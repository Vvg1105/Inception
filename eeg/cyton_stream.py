"""
cyton_stream.py — BrainFlow decoder for OpenBCI Cyton board (USB dongle).

Architecture:
  OpenBCI Cyton (USB serial dongle)
    → BrainFlow BoardShim (polling thread, 40 ms drain interval)
      → raw EEG fed to BlinkDetector (BLINK algorithm, eeg/blink_detector.py)
      → CytonBuffer (thread-safe ring buffer + scipy IIR filter bank)
        → _run_brainflow_emotion() — BrainFlow RESTFULNESS classifier
              relaxed (≥ 0.8) → "happy"
              neutral / stressed (< 0.8) → "sad"
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import time
from typing import Any, Optional

import numpy as np

# Ensure project root on path so blink_detector can be imported stand-alone
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eeg.blink_detector import BlinkDetector

try:
    from scipy.signal import butter, sosfilt, sosfilt_zi
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
    from brainflow.data_filter import DataFilter
    from brainflow.ml_model import MLModel, BrainFlowModelParams, BrainFlowMetrics, BrainFlowClassifiers
    HAS_BRAINFLOW = True
except ImportError:
    HAS_BRAINFLOW = False

# ── Signal constants (match eeg_stream.py) ────────────────────────────────────
FS             = 250          # Cyton default sample rate (Hz)
N_CHANNELS     = 8            # EEG channels
BUFFER_SAMPLES = FS * 4       # 4-second rolling buffer to satisfy BrainFlow emotion window
BLINK_WINDOW   = FS // 2      # 500 ms = 125 samples

# BrainFlow RESTFULNESS classifier — min recommended window is 4 s
BF_EMOTION_SAMPLES  = FS * 4  # 1000 samples @ 250 Hz
# Relaxation score thresholds → emotion label
BF_RELAX_HAPPY      = 0.8     # score ≥ 0.8 → happy / relaxed
BF_RELAX_SAD        = 0.4     # score 0.4–0.8 → sad / neutral
                               # score < 0.4 → angry / stressed

# Blink amplitude thresholds (µV) — same as eeg_stream.py
BLINK_MIN           = 100.0
BLINK_MAX           = 200.0
BLINK2_BASELINE_MAX = 50.0
BLINK2_SPIKE_MIN    = 100.0
BLINK2_SPIKE_MAX    = 200.0

# Center band for neutral / ambiguous RESTFULNESS scores
BF_RELAX_NEUTRAL_DELTA = 0.10

# Emotion smoothing history window
EMOTION_INTERVAL_S = 0.5
EMOTION_HISTORY_N  = 6

# Emotion label → (arousal, valence) circumplex mapping
EMOTION_AV = {
    "relaxed":     (0.70, 0.85),   # relaxed / positive
    "not relaxed":(0.25, 0.20),   # low-arousal negative / stressed
    "neutral":     (0.50, 0.50),   # ambiguous/center
}


# ── Filter bank (scipy IIR, causal, sample-by-sample friendly) ───────────────
class _FilterBank:
    """
    Per-channel IIR filter bank: bandpass(0.5–45 Hz) + notch50 + notch60.
    Uses second-order sections (SOS) with persistent state so each new chunk
    of samples arrives with correct initial conditions — no discontinuities.
    """

    def __init__(self, n_channels: int = N_CHANNELS, fs: int = FS):
        if not HAS_SCIPY:
            raise ImportError("pip install scipy")
        self._n_ch = n_channels
        self._sos_bp  = butter(4, [0.5, 45.0],   btype="bandpass", fs=fs, output="sos")
        self._sos_n50 = butter(4, [48.0, 52.0],  btype="bandstop", fs=fs, output="sos")
        self._sos_n60 = butter(4, [58.0, 62.0],  btype="bandstop", fs=fs, output="sos")
        # Filter state per channel (zi shape = (n_sections, 2))
        zi0 = sosfilt_zi(self._sos_bp)
        zi1 = sosfilt_zi(self._sos_n50)
        zi2 = sosfilt_zi(self._sos_n60)
        self._zi_bp  = [zi0.copy() for _ in range(n_channels)]
        self._zi_n50 = [zi1.copy() for _ in range(n_channels)]
        self._zi_n60 = [zi2.copy() for _ in range(n_channels)]

    def process(self, raw: np.ndarray) -> np.ndarray:
        """
        raw   : shape (n_samples, n_channels) float64
        return: shape (n_samples, n_channels) float32, filtered
        """
        n_samples, n_ch = raw.shape
        out = np.zeros((n_samples, min(n_ch, self._n_ch)), dtype=np.float32)
        for c in range(min(n_ch, self._n_ch)):
            ch = raw[:, c].astype(np.float32)
            ch, self._zi_bp[c]  = sosfilt(self._sos_bp,  ch, zi=self._zi_bp[c])
            ch, self._zi_n50[c] = sosfilt(self._sos_n50, ch, zi=self._zi_n50[c])
            ch, self._zi_n60[c] = sosfilt(self._sos_n60, ch, zi=self._zi_n60[c])
            out[:, c] = ch
        return out


# ── Rolling EEG buffer ────────────────────────────────────────────────────────
class CytonBuffer:
    """
    Thread-safe rolling buffer of *filtered* EEG samples.
    Mirrors EEGBuffer in eeg_stream.py but is instance-based (no global state).
    """

    def __init__(self, capacity: int = BUFFER_SAMPLES, n_channels: int = N_CHANNELS):
        self._deque: collections.deque = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._filters = _FilterBank(n_channels=n_channels)

    def push(self, raw: np.ndarray) -> None:
        """raw: (n_samples, n_channels)"""
        filtered = self._filters.process(raw)
        with self._lock:
            for sample in filtered:
                self._deque.append(sample.copy())

    def latest(self, n: int) -> "np.ndarray | None":
        """Return the most recent n samples as (n, n_channels) float32, or None."""
        with self._lock:
            if len(self._deque) < n:
                return None
            return np.array(list(self._deque)[-n:], dtype=np.float32)


# ── Main decoder class ────────────────────────────────────────────────────────
class CytonDecoder:
    """
    Full EEG decode pipeline for OpenBCI Cyton via BrainFlow.

    Thread-safe.  Call setup() → start() → poll decode() → stop().
    """

    def __init__(
        self,
        serial_port: str = "",
        *,
        blink_profile: Optional[str] = None,
        frontal_ch: int = 0,
        use_blink_paper: bool = True,
    ):
        if not HAS_BRAINFLOW:
            raise ImportError("pip install brainflow")
        if not HAS_SCIPY:
            raise ImportError("pip install scipy")

        self._serial_port = serial_port
        self._board: "BoardShim | None" = None
        self._eeg_channels: list[int] = []

        self._buf = CytonBuffer()

        self._running = False
        self._poll_thread: "threading.Thread | None" = None

        # BrainFlow RESTFULNESS classifier (prepared in start(), released in stop())
        self._bf_model: "MLModel | None" = None

        # Emotion smoothing state
        self._score_history: collections.deque = collections.deque(maxlen=EMOTION_HISTORY_N)
        self._next_emotion_t = 0.0
        self._last_arousal = 0.5
        self._last_valence = 0.5
        self._last_focus   = 0.5
        self._last_label   = ""
        self._emotion_debug_started = time.monotonic()
        self._emotion_debug_warned = False

        self._use_blink_paper = use_blink_paper
        self._blink_det: "BlinkDetector | None" = None
        if use_blink_paper:
            self._blink_det = BlinkDetector(
                fs=FS,
                frontal_ch=frontal_ch,
                profile=blink_profile,
            )

    def start(self) -> None:
        """Open BrainFlow session, prepare ML classifier, start polling thread."""
        BoardShim.disable_board_logger()   # suppress noisy BrainFlow logs

        params = BrainFlowInputParams()
        if self._serial_port:
            params.serial_port = self._serial_port

        self._board = BoardShim(BoardIds.CYTON_BOARD.value, params)
        print(f"  [Cyton] opening board session on {self._serial_port or '(auto)'}", flush=True)
        self._board.prepare_session()
        print("  [Cyton] board session prepared", flush=True)
        self._board.start_stream()
        print("  [Cyton] board stream started", flush=True)
        self._eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
        print(f"  [Cyton] eeg channels: {self._eeg_channels}", flush=True)

        # Prepare the BrainFlow RESTFULNESS emotion classifier once for the session
        bf_params = BrainFlowModelParams(
            BrainFlowMetrics.RESTFULNESS.value,
            BrainFlowClassifiers.DEFAULT_CLASSIFIER.value,
        )
        self._bf_model = MLModel(bf_params)
        self._bf_model.prepare()

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="cyton-poll"
        )
        self._poll_thread.start()
        print(f"  Cyton streaming on {self._serial_port or '(auto)'} @ {FS} Hz")

    def stop(self) -> None:
        """Stop streaming and release the BrainFlow session and ML model."""
        self._running = False
        if self._bf_model is not None:
            try:
                self._bf_model.release()
            except Exception:
                pass
            self._bf_model = None
        if self._board is not None:
            try:
                self._board.stop_stream()
                self._board.release_session()
            except Exception:
                pass
            self._board = None

    # ── Internal polling thread ───────────────────────────────────────────────
    def _poll_loop(self) -> None:
        """Drains the BrainFlow ring buffer every 40 ms (~25 Hz drain rate)."""
        while self._running:
            try:
                if self._board is not None:
                    count = self._board.get_board_data_count()
                    if count > 0:
                        data = self._board.get_board_data(count)
                        # data shape: (n_total_channels, n_samples)
                        # transpose to (n_samples, n_channels)
                        eeg = data[self._eeg_channels, :].T.astype(np.float64)
                        if self._blink_det is not None:
                            self._blink_det.feed(eeg)
                        self._buf.push(eeg)
            except Exception:
                pass
            time.sleep(0.04)

    # ── Blink detection ───────────────────────────────────────────────────────
    def check_blink(self) -> bool:
        """
        Double-blink detector — same algorithm as check_blink_state_old() in eeg_stream.py.
        Returns True when 2+ separate amplitude spikes are found in the last 500 ms,
        each within [BLINK_MIN, BLINK_MAX] µV.
        """
        window = self._buf.latest(BLINK_WINDOW)
        if window is None:
            return False
        amp = np.abs(window).mean(axis=1)          # shape: (T,)
        active = (amp > BLINK_MIN) & (amp < BLINK_MAX)
        spike_count = int((np.diff(active.astype(np.int8)) == 1).sum())
        return spike_count >= 2

    # ── Emotion inference via BrainFlow RESTFULNESS classifier ───────────────
    def _run_brainflow_emotion(self) -> "float | None":
        """
        Returns a relaxation score in [0, 1] using BrainFlow's built-in
        RESTFULNESS classifier (band-power feature vector → logistic model).

        Requires at least 4 seconds of board data (BrainFlow recommendation).
        Returns None if the board or model is not ready yet.
        """
        if self._board is None or self._bf_model is None:
            return None
        try:
            # Prefer the live BrainFlow board buffer; fallback to the filtered
            # rolling buffer if the board buffer is not yet ready.
            data = self._board.get_current_board_data(BF_EMOTION_SAMPLES)
            if data.shape[1] < BF_EMOTION_SAMPLES:
                buf = self._buf.latest(BF_EMOTION_SAMPLES)
                if buf is None or buf.shape[0] < BF_EMOTION_SAMPLES:
                    if not self._emotion_debug_warned and time.monotonic() - self._emotion_debug_started > 6.0:
                        print(
                            f"  [!] Cyton emotion waiting for {BF_EMOTION_SAMPLES} samples; "
                            f"buffer has {0 if buf is None else buf.shape[0]}.",
                            flush=True,
                        )
                        self._emotion_debug_warned = True
                    return None
                data = np.ascontiguousarray(buf.T.astype(np.float64))
                channels = list(range(data.shape[0]))
                bands = DataFilter.get_avg_band_powers(
                    data, channels, FS, apply_filter=False
                )
            else:
                data = np.ascontiguousarray(data)
                bands = DataFilter.get_avg_band_powers(
                    data, self._eeg_channels, FS, apply_filter=True
                )
            feature_vector = bands[0]
            raw = self._bf_model.predict(feature_vector)
            return float(np.atleast_1d(raw)[0])
        except Exception as exc:
            if not self._emotion_debug_warned:
                print(f"  [!] Cyton emotion classifier error: {exc}", flush=True)
                self._emotion_debug_warned = True
            return None

    # ── Public decode interface ───────────────────────────────────────────────
    def decode(self) -> dict[str, Any]:
        """
        Returns the current decoded state.
        Same contract as LiveDecoder.decode() in eeg_decode.py.

        Returns
        -------
        dict with keys:
            arousal, valence, focus : float 0..1
            label                  : str  ("happy" | "sad" | "")
            blink                  : bool
        """
        if self._blink_det is not None and self._blink_det.ready:
            blink = self._blink_det.check()
        else:
            blink = self.check_blink()

        now = time.monotonic()
        if now >= self._next_emotion_t:
            self._next_emotion_t = now + EMOTION_INTERVAL_S
            score = self._run_brainflow_emotion()
            if score is not None:
                self._score_history.append(score)

        # Smooth by averaging score history, then threshold to emotion label
        if self._score_history:
            avg_score = sum(self._score_history) / len(self._score_history)

            if abs(avg_score - 0.5) <= BF_RELAX_NEUTRAL_DELTA:
                label = "neutral"
            elif avg_score > 0.5:
                label = "relaxed"
            else:
                label = "not relaxed"

            # Focus = how far the score is from ambiguous (0.5 center)
            conf = min(1.0, abs(avg_score - 0.5) * 2.0)
            av = EMOTION_AV[label]
            self._last_label   = label
            self._last_focus   = conf
            self._last_arousal = av[0] * conf + 0.5 * (1.0 - conf)
            self._last_valence = av[1] * conf + 0.5 * (1.0 - conf)

        return {
            "arousal": self._last_arousal,
            "valence": self._last_valence,
            "focus":   self._last_focus,
            "label":   self._last_label,
            "blink":   blink,
        }
