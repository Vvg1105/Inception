"""
cyton_stream.py — BrainFlow decoder for OpenBCI Cyton board (USB dongle).

Architecture:
  OpenBCI Cyton (USB serial dongle)
    → BrainFlow BoardShim (polling thread, 40 ms drain interval)
      → CytonBuffer (thread-safe ring buffer + scipy IIR filter bank)
        → check_blink()    — same amplitude-spike algorithm as eeg_stream.py
        → decode_emotion() — same EEGNet inference, same model weights

Usage:
    from eeg.cyton_stream import CytonDecoder
    import json, torch
    from eeg.eegnet import EEGNet

    with open("eeg/models/eegnet_config.json") as f:
        cfg = json.load(f)

    model = EEGNet(...).to(device)
    model.load_state_dict(torch.load("eeg/models/eegnet_emotion.pt", ...))
    model.eval()

    dec = CytonDecoder(serial_port="/dev/cu.usbserial-XXXX")
    dec.setup(model, cfg)
    dec.start()

    while True:
        state = dec.decode()
        # {"arousal": 0..1, "valence": 0..1, "focus": 0..1, "label": str, "blink": bool}

    dec.stop()
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any

import numpy as np

try:
    from scipy.signal import butter, sosfilt, sosfilt_zi
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
    HAS_BRAINFLOW = True
except ImportError:
    HAS_BRAINFLOW = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Signal constants (match eeg_stream.py) ────────────────────────────────────
FS             = 250          # Cyton default sample rate (Hz)
N_CHANNELS     = 8            # EEG channels
BUFFER_SAMPLES = FS * 2       # 2-second rolling buffer
EMOTION_WINDOW = FS           # 1000 ms = 250 samples
BLINK_WINDOW   = FS // 2      # 500 ms = 125 samples

# Blink amplitude thresholds (µV) — same as eeg_stream.py
BLINK_MIN           = 100.0
BLINK_MAX           = 200.0
BLINK2_BASELINE_MAX = 50.0
BLINK2_SPIKE_MIN    = 100.0
BLINK2_SPIKE_MAX    = 200.0

# Emotion-smoothing history window
EMOTION_INTERVAL_S = 0.5
EMOTION_HISTORY_N  = 6

# Emotion label → (arousal, valence) circumplex mapping
EMOTION_AV = {
    "happy": (0.70, 0.85),
    "sad":   (0.25, 0.20),
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

    def __init__(self, serial_port: str = ""):
        if not HAS_BRAINFLOW:
            raise ImportError("pip install brainflow")
        if not HAS_SCIPY:
            raise ImportError("pip install scipy")

        self._serial_port = serial_port
        self._board: "BoardShim | None" = None
        self._eeg_channels: list[int] = []

        self._buf = CytonBuffer()
        self._model = None
        self._cfg: dict = {}
        self._device = None

        self._running = False
        self._poll_thread: "threading.Thread | None" = None

        # Emotion smoothing state
        self._prob_history: collections.deque = collections.deque(maxlen=EMOTION_HISTORY_N)
        self._next_emotion_t = 0.0
        self._last_arousal = 0.5
        self._last_valence = 0.5
        self._last_focus   = 0.5
        self._last_label   = ""

    def setup(self, model: Any, cfg: dict) -> None:
        """
        Register the pre-loaded EEGNet emotion model.
        Same weights file as GTECH user — both headsets use identical 8ch 250Hz EEGNet.

        Parameters
        ----------
        model : EEGNet (torch.nn.Module), already on device, in eval() mode
        cfg   : dict from eegnet_config.json
        """
        self._model = model
        self._cfg = cfg
        if HAS_TORCH:
            self._device = next(model.parameters()).device

    def start(self) -> None:
        """Open BrainFlow session and start background polling thread."""
        BoardShim.disable_board_logger()   # suppress noisy BrainFlow logs

        params = BrainFlowInputParams()
        if self._serial_port:
            params.serial_port = self._serial_port

        self._board = BoardShim(BoardIds.CYTON_BOARD.value, params)
        self._board.prepare_session()
        self._board.start_stream()
        self._eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="cyton-poll"
        )
        self._poll_thread.start()
        print(f"  Cyton streaming on {self._serial_port or '(auto)'} @ {FS} Hz")

    def stop(self) -> None:
        """Stop streaming and release the BrainFlow session."""
        self._running = False
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

    # ── Emotion inference ─────────────────────────────────────────────────────
    def _run_emotion_model(self) -> "tuple[str, float, dict] | None":
        """Single EEGNet forward pass on the most recent 1 s of data."""
        if self._model is None:
            return None
        window = self._buf.latest(EMOTION_WINDOW)
        if window is None:
            return None

        mean = np.array(self._cfg["ch_mean"], dtype=np.float32)
        std  = np.array(self._cfg["ch_std"],  dtype=np.float32)
        win  = (window - mean) / (std + 1e-8)      # (T, C)

        x = torch.from_numpy(win.T[np.newaxis, np.newaxis]).float().to(self._device)
        with torch.no_grad():
            logits = self._model(x)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx        = int(probs.argmax())
        label      = self._cfg["emotions"][idx]
        conf       = float(probs[idx])
        probs_dict = {name: float(probs[i])
                      for i, name in enumerate(self._cfg["emotions"])}
        return label, conf, probs_dict

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
        blink = self.check_blink()

        now = time.monotonic()
        if now >= self._next_emotion_t:
            self._next_emotion_t = now + EMOTION_INTERVAL_S
            result = self._run_emotion_model()
            if result is not None:
                _label, _conf, probs = result
                self._prob_history.append(probs)

        # Smooth by averaging probability history
        if self._prob_history:
            emotions = list(next(iter(self._prob_history)).keys())
            n = len(self._prob_history)
            avg = {e: sum(p[e] for p in self._prob_history) / n for e in emotions}
            best = max(avg, key=avg.__getitem__)
            conf = avg[best]
            av = EMOTION_AV.get(best, (0.5, 0.5))
            self._last_label   = best
            self._last_focus   = conf
            self._last_arousal = av[0] * conf + 0.5 * (1 - conf)
            self._last_valence = av[1] * conf + 0.5 * (1 - conf)

        return {
            "arousal": self._last_arousal,
            "valence": self._last_valence,
            "focus":   self._last_focus,
            "label":   self._last_label,
            "blink":   blink,
        }
