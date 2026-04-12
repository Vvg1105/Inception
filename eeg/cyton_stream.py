"""
cyton_stream.py — BrainFlow decoder for OpenBCI Cyton board (USB dongle).

Architecture:
  OpenBCI Cyton (USB serial dongle)
    → BrainFlow BoardShim (polling thread, 40 ms drain interval)
      → raw EEG fed to BlinkDetector (BLINK algorithm, eeg/blink_detector.py)
      → CytonBuffer (thread-safe ring buffer + scipy IIR filter bank)
        → _run_eegnet_emotion() — EEGNet/EmotionMLP from eeg/models/cyton/
              sad / happy / neutral → arousal/valence circumplex
"""

from __future__ import annotations

import collections
import json
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
BUFFER_SAMPLES = FS * 4       # 4-second rolling buffer
BLINK_WINDOW   = FS // 2      # 500 ms = 125 samples
EMOTION_WINDOW = int(FS * 1.0)  # 1 s = 250 samples — EEGNet input window

# Default emotion model paths — mirrors gtech pattern at root of eeg/models/
_DEFAULT_EMOTION_WEIGHTS = os.path.join(_PROJECT_ROOT, "eeg", "models", "eegnet_emotion_cyton.pt")
_DEFAULT_EMOTION_CONFIG  = os.path.join(_PROJECT_ROOT, "eeg", "models", "eegnet_config_cyton.json")

# Blink amplitude thresholds (µV) — same as eeg_stream.py
BLINK_MIN           = 100.0
BLINK_MAX           = 200.0
BLINK2_BASELINE_MAX = 50.0
BLINK2_SPIKE_MIN    = 100.0
BLINK2_SPIKE_MAX    = 200.0

# Emotion smoothing history window
EMOTION_INTERVAL_S = 0.5
EMOTION_HISTORY_N  = 6

# Emotion label → (arousal, valence) circumplex mapping
EMOTION_AV = {
    "happy":   (0.70, 0.85),
    "sad":     (0.25, 0.20),
    "neutral": (0.50, 0.50),
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

    Emotion is classified by EEGNet/EmotionMLP loaded from eeg/models/cyton/.
    Blink detection uses BlinkDetector (BLINK paper) when a profile is provided,
    otherwise falls back to amplitude double-blink.

    Thread-safe.  Call start() → poll decode() → stop().
    """

    def __init__(
        self,
        serial_port: str = "",
        *,
        blink_profile: Optional[str] = None,
        frontal_ch: int = 0,
        use_blink_paper: bool = True,
        emotion_weights: Optional[str] = None,
        emotion_config: Optional[str] = None,
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

        # EEGNet emotion model (loaded in start())
        self._emotion_model = None
        self._emotion_cfg: dict = {}
        self._emotion_device = None
        self._emotion_weights_path = emotion_weights or _DEFAULT_EMOTION_WEIGHTS
        self._emotion_config_path  = emotion_config  or _DEFAULT_EMOTION_CONFIG

        # Emotion smoothing — average per-class probability vectors
        self._prob_history: collections.deque = collections.deque(maxlen=EMOTION_HISTORY_N)
        self._next_emotion_t = 0.0
        self._last_arousal = 0.5
        self._last_valence = 0.5
        self._last_focus   = 0.5
        self._last_label   = ""

        self._use_blink_paper = use_blink_paper
        self._blink_det: "BlinkDetector | None" = None
        if use_blink_paper:
            self._blink_det = BlinkDetector(
                fs=FS,
                frontal_ch=frontal_ch,
                profile=blink_profile,
            )

    def start(self) -> None:
        """Open BrainFlow session, load EEGNet emotion model, start polling thread."""
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

        # Load EEGNet / EmotionMLP emotion model from eeg/models/cyton/
        self._load_emotion_model()

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="cyton-poll"
        )
        self._poll_thread.start()
        print(f"  Cyton streaming on {self._serial_port or '(auto)'} @ {FS} Hz")

    def _load_emotion_model(self) -> None:
        """Load the EEGNet/EmotionMLP model for emotion classification."""
        if not HAS_TORCH:
            print("  [!] Cyton: torch not available — emotion will be neutral", flush=True)
            return
        try:
            from eeg.eegnet import EEGNet, EmotionMLP

            with open(self._emotion_config_path) as f:
                self._emotion_cfg = json.load(f)

            device = torch.device(
                "mps" if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model_cls = EmotionMLP if self._emotion_cfg.get("model") == "mlp" else EEGNet
            model = model_cls(
                n_channels=self._emotion_cfg["n_channels"],
                n_timepoints=self._emotion_cfg["n_timepoints"],
                n_classes=self._emotion_cfg["n_classes"],
            ).to(device)
            state_dict = torch.load(self._emotion_weights_path, map_location=device)
            # Remap weights saved without the Dropout layer in the classifier
            # (classifier.1.* → classifier.2.* when Flatten/Dropout/Linear order is used)
            if "classifier.1.weight" in state_dict and "classifier.2.weight" not in state_dict:
                state_dict = {
                    ("classifier.2." + k[len("classifier.1."):] if k.startswith("classifier.1.") else k): v
                    for k, v in state_dict.items()
                }
            model.load_state_dict(state_dict)
            model.eval()
            self._emotion_model  = model
            self._emotion_device = device
            print(
                f"  [Cyton] emotion model loaded  {model_cls.__name__} "
                f"({self._emotion_cfg['n_classes']} classes: "
                f"{self._emotion_cfg.get('emotions', [])})",
                flush=True,
            )
        except Exception as exc:
            print(f"  [!] Cyton emotion model load failed: {exc} — emotion will be neutral",
                  flush=True)

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
                        if self._blink_det is not None:
                            self._blink_det.feed(eeg)
                        self._buf.push(eeg)
            except Exception:
                pass
            time.sleep(0.04)

    # ── Blink detection ───────────────────────────────────────────────────────
    def check_blink(self) -> bool:
        """
        Double-blink detector — amplitude spike method.
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

    # ── Emotion inference via EEGNet ──────────────────────────────────────────
    def _run_eegnet_emotion(self) -> "dict[str, float] | None":
        """
        Run the EEGNet/EmotionMLP model on the most recent 1 s of EEG.

        Returns a {emotion_name: probability} dict, or None if the model
        is not loaded or the buffer has not filled yet.
        """
        if self._emotion_model is None:
            return None

        n_tp = self._emotion_cfg["n_timepoints"]
        window = self._buf.latest(n_tp)   # (n_tp, n_channels) or None
        if window is None:
            return None

        mean = np.array(self._emotion_cfg["ch_mean"], dtype=np.float32)
        std  = np.array(self._emotion_cfg["ch_std"],  dtype=np.float32)
        win  = (window - mean) / (std + 1e-8)   # (T, C), z-scored

        # EEGNet input: (batch=1, in_channels=1, eeg_channels=C, timepoints=T)
        x = torch.from_numpy(win.T[np.newaxis, np.newaxis]).float().to(self._emotion_device)

        with torch.no_grad():
            logits = self._emotion_model(x)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (n_classes,)

        return {name: float(probs[i])
                for i, name in enumerate(self._emotion_cfg["emotions"])}

    # ── Public decode interface ───────────────────────────────────────────────
    def decode(self) -> dict[str, Any]:
        """
        Returns the current decoded state.
        Same contract as LiveDecoder.decode() in eeg_decode.py.

        Returns
        -------
        dict with keys:
            arousal, valence, focus : float 0..1
            label                  : str  ("happy" | "sad" | "neutral" | "")
            blink                  : bool
        """
        if self._blink_det is not None and self._blink_det.ready:
            blink = self._blink_det.check()
        else:
            blink = self.check_blink()

        now = time.monotonic()
        if now >= self._next_emotion_t:
            self._next_emotion_t = now + EMOTION_INTERVAL_S
            probs = self._run_eegnet_emotion()
            if probs is not None:
                self._prob_history.append(probs)

        # Smooth by averaging raw per-class probabilities over the history window.
        if self._prob_history:
            emotions = list(next(iter(self._prob_history)).keys())
            n = len(self._prob_history)
            avg_probs = {e: sum(p[e] for p in self._prob_history) / n for e in emotions}
            best = max(avg_probs, key=avg_probs.__getitem__)
            confidence = avg_probs[best]
            av = EMOTION_AV.get(best, (0.5, 0.5))
            self._last_label   = best
            self._last_focus   = confidence
            self._last_arousal = av[0] * confidence + 0.5 * (1.0 - confidence)
            self._last_valence = av[1] * confidence + 0.5 * (1.0 - confidence)

        return {
            "arousal": self._last_arousal,
            "valence": self._last_valence,
            "focus":   self._last_focus,
            "label":   self._last_label,
            "blink":   blink,
        }
