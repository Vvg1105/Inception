"""
eeg_stream.py — Shared EEG rolling buffer + real-time decode functions.

This is the single entry point for live EEG data in this project.
All other modules should import from here rather than touching gpype directly.

━━━ PIPELINE SETUP (do this once at startup) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    import json, torch
    import gpype as gp
    from eeg.eegnet import EEGNet
    from eeg.eeg_stream import EEGBuffer, load_emotion_model

    # 1. Build the gpype pipeline as normal, ending with an EEGBuffer sink.
    p        = gp.Pipeline()
    source   = gp.BCICore8()
    bandpass = gp.Bandpass(f_lo=0.5, f_hi=45)
    notch50  = gp.Bandstop(f_lo=48,  f_hi=52)
    notch60  = gp.Bandstop(f_lo=58,  f_hi=62)
    buf      = EEGBuffer()                      # registers as module singleton

    p.connect(source, bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50,  notch60)
    p.connect(notch60,  buf)

    # 2. (Optional) Load the emotion model if you need decode_emotion().
    #    Skip this if you only need blink detection.
    with open("eeg/models/eegnet_config.json") as f:
        cfg = json.load(f)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNet(n_channels=cfg["n_channels"],
                   n_timepoints=cfg["n_timepoints"],
                   n_classes=cfg["n_classes"]).to(device)
    model.load_state_dict(torch.load("eeg/models/eegnet_emotion.pt",
                                     map_location=device))
    model.eval()
    load_emotion_model(model, cfg)

    p.start()

━━━ CALLING FROM ANY MODULE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from eeg.eeg_stream import check_blink_state, decode_emotion

    # Blink detection — no model needed, pure spike detector.
    if check_blink_state():
        print("blink detected")

    # Emotion decoding — requires load_emotion_model() called at startup.
    result = decode_emotion()
    if result:
        label, confidence = result
        print(label, confidence)

Both functions read the latest 100 ms (25 samples @ 250 Hz) from the shared
rolling buffer each time they are called.  check_blink_state() also holds
its True result for 1 second after the last spike so callers polling at a
low rate don't miss a blink.
"""

import threading
import collections

import numpy as np
import torch
from gpype.backend.core.i_node import INode
from gpype.common.constants import Constants

# ── Constants ─────────────────────────────────────────────────────────────────
FS             = 250                    # expected sample rate (Hz)
EMOTION_WINDOW = int(FS * 0.1)         # 100 ms = 25 samples  (for emotion)
BLINK_WINDOW   = int(FS * 0.5)         # 500 ms = 125 samples (for blink)
BUFFER_SAMPLES = FS                    # 1-second history (250 samples)

# Average absolute amplitude threshold (µV) for eye-closure detection.
# Eyes-closed sustained signal typically runs well above resting EEG.
# Raise this if you get false positives; lower it if closures are missed.
DEFAULT_BLINK_THRESHOLD = 40.0

PORT_IN = Constants.Defaults.PORT_IN

# ── Module-level shared state ─────────────────────────────────────────────────
_buffer: "EEGBuffer | None" = None

_emotion_model  = None
_emotion_cfg    = None
_emotion_device = None


# ── Shared rolling buffer (gpype node) ───────────────────────────────────────
class EEGBuffer(INode):
    """
    gpype pipeline node — thread-safe rolling buffer of EEG samples.

    Registers itself as the module-level singleton on instantiation so
    check_blink_state() and decode_emotion() can access it automatically.

    Parameters
    ----------
    capacity : int
        Number of samples to keep.  Default: 1 second (250 samples @ 250 Hz).
    """

    def __init__(self, capacity: int = BUFFER_SAMPLES, **kwargs):
        super().__init__(**kwargs)
        self._deque: collections.deque = collections.deque(maxlen=capacity)
        self._lock  = threading.Lock()

        global _buffer
        _buffer = self

    def step(self, data: dict) -> None:
        sample = data[PORT_IN][0].copy()   # (n_channels,)
        with self._lock:
            self._deque.append(sample)
        return None

    def latest(self, n: int) -> "np.ndarray | None":
        """
        Return the most recent `n` samples as an array of shape (n, C),
        or None if fewer than `n` samples have arrived yet.
        """
        with self._lock:
            if len(self._deque) < n:
                return None
            return np.array(list(self._deque)[-n:], dtype=np.float32)


# ── Emotion decoder setup ─────────────────────────────────────────────────────
def load_emotion_model(model, cfg: dict) -> None:
    """
    Register the EEGNet model and its config so decode_emotion() can use them.

    Parameters
    ----------
    model : EEGNet (torch.nn.Module)
        Trained model, already on the target device, in eval mode.
    cfg   : dict
        Config dict as produced by train.py — must contain keys:
        'emotions', 'ch_mean', 'ch_std'.
    """
    global _emotion_model, _emotion_cfg, _emotion_device
    _emotion_model  = model
    _emotion_cfg    = cfg
    _emotion_device = next(model.parameters()).device


# ── Public decode functions ───────────────────────────────────────────────────
def check_blink_state(threshold: float = DEFAULT_BLINK_THRESHOLD) -> bool:
    """
    Return True if the eyes appear to be closed right now.

    Looks at the last 500 ms (125 samples) of EEG and computes the mean
    absolute amplitude across all channels.  Sustained eye closure (~0.7 s)
    produces a broad elevation of signal magnitude that reliably exceeds
    resting EEG, making this far more robust than single-spike detection.

    Parameters
    ----------
    threshold : float
        Mean absolute amplitude (µV) that counts as eyes-closed.
        Default: 40 µV.  Tune up to reduce false positives, down if
        genuine closures are missed.
    """
    if _buffer is None:
        raise RuntimeError(
            "No EEGBuffer has been instantiated — add one to your pipeline."
        )

    window = _buffer.latest(BLINK_WINDOW)   # (125, C) or None
    if window is None:
        return False

    # Mean absolute amplitude across all samples and all channels
    mean_abs = float(np.abs(window).mean())
    return mean_abs > threshold


def decode_emotion() -> "tuple[str, float] | None":
    """
    Run EEGNet on the most recent 100 ms of EEG and return the result.

    Returns
    -------
    (label, confidence) : tuple[str, float]
        Predicted emotion label and softmax confidence (0–1).
    None
        If the model hasn't been loaded or the buffer isn't full yet.

    Requires load_emotion_model() to have been called first.
    """
    if _emotion_model is None or _buffer is None:
        return None

    window = _buffer.latest(EMOTION_WINDOW)   # (25, C) or None
    if window is None:
        return None

    # Z-score normalise using training statistics
    mean = np.array(_emotion_cfg["ch_mean"], dtype=np.float32)  # (C,)
    std  = np.array(_emotion_cfg["ch_std"],  dtype=np.float32)  # (C,)
    win  = (window - mean) / (std + 1e-8)                       # (T, C)

    # Reshape to (1, 1, C, T) as EEGNet expects
    x = torch.from_numpy(win.T[np.newaxis, np.newaxis]).float()
    x = x.to(_emotion_device)

    with torch.no_grad():
        logits = _emotion_model(x)                 # (1, n_classes)
        probs  = torch.softmax(logits, dim=1)[0]   # (n_classes,)

    idx   = int(probs.argmax())
    label = _emotion_cfg["emotions"][idx]
    conf  = float(probs[idx])
    return label, conf
