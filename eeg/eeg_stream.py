"""
eeg_stream.py — Shared EEG rolling buffer + real-time decode functions.

This is the single entry point for live EEG data in this project.
All other modules should import from here rather than touching gpype directly.

━━━ PIPELINE SETUP (do this once at startup) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Build a standard gpype signal chain ending with EEGBuffer as the final
    sink node.  EEGBuffer registers itself as a module-level singleton on
    instantiation, so check_blink_state() and decode_emotion() will find it
    automatically without any extra wiring.

    # ── Minimal setup (blink detection only) ──────────────────────────────
    import gpype as gp
    from eeg.eeg_stream import EEGBuffer, check_blink_state

    p        = gp.Pipeline()
    source   = gp.BCICore8()             # g.tec hardware source node
    bandpass = gp.Bandpass(f_lo=0.5, f_hi=45)   # remove DC drift + HF noise
    notch50  = gp.Bandstop(f_lo=48,  f_hi=52)   # suppress 50 Hz mains
    notch60  = gp.Bandstop(f_lo=58,  f_hi=62)   # suppress 60 Hz mains
    buf      = EEGBuffer()               # registers as module singleton

    p.connect(source,   bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50,  notch60)
    p.connect(notch60,  buf)             # buf is the final sink

    p.start()   # must be called before any data arrives in the buffer

    # ── Full setup (blink + emotion decoding) ──────────────────────────────
    import json, torch
    import gpype as gp
    from eeg.eegnet import EEGNet
    from eeg.eeg_stream import EEGBuffer, load_emotion_model

    # Build and start the pipeline exactly as above, then additionally:

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
    load_emotion_model(model, cfg)   # registers model as module singleton

    p.start()

━━━ CALLING FROM ANY MODULE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from eeg.eeg_stream import check_blink_state, decode_emotion

    # ── Double-blink detection ────────────────────────────────────────────
    # No model required.  Returns True when two separate amplitude spikes
    # are found within the last 500 ms, each within [MIN, MAX] µV band.
    # Returns False (not raises) if the buffer hasn't filled yet.
    if check_blink_state():
        print("double blink detected")

    # ── Emotion decoding ──────────────────────────────────────────────────
    # Requires load_emotion_model() to have been called at startup.
    # Returns None if the model isn't loaded or the buffer hasn't filled yet.
    result = decode_emotion()
    if result:
        label, confidence = result   # e.g. ("happy", 0.87)
        print(label, confidence)

━━━ DATA WINDOWS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    check_blink_state() — inspects the last 500 ms (125 samples @ 250 Hz).
    decode_emotion()    — inspects the last 100 ms  (25  samples @ 250 Hz).

    Both functions read directly from the live buffer on each call; there is
    no caching or hold-time applied.  If you poll at a low rate you may miss
    short blinks — poll at least every 100–200 ms for reliable detection.
"""

import threading
import collections

import numpy as np
import torch
from gpype.backend.core.i_node import INode
from gpype.common.constants import Constants

# ── Constants ─────────────────────────────────────────────────────────────────
FS             = 250                    # hardware sample rate (Hz)
EMOTION_WINDOW = int(FS * 0.1)         # 100 ms = 25 samples  — EEGNet input window
BLINK_WINDOW   = int(FS * 0.5)         # 500 ms = 125 samples — eye-closure window
BUFFER_SAMPLES = FS                    # rolling buffer capacity: 1 second (250 samples)

# Amplitude band (µV) used by check_blink_state().
# A genuine eye closure produces a sustained broad-band elevation that
# lands between these two values.  Signals below MIN are resting EEG;
# signals above MAX are muscle artifacts, electrode pop, or saturation —
# not real blinks.  Both constants are intentionally not exposed as
# parameters — edit here to retune for your hardware / subject.
DEFAULT_BLINK_MIN_THRESHOLD = 75.0    # µV — lower bound: must exceed resting EEG
DEFAULT_BLINK_MAX_THRESHOLD = 300.0   # µV — upper bound: above this is artifact/noise

# gpype constant for the default input port name on all pipeline nodes.
PORT_IN = Constants.Defaults.PORT_IN

# ── Module-level shared state ─────────────────────────────────────────────────
# Set by EEGBuffer.__init__(); read by check_blink_state() and decode_emotion().
_buffer: "EEGBuffer | None" = None

# Set by load_emotion_model(); read by decode_emotion().
_emotion_model  = None   # torch.nn.Module — trained EEGNet
_emotion_cfg    = None   # dict from eegnet_config.json (emotions, ch_mean, ch_std)
_emotion_device = None   # torch.device the model lives on


# ── Shared rolling buffer (gpype node) ───────────────────────────────────────
class EEGBuffer(INode):
    """
    gpype pipeline node — thread-safe rolling buffer of raw EEG samples.

    Place this as the final sink in the gpype pipeline.  On instantiation it
    registers itself as the module-level singleton (_buffer), making the live
    data stream available to check_blink_state() and decode_emotion() from
    any module without any additional wiring.

    Only one EEGBuffer should exist at a time; instantiating a second one
    replaces the singleton reference.

    Parameters
    ----------
    capacity : int
        Number of samples to retain.  Default: BUFFER_SAMPLES (250 samples =
        1 second at 250 Hz).  Increase if you need a longer history window.
    """

    def __init__(self, capacity: int = BUFFER_SAMPLES, **kwargs):
        super().__init__(**kwargs)
        self._deque: collections.deque = collections.deque(maxlen=capacity)
        self._lock  = threading.Lock()

        # Register this instance as the module singleton so the decode
        # functions can find it without being passed a reference explicitly.
        global _buffer
        _buffer = self

    def step(self, data: dict) -> None:
        """
        Called by the gpype runtime for every incoming sample.

        Appends the sample vector to the rolling deque under a lock so that
        concurrent reads from check_blink_state() / decode_emotion() are safe.
        Oldest samples are automatically discarded once the deque reaches capacity.

        Parameters
        ----------
        data : dict
            gpype port dict.  data[PORT_IN][0] is a 1-D array of shape
            (n_channels,) representing one EEG sample across all channels.
        """
        sample = data[PORT_IN][0].copy()   # shape: (n_channels,) — copy to avoid aliasing
        with self._lock:
            self._deque.append(sample)
        return None

    def latest(self, n: int) -> "np.ndarray | None":
        """
        Return the most recent `n` samples as a float32 array of shape (n, C).

        Returns None if fewer than `n` samples have arrived yet (e.g. shortly
        after pipeline start).  Callers should treat None as "not ready" and
        skip processing for that tick.

        Parameters
        ----------
        n : int
            Number of samples to retrieve.  Must be <= capacity.
        """
        with self._lock:
            if len(self._deque) < n:
                return None
            return np.array(list(self._deque)[-n:], dtype=np.float32)


# ── Emotion model registration ────────────────────────────────────────────────
def load_emotion_model(model, cfg: dict) -> None:
    """
    Register the trained EEGNet model so decode_emotion() can use it.

    Call this once at startup after loading the model from disk.  The model
    must already be on the desired device and in eval mode before passing it
    here.  decode_emotion() will return None until this is called.

    Parameters
    ----------
    model : EEGNet (torch.nn.Module)
        Trained model, moved to device and set to eval() by the caller.
    cfg   : dict
        Config dict loaded from eegnet_config.json.  Required keys:
          'emotions'  — list[str] mapping class index → label (e.g. "happy")
          'ch_mean'   — list[float] per-channel mean used during training
          'ch_std'    — list[float] per-channel std  used during training
          'n_channels', 'n_timepoints', 'n_classes' — model architecture params
    """
    global _emotion_model, _emotion_cfg, _emotion_device
    _emotion_model  = model
    _emotion_cfg    = cfg
    _emotion_device = next(model.parameters()).device   # infer device from model weights


# ── Public decode functions ───────────────────────────────────────────────────
def check_blink_state() -> bool:
    """
    Return True if a double blink is detected in the last 500 ms.

    A double blink is defined as two separate amplitude spikes within the
    500 ms (BLINK_WINDOW = 125 samples) rolling window, where each spike
    must fall within the valid amplitude band:
      DEFAULT_BLINK_MIN_THRESHOLD (40 µV)  — must exceed resting eyes-open EEG.
      DEFAULT_BLINK_MAX_THRESHOLD (150 µV) — must not exceed this; values above
        indicate muscle artifact, electrode pop, or ADC saturation.

    Algorithm:
      1. Compute per-sample mean absolute amplitude across all channels → (T,).
      2. Mark each sample as "active" if its amplitude is within [MIN, MAX].
      3. Count the number of contiguous active runs (rising edges on the mask).
      4. Return True only if there are at least 2 separate runs (double blink).

    A single sustained closure or a single spike will not trigger this —
    two distinct bursts separated by at least one sub-threshold sample are
    required.  Both thresholds are fixed constants and not configurable
    by callers; edit DEFAULT_BLINK_MIN/MAX_THRESHOLD in this file to retune.

    Returns
    -------
    bool
        True  — two or more valid amplitude spikes found within the window.
        False — fewer than two spikes, all above MAX, or buffer not yet full.

    Raises
    ------
    RuntimeError
        If no EEGBuffer has been added to the pipeline yet.
    """
    if _buffer is None:
        raise RuntimeError(
            "No EEGBuffer has been instantiated — add one to your pipeline."
        )

    window = _buffer.latest(BLINK_WINDOW)   # shape: (125, n_channels) or None
    if window is None:
        # Buffer hasn't accumulated enough samples yet (pipeline just started).
        return False

    # Per-sample mean absolute amplitude across all channels → shape: (T,).
    # Averaging across channels suppresses single-channel noise so only
    # correlated, broad-band events (genuine blinks) pass the threshold.
    amp = np.abs(window).mean(axis=1)

    # Boolean mask: True where amplitude is inside the valid [MIN, MAX] band.
    # Samples below MIN are resting EEG; samples above MAX are artifacts.
    active = (amp > DEFAULT_BLINK_MIN_THRESHOLD) & (amp < DEFAULT_BLINK_MAX_THRESHOLD)

    # Count rising edges (False → True transitions) in the mask.
    # Each rising edge is the start of a new spike burst.
    # np.diff on int8 gives +1 at each False→True transition.
    spike_count = int((np.diff(active.astype(np.int8)) == 1).sum())

    # Require at least 2 separate spikes to confirm a double blink.
    return spike_count >= 2


def decode_emotion() -> "tuple[str, float] | None":
    """
    Run EEGNet on the most recent 100 ms of EEG and return the predicted emotion.

    Normalises the window using the per-channel mean and std recorded during
    training, then runs a forward pass through the registered EEGNet model.

    Returns
    -------
    (label, confidence) : tuple[str, float]
        label      — emotion string (e.g. "happy", "neutral") from cfg['emotions'].
        confidence — softmax probability of the top class, in [0, 1].
    None
        If load_emotion_model() has not been called, or if the buffer hasn't
        accumulated enough samples yet (returns None silently, not an error).

    Requires
    --------
    load_emotion_model(model, cfg) must be called once at startup.
    """
    if _emotion_model is None or _buffer is None:
        return None

    window = _buffer.latest(EMOTION_WINDOW)   # shape: (25, n_channels) or None
    if window is None:
        # Buffer hasn't accumulated enough samples yet.
        return None

    # Z-score normalise each channel using the training statistics stored in cfg.
    # This must match exactly the normalisation applied during training.
    mean = np.array(_emotion_cfg["ch_mean"], dtype=np.float32)  # shape: (C,)
    std  = np.array(_emotion_cfg["ch_std"],  dtype=np.float32)  # shape: (C,)
    win  = (window - mean) / (std + 1e-8)                       # shape: (T, C); 1e-8 avoids ÷0

    # EEGNet expects input shape (batch=1, in_channels=1, eeg_channels=C, timepoints=T).
    # win is (T, C) → transpose to (C, T) → add batch + in_channel dims.
    x = torch.from_numpy(win.T[np.newaxis, np.newaxis]).float()
    x = x.to(_emotion_device)

    with torch.no_grad():
        logits = _emotion_model(x)                 # shape: (1, n_classes)
        probs  = torch.softmax(logits, dim=1)[0]   # shape: (n_classes,) — sum to 1

    idx   = int(probs.argmax())
    label = _emotion_cfg["emotions"][idx]
    conf  = float(probs[idx])
    return label, conf
