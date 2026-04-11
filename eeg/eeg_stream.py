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
EMOTION_WINDOW = int(FS * 1.0)         # 1000 ms = 250 samples — EEGNet input window
BLINK_WINDOW   = int(FS * 0.5)         # 500 ms = 125 samples — eye-closure window
BUFFER_SAMPLES = FS * 2                # rolling buffer capacity: 2 seconds (500 samples)

# Amplitude band (µV) used by check_blink_state().
# A genuine eye closure produces a sustained broad-band elevation that
# lands between these two values.  Signals below MIN are resting EEG;
# signals above MAX are muscle artifacts, electrode pop, or saturation —
# not real blinks.  Both constants are intentionally not exposed as
# parameters — edit here to retune for your hardware / subject.
DEFAULT_BLINK_MIN_THRESHOLD = 100.0    # µV — lower bound: must exceed resting EEG
DEFAULT_BLINK_MAX_THRESHOLD = 200.0   # µV — upper bound: above this is artifact/noise

# check_blink_state_v2() thresholds.
# The v2 detector first confirms the signal is at a quiet baseline before
# looking for spikes, so noisy / high-movement conditions always yield False.
BLINK2_WINDOW       = FS                # 1 s = 250 samples — detection window
BLINK2_BASELINE_MAX = 50.0             # µV — median of the window must be below this;
                                        #       above → too much noise / movement, skip
BLINK2_SPIKE_MIN    = 100.0             # µV — spike must rise above this relative to 0 V
BLINK2_SPIKE_MAX    = 200.0            # µV — spike must not exceed this (artifact above)

# gpype constant for the default input port name on all pipeline nodes.
PORT_IN = Constants.Defaults.PORT_IN

# ── Module-level shared state ─────────────────────────────────────────────────
# Set by EEGBuffer.__init__(); read by check_blink_state() and decode_emotion().
_buffer: "EEGBuffer | None" = None

# Set by load_emotion_model(); read by decode_emotion().
_emotion_model  = None   # torch.nn.Module — trained EEGNet
_emotion_cfg    = None   # dict from eegnet_config.json (emotions, ch_mean, ch_std)
_emotion_device = None   # torch.device the model lives on

# Set by load_blink_model(); read by check_blink_model().
_blink_model  = None     # torch.nn.Module — trained binary EEGNet
_blink_cfg    = None     # dict from blinknet_config.json
_blink_device = None     # torch.device


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
        Called by the gpype runtime for every incoming frame.

        BCICore8 delivers frames of multiple samples (e.g. 10 samples per
        40 ms frame at 250 Hz).  All samples in the frame are appended so the
        buffer runs at the true hardware sample rate.  Taking only [0] would
        leave the effective rate at 25 Hz, corrupting frequency content and
        window duration for both band-power and EEGNet inference.

        Parameters
        ----------
        data : dict
            gpype port dict.  data[PORT_IN] has shape (frame_size, n_channels).
        """
        frame = data[PORT_IN]              # shape: (frame_size, n_channels)
        with self._lock:
            for sample in frame:
                self._deque.append(sample.copy())
        return None

    def latest(self, n: int) -> "np.ndarray | None":
        """
        Return the most recent `n` samples as a float32 array of shape (n, C).

        Returns None if fewer than `n` samples have arrived yet (e.g. shortly
        after pipeline start — the buffer fills at the true hardware rate now
        that step() stores every sample in each frame, not just the first one).
        Callers should treat None as "not ready" and skip processing for that tick.

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


def load_blink_model(model, cfg: dict) -> None:
    """
    Register the trained binary blink EEGNet so check_blink_model() can use it.

    Call once at startup after loading blinknet.pt from disk.  The model must
    already be on the desired device and in eval mode.

    Parameters
    ----------
    model : EEGNet (torch.nn.Module)
        Binary classifier (n_classes=2): class 0 = blink, class 1 = open.
    cfg   : dict
        Config dict from blinknet_config.json.  Required keys:
          'ch_mean', 'ch_std'    — per-channel normalisation
          'n_timepoints'         — window length in samples (e.g. 125)
    """
    global _blink_model, _blink_cfg, _blink_device
    _blink_model  = model
    _blink_cfg    = cfg
    _blink_device = next(model.parameters()).device


# ── Public decode functions ───────────────────────────────────────────────────
def check_blink_state_old() -> bool:
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


def check_blink_state() -> bool:
    """
    Return True if a genuine double blink is detected in the last 1 second.

    Unlike check_blink_state_old(), this version enforces a quiet-baseline gate
    before looking for spikes.  If the rolling median amplitude across the
    1 s window is elevated — caused by movement, loose electrodes, or other
    broadband noise — the function returns False immediately without inspecting
    individual spikes.  Blinks are only reported when the resting signal is
    provably quiet.

    Algorithm
    ---------
    1. Grab the last 1 s (BLINK2_WINDOW = 250 samples) from the buffer.
    2. Compute per-sample mean absolute amplitude across all channels → (T,).
    3. Take the **median** of that amplitude trace as the baseline estimate.
       The median is robust to 1–2 brief spike events, so it still reflects
       the true resting level even when blinks are present.
    4. If baseline > BLINK2_BASELINE_MAX (50 µV), return False immediately.
       High median → high broadband noise → cannot trust any spikes found.
    5. Mark each sample as "spiking" when its amplitude is within the band
       [BLINK2_SPIKE_MIN, BLINK2_SPIKE_MAX] (80–200 µV).
       Below MIN  → resting EEG, not a spike.
       Above MAX  → muscle artifact or electrode pop, not a real blink.
    6. Count contiguous spike runs via rising edges (False→True) on the mask.
    7. Return True only if there are >= 2 separate spike runs.

    Threshold summary
    -----------------
    BLINK2_BASELINE_MAX = 50  µV  — median of window; above this → noisy, skip
    BLINK2_SPIKE_MIN    = 80  µV  — minimum spike amplitude
    BLINK2_SPIKE_MAX    = 200 µV  — maximum spike amplitude (above = artifact)

    Returns
    -------
    bool
        True  — baseline is quiet AND two valid amplitude spikes were found.
        False — baseline too noisy, fewer than 2 spikes, or buffer not ready.

    Raises
    ------
    RuntimeError
        If no EEGBuffer has been added to the pipeline yet.
    """
    if _buffer is None:
        raise RuntimeError(
            "No EEGBuffer has been instantiated — add one to your pipeline."
        )

    window = _buffer.latest(BLINK2_WINDOW)   # shape: (250, n_channels) or None
    if window is None:
        # Buffer hasn't accumulated a full second yet (pipeline just started).
        return False

    # Per-sample mean absolute amplitude across all channels → shape: (T,).
    amp = np.abs(window).mean(axis=1)

    # Baseline gate: use the median — robust to the brief spike events themselves.
    # If the typical signal level is elevated, the person is moving or the
    # electrodes are noisy; don't report blinks in that state.
    baseline = float(np.median(amp))
    if baseline > BLINK2_BASELINE_MAX:
        return False

    # Spike mask: samples within the valid blink amplitude band.
    spike_mask = (amp > BLINK2_SPIKE_MIN) & (amp < BLINK2_SPIKE_MAX)

    # Count rising edges (False→True transitions) = number of separate spikes.
    spike_count = int((np.diff(spike_mask.astype(np.int8)) == 1).sum())

    # Require exactly (at least) 2 separate spikes to confirm a double blink.
    return spike_count >= 2


def check_blink_model() -> bool:
    """
    Return True if the trained binary blink model predicts a blink in the
    last 500 ms (BLINK2_WINDOW = 125 samples).

    Requires load_blink_model() to have been called at startup.
    Returns False silently if the model is not loaded or the buffer is not
    full yet — identical contract to check_blink_state().

    Returns
    -------
    bool
        True  — model predicts class 0 (blink).
        False — model predicts class 1 (open), not loaded, or not ready.
    """
    if _blink_model is None or _buffer is None:
        return False

    n_timepoints = _blink_cfg["n_timepoints"]
    window = _buffer.latest(n_timepoints)    # (n_timepoints, n_channels) or None
    if window is None:
        return False

    mean = np.array(_blink_cfg["ch_mean"], dtype=np.float32)
    std  = np.array(_blink_cfg["ch_std"],  dtype=np.float32)
    win  = (window - mean) / (std + 1e-8)    # (T, C)

    # EEGNet input: (batch=1, in_channels=1, eeg_channels=C, timepoints=T)
    x = torch.from_numpy(win.T[np.newaxis, np.newaxis]).float().to(_blink_device)

    with torch.no_grad():
        logits = _blink_model(x)             # (1, 2)
        pred   = int(logits.argmax(dim=1))   # 0 = blink, 1 = open

    return pred == 0


def decode_emotion() -> "tuple[str, float, dict[str, float]] | None":
    """
    Run the emotion model on the most recent 1 s of EEG.

    Returns
    -------
    (label, confidence, probs) : tuple[str, float, dict[str, float]]
        label      — winning emotion string from cfg['emotions'].
        confidence — softmax probability of the top class, in [0, 1].
        probs      — {emotion_name: probability} for every class.
    None
        If the model isn't loaded or the buffer hasn't filled yet.
    """
    if _emotion_model is None or _buffer is None:
        return None

    window = _buffer.latest(EMOTION_WINDOW)   # (EMOTION_WINDOW, n_channels) or None
    if window is None:
        return None

    mean = np.array(_emotion_cfg["ch_mean"], dtype=np.float32)  # (C,)
    std  = np.array(_emotion_cfg["ch_std"],  dtype=np.float32)  # (C,)
    win  = (window - mean) / (std + 1e-8)                       # (T, C)

    x = torch.from_numpy(win.T[np.newaxis, np.newaxis]).float().to(_emotion_device)

    with torch.no_grad():
        logits = _emotion_model(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (n_classes,)

    idx    = int(probs.argmax())
    label  = _emotion_cfg["emotions"][idx]
    conf   = float(probs[idx])
    probs_dict = {name: float(probs[i])
                  for i, name in enumerate(_emotion_cfg["emotions"])}
    return label, conf, probs_dict
