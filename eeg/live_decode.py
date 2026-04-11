"""
Real-time EEG emotion decoding — BCI Core-8 / g.tec gtec headset.

Loads the trained EEGNet weights + normalisation config, then runs a
gpype pipeline that feeds each new sample into a sliding ring-buffer.
Every DECODE_EVERY samples it runs one inference pass and prints the
predicted emotion + confidence to the terminal.

Run after training:
  conda run -n base python eeg/live_decode.py
"""
import os
import sys
import json
import time
import threading
import collections

import numpy as np
import torch

import gpype as gp
from gpype.backend.core.i_node import INode
from gpype.common.constants import Constants

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from eegnet import EEGNet

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
WEIGHTS   = os.path.join(MODEL_DIR, "eegnet_emotion.pt")
CFG_PATH  = os.path.join(MODEL_DIR, "eegnet_config.json")

PORT_IN = Constants.Defaults.PORT_IN

# ── How often to run inference (every N new samples, i.e. N/250 seconds) ──────
DECODE_EVERY = 5     # 5 samples = 20 ms between predictions @ 250 Hz


# ── Custom gpype sink ─────────────────────────────────────────────────────────
class EmotionDecoder(INode):
    """
    Pipeline sink that maintains a sliding window and runs EEGNet inference.

    Parameters
    ----------
    model      : loaded EEGNet
    window     : number of samples per window (int)
    ch_mean    : per-channel z-score mean  (np.ndarray, shape (C,))
    ch_std     : per-channel z-score std   (np.ndarray, shape (C,))
    emotions   : list of emotion label strings
    device     : torch device
    """

    def __init__(self, model, window: int, ch_mean: np.ndarray,
                 ch_std: np.ndarray, emotions: list,
                 device: torch.device, **kwargs):
        super().__init__(**kwargs)
        self._model    = model
        self._window   = window
        self._mean     = ch_mean          # (C,)
        self._std      = ch_std           # (C,)
        self._emotions = emotions
        self._device   = device
        self._buf      = collections.deque(maxlen=window)  # ring buffer of (C,) arrays
        self._counter  = 0

    def step(self, data: dict) -> None:
        sample = data[PORT_IN][0].copy()   # shape (n_channels,)
        self._buf.append(sample)
        self._counter += 1

        # Run inference once per DECODE_EVERY new samples and buffer is full
        if len(self._buf) == self._window and self._counter % DECODE_EVERY == 0:
            self._infer()

        return None

    def _infer(self):
        # Stack ring buffer → (window, C)
        win = np.stack(self._buf)                       # (T, C)
        # Z-score normalise
        win = (win - self._mean) / (self._std + 1e-8)  # (T, C)
        # Reshape to (1, 1, C, T)
        x   = torch.from_numpy(win.T[np.newaxis, np.newaxis]).float()
        x   = x.to(self._device)

        with torch.no_grad():
            logits = self._model(x)                     # (1, n_classes)
            probs  = torch.softmax(logits, dim=1)[0]    # (n_classes,)

        pred_idx = int(probs.argmax())
        pred_lbl = self._emotions[pred_idx]
        conf     = float(probs[pred_idx])

        # ── Terminal display ──────────────────────────────────────────────────
        bar  = _make_prob_bar(probs.cpu().numpy(), self._emotions, width=20)
        time_str = time.strftime("%H:%M:%S")
        print(f"\r[{time_str}]  {pred_lbl.upper():>8s}  ({conf:.0%})    {bar}",
              end="", flush=True)


def _make_prob_bar(probs, emotions, width=20):
    """Compact inline bar: angry[####    ] sad[#       ] ..."""
    parts = []
    for p, e in zip(probs, emotions):
        filled = int(p * width)
        bar    = "#" * filled + " " * (width - filled)
        parts.append(f"{e[:3]}[{bar}]")
    return "  ".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Load config ───────────────────────────────────────────────────────────
    if not os.path.exists(CFG_PATH):
        sys.exit(f"[ERROR] {CFG_PATH} not found — run train.py first.")
    with open(CFG_PATH) as f:
        cfg = json.load(f)

    n_channels   = cfg["n_channels"]
    n_timepoints = cfg["n_timepoints"]
    n_classes    = cfg["n_classes"]
    emotions     = cfg["emotions"]
    ch_mean      = np.array(cfg["ch_mean"],  dtype=np.float32)
    ch_std       = np.array(cfg["ch_std"],   dtype=np.float32)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = EEGNet(n_channels=n_channels, n_timepoints=n_timepoints,
                   n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()
    print(f"Loaded weights: {WEIGHTS}")

    # ── Build gpype pipeline ──────────────────────────────────────────────────
    p = gp.Pipeline()

    source   = gp.BCICore8()
    bandpass = gp.Bandpass(f_lo=0.5, f_hi=45)
    notch50  = gp.Bandstop(f_lo=48,  f_hi=52)
    notch60  = gp.Bandstop(f_lo=58,  f_hi=62)
    decoder  = EmotionDecoder(
        model    = model,
        window   = n_timepoints,
        ch_mean  = ch_mean,
        ch_std   = ch_std,
        emotions = emotions,
        device   = device,
    )

    p.connect(source,   bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50,  notch60)
    p.connect(notch60,  decoder)

    # ── Run ───────────────────────────────────────────────────────────────────
    print("\nConnecting to BCI Core-8 ...")
    print("Live decoding — press Ctrl+C to stop.\n")
    p.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nStopping pipeline ...")
        p.stop()
        print("Done.")


if __name__ == "__main__":
    main()
