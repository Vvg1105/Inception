"""
EEG emotion data collection — BCI Core-8 / g.tec gtec headset.

Records 4 emotions (angry, sad, happy, fear) for RECORD_SECS each.
Prompts you between trials so you can compose yourself.

Output
------
  eeg/data/eeg_raw.npy    float32 (N, 8)  — filtered EEG samples
  eeg/data/eeg_labels.npy int64   (N,)    — 0=angry 1=sad 2=happy 3=fear

Run with conda base (has gpype + numpy):
  conda run -n base python eeg/collect_data.py
"""
import os
import sys
import threading
import time

import numpy as np
import gpype as gp

from gpype.backend.core.i_node import INode
from gpype.common.constants import Constants

# ── Configuration ─────────────────────────────────────────────────────────────
EMOTIONS     = ["angry", "sad", "happy", "fear"]
RECORD_SECS  = 10
FS           = 250
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")

PORT_IN = Constants.Defaults.PORT_IN


# ── Custom gpype sink node ────────────────────────────────────────────────────
class EEGRecorder(INode):
    """
    Pipeline sink that accumulates EEG samples when recording is active.

    Thread-safe: step() is called from the gpype pipeline thread;
    start_recording / stop_recording are called from the main thread.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lock          = threading.Lock()
        self._recording     = False
        self._current_label = -1
        self._samples: list = []
        self._labels: list  = []

    # ── Called from main thread ───────────────────────────────────────────────
    def start_recording(self, label: int):
        with self._lock:
            self._current_label = label
            self._recording     = True

    def stop_recording(self):
        with self._lock:
            self._recording = False

    def get_data(self):
        """Return (samples, labels) as numpy arrays after pipeline stops."""
        return np.array(self._samples, dtype=np.float32), \
               np.array(self._labels,  dtype=np.int64)

    # ── Called from pipeline thread ───────────────────────────────────────────
    def step(self, data: dict) -> None:
        with self._lock:
            if self._recording:
                # data[PORT_IN] shape: (1, n_channels) — one sample at a time
                self._samples.append(data[PORT_IN][0].copy())
                self._labels.append(self._current_label)
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────
def countdown(secs: int):
    for s in range(secs, 0, -1):
        print(f"  Starting in {s}...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 30, end="\r")


def progress_bar(secs: int, label: str):
    """Blocking progress bar while recording."""
    bar_width = 40
    start     = time.time()
    while True:
        elapsed  = time.time() - start
        if elapsed >= secs:
            break
        frac  = elapsed / secs
        filled = int(bar_width * frac)
        bar   = "#" * filled + "-" * (bar_width - filled)
        remaining = secs - elapsed
        print(f"  [{bar}] {remaining:.1f}s  FEEL {label.upper()}", end="\r", flush=True)
        time.sleep(0.05)
    print(" " * 70, end="\r")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("  EEG Emotion Data Collection")
    print(f"  {len(EMOTIONS)} emotions × {RECORD_SECS}s  @  {FS} Hz  ×  8 channels")
    print("=" * 60)

    # ── Build gpype pipeline ──────────────────────────────────────────────────
    p = gp.Pipeline()

    source      = gp.BCICore8()
    bandpass    = gp.Bandpass(f_lo=0.5, f_hi=45)
    notch50     = gp.Bandstop(f_lo=48,  f_hi=52)
    notch60     = gp.Bandstop(f_lo=58,  f_hi=62)
    recorder    = EEGRecorder()

    p.connect(source,   bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50,  notch60)
    p.connect(notch60,  recorder)

    # ── Start pipeline (runs in background threads) ───────────────────────────
    print("\nConnecting to BCI Core-8 ...")
    p.start()
    time.sleep(2)   # let amplifier settle

    # ── Recording loop ────────────────────────────────────────────────────────
    for idx, emotion in enumerate(EMOTIONS):
        print(f"\n{'─'*50}")
        print(f"  Trial {idx+1}/{len(EMOTIONS)}: {emotion.upper()}")
        input("  Press Enter when ready ...")
        countdown(3)
        print(f"  [REC] Feeling {emotion.upper()} for {RECORD_SECS}s ...")
        recorder.start_recording(idx)
        progress_bar(RECORD_SECS, emotion)
        recorder.stop_recording()
        samples_so_far = recorder._samples
        print(f"  Done. Collected {sum(1 for l in recorder._labels if l == idx)} samples.")

    # ── Tear down ─────────────────────────────────────────────────────────────
    p.stop()

    raw, labels = recorder.get_data()
    print(f"\n{'='*50}")
    print(f"  Total samples : {len(raw)}")
    for i, e in enumerate(EMOTIONS):
        count = int((labels == i).sum())
        print(f"    {e:>8s} ({i}): {count} samples  ({count/FS:.1f}s)")

    # ── Save ──────────────────────────────────────────────────────────────────
    raw_path    = os.path.join(DATA_DIR, "eeg_raw.npy")
    labels_path = os.path.join(DATA_DIR, "eeg_labels.npy")
    np.save(raw_path,    raw)
    np.save(labels_path, labels)
    print(f"\n  Saved: {raw_path}")
    print(f"  Saved: {labels_path}")
    print("  Collection complete.\n")


if __name__ == "__main__":
    main()
