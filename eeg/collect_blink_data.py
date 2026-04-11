"""
Blink data collection — BCI Core-8 / g.tec headset.

Records two 20-second trials:
  0 = BLINK  — blink continuously and naturally throughout the window
  1 = OPEN   — keep eyes open and still, no blinking

Output
------
  eeg/data/blink_raw.npy    float32 (N, 8)  — filtered EEG samples
  eeg/data/blink_labels.npy int64   (N,)    — 0=blink  1=open

Run with conda base:
  conda run -n base python eeg/collect_blink_data.py
"""
import os
import sys
import threading
import time

import numpy as np
import gpype as gp
import gtec_ble as ble

from gpype.backend.core.i_node import INode
from gpype.common.constants import Constants

# ── Configuration ──────────────────────────────────────────────────────────────
CLASSES     = [("blink", 0), ("open", 1)]
RECORD_SECS = 30
FS          = 250
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
PORT_IN     = Constants.Defaults.PORT_IN


# ── Recorder node ──────────────────────────────────────────────────────────────
class EEGRecorder(INode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lock          = threading.Lock()
        self._recording     = False
        self._current_label = -1
        self._samples: list = []
        self._labels: list  = []
        self._total_received: int = 0

    def is_flowing(self) -> bool:
        with self._lock:
            return self._total_received > 0

    def start_recording(self, label: int):
        with self._lock:
            self._current_label = label
            self._recording     = True

    def stop_recording(self):
        with self._lock:
            self._recording = False

    def get_data(self):
        return (np.array(self._samples, dtype=np.float32),
                np.array(self._labels,  dtype=np.int64))

    def step(self, data: dict) -> None:
        frame = data[PORT_IN]
        with self._lock:
            self._total_received += len(frame)
            if self._recording:
                for sample in frame:
                    self._samples.append(sample.copy())
                    self._labels.append(self._current_label)
        return None


# ── Helpers ────────────────────────────────────────────────────────────────────
def countdown(secs: int):
    for s in range(secs, 0, -1):
        print(f"  Starting in {s}...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 30, end="\r")


def progress_bar(secs: int, label: str):
    bar_width = 40
    start     = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed >= secs:
            break
        frac    = elapsed / secs
        filled  = int(bar_width * frac)
        bar     = "#" * filled + "-" * (bar_width - filled)
        print(f"  [{bar}] {secs - elapsed:.1f}s  {label.upper()}", end="\r", flush=True)
        time.sleep(0.05)
    print(" " * 70, end="\r")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("  EEG Blink Data Collection")
    print(f"  2 classes × {RECORD_SECS}s  @  {FS} Hz  ×  8 channels")
    print("=" * 60)

    # ── BLE pre-check ─────────────────────────────────────────────────────────
    print("\nScanning for BCI Core-8 (up to 6 s) ...")
    devices = ble.Amplifier.get_connected_devices()
    if not devices:
        print("  ERROR: No BCI Core-8 found — power on the headset and retry.")
        return
    print(f"  Found: {devices}")

    # ── Build pipeline ────────────────────────────────────────────────────────
    p        = gp.Pipeline()
    source   = gp.BCICore8()
    bandpass = gp.Bandpass(f_lo=0.5, f_hi=45)
    notch50  = gp.Bandstop(f_lo=48,  f_hi=52)
    notch60  = gp.Bandstop(f_lo=58,  f_hi=62)
    recorder = EEGRecorder()

    p.connect(source,   bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50,  notch60)
    p.connect(notch60,  recorder)

    # ── Start + wait for data flow ────────────────────────────────────────────
    print("Starting pipeline ...")
    p.start()

    CONNECT_TIMEOUT_S = 15
    print("  Waiting for data flow", end="", flush=True)
    deadline = time.time() + CONNECT_TIMEOUT_S
    while not recorder.is_flowing():
        if time.time() > deadline:
            print(f"\n  ERROR: No samples after {CONNECT_TIMEOUT_S}s.")
            p.stop()
            return
        print(".", end="", flush=True)
        time.sleep(0.5)
    print(f"  flowing")
    time.sleep(1)

    # ── Recording loop ────────────────────────────────────────────────────────
    for name, label in CLASSES:
        print(f"\n{'─'*50}")
        if name == "blink":
            print(f"  Trial: BLINK")
            print("  Blink naturally and continuously for the full 20 seconds.")
        else:
            print(f"  Trial: EYES OPEN")
            print("  Keep your eyes open and still. Do NOT blink.")
        input("  Press Enter when ready ...")
        countdown(3)
        print(f"  [REC] {RECORD_SECS}s ...")
        recorder.start_recording(label)
        progress_bar(RECORD_SECS, name)
        recorder.stop_recording()
        count = sum(1 for l in recorder._labels if l == label)
        print(f"  Done. Collected {count} samples ({count/FS:.1f}s)")

    p.stop()

    # ── Save ──────────────────────────────────────────────────────────────────
    raw, labels = recorder.get_data()
    print(f"\n{'='*50}")
    print(f"  Total samples : {len(raw)}")
    for name, i in CLASSES:
        count = int((labels == i).sum())
        print(f"  {name:>6} ({i}): {count} samples  ({count/FS:.1f}s)")

    raw_path    = os.path.join(DATA_DIR, "blink_raw.npy")
    labels_path = os.path.join(DATA_DIR, "blink_labels.npy")
    np.save(raw_path,    raw)
    np.save(labels_path, labels)
    print(f"\n  Saved: {raw_path}")
    print(f"  Saved: {labels_path}")
    print("  Collection complete.\n")


if __name__ == "__main__":
    main()
