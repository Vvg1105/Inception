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
import gtec_ble as ble

from gpype.backend.core.i_node import INode
from gpype.common.constants import Constants

# ── Configuration ─────────────────────────────────────────────────────────────
EMOTIONS     = ["sad", "happy"]
N_TRIALS     = 3       # trials per emotion — more trials = more onset events = better model
TRIAL_SECS   = 7       # seconds per trial — short enough that emotion stays strong throughout
REST_SECS    = 4       # rest between trials (not recorded, just a break)
FS           = 250
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
REST_LABEL   = len(EMOTIONS)   # sentinel label (2) used to mark trial boundaries in the array

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
        self._total_received: int = 0   # all samples ever seen, recording or not

    # ── Called from main thread ───────────────────────────────────────────────
    def is_flowing(self) -> bool:
        """Return True once the pipeline has delivered at least one sample."""
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
        """Return (samples, labels) as numpy arrays after pipeline stops."""
        return np.array(self._samples, dtype=np.float32), \
               np.array(self._labels,  dtype=np.int64)

    # ── Called from pipeline thread ───────────────────────────────────────────
    def step(self, data: dict) -> None:
        # data[PORT_IN] shape: (frame_size, n_channels)
        # frame_size may be > 1 depending on BCICore8 buffer config, so
        # iterate all rows rather than taking only [0].
        frame = data[PORT_IN]
        with self._lock:
            self._total_received += len(frame)
            if self._recording:
                for sample in frame:
                    self._samples.append(sample.copy())
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
    print(f"  {len(EMOTIONS)} emotions × {N_TRIALS} trials × {TRIAL_SECS}s  @  {FS} Hz  ×  8 channels")
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

    # ── Verify BCI Core-8 is reachable before starting the pipeline ──────────
    # Amplifier.__init__ scans BLE for ~6 s and throws ValueError if nothing is
    # found.  That exception is swallowed inside gpype's thread, leaving the
    # pipeline "Healthy" with 0 samples forever.  Scan here first so we get a
    # clear error message before wasting time.
    import gtec_ble as ble
    print("\nScanning for BCI Core-8 (up to 6 s) ...")
    devices = ble.Amplifier.get_connected_devices()
    if not devices:
        print("  ERROR: No BCI Core-8 found over BLE.")
        print("  → Power on the headset and make sure Bluetooth is enabled.")
        return
    print(f"  Found device(s): {devices}")

    # ── Start pipeline (runs in background threads) ───────────────────────────
    print("Starting pipeline ...")
    p.start()

    # Wait until the first sample actually arrives at the recorder.
    CONNECT_TIMEOUT_S = 15
    print("  Waiting for data flow", end="", flush=True)
    deadline = time.time() + CONNECT_TIMEOUT_S
    while not recorder.is_flowing():
        if time.time() > deadline:
            print(f"\n  ERROR: Pipeline started but no samples received after "
                  f"{CONNECT_TIMEOUT_S}s — check BLE connection.")
            p.stop()
            return
        print(".", end="", flush=True)
        time.sleep(0.5)
    print(f"  flowing ({recorder._total_received} samples received)")
    time.sleep(1)   # brief extra settle time

    # ── Recording loop ────────────────────────────────────────────────────────
    # Each emotion gets N_TRIALS short bursts.  Short trials keep the emotion
    # signal strong (emotion fades after ~10-12 s of sustained effort).
    # REST_SECS between trials is unrecorded — lets the person reset before the next onset.
    for idx, emotion in enumerate(EMOTIONS):
        print(f"\n{'═'*50}")
        print(f"  Emotion: {emotion.upper()}  ({N_TRIALS} trials × {TRIAL_SECS}s)")
        print(f"{'─'*50}")
        for trial in range(N_TRIALS):
            input(f"  [Trial {trial+1}/{N_TRIALS}] Press Enter when ready ...")
            countdown(3)
            print(f"  [REC] Feel {emotion.upper()} intensely for {TRIAL_SECS}s ...")
            recorder.start_recording(idx)
            progress_bar(TRIAL_SECS, emotion)
            recorder.stop_recording()
            n_recorded = sum(1 for l in recorder._labels if l == idx)
            print(f"  Done. ({n_recorded} samples total for {emotion})")
            if trial < N_TRIALS - 1:
                print(f"  Rest {REST_SECS}s — clear your mind ...")
                time.sleep(REST_SECS)

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
