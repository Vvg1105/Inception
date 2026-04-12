"""
EEG emotion data collection — g.tec BCI Core-8 (gtec) or OpenBCI Cyton.

Records emotions for TRIAL_SECS each, N_TRIALS per emotion.
Each run is saved as a numbered file so multiple sessions can be combined
during training without overwriting previous data.

Output (one pair per run)
------
  eeg/data/run_001_raw.npy    float32 (N, 8)  — filtered EEG samples
  eeg/data/run_001_labels.npy int64   (N,)    — 0=sad 1=happy ...
  eeg/data/run_002_raw.npy    ...
  ...

Usage:
  # g.tec (default, requires conda base with gpype)
  conda run -n base python eeg/collect_data.py

  # OpenBCI Cyton (USB dongle, requires brainflow + scipy)
  python eeg/collect_data.py --board cyton --port /dev/ttyUSB0
  ******RUN "ls -l /dev/cu.* /dev/tty.*" IN TERMINAL TO FIND CORRECT SERIAL PORT FOR CYTON
"""
import argparse
import os
import threading
import time

import numpy as np


# ── Configuration ─────────────────────────────────────────────────────────────
EMOTIONS     = ["sad", "happy"]
N_TRIALS     = 10    # trials per emotion
TRIAL_SECS   = 5         # seconds per trial
REST_SECS    = 0.5       # rest between trials (not recorded)
FS           = 250
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
REST_LABEL   = len(EMOTIONS)   # sentinel label (2)


# ── gtec recorder (gpype INode sink) ─────────────────────────────────────────
def _make_gtec_recorder():
    """Import gpype and return an EEGRecorder INode instance."""
    import gpype as gp
    from gpype.backend.core.i_node import INode
    from gpype.common.constants import Constants

    PORT_IN = Constants.Defaults.PORT_IN

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
            frame = data[PORT_IN]  # (frame_size, n_channels)
            with self._lock:
                self._total_received += len(frame)
                if self._recording:
                    for sample in frame:
                        self._samples.append(sample.copy())
                        self._labels.append(self._current_label)
            return None

    return EEGRecorder()


# ── Cyton recorder (BrainFlow + scipy) ───────────────────────────────────────
class CytonEEGRecorder:
    """
    Polls BrainFlow's Cyton board in a background thread and accumulates
    raw EEG samples when recording is active.  Filters are applied in batch
    via BrainFlow DataFilter when get_data() is called.

    Thread-safe: _poll_loop runs in a daemon thread; start_recording /
    stop_recording / is_flowing are called from the main thread.
    """

    def __init__(self, serial_port: str = ""):
        try:
            from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
            from brainflow.data_filter import DataFilter, FilterTypes
        except ImportError:
            raise ImportError("pip install brainflow")

        self._BoardShim          = BoardShim
        self._BoardIds           = BoardIds
        self._BrainFlowInputParams = BrainFlowInputParams
        self._DataFilter         = DataFilter
        self._FilterTypes        = FilterTypes

        self._serial_port  = serial_port
        self._board        = None
        self._eeg_channels: list = []

        self._lock          = threading.Lock()
        self._recording     = False
        self._current_label = -1
        self._raw_samples: list = []   # list of 1-D arrays (n_channels,)
        self._labels: list      = []
        self._total_received    = 0

        self._running      = False
        self._poll_thread  = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start(self):
        self._BoardShim.disable_board_logger()
        params = self._BrainFlowInputParams()
        if self._serial_port:
            params.serial_port = self._serial_port

        self._board = self._BoardShim(self._BoardIds.CYTON_BOARD.value, params)
        print(f"  [Cyton] opening board on {self._serial_port or '(auto)'} ...", flush=True)
        self._board.prepare_session()
        self._board.start_stream()
        self._eeg_channels = self._BoardShim.get_eeg_channels(self._BoardIds.CYTON_BOARD.value)
        print(f"  [Cyton] streaming — EEG channels: {self._eeg_channels}", flush=True)

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="cyton-collect"
        )
        self._poll_thread.start()

    def stop(self):
        self._running = False
        if self._board is not None:
            try:
                self._board.stop_stream()
                self._board.release_session()
            except Exception:
                pass
            self._board = None

    # ── Recording control ─────────────────────────────────────────────────────
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
        """
        Return (raw_filtered, labels) as numpy arrays.
        Applies bandpass (0.5–45 Hz) + notch 50 Hz + notch 60 Hz per channel.
        """
        raw    = np.array(self._raw_samples, dtype=np.float64)  # (N, n_ch)
        labels = np.array(self._labels, dtype=np.int64)

        if raw.size == 0:
            return raw.astype(np.float32), labels

        for c in range(raw.shape[1]):
            ch = np.ascontiguousarray(raw[:, c])
            self._DataFilter.perform_bandpass(
                ch, FS, 0.5, 45.0, 4,
                self._FilterTypes.BUTTERWORTH.value, 0
            )
            self._DataFilter.perform_bandstop(
                ch, FS, 48.0, 52.0, 4,
                self._FilterTypes.BUTTERWORTH.value, 0
            )
            self._DataFilter.perform_bandstop(
                ch, FS, 58.0, 62.0, 4,
                self._FilterTypes.BUTTERWORTH.value, 0
            )
            raw[:, c] = ch

        return raw.astype(np.float32), labels

    # ── Internal poll loop ────────────────────────────────────────────────────
    def _poll_loop(self):
        """Drains BrainFlow ring buffer every 40 ms and stores EEG samples."""
        while self._running:
            try:
                if self._board is not None:
                    count = self._board.get_board_data_count()
                    if count > 0:
                        data = self._board.get_board_data(count)
                        # data: (n_total_channels, n_samples) → transpose
                        eeg = data[self._eeg_channels, :].T  # (n_samples, n_ch)
                        with self._lock:
                            self._total_received += len(eeg)
                            if self._recording:
                                for sample in eeg:
                                    self._raw_samples.append(sample.copy())
                                    self._labels.append(self._current_label)
            except Exception:
                pass
            time.sleep(0.04)


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
        elapsed = time.time() - start
        if elapsed >= secs:
            break
        frac   = elapsed / secs
        filled = int(bar_width * frac)
        bar    = "#" * filled + "-" * (bar_width - filled)
        print(f"  [{bar}] {secs - elapsed:.1f}s  FEEL {label.upper()}", end="\r", flush=True)
        time.sleep(0.05)
    print(" " * 70, end="\r")


# ── Run helpers ───────────────────────────────────────────────────────────────
def _find_runs(data_dir: str) -> list:
    import re
    runs = []
    for fname in os.listdir(data_dir):
        m = re.match(r"run_(\d+)_raw\.npy$", fname)
        if m:
            n = int(m.group(1))
            if os.path.exists(os.path.join(data_dir, f"run_{n:03d}_labels.npy")):
                runs.append(n)
    return sorted(runs)


def _next_run_number(data_dir: str) -> int:
    existing = _find_runs(data_dir)
    return (max(existing) + 1) if existing else 1


# ── Board setup helpers ───────────────────────────────────────────────────────
def _setup_gtec():
    """Build and start a gpype pipeline for the g.tec BCI Core-8."""
    import gpype as gp

    p        = gp.Pipeline()
    source   = gp.BCICore8()
    bandpass = gp.Bandpass(f_lo=0.5, f_hi=45)
    notch50  = gp.Bandstop(f_lo=48,  f_hi=52)
    notch60  = gp.Bandstop(f_lo=58,  f_hi=62)
    recorder = _make_gtec_recorder()

    p.connect(source,   bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50,  notch60)
    p.connect(notch60,  recorder)

    print("Starting gpype pipeline ...")
    p.start()
    return p, recorder


def _setup_cyton(serial_port: str):
    """Start a BrainFlow session for the OpenBCI Cyton board."""
    recorder = CytonEEGRecorder(serial_port=serial_port)
    recorder.start()
    return recorder


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EEG emotion data collection")
    parser.add_argument(
        "--board", choices=["gtec", "cyton"], default="gtec",
        help="EEG board to use (default: gtec)"
    )
    parser.add_argument(
        "--port", default="",
        help="Serial port for Cyton USB dongle, e.g. /dev/ttyUSB0 or COM3"
    )
    args = parser.parse_args()

    # Save into a board-specific subdirectory so train.py --gtec/--cyton finds the right data
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data", args.board)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("  EEG Emotion Data Collection")
    print(f"  Board  : {args.board.upper()}")
    print(f"  {len(EMOTIONS)} emotions × {N_TRIALS} trials × {TRIAL_SECS}s  @  {FS} Hz  ×  8 channels")
    print("=" * 60)

    # ── Board-specific setup ──────────────────────────────────────────────────
    pipeline = None   # only used for gtec

    if args.board == "gtec":
        pipeline, recorder = _setup_gtec()
    else:  # cyton
        recorder = _setup_cyton(args.port)

    # ── Wait for data flow ────────────────────────────────────────────────────
    CONNECT_TIMEOUT_S = 15
    print("  Waiting for data flow", end="", flush=True)
    deadline = time.time() + CONNECT_TIMEOUT_S
    while not recorder.is_flowing():
        if time.time() > deadline:
            print(f"\n  ERROR: No samples received after {CONNECT_TIMEOUT_S}s.")
            if pipeline:
                pipeline.stop()
            else:
                recorder.stop()
            return
        print(".", end="", flush=True)
        time.sleep(0.5)
    print(f"  flowing ({recorder._total_received} samples received)")
    time.sleep(1)   # brief extra settle time

    # ── Recording loop ────────────────────────────────────────────────────────
    for idx, emotion in enumerate(EMOTIONS):
        print(f"\n{'═'*50}")
        print(f"  Emotion: {emotion.upper()}  ({N_TRIALS} trials × {TRIAL_SECS}s)")
        print(f"{'─'*50}")
        for trial in range(N_TRIALS):
            input(f"  [Trial {trial+1}/{N_TRIALS}] Press Enter when ready ...")
            print(f"  [REC] Feel {emotion.upper()} intensely for {TRIAL_SECS}s ...")
            recorder.start_recording(idx)
            progress_bar(TRIAL_SECS, emotion)
            recorder.stop_recording()
            n_recorded = sum(1 for lbl in recorder._labels if lbl == idx)
            print(f"  Done. ({n_recorded} samples total for {emotion})")
            if trial < N_TRIALS - 1:
                print(f"  Rest {REST_SECS}s — clear your mind ...")
                time.sleep(REST_SECS)

    # ── Tear down ─────────────────────────────────────────────────────────────
    if pipeline:
        pipeline.stop()
    else:
        recorder.stop()

    raw, labels = recorder.get_data()
    print(f"\n{'='*50}")
    print(f"  Total samples : {len(raw)}")
    for i, e in enumerate(EMOTIONS):
        count = int((labels == i).sum())
        print(f"    {e:>8s} ({i}): {count} samples  ({count/FS:.1f}s)")

    # ── Save (numbered run) ───────────────────────────────────────────────────
    import json
    run_num     = _next_run_number(DATA_DIR)
    raw_path    = os.path.join(DATA_DIR, f"run_{run_num:03d}_raw.npy")
    labels_path = os.path.join(DATA_DIR, f"run_{run_num:03d}_labels.npy")
    meta_path   = os.path.join(DATA_DIR, f"run_{run_num:03d}_meta.json")
    np.save(raw_path,    raw)
    np.save(labels_path, labels)
    with open(meta_path, "w") as f:
        json.dump({"emotions": EMOTIONS}, f)
    print(f"\n  Saved run {run_num:03d}: {raw_path}")
    print(f"  Saved run {run_num:03d}: {labels_path}")
    print(f"  Saved run {run_num:03d}: {meta_path}  {EMOTIONS}")
    existing = sorted(_find_runs(DATA_DIR))
    print(f"  Total runs on disk: {len(existing)} "
          f"({', '.join(f'run_{r:03d}' for r in existing)})")
    print("  Collection complete.\n")


if __name__ == "__main__":
    main()
