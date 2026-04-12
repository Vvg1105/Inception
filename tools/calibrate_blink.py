#!/usr/bin/env python
"""
calibrate_blink.py — Pre-demo blink calibration for the BLINK algorithm.

Records EEG from a frontal channel, runs the two-pass BLINK unsupervised
algorithm, and saves the learned blink profile (delta_new + template waveforms)
to eeg/models/blink_{label}.npz.  Load it at demo time for zero warm-up.

Usage
-----
  # User 2 — OpenBCI Cyton (USB serial):
  python tools/calibrate_blink.py --headset cyton --label user2
  python tools/calibrate_blink.py --headset cyton --label user2 \\
      --port /dev/cu.usbserial-XXXX --ch 0

  # User 1 — GTECH BCICore-8 (Bluetooth via gpype):
  python tools/calibrate_blink.py --headset gtech --label user1 --ch 0

  # Smoke-test with synthetic data (no hardware):
  python tools/calibrate_blink.py --headset mock --label test

Instructions for the subject
-----------------------------
  When prompted, blink naturally at your normal rate for the recording duration
  (default 60 seconds).  Avoid large movements — stay relatively still.
  The algorithm needs about 10–20 natural blinks to learn your fingerprint.

Channel selection
-----------------
  --ch 0  is Fp2 on GTECH BCICore-8 (default, ideal for blinks)
          is the first EEG channel on Cyton (place electrode at Fp1/Fp2!)
  Use the most frontal electrode available — Fp1 or Fp2 are best.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eeg.blink_detector import _lowpass, run_blink_algorithm, CALIB_SECS


# ── Per-headset recorders ─────────────────────────────────────────────────────

def record_cyton(port: str, frontal_ch: int, duration_s: float) -> np.ndarray:
    """Record from OpenBCI Cyton via BrainFlow and return raw frontal channel."""
    try:
        from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
    except ImportError:
        raise SystemExit("pip install brainflow")

    BoardShim.disable_board_logger()
    params = BrainFlowInputParams()
    if port:
        params.serial_port = port

    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()
    board.start_stream()
    eeg_chs = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
    fs = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)

    print(f"  Recording {duration_s:.0f} s @ {fs} Hz  (EEG ch index {frontal_ch}) …")
    _countdown(duration_s)

    data = board.get_current_board_data(int(duration_s * fs) + 500)
    board.stop_stream()
    board.release_session()

    ch_idx = eeg_chs[frontal_ch]
    raw    = data[ch_idx, :].astype(np.float64)
    print(f"  Captured {len(raw)} samples ({len(raw)/fs:.1f} s)")
    return raw, float(fs)


def record_gtech(frontal_ch: int, duration_s: float) -> tuple[np.ndarray, float]:
    """Record from GTECH BCICore-8 via gpype pipeline."""
    try:
        import gpype as gp
    except ImportError:
        raise SystemExit("gpype not installed — cannot record from GTECH headset")

    from eeg.eeg_stream import EEGBuffer, FS

    p        = gp.Pipeline()
    source   = gp.BCICore8()
    bandpass = gp.Bandpass(f_lo=0.5, f_hi=45)
    notch50  = gp.Bandstop(f_lo=48, f_hi=52)
    notch60  = gp.Bandstop(f_lo=58, f_hi=62)
    buf      = EEGBuffer(capacity=int((duration_s + 5) * FS))

    p.connect(source,   bandpass)
    p.connect(bandpass, notch50)
    p.connect(notch50,  notch60)
    p.connect(notch60,  buf)
    p.start()
    print(f"  GTECH BCICore-8 connected.  Recording {duration_s:.0f} s …")
    _countdown(duration_s)

    win = buf.latest(int(duration_s * FS))
    p.stop()

    if win is None:
        raise RuntimeError("Buffer did not fill during recording — check headset connection.")

    raw = win[:, frontal_ch].astype(np.float64)
    print(f"  Captured {len(raw)} samples ({len(raw)/FS:.1f} s)")
    return raw, float(FS)


def record_mock(frontal_ch: int, duration_s: float, fs: float = 250.0) -> tuple[np.ndarray, float]:
    """
    Generate synthetic EEG with injected blink waveforms for smoke-testing.
    Produces a 1 Hz blink rate — 60 s → ~60 blinks, plenty for calibration.
    """
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    # Background: band-limited noise (0.5–20 Hz)
    rng = np.random.default_rng(42)
    sig = rng.normal(0, 15.0, n)
    for fc in [1.0, 4.0, 8.0, 13.0]:
        sig += rng.normal(0, 5.0) * np.sin(2 * np.pi * fc * t + rng.uniform(0, 6.28))

    # Inject blink waveforms every ~1 s (±0.2 s jitter)
    blink_ts = np.arange(2.0, duration_s - 2.0, 1.0)
    blink_ts += rng.uniform(-0.2, 0.2, len(blink_ts))
    for bt in blink_ts:
        onset = int(bt * fs)
        dur_s = rng.uniform(0.25, 0.50)
        dur_n = int(dur_s * fs)
        x     = np.linspace(0, np.pi, dur_n)
        amp   = rng.uniform(120.0, 180.0)
        sig[onset : onset + dur_n] -= amp * np.sin(x)  # downward blink trough

    print(f"  [mock] Generated {n} samples  ({len(blink_ts)} injected blinks)")
    return sig, fs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _countdown(duration_s: float) -> None:
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  Blink naturally at your normal rate.           │")
    print("  │  Avoid large head/body movements.               │")
    print("  │  You need about 10–20 blinks for calibration.   │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    end = time.monotonic() + duration_s
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        print(f"\r  Recording …  {remaining:5.1f} s remaining", end="", flush=True)
        time.sleep(0.25)
    print("\r  Recording complete.                               ", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-demo blink calibration — saves a BLINK profile to disk"
    )
    parser.add_argument("--headset",  choices=["cyton", "gtech", "mock"], required=True)
    parser.add_argument("--label",    default="user",
                        help="Profile name suffix, e.g. 'user1' → blink_user1.npz")
    parser.add_argument("--port",     default="",
                        help="Serial port for Cyton dongle, e.g. /dev/cu.usbserial-XXXX")
    parser.add_argument("--ch",       type=int, default=0,
                        help="Frontal EEG channel column index (default: 0)")
    parser.add_argument("--duration", type=float, default=CALIB_SECS,
                        help=f"Recording duration in seconds (default: {CALIB_SECS:.0f})")
    parser.add_argument("--out-dir",  default=os.path.join(PROJECT_ROOT, "eeg", "models"),
                        help="Directory to save the profile (default: eeg/models/)")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"blink_{args.label}.npz")

    print()
    print(f"  ── Blink calibration  ({args.headset.upper()}, ch={args.ch}) ──")
    print(f"  Output: {out_path}")
    print()

    # ── Record ──────────────────────────────────────────────────────────────
    if args.headset == "cyton":
        raw, fs = record_cyton(args.port, args.ch, args.duration)
    elif args.headset == "gtech":
        raw, fs = record_gtech(args.ch, args.duration)
    else:
        raw, fs = record_mock(args.ch, args.duration)

    # ── Run BLINK algorithm ──────────────────────────────────────────────────
    print()
    print("  Running BLINK algorithm …", flush=True)
    sig    = _lowpass(raw, fc=10.0, fs=fs)
    result = run_blink_algorithm(sig, fs)

    delta  = result["delta_new"]
    tmpls  = result["template_wavs"]
    blinks = result["final_blinks"]

    print(f"  delta_new    : {delta:.2f} µV")
    print(f"  blinks found : {len(blinks)}")
    print(f"  templates    : {len(tmpls)}")

    if len(blinks) < 5:
        print()
        print("  ⚠  Fewer than 5 blinks detected. Consider re-running and blinking")
        print("     more naturally.  The profile will still be saved.")

    # ── Save ─────────────────────────────────────────────────────────────────
    tmpl_arr = np.empty(len(tmpls), dtype=object)
    for i, t in enumerate(tmpls):
        tmpl_arr[i] = t
    np.savez(out_path,
             delta_new=np.array([delta]),
             templates=tmpl_arr)

    print()
    print(f"  ✓  Profile saved → {out_path}")
    print()
    print("  Load at demo time:")
    print(f"      BlinkDetector(fs=250, frontal_ch={args.ch},")
    print(f"                    profile=\"eeg/models/blink_{args.label}.npz\")")
    print()


if __name__ == "__main__":
    main()
