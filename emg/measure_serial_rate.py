#!/usr/bin/env python3
"""
Measure how many EMG integer lines per second the Arduino actually sends.

Run while the board is streaming (same as emg_viewer / collect_dataset).
Uses wall-clock time, so you get real Hz, not whatever you assumed.

Examples:
  python measure_serial_rate.py
  python measure_serial_rate.py --duration 15
  python measure_serial_rate.py --port /dev/tty.usbmodem1101
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_EMG_DIR = os.path.dirname(os.path.abspath(__file__))
if _EMG_DIR not in sys.path:
    sys.path.insert(0, _EMG_DIR)

from serial_emg import open_emg_serial


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default=None)
    p.add_argument("--baud", type=int, default=9600)
    p.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Seconds to average over (default: 10)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.duration <= 0:
        print("--duration must be positive")
        sys.exit(1)

    ser, port = open_emg_serial(args.port, baud=args.baud)
    if not ser:
        print("Serial device not found.")
        sys.exit(1)
    print(f"Port: {port}  |  collecting for {args.duration:g} s…\n")

    ser.reset_input_buffer()
    t0 = time.perf_counter()
    n_ok = 0
    n_other = 0
    deadline = t0 + args.duration

    while time.perf_counter() < deadline:
        try:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if raw.isdigit():
                n_ok += 1
            elif raw:
                n_other += 1
        except OSError:
            break

    elapsed = time.perf_counter() - t0
    ser.close()

    if elapsed < 0.5:
        print("Stopped almost immediately; check connection.")
        sys.exit(1)

    hz = n_ok / elapsed
    print(f"Valid integer lines: {n_ok}  in  {elapsed:.3f} s")
    print(f"Estimated rate:      {hz:.2f}  lines/s (Hz)")
    if n_other:
        print(f"(Also saw {n_other} non-empty non-integer lines.)")
    print()
    print("Use this value as --hz in collect_dataset.py, e.g.:")
    print(f"  python collect_dataset.py --seconds 1.5 --hz {hz:.0f}")


if __name__ == "__main__":
    main()
