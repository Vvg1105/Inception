#!/usr/bin/env python3
"""
Real-time two-class EMG prediction using a trained joblib model.

Uses the same window length and features as training (read from the model bundle).
Fill the rolling buffer with live serial samples, then classify every --stride samples.
Optional majority vote over the last few predictions reduces flicker.

Train first:
  python train_classifier.py --data data/emg_two_movements.npz --out models/emg_two_movements.joblib

Then:
  python live_classify.py --model models/emg_two_movements.joblib

Your ~2 s gesture should match the number of samples you used when collecting
(--window in collect_dataset.py). If the bundle says window=200, that is 200
serial samples ≈ 2 s only if your Arduino sends ~100 samples/s.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque

_EMG_DIR = os.path.dirname(os.path.abspath(__file__))
if _EMG_DIR not in sys.path:
    sys.path.insert(0, _EMG_DIR)

import joblib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from features import emg_features
from serial_emg import open_emg_serial


def parse_args():
    p = argparse.ArgumentParser(description="Live EMG movement classification")
    p.add_argument(
        "--model",
        default="models/emg_two_movements.joblib",
        help="joblib from train_classifier.py",
    )
    p.add_argument("--port", default=None)
    p.add_argument("--baud", type=int, default=9600)
    p.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Predict every N new samples (after buffer is full)",
    )
    p.add_argument(
        "--votes",
        type=int,
        default=7,
        help="Majority vote over last K predictions (1 = no smoothing)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    try:
        bundle = joblib.load(args.model)
    except OSError as e:
        print(f"Could not load model {args.model}: {e}")
        sys.exit(1)

    pipeline = bundle["pipeline"]
    window = int(bundle.get("window", 200))
    names = bundle.get("class_names")
    if names is None:
        names = ["class_0", "class_1"]
    stride = max(1, args.stride)
    vote_n = max(1, args.votes)

    ser, port = open_emg_serial(args.port, baud=args.baud)
    if not ser:
        print("Serial device not found.")
        sys.exit(1)
    print(f"Port: {port}, window={window} samples, stride={stride}, votes={vote_n}")
    print("Click the plot window for keyboard focus.  q  quit")

    buf = deque(maxlen=window)
    pred_history: deque[int] = deque(maxlen=vote_n)
    sample_i = 0
    last_display = "…filling buffer…"
    last_probs = ""

    fig, ax = plt.subplots(figsize=(11, 4.5))
    plot_buf = deque([0] * window, maxlen=window)
    line, = ax.plot(range(window), list(plot_buf), color="cyan", linewidth=1.2)
    status = ax.text(
        0.5,
        1.06,
        "",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#e8e8ef",
    )
    sub = ax.text(
        0.5,
        1.015,
        "",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#a0a0b8",
    )
    ax.set_ylim(0, 1023)
    ax.set_xlim(0, window - 1)
    ax.set_ylabel("Raw ADC")
    ax.set_xlabel("Sample (oldest → newest)")
    ax.set_facecolor("#16213e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for l in (ax.xaxis.label, ax.yaxis.label):
        l.set_color("white")

    def on_key(event):
        if event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame):
        nonlocal sample_i, last_display, last_probs
        try:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if raw.isdigit():
                v = int(raw)
                buf.append(v)
                plot_buf.append(v)
                line.set_ydata(list(plot_buf))
                sample_i += 1
                if len(buf) == window and sample_i % stride == 0:
                    w = np.asarray(buf, dtype=np.float64)
                    feat = emg_features(w).reshape(1, -1)
                    pred = int(pipeline.predict(feat)[0])
                    pred_history.append(pred)
                    voted = max(set(pred_history), key=list(pred_history).count)
                    last_display = names[voted] if voted < len(names) else str(voted)
                    if hasattr(pipeline, "predict_proba"):
                        p = pipeline.predict_proba(feat)[0]
                        parts = [
                            f"{names[i][:10]}:{p[i]:.2f}"
                            for i in range(min(len(names), len(p)))
                        ]
                        last_probs = "  ".join(parts)
        except (OSError, ValueError):
            pass

        fill = len(buf)
        if fill < window:
            status.set_text(f"Collecting… {fill}/{window} samples")
            sub.set_text("Perform your ~2 s gesture once buffer is full")
        else:
            status.set_text(f"Prediction →  {last_display}")
            sub.set_text(last_probs)
        return (line, status, sub)

    ani = animation.FuncAnimation(
        fig, update, interval=10, blit=False, cache_frame_data=False
    )
    plt.tight_layout()
    try:
        plt.show()
    finally:
        ser.close()


if __name__ == "__main__":
    main()
