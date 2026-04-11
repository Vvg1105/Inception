#!/usr/bin/env python3
"""
Collect labeled EMG trials for two movements.

Two ways to mark a trial:

  A) Segment (recommended): you signal start/end with keys s and e (left-hand friendly):
     Arm with 1/2, press s at movement start, perform, press e at end.
     Samples between s and e are saved (cropped/padded to --window; see --align).

  B) Quick snap: w saves the last WINDOW samples ending now (no start/end).

Keys:
  1 / 2       — arm class
  s / e       — begin / end segment recording
  escape / x  — cancel segment
  w           — quick snap (rolling buffer)
  d / l / q   — save npz / print counts / quit

Add more data later: run with  --append  and the same  --out  to load existing
trials and stack new ones (then  d  overwrites the file with the combined set).

Duration: use  --seconds 1.5 --hz 100  to set window ≈ 150 (adjust --hz to match
your real Serial rate, or timing will be wrong).

The window shows a Status panel: last key pressed, armed class (1/2),
what to do next, and a short action log.

Arduino should send one integer per line (0–1023), same as emg_viewer.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque

STATUS_LOG_MAX = 6

KEYS_BAR = (
    "1/2 arm   ·   s start   e end   ·   w snap   ·   Esc/x cancel   ·   d save   ·   l log   ·   q quit"
)

_EMG_DIR = os.path.dirname(os.path.abspath(__file__))
if _EMG_DIR not in sys.path:
    sys.path.insert(0, _EMG_DIR)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from serial_emg import open_emg_serial


def parse_args():
    p = argparse.ArgumentParser(description="Collect two-class EMG dataset")
    p.add_argument("--port", default=None, help="Serial port (default: auto-detect)")
    p.add_argument("--baud", type=int, default=9600)
    p.add_argument(
        "--window",
        type=int,
        default=200,
        help="Samples per saved trial (ignored if --seconds and --hz are set)",
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=None,
        metavar="SEC",
        help="Target gesture length in seconds; use with --hz to set --window",
    )
    p.add_argument(
        "--hz",
        type=float,
        default=None,
        metavar="RATE",
        help="Approx. Arduino samples per second (Serial lines/s); use with --seconds",
    )
    p.add_argument(
        "--max-segment",
        type=int,
        default=None,
        help="Max samples per s–e segment (default: 15 × window)",
    )
    p.add_argument(
        "--align",
        choices=("end", "center", "start"),
        default="end",
        help="If segment length != window: end=keep samples before e; center crop; start=keep after s",
    )
    p.add_argument(
        "--buffer",
        type=int,
        default=None,
        help="Rolling buffer length (default: max(window, 256))",
    )
    p.add_argument(
        "--out",
        default="data/emg_two_movements.npz",
        help="Output .npz path (relative to cwd unless absolute)",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="If --out already exists, load its trials and add new ones (same window length)",
    )
    p.add_argument(
        "--names",
        nargs=2,
        default=["movement_1", "movement_2"],
        metavar=("NAME1", "NAME2"),
        help="Human-readable class names stored in npz",
    )
    return p.parse_args()


def normalize_segment(
    samples: list[int], window: int, align: str
) -> np.ndarray:
    """Crop or pad to fixed `window` length for sklearn pipeline."""
    x = np.asarray(samples, dtype=np.float32)
    n = x.size
    if n == window:
        return x
    if n > window:
        if align == "end":
            return x[-window:]
        if align == "start":
            return x[:window]
        # center
        start = (n - window) // 2
        return x[start : start + window]
    pad = window - n
    if align == "end":
        return np.pad(x, (pad, 0), mode="edge")
    if align == "start":
        return np.pad(x, (0, pad), mode="edge")
    # center pad: equal sides (prefer left one extra if odd)
    left = pad // 2
    right = pad - left
    return np.pad(x, (left, right), mode="edge")


def main():
    args = parse_args()
    if args.seconds is not None:
        if args.hz is None or args.hz <= 0:
            print("Error: --seconds requires a positive --hz (approximate samples per second).")
            sys.exit(1)
        window = max(16, int(round(args.seconds * args.hz)))
        print(
            f"Window = {window} samples (~{args.seconds:g} s × ~{args.hz:g} Hz from Arduino)"
        )
    else:
        window = max(16, args.window)
    trials_x: list[np.ndarray] = []
    trials_y: list[int] = []

    out_abs = os.path.abspath(args.out)
    if args.append:
        if os.path.isfile(out_abs):
            try:
                b = np.load(out_abs, allow_pickle=True)
            except OSError as e:
                print(f"Could not load {args.out} for --append: {e}")
                sys.exit(1)
            X0 = b["X"]
            if X0.ndim != 2:
                print("Append failed: existing X must be 2D (n_trials, n_samples).")
                sys.exit(1)
            w_file = int(b["window"]) if "window" in b else int(X0.shape[1])
            if w_file != window:
                print(
                    f"Using window={w_file} from existing file (not --window {args.window})."
                )
                window = w_file
            y0 = np.asarray(b["y"], dtype=np.int64)
            for i in range(X0.shape[0]):
                trials_x.append(np.asarray(X0[i], dtype=np.float32))
                trials_y.append(int(y0[i]))
            cn = b.get("class_names")
            if cn is not None:
                ex = [str(x) for x in cn.tolist()]
                if ex != list(args.names):
                    print(
                        "Note: class_names in file differ from --names "
                        f"(file {ex}, args {list(args.names)}); class ids 0/1 unchanged."
                    )
            print(f"--append: loaded {len(trials_x)} trials from {args.out}")
        else:
            print(f"--append: {args.out} not found, starting with an empty dataset.")

    max_segment = args.max_segment if args.max_segment is not None else 15 * window
    max_segment = max(max_segment, window)
    buf_len = args.buffer if args.buffer is not None else max(window, 256)

    ser, port = open_emg_serial(args.port, baud=args.baud)
    if not ser:
        print("Serial device not found. Use --port or plug in the board.")
        sys.exit(1)
    print(f"Using serial port: {port}")
    selected: int | None = None
    recording = False
    recording_class: int | None = None
    segment_samples: list[int] = []

    data = deque([0] * buf_len, maxlen=buf_len)

    last_key_display = "—"
    action_log: deque[str] = deque(maxlen=STATUS_LOG_MAX)

    fig = plt.figure(figsize=(13, 5.4))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        KEYS_BAR,
        fontsize=8.5,
        color="#c8c8e0",
        family="monospace",
        y=0.998,
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.05], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_panel = fig.add_subplot(gs[0, 1])

    line, = ax.plot(list(data), color="cyan", linewidth=1.2)
    rec_label = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        color="#ff6b6b",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, 1023)
    ax.set_xlim(0, buf_len)
    ax.set_ylabel("Raw ADC (0–1023)")
    ax.set_xlabel("Samples (most recent at right)")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    for l in (ax.title, ax.xaxis.label, ax.yaxis.label):
        l.set_color("white")

    ax_panel.set_facecolor("#12121f")
    ax_panel.set_xticks([])
    ax_panel.set_yticks([])
    ax_panel.set_title(
        "Status & keys",
        color="#e8e8ef",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    for s in ax_panel.spines.values():
        s.set_color("#3d3d52")
    panel_text = ax_panel.text(
        0.04,
        0.97,
        "",
        transform=ax_panel.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        family="monospace",
        color="#c8c8d8",
        linespacing=1.35,
    )

    def counts_str():
        n0 = sum(1 for y in trials_y if y == 0)
        n1 = sum(1 for y in trials_y if y == 1)
        return f"n={len(trials_y)}  [{args.names[0]}: {n0}]  [{args.names[1]}: {n1}]"

    def indent_block(text: str, pad: str = "    ") -> str:
        return "\n".join(pad + ln if ln else "" for ln in text.splitlines())

    def keys_reference_block() -> str:
        return (
            "▸ Keys to press\n"
            f"    1 / 2      Arm: {args.names[0]} / {args.names[1]}\n"
            "    s          Start segment (then move)\n"
            "    e          End segment → save this trial\n"
            f"    w          Quick snap (last {window} samples)\n"
            "    Esc / x    Cancel segment\n"
            "    d          Save all trials → .npz\n"
            "    l          Trial counts (terminal)\n"
            "    q          Quit\n"
        )

    def next_step_message() -> str:
        if recording:
            return (
                "Press  e  when you FINISH the movement.\n"
                "Press  Esc  or  x  to cancel this segment."
            )
        if selected is None:
            return (
                "Use the KEYBOARD (click this window first if\n"
                "keys do nothing).\n"
                "\n"
                "Press  1  → first movement  (class 0)\n"
                "Press  2  → second movement (class 1)\n"
                "\n"
                "The EMG numbers from Arduino are read\n"
                "automatically — you do not type them."
            )
        return (
            "Press  s  to START recording, then do the\n"
            "movement, then  e  to STOP.\n"
            "\n"
            "Or press  w  for quick snap (last "
            f"{window} samples).\n"
            "d  save file · l  counts in terminal · q  quit"
        )

    def refresh_panel():
        armed = (
            "None — press 1 or 2"
            if selected is None
            else f"Key {'1' if selected == 0 else '2'} → "
            f"{args.names[selected]}  (class {selected})"
        )
        rec_line = (
            "▸ Segment status\n"
            "    Recording ON — red counter on chart shows\n"
            "    how many samples so far.\n\n"
            if recording
            else ""
        )
        log_lines = list(action_log)
        log_block = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(log_lines))
        if not log_block:
            log_block = "  (no actions yet)"

        body = (
            f"{keys_reference_block()}"
            "\n"
            f"▸ Last key you pressed\n"
            f"    {last_key_display}\n"
            "\n"
            f"▸ Armed movement (from 1 or 2)\n"
            f"    {armed}\n"
            "\n"
            f"{rec_line}"
            f"▸ What to do NEXT\n"
            f"{indent_block(next_step_message())}\n"
            "\n"
            f"▸ Recent actions (newest at bottom)\n"
            f"{log_block}\n"
        )
        panel_text.set_text(body)
        ax.set_title(f"{counts_str()}   |   live EMG")
        fig.canvas.draw_idle()

    def log_action(msg: str):
        action_log.append(msg)

    def set_last_key(key_repr: str, action_msg: str | None = None):
        nonlocal last_key_display
        last_key_display = key_repr
        if action_msg:
            log_action(action_msg)
        refresh_panel()

    def save_npz() -> bool:
        if not trials_x:
            print("No trials to save.")
            return False
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        X = np.stack(trials_x, axis=0)
        y = np.asarray(trials_y, dtype=np.int64)
        np.savez(
            args.out,
            X=X,
            y=y,
            window=np.int32(window),
            class_names=np.array(args.names, dtype=object),
        )
        print(f"Saved {X.shape[0]} trials, X shape {X.shape} -> {args.out}")
        return True

    def on_key(event):
        nonlocal selected, recording, recording_class, segment_samples
        k = event.key
        if k is None:
            return
        if k in ("1", "2"):
            selected = int(k) - 1
            set_last_key(
                f"Key  {k}",
                f"Armed {args.names[selected]} (class {selected}) — next: s or w",
            )
        elif k == "s":
            if selected is None:
                print("Press 1 or 2 first, then s to start the segment.")
                set_last_key("Key  s", "Ignored: press 1 or 2 first to pick a movement")
                return
            recording = True
            recording_class = selected
            segment_samples = []
            set_last_key(
                "Key  s",
                f"Segment started for {args.names[recording_class]} — do move, then e",
            )
            print("Recording… press e when the movement is finished (escape to cancel).")
        elif k == "e":
            if not recording:
                print("Press s first to start a segment.")
                set_last_key("Key  e", "Ignored: not recording — press s first")
                return
            recording = False
            if not segment_samples:
                print("No samples in segment; try again.")
                recording_class = None
                set_last_key("Key  e", "Empty segment (no samples) — try s again")
                return
            raw_n = len(segment_samples)
            snap = normalize_segment(segment_samples, window, args.align)
            trials_x.append(snap)
            trials_y.append(recording_class)
            cls = recording_class
            cname = args.names[cls] if cls is not None else "?"
            recording_class = None
            segment_samples = []
            print(f"Saved trial: class {cls}, raw_len={raw_n}, stored_len={window}")
            set_last_key(
                "Key  e",
                f"Saved trial: {cname}, raw {raw_n} pts → stored {window} pts",
            )
        elif k in ("escape", "x"):
            if recording:
                recording = False
                recording_class = None
                segment_samples = []
                set_last_key("Esc" if k == "escape" else "Key  x", "Segment cancelled (not saved)")
                print("Recording cancelled.")
        elif k == "w":
            if selected is None:
                print("Press 1 or 2 first to choose the movement class.")
                set_last_key("Key  w", "Ignored: press 1 or 2 first")
                return
            snap = np.array(list(data)[-window:], dtype=np.float32)
            if snap.size < window:
                snap = np.pad(snap, (window - snap.size, 0), mode="constant")
            trials_x.append(snap)
            trials_y.append(selected)
            set_last_key(
                "Key  w",
                f"Quick snap saved as {args.names[selected]} ({window} samples)",
            )
        elif k == "d":
            if save_npz():
                set_last_key(
                    "Key  d",
                    f"Wrote {len(trials_x)} trials → {args.out}",
                )
            else:
                set_last_key("Key  d", "No trials yet — record with s/e or w")
        elif k == "l":
            print(counts_str())
            set_last_key("Key  l", counts_str())
        elif k == "q":
            if trials_x and save_npz():
                set_last_key(
                    "Key  q",
                    f"Saved {len(trials_x)} trials to {args.out}, then quit",
                )
            else:
                set_last_key("Key  q", "Quit (no trials to save)")
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame):
        nonlocal recording, recording_class, segment_samples
        try:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if raw.isdigit():
                v = int(raw)
                data.append(v)
                line.set_ydata(list(data))
                if recording:
                    if len(segment_samples) >= max_segment:
                        recording = False
                        recording_class = None
                        print(
                            f"Segment hit --max-segment ({max_segment}); stopped without saving. Press s again."
                        )
                        segment_samples = []
                        set_last_key(
                            "(auto: max length)",
                            f"Segment exceeded max length ({max_segment}); not saved",
                        )
                    else:
                        segment_samples.append(v)
        except (OSError, ValueError):
            pass
        if recording:
            rec_label.set_text(f"● REC  {len(segment_samples)}")
        else:
            rec_label.set_text("")
        return (line, rec_label)

    ani = animation.FuncAnimation(
        fig, update, interval=10, blit=False, cache_frame_data=False
    )
    if trials_x:
        log_action(
            f"Ready — {len(trials_x)} trials already in memory; add more, then d saves all"
        )
    else:
        log_action("Ready — press 1 or 2 to choose a movement class")
    refresh_panel()
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.07)
    try:
        plt.show()
    finally:
        ser.close()


if __name__ == "__main__":
    main()
