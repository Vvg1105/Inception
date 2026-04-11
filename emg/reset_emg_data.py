#!/usr/bin/env python3
"""Remove collected EMG datasets and trained models under emg/data and emg/models."""

from __future__ import annotations

import argparse
import os
import sys

_EMG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_EMG_DIR, "data")
MODELS_DIR = os.path.join(_EMG_DIR, "models")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Do not ask for confirmation",
    )
    args = p.parse_args()

    candidates: list[str] = []
    for root in (DATA_DIR, MODELS_DIR):
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if name.endswith((".npz", ".joblib", ".pkl")):
                candidates.append(os.path.join(root, name))

    if not candidates:
        print("Nothing to remove (no .npz / .joblib / .pkl in data/ or models/).")
        return

    print("Will delete:")
    for path in sorted(candidates):
        print(f"  {path}")
    if not args.yes:
        if input("Continue? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            sys.exit(1)

    for path in candidates:
        os.remove(path)
        print(f"Removed {path}")
    print("Done. Recollect with collect_dataset.py, then train_classifier.py.")


if __name__ == "__main__":
    main()
