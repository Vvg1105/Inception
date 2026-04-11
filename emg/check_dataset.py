#!/usr/bin/env python3
"""
Check whether you collected enough EMG data and how separable the two classes are.

Runs stratified k-fold cross-validation on the same features + model family as
train_classifier.py (so scores match what you can expect offline).

Usage:
  python check_dataset.py --data data/emg_two_movements.npz
"""

from __future__ import annotations

import argparse
import os
import sys

_EMG_DIR = os.path.dirname(os.path.abspath(__file__))
if _EMG_DIR not in sys.path:
    sys.path.insert(0, _EMG_DIR)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import featurize_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Check EMG dataset size and separability")
    p.add_argument("--data", default="data/emg_two_movements.npz")
    p.add_argument("--folds", type=int, default=5, help="CV folds (min class must be >= folds)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        bundle = np.load(args.data, allow_pickle=True)
    except OSError as e:
        print(f"Could not load {args.data}: {e}")
        sys.exit(1)

    X_raw = bundle["X"]
    y = bundle["y"].astype(np.int64)
    names = bundle.get("class_names")
    if X_raw.ndim != 2:
        print("Expected X shape (n_trials, n_samples)")
        sys.exit(1)
    classes = np.unique(y)
    if len(classes) < 2:
        print("Need both classes in y.")
        sys.exit(1)

    labels = (
        [str(n) for n in names.tolist()]
        if names is not None
        else [str(int(c)) for c in classes]
    )
    label_for = {int(c): labels[i] for i, c in enumerate(sorted(classes))}

    print(f"File: {args.data}")
    print(f"Trials: {X_raw.shape[0]}, samples per trial: {X_raw.shape[1]}")
    if "window" in bundle:
        print(f"Stored window (from collection): {int(bundle['window'])}")
    print()

    print("Counts per class:")
    min_n = X_raw.shape[0]
    for c in sorted(classes):
        n = int(np.sum(y == c))
        min_n = min(min_n, n)
        print(f"  {label_for[int(c)]} (class {int(c)}): {n}")
    print()

    # Heuristic guidance
    if min_n < 10:
        print(
            "[!] Very few trials in the smallest class. Aim for at least 20–40 per "
            "movement for steadier results.\n"
        )
    elif min_n < 25:
        print(
            "[*] Usable, but more trials per class (e.g. 30+) usually improves live "
            "performance.\n"
        )
    else:
        print("[OK] Reasonable trial counts for a simple two-gesture problem.\n")

    folds = min(args.folds, min_n)
    if folds < 2:
        print("Not enough trials for cross-validation (need >= 2 in smallest class).")
        sys.exit(0)

    F = featurize_dataset(X_raw)
    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    random_state=args.seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    scores = cross_val_score(clf, F, y, cv=cv, scoring="accuracy")
    print(f"Stratified {folds}-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  (per-fold: {', '.join(f'{s:.3f}' for s in scores)})")
    print()

    y_hat = cross_val_predict(clf, F, y, cv=cv)
    print("Pooled CV confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y, y_hat, labels=sorted(classes))
    header = "              " + "  ".join(f"{lab[:12]:>12}" for lab in labels)
    print(header)
    for i, row_c in enumerate(sorted(classes)):
        row_lab = label_for[int(row_c)][:12]
        print(
            f"  {row_lab:>12}"
            + "".join(f"{cm[i, j]:12d}" for j in range(len(classes)))
        )
    print()
    print("Pooled CV classification report:")
    print(
        classification_report(
            y, y_hat, target_names=labels, digits=3, zero_division=0
        )
    )

    if scores.mean() < 0.65:
        print(
            "\nCV accuracy is modest: collect more varied trials, check electrode "
            "placement, or increase window length to cover the full ~2 s gesture "
            "(see collect_dataset.py --window)."
        )
    elif scores.mean() < 0.85:
        print(
            "\nDecent offline separation. If live feels worse, use live_classify.py "
            "and try smoothing; timing must match training window length."
        )


if __name__ == "__main__":
    main()
