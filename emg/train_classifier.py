#!/usr/bin/env python3
"""
Train a two-class classifier from collect_dataset.py output (.npz).

Loads X, y, applies time-domain features, fits RandomForest, saves joblib model.
"""

from __future__ import annotations

import argparse
import os
import sys

_EMG_DIR = os.path.dirname(os.path.abspath(__file__))
if _EMG_DIR not in sys.path:
    sys.path.insert(0, _EMG_DIR)

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import featurize_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Train EMG two-movement classifier")
    p.add_argument(
        "--data",
        default="data/emg_two_movements.npz",
        help="Path to .npz from collect_dataset.py",
    )
    p.add_argument(
        "--out",
        default="models/emg_two_movements.joblib",
        help="Output joblib path for fitted pipeline",
    )
    p.add_argument("--test-size", type=float, default=0.25, help="Holdout fraction")
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
    y = bundle["y"]
    names = bundle.get("class_names")
    if X_raw.ndim != 2:
        print("Expected X with shape (n_trials, n_samples)")
        sys.exit(1)
    if len(np.unique(y)) < 2:
        print("Need at least two classes in y to train a classifier.")
        sys.exit(1)

    F = featurize_dataset(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        F, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

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
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    target_names = (
        [str(n) for n in names.tolist()]
        if names is not None
        else [str(i) for i in sorted(np.unique(y))]
    )
    print(classification_report(y_test, pred, target_names=target_names))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    joblib.dump(
        {
            "pipeline": clf,
            "class_names": names.tolist() if names is not None else None,
            "window": int(bundle["window"]) if "window" in bundle else X_raw.shape[1],
        },
        args.out,
    )
    print(f"Saved model bundle -> {args.out}")


if __name__ == "__main__":
    main()
