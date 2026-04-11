"""
Training script for fMRI decoders using leave-one-out cross-validation.

Usage:
    python train.py --data <path_to_data.npz>

Expected data format (.npz):
    X              : (n_trials, ~20000) fMRI encodings from TRIBE v2
    y_object       : (n_trials,) integer labels 0-7 (see OBJECT_CLASSES)
    y_size         : (n_trials,) binary labels 0=small, 1=large

Outputs per-fold predictions and prints final LOO accuracy for each decoder.
"""

import argparse
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model import build_object_decoder, build_size_decoder, OBJECT_CLASSES, N_COMPONENTS


def load_data(path: str):
    data = np.load(path)
    X = data["X"]               # (n_trials, ~20000)
    y_object = data["y_object"] # (n_trials,)  values 0-7
    y_size = data["y_size"]     # (n_trials,)  values 0 or 1
    print(f"Loaded {X.shape[0]} trials, {X.shape[1]} features")
    return X, y_object, y_size


def loo_evaluate(decoder_name: str, pipeline, X: np.ndarray, y: np.ndarray):
    """
    Run leave-one-out cross-validation and return per-fold predictions.

    Args:
        decoder_name: Label for logging.
        pipeline: Unfitted sklearn Pipeline.
        X: Feature matrix (n_trials, n_features).
        y: Label vector (n_trials,).

    Returns:
        y_true, y_pred arrays aligned trial-by-trial.
    """
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    n_trials = len(y)

    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # Clone-free: re-instantiate from the factory each fold to avoid state leakage.
        # (sklearn pipelines are stateful after fit)
        from sklearn.base import clone
        fold_pipeline = clone(pipeline)

        fold_pipeline.fit(X_train, y_train)
        pred = fold_pipeline.predict(X_test)

        y_true.append(y[test_idx[0]])
        y_pred.append(pred[0])

        if (fold + 1) % max(1, n_trials // 10) == 0 or fold == n_trials - 1:
            print(f"  [{decoder_name}] fold {fold + 1}/{n_trials}")

    return np.array(y_true), np.array(y_pred)


def report_object_decoder(y_true: np.ndarray, y_pred: np.ndarray):
    acc = accuracy_score(y_true, y_pred)
    chance = 1.0 / len(OBJECT_CLASSES)
    print(f"\n=== Object Decoder (8-way) ===")
    print(f"LOO Accuracy : {acc:.4f}  (chance = {chance:.4f})")
    print(f"\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=OBJECT_CLASSES, zero_division=0
        )
    )
    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred)
    header = "         " + "  ".join(f"{c[:4]:>4}" for c in OBJECT_CLASSES)
    print(header)
    for i, row in enumerate(cm):
        label = f"{OBJECT_CLASSES[i][:8]:<8}"
        print(label + "  " + "  ".join(f"{v:>4}" for v in row))


def report_size_decoder(y_true: np.ndarray, y_pred: np.ndarray):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== Size Decoder (binary) ===")
    print(f"LOO Accuracy : {acc:.4f}  (chance = 0.5000)")
    print(f"\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=["small", "large"], zero_division=0
        )
    )
    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred)
    print("          small  large")
    print(f"small     {cm[0,0]:>5}  {cm[0,1]:>5}")
    print(f"large     {cm[1,0]:>5}  {cm[1,1]:>5}")


def main():
    parser = argparse.ArgumentParser(description="Train and LOO-evaluate fMRI decoders")
    parser.add_argument(
        "--data", required=True, help="Path to .npz file with X, y_object, y_size"
    )
    args = parser.parse_args()

    X, y_object, y_size = load_data(args.data)

    print(f"\nRunning LOO for Object Decoder (n_components={N_COMPONENTS})...")
    obj_pipeline = build_object_decoder()
    y_true_obj, y_pred_obj = loo_evaluate("Object", obj_pipeline, X, y_object)
    report_object_decoder(y_true_obj, y_pred_obj)

    print(f"\nRunning LOO for Size Decoder (n_components={N_COMPONENTS})...")
    size_pipeline = build_size_decoder()
    y_true_sz, y_pred_sz = loo_evaluate("Size", size_pipeline, X, y_size)
    report_size_decoder(y_true_sz, y_pred_sz)


if __name__ == "__main__":
    main()
