"""
Evaluate a classifier from ``train_element_classifier`` on a neural feature ``.npz``.

Use this for **holdout** data (never used during training). Do not call
``train_element_classifier --data`` on the holdout file.

  python -m pipeline.eval_element_classifier \\
      --model outputs/photo_element_logreg.joblib \\
      --data outputs/photo_tribe_neural_holdout.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report

from pipeline.neural_matrix import load_npz_bundle, normalize_class_label

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_artifact(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Classifier artifact not found: {path}")
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "pipeline" not in obj or "element_classes" not in obj:
        raise ValueError(
            f"{path} is not a train_element_classifier joblib (missing pipeline / element_classes)"
        )
    return obj


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "element_logreg.joblib",
        help="Joblib from pipeline.train_element_classifier",
    )
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Neural matrix .npz (e.g. photo_tribe_neural_holdout.npz)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print classification_report as JSON (precision/recall/f1 per label)",
    )
    args = p.parse_args(argv)

    if not args.data.is_file():
        logger.error("Data file not found: %s", args.data)
        return 1

    try:
        art = _load_artifact(args.model)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1

    pipe = art["pipeline"]
    class_names = [normalize_class_label(str(x)) for x in art["element_classes"]]
    expected_nv = art.get("n_vertices")

    bundle = load_npz_bundle(args.data)
    X = np.asarray(bundle["X"], dtype=np.float64)
    if expected_nv is not None and X.shape[1] != int(expected_nv):
        logger.error(
            "Feature dim %s != model n_vertices %s (checkpoint / pooling mismatch?)",
            X.shape[1],
            expected_nv,
        )
        return 1

    y_true = [
        normalize_class_label(str(x)) for x in np.asarray(bundle["labels_combined"])
    ]
    unknown = sorted(set(y_true) - set(class_names))
    if unknown:
        logger.error(
            "Labels in data not seen at training (element_classes): %s",
            unknown,
        )
        return 1

    y_pred_idx = pipe.predict(X)
    y_pred = [class_names[int(i)] for i in y_pred_idx]

    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    logger.info("Accuracy on %s: %.4f (%d rows)", args.data.name, acc, len(y_true))

    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        zero_division=0,
    )
    if args.json:
        rep_dict = classification_report(
            y_true,
            y_pred,
            labels=class_names,
            output_dict=True,
            zero_division=0,
        )
        print(json.dumps(rep_dict, indent=2))
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
