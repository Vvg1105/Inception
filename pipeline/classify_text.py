"""
Run arbitrary text through TRIBE (pooled neural vector), then the trained element classifier.

  python -m pipeline.classify_text "quiet side street"
  python -m pipeline.classify_text --model outputs/element_logreg_partial.joblib

Omit the text argument to type a line at the prompt (stdin).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

from tribe.model import load_model, predict_from_text_string

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


def classify_one(
    *,
    text: str,
    tribe_model,
    pipeline,
    element_classes: list[str],
    expected_n_vertices: int | None,
    verbose_tribe: bool = False,
) -> tuple[str, float, dict[str, float]]:
    pooled, _, _ = predict_from_text_string(
        tribe_model, text, verbose=verbose_tribe
    )
    x = np.asarray(pooled, dtype=np.float64).reshape(1, -1)
    if expected_n_vertices is not None and x.shape[1] != expected_n_vertices:
        raise ValueError(
            f"Feature dim {x.shape[1]} != model training dim {expected_n_vertices} "
            "(TRIBE checkpoint or pooling changed?)"
        )
    proba = pipeline.predict_proba(x)[0]
    names = list(element_classes)
    if len(proba) != len(names):
        raise ValueError(
            f"predict_proba length {len(proba)} != element_classes length {len(names)}"
        )
    idx = int(np.argmax(proba))
    label = names[idx]
    confidence = float(proba[idx])
    scores = {names[i]: float(proba[i]) for i in range(len(names))}
    return label, confidence, scores


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Phrase to classify (omit to read one line from stdin)",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "element_logreg.joblib",
        help="Joblib from pipeline.train_element_classifier",
    )
    p.add_argument(
        "--cache-folder",
        default=None,
        help="TRIBE cache folder (default: ./cache)",
    )
    p.add_argument(
        "--verbose-tribe",
        action="store_true",
        help="TRIBE tqdm per sentence",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON object (label, confidence, probabilities)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=0,
        help="If >0, print this many classes by probability (text mode only)",
    )
    args = p.parse_args(argv)

    raw = args.text
    if raw is None:
        raw = sys.stdin.readline()
    text = (raw or "").strip()
    if not text:
        logger.error("Empty text; pass a phrase or type a line when prompted.")
        return 1

    try:
        art = _load_artifact(args.model)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1

    pipeline = art["pipeline"]
    names = list(art["element_classes"])
    n_vert = art.get("n_vertices")
    if n_vert is not None:
        n_vert = int(n_vert)

    logger.info("Loading TRIBE (first run may download weights)...")
    tribe = load_model(cache_folder=args.cache_folder)

    try:
        label, confidence, scores = classify_one(
            text=text,
            tribe_model=tribe,
            pipeline=pipeline,
            element_classes=names,
            expected_n_vertices=n_vert,
            verbose_tribe=args.verbose_tribe,
        )
    except Exception as e:
        logger.error("%s", e)
        return 1

    if args.json:
        out = {
            "text": text,
            "label": label,
            "confidence": confidence,
            "probabilities": scores,
        }
        print(json.dumps(out, indent=2))
        return 0

    print(f"label:       {label}")
    print(f"confidence:  {confidence:.4f}")
    if args.top and args.top > 0:
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[: args.top]
        print("top classes:")
        for name, pr in ranked:
            print(f"  {pr:.4f}  {name}")
    else:
        print("probabilities:")
        for name in sorted(scores.keys()):
            print(f"  {scores[name]:.4f}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
