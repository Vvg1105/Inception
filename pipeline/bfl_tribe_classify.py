"""
Text prompt → BFL FLUX image → still looped MP4 → TRIBE (video) → element classifier.

Loads ``BFL_API_KEY`` from the environment (optionally from repo ``.env``). Matches the
photo pipeline: video TRIBE path with optional Whisper skip and video-only extractors.

  python -m pipeline.bfl_tribe_classify
  python -m pipeline.bfl_tribe_classify "a row of Victorian houses at sunset"

Requires ``ffmpeg`` on PATH and a trained ``train_element_classifier`` joblib.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

from pipeline.bfl_api import BFLAPIError, bfl_generate_image_bytes
from pipeline.photo_neural_matrix import _check_ffmpeg, image_to_looped_mp4
from tribe.model import load_model, predict_from_video_pooled

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_classifier_artifact(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Classifier artifact not found: {path}")
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "pipeline" not in obj or "element_classes" not in obj:
        raise ValueError(
            f"{path} is not a train_element_classifier joblib (missing pipeline / element_classes)"
        )
    return obj


def _load_dotenv(path: Path) -> None:
    """Set missing env vars from KEY=VALUE lines (no extra dependency)."""
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def _default_classifier_candidates() -> list[Path]:
    """Prefer repo ``outputs/``; on RunPod artifacts often live under ``/workspace/outputs/``."""
    name = "photo_element_logreg.joblib"
    return [
        PROJECT_ROOT / "outputs" / name,
        Path("/workspace/outputs") / name,
    ]


def _resolve_classifier_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    for cand in _default_classifier_candidates():
        if cand.is_file():
            logger.info("Using classifier %s", cand)
            return cand.resolve()
    return _default_classifier_candidates()[0].resolve()


def _suffix_for_mime(mime: str) -> str:
    m = (mime or "").lower()
    if "png" in m:
        return ".png"
    if "webp" in m:
        return ".webp"
    return ".jpg"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Image prompt (omit to type when asked)",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Classifier joblib (default: ./outputs/ or /workspace/outputs/ photo_element_logreg.joblib)",
    )
    p.add_argument(
        "--cache-folder",
        default=None,
        help="TRIBE cache folder (default: ./cache)",
    )
    p.add_argument(
        "--bfl-model",
        default="flux-2-klein-4b",
        help="BFL endpoint name, e.g. flux-2-klein-4b, flux-2-pro-preview, flux-dev",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output width (multiple of 16/32 per model docs)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output height",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Looped MP4 duration for TRIBE (seconds)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=24,
        help="FPS for looped MP4",
    )
    p.add_argument(
        "--save-image",
        type=Path,
        default=None,
        help="If set, write the generated image to this path",
    )
    p.add_argument(
        "--verbose-tribe",
        action="store_true",
        help="TRIBE tqdm",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON object with label, confidence, probabilities, prompt",
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="Load env vars from this file if keys are missing",
    )
    args = p.parse_args(argv)
    model_path = _resolve_classifier_path(args.model)

    _load_dotenv(args.env_file.resolve())

    os.environ.setdefault("TRIBE_VIDEO_SKIP_WHISPER", "1")
    os.environ.setdefault("TRIBE_FEATURES_VIDEO_ONLY", "1")

    api_key = (os.environ.get("BFL_API_KEY") or "").strip()
    if not api_key:
        logger.error(
            "Missing BFL_API_KEY. Set it in the environment or in %s",
            args.env_file,
        )
        return 1

    prompt = (args.prompt or "").strip()
    if not prompt:
        try:
            prompt = input("Describe the image to generate: ").strip()
        except EOFError:
            prompt = ""
    if not prompt:
        logger.error("Empty prompt.")
        return 1

    try:
        art = _load_classifier_artifact(model_path)
    except FileNotFoundError as e:
        logger.error("%s", e)
        if args.model is None:
            logger.error(
                "Tried: %s",
                ", ".join(str(p) for p in _default_classifier_candidates()),
            )
        return 1
    except ValueError as e:
        logger.error("%s", e)
        return 1

    pipe = art["pipeline"]
    element_classes = list(art["element_classes"])
    expected_nv = art.get("n_vertices")

    try:
        logger.info("BFL model %s — generating image…", args.bfl_model)
        img_bytes, mime = bfl_generate_image_bytes(
            api_key=api_key,
            prompt=prompt,
            model=args.bfl_model,
            width=args.width,
            height=args.height,
        )
    except BFLAPIError as e:
        logger.error("%s", e)
        return 1
    except OSError as e:
        logger.error("Network error: %s", e)
        return 1

    if args.save_image is not None:
        args.save_image.parent.mkdir(parents=True, exist_ok=True)
        args.save_image.write_bytes(img_bytes)
        logger.info("Saved image %s", args.save_image)

    ffmpeg = _check_ffmpeg()
    tmp_dir = Path(tempfile.mkdtemp(prefix="bfl_tribe_"))
    try:
        img_path = tmp_dir / f"generated{_suffix_for_mime(mime)}"
        img_path.write_bytes(img_bytes)
        mp4_path = tmp_dir / "generated.mp4"
        image_to_looped_mp4(
            image_path=img_path,
            out_mp4=mp4_path,
            duration_sec=args.duration,
            fps=args.fps,
            ffmpeg_exe=ffmpeg,
        )

        logger.info("Loading TRIBE…")
        tribe = load_model(cache_folder=args.cache_folder)
        pooled, _, _ = predict_from_video_pooled(
            tribe,
            str(mp4_path),
            verbose=args.verbose_tribe,
        )
        x = np.asarray(pooled, dtype=np.float64).reshape(1, -1)
        if expected_nv is not None and x.shape[1] != int(expected_nv):
            logger.error(
                "Feature dim %s != classifier n_vertices %s",
                x.shape[1],
                expected_nv,
            )
            return 1

        proba = pipe.predict_proba(x)[0]
        if len(proba) != len(element_classes):
            logger.error("Classifier / class list length mismatch")
            return 1
        idx = int(np.argmax(proba))
        label = element_classes[idx]
        confidence = float(proba[idx])
        scores = {element_classes[i]: float(proba[i]) for i in range(len(element_classes))}

        if args.json:
            print(
                json.dumps(
                    {
                        "prompt": prompt,
                        "label": label,
                        "confidence": confidence,
                        "probabilities": scores,
                    },
                    indent=2,
                )
            )
        else:
            print(f"Predicted class: {label} (confidence {confidence:.3f})")
            for name, sc in sorted(scores.items(), key=lambda t: -t[1]):
                print(f"  {name}: {sc:.3f}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
