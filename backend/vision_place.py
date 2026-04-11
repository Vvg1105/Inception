"""
BFL image → TRIBE video features → sklearn element classifier.

Used by FastAPI ``/api/vision-classify``. Expects repo root on ``sys.path`` (see ``app.py``).
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# Training class name → key used in index.html MODEL_MAP / PROCEDURAL_MAP
CLASSIFIER_TO_PLACE_KEY: dict[str, str] = {
    "bridge": "bridge",
    "lake": "lake",
    "skyscrapers": "skyscraper",
    "trees": "tree",
    "houses": "house",
}

_tribe_model = None
_classifier_artifact: dict | None = None


def _resolve_classifier_path() -> Path:
    name = "photo_element_logreg.joblib"
    for cand in (ROOT / "outputs" / name, Path("/workspace/outputs") / name):
        if cand.is_file():
            return cand
    return ROOT / "outputs" / name


def _load_classifier() -> dict:
    global _classifier_artifact
    if _classifier_artifact is not None:
        return _classifier_artifact
    path = _resolve_classifier_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Classifier not found at {path}. Train with pipeline.train_element_classifier "
            "or set outputs/photo_element_logreg.joblib"
        )
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "pipeline" not in obj or "element_classes" not in obj:
        raise ValueError("Invalid classifier joblib (need pipeline + element_classes)")
    _classifier_artifact = obj
    logger.info("Loaded classifier from %s", path)
    return _classifier_artifact


def _load_tribe(cache_folder: str | None):
    global _tribe_model
    if _tribe_model is not None:
        return _tribe_model
    os.environ.setdefault("TRIBE_VIDEO_SKIP_WHISPER", "1")
    os.environ.setdefault("TRIBE_FEATURES_VIDEO_ONLY", "1")
    from tribe.model import load_model

    cache = cache_folder or str(ROOT / "cache")
    _tribe_model = load_model(cache_folder=cache)
    logger.info("TRIBE model loaded (video-only path)")
    return _tribe_model


def run_vision_classify(
    *,
    prompt: str,
    api_key: str,
    bfl_model: str = "flux-2-klein-4b",
    width: int = 1024,
    height: int = 1024,
    duration_sec: float = 5.0,
    fps: int = 24,
    cache_folder: str | None = None,
) -> tuple[str, str, float, dict[str, float]]:
    """Return ``(classified_label, place_key, confidence, probabilities)``."""
    from pipeline.bfl_api import bfl_generate_image_bytes
    from pipeline.photo_neural_matrix import _check_ffmpeg, image_to_looped_mp4
    from tribe.model import predict_from_video_pooled

    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt is empty")

    img_bytes, mime = bfl_generate_image_bytes(
        api_key=api_key,
        prompt=prompt,
        model=bfl_model,
        width=width,
        height=height,
    )

    ext = ".jpg"
    if "png" in (mime or "").lower():
        ext = ".png"
    elif "webp" in (mime or "").lower():
        ext = ".webp"

    ffmpeg = _check_ffmpeg()
    tmp = Path(tempfile.mkdtemp(prefix="vision_place_"))
    try:
        img_path = tmp / f"gen{ext}"
        img_path.write_bytes(img_bytes)
        mp4_path = tmp / "clip.mp4"
        image_to_looped_mp4(
            image_path=img_path,
            out_mp4=mp4_path,
            duration_sec=duration_sec,
            fps=fps,
            ffmpeg_exe=ffmpeg,
        )

        tribe = _load_tribe(cache_folder)
        pooled, _, _ = predict_from_video_pooled(tribe, str(mp4_path), verbose=False)
        art = _load_classifier()
        pipe = art["pipeline"]
        names = [str(x).strip().lower() for x in art["element_classes"]]
        expected_nv = art.get("n_vertices")
        x = np.asarray(pooled, dtype=np.float64).reshape(1, -1)
        if expected_nv is not None and x.shape[1] != int(expected_nv):
            raise ValueError(
                f"TRIBE dim {x.shape[1]} != classifier n_vertices {expected_nv}"
            )

        proba = pipe.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        classified = names[idx]
        confidence = float(proba[idx])
        probs = {names[i]: float(proba[i]) for i in range(len(names))}
        place_key = CLASSIFIER_TO_PLACE_KEY.get(classified, classified)
        return classified, place_key, confidence, probs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
