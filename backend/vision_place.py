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


def generate_bfl_image(
    *,
    prompt: str,
    api_key: str,
    bfl_model: str = "flux-2-klein-4b",
    width: int = 1024,
    height: int = 1024,
) -> tuple[bytes, str]:
    """Step 1: generate the BFL image. Returns ``(image_bytes, mime_type)``."""
    from pipeline.bfl_api import bfl_generate_image_bytes

    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt is empty")
    return bfl_generate_image_bytes(
        api_key=api_key, prompt=prompt, model=bfl_model, width=width, height=height,
    )


def _fast_image_to_mp4(
    image_path: Path, out_mp4: Path, ffmpeg_exe: str,
    duration_sec: float = 2.0, fps: int = 8, res: int = 512,
) -> None:
    """Ultrafast MP4 for real-time classification (not training)."""
    import subprocess

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={res}:{res}:force_original_aspect_ratio=decrease,pad={res}:{res}:(ow-iw)/2:(oh-ih)/2,fps={fps},format=yuv420p"
    cmd = [
        ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-i", str(image_path),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", str(duration_sec),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-vf", vf,
        "-c:a", "aac", "-shortest",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def classify_from_image_bytes(
    *,
    img_bytes: bytes,
    mime: str,
    duration_sec: float = 3.0,
    fps: int = 16,
    fast: bool = True,
    cache_folder: str | None = None,
) -> tuple[str, str, float, dict[str, float], np.ndarray]:
    """Step 2: image bytes → looped MP4 → TRIBE → classifier.

    Returns ``(classified_label, place_key, confidence, probabilities, tribe_pooled)``.
    When ``fast=True`` uses ultrafast ffmpeg preset with minimal frames (for real-time use).
    """
    from pipeline.photo_neural_matrix import _check_ffmpeg, image_to_looped_mp4
    from tribe.model import predict_from_video_pooled

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
        if fast:
            _fast_image_to_mp4(img_path, mp4_path, ffmpeg, duration_sec, fps)
        else:
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
        pooled_out = np.asarray(pooled, dtype=np.float32).reshape(-1).copy()
        return classified, place_key, confidence, probs, pooled_out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def preload_models(cache_folder: str | None = None) -> None:
    """Pre-warm TRIBE + classifier so first request doesn't pay load time."""
    _load_tribe(cache_folder)
    _load_classifier()


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
) -> tuple[str, str, float, dict[str, float], np.ndarray]:
    """Full pipeline (legacy single-call path)."""
    img_bytes, mime = generate_bfl_image(
        prompt=prompt, api_key=api_key, bfl_model=bfl_model, width=width, height=height,
    )
    return classify_from_image_bytes(
        img_bytes=img_bytes, mime=mime,
        duration_sec=duration_sec, fps=fps, cache_folder=cache_folder,
    )
