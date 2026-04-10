"""
TRIBE v2 loading and prediction helpers for the imagine project.

Mirrors insilico's core.model behavior; weights cache defaults to ./cache here.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tribe.env_flags import force_cpu_requested

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = PROJECT_ROOT / "cache"


def _get_device() -> str:
    import torch

    if force_cpu_requested():
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(cache_folder: Optional[str] = None):
    """Load pretrained TRIBE v2; weights download on first run (~2GB) into cache_folder."""
    if force_cpu_requested():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from tribe.whisper_patch import apply_whisper_compute_patch
    from tribev2 import TribeModel

    apply_whisper_compute_patch()

    cache = cache_folder or str(DEFAULT_CACHE)
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache)

    device = _get_device()
    if device != "cuda":
        if getattr(model, "_model", None) is not None:
            model._model.cpu()

        for attr in ("text_feature", "audio_feature", "video_feature", "image_feature"):
            ext = getattr(model.data, attr, None)
            if ext is not None and getattr(ext, "device", None) == "cuda":
                ext.device = "cpu"

        model.data.num_workers = 2
    return model


def predict_from_video(model, video_path: str) -> tuple[np.ndarray, pd.DataFrame]:
    df = model.get_events_dataframe(video_path=video_path)
    preds, segments = model.predict(events=df)
    return preds, segments


def predict_from_audio(model, audio_path: str) -> tuple[np.ndarray, pd.DataFrame]:
    df = model.get_events_dataframe(audio_path=audio_path)
    preds, segments = model.predict(events=df)
    return preds, segments


def predict_from_text(model, text_path: str) -> tuple[np.ndarray, pd.DataFrame]:
    df = model.get_events_dataframe(text_path=text_path)
    preds, segments = model.predict(events=df)
    return preds, segments


def predict_from_text_string(
    model, text: str, *, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, list]:
    """TTS → words → TRIBE; returns mean-pooled ``(n_vertices,)``, raw preds, segments."""
    from tribe.whisper_patch import apply_whisper_compute_patch

    apply_whisper_compute_patch()

    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("text must be non-empty")

    path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            encoding="utf-8",
            delete=False,
        ) as f:
            f.write(stripped)
            path = f.name

        events = model.get_events_dataframe(text_path=path)
        preds, segments = model.predict(events=events, verbose=verbose)
    finally:
        if path is not None:
            Path(path).unlink(missing_ok=True)

    preds = np.asarray(preds, dtype=np.float32)
    if preds.size == 0:
        raise RuntimeError(
            "TRIBE returned no segments for this text (empty preds). "
            "Try a longer sentence or check TTS/transcription logs."
        )
    pooled = preds.mean(axis=0).astype(np.float32, copy=False)
    return pooled, preds, segments


def predict_from_events(model, events: pd.DataFrame) -> tuple[np.ndarray, list]:
    preds, segments = model.predict(events=events)
    return preds, segments
