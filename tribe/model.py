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


def _cuda_really_works() -> bool:
    """True only if this PyTorch build can allocate on CUDA (not just ``is_available()``)."""
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.zeros(1, device="cuda")
        del x
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
    except Exception:
        return False


def _get_device() -> str:
    import torch

    if force_cpu_requested():
        return "cpu"
    if _cuda_really_works():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_cuda_device(d: object) -> bool:
    import torch

    if isinstance(d, str):
        return d.lower() == "cuda"
    if isinstance(d, torch.device):
        return d.type == "cuda"
    return False


def _force_extractor_tree_cpu(ext: object) -> None:
    """Move extractors that were configured for CUDA onto CPU (nested ``video.image``, etc.)."""
    if ext is None:
        return
    if _is_cuda_device(getattr(ext, "device", None)):
        ext.device = "cpu"
    sub = getattr(ext, "image", None)
    if sub is not None:
        if _is_cuda_device(getattr(sub, "device", None)):
            sub.device = "cpu"
        m = getattr(sub, "_model", None)
        if m is not None:
            try:
                m.cpu()
            except Exception:
                pass


def load_model(cache_folder: Optional[str] = None):
    """Load pretrained TRIBE v2; weights download on first run (~2GB) into cache_folder.

    Uses **CUDA** when a real GPU allocation succeeds and ``TRIBE_FORCE_CPU`` is unset
    (e.g. RunPod). Set ``TRIBE_FORCE_CPU=1`` for Mac / CPU-only PyTorch.
    """
    cuda_ok = _cuda_really_works()
    use_cpu = force_cpu_requested() or not cuda_ok
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from tribe.whisper_patch import apply_whisper_compute_patch
    from tribev2 import TribeModel

    apply_whisper_compute_patch()

    cache = cache_folder or str(DEFAULT_CACHE)
    brain_device = "cpu" if use_cpu else _get_device()
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=cache,
        device=brain_device,
    )

    if brain_device != "cuda":
        if getattr(model, "_model", None) is not None:
            model._model.cpu()
        for attr in ("text_feature", "audio_feature", "video_feature", "image_feature"):
            _force_extractor_tree_cpu(getattr(model.data, attr, None))
    else:
        import torch

        if os.environ.get("TRIBE_CUDNN_BENCHMARK", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        ):
            torch.backends.cudnn.benchmark = True

    nw_env = os.environ.get("TRIBE_DATALOADER_WORKERS", "").strip()
    if nw_env.isdigit():
        num_workers = max(0, int(nw_env))
    else:
        num_workers = 4 if brain_device == "cuda" else 2
    model.data.num_workers = min(num_workers, 8)

    return model


def predict_from_video(model, video_path: str) -> tuple[np.ndarray, pd.DataFrame]:
    df = model.get_events_dataframe(video_path=video_path)
    preds, segments = model.predict(events=df)
    return preds, segments


def predict_from_video_pooled(
    model, video_path: str, *, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, list]:
    """Run TRIBE on a video file; return mean-pooled ``(n_vertices,)``, raw preds, segments."""
    df = model.get_events_dataframe(video_path=video_path)
    preds, segments = model.predict(events=df, verbose=verbose)
    preds = np.asarray(preds, dtype=np.float32)
    if preds.size == 0:
        raise RuntimeError(
            "TRIBE returned no segments for this video (empty preds). "
            "Try a longer clip or check the file."
        )
    pooled = preds.mean(axis=0).astype(np.float32, copy=False)
    return pooled, preds, segments


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
