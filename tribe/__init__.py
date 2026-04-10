"""TRIBE v2 helpers for imagine. Install: ``pip install -r requirements-tribe.txt``."""

from tribe.model import (
    load_model,
    predict_from_audio,
    predict_from_events,
    predict_from_text,
    predict_from_text_string,
    predict_from_video,
)

__all__ = [
    "load_model",
    "predict_from_audio",
    "predict_from_events",
    "predict_from_text",
    "predict_from_text_string",
    "predict_from_video",
]
