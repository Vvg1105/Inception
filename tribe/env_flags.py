"""Runtime flags via environment variables."""

import os


def force_cpu_requested() -> bool:
    """If true, TRIBE + WhisperX stay on CPU (no ``torch.cuda`` / GPU Whisper)."""
    v = os.environ.get("TRIBE_FORCE_CPU", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def video_skip_whisper_for_video_path() -> bool:
    """If true, video inputs skip ASR/Whisper (``get_audio_and_text_events(..., audio_only=True)``).

    Set ``TRIBE_VIDEO_SKIP_WHISPER=1`` before loading the model / calling video predict.
    ``pipeline.photo_neural_matrix`` defaults this to ``1`` unless ``--video-whisper`` is passed.
    """
    v = os.environ.get("TRIBE_VIDEO_SKIP_WHISPER", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def features_video_only_requested() -> bool:
    """If true, ``load_model`` passes ``config_update`` so ``data.features_to_use`` is ``[\"video\"]``.

    Text (Llama) and audio (Wav2Vec) extractors are not prepared for inference — large VRAM/time
    savings. Brain weights still load all modality projectors from the checkpoint; missing
    batch keys are zero-filled in ``FmriEncoderModel.aggregate_features``.

    ``pipeline.photo_neural_matrix`` defaults ``TRIBE_FEATURES_VIDEO_ONLY=1`` unless
    ``--tribe-all-modalities`` is passed.
    """
    v = os.environ.get("TRIBE_FEATURES_VIDEO_ONLY", "0").strip().lower()
    return v in ("1", "true", "yes", "on")
