"""Runtime flags via environment variables."""

import os


def force_cpu_requested() -> bool:
    """If true, TRIBE + WhisperX stay on CPU (no ``torch.cuda`` / GPU Whisper)."""
    v = os.environ.get("TRIBE_FORCE_CPU", "").strip().lower()
    return v in ("1", "true", "yes", "on")
