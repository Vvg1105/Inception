"""Patch tribev2 WhisperX invocation: float16 on CUDA, float32 on CPU (ctranslate2)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_applied = False


def _get_transcript_from_audio(wav_filename: Path, language: str) -> pd.DataFrame:
    from tribe.env_flags import force_cpu_requested

    language_codes = dict(
        english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
    )
    if language not in language_codes:
        raise ValueError(f"Language {language} not supported")

    env_dev = os.environ.get("TRIBE_WHISPER_DEVICE", "").strip().lower()
    if env_dev in ("cpu", "cuda"):
        device = env_dev
    elif force_cpu_requested():
        device = "cpu"
    elif sys.platform == "darwin":
        # ctranslate2 cannot use efficient float16 on macOS CPU; some PyTorch builds
        # also report CUDA in ways that still run Whisper on CPU but pass float16.
        device = "cpu"
    else:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    override = os.environ.get("TRIBE_WHISPER_COMPUTE_TYPE", "").strip()
    if override:
        compute_type = override
    else:
        compute_type = "float16" if device == "cuda" else "float32"

    with tempfile.TemporaryDirectory() as output_dir:
        logger.info(
            "Running whisperx via uvx (device=%s, compute_type=%s)...",
            device,
            compute_type,
        )
        cmd = [
            "uvx",
            "whisperx",
            str(wav_filename),
            "--model",
            "large-v3",
            "--language",
            language_codes[language],
            "--device",
            device,
            "--compute_type",
            compute_type,
            "--batch_size",
            "16",
            "--align_model",
            "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
            "--output_dir",
            output_dir,
            "--output_format",
            "json",
        ]
        cmd = [c for c in cmd if c]
        env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"whisperx failed:\n{result.stderr}")

        json_path = Path(output_dir) / f"{wav_filename.stem}.json"
        transcript = json.loads(json_path.read_text())

    words = []
    for i, segment in enumerate(transcript["segments"]):
        sentence = segment["text"].replace('"', "")
        for word in segment["words"]:
            if "start" not in word:
                continue
            words.append(
                {
                    "text": word["word"].replace('"', ""),
                    "start": word["start"],
                    "duration": word["end"] - word["start"],
                    "sequence_id": i,
                    "sentence": sentence,
                }
            )

    return pd.DataFrame(words)


def apply_whisper_compute_patch() -> None:
    """Idempotent: replace tribev2 ExtractWordsFromAudio transcript helper."""
    global _applied
    if _applied:
        return
    import tribev2.eventstransforms as et

    et.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(
        _get_transcript_from_audio
    )
    _applied = True
