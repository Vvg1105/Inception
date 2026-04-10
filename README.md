# imagine

TRIBE-based neural feature pipeline for the city-elements sentence dataset.

- **Local:** `pip install -r requirements-tribe.txt` then `python -m pipeline.neural_matrix --help`
- **RunPod / GPU:** see [RUNPOD.md](RUNPOD.md)

WhisperX is patched via `tribe/whisper_patch.py`: **float16 on CUDA**, **float32 on CPU**.
