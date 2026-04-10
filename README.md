# imagine

TRIBE-based neural feature pipeline for the city-elements sentence dataset.

- **Inception / LLM placement API (FastAPI):** see [backend/README.md](backend/README.md) — `uvicorn app:app --port 8000` so `index.html` can POST emotion + object label and receive `material_params`.

- **Local:** `pip install -r requirements-tribe.txt` then `python -m pipeline.neural_matrix --help`
- **RunPod / GPU:** see [RUNPOD.md](RUNPOD.md)

WhisperX is patched via `tribe/whisper_patch.py`: **float16 on CUDA**, **float32 on CPU**.

### macOS

WhisperX is forced to **`--device cpu` + `float32`** on Darwin so ctranslate2 does not request float16 on CPU (avoids a common crash). Override with `TRIBE_WHISPER_DEVICE` / `TRIBE_WHISPER_COMPUTE_TYPE` if you know what you’re doing.

### CPU-only (no GPU / broken CUDA)

Set this **before** starting Python so TRIBE never loads weights on GPU and WhisperX uses CPU:

```bash
export TRIBE_FORCE_CPU=1
export CUDA_VISIBLE_DEVICES=
python -m pipeline.neural_matrix --csv city_elements_dataset.csv --output outputs/city_elements_neural.npz
```
