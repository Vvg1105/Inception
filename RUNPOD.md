# RunPod (GPU) — neural matrix pipeline

Use a **PyTorch + CUDA** template (e.g. RunPod `runpod/pytorch` with CUDA). Mount a **volume** on `/workspace` for caches and outputs.

## One-time setup (SSH into the pod)

```bash
# Persist caches on the volume
export HF_HOME=/workspace/.cache/huggingface
export UV_CACHE_DIR=/workspace/.cache/uv
mkdir -p /workspace/imagine /workspace/outputs /workspace/.cache

# uv — tribev2 calls `uvx whisperx`
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspace
git clone <YOUR_REPO_URL> imagine
cd imagine

python3 -m venv /workspace/.venv
source /workspace/.venv/bin/activate

# Prefer image torch; see header in requirements-runpod.txt if tribev2 conflicts on torch version
pip install -U pip
pip install -r requirements-runpod.txt
```

Verify GPU:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## Run the pipeline

```bash
source /workspace/.venv/bin/activate
cd /workspace/imagine

python -m pipeline.neural_matrix \
  --csv city_elements_dataset.csv \
  --cache-folder /workspace/cache/tribe \
  --row-cache-dir /workspace/cache/neural_rows \
  --output /workspace/outputs/city_elements_neural.npz
```

- **CUDA**: PyTorch sees the GPU → WhisperX runs with `--device cuda` and **float16** (via `tribe/whisper_patch.py`).
- **CPU / Mac**: same patch uses **float32** for Whisper so ctranslate2 does not error.
- **Override** (optional): `export TRIBE_WHISPER_COMPUTE_TYPE=int8_float16` (or `float32`) if you need a different Whisper backend mode.

## Fetch results

Download `/workspace/outputs/city_elements_neural.npz` and optionally `/workspace/cache/neural_rows/` (resume cache) before terminating the pod if you do not use a persistent volume for those paths.
