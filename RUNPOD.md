# RunPod (GPU) — TRIBE pipelines

Use a template with **NVIDIA GPU + PyTorch built with CUDA** (e.g. RunPod **PyTorch** images). Mount a **volume** on `/workspace` for caches and outputs.

## GPU and VRAM (what to pick)

TRIBE loads several heavy pieces at once (video encoder e.g. VideoMAE, audio/Wav2Vec stack, brain model, plus WhisperX on the **text** path and **audio extracted from video**). Peak VRAM is usually **well above 8 GB**.

| Tier | Typical GPUs (examples) | Notes |
|------|-------------------------|--------|
| **Minimum** | **16 GB** — RTX 4000 Ada / A4000 / RTX 4080 (16 GB) | Often works; if you hit CUDA OOM, use a **24 GB** pod or reduce parallelism (one job at a time, shorter `--duration` on photo pipeline). |
| **Recommended** | **24 GB** — **RTX 3090**, **RTX 4090**, **L4**, **A10** | Comfortable default for `neural_matrix` and `photo_neural_matrix`. |
| **Headroom** | **40–48 GB** — A6000, A100 40GB | Use if you experiment with larger batches or very long clips. |

**Avoid** picking a template that reinstalls **CPU-only** PyTorch over the image’s CUDA build. Follow `requirements-runpod.txt` (install **tribev2** without replacing `torch`).

## One-time setup (SSH into the pod)

```bash
# Do NOT set TRIBE_FORCE_CPU on RunPod — you want CUDA.
unset TRIBE_FORCE_CPU

# Persist caches on the volume
export HF_HOME=/workspace/.cache/huggingface
export UV_CACHE_DIR=/workspace/.cache/uv
mkdir -p /workspace/imagine /workspace/outputs /workspace/.cache

# ffmpeg (needed for photo → MP4 and many video tools)
sudo apt-get update && sudo apt-get install -y ffmpeg

# uv — tribev2 calls `uvx whisperx`
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspace
git clone <YOUR_REPO_URL> imagine
cd imagine

python3 -m venv /workspace/.venv
source /workspace/.venv/bin/activate

pip install -U pip
pip install -r requirements-runpod.txt
```

Verify GPU and PyTorch CUDA:

```bash
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Optional tuning (GPU):

```bash
# dataloader workers for video feature extraction (default 4 on CUDA if unset)
export TRIBE_DATALOADER_WORKERS=4

# disable cudnn.benchmark if you debug nondeterminism (default on for CUDA)
# export TRIBE_CUDNN_BENCHMARK=0
```

## Text CSV → TRIBE → `city_elements_neural.npz`

```bash
source /workspace/.venv/bin/activate
cd /workspace/imagine

python -m pipeline.neural_matrix \
  --csv city_elements_dataset.csv \
  --cache-folder /workspace/cache/tribe \
  --row-cache-dir /workspace/cache/neural_rows \
  --output /workspace/outputs/city_elements_neural.npz
```

## Photos → MP4 → TRIBE → `photo_tribe_neural.npz`

Put images under `data/photo_dataset/source/<class>/` (see `pipeline/photo_neural_matrix.py` defaults), then:

```bash
source /workspace/.venv/bin/activate
cd /workspace/imagine

python -m pipeline.photo_neural_matrix \
  --dataset-root /workspace/imagine/data/photo_dataset \
  --cache-folder /workspace/cache/tribe \
  --row-cache-dir /workspace/cache/photo_neural_rows \
  --output /workspace/outputs/photo_tribe_neural.npz
```

**Video speed:** this pipeline **defaults to skipping Whisper/ASR** on extracted audio (`TRIBE_VIDEO_SKIP_WHISPER=1`) and **only preparing the video feature extractor** (`TRIBE_FEATURES_VIDEO_ONLY=1`), which avoids loading Llama and Wav2Vec for inference. For the full tribev2 multimodal stack, pass **`--video-whisper`** and **`--tribe-all-modalities`** (or set `TRIBE_VIDEO_SKIP_WHISPER=0` and `TRIBE_FEATURES_VIDEO_ONLY=0` before Python starts).

Adjust `--row-cache-dir` if you want shards on the volume only.

## CUDA vs CPU behavior

- **RunPod (Linux + CUDA PyTorch):** `load_model()` uses **`device="cuda"`** for the TRIBE brain model when `TRIBE_FORCE_CPU` is unset and a test allocation on GPU succeeds. Nested vision/audio extractors stay on GPU with that config. WhisperX uses **`--device cuda`** and **float16** (see `tribe/whisper_patch.py`) unless you override `TRIBE_WHISPER_DEVICE` / `TRIBE_WHISPER_COMPUTE_TYPE`.
- **Mac CPU:** set `export TRIBE_FORCE_CPU=1` and use `requirements-tribe.txt` CPU wheels.

## Fetch results

Download `/workspace/outputs/*.npz` and row-cache directories before terminating the pod if those paths are not on a persistent volume.

## Troubleshooting

- **`Torch not compiled with CUDA`:** Wrong PyTorch wheel; use a CUDA image and do not `pip install torch` from PyPI CPU builds on top of it.
- **CUDA OOM:** Use a **24 GB+** GPU, run **one pipeline at a time**, shorten `--duration` on `photo_neural_matrix`, or try `export TRIBE_WHISPER_COMPUTE_TYPE=float32` (more VRAM for Whisper) only if you understand the tradeoff.
- **Override Whisper device:** `export TRIBE_WHISPER_DEVICE=cuda` or `cpu`; compute type: `TRIBE_WHISPER_COMPUTE_TYPE=int8_float16` or `float32`.
