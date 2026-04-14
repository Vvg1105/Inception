# Inception

> An interactive 3D city builder powered by emotion, neural signals, and AI vision — built with Three.js, FastAPI, and Meta's TRIBE v2.

<!-- Add your hero screenshot here -->
![Inception Hero](docs/images/hero.png)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Demo](#demo)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [EEG & Neural Symbiosis](#eeg--neural-symbiosis)
- [Vision Pipeline (BFL → TRIBE → Classifier)](#vision-pipeline-bfl--tribe--classifier)
- [TRIBE v2 CLI Pipelines](#tribe-v2-cli-pipelines)
- [RunPod / GPU Setup](#runpod--gpu-setup)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [EEG Channel Map](#eeg-channel-map)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

Inception is a browser-based 3D city builder where every building, prop, and environmental detail is shaped by **emotion** (arousal, valence, focus), **mood**, and **environment** (sun elevation, fog density). Users place objects on a grid via text prompts or EEG blink triggers, and an AI backend (Claude) determines material properties — emissive glow, roughness, metalness, scale — to match the emotional context.

A second dimension uses **Meta's TRIBE v2** brain encoder: real photos or AI-generated images are converted to neural feature vectors, classified by a scikit-learn model, and mapped back into the city as specific asset types (bridge, skyscraper, lake, tree, etc.).

<!-- Add a GIF or screenshot of placing a building -->
![Placing a building](docs/images/place-building.gif)

---

## Features

### 3D City Scene
- **100+ Kenney city kit GLB assets** — commercial, industrial, suburban, roads
- **Procedural objects** — water, stadiums, gardens, parks, hills, clouds generated in code
- **Post-processing** — Unreal bloom, ambient occlusion, sky system with real-time sun/fog

### Emotion-Driven Placement
- **Emotion pad** — arousal, valence, and focus control material appearance
- **Mood quadrants** — liminal, euphoric, melancholic, chaotic states shape the environment
- **AI materials** — Claude interprets object + emotion → Three.js material parameters
- **Offline fallback** — heuristic engine mirrors server behavior when backend is down

### Neural Interface (EEG)
- **Dual-headset "Neural Symbiosis"** — g.tec + OpenBCI Cyton, two users, one shared world
- **Blink-to-build** — EEG blink detection opens the placement dialog
- **Live emotion stream** — 10 Hz WebSocket feed drives scene atmosphere in real-time

### AI Vision Pipeline
- **Text → Image → Video → Neural features → Classification** via BFL FLUX + TRIBE v2
- **Streaming NDJSON** responses for progressive UI updates
- **Brain surface visualization** of TRIBE vectors via nilearn

### Eye Tracking
- **MediaPipe iris tracking** via webcam → WebSocket gaze data at 30 fps
- Camera follows gaze direction in the 3D scene

### 3D Model Search
- **Perplexity Sonar** or **Sketchfab** fallback for finding and importing external GLB models
- Built-in CORS proxy for cross-origin model downloads

### Voice Narration
- **ElevenLabs TTS** integration — AI narration played in-scene after placement

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (index.html)                     │
│  Three.js scene · Emotion pad · Mood HUD · Voice · WebSockets  │
│  world-camera/  → orbit rig, pointer, EEG bridge, eye bridge   │
└──────────┬──────────────┬──────────────┬───────────────┬────────┘
           │ HTTP         │ WS :8765     │ WS :8766      │ WS :8767
           ▼              ▼              ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────┐
│  FastAPI     │  │ EEG Decoder  │  │ Eye      │  │ TRIBE WS  │
│  :8000       │  │ (dual/single)│  │ Tracker  │  │ (fMRI     │
│              │  │              │  │          │  │  decoder)  │
│ /api/place   │  │ g.tec +      │  │ MediaPipe│  │           │
│ /api/vision-*│  │ OpenBCI      │  │ + OpenCV │  │ tribe_    │
│ /api/tts     │  │              │  │          │  │ decoding/ │
│ /api/search  │  └──────────────┘  └──────────┘  └───────────┘
│ /api/proxy   │
└──────┬───────┘
       │
  ┌────┴─────────────────────────┐
  │  External APIs               │
  │  · Anthropic Claude          │
  │  · BFL FLUX (image gen)      │
  │  · ElevenLabs (TTS)          │
  │  · Perplexity / Sketchfab    │
  └──────────────────────────────┘
```

<!-- Add architecture diagram image if you have one -->
<!-- ![Architecture Diagram](docs/images/architecture.png) -->

---

## Demo

<!-- Add screenshots or GIFs for each section -->

| Feature | Preview |
|---------|---------|
| City scene overview | ![City Overview](docs/images/city-overview.png) |
| Emotion pad + placement | ![Emotion Pad](docs/images/emotion-pad.png) |
| EEG Neural Symbiosis | ![EEG Symbiosis](docs/images/eeg-symbiosis.png) |
| Vision pipeline result | ![Vision Pipeline](docs/images/vision-pipeline.png) |

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js** is not required — the frontend uses vanilla ES modules with Three.js via CDN
- **ffmpeg** on `PATH` (required for vision pipeline)
- EEG hardware (optional): g.tec BCICore-8 and/or OpenBCI Cyton

### 1. Clone the repository

```bash
git clone <YOUR_REPO_URL> Inception
cd Inception
```

### 2. Set up the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys (see Environment Variables below)
```

### 4. Start the servers

Open **two terminals**:

**Terminal 1 — API server:**

```bash
cd backend
source .venv/bin/activate
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — Static file server:**

```bash
# From repo root
python -m http.server 8080
```

Open **http://localhost:8080/index.html** — the console should show `[Backend] online`.

### 5. Optional services

**Eye tracking** (Terminal 3):

```bash
cd backend
python eye_track.py                  # opens webcam debug preview
python eye_track.py --no-preview     # headless mode
python eye_track.py --sens 1.5       # increase gaze sensitivity
```

**EEG dual decoder** (Terminal 4):

```bash
cd backend
python eeg_decode_dual.py --mock     # simulated (no hardware)
python eeg_decode_dual.py            # real hardware (BLE + USB)
```

---

## Environment Variables

Create a `.env` file at the repo root (loaded by the backend automatically).

### API Keys

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Recommended | Claude API for intelligent material generation. Alias: `CLAUDE_API_KEY` |
| `ANTHROPIC_MODEL` | No | Model override (default: `claude-3-5-sonnet-20241022`) |
| `BFL_API_KEY` | For vision | BFL FLUX image generation for the vision pipeline |
| `BFL_MODEL` | No | FLUX model (default: `flux-2-klein-4b`) |
| `PERPLEXITY_API_KEY` | For model search | 3D model search via Perplexity Sonar. Alias: `PPLX_API_KEY` |
| `SKETCHFAB_API_TOKEN` | For model search | Fallback 3D model search + download |
| `ELEVEN_LABS_API_KEY` | For narration | ElevenLabs text-to-speech |

### TRIBE Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRIBE_FORCE_CPU` | unset | Force CPU inference (set to `1` on Mac / no GPU) |
| `TRIBE_VIDEO_SKIP_WHISPER` | `1` | Skip Whisper ASR on video audio |
| `TRIBE_FEATURES_VIDEO_ONLY` | `1` | Only load video extractor (skip Llama/Wav2Vec) |
| `TRIBE_CACHE_FOLDER` | — | Override model cache directory |
| `TRIBE_DATALOADER_WORKERS` | `4` (CUDA) | DataLoader worker count |
| `TRIBE_CUDNN_BENCHMARK` | `1` (CUDA) | Enable cuDNN benchmark |
| `TRIBE_WHISPER_DEVICE` | auto | Override Whisper device (`cpu` / `cuda`) |
| `TRIBE_WHISPER_COMPUTE_TYPE` | auto | Override Whisper compute type (`float16` / `float32`) |

Without an Anthropic key, the server falls back to a deterministic **heuristic engine** that mirrors the client-side `buildLocalParams()` function.

---

## Usage

### Placing Objects

1. **Blink** (with EEG + "open build after blink" enabled) or **click the ground** to open the placement dialog
2. Type a building name (e.g. "red massive bridge", "glass skyscraper", "small cottage")
3. The client sends the label + emotion state + environment to `POST /api/place`
4. Claude (or the heuristic fallback) returns material parameters
5. The object spawns on the grid with emotion-driven materials

<!-- Screenshot of the place dialog -->
![Place Dialog](docs/images/place-dialog.png)

### Emotion & Mood

The **emotion pad** controls three axes:
- **Arousal** — energy level (calm → excited)
- **Valence** — positivity (negative → positive)
- **Focus** — attention (diffuse → concentrated)

These values influence material properties (emissive intensity, roughness) and environment (fog density, sun position, ambient color).

<!-- Screenshot of the emotion pad -->
![Emotion Pad UI](docs/images/emotion-pad-ui.png)

### Vision Placement

Enable the **"vision (BFL→TRIBE→class)"** checkbox in the placement dialog:
1. Your text prompt generates an image via BFL FLUX
2. The image is converted to a short video
3. TRIBE v2 extracts neural features from the video
4. A sklearn classifier predicts the city element class
5. The corresponding 3D asset spawns in the scene

<!-- Screenshot of vision placement result -->
![Vision Placement](docs/images/vision-placement.png)

---

## EEG & Neural Symbiosis

Inception supports live EEG input from one or two headsets simultaneously.

### Dual Mode (Neural Symbiosis)

Two users wear EEG headsets (g.tec BCICore-8 + OpenBCI Cyton) and co-create in a shared world. The system tracks:

- **Per-user emotion** (arousal, valence, focus)
- **Blink detection** for hands-free building placement
- **Symbiosis scores** — correlation metrics between the two brain signals
- **Active user** — determines who has creation permission (toggled via stdin)

```
Python (eeg_decode_dual.py)                Browser (index.html)
───────────────────────────                ─────────────────────
User 1: g.tec BCICore-8                    ← ws://127.0.0.1:8765
User 2: OpenBCI Cyton                      dual EEG bridge + symbiosis UI
→ JSON: { user1, user2, active_user, symbiosis, capture }
```

### Single Mode

One g.tec headset streams flat JSON at 10 Hz:

```json
{
  "present": true,
  "blink": false,
  "emotion": { "arousal": 0.72, "valence": 0.41, "focus": 0.60 }
}
```

### Mock Mode

Test without hardware:

```bash
python eeg_decode_dual.py --mock     # dual mock
python eeg_decode.py --mock          # single mock
python eeg_mock_ws.py                # minimal blink-only mock
```

<!-- Screenshot of EEG visualization -->
![EEG Live Feed](docs/images/eeg-live.png)

---

## Vision Pipeline (BFL → TRIBE → Classifier)

The vision pipeline transforms text prompts into classified city elements through a multi-stage neural pipeline.

```
Text prompt
    │
    ▼
BFL FLUX API ──── generates image
    │
    ▼
ffmpeg ────────── image → MP4 (short clip)
    │
    ▼
TRIBE v2 ──────── extracts neural feature vector
    │
    ▼
sklearn ───────── classifies into city element
classifier        (bridge, lake, skyscraper, tree, house, …)
    │
    ▼
3D asset spawns in scene
```

### Requirements

- `BFL_API_KEY` in `.env`
- TRIBE v2 installed: `pip install -r requirements-tribe.txt`
- Trained classifier at `outputs/photo_element_logreg.joblib`
- `ffmpeg` on PATH

### Brain Visualization

The backend can render TRIBE feature vectors as cortical surface maps using nilearn, returned as part of the streaming vision pipeline response.

<!-- Brain visualization screenshot -->
![Brain Render](docs/images/brain-render.png)

---

## TRIBE v2 CLI Pipelines

Train and evaluate classifiers from the command line.

| Command | Description |
|---------|-------------|
| `python -m pipeline.neural_matrix --help` | Text CSV → TRIBE neural feature matrix |
| `python -m pipeline.photo_neural_matrix --help` | Photos → MP4 → TRIBE feature matrix |
| `python -m pipeline.train_element_classifier --help` | Train sklearn classifier on `.npz` |
| `python -m pipeline.eval_element_classifier --help` | Evaluate on holdout `.npz` |
| `python -m pipeline.classify_text --help` | Classify a single text phrase |
| `python -m pipeline.bfl_tribe_classify --help` | BFL text→image→TRIBE→class (needs `BFL_API_KEY`) |

### Photo Pipeline

```bash
# Place images under data/photo_dataset/source/<class>/
# e.g. data/photo_dataset/source/bridge/photo1.jpg

python -m pipeline.photo_neural_matrix \
  --dataset-root data/photo_dataset \
  --output outputs/photo_tribe_neural.npz \
  --holdout-per-class 2   # reserve 2 images per class for testing
```

### Training

```bash
python -m pipeline.train_element_classifier \
  --input outputs/photo_tribe_neural.npz \
  --output outputs/photo_element_logreg.joblib

python -m pipeline.eval_element_classifier \
  --model outputs/photo_element_logreg.joblib \
  --input outputs/photo_tribe_neural_holdout.npz
```

### Platform Notes

**macOS / CPU:**

```bash
export TRIBE_FORCE_CPU=1
export CUDA_VISIBLE_DEVICES=
pip install -r requirements-tribe.txt
```

WhisperX is forced to `--device cpu` + `float32` on Darwin to avoid ctranslate2 float16 crashes.

**GPU / RunPod:** See [RunPod Setup](#runpod--gpu-setup) below.

---

## RunPod / GPU Setup

For heavy TRIBE workloads, use a GPU cloud instance.

### Recommended GPUs

| Tier | GPUs | VRAM |
|------|------|------|
| Minimum | RTX 4000 Ada, A4000, RTX 4080 | 16 GB |
| **Recommended** | **RTX 3090, RTX 4090, L4, A10** | **24 GB** |
| Headroom | A6000, A100 | 40–48 GB |

### Setup

```bash
unset TRIBE_FORCE_CPU
export HF_HOME=/workspace/.cache/huggingface
export UV_CACHE_DIR=/workspace/.cache/uv

# Install ffmpeg
sudo apt-get update && sudo apt-get install -y ffmpeg

# Install uv (tribev2 calls uvx whisperx)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone and set up
cd /workspace
git clone <YOUR_REPO_URL> imagine && cd imagine
python3 -m venv /workspace/.venv && source /workspace/.venv/bin/activate
pip install -U pip && pip install -r requirements-runpod.txt
```

Verify:

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

See [RUNPOD.md](RUNPOD.md) for full details and troubleshooting.

---

## API Reference

All endpoints are served by the FastAPI backend at `http://localhost:8000`.

### `GET /health`

Health check — returns `{ "status": "ok" }`.

### `POST /api/place`

Place an object with emotion-driven materials.

**Request:**

```json
{
  "label": "red massive bridge",
  "base_label": "bridge",
  "hints": {
    "color": "#cc3333",
    "size": 1.2,
    "material": { "roughness": 0.4, "metalness": 0.1 }
  },
  "emotion": { "arousal": 0.7, "valence": 0.4, "focus": 0.6 },
  "mood": "liminal",
  "environment": {
    "mood_quadrant": "liminal",
    "sun_elevation_deg": 18,
    "fog_density": 0.3
  }
}
```

**Response:**

```json
{
  "material_params": {
    "emissive": "#221100",
    "emissiveIntensity": 0.08,
    "roughness": 0.45,
    "metalness": 0.2,
    "scale": 1.0,
    "pointLight": null
  },
  "narration": "optional line shown in the scene",
  "audio_b64": null
}
```

### `POST /api/vision-pipeline`

Streaming NDJSON — generates image, classifies it, and returns brain visualization.

### `POST /api/vision-classify`

BFL image → MP4 → TRIBE → classifier label.

### `POST /api/vision-imagine`

Generate an image via BFL FLUX (returns base64).

### `POST /api/vision-classify-image`

Classify an existing image via TRIBE.

### `GET /api/search-model?q=...`

Search for 3D models via Perplexity Sonar or Sketchfab fallback.

### `GET /api/proxy-glb?url=...`

Proxy external GLB files to avoid CORS issues.

### `POST /api/tts`

Text-to-speech via ElevenLabs — returns audio.

---

## Project Structure

```
Inception/
├── index.html                  # Main SPA — Three.js scene, UI, placement logic
├── world-camera/               # ES modules for camera, input, and bridges
│   ├── index.js                # Orbit rig, pointer ground follow
│   ├── eeg-bridge.js           # Single EEG WebSocket bridge
│   ├── eeg-bridge-dual.js      # Dual EEG (Neural Symbiosis) bridge
│   ├── eye-bridge.js           # Eye tracking WebSocket bridge
│   └── tribe-bridge.js         # TRIBE fMRI WebSocket bridge
├── backend/                    # FastAPI placement & vision API
│   ├── app.py                  # All HTTP routes, CORS, env loading
│   ├── vision_place.py         # BFL → TRIBE → sklearn classification
│   ├── brain_render.py         # Cortical surface visualization (nilearn)
│   ├── eye_track.py            # MediaPipe iris → WebSocket (:8766)
│   ├── eeg_decode_dual.py      # Dual EEG decoder → WebSocket (:8765)
│   ├── eeg_decode.py           # Single EEG decoder
│   ├── eeg_mock_ws.py          # Minimal blink-only mock
│   ├── tribe_ws.py             # TRIBE fMRI WebSocket server
│   └── requirements.txt
├── pipeline/                   # TRIBE CLI tools
│   ├── neural_matrix.py        # Text CSV → TRIBE features
│   ├── photo_neural_matrix.py  # Photos → MP4 → TRIBE features
│   ├── train_element_classifier.py
│   ├── eval_element_classifier.py
│   ├── classify_text.py
│   └── bfl_api.py              # BFL FLUX API client
├── tribe/                      # TRIBE v2 helpers
│   ├── model.py                # load_model, device selection
│   ├── env_flags.py            # Environment variable configuration
│   └── whisper_patch.py        # Platform-aware WhisperX patching
├── tribe_decoding/             # sklearn PCA + logistic decoders (fMRI)
├── eeg/                        # EEGNet training, blink detection, data
│   ├── models/                 # Trained EEG models
│   ├── data/                   # Collected EEG data
│   └── eegnet.py               # EEGNet architecture
├── emg/                        # EMG utilities (separate requirements)
├── gpype/                      # Vendored g.tec BCI Python package
├── assets/                     # Kenney city kit GLB models
│   ├── commercial/
│   ├── industrial/
│   ├── suburban/
│   └── roads/
├── data/                       # Photo datasets for training
│   └── photo_dataset/source/
├── tools/                      # Calibration scripts
├── scripts/                    # RunPod SSH, venv helpers
├── city_elements_dataset.csv   # Training data for text classifier
├── requirements-tribe.txt      # TRIBE dependencies (CPU / Mac)
├── requirements-runpod.txt     # TRIBE dependencies (GPU / RunPod)
├── .env.example                # Template for API keys
├── RUNPOD.md                   # GPU cloud setup guide
└── README.md                   # This file
```

---

## EEG Channel Map

### EEG 1 — g.tec BCICore-8

| Channel | 10-20 Position |
|---------|---------------|
| Ch1 | Fp2 |
| Ch2 | F7 |
| Ch3 | FC5 |
| Ch4 | Fp1 |
| Ch5 | P7 |
| Ch6 | T7 |
| Ch7 | T8 |
| Ch8 | O2 |

### EEG 2 — OpenBCI Cyton

| Channel | Wire Color | 10-20 Position |
|---------|-----------|---------------|
| Ch0 | Grey | Fp2 |
| Ch1 | Purple | C4 |
| Ch2 | Blue | Cz |
| Ch3 | Green | T3 |
| Ch4 | Yellow | T4 |
| Ch5 | Orange | F3 |
| Ch6 | Red | Fz |
| Ch7 | Brown | F4 |

<!-- Photo of the EEG headset setup -->
![EEG Setup](docs/images/eeg-setup.png)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `[Backend] offline` in browser console | Ensure FastAPI is running on port 8000 |
| No materials applied (flat grey) | Check `ANTHROPIC_API_KEY` in `.env` — heuristic fallback still works |
| Vision pipeline returns 503 | Missing `BFL_API_KEY`, TRIBE not installed, or classifier `.joblib` not found |
| `Torch not compiled with CUDA` | Wrong PyTorch wheel — use the CUDA image, don't install CPU torch on top |
| CUDA OOM | Use 24 GB+ GPU, run one pipeline at a time, or shorten `--duration` |
| WhisperX crash on macOS | Already handled — `tribe/whisper_patch.py` forces CPU + float32 on Darwin |
| `ffmpeg: command not found` | Install ffmpeg: `brew install ffmpeg` (Mac) or `apt install ffmpeg` (Linux) |
| ES module import errors | Serve via HTTP server, not `file://` — `python -m http.server 8080` |

---

## License

<!-- Add your license here -->

---

<p align="center">
  Built with Three.js · FastAPI · TRIBE v2 · Claude · BFL FLUX
</p>
