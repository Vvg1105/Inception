# imagine

TRIBE v2 neural features for **text** and **photo→video** inputs, plus a small **sklearn** classifier on top.

- **Inception / LLM placement API (FastAPI):** see [backend/README.md](backend/README.md) — `uvicorn app:app --port 8000` so `index.html` can POST emotion + object label and receive `material_params`.

| Piece | Command |
|--------|---------|
| Text CSV → matrix | `python -m pipeline.neural_matrix --help` |
| Photos → MP4 → matrix | `python -m pipeline.photo_neural_matrix --help` |
| Train classifier | `python -m pipeline.train_element_classifier --help` |
| Eval on holdout `.npz` | `python -m pipeline.eval_element_classifier --help` |
| Classify one phrase | `python -m pipeline.classify_text --help` |
| BFL text→image→TRIBE→class | `python -m pipeline.bfl_tribe_classify --help` (needs `BFL_API_KEY` in `.env`) |

**Photo / video:** by default **no Whisper/ASR** on video audio (`TRIBE_VIDEO_SKIP_WHISPER=1`) and **video-only TRIBE extractors** (`TRIBE_FEATURES_VIDEO_ONLY=1`, skips loading Llama/Wav2Vec for inference). Use **`--video-whisper`** and **`--tribe-all-modalities`** on `photo_neural_matrix` for the full multimodal stack.

**Photos on GitHub → RunPod:** commit/push images under `data/photo_dataset/source/<class>/`, then on the pod `cd` to the repo and `git pull`. Run `python -m pipeline.photo_neural_matrix ...` (see `--help`). To hold out **K** images per class for testing only, add **`--holdout-per-class K`**; training uses **`--output`**; the holdout matrix defaults to `<output_stem>_holdout.npz` — **train the classifier only on `--output`**, not on the holdout file.

- **Local (Mac / CPU):** `pip install -r requirements-tribe.txt`
- **RunPod / GPU:** [RUNPOD.md](RUNPOD.md) and `requirements-runpod.txt` (do not replace the image’s CUDA PyTorch with CPU wheels)
- **EMG utilities (separate):** `emg/` — own `requirements.txt`

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


# EEG

Understanding which channels correspond to what placement will help you create a visualization map for EEG, live. 
### EEG 1 - GTECH
Ch8 - O2
ch5 - p7
ch6 - t7
ch3 - fc5
ch2 - f7
ch4 - fp1
ch7 - t8
ch1 - fp2


### EEG 2 - OPENBCI
ch0 - grey C3
ch1 - purple C4
ch2 - blue CZ
ch3 - green T3
ch4 - yellow T4
ch5 - orange F3
ch6 - red FZ
ch7 - brown F4