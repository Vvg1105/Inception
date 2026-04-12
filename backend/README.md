# Inception — placement API (Python)

Serves `POST /api/place` for the static front end (`index.html` → `getLLMMaterials`). The browser sends **object label + parsed hints + emotion**; you return **Three.js–compatible `material_params`** (and optional **narration**).

## Run (four terminals)

**1 — API (this folder)** loads `.env` from **repo root** or `backend/.env`.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

**2 — Static site** (ES modules need HTTP, not `file://`):

```bash
cd ..   # repo root (Inception)
python -m http.server 8080
```

**3 — Eye tracking** (iris position via webcam + MediaPipe → WebSocket at 30 fps):

```bash
cd backend
python eye_track.py                  # opens webcam, shows debug preview
python eye_track.py --no-preview     # headless (no OpenCV window)
python eye_track.py --sens 1.5       # increase gaze sensitivity
```

**4 — EEG decoder** (emotion + blink every 100 ms):

**Neural Symbiosis — two headsets** (g.tec BCICore-8 + OpenBCI Cyton on one machine). This is what `index.html` auto-connects to on port 8765:

```bash
cd backend
python eeg_decode_dual.py --mock                    # both users fake
python eeg_decode_dual.py                           # both real (BLE + USB Cyton)
python eeg_decode_dual.py --cyton-port /dev/cu.usbserial-XXXX
```

**Single headset** (g.tec only — use the World panel “EEG (single)” connect instead of dual):

```bash
python eeg_decode.py --mock
python eeg_decode.py
```

Open **http://localhost:8080/index.html**. The page calls **http://localhost:8000** — keep both running. Console should show `[Backend] online`.

### Vision placement (BFL → TRIBE → classifier)

With **`BFL_API_KEY`** set and **TRIBE + classifier** installed in the **same** venv as the API
(e.g. `pip install -r ../requirements-tribe.txt` from repo root, then your classifier at
`outputs/photo_element_logreg.joblib`), the place popup checkbox **vision (BFL→TRIBE→class)**
calls **`POST /api/vision-classify`**. The returned class sets which asset is spawned
(bridge, lake, skyscraper, tree, house, …). First request loads TRIBE (slow). Needs **ffmpeg** on `PATH`.

### What happens when you place

1. **Blink** (with "open build after blink") or **click the ground** → place dialog opens.  
2. Type a building name → **place**. The client POSTs **label + emotion pad + mood text + environment** (sun, fog, mood quadrant).  
3. The API returns **`material_params`** including **`scale`** (building size) and surface fields; the scene **already** uses emotion for fog/light via the mood pad (`applyEmotion`).

CORS is open for local dev.

## Environment

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | **Anthropic Claude** API key. `CLAUDE_API_KEY` is accepted as an alias. |
| `ANTHROPIC_MODEL` | Optional — default `claude-3-5-sonnet-20241022` (override with e.g. Sonnet 4 if your key supports it). |

Without an Anthropic key, the server uses the **heuristic** in `heuristic_materials()` (aligned with the client's `buildLocalParams`).

## Payload (request JSON)

Sent by the client:

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
  "environment": { "mood_quadrant": "liminal", "sun_elevation_deg": 18, "fog_density": 0.3 }
}
```

## Response (JSON)

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

`material_params` is passed straight into `applyMaterialParams()` in `index.html` (emissive, roughness, metalness, color, scale, optional `pointLight`, etc.).

## Health check

`GET /health` → `{ "status": "ok" }` — the page uses this to set `backendOnline` and switch off the local JS fallback.

---

## EEG live decode → emotion + blink → scene

### Dual mode (default UI): `eeg_decode_dual.py`

Two boards in parallel → one WebSocket. The client uses `world-camera/eeg-bridge-dual.js` and fills `window.__dreamEEG1`, `__dreamEEG2`, `__dreamEEG` (active user), plus symbiosis scores.

```
Python (eeg_decode_dual.py)                Browser (index.html)
───────────────────────────                ─────────────────────
User1: LiveDecoder (GTECH)                 ← ws://127.0.0.1:8765
User2: CytonDecoder (OpenBCI)              dual EEG bridge + Neural Symbiosis UI
→ JSON with user1, user2, active_user, symbiosis, capture
```

### Single-board mode: `eeg_decode.py`

One g.tec headset — JSON shape is flat (`present`, `blink`, `emotion`). Use when only `eeg_decode.py` is running.

### JSON (single decoder) every 100 ms

```json
{
  "present": true,
  "blink": false,
  "emotion": { "arousal": 0.72, "valence": 0.41, "focus": 0.60 }
}
```

When the active user **blinks** (dual) or a capture fires (single), the decoder may add `"capture": true`. The browser opens the place dialog on that event.

### Plugging in real hardware

- **Dual:** see `backend/eeg_decode_dual.py` (threads + stdin to swap active user).  
- **Single:** subclass `EEGDecoder` in `eeg_decode.py` — override `setup()`, `read_raw()`, `decode()`. See `MockDecoder` as an example.

### Old simple mock

`eeg_mock_ws.py` still works (blink-only, no emotion stream) for quick capture testing.
