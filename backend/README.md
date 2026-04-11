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

```bash
cd backend
python eeg_decode.py --mock          # fake sine-wave emotions + blink
python eeg_decode.py                 # real hardware (edit EEGDecoder subclass)
```

Open **http://localhost:8080/index.html**. The page calls **http://localhost:8000** — keep both running. Console should show `[Backend] online`.

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

### How it works

```
Python (eeg_decode.py)                     Browser (index.html)
──────────────────────                     ─────────────────────
EEG hardware / LSL / serial                ← WebSocket (100 ms) ←
↓                                          reads window.__dreamEEG
decode() → arousal, valence, focus, blink  ↓
→ JSON over ws://127.0.0.1:8765  ──────►  1. HUD shows A / V / F numbers
                                           2. mood pad + focus slider move
                                           3. applyEmotion() → fog, sky, light
                                           4. blink edge → "capture" → place UI
```

### JSON sent every 100 ms

```json
{
  "present": true,
  "blink": false,
  "emotion": { "arousal": 0.72, "valence": 0.41, "focus": 0.60 }
}
```

When a **blink ends** (eyes close → open), the Python decoder adds `"capture": true` to that frame. The browser treats this like `dream-blink-up` and opens the place dialog.

### Plugging in real hardware

Subclass `EEGDecoder` in `eeg_decode.py` — override `setup()`, `read_raw()`, `decode()`. See the file for the full template and `MockDecoder` as an example.

### Old simple mock

`eeg_mock_ws.py` still works (blink-only, no emotion stream) for quick capture testing.
