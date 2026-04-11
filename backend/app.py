"""
Dream City Void — placement LLM API + 3D model search

The static site (`index.html`) POSTs here when you place an object. Payload carries
the typed label, parsed hints, and current emotion pad. This server returns
`material_params` for Three.js `applyMaterialParams`, plus optional narration.

Also provides `/api/search-model` for finding downloadable 3D models via Sketchfab.

Run:
  cd backend && pip install -r requirements.txt && uvicorn app:app --reload --port 8000

Env:
  ANTHROPIC_API_KEY (or CLAUDE_API_KEY) — Claude API key for `/api/place`
  ANTHROPIC_MODEL     — optional, default claude-3-5-sonnet-20241022
  PERPLEXITY_API_KEY (or PPLX_API_KEY) — Perplexity Sonar API key for 3D model search
  BFL_API_KEY — Black Forest Labs FLUX API for ``/api/vision-classify`` (same as pipeline.bfl_tribe_classify)
  BFL_MODEL — optional FLUX endpoint name (default flux-2-klein-4b)

``/api/vision-classify`` also needs TRIBE + the photo classifier joblib in the **same** venv
(``pip install -r ../requirements-tribe.txt`` from repo root, or use your RunPod venv).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import asyncio

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

# pipeline/, tribe/, etc. live at repo root (parent of backend/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Repo root .env (run `uvicorn` from `backend/` or project root)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(title="Inception place API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _preload_tribe():
    """Pre-warm TRIBE + classifier in a background thread so first request is fast."""
    if os.getenv("TRIBE_PRELOAD", "1").strip().lower() in ("1", "true", "yes"):
        try:
            from vision_place import preload_models

            loop = asyncio.get_running_loop()
            loop.run_in_executor(
                None,
                lambda: preload_models(os.getenv("TRIBE_CACHE_FOLDER")),
            )
        except Exception as exc:
            print(f"[startup] TRIBE preload skipped: {exc}")


# ── Request / response shapes (match `getLLMMaterials` in index.html) ─────────


class Emotion(BaseModel):
    arousal: float = Field(ge=0, le=1, default=0.5)
    valence: float = Field(ge=0, le=1, default=0.5)
    focus: float = Field(ge=0, le=1, default=0.5)


class PlaceRequest(BaseModel):
    """JSON body from the browser — object label + mood + emotion + scene snapshot."""

    label: str = ""
    base_label: str = ""
    hints: dict[str, Any] = Field(default_factory=dict)
    emotion: Emotion = Field(default_factory=Emotion)
    mood: str = ""
    environment: dict[str, Any] = Field(default_factory=dict)


class PlaceResponse(BaseModel):
    material_params: dict[str, Any]
    narration: str | None = None
    audio_b64: str | None = None


class VisionClassifyRequest(BaseModel):
    """Prompt for BFL → TRIBE → element classifier (matches client emotion payload)."""

    prompt: str
    emotion: Emotion = Field(default_factory=Emotion)
    mood: str = ""
    environment: dict[str, Any] = Field(default_factory=dict)


class VisionClassifyResponse(BaseModel):
    classified_label: str
    place_key: str
    confidence: float
    probabilities: dict[str, float]
    narration: str
    tribe_dim: int = 0
    brain_image_b64: str = ""


class VisionImagineResponse(BaseModel):
    """Step 1: just the BFL-generated image (returned fast, before TRIBE runs)."""
    image_b64: str
    mime: str


class VisionClassifyFromImageRequest(BaseModel):
    """Step 2: send the previously-generated image back for TRIBE → classify."""
    image_b64: str
    mime: str = "image/jpeg"
    prompt: str = ""


# ── Local heuristic (same idea as `buildLocalParams` in index.html) ──────────


def _stable01(seed: str) -> float:
    h = hashlib.sha256(seed.encode()).digest()
    return int.from_bytes(h[:4], "big") / 0xFFFFFFFF


def heuristic_materials(req: PlaceRequest) -> dict[str, Any]:
    e = req.emotion
    hints = req.hints or {}
    color = hints.get("color")
    size = hints.get("size")
    mat_hints: dict[str, Any] = dict(hints.get("material") or {})

    ei = e.arousal * 0.18
    if color:
        emissive_hex = color
    else:
        emissive_hex = f"rgb({round(e.arousal * 60)},0,0)"

    rnd = _stable01(f"{req.label}|{req.base_label}|{e.arousal:.3f}")
    base_scale = size if size is not None else (0.9 + rnd * 0.4)
    scale = base_scale * (1 + e.arousal * 0.1)
    env = req.environment or {}
    fog = float(env.get("fog_density") or 0)
    if fog > 0.5:
        scale *= 0.92
    if float(env.get("sun_elevation_deg") or 0) < 5:
        scale *= 0.95

    wants_light = float(mat_hints.get("emissiveIntensity") or 0) > 0.4
    params: dict[str, Any] = {
        "emissive": emissive_hex,
        "emissiveIntensity": ei,
        "scale": scale,
    }
    if wants_light:
        params["pointLight"] = {
            "color": color or "#ffccaa",
            "intensity": 0.8 + e.arousal * 1.2,
            "distance": 8 + e.focus * 8,
        }
    else:
        params["pointLight"] = None
    # Same order as client: explicit matHints override base fields
    params.update(mat_hints)
    if color:
        params["color"] = color
    return params


MATERIAL_JSON_INSTRUCTION = """Return a single JSON object with this shape:
{
  "material_params": {
    "emissive": "<css color string, e.g. #aabbcc or rgb(r,g,b)>",
    "emissiveIntensity": <0..1 number>,
    "roughness": <0..1 or omit>,
    "metalness": <0..1 or omit>,
    "color": "<hex if user asked for tint, else omit>",
    "scale": <positive number — size of the building/prop; use emotion + environment to decide>,
    "pointLight": null | { "color": "#hex", "intensity": number, "distance": number }
  },
  "narration": "<one short line tying object + emotion + environment together, or empty>"
}
Use the user's emotion (arousal/valence/focus), mood label, and environment.sun_elevation / fog_density / mood_quadrant so materials and scale feel at home in that world (e.g. night + fog → dimmer emissive, slightly smaller silhouette; high arousal → bolder scale or glow).
Only valid JSON. Numbers must be JSON numbers. Output ONLY the JSON object, no markdown fences."""


def _anthropic_api_key() -> str | None:
    return os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")


def _parse_json_loose(text: str) -> dict[str, Any]:
    """Claude sometimes wraps JSON in ``` fences — strip and parse."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def llm_materials(req: PlaceRequest) -> tuple[dict[str, Any], str | None]:
    api_key = _anthropic_api_key()
    if not api_key:
        raise RuntimeError("no ANTHROPIC_API_KEY or CLAUDE_API_KEY")

    try:
        import anthropic
    except ImportError as err:
        raise RuntimeError("anthropic package not installed") from err

    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    client = anthropic.Anthropic(api_key=api_key)

    user_payload = {
        "object_label": req.label,
        "base_object": req.base_label,
        "parsed_hints": req.hints,
        "emotion": {
            "arousal": req.emotion.arousal,
            "valence": req.emotion.valence,
            "focus": req.emotion.focus,
        },
        "mood_word": req.mood,
        "environment": req.environment or {},
    }

    system = (
        "You help style 3D city props for a dreamlike scene. "
        + MATERIAL_JSON_INSTRUCTION
    )

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
    )
    raw_text = "".join(
        getattr(b, "text", str(b)) for b in message.content
    ).strip() or "{}"
    data = _parse_json_loose(raw_text)
    mp = data.get("material_params")
    if not isinstance(mp, dict):
        raise ValueError("missing material_params object")
    narration = data.get("narration")
    if isinstance(narration, str) and narration.strip():
        return mp, narration.strip()
    return mp, None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/place", response_model=PlaceResponse)
def place(req: PlaceRequest) -> PlaceResponse:
    """
    Main hook: emotion + object label in, material JSON out.
    Uses Claude when ANTHROPIC_API_KEY (or CLAUDE_API_KEY) is set; otherwise heuristic.
    """
    narration: str | None = None
    if _anthropic_api_key():
        try:
            material_params, narration = llm_materials(req)
            return PlaceResponse(material_params=material_params, narration=narration)
        except Exception as exc:  # noqa: BLE001
            print("[place] LLM failed, using heuristic:", exc)

    material_params = heuristic_materials(req)
    snippet = (req.base_label or req.label or "something").strip()[:48]
    narration = f"{req.mood or 'still'} — {snippet} settles into the grid."
    return PlaceResponse(material_params=material_params, narration=narration)


@app.post("/api/vision-classify", response_model=VisionClassifyResponse)
def vision_classify(req: VisionClassifyRequest) -> VisionClassifyResponse:
    """
    Text prompt → BFL image → looped MP4 → TRIBE video features → sklearn class.
    Returns ``place_key`` for ``index.html`` MODEL_MAP / PROCEDURAL_MAP.
    """
    api_key = (os.getenv("BFL_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="BFL_API_KEY not set (repo root or backend .env)",
        )
    try:
        from vision_place import run_vision_classify
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Vision stack import failed (install tribev2/torch in this venv): {exc}",
        ) from exc

    try:
        classified, place_key, confidence, probs, tribe_pooled = run_vision_classify(
            prompt=req.prompt.strip(),
            api_key=api_key,
            bfl_model=os.getenv("BFL_MODEL", "flux-2-klein-4b"),
            cache_folder=os.getenv("TRIBE_CACHE_FOLDER"),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    snippet = req.prompt.strip()[:72]
    narration = f"Vision: {classified} ({confidence:.0%}) — {snippet}"

    brain_b64 = ""
    try:
        from brain_render import render_tribe_brain_b64
        brain_b64 = render_tribe_brain_b64(tribe_pooled)
    except Exception as exc:  # noqa: BLE001
        print(f"[brain_render] skipped (nilearn may not be installed): {exc}")

    return VisionClassifyResponse(
        classified_label=classified,
        place_key=place_key,
        confidence=confidence,
        probabilities=probs,
        narration=narration,
        tribe_dim=int(tribe_pooled.shape[0]),
        brain_image_b64=brain_b64,
    )


@app.post("/api/vision-imagine", response_model=VisionImagineResponse)
def vision_imagine(req: VisionClassifyRequest) -> VisionImagineResponse:
    """Step 1: Generate BFL image only (fast). Returns base64 image."""
    import base64 as _b64

    api_key = (os.getenv("BFL_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="BFL_API_KEY not set")
    try:
        from vision_place import generate_bfl_image
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        img_bytes, mime = generate_bfl_image(
            prompt=req.prompt.strip(),
            api_key=api_key,
            bfl_model=os.getenv("BFL_MODEL", "flux-2-klein-4b"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return VisionImagineResponse(
        image_b64=_b64.b64encode(img_bytes).decode("ascii"),
        mime=mime or "image/jpeg",
    )


@app.post("/api/vision-classify-image", response_model=VisionClassifyResponse)
def vision_classify_image(req: VisionClassifyFromImageRequest) -> VisionClassifyResponse:
    """Step 2: Previously-generated BFL image → TRIBE → classifier + brain render."""
    import base64 as _b64

    try:
        from vision_place import classify_from_image_bytes
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        img_bytes = _b64.b64decode(req.image_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"bad base64: {exc}") from exc
    try:
        classified, place_key, confidence, probs, tribe_pooled = classify_from_image_bytes(
            img_bytes=img_bytes,
            mime=req.mime,
            cache_folder=os.getenv("TRIBE_CACHE_FOLDER"),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    snippet = req.prompt.strip()[:72] if req.prompt else ""
    narration = f"Vision: {classified} ({confidence:.0%})" + (f" — {snippet}" if snippet else "")

    brain_b64 = ""
    try:
        from brain_render import render_tribe_brain_b64
        brain_b64 = render_tribe_brain_b64(tribe_pooled)
    except Exception as exc:  # noqa: BLE001
        print(f"[brain_render] skipped: {exc}")

    return VisionClassifyResponse(
        classified_label=classified,
        place_key=place_key,
        confidence=confidence,
        probabilities=probs,
        narration=narration,
        tribe_dim=int(tribe_pooled.shape[0]),
        brain_image_b64=brain_b64,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Combined streaming pipeline: BFL → image (streamed) → TRIBE → classify
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/vision-pipeline")
async def vision_pipeline(req: VisionClassifyRequest):
    """BFL + TRIBE in one call, streamed as NDJSON.

    Eliminates the base64 round trip: the backend keeps the image bytes in
    memory between generation and classification instead of sending them to the
    browser and back.  Also uses shorter video params (1.5 s / 8 fps = 12 frames
    instead of 5 s / 24 fps = 120 frames) since TRIBE pools identical frames.
    """
    import base64 as _b64

    api_key = (os.getenv("BFL_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="BFL_API_KEY not set")
    try:
        from vision_place import classify_from_image_bytes, generate_bfl_image
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    bfl_model = os.getenv("BFL_MODEL", "flux-2-klein-4b")
    pipeline_w = int(os.getenv("BFL_PIPELINE_WIDTH", "1024"))
    pipeline_h = int(os.getenv("BFL_PIPELINE_HEIGHT", "1024"))

    async def _stream():
        loop = asyncio.get_running_loop()

        img_bytes, mime = await loop.run_in_executor(
            None,
            lambda: generate_bfl_image(
                prompt=req.prompt.strip(),
                api_key=api_key,
                bfl_model=bfl_model,
                width=pipeline_w,
                height=pipeline_h,
            ),
        )
        img_b64 = _b64.b64encode(img_bytes).decode("ascii")
        yield json.dumps({"stage": "image", "image_b64": img_b64, "mime": mime}) + "\n"

        try:
            result = await loop.run_in_executor(
                None,
                lambda: classify_from_image_bytes(
                    img_bytes=img_bytes,
                    mime=mime,
                    duration_sec=1.0,
                    fps=4,
                    fast=True,
                    cache_folder=os.getenv("TRIBE_CACHE_FOLDER"),
                ),
            )
            classified, place_key, confidence, probs, tribe_pooled = result

            snippet = req.prompt.strip()[:72]
            narration = (
                f"Vision: {classified} ({confidence:.0%})"
                + (f" — {snippet}" if snippet else "")
            )

            brain_b64 = ""
            try:
                from brain_render import render_tribe_brain_b64

                brain_b64 = await loop.run_in_executor(
                    None, lambda: render_tribe_brain_b64(tribe_pooled)
                )
            except Exception:
                pass

            yield json.dumps({
                "stage": "classified",
                "classified_label": classified,
                "place_key": place_key,
                "confidence": confidence,
                "probabilities": probs,
                "narration": narration,
                "tribe_dim": int(tribe_pooled.shape[0]),
                "brain_image_b64": brain_b64,
            }) + "\n"
        except Exception as exc:
            yield json.dumps({"stage": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


# ═══════════════════════════════════════════════════════════════════════════
# 3D model search via Perplexity Sonar
# ═══════════════════════════════════════════════════════════════════════════

SONAR_URL = "https://api.perplexity.ai/v1/sonar"

SONAR_SYSTEM = """You help find free downloadable 3D models (GLB or GLTF format) on the web.
When the user asks for a 3D model of something, search the web and return ONLY a JSON object with this exact shape:
{
  "results": [
    {
      "name": "Model name",
      "glb_url": "https://direct-download-link-to.glb",
      "source": "website name",
      "author": "creator name"
    }
  ]
}

Rules:
- Only include models with DIRECT download links to .glb or .gltf files (not page links).
- Look on sites like Sketchfab, Poly Pizza, KennyNL, Quaternius, poly.pizza, Smithsonian 3D, NASA 3D, market.pmnd.rs, etc.
- Prefer free, Creative Commons licensed models.
- Return 1-3 results maximum.
- If you cannot find any direct GLB/GLTF download links, return {"results": []}.
- Output ONLY valid JSON, no markdown fences, no explanation."""


def _perplexity_api_key() -> str | None:
    return os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY")


class ModelSearchResult(BaseModel):
    name: str
    glb_url: str = ""
    source: str = ""
    author: str = ""


class ModelSearchResponse(BaseModel):
    results: list[ModelSearchResult]
    query: str


@app.get("/api/search-model", response_model=ModelSearchResponse)
def search_model(q: str = Query(..., min_length=1)) -> ModelSearchResponse:
    """
    Search for downloadable 3D models using Perplexity Sonar.
    Returns results with direct GLB/GLTF download URLs.
    """
    api_key = _perplexity_api_key()
    if not api_key:
        print("[search-model] No PERPLEXITY_API_KEY set")
        return ModelSearchResponse(results=[], query=q)

    try:
        resp = httpx.post(
            SONAR_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": SONAR_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Find a free downloadable 3D model (GLB/GLTF) of: {q}",
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[search-model] Perplexity Sonar request failed: {exc}")
        return ModelSearchResponse(results=[], query=q)

    raw_text = ""
    try:
        raw_text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        print(f"[search-model] unexpected response shape: {data}")
        return ModelSearchResponse(results=[], query=q)

    parsed = _parse_json_loose(raw_text)
    results: list[ModelSearchResult] = []
    for item in parsed.get("results", []):
        if not isinstance(item, dict):
            continue
        glb_url = item.get("glb_url", "")
        if not glb_url:
            continue
        results.append(ModelSearchResult(
            name=item.get("name", q),
            glb_url=glb_url,
            source=item.get("source", ""),
            author=item.get("author", ""),
        ))

    print(f"[search-model] query={q!r} → {len(results)} result(s)")
    return ModelSearchResponse(results=results, query=q)
