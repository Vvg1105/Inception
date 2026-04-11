"""
Dream City Void — placement LLM API

The static site (`index.html`) POSTs here when you place an object. Payload carries
the typed label, parsed hints, and current emotion pad. This server returns
`material_params` for Three.js `applyMaterialParams`, plus optional narration.

Run:
  cd backend && pip install -r requirements.txt && uvicorn app:app --reload --port 8000

Env:
  ANTHROPIC_API_KEY (or CLAUDE_API_KEY) — Claude API key for `/api/place`
  ANTHROPIC_MODEL     — optional, default claude-3-5-sonnet-20241022
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
