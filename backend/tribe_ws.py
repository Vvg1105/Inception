"""
TRIBE v2 fMRI decoder — WebSocket server for brain-decoded object placement.

Takes fMRI feature vectors via WebSocket, runs the trained object + size decoders,
and sends predicted object class + size back to the browser for auto-placement.

Run:
  python backend/tribe_ws.py --model <path_to_trained.npz>       # with trained model
  python backend/tribe_ws.py --mock                               # cycles through classes
  python backend/tribe_ws.py --port 8766                          # custom port

Protocol (browser → server):
  { "fmri_vector": [float, ...] }    # ~20,000 dim TRIBE v2 encoding
  OR
  { "trigger": true }                # mock mode: trigger next prediction

Protocol (server → browser):
  {
    "tribe": true,
    "object_class": "Building",      # one of 8 OBJECT_CLASSES
    "object_idx": 0,                 # class index 0-7
    "size": "large",                 # "small" or "large"
    "confidence": 0.85,              # softmax probability of predicted class
    "probabilities": { ... }         # per-class probabilities
  }
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

try:
    import websockets
except ImportError:
    raise SystemExit("pip install websockets") from None

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tribe_decoding.model import (
    build_object_decoder,
    build_size_decoder,
    OBJECT_CLASSES,
    SIZE_CLASSES,
)

# ═══════════════════════════════════════════════════════════════════════════
# Decoder wrapper
# ═══════════════════════════════════════════════════════════════════════════

class TRIBEDecoder:
    """Wraps trained object + size pipelines for inference."""

    def __init__(self, data_path: str | None = None):
        self.obj_pipeline = build_object_decoder()
        self.size_pipeline = build_size_decoder()
        self._trained = False

        if data_path:
            self._train(data_path)

    def _train(self, path: str):
        data = np.load(path)
        X = data["X"]
        y_object = data["y_object"]
        y_size = data["y_size"]
        print(f"[TRIBE] Training on {X.shape[0]} trials, {X.shape[1]} features")
        self.obj_pipeline.fit(X, y_object)
        self.size_pipeline.fit(X, y_size)
        self._trained = True
        print("[TRIBE] Models trained and ready")

    def predict(self, fmri_vector: np.ndarray) -> dict[str, Any]:
        if not self._trained:
            return {"error": "Models not trained"}

        X = fmri_vector.reshape(1, -1)

        obj_idx = int(self.obj_pipeline.predict(X)[0])
        obj_probs = self.obj_pipeline.predict_proba(X)[0]
        obj_class = OBJECT_CLASSES[obj_idx]
        obj_confidence = float(obj_probs[obj_idx])

        size_idx = int(self.size_pipeline.predict(X)[0])
        size_label = SIZE_CLASSES[size_idx]

        return {
            "tribe": True,
            "object_class": obj_class,
            "object_idx": obj_idx,
            "size": size_label,
            "confidence": round(obj_confidence, 4),
            "probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(OBJECT_CLASSES, obj_probs)
            },
        }


class MockTRIBEDecoder:
    """Cycles through object classes for testing without fMRI data."""

    def __init__(self):
        self._idx = 0
        self._last_trigger = 0.0

    def predict(self, fmri_vector: np.ndarray | None = None) -> dict[str, Any]:
        obj_class = OBJECT_CLASSES[self._idx]
        size = "large" if self._idx % 2 == 0 else "small"

        probs = {cls: 0.02 for cls in OBJECT_CLASSES}
        probs[obj_class] = 0.86

        result = {
            "tribe": True,
            "object_class": obj_class,
            "object_idx": self._idx,
            "size": size,
            "confidence": 0.86,
            "probabilities": probs,
        }

        self._idx = (self._idx + 1) % len(OBJECT_CLASSES)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# WebSocket server
# ═══════════════════════════════════════════════════════════════════════════

clients: set[Any] = set()
decoder: TRIBEDecoder | MockTRIBEDecoder | None = None


async def ws_handler(websocket: Any) -> None:
    clients.add(websocket)
    print(f"[TRIBE] client connected ({len(clients)} total)")
    try:
        async for raw_msg in websocket:
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue

            if "fmri_vector" in msg:
                vec = np.array(msg["fmri_vector"], dtype=np.float64)
                result = decoder.predict(vec)
            elif msg.get("trigger"):
                result = decoder.predict(None)
            else:
                continue

            out = json.dumps(result, separators=(",", ":"))
            print(f"[TRIBE] → {result['object_class']} ({result['size']}, "
                  f"conf={result['confidence']})")
            await websocket.send(out)
    finally:
        clients.discard(websocket)
        print(f"[TRIBE] client disconnected ({len(clients)} total)")


async def mock_auto_trigger(interval: float = 6.0) -> None:
    """In mock mode, auto-trigger a prediction every N seconds."""
    while True:
        await asyncio.sleep(interval)
        if not clients:
            continue
        result = decoder.predict(None)
        out = json.dumps(result, separators=(",", ":"))
        print(f"[TRIBE] auto → {result['object_class']} ({result['size']})")
        dead = set()
        for ws in clients:
            try:
                await ws.send(out)
            except Exception:
                dead.add(ws)
        clients.difference_update(dead)


async def main(port: int, mock: bool) -> None:
    global decoder
    if mock:
        decoder = MockTRIBEDecoder()
        print("[TRIBE] Mock mode — cycling through object classes")
    else:
        print("[TRIBE] ERROR: --model <path.npz> required for real mode")
        return

    print(f"[TRIBE] WebSocket → ws://127.0.0.1:{port}")
    print(f"[TRIBE] Object classes: {', '.join(OBJECT_CLASSES)}")
    print()

    async with websockets.serve(ws_handler, "127.0.0.1", port):
        if mock:
            await mock_auto_trigger(6.0)
        else:
            await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRIBE fMRI decoder → WebSocket")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--mock", action="store_true",
                        help="Use mock decoder that cycles through classes")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained .npz data file")
    args = parser.parse_args()

    if args.model and not args.mock:
        dec = TRIBEDecoder(args.model)
        async def run():
            global decoder
            decoder = dec
            print(f"[TRIBE] WebSocket → ws://127.0.0.1:{args.port}")
            async with websockets.serve(ws_handler, "127.0.0.1", args.port):
                await asyncio.Future()
        asyncio.run(run())
    else:
        asyncio.run(main(args.port, mock=True))
