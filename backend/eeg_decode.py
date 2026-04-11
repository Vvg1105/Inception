"""
EEG live decoder — runs in Python, sends emotion + blink to the browser every 100ms.

Architecture:
  Python (this file)                        Browser (index.html)
  ─────────────────                         ───────────────────
  g.tec BCICore-8 → gpype pipeline          ← WebSocket ←
  ↓ EEGBuffer (eeg/eeg_stream.py)           reads window.__dreamEEG
  check_blink_state() + decode_emotion()    ↓
  → { emotion, blink, capture }  ──ws──►   mood pad + fog + light + colour
                                            blink → open place UI

Run:
  python backend/eeg_decode.py              # real hardware (g.tec BCICore-8)
  python backend/eeg_decode.py --mock       # fake sine-wave data for testing
  python backend/eeg_decode.py --port 8765  # custom port
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
import threading
from typing import Any

try:
    import websockets
except ImportError:
    raise SystemExit("pip install websockets") from None

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# Decoder base + mock
# ═══════════════════════════════════════════════════════════════════════════


class EEGDecoder:
    """
    Base class. Subclass and override `setup()` / `read_raw()` / `decode()`.

    decode() must return:
      {
        "arousal": float 0..1,
        "valence": float 0..1,
        "focus":   float 0..1,
        "blink":   bool,        # true while eyes closed
      }
    """

    def setup(self) -> None: ...
    def read_raw(self) -> Any: return None
    def decode(self, raw: Any) -> dict[str, Any]:
        return {"arousal": 0.5, "valence": 0.5, "focus": 0.5, "blink": False}
    def cleanup(self) -> None: ...


MOCK_EMOTIONS = [
    {"label": "happy", "arousal": 0.70, "valence": 0.85, "focus": 0.75},
    {"label": "sad",   "arousal": 0.25, "valence": 0.20, "focus": 0.40},
    {"label": "angry", "arousal": 0.85, "valence": 0.15, "focus": 0.60},
    {"label": "fear",  "arousal": 0.80, "valence": 0.20, "focus": 0.35},
]
MOCK_HOLD_SECONDS = 10  # hold each emotion for this long before switching


class MockDecoder(EEGDecoder):
    """Fake decoder that cycles through emotions for testing weather effects."""

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self._last_blink = 0.0
        self._blink_on = False
        self._emo_idx = 0
        self._emo_switch_at = self._t0 + MOCK_HOLD_SECONDS

    def read_raw(self) -> float:
        return time.monotonic() - self._t0

    def decode(self, raw: float) -> dict[str, Any]:
        now = time.monotonic()

        if now >= self._emo_switch_at:
            self._emo_idx = (self._emo_idx + 1) % len(MOCK_EMOTIONS)
            self._emo_switch_at = now + MOCK_HOLD_SECONDS

        emo = MOCK_EMOTIONS[self._emo_idx]

        if not self._blink_on and (now - self._last_blink) > 8.0:
            self._blink_on = True
            self._last_blink = now
        elif self._blink_on and (now - self._last_blink) > 0.35:
            self._blink_on = False

        return {
            "arousal": emo["arousal"],
            "valence": emo["valence"],
            "focus":   emo["focus"],
            "label":   emo["label"],
            "blink":   self._blink_on,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Real decoder — g.tec BCICore-8 via gpype + EEGNet
# ═══════════════════════════════════════════════════════════════════════════

# Emotion label → (arousal, valence) in the circumplex model.
# Focus is derived from confidence: high confidence → high focus.
EMOTION_AV = {
    "angry": (0.85, 0.15),
    "fear":  (0.80, 0.20),
    "happy": (0.70, 0.85),
    "sad":   (0.25, 0.20),
}


class LiveDecoder(EEGDecoder):
    """
    Real EEG decoder using the gpype pipeline and eeg/eeg_stream.py.

    Pipeline: BCICore8 → Bandpass(0.5–45) → Notch 50 → Notch 60 → EEGBuffer
    Blink:    check_blink_state() — threshold on mean absolute amplitude
    Emotion:  decode_emotion() — EEGNet inference on the last 100 ms
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._last_arousal = 0.5
        self._last_valence = 0.5
        self._last_focus   = 0.5
        self._last_label   = "happy"

    def setup(self) -> None:
        import torch
        import gpype as gp
        from eeg.eegnet import EEGNet
        from eeg.eeg_stream import EEGBuffer, load_emotion_model

        p        = gp.Pipeline()
        source   = gp.BCICore8()
        bandpass = gp.Bandpass(f_lo=0.5, f_hi=45)
        notch50  = gp.Bandstop(f_lo=48,  f_hi=52)
        notch60  = gp.Bandstop(f_lo=58,  f_hi=62)
        buf      = EEGBuffer()

        p.connect(source,   bandpass)
        p.connect(bandpass, notch50)
        p.connect(notch50,  notch60)
        p.connect(notch60,  buf)

        config_path = os.path.join(PROJECT_ROOT, "eeg", "models", "eegnet_config.json")
        weights_path = os.path.join(PROJECT_ROOT, "eeg", "models", "eegnet_emotion.pt")

        with open(config_path) as f:
            cfg = json.load(f)

        device = torch.device(
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = EEGNet(
            n_channels=cfg["n_channels"],
            n_timepoints=cfg["n_timepoints"],
            n_classes=cfg["n_classes"],
        ).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        load_emotion_model(model, cfg)

        p.start()
        self._pipeline = p
        print("  gpype pipeline started")

    def read_raw(self) -> Any:
        return None

    def decode(self, raw: Any) -> dict[str, Any]:
        from eeg.eeg_stream import check_blink_state, decode_emotion

        blink = check_blink_state()

        result = decode_emotion()
        if result is not None:
            label, confidence = result
            av = EMOTION_AV.get(label, (0.5, 0.5))
            self._last_arousal = av[0] * confidence + 0.5 * (1 - confidence)
            self._last_valence = av[1] * confidence + 0.5 * (1 - confidence)
            self._last_focus   = confidence
            self._last_label   = label

        return {
            "arousal": self._last_arousal,
            "valence": self._last_valence,
            "focus":   self._last_focus,
            "label":   self._last_label,
            "blink":   blink,
        }

    def cleanup(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


# ═══════════════════════════════════════════════════════════════════════════
# WebSocket server — pushes decoded state to browser at 10 Hz
# ═══════════════════════════════════════════════════════════════════════════

TICK_MS = 100
clients: set[Any] = set()


def build_message(state: dict[str, Any], prev_blink: bool) -> tuple[str, bool]:
    """
    Build JSON to send. If blink just ended (True → False), add capture flag
    so the browser opens the place UI.
    """
    blink_now = bool(state.get("blink"))
    capture = prev_blink and not blink_now

    msg: dict[str, Any] = {
        "present": True,
        "blink": blink_now,
        "emotion": {
            "arousal": round(state["arousal"], 4),
            "valence": round(state["valence"], 4),
            "focus": round(state["focus"], 4),
            "label": state.get("label", ""),
        },
    }
    if capture:
        msg["capture"] = True

    return json.dumps(msg, separators=(",", ":")), blink_now


async def decode_loop(decoder: EEGDecoder) -> None:
    prev_blink = False
    while True:
        raw = decoder.read_raw()
        state = decoder.decode(raw)
        msg, prev_blink = build_message(state, prev_blink)

        dead = set()
        for ws in clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        clients.difference_update(dead)

        a = state["arousal"]
        v = state["valence"]
        f = state["focus"]
        lbl = state.get("label", "")
        b = "BLINK" if state["blink"] else "     "
        print(
            f"\r  {lbl:6s}  A={a:.2f}  V={v:.2f}  F={f:.2f}  {b}  clients={len(clients)}",
            end="",
            flush=True,
        )

        await asyncio.sleep(TICK_MS / 1000)


async def ws_handler(websocket: Any) -> None:
    clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        clients.discard(websocket)


async def main(decoder: EEGDecoder, port: int) -> None:
    decoder.setup()
    print(f"EEG decode → ws://127.0.0.1:{port}  (every {TICK_MS}ms)")
    print("Connect in World panel → EEG → connect")
    print()

    async with websockets.serve(ws_handler, "127.0.0.1", port):
        try:
            await decode_loop(decoder)
        finally:
            decoder.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG live decode → WebSocket")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use fake sine-wave decoder (no hardware needed)",
    )
    args = parser.parse_args()

    decoder: EEGDecoder
    if args.mock:
        decoder = MockDecoder()
    else:
        decoder = LiveDecoder()

    asyncio.run(main(decoder, args.port))
