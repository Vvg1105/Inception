"""
EEG live decoder — runs in Python, sends emotion + blink to the browser every 100ms.

**Two headsets (GTECH BCICore-8 + OpenBCI Cyton)** — use the dual decoder instead of this file:

  python backend/eeg_decode_dual.py

This single-board script is for **one** g.tec headset only. The browser’s default
Neural Symbiosis flow connects to the **dual** WebSocket (`eeg_decode_dual.py`).

Architecture:
  Python (this file)                        Browser (index.html)
  ─────────────────                         ───────────────────
  g.tec BCICore-8 → gpype pipeline          ← WebSocket ←
  ↓ EEGBuffer (eeg/eeg_stream.py)           reads window.__dreamEEG
  BlinkDetector (BLINK paper) + decode_emotion()    ↓
  → { emotion, blink, capture }  ──ws──►   mood pad + fog + light + colour
                                            blink → open place UI

Run:
  python backend/eeg_decode.py              # real hardware (g.tec BCICore-8)
  python backend/eeg_decode.py --mock       # fake sine-wave data for testing
  python backend/eeg_decode.py --port 8765  # custom port

  RUN "ls -l /dev/cu.* /dev/tty.*" IN TERMINAL TO FIND CORRECT SERIAL PORT FOR CYTON
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import os
import sys
import time
import threading
from typing import Any, Optional

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
]
MOCK_HOLD_SECONDS = 3   # hold each emotion for this long before switching


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
    "happy": (0.70, 0.85),
    "sad":   (0.25, 0.20),
}


EMOTION_INTERVAL_S  = 0.5   # call decode_emotion() every 500 ms
EMOTION_HISTORY_N   = 6     # average last 6 results = 3 s


class LiveDecoder(EEGDecoder):
    """
    Real EEG decoder using the gpype pipeline and eeg/eeg_stream.py.

    Pipeline: BCICore8 → Bandpass(0.5–45) → Notch 50 → Notch 60 → EEGBuffer
    Blink:    BlinkDetector (BLINK paper, eeg/blink_detector.py) when enabled —
              feed from EEGBuffer; falls back to amplitude double-blink until calibrated.
    Emotion:  decode_emotion() — called every 500 ms, averaged over last 3 s
    """

    def __init__(
        self,
        *,
        blink_profile: Optional[str] = None,
        blink_frontal_ch: int = 0,
        use_blink_paper: bool = True,
    ) -> None:
        self._pipeline = None
        self._last_arousal = 0.5
        self._last_valence = 0.5
        self._last_focus   = 0.5
        self._last_label   = ""
        # History stores raw per-class probability dicts for smoothing.
        self._prob_history: collections.deque = collections.deque(maxlen=EMOTION_HISTORY_N)
        self._next_emotion_t = 0.0  # monotonic time for next decode_emotion() call
        self._use_blink_paper = use_blink_paper
        self._blink_det: Any = None
        self._blink_frontal_ch = blink_frontal_ch
        self._blink_profile_path = blink_profile
        self._last_blink_feed_t: Optional[float] = None

    def setup(self) -> None:
        import torch
        import gpype as gp
        from eeg.eegnet import EEGNet, EmotionMLP
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
        model_cls = EmotionMLP if cfg.get("model", "eegnet") == "mlp" else EEGNet
        model = model_cls(
            n_channels=cfg["n_channels"],
            n_timepoints=cfg["n_timepoints"],
            n_classes=cfg["n_classes"],
        ).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        load_emotion_model(model, cfg)

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                p.start()
                break
            except (ValueError, RuntimeError, OSError) as exc:
                print(f"  BLE connect attempt {attempt}/{max_attempts} failed: {exc}")
                if attempt == max_attempts:
                    raise RuntimeError(
                        "Could not connect to BCI Core-8 after "
                        f"{max_attempts} attempts. Power-cycle the headset "
                        "and toggle Bluetooth, then retry."
                    ) from exc
                time.sleep(3)
                # Rebuild the pipeline — a failed start leaves nodes in a bad state.
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

        self._pipeline = p
        print("  gpype pipeline started")

        if self._use_blink_paper:
            from eeg.blink_detector import BlinkDetector
            from eeg.eeg_stream import FS as _FS

            self._blink_det = BlinkDetector(
                fs=int(_FS),
                frontal_ch=self._blink_frontal_ch,
                profile=self._blink_profile_path,
            )

    def read_raw(self) -> Any:
        return None

    def decode(self, raw: Any) -> dict[str, Any]:
        from eeg.eeg_stream import FS, _buffer, check_blink_state_old, decode_emotion

        if self._blink_det is not None:
            now = time.monotonic()
            if self._last_blink_feed_t is None:
                self._last_blink_feed_t = now
            else:
                dt = now - self._last_blink_feed_t
                n_new = min(max(1, int(dt * FS)), 500)
                if _buffer is not None:
                    w = _buffer.latest(n_new)
                    if w is not None:
                        self._blink_det.feed(w)
                self._last_blink_feed_t = now
            if self._blink_det.ready:
                blink = self._blink_det.check()
            else:
                blink = check_blink_state_old()
        else:
            blink = check_blink_state_old()

        # Only call decode_emotion() every 500 ms to avoid over-sampling.
        now = time.monotonic()
        if now >= self._next_emotion_t:
            self._next_emotion_t = now + EMOTION_INTERVAL_S
            result = decode_emotion()
            if result is not None:
                *_, probs = result
                self._prob_history.append(probs)

        # Smooth by averaging raw per-class probabilities over the history window.
        # This is more stable than majority-voting labels — a slight consistent
        # lean in probability space accumulates rather than being rounded away.
        if self._prob_history:
            emotions = list(next(iter(self._prob_history)).keys())
            n = len(self._prob_history)
            avg_probs = {e: sum(p[e] for p in self._prob_history) / n
                         for e in emotions}
            best = max(avg_probs, key=avg_probs.__getitem__)
            confidence = avg_probs[best]
            av = EMOTION_AV.get(best, (0.5, 0.5))
            self._last_label   = best
            self._last_focus   = confidence
            self._last_arousal = av[0] * confidence + 0.5 * (1 - confidence)
            self._last_valence = av[1] * confidence + 0.5 * (1 - confidence)

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
    Build JSON to send. If blink just started (False → True), add capture flag
    so the browser opens the place UI immediately — no waiting for blink to end.
    """
    blink_now = bool(state.get("blink"))
    capture = not prev_blink and blink_now  # rising edge: fire on first detection

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

    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    # Register SIGINT/SIGTERM so Ctrl-C resolves the future instead of raising
    # into the middle of the event loop (which leaves the port open).
    import signal as _signal
    for sig in (_signal.SIGINT, _signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set_result, None)

    async with websockets.serve(ws_handler, "127.0.0.1", port):
        decode_task = asyncio.ensure_future(decode_loop(decoder))
        try:
            # Block until Ctrl-C / SIGTERM fires the stop future.
            await asyncio.wait(
                [decode_task, asyncio.ensure_future(stop)],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            decode_task.cancel()
            try:
                await decode_task
            except asyncio.CancelledError:
                pass
            decoder.cleanup()
            print("\nStopped.")


def free_port(port: int) -> None:
    """Kill any process already bound to `port` so we can reuse it."""
    import signal
    import subprocess
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True, text=True
    )
    pids = [p for p in result.stdout.strip().split() if p]
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed stale process {pid} on port {port}")
        except (ProcessLookupError, ValueError):
            pass
    if pids:
        time.sleep(1.0)   # give the OS time to release the port after SIGKILL


def _default_blink_npz(name: str) -> Optional[str]:
    p = os.path.join(PROJECT_ROOT, "eeg", "models", name)
    return p if os.path.isfile(p) else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG live decode → WebSocket")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use fake sine-wave decoder (no hardware needed)",
    )
    parser.add_argument(
        "--blink-profile",
        default="",
        help="Path to blink_*.npz from tools/calibrate_blink.py; "
             "if omitted, uses eeg/models/blink_user1.npz when present",
    )
    parser.add_argument(
        "--no-blink-paper",
        action="store_true",
        help="Disable BLINK paper detector; use amplitude double-blink only",
    )
    parser.add_argument(
        "--blink-ch",
        type=int,
        default=0,
        help="Frontal EEG column index for BlinkDetector (default: 0 = Fp2 on BCICore-8)",
    )
    args = parser.parse_args()

    free_port(args.port)

    decoder: EEGDecoder
    if args.mock:
        decoder = MockDecoder()
    else:
        blink_path = args.blink_profile.strip() or _default_blink_npz("blink_user1.npz")
        decoder = LiveDecoder(
            blink_profile=blink_path,
            blink_frontal_ch=args.blink_ch,
            use_blink_paper=not args.no_blink_paper,
        )

    asyncio.run(main(decoder, args.port))
