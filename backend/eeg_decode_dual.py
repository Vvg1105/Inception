"""
eeg_decode_dual.py — Dual-user EEG decoder for the Neural Symbiosis demo.

**This is the EEG entry point when both headsets are used** (with `index.html`
auto-connecting the dual WebSocket on port 8765). For a single g.tec board only,
use `eeg_decode.py` instead.

User 1 : GTECH BCICore-8 (Bluetooth via gpype)   → existing LiveDecoder
User 2 : OpenBCI Cyton  (USB dongle via BrainFlow) → new CytonDecoder

Architecture:
  Thread-1 : decoder for User 1 → writes shared state['user1']
  Thread-2 : decoder for User 2 → writes shared state['user2']
  Thread-3 : stdin listener — press Enter to toggle active_user
  Main     : asyncio WebSocket server → broadcasts dual JSON at 10 Hz

WebSocket JSON payload (sent every 100 ms):
  {
    "user1"       : { "present": bool, "blink": bool,
                      "emotion": { "arousal", "valence", "focus", "label" } },
    "user2"       : { ...same... },
    "active_user" : 1 or 2,            // who has creation permission
    "capture"     : true,              // only on blink rising-edge of active user
    "symbiosis"   : {
        "arousal" : 0..1,              // combined average
        "valence" : 0..1,
        "score"   : 0..1,              // 1 = fully aligned, 0 = divergent
        "label"   : "resonant"|"entangled"|"drifting"|"divergent"
    }
  }

Run:
  python backend/eeg_decode_dual.py                              # both real
  python backend/eeg_decode_dual.py --mock                       # both fake
  python backend/eeg_decode_dual.py --mock-user1                 # user1 fake
  python backend/eeg_decode_dual.py --cyton-port /dev/cu.usbserial-XXXX
  python backend/eeg_decode_dual.py --port 8765

  After running tools/calibrate_blink.py for each headset, profiles load from
  eeg/models/blink_user1.npz (GTECH) and blink_user2.npz (Cyton) unless overridden.

Press Enter in the terminal at any time to hand creation permission to the
other user.  The active user is shown in the terminal readout and pushed to
the browser in every WebSocket message.
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
import traceback
from typing import Any, Optional

try:
    import websockets
except ImportError:
    raise SystemExit("pip install websockets") from None

try:
    import numpy as np
except ImportError:
    raise SystemExit("pip install numpy") from None

# Ensure project root is on sys.path so we can import eeg.* and backend.*
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Emotion circumplex mapping (shared by both decoders) ──────────────────────
EMOTION_AV = {
    "happy": (0.70, 0.85),
    "sad":   (0.25, 0.20),
}

EMOTION_INTERVAL_S = 0.5
EMOTION_HISTORY_N  = 6


# ═══════════════════════════════════════════════════════════════════════════════
# Shared state — written by decoder threads, read by the WS broadcast loop
# ═══════════════════════════════════════════════════════════════════════════════

# Tracks which users have successfully connected (for startup confirmation)
_connected: dict[int, bool] = {1: False, 2: False}
_connected_lock = threading.Lock()
_both_connected = threading.Event()   # set when both boards are up

def _mark_connected(user: int) -> None:
    with _connected_lock:
        _connected[user] = True
        u1_ok = _connected[1]
        u2_ok = _connected[2]
    if u1_ok and u2_ok:
        print("\n  ✓  both boards connected — streaming")
        print("  Press  Enter ↵  to pass creation to the other user\n", flush=True)
        _both_connected.set()

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "user1": {
        "arousal": 0.5, "valence": 0.5, "focus": 0.5,
        "label": "", "blink": False, "present": False,
        "ch_amplitudes": [0.5] * 8,
    },
    "user2": {
        "arousal": 0.5, "valence": 0.5, "focus": 0.5,
        "label": "", "blink": False, "present": False,
        "ch_amplitudes": [0.5] * 8,
    },
    "active_user": 1,
}


def _snapshot() -> dict[str, Any]:
    with _state_lock:
        return {
            "user1":       dict(_state["user1"]),
            "user2":       dict(_state["user2"]),
            "active_user": _state["active_user"],
        }


def _set_user(user: int, data: dict[str, Any]) -> None:
    with _state_lock:
        _state[f"user{user}"].update(data)
        _state[f"user{user}"]["present"] = True


def _toggle_active() -> int:
    with _state_lock:
        _state["active_user"] = 2 if _state["active_user"] == 1 else 1
        return _state["active_user"]


# ═══════════════════════════════════════════════════════════════════════════════
# Mock decoders (--mock / --mock-user1)
# ═══════════════════════════════════════════════════════════════════════════════

_MOCK_U1 = [
    {"label": "happy", "arousal": 0.72, "valence": 0.85, "focus": 0.78},
    {"label": "sad",   "arousal": 0.22, "valence": 0.18, "focus": 0.38},
]
_MOCK_U2 = [
    {"label": "sad",   "arousal": 0.30, "valence": 0.22, "focus": 0.42},
    {"label": "happy", "arousal": 0.65, "valence": 0.80, "focus": 0.70},
]
_MOCK_HOLD_S = 4

# Mock per-channel amplitude baselines by emotion and user
# User1 ch order: [FP2, F7, FC5, FP1, P7, T7, T8, O2]
# User2 ch order: [C3, C4, CZ, T3, T4, F3, FZ, F4]
_MOCK_CH_BASE: dict[int, dict[str, list[float]]] = {
    1: {
        "happy": [0.80, 0.55, 0.55, 0.75, 0.18, 0.28, 0.35, 0.22],
        "sad":   [0.22, 0.18, 0.18, 0.20, 0.72, 0.48, 0.50, 0.82],
    },
    2: {
        "happy": [0.40, 0.48, 0.62, 0.28, 0.32, 0.68, 0.78, 0.68],
        "sad":   [0.62, 0.52, 0.42, 0.70, 0.68, 0.25, 0.18, 0.28],
    },
}


class _MockUser:
    def __init__(self, offset: int, emotions: list, user: int = 1) -> None:
        self._emotions = emotions
        self._emo_i = offset % len(emotions)
        self._switch_at = time.monotonic() + _MOCK_HOLD_S
        self._blink_on = False
        self._last_blink = time.monotonic() + offset * 5   # stagger
        self._user = user

    def tick(self) -> dict[str, Any]:
        now = time.monotonic()
        if now >= self._switch_at:
            self._emo_i = (self._emo_i + 1) % len(self._emotions)
            self._switch_at = now + _MOCK_HOLD_S
        emo = self._emotions[self._emo_i]
        if not self._blink_on and (now - self._last_blink) > 10.0:
            self._blink_on = True
            self._last_blink = now
        elif self._blink_on and (now - self._last_blink) > 0.35:
            self._blink_on = False
        # Synthetic per-channel amplitudes with sinusoidal variation
        label = emo.get("label", "happy")
        base = _MOCK_CH_BASE.get(self._user, _MOCK_CH_BASE[1]).get(label, [0.5]*8)
        ch_amplitudes = [
            max(0.0, min(1.0, base[i] + 0.08 * math.sin(now * (1.0 + i * 0.4) + i)))
            for i in range(8)
        ]
        return {**emo, "blink": self._blink_on, "ch_amplitudes": ch_amplitudes}


def _run_mock(user: int, mock: _MockUser, interval: float = 0.1) -> None:
    _mark_connected(user)
    while True:
        _set_user(user, mock.tick())
        time.sleep(interval)


# ═══════════════════════════════════════════════════════════════════════════════
# Blink profile resolution (BLINK paper, eeg/blink_detector.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_blink_npz(cli: str, default_name: str) -> Optional[str]:
    """Explicit path from CLI wins; else use eeg/models/{default_name} if it exists."""
    if cli.strip():
        return cli.strip()
    p = os.path.join(PROJECT_ROOT, "eeg", "models", default_name)
    return p if os.path.isfile(p) else None


# ═══════════════════════════════════════════════════════════════════════════════
# Real User 1 — GTECH BCICore-8 via existing LiveDecoder
# ═══════════════════════════════════════════════════════════════════════════════

def _run_live_user1(
    interval: float,
    blink_profile: Optional[str],
    blink_ch: int,
    no_blink_paper: bool,
) -> None:
    from backend.eeg_decode import LiveDecoder
    import eeg.eeg_stream as _es
    print("  [ ] User 1  GTECH BCICore-8   connecting via Bluetooth …", flush=True)
    dec = LiveDecoder(
        blink_profile=blink_profile,
        blink_frontal_ch=blink_ch,
        use_blink_paper=not no_blink_paper,
    )
    dec.setup()
    print("  [✓] User 1  GTECH BCICore-8   connected", flush=True)
    _mark_connected(1)
    while True:
        state = dec.decode(None)
        # Per-channel RMS amplitudes from the rolling buffer (~50 samples = 200ms)
        try:
            buf = _es._buffer
            if buf is not None:
                win = buf.latest(50)
                if win is not None:
                    rms = np.sqrt(np.mean(np.asarray(win, dtype=np.float32)**2, axis=0))
                    state["ch_amplitudes"] = np.clip(rms[:8] / 50.0, 0.0, 1.0).tolist()
        except Exception:
            pass
        _set_user(1, state)
        time.sleep(interval)


# ═══════════════════════════════════════════════════════════════════════════════
# Real User 2 — OpenBCI Cyton via BrainFlow + CytonDecoder
# ═══════════════════════════════════════════════════════════════════════════════

def _run_cyton_user2(
    serial_port: str,
    interval: float,
    blink_profile: Optional[str],
    blink_ch: int,
    no_blink_paper: bool,
) -> None:
    """
    Streams from the Cyton board and classifies emotion via BrainFlow's built-in
    RESTFULNESS classifier (no external model weights required).
    """
    from eeg.cyton_stream import CytonDecoder

    port_label = serial_port or "auto-detect"
    print(f"  [ ] User 2  OpenBCI Cyton      connecting via USB ({port_label}) …", flush=True)
    try:
        dec = CytonDecoder(
            serial_port=serial_port,
            blink_profile=blink_profile,
            frontal_ch=blink_ch,
            use_blink_paper=not no_blink_paper,
        )
        dec.start()
    except Exception as exc:
        print(f"  [!] User 2  OpenBCI Cyton failed to start: {exc}", flush=True)
        traceback.print_exc()
        return

    print("  [✓] User 2  OpenBCI Cyton      connected", flush=True)
    _mark_connected(2)

    try:
        while True:
            try:
                state = dec.decode()
                # Per-channel RMS amplitudes from the Cyton rolling buffer
                win = dec._buf.latest(50)
                if win is not None:
                    rms = np.sqrt(np.mean(np.asarray(win, dtype=np.float32)**2, axis=0))
                    state["ch_amplitudes"] = np.clip(rms[:8] / 50.0, 0.0, 1.0).tolist()
                _set_user(2, state)
            except Exception as exc:
                print(f"  [!] User 2  OpenBCI Cyton decode loop error: {exc}", flush=True)
                traceback.print_exc()
                time.sleep(1.0)
            time.sleep(interval)
    finally:
        dec.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# Stdin permission switch — press Enter to hand creation to the other user
# ═══════════════════════════════════════════════════════════════════════════════

def _stdin_thread() -> None:
    _both_connected.wait()   # don't accept input until both boards are live
    while True:
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            break
        new = _toggle_active()
        other = 2 if new == 1 else 1
        print(f"\n  ↳  creation passed to  USER {new}  (USER {other} now watching)\n",
              flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Neural symbiosis computation
# ═══════════════════════════════════════════════════════════════════════════════

def _symbiosis(s1: dict, s2: dict) -> dict[str, Any]:
    """
    Alignment score + label. Previously almost everything read as "resonant" because
    Manhattan distance stays small when both users sit in the same quadrant.

    Uses Euclidean distance in A/V space (normalized), optional penalty when
    classifiers disagree (happy vs sad/angry), and wider tier thresholds.
    """
    a1, v1 = s1["arousal"], s1["valence"]
    a2, v2 = s2["arousal"], s2["valence"]
    ca = (a1 + a2) / 2
    cv = (v1 + v2) / 2

    # Max distance across unit square diagonal = sqrt(2); normalize to ~[0, 1]
    d_euc = math.hypot(a1 - a2, v1 - v2) / math.sqrt(2.0)

    # Keep happy/sad from decoders: opposing labels add virtual distance
    t1 = (s1.get("label") or "").strip().lower()
    t2 = (s2.get("label") or "").strip().lower()
    pos = {"happy"}
    neg = {"sad", "angry"}
    lab_penalty = 0.0
    if t1 and t2:
        if (t1 in pos and t2 in neg) or (t1 in neg and t2 in pos):
            lab_penalty = 0.18

    divergence = min(1.0, d_euc + lab_penalty)
    score = round(max(0.0, min(1.0, 1.0 - divergence)), 4)

    # Wider bands so "resonant" is rare (true lock-step + matching labels)
    if score >= 0.88:
        label = "resonant"
    elif score >= 0.68:
        label = "entangled"
    elif score >= 0.42:
        label = "drifting"
    else:
        label = "divergent"
    return {"arousal": round(ca, 4), "valence": round(cv, 4),
            "score": score, "label": label}


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket broadcast loop
# ═══════════════════════════════════════════════════════════════════════════════

TICK_MS = 100
_clients: set[Any] = set()


def _build_msg(
    snap: dict[str, Any],
    prev_b1: bool,
    prev_b2: bool,
) -> tuple[str, bool, bool]:
    s1, s2 = snap["user1"], snap["user2"]
    active = snap["active_user"]
    b1, b2 = bool(s1.get("blink")), bool(s2.get("blink"))
    capture = (
        (active == 1 and not prev_b1 and b1) or
        (active == 2 and not prev_b2 and b2)
    )
    sym = _symbiosis(s1, s2)
    def _round_amps(amps: list) -> list:
        return [round(float(a), 3) for a in (amps or [])]

    msg: dict[str, Any] = {
        "user1": {
            "present": s1["present"],
            "blink":   b1,
            "emotion": {
                "arousal": round(s1["arousal"], 4),
                "valence": round(s1["valence"], 4),
                "focus":   round(s1["focus"],   4),
                "label":   s1.get("label", ""),
            },
            "ch_amplitudes": _round_amps(s1.get("ch_amplitudes", [])),
        },
        "user2": {
            "present": s2["present"],
            "blink":   b2,
            "emotion": {
                "arousal": round(s2["arousal"], 4),
                "valence": round(s2["valence"], 4),
                "focus":   round(s2["focus"],   4),
                "label":   s2.get("label", ""),
            },
            "ch_amplitudes": _round_amps(s2.get("ch_amplitudes", [])),
        },
        "active_user": active,
        "symbiosis":   sym,
    }
    if capture:
        msg["capture"] = True
    return json.dumps(msg, separators=(",", ":")), b1, b2


async def _broadcast_loop() -> None:
    # Wait until both boards have confirmed a connection before doing anything
    await asyncio.get_event_loop().run_in_executor(None, _both_connected.wait)
    pb1, pb2 = False, False
    while True:
        snap = _snapshot()
        msg, pb1, pb2 = _build_msg(snap, pb1, pb2)
        dead: set[Any] = set()
        for ws in _clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        _clients.difference_update(dead)

        # Terminal status line
        s1, s2 = snap["user1"], snap["user2"]
        sym    = _symbiosis(s1, s2)
        active = snap["active_user"]
        b1s = "◉" if s1["blink"] else "○"
        b2s = "◉" if s2["blink"] else "○"
        l1  = s1.get("label", "") or "—"
        l2  = s2.get("label", "") or "—"
        creating = f"USER {active} creating"
        print(
            f"\r  {creating:16s}  │  U1: {l1:5s} {b1s}"
            f"  │  U2: {l2:5s} {b2s}"
            f"  │  {sym['label']}",
            end="          ", flush=True,
        )
        await asyncio.sleep(TICK_MS / 1000)


async def _ws_handler(websocket: Any) -> None:
    _clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        _clients.discard(websocket)


async def _main_async(port: int) -> None:
    print(f"  WebSocket ready on ws://127.0.0.1:{port}\n")

    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    import signal as _signal
    for sig in (_signal.SIGINT, _signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set_result, None)

    async with websockets.serve(_ws_handler, "127.0.0.1", port):
        task = asyncio.ensure_future(_broadcast_loop())
        try:
            await asyncio.wait(
                [task, asyncio.ensure_future(stop)],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            print("\nStopped.")


def _free_port(port: int) -> None:
    import signal as _signal, subprocess
    res = subprocess.run(["lsof", "-ti", f"tcp:{port}"],
                         capture_output=True, text=True)
    pids = [p for p in res.stdout.strip().split() if p]
    for pid in pids:
        try:
            os.kill(int(pid), _signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass
    if pids:
        time.sleep(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dual-user EEG decoder — Neural Symbiosis demo"
    )
    parser.add_argument("--port",        type=int, default=8765)
    parser.add_argument("--mock",        action="store_true",
                        help="Fake both users (no hardware)")
    parser.add_argument("--mock-user1",  action="store_true",
                        help="Fake User 1 only; real Cyton for User 2")
    parser.add_argument("--cyton-port",  default="",
                        help="Serial port for Cyton dongle, e.g. /dev/cu.usbserial-XXXX")
    parser.add_argument(
        "--blink-profile-u1", default="",
        help="Path to blink .npz for User 1 (GTECH); default eeg/models/blink_user1.npz",
    )
    parser.add_argument(
        "--blink-profile-u2", default="",
        help="Path to blink .npz for User 2 (Cyton); default eeg/models/blink_user2.npz",
    )
    parser.add_argument(
        "--blink-ch-u1", type=int, default=0,
        help="Frontal EEG column for User 1 BlinkDetector (default: 0)",
    )
    parser.add_argument(
        "--blink-ch-u2", type=int, default=0,
        help="Frontal EEG column for User 2 BlinkDetector (default: 0)",
    )
    parser.add_argument(
        "--no-blink-paper",
        action="store_true",
        help="Disable BLINK paper detector for both users; amplitude double-blink only",
    )
    args = parser.parse_args()

    print("\n  ── Neural Symbiosis  ─────────────────────────────")
    print(f"  port: ws://127.0.0.1:{args.port}\n")

    _free_port(args.port)

    blink_u1 = _resolve_blink_npz(args.blink_profile_u1, "blink_user1.npz")
    blink_u2 = _resolve_blink_npz(args.blink_profile_u2, "blink_user2.npz")

    # ── Start decoder threads ──────────────────────────────────────────────────
    if args.mock:
        threading.Thread(
            target=_run_mock, args=(1, _MockUser(0, _MOCK_U1, user=1)),
            daemon=True, name="mock-u1"
        ).start()
        threading.Thread(
            target=_run_mock, args=(2, _MockUser(1, _MOCK_U2, user=2)),
            daemon=True, name="mock-u2"
        ).start()
        print("  [✓] User 1  GTECH BCICore-8   mock")
        print("  [✓] User 2  OpenBCI Cyton      mock\n")

    elif args.mock_user1:
        print("  [✓] User 1  GTECH BCICore-8   mock")
        threading.Thread(
            target=_run_mock, args=(1, _MockUser(0, _MOCK_U1, user=1)),
            daemon=True, name="mock-u1"
        ).start()
        threading.Thread(
            target=_run_cyton_user2,
            args=(
                args.cyton_port,
                0.1,
                blink_u2,
                args.blink_ch_u2,
                args.no_blink_paper,
            ),
            daemon=True, name="cyton-u2"
        ).start()

    else:
        threading.Thread(
            target=_run_live_user1,
            args=(0.1, blink_u1, args.blink_ch_u1, args.no_blink_paper),
            daemon=True, name="gtech-u1"
        ).start()
        threading.Thread(
            target=_run_cyton_user2,
            args=(
                args.cyton_port,
                0.1,
                blink_u2,
                args.blink_ch_u2,
                args.no_blink_paper,
            ),
            daemon=True, name="cyton-u2"
        ).start()

    # ── stdin permission switcher ──────────────────────────────────────────────
    threading.Thread(target=_stdin_thread, daemon=True, name="stdin-perm").start()

    asyncio.run(_main_async(args.port))
