"""
EEG live decoder for OpenBCI Cyton (USB dongle) → WebSocket.

Mirrors backend/eeg_decode.py but swaps the g.tec/gpype pipeline for BrainFlow.

Emotion detection:
  Default   : BrainFlow RESTFULNESS classifier (no model weights needed)
  --eegnet  : Trained EEGNet/MLP from eeg/models/ (requires eeg/train.py first)

Run:
  python backend/eeg_decode_cyton.py --serial-port /dev/cu.usbserial-XXXX
  python backend/eeg_decode_cyton.py --serial-port /dev/cu.usbserial-XXXX --eegnet
  python backend/eeg_decode_cyton.py --mock
  python backend/eeg_decode_cyton.py --serial-port /dev/cu.usbserial-XXXX --ws-port 8765

  RUN "ls -l /dev/cu.* /dev/tty.*" IN TERMINAL TO FIND CORRECT SERIAL PORT
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

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════════
# Emotion circumplex mapping
# ═══════════════════════════════════════════════════════════════════════════════

EMOTION_AV: dict[str, tuple[float, float]] = {
    "happy":       (0.70, 0.85),
    "sad":         (0.25, 0.20),
    "neutral":     (0.50, 0.50),
    "relaxed":     (0.70, 0.85),
    "not relaxed": (0.25, 0.20),
}

EMOTION_INTERVAL_S = 0.5
EMOTION_HISTORY_N  = 6


# ═══════════════════════════════════════════════════════════════════════════════
# Mock decoder (no hardware)
# ═══════════════════════════════════════════════════════════════════════════════

MOCK_EMOTIONS = [
    {"label": "happy",   "arousal": 0.70, "valence": 0.85, "focus": 0.75},
    {"label": "sad",     "arousal": 0.25, "valence": 0.20, "focus": 0.40},
    {"label": "neutral", "arousal": 0.50, "valence": 0.50, "focus": 0.55},
]
MOCK_HOLD_SECONDS = 3


class MockDecoder:
    """Cycles through emotions for UI testing without hardware."""

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self._last_blink = 0.0
        self._blink_on = False
        self._emo_idx = 0
        self._emo_switch_at = self._t0 + MOCK_HOLD_SECONDS

    def setup(self) -> None: ...
    def cleanup(self) -> None: ...

    def decode(self) -> dict[str, Any]:
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


# ═══════════════════════════════════════════════════════════════════════════════
# Live decoder — BrainFlow RESTFULNESS (default, no model weights needed)
# ═══════════════════════════════════════════════════════════════════════════════

class CytonLiveDecoder:
    """
    Wraps CytonDecoder from eeg/cyton_stream.py.
    Uses BrainFlow's built-in RESTFULNESS classifier for emotion.
    """

    def __init__(
        self,
        serial_port: str = "",
        *,
        blink_profile: Optional[str] = None,
        blink_frontal_ch: int = 0,
        use_blink_paper: bool = True,
    ) -> None:
        self._serial_port = serial_port
        self._blink_profile = blink_profile
        self._blink_frontal_ch = blink_frontal_ch
        self._use_blink_paper = use_blink_paper
        self._dec: Any = None

    def setup(self) -> None:
        from eeg.cyton_stream import CytonDecoder

        self._dec = CytonDecoder(
            serial_port=self._serial_port,
            blink_profile=self._blink_profile,
            frontal_ch=self._blink_frontal_ch,
            use_blink_paper=self._use_blink_paper,
        )
        self._dec.start()

    def decode(self) -> dict[str, Any]:
        return self._dec.decode()

    def cleanup(self) -> None:
        if self._dec is not None:
            self._dec.stop()
            self._dec = None

    @property
    def buffer(self):
        return self._dec._buf if self._dec else None


# ═══════════════════════════════════════════════════════════════════════════════
# Live decoder — EEGNet / EmotionMLP (trained model from eeg/models/)
# ═══════════════════════════════════════════════════════════════════════════════

class CytonEEGNetDecoder:
    """
    Uses BrainFlow for data acquisition (same as CytonLiveDecoder) but runs
    the trained EEGNet/EmotionMLP model for emotion classification instead of
    BrainFlow RESTFULNESS.

    Requires:
      eeg/models/eegnet_emotion.pt
      eeg/models/eegnet_config.json
    Produced by: python eeg/train.py
    """

    def __init__(
        self,
        serial_port: str = "",
        *,
        blink_profile: Optional[str] = None,
        blink_frontal_ch: int = 0,
        use_blink_paper: bool = True,
    ) -> None:
        self._serial_port = serial_port
        self._blink_profile = blink_profile
        self._blink_frontal_ch = blink_frontal_ch
        self._use_blink_paper = use_blink_paper

        self._board: Any = None
        self._buf: Any = None
        self._blink_det: Any = None
        self._poll_thread: Optional[threading.Thread] = None
        self._running = False
        self._eeg_channels: list[int] = []

        self._model: Any = None
        self._cfg: dict = {}
        self._device: Any = None

        self._prob_history: collections.deque = collections.deque(maxlen=EMOTION_HISTORY_N)
        self._next_emotion_t = 0.0
        self._last_arousal = 0.5
        self._last_valence = 0.5
        self._last_focus   = 0.5
        self._last_label   = ""

    def setup(self) -> None:
        import torch
        from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
        from eeg.cyton_stream import CytonBuffer, FS
        from eeg.eegnet import EEGNet, EmotionMLP

        config_path  = os.path.join(PROJECT_ROOT, "eeg", "models", "eegnet_config.json")
        weights_path = os.path.join(PROJECT_ROOT, "eeg", "models", "eegnet_emotion.pt")

        if not os.path.isfile(config_path) or not os.path.isfile(weights_path):
            raise FileNotFoundError(
                "EEGNet model not found. Run 'python eeg/train.py' first, "
                "or use the default BrainFlow RESTFULNESS mode (omit --eegnet)."
            )

        with open(config_path) as f:
            self._cfg = json.load(f)

        self._device = torch.device(
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_cls = EmotionMLP if self._cfg.get("model", "eegnet") == "mlp" else EEGNet
        self._model = model_cls(
            n_channels=self._cfg["n_channels"],
            n_timepoints=self._cfg["n_timepoints"],
            n_classes=self._cfg["n_classes"],
        ).to(self._device)
        self._model.load_state_dict(
            torch.load(weights_path, map_location=self._device)
        )
        self._model.eval()
        print(f"  [EEGNet] loaded {model_cls.__name__} "
              f"({self._cfg['n_classes']} classes: {self._cfg.get('emotions', [])})")

        BoardShim.disable_board_logger()
        params = BrainFlowInputParams()
        if self._serial_port:
            params.serial_port = self._serial_port

        self._board = BoardShim(BoardIds.CYTON_BOARD.value, params)
        print(f"  [Cyton] opening board on {self._serial_port or '(auto)'} ...", flush=True)
        self._board.prepare_session()
        self._board.start_stream()
        self._eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
        print(f"  [Cyton] streaming — EEG channels: {self._eeg_channels}", flush=True)

        self._buf = CytonBuffer()

        if self._use_blink_paper:
            from eeg.blink_detector import BlinkDetector
            self._blink_det = BlinkDetector(
                fs=FS,
                frontal_ch=self._blink_frontal_ch,
                profile=self._blink_profile,
            )

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="cyton-eegnet-poll"
        )
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        while self._running:
            try:
                if self._board is not None:
                    count = self._board.get_board_data_count()
                    if count > 0:
                        data = self._board.get_board_data(count)
                        eeg = data[self._eeg_channels, :].T.astype(np.float64)
                        if self._blink_det is not None:
                            self._blink_det.feed(eeg)
                        self._buf.push(eeg)
            except Exception:
                pass
            time.sleep(0.04)

    def decode(self) -> dict[str, Any]:
        import torch

        if self._blink_det is not None and self._blink_det.ready:
            blink = self._blink_det.check()
        else:
            blink = self._check_blink_amplitude()

        now = time.monotonic()
        if now >= self._next_emotion_t:
            self._next_emotion_t = now + EMOTION_INTERVAL_S
            probs = self._run_eegnet()
            if probs is not None:
                self._prob_history.append(probs)

        if self._prob_history:
            emotions = list(next(iter(self._prob_history)).keys())
            n = len(self._prob_history)
            avg_probs = {e: sum(p[e] for p in self._prob_history) / n for e in emotions}
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

    def _run_eegnet(self) -> Optional[dict[str, float]]:
        import torch

        n_tp = self._cfg["n_timepoints"]
        window = self._buf.latest(n_tp)
        if window is None:
            return None

        mean = np.array(self._cfg["ch_mean"], dtype=np.float32)
        std  = np.array(self._cfg["ch_std"],  dtype=np.float32)
        win  = (window - mean) / (std + 1e-8)

        x = torch.from_numpy(win.T[np.newaxis, np.newaxis]).float().to(self._device)
        with torch.no_grad():
            logits = self._model(x)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        return {name: float(probs[i])
                for i, name in enumerate(self._cfg["emotions"])}

    def _check_blink_amplitude(self) -> bool:
        from eeg.cyton_stream import BLINK_WINDOW, BLINK_MIN, BLINK_MAX
        window = self._buf.latest(BLINK_WINDOW)
        if window is None:
            return False
        amp = np.abs(window).mean(axis=1)
        active = (amp > BLINK_MIN) & (amp < BLINK_MAX)
        spike_count = int((np.diff(active.astype(np.int8)) == 1).sum())
        return spike_count >= 2

    def cleanup(self) -> None:
        self._running = False
        if self._board is not None:
            try:
                self._board.stop_stream()
                self._board.release_session()
            except Exception:
                pass
            self._board = None

    @property
    def buffer(self):
        return self._buf


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket server — pushes decoded state to browser at 10 Hz
# ═══════════════════════════════════════════════════════════════════════════════

TICK_MS = 100
clients: set[Any] = set()


def build_message(state: dict[str, Any], prev_blink: bool) -> tuple[str, bool]:
    blink_now = bool(state.get("blink"))
    capture = not prev_blink and blink_now

    msg: dict[str, Any] = {
        "present": True,
        "blink": blink_now,
        "emotion": {
            "arousal": round(state["arousal"], 4),
            "valence": round(state["valence"], 4),
            "focus":   round(state["focus"], 4),
            "label":   state.get("label", ""),
        },
    }
    if capture:
        msg["capture"] = True

    return json.dumps(msg, separators=(",", ":")), blink_now


async def decode_loop(decoder) -> None:
    prev_blink = False
    while True:
        state = decoder.decode()
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
            f"\r  {lbl:12s}  A={a:.2f}  V={v:.2f}  F={f:.2f}  {b}  clients={len(clients)}",
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


async def main(decoder, port: int) -> None:
    decoder.setup()
    print(f"\n  EEG Cyton decode → ws://127.0.0.1:{port}  (every {TICK_MS}ms)")
    print("  Connect in World panel → EEG → connect\n")

    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    import signal as _signal
    for sig in (_signal.SIGINT, _signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set_result, None)

    async with websockets.serve(ws_handler, "127.0.0.1", port):
        decode_task = asyncio.ensure_future(decode_loop(decoder))
        try:
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
    import signal, subprocess
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True, text=True,
    )
    pids = [p for p in result.stdout.strip().split() if p]
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed stale process {pid} on port {port}")
        except (ProcessLookupError, ValueError):
            pass
    if pids:
        time.sleep(1.0)


def _default_blink_npz(name: str) -> Optional[str]:
    p = os.path.join(PROJECT_ROOT, "eeg", "models", name)
    return p if os.path.isfile(p) else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG live decode (OpenBCI Cyton) → WebSocket"
    )
    parser.add_argument(
        "--serial-port", default="",
        help="Serial port for Cyton USB dongle, e.g. /dev/cu.usbserial-DP05J1WH",
    )
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port")
    parser.add_argument(
        "--mock", action="store_true",
        help="Use fake decoder (no hardware needed)",
    )
    parser.add_argument(
        "--eegnet", action="store_true",
        help="Use trained EEGNet/MLP model instead of BrainFlow RESTFULNESS",
    )
    parser.add_argument(
        "--blink-profile", default="",
        help="Path to blink_*.npz from tools/calibrate_blink.py",
    )
    parser.add_argument(
        "--no-blink-paper", action="store_true",
        help="Disable BLINK paper detector; use amplitude double-blink only",
    )
    parser.add_argument(
        "--blink-ch", type=int, default=0,
        help="Frontal EEG column index for BlinkDetector (default: 0)",
    )
    args = parser.parse_args()

    free_port(args.ws_port)

    blink_path = (args.blink_profile.strip()
                  or _default_blink_npz("blink_user1.npz")
                  or _default_blink_npz("blink_user2.npz"))

    if args.mock:
        decoder = MockDecoder()
    elif args.eegnet:
        decoder = CytonEEGNetDecoder(
            serial_port=args.serial_port,
            blink_profile=blink_path,
            blink_frontal_ch=args.blink_ch,
            use_blink_paper=not args.no_blink_paper,
        )
    else:
        decoder = CytonLiveDecoder(
            serial_port=args.serial_port,
            blink_profile=blink_path,
            blink_frontal_ch=args.blink_ch,
            use_blink_paper=not args.no_blink_paper,
        )

    asyncio.run(main(decoder, args.ws_port))
