"""
Minimal WebSocket server for testing EEG "live decode" → browser.

Sends a JSON blink capture every ~5s:
  {"capture": true}

Run (needs `pip install websockets`):
  python eeg_mock_ws.py

Then in the app: World panel → EEG → connect to ws://127.0.0.1:8765
"""
from __future__ import annotations

import asyncio
import json

try:
    import websockets
except ImportError:
    raise SystemExit("pip install websockets") from None


async def blink_loop(websocket):  # noqa: ARG001
    await websocket.send(json.dumps({"present": True, "blink": False}))
    while True:
        await asyncio.sleep(5.0)
        await websocket.send(json.dumps({"capture": True}))
        await asyncio.sleep(0.05)
        await websocket.send(json.dumps({"present": True, "blink": False}))


async def handler(websocket):
    await blink_loop(websocket)


async def main():
    async with websockets.serve(handler, "127.0.0.1", 8765):
        print("EEG mock WebSocket ws://127.0.0.1:8765 (sends capture every 5s)")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
