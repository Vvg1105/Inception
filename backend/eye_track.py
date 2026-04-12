"""
Eye tracking via iris landmarks — Python (OpenCV + MediaPipe Tasks) → WebSocket → browser.

Tracks eye rolling (iris position inside the eye socket), NOT head movement.
Sends gaze NDC + blink state to the browser at ~30 fps.

Run:
  pip install opencv-python mediapipe websockets
  python eye_track.py [--port 8766] [--no-preview]

Then in the app: World panel → Eye tracking → connect to ws://127.0.0.1:8766
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import urllib.request
from typing import Any

import cv2

try:
    import websockets
except ImportError:
    raise SystemExit("pip install websockets") from None

# MediaPipe Tasks API
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def ensure_model() -> str:
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    print(f"Downloading face_landmarker model → {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")
    return MODEL_PATH


# ═══════════════════════════════════════════════════════════════════════════
# Landmark indices (same as MediaPipe FaceMesh 478-point mesh)
# ═══════════════════════════════════════════════════════════════════════════
L_IRIS = 468
R_IRIS = 473
L_INNER, L_OUTER = 133, 33
R_INNER, R_OUTER = 362, 263
L_TOP, L_BOT = 159, 145
R_TOP, R_BOT = 386, 374


# ═══════════════════════════════════════════════════════════════════════════
# Iris gaze computation
# ═══════════════════════════════════════════════════════════════════════════

def iris_offset_x(lm: list, iris: int, inner: int, outer: int) -> float | None:
    """Iris displacement from eye center, normalized by half eye-width. Flip-safe."""
    cx = (lm[inner].x + lm[outer].x) / 2
    half_w = abs(lm[inner].x - lm[outer].x) / 2
    if half_w < 1e-6:
        return None
    return (lm[iris].x - cx) / half_w


def iris_offset_y(lm: list, iris: int, top: int, bot: int) -> float | None:
    """Iris displacement from eye center, normalized by half eye-height. Positive = down."""
    cy = (lm[top].y + lm[bot].y) / 2
    half_h = abs(lm[bot].y - lm[top].y) / 2
    if half_h < 1e-6:
        return None
    return (lm[iris].y - cy) / half_h


def eye_openness(lm: list, top: int, bot: int) -> float:
    return abs(lm[bot].y - lm[top].y)


H_GAIN = 4.0
V_GAIN = 3.0


def compute_gaze(lm: list, sens: float = 1.0) -> dict[str, Any]:
    lx = iris_offset_x(lm, L_IRIS, L_INNER, L_OUTER)
    rx = iris_offset_x(lm, R_IRIS, R_INNER, R_OUTER)
    ly = iris_offset_y(lm, L_IRIS, L_TOP, L_BOT)
    ry = iris_offset_y(lm, R_IRIS, R_TOP, R_BOT)

    hx = [v for v in (lx, rx) if v is not None]
    vy = [v for v in (ly, ry) if v is not None]

    avg_x = sum(hx) / len(hx) if hx else 0.0
    avg_y = sum(vy) / len(vy) if vy else 0.0

    x = avg_x * H_GAIN * sens
    y = -avg_y * V_GAIN * sens

    x = max(-1.4, min(1.4, x))
    y = max(-1.4, min(1.4, y))

    lo = eye_openness(lm, L_TOP, L_BOT)
    ro = eye_openness(lm, R_TOP, R_BOT)
    blink = False

    return {
        "x": round(x, 4),
        "y": round(y, 4),
        "blink": blink,
        "avg_x": round(avg_x, 4),
        "avg_y": round(avg_y, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# OpenCV debug preview
# ═══════════════════════════════════════════════════════════════════════════

def draw_preview(frame, lm: list, gaze: dict[str, Any]) -> None:
    h, w = frame.shape[:2]

    for idx in (L_IRIS, R_IRIS):
        cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    for idx in (L_INNER, L_OUTER, R_INNER, R_OUTER):
        cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
        cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)

    txt = f"gaze ({gaze['x']:+.2f}, {gaze['y']:+.2f})"
    if gaze["blink"]:
        txt += "  BLINK"
    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# ═══════════════════════════════════════════════════════════════════════════
# WebSocket server + capture loop
# ═══════════════════════════════════════════════════════════════════════════

clients: set[Any] = set()


async def ws_handler(websocket: Any) -> None:
    clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        clients.discard(websocket)


async def broadcast(msg: str) -> None:
    dead: set[Any] = set()
    for ws in clients:
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


async def track_loop(port: int, show_preview: bool, sens: float) -> None:
    model_path = ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Eye tracking → ws://127.0.0.1:{port}  (~30 fps)")
    print("Connect in World panel → Eye tracking → connect")
    if show_preview:
        print("Preview window open — press Q to quit")
    print()

    prev_blink = False
    smooth_x, smooth_y = 0.0, 0.0
    frame_ts = 0

    async with websockets.serve(ws_handler, "127.0.0.1", port):
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                frame_ts += 33
                result = landmarker.detect_for_video(mp_image, frame_ts)

                if result.face_landmarks and len(result.face_landmarks) > 0:
                    lm = result.face_landmarks[0]

                    if len(lm) >= 474:
                        gaze = compute_gaze(lm, sens)

                        smooth_x += (gaze["x"] - smooth_x) * 0.55
                        smooth_y += (gaze["y"] - smooth_y) * 0.45

                        blink_now = gaze["blink"]
                        capture = prev_blink and not blink_now
                        prev_blink = blink_now

                        msg: dict[str, Any] = {
                            "gaze": {
                                "x": round(smooth_x, 4),
                                "y": round(smooth_y, 4),
                            },
                            "blink": blink_now,
                        }
                        if capture:
                            msg["capture"] = True

                        await broadcast(json.dumps(msg, separators=(",", ":")))

                        b = "BLINK" if blink_now else "     "
                        print(
                            f"\r  gaze=({smooth_x:+.2f}, {smooth_y:+.2f})  "
                            f"iris=({gaze['avg_x']:+.3f}, {gaze['avg_y']:+.3f})  "
                            f"{b}  clients={len(clients)}",
                            end="", flush=True,
                        )

                        if show_preview:
                            draw_preview(frame, lm, gaze)
                    else:
                        no_iris_msg: dict[str, Any] = {
                            "gaze": {"x": round(smooth_x, 4), "y": round(smooth_y, 4)},
                            "blink": False,
                            "no_face": True,
                        }
                        await broadcast(json.dumps(no_iris_msg, separators=(",", ":")))
                else:
                    msg_no_face: dict[str, Any] = {
                        "gaze": {"x": round(smooth_x, 4), "y": round(smooth_y, 4)},
                        "blink": False,
                        "no_face": True,
                    }
                    await broadcast(json.dumps(msg_no_face, separators=(",", ":")))

                if show_preview:
                    cv2.imshow("Eye Tracking", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                await asyncio.sleep(1 / 30)
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
            landmarker.close()


async def main(port: int, show_preview: bool, sens: float) -> None:
    await track_loop(port, show_preview, sens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eye tracking → WebSocket")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-preview", action="store_true", help="Hide OpenCV debug window")
    parser.add_argument("--sens", type=float, default=1.0, help="Gaze sensitivity multiplier")
    args = parser.parse_args()

    asyncio.run(main(args.port, not args.no_preview, args.sens))
