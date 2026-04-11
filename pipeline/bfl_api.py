"""Minimal Black Forest Labs (BFL) FLUX text-to-image client (async poll + download)."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

BFL_API_BASE = "https://api.bfl.ai/v1"


class BFLAPIError(RuntimeError):
    pass


def bfl_generate_image_bytes(
    *,
    api_key: str,
    prompt: str,
    model: str = "flux-2-klein-4b",
    width: int = 1024,
    height: int = 1024,
    poll_interval_sec: float = 0.5,
    max_wait_sec: float = 420.0,
    session: requests.Session | None = None,
) -> tuple[bytes, str]:
    """Submit prompt to BFL, poll until ready, download image. Returns (image_bytes, mime_guess).

    Uses ``POST /v1/{model}`` with header ``x-key`` and JSON body ``prompt``, ``width``, ``height``
    (FLUX.2 klein / dev style). See https://docs.bfl.ml/
    """
    sess = session or requests.Session()
    url = f"{BFL_API_BASE}/{model.lstrip('/')}"
    headers = {
        "accept": "application/json",
        "x-key": api_key,
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "prompt": prompt,
        "width": int(width),
        "height": int(height),
    }

    r = sess.post(url, headers=headers, json=body, timeout=120)
    if r.status_code == 402:
        raise BFLAPIError("BFL API returned 402 (insufficient credits).")
    if r.status_code == 429:
        raise BFLAPIError("BFL API returned 429 (rate limit / too many active tasks).")
    r.raise_for_status()
    data = r.json()
    polling_url = data.get("polling_url")
    if not polling_url:
        raise BFLAPIError(f"BFL response missing polling_url: {data!r}")

    deadline = time.monotonic() + max_wait_sec
    last_status = ""
    while time.monotonic() < deadline:
        time.sleep(poll_interval_sec)
        pr = sess.get(
            polling_url,
            headers={"accept": "application/json", "x-key": api_key},
            timeout=60,
        )
        pr.raise_for_status()
        result = pr.json()
        status = result.get("status", "")
        if status != last_status:
            logger.info("BFL status: %s", status)
            last_status = status

        if status == "Ready":
            sample_url = (result.get("result") or {}).get("sample")
            if not sample_url:
                raise BFLAPIError(f"BFL Ready but no result.sample: {result!r}")
            ir = sess.get(sample_url, timeout=120)
            ir.raise_for_status()
            ctype = ir.headers.get("Content-Type", "image/jpeg")
            return ir.content, ctype

        if status in ("Error", "Failed"):
            raise BFLAPIError(f"BFL generation failed: {result!r}")

    raise BFLAPIError(f"BFL polling timed out after {max_wait_sec:.0f}s (last={last_status!r})")
