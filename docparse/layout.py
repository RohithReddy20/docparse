"""Thin client for the PP-DocLayoutV3 server (serve/modal_paddle_layout.py).

Used by the composite pipeline; parse-bench runs go through
docparse.providers.paddle_layout_modal instead.
"""

from __future__ import annotations

import io
import os
from typing import Any, TypedDict

import requests
from PIL import Image


class LayoutBox(TypedDict):
    bbox_xyxy_norm: list[float]  # [x1, y1, x2, y2] in [0, 1] of the image
    bbox_xyxy_px: list[float]
    label: str  # PP-DocLayout 20-class name (table, chart, text, doc_title, ...)
    raw_label: str  # PP-DocLayoutV3 25-class name
    score: float
    order: int  # model reading order


def detect_layout(
    image: Image.Image,
    endpoint: str | None = None,
    *,
    timeout: int = 300,
    min_score: float = 0.0,
) -> list[LayoutBox]:
    """POST a PIL image to `{endpoint}/predict` and return boxes in reading order."""
    endpoint = (endpoint or os.getenv("PADDLE_LAYOUT_ENDPOINT_URL") or "").rstrip("/")
    if not endpoint:
        raise ValueError("No endpoint: pass `endpoint` or set PADDLE_LAYOUT_ENDPOINT_URL")

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    resp = requests.post(
        f"{endpoint}/predict", data=buf.getvalue(), headers={"Content-Type": "image/png"}, timeout=timeout
    )
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()

    w = float(payload.get("image_width") or image.width)
    h = float(payload.get("image_height") or image.height)
    boxes: list[LayoutBox] = []
    for i, p in enumerate(payload.get("predictions", [])):
        score = float(p.get("score", 0.0))
        if score < min_score:
            continue
        x1, y1, x2, y2 = (float(v) for v in p["coordinate"][:4])
        boxes.append(
            LayoutBox(
                bbox_xyxy_norm=[x1 / w, y1 / h, x2 / w, y2 / h],
                bbox_xyxy_px=[x1, y1, x2, y2],
                label=str(p.get("label", "")),
                raw_label=str(p.get("raw_label", p.get("label", ""))),
                score=score,
                order=int(p.get("order", i)),
            )
        )
    boxes.sort(key=lambda b: b["order"])
    return boxes
