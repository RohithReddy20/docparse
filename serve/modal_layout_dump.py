"""Run the PP-DocLayout-V3 server over every dataset page and save the detections.

Runs on Modal next to the dataset volume: pages are rendered at DPI, posted to the
layout endpoint (PADDLE_LAYOUT_ENDPOINT_URL from .env), and one JSON per PDF is
written under /vol/bench/layout_v3/<group>/<name>.json:

    {"dpi": 200, "pages": [{"width": W, "height": H, "predictions": [
        {"bbox": [x1, y1, x2, y2], "label": "...", "raw_label": "...", "score": s, "order": i}]}]}

Usage:
    uv run modal run --detach serve/modal_layout_dump.py            # all groups
    uv run modal run serve/modal_layout_dump.py --groups layout     # one group
    uv run modal run serve/modal_layout_dump.py --mode pack         # tgz for download
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
ENV = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v}
VOL_PATH = "/vol"
BENCH_DIR = f"{VOL_PATH}/bench"
DPI = 200

image = modal.Image.debian_slim(python_version="3.12").pip_install("pymupdf>=1.24", "requests").env(ENV)
vol = modal.Volume.from_name("docparse-output", create_if_missing=True)
app = modal.App("docparse-layout-dump")


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=2, memory=4096, timeout=4 * 3600)
def dump(groups: str = "chart,layout,table,text", limit: int = 0) -> str:
    import threading
    import time

    import pymupdf
    import requests

    url = os.environ["PADDLE_LAYOUT_ENDPOINT_URL"].rstrip("/")
    # wait for the scale-to-zero container
    for _ in range(90):
        try:
            if requests.get(f"{url}/health", timeout=20).ok:
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(10)
    out_root = Path(BENCH_DIR) / "layout_v3"
    done = failed = 0
    stop = threading.Event()

    def committer() -> None:
        while not stop.wait(120):
            vol.commit()

    threading.Thread(target=committer, daemon=True).start()
    for group in groups.split(","):
        pdfs = sorted((Path(BENCH_DIR) / "data" / "docs" / group).glob("*.pdf"))
        if limit:
            pdfs = pdfs[:limit]
        (out_root / group).mkdir(parents=True, exist_ok=True)
        for pdf in pdfs:
            target = out_root / group / (pdf.stem + ".json")
            if target.exists():
                done += 1
                continue
            try:
                doc = pymupdf.open(pdf)
                pages = []
                for page in doc:
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(DPI / 72, DPI / 72), alpha=False)
                    r = requests.post(f"{url}/predict", data=pix.tobytes("png"), headers={"Content-Type": "image/png"}, timeout=300)
                    r.raise_for_status()
                    j = r.json()
                    pages.append({
                        "width": j.get("image_width", pix.width), "height": j.get("image_height", pix.height),
                        "predictions": [
                            {"bbox": p["coordinate"], "label": p["label"], "raw_label": p.get("raw_label", p["label"]),
                             "score": p["score"], "order": p.get("order", i)}
                            for i, p in enumerate(j["predictions"])
                        ],
                    })
                doc.close()
                target.write_text(json.dumps({"dpi": DPI, "pages": pages}))
                done += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAILED {pdf.name}: {e}", flush=True)
            if (done + failed) % 100 == 0:
                print(f"progress {done} done, {failed} failed", flush=True)
    stop.set()
    vol.commit()
    return f"done={done} failed={failed}"


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=1, memory=2048, timeout=900)
def pack() -> str:
    import tarfile

    src = Path(BENCH_DIR) / "layout_v3"
    tgz = Path(BENCH_DIR) / "layout_v3.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(src, arcname="layout_v3")
    vol.commit()
    return f"{tgz.relative_to(VOL_PATH)} {tgz.stat().st_size / 1e6:.1f}MB"


@app.local_entrypoint()
def main(mode: str = "dump", groups: str = "chart,layout,table,text", limit: int = 0):
    if mode == "dump":
        print(dump.remote(groups, limit))
    else:
        print(pack.remote())
