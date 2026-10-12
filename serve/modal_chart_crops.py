"""Selective chart-crop fallback: run PaddleOCR-VL-1.6's element-level chart mode
(prompt ``Chart Recognition:``, Apache-2.0, 0.9B) on every chart block the page VLM
found, and save the returned tables so the hybrid provider can add them as extra
candidate tables (docparse/charts/vlm_fallback.py).

Chart blocks come from the saved LightOnOCR raw outputs on the volume
(``/vol/bench/output/lightonocr3_0_8b_grounding/<group>/<name>.raw.json``):
``![chart](x1,y1,x2,y2) <table>…`` blocks on a 0–1000 grid, plus ``image`` blocks whose
text contains a ``<table``. Each block is cropped from a 200-DPI render, padded by 6% so
title/axes/legend stay in frame, and sent to the model.

Output: ``/vol/bench/chart_vlm/<group>/<name>.json`` =
``{"pages": [{"charts": [{"bbox": [x1,y1,x2,y2], "text": <model output>, "ms": n}]}]}``.

    uv run modal run --detach serve/modal_chart_crops.py                # --groups chart
    uv run modal run serve/modal_chart_crops.py --mode pack             # -> /vol/bench/chart_vlm.tgz
    DOCPARSE_CROP_GPU=L4 uv run modal run --detach serve/modal_chart_crops.py --limit 20
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import modal

VOL_PATH = "/vol"
BENCH_DIR = f"{VOL_PATH}/bench"
RAW_DIR = f"{BENCH_DIR}/output/lightonocr3_0_8b_grounding"
OUT_DIR = f"{BENCH_DIR}/chart_vlm"
DPI = 200
PAD = 0.06
MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
PROMPT = "Chart Recognition:"  # model card: element-level chart task prompt
MAX_PIXELS = 1280 * 28 * 28  # model card default for non-spotting tasks
MAX_NEW_TOKENS = 2048  # card example uses 512; dense chart tables need more

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "transformers>=5.0.0", "pillow", "pymupdf>=1.24", "huggingface_hub", "accelerate")
    .env({"HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("docparse")
)
vol = modal.Volume.from_name("docparse-output", create_if_missing=True)
app = modal.App("docparse-chart-crops")

_PLACEHOLDER_RE = re.compile(r"^!\[([A-Za-z_\-]+?)(\+?)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]?(.*)$")


def chart_blocks(raw_text: str) -> list[list[int]]:
    """bboxes (0–1000 grid) of chart blocks, and image blocks that carry a table."""
    import re as _re

    _PH = _re.compile(r"^!\[([A-Za-z_\-]+?)(\+?)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]?(.*)$")

    def parse_grounding(text):  # minimal copy of docparse.providers.lightonocr3.parse_grounding (no parse_bench import)
        blocks, cur, buf = [], None, []
        for line in text.splitlines():
            m = _PH.match(line.strip())
            if m:
                if cur is not None:
                    cur["text"] = "\n".join(buf).strip(); blocks.append(cur)
                label, cont, x1, y1, x2, y2, rest = m.groups()
                cur = {"label": label.lower(), "continuation": cont == "+", "bbox": [int(x1), int(y1), int(x2), int(y2)]}
                buf = [rest] if rest else []
            elif cur is not None:
                buf.append(line)
        if cur is not None:
            cur["text"] = "\n".join(buf).strip(); blocks.append(cur)
        return blocks

    out: list[list[int]] = []
    for b in parse_grounding(raw_text):
        if b.get("bbox") is None:
            continue
        if b["label"] == "chart" or (b["label"] == "image" and "<table" in b["text"]):
            out.append([int(v) for v in b["bbox"]])
    return out


@app.function(
    image=image,
    volumes={VOL_PATH: vol},
    gpu=os.environ.get("DOCPARSE_CROP_GPU", "T4"),
    cpu=2,
    memory=12288,
    timeout=3 * 3600,
)
def dump(groups: str = "chart", limit: int = 0) -> str:
    import threading

    import pymupdf
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # T4 has no bf16: fall back to fp16 there (card loads bfloat16).
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    min_pixels = getattr(processor.image_processor, "min_pixels", 56 * 56)

    def recognize(img: Image.Image) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={"size": {"shortest_edge": min_pixels, "longest_edge": MAX_PIXELS}},
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    stop = threading.Event()

    def committer() -> None:
        while not stop.wait(120):
            vol.commit()

    threading.Thread(target=committer, daemon=True).start()
    done = failed = n_charts = 0
    for group in groups.split(","):
        raws = sorted(Path(RAW_DIR, group).glob("*.raw.json"))
        if limit:
            raws = raws[:limit]
        Path(OUT_DIR, group).mkdir(parents=True, exist_ok=True)
        for raw_path in raws:
            name = raw_path.name[: -len(".raw.json")]
            target = Path(OUT_DIR, group, name + ".json")
            if target.exists():
                done += 1
                continue
            try:
                raw = json.loads(raw_path.read_text())["raw_output"]
                page_results = raw.get("page_results", raw.get("pages", []))
                pdf = Path(BENCH_DIR, "data", "docs", group, name + ".pdf")
                doc = pymupdf.open(pdf)
                pages = []
                for p in page_results:
                    idx = p["page_index"]
                    charts = []
                    boxes = chart_blocks(p["raw_text"])
                    if boxes and idx < len(doc):
                        page = doc[idx]
                        pix = page.get_pixmap(matrix=pymupdf.Matrix(DPI / 72, DPI / 72), alpha=False)
                        full = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        W, H = full.size
                        for bb in boxes:
                            x0, y0, x1, y1 = (bb[0] / 1000 * W, bb[1] / 1000 * H, bb[2] / 1000 * W, bb[3] / 1000 * H)
                            pw, ph = PAD * (x1 - x0), PAD * (y1 - y0)
                            crop = full.crop((max(0, int(x0 - pw)), max(0, int(y0 - ph)), min(W, int(x1 + pw)), min(H, int(y1 + ph))))
                            if crop.width < 16 or crop.height < 16:
                                continue
                            t0 = time.time()
                            try:
                                text = recognize(crop)
                            except Exception as e:  # noqa: BLE001
                                text = ""
                                print(f"crop failed {name} p{idx}: {e}", flush=True)
                            charts.append({"bbox": bb, "text": text, "ms": int((time.time() - t0) * 1000)})
                            n_charts += 1
                    pages.append({"charts": charts})
                doc.close()
                target.write_text(json.dumps({"model": MODEL, "prompt": PROMPT, "dpi": DPI, "pad": PAD, "pages": pages}))
                done += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAILED {name}: {e}", flush=True)
            if (done + failed) % 50 == 0:
                print(f"progress {done} done, {failed} failed, {n_charts} charts", flush=True)
    stop.set()
    vol.commit()
    return f"done={done} failed={failed} charts={n_charts}"


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=1, memory=2048, timeout=900)
def pack() -> str:
    import tarfile

    src = Path(OUT_DIR)
    tgz = Path(BENCH_DIR, "chart_vlm.tgz")
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(src, arcname="chart_vlm")
    vol.commit()
    return f"{tgz.relative_to(VOL_PATH)} {tgz.stat().st_size / 1e6:.1f}MB"


@app.local_entrypoint()
def main(mode: str = "dump", groups: str = "chart", limit: int = 0):
    print(dump.remote(groups, limit) if mode == "dump" else pack.remote())
