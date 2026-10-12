"""Run Docling's `docling-layout-heron` (RT-DETRv2, Apache-2.0, 42.9M params; the
17 Canonical17 classes ParseBench's GT uses) over every dataset page on Modal and
save detections in the same JSON shape as serve/modal_layout_dump.py, under
/vol/bench/layout_heron/<group>/<name>.json. Labels are the canonical names
("List-item", "Section-header", ...), `raw_label` identical.

    uv run modal run --detach serve/modal_heron_dump.py
    uv run modal run serve/modal_heron_dump.py --mode pack
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

VOL_PATH = "/vol"
BENCH_DIR = f"{VOL_PATH}/bench"
DPI = 200
MODEL = "docling-project/docling-layout-heron"
# Docling's per-class acceptance thresholds (layout_postprocessor.py)
THRESH = {"Section-header": 0.45, "Title": 0.45, "Code": 0.45, "Form": 0.45, "Key-Value Region": 0.45, "Picture": 0.3}
DEFAULT_THRESH = 0.5

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "transformers>=4.50", "pillow", "pymupdf>=1.24", "huggingface_hub")
    .env({"HF_HOME": "/vol/hf"})
)
vol = modal.Volume.from_name("docparse-output", create_if_missing=True)
app = modal.App("docparse-heron-dump")


@app.function(image=image, volumes={VOL_PATH: vol}, gpu=os.environ.get("DOCPARSE_HERON_GPU", "T4"), cpu=2, memory=8192, timeout=3 * 3600)
def dump(groups: str = "chart,layout,table,text", limit: int = 0) -> str:
    import threading

    import pymupdf
    import torch
    from PIL import Image
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = RTDetrImageProcessor.from_pretrained(MODEL)
    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL).to(device).eval()
    id2label = model.config.id2label
    out_root = Path(BENCH_DIR) / "layout_heron"
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
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    inputs = proc(images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    res = proc.post_process_object_detection(outputs, target_sizes=torch.tensor([[img.height, img.width]]), threshold=0.25)[0]
                    preds = []
                    for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                        name = id2label[int(label)]
                        if float(score) < THRESH.get(name, DEFAULT_THRESH):
                            continue
                        preds.append({"bbox": [float(v) for v in box], "label": name, "raw_label": name, "score": float(score)})
                    preds.sort(key=lambda p: (p["bbox"][1], p["bbox"][0]))
                    for i, p in enumerate(preds):
                        p["order"] = i
                    pages.append({"width": img.width, "height": img.height, "predictions": preds})
                doc.close()
                target.write_text(json.dumps({"dpi": DPI, "model": MODEL, "pages": pages}))
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

    src = Path(BENCH_DIR) / "layout_heron"
    tgz = Path(BENCH_DIR) / "layout_heron.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(src, arcname="layout_heron")
    vol.commit()
    return f"{tgz.relative_to(VOL_PATH)} {tgz.stat().st_size / 1e6:.1f}MB"


@app.local_entrypoint()
def main(mode: str = "dump", groups: str = "chart,layout,table,text", limit: int = 0):
    print(dump.remote(groups, limit) if mode == "dump" else pack.remote())
