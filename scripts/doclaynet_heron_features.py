"""Generate detector-aware training examples on Kaggle GPUs, using DocLayNet only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

from docparse.fusion import DET_LABEL, HEADING, _heading_signature, text_digit_heavy
from docparse.section_text import context_features
from docparse.textlayer import extract_page

MODEL = "docling-project/docling-layout-heron"
SEED = 13
TARGETS = {"text", "section_header"}


def _iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / max(union, 1e-6)


def _target(box, annotations, classes):
    best, label = 0.0, None
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        overlap = _iou(box, [x, y, x + w, y + h])
        if overlap > best:
            best, label = overlap, classes[annotation["category_id"]]
    if best < 0.5:
        return None
    if label == "Section-header":
        return 1
    if label == "Text":
        return 0
    return None


def _detect(engine, pages):
    import torch

    processor, model, device = engine
    images = [p[1] for p in pages]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    sizes = torch.tensor([[im.height, im.width] for im in images], device=device)
    results = processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=0.25)
    predictions = []
    for result in results:
        boxes = []
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            raw = model.config.id2label[int(label_id)]
            label = DET_LABEL.get(raw)
            threshold = 0.45 if label in HEADING else 0.5
            if label in TARGETS and float(score) >= threshold:
                boxes.append((label, float(score), box.cpu().tolist()))
        predictions.append(boxes)
    return predictions


def collect(coco, pdf_root, png_root, limit, forbidden, engines, batch_size, progress_path=None):
    print("Loading annotations:", coco, flush=True)
    data = json.loads(coco.read_text())
    classes = {c["id"]: c["name"] for c in data["categories"]}
    annotations = defaultdict(list)
    for annotation in data["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    images = [im for im in data["images"] if im.get("precedence", 0) == 0 and im["doc_name"] not in forbidden]
    random.Random(SEED).shuffle(images)
    x, y, documents, counts = [], [], set(), Counter()
    start = time.monotonic()
    chunk = []
    with ThreadPoolExecutor(max_workers=len(engines)) as pool:
        for im in images:
            if counts["pages"] + len(chunk) >= limit:
                break
            stem = Path(im["file_name"]).stem
            pdf, png = pdf_root / (stem + ".pdf"), png_root / im["file_name"]
            if not pdf.exists() or not png.exists():
                counts["missing"] += 1
                continue
            with pymupdf.open(pdf) as doc:
                tl = extract_page(doc[0])
            if tl.scanned or not tl.lines:
                counts["no_text_page"] += 1
                continue
            with Image.open(png) as image:
                image = image.convert("RGB")
            chunk.append((im, image, tl))
            if len(chunk) == batch_size * len(engines):
                _consume(chunk, engines, batch_size, pool, annotations, classes, x, y, documents, counts)
                chunk = []
                if progress_path is not None:
                    progress_path.write_text(json.dumps({"split": coco.stem, "counts": dict(counts),
                                                         "elapsed_seconds": time.monotonic() - start}))
                if counts["pages"] == batch_size * len(engines) or counts["pages"] % 256 == 0:
                    print(coco.stem, counts["pages"], "pages", len(y), "matched predictions", round(time.monotonic() - start), "seconds", flush=True)
        if chunk:
            _consume(chunk, engines, batch_size, pool, annotations, classes, x, y, documents, counts)
    if progress_path is not None:
        progress_path.write_text(json.dumps({"split": coco.stem, "counts": dict(counts),
                                             "elapsed_seconds": time.monotonic() - start}))
    if counts["pages"] < min(limit, 100):
        raise RuntimeError(f"Insufficient training pages: {dict(counts)}")
    print(coco.stem, dict(counts), "elapsed", round(time.monotonic() - start), flush=True)
    if not y:
        raise RuntimeError(f"No matched text/heading boxes: {dict(counts)}")
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64), {
        "counts": dict(counts), "documents": sorted(documents), "elapsed_seconds": time.monotonic() - start,
        "annotation_sha256": hashlib.sha256(coco.read_bytes()).hexdigest()}


def _consume(pages, engines, batch_size, pool, annotations, classes, x, y, documents, counts):
    batches = [pages[i:i + batch_size] for i in range(0, len(pages), batch_size)]
    futures = [pool.submit(_detect, engine, batch) for engine, batch in zip(engines, batches)]
    predictions = [items for future in futures for items in future.result()]
    if len(predictions) != len(pages):
        raise RuntimeError("Detector did not return one prediction list per page")
    for (im, image, tl), boxes in zip(pages, predictions):
        counts["pages"] += 1
        documents.add(im["doc_name"])
        for detector_label, score, box in boxes:
            target = _target(box, annotations[im["id"]], classes)
            if target is None:
                counts["unmatched_or_other_class"] += 1
                continue
            bbox = [box[i] / (image.width if i % 2 == 0 else image.height) * 1000 for i in range(4)]
            baseline = detector_label
            if baseline in HEADING and text_digit_heavy(bbox, tl):
                baseline = "text"
            if baseline == "text" and _heading_signature(bbox, tl):
                baseline = "section_header"
            values = context_features(tl, bbox, baseline, detector_label, score)
            if values is None:
                counts["no_text_box"] += 1
                continue
            x.append(values)
            y.append(target)
            counts["Heading" if target else "Text"] += 1
            counts["baseline_correct"] += int((baseline in HEADING) == bool(target))


def main():
    import torch
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    parser = argparse.ArgumentParser()
    for name in ("coco", "pdf", "png", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-pages", type=int, default=5000)
    parser.add_argument("--validation-pages", type=int, default=1500)
    parser.add_argument("--test-pages", type=int, default=1000)
    args = parser.parse_args()
    if min(args.batch_size, args.train_pages, args.validation_pages, args.test_pages) < 1:
        parser.error("Batch size and page limits must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle T4 GPU is required for detector-aware feature generation")
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print("GPUs:", [torch.cuda.get_device_name(i) for i in range(len(devices))], flush=True)
    if any("T4" not in torch.cuda.get_device_name(i) for i in range(len(devices))):
        raise RuntimeError("Expected Kaggle T4 GPUs")
    print("Loading detector:", MODEL, flush=True)
    engines = [(RTDetrImageProcessor.from_pretrained(MODEL), RTDetrV2ForObjectDetection.from_pretrained(MODEL).to(d).eval(), d) for d in devices]
    print("Detector ready", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    forbidden = set()
    for name, filename, limit in (("train", "train.json", args.train_pages), ("validation", "val.json", args.validation_pages), ("test", "test.json", args.test_pages)):
        x, y, meta = collect(args.coco / filename, args.pdf, args.png, limit, forbidden, engines, args.batch_size,
                             args.output / "progress.json")
        forbidden.update(meta["documents"])
        meta["detector"] = MODEL
        meta["matching"] = "Highest IoU GT, IoU >= 0.5; detector and GT restricted to Text/Section-header"
        np.savez_compressed(args.output / f"{name}.npz", x=x, y=y)
        (args.output / f"{name}.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()
