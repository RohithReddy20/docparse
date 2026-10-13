"""DocLayNet-trained Section/Text classifier; dependency-free forest inference."""

from __future__ import annotations

import json
import math
import os
import re
import struct
from functools import lru_cache
from pathlib import Path

from docparse.textlayer import Line, TextLayerPage

BASE_FEATURES = (
    "lines", "chars", "words", "bold", "italic", "size_mean", "size_min", "size_max",
    "uppercase", "digits", "sentence_end", "colon_end", "numbered", "bullet",
    "left", "top", "width", "height", "gap_above", "gap_below", "size_above", "size_below",
)
FEATURES = BASE_FEATURES + ("baseline_heading", "detector_heading", "detector_score")
VERSION = 2
HEADING = {"title", "section_header"}


def _overlap(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _neighbors(tl: TextLayerPage, box: list[float], picked: list[Line]) -> tuple[Line | None, Line | None]:
    rest = [ln for ln in tl.lines if ln not in picked and ln.text.strip()
            and min(ln.bbox[2], box[2]) - max(ln.bbox[0], box[0]) > 0.3 * min(ln.bbox[2] - ln.bbox[0], box[2] - box[0])]
    above = [ln for ln in rest if ln.bbox[3] <= box[1] + 1]
    below = [ln for ln in rest if ln.bbox[1] >= box[3] - 1]
    return max(above, key=lambda ln: ln.bbox[3], default=None), min(below, key=lambda ln: ln.bbox[1], default=None)


def features(tl: TextLayerPage | None, bbox1000: list[float]) -> list[float] | None:
    if tl is None or tl.scanned or tl.body_size <= 0:
        return None
    box = [bbox1000[i] / 1000 * (tl.width if i % 2 == 0 else tl.height) for i in range(4)]
    picked: list[Line] = []
    spans = []
    for ln in tl.lines:
        selected = [s for s in ln.spans if s.text.strip() and _overlap(s.bbox, box) >= 0.5 * max(1e-6, (s.bbox[2] - s.bbox[0]) * (s.bbox[3] - s.bbox[1]))]
        if selected:
            picked.append(ln)
            spans.extend(selected)
    if not spans:
        return None
    text = " ".join(s.text.strip() for s in spans)
    weights = [len(s.text.strip()) for s in spans]
    total = sum(weights)
    sizes = [s.size / tl.body_size for s in spans]
    letters = [c for c in text if c.isalpha()]
    line_height = max(1.0, sum(ln.bbox[3] - ln.bbox[1] for ln in picked) / len(picked))
    above, below = _neighbors(tl, box, picked)
    gap_above = min(10.0, max(0.0, box[1] - above.bbox[3]) / line_height) if above else 10.0
    gap_below = min(10.0, max(0.0, below.bbox[1] - box[3]) / line_height) if below else 10.0
    mean_size = lambda ln: sum(s.size for s in ln.spans) / max(1, len(ln.spans)) / tl.body_size if ln else 0.0
    values = [
        len(picked), math.log1p(total), math.log1p(len(text.split())),
        sum(w * s.bold for w, s in zip(weights, spans)) / total,
        sum(w * s.italic for w, s in zip(weights, spans)) / total,
        sum(w * v for w, v in zip(weights, sizes)) / total, min(sizes), max(sizes),
        sum(c.isupper() for c in letters) / max(1, len(letters)),
        sum(c.isdigit() for c in text) / len(text), text.endswith((".", "!", "?", ";", ",")), text.endswith(":"),
        bool(re.match(r"^(?:\d+(?:\.\d+)*[.)]?|[IVX]+[.)])\s", text)), bool(re.match(r"^[•●▪◦*-]\s", text)),
        bbox1000[0] / 1000, bbox1000[1] / 1000,
        (bbox1000[2] - bbox1000[0]) / 1000, (bbox1000[3] - bbox1000[1]) / 1000,
        gap_above, gap_below, mean_size(above), mean_size(below),
    ]
    return [float(v) for v in values]


def probability(model: dict, values: list[float]) -> float:
    values = [struct.unpack("f", struct.pack("f", v))[0] for v in values]
    predictions = []
    for tree in model["trees"]:
        node = tree[0]
        while node[0] >= 0:
            feature, threshold, left, right, _ = node
            node = tree[left if values[feature] <= threshold else right]
        predictions.append(node[4])
    return sum(predictions) / len(predictions)


def context_features(
    tl: TextLayerPage | None, bbox1000: list[float], baseline_label: str, detector_label: str, detector_score: float
) -> list[float] | None:
    values = features(tl, bbox1000)
    if values is None:
        return None
    return values + [float(baseline_label in HEADING), float(detector_label in HEADING), float(detector_score)]


@lru_cache(maxsize=2)
def load_model(path: str) -> dict:
    model = json.loads(Path(path).read_text())
    if model.get("version") != VERSION or model.get("features") != list(FEATURES) or not model.get("trees"):
        raise ValueError("Incompatible Section/Text model")
    if not 0 <= model["demote_threshold"] <= 0.1 < 0.9 <= model["promote_threshold"] <= 1:
        raise ValueError("Section/Text model must use conservative thresholds")
    return model


def rerank(
    label: str, bbox1000: list[float], tl: TextLayerPage | None, detector_label: str | None = None, detector_score: float = 0.0
) -> str:
    path = os.getenv("DOCPARSE_SECTION_TEXT_MODEL")
    if not path or label not in HEADING | {"text", "list"}:
        return label
    values = context_features(tl, bbox1000, label, detector_label or label, detector_score)
    if values is None:
        return label
    model = load_model(path)
    p = probability(model, values)
    if model.get("demote_enabled", True) and label in HEADING and p <= model["demote_threshold"]:
        return "text"
    if model.get("promote_enabled", True) and label in {"text", "list"} and p >= model["promote_threshold"]:
        return "section_header"
    return label
