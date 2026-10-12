"""Fuse PP-DocLayout-V3 detections with the hybrid blocks of a page.

The detector is the authority on element *extent and class* (it is trained for
exactly that; the VLM merges neighbours and the GT's Text-vs-Section decision is
contextual); the blocks keep the *content*. Per detector box, three cases:

- exactly one block lies inside it → the block takes the detector's class
  (and the detector's box for pictures / tables / formulas);
- one block spans several detector boxes → the block is split, its text cut at the
  text-layer lines inside each box (VLM characters kept);
- several blocks lie inside one box → ours are finer; keep them, relabel to the
  detector's class only when it is not a text class.

Pictures the detector finds with no block of ours become empty picture blocks (the
layout metric scores their localization).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from docparse.hybrid import _lines_in_bbox, transfer_styles
from docparse.textlayer import TextLayerPage

logger = logging.getLogger(__name__)

MIN_SCORE = 0.45

# PP-DocLayout(-V3) class -> our block label (LightOnOCR vocabulary)
DET_LABEL: dict[str, str] = {
    "doc_title": "title", "paragraph_title": "section_header",
    "text": "text", "abstract": "text", "content": "text", "reference": "text", "reference_content": "text",
    "aside_text": "aside_text", "vertical_text": "text", "algorithm": "text", "list": "list",
    "image": "image", "figure": "image", "seal": "image", "chart": "chart",
    "table": "table", "formula": "formula", "display_formula": "formula", "inline_formula": "formula",
    "formula_number": "text",
    "figure_title": "caption", "table_title": "caption", "chart_title": "caption",
    "footnote": "footnote", "vision_footnote": "footnote",
    "header": "header", "header_image": "header_image", "footer": "footer", "footer_image": "footer_image",
    "number": "page_number",
    # Docling heron / Canonical17 names (docling-layout-heron)
    "Caption": "caption", "Footnote": "footnote", "Formula": "formula", "List-item": "list",
    "Page-footer": "footer", "Page-header": "header", "Picture": "image", "Section-header": "section_header",
    "Table": "table", "Text": "text", "Title": "title", "Document Index": "text", "Code": "text",
    "Checkbox-Selected": "text", "Checkbox-Unselected": "text", "Form": "text", "Key-Value Region": "text",
    # the same classes as emitted by the transformers checkpoint's id2label
    "picture": "image", "section_header": "section_header", "page_header": "header", "page_footer": "footer",
    "list_item": "list", "document_index": "text", "code": "text", "checkbox_selected": "text",
    "checkbox_unselected": "text", "form": "text", "key_value_region": "text",
}
TEXTUAL = {"title", "section_header", "text", "aside_text", "list", "caption", "footnote", "header", "footer", "page_number"}
VISUAL = {"image", "chart", "table", "formula", "header_image", "footer_image"}


def load_detections(source_pdf: str, layout_dir: str | None = None) -> dict[str, Any] | None:
    """Per-page detections dumped by serve/modal_layout_dump.py, if present."""
    root = Path(layout_dir or os.getenv("DOCPARSE_LAYOUT_DIR", "output/layout_heron"))
    p = Path(source_pdf)
    cand = root / p.parent.name / (p.stem + ".json")
    if not cand.exists():
        return None
    try:
        return json.loads(cand.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("bad detections file %s: %s", cand, e)
        return None


def _det_boxes(page_dets: dict[str, Any]) -> list[dict[str, Any]]:
    w, h = float(page_dets["width"]), float(page_dets["height"])
    out = []
    for p in page_dets["predictions"]:
        if p["score"] < MIN_SCORE:
            continue
        lab = DET_LABEL.get(p.get("raw_label") or p["label"]) or DET_LABEL.get(p["label"])
        if not lab:
            continue
        x1, y1, x2, y2 = p["bbox"]
        out.append({"bbox": [x1 / w * 1000, y1 / h * 1000, x2 / w * 1000, y2 / h * 1000], "label": lab, "score": p["score"], "order": p.get("order", len(out))})
    # PP-DocLayoutV3 predicts reading order in the decoder; the metric scores it.
    out.sort(key=lambda d: d["order"])
    return out


def _inter(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(a: list[float]) -> float:
    return max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def fuse(blocks: list[dict[str, Any]], page_dets: dict[str, Any], tl: TextLayerPage | None) -> list[dict[str, Any]]:
    dets = _det_boxes(page_dets)
    if not dets:
        return blocks
    located = [(i, b) for i, b in enumerate(blocks) if b.get("bbox") is not None]
    # block -> dets it substantially overlaps (fraction of block area inside det)
    cover: dict[int, list[int]] = {}
    for i, b in located:
        cover[i] = [j for j, d in enumerate(dets) if _inter(b["bbox"], d["bbox"]) / _area(b["bbox"]) >= 0.5]
    members: dict[int, list[int]] = {j: [] for j in range(len(dets))}
    for i, js in cover.items():
        for j in js:
            members[j].append(i)

    out: list[dict[str, Any] | None] = [dict(b) for b in blocks]
    used_dets: set[int] = set()
    for j, d in enumerate(dets):
        ms = members[j]
        if len(ms) == 1:
            i = ms[0]
            b = out[i]
            if b is None or len(cover.get(i, [])) != 1:
                continue
            frac_d = _inter(b["bbox"], d["bbox"]) / _area(d["bbox"])
            if d["label"] in VISUAL and b["label"] in ("image", "chart", "table", "formula", "header_image", "footer_image"):
                b["bbox"] = list(d["bbox"])
                if d["label"] != "table" and b["label"] not in ("chart", "table"):
                    b["label"] = d["label"]
                used_dets.add(j)
            elif d["label"] in TEXTUAL and b["label"] in TEXTUAL and frac_d >= 0.3:
                new_label = _relabel(b, d["label"], tl)
                if new_label != b["label"]:
                    b["label"] = new_label
                    if new_label in ("title", "section_header") and not re.match(r"^\s*#", b["text"]):
                        b["text"] = _retitle(b["text"], new_label)
                used_dets.add(j)
        elif len(ms) > 1:
            used_dets.add(j)
    # blocks spanning several detector boxes -> split by text-layer lines
    for i, b in located:
        js = [j for j in cover.get(i, []) if dets[j]["label"] in TEXTUAL]
        if len(js) < 2 or out[i] is None or b["label"] not in TEXTUAL or tl is None or not tl.has_text or tl.scanned:
            continue
        pieces = _split_by_dets(b, [dets[j] for j in sorted(js, key=lambda j: (dets[j]["bbox"][1], dets[j]["bbox"][0]))], tl)
        if pieces:
            out[i] = None
            out.extend(pieces)
            used_dets.update(js)
    # detector pictures with nothing of ours inside
    extras: list[dict[str, Any]] = []
    for j, d in enumerate(dets):
        if j in used_dets or members[j]:
            continue
        if d["label"] in ("image", "chart") and _area(d["bbox"]) >= 400:
            if not any(_inter(d["bbox"], b["bbox"]) / _area(d["bbox"]) > 0.5 for _, b in located):
                extras.append({"label": d["label"], "continuation": False, "bbox": list(d["bbox"]), "text": "", "source": "detector"})
    result = [b for b in out if b is not None] + extras
    return result


HEADING = {"title", "section_header"}
FURNITURE = {"header", "footer", "page_number", "header_image", "footer_image"}


def _emphasized(block: dict[str, Any], tl: TextLayerPage | None) -> bool:
    """DocLayNet heading test on the text layer: own short line(s), bold or larger."""
    if tl is None or not tl.has_text or tl.scanned:
        return False
    lines = _lines_in_bbox(tl, block["bbox"])
    spans = [s for ln in lines for s in ln.spans if s.text.strip()]
    if not spans or len(lines) > 3 or len(" ".join(ln.text for ln in lines)) > 120:
        return False
    bold = sum(s.bold for s in spans) / len(spans) >= 0.8
    big = all(s.size >= tl.body_size + 0.9 for s in spans)
    return bold or big


def _heading_signature(bbox: list[float], tl: TextLayerPage | None) -> bool:
    """Single line, <= 80 chars, every span bold or larger than body, no terminal period."""
    if tl is None or not tl.has_text or tl.scanned:
        return False
    lines = _lines_in_bbox(tl, bbox)
    if len(lines) != 1:
        return False
    text = lines[0].text.strip()
    spans = [s for s in lines[0].spans if s.text.strip()]
    if not spans or len(text) > 80 or text.endswith((".", ":", ";", ",")) or sum(c.isdigit() for c in text) > len(text) / 2:
        return False
    bold = all(s.bold for s in spans)
    big = all(s.size >= tl.body_size + 0.9 for s in spans)
    return bold or big


def _relabel(block: dict[str, Any], det_label: str, tl: TextLayerPage | None) -> str:
    """Detector class vs VLM class. Headings are never demoted by the detector's
    generic `text`; promotion to a heading needs the text layer to agree; furniture
    and caption/footnote labels come from the detector."""
    cur = block["label"]
    if det_label == cur:
        return cur
    if det_label in HEADING:
        return det_label if (cur in HEADING or _emphasized(block, tl)) else cur
    if cur in HEADING:
        return cur
    if det_label in FURNITURE or det_label in ("caption", "footnote"):
        return det_label
    if det_label in ("text", "list", "aside_text") and cur in FURNITURE:
        return cur
    return det_label


def _retitle(text: str, label: str) -> str:
    plain = re.sub(r"^\s*#{1,6}\s*", "", text.strip())
    if label == "title":
        return f"# {plain}"
    if label == "section_header":
        return f"## {plain}"
    return plain


def _split_by_dets(block: dict[str, Any], dets: list[dict[str, Any]], tl: TextLayerPage) -> list[dict[str, Any]]:
    vlm = block["text"]
    plain = re.sub(r"^\s*#{1,6}\s*", "", vlm)
    pieces: list[tuple[int, dict[str, Any], list]] = []
    lo = 0
    for d in dets:
        lines = _lines_in_bbox(tl, d["bbox"])
        if not lines:
            return []
        q = re.sub(r"\s+", " ", " ".join(ln.text for ln in lines)).strip()
        if len(q) < 3:
            return []
        al = fuzz.partial_ratio_alignment(q, plain[lo:])
        if al is None or al.score < 80:
            return []
        pieces.append((lo + al.dest_start, d, lines))
        lo = lo + al.dest_end
    out: list[dict[str, Any]] = []
    for k, (start, d, lines) in enumerate(pieces):
        a = start if k > 0 else 0
        b = pieces[k + 1][0] if k + 1 < len(pieces) else len(plain)
        seg = plain[a:b].strip()
        if not seg:
            continue
        nb = dict(block)
        nb.update(label=d["label"], bbox=list(d["bbox"]), text=_retitle(transfer_styles(seg, lines), d["label"]) if d["label"] in ("title", "section_header") else transfer_styles(seg, lines), source="fused-split")
        out.append(nb)
    return out


def detector_items(
    blocks: list[dict[str, Any]], page_dets: dict[str, Any], tl: TextLayerPage | None
) -> list[dict[str, Any]]:
    """Layout elements for the grounding metric, detector-primary: one item per
    detector box (class from the detector, heading class only if the text layer or
    the VLM agree), text attributed from the text-layer lines inside the box (or the
    overlapping VLM block). Our blocks are added only where the detector has nothing
    (lists, elements below its confidence). The page markdown is not touched."""
    dets = _det_boxes(page_dets)
    items: list[dict[str, Any]] = []
    located = [b for b in blocks if b.get("bbox") is not None]
    for d in dets:
        label = d["label"]
        over = [b for b in located if _inter(b["bbox"], d["bbox"]) / _area(d["bbox"]) >= 0.3]
        inside = [b for b in located if _inter(b["bbox"], d["bbox"]) / _area(b["bbox"]) >= 0.6 and b["label"] in TEXTUAL]
        # The detector has no list-item class and merges stacked short elements: when
        # several of our (finer) blocks sit inside one detector text box, emit ours.
        # ParseBench localization fails only when a box exceeds ~5x an element it
        # covers; splitting smaller merges costs more than it gains.
        if (
            label in ("text", "list") and len(inside) >= 2
            and sum(_area(b["bbox"]) for b in inside) >= 0.5 * _area(d["bbox"])
            and _area(d["bbox"]) >= 4.0 * min(_area(b["bbox"]) for b in inside)
        ):
            for b in inside:
                nb = dict(b)
                nb["_key"] = (located.index(b), 0)
                items.append(nb)
            continue
        if label == "text":
            # A VLM heading that the detector folded into generic text, or a line
            # with DocLayNet's heading signature (own short line, bold / larger).
            heads = [b for b in inside if b["label"] in HEADING]
            if heads and len(inside) == 1 and len(re.sub(r"\s+", " ", heads[0]["text"])) <= 120:
                label = heads[0]["label"]
            elif len(inside) <= 1 and _heading_signature(d["bbox"], tl):
                label = "section_header"
        text = ""
        if tl is not None and tl.has_text and not tl.scanned and label in TEXTUAL:
            text = " ".join(ln.text.strip() for ln in _lines_in_bbox(tl, d["bbox"]))
        if not text and over:
            best = max(over, key=lambda b: _inter(b["bbox"], d["bbox"]))
            text = re.sub(r"<[^>]+>", " ", best["text"]) if best["label"] in ("table", "chart") else best["text"]
        # Order key: the VLM's reading order (scored higher than the detector's) via
        # the block each box overlaps most, then the detector order within a block.
        vlm_pos = max(range(len(located)), key=lambda k: _inter(located[k]["bbox"], d["bbox"]), default=None) if over else None
        key = (vlm_pos if vlm_pos is not None else 10_000, d["order"])
        items.append({"label": label, "bbox": list(d["bbox"]), "text": text, "source": "detector", "_key": key})
    items.sort(key=lambda it: it["_key"])
    for it in items:
        it.pop("_key", None)
    # our blocks the detector missed, inserted by vertical position
    emitted = {tuple(b["bbox"]) for b in items if b.get("source") != "detector"}
    for b in located:
        if tuple(b["bbox"]) in emitted or (not b["text"] and b["label"] in TEXTUAL):
            continue
        if any(_inter(b["bbox"], d["bbox"]) / _area(b["bbox"]) >= 0.5 for d in dets):
            continue
        cy = (b["bbox"][1] + b["bbox"][3]) / 2
        pos = next((k for k, it in enumerate(items) if (it["bbox"][1] + it["bbox"][3]) / 2 > cy), len(items))
        items.insert(pos, b)
    return items
