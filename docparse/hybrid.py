"""Reconcile VLM grounding blocks with the PDF text layer.

The VLM (LightOnOCR-3 grounding mode) supplies structure: block order, labels,
bboxes, table/chart HTML. On pages with a usable text layer, each text-like block
is re-rendered from the text-layer spans inside its bbox, which restores exact
characters and the styling the VLM cannot see (bold, underline, strike, highlight,
heading levels). Blocks whose text-layer content does not resemble the VLM text
(OCR layers from scanners, mislocated boxes) keep the VLM text.
"""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz

from docparse.textlayer import Line, Span, TextLayerPage

TEXT_LABELS = {"text", "title", "section_header", "list", "caption", "footnote", "header", "footer", "aside_text", "page_number"}
MIN_SIMILARITY = 55  # rapidfuzz token_set_ratio between VLM text and text-layer text
BBOX_PAD = 0.004  # fraction of page, tolerance when testing span centres against the block bbox


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*_~`#<>/\\]", "", s)).strip().lower()


def _lines_in_bbox(tl: TextLayerPage, bbox1000: list[float]) -> list[Line]:
    x0 = bbox1000[0] / 1000 * tl.width - BBOX_PAD * tl.width
    y0 = bbox1000[1] / 1000 * tl.height - BBOX_PAD * tl.height
    x1 = bbox1000[2] / 1000 * tl.width + BBOX_PAD * tl.width
    y1 = bbox1000[3] / 1000 * tl.height + BBOX_PAD * tl.height
    picked: list[Line] = []
    for line in tl.lines:
        cx = (line.bbox[0] + line.bbox[2]) / 2
        cy = (line.bbox[1] + line.bbox[3]) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1 and line.text.strip():
            picked.append(line)
    picked.sort(key=lambda ln: (round(ln.bbox[1] / 3), ln.bbox[0]))
    return picked


def _style_key(s: Span) -> tuple[bool, bool, bool, bool]:
    return (s.bold, s.underline, s.strike, s.mark)


def _wrap(text: str, key: tuple[bool, bool, bool, bool]) -> str:
    bold, underline, strike, mark = key
    lead = re.match(r"^\s*", text).group(0)
    trail = re.search(r"\s*$", text).group(0)
    core = text.strip()
    if not core:
        return text
    if mark:
        core = f"<mark>{core}</mark>"
    if strike:
        core = f"~~{core}~~"
    if underline:
        core = f"<u>{core}</u>"
    if bold:
        core = f"**{core}**"
    return f"{lead}{core}{trail}"


def render_line(line: Line) -> str:
    """Line text with styling markers; adjacent spans of equal style are merged."""
    out: list[str] = []
    buf = ""
    key: tuple[bool, bool, bool, bool] | None = None
    for s in line.spans:
        k = _style_key(s)
        if key is None or k == key:
            buf += s.text
            key = k
        else:
            out.append(_wrap(buf, key))
            buf, key = s.text, k
    if key is not None:
        out.append(_wrap(buf, key))
    return "".join(out).strip()


def _heading_level(lines: list[Line], tl: TextLayerPage) -> int:
    """1 for the largest font on the page, growing with smaller sizes; bold body size → 3."""
    sizes = sorted({s.size for ln in lines for s in ln.spans if s.text.strip()}, reverse=True)
    if not sizes:
        return 2
    page_sizes = sorted({s.size for ln in tl.lines for s in ln.spans if s.size > tl.body_size + 0.4}, reverse=True)
    top = sizes[0]
    if top > tl.body_size + 0.4 and top in page_sizes:
        return min(6, 1 + page_sizes.index(top))
    return min(6, 1 + len(page_sizes))


def _paragraphs(lines: list[Line]) -> list[list[Line]]:
    """Split consecutive lines on vertical gaps clearly larger than the running
    line pitch (a blank line between paragraphs); single lines are their own group."""
    if len(lines) < 2:
        return [lines]
    heights = sorted(ln.bbox[3] - ln.bbox[1] for ln in lines)
    line_h = heights[len(heights) // 2]
    gaps = [max(0.0, b.bbox[1] - a.bbox[3]) for a, b in zip(lines, lines[1:])]
    nonzero = sorted(g for g in gaps if g > 0.5)
    base_gap = nonzero[len(nonzero) // 2] if nonzero else 0.0
    groups: list[list[Line]] = [[lines[0]]]
    for gap, ln in zip(gaps, lines[1:]):
        is_break = gap > 0.45 * line_h and (base_gap == 0.0 or gap > 1.7 * base_gap)
        if is_break:
            groups.append([ln])
        else:
            groups[-1].append(ln)
    return groups


def _lines_bbox1000(lines: list[Line], tl: TextLayerPage) -> list[float]:
    x0 = min(ln.bbox[0] for ln in lines)
    y0 = min(ln.bbox[1] for ln in lines)
    x1 = max(ln.bbox[2] for ln in lines)
    y1 = max(ln.bbox[3] for ln in lines)
    return [x0 / tl.width * 1000, y0 / tl.height * 1000, x1 / tl.width * 1000, y1 / tl.height * 1000]


def _render_group(label: str, lines: list[Line], tl: TextLayerPage) -> str:
    rendered = [r for r in (render_line(ln) for ln in lines) if r]
    if label == "title":
        # One heading per text-layer line: a title block often stacks several
        # headings (company / programme / section), each scored separately.
        level = _heading_level(lines, tl)
        return "\n\n".join(f"{'#' * level} {_strip_outer_bold(r)}" for r in rendered)
    return " ".join(rendered) if label != "list" else "\n".join(rendered)


def render_blocks(block: dict[str, Any], tl: TextLayerPage) -> list[dict[str, Any]] | None:
    """Text-layer rendering of a VLM block as one or more blocks (split on paragraph
    gaps, bboxes tightened to the text), or None to keep the VLM block."""
    if block["label"] not in TEXT_LABELS or block.get("bbox") is None:
        return None
    lines = _lines_in_bbox(tl, block["bbox"])
    if not lines:
        return None
    layer_text = " ".join(ln.text for ln in lines)
    if fuzz.token_set_ratio(_norm(layer_text), _norm(block["text"])) < MIN_SIMILARITY:
        return None
    groups = _paragraphs(lines) if block["label"] in ("text", "list", "footnote") else [lines]
    out: list[dict[str, Any]] = []
    for g in groups:
        for label, sub in _split_items_and_headings(block["label"], g, tl):
            text = _render_group(label, sub, tl)
            if not text:
                continue
            nb = dict(block)
            nb.update(label=label, text=text, bbox=_lines_bbox1000(sub, tl), source="textlayer")
            out.append(nb)
    return out or None


_LIST_MARKER_RE = re.compile(r"^\s*(?:[\u2022\u25aa\u25cf\u2013\u2014\-\*\u25a0\u2023\u2043\u25e6]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|[ivxIVX]{1,5}[.)])\s+")


def _is_heading_line(ln: Line, tl: TextLayerPage) -> bool:
    spans = [s for s in ln.spans if s.text.strip()]
    if not spans or len(ln.text.strip()) > 120:
        return False
    all_bold = all(s.bold for s in spans)
    bigger = all(s.size >= tl.body_size + 0.9 for s in spans)
    return all_bold or bigger


def _split_items_and_headings(label: str, lines: list[Line], tl: TextLayerPage) -> list[tuple[str, list[Line]]]:
    """ParseBench treats each list item and each bold/larger lead line as its own
    element; VLM blocks routinely merge them."""
    if label not in ("text", "list") or len(lines) < 2:
        return [(label, lines)]
    n_markers = sum(bool(_LIST_MARKER_RE.match(ln.text)) for ln in lines)
    out: list[tuple[str, list[Line]]] = []
    cur: list[Line] = []
    for i, ln in enumerate(lines):
        starts_item = n_markers >= 2 and _LIST_MARKER_RE.match(ln.text) is not None
        heading = i == 0 and len(lines) >= 2 and _is_heading_line(ln, tl) and not _is_heading_line(lines[1], tl)
        if heading:
            out.append(("section_header", [ln]))
            continue
        if starts_item and cur:
            out.append(("list" if label == "list" else "text", cur))
            cur = []
        cur.append(ln)
    if cur:
        out.append(("list" if label == "list" else "text", cur))
    return out


def render_block(block: dict[str, Any], tl: TextLayerPage) -> str | None:
    """Text-layer rendering of a VLM block's text, or None to keep the VLM text."""
    blocks = render_blocks(block, tl)
    return "\n\n".join(b["text"] for b in blocks) if blocks else None


def _strip_outer_bold(text: str) -> str:
    """Heading markers already imply emphasis; ``# **x**`` double-scores nothing."""
    return re.sub(r"^\*\*(.*)\*\*$", r"\1", text)


def reconcile_page(blocks: list[dict[str, Any]], tl: TextLayerPage | None) -> list[dict[str, Any]]:
    """Return a copy of ``blocks`` with ``text`` replaced from the text layer where it aligns."""
    if tl is None or not tl.has_text:
        return blocks
    out: list[dict[str, Any]] = []
    covered: set[int] = set()
    for b in blocks:
        rendered = render_blocks(b, tl)
        if b.get("bbox") is not None:
            covered.update(id(ln) for ln in _lines_in_bbox(tl, b["bbox"]))
        out.extend(rendered if rendered is not None else [dict(b)])
    return _insert_orphans(out, tl, covered)


def _orphan_label(ln: Line, tl: TextLayerPage) -> str:
    cy = (ln.bbox[1] + ln.bbox[3]) / 2
    text = ln.text.strip()
    if re.fullmatch(r"[-–—\s]*(page\s*)?\d{1,4}([\s/]+(of\s+)?\d{1,4})?[-–—\s]*", text, flags=re.I):
        return "page_number"
    if cy < 0.07 * tl.height:
        return "header"
    if cy > 0.93 * tl.height:
        return "footer"
    return "text"


def _insert_orphans(blocks: list[dict[str, Any]], tl: TextLayerPage, covered: set[int]) -> list[dict[str, Any]]:
    """Text-layer lines outside every VLM bbox are real page text the VLM skipped
    (bullets, marginal marks, stray headings). Add them as text blocks, placed by
    vertical position among the located blocks."""
    orphans = [ln for ln in tl.lines if id(ln) not in covered and (len(ln.text.strip()) >= 2 or ln.text.strip().isdigit())]
    if not orphans:
        return blocks
    located = [(b["bbox"][1] / 1000 * tl.height, i) for i, b in enumerate(blocks) if b.get("bbox") is not None]
    inserts: list[tuple[int, dict[str, Any]]] = []
    for ln in orphans:
        cy = (ln.bbox[1] + ln.bbox[3]) / 2
        pos = next((i for y, i in located if y > cy), len(blocks))
        bbox = [
            ln.bbox[0] / tl.width * 1000, ln.bbox[1] / tl.height * 1000,
            ln.bbox[2] / tl.width * 1000, ln.bbox[3] / tl.height * 1000,
        ]
        inserts.append((pos, {"label": _orphan_label(ln, tl), "continuation": False, "bbox": bbox, "text": render_line(ln), "source": "orphan"}))
    merged: list[dict[str, Any]] = []
    for i, b in enumerate(blocks):
        merged.extend(nb for pos, nb in inserts if pos == i)
        merged.append(b)
    merged.extend(nb for pos, nb in inserts if pos >= len(blocks))
    return merged
