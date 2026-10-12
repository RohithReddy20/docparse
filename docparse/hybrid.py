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


def _overlap_frac(line: Line, box: tuple[float, float, float, float]) -> float:
    """Fraction of the line's area inside ``box``."""
    lx0, ly0, lx1, ly1 = line.bbox
    ix = max(0.0, min(lx1, box[2]) - max(lx0, box[0]))
    iy = max(0.0, min(ly1, box[3]) - max(ly0, box[1]))
    area = max(1e-6, (lx1 - lx0) * (ly1 - ly0))
    return ix * iy / area


def _lines_in_bbox(tl: TextLayerPage, bbox1000: list[float], claimed: set[int] | None = None) -> list[Line]:
    """Text-layer lines belonging to a VLM block: at least half the line's area lies
    inside the (slightly padded) bbox. With ``claimed``, lines already given to an
    earlier block are skipped and the picked ones are recorded, so every line is
    rendered exactly once even when VLM boxes overlap."""
    x0 = bbox1000[0] / 1000 * tl.width - BBOX_PAD * tl.width
    y0 = bbox1000[1] / 1000 * tl.height - BBOX_PAD * tl.height
    x1 = bbox1000[2] / 1000 * tl.width + BBOX_PAD * tl.width
    y1 = bbox1000[3] / 1000 * tl.height + BBOX_PAD * tl.height
    picked: list[Line] = []
    for line in tl.lines:
        if not line.text.strip() or (claimed is not None and id(line) in claimed):
            continue
        if _overlap_frac(line, (x0, y0, x1, y1)) >= 0.5:
            picked.append(line)
    if claimed is not None:
        claimed.update(id(ln) for ln in picked)
    return merge_rows(picked)


def merge_rows(lines: list[Line]) -> list[Line]:
    """PyMuPDF emits separate ``Line`` objects for spans that sit on one visual row
    (tab-separated cells, left/right footer parts, bold lead-ins). Merge rows whose
    vertical centres fall within half a line height; order left to right."""
    lines = sorted(lines, key=lambda ln: ((ln.bbox[1] + ln.bbox[3]) / 2, ln.bbox[0]))
    rows: list[Line] = []
    for ln in lines:
        if rows:
            prev = rows[-1]
            h = min(prev.bbox[3] - prev.bbox[1], ln.bbox[3] - ln.bbox[1]) or 1.0
            if abs((prev.bbox[1] + prev.bbox[3]) / 2 - (ln.bbox[1] + ln.bbox[3]) / 2) <= 0.5 * h:
                # Keep left-to-right order whichever line came first.
                left, right = (prev, ln) if prev.bbox[0] <= ln.bbox[0] else (ln, prev)
                gap = right.bbox[0] - left.bbox[2]
                spans = list(left.spans)
                if gap > 0.25 * h and spans and not spans[-1].text.endswith(" "):
                    spans[-1] = Span(**{**spans[-1].__dict__, "text": spans[-1].text + " "})
                merged = Line(spans=spans + list(right.spans), bbox=[
                    min(prev.bbox[0], ln.bbox[0]), min(prev.bbox[1], ln.bbox[1]),
                    max(prev.bbox[2], ln.bbox[2]), max(prev.bbox[3], ln.bbox[3]),
                ])
                rows[-1] = merged
                continue
        rows.append(Line(spans=list(ln.spans), bbox=list(ln.bbox)))
    return rows


def _inner(text: str, s: Span) -> str:
    lead = re.match(r"^\s*", text).group(0)
    trail = re.search(r"\s*$", text).group(0)
    core = text.strip()
    if not core:
        return text
    if s.sup:
        core = f"<sup>{core}</sup>"
    if s.mark:
        core = f"<mark>{core}</mark>"
    if s.strike:
        core = f"~~{core}~~"
    if s.underline:
        core = f"<u>{core}</u>"
    if s.italic and not s.bold:
        core = f"*{core}*"
    return f"{lead}{core}{trail}"


def render_line(line: Line) -> str:
    """Line text with styling markers. Bold is applied per run of consecutive bold
    spans (so ``**Ed. (<mark>6-11</mark>)**``, never ``**Ed. (****<mark>…``); the
    other marks are applied per span inside the run."""
    out: list[str] = []
    run: list[str] = []
    run_bold: bool | None = None

    def flush() -> None:
        if not run:
            return
        text = "".join(run)
        if run_bold and text.strip():
            lead = re.match(r"^\s*", text).group(0)
            trail = re.search(r"\s*$", text).group(0)
            text = f"{lead}**{text.strip()}**{trail}"
        out.append(text)

    for s in line.spans:
        if run_bold is None or s.bold == run_bold:
            run.append(_inner(s.text, s))
            run_bold = s.bold
        else:
            flush()
            run, run_bold = [_inner(s.text, s)], s.bold
    flush()
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
        return "\n\n".join(f"{'#' * level} {_heading_text(r)}" for r in rendered)
    return " ".join(rendered) if label != "list" else "\n".join(rendered)


_GARBAGE_RE = re.compile(r"[\u00a4\ufffd\ue000-\uf8ff\x00-\x08\x0b\x0c\x0e-\x1f]")
IDENTICAL = 97  # char-level ratio above which layer text replaces VLM text outright


def _styled_runs(lines: list[Line]) -> list[tuple[str, Span]]:
    """Maximal runs of consecutive spans sharing a non-default style, as (text, style)."""
    runs: list[tuple[str, Span]] = []
    for ln in lines:
        cur: list[str] = []
        key: tuple | None = None
        proto: Span | None = None
        for sp in ln.spans:
            k = (sp.bold, sp.italic, sp.underline, sp.strike, sp.mark, sp.sup)
            if key is not None and k == key:
                cur.append(sp.text)
                continue
            if cur and any(key):
                runs.append(("".join(cur), proto))
            cur, key, proto = [sp.text], k, sp
        if cur and key is not None and any(key):
            runs.append(("".join(cur), proto))
    return runs


def transfer_styles(vlm_text: str, lines: list[Line]) -> str:
    """Wrap the VLM's characters with the text layer's styling: each styled run is
    located in ``vlm_text`` by fuzzy alignment and wrapped in place. Characters stay
    the VLM's (symbol fonts and clipped layers make the layer's unreliable)."""
    runs = _styled_runs(lines)
    if not runs:
        return vlm_text
    edits: list[tuple[int, int, Span]] = []
    taken: list[tuple[int, int]] = []
    for text, style in runs:
        q = re.sub(r"\s+", " ", text).strip()
        if len(q) < 2:
            continue
        al = fuzz.partial_ratio_alignment(q, vlm_text)
        if al is None or al.score < (85 if len(q) >= 10 else 75):
            continue
        a, b = al.dest_start, al.dest_end
        # snap to word boundaries
        while a > 0 and vlm_text[a - 1].isalnum() and vlm_text[a].isalnum():
            a -= 1
        while b < len(vlm_text) and vlm_text[b - 1].isalnum() and vlm_text[b].isalnum():
            b += 1
        if any(not (b <= x0 or a >= x1) for x0, x1 in taken):
            continue
        taken.append((a, b))
        edits.append((a, b, style))
    for a, b, style in sorted(edits, reverse=True):
        seg = vlm_text[a:b]
        if seg.strip():
            vlm_text = vlm_text[:a] + _wrap_vlm(seg, style) + vlm_text[b:]
    return vlm_text


def _wrap_vlm(seg: str, style: Span) -> str:
    lead = re.match(r"^\s*", seg).group(0)
    trail = re.search(r"\s*$", seg).group(0)
    core = seg.strip()
    already_bold = core.startswith("**") and core.endswith("**")
    inner = _inner(core, Span(text=core, bbox=style.bbox, size=style.size, italic=style.italic and not style.bold,
                               underline=style.underline, strike=style.strike, mark=style.mark, sup=style.sup))
    if style.bold and not already_bold:
        inner = f"**{inner}**"
    return f"{lead}{inner}{trail}"


def _align_slice(text: str, query: str, lo: int) -> tuple[int, int] | None:
    """Locate ``query`` (a layer group's text) inside ``text`` at or after ``lo``."""
    q = re.sub(r"\s+", " ", query).strip()
    if len(q) < 3 or lo >= len(text):
        return None
    al = fuzz.partial_ratio_alignment(q, text[lo:])
    if al is None or al.score < 80:
        return None
    return lo + al.dest_start, lo + al.dest_end


def render_blocks(
    block: dict[str, Any], tl: TextLayerPage, claimed: set[int] | None = None
) -> list[dict[str, Any]] | None:
    """VLM characters + text-layer styling, split into the layer's paragraphs / list
    items / headings with tight bboxes. Returns None to keep the VLM block as is."""
    if block["label"] not in TEXT_LABELS or block.get("bbox") is None:
        return None
    lines = _lines_in_bbox(tl, block["bbox"], claimed)
    if not lines:
        return None
    vlm = block["text"]
    if block["label"] == "title":
        # Headings are short; the layer's characters are reliable there and the
        # per-line split (stacked headings) matters for the title rules.
        layer_text = " ".join(ln.text for ln in lines)
        if _GARBAGE_RE.search(layer_text) or fuzz.token_set_ratio(_norm(layer_text), _norm(vlm)) < MIN_SIMILARITY:
            return None
        text = _render_group("title", lines, tl)
        nb = dict(block)
        nb.update(text=text, bbox=_lines_bbox1000(lines, tl), source="textlayer")
        return [nb]

    groups = _paragraphs(lines) if block["label"] in ("text", "list", "footnote") else [lines]
    parts: list[tuple[str, list[Line]]] = []
    for g in groups:
        parts.extend(_split_items_and_headings(block["label"], g, tl))
    if len(parts) <= 1:
        nb = dict(block)
        nb.update(text=transfer_styles(vlm, lines), source="styled-vlm")
        return [nb]

    # Cut the VLM text where the layer's groups start, keeping VLM characters.
    cuts: list[tuple[int, int, str, list[Line]]] = []
    lo = 0
    for label, sub in parts:
        sl = _align_slice(vlm, " ".join(ln.text for ln in sub), lo)
        if sl is None:
            cuts = []
            break
        cuts.append((sl[0], sl[1], label, sub))
        lo = sl[1]
    if not cuts:
        nb = dict(block)
        nb.update(text=transfer_styles(vlm, lines), source="styled-vlm")
        return [nb]
    out: list[dict[str, Any]] = []
    for i, (a, b, label, sub) in enumerate(cuts):
        start = a if i > 0 else 0
        stop = cuts[i + 1][0] if i + 1 < len(cuts) else len(vlm)
        seg = vlm[start:stop].strip()
        if not seg:
            continue
        if label in ("section_header", "section-header"):
            seg = f"## {_heading_text(seg)}"
        else:
            seg = transfer_styles(seg, sub)
        nb = dict(block)
        nb.update(label=label, text=seg, bbox=_lines_bbox1000(sub, tl), source="styled-vlm-split")
        out.append(nb)
    return out or None


_LIST_MARKER_RE = re.compile(r"^\s*(?:[\u2022\u25aa\u25cf\u2013\u2014\-\*\u25a0\u2023\u2043\u25e6]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|[ivxIVX]{1,5}[.)])\s+")


_CAPTION_RE = re.compile(r"^\s*(figure|fig\.|table|chart|exhibit|source|note|photo|image)\b", re.I)


def _is_heading_line(ln: Line, tl: TextLayerPage) -> bool:
    """A real section heading: larger than body text, or bold *and* short, not a
    caption, list item or sentence. ParseBench labels bold lead-ins inside
    paragraphs as Text, so bold alone is not enough."""
    spans = [s for s in ln.spans if s.text.strip()]
    text = ln.text.strip()
    if not spans or len(text) > 80 or _LIST_MARKER_RE.match(text) or _CAPTION_RE.match(text):
        return False
    bigger = all(s.size >= tl.body_size + 0.9 for s in spans)
    if bigger:
        return True
    all_bold = all(s.bold for s in spans)
    return all_bold and len(text.split()) <= 8 and not text.endswith((".", ":", ";", ","))


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


def _heading_text(text: str) -> str:
    """Heading markers already imply emphasis; inline tags only break the title regex."""
    text = re.sub(r"</?u>|\*\*|~~", "", text)
    return re.sub(r"\s+", " ", text).strip()


def reconcile_page(blocks: list[dict[str, Any]], tl: TextLayerPage | None) -> list[dict[str, Any]]:
    """Return a copy of ``blocks`` with ``text`` replaced from the text layer where it aligns."""
    if tl is None or not tl.has_text or tl.scanned:
        # Scanner OCR layers are worse than the VLM; keep the VLM output whole.
        return blocks
    out: list[dict[str, Any]] = []
    covered: set[int] = set()
    for b in blocks:
        before = set(covered)
        rendered = render_blocks(b, tl, covered)
        if rendered is None and b.get("bbox") is not None:
            # VLM text kept: its box still accounts for the lines inside it.
            covered.update(id(ln) for ln in tl.lines if _overlap_frac(ln, _box_pts(b["bbox"], tl)) >= 0.5)
        if rendered is None:
            covered |= before
        out.extend(rendered if rendered is not None else [dict(b)])
    return _insert_orphans(out, tl, covered)


def _box_pts(bbox1000: list[float], tl: TextLayerPage) -> tuple[float, float, float, float]:
    return (
        bbox1000[0] / 1000 * tl.width, bbox1000[1] / 1000 * tl.height,
        bbox1000[2] / 1000 * tl.width, bbox1000[3] / 1000 * tl.height,
    )


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


_TRACKED_CHANGE_RE = re.compile(r"^\s*(Deleted|Inserted|Formatted|Moved|Comment \[|Commented \[)[:\s]", re.I)


def _insert_orphans(blocks: list[dict[str, Any]], tl: TextLayerPage, covered: set[int]) -> list[dict[str, Any]]:
    """Text-layer lines outside every VLM bbox are real page text the VLM skipped
    (bullets, marginal marks, stray headings). Add them as text blocks, placed by
    vertical position among the located blocks."""
    orphans = [
        ln for ln in merge_rows([ln for ln in tl.lines if id(ln) not in covered and ln.text.strip()])
        if (len(ln.text.strip()) >= 2 or ln.text.strip().isdigit()) and not _TRACKED_CHANGE_RE.match(ln.text)
    ]
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
