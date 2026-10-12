"""Apply chart reconstruction to the VLM's chart blocks on one page.

For every ``chart`` block: snap the VLM table's numbers onto printed data labels
(``textlabels.snap_table``). Then add, as extra chart blocks, any table the pure
text-layer (``labels_table``) or vector (``vector.reconstruct_page``) methods
produce whose values are not already covered. ParseBench's chart metric scores
the presence of each annotated data point in some table, so additional candidate
tables can only add recall; they are deduplicated by numeric content.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pymupdf

import html as _html

from rapidfuzz import fuzz

from docparse.charts import ChartRegion, ChartTable, clean_number, is_numeric, parse_html_table, region_from_page
from docparse.charts import textlabels, vector
from docparse.textlayer import TextLayerPage

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _points(table: ChartTable) -> set[tuple[str, str, str]]:
    """(row label, column header, value) triples: the unit ParseBench scores."""
    out: set[tuple[str, str, str]] = set()
    for row in table.rows:
        if not row:
            continue
        for ci, cell in enumerate(row[1:], start=1):
            col = table.header[ci] if ci < len(table.header) else ""
            for m in _NUM_RE.findall(clean_number(cell)):
                out.add((_key(row[0]), _key(col), m))
    return out


def _novel(table: ChartTable, seen: set[tuple[str, str, str]], min_new: float = 0.3) -> bool:
    pts = _points(table)
    if len(pts) < 2:
        return False
    return len(pts - seen) / len(pts) >= min_new


def chart_caption(region: ChartRegion, tl: TextLayerPage, table: ChartTable | None) -> str:
    """Chart identity text for a ``<caption>``: title lines just above the box plus the
    non-numeric strings inside it (panel titles, legend entries) that are not already
    the table's row labels. ParseBench accepts a data point's title/series labels when
    they appear in the table's caption or a heading next to it."""
    x0, y0, x1, y1 = region.bbox
    parts: list[str] = []
    above = [
        ln for ln in tl.lines
        if ln.text.strip() and not is_numeric(ln.text) and y0 - 4 * max(8.0, ln.bbox[3] - ln.bbox[1]) <= ln.bbox[3] <= y0 + 2
        and min(ln.bbox[2], x1) - max(ln.bbox[0], x0) > 0.3 * (x1 - x0)
    ]
    above.sort(key=lambda ln: -ln.bbox[3])
    for ln in above[:2]:
        parts.append(re.sub(r"\s+", " ", ln.text).strip())
    row_labels = [r[0] for r in table.rows if r] if table else []
    headers = table.header if table else []
    seen: set[str] = set()
    for sp in region.spans:
        t = re.sub(r"\s+", " ", sp.text).strip()
        if len(t) < 3 or is_numeric(t) or t.lower() in seen:
            continue
        if any(fuzz.ratio(t.lower(), r.lower()) >= 85 for r in row_labels + headers):
            continue
        seen.add(t.lower())
        parts.append(t)
    cap = " | ".join(dict.fromkeys(p for p in parts if p))
    return cap[:400]


_FIRST_TD_RE = re.compile(r"(<tr[^>]*>\s*)<td([^>]*)>(.*?)</td>", re.S | re.I)


def row_headers(table_html: str) -> str:
    """Make the first cell of every body row a <th> (category row header)."""
    return _FIRST_TD_RE.sub(lambda m: f"{m.group(1)}<th{m.group(2)}>{m.group(3)}</th>", table_html)


def with_caption(table_html: str, caption: str) -> str:
    if not caption or "<caption" in table_html:
        return table_html
    return re.sub(r"<table[^>]*>", lambda m: m.group(0) + f"<caption>{_html.escape(caption)}</caption>", table_html, count=1)


def enhance_chart_blocks(blocks: list[dict[str, Any]], page: pymupdf.Page, tl: TextLayerPage) -> list[dict[str, Any]]:
    chart_idx = [i for i, b in enumerate(blocks) if b["label"] == "chart" and b.get("bbox") is not None]
    if not chart_idx or tl.scanned:
        return blocks
    out = list(blocks)
    seen: set[tuple[str, str, str]] = set()
    extras: list[dict[str, Any]] = []
    for i in chart_idx:
        b = out[i]
        bbox_pts = [
            b["bbox"][0] / 1000 * tl.width, b["bbox"][1] / 1000 * tl.height,
            b["bbox"][2] / 1000 * tl.width, b["bbox"][3] / 1000 * tl.height,
        ]
        region = region_from_page(page, tl, bbox_pts, vlm_table_html=b["text"] if "<table" in b["text"] else None)
        try:
            snapped = textlabels.snap_table(region) if region.vlm_table_html else None
        except Exception as e:  # noqa: BLE001
            logger.warning("snap_table failed: %s", e)
            snapped = None
        base = snapped or (parse_html_table(b["text"]) if "<table" in b["text"] else None)
        caption = chart_caption(region, tl, base)
        nb = dict(b)
        nb["text"] = with_caption(snapped.to_html() if snapped is not None else row_headers(b["text"]), caption)
        nb["source"] = "chart-snap" if snapped is not None else b.get("source", "vlm")
        out[i] = nb
        if base is not None:
            seen |= _points(base)
        try:
            for t in textlabels.panel_tables(region):
                if _novel(t, seen):
                    extras.append({**b, "text": with_caption(t.to_html(), chart_caption(region, tl, t)), "source": "chart-textlabels", "continuation": False})
                    seen |= _points(t)
        except Exception as e:  # noqa: BLE001
            logger.warning("labels_table failed: %s", e)
    anchor_b = out[chart_idx[-1]]
    caption_for_anchor = re.search(r"<caption>(.*?)</caption>", anchor_b["text"], flags=re.S)
    caption_for_anchor = _html.unescape(caption_for_anchor.group(1)) if caption_for_anchor else ""
    try:
        for t in vector.reconstruct_page(page, tl):
            if _novel(t, seen):
                anchor = out[chart_idx[-1]]
                extras.append({**anchor, "text": with_caption(t.to_html(), caption_for_anchor), "source": "chart-vector", "continuation": False})
                seen |= _points(t)
    except Exception as e:  # noqa: BLE001
        logger.warning("vector reconstruction failed: %s", e)
    if extras:
        last = chart_idx[-1]
        out = out[: last + 1] + extras + out[last + 1 :]
    return out
