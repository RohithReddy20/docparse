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

from docparse.charts import ChartTable, clean_number, parse_html_table, region_from_page
from docparse.charts import textlabels, vector
from docparse.textlayer import TextLayerPage

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers(table: ChartTable) -> set[str]:
    out: set[str] = set()
    for row in table.rows:
        for cell in row[1:]:
            for m in _NUM_RE.findall(clean_number(cell)):
                out.add(m)
    return out


def _novel(table: ChartTable, seen: set[str], min_new: float = 0.3) -> bool:
    nums = _numbers(table)
    if len(nums) < 2:
        return False
    new = nums - seen
    return len(new) / len(nums) >= min_new


def enhance_chart_blocks(blocks: list[dict[str, Any]], page: pymupdf.Page, tl: TextLayerPage) -> list[dict[str, Any]]:
    chart_idx = [i for i, b in enumerate(blocks) if b["label"] == "chart" and b.get("bbox") is not None]
    if not chart_idx or tl.scanned:
        return blocks
    out = list(blocks)
    seen: set[str] = set()
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
        if snapped is not None:
            nb = dict(b)
            nb["text"] = snapped.to_html()
            nb["source"] = "chart-snap"
            out[i] = nb
        base = snapped or (parse_html_table(b["text"]) if "<table" in b["text"] else None)
        if base is not None:
            seen |= _numbers(base)
        try:
            for t in textlabels.panel_tables(region):
                if _novel(t, seen):
                    extras.append({**b, "text": t.to_html(), "source": "chart-textlabels", "continuation": False})
                    seen |= _numbers(t)
        except Exception as e:  # noqa: BLE001
            logger.warning("labels_table failed: %s", e)
    try:
        for t in vector.reconstruct_page(page, tl):
            if _novel(t, seen):
                anchor = out[chart_idx[-1]]
                extras.append({**anchor, "text": t.to_html(), "source": "chart-vector", "continuation": False})
                seen |= _numbers(t)
    except Exception as e:  # noqa: BLE001
        logger.warning("vector reconstruction failed: %s", e)
    if extras:
        last = chart_idx[-1]
        out = out[: last + 1] + extras + out[last + 1 :]
    return out
