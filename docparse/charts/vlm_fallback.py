"""Chart tables from the chart-crop VLM pass (serve/modal_chart_crops.py).

The dump stores, per PDF, one record per page with the model's raw text per chart
block. ``fallback_tables`` parses that text (HTML table, Markdown pipe table or TSV)
into ``ChartTable`` objects for the chart block whose bbox matches.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from docparse.charts import ChartTable, parse_html_table

logger = logging.getLogger(__name__)

MIN_IOU = 0.5


def load_chart_vlm(source_pdf: str, vlm_dir: str | None = None) -> dict[str, Any] | None:
    """Per-page chart-crop outputs for ``source_pdf`` (same <group>/<stem>.json layout as
    ``docparse.fusion.load_detections``), or None when the dump has no record."""
    root = Path(vlm_dir or os.getenv("DOCPARSE_CHART_VLM_DIR", "output/chart_vlm"))
    p = Path(source_pdf)
    cand = root / p.parent.name / (p.stem + ".json")
    if not cand.exists():
        return None
    try:
        return json.loads(cand.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("bad chart_vlm file %s: %s", cand, e)
        return None


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _clean(cell: str) -> str:
    return re.sub(r"\s+", " ", cell).strip()


def parse_pipe_table(text: str) -> ChartTable | None:
    """Markdown pipe table: header row, optional ``|---|`` separator, body rows."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("|") and "|" not in s:
            continue
        cells = [_clean(c) for c in s.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{2,}:?", c or "--") for c in cells):
            continue  # separator
        if any(cells):
            rows.append(cells)
    if len(rows) < 2:
        return None
    return ChartTable(header=rows[0], rows=rows[1:], source="chart-vlm")


def parse_tsv_table(text: str) -> ChartTable | None:
    """Tab- (or multi-space-) separated rows, first row = header."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cells = [_clean(c) for c in (line.split("\t") if "\t" in line else re.split(r"\s{2,}", line.strip()))]
        if len(cells) >= 2:
            rows.append(cells)
    if len(rows) < 2:
        return None
    return ChartTable(header=rows[0], rows=rows[1:], source="chart-vlm")


_NUMERIC = re.compile(r"^[+\-−–]?\s*[$€£¥]?\s*\d[\d,.]*\s*(%|[A-Za-z%./]{1,8}|p\.p\.|%pts)?\s*$", re.I)
_BULLET_KV = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*\**\s*(.+?)\s*\**\s*[:：]\s*\**\s*(.+?)\**\s*$")
_HEADING = re.compile(r"^\s*(?:#{1,6}\s+(.+?)\s*#*|\*\*(.+?)\**\s*:?\*\*|([^\-*•#].+?):)\s*$")
_SKIP_HEADINGS = {"chart recognition", "chart description", "data", "data points", "legend", "axes", "values", "series", "categories"}


def parse_bullet_table(text: str) -> ChartTable | None:
    """Markdown description with ``- label: value`` bullets, optionally grouped under
    headings (``### Series`` or ``- **Group:**`` with indented children). Groups become
    columns, bullet labels become rows; a single group gives a two-column table."""
    triples: list[tuple[str, str, str]] = []  # (group, label, value)
    group = ""
    group_indent = -1
    for raw in text.splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        m = _BULLET_KV.match(raw)
        if m and _NUMERIC.match(m.group(2)):
            if indent <= group_indent:
                group, group_indent = "", -1
            triples.append((group, _clean(m.group(1)), _clean(m.group(2))))
            continue
        h = _HEADING.match(raw)
        if h:
            name = _clean(h.group(1) or h.group(2) or h.group(3) or "")
            if name and name.lower() not in _SKIP_HEADINGS and not _NUMERIC.match(name):
                group, group_indent = name, -1
            continue
        b = re.match(r"^\s*(?:[-*•]|\d+[.)])\s*\**\s*(.+?)\s*:?\s*\**\s*$", raw)
        if b and _clean(b.group(1)).lower() not in _SKIP_HEADINGS:
            group, group_indent = _clean(b.group(1)).rstrip(":"), indent
    if len(triples) < 2:
        return None
    groups: list[str] = []
    labels: list[str] = []
    for g, lab, _ in triples:
        if g not in groups:
            groups.append(g)
        if lab not in labels:
            labels.append(lab)
    cells: dict[tuple[str, str], str] = {(g, lab): v for g, lab, v in triples}
    if groups == [""]:
        return ChartTable(header=["", "Value"], rows=[[lab, cells[("", lab)]] for lab in labels], source="chart-vlm")
    if len(labels) == 1 and all(g for g in groups):
        return ChartTable(header=["", labels[0]], rows=[[g, cells.get((g, labels[0]), "")] for g in groups], source="chart-vlm")
    header = [""] + [g or "Value" for g in groups]
    rows = [[lab] + [cells.get((g, lab), "") for g in groups] for lab in labels]
    return ChartTable(header=header, rows=rows, source="chart-vlm")


def parse_model_output(text: str) -> list[ChartTable]:
    """All tables in a model answer: HTML first, then pipe tables, then TSV, then
    ``- label: value`` bullet descriptions."""
    if not text or not text.strip():
        return []
    tables: list[ChartTable] = []
    for match in re.finditer(r"<table.*?</table>", text, flags=re.S | re.I):
        t = parse_html_table(match.group())
        if t is not None and t.rows:
            t.source = "chart-vlm"
            _restore_labels(t, text[:match.start()])
            tables.append(t)
    if tables:
        return tables
    if "|" in text:
        t = parse_pipe_table(text)
        if t is not None:
            _restore_labels(t, text.split("|", 1)[0])
            return [t]
    t = parse_bullet_table(text)
    if t is not None:
        return [t]
    t = parse_tsv_table(text)
    return [t] if t is not None else []


def _restore_labels(table: ChartTable, prefix: str) -> None:
    """Keep the model's explicit title and leading categorical columns."""
    headings = re.findall(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$", prefix, flags=re.M)
    if headings and not table.caption and headings[-1].lower() not in _SKIP_HEADINGS:
        table.caption = headings[-1]
    for col in range(1, len(table.header) - 1):
        values = [r[col] for r in table.rows if len(r) > col and r[col].strip()]
        if not values or any(re.search(r"\d", v) or not any(c.isalpha() for c in v) for v in values):
            break
        table.row_header_columns = col + 1


def fallback_tables(page_record: dict[str, Any] | None, bbox1000: list[float]) -> list[ChartTable]:
    """Tables the crop model produced for the chart block at ``bbox1000`` (IoU ≥ 0.5)."""
    if not page_record:
        return []
    best = None
    best_iou = 0.0
    for c in page_record.get("charts", []):
        iou = _iou([float(v) for v in c["bbox"]], [float(v) for v in bbox1000])
        if iou > best_iou:
            best, best_iou = c, iou
    if best is None or best_iou < MIN_IOU:
        return []
    return parse_model_output(best.get("text", ""))
