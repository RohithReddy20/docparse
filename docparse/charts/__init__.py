"""Chart reconstruction from born-digital PDFs.

A chart region is reconstructed into a ``ChartTable`` (header row + rows, first
cell of each row is the category label) from two deterministic sources:

- ``textlabels``: printed data labels / axis ticks / legend text in the text layer,
  snapped onto the VLM's table skeleton when one exists;
- ``vector``: bar rectangles and line paths from ``page.get_drawings()``, scaled by
  the numeric axis ticks, series resolved through legend swatch colours.

Both operate on ``ChartRegion`` and return ``ChartTable``; ``scripts/chart_eval.py``
scores any reconstruction against ParseBench's chart rules locally.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import pymupdf

from docparse.textlayer import Span, TextLayerPage

# "1,200", "−17", "45%", "$1,200.5", "$600B", "(200)" (accounting negative), "5bn", "12.5x"
_UNIT = r"(?:%|bn|mn|tn|b|m|k|t|x|pp|pts|p\.p\.)?"
NUMERIC_RE = re.compile(r"^(?:\(\s*[$€£¥]?\s*\d[\d\s,.]*\s*" + _UNIT + r"\s*\)|[-−+]?\s*[$€£¥]?\s*\d[\d\s,.]*\s*" + _UNIT + r")$", re.I)


@dataclass
class Drawing:
    kind: str  # "rect" (filled) | "path" (stroked polyline)
    bbox: list[float]  # x0, y0, x1, y1 (page points)
    fill: tuple[float, ...] | None
    color: tuple[float, ...] | None
    points: list[tuple[float, float]] = field(default_factory=list)  # for paths


@dataclass
class ChartRegion:
    bbox: list[float]  # x0, y0, x1, y1 (page points)
    spans: list[Span]  # text-layer spans whose centre lies in bbox
    drawings: list[Drawing]  # vector drawings intersecting bbox
    page_width: float
    page_height: float
    vlm_table_html: str | None = None  # the VLM's chart table, when available


@dataclass
class ChartTable:
    header: list[str]  # first cell is the category column name (may be "")
    rows: list[list[str]]  # each row: [category, v1, v2, ...]
    confidence: float = 0.0
    source: str = ""
    caption: str = ""  # chart / panel title; emitted as <caption> (a label source for the scorer)
    row_header_columns: int = 1

    def __post_init__(self) -> None:
        # Ligatures ("Inﬂation") never match annotated labels: fold them.
        self.header = [unicodedata.normalize("NFKC", h) for h in self.header]
        self.rows = [[unicodedata.normalize("NFKC", c) for c in r] for r in self.rows]
        self.caption = unicodedata.normalize("NFKC", self.caption)

    def to_html(self) -> str:
        """Series as a <th> header row, categories as <th> row headers: ParseBench
        associates a value's labels through the <th> chain of its row and column."""
        cap = f"<caption>{html.escape(self.caption)}</caption>" if self.caption else ""
        head = "".join(f"<th>{html.escape(h)}</th>" for h in self.header)
        body = "".join(
            "<tr>" + "".join(
                (f"<th>{html.escape(c)}</th>" if i < self.row_header_columns else f"<td>{html.escape(c)}</td>") for i, c in enumerate(r)
            ) + "</tr>"
            for r in self.rows
        )
        return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def is_numeric(text: str) -> bool:
    return bool(NUMERIC_RE.match(text.strip()))


def clean_number(text: str) -> str:
    """'14 000' -> '14000', '−17' -> '-17', '45%' -> '45', '$1,200.5' -> '1200.5'."""
    s = text.strip().replace("−", "-").replace("%", "")
    if re.fullmatch(r"\(.*\)", s):
        s = "-" + s[1:-1]  # accounting negative
    s = re.sub(r"[$€£¥\s]", "", s)
    s = re.sub(r"(?<=\d)\s*(?:bn|mn|tn|b|m|k|t|x|pp|pts|p\.p\.)$", "", s, flags=re.I)
    if s.count(",") and s.count(".") == 0 and re.search(r",\d{1,2}$", s):
        s = s.replace(",", ".")  # decimal comma
    return s.replace(",", "")


def region_from_page(
    page: pymupdf.Page,
    tl: TextLayerPage,
    bbox: list[float],
    vlm_table_html: str | None = None,
    pad: float = 2.0,
) -> ChartRegion:
    x0, y0, x1, y1 = bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad
    spans = [
        s
        for ln in tl.lines
        for s in ln.spans
        if s.text.strip() and x0 <= (s.bbox[0] + s.bbox[2]) / 2 <= x1 and y0 <= (s.bbox[1] + s.bbox[3]) / 2 <= y1
    ]
    clip = pymupdf.Rect(x0, y0, x1, y1)
    drawings: list[Drawing] = []
    for dr in page.get_drawings():
        r = dr.get("rect")
        if r is None or not r.intersects(clip):
            continue
        box = [r.x0, r.y0, r.x1, r.y1]
        if dr.get("fill") and r.width >= 1 and r.height >= 1:
            drawings.append(Drawing("rect", box, tuple(dr["fill"]), tuple(dr["color"]) if dr.get("color") else None))
        elif dr.get("color"):
            pts: list[tuple[float, float]] = []
            for it in dr.get("items", []):
                if it[0] == "l":
                    pts.extend([(it[1].x, it[1].y), (it[2].x, it[2].y)])
                elif it[0] == "c":
                    pts.extend([(it[1].x, it[1].y), (it[4].x, it[4].y)])
            drawings.append(Drawing("path", box, None, tuple(dr["color"]), pts))
    return ChartRegion(
        bbox=[bbox[0], bbox[1], bbox[2], bbox[3]],
        spans=spans,
        drawings=drawings,
        page_width=page.rect.width,
        page_height=page.rect.height,
        vlm_table_html=vlm_table_html,
    )


def parse_html_table(table_html: str) -> ChartTable | None:
    """Minimal parser for the VLM's <table> (thead/tbody or plain rows)."""
    rows: list[list[str]] = []
    pending: dict[int, tuple[str, int]] = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S | re.I):
        cells = re.findall(r"<t[hd]\b([^>]*)>(.*?)</t[hd]>", tr, flags=re.S | re.I)
        row = {col: text for col, (text, _) in pending.items()}
        pending = {col: (text, remaining - 1) for col, (text, remaining) in pending.items() if remaining > 1}
        col = 0
        for attrs, content in cells:
            text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", content))).strip()
            spans = {name.lower(): min(256, max(1, int(n))) for name, n in re.findall(r"\b(rowspan|colspan)\s*=\s*['\"]?(\d+)", attrs, re.I)}
            while col in row:
                col += 1
            for _ in range(spans.get("colspan", 1)):
                row[col] = text
                if spans.get("rowspan", 1) > 1:
                    pending[col] = (text, spans["rowspan"] - 1)
                col += 1
        if row:
            rows.append([row.get(i, "") for i in range(max(row) + 1)])
    if not rows:
        return None
    if len(rows) > 1 and all(len(row) == len(rows[0]) + 1 for row in rows[1:]):
        rows[0].insert(0, "")
    caption = re.search(r"<caption\b[^>]*>(.*?)</caption>", table_html, flags=re.S | re.I)
    caption_text = html.unescape(re.sub(r"<[^>]+>", " ", caption.group(1))).strip() if caption else ""
    return ChartTable(header=rows[0], rows=rows[1:], source="vlm", caption=caption_text)


def to_dict(table: ChartTable) -> dict[str, Any]:
    return {"header": table.header, "rows": table.rows, "confidence": table.confidence, "source": table.source}
