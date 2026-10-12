"""PDF text layer with styling, per page (PyMuPDF, CPU).

Born-digital PDFs carry exact characters plus the styling pixel-only parsers lose:
bold (font flag/name), underline and strikethrough (thin filled rects drawn at the
baseline / mid-line of a span), highlight (coloured rect behind a span or a markup
annotation), and font size (heading hierarchy). Everything is returned in page
points with bboxes as ``[x0, y0, x1, y1]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

MIN_CHARS_FOR_TEXT_LAYER = 80
_BOLD_FLAG = 16
_WHITE = (1.0, 1.0, 1.0)


@dataclass
class Span:
    text: str
    bbox: list[float]  # x0, y0, x1, y1 (page points)
    size: float
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strike: bool = False
    mark: bool = False
    sup: bool = False


@dataclass
class Line:
    spans: list[Span]
    bbox: list[float]

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)


@dataclass
class TextLayerPage:
    width: float
    height: float
    lines: list[Line] = field(default_factory=list)
    body_size: float = 0.0  # most common font size on the page
    scanned: bool = False  # a raster image covers most of the page: the text layer is OCR output, not source text

    @property
    def has_text(self) -> bool:
        return sum(len(line.text.strip()) for line in self.lines) >= MIN_CHARS_FOR_TEXT_LAYER


def _x_overlap(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return inter / max(1e-6, a[2] - a[0])


def _is_coloured(rgb: tuple[float, ...] | None) -> bool:
    """True for saturated colours; black, white and greys are not highlights."""
    if not rgb or len(rgb) < 3:
        return False
    return max(rgb[:3]) - min(rgb[:3]) > 0.25


def _apply_drawings(lines: list[Line], page: pymupdf.Page) -> None:
    """Mark underline / strikethrough / highlight from vector drawings."""
    thin: list[tuple[list[float], bool]] = []  # (box, coloured)
    fills: list[list[float]] = []
    for dr in page.get_drawings():
        r = dr.get("rect")
        if r is None or r.width < 4:
            continue
        box = [r.x0, r.y0, r.x1, r.y1]
        if r.height <= 2.5:
            thin.append((box, _is_coloured(dr.get("fill") or dr.get("color"))))
        elif dr.get("fill") and tuple(round(c, 2) for c in dr["fill"]) != _WHITE and 4 <= r.height <= 24:
            fills.append(box)
    if not thin and not fills:
        return
    for line in lines:
        for s in line.spans:
            x0, y0, x1, y1 = s.bbox
            h = y1 - y0
            for t, coloured in thin:
                if _x_overlap(s.bbox, t) < 0.5:
                    continue
                ty = (t[1] + t[3]) / 2
                if y1 - 0.15 * h <= ty <= y1 + 0.3 * h:
                    # A coloured rule under text is an editor highlight, not an underline.
                    if coloured:
                        s.mark = True
                    else:
                        s.underline = True
                elif y0 + 0.3 * h <= ty <= y0 + 0.7 * h:
                    s.strike = True
            for f in fills:
                if _x_overlap(s.bbox, f) >= 0.5 and f[1] <= y0 + 0.5 * h <= f[3]:
                    s.mark = True


def _apply_annots(lines: list[Line], page: pymupdf.Page) -> None:
    boxes: dict[str, list[list[float]]] = {"mark": [], "underline": [], "strike": []}
    for a in page.annots():
        kind = {8: "mark", 9: "underline", 11: "strike"}.get(a.type[0])
        if kind:
            r = a.rect
            boxes[kind].append([r.x0, r.y0, r.x1, r.y1])
    if not any(boxes.values()):
        return
    for line in lines:
        for s in line.spans:
            cy = (s.bbox[1] + s.bbox[3]) / 2
            for kind, bl in boxes.items():
                for b in bl:
                    if _x_overlap(s.bbox, b) >= 0.5 and b[1] <= cy <= b[3]:
                        setattr(s, kind, True)


def _is_scanned(page: pymupdf.Page) -> bool:
    """True when one raster image covers >= 60% of the page (a scan with an OCR layer)."""
    area = page.rect.width * page.rect.height
    for img in page.get_images(full=True):
        try:
            rects = page.get_image_rects(img[0])
        except Exception:  # noqa: BLE001
            continue
        if any(r.width * r.height >= 0.6 * area for r in rects):
            return True
    return False


def extract_page(page: pymupdf.Page) -> TextLayerPage:
    d = page.get_text("dict")
    out = TextLayerPage(width=page.rect.width, height=page.rect.height)
    sizes: dict[float, int] = {}
    for block in d["blocks"]:
        for raw_line in block.get("lines", []):
            spans: list[Span] = []
            for rs in raw_line["spans"]:
                if not rs["text"]:
                    continue
                font = rs["font"].lower()
                s = Span(
                    text=rs["text"],
                    bbox=list(rs["bbox"]),
                    size=round(rs["size"], 1),
                    bold=bool(rs["flags"] & _BOLD_FLAG) or "bold" in font or "black" in font or "heavy" in font,
                    italic=bool(rs["flags"] & 2) or "italic" in font or "oblique" in font,
                    sup=bool(rs["flags"] & 1),
                )
                spans.append(s)
                if s.text.strip():
                    sizes[s.size] = sizes.get(s.size, 0) + len(s.text.strip())
            if spans:
                out.lines.append(Line(spans=spans, bbox=list(raw_line["bbox"])))
    if sizes:
        out.body_size = max(sizes.items(), key=lambda kv: kv[1])[0]
    out.scanned = _is_scanned(page)
    _apply_drawings(out.lines, page)
    _apply_annots(out.lines, page)
    return out


def extract(pdf_path: str) -> list[TextLayerPage]:
    doc = pymupdf.open(pdf_path)
    pages = [extract_page(p) for p in doc]
    doc.close()
    return pages
