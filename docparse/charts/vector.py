"""Bar / line chart reconstruction from a born-digital PDF's vector drawings.

Bars are axis-aligned filled rectangles (``re`` items, or closed four-segment
``l`` quads, inside filled drawings), lines are stroked polylines. Numeric tick
spans aligned along one axis give a linear value scale, non-numeric spans along
the other axis give categories, legend swatches (tiny filled rects or short
stroked segments beside a text span) name the series by colour. A numeric span
printed at a bar's end is preferred over the geometric estimate.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

import pymupdf

from docparse.charts import ChartRegion, ChartTable, clean_number, is_numeric, region_from_page
from docparse.textlayer import Span, TextLayerPage

Color = tuple[float, ...]

GREY_SAT = 0.08  # max channel spread for a colour to count as grey (gridlines, axes)
MIN_BAR = 0.8  # points; thinner filled rects are rules/gridlines
MAX_SWATCH = 26.0  # legend swatch max side
ALIGN_TOL = 6.0  # points; tick centres aligned on an axis
CORE_GAP = 24.0  # points; clustering gap between bars/lines of one chart
LEGEND_REACH = 120.0  # points; a legend may sit this far from its plot
SPAN_GAP = 40.0  # points; tick/label spans this close to the core belong to the chart
ZERO_TOL = 1.5  # points; bar edge sits on the zero line


@dataclass
class Bar:
    x0: float
    y0: float
    x1: float
    y1: float
    fill: Color

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Poly:
    points: list[tuple[float, float]]
    color: Color
    bbox: list[float] = field(default_factory=list)


@dataclass
class Axis:
    orient: str  # "x" (ticks along x, values vary with x) | "y"
    a: float  # value = a * coord + b
    b: float
    step: float  # value difference between consecutive ticks
    coords: list[float]
    values: list[float]
    pos: float  # perpendicular coordinate of the tick row/column
    span_ids: list[int] = field(default_factory=list)

    def value(self, coord: float) -> float:
        return self.a * coord + self.b

    def coord(self, value: float) -> float:
        return (value - self.b) / self.a if self.a else 0.0


# --------------------------------------------------------------------------- primitives


def _rgb(c: Color | None) -> Color | None:
    if c is None:
        return None
    return tuple(round(float(v), 2) for v in c[:3])


def _is_grey(c: Color | None) -> bool:
    if c is None:
        return True
    return max(c[:3]) - min(c[:3]) <= GREY_SAT


def _quads_from_lines(items: list) -> list[tuple[float, float, float, float]]:
    """Closed axis-aligned rectangles drawn as four consecutive ``l`` segments."""
    segs = [(it[1].x, it[1].y, it[2].x, it[2].y) for it in items if it[0] == "l"]
    out: list[tuple[float, float, float, float]] = []
    i = 0
    while i + 3 < len(segs):
        quad = segs[i : i + 4]
        xs = {round(v, 1) for s in quad for v in (s[0], s[2])}
        ys = {round(v, 1) for s in quad for v in (s[1], s[3])}
        closed = (round(quad[0][0], 1), round(quad[0][1], 1)) == (round(quad[3][2], 1), round(quad[3][3], 1))
        if len(xs) == 2 and len(ys) == 2 and closed:
            out.append((min(xs), min(ys), max(xs), max(ys)))
            i += 4
        else:
            i += 1
    return out


def page_primitives(page: pymupdf.Page) -> tuple[list[Bar], list[Poly]]:
    bars, polys, _frames = page_primitives_full(page)
    return bars, polys


def page_primitives_full(page: pymupdf.Page) -> tuple[list[Bar], list[Poly], list[list[float]]]:
    """Bars, coloured polylines, and grey frame/grid path bboxes (plot-area anchors)."""
    bars: list[Bar] = []
    polys: list[Poly] = []
    frames: list[list[float]] = []
    page_area = page.rect.width * page.rect.height
    for dr in page.get_drawings():
        fill = _rgb(dr.get("fill"))
        color = _rgb(dr.get("color"))
        items = dr.get("items", [])
        if fill is not None:
            rects = [tuple(it[1]) for it in items if it[0] == "re"]
            rects += _quads_from_lines(items)
            for x0, y0, x1, y1 in rects:
                x0, x1 = min(x0, x1), max(x0, x1)
                y0, y1 = min(y0, y1), max(y0, y1)
                w, h = x1 - x0, y1 - y0
                if w < MIN_BAR or h < MIN_BAR or w * h > 0.35 * page_area:
                    continue
                if _is_grey(fill) and (w > 200 and h > 200):
                    continue  # background panel
                bars.append(Bar(x0, y0, x1, y1, fill))
        elif color is not None and _is_grey(color):
            r = dr["rect"]
            if (r.width > 20 or r.height > 20) and r.width * r.height < 0.35 * page_area:
                frames.append([r.x0, r.y0, r.x1, r.y1])
        elif color is not None:
            pts: list[tuple[float, float]] = []
            for it in items:
                if it[0] == "l":
                    if not pts or pts[-1] != (it[1].x, it[1].y):
                        pts.append((it[1].x, it[1].y))
                    pts.append((it[2].x, it[2].y))
                elif it[0] == "c":
                    if not pts:
                        pts.append((it[1].x, it[1].y))
                    pts.append((it[4].x, it[4].y))
            if len(pts) >= 3:
                r = dr["rect"]
                polys.append(Poly(pts, color, [r.x0, r.y0, r.x1, r.y1]))
    return bars, polys, frames


# --------------------------------------------------------------------------- regions


def _union(a: list[float], b: list[float]) -> list[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _near(a: list[float], b: list[float], gap: float) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def _cluster(boxes: list[list[float]], gap: float) -> list[list[float]]:
    clusters: list[list[float]] = []
    for box in boxes:
        merged = list(box)
        rest: list[list[float]] = []
        for c in clusters:
            if _near(c, merged, gap):
                merged = _union(c, merged)
            else:
                rest.append(c)
        clusters = rest + [merged]
    # second pass: clusters may have become adjacent through growth
    changed = True
    while changed:
        changed = False
        out: list[list[float]] = []
        for c in clusters:
            for i, o in enumerate(out):
                if _near(o, c, gap):
                    out[i] = _union(o, c)
                    changed = True
                    break
            else:
                out.append(c)
        clusters = out
    return clusters


def _dist(box: list[float], s: Span) -> float:
    """Distance from a span's centre to a box (0 inside)."""
    cx, cy = _span_center(s)
    dx = max(box[0] - cx, 0.0, cx - box[2])
    dy = max(box[1] - cy, 0.0, cy - box[3])
    return math.hypot(dx, dy)


def _span_center(s: Span) -> tuple[float, float]:
    return (s.bbox[0] + s.bbox[2]) / 2, (s.bbox[1] + s.bbox[3]) / 2


def find_plot_areas(page: pymupdf.Page, tl: TextLayerPage) -> list[list[float]]:
    return [grown for _core, grown in _plot_cores(page, tl)]


def _page_swatches(bars: list[Bar], polys: list[Poly], tl: TextLayerPage) -> set[int]:
    all_spans = [sp for ln in tl.lines for sp in ln.spans]
    return _legend(bars, polys, all_spans)[1]


def _plot_cores(page: pymupdf.Page, tl: TextLayerPage) -> list[tuple[list[float], list[float]]]:
    """(core bbox of bars/lines, grown bbox including nearby tick/label spans)."""
    bars, polys, frames = page_primitives_full(page)
    swatches = _page_swatches(bars, polys, tl)
    big = [b for b in bars if id(b) not in swatches and (b.w > MAX_SWATCH or b.h > MAX_SWATCH or b.w * b.h >= 30)]
    boxes = [[b.x0, b.y0, b.x1, b.y1] for b in big]
    boxes += [p.bbox for p in polys if (p.bbox[2] - p.bbox[0]) > 20 or (p.bbox[3] - p.bbox[1]) > 20]
    if not boxes:
        return []
    # grey gridlines/axes anchor the plot area but never form a chart on their own
    anchored = [f for f in frames if any(_near(f, b, CORE_GAP) for b in boxes)]
    out: list[tuple[list[float], list[float]]] = []
    for core in _cluster(boxes + anchored, CORE_GAP):
        n_bars = sum(1 for b in big if _near(core, [b.x0, b.y0, b.x1, b.y1], 0))
        n_poly = sum(1 for p in polys if _near(core, p.bbox, 0))
        if n_bars < 2 and n_poly < 1:
            continue
        grown = list(core)
        cw = core[2] - core[0]
        for ln in tl.lines:
            for sp in ln.spans:
                if not sp.text.strip() or (sp.bbox[2] - sp.bbox[0]) > 1.2 * cw:
                    continue  # page-wide titles/captions are not chart furniture
                cx, cy = _span_center(sp)
                if core[0] - SPAN_GAP <= cx <= core[2] + SPAN_GAP and core[1] - SPAN_GAP <= cy <= core[3] + SPAN_GAP:
                    grown = _union(grown, list(sp.bbox))
        if (grown[2] - grown[0]) < 40 or (grown[3] - grown[1]) < 25:
            continue
        out.append((core, grown))
    return out


# --------------------------------------------------------------------------- axes


def _fit(coords: list[float], values: list[float]) -> tuple[float, float, float]:
    n = len(coords)
    mx, my = sum(coords) / n, sum(values) / n
    sxx = sum((c - mx) ** 2 for c in coords)
    if sxx == 0:
        return 0.0, my, 0.0
    a = sum((c - mx) * (v - my) for c, v in zip(coords, values)) / sxx
    b = my - a * mx
    ss_res = sum((v - (a * c + b)) ** 2 for c, v in zip(coords, values))
    ss_tot = sum((v - my) ** 2 for v in values) or 1e-9
    return a, b, 1 - ss_res / ss_tot


def _to_float(text: str) -> float | None:
    s = clean_number(text)
    try:
        return float(s)
    except ValueError:
        return None


def _is_categorical(values: list[float], step: float) -> bool:
    """Year-like integer ticks are category labels, not a value scale."""
    return all(v == int(v) and 1900 <= v <= 2100 for v in values)


def detect_axes(spans: list[Span]) -> list[Axis]:
    nums = [(s, _to_float(s.text)) for s in spans if is_numeric(s.text) and len(s.text.split()) <= 2]
    nums = [(s, v) for s, v in nums if v is not None]
    axes: list[Axis] = []
    seen: set[tuple[int, ...]] = set()
    keyings = {
        "x": [lambda s: _span_center(s)[1]],
        "y": [lambda s: _span_center(s)[0], lambda s: s.bbox[2], lambda s: s.bbox[0]],
    }
    for orient, keyfns in keyings.items():
        for keyfn in keyfns:
            groups: dict[int, list[tuple[Span, float]]] = defaultdict(list)
            for s, v in nums:
                groups[round(keyfn(s) / ALIGN_TOL)].append((s, v))
            for key, members in groups.items():
                if len(members) < 3:
                    continue
                members.sort(key=lambda m: _span_center(m[0])[0 if orient == "x" else 1])
                ids = tuple(sorted(id(m[0]) for m in members))
                if ids in seen:
                    continue
                coords = [_span_center(m[0])[0 if orient == "x" else 1] for m in members]
                values = [m[1] for m in members]
                diffs = [b - a for a, b in zip(values, values[1:])]
                if not (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)):
                    continue
                a, b, r2 = _fit(coords, values)
                if r2 < 0.995 or a == 0:
                    continue
                seen.add(ids)
                step = min(abs(d) for d in diffs) or 1.0
                axes.append(Axis(orient, a, b, step, coords, values, key * ALIGN_TOL, list(ids)))
    return axes


def _round_value(v: float, step: float) -> str:
    """Round a geometric estimate the way a human reads an axis: whole units when
    ticks are ≥2 apart, otherwise a tenth of the tick step."""
    if step <= 0:
        return f"{v:.2f}"
    if step >= 2:
        return str(int(round(v))) if round(v) != 0 else "0"
    decimals = max(0, min(4, int(math.ceil(-math.log10(step / 10)))))
    s = f"{v:.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


# --------------------------------------------------------------------------- legend / series


def _legend(bars: list[Bar], polys: list[Poly], spans: list[Span]) -> tuple[dict[Color, str], set[int]]:
    """Series name per colour from legend swatches, plus the ids of swatch bars."""
    names: dict[Color, str] = {}
    swatches: set[int] = set()
    # legend entries are words, or years ("2022" / "2024" series)
    texts = [
        s for s in spans
        if s.text.strip() and (not is_numeric(s.text) or re.fullmatch(r"(19|20)\d\d", s.text.strip()))
    ]

    def beside(x1: float, cy: float, h: float) -> str | None:
        best: tuple[float, str] | None = None
        for s in texts:
            sx0, sy0, sx1, sy1 = s.bbox
            scy = (sy0 + sy1) / 2
            if abs(scy - cy) <= max(h, 6) and 0 <= sx0 - x1 <= 16:
                d = sx0 - x1
                if best is None or d < best[0]:
                    best = (d, s.text.strip())
        return best[1] if best else None

    for b in bars:
        if b.w <= MAX_SWATCH and b.h <= MAX_SWATCH and 0.5 <= b.w / max(b.h, 0.1) <= 2.0:
            name = beside(b.x1, b.cy, b.h)
            if name:
                swatches.add(id(b))
                if b.fill not in names:
                    names[b.fill] = name
    for p in polys:
        w, h = p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]
        if w <= 30 and h <= 4 and len(p.points) <= 6:
            name = beside(p.bbox[2], (p.bbox[1] + p.bbox[3]) / 2, 6)
            if name and p.color not in names:
                names[p.color] = name
    return names, swatches


# --------------------------------------------------------------------------- reconstruction


def _split_tokens(s: Span) -> list[tuple[float, float, str]]:
    """A single span holding several evenly spaced labels ('2015 2016 ... 2024')
    becomes one pseudo-span per token, centres spread across the span width."""
    toks = s.text.split()
    if len(toks) < 3:
        return [((s.bbox[0] + s.bbox[2]) / 2, (s.bbox[1] + s.bbox[3]) / 2, s.text.strip())]
    x0, x1 = s.bbox[0], s.bbox[2]
    cy = (s.bbox[1] + s.bbox[3]) / 2
    total = sum(len(t) for t in toks) + len(toks) - 1
    out: list[tuple[float, float, str]] = []
    pos = 0
    for t in toks:
        cx = x0 + (x1 - x0) * (pos + len(t) / 2) / total
        out.append((cx, cy, t))
        pos += len(t) + 1
    return out


_FURNITURE_RE = re.compile(r"^(source|note|notes|figure|fig\.|table|chart)\b", re.I)


def _categories(
    spans: list[Span], orient: str, plot: list[float], exclude: set[int], n_groups: int = 0
) -> list[tuple[float, str]]:
    """Category labels. Horizontal bars (value axis "x"): non-numeric spans outside
    the plot's x-range, keyed by their y. Vertical bars: spans below the plot keyed
    by x; a single span holding every label ("2015 2016 …") is split when its token
    count matches the number of bar groups. Returns (coordinate, text)."""
    pool = [
        s for s in spans
        if s.text.strip() and id(s) not in exclude and not _FURNITURE_RE.match(s.text.strip())
    ]
    cats: list[tuple[float, str]] = []
    if orient == "x":
        side = [
            s for s in pool
            if (s.bbox[2] <= plot[0] + 4 or s.bbox[0] >= plot[2] - 4)
            and s.bbox[3] >= plot[1] - 6 and s.bbox[1] <= plot[3] + 6
        ]
        left = [s for s in side if s.bbox[2] <= plot[0] + 4]
        right = [s for s in side if s not in left]
        chosen = left if len(left) >= len(right) else right
        for s in chosen:
            cats.append(((s.bbox[1] + s.bbox[3]) / 2, s.text.strip()))
    else:
        below = [
            s for s in pool
            if plot[3] - 6 <= s.bbox[1] <= plot[3] + 40 and s.bbox[2] >= plot[0] - 10 and s.bbox[0] <= plot[2] + 10
        ]
        single = [s for s in below if len(s.text.split()) <= 2 or (s.bbox[2] - s.bbox[0]) < 0.5 * (plot[2] - plot[0])]
        multi = [s for s in below if s not in single]
        for s in single:
            cats.append(((s.bbox[0] + s.bbox[2]) / 2, s.text.strip()))
        if n_groups and len(single) < n_groups / 2:
            for s in multi:
                if len(s.text.split()) == n_groups:
                    for cx, _cy, t in _split_tokens(s):
                        cats.append((cx, t))
    cats.sort()
    merged: list[tuple[float, str]] = []
    for c, t in cats:
        if merged and abs(merged[-1][0] - c) <= 2.5:
            merged[-1] = (merged[-1][0], merged[-1][1] + " " + t)
        else:
            merged.append((c, t))
    return merged


def _outside(axis: Axis, plot: list[float], tol: float = 3.0) -> bool:
    """Tick labels live beside/below the plot, never inside the bars' box."""
    if axis.orient == "x":
        return axis.pos <= plot[1] + tol or axis.pos >= plot[3] - tol
    return axis.pos <= plot[0] + tol or axis.pos >= plot[2] - tol


def _nearest(coord: float, cats: list[tuple[float, str]], tol: float) -> str | None:
    best = None
    for c, t in cats:
        d = abs(c - coord)
        if d <= tol and (best is None or d < best[0]):
            best = (d, t)
    return best[1] if best else None


def _label_hit(bar: Bar, orient: str, s: Span, zero: float | None) -> bool:
    cx, cy = _span_center(s)
    if orient == "x":
        if not (bar.y0 - 1 <= cy <= bar.y1 + 1):
            return False
        if zero is not None and bar.x1 <= zero + ZERO_TOL:  # negative bar, label at left end
            return bar.x0 - 22 <= cx <= bar.x0 + max(6, bar.w * 0.35)
        return bar.x1 - max(6, bar.w * 0.35) <= cx <= bar.x1 + 22
    if not (bar.x0 - 1 <= cx <= bar.x1 + 1):
        return False
    if bar.y0 <= cy <= bar.y1 and bar.h >= 12:  # printed inside the bar
        return True
    if zero is not None and bar.y0 >= zero - ZERO_TOL:  # negative bar (below zero), label beneath
        return bar.y1 - max(6, bar.h * 0.35) <= cy <= bar.y1 + 14
    return bar.y0 - 14 <= cy <= bar.y0 + max(6, bar.h * 0.35)


def _printed_label(bar: Bar, orient: str, spans: list[Span], zero: float | None) -> str | None:
    """A numeric span sitting at (or just past) the bar's value end."""
    for s in spans:
        if is_numeric(s.text) and len(s.text.split()) <= 2 and _label_hit(bar, orient, s, zero):
            return clean_number(s.text)
    return None


def _bar_value(bar: Bar, axis: Axis) -> float:
    if axis.orient == "x":
        v0, v1 = axis.value(bar.x0), axis.value(bar.x1)
    else:
        v0, v1 = axis.value(bar.y1), axis.value(bar.y0)  # y grows downward
    tol = axis.step * 0.05
    if abs(v0) <= tol:
        return v1
    if abs(v1) <= tol:
        return v0
    return v1 - v0  # stacked segment


def reconstruct_region(region: ChartRegion, bars: list[Bar] | None = None, polys: list[Poly] | None = None) -> list[ChartTable]:
    if bars is None or polys is None:
        return []
    spans = region.spans
    all_axes = detect_axes(spans)
    axes = [a for a in all_axes if not _is_categorical(a.values, a.step)]
    cat_axes = [a for a in all_axes if _is_categorical(a.values, a.step)]
    legend, swatch_ids = _legend(bars, polys, spans)
    big = [b for b in bars if id(b) not in swatch_ids and (b.w >= 2 or b.h >= 2)]
    tables: list[ChartTable] = []

    if big:
        horizontal = sum(1 for b in big if b.w > b.h) >= len(big) / 2
        want = "x" if horizontal else "y"
        plot = [min(b.x0 for b in big), min(b.y0 for b in big), max(b.x1 for b in big), max(b.y1 for b in big)]
        cand = [a for a in axes if a.orient == want and _outside(a, plot)]
        axis = max(cand, key=lambda a: len(a.coords)) if cand else None
        labelled = sum(1 for b in big if _printed_label(b, want, spans, None) is not None)
        if axis is not None or labelled >= 0.6 * len(big):
            exclude = {id(sp) for sp in spans if sp.text.strip() in legend.values()}
            for other in axes:  # tick labels are not categories
                exclude.update(other.span_ids)
            # printed data labels are not categories either
            for b in big:
                for sp in spans:
                    if is_numeric(sp.text) and _label_hit(b, want, sp, None):
                        exclude.add(id(sp))
            thickness = sorted((b.h if horizontal else b.w) for b in big)[len(big) // 2]
            groups: list[float] = []
            for b in sorted(big, key=lambda b: (b.cy if horizontal else b.cx)):
                c = b.cy if horizontal else b.cx
                if not groups or c - groups[-1] > thickness * 1.2:
                    groups.append(c)
            cats = _categories(spans, want, plot, exclude, n_groups=len(groups))
            year_axis = next((a for a in cat_axes if a.orient == ("y" if horizontal else "x")), None)
            zero = axis.coord(0.0) if axis is not None else None
            colors: list[Color] = []
            for b in big:
                if b.fill not in colors:
                    colors.append(b.fill)
            # order series as in the legend when known
            colors.sort(key=lambda c: (c not in legend, list(legend).index(c) if c in legend else 0))
            series = [legend.get(c, f"Series {i + 1}") for i, c in enumerate(colors)]
            if len(series) == 1 and colors[0] not in legend:
                series = [_chart_title(spans, plot) or "Value"]
            rows: dict[str, list[str]] = {}
            order: list[str] = []
            for b in sorted(big, key=lambda b: (b.cy if horizontal else b.cx)):
                coord = b.cy if horizontal else b.cx
                if year_axis is not None:
                    cat = str(int(round(year_axis.value(coord))))
                else:
                    cat = _nearest(coord, cats, max(thickness * 1.6, 8))
                if cat is None:
                    cat = _round_value(coord, 1)
                printed = _printed_label(b, want, spans, zero)
                if printed is not None:
                    val = printed
                elif axis is not None:
                    val = _round_value(_bar_value(b, axis), axis.step)
                else:
                    continue
                if cat not in rows:
                    rows[cat] = [""] * len(series)
                    order.append(cat)
                ci = colors.index(b.fill)
                if rows[cat][ci] == "":
                    rows[cat][ci] = val
                else:  # same colour twice in a category: stacked/grouped w/o colour difference
                    rows[cat].append(val)
            tables.append(
                ChartTable(header=[""] + series, rows=[[c] + rows[c] for c in order], confidence=0.6, source="vector")
            )

    # line series
    y_axes = [a for a in axes if a.orient == "y"]
    if polys and y_axes:
        axis = max(y_axes, key=lambda a: len(a.coords))
        lines = [p for p in polys if (p.bbox[2] - p.bbox[0]) > 30 and len(p.points) >= 3]
        if lines:
            plot = [
                min(p.bbox[0] for p in lines),
                min(p.bbox[1] for p in lines),
                max(p.bbox[2] for p in lines),
                max(p.bbox[3] for p in lines),
            ]
            exclude = {id(sp) for sp in spans if sp.text.strip() in legend.values()}
            for other in axes:
                exclude.update(other.span_ids)
            cats = _categories(spans, "y", plot, exclude)
            if cats:
                series = [legend.get(p.color) or _end_label(p, spans) or f"Line {i + 1}" for i, p in enumerate(lines)]
                rows_l: list[list[str]] = []
                for cx, name in cats:
                    if cx < plot[0] - 10 or cx > plot[2] + 10:
                        continue
                    row = [name]
                    for p in lines:
                        row.append(_round_value(axis.value(_sample(p.points, cx)), axis.step))
                    rows_l.append(row)
                if rows_l:
                    tables.append(ChartTable(header=[""] + series, rows=rows_l, confidence=0.4, source="vector-line"))
    return tables


def _chart_title(spans: list[Span], plot: list[float]) -> str | None:
    """Nearest text line above the plot that is not figure furniture or an axis tick."""
    cands = [
        s for s in spans
        if s.text.strip() and not is_numeric(s.text) and not _FURNITURE_RE.match(s.text.strip())
        and s.bbox[3] <= plot[1] + 2 and len(s.text.strip()) >= 3
    ]
    if not cands:
        return None
    best = max(cands, key=lambda s: s.bbox[3])  # closest to the plot top
    # merge spans on the same line (title split into several spans)
    line = [s for s in cands if abs(s.bbox[1] - best.bbox[1]) <= 2]
    return " ".join(s.text.strip() for s in sorted(line, key=lambda s: s.bbox[0]))


def _end_label(p: Poly, spans: list[Span]) -> str | None:
    """Direct labelling: the text span just right of the line's last point."""
    xe, ye = max(p.points)
    best: tuple[float, str] | None = None
    for s in spans:
        if is_numeric(s.text) or not s.text.strip():
            continue
        cy = (s.bbox[1] + s.bbox[3]) / 2
        if 0 <= s.bbox[0] - xe <= 30 and abs(cy - ye) <= 8:
            d = (s.bbox[0] - xe) + abs(cy - ye)
            if best is None or d < best[0]:
                best = (d, s.text.strip())
    return best[1] if best else None


def _sample(points: list[tuple[float, float]], x: float) -> float:
    pts = sorted(points)
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def reconstruct_page(page: pymupdf.Page, tl: TextLayerPage) -> list[ChartTable]:
    bars, polys, _frames = page_primitives_full(page)
    swatches = _page_swatches(bars, polys, tl)
    out: list[ChartTable] = []
    cores = _plot_cores(page, tl)
    for core, grown in cores:
        region = region_from_page(page, tl, grown, pad=0.0)
        others = [c for c, _g in cores if c is not core]
        region.spans = [sp for sp in region.spans if not any(_dist(o, sp) < _dist(core, sp) for o in others)]
        rb = [b for b in bars if id(b) not in swatches and _near(core, [b.x0, b.y0, b.x1, b.y1], 0.5)]
        # legend swatches (and their names) belong to the nearest chart, up to LEGEND_REACH away
        for b in bars:
            if id(b) in swatches:
                box = [b.x0, b.y0, b.x1, b.y1]
                if _near(core, box, LEGEND_REACH) and not any(_near(o, box, 0.0) for o in others):
                    rb.append(b)
                    region.spans.extend(
                        sp for ln in tl.lines for sp in ln.spans
                        if sp not in region.spans and 0 <= sp.bbox[0] - b.x1 <= 16 and abs((sp.bbox[1] + sp.bbox[3]) / 2 - b.cy) <= max(b.h, 6)
                    )
        rp = [p for p in polys if _near(grown, p.bbox, 0.5)]
        out.extend(reconstruct_region(region, rb, rp))
    return out
