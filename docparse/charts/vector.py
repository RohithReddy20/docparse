"""Bar / line / area chart reconstruction from a born-digital PDF's drawings.

Bars are axis-aligned filled rectangles (``re`` items, or closed four-segment
``l`` quads, inside filled drawings), lines are stroked polylines. Numeric tick
spans aligned along one axis give a linear value scale, non-numeric spans along
the other axis give categories, legend swatches (tiny filled rects or short
stroked segments beside a text span) name the series by colour. A numeric span
printed at a bar's end is preferred over the geometric estimate. Filled area
bands are sampled at category positions; their thickness gives each series value.
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
    width: float = 1.0
    dash: str = ""
    filled: bool = False

    @property
    def style(self) -> tuple:
        return (self.color, round(self.width, 1), self.dash)


@dataclass
class Slice:
    cx: float
    cy: float
    a0: float  # start angle, degrees (0 = +x, counter-clockwise in page space)
    a1: float  # end angle
    sweep: float  # degrees
    radius: float
    fill: Color

    @property
    def mid(self) -> float:
        return (self.a0 + self.sweep / 2) % 360


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


def _angle(cx: float, cy: float, x: float, y: float) -> float:
    return math.degrees(math.atan2(-(y - cy), x - cx)) % 360


def page_slices(page: pymupdf.Page) -> list[Slice]:
    """Pie / donut slices: a filled path made of straight radii meeting at a common
    centre plus a chain of Bézier arcs. The sweep is read from the arc points."""
    out: list[Slice] = []
    for dr in page.get_drawings():
        fill = _rgb(dr.get("fill"))
        items = dr.get("items", [])
        curves = [it for it in items if it[0] == "c"]
        lines = [it for it in items if it[0] == "l"]
        if fill is None or not curves or _is_grey(fill) and min(fill) > 0.85:
            continue
        # arc chain endpoints and sample points (curve start/end/control)
        arc_pts = [(it[1].x, it[1].y) for it in curves] + [(it[4].x, it[4].y) for it in curves]
        # centre: endpoint shared by the straight segments (or the arc's two ends' radii)
        cands = [(it[1].x, it[1].y) for it in lines] + [(it[2].x, it[2].y) for it in lines]
        centre = None
        for c in cands:
            if sum(1 for o in cands if abs(o[0] - c[0]) <= 0.6 and abs(o[1] - c[1]) <= 0.6) >= 2:
                centre = c
                break
        if centre is None:
            continue
        cx, cy = centre
        radii = [math.hypot(x - cx, y - cy) for x, y in arc_pts]
        if not radii or max(radii) < 8:
            continue
        r = max(radii)
        # keep arc points on the outer rim (donuts also carry an inner arc)
        rim = [(x, y) for (x, y), rr in zip(arc_pts, radii) if rr >= 0.75 * r]
        angs = sorted(_angle(cx, cy, x, y) for x, y in rim)
        if len(angs) < 2:
            continue
        # the slice spans the arc that contains the sampled rim points: find the
        # largest angular gap between consecutive samples; the slice is the rest
        gaps = [(angs[(i + 1) % len(angs)] - angs[i]) % 360 for i in range(len(angs))]
        k = max(range(len(gaps)), key=lambda i: gaps[i])
        sweep = 360 - gaps[k]
        a0 = angs[(k + 1) % len(angs)]
        if sweep < 1 or sweep > 359:
            continue
        out.append(Slice(cx, cy, a0, (a0 + sweep) % 360, sweep, r, fill))
    return out


def pie_tables(slices: list[Slice], spans: list[Span], legend: dict[Color, str], title: str | None) -> list[ChartTable]:
    """Group slices by centre; one table per pie with category, printed value
    (or the geometric share) per slice. Labels are read along each slice's
    bisector (inside the slice or just past the rim); wrapped label lines are
    joined; the text line above the pie is its title."""
    groups: list[list[Slice]] = []
    for sl in slices:
        for g in groups:
            if abs(g[0].cx - sl.cx) <= 3 and abs(g[0].cy - sl.cy) <= 3:
                g.append(sl)
                break
        else:
            groups.append([sl])
    tables: list[ChartTable] = []
    text_spans = [sp for sp in spans if sp.text.strip() and not is_numeric(sp.text) and not _FURNITURE_RE.match(sp.text.strip())]
    labels = _merge_wrapped_labels(text_spans)
    for g in groups:
        total = sum(sl.sweep for sl in g)
        if len(g) < 2 or not (300 <= total <= 375):
            continue
        cx, cy, r = g[0].cx, g[0].cy, max(sl.radius for sl in g)
        rows: list[list[str]] = []
        used_labels: set[str] = set()
        for sl in sorted(g, key=lambda sl: -sl.sweep):
            label = legend.get(sl.fill)
            best_t: tuple[float, str] | None = None
            best_v: tuple[float, str] | None = None
            for b, t in labels:
                if re.search(r"\btotal\b", t, re.I):
                    continue  # "Total: $750,000" is the pie's title line
                if b[3] <= cy - r and not re.search(r"\d", t):
                    continue  # a line above the pie with no number is its title
                scx, scy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                dist = math.hypot(scx - sl.cx, scy - sl.cy)
                if dist > 2.0 * sl.radius or dist < 0.15 * sl.radius:
                    continue
                ang = _angle(sl.cx, sl.cy, scx, scy)
                diff = min((ang - sl.mid) % 360, (sl.mid - ang) % 360)
                if diff > max(sl.sweep / 2, 8):
                    continue
                score = diff + 4 * dist / sl.radius
                m = re.match(r"^(.*?)[\s:–-]*([-−]?\d[\d.,]*\s*%?)$", t)
                if m and len(m.group(1).strip()) >= 2:
                    t_label, t_val = m.group(1).strip(), clean_number(m.group(2))
                    if best_v is None or score < best_v[0]:
                        best_v = (score, t_val)
                    t = t_label
                if len(t) <= 80 and (best_t is None or score < best_t[0]):
                    best_t = (score, t)
            for sp in spans:
                t = sp.text.strip()
                if not is_numeric(t):
                    continue
                scx, scy = _span_center(sp)
                dist = math.hypot(scx - sl.cx, scy - sl.cy)
                if dist > 2.4 * sl.radius or dist < 0.1 * sl.radius:
                    continue
                ang = _angle(sl.cx, sl.cy, scx, scy)
                diff = min((ang - sl.mid) % 360, (sl.mid - ang) % 360)
                if diff > max(sl.sweep / 2, 8):
                    continue
                score = diff + 4 * dist / sl.radius
                if best_v is None or score < best_v[0]:
                    best_v = (score, clean_number(t))
            if label is None and best_t is not None:
                label = best_t[1]
            value = best_v[1] if best_v is not None else _round_value(100 * sl.sweep / total, 0.5)
            if label:
                rows.append([label, value])
                used_labels.add(label)
        # title: nearest unused text line above the pie, overlapping it horizontally
        above = [
            (b, t) for b, t in labels
            if b[3] <= cy - r + 2 and cy - r - b[3] <= 80 and min(b[2], cx + r) - max(b[0], cx - r) > 0
            and not any(t == u or t.startswith(u) for u in used_labels)
        ]
        pie_title = title or (max(above, key=lambda bt: bt[0][3])[1] if above else "")
        if len(rows) >= 2:
            tables.append(ChartTable(header=[pie_title, "Share (%)"], rows=rows, confidence=0.5, source="vector-pie", caption=pie_title))
    return tables


def _filled_polygons(items: list, fill: Color) -> list[Poly]:
    if not items or any(it[0] != "l" for it in items):
        return []
    paths: list[list[tuple[float, float]]] = []
    for _, start, end in items:
        p0, p1 = (start.x, start.y), (end.x, end.y)
        if not paths or math.dist(paths[-1][-1], p0) > 0.01:
            paths.append([p0])
        paths[-1].append(p1)
    out: list[Poly] = []
    for points in paths:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        if len(set(points)) < 4 or max(xs) - min(xs) <= 30 or max(ys) - min(ys) < MIN_BAR:
            continue
        if len({round(x, 1) for x in xs}) == len({round(y, 1) for y in ys}) == 2:
            continue  # rectangles already belong to the bar extractor
        out.append(Poly(points, fill, [min(xs), min(ys), max(xs), max(ys)], filled=True))
    return out


def page_primitives_full(page: pymupdf.Page) -> tuple[list[Bar], list[Poly], list[list[float]]]:
    """Bars, filled polygons, coloured lines, and grey frame/grid path bboxes."""
    bars: list[Bar] = []
    polys: list[Poly] = []
    frames: list[list[float]] = []
    page_area = page.rect.width * page.rect.height
    for dr in page.get_drawings():
        fill = _rgb(dr.get("fill"))
        color = _rgb(dr.get("color"))
        items = dr.get("items", [])
        if fill is not None:
            if min(fill[:3]) < 0.97 and dr["rect"].get_area() < 0.35 * page_area:
                polys.extend(_filled_polygons(items, fill))
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
                if min(fill[:3]) >= 0.97 and w > 40 and h > 40:
                    continue  # white plot background / card
                if min(fill[:3]) >= 0.9 and ((w > 60 and h > 20) or (h > 60 and w > 20)) and w * h > 3000:
                    continue  # pale title band / panel background
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
            if len(pts) >= 2:
                r = dr["rect"]
                polys.append(Poly(pts, color, [r.x0, r.y0, r.x1, r.y1], float(dr.get("width") or 1.0), str(dr.get("dashes") or "")))
    return bars, _chain(polys), frames


def _chain(polys: list[Poly]) -> list[Poly]:
    """Lines are often emitted one segment per drawing (dashed series especially).
    Join consecutive same-style segments whose endpoints meet into one polyline."""
    by_style: dict[tuple, list[Poly]] = defaultdict(list)
    for p in polys:
        by_style[p.style].append(p)
    out = [p for p in polys if p.filled]
    for style, group in by_style.items():
        group = [p for p in group if not p.filled]
        group.sort(key=lambda p: (min(x for x, _ in p.points), min(y for _, y in p.points)))
        chains: list[Poly] = []
        for p in group:
            attached = False
            for c in chains:
                tail = c.points[-1]
                head = p.points[0]
                if abs(tail[0] - head[0]) <= 0.6 and abs(tail[1] - head[1]) <= 0.6:
                    c.points.extend(p.points[1:])
                    c.bbox = _union(c.bbox, p.bbox)
                    attached = True
                    break
                if abs(c.points[0][0] - p.points[-1][0]) <= 0.6 and abs(c.points[0][1] - p.points[-1][1]) <= 0.6:
                    c.points = p.points[:-1] + c.points
                    c.bbox = _union(c.bbox, p.bbox)
                    attached = True
                    break
            if not attached:
                chains.append(Poly(list(p.points), p.color, list(p.bbox), p.width, p.dash))
        out.extend(chains)
    return out


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
    legend, swatches = _legend(bars, polys, [sp for ln in tl.lines for sp in ln.spans])
    polys = [p for p in polys if not p.filled or p.color in legend]
    big = [b for b in bars if id(b) not in swatches and (b.w > MAX_SWATCH or b.h > MAX_SWATCH or b.w * b.h >= 30)]
    boxes = [[b.x0, b.y0, b.x1, b.y1] for b in big]
    boxes += [p.bbox for p in polys if (p.bbox[2] - p.bbox[0]) > 20 or (p.bbox[3] - p.bbox[1]) > 20]
    if not boxes:
        return []
    # grey gridlines/axes extend a plot area; they bridge two bar clusters only when
    # one frame encloses both (positive and negative bars of one chart), never
    # side-by-side panels sharing a page-wide background.
    cores0 = _cluster(boxes, CORE_GAP)

    def inside(a: list[float], f: list[float]) -> bool:
        ix = max(0.0, min(a[2], f[2]) - max(a[0], f[0]))
        iy = max(0.0, min(a[3], f[3]) - max(a[1], f[1]))
        return ix * iy >= 0.8 * max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))

    merged = True
    while merged and len(cores0) > 1:
        merged = False
        for i in range(len(cores0)):
            for j in range(i + 1, len(cores0)):
                a, b = cores0[i], cores0[j]
                wide = max(a[2] - a[0], b[2] - b[0])
                if any(inside(a, f) and inside(b, f) and (f[2] - f[0]) <= 1.6 * wide for f in frames):
                    cores0[i] = _union(a, b)
                    del cores0[j]
                    merged = True
                    break
            if merged:
                break
    extended: list[list[float]] = []
    for core in cores0:
        ext = list(core)
        for f in frames:
            if _near(f, core, 2.0) and not any(_near(f, o, 2.0) for o in cores0 if o is not core):
                ext = _union(ext, f)
        extended.append(ext)
    out: list[tuple[list[float], list[float]]] = []
    for core in extended:
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
                if _dist(core, sp) <= SPAN_GAP or _near(core, list(sp.bbox), SPAN_GAP * 0.5):
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
            # 1-D gap clustering on the alignment key (rounding splits a column
            # whose key straddles a bucket boundary)
            last_key = None
            gid = -1
            for s, v in sorted(nums, key=lambda m: keyfn(m[0])):
                k = keyfn(s)
                if last_key is None or k - last_key > ALIGN_TOL:
                    gid += 1
                last_key = k
                groups[gid].append((s, v))
            for key, members in groups.items():
                if len(members) < 3:
                    continue
                key = sum(keyfn(m[0]) for m in members) / len(members) / ALIGN_TOL
                members.sort(key=lambda m: _span_center(m[0])[0 if orient == "x" else 1])
                # stacked panels share one tick column: split into monotone runs
                for run in _monotone_runs(members):
                    if len(run) < 3:
                        continue
                    ids = tuple(sorted(id(m[0]) for m in run))
                    if ids in seen:
                        continue
                    coords = [_span_center(m[0])[0 if orient == "x" else 1] for m in run]
                    values = [m[1] for m in run]
                    diffs = [b - a for a, b in zip(values, values[1:])]
                    a, b, r2 = _fit(coords, values)
                    if r2 < 0.995 or a == 0:
                        continue
                    seen.add(ids)
                    step = min(abs(d) for d in diffs) or 1.0
                    axes.append(Axis(orient, a, b, step, coords, values, key * ALIGN_TOL, list(ids)))
    return axes


def _monotone_runs(members: list[tuple[Span, float]]) -> list[list[tuple[Span, float]]]:
    """Maximal strictly monotone runs (by value) of coordinate-sorted tick spans."""
    runs: list[list[tuple[Span, float]]] = []
    cur: list[tuple[Span, float]] = []
    sign = 0
    for m in members:
        if cur:
            d = m[1] - cur[-1][1]
            s = (d > 0) - (d < 0)
            if s == 0 or (sign and s != sign):
                runs.append(cur)
                cur, sign = [cur[-1]] if s != 0 else [], 0
                if cur:
                    sign = s
            else:
                sign = s
        cur.append(m)
    if cur:
        runs.append(cur)
    return runs


def _round_value(v: float, step: float, fine: bool = False) -> str:
    """Round a geometric estimate the way a human reads an axis: whole units when
    ticks are ≥2 apart, otherwise a tenth of the tick step. ``fine`` (line charts,
    whose annotated values carry a decimal) keeps a twentieth of the step."""
    if step <= 0:
        return f"{v:.2f}"
    if step >= 2 and not fine:
        return str(int(round(v))) if round(v) != 0 else "0"
    decimals = max(0, min(4, int(math.ceil(-math.log10(step / (20 if fine else 10))))))
    s = f"{v:.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


# --------------------------------------------------------------------------- legend / series


def _legend(
    bars: list[Bar], polys: list[Poly], spans: list[Span], near: list[float] | None = None
) -> tuple[dict[Color, str], set[int]]:
    """Series name per colour from legend swatches, plus the ids of swatch bars.
    With ``near``, only swatches within LEGEND_REACH of that box count (a page
    with several charts reuses colours across legends)."""
    names: dict[Color, str] = {}
    swatches: set[int] = set()
    if near is not None:
        bars = [b for b in bars if _near(near, [b.x0, b.y0, b.x1, b.y1], LEGEND_REACH)]
        polys = [p for p in polys if _near(near, p.bbox, LEGEND_REACH)]
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
        if p.filled:
            continue
        w, h = p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]
        if 4 <= w <= 30 and h <= 4 and len(p.points) <= 6:
            name = beside(p.bbox[2], (p.bbox[1] + p.bbox[3]) / 2, 6)
            if name and p.color not in names:
                names[p.color] = name
    return names, swatches



def _infer_zero(bars: list[Bar], horizontal: bool) -> float | None:
    """Bars grow from a common baseline: the most shared edge coordinate (positive
    bars end on it, negative bars start on it)."""
    edges: dict[float, int] = defaultdict(int)
    for b in bars:
        for e in ((b.x0, b.x1) if horizontal else (b.y0, b.y1)):
            edges[round(e, 0)] += 1
    if not edges:
        return None
    coord, n = max(edges.items(), key=lambda kv: kv[1])
    if n >= 2 and n >= 0.4 * len(bars):
        return float(coord)
    return None


def _merge_wrapped_labels(spans: list[Span]) -> list[tuple[list[float], str]]:
    """Short text lines that wrap (same left edge, next line within a third of a
    line height) read as one label. Merging runs per left-edge column so labels
    of another column printed between two wrapped lines do not break the join."""
    items = sorted(((list(s.bbox), s.text.strip()) for s in spans if s.text.strip()), key=lambda t: (t[0][0], t[0][1]))
    columns: list[list[tuple[list[float], str]]] = []

    def same_column(a: list[float], b: list[float]) -> bool:  # left-aligned or centred
        return abs(a[0] - b[0]) <= 8 or abs((a[0] + a[2]) - (b[0] + b[2])) / 2 <= 8

    for box, text in items:
        for col in columns:
            if any(same_column(c[0], box) for c in col[-2:]):
                col.append((box, text))
                break
        else:
            columns.append([(box, text)])
    out: list[tuple[list[float], str]] = []
    for col in columns:
        col.sort(key=lambda t: t[0][1])
        merged: list[tuple[list[float], str]] = []
        last_h = 1.0
        for box, text in col:
            h = max(box[3] - box[1], 1.0)
            if merged:
                pb, pt = merged[-1]
                if -0.35 * last_h <= box[1] - pb[3] <= 0.3 * last_h:
                    merged[-1] = (_union(pb, box), pt + " " + text)
                    last_h = h
                    continue
            merged.append((box, text))
            last_h = h
        out.extend(merged)
    out.sort(key=lambda t: (t[0][1], t[0][0]))
    return out


def _side_labels(spans: list[Span], plot: list[float], exclude: set[int]) -> list[tuple[list[float], str]]:
    """Non-numeric labels left/right of the plot (legend entries without swatches,
    sign labels of diverging charts), wrapped lines merged."""
    pool = [
        s for s in spans
        if s.text.strip() and not is_numeric(s.text) and id(s) not in exclude
        and not _FURNITURE_RE.match(s.text.strip()) and len(s.text.strip()) <= 60
        and (s.bbox[2] <= plot[0] + 2 or s.bbox[0] >= plot[2] - 2)
        and s.bbox[3] >= plot[1] - 10 and s.bbox[1] <= plot[3] + 40
        and (s.bbox[2] - s.bbox[0]) <= max(0.6 * (plot[2] - plot[0]), 60.0)
    ]
    return _merge_wrapped_labels(pool)


def _sign_legend(spans: list[Span], plot: list[float], zero: float, horizontal: bool, exclude: set[int]) -> tuple[str, str] | None:
    """Diverging charts label the two sides of the baseline ('Favorable attitude' /
    'Unfavorable attitude') instead of drawing swatches."""
    if horizontal:
        return None
    labels = _side_labels(spans, plot, exclude)
    for side in (
        [t for t in labels if t[0][2] <= plot[0] + 2],
        [t for t in labels if t[0][0] >= plot[2] - 2],
    ):
        above = [t for b, t in side if (b[1] + b[3]) / 2 < zero - 2]
        below = [t for b, t in side if (b[1] + b[3]) / 2 > zero + 2]
        if len(above) == 1 and len(below) == 1:
            return above[0], below[0]
    return None


def _ordinal_legend(spans: list[Span], plot: list[float], colors: list[Color], exclude: set[int]) -> list[str] | None:
    """Legend entries printed as a stack of short labels beside the plot with no
    swatch primitives: map them to series colours in order. The legend is the
    side whose label count equals the series count (categories fill the other)."""
    if len(colors) < 2:
        return None
    labels = _side_labels(spans, plot, exclude)
    left = [t for t in labels if t[0][2] <= plot[0] + 2]
    right = [t for t in labels if t not in left]
    for side in (right, left):
        if len(side) == len(colors):
            side.sort(key=lambda t: (t[0][1], t[0][0]))
            return [t for _b, t in side]
    return None


def _panel_title(spans: list[Span], core: list[float], reach: float = 40.0) -> str | None:
    """Bold / short text line just above a plot core (a small-multiple's name)."""
    cands = [
        s for s in spans
        if s.text.strip() and not is_numeric(s.text) and not _FURNITURE_RE.match(s.text.strip())
        and 0 <= core[1] - s.bbox[3] <= reach
        and min(s.bbox[2], core[2]) - max(s.bbox[0], core[0]) >= 0.5 * (s.bbox[2] - s.bbox[0])
        and len(s.text.strip()) <= 60
    ]
    if not cands:
        return None
    best = max(cands, key=lambda s: s.bbox[3])
    line = sorted((s for s in cands if abs(s.bbox[1] - best.bbox[1]) <= 2), key=lambda s: s.bbox[0])
    return " ".join(s.text.strip() for s in line)


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
        return bar.y1 - max(6, bar.h * 0.35) <= cy <= bar.y1 + 18
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


def _horizontal(big: list[Bar], tol: float = 1.0) -> bool:
    """Bar orientation from arrangement, not aspect: stacked segments of one bar
    share both edges across the stacking axis; simple bars share a baseline."""
    same_col = same_row = 0
    for i, a in enumerate(big):
        for b in big[i + 1 :]:
            # touching segments of one stacked bar
            if abs(a.x0 - b.x0) <= tol and abs(a.x1 - b.x1) <= tol and (abs(a.y1 - b.y0) <= tol or abs(b.y1 - a.y0) <= tol):
                same_col += 1
            if abs(a.y0 - b.y0) <= tol and abs(a.y1 - b.y1) <= tol and (abs(a.x1 - b.x0) <= tol or abs(b.x1 - a.x0) <= tol):
                same_row += 1

    def most_common(vals: list[float]) -> int:
        vals = sorted(vals)
        best = run = 1
        for p, q in zip(vals, vals[1:]):
            run = run + 1 if q - p <= tol else 1
            best = max(best, run)
        return best

    v = same_col + max(most_common([b.y1 for b in big]), most_common([b.y0 for b in big]))
    h = same_row + max(most_common([b.x0 for b in big]), most_common([b.x1 for b in big]))
    if v != h:
        return h > v
    return sum(1 for b in big if b.w > b.h) >= len(big) / 2


def _drop_containers(big: list[Bar]) -> list[Bar]:
    """Filled rects enclosing several bars are backgrounds, not bars."""
    out: list[Bar] = []
    for b in big:
        inside = sum(
            1 for o in big
            if o is not b and o.x0 >= b.x0 - 0.5 and o.x1 <= b.x1 + 0.5 and o.y0 >= b.y0 - 0.5 and o.y1 <= b.y1 + 0.5
        )
        if inside < 3:
            out.append(b)
    return out


def _area_bounds(poly: Poly, x: float) -> tuple[float, float] | None:
    if not poly.bbox[0] - 0.5 <= x <= poly.bbox[2] + 0.5:
        return None
    x = min(max(x, poly.bbox[0]), poly.bbox[2])
    hits: list[float] = []
    for (x0, y0), (x1, y1) in zip(poly.points, poly.points[1:] + poly.points[:1]):
        if x0 == x1:
            if abs(x - x0) < 0.01:
                hits.extend((y0, y1))
        elif min(x0, x1) <= x <= max(x0, x1):
            hits.append(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    distinct: list[float] = []
    for y in sorted(hits):
        if not distinct or y - distinct[-1] > 0.01:
            distinct.append(y)
    if not 1 <= len(distinct) <= 2:
        return None
    return distinct[0], distinct[-1]


def _area_tables(
    polys: list[Poly], spans: list[Span], axes: list[Axis], legend: dict[Color, str], panel: str | None
) -> list[ChartTable]:
    areas = [p for p in polys if p.filled and p.color in legend]
    if not areas:
        return []
    plot = [min(p.bbox[0] for p in areas), min(p.bbox[1] for p in areas),
            max(p.bbox[2] for p in areas), max(p.bbox[3] for p in areas)]
    y_axes = [a for a in axes if a.orient == "y" and not _is_categorical(a.values, a.step) and _outside(a, plot)]
    if not y_axes:
        return []
    axis = max(y_axes, key=lambda a: len(a.coords))
    exclude = {id(sp) for sp in spans if sp.text.strip() in legend.values()}
    for a in y_axes:
        exclude.update(a.span_ids)
    year_axes = [a for a in axes if a.orient == "x" and _is_categorical(a.values, a.step) and _outside(a, plot)]
    if year_axes:
        years = max(year_axes, key=lambda a: len(a.coords))
        cats = [(x, str(int(v))) for x, v in zip(years.coords, years.values)]
    else:
        below = [sp for sp in spans if id(sp) not in exclude and plot[3] - 6 <= sp.bbox[1] <= plot[3] + 40]
        top = min((sp.bbox[1] for sp in below), default=plot[3])
        cats = _categories([sp for sp in below if sp.bbox[1] <= top + 4], "y", plot, exclude)
    if len(cats) < 2:
        return []
    colors = [c for c in legend if any(p.color == c for p in areas)]
    rows: list[list[str]] = []
    for x, name in cats:
        values: list[str] = []
        for color in colors:
            bands = [_area_bounds(p, x) for p in areas if p.color == color]
            bands = [b for b in bands if b is not None]
            value = sum(abs(axis.value(top) - axis.value(bottom)) for top, bottom in bands)
            values.append(_round_value(value, axis.step, fine=True) if bands else "")
        if any(values):
            rows.append([f"{panel} – {name}" if panel else name] + values)
    if len(rows) < 2:
        return []
    title = _chart_title(spans, plot) or ""
    return [ChartTable(header=[title] + [legend[c] for c in colors], rows=rows,
                       confidence=0.6, source="vector-area", caption=_caption(title, panel))]


def reconstruct_region(
    region: ChartRegion,
    bars: list[Bar] | None = None,
    polys: list[Poly] | None = None,
    page_legend: dict[Color, str] | None = None,
    panel: str | None = None,
    page_axes: list[Axis] | None = None,
) -> list[ChartTable]:
    if bars is None or polys is None:
        return []
    spans = region.spans
    all_axes = detect_axes(spans)
    axes = [a for a in all_axes if not _is_categorical(a.values, a.step)]
    cat_axes = [a for a in all_axes if _is_categorical(a.values, a.step)]
    legend, swatch_ids = _legend(bars, polys, spans)
    for c, n in (page_legend or {}).items():
        legend[c] = n  # caller resolved colour collisions by proximity
    big = _drop_containers([b for b in bars if id(b) not in swatch_ids and (b.w >= 2 or b.h >= 2)])
    tables: list[ChartTable] = []

    if big:
        horizontal = _horizontal(big)
        # bullets / tick marks are far thinner than the bars
        thick = sorted((b.h if horizontal else b.w) for b in big)[len(big) // 2]
        big = [b for b in big if (b.h if horizontal else b.w) >= 0.4 * thick] or big
        want = "x" if horizontal else "y"
        plot = [min(b.x0 for b in big), min(b.y0 for b in big), max(b.x1 for b in big), max(b.y1 for b in big)]
        cand = [a for a in axes if a.orient == want and _outside(a, plot)]
        if not cand and page_axes:  # panels sharing one axis (small multiples)
            lo, hi = (plot[0], plot[2]) if want == "x" else (plot[1], plot[3])
            cand = [
                a for a in page_axes
                if a.orient == want and not _is_categorical(a.values, a.step) and _outside(a, plot)
                and min(hi, max(a.coords)) - max(lo, min(a.coords)) >= 0.5 * (hi - lo)
            ]
        axis = max(cand, key=lambda a: len(a.coords)) if cand else None
        zero = axis.coord(0.0) if axis is not None else _infer_zero(big, horizontal)
        labelled = sum(1 for b in big if _printed_label(b, want, spans, zero) is not None)
        if axis is not None or labelled >= 0.6 * len(big):
            exclude = {id(sp) for sp in spans if sp.text.strip() in legend.values()}
            for other in axes:  # tick labels are not categories
                exclude.update(other.span_ids)
            # printed data labels are not categories either
            for b in big:
                for sp in spans:
                    if is_numeric(sp.text) and _label_hit(b, want, sp, zero):
                        exclude.add(id(sp))
            thickness = sorted((b.h if horizontal else b.w) for b in big)[len(big) // 2]
            groups: list[float] = []
            for b in sorted(big, key=lambda b: (b.cy if horizontal else b.cx)):
                c = b.cy if horizontal else b.cx
                if not groups or c - groups[-1] > thickness * 1.2:
                    groups.append(c)
            colors: list[Color] = []
            for b in sorted(big, key=lambda b: ((b.cy, b.x0) if horizontal else (b.cx, -b.y1))):
                if b.fill not in colors:
                    colors.append(b.fill)
            # order series as in the legend when known
            colors.sort(key=lambda c: (c not in legend, list(legend).index(c) if c in legend else 0))
            sign_names: tuple[str, str] | None = None
            if not any(c in legend for c in colors):
                if zero is not None:
                    sign_names = _sign_legend(spans, plot, zero, horizontal, exclude)
                if not sign_names:
                    ordinal = _ordinal_legend(spans, plot, colors, exclude)
                    if ordinal:
                        legend.update(zip(colors, ordinal))
            exclude |= {id(sp) for sp in spans if sp.text.strip() in legend.values()}
            if sign_names:
                exclude |= {id(sp) for sp in spans if any(sp.text.strip() in n for n in sign_names)}
            cats = _categories(spans, want, plot, exclude, n_groups=len(groups))
            year_axis = next((a for a in cat_axes if a.orient == ("y" if horizontal else "x")), None)
            if sign_names:
                series = list(sign_names)
            else:
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
                if sign_names:
                    try:
                        neg = float(val) < 0
                    except ValueError:
                        neg = (b.y0 >= zero - ZERO_TOL) if zero is not None else False
                    ci = 1 if neg else 0
                else:
                    ci = colors.index(b.fill)
                if rows[cat][ci] == "":
                    rows[cat][ci] = val
                else:  # same colour twice in a category: stacked/grouped w/o colour difference
                    rows[cat].append(val)
            title = _chart_title(spans, plot) or ""
            cap = _caption(title, panel)
            pre = f"{panel} – " if panel else ""
            tables.append(
                ChartTable(header=[title] + series, rows=[[pre + c] + rows[c] for c in order], confidence=0.6, source="vector", caption=cap)
            )

    tables.extend(_area_tables(polys, spans, all_axes, legend, panel))

    # line series
    y_axes = [a for a in axes if a.orient == "y"]
    if polys and y_axes:
        axis = max(y_axes, key=lambda a: len(a.coords))
        chains = [p for p in polys if not p.filled and (p.bbox[2] - p.bbox[0]) > 30 and len(p.points) >= 3]
        # one series per stroke style; a series may be drawn as several chains
        by_style: dict[tuple, list[Poly]] = {}
        for p in chains:
            by_style.setdefault(p.style, []).append(p)
        lines = [
            Poly(
                sorted(pt for c in group for pt in c.points), group[0].color,
                [min(c.bbox[0] for c in group), min(c.bbox[1] for c in group), max(c.bbox[2] for c in group), max(c.bbox[3] for c in group)],
                group[0].width, group[0].dash,
            )
            for group in by_style.values()
        ]
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
                per_series = [_side_axis(name, y_axes, plot) or axis for name in series]
                rows_l: list[list[str]] = []
                pre = f"{panel} – " if panel else ""
                for cx, name in cats:
                    if cx < plot[0] - 10 or cx > plot[2] + 10:
                        continue
                    row = [pre + name]
                    for p, ax in zip(lines, per_series):
                        row.append(_round_value(ax.value(_sample(p.points, cx)), ax.step, fine=True))
                    rows_l.append(row)
                if rows_l:
                    title = _chart_title(spans, plot) or ""
                    cap = _caption(title, panel)
                    tables.append(ChartTable(header=[title] + series, rows=rows_l, confidence=0.4, source="vector-line", caption=cap))
    return tables


def _caption(title: str, panel: str | None) -> str:
    parts = [t for t in (title, panel) if t]
    if len(parts) == 2 and parts[0].strip().lower() == parts[1].strip().lower():
        parts = parts[:1]
    return " – ".join(parts)


def _chart_title(spans: list[Span], plot: list[float]) -> str | None:
    """Nearest text line above the plot that is not figure furniture or an axis tick."""
    cands = [
        s for s in spans
        if s.text.strip() and not is_numeric(s.text) and not _FURNITURE_RE.match(s.text.strip())
        and s.bbox[3] <= plot[1] + 2 and len(s.text.strip()) >= 3
        and (
            min(s.bbox[2], plot[2]) - max(s.bbox[0], plot[0]) >= 0.5 * (s.bbox[2] - s.bbox[0])
            or (s.bbox[2] <= plot[2] and plot[0] - s.bbox[2] <= 80)  # left-aligned title outside the bars
        )
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


_RHS = re.compile(r"\b(rhs|rha|right(?:[- ]hand)?(?: axis| scale)?|r\.h\.s\.?)\b", re.I)
_LHS = re.compile(r"\b(lhs|lha|left(?:[- ]hand)?(?: axis| scale)?|l\.h\.s\.?)\b", re.I)


def _side_axis(name: str, y_axes: list[Axis], plot: list[float]) -> Axis | None:
    """Dual-axis charts: a series tagged (RHS)/(LHS) reads off that side's axis."""
    if len(y_axes) < 2:
        return None
    if _RHS.search(name):
        right = [a for a in y_axes if a.pos >= plot[2] - 3]
        return max(right, key=lambda a: len(a.coords)) if right else None
    if _LHS.search(name):
        left = [a for a in y_axes if a.pos <= plot[0] + 3]
        return max(left, key=lambda a: len(a.coords)) if left else None
    return None


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
    all_spans = [sp for ln in tl.lines for sp in ln.spans]
    page_legend, swatches = _legend(bars, polys, all_spans)
    out: list[ChartTable] = []
    slices = page_slices(page)
    if slices:
        out.extend(pie_tables(slices, all_spans, page_legend, None))
    cores = _plot_cores(page, tl)
    multi = len(cores) >= 2
    page_axes = detect_axes(all_spans) if multi else []
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
        panel = _panel_title(all_spans, core) if multi else None
        local, _ = _legend(bars, polys, all_spans, near=core)
        legend = dict(local)
        for c, n in page_legend.items():
            legend.setdefault(c, n)
        out.extend(reconstruct_region(region, rb, rp, page_legend=legend, panel=panel, page_axes=page_axes))
    return out
