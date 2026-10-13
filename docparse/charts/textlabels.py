"""Chart tables from the PDF text layer: printed data labels, category labels, legend.

Born-digital charts print their data labels, axis ticks, categories and legend as
positioned text spans. This module recovers a ``ChartTable`` from those spans alone
(``labels_table``), and corrects a VLM's chart table with them (``snap_table``).
No vision, no GPU.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, replace

import pymupdf
from rapidfuzz import fuzz

from docparse.charts import ChartRegion, ChartTable, clean_number, is_numeric, parse_html_table, region_from_page
from docparse.textlayer import Span, TextLayerPage

# --------------------------------------------------------------------------- geometry

_Y_TOL = 2.5  # pts: spans on the same text line
_X_TOL = 4.0  # pts: spans aligned on the same column edge


@dataclass
class _S:
    span: Span
    text: str
    num: bool

    @property
    def cx(self) -> float:
        return (self.span.bbox[0] + self.span.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.span.bbox[1] + self.span.bbox[3]) / 2

    @property
    def x0(self) -> float:
        return self.span.bbox[0]

    @property
    def x1(self) -> float:
        return self.span.bbox[2]

    @property
    def y0(self) -> float:
        return self.span.bbox[1]

    @property
    def y1(self) -> float:
        return self.span.bbox[3]

    @property
    def h(self) -> float:
        return self.span.bbox[3] - self.span.bbox[1]


_PLACEHOLDER_RE = re.compile(r"^(n/?a|n\.a\.|-|–|—|x|\*)$", re.I)
_YEAR_RE = re.compile(r"^(19|20)\d\d(/\d{2,4})?$")


_PAREN_RE = re.compile(r"^([-−+]?\s*[$€£¥]?\s*\d[\d\s,.]*\s*%?)\s*\(([-−+]?\s*\d[\d\s,.]*\s*%?)\)$")


def _is_datum(t: str) -> bool:
    if _PAREN_RE.match(t):
        return True
    return (is_numeric(t) and not _YEAR_RE.match(t)) or bool(_PLACEHOLDER_RE.match(t))


def _cell(t: str) -> str:
    """Cell text for a datum: numbers cleaned, 'a (b%)' kept as printed."""
    t = t.strip().replace("−", "-")
    if _PAREN_RE.match(t) or _PLACEHOLDER_RE.match(t):
        return re.sub(r"\s+", " ", t)
    return clean_number(t)


def _wrap(spans: list[Span]) -> list[_S]:
    out: list[_S] = []
    for s in spans:
        t = s.text.strip()
        if t:
            out.append(_S(s, t, _is_datum(t)))
    return out


def _merge_line_fragments(items: list[_S]) -> list[_S]:
    """Join spans that are pieces of one label (same line, tiny gap, same numeric-ness)."""
    items = sorted(items, key=lambda s: (round(s.cy / _Y_TOL), s.x0))
    out: list[_S] = []
    for s in items:
        if out:
            p = out[-1]
            same_line = abs(p.cy - s.cy) <= _Y_TOL
            gap = s.x0 - p.x1
            pct = s.text in ("%", "％") and p.num
            if same_line and -1 <= gap <= 0.6 * max(p.h, s.h) and (pct or (p.num == s.num and not (p.num and s.num and gap > 2))):
                merged = Span(
                    text=p.span.text.rstrip() + (" " if gap > 0.5 else "") + s.span.text.lstrip(),
                    bbox=[p.x0, min(p.y0, s.y0), s.x1, max(p.y1, s.y1)],
                    size=p.span.size,
                    bold=p.span.bold,
                )
                out[-1] = _S(merged, merged.text.strip(), _is_datum(merged.text))
                continue
        out.append(s)
    return out


def _value(s: _S) -> float | None:
    try:
        return float(clean_number(s.text))
    except ValueError:
        return None


def _is_arithmetic(vals: list[float]) -> bool:
    if len(vals) < 3:
        return False
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    if any(d == 0 for d in diffs):
        return False
    if not (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)):
        return False
    ref = statistics.median(abs(d) for d in diffs)
    return all(abs(abs(d) - ref) <= 0.08 * ref + 1e-9 for d in diffs)


def _tick_groups(nums: list[_S]) -> set[int]:
    """ids of numeric spans that form axis tick scales (aligned, evenly stepped values)."""
    ticks: set[int] = set()
    # horizontal axes: same y
    for key in ("y", "x"):
        groups: dict[int, list[_S]] = {}
        for s in nums:
            k = round((s.cy if key == "y" else s.x1) / (_Y_TOL if key == "y" else _X_TOL))
            groups.setdefault(k, []).append(s)
        # also tolerate neighbouring bins
        keys = sorted(groups)
        merged: list[list[_S]] = []
        for k in keys:
            if merged and k - merged[-1][-1]._bin <= 1:  # type: ignore[attr-defined]
                merged[-1].extend(groups[k])
            else:
                merged.append(list(groups[k]))
            for s in groups[k]:
                s._bin = k  # type: ignore[attr-defined]
        for g in merged:
            if len(g) < 3:
                continue
            g = sorted(g, key=lambda s: s.cx if key == "y" else s.cy)
            vals = [_value(s) for s in g]
            if any(v is None for v in vals):
                continue
            # positions must be evenly spaced too
            pos = [s.cx if key == "y" else s.cy for s in g]
            if _is_arithmetic(vals) and _is_arithmetic(pos):  # type: ignore[arg-type]
                ticks.update(id(s) for s in g)
            else:
                # try the longest arithmetic run of length >= 4
                for i in range(len(g)):
                    for j in range(len(g), i + 3, -1):
                        sub = g[i:j]
                        sv = [_value(s) for s in sub]
                        sp = [s.cx if key == "y" else s.cy for s in sub]
                        if len(sub) >= 4 and _is_arithmetic(sv) and _is_arithmetic(sp):  # type: ignore[arg-type]
                            ticks.update(id(s) for s in sub)
                            break
    return ticks


def _alignment_groups(items: list[_S], axis: str, edges: tuple[str, ...] = ("cy",)) -> list[list[_S]]:
    """Group spans sharing a y line (axis='y'; ``edges`` adds top/bottom alignment for
    rotated labels) or an x edge (axis='x')."""
    groups: list[list[_S]] = []
    if axis == "y":
        for edge in edges:
            got: list[list[_S]] = []
            for s in sorted(items, key=lambda s: getattr(s, edge)):
                if got and abs(getattr(got[-1][-1], edge) - getattr(s, edge)) <= _Y_TOL:
                    got[-1].append(s)
                else:
                    got.append([s])
            groups.extend(g for g in got if edge == "cy" or len(g) >= 2)
    else:
        # right-aligned or left-aligned or centred columns
        for edge in ("x1", "x0", "cx"):
            got: list[list[_S]] = []
            for s in sorted(items, key=lambda s: getattr(s, edge)):
                if got and abs(getattr(got[-1][-1], edge) - getattr(s, edge)) <= _X_TOL:
                    got[-1].append(s)
                else:
                    got.append([s])
            groups.extend(g for g in got if len(g) >= 2)
    return groups


def _best_category_group(cats: list[_S], data: list[_S] | None = None) -> tuple[str, list[_S]] | None:
    """Largest set of distinct non-numeric spans aligned as a category axis; ties go
    to the group sitting closest to the data labels (a legend sits further away)."""
    best: tuple[str, list[_S]] | None = None
    best_key: tuple[int, float] | None = None
    for axis in ("x", "y"):
        for g in _alignment_groups(cats, axis, edges=("cy", "y0", "y1")):
            g = [s for s in g if len(s.text) <= 90]
            seen: set[int] = set()
            g = [s for s in g if not (id(s) in seen or seen.add(id(s)))]
            score = len({s.text.lower() for s in g})
            if len(g) < 2 or score < 2:
                continue
            dist = 0.0
            if data:
                dist = statistics.mean(min(abs(c.cx - d.cx) + abs(c.cy - d.cy) for d in data) for c in g)
            key = (score, -dist)
            if best_key is None or key > best_key:
                best, best_key = (axis, g), key
    return best


def _merge_wrapped(cats: list[_S], all_items: list[_S]) -> list[_S]:
    """A category label wrapped onto following lines: same left edge, small gap,
    nothing else (data) between. The merged text is what the annotators quote."""
    cat_ids = {id(c) for c in cats}
    others = [s for s in all_items if id(s) not in cat_ids and not s.num]
    out: list[_S] = []
    for c in cats:
        text, x0, y0, x1, y1 = c.text, c.x0, c.y0, c.x1, c.y1
        cur = c
        while True:
            nxt = [
                o for o in others
                if abs(o.x0 - cur.x0) <= 3 and 0 <= o.y0 - cur.y1 <= 0.6 * cur.h and len(o.text) <= 90
            ]
            if not nxt:
                break
            o = min(nxt, key=lambda o: o.y0)
            if any(abs(k.x0 - cur.x0) <= 3 and cur.y1 < k.y0 < o.y0 for k in cats):
                break
            text, x1, y1 = text + " " + o.text, max(x1, o.x1), o.y1
            others = [k for k in others if k is not o]
            cur = o
        sp = Span(text=text, bbox=[x0, y0, x1, y1], size=c.span.size, bold=c.span.bold)
        out.append(_S(sp, text, False))
    return out


# --------------------------------------------------------------------------- panels


def _cluster(items: list[_S], dx: float = 14.0, dy: float = 22.0) -> list[list[_S]]:
    """Connected components of spans whose padded bboxes touch."""
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        a = items[i]
        for j in range(i + 1, n):
            b = items[j]
            if a.x0 - dx <= b.x1 and b.x0 - dx <= a.x1 and a.y0 - dy <= b.y1 and b.y0 - dy <= a.y1:
                parent[find(i)] = find(j)
    comps: dict[int, list[_S]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(items[i])
    return list(comps.values())


# --------------------------------------------------------------------------- tables


def _assign(data: list[_S], cats: list[_S], axis: str) -> dict[int, list[_S]]:
    """Map category index -> data labels, by nearest centre along the category axis."""
    if axis == "x":  # categories stacked vertically -> horizontal bars: match by y
        pos = [c.cy for c in cats]
        get = lambda s: s.cy  # noqa: E731
    else:
        pos = [c.cx for c in cats]
        get = lambda s: s.cx  # noqa: E731
    spacing = statistics.median([abs(b - a) for a, b in zip(sorted(pos), sorted(pos)[1:])]) if len(pos) > 1 else 50.0
    out: dict[int, list[_S]] = {i: [] for i in range(len(cats))}
    limit = max(0.6 * spacing, 1.2 * (data[0].h if data else 10.0))
    near = [d for d in data if min(abs(get(d) - p) for p in pos) <= limit]
    if near and len(near) <= len(cats):
        # one label per category at most: match in position order (no two labels
        # may swap categories), which fixes labels nudged towards a neighbour
        order = sorted(range(len(pos)), key=lambda i: pos[i])
        ds = sorted(near, key=get)
        j = 0
        for d in ds:
            # advance to the best category not before the previous one
            best_i, best_d = None, None
            for k in range(j, len(order)):
                dd = abs(get(d) - pos[order[k]])
                if best_d is None or dd < best_d:
                    best_i, best_d = k, dd
                elif dd > best_d:
                    break
            if best_i is not None and best_d is not None and best_d <= limit:
                out[order[best_i]].append(d)
                j = best_i + 1
        return out
    for d in data:
        i, dist = min(((i, abs(get(d) - p)) for i, p in enumerate(pos)), key=lambda t: t[1])
        if dist <= limit:
            out[i].append(d)
    return out


def _legend_candidates(others: list[_S], cats: list[_S], data: list[_S]) -> list[_S]:
    used = {id(s) for s in cats}
    cands = [s for s in others if id(s) not in used and 1 <= len(s.text) <= 60]
    # legends sit on one line (horizontal) or stacked near the top/bottom; prefer spans
    # that are not axis titles (axis titles are rotated, which the text layer gives as
    # tall narrow boxes) and not sentence-like.
    out = []
    for s in cands:
        if s.h > 3 * s.span.size:  # rotated axis title
            continue
        if "=" in s.text or (re.search(r"\d", s.text) and not _YEAR_RE.match(s.text) and not re.search(r"\d{4}", s.text)):
            continue
        if s.text.endswith((".", ":")) and len(s.text) > 25:
            continue
        out.append(s)
    return out


def _column_headers(items: list[_S], cats: list[_S], data: list[_S], axis: str) -> list[list[_S]]:
    """Rows of short non-numeric spans sitting above the data region (panel titles,
    series names printed over column groups)."""
    cat_ids = {id(c) for c in cats}
    top = min(d.y0 for d in data)
    left = min(d.x0 for d in data) - 4
    right = max(d.x1 for d in data) + 4
    width = right - left
    cands = [
        s for s in items
        if not s.num and id(s) not in cat_ids and s.y1 <= top + 2 and len(s.text) <= 60
        and s.x1 >= left and s.x0 <= right and s.h <= 3 * s.span.size
    ]
    rows = [sorted(g, key=lambda s: s.x0) for g in _alignment_groups(cands, "y") if g]
    # a header row names column groups: several spans, none spanning most of the data
    rows = [
        g for g in rows
        if len(g) >= 2 and all((h.x1 - h.x0) < 0.6 * width for h in g) and (g[-1].x1 - g[0].x0) >= 0.45 * width
    ]
    return rows[-3:]  # the three rows closest to the data


def _header_for(d: _S, header_rows: list[list[_S]]) -> list[str]:
    parts: list[str] = []
    for row in header_rows:
        hit = None
        for h in row:
            if h.x0 - 2 <= d.cx <= h.x1 + 2:
                hit = h
                break
        if hit is None and len(row) > 1:
            # nearest header whose x gap is smaller than half the header spacing
            xs = sorted(h.cx for h in row)
            spacing = statistics.median([b - a for a, b in zip(xs, xs[1:])])
            h = min(row, key=lambda h: abs(h.cx - d.cx))
            if abs(h.cx - d.cx) <= 0.5 * spacing:
                hit = h
        if hit is not None:
            parts.append(hit.text)
    return parts


def _panel_table(items: list[_S]) -> ChartTable | None:
    items = _merge_line_fragments(items)
    nums = [s for s in items if s.num]
    ticks = _tick_groups([s for s in nums if is_numeric(s.text)])
    data = [s for s in nums if id(s) not in ticks]
    cats_all = [s for s in items if not s.num]
    if not data:
        return None
    best = _best_category_group(cats_all, data)
    if best is None:
        rows = []
        for d in data:
            if not cats_all:
                break
            c = min(cats_all, key=lambda c: (c.cx - d.cx) ** 2 + (c.cy - d.cy) ** 2)
            rows.append([c.text, _cell(d.text)])
        return ChartTable(header=["", ""], rows=rows, confidence=0.3, source="textlabels") if rows else None
    axis, cats = best
    cats = sorted(cats, key=lambda c: c.cy if axis == "x" else c.cx)
    raw_cats = cats
    cats = _merge_wrapped(cats, items)
    assigned = _assign(data, cats, axis)
    if not any(assigned.values()):
        return None
    data = [d for ds in assigned.values() for d in ds]  # only labels that belong to a category
    header_rows = _column_headers(items, cats, data, axis)
    legends = _legend_candidates(cats_all, raw_cats, data)

    # columns: combined header text when header rows exist, else series order
    col_names: list[str] = []
    cells: dict[int, dict[int, str]] = {}
    for i in assigned:
        assigned[i].sort(key=lambda s: (s.cx, s.cy) if axis == "x" else (s.cx, -s.cy))
    use_headers = bool(header_rows) and any(_header_for(d, header_rows) for d in data)
    if use_headers:
        # Stacked bars: labels of one category share x, so x-range headers would
        # name them all alike. Then the header row is really the legend, and the
        # series dimension runs along the stack.
        stacked = 0
        for ds in assigned.values():
            if len(ds) >= 2:
                names = [" ".join(_header_for(d, header_rows)) for d in ds]
                stacked += len(names) - len(set(names))
        if stacked >= 0.5 * max(1, sum(len(ds) for ds in assigned.values() if len(ds) >= 2)):
            legends = header_rows[-1] + legends
            use_headers = False
    if use_headers:
        for i, ds in assigned.items():
            for d in ds:
                name = " ".join(_header_for(d, header_rows)) or ""
                if name not in col_names:
                    col_names.append(name)
                cells.setdefault(i, {})
                j = col_names.index(name)
                if j in cells[i]:  # second datum in the same column: open a new unnamed one
                    name2 = name + " "
                    if name2 not in col_names:
                        col_names.append(name2)
                    j = col_names.index(name2)
                cells[i][j] = _cell(d.text)
    else:
        nseries = max(len(v) for v in assigned.values())
        col_names = [""] * nseries
        lg_groups = [g for g in _alignment_groups(legends, "y")] + [g for g in _alignment_groups(legends, "x")]
        lg_groups = [g for g in lg_groups if len(g) == nseries and nseries >= 2]
        if lg_groups:
            g = sorted(lg_groups[0], key=lambda s: (s.cy, s.cx))
            col_names = [s.text for s in g]
        elif nseries == 1 and legends:
            dcx = statistics.mean(d.cx for d in data)
            dcy = statistics.mean(d.cy for d in data)
            top = min(d.y0 for d in data)
            # a legend sits beside or below the plot; what is above it is the title
            pool = [s for s in legends if s.cy >= top] or legends
            near = min(pool, key=lambda s: (s.cx - dcx) ** 2 + (s.cy - dcy) ** 2)
            col_names = [near.text]
        for i, ds in assigned.items():
            if axis == "y" and len(ds) >= 2 and max(d.cx for d in ds) - min(d.cx for d in ds) < 0.6 * ds[0].h:
                ds = sorted(ds, key=lambda d: -d.cy)  # vertical stack: bottom -> top
            cells[i] = {j: _cell(d.text) for j, d in enumerate(ds)}

    used = set(col_names)
    title_parts = [s.text for s in legends if s.text not in used and s.y1 <= min(d.y0 for d in data) + 2]
    title = " | ".join(title_parts[:4])
    rows: list[list[str]] = []
    for i, c in enumerate(cats):
        vals = [cells.get(i, {}).get(j, "") for j in range(len(col_names))]
        if any(vals):
            rows.append([c.text] + vals)
    if not rows:
        return None
    conf = min(1.0, len(rows) / max(1, len(cats)))
    return ChartTable(header=[title] + col_names, rows=rows, confidence=conf, source="textlabels")


def labels_table(region: ChartRegion) -> ChartTable | None:
    """Reconstruct a chart table from the text layer inside the region (no VLM)."""
    items = _wrap(region.spans)
    return _panel_table(items) if items else None


def panel_tables(region: ChartRegion) -> list[ChartTable]:
    """Split the region into panels (small multiples) and reconstruct each."""
    w = region.bbox[2] - region.bbox[0]
    items = [s for s in _wrap(region.spans) if (s.x1 - s.x0) < 0.55 * w]
    out: list[ChartTable] = []
    for comp in _cluster(items):
        if sum(s.num for s in comp) < 2:
            continue
        t = _panel_table(comp)
        if t:
            out.append(t)
    whole = _panel_table(items) if items else None
    if whole:
        out.append(whole)
    return out


# --------------------------------------------------------------------------- VLM snap


def _sim(a: str, b: str) -> float:
    return fuzz.ratio(re.sub(r"[^a-z0-9]", "", a.lower()), re.sub(r"[^a-z0-9]", "", b.lower()))


def _close(a: str, b: str) -> bool:
    try:
        fa, fb = float(clean_number(a)), float(clean_number(b))
    except ValueError:
        return False
    if fa == fb:
        return True
    if abs(fa - fb) <= 0.1 * max(abs(fa), abs(fb), 1e-9):
        return True
    da, db = re.sub(r"\D", "", a), re.sub(r"\D", "", b)
    return bool(da) and sorted(da) == sorted(db)


def snap_table(region: ChartRegion) -> ChartTable | None:
    """Correct the VLM's chart table with text-layer data labels."""
    if not region.vlm_table_html:
        return None
    vlm = parse_html_table(region.vlm_table_html)
    if vlm is None or not vlm.rows:
        return None
    ref = labels_table(region)
    if ref is None:
        vlm.source = "vlm"
        return vlm
    ncols = max(len(r) for r in vlm.rows)
    used_ref: set[int] = set()
    rows: list[list[str]] = []
    for row in vlm.rows:
        row = list(row) + [""] * (ncols - len(row))
        best_i, best_s = -1, 0.0
        for i, rr in enumerate(ref.rows):
            s = _sim(row[0], rr[0])
            if s > best_s:
                best_i, best_s = i, s
        if best_s >= 85:
            used_ref.add(best_i)
            rr = ref.rows[best_i]
            # column alignment: by header name when possible, else positional
            for ci in range(1, ncols):
                rj = ci
                if ci < len(vlm.header) and ref.header:
                    named = [j for j, h in enumerate(ref.header) if j > 0 and h and _sim(h, vlm.header[ci]) >= 85]
                    if named:
                        rj = named[0]
                if rj < len(rr) and rr[rj]:
                    if not row[ci] or not is_numeric(row[ci]) or _close(row[ci], rr[rj]):
                        row[ci] = rr[rj]
        rows.append(row)
    for i, rr in enumerate(ref.rows):
        if i not in used_ref and rr[0] and all(_sim(rr[0], r[0]) < 85 for r in rows):
            rows.append(list(rr) + [""] * max(0, ncols - len(rr)))
    header = list(vlm.header) + [""] * max(0, ncols - len(vlm.header))
    if ref.header and ref.header[0] and not header[0]:
        header[0] = ref.header[0]
    return replace(vlm, header=header, rows=rows, confidence=ref.confidence, source="snap")


# --------------------------------------------------------------------------- page entry


def reconstruct_page(page: pymupdf.Page, tl: TextLayerPage) -> list[ChartTable]:
    region = region_from_page(page, tl, [0.0, 0.0, page.rect.width, page.rect.height])
    return panel_tables(region)
