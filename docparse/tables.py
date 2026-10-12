"""Header canonicalization for VLM table HTML (TableRecordMatch-aware).

ParseBench keys table records by the ``<th>`` chain above each cell (header-row
texts joined top-to-bottom, colspan text repeated per column). VLMs usually emit
exactly one ``<th>`` row, so multi-row headers lose their second level and
"title + header" tables lose the title. Conservative fixes, applied only when the
table's own content supports them. **Measured on the full table set (503 docs) each
rule is net negative** (GTRM 0.8346 baseline; a: 0.8320, b: 0.8321, c: 0.8342 —
GT header conventions are inconsistent and the VLM's single header row is the
better average bet), so the provider enables none by default; set
``DOCPARSE_TABLE_RULES=abc`` (any subset) to experiment.

(a) extend the header block downward while some column still has an empty header
    cell and the next row is not numeric-majority (TATR canonicalization);
(b) a leading row with exactly one non-empty cell becomes a ``<th colspan=N>``
    title row;
(c) when the VLM's header row has the same "shape" as the body's first column
    (it promoted a data row) and the block right above the table is a short
    title/caption, insert that title as a ``<th colspan=N>`` row and demote the
    VLM header to body.
"""

from __future__ import annotations

import html as _html
import re
from typing import Any

_NUM_RE = re.compile(r"^[\s$€£¥(]*[-−+]?\d[\d\s,.%)]*$|^[\s-]*$")
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<(t[hd])([^>]*)>(.*?)</t[hd]>", re.S | re.I)
_TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.S | re.I)
_ATTR_INT_RE = re.compile(r'(colspan|rowspan)\s*=\s*"?(\d+)"?', re.I)


class Cell:
    __slots__ = ("tag", "attrs", "inner", "colspan", "rowspan")

    def __init__(self, tag: str, attrs: str, inner: str):
        self.tag = tag.lower()
        self.attrs = attrs
        self.inner = inner
        spans = {k.lower(): int(v) for k, v in _ATTR_INT_RE.findall(attrs)}
        self.colspan = max(1, spans.get("colspan", 1))
        self.rowspan = max(1, spans.get("rowspan", 1))

    @property
    def text(self) -> str:
        return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", self.inner))).strip()

    def render(self) -> str:
        attrs = re.sub(r'\s*colspan\s*=\s*"?\d+"?', "", self.attrs, flags=re.I)
        if self.colspan > 1:
            attrs += f' colspan="{self.colspan}"'
        return f"<{self.tag}{attrs}>{self.inner}</{self.tag}>"


def parse_rows(table_html: str) -> list[list[Cell]]:
    rows: list[list[Cell]] = []
    for tr in _TR_RE.findall(table_html):
        cells = [Cell(t, a, i) for t, a, i in _CELL_RE.findall(tr)]
        if cells:
            rows.append(cells)
    return rows


def _width(row: list[Cell]) -> int:
    return sum(c.colspan for c in row)


def _is_num(text: str) -> bool:
    t = text.strip()
    return bool(t) and bool(_NUM_RE.match(t))


def _numeric_majority(row: list[Cell]) -> bool:
    vals = [c.text for c in row if c.text]
    if not vals:
        return False
    return sum(_is_num(v) for v in vals) >= 0.5 * len(vals)


def _shape(text: str) -> str:
    """Coarse pattern of a cell: digits→9, letters→a, keeps punctuation; first 8 chars."""
    s = re.sub(r"\d", "9", text.strip().lower())
    s = re.sub(r"[a-z]", "a", s)
    return s[:8]


def render_rows(rows: list[list[Cell]], *, thead_rows: int = 0) -> str:
    parts = ["<table>"]
    if thead_rows:
        parts.append("<thead>")
        for r in rows[:thead_rows]:
            parts.append("<tr>" + "".join(c.render() for c in r) + "</tr>")
        parts.append("</thead><tbody>")
    else:
        parts.append("<tbody>")
    for r in rows[thead_rows:]:
        parts.append("<tr>" + "".join(c.render() for c in r) + "</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _header_rows(rows: list[list[Cell]]) -> int:
    """Leading rows whose cells are all <th>."""
    n = 0
    for r in rows:
        if all(c.tag == "th" for c in r):
            n += 1
        else:
            break
    return n


def extend_header(rows: list[list[Cell]]) -> tuple[list[list[Cell]], bool]:
    """(a): while a header column is empty and the next row is non-numeric and
    not the last row, promote that row to <th>."""
    changed = False
    n_head = _header_rows(rows)
    if n_head == 0 or n_head >= len(rows) - 1:
        return rows, False
    width = _width(rows[0])
    for _ in range(2):  # at most two extra header levels
        # column coverage from the current header block
        covered = [False] * width
        for r in rows[:n_head]:
            col = 0
            for c in r:
                if c.text:
                    for k in range(col, min(width, col + c.colspan)):
                        covered[k] = True
                col += c.colspan
        if all(covered) or n_head >= len(rows) - 1:
            break
        nxt = rows[n_head]
        if _width(nxt) != width:
            break
        # A numeric row still continues the header when its label column is empty
        # (sub-headers like "$2,000 / $3,000" under "Basic Limit").
        if _numeric_majority(nxt) and nxt[0].text:
            break
        if not any(c.text for c in nxt):
            break
        for c in nxt:
            c.tag = "th"
        n_head += 1
        changed = True
    return rows, changed


def title_row(rows: list[list[Cell]]) -> tuple[list[list[Cell]], bool]:
    """(b): a leading row with exactly one non-empty cell spanning the table is a title."""
    if len(rows) < 2:
        return rows, False
    first = rows[0]
    non_empty = [c for c in first if c.text]
    width = _width(rows[1]) if len(rows) > 1 else _width(first)
    if len(non_empty) == 1 and len(first) > 1 and width > 1 and not _is_num(non_empty[0].text):
        t = non_empty[0]
        t.tag = "th"
        t.colspan = width
        rows[0] = [t]
        return rows, True
    return rows, False


def _looks_like_data(header: list[Cell], body: list[list[Cell]]) -> bool:
    """The VLM promoted a data row: first header cell shares its shape with the
    body's first column and the header cells are mostly numeric-or-same-shape."""
    if not header or not body or len(body) < 2:
        return False
    h0 = header[0].text
    if not h0:
        return False
    first_col = [r[0].text for r in body if r and r[0].text]
    if len(first_col) < 2:
        return False
    same = sum(_shape(v) == _shape(h0) for v in first_col)
    return same >= 0.6 * len(first_col) and len(_shape(h0)) >= 4


def insert_title_from_context(
    rows: list[list[Cell]], title: str | None
) -> tuple[list[list[Cell]], bool]:
    """(c): VLM header looks like data and a short title sits right above the table."""
    if not title or len(rows) < 3:
        return rows, False
    n_head = _header_rows(rows)
    if n_head != 1:
        return rows, False
    if not _looks_like_data(rows[0], rows[1:]):
        return rows, False
    width = _width(rows[0])
    for c in rows[0]:
        c.tag = "td"
    t = Cell("th", "", _html.escape(title.strip()))
    t.colspan = width
    return [[t]] + rows, True


def context_title(context_blocks: list[dict[str, Any]], table_index: int) -> str | None:
    """Short title/caption immediately above the table in the block list."""
    for k in range(table_index - 1, max(-1, table_index - 3), -1):
        b = context_blocks[k]
        if b.get("label") in ("image", "chart", "table"):
            return None
        text = re.sub(r"^\s*#{1,6}\s*", "", b.get("text", "")).strip().strip("*")
        if not text:
            continue
        short = len(text) <= 70 and "\n" not in text
        titled = b.get("label") in ("title", "section_header", "caption") or text.endswith(":")
        return text if short and titled else None
    return None


def canonicalize_table_html(
    table_html: str,
    context_blocks: list[dict[str, Any]] | None = None,
    table_index: int | None = None,
    rules: tuple[str, ...] = ("a", "b", "c"),
) -> str:
    """Apply the enabled rules to the first <table> in ``table_html``; untouched on any oddity."""
    m = _TABLE_RE.search(table_html)
    if not m:
        return table_html
    rows = parse_rows(m.group(0))
    if len(rows) < 2:
        return table_html
    widths = {_width(r) for r in rows}
    if len(widths) > 2:  # ragged tables: leave alone
        return table_html
    changed = False
    if "b" in rules:
        rows, ch = title_row(rows)
        changed |= ch
    if "a" in rules:
        rows, ch = extend_header(rows)
        changed |= ch
    if "c" in rules and context_blocks is not None and table_index is not None:
        rows, ch = insert_title_from_context(rows, context_title(context_blocks, table_index))
        changed |= ch
    if not changed:
        return table_html
    new_html = render_rows(rows, thead_rows=_header_rows(rows))
    return table_html[: m.start()] + new_html + table_html[m.end():]


def canonicalize_blocks(blocks: list[dict[str, Any]], rules: tuple[str, ...] = ("a", "b", "c")) -> list[dict[str, Any]]:
    out = []
    for i, b in enumerate(blocks):
        if b.get("label") == "table" and "<table" in b.get("text", ""):
            nb = dict(b)
            nb["text"] = canonicalize_table_html(b["text"], blocks, i, rules)
            out.append(nb)
        else:
            out.append(b)
    return out
