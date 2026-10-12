"""How much of ParseBench is born-digital? Per category: share of pages with a usable
text layer, and what styling the layer carries (bold spans, distinct font sizes,
underline/strike drawings, highlight annotations). CPU only.

Usage: uv run python scripts/analyze_text_layer.py [data/docs]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pymupdf

MIN_CHARS_PER_PAGE = 80  # below this the "text layer" is noise (stamps, OCR fragments)


def page_stats(page: pymupdf.Page) -> dict:
    d = page.get_text("dict")
    chars = bold = spans = 0
    sizes: set[float] = set()
    for block in d["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                t = span["text"].strip()
                if not t:
                    continue
                spans += 1
                chars += len(t)
                sizes.add(round(span["size"], 1))
                if span["flags"] & 16 or "bold" in span["font"].lower():
                    bold += 1
    # Thin horizontal rects/lines are how underline & strikethrough are drawn.
    thin = 0
    for dr in page.get_drawings():
        r = dr.get("rect")
        if r is not None and r.height < 2.5 and r.width > 8:
            thin += 1
    annots = sum(1 for a in page.annots() if a.type[0] in (8, 9, 10, 11))  # highlight/underline/squiggly/strike
    return {
        "has_text": chars >= MIN_CHARS_PER_PAGE,
        "chars": chars,
        "bold_frac": bold / spans if spans else 0.0,
        "n_sizes": len(sizes),
        "thin_lines": thin,
        "markup_annots": annots,
    }


def main(root: Path) -> None:
    per_cat: dict[str, list[dict]] = defaultdict(list)
    for pdf in sorted(root.rglob("*.pdf")):
        cat = pdf.parent.name
        try:
            doc = pymupdf.open(pdf)
            for page in doc:
                per_cat[cat].append(page_stats(page))
            doc.close()
        except Exception as e:  # noqa: BLE001
            print(f"skip {pdf.name}: {e}", file=sys.stderr)

    print(f"| category | pages | text layer | bold spans | ≥3 font sizes | thin lines/pg | markup annots/pg |")
    print("|---|---|---|---|---|---|---|")
    total: list[dict] = []
    for cat, rows in sorted(per_cat.items()):
        total += rows
        _print_row(cat, rows)
    _print_row("**all**", total)


def _print_row(name: str, rows: list[dict]) -> None:
    n = len(rows)
    digital = [r for r in rows if r["has_text"]]
    nd = len(digital) or 1
    print(
        f"| {name} | {n} | {100 * len(digital) / n:.0f}% "
        f"| {100 * sum(r['bold_frac'] > 0 for r in digital) / nd:.0f}% "
        f"| {100 * sum(r['n_sizes'] >= 3 for r in digital) / nd:.0f}% "
        f"| {sum(r['thin_lines'] for r in digital) / nd:.1f} "
        f"| {sum(r['markup_annots'] for r in digital) / nd:.2f} |"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/docs"))
