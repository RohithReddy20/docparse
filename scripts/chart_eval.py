"""Score a chart reconstruction against ParseBench's chart_data_point rules, locally.

A rule passes when its value appears in some table cell (parse-bench's own
``numbers_match``, 1% tolerance) and every rule label fuzzy-matches that cell's
row header (first cell of its row) or column header. This is a faithful local proxy
for ChartDataPointMatch so reconstruction code can iterate on all 568 chart pages
with no GPU and no VLM.

Usage:
  uv run python scripts/chart_eval.py docparse.charts.vector:reconstruct_page [--limit 50] [--verbose]

The target is ``module:function`` with signature
``fn(page: pymupdf.Page, tl: TextLayerPage) -> list[ChartTable]`` (whole-page
reconstruction; the function decides where the charts are).
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pymupdf
from parse_bench.evaluation.metrics.parse.rules_chart import numbers_match
from rapidfuzz import fuzz

from docparse import textlayer
from docparse.charts import ChartTable

LABEL_SIM = 85


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _label_ok(label: str, header: str) -> bool:
    a, b = _norm(label), _norm(header)
    if not a or not b:
        return False
    return a == b or a in b or b in a or fuzz.ratio(a, b) >= LABEL_SIM


_NUM_TOKEN = re.compile(r"[-−]?\d[\d,.]*%?")


def _single_token_match(value: str, cell: str, tol: float) -> bool:
    toks = _NUM_TOKEN.findall(cell)
    return len(toks) == 1 and numbers_match(value, toks[0], tol)


def rule_passes(rule: dict, tables: list[ChartTable]) -> bool:
    value = str(rule["value"])
    labels = [str(x) for x in rule.get("labels", [])]
    tol = float(rule.get("relative_tolerance", 0.01))
    for t in tables:
        for row in t.rows:
            if not row:
                continue
            for ci, cell in enumerate(row):
                if ci == 0 or not cell:
                    continue
                # parse-bench also accepts the printed composite form ("25 (13.0%)")
                # and a cell whose single numeric token matches the value
                if not (numbers_match(value, cell, tol) or _norm(cell) == _norm(value) or _single_token_match(value, cell, tol)):
                    continue
                col = t.header[ci] if ci < len(t.header) else ""
                row_label = row[0]
                if all(_label_ok(lb, row_label) or _label_ok(lb, col) or any(_label_ok(lb, h) for h in t.header[:1]) for lb in labels):
                    return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="module:function")
    ap.add_argument("--data", default="data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="substring filter on pdf name")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    mod, fn_name = args.target.split(":")
    fn = getattr(importlib.import_module(mod), fn_name)

    rules: dict[str, list[dict]] = defaultdict(list)
    for line in open(Path(args.data) / "chart.jsonl"):
        gt = json.loads(line)
        rules[gt["pdf"]].append(json.loads(gt["rule"]) if isinstance(gt["rule"], str) else gt["rule"])

    pdfs = sorted(p for p in rules if args.only in p)
    if args.limit:
        pdfs = pdfs[: args.limit]
    passed = total = docs_ok = 0
    for pdf in pdfs:
        path = Path(args.data) / pdf
        try:
            doc = pymupdf.open(path)
            tl_pages = textlayer.extract(str(path))
            tables: list[ChartTable] = []
            for i, page in enumerate(doc):
                tables.extend(fn(page, tl_pages[i]))
            doc.close()
        except Exception as e:  # noqa: BLE001
            print(f"ERR {pdf}: {e}", file=sys.stderr)
            tables = []
        p = sum(rule_passes(r, tables) for r in rules[pdf])
        n = len(rules[pdf])
        passed += p
        total += n
        docs_ok += p == n
        if args.verbose:
            print(f"{p:3d}/{n:<3d} {pdf}")
    print(f"docs={len(pdfs)} rules={total} pass={passed} rate={100 * passed / max(1, total):.1f}% docs_perfect={docs_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
