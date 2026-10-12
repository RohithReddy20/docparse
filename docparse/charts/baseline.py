"""Sanity baseline for scripts/chart_eval.py: no reconstruction at all."""

from __future__ import annotations

import pymupdf

from docparse.charts import ChartTable
from docparse.textlayer import TextLayerPage


def reconstruct_page(page: pymupdf.Page, tl: TextLayerPage) -> list[ChartTable]:  # noqa: ARG001
    return []
