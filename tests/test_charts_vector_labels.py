"""Label association in docparse.charts.vector: sign legends, ordinal legends,
segment chaining, pie slices."""

from __future__ import annotations

from docparse.charts import ChartRegion
from docparse.charts.vector import (
    Bar,
    Poly,
    Slice,
    _chain,
    _infer_zero,
    _merge_wrapped_labels,
    pie_tables,
    reconstruct_region,
)
from docparse.textlayer import Span


def _span(text: str, x0: float, y0: float, x1: float, y1: float, bold: bool = False) -> Span:
    return Span(text=text, bbox=[x0, y0, x1, y1], size=8.0, bold=bold)


def _region(spans: list[Span]) -> ChartRegion:
    return ChartRegion(bbox=[0, 0, 400, 300], spans=spans, drawings=[], page_width=600, page_height=800)


def test_infer_zero_from_shared_baseline():
    bars = [Bar(10, 50, 20, 100, (0.1, 0.2, 0.8)), Bar(30, 70, 40, 100, (0.1, 0.2, 0.8)), Bar(50, 100, 60, 130, (0.1, 0.2, 0.8))]
    assert _infer_zero(bars, horizontal=False) == 100.0


def test_sign_legend_names_series_by_side_of_baseline():
    # diverging vertical bars: positives end on y=100, negatives start there; no swatches
    blue = (0.2, 0.3, 0.9)
    bars = [Bar(100, 60, 110, 100, blue), Bar(130, 80, 140, 100, blue), Bar(160, 100, 170, 125, blue)]
    spans = [
        _span("40", 101, 50, 109, 58), _span("20", 131, 70, 139, 78), _span("-25", 160, 126, 170, 134),
        _span("A", 101, 140, 109, 148), _span("B", 131, 140, 139, 148), _span("C", 161, 140, 169, 148),
        _span("Favorable", 40, 80, 80, 90, bold=True), _span("attitude", 44, 90, 76, 100, bold=True),
        _span("Unfavorable", 36, 106, 84, 116, bold=True), _span("attitude", 44, 116, 76, 126, bold=True),
    ]
    tables = reconstruct_region(_region(spans), bars, [])
    assert tables, "no table"
    t = tables[0]
    assert t.header[1:] == ["Favorable attitude", "Unfavorable attitude"]
    rows = {r[0]: r[1:] for r in t.rows}
    assert rows["A"] == ["40", ""] and rows["C"] == ["", "-25"]


def test_ordinal_legend_maps_stacked_labels_to_colours():
    # horizontal stacked bars, two colours, legend printed as text at the right
    c1, c2 = (0.5, 0.8, 1.0), (0.0, 0.6, 0.8)
    bars = [
        Bar(100, 20, 130, 30, c1), Bar(130, 20, 150, 30, c2),
        Bar(100, 40, 120, 50, c1), Bar(120, 40, 160, 50, c2),
    ]
    spans = [
        _span("0", 96, 60, 104, 68), _span("50", 146, 60, 154, 68), _span("100", 196, 60, 204, 68),
        _span("Alpha", 60, 20, 95, 30), _span("Beta", 60, 40, 95, 50),
        _span("proved", 230, 20, 260, 30, bold=True), _span("reserves", 230, 31, 265, 41, bold=True),
        _span("unproved", 230, 60, 268, 70, bold=True), _span("reserves", 230, 71, 265, 81, bold=True),
    ]
    tables = reconstruct_region(_region(spans), bars, [])
    assert tables
    assert tables[0].header[1:] == ["proved reserves", "unproved reserves"]
    assert tables[0].rows[0][0] == "Alpha"


def test_chain_joins_single_segment_strokes():
    red = (0.9, 0.1, 0.1)
    segs = [Poly([(0, 0), (10, 5)], red, [0, 0, 10, 5]), Poly([(10, 5), (20, 2)], red, [10, 2, 20, 5]), Poly([(20, 2), (30, 8)], red, [20, 2, 30, 8])]
    out = _chain(segs)
    assert len(out) == 1 and len(out[0].points) == 4 and out[0].bbox == [0, 0, 30, 8]


def test_merge_wrapped_labels_per_column():
    spans = [_span("proved reserve", 608, 230, 686, 241), _span("Expand Energy", 114, 237, 262, 250), _span("acquisition", 608, 242, 663, 254)]
    labels = {t for _b, t in _merge_wrapped_labels(spans)}
    assert "proved reserve acquisition" in labels and "Expand Energy" in labels


def test_pie_tables_use_printed_values_and_labels_on_bisectors():
    # three slices around (100, 100), r=40: 50% right-up, 30%, 20%
    slices = [Slice(100, 100, 0, 180, 180, 40, (1, 0, 0)), Slice(100, 100, 180, 288, 108, 40, (0, 1, 0)), Slice(100, 100, 288, 360, 72, 40, (0, 0, 1))]
    spans = [
        _span("Cars 50%", 92, 50, 130, 58),      # bisector at 90° (straight up)
        _span("Bikes 30%", 50, 125, 90, 133),    # bisector at 234° (down-left)
        _span("Buses", 130, 118, 160, 126), _span("20%", 134, 128, 150, 136),  # bisector 324° (down-right)
        _span("Modal split", 70, 30, 130, 40),
    ]
    tables = pie_tables(slices, spans, {}, None)
    assert len(tables) == 1
    rows = {r[0]: r[1] for r in tables[0].rows}
    assert rows == {"Cars": "50", "Bikes": "30", "Buses": "20"}
    assert tables[0].caption == "Modal split"
