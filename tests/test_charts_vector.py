"""Synthetic chart regions for the vector reconstruction."""

import pymupdf
import pytest

from docparse.charts import ChartRegion
from docparse.charts.vector import Bar, Poly, page_primitives_full, reconstruct_region
from docparse.textlayer import Span


def _span(text: str, x0: float, y0: float, x1: float, y1: float) -> Span:
    return Span(text=text, bbox=[x0, y0, x1, y1], size=8.0)


def _region(spans: list[Span]) -> ChartRegion:
    return ChartRegion(bbox=[0, 0, 400, 300], spans=spans, drawings=[], page_width=600, page_height=800)


def test_vertical_grouped_bars_two_series() -> None:
    # y axis: 0 at y=200, 100 at y=100 (value = -1*y + 200)
    spans = [_span(str(v), 10, 200 - v - 4, 25, 200 - v + 4) for v in (0, 25, 50, 75, 100)]
    spans += [_span("A", 55, 205, 65, 213), _span("B", 115, 205, 125, 213)]
    # legend: swatch + name
    spans += [_span("Red", 212, 251, 230, 259), _span("Blue", 262, 251, 282, 259)]
    red, blue = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    bars = [
        Bar(50, 200 - 40, 58, 200, red), Bar(60, 200 - 60, 68, 200, blue),   # A: 40 / 60
        Bar(110, 200 - 80, 118, 200, red), Bar(120, 200 - 20, 128, 200, blue),  # B: 80 / 20
        Bar(200, 250, 210, 260, red), Bar(250, 250, 260, 260, blue),  # swatches
    ]
    (table,) = reconstruct_region(_region(spans), bars, [])
    assert table.header == ["", "Red", "Blue"]
    assert table.rows == [["A", "40", "60"], ["B", "80", "20"]]


def test_horizontal_bars_with_negative_value() -> None:
    # x axis: -50 at x=100, 0 at x=150, 50 at x=200 (value = x - 150)
    spans = [_span(str(v), 150 + v - 6, 215, 150 + v + 6, 223) for v in (-50, -25, 0, 25, 50)]
    spans += [_span("North", 60, 96, 95, 104), _span("South", 60, 136, 95, 144)]
    spans += [_span("Sales", 5, 60, 60, 70)]  # chart title names the single series
    green = (0.0, 0.6, 0.0)
    bars = [Bar(150, 95, 190, 105, green), Bar(130, 135, 150, 145, green)]  # +40, -20
    (table,) = reconstruct_region(_region(spans), bars, [])
    assert table.header[1] == "Sales" and table.header[0] in ("", "Sales")  # the title may also name column 0
    assert table.rows == [["North", "40"], ["South", "-20"]]


def test_printed_label_beats_geometry() -> None:
    spans = [_span(str(v), 10, 200 - v - 4, 25, 200 - v + 4) for v in (0, 50, 100)]
    spans += [_span("Q1", 52, 205, 62, 213), _span("41.7", 50, 150, 66, 158)]  # label above the bar
    bars = [Bar(50, 200 - 43, 66, 200, (0.2, 0.2, 0.8))]
    (table,) = reconstruct_region(_region(spans), bars, [])
    assert table.rows == [["Q1", "41.7"]]


def test_line_series_sampled_at_categories() -> None:
    spans = [_span(str(v), 10, 200 - v - 4, 25, 200 - v + 4) for v in (0, 50, 100)]
    spans += [_span("2020", 45, 205, 65, 213), _span("2021", 95, 205, 115, 213), _span("2022", 145, 205, 165, 213)]
    spans += [_span("Growth", 170, 116, 200, 124)]  # direct label at the line end
    line = Poly(points=[(55, 200 - 10), (105, 200 - 50), (155, 200 - 80)], color=(0.9, 0.1, 0.1), bbox=[55, 120, 155, 190])
    tables = reconstruct_region(_region(spans), [], [line])
    assert tables and tables[-1].header == ["", "Growth"]
    assert [r[1] for r in tables[-1].rows] == ["10", "50", "80"]


@pytest.mark.parametrize("reverse", [False, True])
def test_stacked_areas_use_band_thickness_with_offset_axis(reverse: bool) -> None:
    spans = [_span(str(v + 100), 10, 200 - v - 4, 25, 200 - v + 4) for v in (0, 50, 100)]
    spans += [_span(str(2020 + i), x - 10, 205, x + 10, 213) for i, x in enumerate((55, 105, 155))]
    spans += [_span("Unmatched legend", 90, 235, 120, 243)]
    red, blue = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    points = [
        [(55, 180), (105, 150), (155, 160), (155, 200), (55, 200)],
        [(55, 150), (105, 130), (155, 150), (155, 160), (105, 150), (55, 180)],
    ]
    areas = [Poly(p[::-1] if reverse else p, c, [55, min(y for _, y in p), 155, max(y for _, y in p)], filled=True)
             for p, c in zip(points, (red, blue))]
    (table,) = reconstruct_region(_region(spans), [], areas, {red: "Red", blue: "Blue"})
    assert table.source == "vector-area"
    assert table.header == ["", "Red", "Blue"]
    assert table.rows == [["2020", "20", "30"], ["2021", "50", "20"], ["2022", "40", "10"]]


def test_area_requires_legend_and_value_axis() -> None:
    red = (1.0, 0.0, 0.0)
    area = Poly([(50, 170), (150, 140), (150, 200), (50, 200)], red, [50, 140, 150, 200], filled=True)
    cats = [_span("A", 45, 205, 55, 213), _span("B", 145, 205, 155, 213)]
    ticks = [_span(str(v), 10, 200 - v - 4, 25, 200 - v + 4) for v in (0, 50, 100)]
    assert reconstruct_region(_region(cats), [], [area], {red: "Red"}) == []
    assert reconstruct_region(_region(cats + ticks), [], [area]) == []


def test_area_interpolates_named_categories_without_extrapolating() -> None:
    red = (1.0, 0.0, 0.0)
    area = Poly([(50, 170), (150, 140), (150, 200), (50, 200)], red, [50, 140, 150, 200], filled=True)
    spans = [_span(str(v), 10, 200 - v - 4, 25, 200 - v + 4) for v in (0, 50, 100)]
    spans += [_span(name, x - 5, 205, x + 5, 213) for x, name in ((50, "A"), (100, "B"), (150, "C"), (200, "D"))]
    (table,) = reconstruct_region(_region(spans), [], [area], {red: "Red"})
    assert table.rows == [["A", "30"], ["B", "45"], ["C", "60"]]


def test_filled_subpaths_stay_separate_from_stroked_lines() -> None:
    with pymupdf.open() as doc:
        page = doc.new_page()
        shape = page.new_shape()
        for dx in (0, 200):
            shape.draw_polyline([(50 + dx, 180), (150 + dx, 150), (150 + dx, 200), (50 + dx, 200), (50 + dx, 180)])
        shape.finish(fill=(1, 0, 0), color=None)
        shape.commit()
        _, polys, _ = page_primitives_full(page)
        assert len(polys) == 2
        assert all(p.filled for p in polys)
        assert [p.bbox for p in polys] == [[50, 150, 150, 200], [250, 150, 350, 200]]
