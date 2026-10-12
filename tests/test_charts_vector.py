"""Synthetic chart regions for the vector reconstruction."""

from docparse.charts import ChartRegion
from docparse.charts.vector import Bar, Poly, reconstruct_region
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
    assert table.header == ["", "Sales"]
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
