from docparse.charts import ChartRegion
from docparse.charts.textlabels import labels_table, snap_table
from docparse.textlayer import Span


def _span(text: str, x0: float, y0: float, w: float = 20.0, h: float = 8.0) -> Span:
    return Span(text=text, bbox=[x0, y0, x0 + w, y0 + h], size=8.0)


def _region(spans: list[Span], vlm: str | None = None) -> ChartRegion:
    return ChartRegion(bbox=[0, 0, 400, 300], spans=spans, drawings=[], page_width=400, page_height=300, vlm_table_html=vlm)


def test_horizontal_bars_with_axis_ticks_excluded():
    # categories stacked at the left, one data label per bar, x-axis ticks at the bottom
    spans = [
        _span("Alpha", 10, 50), _span("Beta", 10, 80), _span("Gamma", 10, 110),
        _span("12.5", 120, 50, w=16), _span("7.25", 90, 80, w=16), _span("30", 200, 110, w=12),
        _span("0", 40, 150, w=6), _span("10", 100, 150, w=10), _span("20", 160, 150, w=10), _span("30", 220, 150, w=10),
        _span("Score", 150, 170, w=30),
    ]
    t = labels_table(_region(spans))
    assert t is not None
    assert [r[0] for r in t.rows] == ["Alpha", "Beta", "Gamma"]
    assert [r[1] for r in t.rows] == ["12.5", "7.25", "30"]
    assert t.header[1] == "Score"  # legend nearest the plot names the single series


def test_grouped_vertical_bars_use_legend_order():
    spans = [
        _span("2022", 20, 20, w=16), _span("2024", 44, 20, w=16),  # legend row (top-left)
        _span("Q1", 40, 200, w=12), _span("Q2", 140, 200, w=12),
        _span("5", 30, 150, w=6), _span("8", 50, 130, w=6),
        _span("6", 130, 140, w=6), _span("9", 150, 120, w=6),
    ]
    t = labels_table(_region(spans))
    assert t is not None
    assert t.header[1:] == ["2022", "2024"]
    assert t.rows == [["Q1", "5", "8"], ["Q2", "6", "9"]]


def test_snap_replaces_misread_values_and_keeps_structure():
    spans = [
        _span("Alpha", 10, 50), _span("Beta", 10, 80),
        _span("12.5", 120, 50, w=16), _span("7.25", 90, 80, w=16),
    ]
    vlm = "<table><tr><th>Item</th><th>Value</th></tr><tr><td>Alpha</td><td>12.6</td></tr><tr><td>Beta</td><td>72.5</td></tr></table>"
    t = snap_table(_region(spans, vlm))
    assert t is not None and t.source == "snap"
    assert t.header == ["Item", "Value"]
    assert t.rows == [["Alpha", "12.5"], ["Beta", "7.25"]]


def test_snap_keeps_chart_caption_when_correcting_values():
    spans = [
        _span("Alpha", 10, 50), _span("Beta", 10, 80),
        _span("12.5", 120, 50, w=16), _span("7.25", 90, 80, w=16),
    ]
    vlm = (
        "<table><caption>Revenue by region &amp; year</caption>"
        "<tr><th>Item</th><th>Value</th></tr>"
        "<tr><td>Alpha</td><td>12.6</td></tr>"
        "<tr><td>Beta</td><td>72.5</td></tr></table>"
    )
    t = snap_table(_region(spans, vlm))
    assert t is not None and t.source == "snap"
    assert t.caption == "Revenue by region & year"
    assert t.header == ["Item", "Value"]
    assert t.rows == [["Alpha", "12.5"], ["Beta", "7.25"]]
    assert "<caption>Revenue by region &amp; year</caption>" in t.to_html()
