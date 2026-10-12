import json
from pathlib import Path

from docparse.charts.vlm_fallback import fallback_tables, load_chart_vlm, parse_model_output

HTML = "<table><tr><th>Year</th><th>OECD</th></tr><tr><td>2015</td><td>7200</td></tr><tr><td>2016</td><td>7500</td></tr></table>"
PIPE = "| Region | 2024 | 2022 |\n|---|---|---|\n| Global | 16 | 12 |\n| China | 29 | 25 |"
TSV = "Month\tInflation\tRate\n2018:12\t2.0\t2.5\n2019:01\t1.8\t2.4"


def test_parse_html():
    (t,) = parse_model_output(HTML)
    assert t.header == ["Year", "OECD"] and t.rows == [["2015", "7200"], ["2016", "7500"]] and t.source == "chart-vlm"


def test_parse_pipe():
    (t,) = parse_model_output(PIPE)
    assert t.header == ["Region", "2024", "2022"] and t.rows[1] == ["China", "29", "25"]


def test_parse_tsv():
    (t,) = parse_model_output(TSV)
    assert t.header == ["Month", "Inflation", "Rate"] and t.rows[0] == ["2018:12", "2.0", "2.5"]


def test_empty_and_prose():
    assert parse_model_output("") == []
    assert parse_model_output("This chart shows revenue growing over time.") == []


def test_load_and_match(tmp_path: Path, monkeypatch):
    root = tmp_path / "chart_vlm"
    (root / "chart").mkdir(parents=True)
    rec = {"pages": [{"charts": [{"bbox": [100, 100, 500, 400], "text": PIPE, "ms": 10}]}]}
    (root / "chart" / "doc_p1.json").write_text(json.dumps(rec))
    monkeypatch.setenv("DOCPARSE_CHART_VLM_DIR", str(root))
    dump = load_chart_vlm(str(tmp_path / "data" / "docs" / "chart" / "doc_p1.pdf"))
    assert dump is not None
    page = dump["pages"][0]
    assert len(fallback_tables(page, [110, 95, 505, 410])) == 1  # IoU well above 0.5
    assert fallback_tables(page, [600, 600, 900, 900]) == []  # no overlap
    assert fallback_tables(None, [100, 100, 500, 400]) == []
    assert load_chart_vlm(str(tmp_path / "data" / "docs" / "chart" / "missing.pdf")) is None
