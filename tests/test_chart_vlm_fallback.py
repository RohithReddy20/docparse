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


def test_model_titles_and_rowspan_category_labels_survive_conversion():
    source = '''# Share by region
<table><tr><th>Region</th><th>Country</th><th>Share</th></tr>
<tr><td rowspan="2">North</td><td>Aland</td><td>17</td></tr>
<tr><td>Boland</td><td>23</td></tr></table>'''
    (table,) = parse_model_output(source)
    assert table.caption == "Share by region"
    assert table.rows == [["North", "Aland", "17"], ["North", "Boland", "23"]]
    assert table.row_header_columns == 2
    assert "<th>North</th><th>Boland</th><td>23</td>" in table.to_html()


def test_explicit_captions_are_preserved_and_numeric_columns_stay_values():
    source = '# Ignored title\n' + HTML.replace("<table>", "<table><caption>Actual title</caption>")
    (table,) = parse_model_output(source)
    assert table.caption == "Actual title"
    assert table.row_header_columns == 1


def test_separate_tables_keep_their_own_model_titles():
    tables = parse_model_output("# First chart\n" + HTML + "\n# Second chart\n" + HTML)
    assert [t.caption for t in tables] == ["First chart", "Second chart"]


def test_missing_category_header_does_not_shift_year_columns():
    (table,) = parse_model_output('<table><tr><th>2024</th><th>2023</th></tr>'
                                '<tr><td>North</td><td>17</td><td>16</td></tr>'
                                '<tr><td>South</td><td>23</td><td>22</td></tr></table>')
    assert table.header == ["", "2024", "2023"]
    assert table.rows[0] == ["North", "17", "16"]


def test_scanned_charts_use_cached_crop_tables_without_native_reconstruction(tmp_path, monkeypatch):
    import pymupdf
    from docparse.charts import integrate
    from docparse.textlayer import TextLayerPage

    pdf = tmp_path / "chart.pdf"
    with pymupdf.open() as doc:
        doc.new_page(width=500, height=500)
        doc.save(pdf)
    monkeypatch.setattr(integrate, "load_chart_vlm", lambda _: {
        "pages": [{"charts": [{"bbox": [100, 100, 500, 400], "text": PIPE}]}]})
    def no_native(*args, **kwargs):
        raise AssertionError("Scanned pages must not use native chart reconstruction")
    monkeypatch.setattr(integrate, "region_from_page", no_native)
    monkeypatch.setattr(integrate.vector, "reconstruct_page", no_native)
    block = {"label": "chart", "bbox": [100, 100, 500, 400], "text": "Chart image"}
    with pymupdf.open(pdf) as doc:
        result = integrate.enhance_chart_blocks([block], doc[0], TextLayerPage(500, 500, scanned=True))
    assert len(result) == 2 and result[1]["source"] == "chart-vlm"
    assert "<th>China</th><td>29</td><td>25</td>" in result[1]["text"]


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
