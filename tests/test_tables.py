from docparse.tables import canonicalize_table_html, parse_rows


def _rows(html):
    return [[(c.tag, c.text, c.colspan) for c in r] for r in parse_rows(html)]


def test_extend_header_promotes_subheader_row():
    h = (
        "<table><thead><tr><th>Rule Name</th><th>Basic Limit</th><th></th><th></th></tr></thead>"
        "<tbody><tr><td></td><td>$2,000</td><td>$4,000</td><td>$5,000</td></tr>"
        "<tr><td>Rule 1</td><td>1.00</td><td>1.10</td><td>1.20</td></tr></tbody></table>"
    )
    out = _rows(canonicalize_table_html(h))
    assert all(tag == "th" for tag, _, _ in out[1])
    assert out[2][0] == ("td", "Rule 1", 1)


def test_title_row_single_cell_becomes_colspan_th():
    h = (
        "<table><tr><td>Sundays to Manhattan</td><td></td><td></td></tr>"
        "<tr><td>Van Cortlandt</td><td>238 St</td><td>Dyckman</td></tr>"
        "<tr><td>12:01</td><td>12:03</td><td>12:08</td></tr></table>"
    )
    out = _rows(canonicalize_table_html(h))
    assert out[0] == [("th", "Sundays to Manhattan", 3)]


def test_context_title_inserted_when_header_is_data():
    h = (
        "<table><thead><tr><th>ICC18 ENT-06 1805</th><th>Single Premium</th></tr></thead>"
        "<tbody><tr><td>ICC18 ENT-06 1805 DP</td><td>Data Page</td></tr>"
        "<tr><td>ICC18 ENT-06 1805 DP-A</td><td>Data Page</td></tr>"
        "<tr><td>ICC22 EE.48 2208</td><td>Endorsement</td></tr></tbody></table>"
    )
    blocks = [{"label": "section_header", "text": "## Supporting Policy Form:"}, {"label": "table", "text": h}]
    out = _rows(canonicalize_table_html(h, blocks, 1))
    assert out[0] == [("th", "Supporting Policy Form:", 2)]
    assert out[1][0][0] == "td"


def test_untouched_when_nothing_applies():
    h = "<table><thead><tr><th>A</th><th>B</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
    assert canonicalize_table_html(h) == h
