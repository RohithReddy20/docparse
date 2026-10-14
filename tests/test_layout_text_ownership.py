import pymupdf

from docparse.fusion import _native_box_text
from docparse.textlayer import Line, Span, TextLayerPage, extract_page


def _page(doc):
    page = doc.new_page(width=300, height=100)
    page.insert_text((20, 30), "Open" + " " * 34 + "Closed", fontsize=12)
    return page


def test_adjacent_words_belong_to_their_own_box(monkeypatch):
    monkeypatch.setenv("DOCPARSE_LAYOUT_TEXT", "words")
    with pymupdf.open() as doc:
        layer = extract_page(_page(doc))
    assert _native_box_text(layer, [0, 100, 400, 400]) == "Open"
    assert _native_box_text(layer, [500, 100, 1000, 400]) == "Closed"


def test_word_is_not_truncated_at_a_detector_boundary(monkeypatch):
    monkeypatch.setenv("DOCPARSE_LAYOUT_TEXT", "words")
    with pymupdf.open() as doc:
        layer = extract_page(_page(doc))
    assert _native_box_text(layer, [0, 100, 130, 400]) == "Open"


def test_extraction_retains_native_spelling_and_character_geometry():
    with pymupdf.open() as doc:
        page = _page(doc)
        expected = [s["text"] for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]
        layer = extract_page(page)
    spans = [s for line in layer.lines for s in line.spans]
    assert [s.text for s in spans] == expected
    assert all("".join(c["c"] for c in s.characters) == s.text for s in spans)


def test_rotated_pages_keep_legacy_attribution(monkeypatch):
    with pymupdf.open() as doc:
        page = _page(doc)
        page.set_rotation(90)
        layer = extract_page(page)
    monkeypatch.setenv("DOCPARSE_LAYOUT_TEXT", "legacy")
    expected = _native_box_text(layer, [0, 0, 1000, 1000])
    monkeypatch.setenv("DOCPARSE_LAYOUT_TEXT", "words")
    assert _native_box_text(layer, [0, 0, 1000, 1000]) == expected


def test_word_across_font_spans_is_not_given_extra_spaces(monkeypatch):
    monkeypatch.setenv("DOCPARSE_LAYOUT_TEXT", "words")
    a = Span("AB", [10, 10, 30, 20], 10, bold=True, characters=[
        {"c": "A", "bbox": [10, 10, 20, 20]}, {"c": "B", "bbox": [20, 10, 30, 20]}])
    b = Span("CD", [30, 10, 50, 20], 10, characters=[
        {"c": "C", "bbox": [30, 10, 40, 20]}, {"c": "D", "bbox": [40, 10, 50, 20]}])
    layer = TextLayerPage(100, 100, [Line([a, b], [10, 10, 50, 20])])
    assert _native_box_text(layer, [0, 0, 1000, 1000]) == "ABCD"


def test_multiple_characters_from_one_glyph_retain_geometry(monkeypatch):
    monkeypatch.setenv("DOCPARSE_LAYOUT_TEXT", "words")
    span = Span("fi sh", [10, 10, 50, 20], 10, characters=[
        {"c": "fi", "bbox": [10, 10, 20, 20]}, {"c": " ", "bbox": [20, 10, 30, 20]},
        {"c": "s", "bbox": [30, 10, 40, 20]}, {"c": "h", "bbox": [40, 10, 50, 20]}])
    layer = TextLayerPage(100, 100, [Line([span], [10, 10, 50, 20])])
    assert _native_box_text(layer, [300, 0, 1000, 300]) == "sh"


def test_default_uses_local_word_attribution(monkeypatch):
    monkeypatch.delenv("DOCPARSE_LAYOUT_TEXT", raising=False)
    with pymupdf.open() as doc:
        layer = extract_page(_page(doc))
    assert _native_box_text(layer, [0, 100, 400, 400]) == "Open"
    assert _native_box_text(layer, [0, 500, 400, 600]) == ""
