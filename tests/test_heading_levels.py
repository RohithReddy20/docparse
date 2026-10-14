import pytest

from docparse.hybrid import render_blocks
from docparse.textlayer import Line, Span, TextLayerPage


def _page():
    line = Line([Span("Methods", [10, 10, 90, 25], 12, bold=True)], [10, 10, 90, 25])
    return TextLayerPage(100, 100, [line], body_size=10)


@pytest.mark.parametrize("level", range(1, 7))
def test_keeps_explicit_level_when_native_font_rank_differs(monkeypatch, level):
    monkeypatch.setenv("DOCPARSE_HEADING_LEVEL_POLICY", "preserve_explicit")
    block = {"label": "title", "bbox": [100, 100, 900, 250], "text": f"{'#' * level} Methods"}
    result = render_blocks(block, _page())
    assert result[0]["text"] == block["text"]


def test_title_without_explicit_level_uses_native_font_context(monkeypatch):
    monkeypatch.setenv("DOCPARSE_HEADING_LEVEL_POLICY", "preserve_explicit")
    block = {"label": "title", "bbox": [100, 100, 900, 250], "text": "Methods"}
    assert render_blocks(block, _page())[0]["text"] == "# Methods"


def test_legacy_control_uses_native_level(monkeypatch):
    monkeypatch.setenv("DOCPARSE_HEADING_LEVEL_POLICY", "legacy")
    block = {"label": "title", "bbox": [100, 100, 900, 250], "text": "### Methods"}
    assert render_blocks(block, _page())[0]["text"] == "# Methods"


def test_default_keeps_explicit_level(monkeypatch):
    monkeypatch.delenv("DOCPARSE_HEADING_LEVEL_POLICY", raising=False)
    block = {"label": "title", "bbox": [100, 100, 900, 250], "text": "### Methods"}
    assert render_blocks(block, _page())[0]["text"] == "### Methods"
