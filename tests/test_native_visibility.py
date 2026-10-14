import io

import pymupdf
from PIL import Image

from docparse.textlayer import extract_page
from docparse.hybrid import reconcile_page
from docparse.textlayer import Line, Span, TextLayerPage


TEXT = "Visible native text retains its real font styling above a background image. " * 3


def _background(page):
    image = Image.new("RGB", (4, 4), (220, 230, 240))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    page.insert_image(page.rect, stream=buffer.getvalue())


def _text(page, render_mode=0):
    page.insert_textbox(pymupdf.Rect(20, 20, 380, 280), TEXT, fontsize=12, render_mode=render_mode)


def test_visible_native_text_above_background_is_usable(monkeypatch):
    monkeypatch.setenv("DOCPARSE_NATIVE_IMAGE_TEXT", "visible")
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        _background(page)
        _text(page)
        layer = extract_page(page)
    assert layer.has_text
    assert not layer.scanned


def test_invisible_ocr_text_stays_excluded(monkeypatch):
    monkeypatch.setenv("DOCPARSE_NATIVE_IMAGE_TEXT", "visible")
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        _background(page)
        _text(page, render_mode=3)
        assert extract_page(page).scanned


def test_text_covered_by_later_image_stays_excluded(monkeypatch):
    monkeypatch.setenv("DOCPARSE_NATIVE_IMAGE_TEXT", "visible")
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        _text(page)
        _background(page)
        assert extract_page(page).scanned


def test_mixed_visible_and_invisible_text_stays_excluded(monkeypatch):
    monkeypatch.setenv("DOCPARSE_NATIVE_IMAGE_TEXT", "visible")
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        _background(page)
        _text(page)
        page.insert_text((20, 290), "Hidden OCR", render_mode=3)
        assert extract_page(page).scanned


def test_legacy_scan_control_stays_available(monkeypatch):
    monkeypatch.setenv("DOCPARSE_NATIVE_IMAGE_TEXT", "legacy")
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        _background(page)
        _text(page)
        assert extract_page(page).scanned


def test_default_uses_visible_native_text(monkeypatch):
    monkeypatch.delenv("DOCPARSE_NATIVE_IMAGE_TEXT", raising=False)
    monkeypatch.delenv("DOCPARSE_NATIVE_IMAGE_ORPHANS", raising=False)
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        _background(page)
        _text(page)
        layer = extract_page(page)
        assert not layer.scanned
        assert not layer.recover_orphans


def test_image_native_text_does_not_enable_unowned_insertions(monkeypatch):
    monkeypatch.setenv("DOCPARSE_NATIVE_IMAGE_TEXT", "visible")
    monkeypatch.setenv("DOCPARSE_NATIVE_IMAGE_ORPHANS", "skip")
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        _background(page)
        _text(page)
        assert not extract_page(page).recover_orphans


def test_unmatched_text_is_not_inserted_without_ownership():
    body = "This paragraph contains enough native text to use the text layer and retain the model's body content."
    layer = TextLayerPage(100, 100, [
        Line([Span(body, [10, 10, 90, 25], 10)], [10, 10, 90, 25]),
        Line([Span("Image credit", [10, 80, 90, 90], 10)], [10, 80, 90, 90]),
    ], body_size=10, recover_orphans=False)
    block = {"label": "text", "text": body, "bbox": [100, 100, 900, 250]}
    result = reconcile_page([block], layer)
    assert len(result) == 1
    assert result[0]["text"] == body
