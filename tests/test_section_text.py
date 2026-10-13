import json

import pytest

from docparse.section_text import BASE_FEATURES, FEATURES, VERSION, context_features, features, load_model, probability, rerank
from docparse.textlayer import Line, Span, TextLayerPage


def _page() -> TextLayerPage:
    heading = Span("1. Overview", [50, 50, 200, 65], 15, bold=True)
    body = Span("This paragraph explains the document.", [50, 80, 450, 90], 10)
    return TextLayerPage(500, 500, [Line([heading], heading.bbox), Line([body], body.bbox)], body_size=10)


def _model(tmp_path, probability: float):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"version": VERSION, "features": list(FEATURES), "trees": [[[-1, 0, -1, -1, probability]]],
                               "promote_threshold": 0.95, "demote_threshold": 0.05}))
    return path


def test_features_distinguish_heading_from_body_and_use_page_scale():
    page = _page()
    heading = dict(zip(BASE_FEATURES, features(page, [100, 100, 400, 130])))
    body = dict(zip(BASE_FEATURES, features(page, [100, 160, 900, 180])))
    assert heading["bold"] == 1 and body["bold"] == 0
    assert heading["size_mean"] == 1.5 and body["size_mean"] == 1
    assert heading["numbered"] == 1 and body["sentence_end"] == 1
    assert heading["gap_below"] == 1 and body["gap_above"] == 1.5
    assert features(page, [0, 900, 1000, 1000]) is None
    page.scanned = True
    assert features(page, [100, 100, 400, 130]) is None


def test_detector_context_tracks_raw_and_baseline_labels():
    values = context_features(_page(), [100, 100, 400, 130], "section_header", "text", 0.97)
    assert len(values) == len(FEATURES)
    assert values[-3:] == [1, 0, 0.97]


def test_forest_uses_training_float32_precision_at_split_boundaries():
    model = {"trees": [[[0, 0.89999999, 1, 2, 0], [-1, 0, -1, -1, 0], [-1, 0, -1, -1, 1]]]}
    assert probability(model, [0.9]) == 0


def test_reranker_is_opt_in_and_preserves_non_text_classes(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCPARSE_SECTION_TEXT_MODEL", raising=False)
    assert rerank("text", [100, 100, 400, 130], _page()) == "text"
    monkeypatch.setenv("DOCPARSE_SECTION_TEXT_MODEL", str(_model(tmp_path, 1)))
    assert rerank("text", [100, 100, 400, 130], _page()) == "section_header"
    assert rerank("image", [100, 100, 400, 130], _page()) == "image"
    assert rerank("title", [100, 100, 400, 130], _page()) == "title"
    assert rerank("list", [100, 100, 400, 130], _page()) == "list"
    assert rerank("text", [100, 100, 400, 130], _page(), "title", 0.99) == "text"
    assert rerank("text", [100, 100, 400, 130], None) == "text"


def test_reranker_demotes_only_confident_text_and_keeps_uncertain_heading(tmp_path, monkeypatch):
    path = _model(tmp_path, 0.02)
    monkeypatch.setenv("DOCPARSE_SECTION_TEXT_MODEL", str(path))
    assert rerank("section_header", [100, 100, 400, 130], _page()) == "text"
    path = _model(tmp_path, 0.5)
    load_model.cache_clear()
    assert rerank("section_header", [100, 100, 400, 130], _page()) == "section_header"


def test_incompatible_or_unconstrained_model_is_rejected(tmp_path):
    path = _model(tmp_path, 1)
    model = json.loads(path.read_text())
    model["features"] = model["features"][::-1]
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="Incompatible"):
        load_model(str(path))
    model["features"] = list(FEATURES)
    model["promote_threshold"] = 0.5
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="conservative"):
        load_model(str(path))


def test_layout_reranking_does_not_mutate_markdown_blocks(tmp_path, monkeypatch):
    from docparse.fusion import detector_items
    from docparse.providers.lightonocr3 import blocks_to_markdown

    monkeypatch.setenv("DOCPARSE_SECTION_TEXT_MODEL", str(_model(tmp_path, 0.02)))
    blocks = [{"label": "section_header", "bbox": [100, 100, 400, 130], "text": "## 1. Overview", "continuation": False}]
    detections = {"width": 500, "height": 500, "predictions": [
        {"bbox": [50, 50, 200, 65], "label": "Section-header", "score": 0.99, "order": 0}]}
    before = blocks_to_markdown(blocks)
    (item,) = detector_items(blocks, detections, _page())
    assert item["label"] == "text"
    assert blocks[0]["label"] == "section_header"
    assert blocks_to_markdown(blocks) == before


def test_vlm_heading_promotion_is_preserved(tmp_path, monkeypatch):
    from docparse.fusion import detector_items

    monkeypatch.setenv("DOCPARSE_SECTION_TEXT_MODEL", str(_model(tmp_path, 0.02)))
    blocks = [{"label": "section_header", "bbox": [100, 100, 400, 130], "text": "## 1. Overview", "continuation": False}]
    dets = {"width": 500, "height": 500, "predictions": [{"bbox": [50, 50, 200, 65], "label": "text", "score": 0.99}]}
    (item,) = detector_items(blocks, dets, _page())
    assert item["label"] == "section_header"


def test_font_inferred_heading_can_be_reranked_without_changing_markdown(tmp_path, monkeypatch):
    from docparse.fusion import detector_items
    from docparse.providers.lightonocr3 import blocks_to_markdown

    monkeypatch.setenv("DOCPARSE_SECTION_TEXT_MODEL", str(_model(tmp_path, 0.02)))
    block = {"label": "section_header", "_vlm_label": "text", "source": "styled-vlm-split",
             "bbox": [100, 100, 400, 130], "text": "## 1. Overview", "continuation": False}
    detections = {"width": 500, "height": 500, "predictions": [
        {"bbox": [50, 50, 200, 65], "label": "text", "score": 0.99}]}
    before = blocks_to_markdown([block])
    (item,) = detector_items([block], detections, _page())
    assert item["label"] == "text"
    assert blocks_to_markdown([block]) == before


def test_gate_calibration_uses_changed_labels_only():
    import numpy as np
    from scripts.train_section_text import select_gate

    y = np.array([0] * 1000 + [1] * 100)
    p = np.array([0.01] * 1100)
    baseline = np.array([0] * 1000 + [1] * 100)
    assert not select_gate(y, p, baseline, False)["enabled"]


def test_detector_training_excludes_unmatched_and_visual_ground_truth():
    from scripts.doclaynet_heron_features import _target

    classes = {1: "Text", 2: "Title", 3: "Picture"}
    annotations = [
        {"bbox": [0, 0, 100, 100], "category_id": 3},
        {"bbox": [10, 10, 20, 10], "category_id": 1},
        {"bbox": [50, 50, 20, 10], "category_id": 2},
        {"bbox": [80, 50, 20, 10], "category_id": 4}]
    classes[4] = "Section-header"
    assert _target([0, 0, 100, 100], annotations, classes) is None
    assert _target([10, 10, 30, 20], annotations, classes) == 0
    assert _target([50, 50, 70, 60], annotations, classes) is None
    assert _target([80, 50, 100, 60], annotations, classes) == 1
    assert _target([200, 200, 300, 300], annotations, classes) is None
