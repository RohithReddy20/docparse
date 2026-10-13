from docparse.fusion import _det_boxes


def test_lowercase_caption_and_title_detections_are_retained():
    detections = {"width": 500, "height": 500, "predictions": [
        {"bbox": [10, 10, 100, 30], "label": "caption", "score": 0.9},
        {"bbox": [10, 50, 100, 70], "label": "title", "score": 0.9}]}
    assert [d["label"] for d in _det_boxes(detections)] == ["caption", "title"]
