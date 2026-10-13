import json

import pymupdf
from PIL import Image

from scripts import doclaynet_heron_features as collector


def test_collection_keeps_partial_batch_and_excludes_other_split_documents(tmp_path, monkeypatch):
    images, annotations = [], []
    for i in range(5):
        with pymupdf.open() as doc:
            page = doc.new_page(width=500, height=500)
            page.insert_text((50, 100), "A paragraph for detector feature extraction.", fontsize=10)
            doc.save(tmp_path / f"{i}.pdf")
        Image.new("RGB", (500, 500), "white").save(tmp_path / f"{i}.png")
        images.append({"id": i, "file_name": f"{i}.png", "doc_name": f"doc-{i}", "precedence": 0})
        annotations.append({"image_id": i, "bbox": [45, 85, 350, 25], "category_id": 1})
    coco = tmp_path / "train.json"
    coco.write_text(json.dumps({"images": images, "annotations": annotations,
                                "categories": [{"id": 1, "name": "Text"}]}))
    monkeypatch.setattr(collector, "_detect", lambda engine, pages: [
        [("text", 0.99, [45, 85, 395, 110])] for _ in pages])
    progress = tmp_path / "progress.json"
    x, y, meta = collector.collect(coco, tmp_path, tmp_path, 3, {"doc-0"}, [None, None], 1, progress)
    assert x.shape == (3, 25)
    assert y.tolist() == [0, 0, 0]
    assert meta["counts"]["pages"] == 3
    assert "doc-0" not in meta["documents"]
    assert json.loads(progress.read_text())["counts"]["pages"] == 3


def test_collection_rejects_detector_that_drops_a_page(tmp_path, monkeypatch):
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    import pytest

    monkeypatch.setattr(collector, "_detect", lambda engine, pages: [])
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="one prediction list per page"):
            collector._consume([(None, None, None)], [None], 1, pool, {}, {}, [], [], set(), Counter())
