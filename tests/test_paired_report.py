import json

import pytest

from docparse.oracle import DIMENSIONS, HEADLINE_METRIC
from scripts.paired_report import compare_runs


def _run(root, score=0.5, version="1.0.4", missing=False):
    for dimension in DIMENSIONS:
        folder = root / dimension
        folder.mkdir(parents=True)
        (folder / "example.raw.json").write_text('{"request": {"example_id": "example"}}')
        metrics = [{"metric_name": HEADLINE_METRIC[dimension], "value": score}]
        examples = [{"example_id": "example", "success": True, "error": None, "metrics": metrics}]
        if missing:
            examples.append({"example_id": "no_rules", "success": True, "error": None, "metrics": []})
            (folder / "no_rules.raw.json").write_text('{"request": {"example_id": "no_rules"}}')
        report = {"total_examples": len(examples), "failed": 0, "skipped": 0,
                  "parse_bench_version": version, "per_example_results": examples}
        (folder / "_evaluation_report.json").write_text(json.dumps(report))
    return root


def _modify(root, fn):
    path = root / "chart" / "_evaluation_report.json"
    report = json.loads(path.read_text())
    fn(report)
    path.write_text(json.dumps(report))


def test_paired_delta_uses_category_means(tmp_path):
    before = _run(tmp_path / "before")
    after = _run(tmp_path / "after")
    _modify(after, lambda r: r["per_example_results"][0]["metrics"][0].update(value=0.75))
    result = compare_runs(before, after, 1)
    assert result["dimensions"]["chart"]["delta"] == 25
    assert result["overall"] == {"baseline": 50, "candidate": 55, "delta": 5}


def test_matching_inapplicable_metrics_are_allowed(tmp_path):
    before = _run(tmp_path / "before", missing=True)
    after = _run(tmp_path / "after", missing=True)
    result = compare_runs(before, after, 2)
    assert result["dimensions"]["chart"]["scored_files"] == 1
    assert result["overall"]["delta"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_rejects_invalid_metric_values(tmp_path, value):
    before = _run(tmp_path / "before")
    after = _run(tmp_path / "after", score=value)
    with pytest.raises(ValueError, match="invalid headline"):
        compare_runs(before, after, 1)


def test_rejects_duplicate_example_ids(tmp_path):
    before = _run(tmp_path / "before")
    after = _run(tmp_path / "after")
    def duplicate(report):
        report["per_example_results"].append(report["per_example_results"][0])
        report["total_examples"] += 1
    _modify(after, duplicate)
    with pytest.raises(ValueError, match="duplicate example"):
        compare_runs(before, after, 1)


def test_rejects_changed_metric_coverage(tmp_path):
    before = _run(tmp_path / "before", missing=True)
    after = _run(tmp_path / "after", missing=True)
    def add_metric(report):
        report["per_example_results"][1]["metrics"] = [{"metric_name": "rule_pass_rate", "value": 1}]
    _modify(after, add_metric)
    with pytest.raises(ValueError, match="headline metric IDs differ"):
        compare_runs(before, after, 2)


def test_rejects_different_evaluator_versions(tmp_path):
    before = _run(tmp_path / "before")
    after = _run(tmp_path / "after", version="2.0.0")
    with pytest.raises(ValueError, match="Evaluator versions differ"):
        compare_runs(before, after, 1)


def test_rejects_incomplete_full_set(tmp_path):
    before = _run(tmp_path / "before")
    after = _run(tmp_path / "after")
    with pytest.raises(ValueError, match="Expected 2078"):
        compare_runs(before, after)


def test_rejects_failed_examples(tmp_path):
    before = _run(tmp_path / "before")
    after = _run(tmp_path / "after")
    _modify(after, lambda r: r["per_example_results"][0].update(success=False, error="failed"))
    with pytest.raises(ValueError, match="failed example"):
        compare_runs(before, after, 1)


def test_report_skipped_counter_does_not_hide_successful_examples(tmp_path):
    before = _run(tmp_path / "before")
    after = _run(tmp_path / "after")
    _modify(before, lambda r: r.update(skipped=1))
    _modify(after, lambda r: r.update(skipped=1))
    assert compare_runs(before, after, 1)["overall"]["delta"] == 0


def test_rejects_duplicate_headline_metrics(tmp_path):
    before = _run(tmp_path / "before")
    after = _run(tmp_path / "after")
    def duplicate(report):
        metrics = report["per_example_results"][0]["metrics"]
        metrics.append(metrics[0])
    _modify(after, duplicate)
    with pytest.raises(ValueError, match="duplicate headline"):
        compare_runs(before, after, 1)


def test_rejects_missing_raw_cache_coverage(tmp_path):
    before = _run(tmp_path / "before")
    after = _run(tmp_path / "after")
    for p in after.glob("*/*.raw.json"):
        p.unlink()
    with pytest.raises(ValueError, match="raw cache and evaluated IDs differ"):
        compare_runs(before, after, 1)
