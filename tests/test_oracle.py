import json
from pathlib import Path

import pytest

from docparse.oracle import load_scores, oracle, render_markdown

FIXTURES = Path(__file__).parent / "fixtures"
OUTPUT = FIXTURES / "output"


def test_load_scores_picks_headline_metric():
    s = load_scores(OUTPUT, "fake_a")
    assert s == {"table": {"ex1": 0.9, "ex2": 0.5}, "chart": {"ex1": 0.2, "ex2": 0.8}}


def test_missing_pipeline_raises():
    with pytest.raises(FileNotFoundError):
        load_scores(OUTPUT, "nope")


def test_oracle_by_dimension_and_by_example():
    r = oracle(["fake_a", "fake_b"], OUTPUT)
    assert r["dimensions"] == ["table", "chart"]

    # Leaderboard-style means (0-100).
    assert r["per_pipeline"]["fake_a"] == {"table": 70.0, "chart": 50.0}
    assert r["per_pipeline"]["fake_b"] == pytest.approx({"table": 60.0, "chart": 60.0})
    assert r["overall"]["fake_a"] == 60.0
    assert r["overall"]["fake_b"] == pytest.approx(60.0)

    # Best single pipeline per dimension.
    assert r["by_dimension"]["table"] == {"pipeline": "fake_a", "score": 70.0}
    assert r["by_dimension"]["chart"] == {"pipeline": "fake_b", "score": pytest.approx(60.0)}
    assert r["overall"]["oracle_by_dimension"] == pytest.approx(65.0)

    # Per-example max: table max(.9,.4), max(.5,.8) -> .85; chart max(.2,.6), max(.8,.6) -> .7
    assert r["by_example"]["table"] == pytest.approx(85.0)
    assert r["by_example"]["chart"] == pytest.approx(70.0)
    assert r["overall"]["oracle_by_example"] == pytest.approx(77.5)


def test_cheapest_within():
    costs = json.loads((FIXTURES / "costs.json").read_text())
    r = oracle(["fake_a", "fake_b"], OUTPUT, costs=costs, within=10.0)
    # table oracle 70 (fake_a); fake_b at 60 is within 10 and cheaper.
    assert r["cheapest_within"]["table"]["pipeline"] == "fake_b"
    r2 = oracle(["fake_a", "fake_b"], OUTPUT, costs=costs, within=2.0)
    assert r2["cheapest_within"]["table"]["pipeline"] == "fake_a"


def test_render_markdown_has_oracle_rows():
    md = render_markdown(oracle(["fake_a", "fake_b"], OUTPUT))
    assert "oracle-by-dimension" in md and "oracle-by-example" in md
    assert "77.50" in md
