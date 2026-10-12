"""Oracle analysis over parse-bench results: what could a router achieve?

Reads ``output/<pipeline>/<category>/_evaluation_report.json`` for each pipeline
and reports, per ParseBench dimension:

- each pipeline's mean score (the number the leaderboard shows),
- **oracle-by-dimension**: pick the best single pipeline per dimension,
- **oracle-by-example**: per example take the max over pipelines — the ceiling
  for a perfect per-page router,
- optionally the cheapest pipeline within ``--within`` points of the
  per-dimension oracle, given a ``costs.json`` of ``{pipeline: cents_per_page}``.

Overall is parse-bench's definition: the unweighted mean of the per-category
headline metric over the categories present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

# Headline metric per category (mirrors parse_bench.analysis.aggregation_report
# ``_DEFAULT_METRICS``; categories not listed fall back to ``rule_pass_rate``).
HEADLINE_METRIC: dict[str, str] = {
    "table": "grits_trm_composite",
    "layout": "layout_element_rule_pass_rate",
    "text_content": "content_faithfulness",
    "text_formatting": "semantic_formatting",
    "chart": "rule_pass_rate",
}
DIMENSIONS = ["table", "chart", "text_content", "text_formatting", "layout"]

Scores = dict[str, dict[str, float]]  # dimension -> example_id -> score


def headline_metric(dimension: str) -> str:
    return HEADLINE_METRIC.get(dimension, "rule_pass_rate")


def load_scores(output_dir: Path, pipeline: str) -> Scores:
    """Per-example headline scores for every evaluated category of ``pipeline``."""
    pipeline_dir = Path(output_dir) / pipeline
    if not pipeline_dir.is_dir():
        raise FileNotFoundError(f"No results for pipeline '{pipeline}' under {output_dir}")
    scores: Scores = {}
    for report in sorted(pipeline_dir.glob("*/_evaluation_report.json")):
        dim = report.parent.name
        metric = headline_metric(dim)
        summary = json.loads(report.read_text(encoding="utf-8"))
        per_example: dict[str, float] = {}
        for res in summary.get("per_example_results", []):
            for m in res.get("metrics", []):
                if m.get("metric_name") == metric:
                    per_example[res["example_id"]] = float(m["value"])
                    break
        if per_example:
            scores[dim] = per_example
    return scores


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def oracle(
    pipelines: list[str],
    output_dir: Path = Path("output"),
    costs: dict[str, float] | None = None,
    within: float = 2.0,
) -> dict:
    """Compute per-dimension means, oracle-by-dimension and oracle-by-example.

    Scores are returned on a 0-100 scale. Returns a dict with keys
    ``dimensions``, ``per_pipeline``, ``by_dimension``, ``by_example``,
    ``overall`` and (when ``costs`` is given) ``cheapest_within``.
    """
    loaded = {p: load_scores(output_dir, p) for p in pipelines}
    dims = [d for d in DIMENSIONS if any(d in s for s in loaded.values())]
    dims += sorted({d for s in loaded.values() for d in s} - set(dims))

    per_pipeline: dict[str, dict[str, float | None]] = {p: {} for p in pipelines}
    by_dimension: dict[str, dict] = {}
    by_example: dict[str, float | None] = {}
    cheapest_within: dict[str, dict | None] = {}

    for d in dims:
        for p in pipelines:
            vals = list(loaded[p].get(d, {}).values())
            m = _mean(vals)
            per_pipeline[p][d] = 100 * m if m is not None else None

        scored = [(per_pipeline[p][d], p) for p in pipelines if per_pipeline[p][d] is not None]
        best_score, best_p = max(scored) if scored else (None, None)
        by_dimension[d] = {"pipeline": best_p, "score": best_score}

        ids = sorted({eid for p in pipelines for eid in loaded[p].get(d, {})})
        maxes = [max(loaded[p][d][eid] for p in pipelines if eid in loaded[p].get(d, {})) for eid in ids]
        ex_mean = _mean(maxes)
        by_example[d] = 100 * ex_mean if ex_mean is not None else None

        if costs is not None and best_score is not None:
            ok = [
                (costs[p], per_pipeline[p][d], p)
                for p in pipelines
                if p in costs and per_pipeline[p][d] is not None and per_pipeline[p][d] >= best_score - within
            ]
            cheapest_within[d] = (
                {"pipeline": ok[0][2], "score": ok[0][1], "cents_per_page": ok[0][0]} if (ok := sorted(ok)) else None
            )

    def overall(values: dict[str, float | None]) -> float | None:
        return _mean([v for v in values.values() if v is not None])

    result = {
        "dimensions": dims,
        "per_pipeline": per_pipeline,
        "by_dimension": by_dimension,
        "by_example": by_example,
        "overall": {
            **{p: overall(per_pipeline[p]) for p in pipelines},
            "oracle_by_dimension": overall({d: by_dimension[d]["score"] for d in dims}),
            "oracle_by_example": overall(by_example),
        },
    }
    if costs is not None:
        result["cheapest_within"] = cheapest_within
        result["within"] = within
    return result


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def render_markdown(r: dict) -> str:
    dims = r["dimensions"]
    pipelines = list(r["per_pipeline"])
    header = "| Pipeline | " + " | ".join(dims) + " | Overall |"
    sep = "|" + "---|" * (len(dims) + 2)
    rows = [header, sep]
    for p in pipelines:
        cells = [_fmt(r["per_pipeline"][p][d]) for d in dims]
        rows.append(f"| {p} | " + " | ".join(cells) + f" | {_fmt(r['overall'][p])} |")
    rows.append(
        "| **oracle-by-dimension** | "
        + " | ".join(f"{_fmt(r['by_dimension'][d]['score'])} ({r['by_dimension'][d]['pipeline']})" for d in dims)
        + f" | **{_fmt(r['overall']['oracle_by_dimension'])}** |"
    )
    rows.append(
        "| **oracle-by-example** | "
        + " | ".join(_fmt(r["by_example"][d]) for d in dims)
        + f" | **{_fmt(r['overall']['oracle_by_example'])}** |"
    )
    if "cheapest_within" in r:
        rows.append(
            f"| cheapest within {r['within']:g} pts | "
            + " | ".join(
                (f"{c['pipeline']} ({c['cents_per_page']:g}¢)" if (c := r["cheapest_within"].get(d)) else "—")
                for d in dims
            )
            + " | |"
        )
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Oracle (best-per-dimension / per-example) over parse-bench results.")
    ap.add_argument("pipelines", nargs="+", help="pipeline names under --output_dir")
    ap.add_argument("--output_dir", default="output")
    ap.add_argument("--costs", help="JSON file: {pipeline: cents_per_page}")
    ap.add_argument("--within", type=float, default=2.0, help="points below the oracle still acceptable (default 2)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a markdown table")
    args = ap.parse_args()

    costs = json.loads(Path(args.costs).read_text()) if args.costs else None
    r = oracle(args.pipelines, Path(args.output_dir), costs=costs, within=args.within)
    print(json.dumps(r, indent=2) if args.json else render_markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
