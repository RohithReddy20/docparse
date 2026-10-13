"""Compare complete, paired ParseBench runs without changing aggregation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from docparse.oracle import DIMENSIONS, HEADLINE_METRIC


def load_run(root: Path) -> dict:
    scores = {}
    evaluated = {}
    versions = set()
    for dimension in DIMENSIONS:
        path = root / dimension / "_evaluation_report.json"
        report = json.loads(path.read_text())
        examples = report["per_example_results"]
        if report["total_examples"] != len(examples):
            raise ValueError(f"{path}: inconsistent example count")
        # The release can report nonzero `skipped` alongside successful results
        # for every ID. Check the actual results and paired metric eligibility.
        if report.get("failed", 0):
            raise ValueError(f"{path}: failed evaluations")
        versions.add(report["parse_bench_version"])
        ids = set()
        values = {}
        for example in examples:
            example_id = example["example_id"]
            if example_id in ids:
                raise ValueError(f"{path}: duplicate example {example_id}")
            ids.add(example_id)
            if not example["success"] or example.get("error"):
                raise ValueError(f"{path}: failed example {example_id}")
            metrics = [
                m for m in example["metrics"]
                if m["metric_name"] == HEADLINE_METRIC[dimension]
            ]
            if len(metrics) > 1:
                raise ValueError(f"{path}: duplicate headline for {example_id}")
            if metrics:
                value = float(metrics[0]["value"])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{path}: invalid headline for {example_id}")
                values[example_id] = value
        if not values:
            raise ValueError(f"{path}: no headline metrics")
        scores[dimension] = values
        evaluated[dimension] = ids
    if len(versions) != 1:
        raise ValueError(f"{root}: mixed evaluator versions")
    raw_ids = {
        json.loads(p.read_text())["request"]["example_id"]
        for p in root.glob("*/*.raw.json")
    }
    all_ids = set().union(*evaluated.values())
    if raw_ids != all_ids:
        raise ValueError(f"{root}: raw cache and evaluated IDs differ")
    return {"scores": scores, "evaluated": evaluated, "version": versions.pop(), "ids": all_ids}


def compare_runs(baseline: Path, candidate: Path, expected_files: int = 2078) -> dict:
    before, after = load_run(baseline), load_run(candidate)
    if before["version"] != after["version"]:
        raise ValueError("Evaluator versions differ")
    if len(before["ids"]) != expected_files or len(after["ids"]) != expected_files:
        raise ValueError(f"Expected {expected_files} unique files in each run")
    result = {"files": expected_files, "parse_bench_version": before["version"], "dimensions": {}}
    for dimension in DIMENSIONS:
        if before["evaluated"][dimension] != after["evaluated"][dimension]:
            raise ValueError(f"{dimension}: evaluated IDs differ")
        left, right = before["scores"][dimension], after["scores"][dimension]
        # Some examples legitimately have no applicable headline metric. Their
        # availability must agree, rather than silently changing the denominator.
        if left.keys() != right.keys():
            raise ValueError(f"{dimension}: headline metric IDs differ")
        deltas = [(i, 100 * (right[i] - left[i])) for i in sorted(left)]
        changed = [{"example_id": i, "delta": delta} for i, delta in deltas if abs(delta) > 1e-10]
        result["dimensions"][dimension] = {
            "evaluated_files": len(before["evaluated"][dimension]),
            "scored_files": len(left),
            "baseline": 100 * mean(left.values()),
            "candidate": 100 * mean(right.values()),
            "delta": mean(delta for _, delta in deltas),
            "improved": sum(delta > 1e-10 for _, delta in deltas),
            "regressed": sum(delta < -1e-10 for _, delta in deltas),
            "changed": changed,
        }
    result["overall"] = {
        key: mean(d[key] for d in result["dimensions"].values())
        for key in ("baseline", "candidate", "delta")
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline pipeline directory")
    parser.add_argument("candidate", type=Path, help="Candidate pipeline directory")
    parser.add_argument("--expected-files", type=int, default=2078)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = compare_runs(args.baseline, args.candidate, args.expected_files)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
