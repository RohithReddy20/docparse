"""Train on DocLayNet's official splits in Kaggle; export a small portable forest.

Requires scikit-learn in the training environment only. Generate features with
`doclaynet_heron_features.py`. No ParseBench files or labels are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

from docparse.section_text import FEATURES, VERSION, probability

SEED = 13


def load_split(root: Path, name: str):
    import numpy as np

    data = np.load(root / f"{name}.npz")
    meta = json.loads((root / f"{name}.json").read_text())
    assert data["x"].shape[1] == len(FEATURES)
    return data["x"], data["y"], meta


def export_forest(forest) -> list[list[list]]:
    trees = []
    for estimator in forest.estimators_:
        t = estimator.tree_
        nodes = []
        for i in range(t.node_count):
            leaf = t.children_left[i] < 0
            values = t.value[i][0]
            p = float(values[1] / sum(values))
            nodes.append([-1 if leaf else int(t.feature[i]), float(t.threshold[i]),
                          int(t.children_left[i]), int(t.children_right[i]), p])
        trees.append(nodes)
    return trees


def select_gate(y, probabilities, baseline, promote: bool, *, min_count=20, min_precision=0.95) -> dict:
    import numpy as np

    thresholds = [0.9, 0.92, 0.95, 0.97, 0.98, 0.99, 1.0] if promote else [0.1, 0.075, 0.05, 0.03, 0.02, 0.01, 0.005, 0.0]
    eligible = []
    for threshold in thresholds:
        mask = (probabilities >= threshold if promote else probabilities <= threshold) & (baseline != int(promote))
        count = int(mask.sum())
        precision = float(np.mean(y[mask] == int(promote))) if count else 0.0
        if count >= min_count and precision >= min_precision:
            eligible.append({"enabled": True, "threshold": threshold, "count": count, "precision": precision})
    if eligible:
        return max(eligible, key=lambda gate: gate["count"] * (2 * gate["precision"] - 1))
    return {"enabled": False, "threshold": 1.0 if promote else 0.0, "count": 0, "precision": None}


def evaluate(y, p, baseline, model: dict) -> dict:
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

    pred = p >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    result = {"baseline_accuracy": float(np.mean(baseline == y)), "boxes": len(y), "heading_fraction": float(np.mean(y)), "accuracy": float(np.mean(pred == y)),
              "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "precision": float(precision), "recall": float(recall), "f1": float(f1)}
    final = baseline.copy()
    for name, positive in (("promote", True), ("demote", False)):
        mask = (p >= model["promote_threshold"] if positive else p <= model["demote_threshold"]) & (baseline != int(positive))
        if not model[f"{name}_enabled"]:
            mask[:] = False
        final[mask] = int(positive)
        result[name] = {"count": int(mask.sum()), "precision": float(np.mean(y[mask] == int(positive))) if mask.any() else None}
    result["gated_accuracy"] = float(np.mean(final == y))
    result["gated_net_correct"] = int(np.sum(final == y) - np.sum(baseline == y))
    return result


def main() -> None:
    import numpy as np
    import sklearn
    from sklearn.ensemble import RandomForestClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--baseline-error-weight", type=float, default=1.0)
    parser.add_argument("--min-gate-count", type=int, default=20)
    parser.add_argument("--gate-precision", type=float, default=0.95)
    args = parser.parse_args()
    if not math.isfinite(args.baseline_error_weight) or args.baseline_error_weight < 1:
        parser.error("Baseline error weight must be finite and at least 1")
    if args.min_gate_count < 1 or not 0.5 < args.gate_precision <= 1:
        parser.error("Gate count must be positive and precision must be greater than 0.5 and at most 1")
    x_train, y_train, train = load_split(args.features, "train")
    x_val, y_val, val = load_split(args.features, "validation")
    x_test, y_test, test = load_split(args.features, "test")
    assert not set(train["documents"]) & set(val["documents"])
    assert not (set(train["documents"]) | set(val["documents"])) & set(test["documents"])
    baseline_index = FEATURES.index("baseline_heading")
    base_val, base_test = x_val[:, baseline_index], x_test[:, baseline_index]
    params = dict(n_estimators=120, max_depth=12, min_samples_leaf=10, max_features=0.7, n_jobs=-1, random_state=SEED)
    sample_weight = np.where(x_train[:, baseline_index] == y_train, 1.0, args.baseline_error_weight)
    forest = RandomForestClassifier(**params).fit(np.asarray(x_train, dtype=np.float32), y_train, sample_weight=sample_weight)
    assert list(forest.classes_) == [0, 1]
    p_val = forest.predict_proba(np.asarray(x_val, dtype=np.float32))[:, 1]
    policy = {"min_count": args.min_gate_count, "min_precision": args.gate_precision}
    promote = select_gate(np.asarray(y_val), p_val, base_val, True, **policy)
    demote = select_gate(np.asarray(y_val), p_val, base_val, False, **policy)
    model = {"version": VERSION, "features": list(FEATURES), "trees": export_forest(forest),
             "promote_threshold": promote["threshold"], "demote_threshold": demote["threshold"],
             "promote_enabled": promote["enabled"], "demote_enabled": demote["enabled"],
             "training": {"dataset": "DocLayNet", "source": "https://github.com/DS4SD/DocLayNet", "seed": SEED,
                          "splits": "official train/val/test, sampled without cross-split document overlap", "detector": train["detector"],
                          "matching": train["matching"], "calibration": "precision of changed baseline labels on validation only", "sklearn": sklearn.__version__, "parameters": params,
                          "baseline_error_weight": args.baseline_error_weight,
                          "calibration_policy": {"minimum_changed_cases": args.min_gate_count, "minimum_precision": args.gate_precision}}}
    probe = np.asarray(x_val[:1000], dtype=np.float32)
    parity = max(abs(probability(model, list(v)) - float(p)) for v, p in zip(probe, forest.predict_proba(probe)[:, 1]))
    assert parity < 1e-10, parity
    report = {"training": model["training"], "features": list(FEATURES), "train": train, "validation": val, "test": test,
              "export_max_error": parity, "selected_gates": {"promote": promote, "demote": demote},
              "validation_metrics": evaluate(np.asarray(y_val), p_val, base_val, model),
              "test_metrics": evaluate(np.asarray(y_test), forest.predict_proba(np.asarray(x_test, dtype=np.float32))[:, 1], base_test, model),
              "feature_importances": dict(zip(FEATURES, map(float, forest.feature_importances_)))}
    args.output.mkdir(parents=True, exist_ok=True)
    model_path, report_path = args.output / "section_text_model.json", args.output / "section_text_report.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))
    report["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report, indent=2))
    with zipfile.ZipFile(args.output / "section_text_artifacts.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(model_path, model_path.name)
        bundle.write(report_path, report_path.name)
    print(json.dumps({k: report[k] for k in ("selected_gates", "validation_metrics", "test_metrics", "export_max_error")}, indent=2), flush=True)
    print("Compressed artifact bytes:", (args.output / "section_text_artifacts.zip").stat().st_size, flush=True)


if __name__ == "__main__":
    main()
