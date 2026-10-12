#!/usr/bin/env bash
# Score a normalization-only pipeline against another pipeline's saved raw inference
# outputs (no GPU). Usage: scripts/reuse_raw.sh <source_pipeline> <target_pipeline>
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:?source pipeline}"; DST="${2:?target pipeline}"
mkdir -p "output/$DST"
for d in output/"$SRC"/*/; do
  g=$(basename "$d"); mkdir -p "output/$DST/$g"
  cp "$d"/*.raw.json "output/$DST/$g/" 2>/dev/null || true
done
uv run python - "$SRC" "$DST" <<'PY'
import json, sys
src, dst = sys.argv[1:3]
m = json.load(open(f"output/{src}/_metadata.json"))
from parse_bench.inference.pipelines import get_pipeline
import docparse.pipelines  # noqa
m["pipeline"] = get_pipeline(dst).model_dump()
json.dump(m, open(f"output/{dst}/_metadata.json", "w"), indent=2)
PY
uv run docparse inference renormalize "output/$DST" --pipeline_name "$DST" --force | grep -E "Success|Errors"
uv run docparse run "$DST" --test --skip_inference > "output/$DST/_eval.log" 2>&1 || true
echo "failed examples: $(grep -cE FAILED "output/$DST/_eval.log" || true)"
