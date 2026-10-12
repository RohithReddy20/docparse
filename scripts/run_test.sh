#!/usr/bin/env bash
# Run a pipeline on the ParseBench --test split (3 files/category, ~12 PDFs).
# Usage: scripts/run_test.sh <pipeline_name>
set -euo pipefail
cd "$(dirname "$0")/.."
PIPELINE="${1:?pipeline name required}"
uv run docparse run "$PIPELINE" --test --max_concurrent 4
echo
echo "Reports: output/$PIPELINE/   (view: uv run docparse serve $PIPELINE)"
