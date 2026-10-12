#!/usr/bin/env bash
# Deploy infly/Infinity-Parser2-Flash as a vLLM server on Modal.
# Prints the command; only runs it with --yes (deploy itself is a CPU image
# build; GPU cost starts on the first request, ~$0.80/hr on L4, scale-to-zero).
#
# Notes:
#  - served name MUST be the HF id: the infinity_parser2 SDK sends model=<model_name>.
#  - SDK requests up to max_new_tokens (16384 in our pipeline) → raise max-model-len.
#  - thinking is disabled by serve/modal_vllm.py's --default-chat-template-kwargs.
set -euo pipefail
cd "$(dirname "$0")/.."

CMD='DOCPARSE_MODEL=infly/Infinity-Parser2-Flash \
DOCPARSE_SERVED_NAME=infly/Infinity-Parser2-Flash \
DOCPARSE_GPU=L4 \
DOCPARSE_VLLM_ARGS="--max-model-len 32768" \
modal deploy serve/modal_vllm.py'

echo "$CMD"
if [[ "${1:-}" == "--yes" ]]; then
  eval "$CMD"
  echo
  echo "Set INFINITY_ENDPOINT_URL=<printed URL>/v1 in .env, then: scripts/run_test.sh infinity_parser2_flash_modal"
else
  echo
  echo "(dry run — re-run with --yes to deploy)"
fi
