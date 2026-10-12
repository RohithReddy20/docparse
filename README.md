# docparse

Open, multi-step document parser targeting [ParseBench](https://github.com/run-llama/ParseBench).
Goal: beat LlamaParse Cost Effective (80.6 overall @ 0.375¢/page) and approach
Agentic (87.0 @ 1.25¢) using only permissively licensed open models, with cost
measured from real GPU-seconds.

## Why multi-step

No single open model exceeds ~77 on ParseBench; the gaps are charts and semantic
formatting, and most VLMs score 0 on visual grounding because they emit no
bboxes. LlamaParse Agentic wins by orchestrating. Plan:

```
page → triage (text layer?) → layout detection → per-region specialists
     → verifier (cheap rule checks) → escalation on low confidence → assemble
```

## Components (all Apache/MIT)

| Role | Model | ParseBench (single) |
|---|---|---|
| Text + grounding | LightOnOCR-3-0.8B | 74.6 |
| Tables / JSON layout | Infinity-Parser2-Flash (2B) | 73.3 |
| Layout detector | PP-DocLayout-V3 | — |
| Charts (escalation) | LightOnOCR-3-4B → Qwen3.8-27B | 66 / 69 |

Avoided: MinerU, KDL/florin/rakedoc-nano (AGPL), Chandra (non-compete license).

## Budget rules

Zero out-of-pocket. Modal free credit (~$20) is the hard cap and is used only for
benchmark runs; iteration happens on Kaggle. `serve/modal_vllm.py` enforces one
L4 container max, 2-minute idle scale-down, 1-hour hard timeout.

## Quickstart

```bash
uv sync
uv run parse-bench download --test                      # 12 PDFs, 6 MB

DOCPARSE_MODEL=lightonai/LightOnOCR-3-0.8B DOCPARSE_SERVED_NAME=lightonocr3 \
  modal deploy serve/modal_vllm.py                      # prints the endpoint URL
cp .env.example .env                                    # paste URL (+ /v1)

scripts/run_test.sh lightonocr3_0_8b_grounding
uv run docparse serve lightonocr3_0_8b_grounding        # browse reports
```

`docparse` is `parse-bench` with this repo's providers/pipelines registered.

## Layout

- `docparse/providers/` – parse-bench `Provider` implementations
- `docparse/pipelines.py` – named pipeline specs
- `serve/modal_vllm.py` – generic vLLM server on Modal (one app per model)
- `scripts/` – run helpers
