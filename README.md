# docparse

Open, multi-step document parser targeting [ParseBench](https://github.com/run-llama/ParseBench).
Goal: beat LlamaParse Cost Effective (80.6 overall @ 0.375¢/page) and approach
Agentic (87.0 @ 1.25¢) using only permissively licensed open models, with cost
measured from real GPU-seconds.

## Method: text layer + small VLM

No single open model exceeds ~77 on ParseBench, and they all share the same
ceiling because they only look at pixels. 93% of ParseBench pages are born-digital
(`scripts/analyze_text_layer.py`): the PDF text layer carries exact characters,
bold/underline/strikethrough/highlight, font sizes (heading hierarchy), printed
chart data labels, and the vector bars of 89% of chart pages. `docparse_hybrid_v0`
uses the text layer for *content* and a 0.8B VLM for *structure*:

```
PDF page ─┬─ text layer (PyMuPDF, CPU)  : spans + styling + drawings
          └─ LightOnOCR-3-0.8B grounding: block order, labels, bboxes, table/chart HTML
                     │
          reconcile  (docparse/hybrid.py)
            • each VLM text block re-rendered from the text-layer lines it covers
              (styling markers, heading levels from font size, exact digits)
            • merged blocks split on paragraph gaps / list items / bold lead lines
            • lines the VLM skipped recovered as blocks; header/footer/page-number
              lifted into structured fields
            • scanned pages (raster covers the page) keep the VLM output whole
          layout     (docparse/fusion.py) — Docling heron (RT-DETRv2, the 17
            Canonical17 classes ParseBench's GT uses; serve/modal_heron_dump.py)
            boxes are the layout elements, text attributed from the layer; VLM
            blocks fill what the detector misses; markdown untouched
          charts     (docparse/charts/)
            • VLM table snapped to printed data labels (textlabels.snap_table)
            • extra candidate tables from text labels and from vector bars/lines
              scaled by axis ticks (vector.reconstruct_page)
            • chart crops re-read by PaddleOCR-VL-1.6 ("Chart Recognition:",
              serve/modal_chart_crops.py → output/chart_vlm/); HTML / pipe /
              bullet 'label: value' answers become extra candidate tables
```

Scores on the **full ParseBench set** (2,078 files; identical VLM output for both
rows, so every difference is the CPU-side reconciliation):

| | table | chart | content | formatting | layout | overall |
|---|---|---|---|---|---|---|
| LightOnOCR-3-0.8B alone (our provider) | 83.5 | 61.0 | 88.9 | 64.7 | 66.0 | 72.8 |
| docparse_hybrid_v0 (+ PP-DocLayout-V3 fusion) | 83.5 | 70.4 | 89.0 | 70.6 | 73.8 | 77.5 |
| docparse_hybrid_v0 (+ Docling heron fusion, legend/panel/pie charts) | 83.5 | 72.5 | 88.9 | 70.7 | 80.4 | 79.2 |
| docparse_hybrid_v0 (+ PaddleOCR-VL-1.6 chart-crop fallback) | 83.5 | **76.0** | 88.9 | 70.4 | **80.4** | **79.8** |
| LlamaParse Cost Effective (leaderboard) | 84.2 | 77.9 | 89.9 | 67.3 | 83.8 | 80.6 |
| best open-weight single model (leaderboard) | 86.4 | 64.9 | 88.8 | 71.7 | 74.3 | 77.2 |

Everything above the first row costs CPU only. Iterate without GPU:
`scripts/reuse_raw.sh lightonocr3_0_8b_grounding docparse_hybrid_v0` re-normalizes
saved VLM outputs under the hybrid pipeline and re-evaluates;
`scripts/chart_eval.py <module:function>` scores a chart reconstruction against the
chart rules on all 568 chart pages.

## Components (all Apache/MIT)

| Role | Model | ParseBench (single) |
|---|---|---|
| Structure + grounding | LightOnOCR-3-0.8B | 74.6 |
| Tables / JSON layout (optional) | Infinity-Parser2-Flash (2B) | 73.3 |
| Layout detector (optional) | PP-DocLayout-V3 | — |

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

# Infinity-Parser2-Flash (served name must be the HF id; SDK sends model=<id>)
scripts/deploy_infinity.sh --yes                        # prints the endpoint URL
#   → INFINITY_ENDPOINT_URL=<url>/v1 in .env
scripts/run_test.sh infinity_parser2_flash_modal

# PP-DocLayoutV3 layout detection (T4 by default; DOCPARSE_LAYOUT_GPU=cpu for CPU)
modal deploy serve/modal_paddle_layout.py               # prints the endpoint URL
#   → PADDLE_LAYOUT_ENDPOINT_URL=<url> in .env (no /v1)
scripts/run_test.sh ppdoclayout_v3_modal
```

Layout detector: `PaddlePaddle/PP-DocLayoutV3` — Apache-2.0 weights (HF model
card + PaddleOCR LICENSE), 25 classes, RT-DETR @ 800×800, ~70 ms/page on A100.
The server (`serve/modal_paddle_layout.py`) folds V3-only labels into the 20
PP-DocLayout classes parse-bench maps, keeps the original in `raw_label`, and
returns boxes in model reading order. `docparse.layout.detect_layout()` is the
client the composite pipeline uses.

Pipelines: `lightonocr3_0_8b_grounding`, `lightonocr3_0_8b_md`,
`infinity_parser2_flash_modal` (built-in `infinity_parser2_flash` settings +
env-driven endpoint + cold-start wait; uses the `infinity-parser2` PyPI SDK,
which needs poppler: `brew install poppler`).

`docparse` is `parse-bench` with this repo's providers/pipelines registered.

## Oracle analysis

Once several pipelines have results under `output/`, ask what a router could
achieve by combining them:

```bash
uv run oracle lightonocr3_0_8b_grounding infinity_parser2_flash_modal \
  [--output_dir output] [--costs costs.json] [--within 2] [--json]
```

Prints each pipeline's per-dimension mean (parse-bench's headline metric per
category: GTRM, chart `rule_pass_rate`, content faithfulness, semantic
formatting, layout element pass rate) and two oracles:

- **oracle-by-dimension** – best single pipeline per dimension (a static router);
- **oracle-by-example** – per-example max over pipelines (the ceiling for a
  perfect per-page router).

Overall is parse-bench's definition: the unweighted mean over categories.
With `--costs costs.json` (`{"pipeline": cents_per_page}`) it also lists the
cheapest pipeline within `--within` points of the per-dimension oracle.

## Layout

- `docparse/providers/` – parse-bench `Provider` implementations
- `docparse/pipelines.py` – named pipeline specs
- `serve/modal_vllm.py` – generic vLLM server on Modal (one app per model)
- `scripts/` – run helpers
