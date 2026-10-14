# docparse

Open, multi-step document parser targeting [ParseBench](https://github.com/run-llama/ParseBench).
Goal: beat LlamaParse Cost Effective (80.6 overall @ 0.375¢/page) and approach
Agentic (87.0 @ 1.25¢) using only permissively licensed open models, with cost
measured from real GPU-seconds.

## Method: text layer + small VLM

Small pixel-only VLMs miss evidence already present in the source PDF.
93% of ParseBench pages are born-digital
(`scripts/analyze_text_layer.py`): the PDF text layer carries exact characters,
bold/underline/strikethrough/highlight, font sizes (heading hierarchy), printed
chart data labels, and the vector bars of 89% of chart pages. `docparse_hybrid_v0`
combines native styling, text attribution and chart values with a 0.8B VLM for
text and structure:

```
PDF page ─┬─ text layer (PyMuPDF, CPU)  : spans + styling + drawings
          └─ LightOnOCR-3-0.8B grounding: block order, labels, bboxes, table/chart HTML
                     │
          reconcile  (docparse/hybrid.py)
            • VLM characters retained; styling aligned from native spans,
              explicit heading levels preserved, missing levels from font context
            • merged blocks split on paragraph gaps / list items / bold lead lines
            • lines the VLM skipped recovered as blocks; header/footer/page-number
              lifted into structured fields
            • invisible OCR layers keep the VLM output whole; visible native text
              above background images supplies styling without orphan insertion
          layout     (docparse/fusion.py) — Docling heron (RT-DETRv2, the 17
            Canonical17 classes ParseBench's GT uses; serve/modal_heron_dump.py)
            boxes are the layout elements, complete native words attributed
            locally; VLM blocks fill what the detector misses
          charts     (docparse/charts/)
            • VLM table snapped to printed data labels (textlabels.snap_table)
            • extra candidate tables from text labels and from vector bars/lines
              scaled by axis ticks (vector.reconstruct_page); filled area bands
              sampled at category positions, with legend colours naming series
            • chart crops re-read by PaddleOCR-VL-1.6 ("Chart Recognition:",
              serve/modal_chart_crops.py → output/chart_vlm/); HTML / pipe /
              bullet 'label: value' answers become extra candidate tables
```

Scores on the **full ParseBench set** (2,078 files; identical base VLM outputs
across variants; the hybrid also uses saved layout detections and chart crops):

| | table | chart | content | formatting | layout | overall |
|---|---|---|---|---|---|---|
| LightOnOCR-3-0.8B alone (our provider) | 83.5 | 61.0 | 88.9 | 64.7 | 66.0 | 72.8 |
| docparse_hybrid_v0 (+ PP-DocLayout-V3 fusion) | 83.5 | 70.4 | 89.0 | 70.6 | 73.8 | 77.5 |
| docparse_hybrid_v0 (+ Docling heron fusion, legend/panel/pie charts) | 83.5 | 72.5 | 88.9 | 70.7 | 80.4 | 79.2 |
| docparse_hybrid_v0 (+ PaddleOCR-VL-1.6 crops, vector bars/lines/areas, chart table labels, complete Heron class mapping) | 83.46 | 76.79 | 88.84 | 70.43 | 81.01 | 80.11 |
| docparse_hybrid_v0 (+ calibrated DocLayNet Section/Text reranker; 2026-10-13) | 83.46 | 76.79 | 88.85 | 70.43 | 81.33 | 80.17 |
| docparse_hybrid_v0 (+ shared native-style alignment; 2026-10-13) | 83.46 | 76.79 | **88.92** | 71.18 | 81.34 | 80.34 |
| docparse_hybrid_v0 (+ heading/native-text preservation and word attribution; 2026-10-14) | 83.46 | **76.79** | 88.91 | **72.29** | **81.54** | **80.60** |
| LlamaParse Cost Effective (leaderboard) | 84.2 | 77.9 | 89.9 | 67.3 | 83.8 | 80.6 |
| best open-weight single model (leaderboard) | 86.4 | 64.9 | 88.8 | 71.7 | 74.3 | 77.2 |

The latest exact overall is **80.597947**, still **0.012053** below the
80.61 Cost Effective reference. Rounded 80.60 is not a claim of surpassing it.
The [native-preservation validation](docs/native-preservation.md) records paired
gains and regressions; production cost remains to be measured.

Re-normalizing saved VLM, layout, and chart-crop outputs uses CPU only. The full-set
rescore above reused all 2,078 raw outputs; initial model inference uses a GPU.
With the cached outputs and dataset already local, rescore without GPU or downloads:

```bash
uv run docparse inference renormalize output_heron/docparse_hybrid_v0 --pipeline_name docparse_hybrid_v0 --force
uv run docparse run docparse_hybrid_v0 --skip_inference --input_dir data --output_dir output_heron --open_report False
uv run oracle docparse_hybrid_v0 --output_dir output_heron
```

Compare a candidate with the same evaluator and cached inputs using paired
category reports. This rejects missing files, changed metric coverage, failed
examples and mismatched evaluator versions:

```bash
uv run python scripts/paired_report.py output_heron/docparse_hybrid_v0 output_candidate/docparse_hybrid_v0 --out output_candidate/paired_delta.json
```

`scripts/chart_eval.py <module:function>` scores a chart reconstruction against the
chart rules on all 568 chart pages.

The [DocLayNet Section/Text reranker](docs/section-text-reranker.md) uses a bundled
forest calibrated on validation data. It is enabled by default after full-set
validation. Set `DOCPARSE_SECTION_TEXT_MODEL=` to disable it for a paired control.

[Native-style alignment](docs/style-alignment.md) uses one monotone character
mapping for all styles, retains occurrence identity and supports single-character
superscripts. It is enabled by default; set `DOCPARSE_STYLE_ALIGNMENT=legacy` for
the previous style-transfer control. The full-set gain includes regressions on
eight formatting examples, alongside improvements on 22.

## Model components

| Role | Model | ParseBench (single) |
|---|---|---|
| Structure + grounding | LightOnOCR-3-0.8B | 74.6 |
| Tables / JSON layout (optional) | Infinity-Parser2-Flash (2B) | 73.3 |
| Layout detector (optional) | PP-DocLayout-V3 | — |

Avoided: MinerU, KDL/florin/rakedoc-nano (AGPL), Chandra (non-compete license).

## Compute

Text-layer extraction, fusion, vector reconstruction and rescoring cached outputs
use CPU only. Initial VLM, layout-detector and chart-crop inference use GPUs.
`serve/modal_vllm.py` runs the VLM on one L4 container (2-minute idle scale-down,
1-hour hard timeout); a full ParseBench VLM run is about an L4-hour.

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
