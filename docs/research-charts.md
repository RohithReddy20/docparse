# Chart extraction — research notes (2026-10-12)

Our state: chart 70.5 (LlamaParse CE 77.9). Failures: ~1,200 "value found, labels not
associated", ~780 "value not found". 89% of chart pages are vector PDFs.

## Models worth trying on chart crops (GPU)

| Model | Size / license | Why |
|---|---|---|
| Qwen3-VL-8B/4B | Apache-2.0 | Best open OOD chart-to-table: 52.9 WB-ChartExtract (next open ~30–35); 91.4 ChartQA RMS-F1 ([self-ensembling paper](https://arxiv.org/html/2605.27298), [EpiCurveBench](https://arxiv.org/html/2605.27195)) |
| PaddleOCR-VL-1.6 | 0.9B, Apache-2.0 | Native chart→table, 91.7 RMS-F1 in-house; cheap second opinion ([paper](https://arxiv.org/html/2606.03264v1)) |
| PP-Chart2Table | 0.58B | Header row carries series names; "comparable to 7B VLMs" ([docs](https://www.paddleocr.ai/v3.4.1/en/version3.x/module_usage/chart_parsing.html)) |
| OneChart | 0.2B, Apache-2.0 | Outputs `{title, x_axis, y_axis, value:{series→{category→number}}}` + reliability score; strong when no data labels ([paper](https://arxiv.org/html/2404.09987v2)) |

Skip: TinyChart/DePlot/UniChart/ChartGemma (collapse out of distribution).

## Deterministic recipes for vector charts (sources: [Mystique](https://arxiv.org/html/2307.13567v2), [ChartDetective](https://github.com/m-damien/ChartDetective), [Reducto](https://reducto.ai/blog/reducto-chart-extraction))

- **Legend swatch ↔ series** (our biggest label failure): legend = run of ≥2 small same-size filled rects / short stroked segments evenly spaced with a word to the right; match exact fill RGB to bars, stroke RGB+dash+width to lines; fallback ordinal order (legend order == left→right in a group / bottom→top in a stack).
- **Stacked vs grouped**: stack = same x-interval (IoU>0.8), vertically touching; emit per-segment values. Grid = equal-width adjacent rects; series by colour.
- **Lines/areas**: polylines by identical stroke; sample y at each x tick (interpolate); areas = upper boundary; stacked areas subtract.
- **Pies**: slice = closed path with two radii + Bézier arc chain; share = sweep/360; label by legend colour or leader-line text on the bisector.
- **Small multiples**: panel = repeated axis frame; nearest larger text above = panel title; one table per panel with `<caption>` = title – panel title, plus a merged table with "Panel – Series" headers.
- **Dual axes**: right-side tick column → second scale; assign series by range.
- **Confidence gate**: emit vector values only when axis fit R²≈1 and swatch match passes; agreement with VLM within 10% as reliability signal.

## Prompt/format recipes

1. Inject vector-derived series names and categories into the VLM prompt as fixed headers.
2. Header rule (from EpiCurveBench): single series → y-axis label; multiple → legend labels; copy headers exactly; `nan` for missing; chart title in `<caption>`.
3. Two-pass: structure (`title, panels, legend order, categories, units`) then values.
4. Per-panel crops with the panel title stated.
5. Multi-level headers: flatten "Region – Category" AND emit the transposed table (matcher accepts row or column headers).
6. Self-ensembling (sample N tables, per-cell median, prune rare labels): +1.7 to +8 RMS-F1; treat the vector table as one sample.

## How commercial parsers do it

- LlamaParse: `specialized_chart_parsing` tiers; chart emitted as a table item, series as columns, categories as rows, units in headers ([docs](https://developers.llamaindex.ai/llamaparse/parse/features/charts-and-figures/)).
- Reducto: multi-stage component detectors (axes, ticks, legends, series), values tick-aligned, bars as (bottom, top); falls back to VLM for pies/horizontal bars/multi-chart ([docs](https://docs.reducto.ai/configs/parse/chart-extraction.md)).
- Azure Content Understanding: chart.js JSON (`labels[]`, `datasets[{label,data}]`) — good internal schema.
