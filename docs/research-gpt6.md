# External deep-research report (GPT, 2026-10-13) — key points we act on

Full text was pasted in chat; this records what changed our plan.

1. **PyMuPDF is AGPL / commercial** — fine for benchmarking, not for a sellable product.
   A permissive backend port (pypdfium2 BSD/Apache for text + char boxes + fonts,
   pdfplumber/pdfminer.six MIT for drawings/colors) is needed before commercial use.
   Tracked as a TODO; does not block the benchmark work.
2. Evaluator notes: chart rules accept Markdown and HTML tables; label matching uses
   candidate-local row evidence + authored header chains. "Every legend string in every
   caption" passes the matcher but is not a correct association — keep but prefer real
   series headers.
3. Docling detector comparison (paper 2509.11720): heron 69.9 raw DocLayNet mAP @ 0.64 s/page
   CPU; egret-medium 68.6 @ 0.33 s/page (cheaper for production); heron-101 not better.
   We measured heron: layout 74.2 → 80.4. Docling's own post-processor *lowers* AP — adopt
   rules selectively.
4. Chart-to-table OOD numbers: TinyChart 95 ChartQA → 28.6 WB-ChartExtract; DePlot 23;
   OneChart 26.5; Qwen3-VL **235B** 52.9 (not the 4B/8B). PaddleOCR-VL-1.6 (`Chart
   Recognition:` prompt) and PP-Chart2Table are the permissive crop-level fallbacks to try;
   OneChart's weights have research-only language; TinyChart license unverified.
5. Ranked plan (their estimate of gain/day): chart IR + candidate-local headers → panel
   separation → legend↔series assignment (colour, dash, marker, hatch; global bipartite
   assignment; reject when best cost is not ≥20% better than runner-up) → conservative
   granularity repair → selective PaddleOCR-VL-1.6 chart-crop fallback → axis calibration
   (≥3 ticks, RANSAC residual max(1pt, 0.5% span), log only with evidence, dual axes
   separate) → character-provenance attribution after splits → picture identity separate
   from header/footer role → DocLayNet-trained Section/Text reranker (promote only at
   P≥0.9, demote only at P≤0.1).
6. Their milestone math: charts 80 + grounding 85 with our other dims fixed → ≈81.6 overall
   (beats Cost Effective); matching Agentic's chart/grounding alone → ≈83.2.
