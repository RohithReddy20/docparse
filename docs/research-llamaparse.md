# LlamaParse internals & ParseBench grading — research notes (2026-10-12)

## How LlamaParse works (public + inferable)

- Text layer first, OCR second; "segment before you read": a layout-detection CV model splits
  the page, each component is routed to OCR / LLM / VLM, with validation loops
  ([blog](https://www.llamaindex.ai/blog/llm-ocr), [HN 2024](https://news.ycombinator.com/item?id=39443972)).
  LiteParse = the open-sourced text-layer core (spatial grid, bboxes, no models).
- Layout labels are visible in ParseBench's own code (`layout_label_mapping.py`): "v2" = the
  PP-DocLayout vocabulary (`doc_title, paragraph_title, text, abstract, aside_text, header, footer,
  footnote, image, chart, seal, figure_title, table, formula, …`); "v3" = Docling-Heron
  Canonical17 via an RT-DETRv2/ResNet-50 detector + a 16-class figure classifier
  (`layout_v3.py`). Cost Effective's visual grounding jumped 58.6 → 83.8 with the v2→v3 switch,
  i.e. a 17-class detector at paragraph/list-item granularity. **The GT boxes were seeded by the same
  detector family (PP-DocLayout → Heron).**
- Tiers: Fast = text layer only. Cost Effective = layered text + single LLM pass per page.
  Agentic = page text + screenshot to LLM/VLM in an agentic loop with tools, iterative refinement
  (historically Gemini Flash/Pro). Agentic Plus = different multi-pass pipeline, chart parsing on,
  verification rounds for bboxes
  ([modes](https://developers.llamaindex.ai/llamaparse/parse/v1/presets_and_modes/advance_parsing_modes/index.md),
  [tiers](https://developers.llamaindex.ai/python/cloud/llamaparse/tiers/)).
- Headers/footers: VLM prompt asks for `<page_header>/<page_footer>/<page_number>` tags; post-processing
  lifts them into structured fields.
- Charts become a `table` item: header row, one column per series, rows keyed by category
  ([docs](https://developers.llamaindex.ai/llamaparse/parse/features/charts-and-figures/index.md)).
- Confidence = char-count ratio + bag-of-letters + bag-of-words between page and markdown.

## ParseBench grading facts that change strategy (from the evaluator source)

**Visual grounding** (`evaluators/layoutdet.py`, ontology "basic", 6 classes):
- GT classes: Text 9223, Section 3001 (= Title + Section-header), Picture 2286, Page-header 531,
  Page-footer 415, Table 232. Classification is strict.
- Granularity: every list item, caption, footnote, key-value line, TOC row, form field is its own
  Text element. GT `source_label`s are PP-DocLayout names (`paragraph_title` 2485, …).
- Pictures: ~45% are icons/logos (icon 965, logo 270); captions on pictures skip attribution → a box
  with the right class is nearly free.
- Localization passes when `IoA(GT,Pred) ≥ 0.50` AND `IoA(Pred,GT) ≥ 0.20` → a merged box
  passes for every element it covers as long as it is ≤ 5× that element's area.
- Attribution: token F1 ≥ 0.80, **merge-aware** (tokens of other overlapping GT elements are
  subtracted) → merging same-sized neighbours costs nothing; it fails only when the box exceeds
  5× a small element, when a Section is merged into a Text box (class), or on reading order.
- Page-header/footer: grouped bands; union must cover ≥80% of the GT x-span and ≥50% of y.

**Formatting** (`rules_formatting.py`): `is_title` passes for an ATX heading, `<h1-6>`, or a
standalone bold line; `title_hierarchy_percent` only needs parent shallower than child (bold = level 7).
No penalty for extra headings in the current rule set. Styling uses F-β(0.5): false bold/strike is
punished harder than missed.

**Charts** (`rules_chart.py`): per point labels are typically [series, category] (+ title in 15%);
label must be a substring (`partial_ratio`) of a `<th>` in the value's row-header chain, column-header
chain, or the same row. Flat combined headers ("Primary 2015") satisfy both labels. All-`<td>` tables
count only when the match is unique. Labels can also be satisfied from context *before* the table:
heading / bold / `<caption>` (ratio ≥ 0.6). Orientation-agnostic. A leading single-cell
`<td colspan>` row is treated as a table title.

**Tables** (GTRM = mean of GriTS and TableRecordMatch): GT uses `<th>` in 473/503 tables but
`<thead>` in only 67; 191 tables put `<th>` in `<tbody>` as a row-header first column; 246 use
colspan, 99 rowspan. TRM keys records by the `<th>` chain above each cell; dropped/transposed
headers are heavily penalised, column reordering is free.

**Content**: `is_header`/`is_footer` need the text in `<page_header>`/`<page_footer>` tags or the
structured fields; the order rule compares first occurrence of "before" with last of "after".

## Ranked ideas (from the report)

1. Split VLM blocks only where the eval punishes it: never let a block absorb a detector
   `paragraph_title / list-item / caption / footnote / key-value` box; split when
   `area(block)/area(det) > ~4`; otherwise keep merges.
2. Detector labels are the Section-vs-Text truth (`doc_title/paragraph_title → Section`).
3. Emit a Picture box for every detector image + every vector/raster cluster with no text inside.
4. Running header/footer bands from the text layer + `<page_header>` tags in markdown.
5. Chart tables: categories as `<th>` row headers, series as a `<th>` header row, flat combined
   headers for grouped/stacked, title as heading/bold/caption before the table.
6. Legend/axis association from vector geometry (fill colour ↔ swatch).
7. Table `<th>` promotion: rows above the first numeric-majority row; first column when
   non-numeric and the rest numeric; colspan/rowspan from text-layer geometry; no `<thead>` needed.
8. Always emit `#` headings for Sections (levels by font-size rank); never bold text that is not bold.
9. Column-aware reading order from the text layer (LiteParse-style grid).
10. Verification pass on flagged crops instead of a bigger VLM.
