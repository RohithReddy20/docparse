# Layout & table structure — research notes (2026-10-12)

## What the ParseBench scorer rewards (evaluator source)

- Ontology: Canonical17 = Docling Heron classes; default view Core11, "basic" view collapses to
  Section/Text/Picture/Page-header/Page-footer/Table.
- Localization: per-GT argmax with `IoA(GT,Pred) ≥ 0.50` and `IoA(Pred,GT) ≥ 0.20`, not one-to-one.
  → a merged box passes for each covered element if ≤ 5× its area; a **split** passes only if one
  fragment alone covers ≥ 50% of the GT. Over-splitting is punished harder than mild merging.
- Attribution: IoA ≥ 0.30 candidates, token-F1 ≥ 0.80, merge-aware; skipped for captions/formulas.
- Tables (TRM): column keys = header-row texts joined top-to-bottom, colspan text repeated into each
  column, duplicates suffixed `_0, _1`. `rate adjustments_0/_1` = one `<th colspan=2>` with no sub-row.
  If GT has `<th>` and the prediction has none, the first data row is promoted when a cell fuzzy-matches
  a GT header ≥ 0.9; if GT has none, predicted headers are demoted.

## Docling rules worth copying ([paper](https://arxiv.org/html/2408.09869v5), `layout_postprocessor.py`)

- Per-class confidence 0.5 (0.45 for Section-header/Title/Code/Form/KV); TITLE → SECTION_HEADER.
- Span → cluster: intersection-over-span-area > 0.2, highest wins; clusters with no spans dropped
  (except Picture/Formula/Table); orphan spans become Text clusters.
- Overlaps: link if IoU > 0.8 or IoS > 0.8; List-item beats Text when areas within 20%; else keep
  the larger unless the smaller is > 0.05 more confident. Clusters with IoS > 0.8 inside a
  table/picture become its children.
- Bbox := union of its spans (up to 3 passes). Captions matched to the adjacent run of graphics by
  box gap (one caption per graphic). Cross-column paragraph merge via hyphen/lowercase regexes.
- Heading levels: numbering regexes > style (median cell height clustered at 5%, then weight, caps).
  Docling does **not** relabel Text↔Section from fonts; it trusts the detector.

## PaddleX / PP-StructureV3 fusion rules ([PaddleX](https://github.com/PaddlePaddle/PaddleX))

- Merge mode per class: "large" for paragraph_title/image, "union" for text/table/doc_title;
  `remove_overlap_blocks(0.65, smaller=True)`; if one of an overlapping pair is an image, drop the image.
- Text line ↔ block: intersection > 3 px both dims; a line spanning two blocks is split; text with no
  block becomes a new text block.
- Caption association: nearest graphic in block direction within 2 × line height.
- Stacked headings: adjacent paragraph_title blocks within 2 × line height chained but kept separate.
- Paragraph continuation only if end-gap < 10 px, start-gap < 10 px, previous block > 1 line.
- PP-DocLayoutV3 has no List-item class → lists from text-layer geometry (hanging indent / bullets).

## DocLayNet labeling guide (definition of the GT classes)

- Section-header: any heading except the title; own line; visible emphasis; usually < 1 line;
  consecutive headers of different levels are separate boxes. **Bold/caps text that starts a line
  and is followed by normal text on the same line is Text.** Emphasized intro statements after a
  heading are Text.
- List-item: hanging shape (first line starts left of continuation lines); each item = one box incl.
  bullet; indented (not hanging) numbered paragraphs are Text.
- Picture: graphic/plot/photo/logo; connected sub-figures = one box; scattered logos = one Picture;
  only if taller than 2 rows of normal text; tiny header/footer logos ignored.
- Human agreement: Section-header 83–84, Text 84–86, Picture 69–71, Title 60–72 mAP.

## Marker / Surya

- Line → block by max intersection; unassigned lines → nearest block (y weighted 5×); leftovers →
  new Text blocks. Picture boxes expanded ≤ 5% toward the nearest neighbour.
- Section levels: 1-D k-means (k=4) on line height.
- Text-layer tables: header band = lines above the first numeric-transition row; header spans
  stitched top-to-bottom per column (= TRM keys); score gate 0.75.

## Table structure

- TATR canonicalization: first row starting blank → header; extend header downward until every
  column has a non-spanning cell; a row with one non-blank cell in column 1 is a projected row
  header; merge header cells with blank cells beneath.
- UniTable (MIT, 30M/125M): TEDS 96.5 PubTabNet; use on crops where the text-layer grid score is low.

## Ranked ideas

1. Split merged boxes by text-layer geometry: hanging indent / bullet regex → List-item per item;
   gap > 1.3–1.5 × line height; font size/bold change with ≥ 1 line each side; block > 5× smallest
   unit. Snap each piece to its spans.
2. Text vs Section relabel: DocLayNet rules + a small model (logistic/GBM) over span features
   (own line, n_lines ≤ 3, no period, size ratio, bold/caps fraction, width, gap after, precedes a
   graphic) with detector label + VLM label as inputs, fit on the 14k GT elements.
3. Picture recovery: `get_image_info()` ∪ `cluster_drawings()` ∪ detector image/chart/seal boxes;
   height > 2 lines; merge touching; absorb spans with IoS > 0.8 (deletes spurious KPI Text boxes).
4. Table headers: TATR canonicalization on the text-layer grid; `<th colspan>` for spanned headers
   with no sub-row; row 1 `<th>` when non-numeric and rows 2+ numeric.
5. Docling-style three-way fusion (VLM boxes + detector boxes + spans) with overlap resolution.
6. Split multi-table boxes on whitespace gaps > 2 × line height / column-interval changes / ruling clusters.
7. Caption linkage for consistent relabeling.
8. Reading order: detector decoder order primary; Docling up/down maps fallback.
