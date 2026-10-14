# Native text preservation

The hybrid parser keeps the VLM's characters and uses native PDF evidence for
styling and grounding. The following changes repair losses after inference.

## Explicit heading levels

A single explicit Markdown heading keeps its authored level. Native font ranking
supplies a level when the VLM supplied none. Font ranking alone previously changed
an explicit `###` to `#` when a page had no larger font, losing document hierarchy.
Multiple headings inside one title block still use the existing splitting path.

`DOCPARSE_HEADING_LEVEL_POLICY=legacy` restores the font-ranking control.

## Visible text above images

A large image is evidence of a scan, but a background image can coexist with real
native text. The parser checks drawing order, text rendering mode and opacity.
Native text must contain at least 80 non-whitespace characters; every meaningful
text trace must be visible, have opacity at least 0.9 and remain uncovered by
later images. Invisible OCR and GlyphlessFont layers remain excluded. Mixed
visible and invisible layers retain the scan policy.

Newly enabled image pages supply styling and grounding, but do not insert
unmatched native lines into Markdown. An earlier variant duplicated table-of-
contents fragments: visibility establishes that text is drawn, not which output
region owns it. Omission recovery needs independent token ownership.

`DOCPARSE_NATIVE_IMAGE_TEXT=legacy` restores the large-image exclusion.
`DOCPARSE_NATIVE_IMAGE_ORPHANS=legacy` restores orphan insertion on these pages.

## Adjacent bold boundaries

When native bold extends an existing Markdown bold run, the serializer joins
only the touching generated and existing boundaries. For example,
`**Ques**: An example question.` becomes `**Ques: An example question.**`,
instead of the invalid `**Ques****: An example question.**`. Existing combined
emphasis, code and math remain protected; target characters remain unchanged.

`DOCPARSE_STYLE_JOIN_ADJACENT=0` restores separate bold wrappers.

## Local layout attribution

Native character rectangles form complete words across font spans. The word
rectangles are cached once per page. Grounding text is selected locally, rather
than copying a whole native line into neighboring layout boxes. Words retain
their source spelling and are not truncated at detector boundaries.

Rotated pages retain legacy attribution until their coordinate transforms are
handled explicitly. Text without character geometry uses span rectangles.

Word attribution is enabled by default and requires at least 50% word-area
overlap, using the existing 0.4% page padding. `DOCPARSE_LAYOUT_TEXT=legacy`
restores line attribution. A vertical-centre alternative was tested separately;
it changed 30 outputs but left all five headline means unchanged and was removed.

## Validation

Experiments reuse the same 2,078 raw outputs, Heron detections and chart crops,
with ParseBench 1.0.4. All five category reports must have paired evaluated IDs,
headline eligibility and complete raw-cache coverage. No benchmark annotations
are used by the native extraction, attribution or routing code.

The source fixtures cover heading levels, hidden and occluded OCR, background
images, unowned text, adjoining bold containers, font changes inside words,
multi-character glyphs, short detector boxes and rotated-page fallback.

| Dimension | Previous default | Native preservation | Improved examples | Regressed examples |
|---|---:|---:|---:|---:|
| Tables | 83.461169 | 83.461238 | 1 | 1 |
| Charts | 76.792377 | 76.792377 | 0 | 0 |
| Content | 88.922088 | 88.907104 | 4 | 7 |
| Formatting | 71.180703 | 72.286532 | 31 | 11 |
| Layout | 81.338546 | 81.542486 | 19 | 7 |
| Overall | 80.338977 | 80.597947 | | |

The exact overall gain is 0.258971 points. Formatting has 456 eligible examples
out of 476 evaluations; layout has 436 out of 500. The overall is the unweighted
mean of the five category headlines. All 2,078 unique files have successful
evaluations and matching raw-cache coverage. The result remains 0.012053 below
the 80.61 Cost Effective reference; rounding is not a surpassing claim.

Intermediate full-set results were 80.406396 with explicit heading levels,
80.464248 with visible native text, and 80.580690 after bold-boundary repair and
disabling orphan insertion on newly enabled image pages. Local word attribution
adds the remaining gain. These stages do not imply that each individual change
has an additive effect outside the measured configuration.

Panel-owned chart integration was tested separately and rejected: two chart-only
variants scored 72.58 and 73.64 against 76.79. Explicit titles alone did not retain
the required panel, series and unit associations. A missing-header correctness
patch scored unchanged. A separate semantic/grounding Markdown projection changed
115 outputs with zero grounding changes but slightly decreased the full-set
score; it was also removed.
