# Native-style alignment

The default parser transfers native PDF styling through one shared monotone
character alignment. This replaces independent fuzzy searches for each styled
string, which could select the wrong occurrence of a repeated word.

`docparse/textalign.py` retains original character offsets while normalizing
whitespace, case and ligatures for matching. `docparse/styles.py` maps native
style intervals to those offsets without replacing VLM characters. It preserves
existing emphasis, splits additions at markup boundaries and protects code and
math. Superscripts keep exact character bounds, including single characters;
whole native words retain whole-word styling when the target spelling differs
by a suffix. Ambiguous omitted repetitions are left unchanged.

The implementation requires at least 0.6 global alignment similarity and 85%
character coverage of a style interval, including matched endpoints. Inputs
over 12,000 characters retain their original text. These conservative limits
were exercised with authored fixtures, not fitted to ParseBench labels.
Existing incorrect VLM styling is not removed. Subscript inference and a
document-level heading model are not included.

Set `DOCPARSE_STYLE_ALIGNMENT=legacy` to use the previous transfer method.
The bundled Section/Text reranker remains independent of this switch.

## Full-set validation

All 2,078 cached raw outputs were renormalized and all five categories were
evaluated with the installed ParseBench 1.0.4 evaluator. Paired IDs and headline
metric eligibility matched; there were no inference or evaluation errors.

| Dimension | Previous | Aligned styles | Improved examples | Regressed examples |
|---|---:|---:|---:|---:|
| Tables | 83.4612 | 83.4612 | 0 | 0 |
| Charts | 76.7924 | 76.7924 | 0 | 0 |
| Content | 88.8465 | 88.9221 | 25 | 15 |
| Formatting | 70.4335 | 71.1807 | 22 | 8 |
| Layout | 81.3316 | 81.3385 | 3 | 2 |
| Overall | 80.1730 | 80.3390 | | |

Overall is the unweighted mean of the five category headline means. Formatting
has 456 eligible examples out of 476 evaluations; layout has 436 out of 500.
`scripts/paired_report.py` checks both evaluated IDs and metric eligibility
rather than silently changing those denominators.

The caption-preservation fix in chart snapping was also checked independently:
all 2,078 normalized outputs were unchanged, so it adds no benchmark gain.

## Rejected omission recovery

A separate experiment inserted short unmatched native phrases between exact,
unique word anchors at at least 97% overall text agreement. Only two outputs
changed: an image credit and a review watermark entered unrelated body prose.
Full-set content declined by 0.0130 points and overall by 0.0026 points relative
to aligned styles. The recovery code was removed from the parser.

High text agreement and geometric coverage do not establish token ownership.
Recovery needs local source-span geometry, page roles and text ownership before
adding visible text that the VLM omitted.
