# Section/Text reranker

The reranker changes layout labels using PDF text, font, geometry and detector
confidence. It is experimental and disabled unless `DOCPARSE_SECTION_TEXT_MODEL`
points to a compatible model JSON. It changes only Text/Section-header detector
labels and preserves titles, list items, markdown, scanned pages, visual
classes and explicit VLM headings. Font-inferred headings may be reconsidered;
the original VLM label is retained internally to distinguish their origin.

The published full-set score is **80.11 overall / 81.01 layout** without the
reranker. An earlier annotation-box-only forest achieved 96.74% accuracy on
held-out DocLayNet boxes but regressed its paired full ParseBench control from
79.95 to 79.91 overall, and 80.41 to 80.20 layout. It was rejected. DocLayNet box
accuracy alone is insufficient evidence of a ParseBench improvement.

## Training

`scripts/doclaynet_heron_features.py` runs frozen Heron detections on DocLayNet
images. It matches predictions to the highest-IoU annotation at IoU >= 0.5,
restricts both detector and annotation classes to Text/Section-header, excludes
unmatched boxes, and extracts 25 features from
the predicted boxes and native PDFs. It samples 5,000 training, 1,500 validation
and 1,000 test pages from the official splits, rejecting document overlap.
ParseBench documents and labels are not used for training or calibration.

`scripts/train_section_text.py` trains a 120-tree forest and exports JSON trees
for inference without scikit-learn. Promotion and demotion gates are selected
on validation examples whose baseline label would change: at least 50 examples
and 98.5% precision are required. A direction is disabled when no gate qualifies.
Held-out test metrics include the net number of corrected baseline labels.
The exported evaluator must match scikit-learn probabilities within 1e-10,
including its float32 input conversion.

Create a private Kaggle script using a DocLayNet input that contains the core
COCO/PNG and extra PDF directories:

```bash
uv run python scripts/kaggle_section_text.py \
  --output /tmp/docparse-section-training \
  --kernel USER/section-text-training --enable-gpu
kaggle kernels push -p /tmp/docparse-section-training \
  -t 7200 --accelerator NvidiaTeslaT4
```

For a first run, add `--smoke` when creating the bundle and use `-t 1200`
when pushing it. This exercises detection, feature extraction, training and
export on 128/64/64 pages; its model is for checking the workflow, not promotion.
Only start the full run after this completes successfully.
If a CPU notebook has prepared a smaller DocLayNet input with `COCO`, `PDF` and
`PNG` directories, pass `--kernel-source USER/doclaynet-subset` when creating
the training bundle. This mounts that notebook's output instead of the full
dataset. Keep the official splits and reject document overlap when preparing it.

The script uses both T4s and keeps dataset/model downloads on Kaggle. Training
requires NumPy, scikit-learn, PyTorch, Transformers, PyMuPDF, Pillow and RapidFuzz;
the Kaggle bootstrap installs missing PyMuPDF, Transformers and RapidFuzz.
Completed split features are retained under `/kaggle/working/section_text_features`.
`section_text_status.json`, `features.log`, `training.log` and
`section_text_features/progress.json` record stage and batch progress. The
feature subprocess is limited to 90 minutes (10 minutes in smoke mode), and
training to five minutes. These limits begin after Kaggle starts the script;
they cannot bound platform provisioning. A run stuck without its startup status
file should be checked in Kaggle's version history and cancelled if necessary.
Download only the model/report archive:

```bash
kaggle kernels output USER/section-text-training \
  -p output_reranker/artifacts --file-pattern '^section_text_artifacts\.zip$'
```

The archive contains `section_text_model.json` and `section_text_report.json`.
Check its reported SHA-256, split provenance, export parity and selected gates
before evaluation. Annotation-box-only version 1 models are incompatible with
the detector-aware version 2 runtime.

## Full-set evaluation

Use an isolated output directory containing all 2,078 cached raw files and the
pipeline metadata. With the dataset, layout detections and chart crops local:

```bash
DOCPARSE_SECTION_TEXT_MODEL=output_reranker/artifacts/section_text_model.json \
  UV_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  uv run docparse inference renormalize output_reranker/docparse_hybrid_v0 \
    --pipeline_name docparse_hybrid_v0 --force
UV_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  uv run docparse run docparse_hybrid_v0 --skip_inference --input_dir data \
    --output_dir output_reranker --open_report False
uv run oracle docparse_hybrid_v0 --output_dir output_reranker
```

Verify that all examples succeeded and that markdown, text, boxes and order are
unchanged from the paired control. Promote only after full-set scores show a
gain; the 12-document test split is insufficient. Leaving the environment
variable unset or empty retains the published parser.
