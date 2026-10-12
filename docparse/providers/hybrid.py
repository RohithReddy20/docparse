"""docparse hybrid provider: LightOnOCR-3 grounding for structure, PDF text layer for
characters and styling. Inference is identical to ``lightonocr3`` (so its saved raw
outputs can be re-normalized under this pipeline with no GPU); only ``normalize``
differs.
"""

from __future__ import annotations

import logging

from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceResult, RawInferenceResult

from docparse import textlayer
from docparse.hybrid import reconcile_page
from docparse.providers.lightonocr3 import (
    _LABEL_TO_CANONICAL_KEY,
    LightOnOCR3Provider,
    blocks_to_markdown,
    build_layout_page,
    page_furniture,
    parse_grounding,
)

logger = logging.getLogger(__name__)


@register_provider("docparse_hybrid")
class DocparseHybridProvider(LightOnOCR3Provider):
    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        raw = dict(raw_result.raw_output)
        if "pages" in raw:
            raw["page_results"] = raw.pop("pages")
        try:
            layers = textlayer.extract(raw_result.request.source_file_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("text layer extraction failed for %s: %s", raw_result.request.source_file_path, e)
            layers = []

        pages: list[PageIR] = []
        layout_pages: list[ParseLayoutPageIR] = []
        for p in raw["page_results"]:
            idx = p["page_index"]
            blocks = parse_grounding(p["raw_text"])
            tl = layers[idx] if idx < len(layers) else None
            blocks = reconcile_page(blocks, tl)
            md = blocks_to_markdown(blocks)
            items = [
                {"bbox": b["bbox"], "label": _LABEL_TO_CANONICAL_KEY.get(b["label"], "text"), "text": b["text"]}
                for b in blocks
                if b["bbox"] is not None
            ]
            lp = build_layout_page(items, p["width"], p["height"], md, idx + 1)
            furniture = page_furniture(blocks)
            lp.page_header_markdown = furniture["header"]
            lp.page_footer_markdown = furniture["footer"]
            lp.printed_page_number = furniture["page_number"]
            layout_pages.append(lp)
            pages.append(PageIR(page_index=idx, markdown=md))

        output = ParseOutput(
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown="\n\n".join(pg.markdown for pg in pages),
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
