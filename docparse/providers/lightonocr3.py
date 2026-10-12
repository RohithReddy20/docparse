"""LightOnOCR-3 parse provider (lightonai/LightOnOCR-3-0.8B, -4B; Apache-2.0).

Talks to a vLLM OpenAI-compatible endpoint (see serve/modal_vllm.py). Two prompt
modes, per the model card:

- ``grounding``: each block is preceded by a placeholder line
  ``![label](x1,y1,x2,y2)`` with 0-1000 page coordinates; ``label+`` marks a
  continuation block. Charts/tables come back as HTML ``<table>``.
- transcription (empty prompt): plain markdown, no bboxes.

Config (env fallback in parentheses):
  endpoint_url  vLLM base URL ending in /v1   (LIGHTONOCR_ENDPOINT_URL)
  model         served model name             (LIGHTONOCR_MODEL, default lightonocr3)
  prompt_mode   "grounding" | "transcribe"    (default grounding)
  dpi           render DPI                    (default 400, model card)
  max_edge      longest image edge in px      (default 2048, model card)
  max_tokens / temperature / top_p / timeout
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from datetime import datetime
from typing import Any

import pymupdf as fitz
from openai import OpenAI
from PIL import Image

from parse_bench.inference.providers.parse._layout_utils import build_layout_pages
from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"^!\[([A-Za-z_\-]+?)(\+?)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$")

# LightOnOCR label -> key understood by parse-bench's LABEL_MAP (Canonical17).
_LABEL_TO_CANONICAL_KEY: dict[str, str] = {
    "title": "title",
    "section_header": "section-header",
    "section-header": "section-header",
    "header": "page-header",
    "page_header": "page-header",
    "text": "text",
    "list": "list-item",
    "list_item": "list-item",
    "image": "picture",
    "figure": "picture",
    "picture": "picture",
    "chart": "picture",
    "table": "table",
    "formula": "formula",
    "caption": "caption",
    "footnote": "footnote",
    "footer": "page-footer",
    "page_footer": "page-footer",
}


def render_pdf_pages(pdf_path: str, dpi: int, max_edge: int) -> list[Image.Image]:
    """Render every page at ``dpi`` then shrink so the longest edge is ``max_edge``."""
    doc = fitz.open(pdf_path)
    images: list[Image.Image] = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        longest = max(img.size)
        if longest > max_edge:
            scale = max_edge / longest
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        images.append(img)
    doc.close()
    return images


def parse_grounding(text: str) -> list[dict[str, Any]]:
    """Split grounding-mode output into ``{label, bbox, text, continuation}`` blocks."""
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    buf: list[str] = []

    def flush() -> None:
        if current is not None:
            current["text"] = "\n".join(buf).strip()
            blocks.append(current)

    for line in text.splitlines():
        m = _PLACEHOLDER_RE.match(line.strip())
        if m:
            flush()
            label, cont, x1, y1, x2, y2 = m.groups()
            current = {
                "label": label.lower(),
                "continuation": cont == "+",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            }
            buf = []
        elif current is not None:
            buf.append(line)
        elif line.strip():
            # Text before the first placeholder: keep it as an unlocated block.
            current = {"label": "text", "continuation": False, "bbox": None}
            buf = [line]
    flush()
    return blocks


def blocks_to_markdown(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for b in blocks:
        t = b["text"]
        if not t:
            continue
        label = b["label"]
        if label == "title" and not t.lstrip().startswith("#"):
            t = f"# {t}"
        elif label in ("section_header", "section-header") and not t.lstrip().startswith("#"):
            t = f"## {t}"
        elif label == "formula" and not t.lstrip().startswith("$"):
            t = f"$$\n{t}\n$$"
        parts.append(t)
    return "\n\n".join(parts)


@register_provider("lightonocr3")
class LightOnOCR3Provider(Provider):
    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None) -> None:
        super().__init__(provider_name, base_config)
        cfg = self.base_config
        endpoint = cfg.get("endpoint_url") or os.getenv("LIGHTONOCR_ENDPOINT_URL")
        if not endpoint:
            raise ProviderConfigError("Set LIGHTONOCR_ENDPOINT_URL (vLLM base URL ending in /v1).")
        self._client = OpenAI(base_url=endpoint, api_key=os.getenv("LIGHTONOCR_API_KEY", "EMPTY"))
        self._model = cfg.get("model") or os.getenv("LIGHTONOCR_MODEL") or "lightonocr3"
        self._prompt_mode = cfg.get("prompt_mode", "grounding")
        self._dpi = int(cfg.get("dpi", 400))
        self._max_edge = int(cfg.get("max_edge", 2048))
        self._max_tokens = int(cfg.get("max_tokens", 8192))
        self._temperature = float(cfg.get("temperature", 0.2))
        self._top_p = float(cfg.get("top_p", 0.9))
        self._timeout = int(cfg.get("timeout", 600))

    # -- inference ---------------------------------------------------------

    def _call(self, img: Image.Image) -> tuple[str, dict[str, Any]]:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        if self._prompt_mode == "grounding":
            content.append({"type": "text", "text": "grounding"})
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": content}],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                timeout=self._timeout,
            )
        except Exception as e:  # noqa: BLE001
            status = getattr(e, "status_code", None)
            if status and 400 <= status < 500:
                raise ProviderPermanentError(f"LightOnOCR request rejected ({status}): {e}") from e
            raise ProviderTransientError(f"LightOnOCR request failed: {e}") from e
        text = resp.choices[0].message.content or ""
        usage = resp.usage.model_dump() if resp.usage else {}
        return text, usage

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        started = datetime.now()
        images = render_pdf_pages(request.source_file_path, self._dpi, self._max_edge)
        pages: list[dict[str, Any]] = []
        for i, img in enumerate(images):
            text, usage = self._call(img)
            pages.append({"page_index": i, "width": img.width, "height": img.height, "raw_text": text, "usage": usage})
        completed = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={"model": self._model, "prompt_mode": self._prompt_mode, "pages": pages},
            started_at=started,
            completed_at=completed,
            latency_in_ms=int((completed - started).total_seconds() * 1000),
        )

    # -- normalization -----------------------------------------------------

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        raw = raw_result.raw_output
        pages: list[PageIR] = []
        layout_pages: list[ParseLayoutPageIR] = []
        for p in raw["pages"]:
            idx = p["page_index"]
            if raw.get("prompt_mode") == "grounding":
                blocks = parse_grounding(p["raw_text"])
                md = blocks_to_markdown(blocks)
                items = [
                    {"bbox": b["bbox"], "label": _LABEL_TO_CANONICAL_KEY.get(b["label"], "text"), "text": b["text"]}
                    for b in blocks
                    if b["bbox"] is not None
                ]
                layout_pages.extend(
                    build_layout_pages(items, p["width"], p["height"], md, page_number=idx + 1, bbox_scale=1000.0)
                )
            else:
                md = p["raw_text"].strip()
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
