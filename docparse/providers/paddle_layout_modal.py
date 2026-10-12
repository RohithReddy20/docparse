"""PP-DocLayoutV3 layout-detection provider against serve/modal_paddle_layout.py.

parse-bench's built-in `paddle_layout` provider POSTs raw PNG bytes to a fixed
HuggingFace endpoint, needs HF_TOKEN, and only accepts image inputs. ParseBench's
layout split ships PDFs, so this provider:

- reads the endpoint from PADDLE_LAYOUT_ENDPOINT_URL (no token),
- renders the requested PDF page (config `dpi`, default 200; `page_index`
  override honoured like the dots_ocr layout provider),
- POSTs the PNG to `{endpoint}/predict` and parses the same
  `{"predictions": [{"coordinate", "label", "score"}]}` shape via the parent.

Labels are the 20-class PP-DocLayout names, so the stock PPDocLayout adapter
(label ints 0-19 -> Canonical17) applies unchanged.
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Any

import pymupdf
import requests
from PIL import Image

from parse_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.layoutdet.paddle import PaddleLayoutProvider
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.layout_detection_output import LayoutOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType


def render_pdf_page(pdf_path: str, page_index: int, dpi: int) -> Image.Image:
    doc = pymupdf.open(pdf_path)
    if page_index >= len(doc):
        doc.close()
        raise ProviderPermanentError(f"PDF has no page at index {page_index}: {pdf_path}")
    page = doc[page_index]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


@register_provider("paddle_layout_modal")
class PaddleLayoutModalProvider(PaddleLayoutProvider):
    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None) -> None:
        cfg = dict(base_config or {})
        endpoint = cfg.get("endpoint_url") or os.getenv("PADDLE_LAYOUT_ENDPOINT_URL")
        if not endpoint:
            raise ProviderConfigError("Set PADDLE_LAYOUT_ENDPOINT_URL (URL printed by `modal deploy serve/modal_paddle_layout.py`).")
        # Parent requires HF_TOKEN; our server is unauthenticated.
        cfg.setdefault("hf_token", "unused")
        super().__init__(provider_name, cfg)
        self.endpoint_url = endpoint.rstrip("/")
        self._dpi = int(cfg.get("dpi", 200))
        self._page_index = int(cfg.get("page_index", 0))
        self._timeout = int(cfg.get("timeout", 300))

    def _call_endpoint(self, image: Image.Image) -> dict[str, Any]:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        try:
            resp = requests.post(
                f"{self.endpoint_url}/predict",
                data=buf.getvalue(),
                headers={"Content-Type": "image/png"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]
        except requests.exceptions.Timeout as e:
            raise ProviderTransientError(f"Request timed out: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ProviderTransientError(f"Connection error: {e}") from e
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 429:
                raise ProviderRateLimitError(str(e)) from e
            if status and status >= 500:
                raise ProviderTransientError(f"Server error ({status}): {e}") from e
            raise ProviderPermanentError(f"HTTP error ({status}): {e}") from e

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.LAYOUT_DETECTION:
            raise ProviderPermanentError(f"{type(self).__name__} only supports LAYOUT_DETECTION")
        started = datetime.now()
        page_index = self._page_index
        if request.config_override and "page_index" in request.config_override:
            page_index = int(request.config_override["page_index"])

        src = request.source_file_path
        if src.lower().endswith(".pdf"):
            image = render_pdf_page(src, page_index, self._dpi)
        else:
            image = Image.open(src).convert("RGB")

        response = self._call_endpoint(image)
        completed = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={
                "response": response,
                "image_width": image.width,
                "image_height": image.height,
                "page_number": page_index + 1,
                "dpi": self._dpi,
            },
            started_at=started,
            completed_at=completed,
            latency_in_ms=int((completed - started).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        result = super().normalize(raw_result)
        page = raw_result.raw_output.get("page_number")
        output = result.output
        if isinstance(output, LayoutOutput) and page is not None:
            for pred in output.predictions:
                pred.page = int(page)
        return result
