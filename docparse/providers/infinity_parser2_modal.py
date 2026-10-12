"""Infinity-Parser2 (infly/Infinity-Parser2-Flash, Apache-2.0) against a Modal vLLM endpoint.

Thin wrapper over parse-bench's built-in ``infinity_parser2`` provider:

- reads the endpoint from ``INFINITY_ENDPOINT_URL`` (either the base URL ending
  in ``/v1`` or the full ``/v1/chat/completions``; the SDK wants the full path),
- waits for the scale-to-zero container to come up before the SDK's 5-second
  connectivity ping runs in ``__init__`` (a cold Modal boot takes minutes),
- caps ``max_new_tokens`` so it fits the server's ``--max-model-len``.

The SDK sends ``model=<model_name>`` on every request, so the vLLM server must
be started with ``--served-model-name infly/Infinity-Parser2-Flash``. Thinking
is disabled server-side via ``--default-chat-template-kwargs`` (the SDK's
server backend does not pass ``enable_thinking=False`` itself).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import urllib.error
import urllib.request

from parse_bench.inference.providers.base import ProviderConfigError
from parse_bench.inference.providers.parse.infinity_parser2 import InfinityParser2Provider
from parse_bench.inference.providers.registry import register_provider

logger = logging.getLogger(__name__)


def _chat_completions_url(raw: str) -> str:
    raw = raw.rstrip("/")
    return raw if raw.endswith("/chat/completions") else f"{raw}/chat/completions"


def wait_for_endpoint(chat_url: str, timeout_s: int) -> None:
    """Poll ``/v1/models`` until the vLLM server answers (cold start)."""
    models_url = chat_url.rsplit("/chat/completions", 1)[0] + "/models"
    deadline = time.time() + timeout_s
    while True:
        try:
            with urllib.request.urlopen(models_url, timeout=30) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        if time.time() > deadline:
            raise ProviderConfigError(f"vLLM endpoint not ready after {timeout_s}s: {models_url}")
        time.sleep(10)


@register_provider("infinity_parser2_modal")
class InfinityParser2ModalProvider(InfinityParser2Provider):
    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None) -> None:
        cfg = dict(base_config or {})
        endpoint = cfg.get("api_url") or os.getenv("INFINITY_ENDPOINT_URL")
        if not endpoint:
            raise ProviderConfigError("Set INFINITY_ENDPOINT_URL (Modal vLLM URL, base /v1 or full /v1/chat/completions).")
        cfg["api_url"] = _chat_completions_url(endpoint)
        cfg.setdefault("model_name", "infly/Infinity-Parser2-Flash")
        cfg.setdefault("max_new_tokens", 16384)
        wait_for_endpoint(cfg["api_url"], timeout_s=int(cfg.pop("startup_timeout", 900)))
        super().__init__(provider_name, cfg)
