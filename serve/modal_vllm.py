"""Generic vLLM OpenAI-compatible server on Modal, scale-to-zero.

Deploy one app per model:
    DOCPARSE_MODEL=lightonai/LightOnOCR-3-0.8B DOCPARSE_SERVED_NAME=lightonocr3 \
        modal deploy serve/modal_vllm.py

Budget guards (the whole project runs on ~$20 of free credit):
  - max_containers=1, scaledown_window=120s: one L4 at most, gone 2 min after idle.
  - timeout=3600: a container can never live longer than an hour per input.
  - HF weights cached on a Volume so re-deploys don't re-download.
Estimated cost: L4 ≈ $0.80/hr → a full 2,078-page ParseBench run at 0.5–1 pg/s
is ~$0.50–1.00; the --test split (12 PDFs) is a few cents.
"""

import os
import re
import subprocess

import modal

MODEL = os.environ.get("DOCPARSE_MODEL", "lightonai/LightOnOCR-3-0.8B")
SERVED_NAME = os.environ.get("DOCPARSE_SERVED_NAME", "lightonocr3")
GPU = os.environ.get("DOCPARSE_GPU", "L4")
EXTRA_ARGS = os.environ.get("DOCPARSE_VLLM_ARGS", "")
APP_NAME = "docparse-" + re.sub(r"[^a-z0-9]+", "-", SERVED_NAME.lower()).strip("-")

PORT = 8000
CACHE_DIR = "/root/.cache"  # HF weights + vLLM torch.compile artifacts persist across cold starts

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.31.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)

hf_cache = modal.Volume.from_name("docparse-hf-cache", create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    volumes={CACHE_DIR: hf_cache},
    scaledown_window=120,
    timeout=3600,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=900)
def serve() -> None:
    cmd = [
        "vllm", "serve", MODEL,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "16384",
        "--limit-mm-per-prompt", '{"image": 1}',
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
    ]
    if EXTRA_ARGS:
        cmd += EXTRA_ARGS.split()
    subprocess.Popen(cmd)
