"""Run the parse-bench harness inside Modal (CPU), next to the GPU endpoint.

Nothing heavy crosses the local connection: the dataset is downloaded from
Hugging Face into a Modal volume, page images go container-to-container, and
results (raw VLM output, normalized output, evaluation reports) persist on the
volume. Pull only what you need afterwards:

    modal run --detach serve/modal_bench.py --pipeline lightonocr3_0_8b_grounding
    modal volume ls docparse-output bench/output
    modal volume get docparse-output bench/output/lightonocr3_0_8b_grounding ./output/

Cost: a 4-CPU container at a few cents per hour; GPU time is billed by the
separate vLLM app only while requests are in flight.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal
import threading
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
ENV = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v}

VOL_PATH = "/vol"
BENCH_DIR = f"{VOL_PATH}/bench"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("poppler-utils")
    .pip_install(
        "parse-bench==1.0.4",
        "infinity-parser2>=0.4.0",
        "llama-cloud>=2.18.0",
        "openai>=1.0",
        "pdf2image>=1.17.0",
        "pillow>=10.0",
        "pymupdf>=1.24",
        "python-dotenv>=1.0",
        "rapidfuzz>=3.14",
    )
    .env({"PYTHONUNBUFFERED": "1", **ENV})
    .add_local_python_source("docparse")
)

vol = modal.Volume.from_name("docparse-output", create_if_missing=True)
app = modal.App("docparse-bench")


def _sh(cmd: list[str], cwd: str) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=4, memory=8192, timeout=4 * 3600)
def run(pipeline: str, max_concurrent: int = 16, test: bool = False, extra: str = "") -> str:
    os.makedirs(BENCH_DIR, exist_ok=True)
    data_dir = Path(BENCH_DIR) / ("data/test" if test else "data")
    if not (data_dir / "docs").exists():
        _sh([sys.executable, "-c", "import docparse.cli; docparse.cli.main()", "download", *(["--test"] if test else [])], BENCH_DIR)
        vol.commit()
    # Commit partial results periodically: CPU containers can be preempted, and a
    # restarted run skips files that already have results on the volume.
    stop = threading.Event()

    def committer() -> None:
        while not stop.wait(120):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print("volume commit failed:", e, flush=True)

    threading.Thread(target=committer, daemon=True).start()
    cmd = [
        sys.executable, "-c", "import docparse.cli; docparse.cli.main()",
        "run", pipeline, "--max_concurrent", str(max_concurrent),
    ]
    if test:
        cmd.append("--test")
    if extra:
        cmd += extra.split()
    rc = _sh(cmd, BENCH_DIR)
    stop.set()
    vol.commit()
    out = Path(BENCH_DIR) / "output" / pipeline
    n = len(list(out.rglob("*.result.json"))) if out.exists() else 0
    return f"rc={rc} results={n} output={out}"


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=4, memory=8192, timeout=2 * 3600)
def renormalize_eval(source: str, target: str, test: bool = False) -> str:
    """Score a normalization-only pipeline on another pipeline's saved raw outputs."""
    import json
    import shutil

    out_root = Path(BENCH_DIR) / "output"
    src, dst = out_root / source, out_root / target
    for d in src.iterdir():
        if d.is_dir():
            (dst / d.name).mkdir(parents=True, exist_ok=True)
            for f in d.glob("*.raw.json"):
                shutil.copy(f, dst / d.name / f.name)
    import docparse.pipelines  # noqa: F401
    from parse_bench.inference.pipelines import get_pipeline

    meta = json.load(open(src / "_metadata.json"))
    meta["pipeline"] = get_pipeline(target).model_dump()
    json.dump(meta, open(dst / "_metadata.json", "w"), indent=2)
    py = [sys.executable, "-c", "import docparse.cli; docparse.cli.main()"]
    _sh([*py, "inference", "renormalize", str(dst), "--pipeline_name", target, "--force"], BENCH_DIR)
    rc = _sh([*py, "run", target, "--skip_inference", *(["--test"] if test else [])], BENCH_DIR)
    vol.commit()
    table = subprocess.run(
        [sys.executable, "-c", "import docparse.oracle; docparse.oracle.main()", source, target],
        cwd=BENCH_DIR, capture_output=True, text=True,
    )
    return f"rc={rc}\n{table.stdout}{table.stderr[-500:]}"


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=1, memory=2048, timeout=600)
def sizes(pipeline: str) -> str:
    out = Path(BENCH_DIR) / "output" / pipeline
    def total(pattern: str) -> int:
        return sum(f.stat().st_size for f in out.rglob(pattern))
    return (
        f"raw={total('*.raw.json') / 1e6:.1f}MB result={total('*.result.json') / 1e6:.1f}MB "
        f"reports={total('_evaluation_report.json') / 1e6:.1f}MB html={total('*.html') / 1e6:.1f}MB"
    )


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=1, memory=2048, timeout=900)
def pack_raw(pipeline: str) -> str:
    """Tar+gzip a pipeline's raw VLM outputs (small) for download: output/<pipeline>/<group>/*.raw.json."""
    import tarfile

    out = Path(BENCH_DIR) / "output"
    tgz = Path(BENCH_DIR) / f"raw_{pipeline}.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        for f in sorted((out / pipeline).rglob("*.raw.json")):
            tf.add(f, arcname=str(f.relative_to(out)))
        tf.add(out / pipeline / "_metadata.json", arcname=f"{pipeline}/_metadata.json")
    vol.commit()
    return f"{tgz.relative_to(VOL_PATH)} {tgz.stat().st_size / 1e6:.1f}MB"


@app.local_entrypoint()
def main(
    pipeline: str = "lightonocr3_0_8b_grounding",
    max_concurrent: int = 16,
    test: bool = False,
    extra: str = "",
    mode: str = "run",
    target: str = "docparse_hybrid_v0",
):
    if mode == "run":
        print(run.remote(pipeline, max_concurrent, test, extra))
    elif mode == "renorm":
        print(renormalize_eval.remote(pipeline, target, test))
    elif mode == "sizes":
        print(sizes.remote(pipeline))
    elif mode == "pack":
        print(pack_raw.remote(pipeline))
