"""Bundle detector-aware Section/Text training for a private Kaggle T4 run.

    uv run python scripts/kaggle_section_text.py --output /tmp/docparse-kaggle --kernel USER/SLUG --enable-gpu
    kaggle kernels push -p /tmp/docparse-kaggle -t 7200 --accelerator NvidiaTeslaT4

The DocLayNet dataset stays on Kaggle. Download section_text_artifacts.zip only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BOOTSTRAP = """import importlib.util, json, os, pathlib, subprocess, sys, time
working=pathlib.Path('/kaggle/working')
status=working/'section_text_status.json'
def stage(name, **details):
    status.write_text(json.dumps({'stage':name,'time':time.time(),**details}))
    print(name,details,flush=True)
stage('started',smoke=smoke)
print('Starting detector-aware DocLayNet training',flush=True)
requirements={'pymupdf':'pymupdf>=1.24','rapidfuzz':'rapidfuzz>=3.14','transformers':'transformers>=4.50'}
missing=[package for module,package in requirements.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',*missing], check=True,timeout=300)
runtime=pathlib.Path('/tmp/docparse-section-runtime')
for name, content in sources.items():
    path=runtime/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content)
(runtime/'docparse/__init__.py').write_text('')
sys.path.insert(0,str(runtime))
data=pathlib.Path('/kaggle/input')
roots=[data,*data.glob('*'),*data.glob('*/*'),*data.glob('*/*/*')]
roots += [base/'DocLayNet_subset' for base in roots if (base/'DocLayNet_subset').is_dir()]
train=next((p for base in roots for p in (base/'COCO/train.json',base/'DocLayNet_core/COCO/train.json') if p.is_file()),None)
pdf=next((p for base in roots for p in (base/'PDF',base/'DocLayNet_extra/PDF') if p.is_dir()),None)
png=next((p for base in roots for p in (base/'PNG',base/'DocLayNet_core/PNG') if p.is_dir()),None)
print('COCO:',train,'PDF:',pdf,'PNG:',png,flush=True)
if train is None or pdf is None or png is None:
    print('Input directories:',[str(p) for p in data.glob('*/*') if p.is_dir()],flush=True)
    raise RuntimeError('DocLayNet core COCO and extra PDF folders required')
print('PDF count:',sum(1 for p in pdf.glob('*.pdf')),flush=True)
features=working/'section_text_features'
env={**os.environ,'PYTHONPATH':str(runtime),'PYTHONUNBUFFERED':'1'}
def run_stage(name, script, args, timeout):
    stage(name,timeout_seconds=timeout)
    try:
        with (working/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-u',str(runtime/'scripts'/script),*args],
                           check=True,timeout=timeout,env=env,stdout=log,stderr=subprocess.STDOUT)
    except Exception as error:
        stage('failed',failed_stage=name,error=str(error))
        print((working/(name+'.log')).read_text()[-12000:],flush=True)
        raise
feature_args=['--coco',str(train.parent),'--pdf',str(pdf),'--png',str(png),'--output',str(features)]
if smoke:
    feature_args += ['--train-pages','128','--validation-pages','64','--test-pages','64']
run_stage('features','doclaynet_heron_features.py',feature_args,600 if smoke else 5400)
run_stage('training','train_section_text.py',['--features',str(features)],300)
stage('complete',smoke=smoke)
print((working/'training.log').read_text(),flush=True)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--dataset", default="ayoubelmendoub/doclaynet")
    parser.add_argument("--kernel-source", help="Use a prepared DocLayNet subset notebook instead of the full dataset")
    parser.add_argument("--enable-gpu", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run 128/64/64 pages to verify the training path")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    sources = {name: (root / name).read_text() for name in (
        "docparse/textlayer.py", "docparse/section_text.py", "docparse/hybrid.py", "docparse/fusion.py",
        "scripts/doclaynet_heron_features.py", "scripts/train_section_text.py")}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train.py").write_text("sources = " + repr(sources) + "\nsmoke = " + repr(args.smoke) + "\n" + BOOTSTRAP)
    title = args.kernel.split("/")[-1].replace("-", " ").title()
    meta = {"id": args.kernel, "title": title, "code_file": "train.py", "language": "python", "kernel_type": "script",
            "is_private": "true", "enable_gpu": str(args.enable_gpu).lower(), "enable_internet": "true",
            "dataset_sources": [] if args.kernel_source else [args.dataset],
            "competition_sources": [], "kernel_sources": [args.kernel_source] if args.kernel_source else [], "model_sources": []}
    if args.enable_gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (args.output / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("Private kernel bundle:", args.output, "GPU:", args.enable_gpu)


if __name__ == "__main__":
    main()
