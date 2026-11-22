#!/usr/bin/env bash
# Generic bootstrap: same pinned image and libraries as all previous runs.
set -euo pipefail
export BOOT_UNIX
BOOT_UNIX=$(date +%s.%N)
export PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface PIP_BREAK_SYSTEM_PACKAGES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m pip install --disable-pip-version-check --no-cache-dir diffusers==0.39.0 transformers==5.13.1 accelerate==1.14.0 sentencepiece==0.2.1 protobuf==6.33.0 huggingface_hub==1.16.1 'click<8.4'
exec python /app/worker.py --spec /app/keyframe-spec.json --out "/data/runs/keyframes/$RUN_ID"
