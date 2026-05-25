#!/bin/bash
#SBATCH --job-name=prestage-qwen25-7b
#SBATCH --partition=earlytalent
#SBATCH --qos=gpulimit
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prestage-%j.out

# Pre-download Qwen/Qwen2.5-7B-Instruct into the shared scratch HF cache so
# training jobs don't pay the pull cost during a real wall-clock allocation.
# CPU-only, no GPU.
set -uo pipefail
source ~/.hf_env

mkdir -p logs "$HF_HOME"

# Path-preserving mount: HF_HOME resolves the same inside and outside the container.
srun --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/mnt/data/jg:/mnt/data/jg,/home/jg:/home/jg \
     --container-workdir=/home/jg/Nebius-Take-Home \
     bash -c '
set -euo pipefail
# Pyxis does not inherit env from the sbatch shell, re-source inside.
# `set -a` auto-exports every variable in the sourced file so Python
# subprocesses can read them via os.environ.
set -a; source /home/jg/.hf_env; set +a
pip install --quiet --no-cache-dir "huggingface_hub[hf_transfer]>=0.26.0"
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "HF_HOME=$HF_HOME  HF_TOKEN_set=$([ -n \"$HF_TOKEN\" ] && echo yes || echo no)"

python - <<PY
import os, time
from huggingface_hub import snapshot_download

repo = "Qwen/Qwen2.5-7B-Instruct"
t0 = time.time()
path = snapshot_download(
    repo_id=repo,
    cache_dir=os.environ["HF_HOME"],
    token=os.environ.get("HF_TOKEN"),
    max_workers=8,
    allow_patterns=[
        "*.json", "*.txt", "tokenizer.*", "merges.txt",
        "*.safetensors",
    ],
)
dt = time.time() - t0
print(f"snapshot_download: {repo} -> {path}")
print(f"elapsed: {dt:.1f}s")
PY

echo "--- cache footprint ---"
du -sh "$HF_HOME"
echo "--- repo contents ---"
SNAP=$(ls -d "$HF_HOME"/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ 2>/dev/null | head -1)
ls -la "$SNAP" 2>&1 | head -40
'
