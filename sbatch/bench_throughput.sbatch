#!/bin/bash
#SBATCH --job-name=bench-throughput
#SBATCH --partition=earlytalent
#SBATCH --qos=gpulimit
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/bench-throughput-%j.out

# transformers.generate vs vLLM, same prompts and same node. HF and vLLM
# use SEPARATE PYTHONUSERBASE prefixes so vLLM's torch 2.4 can't ABI-break
# the container's prebuilt flash_attn that the HF run depends on.
set -uo pipefail
mkdir -p logs results/eval

MODEL="${MODEL:-/mnt/data/jg/checkpoints/qwen25-7b-mmlu-sft-merged}"
PROMPTS=/mnt/data/jg/datasets/bench_prompts.jsonl
OUT_DIR=results/eval
HF_BASE="$HOME/.local/bench-env-hf"
VLLM_BASE="$HOME/.local/bench-env-vllm"

# 1) HF preinstall (clean prefix).
srun --nodes=1 --ntasks=1 \
     --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/home/jg:/home/jg \
     bash -c "
set -uo pipefail
export PYTHONUSERBASE=$HF_BASE
rm -rf $HF_BASE 2>/dev/null || true
mkdir -p $HF_BASE
pip install --user --quiet --no-cache-dir \
    'transformers==4.46.0' 'datasets==3.0.2' 'hf_transfer'
echo 'HF preinstall OK'
"

# 2) Build prompts (idempotent, uses HF env).
srun --nodes=1 --ntasks=1 --gres=gpu:0 \
     --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/mnt/data/jg:/mnt/data/jg,/home/jg:/home/jg \
     --container-workdir=/home/jg/Nebius-Take-Home \
     bash -c "
set -euo pipefail
set -a; source /home/jg/.hf_env; set +a
export PYTHONUSERBASE=$HF_BASE
export PYTHONPATH=$HF_BASE/lib/python3.10/site-packages:\${PYTHONPATH:-}
[ -s '$PROMPTS' ] || python evaluation/bench/load_prompts.py \
    --subject high_school_mathematics --n 500 --out '$PROMPTS'
wc -l '$PROMPTS'
"

# 3) HF baseline bench (uses HF env).
srun --nodes=1 --ntasks=1 --gres=gpu:1 \
     --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/mnt/data/jg:/mnt/data/jg,/home/jg:/home/jg \
     --container-workdir=/home/jg/Nebius-Take-Home \
     bash -c "
set -euo pipefail
set -a; source /home/jg/.hf_env; set +a
export PYTHONUSERBASE=$HF_BASE
export PYTHONPATH=$HF_BASE/lib/python3.10/site-packages:\${PYTHONPATH:-}
export PATH=$HF_BASE/bin:\$PATH
export PYTHONUNBUFFERED=1
python evaluation/bench/bench_hf.py \
    --model '$MODEL' --prompts '$PROMPTS' --n 500 \
    --batch-size 8 --max-new-tokens 128 \
    --out $OUT_DIR/throughput_hf.json
"

# 4) vLLM preinstall (separate prefix; zstandard upgrade fixes httpx compat).
srun --nodes=1 --ntasks=1 \
     --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/home/jg:/home/jg \
     bash -c "
set -uo pipefail
export PYTHONUSERBASE=$VLLM_BASE
rm -rf $VLLM_BASE 2>/dev/null || true
mkdir -p $VLLM_BASE
pip install --user --quiet --no-cache-dir \
    'zstandard>=0.20.0' \
    'vllm==0.6.3' 'transformers==4.46.0' 'datasets==3.0.2' 'hf_transfer'
echo 'vLLM preinstall OK'
"

# 5) vLLM bench.
srun --nodes=1 --ntasks=1 --gres=gpu:2 \
     --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/mnt/data/jg:/mnt/data/jg,/home/jg:/home/jg \
     --container-workdir=/home/jg/Nebius-Take-Home \
     bash -c "
set -euo pipefail
set -a; source /home/jg/.hf_env; set +a
export PYTHONUSERBASE=$VLLM_BASE
export PYTHONPATH=$VLLM_BASE/lib/python3.10/site-packages:\${PYTHONPATH:-}
export PATH=$VLLM_BASE/bin:\$PATH
export PYTHONUNBUFFERED=1
python evaluation/bench/bench_vllm.py \
    --model '$MODEL' --prompts '$PROMPTS' --n 500 \
    --tensor-parallel-size 2 --max-num-seqs 256 --max-new-tokens 128 \
    --out $OUT_DIR/throughput_vllm.json
"

# 6) Compare.
srun --nodes=1 --ntasks=1 --gres=gpu:0 \
     --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/home/jg:/home/jg \
     --container-workdir=/home/jg/Nebius-Take-Home \
     bash -c "
set -euo pipefail
python evaluation/bench/compare_throughput.py \
    --hf   $OUT_DIR/throughput_hf.json \
    --vllm $OUT_DIR/throughput_vllm.json \
    --out-md $OUT_DIR/throughput.md
cat $OUT_DIR/throughput.md
"
