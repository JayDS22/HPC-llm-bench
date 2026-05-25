#!/bin/bash
#SBATCH --job-name=eval-accuracy
#SBATCH --partition=earlytalent
#SBATCH --qos=gpulimit
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval-accuracy-%j.out

# lm-evaluation-harness MMLU eval: base vs SFT on target subject + forgetting
# check. Uses a dedicated PYTHONUSERBASE (~/.local/eval-env) so a parallel
# bench job can't nuke this pip env mid-run.
set -uo pipefail
mkdir -p logs results/eval

TARGET="${TARGET:-mmlu_high_school_mathematics}"
FORGETTING="${FORGETTING:-mmlu_college_biology,mmlu_world_religions,mmlu_high_school_us_history}"
TASKS="${TARGET},${FORGETTING}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SFT_MODEL="${SFT_MODEL:-/mnt/data/jg/checkpoints/qwen25-7b-mmlu-sft-merged}"
OUT_DIR="${OUT_DIR:-/home/jg/Nebius-Take-Home/results/eval}"
ENV_BASE="$HOME/.local/eval-env"

# Preinstall lm-eval from a git tag. PyPI 0.4.5 wheel is missing task YAMLs.
srun --nodes=1 --ntasks=1 \
     --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/home/jg:/home/jg \
     bash -c "
set -uo pipefail
export PYTHONUSERBASE=$ENV_BASE
rm -rf $ENV_BASE 2>/dev/null || true
mkdir -p $ENV_BASE
pip install --user --quiet --no-cache-dir \
    'git+https://github.com/EleutherAI/lm-evaluation-harness.git@v0.4.5' \
    'transformers==4.46.0' 'datasets==3.0.2' 'hf_transfer'
test -f $ENV_BASE/lib/python3.10/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_mathematics.yaml \
    && echo 'preinstall OK: mmlu YAMLs present' \
    || { echo 'FAIL: mmlu YAMLs missing'; exit 1; }
"

srun --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/mnt/data/jg:/mnt/data/jg,/home/jg:/home/jg \
     --container-workdir=/home/jg/Nebius-Take-Home \
     bash -c "
set -euo pipefail
set -a; source /home/jg/.hf_env; set +a
export PYTHONUSERBASE=$ENV_BASE
export PYTHONPATH=$ENV_BASE/lib/python3.10/site-packages:\${PYTHONPATH:-}
export PATH=$ENV_BASE/bin:\$PATH
export PYTHONUNBUFFERED=1

# Both base and SFT scored with --apply-chat-template so the comparison
# stays apples-to-apples in the model's native inference format.
echo --- base model ---
python evaluation/eval_mmlu.py \
    --model '$BASE_MODEL' --tasks '$TASKS' \
    --out-dir '$OUT_DIR' --label base --batch-size 8 --num-fewshot 5 \
    --apply-chat-template

echo --- SFT model ---
python evaluation/eval_mmlu.py \
    --model '$SFT_MODEL' --tasks '$TASKS' \
    --out-dir '$OUT_DIR' --label sft --batch-size 8 --num-fewshot 5 \
    --apply-chat-template

echo --- comparison ---
python evaluation/compare_accuracy.py \
    --base '$OUT_DIR/base/summary.json' \
    --sft  '$OUT_DIR/sft/summary.json' \
    --target-task '$TARGET' \
    --out-json '$OUT_DIR/accuracy.json' \
    --out-md   '$OUT_DIR/accuracy.md'

cat '$OUT_DIR/accuracy.md'
"
