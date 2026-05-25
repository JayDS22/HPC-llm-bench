#!/bin/bash
#SBATCH --job-name=qwen-smoke
#SBATCH --partition=earlytalent
#SBATCH --qos=gpulimit
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/train-smoke-%j.out

# 1-GPU dry run that exercises the full TRL/SFTTrainer pipeline for 10 steps.
# Catches API drift, dataset shape problems, OOMs at seq_len, etc., before
# paying for a 4-GPU multi-node allocation. Expect ~5 min wall clock.
set -uo pipefail
mkdir -p logs

srun --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/mnt/data/jg:/mnt/data/jg,/home/jg:/home/jg \
     --container-workdir=/home/jg/Nebius-Take-Home \
     bash -c '
set -euo pipefail
set -a; source /home/jg/.hf_env; set +a
pip install --quiet --no-cache-dir \
    "trl==0.11.4" "accelerate==1.0.1" "transformers==4.46.0" \
    "datasets==3.0.2" "pyyaml" "hf_transfer" 2>&1 | tail -3
export PYTHONUNBUFFERED=1
python training/train_sft.py --config training/config/sft_smoke.yaml
'
