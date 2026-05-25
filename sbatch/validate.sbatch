#!/bin/bash
#SBATCH --job-name=nebius-validate
#SBATCH --partition=earlytalent
#SBATCH --qos=gpulimit
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/validate-%j.out

set -uo pipefail

# HF token + HF_HOME=/mnt/data/jg/hf_cache.
source ~/.hf_env

# torch.distributed rendezvous coordinates.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29400

# Loud NCCL the first few runs. Set to WARN once the fabric is well understood.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

mkdir -p logs results/validation

srun --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/mnt/data/jg:/scratch,/home/jg:/home/jg \
     --container-workdir=/home/jg/Nebius-Take-Home \
     bash validation/run_all.sh
