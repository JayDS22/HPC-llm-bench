#!/bin/bash
#SBATCH --job-name=qwen25-7b-mmlu-sft
#SBATCH --partition=earlytalent
#SBATCH --qos=gpulimit
#SBATCH -N 2
#SBATCH --ntasks-per-node=1          # one accelerate-launcher per node
#SBATCH --gres=gpu:2                 # 2 GPUs/node, 4 GPUs total
#SBATCH --cpus-per-task=32           # CPU pool the single launcher can use
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train-%j.out

set -uo pipefail
mkdir -p logs

# Single-task preinstall into NFS-shared ~/.local. Concurrent pip installs
# from 4 ranks on the same node clobber each other's temp files in the
# container's writable layer; do it once instead.
srun --nodes=1 --ntasks=1 \
     --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/home/jg:/home/jg \
     bash -c '
set -euo pipefail
pip install --user --quiet --no-cache-dir \
    "trl==0.11.4" "accelerate==1.0.1" "transformers==4.46.0" \
    "datasets==3.0.2" "pyyaml" "hf_transfer"
python -c "import trl, accelerate, transformers, datasets; \
  print(\"preinstall OK:\", trl.__version__, accelerate.__version__, \
        transformers.__version__, datasets.__version__)"
'

# Rendezvous coords for accelerate / torch.distributed.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29400

# Main multi-node training.
srun --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/mnt/data/jg:/mnt/data/jg,/home/jg:/home/jg \
     --container-workdir=/home/jg/Nebius-Take-Home \
     bash -c '
set -euo pipefail
set -a; source /home/jg/.hf_env; set +a

# Surface the --user-installed packages to Python.
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}
export PATH=$HOME/.local/bin:$PATH

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# accelerate launch: --num_processes is TOTAL workers across all machines,
# not per-machine. 2 nodes x 2 GPUs/node = 4 workers total.
accelerate launch \
  --config_file training/config/accelerate_fsdp.yaml \
  --num_machines $SLURM_NNODES \
  --num_processes 4 \
  --machine_rank $SLURM_NODEID \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  training/train_sft.py --config training/config/sft.yaml
'
