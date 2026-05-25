#!/bin/bash
# Validation entry point. Runs inside the NGC PyTorch container, once per Slurm task.
set -uo pipefail

REPO=/home/jg/Nebius-Take-Home
RUN_ID="${SLURM_JOB_ID:-local}"
OUT_DIR="$REPO/results/validation/${RUN_ID}"
mkdir -p "$OUT_DIR"

RANK="${SLURM_PROCID:-0}"
WORLD="${SLURM_NTASKS:-1}"
LOCAL="${SLURM_LOCALID:-0}"
NODE="${SLURMD_NODENAME:-$(hostname)}"

# Standard FSDP pattern: every task on a node sees all node-local GPUs;
# each process picks its own via torch.cuda.set_device(SLURM_LOCALID).
# Restricting CUDA_VISIBLE_DEVICES per task breaks intra-node NCCL P2P.
echo "[rank $RANK/$WORLD local=$LOCAL] node=$NODE  visible_gpus=${CUDA_VISIBLE_DEVICES:-?}"

# Extras the NGC image doesn't ship. Rank 0 installs once into ~/.local;
# NFS-shared home means all ranks see it. Quiet pip; show last lines if it errors.
if [ "$RANK" = "0" ]; then
  pip install --quiet --no-cache-dir --user \
      "sentence-transformers==3.2.1" \
      "huggingface_hub>=0.26.0" 2>&1 | tail -3
fi
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"

# Per-rank checks
python "$REPO/validation/checks/gpu_compute.py" \
    --out "$OUT_DIR/gpu_compute_rank${RANK}.json"

python "$REPO/validation/checks/nccl_bench.py" \
    --out "$OUT_DIR/nccl_bench_rank${RANK}.json"

# Rank-0-only checks
if [ "$RANK" = "0" ]; then
  python "$REPO/validation/checks/gpu_info.py" \
      --out "$OUT_DIR/gpu_info.json"
  python "$REPO/validation/checks/hf_smoke.py" \
      --out "$OUT_DIR/hf_smoke.json"
fi

# Filesystem barrier so rank 0 sees everyone's JSONs before aggregating.
sleep 10
if [ "$RANK" = "0" ]; then
  python "$REPO/validation/aggregate.py" \
      --in-dir "$OUT_DIR" \
      --out "$OUT_DIR/report.json" \
      --md "$OUT_DIR/report.md"
  echo
  echo "validation report:"
  cat "$OUT_DIR/report.md"
fi
