# Runbook

Copy-paste path from a clean clone to a full set of results. One page.

## Prerequisites

- SSH access to the cluster (`ssh jg@<login-ip>`) with `~/.ssh/<key>` and a matching `~/.ssh/config` `Host nebius` block.
- A read-only Hugging Face token from https://huggingface.co/settings/tokens.
- Cluster has Pyxis + enroot wired into Slurm (`srun --container-image=docker://...` works).
- Shared scratch path `/mnt/data/$USER` exists and is writable.

## One-time setup

```bash
# 1. Stage secrets in the gitignored .env
cat > .env <<'EOF'
HF_TOKEN=<your-read-token>
HF_HOME=/mnt/data/$USER/hf_cache
HF_HUB_ENABLE_HF_TRANSFER=1
EOF
chmod 600 .env

# 2. Sync .env to the cluster
scp .env nebius:~/.hf_env
ssh nebius 'chmod 600 ~/.hf_env'

# 3. Sync the repo
rsync -av --delete \
  --exclude='.git/' --exclude='.env' --exclude='.env.*' \
  --exclude='results/' --exclude='logs/' \
  ./ nebius:Nebius-Take-Home/

# 4. Pre-stage the base model (saves ~5 min on every later job)
ssh nebius 'cd Nebius-Take-Home && sbatch sbatch/prestage_qwen.sbatch'

# 5. Build the SFT dataset (one-shot, ~1 min)
ssh nebius 'srun --partition=earlytalent --qos=gpulimit \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem=16G --time=00:10:00 \
  --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
  --container-mounts=/mnt/data/jg:/mnt/data/jg,/home/jg:/home/jg \
  --container-workdir=/home/jg/Nebius-Take-Home \
  bash -c "set -a; source /home/jg/.hf_env; set +a; \
           pip install --quiet datasets==3.0.2 hf_transfer; \
           python training/data/build_mmlu_sft.py \
             --out /mnt/data/jg/datasets/mmlu_sft.jsonl --n 20000"'
```

## Phase 0: validate the cluster (once)

```bash
ssh nebius 'cd Nebius-Take-Home && sbatch sbatch/validate.sbatch'
# After it finishes:
ssh nebius 'cat ~/Nebius-Take-Home/results/validation/<JOB_ID>/report.md'
```

Expect 4 PASS sections (gpu_info, gpu_compute, nccl_bench, hf_smoke). Numbers archived under `results/validation/<JOB_ID>/`.

## Phase 1: train

```bash
# Optional 1-GPU smoke test first (~5 min)
ssh nebius 'cd Nebius-Take-Home && sbatch sbatch/train_smoke.sbatch'

# Full multi-node SFT (~15 min on 2 nodes x 2 H200 with FSDP)
ssh nebius 'cd Nebius-Take-Home && sbatch sbatch/train.sbatch'

# Consolidate sharded FSDP checkpoint into an HF-loadable directory (~2 min)
ssh nebius 'srun --partition=earlytalent --qos=gpulimit \
  --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:15:00 \
  --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
  --container-mounts=/mnt/data/jg:/mnt/data/jg,/home/jg:/home/jg \
  --container-workdir=/home/jg/Nebius-Take-Home \
  bash -c "set -a; source /home/jg/.hf_env; set +a; \
           python training/consolidate_fsdp.py \
             --base-model Qwen/Qwen2.5-7B-Instruct \
             --sharded-dir /mnt/data/jg/checkpoints/qwen25-7b-mmlu-sft/checkpoint-312/pytorch_model_fsdp_0 \
             --tokenizer-dir /mnt/data/jg/checkpoints/qwen25-7b-mmlu-sft \
             --out-dir /mnt/data/jg/checkpoints/qwen25-7b-mmlu-sft-merged"'
```

## Phase 2: evaluate

```bash
# Accuracy: base vs SFT on target subject + forgetting check (~7 min)
ssh nebius 'cd Nebius-Take-Home && sbatch sbatch/eval_accuracy.sbatch'

# Throughput: transformers vs vLLM head-to-head (~8 min)
ssh nebius 'cd Nebius-Take-Home && sbatch sbatch/bench_throughput.sbatch'
```

Both fit within the 2-concurrent-job QOS cap, so submit them together. They use separate `PYTHONUSERBASE` prefixes so the parallel pip installs don't step on each other.

## Pull results back to laptop

```bash
rsync -av nebius:Nebius-Take-Home/results/ ./results/
```

Key files:

- `results/validation/<JOB>/report.md`: cluster validation summary
- `results/eval/accuracy.md`: base vs SFT MMLU table
- `results/eval/throughput.md`: transformers vs vLLM bench

## Generate plots

Needs `matplotlib` + `numpy` locally.

```bash
mkdir -p results/plots
python3 evaluation/plots.py loss --log logs/train-<JOB>.out --out results/plots/loss.png
python3 evaluation/plots.py accuracy --accuracy-json results/eval/accuracy.json --out results/plots/accuracy.png
python3 evaluation/plots.py throughput --hf-json results/eval/throughput_hf.json --vllm-json results/eval/throughput_vllm.json --out results/plots/throughput.png
```

## Knobs

- Subject of fine-tuning: edit `TARGET` env var when sbatch'ing `eval_accuracy.sbatch`, and rebuild the SFT dataset with a filtered `auxiliary_train` if you want subject-specific gains.
- Batch / LR: `training/config/sft.yaml`. The defaults (lr 2e-5, eff. bs 64) are good starting points for any Qwen2.5-7B-class model.
- vLLM perf: `--tensor-parallel-size`, `--max-num-seqs`, and `--kv-cache-dtype fp8` in `evaluation/bench/bench_vllm.py` are the highest-leverage knobs.
