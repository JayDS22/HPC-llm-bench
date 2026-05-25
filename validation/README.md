# Cluster validation

A small set of checks that proves the cluster can run training and inference workloads. Runs as a single Slurm job that pulls the NGC PyTorch container via Pyxis. No separate image build.

## Checks

| Check | Where it runs | What it measures |
| --- | --- | --- |
| `gpu_info` | rank 0 | Driver, CUDA, GPU model, HBM per GPU |
| `gpu_compute` | every rank | Sustained bf16 matmul TFLOPs on each visible GPU |
| `nccl_bench` | all ranks (collective) | torch.distributed all-reduce bus bandwidth, 8 MiB to 1 GiB |
| `hf_smoke` | rank 0 | HF model pull + GPU inference (sentence-transformers/all-MiniLM-L6-v2) |

Pass criteria are encoded in `aggregate.py`:

- per-GPU TFLOPs >= 600 bf16 (H200 spec ~1000, threshold conservative)
- peak NCCL bus bandwidth >= 50 GB/s (well below what NDR IB delivers)
- HF model load <= 60 s

These thresholds are intentionally loose for first runs. Tighten once we have a few reference numbers.

## Run

From the repo root on the cluster:

```bash
sbatch sbatch/validate.sbatch
```

The job uses 2 nodes x 2 GPUs = 4 GPUs total (within the QOS cap), 30-minute wall clock, 128 GB RAM/node. NGC image cache is per-node, so pre-warm `worker-1` if the multi-node pull would otherwise take ~15 min on the second node.

Outputs land in `results/validation/<JOB_ID>/`:

- `gpu_info.json`, `gpu_compute_rank{0..3}.json`, `nccl_bench_rank{0..3}.json`, `hf_smoke.json`: raw per-check data
- `report.json`: machine-readable aggregate with `overall_pass: true|false`
- `report.md`: same data, human-readable. This is what to paste into the demo.

## Re-running

Each invocation writes to a fresh `results/validation/<JOB_ID>/` directory, so reruns don't clobber previous data. Point at the right job id to compare.

## Where to look first

1. `logs/validate-<JOB_ID>.out`: full stdout/stderr including NCCL init logs.
2. `results/validation/<JOB_ID>/report.md`: pass/fail summary with numbers.
3. If `nccl_bench` fails or shows low bandwidth, check the `NCCL_DEBUG=INFO` lines in the log for which transport NCCL selected (`IB` for inter-node, `P2P/NVLink` for intra-node).
