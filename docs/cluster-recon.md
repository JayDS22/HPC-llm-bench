# Cluster recon

Captured 2026-05-22 from the login node via `scontrol`, `sacctmgr`, and a 5-minute `srun` on a compute node. Raw log: `~/recon/recon-2026-05-22.log` on the cluster.

## Hardware

- 2 worker nodes: `worker-0`, `worker-1`.
- Per node: 8x NVIDIA H200 SXM (`gpu:nvidia_h200:8`), AMD EPYC-Genoa (128 vCPU, 64 physical cores, 2 sockets), 1438 GB RAM.
- Node features: `gpu_h200_sxm,network_ssd`.
- Interconnect: InfiniBand. Each node has an `ib_pod` value in its Slurm `Extra` field (one IB pod per node).
- Login node: `login-0`, 32 vCPU, 125 GB RAM, no GPU.
- OS: Ubuntu 24.04.3 LTS, NVIDIA-tuned kernel `6.11.0-1016-nvidia`.

Per-node compute inspection (`srun ~/inspect.sh` on worker-0):

- H200 SXM, 143,771 MiB HBM, 700 W TDP, no MIG. Driver 570.211.01, CUDA 12.8.
- 8 Mellanox ConnectX HCAs (`mlx5_0`..`mlx5_7`). Every Port 1 Active/LinkUp at 400 Gb/s (NDR IB). FW 28.39.3004.
- Aggregate per-node IB: 8 x 400 Gb/s = 3.2 Tb/s.
- GPU-NIC topology: each GPU has a closely paired NIC (`PIX` distance, one PCIe bridge), supporting GPUDirect RDMA without crossing the host bridge. 1:1 GPU:NIC.
- Local `/tmp`: ext4, 495 GB total, 464 GB free.
- 2 NUMA nodes per worker (CPUs 0-63 and 64-127), ~800 GB RAM each, distances 10/20.

## Slurm

- Version 25.11.3.
- One partition, `earlytalent` (default), time limit 12:00:00.
- Both worker nodes in `IDLE+CLOUD` at recon time.
- User QOS `gpulimit`: `gres/gpu=4` (max 4 GPUs per job), `MaxJobs=2` (2 concurrent jobs per user).
- `~/.slurm/defaults` contains `cpu-bind=verbose`, applied to every `srun` automatically.

## Storage

| Mount | Type | Free | Use |
| --- | --- | --- | --- |
| `/home/jg` | NFS, shared across nodes | 3.5 TB of 3.6 TB | code, configs, small artifacts |
| `/mnt/data/jg` | virtiofs, shared | ~20 TB | model weights, HF cache, checkpoints |
| `/mnt/memory` | tmpfs | 112 GB | fast scratch for temp files |

No `quota` command, quotas unenforced. Shared cluster, so keep usage reasonable.

## Container runtime

- Pyxis + enroot 4.0.1 wired into Slurm. Confirmed via `srun --help | grep container` showing `[pyxis]`-tagged flags.
- Pull and run any OCI image directly from `srun`/`sbatch`:

  ```bash
  srun --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
       --container-mounts=/mnt/data/jg:/scratch,/home/jg:/home/jg \
       bash -c 'nvidia-smi'
  ```

- Docker binary at `/usr/bin/docker` but the daemon does not run on the login node. All container work goes through Pyxis on compute nodes.
- Apptainer/Singularity not installed.
- No `module` system. Environments live in containers, not LMOD.

### Pyxis smoke test (2026-05-22)

Pulled `docker://nvcr.io/nvidia/pytorch:24.10-py3`. First pull is ~11.5 GB; squashfs build plus extraction needed ~15 min wall clock. Subsequent runs on the same node hit the cache in seconds.

Inside the container:

- Ubuntu 22.04.5 LTS.
- PyTorch `2.5.0a0+e000cf0ad9.nv24.10`, CUDA 12.6.
- `nccl-tests` prebuilt at `/usr/local/bin/all_reduce_perf` (no need to build from source).
- Bind mount `/mnt/data/jg -> /scratch` works.

Cache locality: the squashfs is cached per node under `/var/cache/enroot-container-images/$UID/`. Pre-warm `worker-1` before submitting any `-N 2` job to avoid a 15-minute cold pull on the second node. Killing a srun mid-extract leaves a corrupt `.squashfs` in that path; remove it manually before retrying.

## Implications for the assignment

- Validation container is not a `.sif` we build. It's a Slurm job that pulls an NGC image via Pyxis and runs checks inside.
- Set `HF_HOME=/mnt/data/jg/hf_cache` so model weights stay off `/home`.
- Job sizing constraints: <=4 GPUs/job, <=2 concurrent jobs, 12h cap. Multi-stage pipelines (eval-while-training) need to live inside those bounds.
- Per-node memory: ~1.4 TB. `--mem=256G` is comfortable when justified.

## Validation run (2026-05-22, job 386)

`sbatch sbatch/validate.sbatch` end-to-end in 53 s on 2 nodes x 2 GPUs (4 ranks). All checks PASS. Full report at `results/validation/386/report.md`.

| Check | Result |
| --- | --- |
| `gpu_info` | 2 GPUs/node visible, driver 570.211.01, H200 SXM, 140 GB HBM each. |
| `gpu_compute` (bf16 8192^2 matmul) | 799-816 TFLOPs per rank (mean 806). H200 spec ~1000, threshold 600. |
| `nccl_bench` (torch.distributed all-reduce, 4 ranks) | peak 136 GB/s bus bw at 1 GiB; 96 GB/s @ 256 MiB; 89 GB/s @ 64 MiB. Threshold 50. |
| `hf_smoke` (sentence-transformers/all-MiniLM-L6-v2) | load 2.1 s, encode 16 sentences 0.24 s. Confirms HF Hub egress + token auth. |

Notes from getting it green:

- GPU binding for NCCL P2P: each task on a node should see all node-local GPUs (do not use `--gpus-per-task=1`); each Python process then claims its own via `torch.cuda.set_device(SLURM_LOCALID)`. Restricting `CUDA_VISIBLE_DEVICES` per task breaks intra-node P2P with `Cuda failure 101 'invalid device ordinal'`.
- `MASTER_ADDR` via hostname works. `scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1` returns `worker-0` etc., and PyTorch resolves it for NCCL rendezvous.
- NCCL auto-detected all 8 ConnectX HCAs and enabled GPU Direct RDMA on every one. No NCCL tuning env vars required.

Cluster is ready for Phase 1 multi-node training.
