#!/usr/bin/env python3
"""NCCL all-reduce bus-bandwidth sweep, torch.distributed env://.

sbatch sets MASTER_ADDR and MASTER_PORT for rendezvous; we bridge
SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID to RANK / WORLD_SIZE / LOCAL_RANK
and sweep 8 MiB .. 1 GiB. Bus bandwidth = 2*(n-1)/n * size / time.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import time

import torch
import torch.distributed as dist


def setup() -> int:
    # Bridge Slurm env vars to what torch.distributed env:// expects.
    os.environ.setdefault("RANK", os.environ["SLURM_PROCID"])
    os.environ.setdefault("WORLD_SIZE", os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    # Bind to the node-local GPU before NCCL init so the P2P transport
    # sees a coherent set of sibling GPUs on the same node.
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=dt.timedelta(seconds=120),
    )
    return local_rank


def bench(local_rank: int, size_bytes: int, iters: int = 20, warmup: int = 5) -> dict:
    n = size_bytes // 4  # fp32 elements
    x = torch.ones(n, dtype=torch.float32, device=f"cuda:{local_rank}")
    for _ in range(warmup):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters
    world = dist.get_world_size()
    bus_bw = (2.0 * size_bytes * (world - 1) / world) / elapsed
    return {
        "size_bytes": size_bytes,
        "size_mib": size_bytes / (1024 * 1024),
        "time_per_iter_s": elapsed,
        "bus_bandwidth_gbps": bus_bw / 1e9,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    local_rank = setup()
    rank = dist.get_rank()
    world = dist.get_world_size()

    sizes = [8 << 20, 64 << 20, 256 << 20, 1 << 30]  # 8 MiB, 64 MiB, 256 MiB, 1 GiB
    results = [bench(local_rank, s) for s in sizes]

    info = {
        "rank": rank,
        "world_size": world,
        "node": os.environ.get("SLURMD_NODENAME", ""),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(info, f, indent=2)

    if rank == 0:
        for r in results:
            print(
                f"nccl_bench size={r['size_mib']:6.0f} MiB  "
                f"busbw={r['bus_bandwidth_gbps']:6.1f} GB/s  "
                f"time={r['time_per_iter_s']*1000:.2f} ms"
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
