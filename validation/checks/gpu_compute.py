#!/usr/bin/env python3
"""Per-GPU bf16 matmul TFLOPs.

Each task picks its node-local GPU via SLURM_LOCALID and times a square
matmul. H200 sustains ~700-800 TFLOPs bf16 (spec ~1000).
"""
from __future__ import annotations
import argparse
import json
import os
import time

import torch


def benchmark(local_rank: int, size: int = 8192, iters: int = 20, warmup: int = 5,
              dtype: torch.dtype = torch.bfloat16) -> dict:
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    # Warmup
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters
    flops = 2.0 * size ** 3  # matmul is ~2*N^3 FLOPs
    return {
        "matmul_size": size,
        "dtype": str(dtype).replace("torch.", ""),
        "iters": iters,
        "time_per_iter_s": elapsed,
        "tflops": flops / elapsed / 1e12,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=8192)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    info = {
        "rank": int(os.environ.get("SLURM_PROCID", 0)),
        "local_rank": local_rank,
        "node": os.environ.get("SLURMD_NODENAME", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "result": benchmark(local_rank=local_rank, size=args.size, iters=args.iters),
    }
    with open(args.out, "w") as f:
        json.dump(info, f, indent=2)
    print(
        f"gpu_compute rank={info['rank']} node={info['node']} "
        f"gpu={info['gpu_name']} tflops={info['result']['tflops']:.1f}"
    )


if __name__ == "__main__":
    main()
