# Cluster validation report - overall PASS

## gpu_info - PASS
- 2 GPU(s) visible on rank-0 host. First: **NVIDIA H200**, driver 570.211.01, 143771 MiB HBM.

## gpu_compute - PASS
- bf16 matmul TFLOPs across 4 rank(s): min 799.4, mean 806.2, max 816.4 (threshold 600).
  - rank 0 (worker-0): 800.5 TFLOPs
  - rank 1 (worker-0): 799.4 TFLOPs
  - rank 2 (worker-1): 816.4 TFLOPs
  - rank 3 (worker-1): 808.5 TFLOPs

## nccl_bench - PASS
- all-reduce world_size=4, peak bus bandwidth 136.2 GB/s (threshold 50).
  - 8 MiB -> 69.1 GB/s
  - 64 MiB -> 88.8 GB/s
  - 256 MiB -> 95.5 GB/s
  - 1024 MiB -> 136.2 GB/s

## hf_smoke - PASS
- load 2.1s, encode 0.24s, embedding dim 384 (threshold 60s load).
