# Cluster validation report - overall FAIL

## gpu_info - PASS
- 2 GPU(s) visible on rank-0 host. First: **NVIDIA H200**, driver 570.211.01, 143771 MiB HBM.

## gpu_compute - FAIL
- bf16 matmul TFLOPs across 4 rank(s): min 356.0, mean 585.8, max 810.1 (threshold 400).
  - rank 0 (worker-0): 799.1 TFLOPs
  - rank 1 (worker-0): 810.1 TFLOPs
  - rank 2 (worker-1): 378.1 TFLOPs
  - rank 3 (worker-1): 356.0 TFLOPs

## hf_smoke - PASS
- load 4.8s, encode 0.25s, embedding dim 384 (threshold 60s load).
