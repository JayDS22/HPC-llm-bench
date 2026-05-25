# Cluster validation report - overall PASS

## gpu_info - PASS
- 1 GPU(s) visible on rank-0 host. First: **NVIDIA H200**, driver 570.211.01, 143771 MiB HBM.

## gpu_compute - PASS
- bf16 matmul TFLOPs across 4 rank(s): min 799.6, mean 806.9, max 816.3 (threshold 600).
  - rank 0 (worker-0): 810.6 TFLOPs
  - rank 1 (worker-0): 799.6 TFLOPs
  - rank 2 (worker-1): 816.3 TFLOPs
  - rank 3 (worker-1): 801.2 TFLOPs

## hf_smoke - PASS
- load 2.0s, encode 0.24s, embedding dim 384 (threshold 60s load).
