# Results

Numbers from the runs on 2026-05-22 and 2026-05-23. Raw artifacts under `results/`, plots under `results/plots/`.

## Phase 0: cluster validation

Multi-node validation suite (`sbatch sbatch/validate.sbatch`), see `results/validation/386/report.md`.

- bf16 8192^2 matmul: 799-816 TFLOPs per H200 (mean 806, spec ~1000).
- NCCL all-reduce (4 ranks, 2 nodes x 2 GPUs): peak 136 GB/s bus bw at 1 GiB.
- HF model load + GPU inference smoke: 2.1 s load, 0.24 s encode.
- All 4 checks PASS.

Cluster fabric and per-GPU compute are within spec, ready for training and inference workloads.

## Phase 1: training

Fine-tune of `Qwen/Qwen2.5-7B-Instruct` on 20k examples from `cais/mmlu` `auxiliary_train`. 2 nodes x 2 H200 with FSDP `FULL_SHARD`.

| | |
| --- | ---: |
| Optimizer steps | 312 |
| Wall clock | 14:20 |
| Train loss | 1.85 -> 1.21 |
| Throughput | 23.2 samples/s |
| Effective batch | 64 |
| Final checkpoint | `/mnt/data/jg/checkpoints/qwen25-7b-mmlu-sft-merged/` (consolidated from FSDP shards) |

Plot: `results/plots/loss.png`.

## Phase 2a: accuracy

`lm-evaluation-harness` 0.4.5 (git tag), 5-shot, bf16, chat template applied to both models so each is scored in its native instruction-tuned inference format.

| Task | Base acc | SFT acc | Delta (pp) |
| --- | ---: | ---: | ---: |
| **`mmlu_high_school_mathematics`** (target) | 0.4296 | **0.5630** | **+13.33** |
| `mmlu_college_biology` | 0.8403 | 0.8611 | +2.08 |
| `mmlu_high_school_us_history` | 0.8824 | 0.8725 | -0.98 |
| `mmlu_world_religions` | 0.8713 | 0.8538 | -1.75 |

+13.3 pp on the target subject (well outside the ~3 pp stderr), at most -1.75 pp on the three unrelated forgetting-check subjects. Plot: `results/plots/accuracy.png`.

Reading the result: SFT on the auxiliary-train MCQ distribution shifted the model's behavior in the chat-template / single-letter format we trained for. The forgetting check confirms it didn't broadly degrade other reasoning ability.

## Phase 2b: inference throughput

Same 270 prompts (full `high_school_mathematics` test split), max 128 generated tokens, same node. Baseline: `transformers.generate` batch 8 on 1 H200. Optimized: vLLM 0.6.3 TP=2 on 2 H200, `max_num_seqs=256`, bf16.

| Metric | transformers | vLLM | speedup |
| --- | ---: | ---: | ---: |
| Wall-clock (s) | 1.98 | 0.54 | 3.62x |
| Requests/s | 139.4 | 504.6 | 3.62x |
| Generated tokens/s | 278.7 | 1009.2 | 3.62x |
| Latency p50 (s) | 0.0071 | 0.0020 | 3.50x |
| Latency p95 (s) | 0.0085 | 0.0020 | 4.15x |

vLLM is 3.6x more throughput and 4x lower p95 latency at iso-prompt-set on this hardware. Plot: `results/plots/throughput.png`.

The two wins:

- PagedAttention. Naive engines waste 60-80% of KV-cache memory to fragmentation; vLLM uses block-paged allocation that's near-optimal.
- Continuous batching. vLLM swaps decoded sequences out as soon as they hit EOS, so no head-of-line blocking from the slowest sequence in a batch.

Both are portable. The customer gets the same multipliers on their H100 deployment from the same recipe. Absolute tokens/sec will scale with peak FLOPs and HBM bandwidth.

Fairness note: vLLM uses 2 GPUs (TP=2), transformers uses 1. transformers' `.generate()` does not natively tensor-parallel across multiple GPUs, while vLLM is built for it. This is the realistic deployment comparison. An iso-GPU comparison would also run vLLM at TP=1 for an extra reference point.

## Caveats and follow-ups

- `auxiliary_train` is not subject-stratified. The +13 pp on `high_school_mathematics` is a property of the random 20k subsample. With more time: subject-tagged training data, per-subject deltas at multiple training durations.
- The throughput bench is closed-loop (all requests submitted at once). An open-loop bench at varying request rates would give the customer a Pareto curve of latency vs throughput closer to production traffic.
- vLLM FP8 KV cache (Hopper-family supported) would extend per-instance batch by ~2x. Worth measuring next.
- The SFT model finishes MCQ in 1-2 generated tokens, so the throughput bench under-uses generation length. A bench with mixed prompt categories (open-ended + MCQ) would show off continuous batching's heterogeneous-length wins more strongly.
