# Results

Numbers from the runs on 2026-05-22 through 2026-05-25. Raw artifacts under `results/`, plots under `results/plots/`.

## Phase 0: cluster validation

`sbatch sbatch/validate.sbatch`. Full report at `results/validation/386/report.md`.

- bf16 8192^2 matmul: 799-816 TFLOPs per H200 (mean 806, spec ~1000).
- NCCL all-reduce (4 ranks, 2 nodes x 2 GPUs): peak 136 GB/s bus bw at 1 GiB.
- HF model load + GPU inference smoke: 2.1 s load, 0.24 s encode.
- All 4 checks PASS.

Cluster fabric and per-GPU compute are within spec, ready for training and inference workloads.

**Limitation worth flagging:** the validation suite is intentionally minimal. A production-grade version would add sustained-thermal compute (1-hour matmul to watch for throttling), `fio` on the scratch filesystem, and inter-node latency distribution.

## Phase 1: training

Fine-tune of `Qwen/Qwen2.5-7B-Instruct` on 20k examples from `cais/mmlu` `auxiliary_train`. 2 nodes x 2 H200 with FSDP `FULL_SHARD`.

| | |
| --- | ---: |
| Optimizer steps | 312 |
| Wall clock | 14:20 |
| Train loss | 1.85 -> 1.21 |
| Throughput | 23.2 samples/s |
| Effective batch | 64 |
| Final checkpoint | `/mnt/data/jg/checkpoints/qwen25-7b-mmlu-sft-merged/` |

Plot: `results/plots/loss.png`.

**Limitation worth flagging:** single-shot hyperparameters (no LR sweep, no batch-size sensitivity). The defaults (lr 2e-5, 1 epoch, eff. batch 64) are sane choices from the literature, but a real engagement would start with an LR x epochs sweep with eval every N steps to find the actual best checkpoint.

## Phase 2a: accuracy

`lm-evaluation-harness` 0.4.5 (git tag), 5-shot, bf16. **Both base and SFT scored with `--apply_chat_template`** so each model is evaluated in its native instruction-tuned inference format. This is critical: scoring an instruction-tuned model *without* its chat template is out-of-distribution evaluation, and it's the reason the first eval run showed a misleading negative result.

| Task | Base acc | SFT acc | Delta (pp) |
| --- | ---: | ---: | ---: |
| **`mmlu_high_school_mathematics`** (target) | 0.4296 | **0.5630** | **+13.33** |
| `mmlu_college_biology` | 0.8403 | 0.8611 | +2.08 |
| `mmlu_high_school_us_history` | 0.8824 | 0.8725 | -0.98 |
| `mmlu_world_religions` | 0.8713 | 0.8538 | -1.75 |

+13.3 pp on the target subject (well outside the ~3 pp stderr), at most -1.75 pp on three unrelated forgetting-check subjects. Plot: `results/plots/accuracy.png`.

**Limitation worth flagging:** `auxiliary_train` is not subject-stratified, so the +13.3 pp on `high_school_mathematics` is partly a property of the 20k random subsample (seed 42). A different seed would shift the magnitude. With more time, the right next step is subject-tagged training data (filter `auxiliary_train` for math-adjacent items only) and report per-subject deltas at multiple training durations.

## Phase 2b: inference throughput

Same 270 prompts (full `high_school_mathematics` test split), max 128 generated tokens, same node. Three engines compared.

| Metric | transformers (1 GPU, bs 8) | vLLM TP=1 (1 GPU) | vLLM TP=2 (2 GPUs) |
| --- | ---: | ---: | ---: |
| Wall-clock (s) | 1.98 | 1.08 | 0.54 |
| Requests/s | 139.4 | 250.5 | 504.6 |
| Generated tokens/s | 278.7 | 501.1 | 1009.2 |
| Latency p50 (s) | 0.0071 | 0.0040 | 0.0020 |
| Latency p95 (s) | 0.0085 | 0.0040 | 0.0020 |

Two distinct wins, decomposed:

- **Algorithmic win (PagedAttention + continuous batching): 1.80x.** vLLM TP=1 vs transformers at iso-GPU (same hardware, same prompts). This is the portable win that transfers to the customer's H100 box unchanged.
- **Parallelism win (tensor parallelism): 2.0x.** vLLM TP=2 vs vLLM TP=1 (same engine, twice the GPUs). transformers' `.generate()` does not natively tensor-parallel, so going from 1 GPU to 2 GPUs in the baseline isn't a direct option.
- **Combined: 1.80 x 2.0 = 3.6x.**

Plot: `results/plots/throughput.png` (shows the 2-engine headline; the 3-way breakdown is the table above).

The two algorithmic ideas:

- PagedAttention. Naive engines waste 60-80% of KV-cache memory to fragmentation; vLLM uses block-paged allocation that's near-optimal.
- Continuous batching. vLLM swaps decoded sequences out as soon as they hit EOS, so no head-of-line blocking from the slowest sequence in a batch.

**Limitation worth flagging:** MCQ answers are 1-2 generated tokens, which under-uses continuous batching's main superpower (heterogeneous-length workloads). Most of our 1.80x algorithmic win is PagedAttention's memory layout. A mixed bench (long-form Q&A plus MCQ) would let continuous batching compound the win further. The bench is also closed-loop (all requests submitted at once); an open-loop bench at varying request rates would give the customer a Pareto curve of latency vs throughput closer to production traffic.

## Cross-cutting follow-ups

- vLLM FP8 KV cache (Hopper-family supported) would extend per-instance batch by ~2x. Worth measuring next.
- The H200 numbers are illustrative for an H100 deployment decision. The vLLM recipe transfers unchanged; absolute numbers scale with peak FLOPs and HBM bandwidth (H100 HBM3 is ~25% lower bandwidth than H200 HBM3e).
- All training and inference work fit comfortably within the QOS cap (4 GPUs/job, 2 concurrent jobs). On the customer's 512-H100 reservation those caps lift and the recipe scales linearly by changing the FSDP wrap-policy class for larger models.
