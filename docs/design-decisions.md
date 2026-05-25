# Design decisions

The technical choices behind the PoC, with the reasoning. Written for the customer's ML eng team.

## 1. Model: `Qwen/Qwen2.5-7B-Instruct`

| Constraint | How the choice meets it |
| --- | --- |
| Full fine-tune on <=4 H200 | 7B params bf16 (~16 GB weights) + AdamW state (~48 GB) shards comfortably across 4 GPUs under FSDP FULL_SHARD. Per-GPU peak memory observed well under H200's 140 GB. |
| Open license, no gating | Apache-2.0. No HF token approval, no acceptance prompts. |
| Strong base on MMLU so the delta is meaningful | Reported base ~74% MMLU overall. High enough that movement in one subject is interpretable, low enough that there's room to improve. |
| Downstream tooling | Supported by vLLM and lm-evaluation-harness without custom adapters. |

Alternatives considered:

- Llama-3.1-8B-Instruct: comparable, but requires HF token + license acceptance. Adds a failure mode to the demo.
- Mistral-7B-Instruct-v0.3: older, weaker base MMLU score. Less to show in the delta plot.
- 70B with QLoRA: would be a more impressive "scale" demo but needs more than 4 GPUs to fine-tune reliably even with QLoRA, plus adapter-merging complexity.

## 2. Fine-tuning stack: `transformers` + `trl.SFTTrainer` + `accelerate`

- `SFTTrainer` handles chat-template application and completion-only loss masking, so the script stays small.
- `accelerate` provides the FSDP plumbing without dropping into raw `torch.distributed`.
- This is the same stack the customer's ML engineers would reach for, which keeps the reproduction guide short.

Pinned versions: `trl==0.11.4`, `accelerate==1.0.1`, `transformers==4.46.0`, `datasets==3.0.2`.

## 3. Distributed strategy: FSDP (FULL_SHARD), 2 nodes x 2 GPUs

| Knob | Value |
| --- | --- |
| Sharding | `FULL_SHARD` (params, grads, optimizer state all sharded across 4 ranks) |
| Wrap policy | `TRANSFORMER_BASED_WRAP` with `Qwen2DecoderLayer` (one FSDP unit per transformer block) |
| State dict | `SHARDED_STATE_DICT` for intermediate saves; consolidated post-training via `accelerate.utils.fsdp_utils.merge_fsdp_weights` |
| Precision | bf16 model + optimizer; H200 has native bf16 support |
| Gradient checkpointing | On. Trades ~25% step time for activation memory headroom. |

Why FSDP rather than DDP: with 4 ranks, DDP would replicate the full 7B model + AdamW state on each GPU. Workable for inference but wasteful for training, and it forces a refactor if the customer scales to a larger model later. FSDP keeps the same code path open for 13B / 32B with no structural changes.

Why not pipeline or tensor parallelism: 7B is small enough that intra-layer parallelism adds complexity for no benefit. Data-parallel sharding is the right abstraction at this size.

## 4. Data: `cais/mmlu` `auxiliary_train`, 20k subsample

The brief says "fine-tune against at least one category of your choosing from the cais/mmlu dataset." MMLU has no real training set; the options are:

| Option | Pros | Cons |
| --- | --- | --- |
| `auxiliary_train` (99k MCQs) | Real training data, covers many MMLU-adjacent topics | Not organized by subject, so "category" is a loose fit |
| Per-subject dev + validation | Strict reading of "category" | Tiny (~85-90 examples), loss signal washes out |
| `auxiliary_train` filtered to one subject | Combines both | `auxiliary_train` has no subject field, filtering needs synthetic labels |

Picked Option 1 with a 20k random subsample for fast iteration. The subject we evaluate on (the one the model should be measurably better at) is `high_school_mathematics`. Base accuracy under chat-template eval is ~43%, so there's clear room above noise.

The "choose one of the following options" wording in the brief is ambiguous. Phase 2 explicitly references "the model you trained in Phase 1," so we treat Phase 1 + Phase 2 as a required sequence, not a fork.

## 5. Prompt template

The brief lists "prompt" as a justification target. We use a 3-turn chat with the model's native template:

- System: "You are a careful assistant. Answer the multiple-choice question by selecting exactly one of A, B, C, or D."
- User: `{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:`
- Assistant: single letter (`A` / `B` / `C` / `D`).

Two reasons for this shape:

1. Train/eval distribution match. We score with `lm_eval --apply_chat_template`, so both base and SFT are evaluated in the same instruction-tuned format the model was trained for. Training with verbose chain-of-thought answers would have hurt single-letter logprob scoring.
2. Concentrated loss signal. With completion-only loss masking (TRL default for the `messages` format), each gradient step is computed on one answer token. No noise from generating reasoning we never score.

## 6. Hyperparameters

| | Value | Why |
| --- | ---: | --- |
| Epochs | 1 | 20k x bs 64 = 312 steps. Enough for a clear delta on one subject without overfitting. |
| Effective batch | 64 (per_device 4 x accum 4 x world 4) | Common SFT range, large enough to dampen MCQ noise. |
| LR | 2e-5 | Standard SFT LR for 7B instruction-tuned models. Higher forgets the base, lower under-trains in 1 epoch. |
| Scheduler | Cosine, 3% warmup | Warmup prevents early grad-norm spikes that hurt FSDP comm timings. |
| Optimizer | `adamw_torch_fused` | Single CUDA kernel per param group, faster than the unfused path. |
| Sequence length | 1024 | MMLU prompts + answer fit easily; avoids wasted compute on padding. |

## 7. Evaluation

- Tool: `lm-evaluation-harness` (git tag v0.4.5). The PyPI 0.4.5 wheel is missing task YAML files; git install ships them.
- 5-shot, bf16, `--apply_chat_template` for both base and SFT.
- Target subject: `mmlu_high_school_mathematics`.
- Forgetting check: `mmlu_college_biology`, `mmlu_world_religions`, `mmlu_high_school_us_history` (chosen because they are unrelated to math reasoning).

Why include the forgetting check: a +5 pp gain on math that costs -10 pp on biology is not a real improvement. The forgetting plot makes the trade-off (or lack of one) honest.

## 8. Inference performance optimization

We picked throughput rather than latency. The customer's "should we reserve 512 H100s for 6 months" question is a capacity question, and throughput maps to it directly.

| Engine | Settings |
| --- | --- |
| Baseline | `transformers.generate()`, batch 8, 1 H200, bf16 |
| Optimized | vLLM 0.6.3, tensor-parallel 2, `max_num_seqs=256`, bf16 |

vLLM's win comes from two algorithmic changes, both portable to the customer's H100 deployment:

- PagedAttention removes KV-cache fragmentation. Naive engines waste 60-80% of KV memory; vLLM is near-optimal.
- Continuous batching never leaves a GPU idle waiting for the slowest sequence in a batch to finish.

Reported speedup is the customer-transferable artifact. Absolute tokens/sec scale with peak FLOPs and HBM bandwidth.

Fairness note: vLLM uses 2 GPUs (TP=2), transformers uses 1. transformers' `.generate()` does not natively tensor-parallel across GPUs, so this is the realistic deployment comparison. An iso-GPU comparison would also run vLLM at TP=1.

## 9. What we did not do

- LoRA / PEFT. Faster, but it would obscure the FSDP story this PoC is meant to validate. We recommend LoRA for the customer's own iteration loop once the infra is proven.
- DeepSpeed. Comparable to FSDP at this scale, but FSDP is the PyTorch-native path now, which means less drift risk over a 6-month engagement.
- Multi-epoch training. With 20k examples and a strong base, a second epoch on this subset is mostly memorisation.
- Curriculum or packed sequences. TRL's packing fragments completion-only loss masking. Left off to keep the loss interpretable.
