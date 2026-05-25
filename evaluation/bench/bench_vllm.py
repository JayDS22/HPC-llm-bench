#!/usr/bin/env python3
"""Optimized inference throughput: vLLM (PagedAttention + continuous batching).

Same prompt set and max_new_tokens as bench_hf.py for an apples-to-apples
comparison on the same hardware. Tensor parallelism set via CLI flag.
Reports the same metric set as bench_hf for the head-to-head plot.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path


def _load_prompts(path: Path, n: int) -> list[list[dict]]:
    out = []
    for i, line in enumerate(path.open()):
        if i >= n:
            break
        d = json.loads(line)
        out.append([
            {"role": "system", "content": d["system"]},
            {"role": "user", "content": d["user"]},
        ])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--kv-cache-dtype", default="auto",
                   help="auto | fp8 (only on supported GPUs)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    msgs = _load_prompts(Path(args.prompts), args.n)
    prompts_text = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in msgs
    ]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.85,
    )
    sp = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)

    _ = llm.generate(prompts_text[:8], sp)  # warmup

    t0 = time.perf_counter()
    outs = llm.generate(prompts_text, sp)
    total = time.perf_counter() - t0

    n_prompts = len(outs)
    n_prompt_tokens = sum(len(o.prompt_token_ids) for o in outs)
    n_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    # vLLM doesn't expose per-request finish times by default. Approximate
    # per-request latency by total / n_prompts (under closed-loop submission
    # this is the mean steady-state request latency).
    per_request = sorted([total / n_prompts] * n_prompts)
    p50 = per_request[int(0.50 * len(per_request))]
    p95 = per_request[int(0.95 * len(per_request))]

    result = {
        "engine": "vllm",
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_seqs": args.max_num_seqs,
        "kv_cache_dtype": args.kv_cache_dtype,
        "n_prompts": n_prompts,
        "max_new_tokens": args.max_new_tokens,
        "total_s": total,
        "requests_per_s": n_prompts / total,
        "prompt_tokens_per_s": n_prompt_tokens / total,
        "generated_tokens_per_s": n_gen_tokens / total,
        "latency_p50_s": p50,
        "latency_p95_s": p95,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
