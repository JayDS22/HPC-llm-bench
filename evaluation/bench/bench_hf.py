#!/usr/bin/env python3
"""Baseline inference throughput: transformers.generate on one GPU.

Loads the model, formats prompts with the model's chat template, runs
generation in fixed batches, and reports requests/sec, prompt tokens/sec,
generated tokens/sec, and p50/p95 latency. JSON output pairs with
bench_vllm.py for the head-to-head plot.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # decoder-only generation needs left padding

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()

    msgs = _load_prompts(Path(args.prompts), args.n)
    prompts_text = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in msgs
    ]

    bs = args.batch_size
    batches = [prompts_text[i:i + bs] for i in range(0, len(prompts_text), bs)]

    # Warmup
    with torch.inference_mode():
        for _ in range(args.warmup):
            enc = tok(batches[0], return_tensors="pt", padding=True, truncation=True).to("cuda")
            _ = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)

    torch.cuda.synchronize()

    per_batch_latency: list[float] = []
    n_prompts = 0
    n_prompt_tokens = 0
    n_gen_tokens = 0

    t0 = time.perf_counter()
    with torch.inference_mode():
        for batch in batches:
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
            bs_t = time.perf_counter()
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            torch.cuda.synchronize()
            per_batch_latency.append(time.perf_counter() - bs_t)
            n_prompts += len(batch)
            n_prompt_tokens += int(enc["attention_mask"].sum().item())
            n_gen_tokens += int((out.shape[1] - enc["input_ids"].shape[1]) * out.shape[0])
    total = time.perf_counter() - t0

    # Per-request latency approximated as per-batch latency / batch_size.
    per_request = sorted(l / bs for l in per_batch_latency for _ in range(bs))
    p50 = per_request[int(0.50 * len(per_request))]
    p95 = per_request[int(0.95 * len(per_request))]

    result = {
        "engine": "transformers.generate",
        "model": args.model,
        "batch_size": bs,
        "n_prompts": n_prompts,
        "max_new_tokens": args.max_new_tokens,
        "total_s": total,
        "requests_per_s": n_prompts / total,
        "prompt_tokens_per_s": n_prompt_tokens / total,
        "generated_tokens_per_s": n_gen_tokens / total,
        "latency_p50_s": p50,
        "latency_p95_s": p95,
        "peak_gpu_mem_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
