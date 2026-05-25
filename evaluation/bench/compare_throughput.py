#!/usr/bin/env python3
"""Take two bench JSONs (HF baseline + vLLM) and emit a markdown comparison."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True)
    p.add_argument("--vllm", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args()

    hf = json.loads(Path(args.hf).read_text())
    vl = json.loads(Path(args.vllm).read_text())

    def speedup(a: float, b: float) -> str:
        return f"{(b / a):.2f}x" if a else "-"

    lines = [
        "# Inference throughput: transformers vs vLLM",
        "",
        f"- Model: `{hf['model']}`",
        f"- Prompts: {hf['n_prompts']}, max_new_tokens: {hf['max_new_tokens']}",
        "",
        "| Metric | transformers | vLLM | speedup |",
        "| --- | ---: | ---: | ---: |",
        f"| Wall-clock (s) | {hf['total_s']:.1f} | {vl['total_s']:.1f} | {speedup(vl['total_s'], hf['total_s'])} |",
        f"| Requests/s | {hf['requests_per_s']:.2f} | {vl['requests_per_s']:.2f} | {speedup(hf['requests_per_s'], vl['requests_per_s'])} |",
        f"| Generated tokens/s | {hf['generated_tokens_per_s']:.1f} | {vl['generated_tokens_per_s']:.1f} | {speedup(hf['generated_tokens_per_s'], vl['generated_tokens_per_s'])} |",
        f"| Latency p50 (s) | {hf['latency_p50_s']:.3f} | {vl['latency_p50_s']:.3f} | {speedup(vl['latency_p50_s'], hf['latency_p50_s'])} |",
        f"| Latency p95 (s) | {hf['latency_p95_s']:.3f} | {vl['latency_p95_s']:.3f} | {speedup(vl['latency_p95_s'], hf['latency_p95_s'])} |",
        "",
        f"transformers: batch_size={hf['batch_size']}, single GPU.",
        f"vLLM: TP={vl['tensor_parallel_size']}, max_num_seqs={vl['max_num_seqs']}, kv_cache_dtype={vl['kv_cache_dtype']}.",
        "vLLM wins from PagedAttention (no KV-cache fragmentation) and continuous batching.",
        "Same recipe applies to the customer's H100 box; absolute numbers scale with peak FLOPs and HBM bandwidth.",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
