#!/usr/bin/env python3
"""Build a fixed prompt set for throughput benchmarks.

Reads a subset of MMLU test items and formats each as system + user MCQ
("Answer:" cue) matching the training format. JSONL output is loaded by
bench_hf.py and bench_vllm.py so the head-to-head comparison hits the same
prompts in the same order.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the multiple-choice question by "
    "selecting exactly one of A, B, C, or D."
)


def _fmt(question: str, choices: list[str]) -> str:
    return (
        f"{question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        f"Answer:"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="high_school_mathematics",
                   help="MMLU subject for test prompts")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ds = load_dataset("cais/mmlu", args.subject, split="test")
    print(f"loaded {len(ds)} {args.subject} test examples")
    n = min(args.n, len(ds))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i in range(n):
            ex = ds[i]
            prompt = {
                "system": SYSTEM_PROMPT,
                "user": _fmt(ex["question"], ex["choices"]),
            }
            f.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    print(f"wrote {n} prompts to {out}")


if __name__ == "__main__":
    main()
