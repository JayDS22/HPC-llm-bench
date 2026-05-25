#!/usr/bin/env python3
"""Build a JSONL SFT corpus from cais/mmlu auxiliary_train.

auxiliary_train is ~99k MCQs that the MMLU authors curated as supplementary
training data. We subsample (default 20k), format each as a 3-turn chat
(system, user, assistant) and write JSONL for TRL SFTTrainer.

Single-letter assistant answer matches the lm-eval-harness MMLU scoring
shape (logprob on the letter token) so train and eval distributions line up.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the multiple-choice question by "
    "selecting exactly one of A, B, C, or D."
)

LETTERS = ["A", "B", "C", "D"]


def _fmt_user(question: str, choices: list[str]) -> str:
    return (
        f"{question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        f"Answer:"
    )


def _to_record(ex: dict) -> dict | None:
    q = (ex.get("question") or "").strip()
    cs = ex.get("choices") or []
    a = ex.get("answer")
    if not q or len(cs) != 4 or a is None or not (0 <= int(a) < 4):
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _fmt_user(q, cs)},
            {"role": "assistant", "content": LETTERS[int(a)]},
        ]
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Path to write JSONL")
    p.add_argument("--n", type=int, default=20000,
                   help="Subset size (0 = use all auxiliary_train)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    print(f"loaded cais/mmlu auxiliary_train: {len(ds)} examples")

    if args.n and args.n < len(ds):
        rng = random.Random(args.seed)
        idxs = sorted(rng.sample(range(len(ds)), args.n))
        ds = ds.select(idxs)
        print(f"subsampled to {len(ds)} (seed={args.seed})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_kept = 0
    n_dropped = 0
    with out_path.open("w") as f:
        for ex in ds:
            r = _to_record(ex)
            if r is None:
                n_dropped += 1
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_kept += 1
    print(f"wrote {n_kept} records to {out_path} (dropped {n_dropped} malformed)")


if __name__ == "__main__":
    main()
