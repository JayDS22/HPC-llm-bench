#!/usr/bin/env python3
"""HF smoke test: pull a small public model, run a GPU embed.

Confirms HF token auth, HF cache writability on shared scratch, and that
GPU-backed inference works inside the container.
"""
from __future__ import annotations
import argparse
import json
import os
import time


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF model id",
    )
    args = p.parse_args()

    sentences = [
        "Validation runs on shared cluster.",
        "H200 GPUs with 140 GB HBM.",
        "Pyxis pulls images from NGC.",
        "NCCL all-reduce over InfiniBand.",
    ] * 4

    timings: dict = {}

    t0 = time.perf_counter()
    from sentence_transformers import SentenceTransformer
    timings["import_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = SentenceTransformer(args.model, device="cuda")
    timings["load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    emb = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    timings["encode_s"] = time.perf_counter() - t0

    info = {
        "model": args.model,
        "hf_home": os.environ.get("HF_HOME", ""),
        "num_sentences": len(sentences),
        "embedding_dim": int(emb.shape[1]),
        "timings_s": timings,
    }
    with open(args.out, "w") as f:
        json.dump(info, f, indent=2)
    print(
        f"hf_smoke model={args.model} sentences={len(sentences)} "
        f"dim={emb.shape[1]} load={timings['load_s']:.1f}s "
        f"encode={timings['encode_s']:.2f}s"
    )


if __name__ == "__main__":
    main()
