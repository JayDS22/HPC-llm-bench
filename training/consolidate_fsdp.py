#!/usr/bin/env python3
"""Consolidate a sharded-FSDP checkpoint into an HF-loadable directory.

Training writes SHARDED_STATE_DICT for fast per-rank saves, so the final
checkpoint lives at checkpoint-XXX/pytorch_model_fsdp_0/ in distributed-
checkpoint format. lm-eval-harness and AutoModelForCausalLM.from_pretrained
can't load that directly. accelerate's merge_fsdp_weights strips the FSDP
key prefixes (_fsdp_wrapped_module.*) and emits standard safetensors.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

from accelerate.utils.fsdp_utils import merge_fsdp_weights
from transformers import AutoConfig, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True,
                   help="HF id of the original base model (for config + tokenizer)")
    p.add_argument("--sharded-dir", required=True,
                   help="Path to pytorch_model_fsdp_0/ from the FSDP checkpoint")
    p.add_argument("--tokenizer-dir", default=None,
                   help="Directory holding the saved tokenizer files. "
                        "Defaults to the parent of --sharded-dir's parent "
                        "(i.e. the trainer output_dir).")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"merging FSDP shards from {args.sharded_dir}")
    merge_fsdp_weights(
        args.sharded_dir,
        str(out_dir),
        safe_serialization=True,
        remove_checkpoint_dir=False,
    )

    print(f"writing config from {args.base_model}")
    cfg = AutoConfig.from_pretrained(args.base_model)
    cfg.save_pretrained(out_dir)

    tok_src = args.tokenizer_dir or str(Path(args.sharded_dir).parent.parent)
    tok = AutoTokenizer.from_pretrained(tok_src)
    tok.save_pretrained(out_dir)

    safetensors = list(out_dir.glob("*.safetensors"))
    print(f"safetensors in output: {len(safetensors)}")
    for s in safetensors[:6]:
        print(f"  {s.name}")
    print("done")


if __name__ == "__main__":
    main()
