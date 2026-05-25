#!/usr/bin/env python3
"""SFT of an instruction-tuned LLM on MMLU SFT JSONL.

Launch via `accelerate launch` with training/config/accelerate_fsdp.yaml for
multi-GPU FSDP; plain `python ...` works for a single-GPU smoke run.

Dataset uses TRL's conversational format ("messages" field). SFTTrainer
applies the model's chat template and masks loss to assistant tokens only.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    train_ds = load_dataset("json", data_files=cfg["dataset_path"], split="train")
    print(f"[train_sft] loaded {len(train_ds)} examples from {cfg['dataset_path']}")

    tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
    )

    sft_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("epochs", 1),
        max_steps=cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg.get("per_device_batch", 4),
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        learning_rate=cfg.get("lr", 2.0e-5),
        lr_scheduler_type=cfg.get("lr_schedule", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.0),
        max_seq_length=cfg.get("max_seq_length", 1024),
        bf16=True,
        gradient_checkpointing=cfg.get("grad_checkpointing", True),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 250),
        save_total_limit=cfg.get("save_total_limit", 2),
        report_to=[],
        seed=cfg.get("seed", 42),
        packing=cfg.get("packing", False),
        optim=cfg.get("optim", "adamw_torch_fused"),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        tokenizer=tok,    # TRL 0.11 API; renamed to processing_class in 0.13+.
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    if trainer.is_world_process_zero():
        tok.save_pretrained(cfg["output_dir"])
        print(f"[train_sft] saved final model to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
