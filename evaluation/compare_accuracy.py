#!/usr/bin/env python3
"""Diff two eval_mmlu summary JSONs, write a JSON delta and a markdown table.

Output feeds the demo slide showing per-subject deltas (target bolded, plus
forgetting-check subjects).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Path to base summary.json")
    p.add_argument("--sft", required=True, help="Path to SFT summary.json")
    p.add_argument("--target-task", required=True,
                   help="MMLU task name we fine-tuned for (header bolding)")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args()

    base = json.loads(Path(args.base).read_text())
    sft = json.loads(Path(args.sft).read_text())

    rows = []
    for task in sorted(set(base["tasks"]) | set(sft["tasks"])):
        b = base["tasks"].get(task, {}).get("acc")
        s = sft["tasks"].get(task, {}).get("acc")
        delta = (s - b) if (b is not None and s is not None) else None
        rows.append({
            "task": task,
            "base_acc": b,
            "sft_acc": s,
            "delta_pp": (delta * 100) if delta is not None else None,
            "is_target": task == args.target_task,
        })

    Path(args.out_json).write_text(json.dumps({
        "base_model": base["model"],
        "sft_model": sft["model"],
        "rows": rows,
    }, indent=2))

    lines = [
        "# Accuracy: base vs SFT",
        "",
        f"- Base: `{base['model']}`",
        f"- SFT:  `{sft['model']}`",
        "",
        "| Task | Base acc | SFT acc | Delta (pp) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        name = f"**{r['task']}**" if r["is_target"] else r["task"]
        b_str = f"{r['base_acc']:.4f}" if r["base_acc"] is not None else "-"
        s_str = f"{r['sft_acc']:.4f}" if r["sft_acc"] is not None else "-"
        d_str = (f"{r['delta_pp']:+.2f}" if r["delta_pp"] is not None else "-")
        lines.append(f"| {name} | {b_str} | {s_str} | {d_str} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
