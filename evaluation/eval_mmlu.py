#!/usr/bin/env python3
"""Accuracy eval on MMLU via lm-evaluation-harness.

Invoke lm_eval against one model on a list of MMLU subjects, write a
summary JSON. Run twice from the sbatch (base, then SFT) so compare_accuracy
can diff them.

The target subject is the one we fine-tuned for. The forgetting-check
subjects are unrelated MMLU subjects we expect the SFT not to degrade.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path


def run_lm_eval(model: str, tasks: list[str], out_dir: Path,
                batch_size: int = 8, num_fewshot: int = 5,
                apply_chat_template: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model},dtype=bfloat16",
        "--tasks", ",".join(tasks),
        "--batch_size", str(batch_size),
        "--num_fewshot", str(num_fewshot),
        "--output_path", str(out_dir),
    ]
    if apply_chat_template:
        cmd.append("--apply_chat_template")
    print(f"[eval] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    # lm-eval writes a nested results file; grab the most recent.
    results = sorted(out_dir.glob("**/results_*.json"), key=lambda p: p.stat().st_mtime)
    if not results:
        raise RuntimeError(f"no lm-eval results JSON found under {out_dir}")
    return results[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF model id or local checkpoint path")
    p.add_argument("--tasks", required=True,
                   help="Comma-separated MMLU subject task names "
                        "(e.g. mmlu_high_school_mathematics)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--label", default="model",
                   help="Short label for this run (used in output filename)")
    p.add_argument("--apply-chat-template", action="store_true",
                   help="Wrap prompts in the model's chat template before "
                        "scoring (recommended for instruction-tuned + SFT models)")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_dir = Path(args.out_dir) / args.label

    results_path = run_lm_eval(
        model=args.model,
        tasks=tasks,
        out_dir=out_dir,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        apply_chat_template=args.apply_chat_template,
    )
    print(f"[eval] results -> {results_path}")

    data = json.loads(results_path.read_text())
    summary = {
        "label": args.label,
        "model": args.model,
        "tasks": {},
    }
    for task_name, task_results in (data.get("results") or {}).items():
        acc = task_results.get("acc,none") or task_results.get("acc")
        acc_err = task_results.get("acc_stderr,none") or task_results.get("acc_stderr")
        summary["tasks"][task_name] = {"acc": acc, "acc_stderr": acc_err}

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[eval] summary -> {summary_path}")
    for t, r in summary["tasks"].items():
        print(f"  {t}: acc={r['acc']:.4f}  +/-{r['acc_stderr']:.4f}")


if __name__ == "__main__":
    main()
