#!/usr/bin/env python3
"""Generate the three demo plots.

Subcommands:
  loss        training loss curve from logs/train-<JOB>.out
  accuracy    base vs SFT bars on target + forgetting-check subjects
  throughput  transformers vs vLLM bars (req/s, gen tok/s, p95 latency)

Writes PNGs (matplotlib Agg backend).
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path


def _ensure_mpl():
    """Import matplotlib lazily; configure for headless rendering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    return plt


def plot_loss(log_path: Path, out_path: Path) -> None:
    plt = _ensure_mpl()
    pat = re.compile(r"'loss':\s*([0-9.]+)")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    losses = [float(m.group(1)) for m in pat.finditer(text)]
    steps = list(range(1, len(losses) + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses, color="#1f77b4", linewidth=1.6)
    ax.set_xlabel("logging step")
    ax.set_ylabel("training loss")
    ax.set_title(f"SFT training loss ({len(losses)} log points, final {losses[-1]:.3f})")
    ax.grid(alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"loss plot -> {out_path}")


def plot_accuracy(accuracy_json: Path, out_path: Path) -> None:
    plt = _ensure_mpl()
    data = json.loads(accuracy_json.read_text())
    rows = data["rows"]
    rows.sort(key=lambda r: (not r["is_target"], r["task"]))  # target first

    labels = [r["task"].replace("mmlu_", "") for r in rows]
    base = [r["base_acc"] for r in rows]
    sft = [r["sft_acc"] for r in rows]

    import numpy as np
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_b = ax.bar(x - w / 2, base, w, label="base", color="#9ecae1")
    bars_s = ax.bar(x + w / 2, sft, w, label="SFT", color="#3182bd")
    for r, b in zip(rows, bars_s):
        d_pp = (r["sft_acc"] - r["base_acc"]) * 100
        ax.annotate(
            f"{d_pp:+.1f} pp",
            xy=(b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=9,
            color=("#0a8a3f" if d_pp > 0 else "#b03a2e"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, max(max(base), max(sft)) + 0.10)
    ax.set_title("MMLU accuracy: base vs SFT (target bolded)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for i, r in enumerate(rows):
        if r["is_target"]:
            ax.get_xticklabels()[i].set_fontweight("bold")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"accuracy plot -> {out_path}")


def plot_throughput(hf_json: Path, vllm_json: Path, out_path: Path) -> None:
    plt = _ensure_mpl()
    import numpy as np
    hf = json.loads(hf_json.read_text())
    vl = json.loads(vllm_json.read_text())
    metrics = [
        ("Requests/s", "requests_per_s"),
        ("Generated tok/s", "generated_tokens_per_s"),
        ("Latency p95 (s)", "latency_p95_s"),
    ]
    labels = [m[0] for m in metrics]
    hf_vals = [hf[m[1]] for m in metrics]
    vl_vals = [vl[m[1]] for m in metrics]

    x = np.arange(len(labels))
    w = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, label, hv, vv in zip(axes, labels, hf_vals, vl_vals):
        bars = ax.bar([0, 1], [hv, vv],
                      color=["#9ecae1", "#3182bd"], width=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["transformers", "vLLM"])
        ax.set_title(label, fontsize=11)
        # Speedup. For latency, lower is better, so invert the ratio.
        if "Latency" in label:
            mult = hv / vv if vv else 0
            arrow = "v"
        else:
            mult = vv / hv if hv else 0
            arrow = "^"
        ax.annotate(
            f"{mult:.1f}x {arrow}",
            xy=(1, vv), xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold",
            color="#0a8a3f" if mult > 1 else "#b03a2e",
        )
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Inference throughput: transformers vs vLLM (same prompts)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"throughput plot -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub_loss = sub.add_parser("loss"); sub_loss.add_argument("--log", required=True); sub_loss.add_argument("--out", required=True)
    sub_acc = sub.add_parser("accuracy"); sub_acc.add_argument("--accuracy-json", required=True); sub_acc.add_argument("--out", required=True)
    sub_thr = sub.add_parser("throughput"); sub_thr.add_argument("--hf-json", required=True); sub_thr.add_argument("--vllm-json", required=True); sub_thr.add_argument("--out", required=True)
    args = p.parse_args()

    if args.cmd == "loss":
        plot_loss(Path(args.log), Path(args.out))
    elif args.cmd == "accuracy":
        plot_accuracy(Path(args.accuracy_json), Path(args.out))
    elif args.cmd == "throughput":
        plot_throughput(Path(args.hf_json), Path(args.vllm_json), Path(args.out))


if __name__ == "__main__":
    main()
