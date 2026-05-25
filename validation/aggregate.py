#!/usr/bin/env python3
"""Aggregate per-rank validation JSONs into a single report (JSON + markdown)."""
from __future__ import annotations
import argparse
import json
import pathlib
from statistics import mean


# Pass thresholds. Conservative; tighten as we collect more reference runs.
TFLOPS_MIN_BF16 = 600.0      # H200 spec ~1000 bf16 TFLOPs.
NCCL_BUSBW_MIN_GBPS = 50.0   # H200 + NDR IB should clear this easily.
HF_LOAD_MAX_S = 60.0         # network + cache write should finish in a minute.


def _load_json(p: pathlib.Path) -> dict:
    return json.loads(p.read_text())


def aggregate(in_dir: pathlib.Path) -> dict:
    files = sorted(in_dir.glob("*.json"))
    by_name: dict[str, list[dict]] = {}
    for f in files:
        kind = f.stem.split("_rank")[0]
        by_name.setdefault(kind, []).append(_load_json(f))

    report: dict = {"checks": {}}

    # gpu_info (single, from rank 0)
    if "gpu_info" in by_name:
        info = by_name["gpu_info"][0]
        report["checks"]["gpu_info"] = {
            "pass": len(info["gpus"]) > 0,
            "n_gpus_visible": len(info["gpus"]),
            "first_gpu": info["gpus"][0] if info["gpus"] else None,
        }

    # gpu_compute (one per rank)
    if "gpu_compute" in by_name:
        ranks = by_name["gpu_compute"]
        tflops = [r["result"]["tflops"] for r in ranks]
        report["checks"]["gpu_compute"] = {
            "pass": all(t >= TFLOPS_MIN_BF16 for t in tflops),
            "n_ranks": len(ranks),
            "tflops_min": min(tflops),
            "tflops_mean": mean(tflops),
            "tflops_max": max(tflops),
            "tflops_threshold": TFLOPS_MIN_BF16,
            "per_rank": [
                {"rank": r["rank"], "node": r["node"], "tflops": r["result"]["tflops"]}
                for r in ranks
            ],
        }

    # nccl_bench (one per rank, rank 0's numbers are representative)
    if "nccl_bench" in by_name:
        ranks = sorted(by_name["nccl_bench"], key=lambda r: r["rank"])
        rank0 = ranks[0]
        peak_busbw = max(x["bus_bandwidth_gbps"] for x in rank0["results"])
        report["checks"]["nccl_bench"] = {
            "pass": peak_busbw >= NCCL_BUSBW_MIN_GBPS,
            "world_size": rank0["world_size"],
            "peak_busbw_gbps": peak_busbw,
            "threshold_gbps": NCCL_BUSBW_MIN_GBPS,
            "sweep": rank0["results"],
        }

    # hf_smoke (rank 0 only)
    if "hf_smoke" in by_name:
        s = by_name["hf_smoke"][0]
        load_s = s["timings_s"].get("load_s", 1e9)
        report["checks"]["hf_smoke"] = {
            "pass": load_s <= HF_LOAD_MAX_S,
            "load_s": load_s,
            "encode_s": s["timings_s"].get("encode_s"),
            "threshold_s": HF_LOAD_MAX_S,
            "embedding_dim": s["embedding_dim"],
        }

    report["overall_pass"] = all(c.get("pass") for c in report["checks"].values())
    return report


def to_markdown(report: dict) -> str:
    lines: list[str] = []
    overall = "PASS" if report["overall_pass"] else "FAIL"
    lines.append(f"# Cluster validation report - overall {overall}\n")
    for name, c in report["checks"].items():
        tag = "PASS" if c.get("pass") else "FAIL"
        lines.append(f"## {name} - {tag}")
        if name == "gpu_info":
            g = c["first_gpu"]
            lines.append(
                f"- {c['n_gpus_visible']} GPU(s) visible on rank-0 host. "
                f"First: **{g['name']}**, driver {g['driver']}, "
                f"{g['memory_total_mib']} MiB HBM."
            )
        elif name == "gpu_compute":
            lines.append(
                f"- bf16 matmul TFLOPs across {c['n_ranks']} rank(s): "
                f"min {c['tflops_min']:.1f}, mean {c['tflops_mean']:.1f}, "
                f"max {c['tflops_max']:.1f} (threshold {c['tflops_threshold']:.0f})."
            )
            for r in c["per_rank"]:
                lines.append(f"  - rank {r['rank']} ({r['node']}): {r['tflops']:.1f} TFLOPs")
        elif name == "nccl_bench":
            lines.append(
                f"- all-reduce world_size={c['world_size']}, "
                f"peak bus bandwidth {c['peak_busbw_gbps']:.1f} GB/s "
                f"(threshold {c['threshold_gbps']:.0f})."
            )
            for s in c["sweep"]:
                lines.append(
                    f"  - {s['size_mib']:.0f} MiB -> {s['bus_bandwidth_gbps']:.1f} GB/s"
                )
        elif name == "hf_smoke":
            lines.append(
                f"- load {c['load_s']:.1f}s, encode {c['encode_s']:.2f}s, "
                f"embedding dim {c['embedding_dim']} (threshold {c['threshold_s']:.0f}s load)."
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--md", required=True)
    args = p.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    report = aggregate(in_dir)
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
    pathlib.Path(args.md).write_text(to_markdown(report))
    print(f"aggregate -> {args.out}, {args.md}")


if __name__ == "__main__":
    main()
