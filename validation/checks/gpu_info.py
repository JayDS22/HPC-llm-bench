#!/usr/bin/env python3
"""nvidia-smi snapshot. Runs on rank 0. Captures driver, model, HBM per visible GPU."""
from __future__ import annotations
import argparse
import json
import subprocess


def _query(fields: list[str]) -> list[list[str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return [[x.strip() for x in line.split(",")] for line in out.splitlines()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()

    fields = [
        "index", "name", "driver_version",
        "memory.total", "memory.free",
        "compute_mode", "uuid", "pstate",
    ]
    rows = _query(fields)
    gpus = [
        {
            "index": int(r[0]),
            "name": r[1],
            "driver": r[2],
            "memory_total_mib": int(r[3]),
            "memory_free_mib": int(r[4]),
            "compute_mode": r[5],
            "uuid": r[6],
            "pstate": r[7],
        }
        for r in rows
    ]
    with open(args.out, "w") as f:
        json.dump({"gpus": gpus}, f, indent=2)
    print(f"gpu_info: {len(gpus)} GPU(s) -> {args.out}")


if __name__ == "__main__":
    main()
