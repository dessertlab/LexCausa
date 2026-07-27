#!/usr/bin/env python
"""Merge the per-shard outputs of a split multi-DoE campaign into one run dir.

Built for the two-machine split (PC A = global shards 0-3, PC B = 4-7): copy
PC B's ``*_g4..g7`` directories next to PC A's, then point this at all of them.

It concatenates every ``metrics.csv``, copies the raw per-run JSONs into a single
``runs/`` folder, and verifies the result: no duplicate ``run_id`` (the shards are
disjoint by construction, so any duplicate means the two machines built different
matrices) and, optionally, that the expected total was reached.

Usage:
    poetry run python scripts/merge_doe_shards.py \\
        --shards experiments/multi_doe/runs/or_qwen80_pc*_g* \\
        --out    experiments/multi_doe/runs/or_qwen80_merged \\
        --expect 1320
"""
from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="shard dirs (globs ok)")
    ap.add_argument("--out", required=True, help="output merged run dir")
    ap.add_argument(
        "--expect", type=int, default=0, help="expected total runs (0=skip)"
    )
    ap.add_argument("--copy-raw", action="store_true", help="also copy runs/*.json")
    args = ap.parse_args()

    dirs: list[Path] = []
    for pat in args.shards:
        dirs.extend(Path(p) for p in sorted(glob.glob(pat)) if Path(p).is_dir())
    if not dirs:
        print("no shard directories matched", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for d in dirs:
        m = d / "metrics.csv"
        if not m.exists():
            print(f"  !! {d.name}: metrics.csv MISSING (shard incomplete?)")
            continue
        df = pd.read_csv(m)
        df["shard_dir"] = d.name
        frames.append(df)
        print(f"  {d.name}: {len(df)} runs")

    if not frames:
        print("nothing to merge", file=sys.stderr)
        return 1

    merged = pd.concat(frames, ignore_index=True)

    dups = merged.run_id.duplicated().sum()
    print(
        f"\ntotal rows: {len(merged)} | unique run_id: {merged.run_id.nunique()} | duplicates: {dups}"
    )
    if dups:
        print(
            "ERROR: duplicate run_id across shards. The machines did NOT build the\n"
            "same matrix (different claims file, models, replicates or seed).",
            file=sys.stderr,
        )
        return 2

    if args.expect and len(merged) != args.expect:
        print(
            f"WARNING: expected {args.expect} runs, got {len(merged)} — a shard is missing or short."
        )

    ok = (merged.status == "completed").sum()
    print(f"completed: {ok} | failed: {len(merged) - ok}")

    merged.to_csv(out / "metrics.csv", index=False)
    try:
        merged.to_parquet(out / "metrics.parquet", index=False)
    except Exception:
        pass

    if args.copy_raw:
        raw = out / "runs"
        raw.mkdir(exist_ok=True)
        n = 0
        for d in dirs:
            for j in (d / "runs").glob("*.json"):
                shutil.copy2(j, raw / j.name)
                n += 1
        print(f"copied {n} raw responses -> {raw}")

    print(f"\nmerged -> {out/'metrics.csv'}")
    print(
        f"analyze: poetry run python scripts/analyze_multi_doe.py --run-dir {out} --output <out>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
