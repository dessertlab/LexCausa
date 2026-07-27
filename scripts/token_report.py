#!/usr/bin/env python
"""Per-phase token report for LexCausa runs.

Two modes:

  # Single full-pipeline run: reads _token_stats.by_phase from a saved response
  python scripts/token_report.py --response path/to/run_response.json

  # A whole DoE run directory: aggregates the tok_<phase>_* columns of metrics.csv
  python scripts/token_report.py --doe experiments/full_factorial/runs/<run_dir>

The DoE runner persists each raw response under <out>/runs/<run_id>.json and the
flattened per-phase columns in metrics.csv (tok_<phase>_prompt / _completion).
"""
import argparse
import json
from pathlib import Path

PHASES = ["retrieval", "reasoner", "counter_reasoner", "evaluator"]


def _fmt(n: float) -> str:
    return f"{int(n):,}"


def report_single(response_path: Path) -> None:
    data = json.loads(response_path.read_text(encoding="utf-8"))
    ts = data.get("_token_stats", {}) or {}
    by_phase = ts.get("by_phase", {}) or {}
    if not by_phase:
        print("No per-phase token stats in this response (old run?).")
        return
    print(f"\nPer-phase tokens — {response_path.name}\n")
    print(f"{'phase':18} {'input':>12} {'output':>12} {'total':>12}")
    print("-" * 56)
    ti = to = 0
    for ph in PHASES:
        p = by_phase.get(ph, {}) or {}
        pi, po = int(p.get("prompt", 0)), int(p.get("completion", 0))
        ti += pi
        to += po
        print(f"{ph:18} {_fmt(pi):>12} {_fmt(po):>12} {_fmt(pi + po):>12}")
    print("-" * 56)
    print(f"{'TOTAL':18} {_fmt(ti):>12} {_fmt(to):>12} {_fmt(ti + to):>12}")
    print(
        f"\nreported totals: input={_fmt(ts.get('total_prompt_tokens', 0))} "
        f"output={_fmt(ts.get('total_completion_tokens', 0))} "
        f"all={_fmt(ts.get('total_tokens', 0))}"
    )


def report_doe(run_dir: Path) -> None:
    import pandas as pd

    csv = run_dir / "metrics.csv"
    df = pd.read_csv(csv)
    df = df[df.get("status", "completed") == "completed"] if "status" in df else df
    print(
        f"\nDoE per-phase tokens (mean over {len(df)} completed runs) — {run_dir.name}\n"
    )
    print(f"{'phase':18} {'input':>12} {'output':>12} {'total':>12}")
    print("-" * 56)
    ti = to = 0.0
    for ph in PHASES:
        pi = df.get(f"tok_{ph}_prompt")
        po = df.get(f"tok_{ph}_completion")
        mi = float(pi.mean()) if pi is not None else 0.0
        mo = float(po.mean()) if po is not None else 0.0
        ti += mi
        to += mo
        print(f"{ph:18} {_fmt(mi):>12} {_fmt(mo):>12} {_fmt(mi + mo):>12}")
    print("-" * 56)
    print(f"{'TOTAL/run':18} {_fmt(ti):>12} {_fmt(to):>12} {_fmt(ti + to):>12}")
    if "total_prompt_tokens" in df:
        print(f"\nmean input/run:  {_fmt(df['total_prompt_tokens'].mean())}")
    if "total_tokens" in df:
        print(f"mean output/run: {_fmt(df['total_tokens'].mean())}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-phase token report")
    ap.add_argument("--response", help="Path to a single saved pipeline response JSON")
    ap.add_argument("--doe", help="Path to a DoE run directory (with metrics.csv)")
    args = ap.parse_args()
    if args.response:
        report_single(Path(args.response))
    elif args.doe:
        report_doe(Path(args.doe))
    else:
        ap.error("provide --response or --doe")


if __name__ == "__main__":
    main()
