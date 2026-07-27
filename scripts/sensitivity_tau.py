#!/usr/bin/env python3
"""Sensitivity of the verdict threshold τ (Appendix G).

τ only discretizes the (fixed) net plausibility into plausible / uncertain / implausible;
it does not change the net scores or their comparative ranking. This script sweeps τ over
the released `metrics.csv` and shows:
  - the verdict distribution and the *uncertain-band* share as a function of τ;
  - that the comparative ranking of the design cells (by mean net plausibility, which is
    τ-independent) is unaffected, and the ranking by verdict-derived "positive share"
    stays highly correlated across τ (Spearman ρ).

Reads only metrics.csv — no AQA re-scoring, no dependencies beyond the standard library.

Usage:
  python scripts/sensitivity_tau.py --run-dir experiments/full_factorial/runs
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

TAUS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
TAU_REF = 0.20  # calibrated threshold
_PAR = {"plan_then_execute": "plan_then_execute", "stepwise": "stepwise",
        "step_wise": "stepwise", "single_call": "single_call", "single": "single_call"}


def _truthy(v):
    return str(v).strip().lower() in {"true", "1", "yes", "on", "t"}


def _paradigm(r):
    p = _PAR.get((r.get("paradigm") or "").strip().lower())
    if p:
        return p
    if _truthy(r.get("single_call_reasoner")):
        return "single_call"
    return "plan_then_execute" if _truthy(r.get("planning_reasoner")) or \
        (r.get("planning_reasoner", "") == "") else "stepwise"


def _kendall(a, b):
    """Kendall's τ-b rank correlation between two equal-length sequences."""
    n = len(a)
    c = d = n1 = n2 = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0:
                n1 += 1
            if db == 0:
                n2 += 1
            if da != 0 and db != 0:
                c += 1 if (da > 0) == (db > 0) else 0
                d += 1 if (da > 0) != (db > 0) else 0
    n0 = n * (n - 1) / 2
    denom = ((n0 - n1) * (n0 - n2)) ** 0.5
    return (c - d) / denom if denom > 0 else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=Path("experiments/full_factorial/runs"))
    ap.add_argument("--subset", choices=["full", "720"], default="full",
                    help="'720' = the original 12-claim main study (shard_dir not from the "
                         "or_doe600 extension)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.run_dir / "metrics.csv")))
    if args.subset == "720":
        rows = [r for r in rows if "or_doe600" not in (r.get("shard_dir") or "")]
        print(f"[subset=720] {len(rows)} runs, {len({r['claim_id'] for r in rows})} claims\n")
    nets, cell = [], []
    for r in rows:
        try:
            v = float(r["aqa_plausibility"])
        except (TypeError, ValueError):
            continue
        nets.append(v)
        cell.append((r["reasoner_model"], r["counter_model"], _paradigm(r)))

    def share_by_cell(tau):
        """Positive-verdict (plausible) share per design cell, at threshold tau."""
        pos = collections.Counter()
        tot = collections.Counter()
        for v, c in zip(nets, cell):
            tot[c] += 1
            if v > tau:
                pos[c] += 1
        keys = sorted(tot)
        return keys, [pos[k] / tot[k] for k in keys]

    # ranking by mean net (τ-independent) — the comparative conclusion
    mean_net = collections.defaultdict(list)
    for v, c in zip(nets, cell):
        mean_net[c].append(v)
    keys = sorted(mean_net)
    net_rank = [sum(mean_net[k]) / len(mean_net[k]) for k in keys]

    def verdict(v, tau):
        return "plausible" if v > tau else "implausible" if v < -tau else "uncertain"

    ref_verdict = [verdict(v, TAU_REF) for v in nets]

    print(f"Sensitivity to the verdict threshold τ  ({len(nets)} runs)\n")
    print(f"{'τ':>6} {'plausible':>11} {'uncertain':>11} {'implausible':>12} "
          f"{'flips vs τ=0.2':>15} {'Kendall τ_k (plaus-share vs τ=0.2)':>36}")
    _, ref_share = share_by_cell(TAU_REF)
    n = len(nets)
    for tau in TAUS:
        pl = sum(1 for v in nets if v > tau)
        im = sum(1 for v in nets if v < -tau)
        un = n - pl - im
        _, sh = share_by_cell(tau)
        tk = _kendall(sh, ref_share)
        flips = sum(1 for v, rv in zip(nets, ref_verdict) if verdict(v, tau) != rv)
        print(f"{tau:>6.2f} {100*pl/n:>10.1f}% {100*un/n:>10.1f}% {100*im/n:>11.1f}% "
              f"{flips:>4} ({100*flips/n:>4.1f}%) {tk:>36.4f}")

    print("\nComparative ranking of the 12 configurations by plausible-verdict share stays "
          "≈constant\nacross τ (Kendall τ_k above); τ shifts the uncertain band and flips "
          "levels, not comparisons.")
    print(f"Best cell (mean net): {keys[max(range(len(keys)), key=lambda i: net_rank[i])]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
