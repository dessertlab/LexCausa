#!/usr/bin/env python3
"""Sensitivity of the AQA scoring parameters (paper Appendix G.1, Table 25).

Re-scores the *stored* per-run artifacts under perturbed specifications of the evaluator
and checks that every conclusion of the paper survives. Six variants bracket the
calibrated constants of Appendix E:

  Uniform weights     (alpha,beta,gamma) = (1/3, 1/3, 1/3)
  Norm-heavy          (alpha,beta,gamma) = (0.2, 0.6, 0.2)
  m_t x 0.75 / 1.25   attack-type multipliers scaled by -+25%
  lambda x 0.75/1.25  damage factor scaled by -+25%

Reported per variant, against the calibrated (default) setting:
  rho_S, tau_K   rank correlation of the 12-configuration (Reasoner x Counter x paradigm)
                 ordering by mean net plausibility
  Flips          share of runs whose verdict changes (%, at tau = 0.2)
  eta2_int       Reasoner x Counter interaction eta^2 (the RQ2 null result)
  RQ1            claims on which the instruction-tuned Reasoner has the higher mean net
  Top            best paradigm by mean net plausibility

Rescoring mechanism (reverse-validated on the released artifacts, exact at the default):

    B_i^s   = alpha*Cogency_i + beta*NormSupport_i + gamma*Semantics_i     per side s
    v_ij    = lam_s * lambda_ij * max(0, mt_s * m_ij * overlap_ij * B_j^r - B_i^s)
    P_i     = clamp01(B_i^s - sum_j v_ij + delta_i)
    side_s  = clamp01(mean_i P_i - redundancy_s + bonus_s)
    net     = side_pro - side_contra

The attacker's base score is looked up on the side given by the stored `attacker_role`
(r), *not* on the target's side: PRO links are attacked by CONTRA links and vice versa,
and the link ids ("S1->S2", ...) are reused across the two chains, so a side-agnostic
lookup would resolve every attacker to the wrong link.

The contra-side attack-coverage bonus is *recomputed* under perturbation (it depends on
the perturbed damages) from the stored axis partition:

    axis_q  = top + 0.3*2nd + 0.1*3rd   over the active contra->pro values on that axis
                                        retained at v >= 0.08 and overlap >= 0.45
    cov     = covered axes / total axes
    bonus   = min(0.15, 0.12 * (0.4*cov + 0.6*mean_axes(axis_q)))

Held fixed as observed, per Appendix G: the admitted-attack set, the axis partition, the
redundancy penalties and the precedent deltas. Their determinants are either independent
of the perturbed parameters (the semantic-overlap, NLI and domain-severity gates) or not
identifiable from the logs (the top-K retention).

Standard library only (reads the per-run JSON + metrics.csv).

Usage:
  python scripts/sensitivity_full.py                  # full factorial (1320 runs, 22 claims)
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import statistics
from pathlib import Path

# calibrated constants (src/config.py, paper Appendix E)
W0 = (0.3, 0.4, 0.3)          # alpha cogency, beta norm_support, gamma semantics
TAU = 0.20                    # verdict threshold
THIRD = 1.0 / 3.0

# attack-coverage bonus (paper Appendix E.4)
COV_MIN_VALUE = 0.08          # an attack counts toward an axis only above this damage
COV_MIN_OVERLAP = 0.45        # ... and above this overlap
COV_CAP = 0.15                # bonus ceiling
COV_SCALE = 0.12
COV_W_RATIO = 0.4             # weight of the covered-axis ratio vs the mean axis quality

INSTRUCTION_TUNED = "llama_3_3_70b"
PAR_SHORT = {"plan_then_execute": "PtE", "stepwise": "Step-wise",
             "single_call": "Single-call"}
_PAR = {"plan_then_execute": "plan_then_execute", "stepwise": "stepwise",
        "step_wise": "stepwise", "single_call": "single_call", "single": "single_call"}

# (label, weights, m_t scale, lambda scale)
VARIANTS = [
    ("Uniform weights", (THIRD, THIRD, THIRD), 1.0, 1.0),
    ("Norm-heavy (.2,.6,.2)", (0.2, 0.6, 0.2), 1.0, 1.0),
    ("m_t x 0.75", W0, 0.75, 1.0),
    ("m_t x 1.25", W0, 1.25, 1.0),
    ("lambda x 0.75", W0, 1.0, 0.75),
    ("lambda x 1.25", W0, 1.0, 1.25),
]


# ---- parsing -----------------------------------------------------------------
def _truthy(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "on", "t"}


def _f(x, d=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _paradigm(r) -> str:
    """Canonical paradigm of a run, falling back to the boolean design columns."""
    p = _PAR.get((r.get("paradigm") or "").strip().lower())
    if p:
        return p
    if _truthy(r.get("single_call_reasoner")):
        return "single_call"
    return "plan_then_execute" if _truthy(r.get("planning_reasoner")) else "stepwise"


def load(run_dir: Path, subset: str):
    """(rows by run_id, aqa_report by run_id) for the completed runs of the subset."""
    rows = {}
    for r in csv.DictReader(open(run_dir / "metrics.csv")):
        if (r.get("status") or "completed") != "completed":
            continue
        if subset == "720" and "or_doe600" in (r.get("shard_dir") or ""):
            continue
        rows[r["run_id"]] = r
    reports = {}
    for p in glob.glob(str(run_dir / "runs" / "*.json")):
        rid = os.path.splitext(os.path.basename(p))[0]
        if rid not in rows:
            continue
        try:
            rep = json.load(open(p)).get("evaluation", {}).get("aqa_report", {})
        except Exception:
            continue
        if rep.get("links"):
            reports[rid] = rep
    return rows, reports


# ---- rescoring engine --------------------------------------------------------
def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _coverage_bonus(report: dict, active_on_pro: dict) -> float:
    """Contra-side attack-coverage bonus, recomputed from the perturbed damages.

    The axis partition is read back from the stored artifacts; only the per-axis quality
    is re-derived, from the contra->pro attacks that survive the retention thresholds.
    """
    notes = report.get("notes") or {}
    sa = notes.get("structural_adjustments") or report.get("structural_adjustments") or {}
    ac = (sa.get("contra") or {}).get("attack_coverage") or {}
    axes = ac.get("axis_details") or []
    if not axes or not ac.get("enabled", True):
        return 0.0
    qs = []
    for ax in axes:
        vals = sorted((v for lid in ax.get("link_ids", [])
                       for v, o in active_on_pro.get(lid, [])
                       if v >= COV_MIN_VALUE and o >= COV_MIN_OVERLAP), reverse=True)
        q = 0.0
        if vals:
            q = vals[0]
            if len(vals) > 1:
                q += 0.3 * vals[1]
            if len(vals) > 2:
                q += 0.1 * vals[2]
        qs.append(q)
    cov = sum(1 for q in qs if q > 0) / len(qs)
    quality = statistics.mean(qs)
    return min(COV_CAP, COV_SCALE * (COV_W_RATIO * cov + (1.0 - COV_W_RATIO) * quality))


def rescore(report: dict, w=W0, mt_s: float = 1.0, lam_s: float = 1.0) -> float:
    """Net plausibility of one run under perturbed scoring constants."""
    links = report.get("links") or {}
    # base scores keyed by (side, link_id): link ids are reused across the two chains
    base = {(s, l.get("link_id")):
            w[0] * _f(l.get("cogency")) + w[1] * _f(l.get("norm_support"))
            + w[2] * _f(l.get("semantics"))
            for s in ("pro", "contra") for l in (links.get(s) or [])}

    side_val, active_on_pro = {}, {}
    for s in ("pro", "contra"):
        ls = links.get(s) or []
        if not ls:
            side_val[s] = 0.0
            continue
        plaus = []
        for l in ls:
            b_target = base[(s, l.get("link_id"))]
            active = []
            for a in l.get("attacks_received") or []:
                if a.get("filtered"):
                    continue
                # the attacker sits on the opposite chain; resolve it by its stored role
                role = "pro" if a.get("attacker_role") in ("support", "pro") else "contra"
                b_att = base.get((role, a.get("attacker_link_id")),
                                 _f(a.get("attacker_base_score")))
                boosted = (mt_s * _f(a.get("type_multiplier"), 1.0)
                           * _f(a.get("overlap")) * b_att)
                v = lam_s * _f(a.get("damage_factor")) * max(0.0, boosted - b_target)
                active.append((v, _f(a.get("overlap"))))
            if s == "pro":
                active_on_pro[l.get("link_id")] = active
            plaus.append(clamp01(b_target - sum(v for v, _o in active)
                                 + _f(l.get("precedent_delta"))))
        cs = (report.get("chain_scores") or {}).get(s) or {}
        bonus = _coverage_bonus(report, active_on_pro) if s == "contra" else 0.0
        side_val[s] = clamp01(statistics.mean(plaus)
                              - _f(cs.get("redundancy_penalty")) + bonus)
    return side_val["pro"] - side_val["contra"]


# ---- rank correlations -------------------------------------------------------
def _ranks(a):
    """Average ranks (ties shared), as needed by Spearman."""
    order = sorted(range(len(a)), key=lambda i: a[i])
    r = [0.0] * len(a)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b) -> float:
    """Spearman rho (Pearson on average ranks, so ties are handled)."""
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 1.0


def kendall(a, b) -> float:
    """Kendall tau-b rank correlation."""
    n = len(a)
    c = d = n1 = n2 = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0:
                n1 += 1
            if db == 0:
                n2 += 1
            if da and db:
                c += 1 if (da > 0) == (db > 0) else 0
                d += 1 if (da > 0) != (db > 0) else 0
    n0 = n * (n - 1) / 2
    den = ((n0 - n1) * (n0 - n2)) ** 0.5
    return (c - d) / den if den > 0 else 1.0


# ---- design-level statistics -------------------------------------------------
def verdict(net: float, tau: float = TAU) -> str:
    return "plausible" if net > tau else "implausible" if net < -tau else "uncertain"


def eta2(nets: dict, factors: dict, order) -> dict:
    """eta^2 of the three main effects and of the Reasoner x Counter interaction."""
    ys = [nets[r] for r in order]
    gm = statistics.mean(ys)
    sst = sum((y - gm) ** 2 for y in ys)
    if not sst:
        return {"Reasoner": 0.0, "Counter": 0.0, "Paradigm": 0.0, "RxC": 0.0}

    def means(key):
        d = collections.defaultdict(list)
        for r in order:
            d[key(r)].append(nets[r])
        return {k: statistics.mean(v) for k, v in d.items()}

    out = {}
    for i, lab in ((0, "Reasoner"), (1, "Counter"), (2, "Paradigm")):
        m = means(lambda r, i=i: factors[r][i])
        out[lab] = sum((m[factors[r][i]] - gm) ** 2 for r in order) / sst
    mR = means(lambda r: factors[r][0])
    mC = means(lambda r: factors[r][1])
    mRC = means(lambda r: (factors[r][0], factors[r][1]))
    out["RxC"] = sum((mRC[(factors[r][0], factors[r][1])] - mR[factors[r][0]]
                      - mC[factors[r][1]] + gm) ** 2 for r in order) / sst
    return out


def cell_mean_net(nets: dict, factors: dict, cells):
    """Mean net plausibility per design cell — the Table 25 ranking basis."""
    d = collections.defaultdict(list)
    for rid, v in nets.items():
        d[factors[rid][:3]].append(v)
    return [statistics.mean(d[c]) if d[c] else 0.0 for c in cells]


def claims_won_by_instruction_tuned(nets: dict, factors: dict):
    """(wins, total): claims where the instruction-tuned Reasoner has the higher mean."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for rid, v in nets.items():
        by[factors[rid][3]][factors[rid][0]].append(v)
    wins = total = 0
    for _claim, d in by.items():
        if INSTRUCTION_TUNED not in d or len(d) < 2:
            continue
        total += 1
        native = [v for m, vs in d.items() if m != INSTRUCTION_TUNED for v in vs]
        if statistics.mean(d[INSTRUCTION_TUNED]) > statistics.mean(native):
            wins += 1
    return wins, total


def by_factor(nets: dict, factors: dict, i: int) -> dict:
    d = collections.defaultdict(list)
    for rid, v in nets.items():
        d[factors[rid][i]].append(v)
    return {k: statistics.mean(v) for k, v in d.items()}


def analyze(nets, factors, cells, order, ref=None) -> dict:
    res = {
        "nets": nets,
        "cellnet": cell_mean_net(nets, factors, cells),
        "verdicts": {r: verdict(nets[r]) for r in order},
        "eta": eta2(nets, factors, order),
        "by_paradigm": by_factor(nets, factors, 2),
    }
    res["top"] = max(res["by_paradigm"], key=res["by_paradigm"].get)
    res["rq1"] = claims_won_by_instruction_tuned(nets, factors)
    if ref is not None:
        res["rho"] = spearman(res["cellnet"], ref["cellnet"])
        res["tk"] = kendall(res["cellnet"], ref["cellnet"])
        res["flips"] = 100.0 * sum(1 for r in order
                                   if res["verdicts"][r] != ref["verdicts"][r]) / len(order)
    else:
        res["rho"], res["tk"], res["flips"] = 1.0, 1.0, 0.0
    return res


# ---- CLI ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path,
                    default=Path("experiments/full_factorial/runs"))
    ap.add_argument("--subset", choices=["full", "720"], default="full",
                    help="'720' = the original 12-claim main study (shard_dir not from "
                         "the or_doe600 extension)")
    args = ap.parse_args()

    rows, reports = load(args.run_dir, args.subset)
    factors = {rid: (rows[rid]["reasoner_model"], rows[rid]["counter_model"],
                     _paradigm(rows[rid]), rows[rid]["claim_id"]) for rid in reports}
    order = sorted(reports)
    cells = sorted({factors[rid][:3] for rid in reports})
    print(f"[subset={args.subset}] {len(rows)} runs, {len(reports)} with AQA artifacts, "
          f"{len({factors[r][3] for r in order})} claims, "
          f"{len(cells)} design configurations\n")

    ref = analyze({rid: rescore(rep) for rid, rep in reports.items()},
                  factors, cells, order)

    errs = [abs(ref["nets"][r] - _f(rows[r].get("aqa_plausibility"))) for r in order]
    print(f"[validation] re-scorer vs stored `aqa_plausibility` at the calibrated "
          f"constants:\n             mean|d| = {statistics.mean(errs):.2e}   "
          f"max|d| = {max(errs):.2e}   over {len(errs)} runs")
    print("             (the residual is the four-decimal rounding of the logged "
          "overlaps)\n")

    results = [(lab, analyze({rid: rescore(rep, w, mt, lam)
                              for rid, rep in reports.items()},
                             factors, cells, order, ref))
               for lab, w, mt, lam in VARIANTS]

    print(f"{'Variant':<24} {'rho_S':>7} {'tau_K':>7} {'Flips':>7} "
          f"{'eta2_int':>9} {'RQ1':>7} {'Top':>14}")
    print("-" * 82)
    for lab, r in results:
        w, t = r["rq1"]
        print(f"{lab:<24} {r['rho']:>7.3f} {r['tk']:>7.3f} {r['flips']:>6.1f}% "
              f"{r['eta']['RxC']:>9.3f} {f'{w}/{t}':>7} {PAR_SHORT[r['top']]:>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
