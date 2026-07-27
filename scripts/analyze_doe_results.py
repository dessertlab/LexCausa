#!/usr/bin/env python3
"""
Analisi statistica dei risultati DoE A/B per LexCausa.

Legge il file run_summary.csv prodotto dal batch DoE e calcola:
  1. Statistiche descrittive (media ± σ) per contra, pro, final plausibility
  2. Paired t-test bilaterale su Δ_contra
  3. Sign test binomiale unilaterale (H1: B > A)
  4. Cohen's d per dati accoppiati
  5. Breakdown per dominio (con d e sign test per ciascuno)
  6. Sotto-metriche (cogency, semantics, norm_support)
  7. Metriche strutturali (links, steps, density, selected_attacks)
  8. Consistenza intra-claim (unanime B / maggioranza B / maggioranza A / unanime A)
  9. Verdict flip (cambi plausible ↔ uncertain ↔ implausible)
 10. Error analysis: astensioni, riparazioni fallite, gate distribution
 11. Isolamento PRO (|Δ_pro| per coppia)

Uso:
    python scripts/analyze_doe_results.py [--csv PATH] [--out PATH]

Se --csv non è specificato, cerca il file in:
    experiments/causal_taxonomy_ablation/run_001/run_summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
DEFAULT_CSV = Path("experiments/causal_taxonomy_ablation/run_001/run_summary.csv")
VERDICT_COL_A = "verdict_label_A"
VERDICT_COL_B = "verdict_label_B"

# ---------------------------------------------------------------------------
# Mappatura colonne CSV
# ---------------------------------------------------------------------------
# Le colonne nel run_summary.csv seguono il pattern {setup}_{metric}.
# Definiamo alias leggibili → nome colonna effettivo.
COL = {
    "contra_A": "A_aqa_net_contra",
    "contra_B": "B_aqa_net_contra",
    "pro_A": "A_aqa_net_pro",
    "pro_B": "B_aqa_net_pro",
    "final_A": "A_aqa_net_final",
    "final_B": "B_aqa_net_final",
    "contra_cogency_A": "A_contra_cogency_avg",
    "contra_cogency_B": "B_contra_cogency_avg",
    "contra_semantics_A": "A_contra_semantics_avg",
    "contra_semantics_B": "B_contra_semantics_avg",
    "contra_norm_support_A": "A_contra_norm_support_avg",
    "contra_norm_support_B": "B_contra_norm_support_avg",
    "contra_links_A": "A_contra_links_count",
    "contra_links_B": "B_contra_links_count",
    "counter_chain_steps_A": "A_counter_chain_steps",
    "counter_chain_steps_B": "B_counter_chain_steps",
    "counter_density_A": "A_counter_density",
    "counter_density_B": "B_counter_density",
    "selected_attacks_n_A": "A_counter_selected_attacks_n",
    "selected_attacks_n_B": "B_counter_selected_attacks_n",
    "gate_label_A": "A_counter_gate_label",
    "gate_label_B": "B_counter_gate_label",
    "abstain_A": "abstain_A",
    "abstain_B": "abstain_B",
    "abstention_type_A": "abstention_type_A",
    "abstention_type_B": "abstention_type_B",
}


DOMAIN_MAP = {
    "C": "CIVILE",
    "P": "PENALE",
    "M": "MISTO",
    "A": "AMMINISTRATIVO",
}


# ---------------------------------------------------------------------------
# Utilità
# ---------------------------------------------------------------------------
def claim_domain(claim_id: str) -> str:
    """Ricava il dominio dalla prima lettera del claim_id (C/P/M/A)."""
    prefix = claim_id[0].upper()
    return DOMAIN_MAP.get(prefix, "UNKNOWN")


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d per dati accoppiati: d = mean(Δ) / std(Δ, ddof=1)."""
    delta = b - a
    return float(np.mean(delta) / np.std(delta, ddof=1))


def sign_test(a: np.ndarray, b: np.ndarray, alternative: str = "greater"):
    """Sign test binomiale: conta quante volte b > a."""
    n_b_wins = int(np.sum(b > a))
    n_clean = int(np.sum(b != a))  # escludi pareggi esatti
    result = stats.binomtest(n_b_wins, n_clean, 0.5, alternative=alternative)
    return n_b_wins, n_clean, result.pvalue


def verdict_label(final_plaus: float) -> str:
    """Classifica il verdetto come plausible / uncertain / implausible."""
    if final_plaus >= 0.5:
        return "plausible"
    elif final_plaus >= 0.3:
        return "uncertain"
    else:
        return "implausible"


# ---------------------------------------------------------------------------
# Analisi principale
# ---------------------------------------------------------------------------
def analyze(csv_path: Path) -> dict:
    """Esegue l'intera batteria di analisi e restituisce un dizionario di risultati."""

    df = pd.read_csv(csv_path)
    results: dict = {}

    # --- 0. Overview ---
    n_total = len(df)
    results["n_total"] = n_total

    # Identifica astensioni (contra == 0 in almeno un setup)
    mask_abstain = (df[COL["contra_A"]] == 0) | (df[COL["contra_B"]] == 0)
    abstentions = df[mask_abstain].copy()
    results["n_abstentions"] = int(mask_abstain.sum())
    results["abstentions"] = abstentions[["claim_id", "replicate"]].to_dict(
        orient="records"
    )

    # Dataset pulito
    clean = df[~mask_abstain].copy()
    n = len(clean)
    results["n_clean"] = n

    # Vettori principali
    contra_a = clean[COL["contra_A"]].values
    contra_b = clean[COL["contra_B"]].values
    delta = contra_b - contra_a

    pro_a = clean[COL["pro_A"]].values
    pro_b = clean[COL["pro_B"]].values

    final_a = clean[COL["final_A"]].values
    final_b = clean[COL["final_B"]].values

    # --- 1. Statistiche descrittive ---
    results["descriptive"] = {
        "contra_A": {
            "mean": float(np.mean(contra_a)),
            "std": float(np.std(contra_a, ddof=1)),
        },
        "contra_B": {
            "mean": float(np.mean(contra_b)),
            "std": float(np.std(contra_b, ddof=1)),
        },
        "delta_contra": {
            "mean": float(np.mean(delta)),
            "std": float(np.std(delta, ddof=1)),
        },
        "pro_A": {"mean": float(np.mean(pro_a)), "std": float(np.std(pro_a, ddof=1))},
        "pro_B": {"mean": float(np.mean(pro_b)), "std": float(np.std(pro_b, ddof=1))},
        "final_A": {
            "mean": float(np.mean(final_a)),
            "std": float(np.std(final_a, ddof=1)),
        },
        "final_B": {
            "mean": float(np.mean(final_b)),
            "std": float(np.std(final_b, ddof=1)),
        },
    }

    # --- 2. Paired t-test ---
    t_stat, p_two = stats.ttest_rel(contra_b, contra_a)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    results["ttest"] = {
        "t_statistic": float(t_stat),
        "df": n - 1,
        "p_two_tailed": float(p_two),
        "p_one_tailed": float(p_one),
    }

    # --- 3. Sign test ---
    n_b_wins, n_compared, p_sign = sign_test(contra_a, contra_b)
    results["sign_test"] = {
        "B_wins": n_b_wins,
        "n_compared": n_compared,
        "B_pct": round(100 * n_b_wins / n_compared, 1),
        "p_value": float(p_sign),
    }

    # --- 4. Cohen's d ---
    d = cohens_d_paired(contra_a, contra_b)
    results["cohens_d"] = round(d, 3)

    # --- 5. Breakdown per dominio ---
    if "domain" not in clean.columns:
        clean["domain"] = clean["claim_id"].apply(claim_domain)
    domain_results = {}
    for dom, grp in clean.groupby("domain"):
        ca = grp[COL["contra_A"]].values
        cb = grp[COL["contra_B"]].values
        n_dom = len(grp)
        bw, nc, ps = sign_test(ca, cb)
        domain_results[dom] = {
            "n": n_dom,
            "B_wins": bw,
            "B_pct": round(100 * bw / nc, 1) if nc > 0 else None,
            "mean_delta": round(float(np.mean(cb - ca)), 3),
            "cohens_d": round(cohens_d_paired(ca, cb), 2),
            "sign_p": round(ps, 4),
        }
    results["per_domain"] = domain_results

    # --- 6. Sotto-metriche ---
    sub_metrics = {}
    for short_name in ["contra_cogency", "contra_semantics", "contra_norm_support"]:
        col_a = COL.get(f"{short_name}_A")
        col_b = COL.get(f"{short_name}_B")
        if col_a and col_b and col_a in clean.columns and col_b in clean.columns:
            ma = float(clean[col_a].mean())
            mb = float(clean[col_b].mean())
            sub_metrics[short_name] = {
                "A": round(ma, 3),
                "B": round(mb, 3),
                "delta": round(mb - ma, 3),
                "delta_pct": round(100 * (mb - ma) / ma, 1) if ma != 0 else None,
            }
    results["sub_metrics"] = sub_metrics

    # --- 7. Metriche strutturali ---
    structural = {}
    for short_name in [
        "contra_links",
        "counter_chain_steps",
        "counter_density",
        "selected_attacks_n",
    ]:
        col_a_key = f"{short_name}_A"
        col_b_key = f"{short_name}_B"
        col_a = COL.get(col_a_key)
        col_b = COL.get(col_b_key)
        if col_a and col_b and col_a in clean.columns and col_b in clean.columns:
            ma = float(clean[col_a].mean())
            mb = float(clean[col_b].mean())
            structural[short_name] = {
                "A": round(ma, 2),
                "B": round(mb, 2),
                "delta": round(mb - ma, 2),
            }
        elif col_b and col_b in clean.columns:
            structural[short_name] = {
                "A": 0.0,
                "B": round(float(clean[col_b].mean()), 2),
            }
    results["structural"] = structural

    # --- 8. Consistenza intra-claim ---
    claim_consistency = {}
    for cid, grp in clean.groupby("claim_id"):
        ca = grp[COL["contra_A"]].values
        cb = grp[COL["contra_B"]].values
        b_wins_claim = int(np.sum(cb > ca))
        n_reps = len(grp)
        claim_consistency[cid] = {
            "n_reps_clean": n_reps,
            "B_wins": b_wins_claim,
            "mean_delta": round(float(np.mean(cb - ca)), 3),
        }
    # Classificazione
    all_for_b = [
        c for c, v in claim_consistency.items() if v["B_wins"] == v["n_reps_clean"]
    ]
    majority_b = [
        c
        for c, v in claim_consistency.items()
        if 0 < v["B_wins"] < v["n_reps_clean"] and v["B_wins"] > v["n_reps_clean"] / 2
    ]
    majority_a = [
        c
        for c, v in claim_consistency.items()
        if v["B_wins"] < v["n_reps_clean"] / 2 and v["B_wins"] > 0
    ]
    all_for_a = [c for c, v in claim_consistency.items() if v["B_wins"] == 0]
    results["consistency"] = {
        "all_B": sorted(all_for_b),
        "majority_B": sorted(majority_b),
        "majority_A": sorted(majority_a),
        "all_A": sorted(all_for_a),
        "per_claim": claim_consistency,
    }

    # --- 9. Verdict flip ---
    clean = clean.copy()
    clean["verdict_A"] = clean[COL["final_A"]].apply(verdict_label)
    clean["verdict_B"] = clean[COL["final_B"]].apply(verdict_label)
    flips = clean[clean["verdict_A"] != clean["verdict_B"]].copy()
    flip_list = []
    for _, row in flips.iterrows():
        flip_list.append(
            {
                "claim_id": row["claim_id"],
                "replicate": int(row["replicate"]),
                "verdict_A": row["verdict_A"],
                "verdict_B": row["verdict_B"],
                "delta_final": round(
                    float(row[COL["final_B"]] - row[COL["final_A"]]), 3
                ),
            }
        )
    results["verdict_flips"] = {
        "n_flips": len(flip_list),
        "pct": round(100 * len(flip_list) / n, 1),
        "to_uncertain": sum(1 for f in flip_list if f["verdict_B"] == "uncertain"),
        "to_plausible": sum(1 for f in flip_list if f["verdict_B"] == "plausible"),
        "details": flip_list,
    }

    # --- 10. Isolamento PRO ---
    delta_pro = np.abs(pro_b - pro_a)
    results["pro_isolation"] = {
        "mean_abs_delta": float(np.mean(delta_pro)),
        "max_abs_delta": float(np.max(delta_pro)),
        "min_abs_delta": float(np.min(delta_pro)),
    }

    # --- 11. Gate distribution (su tutto il dataset, incluse astensioni) ---
    gate_a = COL["gate_label_A"]
    gate_b = COL["gate_label_B"]
    if gate_a in df.columns and gate_b in df.columns:
        results["gate_distribution"] = {
            "A": df[gate_a].value_counts().to_dict(),
            "B": df[gate_b].value_counts().to_dict(),
        }

    # --- 12. Repair failures (su tutto il dataset) ---
    repair_cols = [c for c in df.columns if "repair_fail_rate" in c.lower()]
    if repair_cols:
        repairs = []
        for _, row in df.iterrows():
            for rc in repair_cols:
                val = row.get(rc)
                if pd.notna(val) and float(val) > 0:
                    repairs.append(
                        {
                            "claim_id": row["claim_id"],
                            "replicate": int(row["replicate"]),
                            "column": rc,
                            "fail_rate": round(float(val), 3),
                        }
                    )
        results["repair_failures"] = repairs

    return results


# ---------------------------------------------------------------------------
# Stampa formattata
# ---------------------------------------------------------------------------
def print_results(r: dict) -> None:
    """Stampa i risultati in formato leggibile."""

    print("=" * 70)
    print("  ANALISI DOE — RISULTATI STATISTICI")
    print("=" * 70)

    # Overview
    print(
        f"\nDataset: {r['n_total']} coppie totali, {r['n_abstentions']} astensioni → {r['n_clean']} pulite"
    )
    if r["abstentions"]:
        print(
            "  Astensioni:",
            ", ".join(f"{a['claim_id']}-R{a['replicate']}" for a in r["abstentions"]),
        )

    # Descrittive
    d = r["descriptive"]
    print("\n--- Statistiche descrittive (n = {}) ---".format(r["n_clean"]))
    print(f"  contra_A:  {d['contra_A']['mean']:.3f} ± {d['contra_A']['std']:.3f}")
    print(f"  contra_B:  {d['contra_B']['mean']:.3f} ± {d['contra_B']['std']:.3f}")
    print(
        f"  Δ_contra:  {d['delta_contra']['mean']:+.3f} ± {d['delta_contra']['std']:.3f}"
    )
    print(f"  pro_A:     {d['pro_A']['mean']:.3f} ± {d['pro_A']['std']:.3f}")
    print(f"  pro_B:     {d['pro_B']['mean']:.3f} ± {d['pro_B']['std']:.3f}")
    print(f"  final_A:   {d['final_A']['mean']:.3f} ± {d['final_A']['std']:.3f}")
    print(f"  final_B:   {d['final_B']['mean']:.3f} ± {d['final_B']['std']:.3f}")

    # t-test
    t = r["ttest"]
    print("\n--- Paired t-test ---")
    print(f"  t({t['df']}) = {t['t_statistic']:.2f}")
    print(f"  p (bilaterale) = {t['p_two_tailed']:.6f}")
    print(f"  p (unilaterale) = {t['p_one_tailed']:.6f}")

    # Sign test
    s = r["sign_test"]
    print("\n--- Sign test (unilaterale, H1: B > A) ---")
    print(f"  B > A in {s['B_wins']}/{s['n_compared']} coppie ({s['B_pct']} %)")
    print(f"  p = {s['p_value']:.6f}")

    # Cohen's d
    print("\n--- Cohen's d (paired) ---")
    print(f"  d = {r['cohens_d']}")

    # Per dominio
    print("\n--- Breakdown per dominio ---")
    print(
        f"  {'Dominio':<16} {'n':>4} {'B>A':>5} {'B%':>7} {'Δ_mean':>8} {'d':>6} {'p_sign':>8}"
    )
    for dom in sorted(r["per_domain"]):
        v = r["per_domain"][dom]
        print(
            f"  {dom:<16} {v['n']:>4} {v['B_wins']:>5} {v['B_pct']:>6.1f}% {v['mean_delta']:>+8.3f} {v['cohens_d']:>6.2f} {v['sign_p']:>8.4f}"
        )

    # Sotto-metriche
    if r.get("sub_metrics"):
        print("\n--- Sotto-metriche ---")
        for k, v in r["sub_metrics"].items():
            print(
                f"  {k:<28} A={v['A']:.3f}  B={v['B']:.3f}  Δ={v['delta']:+.3f}  ({v['delta_pct']:+.1f}%)"
            )

    # Strutturali
    if r.get("structural"):
        print("\n--- Metriche strutturali ---")
        for k, v in r["structural"].items():
            print(
                f"  {k:<28} A={v.get('A', 0):.2f}  B={v['B']:.2f}  Δ={v.get('delta', v['B']):+.2f}"
            )

    # Consistenza
    c = r["consistency"]
    print("\n--- Consistenza intra-claim ---")
    print(f"  Tutte per B ({len(c['all_B'])}):       {', '.join(c['all_B'])}")
    print(f"  Maggioranza B ({len(c['majority_B'])}):  {', '.join(c['majority_B'])}")
    print(f"  Maggioranza A ({len(c['majority_A'])}):  {', '.join(c['majority_A'])}")
    print(f"  Tutte per A ({len(c['all_A'])}):       {', '.join(c['all_A'])}")

    # Verdict flips
    vf = r["verdict_flips"]
    print("\n--- Verdict flip ---")
    print(f"  {vf['n_flips']} flip su {r['n_clean']} coppie ({vf['pct']} %)")
    print(f"  → uncertain: {vf['to_uncertain']},  → plausible: {vf['to_plausible']}")
    for f in vf["details"]:
        print(
            f"    {f['claim_id']}-R{f['replicate']}: {f['verdict_A']} → {f['verdict_B']}  (Δ_final = {f['delta_final']:+.3f})"
        )

    # Isolamento PRO
    pi = r["pro_isolation"]
    print("\n--- Isolamento PRO ---")
    print(f"  mean |Δ_pro| = {pi['mean_abs_delta']:.6f}")
    print(f"  max  |Δ_pro| = {pi['max_abs_delta']:.6f}")

    # Gate
    if r.get("gate_distribution"):
        print("\n--- Gate distribution ---")
        for setup, dist in r["gate_distribution"].items():
            print(f"  Setup {setup}: {dict(dist)}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analisi statistica DoE A/B per LexCausa"
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV, help="Path al file run_summary.csv"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Se specificato, salva i risultati in JSON",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERRORE: file non trovato: {args.csv}", file=sys.stderr)
        sys.exit(1)

    results = analyze(args.csv)
    print_results(results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nRisultati salvati in: {args.out}")


if __name__ == "__main__":
    main()
