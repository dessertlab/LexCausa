#!/usr/bin/env python3
"""
AQA tuning with gold labels (verdict-level) and overkill control.

Pipeline:
1) Tune attack parameters with fixed decision threshold T=0.20
   Objective = macro_f1_present - lambda_overkill * overkill_rate
2) For top attack settings, tune score weights (alpha,beta,gamma)
3) For top configurations, tune global threshold T in {0.15,0.20,0.25,0.30}

This script replays AQA on saved reports in logs/aqa_reports and aligns
them with the provided JSONL dataset keyed by run_id.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(
    "tuning_data/tuning_dataset_aqa_with_gold_threshold_hint.jsonl"
)
REPORTS_DIR = PROJECT_ROOT / "logs" / "aqa_reports"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "tuning"


DEFAULT_MULTIPLIERS = {
    "contradiction": 2.0,
    "exception": 1.7,
    "derogation": 2.2,
    "extinction": 2.5,
    "factual_impediment": 1.2,  # fixed / low-variance
    "general_opposition": 1.05,  # fixed / low-variance
}

DEFAULT_RATIO_BY_TYPE = {
    "contradiction": 0.10,
    "exception": 0.10,
    "derogation": 0.05,
    "extinction": 0.00,
    "factual_impediment": 0.55,
    "general_opposition": 0.55,
}

DEFAULT_ALPHA_BETA_GAMMA = (0.30, 0.40, 0.30)


@dataclass
class GoldSample:
    run_id: str
    gold_label: str
    hint_threshold: float | None
    hint_region: str | None


@dataclass
class AttackParams:
    top_k: int
    damage_factor: float
    min_semantic_overlap: float
    min_strength_ratio: float
    multipliers: dict[str, float]
    ratio_by_type: dict[str, float]


@dataclass
class ScoreParams:
    alpha: float
    beta: float
    gamma: float


@dataclass
class EvalResult:
    objective: float
    macro_f1_present: float
    macro_f1_3way: float
    accuracy: float
    overkill_rate: float
    mean_final_score: float
    pred_distribution: dict[str, int]
    confusion: dict[str, dict[str, int]]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _map_gold_label(v: str | None) -> str:
    t = str(v or "").strip().lower()
    if t in {"support", "plausible", "pro"}:
        return "plausible"
    if t in {"counter", "implausible", "contra"}:
        return "implausible"
    return "uncertain"


def _predict_label(final_score: float, threshold: float) -> str:
    if final_score >= threshold:
        return "plausible"
    if final_score <= -threshold:
        return "implausible"
    return "uncertain"


def _macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    f1s = []
    for lbl in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp == lbl)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lbl and yp == lbl)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp != lbl)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return (
        sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
        if y_true
        else 0.0
    )


def _confusion(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {t: {p: 0 for p in labels} for t in labels}
    for yt, yp in zip(y_true, y_pred):
        if yt not in out:
            out[yt] = {p: 0 for p in labels}
        if yp not in out[yt]:
            out[yt][yp] = 0
        out[yt][yp] += 1
    return out


def _load_gold_dataset(path: Path) -> list[GoldSample]:
    rows: list[GoldSample] = []
    with path.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            item = json.loads(ln)
            hint = item.get("gold_threshold_hint") or {}
            rows.append(
                GoldSample(
                    run_id=str(item.get("run_id") or "").strip(),
                    gold_label=_map_gold_label(item.get("gold_verdict")),
                    hint_threshold=(
                        float(hint["T"])
                        if isinstance(hint.get("T"), (int, float))
                        else None
                    ),
                    hint_region=(
                        str(hint.get("expected_region")).strip().lower()
                        if hint.get("expected_region") is not None
                        else None
                    ),
                )
            )
    return rows


def _load_reports_map(reports_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(reports_dir.glob("*_aqa_report.json")):
        run_id = p.name.replace("_aqa_report.json", "")
        out[run_id] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _replay_single_report(
    report: dict[str, Any],
    attack_params: AttackParams,
    score_params: ScoreParams,
) -> tuple[float, float]:
    """
    Returns:
    - final_score (pro_net - contra_net)
    - overkill_rate for this run (% links with nesso=0)
    """
    links = {
        "pro": copy.deepcopy(report.get("links", {}).get("pro", [])),
        "contra": copy.deepcopy(report.get("links", {}).get("contra", [])),
    }

    # 1) Recompute base_score from alpha/beta/gamma
    for side in ("pro", "contra"):
        for link in links[side]:
            cog = float(link.get("cogency", 0.0) or 0.0)
            sem = float(link.get("semantics", 0.0) or 0.0)
            norm = float(link.get("norm_support", 0.0) or 0.0)
            base_new = _clamp01(
                score_params.alpha * cog
                + score_params.beta * norm
                + score_params.gamma * sem
            )
            link["_base_new"] = base_new

    # attacker base lookups
    pro_by_id = {str(link.get("link_id")): link for link in links["pro"]}
    contra_by_id = {str(link.get("link_id")): link for link in links["contra"]}

    # 2) Recompute attacks from stored attempts
    for side in ("pro", "contra"):
        for target in links[side]:
            target_base = float(target.get("_base_new", 0.0) or 0.0)
            recomputed = []
            attempts = target.get("attacks_received") or []
            for atk in attempts:
                rec = dict(atk)
                stage = str(rec.get("filter_stage", "") or "").strip().lower()
                if stage in {"domain", "nli_entailment"}:
                    rec["filtered"] = True
                    rec["attack_value"] = 0.0
                    recomputed.append(rec)
                    continue

                overlap = float(rec.get("overlap", 0.0) or 0.0)
                if overlap < attack_params.min_semantic_overlap:
                    rec["filtered"] = True
                    rec["filter_stage"] = "overlap"
                    rec["attack_value"] = 0.0
                    recomputed.append(rec)
                    continue

                nli_label = str(rec.get("nli_label", "") or "").strip().lower()
                if nli_label == "entailment":
                    rec["filtered"] = True
                    rec["filter_stage"] = "nli_entailment"
                    rec["attack_value"] = 0.0
                    recomputed.append(rec)
                    continue

                attacker_role = str(rec.get("attacker_role", "") or "").strip().lower()
                attacker_id = str(rec.get("attacker_link_id", "") or "")
                if attacker_role in {"counter", "contra"}:
                    attacker_link = contra_by_id.get(attacker_id)
                else:
                    attacker_link = pro_by_id.get(attacker_id)
                attacker_base = (
                    float(attacker_link.get("_base_new", 0.0))
                    if isinstance(attacker_link, dict)
                    else float(rec.get("attacker_base_score", 0.0) or 0.0)
                )

                atk_type = (
                    str(
                        rec.get("attack_type", "general_opposition")
                        or "general_opposition"
                    )
                    .strip()
                    .lower()
                )
                if atk_type not in attack_params.multipliers:
                    atk_type = "general_opposition"

                nli_bypass = (
                    bool(rec.get("nli_bypass", False)) or nli_label == "contradiction"
                )
                if not nli_bypass:
                    ratio = float(
                        attack_params.ratio_by_type.get(
                            atk_type,
                            attack_params.min_strength_ratio,
                        )
                    )
                    if ratio > 0.0 and attacker_base < ratio * target_base:
                        rec["filtered"] = True
                        rec["filter_stage"] = "strength_ratio"
                        rec["attack_value"] = 0.0
                        recomputed.append(rec)
                        continue

                raw_attack = overlap * attacker_base
                mult = float(attack_params.multipliers.get(atk_type, 1.0))
                boosted = raw_attack * mult
                excess = max(0.0, boosted - target_base)
                attack_value = excess * attack_params.damage_factor

                rec["filtered"] = False
                rec["attack_type"] = atk_type
                rec["type_multiplier"] = mult
                rec["boosted_attack"] = boosted
                rec["target_base_score"] = target_base
                rec["excess"] = excess
                rec["damage_factor"] = attack_params.damage_factor
                rec["attacker_base_score"] = attacker_base
                rec["attack_value"] = attack_value
                recomputed.append(rec)

            recomputed.sort(
                key=lambda x: float(x.get("attack_value", 0.0) or 0.0), reverse=True
            )

            active = [
                a
                for a in recomputed
                if not a.get("filtered", False)
                and float(a.get("attack_value", 0.0) or 0.0) >= 0.01
            ]
            if len(active) > attack_params.top_k:
                for overflow in active[attack_params.top_k :]:
                    overflow["filtered"] = True
                    overflow["filter_stage"] = "top_k"

            attacks_sum = sum(
                float(a.get("attack_value", 0.0) or 0.0)
                for a in recomputed
                if not a.get("filtered", False)
                and float(a.get("attack_value", 0.0) or 0.0) > 0.0
            )
            delta = float(target.get("precedent_delta", 0.0) or 0.0)
            nesso = _clamp01(target_base - attacks_sum + delta)

            target["_attacks_sum_new"] = attacks_sum
            target["_nesso_new"] = nesso

    def _avg(side: str, field: str) -> float:
        arr = links[side]
        if not arr:
            return 0.0
        return sum(float(x.get(field, 0.0) or 0.0) for x in arr) / len(arr)

    pro_net = _avg("pro", "_nesso_new")
    contra_net = _avg("contra", "_nesso_new")
    final_score = pro_net - contra_net

    all_links = links["pro"] + links["contra"]
    if not all_links:
        return final_score, 0.0
    overkill = sum(
        1 for link in all_links if float(link.get("_nesso_new", 0.0) or 0.0) <= 1e-12
    )
    overkill_rate = overkill / len(all_links)
    return final_score, overkill_rate


def _evaluate(
    samples: list[GoldSample],
    reports_map: dict[str, dict[str, Any]],
    attack_params: AttackParams,
    score_params: ScoreParams,
    threshold: float,
    lambda_overkill: float,
) -> EvalResult:
    y_true = []
    y_pred = []
    finals = []
    overkills = []

    for s in samples:
        report = reports_map.get(s.run_id)
        if report is None:
            continue
        final_score, overkill_rate = _replay_single_report(
            report=report,
            attack_params=attack_params,
            score_params=score_params,
        )
        pred = _predict_label(final_score, threshold)
        y_true.append(s.gold_label)
        y_pred.append(pred)
        finals.append(final_score)
        overkills.append(overkill_rate)

    labels_all = ["plausible", "uncertain", "implausible"]
    labels_present = sorted(set(y_true)) if y_true else labels_all
    mf1_3 = _macro_f1(y_true, y_pred, labels_all)
    mf1_present = _macro_f1(y_true, y_pred, labels_present)
    acc = _accuracy(y_true, y_pred)
    overkill = sum(overkills) / len(overkills) if overkills else 0.0
    mean_final = sum(finals) / len(finals) if finals else 0.0
    objective = mf1_present - lambda_overkill * overkill

    return EvalResult(
        objective=objective,
        macro_f1_present=mf1_present,
        macro_f1_3way=mf1_3,
        accuracy=acc,
        overkill_rate=overkill,
        mean_final_score=mean_final,
        pred_distribution=dict(Counter(y_pred)),
        confusion=_confusion(y_true, y_pred, labels_all),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune AQA weights + threshold from gold JSONL."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--lambda-overkill", type=float, default=0.25)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    samples = _load_gold_dataset(args.dataset)
    reports_map = _load_reports_map(args.reports_dir)
    samples = [s for s in samples if s.run_id in reports_map]

    if not samples:
        raise SystemExit("No overlapping runs between dataset and reports.")

    print(f"Samples used: {len(samples)}")
    print(f"Gold distribution: {dict(Counter(s.gold_label for s in samples))}")

    # ------------------------------------------------------------
    # STEP A: tune attack parameters with fixed threshold T=0.20
    # ------------------------------------------------------------
    fixed_threshold = 0.20
    step_a_rows: list[dict[str, Any]] = []
    for top_k in (2, 3):
        for damage in (0.40, 0.50):
            for m_contr in (1.8, 2.0):
                for m_exc in (1.5, 1.7):
                    for m_der in (2.0, 2.2):
                        for m_ext in (2.3, 2.5):
                            for r_contr in (0.05, 0.10):
                                for r_exc in (0.05, 0.10):
                                    for r_der in (0.00, 0.05):
                                        for r_ext in (0.00, 0.05):
                                            multipliers = dict(DEFAULT_MULTIPLIERS)
                                            multipliers.update(
                                                {
                                                    "contradiction": m_contr,
                                                    "exception": m_exc,
                                                    "derogation": m_der,
                                                    "extinction": m_ext,
                                                }
                                            )
                                            ratios = dict(DEFAULT_RATIO_BY_TYPE)
                                            ratios.update(
                                                {
                                                    "contradiction": r_contr,
                                                    "exception": r_exc,
                                                    "derogation": r_der,
                                                    "extinction": r_ext,
                                                }
                                            )
                                            a_params = AttackParams(
                                                top_k=top_k,
                                                damage_factor=damage,
                                                min_semantic_overlap=0.50,
                                                min_strength_ratio=1.20,
                                                multipliers=multipliers,
                                                ratio_by_type=ratios,
                                            )
                                            s_params = ScoreParams(
                                                *DEFAULT_ALPHA_BETA_GAMMA
                                            )
                                            ev = _evaluate(
                                                samples=samples,
                                                reports_map=reports_map,
                                                attack_params=a_params,
                                                score_params=s_params,
                                                threshold=fixed_threshold,
                                                lambda_overkill=args.lambda_overkill,
                                            )
                                            step_a_rows.append(
                                                {
                                                    "attack_params": asdict(a_params),
                                                    "score_params": asdict(s_params),
                                                    "threshold": fixed_threshold,
                                                    "eval": asdict(ev),
                                                }
                                            )

    step_a_rows.sort(key=lambda x: x["eval"]["objective"], reverse=True)
    top_attack_rows = step_a_rows[:8]

    # ------------------------------------------------------------
    # STEP A2: tune alpha/beta/gamma over top attack sets
    # ------------------------------------------------------------
    abg_candidates = [
        (0.30, 0.40, 0.30),
        (0.35, 0.35, 0.30),
        (0.25, 0.45, 0.30),
        (0.30, 0.35, 0.35),
        (0.25, 0.40, 0.35),
    ]
    step_a2_rows: list[dict[str, Any]] = []
    for row in top_attack_rows:
        attack_params_row = row["attack_params"]
        if not isinstance(attack_params_row, dict):
            continue
        ap = AttackParams(**attack_params_row)
        for abg in abg_candidates:
            sp = ScoreParams(*abg)
            ev = _evaluate(
                samples=samples,
                reports_map=reports_map,
                attack_params=ap,
                score_params=sp,
                threshold=fixed_threshold,
                lambda_overkill=args.lambda_overkill,
            )
            step_a2_rows.append(
                {
                    "attack_params": asdict(ap),
                    "score_params": asdict(sp),
                    "threshold": fixed_threshold,
                    "eval": asdict(ev),
                }
            )

    step_a2_rows.sort(key=lambda x: x["eval"]["objective"], reverse=True)
    top_joint = step_a2_rows[: args.top_n]

    # ------------------------------------------------------------
    # STEP B: threshold sweep for top joint configs
    # ------------------------------------------------------------
    final_rows: list[dict[str, Any]] = []
    for row in top_joint:
        attack_params_row = row["attack_params"]
        score_params_row = row["score_params"]
        if not isinstance(attack_params_row, dict) or not isinstance(
            score_params_row, dict
        ):
            continue
        ap = AttackParams(**attack_params_row)
        sp = ScoreParams(**score_params_row)
        best_for_cfg: dict[str, Any] | None = None
        for t in (0.15, 0.20, 0.25, 0.30):
            ev = _evaluate(
                samples=samples,
                reports_map=reports_map,
                attack_params=ap,
                score_params=sp,
                threshold=t,
                lambda_overkill=args.lambda_overkill,
            )
            candidate: dict[str, Any] = {
                "attack_params": asdict(ap),
                "score_params": asdict(sp),
                "threshold": t,
                "eval": asdict(ev),
            }
            if best_for_cfg is None or (
                isinstance(best_for_cfg.get("eval"), dict)
                and candidate["eval"]["objective"] > best_for_cfg["eval"]["objective"]
            ):
                best_for_cfg = candidate
        if best_for_cfg:
            final_rows.append(best_for_cfg)

    final_rows.sort(key=lambda x: x["eval"]["objective"], reverse=True)
    best = final_rows[0]

    # Baseline (current defaults)
    baseline_attack = AttackParams(
        top_k=2,
        damage_factor=0.5,
        min_semantic_overlap=0.5,
        min_strength_ratio=1.2,
        multipliers={
            "contradiction": 1.9,
            "exception": 1.6,
            "derogation": 2.1,
            "extinction": 2.4,
            "factual_impediment": 1.2,
            "general_opposition": 1.05,
        },
        ratio_by_type={
            "contradiction": 0.0,
            "exception": 0.0,
            "derogation": 0.0,
            "extinction": 0.0,
            "factual_impediment": 0.55,
            "general_opposition": 0.55,
        },
    )
    baseline_score = ScoreParams(0.30, 0.40, 0.30)
    baseline_eval = _evaluate(
        samples=samples,
        reports_map=reports_map,
        attack_params=baseline_attack,
        score_params=baseline_score,
        threshold=0.20,
        lambda_overkill=args.lambda_overkill,
    )

    print("\nBaseline (current config @T=0.20):")
    print(json.dumps(asdict(baseline_eval), ensure_ascii=False, indent=2))
    print("\nBest tuned:")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    args.output.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output / f"aqa_gold_tuning_{ts}.json"
    latest = args.output / "aqa_gold_tuning_latest.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "reports_dir": str(args.reports_dir),
        "samples_used": len(samples),
        "gold_distribution": dict(Counter(s.gold_label for s in samples)),
        "lambda_overkill": args.lambda_overkill,
        "step_a_grid_size": len(step_a_rows),
        "step_a_top": step_a_rows[: args.top_n],
        "step_a2_top": step_a2_rows[: args.top_n],
        "final_top": final_rows[: args.top_n],
        "best": best,
        "baseline_current_config": {
            "attack_params": asdict(baseline_attack),
            "score_params": asdict(baseline_score),
            "threshold": 0.20,
            "eval": asdict(baseline_eval),
        },
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved:\n- {out}\n- {latest}")


if __name__ == "__main__":
    main()
