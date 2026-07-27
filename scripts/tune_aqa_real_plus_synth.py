#!/usr/bin/env python3
"""
Tune AQA attack/score parameters using:
- real gold dataset (verdict supervision)
- synthetic attack-unit dataset (stability / mechanics)

Design:
1) Tune attack parameters with fixed global threshold T=0.20
2) Tune alpha/beta/gamma on top attack candidates
3) Sweep global threshold T in {0.15, 0.20, 0.25, 0.30}

Objective (combined):
  w_real  * (macro_f1_real_present  - lambda_real  * overkill_real)
+ w_synth * (macro_f1_synth_present - lambda_synth * overkill_synth)
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
DEFAULT_REAL_DATASET = Path(
    "tuning_data/tuning_dataset_aqa_with_gold_threshold_hint.jsonl"
)
DEFAULT_SYNTH_DATASET = Path(
    "tuning_data/aqa_synthetic_cases_uncertain_counter.jsonl"
)
REPORTS_DIR = PROJECT_ROOT / "logs" / "aqa_reports"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "tuning"


DEFAULT_MULTIPLIERS = {
    "contradiction": 2.0,
    "exception": 1.7,
    "derogation": 2.2,
    "extinction": 2.5,
    # Keep these low-variance to avoid overfitting
    "factual_impediment": 1.2,
    "general_opposition": 1.05,
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
class RealSample:
    run_id: str
    gold_label: str


@dataclass
class SyntheticSample:
    run_id: str
    gold_label: str
    support_links: list[dict[str, Any]]
    counter_links: list[dict[str, Any]]
    attacks: list[dict[str, Any]]


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
    scores = []
    for lbl in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp == lbl)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lbl and yp == lbl)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp != lbl)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        scores.append(f1)
    return sum(scores) / len(scores) if scores else 0.0


def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return (
        sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
        if y_true
        else 0.0
    )


def _confusion(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict[str, dict[str, int]]:
    out = {t: {p: 0 for p in labels} for t in labels}
    for yt, yp in zip(y_true, y_pred):
        if yt not in out:
            out[yt] = {p: 0 for p in labels}
        if yp not in out[yt]:
            out[yt][yp] = 0
        out[yt][yp] += 1
    return out


def _load_real_dataset(path: Path) -> list[RealSample]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            item = json.loads(ln)
            rows.append(
                RealSample(
                    run_id=str(item.get("run_id") or "").strip(),
                    gold_label=_map_gold_label(item.get("gold_verdict")),
                )
            )
    return rows


def _load_synth_dataset(path: Path) -> list[SyntheticSample]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            item = json.loads(ln)
            rows.append(
                SyntheticSample(
                    run_id=str(item.get("run_id") or "").strip(),
                    gold_label=_map_gold_label(item.get("gold_verdict")),
                    support_links=list(item.get("support_links") or []),
                    counter_links=list(item.get("counter_links") or []),
                    attacks=list(item.get("attacks") or []),
                )
            )
    return rows


def _load_reports_map(reports_dir: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted(reports_dir.glob("*_aqa_report.json")):
        run_id = p.name.replace("_aqa_report.json", "")
        out[run_id] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _replay_real_report(
    report: dict[str, Any],
    attack_params: AttackParams,
    score_params: ScoreParams,
) -> tuple[float, float]:
    links = {
        "pro": copy.deepcopy(report.get("links", {}).get("pro", [])),
        "contra": copy.deepcopy(report.get("links", {}).get("contra", [])),
    }

    # Recompute base score from components
    for side in ("pro", "contra"):
        for link in links[side]:
            cog = float(link.get("cogency", 0.0) or 0.0)
            sem = float(link.get("semantics", 0.0) or 0.0)
            norm = float(link.get("norm_support", 0.0) or 0.0)
            link["_base_new"] = _clamp01(
                score_params.alpha * cog
                + score_params.beta * norm
                + score_params.gamma * sem
            )

    pro_by_id = {str(link.get("link_id")): link for link in links["pro"]}
    contra_by_id = {str(link.get("link_id")): link for link in links["contra"]}

    for side in ("pro", "contra"):
        for target in links[side]:
            target_base = float(target.get("_base_new", 0.0) or 0.0)
            updated = []
            for atk in target.get("attacks_received") or []:
                rec = dict(atk)
                stage = str(rec.get("filter_stage", "") or "").strip().lower()
                if stage in {"domain", "nli_entailment"}:
                    rec["filtered"] = True
                    rec["attack_value"] = 0.0
                    updated.append(rec)
                    continue

                overlap = float(rec.get("overlap", 0.0) or 0.0)
                if overlap < attack_params.min_semantic_overlap:
                    rec["filtered"] = True
                    rec["filter_stage"] = "overlap"
                    rec["attack_value"] = 0.0
                    updated.append(rec)
                    continue

                nli_label = str(rec.get("nli_label", "") or "").strip().lower()
                if nli_label == "entailment":
                    rec["filtered"] = True
                    rec["filter_stage"] = "nli_entailment"
                    rec["attack_value"] = 0.0
                    updated.append(rec)
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
                            atk_type, attack_params.min_strength_ratio
                        )
                    )
                    if ratio > 0.0 and attacker_base < ratio * target_base:
                        rec["filtered"] = True
                        rec["filter_stage"] = "strength_ratio"
                        rec["attack_value"] = 0.0
                        updated.append(rec)
                        continue

                raw_attack = overlap * attacker_base
                mult = float(attack_params.multipliers.get(atk_type, 1.0))
                boosted = raw_attack * mult
                excess = max(0.0, boosted - target_base)
                attack_value = excess * attack_params.damage_factor

                rec["filtered"] = False
                rec["attack_type"] = atk_type
                rec["attacker_base_score"] = attacker_base
                rec["type_multiplier"] = mult
                rec["boosted_attack"] = boosted
                rec["target_base_score"] = target_base
                rec["excess"] = excess
                rec["damage_factor"] = attack_params.damage_factor
                rec["attack_value"] = attack_value
                updated.append(rec)

            updated.sort(
                key=lambda x: float(x.get("attack_value", 0.0) or 0.0), reverse=True
            )
            active = [
                a
                for a in updated
                if not a.get("filtered", False)
                and float(a.get("attack_value", 0.0) or 0.0) >= 0.01
            ]
            if len(active) > attack_params.top_k:
                for overflow in active[attack_params.top_k :]:
                    overflow["filtered"] = True
                    overflow["filter_stage"] = "top_k"

            attacks_sum = sum(
                float(a.get("attack_value", 0.0) or 0.0)
                for a in updated
                if not a.get("filtered", False)
                and float(a.get("attack_value", 0.0) or 0.0) > 0.0
            )
            target["_attacks_sum_new"] = attacks_sum
            delta = float(target.get("precedent_delta", 0.0) or 0.0)
            target["_nesso_new"] = _clamp01(target_base - attacks_sum + delta)

    def _avg(side: str, field: str) -> float:
        arr = links[side]
        return (
            sum(float(x.get(field, 0.0) or 0.0) for x in arr) / len(arr) if arr else 0.0
        )

    final_score = _avg("pro", "_nesso_new") - _avg("contra", "_nesso_new")
    all_links = links["pro"] + links["contra"]
    overkill = (
        sum(
            1
            for link in all_links
            if float(link.get("_nesso_new", 0.0) or 0.0) <= 1e-12
        )
        / len(all_links)
        if all_links
        else 0.0
    )
    return final_score, overkill


def _replay_synthetic_sample(
    sample: SyntheticSample,
    attack_params: AttackParams,
) -> tuple[float, float]:
    support = [copy.deepcopy(x) for x in sample.support_links]
    counter = [copy.deepcopy(x) for x in sample.counter_links]
    links = {
        "support": {str(link.get("link_id")): link for link in support},
        "counter": {str(link.get("link_id")): link for link in counter},
    }
    all_by_id = {**links["support"], **links["counter"]}

    for link in all_by_id.values():
        link["_base_new"] = float(link.get("base", 0.0) or 0.0)
        link["_attacks_received"] = []

    # Build attack candidates from synthetic edges
    for atk in sample.attacks:
        src_id = str(atk.get("source_link_id") or "")
        tgt_id = str(atk.get("target_link_id") or "")
        src = all_by_id.get(src_id)
        tgt = all_by_id.get(tgt_id)
        if src is None or tgt is None:
            continue

        atk_type = str(atk.get("type") or "general_opposition").strip().lower()
        if atk_type not in attack_params.multipliers:
            atk_type = "general_opposition"

        # "strength_raw" is pre-multiplier (provided by dataset author).
        raw_attack = float(atk.get("strength_raw", 0.0) or 0.0)
        source_base = float(src.get("_base_new", 0.0))
        target_base = float(tgt.get("_base_new", 0.0))

        ratio = float(
            attack_params.ratio_by_type.get(atk_type, attack_params.min_strength_ratio)
        )
        if ratio > 0.0 and source_base < ratio * target_base:
            tgt["_attacks_received"].append(
                {
                    "attack_value": 0.0,
                    "filtered": True,
                    "filter_stage": "strength_ratio",
                }
            )
            continue

        mult = float(attack_params.multipliers.get(atk_type, 1.0))
        boosted = raw_attack * mult
        excess = max(0.0, boosted - target_base)
        attack_value = excess * attack_params.damage_factor

        tgt["_attacks_received"].append(
            {
                "attack_type": atk_type,
                "source_link_id": src_id,
                "target_link_id": tgt_id,
                "raw_attack": raw_attack,
                "source_base": source_base,
                "target_base": target_base,
                "attack_value": attack_value,
                "filtered": False,
            }
        )

    # top-k per target
    for link in all_by_id.values():
        arr = sorted(
            link["_attacks_received"],
            key=lambda x: float(x.get("attack_value", 0.0) or 0.0),
            reverse=True,
        )
        active = [
            a
            for a in arr
            if not a.get("filtered", False)
            and float(a.get("attack_value", 0.0) or 0.0) >= 0.01
        ]
        if len(active) > attack_params.top_k:
            for overflow in active[attack_params.top_k :]:
                overflow["filtered"] = True
                overflow["filter_stage"] = "top_k"
        attacks_sum = sum(
            float(a.get("attack_value", 0.0) or 0.0)
            for a in arr
            if not a.get("filtered", False)
            and float(a.get("attack_value", 0.0) or 0.0) > 0.0
        )
        link["_nesso_new"] = _clamp01(float(link.get("_base_new", 0.0)) - attacks_sum)

    def _avg(arr: list[dict[str, Any]]) -> float:
        return (
            sum(float(x.get("_nesso_new", 0.0) or 0.0) for x in arr) / len(arr)
            if arr
            else 0.0
        )

    final_score = _avg(support) - _avg(counter)
    all_links = support + counter
    overkill = (
        sum(
            1
            for link in all_links
            if float(link.get("_nesso_new", 0.0) or 0.0) <= 1e-12
        )
        / len(all_links)
        if all_links
        else 0.0
    )
    return final_score, overkill


def _evaluate_predictions(
    y_true: list[str],
    y_pred: list[str],
    finals: list[float],
    overkills: list[float],
    lambda_overkill: float,
) -> EvalResult:
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


def _evaluate_real(
    samples: list[RealSample],
    reports_map: dict[str, dict[str, Any]],
    attack_params: AttackParams,
    score_params: ScoreParams,
    threshold: float,
    lambda_overkill: float,
) -> EvalResult:
    y_true: list[str] = []
    y_pred: list[str] = []
    finals: list[float] = []
    overkills: list[float] = []
    for s in samples:
        report = reports_map.get(s.run_id)
        if report is None:
            continue
        final, overkill = _replay_real_report(report, attack_params, score_params)
        y_true.append(s.gold_label)
        y_pred.append(_predict_label(final, threshold))
        finals.append(final)
        overkills.append(overkill)
    return _evaluate_predictions(y_true, y_pred, finals, overkills, lambda_overkill)


def _evaluate_synth(
    samples: list[SyntheticSample],
    attack_params: AttackParams,
    threshold: float,
    lambda_overkill: float,
) -> EvalResult:
    y_true: list[str] = []
    y_pred: list[str] = []
    finals: list[float] = []
    overkills: list[float] = []
    for s in samples:
        final, overkill = _replay_synthetic_sample(s, attack_params)
        y_true.append(s.gold_label)
        y_pred.append(_predict_label(final, threshold))
        finals.append(final)
        overkills.append(overkill)
    return _evaluate_predictions(y_true, y_pred, finals, overkills, lambda_overkill)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune AQA with real + synthetic datasets."
    )
    parser.add_argument("--real-dataset", type=Path, default=DEFAULT_REAL_DATASET)
    parser.add_argument("--synthetic-dataset", type=Path, default=DEFAULT_SYNTH_DATASET)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--lambda-overkill-real", type=float, default=0.25)
    parser.add_argument("--lambda-overkill-synth", type=float, default=0.35)
    parser.add_argument("--weight-real", type=float, default=0.70)
    parser.add_argument("--weight-synth", type=float, default=0.30)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reports_map = _load_reports_map(args.reports_dir)
    real_samples = [
        s for s in _load_real_dataset(args.real_dataset) if s.run_id in reports_map
    ]
    synth_samples = _load_synth_dataset(args.synthetic_dataset)

    if not real_samples:
        raise SystemExit("No overlapping real samples with local AQA reports.")
    if not synth_samples:
        raise SystemExit("Synthetic dataset is empty.")

    print(f"Real samples used: {len(real_samples)}")
    print(
        f"Real gold distribution: {dict(Counter(s.gold_label for s in real_samples))}"
    )
    print(f"Synthetic samples used: {len(synth_samples)}")
    print(
        f"Synthetic gold distribution: {dict(Counter(s.gold_label for s in synth_samples))}"
    )

    # ------------------------------------------------------------
    # STEP A: attack params @ fixed T=0.20
    # ------------------------------------------------------------
    fixed_threshold = 0.20
    step_a = []
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
                                            ap = AttackParams(
                                                top_k=top_k,
                                                damage_factor=damage,
                                                min_semantic_overlap=0.50,
                                                min_strength_ratio=1.20,
                                                multipliers=multipliers,
                                                ratio_by_type=ratios,
                                            )
                                            sp = ScoreParams(*DEFAULT_ALPHA_BETA_GAMMA)
                                            ev_real = _evaluate_real(
                                                real_samples,
                                                reports_map,
                                                ap,
                                                sp,
                                                fixed_threshold,
                                                args.lambda_overkill_real,
                                            )
                                            ev_synth = _evaluate_synth(
                                                synth_samples,
                                                ap,
                                                fixed_threshold,
                                                args.lambda_overkill_synth,
                                            )
                                            obj = (
                                                args.weight_real * ev_real.objective
                                                + args.weight_synth * ev_synth.objective
                                            )
                                            step_a.append(
                                                {
                                                    "attack_params": asdict(ap),
                                                    "score_params": asdict(sp),
                                                    "threshold": fixed_threshold,
                                                    "objective_combined": obj,
                                                    "eval_real": asdict(ev_real),
                                                    "eval_synth": asdict(ev_synth),
                                                }
                                            )

    step_a.sort(key=lambda x: x["objective_combined"], reverse=True)
    top_attack = step_a[:8]

    # ------------------------------------------------------------
    # STEP A2: alpha/beta/gamma over top attack settings
    # ------------------------------------------------------------
    abg_candidates = [
        (0.30, 0.40, 0.30),
        (0.35, 0.35, 0.30),
        (0.25, 0.45, 0.30),
        (0.30, 0.35, 0.35),
        (0.25, 0.40, 0.35),
    ]
    step_a2 = []
    for row in top_attack:
        ap = AttackParams(**row["attack_params"])
        for abg in abg_candidates:
            sp = ScoreParams(*abg)
            ev_real = _evaluate_real(
                real_samples,
                reports_map,
                ap,
                sp,
                fixed_threshold,
                args.lambda_overkill_real,
            )
            ev_synth = _evaluate_synth(
                synth_samples,
                ap,
                fixed_threshold,
                args.lambda_overkill_synth,
            )
            obj = (
                args.weight_real * ev_real.objective
                + args.weight_synth * ev_synth.objective
            )
            step_a2.append(
                {
                    "attack_params": asdict(ap),
                    "score_params": asdict(sp),
                    "threshold": fixed_threshold,
                    "objective_combined": obj,
                    "eval_real": asdict(ev_real),
                    "eval_synth": asdict(ev_synth),
                }
            )

    step_a2.sort(key=lambda x: x["objective_combined"], reverse=True)
    top_joint = step_a2[: args.top_n]

    # ------------------------------------------------------------
    # STEP B: threshold sweep (global threshold only)
    # ------------------------------------------------------------
    final_rows = []
    for row in top_joint:
        ap = AttackParams(**row["attack_params"])
        sp = ScoreParams(**row["score_params"])
        best_for_cfg = None
        for t in (0.15, 0.20, 0.25, 0.30):
            ev_real = _evaluate_real(
                real_samples,
                reports_map,
                ap,
                sp,
                t,
                args.lambda_overkill_real,
            )
            ev_synth = _evaluate_synth(
                synth_samples,
                ap,
                t,
                args.lambda_overkill_synth,
            )
            obj = (
                args.weight_real * ev_real.objective
                + args.weight_synth * ev_synth.objective
            )
            candidate = {
                "attack_params": asdict(ap),
                "score_params": asdict(sp),
                "threshold": t,
                "objective_combined": obj,
                "eval_real": asdict(ev_real),
                "eval_synth": asdict(ev_synth),
            }
            if (
                best_for_cfg is None
                or candidate["objective_combined"] > best_for_cfg["objective_combined"]
            ):
                best_for_cfg = candidate
        if best_for_cfg:
            final_rows.append(best_for_cfg)

    final_rows.sort(key=lambda x: x["objective_combined"], reverse=True)
    best = final_rows[0]

    print("\nBest tuned (real+synth):")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    args.output.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output / f"aqa_real_synth_tuning_{ts}.json"
    latest = args.output / "aqa_real_synth_tuning_latest.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "real_dataset": str(args.real_dataset),
        "synthetic_dataset": str(args.synthetic_dataset),
        "reports_dir": str(args.reports_dir),
        "samples_real_used": len(real_samples),
        "samples_synth_used": len(synth_samples),
        "real_distribution": dict(Counter(s.gold_label for s in real_samples)),
        "synth_distribution": dict(Counter(s.gold_label for s in synth_samples)),
        "objective_weights": {
            "weight_real": args.weight_real,
            "weight_synth": args.weight_synth,
            "lambda_overkill_real": args.lambda_overkill_real,
            "lambda_overkill_synth": args.lambda_overkill_synth,
        },
        "step_a_grid_size": len(step_a),
        "step_a_top": step_a[: args.top_n],
        "step_a2_top": step_a2[: args.top_n],
        "final_top": final_rows[: args.top_n],
        "best": best,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved:\n- {out}\n- {latest}")


if __name__ == "__main__":
    main()
