#!/usr/bin/env python3
"""
LexCausa – DoE Batch Runner
============================
Esegue il Design of Experiments su tutti i claim coperti definiti in claims.md.

Per ogni claim e ogni replica viene eseguita la pipeline completa in due setup:
  * Setup A (baseline):  counter_enable_causality = False
  * Setup B (treatment): counter_enable_causality = True

I risultati vengono organizzati in:
  experiments/causal_taxonomy_ablation/<run>/<CLAIM_ID>/R<n>/setup_A/  (response, AQA, log)
  experiments/causal_taxonomy_ablation/<run>/<CLAIM_ID>/R<n>/setup_B/
  experiments/causal_taxonomy_ablation/<run>/<CLAIM_ID>/R<n>/doe/       (A/B comparison report)

Al termine della run viene generato un file di riepilogo con tutte le metriche
(run_summary.json + run_summary.csv) nella directory radice della run.

Uso:
  python scripts/run_doe_batch.py \
      --config ablation_settings.json \
      --replicates 2 \
      --domains CIVILE,PENALE,AMMINISTRATIVO,MIXED

Flags utili:
  --resume          Salta claim/repliche con artefatti già presenti
  --run-name XXX    Nome esplicito della run (default: auto-incrementale)
  --start-from C3   Riprendi dall'ID claim specificato
  --only C1,P2      Esegui solo i claim indicati
  --dry-run         Mostra il piano senza eseguire
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLAIMS = PROJECT_ROOT / "claims.md"
DEFAULT_CONFIG = PROJECT_ROOT / "ablation_settings.json"
BATCH_RUNS_DIR = PROJECT_ROOT / "experiments" / "causal_taxonomy_ablation"


# ──────────────────────────────────────────────────────────────────────
# Claims parser (aligned with generate_run_plan.py)
# ──────────────────────────────────────────────────────────────────────
class ClaimCase:
    """A single claim parsed from claims.md."""

    def __init__(
        self, claim_id: str, title: str, text: str, domain: str, covered: bool
    ):
        self.claim_id = claim_id
        self.title = title
        self.text = text
        self.domain = domain
        self.covered = covered

    def __repr__(self) -> str:
        return f"<Claim {self.claim_id} [{self.domain}] covered={self.covered}>"


def _normalize_domain(heading: str) -> str:
    h = heading.upper()
    if "CIVILI" in h:
        return "CIVILE"
    if "PENALI" in h:
        return "PENALE"
    if "AMMINISTRATIVI" in h:
        return "AMMINISTRATIVO"
    if "MIXED" in h:
        return "MIXED"
    if "NON COPERTI" in h:
        return "NON_COPERTO"
    return "UNKNOWN"


def parse_claims_md(path: Path) -> list[ClaimCase]:
    """Parse claims.md and return structured claim list."""
    lines = path.read_text(encoding="utf-8").splitlines()
    block_domain = "UNKNOWN"
    block_covered = False
    claims: list[ClaimCase] = []
    cur_id = ""
    cur_title = ""
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_id, cur_title, cur_lines
        if not cur_id:
            return
        text = re.sub(r"\s+", " ", " ".join(cur_lines)).strip()
        if text:
            claims.append(
                ClaimCase(cur_id, cur_title, text, block_domain, block_covered)
            )
        cur_id = ""
        cur_title = ""
        cur_lines = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("## "):
            flush()
            block_domain = _normalize_domain(line)
            block_covered = "(COPERTI)" in line.upper()
            continue
        m = re.match(r"^###\s+([A-Z0-9]+)\s*-\s*(.+)$", line)
        if m:
            flush()
            cur_id = m.group(1).strip()
            cur_title = m.group(2).strip()
            continue
        if line.startswith("- Gap normativo:"):
            continue
        if line.startswith("Testo completo:"):
            line = line.replace("Testo completo:", "", 1).strip()
        if cur_id:
            cur_lines.append(line)

    flush()
    return claims


# ──────────────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────────────
def post_json(url: str, payload: dict, timeout_sec: int) -> tuple[int, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            return int(resp.getcode()), resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")
    except (TimeoutError, error.URLError, OSError) as exc:
        # Socket timeout, connection refused, DNS failure, etc.
        return 0, json.dumps({"error": f"network_error: {exc}"})


def post_json_with_retries(
    url: str,
    payload: dict,
    *,
    timeout_sec: int = 3600,
    max_retries: int = 0,
    backoff_sec: int = 30,
) -> tuple[str, dict]:
    """POST with retries. Returns (status, parsed_json).

    *max_retries=0* (default) means **infinite retries** – the call will
    keep retrying on timeouts and transient errors (5xx, 429, network)
    until it either succeeds or receives a deterministic client error
    (4xx other than 429).  Set a positive value to cap attempts.
    """
    attempt = 0
    data: dict = {}
    max_backoff = 300  # cap wait at 5 min

    while True:
        attempt += 1
        http_code, text = post_json(url, payload, timeout_sec)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"raw_response": text}
        if not isinstance(data, dict):
            data = {"raw_response": data}

        # Success
        if 200 <= http_code < 300:
            return "ok", data

        # Deterministic client error (except 429 rate-limit) → stop
        if 400 <= http_code < 500 and http_code != 429:
            err = str(data.get("error", text))[:200]
            print(f"      [FAIL] http={http_code} (non-retryable) err={err}")
            return "failed", data

        # Retryable failure (timeout/network=0, 5xx, 429)
        is_timeout = http_code == 0
        err = str(data.get("error", text))[:500]
        label = "TIMEOUT" if is_timeout else f"http={http_code}"
        limit_str = str(max_retries) if max_retries > 0 else "∞"
        print(f"      [WARN] attempt {attempt}/{limit_str} " f"{label} err={err[:120]}")

        # If we have a finite cap and exceeded it, give up
        if max_retries > 0 and attempt >= max_retries:
            return "failed", data

        wait = min(backoff_sec * attempt, max_backoff)
        print(f"      [WAIT] retrying in {wait}s ...")
        time.sleep(wait)


# ──────────────────────────────────────────────────────────────────────
# Metric extraction helpers (aligned with extract_metrics.py)
# ──────────────────────────────────────────────────────────────────────
def dig(data: Any, *keys: str, default: Any = None) -> Any:
    cur = data
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _f(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _count_repair_failed(report: dict) -> int:
    checks = report.get("citation_checks") or []
    return sum(
        1
        for c in checks
        if isinstance(c, dict)
        and str(c.get("mismatch_action", "")).lower() == "repair_failed"
    )


def _classify_abstention(gate: dict, counter: dict) -> str:
    """Classify abstention into reasoned / gate / empty."""
    label = str(gate.get("label", "")).upper()
    counter_reason = str(counter.get("abstention_reason", "")).lower()

    # Counter explicitly decided there are no valid attacks
    if (
        label == "ALREADY_ABSTAINED"
        or "no_attacks" in counter_reason
        or "safety" in counter_reason
    ):
        return "reasoned"
    # Polisher gate blocked (insufficient material or non-opposing)
    if label in ("INSUFFICIENT_MATERIAL", "NON_OPPOSING"):
        return "gate"
    # Empty chain or generic failure
    chain = counter.get("reasoning_chain") or []
    if not chain and not counter.get("raw_response"):
        return "empty"
    # Fallback: if gate says abstain but we can't classify further
    return "gate"


def extract_setup_metrics(response: dict) -> dict[str, Any]:
    """Extract AQA and consistency metrics from a single pipeline response."""
    evaluation = dig(response, "evaluation", default={}) or {}
    aqa = dig(evaluation, "aqa_report", default={}) or {}
    consistency = dig(evaluation, "consistency_report", default={}) or {}
    reasoner_report = dig(consistency, "reasoner", default={}) or {}
    counter_report = dig(consistency, "counter_reasoner", default={}) or {}
    gate = (
        dig(evaluation, "counter_reasoner_gate", default={})
        or dig(consistency, "counter_reasoner_gate", default={})
        or {}
    )
    counter = dig(response, "counter_reasoner", default={}) or {}
    chain_scores = dig(aqa, "chain_scores", default={}) or {}
    chain_pro = dig(chain_scores, "pro", default={}) or {}
    chain_contra = dig(chain_scores, "contra", default={}) or {}
    net = dig(aqa, "net_plausibility", default={}) or {}
    aqa_links = dig(aqa, "links", default={}) or {}

    reasoner_total = _i(reasoner_report.get("total_citations"))
    counter_total = _i(counter_report.get("total_citations"))
    reasoner_fail = _count_repair_failed(reasoner_report)
    counter_fail = _count_repair_failed(counter_report)

    reasoner_data = dig(response, "reasoner", default={}) or {}
    counter_data = dig(response, "counter_reasoner", default={}) or {}
    reasoner_chain = reasoner_data.get("reasoning_chain") or []
    counter_chain = counter_data.get("reasoning_chain") or []

    return {
        # Verdict & net plausibility
        "aqa_verdict": aqa.get("verdict"),
        "aqa_net_final": _f(net.get("final")),
        "aqa_net_pro": _f(net.get("pro")),
        "aqa_net_contra": _f(net.get("contra")),
        # Link counts (mirrors frontend extractDoeMetrics)
        "pro_links_count": (
            len(aqa_links.get("pro") or [])
            if isinstance(aqa_links.get("pro"), list)
            else 0
        ),
        "contra_links_count": (
            len(aqa_links.get("contra") or [])
            if isinstance(aqa_links.get("contra"), list)
            else 0
        ),
        # Chain-level scores
        "pro_cogency_avg": _f(chain_pro.get("cogency_avg")),
        "pro_semantics_avg": _f(chain_pro.get("semantics_avg")),
        "pro_norm_support_avg": _f(chain_pro.get("norm_support_avg")),
        "contra_cogency_avg": _f(chain_contra.get("cogency_avg")),
        "contra_semantics_avg": _f(chain_contra.get("semantics_avg")),
        "contra_norm_support_avg": _f(chain_contra.get("norm_support_avg")),
        # Consistency / repair
        "reasoner_total_citations": reasoner_total,
        "counter_total_citations": counter_total,
        "reasoner_repaired_citations": _i(reasoner_report.get("repaired_citations")),
        "counter_repaired_citations": _i(counter_report.get("repaired_citations")),
        "reasoner_dropped_citations": _i(reasoner_report.get("dropped_citations")),
        "counter_dropped_citations": _i(counter_report.get("dropped_citations")),
        "reasoner_repair_fail_count": reasoner_fail,
        "counter_repair_fail_count": counter_fail,
        "reasoner_repair_fail_rate": (
            reasoner_fail / reasoner_total
            if reasoner_total and reasoner_total > 0
            else None
        ),
        "counter_repair_fail_rate": (
            counter_fail / counter_total
            if counter_total and counter_total > 0
            else None
        ),
        # Counter gate
        "counter_gate_checked": gate.get("checked"),
        "counter_gate_label": gate.get("label"),
        "counter_gate_abstain": gate.get("abstain"),
        "counter_gate_reason": gate.get("reason"),
        # Chain lengths
        "reasoner_chain_steps": len(reasoner_chain),
        "counter_chain_steps": len(counter_chain),
        # Routing
        "domain": dig(response, "routing", "domain"),
        "causal_type_id": dig(response, "final_routing", "causal_type_id"),
        "theory_id": dig(response, "final_routing", "theory_id"),
        # Counter planning
        "counter_planning_mode": counter.get("planning_mode"),
        "counter_selected_attacks_n": (
            len(counter.get("selected_attack_ids") or [])
            if isinstance(counter.get("selected_attack_ids"), list)
            else None
        ),
        # Winning side
        "winning_side": evaluation.get("winning_side"),
        "confidence": _f(evaluation.get("confidence")),
        # ── Abstention Quality ──────────────────────────────────────
        # Classify *why* the counter abstained:
        #   "reasoned"       – counter explicitly decided no valid attacks exist
        #   "gate"           – polisher gate blocked (insufficient material / non-opposing)
        #   "empty"          – output empty / generation failure
        #   None             – counter did not abstain
        "abstention_type": (
            _classify_abstention(gate, counter) if gate.get("abstain") else None
        ),
        # ── Counter Density (structural richness) ─────────────────
        # ratio = contra_links / max(pro_links, 1)
        # Higher → counter built a denser attack graph relative to pro
        "counter_density": (
            round(
                (
                    len(aqa_links.get("contra") or [])
                    if isinstance(aqa_links.get("contra"), list)
                    else 0
                )
                / max(
                    (
                        len(aqa_links.get("pro") or [])
                        if isinstance(aqa_links.get("pro"), list)
                        else 0
                    ),
                    1,
                ),
                4,
            )
        ),
    }


def compute_doe_delta(metrics_a: dict, metrics_b: dict) -> dict[str, Any]:
    """Compute B-minus-A deltas for key numeric metrics."""
    delta_keys = [
        "aqa_net_final",
        "aqa_net_pro",
        "aqa_net_contra",
        "pro_links_count",
        "contra_links_count",
        "pro_cogency_avg",
        "pro_semantics_avg",
        "pro_norm_support_avg",
        "contra_cogency_avg",
        "contra_semantics_avg",
        "contra_norm_support_avg",
        "reasoner_repair_fail_rate",
        "counter_repair_fail_rate",
        "confidence",
        "counter_density",
    ]
    delta: dict[str, Any] = {}
    for k in delta_keys:
        a, b = metrics_a.get(k), metrics_b.get(k)
        if a is not None and b is not None:
            delta[f"delta_{k}"] = round(b - a, 6)
        else:
            delta[f"delta_{k}"] = None

    # Verdict flip
    delta["verdict_A"] = metrics_a.get("aqa_verdict")
    delta["verdict_B"] = metrics_b.get("aqa_verdict")
    delta["verdict_changed"] = metrics_a.get("aqa_verdict") != metrics_b.get(
        "aqa_verdict"
    )
    delta["plausible_A"] = int(
        str(metrics_a.get("aqa_verdict", "")).lower() == "plausible"
    )
    delta["plausible_B"] = int(
        str(metrics_b.get("aqa_verdict", "")).lower() == "plausible"
    )
    delta["abstain_A"] = bool(metrics_a.get("counter_gate_abstain"))
    delta["abstain_B"] = bool(metrics_b.get("counter_gate_abstain"))
    # Abstention quality
    delta["abstention_type_A"] = metrics_a.get("abstention_type")
    delta["abstention_type_B"] = metrics_b.get("abstention_type")
    return delta


# ──────────────────────────────────────────────────────────────────────
# Payload builder
# ──────────────────────────────────────────────────────────────────────
def build_setup_a_payload(claim_text: str, cfg: dict) -> dict:
    """Build full-pipeline payload for Setup A (baseline).

    Mirrors the frontend DoE handleDoeSubmit():
    - Pass flags come from config (= frontend doeSettings panel defaults)
    - counter_enable_causality is derived as OR of the three pass flags
    - aqa_lock_reasoner_plausibility is forced True
    """
    s = dict(cfg.get("settings", {}))
    pass_ci = bool(s.get("counter_pass_causal_identity", False))
    pass_ta = bool(s.get("counter_pass_taxonomy_attacks", False))
    pass_norms = bool(s.get("counter_pass_norms", False))
    s["counter_enable_causality"] = pass_ci or pass_ta or pass_norms
    s["counter_pass_causal_identity"] = pass_ci
    s["counter_pass_taxonomy_attacks"] = pass_ta
    s["counter_pass_norms"] = pass_norms
    s["aqa_lock_reasoner_plausibility"] = True
    s.setdefault("reasoner_enable_causality", True)
    return {
        "claim": claim_text,
        "include_precedents": bool(cfg.get("include_precedents", True)),
        "max_statutes": int(cfg.get("max_statutes", 100)),
        "max_precedents": int(cfg.get("max_precedents", 5)),
        "claim_context_memory_enabled": True,
        "claim_context_memory_overwrite": False,
        "settings": s,
    }


def build_setup_b_counter_payload(
    claim_text: str, cfg: dict, run_a_response: dict
) -> dict:
    """Build counter-only payload for Setup B (treatment).

    Mirrors the frontend DoE:
    - Reuses routing, retrieval context and reasoner_conclusion from Run A
    - All counter taxonomy flags forced True
    - Calls /api/counter_reason (NOT /api/pipeline)
    """
    s = dict(cfg.get("settings", {}))
    routing = run_a_response.get("final_routing") or run_a_response.get("routing") or {}
    retrieval = run_a_response.get("retrieval_context", {}) or {}
    reasoner_conclusion = (
        (run_a_response.get("reasoner") or {}).get("conclusion", "") or ""
    ).strip()
    statutes = retrieval.get("statutes") or []
    precedents = retrieval.get("precedents") or []

    payload: dict[str, Any] = {
        "claim": claim_text,
        "include_precedents": bool(cfg.get("include_precedents", True)),
        "max_statutes": int(cfg.get("max_statutes", 100)),
        "max_precedents": int(cfg.get("max_precedents", 5)),
        "claim_context_memory_enabled": True,
        "claim_context_memory_overwrite": False,
        "reasoner_conclusion": reasoner_conclusion,
        "routing": routing,
        "settings": {
            "counter_model": s.get("counter_model"),
            "counter_temperature": s.get("counter_temperature"),
            "llm_max_tokens": s.get("llm_max_tokens"),
            "counter_enable_causality": True,
            "counter_pass_causal_identity": True,
            "counter_pass_taxonomy_attacks": True,
            "counter_pass_norms": True,
        },
    }
    if isinstance(statutes, list) and isinstance(precedents, list) and statutes:
        payload["pre_retrieved_statutes"] = statutes
        payload["pre_retrieved_precedents"] = precedents
    return payload


def build_setup_b_evaluate_payload(
    claim_text: str, cfg: dict, run_a_response: dict, counter_b_result: dict
) -> dict:
    """Build evaluate-only payload for Setup B (treatment).

    Mirrors the frontend DoE:
    - Shared Reasoner output from A + fresh Counter output from B
    - AQA settings from config; lock forced True
    - Calls /api/evaluate (NOT /api/pipeline)
    """
    s = dict(cfg.get("settings", {}))
    domain = (
        dig(run_a_response, "final_routing", "domain")
        or dig(run_a_response, "routing", "domain")
        or "ENTRAMBI"
    )
    return {
        "claim": claim_text,
        "domain": domain,
        "reasoner_output": run_a_response.get("reasoner", {}),
        "counter_output": counter_b_result,
        "settings": {
            "aqa_alpha": s.get("aqa_alpha"),
            "aqa_beta": s.get("aqa_beta"),
            "aqa_gamma": s.get("aqa_gamma"),
            "aqa_min_semantic_overlap": s.get("aqa_min_semantic_overlap"),
            "aqa_min_strength_ratio": s.get("aqa_min_strength_ratio"),
            "aqa_damage_factor": s.get("aqa_damage_factor"),
            "aqa_allow_factual_attacks": s.get("aqa_allow_factual_attacks"),
            "aqa_allow_cross_codice": s.get("aqa_allow_cross_codice"),
            "aqa_lock_reasoner_plausibility": True,
        },
    }


def compose_setup_b_response(
    run_a_response: dict, counter_b_result: dict, evaluate_b_result: dict
) -> dict:
    """Compose a full-pipeline-like response for Setup B from partial results.

    Mirrors the frontend:
    const runBResult = { ...runAResult, counter_reasoner: B_counter, evaluation: B_eval }
    """
    composed = dict(run_a_response)
    composed["counter_reasoner"] = counter_b_result
    composed["evaluation"] = evaluate_b_result
    return composed


# ──────────────────────────────────────────────────────────────────────
# Run-level summary generator
# ──────────────────────────────────────────────────────────────────────
def build_run_summary(
    all_results: list[dict],
    run_dir: Path,
    manifest: dict,
) -> dict[str, Any]:
    """Build the final run-level summary JSON with aggregated metrics."""
    summary: dict[str, Any] = {
        "run_name": manifest.get("run_name"),
        "generated_at": datetime.now().isoformat(),
        "total_claims": len({r["claim_id"] for r in all_results}),
        "total_replicate_pairs": len(all_results),
    }

    # Per-claim detail rows
    detail_rows: list[dict[str, Any]] = []
    for r in all_results:
        row: dict[str, Any] = {
            "claim_id": r["claim_id"],
            "claim_title": r["claim_title"],
            "domain": r["domain"],
            "replicate": r["replicate"],
            "status_A": r.get("status_A", "missing"),
            "status_B": r.get("status_B", "missing"),
            "duration_A_sec": r.get("duration_A_sec"),
            "duration_B_sec": r.get("duration_B_sec"),
        }
        ma = r.get("metrics_A") or {}
        mb = r.get("metrics_B") or {}
        delta = r.get("delta") or {}

        # Setup A metrics
        for k, v in ma.items():
            row[f"A_{k}"] = v
        # Setup B metrics
        for k, v in mb.items():
            row[f"B_{k}"] = v
        # Deltas
        for k, v in delta.items():
            row[k] = v

        detail_rows.append(row)

    summary["claims"] = detail_rows

    # ── Aggregated statistics ──────────────────────────────────────────
    ok_pairs = [
        r
        for r in all_results
        if r.get("status_A") == "ok" and r.get("status_B") == "ok"
    ]
    n = len(ok_pairs)

    def _safe_avg(values: list[float | None]) -> float | None:
        clean = [v for v in values if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else None

    def _safe_std(values: list[float | None]) -> float | None:
        clean = [v for v in values if v is not None]
        if len(clean) < 2:
            return None
        mean = sum(clean) / len(clean)
        var = sum((x - mean) ** 2 for x in clean) / (len(clean) - 1)
        return round(var**0.5, 4)

    def _abstention_breakdown(types: list[str | None]) -> dict[str, int]:
        """Count abstention types: reasoned / gate / empty / none."""
        counts: dict[str, int] = {"reasoned": 0, "gate": 0, "empty": 0, "none": 0}
        for t in types:
            counts[t if t in ("reasoned", "gate", "empty") else "none"] += 1
        return counts

    def _compute_stability(
        results: list[dict], setup_key: str, metric_key: str = "aqa_net_final"
    ) -> dict[str, Any]:
        """Compute inter-replica stability: group by claim_id, compute
        per-claim stdev across replicates, then average those stdevs.

        Lower mean_std → more stable (deterministic) system.
        """
        from collections import defaultdict

        by_claim: dict[str, list[float]] = defaultdict(list)
        for r in results:
            v = _f((r.get(f"metrics_{setup_key}") or {}).get(metric_key))
            if v is not None:
                by_claim[r["claim_id"]].append(v)

        claim_stds: list[float] = []
        for cid, vals in by_claim.items():
            if len(vals) >= 2:
                mean = sum(vals) / len(vals)
                var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
                claim_stds.append(var**0.5)

        if not claim_stds:
            return {"claims_with_2plus_reps": 0, "mean_std": None, "max_std": None}

        return {
            "claims_with_2plus_reps": len(claim_stds),
            "mean_std": round(sum(claim_stds) / len(claim_stds), 6),
            "max_std": round(max(claim_stds), 6),
        }

    if n > 0:
        net_finals_a = [
            _f((r.get("metrics_A") or {}).get("aqa_net_final")) for r in ok_pairs
        ]
        net_finals_b = [
            _f((r.get("metrics_B") or {}).get("aqa_net_final")) for r in ok_pairs
        ]
        deltas_net = [
            _f((r.get("delta") or {}).get("delta_aqa_net_final")) for r in ok_pairs
        ]

        plausible_a = sum(
            1
            for r in ok_pairs
            if str((r.get("metrics_A") or {}).get("aqa_verdict", "")).lower()
            == "plausible"
        )
        plausible_b = sum(
            1
            for r in ok_pairs
            if str((r.get("metrics_B") or {}).get("aqa_verdict", "")).lower()
            == "plausible"
        )
        abstain_a = sum(
            1
            for r in ok_pairs
            if bool((r.get("metrics_A") or {}).get("counter_gate_abstain"))
        )
        abstain_b = sum(
            1
            for r in ok_pairs
            if bool((r.get("metrics_B") or {}).get("counter_gate_abstain"))
        )
        verdict_changed = sum(
            1 for r in ok_pairs if bool((r.get("delta") or {}).get("verdict_changed"))
        )

        summary["aggregated"] = {
            "completed_pairs": n,
            "failed_pairs": len(all_results) - n,
            # AQA net final
            "mean_aqa_net_final_A": _safe_avg(net_finals_a),
            "mean_aqa_net_final_B": _safe_avg(net_finals_b),
            "std_aqa_net_final_A": _safe_std(net_finals_a),
            "std_aqa_net_final_B": _safe_std(net_finals_b),
            # Delta B-A
            "mean_delta_aqa_net_final": _safe_avg(deltas_net),
            "std_delta_aqa_net_final": _safe_std(deltas_net),
            # Verdict rates
            "plausible_rate_A": round(plausible_a / n, 4),
            "plausible_rate_B": round(plausible_b / n, 4),
            "verdict_changed_rate": round(verdict_changed / n, 4),
            # Abstention rates
            "abstain_rate_A": round(abstain_a / n, 4),
            "abstain_rate_B": round(abstain_b / n, 4),
            # Per-metric averages
            "mean_pro_cogency_A": _safe_avg(
                [
                    _f((r.get("metrics_A") or {}).get("pro_cogency_avg"))
                    for r in ok_pairs
                ]
            ),
            "mean_pro_cogency_B": _safe_avg(
                [
                    _f((r.get("metrics_B") or {}).get("pro_cogency_avg"))
                    for r in ok_pairs
                ]
            ),
            "mean_contra_cogency_A": _safe_avg(
                [
                    _f((r.get("metrics_A") or {}).get("contra_cogency_avg"))
                    for r in ok_pairs
                ]
            ),
            "mean_contra_cogency_B": _safe_avg(
                [
                    _f((r.get("metrics_B") or {}).get("contra_cogency_avg"))
                    for r in ok_pairs
                ]
            ),
            "mean_pro_norm_support_A": _safe_avg(
                [
                    _f((r.get("metrics_A") or {}).get("pro_norm_support_avg"))
                    for r in ok_pairs
                ]
            ),
            "mean_pro_norm_support_B": _safe_avg(
                [
                    _f((r.get("metrics_B") or {}).get("pro_norm_support_avg"))
                    for r in ok_pairs
                ]
            ),
            "mean_pro_semantics_A": _safe_avg(
                [
                    _f((r.get("metrics_A") or {}).get("pro_semantics_avg"))
                    for r in ok_pairs
                ]
            ),
            "mean_pro_semantics_B": _safe_avg(
                [
                    _f((r.get("metrics_B") or {}).get("pro_semantics_avg"))
                    for r in ok_pairs
                ]
            ),
            # Link counts
            "mean_pro_links_count_A": _safe_avg(
                [
                    _f((r.get("metrics_A") or {}).get("pro_links_count"))
                    for r in ok_pairs
                ]
            ),
            "mean_pro_links_count_B": _safe_avg(
                [
                    _f((r.get("metrics_B") or {}).get("pro_links_count"))
                    for r in ok_pairs
                ]
            ),
            "mean_contra_links_count_A": _safe_avg(
                [
                    _f((r.get("metrics_A") or {}).get("contra_links_count"))
                    for r in ok_pairs
                ]
            ),
            "mean_contra_links_count_B": _safe_avg(
                [
                    _f((r.get("metrics_B") or {}).get("contra_links_count"))
                    for r in ok_pairs
                ]
            ),
            # ── Counter Density ────────────────────────────────────
            "mean_counter_density_A": _safe_avg(
                [
                    _f((r.get("metrics_A") or {}).get("counter_density"))
                    for r in ok_pairs
                ]
            ),
            "mean_counter_density_B": _safe_avg(
                [
                    _f((r.get("metrics_B") or {}).get("counter_density"))
                    for r in ok_pairs
                ]
            ),
            # ── Abstention Quality ─────────────────────────────────
            "abstention_breakdown_A": _abstention_breakdown(
                [(r.get("metrics_A") or {}).get("abstention_type") for r in ok_pairs]
            ),
            "abstention_breakdown_B": _abstention_breakdown(
                [(r.get("metrics_B") or {}).get("abstention_type") for r in ok_pairs]
            ),
        }

        # ── Stability inter-replica ───────────────────────────────────
        # Per-claim stdev of aqa_net_final across replicates.
        # Lower mean_std → higher determinism.
        summary["stability"] = {
            "setup_A": _compute_stability(ok_pairs, "A"),
            "setup_B": _compute_stability(ok_pairs, "B"),
        }

        # Per-domain breakdown
        domain_stats: dict[str, dict[str, Any]] = {}
        for r in ok_pairs:
            d = r.get("domain", "UNKNOWN")
            if d not in domain_stats:
                domain_stats[d] = {"n": 0, "net_A": [], "net_B": [], "deltas": []}
            domain_stats[d]["n"] += 1
            domain_stats[d]["net_A"].append(
                _f((r.get("metrics_A") or {}).get("aqa_net_final"))
            )
            domain_stats[d]["net_B"].append(
                _f((r.get("metrics_B") or {}).get("aqa_net_final"))
            )
            domain_stats[d]["deltas"].append(
                _f((r.get("delta") or {}).get("delta_aqa_net_final"))
            )

        summary["by_domain"] = {
            domain: {
                "pairs": stats["n"],
                "mean_aqa_net_final_A": _safe_avg(stats["net_A"]),
                "mean_aqa_net_final_B": _safe_avg(stats["net_B"]),
                "mean_delta": _safe_avg(stats["deltas"]),
                "std_delta": _safe_std(stats["deltas"]),
            }
            for domain, stats in sorted(domain_stats.items())
        }
    else:
        summary["aggregated"] = {"completed_pairs": 0, "failed_pairs": len(all_results)}
        summary["by_domain"] = {}

    return summary


def write_summary_csv(summary: dict, csv_path: Path) -> None:
    """Write the claims detail section of the summary as CSV for easy analysis."""
    claims = summary.get("claims", [])
    if not claims:
        return
    fieldnames = list(claims[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(claims)


# ──────────────────────────────────────────────────────────────────────
# Next run number
# ──────────────────────────────────────────────────────────────────────
def next_run_name(batch_dir: Path) -> str:
    """Auto-increment run_XXX based on existing folders."""
    existing = (
        sorted(
            int(m.group(1))
            for d in batch_dir.iterdir()
            if d.is_dir() and (m := re.match(r"^run_(\d+)$", d.name))
        )
        if batch_dir.exists()
        else []
    )
    n = (existing[-1] + 1) if existing else 1
    return f"run_{n:03d}"


def latest_run_name(batch_dir: Path) -> str | None:
    """Return the name of the most recent run_XXX folder, or None."""
    existing = (
        sorted(
            (int(m.group(1)), d.name)
            for d in batch_dir.iterdir()
            if d.is_dir() and (m := re.match(r"^run_(\d+)$", d.name))
        )
        if batch_dir.exists()
        else []
    )
    return existing[-1][1] if existing else None


# ──────────────────────────────────────────────────────────────────────
# Core batch runner
# ──────────────────────────────────────────────────────────────────────
def run_batch(
    claims: list[ClaimCase],
    cfg: dict,
    run_dir: Path,
    *,
    replicates: int,
    resume: bool,
    start_from: str | None,
    only_ids: set[str] | None,
    dry_run: bool,
) -> None:
    """Execute the full DoE batch: for each claim × replicate, run A then B."""
    endpoint = str(cfg.get("endpoint", "http://127.0.0.1:8000/api/pipeline")).strip()
    # Derive counter-only and evaluate-only endpoints from pipeline URL
    base_url = (
        endpoint.rsplit("/api/pipeline", 1)[0]
        if "/api/pipeline" in endpoint
        else endpoint.rstrip("/")
    )
    endpoint_counter = f"{base_url}/api/counter_reason"
    endpoint_evaluate = f"{base_url}/api/evaluate"
    timeout_sec = int(cfg.get("timeout_sec", 1800))
    max_retries = int(cfg.get("max_retries", 3))
    backoff_sec = int(cfg.get("retry_backoff_sec", 20))

    # ── Build execution plan ───────────────────────────────────────────
    plan: list[tuple[ClaimCase, int]] = []
    started = start_from is None
    for claim in claims:
        if only_ids and claim.claim_id not in only_ids:
            continue
        if not started:
            if claim.claim_id == start_from:
                started = True
            else:
                continue
        for rep in range(1, replicates + 1):
            plan.append((claim, rep))

    total = len(plan)
    print(f"\n{'='*70}")
    print("  LexCausa DoE Batch Runner")
    print(f"{'='*70}")
    print(f"  Run dir:     {run_dir}")
    print(f"  Pipeline:    {endpoint}")
    print(f"  Counter:     {endpoint_counter}")
    print(f"  Evaluate:    {endpoint_evaluate}")
    print("  Protocol:    A=full pipeline, B=counter+evaluate (shared Reasoner)")
    print(f"  Claims:      {len({c.claim_id for c, _ in plan})}")
    print(f"  Replicates:  {replicates}")
    print(f"  Total pairs: {total}")
    print(f"  Resume:      {resume}")
    print(f"{'='*70}\n")

    if dry_run:
        print("[DRY RUN] Piano di esecuzione:")
        for i, (claim, rep) in enumerate(plan, 1):
            print(
                f"  {i:3d}. {claim.claim_id} R{rep} [{claim.domain}] "
                f"- {claim.title[:60]}"
            )
        print(f"\nTotale: {total} coppie A/B da eseguire.")
        return

    # ── Manifest ───────────────────────────────────────────────────────
    manifest = {
        "run_name": run_dir.name,
        "started_at": datetime.now().isoformat(),
        "config_file": str(Path(cfg.get("_config_path", "ablation_settings.json"))),
        "config_hash_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()
        ).hexdigest(),
        "protocol": "batch_doe_ab_shared_reasoner",
        "replicates": replicates,
        "claims_selected": len({c.claim_id for c, _ in plan}),
        "rows_total": total * 2,  # A + B per pair
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Execute ────────────────────────────────────────────────────────
    all_results: list[dict[str, Any]] = []
    status_count = {"ok": 0, "warning": 0, "failed": 0}

    for idx, (claim, rep) in enumerate(plan, 1):
        rep_dir = run_dir / claim.claim_id / f"R{rep}"
        dir_a = rep_dir / "setup_A"
        dir_b = rep_dir / "setup_B"
        dir_doe = rep_dir / "doe"

        # Resume check
        if (
            resume
            and (dir_a / "pipeline_response.json").exists()
            and (dir_b / "pipeline_response.json").exists()
        ):
            print(f"[SKIP] {claim.claim_id} R{rep} (artifacts exist)")
            # Still load existing results for summary
            try:
                resp_a = json.loads(
                    (dir_a / "pipeline_response.json").read_text("utf-8")
                )
                resp_b = json.loads(
                    (dir_b / "pipeline_response.json").read_text("utf-8")
                )
                m_a = extract_setup_metrics(resp_a)
                m_b = extract_setup_metrics(resp_b)
                all_results.append(
                    {
                        "claim_id": claim.claim_id,
                        "claim_title": claim.title,
                        "domain": claim.domain,
                        "replicate": rep,
                        "status_A": "ok",
                        "status_B": "ok",
                        "metrics_A": m_a,
                        "metrics_B": m_b,
                        "delta": compute_doe_delta(m_a, m_b),
                    }
                )
                status_count["ok"] += 1
            except Exception:
                pass
            continue

        dir_a.mkdir(parents=True, exist_ok=True)
        dir_b.mkdir(parents=True, exist_ok=True)
        dir_doe.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*70}")
        print(
            f"[{idx}/{total}] {claim.claim_id} R{rep} [{claim.domain}] "
            f"- {claim.title[:50]}"
        )
        print(f"{'─'*70}")

        result_entry: dict[str, Any] = {
            "claim_id": claim.claim_id,
            "claim_title": claim.title,
            "domain": claim.domain,
            "replicate": rep,
        }

        # ── Setup A (baseline) ────────────────────────────────────────
        print("   ▶ Setup A (baseline: pass flags from config)...")
        payload_a = build_setup_a_payload(claim.text, cfg)
        (dir_a / "request.json").write_text(
            json.dumps(payload_a, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        t0 = time.time()
        status_a, resp_a = post_json_with_retries(
            endpoint,
            payload_a,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_sec=backoff_sec,
        )
        dur_a = round(time.time() - t0, 2)

        (dir_a / "pipeline_response.json").write_text(
            json.dumps(resp_a, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result_entry["status_A"] = status_a
        result_entry["duration_A_sec"] = dur_a

        # Save AQA report separately
        aqa_a = dig(resp_a, "evaluation", "aqa_report")
        if isinstance(aqa_a, dict):
            (dir_a / "aqa_report.json").write_text(
                json.dumps(aqa_a, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        # Save pipeline log
        _save_pipeline_log(
            dir_a / "pipeline.log", claim.text, payload_a, resp_a, status_a, dur_a, "A"
        )
        print(f"      [{status_a.upper()}] Setup A done in {dur_a}s")

        # ── Setup B (treatment: counter-only + evaluate-only) ─────────
        # Mirrors the frontend DoE: Reasoner is shared from A;
        # only the Counter-Reasoner and Evaluator are re-run with B settings.
        print("   ▶ Setup B (treatment: counter_causality=ON, shared Reasoner)...")

        if status_a != "ok":
            print("      [SKIP] Setup A failed; cannot run B without Reasoner output")
            result_entry["status_B"] = "skipped"
            result_entry["duration_B_sec"] = 0
            status_count["failed"] += 1
            all_results.append(result_entry)
            continue

        # Check reasoner conclusion is available
        reasoner_conclusion_a = (
            (resp_a.get("reasoner") or {}).get("conclusion", "") or ""
        ).strip()
        if not reasoner_conclusion_a:
            print("      [SKIP] Setup A Reasoner has no conclusion; cannot run B")
            result_entry["status_B"] = "skipped"
            result_entry["duration_B_sec"] = 0
            status_count["failed"] += 1
            all_results.append(result_entry)
            continue

        t0 = time.time()

        # Step B.1: Counter-Reasoner only (/api/counter_reason)
        payload_b_counter = build_setup_b_counter_payload(claim.text, cfg, resp_a)
        (dir_b / "counter_request.json").write_text(
            json.dumps(payload_b_counter, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("      B.1 Counter-Reasoner (shared context from A)...")
        status_b_counter, resp_b_counter = post_json_with_retries(
            endpoint_counter,
            payload_b_counter,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_sec=backoff_sec,
        )
        (dir_b / "counter_response.json").write_text(
            json.dumps(resp_b_counter, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if status_b_counter != "ok":
            dur_b = round(time.time() - t0, 2)
            print(f"      [FAILED] Setup B counter failed in {dur_b}s")
            result_entry["status_B"] = "failed"
            result_entry["duration_B_sec"] = dur_b
            status_count["failed"] += 1
            _save_pipeline_log(
                dir_b / "pipeline.log",
                claim.text,
                payload_b_counter,
                resp_b_counter,
                status_b_counter,
                dur_b,
                "B",
            )
            all_results.append(result_entry)
            continue

        # Step B.2: Evaluate only (/api/evaluate)
        payload_b_eval = build_setup_b_evaluate_payload(
            claim.text, cfg, resp_a, resp_b_counter
        )
        (dir_b / "evaluate_request.json").write_text(
            json.dumps(payload_b_eval, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("      B.2 Evaluator (Reasoner A + Counter B)...")
        status_b_eval, resp_b_eval = post_json_with_retries(
            endpoint_evaluate,
            payload_b_eval,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_sec=backoff_sec,
        )
        (dir_b / "evaluate_response.json").write_text(
            json.dumps(resp_b_eval, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        dur_b = round(time.time() - t0, 2)

        if status_b_eval != "ok":
            print(f"      [FAILED] Setup B evaluate failed in {dur_b}s")
            result_entry["status_B"] = "failed"
            result_entry["duration_B_sec"] = dur_b
            status_count["failed"] += 1
            _save_pipeline_log(
                dir_b / "pipeline.log",
                claim.text,
                payload_b_eval,
                resp_b_eval,
                status_b_eval,
                dur_b,
                "B",
            )
            all_results.append(result_entry)
            continue

        # Step B.3: Compose full response (like frontend: { ...A, counter_reasoner: B, evaluation: B })
        resp_b = compose_setup_b_response(resp_a, resp_b_counter, resp_b_eval)
        status_b = "ok"

        (dir_b / "pipeline_response.json").write_text(
            json.dumps(resp_b, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result_entry["status_B"] = status_b
        result_entry["duration_B_sec"] = dur_b

        aqa_b = dig(resp_b, "evaluation", "aqa_report")
        if isinstance(aqa_b, dict):
            (dir_b / "aqa_report.json").write_text(
                json.dumps(aqa_b, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        _save_pipeline_log(
            dir_b / "pipeline.log",
            claim.text,
            payload_b_counter,
            resp_b,
            status_b,
            dur_b,
            "B",
        )
        print(f"      [OK] Setup B done in {dur_b}s")

        # ── Extract metrics & delta ───────────────────────────────────
        metrics_a = extract_setup_metrics(resp_a) if status_a == "ok" else {}
        metrics_b = extract_setup_metrics(resp_b) if status_b == "ok" else {}
        result_entry["metrics_A"] = metrics_a
        result_entry["metrics_B"] = metrics_b

        if status_a == "ok" and status_b == "ok":
            delta = compute_doe_delta(metrics_a, metrics_b)
            result_entry["delta"] = delta
            status_count["ok"] += 1

            # ── DoE comparison report ─────────────────────────────────
            doe_report = {
                "claim_id": claim.claim_id,
                "claim_title": claim.title,
                "domain": claim.domain,
                "replicate": rep,
                "generated_at": datetime.now().isoformat(),
                "setup_A": {
                    "label": "Baseline (counter_causality=OFF)",
                    "status": status_a,
                    "duration_sec": dur_a,
                    "metrics": metrics_a,
                },
                "setup_B": {
                    "label": "Treatment (counter_causality=ON)",
                    "status": status_b,
                    "duration_sec": dur_b,
                    "metrics": metrics_b,
                },
                "delta": delta,
            }
            (dir_doe / "doe_report.json").write_text(
                json.dumps(doe_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _save_doe_log(dir_doe / "doe.log", claim, rep, metrics_a, metrics_b, delta)

            # Print quick delta
            d_net = delta.get("delta_aqa_net_final")
            v_a = delta.get("verdict_A", "?")
            v_b = delta.get("verdict_B", "?")
            print(f"   📊 DoE delta: net_final={d_net}, " f"verdict {v_a} -> {v_b}")
        else:
            result_entry["delta"] = {}
            status_count["failed"] += 1
            if status_a == "ok" or status_b == "ok":
                status_count["warning"] += 1

        all_results.append(result_entry)

    # ── Final summary ──────────────────────────────────────────────────
    manifest["finished_at"] = datetime.now().isoformat()
    manifest["status_count"] = status_count
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = build_run_summary(all_results, run_dir, manifest)
    summary_json = run_dir / "run_summary.json"
    summary_csv = run_dir / "run_summary.csv"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_csv(summary, summary_csv)

    print(f"\n{'='*70}")
    print("  BATCH COMPLETED")
    print(f"{'='*70}")
    print(f"  Manifest:  {manifest_path}")
    print(f"  Summary:   {summary_json}")
    print(f"  CSV:       {summary_csv}")
    print(
        f"  OK: {status_count['ok']}  WARN: {status_count['warning']}  FAIL: {status_count['failed']}"
    )
    agg = summary.get("aggregated", {})
    if agg.get("completed_pairs", 0) > 0:
        print("\n  === Risultati aggregati ===")
        print(f"  Media AQA net final A: {agg.get('mean_aqa_net_final_A')}")
        print(f"  Media AQA net final B: {agg.get('mean_aqa_net_final_B')}")
        print(f"  Media delta (B-A):     {agg.get('mean_delta_aqa_net_final')}")
        print(f"  Plausible rate A:      {agg.get('plausible_rate_A')}")
        print(f"  Plausible rate B:      {agg.get('plausible_rate_B')}")
        print(f"  Verdict changed rate:  {agg.get('verdict_changed_rate')}")
        print(f"  Abstain rate A:        {agg.get('abstain_rate_A')}")
        print(f"  Abstain rate B:        {agg.get('abstain_rate_B')}")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────────────────
# Log helpers
# ──────────────────────────────────────────────────────────────────────
def _save_pipeline_log(
    path: Path,
    claim: str,
    payload: dict,
    response: dict,
    status: str,
    duration: float,
    setup_label: str,
) -> None:
    """Save a human-readable pipeline execution log."""
    evaluation = dig(response, "evaluation", default={}) or {}
    aqa = dig(evaluation, "aqa_report", default={}) or {}
    net = aqa.get("net_plausibility", {}) or {}
    gate = dig(evaluation, "counter_reasoner_gate", default={}) or {}

    reasoner = dig(response, "reasoner", default={}) or {}
    counter = dig(response, "counter_reasoner", default={}) or {}

    lines = [
        f"[{datetime.now().isoformat()}] Pipeline log - Setup {setup_label}",
        f"Claim: {claim[:200]}",
        "=" * 70,
        f"Status: {status}",
        f"Duration: {duration}s",
        "",
        "[SETTINGS]",
        json.dumps(payload.get("settings", {}), ensure_ascii=False, indent=2),
        "",
        "[REASONER]",
        f"  Chain steps: {len(reasoner.get('reasoning_chain', []))}",
        f"  Arguments: {len(reasoner.get('arguments', []))}",
        f"  Causal type: {reasoner.get('causal_type_id', 'N/A')}",
        "",
        "[COUNTER]",
        f"  Chain steps: {len(counter.get('reasoning_chain', []))}",
        f"  Arguments: {len(counter.get('counter_arguments', []))}",
        f"  Abstained: {counter.get('abstained', False)}",
        f"  Planning mode: {counter.get('planning_mode', 'N/A')}",
        "",
        "[EVALUATION]",
        f"  Verdict: {aqa.get('verdict', 'N/A')}",
        f"  Net pro: {net.get('pro', 'N/A')}",
        f"  Net contra: {net.get('contra', 'N/A')}",
        f"  Net final: {net.get('final', 'N/A')}",
        f"  Counter gate: label={gate.get('label', 'N/A')}, abstain={gate.get('abstain', 'N/A')}",
    ]
    try:
        path.write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        print(f"      [WARN] Could not write log: {e}")


def _save_doe_log(
    path: Path,
    claim: ClaimCase,
    replicate: int,
    metrics_a: dict,
    metrics_b: dict,
    delta: dict,
) -> None:
    """Save a human-readable DoE comparison log."""
    lines = [
        f"[{datetime.now().isoformat()}] DoE Comparison Log",
        f"Claim: {claim.claim_id} - {claim.title}",
        f"Domain: {claim.domain}",
        f"Replicate: R{replicate}",
        "=" * 70,
        "",
        "[SETUP A - Baseline (counter_causality=OFF)]",
        f"  Verdict:        {metrics_a.get('aqa_verdict', 'N/A')}",
        f"  Net final:      {metrics_a.get('aqa_net_final', 'N/A')}",
        f"  Net pro:        {metrics_a.get('aqa_net_pro', 'N/A')}",
        f"  Net contra:     {metrics_a.get('aqa_net_contra', 'N/A')}",
        f"  Pro links:      {metrics_a.get('pro_links_count', 'N/A')}",
        f"  Contra links:   {metrics_a.get('contra_links_count', 'N/A')}",
        f"  Pro cogency:    {metrics_a.get('pro_cogency_avg', 'N/A')}",
        f"  Pro semantics:  {metrics_a.get('pro_semantics_avg', 'N/A')}",
        f"  Pro norm supp:  {metrics_a.get('pro_norm_support_avg', 'N/A')}",
        f"  Counter gate:   {metrics_a.get('counter_gate_label') or 'N/A'} (abstain={metrics_a.get('counter_gate_abstain', 'N/A')})",
        f"  Abstention type:{str(metrics_a.get('abstention_type') or 'N/A'):>10}",
        f"  Counter density:{str(metrics_a['counter_density'] if metrics_a.get('counter_density') is not None else 'N/A'):>10}",
        "",
        "[SETUP B - Treatment (counter_causality=ON)]",
        f"  Verdict:        {metrics_b.get('aqa_verdict', 'N/A')}",
        f"  Net final:      {metrics_b.get('aqa_net_final', 'N/A')}",
        f"  Net pro:        {metrics_b.get('aqa_net_pro', 'N/A')}",
        f"  Net contra:     {metrics_b.get('aqa_net_contra', 'N/A')}",
        f"  Pro links:      {metrics_b.get('pro_links_count', 'N/A')}",
        f"  Contra links:   {metrics_b.get('contra_links_count', 'N/A')}",
        f"  Pro cogency:    {metrics_b.get('pro_cogency_avg', 'N/A')}",
        f"  Pro semantics:  {metrics_b.get('pro_semantics_avg', 'N/A')}",
        f"  Pro norm supp:  {metrics_b.get('pro_norm_support_avg', 'N/A')}",
        f"  Counter gate:   {metrics_b.get('counter_gate_label') or 'N/A'} (abstain={metrics_b.get('counter_gate_abstain', 'N/A')})",
        f"  Abstention type:{str(metrics_b.get('abstention_type') or 'N/A'):>10}",
        f"  Counter density:{str(metrics_b['counter_density'] if metrics_b.get('counter_density') is not None else 'N/A'):>10}",
        "",
        "[DELTA (B - A)]",
        f"  Delta net final:              {delta.get('delta_aqa_net_final', 'N/A')}",
        f"  Delta net pro:                {delta.get('delta_aqa_net_pro', 'N/A')}",
        f"  Delta net contra:             {delta.get('delta_aqa_net_contra', 'N/A')}",
        f"  Delta pro links:              {delta.get('delta_pro_links_count', 'N/A')}",
        f"  Delta contra links:           {delta.get('delta_contra_links_count', 'N/A')}",
        f"  Delta pro cogency:            {delta.get('delta_pro_cogency_avg', 'N/A')}",
        f"  Delta pro semantics:          {delta.get('delta_pro_semantics_avg', 'N/A')}",
        f"  Delta pro norm support:       {delta.get('delta_pro_norm_support_avg', 'N/A')}",
        f"  Delta contra cogency:         {delta.get('delta_contra_cogency_avg', 'N/A')}",
        f"  Delta contra semantics:       {delta.get('delta_contra_semantics_avg', 'N/A')}",
        f"  Delta contra norm support:    {delta.get('delta_contra_norm_support_avg', 'N/A')}",
        f"  Delta confidence:             {delta.get('delta_confidence', 'N/A')}",
        f"  Verdict A:                    {delta.get('verdict_A', 'N/A')}",
        f"  Verdict B:                    {delta.get('verdict_B', 'N/A')}",
        f"  Verdict changed:              {delta.get('verdict_changed', 'N/A')}",
        f"  Delta counter density:        {delta.get('delta_counter_density', 'N/A')}",
        f"  Abstention type A:            {delta.get('abstention_type_A', 'N/A')}",
        f"  Abstention type B:            {delta.get('abstention_type_B', 'N/A')}",
    ]
    try:
        path.write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        print(f"      [WARN] Could not write DoE log: {e}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run DoE batch on all covered claims from claims.md"
    )
    p.add_argument(
        "--claims-file", type=Path, default=DEFAULT_CLAIMS, help="Path to claims.md"
    )
    p.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="DoE settings JSON"
    )
    p.add_argument(
        "--replicates", type=int, default=2, help="Number of replicates per claim"
    )
    p.add_argument(
        "--domains",
        type=str,
        default="CIVILE,PENALE,AMMINISTRATIVO,MIXED",
        help="Comma-separated domains to include",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Explicit run folder name (e.g. run_005)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip claim/replicates with existing artifacts",
    )
    p.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Start from this claim ID (e.g. C3)",
    )
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated claim IDs to run (e.g. C1,P2,A4)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Show execution plan without running"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Parse claims
    all_claims = parse_claims_md(args.claims_file)
    allowed_domains = {d.strip().upper() for d in args.domains.split(",") if d.strip()}
    eligible = [c for c in all_claims if c.covered and c.domain in allowed_domains]

    if not eligible:
        print("ERROR: No eligible covered claims found.")
        sys.exit(1)

    print(
        f"[INFO] Parsed {len(all_claims)} claims, {len(eligible)} eligible covered claims"
    )
    for c in eligible:
        print(f"  {c.claim_id:4s} [{c.domain:15s}] {c.title[:60]}")

    # Load config
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(args.config)

    # Determine run folder
    if args.run_name.strip():
        run_name = args.run_name.strip()
    elif args.resume:
        # --resume without --run-name: reuse the latest existing run
        run_name = latest_run_name(BATCH_RUNS_DIR)
        if not run_name:
            print(
                "ERROR: --resume specified but no existing run found in "
                "experiments/causal_taxonomy_ablation/"
            )
            sys.exit(1)
        print(f"[INFO] --resume: riprendo dalla run esistente '{run_name}'")
    else:
        run_name = next_run_name(BATCH_RUNS_DIR)
    run_dir = BATCH_RUNS_DIR / run_name

    # Only filter
    only_ids = (
        {x.strip().upper() for x in args.only.split(",") if x.strip()}
        if args.only.strip()
        else None
    )

    try:
        run_batch(
            claims=eligible,
            cfg=cfg,
            run_dir=run_dir,
            replicates=args.replicates,
            resume=args.resume,
            start_from=args.start_from,
            only_ids=only_ids,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Batch interrotto. Usa --resume per riprendere.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
