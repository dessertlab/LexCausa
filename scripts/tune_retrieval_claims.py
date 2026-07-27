#!/usr/bin/env python3
"""
Domain-specific retrieval tuning from claims.md.

Modes:
- Supervised (preferred): optimize on gold labels with robust multi-run scoring
  built on Hit@k, MRR, nDCG@k.
- Unsupervised fallback: proxy lexical metrics when gold labels are missing.

Notes:
- Supports structured gold labels (`gold`, `must_have`, `good_to_have`, `confusers`)
  while remaining backward-compatible with flat label lists.
- Can enforce optional per-claim supervised constraints (e.g. `P2.min_hit_at_5`)
  during tuning.
- Per-claim constraints are evaluated on the benchmark-aligned retrieval view
  (hybrid ranked list + citation expansion) to avoid tuning/benchmark mismatch.

Inputs:
- claims.md (covered claims sections)
- claims_gold_labels.json (claim_id -> expected articles)

Outputs:
- logs/tuning/retrieval_tuning_YYYYMMDD_HHMMSS.json
- logs/tuning/retrieval_tuning_latest.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agents.base import AgentConfig  # noqa: E402
from agents.retrieval_filter_agent import RetrievalFilterAgent  # noqa: E402
from config import settings  # noqa: E402
from services.legal_search import LegalSearchPipeline  # noqa: E402

CLAIMS_MD_PATH = PROJECT_ROOT / "claims.md"
GOLD_LABELS_PATH = PROJECT_ROOT / "claims_gold_labels.json"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "tuning"
DEFAULT_TOP_K = 30


@dataclass
class ClaimEntry:
    claim_id: str
    section: str
    domain: str
    title: str
    text: str


@dataclass
class EvalEntry:
    claim_id: str
    domain: str
    source: str
    proxy_score: float
    mean_overlap_top10: float
    coverage_terms_top10: float
    top_sources: dict[str, int]
    top_articles: list[str]
    gold_count: int = 0
    hit_at_5: Optional[float] = None
    hit_at_10: Optional[float] = None
    mrr: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    runs: int = 1
    proxy_score_std: Optional[float] = None
    mean_overlap_top10_std: Optional[float] = None
    coverage_terms_top10_std: Optional[float] = None
    hit_at_5_std: Optional[float] = None
    hit_at_10_std: Optional[float] = None
    mrr_std: Optional[float] = None
    ndcg_at_10_std: Optional[float] = None


@dataclass
class StageEvalEntry:
    claim_id: str
    domain: str
    source: str
    gold_count: int
    must_have_count: int
    confusers_count: int
    runs: int
    raw_topk_count_mean: float
    cites_topk_count_mean: float
    relevance_kept_count_mean: float
    applicability_kept_count_mean: float
    raw_gold_recall_at_k_mean: Optional[float] = None
    cites_gold_recall_at_k_mean: Optional[float] = None
    post_rel_gold_recall_mean: Optional[float] = None
    post_app_gold_recall_mean: Optional[float] = None
    raw_must_have_recall_at_k_mean: Optional[float] = None
    cites_must_have_recall_at_k_mean: Optional[float] = None
    post_rel_must_have_recall_mean: Optional[float] = None
    post_app_must_have_recall_mean: Optional[float] = None
    raw_must_have_all_present_rate: Optional[float] = None
    cites_must_have_all_present_rate: Optional[float] = None
    post_rel_must_have_all_present_rate: Optional[float] = None
    post_app_must_have_all_present_rate: Optional[float] = None
    rel_filter_keep_ratio_mean: Optional[float] = None
    app_filter_keep_ratio_over_rel_mean: Optional[float] = None
    must_have_dropped_by_relevance_mean: Optional[float] = None
    must_have_dropped_by_applicability_mean: Optional[float] = None
    confuser_penalty_top10_raw_mean: Optional[float] = None
    confuser_penalty_top10_cites_mean: Optional[float] = None
    confuser_penalty_top10_post_app_mean: Optional[float] = None
    raw_top_articles: list[str] = field(default_factory=list)
    cites_top_articles: list[str] = field(default_factory=list)
    post_app_top_articles: list[str] = field(default_factory=list)


@dataclass
class GoldLabelSpec:
    """Structured gold labels for robust tuning (retro-compatible with flat lists)."""

    gold: set[str]
    must_have: set[str] = field(default_factory=set)
    good_to_have: set[str] = field(default_factory=set)
    confusers: set[str] = field(default_factory=set)


ROBUST_TUNING_POLICY: dict[str, float] = {
    # Blend average claim score with lower-quantile claim score to avoid overfitting
    # to domains where a few claims perform well and others collapse.
    "worst_claim_weight": 0.35,
    "worst_claim_quantile": 0.25,
    # Hard constraint on explicit must-have stability (applies only when enough
    # claims in the domain are annotated with must-have labels).
    "must_have_min_rate": 0.80,
    # Avoid overfitting the whole domain to too few annotated claims.
    "must_have_min_annotated_claims_for_hard_gate": 2.0,
    # Soft bonuses/penalties (supervised mode only; only if annotations exist).
    "must_have_bonus": 0.12,
    "good_to_have_bonus": 0.04,
    "confuser_penalty": 0.10,
}


def _iter_progress(iterable, enabled: bool, **kwargs):
    """Return tqdm wrapper when enabled, otherwise plain iterable."""
    if not enabled:
        return iterable
    return tqdm(iterable, **kwargs)


def _extract_block(text: str, start_marker: str, end_markers: list[str]) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    end = len(text)
    for marker in end_markers:
        idx = text.find(marker, start)
        if idx >= 0:
            end = min(end, idx)
    return text[start:end].strip()


def _parse_claims_from_block(
    block: str,
    pattern: re.Pattern,
    section: str,
    domain: str,
) -> list[ClaimEntry]:
    claims: list[ClaimEntry] = []
    matches = list(pattern.finditer(block))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        body = block[start:end].strip()
        if not body:
            continue
        claim_id = str(match.group(1)).strip().upper()
        title = str(match.group(2)).strip()
        claims.append(
            ClaimEntry(
                claim_id=claim_id,
                section=section,
                domain=domain,
                title=title,
                text=body,
            )
        )
    return claims


def parse_claims_md(path: Path) -> tuple[list[ClaimEntry], int]:
    text = path.read_text(encoding="utf-8")

    civile_block = _extract_block(
        text,
        "## CLAIM CIVILI (COPERTI)",
        [
            "## CLAIM PENALI (COPERTI)",
            "## CLAIM MIXED (COPERTI)",
            "## CLAIM AMMINISTRATIVI (COPERTI)",
            "## CLAIM NON COPERTI",
        ],
    )
    penale_block = _extract_block(
        text,
        "## CLAIM PENALI (COPERTI)",
        [
            "## CLAIM MIXED (COPERTI)",
            "## CLAIM AMMINISTRATIVI (COPERTI)",
            "## CLAIM NON COPERTI",
        ],
    )
    mixed_block = _extract_block(
        text,
        "## CLAIM MIXED (COPERTI)",
        ["## CLAIM AMMINISTRATIVI (COPERTI)", "## CLAIM NON COPERTI"],
    )
    admin_block = _extract_block(
        text,
        "## CLAIM AMMINISTRATIVI (COPERTI)",
        ["## CLAIM NON COPERTI"],
    )
    non_covered_block = _extract_block(text, "## CLAIM NON COPERTI", [])

    claim_pattern = re.compile(r"(?m)^\s*###\s*([A-Z]\d+)\s*-\s*(.+?)\s*$")

    civile_claims = _parse_claims_from_block(
        civile_block,
        claim_pattern,
        section="CLAIM CIVILI (COPERTI)",
        domain="civile",
    )
    penale_claims = _parse_claims_from_block(
        penale_block,
        claim_pattern,
        section="CLAIM PENALI (COPERTI)",
        domain="penale",
    )
    mixed_claims = _parse_claims_from_block(
        mixed_block,
        claim_pattern,
        section="CLAIM MIXED (COPERTI)",
        domain="misto",
    )
    admin_claims = _parse_claims_from_block(
        admin_block,
        claim_pattern,
        section="CLAIM AMMINISTRATIVI (COPERTI)",
        domain="amministrativo",
    )

    all_claims = civile_claims + penale_claims + mixed_claims + admin_claims
    non_covered_count = len(
        re.findall(r"(?m)^\s*###\s*NC\d+\s*-\s*.+$", non_covered_block)
    )
    return all_claims, non_covered_count


def _normalize_source(value: str) -> str:
    raw = (value or "").strip().lower().replace(" ", "")
    mapping = {
        "codice_penale": "codice_penale",
        "cp": "codice_penale",
        "c.p.": "codice_penale",
        "codice_civile": "codice_civile",
        "cc": "codice_civile",
        "c.c.": "codice_civile",
        "codice_amministrativo": "codice_amministrativo",
        "amm": "codice_amministrativo",
        "l241": "codice_amministrativo",
        "l.241/1990": "codice_amministrativo",
    }
    return mapping.get(raw, raw)


def _normalize_articolo(value: str) -> str:
    art = (value or "").strip().lower()
    art = art.replace("–", "-").replace("—", "-")
    art = re.sub(r"^art\.?\s*", "", art)
    art = re.sub(r"\s+", "", art)
    art = art.strip(". ,;:")
    return art


def _normalize_articolo_for_source(source: str, value: str) -> str:
    art = _normalize_articolo(value)
    src = _normalize_source(source)
    # codice_civile_normattiva uses article_id style: art1, art2, ...
    if src == "codice_civile" and art and not art.startswith("art"):
        art = f"art{art}"
    return art


def _article_key(source: str, articolo: str) -> str:
    src = _normalize_source(source)
    art = _normalize_articolo_for_source(src, articolo)
    return f"{src}:art.{art}"


def _normalize_label_entries(entries: Any) -> set[str]:
    normalized: set[str] = set()
    if not isinstance(entries, list):
        return normalized

    for item in entries:
        source = ""
        articolo = ""
        if isinstance(item, dict):
            source = str(item.get("source", "")).strip()
            articolo = str(item.get("articolo", "")).strip()
        elif isinstance(item, str):
            text = item.strip()
            if ":" in text:
                left, right = text.split(":", 1)
                source = left.strip()
                articolo = right.replace("art.", "").strip()
            else:
                articolo = text
        if not source or not articolo:
            continue
        normalized.add(_article_key(source, articolo))
    return normalized


def load_gold_label_specs(path: Path) -> dict[str, GoldLabelSpec]:
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    labels_raw = payload.get("labels", payload)
    if not isinstance(labels_raw, dict):
        return {}

    labels: dict[str, GoldLabelSpec] = {}
    for claim_id, entries in labels_raw.items():
        cid = str(claim_id).strip().upper()
        gold: set[str] = set()
        must_have: set[str] = set()
        good_to_have: set[str] = set()
        confusers: set[str] = set()

        if isinstance(entries, list):
            gold = _normalize_label_entries(entries)
        elif isinstance(entries, dict):
            # Structured schema (backward-compatible): if "gold" is absent we also
            # accept "labels" or a legacy "entries" list.
            gold = _normalize_label_entries(
                entries.get("gold", entries.get("labels", entries.get("entries", [])))
            )
            must_have = _normalize_label_entries(entries.get("must_have", []))
            good_to_have = _normalize_label_entries(entries.get("good_to_have", []))
            confusers = _normalize_label_entries(entries.get("confusers", []))
        else:
            continue

        if not gold and not must_have and not good_to_have and not confusers:
            continue

        labels[cid] = GoldLabelSpec(
            gold=gold,
            must_have=must_have,
            good_to_have=good_to_have,
            confusers=confusers,
        )

    return labels


def _mean_or_none(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.fmean(values))


def _std_or_zero(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] + frac * (vals[hi] - vals[lo])


def _must_have_all_present_rate(
    ranked_keys: list[str], must_have: set[str], k: int
) -> float:
    if not must_have:
        return 1.0
    top = set(ranked_keys[: max(0, k)])
    return 1.0 if must_have.issubset(top) else 0.0


def _set_recall_at_k(ranked_keys: list[str], target_keys: set[str], k: int) -> float:
    if not target_keys:
        return 0.0
    top = set(ranked_keys[: max(0, k)])
    return len(top & target_keys) / max(1, len(target_keys))


def _confuser_rank_penalty(
    ranked_keys: list[str], confusers: set[str], k: int = 10
) -> float:
    """Rank-weighted confuser penalty in [0,1] over the top-k list."""
    if not confusers or k <= 0:
        return 0.0
    penalty = 0.0
    norm = 0.0
    for idx in range(1, k + 1):
        weight = 1.0 / idx
        norm += weight
        if idx - 1 < len(ranked_keys) and ranked_keys[idx - 1] in confusers:
            penalty += weight
    return penalty / max(1e-9, norm)


def _clear_query_terms_cache(pipeline: LegalSearchPipeline) -> None:
    cache = getattr(pipeline, "_query_terms_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def _score_claim_proxy(
    pipeline: LegalSearchPipeline,
    claim_text: str,
    results,
) -> tuple[float, float, float]:
    terms = pipeline.get_search_query_terms(claim_text, mode="llm")
    top = results[:10]
    if not top:
        return 0.0, 0.0, 0.0
    if not terms:
        top1 = max(0.0, float(top[0].score))
        top5 = float(top[min(len(top) - 1, 4)].score)
        margin = max(0.0, top1 - top5)
        return max(0.0, min(1.0, margin)), 0.0, 0.0

    covered_terms: set[str] = set()
    weighted_overlap = 0.0
    weight_sum = 0.0
    overlap_values: list[float] = []

    for rank, item in enumerate(top, start=1):
        text = f"{item.articolo} {item.titolo} {item.testo[:600]}".lower()
        item_terms = set(re.findall(r"[a-zA-Zàèéìòù0-9\\-]+", text))
        overlap = len(terms & item_terms) / max(1, len(terms))
        overlap_values.append(overlap)
        covered_terms.update(terms & item_terms)
        weight = 1.0 / rank
        weighted_overlap += overlap * weight
        weight_sum += weight

    top1 = max(0.0, float(top[0].score))
    top5 = float(top[min(len(top) - 1, 4)].score)
    margin = max(0.0, top1 - top5)
    weighted_overlap = weighted_overlap / max(1e-9, weight_sum)
    coverage_ratio = len(covered_terms) / max(1, len(terms))
    mean_overlap = statistics.fmean(overlap_values) if overlap_values else 0.0
    proxy_score = (0.55 * weighted_overlap) + (0.35 * coverage_ratio) + (0.10 * margin)
    proxy_score = max(0.0, min(1.0, proxy_score))
    return proxy_score, mean_overlap, coverage_ratio


def _ranked_article_keys(results) -> list[str]:
    return [_article_key(item.source, item.articolo) for item in results]


def _ranked_statute_dict_keys(items: list[dict[str, Any]]) -> list[str]:
    return [
        _article_key(str(item.get("source", "")), str(item.get("articolo", "")))
        for item in items
    ]


def _results_to_statute_dicts(results) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in results or []:
        out.append(
            {
                "source": getattr(item, "source", None),
                "articolo": getattr(item, "articolo", None),
                "titolo": getattr(item, "titolo", None),
                "testo": getattr(item, "testo", None),
                "libro": getattr(item, "libro", None),
                "score": float(getattr(item, "score", 0.0)),
            }
        )
    return out


def _make_retrieval_filter_agent() -> RetrievalFilterAgent:
    return RetrievalFilterAgent(
        config=AgentConfig(
            model_name=settings.retrieval_default_model,
            temperature=0.0,
            max_tokens=settings.llm_max_tokens,
        )
    )


def _hit_at_k(ranked_keys: list[str], gold_keys: set[str], k: int) -> float:
    if not ranked_keys or not gold_keys or k <= 0:
        return 0.0
    return 1.0 if any(key in gold_keys for key in ranked_keys[:k]) else 0.0


def _mrr(ranked_keys: list[str], gold_keys: set[str]) -> float:
    if not ranked_keys or not gold_keys:
        return 0.0
    for idx, key in enumerate(ranked_keys, start=1):
        if key in gold_keys:
            return 1.0 / idx
    return 0.0


def _ndcg_at_k(ranked_keys: list[str], gold_keys: set[str], k: int) -> float:
    if not ranked_keys or not gold_keys or k <= 0:
        return 0.0
    dcg = 0.0
    for idx, key in enumerate(ranked_keys[:k], start=1):
        if key in gold_keys:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(gold_keys), k)
    if ideal_hits <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _supervised_metrics(
    ranked_keys: list[str],
    gold_keys: set[str],
) -> dict[str, float]:
    hit5 = _hit_at_k(ranked_keys, gold_keys, 5)
    hit10 = _hit_at_k(ranked_keys, gold_keys, 10)
    mrr = _mrr(ranked_keys, gold_keys)
    ndcg10 = _ndcg_at_k(ranked_keys, gold_keys, 10)
    return {
        "hit_at_5": hit5,
        "hit_at_10": hit10,
        "mrr": mrr,
        "ndcg_at_10": ndcg10,
    }


def _supervised_objective(metrics: dict[str, float]) -> float:
    # Weighted objective to optimize ranking quality.
    return (
        (0.15 * metrics["hit_at_5"])
        + (0.15 * metrics["hit_at_10"])
        + (0.30 * metrics["mrr"])
        + (0.40 * metrics["ndcg_at_10"])
    )


def _domain_keys(domain: str) -> dict[str, str]:
    if domain == "penale":
        return {
            "vector_weight": "search_hybrid_penale_vector_weight",
            "fulltext_weight": "search_hybrid_penale_fulltext_weight",
            "keyword_min_overlap_count": "search_hybrid_penale_keyword_min_overlap_count",
            "zero_overlap_multiplier": "search_hybrid_penale_zero_overlap_multiplier",
            "low_overlap_multiplier": "search_hybrid_penale_low_overlap_multiplier",
        }
    if domain == "civile":
        return {
            "vector_weight": "search_hybrid_civile_vector_weight",
            "fulltext_weight": "search_hybrid_civile_fulltext_weight",
            "keyword_min_overlap_count": "search_hybrid_civile_keyword_min_overlap_count",
            "zero_overlap_multiplier": "search_hybrid_civile_zero_overlap_multiplier",
            "low_overlap_multiplier": "search_hybrid_civile_low_overlap_multiplier",
        }
    if domain == "amministrativo":
        return {
            "vector_weight": "search_hybrid_admin_vector_weight",
            "fulltext_weight": "search_hybrid_admin_fulltext_weight",
            "keyword_min_overlap_count": "search_hybrid_admin_keyword_min_overlap_count",
            "zero_overlap_multiplier": "search_hybrid_admin_zero_overlap_multiplier",
            "low_overlap_multiplier": "search_hybrid_admin_low_overlap_multiplier",
        }
    raise ValueError(f"Unsupported domain: {domain}")


def _source_for_domain(domain: str) -> str:
    return {
        "penale": "codice_penale",
        "civile": "codice_civile",
        "amministrativo": "codice_amministrativo",
    }[domain]


def _candidate_grid_for_domain(domain: str) -> list[dict[str, Any]]:
    if domain == "penale":
        vector_weights = [0.45, 0.55, 0.60, 0.65, 0.75]
        min_overlaps = [1, 2, 3]
        zero_mults = [0.75, 0.85, 0.95, 1.0]
        low_mults = [0.88, 0.94, 1.0]
    elif domain == "civile":
        vector_weights = [0.20, 0.30, 0.40, 0.50, 0.60]
        min_overlaps = [1, 2, 3]
        zero_mults = [0.80, 0.90, 1.0]
        low_mults = [0.90, 0.95, 1.0]
    else:  # amministrativo
        vector_weights = [0.20, 0.35, 0.50]
        min_overlaps = [1, 2, 3]
        zero_mults = [0.85, 0.95, 1.0]
        low_mults = [0.95, 1.0]

    candidates: list[dict[str, Any]] = []
    for vw in vector_weights:
        fw = round(1.0 - vw, 2)
        for mo in min_overlaps:
            for zm in zero_mults:
                for lm in low_mults:
                    if lm < zm:
                        continue
                    candidates.append(
                        {
                            "vector_weight": float(vw),
                            "fulltext_weight": float(fw),
                            "keyword_min_overlap_count": int(mo),
                            "zero_overlap_multiplier": float(zm),
                            "low_overlap_multiplier": float(lm),
                        }
                    )
    return candidates


def _get_current_params(domain: str) -> dict[str, Any]:
    keys = _domain_keys(domain)
    return {k: getattr(settings, attr) for k, attr in keys.items()}


def _apply_params(domain: str, params: dict[str, Any]) -> None:
    keys = _domain_keys(domain)
    for p_name, attr in keys.items():
        setattr(settings, attr, params[p_name])


def _precompute_ranked_lists(
    pipeline: LegalSearchPipeline,
    claims: list[ClaimEntry],
    source: str,
    top_k: int,
    query_terms_mode: str,
    show_progress: bool = True,
    run_idx: int = 1,
    total_runs: int = 1,
) -> dict[str, dict[str, Any]]:
    precomputed: dict[str, dict[str, Any]] = {}
    candidate_limit = max(
        top_k * settings.search_hybrid_candidate_multiplier,
        settings.search_hybrid_candidate_min,
    )
    fused_limit = max(top_k * settings.search_hybrid_fused_pool_multiplier, top_k)

    with pipeline.driver.session() as session:
        for claim in _iter_progress(
            claims,
            enabled=show_progress,
            desc=f"[{source}] precompute r{run_idx}/{total_runs}",
            unit="claim",
            leave=False,
        ):
            embedding = pipeline.embed_text(claim.text)
            query_text = claim.text.strip()
            query_terms = pipeline.get_search_query_terms(
                query_text, mode=query_terms_mode
            )
            fulltext_query = " ".join(sorted(query_terms)) if query_terms else ""
            vector_results = pipeline._vector_exact_search(
                session=session,
                embedding=embedding,
                source=source,
                libro="",
                limit=candidate_limit,
            )
            fulltext_results = pipeline._fulltext_search(
                session=session,
                query_text=fulltext_query,
                source=source,
                libro="",
                limit=candidate_limit,
            )
            precomputed[claim.claim_id] = {
                "claim": claim,
                "query_text": query_text,
                "vector_results": vector_results,
                "fulltext_results": fulltext_results,
                "fused_limit": fused_limit,
            }
    return precomputed


def tune_domain(
    pipeline: LegalSearchPipeline,
    domain: str,
    claims: list[ClaimEntry],
    gold_labels: dict[str, set[str]],
    gold_specs: Optional[dict[str, GoldLabelSpec]],
    supervised: bool,
    query_terms_mode: str,
    per_claim_supervised_constraints: Optional[dict[str, dict[str, float]]] = None,
    top_k: int = DEFAULT_TOP_K,
    show_progress: bool = True,
    num_iterations: int = 1,
    variance_penalty: float = 0.0,
) -> dict[str, Any]:
    base_params = _get_current_params(domain)
    normalized_claim_constraints: dict[str, dict[str, float]] = {
        str(claim_id)
        .strip()
        .upper(): {
            str(metric_name).strip(): float(metric_val)
            for metric_name, metric_val in (constraint_spec or {}).items()
        }
        for claim_id, constraint_spec in (
            per_claim_supervised_constraints or {}
        ).items()
        if isinstance(constraint_spec, dict)
    }
    if not claims:
        return {
            "domain": domain,
            "claims_count": 0,
            "labeled_claims_count": 0,
            "status": "skipped_no_covered_claims",
            "best_params": base_params,
            "best_score": None,
            "num_candidates": 0,
            "tuning_mode": "n/a",
            "query_terms_mode": query_terms_mode,
        }

    labeled_claim_ids = {
        c.claim_id
        for c in claims
        if c.claim_id in gold_labels and gold_labels[c.claim_id]
    }
    must_have_annotated_claim_ids = {
        c.claim_id
        for c in claims
        if (gold_specs or {}).get(c.claim_id) is not None
        and bool((gold_specs or {})[c.claim_id].must_have)
    }
    use_supervised = supervised and len(labeled_claim_ids) > 0

    source = _source_for_domain(domain)
    precomputed_runs: list[dict[str, dict[str, Any]]] = []
    for run_idx in range(1, max(1, num_iterations) + 1):
        _clear_query_terms_cache(pipeline)
        precomputed_runs.append(
            _precompute_ranked_lists(
                pipeline,
                claims,
                source,
                top_k=top_k,
                query_terms_mode=query_terms_mode,
                show_progress=show_progress,
                run_idx=run_idx,
                total_runs=max(1, num_iterations),
            )
        )
    candidates = _candidate_grid_for_domain(domain)

    best_score = -math.inf
    best_score_mean = -math.inf
    best_score_std = 0.0
    best_score_components: dict[str, float] = {}
    best_params = dict(base_params)
    best_relaxed_score = -math.inf
    best_relaxed_score_mean = -math.inf
    best_relaxed_score_std = 0.0
    best_relaxed_score_components: dict[str, float] = {}
    best_relaxed_params = dict(base_params)

    for candidate in _iter_progress(
        candidates,
        enabled=show_progress,
        desc=f"[{domain}] grid",
        unit="cand",
        leave=False,
    ):
        _apply_params(domain, candidate)
        run_scores: list[float] = []
        claim_run_scores: dict[str, list[float]] = {}
        claim_supervised_metric_runs: dict[str, dict[str, list[float]]] = {}
        must_have_rates: list[float] = []
        good_to_have_recalls: list[float] = []
        confuser_penalties: list[float] = []
        for precomputed in precomputed_runs:
            claim_scores: list[float] = []
            run_must_have_flags: list[float] = []
            run_good_recalls: list[float] = []
            run_confuser_vals: list[float] = []
            for entry in precomputed.values():
                claim = entry["claim"]
                fused = pipeline._fuse_ranked_results(
                    source=source,
                    vector_results=entry["vector_results"],
                    fulltext_results=entry["fulltext_results"],
                    query_text=entry["query_text"],
                    limit=entry["fused_limit"],
                )[:top_k]

                if use_supervised:
                    gold = gold_labels.get(claim.claim_id, set())
                    if not gold:
                        continue
                    ranked_keys = _ranked_article_keys(fused)
                    metrics = _supervised_metrics(ranked_keys, gold)
                    claim_score = _supervised_objective(metrics)
                    metrics_for_constraints = metrics
                    if claim.claim_id in normalized_claim_constraints:
                        # Align per-claim constraints with the benchmark view
                        # used by evaluate_domain (hybrid ranked list + CITES).
                        fused_for_constraints = list(fused)
                        fused_with_cites = pipeline.expand_with_cited_articles(
                            fused_for_constraints
                        )[:top_k]
                        metrics_for_constraints = _supervised_metrics(
                            _ranked_article_keys(fused_with_cites), gold
                        )
                    claim_metrics_bucket = claim_supervised_metric_runs.setdefault(
                        claim.claim_id, {}
                    )
                    for metric_name in ("hit_at_5", "hit_at_10", "mrr", "ndcg_at_10"):
                        claim_metrics_bucket.setdefault(metric_name, []).append(
                            float(metrics_for_constraints[metric_name])
                        )
                    spec = (gold_specs or {}).get(claim.claim_id)
                    if spec is not None:
                        if spec.must_have:
                            must_ok = _must_have_all_present_rate(
                                ranked_keys, spec.must_have, top_k
                            )
                            claim_score += (
                                ROBUST_TUNING_POLICY["must_have_bonus"] * must_ok
                            )
                            run_must_have_flags.append(must_ok)
                        if spec.good_to_have:
                            good_recall = _set_recall_at_k(
                                ranked_keys, spec.good_to_have, top_k
                            )
                            claim_score += (
                                ROBUST_TUNING_POLICY["good_to_have_bonus"] * good_recall
                            )
                            run_good_recalls.append(good_recall)
                        if spec.confusers:
                            conf_pen = _confuser_rank_penalty(
                                ranked_keys, spec.confusers, k=min(10, top_k)
                            )
                            claim_score -= (
                                ROBUST_TUNING_POLICY["confuser_penalty"] * conf_pen
                            )
                            run_confuser_vals.append(conf_pen)
                    claim_scores.append(claim_score)
                    claim_run_scores.setdefault(claim.claim_id, []).append(claim_score)
                else:
                    proxy, _, _ = _score_claim_proxy(pipeline, claim.text, fused)
                    claim_scores.append(proxy)
                    claim_run_scores.setdefault(claim.claim_id, []).append(proxy)

            run_scores.append(statistics.fmean(claim_scores) if claim_scores else 0.0)
            if run_must_have_flags:
                must_have_rates.append(float(statistics.fmean(run_must_have_flags)))
            if run_good_recalls:
                good_to_have_recalls.append(float(statistics.fmean(run_good_recalls)))
            if run_confuser_vals:
                confuser_penalties.append(float(statistics.fmean(run_confuser_vals)))

        score_mean = statistics.fmean(run_scores) if run_scores else 0.0
        score_std = _std_or_zero(run_scores)

        claim_means = [
            statistics.fmean(scores) for scores in claim_run_scores.values() if scores
        ]
        claim_mean_stds = [
            _std_or_zero(scores) for scores in claim_run_scores.values() if scores
        ]
        worst_claim_score = _quantile(
            claim_means, ROBUST_TUNING_POLICY["worst_claim_quantile"]
        )
        robust_mean = (
            (1.0 - ROBUST_TUNING_POLICY["worst_claim_weight"]) * score_mean
        ) + (ROBUST_TUNING_POLICY["worst_claim_weight"] * worst_claim_score)
        claim_dispersion = statistics.fmean(claim_mean_stds) if claim_mean_stds else 0.0
        must_have_rate = statistics.fmean(must_have_rates) if must_have_rates else 1.0
        good_to_have_rate = (
            statistics.fmean(good_to_have_recalls) if good_to_have_recalls else 0.0
        )
        confuser_rate = (
            statistics.fmean(confuser_penalties) if confuser_penalties else 0.0
        )

        passes_must_have_gate = True
        # Hard gate for explicitly annotated must-haves only.
        if (
            len(must_have_annotated_claim_ids)
            >= int(ROBUST_TUNING_POLICY["must_have_min_annotated_claims_for_hard_gate"])
            and must_have_rates
            and must_have_rate < ROBUST_TUNING_POLICY["must_have_min_rate"]
        ):
            passes_must_have_gate = False

        passes_per_claim_constraints = True
        per_claim_constraint_values: dict[str, float] = {}
        for claim_id, constraint_spec in normalized_claim_constraints.items():
            claim_metric_runs = claim_supervised_metric_runs.get(claim_id, {})
            for constraint_name, threshold in constraint_spec.items():
                metric_value: Optional[float] = None
                if constraint_name.startswith("min_"):
                    metric_key = constraint_name[4:]
                    vals = claim_metric_runs.get(metric_key, [])
                    metric_value = float(statistics.fmean(vals)) if vals else None
                    if metric_value is None or metric_value < float(threshold):
                        passes_per_claim_constraints = False
                elif constraint_name.startswith("max_"):
                    metric_key = constraint_name[4:]
                    vals = claim_metric_runs.get(metric_key, [])
                    metric_value = float(statistics.fmean(vals)) if vals else None
                    if metric_value is None or metric_value > float(threshold):
                        passes_per_claim_constraints = False
                else:
                    # Unknown constraint format -> fail closed to avoid silent misuse.
                    passes_per_claim_constraints = False
                key_safe = re.sub(
                    r"[^A-Za-z0-9_]+", "_", f"{claim_id}_{constraint_name}"
                )
                per_claim_constraint_values[f"constraint_{key_safe}_target"] = float(
                    threshold
                )
                if metric_value is not None:
                    per_claim_constraint_values[f"constraint_{key_safe}_value"] = float(
                        metric_value
                    )
                per_claim_constraint_values[f"constraint_{key_safe}_passed"] = (
                    1.0
                    if (
                        metric_value is not None
                        and (
                            (
                                constraint_name.startswith("min_")
                                and metric_value >= float(threshold)
                            )
                            or (
                                constraint_name.startswith("max_")
                                and metric_value <= float(threshold)
                            )
                        )
                    )
                    else 0.0
                )

        score = robust_mean - (variance_penalty * (score_std + claim_dispersion))
        components = {
            "robust_mean": float(robust_mean),
            "worst_claim_score_q": float(worst_claim_score),
            "worst_claim_quantile": float(ROBUST_TUNING_POLICY["worst_claim_quantile"]),
            "run_mean_std": float(score_std),
            "claim_mean_std": float(claim_dispersion),
            "must_have_rate": float(must_have_rate),
            "good_to_have_recall": float(good_to_have_rate),
            "confuser_penalty_rate": float(confuser_rate),
            "must_have_annotated_claims": float(len(must_have_annotated_claim_ids)),
            "must_have_gate_passed": 1.0 if passes_must_have_gate else 0.0,
            "per_claim_constraints_passed": (
                1.0 if passes_per_claim_constraints else 0.0
            ),
        }
        components.update(per_claim_constraint_values)

        if score > best_relaxed_score:
            best_relaxed_score = score
            best_relaxed_score_mean = score_mean
            best_relaxed_score_std = score_std
            best_relaxed_score_components = dict(components)
            best_relaxed_params = dict(candidate)

        if not (passes_must_have_gate and passes_per_claim_constraints):
            continue

        if score > best_score:
            best_score = score
            best_score_mean = score_mean
            best_score_std = score_std
            best_score_components = dict(components)
            best_params = dict(candidate)

    if not math.isfinite(best_score):
        if math.isfinite(best_relaxed_score):
            _apply_params(domain, best_relaxed_params)
            return {
                "domain": domain,
                "claims_count": len(claims),
                "labeled_claims_count": len(labeled_claim_ids),
                "status": "ok_relaxed_constraints",
                "best_params": best_relaxed_params,
                "best_score": round(float(best_relaxed_score), 6),
                "best_score_mean": round(float(best_relaxed_score_mean), 6),
                "best_score_std": round(float(best_relaxed_score_std), 6),
                "best_score_components": {
                    k: round(float(v), 6)
                    for k, v in best_relaxed_score_components.items()
                },
                "num_candidates": len(candidates),
                "tuning_mode": "supervised" if use_supervised else "proxy_fallback",
                "query_terms_mode": query_terms_mode,
                "num_iterations": max(1, int(num_iterations)),
                "variance_penalty": float(variance_penalty),
                "robust_tuning_policy": dict(ROBUST_TUNING_POLICY),
                "constraints_relaxed": (
                    (["must_have_min_rate"] if must_have_annotated_claim_ids else [])
                    + (
                        ["per_claim_supervised_constraints"]
                        if normalized_claim_constraints
                        else []
                    )
                ),
            }
        _apply_params(domain, base_params)
        return {
            "domain": domain,
            "claims_count": len(claims),
            "labeled_claims_count": len(labeled_claim_ids),
            "status": "skipped_no_candidate_passed_constraints",
            "best_params": base_params,
            "best_score": None,
            "best_score_mean": None,
            "best_score_std": None,
            "best_score_components": {},
            "num_candidates": len(candidates),
            "tuning_mode": "supervised" if use_supervised else "proxy_fallback",
            "query_terms_mode": query_terms_mode,
            "num_iterations": max(1, int(num_iterations)),
            "variance_penalty": float(variance_penalty),
            "robust_tuning_policy": dict(ROBUST_TUNING_POLICY),
            "per_claim_supervised_constraints": dict(normalized_claim_constraints),
        }

    _apply_params(domain, best_params)

    return {
        "domain": domain,
        "claims_count": len(claims),
        "labeled_claims_count": len(labeled_claim_ids),
        "status": "ok",
        "best_params": best_params,
        "best_score": round(float(best_score), 6),
        "best_score_mean": round(float(best_score_mean), 6),
        "best_score_std": round(float(best_score_std), 6),
        "best_score_components": {
            k: round(float(v), 6) for k, v in best_score_components.items()
        },
        "num_candidates": len(candidates),
        "tuning_mode": "supervised" if use_supervised else "proxy_fallback",
        "query_terms_mode": query_terms_mode,
        "num_iterations": max(1, int(num_iterations)),
        "variance_penalty": float(variance_penalty),
        "robust_tuning_policy": dict(ROBUST_TUNING_POLICY),
        "per_claim_supervised_constraints": dict(normalized_claim_constraints),
    }


def evaluate_domain(
    pipeline: LegalSearchPipeline,
    domain: str,
    claims: list[ClaimEntry],
    gold_labels: dict[str, set[str]],
    query_terms_mode: str,
    top_k: int = DEFAULT_TOP_K,
    show_progress: bool = True,
    num_iterations: int = 1,
) -> dict[str, Any]:
    if not claims:
        return {
            "domain": domain,
            "claims_count": 0,
            "labeled_claims_count": 0,
            "status": "skipped_no_covered_claims",
            "mean_proxy_score": None,
            "mean_overlap_top10": None,
            "mean_coverage_terms_top10": None,
            "mean_hit_at_5": None,
            "mean_hit_at_10": None,
            "mean_mrr": None,
            "mean_ndcg_at_10": None,
            "entries": [],
        }

    source = _source_for_domain(domain)
    filters = [(source, "")]
    entries: list[EvalEntry] = []

    hit5_vals: list[float] = []
    hit10_vals: list[float] = []
    mrr_vals: list[float] = []
    ndcg_vals: list[float] = []
    proxy_std_vals: list[float] = []
    hit5_std_vals: list[float] = []
    hit10_std_vals: list[float] = []
    mrr_std_vals: list[float] = []
    ndcg_std_vals: list[float] = []

    for claim in _iter_progress(
        claims,
        enabled=show_progress,
        desc=f"[{domain}] benchmark",
        unit="claim",
        leave=False,
    ):
        embedding = pipeline.embed_text(claim.text)
        run_proxies: list[float] = []
        run_overlaps: list[float] = []
        run_coverages: list[float] = []
        run_hit5: list[float] = []
        run_hit10: list[float] = []
        run_mrr: list[float] = []
        run_ndcg: list[float] = []
        top_sources: dict[str, int] = {}
        top_articles: list[str] = []
        gold = gold_labels.get(claim.claim_id, set())

        for run_idx in range(max(1, int(num_iterations))):
            _clear_query_terms_cache(pipeline)
            results = pipeline.vector_search(
                embedding=embedding,
                libri_filters=filters,
                top_k=top_k,
                query_text=claim.text,
            )
            results = pipeline.expand_with_cited_articles(results)[:top_k]

            proxy, mean_overlap, coverage_terms = _score_claim_proxy(
                pipeline, claim.text, results
            )
            run_proxies.append(float(proxy))
            run_overlaps.append(float(mean_overlap))
            run_coverages.append(float(coverage_terms))

            if run_idx == 0:
                for item in results[:10]:
                    top_sources[item.source] = top_sources.get(item.source, 0) + 1
                    top_articles.append(_article_key(item.source, item.articolo))

            if gold:
                run_metrics = _supervised_metrics(_ranked_article_keys(results), gold)
                run_hit5.append(run_metrics["hit_at_5"])
                run_hit10.append(run_metrics["hit_at_10"])
                run_mrr.append(run_metrics["mrr"])
                run_ndcg.append(run_metrics["ndcg_at_10"])

        metrics: Optional[dict[str, float]] = None
        if gold and run_hit5:
            metrics = {
                "hit_at_5": statistics.fmean(run_hit5),
                "hit_at_10": statistics.fmean(run_hit10),
                "mrr": statistics.fmean(run_mrr),
                "ndcg_at_10": statistics.fmean(run_ndcg),
            }
            hit5_vals.append(metrics["hit_at_5"])
            hit10_vals.append(metrics["hit_at_10"])
            mrr_vals.append(metrics["mrr"])
            ndcg_vals.append(metrics["ndcg_at_10"])
            hit5_std_vals.append(_std_or_zero(run_hit5))
            hit10_std_vals.append(_std_or_zero(run_hit10))
            mrr_std_vals.append(_std_or_zero(run_mrr))
            ndcg_std_vals.append(_std_or_zero(run_ndcg))
        proxy_std_vals.append(_std_or_zero(run_proxies))

        entries.append(
            EvalEntry(
                claim_id=claim.claim_id,
                domain=domain,
                source=source,
                proxy_score=float(statistics.fmean(run_proxies)),
                mean_overlap_top10=float(statistics.fmean(run_overlaps)),
                coverage_terms_top10=float(statistics.fmean(run_coverages)),
                top_sources=top_sources,
                top_articles=top_articles,
                gold_count=len(gold),
                hit_at_5=(metrics["hit_at_5"] if metrics else None),
                hit_at_10=(metrics["hit_at_10"] if metrics else None),
                mrr=(metrics["mrr"] if metrics else None),
                ndcg_at_10=(metrics["ndcg_at_10"] if metrics else None),
                runs=max(1, int(num_iterations)),
                proxy_score_std=_std_or_zero(run_proxies),
                mean_overlap_top10_std=_std_or_zero(run_overlaps),
                coverage_terms_top10_std=_std_or_zero(run_coverages),
                hit_at_5_std=(_std_or_zero(run_hit5) if run_hit5 else None),
                hit_at_10_std=(_std_or_zero(run_hit10) if run_hit10 else None),
                mrr_std=(_std_or_zero(run_mrr) if run_mrr else None),
                ndcg_at_10_std=(_std_or_zero(run_ndcg) if run_ndcg else None),
            )
        )

    mean_hit5 = _mean_or_none(hit5_vals)
    mean_hit10 = _mean_or_none(hit10_vals)
    mean_mrr = _mean_or_none(mrr_vals)
    mean_ndcg = _mean_or_none(ndcg_vals)

    return {
        "domain": domain,
        "claims_count": len(claims),
        "labeled_claims_count": len([c for c in claims if gold_labels.get(c.claim_id)]),
        "status": "ok",
        "query_terms_mode": query_terms_mode,
        "num_iterations": max(1, int(num_iterations)),
        "mean_proxy_score": round(statistics.fmean(e.proxy_score for e in entries), 6),
        "mean_proxy_score_std_across_runs": round(
            _mean_or_none(proxy_std_vals) or 0.0, 6
        ),
        "mean_overlap_top10": round(
            statistics.fmean(e.mean_overlap_top10 for e in entries), 6
        ),
        "mean_coverage_terms_top10": round(
            statistics.fmean(e.coverage_terms_top10 for e in entries), 6
        ),
        "mean_hit_at_5": (round(mean_hit5, 6) if mean_hit5 is not None else None),
        "mean_hit_at_10": (round(mean_hit10, 6) if mean_hit10 is not None else None),
        "mean_mrr": (round(mean_mrr, 6) if mean_mrr is not None else None),
        "mean_ndcg_at_10": (round(mean_ndcg, 6) if mean_ndcg is not None else None),
        "mean_hit_at_5_std_across_runs": (
            round(_mean_or_none(hit5_std_vals) or 0.0, 6) if hit5_std_vals else None
        ),
        "mean_hit_at_10_std_across_runs": (
            round(_mean_or_none(hit10_std_vals) or 0.0, 6) if hit10_std_vals else None
        ),
        "mean_mrr_std_across_runs": (
            round(_mean_or_none(mrr_std_vals) or 0.0, 6) if mrr_std_vals else None
        ),
        "mean_ndcg_at_10_std_across_runs": (
            round(_mean_or_none(ndcg_std_vals) or 0.0, 6) if ndcg_std_vals else None
        ),
        "entries": [asdict(e) for e in entries],
    }


def evaluate_mixed(
    pipeline: LegalSearchPipeline,
    claims: list[ClaimEntry],
    gold_labels: dict[str, set[str]],
    query_terms_mode: str,
    top_k: int = DEFAULT_TOP_K,
    show_progress: bool = True,
    num_iterations: int = 1,
) -> dict[str, Any]:
    if not claims:
        return {
            "domain": "misto",
            "claims_count": 0,
            "labeled_claims_count": 0,
            "status": "skipped_no_covered_claims",
            "mean_proxy_score": None,
            "mean_hit_at_5": None,
            "mean_hit_at_10": None,
            "mean_mrr": None,
            "mean_ndcg_at_10": None,
            "entries": [],
        }

    filters = [
        ("codice_civile", ""),
        ("codice_penale", ""),
        ("codice_amministrativo", ""),
    ]

    entries: list[EvalEntry] = []
    hit5_vals: list[float] = []
    hit10_vals: list[float] = []
    mrr_vals: list[float] = []
    ndcg_vals: list[float] = []
    proxy_std_vals: list[float] = []
    hit5_std_vals: list[float] = []
    hit10_std_vals: list[float] = []
    mrr_std_vals: list[float] = []
    ndcg_std_vals: list[float] = []

    for claim in _iter_progress(
        claims,
        enabled=show_progress,
        desc="[misto] benchmark",
        unit="claim",
        leave=False,
    ):
        embedding = pipeline.embed_text(claim.text)
        run_proxies: list[float] = []
        run_overlaps: list[float] = []
        run_coverages: list[float] = []
        run_hit5: list[float] = []
        run_hit10: list[float] = []
        run_mrr: list[float] = []
        run_ndcg: list[float] = []
        top_sources: dict[str, int] = {}
        top_articles: list[str] = []
        gold = gold_labels.get(claim.claim_id, set())

        for run_idx in range(max(1, int(num_iterations))):
            _clear_query_terms_cache(pipeline)
            results = pipeline.vector_search(
                embedding=embedding,
                libri_filters=filters,
                top_k=top_k,
                query_text=claim.text,
            )
            results = pipeline.expand_with_cited_articles(results)[:top_k]

            proxy, mean_overlap, coverage_terms = _score_claim_proxy(
                pipeline, claim.text, results
            )
            run_proxies.append(float(proxy))
            run_overlaps.append(float(mean_overlap))
            run_coverages.append(float(coverage_terms))

            if run_idx == 0:
                for item in results[:10]:
                    top_sources[item.source] = top_sources.get(item.source, 0) + 1
                    top_articles.append(_article_key(item.source, item.articolo))

            if gold:
                run_metrics = _supervised_metrics(_ranked_article_keys(results), gold)
                run_hit5.append(run_metrics["hit_at_5"])
                run_hit10.append(run_metrics["hit_at_10"])
                run_mrr.append(run_metrics["mrr"])
                run_ndcg.append(run_metrics["ndcg_at_10"])

        metrics: Optional[dict[str, float]] = None
        if gold and run_hit5:
            metrics = {
                "hit_at_5": statistics.fmean(run_hit5),
                "hit_at_10": statistics.fmean(run_hit10),
                "mrr": statistics.fmean(run_mrr),
                "ndcg_at_10": statistics.fmean(run_ndcg),
            }
            hit5_vals.append(metrics["hit_at_5"])
            hit10_vals.append(metrics["hit_at_10"])
            mrr_vals.append(metrics["mrr"])
            ndcg_vals.append(metrics["ndcg_at_10"])
            hit5_std_vals.append(_std_or_zero(run_hit5))
            hit10_std_vals.append(_std_or_zero(run_hit10))
            mrr_std_vals.append(_std_or_zero(run_mrr))
            ndcg_std_vals.append(_std_or_zero(run_ndcg))
        proxy_std_vals.append(_std_or_zero(run_proxies))

        entries.append(
            EvalEntry(
                claim_id=claim.claim_id,
                domain="misto",
                source="multi",
                proxy_score=float(statistics.fmean(run_proxies)),
                mean_overlap_top10=float(statistics.fmean(run_overlaps)),
                coverage_terms_top10=float(statistics.fmean(run_coverages)),
                top_sources=top_sources,
                top_articles=top_articles,
                gold_count=len(gold),
                hit_at_5=(metrics["hit_at_5"] if metrics else None),
                hit_at_10=(metrics["hit_at_10"] if metrics else None),
                mrr=(metrics["mrr"] if metrics else None),
                ndcg_at_10=(metrics["ndcg_at_10"] if metrics else None),
                runs=max(1, int(num_iterations)),
                proxy_score_std=_std_or_zero(run_proxies),
                mean_overlap_top10_std=_std_or_zero(run_overlaps),
                coverage_terms_top10_std=_std_or_zero(run_coverages),
                hit_at_5_std=(_std_or_zero(run_hit5) if run_hit5 else None),
                hit_at_10_std=(_std_or_zero(run_hit10) if run_hit10 else None),
                mrr_std=(_std_or_zero(run_mrr) if run_mrr else None),
                ndcg_at_10_std=(_std_or_zero(run_ndcg) if run_ndcg else None),
            )
        )

    mean_hit5 = _mean_or_none(hit5_vals)
    mean_hit10 = _mean_or_none(hit10_vals)
    mean_mrr = _mean_or_none(mrr_vals)
    mean_ndcg = _mean_or_none(ndcg_vals)

    return {
        "domain": "misto",
        "claims_count": len(claims),
        "labeled_claims_count": len([c for c in claims if gold_labels.get(c.claim_id)]),
        "status": "ok",
        "query_terms_mode": query_terms_mode,
        "num_iterations": max(1, int(num_iterations)),
        "mean_proxy_score": round(statistics.fmean(e.proxy_score for e in entries), 6),
        "mean_proxy_score_std_across_runs": round(
            _mean_or_none(proxy_std_vals) or 0.0, 6
        ),
        "mean_overlap_top10": round(
            statistics.fmean(e.mean_overlap_top10 for e in entries), 6
        ),
        "mean_coverage_terms_top10": round(
            statistics.fmean(e.coverage_terms_top10 for e in entries), 6
        ),
        "mean_hit_at_5": (round(mean_hit5, 6) if mean_hit5 is not None else None),
        "mean_hit_at_10": (round(mean_hit10, 6) if mean_hit10 is not None else None),
        "mean_mrr": (round(mean_mrr, 6) if mean_mrr is not None else None),
        "mean_ndcg_at_10": (round(mean_ndcg, 6) if mean_ndcg is not None else None),
        "mean_hit_at_5_std_across_runs": (
            round(_mean_or_none(hit5_std_vals) or 0.0, 6) if hit5_std_vals else None
        ),
        "mean_hit_at_10_std_across_runs": (
            round(_mean_or_none(hit10_std_vals) or 0.0, 6) if hit10_std_vals else None
        ),
        "mean_mrr_std_across_runs": (
            round(_mean_or_none(mrr_std_vals) or 0.0, 6) if mrr_std_vals else None
        ),
        "mean_ndcg_at_10_std_across_runs": (
            round(_mean_or_none(ndcg_std_vals) or 0.0, 6) if ndcg_std_vals else None
        ),
        "entries": [asdict(e) for e in entries],
    }


def _evaluate_pipeline_stage_metrics(
    pipeline: LegalSearchPipeline,
    retrieval_agent: RetrievalFilterAgent,
    claims: list[ClaimEntry],
    gold_labels: dict[str, set[str]],
    gold_specs: dict[str, GoldLabelSpec],
    top_k: int,
    show_progress: bool,
    num_iterations: int,
    progress_desc: str,
) -> dict[str, Any]:
    entries: list[StageEvalEntry] = []
    claim_iter = _iter_progress(
        claims,
        enabled=show_progress,
        desc=progress_desc,
        unit="claim",
        leave=False,
    )
    claim_bar = claim_iter if show_progress else None

    for claim in claim_iter:
        if claim_bar is not None:
            try:
                claim_bar.set_postfix_str(f"{claim.claim_id}: setup")
            except Exception:
                pass
        gold = gold_labels.get(claim.claim_id, set())
        spec = gold_specs.get(claim.claim_id)
        must_have = set(spec.must_have) if spec else set()
        confusers = set(spec.confusers) if spec else set()

        raw_counts: list[float] = []
        cites_counts: list[float] = []
        rel_counts: list[float] = []
        app_counts: list[float] = []
        rel_keep_ratios: list[float] = []
        app_keep_over_rel_ratios: list[float] = []

        raw_gold_recalls: list[float] = []
        cites_gold_recalls: list[float] = []
        rel_gold_recalls: list[float] = []
        app_gold_recalls: list[float] = []

        raw_must_recalls: list[float] = []
        cites_must_recalls: list[float] = []
        rel_must_recalls: list[float] = []
        app_must_recalls: list[float] = []
        raw_must_all: list[float] = []
        cites_must_all: list[float] = []
        rel_must_all: list[float] = []
        app_must_all: list[float] = []
        must_drop_rel_vals: list[float] = []
        must_drop_app_vals: list[float] = []

        conf_pen_raw: list[float] = []
        conf_pen_cites: list[float] = []
        conf_pen_post_app: list[float] = []

        raw_top_articles: list[str] = []
        cites_top_articles: list[str] = []
        post_app_top_articles: list[str] = []

        for run_idx in range(max(1, int(num_iterations))):
            if claim_bar is not None:
                try:
                    claim_bar.set_postfix_str(
                        f"{claim.claim_id}: run {run_idx+1}/{max(1, int(num_iterations))} search"
                    )
                except Exception:
                    pass
            _clear_query_terms_cache(pipeline)
            classification = pipeline.classifier.classify(claim.text)
            embedding = pipeline.embed_text(claim.text)
            libri_filters = pipeline.build_search_filters(
                classification, settings.search_use_top_n_libri
            )

            raw_results = pipeline.vector_search(
                embedding=embedding,
                libri_filters=libri_filters,
                top_k=top_k,
                query_text=claim.text,
            )[:top_k]
            if claim_bar is not None:
                try:
                    claim_bar.set_postfix_str(f"{claim.claim_id}: +CITES")
                except Exception:
                    pass
            cites_results = pipeline.expand_with_cited_articles(list(raw_results))[
                :top_k
            ]

            cites_statutes = _results_to_statute_dicts(cites_results)
            if claim_bar is not None:
                try:
                    claim_bar.set_postfix_str(
                        f"{claim.claim_id}: legal-context ({len(cites_statutes)} cand)"
                    )
                except Exception:
                    pass
            legal_context = retrieval_agent._extract_legal_context(claim.text)
            if claim_bar is not None:
                try:
                    claim_bar.set_postfix_str(
                        f"{claim.claim_id}: relevance filter ({len(cites_statutes)})"
                    )
                except Exception:
                    pass
            rel_kept = retrieval_agent.filter_irrelevant_statutes(
                claim.text, cites_statutes
            )
            if claim_bar is not None:
                try:
                    claim_bar.set_postfix_str(
                        f"{claim.claim_id}: applicability filter ({len(rel_kept)})"
                    )
                except Exception:
                    pass
            app_kept = retrieval_agent.filter_applicable_statutes(
                claim.text, rel_kept, legal_context
            )
            if claim_bar is not None:
                try:
                    claim_bar.set_postfix_str(
                        f"{claim.claim_id}: done raw={len(raw_results)} cites={len(cites_results)} rel={len(rel_kept)} app={len(app_kept)}"
                    )
                except Exception:
                    pass

            raw_keys = _ranked_article_keys(raw_results)
            cites_keys = _ranked_article_keys(cites_results)
            rel_keys = _ranked_statute_dict_keys(rel_kept)
            app_keys = _ranked_statute_dict_keys(app_kept)

            raw_counts.append(float(len(raw_results)))
            cites_counts.append(float(len(cites_results)))
            rel_counts.append(float(len(rel_kept)))
            app_counts.append(float(len(app_kept)))
            rel_keep_ratios.append(float(len(rel_kept) / max(1, len(cites_results))))
            app_keep_over_rel_ratios.append(
                float(len(app_kept) / max(1, len(rel_kept)))
            )

            if gold:
                raw_gold_recalls.append(_set_recall_at_k(raw_keys, gold, top_k))
                cites_gold_recalls.append(_set_recall_at_k(cites_keys, gold, top_k))
                rel_gold_recalls.append(_set_recall_at_k(rel_keys, gold, top_k))
                app_gold_recalls.append(_set_recall_at_k(app_keys, gold, top_k))

            if must_have:
                raw_must_recalls.append(_set_recall_at_k(raw_keys, must_have, top_k))
                cites_must_recalls.append(
                    _set_recall_at_k(cites_keys, must_have, top_k)
                )
                rel_must_recalls.append(_set_recall_at_k(rel_keys, must_have, top_k))
                app_must_recalls.append(_set_recall_at_k(app_keys, must_have, top_k))
                raw_must_all.append(
                    _must_have_all_present_rate(raw_keys, must_have, top_k)
                )
                cites_must_all.append(
                    _must_have_all_present_rate(cites_keys, must_have, top_k)
                )
                rel_must_all.append(
                    _must_have_all_present_rate(rel_keys, must_have, top_k)
                )
                app_must_all.append(
                    _must_have_all_present_rate(app_keys, must_have, top_k)
                )

                cites_set = set(cites_keys)
                rel_set = set(rel_keys)
                app_set = set(app_keys)
                mh_cites = must_have & cites_set
                mh_rel = must_have & rel_set
                mh_app = must_have & app_set
                must_drop_rel_vals.append(
                    float((len(mh_cites) - len(mh_rel)) / max(1, len(mh_cites)))
                    if mh_cites
                    else 0.0
                )
                must_drop_app_vals.append(
                    float((len(mh_rel) - len(mh_app)) / max(1, len(mh_rel)))
                    if mh_rel
                    else 0.0
                )

            if confusers:
                conf_pen_raw.append(
                    _confuser_rank_penalty(raw_keys, confusers, k=min(10, top_k))
                )
                conf_pen_cites.append(
                    _confuser_rank_penalty(cites_keys, confusers, k=min(10, top_k))
                )
                conf_pen_post_app.append(
                    _confuser_rank_penalty(app_keys, confusers, k=min(10, top_k))
                )

            if run_idx == 0:
                raw_top_articles = raw_keys[:10]
                cites_top_articles = cites_keys[:10]
                post_app_top_articles = app_keys[:10]

        classification = (
            "multi" if claim.domain == "misto" else _source_for_domain(claim.domain)
        )
        entries.append(
            StageEvalEntry(
                claim_id=claim.claim_id,
                domain=claim.domain,
                source=classification,
                gold_count=len(gold),
                must_have_count=len(must_have),
                confusers_count=len(confusers),
                runs=max(1, int(num_iterations)),
                raw_topk_count_mean=(
                    float(statistics.fmean(raw_counts)) if raw_counts else 0.0
                ),
                cites_topk_count_mean=(
                    float(statistics.fmean(cites_counts)) if cites_counts else 0.0
                ),
                relevance_kept_count_mean=(
                    float(statistics.fmean(rel_counts)) if rel_counts else 0.0
                ),
                applicability_kept_count_mean=(
                    float(statistics.fmean(app_counts)) if app_counts else 0.0
                ),
                raw_gold_recall_at_k_mean=_mean_or_none(raw_gold_recalls),
                cites_gold_recall_at_k_mean=_mean_or_none(cites_gold_recalls),
                post_rel_gold_recall_mean=_mean_or_none(rel_gold_recalls),
                post_app_gold_recall_mean=_mean_or_none(app_gold_recalls),
                raw_must_have_recall_at_k_mean=_mean_or_none(raw_must_recalls),
                cites_must_have_recall_at_k_mean=_mean_or_none(cites_must_recalls),
                post_rel_must_have_recall_mean=_mean_or_none(rel_must_recalls),
                post_app_must_have_recall_mean=_mean_or_none(app_must_recalls),
                raw_must_have_all_present_rate=_mean_or_none(raw_must_all),
                cites_must_have_all_present_rate=_mean_or_none(cites_must_all),
                post_rel_must_have_all_present_rate=_mean_or_none(rel_must_all),
                post_app_must_have_all_present_rate=_mean_or_none(app_must_all),
                rel_filter_keep_ratio_mean=_mean_or_none(rel_keep_ratios),
                app_filter_keep_ratio_over_rel_mean=_mean_or_none(
                    app_keep_over_rel_ratios
                ),
                must_have_dropped_by_relevance_mean=_mean_or_none(must_drop_rel_vals),
                must_have_dropped_by_applicability_mean=_mean_or_none(
                    must_drop_app_vals
                ),
                confuser_penalty_top10_raw_mean=_mean_or_none(conf_pen_raw),
                confuser_penalty_top10_cites_mean=_mean_or_none(conf_pen_cites),
                confuser_penalty_top10_post_app_mean=_mean_or_none(conf_pen_post_app),
                raw_top_articles=raw_top_articles,
                cites_top_articles=cites_top_articles,
                post_app_top_articles=post_app_top_articles,
            )
        )
        if claim_bar is not None:
            try:
                claim_bar.set_postfix_str(f"{claim.claim_id}: summarized")
            except Exception:
                pass

    def _domain_stage_summary(domain_name: str) -> dict[str, Any]:
        domain_entries = [e for e in entries if e.domain == domain_name]
        if not domain_entries:
            return {
                "domain": domain_name,
                "claims_count": 0,
                "status": "skipped_no_covered_claims",
                "entries": [],
            }

        def _vals(field_name: str) -> list[float]:
            vals: list[float] = []
            for e in domain_entries:
                v = getattr(e, field_name)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            return vals

        return {
            "domain": domain_name,
            "claims_count": len(domain_entries),
            "status": "ok",
            "top_k_stage": int(top_k),
            "num_iterations": max(1, int(num_iterations)),
            "mean_raw_gold_recall_at_k": _mean_or_none(
                _vals("raw_gold_recall_at_k_mean")
            ),
            "mean_cites_gold_recall_at_k": _mean_or_none(
                _vals("cites_gold_recall_at_k_mean")
            ),
            "mean_post_rel_gold_recall": _mean_or_none(
                _vals("post_rel_gold_recall_mean")
            ),
            "mean_post_app_gold_recall": _mean_or_none(
                _vals("post_app_gold_recall_mean")
            ),
            "mean_raw_must_have_recall_at_k": _mean_or_none(
                _vals("raw_must_have_recall_at_k_mean")
            ),
            "mean_cites_must_have_recall_at_k": _mean_or_none(
                _vals("cites_must_have_recall_at_k_mean")
            ),
            "mean_post_rel_must_have_recall": _mean_or_none(
                _vals("post_rel_must_have_recall_mean")
            ),
            "mean_post_app_must_have_recall": _mean_or_none(
                _vals("post_app_must_have_recall_mean")
            ),
            "mean_raw_must_have_all_present_rate": _mean_or_none(
                _vals("raw_must_have_all_present_rate")
            ),
            "mean_cites_must_have_all_present_rate": _mean_or_none(
                _vals("cites_must_have_all_present_rate")
            ),
            "mean_post_app_must_have_all_present_rate": _mean_or_none(
                _vals("post_app_must_have_all_present_rate")
            ),
            "mean_rel_filter_keep_ratio": _mean_or_none(
                _vals("rel_filter_keep_ratio_mean")
            ),
            "mean_app_filter_keep_over_rel_ratio": _mean_or_none(
                _vals("app_filter_keep_ratio_over_rel_mean")
            ),
            "mean_must_have_dropped_by_relevance": _mean_or_none(
                _vals("must_have_dropped_by_relevance_mean")
            ),
            "mean_must_have_dropped_by_applicability": _mean_or_none(
                _vals("must_have_dropped_by_applicability_mean")
            ),
            "mean_confuser_penalty_top10_raw": _mean_or_none(
                _vals("confuser_penalty_top10_raw_mean")
            ),
            "mean_confuser_penalty_top10_cites": _mean_or_none(
                _vals("confuser_penalty_top10_cites_mean")
            ),
            "mean_confuser_penalty_top10_post_app": _mean_or_none(
                _vals("confuser_penalty_top10_post_app_mean")
            ),
            "entries": [asdict(e) for e in domain_entries],
        }

    return {
        domain_name: _domain_stage_summary(domain_name)
        for domain_name in ("penale", "civile", "amministrativo", "misto")
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune retrieval parameters on claims set."
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--gold-labels",
        type=Path,
        default=GOLD_LABELS_PATH,
        help="Path to claims gold labels JSON.",
    )
    parser.add_argument(
        "--query-terms-mode",
        choices=["llm"],
        default="llm",
        help="Term extraction mode used to build fulltext query (LLM-only).",
    )
    parser.add_argument(
        "--unsupervised",
        action="store_true",
        help="Force proxy-only tuning even if gold labels are available.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Repeat retrieval runs per claim/config and aggregate to reduce variance.",
    )
    parser.add_argument(
        "--variance-penalty",
        type=float,
        default=0.15,
        help="Penalty multiplier on per-candidate std across iterations (tuning only).",
    )
    parser.add_argument(
        "--include-stage-metrics",
        action="store_true",
        help="Compute pipeline-aligned stage metrics (raw@K, +CITES, post relevance/applicability) in benchmark output.",
    )
    parser.add_argument(
        "--stage-top-k",
        type=int,
        default=100,
        help="Top-k for pipeline-aligned stage metrics (default: 100).",
    )
    parser.add_argument(
        "--stage-metrics-domains",
        nargs="*",
        choices=["penale", "civile", "amministrativo", "misto"],
        default=[],
        help="Optional subset of domains for stage metrics (default: all).",
    )
    parser.add_argument(
        "--stage-metrics-claim-ids",
        nargs="*",
        default=[],
        help="Optional subset of claim IDs for stage metrics only (e.g. P2 P4 P6).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    top_k = max(1, int(args.top_k))
    query_terms_mode = str(args.query_terms_mode).strip().lower()
    show_progress = not args.no_progress
    num_iterations = max(1, int(args.num_iterations))
    variance_penalty = max(0.0, float(args.variance_penalty))
    include_stage_metrics = bool(args.include_stage_metrics)
    stage_top_k = max(1, int(args.stage_top_k))
    stage_metrics_domains = (
        set(args.stage_metrics_domains)
        if args.stage_metrics_domains
        else {"penale", "civile", "amministrativo", "misto"}
    )
    stage_metrics_claim_ids = {
        str(cid).strip().upper()
        for cid in args.stage_metrics_claim_ids
        if str(cid).strip()
    }

    covered_claims, non_covered_count = parse_claims_md(CLAIMS_MD_PATH)
    gold_specs = load_gold_label_specs(args.gold_labels)
    gold_labels = {cid: set(spec.gold) for cid, spec in gold_specs.items() if spec.gold}

    by_domain: dict[str, list[ClaimEntry]] = {
        "penale": [c for c in covered_claims if c.domain == "penale"],
        "civile": [c for c in covered_claims if c.domain == "civile"],
        "amministrativo": [c for c in covered_claims if c.domain == "amministrativo"],
        "misto": [c for c in covered_claims if c.domain == "misto"],
    }

    supervised = (not args.unsupervised) and bool(gold_labels)

    print("=== Claims dataset ===")
    print(f"Totale claims coperti parsed: {len(covered_claims)}")
    print(f"Non coperti dichiarati: {non_covered_count}")
    print(f"Claims coperti usati: {len(covered_claims)}")
    print(f"Gold labels loaded: {len(gold_labels)} claim(s) from {args.gold_labels}")
    print(f"Tuning mode requested: {'supervised' if supervised else 'proxy_fallback'}")
    print(f"Query terms mode: {query_terms_mode}")
    print(f"Retrieval iterations per claim: {num_iterations}")
    print(f"Variance penalty (tuning): {variance_penalty}")
    print(f"Stage metrics enabled: {include_stage_metrics}")
    if include_stage_metrics:
        print(f"Stage metrics top_k: {stage_top_k}")
        print(f"Stage metrics domains: {sorted(stage_metrics_domains)}")
        if stage_metrics_claim_ids:
            print(f"Stage metrics claim_ids: {sorted(stage_metrics_claim_ids)}")
    for k in ("penale", "civile", "amministrativo", "misto"):
        print(f"- {k}: {len(by_domain[k])}")

    if not supervised:
        print("⚠️ Using proxy fallback (no gold labels or --unsupervised).")

    pipeline = LegalSearchPipeline()
    try:
        original_params = {
            "penale": _get_current_params("penale"),
            "civile": _get_current_params("civile"),
            "amministrativo": _get_current_params("amministrativo"),
        }

        tuning_summary = {
            "penale": tune_domain(
                pipeline,
                "penale",
                by_domain["penale"],
                gold_labels,
                gold_specs,
                supervised,
                query_terms_mode=query_terms_mode,
                top_k=top_k,
                show_progress=show_progress,
                num_iterations=num_iterations,
                variance_penalty=variance_penalty,
            ),
            "civile": tune_domain(
                pipeline,
                "civile",
                by_domain["civile"],
                gold_labels,
                gold_specs,
                supervised,
                query_terms_mode=query_terms_mode,
                top_k=top_k,
                show_progress=show_progress,
                num_iterations=num_iterations,
                variance_penalty=variance_penalty,
            ),
            "amministrativo": tune_domain(
                pipeline,
                "amministrativo",
                by_domain["amministrativo"],
                gold_labels,
                gold_specs,
                supervised,
                query_terms_mode=query_terms_mode,
                top_k=top_k,
                show_progress=show_progress,
                num_iterations=num_iterations,
                variance_penalty=variance_penalty,
            ),
        }

        benchmark_summary = {
            "penale": evaluate_domain(
                pipeline,
                "penale",
                by_domain["penale"],
                gold_labels,
                query_terms_mode=query_terms_mode,
                top_k=top_k,
                show_progress=show_progress,
                num_iterations=num_iterations,
            ),
            "civile": evaluate_domain(
                pipeline,
                "civile",
                by_domain["civile"],
                gold_labels,
                query_terms_mode=query_terms_mode,
                top_k=top_k,
                show_progress=show_progress,
                num_iterations=num_iterations,
            ),
            "amministrativo": evaluate_domain(
                pipeline,
                "amministrativo",
                by_domain["amministrativo"],
                gold_labels,
                query_terms_mode=query_terms_mode,
                top_k=top_k,
                show_progress=show_progress,
                num_iterations=num_iterations,
            ),
            "misto": evaluate_mixed(
                pipeline,
                by_domain["misto"],
                gold_labels,
                query_terms_mode=query_terms_mode,
                top_k=top_k,
                show_progress=show_progress,
                num_iterations=num_iterations,
            ),
        }
        stage_metrics_summary: dict[str, Any] = {}
        if include_stage_metrics:
            retrieval_agent = _make_retrieval_filter_agent()
            claims_for_stage_metrics = [
                c for c in covered_claims if c.domain in stage_metrics_domains
            ]
            if stage_metrics_claim_ids:
                claims_for_stage_metrics = [
                    c
                    for c in claims_for_stage_metrics
                    if c.claim_id in stage_metrics_claim_ids
                ]
            stage_metrics_summary = _evaluate_pipeline_stage_metrics(
                pipeline=pipeline,
                retrieval_agent=retrieval_agent,
                claims=claims_for_stage_metrics,
                gold_labels=gold_labels,
                gold_specs=gold_specs,
                top_k=stage_top_k,
                show_progress=show_progress,
                num_iterations=1,
                progress_desc="[stage] benchmark",
            )
    finally:
        pipeline.close()

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "claims_path": str(CLAIMS_MD_PATH),
        "gold_labels_path": str(args.gold_labels),
        "query_terms_mode": query_terms_mode,
        "num_iterations": num_iterations,
        "variance_penalty": variance_penalty,
        "include_stage_metrics": include_stage_metrics,
        "stage_top_k": stage_top_k,
        "stage_metrics_domains": sorted(stage_metrics_domains),
        "stage_metrics_claim_ids": sorted(stage_metrics_claim_ids),
        "robust_tuning_policy": dict(ROBUST_TUNING_POLICY),
        "supervised_requested": not args.unsupervised,
        "supervised_effective": supervised,
        "claims_counts": {
            "parsed_total": len(covered_claims),
            "non_covered_excluded": non_covered_count,
            "covered_used": len(covered_claims),
            "labeled_claims": len(gold_labels),
            "structured_label_specs": len(gold_specs),
            "claims_with_must_have": sum(1 for s in gold_specs.values() if s.must_have),
            "claims_with_confusers": sum(1 for s in gold_specs.values() if s.confusers),
            "by_domain": {k: len(v) for k, v in by_domain.items()},
        },
        "original_params": original_params,
        "tuning_summary": tuning_summary,
        "benchmark_summary": benchmark_summary,
        "pipeline_stage_metrics_summary": stage_metrics_summary,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"retrieval_tuning_{ts}.json"
    latest_path = OUTPUT_DIR / "retrieval_tuning_latest.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    latest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Tuning summary ===")
    for domain in ("penale", "civile", "amministrativo"):
        item = tuning_summary[domain]
        print(
            f"- {domain}: status={item['status']}, claims={item['claims_count']}, "
            f"labeled={item['labeled_claims_count']}, "
            f"mode={item['tuning_mode']}, best_score={item['best_score']}, "
            f"mean={item.get('best_score_mean')}, std={item.get('best_score_std')}"
        )
        print(f"  best_params={item['best_params']}")

    print("\n=== Benchmark summary ===")
    for domain in ("penale", "civile", "amministrativo", "misto"):
        item = benchmark_summary[domain]
        print(
            f"- {domain}: status={item['status']}, claims={item['claims_count']}, "
            f"labeled={item['labeled_claims_count']}, "
            f"proxy={item['mean_proxy_score']}, "
            f"proxy_std~={item.get('mean_proxy_score_std_across_runs')}, "
            f"hit@5={item.get('mean_hit_at_5')}, hit@10={item.get('mean_hit_at_10')}, "
            f"mrr={item.get('mean_mrr')}, ndcg@10={item.get('mean_ndcg_at_10')}"
        )

    if include_stage_metrics:
        print("\n=== Pipeline Stage Metrics (raw@K -> +CITES -> post filters) ===")
        for domain in ("penale", "civile", "amministrativo", "misto"):
            item = stage_metrics_summary.get(domain, {})
            print(
                f"- {domain}: status={item.get('status')}, claims={item.get('claims_count')}, "
                f"raw_mh_recall@{stage_top_k}={item.get('mean_raw_must_have_recall_at_k')}, "
                f"cites_mh_recall@{stage_top_k}={item.get('mean_cites_must_have_recall_at_k')}, "
                f"post_app_mh_recall={item.get('mean_post_app_must_have_recall')}, "
                f"rel_keep={item.get('mean_rel_filter_keep_ratio')}, "
                f"app_keep_over_rel={item.get('mean_app_filter_keep_over_rel_ratio')}"
            )

    print("\nSaved report:")
    print(f"- {out_path}")
    print(f"- {latest_path}")


if __name__ == "__main__":
    main()
