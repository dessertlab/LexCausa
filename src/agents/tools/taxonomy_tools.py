"""
LangChain Tools for Causality Taxonomy (config_taxonomy.json schema).

Provides tools for:
- Classifying causality type using LLM over config-defined causal_types
- Retrieving causality theory and associated norms (anchor + principle tests)
"""

import json
import sys
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .prompt_registry import render_prompt

# Cross-platform path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings  # noqa: E402
from services.groq_client import resilient_groq_call  # noqa: E402

_taxonomy_cache: Optional[dict] = None
_theory_cache: dict = {}


def get_taxonomy() -> dict:
    """Load and cache the causality taxonomy (config schema)."""
    global _taxonomy_cache
    if _taxonomy_cache is None:
        taxonomy_path = Path(__file__).parent / "config_taxonomy.json"
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            _taxonomy_cache = json.load(f)
    return _taxonomy_cache


def _causal_type_index(cfg: dict) -> dict:
    return {ct["id"]: ct for ct in cfg.get("causal_types", []) if ct.get("id")}


def _theory_index(cfg: dict) -> dict:
    return {th["id"]: th for th in cfg.get("theories", []) if th.get("id")}


def _build_classification_prompt(claim: str, cfg: dict) -> str:
    types = cfg.get("causal_types", [])
    lines = []
    for ct in types:
        lines.append(f"- {ct.get('id')}: {ct.get('name','')} [{ct.get('domain','')}]")
    options_text = "\n".join(lines)
    return render_prompt(
        "taxonomy_tools.classification",
        options_text=options_text,
        claim=claim,
    )


class ClassifyCausalityInput(BaseModel):
    """Input schema for causality classification."""

    claim: str = Field(
        description="The legal claim to analyze to determine the type of causality"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context (e.g., relevant articles already identified)",
    )


def _build_classification_messages(
    claim: str, context: Optional[str] = None
) -> list[dict]:
    """Build the message chain with few-shot examples for causality classification."""
    cfg = get_taxonomy()
    prompt = _build_classification_prompt(claim, cfg)
    return [{"role": "user", "content": prompt}]


@tool("classify_causality", args_schema=ClassifyCausalityInput)
def classify_causality_tool(claim: str, context: Optional[str] = None) -> dict:
    """
    Classify the type of causality in a legal claim using LLM.

    Analyzes the claim and determines whether it involves:
    - Material Causality: factual connection between conduct and event (criminal law)
    - Legal Causality: connection between event and compensable damage (civil law)
    - Concurrent/Supervening Causes: interaction between multiple causal factors

    Returns the causality type, associated warrant, and relevant norms from taxonomy.
    """
    messages = _build_classification_messages(claim, context)

    def _call(client, model):
        return client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=settings.classifier_temperature,
            max_tokens=settings.taxonomy_max_tokens,
        )

    response = resilient_groq_call(
        _call,
        model_order=settings.retrieval_model_fallback_order,
    )

    # Parse response
    content = response.choices[0].message.content
    raw_response = (content or "").strip()

    # Extract causality type ID
    taxonomy = get_taxonomy()
    ct_index = _causal_type_index(taxonomy)
    causality_id = None
    for type_id in ct_index.keys():
        if type_id in raw_response:
            causality_id = type_id
            break

    if causality_id is None:
        causality_id = next(iter(ct_index.keys()), "")

    ct_entry = ct_index.get(causality_id, {})
    return {
        "causality_type": ct_entry.get("id", ""),
        "causality_id": ct_entry.get("id", ""),
        "anchor_norms": ct_entry.get("anchor_norms", {}),
        "principle_tests": ct_entry.get("principle_tests", []),
    }


class GetCausalityTheoryInput(BaseModel):
    """Input schema for getting causality theory."""

    causality_type: str = Field(description="causal_type_id from config_taxonomy.json")
    claim: Optional[str] = Field(
        default=None,
        description="Legal claim text to softly filter taxonomy norms for relevance.",
    )


@tool("get_causality_theory", args_schema=GetCausalityTheoryInput)
def get_causality_theory_tool(
    causality_type: str,
    claim: Optional[str] = None,
) -> dict:
    """
    Retrieve the complete theory associated with a causality type.

    Given a causality classification, returns:
    - Anchor norms (core/accessory)
    - Principle tests
    - Theory info (default_counter_attacks if applicable)

    Use this function to build counter-arguments based on causal theory.
    """
    taxonomy = get_taxonomy()

    normalized_type = causality_type.strip()
    cache_key = (normalized_type.lower(), (claim or "").strip())
    if cache_key in _theory_cache:
        return _theory_cache[cache_key]

    ct_index = _causal_type_index(taxonomy)
    th_index = _theory_index(taxonomy)

    ct_entry = ct_index.get(normalized_type) or ct_index.get(normalized_type.upper())
    if not ct_entry:
        return {
            "error": f"causal_type_id '{causality_type}' not found in taxonomy",
            "available_types": list(ct_index.keys()),
        }

    # collect principle_tests and anchor_norms
    core_full = (ct_entry.get("anchor_norms") or {}).get("core_norms", [])
    access_full = (ct_entry.get("anchor_norms") or {}).get("accessory_norms", [])
    principle_tests = ct_entry.get("principle_tests", [])

    result = {
        "causal_type_id": ct_entry.get("id"),
        "name": ct_entry.get("name", ""),
        "anchor_norms": ct_entry.get("anchor_norms", {}),
        "principle_tests": principle_tests,
        "norme_core": core_full,
        "norme_accessorie": access_full,
    }

    # attach applicable theories default attacks
    applicable_theories = [
        th
        for th in th_index.values()
        if ct_entry.get("id") in th.get("applicable_causal_types", [])
    ]
    if applicable_theories:
        result["applicable_theories"] = [
            {
                "id": th.get("id"),
                "name": th.get("name"),
                "default_counter_attacks": th.get("default_counter_attacks", []),
            }
            for th in applicable_theories
        ]

    if not claim:
        _theory_cache[cache_key] = result
        return result

    # Relevance filtering against the claim (soft keep-by-default)
    core_rel = _filter_norms_for_claim(core_full, claim)
    access_rel = _filter_norms_for_claim(access_full, claim)
    result["norme_core_rilevanti"] = core_rel
    result["norme_accessorie_rilevanti"] = access_rel

    _theory_cache[cache_key] = result
    return result


def _filter_norms_for_claim(norms: list[dict], claim: str) -> list[dict]:
    """Soft-filter taxonomy norms for claim relevance (default keep)."""
    if not norms:
        return norms

    relevant: list[dict] = []

    for norm in norms:
        # Support both old keys (riferimento/nota) and new keys (ref/role)
        ref = norm.get("ref") or norm.get("riferimento", "N/A")
        role = norm.get("role") or norm.get("nota", "")

        prompt = render_prompt(
            "taxonomy_tools.filter_norm",
            claim=claim,
            ref=ref,
            role=role,
        )

        try:

            def _call(client, model):
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.classifier_temperature,
                    max_tokens=settings.taxonomy_filter_max_tokens,
                )

            resp = resilient_groq_call(
                _call,
                model_order=settings.retrieval_model_fallback_order,
            )

            answer = (resp.choices[0].message.content or "").strip().upper()
        except Exception:
            answer = "YES"

        token = answer.split()[0] if answer else ""
        keep = token != "NO" and (
            token == "YES" or "YES" in answer or "NO" not in answer
        )

        if keep:
            relevant.append(norm)

    return relevant
