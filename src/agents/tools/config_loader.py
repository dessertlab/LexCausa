"""
Utilities for loading and working with the causal configuration defined in
`config_taxonomy.json`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CONFIG_PATH = Path(__file__).parent / "config_taxonomy.json"


@lru_cache()
def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the causal configuration once and cache the result.

    Args:
        path: Optional override path for the configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    cfg_path = path or CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def causal_types_by_id(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Index causal types by id."""
    return {ct["id"]: ct for ct in cfg.get("causal_types", []) if ct.get("id")}


def theories_by_id(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Index theories by id."""
    return {th["id"]: th for th in cfg.get("theories", []) if th.get("id")}


def default_mapping_by_causal(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Index default mapping entries by causal_type."""
    return {
        dm["causal_type"]: dm
        for dm in cfg.get("default_mapping", [])
        if dm.get("causal_type")
    }


def claim_classifier_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return claim-classifier config section from taxonomy."""
    config = cfg or load_config()
    section = config.get("claim_classifier", {})
    return section if isinstance(section, dict) else {}


def claim_classifier_categories(
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return configured claim-classifier category catalog."""
    section = claim_classifier_config(cfg)
    categories = section.get("categories", [])
    return categories if isinstance(categories, list) else []


def claim_classifier_few_shots(
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return configured few-shot examples for claim classifier."""
    section = claim_classifier_config(cfg)
    examples = section.get("few_shot_examples", [])
    return examples if isinstance(examples, list) else []


def claim_classifier_max_categories(cfg: Optional[Dict[str, Any]] = None) -> int:
    """Return max number of categories to keep from classifier output."""
    section = claim_classifier_config(cfg)
    value = section.get("max_categories", 3)
    try:
        parsed = int(value)
    except Exception:
        return 3
    return parsed if parsed > 0 else 3


def anchor_norms_for(
    causal_type_id: str, cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, List[Dict[str, str]]]:
    """Return anchor norms (core/accessory) for the given causal type id."""
    config = cfg or load_config()
    ct = causal_types_by_id(config).get(causal_type_id, {})
    return ct.get("anchor_norms", {"core_norms": [], "accessory_norms": []})


def principle_tests_for(
    causal_type_id: str, cfg: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    """Return principle tests for the given causal type id."""
    config = cfg or load_config()
    ct = causal_types_by_id(config).get(causal_type_id, {})
    return ct.get("principle_tests", [])


def applicable_theories_for(
    causal_type_id: str, cfg: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Return theory ids that apply to the given causal type id."""
    config = cfg or load_config()
    applicable: List[str] = []
    for theory in config.get("theories", []):
        if causal_type_id in theory.get("applicable_causal_types", []):
            applicable.append(theory["id"])
    return applicable


def pick_default_theory(
    causal_type_id: str, cfg: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Pick the default theory for a causal type based on default_mapping.

    Returns:
        theory_id if found, else None.
    """
    config = cfg or load_config()
    mapping = default_mapping_by_causal(config).get(causal_type_id)
    if mapping:
        return mapping.get("reasoner_primary_theory")
    return None


def counter_attack_pool_for(
    causal_type_id: str, cfg: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Return the counter attack pool defined for a causal type."""
    config = cfg or load_config()
    mapping = default_mapping_by_causal(config).get(causal_type_id, {})
    return mapping.get("counter_attack_pool", [])


def theory_counter_attacks(
    theory_id: str, cfg: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Return default counter attacks defined inside a theory."""
    config = cfg or load_config()
    theory = theories_by_id(config).get(theory_id, {})
    return theory.get("default_counter_attacks", [])


def counter_attack_definitions(
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, str]]:
    """Return metadata for counter attack IDs (descriptions, labels, etc.)."""
    config = cfg or load_config()
    definitions = config.get("counter_attack_definitions", {})
    return definitions if isinstance(definitions, dict) else {}


def counter_attack_descriptions(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    locale: str = "en",
) -> Dict[str, str]:
    """Return ``attack_id -> description`` resolved from taxonomy metadata."""
    definitions = counter_attack_definitions(cfg)
    use_italian = locale.lower().startswith("it")
    preferred_key = "description_it" if use_italian else "description"

    descriptions: Dict[str, str] = {}
    for attack_id, meta in definitions.items():
        if isinstance(meta, str):
            descriptions[str(attack_id)] = meta.strip()
            continue
        if not isinstance(meta, dict):
            continue
        text = (
            meta.get(preferred_key)
            or meta.get("description")
            or meta.get("description_it")
            or ""
        )
        descriptions[str(attack_id)] = str(text).strip()
    return descriptions


def validate_ids(
    causal_type_id: str,
    theory_id: Optional[str],
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Validate causal_type_id and theory_id against config.

    Returns potentially corrected (causal_type_id, theory_id).
    """
    config = cfg or load_config()
    ct_index = causal_types_by_id(config)

    chosen_ct = (
        causal_type_id if causal_type_id in ct_index else next(iter(ct_index), "")
    )

    valid_theories = applicable_theories_for(chosen_ct, config)
    chosen_th = theory_id if theory_id in valid_theories else None
    if chosen_th is None:
        fallback = pick_default_theory(chosen_ct, config)
        if fallback in valid_theories:
            chosen_th = fallback
        elif valid_theories:
            chosen_th = valid_theories[0]

    return chosen_ct, chosen_th
