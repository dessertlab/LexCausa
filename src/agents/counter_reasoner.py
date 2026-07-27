"""
LexCausa Counter-Reasoner Agent.

Generates counter-arguments on the same claim, targeting the
Reasoner's conclusion (required input).
Receives the claim + causal_type_id/theory_id from the Router and the
pre-retrieved relevant sources.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage

from .aspic_formatter import AspicFormatter
from .base import AgentConfig, BaseAgent
from .citation_utils import (
    extract_article_mentions,
    format_article_citation,
    infer_source_hint,
    normalize_article_id,
)
from .router import RoutingDecision
from .tools import config_loader
from .tools.neo4j_tools import get_legal_search_pipeline, get_statute_by_article_tool
from .tools.prompt_registry import get_response_language, render_prompt
from .tools.taxonomy_tools import get_causality_theory_tool

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings  # noqa: E402
from services.groq_client import (  # noqa: E402
    RequestTooLargeError,
    shrink_max_tokens_progressive,
)
from services.pipeline_control import PipelineCancelled  # noqa: E402


@dataclass
class CounterArgument:
    """Structured representation of a counter-argument."""

    premise: str = ""
    norm: str = ""
    link: str = ""
    conclusion: str = ""


@dataclass
class CounterReasonerOutput:
    """Structured output from the Counter-Reasoner."""

    claim: str
    causal_type_id: str
    theory_id: str
    selected_attack_id: str = ""
    selected_attack_ids: List[str] = field(default_factory=list)
    reasoner_causality: dict = field(default_factory=dict)  # compatibility alias
    relevant_statutes: List[dict] = field(default_factory=list)
    relevant_precedents: List[dict] = field(default_factory=list)
    counter_arguments: List[CounterArgument] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    raw_response: str = ""
    aspic_ir: dict = field(default_factory=dict)
    abstained: bool = False
    abstention_reason: str = ""
    reasoner_conclusion_context: str = ""

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "causal_type_id": self.causal_type_id,
            "theory_id": self.theory_id,
            "selected_attack_id": self.selected_attack_id,
            "selected_attack_ids": self.selected_attack_ids,
            "reasoner_causality": self.reasoner_causality,
            "statutes": self.relevant_statutes,
            "precedents": self.relevant_precedents,
            "counter_arguments": [
                {
                    "premise": arg.premise,
                    "norm": arg.norm,
                    "link": arg.link,
                    "conclusion": arg.conclusion,
                }
                for arg in self.counter_arguments
            ],
            "reasoning_chain": self.reasoning_chain,
            "raw_response": self.raw_response,
            "aspic_ir": self.aspic_ir,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "reasoner_conclusion_context": self.reasoner_conclusion_context,
        }


_DEFAULT_ATTACK_DESCRIPTION_EN = "Counter-argument to weaken the primary legal thesis."
_DEFAULT_ATTACK_DESCRIPTION_IT = (
    "le norme citate indeboliscono la tesi giuridica primaria"
)


class CounterReasoner(BaseAgent):
    """
    Legal Counter-Reasoner Agent.

    Generates counter-arguments against the primary legal thesis,
    using the Reasoner's conclusion as mandatory target.
    Selects multiple attacks from config (counter_attack_pool) based on causal_type_id/theory_id.

    Flow:
    1. api_server pre-retrieves statutes and precedents
    2. CounterReasoner.run() receives the Router decision + pre-retrieved knowledge
    3. Planner/executor prompts build counter-arguments using the retrieved relevant sources
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Counter-Reasoner agent."""
        super().__init__(config)
        self._config = config_loader.load_config()
        self._attack_descriptions_en = config_loader.counter_attack_descriptions(
            self._config,
            locale="en",
        )
        self._attack_descriptions_it = config_loader.counter_attack_descriptions(
            self._config,
            locale="it",
        )
        self._max_plan_retries = 3
        self._max_step_rewrites = 3
        self._planner_max_tokens_cap = int(settings.counter_planner_max_tokens_cap)
        self._planner_min_tokens = int(settings.counter_planner_min_tokens)
        self._support_step_max_tokens_cap = int(
            settings.counter_support_step_max_tokens_cap
        )
        self._support_step_min_tokens = int(settings.counter_support_step_min_tokens)
        self._new_facts_check_cache: Dict[tuple[str, str], tuple[bool, str]] = {}
        self._counter_fact_lock_cache: Dict[tuple[str, str], tuple[bool, str]] = {}
        self._target_map_cache: Dict[tuple[str, str], dict] = {}
        self._conclusion_points_cache: Dict[tuple[str, str], dict] = {}
        self._attack_safety_cache: Dict[
            tuple[str, str, tuple[str, ...]], tuple[List[str], Dict[str, str]]
        ] = {}
        self._attack_compat_cache: Dict[tuple[str, str, str], tuple[bool, str]] = {}
        self._attack_precondition_cache: Dict[tuple[str, str, str], str] = {}
        self._plan_target_alignment_cache: Dict[
            tuple[str, str, str, str, str], tuple[bool, str]
        ] = {}
        self._hard_failure_threshold = 3

    def _bounded_max_tokens(self, cap: int) -> int:
        """Cap per-call max tokens against the active agent configuration."""
        base_max_tokens = int(
            getattr(self.config, "max_tokens", settings.llm_max_tokens)
            or settings.llm_max_tokens
        )
        if base_max_tokens <= 0:
            base_max_tokens = settings.llm_max_tokens
        return max(1, min(base_max_tokens, int(cap)))

    @staticmethod
    def _is_request_too_large_error(exc: Exception) -> bool:
        """Detect provider-side oversized-request failures."""
        if isinstance(exc, RequestTooLargeError):
            return True
        message = str(exc or "").lower()
        if "error code: 429" in message or "rate limit reached" in message:
            return False
        return "request too large" in message

    @staticmethod
    def _extract_json_object_payload(raw_text: str) -> str:
        """Best-effort extraction of a JSON object payload from model text."""
        payload = (
            str(raw_text or "")
            .strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        if not payload.startswith("{"):
            match = re.search(r"\{[\s\S]*\}", payload)
            if match:
                payload = match.group(0)
        return payload

    def _invoke_json_object(
        self,
        *,
        prompt: str,
        max_tokens: int,
        log_label: str,
    ) -> Dict[str, object]:
        """Invoke LLM expecting a JSON object, with response_format fallback.

        Parse failures (truncated/malformed payloads, e.g. providers that do
        not enforce json mode server-side, like DeepInfra) are retried once: a
        single bad generation must not silently disable the extraction for the
        whole run (decomposition/target-map fallbacks are cached per claim).
        """
        use_json_mode = True
        last_exc: Optional[Exception] = None
        parse_retries = 0
        attempts = 0
        while attempts < 4:
            attempts += 1
            invoke_kwargs: Dict[str, object] = {
                "max_tokens": max(1, int(max_tokens)),
                **self._low_reasoning_effort_kwargs(),
            }
            if use_json_mode:
                invoke_kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self._resilient_llm_invoke(
                    [HumanMessage(content=prompt)],
                    **invoke_kwargs,
                )
                raw = self._extract_json_object_payload(resp.content or "")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("JSON payload is not an object")
                return parsed
            except Exception as exc:
                last_exc = exc
                if use_json_mode and self._is_response_format_error(str(exc)):
                    use_json_mode = False
                    self._log(
                        f"⚠️ {log_label} JSON mode not accepted; retrying without response_format",
                        "warning",
                    )
                    continue
                if (
                    isinstance(exc, (json.JSONDecodeError, ValueError))
                    and parse_retries < 1
                ):
                    parse_retries += 1
                    self._log(
                        f"⚠️ {log_label}: invalid JSON payload; retrying once ({exc})",
                        "warning",
                    )
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"{log_label} JSON invocation failed")

    def _known_attack_ids(self) -> List[str]:
        """Return all known attack IDs from taxonomy metadata."""
        ids: List[str] = []

        def _append_if_new(values: List[str]) -> None:
            for value in values:
                attack_id = str(value).strip()
                if attack_id and attack_id not in ids:
                    ids.append(attack_id)

        _append_if_new(list(self._attack_descriptions_en.keys()))
        _append_if_new(list(self._attack_descriptions_it.keys()))
        for mapping in self._config.get("default_mapping", []):
            _append_if_new(mapping.get("counter_attack_pool", []) or [])
        for theory in self._config.get("theories", []):
            _append_if_new(theory.get("default_counter_attacks", []) or [])
        for causal_type in self._config.get("causal_types", []):
            _append_if_new(causal_type.get("counter_attack_catalog", []) or [])
        return ids

    def _attack_description(
        self,
        attack_id: str,
        *,
        locale: str = "en",
        default: str = "",
    ) -> str:
        """Resolve attack description from taxonomy metadata with fallback."""
        if locale.lower().startswith("it"):
            return (
                self._attack_descriptions_it.get(attack_id)
                or self._attack_descriptions_en.get(attack_id)
                or default
            )
        return (
            self._attack_descriptions_en.get(attack_id)
            or self._attack_descriptions_it.get(attack_id)
            or default
        )

    @staticmethod
    def _claim_fact_anchors(claim: str, max_items: int = 8) -> List[str]:
        """Extract compact factual anchors from claim text for prompt grounding."""
        text = re.sub(r"\s+", " ", (claim or "").strip())
        if not text:
            return []
        chunks = re.split(r"(?<=[.;])\s+|,\s+(?=[A-Za-zÀ-ÿ])", text)
        anchors: List[str] = []
        seen = set()
        for chunk in chunks:
            c = chunk.strip(" .;:-")
            if len(c) < 18:
                continue
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            anchors.append(c)
            if len(anchors) >= max_items:
                break
        return anchors

    def _claim_fact_anchors_text(self, claim: str, max_items: int = 8) -> str:
        anchors = self._claim_fact_anchors(claim, max_items=max_items)
        if not anchors:
            return "- none"
        return "\n".join(f"- {a}" for a in anchors)

    def _evaluate_attack_precondition(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        precondition: str,
    ) -> str:
        """Evaluate one attack precondition with conservative fail-open behavior."""
        claim_text = re.sub(r"\s+", " ", (claim or "").strip())[
            : settings.truncation_counter_claim
        ]
        reasoner_text = re.sub(r"\s+", " ", (reasoner_conclusion or "").strip())[
            : settings.truncation_counter_reasoner_conclusion
        ]
        condition = re.sub(r"\s+", " ", (precondition or "").strip())[
            : settings.truncation_counter_precondition
        ]
        if not condition:
            return "UNCLEAR"

        cache_key = (claim_text, reasoner_text, condition)
        cached = self._attack_precondition_cache.get(cache_key)
        if cached is not None:
            return cached

        status = "UNCLEAR"
        prompt = render_prompt(
            "counter_reasoner.attack_precondition_check",
            claim=claim_text,
            reasoner_conclusion=reasoner_text,
            precondition=condition,
        )
        try:
            resp = self._resilient_llm_invoke(
                [HumanMessage(content=prompt)],
                max_tokens=self._ancillary_max_tokens(),
            )
            answer = (resp.content or "").strip().upper()
            if "UNSATISFIED" in answer:
                status = "UNSATISFIED"
            elif "SATISFIED" in answer:
                status = "SATISFIED"
        except Exception as exc:
            self._log(
                f"Counter attack precondition check failed (fallback UNCLEAR): {exc}",
                "warning",
            )
            status = "UNCLEAR"

        self._attack_precondition_cache[cache_key] = status
        return status

    def _attack_preconditions_satisfied(
        self,
        *,
        attack_id: str,
        claim: str,
        reasoner_conclusion: str,
    ) -> tuple[bool, str]:
        """Check optional attack preconditions from taxonomy metadata."""
        meta = self._attack_definition_meta(attack_id)
        preconditions = meta.get("preconditions", {})
        if not isinstance(preconditions, dict):
            return True, ""

        requires_any = preconditions.get("requires_any", []) or []
        requires_all = preconditions.get("requires_all", []) or []
        requires_any = [str(x).strip() for x in requires_any if str(x).strip()]
        requires_all = [str(x).strip() for x in requires_all if str(x).strip()]

        if requires_any:
            any_supported = False
            for condition in requires_any:
                status = self._evaluate_attack_precondition(
                    claim=claim,
                    reasoner_conclusion=reasoner_conclusion,
                    precondition=condition,
                )
                if status in {"SATISFIED", "UNCLEAR"}:
                    any_supported = True
                    break
            if not any_supported:
                return False, "attack precondition (requires_any) unsatisfied"

        if requires_all:
            for condition in requires_all:
                status = self._evaluate_attack_precondition(
                    claim=claim,
                    reasoner_conclusion=reasoner_conclusion,
                    precondition=condition,
                )
                if status == "UNSATISFIED":
                    return False, "attack precondition (requires_all) unsatisfied"

        return True, ""

    def _is_attack_compatible_with_claim(
        self,
        attack_id: str,
        claim: str,
        *,
        reasoner_conclusion: str = "",
    ) -> bool:
        """Semantic attack-claim compatibility filter with precondition checks."""
        attack = str(attack_id or "").strip()
        if not attack:
            return False
        claim_text = re.sub(r"\s+", " ", (claim or "").strip())[
            : settings.truncation_counter_claim
        ]
        reasoner_text = re.sub(r"\s+", " ", (reasoner_conclusion or "").strip())[
            : settings.truncation_counter_reasoner_conclusion
        ]
        if not claim_text:
            return True

        cache_key = (attack, claim_text, reasoner_text)
        cached = self._attack_compat_cache.get(cache_key)
        if cached is not None:
            return cached[0]

        precond_ok, precond_reason = self._attack_preconditions_satisfied(
            attack_id=attack,
            claim=claim_text,
            reasoner_conclusion=reasoner_text,
        )
        if not precond_ok:
            self._attack_compat_cache[cache_key] = (False, precond_reason)
            return False

        attack_desc = self._attack_description(
            attack,
            locale="en",
            default=_DEFAULT_ATTACK_DESCRIPTION_EN,
        )
        compatible = True
        compat_reason = ""
        prompt = render_prompt(
            "counter_reasoner.attack_compatibility",
            claim=claim_text,
            attack_id=attack,
            attack_desc=attack_desc,
        )
        try:
            resp = self._resilient_llm_invoke(
                [HumanMessage(content=prompt)],
                max_tokens=self._ancillary_max_tokens(),
            )
            answer = (resp.content or "").strip().upper()
            if "MISMATCH" in answer:
                compatible = False
                compat_reason = "semantic mismatch"
            elif "COMPATIBLE" in answer or "WEAK" in answer:
                compatible = True
            else:
                compatible = True
        except Exception as exc:
            self._log(
                f"Counter attack compatibility check failed for {attack} (fallback keep): {exc}",
                "warning",
            )
            compatible = True

        self._attack_compat_cache[cache_key] = (compatible, compat_reason)
        return compatible

    def _attack_definition_meta(self, attack_id: str) -> Dict[str, object]:
        """Return attack metadata from taxonomy definitions."""
        defs = self._config.get("counter_attack_definitions", {})
        meta = defs.get(attack_id, {})
        return meta if isinstance(meta, dict) else {}

    @staticmethod
    def _truncate_words(text: str, max_words: int = 22) -> str:
        words = [w for w in re.split(r"\s+", str(text or "").strip()) if w]
        if len(words) <= max_words:
            return " ".join(words)
        return " ".join(words[:max_words]).strip()

    def _limited_attack_description(self, attack_id: str) -> str:
        """Fallback LIMITED description when safety-rewriter omits one."""
        base = self._attack_description(
            attack_id,
            locale="en",
            default=_DEFAULT_ATTACK_DESCRIPTION_EN,
        )
        return self._build_operational_attack_description(
            attack_id=attack_id,
            status="LIMITED",
            raw_desc=base,
        )

    def _build_operational_attack_description(
        self,
        *,
        attack_id: str,
        status: str,
        raw_desc: str,
    ) -> str:
        """
        Build concise prompt-facing attack descriptions with explicit factual limits.
        """
        base_desc = self._truncate_words(raw_desc or "", max_words=16)
        if not base_desc:
            base_desc = self._truncate_words(
                self._attack_description(
                    attack_id,
                    locale="en",
                    default=_DEFAULT_ATTACK_DESCRIPTION_EN,
                ),
                max_words=16,
            )
        status_norm = (status or "SAFE").strip().upper()
        if status_norm == "LIMITED":
            text = (
                f"LIMITED: only narrow scope/effects using {base_desc}; "
                "do not deny explicit claim facts or add new facts."
            )
        else:
            text = (
                f"SAFE: challenge legal inference/effects via {base_desc}; "
                "keep explicit claim facts fixed, no new facts."
            )
        return self._truncate_words(text, max_words=32)

    def _adapt_attack_pool_for_claim(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        candidate_attack_ids: List[str],
    ) -> tuple[List[str], Dict[str, str]]:
        """
        Pre-classify attacks for factual safety and rephrase risky ones as LIMITATION lines.
        """
        ordered_ids = [aid for aid in dict.fromkeys(candidate_attack_ids) if aid]
        if not ordered_ids:
            return [], {}

        claim_text = re.sub(r"\s+", " ", (claim or "").strip())[
            : settings.truncation_counter_claim
        ]
        reasoner_text = re.sub(r"\s+", " ", (reasoner_conclusion or "").strip())[
            : settings.truncation_counter_reasoner_conclusion
        ]
        cache_key = (claim_text, reasoner_text, tuple(ordered_ids))
        cached = self._attack_safety_cache.get(cache_key)
        if cached is not None:
            return cached

        attack_catalog = "\n".join(
            f"- {aid}: {self._attack_description(aid, locale='en', default=_DEFAULT_ATTACK_DESCRIPTION_EN)}"
            for aid in ordered_ids
        )
        claim_facts = self._claim_fact_anchors_text(claim, max_items=10)
        prompt = render_prompt(
            "counter_reasoner.attack_safety",
            claim=claim_text,
            reasoner_conclusion=reasoner_text,
            claim_facts=claim_facts,
            attack_catalog=attack_catalog,
        )

        status_map: Dict[str, str] = {aid: "SAFE" for aid in ordered_ids}
        desc_map: Dict[str, str] = {}
        try:
            parsed = self._invoke_json_object(
                prompt=prompt,
                max_tokens=self._ancillary_max_tokens(),
                log_label="Counter attack safety preprocessing",
            )
            rows = parsed.get("attacks", []) if isinstance(parsed, dict) else []
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    aid = str(row.get("id", "")).strip()
                    if aid not in status_map:
                        continue
                    status = str(row.get("status", "SAFE")).strip().upper()
                    if status not in {"SAFE", "LIMITED", "UNSAFE"}:
                        status = "SAFE"
                    status_map[aid] = status
                    raw_desc = str(row.get("description", "")).strip()
                    if raw_desc and status in {"SAFE", "LIMITED"}:
                        desc_map[aid] = self._build_operational_attack_description(
                            attack_id=aid,
                            status=status,
                            raw_desc=raw_desc,
                        )
        except Exception as exc:
            self._log(f"Counter attack safety preprocessing failed: {exc}", "warning")
            original = (
                ordered_ids,
                {
                    aid: self._build_operational_attack_description(
                        attack_id=aid,
                        status="SAFE",
                        raw_desc=self._attack_description(
                            aid,
                            locale="en",
                            default=_DEFAULT_ATTACK_DESCRIPTION_EN,
                        ),
                    )
                    for aid in ordered_ids
                },
            )
            self._attack_safety_cache[cache_key] = original
            return original

        kept_ids: List[str] = []
        limited_count = 0
        unsafe_count = 0
        for aid in ordered_ids:
            status = status_map.get(aid, "SAFE")
            if status == "UNSAFE":
                unsafe_count += 1
                continue
            if status == "LIMITED":
                limited_count += 1
                if aid not in desc_map:
                    desc_map[aid] = self._limited_attack_description(aid)
            kept_ids.append(aid)

        if not kept_ids:
            # Fail-open to avoid collapsing the planner.
            kept_ids = list(ordered_ids)
            desc_map = {
                aid: self._limited_attack_description(aid) for aid in ordered_ids
            }
            self._log(
                "Counter attack safety marked all attacks unsafe; fallback to limited reformulations",
                "warning",
            )
        else:
            for aid in kept_ids:
                if aid not in desc_map:
                    desc_map[aid] = self._build_operational_attack_description(
                        attack_id=aid,
                        status=status_map.get(aid, "SAFE"),
                        raw_desc=self._attack_description(
                            aid,
                            locale="en",
                            default=_DEFAULT_ATTACK_DESCRIPTION_EN,
                        ),
                    )
            if limited_count or unsafe_count:
                self._log(
                    f"Counter attack safety: limited={limited_count}, dropped_unsafe={unsafe_count}",
                    "info",
                )

        result = (kept_ids, desc_map)
        self._attack_safety_cache[cache_key] = result
        return result

    def _extract_counter_target_map(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
    ) -> Dict[str, List[str]]:
        """Extract target map for planner scope control."""
        claim_text = re.sub(r"\s+", " ", (claim or "").strip())[
            : settings.truncation_counter_claim
        ]
        reasoner_text = re.sub(r"\s+", " ", (reasoner_conclusion or "").strip())[
            : settings.truncation_counter_reasoner_conclusion
        ]
        if not claim_text:
            return {
                "allowed_targets": [],
                "forbidden_assumptions": [],
                "priority_targets": [],
            }
        cache_key = (claim_text, reasoner_text)
        cached = self._target_map_cache.get(cache_key)
        if cached is not None:
            return cached
        prompt = render_prompt(
            "counter_reasoner.target_map",
            claim=claim_text,
            reasoner_conclusion=reasoner_text,
        )
        fallback: Dict[str, List[str]] = {
            "allowed_targets": [],
            "forbidden_assumptions": [],
            "priority_targets": [],
        }
        try:
            parsed = self._invoke_json_object(
                prompt=prompt,
                max_tokens=self._ancillary_max_tokens(),
                log_label="Counter target-map extraction",
            )
            if not isinstance(parsed, dict):
                parsed = {}
            for k in ("allowed_targets", "forbidden_assumptions", "priority_targets"):
                vals = parsed.get(k, [])
                if not isinstance(vals, list):
                    vals = []
                fallback[k] = [
                    str(v).strip()
                    for v in vals
                    if isinstance(v, str) and str(v).strip()
                ][:10]
        except Exception as exc:
            self._log(f"⚠️ Counter target-map extraction failed: {exc}", "warning")
        self._target_map_cache[cache_key] = fallback
        return fallback

    @staticmethod
    def _target_map_text(target_map: Dict[str, List[str]]) -> str:
        """Serialize target map into concise planner text."""
        if not isinstance(target_map, dict):
            return "- none"
        allowed = target_map.get("allowed_targets", []) or []
        forbidden = target_map.get("forbidden_assumptions", []) or []
        priority = target_map.get("priority_targets", []) or []
        parts: List[str] = []
        if allowed:
            parts.append("Allowed targets:\n" + "\n".join(f"- {x}" for x in allowed))
        if priority:
            parts.append("Priority targets:\n" + "\n".join(f"- {x}" for x in priority))
        if forbidden:
            parts.append(
                "Forbidden assumptions:\n" + "\n".join(f"- {x}" for x in forbidden)
            )
        return "\n\n".join(parts) if parts else "- none"

    def _decompose_reasoner_conclusion(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
    ) -> Dict[str, object]:
        """
        Decompose reasoner conclusion into attackable legal commitments.

        Works only on reasoner conclusion (not on reasoner chain) to keep
        counter-argument generation independent from the original path.
        """
        claim_text = re.sub(r"\s+", " ", (claim or "").strip())[
            : settings.truncation_counter_claim
        ]
        conclusion_text = re.sub(r"\s+", " ", (reasoner_conclusion or "").strip())[
            : settings.truncation_counter_conclusion
        ]
        fallback: Dict[str, object] = {
            "attack_points": [],
            "fixed_commitments": [],
        }
        if not conclusion_text:
            return fallback
        cache_key = (claim_text, conclusion_text)
        cached = self._conclusion_points_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = render_prompt(
            "counter_reasoner.decompose_conclusion",
            claim=claim_text,
            reasoner_conclusion=conclusion_text,
        )
        try:
            parsed = self._invoke_json_object(
                prompt=prompt,
                max_tokens=self._ancillary_max_tokens(),
                log_label="Counter conclusion decomposition",
            )
            if not isinstance(parsed, dict):
                parsed = {}

            points_raw = parsed.get("attack_points", [])
            fixed_raw = parsed.get("fixed_commitments", [])
            points: List[Dict[str, str]] = []
            if isinstance(points_raw, list):
                for idx, item in enumerate(points_raw, start=1):
                    if not isinstance(item, dict):
                        continue
                    pid = str(item.get("id", "")).strip().upper()
                    if not pid:
                        pid = f"P{idx}"
                    statement = str(item.get("statement", "")).strip()
                    point_type = str(item.get("point_type", "")).strip().lower()
                    attack_vector = str(item.get("attack_vector", "")).strip()
                    if not statement:
                        continue
                    points.append(
                        {
                            "id": pid[:8],
                            "statement": statement[
                                : settings.truncation_counter_attack_statement
                            ],
                            "point_type": point_type[
                                : settings.truncation_counter_novelty_key
                            ],
                            "attack_vector": attack_vector[
                                : settings.truncation_counter_attack_vector
                            ],
                        }
                    )
            fixed: List[str] = []
            if isinstance(fixed_raw, list):
                fixed = [
                    str(v).strip()[: settings.truncation_counter_attack_vector]
                    for v in fixed_raw
                    if isinstance(v, str) and str(v).strip()
                ][:8]

            if points:
                fallback["attack_points"] = points[:8]
            if fixed:
                fallback["fixed_commitments"] = fixed
        except Exception as exc:
            self._log(f"Counter conclusion decomposition failed: {exc}", "warning")

        self._conclusion_points_cache[cache_key] = fallback
        return fallback

    @staticmethod
    def _conclusion_points_text(points_map: Dict[str, object]) -> str:
        """Serialize conclusion decomposition for planner prompt."""
        if not isinstance(points_map, dict):
            return "- none"
        lines: List[str] = []
        attack_points = points_map.get("attack_points", []) or []
        fixed = points_map.get("fixed_commitments", []) or []
        if isinstance(attack_points, list) and attack_points:
            lines.append("Attackable commitments:")
            for item in attack_points:
                if not isinstance(item, dict):
                    continue
                pid = str(item.get("id", "")).strip() or "P?"
                statement = str(item.get("statement", "")).strip()
                ptype = str(item.get("point_type", "")).strip() or "generic"
                vector = str(item.get("attack_vector", "")).strip()
                if not statement:
                    continue
                base = f"- {pid} [{ptype}]: {statement}"
                if vector:
                    base += f" | attack_hint: {vector}"
                lines.append(base)
        if isinstance(fixed, list) and fixed:
            lines.append("Fixed commitments to preserve:")
            for item in fixed:
                text = str(item).strip()
                if text:
                    lines.append(f"- {text}")
        return "\n".join(lines) if lines else "- none"

    def _resilient_model_order(self) -> list[str] | None:
        """Counter fallback chain from settings (selected model first)."""
        preferred_chain = settings.counter_model_fallback_order
        selected = settings.resolve_model_name(self.config.model_name)
        order = [selected] + [m for m in preferred_chain if m != selected]
        return order

    # ------------------------------------------------------------------
    # Attack selection logic (config-driven)
    # ------------------------------------------------------------------
    def _select_attacks(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        reasoner_conclusion: str,
    ) -> AttackSelection:
        """Select 2-3 counter attacks from config pools."""
        raw_causal_pool: List[str] = config_loader.counter_attack_pool_for(
            routing_decision.causal_type_id, self._config
        )
        raw_theory_attacks: List[str] = config_loader.theory_counter_attacks(
            routing_decision.theory_id, self._config
        )

        def _compatible(aid: str) -> bool:
            return self._is_attack_compatible_with_claim(
                aid,
                claim,
                reasoner_conclusion=reasoner_conclusion,
            )

        causal_pool = [aid for aid in raw_causal_pool if _compatible(aid)]
        theory_attacks = [aid for aid in raw_theory_attacks if _compatible(aid)]
        removed_by_compat = (len(raw_causal_pool) - len(causal_pool)) + (
            len(raw_theory_attacks) - len(theory_attacks)
        )
        if removed_by_compat:
            self._log(
                "Counter attack compatibility filtered "
                f"{removed_by_compat} attack candidate(s)",
                "info",
            )
        pool: List[str] = list(causal_pool)

        if theory_attacks:
            intersection = [a for a in pool if a in theory_attacks]
            # Keep theory-consistent attacks first, but do not collapse the pool:
            # over-pruning to a tiny intersection makes the planner brittle.
            if intersection:
                ordered_pool: List[str] = []
                for aid in intersection + pool + theory_attacks:
                    if aid and aid not in ordered_pool:
                        ordered_pool.append(aid)
                pool = ordered_pool

        if not pool:
            # Fallback to theory attacks or all known attacks
            known: List[str] = []
            for aid in self._known_attack_ids():
                precond_ok, _ = self._attack_preconditions_satisfied(
                    attack_id=aid,
                    claim=claim,
                    reasoner_conclusion=reasoner_conclusion,
                )
                if precond_ok:
                    known.append(aid)
            pool = theory_attacks or known

        pool, adapted_desc_map = self._adapt_attack_pool_for_claim(
            claim=claim,
            reasoner_conclusion=reasoner_conclusion,
            candidate_attack_ids=pool,
        )
        if not pool:
            return AttackSelection(
                pool=[],
                attack_ids=[],
                descriptions={},
                causal_pool=causal_pool,
                theory_pool=theory_attacks,
            )

        selected_ids = self._pick_attacks_with_llm(
            claim=claim,
            causal_type_id=routing_decision.causal_type_id,
            theory_id=routing_decision.theory_id,
            pool=pool,
            description_overrides=adapted_desc_map,
        )
        if not selected_ids:
            fallback_count = min(3, len(pool))
            selected_ids = pool[:fallback_count]

        # Prefer 3 active attacks when available to avoid early collapse.
        if len(pool) >= 3:
            min_target = 3
        elif len(pool) >= 2:
            min_target = 2
        else:
            min_target = 1

        selected_ids = [aid for aid in dict.fromkeys(selected_ids) if aid in pool]
        if selected_ids:
            selected_ids = selected_ids[: min(3, len(selected_ids))]
            if len(selected_ids) < min_target:
                for aid in pool:
                    if aid in selected_ids:
                        continue
                    selected_ids.append(aid)
                    if len(selected_ids) >= min_target:
                        break
        else:
            selected_ids = []
            fallback_taxonomy_ids: List[str] = []
            for aid in list(causal_pool) + list(theory_attacks):
                if aid and aid not in pool and aid not in fallback_taxonomy_ids:
                    fallback_taxonomy_ids.append(aid)
            if fallback_taxonomy_ids:
                fallback_taxonomy_ids, fallback_desc_map = (
                    self._adapt_attack_pool_for_claim(
                        claim=claim,
                        reasoner_conclusion=reasoner_conclusion,
                        candidate_attack_ids=fallback_taxonomy_ids,
                    )
                )
                adapted_desc_map.update(fallback_desc_map)
                if fallback_taxonomy_ids:
                    selected_ids = fallback_taxonomy_ids[
                        : min(3, len(fallback_taxonomy_ids))
                    ]
                    self._log(
                        "Warning: primary taxonomy pool exhausted; using fallback taxonomy attacks",
                        "warning",
                    )

        description_ids = list(dict.fromkeys(pool + selected_ids))
        descriptions = {
            aid: adapted_desc_map.get(aid)
            or self._attack_description(
                aid,
                locale="en",
                default=_DEFAULT_ATTACK_DESCRIPTION_EN,
            )
            for aid in description_ids
        }
        return AttackSelection(
            pool=pool,
            attack_ids=selected_ids,
            descriptions=descriptions,
            causal_pool=causal_pool,
            theory_pool=theory_attacks,
        )

    def _select_open_attacks(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        min_attacks: int = 2,
        max_attacks: int = 3,
    ) -> AttackSelection:
        """
        Select attack strategies without taxonomy IDs (open mode).
        """
        prompt = render_prompt(
            "counter_reasoner.open_attacks",
            claim=claim,
            reasoner_conclusion=reasoner_conclusion,
            min_attacks=min_attacks,
            max_attacks=max_attacks,
        )
        attacks_raw: List[dict] = []
        try:
            parsed = self._invoke_json_object(
                prompt=prompt,
                max_tokens=self._ancillary_max_tokens(),
                log_label="Open attack generation",
            )
            attacks_field = parsed.get("attacks", [])
            if isinstance(attacks_field, list):
                attacks_raw = [a for a in attacks_field if isinstance(a, dict)]
        except Exception as exc:
            self._log(f"⚠️ Open attack generation failed: {exc}", "warning")

        normalized: List[tuple[str, str]] = []
        seen_ids = set()
        for idx, item in enumerate(attacks_raw, start=1):
            raw_id = str(item.get("id", "")).strip().lower()
            raw_desc = str(item.get("description", "")).strip()
            attack_id = re.sub(r"[^a-z0-9_]+", "_", raw_id).strip("_")
            if not attack_id:
                attack_id = f"open_attack_{idx}"
            if len(attack_id) > 40:
                attack_id = attack_id[:40].rstrip("_")
            if not raw_desc:
                continue
            if attack_id in seen_ids:
                continue
            seen_ids.add(attack_id)
            normalized.append((attack_id, raw_desc))
            if len(normalized) >= max_attacks:
                break

        if len(normalized) < min_attacks:
            fallback = [
                (
                    "open_evidence_weight",
                    "Contestare sufficienza e univocità probatoria del nesso causale senza negare i fatti espliciti.",
                ),
                (
                    "open_subjective_element",
                    "Ridimensionare il rimprovero soggettivo e il grado di colpa alla luce dell'urgenza familiare dichiarata.",
                ),
                (
                    "open_sanction_balancing",
                    "Controbilanciare aggravanti e attenuanti per limitare la portata della conclusione punitiva.",
                ),
            ]
            normalized = fallback[:max_attacks]

        normalized = normalized[:max_attacks]

        pool = [aid for aid, _ in normalized]
        descriptions = {
            aid: self._build_operational_attack_description(
                attack_id=aid,
                status="SAFE",
                raw_desc=desc,
            )
            for aid, desc in normalized
        }
        return AttackSelection(pool=pool, attack_ids=pool, descriptions=descriptions)

    def _pick_attacks_with_llm(
        self,
        claim: str,
        causal_type_id: str,
        theory_id: str,
        pool: List[str],
        description_overrides: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Use LLM to pick 2-3 suitable attack ids from pool."""
        if not pool:
            return []

        if len(pool) == 1:
            return [pool[0]]

        min_attacks = 2
        max_attacks = min(3, len(pool))
        desc_overrides = description_overrides or {}

        options_text = "\n".join(
            f"- {aid}: {desc_overrides.get(aid) or self._attack_description(aid, locale='en', default=_DEFAULT_ATTACK_DESCRIPTION_EN)}"
            for aid in pool
        )
        prompt = render_prompt(
            "counter_reasoner.pick_attacks",
            claim=claim,
            causal_type_id=causal_type_id,
            theory_id=theory_id,
            min_attacks=min_attacks,
            max_attacks=max_attacks,
            options_text=options_text,
        )
        try:
            parsed = self._invoke_json_object(
                prompt=prompt,
                max_tokens=self._ancillary_max_tokens(),
                log_label="Counter attack selection",
            )
            answer = json.dumps(parsed, ensure_ascii=False)
            attack_ids = self._clean_attack_choices(
                raw=answer,
                pool=pool,
                min_count=min_attacks,
                max_count=max_attacks,
            )
            if attack_ids:
                return attack_ids
        except Exception as e:
            self._log(f"⚠️ LLM attack selection failed: {e}", "warning")

        return pool[:max_attacks]

    def _clean_attack_choices(
        self,
        raw: str,
        pool: List[str],
        min_count: int,
        max_count: int,
    ) -> List[str]:
        """Normalize LLM output to a list of valid attack IDs."""
        pool_set = set(pool)
        payload = raw.strip().replace("```json", "").replace("```", "").strip()
        candidates: List[str] = []

        if payload:
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    attack_ids = parsed.get("attack_ids", [])
                    if isinstance(attack_ids, list):
                        candidates.extend(str(a).strip() for a in attack_ids)
            except Exception:
                pass

        if not candidates:
            lowered = payload.lower()
            for attack_id in pool:
                if attack_id.lower() in lowered:
                    candidates.append(attack_id)

        cleaned: List[str] = []
        seen = set()
        for attack_id in candidates:
            normalized = attack_id.strip()
            if normalized not in pool_set or normalized in seen:
                continue
            cleaned.append(normalized)
            seen.add(normalized)
            if len(cleaned) >= max_count:
                break

        if len(cleaned) < min_count:
            for attack_id in pool:
                if attack_id in seen:
                    continue
                cleaned.append(attack_id)
                seen.add(attack_id)
                if len(cleaned) >= min_count:
                    break

        return cleaned[:max_count]

    def _expand_with_cross_references(self, statutes: List[dict]) -> List[dict]:
        """
        Add statutes explicitly referenced inside the text of already-retrieved articles.

        Forward-only: parses each article's text for patterns like
        "art. 624", "artt. 1223" and fetches the cited article by exact
        number from the KB via get_statute_by_article_tool.
        """
        try:
            import re
        except Exception:
            return statutes

        seen = {(s.get("articolo"), s.get("source")) for s in statutes}
        extra: List[dict] = []

        pattern = re.compile(r"art(?:t)?\.?\s*(\d{2,4})", re.IGNORECASE)

        for s in statutes:
            text = s.get("testo") or ""
            source = s.get("source", "codice_civile")

            refs = set(pattern.findall(text))

            added_refs: List[str] = []
            for ref in refs:
                key = (ref, source)
                if key in seen:
                    continue
                seen.add(key)
                result = get_statute_by_article_tool.invoke(
                    {"articolo": ref, "codice": source}
                )
                if result and not result.get("error") and result.get("articolo"):
                    extra.append(result)
                    added_refs.append(ref)

            if added_refs:
                self._log(
                    f"🔗 Cross-ref from Art. {s.get('articolo')} -> {', '.join(sorted(set(added_refs)))}"
                )

        merged = statutes + extra
        deduped: List[dict] = []
        seen_final = set()
        for st in merged:
            k = (st.get("articolo"), st.get("source"))
            if k in seen_final:
                continue
            seen_final.add(k)
            deduped.append(st)
        return deduped

    def _filtered_anchor_statutes_for_types(
        self, causal_types: List[str], claim: str
    ) -> List[dict]:
        """
        Get taxonomy anchor statutes for the given causal types.

        Alignment with Reasoner:
        - taxonomy claim-relevance filter via get_causality_theory_tool(claim=...)
        - applicability filter on converted statutes (soft core / hard accessory)
        """
        statutes: List[dict] = []
        seen_refs = set()
        unique_cts = list(dict.fromkeys(ct for ct in causal_types if ct))

        for ct in unique_cts:
            try:
                theory = get_causality_theory_tool.invoke(
                    {"causality_type": ct, "claim": claim}
                )
            except Exception as e:
                self._log(f"⚠️ Failed to load theory for {ct}: {e}", "warning")
                continue

            core_rel = theory.get("norme_core_rilevanti", []) or []
            acc_rel = theory.get("norme_accessorie_rilevanti", []) or []
            core_full = theory.get("norme_core", []) or []
            acc_full = theory.get("norme_accessorie", []) or []

            if not core_rel and core_full:
                core_rel = core_full
            if not acc_rel and acc_full:
                acc_rel = acc_full

            taxonomy_norms = core_rel + acc_rel
            kept_refs = [
                n.get("ref") or n.get("riferimento")
                for n in taxonomy_norms
                if n.get("ref") or n.get("riferimento")
            ]
            kept_set = {r for r in kept_refs if r}
            discarded_refs = [
                n.get("ref") or n.get("riferimento")
                for n in (core_full + acc_full)
                if (n.get("ref") or n.get("riferimento"))
                and (n.get("ref") or n.get("riferimento")) not in kept_set
            ]
            self._log(
                f"🔎 [taxonomy] Causality {ct}: core {len(core_rel)}/{len(core_full)}, accessory {len(acc_rel)}/{len(acc_full)}"
            )
            if kept_refs:
                self._log(f"   ✔️ Kept: {', '.join(kept_refs)}")
            if discarded_refs:
                self._log(f"   ❌ Discarded: {', '.join(discarded_refs)}")

            # Align with Reasoner:
            # - taxonomy relevance (soft) from get_causality_theory_tool(claim=...)
            # - applicability: hard on accessory, soft on core
            core_pairs = [
                (n, self._norm_to_statute_dict(n))
                for n in core_rel
                if n.get("ref") or n.get("riferimento")
            ]
            acc_pairs = [
                (n, self._norm_to_statute_dict(n))
                for n in acc_rel
                if n.get("ref") or n.get("riferimento")
            ]
            core_statutes_raw = [st for _, st in core_pairs]
            acc_statutes_raw = [st for _, st in acc_pairs]
            (
                _core_applicable,
                acc_applicable,
                _core_rejected,
                _acc_rejected,
            ) = self._filter_taxonomy_anchor_statutes_by_applicability(
                claim,
                core_statutes=core_statutes_raw,
                accessory_statutes=acc_statutes_raw,
                log_prefix=f"counter/taxonomy/{ct}",
            )

            acc_keep_keys = {self._statute_identity_key(st) for st in acc_applicable}
            final_core_statutes = core_statutes_raw  # soft keep
            final_acc_statutes = [
                st
                for st in acc_statutes_raw
                if self._statute_identity_key(st) in acc_keep_keys
            ]

            for st in final_core_statutes:
                ref = str(st.get("statute_id") or "")
                if ref and ref not in seen_refs:
                    seen_refs.add(ref)
                    payload = dict(st)
                    payload["_kb_origin"] = "taxonomy_core"
                    statutes.append(payload)
            for st in final_acc_statutes:
                ref = str(st.get("statute_id") or "")
                if ref and ref not in seen_refs:
                    seen_refs.add(ref)
                    payload = dict(st)
                    payload["_kb_origin"] = "taxonomy_accessory"
                    statutes.append(payload)

        return statutes

    def run(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        pre_retrieved_statutes: List[dict],
        pre_retrieved_precedents: List[dict],
        reasoner_conclusion: str,
        enable_causality: bool = True,
        enable_planning: bool = True,
        single_call: bool = False,
        stream_callback: Optional[Callable[[dict], None]] = None,
    ) -> CounterReasonerOutput:
        """
        Execute the counter-reasoning process with pre-retrieved knowledge.

        Args:
            claim: The legal claim to counter-argue.
            routing_decision: Output of the Router with causal_type_id/theory_id.
            pre_retrieved_statutes: Already retrieved and filtered statute articles.
            pre_retrieved_precedents: Already retrieved precedents.

        Returns:
            CounterReasonerOutput with counter-arguments and reasoning chain.
        """
        self._log(f"Counter-analyzing claim: {claim[:100]}...")
        self._log(
            f"📚 Knowledge base: {len(pre_retrieved_statutes)} statutes, {len(pre_retrieved_precedents)} precedents"
        )

        if not routing_decision:
            raise ValueError("routing_decision is required")
        if enable_causality and not routing_decision.causal_type_id:
            raise ValueError(
                "routing_decision with causal_type_id/theory_id is required when causality is enabled"
            )
        if not (reasoner_conclusion or "").strip():
            raise ValueError(
                "reasoner_conclusion is required for CounterReasoner (counter must oppose the Reasoner thesis)"
            )

        use_taxonomy_mode = False
        attack_source = "open"
        allow_open_attacks = bool(enable_causality)
        if enable_causality:
            # Select counter attacks from config pools
            taxonomy_selection = self._select_attacks(
                claim,
                routing_decision,
                reasoner_conclusion,
            )
            self._log(
                f"⚔️ Selected counter attacks: "
                f"{', '.join(taxonomy_selection.attack_ids) if taxonomy_selection.attack_ids else 'N/A'} "
                f"(pool size {len(taxonomy_selection.pool)})"
            )
            if taxonomy_selection.attack_ids:
                attack_selection = taxonomy_selection
                use_taxonomy_mode = True
                attack_source = "taxonomy"
            else:
                self._log(
                    "⚠️ No taxonomy attacks available; switching to open attack mode",
                    "warning",
                )
                attack_selection = self._select_open_attacks(
                    claim=claim,
                    reasoner_conclusion=reasoner_conclusion,
                    min_attacks=2,
                    max_attacks=3,
                )
                use_taxonomy_mode = False
                attack_source = "open"
            if attack_selection.attack_ids:
                descs = [
                    f"{aid}: {attack_selection.descriptions.get(aid, 'N/A')}"
                    for aid in attack_selection.attack_ids
                ]
                self._log(f"📝 Attack descriptions: {' | '.join(descs)}")
            # Keep taxonomy anchor norms in the counter KB when taxonomy mode is active.
            if use_taxonomy_mode:
                anchor_statutes = self._filtered_anchor_statutes_for_types(
                    [routing_decision.causal_type_id]
                    + (routing_decision.additional_causal_types or []),
                    claim,
                )
                if anchor_statutes:
                    self._log(
                        f"Anchor norms added to KB (counter): {len(anchor_statutes)}",
                        "info",
                    )
                boosted_counter_statutes = self._retrieve_targeted_counter_boost(
                    claim=claim,
                    attack_ids=attack_selection.attack_ids,
                    existing_statutes=pre_retrieved_statutes,
                )
            else:
                anchor_statutes = []
                boosted_counter_statutes = []
        else:
            self._log(
                "Causality DISABLED - attack-agnostic counter mode (no attack selection)"
            )
            attack_selection = AttackSelection(
                pool=[],
                attack_ids=[],
                descriptions={},
            )
            attack_source = "none"
            anchor_statutes = []
            boosted_counter_statutes = []

        if enable_causality and not attack_selection.attack_ids:
            self._log(
                "⚠️ No counter attacks available after safety adaptation: abstaining",
                "warning",
            )
            return self._build_abstention_output(
                claim=claim,
                routing_decision=routing_decision,
                reasoner_conclusion=reasoner_conclusion,
                reason="no_attacks_after_safety_adaptation",
                relevant_statutes=pre_retrieved_statutes,
                relevant_precedents=pre_retrieved_precedents,
                attack_selection=attack_selection,
            )

        all_statutes = (
            pre_retrieved_statutes + boosted_counter_statutes + anchor_statutes
        )
        statute_origin_map: dict[tuple[str, str], set[str]] = {}
        for s in pre_retrieved_statutes:
            k = self._statute_identity_key(s)
            statute_origin_map.setdefault(k, set()).add("pre_retrieval")
        for s in boosted_counter_statutes:
            k = self._statute_identity_key(s)
            statute_origin_map.setdefault(k, set()).add("counter_second_pass")
        for s in anchor_statutes:
            k = self._statute_identity_key(s)
            statute_origin_map.setdefault(k, set()).add(
                str(s.get("_kb_origin") or "taxonomy_anchor")
            )
        # Deduplicate
        seen_keys = set()
        deduped_statutes = []
        for s in all_statutes:
            key = (s.get("articolo"), s.get("source"))
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_statutes.append(s)

        before_expand = len(deduped_statutes)
        before_expand_keys = {self._statute_identity_key(s) for s in deduped_statutes}
        deduped_statutes = self._expand_with_cross_references(deduped_statutes)
        after_expand_keys = {self._statute_identity_key(s) for s in deduped_statutes}
        for k in after_expand_keys - before_expand_keys:
            statute_origin_map.setdefault(k, set()).add("cross_ref")
        if len(deduped_statutes) > before_expand:
            self._log(
                f"➕ Added {len(deduped_statutes) - before_expand} statutes via cross-ref",
                "info",
            )
        self._log_final_statute_origins(
            deduped_statutes,
            statute_origin_map,
            label="Counter KB",
        )

        # Format knowledge base for prompt
        knowledge_base = self._format_context_for_prompt(
            deduped_statutes, pre_retrieved_precedents
        )

        allowed_statutes = [
            f"Art. {s.get('articolo')} ({self._source_short_label(s.get('source', ''))})"
            for s in deduped_statutes
        ]
        allowed_precedents = [
            p.get("title", "Untitled") for p in pre_retrieved_precedents
        ]

        # ----------------------------------------------------------
        # Single-call ablation: one LLM call produces the whole counter-chain,
        # bypassing attack selection, decomposition, planning and per-step checks.
        # ----------------------------------------------------------
        if single_call:
            return self._generate_counter_single_call(
                claim=claim,
                routing_decision=routing_decision,
                reasoner_conclusion=reasoner_conclusion,
                knowledge_base=knowledge_base,
                allowed_statutes=allowed_statutes,
                allowed_precedents=allowed_precedents,
                deduped_statutes=deduped_statutes,
                pre_retrieved_precedents=pre_retrieved_precedents,
                stream_callback=stream_callback,
            )

        # ----------------------------------------------------------
        # Execute with iterative step-by-step chain generation
        # ----------------------------------------------------------
        target_map = self._extract_counter_target_map(
            claim=claim,
            reasoner_conclusion=reasoner_conclusion,
        )
        conclusion_points_map = self._decompose_reasoner_conclusion(
            claim=claim,
            reasoner_conclusion=reasoner_conclusion,
        )
        attack_points_raw = conclusion_points_map.get("attack_points", [])
        attack_points_list = (
            attack_points_raw if isinstance(attack_points_raw, list) else []
        )
        points_count = len(attack_points_list)
        if points_count:
            self._log(
                f"Counter conclusion decomposition: {points_count} attackable commitment(s)",
                "info",
            )
        else:
            self._log(
                "Counter conclusion decomposition unavailable; planner fallback to direct conclusion targeting",
                "warning",
            )
        MAX_CHAIN_RETRIES = settings.chain_max_retries
        output = None
        attack_blacklist: set[str] = set()

        for attempt in range(1, MAX_CHAIN_RETRIES + 1):
            self._log(
                f"🔄 Counter-Reasoner generation attempt {attempt}/{MAX_CHAIN_RETRIES}"
            )
            if attack_blacklist:
                self._log(
                    "⚠️ Counter attack blacklist active: "
                    + ", ".join(sorted(attack_blacklist)),
                    "warning",
                )

            try:
                raw_output, iterative_chain, step_attack_ids_by_step = (
                    self._generate_counter_chain_iteratively(
                        claim=claim,
                        routing_decision=routing_decision,
                        attack_selection=attack_selection,
                        knowledge_base=knowledge_base,
                        allowed_statutes=allowed_statutes,
                        available_statutes=deduped_statutes,
                        allowed_precedents=allowed_precedents,
                        reasoner_conclusion=reasoner_conclusion,
                        target_map=target_map,
                        conclusion_points_map=conclusion_points_map,
                        attack_blacklist=attack_blacklist,
                        allow_open_attacks=allow_open_attacks,
                        taxonomy_mode_active=use_taxonomy_mode,
                        enable_planning=enable_planning,
                        stream_callback=stream_callback,
                    )
                )
            except Exception as gen_exc:
                self._log(
                    f"⚠️ Attempt {attempt}/{MAX_CHAIN_RETRIES}: counter planner/executor failed ({gen_exc})",
                    "warning",
                )
                if attempt == MAX_CHAIN_RETRIES:
                    return self._build_abstention_output(
                        claim=claim,
                        routing_decision=routing_decision,
                        reasoner_conclusion=reasoner_conclusion,
                        reason=f"generation_failed_after_retries: {gen_exc}",
                        relevant_statutes=deduped_statutes,
                        relevant_precedents=pre_retrieved_precedents,
                        attack_selection=attack_selection,
                    )
                continue

            resolved_attack_ids = [
                aid
                for aid in dict.fromkeys(
                    aid
                    for per_step in step_attack_ids_by_step
                    for aid in (per_step or [])
                    if aid
                )
            ] or list(attack_selection.attack_ids)
            resolved_primary_attack = (
                resolved_attack_ids[0]
                if resolved_attack_ids
                else attack_selection.attack_id
            )
            if any(aid.startswith("open_") for aid in resolved_attack_ids):
                attack_source = "open"

            # Build output
            output = CounterReasonerOutput(
                claim=claim,
                causal_type_id=routing_decision.causal_type_id,
                theory_id=routing_decision.theory_id,
                selected_attack_id=resolved_primary_attack,
                selected_attack_ids=resolved_attack_ids,
                reasoner_causality={
                    "causal_type_id": routing_decision.causal_type_id,
                    "theory_id": routing_decision.theory_id,
                },
                relevant_statutes=deduped_statutes,
                relevant_precedents=pre_retrieved_precedents,
                raw_response=raw_output,
            )

            # Use iterative chain directly; fall back to extraction if empty
            output.reasoning_chain = (
                iterative_chain
                if iterative_chain
                else self._extract_reasoning_chain(raw_output)
            )
            output.counter_arguments = self._extract_arguments(raw_output)
            output.reasoning_chain = self._sanitize_reasoning_chain(
                output.reasoning_chain, pre_retrieved_precedents
            )

            formatter = AspicFormatter(
                role="counter",
                statutes=deduped_statutes,
                precedents=pre_retrieved_precedents,
            )
            output.aspic_ir = formatter.format(
                claim=claim,
                raw_response=raw_output,
                reasoning_chain=output.reasoning_chain,
                arguments=output.counter_arguments,
                metadata={
                    "selected_attack_id": resolved_primary_attack,
                    "selected_attack_ids": resolved_attack_ids,
                    "attack_source": attack_source,
                    "selected_attack_by_step": [
                        {
                            "step": idx + 1,
                            "attack_id": (attack_ids[0] if attack_ids else ""),
                            "attack_ids": list(attack_ids),
                        }
                        for idx, attack_ids in enumerate(step_attack_ids_by_step)
                    ],
                    "conclusion_points_count": points_count,
                    "conclusion_point_ids": [
                        str(item.get("id", "")).strip()
                        for item in attack_points_list
                        if isinstance(item, dict) and str(item.get("id", "")).strip()
                    ],
                    "causal_type_id": routing_decision.causal_type_id,
                    "theory_id": routing_decision.theory_id,
                },
            )

            # Validate: ASPIC_IR must contain reasoning chain nodes (S1, S2, …)
            if self._has_valid_reasoning_chain(output.aspic_ir):
                break
            else:
                self._log(
                    f"⚠️ Attempt {attempt}/{MAX_CHAIN_RETRIES}: empty reasoning chain "
                    f"(no S* nodes in ASPIC_IR) — retrying…",
                    "warning",
                )
                if attempt == MAX_CHAIN_RETRIES:
                    self._log(
                        f"❌ Failed to generate a valid reasoning chain after "
                        f"{MAX_CHAIN_RETRIES} attempts",
                        "error",
                    )

        assert (
            output is not None
        ), "CounterReasonerOutput was never assigned"  # guard for mypy

        chain_len = (
            len(output.aspic_ir.get("reasoning_chain", [])) if output.aspic_ir else 0
        )
        self._log(
            f"✅ Generated {len(output.counter_arguments)} counter-argument(s), "
            f"{chain_len} reasoning steps",
            "success",
        )
        return output

    def _parse_single_call_chain(self, raw: str) -> tuple[list[str], str]:
        """Parse a single-call output ('STEP N: ...' lines + 'CONCLUSION: ...').

        Returns (steps, conclusion), tolerant of a leading ```-fence or prose
        before the first STEP marker. 'NO_VALID_OPPOSITION' yields no steps.
        """
        text = str(raw or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text[:4].lower() == "json":
                text = text[4:]
            text = text.strip()
        conclusion = ""
        cm = re.search(r"(?ims)^\s*(?:CONCLUSION|CONCLUSIONE)\s*:\s*(.+)\Z", text)
        body = text
        if cm:
            conclusion = re.sub(r"\s+", " ", cm.group(1)).strip()
            body = text[: cm.start()]
        steps: list[str] = []
        for m in re.finditer(
            r"(?ims)^\s*(?:STEP|PASSO)\s*\d+\s*:\s*(.*?)(?=^\s*(?:STEP|PASSO)\s*\d+\s*:|\Z)",
            body,
        ):
            t = re.sub(r"\s+", " ", m.group(1)).strip()
            if t and t.upper() != "NO_VALID_OPPOSITION":
                steps.append(t)
        return steps, conclusion

    def _generate_counter_single_call(
        self,
        *,
        claim: str,
        routing_decision: RoutingDecision,
        reasoner_conclusion: str,
        knowledge_base: str,
        allowed_statutes: List[str],
        allowed_precedents: List[str],
        deduped_statutes: List[dict],
        pre_retrieved_precedents: List[dict],
        stream_callback: Optional[Callable[[dict], None]] = None,
    ) -> CounterReasonerOutput:
        """Single-call ablation: produce the whole counter-chain in ONE LLM call.

        No attack selection, decomposition, planner or per-step guardrails; the
        model emits a numbered counter-chain and conclusion, parsed into steps.
        Abstains when the model emits no opposing step.
        """
        statutes_list = (
            "\n".join(f"- {a}" for a in allowed_statutes) or "- No statutes available"
        )
        precedents_list = (
            "\n".join(f"- {p}" for p in allowed_precedents)
            or "- No precedents available"
        )
        prompt = render_prompt(
            "counter_reasoner.single_call",
            claim=claim,
            reasoner_conclusion=reasoner_conclusion or "(no conclusion provided)",
            routing_domain=routing_decision.domain,
            knowledge_base=knowledge_base,
            statutes_list=statutes_list,
            precedents_list=precedents_list,
            min_steps=settings.chain_min_steps,
            max_steps=settings.chain_max_steps,
        )
        resp = self._resilient_llm_invoke(
            [HumanMessage(content=prompt)],
            max_tokens=settings.llm_max_tokens,
            stream_callback=stream_callback,
        )
        raw = (resp.content or "").strip()
        steps, _conclusion = self._parse_single_call_chain(raw)
        self._log(f"⚔️ Single-call counter: {len(steps)} step(s) parsed", "info")
        if not steps:
            return self._build_abstention_output(
                claim=claim,
                routing_decision=routing_decision,
                reasoner_conclusion=reasoner_conclusion,
                reason="single-call generation produced no opposing steps",
                relevant_statutes=deduped_statutes,
                relevant_precedents=pre_retrieved_precedents,
            )
        output = CounterReasonerOutput(
            claim=claim,
            causal_type_id=routing_decision.causal_type_id,
            theory_id=routing_decision.theory_id,
            selected_attack_id="",
            selected_attack_ids=[],
            reasoner_causality={
                "causal_type_id": routing_decision.causal_type_id,
                "theory_id": routing_decision.theory_id,
            },
            relevant_statutes=deduped_statutes,
            relevant_precedents=pre_retrieved_precedents,
            raw_response=raw,
            reasoner_conclusion_context=reasoner_conclusion,
        )
        output.reasoning_chain = self._sanitize_reasoning_chain(
            steps, pre_retrieved_precedents
        )
        output.counter_arguments = self._extract_arguments(raw)
        formatter = AspicFormatter(
            role="counter",
            statutes=deduped_statutes,
            precedents=pre_retrieved_precedents,
        )
        output.aspic_ir = formatter.format(
            claim=claim,
            raw_response=raw,
            reasoning_chain=output.reasoning_chain,
            arguments=output.counter_arguments,
            metadata={
                "selected_attack_id": "",
                "selected_attack_ids": [],
                "attack_source": "single_call",
                "causal_type_id": routing_decision.causal_type_id,
                "theory_id": routing_decision.theory_id,
            },
        )
        self._log(
            f"✅ Single-call counter: {len(output.counter_arguments)} argument(s), "
            f"{len(output.reasoning_chain)} steps",
            "success",
        )
        return output

    def _build_abstention_output(
        self,
        *,
        claim: str,
        routing_decision: RoutingDecision,
        reasoner_conclusion: str,
        reason: str,
        relevant_statutes: List[dict],
        relevant_precedents: List[dict],
        attack_selection: Optional["AttackSelection"] = None,
    ) -> CounterReasonerOutput:
        """Build a structured abstention output instead of raising hard errors."""
        raw_msg = (
            "Counter-Reasoner abstained: "
            f"{reason}. Non è stato possibile costruire una contro-catena valida "
            "senza contraddire i fatti espliciti o introdurre fatti nuovi."
        )
        return CounterReasonerOutput(
            claim=claim,
            causal_type_id=routing_decision.causal_type_id,
            theory_id=routing_decision.theory_id,
            selected_attack_id=(attack_selection.attack_id if attack_selection else ""),
            selected_attack_ids=(
                attack_selection.attack_ids if attack_selection else []
            ),
            reasoner_causality={
                "causal_type_id": routing_decision.causal_type_id,
                "theory_id": routing_decision.theory_id,
            },
            relevant_statutes=relevant_statutes,
            relevant_precedents=relevant_precedents,
            counter_arguments=[],
            reasoning_chain=[],
            raw_response=raw_msg,
            aspic_ir={
                "role": "counter",
                "reasoning_chain": [],
                "abstained": True,
                "abstention_reason": reason,
            },
            abstained=True,
            abstention_reason=reason,
            reasoner_conclusion_context=reasoner_conclusion,
        )

    def _extract_cited_articles(self, text: str) -> List[str]:
        """Extract article references cited in the text."""
        mentions = extract_article_mentions(text, require_code=True)
        articles: List[str] = [
            format_article_citation(m.article_id, m.source_hint) for m in mentions
        ]
        seen: set[str] = set()
        unique: List[str] = []
        for a in articles:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique

    @staticmethod
    def _article_result_to_dict(article) -> dict:
        """Convert ``ArticleResult`` into statute dict format used by agents."""
        return {
            "statute_id": article.statute_id,
            "articolo": article.articolo,
            "titolo": article.titolo,
            "testo": article.text,
            "libro": article.libro,
            "source": article.source,
        }

    def _estimate_counter_context_coverage(
        self,
        *,
        existing_statutes: List[dict],
        attack_ids: List[str],
    ) -> tuple[float, int]:
        """
        Heuristic estimate of how well current statutes cover selected counter attacks.

        Returns:
            (coverage_ratio, covered_attacks_count)
        """
        unique_attacks = [a for a in dict.fromkeys(attack_ids) if a]
        if not unique_attacks:
            return 1.0, 0
        if not existing_statutes:
            return 0.0, 0

        attack_lines = "\n".join(
            f"- {aid}: {self._attack_description(aid, locale='it', default=_DEFAULT_ATTACK_DESCRIPTION_IT)}"
            for aid in unique_attacks
        )

        statute_lines: List[str] = []
        for statute in existing_statutes[:18]:
            articolo = str(statute.get("articolo", "") or "")
            titolo = str(statute.get("titolo", "") or "")
            testo = re.sub(r"\s+", " ", str(statute.get("testo", "") or "")).strip()
            if len(testo) > 240:
                testo = f"{testo[:240]}..."
            statute_lines.append(f"- Art. {articolo} | {titolo} | {testo}")
        statutes_block = "\n".join(statute_lines) or "- none"

        prompt = (
            "Valuta la copertura del contesto normativo per i tipi di contro-attacco.\n\n"
            "TIPI DI ATTACCO (ID + descrizione):\n"
            f"{attack_lines}\n\n"
            "CONTESTO NORMATIVO DISPONIBILE:\n"
            f"{statutes_block}\n\n"
            "Compito:\n"
            "- indica per quali attack_id il contesto contiene almeno una norma idonea a sostenere quel tipo di attacco;\n"
            "- usa SOLO gli ID forniti.\n\n"
            'Rispondi SOLO in JSON compatto: {"covered_attack_ids":["id1","id2"]}'
        )

        try:
            data = self._invoke_json_object(
                prompt=prompt,
                max_tokens=self._ancillary_max_tokens(),
                log_label="Counter coverage estimation",
            )
            covered_raw_obj = data.get("covered_attack_ids", [])
            covered_raw = covered_raw_obj if isinstance(covered_raw_obj, list) else []
            covered_ids = [
                str(x).strip()
                for x in covered_raw
                if str(x).strip() in set(unique_attacks)
            ]
            covered_ids = list(dict.fromkeys(covered_ids))
            covered = len(covered_ids)
            return covered / len(unique_attacks), covered
        except Exception as exc:
            self._log(
                f"⚠️ Counter coverage estimation failed (fallback quantitative): {exc}",
                "warning",
            )
            ratio = min(
                1.0,
                len(existing_statutes)
                / max(1, settings.counter_second_pass_min_against_statutes),
            )
            covered = min(len(unique_attacks), int(round(ratio * len(unique_attacks))))
            return ratio, covered

    def _retrieve_targeted_counter_boost(
        self,
        claim: str,
        attack_ids: List[str],
        existing_statutes: List[dict],
        context_statutes_count: Optional[int] = None,
    ) -> List[dict]:
        """Run a second-pass retrieval guided by selected counter attacks."""
        if not settings.counter_second_pass_enabled:
            return []
        effective_context_size = (
            context_statutes_count
            if context_statutes_count is not None
            else len(existing_statutes)
        )

        coverage_ratio, covered_attacks = self._estimate_counter_context_coverage(
            existing_statutes=existing_statutes,
            attack_ids=attack_ids,
        )
        size_gate = (
            effective_context_size < settings.counter_second_pass_min_against_statutes
        )
        quality_gate = coverage_ratio < 0.50
        self._log(
            "Counter second-pass pre-check: "
            f"context_size={effective_context_size}, "
            f"attack_coverage={coverage_ratio:.2f} ({covered_attacks}/{max(1, len(set(attack_ids)))})",
            "info",
        )
        # Run second pass only when context is quantitatively small OR qualitatively
        # weak for the currently selected attacks.
        if not (size_gate or quality_gate):
            self._log(
                "Counter second-pass skipped: context already sufficient for selected attacks",
                "info",
            )
            return []

        hints: List[str] = []
        for attack_id in attack_ids:
            hint = self._attack_description(
                attack_id,
                locale="it",
                default=_DEFAULT_ATTACK_DESCRIPTION_IT,
            ).strip()
            if hint and hint not in hints:
                hints.append(hint)
        if not hints:
            hints.append("contestazione nesso causale e responsabilità")

        queries = [f"{claim}. Focus difensivo: {hint}." for hint in hints]
        queries = queries[: max(1, settings.counter_second_pass_max_queries)]

        self._log(
            "🔎 Counter second-pass retrieval attivato: "
            f"{len(queries)} query mirate da attack template",
            "info",
        )

        try:
            pipe = get_legal_search_pipeline()
        except Exception as exc:
            self._log(f"⚠️ Counter second-pass init failed: {exc}", "warning")
            return []

        try:
            classification = pipe.classifier.classify(claim)
            filters = pipe.build_search_filters(
                classification,
                settings.search_use_top_n_libri,
            )

            boosted_by_id: Dict[str, dict] = {}
            existing_ids = {
                (s.get("statute_id") or "").strip()
                for s in existing_statutes
                if (s.get("statute_id") or "").strip()
            }

            legal_context = self._extract_legal_context(claim)
            for q_idx, query_text in enumerate(queries, start=1):
                embedding = pipe.embed_text(query_text)
                articles = pipe.vector_search(
                    embedding=embedding,
                    book_filters=filters,
                    top_k=max(1, settings.counter_second_pass_top_k),
                    query_text=query_text,
                )
                article_by_id = {
                    (a.statute_id or "").strip(): a
                    for a in articles
                    if (a.statute_id or "").strip()
                }
                direct_candidates = []
                for article in articles:
                    statute_id = (article.statute_id or "").strip()
                    if (
                        not statute_id
                        or statute_id in existing_ids
                        or statute_id in boosted_by_id
                    ):
                        continue
                    payload = self._article_result_to_dict(article)
                    payload["_score"] = float(article.score)
                    direct_candidates.append(payload)

                if not direct_candidates:
                    continue

                direct_kept = self.filter_irrelevant_statutes(claim, direct_candidates)
                direct_kept = self.filter_applicable_statutes(
                    claim,
                    direct_kept,
                    legal_context,
                    cache_scope="counter_second_pass:direct",
                )

                for item in direct_kept:
                    statute_id = (item.get("statute_id") or "").strip()
                    if not statute_id:
                        continue
                    current = boosted_by_id.get(statute_id)
                    if current is None or float(item.get("_score", 0.0)) > float(
                        current.get("_score", 0.0)
                    ):
                        boosted_by_id[statute_id] = item

                seed_articles = [
                    article_by_id[(s.get("statute_id") or "").strip()]
                    for s in direct_kept
                    if (s.get("statute_id") or "").strip() in article_by_id
                ]
                cites_kept_count = 0
                if seed_articles:
                    expanded_articles = pipe.expand_with_cited_articles(seed_articles)
                    seed_ids = {
                        (a.statute_id or "").strip()
                        for a in seed_articles
                        if a.statute_id
                    }
                    cites_candidates = []
                    for article in expanded_articles:
                        statute_id = (article.statute_id or "").strip()
                        if (
                            not statute_id
                            or statute_id in seed_ids
                            or statute_id in existing_ids
                            or statute_id in boosted_by_id
                        ):
                            continue
                        payload = self._article_result_to_dict(article)
                        payload["_score"] = float(article.score)
                        cites_candidates.append(payload)
                    if cites_candidates:
                        cites_kept = self.filter_irrelevant_statutes(
                            claim, cites_candidates
                        )
                        cites_kept = self.filter_applicable_statutes(
                            claim,
                            cites_kept,
                            legal_context,
                            cache_scope="counter_second_pass:cites",
                        )
                        cites_kept_count = len(cites_kept)
                        for item in cites_kept:
                            statute_id = (item.get("statute_id") or "").strip()
                            if not statute_id:
                                continue
                            current = boosted_by_id.get(statute_id)
                            if current is None or float(
                                item.get("_score", 0.0)
                            ) > float(current.get("_score", 0.0)):
                                boosted_by_id[statute_id] = item

                self._log(
                    "📎 Counter second-pass query "
                    f"{q_idx}/{len(queries)}: direct_kept={len(direct_kept)}, "
                    f"cites_kept={cites_kept_count}",
                    "info",
                )

            boosted = sorted(
                boosted_by_id.values(),
                key=lambda x: float(x.get("_score", 0.0)),
                reverse=True,
            )
            boosted = boosted[: max(1, settings.counter_second_pass_max_additional)]
            for item in boosted:
                item.pop("_score", None)

            if not boosted:
                return []

            self._log(
                f"✅ Counter second-pass: {len(boosted)} articoli aggiuntivi kept",
                "info",
            )
            return boosted

        except Exception as exc:
            self._log(f"⚠️ Counter second-pass retrieval failed: {exc}", "warning")
            return []

    def _generate_counter_chain_iteratively(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        attack_selection: AttackSelection,
        knowledge_base: str,
        allowed_statutes: List[str],
        available_statutes: List[dict],
        allowed_precedents: List[str],
        reasoner_conclusion: str,
        target_map: Optional[Dict[str, List[str]]] = None,
        conclusion_points_map: Optional[Dict[str, object]] = None,
        attack_blacklist: Optional[set[str]] = None,
        allow_open_attacks: bool = True,
        taxonomy_mode_active: bool = False,
        enable_planning: bool = True,
        stream_callback: Optional[Callable[[dict], None]] = None,
    ) -> tuple[str, List[str], List[List[str]]]:
        """Generate counter-reasoning chain with plan -> execute -> residual replan workflow.

        When enable_planning=False (DoE ablation), the planner LLM call is skipped and
        generic synthetic plan steps are used so the executor still runs step-by-step
        but without structured plan guidance.
        """
        max_steps = settings.chain_max_steps
        min_steps = settings.chain_min_steps
        statutes_list = (
            "\n".join(f"- {a}" for a in allowed_statutes) or "- No statutes available"
        )
        precedents_list = (
            "\n".join(f"- {p}" for p in allowed_precedents)
            or "- No precedents available"
        )
        target_map = target_map or {
            "allowed_targets": [],
            "forbidden_assumptions": [],
            "priority_targets": [],
        }
        target_map_text = self._target_map_text(target_map)
        conclusion_points_map = conclusion_points_map or {
            "attack_points": [],
            "fixed_commitments": [],
        }
        conclusion_points_text = self._conclusion_points_text(conclusion_points_map)

        selected_attack_ids = [
            aid
            for aid in dict.fromkeys(
                attack_selection.attack_ids or [attack_selection.attack_id]
            )
            if aid
        ]
        blacklist = attack_blacklist if attack_blacklist is not None else set()
        selected_attack_ids = [
            aid for aid in selected_attack_ids if aid not in blacklist
        ]
        primary_pool = [
            aid
            for aid in dict.fromkeys(attack_selection.pool)
            if aid and aid not in blacklist
        ]
        causal_pool = [
            aid
            for aid in dict.fromkeys(attack_selection.causal_pool or [])
            if aid and aid not in blacklist
        ]
        theory_pool = [
            aid
            for aid in dict.fromkeys(attack_selection.theory_pool or [])
            if aid and aid not in blacklist
        ]
        causal_residual = [aid for aid in causal_pool if aid not in primary_pool]
        theory_residual = [
            aid
            for aid in theory_pool
            if aid not in primary_pool and aid not in causal_residual
        ]
        backup_attack_ids = [
            aid
            for aid in dict.fromkeys(primary_pool + causal_residual + theory_residual)
            if aid and aid not in selected_attack_ids
        ]
        attack_desc_map: Dict[str, str] = dict(attack_selection.descriptions or {})
        for attack_id in selected_attack_ids + backup_attack_ids:
            attack_desc_map.setdefault(
                attack_id,
                self._attack_description(
                    attack_id,
                    locale="en",
                    default=_DEFAULT_ATTACK_DESCRIPTION_EN,
                ),
            )
        attack_fail_count: Dict[str, int] = {
            attack_id: 0 for attack_id in selected_attack_ids + backup_attack_ids
        }
        attack_failed_goals: Dict[str, set[str]] = {
            attack_id: set() for attack_id in selected_attack_ids + backup_attack_ids
        }
        attack_failure_patterns: Dict[str, Dict[str, int]] = {
            attack_id: defaultdict(int)
            for attack_id in selected_attack_ids + backup_attack_ids
        }
        attack_local_cooldown: Dict[str, int] = {}
        hard_goal_blacklist_threshold = self._hard_failure_threshold

        def _goal_key(step: Dict[str, str]) -> str:
            novelty = str(step.get("novelty_key", "")).strip().lower()
            if novelty:
                return novelty
            goal = str(step.get("goal", "")).strip().lower()
            focus = str(step.get("focus", "")).strip().lower()
            key = re.sub(r"[^a-z0-9_]+", "_", f"{goal}_{focus}").strip("_")
            return key or "generic_goal"

        def _cooldown_blocked_ids() -> set[str]:
            return {
                aid
                for aid, rounds in attack_local_cooldown.items()
                if rounds > 0 and aid not in blacklist
            }

        def _rebuild_attack_catalog(current_ids: List[str]) -> str:
            ids = [aid for aid in current_ids if aid]
            if not ids:
                return "- no attacks available"
            return "\n".join(
                f"- {aid}: {attack_desc_map.get(aid, _DEFAULT_ATTACK_DESCRIPTION_EN)}"
                for aid in ids
            )

        attack_catalog = _rebuild_attack_catalog(selected_attack_ids)
        steps: List[str] = []
        step_summaries: List[str] = []
        used_norms: List[str] = []
        step_attacks: List[List[str]] = []
        planned_steps_count = 0
        extra_steps_count = 0
        max_extra_total = max(0, int(settings.counter_step_expansion_max_extra_total))
        allowed_statute_index = self._build_allowed_statute_index(available_statutes)
        plan_round = 0
        stalled_rounds = 0
        max_plan_rounds = max(1, self._max_plan_retries + 1)

        while planned_steps_count < min_steps and planned_steps_count < max_steps:
            plan_round += 1
            if plan_round > max_plan_rounds:
                break

            def _attack_is_active(aid: str) -> bool:
                return bool(aid and aid not in blacklist)

            selected_attack_ids = [
                aid for aid in selected_attack_ids if _attack_is_active(aid)
            ]
            backup_attack_ids = [
                aid for aid in backup_attack_ids if _attack_is_active(aid)
            ]

            if not selected_attack_ids and allow_open_attacks:
                if backup_attack_ids:
                    replacement_idx = 0
                    blocked_ids = _cooldown_blocked_ids()
                    while replacement_idx < len(backup_attack_ids):
                        candidate = backup_attack_ids[replacement_idx]
                        if candidate not in blocked_ids:
                            selected_attack_ids.append(candidate)
                            backup_attack_ids.pop(replacement_idx)
                            break
                        replacement_idx += 1
                    if not selected_attack_ids and backup_attack_ids:
                        selected_attack_ids.append(backup_attack_ids.pop(0))
                    attack_catalog = _rebuild_attack_catalog(selected_attack_ids)
                    self._log(
                        f"Warning: switching to backup attack {selected_attack_ids[0]}",
                        "warning",
                    )
                else:
                    open_selection = self._select_open_attacks(
                        claim=claim,
                        reasoner_conclusion=reasoner_conclusion,
                        min_attacks=1,
                        max_attacks=3,
                    )
                    open_ids = [
                        aid
                        for aid in dict.fromkeys(open_selection.attack_ids)
                        if aid and aid not in blacklist
                    ]
                    if not open_ids:
                        self._log(
                            "Warning: no additional attacks available after rotations",
                            "warning",
                        )
                        break
                    selected_attack_ids = open_ids
                    for aid, desc in (open_selection.descriptions or {}).items():
                        if aid and desc:
                            attack_desc_map[aid] = desc
                    for aid in selected_attack_ids:
                        attack_fail_count.setdefault(aid, 0)
                        attack_failed_goals.setdefault(aid, set())
                        attack_failure_patterns.setdefault(aid, defaultdict(int))
                    attack_catalog = _rebuild_attack_catalog(selected_attack_ids)
                    self._log(
                        "Warning: taxonomy attacks exhausted; switching to OPEN attack mode in-run",
                        "warning",
                    )

            remaining_min = max(1, min_steps - planned_steps_count)
            remaining_max = max(1, max_steps - planned_steps_count)
            planner_mode = "RESUME" if planned_steps_count else "FULL"
            if enable_planning:
                plan = self._generate_counter_plan(
                    claim=claim,
                    routing_decision=routing_decision,
                    selected_attack_ids=selected_attack_ids,
                    attack_catalog=attack_catalog,
                    reasoner_conclusion=reasoner_conclusion,
                    knowledge_base=knowledge_base,
                    statutes_list=statutes_list,
                    precedents_list=precedents_list,
                    target_map_text=target_map_text,
                    conclusion_points_text=conclusion_points_text,
                    min_steps=remaining_min,
                    max_steps=remaining_max,
                    planner_mode=planner_mode,
                    resume_from_step=planned_steps_count + 1,
                    existing_summaries=step_summaries,
                )
                plan = self._coerce_counter_plan_to_allowed_norms(
                    plan=plan,
                    allowed_statutes=available_statutes,
                )
                plan = self._prune_counter_plan_against_existing_history(
                    plan=plan,
                    previous_summaries=step_summaries,
                )
                plan = self._filter_counter_plan_by_feasibility(
                    claim=claim,
                    reasoner_conclusion=reasoner_conclusion,
                    target_map=target_map,
                    plan=plan,
                    previous_summaries=step_summaries,
                )
            else:
                # Planning ablation: skip planner LLM call, use trivial synthetic steps.
                plan = [
                    {
                        "goal": "",
                        "focus": "",
                        "expected_norm": "",
                        "step_type": "ATTACK",
                        "citation_requirement": "optional",
                        "summary": f"Counter-step {planned_steps_count + i + 1}",
                    }
                    for i in range(remaining_max)
                ]

            if not plan:
                stalled_rounds += 1
                self._log(
                    "Warning: counter planner produced only redundant residual steps; retrying residual plan",
                    "warning",
                )
                if stalled_rounds >= 2:
                    break
                continue
            if enable_planning and len(plan) < remaining_min:
                stalled_rounds += 1
                self._log(
                    "Warning: counter planner returned too few feasible steps after pruning; replanning",
                    "warning",
                )
                if stalled_rounds >= 2:
                    break
                continue

            self._log(
                f"Counter plan {'(synthetic, no-planning ablation)' if not enable_planning else 'generated'}: "
                f"{len(plan)} step(s) [round={plan_round}, mode={planner_mode}, completed={planned_steps_count}]"
            )

            planned_before_round = planned_steps_count
            round_failed = False

            for local_idx, plan_step in enumerate(plan, start=1):
                global_idx = planned_steps_count + 1
                if selected_attack_ids:
                    blocked_ids = _cooldown_blocked_ids()
                    step_attack_pool = [
                        aid
                        for aid in selected_attack_ids
                        if aid and aid not in blocked_ids
                    ] or [aid for aid in selected_attack_ids if aid]
                    if not step_attack_pool:
                        round_failed = True
                        self._log(
                            f"Warning: no attack available for planned counter-step {global_idx}; replanning",
                            "warning",
                        )
                        break
                else:
                    step_attack_pool = []

                self._log(
                    f"Generating planned counter-step {global_idx}/{max_steps}: "
                    f"{plan_step.get('goal', '')[:80]} | attacks={', '.join(step_attack_pool) if step_attack_pool else 'none'}"
                )
                (
                    step_text,
                    step_failure_reason,
                    hard_failure_count,
                    step_used_attacks,
                ) = self._generate_counter_step_from_plan(
                    claim=claim,
                    routing_decision=routing_decision,
                    attack_ids=step_attack_pool,
                    attack_desc_map=attack_desc_map,
                    reasoner_conclusion=reasoner_conclusion,
                    knowledge_base=knowledge_base,
                    statutes_list=statutes_list,
                    precedents_list=precedents_list,
                    plan=plan,
                    plan_index=local_idx,
                    plan_step=plan_step,
                    previous_steps=steps,
                    previous_summaries=step_summaries,
                    used_norms=used_norms,
                    suggested_points_text=conclusion_points_text,
                    allowed_statute_index=allowed_statute_index,
                    stream_callback=stream_callback,
                )
                if not step_text:
                    round_failed = True
                    failed_attack_ids = [
                        aid
                        for aid in dict.fromkeys(
                            step_used_attacks or step_attack_pool[:1]
                        )
                        if aid
                    ]
                    for failed_attack_id in failed_attack_ids:
                        attack_fail_count[failed_attack_id] = (
                            attack_fail_count.get(failed_attack_id, 0) + 1
                        )

                    hard_failure_detected = (
                        self._is_hard_attack_failure(step_failure_reason)
                        or hard_failure_count >= self._hard_failure_threshold
                    )
                    if hard_failure_detected:
                        goal_key = _goal_key(plan_step)
                        pattern = self._normalize_failure_pattern(step_failure_reason)
                        for failed_attack_id in failed_attack_ids:
                            goal_set = attack_failed_goals.setdefault(
                                failed_attack_id, set()
                            )
                            goal_set.add(goal_key)
                            pattern_map = attack_failure_patterns.setdefault(
                                failed_attack_id, defaultdict(int)
                            )
                            pattern_map[pattern] += 1
                            same_pattern_hits = int(pattern_map.get(pattern, 0))
                            if (
                                same_pattern_hits >= self._hard_failure_threshold
                                or len(goal_set) >= hard_goal_blacklist_threshold
                            ):
                                blacklist.add(failed_attack_id)
                                attack_local_cooldown.pop(failed_attack_id, None)
                                selected_attack_ids = [
                                    aid
                                    for aid in selected_attack_ids
                                    if aid != failed_attack_id
                                ]
                                backup_attack_ids = [
                                    aid
                                    for aid in backup_attack_ids
                                    if aid != failed_attack_id
                                ]
                                self._log(
                                    f"Warning: rotating out attack {failed_attack_id} "
                                    f"after repeated hard failures "
                                    f"(goals={len(goal_set)}, pattern={pattern}, "
                                    f"pattern_hits={same_pattern_hits}, "
                                    f"last_reason={step_failure_reason}, "
                                    f"hard_attempts={hard_failure_count})",
                                    "warning",
                                )
                            else:
                                # Temporary quarantine to avoid immediate reuse in next step/round.
                                attack_local_cooldown[failed_attack_id] = max(
                                    attack_local_cooldown.get(failed_attack_id, 0), 2
                                )
                                if failed_attack_id in selected_attack_ids:
                                    selected_attack_ids = [
                                        aid
                                        for aid in selected_attack_ids
                                        if aid != failed_attack_id
                                    ] + [failed_attack_id]
                                self._log(
                                    f"Warning: quarantining attack {failed_attack_id} "
                                    f"after hard failure (goal={goal_key}, "
                                    f"distinct_goals={len(goal_set)}; below blacklist threshold)",
                                    "warning",
                                )

                    if backup_attack_ids and len(selected_attack_ids) < 2:
                        replacement = backup_attack_ids.pop(0)
                        blocked_ids = _cooldown_blocked_ids()
                        while replacement in blocked_ids and backup_attack_ids:
                            replacement = backup_attack_ids.pop(0)
                        if (
                            replacement not in selected_attack_ids
                            and replacement not in blacklist
                        ):
                            selected_attack_ids.append(replacement)
                            self._log(
                                f"Warning: injected backup attack {replacement} into active set",
                                "warning",
                            )
                            attack_failed_goals.setdefault(replacement, set())
                            attack_failure_patterns.setdefault(
                                replacement, defaultdict(int)
                            )

                    attack_catalog = _rebuild_attack_catalog(selected_attack_ids)
                    self._log(
                        f"Planned counter step {global_idx} could not be generated; "
                        "replanning residual steps from accepted prefix "
                        f"(reason: {step_failure_reason or 'unknown'})",
                        "warning",
                    )
                    break

                applied_attack_ids = [
                    aid
                    for aid in dict.fromkeys(step_used_attacks or step_attack_pool[:1])
                    if aid
                ]
                if not applied_attack_ids and step_attack_pool:
                    applied_attack_ids = [step_attack_pool[0]]
                steps.append(step_text)
                step_attacks.append(applied_attack_ids)
                step_summaries.append(self._compact_step_summary(step_text))
                planned_steps_count += 1
                new_norms = self._extract_cited_articles(step_text)
                for norm in new_norms:
                    if norm not in used_norms:
                        used_norms.append(norm)

                prec_mentions = [
                    p for p in allowed_precedents if p.lower() in step_text.lower()
                ]
                prec_info = (
                    f" | prec: {', '.join(prec_mentions)}" if prec_mentions else ""
                )
                self._log(
                    f"Counter-step {global_idx}: {step_text[:80]}... "
                    f"| attacks: {', '.join(applied_attack_ids) if applied_attack_ids else 'none'} "
                    f"| norms: {', '.join(new_norms) if new_norms else 'none'}{prec_info}"
                )

                if (
                    taxonomy_mode_active
                    and applied_attack_ids
                    and extra_steps_count < max_extra_total
                ):
                    expansion_budget = min(
                        max(1, int(settings.counter_step_expansion_max_extra_per_step)),
                        max_extra_total - extra_steps_count,
                    )
                    if expansion_budget > 0:
                        extra_steps, extra_attack_ids = (
                            self._expand_counter_step_satellites(
                                claim=claim,
                                reasoner_conclusion=reasoner_conclusion,
                                knowledge_base=knowledge_base,
                                statutes_list=statutes_list,
                                precedents_list=precedents_list,
                                parent_step=step_text,
                                parent_attack_ids=applied_attack_ids,
                                attack_desc_map=attack_desc_map,
                                previous_steps=steps,
                                previous_summaries=step_summaries,
                                used_norms=used_norms,
                                allowed_statute_index=allowed_statute_index,
                                max_extra_steps=expansion_budget,
                            )
                        )
                        if extra_steps:
                            extra_steps_count += len(extra_steps)
                            for extra_idx, extra_step in enumerate(
                                extra_steps, start=1
                            ):
                                extra_attacks = (
                                    extra_attack_ids[extra_idx - 1]
                                    if extra_idx - 1 < len(extra_attack_ids)
                                    else []
                                )
                                steps.append(extra_step)
                                step_attacks.append(extra_attacks)
                                step_summaries.append(
                                    self._compact_step_summary(extra_step)
                                )
                                extra_norms = self._extract_cited_articles(extra_step)
                                for norm in extra_norms:
                                    if norm not in used_norms:
                                        used_norms.append(norm)
                                self._log(
                                    f"Counter-step {global_idx}.{extra_idx}: {extra_step[:80]}... "
                                    f"| attacks: {', '.join(extra_attacks) if extra_attacks else 'none'} "
                                    f"| norms: {', '.join(extra_norms) if extra_norms else 'none'}"
                                )
                            self._log(
                                f"Counter-step {global_idx}: expansion added {len(extra_steps)} satellite step(s) "
                                f"(extra_budget_used={extra_steps_count}/{max_extra_total})",
                                "info",
                            )

                if planned_steps_count >= max_steps:
                    break

            if planned_steps_count == planned_before_round:
                stalled_rounds += 1
            else:
                stalled_rounds = 0

            if attack_local_cooldown:
                for aid in list(attack_local_cooldown.keys()):
                    attack_local_cooldown[aid] = max(
                        0, int(attack_local_cooldown.get(aid, 0)) - 1
                    )
                    if attack_local_cooldown[aid] <= 0:
                        attack_local_cooldown.pop(aid, None)

            if stalled_rounds >= 2 and planned_steps_count < min_steps:
                break

            if not round_failed and planned_steps_count >= min_steps:
                break

        if planned_steps_count < min_steps:
            adaptive_min = 2 if min_steps >= 3 else 1
            if planned_steps_count >= adaptive_min:
                self._log(
                    "Warning: counter chain below chain_min_steps but accepted due low feasibility "
                    f"(generated={planned_steps_count}, configured_min={min_steps})",
                    "warning",
                )
            else:
                raise RuntimeError(
                    "Counter planner/executor produced fewer steps than chain_min_steps "
                    "after residual replanning"
                )

        self._log(
            f"Planned counter-chain complete: planned={planned_steps_count}, total={len(steps)} steps, "
            f"{len(set(used_norms))} unique norms"
        )
        return (
            self._assemble_counter_raw_response(claim, steps, step_attacks),
            steps,
            step_attacks,
        )

    def _expand_counter_step_satellites(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        knowledge_base: str,
        statutes_list: str,
        precedents_list: str,
        parent_step: str,
        parent_attack_ids: List[str],
        attack_desc_map: Dict[str, str],
        previous_steps: List[str],
        previous_summaries: List[str],
        used_norms: List[str],
        allowed_statute_index: Dict[str, set[str]],
        max_extra_steps: int,
    ) -> tuple[List[str], List[List[str]]]:
        """
        Optionally expand one compressed parent step into additional satellite steps.

        Expansion is best-effort and never blocks the parent step: if parsing or
        validation fails, the caller keeps only the original accepted step.
        """
        if not settings.counter_step_expansion_enabled:
            return [], []
        if max_extra_steps <= 0:
            return [], []

        min_attacks = max(2, int(settings.counter_step_expansion_min_attacks))
        parent_ids = [
            aid
            for aid in dict.fromkeys(parent_attack_ids)
            if aid and not str(aid).startswith("open_")
        ]
        if len(parent_ids) < min_attacks:
            return [], []

        parent_attack_catalog = "\n".join(
            f"- {aid}: {attack_desc_map.get(aid, _DEFAULT_ATTACK_DESCRIPTION_EN)}"
            for aid in parent_ids
        )
        summary_lines = (
            "\n".join(
                f"- Step {idx}: {summary}"
                for idx, summary in enumerate(previous_summaries, start=1)
            )
            if previous_summaries
            else "- none"
        )
        used_norms_text = ", ".join(used_norms) if used_norms else "none"
        reasoner_block = f"\nCONCLUSION TO OPPOSE:\n{reasoner_conclusion}\n"
        claim_facts = self._claim_fact_anchors_text(claim, max_items=10)

        prompt = render_prompt(
            "counter_reasoner.step_expansion",
            claim=claim,
            reasoner_block=reasoner_block,
            claim_facts=claim_facts,
            parent_step=parent_step,
            parent_attack_ids=", ".join(parent_ids),
            parent_attack_catalog=parent_attack_catalog,
            summary_lines=summary_lines,
            used_norms_text=used_norms_text,
            statutes_list=statutes_list,
            precedents_list=precedents_list,
            knowledge_base=knowledge_base,
            max_extra=max_extra_steps,
        )

        try:
            data = self._invoke_json_object(
                prompt=prompt,
                max_tokens=self._bounded_max_tokens(self._support_step_max_tokens_cap),
                log_label="Counter step expansion",
            )
        except Exception as exc:
            self._log(
                f"Counter-step expansion skipped (parser/invoke error): {exc}",
                "warning",
            )
            return [], []

        if not isinstance(data, dict):
            return [], []
        extra_steps_raw = data.get("extra_steps", [])
        if not isinstance(extra_steps_raw, list) or not extra_steps_raw:
            return [], []

        accepted_steps: List[str] = []
        accepted_attacks: List[List[str]] = []
        history_steps = list(previous_steps)
        history_summaries = list(previous_summaries)

        for item in extra_steps_raw[:max_extra_steps]:
            if not isinstance(item, dict):
                continue
            step_candidate_raw = str(item.get("step", "")).strip()
            if not step_candidate_raw:
                continue

            attack_hint = str(item.get("attack_id", "")).strip().lower()
            if attack_hint and attack_hint not in parent_ids:
                attack_hint = ""

            payload_for_parse = (
                f"ATTACKS_USED: {attack_hint}\nSTEP: {step_candidate_raw}"
                if attack_hint
                else f"STEP: {step_candidate_raw}"
            ).strip()
            candidate_step, candidate_attacks = self._parse_counter_step_payload(
                response=payload_for_parse,
                allowed_attack_ids=parent_ids,
                fallback_attack_ids=[attack_hint],
            )
            candidate_step = (candidate_step or "").strip()
            if not candidate_step:
                continue

            if not candidate_attacks:
                candidate_attacks = [parent_ids[0]]
            primary_attack_id = candidate_attacks[0]

            expected_norm = str(item.get("expected_norm", "N/A")).strip() or "N/A"
            citation_requirement = self._normalize_plan_citation_requirement(
                expected_norm=expected_norm,
                raw_value=item.get("citation_requirement", "optional"),
            )

            ok, reason = self._validate_counter_step_candidate(
                candidate_step=candidate_step,
                previous_steps=history_steps,
                claim=claim,
                reasoner_conclusion=reasoner_conclusion,
                expected_norm=expected_norm,
                citation_requirement=citation_requirement,
                allowed_statute_index=allowed_statute_index,
                attack_id=primary_attack_id,
                attack_desc=attack_desc_map.get(primary_attack_id, ""),
                plan_focus="counter_step_expansion",
            )
            if not ok:
                self._log(
                    f"Counter-step expansion candidate rejected ({reason})",
                    "warning",
                )
                continue

            accepted_steps.append(candidate_step)
            accepted_attacks.append(candidate_attacks)
            history_steps.append(candidate_step)
            history_summaries.append(self._compact_step_summary(candidate_step))

        return accepted_steps, accepted_attacks

    def _generate_counter_plan(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        selected_attack_ids: List[str],
        attack_catalog: str,
        reasoner_conclusion: str,
        knowledge_base: str,
        statutes_list: str,
        precedents_list: str,
        target_map_text: str,
        conclusion_points_text: str,
        min_steps: int,
        max_steps: int,
        planner_mode: str = "FULL",
        resume_from_step: int = 1,
        existing_summaries: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Generate and validate an execution plan for counter reasoning."""
        reasoner_block = f"\nReasoner conclusion to oppose:\n{reasoner_conclusion}\n"
        claim_facts = self._claim_fact_anchors_text(claim, max_items=10)
        existing_steps_text = (
            "\n".join(
                f"- Step {idx}: {summary}"
                for idx, summary in enumerate(existing_summaries or [], start=1)
            )
            if existing_summaries
            else "- none"
        )
        last_error = "planner failed"
        planner_max_tokens = self._bounded_max_tokens(self._planner_max_tokens_cap)
        planner_use_json_mode = True
        for attempt in range(1, self._max_plan_retries + 1):
            attempt_prompt = render_prompt(
                "counter_reasoner.generate_plan",
                claim=claim,
                reasoner_block=reasoner_block,
                claim_facts=claim_facts,
                routing_domain=routing_decision.domain,
                selected_attack_ids=", ".join(selected_attack_ids),
                attack_catalog=attack_catalog,
                statutes_list=statutes_list,
                precedents_list=precedents_list,
                knowledge_base=knowledge_base,
                target_map=target_map_text,
                conclusion_points=conclusion_points_text,
                min_steps=min_steps,
                max_steps=max_steps,
                planner_mode=planner_mode,
                resume_from_step=resume_from_step,
                existing_steps=existing_steps_text,
            )
            invoke_kwargs: Dict[str, object] = {
                "max_tokens": planner_max_tokens,
                # Planner emits structured JSON: minimal reasoning effort keeps
                # the CoT of reasoning models from eating the token budget
                # (same protection as the reasoner planner and _invoke_json_object).
                **self._low_reasoning_effort_kwargs(),
            }
            if planner_use_json_mode:
                invoke_kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self._resilient_llm_invoke(
                    [HumanMessage(content=attempt_prompt)],
                    **invoke_kwargs,
                )
                raw = (resp.content or "").strip()
                plan = self._parse_counter_plan(
                    raw=raw,
                    min_steps=min_steps,
                    max_steps=max_steps,
                    allowed_attack_ids=selected_attack_ids,
                )
                if plan:
                    return plan
                last_error = "parsed empty plan"
            except Exception as e:
                last_error = str(e)
                if self._is_response_format_error(last_error) and planner_use_json_mode:
                    planner_use_json_mode = False
                    self._log(
                        "⚠️ Counter planner JSON mode not accepted; retrying without response_format",
                        "warning",
                    )
                if (
                    self._is_request_too_large_error(e)
                    and planner_max_tokens > self._planner_min_tokens
                ):
                    reduced = shrink_max_tokens_progressive(
                        planner_max_tokens,
                        self._planner_min_tokens,
                    )
                    if reduced < planner_max_tokens:
                        self._log(
                            "⚠️ Counter planner request too large; reducing "
                            f"max_tokens {planner_max_tokens} -> {reduced}",
                            "warning",
                        )
                        planner_max_tokens = reduced
            self._log(
                f"⚠️ Counter planner attempt {attempt}/{self._max_plan_retries} failed: {last_error}",
                "warning",
            )
        raise RuntimeError(f"Counter planner failed: {last_error}")

    def _parse_counter_plan(
        self,
        raw: str,
        min_steps: int,
        max_steps: int,
        allowed_attack_ids: List[str],
    ) -> List[Dict[str, str]]:
        """Parse planner JSON and enforce plan quality constraints."""
        payload_text = str(raw or "").strip()
        # Some models (e.g. Llama-3.3-70B) wrap the JSON in a ```json fence or add
        # surrounding prose even in json mode (providers like DeepInfra treat
        # response_format as a hint, not a grammar). Strip the fence and, failing
        # that, fall back to the outermost {...} object before giving up
        # (mirrors the reasoner's _parse_reasoning_plan).
        if payload_text.startswith("```"):
            payload_text = payload_text.strip("`")
            if payload_text[:4].lower() == "json":
                payload_text = payload_text[4:]
            payload_text = payload_text.strip()
        try:
            data = json.loads(payload_text)
        except Exception as exc:
            start = payload_text.find("{")
            end = payload_text.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(payload_text[start : end + 1])
                except Exception:
                    raise ValueError(f"invalid planner JSON payload: {exc}") from exc
            else:
                raise ValueError(f"invalid planner JSON payload: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("planner output is not a JSON object")
        steps_raw = data.get("steps")
        if not isinstance(steps_raw, list):
            raise ValueError("planner output missing 'steps' array")
        cleaned: List[Dict[str, str]] = []
        for idx, item in enumerate(steps_raw, start=1):
            if not isinstance(item, dict):
                continue
            goal = str(item.get("goal", "")).strip()
            focus = str(item.get("focus", "")).strip()
            expected_norm = str(item.get("expected_norm", "")).strip() or "N/A"
            step_type = self._normalize_counter_step_type(item.get("step_type"))
            novelty_key = (
                re.sub(
                    r"[^a-z0-9_]+",
                    "_",
                    str(item.get("novelty_key", "")).strip().lower(),
                ).strip("_")
                or re.sub(r"[^a-z0-9_]+", "_", focus.lower()).strip("_")[
                    : settings.truncation_counter_novelty_key
                ]
                or f"counter_step_{idx}"
            )
            citation_requirement = self._normalize_plan_citation_requirement(
                expected_norm=expected_norm,
                raw_value=item.get("citation_requirement"),
            )
            attack_id = str(item.get("attack_id", "")).strip()
            target_point_id = str(item.get("target_point_id", "")).strip().upper()
            if not goal or not focus:
                continue
            if not allowed_attack_ids:
                attack_id = ""
            if attack_id and attack_id not in allowed_attack_ids:
                attack_id = ""
            if not attack_id and allowed_attack_ids:
                # Keep plan attack-aware even when planner omits attack_id.
                attack_id = allowed_attack_ids[(idx - 1) % len(allowed_attack_ids)]
            cleaned.append(
                {
                    "id": str(item.get("id", f"C{idx}")).strip() or f"C{idx}",
                    "goal": goal,
                    "focus": focus,
                    "expected_norm": expected_norm,
                    "citation_requirement": citation_requirement,
                    "attack_id": attack_id,
                    "step_type": step_type,
                    "novelty_key": novelty_key[:64],
                    "target_point_id": target_point_id[:12],
                }
            )

        if len(cleaned) < min_steps or len(cleaned) > max_steps:
            raise ValueError(
                f"invalid counter-plan length {len(cleaned)} (expected {min_steps}-{max_steps})"
            )
        novelty_keys = [step.get("novelty_key", "") for step in cleaned]
        if len(set(novelty_keys)) != len(novelty_keys):
            raise ValueError("counter planner produced duplicate novelty_key values")
        return cleaned

    @staticmethod
    def _is_response_format_error(error_text: str) -> bool:
        text = str(error_text or "").lower()
        markers = (
            "response_format",
            "json schema",
            "json_object",
            "unsupported",
            "invalid request",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _extract_allowed_counter_article_ids(allowed_statutes: List[dict]) -> set[str]:
        """Parse normalized article ids from available counter statutes."""
        ids: set[str] = set()
        for statute in allowed_statutes or []:
            normalized = normalize_article_id(str(statute.get("articolo", "")))
            if normalized:
                ids.add(normalized)
        return ids

    def _coerce_counter_plan_to_allowed_norms(
        self,
        *,
        plan: List[Dict[str, str]],
        allowed_statutes: List[dict],
    ) -> List[Dict[str, str]]:
        """
        Downgrade impossible citation requirements when planner expects norms not
        present in the current allowed counter statute set.
        """
        allowed_ids = self._extract_allowed_counter_article_ids(allowed_statutes)
        if not allowed_ids:
            return plan

        adjusted = 0
        for step in plan:
            expected_norm = str(step.get("expected_norm", "")).strip()
            if not expected_norm or expected_norm.upper() in {"N/A", "NA", "NONE", "-"}:
                continue
            mentions = extract_article_mentions(expected_norm, require_code=False)
            expected_ids = {
                normalize_article_id(m.article_id)
                for m in mentions
                if getattr(m, "article_id", None)
            }
            expected_ids = {eid for eid in expected_ids if eid}
            if not expected_ids:
                for match in re.findall(
                    r"art\.?\s*([0-9]+(?:-[a-z]+)?)", expected_norm, re.IGNORECASE
                ):
                    normalized = normalize_article_id(match)
                    if normalized:
                        expected_ids.add(normalized)
            if not expected_ids:
                continue
            if expected_ids & allowed_ids:
                continue
            step["expected_norm"] = "N/A"
            if str(step.get("citation_requirement", "")).strip().lower() == "required":
                step["citation_requirement"] = "optional"
            adjusted += 1

        if adjusted:
            self._log(
                "Warning: counter planner normalization downgraded "
                f"{adjusted} step(s) with unavailable expected_norm",
                "warning",
            )
        return plan

    def _is_counter_plan_step_feasible(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        target_map: Dict[str, List[str]],
        plan_step: Dict[str, str],
        previous_summaries: List[str],
    ) -> tuple[bool, str]:
        """
        Feasibility gate for planner steps before step text generation.

        The gate remains permissive, but removes clearly off-target or fact-unsafe
        goals to reduce futile execution/rewrite loops.
        """
        goal = str(plan_step.get("goal", "")).strip()
        focus = str(plan_step.get("focus", "")).strip()
        if not goal or not focus:
            return False, "incomplete plan step"

        candidate = f"{goal}. {focus}".strip()
        if self._is_garbage_text(candidate, min_words=8):
            return False, "degenerate plan step"
        if previous_summaries and self._is_repetitive_step(
            candidate, previous_summaries, threshold=0.45
        ):
            return False, "plan step redundant with accepted history"

        aligned, aligned_reason = self._is_counter_plan_step_target_aligned(
            claim=claim,
            reasoner_conclusion=reasoner_conclusion,
            target_map=target_map,
            plan_goal=goal,
            plan_focus=focus,
        )
        if not aligned:
            return False, aligned_reason or "plan step off target map"

        facts_ok, facts_reason = self._is_counter_step_fact_consistent_with_claim(
            claim=claim,
            candidate_step=candidate,
        )
        if not facts_ok:
            return False, f"plan fact-unsafe: {facts_reason}"

        grounded_ok, grounded_reason = self._is_counter_step_grounded_in_claim_facts(
            claim=claim,
            candidate_step=candidate,
        )
        if not grounded_ok:
            return False, f"plan adds unsupported facts: {grounded_reason}"

        return True, ""

    def _is_counter_plan_step_target_aligned(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        target_map: Dict[str, List[str]],
        plan_goal: str,
        plan_focus: str,
    ) -> tuple[bool, str]:
        """Check whether planned goal/focus stays within extracted target map."""
        goal = re.sub(r"\s+", " ", (plan_goal or "").strip())[
            : settings.truncation_counter_attack_statement
        ]
        focus = re.sub(r"\s+", " ", (plan_focus or "").strip())[
            : settings.truncation_counter_attack_statement
        ]
        if not goal or not focus:
            return False, "incomplete plan step"
        if not target_map:
            return True, ""
        has_constraints = bool(
            (target_map.get("allowed_targets", []) or [])
            or (target_map.get("forbidden_assumptions", []) or [])
            or (target_map.get("priority_targets", []) or [])
        )
        if not has_constraints:
            return True, ""

        claim_text = re.sub(r"\s+", " ", (claim or "").strip())[
            : settings.truncation_counter_claim
        ]
        reasoner_text = re.sub(r"\s+", " ", (reasoner_conclusion or "").strip())[
            : settings.truncation_counter_reasoner_conclusion
        ]
        target_map_text = self._target_map_text(target_map)
        cache_key = (
            goal.lower(),
            focus.lower(),
            claim_text,
            reasoner_text,
            target_map_text,
        )
        cached = self._plan_target_alignment_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = render_prompt(
            "counter_reasoner.plan_target_alignment",
            claim=claim_text,
            reasoner_conclusion=reasoner_text,
            target_map=target_map_text,
            plan_goal=goal,
            plan_focus=focus,
        )
        result: tuple[bool, str] = (True, "")
        try:
            resp = self._resilient_llm_invoke(
                [HumanMessage(content=prompt)],
                max_tokens=self._ancillary_max_tokens(),
            )
            answer = (resp.content or "").strip().upper()
            if "OFF_TARGET" in answer:
                result = (False, "plan step off target map")
            elif "ALIGNED" in answer or "UNCLEAR" in answer:
                result = (True, "")
            else:
                result = (True, "")
        except Exception as exc:
            self._log(
                f"Counter plan target-alignment check failed (fallback keep): {exc}",
                "warning",
            )
            result = (True, "")

        self._plan_target_alignment_cache[cache_key] = result
        return result

    @staticmethod
    def _is_counter_plan_fact_related_failure(reason: str) -> bool:
        """Identify feasibility failures tied to factual unsafety/new-facts violations."""
        text = (reason or "").strip().lower()
        if not text:
            return False
        markers = (
            "plan fact-unsafe",
            "plan adds unsupported facts",
            "contradicts explicit claim fact",
            "adds factual allegations not present in claim",
        )
        return any(marker in text for marker in markers)

    def _rewrite_counter_plan_step_for_fact_safety(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        target_map: Dict[str, List[str]],
        plan_step: Dict[str, str],
        invalid_reason: str,
    ) -> Optional[Dict[str, str]]:
        """
        Rewrite a fact-unsafe planner step before dropping it.

        Returns rewritten step dict on success, otherwise ``None``.
        """
        goal = str(plan_step.get("goal", "")).strip()
        focus = str(plan_step.get("focus", "")).strip()
        if not goal or not focus:
            return None

        prompt = render_prompt(
            "counter_reasoner.plan_feasibility_rewrite",
            claim=(claim or "").strip(),
            reasoner_conclusion=(reasoner_conclusion or "").strip(),
            target_map=self._target_map_text(target_map),
            plan_goal=goal,
            plan_focus=focus,
            expected_norm=str(plan_step.get("expected_norm", "N/A")).strip() or "N/A",
            citation_requirement=str(
                plan_step.get("citation_requirement", "optional")
            ).strip()
            or "optional",
            invalid_reason=(invalid_reason or "").strip(),
        )

        try:
            data = self._invoke_json_object(
                prompt=prompt,
                max_tokens=self._ancillary_max_tokens(),
                log_label="Counter plan-step rewrite",
            )
            if not isinstance(data, dict):
                return None

            new_goal = str(data.get("goal", "")).strip()
            new_focus = str(data.get("focus", "")).strip()
            if not new_goal or not new_focus:
                return None

            rewritten = dict(plan_step)
            rewritten["goal"] = self._truncate_words(new_goal, max_words=25)
            rewritten["focus"] = self._truncate_words(new_focus, max_words=25)
            expected_norm = (
                str(
                    data.get("expected_norm", rewritten.get("expected_norm", "N/A"))
                ).strip()
                or "N/A"
            )
            rewritten["expected_norm"] = expected_norm
            rewritten["citation_requirement"] = (
                self._normalize_plan_citation_requirement(
                    expected_norm=expected_norm,
                    raw_value=data.get(
                        "citation_requirement",
                        rewritten.get("citation_requirement", "optional"),
                    ),
                )
            )
            return rewritten
        except Exception as exc:
            self._log(
                f"Counter plan-step rewrite failed (fallback drop): {exc}",
                "warning",
            )
            return None

    def _filter_counter_plan_by_feasibility(
        self,
        *,
        claim: str,
        reasoner_conclusion: str,
        target_map: Dict[str, List[str]],
        plan: List[Dict[str, str]],
        previous_summaries: List[str],
    ) -> List[Dict[str, str]]:
        """
        Drop planner steps that are impossible under fact-lock/opposition constraints.
        """
        if not plan:
            return []
        kept: List[Dict[str, str]] = []
        dropped = 0
        rewritten_kept = 0
        drop_reasons: Dict[str, int] = defaultdict(int)
        for step in plan:
            ok, reason = self._is_counter_plan_step_feasible(
                claim=claim,
                reasoner_conclusion=reasoner_conclusion,
                target_map=target_map,
                plan_step=step,
                previous_summaries=previous_summaries,
            )
            if ok:
                kept.append(step)
                continue

            # Second chance: rewrite fact-unsafe plan steps instead of immediate drop.
            if self._is_counter_plan_fact_related_failure(reason):
                rewritten_step = self._rewrite_counter_plan_step_for_fact_safety(
                    claim=claim,
                    reasoner_conclusion=reasoner_conclusion,
                    target_map=target_map,
                    plan_step=step,
                    invalid_reason=reason,
                )
                if rewritten_step is not None:
                    ok_rewritten, rewritten_reason = (
                        self._is_counter_plan_step_feasible(
                            claim=claim,
                            reasoner_conclusion=reasoner_conclusion,
                            target_map=target_map,
                            plan_step=rewritten_step,
                            previous_summaries=previous_summaries,
                        )
                    )
                    if ok_rewritten:
                        kept.append(rewritten_step)
                        rewritten_kept += 1
                        continue
                    reason = rewritten_reason or reason

            dropped += 1
            reason_key = str(reason or "unknown").strip().lower()
            if reason_key:
                drop_reasons[reason_key] += 1
        if dropped:
            reason_text = ", ".join(
                f"{key}={count}"
                for key, count in sorted(
                    drop_reasons.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:3]
            )
            self._log(
                "Warning: counter plan feasibility gate pruned "
                f"{dropped} step(s)" + (f" ({reason_text})" if reason_text else ""),
                "warning",
            )
        if rewritten_kept:
            self._log(
                f"Counter plan feasibility rewrite salvaged {rewritten_kept} step(s)",
                "info",
            )
        return kept

    @staticmethod
    def _normalize_counter_step_type(raw_value: object) -> str:
        """Normalize counter planner step_type into a constrained enum-like value."""
        value = str(raw_value or "").strip().upper()
        allowed = {
            "TARGET_FACTS",
            "TARGET_CAUSAL_LINK",
            "TARGET_LEGAL_QUALIFICATION",
            "TARGET_ELEMENT",
            "TARGET_BALANCING",
            "TARGET_OUTCOME",
            "OTHER",
        }
        if value in allowed:
            return value
        aliases = {
            "FACTS": "TARGET_FACTS",
            "CAUSAL_LINK": "TARGET_CAUSAL_LINK",
            "QUALIFICATION": "TARGET_LEGAL_QUALIFICATION",
            "ELEMENTS": "TARGET_ELEMENT",
            "BALANCING": "TARGET_BALANCING",
            "OUTCOME": "TARGET_OUTCOME",
            "CONSEQUENCE": "TARGET_OUTCOME",
        }
        return aliases.get(value, "OTHER")

    @staticmethod
    def _normalize_plan_citation_requirement(
        *,
        expected_norm: str,
        raw_value: object,
    ) -> str:
        """Normalize planner citation policy with backward-compatible defaults."""
        value = str(raw_value or "").strip().lower()
        aliases = {
            "required": "required",
            "must": "required",
            "mandatory": "required",
            "optional": "optional",
            "if_possible": "optional",
            "when_possible": "optional",
            "none": "none",
            "no": "none",
        }
        normalized = aliases.get(value)
        if normalized:
            return normalized

        expected = str(expected_norm or "").strip().upper()
        if expected and expected not in {"N/A", "NA", "NONE", "-"}:
            return "required"
        return "optional"

    @staticmethod
    def _step_requires_citation(
        *, expected_norm: str, citation_requirement: str
    ) -> bool:
        """Decide citation requirement from planner metadata only."""
        requirement = str(citation_requirement or "").strip().lower()
        if requirement == "required":
            return True
        if requirement in {"optional", "none"}:
            return False
        expected = str(expected_norm or "").strip().upper()
        return bool(expected and expected not in {"N/A", "NA", "NONE", "-"})

    def _prune_counter_plan_against_existing_history(
        self,
        *,
        plan: List[Dict[str, str]],
        previous_summaries: List[str],
    ) -> List[Dict[str, str]]:
        """Remove residual counter-plan steps already covered by accepted prefix."""
        if not plan or not previous_summaries:
            return plan
        pruned: List[Dict[str, str]] = []
        removed = 0
        for step in plan:
            candidate = f"{step.get('goal', '')}. {step.get('focus', '')}".strip()
            if not candidate:
                removed += 1
                continue
            if self._is_repetitive_step(candidate, previous_summaries, threshold=0.45):
                removed += 1
                continue
            redundant = False
            for summary in previous_summaries[-4:]:
                rel = self._nli_relation(
                    target_text=summary,
                    attacker_text=candidate,
                    actor_label="CounterPlanner",
                )
                if rel == "entailment":
                    rel_sym = self._nli_relation(
                        target_text=candidate,
                        attacker_text=summary,
                        actor_label="CounterPlanner",
                    )
                    if rel_sym in {"entailment", "neutral"}:
                        redundant = True
                        break
            if redundant:
                removed += 1
                continue
            pruned.append(step)
        if removed:
            self._log(
                f"Warning: counter planner normalization pruned {removed} residual step(s) already covered by accepted prefix",
                "warning",
            )
        return pruned

    @staticmethod
    def _is_hard_attack_failure(reason: str) -> bool:
        """
        Identify failure causes that indicate the current attack line is structurally unproductive.
        """
        reason_norm = (reason or "").strip().lower()
        if not reason_norm:
            return False
        hard_markers = (
            "adds factual allegations not present in claim",
            "contradicts explicit claim fact",
            "candidate contradicts previous step",
            "history incompatibility",
            "citation not grounded in allowed statutes",
            "contains ungrounded citation",
        )
        return any(marker in reason_norm for marker in hard_markers)

    @staticmethod
    def _normalize_failure_pattern(reason: str) -> str:
        """Map raw reject reasons to a stable failure-pattern label."""
        text = (reason or "").strip().lower()
        if "adds factual allegations" in text:
            return "adds_facts"
        if "contradicts explicit claim fact" in text:
            return "contradicts_claim_fact"
        if "candidate contradicts previous step" in text:
            return "history_contradiction"
        if "history incompatibility" in text:
            return "history_incompatibility"
        if "citation not grounded" in text or "ungrounded citation" in text:
            return "ungrounded_citation"
        if "lexical repetition" in text:
            return "repetition"
        if "generation error" in text:
            return "generation_error"
        return "other"

    def _generate_counter_step_from_plan(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        attack_ids: List[str],
        attack_desc_map: Dict[str, str],
        reasoner_conclusion: str,
        knowledge_base: str,
        statutes_list: str,
        precedents_list: str,
        plan: List[Dict[str, str]],
        plan_index: int,
        plan_step: Dict[str, str],
        previous_steps: List[str],
        previous_summaries: List[str],
        used_norms: List[str],
        suggested_points_text: str,
        allowed_statute_index: Dict[str, set[str]],
        stream_callback: Optional[Callable[[dict], None]],
    ) -> tuple[str, str, int, List[str]]:
        """Execute one planned counter step with validation + retries."""
        active_attack_ids = [aid for aid in dict.fromkeys(attack_ids) if aid]

        base_prompt = self._build_counter_step_prompt_from_plan(
            claim=claim,
            routing_decision=routing_decision,
            attack_ids=active_attack_ids,
            attack_desc_map=attack_desc_map,
            reasoner_conclusion=reasoner_conclusion,
            knowledge_base=knowledge_base,
            statutes_list=statutes_list,
            precedents_list=precedents_list,
            plan=plan,
            plan_index=plan_index,
            plan_step=plan_step,
            previous_summaries=previous_summaries,
            used_norms=used_norms,
            suggested_points_text=suggested_points_text,
        )
        last_candidate = ""
        last_reason = "invalid output"
        step_max_tokens = self._bounded_max_tokens(self._support_step_max_tokens_cap)
        hard_failure_count = 0
        last_used_attacks = [active_attack_ids[0]] if active_attack_ids else []
        rewrite_attacks_used_format = (
            "ATTACKS_USED: [comma-separated attack ids chosen from allowed set]"
            if active_attack_ids
            else ""
        )

        for attempt in range(1, self._max_step_rewrites + 2):
            prompt = (
                base_prompt
                if attempt == 1
                else self._build_stance_rewrite_prompt(
                    original_prompt=base_prompt,
                    invalid_step=last_candidate,
                    invalid_reason=last_reason,
                    attacks_used_format=rewrite_attacks_used_format,
                )
            )
            try:
                resp = self._resilient_llm_invoke(
                    [HumanMessage(content=prompt)],
                    stream_callback=(
                        (
                            lambda token: self._emit_stream_token(
                                stream_callback,
                                phase="counter",
                                token=token,
                                step=plan_index,
                            )
                        )
                        if stream_callback
                        else None
                    ),
                    max_tokens=step_max_tokens,
                )
                candidate, used_attacks = self._parse_counter_step_payload(
                    response=(resp.content or "").strip(),
                    allowed_attack_ids=active_attack_ids,
                    fallback_attack_ids=[str(plan_step.get("attack_id", "")).strip()],
                )
                last_used_attacks = used_attacks or last_used_attacks
            except PipelineCancelled:
                raise
            except Exception as exc:
                last_reason = f"generation error: {exc}"
                if (
                    self._is_request_too_large_error(exc)
                    and step_max_tokens > self._support_step_min_tokens
                ):
                    reduced = shrink_max_tokens_progressive(
                        step_max_tokens,
                        self._support_step_min_tokens,
                    )
                    if reduced < step_max_tokens:
                        self._log(
                            "⚠️ Counter step request too large; reducing "
                            f"max_tokens {step_max_tokens} -> {reduced}",
                            "warning",
                        )
                        step_max_tokens = reduced
                if self._is_hard_attack_failure(last_reason):
                    hard_failure_count += 1
                    if hard_failure_count >= self._hard_failure_threshold:
                        self._log(
                            f"Counter-step {plan_index}: hard-failure threshold reached during generation "
                            f"(hard_attempts={hard_failure_count}), rotating attack immediately",
                            "warning",
                        )
                        return "", last_reason, hard_failure_count, last_used_attacks
                self._log(
                    f"Counter-step {plan_index} generation failed (attempt {attempt}): {exc}",
                    "warning",
                )
                continue

            last_candidate = candidate
            primary_attack_id = (
                last_used_attacks[0]
                if last_used_attacks
                else str(plan_step.get("attack_id", "")).strip()
            )
            ok, reason = self._validate_counter_step_candidate(
                candidate_step=candidate,
                previous_steps=previous_steps,
                claim=claim,
                reasoner_conclusion=reasoner_conclusion,
                expected_norm=plan_step.get("expected_norm", "N/A"),
                citation_requirement=plan_step.get("citation_requirement", "optional"),
                allowed_statute_index=allowed_statute_index,
                attack_id=primary_attack_id,
                attack_desc=attack_desc_map.get(primary_attack_id, ""),
                plan_focus=plan_step.get("focus", ""),
            )
            if ok:
                return candidate, "", 0, last_used_attacks

            last_reason = reason
            if self._is_hard_attack_failure(reason):
                hard_failure_count += 1
                if hard_failure_count >= self._hard_failure_threshold:
                    self._log(
                        f"Counter-step {plan_index}: hard-failure threshold reached "
                        f"(hard_attempts={hard_failure_count}), rotating attack immediately",
                        "warning",
                    )
                    return "", last_reason, hard_failure_count, last_used_attacks

            if stream_callback:
                try:
                    stream_callback(
                        {
                            "phase": "counter",
                            "action": "reset_step",
                            "step": plan_index,
                        }
                    )
                except PipelineCancelled:
                    raise
                except Exception:
                    pass

            self._log(
                f"Counter-step {plan_index} rejected ({reason}) "
                f"[attempt {attempt}/{self._max_step_rewrites + 1}]",
                "warning",
            )

        return "", last_reason, hard_failure_count, last_used_attacks

    def _build_counter_step_prompt_from_plan(
        self,
        claim: str,
        routing_decision: RoutingDecision,
        attack_ids: List[str],
        attack_desc_map: Dict[str, str],
        reasoner_conclusion: str,
        knowledge_base: str,
        statutes_list: str,
        precedents_list: str,
        plan: List[Dict[str, str]],
        plan_index: int,
        plan_step: Dict[str, str],
        previous_summaries: List[str],
        used_norms: List[str],
        suggested_points_text: str,
    ) -> str:
        """Create prompt for one planned counter step."""
        active_attack_ids = [aid for aid in dict.fromkeys(attack_ids) if aid]
        plan_attack_hint = str(plan_step.get("attack_id", "")).strip()
        primary_attack_id = (
            plan_attack_hint
            if plan_attack_hint and plan_attack_hint in active_attack_ids
            else (active_attack_ids[0] if active_attack_ids else "")
        )
        primary_attack_desc = attack_desc_map.get(primary_attack_id, "")
        attack_pool_lines = (
            "\n".join(
                f"- {aid}: {attack_desc_map.get(aid, _DEFAULT_ATTACK_DESCRIPTION_EN)}"
                for aid in active_attack_ids
            )
            if active_attack_ids
            else "- none"
        )
        if active_attack_ids:
            attack_usage_rules = (
                f'- Use one or more attacks from this allowed set only: {", ".join(active_attack_ids)}.\n'
                f'- Keep the step focused on the preferred attack "{(plan_step.get("attack_id") or primary_attack_id)}" '
                "unless another allowed attack is clearly better for this step."
            )
            attacks_used_format = (
                "ATTACKS_USED: [comma-separated attack ids chosen from allowed set]"
            )
        else:
            attack_usage_rules = (
                "- No taxonomy attacks are active in this run.\n"
                "- Do not output ATTACKS_USED.\n"
                "- Produce a pure counter-step based only on claim facts, norms and plan goal."
            )
            attacks_used_format = ""
        plan_lines = "\n".join(
            f"{idx}. {step.get('goal', '')} | focus: {step.get('focus', '')} | "
            f"attack: {step.get('attack_id', '')} | type: {step.get('step_type', 'OTHER')} | "
            f"novelty: {step.get('novelty_key', '')}"
            for idx, step in enumerate(plan, start=1)
        )
        summary_lines = (
            "\n".join(
                f"- Step {idx}: {summary}"
                for idx, summary in enumerate(previous_summaries, start=1)
            )
            if previous_summaries
            else "- none"
        )
        used_norms_text = ", ".join(used_norms) if used_norms else "none"
        reasoner_block = f"\nCONCLUSION TO OPPOSE:\n{reasoner_conclusion}\n"
        claim_facts = self._claim_fact_anchors_text(claim, max_items=10)
        return render_prompt(
            "counter_reasoner.step_prompt",
            claim=claim,
            reasoner_block=reasoner_block,
            claim_facts=claim_facts,
            routing_domain=routing_decision.domain,
            attack_id=primary_attack_id,
            attack_desc=primary_attack_desc,
            attack_pool_lines=attack_pool_lines,
            attack_pool_ids=", ".join(active_attack_ids),
            attack_usage_rules=attack_usage_rules,
            attacks_used_format=attacks_used_format,
            knowledge_base=knowledge_base,
            statutes_list=statutes_list,
            precedents_list=precedents_list,
            plan_lines=plan_lines,
            plan_index=plan_index,
            plan_goal=plan_step.get("goal", ""),
            plan_focus=plan_step.get("focus", ""),
            suggested_points_text=suggested_points_text,
            plan_expected_norm=plan_step.get("expected_norm", "N/A"),
            plan_citation_requirement=plan_step.get("citation_requirement", "optional"),
            plan_attack_id=(plan_step.get("attack_id") or primary_attack_id),
            plan_step_type=plan_step.get("step_type", "OTHER"),
            plan_novelty_key=plan_step.get("novelty_key", ""),
            summary_lines=summary_lines,
            used_norms_text=used_norms_text,
        )

    def _validate_counter_step_candidate(
        self,
        candidate_step: str,
        previous_steps: List[str],
        claim: str,
        reasoner_conclusion: str,
        expected_norm: str,
        citation_requirement: str,
        allowed_statute_index: Dict[str, set[str]],
        attack_id: str,
        attack_desc: str,
        plan_focus: str,
    ) -> tuple[bool, str]:
        """Validation checks for one counter step candidate."""
        text = (candidate_step or "").strip()
        if not text or text.upper() == "DONE":
            return False, "empty step"
        if self._is_garbage_text(text):
            return False, "garbage output"
        facts_ok, facts_reason = self._is_counter_step_fact_consistent_with_claim(
            claim=claim,
            candidate_step=text,
        )
        if not facts_ok:
            return False, facts_reason
        grounded_ok, grounded_reason = self._is_counter_step_grounded_in_claim_facts(
            claim=claim,
            candidate_step=text,
        )
        if not grounded_ok:
            return False, grounded_reason
        compatible, compat_reason = self._is_counter_step_compatible_with_history(
            candidate_step=text,
            previous_steps=previous_steps,
            claim=claim,
        )
        if not compatible:
            return False, compat_reason or "history incompatibility"
        _ = (reasoner_conclusion, attack_id, attack_desc, plan_focus)
        mention_matches = extract_article_mentions(text, require_code=False)
        if (
            self._step_requires_citation(
                expected_norm=expected_norm,
                citation_requirement=citation_requirement,
            )
            and not mention_matches
        ):
            return False, "missing statutory citation"
        if mention_matches:
            grounded, grounded_reason = self._has_grounded_citation(
                mentions=mention_matches,
                allowed_statute_index=allowed_statute_index,
            )
            if not grounded:
                return False, grounded_reason
        if previous_steps and self._is_repetitive_step(text, previous_steps):
            return False, "lexical repetition"
        return True, ""

    def _is_counter_step_compatible_with_history(
        self,
        *,
        candidate_step: str,
        previous_steps: List[str],
        claim: str,
    ) -> tuple[bool, str]:
        """
        Ensure candidate counter-step is semantically compatible with accepted chain history.
        """
        _ = claim
        if not previous_steps:
            return True, ""
        window = previous_steps[-3:]
        start_idx = len(previous_steps) - len(window) + 1
        for offset, prev_step in enumerate(window):
            relation = self._nli_relation(
                target_text=prev_step,
                attacker_text=candidate_step,
                actor_label="CounterReasoner",
            )
            if relation != "contradiction":
                continue
            # Reject only when contradiction is stable in both directions.
            relation_sym = self._nli_relation(
                target_text=candidate_step,
                attacker_text=prev_step,
                actor_label="CounterReasoner",
            )
            if relation_sym == "contradiction":
                step_no = start_idx + offset
                return False, f"candidate contradicts previous step {step_no}"
        return True, ""

    def _is_counter_step_fact_consistent_with_claim(
        self,
        *,
        claim: str,
        candidate_step: str,
    ) -> tuple[bool, str]:
        """Reject counter-steps that contradict explicit facts stated in the claim.

        The counter can attack legal qualification and inference, but must not
        negate factual premises expressly given in the claim text.
        """
        claim_text = (claim or "").strip()
        step_text = (candidate_step or "").strip()
        if not claim_text or not step_text:
            return True, ""

        cache_key = (claim_text, step_text)
        cached = self._counter_fact_lock_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = render_prompt(
            "counter_reasoner.fact_lock_check",
            claim=claim_text,
            candidate_step=step_text,
        )
        try:
            resp = self._resilient_llm_invoke(
                [HumanMessage(content=prompt)],
                max_tokens=self._ancillary_max_tokens(),
            )
            answer = (resp.content or "").strip().upper()
            if "DIRECT_CONTRADICTION" in answer or (
                "CONTRADICT" in answer and "INTERPRET" not in answer
            ):
                result = (False, "contradicts explicit claim fact")
            else:
                result = (True, "")
        except Exception as exc:
            # Keep counter permissive on checker outages/rate limits.
            self._log(
                f"CounterReasoner fact-lock check failed (fallback keep): {exc}",
                "warning",
            )
            result = (True, "")

        self._counter_fact_lock_cache[cache_key] = result
        return result

    def _is_counter_step_grounded_in_claim_facts(
        self,
        *,
        claim: str,
        candidate_step: str,
    ) -> tuple[bool, str]:
        """
        Reject counter-steps that add new factual allegations not present in the claim.

        Legal inferences from existing claim facts are allowed.
        """
        claim_text = (claim or "").strip()
        step_text = (candidate_step or "").strip()
        if not claim_text or not step_text:
            return True, ""

        cache_key = (claim_text, step_text)
        cached = self._new_facts_check_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = render_prompt(
            "counter_reasoner.no_new_facts",
            claim=claim_text,
            candidate_step=step_text,
        )
        try:
            resp = self._resilient_llm_invoke(
                [HumanMessage(content=prompt)],
                max_tokens=self._ancillary_max_tokens(),
            )
            answer = (resp.content or "").strip().upper()
            if "ADDS_FACTS" in answer:
                result = (
                    False,
                    "adds factual allegations not present in claim",
                )
            elif "LEGAL_INFERENCE" in answer:
                result = (True, "")
            else:
                result = (True, "")
        except Exception as exc:
            # Avoid over-blocking on checker outages/rate limits.
            self._log(
                f"⚠️ Counter new-facts check failed (fallback keep): {exc}",
                "warning",
            )
            result = (True, "")

        self._new_facts_check_cache[cache_key] = result
        return result

    @staticmethod
    def _normalize_source_for_match(source_raw: str) -> str:
        """Normalize source labels to internal statute-source keys."""
        source = (source_raw or "").strip().lower()
        if not source:
            return ""
        if source in {"codice_amministrativo", "amministrativo"}:
            return "codice_amministrativo"
        if source in {"codice_civile", "civile"}:
            return "codice_civile"
        if source in {"codice_penale", "penale"}:
            return "codice_penale"
        if "241" in source or "l. 241/1990" in source or "amm" in source:
            return "codice_amministrativo"
        if "c.c" in source or ("cod" in source and "civ" in source):
            return "codice_civile"
        if "c.p" in source or ("cod" in source and "pen" in source):
            return "codice_penale"
        return source

    def _build_allowed_statute_index(self, statutes: List[dict]) -> Dict[str, set[str]]:
        """
        Build article->source index for grounding checks.

        Example:
            {"21-octies": {"codice_amministrativo"}}
        """
        index: Dict[str, set[str]] = {}
        for statute in statutes:
            article = normalize_article_id(str(statute.get("articolo", "")))
            if not article:
                continue
            source = self._normalize_source_for_match(str(statute.get("source", "")))
            if article not in index:
                index[article] = set()
            if source:
                index[article].add(source)
        return index

    def _has_grounded_citation(
        self,
        mentions: List,
        allowed_statute_index: Dict[str, set[str]],
    ) -> tuple[bool, str]:
        """
        Ensure cited articles are grounded in the allowed counter statute set.
        """
        if not mentions:
            return False, "missing statutory citation"
        if not allowed_statute_index:
            return True, ""

        grounded = 0
        ungrounded_details: List[str] = []
        seen_keys = set()

        for mention in mentions:
            article_id = normalize_article_id(getattr(mention, "article_id", ""))
            if not article_id:
                continue
            mention_source = self._normalize_source_for_match(
                getattr(mention, "source_hint", "")
                or infer_source_hint(getattr(mention, "raw_code", ""))
            )
            key = (article_id, mention_source)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            allowed_sources = allowed_statute_index.get(article_id)
            if not allowed_sources:
                ungrounded_details.append(f"Art. {article_id} not in allowed set")
                continue
            if mention_source:
                if mention_source in allowed_sources:
                    grounded += 1
                else:
                    ungrounded_details.append(
                        f"Art. {article_id} source mismatch ({mention_source})"
                    )
                continue

            # No source provided in text: allow only if article is unambiguous.
            if len(allowed_sources) == 1:
                grounded += 1
            else:
                ungrounded_details.append(
                    f"Art. {article_id} ambiguous source (missing code)"
                )

        if grounded == 0:
            return False, "citation not grounded in allowed statutes"
        if ungrounded_details:
            # Accept partially grounded citations to avoid rejecting otherwise valid steps.
            return True, ""
        return True, ""

    @staticmethod
    def _first_sentence_legal_safe(text: str) -> str:
        """
        Return first sentence without splitting on legal abbreviations
        such as ``art.``, ``c.p.``, ``c.c.``, ``c.p.p.``.
        """
        if not text:
            return ""
        protected = str(text)
        replacements = [
            (r"\bc\.p\.p\.", "CPPTOKEN"),
            (r"\bc\.p\.", "CPTOKEN"),
            (r"\bc\.c\.", "CCTOKEN"),
            (r"\bart\.", "ARTTOKEN"),
            (r"\bd\.lgs\.", "DLGSTOKEN"),
            (r"\bd\.p\.r\.", "DPRTOKEN"),
            (r"\bn\.", "NTOKEN"),
            (r"\bl\.", "LTOKEN"),
        ]
        for pattern, token in replacements:
            protected = re.sub(pattern, token, protected, flags=re.IGNORECASE)

        first = re.split(r"(?<=[.!?])\s+", protected.strip())[0]

        restores = [
            ("CPPTOKEN", "c.p.p."),
            ("CPTOKEN", "c.p."),
            ("CCTOKEN", "c.c."),
            ("ARTTOKEN", "art."),
            ("DLGSTOKEN", "d.lgs."),
            ("DPRTOKEN", "d.p.r."),
            ("NTOKEN", "n."),
            ("LTOKEN", "l."),
        ]
        for token, value in restores:
            first = first.replace(token, value)
        return first.strip()

    @staticmethod
    def _compact_step_summary(step_text: str) -> str:
        """Compact summary used as execution memory for following steps."""
        first_sentence = CounterReasoner._first_sentence_legal_safe(step_text)
        first_sentence = re.sub(r"\s+", " ", first_sentence).strip()
        return first_sentence[: settings.truncation_counter_attack_statement]

    @staticmethod
    def _emit_stream_token(
        stream_callback: Optional[Callable[[dict], None]],
        *,
        phase: str,
        token: str,
        step: Optional[int] = None,
    ) -> None:
        """Emit one token chunk to external streaming callback."""
        if not stream_callback or not token:
            return
        payload: dict[str, str | int] = {"phase": phase, "token": token}
        if step is not None:
            payload["step"] = step
        try:
            stream_callback(payload)
        except PipelineCancelled:
            raise
        except Exception:
            # Streaming callback errors must never break counter-generation.
            pass

    def _parse_counter_step_payload(
        self,
        *,
        response: str,
        allowed_attack_ids: List[str],
        fallback_attack_ids: Optional[List[str]] = None,
    ) -> tuple[str, List[str]]:
        """Parse counter-step text plus optional ATTACKS_USED metadata."""
        allowed = [aid for aid in dict.fromkeys(allowed_attack_ids) if aid]
        attack_ids: List[str] = []
        lines = (response or "").splitlines()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if not re.match(
                r"^(ATTACKS?_USED|ATTACCHI?_USATI)\s*:",
                stripped,
                flags=re.IGNORECASE,
            ):
                continue
            payload = stripped.split(":", 1)[1]
            raw_tokens = [
                tok.strip()
                for tok in re.split(r"[,;|]", payload)
                if tok and tok.strip()
            ]
            for token in raw_tokens:
                normalized = token.strip().lower()
                if normalized in allowed and normalized not in attack_ids:
                    attack_ids.append(normalized)
                    continue
                for allowed_id in allowed:
                    if allowed_id in normalized and allowed_id not in attack_ids:
                        attack_ids.append(allowed_id)

        if not attack_ids:
            lowered = (response or "").lower()
            for attack_id in allowed:
                if attack_id.lower() in lowered and attack_id not in attack_ids:
                    attack_ids.append(attack_id)

        if not attack_ids:
            for fallback in fallback_attack_ids or []:
                fallback_id = str(fallback or "").strip().lower()
                if (
                    fallback_id
                    and fallback_id in allowed
                    and fallback_id not in attack_ids
                ):
                    attack_ids.append(fallback_id)

        if not attack_ids and allowed:
            attack_ids = [allowed[0]]

        return self._parse_step_text(response or ""), attack_ids

    def _parse_step_text(self, response: str) -> str:
        """Extract the step text from an LLM response.

        Looks for a ``STEP:`` / ``STEP N:`` (or ``PASSO:``) marker and
        returns everything after it.  Falls back to the full response
        if no marker is found.

        Also removes any leading numeric prefixes (e.g., "3 Per valutare...")
        that the LLM may have included.
        """
        lines = response.strip().split("\n")
        step_lines: List[str] = []
        found_step = False

        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()
            if re.match(
                r"^(ATTACKS?_USED|ATTACCHI?_USATI)\s*:",
                upper,
                flags=re.IGNORECASE,
            ):
                continue

            # Match STEP:, STEP 1:, STEP 12:, PASSO:, PASSO 1: etc.
            if re.match(r"^(STEP|PASSO)\s*\d*\s*:", upper):
                content = re.sub(
                    r"^(?:STEP|PASSO)\s*\d*\s*:\s*",
                    "",
                    stripped,
                    flags=re.IGNORECASE,
                ).strip()
                if content:
                    step_lines.append(content)
                found_step = True
                continue

            if found_step and stripped:
                step_lines.append(stripped)

        step_text = " ".join(step_lines).strip()

        # Fallback: use entire response (skip DECISION / STEP N: prefixes)
        if not step_text:
            fallback_lines = []
            for ln in lines:
                s = ln.strip()
                if not s:
                    continue
                su = s.upper()
                if su.startswith("DECISION:") or su.startswith("DECISIONE:"):
                    continue
                if re.match(
                    r"^(ATTACKS?_USED|ATTACCHI?_USATI)\s*:",
                    su,
                    flags=re.IGNORECASE,
                ):
                    continue
                # Strip any leading STEP N: prefix
                s = re.sub(
                    r"^(?:STEP|PASSO)\s*\d*\s*:\s*",
                    "",
                    s,
                    flags=re.IGNORECASE,
                )
                if s:
                    fallback_lines.append(s)
            step_text = " ".join(fallback_lines).strip()

        # P6 FIX: Remove leading numeric prefixes like "3 Per valutare..."
        # This handles cases where LLM responds with just "3 Per valutare" without STEP:
        step_text = re.sub(r"^\d+\s+", "", step_text)

        # FIX: Strip metadata leakage (prompt context echoed by LLM)
        step_text = re.sub(
            r"\s*(?:Norms already used|Current step|New norm):.*$",
            "",
            step_text,
            flags=re.DOTALL,
        ).strip()

        return step_text

    def _build_stance_rewrite_prompt(
        self,
        original_prompt: str,
        invalid_step: str,
        invalid_reason: str = "",
        attacks_used_format: str = "",
    ) -> str:
        """Ask the model to rewrite a step that violates counter-step consistency rules."""
        reason_text = invalid_reason or "it is not a coherent counter-step."
        return render_prompt(
            "counter_reasoner.stance_rewrite",
            original_prompt=original_prompt,
            invalid_reason=reason_text,
            invalid_step=invalid_step,
            attacks_used_format=attacks_used_format,
        )

    def _derive_counter_conclusion_ground(self, steps: List[str]) -> str:
        """Pick a final counter-rationale from the last coherent step."""
        for step in reversed(steps):
            first_sentence = self._first_sentence_legal_safe(step)
            first_sentence = re.sub(
                r"^(?:pertanto|quindi|dunque|in\s+conclusione|therefore|thus|hence|in\s+conclusion)\s*,?\s*",
                "",
                first_sentence,
                flags=re.IGNORECASE,
            ).strip()
            first_sentence = first_sentence.rstrip(" .")
            if first_sentence:
                return first_sentence
        if get_response_language() == "en":
            return "the cited norms and alleged facts do not unambiguously support the primary legal thesis"
        return (
            "le norme richiamate e i fatti allegati non giustificano "
            "in modo univoco la tesi principale"
        )

    def _assemble_counter_raw_response(
        self,
        claim: str,
        steps: List[str],
        step_attack_ids: List[List[str]],
    ) -> str:
        """Assemble counter-argument raw response from iterative steps."""
        chain_section = "**Catena di ragionamento**:\n"
        for i, step in enumerate(steps, 1):
            chain_section += f"{i}. {step}\n"

        premise_steps = steps[:-1] if len(steps) > 1 else steps

        premise_text = " ".join(premise_steps)
        norms = self._extract_cited_articles(" ".join(steps))
        norms_text = "\n".join(f"- {n}" for n in norms) if norms else "N/D"
        conclusion_ground = self._derive_counter_conclusion_ground(steps)

        unique_attacks = list(
            dict.fromkeys(
                attack_id
                for per_step in step_attack_ids
                for attack_id in (per_step or [])
                if attack_id
            )
        )
        lang = get_response_language()
        attack_locale = lang if lang in ("it", "en") else "en"
        _default_attack_desc = (
            "the cited norms do not unambiguously support the main thesis"
            if lang == "en"
            else _DEFAULT_ATTACK_DESCRIPTION_IT
        )
        attack_desc_list = [
            self._attack_description(
                attack_id,
                locale=attack_locale,
                default=_default_attack_desc,
            )
            for attack_id in unique_attacks
        ]
        if not attack_desc_list:
            attack_desc_list = [_default_attack_desc]
        attack_desc_str = "; ".join(d.rstrip(". ") for d in attack_desc_list[:3])

        if lang == "en":
            conclusion_text = (
                "Therefore, the primary legal thesis must be contested or "
                f"limited because {conclusion_ground}."
            )
            causal_link_text = (
                f"The legal analysis shows that {attack_desc_str}. "
                "The argumentative chain highlights how the norms applicable to the case "
                "allow for an alternative or limiting reconstruction relative to the primary thesis."
            )
        else:
            conclusion_text = (
                "Pertanto, la tesi giuridica principale deve essere contestata o "
                f"ridimensionata poiché {conclusion_ground}."
            )
            causal_link_text = (
                f"L'analisi giuridica dimostra che {attack_desc_str}. "
                "La catena argomentativa evidenzia come "
                "le norme applicabili al caso consentano una ricostruzione alternativa "
                "o limitativa rispetto alla tesi principale."
            )

        raw = (
            f"**Premessa Alternativa**: {premise_text}\n\n"
            f"**Norma**:\n{norms_text}\n\n"
            f"**Nesso Causale Alternativo**: {causal_link_text}\n\n"
            f"**Conclusione Contraria**: {conclusion_text}\n\n"
            f"{chain_section}"
        )
        return raw

    def _extract_arguments(self, response: str) -> List[CounterArgument]:
        """Estrae contro-argomenti strutturati dalla risposta."""
        arguments = []
        sections = response.split("**")
        current_arg: dict[str, str] = {}

        for section in sections:
            section = section.strip()
            lower_section = section.lower()

            if "premessa" in lower_section:
                if current_arg:
                    # Create CounterArgument with defaults for missing fields
                    arguments.append(
                        CounterArgument(
                            premise=current_arg.get("premise", ""),
                            norm=current_arg.get("norm", ""),
                            link=current_arg.get("link", ""),
                            conclusion=current_arg.get("conclusion", ""),
                        )
                    )
                current_arg = {"premise": section}
            elif "norma" in lower_section:
                current_arg["norm"] = section
            elif "nesso" in lower_section:
                current_arg["link"] = section
            elif "conclusione" in lower_section:
                current_arg["conclusion"] = section

        if current_arg:
            arguments.append(
                CounterArgument(
                    premise=current_arg.get("premise", ""),
                    norm=current_arg.get("norm", ""),
                    link=current_arg.get("link", ""),
                    conclusion=current_arg.get("conclusion", ""),
                )
            )

        return arguments


@dataclass
class AttackSelection:
    pool: List[str]
    attack_ids: List[str]
    descriptions: Dict[str, str]
    causal_pool: List[str] = field(default_factory=list)
    theory_pool: List[str] = field(default_factory=list)

    @property
    def attack_id(self) -> str:
        """Primary attack id (first selected), kept for compatibility."""
        return self.attack_ids[0] if self.attack_ids else ""

    @property
    def description(self) -> str:
        """Primary attack description (first selected), kept for compatibility."""
        return self.descriptions.get(self.attack_id, "")
