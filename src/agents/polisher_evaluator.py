"""
LexCausa Polisher-Evaluator Agent.

The Polisher-Evaluator is responsible for:
1. Receiving arguments from Reasoner and Counter-Reasoner
2. Evaluating the dialectical exchange
3. Checking consistency of reasoning chains against the knowledge base
4. Determining which arguments prevail via AQA scoring
5. Repairing and polishing the final output for presentation

This agent acts as a judge/evaluator of the argumentation.

The implementation is split across several mixin modules under
``agents.evaluation``:

- :class:`~agents.evaluation.nlp_utils.NLPUtils` -- text helpers
- :class:`~agents.evaluation.scoring.ScoringMixin` -- quality scoring
- :class:`~agents.evaluation.consistency_checker.ConsistencyMixin` -- citation
  verification & chain repair
- :class:`~agents.evaluation.aqa_engine.AQAEngineMixin` -- ASPIC+ AQA pipeline
"""

import re
from typing import Any, Callable, Optional

from langchain_core.messages import HumanMessage

from config import settings

from .base import AgentConfig, BaseAgent
from .evaluation import AQAEngineMixin, ConsistencyMixin, NLPUtils, ScoringMixin
from .evaluation.models import (  # noqa: F401 -- re-exported
    CitationCheck,
    ConsistencyReport,
    EvaluationResult,
    MismatchAction,
)
from .tools.prompt_registry import render_prompt

# ---------------------------------------------------------------------------
# Main class -- composes all mixins
# ---------------------------------------------------------------------------


class PolisherEvaluator(
    NLPUtils,
    ScoringMixin,
    ConsistencyMixin,
    AQAEngineMixin,
    BaseAgent,
):
    """
    Legal Polisher-Evaluator Agent.

    Evaluates the dialectical exchange between Reasoner and Counter-Reasoner,
    determines which arguments prevail, and produces a polished final output.

    Implementation is split across mixin classes for maintainability:
    - **NLPUtils**: text normalization, embedding, similarity
    - **ScoringMixin**: argument quality, coherence, readability
    - **ConsistencyMixin**: citation verification, mismatch repair
    - **AQAEngineMixin**: ASPIC+ link building, cross-attack, verdict
    """

    STATUTE_PATTERN = re.compile(
        r"(?:art(?:icolo)?\.?\s*)"
        r"(\d{1,4}(?:"
        r"-(?:[a-z0-9]{2,})|"
        r"(?:noviesdecies|octiesdecies|septiesdecies|sexiesdecies|"
        r"quinquiesdecies|quaterdecies|terdecies|duodecies|undecies|"
        r"quinquies|septies|quater|sexies|octies|nonies|decies|vicies|ter|bis)|"
        r"\s+(?:noviesdecies|octiesdecies|septiesdecies|sexiesdecies|"
        r"quinquiesdecies|quaterdecies|terdecies|duodecies|undecies|"
        r"quinquies|septies|quater|sexies|octies|nonies|decies|vicies|ter|bis)"
        r")?)\s*"
        r"(c\.?\s*[cp]\.?|"
        r"cod(?:ice)?\.?\s*(?:civ(?:ile)?|pen(?:ale)?|amm(?:inistrativ[oa])?)|"
        r"l\.?\s*241(?:\s*/\s*1990)?|"
        r"legge\s*241(?:\s*/\s*1990)?)?",
        re.IGNORECASE,
    )

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Polisher-Evaluator agent."""
        super().__init__(config)
        self._log("Polisher-Evaluator initialized")
        self._embedder = None
        self._arg_quality = None
        self._tfidf_vectorizer = None
        self._tfidf_cache: dict[str, Any] = {}
        self._embed_cache: dict[str, Any] = {}
        self._statute_meta_cache: dict[tuple[str, str], dict] = {}
        self._aqa_enabled = settings.aqa_enabled
        self._aqa_alpha = settings.aqa_alpha
        self._aqa_beta = settings.aqa_beta
        self._aqa_gamma = settings.aqa_gamma
        self._aqa_attack_top_k = settings.aqa_attack_top_k
        self._aqa_min_semantic_overlap = settings.aqa_min_semantic_overlap
        self._aqa_min_strength_ratio = settings.aqa_min_strength_ratio
        self._aqa_damage_factor = settings.aqa_damage_factor
        self._aqa_allow_factual_attacks = settings.aqa_allow_factual_attacks
        self._aqa_allow_cross_codice = settings.aqa_allow_cross_codice

        self._aqa_attack_type_multipliers = settings.aqa_attack_type_multipliers
        self._aqa_strength_ratio_by_type = settings.aqa_strength_ratio_by_type
        self._aqa_severity_book_map = settings.aqa_severity_book_map
        self._aqa_procedural_categories = set(
            settings.aqa_procedural_severity_categories
        )
        self._aqa_valid_attack_types = set(settings.aqa_valid_attack_types)
        self._aqa_default_attack_type = settings.aqa_default_attack_type
        self._aqa_verdict_pos = settings.aqa_verdict_pos_threshold
        self._aqa_verdict_neg = settings.aqa_verdict_neg_threshold
        self._aqa_embedding_model = settings.aqa_embedding_model
        self._aqa_arg_quality_model = settings.aqa_argument_quality_model
        self._aqa_arg_quality_use_model = settings.aqa_argument_quality_use_model
        self._aqa_tfidf_max_features = settings.aqa_tfidf_max_features
        self._aqa_normsupport_max_citations = settings.aqa_normsupport_max_citations
        self._aqa_normsupport_citation_weight = settings.aqa_normsupport_citation_weight
        self._aqa_normsupport_retrieved_weight = (
            settings.aqa_normsupport_retrieved_weight
        )
        self._aqa_normsupport_retrieved_agg = settings.aqa_normsupport_retrieved_agg
        self._aqa_severity_map_penale = settings.aqa_severity_map_penale
        self._aqa_severity_map_civile = settings.aqa_severity_map_civile
        self._aqa_structural_adjustments_enabled = (
            settings.aqa_structural_adjustments_enabled
        )
        self._aqa_redundancy_similarity_threshold = (
            settings.aqa_redundancy_similarity_threshold
        )
        self._aqa_redundancy_penalty_weight = settings.aqa_redundancy_penalty_weight
        self._aqa_redundancy_max_penalty = settings.aqa_redundancy_max_penalty
        self._aqa_attack_coverage_enabled = settings.aqa_attack_coverage_enabled
        self._aqa_attack_coverage_similarity_threshold = (
            settings.aqa_attack_coverage_similarity_threshold
        )
        self._aqa_attack_coverage_overlap_threshold = (
            settings.aqa_attack_coverage_overlap_threshold
        )
        self._aqa_attack_coverage_min_attack_value = (
            settings.aqa_attack_coverage_min_attack_value
        )
        self._aqa_attack_coverage_bonus_weight = (
            settings.aqa_attack_coverage_bonus_weight
        )
        self._aqa_attack_coverage_max_bonus = settings.aqa_attack_coverage_max_bonus
        self._aqa_attack_coverage_second_hit_weight = (
            settings.aqa_attack_coverage_second_hit_weight
        )
        self._aqa_attack_coverage_third_hit_weight = (
            settings.aqa_attack_coverage_third_hit_weight
        )
        self._aqa_verdict_use_adjusted_score = settings.aqa_verdict_use_adjusted_score
        self._aqa_lock_reasoner_plausibility = settings.aqa_lock_reasoner_plausibility

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        claim: str,
        domain: str = "BOTH",
        reasoner_output: dict | None = None,
        counter_reasoner_output: dict | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate the dialectical exchange and produce final output."""
        progress_callback: Optional[Callable[[str, dict], None]] = kwargs.get(
            "progress_callback"
        )

        def _emit(event_name: str, payload: dict) -> None:
            if not progress_callback:
                return
            try:
                progress_callback(event_name, payload)
            except Exception:
                # Progress callbacks must not break evaluation flow.
                pass

        self._log("Starting consistency evaluation...")
        self._log(f"Domain: {domain}")
        _emit(
            "evaluation_status",
            {
                "stage": "start",
                "message": "Starting knowledge base consistency check",
            },
        )

        # Refresh AQA parameters (supports frontend overrides)
        self._aqa_alpha = settings.aqa_alpha
        self._aqa_beta = settings.aqa_beta
        self._aqa_gamma = settings.aqa_gamma
        self._aqa_min_semantic_overlap = settings.aqa_min_semantic_overlap
        self._aqa_min_strength_ratio = settings.aqa_min_strength_ratio
        self._aqa_damage_factor = settings.aqa_damage_factor
        self._aqa_allow_factual_attacks = settings.aqa_allow_factual_attacks
        self._aqa_allow_cross_codice = settings.aqa_allow_cross_codice
        self._aqa_strength_ratio_by_type = settings.aqa_strength_ratio_by_type
        self._aqa_procedural_categories = set(
            settings.aqa_procedural_severity_categories
        )
        self._aqa_valid_attack_types = set(settings.aqa_valid_attack_types)
        self._aqa_default_attack_type = settings.aqa_default_attack_type
        self._aqa_structural_adjustments_enabled = (
            settings.aqa_structural_adjustments_enabled
        )
        self._aqa_redundancy_similarity_threshold = (
            settings.aqa_redundancy_similarity_threshold
        )
        self._aqa_redundancy_penalty_weight = settings.aqa_redundancy_penalty_weight
        self._aqa_redundancy_max_penalty = settings.aqa_redundancy_max_penalty
        self._aqa_attack_coverage_enabled = settings.aqa_attack_coverage_enabled
        self._aqa_attack_coverage_similarity_threshold = (
            settings.aqa_attack_coverage_similarity_threshold
        )
        self._aqa_attack_coverage_overlap_threshold = (
            settings.aqa_attack_coverage_overlap_threshold
        )
        self._aqa_attack_coverage_min_attack_value = (
            settings.aqa_attack_coverage_min_attack_value
        )
        self._aqa_attack_coverage_bonus_weight = (
            settings.aqa_attack_coverage_bonus_weight
        )
        self._aqa_attack_coverage_max_bonus = settings.aqa_attack_coverage_max_bonus
        self._aqa_attack_coverage_second_hit_weight = (
            settings.aqa_attack_coverage_second_hit_weight
        )
        self._aqa_attack_coverage_third_hit_weight = (
            settings.aqa_attack_coverage_third_hit_weight
        )
        self._aqa_verdict_use_adjusted_score = settings.aqa_verdict_use_adjusted_score
        self._aqa_lock_reasoner_plausibility = settings.aqa_lock_reasoner_plausibility

        reasoner_output = reasoner_output or {}
        counter_reasoner_output = counter_reasoner_output or {}

        # ----- Consistency checking -----
        reasoner_chain = reasoner_output.get("reasoning_chain", [])
        reasoner_raw = reasoner_output.get("raw_response", "")
        reasoner_aspic_ir = reasoner_output.get("aspic_ir", {})
        reasoner_report = self._check_consistency(
            agent="reasoner",
            reasoning_chain=reasoner_chain,
            raw_response=reasoner_raw,
            domain=domain,
            claim=claim,
            aspic_ir=reasoner_aspic_ir,
            progress_callback=lambda payload: _emit(
                "evaluation_citation_check", payload
            ),
        )
        _emit(
            "evaluation_status",
            {
                "stage": "kb_reasoner_done",
                "message": "Main chain KB check completed",
            },
        )
        _emit(
            "evaluation_partial",
            {"consistency_report": {"reasoner": reasoner_report.to_dict()}},
        )

        counter_chain = counter_reasoner_output.get("reasoning_chain", [])
        counter_raw = counter_reasoner_output.get("raw_response", "")
        counter_aspic_ir = counter_reasoner_output.get("aspic_ir", {})
        counter_report = self._check_consistency(
            agent="counter_reasoner",
            reasoning_chain=counter_chain,
            raw_response=counter_raw,
            domain=domain,
            claim=claim,
            aspic_ir=counter_aspic_ir,
            progress_callback=lambda payload: _emit(
                "evaluation_citation_check", payload
            ),
        )
        _emit(
            "evaluation_status",
            {
                "stage": "kb_counter_done",
                "message": "Counter chain KB check completed",
            },
        )
        _emit(
            "evaluation_partial",
            {
                "consistency_report": {
                    "reasoner": reasoner_report.to_dict(),
                    "counter_reasoner": counter_report.to_dict(),
                }
            },
        )

        self._log(
            "Reasoner consistency checks: "
            f"{reasoner_report.valid_citations}/{reasoner_report.total_citations} valid"
        )
        self._log(
            "Counter consistency checks: "
            f"{counter_report.valid_citations}/{counter_report.total_citations} valid"
        )

        # ----- Counter opposition gate (post-check with both chains) -----
        counter_gate = self._evaluate_counter_opposition_gate(
            claim=claim,
            reasoner_chain=reasoner_chain,
            counter_chain=counter_chain,
            counter_already_abstained=bool(counter_reasoner_output.get("abstained")),
            counter_abstention_reason=counter_reasoner_output.get(
                "abstention_reason", ""
            ),
        )
        if counter_gate.get("abstain"):
            counter_reasoner_output = dict(counter_reasoner_output)
            counter_reasoner_output["abstained"] = True
            counter_reasoner_output["abstention_reason"] = counter_gate.get(
                "reason",
                "The Counter-Reasoner does not have enough material to argue against.",
            )
            self._log("⚠️ Counter gate activated: Counter-Reasoner marked as abstained")
        _emit(
            "evaluation_partial",
            {"consistency_report": {"counter_reasoner_gate": counter_gate}},
        )
        _emit(
            "evaluation_status",
            {"stage": "gate_done", "message": "Opposition check completed"},
        )

        # ----- Repair chains if needed -----
        self._log("Checking if reasoning chains need repair...")
        _emit(
            "evaluation_status",
            {
                "stage": "repair_start",
                "message": "Citation and chain repair check in progress",
            },
        )

        if (
            reasoner_report.repaired_citations > 0
            or reasoner_report.dropped_citations > 0
        ):
            repaired_reasoner_chain = self._regenerate_reasoning_chain_with_llm(
                original_chain=reasoner_raw,
                citation_checks=reasoner_report.citation_checks,
                agent_name="reasoner",
            )
        else:
            repaired_reasoner_chain = reasoner_raw

        if counter_gate.get("abstain"):
            repaired_counter_chain = counter_raw
            self._log(
                "⛔ Counter chain repair skipped: counter marked as abstained by gate"
            )
        elif (
            counter_report.repaired_citations > 0
            or counter_report.dropped_citations > 0
        ):
            repaired_counter_chain = self._regenerate_reasoning_chain_with_llm(
                original_chain=counter_raw,
                citation_checks=counter_report.citation_checks,
                agent_name="counter_reasoner",
            )
        else:
            repaired_counter_chain = counter_raw

        # Repair ASPIC IR structures
        repaired_reasoner_aspic = self._repair_aspic_ir(
            aspic_ir=reasoner_aspic_ir,
            citation_checks=reasoner_report.citation_checks,
            repaired_chain_text=repaired_reasoner_chain,
            claim=claim,
            role="support",
            statutes=reasoner_output.get(
                "statutes", reasoner_output.get("relevant_statutes", [])
            ),
            precedents=reasoner_output.get(
                "precedents", reasoner_output.get("relevant_precedents", [])
            ),
            metadata=reasoner_aspic_ir.get("metadata", {}),
        )
        if counter_gate.get("abstain"):
            repaired_counter_aspic = {}
            self._log(
                "⛔ Counter ASPIC repair skipped: no counter material to consolidate"
            )
        else:
            repaired_counter_aspic = self._repair_aspic_ir(
                aspic_ir=counter_aspic_ir,
                citation_checks=counter_report.citation_checks,
                repaired_chain_text=repaired_counter_chain,
                claim=claim,
                role="counter",
                statutes=counter_reasoner_output.get(
                    "statutes", counter_reasoner_output.get("relevant_statutes", [])
                ),
                precedents=counter_reasoner_output.get(
                    "precedents", counter_reasoner_output.get("relevant_precedents", [])
                ),
                metadata=counter_aspic_ir.get("metadata", {}),
            )
        _emit(
            "evaluation_partial",
            {
                "repaired_reasoner_chain": repaired_reasoner_chain,
                "repaired_counter_chain": repaired_counter_chain,
                "repaired_reasoner_aspic_ir": repaired_reasoner_aspic,
                "repaired_counter_aspic_ir": repaired_counter_aspic,
            },
        )
        _emit(
            "evaluation_status",
            {"stage": "repair_done", "message": "Chain/ASPIC repair completed"},
        )

        # Dialectical tree bundle
        reasoner_ir = reasoner_output.get("aspic_ir")
        counter_ir = counter_reasoner_output.get("aspic_ir")
        dialectical_tree = {}
        if reasoner_ir or counter_ir:
            dialectical_tree = {
                "schema": "aspic_ir_bundle_v1",
                "reasoner": reasoner_ir,
                "counter": counter_ir,
                "repaired_reasoner": repaired_reasoner_aspic,
                "repaired_counter": repaired_counter_aspic,
            }

        summary = self._generate_consistency_summary(reasoner_report, counter_report)
        _emit("evaluation_partial", {"summary": summary})

        # ----- AQA Phase -----
        aqa_reasoner_ir = (
            repaired_reasoner_aspic
            if repaired_reasoner_aspic
            else (reasoner_aspic_ir or {})
        )
        if counter_gate.get("abstain"):
            aqa_counter_ir = {}
            self._log("⛔ Counter excluded from AQA: opposition gate marked abstention")
        else:
            aqa_counter_ir = (
                repaired_counter_aspic
                if repaired_counter_aspic
                else (counter_aspic_ir or {})
            )

        if repaired_reasoner_aspic:
            r_meta = repaired_reasoner_aspic.get("_repair_metadata", {})
            self._log(
                f"AQA using REPAIRED Reasoner IR "
                f"(repaired={r_meta.get('total_repaired', 0)}, "
                f"dropped={r_meta.get('total_dropped', 0)})"
            )
        else:
            self._log("AQA using ORIGINAL Reasoner IR (no repairs needed)")

        if counter_gate.get("abstain"):
            self._log("AQA Counter IR skipped (counter abstained by Polisher gate)")
        elif repaired_counter_aspic:
            c_meta = repaired_counter_aspic.get("_repair_metadata", {})
            self._log(
                f"AQA using REPAIRED Counter IR "
                f"(repaired={c_meta.get('total_repaired', 0)}, "
                f"dropped={c_meta.get('total_dropped', 0)})"
            )
        else:
            self._log("AQA using ORIGINAL Counter IR (no repairs needed)")

        aqa_report = self._run_aqa_phase(
            reasoner_ir=aqa_reasoner_ir,
            counter_ir=aqa_counter_ir,
            domain=domain,
            progress_callback=lambda payload: _emit("evaluation_aqa_progress", payload),
        )
        _emit("evaluation_partial", {"aqa_report": aqa_report})
        _emit(
            "evaluation_status",
            {
                "stage": "aqa_done",
                "message": "Attack analysis and AQA scoring completed",
            },
        )

        self._log("Evaluation complete")
        _emit(
            "evaluation_status",
            {"stage": "done", "message": "Evaluation completed"},
        )

        return EvaluationResult(
            claim=claim,
            winning_side="",
            confidence=0.0,
            consistency_report={
                "reasoner": reasoner_report.to_dict(),
                "counter_reasoner": counter_report.to_dict(),
                "counter_reasoner_gate": counter_gate,
            },
            aqa_report=aqa_report,
            summary=summary,
            polished_response="",
            dialectical_tree=dialectical_tree,
            repaired_reasoner_chain=repaired_reasoner_chain,
            repaired_counter_chain=repaired_counter_chain,
            repaired_reasoner_aspic_ir=repaired_reasoner_aspic,
            repaired_counter_aspic_ir=repaired_counter_aspic,
            counter_reasoner_gate=counter_gate,
        )

    def _substantive_chain_steps(self, chain: list[str]) -> list[str]:
        """Remove placeholder/meta steps that are not legal reasoning content."""
        placeholder_prefixes = (
            "precedents:",
            "reasoning chain not available",
        )
        steps: list[str] = []
        for step in chain or []:
            cleaned = (step or "").strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered.startswith(placeholder_prefixes):
                continue
            steps.append(cleaned)
        return steps

    def _evaluate_counter_opposition_gate(
        self,
        claim: str,
        reasoner_chain: list[str],
        counter_chain: list[str],
        counter_already_abstained: bool,
        counter_abstention_reason: str,
    ) -> dict:
        """Verify that the counter chain is materially opposite to the reasoner.

        If not, mark counter as abstained with an explicit reason.
        """
        gate = {
            "checked": True,
            "abstain": False,
            "label": "OPPOSING",
            "reason": "",
            "reasoner_steps": 0,
            "counter_steps": 0,
        }

        if counter_already_abstained:
            gate.update(
                {
                    "abstain": True,
                    "label": "ALREADY_ABSTAINED",
                    "reason": counter_abstention_reason
                    or "Counter-Reasoner already abstained before Polisher check.",
                }
            )
            self._log(
                "ℹ️ Counter gate: skipping opposition check (counter already abstained)"
            )
            return gate

        reasoner_steps = self._substantive_chain_steps(reasoner_chain)
        counter_steps = self._substantive_chain_steps(counter_chain)
        gate["reasoner_steps"] = len(reasoner_steps)
        gate["counter_steps"] = len(counter_steps)

        self._log(
            "🧪 Counter gate input: "
            f"reasoner_steps={len(reasoner_steps)}, counter_steps={len(counter_steps)}"
        )
        if reasoner_steps:
            self._log(f"   ↳ Reasoner last step: {reasoner_steps[-1][:120]}")
        if counter_steps:
            self._log(f"   ↳ Counter last step: {counter_steps[-1][:120]}")

        if len(counter_steps) < 2:
            gate.update(
                {
                    "abstain": True,
                    "label": "INSUFFICIENT_MATERIAL",
                    "reason": (
                        "The Counter-Reasoner does not have enough material to "
                        "argue against independently and consistently."
                    ),
                }
            )
            self._log("⚠️ Counter gate: insufficient counter material")
            return gate

        if not reasoner_steps:
            gate.update(
                {
                    "abstain": False,
                    "label": "SKIPPED_NO_REASONER_CHAIN",
                    "reason": "Opposition check not performed: Reasoner chain missing.",
                }
            )
            self._log("⚠️ Counter gate: cannot compare, reasoner chain missing")
            return gate

        prompt = render_prompt(
            "polisher.counter_gate",
            claim=claim,
            reasoner_chain=chr(10).join(
                f"{i + 1}. {s}" for i, s in enumerate(reasoner_steps)
            ),
            counter_chain=chr(10).join(
                f"{i + 1}. {s}" for i, s in enumerate(counter_steps)
            ),
        )

        try:
            resp = self._resilient_llm_invoke([HumanMessage(content=prompt)])
            verdict = (resp.content or "").strip().upper()
            self._log(f"🧪 Counter gate LLM verdict raw: {verdict[:120]}")
        except Exception as e:
            self._log(f"⚠️ Counter gate LLM error: {e}", "warning")
            gate.update(
                {
                    "abstain": True,
                    "label": "CHECK_FAILED",
                    "reason": (
                        "The Counter-Reasoner does not have enough verifiable material "
                        "to argue against."
                    ),
                }
            )
            return gate

        # Accept both full opposition and materially limitative opposition.
        if re.search(r"\bOPPOSING_LIMITATIVE\b", verdict):
            gate["label"] = "OPPOSING_LIMITATIVE"
            self._log("✅ Counter gate: limitative opposition confirmed")
            return gate
        if re.search(r"\bOPPOSING(?:_STRONG)?\b", verdict):
            gate["label"] = "OPPOSING_STRONG"
            self._log("✅ Counter gate: opposition confirmed")
            return gate

        label = "UNCLEAR"
        if re.search(r"\bAGREEING\b", verdict):
            label = "AGREEING"
        elif re.search(r"\bUNCLEAR\b", verdict):
            label = "UNCLEAR"

        gate.update(
            {
                "abstain": True,
                "label": label,
                "reason": (
                    "The Counter-Reasoner does not have enough material to "
                    "argue against in a way that is effectively opposite to the Reasoner."
                ),
            }
        )
        self._log(
            f"⚠️ Counter gate: opposition not sufficient (label={label}) -> abstention"
        )
        return gate
