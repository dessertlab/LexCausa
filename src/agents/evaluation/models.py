"""
Data models shared across the evaluation sub-modules.

These classes were originally defined inside ``polisher_evaluator.py`` and are
extracted here to avoid circular imports between the orchestrator and the
mixin modules that need them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MismatchAction(Enum):
    """Action taken when a normative mismatch is detected."""

    NONE = "none"
    MATCH = "match"
    REPAIRED = "repaired"
    DROPPED = "dropped"
    REPAIR_FAILED = "repair_failed"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CitationCheck:
    """Result of checking a single citation."""

    citation: str
    found_in_kb: bool
    source_type: str
    details: str = ""
    text_verified: bool = False
    text_match: bool = False
    text_similarity: float = 0.0
    cited_text: str = ""
    db_text_preview: str = ""
    mismatch_action: str = "none"
    is_core: bool = False
    llm_mismatch_confirmed: bool = False
    llm_validated: bool = False
    repaired_text: str = ""
    repair_success: bool = False

    def to_dict(self) -> dict:
        return {
            "citation": self.citation,
            "found_in_kb": self.found_in_kb,
            "source_type": self.source_type,
            "details": self.details,
            "text_verified": self.text_verified,
            "text_match": self.text_match,
            "text_similarity": self.text_similarity,
            "cited_text": self.cited_text,
            "db_text_preview": self.db_text_preview,
            "mismatch_action": self.mismatch_action,
            "is_core": self.is_core,
            "llm_mismatch_confirmed": self.llm_mismatch_confirmed,
            "llm_validated": self.llm_validated,
            "repaired_text": self.repaired_text,
            "repair_success": self.repair_success,
        }


@dataclass
class ConsistencyReport:
    """Report on the consistency of a reasoning chain with the knowledge base."""

    agent: str
    total_citations: int = 0
    valid_citations: int = 0
    invalid_citations: int = 0
    text_matches: int = 0
    text_mismatches: int = 0
    repaired_citations: int = 0
    dropped_citations: int = 0
    relevance_gate_dropped_citations: int = 0
    citation_checks: list[CitationCheck] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "text_matches": self.text_matches,
            "text_mismatches": self.text_mismatches,
            "repaired_citations": self.repaired_citations,
            "dropped_citations": self.dropped_citations,
            "relevance_gate_dropped_citations": self.relevance_gate_dropped_citations,
            "citation_checks": [c.to_dict() for c in self.citation_checks],
            "issues": self.issues,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    claim: str
    winning_side: str
    confidence: float
    consistency_report: dict = field(default_factory=dict)
    aqa_report: dict = field(default_factory=dict)
    summary: str = ""
    polished_response: str = ""
    dialectical_tree: dict = field(default_factory=dict)
    repaired_reasoner_chain: str = ""
    repaired_counter_chain: str = ""
    repaired_reasoner_aspic_ir: dict = field(default_factory=dict)
    repaired_counter_aspic_ir: dict = field(default_factory=dict)
    counter_reasoner_gate: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "winning_side": self.winning_side,
            "confidence": self.confidence,
            "consistency_report": self.consistency_report,
            "aqa_report": self.aqa_report,
            "summary": self.summary,
            "polished_response": self.polished_response,
            "dialectical_tree": self.dialectical_tree,
            "repaired_reasoner_chain": self.repaired_reasoner_chain,
            "repaired_counter_chain": self.repaired_counter_chain,
            "repaired_reasoner_aspic_ir": self.repaired_reasoner_aspic_ir,
            "repaired_counter_aspic_ir": self.repaired_counter_aspic_ir,
            "counter_reasoner_gate": self.counter_reasoner_gate,
        }
