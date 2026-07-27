"""Services module for LexCausa.

Lazy exports avoid circular imports between ``services`` and ``agents.tools``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["ClaimClassifier", "LegalSearchPipeline", "SearchResult"]

if TYPE_CHECKING:
    from .claim_classifier import ClaimClassifier
    from .legal_search import LegalSearchPipeline, SearchResult


def __getattr__(name: str) -> Any:
    if name == "ClaimClassifier":
        from .claim_classifier import ClaimClassifier

        return ClaimClassifier
    if name in {"LegalSearchPipeline", "SearchResult"}:
        from .legal_search import LegalSearchPipeline, SearchResult

        return {
            "LegalSearchPipeline": LegalSearchPipeline,
            "SearchResult": SearchResult,
        }[name]
    raise AttributeError(name)
