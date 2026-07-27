"""LangChain tools for LexCausa agents (lazy exports to avoid import cycles)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .neo4j_tools import get_statute_by_article_tool, search_precedents_tool
    from .taxonomy_tools import classify_causality_tool, get_causality_theory_tool

__all__ = [
    # Tools for Neo4j lookups
    "search_precedents_tool",
    "get_statute_by_article_tool",
    # Classification tools
    "classify_causality_tool",
    "get_causality_theory_tool",
]


def __getattr__(name: str) -> Any:
    if name in {
        "get_statute_by_article_tool",
        "search_precedents_tool",
    }:
        from . import neo4j_tools

        return getattr(neo4j_tools, name)
    if name in {"classify_causality_tool", "get_causality_theory_tool"}:
        from . import taxonomy_tools

        return getattr(taxonomy_tools, name)
    raise AttributeError(name)
