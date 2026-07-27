"""LexCausa agents module (lazy exports to avoid package import cycles)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import AgentConfig, BaseAgent, ReasoningResult
    from .counter_reasoner import (
        CounterArgument,
        CounterReasoner,
        CounterReasonerOutput,
    )
    from .polisher_evaluator import EvaluationResult, PolisherEvaluator
    from .reasoner import Reasoner, ReasonerOutput
    from .retrieval_filter_agent import RetrievalFilterAgent
    from .router import Router, RoutingDecision

__all__ = [
    # Base
    "BaseAgent",
    "AgentConfig",
    "ReasoningResult",
    # Retrieval Filter
    "RetrievalFilterAgent",
    # Reasoner
    "Reasoner",
    "ReasonerOutput",
    # Counter-Reasoner
    "CounterReasoner",
    "CounterReasonerOutput",
    "CounterArgument",
    # Router
    "Router",
    "RoutingDecision",
    # Polisher-Evaluator
    "PolisherEvaluator",
    "EvaluationResult",
]


def __getattr__(name: str) -> Any:
    if name in {"BaseAgent", "AgentConfig", "ReasoningResult"}:
        from . import base

        return getattr(base, name)
    if name in {"Reasoner", "ReasonerOutput"}:
        from . import reasoner

        return getattr(reasoner, name)
    if name in {"CounterReasoner", "CounterReasonerOutput", "CounterArgument"}:
        from . import counter_reasoner

        return getattr(counter_reasoner, name)
    if name in {"PolisherEvaluator", "EvaluationResult"}:
        from . import polisher_evaluator

        return getattr(polisher_evaluator, name)
    if name in {"RetrievalFilterAgent"}:
        from . import retrieval_filter_agent

        return getattr(retrieval_filter_agent, name)
    if name in {"Router", "RoutingDecision"}:
        from . import router

        return getattr(router, name)
    raise AttributeError(name)
