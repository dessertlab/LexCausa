"""
Retrieval Filter Agent for LexCausa.

Agent tecnico usato esclusivamente in fase di retrieval per:
- estrazione contesto legale del claim
- filtro rilevanza norme/precedenti
- filtro applicabilità norme

Does not execute main/counter reasoning.
"""

from config import settings

from .base import BaseAgent


class RetrievalFilterAgent(BaseAgent):
    """Agent leggero per filtri retrieval con prefisso log dedicato."""

    def _resilient_model_order(self) -> list[str] | None:
        """Use retrieval-specific fallback chain (separate from router/evaluator)."""
        return settings.retrieval_model_fallback_order

    def run(self, claim: str, *args, **kwargs):
        raise NotImplementedError(
            "RetrievalFilterAgent does not execute reasoning runs"
        )

    def _log(self, message: str, level: str = "info"):
        emoji = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(
            level, "•"
        )
        print(f"{emoji} [Retrieval] {message}")
