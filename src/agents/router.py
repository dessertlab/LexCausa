"""
Pre-routing component that selects causal_type_id and theory_id before reasoning.

The router is lightweight and uses the causal configuration as the single source
of truth. It validates the LLM choice against the config and exposes anchor
norms and principle tests for downstream agents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage

from .base import AgentConfig, BaseAgent
from .tools.prompt_registry import get_prompt, render_prompt


@dataclass
class RoutingDecision:
    """Structured output of the router - domain classification, with optional causal type from chain."""

    claim: str
    domain: str  # "CIVILE" | "PENALE" | "AMMINISTRATIVO" | "ENTRAMBI"
    # Fields populated after chain classification by Reasoner
    causal_type_id: str = ""
    theory_id: str = ""
    anchor_norms: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    principle_tests: List[Dict[str, str]] = field(default_factory=list)
    additional_causal_types: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "domain": self.domain,
            "causal_type_id": self.causal_type_id,
            "theory_id": self.theory_id,
            "anchor_norms": self.anchor_norms,
            "principle_tests": self.principle_tests,
            "additional_causal_types": self.additional_causal_types,
        }


class Router(BaseAgent):
    """
    Lightweight router that classifies the claim using config_taxonomy.json.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self._route_max_tokens_cap = 192

    def route(self, claim: str) -> RoutingDecision:
        """Route a claim to a domain classification."""
        self._log(f"🔀 Routing claim: {claim[:80]}...")
        candidate = self._route_with_llm(claim)

        # Validate domain
        domain = candidate.get("domain", "").upper()
        if domain not in ("CIVILE", "PENALE", "AMMINISTRATIVO", "ENTRAMBI"):
            domain = "ENTRAMBI"  # fallback

        self._log(f"🎯 Router decision -> domain={domain}")

        return RoutingDecision(
            claim=claim,
            domain=domain,
        )

    # BaseAgent abstract method compatibility
    def run(self, claim: str, *args, **kwargs) -> RoutingDecision:  # type: ignore[override]
        """Alias for route() to satisfy BaseAgent interface."""
        return self.route(claim)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _route_with_llm(self, claim: str) -> Dict[str, str]:
        """Use LLM to classify domain, with resilient retry/key-rotation/fallback."""
        prompt = self._build_prompt(claim)
        try:
            resp = self._resilient_llm_invoke(
                [HumanMessage(content=prompt)],
                max_tokens=self._router_max_tokens(),
            )
            content = (resp.content or "").strip()
            parsed = self._parse_json_like(content)
            if parsed:
                return parsed
        except Exception as e:
            self._log(f"⚠️ Router LLM fallito: {e}", "warning")

        return {"domain": "ENTRAMBI"}

    def _build_prompt(self, claim: str) -> str:
        """Build a compact instruction for domain classification."""
        return render_prompt(
            "router.user",
            router_system=get_prompt("router.system"),
            claim=claim,
        )

    def _router_max_tokens(self) -> int:
        """Bound router output length to keep request envelopes lightweight."""
        base_max_tokens = int(getattr(self.config, "max_tokens", 512) or 512)
        if base_max_tokens <= 0:
            base_max_tokens = 512
        return max(1, min(base_max_tokens, int(self._route_max_tokens_cap)))

    def _parse_json_like(self, text: str) -> Dict[str, str]:
        """Parse JSON, tolerating fenced code."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return {
                    "domain": str(data.get("domain", "")).strip().upper(),
                }
        except Exception:
            pass
        return {}
