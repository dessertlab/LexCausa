"""
Scoring functions for AQA evaluation.

Provides argument quality, readability, coherence, norm support,
and semantics scoring for reasoning chain links.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings  # noqa: E402

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None  # type: ignore[assignment]


class ScoringMixin:
    """Mixin providing scoring methods to the evaluator."""

    def _get_arg_quality_model(self):
        if self._arg_quality is not None:
            return self._arg_quality
        if not self._aqa_arg_quality_use_model or not self._aqa_arg_quality_model:
            self._arg_quality = None
            return None
        if pipeline is None:
            self._arg_quality = None
            return None
        try:
            self._arg_quality = pipeline(
                "text-classification",
                model=self._aqa_arg_quality_model,
                top_k=None,
            )
        except Exception:
            self._arg_quality = None
        return self._arg_quality

    def _argument_quality_score(
        self, premise_text: str, rule_text: str, conclusion_text: str
    ) -> float:
        structure = 0
        structure += 1 if premise_text else 0
        structure += 1 if rule_text else 0
        structure += 1 if conclusion_text else 0
        structure_score = structure / 3.0
        quality_text = f"{premise_text}\n{rule_text}\n{conclusion_text}".strip()
        model_score = None
        model = self._get_arg_quality_model()
        if model is not None and quality_text:
            try:
                result = model(quality_text)
                scores = []
                for item in result[0] if isinstance(result, list) else []:
                    if isinstance(item, dict) and "score" in item:
                        scores.append(float(item["score"]))
                if scores:
                    model_score = max(scores)
            except Exception:
                model_score = None
        if model_score is not None:
            quality_score = self._clamp01(model_score)
        else:
            similarity_score = self._tfidf_similarity(
                premise_text + " " + rule_text, conclusion_text
            )
            if similarity_score == 0.0:
                similarity_score = self._similarity(
                    premise_text + " " + rule_text, conclusion_text
                )
            quality_score = self._clamp01(similarity_score)
        return self._clamp01(
            settings.readability_structure_weight * structure_score
            + settings.readability_quality_weight * quality_score
        )

    def _coherence_score(
        self, premise_text: str, conclusion_text: str, rule_text: str
    ) -> float:
        base = self._similarity(premise_text + " " + rule_text, conclusion_text)
        sentences = self._split_sentences(rule_text)
        if len(sentences) < 2:
            return base
        sims = []
        for idx in range(len(sentences) - 1):
            sims.append(self._similarity(sentences[idx], sentences[idx + 1]))
        if not sims:
            return base
        return self._clamp01(
            settings.coherence_base_weight * base
            + settings.coherence_chain_weight * (sum(sims) / len(sims))
        )

    def _norm_support_score(
        self,
        text: str,
        retrieved_norms: list[dict] | None = None,
        max_citations: int | None = None,
    ) -> tuple[float, dict]:
        citations = self._extract_statute_citations(text)
        max_citations = max(1, max_citations or self._aqa_normsupport_max_citations)

        # Quantity: saturates at max_citations with diminishing marginal gains.
        citation_count = len(citations)
        capped_count = min(citation_count, max_citations)
        citation_count_score = (capped_count / max_citations) ** 0.5

        # Specificity: reward fully-specified citations (code/source explicit).
        explicit_code_count = sum(
            1
            for c in citations
            if any(token in c.lower() for token in ("c.c", "c.p", "241/1990", "l. 241"))
        )
        citation_specificity_score = (
            explicit_code_count / citation_count if citation_count > 0 else 0.0
        )

        # Overall citation quality (count + precision of legal reference).
        citation_score = self._clamp01(
            0.75 * citation_count_score + 0.25 * citation_specificity_score
        )

        # Retrieved grounding score: how well cited norms are supported by context norms.
        retrieved_score = 0.0
        retrieved_count = 0
        if retrieved_norms:
            scores = []
            for item in retrieved_norms:
                if not isinstance(item, dict):
                    continue
                val = item.get("similarity")
                if val is None:
                    val = item.get("score")
                if isinstance(val, (int, float)):
                    scores.append(float(val))
            if scores:
                retrieved_count = len(scores)
                if self._aqa_normsupport_retrieved_agg == "max":
                    retrieved_score = max(scores)
                else:
                    # Slightly robust average: mean of top-K (K<=3) if many entries.
                    top = sorted(scores, reverse=True)[: min(3, len(scores))]
                    retrieved_score = sum(top) / len(top)

        final_score = self._clamp01(
            self._aqa_normsupport_citation_weight * citation_score
            + self._aqa_normsupport_retrieved_weight * retrieved_score
        )
        details = {
            "citation_count": citation_count,
            "citation_count_score": citation_count_score,
            "citation_specificity_score": citation_specificity_score,
            "citation_score": citation_score,
            "retrieved_score": retrieved_score,
            "retrieved_norms_count": retrieved_count,
            "final": final_score,
        }
        return final_score, details

    def _semantics_score(
        self, premise_text: str, conclusion_text: str
    ) -> tuple[float, dict]:
        """Semantic coherence between premise and conclusion using cosine similarity."""
        score = self._similarity(premise_text, conclusion_text)
        return self._clamp01(score), {"method": "similarity", "score": score}
