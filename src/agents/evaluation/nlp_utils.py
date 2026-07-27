"""
NLP utility functions for the Polisher-Evaluator.

Provides text normalization, embedding, TF-IDF vectorization,
cosine similarity, and LLM-based contradiction detection.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from ..tools.prompt_registry import render_prompt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings  # noqa: E402
from services.groq_client import get_chat_groq, resilient_chat_call  # noqa: E402


class NLPUtils:
    """Mixin providing NLP helper methods to the evaluator."""

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"[.!?]\s+", self._normalize_text(text))
        return [p.strip() for p in parts if p.strip()]

    def _get_sentence_transformer(self):
        if self._embedder is not None:
            return self._embedder
        if not self._aqa_embedding_model:
            self._embedder = None
            return None
        if SentenceTransformer is None:
            self._embedder = None
            return None
        try:
            self._embedder = SentenceTransformer(self._aqa_embedding_model)
        except Exception:
            self._embedder = None
        return self._embedder

    def _check_nli_contradiction(
        self, target_text: str, attacker_text: str
    ) -> tuple[str, float]:
        """Run LLM inference to detect genuine contradiction.

        Uses the Groq LLM to classify the relationship between target and
        attacker reasoning as *contradiction*, *entailment*, or *neutral*.

        This replaces the former DeBERTa MNLI model which did not perform
        well on long Italian legal texts.

        Returns:
            (label, score) -- the LLM-determined label and a confidence
            score (1.0 for definite answers, 0.5 for fallback).
        """
        t = self._normalize_text(target_text).strip()[: settings.truncation_nli_text]
        a = self._normalize_text(attacker_text).strip()[: settings.truncation_nli_text]
        if not t or not a:
            return ("neutral", 0.0)

        try:
            llm = get_chat_groq(
                temperature=settings.classifier_temperature,
                max_tokens=settings.nli_max_tokens,
            )

            system_prompt = render_prompt("nlp_utils.nli_system")

            user_prompt = render_prompt(
                "nlp_utils.nli_user",
                target_text=t,
                attacker_text=a,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = resilient_chat_call(llm, messages)
            answer = response.content.strip().upper()

            if "CONTRADICTION" in answer:
                return ("contradiction", 1.0)
            elif "ENTAILMENT" in answer:
                return ("entailment", 1.0)
            elif "NEUTRAL" in answer:
                return ("neutral", 1.0)
            else:
                self._log(
                    f"      \u26a0\ufe0f LLM NLI unrecognised: "
                    f'"{answer}", falling back to neutral'
                )
                return ("neutral", 0.5)

        except Exception as exc:
            self._log(
                f"      \u26a0\ufe0f LLM NLI classification failed: {exc}",
                "warning",
            )
            return ("neutral", 0.0)

    def _get_tfidf_vectorizer(self):
        if self._tfidf_vectorizer is not None:
            return self._tfidf_vectorizer
        if TfidfVectorizer is None:
            self._tfidf_vectorizer = None
            return None
        try:
            self._tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words=None,
                max_features=self._aqa_tfidf_max_features,
            )
        except Exception:
            self._tfidf_vectorizer = None
        return self._tfidf_vectorizer

    def _vector_to_list(self, vec: Any) -> list[float]:
        if hasattr(vec, "toarray"):
            vec = vec.toarray()
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        if isinstance(vec, list) and vec and isinstance(vec[0], list):
            return vec[0]
        return vec or []

    def _cosine_sim(self, v1: list[float], v2: list[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    def _embed_text(self, text: str) -> list[float]:
        text = self._normalize_text(text)
        if text in self._embed_cache:
            return self._embed_cache[text]
        embedder = self._get_sentence_transformer()
        if embedder is None:
            return []
        try:
            vec = embedder.encode([text], convert_to_numpy=True)[0]
            result = vec.tolist()
            self._embed_cache[text] = result
            return result
        except Exception:
            return []

    def _tfidf_vector(self, text: str, other: str) -> tuple[list[float], list[float]]:
        vectorizer = self._get_tfidf_vectorizer()
        if vectorizer is None:
            return [], []
        key = f"{text}||{other}"
        if key in self._tfidf_cache:
            return self._tfidf_cache[key]
        try:
            matrix = vectorizer.fit_transform([text, other])
            v1 = self._vector_to_list(matrix[0])
            v2 = self._vector_to_list(matrix[1])
            self._tfidf_cache[key] = (v1, v2)
            return v1, v2
        except Exception:
            return [], []

    def _tfidf_similarity(self, text_a: str, text_b: str) -> float:
        v1, v2 = self._tfidf_vector(text_a, text_b)
        if v1 and v2:
            return self._clamp01(self._cosine_sim(v1, v2))
        return 0.0

    def _similarity(self, text_a: str, text_b: str) -> float:
        text_a = self._normalize_text(text_a)
        text_b = self._normalize_text(text_b)
        if not text_a or not text_b:
            return 0.0
        vec_a = self._embed_text(text_a)
        vec_b = self._embed_text(text_b)
        if vec_a and vec_b:
            return self._clamp01(self._cosine_sim(vec_a, vec_b))
        v1, v2 = self._tfidf_vector(text_a, text_b)
        if v1 and v2:
            return self._clamp01(self._cosine_sim(v1, v2))
        # Fallback to Jaccard
        tokens_a = set(re.findall(r"\b\w{4,}\b", text_a.lower()))
        tokens_b = set(re.findall(r"\b\w{4,}\b", text_b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        return self._clamp01(len(tokens_a & tokens_b) / len(tokens_a | tokens_b))
