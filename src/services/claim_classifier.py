"""
Legal Claim Classifier using Groq Cloud.

Classifies legal claims into the appropriate book (libro) of the
Italian Civil Code (Codice Civile), Penal Code (Codice Penale),
or the administrative procedure law (L. 241/1990).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from groq import Groq

# Cross-platform path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.tools import config_loader  # noqa: E402
from agents.tools.prompt_registry import get_prompt, render_prompt  # noqa: E402
from config import settings  # noqa: E402
from services.groq_client import get_groq_client, resilient_groq_call  # noqa: E402


@dataclass
class ClassificationResult:
    """Result of claim classification."""

    claim: str
    categories: list[str]
    descriptions: list[str]
    libro_mappings: list[tuple[str, str]]  # (codice, libro)

    def __str__(self) -> str:
        lines = [f"Claim: {self.claim}", "", "Classifications:"]
        for i, (cat, desc, mapping) in enumerate(
            zip(self.categories, self.descriptions, self.libro_mappings), 1
        ):
            codice, libro = mapping
            scope = libro or "N/A (intero codice)"
            lines.append(f"  {i}. {cat} -> {desc}")
            lines.append(f"     Neo4j: {codice} / {scope}")
        return "\n".join(lines)


class ClaimClassifier:
    """
    Classifies legal claims using Groq Cloud LLM.

    Uses LLM to route claims to the appropriate
    book of the Italian Civil or Penal Code.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the classifier.

        Args:
            api_key: Groq API key. If None, uses the resilient client with key rotation.
            model: Model to use for classification. If None, reads from settings.
        """
        self.api_key = api_key  # May be None → resilient client handles rotation
        if not api_key:
            if settings.llm_backend == "openrouter":
                if not settings.openrouter_api_key:
                    raise ValueError(
                        "OPENROUTER_API_KEY not found. Set it in .env "
                        "(required for LLM_BACKEND=openrouter)."
                    )
            elif settings.llm_backend != "local" and not settings.groq_api_keys:
                raise ValueError(
                    "GROQ_API_KEY not found. Set GROQ_API_KEY_V1/V2/V3 in .env."
                )

        self.client = get_groq_client(api_key=self.api_key)
        self.model = model or settings.retrieval_default_model
        self._categories = config_loader.claim_classifier_categories()
        self._few_shots = config_loader.claim_classifier_few_shots()
        self._max_categories = config_loader.claim_classifier_max_categories()

        self._category_to_libro: dict[str, tuple[str, str]] = {}
        self._category_to_description: dict[str, str] = {}
        for item in self._categories:
            if not isinstance(item, dict):
                continue
            cat_id = str(item.get("id", "")).strip()
            if not cat_id:
                continue
            codice = str(item.get("codice", "unknown")).strip() or "unknown"
            libro = str(item.get("libro", "unknown")).strip()
            description = str(item.get("description", "Unknown")).strip() or "Unknown"
            self._category_to_libro[cat_id] = (codice, libro)
            self._category_to_description[cat_id] = description

        if not self._category_to_libro:
            raise ValueError(
                "Missing claim_classifier.categories in config_taxonomy.json"
            )

        self._taxonomy_block = self._build_taxonomy_block()

    def _build_taxonomy_block(self) -> str:
        """Build taxonomy lines from config categories."""
        taxonomy_lines = []
        for item in self._categories:
            if not isinstance(item, dict):
                continue
            cat_id = str(item.get("id", "")).strip()
            description = str(item.get("description", "")).strip()
            if not cat_id:
                continue
            taxonomy_lines.append(f"{cat_id} -> {description}")
        return "\n".join(taxonomy_lines)

    @staticmethod
    def _format_example_response(raw: Any) -> str:
        """Normalize few-shot response to newline-separated taxonomy IDs."""
        if isinstance(raw, list):
            values = [str(v).strip() for v in raw if str(v).strip()]
            return "\n".join(values)
        return str(raw or "").strip()

    def _build_messages(self, claim: str) -> list[dict]:
        """Build the message chain with few-shot examples."""
        messages = [
            {"role": "system", "content": get_prompt("claim_classifier.system")}
        ]

        # Add few-shot examples
        for example in self._few_shots:
            if not isinstance(example, dict):
                continue
            example_claim = str(example.get("claim", "")).strip()
            example_response = self._format_example_response(example.get("response"))
            if not example_claim or not example_response:
                continue
            messages.append(
                {
                    "role": "user",
                    "content": render_prompt(
                        "claim_classifier.taxonomy_user",
                        taxonomy_block=self._taxonomy_block,
                        claim=example_claim,
                    ),
                }
            )
            messages.append({"role": "assistant", "content": example_response})

        # Add the actual claim
        messages.append(
            {
                "role": "user",
                "content": render_prompt(
                    "claim_classifier.taxonomy_user",
                    taxonomy_block=self._taxonomy_block,
                    claim=claim,
                ),
            }
        )

        return messages

    def classify(
        self,
        claim: str,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> ClassificationResult:
        """
        Classify a legal claim into taxonomy categories.

        Args:
            claim: The legal claim text to classify.
            temperature: LLM temperature (lower = more deterministic).
            stream: Whether to stream the response.

        Returns:
            ClassificationResult with top 3 categories.
        """
        resolved_temperature = (
            settings.classifier_temperature
            if temperature is None
            else float(temperature)
        )
        messages = self._build_messages(claim)

        if stream:
            return self._classify_stream(claim, messages, resolved_temperature)
        else:
            return self._classify_sync(claim, messages, resolved_temperature)

    def _classify_sync(
        self,
        claim: str,
        messages: list[dict],
        temperature: float,
    ) -> ClassificationResult:
        """Synchronous classification with resilient retry/key-rotation/fallback."""

        def _call(client: Groq, model: str):
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=settings.claim_classifier_max_tokens,
                top_p=1,
                stream=False,
            )

        completion = resilient_groq_call(
            _call,
            model_order=settings.retrieval_model_fallback_order,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        return self._parse_response(claim, response_text)

    def _classify_stream(
        self,
        claim: str,
        messages: list[dict],
        temperature: float,
    ) -> ClassificationResult:
        """Streaming classification with resilient retry/key-rotation/fallback."""

        def _call(client: Groq, model: str):
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=settings.claim_classifier_max_tokens,
                top_p=1,
                stream=True,
            )
            response_text = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                response_text += content
            return response_text.strip()

        response_text = resilient_groq_call(
            _call,
            model_order=settings.retrieval_model_fallback_order,
        )
        return self._parse_response(claim, response_text)

    def _parse_response(
        self,
        claim: str,
        response_text: str,
    ) -> ClassificationResult:
        """Parse LLM response into ClassificationResult."""
        # Split by newlines and filter valid categories
        lines = [line.strip() for line in response_text.split("\n")]
        categories = [line for line in lines if line in self._category_to_libro][
            : self._max_categories
        ]

        # If we got fewer than 3, that's still okay
        if not categories:
            # Fallback: try to find any valid category in response
            for cat in self._category_to_libro.keys():
                if cat in response_text:
                    categories.append(cat)
                    if len(categories) >= self._max_categories:
                        break

        descriptions = [
            self._category_to_description.get(cat, "Unknown") for cat in categories
        ]
        libro_mappings = [
            self._category_to_libro.get(cat, ("unknown", "unknown"))
            for cat in categories
        ]

        return ClassificationResult(
            claim=claim,
            categories=categories,
            descriptions=descriptions,
            libro_mappings=libro_mappings,
        )


def main():
    """Interactive CLI for testing the classifier."""
    print("=" * 60)
    print("LexCausa - Legal Claim Classifier")
    print("=" * 60)
    print()
    print("This tool classifies legal claims into Italian law categories.")
    print("Type 'quit' or 'exit' to stop.")
    print()

    try:
        classifier = ClaimClassifier()
        print("✅ Groq client initialized successfully.")
        print(f"   Model: {classifier.model}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    print()

    while True:
        print("-" * 60)
        claim = input("📝 Enter your legal claim:\n> ").strip()

        if claim.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break

        if not claim:
            print("⚠️ Please enter a valid claim.")
            continue

        print("\n🔄 Classifying...")

        try:
            result = classifier.classify(claim)
            print()
            print("📊 Result:")
            print(result)
            print()
        except Exception as e:
            print(f"❌ Error during classification: {e}")


if __name__ == "__main__":
    main()
