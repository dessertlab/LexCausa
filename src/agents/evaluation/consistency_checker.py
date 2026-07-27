"""
Consistency checking and repair for reasoning chains.

Verifies that citations in the reasoning chain match the knowledge base,
handles mismatches (repair or drop), and regenerates chains via LLM.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..aspic_formatter import AspicFormatter
from ..citation_utils import article_id_to_regex as build_article_id_regex
from ..citation_utils import (
    extract_article_mentions,
    format_article_citation,
    infer_source_hint,
    normalize_article_id,
)
from ..tools.neo4j_tools import get_driver
from ..tools.prompt_registry import render_prompt
from .models import CitationCheck, ConsistencyReport, MismatchAction

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings  # noqa: E402
from services.groq_client import get_chat_groq, resilient_chat_call  # noqa: E402


class ConsistencyMixin:
    """Mixin providing consistency-checking methods to the evaluator."""

    def _verify_statute_in_neo4j(
        self, article_num: str, domain: str, citation_str: str = ""
    ) -> tuple[bool, str, str]:
        """
        Verify if an article exists in Neo4j and return its text.

        Args:
            article_num: The article number (e.g., "1223")
            domain: Legal domain ("CIVILE", "PENALE", "AMMINISTRATIVO", or "ENTRAMBI")
            citation_str: Original citation string (e.g., "Art. 52 c.p.") used
                          to disambiguate codice when domain is ENTRAMBI.

        Returns:
            Tuple of (exists: bool, text: str, title: str).
            Text/title are empty if not found.
        """
        driver = get_driver()
        article_id = self._normalize_article_id(article_num)
        if not article_id:
            return False, "", ""

        domain = (domain or "").strip().upper()
        citation_lower = citation_str.lower() if citation_str else ""

        if citation_lower:
            if "241/1990" in citation_lower or "legge 241" in citation_lower:
                self._log(
                    f"      🔍 detected L. 241/1990 in '{citation_str}', searching AMMINISTRATIVO"
                )
                domain = "AMMINISTRATIVO"
            elif "amministrativ" in citation_lower and "codice" in citation_lower:
                self._log(
                    f"      🔍 detected codice amministrativo in '{citation_str}', searching AMMINISTRATIVO"
                )
                domain = "AMMINISTRATIVO"

        # Resolve candidate domains once, then query iteratively (no recursion).
        if domain == "ENTRAMBI" and citation_lower:
            if "c.p" in citation_lower or (
                "cod" in citation_lower and "pen" in citation_lower
            ):
                self._log(
                    f"      🔍 ENTRAMBI → detected c.p. in '{citation_str}', searching PENALE"
                )
                candidate_domains = ["PENALE"]
            elif "c.c" in citation_lower or (
                "cod" in citation_lower and "civ" in citation_lower
            ):
                self._log(
                    f"      🔍 ENTRAMBI → detected c.c. in '{citation_str}', searching CIVILE"
                )
                candidate_domains = ["CIVILE"]
            else:
                candidate_domains = ["PENALE", "CIVILE", "AMMINISTRATIVO"]
        elif domain in ("CIVILE", "PENALE", "AMMINISTRATIVO"):
            candidate_domains = [domain]
        else:
            candidate_domains = ["PENALE", "CIVILE", "AMMINISTRATIVO"]

        query = """
            MATCH (s:Statute)
            WHERE s.articolo = $articolo AND s.source = $codice
            RETURN s.articolo AS articolo, s.testo AS testo, s.titolo AS titolo
            LIMIT 1
        """

        try:
            with driver.session() as session:
                for candidate in candidate_domains:
                    if candidate == "CIVILE":
                        codice = "codice_civile"
                    elif candidate == "PENALE":
                        codice = "codice_penale"
                    else:  # AMMINISTRATIVO
                        codice = "codice_amministrativo"

                    for articolo_normalized in self._build_lookup_variants(
                        article_id, candidate
                    ):
                        result = session.run(
                            query,
                            parameters={
                                "articolo": articolo_normalized,
                                "codice": codice,
                            },
                        )
                        record = result.single()
                        if not record:
                            continue

                        testo = record.get("testo", "") or ""
                        titolo = record.get("titolo", "") or ""
                        if testo:
                            self._log(
                                f"      Neo4j: Art. {article_id} found - '{titolo[:50]}...'"
                            )
                            self._log(f"      DB text preview: '{testo[:100]}...'")
                        else:
                            self._log(
                                f"      Neo4j: Art. {article_id} found but NO TEXT in DB"
                            )
                        return True, testo, titolo
                return False, "", ""
        except Exception as e:
            self._log(f"⚠️ Neo4j query failed: {e}", "warning")
            return False, "", ""

    @staticmethod
    def _normalize_article_id(raw: str) -> str:
        """Normalize article identifiers across formats (e.g. 21 quinquies -> 21-quinquies)."""
        return normalize_article_id(raw)

    @staticmethod
    def _article_id_to_regex(article_id: str) -> str:
        """Build a regex fragment that accepts both hyphen and space suffixes."""
        return build_article_id_regex(article_id)

    @staticmethod
    def _build_lookup_variants(article_id: str, domain: str) -> list[str]:
        """Build lookup variants for article ids across DB storage formats."""
        base = article_id[3:] if article_id.startswith("art") else article_id
        base = (base or "").strip()
        compact = base.replace("-", "")

        if domain == "CIVILE":
            variants = [f"art{base}"]
            if compact and compact != base:
                variants.append(f"art{compact}")
        else:
            variants = [base]
            if compact and compact != base:
                variants.append(compact)

        return list(dict.fromkeys(v for v in variants if v))

    @staticmethod
    def _normalize_text_for_match(text: str) -> str:
        """Normalize legal text for robust matching across noisy DB formats."""
        if not text:
            return ""
        normalized = str(text)
        # Remove invisible chars frequently found in itacasehold exports.
        normalized = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", normalized)
        normalized = normalized.replace("\u00a0", " ")
        # Unify punctuation variants.
        normalized = (
            normalized.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _extract_quote_candidate(text: str) -> str:
        """Extract quoted span from model output (supports «...» and \"...\")."""
        if not text:
            return ""
        for pattern in (r"«([^»]+)»", r'"([^"]+)"'):
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return ""

    def _quote_matches_db_summary(self, quote: str, db_summary: str) -> bool:
        """
        Validate quoted text against DB summary with exact+fuzzy checks.

        Exact inclusion is preferred; fuzzy token/4-gram overlap handles
        whitespace/invisible-char noise from the precedent dataset.
        """
        quote_norm = self._normalize_text_for_match(quote).lower()
        db_norm = self._normalize_text_for_match(db_summary).lower()
        if not quote_norm or not db_norm:
            return False

        quote_tokens = re.findall(r"\b\w+\b", quote_norm)
        if len(quote_tokens) < 8:
            return False

        if quote_norm in db_norm:
            return True

        db_tokens = set(re.findall(r"\b\w+\b", db_norm))
        quote_set = set(quote_tokens)
        if not quote_set:
            return False

        token_recall = len(quote_set & db_tokens) / len(quote_set)
        if token_recall >= 0.9 and len(quote_tokens) >= 10:
            return True

        n = 4
        quote_ngrams = {
            " ".join(quote_tokens[i : i + n])
            for i in range(0, max(len(quote_tokens) - n + 1, 1))
        }
        if not quote_ngrams:
            return False
        matched_ngrams = sum(1 for ng in quote_ngrams if ng in db_norm)
        ngram_recall = matched_ngrams / len(quote_ngrams)
        return ngram_recall >= 0.6

    def _build_precedent_fallback_repair(
        self, precedent_title: str, db_summary: str
    ) -> str:
        """Build deterministic repaired precedent citation from DB summary."""
        title = self._normalize_text_for_match(precedent_title)
        summary = self._normalize_text_for_match(db_summary)
        if not summary:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", summary)
        holding = ""
        for sentence in sentences:
            if len(sentence.split()) >= 12:
                holding = sentence.strip()
                break
        if not holding:
            holding = summary[:420].rstrip(" ,;:")

        if len(holding) > 450:
            holding = holding[:450].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."

        return f"Come evidenziato dalla giurisprudenza in «{title}», " f"«{holding}»."

    def _extract_article_id_from_citation(self, citation: str) -> str:
        """Extract full article id from a citation (supports suffixes like -bis/-quinquies)."""
        if not citation:
            return ""
        mentions = extract_article_mentions(citation, require_code=False)
        if mentions:
            return mentions[0].article_id
        return ""

    @staticmethod
    def _infer_source_hint_from_citation(citation: str) -> str:
        """Infer statute source from citation text."""
        mentions = extract_article_mentions(citation, require_code=False)
        if mentions and mentions[0].source_hint:
            return mentions[0].source_hint

        lower = (citation or "").lower()
        if (
            "241/1990" in lower
            or "legge 241" in lower
            or "codice amministrativ" in lower
        ):
            return "codice_amministrativo"
        if "c.c" in lower or ("cod" in lower and "civ" in lower):
            return "codice_civile"
        if "c.p" in lower or ("cod" in lower and "pen" in lower):
            return "codice_penale"
        return infer_source_hint(lower)

    def _extract_cited_text_for_article_with_source(
        self, full_text: str, article_num: str, aspic_ir: dict | None = None
    ) -> tuple[str, str]:
        """
        Extract cited text and report the source pattern used.

        Returns:
            Tuple(extracted_text, source_pattern)
            source_pattern in {"pattern0","pattern1","pattern2","pattern3",
            "pattern4","none"}.
        """
        article_id = self._normalize_article_id(article_num)
        article_pattern = self._article_id_to_regex(article_id or article_num)
        article_anchor = rf"{article_pattern}(?![-\w])"
        code_pattern = (
            r"(?:c\.?\s*c\.?|c\.?\s*p\.?|"
            r"l\.?\s*241(?:\s*/\s*1990)?|legge\s*241(?:\s*/\s*1990)?|"
            r"cod(?:ice)?\.?\s*amm(?:inistrativ[oa])?)?"
        )

        # Pattern 0: block with Norma + Testo lines
        block_pattern = (
            rf"Norma.*?Art(?:icolo)?\.?\s*{article_anchor}.*?\n"
            rf".*?Testo\s*:\s*\"([^\"]{{20,}})\""
        )
        match = re.search(block_pattern, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r"\s+", " ", extracted)
            if len(extracted) >= 20:
                self._log("      🎯 Found 'Norma + Testo' block pattern")
                return extracted, "pattern0"

        # Pattern 1: "L'Art. 1223 c.c. stabilisce/prevede/limita che..."
        # Captures the verb + "che" + text until period
        pattern1 = rf"[Ll][''']?Art(?:icolo)?\.?\s*{article_anchor}\s*{code_pattern}\s+(?:stabilisce|prevede|limita|dispone|sancisce|afferma)\s+(?:che\s+)?(.+?)(?:\.|$)"
        match = re.search(pattern1, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r"\s+", " ", extracted)
            if len(extracted) >= 20:
                self._log(
                    "      🎯 Found 'Art. "
                    + str(article_num)
                    + " stabilisce/prevede...' pattern"
                )
                return extracted, "pattern1"

        # Pattern 2: "Art. 1223 c.c. - [testo]"
        pattern2 = rf"Art(?:icolo)?\.?\s*{article_anchor}\s*{code_pattern}\s*[\-:]\s*(.+?)(?:\.|$)"
        match = re.search(pattern2, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r"\s+", " ", extracted)
            # Avoid if it contains other article references (means it's a list)
            if not re.search(
                r"Art(?:icolo)?\.?\s*\d{1,4}(?:[-\s]?[a-z0-9]+)?",
                extracted,
                re.IGNORECASE,
            ):
                if len(extracted) >= 20:
                    self._log(
                        "      🎯 Found 'Art. " + str(article_num) + " - ...' pattern"
                    )
                    return extracted, "pattern2"

        # Pattern 3: "Secondo l'Art. 1223 c.c., [testo]"
        pattern3 = rf"[Ss]econdo\s+[Ll][''']?Art(?:icolo)?\.?\s*{article_anchor}\s*{code_pattern},?\s+(.+?)(?:\.|$)"
        match = re.search(pattern3, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r"\s+", " ", extracted)
            if len(extracted) >= 20:
                self._log(
                    "      🎯 Found 'Secondo l'Art. "
                    + str(article_num)
                    + "...' pattern"
                )
                return extracted, "pattern3"

        # Pattern 4: "(Art. 1223 c.c.)" in parentheses after text - capture text before
        pattern4 = (
            rf"([^.]+?)\s*\(Art(?:icolo)?\.?\s*{article_anchor}\s*{code_pattern}\)"
        )
        match = re.search(pattern4, full_text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            # Take last sentence before the article reference
            sentences = extracted.split(".")
            if sentences:
                last_sentence = sentences[-1].strip()
                if len(last_sentence) >= 20:
                    self._log(
                        "      🎯 Found text before '(Art. "
                        + str(article_num)
                        + ")' pattern"
                    )
                    return last_sentence, "pattern4"

        self._log("      ⚠️ No text found for Art. " + str(article_num))
        return "", "none"

    def _compute_text_similarity(self, cited_text: str, db_text: str) -> float:
        """
        Compute similarity between cited text and database text.

        Uses a simple approach: check if key phrases from cited text appear in db_text.

        Args:
            cited_text: Text extracted from reasoning chain
            db_text: Text from database

        Returns:
            Similarity score from 0.0 to 1.0
        """
        if not cited_text or not db_text:
            return 0.0

        # Normalize both texts
        cited_lower = self._normalize_text_for_match(cited_text).lower()
        db_lower = self._normalize_text_for_match(db_text).lower()

        # Check direct substring match
        if cited_lower in db_lower:
            return 1.0

        # Extract meaningful words (>3 chars)
        cited_words = set(w for w in re.findall(r"\b\w{4,}\b", cited_lower))
        db_words = set(w for w in re.findall(r"\b\w{4,}\b", db_lower))

        if not cited_words:
            return 0.0

        # Calculate recall-oriented overlap and blend with F1
        common_words = cited_words & db_words
        recall = len(common_words) / len(cited_words)
        precision = len(common_words) / len(db_words) if db_words else 0.0
        if recall + precision == 0:
            similarity = 0.0
        else:
            f1 = 2 * recall * precision / (recall + precision)
            similarity = max(recall, f1)

        return min(similarity, 1.0)

    def _verify_mismatch_with_llm(
        self,
        article_num: str,
        cited_text: str,
        db_text: str,
    ) -> bool:
        """
        Use LLM to verify if the cited text is logically different from DB text.

        Args:
            article_num: Article number for context
            cited_text: Text extracted from reasoning chain
            db_text: Official text from database

        Returns:
            True if LLM confirms the texts are logically different (mismatch).
        """
        try:
            llm = get_chat_groq(
                temperature=settings.classifier_temperature,
                max_tokens=settings.repair_max_tokens,
            )

            system_prompt = render_prompt("consistency.verify_mismatch_system")

            user_prompt = render_prompt(
                "consistency.verify_mismatch_user",
                article_num=article_num,
                cited_text=cited_text,
                db_text=db_text,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = resilient_chat_call(llm, messages)
            answer = response.content.strip().upper()

            is_different = "DIVERSI" in answer
            self._log(
                f"      🤖 LLM mismatch verification: {'DIVERSI ⚠️' if is_different else 'EQUIVALENTI ✅'}"
            )
            return is_different

        except Exception as e:
            self._log(f"      ⚠️ LLM verification failed: {e}", "warning")
            # In case of error, assume mismatch to be conservative
            return True

    def _check_pertinence_with_llm(
        self,
        article_num: str,
        cited_text: str,
        full_text: str,
        db_text: str = "",
        db_title: str = "",
    ) -> bool:
        """
        Use LLM to check if a peripheral norm is still logically pertinent
        to the reasoning chain, even if not core.

        A norm is PERTINENT if it contributes meaningfully to the legal
        reasoning — e.g., it provides context, reinforces a conclusion,
        or integrates the normative framework. It should be kept and repaired.

        A norm is NOT PERTINENT if it is tangential, redundant, or does not
        add value to the argument. It can safely be dropped.

        Args:
            article_num: Article number
            cited_text: Text cited in the reasoning chain for this article
            full_text: Full reasoning chain text
            db_text: Official DB text (optional, strengthens pertinence check)
            db_title: Official DB title (optional)

        Returns:
            True if the norm is pertinent (should be repaired),
            False if not pertinent (can be dropped).
        """
        try:
            llm = get_chat_groq(
                temperature=settings.classifier_temperature,
                max_tokens=settings.repair_max_tokens,
            )

            system_prompt = render_prompt("consistency.pertinence_system")

            user_prompt = render_prompt(
                "consistency.pertinence_user",
                full_text=full_text[: settings.truncation_chain_text],
                article_num=article_num,
                cited_text=cited_text,
                db_title=(db_title or "")[: settings.truncation_consistency_db_title],
                db_text=(db_text or "")[: settings.truncation_consistency_db_text],
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = resilient_chat_call(llm, messages)
            answer = response.content.strip().upper()

            is_pertinent = (
                "NOT_PERTINENT" not in answer and "NOT PERTINENT" not in answer
            )
            self._log(
                f"      🤖 LLM pertinence check Art. {article_num}: "
                f"{'PERTINENTE ✅' if is_pertinent else 'NON PERTINENTE 🗑️'}"
            )
            return is_pertinent

        except Exception as e:
            self._log(f"      ⚠️ LLM pertinence check failed: {e}", "warning")
            # In case of error, assume pertinent to be conservative (repair > drop)
            return True

    def _pre_repair_statute_relevance_gate(
        self,
        *,
        claim: str,
        full_text: str,
        article_num: str,
        citation: str,
        cited_text: str,
        db_text: str,
        db_title: str = "",
    ) -> tuple[bool, str]:
        """
        Relevance gate executed BEFORE any textual repair/drop workflow.

        Returns:
            (is_relevant, label) where label is one of:
            - RELEVANT
            - IRRELEVANT
            - UNCERTAIN_KEEP (on checker failures)
        """
        cache: dict[tuple[str, str, str], tuple[bool, str]] = getattr(
            self, "_statute_relevance_gate_cache", {}
        )
        if not hasattr(self, "_statute_relevance_gate_cache"):
            self._statute_relevance_gate_cache = cache

        claim_sig = self._normalize_text_for_match(claim or "")[
            : settings.truncation_consistency_signature
        ]
        db_sig = self._normalize_text_for_match(db_text or "")[
            : settings.truncation_consistency_signature
        ]
        key = (str(article_num).strip(), claim_sig, db_sig)
        if key in cache:
            return cache[key]

        # Build a "case-aware" context without changing the old pertinence helper.
        case_context = "\n".join(
            part
            for part in [
                f"CLAIM: {claim.strip()}" if (claim or "").strip() else "",
                f"CHAIN: {full_text.strip()}" if (full_text or "").strip() else "",
            ]
            if part
        ).strip()
        if not case_context:
            case_context = (full_text or claim or "").strip()

        evidence = (cited_text or citation or "").strip()
        if not evidence and db_text:
            evidence = db_text[: settings.truncation_consistency_signature]
        cited_for_gate = evidence or f"Art. {article_num}"
        try:
            pertinent = self._check_pertinence_with_llm(
                article_num=article_num,
                cited_text=cited_for_gate,
                full_text=case_context,
                db_text=db_text,
                db_title=db_title,
            )
            result = (bool(pertinent), "RELEVANT" if pertinent else "IRRELEVANT")
        except Exception:
            # Conservative fallback: keep citation for downstream repair logic.
            result = (True, "UNCERTAIN_KEEP")

        cache[key] = result
        return result

    def _is_article_core(
        self,
        article_num: str,
        aspic_ir: dict | None,
        full_text: str,
    ) -> bool:
        """
        Determine if an article is CORE (essential) or PERIPHERAL to the argument.

        An article is CORE if:
        - It's the only normative support for a conclusion
        - It appears in the main norm/rule of an argument
        - It's cited multiple times in the reasoning chain
        - It's in the conclusion's citations

        Args:
            article_num: Article number to check
            aspic_ir: ASPIC IR structure
            full_text: Full text of reasoning chain

        Returns:
            True if the article is core, False if peripheral.
        """
        core_indicators = 0

        # Check ASPIC IR for core indicators
        if aspic_ir:
            for arg in aspic_ir.get("arguments", []):
                # Check premises with type="norm"
                norm_premises = [
                    p for p in arg.get("premises", []) if p.get("type") == "norm"
                ]

                for premise in norm_premises:
                    citations = premise.get("citations", {})
                    statutes = citations.get("statutes", [])

                    # Article is in a norm premise
                    article_cited = any(
                        str(s.get("articolo", "")).strip() == article_num
                        for s in statutes
                    )

                    if article_cited:
                        # If it's the ONLY statute in a norm premise → definitely core
                        if len(statutes) == 1:
                            self._log(
                                f"      🎯 Art. {article_num} is CORE: only statute in norm premise"
                            )
                            return True
                        core_indicators += 1

                # Check conclusion citations
                conclusion = arg.get("conclusion", {})
                if conclusion:
                    conclusion_citations = conclusion.get("citations", {})
                    for s in conclusion_citations.get("statutes", []):
                        if str(s.get("articolo", "")).strip() == article_num:
                            core_indicators += (
                                settings.cc_conclusion_bonus
                            )  # In conclusion = more important
                            self._log(
                                f"      🎯 Art. {article_num} cited in conclusion"
                            )

        # Count occurrences in full text
        pattern = rf"art(?:icolo)?\.?\s*{article_num}"
        occurrences = len(re.findall(pattern, full_text, re.IGNORECASE))
        if occurrences >= settings.cc_occurrence_threshold:
            core_indicators += 1
            self._log(
                f"      📊 Art. {article_num} appears {occurrences} times in text"
            )

        # Threshold: configurable via settings.cc_core_threshold
        is_core = core_indicators >= settings.cc_core_threshold
        self._log(
            f"      {'🎯 CORE' if is_core else '📎 PERIPHERAL'}: Art. {article_num} (indicators: {core_indicators})"
        )
        return is_core

    def _repair_with_db_constraint(
        self,
        article_num: str,
        db_text: str,
        original_context: str,
    ) -> tuple[bool, str]:
        """
        Attempt to repair the normative citation using the official DB text.

        The LLM must rewrite the norm/causal link using ONLY the DB text,
        and include a verbatim quote from the DB.

        Args:
            article_num: Article number
            db_text: Official text from database
            original_context: Original context where the article was used

        Returns:
            Tuple of (success: bool, repaired_text: str).
            Success is True only if the repaired text contains a verbatim DB quote.
        """
        try:
            llm = get_chat_groq(
                temperature=settings.classifier_temperature,
                max_tokens=settings.repair_max_tokens,
            )

            system_prompt = render_prompt("consistency.repair_db_system")

            user_prompt = render_prompt(
                "consistency.repair_db_user",
                article_num=article_num,
                db_text=db_text,
                original_context=original_context[: settings.truncation_context],
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = resilient_chat_call(llm, messages)
            repaired_text = response.content.strip()

            # Validate: check if there's a verbatim quote from DB
            # Extract text between «»
            quote_match = re.search(r"«([^»]+)»", repaired_text)
            if quote_match:
                quote = quote_match.group(1).lower().strip()
                db_lower = db_text.lower()

                # Check if quote is a substring of DB text
                if quote in db_lower and len(quote) >= 15:
                    self._log(
                        f"      ✅ Repair SUCCESS: verbatim quote found ({len(quote)} chars)"
                    )
                    return True, repaired_text
                else:
                    self._log(
                        "      ❌ Repair FAILED: quote not found in DB or too short"
                    )
                    return False, ""
            else:
                self._log("      ❌ Repair FAILED: no «» quote in response")
                return False, ""

        except Exception as e:
            self._log(f"      ⚠️ Repair failed with error: {e}", "warning")
            return False, ""

    def _handle_norm_mismatch(
        self,
        check: CitationCheck,
        article_num: str,
        cited_text: str,
        db_text: str,
        aspic_ir: dict | None,
        full_text: str,
        report: ConsistencyReport,
        db_title: str = "",
        precomputed_is_core: bool | None = None,
        prechecked_pertinent: bool | None = None,
    ) -> None:
        """
        Handle a normative mismatch by verifying with LLM and taking appropriate action.

        Flow:
        1. Verify mismatch with LLM
        2. If confirmed, classify as core/peripheral
        3. If core: attempt repair with DB constraint
        4. If peripheral: check pertinence with LLM
           4a. If pertinent: attempt repair (keep useful norms)
           4b. If not pertinent: drop the citation
        5. If repair fails (core or peripheral): drop the citation

        Args:
            check: CitationCheck object to update
            article_num: Article number
            cited_text: Cited text from reasoning
            db_text: Official DB text
            aspic_ir: ASPIC IR structure
            full_text: Full reasoning text
            report: ConsistencyReport to update
            db_title: Official DB title (optional)
            precomputed_is_core: If provided, skips core re-classification
            prechecked_pertinent: If provided for peripheral norms, skips
                pertinence re-check.
        """
        self._log(f"      🔧 Handling mismatch for Art. {article_num}...")

        # Step 1: Verify mismatch with LLM
        llm_confirmed = self._verify_mismatch_with_llm(article_num, cited_text, db_text)
        check.llm_mismatch_confirmed = llm_confirmed

        if not llm_confirmed:
            # LLM says they're equivalent → treat as match despite low similarity
            self._log("      ✅ LLM says texts are equivalent - treating as match")
            check.mismatch_action = MismatchAction.MATCH.value
            check.text_match = True
            check.llm_validated = True  # Mark as LLM-rescued
            report.text_matches += 1
            report.text_mismatches -= 1  # Correct the earlier increment
            return

        # Step 2: Classify as core or peripheral
        if precomputed_is_core is None:
            is_core = self._is_article_core(article_num, aspic_ir, full_text)
        else:
            is_core = bool(precomputed_is_core)
        check.is_core = is_core

        if is_core:
            # Step 3: Attempt repair
            self._log("      🔄 Attempting repair for CORE article...")
            success, repaired_text = self._repair_with_db_constraint(
                article_num, db_text, cited_text
            )

            if success:
                check.mismatch_action = MismatchAction.REPAIRED.value
                check.repaired_text = repaired_text
                check.repair_success = True
                check.details += " [REPAIRED with DB text]"
                report.repaired_citations += 1
                self._log(f"      ✅ Art. {article_num} REPAIRED successfully")
            else:
                check.mismatch_action = MismatchAction.REPAIR_FAILED.value
                check.repair_success = False
                check.details += " [REPAIR FAILED - citation unreliable]"
                report.dropped_citations += 1
                report.issues.append(
                    f"Art. {article_num}: CORE citation repair failed - argument may be invalid"
                )
                self._log(
                    f"      ❌ Art. {article_num} repair FAILED - marked as unreliable"
                )
        else:
            # Peripheral: check if it's still pertinent to the reasoning
            self._log(
                f"      🔍 Art. {article_num} is PERIPHERAL - checking pertinence..."
            )
            if prechecked_pertinent is None:
                is_pertinent = self._check_pertinence_with_llm(
                    article_num,
                    cited_text,
                    full_text,
                    db_text=db_text,
                    db_title=db_title,
                )
            else:
                is_pertinent = bool(prechecked_pertinent)

            if is_pertinent:
                # Pertinent peripheral norm → attempt repair (keep it)
                self._log(
                    f"      🔄 Art. {article_num} is PERTINENT - attempting repair..."
                )
                success, repaired_text = self._repair_with_db_constraint(
                    article_num, db_text, cited_text
                )

                if success:
                    check.mismatch_action = MismatchAction.REPAIRED.value
                    check.repaired_text = repaired_text
                    check.repair_success = True
                    check.details += " [REPAIRED - pertinent peripheral citation]"
                    report.repaired_citations += 1
                    self._log(
                        f"      ✅ Art. {article_num} REPAIRED (pertinent peripheral)"
                    )
                else:
                    # Repair failed even for pertinent norm → drop
                    check.mismatch_action = MismatchAction.REPAIR_FAILED.value
                    check.repair_success = False
                    check.details += (
                        " [REPAIR FAILED - pertinent peripheral, repair unsuccessful]"
                    )
                    report.dropped_citations += 1
                    report.issues.append(
                        f"Art. {article_num}: pertinent peripheral citation repair failed"
                    )
                    self._log(
                        f"      ❌ Art. {article_num} repair FAILED (pertinent peripheral)"
                    )
            else:
                # Not pertinent → safe to drop
                check.mismatch_action = MismatchAction.DROPPED.value
                check.details += " [DROPPED - non-pertinent peripheral citation]"
                report.dropped_citations += 1
                self._log(
                    f"      🗑️ Art. {article_num} DROPPED (non-pertinent peripheral)"
                )

    # ------------------------------------------------------------------
    # Precedent verification helpers
    # ------------------------------------------------------------------

    def _verify_precedent_in_neo4j(
        self, title: str, precedent_id: str | None = None
    ) -> tuple[bool, str]:
        """
        Verify if a precedent exists in Neo4j and return its summary.

        Prefers lookup by ``precedent_id`` (exact, no whitespace issues).
        Falls back to a case-insensitive, trim-safe title comparison so
        that trailing whitespace or casing differences do not cause false
        negatives.

        Returns:
            Tuple of (exists: bool, summary: str). Summary is empty if
            not found.
        """
        driver = get_driver()

        # --- Strategy 1: lookup by precedent_id (most reliable) ---
        if precedent_id:
            query_by_id = """
                MATCH (p:Precedent)
                WHERE p.precedent_id = $id
                RETURN p.precedent_id AS id,
                       p.title AS title,
                       p.summary AS summary
                LIMIT 1
            """
            try:
                with driver.session() as session:
                    result = session.run(query_by_id, parameters={"id": precedent_id})
                    record = result.single()
                    if record:
                        summary = self._normalize_text_for_match(
                            record.get("summary", "") or ""
                        )
                        title_clean = self._normalize_text_for_match(
                            record.get("title", "") or ""
                        )
                        self._log(
                            f"      🗄️ Neo4j: precedent found by ID ({precedent_id}) - "
                            f"'{title_clean[:60]}...'"
                        )
                        return True, summary
            except Exception as e:
                self._log(f"⚠️ Neo4j precedent query by ID failed: {e}", "warning")

        # --- Strategy 2: fallback to title with trim() for whitespace safety ---
        query_by_title = """
            MATCH (p:Precedent)
            WHERE toLower(trim(p.title)) = toLower(trim($title))
            RETURN p.precedent_id AS id,
                   p.title AS title,
                   p.summary AS summary
            LIMIT 1
        """
        try:
            with driver.session() as session:
                result = session.run(query_by_title, parameters={"title": title})
                record = result.single()
                if record:
                    summary = self._normalize_text_for_match(
                        record.get("summary", "") or ""
                    )
                    title_clean = self._normalize_text_for_match(
                        record.get("title", "") or ""
                    )
                    self._log(
                        f"      🗄️ Neo4j: precedent found by title - "
                        f"'{title_clean[:60]}...'"
                    )
                    return True, summary
                return False, ""
        except Exception as e:
            self._log(f"⚠️ Neo4j precedent query failed: {e}", "warning")
            return False, ""

    def _extract_cited_text_for_precedent(
        self, full_text: str, precedent_title: str
    ) -> str:
        """
        Extract the text the LLM cited around a precedent title.

        Looks for patterns like:
        - «Titolo del precedente», il quale stabilisce che [testo]
        - Come confermato in «Titolo del precedente», [testo]
        - ... «Titolo del precedente» ... (sentence containing the title)

        Returns the surrounding sentence/context or empty string.
        """
        precedent_title = self._normalize_text_for_match(precedent_title)
        title_lower = precedent_title.lower()
        text_lower = full_text.lower()

        idx = text_lower.find(title_lower)
        if idx == -1:
            self._log(
                f"      ⚠️ Precedent title not found in text: "
                f"'{precedent_title[:60]}...'"
            )
            return ""

        # Extract a window around the title mention (the full sentence)
        # Go backwards to find sentence start
        start = max(0, idx - 200)
        end = min(len(full_text), idx + len(precedent_title) + 300)

        # Try to find sentence boundaries
        window = full_text[start:end]

        # Find the sentence containing the title
        sentences = re.split(r"(?<=[.!?])\s+", window)
        for sentence in sentences:
            if title_lower in sentence.lower():
                cleaned = sentence.strip()
                if len(cleaned) >= 20:
                    cleaned = self._normalize_text_for_match(cleaned)
                    self._log(
                        f"      🎯 Extracted precedent context: " f"'{cleaned[:80]}...'"
                    )
                    return cleaned

        # Fallback: return the raw window
        raw = window.strip()
        if len(raw) >= 20:
            return self._normalize_text_for_match(raw)
        return ""

    def _handle_precedent_mismatch(
        self,
        check: CitationCheck,
        precedent_title: str,
        cited_text: str,
        db_summary: str,
        full_text: str,
        report: ConsistencyReport,
    ) -> None:
        """
        Handle a precedent citation mismatch.

        Simpler than the statute flow because precedents are always
        treated as *pertinent* (they were explicitly cited by the LLM)
        and repair always uses the DB summary.

        Flow:
        1. Verify mismatch with LLM
        2. If confirmed → repair by rewriting with correct DB summary
        3. If repair fails → drop the citation
        """
        self._log(
            f"      🔧 Handling precedent mismatch for "
            f"'{precedent_title[:50]}...'..."
        )

        # Step 1: Verify mismatch with LLM (reuse the existing method
        # but with "Precedent" context instead of article number)
        try:
            llm = get_chat_groq(
                temperature=settings.classifier_temperature,
                max_tokens=settings.repair_max_tokens,
            )
            system_prompt = render_prompt("consistency.precedent_mismatch_system")
            user_prompt = render_prompt(
                "consistency.precedent_mismatch_user",
                cited_text=cited_text,
                db_summary=db_summary[
                    : settings.truncation_consistency_db_summary_mismatch
                ],
            )
            response = resilient_chat_call(
                llm,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
            )
            answer = response.content.strip().upper()
            llm_confirmed = "DIVERSI" in answer
        except Exception as e:
            self._log(
                f"      ⚠️ LLM precedent mismatch check failed: {e}",
                "warning",
            )
            llm_confirmed = True

        check.llm_mismatch_confirmed = llm_confirmed

        if not llm_confirmed:
            self._log(
                "      ✅ LLM says precedent texts are equivalent "
                "- treating as match"
            )
            check.mismatch_action = MismatchAction.MATCH.value
            check.text_match = True
            check.llm_validated = True
            report.text_matches += 1
            report.text_mismatches -= 1
            return

        # Step 2: Attempt repair — rewrite the passage with the DB summary
        try:
            llm = get_chat_groq(
                temperature=settings.classifier_temperature,
                max_tokens=settings.repair_max_tokens,
            )
            system_prompt = render_prompt("consistency.precedent_repair_system")
            user_prompt = render_prompt(
                "consistency.precedent_repair_user",
                precedent_title=precedent_title,
                db_summary=db_summary[
                    : settings.truncation_consistency_db_summary_repair
                ],
                cited_text=cited_text[: settings.truncation_context],
            )
            response = resilient_chat_call(
                llm,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
            )
            repaired_text = self._normalize_text_for_match(response.content.strip())

            # Validate quote with robust normalization/fuzzy matching.
            quote = self._extract_quote_candidate(repaired_text)
            if quote and self._quote_matches_db_summary(quote, db_summary):
                check.mismatch_action = MismatchAction.REPAIRED.value
                check.repaired_text = repaired_text
                check.repair_success = True
                check.details += " [REPAIRED with DB summary]"
                report.repaired_citations += 1
                self._log(
                    f"      ✅ Precedent REPAIRED successfully "
                    f"(quote {len(quote)} chars)"
                )
                return

            # Fallback acceptance: high semantic overlap even without valid quote.
            repaired_similarity = self._compute_text_similarity(
                repaired_text, db_summary
            )
            if repaired_similarity >= 0.75:
                check.mismatch_action = MismatchAction.REPAIRED.value
                check.repaired_text = repaired_text
                check.repair_success = True
                check.details += f" [REPAIRED with DB summary - semantic overlap {repaired_similarity:.0%}]"
                report.repaired_citations += 1
                self._log(
                    f"      ✅ Precedent REPAIRED (semantic overlap {repaired_similarity:.0%})"
                )
                return

            self._log(
                "      ⚠️ Precedent LLM repair not valid enough, trying DB fallback"
            )
        except Exception as e:
            self._log(f"      ⚠️ Precedent repair failed: {e}", "warning")

        # Deterministic fallback using DB summary snippet (prevents noisy false drops).
        fallback_text = self._build_precedent_fallback_repair(
            precedent_title, db_summary
        )
        if fallback_text:
            check.mismatch_action = MismatchAction.REPAIRED.value
            check.repaired_text = fallback_text
            check.repair_success = True
            check.details += " [REPAIRED with deterministic DB fallback]"
            report.repaired_citations += 1
            self._log("      ✅ Precedent REPAIRED with deterministic DB fallback")
            return

        # Repair failed → drop (only when DB summary unavailable/empty).
        check.mismatch_action = MismatchAction.REPAIR_FAILED.value
        check.repair_success = False
        check.details += " [REPAIR FAILED - precedent citation unreliable]"
        report.dropped_citations += 1
        report.issues.append(
            f"Precedent '{precedent_title[:40]}': citation repair failed"
        )
        self._log(f"      ❌ Precedent '{precedent_title[:40]}' repair FAILED")

    def _check_consistency(
        self,
        agent: str,
        reasoning_chain: list[str],
        raw_response: str,
        domain: str,
        claim: str = "",
        aspic_ir: dict | None = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> ConsistencyReport:
        """
        Check the consistency of a reasoning chain by verifying citations in Neo4j.

        Performs two checks:
        1. Existence check: Does the article exist in Neo4j?
        2. Text verification: Does the cited text match the actual article text?

        Args:
            agent: Name of the agent ("reasoner" or "counter_reasoner")
            reasoning_chain: List of reasoning steps
            raw_response: Raw LLM response text
            domain: Legal domain ("CIVILE", "PENALE", "AMMINISTRATIVO", or "ENTRAMBI")
            claim: Original legal claim (used by relevance gate)
            aspic_ir: ASPIC IR structured output for text extraction

        Returns:
            ConsistencyReport with verification results.
        """
        report = ConsistencyReport(agent=agent)

        def _emit_citation_progress(check_obj: CitationCheck) -> None:
            if not progress_callback:
                return
            try:
                progress_callback(
                    {
                        "agent": agent,
                        "check": check_obj.to_dict(),
                        "totals": {
                            "processed": report.total_citations,
                            "valid": report.valid_citations,
                            "invalid": report.invalid_citations,
                            "text_matches": report.text_matches,
                            "text_mismatches": report.text_mismatches,
                            "repaired": report.repaired_citations,
                            "dropped": report.dropped_citations,
                            "expected_total": expected_total_checks,
                        },
                    }
                )
            except Exception:
                # Progress callback failures must not affect consistency logic.
                pass

        # Parse citations only from the original reasoning steps.
        # This avoids pulling artifacts from repaired/enriched text blocks.
        reasoning_text = "\n".join(reasoning_chain or [])
        citation_source_text = reasoning_text.strip() or (raw_response or "")
        verification_text = (
            reasoning_text + "\n" + raw_response
        ).strip() or citation_source_text

        # Extract statute citations from the text
        statute_citations = self._extract_statute_citations(citation_source_text)
        self._log(
            f"📜 [{agent}] Found {len(statute_citations)} statute citations to verify"
        )

        # Prepare precedent candidates early so progress events can expose
        # an accurate total expected checks count for live progress bars.
        precedent_titles_in_ir: list[dict] = []
        if aspic_ir:
            for p in aspic_ir.get("sources", {}).get("precedents", []):
                title = (p.get("title") or "").strip()
                if title:
                    precedent_titles_in_ir.append(p)

        # Also collect titles from precedent_nodes (richer data)
        prec_node_map: dict[str, dict] = {}
        if aspic_ir:
            for pn in aspic_ir.get("precedent_nodes", []):
                title = (pn.get("title") or "").strip()
                if title:
                    prec_node_map[title.lower()] = pn

        # Build a list of all known precedent titles (from sources)
        # and a title → precedent_id map for reliable Neo4j lookup
        known_prec_titles: list[str] = []
        known_prec_titles_seen: set[str] = set()
        prec_id_by_title: dict[str, str] = {}
        for p in precedent_titles_in_ir:
            title = (p.get("title") or "").strip()
            title_key = title.lower()
            if title and title_key not in known_prec_titles_seen:
                known_prec_titles_seen.add(title_key)
                known_prec_titles.append(title)
                prec_id = (p.get("precedent_id") or "").strip()
                if prec_id:
                    prec_id_by_title[title_key] = prec_id

        # Enrich map from precedent_nodes (they carry precedent_id too)
        for pn in prec_node_map.values():
            title = (pn.get("title") or "").strip().lower()
            prec_id = (pn.get("precedent_id") or "").strip()
            if title and prec_id and title not in prec_id_by_title:
                prec_id_by_title[title] = prec_id

        # Find which precedents are actually cited in the original reasoning steps.
        citation_source_lower = citation_source_text.lower()
        cited_prec_titles = [
            t for t in known_prec_titles if t.lower() in citation_source_lower
        ]
        parsed_statute_citations: list[tuple[str, str, str]] = []
        for citation in statute_citations:
            article_id = self._extract_article_id_from_citation(citation)
            if not article_id:
                continue
            source_hint = self._infer_source_hint_from_citation(citation)
            parsed_statute_citations.append((citation, article_id, source_hint))

        unique_statute_citations = {(a, s) for _, a, s in parsed_statute_citations}
        expected_total_checks = len(unique_statute_citations) + len(cited_prec_titles)

        # Track already verified articles to avoid duplicates
        verified_articles: set[tuple[str, str]] = set()
        verified_precedent_titles: set[str] = set()

        for citation, article_num, source_hint in parsed_statute_citations:
            article_key = (article_num, source_hint)

            # Skip if already verified
            if article_key in verified_articles:
                self._log(
                    f"   ⏭️ Art. {article_num}"
                    f"{' (' + source_hint + ')' if source_hint else ''} already verified, skipping"
                )
                continue
            verified_articles.add(article_key)

            # Verify existence in Neo4j and get text
            found, db_text, db_title = self._verify_statute_in_neo4j(
                article_num, domain, citation
            )

            # Initialize check object
            check = CitationCheck(
                citation=citation,
                found_in_kb=found,
                source_type="statute",
            )

            if found:
                report.valid_citations += 1
                self._log(f"   ✅ Art. {article_num} -> EXISTS in Neo4j")

                # Extract cited text from the reasoning chain / raw response
                cited_text, cited_source = (
                    self._extract_cited_text_for_article_with_source(
                        verification_text, article_num, aspic_ir
                    )
                )

                if cited_text:
                    self._log(f"      📖 Cited text extracted: '{cited_text[:80]}...'")
                    self._log(f"      🔎 Citation extraction source: {cited_source}")

                # Classify core/peripheral BEFORE applying any relevance gate.
                # Core norms must never be dropped by the pre-repair gate.
                is_core = self._is_article_core(
                    article_num, aspic_ir, verification_text
                )
                check.is_core = is_core

                run_relevance_gate = (
                    not is_core and bool(cited_text) and cited_source != "pattern4"
                )
                is_relevant = True
                relevance_label = "SKIPPED"
                if run_relevance_gate:
                    is_relevant, relevance_label = (
                        self._pre_repair_statute_relevance_gate(
                            claim=claim,
                            full_text=verification_text,
                            article_num=article_num,
                            citation=citation,
                            cited_text=cited_text,
                            db_text=db_text,
                            db_title=db_title,
                        )
                    )
                elif not is_core:
                    if not cited_text:
                        relevance_label = "SKIPPED_NO_EVIDENCE_KEEP"
                    elif cited_source == "pattern4":
                        relevance_label = "SKIPPED_FRAGILE_EVIDENCE_KEEP"
                    self._log(
                        f"      ↩️ Relevance gate skipped for Art. {article_num} "
                        f"(label={relevance_label})"
                    )

                if not is_relevant:
                    check.text_verified = bool(cited_text or db_text)
                    check.text_match = False
                    check.text_similarity = 0.0
                    check.cited_text = cited_text
                    check.db_text_preview = db_text
                    check.mismatch_action = MismatchAction.DROPPED.value
                    check.details = (
                        f"Verified in Neo4j ({domain}), dropped by relevance gate "
                        f"({relevance_label})"
                    )
                    report.dropped_citations += 1
                    report.relevance_gate_dropped_citations += 1
                    report.issues.append(
                        f"Art. {article_num}: dropped as non-pertinent to the case"
                    )
                    self._log(
                        f"      🗑️ Art. {article_num} DROPPED by relevance gate "
                        f"(label={relevance_label})"
                    )
                elif cited_text and db_text:
                    # Perform text verification
                    similarity = self._compute_text_similarity(cited_text, db_text)
                    text_match = similarity >= settings.cc_text_match_threshold

                    check.text_verified = True
                    check.text_match = text_match
                    check.text_similarity = similarity
                    check.cited_text = cited_text
                    check.db_text_preview = db_text

                    if text_match:
                        report.text_matches += 1
                        check.mismatch_action = MismatchAction.MATCH.value
                        check.details = f"Verified in Neo4j ({domain}), text match: {similarity:.0%}"
                        self._log(
                            f"      📝 Text similarity: {similarity:.0%} ✅ MATCH"
                        )
                    else:
                        report.text_mismatches += 1
                        check.details = f"Verified in Neo4j ({domain}), text mismatch: {similarity:.0%}"
                        self._log(
                            f"      📝 Text similarity: {similarity:.0%} ⚠️ MISMATCH"
                        )
                        report.issues.append(
                            f"Art. {article_num}: cited text differs from DB (similarity: {similarity:.0%})"
                        )

                        # Handle mismatch: LLM verification + repair/drop
                        self._handle_norm_mismatch(
                            check=check,
                            article_num=article_num,
                            cited_text=cited_text,
                            db_text=db_text,
                            aspic_ir=aspic_ir,
                            full_text=verification_text,
                            report=report,
                            db_title=db_title,
                            precomputed_is_core=is_core,
                            prechecked_pertinent=(
                                True
                                if run_relevance_gate and relevance_label == "RELEVANT"
                                else None
                            ),
                        )
                elif not cited_text and db_text:
                    # Article exists in DB but no text was cited in the response.
                    # Treat as mismatch → repair by injecting DB text.
                    self._log(
                        f"      ⚠️ Art. {article_num}: no text cited but DB text available → forcing repair"
                    )
                    check.text_verified = True
                    check.text_match = False
                    check.text_similarity = 0.0
                    check.cited_text = ""
                    check.db_text_preview = db_text
                    check.is_core = True  # no text at all → force repair
                    check.mismatch_action = MismatchAction.REPAIRED.value
                    check.repaired_text = db_text
                    check.repair_success = True
                    check.details = f"Verified in Neo4j ({domain}), no text cited → repaired with DB text"
                    report.text_mismatches += 1
                    report.repaired_citations += 1
                    report.issues.append(
                        f"Art. {article_num}: no text cited in response, repaired with DB text"
                    )
                else:
                    check.details = f"Verified in Neo4j ({domain}), no text available for verification"
                    self._log("      📝 No cited text and no DB text for verification")
            else:
                report.invalid_citations += 1
                check.details = f"Not found in Neo4j ({domain})"
                report.issues.append(f"Art. {article_num} not found in {domain} codice")
                self._log(f"   ❌ Art. {article_num} -> NOT FOUND in Neo4j")

            report.citation_checks.append(check)
            report.total_citations += 1
            _emit_citation_progress(check)

        # ----- Precedent citation verification -----
        # `cited_prec_titles` and `prec_id_by_title` are precomputed above.

        if cited_prec_titles:
            self._log(
                f"📜 [{agent}] Found {len(cited_prec_titles)} precedent "
                f"citations to verify"
            )
        else:
            self._log(f"📜 [{agent}] No precedent citations detected in text")

        for prec_title in cited_prec_titles:
            if prec_title.lower() in verified_precedent_titles:
                continue
            verified_precedent_titles.add(prec_title.lower())

            # Verify existence in Neo4j and get summary
            # Use precedent_id when available for reliable lookup
            prec_id = prec_id_by_title.get(prec_title.lower())
            found, db_summary = self._verify_precedent_in_neo4j(
                prec_title, precedent_id=prec_id
            )

            check = CitationCheck(
                citation=f"Prec: {prec_title[:80]}",
                found_in_kb=found,
                source_type="precedent",
            )

            if found:
                report.valid_citations += 1
                self._log(f"   ✅ Precedent '{prec_title[:50]}...' -> EXISTS in Neo4j")

                # Extract cited text around the precedent mention
                cited_text = self._extract_cited_text_for_precedent(
                    verification_text, prec_title
                )

                if cited_text and db_summary:
                    similarity = self._compute_text_similarity(cited_text, db_summary)
                    text_match = similarity >= settings.cc_text_match_threshold

                    check.text_verified = True
                    check.text_match = text_match
                    check.text_similarity = similarity
                    check.cited_text = cited_text
                    check.db_text_preview = db_summary[:500]

                    if text_match:
                        report.text_matches += 1
                        check.mismatch_action = MismatchAction.MATCH.value
                        check.details = (
                            f"Precedent verified in Neo4j, "
                            f"text match: {similarity:.0%}"
                        )
                        self._log(
                            f"      📝 Text similarity: " f"{similarity:.0%} ✅ MATCH"
                        )
                    else:
                        report.text_mismatches += 1
                        check.details = (
                            f"Precedent verified in Neo4j, "
                            f"text mismatch: {similarity:.0%}"
                        )
                        self._log(
                            f"      📝 Text similarity: " f"{similarity:.0%} ⚠️ MISMATCH"
                        )
                        report.issues.append(
                            f"Precedent '{prec_title[:40]}': "
                            f"cited text differs from DB "
                            f"(similarity: {similarity:.0%})"
                        )
                        self._handle_precedent_mismatch(
                            check=check,
                            precedent_title=prec_title,
                            cited_text=cited_text,
                            db_summary=db_summary,
                            full_text=verification_text,
                            report=report,
                        )
                elif not cited_text and db_summary:
                    # Title is present but no surrounding text extracted.
                    # Treat as match (the LLM cited it correctly by title).
                    check.text_verified = True
                    check.text_match = True
                    check.text_similarity = 1.0
                    check.cited_text = prec_title
                    check.db_text_preview = db_summary[:500]
                    check.mismatch_action = MismatchAction.MATCH.value
                    check.details = (
                        "Precedent verified in Neo4j, "
                        "cited by exact title (no surrounding text)"
                    )
                    report.text_matches += 1
                    self._log("      📝 Precedent cited by exact title ✅ MATCH")
                else:
                    check.details = (
                        "Precedent verified in Neo4j, "
                        "no text available for verification"
                    )
            else:
                report.invalid_citations += 1
                check.details = "Precedent not found in Neo4j"
                report.issues.append(
                    f"Precedent '{prec_title[:40]}' not found in Neo4j"
                )
                self._log(
                    f"   ❌ Precedent '{prec_title[:50]}...' " f"-> NOT FOUND in Neo4j"
                )

            report.citation_checks.append(check)
            report.total_citations += 1
            _emit_citation_progress(check)

        report.issues = self._deduplicate_issue_list(report.issues)

        return report

    def _extract_statute_citations(self, text: str) -> list[str]:
        """Extract statute citations from text."""
        citations: list[str] = []
        seen: set[str] = set()

        for mention in extract_article_mentions(text, require_code=False):
            citation = format_article_citation(
                mention.article_id,
                mention.source_hint,
            )
            key = citation.lower()
            if key in seen:
                continue
            seen.add(key)
            citations.append(citation)

        return citations

    def _deduplicate_issue_list(self, issues: list[str]) -> list[str]:
        """
        Keep one final issue per citation key (article/precedent) to avoid
        double-reporting intermediate states.
        """
        if not issues:
            return []

        def issue_key(text: str) -> str:
            m = re.match(r"^(Art\.\s*[^:]+):", text, re.IGNORECASE)
            if m:
                return m.group(1).strip().lower()
            m = re.match(r"^(Precedent\s+'[^']+')", text, re.IGNORECASE)
            if m:
                return m.group(1).strip().lower()
            return text.strip().lower()

        def issue_priority(text: str) -> int:
            t = text.lower()
            if "attribution unresolved" in t:
                return 100
            if "attribution" in t:
                return 90
            if "not found" in t:
                return 80
            if "repair failed" in t:
                return 70
            if "dropped" in t:
                return 60
            if "differs from db" in t:
                return 50
            if "no text cited" in t:
                return 40
            return 10

        kept: dict[str, str] = {}
        for issue in issues:
            key = issue_key(issue)
            if key not in kept:
                kept[key] = issue
                continue
            if issue_priority(issue) >= issue_priority(kept[key]):
                kept[key] = issue
        return list(kept.values())

    def _generate_consistency_summary(
        self,
        reasoner_report: ConsistencyReport,
        counter_report: ConsistencyReport,
    ) -> str:
        """Generate a human-readable summary of the consistency analysis."""
        lines = []

        lines.append("## Analisi di Consistenza\n")

        # Reasoner summary
        lines.append("### Reasoner")
        lines.append(
            f"- Citazioni valide: {reasoner_report.valid_citations}/{reasoner_report.total_citations}"
        )
        lines.append(
            f"- Testo verificato: {reasoner_report.text_matches} match / {reasoner_report.text_matches + reasoner_report.text_mismatches} verificati"
        )
        if reasoner_report.repaired_citations > 0:
            lines.append(f"- Citazioni riparate: {reasoner_report.repaired_citations}")
        if reasoner_report.dropped_citations > 0:
            lines.append(f"- Citazioni scartate: {reasoner_report.dropped_citations}")
        if reasoner_report.relevance_gate_dropped_citations > 0:
            lines.append(
                "- Scarti da relevance gate: "
                f"{reasoner_report.relevance_gate_dropped_citations}"
            )
        if reasoner_report.issues:
            lines.append(f"- Problemi: {len(reasoner_report.issues)}")

        lines.append("")

        # Counter-Reasoner summary
        lines.append("### Counter-Reasoner")
        lines.append(
            f"- Citazioni valide: {counter_report.valid_citations}/{counter_report.total_citations}"
        )
        lines.append(
            f"- Testo verificato: {counter_report.text_matches} match / {counter_report.text_matches + counter_report.text_mismatches} verificati"
        )
        if counter_report.repaired_citations > 0:
            lines.append(f"- Citazioni riparate: {counter_report.repaired_citations}")
        if counter_report.dropped_citations > 0:
            lines.append(f"- Citazioni scartate: {counter_report.dropped_citations}")
        if counter_report.relevance_gate_dropped_citations > 0:
            lines.append(
                "- Scarti da relevance gate: "
                f"{counter_report.relevance_gate_dropped_citations}"
            )
        if counter_report.issues:
            lines.append(f"- Problemi: {len(counter_report.issues)}")

        lines.append("")

        return "\n".join(lines)

    def _regenerate_reasoning_chain_with_llm(
        self,
        original_chain: str,
        citation_checks: list[CitationCheck],
        agent_name: str,
    ) -> str:
        """Apply surgical repairs to the reasoning chain text.

        Strategy:
        - no cited text -> inject DB text after article mention
        - cited text mismatch -> in-place replacement with repaired text
        - dropped/failed citations -> mark unreliable citation

        Args:
            original_chain: The original reasoning chain text
            citation_checks: List of CitationCheck objects with mismatch actions
            agent_name: Name of the agent ("reasoner" or "counter_reasoner")

        Returns:
            The repaired chain text.
        """
        surgical_count = 0
        dropped_count = 0
        chain = original_chain

        for check in citation_checks:
            # --- REPAIRED citations ---
            if check.mismatch_action == MismatchAction.REPAIRED.value:
                # Precedent citations: surgical replace of cited context
                if check.source_type == "precedent":
                    if check.cited_text and check.repaired_text:
                        if check.cited_text in chain:
                            chain = chain.replace(
                                check.cited_text, check.repaired_text, 1
                            )
                            surgical_count += 1
                        else:
                            idx = chain.lower().find(check.cited_text.lower())
                            if idx >= 0:
                                chain = (
                                    chain[:idx]
                                    + check.repaired_text
                                    + chain[idx + len(check.cited_text) :]
                                )
                                surgical_count += 1
                    continue

                # Statute citations (original logic)
                if not check.cited_text and check.repaired_text:
                    # Case A: no text was cited → inject DB text (surgical)
                    chain = self._inject_text_after_article(
                        chain, check.citation, check.repaired_text
                    )
                    surgical_count += 1

                elif check.cited_text and check.repaired_text:
                    # Case B: text cited and repaired text available → surgical replace
                    if check.cited_text in chain:
                        chain = chain.replace(check.cited_text, check.repaired_text, 1)
                        surgical_count += 1
                    else:
                        idx = chain.lower().find(check.cited_text.lower())
                        if idx >= 0:
                            chain = (
                                chain[:idx]
                                + check.repaired_text
                                + chain[idx + len(check.cited_text) :]
                            )
                            surgical_count += 1
                        else:
                            chain = self._inject_text_after_article(
                                chain, check.citation, check.repaired_text
                            )
                            surgical_count += 1

            # --- DROPPED / REPAIR_FAILED citations ---
            elif check.mismatch_action in (
                MismatchAction.DROPPED.value,
                MismatchAction.REPAIR_FAILED.value,
            ):
                if check.cited_text and check.cited_text in chain:
                    chain = chain.replace(
                        check.cited_text,
                        "[Citation removed - unreliable source]",
                        1,
                    )
                dropped_count += 1

        total = surgical_count + dropped_count
        if total == 0:
            self._log(f"   ✅ [{agent_name}] No repairs needed - chain unchanged")
            return original_chain

        self._log(
            f"   🔄 [{agent_name}] Repair complete: "
            f"{surgical_count} surgical, {dropped_count} dropped"
        )
        return chain

    # ------------------------------------------------------------------
    # Per-step LLM regeneration (only for severe mismatches)
    # ------------------------------------------------------------------

    def _inject_text_after_article(
        self, chain: str, citation: str, repaired_text: str
    ) -> str:
        """Insert a normative-text annotation after the first mention of *citation*.

        For example, if ``citation`` is ``"Art. 43 c.p."`` and the chain
        contains ``"… Art. 43 c.p. …"``, this method turns it into
        ``"… Art. 43 c.p. [Testo: «…»] …"``.

        The ``repaired_text`` is truncated to a reasonable length to
        avoid bloating the chain.
        """
        article_id = self._extract_article_id_from_citation(citation)
        if not article_id:
            return chain
        article_pattern = self._article_id_to_regex(article_id)

        # Build a regex that matches "Art. X c.p./c.c./L.241/1990".
        pattern = re.compile(
            rf"(Art(?:icolo)?\.?\s*{article_pattern}(?![-\w])\s*"
            rf"(?:c\.?\s*[cp]\.?|"
            rf"l\.?\s*241(?:\s*/\s*1990)?|legge\s*241(?:\s*/\s*1990)?|"
            rf"cod(?:ice)?\.?\s*(?:civ(?:ile)?|pen(?:ale)?|amm(?:inistrativ[oa])?))?)",
            re.IGNORECASE,
        )

        match = pattern.search(chain)
        if not match:
            return chain

        # Truncate repaired text to first 200 chars if very long
        snippet = repaired_text[:200].rstrip()
        if len(repaired_text) > 200:
            snippet += "…"

        insert_pos = match.end()
        annotation = f", il quale dispone che «{snippet}»,"
        chain = chain[:insert_pos] + annotation + chain[insert_pos:]
        return chain

    def _repair_aspic_ir(
        self,
        aspic_ir: dict,
        citation_checks: list[CitationCheck],
        repaired_chain_text: str = "",
        claim: str = "",
        role: str = "support",
        statutes: list[dict] | None = None,
        precedents: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """
        Rebuild the ASPIC IR from the repaired reasoning chain.

        Preserves the original step order by using indexed extraction
        rather than relying on AspicFormatter's re-parsing which may
        reorder steps.

        Args:
            aspic_ir: Original ASPIC IR structure (used for fallback/metadata)
            citation_checks: List of CitationCheck objects with mismatch actions
            repaired_chain_text: The LLM-regenerated reasoning chain text
            claim: The original legal claim
            role: "support" or "counter"
            statutes: Relevant statutes for the formatter
            precedents: Relevant precedents for the formatter
            metadata: Additional metadata (causal_type_id, theory_id, etc.)

        Returns:
            Rebuilt ASPIC IR structure from the repaired chain, or empty dict
            if no repairs were needed.
        """
        if not aspic_ir:
            return {}

        # Count repairs and drops
        total_repaired = sum(
            1
            for c in citation_checks
            if c.mismatch_action == MismatchAction.REPAIRED.value
        )
        total_dropped = sum(
            1
            for c in citation_checks
            if c.mismatch_action
            in (MismatchAction.DROPPED.value, MismatchAction.REPAIR_FAILED.value)
        )

        # If no changes were needed, return empty (signals "use original")
        if total_repaired == 0 and total_dropped == 0:
            return {}

        # If no repaired text available, fall back to original
        if not repaired_chain_text:
            self._log(
                f"   ⚠️ [{role}] No repaired chain text available, skipping IR rebuild"
            )
            return {}

        self._log(
            f"   🔄 [{role.upper()}] Rebuilding ASPIC IR from repaired chain "
            f"(repaired={total_repaired}, dropped={total_dropped})"
        )

        # Parse the repaired chain text - extract steps in order.
        # NOTE: we intentionally do NOT call _sanitize_reasoning_chain here.
        # That method uses keyword filtering ("precedent"/"precedente") and
        # removes valid reasoning steps that happen to cite precedents when
        # the precedents list is empty.  _build_chain_steps inside
        # AspicFormatter already handles noise/dedup without destroying
        # legitimate steps.
        reasoning_chain = self._extract_reasoning_chain_ordered(repaired_chain_text)

        # Fallback: if the repair LLM dropped the numbered chain format,
        # the parser may extract < 2 steps. In that case, reuse the
        # original IR's reasoning_chain so AQA can still build links.
        original_chain_steps = aspic_ir.get("reasoning_chain", [])
        if len(reasoning_chain) < 2 and len(original_chain_steps) >= 2:
            self._log(
                f"   ⚠️ [{role.upper()}] Repaired chain produced only "
                f"{len(reasoning_chain)} parsed step(s); falling back to "
                f"original chain ({len(original_chain_steps)} steps)"
            )
            reasoning_chain = [
                s.get("text", s) if isinstance(s, dict) else s
                for s in original_chain_steps
            ]

        arguments = self._extract_arguments_from_text(repaired_chain_text)

        # Build new IR using AspicFormatter (same as Reasoner/Counter-Reasoner)
        formatter = AspicFormatter(
            role=role,
            statutes=statutes or [],
            precedents=precedents or [],
        )
        rebuilt_ir = formatter.format(
            claim=claim or aspic_ir.get("claim", ""),
            raw_response=repaired_chain_text,
            reasoning_chain=reasoning_chain,
            arguments=arguments,
            metadata=metadata or aspic_ir.get("metadata", {}),
        )

        # Add repair metadata
        rebuilt_ir["_repair_metadata"] = {
            "total_repaired": total_repaired,
            "total_dropped": total_dropped,
        }

        self._log(
            f"   ✅ [{role.upper()}] ASPIC IR rebuilt: "
            f"{len(rebuilt_ir.get('arguments', []))} arguments, "
            f"{len(rebuilt_ir.get('reasoning_chain', []))} chain steps"
        )

        return rebuilt_ir

    def _extract_reasoning_chain_ordered(self, text: str) -> list[str]:
        """Extract reasoning chain steps preserving their original numbered order.

        Parses numbered steps (1. 2. 3. ...) from the chain section and
        returns them in their original order, not re-sorted.
        """
        steps: list[str] = []

        # Find the chain section
        chain_start = text.lower().find("catena di ragionamento")
        if chain_start == -1:
            chain_start = text.lower().find("reasoning chain")
        if chain_start == -1:
            chain_start = 0

        chain_text = text[chain_start:]

        # Pattern for numbered steps: "1. text" or "1) text"
        step_pattern = re.compile(
            r"^\s*(\d+)[.)\s]+(.+?)(?=^\s*\d+[.)\s]|\Z)", re.MULTILINE | re.DOTALL
        )

        # Extract all numbered steps
        numbered_steps: list[tuple[int, str]] = []
        for match in step_pattern.finditer(chain_text):
            step_num = int(match.group(1))
            step_text = match.group(2).strip()
            # Clean up the step text
            step_text = re.sub(r"\s+", " ", step_text)
            if step_text and len(step_text) > 10:
                numbered_steps.append((step_num, step_text))

        # Sort by step number to preserve original order
        numbered_steps.sort(key=lambda x: x[0])

        # Extract just the text
        steps = [text for _, text in numbered_steps]

        return steps

    def _extract_arguments_from_text(self, response: str) -> list[dict]:
        """
        Extract argument blocks from raw response text.

        Reuses the same logic as Reasoner._extract_arguments and
        CounterReasoner._extract_arguments to parse structured blocks.
        """
        arguments = []
        current_arg: dict[str, str] = {}
        current_section = None

        section_markers = {
            "premessa": "premise",
            "premessa alternativa": "premise",
            "norma": "norm",
            "nesso causale": "link",
            "nesso causale alternativo": "link",
            "causal link": "link",
            "conclusione": "conclusion",
            "conclusione contraria": "conclusion",
        }

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Remove markdown bold markers
            clean = line.replace("**", "").strip()
            lower_clean = clean.lower()

            # Check for section headers
            matched_section = None
            for marker, section_key in section_markers.items():
                if lower_clean.startswith(marker):
                    matched_section = section_key
                    # Extract content after the marker
                    content = clean[len(marker) :].lstrip(" :.-").strip()
                    break

            if matched_section:
                if matched_section == "premise" and current_arg.get("conclusion"):
                    # New argument block
                    if current_arg:
                        arguments.append(current_arg)
                    current_arg = {}

                current_section = matched_section
                if content:
                    current_arg[current_section] = content
                continue

            # Check for chain/reasoning section → stop parsing arguments
            if any(
                kw in lower_clean
                for kw in [
                    "ragionamento",
                    "chain of reasoning",
                    "catena di ragionamento",
                ]
            ):
                break

            # Continuation of current section
            if current_section and line:
                existing = current_arg.get(current_section, "")
                current_arg[current_section] = (
                    (existing + " " + line).strip() if existing else line
                )

        if current_arg:
            arguments.append(current_arg)

        return arguments
