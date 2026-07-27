"""
AQA (Argument Quality Assessment) engine.

Implements the full ASPIC+ evaluation pipeline:
- Link building from ASPIC IR structures
- Cross-attack computation with domain-aware rules
- Precedent influence scoring
- Final verdict aggregation (Option F – LLM-unified)
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..tools.neo4j_tools import get_driver  # noqa: E402
from ..tools.prompt_registry import render_prompt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings  # noqa: E402
from services.groq_client import get_chat_groq, resilient_chat_call  # noqa: E402


class AQAEngineMixin:
    """Mixin providing AQA methods to the evaluator."""

    def _extract_statute_refs_from_text(self, text: str) -> list[dict]:
        refs = []
        for match in self.STATUTE_PATTERN.finditer(text):
            art_raw = match.group(1)
            normalizer = getattr(self, "_normalize_article_id", None)
            art = (
                normalizer(art_raw)
                if callable(normalizer)
                else str(art_raw or "").strip().lower()
            )
            if not art:
                continue
            code_part = match.group(2) or ""
            source = ""
            if code_part:
                code_lower = code_part.lower()
                if "241" in code_lower or "amm" in code_lower:
                    source = "codice_amministrativo"
                else:
                    source = (
                        "codice_civile"
                        if "c" in code_lower and "p" not in code_lower
                        else "codice_penale"
                    )
            refs.append({"articolo": art, "source": source})
        return refs

    def _get_statute_meta(
        self, article_num: str, domain: str, source_hint: str = ""
    ) -> dict:
        normalizer = getattr(self, "_normalize_article_id", None)
        article_id = (
            normalizer(article_num)
            if callable(normalizer)
            else str(article_num or "").strip().lower()
        )
        if not article_id:
            return {}

        # If a source_hint is provided (e.g. "codice_penale"), resolve domain
        if source_hint and domain == "ENTRAMBI":
            if source_hint == "codice_penale":
                domain = "PENALE"
            elif source_hint == "codice_civile":
                domain = "CIVILE"
            elif source_hint == "codice_amministrativo":
                domain = "AMMINISTRATIVO"

        key = (article_id, domain)
        if key in self._statute_meta_cache:
            return self._statute_meta_cache[key]
        driver = get_driver()
        if domain == "CIVILE":
            codice = "codice_civile"
            lookup_variants = (
                self._build_lookup_variants(article_id, "CIVILE")
                if hasattr(self, "_build_lookup_variants")
                else [
                    article_id if article_id.startswith("art") else f"art{article_id}"
                ]
            )
        elif domain == "PENALE":
            codice = "codice_penale"
            lookup_variants = (
                self._build_lookup_variants(article_id, "PENALE")
                if hasattr(self, "_build_lookup_variants")
                else [article_id[3:] if article_id.startswith("art") else article_id]
            )
        elif domain == "AMMINISTRATIVO":
            codice = "codice_amministrativo"
            lookup_variants = (
                self._build_lookup_variants(article_id, "AMMINISTRATIVO")
                if hasattr(self, "_build_lookup_variants")
                else [article_id[3:] if article_id.startswith("art") else article_id]
            )
        else:
            # ENTRAMBI without hint: try PENALE, then CIVILE, then AMMINISTRATIVO
            meta = self._get_statute_meta(article_id, "PENALE")
            if meta:
                return meta
            meta = self._get_statute_meta(article_id, "CIVILE")
            if meta:
                return meta
            meta = self._get_statute_meta(article_id, "AMMINISTRATIVO")
            if meta:
                return meta
            return {}
        query = """
            MATCH (s:Statute)
            WHERE s.articolo = $articolo AND s.source = $codice
            RETURN s.libro AS libro, s.source AS source
            LIMIT 1
        """
        try:
            with driver.session() as session:
                for articolo in lookup_variants:
                    record = session.run(
                        query, {"articolo": articolo, "codice": codice}
                    ).single()
                    if not record:
                        continue
                    meta = {
                        "libro": record.get("libro"),
                        "source": record.get("source"),
                    }
                    self._statute_meta_cache[key] = meta
                    return meta
        except Exception:
            pass
        self._statute_meta_cache[key] = {}
        return {}

    def _derive_severity_category(self, meta: dict) -> str:
        source = meta.get("source") or ""
        libro = (meta.get("libro") or "").lower()
        match = re.search(
            r"libro\s+(iv|vi|v|iii|ii|i|primo|secondo|terzo|quarto|quinto|sesto)",
            libro,
        )
        token = match.group(1) if match else ""
        if source == "codice_penale":
            return (self._aqa_severity_map_penale or {}).get(token, "")
        if source == "codice_civile":
            if "fuori range" in libro:
                return ""
            return (self._aqa_severity_map_civile or {}).get(token, "")
        if source == "codice_amministrativo":
            return "amministrativo"
        return ""

    def _collect_links(self, aspic_ir: dict, role: str, domain: str) -> list[dict]:
        links: list[dict] = []
        reasoning_chain = aspic_ir.get("reasoning_chain", [])
        source_statutes = (
            aspic_ir.get("sources", {}).get("statutes", [])
            if isinstance(aspic_ir.get("sources"), dict)
            else []
        )

        normalizer = getattr(self, "_normalize_article_id", None)

        def _norm_article(raw: object) -> str:
            text = str(raw or "").strip()
            if not text:
                return ""
            if callable(normalizer):
                try:
                    return str(normalizer(text) or "").strip()
                except Exception:
                    pass
            return text.lower()

        retrieved_exact: dict[tuple[str, str], dict] = {}
        retrieved_by_article: dict[str, dict] = {}
        for item in source_statutes:
            if not isinstance(item, dict):
                continue
            articolo = _norm_article(item.get("articolo"))
            source = str(item.get("source", "") or "").strip()
            if not articolo:
                continue
            payload = {
                "statute_id": item.get("statute_id"),
                "articolo": articolo,
                "source": source,
                "title": item.get("title") or "",
            }
            retrieved_exact[(articolo, source)] = payload
            retrieved_by_article.setdefault(articolo, payload)

        # Check if this is a repaired IR
        repair_meta = aspic_ir.get("_repair_metadata", {})
        if repair_meta:
            self._log(
                f"   🔧 [{role.upper()}] Using REPAIRED IR: "
                f"{repair_meta.get('total_repaired', 0)} repaired, "
                f"{repair_meta.get('total_dropped', 0)} dropped"
            )

        # Filter reasoning_chain to exclude noise steps and citation-removal markers
        valid_chain_steps = [
            step
            for step in reasoning_chain
            if step.get("text")
            and not step.get("text", "").lower().startswith("precedent")
            and not step.get("text", "").lower().startswith("[citation removed")
            and not step.get("text", "").lower().startswith("citation removed")
            and "[citation removed" not in step.get("text", "").lower()
        ]

        if len(valid_chain_steps) < 2:
            self._log(
                f"   ⚠️ [{role.upper()}] Not enough chain steps to build links (found {len(valid_chain_steps)})"
            )
            return links

        self._log(
            f"   📋 [{role.upper()}] Building {len(valid_chain_steps) - 1} links from reasoning_chain"
        )

        for idx in range(len(valid_chain_steps) - 1):
            step_from = valid_chain_steps[idx]
            step_to = valid_chain_steps[idx + 1]

            step_from_id = step_from.get("id") or f"S{idx + 1}"
            step_to_id = step_to.get("id") or f"S{idx + 2}"
            link_id = f"{step_from_id}->{step_to_id}"

            premise_text = self._normalize_text(step_from.get("text", ""))
            conclusion_text = self._normalize_text(step_to.get("text", ""))
            # The "rule" or nesso is the logical connection between the two steps
            # We use the premise text as it contains the normative basis
            link_text = premise_text

            # Extract citations from both steps
            citations_text = f"{premise_text} {conclusion_text}"
            statute_refs = self._extract_statute_refs_from_text(citations_text)
            retrieved_norms: list[dict] = []
            for ref in statute_refs:
                articolo = _norm_article(ref.get("articolo", ""))
                source_hint = str(ref.get("source", "") or "").strip()
                exact = retrieved_exact.get((articolo, source_hint))
                fallback = retrieved_by_article.get(articolo)
                if exact:
                    retrieved_norms.append(
                        {
                            **exact,
                            "score": 1.0,
                            "similarity": 1.0,
                            "match_type": "exact",
                        }
                    )
                elif fallback:
                    retrieved_norms.append(
                        {
                            **fallback,
                            "score": 0.85,
                            "similarity": 0.85,
                            "match_type": "article_fallback",
                        }
                    )
                elif articolo:
                    retrieved_norms.append(
                        {
                            "articolo": articolo,
                            "source": source_hint,
                            "score": 0.0,
                            "similarity": 0.0,
                            "match_type": "missing",
                        }
                    )

            libri = set()
            severities = set()
            sources = set()
            for ref in statute_refs:
                meta = self._get_statute_meta(
                    ref.get("articolo", ""), domain, ref.get("source", "")
                )
                libro = meta.get("libro")
                if libro:
                    libri.add(libro)
                source = meta.get("source")
                if source:
                    sources.add(source)
                severity = self._derive_severity_category(meta)
                if severity:
                    severities.add(severity)

            link_libro = list(libri)[0] if len(libri) == 1 else ""
            link_source = (
                list(sources)[0]
                if len(sources) == 1
                else (list(sources)[0] if sources else "")
            )
            # P3 fix: prefer the most specific (non-generale) category
            if severities:
                specific = [s for s in severities if "generale" not in s]
                severity_category = specific[0] if specific else list(severities)[0]
            else:
                severity_category = ""

            # Log the link
            self._log(
                f"      🔗 Link {idx + 1}: {link_id} | "
                f"from={step_from_id} to={step_to_id} | norms={len(statute_refs)}"
            )
            if statute_refs:
                refs_str = ", ".join(
                    f"Art. {r.get('articolo', '?')}" for r in statute_refs[:3]
                )
                self._log(f"         📜 Statutes: {refs_str}")
            if link_libro or severity_category:
                self._log(
                    f"         🗂️ Libro: {link_libro or '-'} | Severity: {severity_category or '-'}"
                )

            links.append(
                {
                    "link_id": link_id,
                    "argument_id": f"chain_{idx + 1}",
                    "premise_ids": [step_from_id],
                    "conclusion_id": step_to_id,
                    "rule_id": "",
                    "role": role,
                    "text": link_text,
                    "premise_text": premise_text,
                    "conclusion_text": conclusion_text,
                    "libro": link_libro,
                    "source": link_source,
                    "severity_category": severity_category,
                    "retrieved_norms": retrieved_norms or None,
                }
            )

        self._log(
            f"   ✅ [{role.upper()}] Collected {len(links)} links from reasoning_chain"
        )
        return links

    def _same_book(self, sev1: str, sev2: str) -> bool:
        """Return True if two severity categories belong to the same libro."""
        book1 = self._aqa_severity_book_map.get(sev1, "")
        book2 = self._aqa_severity_book_map.get(sev2, "")
        return book1 == book2 if book1 and book2 else False

    def _is_procedural_vs_substantive(self, sev1: str, sev2: str) -> bool:
        """Return True if one severity is procedural and the other substantive."""
        procedural = getattr(self, "_aqa_procedural_categories", set())
        return (sev1 in procedural) != (sev2 in procedural)

    def _classify_attack_type(self, target: dict, attacker: dict) -> str:
        """Classify the legal attack type via LLM for damage modulation.

        Uses an LLM prompt to determine the logical relationship between
        the target reasoning link and the attacking link.  This replaces
        the former keyword-based heuristic so that the classification
        generalises to any legal domain and phrasing.

        Categories:
        - contradiction:       direct opposition on the same legal object
        - exception:           condition blocking norm application
        - derogation:          lex specialis overriding generalis
        - extinction:          prescription / forfeiture / estinzione
        - factual_impediment:  factual argument without normative basis
        - general_opposition:  fallback / generic rebuttal

        Falls back to ``general_opposition`` if the LLM is unavailable or
        returns an unparseable answer.
        """
        valid_types = set(
            getattr(
                self,
                "_aqa_valid_attack_types",
                {
                    "contradiction",
                    "exception",
                    "derogation",
                    "extinction",
                    "factual_impediment",
                    "general_opposition",
                },
            )
        )
        default_type = (
            str(getattr(self, "_aqa_default_attack_type", "general_opposition"))
            .strip()
            .lower()
            or "general_opposition"
        )
        if default_type not in valid_types:
            default_type = "general_opposition"

        target_text = self._normalize_text(
            f"{target.get('premise_text', '')} " f"{target.get('conclusion_text', '')}"
        ).strip()
        attacker_text = self._normalize_text(
            f"{attacker.get('premise_text', '')} "
            f"{attacker.get('conclusion_text', '')}"
        ).strip()

        if not target_text or not attacker_text:
            return default_type

        try:
            llm = get_chat_groq(
                temperature=settings.classifier_temperature,
                max_tokens=settings.attack_type_max_tokens,
            )

            system_prompt = render_prompt("aqa_engine.attack_type_system")

            user_prompt = render_prompt(
                "aqa_engine.attack_type_user",
                target_text=target_text[: settings.truncation_nli_text],
                attacker_text=attacker_text[: settings.truncation_nli_text],
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = resilient_chat_call(llm, messages)
            answer = response.content.strip().replace(" ", "_").lower()

            # Normalise common LLM variations
            if answer in valid_types:
                return answer
            # Partial match (e.g. "CONTRADICTION." or "- CONTRADICTION")
            for vt in valid_types:
                if vt in answer:
                    return vt

            self._log(
                f"      \u26a0\ufe0f LLM attack-type unrecognised: "
                f'"{answer}", falling back to {default_type}'
            )
            return default_type

        except Exception as exc:
            self._log(
                f"      \u26a0\ufe0f LLM attack-type classification failed: {exc}",
                "warning",
            )
            return default_type

    def _get_attack_type_multiplier(self, attack_type: str) -> float:
        """Return the damage multiplier for a given attack type."""
        return self._aqa_attack_type_multipliers.get(attack_type.lower(), 1.0)

    def _domain_rules_allow(self, target: dict, attacker: dict) -> tuple[bool, str]:
        """Improved domain rules for realistic legal attacks.

        Allows:
        - Factual attacks on normative arguments
        - Double penale-civile relevance
        - Bridge norms (Art. 185, 198 c.p.)
        - Same-book severity compatibility
        - Constitutional principles overriding ordinary norms
        - Procedural vs substantive interactions

        Blocks:
        - Fatto vs Fatto (no legal value)
        - Different codice without a recognised exception
        - Incompatible severity categories (different libro, no special rule)
        """
        t_sev = target.get("severity_category", "")
        a_sev = attacker.get("severity_category", "")

        # ==================================================
        # RULE 1 – Normative content
        # ==================================================
        # Block only: both lack normative content (fact vs fact)
        if not t_sev and not a_sev:
            return False, "denied: both lack normative content"

        # Factual attack (one side has no norm)
        if not t_sev or not a_sev:
            if self._aqa_allow_factual_attacks:
                return True, "allowed: factual attack"
            return False, "denied: factual attacks disabled"

        # ==================================================
        # RULE 2 – Codice compatibility
        # ==================================================
        t_src = target.get("source", "")
        a_src = attacker.get("source", "")

        # Missing source → treat as factual, allow
        if not t_src or not a_src:
            return True, "allowed: missing source (factual)"

        if t_src == a_src:
            # Same codice → fall through to severity check
            pass
        elif self._aqa_allow_cross_codice:
            return True, f"allowed: cross-codice ({t_src} vs {a_src})"
        else:
            return False, f"denied: different codice ({t_src} vs {a_src})"

        # ==================================================
        # RULE 3 – Severity compatibility
        # ==================================================
        # Same severity
        if t_sev == a_sev:
            return True, "allowed: same severity"

        # Norme generali cross-cut everything
        if "generale" in t_sev or "generale" in a_sev:
            return True, "allowed: generale cross-cutting"

        # Same libro (e.g. contratti generali vs speciali)
        if self._same_book(t_sev, a_sev):
            return True, "allowed: same libro"

        # Constitutional principles
        if "costituzionale" in t_sev or "costituzionale" in a_sev:
            return True, "allowed: constitutional principle"

        # Procedural vs substantive (e.g. prescrizione vs responsabilità)
        if self._is_procedural_vs_substantive(t_sev, a_sev):
            return True, "allowed: procedural vs substantive"

        return False, f"denied: severity mismatch ({t_sev} vs {a_sev})"

    def _compute_cross_attacks(
        self,
        pro_links: list[dict],
        contra_links: list[dict],
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """Bidirectional cross-attacks between PRO and CONTRA links (ASPIC+).

        **Option F – LLM-unified classification + type-based bypass**

        Pipeline per ogni coppia (target, attacker):

        1. **Hard gate** – domain rules (``_domain_rules_allow``).
        2. **Soft filter 1** – semantic overlap >= MIN_SEMANTIC_OVERLAP.
        3. **Classify attack type** – LLM-based (Groq).
        4. **LLM contradiction gate** – LLM inference:
           - *contradiction*  ->  bypass ratio, attack always proceeds.
           - *entailment*  ->  block attack
             (the two nessi *support* each other – not a real attack).
           - *neutral*  ->  fall through to step 5.
        5. **Type-specific strength ratio** – ``aqa_strength_ratio_by_type``.
           Specific types (contradiction, exception, derogation, extinction)
           have ratio=0.0, so they always pass.
           Only general_opposition and factual_impediment are gated.
        6. **Damage** – excess formula with type multiplier.

        Damage formula (only the *excess* counts):
            raw_attack       = overlap x attacker_base_score
            boosted_attack   = raw_attack x attack_type_multiplier
            attack_value     = max(0, boosted_attack - target_base) x DAMAGE_FACTOR
        """
        MIN_SEMANTIC_OVERLAP = self._aqa_min_semantic_overlap
        DAMAGE_FACTOR = self._aqa_damage_factor

        # Initialise
        for link in pro_links + contra_links:
            link["attacks_received"] = []
            link["attacks_sum"] = 0.0

        def _apply_attacks(
            targets: list[dict],
            attackers: list[dict],
            attacker_role: str,
            attack_direction: str,
        ) -> None:
            total_targets = max(1, len(targets))
            for target_idx, target in enumerate(targets, start=1):
                target_base = target.get("base_score", 0.0)
                target_text = (
                    f"{target.get('premise_text', '')} "
                    f"{target.get('conclusion_text', '')}"
                )

                for attacker in attackers:
                    allowed, reason = self._domain_rules_allow(target, attacker)

                    # ---- Stage 1: Hard gate (domain rules) ----
                    if not allowed:
                        target["attacks_received"].append(
                            {
                                "attacker_link_id": attacker.get("link_id"),
                                "attacker_role": attacker_role,
                                "overlap": 0.0,
                                "attacker_base_score": round(
                                    attacker.get("base_score", 0.0), 4
                                ),
                                "attack_value": 0.0,
                                "reason": reason,
                                "filtered": True,
                                "filter_stage": "domain",
                            }
                        )
                        continue

                    attacker_text = (
                        f"{attacker.get('premise_text', '')} "
                        f"{attacker.get('conclusion_text', '')}"
                    )
                    overlap = self._similarity(target_text, attacker_text)

                    # ---- Stage 2: Semantic overlap threshold ----
                    if overlap < MIN_SEMANTIC_OVERLAP:
                        target["attacks_received"].append(
                            {
                                "attacker_link_id": attacker.get("link_id"),
                                "attacker_role": attacker_role,
                                "overlap": round(overlap, 4),
                                "attacker_base_score": round(
                                    attacker.get("base_score", 0.0), 4
                                ),
                                "attack_value": 0.0,
                                "reason": (
                                    f"filtered: overlap {overlap:.3f} "
                                    f"< {MIN_SEMANTIC_OVERLAP}"
                                ),
                                "filtered": True,
                                "filter_stage": "overlap",
                            }
                        )
                        continue

                    attacker_base = attacker.get("base_score", 0.0)
                    raw_attack = overlap * attacker_base

                    # ---- Stage 3: Classify attack type ----
                    attack_type = self._classify_attack_type(target, attacker).lower()
                    type_multiplier = self._get_attack_type_multiplier(attack_type)

                    # ---- Stage 4: LLM contradiction gate ----
                    nli_label, nli_score = self._check_nli_contradiction(
                        target_text, attacker_text
                    )
                    nli_bypass = False

                    if nli_label == "contradiction":
                        # LLM confirms genuine contradiction -> bypass ratio
                        nli_bypass = True
                    elif nli_label == "entailment":
                        # LLM says these two nessi *support* each other
                        target["attacks_received"].append(
                            {
                                "attacker_link_id": attacker.get("link_id"),
                                "attacker_role": attacker_role,
                                "overlap": round(overlap, 4),
                                "attacker_base_score": round(attacker_base, 4),
                                "attack_value": 0.0,
                                "attack_type": attack_type,
                                "nli_label": nli_label,
                                "nli_score": nli_score,
                                "reason": (
                                    "filtered: LLM entailment "
                                    "- nessi non contraddittori"
                                ),
                                "filtered": True,
                                "filter_stage": "nli_entailment",
                            }
                        )
                        continue

                    # ---- Stage 5: Type-specific strength ratio ----
                    if not nli_bypass:
                        type_ratio = self._aqa_strength_ratio_by_type.get(
                            attack_type.lower(),
                            self._aqa_min_strength_ratio,  # fallback
                        )

                        # Base vs base comparison (NO overlap)
                        if type_ratio > 0 and attacker_base < type_ratio * target_base:
                            target["attacks_received"].append(
                                {
                                    "attacker_link_id": attacker.get("link_id"),
                                    "attacker_role": attacker_role,
                                    "overlap": round(overlap, 4),
                                    "attacker_base_score": round(attacker_base, 4),
                                    "attack_value": 0.0,
                                    "attack_type": attack_type,
                                    "nli_label": nli_label,
                                    "nli_score": nli_score,
                                    "reason": (
                                        f"filtered: attacker_base {attacker_base:.3f} < "
                                        f"{type_ratio} × {target_base:.3f} "
                                        f"(ratio for {attack_type})"
                                    ),
                                    "filtered": True,
                                    "filter_stage": "strength_ratio",
                                }
                            )
                            continue

                    # ---- Stage 6: Compute damage (excess formula) ----
                    # Only the EXCESS of the boosted attack over the target's
                    # own quality counts as damage.  This ensures that only
                    # genuinely stronger attacks inflict damage, and only
                    # proportional to the surplus.
                    boosted_attack = raw_attack * type_multiplier
                    excess = max(0.0, boosted_attack - target_base)
                    attack_value = excess * DAMAGE_FACTOR

                    bypass_tag = " [NLI-bypass]" if nli_bypass else ""
                    target["attacks_received"].append(
                        {
                            "attacker_link_id": attacker.get("link_id"),
                            "attacker_role": attacker_role,
                            "overlap": round(overlap, 4),
                            "attacker_base_score": round(attacker_base, 4),
                            "boosted_attack": round(boosted_attack, 4),
                            "target_base_score": round(target_base, 4),
                            "excess": round(excess, 4),
                            "damage_factor": DAMAGE_FACTOR,
                            "attack_value": round(attack_value, 4),
                            "attack_type": attack_type,
                            "type_multiplier": round(type_multiplier, 2),
                            "nli_label": nli_label,
                            "nli_score": nli_score,
                            "nli_bypass": nli_bypass,
                            "reason": f"active: {attack_type}{bypass_tag}",
                            "filtered": False,
                        }
                    )
                    if attack_value > 0:
                        target["attacks_sum"] += attack_value

                if progress_callback:
                    try:
                        progress_callback(
                            {
                                "stage": "attacks_progress",
                                "message": (
                                    f"Attacchi {attack_direction}: "
                                    f"{target_idx}/{total_targets}"
                                ),
                                "direction": attack_direction,
                                "current_target": target_idx,
                                "total_targets": total_targets,
                                "target_link_id": target.get("link_id"),
                            }
                        )
                    except Exception:
                        pass

        # CONTRA attacks PRO
        _apply_attacks(
            pro_links,
            contra_links,
            attacker_role="contra",
            attack_direction="counter->support",
        )
        # PRO attacks CONTRA
        _apply_attacks(
            contra_links,
            pro_links,
            attacker_role="support",
            attack_direction="support->counter",
        )

        # Sort attacks by value (descending) and apply top-K limiting
        top_k = self._aqa_attack_top_k
        for link in pro_links + contra_links:
            link["attacks_received"] = sorted(
                link.get("attacks_received", []),
                key=lambda x: x.get("attack_value", 0.0),
                reverse=True,
            )
            # Only the top-K active attacks contribute to attacks_sum
            active_attacks = [
                a
                for a in link["attacks_received"]
                if not a.get("filtered", False) and a.get("attack_value", 0.0) >= 0.01
            ]
            if len(active_attacks) > top_k:
                for overflow in active_attacks[top_k:]:
                    overflow["reason"] = (
                        f"capped: top-K overflow (rank > {top_k}), "
                        f"original={overflow.get('reason', '')}"
                    )
                    overflow["filtered"] = True
                    overflow["filter_stage"] = "top_k"
            # Recalculate attacks_sum from surviving active attacks only
            link["attacks_sum"] = sum(
                a.get("attack_value", 0.0)
                for a in link["attacks_received"]
                if not a.get("filtered", False) and a.get("attack_value", 0.0) > 0
            )

    def _compute_redundancy_penalty(self, links: list[dict]) -> dict:
        """Estimate and penalize redundant intra-chain links for a single side."""
        texts: list[str] = []
        for link in links:
            text = self._normalize_text(
                " ".join(
                    [
                        str(link.get("premise_text", "") or ""),
                        str(link.get("text", "") or ""),
                        str(link.get("conclusion_text", "") or ""),
                    ]
                )
            )
            if text:
                texts.append(text)

        n_links = len(texts)
        if n_links < 2:
            return {
                "links_count": n_links,
                "pair_count": 0,
                "avg_pair_similarity": 0.0,
                "high_similarity_pair_ratio": 0.0,
                "redundancy_index": 0.0,
                "penalty": 0.0,
            }

        sims: list[float] = []
        for i in range(n_links):
            for j in range(i + 1, n_links):
                sim = self._clamp01(
                    self._safe_float(self._similarity(texts[i], texts[j]), 0.0)
                )
                sims.append(sim)

        pair_count = len(sims)
        threshold = self._clamp01(
            self._safe_float(self._aqa_redundancy_similarity_threshold, 0.72)
        )
        denom = max(1e-6, 1.0 - threshold)
        high_pairs = [s for s in sims if s >= threshold]
        high_ratio = (
            float(len(high_pairs)) / float(pair_count) if pair_count > 0 else 0.0
        )
        normalized_excess = [max(0.0, (s - threshold) / denom) for s in high_pairs]
        redundancy_index = (
            sum(normalized_excess) / len(normalized_excess)
            if normalized_excess
            else 0.0
        )
        length_factor = min(1.0, float(max(0, n_links - 1)) / 4.0)
        raw_penalty = (
            self._safe_float(self._aqa_redundancy_penalty_weight, 0.0)
            * redundancy_index
            * length_factor
        )
        penalty = min(
            self._safe_float(self._aqa_redundancy_max_penalty, 0.18),
            max(0.0, raw_penalty),
        )

        return {
            "links_count": n_links,
            "pair_count": pair_count,
            "avg_pair_similarity": (sum(sims) / pair_count) if pair_count > 0 else 0.0,
            "high_similarity_pair_ratio": high_ratio,
            "redundancy_index": redundancy_index,
            "penalty": penalty,
        }

    def _link_axis_signature(self, link: dict) -> str:
        """Build a semantic signature for weak-point axis clustering."""
        return self._normalize_text(
            " ".join(
                [
                    str(link.get("premise_text", "") or ""),
                    str(link.get("text", "") or ""),
                    str(link.get("conclusion_text", "") or ""),
                ]
            )
        )

    def _cluster_reasoner_axes(self, pro_links: list[dict]) -> list[dict]:
        """
        Cluster reasoner links into weak-point axes using semantic similarity.

        Axis clustering is intentionally simple/greedy for runtime stability.
        """
        if not pro_links:
            return []
        threshold = self._clamp01(
            self._safe_float(self._aqa_attack_coverage_similarity_threshold, 0.78)
        )

        ordered = sorted(
            pro_links,
            key=lambda x: self._safe_float(x.get("base_score"), 0.0),
            reverse=True,
        )
        axes: list[dict] = []
        for link in ordered:
            link_id = str(link.get("link_id") or "").strip()
            if not link_id:
                continue
            signature = self._link_axis_signature(link)
            base_score = self._clamp01(self._safe_float(link.get("base_score"), 0.0))
            if not signature:
                axes.append(
                    {
                        "axis_id": f"AX{len(axes) + 1}",
                        "representative_text": f"link:{link_id}",
                        "link_ids": [link_id],
                        "link_objects": [link],
                        "base_scores": [base_score],
                    }
                )
                continue

            best_idx = -1
            best_sim = 0.0
            for idx, axis in enumerate(axes):
                rep = str(axis.get("representative_text", "") or "")
                if not rep:
                    continue
                sim = self._clamp01(
                    self._safe_float(self._similarity(signature, rep), 0.0)
                )
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx

            if best_idx >= 0 and best_sim >= threshold:
                axis = axes[best_idx]
                axis["link_ids"].append(link_id)
                axis["link_objects"].append(link)
                axis["base_scores"].append(base_score)
            else:
                axes.append(
                    {
                        "axis_id": f"AX{len(axes) + 1}",
                        "representative_text": signature,
                        "link_ids": [link_id],
                        "link_objects": [link],
                        "base_scores": [base_score],
                    }
                )
        return axes

    def _compute_attack_coverage_bonus(self, pro_links: list[dict]) -> dict:
        """
        Compute Weak-Point Coverage Score (WPCS) for counter attacks on reasoner axes.

        The score rewards distinct, credible coverage of pro-side weak points.
        """
        if not bool(getattr(self, "_aqa_attack_coverage_enabled", True)):
            return {
                "enabled": False,
                "axes_total": 0,
                "axes_covered": 0,
                "coverage_ratio": 0.0,
                "quality_avg": 0.0,
                "wpcs": 0.0,
                "bonus": 0.0,
                "axis_details": [],
            }

        axes = self._cluster_reasoner_axes(pro_links)
        if not axes:
            return {
                "enabled": True,
                "axes_total": 0,
                "axes_covered": 0,
                "coverage_ratio": 0.0,
                "quality_avg": 0.0,
                "wpcs": 0.0,
                "bonus": 0.0,
                "axis_details": [],
            }

        overlap_threshold = self._clamp01(
            self._safe_float(self._aqa_attack_coverage_overlap_threshold, 0.45)
        )
        min_attack_value = self._clamp01(
            self._safe_float(self._aqa_attack_coverage_min_attack_value, 0.08)
        )
        w2 = self._clamp01(
            self._safe_float(self._aqa_attack_coverage_second_hit_weight, 0.30)
        )
        w3 = self._clamp01(
            self._safe_float(self._aqa_attack_coverage_third_hit_weight, 0.10)
        )

        axes_total = len(axes)
        axes_covered = 0
        quality_values: list[float] = []
        axis_details: list[dict] = []

        for axis in axes:
            active_vals: list[float] = []
            active_count = 0
            for link in axis.get("link_objects", []):
                attacks = link.get("attacks_received", []) or []
                for atk in attacks:
                    if not isinstance(atk, dict):
                        continue
                    if str(atk.get("attacker_role", "")).strip().lower() != "contra":
                        continue
                    if atk.get("filtered", False):
                        continue
                    attack_value = self._safe_float(atk.get("attack_value"), 0.0)
                    overlap = self._safe_float(atk.get("overlap"), 0.0)
                    if attack_value < min_attack_value:
                        continue
                    if overlap < overlap_threshold:
                        continue
                    active_vals.append(self._clamp01(attack_value))
                    active_count += 1

            active_vals.sort(reverse=True)
            axis_quality = 0.0
            if active_vals:
                axis_quality += active_vals[0]
                if len(active_vals) > 1:
                    axis_quality += w2 * active_vals[1]
                if len(active_vals) > 2:
                    axis_quality += w3 * active_vals[2]
            axis_quality = self._clamp01(axis_quality)
            if axis_quality > 0:
                axes_covered += 1
            quality_values.append(axis_quality)
            axis_details.append(
                {
                    "axis_id": axis.get("axis_id"),
                    "link_ids": list(axis.get("link_ids", [])),
                    "active_attacks": active_count,
                    "axis_quality": axis_quality,
                    "axis_base_avg": (
                        sum(axis.get("base_scores", []))
                        / len(axis.get("base_scores", []))
                        if axis.get("base_scores")
                        else 0.0
                    ),
                }
            )

        coverage_ratio = (
            float(axes_covered) / float(axes_total) if axes_total > 0 else 0.0
        )
        quality_avg = (
            sum(quality_values) / len(quality_values) if quality_values else 0.0
        )
        wpcs = self._clamp01(0.4 * coverage_ratio + 0.6 * quality_avg)
        raw_bonus = (
            self._safe_float(self._aqa_attack_coverage_bonus_weight, 0.12) * wpcs
        )
        bonus = min(
            self._safe_float(self._aqa_attack_coverage_max_bonus, 0.15),
            max(0.0, raw_bonus),
        )
        axis_details.sort(
            key=lambda x: self._safe_float(x.get("axis_quality"), 0.0), reverse=True
        )
        return {
            "enabled": True,
            "axes_total": axes_total,
            "axes_covered": axes_covered,
            "coverage_ratio": coverage_ratio,
            "quality_avg": quality_avg,
            "wpcs": wpcs,
            "bonus": bonus,
            "axis_details": axis_details[:8],
        }

    def _extract_year(self, text: str) -> int | None:
        if not text:
            return None
        matches = re.findall(r"(?:19|20)\d{2}", text)
        if not matches:
            return None
        try:
            return int(matches[-1])
        except ValueError:
            return None

    def _compute_recency(self, year: int | None) -> float:
        if not year:
            return 0.0
        current_year = datetime.now().year
        age = max(0, current_year - year)
        max_age = settings.aqa_max_age
        return self._clamp01(1.0 - (age / max_age))

    def _stance_from_link_type(self, link_type: str | None) -> int:
        link_type = (link_type or "").lower()
        if link_type in {"supports", "support"}:
            return 1
        if link_type in {"attacks", "attack", "contradicts", "opposes", "against"}:
            return -1
        return 0

    def _build_precedent_text(self, meta: dict) -> str:
        parts = [
            meta.get("title") or "",
            meta.get("summary") or "",
        ]
        return self._normalize_text(" ".join(p for p in parts if p))

    def _build_precedent_influence(
        self, prec_meta: dict, link: dict, edge: dict, match_reason: str
    ) -> dict:
        link_text = self._normalize_text(
            " ".join(
                [
                    link.get("premise_text", ""),
                    link.get("text", ""),
                    link.get("conclusion_text", ""),
                ]
            )
        )
        prec_text = self._build_precedent_text(prec_meta)
        similarity = prec_meta.get("similarity")
        if similarity is None:
            similarity = prec_meta.get("score")
        if similarity is None and prec_text and link_text:
            similarity = self._similarity(link_text, prec_text)
        similarity = self._clamp01(self._safe_float(similarity, 0.0))

        stance = self._stance_from_link_type(edge.get("type"))

        court = (
            prec_meta.get("court")
            or prec_meta.get("court_level")
            or prec_meta.get("organo")
            or ""
        )
        # If no explicit court field, search title+summary for court keywords
        if not court:
            court = f"{prec_meta.get('title', '')} {prec_meta.get('summary', '')}"
        bindingness = prec_meta.get("bindingness")
        if bindingness is None:
            bindingness = self._bindingness_from_court(court)
        bindingness = self._clamp01(self._safe_float(bindingness, 0.0))

        year = prec_meta.get("year") or self._extract_year(
            f"{prec_meta.get('title','')} {prec_meta.get('summary','')}"
        )
        recency = prec_meta.get("recency")
        if recency is None:
            recency = self._compute_recency(year)
        recency = self._clamp01(self._safe_float(recency, 0.0))

        confidence = prec_meta.get("confidence")
        if confidence is None:
            confidence = prec_meta.get("stance_confidence")
        if confidence is None:
            confidence = settings.aqa_default_confidence
        confidence = self._clamp01(
            self._safe_float(confidence, settings.aqa_default_confidence)
        )

        prec_id = prec_meta.get("precedent_id") or prec_meta.get("title")
        delta_preview = bindingness * similarity * recency * stance * confidence
        self._log(
            f"       📊 Precedent {prec_id}: bind={bindingness:.2f} "
            f"sim={similarity:.2f} rec={recency:.2f} stance={stance} "
            f"conf={confidence:.2f} → δ={delta_preview:.4f} "
            f"(court='{court[:60]}…' year={year})"
        )
        return {
            "precedent_id": prec_id,
            "stance": stance,
            "bindingness": bindingness,
            "similarity": similarity,
            "recency": recency,
            "confidence": confidence,
            "court": court,
            "year": year,
            "link_match": match_reason,
        }

    def _populate_precedent_influences(self, aspic_ir: dict, links: list[dict]) -> None:
        if not aspic_ir or not links:
            return
        precedent_nodes = {
            node.get("id"): node
            for node in aspic_ir.get("precedent_nodes", [])
            if node.get("id")
        }
        precedent_by_id = {
            node.get("precedent_id"): node
            for node in aspic_ir.get("precedent_nodes", [])
            if node.get("precedent_id")
        }

        target_to_link: dict[str, dict] = {}
        for link in links:
            targets = set()
            for key in (
                link.get("link_id"),
                link.get("argument_id"),
                link.get("rule_id"),
                link.get("conclusion_id"),
            ):
                if key:
                    targets.add(key)
            for pid in link.get("premise_ids", []) or []:
                if pid:
                    targets.add(pid)
            link["precedent_influences"] = link.get("precedent_influences", [])
            for target in targets:
                target_to_link[target] = link

        step_text_by_id = {
            step.get("id"): self._normalize_text(step.get("text", ""))
            for step in aspic_ir.get("reasoning_chain", [])
            if step.get("id")
        }

        used = set()
        for edge in aspic_ir.get("precedent_links", []) or []:
            prec_node_id = edge.get("from") or ""
            target_id = edge.get("to") or ""
            prec_meta = precedent_nodes.get(prec_node_id) or precedent_by_id.get(
                prec_node_id
            )
            if not prec_meta:
                continue

            matched_link: dict | None = target_to_link.get(target_id)
            match_reason = "direct"
            if matched_link is None and target_id in step_text_by_id:
                step_text = step_text_by_id.get(target_id, "")
                best_link = None
                best_score = 0.0
                for candidate in links:
                    candidate_text = self._normalize_text(
                        " ".join(
                            [
                                candidate.get("premise_text", ""),
                                candidate.get("text", ""),
                                candidate.get("conclusion_text", ""),
                            ]
                        )
                    )
                    score = self._similarity(step_text, candidate_text)
                    if score > best_score:
                        best_score = score
                        best_link = candidate
                if best_link and best_score >= settings.aqa_precedent_sim_threshold:
                    matched_link = best_link
                    match_reason = "chain_step_similarity"

            if matched_link is None:
                continue

            prec_id = prec_meta.get("precedent_id") or prec_meta.get("title")
            link_id = matched_link.get("link_id")
            dedup_key = (link_id, prec_id)
            if dedup_key in used:
                continue
            used.add(dedup_key)

            influence = self._build_precedent_influence(
                prec_meta=prec_meta,
                link=matched_link,
                edge=edge,
                match_reason=match_reason,
            )
            matched_link["precedent_influences"].append(influence)

    def _compute_precedent_delta(self, link: dict) -> tuple[float, list[dict]]:
        influences = []
        total_delta = 0.0
        precedents = link.get("precedent_influences") or []
        for prec in precedents:
            stance = prec.get("stance", 0)
            if isinstance(stance, str):
                stance_norm = stance.strip().lower()
                if stance_norm in {"support", "pro", "favour"}:
                    stance = 1
                elif stance_norm in {"contradict", "contra", "against"}:
                    stance = 0
                else:
                    stance = 0
            try:
                stance = float(stance)
            except Exception:
                stance = 0.0
            # Defensive clamp: no negative precedent contribution.
            if stance < 0:
                stance = 0.0
            elif stance > 0:
                stance = 1.0
            bindingness = prec.get("bindingness", 0.0)
            if not bindingness:
                court = (
                    prec.get("court")
                    or prec.get("court_level")
                    or prec.get("organo")
                    or ""
                )
                bindingness = self._bindingness_from_court(court)
            similarity = prec.get("similarity", 0.0)
            recency = prec.get("recency", 0.0)
            confidence = prec.get("confidence", 0.0)
            try:
                delta = (
                    float(bindingness)
                    * float(similarity)
                    * float(recency)
                    * float(stance)
                    * float(confidence)
                )
            except Exception:
                delta = 0.0
            influences.append(
                {
                    "precedent_id": prec.get("precedent_id"),
                    "delta": delta,
                    "stance": stance,
                    "bindingness": bindingness,
                    "similarity": similarity,
                    "recency": recency,
                    "confidence": confidence,
                    "court": prec.get("court"),
                    "year": prec.get("year"),
                    "link_match": prec.get("link_match"),
                }
            )
            total_delta += delta
        return total_delta, influences

    def _bindingness_from_court(self, court: str) -> float:
        court_norm = (court or "").lower()
        bmap = settings.aqa_bindingness_map
        if "consiglio di stato" in court_norm:
            return bmap.get("cassazione", 1.0)
        if "corte costituzionale" in court_norm:
            return bmap.get("cassazione", 1.0)
        if "consiglio di giustizia amministrativa" in court_norm:
            return bmap.get("appello", 0.7)
        if "tribunale amministrativo regionale" in court_norm or re.search(
            r"\btar\b", court_norm
        ):
            return bmap.get("tribunale", 0.4)
        if "cass" in court_norm:
            return bmap.get("cassazione", 1.0)
        if "appello" in court_norm:
            return bmap.get("appello", 0.7)
        if "tribunale" in court_norm:
            return bmap.get("tribunale", 0.4)
        return bmap.get("other", 0.0)

    def _run_aqa_phase(
        self,
        reasoner_ir: dict,
        counter_ir: dict,
        domain: str,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        def _emit(
            stage: str, message: str, progress: float | None = None, **extra
        ) -> None:
            if not progress_callback:
                return
            payload = {"stage": stage, "message": message}
            if progress is not None:
                payload["progress"] = max(0.0, min(1.0, float(progress)))
            if extra:
                payload.update(extra)
            try:
                progress_callback(payload)
            except Exception:
                pass

        if not self._aqa_enabled:
            self._log("🧪 AQA disabled - skipping scoring")
            return {"enabled": False}
        self._log("🧪 Starting AQA scoring on REPAIRED reasoning chains...")
        self._log(f"🧠 AQA domain: {domain}")
        _emit("start", "AQA avviata: costruzione link", 0.02)

        # Check if we're using repaired IRs
        reasoner_repaired = bool(reasoner_ir.get("_repair_metadata"))
        counter_repaired = bool(counter_ir.get("_repair_metadata"))
        if reasoner_repaired or counter_repaired:
            self._log(
                f"   ✅ Using repaired chains: Reasoner={reasoner_repaired}, Counter={counter_repaired}"
            )
        else:
            self._log("   ℹ️ No repairs were needed - using original chains")

        pro_links = self._collect_links(reasoner_ir, "support", domain)
        contra_links = self._collect_links(counter_ir, "counter", domain)
        total_links = len(pro_links) + len(contra_links)
        self._log(
            f"🔗 AQA links collected: pro={len(pro_links)} contra={len(contra_links)}"
        )
        _emit(
            "links_collected",
            f"ASPIC+ links collected: pro={len(pro_links)}, counter={len(contra_links)}",
            0.12,
            pro_links=len(pro_links),
            contra_links=len(contra_links),
            total_links=total_links,
        )
        self._populate_precedent_influences(reasoner_ir, pro_links)
        self._populate_precedent_influences(counter_ir, contra_links)
        _emit("precedents_linked", "Precedent alignment completed", 0.2)

        for idx, link in enumerate(pro_links + contra_links, start=1):
            link_id = link.get("link_id") or f"L{idx}"
            role = (link.get("role") or "link").upper()
            self._log(f"   🔹 [{idx}/{total_links}] {role} link {link_id}")
            premise_text = link.get("premise_text", "")
            rule_text = link.get("text", "")
            conclusion_text = link.get("conclusion_text", "")

            # Fallback: if premise_text is empty, use rule_text for semantics
            semantics_premise = premise_text if premise_text else rule_text
            if not premise_text and rule_text:
                self._log("      ⚠️ premise_text empty, using rule_text for semantics")

            semantics, semantics_details = self._semantics_score(
                semantics_premise, conclusion_text
            )
            # Add fallback info to semantics_details
            if not premise_text and rule_text:
                semantics_details["used_fallback"] = True
                semantics_details["fallback_source"] = "rule_text"

            coherence = self._coherence_score(premise_text, conclusion_text, rule_text)
            arg_quality = self._argument_quality_score(
                premise_text, rule_text, conclusion_text
            )
            # Cogency: average of coherence and argument_quality (readability removed)
            cogency = self._clamp01((coherence + arg_quality) / 2.0)
            cogency_details = {
                "coherence_score": coherence,
                "argument_quality_score": arg_quality,
                "explanation": "coherence/argument_quality averaged (readability removed)",
            }

            # ----------------------------------------------------------
            # Per-link NormSupport (β component)
            # ----------------------------------------------------------
            link_citations_text = f"{premise_text} {conclusion_text}"
            norm_support_link, norm_details_link = self._norm_support_score(
                text=link_citations_text,
                retrieved_norms=link.get("retrieved_norms"),
            )

            # Link base_score = α·cogency + β·normSupport + γ·semantics
            base = self._clamp01(
                self._aqa_alpha * cogency
                + self._aqa_beta * norm_support_link
                + self._aqa_gamma * semantics
            )

            self._log(
                "      🧠 Cogency "
                f"{cogency:.2f} (coh {coherence:.2f}, qual {arg_quality:.2f})"
            )
            self._log(
                "      📚 NormSupport "
                f"{norm_support_link:.2f} (cit={norm_details_link.get('citation_count', 0)})"
            )
            self._log(
                "      🧩 Semantics "
                f"{semantics:.2f} ({semantics_details.get('method', 'n/a')})"
            )
            self._log(f"      ➕ Link base score {base:.2f}")
            if link.get("libro") or link.get("severity_category"):
                self._log(
                    "      🗂️ Severity "
                    f"{link.get('severity_category') or '-'} / libro "
                    f"{link.get('libro') or '-'}"
                )

            link["cogency"] = cogency
            link["semantics"] = semantics
            link["norm_support"] = norm_support_link
            link["cogency_details"] = cogency_details
            link["semantics_details"] = semantics_details
            link["norm_support_details"] = norm_details_link
            link["base_score"] = base
            if total_links > 0:
                _emit(
                    "link_scored",
                    f"Scoring link {idx}/{total_links}",
                    0.2 + (idx / total_links) * 0.4,
                    current=idx,
                    total=total_links,
                    link_id=link_id,
                    role=role.lower(),
                )

        # ==============================================================
        # Cross-attacks (ASPIC+ bidirectional)
        # ==============================================================
        self._log("⚔️ Computing bidirectional cross-attacks (ASPIC+)...")
        _emit("attacks_start", "Bidirectional attack computation in progress", 0.64)
        self._compute_cross_attacks(
            pro_links,
            contra_links,
            progress_callback=lambda payload: _emit(
                payload.get("stage", "attacks_progress"),
                payload.get("message", "Attack computation"),
                0.64
                + (
                    (
                        payload.get("current_target", 0)
                        / max(1, payload.get("total_targets", 1))
                    )
                    * 0.18
                ),
                direction=payload.get("direction"),
                current_target=payload.get("current_target"),
                total_targets=payload.get("total_targets"),
                target_link_id=payload.get("target_link_id"),
            ),
        )
        _emit("attacks_done", "Attack computation completed", 0.84)

        # Log attack summaries (active + filtered breakdown)
        for link in pro_links + contra_links:
            all_attacks = link.get("attacks_received", [])
            active = [a for a in all_attacks if a.get("attack_value", 0.0) > 0]
            filtered = [a for a in all_attacks if a.get("filtered", False)]
            role_tag = (link.get("role") or "link").upper()
            link_id = link.get("link_id", "?")

            if active:
                self._log(
                    f"   \u2694\ufe0f [{role_tag}] {link_id} received "
                    f"{len(active)} active attack(s), "
                    f"\u03a3attack={link.get('attacks_sum', 0.0):.4f}"
                )

            if filtered:
                # Breakdown by filter stage
                from collections import Counter as _Counter

                stage_counts = _Counter(
                    a.get("filter_stage", "unknown") for a in filtered
                )
                parts = [
                    f"{stage}={cnt}" for stage, cnt in sorted(stage_counts.items())
                ]
                self._log(
                    f"   \U0001f6ab [{role_tag}] {link_id} "
                    f"filtered {len(filtered)} attack(s) "
                    f"({', '.join(parts)})"
                )
                # Detail: log each filtered attack reason
                for fa in filtered:
                    self._log(
                        f"      \u21b3 attacker={fa.get('attacker_link_id')} "
                        f"({fa.get('attacker_role')}) \u2014 {fa.get('reason')}"
                    )

            zero_damage = [
                a
                for a in all_attacks
                if not a.get("filtered", False) and a.get("attack_value", 0.0) <= 0
            ]
            if zero_damage:
                self._log(
                    f"   \u26a0\ufe0f [{role_tag}] {link_id} "
                    f"{len(zero_damage)} attack(s) passed filters but zero/negative damage"
                )
                for zd in zero_damage:
                    self._log(
                        f"      \u21b3 attacker={zd.get('attacker_link_id')} "
                        f"({zd.get('attacker_role')}) \u2014 "
                        f"damage={zd.get('attack_value', 0.0):.4f}, "
                        f"type={zd.get('attack_type')}, "
                        f"overlap={zd.get('overlap', 0.0):.3f}"
                    )

            if not active and not filtered and not zero_damage:
                self._log(
                    f"   \u2139\ufe0f [{role_tag}] {link_id} no attacks attempted"
                )

        # ==============================================================
        # Precedent delta + nesso_plausibility per link
        # ==============================================================
        # Formula: nesso_plausibility = clamp01(base_score - Σattacks ± δ)
        for idx, link in enumerate(pro_links + contra_links, start=1):
            link_id = link.get("link_id") or f"L{idx}"
            role = (link.get("role") or "link").upper()
            delta, influences = self._compute_precedent_delta(link)
            link["precedent_delta"] = delta
            link["precedent_influences"] = influences
            # Intrinsic link strength (independent from incoming attacks).
            intrinsic_nesso = self._clamp01(link.get("base_score", 0.0) + delta)
            link["nesso_intrinsic"] = intrinsic_nesso
            nesso = self._clamp01(
                link.get("base_score", 0.0) - link.get("attacks_sum", 0.0) + delta
            )
            link["nesso_plausibility"] = nesso
            self._log(
                "   🔸 "
                f"[{idx}/{total_links}] {role} link {link_id} "
                f"base={link.get('base_score', 0.0):.2f} "
                f"- attacks={link.get('attacks_sum', 0.0):.2f} "
                f"+ Δprec={delta:.2f} "
                f"→ nesso={nesso:.2f}"
            )
            if influences:
                self._log(f"      📚 Precedent influences: {len(influences)}")
            if total_links > 0:
                _emit(
                    "aggregation_progress",
                    f"Aggregazione finale link {idx}/{total_links}",
                    0.84 + (idx / total_links) * 0.12,
                    current=idx,
                    total=total_links,
                    link_id=link_id,
                    role=role.lower(),
                )

        # ==============================================================
        # Chain-level aggregation
        # ==============================================================
        # net_plausibility = mean(nesso_plausibility_i) per side
        # final = net_PRO − net_CONTRA
        def avg_field(items: list[dict], field: str) -> float:
            if not items:
                return 0.0
            return sum(i.get(field, 0.0) for i in items) / len(items)

        # Dialectical pro score includes incoming counter attacks; intrinsic does not.
        pro_net_dialectical_raw = avg_field(pro_links, "nesso_plausibility")
        pro_net_intrinsic_raw = avg_field(pro_links, "nesso_intrinsic")
        lock_reasoner = bool(getattr(self, "_aqa_lock_reasoner_plausibility", False))
        pro_net_raw = (
            pro_net_intrinsic_raw if lock_reasoner else pro_net_dialectical_raw
        )
        contra_net_raw = avg_field(contra_links, "nesso_plausibility")
        final_plausibility_raw = pro_net_raw - contra_net_raw

        pro_redundancy = self._compute_redundancy_penalty(pro_links)
        contra_redundancy = self._compute_redundancy_penalty(contra_links)
        contra_attack_coverage = self._compute_attack_coverage_bonus(pro_links)

        structural_enabled = bool(self._aqa_structural_adjustments_enabled)
        if structural_enabled:
            # In counter-ablation mode keep Reasoner-side score locked.
            if lock_reasoner:
                pro_net_adjusted = pro_net_raw
            else:
                pro_net_adjusted = self._clamp01(
                    pro_net_raw - pro_redundancy.get("penalty", 0.0)
                )
            contra_cov_bonus = self._safe_float(
                contra_attack_coverage.get("bonus"), 0.0
            )
            contra_net_adjusted = self._clamp01(
                contra_net_raw
                - contra_redundancy.get("penalty", 0.0)
                + contra_cov_bonus
            )
        else:
            pro_net_adjusted = pro_net_raw
            contra_net_adjusted = contra_net_raw

        final_plausibility_adjusted = pro_net_adjusted - contra_net_adjusted
        use_adjusted = bool(self._aqa_verdict_use_adjusted_score)
        if structural_enabled and use_adjusted:
            pro_net = pro_net_adjusted
            contra_net = contra_net_adjusted
            final_plausibility = final_plausibility_adjusted
        else:
            pro_net = pro_net_raw
            contra_net = contra_net_raw
            final_plausibility = final_plausibility_raw

        # Component averages (kept for reporting / explainability)
        pro_cogency_avg = avg_field(pro_links, "cogency")
        pro_semantics_avg = avg_field(pro_links, "semantics")
        pro_norm_support_avg = avg_field(pro_links, "norm_support")
        pro_base_avg = avg_field(pro_links, "base_score")
        pro_attacks_avg = avg_field(pro_links, "attacks_sum")
        pro_precedent_delta = avg_field(pro_links, "precedent_delta")

        contra_cogency_avg = avg_field(contra_links, "cogency")
        contra_semantics_avg = avg_field(contra_links, "semantics")
        contra_norm_support_avg = avg_field(contra_links, "norm_support")
        contra_base_avg = avg_field(contra_links, "base_score")
        contra_attacks_avg = avg_field(contra_links, "attacks_sum")
        contra_precedent_delta = avg_field(contra_links, "precedent_delta")

        self._log(
            f"📊 Chain averages: "
            f"pro(cog={pro_cogency_avg:.2f}, norm={pro_norm_support_avg:.2f}, "
            f"sem={pro_semantics_avg:.2f}), "
            f"contra(cog={contra_cogency_avg:.2f}, norm={contra_norm_support_avg:.2f}, "
            f"sem={contra_semantics_avg:.2f})"
        )
        self._log(
            f"📚 base_score = α({self._aqa_alpha:.2f})*cogency "
            f"+ β({self._aqa_beta:.2f})*normSupport "
            f"+ γ({self._aqa_gamma:.2f})*semantics"
        )
        self._log(
            f"📈 Chain base avg: pro={pro_base_avg:.2f}, "
            f"contra={contra_base_avg:.2f}"
        )
        self._log(
            f"⚔️ Chain attacks avg: pro={pro_attacks_avg:.2f}, "
            f"contra={contra_attacks_avg:.2f}"
        )
        self._log(
            f"⚖️ Chain precedent Δ avg: pro={pro_precedent_delta:+.2f}, "
            f"contra={contra_precedent_delta:+.2f}"
        )
        self._log(
            f"📈 net_plausibility: pro={pro_net:.2f}, "
            f"contra={contra_net:.2f}, final={final_plausibility:+.2f}"
        )
        if structural_enabled:
            self._log(
                "🧱 Structural adjust PRO: "
                f"-red={pro_redundancy.get('penalty', 0.0):.3f} "
                f"(raw={pro_net_raw:.3f} -> adj={pro_net_adjusted:.3f})"
            )
            self._log(
                "🧱 Structural adjust CONTRA: "
                f"-red={contra_redundancy.get('penalty', 0.0):.3f} "
                f"+cov={contra_attack_coverage.get('bonus', 0.0):.3f} "
                f"(raw={contra_net_raw:.3f} -> adj={contra_net_adjusted:.3f})"
            )
            self._log(
                "🎯 Counter attack-coverage: "
                f"axes={contra_attack_coverage.get('axes_covered', 0)}/"
                f"{contra_attack_coverage.get('axes_total', 0)}, "
                f"coverage={contra_attack_coverage.get('coverage_ratio', 0.0):.3f}, "
                f"wpcs={contra_attack_coverage.get('wpcs', 0.0):.3f}"
            )
            self._log(
                "🧱 Final raw/adj: "
                f"{final_plausibility_raw:+.3f} -> {final_plausibility_adjusted:+.3f} "
                f"(effective={'adjusted' if (structural_enabled and use_adjusted) else 'raw'})"
            )
        if lock_reasoner:
            self._log(
                "🧷 Reasoner net locked to intrinsic plausibility "
                f"(dialectical={pro_net_dialectical_raw:.3f}, intrinsic={pro_net_intrinsic_raw:.3f})"
            )

        if final_plausibility >= self._aqa_verdict_pos:
            verdict = "plausible"
        elif final_plausibility <= self._aqa_verdict_neg:
            verdict = "implausible"
        else:
            verdict = "uncertain"

        self._log(
            "🧾 AQA verdict "
            f"{verdict} (pos≥{self._aqa_verdict_pos:.2f}, "
            f"neg≤{self._aqa_verdict_neg:.2f})"
        )
        _emit("done", f"AQA completed: verdict {verdict}", 1.0, verdict=verdict)

        weakest = sorted(
            pro_links + contra_links,
            key=lambda x: x.get("nesso_plausibility", 0.0),
        )[:3]
        if weakest:
            weakest_ids = ", ".join(
                str(item.get("link_id") or "link") for item in weakest
            )
            self._log(f"🔎 Weakest links: {weakest_ids}")

        # Collect dominant attacks for reporting
        dominant_attacks = []
        for link in pro_links + contra_links:
            target_role = link.get("role", "")
            for atk in link.get("attacks_received", []):
                if atk.get("attack_value", 0.0) > 0:
                    dominant_attacks.append(
                        {
                            "target": link.get("link_id"),
                            "target_role": target_role,
                            "attacker": atk.get("attacker_link_id"),
                            "attacker_role": atk.get("attacker_role", ""),
                            "value": atk.get("attack_value"),
                            "overlap": atk.get("overlap"),
                        }
                    )
        dominant_attacks.sort(key=lambda x: x.get("value", 0.0), reverse=True)
        dominant_attacks = dominant_attacks[: settings.aqa_dominant_attacks_limit]
        if dominant_attacks:
            self._log(f"⚔️ Dominant attacks: {len(dominant_attacks)}")

        precedent_swings = [
            {
                "link_id": link.get("link_id"),
                "delta": link.get("precedent_delta"),
            }
            for link in pro_links + contra_links
            if abs(link.get("precedent_delta", 0.0)) > 0.0
        ]
        if precedent_swings:
            self._log(f"📚 Precedent swings: {len(precedent_swings)}")

        severity_debug = [
            {
                "link_id": link.get("link_id"),
                "role": link.get("role"),
                "source": link.get("source"),
                "libro": link.get("libro"),
                "severity_category": link.get("severity_category"),
            }
            for link in pro_links + contra_links
        ]

        return {
            "enabled": True,
            "weights": {
                "alpha": self._aqa_alpha,
                "beta": self._aqa_beta,
                "gamma": self._aqa_gamma,
            },
            "links": {
                "pro": pro_links,
                "contra": contra_links,
            },
            "chain_scores": {
                "pro": {
                    "cogency_avg": pro_cogency_avg,
                    "semantics_avg": pro_semantics_avg,
                    "norm_support_avg": pro_norm_support_avg,
                    "base_score_avg": pro_base_avg,
                    "attacks_avg": pro_attacks_avg,
                    "precedent_delta_avg": pro_precedent_delta,
                    "net_plausibility": pro_net,
                    "net_plausibility_raw": pro_net_raw,
                    "net_plausibility_dialectical_raw": pro_net_dialectical_raw,
                    "net_plausibility_intrinsic_raw": pro_net_intrinsic_raw,
                    "net_plausibility_adjusted": pro_net_adjusted,
                    "redundancy_penalty": pro_redundancy.get("penalty", 0.0),
                    "redundancy_index": pro_redundancy.get("redundancy_index", 0.0),
                    "reasoner_plausibility_locked": lock_reasoner,
                },
                "contra": {
                    "cogency_avg": contra_cogency_avg,
                    "semantics_avg": contra_semantics_avg,
                    "norm_support_avg": contra_norm_support_avg,
                    "base_score_avg": contra_base_avg,
                    "attacks_avg": contra_attacks_avg,
                    "precedent_delta_avg": contra_precedent_delta,
                    "net_plausibility": contra_net,
                    "net_plausibility_raw": contra_net_raw,
                    "net_plausibility_adjusted": contra_net_adjusted,
                    "redundancy_penalty": contra_redundancy.get("penalty", 0.0),
                    "redundancy_index": contra_redundancy.get("redundancy_index", 0.0),
                    "attack_coverage_bonus": contra_attack_coverage.get("bonus", 0.0),
                    "attack_coverage_score": contra_attack_coverage.get("wpcs", 0.0),
                    "attack_coverage_ratio": contra_attack_coverage.get(
                        "coverage_ratio", 0.0
                    ),
                    "attack_coverage_axes_covered": contra_attack_coverage.get(
                        "axes_covered", 0
                    ),
                    "attack_coverage_axes_total": contra_attack_coverage.get(
                        "axes_total", 0
                    ),
                },
            },
            "net_plausibility": {
                "pro": pro_net,
                "contra": contra_net,
                "final": final_plausibility,
                "pro_raw": pro_net_raw,
                "pro_raw_dialectical": pro_net_dialectical_raw,
                "pro_raw_intrinsic": pro_net_intrinsic_raw,
                "contra_raw": contra_net_raw,
                "final_raw": final_plausibility_raw,
                "pro_adjusted": pro_net_adjusted,
                "contra_adjusted": contra_net_adjusted,
                "final_adjusted": final_plausibility_adjusted,
                "uses_adjusted_score": bool(structural_enabled and use_adjusted),
            },
            "verdict": verdict,
            "notes": {
                "weakest_links": weakest,
                "dominant_attacks": dominant_attacks,
                "attacks_enabled": True,
                "precedent_swings": precedent_swings,
                "severity_debug": severity_debug,
                "structural_adjustments": {
                    "enabled": structural_enabled,
                    "verdict_uses_adjusted_score": bool(
                        structural_enabled and use_adjusted
                    ),
                    "reasoner_plausibility_locked": lock_reasoner,
                    "pro": {
                        "redundancy": pro_redundancy,
                    },
                    "contra": {
                        "redundancy": contra_redundancy,
                        "attack_coverage": contra_attack_coverage,
                    },
                },
            },
        }
