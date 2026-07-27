"""Prompt registry for LexCausa.

This module contains:
- prompt identifiers (PromptKey)
- prompt templates (PROMPTS)
- helper functions (get_prompt / render_prompt)
"""

from __future__ import annotations

import re
import threading
from enum import StrEnum

_lang_local = threading.local()


def set_response_language(lang: str) -> None:
    """Set the response language for the current thread (request-scoped override)."""
    _lang_local.value = (lang or "it").lower()


def get_response_language() -> str:
    """Return the current per-request language ('it' or 'en')."""
    lang = getattr(_lang_local, "value", None)
    if lang is None:
        try:
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from config import settings

            lang = (settings.response_language or "it").lower()
        except Exception:
            lang = "it"
    return lang


class PromptKey(StrEnum):
    """Prompt identifiers used by the registry."""

    CLAIM_CLASSIFIER_SYSTEM = "claim_classifier.system"
    CLAIM_CLASSIFIER_TAXONOMY_USER = "claim_classifier.taxonomy_user"
    ROUTER_SYSTEM = "router.system"
    ROUTER_USER = "router.user"
    BASE_FILTER_RELEVANT_STATUTES = "base.filter_relevant_statutes"
    BASE_EXTRACT_LEGAL_CONTEXT = "base.extract_legal_context"
    BASE_FILTER_APPLICABLE_STATUTES = "base.filter_applicable_statutes"
    BASE_FILTER_PRECEDENTS = "base.filter_precedents"
    BASE_FACT_LOCK_CHECK = "base.fact_lock_check"
    LEGAL_SEARCH_QUERY_TERMS_SYSTEM = "legal_search.query_terms_system"
    LEGAL_SEARCH_QUERY_TERMS_USER = "legal_search.query_terms_user"
    NEO4J_TOOLS_EXTRACT_KEYWORDS = "neo4j_tools.extract_keywords"
    TAXONOMY_TOOLS_CLASSIFICATION = "taxonomy_tools.classification"
    TAXONOMY_TOOLS_FILTER_NORM = "taxonomy_tools.filter_norm"
    REASONER_CLASSIFY_CAUSALITY = "reasoner.classify_causality"
    REASONER_GENERATE_PLAN = "reasoner.generate_plan"
    REASONER_SUPPORT_STEP = "reasoner.support_step"
    REASONER_SUPPORT_PLAN_REWRITE = "reasoner.support_plan_rewrite"
    REASONER_GENERATE_CONCLUSION = "reasoner.generate_conclusion"
    REASONER_SINGLE_CALL = "reasoner.single_call"
    COUNTER_REASONER_PICK_ATTACKS = "counter_reasoner.pick_attacks"
    COUNTER_REASONER_OPEN_ATTACKS = "counter_reasoner.open_attacks"
    COUNTER_REASONER_TARGET_MAP = "counter_reasoner.target_map"
    COUNTER_REASONER_DECOMPOSE_CONCLUSION = "counter_reasoner.decompose_conclusion"
    COUNTER_REASONER_GENERATE_PLAN = "counter_reasoner.generate_plan"
    COUNTER_REASONER_PLAN_TARGET_ALIGNMENT = "counter_reasoner.plan_target_alignment"
    COUNTER_REASONER_STEP_PROMPT = "counter_reasoner.step_prompt"
    COUNTER_REASONER_SINGLE_CALL = "counter_reasoner.single_call"
    COUNTER_REASONER_ATTACK_ALIGNMENT = "counter_reasoner.attack_alignment"
    COUNTER_REASONER_ATTACK_SAFETY = "counter_reasoner.attack_safety"
    COUNTER_REASONER_ATTACK_COMPATIBILITY = "counter_reasoner.attack_compatibility"
    COUNTER_REASONER_ATTACK_PRECONDITION_CHECK = (
        "counter_reasoner.attack_precondition_check"
    )
    COUNTER_REASONER_STEP_OPPOSITION_CHECK = "counter_reasoner.step_opposition_check"
    COUNTER_REASONER_STEP_EXPANSION = "counter_reasoner.step_expansion"
    COUNTER_REASONER_STANCE_REWRITE = "counter_reasoner.stance_rewrite"
    COUNTER_REASONER_FACT_LOCK_CHECK = "counter_reasoner.fact_lock_check"
    COUNTER_REASONER_NO_NEW_FACTS = "counter_reasoner.no_new_facts"
    COUNTER_REASONER_PLAN_FEASIBILITY_REWRITE = (
        "counter_reasoner.plan_feasibility_rewrite"
    )
    AQA_ENGINE_ATTACK_TYPE_SYSTEM = "aqa_engine.attack_type_system"
    AQA_ENGINE_ATTACK_TYPE_USER = "aqa_engine.attack_type_user"
    NLP_UTILS_NLI_SYSTEM = "nlp_utils.nli_system"
    NLP_UTILS_NLI_USER = "nlp_utils.nli_user"
    POLISHER_COUNTER_GATE = "polisher.counter_gate"
    CONSISTENCY_VERIFY_MISMATCH_SYSTEM = "consistency.verify_mismatch_system"
    CONSISTENCY_VERIFY_MISMATCH_USER = "consistency.verify_mismatch_user"
    CONSISTENCY_PERTINENCE_SYSTEM = "consistency.pertinence_system"
    CONSISTENCY_PERTINENCE_USER = "consistency.pertinence_user"
    CONSISTENCY_REPAIR_DB_SYSTEM = "consistency.repair_db_system"
    CONSISTENCY_REPAIR_DB_USER = "consistency.repair_db_user"
    CONSISTENCY_PRECEDENT_MISMATCH_SYSTEM = "consistency.precedent_mismatch_system"
    CONSISTENCY_PRECEDENT_MISMATCH_USER = "consistency.precedent_mismatch_user"
    CONSISTENCY_PRECEDENT_REPAIR_SYSTEM = "consistency.precedent_repair_system"
    CONSISTENCY_PRECEDENT_REPAIR_USER = "consistency.precedent_repair_user"


PROMPTS: dict[PromptKey, str] = {
    # ---------------------------------------------------------------------
    # Claim Classification & Routing
    # ---------------------------------------------------------------------
    PromptKey.CLAIM_CLASSIFIER_SYSTEM: """You are a legal-domain routing classifier for Italian law.

Your task is to assign a legal claim to the most relevant category
chosen ONLY from the provided taxonomy.

Rules:
- Output ONE category ID by default.
- Output MORE THAN ONE category ONLY if multiple categories are clearly and independently relevant.
- Output AT MOST 2 category IDs.
- If only one category applies, output ONLY one.
- Do NOT explain the decision.
- Do NOT add any text, symbols, or formatting.
- Do NOT cite articles or laws.
- Do NOT invent new categories.

You must follow these rules strictly.
If uncertain, prefer fewer categories.
The response language must be Italian.
""",
    PromptKey.CLAIM_CLASSIFIER_TAXONOMY_USER: """TAXONOMY

[[taxonomy_block]]

CLAIM
<<<
[[claim]]
>>>

Respond in Italian.""",
    PromptKey.ROUTER_SYSTEM: """You are a preliminary router for a legal causal reasoning system.
Your only task is to classify the DOMAIN of the claim:
- "CIVILE" if the claim pertains to civil law (contracts, torts, damages, compensation)
- "PENALE" if the claim pertains to criminal law (crimes, criminal liability, punishment)
- "AMMINISTRATIVO" if the claim pertains to administrative procedure/public administration acts
- "ENTRAMBI" if the claim involves multiple domains (civil/criminal/administrative)

Rules:
- Respond ONLY with compact JSON: {"domain": "CIVILE" | "PENALE" | "AMMINISTRATIVO" | "ENTRAMBI"}
- Do not add text, comments, or explanations.
- If uncertain, prefer "ENTRAMBI".
""",
    PromptKey.ROUTER_USER: """[[router_system]]

Claim:
\"\"\"[[claim]]\"\"\"

Domain options:
- CIVILE: civil liability, damages, breach of contract, tort liability
- PENALE: crimes, criminal liability, causal nexus between conduct and harmful event
- AMMINISTRATIVO: administrative procedure, access to documents, deadlines, motivation, defects of administrative acts (L. 241/1990)
- ENTRAMBI: cases involving more than one domain among civil, criminal, administrative

Respond with compact JSON.""",
    # ---------------------------------------------------------------------
    # Base Agent (shared retrieval filters)
    # ---------------------------------------------------------------------
    PromptKey.BASE_FILTER_RELEVANT_STATUTES: """Legal Claim:
"[[claim]]"

Article:
"[[article_number]] - [[article_title]] - [[article_desc]]"

Instruction:
Determine whether the main topic of the article is directly mentioned or implied in the claim.

Rules:
- Do NOT evaluate whether the article fully resolves the issue.
- Do NOT suggest any additional articles.
- Do NOT use external knowledge; only consider the claim and this article.
- Do NOT add explanations or comments.
- Answer YES in all cases with even indirect connection.
- Use NO only when the article is clearly about a different domain.
- If uncertain, answer YES.

Respond with EXACTLY one token: YES or NO.
No punctuation. No new lines. No extra spaces.
""",
    PromptKey.BASE_EXTRACT_LEGAL_CONTEXT: """You are a legal triage assistant.

Given this claim:
"[[claim]]"

Extract a compact legal context string (max 20 words) including:
- legal domain (criminal/civil/administrative/labour/commercial/etc.)
- party relationship (private-private, citizen-state, company-shareholders, etc.)
- procedural posture (investigation/trial/enforcement/contract dispute/etc.)

If uncertain, provide the most plausible generic context.

Respond with EXACTLY one short line and no extra text.""",
    PromptKey.BASE_FILTER_APPLICABLE_STATUTES: """Legal Situation:
"[[claim]]"

Legal Context: [[legal_context]][[taxonomy_role_block]]

Statute:
"[[article_number]] - [[article_title]]"
"[[article_text]]"

Question:
Does this statute APPLY to the legal situation described?

Evaluation Criteria:
1. Subject Scope: Does the statute apply to the TYPE of parties involved?
2. Substantive Scope: Does the statute regulate the LEGAL ISSUE at stake?
3. Temporal Scope: Is the statute relevant to the PROCEDURAL PHASE?

Rules:
- Answer YES only if the statute directly regulates THIS situation.
- Answer NO if it applies to a different:
  * relationship type
  * legal domain
  * offense class
  * procedural phase
- Answer NO for merely topically related statutes that belong to a different
  offense/legal institute (e.g., intentional homicide vs negligent road homicide;
  healthcare malpractice vs road accident liability).
- If the statute is a broad/general provision and a more specific statute clearly
  governs the claim, answer NO unless the general provision is still directly
  needed as a general principle to decide the case.
- Generic labels shared across unrelated sectors (e.g., "aggravating circumstances")
  are NOT enough for YES.
- If uncertain but potentially on-point, answer YES only when a competent lawyer
  could realistically use the statute in the main reasoning for this claim.

Respond with EXACTLY one token: YES or NO.""",
    PromptKey.BASE_FILTER_PRECEDENTS: """You are a senior Italian legal expert.

CLAIM (the legal case under evaluation):
"[[claim]]"

PRECEDENT:
Title: "[[title]]"[[materia_line]]
Summary: "[[summary]]"

TASK — Decide whether a competent lawyer would cite this precedent
when arguing the above claim (either to support or to counter it).

Answer YES when ANY of the following is true:
1. The precedent addresses the SAME or a closely analogous legal
   question (e.g. same offence, same cause of action, same defence).
2. The precedent establishes a legal PRINCIPLE (causation test,
   evidentiary standard, constitutional interpretation, procedural
   rule) that directly applies to the claim.
3. The factual scenario of the precedent is substantially similar to
   the claim, making the ruling transferable.

Answer NO when:
- The precedent concerns a completely unrelated area of law with no
  transferable principle (e.g. tax evasion vs. divorce).
- The connection is merely superficial (shared keywords but different
  legal substance).

If uncertain, answer YES.

Respond with EXACTLY one token: YES or NO.""",
    PromptKey.BASE_FACT_LOCK_CHECK: """You are a factual consistency checker for legal reasoning steps.

CLAIM (all explicit facts are fixed and true):
\"\"\"[[claim]]\"\"\"

CANDIDATE STEP:
\"\"\"[[candidate_step]]\"\"\"

Task:
Decide whether the candidate step CONTRADICTS any explicit fact stated in the claim.

Important rules:
- The step MAY discuss legal qualification, applicability of norms, causal inference, or legal conclusions.
- The step MAY NOT deny, reverse, or weaken an explicit factual statement in the claim.
- If the claim reports a factual finding/perizia as given, the step cannot claim that the finding is absent or says the opposite.
- If doubtful, choose CONSISTENT unless there is a clear factual contradiction.

Answer with EXACTLY one word: CONSISTENT or CONTRADICTS.
""",
    # ---------------------------------------------------------------------
    # Legal Search & Neo4j keyword extraction
    # ---------------------------------------------------------------------
    PromptKey.LEGAL_SEARCH_QUERY_TERMS_SYSTEM: """You are a legal information-retrieval assistant.
Given a claim, extract ONLY legal keywords useful for search (offenses, legal institutes, qualifications, decisive factual elements).
Prioritize legal concepts over narrative details.
Avoid generic words that create lexical noise (e.g., "problem", "situation", "event") unless legally meaningful.

Output format: only a comma-separated list, with no explanations.""",
    PromptKey.LEGAL_SEARCH_QUERY_TERMS_USER: """Extract up to [[max_terms]] keywords.
CLAIM:
[[query_text]]""",
    PromptKey.NEO4J_TOOLS_EXTRACT_KEYWORDS: """Extract the most important legal keywords from this claim.
Focus on: legal concepts, legal domains, types of offenses/violations,
key factual elements, and relevant legal categories.

CLAIM:
"[[claim]]"

RULES:
- Extract 5 to 10 keywords or short phrases (max 3 words each).
- Use Italian legal terminology.
- One keyword per line.
- Do NOT add numbering, bullets, or explanations.
- Do NOT repeat the claim.
- Output ONLY the keywords, nothing else.
""",
    # ---------------------------------------------------------------------
    # Taxonomy Tools
    # ---------------------------------------------------------------------
    PromptKey.TAXONOMY_TOOLS_CLASSIFICATION: """You are an expert legal classifier. Choose the best causal_type_id from the list.

Allowed causal_type_id values:
[[options_text]]

Rules:
- Respond ONLY with the id.
- No explanations, no extra text.

CLAIM:
[[claim]]""",
    PromptKey.TAXONOMY_TOOLS_FILTER_NORM: """Legal Claim:
"[[claim]]"

Norm from taxonomy:
"[[ref]]" - "[[role]]"

Instruction:
Assess whether this norm is relevant to the claim. Answer YES unless the norm is clearly outside the domain of the facts and legal institutes in the claim. If uncertain, YES.

Respond with a single token: YES or NO.""",
    # ---------------------------------------------------------------------
    # Reasoner
    # ---------------------------------------------------------------------
    PromptKey.REASONER_CLASSIFY_CAUSALITY: """You are a classifier. Based PRIMARILY on the CITED ARTICLES from the reasoning chain, choose the most appropriate causal_type_id.

Allowed causal_type_id values (domain=[[domain]]):
[[type_descriptions]]

Classification criteria (based on cited articles):
- If articles include Art. 646/640/624/624-bis/625 c.p. -> PEN_PROPERTY_QUALIFICATION
- If articles include Art. 595/51/610/392/393 c.p. -> PEN_RIGHTS_BALANCING
- If articles include Art. 42/43/61/62-bis/113/133 c.p. -> PEN_COLPA_GRADATION
- If articles include Art. 40/41 c.p. (without stronger signature above) -> PEN_FACTUAL or PEN_INTERVENING
- If articles include Art. 1490-1495 c.c. -> CIV_SALE_DEFECTS
- If articles include Art. 1218/1453/1455/1457/1460/1385/1382/1384 c.c. -> CIV_CONTRACT_REMEDIES
- If articles include Art. 2051/2052/1117/1123/1126 c.c. -> CIV_CUSTODY_DAMAGE
- If articles include Art. 1223/1225/1226/1227/2056 c.c. (without stronger signature above) -> CIV_REMOTENESS
- If articles include Art. 21-novies/21-quinquies L. 241/1990 -> AMM_AUTOTUTELA_BALANCE
- If articles include Art. 2-bis L. 241/1990 -> AMM_DELAY_REMEDIES
- If articles are from L. 241/1990 (without stronger signature above) -> AMM_PROCEDURAL_LEGITIMACY
- If both c.p. and c.c. signatures are strong in the same chain -> MIXED_PEN_CIV_CONCURRENCE
- Consider the combination of articles to determine the most specific causal type

If uncertain, choose the closest from the allowed list.
Respond with ONLY the causal_type_id (no JSON, no explanation, just the id).

ORIGINAL CLAIM (for context only):
[[claim]]

CITED ARTICLES FROM REASONING (primary classification basis):
[[articles_text]]

REASONING CHAIN (for context):
[[chain_text]]
""",
    PromptKey.REASONER_GENERATE_PLAN: """You are a legal planning engine for Italian law.

Create a sequential plan to analyze the claim and build a primary legal thesis.
Each step must add a materially new objective.

Return only ONE valid JSON object (no markdown, no prose, no text outside JSON)
using this schema:
{
  "steps": [
    {
      "id": "P1",
      "goal": "<specific legal objective>",
      "focus": "<single legal/factual focus>",
      "expected_norm": "N/A or article expected to be cited",
      "citation_requirement": "required | optional | none",
      "step_type": "FACTS | QUALIFICATION | CAUSAL_LINK | ELEMENTS | BALANCING | CONSEQUENCE | SYNTHESIS | OTHER",
      "novelty_key": "<unique snake_case key>"
    }
  ]
}

CLAIM:
"[[claim]]"

DOMAIN: [[routing_domain]]
ANCHOR NORMS:
[[anchor_text]]

PRINCIPLE TESTS:
[[principle_text]]

ALLOWED STATUTES:
[[statutes_list]]

ALLOWED PRECEDENTS:
[[precedents_list]]

KNOWLEDGE BASE:
[[knowledge_base]]

PLANNER MODE: [[planner_mode]]
RESUME FROM LOGICAL STEP: [[resume_from_step]]
ALREADY ACCEPTED STEPS (do not duplicate):
[[existing_steps]]

RULES:
- Number of steps must be between [[min_steps]] and [[max_steps]].
- Each step must address a DIFFERENT objective (no overlap/rephrasing).
- Steps must be ordered logically (premise -> legal qualification -> applicability -> consequence -> resulting legal assessment).
- Use only facts explicitly present in claim (no assumptions, no hypothetical factual completions).
- Treat explicit claim facts (including explicit factual findings/perizie stated in the claim) as fixed and true; attack legal implications, not the given facts.
- Prefer using different statutes across steps when possible.
- Set "citation_requirement" to:
  * "required" for norm-application / legal-qualification steps
  * "optional" for inferential bridge or factual-application steps
  * "none" only for pure synthesis/transition steps
- If "expected_norm" is not "N/A", "citation_requirement" should normally be "required".
- "step_type" must be coherent with the objective and should not repeat unless strictly necessary.
- "novelty_key" must be unique across steps and must summarize what is NEW in that step.
- If ALREADY ACCEPTED STEPS is not empty, generate only missing/remaining steps and avoid duplicate objectives.
- Keep each 'goal' and 'focus' concise (max 25 words each).
[[language_instruction]]

Return only the JSON object.
""",
    PromptKey.REASONER_SUPPORT_STEP: """You are an expert Italian jurist.
You must execute ONLY one planned reasoning step.

CLAIM:
"[[claim]]"

DOMAIN: [[routing_domain]]
ANCHOR NORMS:
[[anchor_text]]
PRINCIPLE TESTS:
[[principle_text]]

KNOWLEDGE BASE (use only these sources):
[[knowledge_base]]

ALLOWED STATUTES:
[[statutes_list]]
ALLOWED PRECEDENTS:
[[precedents_list]]

GLOBAL PLAN:
[[plan_lines]]

CURRENT STEP TO EXECUTE: [[plan_index]]
- Goal: [[plan_goal]]
- Focus: [[plan_focus]]
- Expected norm: [[plan_expected_norm]]
- Citation requirement: [[plan_citation_requirement]]
- Step type: [[plan_step_type]]
- Novelty key: [[plan_novelty_key]]

ALREADY GENERATED STEP SUMMARIES:
[[summary_lines]]

NORMS ALREADY USED: [[used_norms_text]]

HARD RULES:
- Generate EXACTLY ONE atomic step in [[response_language_name]] (2-4 sentences).
- It must advance the plan and add NEW information, not paraphrase prior steps.
- It must materially advance the legal analysis of the claim.
- It must realize the declared novelty key by adding a distinct legal point.
- Use only facts explicitly in claim.
- Do not infer unprovided facts (no assumptions, no hypothetical completions of the factual scenario).
- If citation requirement is "required", cite at least one statute.
- If citation requirement is "optional", citation is recommended but not mandatory.
- If citation requirement is "none", do not force a citation.
- If citing a precedent, include its full exact title from allowed list.
[[language_instruction]]

RESPONSE FORMAT:
STEP: [atomic step]
""",
    PromptKey.REASONER_SUPPORT_PLAN_REWRITE: """[[previous_prompt]]

YOUR PREVIOUS STEP WAS REJECTED.
REASON: [[invalid_reason]]
INVALID STEP:
[[invalid_step]]

Rewrite only this step. Keep the same planned objective, but produce NEW,
non-redundant and logically consistent content.
RESPONSE FORMAT:
STEP: [atomic step]
[[language_instruction]]""",
    PromptKey.REASONER_GENERATE_CONCLUSION: """You are an expert Italian jurist. Based on the legal reasoning chain below, generate a concise and precise CONCLUSION.

ORIGINAL CLAIM:
"[[claim]]"

REASONING CHAIN:
[[chain_text]]

CITED NORMS: [[norms_text]]

INSTRUCTIONS:
- Write a conclusion of 2-4 sentences.
- The conclusion must SYNTHESIZE the result of the legal analysis, not repeat the individual steps.
- Clearly state the resulting legal assessment/qualification and WHY, based on the norms analyzed.
- Do NOT introduce norms or facts not mentioned in the reasoning chain.
- Be direct and assertive in the final verdict.
[[language_instruction]]

        CONCLUSION:""",
    PromptKey.REASONER_SINGLE_CALL: """You are an expert Italian jurist. Construct the COMPLETE supporting legal argument for the claim below in a SINGLE response, as a numbered chain of reasoning steps followed by a conclusion.

CLAIM:
"[[claim]]"

DOMAIN: [[routing_domain]]
ANCHOR NORMS:
[[anchor_text]]
PRINCIPLE TESTS:
[[principle_text]]

KNOWLEDGE BASE (use only these sources):
[[knowledge_base]]

ALLOWED STATUTES (cite ONLY these):
[[statutes_list]]
ALLOWED PRECEDENTS:
[[precedents_list]]

INSTRUCTIONS:
- Produce between [[min_steps]] and [[max_steps]] reasoning steps that build the thesis progressively.
- Each step must cite at least one relevant article from the ALLOWED STATUTES using the form "Art. NNN c.c." or "Art. NNN c.p.".
- Do NOT invent facts or norms outside the allowed sources.
- Output the steps in EXACTLY this format, one marker per step, and nothing else before the first step:
STEP 1: <legal reasoning citing the relevant article>
STEP 2: <legal reasoning citing the relevant article>
...
STEP N: <legal reasoning citing the relevant article>
CONCLUSION: <2-4 sentences synthesizing the legal assessment and why>
[[language_instruction]]""",
    # ---------------------------------------------------------------------
    # Counter-Reasoner
    # ---------------------------------------------------------------------
    PromptKey.COUNTER_REASONER_PICK_ATTACKS: """Claim:
"[[claim]]"

Routing context:
- causal_type_id: [[causal_type_id]]
- theory_id: [[theory_id]]

Select the most useful attack IDs among the following ids.
Return ONLY JSON in this format:
{"attack_ids": ["id1", "id2", "id3"]}

Rules:
- choose between [[min_attacks]] and [[max_attacks]] ids
- ids must be from the list below
- avoid near-duplicate attacks

[[options_text]]
""",
    PromptKey.COUNTER_REASONER_OPEN_ATTACKS: """You are a legal counter-argument strategist.

Generate a compact set of NON-TAXONOMIC counter-attacks that can challenge the Reasoner conclusion
using ONLY the facts explicitly present in the claim.

CLAIM:
"[[claim]]"

REASONER CONCLUSION TO OPPOSE:
"[[reasoner_conclusion]]"

Rules:
- Propose between [[min_attacks]] and [[max_attacks]] attacks.
- Do NOT invent new facts and do NOT contradict explicit claim facts.
- Attacks must be materially distinct (no paraphrase duplicates).
- Favor legally meaningful lines (evidence weight, legal qualification boundaries, subjective element, balancing/aggravanti-attenuanti, burden/sufficiency of proof).
- Keep each description short (max 22 words).

[[language_instruction]]
Return ONLY JSON in this exact format:
{
  "attacks": [
    {"id": "open_attack_1", "description": "short description"},
    {"id": "open_attack_2", "description": "short description"}
  ]
}
""",
    PromptKey.COUNTER_REASONER_TARGET_MAP: """You are extracting the valid legal attack surface for a Counter-Reasoner.

CLAIM:
[[claim]]

REASONER CONCLUSION:
[[reasoner_conclusion]]

Return ONLY JSON:
{
  "allowed_targets": [
    "short legal point that can be contested/limited without inventing facts"
  ],
  "forbidden_assumptions": [
    "short statement of factual assumptions that are not explicit in the claim"
  ],
  "priority_targets": [
    "highest-value legal target 1",
    "highest-value legal target 2"
  ]
}

[[language_instruction]]
Rules:
- allowed_targets must be legal-inferential targets (qualification, proof threshold, cumulo limits, proportionality, quantification, aggravants/attenuants, etc.).
- forbidden_assumptions must include hypothetical factual completions not in claim.
- Do not output prose or markdown.
""",
    PromptKey.COUNTER_REASONER_SINGLE_CALL: """You are an expert Italian jurist acting as OPPOSING counsel. Construct the COMPLETE counter-argument (antithesis) against the reasoner's thesis below in a SINGLE response, as a numbered chain of counter-steps followed by a conclusion.

CLAIM:
"[[claim]]"

REASONER CONCLUSION TO OPPOSE:
"[[reasoner_conclusion]]"

DOMAIN: [[routing_domain]]
KNOWLEDGE BASE (use only these sources):
[[knowledge_base]]
ALLOWED STATUTES (cite ONLY these):
[[statutes_list]]
ALLOWED PRECEDENTS:
[[precedents_list]]

INSTRUCTIONS:
- Produce between [[min_steps]] and [[max_steps]] counter-steps that attack the thesis by legal reinterpretation.
- Each step must cite at least one relevant article from the ALLOWED STATUTES using the form "Art. NNN c.c." or "Art. NNN c.p.".
- Attack the thesis WITHOUT inventing new facts and WITHOUT contradicting the explicit facts of the claim: challenge it through legal reinterpretation, exceptions, derogations, or limitations of the remedy.
- If the thesis genuinely cannot be opposed on legal grounds, write "STEP 1: NO_VALID_OPPOSITION" instead of forcing a weak objection.
- Output in EXACTLY this format, one marker per step, and nothing else before the first step:
STEP 1: <counter-reasoning citing the relevant article>
STEP 2: <counter-reasoning citing the relevant article>
...
STEP N: <counter-reasoning citing the relevant article>
CONCLUSION: <2-4 sentences stating why the thesis should be rejected or materially limited>
[[language_instruction]]""",
    PromptKey.COUNTER_REASONER_DECOMPOSE_CONCLUSION: """You are a legal decomposition engine for counter-argument planning.

CLAIM:
[[claim]]

REASONER CONCLUSION:
[[reasoner_conclusion]]

Task:
- Decompose ONLY the reasoner conclusion into atomic legal commitments.
- Identify which commitments are attackable without inventing new facts.
- Keep claim facts fixed; do not introduce new events.

Return ONLY JSON:
{
  "attack_points": [
    {
      "id": "P1",
      "statement": "atomic legal commitment from conclusion",
      "point_type": "norm_application | causal_link | burden_of_proof | quantification | remedy_scope | interpretation | other",
      "attack_vector": "short hint on how this point can be challenged"
    }
  ],
  "fixed_commitments": [
    "commitment that should be preserved as factual/legal baseline"
  ]
}

[[language_instruction]]
Rules:
- Use at most 8 attack_points.
- statement must be concise (max 30 words) and traceable to the conclusion text.
- fixed_commitments should include only non-controversial baseline commitments.
- If no clear decomposition is possible, return empty arrays.
""",
    PromptKey.COUNTER_REASONER_GENERATE_PLAN: """You are a legal planning engine for Italian counter-argumentation.

Create a sequential plan to build a counter-argument against the primary legal thesis and the Reasoner's conclusion.
Each step must add a materially new counter-objective.

Return only ONE valid JSON object (no markdown, no prose, no text outside JSON)
using this schema:
{
  "steps": [
    {
      "id": "C1",
      "goal": "<specific objective to weaken the primary thesis>",
      "focus": "<single weak point for this step>",
      "expected_norm": "N/A or article expected to be cited",
      "citation_requirement": "required | optional | none",
      "attack_id": "one of the selected attack ids",
      "step_type": "TARGET_FACTS | TARGET_CAUSAL_LINK | TARGET_LEGAL_QUALIFICATION | TARGET_ELEMENT | TARGET_BALANCING | TARGET_OUTCOME | OTHER",
      "novelty_key": "<unique snake_case key>"
    }
  ]
}

CLAIM:
"[[claim]]"
[[reasoner_block]]
CLAIM FACT ANCHORS (use only these factual premises):
[[claim_facts]]
TARGET MAP (counter scope):
[[target_map]]
SUGGESTED ATTACK POINTS (optional hints):
[[conclusion_points]]
DOMAIN: [[routing_domain]]
SELECTED ATTACK IDS:
[[selected_attack_ids]]

ATTACK CATALOG:
[[attack_catalog]]

ALLOWED STATUTES:
[[statutes_list]]
ALLOWED PRECEDENTS:
[[precedents_list]]
KNOWLEDGE BASE:
[[knowledge_base]]

PLANNER MODE: [[planner_mode]]
RESUME FROM LOGICAL STEP: [[resume_from_step]]
ALREADY ACCEPTED COUNTER STEPS (do not duplicate):
[[existing_steps]]

RULES:
- Number of steps must be between [[min_steps]] and [[max_steps]].
- Each step must be materially different (no overlap/rephrasing).
- Steps must function as counter-argument steps (they must weaken the primary thesis / Reasoner conclusion).
- Steps must stay within TARGET MAP allowed_targets.
- Steps must avoid TARGET MAP forbidden_assumptions.
- Treat Reasoner conclusion as a thesis-to-attack, NOT as a source of additional facts.
- Never use factual details from Reasoner conclusion unless they are also present in CLAIM FACT ANCHORS.
- Use only facts explicitly present in claim (no assumptions, no hypothetical factual completions).
- Never introduce hypothetical external events not in claim (e.g., mechanical failure, weather conditions, third-party interventions) as if they were case facts.
- For causal-alternative attacks, contest certainty/weight/completeness of the existing evidence or legal imputability; do NOT fabricate alternative factual scenarios.
- If selected attack ids are available, each step should include one attack_id from that set.
- If selected attack ids are empty, set attack_id as empty string and build attack-agnostic steps.
- Suggested attack points are OPTIONAL guidance, not hard constraints.
- You may ignore any suggested point if it risks contradiction with fixed claim facts.
- Set "citation_requirement" using the same policy:
  * "required" for norm-based attacks / legal qualification attacks
  * "optional" for inferential bridge or factual elaboration steps
  * "none" only for pure synthesis/transition steps
- If "expected_norm" is not "N/A", "citation_requirement" should normally be "required".
- "step_type" must be coherent with the target being attacked and should not repeat unless strictly necessary.
- "novelty_key" must be unique across steps and must summarize what is NEW in the counter-attack.
- If ALREADY ACCEPTED COUNTER STEPS is not empty, generate only missing/remaining steps and avoid duplicate objectives.
- Keep each 'goal' and 'focus' concise (max 25 words each).
[[language_instruction]]

Return only the JSON object.
""",
    PromptKey.COUNTER_REASONER_PLAN_TARGET_ALIGNMENT: """You are checking whether a planned counter step is in-scope.

CLAIM:
[[claim]]

REASONER CONCLUSION:
[[reasoner_conclusion]]

TARGET MAP:
[[target_map]]

PLANNED GOAL:
[[plan_goal]]

PLANNED FOCUS:
[[plan_focus]]

Task:
- ALIGNED: step fits allowed_targets and does not imply forbidden_assumptions.
- OFF_TARGET: step is outside allowed_targets or implies forbidden_assumptions.
- UNCLEAR: not enough signal.

Answer with EXACTLY one word: ALIGNED or OFF_TARGET or UNCLEAR.
""",
    PromptKey.COUNTER_REASONER_STEP_PROMPT: """You are an expert Italian jurist.
You must execute ONLY one planned COUNTER step.

CLAIM:
"[[claim]]"
[[reasoner_block]]
CLAIM FACT ANCHORS (fixed facts, do not alter):
[[claim_facts]]
DOMAIN: [[routing_domain]]
ATTACK STRATEGY: [[attack_id]] - [[attack_desc]]
AVAILABLE ATTACKS FOR THIS STEP:
[[attack_pool_lines]]

KNOWLEDGE BASE (use only these sources):
[[knowledge_base]]

ALLOWED STATUTES:
[[statutes_list]]
ALLOWED PRECEDENTS:
[[precedents_list]]

GLOBAL PLAN:
[[plan_lines]]

CURRENT STEP TO EXECUTE: [[plan_index]]
- Goal: [[plan_goal]]
- Focus: [[plan_focus]]
- Suggested attack points (optional):
[[suggested_points_text]]
- Expected norm: [[plan_expected_norm]]
- Citation requirement: [[plan_citation_requirement]]
- Preferred attack id for this step: [[plan_attack_id]]
- Step type: [[plan_step_type]]
- Novelty key: [[plan_novelty_key]]

ALREADY GENERATED STEP SUMMARIES:
[[summary_lines]]

NORMS ALREADY USED: [[used_norms_text]]

HARD RULES:
- Generate EXACTLY ONE atomic step in [[response_language_name]] (2-4 sentences).
- It must advance the plan and add NEW information, not paraphrase prior steps.
- It must realize the declared novelty key by adding a distinct counter-argument point.
- It must function as a counter-step (weakening or challenging the primary thesis / Reasoner conclusion).
- Treat Reasoner conclusion as a thesis-to-attack, NOT as a source of additional facts.
- Never import factual details from Reasoner conclusion unless they are explicitly present in CLAIM FACT ANCHORS.
- Never invent facts outside claim.
- Never assume or complete missing factual details beyond what is explicitly stated in the claim.
- Never introduce hypothetical external events (e.g., mechanical failures, weather events, unknown third-party causes) unless explicitly stated in the claim.
- If you need to counter causal certainty, argue about uncertainty or insufficiency of existing evidence/perizia, not by adding new events.
- Treat explicit claim facts (including explicit factual findings/perizie stated in the claim) as fixed and true; do not deny or reverse them.
- Never contradict previous accepted steps.
- If citation requirement is "required", cite at least one statute.
- If citation requirement is "optional", citation is recommended but not mandatory.
- If citation requirement is "none", do not force a citation.
[[attack_usage_rules]]

[[language_instruction]]

RESPONSE FORMAT:
[[attacks_used_format]]
STEP: [atomic counter-step]
""",
    PromptKey.COUNTER_REASONER_ATTACK_ALIGNMENT: """Assess whether the following counter-argument step is ALIGNED with the planned attack.

CLAIM:
[[claim]]

ATTACK ID:
[[attack_id]]

ATTACK DESCRIPTION:
[[attack_desc]]

PLAN FOCUS:
[[plan_focus]]

STEP CANDIDATE:
[[candidate_step]]

Rules:
- Answer ALIGNED only if the step clearly applies the specified attack and the plan focus.
- Answer MISALIGNED if the step is generic, off-focus, or does not really implement the attack.
- The step must remain a genuine counter-step for the targeted thesis; if it is generic, merely descriptive, or agrees with the targeted thesis, it is MISALIGNED.

Answer with EXACTLY one word: ALIGNED or MISALIGNED.
""",
    PromptKey.COUNTER_REASONER_ATTACK_SAFETY: """You are a legal safety rewriter for counter-attack strategies.

CLAIM (explicit facts are fixed):
[[claim]]

REASONER CONCLUSION TO OPPOSE:
[[reasoner_conclusion]]

CLAIM FACT ANCHORS:
[[claim_facts]]

ATTACK CANDIDATES:
[[attack_catalog]]

Task:
For each candidate attack, classify whether it can be used without contradicting explicit claim facts.

Labels:
- SAFE: can be used directly.
- LIMITED: must be reformulated as limitation/containment of effects (not direct denial of fixed facts).
- UNSAFE: cannot be made compatible without factual contradiction/invention.

[[language_instruction]]
Return ONLY JSON:
{
  "attacks": [
    {
      "id": "attack_id",
      "status": "SAFE | LIMITED | UNSAFE",
      "description": "short operational description (max 22 words, mandatory for SAFE/LIMITED)"
    }
  ]
}

Rules:
- Preserve the attack id exactly.
- If claim contains an explicit fixed fact, do not propose descriptions that deny it.
- Prefer LIMITED over UNSAFE when the attack can be reframed as legal limitation (scope, burden, quantification, effects).
- Do not add new facts or hypothetical events.
""",
    PromptKey.COUNTER_REASONER_ATTACK_COMPATIBILITY: """You are evaluating semantic compatibility between a legal claim and a counter-attack strategy.

CLAIM:
[[claim]]

ATTACK ID:
[[attack_id]]

ATTACK DESCRIPTION:
[[attack_desc]]

Task:
- COMPATIBLE: attack naturally fits the legal institutes/factual frame of the claim.
- WEAK: attack is only partially suitable; it might work but with limited strength.
- MISMATCH: attack targets institutes/facts that are not present in the claim context.

Rules:
- Focus on legal-semantic fit, not lexical overlap.
- If uncertain, prefer WEAK.

Answer with EXACTLY one word: COMPATIBLE or WEAK or MISMATCH.
""",
    PromptKey.COUNTER_REASONER_ATTACK_PRECONDITION_CHECK: """You are checking whether an attack precondition is satisfied.

CLAIM:
[[claim]]

REASONER CONCLUSION:
[[reasoner_conclusion]]

PRECONDITION:
[[precondition]]

Task:
- SATISFIED: precondition is clearly supported by the case context.
- UNSATISFIED: precondition is clearly not supported.
- UNCLEAR: not enough signal.

Answer with EXACTLY one word: SATISFIED or UNSATISFIED or UNCLEAR.
""",
    PromptKey.COUNTER_REASONER_STEP_OPPOSITION_CHECK: """You are checking whether a counter-argument step actually opposes the primary legal thesis.

CLAIM:
\"\"\"[[claim]]\"\"\"

REASONER CONCLUSION TO OPPOSE:
\"\"\"[[reasoner_conclusion]]\"\"\"

CANDIDATE COUNTER STEP:
\"\"\"[[candidate_step]]\"\"\"

Task:
- Answer AGREEING if the candidate step supports, confirms, or materially reinforces the Reasoner conclusion.
- Answer OPPOSING if the candidate step weakens, disputes, or limits the Reasoner conclusion.
- Answer UNCLEAR if the relation is not clear from the text.

Important:
- Use the claim only as factual context.
- Do not require the step to restate the final opposite outcome explicitly; attacking a key premise of the conclusion still counts as OPPOSING.

Answer with EXACTLY one word: OPPOSING or AGREEING or UNCLEAR.
""",
    PromptKey.COUNTER_REASONER_STEP_EXPANSION: """You are expanding one compressed legal counter-step into optional satellite steps.

CLAIM:
"[[claim]]"
[[reasoner_block]]
CLAIM FACT ANCHORS (fixed facts):
[[claim_facts]]

PARENT COUNTER STEP (already accepted):
"[[parent_step]]"

PARENT ATTACK IDS:
[[parent_attack_ids]]

PARENT ATTACK CATALOG:
[[parent_attack_catalog]]

ALREADY GENERATED STEP SUMMARIES:
[[summary_lines]]

NORMS ALREADY USED:
[[used_norms_text]]

ALLOWED STATUTES:
[[statutes_list]]
ALLOWED PRECEDENTS:
[[precedents_list]]
KNOWLEDGE BASE:
[[knowledge_base]]

Task:
- Evaluate whether the parent step can be safely elaborated into additional steps.
- Return 0..[[max_extra]] satellite steps.
- Each satellite must add NEW argumentative value, staying consistent with claim facts.
- Prefer one primary attack per satellite step when possible.
- If expansion is not useful or unsafe, return an empty array.

Hard rules:
- Do not change or replace the parent step.
- Do not invent facts.
- Do not contradict explicit claim facts.
- Do not contradict already generated steps.
- Keep each satellite concise (2-4 sentences) in [[response_language_name]].
- Use attack_id only from PARENT ATTACK IDS.
- If citation_requirement is "required", include at least one grounded statute citation.
- If no strong legal citation is feasible, use expected_norm="N/A" and citation_requirement="optional".

Return ONLY compact JSON:
{
  "extra_steps": [
    {
      "step": "counter-step text",
      "attack_id": "one parent attack id",
      "expected_norm": "article or N/A",
      "citation_requirement": "required | optional | none"
    }
  ]
}
""",
    PromptKey.COUNTER_REASONER_STANCE_REWRITE: """[[original_prompt]]

YOUR PREVIOUS STEP WAS INVALID.
REASON: [[invalid_reason]]
INVALID STEP:
"[[invalid_step]]"

Rewrite the SAME legal point as a coherent counter-step.
Do not add new facts. Do not make it agree with the targeted thesis / Reasoner conclusion.
The rewritten step must clearly weaken or limit the opposing thesis.

RESPONSE FORMAT:
[[attacks_used_format]]
STEP: [text, max 4 sentences]
[[language_instruction]]""",
    PromptKey.COUNTER_REASONER_FACT_LOCK_CHECK: """You are a strict factual contradiction checker for COUNTER legal steps.

CLAIM (explicit facts only):
\"\"\"[[claim]]\"\"\"

CANDIDATE COUNTER STEP:
\"\"\"[[candidate_step]]\"\"\"

Task:
Decide whether the candidate step DIRECTLY contradicts an explicit claim fact.

Label as DIRECT_CONTRADICTION ONLY when at least one explicit claim fact is negated/reversed.

Do NOT label contradiction when the step:
- questions legal qualification, weight, sufficiency, or evidentiary certainty;
- limits legal consequences while keeping facts fixed;
- argues uncertainty without denying an explicit fact.

If doubtful, choose CONSISTENT_OR_INTERPRETIVE.

Answer with EXACTLY one word:
- DIRECT_CONTRADICTION
- CONSISTENT_OR_INTERPRETIVE
""",
    PromptKey.COUNTER_REASONER_NO_NEW_FACTS: """You are a factual grounding checker for legal counter-argument steps.

CLAIM (all explicit facts are fixed and exhaustive):
\"\"\"[[claim]]\"\"\"

CANDIDATE STEP:
\"\"\"[[candidate_step]]\"\"\"

Task:
Decide whether the candidate step introduces NEW FACTUAL ALLEGATIONS that are not explicitly stated in the claim.

Important:
- Legal interpretations, normative qualifications, and evidentiary criticism are allowed.
- It is NOT allowed to add hypothetical events/causes/conditions as case facts.
- If the step only derives legal consequences from existing facts (without adding new events), classify it as LEGAL_INFERENCE.
- If a factual element is merely possible but not in the claim, this counts as ADDS_FACTS.
- If doubtful, prefer GROUNDED unless there is a clear added factual allegation.

Answer with EXACTLY one word: GROUNDED or LEGAL_INFERENCE or ADDS_FACTS.
""",
    PromptKey.COUNTER_REASONER_PLAN_FEASIBILITY_REWRITE: """You are rewriting a planned counter step to make it fact-safe.

CLAIM:
\"\"\"[[claim]]\"\"\"

REASONER CONCLUSION TO OPPOSE:
\"\"\"[[reasoner_conclusion]]\"\"\"

TARGET MAP:
[[target_map]]

ORIGINAL PLAN STEP:
- Goal: [[plan_goal]]
- Focus: [[plan_focus]]
- Expected norm: [[expected_norm]]
- Citation requirement: [[citation_requirement]]

INVALID REASON:
[[invalid_reason]]

Task:
- Rewrite ONLY goal/focus to stay counter-oppositional while preserving explicit claim facts.
- Do NOT add hypothetical events or unstated facts.
- Keep the same argumentative direction when possible.
- Keep expected_norm and citation_requirement coherent.
- If no citation is feasible, set expected_norm to \"N/A\" and citation_requirement to \"optional\".

Return ONLY compact JSON:
{
  "goal": "max 25 words",
  "focus": "max 25 words",
  "expected_norm": "article or N/A",
  "citation_requirement": "required | optional | none"
}
""",
    # ---------------------------------------------------------------------
    # AQA / NLP / Polisher
    # ---------------------------------------------------------------------
    PromptKey.AQA_ENGINE_ATTACK_TYPE_SYSTEM: """You are an expert in Italian law and ASPIC+ argumentation theory.
Given two reasoning steps from opposing legal arguments, classify the TYPE of attack the attacker performs on the target.

Choose EXACTLY ONE of these categories:
- CONTRADICTION: the attacker directly negates the same legal conclusion or factual premise (e.g. the attacker claims the opposite outcome on the same legal question).
- EXCEPTION: the attacker invokes a condition, proviso, or exception that blocks the target norm from applying (e.g. legitimate defence as an exception to criminal liability).
- DEROGATION: the attacker invokes a more specific norm (lex specialis) that overrides or displaces the target norm (lex generalis).
- EXTINCTION: the attacker claims the right/liability has been extinguished (e.g. prescription, forfeiture, settlement, pardon, statute of limitations).
- FACTUAL_IMPEDIMENT: the attacker raises a factual circumstance (without a normative basis) that impedes the target conclusion (e.g. alibi, absence of evidence, factual impossibility).
- GENERAL_OPPOSITION: none of the above; a generic rebuttal or weakening that does not fit any specific category.

Respond with EXACTLY ONE WORD: the category name in upper case.
No punctuation, no explanation, no extra text.""",
    PromptKey.AQA_ENGINE_ATTACK_TYPE_USER: """TARGET REASONING STEP:
"[[target_text]]"

ATTACKER REASONING STEP:
"[[attacker_text]]"

Attack type?""",
    PromptKey.NLP_UTILS_NLI_SYSTEM: """You are an expert in Italian law.
You are comparing two reasoning passages from opposing sides of a legal debate.
Even if they cite the same legal norms, focus on whether their CONCLUSIONS and APPLICATIONS of those norms are incompatible.

Choose EXACTLY ONE of these labels:
- CONTRADICTION: the two passages reach opposite conclusions on the same legal question, or one undermines a premise that the other relies on.
- ENTAILMENT: the two passages support each other and reach compatible conclusions.
- NEUTRAL: the passages address different legal aspects or their relationship is unclear.

Base your judgement solely on the semantic content of the two passages.

Respond with EXACTLY ONE WORD in upper case.
No punctuation, no explanation.""",
    PromptKey.NLP_UTILS_NLI_USER: """PASSAGE A (target):
"[[target_text]]"

PASSAGE B (attacker):
"[[attacker_text]]"

Relationship?""",
    PromptKey.POLISHER_COUNTER_GATE: """You are a legal dialectical verifier.
Compare two reasoning chains on the same claim and determine whether the COUNTER chain is materially oppositional to the REASONER chain.

CLAIM:
\"\"\"[[claim]]\"\"\"

REASONER CHAIN:
[[reasoner_chain]]

COUNTER CHAIN:
[[counter_chain]]

Respond with EXACTLY ONE label:
- OPPOSING_STRONG (the COUNTER chain reaches a clearly opposite outcome)
- OPPOSING_LIMITATIVE (the COUNTER chain does not fully negate liability but materially LIMITS the REASONER outcome, e.g. excludes aggravants, lowers blameworthiness, reduces sanction range, contests certainty threshold)
- AGREEING (the COUNTER chain supports or converges with the REASONER outcome without material limitation)
- UNCLEAR (the relation is not clear)
""",
    # ---------------------------------------------------------------------
    # Consistency Checker (citation verification & repair)
    # ---------------------------------------------------------------------
    PromptKey.CONSISTENCY_VERIFY_MISMATCH_SYSTEM: """You are an expert in Italian law. Your task is to determine whether two normative texts are LOGICALLY EQUIVALENT or DIFFERENT.

Two texts are EQUIVALENT if:
- They express the same legal concept, even with different wording
- One is a faithful paraphrase of the other
- They don't add or omit substantial normative elements

Two texts are DIFFERENT if:
- They add requirements not present in the original
- They omit essential elements
- They change the legal meaning
- They introduce concepts not contained in the original

Respond ONLY with one of these words: EQUIVALENTI or DIVERSI (in Italian)""",
    PromptKey.CONSISTENCY_VERIFY_MISMATCH_USER: """Article [[article_num]]

CITED TEXT (from reasoning):
"[[cited_text]]"

OFFICIAL TEXT (from database):
"[[db_text]]"

Are the two texts EQUIVALENTI or DIVERSI? (Answer in Italian)""",
    PromptKey.CONSISTENCY_PERTINENCE_SYSTEM: """You are an expert in Italian law. Your task is to assess whether a legal norm cited in a reasoning chain is PERTINENT or NOT_PERTINENT to the argumentative logic.

A norm is PERTINENT if:
- It contributes meaningfully to the legal reasoning (e.g., establishes liability, defines scope, sets conditions)
- It reinforces or integrates the legal conclusion
- It provides necessary normative context for the argument's thesis
- It completes the regulatory framework by filling a logical gap
- It is causally or logically connected to other steps in the chain

A norm is NOT_PERTINENT if:
- It is tangential to the main line of reasoning
- It is redundant with respect to other norms already cited that cover the same concept
- It adds no argumentative value to the reasoning chain
- It has no logical or causal link to the conclusion

Respond ONLY with one of these words: PERTINENT or NOT_PERTINENT""",
    PromptKey.CONSISTENCY_PERTINENCE_USER: """COMPLETE REASONING CHAIN:
\"\"\"
[[full_text]]
\"\"\"

NORM TO EVALUATE: Art. [[article_num]]
CITED TEXT: "[[cited_text]]"
OFFICIAL NORM TITLE (DB): "[[db_title]]"
OFFICIAL NORM TEXT (DB EXCERPT): "[[db_text]]"

Is this norm PERTINENT or NOT_PERTINENT to the argumentative logic of the reasoning chain?""",
    PromptKey.CONSISTENCY_REPAIR_DB_SYSTEM: """You are an expert in Italian law. You must rewrite a normative passage using EXCLUSIVELY the official text of the provided article.

MANDATORY RULES:
1. You must include a VERBATIM QUOTE (exact copy) of at least 15 consecutive words from the official text
2. The quote must be enclosed in «» (guillemets/angle quotes)
3. You cannot add concepts not present in the official text
4. The result must be legally correct and coherent

OUTPUT: Write ONLY the rewritten text in Italian, without explanations.""",
    PromptKey.CONSISTENCY_REPAIR_DB_USER: """ARTICLE: Art. [[article_num]]

OFFICIAL TEXT TO USE:
"[[db_text]]"

ORIGINAL CONTEXT (to correct):
"[[original_context]]"

Rewrite the normative passage in Italian using only the official text, including a verbatim quote enclosed in «».""",
    PromptKey.CONSISTENCY_PRECEDENT_MISMATCH_SYSTEM: """You are an expert in Italian law. Determine whether two descriptions of a court precedent are LOGICALLY EQUIVALENT or DIFFERENT.

EQUIVALENT: same legal holding, even with different wording.
DIFFERENT: different holding, added/omitted elements, changed meaning.

Respond ONLY with: EQUIVALENTI or DIVERSI""",
    PromptKey.CONSISTENCY_PRECEDENT_MISMATCH_USER: """CITED TEXT (from reasoning):
"[[cited_text]]"

OFFICIAL SUMMARY (from database):
"[[db_summary]]"

Are the two texts EQUIVALENTI or DIVERSI?""",
    PromptKey.CONSISTENCY_PRECEDENT_REPAIR_SYSTEM: """You are an expert in Italian law. Rewrite a passage that cites a court precedent using EXCLUSIVELY the official summary provided.

RULES:
1. Include a VERBATIM QUOTE of at least 15 words from the official summary enclosed in «»
2. Do not add concepts not in the official summary
3. Write in Italian
4. Output ONLY the rewritten text""",
    PromptKey.CONSISTENCY_PRECEDENT_REPAIR_USER: """PRECEDENT: [[precedent_title]]

OFFICIAL SUMMARY:
"[[db_summary]]"

ORIGINAL CONTEXT (to correct):
"[[cited_text]]"

Rewrite using only the official summary.""",
}


_PLACEHOLDER_RE = re.compile(r"\[\[([a-zA-Z_][a-zA-Z0-9_]*)\]\]")
_PROMPT_KEY_BY_VALUE = {key.value: key for key in PromptKey}
_PROMPT_KEY_BY_NAME = {key.name: key for key in PromptKey}


def _coerce_prompt_key(name: str | PromptKey) -> PromptKey:
    """Normalize prompt key input to PromptKey."""
    if isinstance(name, PromptKey):
        return name

    normalized = str(name or "").strip()
    by_value = _PROMPT_KEY_BY_VALUE.get(normalized)
    if by_value is not None:
        return by_value

    by_name = _PROMPT_KEY_BY_NAME.get(normalized)
    if by_name is not None:
        return by_name

    raise KeyError(f"Prompt not found: {name}")


def get_prompt(name: str | PromptKey) -> str:
    """Return prompt template by registry key."""
    key = _coerce_prompt_key(name)
    if key not in PROMPTS:
        raise KeyError(f"Prompt not found: {name}")
    return PROMPTS[key]


def _get_language_defaults() -> dict[str, str]:
    """Return language-related placeholder defaults derived from settings.

    Resolution order: per-request thread-local (set via set_response_language)
    → settings.response_language → "it".
    """
    lang = getattr(_lang_local, "value", None)
    if lang is None:
        try:
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from config import settings

            lang = (settings.response_language or "it").lower()
        except Exception:
            lang = "it"

    if lang == "en":
        return {
            "language_instruction": (
                "LANGUAGE INSTRUCTION: Reason and respond exclusively in English. "
                "Write all legal analysis, planning steps, argumentation, and conclusions in English. "
                "Legal sources (statutes, precedents) may be quoted verbatim in their original language "
                "inside «» marks, but all surrounding reasoning must be in English."
            ),
            "response_language_name": "English",
        }
    return {
        "language_instruction": (
            "LANGUAGE INSTRUCTION: Reason and respond exclusively in Italian. "
            "Write all legal analysis, planning steps, argumentation, and conclusions in Italian."
        ),
        "response_language_name": "Italian",
    }


def render_prompt(name: str | PromptKey, **values: object) -> str:
    """
    Render prompt replacing placeholders ``[[name]]`` with provided values.

    Missing placeholders are left unchanged.
    Language placeholders (language_instruction, response_language_name) are
    auto-injected from settings.response_language if not explicitly provided.
    """
    template = get_prompt(name)

    lang_defaults = _get_language_defaults()
    merged = {**lang_defaults, **{k: v for k, v in values.items() if v is not None}}

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = merged.get(key)
        return str(value) if value is not None else match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, template)
