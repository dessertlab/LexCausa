"""
LexCausa Configuration Module.

Centralized configuration management using Pydantic Settings.
Loads environment variables from .env file and provides typed access.

Usage:
    from config import settings

    # Access configuration
    neo4j_uri = settings.neo4j_uri
    groq_api_key = settings.groq_api_key
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Find project root and load .env
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
_env_file = _project_root / ".env"

# Load .env file
load_dotenv(_env_file)

# Hugging Face loading is ONLINE by default: models are downloaded if missing and
# served from the local cache otherwise (standard HF behavior). For an air-gapped
# run (e.g. an HPC compute node with no internet), export HF_HUB_OFFLINE=1 to force
# cache-only loading and never reach huggingface.co at runtime.
#
# HF_OFFLINE is the single source of truth for `local_files_only` at every
# `from_pretrained(...)` call.
HF_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "",
)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables.
    The .env file in project root is automatically loaded.
    """

    model_config = SettingsConfigDict(
        env_file=str(_env_file),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Neo4j Configuration
    # =========================================================================
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="neo4jpassword", alias="NEO4J_PASSWORD")

    # =========================================================================
    # Groq Cloud Configuration
    # =========================================================================
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY_V1")
    _groq_api_keys: list[str] = []  # populated dynamically by validator
    # Model catalog is code-owned (not env-driven).
    MODEL_ALIAS_MAP: ClassVar[dict[str, str]] = {
        "groq_llama_scout_17b": "meta-llama/llama-4-scout-17b-16e-instruct",
        "gpt_oss_120b": "openai/gpt-oss-120b",
        "gpt_oss_20b": "openai/gpt-oss-20b",
        "groq_llama_3_3_70b_versatile": "llama-3.3-70b-versatile",
        "qwen_qwen3_32b": "qwen/qwen3-32b",
        # HPC cluster / vLLM models
        "deepseek_r1": "deepseek-ai/DeepSeek-R1",
        "qwen_25_72b": "Qwen/Qwen2.5-72B-Instruct",
    }
    # Used by retrieval-side components (claim classifier, keyword extraction,
    # legal-context extraction, relevance/applicability filters, precedents keywording).
    RETRIEVAL_MODEL_ORDER_ALIASES: ClassVar[list[str]] = [
        "groq_llama_scout_17b",
        "qwen_qwen3_32b",
        "groq_llama_3_3_70b_versatile",
        "gpt_oss_20b",
    ]
    # Used by the rest of the pipeline helpers (router, evaluator/polisher).
    PIPELINE_MODEL_ORDER_ALIASES: ClassVar[list[str]] = [
        "groq_llama_scout_17b",
        "qwen_qwen3_32b",
        "groq_llama_3_3_70b_versatile",
        "gpt_oss_20b",
    ]
    REASONER_DEFAULT_MODEL_ALIAS: ClassVar[str] = "gpt_oss_120b"
    COUNTER_DEFAULT_MODEL_ALIAS: ClassVar[str] = "gpt_oss_120b"
    reasoner_model_fallback_aliases: list[str] = Field(
        default_factory=lambda: [
            "gpt_oss_120b",
            "qwen_qwen3_32b",
            "groq_llama_3_3_70b_versatile",
        ],
        alias="REASONER_MODEL_FALLBACK_ALIASES",
        description="Ordered model aliases used as resilient fallback chain for Reasoner.",
    )
    counter_model_fallback_aliases: list[str] = Field(
        default_factory=lambda: [
            "gpt_oss_120b",
            "qwen_qwen3_32b",
            "groq_llama_3_3_70b_versatile",
        ],
        alias="COUNTER_MODEL_FALLBACK_ALIASES",
        description="Ordered model aliases used as resilient fallback chain for Counter-Reasoner.",
    )

    # =========================================================================
    # Retry / Resilience Configuration
    # =========================================================================
    groq_max_retries: int = Field(
        default=3,
        alias="GROQ_MAX_RETRIES",
        description="Maximum number of retries per API call (key rotation + model fallback).",
    )
    groq_retry_base_delay: float = Field(
        default=1.0,
        alias="GROQ_RETRY_BASE_DELAY",
        description="Base delay in seconds for exponential backoff between retries.",
    )

    # =========================================================================
    # LLM Configuration
    # =========================================================================
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    reasoner_default_temperature: float = Field(
        default=0.0,
        alias="REASONER_DEFAULT_TEMPERATURE",
        description="Default UI temperature for the Reasoner agent.",
    )
    counter_default_temperature: float = Field(
        default=0.3,
        alias="COUNTER_DEFAULT_TEMPERATURE",
        description="Default UI temperature for the Counter-Reasoner agent.",
    )
    llm_max_tokens: int = Field(default=7168, alias="LLM_MAX_TOKENS")

    # =========================================================================
    # Embedding Model Configuration
    # =========================================================================
    embedding_model: str = Field(
        default="nlpaueb/legal-bert-base-uncased", alias="EMBEDDING_MODEL"
    )
    embedding_max_length: int = Field(default=512, alias="EMBEDDING_MAX_LENGTH")

    # =========================================================================
    # API Server Configuration
    # =========================================================================
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    debug: bool = Field(default=False, alias="DEBUG")

    # =========================================================================
    # Search & Retrieval Defaults
    # =========================================================================
    search_top_k_default: int = Field(
        default=100,
        alias="SEARCH_TOP_K_DEFAULT",
        description="Default number of statute results to return when not specified.",
    )
    search_min_kept_statutes: int = Field(
        default=8,
        alias="SEARCH_MIN_KEPT_STATUTES",
        description="Minimum number of statutes to keep after relevance filtering. "
        "If fewer are kept, the search expands progressively by +10 until this threshold is met.",
    )
    search_expansion_step: int = Field(
        default=10,
        alias="SEARCH_EXPANSION_STEP",
        description="Number of additional statutes to fetch per expansion round.",
    )
    search_max_expansions: int = Field(
        default=5,
        alias="SEARCH_MAX_EXPANSIONS",
        description="Maximum number of progressive expansion rounds to avoid infinite loops.",
    )
    search_expansion_max_zero_gain_rounds: int = Field(
        default=1,
        alias="SEARCH_EXPANSION_MAX_ZERO_GAIN_ROUNDS",
        description="Early-stop retrieval expansion after this many consecutive rounds with zero newly kept statutes.",
    )
    search_use_top_n_libri: int = Field(
        default=3,
        alias="SEARCH_USE_TOP_N_LIBRI",
        description="How many top classified libri to query during statute search.",
    )
    search_hybrid_vector_weight: float = Field(
        default=0.30,
        alias="SEARCH_HYBRID_VECTOR_WEIGHT",
        description="Weight of vector-ranked candidates in hybrid statute retrieval score fusion.",
    )
    search_hybrid_fulltext_weight: float = Field(
        default=0.70,
        alias="SEARCH_HYBRID_FULLTEXT_WEIGHT",
        description="Weight of fulltext-ranked candidates in hybrid statute retrieval score fusion.",
    )
    search_hybrid_admin_vector_weight: float = Field(
        default=0.35,
        alias="SEARCH_HYBRID_ADMIN_VECTOR_WEIGHT",
        description="Vector weight used for administrative-code hybrid retrieval.",
    )
    search_hybrid_admin_fulltext_weight: float = Field(
        default=0.65,
        alias="SEARCH_HYBRID_ADMIN_FULLTEXT_WEIGHT",
        description="Fulltext weight used for administrative-code hybrid retrieval.",
    )
    search_hybrid_civile_vector_weight: float = Field(
        default=0.20,
        alias="SEARCH_HYBRID_CIVILE_VECTOR_WEIGHT",
        description="Vector weight used for civil-code hybrid retrieval.",
    )
    search_hybrid_civile_fulltext_weight: float = Field(
        default=0.80,
        alias="SEARCH_HYBRID_CIVILE_FULLTEXT_WEIGHT",
        description="Fulltext weight used for civil-code hybrid retrieval.",
    )
    search_hybrid_penale_vector_weight: float = Field(
        default=0.45,
        alias="SEARCH_HYBRID_PENALE_VECTOR_WEIGHT",
        description="Vector weight used for criminal-code hybrid retrieval.",
    )
    search_hybrid_penale_fulltext_weight: float = Field(
        default=0.55,
        alias="SEARCH_HYBRID_PENALE_FULLTEXT_WEIGHT",
        description="Fulltext weight used for criminal-code hybrid retrieval.",
    )
    search_hybrid_candidate_multiplier: int = Field(
        default=6,
        alias="SEARCH_HYBRID_CANDIDATE_MULTIPLIER",
        description="Candidate expansion factor: preliminary candidates = top_k * multiplier before fusion.",
    )
    search_hybrid_candidate_min: int = Field(
        default=40,
        alias="SEARCH_HYBRID_CANDIDATE_MIN",
        description="Minimum number of preliminary candidates per source/libro before fusion.",
    )
    search_hybrid_fused_pool_multiplier: int = Field(
        default=2,
        alias="SEARCH_HYBRID_FUSED_POOL_MULTIPLIER",
        description="Multiplier for intermediate fused pool size before final top_k truncation.",
    )
    search_hybrid_filter_priority_decay: float = Field(
        default=0.4,
        alias="SEARCH_HYBRID_FILTER_PRIORITY_DECAY",
        description="Priority decay applied to secondary/tertiary classifier filters in fused ranking.",
    )
    search_hybrid_filter_priority_floor: float = Field(
        default=0.35,
        alias="SEARCH_HYBRID_FILTER_PRIORITY_FLOOR",
        description="Lower bound for classifier-filter priority scaling in fused ranking.",
    )
    search_hybrid_keyword_bonus_max: float = Field(
        default=0.20,
        alias="SEARCH_HYBRID_KEYWORD_BONUS_MAX",
        description="Maximum lexical bonus added to fused score when query terms overlap article title/articolo.",
    )
    search_hybrid_keyword_bonus_scale: float = Field(
        default=0.25,
        alias="SEARCH_HYBRID_KEYWORD_BONUS_SCALE",
        description="Linear scale factor for lexical overlap bonus before clamping to max.",
    )
    search_hybrid_keyword_min_overlap_count: int = Field(
        default=1,
        alias="SEARCH_HYBRID_KEYWORD_MIN_OVERLAP_COUNT",
        description="Minimum number of query-term overlaps required before adding keyword bonus (general).",
    )
    search_hybrid_penale_keyword_min_overlap_count: int = Field(
        default=1,
        alias="SEARCH_HYBRID_PENALE_KEYWORD_MIN_OVERLAP_COUNT",
        description="Minimum number of query-term overlaps required before adding keyword bonus for criminal code.",
    )
    search_hybrid_civile_keyword_min_overlap_count: int = Field(
        default=2,
        alias="SEARCH_HYBRID_CIVILE_KEYWORD_MIN_OVERLAP_COUNT",
        description="Minimum number of query-term overlaps required before adding keyword bonus for civil code.",
    )
    search_hybrid_admin_keyword_min_overlap_count: int = Field(
        default=2,
        alias="SEARCH_HYBRID_ADMIN_KEYWORD_MIN_OVERLAP_COUNT",
        description="Minimum number of query-term overlaps required before adding keyword bonus for administrative code.",
    )
    search_hybrid_penale_zero_overlap_multiplier: float = Field(
        default=0.75,
        alias="SEARCH_HYBRID_PENALE_ZERO_OVERLAP_MULTIPLIER",
        description="Score multiplier for criminal-code results with zero lexical overlap against claim terms.",
    )
    search_hybrid_penale_low_overlap_multiplier: float = Field(
        default=0.88,
        alias="SEARCH_HYBRID_PENALE_LOW_OVERLAP_MULTIPLIER",
        description="Score multiplier for criminal-code results with overlap below bonus threshold.",
    )
    search_hybrid_civile_zero_overlap_multiplier: float = Field(
        default=0.8,
        alias="SEARCH_HYBRID_CIVILE_ZERO_OVERLAP_MULTIPLIER",
        description="Score multiplier for civil-code results with zero lexical overlap against claim terms.",
    )
    search_hybrid_civile_low_overlap_multiplier: float = Field(
        default=0.9,
        alias="SEARCH_HYBRID_CIVILE_LOW_OVERLAP_MULTIPLIER",
        description="Score multiplier for civil-code results with overlap below bonus threshold.",
    )
    search_hybrid_admin_zero_overlap_multiplier: float = Field(
        default=0.85,
        alias="SEARCH_HYBRID_ADMIN_ZERO_OVERLAP_MULTIPLIER",
        description="Score multiplier for administrative-code results with zero lexical overlap against claim terms.",
    )
    search_hybrid_admin_low_overlap_multiplier: float = Field(
        default=0.95,
        alias="SEARCH_HYBRID_ADMIN_LOW_OVERLAP_MULTIPLIER",
        description="Score multiplier for administrative-code results with overlap below bonus threshold.",
    )
    search_hybrid_min_keyword_length: int = Field(
        default=4,
        alias="SEARCH_HYBRID_MIN_KEYWORD_LENGTH",
        description="Minimum token length considered for normalized LLM query keywords used in lexical bonus.",
    )
    search_query_terms_mode: str = Field(
        default="llm",
        alias="SEARCH_QUERY_TERMS_MODE",
        description="Query-term extraction mode for fulltext branch. Only 'llm' is supported.",
    )
    search_query_terms_llm_max_terms: int = Field(
        default=12,
        alias="SEARCH_QUERY_TERMS_LLM_MAX_TERMS",
        description="Maximum number of keywords requested when using LLM query-term extraction.",
    )
    search_query_terms_llm_max_tokens: int = Field(
        default=128,
        alias="SEARCH_QUERY_TERMS_LLM_MAX_TOKENS",
        description="Max completion tokens for LLM-based query-term extraction.",
    )
    search_cites_enabled: bool = Field(
        default=True,
        alias="SEARCH_CITES_ENABLED",
        description="Enable citation-based expansion via Neo4j CITES edges before retrieval filters.",
    )
    search_cites_per_article_limit: int = Field(
        default=6,
        alias="SEARCH_CITES_PER_ARTICLE_LIMIT",
        description="Maximum number of cited neighbors to fetch per initially retrieved statute.",
    )
    search_cites_max_additional: int = Field(
        default=30,
        alias="SEARCH_CITES_MAX_ADDITIONAL",
        description="Maximum number of citation-expanded statutes added to a retrieval round.",
    )
    search_cites_score_decay: float = Field(
        default=0.70,
        alias="SEARCH_CITES_SCORE_DECAY",
        description="Score decay applied to cited statutes relative to their strongest parent score.",
    )
    search_cites_multi_seed_bonus: float = Field(
        default=0.03,
        alias="SEARCH_CITES_MULTI_SEED_BONUS",
        description="Bonus added per additional parent statute citing the same target.",
    )
    search_retrieval_debug_top_n: int = Field(
        default=0,
        alias="SEARCH_RETRIEVAL_DEBUG_TOP_N",
        description="How many top retrieved statutes are printed in retrieval debug logs per stage.",
    )
    precedents_limit_default: int = Field(
        default=5,
        alias="PRECEDENTS_LIMIT_DEFAULT",
        description="Default number of precedents to retrieve when not specified.",
    )

    # =========================================================================
    # AQA / Polisher Evaluator Defaults
    # =========================================================================
    aqa_enabled: bool = Field(
        default=True,
        alias="AQA_ENABLED",
        description="Enable the AQA scoring phase in the Polisher-Evaluator.",
    )
    aqa_alpha: float = Field(
        default=0.3,
        alias="AQA_ALPHA",
        description="AQA weight for Cogency.",
    )
    aqa_beta: float = Field(
        default=0.4,
        alias="AQA_BETA",
        description="AQA weight for NormSupport.",
    )
    aqa_gamma: float = Field(
        default=0.3,
        alias="AQA_GAMMA",
        description="AQA weight for Semantics.",
    )
    aqa_attack_top_k: int = Field(
        default=2,
        alias="AQA_ATTACK_TOP_K",
        description="Top-K cross-attacks to keep per link for explainability.",
    )
    aqa_min_semantic_overlap: float = Field(
        default=0.5,
        alias="AQA_MIN_SEMANTIC_OVERLAP",
        description="Minimum semantic similarity between two links for an attack to be valid. "
        "Attacks below this threshold are filtered out.",
    )
    aqa_min_strength_ratio: float = Field(
        default=1.2,
        alias="AQA_MIN_STRENGTH_RATIO",
        description="Attacker base_score must be >= this ratio × target base_score. "
        "Ensures only meaningfully stronger arguments can inflict damage.",
    )
    aqa_damage_factor: float = Field(
        default=0.4,
        alias="AQA_DAMAGE_FACTOR",
        description="Scaling factor applied to the excess damage (attacker_base - target_base). "
        "Lower values make attacks less destructive.",
    )
    aqa_allow_factual_attacks: bool = Field(
        default=True,
        alias="AQA_ALLOW_FACTUAL_ATTACKS",
        description="Allow factual (non-normative) arguments to attack normative ones.",
    )
    aqa_allow_cross_codice: bool = Field(
        default=True,
        alias="AQA_ALLOW_CROSS_CODICE",
        description="Allow cross-codice attacks (penale vs civile).",
    )
    aqa_procedural_severity_categories: list[str] = Field(
        default_factory=lambda: [
            "prescrizione",
            "decadenza",
            "tutela_diritti",
            "processo",
        ],
        alias="AQA_PROCEDURAL_SEVERITY_CATEGORIES",
        description="Severity categories treated as procedural in ASPIC+ domain-rules checks.",
    )
    aqa_valid_attack_types: list[str] = Field(
        default_factory=lambda: [
            "contradiction",
            "exception",
            "derogation",
            "extinction",
            "factual_impediment",
            "general_opposition",
        ],
        alias="AQA_VALID_ATTACK_TYPES",
        description="Allowed attack-type labels for AQA attack classification.",
    )
    aqa_default_attack_type: str = Field(
        default="general_opposition",
        alias="AQA_DEFAULT_ATTACK_TYPE",
        description="Fallback attack type when classifier output is missing/invalid.",
    )

    aqa_attack_type_multipliers: dict = Field(
        default_factory=lambda: {
            "contradiction": 2.0,  # Total logical contradiction
            "exception": 1.7,  # Strongly limits applicability
            "derogation": 2.0,  # Directly invalidates the cited norm
            "extinction": 2.3,  # Completely extinguishes the argument
            "factual_impediment": 1.2,  # Serious factual impediment
            "general_opposition": 1.05,  # Generic but relevant opposition
        },
        alias="AQA_ATTACK_TYPE_MULTIPLIERS",
        description="Damage multipliers by attack type.",
    )
    aqa_strength_ratio_by_type: dict = Field(
        default_factory=lambda: {
            "contradiction": 0.05,
            "exception": 0.05,
            "derogation": 0.0,
            "extinction": 0.0,
            "factual_impediment": 0.55,
            "general_opposition": 0.55,
        },
        alias="AQA_STRENGTH_RATIO_BY_TYPE",
        description="Per-attack-type strength ratio thresholds. "
        "An attack of a given type needs attacker_base >= ratio * target_base. "
        "Specific attack types can use near-zero ratios, while general_opposition "
        "and factual_impediment stay more selective.",
    )
    aqa_structural_adjustments_enabled: bool = Field(
        default=True,
        alias="AQA_STRUCTURAL_ADJUSTMENTS_ENABLED",
        description="Enable structural chain adjustments (redundancy penalty).",
    )
    aqa_redundancy_similarity_threshold: float = Field(
        default=0.72,
        alias="AQA_REDUNDANCY_SIMILARITY_THRESHOLD",
        description="Pairwise similarity threshold above which intra-chain links are considered redundant.",
    )
    aqa_redundancy_penalty_weight: float = Field(
        default=0.25,
        alias="AQA_REDUNDANCY_PENALTY_WEIGHT",
        description="Scaling factor applied to redundancy penalty at chain aggregation level.",
    )
    aqa_redundancy_max_penalty: float = Field(
        default=0.18,
        alias="AQA_REDUNDANCY_MAX_PENALTY",
        description="Upper bound for redundancy penalty applied to a side net plausibility.",
    )
    aqa_attack_coverage_enabled: bool = Field(
        default=True,
        alias="AQA_ATTACK_COVERAGE_ENABLED",
        description="Enable counter attack-coverage bonus based on distinct reasoner weak-point axes hit.",
    )
    aqa_attack_coverage_similarity_threshold: float = Field(
        default=0.78,
        alias="AQA_ATTACK_COVERAGE_SIMILARITY_THRESHOLD",
        description="Similarity threshold used to cluster reasoner links into weak-point axes.",
    )
    aqa_attack_coverage_overlap_threshold: float = Field(
        default=0.45,
        alias="AQA_ATTACK_COVERAGE_OVERLAP_THRESHOLD",
        description="Minimum overlap required for an active counter attack to count toward axis coverage.",
    )
    aqa_attack_coverage_min_attack_value: float = Field(
        default=0.08,
        alias="AQA_ATTACK_COVERAGE_MIN_ATTACK_VALUE",
        description="Minimum attack_value required for an active counter attack to count toward axis coverage.",
    )
    aqa_attack_coverage_bonus_weight: float = Field(
        default=0.12,
        alias="AQA_ATTACK_COVERAGE_BONUS_WEIGHT",
        description="Scaling factor applied to Weak-Point Coverage Score bonus.",
    )
    aqa_attack_coverage_max_bonus: float = Field(
        default=0.15,
        alias="AQA_ATTACK_COVERAGE_MAX_BONUS",
        description="Upper bound for attack-coverage bonus applied to contra net plausibility.",
    )
    aqa_attack_coverage_second_hit_weight: float = Field(
        default=0.30,
        alias="AQA_ATTACK_COVERAGE_SECOND_HIT_WEIGHT",
        description="Diminishing-return weight for second strongest attack on same axis.",
    )
    aqa_attack_coverage_third_hit_weight: float = Field(
        default=0.10,
        alias="AQA_ATTACK_COVERAGE_THIRD_HIT_WEIGHT",
        description="Diminishing-return weight for third strongest attack on same axis.",
    )
    aqa_verdict_use_adjusted_score: bool = Field(
        default=True,
        alias="AQA_VERDICT_USE_ADJUSTED_SCORE",
        description="Use structurally adjusted net plausibility for final AQA verdict/score.",
    )
    aqa_lock_reasoner_plausibility: bool = Field(
        default=False,
        alias="AQA_LOCK_REASONER_PLAUSIBILITY",
        description=(
            "When enabled, the Reasoner-side net plausibility is computed from "
            "intrinsic link strength (base_score + precedent delta), making it "
            "stable across A/B Counter comparisons with shared Reasoner."
        ),
    )

    aqa_severity_book_map: dict = Field(
        default_factory=lambda: {
            "persone_famiglia": "I_civile",
            "successioni": "II_civile",
            "proprieta": "III_civile",
            "diritti_reali": "III_civile",
            "obbligazioni": "IV_civile",
            "contratti_generali": "IV_civile",
            "contratti_speciali": "IV_civile",
            "responsabilita": "IV_civile",
            "lavoro": "V_civile",
            "tutela_diritti": "VI_civile",
            "prescrizione": "VI_civile",
            "decadenza": "VI_civile",
            "generale": "I_penale",
            "delitti": "II_penale",
            "delitti_persona": "II_penale",
            "delitti_patrimonio": "II_penale",
            "delitti_stato": "II_penale",
            "contravvenzioni": "III_penale",
            "amministrativo": "AMM_L241",
        },
        alias="AQA_SEVERITY_BOOK_MAP",
        description="Map severity_category → libro identifier for same-book checks.",
    )
    aqa_verdict_pos_threshold: float = Field(
        default=0.2,
        alias="AQA_VERDICT_POS_THRESHOLD",
        description="Final plausibility threshold for 'plausible'.",
    )
    aqa_verdict_neg_threshold: float = Field(
        default=-0.2,
        alias="AQA_VERDICT_NEG_THRESHOLD",
        description="Final plausibility threshold for 'implausible'.",
    )
    aqa_embedding_model: str = Field(
        default="all-mpnet-base-v2",
        alias="AQA_EMBEDDING_MODEL",
        description="Sentence-transformers model for overlap embeddings.",
    )
    aqa_argument_quality_model: str = Field(
        default="",
        alias="AQA_ARGUMENT_QUALITY_MODEL",
        description="Optional argument-quality classifier model name.",
    )
    aqa_argument_quality_use_model: bool = Field(
        default=False,
        alias="AQA_ARGUMENT_QUALITY_USE_MODEL",
        description="Use argument-quality model when available.",
    )
    aqa_tfidf_max_features: int = Field(
        default=5000,
        alias="AQA_TFIDF_MAX_FEATURES",
        description="Max features for TF-IDF overlap vectors.",
    )
    aqa_normsupport_max_citations: int = Field(
        default=3,
        alias="AQA_NORMSUPPORT_MAX_CITATIONS",
        description="Cap for normalized citation count in NormSupport.",
    )
    aqa_normsupport_citation_weight: float = Field(
        default=0.7,
        alias="AQA_NORMSUPPORT_CITATION_WEIGHT",
        description="Weight for citation count in NormSupport.",
    )
    aqa_normsupport_retrieved_weight: float = Field(
        default=0.3,
        alias="AQA_NORMSUPPORT_RETRIEVED_WEIGHT",
        description="Weight for retrieved_norms similarity in NormSupport.",
    )
    aqa_normsupport_retrieved_agg: str = Field(
        default="avg",
        alias="AQA_NORMSUPPORT_RETRIEVED_AGG",
        description="Aggregation for retrieved_norms similarity: avg or max.",
    )
    aqa_severity_map_penale: dict = Field(
        default_factory=lambda: {
            "i": "generale",
            "ii": "delitti",
            "iii": "contravvenzioni",
            "primo": "generale",
            "secondo": "delitti",
            "terzo": "contravvenzioni",
        },
        alias="AQA_SEVERITY_MAP_PENALE",
        description="Mapping from libro token to severity category (Codice Penale).",
    )
    aqa_severity_map_civile: dict = Field(
        default_factory=lambda: {
            "i": "persone_famiglia",
            "ii": "successioni",
            "iii": "proprieta",
            "iv": "obbligazioni",
            "v": "lavoro",
            "vi": "tutela_diritti",
            "primo": "persone_famiglia",
            "secondo": "successioni",
            "terzo": "proprieta",
            "quarto": "obbligazioni",
            "quinto": "lavoro",
            "sesto": "tutela_diritti",
        },
        alias="AQA_SEVERITY_MAP_CIVILE",
        description="Mapping from libro token to severity category (Codice Civile).",
    )

    # =========================================================================
    # Classifier LLM Defaults (task-specific overrides)
    # =========================================================================
    classifier_temperature: float = Field(
        default=0.0,
        alias="CLASSIFIER_TEMPERATURE",
        description="Temperature for classification tasks (stance, causality, attack-type). "
        "Deterministic by default.",
    )
    nli_max_tokens: int = Field(
        default=64,
        alias="NLI_MAX_TOKENS",
        description="Max tokens for NLI / contradiction detection LLM calls.",
    )
    repair_max_tokens: int = Field(
        default=2048,
        alias="REPAIR_MAX_TOKENS",
        description="Max tokens for chain repair / rewrite LLM calls.",
    )
    attack_type_max_tokens: int = Field(
        default=256,
        alias="ATTACK_TYPE_MAX_TOKENS",
        description="Max tokens for attack-type classification LLM calls.",
    )
    taxonomy_max_tokens: int = Field(
        default=20,
        alias="TAXONOMY_MAX_TOKENS",
        description="Max tokens for causality taxonomy classification LLM calls.",
    )
    taxonomy_filter_max_tokens: int = Field(
        default=10,
        alias="TAXONOMY_FILTER_MAX_TOKENS",
        description="Max tokens for taxonomy norm relevance filter LLM calls.",
    )

    # =========================================================================
    # Resilience / Retry
    # =========================================================================
    model_down_ttl: float = Field(
        default=300.0,
        alias="MODEL_DOWN_TTL",
        description="Seconds to wait before retrying a model marked as down.",
    )
    chain_max_retries: int = Field(
        default=5,
        alias="CHAIN_MAX_RETRIES",
        description="Maximum generation attempts for reasoning / counter-reasoning chains.",
    )
    chain_max_steps: int = Field(
        default=10,
        alias="CHAIN_MAX_STEPS",
        description="Safety cap: maximum reasoning steps per iterative chain (LLM decides when to stop).",
    )
    chain_min_steps: int = Field(
        default=3,
        alias="CHAIN_MIN_STEPS",
        description="Minimum reasoning steps before the LLM is allowed to conclude.",
    )

    # =========================================================================
    # Planning Ablation Flags (DoE Control)
    # =========================================================================
    enable_planning_reasoner: bool = Field(
        default=True,
        alias="ENABLE_PLANNING_REASONER",
        description="Enable planning phase in Reasoner agent (for ablation studies).",
    )
    enable_planning_counter: bool = Field(
        default=True,
        alias="ENABLE_PLANNING_COUNTER",
        description="Enable planning phase in Counter-Reasoner agent (for ablation studies).",
    )
    response_language: str = Field(
        default="it",
        alias="RESPONSE_LANGUAGE",
        description=(
            "Output language for LLM reasoning content. "
            "'it' = Italian (default); 'en' = English. "
            "Internal classification prompts (consistency, NLI) are unaffected."
        ),
    )

    # =========================================================================
    # LLM Backend: Groq Cloud (default) vs local vLLM (HPC)
    # =========================================================================
    llm_backend: str = Field(
        default="groq",
        alias="LLM_BACKEND",
        description="'groq' for Groq Cloud (default); 'local' for vLLM offline inference.",
    )
    # HPC cluster alias → HuggingFace model ID mapping (code-owned, not env-driven)
    VLLM_ALIAS_MAP: ClassVar[dict[str, str]] = {
        "deepseek_r1": "deepseek-ai/DeepSeek-R1",
        "gpt_oss_120b": "openai/gpt-oss-120b",
        "qwen_25_72b": "Qwen/Qwen2.5-72B-Instruct",
        "groq_llama_3_3_70b_versatile": "meta-llama/Llama-3.3-70B-Instruct",
        "llama_4_maverick_17b": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    }

    # Aliases that produce <think>…</think> reasoning tokens (stripped before use)
    VLLM_REASONING_ALIASES: ClassVar[frozenset[str]] = frozenset(
        {"deepseek_r1", "gpt_oss_120b"}
    )

    # =========================================================================
    # OpenRouter backend (OpenAI-compatible, paid API, provider-pinned).
    # Reuses the Groq/ChatGroq clients pointed at the OpenRouter base URL.
    # Select with LLM_BACKEND=openrouter.
    # =========================================================================
    OPENROUTER_ALIAS_MAP: ClassVar[dict[str, str]] = {
        "qwen3_30b_instruct": "qwen/qwen3-30b-a3b-instruct-2507",
        "qwen3_30b_thinking": "qwen/qwen3-30b-a3b-thinking-2507",
        "llama_4_scout": "meta-llama/llama-4-scout",
        # DeepInfra-served pair (pin provider=deepinfra): Llama-3.3-70B has a
        # single DeepInfra endpoint (deepinfra/turbo, fp8) so it is auto-selected;
        # gpt-oss-120b defaults to the cheaper deepinfra/bf16.
        "gpt_oss_120b": "openai/gpt-oss-120b",
        "llama_3_3_70b": "meta-llama/llama-3.3-70b-instruct",
    }
    # Aliases whose OpenRouter model returns reasoning; OpenRouter keeps the
    # chain-of-thought in a separate `reasoning` field, so `content` is already
    # the clean final answer (no <think> stripping needed).
    OPENROUTER_REASONING_ALIASES: ClassVar[frozenset[str]] = frozenset(
        {"qwen3_30b_thinking", "gpt_oss_120b"}
    )
    openrouter_api_key: str = Field(
        default="",
        alias="OPENROUTER_API_KEY",
        description="API key for the OpenRouter backend (llm_backend='openrouter').",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
        description="OpenRouter OpenAI-compatible endpoint.",
    )
    openrouter_provider_only: list[str] = Field(
        default_factory=lambda: ["alibaba", "deepinfra"],
        alias="OPENROUTER_PROVIDER_ONLY",
        description="OpenRouter provider preference order: Alibaba first (R/C Qwen), "
        "then DeepInfra (cheapest for models Alibaba does not serve, e.g. Scout).",
    )
    openrouter_allow_fallbacks: bool = Field(
        default=True,
        alias="OPENROUTER_ALLOW_FALLBACKS",
        description="Allow OpenRouter to fall back to other providers when the preferred "
        "one (e.g. Alibaba) does not serve a model (e.g. Scout). Set false for strict single-provider.",
    )
    openrouter_aux_model: str = Field(
        default="qwen/qwen3-30b-a3b-instruct-2507",
        alias="OPENROUTER_AUX_MODEL",
        description="Fixed OpenRouter slug for retrieval/classifier/evaluator calls.",
    )
    openrouter_reasoning_effort: str = Field(
        default="",
        alias="OPENROUTER_REASONING_EFFORT",
        description="Reasoning effort for thinking models: '', 'low', 'medium' or 'high'. "
        "Empty = provider default. 'low' cuts thinking tokens/latency but can regress "
        "the Counter toward abstention; verify quality before a full DoE.",
    )
    openrouter_reasoning_max_tokens: int = Field(
        default=0,
        alias="OPENROUTER_REASONING_MAX_TOKENS",
        description="Hard cap on reasoning tokens for thinking models (0 = unset). "
        "More portable than 'effort' across OpenRouter providers (e.g. Alibaba Qwen).",
    )

    ancillary_max_tokens_cap: int = Field(
        default=1536,
        alias="ANCILLARY_MAX_TOKENS_CAP",
        description="Max tokens for short ancillary JSON calls (fact-lock, NLI, alignment, "
        "target maps, decomposition, gates). Reasoning models spend their completion "
        "budget on chain-of-thought before the JSON: at 320 the payload came back empty "
        "(Groq: 400 json_validate_failed; OpenRouter: silently truncated). Stress-tested "
        "on the real prompts (gpt-oss effort=low + llama70, DeepInfra): worst observed "
        "total was 1183 tokens (decompose_conclusion) -> 1536 keeps ~30% headroom over "
        "the worst case. Ceiling, not target: models stop at the closing brace.",
    )
    reasoner_planner_max_tokens_cap: int = Field(
        default=1600,
        alias="REASONER_PLANNER_MAX_TOKENS_CAP",
        description="Per-call max_tokens cap for Reasoner planner generation. Sized from "
        "stress tests on the real plan prompt (gpt-oss effort=low): worst observed total "
        "943 tokens (CoT + 10-step plan JSON); 1600 keeps comfortable headroom.",
    )
    reasoner_planner_min_tokens: int = Field(
        default=192,
        alias="REASONER_PLANNER_MIN_TOKENS",
        description="Minimum floor for Reasoner planner max_tokens shrink policy.",
    )
    reasoner_support_step_max_tokens_cap: int = Field(
        default=1400,
        alias="REASONER_SUPPORT_STEP_MAX_TOKENS_CAP",
        description="Per-call max_tokens cap for Reasoner support-step generation.",
    )
    reasoner_support_step_min_tokens: int = Field(
        default=256,
        alias="REASONER_SUPPORT_STEP_MIN_TOKENS",
        description="Minimum floor for Reasoner support-step max_tokens shrink policy.",
    )
    reasoner_causality_classifier_max_tokens_cap: int = Field(
        default=256,
        alias="REASONER_CAUSALITY_CLASSIFIER_MAX_TOKENS_CAP",
        description="Per-call max_tokens cap for post-hoc causality classification.",
    )
    reasoner_conclusion_max_tokens_cap: int = Field(
        default=512,
        alias="REASONER_CONCLUSION_MAX_TOKENS_CAP",
        description="Per-call max_tokens cap for Reasoner conclusion generation.",
    )
    counter_planner_max_tokens_cap: int = Field(
        default=1600,
        alias="COUNTER_PLANNER_MAX_TOKENS_CAP",
        description="Per-call max_tokens cap for Counter planner generation. Sized from "
        "stress tests on the real plan prompt (gpt-oss effort=low): worst observed total "
        "956 tokens (CoT spike 501 + plan JSON); 1600 keeps comfortable headroom.",
    )
    counter_planner_min_tokens: int = Field(
        default=192,
        alias="COUNTER_PLANNER_MIN_TOKENS",
        description="Minimum floor for Counter planner max_tokens shrink policy.",
    )
    counter_support_step_max_tokens_cap: int = Field(
        default=1400,
        alias="COUNTER_SUPPORT_STEP_MAX_TOKENS_CAP",
        description="Per-call max_tokens cap for Counter step generation.",
    )
    counter_support_step_min_tokens: int = Field(
        default=256,
        alias="COUNTER_SUPPORT_STEP_MIN_TOKENS",
        description="Minimum floor for Counter support-step max_tokens shrink policy.",
    )
    counter_second_pass_enabled: bool = Field(
        default=True,
        alias="COUNTER_SECOND_PASS_ENABLED",
        description="Enable targeted second-pass retrieval for Counter-Reasoner when against statutes are scarce.",
    )
    counter_second_pass_min_against_statutes: int = Field(
        default=8,
        alias="COUNTER_SECOND_PASS_MIN_AGAINST_STATUTES",
        description="If initial counter statutes are below this threshold, run targeted second-pass retrieval.",
    )
    counter_second_pass_top_k: int = Field(
        default=40,
        alias="COUNTER_SECOND_PASS_TOP_K",
        description="Top-K statutes requested in each targeted second-pass retrieval query.",
    )
    counter_second_pass_max_queries: int = Field(
        default=2,
        alias="COUNTER_SECOND_PASS_MAX_QUERIES",
        description="Maximum number of targeted retrieval queries generated from selected counter attacks.",
    )
    counter_second_pass_max_additional: int = Field(
        default=25,
        alias="COUNTER_SECOND_PASS_MAX_ADDITIONAL",
        description="Maximum additional statutes retained from the targeted second-pass retrieval.",
    )
    counter_step_expansion_enabled: bool = Field(
        default=True,
        alias="COUNTER_STEP_EXPANSION_ENABLED",
        description="Enable optional post-generation expansion of compressed multi-attack counter steps.",
    )
    counter_step_expansion_min_attacks: int = Field(
        default=2,
        alias="COUNTER_STEP_EXPANSION_MIN_ATTACKS",
        description="Minimum number of attacks used in a step before expansion is attempted.",
    )
    counter_step_expansion_max_extra_per_step: int = Field(
        default=2,
        alias="COUNTER_STEP_EXPANSION_MAX_EXTRA_PER_STEP",
        description="Maximum number of extra satellite steps generated from one compressed parent step.",
    )
    counter_step_expansion_max_extra_total: int = Field(
        default=6,
        alias="COUNTER_STEP_EXPANSION_MAX_EXTRA_TOTAL",
        description="Global cap of extra satellite steps added by expansion during one counter run.",
    )

    # =========================================================================
    # Text Truncation (prompt context limits)
    # =========================================================================
    truncation_nli_text: int = Field(
        default=700,
        alias="TRUNCATION_NLI_TEXT",
        description="Max chars per passage for NLI / attack-type prompts.",
    )
    truncation_chain_text: int = Field(
        default=4000,
        alias="TRUNCATION_CHAIN_TEXT",
        description="Max chars of full chain text for repair prompts.",
    )
    truncation_context: int = Field(
        default=800,
        alias="TRUNCATION_CONTEXT",
        description="Max chars of contextual snippets in prompts.",
    )
    truncation_tool_summary: int = Field(
        default=700,
        alias="TRUNCATION_TOOL_SUMMARY",
        description="Max chars of precedent summary returned by search tools.",
    )
    truncation_prompt_testo: int = Field(
        default=1200,
        alias="TRUNCATION_PROMPT_TESTO",
        description="Max chars of statute text in base agent prompt formatting.",
    )
    truncation_prompt_testo_amministrativo: int = Field(
        default=500,
        alias="TRUNCATION_PROMPT_TESTO_AMMINISTRATIVO",
        description="Max chars of administrative statute text in base agent prompt formatting.",
    )
    truncation_prompt_summary: int = Field(
        default=600,
        alias="TRUNCATION_PROMPT_SUMMARY",
        description="Max chars of precedent summary in base agent prompt formatting.",
    )
    truncation_counter_claim: int = Field(
        default=1500,
        alias="TRUNCATION_COUNTER_CLAIM",
        description="Max chars of claim text used in Counter-Reasoner meta prompts.",
    )
    truncation_counter_reasoner_conclusion: int = Field(
        default=1400,
        alias="TRUNCATION_COUNTER_REASONER_CONCLUSION",
        description="Max chars of reasoner conclusion passed into Counter-Reasoner meta prompts.",
    )
    truncation_counter_precondition: int = Field(
        default=320,
        alias="TRUNCATION_COUNTER_PRECONDITION",
        description="Max chars of one attack precondition string in Counter-Reasoner checks.",
    )
    truncation_counter_conclusion: int = Field(
        default=1400,
        alias="TRUNCATION_COUNTER_CONCLUSION",
        description="Max chars of reasoner conclusion used for counter decomposition prompts.",
    )
    truncation_counter_attack_statement: int = Field(
        default=300,
        alias="TRUNCATION_COUNTER_ATTACK_STATEMENT",
        description="Max chars for one decomposed counter attack statement.",
    )
    truncation_counter_attack_vector: int = Field(
        default=180,
        alias="TRUNCATION_COUNTER_ATTACK_VECTOR",
        description="Max chars for one decomposed counter attack vector/focus string.",
    )
    truncation_counter_novelty_key: int = Field(
        default=48,
        alias="TRUNCATION_COUNTER_NOVELTY_KEY",
        description="Max chars for normalized novelty-key identifiers in Counter planning.",
    )
    truncation_consistency_db_title: int = Field(
        default=200,
        alias="TRUNCATION_CONSISTENCY_DB_TITLE",
        description="Max chars of DB title passed to consistency pertinence checks.",
    )
    truncation_consistency_db_text: int = Field(
        default=1200,
        alias="TRUNCATION_CONSISTENCY_DB_TEXT",
        description="Max chars of DB article text passed to consistency pertinence checks.",
    )
    truncation_consistency_db_summary_mismatch: int = Field(
        default=1500,
        alias="TRUNCATION_CONSISTENCY_DB_SUMMARY_MISMATCH",
        description="Max chars of DB precedent summary used in mismatch validation prompts.",
    )
    truncation_consistency_db_summary_repair: int = Field(
        default=2000,
        alias="TRUNCATION_CONSISTENCY_DB_SUMMARY_REPAIR",
        description="Max chars of DB precedent summary used in repair prompts.",
    )
    truncation_consistency_signature: int = Field(
        default=300,
        alias="TRUNCATION_CONSISTENCY_SIGNATURE",
        description="Max chars for normalized signatures used in consistency matching heuristics.",
    )
    claim_classifier_max_tokens: int = Field(
        default=64,
        alias="CLAIM_CLASSIFIER_MAX_TOKENS",
        description="Max completion tokens for claim classifier LLM calls.",
    )

    # =========================================================================
    # AQA Scoring Weights & Thresholds (extended)
    # =========================================================================

    aqa_max_age: float = Field(
        default=50.0,
        alias="AQA_MAX_AGE",
        description="Maximum precedent age (years) for recency scoring. Older = 0 recency.",
    )
    aqa_default_confidence: float = Field(
        default=0.7,
        alias="AQA_DEFAULT_CONFIDENCE",
        description="Default confidence when stance confidence is missing from precedent metadata.",
    )
    aqa_precedent_sim_threshold: float = Field(
        default=0.5,
        alias="AQA_PRECEDENT_SIM_THRESHOLD",
        description="Minimum similarity score for matching a precedent to a link.",
    )
    aqa_dominant_attacks_limit: int = Field(
        default=10,
        alias="AQA_DOMINANT_ATTACKS_LIMIT",
        description="Max number of dominant attacks to include in AQA report.",
    )
    aqa_bindingness_map: dict = Field(
        default_factory=lambda: {
            "cassazione": 1.0,
            "appello": 0.7,
            "tribunale": 0.4,
            "other": 0.3,
        },
        alias="AQA_BINDINGNESS_MAP",
        description="Bindingness score by court level (substring match).",
    )

    # =========================================================================
    # Readability & Coherence Weights
    # =========================================================================
    readability_structure_weight: float = Field(
        default=0.5,
        alias="READABILITY_STRUCTURE_WEIGHT",
        description="Weight of structure score in argument quality.",
    )
    readability_quality_weight: float = Field(
        default=0.5,
        alias="READABILITY_QUALITY_WEIGHT",
        description="Weight of quality score in argument quality.",
    )
    coherence_base_weight: float = Field(
        default=0.7,
        alias="COHERENCE_BASE_WEIGHT",
        description="Weight for base similarity in coherence scoring.",
    )
    coherence_chain_weight: float = Field(
        default=0.3,
        alias="COHERENCE_CHAIN_WEIGHT",
        description="Weight for inter-sentence similarity chain in coherence scoring.",
    )

    # =========================================================================
    # Consistency Checker Thresholds
    # =========================================================================
    cc_text_match_threshold: float = Field(
        default=0.8,
        alias="CC_TEXT_MATCH_THRESHOLD",
        description="Minimum cosine similarity for a cited text to be considered matching the DB text.",
    )
    cc_core_threshold: int = Field(
        default=2,
        alias="CC_CORE_THRESHOLD",
        description="Minimum number of core indicators for an article to be classified as CORE.",
    )
    cc_conclusion_bonus: int = Field(
        default=2,
        alias="CC_CONCLUSION_BONUS",
        description="Extra core-indicator points when article is cited in the conclusion.",
    )
    cc_occurrence_threshold: int = Field(
        default=3,
        alias="CC_OCCURRENCE_THRESHOLD",
        description="Number of text occurrences that adds a core indicator point.",
    )

    # =========================================================================
    # API Metadata
    # =========================================================================
    api_version: str = Field(
        default="0.2.0",
        alias="API_VERSION",
        description="API version string returned by /health.",
    )

    # =========================================================================
    # Paths
    # =========================================================================
    @model_validator(mode="after")
    def _discover_groq_api_keys(self) -> "Settings":
        """Scan environment for all GROQ_API_KEY_V* variables.

        Discovers keys dynamically (V1, V2, …, V99) so adding more
        keys to .env is all that's needed — no code changes required.
        """
        keys: list[str] = []
        # Start with the primary key (V1) that Pydantic already loaded
        if self.groq_api_key:
            keys.append(self.groq_api_key)
        # Discover V2, V3, … VN
        idx = 2
        while True:
            val = os.environ.get(f"GROQ_API_KEY_V{idx}", "")
            if not val:
                break
            if val not in keys:  # avoid duplicates
                keys.append(val)
            idx += 1
        mode = str(self.search_query_terms_mode or "").strip().lower()
        if mode != "llm":
            object.__setattr__(self, "search_query_terms_mode", "llm")

        reasoner_temp = float(self.reasoner_default_temperature)
        counter_temp = float(self.counter_default_temperature)
        object.__setattr__(
            self,
            "reasoner_default_temperature",
            max(0.0, min(1.0, reasoner_temp)),
        )
        object.__setattr__(
            self,
            "counter_default_temperature",
            max(0.0, min(1.0, counter_temp)),
        )

        reasoner_aliases = [
            str(a).strip()
            for a in self.reasoner_model_fallback_aliases
            if str(a).strip()
        ]
        if not reasoner_aliases:
            reasoner_aliases = [self.REASONER_DEFAULT_MODEL_ALIAS]
        object.__setattr__(self, "reasoner_model_fallback_aliases", reasoner_aliases)

        counter_aliases = [
            str(a).strip()
            for a in self.counter_model_fallback_aliases
            if str(a).strip()
        ]
        if not counter_aliases:
            counter_aliases = [self.COUNTER_DEFAULT_MODEL_ALIAS]
        object.__setattr__(self, "counter_model_fallback_aliases", counter_aliases)

        procedural_categories = [
            str(c).strip().lower()
            for c in self.aqa_procedural_severity_categories
            if str(c).strip()
        ]
        object.__setattr__(
            self,
            "aqa_procedural_severity_categories",
            procedural_categories,
        )

        valid_attack_types = [
            str(t).strip().lower()
            for t in self.aqa_valid_attack_types
            if str(t).strip()
        ]
        if not valid_attack_types:
            valid_attack_types = ["general_opposition"]
        object.__setattr__(self, "aqa_valid_attack_types", valid_attack_types)

        default_attack = str(self.aqa_default_attack_type or "").strip().lower()
        if default_attack not in valid_attack_types:
            default_attack = "general_opposition"
        object.__setattr__(self, "aqa_default_attack_type", default_attack)

        ancillary_cap = max(1, int(self.ancillary_max_tokens_cap))
        object.__setattr__(self, "ancillary_max_tokens_cap", ancillary_cap)

        r_plan_min = max(1, int(self.reasoner_planner_min_tokens))
        r_plan_cap = max(r_plan_min, int(self.reasoner_planner_max_tokens_cap))
        r_step_min = max(1, int(self.reasoner_support_step_min_tokens))
        r_step_cap = max(r_step_min, int(self.reasoner_support_step_max_tokens_cap))
        object.__setattr__(self, "reasoner_planner_min_tokens", r_plan_min)
        object.__setattr__(self, "reasoner_planner_max_tokens_cap", r_plan_cap)
        object.__setattr__(self, "reasoner_support_step_min_tokens", r_step_min)
        object.__setattr__(self, "reasoner_support_step_max_tokens_cap", r_step_cap)
        object.__setattr__(
            self,
            "reasoner_causality_classifier_max_tokens_cap",
            max(1, int(self.reasoner_causality_classifier_max_tokens_cap)),
        )
        object.__setattr__(
            self,
            "reasoner_conclusion_max_tokens_cap",
            max(1, int(self.reasoner_conclusion_max_tokens_cap)),
        )

        c_plan_min = max(1, int(self.counter_planner_min_tokens))
        c_plan_cap = max(c_plan_min, int(self.counter_planner_max_tokens_cap))
        c_step_min = max(1, int(self.counter_support_step_min_tokens))
        c_step_cap = max(c_step_min, int(self.counter_support_step_max_tokens_cap))
        object.__setattr__(self, "counter_planner_min_tokens", c_plan_min)
        object.__setattr__(self, "counter_planner_max_tokens_cap", c_plan_cap)
        object.__setattr__(self, "counter_support_step_min_tokens", c_step_min)
        object.__setattr__(self, "counter_support_step_max_tokens_cap", c_step_cap)

        object.__setattr__(
            self,
            "aqa_redundancy_similarity_threshold",
            max(0.0, min(0.99, float(self.aqa_redundancy_similarity_threshold))),
        )
        object.__setattr__(
            self,
            "aqa_redundancy_penalty_weight",
            max(0.0, min(1.0, float(self.aqa_redundancy_penalty_weight))),
        )
        object.__setattr__(
            self,
            "aqa_redundancy_max_penalty",
            max(0.0, min(0.95, float(self.aqa_redundancy_max_penalty))),
        )
        object.__setattr__(
            self,
            "aqa_attack_coverage_similarity_threshold",
            max(0.0, min(0.99, float(self.aqa_attack_coverage_similarity_threshold))),
        )
        object.__setattr__(
            self,
            "aqa_attack_coverage_overlap_threshold",
            max(0.0, min(1.0, float(self.aqa_attack_coverage_overlap_threshold))),
        )
        object.__setattr__(
            self,
            "aqa_attack_coverage_min_attack_value",
            max(0.0, min(1.0, float(self.aqa_attack_coverage_min_attack_value))),
        )
        object.__setattr__(
            self,
            "aqa_attack_coverage_bonus_weight",
            max(0.0, min(1.0, float(self.aqa_attack_coverage_bonus_weight))),
        )
        object.__setattr__(
            self,
            "aqa_attack_coverage_max_bonus",
            max(0.0, min(0.95, float(self.aqa_attack_coverage_max_bonus))),
        )
        object.__setattr__(
            self,
            "aqa_attack_coverage_second_hit_weight",
            max(0.0, min(1.0, float(self.aqa_attack_coverage_second_hit_weight))),
        )
        object.__setattr__(
            self,
            "aqa_attack_coverage_third_hit_weight",
            max(0.0, min(1.0, float(self.aqa_attack_coverage_third_hit_weight))),
        )
        object.__setattr__(
            self,
            "counter_step_expansion_min_attacks",
            max(2, int(self.counter_step_expansion_min_attacks)),
        )
        object.__setattr__(
            self,
            "counter_step_expansion_max_extra_per_step",
            max(1, int(self.counter_step_expansion_max_extra_per_step)),
        )
        object.__setattr__(
            self,
            "counter_step_expansion_max_extra_total",
            max(0, int(self.counter_step_expansion_max_extra_total)),
        )

        object.__setattr__(self, "_groq_api_keys", keys)
        return self

    @property
    def groq_api_keys(self) -> list[str]:
        """Get all available Groq API keys (non-empty, dynamically discovered)."""
        return list(self._groq_api_keys)

    @property
    def model_alias_map(self) -> dict[str, str]:
        """Centralized alias -> provider model mapping."""
        if self.llm_backend == "openrouter":
            # OpenRouter slugs take precedence, but keep the base map so aux
            # aliases still resolve (then get sanitized to the aux slug).
            return {**self.MODEL_ALIAS_MAP, **self.OPENROUTER_ALIAS_MAP}
        return dict(self.MODEL_ALIAS_MAP)

    # ── OpenRouter helpers ────────────────────────────────────────────────
    @property
    def openrouter_api_keys(self) -> list[str]:
        """OpenRouter API key(s). Single key by default; list for the resilient loop."""
        return [self.openrouter_api_key] if self.openrouter_api_key else []

    def openrouter_reasoning_slugs(self) -> set[str]:
        """OpenRouter slugs that emit reasoning (thinking models)."""
        return {
            self.OPENROUTER_ALIAS_MAP[a]
            for a in self.OPENROUTER_REASONING_ALIASES
            if a in self.OPENROUTER_ALIAS_MAP
        }

    @property
    def openrouter_aux_slug(self) -> str:
        """Aux model as an OpenRouter slug (OPENROUTER_AUX_MODEL may be an alias or a raw slug)."""
        return self.OPENROUTER_ALIAS_MAP.get(
            self.openrouter_aux_model, self.openrouter_aux_model
        )

    def openrouter_known_slugs(self) -> set[str]:
        """OpenRouter slugs we accept as-is (R/C models + fixed aux model)."""
        slugs = set(self.OPENROUTER_ALIAS_MAP.values())
        if self.openrouter_aux_slug:
            slugs.add(self.openrouter_aux_slug)
        return slugs

    def resolve_openrouter_slug(self, model: str | None) -> str:
        """Sanitize an incoming model id to a valid OpenRouter slug.

        R/C models already resolve to Qwen slugs (kept as-is); anything else
        (aux/retrieval/evaluator ids that resolve to Groq/HF ids) falls back
        to the fixed auxiliary OpenRouter model.
        """
        m = (model or "").strip()
        return m if m in self.openrouter_known_slugs() else self.openrouter_aux_slug

    def openrouter_extra_body(self, slug: str) -> dict:
        """Provider preference + reasoning flag injected into the OpenRouter request body.

        Uses ``order`` (preference) rather than ``only`` (hard restriction): the
        configured provider (e.g. Alibaba) is tried first for every model, and a
        model that provider does not serve (e.g. Llama-4-Scout) falls back to its
        own providers instead of 404-ing. Set OPENROUTER_ALLOW_FALLBACKS=false to
        forbid fallback (strict single provider, but then unsupported models fail).
        """
        body: dict = {}
        pref = list(self.openrouter_provider_only or [])
        if pref:
            body["provider"] = {
                "order": pref,
                "allow_fallbacks": bool(self.openrouter_allow_fallbacks),
            }
        if slug in self.openrouter_reasoning_slugs():
            reasoning: dict = {"exclude": False}
            # OpenRouter treats `effort` and `max_tokens` as mutually exclusive;
            # prefer the explicit token cap (more portable across providers).
            if self.openrouter_reasoning_max_tokens > 0:
                reasoning["max_tokens"] = int(self.openrouter_reasoning_max_tokens)
            else:
                effort = (self.openrouter_reasoning_effort or "").strip().lower()
                if effort in ("low", "medium", "high"):
                    reasoning["effort"] = effort
            body["reasoning"] = reasoning
        return body

    @property
    def available_model_aliases(self) -> list[str]:
        """Aliases exposed to frontend model selectors."""
        return list(self.model_alias_map.keys())

    @property
    def retrieval_model_order_aliases(self) -> list[str]:
        """Ordered model aliases used in retrieval-side resilient runtime calls."""
        deduped: list[str] = []
        for alias in self.RETRIEVAL_MODEL_ORDER_ALIASES:
            if alias not in deduped:
                deduped.append(alias)
        return deduped

    @property
    def pipeline_model_order_aliases(self) -> list[str]:
        """Ordered model aliases used in resilient runtime calls."""
        deduped: list[str] = []
        for alias in self.PIPELINE_MODEL_ORDER_ALIASES:
            if alias not in deduped:
                deduped.append(alias)
        return deduped

    def _resolve_model_alias_order(self, aliases: list[str]) -> list[str]:
        """Resolve alias list to provider model ids, preserving order and uniqueness."""
        models: list[str] = []
        for alias in aliases:
            model = self.resolve_model_name(alias)
            if model and model not in models:
                models.append(model)
        return models

    @property
    def reasoner_model_fallback_order(self) -> list[str]:
        """Provider model ids used by Reasoner resilient fallback."""
        return self._resolve_model_alias_order(self.reasoner_model_fallback_aliases)

    @property
    def retrieval_model_fallback_order(self) -> list[str]:
        """Provider model ids used by retrieval-side resilient calls."""
        return self._resolve_model_alias_order(self.retrieval_model_order_aliases)

    @property
    def counter_model_fallback_order(self) -> list[str]:
        """Provider model ids used by Counter-Reasoner resilient fallback."""
        return self._resolve_model_alias_order(self.counter_model_fallback_aliases)

    @property
    def retrieval_default_model(self) -> str:
        """Primary provider model id for retrieval-side components."""
        models = self.retrieval_model_fallback_order
        return models[0] if models else self.resolve_model_name("gpt_oss_20b")

    @property
    def reasoner_default_model(self) -> str:
        """Default frontend alias for Reasoner model selection."""
        return self.REASONER_DEFAULT_MODEL_ALIAS

    @property
    def counter_default_model(self) -> str:
        """Default frontend alias for Counter-Reasoner model selection."""
        return self.COUNTER_DEFAULT_MODEL_ALIAS

    def resolve_model_name(self, alias_or_model: str | None) -> str:
        """Resolve alias to provider model id; pass-through raw ids unchanged."""
        alias_map = self.model_alias_map
        if not alias_or_model:
            first_alias = self.pipeline_model_order_aliases[0]
            return alias_map.get(first_alias, self.MODEL_ALIAS_MAP[first_alias])
        return alias_map.get(alias_or_model, alias_or_model)

    @property
    def groq_models(self) -> list[str]:
        """Ordered provider model ids for runtime resilience loop."""
        models: list[str] = []
        for alias in self.pipeline_model_order_aliases:
            model = self.resolve_model_name(alias)
            if model and model not in models:
                models.append(model)
        return models

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return _project_root

    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return _project_root / "src" / "data"

    @property
    def embeddings_dir(self) -> Path:
        """Get embeddings directory."""
        return self.data_dir / "embeddings"

    @property
    def statutes_dir(self) -> Path:
        """Get statutes directory."""
        return self.data_dir / "statutes"

    @property
    def precedents_dir(self) -> Path:
        """Get precedents directory."""
        return self.data_dir / "precedents"

    @property
    def taxonomy_path(self) -> Path:
        """Get causality taxonomy file path."""
        return self.project_root / "src" / "agents" / "tools" / "config_taxonomy.json"

    def validate_config(self) -> dict:
        """Validate configuration and return status."""
        issues = []

        if not self.groq_api_keys:
            issues.append("No GROQ_API_KEY_V* keys found in environment")

        if not self.taxonomy_path.exists():
            issues.append(f"Taxonomy file not found: {self.taxonomy_path}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "neo4j_uri": self.neo4j_uri,
            "pipeline_models": self.groq_models,
            "embedding_model": self.embedding_model,
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure only one Settings instance is created.
    """
    return Settings()


# Global settings instance for easy import
settings = get_settings()


if __name__ == "__main__":
    # Test configuration
    print("=" * 60)
    print("LexCausa Configuration")
    print("=" * 60)
    print(f"Project Root: {settings.project_root}")
    print(f"Neo4j URI: {settings.neo4j_uri}")
    print(f"Pipeline Models: {settings.groq_models}")
    print(f"Embedding Model: {settings.embedding_model}")
    print()
    validation = settings.validate_config()
    print(f"Valid: {validation['valid']}")
    if validation["issues"]:
        print(f"Issues: {validation['issues']}")
