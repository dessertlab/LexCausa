"""Utilities for parsing and normalizing Italian statute citations."""

from __future__ import annotations

import re
from dataclasses import dataclass

_UNICODE_CITATION_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
        "\u00a0": " ",  # no-break space
        "\u202f": " ",  # narrow no-break space
    }
)

_SUFFIX_TOKENS = (
    "noviesdecies",
    "octiesdecies",
    "septiesdecies",
    "sexiesdecies",
    "quinquiesdecies",
    "quaterdecies",
    "terdecies",
    "duodecies",
    "undecies",
    "quinquies",
    "septies",
    "quater",
    "sexies",
    "octies",
    "nonies",
    "decies",
    "vicies",
    "ter",
    "bis",
)
_SUFFIX_PATTERN = "|".join(_SUFFIX_TOKENS)
_SUFFIX_TOKEN_SET = {token.lower() for token in _SUFFIX_TOKENS}
_ARTICLE_ID_PATTERN = rf"\d{{1,4}}(?:[-\s]?(?:[a-z0-9]{{2,}}|{_SUFFIX_PATTERN}))?"
_CODE_PATTERN = (
    r"c\.?\s*[cp]\.?|"
    r"cod(?:ice)?\.?\s*(?:civ(?:ile)?|pen(?:ale)?|amm(?:inistrativ[oa])?)|"
    r"l(?:egge)?\.?\s*241\s*/?\s*1990|"
    r"241\s*/?\s*1990"
)

_ARTICLE_REF_PATTERN = re.compile(
    rf"(?:\bart(?:icolo|icoli)?\.?\s*)"
    rf"(?P<article>{_ARTICLE_ID_PATTERN})"
    rf"(?P<trailing>\s*(?P<code>{_CODE_PATTERN})?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ArticleMention:
    """Parsed article mention with normalized id and source hint."""

    article_id: str
    source_hint: str
    raw_code: str
    start: int
    end: int


def normalize_article_id(raw: str) -> str:
    """Normalize article identifiers (e.g., ``62 bis`` -> ``62-bis``)."""
    text = (raw or "").translate(_UNICODE_CITATION_TRANSLATION).strip().lower()
    if not text:
        return ""
    text = re.sub(r"^art(?:icolo|icoli)?\.?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("_", "-")
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"^(\d{1,4})([a-z]{2,})$", r"\1-\2", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def _sanitize_article_suffix(article_id: str, raw_article: str) -> str:
    """Drop spurious non-legal suffixes captured from narrative prose.

    Example: ``art. 1218 per ...`` should map to ``1218`` (not ``1218-per``).
    """
    match = re.fullmatch(r"(\d{1,4})-([a-z]{2,})", article_id)
    if not match:
        return article_id

    suffix = match.group(2).lower()
    has_explicit_hyphen = "-" in raw_article.translate(_UNICODE_CITATION_TRANSLATION)
    if has_explicit_hyphen or suffix in _SUFFIX_TOKEN_SET:
        return article_id
    return match.group(1)


def article_id_to_regex(article_id: str) -> str:
    """Build regex fragment that accepts either ``-`` or whitespace suffix separators."""
    normalized = normalize_article_id(article_id)
    return re.escape(normalized).replace(r"\-", r"[-\s]?")


def infer_source_hint(code_fragment: str) -> str:
    """Infer internal source id from citation code fragment."""
    lower = (code_fragment or "").lower()
    if "241" in lower or "amm" in lower:
        return "codice_amministrativo"
    if "c.c" in lower or ("cod" in lower and "civ" in lower):
        return "codice_civile"
    if "c.p" in lower or ("cod" in lower and "pen" in lower):
        return "codice_penale"
    return ""


def format_article_citation(article_id: str, source_hint: str) -> str:
    """Format canonical citation string from article id + source hint."""
    source = (source_hint or "").strip().lower()
    if source == "codice_penale":
        return f"Art. {article_id} c.p."
    if source == "codice_civile":
        return f"Art. {article_id} c.c."
    if source == "codice_amministrativo":
        return f"Art. {article_id} L. 241/1990"
    return f"Art. {article_id}"


def extract_article_mentions(
    text: str, *, require_code: bool = False
) -> list[ArticleMention]:
    """Extract article mentions from free text.

    Args:
        text: Input text.
        require_code: If True, only keep mentions that include a code fragment
            (e.g. ``c.p.``, ``c.c.``, ``L. 241/1990``).
    """
    mentions: list[ArticleMention] = []
    if not text:
        return mentions
    normalized_text = text.translate(_UNICODE_CITATION_TRANSLATION)

    for match in _ARTICLE_REF_PATTERN.finditer(normalized_text):
        raw_article = match.group("article") or ""
        raw_code = (match.group("code") or "").strip()
        if require_code and not raw_code:
            continue

        article_id = normalize_article_id(raw_article)
        if not article_id:
            continue
        article_id = _sanitize_article_suffix(article_id, raw_article)

        source_hint = infer_source_hint(raw_code)
        mentions.append(
            ArticleMention(
                article_id=article_id,
                source_hint=source_hint,
                raw_code=raw_code,
                start=match.start(),
                end=match.end(),
            )
        )

    return mentions
