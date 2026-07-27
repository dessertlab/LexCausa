"""
LangChain Tools for Neo4j Knowledge Base interaction.

Provides tools for agents to search and retrieve:
- Statutes (Codice Civile, Codice Penale, Codice Amministrativo L. 241/1990)
- Precedents (itacasehold)

Uses LegalSearchPipeline for embedding and vector search - the SAME approach as Tab Ricerca.
"""

import sys
import threading
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from neo4j import Driver, GraphDatabase
from pydantic import BaseModel, Field

from .prompt_registry import render_prompt

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings  # noqa: E402
from services.legal_search import LegalSearchPipeline  # noqa: E402

# Singleton driver for tools (only used for direct Neo4j queries like get_statute_by_article)
_driver: Optional[Driver] = None

# Singleton LegalSearchPipeline (the SAME one that works in Tab Ricerca!)
_legal_search_pipeline: Optional[LegalSearchPipeline] = None
_pipeline_lock = threading.Lock()


def get_driver() -> Driver:
    """Get or create Neo4j driver for direct queries."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
    return _driver


def get_legal_search_pipeline() -> LegalSearchPipeline:
    """Get or create LegalSearchPipeline (singleton, thread-safe).

    This is the SAME pipeline that works in the Search Tab!
    It handles embedding generation and vector search internally.
    """
    global _legal_search_pipeline

    if _legal_search_pipeline is None:
        with _pipeline_lock:
            if _legal_search_pipeline is None:
                print("🔧 [neo4j_tools] Initializing LegalSearchPipeline...")
                _legal_search_pipeline = LegalSearchPipeline()
                print("✅ [neo4j_tools] LegalSearchPipeline ready!")

    return _legal_search_pipeline


class GetStatuteByArticleInput(BaseModel):
    """Input schema for getting statute by article number."""

    articolo: str = Field(description="Article number (e.g., '2043', '40')")
    codice: str = Field(
        description="Reference code: 'codice_civile', 'codice_penale', or 'codice_amministrativo'"
    )


@tool("get_statute_by_article", args_schema=GetStatuteByArticleInput)
def get_statute_by_article_tool(articolo: str, codice: str) -> dict:
    """
    Retrieve a specific article given its number and code.

    Use this function when you know the exact legal reference (e.g., Art. 2043 c.c.).
    """
    driver = get_driver()

    # Normalize articolo: database stores as "art1223", input might be "1223" or "art1223"
    articolo_normalized = articolo.lower().replace("art", "").replace(" ", "").strip()
    articolo_with_prefix = f"art{articolo_normalized}"

    query = """
        MATCH (s:Statute)
        WHERE (s.articolo = $articolo OR s.articolo = $articolo_with_prefix) AND s.source = $codice
        RETURN s.statute_id AS id,
               s.articolo AS articolo,
               s.titolo AS titolo,
               s.testo AS testo,
               s.libro AS libro,
               s.source AS source
        LIMIT 1
    """

    try:
        with driver.session() as session:
            result = session.run(
                query,
                parameters={
                    "articolo": articolo_normalized,
                    "articolo_with_prefix": articolo_with_prefix,
                    "codice": codice,
                },
            )
            record = result.single()

            if record:
                return {
                    "statute_id": record["id"] or "",
                    "articolo": record["articolo"] or articolo,
                    "titolo": record["titolo"] or "No title",
                    "testo": record["testo"] or "No text available",
                    "libro": record["libro"] or "",
                    "source": record["source"] or codice,
                    "found": True,
                }
            # Return a valid dict even when not found
            return {
                "error": f"Article {articolo} not found in {codice}",
                "articolo": articolo,
                "codice": codice,
                "found": False,
            }
    except Exception as e:
        return {
            "error": f"Query failed: {str(e)}",
            "articolo": articolo,
            "codice": codice,
            "found": False,
        }


class SearchPrecedentsInput(BaseModel):
    """Input schema for precedent search."""

    query: str = Field(description="The legal claim to extract keywords from")
    limit: int = Field(
        default=settings.precedents_limit_default,
        description="Maximum number of results (defaults to config PRECEDENTS_LIMIT_DEFAULT)",
    )


def _extract_keywords_from_claim(claim: str) -> list[str]:
    """Use LLM to extract legal keywords from a claim for precedent search.

    Returns a list of keyword strings suitable for Neo4j fulltext queries.
    """
    from langchain_core.messages import HumanMessage

    from services.groq_client import get_chat_groq, resilient_chat_call

    prompt = render_prompt("neo4j_tools.extract_keywords", claim=claim)

    try:
        llm = get_chat_groq(
            model=settings.retrieval_default_model,
            temperature=0.0,
            max_tokens=200,
            api_key=settings.groq_api_key or None,
        )
        response = resilient_chat_call(
            llm,
            [HumanMessage(content=prompt)],
            model_order=settings.retrieval_model_fallback_order,
        )
        raw = response.content.strip()
        keywords = [
            kw.strip().strip("-•*").strip()
            for kw in raw.split("\n")
            if kw.strip() and len(kw.strip()) > 1
        ]
        # Deduplicate preserving order
        seen = set()
        unique = []
        for kw in keywords:
            low = kw.lower()
            if low not in seen:
                seen.add(low)
                unique.append(kw)
        print(f"  🔑 Extracted keywords: {unique}")
        return unique[:10]
    except Exception as e:
        print(f"  ⚠️ Keyword extraction failed: {e}")
        # Fallback: split claim into meaningful chunks
        import re

        words = re.findall(r"\b[a-zA-ZàèéìòùÀÈÉÌÒÙ]{4,}\b", claim)
        return list(dict.fromkeys(words))[:8]


def _search_precedents_by_keywords(
    keywords: list[str],
    limit: int = 5,
) -> list[dict]:
    """Search precedents via Neo4j fulltext index using extracted keywords.

    Builds a Lucene OR query from the keywords and queries the
    ``precedents_fulltext_idx`` index (on ``title`` and ``summary``).
    """
    driver = get_driver()

    # Build Lucene query: each keyword OR'd together
    lucene_query = " OR ".join(keywords)
    print(f"  🔄 Fulltext search with query: '{lucene_query}'")

    query_cypher = """
        CALL db.index.fulltext.queryNodes('precedents_fulltext_idx', $query)
        YIELD node, score
        RETURN node.precedent_id AS id,
               node.title AS title,
               node.summary AS summary,
               node.url AS url,
               node.year AS year,
               node.court AS court,
               node.court_level AS court_level,
               score
        ORDER BY score DESC
        LIMIT $limit
    """

    results: list[dict[str, str | float]] = []
    try:
        with driver.session() as session:
            records = session.run(
                query_cypher,
                parameters={"query": lucene_query, "limit": limit},
            )
            for record in records:
                title = str(record["title"] or "Untitled precedent")
                summary_val = record["summary"]
                if isinstance(summary_val, str):
                    summary = (
                        summary_val[: settings.truncation_tool_summary]
                        if summary_val
                        else "No summary available"
                    )
                else:
                    summary = (
                        str(summary_val)[: settings.truncation_tool_summary]
                        if summary_val is not None
                        else "No summary available"
                    )
                result_item = {
                    "precedent_id": record["id"] or "",
                    "title": title,
                    "summary": summary,
                    "url": record["url"] or "",
                    "year": record["year"],
                    "court": record["court"] or "",
                    "court_level": record["court_level"] or "",
                    "score": (
                        float(record["score"]) if record["score"] is not None else 0.0
                    ),
                }
                results.append(result_item)
                print(
                    f"  📋 Found: {title[:60]}... "
                    f"(score: {result_item['score']:.4f})"
                )
    except Exception as e:
        print(f"  ❌ Fulltext search failed: {str(e)}")
        return []

    return results


@tool("search_precedents", args_schema=SearchPrecedentsInput)
def search_precedents_tool(
    query: str,
    limit: int = settings.precedents_limit_default,
) -> list[dict]:
    """
    Search for legal precedents using keyword extraction + fulltext search.

    1. Extracts legal keywords from the claim via LLM.
    2. Queries Neo4j fulltext index (precedents_fulltext_idx) with those keywords.
    3. Returns matching precedents sorted by relevance score.
    """
    print(f"🔍 [search_precedents] Claim: '{query[:80]}...', limit: {limit}")

    # Step 1: Extract keywords from the claim
    keywords = _extract_keywords_from_claim(query)
    if not keywords:
        print("  ⚠️ No keywords extracted, returning empty results")
        return []

    # Step 2: Fulltext search with keywords
    results = _search_precedents_by_keywords(keywords, limit=limit)

    if not results:
        print("  ⚠️ No precedents found via keyword search")
        return []

    print(f"  ✅ Total precedents found: {len(results)} [KEYWORD + FULLTEXT]")
    return results
