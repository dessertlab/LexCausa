"""SQLite-backed cache for pre-retrieved claim context.

Stores the final pre-retrieval outputs (applicable statutes + applicable precedents)
so repeated runs on the same claim can skip retrieval/filtering.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB_PATH = _PROJECT_ROOT / "cache" / "claim_context_cache.sqlite"
_SCHEMA_VERSION = "claim_context_cache_v1"


class ClaimContextMemory:
    """Small SQLite cache keyed by claim + retrieval signature."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = Path(db_path or _DEFAULT_DB_PATH)
        self._init_lock = threading.Lock()
        self._initialized = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS claim_context_cache (
                        cache_key TEXT PRIMARY KEY,
                        claim_text TEXT NOT NULL,
                        request_meta_json TEXT NOT NULL,
                        statutes_json TEXT NOT NULL,
                        precedents_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        hit_count INTEGER NOT NULL DEFAULT 0,
                        last_hit_at TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_claim_context_cache_updated_at
                    ON claim_context_cache(updated_at)
                    """
                )
                conn.commit()
            self._initialized = True

    @staticmethod
    def _normalize_claim(claim: str) -> str:
        return " ".join((claim or "").strip().split())

    def _build_request_meta(
        self,
        *,
        claim: str,
        include_precedents: bool,
        max_statutes: int,
        max_precedents: int,
        signature: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "claim_normalized": self._normalize_claim(claim),
            "include_precedents": bool(include_precedents),
            "max_statutes": int(max_statutes),
            "max_precedents": int(max_precedents),
            "signature": dict(signature or {}),
        }

    def build_cache_key(
        self,
        *,
        claim: str,
        include_precedents: bool,
        max_statutes: int,
        max_precedents: int,
        signature: dict[str, Any],
    ) -> str:
        request_meta = self._build_request_meta(
            claim=claim,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
            signature=signature,
        )
        canonical = json.dumps(request_meta, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(
        self,
        *,
        claim: str,
        include_precedents: bool,
        max_statutes: int,
        max_precedents: int,
        signature: dict[str, Any],
    ) -> dict[str, Any] | None:
        self._ensure_initialized()
        cache_key = self.build_cache_key(
            claim=claim,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
            signature=signature,
        )
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT cache_key, statutes_json, precedents_json, request_meta_json
                FROM claim_context_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            row_cache_key = str(row["cache_key"])
            now = datetime.now().isoformat(timespec="seconds")
            conn.execute(
                """
                UPDATE claim_context_cache
                SET hit_count = hit_count + 1, last_hit_at = ?
                WHERE cache_key = ?
                """,
                (now, row_cache_key),
            )
            conn.commit()

        try:
            statutes = json.loads(row["statutes_json"])
            precedents = json.loads(row["precedents_json"])
            request_meta = json.loads(row["request_meta_json"])
        except Exception:
            return None

        if not isinstance(statutes, list) or not isinstance(precedents, list):
            return None

        return {
            "cache_key": row_cache_key,
            "statutes": statutes,
            "precedents": precedents,
            "request_meta": request_meta if isinstance(request_meta, dict) else {},
        }

    def put(
        self,
        *,
        claim: str,
        include_precedents: bool,
        max_statutes: int,
        max_precedents: int,
        signature: dict[str, Any],
        statutes: list[dict],
        precedents: list[dict],
    ) -> str:
        self._ensure_initialized()
        request_meta = self._build_request_meta(
            claim=claim,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
            signature=signature,
        )
        cache_key = self.build_cache_key(
            claim=claim,
            include_precedents=include_precedents,
            max_statutes=max_statutes,
            max_precedents=max_precedents,
            signature=signature,
        )
        now = datetime.now().isoformat(timespec="seconds")
        statutes_json = json.dumps(statutes or [], ensure_ascii=False)
        precedents_json = json.dumps(precedents or [], ensure_ascii=False)
        request_meta_json = json.dumps(request_meta, ensure_ascii=False, sort_keys=True)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO claim_context_cache (
                    cache_key,
                    claim_text,
                    request_meta_json,
                    statutes_json,
                    precedents_json,
                    created_at,
                    updated_at,
                    hit_count,
                    last_hit_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)
                ON CONFLICT(cache_key) DO UPDATE SET
                    claim_text = excluded.claim_text,
                    request_meta_json = excluded.request_meta_json,
                    statutes_json = excluded.statutes_json,
                    precedents_json = excluded.precedents_json,
                    updated_at = excluded.updated_at
                """,
                (
                    cache_key,
                    str(claim or ""),
                    request_meta_json,
                    statutes_json,
                    precedents_json,
                    now,
                    now,
                ),
            )
            conn.commit()
        return cache_key


@lru_cache(maxsize=1)
def get_claim_context_memory() -> ClaimContextMemory:
    return ClaimContextMemory()
