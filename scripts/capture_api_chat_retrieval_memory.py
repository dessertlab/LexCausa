#!/usr/bin/env python3
"""
Capture /api/chat retrieval outputs for all covered claims and persist them.

Can also warm the backend claim-context memory (SQLite pre-retrieval cache)
because /api/chat reuses the same `prepare_claim_context(...)` path as the
Reasoner/Counter pipeline.

For each claim in claims.md (covered sections), this script calls the local API
endpoint /api/chat with precedents enabled and stores:
- filtered statutes returned by the endpoint
- filtered precedents returned by the endpoint
- classification metadata
- request metadata and status

Outputs:
- logs/api_chat_memory/<timestamp>_<claim_id>_<slug>.json (one per claim)
- logs/api_chat_memory/<timestamp>_manifest.json (aggregate run report)
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAIMS_MD_PATH = PROJECT_ROOT / "claims.md"
OLD_CLAIMS_MD_PATH = PROJECT_ROOT / "old_claims.md"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "api_chat_memory"


@dataclass
class ClaimEntry:
    claim_id: str
    section: str
    domain: str
    title: str
    text: str
    source_file: str


def _slugify_filename(text: str, max_len: int = 60) -> str:
    clean = (text or "").strip().replace("\n", " ")
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in clean)
    safe = "_".join(part for part in safe.split("_") if part)
    return (safe[:max_len] or "claim").strip("_")


def _extract_block(text: str, start_marker: str, end_markers: list[str]) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    end = len(text)
    for marker in end_markers:
        idx = text.find(marker, start)
        if idx >= 0:
            end = min(end, idx)
    return text[start:end].strip()


def _parse_claims_from_block(
    block: str,
    pattern: re.Pattern[str],
    section: str,
    domain: str,
    source_file: str,
) -> list[ClaimEntry]:
    claims: list[ClaimEntry] = []
    matches = list(pattern.finditer(block))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        body = block[start:end].strip()
        if not body:
            continue
        claims.append(
            ClaimEntry(
                claim_id=str(match.group(1)).strip().upper(),
                title=str(match.group(2)).strip(),
                text=body,
                section=section,
                domain=domain,
                source_file=source_file,
            )
        )
    return claims


def parse_claims_md(
    path: Path, *, include_non_covered: bool = False
) -> tuple[list[ClaimEntry], int]:
    text = path.read_text(encoding="utf-8")
    source_file = path.name

    civile_block = _extract_block(
        text,
        "## CLAIM CIVILI (COPERTI)",
        [
            "## CLAIM PENALI (COPERTI)",
            "## CLAIM MIXED (COPERTI)",
            "## CLAIM AMMINISTRATIVI (COPERTI)",
            "## CLAIM NON COPERTI",
        ],
    )
    penale_block = _extract_block(
        text,
        "## CLAIM PENALI (COPERTI)",
        [
            "## CLAIM MIXED (COPERTI)",
            "## CLAIM AMMINISTRATIVI (COPERTI)",
            "## CLAIM NON COPERTI",
        ],
    )
    mixed_block = _extract_block(
        text,
        "## CLAIM MIXED (COPERTI)",
        ["## CLAIM AMMINISTRATIVI (COPERTI)", "## CLAIM NON COPERTI"],
    )
    admin_block = _extract_block(
        text,
        "## CLAIM AMMINISTRATIVI (COPERTI)",
        ["## CLAIM NON COPERTI"],
    )
    non_covered_block = _extract_block(text, "## CLAIM NON COPERTI", [])

    claim_pattern = re.compile(r"(?m)^\s*###\s*([A-Z]\d+)\s*-\s*(.+?)\s*$")

    claims = (
        _parse_claims_from_block(
            civile_block,
            claim_pattern,
            "CLAIM CIVILI (COPERTI)",
            "civile",
            source_file,
        )
        + _parse_claims_from_block(
            penale_block,
            claim_pattern,
            "CLAIM PENALI (COPERTI)",
            "penale",
            source_file,
        )
        + _parse_claims_from_block(
            mixed_block,
            claim_pattern,
            "CLAIM MIXED (COPERTI)",
            "misto",
            source_file,
        )
        + _parse_claims_from_block(
            admin_block,
            claim_pattern,
            "CLAIM AMMINISTRATIVI (COPERTI)",
            "amministrativo",
            source_file,
        )
    )

    if include_non_covered:
        claims += _parse_claims_from_block(
            non_covered_block,
            claim_pattern,
            "CLAIM NON COPERTI (GAP NORMATIVI)",
            "non_coperto",
            source_file,
        )

    non_covered_count = len(claim_pattern.findall(non_covered_block))
    return claims, non_covered_count


def load_claims_from_files(
    paths: list[Path], *, include_non_covered: bool = False
) -> tuple[list[ClaimEntry], dict[str, int]]:
    all_claims: list[ClaimEntry] = []
    non_covered_counts: dict[str, int] = {}
    for path in paths:
        claims, non_covered_count = parse_claims_md(
            path,
            include_non_covered=include_non_covered,
        )
        all_claims.extend(claims)
        non_covered_counts[path.name] = non_covered_count
    return all_claims, non_covered_counts


def _healthcheck(base_url: str, timeout_s: float = 10.0) -> None:
    url = f"{base_url.rstrip('/')}/health"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()


def _call_api_chat(
    base_url: str,
    claim: str,
    top_k: int,
    max_precedents: int,
    timeout_s: float | None,
    settings_payload: dict[str, Any] | None = None,
    claim_context_memory_enabled: bool = False,
    claim_context_memory_overwrite: bool = False,
) -> requests.Response:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {
        "message": claim,
        "top_k": top_k,
        "include_precedents": True,
        "max_precedents": max_precedents,
    }
    if settings_payload:
        payload["settings"] = settings_payload
    if claim_context_memory_enabled:
        payload["claim_context_memory_enabled"] = True
    if claim_context_memory_overwrite:
        payload["claim_context_memory_overwrite"] = True
    return requests.post(url, json=payload, timeout=timeout_s)


def _parse_timeout_arg(value: str) -> float | None:
    raw = str(value).strip().lower()
    if raw in {"none", "no", "null", "inf", "infinite"}:
        return None
    parsed = float(raw)
    if parsed <= 0:
        return None
    return parsed


def _claim_entry_key(entry: ClaimEntry) -> str:
    return f"{Path(entry.source_file).stem}:{entry.claim_id.upper()}"


def _response_text_safe(resp: requests.Response) -> str:
    try:
        return resp.text or ""
    except Exception:
        return ""


def _wipe_claim_context_memory_db() -> int:
    db_path = PROJECT_ROOT / "cache" / "claim_context_cache.sqlite"
    if not db_path.exists():
        return 0
    with sqlite3.connect(str(db_path), timeout=10.0) as conn:
        try:
            deleted = int(
                conn.execute("SELECT COUNT(*) FROM claim_context_cache").fetchone()[0]
            )
        except sqlite3.OperationalError:
            return 0
        conn.execute("DELETE FROM claim_context_cache")
        conn.commit()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except sqlite3.OperationalError:
            pass
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--claims-files",
        nargs="*",
        default=None,
        help="Claim markdown files to process (default: claims.md old_claims.md if present).",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-precedents", type=int, default=5)
    parser.add_argument(
        "--search-min-kept-statutes",
        type=int,
        default=8,
        help="Forwarded to /api/chat settings for deterministic claim-context cache signature.",
    )
    parser.add_argument(
        "--search-use-top-n-libri",
        type=int,
        default=3,
        help="Forwarded to /api/chat settings for deterministic retrieval behavior/signature.",
    )
    parser.add_argument(
        "--timeout",
        type=_parse_timeout_arg,
        default=None,
        help="HTTP timeout seconds. Use 'none' (default) to disable timeout.",
    )
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument(
        "--domains",
        nargs="*",
        default=[],
        help="Optional subset: penale civile amministrativo misto non_coperto",
    )
    parser.add_argument(
        "--include-non-covered",
        action="store_true",
        help="Include NC* claims from 'CLAIM NON COPERTI' sections (default: covered sections only).",
    )
    parser.add_argument(
        "--search-query-terms-mode",
        default="llm",
        choices=["llm"],
        help="Forwarded to /api/chat settings. Kept explicit for reproducibility.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip claims that already have a saved file in logs/api_chat_memory/*_<CLAIM_ID>_*.json",
    )
    parser.add_argument(
        "--claim-context-memory",
        action="store_true",
        help="Enable backend SQLite claim-context memory while calling /api/chat (warm cache).",
    )
    parser.add_argument(
        "--overwrite-claim-context-memory",
        action="store_true",
        help="Force overwrite of existing claim-context memory entries (implies --claim-context-memory).",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Warm backend claim-context memory but skip writing per-claim JSON capture files.",
    )
    parser.add_argument(
        "--retrieval-model-order-aliases",
        nargs="+",
        default=None,
        help="Override retrieval model alias order for /api/chat request settings (e.g. qwen_qwen3_32b).",
    )
    parser.add_argument(
        "--wipe-claim-context-memory",
        action="store_true",
        help="Delete all rows from cache/claim_context_cache.sqlite before running.",
    )
    args = parser.parse_args()

    if args.overwrite_claim_context_memory:
        args.claim_context_memory = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    claim_files_raw = list(args.claims_files or [])
    if not claim_files_raw:
        claim_files_raw = [str(CLAIMS_MD_PATH.name)]
        if OLD_CLAIMS_MD_PATH.exists():
            claim_files_raw.append(str(OLD_CLAIMS_MD_PATH.name))
    claim_paths: list[Path] = []
    for raw in claim_files_raw:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            print(f"Claims file not found: {p}")
            return 1
        claim_paths.append(p)

    claims, non_covered_counts = load_claims_from_files(
        claim_paths,
        include_non_covered=bool(args.include_non_covered),
    )
    domain_filter = {d.strip().lower() for d in args.domains if d.strip()}
    if domain_filter:
        claims = [c for c in claims if c.domain in domain_filter]

    if not claims:
        print("No claims selected.")
        return 1

    try:
        _healthcheck(args.base_url)
    except Exception as exc:
        print(f"Healthcheck failed on {args.base_url}: {exc}")
        return 2

    if args.wipe_claim_context_memory:
        deleted = _wipe_claim_context_memory_db()
        print(f"Wiped claim-context memory rows: {deleted}")

    settings_payload: dict[str, Any] = {
        "search_query_terms_mode": args.search_query_terms_mode,
        "search_min_kept_statutes": int(args.search_min_kept_statutes),
        "search_use_top_n_libri": int(args.search_use_top_n_libri),
    }
    if args.retrieval_model_order_aliases:
        settings_payload["retrieval_model_order_aliases"] = list(
            args.retrieval_model_order_aliases
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = OUTPUT_DIR / f"{ts}_manifest.json"

    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "claims_paths": [str(p) for p in claim_paths],
        "non_covered_excluded": non_covered_counts,
        "request": {
            "top_k": int(args.top_k),
            "include_precedents": True,
            "max_precedents": int(args.max_precedents),
            "settings": settings_payload,
            "timeout": None if args.timeout is None else float(args.timeout),
            "delay": float(args.delay),
            "claim_context_memory_enabled": bool(args.claim_context_memory),
            "claim_context_memory_overwrite": bool(args.overwrite_claim_context_memory),
            "cache_only": bool(args.cache_only),
            "wipe_claim_context_memory": bool(args.wipe_claim_context_memory),
            "include_non_covered": bool(args.include_non_covered),
        },
        "counts": {"selected_claims": len(claims)},
        "results": [],
    }

    ok = 0
    failed = 0
    existing_claim_keys: set[str] = set()
    if args.skip_existing:
        for p in OUTPUT_DIR.glob("*.json"):
            if p.name.endswith("_manifest.json"):
                continue
            m_new = re.search(
                r"^\d{8}_\d{6}_([a-zA-Z0-9_]+)_([A-Z]\d+)_",
                p.name,
            )
            if m_new:
                existing_claim_keys.add(f"{m_new.group(1)}:{m_new.group(2).upper()}")
                continue
        manifest["resume"] = {
            "skip_existing": True,
            "existing_claim_keys_detected": sorted(existing_claim_keys),
        }

    for idx, entry in enumerate(claims, start=1):
        entry_key = _claim_entry_key(entry)
        if entry_key in existing_claim_keys:
            skipped_record = {
                "source_file": entry.source_file,
                "claim_id": entry.claim_id,
                "domain": entry.domain,
                "title": entry.title,
                "status": "skipped_existing",
            }
            manifest["results"].append(skipped_record)
            manifest["counts"]["skipped_existing"] = (
                manifest["counts"].get("skipped_existing", 0) + 1
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[{idx}/{len(claims)}] {entry.source_file}:{entry.claim_id} ({entry.domain}) - skipped existing",
                flush=True,
            )
            continue

        print(
            f"[{idx}/{len(claims)}] {entry.source_file}:{entry.claim_id} ({entry.domain}) - {entry.title}",
            flush=True,
        )
        started = time.time()
        record: dict[str, Any] = {
            "source_file": entry.source_file,
            "claim_id": entry.claim_id,
            "domain": entry.domain,
            "title": entry.title,
            "claim": entry.text,
            "status": "error",
        }
        try:
            while True:
                resp = _call_api_chat(
                    base_url=args.base_url,
                    claim=entry.text,
                    top_k=int(args.top_k),
                    max_precedents=int(args.max_precedents),
                    timeout_s=None if args.timeout is None else float(args.timeout),
                    settings_payload=settings_payload,
                    claim_context_memory_enabled=bool(args.claim_context_memory),
                    claim_context_memory_overwrite=bool(
                        args.overwrite_claim_context_memory
                    ),
                )
                if resp.status_code == 200:
                    break
                break
            elapsed = round(time.time() - started, 3)
            record["elapsed_s"] = elapsed
            record["http_status"] = resp.status_code

            if resp.status_code != 200:
                text = _response_text_safe(resp)
                record["error"] = text[:4000]
                failed += 1
            else:
                payload = resp.json()
                statutes = payload.get("articles") if isinstance(payload, dict) else []
                precedents = (
                    payload.get("precedents") if isinstance(payload, dict) else []
                )
                if not isinstance(statutes, list):
                    statutes = []
                if not isinstance(precedents, list):
                    precedents = []

                record.update(
                    {
                        "status": "ok",
                        "articles_count": len(statutes),
                        "precedents_count": len(precedents),
                    }
                )
                if not args.cache_only:
                    source_stem = _slugify_filename(
                        Path(entry.source_file).stem, max_len=20
                    )
                    claim_file = OUTPUT_DIR / (
                        f"{ts}_{source_stem}_{entry.claim_id}_{_slugify_filename(entry.title, max_len=40)}.json"
                    )
                    claim_payload = {
                        "captured_at": datetime.now().isoformat(timespec="seconds"),
                        "claim_meta": asdict(entry),
                        "request": manifest["request"],
                        "http_status": resp.status_code,
                        "classification": payload.get("classification"),
                        "articles": statutes,
                        "precedents": precedents,
                        "counts": {
                            "articles": len(statutes),
                            "precedents": len(precedents),
                        },
                    }
                    claim_file.write_text(
                        json.dumps(claim_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    record["file"] = {
                        "absolute_path": str(claim_file),
                        "relative_path": str(claim_file.relative_to(PROJECT_ROOT)),
                    }
                ok += 1

        except Exception as exc:
            record["elapsed_s"] = round(time.time() - started, 3)
            record["error"] = str(exc)
            failed += 1

        manifest["results"].append(record)
        manifest["counts"]["ok"] = ok
        manifest["counts"]["failed"] = failed

        # Incremental checkpoint after each claim.
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if args.delay > 0 and idx < len(claims):
            time.sleep(float(args.delay))

    print(f"\nSaved manifest: {manifest_path}")
    print(
        "OK={} | FAILED={} | SKIPPED={}".format(
            ok, failed, manifest["counts"].get("skipped_existing", 0)
        )
    )
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
