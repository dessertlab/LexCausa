"""Runtime usage statistics for API calls and LLM token accounting.

This module provides a process-local, thread-safe collector used by the API
server and Groq client wrappers.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    """Return UTC timestamp in ISO format with stable 'Z' suffix."""
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _as_int(value: Any) -> int | None:
    """Best-effort conversion to non-negative int."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _to_mapping(payload: Any) -> dict[str, Any] | None:
    """Convert supported payload shapes into a plain dictionary."""
    if payload is None:
        return None

    if isinstance(payload, dict):
        return payload

    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    as_dict = getattr(payload, "dict", None)
    if callable(as_dict):
        try:
            dumped = as_dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    # Last-resort attribute probing for common response/message objects.
    probed: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "token_usage",
        "usage",
        "usage_metadata",
        "response_metadata",
        "model",
        "model_name",
    ):
        if hasattr(payload, key):
            try:
                probed[key] = getattr(payload, key)
            except Exception:
                continue

    return probed or None


def extract_token_usage(payload: Any) -> tuple[int | None, int | None, int | None]:
    """Extract prompt/completion/total token usage from mixed payload shapes.

    Supports both OpenAI-style and LangChain-style metadata fields.
    """
    queue: list[Any] = [payload]
    seen_ids: set[int] = set()

    while queue:
        current = queue.pop(0)
        if current is None:
            continue

        current_id = id(current)
        if current_id in seen_ids:
            continue
        seen_ids.add(current_id)

        if isinstance(current, (list, tuple)):
            queue.extend(list(current))
            continue

        mapping = _to_mapping(current)
        if not mapping:
            continue

        prompt_tokens = _as_int(mapping.get("prompt_tokens"))
        if prompt_tokens is None:
            prompt_tokens = _as_int(mapping.get("input_tokens"))

        completion_tokens = _as_int(mapping.get("completion_tokens"))
        if completion_tokens is None:
            completion_tokens = _as_int(mapping.get("output_tokens"))

        total_tokens = _as_int(mapping.get("total_tokens"))
        if total_tokens is None:
            total_tokens = _as_int(mapping.get("total"))

        if (
            total_tokens is None
            and prompt_tokens is not None
            and completion_tokens is not None
        ):
            total_tokens = prompt_tokens + completion_tokens

        if (
            prompt_tokens is not None
            or completion_tokens is not None
            or total_tokens is not None
        ):
            return prompt_tokens, completion_tokens, total_tokens

        for nested_key in (
            "usage",
            "token_usage",
            "usage_metadata",
            "response_metadata",
            "metadata",
        ):
            nested = mapping.get(nested_key)
            if nested is not None:
                queue.append(nested)

    return None, None, None


class UsageStats:
    """Thread-safe process-local usage statistics collector."""

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self._stats_dir = project_root / "logs" / "usage_stats"
        self._stats_dir.mkdir(parents=True, exist_ok=True)

        self._session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        self._session_path = self._stats_dir / f"{self._session_id}.json"
        self._latest_path = self._stats_dir / "latest.json"

        self._lock = threading.Lock()
        self._stats = self._new_snapshot()

        # Persist a bootstrap snapshot so the file always exists.
        with self._lock:
            self._persist_locked()

        atexit.register(self._on_exit)

    def _new_snapshot(self) -> dict[str, Any]:
        timestamp = _utc_now_iso()
        return {
            "session_id": self._session_id,
            "started_at": timestamp,
            "updated_at": timestamp,
            "api": {
                "calls_total": 0,
                "by_endpoint": {},
            },
            "llm": {
                "calls_total": 0,
                "calls_with_usage": 0,
                "calls_without_usage": 0,
                "by_provider": {},
                "by_model": {},
                "by_source": {},
                "tokens": {
                    "prompt": 0,
                    "completion": 0,
                    "total": 0,
                    "max_prompt_per_call": 0,
                },
            },
        }

    @staticmethod
    def _bump(counter: dict[str, int], key: str, delta: int = 1) -> None:
        counter[key] = int(counter.get(key, 0)) + int(delta)

    def _persist_locked(self) -> None:
        self._stats["updated_at"] = _utc_now_iso()
        payload = json.dumps(self._stats, ensure_ascii=False, indent=2, sort_keys=True)

        for target in (self._session_path, self._latest_path):
            tmp_path = target.with_suffix(f"{target.suffix}.tmp")
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(target)

    def record_api_call(self, endpoint: str, method: str | None = None) -> None:
        endpoint_norm = str(endpoint or "").strip() or "unknown"
        method_norm = str(method or "").strip().upper()
        key = f"{method_norm} {endpoint_norm}".strip()

        with self._lock:
            api_stats = self._stats["api"]
            api_stats["calls_total"] = int(api_stats.get("calls_total", 0)) + 1
            by_endpoint = api_stats.get("by_endpoint", {})
            self._bump(by_endpoint, key)
            api_stats["by_endpoint"] = by_endpoint
            self._persist_locked()

    def record_llm_call(
        self,
        *,
        provider: str,
        model: str,
        source: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        usage: Any = None,
    ) -> None:
        usage_prompt, usage_completion, usage_total = extract_token_usage(usage)

        prompt = _as_int(prompt_tokens)
        completion = _as_int(completion_tokens)
        total = _as_int(total_tokens)

        if prompt is None:
            prompt = usage_prompt
        if completion is None:
            completion = usage_completion
        if total is None:
            total = usage_total
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion

        provider_key = str(provider or "unknown").strip() or "unknown"
        model_key = str(model or "unknown").strip() or "unknown"
        source_key = str(source or "unknown").strip() or "unknown"

        with self._lock:
            llm_stats = self._stats["llm"]
            llm_stats["calls_total"] = int(llm_stats.get("calls_total", 0)) + 1
            self._bump(llm_stats.get("by_provider", {}), provider_key)
            self._bump(llm_stats.get("by_model", {}), model_key)
            self._bump(llm_stats.get("by_source", {}), source_key)

            has_usage = (
                prompt is not None or completion is not None or total is not None
            )
            if has_usage:
                llm_stats["calls_with_usage"] = (
                    int(llm_stats.get("calls_with_usage", 0)) + 1
                )
            else:
                llm_stats["calls_without_usage"] = (
                    int(llm_stats.get("calls_without_usage", 0)) + 1
                )

            token_stats = llm_stats.get("tokens", {})
            if prompt is not None:
                token_stats["prompt"] = int(token_stats.get("prompt", 0)) + prompt
                token_stats["max_prompt_per_call"] = max(
                    int(token_stats.get("max_prompt_per_call", 0)), prompt
                )
            if completion is not None:
                token_stats["completion"] = (
                    int(token_stats.get("completion", 0)) + completion
                )
            if total is not None:
                token_stats["total"] = int(token_stats.get("total", 0)) + total
            llm_stats["tokens"] = token_stats

            self._persist_locked()

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._stats)

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._stats = self._new_snapshot()
            self._persist_locked()
            return deepcopy(self._stats)

    def _on_exit(self) -> None:
        try:
            with self._lock:
                self._persist_locked()
                api_calls = int(self._stats.get("api", {}).get("calls_total", 0))
                llm_calls = int(self._stats.get("llm", {}).get("calls_total", 0))
                total_tokens = int(
                    self._stats.get("llm", {}).get("tokens", {}).get("total", 0)
                )
            print(
                "[UsageStats] session="
                f"{self._session_id} api_calls={api_calls} llm_calls={llm_calls} "
                f"total_tokens={total_tokens} file={self._session_path}"
            )
        except Exception:
            # Never fail interpreter shutdown for telemetry.
            return


_USAGE_STATS = UsageStats()


def get_usage_stats() -> UsageStats:
    """Return the process singleton usage stats collector."""
    return _USAGE_STATS
