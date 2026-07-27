"""OpenRouter backend for LexCausa, built on the OpenAI SDK.

The Groq SDK cannot be reused for OpenRouter: it hardcodes the request path
``/openai/v1/chat/completions``, so ``base_url=https://openrouter.ai/api/v1``
yields ``.../api/v1/openai/v1/chat/completions`` (a 404). The OpenAI SDK uses
the relative path ``chat/completions`` and therefore hits the correct
``https://openrouter.ai/api/v1/chat/completions`` endpoint.

Every call is provider-pinned (order preference, e.g. Alibaba) and, for the
reasoning models, keeps the chain-of-thought in the separate ``reasoning`` field
so ``content`` is already the clean final answer. Token usage is recorded like
the Groq path, and ``completion_tokens`` already includes the reasoning tokens.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from config import settings
from services.usage_stats import get_usage_stats

logger = logging.getLogger("lexcausa.openrouter_client")
_usage_stats = get_usage_stats()


def _openai_client(api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=settings.openrouter_base_url, api_key=api_key)


def _record(model: str, usage, source: str) -> None:
    try:
        _usage_stats.record_llm_call(
            provider="openrouter",
            model=model,
            source=source,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            usage=None,
        )
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("usage stats update failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Raw SDK wrapper (for the classifier / aux path via resilient_groq_call).
# Drop-in for groq.Groq: same ``.chat.completions.create`` interface, but hits
# the correct OpenRouter URL and injects the provider-pin/reasoning extra_body.
# ─────────────────────────────────────────────────────────────────────────────
class _ORCompletions:
    def __init__(self, inner):
        self._inner = inner

    def create(self, **kwargs):
        model = str(kwargs.get("model", "") or "")
        model = settings.resolve_openrouter_slug(model)
        kwargs["model"] = model
        merged = dict(settings.openrouter_extra_body(model))
        merged.update(kwargs.get("extra_body") or {})
        if merged:
            kwargs["extra_body"] = merged
        result = self._inner.chat.completions.create(**kwargs)
        if not kwargs.get("stream"):
            _record(model, getattr(result, "usage", None), "openrouter_sdk_call")
        return result


class _ORChat:
    def __init__(self, inner):
        self._inner = inner
        self.completions = _ORCompletions(inner)


class _ORSDKClient:
    def __init__(self, inner):
        self._inner = inner
        self.chat = _ORChat(inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def get_openrouter_sdk_client(api_key: str):
    return _ORSDKClient(_openai_client(api_key))


# ─────────────────────────────────────────────────────────────────────────────
# LangChain chat model (for the agents via resilient_chat_call fast-path).
# ─────────────────────────────────────────────────────────────────────────────
def _to_openai_messages(messages: List[BaseMessage]) -> list[dict]:
    from langchain_core.messages import AIMessage as AI
    from langchain_core.messages import SystemMessage as SM

    out: list[dict] = []
    for m in messages:
        role = (
            "system"
            if isinstance(m, SM)
            else "assistant" if isinstance(m, AI) else "user"
        )
        out.append({"role": role, "content": str(m.content)})
    return out


class ChatOpenRouter(BaseChatModel):
    """LangChain BaseChatModel backed by OpenRouter (OpenAI SDK)."""

    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: str = ""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        slug = settings.resolve_openrouter_slug(self.model)
        client = _openai_client(self.api_key or settings.openrouter_api_key or "")
        extra_body = settings.openrouter_extra_body(slug)
        # Per-call reasoning effort (agents pass reasoning_effort="low" on short
        # JSON utility calls: planners, target maps, gates). Without this, a
        # reasoning model burns the small max_tokens budget on chain-of-thought
        # and returns empty/truncated JSON. Effort replaces the global token cap
        # for this call (OpenRouter treats effort and max_tokens as exclusive).
        effort = str(kwargs.pop("reasoning_effort", "") or "").strip().lower()
        if (
            effort in ("low", "medium", "high")
            and slug in settings.openrouter_reasoning_slugs()
        ):
            extra_body = dict(extra_body)
            extra_body["reasoning"] = {"exclude": False, "effort": effort}
        max_tokens = int(kwargs.get("max_tokens") or self.max_tokens)

        call_kwargs: dict[str, Any] = {
            "model": slug,
            "messages": _to_openai_messages(messages),
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            call_kwargs["extra_body"] = extra_body
        if kwargs.get("response_format") is not None:
            call_kwargs["response_format"] = kwargs["response_format"]
        if stop:
            call_kwargs["stop"] = stop

        retries = max(1, int(settings.groq_max_retries))
        base_delay = float(settings.groq_retry_base_delay)
        last_exc: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                resp = client.chat.completions.create(**call_kwargs)
                content = resp.choices[0].message.content or ""
                _record(slug, getattr(resp, "usage", None), "openrouter_invoke")
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )
            except Exception as exc:
                last_exc = exc
                s = str(exc).lower()
                retryable = any(
                    m in s
                    for m in (
                        "429",
                        "500",
                        "502",
                        "503",
                        "504",
                        "timeout",
                        "overloaded",
                        "rate limit",
                    )
                )
                if attempt < retries and retryable:
                    time.sleep(base_delay * (2 ** (attempt - 1)))
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    def _stream(self, messages, stop=None, **kwargs):
        yield self._generate(messages, stop=stop, **kwargs).generations[0]

    @property
    def _llm_type(self) -> str:
        return "openrouter"
