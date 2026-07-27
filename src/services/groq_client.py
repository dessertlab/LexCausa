"""
Resilient Groq Client for LexCausa.

Provides automatic retry, API key rotation, and model fallback for both:
- groq.Groq (raw SDK, used by ClaimClassifier)
- langchain_groq.ChatGroq (LangChain wrapper, used by agents)

Error-handling strategy:
- Model over-capacity (503 "over capacity") → switch to fallback MODEL immediately
- Rate limit (429 "too many requests")      → rotate API KEY, keep same model
- Transient errors (500, 502, 504, timeout) → retry with backoff, then rotate key
"""

import logging
import re
import threading
import time
from typing import Callable, Optional

from groq import Groq
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq

from config import settings
from services.pipeline_control import PipelineCancelled
from services.usage_stats import extract_token_usage, get_usage_stats

# Strip <think>…</think> blocks emitted by reasoning models (DeepSeek-R1, etc.)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).lstrip()


logger = logging.getLogger("lexcausa.groq_client")
_usage_stats = get_usage_stats()


def _record_llm_stats(
    *,
    provider: str,
    model: str,
    source: str,
    usage_payload=None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
) -> None:
    """Best-effort stats update that never breaks runtime calls."""
    try:
        _usage_stats.record_llm_call(
            provider=provider,
            model=model,
            source=source,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            usage=usage_payload,
        )
    except Exception as exc:
        logger.debug("Usage stats update failed: %s", exc)


class AllKeysRateLimitedError(Exception):
    """Raised when every available API key is rate-limited."""

    def __init__(self, n_keys: int, model: str):
        self.n_keys = n_keys
        self.model = model
        super().__init__(
            f"All {n_keys} Groq API keys are rate-limited for model '{model}'. "
            f"Wait a few minutes or add more GROQ_API_KEY_V* entries to .env."
        )


class RequestTooLargeError(Exception):
    """Raised when the provider rejects a request for exceeding token budget."""

    def __init__(self, model: str, detail: str):
        self.model = model
        self.detail = detail
        super().__init__(
            f"Request too large for model '{model}'. Reduce prompt size or max_tokens. "
            f"Provider detail: {detail}"
        )


def shrink_max_tokens_progressive(current_tokens: int, min_tokens: int) -> int:
    """Reduce max_tokens by 10% with a hard floor and at least 1-token decrement.

    This helper centralizes oversized-request token reduction policy so callers
    can apply the same behavior consistently.
    """
    current = max(1, int(current_tokens))
    floor = max(1, int(min_tokens))
    if current <= floor:
        return floor
    decrement = max(1, current // 10)
    return max(floor, current - decrement)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level state (shared across all callers in the same process)
# ─────────────────────────────────────────────────────────────────────────────
_key_lock = threading.Lock()
_current_key_index: int = 0

# Model-down cache: remembers which models are down so we don't retry them
# on every single call.  Maps model_name → timestamp when it was marked down.
_MODEL_DOWN_TTL: float = settings.model_down_ttl
_model_down_cache: dict[str, float] = {}
_model_lock = threading.Lock()


def _mark_model_down(model: str) -> None:
    """Record that *model* is currently unavailable."""
    with _model_lock:
        _model_down_cache[model] = time.time()
        logger.warning(
            "📛 [groq_client] Model %s marked as DOWN for %ds",
            model,
            int(_MODEL_DOWN_TTL),
        )


def _is_model_cached_down(model: str) -> bool:
    """Return True if *model* was recently marked as down (within TTL)."""
    with _model_lock:
        ts = _model_down_cache.get(model)
        if ts is None:
            return False
        if time.time() - ts > _MODEL_DOWN_TTL:
            # TTL expired → give the model another chance
            del _model_down_cache[model]
            logger.info("🔄 [groq_client] Model %s TTL expired, will retry it", model)
            return False
        return True


def _llm_keys() -> list[str]:
    """API key(s) for the active backend (Groq Cloud or OpenRouter)."""
    if settings.llm_backend == "openrouter":
        keys = settings.openrouter_api_keys
        if not keys:
            raise ValueError(
                "No OpenRouter API key configured. Set OPENROUTER_API_KEY in .env "
                "(required for LLM_BACKEND=openrouter)."
            )
        return keys
    return settings.groq_api_keys


def _build_chat_llm(key: str, model: str, temperature, max_tokens):
    """Construct a ChatGroq bound to the active backend.

    For OpenRouter, the same ChatGroq is pointed at the OpenRouter base URL and
    carries the provider-pin / reasoning flags via extra_body; the model id is
    sanitized to a valid OpenRouter slug (aux calls fall back to the aux model).
    """
    if settings.llm_backend == "openrouter":
        from services.openrouter_client import ChatOpenRouter

        return ChatOpenRouter(
            model=settings.resolve_openrouter_slug(model),
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=key,
        )
    return ChatGroq(
        api_key=key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _build_sdk_client(key: str):
    """Raw SDK client for the active backend.

    OpenRouter uses the OpenAI SDK (the Groq SDK hardcodes an incompatible
    ``/openai/v1/chat/completions`` path); every call is provider-pinned.
    """
    if settings.llm_backend == "openrouter":
        from services.openrouter_client import get_openrouter_sdk_client

        return get_openrouter_sdk_client(key)
    return Groq(api_key=key)


def _rotate_key() -> tuple[str, int]:
    """
    Rotate to the next available API key (thread-safe).

    Returns:
        (new_api_key, new_index)
    """
    global _current_key_index
    keys = _llm_keys()
    if not keys:
        raise ValueError(
            "No API keys configured. Set GROQ_API_KEY_V1 (or OPENROUTER_API_KEY) in .env"
        )
    with _key_lock:
        _current_key_index = (_current_key_index + 1) % len(keys)
        return keys[_current_key_index], _current_key_index


def _current_key() -> str:
    """Get the current API key."""
    keys = _llm_keys()
    if not keys:
        raise ValueError(
            "No API keys configured. Set GROQ_API_KEY_V1 (or OPENROUTER_API_KEY) in .env"
        )
    with _key_lock:
        idx = _current_key_index % len(keys)
        return keys[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Error classification
# ─────────────────────────────────────────────────────────────────────────────


def _is_model_down(exc: Exception) -> bool:
    """True when the error is about the MODEL being unavailable / over capacity."""
    exc_str = str(exc).lower()
    return any(
        marker in exc_str
        for marker in ("over capacity", "currently unavailable", "model not available")
    )


def _is_rate_limit(exc: Exception) -> bool:
    """True when the error is a per-key rate-limit (429)."""
    exc_str = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    return any(
        marker in exc_str
        for marker in ("rate limit", "rate_limit", "too many requests")
    )


def _is_request_too_large(exc: Exception) -> bool:
    """True when provider rejects the request because it exceeds token budget."""
    exc_str = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if status == 413:
        return True
    # Keep 429/rate-limit semantics separate from oversized 413 handling.
    if status == 429 or _is_rate_limit(exc):
        return False
    return any(
        marker in exc_str
        for marker in (
            "request too large",
            "please reduce your message size",
        )
    )


def _is_retryable(exc: Exception) -> bool:
    """True for any transient/retryable error (includes model-down and rate-limit)."""
    if _is_model_down(exc) or _is_rate_limit(exc):
        return True
    exc_str = str(exc).lower()
    retryable_codes = ("503", "429", "500", "502", "504")
    retryable_messages = (
        "timeout",
        "timed out",
        "connection error",
        "service unavailable",
        "internal_server_error",
    )
    for code in retryable_codes:
        if code in exc_str:
            return True
    for msg in retryable_messages:
        if msg in exc_str:
            return True
    status = getattr(exc, "status_code", None)
    if status and status in (429, 500, 502, 503, 504):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Core resilient loop (shared logic for all three wrappers)
# ─────────────────────────────────────────────────────────────────────────────


def _resilient_loop(
    execute_fn,
    *,
    max_retries: Optional[int] = None,
    label: str = "Groq",
    model_order: Optional[list[str]] = None,
    cancel_checker: Optional[Callable[[], bool]] = None,
):
    """
    Generic resilient execution loop.

    Args:
        execute_fn: ``(key: str, model: str) -> result``
                    Called for each attempt.
        max_retries: Per-combination retry cap.
        label: Label for log messages.

    Strategy:
    - On *model-down* → skip remaining retries/keys for this model, jump to
      fallback model immediately.
    - On *rate-limit* → rotate API key, keep the same model.  If **every**
      key is rate-limited, raise ``AllKeysRateLimitedError`` immediately
      instead of looping forever.
    - On other transient errors → retry with exponential backoff, then rotate key.
    """

    def _check_cancel() -> None:
        if cancel_checker is not None and cancel_checker():
            raise PipelineCancelled("Execution manually stopped.")

    def _sleep_with_cancel(delay_s: float) -> None:
        remaining = max(0.0, float(delay_s))
        while remaining > 0:
            _check_cancel()
            chunk = 0.2 if remaining > 0.2 else remaining
            time.sleep(chunk)
            remaining -= chunk

    retries = max_retries or settings.groq_max_retries
    # Allow per-call model order overrides (used by Reasoner/CounterReasoner).
    if model_order:
        models: list[str] = []
        for m in model_order:
            m_norm = str(m or "").strip()
            if m_norm and m_norm not in models:
                models.append(m_norm)
    else:
        models = []
    if not models:
        models = list(settings.groq_models)  # copy so we can iterate safely
    keys = _llm_keys()
    n_keys = len(keys)
    base_delay = settings.groq_retry_base_delay

    if not keys:
        raise ValueError(
            "No API keys configured. "
            "Set GROQ_API_KEY_V1 (or OPENROUTER_API_KEY for LLM_BACKEND=openrouter) in .env"
        )

    logger.info(
        "🔑 [%s] %d API key(s) available, %d model(s) configured",
        label,
        n_keys,
        len(models),
    )

    last_exc: Optional[Exception] = None
    model_idx = 0

    # ── Skip models already known to be down ─────────────────────────────
    while model_idx < len(models) and _is_model_cached_down(models[model_idx]):
        _check_cancel()
        logger.info(
            "⏩ [%s] Skipping model %s (cached as down)",
            label,
            models[model_idx],
        )
        model_idx += 1

    if model_idx >= len(models):
        # Every model is cached-down – reset and try anyway
        logger.warning(
            "⚠️ [%s] All models cached as down, retrying from first model",
            label,
        )
        model_idx = 0

    while model_idx < len(models):
        _check_cancel()
        model = models[model_idx]
        # Track which keys have been rate-limited for THIS model
        rate_limited_keys: set[int] = set()
        keys_tried = 0

        while keys_tried < n_keys:
            _check_cancel()
            key = _current_key()
            cur_idx = _current_key_index

            for attempt in range(1, retries + 1):
                _check_cancel()
                try:
                    return execute_fn(key, model)
                except Exception as exc:
                    last_exc = exc
                    if isinstance(exc, PipelineCancelled):
                        raise

                    if _is_request_too_large(exc):
                        # Keep this at debug level to avoid duplicate user-facing
                        # logs (provider + caller) for the same oversized error.
                        logger.debug(
                            "[%s] Request too large for model %s: %s",
                            label,
                            model,
                            exc,
                        )
                        raise RequestTooLargeError(model, str(exc)) from exc

                    if not _is_retryable(exc):
                        raise  # Non-retryable (e.g. 400 Bad Request) → fail fast

                    # ── Model down → switch model immediately ────────────
                    if _is_model_down(exc):
                        _mark_model_down(model)
                        logger.warning(
                            "🔀 [%s] Model %s is down, switching to fallback "
                            "immediately: %s",
                            label,
                            model,
                            exc,
                        )
                        model_idx += 1
                        if model_idx < len(models):
                            logger.warning(
                                "🔀 [%s] Fallback model: %s",
                                label,
                                models[model_idx],
                            )
                        # break out of both the attempt and keys loops
                        keys_tried = n_keys  # exit keys loop after break
                        break  # exit attempt loop

                    # ── Rate limit → rotate key immediately ──────────────
                    if _is_rate_limit(exc):
                        rate_limited_keys.add(cur_idx)
                        logger.warning(
                            "🔄 [%s] Rate limit hit (key %d/%d), "
                            "rotating API key: %s",
                            label,
                            cur_idx + 1,
                            n_keys,
                            exc,
                        )
                        # Check if ALL keys are now rate-limited
                        if len(rate_limited_keys) >= n_keys:
                            logger.error(
                                "🚫 [%s] All %d API keys are rate-limited "
                                "for model '%s'. Stopping.",
                                label,
                                n_keys,
                                model,
                            )
                            raise AllKeysRateLimitedError(n_keys, model) from exc
                        _rotate_key()
                        keys_tried += 1
                        break  # exit attempt loop, go to next key

                    # ── Other transient error → backoff + retry ──────────
                    logger.warning(
                        "⚠️ [%s] Transient error (model=%s, key %d/%d, "
                        "attempt=%d/%d): %s",
                        label,
                        model,
                        cur_idx + 1,
                        n_keys,
                        attempt,
                        retries,
                        exc,
                    )
                    if attempt < retries:
                        delay = base_delay * (2 ** (attempt - 1))
                        _sleep_with_cancel(delay)
            else:
                # All retry attempts exhausted for this key (no break) → rotate key
                logger.warning(
                    "🔄 [%s] Retries exhausted for key %d/%d, rotating key",
                    label,
                    _current_key_index + 1,
                    n_keys,
                )
                _rotate_key()
                keys_tried += 1
                continue

            # We broke out of the attempt loop (rate-limit or model-down)
            continue

        # If we got here for model-down, model_idx was already incremented above
        if model_idx >= len(models):
            break
        # If all keys exhausted normally, move to next model
        if keys_tried >= n_keys and model == models[model_idx]:
            logger.warning(
                "🔀 [%s] All %d keys exhausted for model %s, trying fallback",
                label,
                n_keys,
                model,
            )
            model_idx += 1

    # Everything exhausted
    raise last_exc  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Resilient Groq SDK client (for ClaimClassifier)
# ─────────────────────────────────────────────────────────────────────────────


def get_groq_client(api_key: Optional[str] = None) -> Groq:
    """
    Create a Groq SDK client with the current (or specified) API key.

    Args:
        api_key: Explicit key override; if None, uses the current rotated key.

    Returns:
        groq.Groq instance.
    """
    key = api_key or _current_key()
    return _build_sdk_client(key)


def resilient_groq_call(
    call_fn,
    *,
    max_retries: Optional[int] = None,
    model_order: Optional[list[str]] = None,
):
    """
    Execute a Groq SDK call with retry + key rotation + model fallback.

    Args:
        call_fn: ``(client: Groq, model: str) -> T``
        max_retries: Override for settings.groq_max_retries.

    Returns:
        The result of call_fn on success.
    """

    def _execute(key: str, model: str):
        if settings.llm_backend == "openrouter":
            model = settings.resolve_openrouter_slug(model)
            provider_label = "openrouter"
        else:
            provider_label = "groq"
        client = _build_sdk_client(key)
        result = call_fn(client, model)
        prompt_tokens, completion_tokens, total_tokens = extract_token_usage(result)
        _record_llm_stats(
            provider=provider_label,
            model=model,
            source="groq_sdk_call",
            usage_payload=result,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return result

    return _resilient_loop(
        _execute,
        max_retries=max_retries,
        label="Groq",
        model_order=model_order,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resilient ChatGroq (LangChain) factory
# ─────────────────────────────────────────────────────────────────────────────


def get_chat_groq(
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    **kwargs,
):
    """Create a chat LLM instance.

    Returns ChatGroq (Groq Cloud) when llm_backend='groq', or ChatVLLMOffline
    when llm_backend='local'. All existing callers (evaluation, tools, agents)
    get the correct backend automatically without any code changes.
    """
    if settings.llm_backend == "local":
        from services.vllm_client import ChatVLLMOffline

        return ChatVLLMOffline(
            model=model or settings.groq_models[0],
            temperature=(
                temperature if temperature is not None else settings.llm_temperature
            ),
            max_tokens=max_tokens or settings.llm_max_tokens,
        )
    if settings.llm_backend == "openrouter":
        key = api_key or _current_key()
        return _build_chat_llm(
            key,
            model or settings.groq_models[0],
            temperature if temperature is not None else settings.llm_temperature,
            max_tokens or settings.llm_max_tokens,
        )
    key = api_key or _current_key()
    return ChatGroq(
        api_key=key,
        model=model or settings.groq_models[0],
        temperature=(
            temperature if temperature is not None else settings.llm_temperature
        ),
        max_tokens=max_tokens or settings.llm_max_tokens,
        **kwargs,
    )


def resilient_chat_call(
    llm_or_factory,
    messages,
    *,
    max_retries: Optional[int] = None,
    model_order: Optional[list[str]] = None,
    cancel_checker: Optional[Callable[[], bool]] = None,
    **invoke_kwargs,
):
    """
    Execute a LangChain ChatGroq .invoke() with smart retry strategy.

    - Model over-capacity → switch model immediately
    - Rate limit          → rotate API key
    - Transient error     → backoff + retry
    """
    from langchain_core.language_models.chat_models import BaseChatModel as _Base

    # Fast path: non-Groq backends (e.g. vLLM) — call directly, no retry loop.
    if isinstance(llm_or_factory, _Base) and not isinstance(llm_or_factory, ChatGroq):
        return llm_or_factory.invoke(messages, **invoke_kwargs)

    ref_llm = llm_or_factory  # used to read temperature / max_tokens

    def _execute(key: str, model: str):
        if callable(ref_llm) and not isinstance(ref_llm, ChatGroq):
            llm = ref_llm(key, model)
        else:
            llm = _build_chat_llm(
                key,
                model,
                getattr(ref_llm, "temperature", settings.llm_temperature),
                getattr(ref_llm, "max_tokens", settings.llm_max_tokens),
            )
        raw = llm.invoke(messages, **invoke_kwargs)
        raw_content = raw.content if isinstance(raw.content, str) else str(raw.content)
        stripped = _strip_thinking(raw_content)
        result = (
            AIMessage(
                content=stripped,
                response_metadata=getattr(raw, "response_metadata", {}),
            )
            if stripped != raw_content
            else raw
        )
        prompt_tokens, completion_tokens, total_tokens = extract_token_usage(result)
        _record_llm_stats(
            provider="openrouter" if settings.llm_backend == "openrouter" else "groq",
            model=model,
            source="chat_groq_invoke",
            usage_payload=result,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return result

    return _resilient_loop(
        _execute,
        max_retries=max_retries,
        label="ChatGroq",
        model_order=model_order,
        cancel_checker=cancel_checker,
    )


def _chunk_to_text(chunk) -> str:
    """Extract text payload from a LangChain stream chunk."""
    if chunk is None:
        return ""

    content = getattr(chunk, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                txt = item.get("text") or item.get("content") or ""
                if txt:
                    parts.append(str(txt))
        return "".join(parts)
    return str(content)


def resilient_chat_stream(
    llm_or_factory,
    messages,
    *,
    on_token: Optional[Callable[[str], None]] = None,
    max_retries: Optional[int] = None,
    model_order: Optional[list[str]] = None,
    cancel_checker: Optional[Callable[[], bool]] = None,
    **stream_kwargs,
):
    """
    Execute a LangChain ChatGroq .stream() with retry/key-rotation/fallback.

    Emits token chunks via ``on_token`` and returns an AIMessage with full text.
    """
    from langchain_core.language_models.chat_models import BaseChatModel as _Base

    # Fast path: non-Groq backends — stream directly, no retry loop.
    if isinstance(llm_or_factory, _Base) and not isinstance(llm_or_factory, ChatGroq):
        pieces: list[str] = []
        for chunk in llm_or_factory.stream(messages, **stream_kwargs):
            text = _chunk_to_text(chunk)
            if text:
                pieces.append(text)
                if on_token is not None:
                    on_token(text)
        return AIMessage(content="".join(pieces))

    ref_llm = llm_or_factory  # used to read temperature / max_tokens

    def _execute(key: str, model: str):
        if callable(ref_llm) and not isinstance(ref_llm, ChatGroq):
            llm = ref_llm(key, model)
        else:
            llm = _build_chat_llm(
                key,
                model,
                getattr(ref_llm, "temperature", settings.llm_temperature),
                getattr(ref_llm, "max_tokens", settings.llm_max_tokens),
            )

        pieces: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None

        for chunk in llm.stream(messages, **stream_kwargs):
            if cancel_checker is not None and cancel_checker():
                raise PipelineCancelled("Execution manually stopped.")

            chunk_prompt, chunk_completion, chunk_total = extract_token_usage(chunk)
            if chunk_prompt is not None:
                prompt_tokens = chunk_prompt
            if chunk_completion is not None:
                completion_tokens = chunk_completion
            if chunk_total is not None:
                total_tokens = chunk_total

            text = _chunk_to_text(chunk)
            if not text:
                continue
            pieces.append(text)
            if on_token is not None:
                on_token(text)

        result = AIMessage(content=_strip_thinking("".join(pieces)))
        _record_llm_stats(
            provider="openrouter" if settings.llm_backend == "openrouter" else "groq",
            model=model,
            source="chat_groq_stream",
            usage_payload=result,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return result

    return _resilient_loop(
        _execute,
        max_retries=max_retries,
        label="ChatGroqStream",
        model_order=model_order,
        cancel_checker=cancel_checker,
    )
