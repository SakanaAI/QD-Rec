"""Chat and embedding requests through OpenRouter, using provider/model IDs.

API requests get up to seven attempts, each capped at 90 seconds. SDK retries
are disabled; evaluation-level retries are handled by the run and seed loops.
"""

from __future__ import annotations

import math
import asyncio as _asyncio
import json as _json
import os
import re as _re

import httpx
import json_repair as _json_repair
from dotenv import load_dotenv
from openai import AsyncOpenAI, AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError


_RETRY_BACKOFF_S = (1, 2, 4, 8, 12, 20)  # waits between attempts: 7 attempts in all
_TIMEOUT_S = 90.0  # per attempt; a high-effort call can think for over a minute
# Errors no retry can fix (a wrong key, model id or request): raised at once.
_NOT_RETRIED = (AuthenticationError, PermissionDeniedError, BadRequestError, NotFoundError)


async def _with_retries(call, problem, what: str):
    """Await ``call()`` until ``problem(result)`` is None, retrying an exception or
    unusable result with backoff; raises after the last attempt. Errors in
    ``_NOT_RETRIED`` are raised at once."""
    last = None
    for i in range(len(_RETRY_BACKOFF_S) + 1):
        try:
            # asyncio.wait_for caps an attempt's TOTAL time: httpx's timeout is per
            # read, so a server that trickles a response never trips it.
            result = await _asyncio.wait_for(call(), timeout=_TIMEOUT_S)
            last = problem(result)
            if last is None:
                return result
        except _NOT_RETRIED:
            raise
        except Exception as e:  # noqa: BLE001 - retry any transient API error
            last = f"{type(e).__name__}: {str(e)[:160]}"
        if i < len(_RETRY_BACKOFF_S):
            await _asyncio.sleep(_RETRY_BACKOFF_S[i])
    raise RuntimeError(f"{what} failed after {len(_RETRY_BACKOFF_S) + 1} attempts: {last}")


def _chat_problem(r) -> str | None:
    """None for a reply with content; otherwise what was wrong. Under load OpenRouter
    can return HTTP 200 with an error body and no choices."""
    if r and r.choices and r.choices[0].message and r.choices[0].message.content:
        return None
    finish = r.choices[0].finish_reason if r and r.choices else None
    return (f"empty choices/content (error={getattr(r, 'error', None)}, finish_reason={finish}, "
            f"usage={getattr(getattr(r, 'usage', None), 'completion_tokens', None)} out-tok)")


async def acreate(client, **kwargs):
    """``client.chat.completions.create`` with the model's reasoning effort and retries."""
    kwargs = _with_effort(kwargs)
    return await _with_retries(lambda: client.chat.completions.create(**kwargs), _chat_problem, "acreate")


async def aembed(**kwargs):
    """``client().embeddings.create`` with retries. The client is created first,
    so a missing API key raises at once instead of being retried."""
    c = client()
    return await _with_retries(
        lambda: c.embeddings.create(**kwargs),
        lambda r: None if r and r.data else "empty embedding response",
        "embedding",
    )


def loads_lenient(content: str | None):
    """Parse a model response as an object, tolerating fences and malformed JSON.

    If the response wraps objects in an array, use the first object.
    """
    return _as_object(_loads_lenient_raw(content))


def parse_json_bool(value: object) -> bool | None:
    """Accept a JSON boolean or its string spelling; reject other values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None



def parse_judge_score(value: object, *, low: int = 0, high: int = 10) -> int:
    """Accept integral numbers or numeric strings in [low, high].

    Reject booleans, fractions and invalid scores so they cannot enter the archive
    or the bandit's rewards.
    """
    if isinstance(value, bool):
        raise ValueError(f"judge score is a boolean: {value!r}")
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise ValueError(f"judge score is not a number: {value!r}") from None
    if not isinstance(value, (int, float)):
        raise ValueError(f"judge score has type {type(value).__name__}: {value!r}")
    if not math.isfinite(value) or value != int(value):
        raise ValueError(f"judge score is not an integer: {value!r}")
    score = int(value)
    if not low <= score <= high:
        raise ValueError(f"judge score {score} is outside {low}-{high}")
    return score


def _as_object(parsed):
    """Normalize a parsed JSON value to a dict (see ``loads_lenient``)."""
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                return item
    return {}


def _loads_lenient_raw(content: str | None):
    if not content or not content.strip():
        raise ValueError("empty LLM response content")
    s = content.strip()
    if s.startswith("```"):
        s = _re.sub(r"^```(?:json)?\s*", "", s, flags=_re.IGNORECASE)
        s = _re.sub(r"\s*```$", "", s).strip()
    try:
        # Accept raw newlines/tabs in strings; malformed escapes need repair below.
        return _json.loads(s, strict=False)
    except _json.JSONDecodeError:
        pass
    # Recover the first JSON value from prose or concatenated objects,
    # then try removing trailing commas.
    start = min((p for p in (s.find("{"), s.find("[")) if p != -1), default=-1)
    if start == -1:
        raise _json.JSONDecodeError("no JSON object/array found", s, 0)
    snippet = s[start:]
    decoder = _json.JSONDecoder(strict=False)
    try:
        obj, _ = decoder.raw_decode(snippet)
        return obj
    except _json.JSONDecodeError:
        pass
    try:
        cleaned = _re.sub(r",(\s*[}\]])", r"\1", snippet)  # drop trailing commas
        obj, _ = decoder.raw_decode(cleaned)
        return obj
    except _json.JSONDecodeError:
        pass
    # Last resort for malformed escapes and stray quotes.
    return _json_repair.loads(s)


class EmptyPayloadError(RuntimeError):
    """A proposer response carried no usable payload field."""


async def agenerate_payload(generate, payload_key: str, *, attempts: int = 3):
    """Return (result_dict, payload_text), retrying a missing or empty payload.

    Raise EmptyPayloadError after exhaustion. Never substitute other response fields:
    an empty hypothesis distorts novelty, and substitute text can spread via examples.
    """
    for _attempt in range(attempts):
        result = await generate()
        if isinstance(result, dict):
            text = (result.get(payload_key) or "").strip()
            if text:
                return result, text
    raise EmptyPayloadError(
        f"proposer returned no '{payload_key}' in {attempts} attempts"
    )


def json_schema_format(name: str, schema: dict, strict: bool = True) -> dict:
    """Build a schema-constrained OpenAI response_format.

    Strict schemas require additionalProperties=False and every property in required.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": strict},
    }

# A bounded connection pool, reused within a batch. The 30 s keepalive expiry
# closes connections left behind by cancelled requests before they exhaust the pool.
_HTTP_LIMITS = httpx.Limits(
    max_connections=64, max_keepalive_connections=10, keepalive_expiry=30.0
)

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_effort_models: set[str] | None = None


def _supports_effort(model: str) -> bool:
    """Whether OpenRouter lists ``reasoning_effort`` for ``model``; the model list
    is fetched once per process."""
    global _effort_models
    if _effort_models is None:
        listing = httpx.get(f"{OPENROUTER_BASE_URL}/models", timeout=60).json()["data"]
        _effort_models = {
            m["id"] for m in listing if "reasoning_effort" in (m.get("supported_parameters") or [])
        }
    return model in _effort_models


def reasoning_effort_for(model: str, effort: str | None) -> str | None:
    """Return effort only when the model's OpenRouter listing supports it.

    Unsupported effort can enable extended thinking. Use the same resolved value
    for requests and archive metadata.
    """
    if not effort or not _supports_effort(model):
        return None
    return effort


def _with_effort(kwargs: dict) -> dict:
    """``kwargs`` with ``reasoning_effort`` kept only if the model supports it."""
    kwargs = dict(kwargs)
    effort = reasoning_effort_for(kwargs.get("model", ""), kwargs.pop("reasoning_effort", None))
    if effort:
        kwargs["reasoning_effort"] = effort
    return kwargs


def get_async_openai_client() -> AsyncOpenAI:
    """An OpenAI SDK client pointed at OpenRouter, with the SDK's own retries off."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; copy .env.example to .env and fill it in")
    return AsyncOpenAI(
        api_key=key, base_url=OPENROUTER_BASE_URL, timeout=_TIMEOUT_S, max_retries=0,
        http_client=httpx.AsyncClient(limits=_HTTP_LIMITS, timeout=_TIMEOUT_S),
    )


_client = None


def client() -> AsyncOpenAI:
    """Return the shared client, creating it on first use."""
    global _client
    if _client is None:
        _client = get_async_openai_client()
    return _client
