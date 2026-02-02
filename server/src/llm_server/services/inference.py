# src/llm_server/services/inference.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.core.metrics import LLM_TOKENS
from llm_server.core.redis import redis_get, redis_set
from llm_server.db.models import CompletionCache, InferenceLog


@dataclass(frozen=True)
class CacheSpec:
    """
    Canonical cache identity for an inference item.

    Contract (Option A):
      - CompletionCache.output stores the cached value as a STRING
      - Redis stores JSON payload {"output": <STRING>}
      - payload key is always "output" (even for extract, where the string is JSON)
    """
    model_id: str
    prompt: str
    prompt_hash: str
    params_fp: str
    redis_key: str
    redis_ttl_seconds: int = 3600


def set_request_meta(request: Any, *, route: str, model_id: str, cached: bool = False) -> None:
    request.state.route = route
    request.state.model_id = model_id
    request.state.cached = cached


def record_token_metrics(model_id: str, prompt_tokens: int | None, completion_tokens: int | None) -> None:
    # Best-effort metrics
    if prompt_tokens is not None:
        LLM_TOKENS.labels(direction="prompt", model_id=model_id).inc(prompt_tokens)
    if completion_tokens is not None:
        LLM_TOKENS.labels(direction="completion", model_id=model_id).inc(completion_tokens)


async def _read_redis_output(redis: Any, *, cache: CacheSpec, kind: str) -> str | None:
    raw = await redis_get(redis, cache.redis_key, model_id=cache.model_id, kind=kind)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
        out = payload.get("output")
        return out if isinstance(out, str) and out != "" else None
    except Exception:
        return None


async def _read_db_output(session: AsyncSession, *, cache: CacheSpec) -> str | None:
    row = await session.execute(
        select(CompletionCache).where(
            CompletionCache.model_id == cache.model_id,
            CompletionCache.prompt_hash == cache.prompt_hash,
            CompletionCache.params_fingerprint == cache.params_fp,
        )
    )
    cached = row.scalar_one_or_none()
    out = cached.output if cached is not None else None
    return out if isinstance(out, str) and out != "" else None


async def get_cached_output(
    session: AsyncSession,
    redis: Any | None,
    *,
    cache: CacheSpec,
    kind: str,
    enabled: bool,
) -> tuple[str | None, bool, str | None]:
    """
    Canonical cache read:
      1) Redis
      2) DB
      3) miss

    Returns:
      (output_str_or_none, cached_flag, layer)
      layer is one of: "redis" | "db" | None
    """
    if not enabled:
        return None, False, None

    # 1) Redis
    if redis is not None:
        out = await _read_redis_output(redis, cache=cache, kind=kind)
        if out is not None:
            return out, True, "redis"

    # 2) DB
    out = await _read_db_output(session, cache=cache)
    if out is None:
        return None, False, None

    # On DB hit, backfill Redis best-effort (store string under "output")
    if redis is not None:
        try:
            await redis_set(
                redis,
                cache.redis_key,
                json.dumps({"output": out}, ensure_ascii=False),
                ex=cache.redis_ttl_seconds,
            )
        except Exception:
            pass

    return out, True, "db"


async def write_cache(
    session: AsyncSession,
    redis: Any | None,
    *,
    cache: CacheSpec,
    output: str,
    enabled: bool,
) -> None:
    """
    Canonical cache write:
      - DB insert (ignore IntegrityError)
      - Redis set best-effort

    Contract: output is a STRING.
    """
    if not enabled:
        return
    if not isinstance(output, str) or output == "":
        return

    session.add(
        CompletionCache(
            model_id=cache.model_id,
            prompt=cache.prompt,
            prompt_hash=cache.prompt_hash,
            params_fingerprint=cache.params_fp,
            output=output,
        )
    )

    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()

    if redis is not None:
        try:
            await redis_set(
                redis,
                cache.redis_key,
                json.dumps({"output": output}, ensure_ascii=False),
                ex=cache.redis_ttl_seconds,
            )
        except Exception:
            pass


async def write_inference_log(
    session: AsyncSession,
    *,
    api_key: str,
    request_id: str | None,
    route: str,
    client_host: str | None,
    model_id: str,
    params_json: Mapping[str, Any],
    prompt: str,
    output: str,
    latency_ms: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    commit: bool = True,
) -> None:
    """
    Canonical log write. Default commit=True matches your early-return pattern.
    For batched endpoints, pass commit=False and commit once at the end.
    """
    session.add(
        InferenceLog(
            api_key=api_key,
            request_id=request_id,
            route=route,
            client_host=client_host,
            model_id=model_id,
            params_json=dict(params_json),
            prompt=prompt,
            output=output,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    )
    if commit:
        await session.commit()