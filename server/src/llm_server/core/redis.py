# src/llm_server/core/redis.py
from __future__ import annotations

import time
from typing import Optional

from fastapi import Request
from redis.asyncio import Redis, from_url

from llm_server.core.config import get_settings
from llm_server.core.metrics import (
    LLM_REDIS_HITS,
    LLM_REDIS_MISSES,
    LLM_REDIS_LATENCY,
)


async def init_redis() -> Optional[Redis]:
    s = get_settings()
    if not bool(s.redis_enabled) or not s.redis_url:
        return None
    return from_url(s.redis_url, decode_responses=True)


async def close_redis(client: Optional[Redis]) -> None:
    if client is not None:
        await client.aclose()


def get_redis_from_request(request: Request) -> Optional[Redis]:
    return getattr(request.app.state, "redis", None)


async def redis_get(
    redis: Optional[Redis],
    key: str,
    *,
    model_id: str = "unknown",
    kind: str = "single",
) -> Optional[str]:
    if redis is None:
        return None

    start = time.perf_counter()
    try:
        val = await redis.get(key)
    finally:
        try:
            LLM_REDIS_LATENCY.labels(model_id=model_id, kind=kind).observe(time.perf_counter() - start)
        except Exception:
            pass

    try:
        if val is None:
            LLM_REDIS_MISSES.labels(model_id=model_id, kind=kind).inc()
        else:
            LLM_REDIS_HITS.labels(model_id=model_id, kind=kind).inc()
    except Exception:
        pass

    return val


async def redis_set(
    redis: Optional[Redis],
    key: str,
    value: str,
    *,
    ex: Optional[int] = None,
) -> None:
    if redis is None:
        return
    if ex is not None:
        await redis.set(key, value, ex=ex)
    else:
        await redis.set(key, value)