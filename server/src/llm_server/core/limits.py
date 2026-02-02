# src/llm_server/core/limits.py
from __future__ import annotations

import asyncio
import time
import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from llm_server.core.config import get_settings

logger = logging.getLogger("llm_server.limits")

# Only guard heavy routes (tweak as needed)
HEAVY_PREFIXES = ("/v1/generate", "/v1/extract")


def _max_concurrency() -> int:
    # Backwards compatible: if you haven't added this setting yet, default to 2.
    s = get_settings()
    return int(getattr(s, "max_concurrent_requests", 2))


class _ConcurrencyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that limits concurrency for heavy endpoints.

    Default behavior: queue (do not reject).
    Adds lightweight logging about wait time to improve observability.
    """

    def __init__(self, app):
        super().__init__(app)
        self._semaphore = asyncio.Semaphore(_max_concurrency())

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path.startswith(HEAVY_PREFIXES):
            t0 = time.time()
            async with self._semaphore:
                wait_ms = (time.time() - t0) * 1000.0
                if wait_ms > 5:
                    logger.info(
                        "concurrency_wait",
                        extra={
                            "request_id": getattr(getattr(request, "state", None), "request_id", None),
                            "path": request.url.path,
                            "wait_ms": round(wait_ms, 2),
                            "max_concurrent": _max_concurrency(),
                        },
                    )
                return await call_next(request)

        return await call_next(request)


def setup(app) -> None:
    """Install concurrency middleware."""
    app.add_middleware(_ConcurrencyMiddleware)