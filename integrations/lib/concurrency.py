from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, List, Tuple


@dataclass(frozen=True)
class ConcurrentResult:
    ok: bool
    value: Any | None
    error: Exception | None
    latency_ms: float


async def _run_one(
    coro_factory: Callable[[], Awaitable[Any]],
) -> ConcurrentResult:
    start = time.perf_counter()
    try:
        val = await coro_factory()
        return ConcurrentResult(
            ok=True,
            value=val,
            error=None,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
    except Exception as e:
        return ConcurrentResult(
            ok=False,
            value=None,
            error=e,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )


async def run_concurrent(
    *,
    n: int,
    task_factory: Callable[[int], Awaitable[Any]],
) -> List[ConcurrentResult]:
    """
    Run `n` concurrent tasks produced by task_factory(i).

    This is intentionally a *thin* wrapper:
    - no retries
    - no rate limiting
    - no fancy pooling
    """
    tasks = [
        _run_one(lambda i=i: task_factory(i))
        for i in range(n)
    ]
    return await asyncio.gather(*tasks)


def assert_all_ok(results: Iterable[ConcurrentResult]) -> None:
    errors = [r.error for r in results if not r.ok]
    if errors:
        raise AssertionError(
            f"{len(errors)} concurrent tasks failed: "
            + ", ".join(type(e).__name__ for e in errors if e)
        )


def assert_latency_reasonable(
    results: Iterable[ConcurrentResult],
    *,
    max_p95_ms: float,
) -> None:
    latencies = sorted(r.latency_ms for r in results)
    if not latencies:
        return

    idx = int(0.95 * (len(latencies) - 1))
    p95 = latencies[idx]

    if p95 > max_p95_ms:
        raise AssertionError(
            f"p95 latency too high: {p95:.1f} ms > {max_p95_ms:.1f} ms"
        )