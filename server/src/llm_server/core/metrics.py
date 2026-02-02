# src/llm_server/core/metrics.py
from __future__ import annotations

import time
from typing import Callable, cast

from fastapi import FastAPI, Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    REGISTRY,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from llm_server.core.config import get_settings


def _metrics_handler() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


def _existing_collector(name: str):
    mapping = getattr(REGISTRY, "_names_to_collectors", None)
    if not isinstance(mapping, dict):
        return None
    return mapping.get(name)


def _assert_labelnames_match(existing, expected: list[str]) -> None:
    try:
        existing_labels = list(getattr(existing, "_labelnames", []))
    except Exception:
        existing_labels = []

    # If an existing collector is already labeled, enforce exact match.
    if existing_labels and existing_labels != expected:
        raise RuntimeError(
            f"Metric '{getattr(existing, '_name', 'unknown')}' already exists with labels={existing_labels}, "
            f"but code expected labels={expected}. Fix the name or align labelnames."
        )


def _get_or_create_counter(name: str, documentation: str, labelnames: list[str]) -> Counter:
    existing = _existing_collector(name)
    if existing is not None:
        if not isinstance(existing, Counter):
            raise RuntimeError(f"Metric '{name}' exists but is not a Counter (got {type(existing)!r}).")
        _assert_labelnames_match(existing, labelnames)
        return cast(Counter, existing)
    return Counter(name, documentation, labelnames)


def _get_or_create_histogram(name: str, documentation: str, labelnames: list[str]) -> Histogram:
    existing = _existing_collector(name)
    if existing is not None:
        if not isinstance(existing, Histogram):
            raise RuntimeError(f"Metric '{name}' exists but is not a Histogram (got {type(existing)!r}).")
        _assert_labelnames_match(existing, labelnames)
        return cast(Histogram, existing)
    return Histogram(name, documentation, labelnames)


def _get_or_create_gauge(name: str, documentation: str, labelnames: list[str] | None = None) -> Gauge:
    existing = _existing_collector(name)
    if existing is not None:
        if not isinstance(existing, Gauge):
            raise RuntimeError(f"Metric '{name}' exists but is not a Gauge (got {type(existing)!r}).")
        _assert_labelnames_match(existing, labelnames or [])
        return cast(Gauge, existing)
    return Gauge(name, documentation, labelnames or [])


def _best_route_label(request: Request) -> str:
    route = getattr(request.state, "route", None)
    if isinstance(route, str) and route:
        return route

    scope_route = request.scope.get("route")
    path = getattr(scope_route, "path", None)
    if isinstance(path, str) and path:
        return path

    return request.url.path


# -----------------------------------------
# Core API metrics
# -----------------------------------------

LLM_TOKENS = _get_or_create_counter(
    "llm_tokens_total",
    "Total tokens processed",
    ["direction", "model_id"],
)

REQUEST_LATENCY = _get_or_create_histogram(
    "llm_api_request_latency_seconds",
    "Request latency in seconds",
    ["route", "model_id", "cached", "status_code"],
)

REQUEST_COUNT = _get_or_create_counter(
    "llm_api_request_total",
    "Total requests",
    ["route", "model_id", "cached", "status_code"],
)

LLM_REDIS_HITS = _get_or_create_counter(
    "llm_redis_hits_total",
    "Redis cache hits",
    ["model_id", "kind"],
)

LLM_REDIS_MISSES = _get_or_create_counter(
    "llm_redis_misses_total",
    "Redis cache misses",
    ["model_id", "kind"],
)

LLM_REDIS_LATENCY = _get_or_create_histogram(
    "llm_redis_latency_seconds",
    "Latency of Redis cache GET operations",
    ["model_id", "kind"],
)

LLM_REDIS_ENABLED = _get_or_create_gauge(
    "llm_redis_enabled",
    "Whether Redis caching is enabled (1=yes, 0=no)",
)

# -----------------------------------------
# Extraction Metrics (Phase 2)
# -----------------------------------------

EXTRACTION_REQUESTS = _get_or_create_counter(
    "llm_extraction_requests_total",
    "Total extraction requests",
    ["schema_id", "model_id"],
)

EXTRACTION_CACHE_HITS = _get_or_create_counter(
    "llm_extraction_cache_hits_total",
    "Extraction cache hits by layer",
    ["schema_id", "model_id", "layer"],  # layer: redis|db
)

EXTRACTION_VALIDATION_FAILURES = _get_or_create_counter(
    "llm_extraction_validation_failures_total",
    "Extraction failures due to parse or schema validation issues",
    ["schema_id", "model_id", "stage"],  # stage: parse|validate|repair_parse|repair_validate
)

EXTRACTION_REPAIR = _get_or_create_counter(
    "llm_extraction_repair_total",
    "Extraction repair attempts and outcomes",
    ["schema_id", "model_id", "outcome"],  # attempted|success|failure
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            route = _best_route_label(request)
            model_id = getattr(request.state, "model_id", "unknown")
            cached = getattr(request.state, "cached", False)
            cached_label = "true" if cached else "false"
            status_label = str(status_code)

            elapsed = time.perf_counter() - start

            REQUEST_LATENCY.labels(
                route=route,
                model_id=model_id,
                cached=cached_label,
                status_code=status_label,
            ).observe(elapsed)

            REQUEST_COUNT.labels(
                route=route,
                model_id=model_id,
                cached=cached_label,
                status_code=status_label,
            ).inc()


def setup(app: FastAPI) -> None:
    # Set gauge here (not at import time) so tests/env overrides are respected.
    s = get_settings()
    LLM_REDIS_ENABLED.set(1 if bool(s.redis_enabled) else 0)

    app.add_middleware(MetricsMiddleware)
    app.add_api_route("/metrics", _metrics_handler, methods=["GET"], include_in_schema=False)