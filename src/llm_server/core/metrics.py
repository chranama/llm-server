# src/llm_server/core/metrics.py
import time
from fastapi import APIRouter, Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# -----------------------------
# Prometheus metrics
# -----------------------------

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Latency of HTTP requests",
    ["method", "endpoint"],
)

LLM_REQUESTS = Counter(
    "llm_requests_total",
    "Total LLM requests (generate/stream) by route, model, and cache status.",
    ["route", "model_id", "cached"],
)

LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Total LLM tokens by direction (prompt/completion) and model.",
    ["direction", "model_id"],
)

LLM_LATENCY = Histogram(
    "llm_request_latency_ms",
    "LLM request latency in milliseconds, by route and model.",
    ["route", "model_id"],
    buckets=(50, 100, 200, 400, 800, 1600, 3200, 6400, 12800),
)

# -----------------------------
# /metrics endpoint
# -----------------------------

router = APIRouter()


@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# -----------------------------
# Middleware wiring
# -----------------------------

def setup(app) -> None:
    """Add latency/count middleware and mount /metrics route."""

    @app.middleware("http")
    async def prometheus_metrics(request: Request, call_next):
        start = time.time()
        response: Response = await call_next(request)
        duration = time.time() - start  # seconds

        endpoint = request.url.path
        method = request.method
        status = response.status_code

        # Basic HTTP metrics
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status=str(status),
        ).inc()

        REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

        # -------------------------
        # LLM-specific metrics
        # -------------------------
        # These are populated by the LLM handlers (e.g. /v1/generate, /v1/stream)
        route = getattr(request.state, "route", endpoint)
        model_id = getattr(request.state, "model_id", "unknown")
        cached_val = getattr(request.state, "cached", None)

        if cached_val is True:
            cached = "true"
        elif cached_val is False:
            cached = "false"
        else:
            cached = "unknown"

        # Only record LLM metrics for LLM routes
        if route in ("/v1/generate", "/v1/stream"):
            # Count of LLM requests
            LLM_REQUESTS.labels(
                route=route,
                model_id=model_id or "unknown",
                cached=cached,
            ).inc()

            # Latency in ms for LLM calls
            LLM_LATENCY.labels(
                route=route,
                model_id=model_id or "unknown",
            ).observe(duration * 1000.0)

        return response

    app.include_router(router)