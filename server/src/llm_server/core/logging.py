from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


# -----------------------------
# JSON log formatter
# -----------------------------

class JsonFormatter(logging.Formatter):
    """
    Minimal JSON formatter.

    Standard fields:
      - ts, level, logger, message

    Plus selected extras if present on the LogRecord:
      - request_id, method, path, status_code, latency_ms, client_ip
      - route, model_id, api_key_id, api_key_role
      - error_type, error_message
    """

    _EXTRA_KEYS = [
        "request_id",
        "method",
        "path",
        "status_code",
        "latency_ms",
        "client_ip",
        "route",
        "model_id",
        "api_key_id",
        "api_key_role",
        "error_type",
        "error_message",
        "cached", 
    ]

    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key in self._EXTRA_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                base[key] = value

        # If there is exception info, include a short form
        if record.exc_info:
            base.setdefault("error_type", record.exc_info[0].__name__)
            base.setdefault("error_message", str(record.exc_info[1]))

        return json.dumps(base, default=str)


# -----------------------------
# Request logging middleware
# -----------------------------

access_logger = logging.getLogger("llm_server.access")
error_logger = logging.getLogger("llm_server.error")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Per-request logging + request_id propagation.

    - Generates request_id and attaches to request.state.request_id
    - Logs a structured "request" record on success
    - Logs a structured "request_error" record on unhandled exceptions
    - Adds X-Request-ID header to the response
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.time()
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        client_ip: Optional[str] = None
        if request.client:
            client_ip = request.client.host

        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000.0

            # Optionally also pull model_id / cached here if you want them on error logs
            model_id = getattr(request.state, "model_id", None)
            cached = getattr(request.state, "cached", None)

            error_logger.exception(
                "request_error",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": client_ip,
                    "latency_ms": latency_ms,
                    "model_id": model_id,
                    "cached": cached,
                },
            )
            raise

        latency_ms = (time.time() - start) * 1000.0

        # Add X-Request-ID header for clients
        response.headers["X-Request-ID"] = request_id

        # Pull values set by handlers (e.g. /v1/generate)
        model_id = getattr(request.state, "model_id", None)
        cached = getattr(request.state, "cached", None)

        extra: Dict[str, Any] = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "client_ip": client_ip,
            "latency_ms": latency_ms,
        }

        if model_id is not None:
            extra["model_id"] = model_id
        if cached is not None:
            extra["cached"] = cached

        access_logger.info("request", extra=extra)

        return response


# -----------------------------
# Setup
# -----------------------------

def _configure_root_logging() -> None:
    """
    Configure root + uvicorn loggers to use JSON formatting.

    Called once from main.create_app().
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    # Root logger
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)

    # Uvicorn loggers
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)


def setup(app: FastAPI) -> None:
    """
    Called from main.create_app().

    - Configures logging
    - Adds RequestLoggingMiddleware
    """
    _configure_root_logging()
    app.add_middleware(RequestLoggingMiddleware)