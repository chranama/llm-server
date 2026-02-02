# backend/src/llm_server/core/errors.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, Request
from fastapi import HTTPException as FastAPIHTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("llm.errors")


class AppError(FastAPIHTTPException):
    """
    Canonical application error.

    By subclassing HTTPException, FastAPI will always turn this into an HTTP response,
    even in tests where the app is instantiated as FastAPI() without custom handlers.
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.extra = extra

        detail: Dict[str, Any] = {"code": code, "message": message}
        if extra:
            detail["extra"] = extra
        super().__init__(status_code=status_code, detail=detail)


def _request_id(request: Request) -> Optional[str]:
    return getattr(getattr(request, "state", None), "request_id", None)


def _to_json_error(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """
    Canonical error envelope.

    Contract:
      - top-level 'code'
      - top-level 'message'
      - top-level 'extra' (optional)
      - top-level 'request_id' (optional)
    Also sets X-Request-ID header if request_id exists.
    """
    payload: Dict[str, Any] = {"code": code, "message": message}

    if extra:
        payload["extra"] = extra

    rid = _request_id(request)
    if rid:
        payload["request_id"] = rid

    resp = JSONResponse(payload, status_code=status_code)
    if rid:
        resp.headers["X-Request-ID"] = rid
    return resp


async def handle_fastapi_http_exception(request: Request, exc: FastAPIHTTPException):
    """
    Catches fastapi.HTTPException.
    `detail` may be:
      - a string
      - a dict (sometimes already shaped)
      - our canonical shape (code/message/extra)
    """
    detail: Union[str, Dict[str, Any]] = exc.detail

    if isinstance(detail, dict):
        code = str(detail.get("code", "http_error"))
        message = str(detail.get("message", "HTTP error"))

        extra = detail.get("extra")
        if not isinstance(extra, dict):
            # Preserve any other keys as extra for debugging/classification
            extra = {k: v for k, v in detail.items() if k not in {"code", "message", "extra"}}

        return _to_json_error(request, status_code=exc.status_code, code=code, message=message, extra=extra or None)

    return _to_json_error(
        request,
        status_code=exc.status_code,
        code="http_error",
        message=str(detail),
    )


async def handle_starlette_http_exception(request: Request, exc: StarletteHTTPException):
    """
    Catches Starlette's HTTPException (e.g., router 404).
    """
    if exc.status_code == 404:
        return _to_json_error(request, status_code=404, code="not_found", message="Route not found")
    return _to_json_error(request, status_code=exc.status_code, code="http_error", message=str(exc.detail))


async def handle_validation_error(request: Request, exc: RequestValidationError):
    """
    Catches Pydantic/validation errors (422).
    """
    logger.debug("validation_error: %s", exc.errors())
    return _to_json_error(
        request,
        status_code=422,
        code="validation_error",
        message="Request validation failed",
        extra={"stage": "request_validation", "fields": exc.errors()},
    )


async def handle_app_error(request: Request, exc: AppError):
    """
    Our explicit app/business logic errors.
    """
    logger.info("app_error code=%s msg=%s", exc.code, exc.message)
    extra = exc.extra if isinstance(exc.extra, dict) else None
    return _to_json_error(
        request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        extra=extra,
    )


async def handle_unhandled_exception(request: Request, exc: Exception):
    """
    Final safety net: avoid leaking internals, but DO provide stable classification signals.
    """
    logger.exception("unhandled_exception path=%s", request.url.path, exc_info=exc)

    safe_extra: Dict[str, Any] = {
        "stage": "unhandled_exception",
        "path": request.url.path,
        "method": request.method,
        "exc_type": type(exc).__name__,
    }

    # If upstream code set a stage (common pattern: request.state.error_stage), include it.
    stage = getattr(getattr(request, "state", None), "error_stage", None)
    if isinstance(stage, str) and stage.strip():
        safe_extra["stage"] = stage.strip()

    return _to_json_error(
        request,
        status_code=500,
        code="internal_error",
        message="An unexpected error occurred",
        extra=safe_extra,
    )


def setup(app: FastAPI) -> None:
    """
    Register all handlers on the FastAPI app.
    Call from main.py early in setup.
    """
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(FastAPIHTTPException, handle_fastapi_http_exception)
    app.add_exception_handler(StarletteHTTPException, handle_starlette_http_exception)
    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(Exception, handle_unhandled_exception)