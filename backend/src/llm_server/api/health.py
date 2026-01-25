# backend/src/llm_server/api/health.py
from __future__ import annotations

import logging
import os
from typing import Any, Tuple

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.core.config import get_settings
from llm_server.db.session import get_session
from llm_server.core.redis import get_redis_from_request

logger = logging.getLogger("llm_server.api.health")
router = APIRouter(tags=["health"])


def _settings_from_request(request: Request) -> Any:
    # Prefer frozen settings from app.state; fallback to global accessor
    return getattr(request.app.state, "settings", None) or get_settings()


def _effective_model_load_mode(request: Request) -> str:
    """
    Single source of truth for model load mode:
      1) app.state.model_load_mode (set during lifespan)
      2) app.state.settings.model_load_mode
      3) derived default from env (prod=>eager else lazy)
    """
    mode = getattr(request.app.state, "model_load_mode", None)
    if isinstance(mode, str) and mode.strip():
        return mode.strip().lower()

    s = _settings_from_request(request)
    raw = getattr(s, "model_load_mode", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()

    env = str(getattr(s, "env", "dev")).strip().lower()
    return "eager" if env == "prod" else "lazy"


def _require_model_loaded_for_readyz(request: Request) -> bool:
    """
    Cloud-readiness policy:
      - In prod, require default model loaded for /readyz by default.
      - Allow override via REQUIRE_MODEL_READY=0/1 (env override is fine here;
        it is *policy*, not core mode).
    """
    raw = os.getenv("REQUIRE_MODEL_READY")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")

    s = _settings_from_request(request)
    return str(getattr(s, "env", "dev")).strip().lower() == "prod"


def get_llm_from_request(request: Request):
    """Canonical place to grab the LLM instance without triggering loading."""
    return getattr(request.app.state, "llm", None)


def _llm_state(llm) -> str:
    """
    Best-effort inference of whether the model is actually loaded.
    Avoid calling ensure_loaded() here because readiness endpoints should not cause loading.
    """
    if llm is None:
        return "not initialized"

    fn = getattr(llm, "is_loaded", None)
    if callable(fn):
        try:
            return "loaded" if bool(fn()) else "not loaded"
        except Exception:
            logger.exception("LLM is_loaded() check failed")
            return "unknown"

    for attr in ("loaded", "is_ready", "ready"):
        if hasattr(llm, attr):
            try:
                return "loaded" if bool(getattr(llm, attr)) else "not loaded"
            except Exception:
                logger.exception("LLM %s attribute check failed", attr)
                return "unknown"

    for attr in ("model", "_model", "pipeline", "_pipeline"):
        if hasattr(llm, attr):
            try:
                return "loaded" if getattr(llm, attr) is not None else "not loaded"
            except Exception:
                logger.exception("LLM %s heuristic check failed", attr)
                return "unknown"

    return "unknown"


async def _db_readiness(session: AsyncSession) -> Tuple[bool, str]:
    try:
        await session.execute(text("SELECT 1"))
        return True, "ok"
    except Exception:
        logger.exception("DB readiness check failed")
        return False, "error"


async def _redis_readiness(request: Request) -> Tuple[bool, str]:
    s = _settings_from_request(request)
    if not bool(getattr(s, "redis_enabled", False)):
        return True, "disabled"

    try:
        redis = get_redis_from_request(request)
        if redis is None:
            logger.error("Redis enabled but app.state.redis is None")
            return False, "not initialized"

        pong = await redis.ping()
        if pong is True:
            return True, "ok"

        logger.error("Redis ping unexpected: %s", pong)
        return False, f"unexpected response: {pong}"
    except Exception:
        logger.exception("Redis readiness check failed")
        return False, "error"


def _model_loaded_flag(request: Request) -> bool | None:
    """
    Prefer explicit app.state.model_loaded if present.
    Returns None if absent.
    """
    if hasattr(request.app.state, "model_loaded"):
        try:
            return bool(getattr(request.app.state, "model_loaded"))
        except Exception:
            return None
    return None


@router.get("/healthz")
async def healthz():
    """Liveness: must be fast and must not touch DB/Redis/model."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Service readiness: checks DB and (optionally) Redis.
    IMPORTANT: never triggers model loading.
    """
    db_ok, db_status = await _db_readiness(session)
    redis_ok, redis_status = await _redis_readiness(request)

    mode = _effective_model_load_mode(request)
    llm = get_llm_from_request(request)

    loaded_flag = _model_loaded_flag(request)
    llm_status = "disabled" if mode == "off" else _llm_state(llm)
    model_loaded = loaded_flag if loaded_flag is not None else (llm_status == "loaded")

    require_model = _require_model_loaded_for_readyz(request)

    model_ok = True
    if require_model:
        model_ok = bool(model_loaded)

    overall_ready = db_ok and redis_ok and model_ok

    payload = {
        "status": "ready" if overall_ready else "not ready",
        "db": db_status,
        "redis": redis_status,
        "model_load_mode": mode,
        "require_model_ready": require_model,
        "model_loaded": bool(model_loaded),
        "llm": llm_status,
    }
    return JSONResponse(content=payload, status_code=200 if overall_ready else 503)


@router.get("/modelz")
async def modelz(request: Request):
    """Model readiness: does NOT trigger loading."""
    mode = _effective_model_load_mode(request)

    model_error = getattr(request.app.state, "model_error", None)

    loaded_flag = _model_loaded_flag(request)
    llm = get_llm_from_request(request)
    llm_status = "disabled" if mode == "off" else _llm_state(llm)
    model_loaded = loaded_flag if loaded_flag is not None else (llm_status == "loaded")

    if model_error:
        payload = {
            "status": "not ready",
            "model_load_mode": mode,
            "model_loaded": bool(model_loaded),
            "model_error": model_error,
            "llm": llm_status,
        }
        return JSONResponse(content=payload, status_code=503)

    payload = {
        "status": "ready" if model_loaded else "not ready",
        "model_load_mode": mode,
        "model_loaded": bool(model_loaded),
        "model_error": None,
        "llm": llm_status,
    }
    return JSONResponse(content=payload, status_code=200 if model_loaded else 503)