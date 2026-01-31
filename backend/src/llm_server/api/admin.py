# backend/src/llm_server/api/admin.py
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from llm_server.api.deps import clear_models_config_cache, get_api_key
from llm_server.core.config import get_settings
from llm_server.core.errors import AppError
from llm_server.db.models import ApiKey
from llm_server.db.session import get_session
from llm_server.reports import queries as report_q
from llm_server.reports import writer as report_w
from llm_server.services.inference import set_request_meta
from llm_server.services.llm import build_llm_from_settings
from llm_server.services.llm_registry import MultiModelManager
from llm_server.services.policy_decisions import get_policy_snapshot, reload_policy_snapshot

logger = logging.getLogger("llm_server.api.admin")

router = APIRouter(tags=["admin"])

_MODEL_LOAD_LOCK = asyncio.Lock()


# -------------------------------------------------------------------
# Models for responses (API contract)
# -------------------------------------------------------------------


class MeUsageResponse(BaseModel):
    api_key: str
    role: Optional[str]
    total_requests: int
    first_request_at: Optional[datetime]
    last_request_at: Optional[datetime]
    total_prompt_tokens: int
    total_completion_tokens: int


class AdminUsageRow(BaseModel):
    api_key: str
    name: Optional[str]
    role: Optional[str]
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    first_request_at: Optional[datetime]
    last_request_at: Optional[datetime]


class AdminUsageResponse(BaseModel):
    results: List[AdminUsageRow]


class AdminApiKeyInfo(BaseModel):
    key_prefix: str
    name: Optional[str]
    role: Optional[str]
    created_at: datetime
    disabled: bool


class AdminApiKeyListResponse(BaseModel):
    results: List[AdminApiKeyInfo]
    total: int
    limit: int
    offset: int


class AdminLogEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime

    api_key: Optional[str] = None
    route: str
    client_host: Optional[str] = None

    model_id: str
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # Full payloads (admin-only)
    prompt: str
    output: Optional[str] = None


class AdminLogsPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[AdminLogEntry]


class AdminModelStats(BaseModel):
    model_id: str
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    avg_latency_ms: float | None


class AdminStatsResponse(BaseModel):
    window_days: int
    since: datetime
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    avg_latency_ms: float | None
    per_model: list[AdminModelStats]


class AdminLoadModelRequest(BaseModel):
    model_id: Optional[str] = None


class AdminLoadModelResponse(BaseModel):
    ok: bool
    already_loaded: bool
    default_model: str
    models: list[str]
    

class AdminPolicySnapshotResponse(BaseModel):
    ok: bool
    model_id: Optional[str] = None
    enable_extract: Optional[bool] = None
    source_path: Optional[str] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = {}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _allowed_model_ids_from_settings(s) -> list[str]:
    allowed = getattr(s, "allowed_models", None)
    if isinstance(allowed, list) and allowed:
        return [str(x) for x in allowed if str(x).strip()]
    legacy = getattr(s, "all_model_ids", None) or []
    return [str(x) for x in legacy if str(x).strip()]


def _summarize_registry(llm_obj: Any, *, fallback_default: str) -> Tuple[str, list[str]]:
    if isinstance(llm_obj, MultiModelManager):
        return llm_obj.default_id, list(llm_obj.models.keys())

    if hasattr(llm_obj, "models") and hasattr(llm_obj, "default_id"):
        default_model = str(getattr(llm_obj, "default_id", "") or "") or fallback_default
        try:
            models_map = getattr(llm_obj, "models", {}) or {}
            model_ids = list(models_map.keys()) if isinstance(models_map, dict) else []
        except Exception:
            model_ids = []
        if not model_ids and default_model:
            model_ids = [default_model]
        return default_model, model_ids

    default_model = str(getattr(llm_obj, "model_id", "") or "") or fallback_default
    return default_model, [default_model] if default_model else []


async def _ensure_admin(api_key: ApiKey, session: AsyncSession) -> None:
    """
    Reload ApiKey with its Role in the current async session and enforce admin role.
    """
    db_key = await report_q.reload_key_with_role(session, api_key_id=api_key.id)
    role_name = db_key.role.name if db_key and db_key.role else None
    if role_name != "admin":
        raise AppError(code="forbidden", message="Admin privileges required", status_code=status.HTTP_403_FORBIDDEN)


# -------------------------------------------------------------------
# /v1/me/usage
# -------------------------------------------------------------------


@router.get("/v1/me/usage", response_model=MeUsageResponse)
async def get_my_usage(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    set_request_meta(request, route="/v1/me/usage", model_id="admin", cached=False)

    role_name = await report_q.fetch_role_name(session, api_key.role_id)
    usage = await report_q.get_me_usage(session, api_key_value=api_key.key, role_name=role_name)

    return MeUsageResponse(
        api_key=usage.api_key,
        role=usage.role,
        total_requests=usage.total_requests,
        first_request_at=usage.first_request_at,
        last_request_at=usage.last_request_at,
        total_prompt_tokens=usage.total_prompt_tokens,
        total_completion_tokens=usage.total_completion_tokens,
    )


# -------------------------------------------------------------------
# /v1/admin/usage
# -------------------------------------------------------------------


@router.get("/v1/admin/usage", response_model=AdminUsageResponse)
async def get_admin_usage(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    set_request_meta(request, route="/v1/admin/usage", model_id="admin", cached=False)
    await _ensure_admin(api_key, session)

    rows = await report_q.get_admin_usage(session)
    return AdminUsageResponse(
        results=[
            AdminUsageRow(
                api_key=r.api_key,
                name=r.name,
                role=r.role,
                total_requests=r.total_requests,
                total_prompt_tokens=r.total_prompt_tokens,
                total_completion_tokens=r.total_completion_tokens,
                first_request_at=r.first_request_at,
                last_request_at=r.last_request_at,
            )
            for r in rows
        ]
    )


# -------------------------------------------------------------------
# /v1/admin/keys
# -------------------------------------------------------------------


@router.get("/v1/admin/keys", response_model=AdminApiKeyListResponse)
async def list_api_keys(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    set_request_meta(request, route="/v1/admin/keys", model_id="admin", cached=False)
    await _ensure_admin(api_key, session)

    page = await report_q.list_api_keys(session, limit=limit, offset=offset)
    return AdminApiKeyListResponse(
        results=[
            AdminApiKeyInfo(
                key_prefix=x.key_prefix,
                name=x.name,
                role=x.role,
                created_at=x.created_at,
                disabled=x.disabled,
            )
            for x in page.items
        ],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


# -------------------------------------------------------------------
# /v1/admin/logs
# -------------------------------------------------------------------


@router.get("/v1/admin/logs", response_model=AdminLogsPage)
async def list_inference_logs(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
    model_id: Optional[str] = Query(default=None, description="Filter by model_id"),
    key: Optional[str] = Query(default=None, alias="api_key", description="Filter by API key value"),
    route: Optional[str] = Query(default=None, description="Filter by route, e.g. /v1/generate"),
    from_ts: Optional[datetime] = Query(default=None, description="Filter logs created_at >= this timestamp (ISO8601)"),
    to_ts: Optional[datetime] = Query(default=None, description="Filter logs created_at <= this timestamp (ISO8601)"),
    limit: int = Query(default=50, ge=1, le=200, description="Max number of rows to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
):
    set_request_meta(request, route="/v1/admin/logs", model_id="admin", cached=False)
    await _ensure_admin(api_key, session)

    page = await report_q.list_inference_logs(
        session,
        model_id=model_id,
        api_key_value=key,
        route=route,
        from_ts=from_ts,
        to_ts=to_ts,
        limit=limit,
        offset=offset,
    )

    items = [AdminLogEntry.model_validate(row) for row in page.items]
    return AdminLogsPage(total=page.total, limit=page.limit, offset=page.offset, items=items)


# -------------------------------------------------------------------
# /v1/admin/stats
# -------------------------------------------------------------------


@router.get("/v1/admin/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
    window_days: int = Query(30, ge=1, le=365),
):
    set_request_meta(request, route="/v1/admin/stats", model_id="admin", cached=False)
    await _ensure_admin(api_key, session)

    stats = await report_q.get_admin_stats(session, window_days=window_days)

    return AdminStatsResponse(
        window_days=stats.window_days,
        since=stats.since,
        total_requests=stats.total_requests,
        total_prompt_tokens=stats.total_prompt_tokens,
        total_completion_tokens=stats.total_completion_tokens,
        avg_latency_ms=stats.avg_latency_ms,
        per_model=[
            AdminModelStats(
                model_id=m.model_id,
                total_requests=m.total_requests,
                total_prompt_tokens=m.total_prompt_tokens,
                total_completion_tokens=m.total_completion_tokens,
                avg_latency_ms=m.avg_latency_ms,
            )
            for m in stats.per_model
        ],
    )


# -------------------------------------------------------------------
# OPTIONAL: /v1/admin/reports/summary
# -------------------------------------------------------------------


@router.get("/v1/admin/reports/summary")
async def admin_report_summary(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
    window_days: int = Query(30, ge=1, le=365),
    format: str = Query("text", pattern="^(text|json|md)$"),
):
    """
    A lightweight, presentation-oriented output that can be consumed by:
      - CLI tools
      - curl
      - GitHub Actions logs
      - docs / PR artifacts

    This does NOT replace /v1/admin/stats; it packages it for humans.
    """
    set_request_meta(request, route="/v1/admin/reports/summary", model_id="admin", cached=False)
    await _ensure_admin(api_key, session)

    stats = await report_q.get_admin_stats(session, window_days=window_days)

    stats_payload: Dict[str, Any] = {
        "window_days": stats.window_days,
        "since": stats.since,
        "total_requests": stats.total_requests,
        "total_prompt_tokens": stats.total_prompt_tokens,
        "total_completion_tokens": stats.total_completion_tokens,
        "avg_latency_ms": stats.avg_latency_ms,
    }
    per_model = [
        {
            "model_id": m.model_id,
            "total_requests": m.total_requests,
            "total_prompt_tokens": m.total_prompt_tokens,
            "total_completion_tokens": m.total_completion_tokens,
            "avg_latency_ms": m.avg_latency_ms,
        }
        for m in stats.per_model
    ]

    return report_w.render_admin_summary(stats_payload=stats_payload, per_model=per_model, fmt=format)


# -------------------------------------------------------------------
# /v1/admin/models/load
# -------------------------------------------------------------------


@router.post("/v1/admin/models/load", response_model=AdminLoadModelResponse)
async def admin_load_model(
    request: Request,
    body: AdminLoadModelRequest,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    set_request_meta(request, route="/v1/admin/models/load", model_id="admin", cached=False)
    await _ensure_admin(api_key, session)

    s = get_settings()

    async with _MODEL_LOAD_LOCK:
        app = request.app

        existing = getattr(app.state, "llm", None)
        model_loaded = bool(getattr(app.state, "model_loaded", False))
        model_error = getattr(app.state, "model_error", None)

        if existing is not None and model_loaded and not model_error:
            default_model, model_ids = _summarize_registry(existing, fallback_default=cast(str, getattr(s, "model_id", "")))
            return AdminLoadModelResponse(ok=True, already_loaded=True, default_model=default_model, models=model_ids)

        if body.model_id:
            allowed = _allowed_model_ids_from_settings(s)
            if body.model_id not in allowed:
                raise AppError(
                    code="model_not_allowed",
                    message=f"Model '{body.model_id}' not allowed.",
                    status_code=status.HTTP_400_BAD_REQUEST,
                    extra={"allowed": allowed},
                )

            s.model_id = body.model_id  # type: ignore[attr-defined]

            try:
                clear_models_config_cache()
            except Exception:
                pass

        app.state.model_error = None
        app.state.model_loaded = False

        try:
            llm = build_llm_from_settings()
            app.state.llm = llm

            if hasattr(llm, "ensure_loaded"):
                llm.ensure_loaded()
                app.state.model_loaded = True
            else:
                app.state.model_loaded = False

        except Exception as e:
            app.state.model_error = repr(e)
            app.state.model_loaded = False
            app.state.llm = None
            raise

        llm = app.state.llm
        default_model, model_ids = _summarize_registry(
            llm,
            fallback_default=cast(str, getattr(get_settings(), "model_id", "")),
        )

        return AdminLoadModelResponse(
            ok=True,
            already_loaded=False,
            default_model=default_model,
            models=model_ids,
        )
    

# -------------------------------------------------------------------
# /v1/admin/policy (inspect/reload)
# -------------------------------------------------------------------

@router.get("/v1/admin/policy", response_model=AdminPolicySnapshotResponse)
async def admin_get_policy_snapshot(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    set_request_meta(request, route="/v1/admin/policy", model_id="admin", cached=False)
    await _ensure_admin(api_key, session)

    snap = get_policy_snapshot(request)
    return AdminPolicySnapshotResponse(
        ok=bool(snap.ok),
        model_id=snap.model_id,
        enable_extract=snap.enable_extract,
        source_path=snap.source_path,
        error=snap.error,
        raw=snap.raw,
    )


@router.post("/v1/admin/policy/reload", response_model=AdminPolicySnapshotResponse)
async def admin_reload_policy_snapshot(
    request: Request,
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    set_request_meta(request, route="/v1/admin/policy/reload", model_id="admin", cached=False)
    await _ensure_admin(api_key, session)

    snap = reload_policy_snapshot(request)
    return AdminPolicySnapshotResponse(
        ok=bool(snap.ok),
        model_id=snap.model_id,
        enable_extract=snap.enable_extract,
        source_path=snap.source_path,
        error=snap.error,
        raw=snap.raw,
    )