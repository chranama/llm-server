# src/llm_server/api/admin.py
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

from llm_server.api.deps import get_api_key
from llm_server.core.config import settings
from llm_server.db.models import ApiKey, InferenceLog, RoleTable
from llm_server.db.session import get_session

logger = logging.getLogger("llm_server.api.admin")

router = APIRouter(tags=["admin"])


# -------------------------------------------------------------------
# Models for responses
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


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

async def _ensure_admin(api_key: ApiKey, session: AsyncSession) -> None:
    """
    Reload the ApiKey with its Role in the current async session and
    enforce that the caller has the 'admin' role.
    """
    result = await session.execute(
        select(ApiKey)
        .options(joinedload(ApiKey.role))
        .where(ApiKey.id == api_key.id)
    )
    db_key = result.scalar_one_or_none()

    role_name = db_key.role.name if db_key and db_key.role else None
    if role_name != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")


# -------------------------------------------------------------------
# /v1/me/usage
# -------------------------------------------------------------------

@router.get("/v1/me/usage", response_model=MeUsageResponse)
async def get_my_usage(
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    # Fetch role name without lazy-loading problems
    role_row = await session.get(RoleTable, api_key.role_id) if api_key.role_id else None
    role_name = role_row.name if role_row else None

    # Aggregate this key's usage from inference_logs
    stmt = (
        select(
            func.count(InferenceLog.id),
            func.min(InferenceLog.created_at),
            func.max(InferenceLog.created_at),
            func.coalesce(func.sum(InferenceLog.prompt_tokens), 0),
            func.coalesce(func.sum(InferenceLog.completion_tokens), 0),
        )
        .where(InferenceLog.api_key == api_key.key)
    )
    res = await session.execute(stmt)
    (
        total_requests,
        first_request_at,
        last_request_at,
        total_prompt_tokens,
        total_completion_tokens,
    ) = res.one()

    return MeUsageResponse(
        api_key=api_key.key,
        role=role_name,
        total_requests=total_requests or 0,
        first_request_at=first_request_at,
        last_request_at=last_request_at,
        total_prompt_tokens=total_prompt_tokens or 0,
        total_completion_tokens=total_completion_tokens or 0,
    )


# -------------------------------------------------------------------
# /v1/admin/usage
# -------------------------------------------------------------------

@router.get("/v1/admin/usage", response_model=AdminUsageResponse)
async def get_admin_usage(
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
):
    await _ensure_admin(api_key, session)

    # Aggregate stats per api_key
    stmt = (
        select(
            InferenceLog.api_key,
            func.count(InferenceLog.id),
            func.min(InferenceLog.created_at),
            func.max(InferenceLog.created_at),
            func.coalesce(func.sum(InferenceLog.prompt_tokens), 0),
            func.coalesce(func.sum(InferenceLog.completion_tokens), 0),
        )
        .group_by(InferenceLog.api_key)
    )

    rows = (await session.execute(stmt)).all()

    # Fetch key metadata in one shot
    key_values = [r[0] for r in rows if r[0] is not None]
    key_map: dict[str, ApiKey] = {}
    if key_values:
        keys_stmt = (
            select(ApiKey)
            .options(selectinload(ApiKey.role))
            .where(ApiKey.key.in_(key_values))
        )
        key_objs = (await session.execute(keys_stmt)).scalars().all()
        key_map = {k.key: k for k in key_objs}

    results: List[AdminUsageRow] = []
    for key_value, total_requests, first_at, last_at, total_prompt, total_completion in rows:
        key_obj = key_map.get(key_value)
        results.append(
            AdminUsageRow(
                api_key=key_value,
                name=getattr(key_obj, "name", None) if key_obj else None,
                role=key_obj.role.name if key_obj and key_obj.role else None,
                total_requests=total_requests or 0,
                total_prompt_tokens=total_prompt or 0,
                total_completion_tokens=total_completion or 0,
                first_request_at=first_at,
                last_request_at=last_at,
            )
        )

    return AdminUsageResponse(results=results)


# -------------------------------------------------------------------
# /v1/admin/keys
# -------------------------------------------------------------------

@router.get("/v1/admin/keys", response_model=AdminApiKeyListResponse)
async def list_api_keys(
    api_key: ApiKey = Depends(get_api_key),
    session: AsyncSession = Depends(get_session),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    List API keys and their metadata.

    - Admin-only
    - Does NOT return full key values, only a key prefix for identification.
    """
    await _ensure_admin(api_key, session)

    # Total count
    total_stmt = select(func.count(ApiKey.id))
    total = (await session.execute(total_stmt)).scalar_one()

    # Page of keys, eager-load role to avoid lazy-load issues
    stmt = (
        select(ApiKey)
        .options(selectinload(ApiKey.role))
        .order_by(ApiKey.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    keys = (await session.execute(stmt)).scalars().all()

    results: List[AdminApiKeyInfo] = []
    for k in keys:
        # Safe prefix only; don't leak full secret
        prefix = k.key[:8] if k.key else ""
        disabled_flag = bool(getattr(k, "disabled_at", None))

        results.append(
            AdminApiKeyInfo(
                key_prefix=prefix,
                name=getattr(k, "name", None),
                role=k.role.name if k.role else None,
                created_at=k.created_at,
                disabled=disabled_flag,
            )
        )

    return AdminApiKeyListResponse(
        results=results,
        total=total or 0,
        limit=limit,
        offset=offset,
    )

# -------------------------------------------------------------------
# /v1/admin/logs
# -------------------------------------------------------------------
@router.get("/v1/admin/logs", response_model=AdminLogsPage)
async def list_inference_logs(
    api_key=Depends(get_api_key),
    session=Depends(get_session),
    # Filters
    model_id: Optional[str] = Query(
        default=None,
        description="Filter by model_id",
    ),
    key: Optional[str] = Query(
        default=None,
        alias="api_key",
        description="Filter by API key value",
    ),
    route: Optional[str] = Query(
        default=None,
        description="Filter by route, e.g. /v1/generate",
    ),
    from_ts: Optional[datetime] = Query(
        default=None,
        description="Filter logs created_at >= this timestamp (ISO8601)",
    ),
    to_ts: Optional[datetime] = Query(
        default=None,
        description="Filter logs created_at <= this timestamp (ISO8601)",
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
        description="Max number of rows to return",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Offset for pagination",
    ),
):
    """
    Admin-only: list inference logs with basic filters + pagination.
    """

    # Ensure caller is admin
    await _ensure_admin(api_key, session)

    # Build filter list once so we can reuse for count + data query
    filters = []

    if model_id:
        filters.append(InferenceLog.model_id == model_id)

    if key:
        filters.append(InferenceLog.api_key == key)

    if route:
        filters.append(InferenceLog.route == route)

    if from_ts:
        filters.append(InferenceLog.created_at >= from_ts)

    if to_ts:
        filters.append(InferenceLog.created_at <= to_ts)

    # ---- total count ----
    count_stmt = select(func.count()).select_from(InferenceLog)
    if filters:
        count_stmt = count_stmt.where(*filters)

    total = await session.scalar(count_stmt)
    total = int(total or 0)

    # ---- page query ----
    stmt = (
        select(InferenceLog)
        .where(*filters) if filters else select(InferenceLog)
    )
    stmt = (
        stmt.order_by(InferenceLog.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    result = await session.execute(stmt)
    rows = result.scalars().all()

    items = [AdminLogEntry.from_orm(row) for row in rows]

    return AdminLogsPage(
        total=total,
        limit=limit,
        offset=offset,
        items=items,
    )

# -------------------------------------------------------------------
# /v1/admin/stats
# -------------------------------------------------------------------
@router.get("/v1/admin/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    window_days: int = 30,
    session: AsyncSession = Depends(get_session),
    api_key=Depends(get_api_key),
):
    """
    Global usage stats over a sliding time window (admin-only).

    - Aggregate totals from `inference_logs`
    - Per-model breakdown
    """
    _ensure_admin(api_key, session)

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    # ---- Global aggregates ----
    global_stmt = (
        select(
            func.count(InferenceLog.id),
            func.coalesce(func.sum(InferenceLog.prompt_tokens), 0),
            func.coalesce(func.sum(InferenceLog.completion_tokens), 0),
            func.avg(InferenceLog.latency_ms),
        )
        .where(InferenceLog.created_at >= since)
    )

    global_row = (await session.execute(global_stmt)).one()
    (
        total_requests,
        total_prompt_tokens,
        total_completion_tokens,
        avg_latency_ms,
    ) = global_row

    # ---- Per-model aggregates ----
    per_model_stmt = (
        select(
            InferenceLog.model_id,
            func.count(InferenceLog.id),
            func.coalesce(func.sum(InferenceLog.prompt_tokens), 0),
            func.coalesce(func.sum(InferenceLog.completion_tokens), 0),
            func.avg(InferenceLog.latency_ms),
        )
        .where(InferenceLog.created_at >= since)
        .group_by(InferenceLog.model_id)
    )

    per_model_rows = await session.execute(per_model_stmt)
    per_model_items: list[AdminModelStats] = []

    for model_id, count, p_tokens, c_tokens, m_avg_latency in per_model_rows:
        per_model_items.append(
            AdminModelStats(
                model_id=model_id,
                total_requests=count,
                total_prompt_tokens=p_tokens or 0,
                total_completion_tokens=c_tokens or 0,
                avg_latency_ms=float(m_avg_latency) if m_avg_latency is not None else None,
            )
        )

    return AdminStatsResponse(
        window_days=window_days,
        since=since,
        total_requests=total_requests or 0,
        total_prompt_tokens=total_prompt_tokens or 0,
        total_completion_tokens=total_completion_tokens or 0,
        avg_latency_ms=float(avg_latency_ms) if avg_latency_ms is not None else None,
        per_model=per_model_items,
    )