# backend/src/llm_server/reports/queries.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from llm_server.db.models import ApiKey, InferenceLog, RoleTable
from llm_server.reports.types import (
    AdminStats,
    AdminUsageRow,
    ApiKeyInfo,
    ApiKeyListPage,
    LogsPage,
    MeUsage,
    ModelStats,
)


async def fetch_role_name(session: AsyncSession, role_id: int | None) -> Optional[str]:
    if not role_id:
        return None
    role_row = await session.get(RoleTable, role_id)
    return role_row.name if role_row else None


async def get_me_usage(session: AsyncSession, *, api_key_value: str, role_name: Optional[str]) -> MeUsage:
    stmt = (
        select(
            func.count(InferenceLog.id),
            func.min(InferenceLog.created_at),
            func.max(InferenceLog.created_at),
            func.coalesce(func.sum(InferenceLog.prompt_tokens), 0),
            func.coalesce(func.sum(InferenceLog.completion_tokens), 0),
        )
        .where(InferenceLog.api_key == api_key_value)
    )
    res = await session.execute(stmt)
    (total_requests, first_request_at, last_request_at, total_prompt_tokens, total_completion_tokens) = res.one()

    return MeUsage(
        api_key=api_key_value,
        role=role_name,
        total_requests=int(total_requests or 0),
        first_request_at=first_request_at,
        last_request_at=last_request_at,
        total_prompt_tokens=int(total_prompt_tokens or 0),
        total_completion_tokens=int(total_completion_tokens or 0),
    )


async def get_admin_usage(session: AsyncSession) -> list[AdminUsageRow]:
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
        keys_stmt = select(ApiKey).options(selectinload(ApiKey.role)).where(ApiKey.key.in_(key_values))
        key_objs = (await session.execute(keys_stmt)).scalars().all()
        key_map = {k.key: k for k in key_objs}

    out: list[AdminUsageRow] = []
    for key_value, total_requests, first_at, last_at, total_prompt, total_completion in rows:
        key_obj = key_map.get(key_value)
        out.append(
            AdminUsageRow(
                api_key=key_value,
                name=getattr(key_obj, "name", None) if key_obj else None,
                role=key_obj.role.name if key_obj and key_obj.role else None,
                total_requests=int(total_requests or 0),
                total_prompt_tokens=int(total_prompt or 0),
                total_completion_tokens=int(total_completion or 0),
                first_request_at=first_at,
                last_request_at=last_at,
            )
        )

    return out


async def list_api_keys(
    session: AsyncSession,
    *,
    limit: int,
    offset: int,
) -> ApiKeyListPage:
    total_stmt = select(func.count(ApiKey.id))
    total = (await session.execute(total_stmt)).scalar_one()
    total_int = int(total or 0)

    stmt = (
        select(ApiKey)
        .options(selectinload(ApiKey.role))
        .order_by(ApiKey.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    keys = (await session.execute(stmt)).scalars().all()

    items: list[ApiKeyInfo] = []
    for k in keys:
        prefix = k.key[:8] if k.key else ""
        disabled_flag = bool(getattr(k, "disabled_at", None))
        items.append(
            ApiKeyInfo(
                key_prefix=prefix,
                name=getattr(k, "name", None),
                role=k.role.name if k.role else None,
                created_at=k.created_at,
                disabled=disabled_flag,
            )
        )

    return ApiKeyListPage(total=total_int, limit=limit, offset=offset, items=items)


async def list_inference_logs(
    session: AsyncSession,
    *,
    model_id: Optional[str],
    api_key_value: Optional[str],
    route: Optional[str],
    from_ts: Optional[datetime],
    to_ts: Optional[datetime],
    limit: int,
    offset: int,
) -> LogsPage:
    filters = []

    if model_id:
        filters.append(InferenceLog.model_id == model_id)
    if api_key_value:
        filters.append(InferenceLog.api_key == api_key_value)
    if route:
        filters.append(InferenceLog.route == route)
    if from_ts:
        filters.append(InferenceLog.created_at >= from_ts)
    if to_ts:
        filters.append(InferenceLog.created_at <= to_ts)

    # total
    count_stmt = select(func.count()).select_from(InferenceLog)
    if filters:
        count_stmt = count_stmt.where(*filters)

    total = await session.scalar(count_stmt)
    total_int = int(total or 0)

    # page
    stmt = select(InferenceLog)
    if filters:
        stmt = stmt.where(*filters)

    stmt = stmt.order_by(InferenceLog.created_at.desc()).offset(offset).limit(limit)
    rows = (await session.execute(stmt)).scalars().all()

    return LogsPage(total=total_int, limit=limit, offset=offset, items=list(rows))


async def get_admin_stats(session: AsyncSession, *, window_days: int) -> AdminStats:
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    global_stmt = (
        select(
            func.count(InferenceLog.id),
            func.coalesce(func.sum(InferenceLog.prompt_tokens), 0),
            func.coalesce(func.sum(InferenceLog.completion_tokens), 0),
            func.avg(InferenceLog.latency_ms),
        )
        .where(InferenceLog.created_at >= since)
    )

    total_requests, total_prompt_tokens, total_completion_tokens, avg_latency_ms = (
        await session.execute(global_stmt)
    ).one()

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

    per_model_rows = (await session.execute(per_model_stmt)).all()
    per_model_items: list[ModelStats] = []
    for mid, count, p_tokens, c_tokens, m_avg_latency in per_model_rows:
        per_model_items.append(
            ModelStats(
                model_id=mid,
                total_requests=int(count or 0),
                total_prompt_tokens=int(p_tokens or 0),
                total_completion_tokens=int(c_tokens or 0),
                avg_latency_ms=float(m_avg_latency) if m_avg_latency is not None else None,
            )
        )

    return AdminStats(
        window_days=window_days,
        since=since,
        total_requests=int(total_requests or 0),
        total_prompt_tokens=int(total_prompt_tokens or 0),
        total_completion_tokens=int(total_completion_tokens or 0),
        avg_latency_ms=float(avg_latency_ms) if avg_latency_ms is not None else None,
        per_model=per_model_items,
    )


async def reload_key_with_role(session: AsyncSession, *, api_key_id: int) -> ApiKey | None:
    """
    Utility used by API layer for admin gating without lazy-load issues.
    """
    result = await session.execute(select(ApiKey).options(joinedload(ApiKey.role)).where(ApiKey.id == api_key_id))
    return result.scalar_one_or_none()