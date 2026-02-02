from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.db.models import ApiKey, Role, RoleTable


@dataclass(frozen=True)
class CreateKeyInput:
    role: Optional[str] = Role.standard.value
    label: Optional[str] = "dev"
    quota_monthly: Optional[int] = None  # None => unlimited
    active: bool = True


async def upsert_role(session: AsyncSession, role_name: Optional[str]) -> Optional[RoleTable]:
    if not role_name:
        return None

    res = await session.execute(select(RoleTable).where(RoleTable.name == role_name))
    role = res.scalar_one_or_none()
    if role:
        return role

    role = RoleTable(name=role_name)
    session.add(role)
    await session.flush()
    return role


async def create_api_key(session: AsyncSession, inp: CreateKeyInput) -> ApiKey:
    # 64 hex chars
    key = secrets.token_hex(32)

    role = await upsert_role(session, inp.role)

    obj = ApiKey(
        key=key,
        label=inp.label,
        active=inp.active,
        role_id=role.id if role else None,
        quota_monthly=inp.quota_monthly,
        quota_used=0,
    )
    session.add(obj)
    await session.commit()
    return obj


async def list_api_keys(session: AsyncSession) -> list[tuple[ApiKey, Optional[str]]]:
    res = await session.execute(
        select(ApiKey, RoleTable.name)
        .join(RoleTable, ApiKey.role_id == RoleTable.id, isouter=True)
        .order_by(ApiKey.created_at.desc())
    )
    return res.all()