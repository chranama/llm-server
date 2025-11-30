# scripts/migrate_data.py
from __future__ import annotations

import os
import asyncio
from typing import AsyncIterator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from llm_server.db.models import (
    RoleTable,
    ApiKey,
    InferenceLog,
    CompletionCache,
)
from sqlalchemy.exc import IntegrityError


# ---------- Config helpers ----------

def get_env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    if not value:
        return default
    return value


# e.g.
#   SOURCE_DB_URL=sqlite+aiosqlite:///./dev.db
#   TARGET_DB_URL=postgresql+asyncpg://llm:llm@localhost:5433/llm
SOURCE_DB_URL = get_env_or_default(
    "SOURCE_DB_URL",
    "sqlite+aiosqlite:///./dev.db",
)
TARGET_DB_URL = get_env_or_default(
    "TARGET_DB_URL",
    "postgresql+asyncpg://llm:llm@localhost:5433/llm",
)


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, future=True, echo=False)


# ---------- Generic batched copier ----------

async def copy_table_batched(
    src_session: AsyncSession,
    dst_session: AsyncSession,
    model,
    batch_size: int = 500,
    skip_on_conflict: bool = True,
) -> None:
    """
    Naive batched copier for a single ORM model.

    Assumes:
    - target table already exists
    - data set is modest in size (OK for portfolio project)
    """

    offset = 0
    total = 0

    while True:
        result = await src_session.execute(
            select(model).offset(offset).limit(batch_size)
        )
        rows = result.scalars().all()
        if not rows:
            break

        for row in rows:
            # Build a new instance detached from source session
            data = {c.name: getattr(row, c.name) for c in model.__table__.columns}
            obj = model(**data)

            dst_session.add(obj)

        try:
            await dst_session.commit()
        except IntegrityError:
            # If we re-run migration or partial data exists,
            # we can optionally ignore duplicates.
            await dst_session.rollback()
            if not skip_on_conflict:
                raise

        total += len(rows)
        offset += batch_size
        print(f"Migrated {total} rows for {model.__name__}")

    print(f"Finished {model.__name__} (total={total})")


# ---------- Top-level migration ----------

async def migrate() -> None:
    print(f"Source DB: {SOURCE_DB_URL}")
    print(f"Target DB: {TARGET_DB_URL}")

    src_engine = make_engine(SOURCE_DB_URL)
    dst_engine = make_engine(TARGET_DB_URL)

    async with src_engine.begin() as src_conn, dst_engine.begin() as dst_conn:
        # Just open connections to verify they are reachable
        print("Connected to source and target databases.")

    async with AsyncSession(src_engine) as src_session, AsyncSession(dst_engine) as dst_session:
        # Order matters due to FK relationships (Role → ApiKey → CompletionCache/InferenceLog)
        await copy_table_batched(src_session, dst_session, RoleTable)
        await copy_table_batched(src_session, dst_session, ApiKey)
        await copy_table_batched(src_session, dst_session, CompletionCache)
        await copy_table_batched(src_session, dst_session, InferenceLog)

    await src_engine.dispose()
    await dst_engine.dispose()
    print("Migration complete.")


def main():
    asyncio.run(migrate())


if __name__ == "__main__":
    main()