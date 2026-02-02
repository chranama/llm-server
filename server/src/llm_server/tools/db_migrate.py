from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Optional, Type

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from llm_server.db.models import ApiKey, CompletionCache, InferenceLog, RoleTable


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default


@dataclass(frozen=True)
class MigrateConfig:
    source_db_url: str
    target_db_url: str
    batch_size: int = 500
    skip_on_conflict: bool = True


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, future=True, echo=False)


async def copy_table_batched(
    src_session: AsyncSession,
    dst_session: AsyncSession,
    model: Type,
    *,
    batch_size: int = 500,
    skip_on_conflict: bool = True,
) -> int:
    """
    Naive batched copier for a single ORM model.

    Assumes:
    - target table already exists
    - dataset is modest (portfolio-scale)
    """
    offset = 0
    total = 0

    while True:
        result = await src_session.execute(select(model).offset(offset).limit(batch_size))
        rows = result.scalars().all()
        if not rows:
            break

        for row in rows:
            data = {c.name: getattr(row, c.name) for c in model.__table__.columns}
            dst_session.add(model(**data))

        try:
            await dst_session.commit()
        except IntegrityError:
            await dst_session.rollback()
            if not skip_on_conflict:
                raise

        total += len(rows)
        offset += batch_size
        print(f"Migrated {total} rows for {model.__name__}")

    print(f"Finished {model.__name__} (total={total})")
    return total


async def migrate(cfg: MigrateConfig) -> None:
    print(f"Source DB: {cfg.source_db_url}")
    print(f"Target DB: {cfg.target_db_url}")

    src_engine = make_engine(cfg.source_db_url)
    dst_engine = make_engine(cfg.target_db_url)

    # quick connectivity check
    async with src_engine.begin() as _, dst_engine.begin() as _:
        print("Connected to source and target databases.")

    async with AsyncSession(src_engine) as src_session, AsyncSession(dst_engine) as dst_session:
        # Order matters due to FK relationships
        await copy_table_batched(src_session, dst_session, RoleTable, batch_size=cfg.batch_size, skip_on_conflict=cfg.skip_on_conflict)
        await copy_table_batched(src_session, dst_session, ApiKey, batch_size=cfg.batch_size, skip_on_conflict=cfg.skip_on_conflict)
        await copy_table_batched(src_session, dst_session, CompletionCache, batch_size=cfg.batch_size, skip_on_conflict=cfg.skip_on_conflict)
        await copy_table_batched(src_session, dst_session, InferenceLog, batch_size=cfg.batch_size, skip_on_conflict=cfg.skip_on_conflict)

    await src_engine.dispose()
    await dst_engine.dispose()
    print("Migration complete.")


def migrate_from_env(
    *,
    source_default: str = "sqlite+aiosqlite:///./dev.db",
    target_default: str = "postgresql+asyncpg://llm:llm@localhost:5433/llm",
    batch_size: Optional[int] = None,
    skip_on_conflict: Optional[bool] = None,
) -> None:
    cfg = MigrateConfig(
        source_db_url=_env("SOURCE_DB_URL", source_default),
        target_db_url=_env("TARGET_DB_URL", target_default),
        batch_size=batch_size if batch_size is not None else int(_env("BATCH_SIZE", "500")),
        skip_on_conflict=skip_on_conflict if skip_on_conflict is not None else (_env("SKIP_ON_CONFLICT", "1") not in ("0", "false", "False")),
    )
    asyncio.run(migrate(cfg))