from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from llm_server.core.config import get_settings

Base = declarative_base()

_ENGINE: AsyncEngine | None = None
_SESSIONMAKER: async_sessionmaker[AsyncSession] | None = None
_ENGINE_OWNED: bool = False  # True if we created it in get_engine()


def _database_url() -> str:
    return get_settings().database_url


def get_engine() -> AsyncEngine:
    global _ENGINE, _ENGINE_OWNED
    if _ENGINE is None:
        _ENGINE = create_async_engine(
            _database_url(),
            echo=False,
            pool_pre_ping=True,
            future=True,
        )
        _ENGINE_OWNED = True
    return _ENGINE


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _SESSIONMAKER
    if _SESSIONMAKER is None:
        _SESSIONMAKER = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _SESSIONMAKER


def set_engine_for_tests(engine: AsyncEngine, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
    global _ENGINE, _SESSIONMAKER, _ENGINE_OWNED
    _ENGINE = engine
    _SESSIONMAKER = sessionmaker
    _ENGINE_OWNED = False  # injected, so we don't dispose it here


async def dispose_engine() -> None:
    """
    Dispose only if we created the engine (production path).
    Always reset module globals.
    """
    global _ENGINE, _SESSIONMAKER, _ENGINE_OWNED
    if _ENGINE is not None and _ENGINE_OWNED:
        await _ENGINE.dispose()
    _ENGINE = None
    _SESSIONMAKER = None
    _ENGINE_OWNED = False


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with get_sessionmaker()() as session:
        yield session


# Backwards-compatible alias (handy during cleanup)
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async for s in get_session():
        yield s


# Optional ergonomic helper
def new_session() -> AsyncSession:
    return get_sessionmaker()()