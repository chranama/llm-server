# tests/integration/conftest.py
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

# ============================================================
# Paths
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parents[1]

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

# ============================================================
# Config routing (root-level config/)
# ============================================================
APP_TEST_YAML = "config/server.test.yaml"
os.environ["APP_ROOT"] = str(REPO_ROOT)
os.environ["APP_CONFIG_PATH"] = APP_TEST_YAML


def _is_generate_only_module(request: pytest.FixtureRequest) -> bool:
    nid = (getattr(request.node, "nodeid", "") or "").lower()
    return "tests/integration/test_generate_integration.py" in nid


# ============================================================
# AnyIO backend
# ============================================================
@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


# ============================================================
# Assert config file exists
# ============================================================
@pytest.fixture(scope="session", autouse=True)
def _assert_test_config_file_exists():
    root = Path(os.environ["APP_ROOT"])
    cfg = (root / os.environ["APP_CONFIG_PATH"]).resolve()
    assert cfg.exists(), f"Missing test config: {cfg}"


# ============================================================
# Per-test env isolation
# ============================================================
@pytest.fixture(autouse=True)
def _integration_env_defaults(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    monkeypatch.setenv("APP_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("APP_CONFIG_PATH", APP_TEST_YAML)

    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("DEBUG", "0")
    monkeypatch.setenv("REDIS_ENABLED", "0")

    # These may be overridden by YAML in your app loader; that's OK.
    monkeypatch.setenv("MODEL_LOAD_MODE", "lazy")
    monkeypatch.setenv("REQUIRE_MODEL_READY", "0")
    monkeypatch.setenv("TOKEN_COUNTING", "0")

    # Capabilities: extract disabled ONLY for generate-only module
    if _is_generate_only_module(request):
        monkeypatch.setenv("ENABLE_EXTRACT", "0")
    else:
        monkeypatch.setenv("ENABLE_EXTRACT", "1")
    monkeypatch.setenv("ENABLE_GENERATE", "1")


# ============================================================
# Assert actual settings object is test config
# ============================================================
@pytest.fixture(autouse=True)
def _assert_test_config_loaded_per_test():
    from llm_server.core.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    assert s.env == "test", f"Expected env=test, got {s.env}"
    assert "test" in s.service_name.lower(), f"service_name not test-like: {s.service_name}"


# ============================================================
# DB helpers
# ============================================================
def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL must be set for integration tests. "
            "Example: postgresql+asyncpg://llm:llm@127.0.0.1:5433/llm"
        )
    return url


# ============================================================
# Engine / Session
# ============================================================
@pytest.fixture
async def test_engine():
    engine = create_async_engine(
        _database_url(),
        echo=False,
        future=True,
        poolclass=NullPool,
        pool_pre_ping=True,
    )
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def test_sessionmaker(test_engine):
    return async_sessionmaker(bind=test_engine, expire_on_commit=False, class_=AsyncSession)


# ============================================================
# Canonical DB injection + teardown reset
# ============================================================
@pytest.fixture(autouse=True)
async def patch_app_db_engine(test_engine, test_sessionmaker):
    """
    Canonical test hook: point llm_server.db.session at the per-test engine,
    then RESET it on teardown so no module-level state leaks across tests.
    """
    import llm_server.db.session as db_session

    db_session.set_engine_for_tests(test_engine, test_sessionmaker)
    try:
        yield
    finally:
        await db_session.dispose_engine()


# ============================================================
# Schema lifecycle
# ============================================================
@pytest.fixture(autouse=True)
async def create_schema(test_engine, patch_app_db_engine):
    from llm_server.db.session import Base
    from llm_server.db import models  # noqa: F401

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    try:
        yield
    finally:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


# ============================================================
# Fake LLM controls
# ============================================================
@pytest.fixture
def llm_outputs():
    return ["ok"]


@pytest.fixture
def llm_sleep_s():
    return 0.0


# ============================================================
# Force-load FakeLLM even if MODEL_LOAD_MODE=off
# ============================================================
def _force_llm_loaded(fake: Any) -> None:
    """
    Your config/server.test.yaml is forcing model_load_mode=off in some contexts.
    That makes /v1/generate and /v1/extract raise llm_not_loaded (503)
    before the endpoint does anything.

    This helper tries a few registry patterns to mark the FakeLLM as loaded,
    without knowing your exact internal implementation.
    """
    # 1) deps registry pattern
    try:
        import llm_server.api.deps as deps

        rl = getattr(deps, "_RL", None)
        if rl is not None:
            # method-based
            for meth in ("set", "set_llm", "set_model", "load", "set_loaded", "register"):
                fn = getattr(rl, meth, None)
                if callable(fn):
                    try:
                        fn(fake)
                        return
                    except TypeError:
                        # some signatures might be (name, model) etc.
                        try:
                            fn("default", fake)
                            return
                        except Exception:
                            pass

            # attribute-based
            for attr in ("llm", "_llm", "model", "_model"):
                if hasattr(rl, attr):
                    try:
                        setattr(rl, attr, fake)
                        # also mark loaded if such a flag exists
                        for flag in ("loaded", "_loaded", "is_loaded"):
                            if hasattr(rl, flag):
                                try:
                                    setattr(rl, flag, True)
                                except Exception:
                                    pass
                        return
                    except Exception:
                        pass
    except Exception:
        pass

    # 2) service-level registry pattern
    try:
        import llm_server.services.llm as llm_svc

        # common global singleton patterns
        for attr in ("LLM", "_LLM", "llm", "_llm", "MODEL", "_MODEL", "model", "_model"):
            if hasattr(llm_svc, attr):
                try:
                    setattr(llm_svc, attr, fake)
                    return
                except Exception:
                    pass

        # common flags
        for flag in ("LOADED", "_LOADED", "loaded", "_loaded", "MODEL_LOADED", "_MODEL_LOADED"):
            if hasattr(llm_svc, flag):
                try:
                    setattr(llm_svc, flag, True)
                except Exception:
                    pass
    except Exception:
        pass

    # 3) last resort: no-op if nothing matches
    return


# ============================================================
# App (lifespan will be handled by client fixture)
# ============================================================
@pytest.fixture
def app(monkeypatch, llm_outputs, llm_sleep_s, patch_app_db_engine, request: pytest.FixtureRequest):
    """
    App wired with FakeLLM. DB is already pointed at test_engine via patch_app_db_engine.

    Critical: even if model_load_mode is forced to OFF by YAML, we mark FakeLLM as loaded
    so /v1/generate works for integration tests.
    """
    from llm_server.core.config import get_settings

    get_settings.cache_clear()

    from fakes import FakeLLM
    import llm_server.api.deps as deps
    import llm_server.main as main
    import llm_server.services.llm as llm_svc

    fake = FakeLLM(outputs=list(llm_outputs), sleep_s=float(llm_sleep_s))

    # Ensure every builder returns our fake
    monkeypatch.setattr(deps, "build_llm_from_settings", lambda: fake, raising=True)
    deps._RL.clear()

    monkeypatch.setattr(main, "build_llm_from_settings", lambda: fake, raising=True)
    monkeypatch.setattr(llm_svc, "build_llm_from_settings", lambda: fake, raising=True)

    app = main.create_app()

    # The YAML is still printing model_load_mode=off in logs.
    # That's fine: we force the model as "loaded" for the tests.
    _force_llm_loaded(fake)

    return app


@pytest.fixture
async def client(app):
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            setattr(c, "app", app)
            yield c


# ============================================================
# App client factory (lifespan aware)
# ============================================================
@pytest.fixture
def app_client(monkeypatch, llm_outputs, llm_sleep_s, test_engine, test_sessionmaker):
    """
    Factory for scenarios where you want a fresh app/client pair per call.
    DB wiring is handled by the autouse patch_app_db_engine fixture.
    """
    from fakes import FakeLLM

    async def _make():
        from llm_server.core.config import get_settings

        get_settings.cache_clear()

        import llm_server.api.deps as deps
        import llm_server.main as main
        import llm_server.services.llm as llm_svc

        fake = FakeLLM(outputs=list(llm_outputs), sleep_s=float(llm_sleep_s))

        monkeypatch.setattr(deps, "build_llm_from_settings", lambda: fake, raising=True)
        deps._RL.clear()
        monkeypatch.setattr(main, "build_llm_from_settings", lambda: fake, raising=True)
        monkeypatch.setattr(llm_svc, "build_llm_from_settings", lambda: fake, raising=True)

        app = main.create_app()
        _force_llm_loaded(fake)

        @contextlib.asynccontextmanager
        async def _ctx():
            async with LifespanManager(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                    setattr(c, "app", app)
                    yield c

        return _ctx()

    return _make


# ============================================================
# API key helpers
# ============================================================
@pytest.fixture
async def api_key(test_sessionmaker):
    from llm_server.db.models import ApiKey
    import uuid

    key = f"test_{uuid.uuid4().hex}"
    async with test_sessionmaker() as session:
        session.add(ApiKey(key=key, active=True, quota_monthly=None, quota_used=0))
        await session.commit()
    return key


@pytest.fixture
async def auth_headers(api_key):
    return {"X-API-Key": api_key}


@pytest.fixture(autouse=True)
def _debug_settings_trace():
    from llm_server.core.config import get_settings
    import os

    get_settings.cache_clear()
    s = get_settings()

    print(
        "\n[debug] ENV MODEL_LOAD_MODE=",
        os.getenv("MODEL_LOAD_MODE"),
        "| settings.model_load_mode=",
        getattr(s, "model_load_mode", None),
    )