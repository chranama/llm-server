# integrations/conftest.py
from __future__ import annotations

import os
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx
import pytest

from integrations.markers import (
    ModelsSnapshot,
    assert_full,
    assert_generate_only,
    MODE_FULL,
    MODE_GENERATE_ONLY,
    REQUIRES_API_KEY,
    REQUIRES_DB,
    REQUIRES_METRICS,
    REQUIRES_REDIS,
    TARGET_COMPOSE,
    TARGET_HOST,
    TARGET_K8S,
)

# ---------------------------------------------------------------------
# small env + url helpers
# ---------------------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v if isinstance(v, str) and v.strip() else default


def _normalize_base_url(x: str) -> str:
    return str(x or "").strip().rstrip("/")


# ---------------------------------------------------------------------
# pytest options + marker registration
# ---------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--base-url",
        action="store",
        default=_env("INTEGRATIONS_BASE_URL", "http://localhost:8000"),
        help="Base URL for API (default: env INTEGRATIONS_BASE_URL or http://localhost:8000)",
    )
    parser.addoption(
        "--api-key",
        action="store",
        default=_env("API_KEY", _env("INTEGRATIONS_API_KEY", "")),
        help="API key (default: env API_KEY or INTEGRATIONS_API_KEY)",
    )
    parser.addoption(
        "--mode",
        action="store",
        default=_env("INTEGRATIONS_MODE", "auto"),
        choices=["auto", MODE_GENERATE_ONLY, MODE_FULL],
        help="Which capability mode to enforce: auto|generate_only|full (default: auto)",
    )
    parser.addoption(
        "--timeout",
        action="store",
        default=float(_env("INTEGRATIONS_TIMEOUT", "30")),
        type=float,
        help="HTTP timeout seconds (default: env INTEGRATIONS_TIMEOUT or 30)",
    )


def pytest_configure(config: pytest.Config) -> None:
    # Mode markers
    config.addinivalue_line("markers", f"{MODE_GENERATE_ONLY}: tests for generate-only deployments")
    config.addinivalue_line("markers", f"{MODE_FULL}: tests for extract-enabled deployments")

    # Capability/infra markers
    config.addinivalue_line("markers", f"{REQUIRES_API_KEY}: test requires API key auth")
    config.addinivalue_line("markers", f"{REQUIRES_REDIS}: test assumes redis-enabled caching")
    config.addinivalue_line("markers", f"{REQUIRES_DB}: test assumes DB persistence is enabled")
    config.addinivalue_line("markers", f"{REQUIRES_METRICS}: test assumes /metrics is reachable")

    # Target markers (where the stack is running)
    config.addinivalue_line("markers", f"{TARGET_COMPOSE}: intended to run against docker compose stack")
    config.addinivalue_line("markers", f"{TARGET_HOST}: intended to run against host-run stack")
    config.addinivalue_line("markers", f"{TARGET_K8S}: intended to run against k8s stack")


# ---------------------------------------------------------------------
# session-scoped config fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="session")
def base_url(pytestconfig: pytest.Config) -> str:
    return _normalize_base_url(str(pytestconfig.getoption("--base-url")))


@pytest.fixture(scope="session")
def api_key(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--api-key") or "").strip()


@pytest.fixture(scope="session")
def http_timeout(pytestconfig: pytest.Config) -> float:
    return float(pytestconfig.getoption("--timeout"))


@pytest.fixture(scope="session")
def auth_headers(api_key: str) -> Dict[str, str]:
    if not api_key:
        return {}
    return {"X-API-Key": api_key}


@pytest.fixture(scope="session")
def integrations_mode(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--mode") or "auto").strip().lower()


@pytest.fixture(scope="session")
def sync_probe(base_url: str, http_timeout: float) -> Callable[[str], int]:
    """
    Synchronous probe for use in pytest hooks (hooks can't await).
    Returns HTTP status code, or 0 if connection error.
    """

    def _probe(path: str) -> int:
        try:
            with httpx.Client(base_url=base_url, timeout=http_timeout) as c:
                r = c.get(path)
                return int(r.status_code)
        except Exception:
            return 0

    return _probe


# ---------------------------------------------------------------------
# shared httpx client + /v1/models snapshot
# ---------------------------------------------------------------------


@pytest.fixture(scope="session")
async def client(base_url: str, auth_headers: Dict[str, str], http_timeout: float) -> httpx.AsyncClient:
    """
    Shared client for all tests. Adds X-API-Key header if provided.
    """
    timeout = httpx.Timeout(http_timeout)
    async with httpx.AsyncClient(base_url=base_url, headers=auth_headers, timeout=timeout) as c:
        yield c


@pytest.fixture(scope="session")
async def models_snapshot(client: httpx.AsyncClient) -> ModelsSnapshot:
    """
    Single source of truth for capability checks: /v1/models.
    """
    r = await client.get("/v1/models")
    r.raise_for_status()
    return ModelsSnapshot.from_json(r.json())


@pytest.fixture(scope="session")
def assert_mode(models_snapshot: ModelsSnapshot, integrations_mode: str) -> bool:
    """
    Enforce the live stack matches --mode intent.
    - generate_only => assert generate-only contract via /v1/models
    - full         => assert full contract via /v1/models
    - auto         => no assertion
    """
    if integrations_mode == MODE_GENERATE_ONLY:
        assert_generate_only(models_snapshot)
    elif integrations_mode == MODE_FULL:
        assert_full(models_snapshot)
    return True


# ---------------------------------------------------------------------
# mode-based skipping
# ---------------------------------------------------------------------


def _test_has_marker(item: pytest.Item, name: str) -> bool:
    return item.get_closest_marker(name) is not None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Enforce mode selection:
      - If --mode=generate_only: skip tests marked full
      - If --mode=full: skip tests marked generate_only
      - If --mode=auto: do nothing
    """
    mode = str(config.getoption("--mode") or "auto").strip().lower()
    if mode == "auto":
        return

    for item in items:
        if mode == MODE_GENERATE_ONLY and _test_has_marker(item, MODE_FULL):
            item.add_marker(pytest.mark.skip(reason="--mode=generate_only (skipping full tests)"))
        if mode == MODE_FULL and _test_has_marker(item, MODE_GENERATE_ONLY):
            item.add_marker(pytest.mark.skip(reason="--mode=full (skipping generate-only tests)"))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """
    Enforce 'requires_*' markers that can be checked without awaiting.
    """
    # Auth gating
    if item.get_closest_marker(REQUIRES_API_KEY) is not None:
        api_key = str(item.config.getoption("--api-key") or "").strip()
        if not api_key:
            pytest.skip("API_KEY not provided (set API_KEY or pass --api-key)")

    # Metrics gating (best-effort: skip if /metrics isn't reachable)
    if item.get_closest_marker(REQUIRES_METRICS) is not None:
        probe = item.config._store.get(("integrations", "sync_probe"))  # type: ignore[attr-defined]
        # If we can't access the probe via store, just don't gate.
        if callable(probe):
            status = int(probe("/metrics"))
            if status in (0, 404):
                pytest.skip("/metrics not reachable (requires_metrics)")

    # NOTE: redis/db gating is intentionally not enforced here, because:
    # - You may not have deterministic probes.
    # - Better to skip inside the specific tests when you can detect capability via /v1/models,
    #   env, or a known endpoint/metric.


# ---------------------------------------------------------------------
# Convenience checks + helpers
# ---------------------------------------------------------------------


@pytest.fixture
def require_api_key(api_key: str) -> bool:
    """
    Use in tests that want a strict runtime check rather than a marker.
    """
    if not api_key:
        pytest.skip("API_KEY not provided (set API_KEY or pass --api-key)")
    return True


# ---- Module-level helpers (so older tests that did `from integrations.conftest import get_json` keep working) ----


async def get_json(
    client: httpx.AsyncClient,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    r = await client.get(path, params=params, headers=headers)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise AssertionError(f"Expected JSON object from GET {path}, got {type(data).__name__}")
    return data


async def post_json(
    client: httpx.AsyncClient,
    path: str,
    payload: Dict[str, Any],
    *,
    headers: Optional[Dict[str, str]] = None,
) -> httpx.Response:
    """
    Returns the raw Response so callers can assert status codes for negative tests.
    If you want a dict + raise_for_status, use the post_json_dict fixture below.
    """
    return await client.post(path, json=payload, headers=headers)


# ---- Pytest fixtures that return callables (nicer in new tests) ----


@pytest.fixture
def get_json_dict(client: httpx.AsyncClient) -> Callable[..., Awaitable[Dict[str, Any]]]:
    async def _get(
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        return await get_json(client, path, params=params, headers=headers)

    return _get


@pytest.fixture
def post_json_dict(client: httpx.AsyncClient) -> Callable[..., Awaitable[Dict[str, Any]]]:
    async def _post(
        path: str,
        payload: Dict[str, Any],
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        r = await client.post(path, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise AssertionError(f"Expected JSON object from POST {path}, got {type(data).__name__}")
        return data

    return _post


@pytest.fixture(autouse=True, scope="session")
def _stash_sync_probe_for_hooks(sync_probe, request: pytest.FixtureRequest) -> None:
    """
    Make sync_probe accessible inside pytest_runtest_setup without imports.
    This avoids creating a second client there.
    """
    request.config._store[("integrations", "sync_probe")] = sync_probe  # type: ignore[attr-defined]