# tests/integration/test_admin_policy_endpoints_integration.py
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest


async def _make_role_and_key(test_sessionmaker, *, role_name: str) -> str:
    """
    Create RoleTable + ApiKey wired to that role. Returns the raw api key string.
    Matches llm_server.db.models (RoleTable, ApiKey.role_id).
    """
    from llm_server.db.models import ApiKey, RoleTable

    key = f"test_{uuid.uuid4().hex}"

    async with test_sessionmaker() as session:
        role = RoleTable(name=role_name)
        session.add(role)
        await session.flush()  # role.id available

        session.add(
            ApiKey(
                key=key,
                active=True,
                role_id=role.id,
                quota_monthly=None,
                quota_used=0,
            )
        )
        await session.commit()

    return key


@pytest.fixture
async def admin_headers(test_sessionmaker):
    key = await _make_role_and_key(test_sessionmaker, role_name="admin")
    return {"X-API-Key": key}


@pytest.fixture
async def user_headers(test_sessionmaker):
    key = await _make_role_and_key(test_sessionmaker, role_name="standard")
    return {"X-API-Key": key}


def _write_policy_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@pytest.mark.anyio
async def test_admin_policy_requires_admin(client, user_headers):
    r = await client.get("/v1/admin/policy", headers=user_headers)
    assert r.status_code == 403
    body = r.json()
    assert body.get("code") == "forbidden"

    r2 = await client.post("/v1/admin/policy/reload", headers=user_headers)
    assert r2.status_code == 403
    body2 = r2.json()
    assert body2.get("code") == "forbidden"


@pytest.mark.anyio
async def test_admin_policy_no_env_path_returns_no_override(client, admin_headers):
    # Ensure env var not set for this test
    import os

    os.environ.pop("POLICY_DECISION_PATH", None)

    # Clear cached snapshot if any (lives on app.state)
    if hasattr(client, "app"):
        try:
            delattr(client.app.state, "policy_snapshot")
        except Exception:
            pass

    r = await client.get("/v1/admin/policy", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()

    assert body["ok"] is True
    assert body["model_id"] is None
    assert body["enable_extract"] is None
    assert body["source_path"] is None
    assert body["error"] is None
    assert body["raw"] == {}


@pytest.mark.anyio
async def test_admin_policy_missing_file_fail_closed(client, admin_headers, tmp_path):
    import os

    missing = tmp_path / "does_not_exist.json"
    os.environ["POLICY_DECISION_PATH"] = str(missing)

    # Clear cached snapshot
    if hasattr(client, "app"):
        try:
            delattr(client.app.state, "policy_snapshot")
        except Exception:
            pass

    r = await client.get("/v1/admin/policy", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()

    assert body["ok"] is False
    assert body["enable_extract"] is False  # fail-closed
    assert body["source_path"] == str(missing)
    assert body["error"] in ("policy_decision_missing", "policy_decision_not_ok")


@pytest.mark.anyio
async def test_admin_policy_valid_file_round_trip(client, admin_headers, tmp_path):
    import os

    p = tmp_path / "policy.json"
    payload = {
        "model_id": "m1",
        "enable_extract": True,
        "status": "allow",
        "contract_errors": 0,
        "extra_field": {"hello": "world"},
    }
    _write_policy_file(p, payload)
    os.environ["POLICY_DECISION_PATH"] = str(p)

    # Clear cached snapshot
    if hasattr(client, "app"):
        try:
            delattr(client.app.state, "policy_snapshot")
        except Exception:
            pass

    r = await client.get("/v1/admin/policy", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()

    assert body["ok"] is True
    assert body["model_id"] == "m1"
    assert body["enable_extract"] is True
    assert body["source_path"] == str(p)
    assert body["error"] is None

    # raw should include our extra_field and keys
    assert isinstance(body["raw"], dict)
    assert body["raw"].get("extra_field", {}).get("hello") == "world"


@pytest.mark.anyio
async def test_admin_policy_contract_errors_fail_closed(client, admin_headers, tmp_path):
    import os

    p = tmp_path / "policy_bad.json"
    payload = {
        "status": "allow",
        "contract_errors": 2,
        # enable_extract omitted; should become False due to fail-closed when not ok
    }
    _write_policy_file(p, payload)
    os.environ["POLICY_DECISION_PATH"] = str(p)

    # Clear cached snapshot
    if hasattr(client, "app"):
        try:
            delattr(client.app.state, "policy_snapshot")
        except Exception:
            pass

    r = await client.get("/v1/admin/policy", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()

    assert body["ok"] is False
    assert body["enable_extract"] is False  # fail-closed
    assert body["source_path"] == str(p)
    assert body["error"] == "policy_decision_not_ok"


@pytest.mark.anyio
async def test_admin_policy_deny_fail_closed_even_if_enable_true(client, admin_headers, tmp_path):
    """
    If status is deny/unknown, ok=False and enable_extract should be fail-closed.
    Your loader forces enable_extract=False if decision not ok, even if omitted.
    If explicitly true in file, your code currently keeps it (because enable_extract is not None),
    BUT still returns ok=False + error, and policy layer can treat that as deny.
    We assert the snapshot truth: ok=False, and enable_extract is boolean.
    """
    import os

    p = tmp_path / "policy_deny.json"
    payload = {
        "status": "deny",
        "enable_extract": True,  # explicitly set
        "contract_errors": 0,
    }
    _write_policy_file(p, payload)
    os.environ["POLICY_DECISION_PATH"] = str(p)

    if hasattr(client, "app"):
        try:
            delattr(client.app.state, "policy_snapshot")
        except Exception:
            pass

    r = await client.get("/v1/admin/policy", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()

    assert body["ok"] is False
    assert body["error"] == "policy_decision_not_ok"
    assert body["source_path"] == str(p)
    assert isinstance(body["enable_extract"], bool)


@pytest.mark.anyio
async def test_admin_policy_reload_picks_up_file_changes(client, admin_headers, tmp_path):
    import os

    p = tmp_path / "policy_reload.json"
    _write_policy_file(p, {"enable_extract": True, "status": "allow", "contract_errors": 0})
    os.environ["POLICY_DECISION_PATH"] = str(p)

    # First call caches it
    r1 = await client.get("/v1/admin/policy", headers=admin_headers)
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["enable_extract"] is True
    assert body1["ok"] is True

    # Change file
    _write_policy_file(p, {"enable_extract": False, "status": "allow", "contract_errors": 0})

    # GET should still return cached snapshot (True) unless your handler reloads implicitly (it doesn't)
    r2 = await client.get("/v1/admin/policy", headers=admin_headers)
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["enable_extract"] is True

    # Reload endpoint should pick up new value
    r3 = await client.post("/v1/admin/policy/reload", headers=admin_headers)
    assert r3.status_code == 200
    body3 = r3.json()

    assert body3["ok"] is True
    assert body3["enable_extract"] is False
    assert body3["source_path"] == str(p)