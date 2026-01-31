# tests/integration/test_reports_endpoints_integration.py
from __future__ import annotations

from datetime import datetime, timedelta, UTC
import uuid

import pytest


async def _mk_role_and_key(test_sessionmaker, *, role_name: str, active: bool = True) -> str:
    from llm_server.db.models import ApiKey, RoleTable

    key = f"test_{uuid.uuid4().hex}"

    async with test_sessionmaker() as session:
        role = RoleTable(name=role_name)
        session.add(role)
        await session.flush()

        session.add(
            ApiKey(
                key=key,
                active=active,
                role_id=role.id,
                quota_monthly=None,
                quota_used=0,
            )
        )
        await session.commit()

    return key


@pytest.fixture
async def admin_headers(test_sessionmaker):
    key = await _mk_role_and_key(test_sessionmaker, role_name="admin")
    return {"X-API-Key": key}


@pytest.fixture
async def standard_headers(test_sessionmaker):
    key = await _mk_role_and_key(test_sessionmaker, role_name="standard")
    return {"X-API-Key": key}


@pytest.fixture
async def other_headers(test_sessionmaker):
    key = await _mk_role_and_key(test_sessionmaker, role_name="standard")
    return {"X-API-Key": key}


async def _insert_logs(test_sessionmaker, rows: list[dict]):
    from llm_server.db.models import InferenceLog

    async with test_sessionmaker() as session:
        for r in rows:
            session.add(InferenceLog(**r))
        await session.commit()


@pytest.mark.anyio
async def test_me_usage_aggregates_only_my_key(
    client,
    test_sessionmaker,
    standard_headers,
    other_headers,
):
    now = datetime.now(UTC)
    my_key = standard_headers["X-API-Key"]
    other_key = other_headers["X-API-Key"]

    await _insert_logs(
        test_sessionmaker,
        [
            {
                "created_at": now - timedelta(days=1),
                "api_key": my_key,
                "request_id": "r1",
                "route": "/v1/generate",
                "client_host": "test",
                "model_id": "m1",
                "params_json": {"x": 1},
                "prompt": "p1",
                "output": "o1",
                "latency_ms": 10.0,
                "prompt_tokens": 11,
                "completion_tokens": 5,
            },
            {
                "created_at": now - timedelta(hours=1),
                "api_key": my_key,
                "request_id": "r2",
                "route": "/v1/extract",
                "client_host": "test",
                "model_id": "m1",
                "params_json": {"schema_id": "ticket_v1"},
                "prompt": "p2",
                "output": "{}",
                "latency_ms": 20.0,
                "prompt_tokens": 7,
                "completion_tokens": 3,
            },
            {
                "created_at": now - timedelta(hours=2),
                "api_key": other_key,
                "request_id": "r3",
                "route": "/v1/generate",
                "client_host": "test",
                "model_id": "m2",
                "params_json": {},
                "prompt": "p3",
                "output": "o3",
                "latency_ms": 30.0,
                "prompt_tokens": 100,
                "completion_tokens": 200,
            },
        ],
    )

    r = await client.get("/v1/me/usage", headers=standard_headers)
    assert r.status_code == 200

    body = r.json()
    assert body["api_key"] == my_key
    # role may be None depending on query join; don’t overconstrain
    assert body["total_requests"] == 2
    assert body["total_prompt_tokens"] == 11 + 7
    assert body["total_completion_tokens"] == 5 + 3
    assert body["first_request_at"] is not None
    assert body["last_request_at"] is not None


@pytest.mark.anyio
async def test_admin_usage_requires_admin(client, standard_headers):
    r = await client.get("/v1/admin/usage", headers=standard_headers)
    assert r.status_code == 403
    assert r.json().get("code") == "forbidden"


@pytest.mark.anyio
async def test_admin_usage_aggregates_all_keys(
    client,
    test_sessionmaker,
    admin_headers,
    standard_headers,
    other_headers,
):
    now = datetime.now(UTC)
    admin_key = admin_headers["X-API-Key"]
    k1 = standard_headers["X-API-Key"]
    k2 = other_headers["X-API-Key"]

    await _insert_logs(
        test_sessionmaker,
        [
            {
                "created_at": now - timedelta(days=2),
                "api_key": k1,
                "request_id": "a1",
                "route": "/v1/generate",
                "client_host": "test",
                "model_id": "m1",
                "params_json": {},
                "prompt": "p",
                "output": "o",
                "latency_ms": 1.0,
                "prompt_tokens": 2,
                "completion_tokens": 3,
            },
            {
                "created_at": now - timedelta(days=1),
                "api_key": k1,
                "request_id": "a2",
                "route": "/v1/extract",
                "client_host": "test",
                "model_id": "m1",
                "params_json": {},
                "prompt": "p",
                "output": "o",
                "latency_ms": 1.0,
                "prompt_tokens": 4,
                "completion_tokens": 5,
            },
            {
                "created_at": now - timedelta(hours=1),
                "api_key": k2,
                "request_id": "b1",
                "route": "/v1/generate",
                "client_host": "test",
                "model_id": "m2",
                "params_json": {},
                "prompt": "p",
                "output": "o",
                "latency_ms": 1.0,
                "prompt_tokens": 6,
                "completion_tokens": 7,
            },
            # admin has no logs; should still be fine
        ],
    )

    r = await client.get("/v1/admin/usage", headers=admin_headers)
    assert r.status_code == 200
    data = r.json()
    assert "results" in data

    by_key = {row["api_key"]: row for row in data["results"]}
    assert k1 in by_key
    assert k2 in by_key

    assert by_key[k1]["total_requests"] == 2
    assert by_key[k1]["total_prompt_tokens"] == 2 + 4
    assert by_key[k1]["total_completion_tokens"] == 3 + 5

    assert by_key[k2]["total_requests"] == 1
    assert by_key[k2]["total_prompt_tokens"] == 6
    assert by_key[k2]["total_completion_tokens"] == 7


@pytest.mark.anyio
async def test_admin_stats_totals_and_per_model(
    client,
    test_sessionmaker,
    admin_headers,
    standard_headers,
):
    now = datetime.now(UTC)
    k1 = standard_headers["X-API-Key"]

    await _insert_logs(
        test_sessionmaker,
        [
            {
                "created_at": now - timedelta(days=1),
                "api_key": k1,
                "request_id": "s1",
                "route": "/v1/generate",
                "client_host": "test",
                "model_id": "m1",
                "params_json": {},
                "prompt": "p",
                "output": "o",
                "latency_ms": 10.0,
                "prompt_tokens": 10,
                "completion_tokens": 20,
            },
            {
                "created_at": now - timedelta(hours=2),
                "api_key": k1,
                "request_id": "s2",
                "route": "/v1/generate",
                "client_host": "test",
                "model_id": "m2",
                "params_json": {},
                "prompt": "p",
                "output": "o",
                "latency_ms": 30.0,
                "prompt_tokens": 1,
                "completion_tokens": 2,
            },
        ],
    )

    r = await client.get("/v1/admin/stats?window_days=30", headers=admin_headers)
    assert r.status_code == 200
    data = r.json()

    assert data["total_requests"] == 2
    assert data["total_prompt_tokens"] == 11
    assert data["total_completion_tokens"] == 22
    assert data["per_model"] and isinstance(data["per_model"], list)

    per = {x["model_id"]: x for x in data["per_model"]}
    assert per["m1"]["total_requests"] == 1
    assert per["m2"]["total_requests"] == 1
    # avg_latency_ms can be None depending on query implementation; don’t overconstrain


@pytest.mark.anyio
async def test_admin_logs_filters_and_paging(
    client,
    test_sessionmaker,
    admin_headers,
    standard_headers,
):
    now = datetime.now(UTC)
    k1 = standard_headers["X-API-Key"]

    await _insert_logs(
        test_sessionmaker,
        [
            {
                "created_at": now - timedelta(hours=3),
                "api_key": k1,
                "request_id": "l1",
                "route": "/v1/generate",
                "client_host": "test",
                "model_id": "m1",
                "params_json": {},
                "prompt": "p1",
                "output": "o1",
                "latency_ms": 1.0,
                "prompt_tokens": 1,
                "completion_tokens": 1,
            },
            {
                "created_at": now - timedelta(hours=2),
                "api_key": k1,
                "request_id": "l2",
                "route": "/v1/extract",
                "client_host": "test",
                "model_id": "m1",
                "params_json": {},
                "prompt": "p2",
                "output": "o2",
                "latency_ms": 1.0,
                "prompt_tokens": 1,
                "completion_tokens": 1,
            },
            {
                "created_at": now - timedelta(hours=1),
                "api_key": k1,
                "request_id": "l3",
                "route": "/v1/generate",
                "client_host": "test",
                "model_id": "m2",
                "params_json": {},
                "prompt": "p3",
                "output": "o3",
                "latency_ms": 1.0,
                "prompt_tokens": 1,
                "completion_tokens": 1,
            },
        ],
    )

    # Filter by model_id=m1 and route=/v1/generate should yield exactly 1
    r = await client.get("/v1/admin/logs?model_id=m1&route=/v1/generate&limit=50&offset=0", headers=admin_headers)
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["model_id"] == "m1"
    assert data["items"][0]["route"] == "/v1/generate"

    # Paging sanity: limit=1 should return 1 item, total stays 3 for no filters
    r2 = await client.get("/v1/admin/logs?limit=1&offset=0", headers=admin_headers)
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["total"] == 3
    assert len(data2["items"]) == 1


@pytest.mark.anyio
async def test_admin_keys_lists_keys(client, admin_headers):
    r = await client.get("/v1/admin/keys?limit=50&offset=0", headers=admin_headers)
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert "total" in data
    assert isinstance(data["results"], list)
    # At least the admin key itself should exist
    assert data["total"] >= 1
    assert "key_prefix" in data["results"][0]
    assert "disabled" in data["results"][0]


@pytest.mark.anyio
async def test_admin_reports_summary_formats(client, admin_headers):
    for fmt in ("text", "json", "md"):
        r = await client.get(f"/v1/admin/reports/summary?window_days=30&format={fmt}", headers=admin_headers)
        assert r.status_code == 200
        # Format-specific shape: json should parse
        if fmt == "json":
            assert isinstance(r.json(), dict)
        else:
            assert isinstance(r.text, str)
            assert len(r.text) > 0