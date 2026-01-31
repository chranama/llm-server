#  integrations/lib/db.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import httpx


@dataclass(frozen=True)
class AdminLogsPage:
    total: int
    limit: int
    offset: int
    items: list[Dict[str, Any]]  # AdminLogEntry-like dicts


def _parse_iso_dt(x: Any) -> Optional[datetime]:
    if not isinstance(x, str) or not x.strip():
        return None
    try:
        # Python 3.12: fromisoformat supports "YYYY-MM-DDTHH:MM:SS(.ffffff)(+00:00)"
        return datetime.fromisoformat(x.replace("Z", "+00:00"))
    except Exception:
        return None


async def get_me_usage(client: httpx.AsyncClient) -> Dict[str, Any]:
    """
    Calls GET /v1/me/usage.
    Requires an API key, but NOT admin privileges.
    """
    r = await client.get("/v1/me/usage")
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise AssertionError("/v1/me/usage must return JSON object")
    return data


async def get_admin_logs(
    *,
    client: httpx.AsyncClient,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    route: Optional[str] = None,
    from_ts: Optional[str] = None,
    to_ts: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> AdminLogsPage:
    """
    Calls GET /v1/admin/logs with filters.
    Admin-only.
    """
    params: Dict[str, Any] = {"limit": int(limit), "offset": int(offset)}
    if model_id:
        params["model_id"] = model_id
    if api_key:
        params["api_key"] = api_key
    if route:
        params["route"] = route
    if from_ts:
        params["from_ts"] = from_ts
    if to_ts:
        params["to_ts"] = to_ts

    r = await client.get("/v1/admin/logs", params=params)
    r.raise_for_status()

    data = r.json()
    if not isinstance(data, dict):
        raise AssertionError("/v1/admin/logs must return JSON object")

    total = int(data.get("total") or 0)
    lim = int(data.get("limit") or limit)
    off = int(data.get("offset") or offset)
    items = data.get("items") or []
    if not isinstance(items, list):
        raise AssertionError("/v1/admin/logs 'items' must be a list")

    # Ensure each item is a dict-ish
    out_items: list[Dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            out_items.append(it)
    return AdminLogsPage(total=total, limit=lim, offset=off, items=out_items)


async def get_admin_stats(
    *,
    client: httpx.AsyncClient,
    window_days: int = 30,
) -> Dict[str, Any]:
    """
    Calls GET /v1/admin/stats.
    Admin-only.
    """
    r = await client.get("/v1/admin/stats", params={"window_days": int(window_days)})
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise AssertionError("/v1/admin/stats must return JSON object")
    return data


async def get_admin_log_count(
    *,
    client: httpx.AsyncClient,
    route: Optional[str] = None,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> int:
    """
    Returns the total count of matching inference_logs via /v1/admin/logs.
    Admin-only.
    """
    page = await get_admin_logs(client=client, route=route, model_id=model_id, api_key=api_key, limit=1, offset=0)
    return int(page.total)


async def assert_admin_log_count_increments(
    *,
    client: httpx.AsyncClient,
    before: int,
    route: Optional[str] = None,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    min_increment: int = 1,
) -> int:
    """
    Assert that matching inference_logs increased by >= min_increment since `before`.
    Returns the new count.
    """
    after = await get_admin_log_count(client=client, route=route, model_id=model_id, api_key=api_key)
    if after < before + int(min_increment):
        raise AssertionError(
            f"inference_logs did not increment as expected: before={before} after={after} "
            f"route={route!r} model_id={model_id!r} api_key={'<set>' if api_key else None}"
        )
    return after


async def latest_admin_log(
    *,
    client: httpx.AsyncClient,
    route: Optional[str] = None,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch the most recent matching log entry (if any).
    Admin-only.
    """
    page = await get_admin_logs(client=client, route=route, model_id=model_id, api_key=api_key, limit=1, offset=0)
    return page.items[0] if page.items else None


def coerce_admin_log_fields(log: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience: normalize a log dict into stable key set with parsed timestamps.
    """
    return {
        "id": log.get("id"),
        "created_at": _parse_iso_dt(log.get("created_at")),
        "api_key": log.get("api_key"),
        "route": log.get("route"),
        "client_host": log.get("client_host"),
        "model_id": log.get("model_id"),
        "latency_ms": log.get("latency_ms"),
        "prompt_tokens": log.get("prompt_tokens"),
        "completion_tokens": log.get("completion_tokens"),
        "prompt": log.get("prompt"),
        "output": log.get("output"),
    }

async def get_me_total_requests(client: httpx.AsyncClient) -> int:
    data = await get_me_usage(client)
    return int(data.get("total_requests") or 0)


async def assert_me_usage_increments(
    *,
    client: httpx.AsyncClient,
    before_total_requests: int,
    min_increment: int = 1,
) -> int:
    after = await get_me_total_requests(client)
    if after < before_total_requests + int(min_increment):
        raise AssertionError(
            f"/v1/me/usage.total_requests did not increment: before={before_total_requests} after={after}"
        )
    return after