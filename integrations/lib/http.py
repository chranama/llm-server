from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import httpx


DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 0.2
DEFAULT_MAX_BACKOFF_SECONDS = 2.0


def _merge_headers(base: Mapping[str, str] | None, extra: Mapping[str, str] | None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if base:
        out.update(dict(base))
    if extra:
        out.update(dict(extra))
    return out


def _is_retryable_status(code: int) -> bool:
    # Conservative: transient / overloaded / gateway problems
    return code in (408, 425, 429, 500, 502, 503, 504)


def _is_retryable_exc(exc: Exception) -> bool:
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError))


def _resp_preview(r: httpx.Response, limit: int = 800) -> str:
    try:
        t = r.text
    except Exception:
        return "<unable to read response body>"
    t = t.strip()
    if len(t) > limit:
        return t[:limit] + "…"
    return t


class HttpError(RuntimeError):
    def __init__(self, msg: str, *, status_code: int | None = None) -> None:
        super().__init__(msg)
        self.status_code = status_code


@dataclass(frozen=True)
class RequestSpec:
    method: str
    path: str
    params: Optional[Dict[str, Any]] = None
    json: Optional[Any] = None
    data: Optional[Any] = None
    headers: Optional[Dict[str, str]] = None


async def request(
    client: httpx.AsyncClient,
    spec: RequestSpec,
    *,
    api_key: str | None = None,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
    retry_on_status: bool = True,
) -> httpx.Response:
    """
    Robust request helper with retries + jitter.

    Notes:
      - Uses client's base_url.
      - Adds X-API-Key if api_key is provided.
      - Retries on network/timeouts and (optionally) on retryable HTTP status codes.
    """
    headers = dict(spec.headers or {})
    if api_key:
        headers.setdefault("X-API-Key", api_key)

    attempt = 0
    last_exc: Exception | None = None

    while True:
        attempt += 1
        try:
            r = await client.request(
                spec.method,
                spec.path,
                params=spec.params,
                json=spec.json,
                data=spec.data,
                headers=headers or None,
            )

            if retry_on_status and _is_retryable_status(r.status_code) and attempt <= retries:
                # Drain response before retrying to avoid connection issues.
                try:
                    _ = r.text
                except Exception:
                    pass
                delay = min(max_backoff_seconds, backoff_seconds * (2 ** (attempt - 1)))
                delay = delay * (0.8 + 0.4 * random.random())  # jitter in [0.8,1.2]
                await asyncio.sleep(delay)
                continue

            return r

        except Exception as exc:
            last_exc = exc
            if attempt <= retries and _is_retryable_exc(exc):
                delay = min(max_backoff_seconds, backoff_seconds * (2 ** (attempt - 1)))
                delay = delay * (0.8 + 0.4 * random.random())
                await asyncio.sleep(delay)
                continue
            raise


async def get_json(
    client: httpx.AsyncClient,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    api_key: str | None = None,
    retries: int = DEFAULT_RETRIES,
) -> Any:
    r = await request(
        client,
        RequestSpec(method="GET", path=path, params=params, headers=headers),
        api_key=api_key,
        retries=retries,
    )
    if r.status_code < 200 or r.status_code >= 300:
        raise HttpError(f"GET {path} -> HTTP {r.status_code}: {_resp_preview(r)}", status_code=r.status_code)
    try:
        return r.json()
    except Exception as e:
        raise HttpError(f"GET {path} returned non-JSON: {_resp_preview(r)}") from e


async def post_json(
    client: httpx.AsyncClient,
    path: str,
    payload: Any,
    *,
    headers: Optional[Dict[str, str]] = None,
    api_key: str | None = None,
    retries: int = DEFAULT_RETRIES,
) -> Any:
    h = _merge_headers({"Content-Type": "application/json"}, headers)
    r = await request(
        client,
        RequestSpec(method="POST", path=path, json=payload, headers=h),
        api_key=api_key,
        retries=retries,
    )
    if r.status_code < 200 or r.status_code >= 300:
        raise HttpError(f"POST {path} -> HTTP {r.status_code}: {_resp_preview(r)}", status_code=r.status_code)
    try:
        return r.json()
    except Exception as e:
        raise HttpError(f"POST {path} returned non-JSON: {_resp_preview(r)}") from e


async def get_text(
    client: httpx.AsyncClient,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    api_key: str | None = None,
    retries: int = DEFAULT_RETRIES,
) -> str:
    r = await request(
        client,
        RequestSpec(method="GET", path=path, params=params, headers=headers),
        api_key=api_key,
        retries=retries,
    )
    if r.status_code < 200 or r.status_code >= 300:
        raise HttpError(f"GET {path} -> HTTP {r.status_code}: {_resp_preview(r)}", status_code=r.status_code)
    return r.text


async def get_metrics_text(client: httpx.AsyncClient, *, api_key: str | None = None) -> str:
    """
    Convenience: scrape Prometheus metrics from /metrics.
    """
    return await get_text(client, "/metrics", api_key=api_key, retries=1)


async def expect_status(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    api_key: str | None = None,
    expected: int | Iterable[int] = 200,
    json_body: Any | None = None,
) -> httpx.Response:
    exp = {expected} if isinstance(expected, int) else set(expected)

    headers: Dict[str, str] = {}
    data = None
    payload = None
    if json_body is not None:
        headers["Content-Type"] = "application/json"
        payload = json_body

    r = await request(
        client,
        RequestSpec(method=method, path=path, json=payload, data=data, headers=headers or None),
        api_key=api_key,
        retries=0,
        retry_on_status=False,
    )

    if r.status_code not in exp:
        raise HttpError(
            f"{method} {path} expected {sorted(exp)}, got {r.status_code}: {_resp_preview(r)}",
            status_code=r.status_code,
        )
    return r