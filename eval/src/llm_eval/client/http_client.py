# eval/src/llm_eval/client/http_client.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx


# =========================
# Typed results
# =========================


@dataclass(frozen=True)
class GenerateOk:
    model: str
    output_text: str
    cached: bool
    latency_ms: float


@dataclass(frozen=True)
class GenerateErr:
    status_code: int
    error_code: str
    message: str
    extra: Optional[Dict[str, Any]]
    latency_ms: float


@dataclass(frozen=True)
class ExtractOk:
    schema_id: str
    model: str
    data: Dict[str, Any]
    cached: bool
    repair_attempted: bool
    latency_ms: float


@dataclass(frozen=True)
class ExtractErr:
    status_code: int
    error_code: str
    message: str
    extra: Optional[Dict[str, Any]]
    latency_ms: float


class HttpEvalClient:
    """
    Talks to the llm-server API endpoints.

    - POST /v1/generate (text generation)
    - POST /v1/extract  (schema-validated extraction)

    Notes:
    - Never raises for HTTP errors: callers get GenerateErr / ExtractErr.
    - Captures classification-friendly metadata in `.extra` on errors:
        - stage
        - content_type
        - request_id (from header or body)
        - response_text preview (safe, truncated)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "test_api_key_123",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)
        self._client: Optional[httpx.AsyncClient] = None

    def _headers(self) -> Dict[str, str]:
        return {"X-API-Key": self.api_key}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _shorten(s: str, n: int = 2000) -> str:
        s = (s or "").strip()
        if len(s) <= n:
            return s
        return s[: n - 3] + "..."

    @staticmethod
    def _safe_json(resp: httpx.Response) -> Tuple[Any, Optional[str]]:
        """
        Returns (json_obj, json_error_str_or_None)
        """
        try:
            return resp.json(), None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    @staticmethod
    def _extract_text_from_generate_payload(payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            for k in ("output", "text", "completion", "result"):
                v = payload.get(k)
                if isinstance(v, str):
                    return v
            data = payload.get("data")
            if isinstance(data, dict):
                for k in ("output", "text", "completion", "result"):
                    v = data.get(k)
                    if isinstance(v, str):
                        return v
        return str(payload)

    @staticmethod
    def _extract_model_from_payload(payload: Any, fallback: str = "unknown") -> str:
        if isinstance(payload, dict):
            for k in ("model", "model_id"):
                v = payload.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            data = payload.get("data")
            if isinstance(data, dict):
                v = data.get("model") or data.get("model_id")
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return fallback

    @staticmethod
    def _extract_request_id(resp: httpx.Response, payload: Any) -> Optional[str]:
        # Prefer header; fall back to body field if present
        rid = resp.headers.get("X-Request-ID") or resp.headers.get("x-request-id")
        if rid and rid.strip():
            return rid.strip()
        if isinstance(payload, dict):
            v = payload.get("request_id")
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def _extract_error_fields(self, resp: httpx.Response) -> tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Normalize llm-server error shape.

        Expected server error payload:
          {"code": "...", "message": "...", "extra": {...}, "request_id": "..."}
        If unavailable/unparseable, fall back to resp.text and attach classification metadata.
        """
        content_type = resp.headers.get("content-type", "") or ""
        text_preview = self._shorten(resp.text or "", 2500)

        j, jerr = self._safe_json(resp)
        rid = self._extract_request_id(resp, j)

        if not isinstance(j, dict):
            # Server returned non-JSON (or JSON parse failed)
            extra: Dict[str, Any] = {
                "stage": "server_json_error" if jerr else "server_non_json",
                "status_code": int(resp.status_code),
                "content_type": content_type,
                "request_id": rid,
                "response_text": text_preview,
            }
            if jerr:
                extra["json_error"] = jerr
            return "http_error", text_preview or "HTTP error", extra

        code = str(j.get("code") or "http_error")
        msg = str(j.get("message") or (resp.text or ""))
        extra_any = j.get("extra") if isinstance(j.get("extra"), dict) else None

        merged: Dict[str, Any] = {}
        if isinstance(extra_any, dict):
            merged.update(extra_any)

        # Always include these classification-friendly fields
        merged.setdefault("stage", "server_error")
        merged["status_code"] = int(resp.status_code)
        merged["content_type"] = content_type
        if rid:
            merged["request_id"] = rid
        # keep preview; policy/eval can redact later if needed
        merged.setdefault("response_text", text_preview)

        return code, msg, merged

    # =========================
    # /v1/generate
    # =========================

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        model: Optional[str] = None,
        cache: Optional[bool] = None,
    ) -> GenerateOk | GenerateErr:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        }
        if model is not None:
            payload["model"] = model
        if cache is not None:
            payload["cache"] = bool(cache)

        t0 = time.time()
        try:
            r = await self._get_client().post(
                f"{self.base_url}/v1/generate",
                json=payload,
                headers=self._headers(),
            )
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000.0
            return GenerateErr(
                status_code=0,
                error_code="transport_error",
                message=f"{type(e).__name__}: {e}",
                extra={"stage": "transport_error", "exc_type": type(e).__name__},
                latency_ms=latency_ms,
            )

        latency_ms = (time.time() - t0) * 1000.0

        if r.status_code == 200:
            data, _ = self._safe_json(r)
            output_text = self._extract_text_from_generate_payload(data).strip()
            model_id = self._extract_model_from_payload(data, fallback=(model or "unknown"))
            cached = bool(data.get("cached", False)) if isinstance(data, dict) else False

            return GenerateOk(
                model=str(model_id),
                output_text=output_text,
                cached=cached,
                latency_ms=latency_ms,
            )

        code, msg, extra = self._extract_error_fields(r)
        return GenerateErr(
            status_code=int(r.status_code),
            error_code=code,
            message=msg,
            extra=extra,
            latency_ms=latency_ms,
        )

    # =========================
    # /v1/extract
    # =========================

    async def extract(
        self,
        *,
        schema_id: str,
        text: str,
        model: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        cache: bool = True,
        repair: bool = True,
    ) -> ExtractOk | ExtractErr:
        payload: Dict[str, Any] = {
            "schema_id": schema_id,
            "text": text,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "cache": bool(cache),
            "repair": bool(repair),
        }
        if model is not None:
            payload["model"] = model

        t0 = time.time()
        try:
            r = await self._get_client().post(
                f"{self.base_url}/v1/extract",
                json=payload,
                headers=self._headers(),
            )
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000.0
            return ExtractErr(
                status_code=0,
                error_code="transport_error",
                message=f"{type(e).__name__}: {e}",
                extra={"stage": "transport_error", "exc_type": type(e).__name__},
                latency_ms=latency_ms,
            )

        latency_ms = (time.time() - t0) * 1000.0

        if r.status_code == 200:
            data, _ = self._safe_json(r)
            if not isinstance(data, dict):
                data = {}

            schema_out = data.get("schema_id", schema_id)
            model_out = data.get("model", model or "unknown")
            obj = data.get("data")
            if not isinstance(obj, dict):
                obj = {}

            return ExtractOk(
                schema_id=str(schema_out),
                model=str(model_out),
                data=dict(obj),
                cached=bool(data.get("cached", False)),
                repair_attempted=bool(data.get("repair_attempted", False)),
                latency_ms=latency_ms,
            )

        code, msg, extra = self._extract_error_fields(r)
        # Normalize stage for eval aggregation: if server JSON decode failed, use server_json_error
        if isinstance(extra, dict) and extra.get("stage") in ("server_non_json", "server_error"):
            # keep whatever server gave; just ensure at least one stable stage exists
            extra.setdefault("stage", "server_error")

        return ExtractErr(
            status_code=int(r.status_code),
            error_code=code,
            message=msg,
            extra=extra,
            latency_ms=latency_ms,
        )