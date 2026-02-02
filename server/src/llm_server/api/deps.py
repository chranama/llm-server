# server/src/llm_server/api/deps.py
from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from typing import Any, Dict, Literal, Optional, Tuple, cast

from fastapi import Depends, Header, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_server.core.config import get_settings
from llm_server.core.errors import AppError
from llm_server.db.models import ApiKey
from llm_server.db.session import get_session
from llm_server.services.llm import build_llm_from_settings
from llm_server.services.llm_config import load_models_config
from llm_server.services.llm_registry import MultiModelManager
from llm_server.io.policy_decisions import policy_capability_overrides

# -----------------------------------------------------------------------------
# Types / constants
# -----------------------------------------------------------------------------

Capability = Literal["generate", "extract"]
_CAP_KEYS: tuple[Capability, ...] = ("generate", "extract")

# -----------------------------------------------------------------------------
# Simple in-memory rate limiting state
# -----------------------------------------------------------------------------
# bucket -> (window_start_ts, count)
_RL: Dict[str, Tuple[float, int]] = {}


def clear_rate_limit_state() -> None:
    """Test helper: clear in-memory rate limit buckets."""
    _RL.clear()


def _now() -> float:
    return time.time()


def _role_rpm(role_obj: Any) -> int:
    # TODO: wire real role-based RPM when roles are implemented
    return 60


def _check_rate_limit(key: str, role_obj: Any) -> None:
    rpm = _role_rpm(role_obj)
    if rpm is None or rpm <= 0:
        return

    now = _now()
    window = 60.0

    # Bucket includes id(_role_rpm) so monkeypatching in tests doesn't share buckets.
    bucket = f"{key}:{id(_role_rpm)}"
    window_start, count = _RL.get(bucket, (now, 0))

    # Reset each minute
    if now - window_start >= window:
        window_start = now
        count = 0

    if count >= rpm:
        retry_after = max(1, int(window - (now - window_start)))
        raise AppError(
            code="rate_limited",
            message="Rate limited",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            extra={"retry_after": retry_after},
        )

    _RL[bucket] = (window_start, count + 1)


def _check_and_consume_quota_in_session(api_key_obj: ApiKey) -> None:
    quota = api_key_obj.quota_monthly

    if quota is None or quota <= 0:
        return

    if api_key_obj.quota_used is None:
        api_key_obj.quota_used = 0

    if api_key_obj.quota_used >= quota:
        raise AppError(
            code="quota_exhausted",
            message="Monthly quota exhausted",
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
        )

    api_key_obj.quota_used += 1


async def get_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    session: AsyncSession = Depends(get_session),
) -> ApiKey:
    """
    Dependency used by protected endpoints.

      - Missing header -> AppError(401) code="missing_api_key"
      - Invalid key    -> AppError(401) code="invalid_api_key"
      - Rate limit hit -> AppError(429) code="rate_limited"
      - Quota exceeded -> AppError(402) code="quota_exhausted"
    """
    if not x_api_key:
        raise AppError(
            code="missing_api_key",
            message="X-API-Key header is required",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    result = await session.execute(select(ApiKey).where(ApiKey.key == x_api_key))
    api_key_obj: ApiKey | None = result.scalar_one_or_none()

    if api_key_obj is None or not getattr(api_key_obj, "active", True):
        raise AppError(
            code="invalid_api_key",
            message="Invalid or inactive API key",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    # IMPORTANT: do NOT touch api_key_obj.role here if it's lazy; keep it None for now.
    role_obj = None

    _check_rate_limit(api_key_obj.key, role_obj)

    _check_and_consume_quota_in_session(api_key_obj)
    session.add(api_key_obj)
    await session.commit()

    return api_key_obj


# -----------------------------------------------------------------------------
# Settings snapshot helpers (prefer app.state.settings)
# -----------------------------------------------------------------------------

def settings_from_request(request: Request | None) -> Any:
    if request is not None:
        s = getattr(request.app.state, "settings", None)
        if s is not None:
            return s
    return get_settings()


# -----------------------------------------------------------------------------
# LLM dependency
# -----------------------------------------------------------------------------

def _effective_model_load_mode_from_request(request: Request) -> str:
    """
    Single source of truth for model mode:
      1) app.state.model_load_mode (set in lifespan)
      2) app.state.settings / get_settings()
      3) derived default from env (prod=>eager else lazy)

    Normalizes "on" -> "eager".
    """
    mode = getattr(request.app.state, "model_load_mode", None)
    if isinstance(mode, str) and mode.strip():
        m = mode.strip().lower()
        return "eager" if m == "on" else m

    s = settings_from_request(request)

    raw = getattr(s, "model_load_mode", None)
    if isinstance(raw, str) and raw.strip():
        m = raw.strip().lower()
        return "eager" if m == "on" else m

    env = str(getattr(s, "env", "dev")).strip().lower()
    return "eager" if env == "prod" else "lazy"


def get_llm(request: Request) -> Any:
    """
    Retrieve the app's LLM object without triggering weight-loading here.

    Policy:
      - model_load_mode == "off": do NOT build on request; require admin/manual load.
      - model_error present: surface 503 instead of silently rebuilding.
      - otherwise: ensure an LLM object exists in app.state (construction only).
    """
    mode = _effective_model_load_mode_from_request(request)

    # If startup recorded an error, do not silently rebuild a new LLM object.
    model_error = getattr(request.app.state, "model_error", None)
    if model_error:
        raise AppError(
            code="llm_unavailable",
            message="LLM is unavailable due to startup/model error",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            extra={"model_error": model_error, "model_load_mode": mode},
        )

    llm = getattr(request.app.state, "llm", None)

    # "off" means: do not build lazily on request
    if mode == "off":
        if llm is not None:
            return llm
        raise AppError(
            code="llm_not_loaded",
            message="LLM is not loaded. Call POST /v1/admin/models/load (MODEL_LOAD_MODE=off).",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # lazy/eager: ensure llm object exists (loading semantics handled elsewhere)
    if llm is None:
        llm = build_llm_from_settings()
        request.app.state.llm = llm

    return llm


# -----------------------------------------------------------------------------
# Capability enforcement (deployment + per-model + policy)
# -----------------------------------------------------------------------------

def deployment_capabilities(request: Request | None = None) -> Dict[str, bool]:
    """
    Deployment-wide capability switches (Settings).
    """
    s = settings_from_request(request)
    return {
        "generate": bool(getattr(s, "enable_generate", True)),
        "extract": bool(getattr(s, "enable_extract", True)),
    }


@lru_cache(maxsize=1)
def _cached_models_config():
    # Disk parse cached; tests can call clear_models_config_cache()
    return load_models_config()


def clear_models_config_cache() -> None:
    _cached_models_config.cache_clear()


def _model_capabilities_from_models_yaml(model_id: str) -> Optional[Dict[str, bool]]:
    """
    Per-model capabilities from models.yaml normalization:
      - defaults.capabilities (if present)
      - overridden by model_spec.capabilities (if present)

    Returns None if models.yaml specifies no capabilities at all.
    """
    cfg = _cached_models_config()

    defaults_caps = cfg.defaults.get("capabilities")
    defaults_caps = cast(Optional[Dict[str, bool]], defaults_caps) if isinstance(defaults_caps, dict) else None

    spec_caps: Optional[Dict[str, bool]] = None
    for sp in cfg.models:
        if sp.id == model_id:
            if isinstance(sp.capabilities, dict):
                spec_caps = dict(sp.capabilities)
            break

    if defaults_caps is None and spec_caps is None:
        return None

    out: Dict[str, bool] = {}
    if defaults_caps:
        for k in _CAP_KEYS:
            if k in defaults_caps:
                out[k] = bool(defaults_caps[k])
    if spec_caps:
        for k in _CAP_KEYS:
            if k in spec_caps:
                out[k] = bool(spec_caps[k])

    # Note: may be partial; missing keys default to True when enforced.
    return out


def model_capabilities(model_id: str, *, request: Request | None = None) -> Optional[Dict[str, bool]]:
    """
    Return per-model capabilities.

    Priority:
      1) If MultiModelManager exists in app.state.llm -> use registry meta (runtime truth).
      2) Else fall back to models.yaml (cached).
      3) Apply POLICY overrides last (can fail-closed extract).
    """
    base_caps: Optional[Dict[str, bool]] = None

    if request is not None:
        llm = getattr(request.app.state, "llm", None)
        if isinstance(llm, MultiModelManager):
            caps_meta = llm._meta.get(model_id, {}).get("capabilities", None)  # intentionally lightweight
            if caps_meta is None:
                base_caps = None
            elif isinstance(caps_meta, dict):
                out: Dict[str, bool] = {}
                for k in _CAP_KEYS:
                    if k in caps_meta:
                        out[k] = bool(caps_meta.get(k))
                base_caps = out or None
            elif isinstance(caps_meta, (list, tuple, set)):
                allowed = {str(x).strip().lower() for x in caps_meta if isinstance(x, str) and str(x).strip()}
                base_caps = {k: (k in allowed) for k in _CAP_KEYS}
            else:
                base_caps = None
        else:
            base_caps = _model_capabilities_from_models_yaml(model_id)
    else:
        base_caps = _model_capabilities_from_models_yaml(model_id)

    # --- POLICY OVERRIDES (last writer wins) ---
    if request is not None:
        pol = policy_capability_overrides(model_id, request=request)
        if pol:
            # Merge onto base_caps; if base is None, policy becomes the only explicit caps map.
            merged: Dict[str, bool] = dict(base_caps or {})
            for k, v in pol.items():
                if k in _CAP_KEYS:
                    merged[k] = bool(v)
            return merged or None

    return base_caps


def effective_capabilities(
    model_id: str,
    *,
    request: Request | None = None,
) -> Dict[str, bool]:
    """
    Effective capabilities = (per-model caps defaulting to True/True if unspecified)
    AND deployment-wide gates from Settings.
    """
    raw = model_capabilities(model_id, request=request)
    if raw is None:
        raw = {"generate": True, "extract": True}

    for k in _CAP_KEYS:
        raw.setdefault(k, True)

    dep = deployment_capabilities(request)
    return {k: bool(raw[k]) and bool(dep.get(k, True)) for k in _CAP_KEYS}


def require_capability(
    model_id: str,
    capability: Capability,
    *,
    request: Request | None = None,
) -> None:
    """
    Enforce that:
      1) deployment allows the capability, AND
      2) the chosen model is capable (registry meta / models.yaml), AND
      3) policy overrides (if any) allow it (wired via model_capabilities()).

    Raises:
      - 501 capability_disabled (deployment off)
      - 400 capability_not_supported (model lacks)
    """
    dep = deployment_capabilities(request)
    if not bool(dep.get(capability, True)):
        raise AppError(
            code="capability_disabled",
            message=f"{capability} is disabled in this deployment mode.",
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            extra={"capability": capability},
        )

    caps = model_capabilities(model_id, request=request)
    if caps is None:
        return

    supported = bool(caps.get(capability, True))
    if not supported:
        raise AppError(
            code="capability_not_supported",
            message=f"Model '{model_id}' does not support capability '{capability}'.",
            status_code=status.HTTP_400_BAD_REQUEST,
            extra={"model_id": model_id, "capability": capability, "model_capabilities": caps},
        )


# -----------------------------------------------------------------------------
# Shared model routing + request fingerprint helpers
# -----------------------------------------------------------------------------

def allowed_model_ids(*, request: Request | None = None) -> list[str]:
    """
    Back-compat: settings may expose allowed models as:
      - allowed_models (preferred; set by llm_config.load_models_config)
      - all_model_ids (legacy)
    """
    s = settings_from_request(request)
    allowed = getattr(s, "allowed_models", None)
    if isinstance(allowed, list) and allowed:
        return [str(x) for x in allowed if str(x).strip()]
    legacy = getattr(s, "all_model_ids", None) or []
    return [str(x) for x in legacy if str(x).strip()]


def default_model_id_from_settings(*, request: Request | None = None) -> str:
    s = settings_from_request(request)
    mid = getattr(s, "model_id", None)
    if not isinstance(mid, str) or not mid.strip():
        return ""
    return mid.strip()


def resolve_model(
    llm: Any,
    model_override: str | None,
    *,
    capability: Capability | None = None,
    request: Request | None = None,
) -> tuple[str, Any]:
    """
    Resolve a concrete (model_id, model_backend) pair.

    NOTE: This function does NOT call require_capability(). Endpoints do that explicitly.
    """
    allowed = allowed_model_ids(request=request)

    # --- Multi-model registry ---
    if isinstance(llm, MultiModelManager):
        if model_override is not None:
            model_id = model_override
            if model_id not in llm:
                raise AppError(
                    code="model_missing",
                    message=f"Model '{model_id}' not found in LLM registry",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    extra={"available": llm.list_models(), "default_id": llm.default_id},
                )
        else:
            model_id = llm.default_id
            if capability:
                fn = getattr(llm, "default_for_capability", None)
                if callable(fn):
                    try:
                        model_id = str(fn(capability))
                    except Exception:
                        model_id = llm.default_id

        if allowed and model_id not in allowed:
            raise AppError(
                code="model_not_allowed",
                message=f"Model '{model_id}' not allowed.",
                status_code=status.HTTP_400_BAD_REQUEST,
                extra={"allowed": allowed},
            )

        return model_id, llm[model_id]

    # --- Dict registry (legacy) ---
    if isinstance(llm, dict):
        model_id = model_override or default_model_id_from_settings(request=request) or next(iter(llm.keys()), "")
        if not model_id:
            raise AppError(
                code="model_config_invalid",
                message="No model configured",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if model_override is not None and allowed and model_id not in allowed:
            raise AppError(
                code="model_not_allowed",
                message=f"Model '{model_id}' not allowed.",
                status_code=status.HTTP_400_BAD_REQUEST,
                extra={"allowed": allowed},
            )

        if model_id not in llm:
            raise AppError(
                code="model_missing",
                message=f"Model '{model_id}' not found in LLM registry",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return model_id, llm[model_id]

    # --- Single backend ---
    model_id = model_override or default_model_id_from_settings(request=request)
    if model_override is not None and allowed and model_id not in allowed:
        raise AppError(
            code="model_not_allowed",
            message=f"Model '{model_id}' not allowed.",
            status_code=status.HTTP_400_BAD_REQUEST,
            extra={"allowed": allowed},
        )

    return model_id or "default", llm


def sha32(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def sha32_json(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def make_cache_redis_key(model_id: str, prompt_hash: str, params_fp: str) -> str:
    return f"llm:cache:{model_id}:{prompt_hash}:{params_fp}"


def make_extract_redis_key(model_id: str, prompt_hash: str, params_fp: str) -> str:
    return f"llm:extract:{model_id}:{prompt_hash}:{params_fp}"


def fingerprint_pydantic(body: Any, *, exclude: set[str], exclude_none: bool = True) -> str:
    params = body.model_dump(exclude=exclude, exclude_none=exclude_none)
    return sha32_json(params)