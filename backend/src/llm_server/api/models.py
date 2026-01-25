# src/llm_server/api/models.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, Request, status
from pydantic import BaseModel

from llm_server.api.deps import deployment_capabilities, effective_capabilities, get_llm
from llm_server.core.config import get_settings
from llm_server.core.errors import AppError
from llm_server.services.inference import set_request_meta
from llm_server.services.llm_registry import MultiModelManager

router = APIRouter()


class ModelInfo(BaseModel):
    id: str
    default: bool
    backend: Optional[str] = None
    capabilities: Optional[Dict[str, bool]] = None
    load_mode: Optional[str] = None
    loaded: Optional[bool] = None


class ModelsResponse(BaseModel):
    default_model: str
    models: List[ModelInfo]
    deployment_capabilities: Dict[str, bool]


def _settings_from_request(request: Request):
    return getattr(request.app.state, "settings", None) or get_settings()


def _effective_model_load_mode(request: Request) -> str:
    mode = getattr(request.app.state, "model_load_mode", None)
    if isinstance(mode, str) and mode.strip():
        m = mode.strip().lower()
        return "eager" if m == "on" else m

    s = _settings_from_request(request)
    raw = getattr(s, "model_load_mode", None)
    if isinstance(raw, str) and raw.strip():
        m = raw.strip().lower()
        return "eager" if m == "on" else m

    env = str(getattr(s, "env", "dev")).strip().lower()
    return "eager" if env == "prod" else "lazy"


def _allowed_model_ids_from_settings(s) -> List[str]:
    allowed = getattr(s, "allowed_models", None)
    if isinstance(allowed, list) and allowed:
        return [str(x) for x in allowed if str(x).strip()]
    legacy = getattr(s, "all_model_ids", None) or []
    return [str(x) for x in legacy if str(x).strip()]


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(request: Request) -> ModelsResponse:
    s = _settings_from_request(request)
    dep_caps = deployment_capabilities(request)
    set_request_meta(request, route="/v1/models", model_id="models", cached=False)

    mode = _effective_model_load_mode(request)

    # In off mode, avoid touching llm dependency entirely.
    if mode == "off":
        default_model = cast(str, getattr(s, "model_id", "") or "")
        ids = _allowed_model_ids_from_settings(s)
        if default_model and default_model not in ids:
            ids = [default_model] + ids

        items: List[ModelInfo] = [
            ModelInfo(
                id=mid,
                default=(mid == default_model),
                backend=None,
                capabilities=effective_capabilities(mid, request=request),
                load_mode="off",
                loaded=None,
            )
            for mid in ids
        ]

        return ModelsResponse(
            default_model=default_model,
            models=items,
            deployment_capabilities=dep_caps,
        )

    # Only now resolve llm
    llm: Any = get_llm(request)

    if llm is None:
        raise AppError(
            code="llm_unavailable",
            message="LLM is not initialized",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    if isinstance(llm, MultiModelManager):
        set_request_meta(request, route="/v1/models", model_id=llm.default_id, cached=False)

        status_map: Dict[str, Dict[str, Any]] = {}
        try:
            for st in llm.status():
                status_map[st.model_id] = {
                    "loaded": st.loaded,
                    "load_mode": st.load_mode,
                    "backend": st.backend,
                }
        except Exception:
            status_map = {}

        items: List[ModelInfo] = []
        for model_id, backend_obj in llm.models.items():
            st = status_map.get(model_id, {})
            items.append(
                ModelInfo(
                    id=model_id,
                    default=(model_id == llm.default_id),
                    backend=str(st.get("backend") or backend_obj.__class__.__name__),
                    capabilities=effective_capabilities(model_id, request=request),
                    load_mode=str(st.get("load_mode") or "unknown"),
                    loaded=cast(Optional[bool], st.get("loaded")),
                )
            )

        return ModelsResponse(
            default_model=llm.default_id,
            models=items,
            deployment_capabilities=dep_caps,
        )

    model_id = cast(str, getattr(llm, "model_id", None) or getattr(s, "model_id", None) or "default")
    set_request_meta(request, route="/v1/models", model_id=model_id, cached=False)

    return ModelsResponse(
        default_model=model_id,
        models=[
            ModelInfo(
                id=model_id,
                default=True,
                backend=llm.__class__.__name__,
                capabilities=effective_capabilities(model_id, request=request),
                load_mode=mode,
                loaded=bool(getattr(request.app.state, "model_loaded", False)) if mode in ("eager", "on") else None,
            )
        ],
        deployment_capabilities=dep_caps,
    )