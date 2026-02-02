# src/llm_server/services/llm_config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml

from llm_server.core.config import get_settings
from llm_server.core.errors import AppError

# -----------------------------
# Types / allowed values
# -----------------------------

Backend = Literal["local", "remote"]
LoadMode = Literal["eager", "lazy", "off"]
Device = Literal["auto", "cuda", "mps", "cpu"]
DType = Literal["float16", "bfloat16", "float32"]

Quantization = Optional[str]  # e.g. "int8", "int4", "nf4", None

# Capabilities are explicit + strictly validated
Capability = Literal["generate", "extract"]
CapabilitiesMap = Dict[Capability, bool]

_ALLOWED_BACKENDS = {"local", "remote"}
_ALLOWED_LOAD_MODES = {"eager", "lazy", "off"}
_ALLOWED_DEVICES = {"auto", "cuda", "mps", "cpu"}
_ALLOWED_DTYPES = {"float16", "bfloat16", "float32"}
_ALLOWED_QUANT = {None, "int8", "int4", "nf4"}  # extend later as you add support
_ALLOWED_CAP_KEYS: set[str] = {"generate", "extract"}


# -----------------------------
# Normalized config objects
# -----------------------------


@dataclass(frozen=True)
class ModelSpec:
    """
    Normalized single-model spec.
    """

    id: str
    backend: Backend = "local"
    load_mode: LoadMode = "lazy"

    # Per-model capabilities for API gating
    capabilities: Optional[CapabilitiesMap] = None

    dtype: Optional[DType] = None
    device: Device = "auto"
    text_only: Optional[bool] = None
    max_context: Optional[int] = None
    trust_remote_code: bool = False
    quantization: Quantization = None
    notes: Optional[str] = None


@dataclass(frozen=True)
class ModelsConfig:
    """
    Normalized model configuration for the service.

    primary_id: the default model id
    model_ids:  ordered unique model ids, primary first
    models:     list of ModelSpec in same order as model_ids
    defaults:   any derived defaults used during normalization (for debugging)
    """

    primary_id: str
    model_ids: List[str]
    models: List[ModelSpec]
    defaults: Dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------


def _app_root() -> Path:
    """
    Resolve APP_ROOT (container-friendly). Falls back to cwd.
    """
    v = (os.environ.get("APP_ROOT") or "").strip()
    if v:
        return Path(v).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_path_maybe_relative(path: str) -> Path:
    """
    Resolve a path relative to APP_ROOT if not absolute.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (_app_root() / p).resolve()


def _resolve_models_yaml_path() -> str:
    """
    Resolve models.yaml path with this priority:

      1) MODELS_YAML env var (compose/k8s explicit override)
      2) Settings.models_config_path (from YAML/server.yaml or env)
      3) default "config/models.yaml"

    Relative paths are resolved against APP_ROOT if set, else cwd.
    """
    # 1) Explicit env override
    env_path = (os.environ.get("MODELS_YAML") or "").strip()
    if env_path:
        return str(_resolve_path_maybe_relative(env_path))

    # 2) Settings (may come from config/server.yaml)
    s = get_settings()
    raw = str(getattr(s, "models_config_path", None) or "config/models.yaml").strip()
    if not raw:
        raw = "config/models.yaml"

    return str(_resolve_path_maybe_relative(raw))


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _as_str(x: Any, *, field: str, path: str) -> str:
    if not isinstance(x, str) or not x.strip():
        raise AppError(
            code="models_yaml_invalid",
            message=f"models.yaml {field} must be a non-empty string",
            status_code=500,
            extra={"path": path, "field": field, "value": x},
        )
    return x.strip()


def _as_opt_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    return None


def _as_opt_int(x: Any, *, field: str, path: str) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        raise AppError(
            code="models_yaml_invalid",
            message=f"models.yaml {field} must be an integer",
            status_code=500,
            extra={"path": path, "field": field, "value": x},
        )
    if isinstance(x, int):
        return x
    raise AppError(
        code="models_yaml_invalid",
        message=f"models.yaml {field} must be an integer",
        status_code=500,
        extra={"path": path, "field": field, "value": x},
    )


def _as_opt_bool(x: Any, *, field: str, path: str) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    raise AppError(
        code="models_yaml_invalid",
        message=f"models.yaml {field} must be a boolean",
        status_code=500,
        extra={"path": path, "field": field, "value": x},
    )


def _validate_enum(
    value: Any,
    *,
    field: str,
    path: str,
    allowed: set[Any],
    coerce_lower: bool = True,
) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and coerce_lower:
        value = value.strip().lower()
    if value not in allowed:
        raise AppError(
            code="models_yaml_invalid",
            message=f"models.yaml {field} has invalid value",
            status_code=500,
            extra={"path": path, "field": field, "value": value, "allowed": sorted(list(allowed))},
        )
    return value


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise AppError(
            code="models_yaml_missing",
            message="models.yaml not found",
            status_code=500,
            extra={"path": path},
        )
    except Exception as e:
        raise AppError(
            code="models_yaml_invalid",
            message="Failed to read models.yaml",
            status_code=500,
            extra={"path": path, "error": str(e)},
        ) from e

    if not data or not isinstance(data, dict):
        raise AppError(
            code="models_yaml_invalid",
            message="models.yaml must be a non-empty mapping (dict)",
            status_code=500,
            extra={"path": path},
        )
    return data


def _normalize_capabilities(
    raw_caps: Any,
    *,
    path: str,
    field: str,
) -> Optional[CapabilitiesMap]:
    """
    Accepts:
      - None
      - dict like {generate: true, extract: false}

    Strict:
      - keys limited to {"generate","extract"}
      - values must be boolean
    Returns a normalized dict[str,bool] with only allowed keys, or None.
    """
    if raw_caps is None:
        return None

    if not isinstance(raw_caps, dict):
        raise AppError(
            code="models_yaml_invalid",
            message=f"models.yaml {field} must be a mapping (dict)",
            status_code=500,
            extra={"path": path, "field": field, "value": raw_caps},
        )

    out: Dict[str, bool] = {}
    for k, v in raw_caps.items():
        if not isinstance(k, str) or not k.strip():
            raise AppError(
                code="models_yaml_invalid",
                message=f"models.yaml {field} keys must be strings",
                status_code=500,
                extra={"path": path, "field": field, "key": k},
            )
        kk = k.strip()
        if kk not in _ALLOWED_CAP_KEYS:
            raise AppError(
                code="models_yaml_invalid",
                message=f"models.yaml {field} has invalid capability key",
                status_code=500,
                extra={"path": path, "field": field, "key": kk, "allowed": sorted(list(_ALLOWED_CAP_KEYS))},
            )
        if not isinstance(v, bool):
            raise AppError(
                code="models_yaml_invalid",
                message=f"models.yaml {field}.{kk} must be a boolean",
                status_code=500,
                extra={"path": path, "field": f"{field}.{kk}", "value": v},
            )
        out[kk] = v

    # Safe cast due to validation above
    return out  # type: ignore[return-value]


def _normalize_model_entry(
    raw: Any,
    *,
    path: str,
    defaults: Dict[str, Any],
) -> ModelSpec:
    """
    Accepts:
      - "model_id" (string)
      - {"id": "..."} (minimal)
      - {"id": "...", backend/load_mode/...} (full)

    Supports per-model capabilities:
      capabilities:
        generate: true/false
        extract: true/false
    """
    if isinstance(raw, str):
        mid = _as_str(raw, field="models[]", path=path)
        return ModelSpec(
            id=mid,
            backend=defaults["backend"],
            load_mode=defaults["load_mode"],
            capabilities=defaults.get("capabilities"),
            dtype=defaults["dtype"],
            device=defaults["device"],
            text_only=defaults["text_only"],
            max_context=defaults["max_context"],
            trust_remote_code=defaults["trust_remote_code"],
            quantization=defaults["quantization"],
            notes=None,
        )

    if not isinstance(raw, dict):
        raise AppError(
            code="models_yaml_invalid",
            message="models.yaml models entries must be strings or objects with an 'id' field",
            status_code=500,
            extra={"path": path, "bad_item": str(raw)},
        )

    if "id" not in raw:
        raise AppError(
            code="models_yaml_invalid",
            message="models.yaml models object entries must contain 'id'",
            status_code=500,
            extra={"path": path, "bad_item": str(raw)},
        )

    mid = _as_str(raw.get("id"), field="models[].id", path=path)

    backend = _validate_enum(
        raw.get("backend", defaults["backend"]),
        field="models[].backend",
        path=path,
        allowed=_ALLOWED_BACKENDS,
    ) or defaults["backend"]

    load_mode = _validate_enum(
        raw.get("load_mode", defaults["load_mode"]),
        field="models[].load_mode",
        path=path,
        allowed=_ALLOWED_LOAD_MODES,
    ) or defaults["load_mode"]

    device = _validate_enum(
        raw.get("device", defaults["device"]),
        field="models[].device",
        path=path,
        allowed=_ALLOWED_DEVICES,
    ) or defaults["device"]

    dtype_raw = raw.get("dtype", defaults["dtype"])
    dtype: Optional[DType] = None
    if dtype_raw is not None:
        dtype = _validate_enum(
            dtype_raw,
            field="models[].dtype",
            path=path,
            allowed=_ALLOWED_DTYPES,
        )

    quant = raw.get("quantization", defaults["quantization"])
    if isinstance(quant, str):
        quant = quant.strip().lower()
    if quant not in _ALLOWED_QUANT:
        raise AppError(
            code="models_yaml_invalid",
            message="models.yaml models[].quantization has invalid value",
            status_code=500,
            extra={
                "path": path,
                "field": "models[].quantization",
                "value": quant,
                "allowed": sorted([x for x in _ALLOWED_QUANT if x is not None]) + [None],
            },
        )

    text_only = _as_opt_bool(raw.get("text_only", defaults["text_only"]), field="models[].text_only", path=path)
    max_context = _as_opt_int(raw.get("max_context", defaults["max_context"]), field="models[].max_context", path=path)

    trc = raw.get("trust_remote_code", defaults["trust_remote_code"])
    if trc is None:
        trc = defaults["trust_remote_code"]
    if not isinstance(trc, bool):
        raise AppError(
            code="models_yaml_invalid",
            message="models.yaml models[].trust_remote_code must be a boolean",
            status_code=500,
            extra={"path": path, "field": "models[].trust_remote_code", "value": trc},
        )

    caps_raw = raw.get("capabilities", defaults.get("capabilities"))
    capabilities = _normalize_capabilities(caps_raw, path=path, field="models[].capabilities")

    notes = _as_opt_str(raw.get("notes"))

    return ModelSpec(
        id=mid,
        backend=backend,  # type: ignore[assignment]
        load_mode=load_mode,  # type: ignore[assignment]
        capabilities=capabilities,
        dtype=dtype,  # type: ignore[assignment]
        device=device,  # type: ignore[assignment]
        text_only=text_only,
        max_context=max_context,
        trust_remote_code=bool(trc),
        quantization=quant,  # type: ignore[assignment]
        notes=notes,
    )


def load_models_config() -> ModelsConfig:
    """
    Load model specs from models.yaml if present, otherwise fall back to Settings.

    IMPORTANT:
      - Path resolution honors MODELS_YAML (compose/k8s override) first.
      - Relative paths resolve against APP_ROOT when set.
      - Updates Settings (best-effort) for legacy call sites.
    """
    s = get_settings()
    path = _resolve_models_yaml_path()

    if path and os.path.exists(path):
        data = _load_yaml(path)

        default_model = data.get("default_model")
        if default_model is not None and not isinstance(default_model, str):
            raise AppError(
                code="models_yaml_invalid",
                message="models.yaml default_model must be a string",
                status_code=500,
                extra={"path": path},
            )

        models_list = data.get("models") or []
        if not isinstance(models_list, list):
            raise AppError(
                code="models_yaml_invalid",
                message="models.yaml models must be a list",
                status_code=500,
                extra={"path": path},
            )

        defaults: Dict[str, Any] = data.get("defaults") or {}
        if defaults and not isinstance(defaults, dict):
            raise AppError(
                code="models_yaml_invalid",
                message="models.yaml defaults must be a mapping (dict) if provided",
                status_code=500,
                extra={"path": path},
            )

        # defaults.capabilities
        caps_defaults = _normalize_capabilities(defaults.get("capabilities", None), path=path, field="defaults.capabilities")

        norm_defaults: Dict[str, Any] = {
            "backend": _validate_enum(
                defaults.get("backend", "local"),
                field="defaults.backend",
                path=path,
                allowed=_ALLOWED_BACKENDS,
            )
            or "local",
            "load_mode": _validate_enum(
                defaults.get("load_mode", "lazy"),
                field="defaults.load_mode",
                path=path,
                allowed=_ALLOWED_LOAD_MODES,
            )
            or "lazy",
            "device": _validate_enum(
                defaults.get("device", "auto"),
                field="defaults.device",
                path=path,
                allowed=_ALLOWED_DEVICES,
            )
            or "auto",
            "dtype": None,
            "capabilities": caps_defaults,
            "text_only": defaults.get("text_only", None),
            "max_context": defaults.get("max_context", None),
            "trust_remote_code": bool(defaults.get("trust_remote_code", False)),
            "quantization": defaults.get("quantization", None),
        }

        if norm_defaults["text_only"] is not None and not isinstance(norm_defaults["text_only"], bool):
            raise AppError(
                code="models_yaml_invalid",
                message="models.yaml defaults.text_only must be a boolean",
                status_code=500,
                extra={"path": path, "field": "defaults.text_only", "value": norm_defaults["text_only"]},
            )

        if norm_defaults["max_context"] is not None and not isinstance(norm_defaults["max_context"], int):
            raise AppError(
                code="models_yaml_invalid",
                message="models.yaml defaults.max_context must be an integer",
                status_code=500,
                extra={"path": path, "field": "defaults.max_context", "value": norm_defaults["max_context"]},
            )

        dtype_default = defaults.get("dtype", None)
        if dtype_default is not None:
            norm_defaults["dtype"] = _validate_enum(dtype_default, field="defaults.dtype", path=path, allowed=_ALLOWED_DTYPES)

        quant_default = defaults.get("quantization", None)
        if isinstance(quant_default, str):
            quant_default = quant_default.strip().lower()
        if quant_default not in _ALLOWED_QUANT:
            raise AppError(
                code="models_yaml_invalid",
                message="models.yaml defaults.quantization has invalid value",
                status_code=500,
                extra={
                    "path": path,
                    "field": "defaults.quantization",
                    "value": quant_default,
                    "allowed": sorted([x for x in _ALLOWED_QUANT if x is not None]) + [None],
                },
            )
        norm_defaults["quantization"] = quant_default

        specs: List[ModelSpec] = []
        for raw in models_list:
            specs.append(_normalize_model_entry(raw, path=path, defaults=norm_defaults))

        ids = _dedupe_preserve_order([m.id for m in specs])

        # Determine primary/default id
        if default_model is None:
            if not ids:
                raise AppError(
                    code="models_yaml_invalid",
                    message="models.yaml must define default_model and/or at least one model id in models",
                    status_code=500,
                    extra={"path": path},
                )
            primary_id = ids[0]
        else:
            primary_id = default_model.strip()
            if not primary_id:
                raise AppError(
                    code="models_yaml_invalid",
                    message="models.yaml default_model must be a non-empty string",
                    status_code=500,
                    extra={"path": path},
                )

            if primary_id not in ids:
                # Keep behavior: allow default_model to be absent from list; add it.
                ids.insert(0, primary_id)
                specs.insert(
                    0,
                    ModelSpec(
                        id=primary_id,
                        backend=norm_defaults["backend"],
                        load_mode=norm_defaults["load_mode"],
                        capabilities=norm_defaults.get("capabilities"),
                        dtype=norm_defaults["dtype"],
                        device=norm_defaults["device"],
                        text_only=norm_defaults["text_only"],
                        max_context=norm_defaults["max_context"],
                        trust_remote_code=norm_defaults["trust_remote_code"],
                        quantization=norm_defaults["quantization"],
                        notes="(auto-added because default_model was not listed)",
                    ),
                )

        # Reorder so primary first; keep first spec per id.
        spec_map: Dict[str, ModelSpec] = {}
        for sp in specs:
            if sp.id not in spec_map:
                spec_map[sp.id] = sp

        ordered_ids = [primary_id] + [x for x in ids if x != primary_id]
        ordered_specs = [spec_map[mid] for mid in ordered_ids if mid in spec_map]

        # Best-effort: keep settings consistent for legacy code paths
        try:
            s.model_id = primary_id  # type: ignore[attr-defined]
            s.allowed_models = ordered_ids  # type: ignore[attr-defined]
            s.models_config_path = path  # type: ignore[attr-defined]
        except Exception:
            pass

        return ModelsConfig(
            primary_id=str(primary_id),
            model_ids=[str(x) for x in ordered_ids],
            models=ordered_specs,
            defaults={"path": path, **norm_defaults},
        )

    # Fallback to Settings (legacy)
    primary_id = getattr(s, "model_id", None)
    model_ids = list(getattr(s, "all_model_ids", []) or [])

    if not primary_id or not isinstance(primary_id, str) or not primary_id.strip():
        raise AppError(
            code="model_config_invalid",
            message="Primary model id is missing or invalid",
            status_code=500,
            extra={"primary_id": str(primary_id)},
        )

    model_ids = [str(x) for x in model_ids if str(x).strip()]
    model_ids = _dedupe_preserve_order(model_ids)
    if primary_id not in model_ids:
        model_ids.insert(0, primary_id)

    try:
        s.model_id = primary_id  # type: ignore[attr-defined]
        s.allowed_models = model_ids  # type: ignore[attr-defined]
    except Exception:
        pass

    specs = [
        ModelSpec(
            id=mid,
            backend="local",
            load_mode="lazy" if mid != primary_id else "eager",
            capabilities=None,  # legacy path has no per-model caps
            dtype=None,
            device="auto",
            text_only=None,
            max_context=None,
            trust_remote_code=False,
            quantization=None,
            notes="(from settings)",
        )
        for mid in model_ids
    ]

    return ModelsConfig(
        primary_id=str(primary_id),
        model_ids=[str(x) for x in model_ids],
        models=specs,
        defaults={"path": None},
    )