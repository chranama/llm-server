# src/llm_server/services/llm.py
from __future__ import annotations

import os
import pwd
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers as tf

from llm_server.core.config import get_settings
from llm_server.core.errors import AppError
from llm_server.services.llm_api import HttpLLMClient
from llm_server.services.llm_config import load_models_config, ModelSpec
from llm_server.services.llm_registry import MultiModelManager

# -----------------------------------
# Configuration helpers (local)
# -----------------------------------

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

DEFAULT_STOPS: List[str] = ["\nUser:", "\nuser:", "User:", "###"]


def _real_user_home() -> str:
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except Exception:
        return os.path.expanduser("~")


def _device_from_settings(cfg) -> str:
    # explicit override wins
    dev = getattr(cfg, "model_device", None)
    if isinstance(dev, str) and dev.strip():
        return dev.strip()
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _resolve_hf_home(cfg) -> str:
    cfg_val = getattr(cfg, "hf_home", None)
    if isinstance(cfg_val, str) and cfg_val.strip():
        return cfg_val.strip()

    env_val = os.environ.get("HF_HOME")
    if env_val and env_val.strip():
        return env_val.strip()

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg and xdg.strip():
        return os.path.join(xdg, "huggingface")

    return os.path.join(_real_user_home(), ".cache", "huggingface")


def _configure_hf_cache_env(cfg) -> dict[str, str]:
    hf_home = _resolve_hf_home(cfg)
    hub_cache = os.environ.get("HF_HUB_CACHE") or os.path.join(hf_home, "hub")

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = hub_cache
    os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE") or hub_cache
    os.environ["XDG_CACHE_HOME"] = os.environ.get("XDG_CACHE_HOME") or os.path.dirname(hf_home)

    try:
        os.makedirs(hf_home, exist_ok=True)
        os.makedirs(hub_cache, exist_ok=True)
    except Exception as e:
        raise AppError(
            code="hf_cache_unwritable",
            message="Hugging Face cache directory is not writable",
            status_code=500,
            extra={
                "hf_home": hf_home,
                "hf_hub_cache": hub_cache,
                "error": str(e),
                "env_HOME": os.environ.get("HOME"),
                "real_user_home": _real_user_home(),
            },
        ) from e

    return {
        "hf_home": hf_home,
        "hf_hub_cache": hub_cache,
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE", ""),
        "env_HOME": os.environ.get("HOME", ""),
        "real_user_home": _real_user_home(),
    }


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _caps_meta(sp: Optional[ModelSpec]) -> Optional[list[str]]:
    """
    Normalize per-model capabilities for registry metadata.

    Contract (aligned with MultiModelManager + unit tests):
      - None => unspecified (registry treats as allow-all)
      - list[str] => allowlist
      - dict[str,bool] => keys with True are enabled
      - str => single capability
    Returned value is a stable, sorted list[str] (or None).
    """
    if sp is None:
        return None

    caps = getattr(sp, "capabilities", None)
    if caps is None:
        return None

    def _norm_one(x: object) -> Optional[str]:
        if not isinstance(x, str):
            return None
        s = x.strip().lower()
        return s or None

    out: list[str] = []

    if isinstance(caps, dict):
        for k, v in caps.items():
            kk = _norm_one(k)
            if not kk:
                continue
            if bool(v):
                out.append(kk)

    elif isinstance(caps, (list, tuple, set)):
        for x in caps:
            s = _norm_one(x)
            if s:
                out.append(s)

    elif isinstance(caps, str):
        s = _norm_one(caps)
        if s:
            out.append(s)

    else:
        # unknown type => treat as unspecified/fail-open
        return None

    # de-dupe + stable ordering (sorted)
    out = sorted(set(out))
    return out or None

def _make_http_client(*, base_url: str, model_id: str, timeout: int = 60):
    try:
        return HttpLLMClient(base_url=base_url, model_id=model_id, timeout=timeout)
    except TypeError:
        # test FakeHttpClient doesn’t accept timeout
        return HttpLLMClient(base_url=base_url, model_id=model_id)

# ===================================
# LOCAL MODEL MANAGER
# ===================================


class ModelManager:
    """
    Local (in-process) HF Transformers backend.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        dtype: torch.dtype,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id: str = model_id
        self._device: str = device
        self._dtype: torch.dtype = dtype
        self._trust_remote_code: bool = bool(trust_remote_code)
        self._tokenizer = None
        self._model = None

    @classmethod
    def from_settings(cls, cfg) -> "ModelManager":
        dtype_str = getattr(cfg, "model_dtype", "float16")
        dtype = DTYPE_MAP.get(dtype_str, torch.float16)
        device = _device_from_settings(cfg)
        model_id = getattr(cfg, "model_id", "mistralai/Mistral-7B-v0.1")
        return cls(model_id=model_id, device=device, dtype=dtype, trust_remote_code=False)

    def _err_ctx(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "device": str(self._device),
            "dtype": str(self._dtype),
            "trust_remote_code": bool(self._trust_remote_code),
            "env_HOME": os.environ.get("HOME"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
            "real_user_home": _real_user_home(),
        }

    def is_loaded(self) -> bool:
        return (self._tokenizer is not None) and (self._model is not None)

    def ensure_loaded(self) -> None:
        try:
            cfg = get_settings()
            cache_ctx = _configure_hf_cache_env(cfg)
            cache_dir = cache_ctx["hf_hub_cache"]

            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    use_fast=True,
                    cache_dir=cache_dir,
                    trust_remote_code=self._trust_remote_code,
                )
                if getattr(self._tokenizer, "pad_token", None) is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

            if self._model is None:
                hf_cfg = AutoConfig.from_pretrained(
                    self.model_id,
                    trust_remote_code=self._trust_remote_code,
                )

                dtype = self._dtype
                if str(self._device) == "mps" and dtype == torch.bfloat16:
                    dtype = torch.float16

                try:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        cache_dir=cache_dir,
                        trust_remote_code=self._trust_remote_code,
                    )
                except ValueError:
                    archs = getattr(hf_cfg, "architectures", None) or []
                    if not archs:
                        raise
                    arch = archs[0]
                    model_cls = getattr(tf, arch, None)
                    if model_cls is None:
                        raise
                    self._model = model_cls.from_pretrained(
                        self.model_id,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        cache_dir=cache_dir,
                        trust_remote_code=self._trust_remote_code,
                    )

                self._model.to(self._device)
                self._model.eval()

        except AppError:
            raise
        except Exception as e:
            raise AppError(
                code="model_load_failed",
                message="Failed to load local model",
                status_code=500,
                extra={**self._err_ctx(), "error": str(e)},
            ) from e

    @staticmethod
    def _truncate_on_stop(text: str, stop: Optional[List[str]]) -> str:
        if not stop:
            return text
        cut_positions = [text.find(s) for s in stop if s in text]
        if cut_positions:
            cut = min(cut_positions)
            if cut >= 0:
                return text[:cut]
        return text

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int | None = 0,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        try:
            self.ensure_loaded()

            tok = self._tokenizer
            model = self._model
            if tok is None or model is None:
                raise RuntimeError("Model not loaded")

            stops = stop if (stop and len(stop) > 0) else DEFAULT_STOPS
            inputs = tok(prompt, return_tensors="pt").to(self._device)

            use_top_k = top_k if (top_k is not None and top_k > 0) else None
            use_temperature = temperature if (temperature is not None and temperature > 0) else 0.0
            use_top_p = top_p if top_p is not None else 0.95

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=use_temperature > 0,
                temperature=use_temperature,
                top_p=use_top_p,
                top_k=use_top_k,
                repetition_penalty=repetition_penalty,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )[0]

            text = tok.decode(output_ids, skip_special_tokens=True)
            tail = text[len(prompt) :]
            return self._truncate_on_stop(tail, stops)

        except AppError:
            raise
        except Exception as e:
            raise AppError(
                code="model_generate_failed",
                message="Local model generation failed",
                status_code=500,
                extra={**self._err_ctx(), "error": str(e)},
            ) from e


# ===================================
# BUILDER / WIRING
# ===================================


def build_llm_from_settings() -> Any:
    """
    Construct the service LLM backend.

    Medium gating behavior (safe default):
      - By default, expose ONLY the primary model locally.
      - If ENABLE_MULTI_MODELS=1, expose additional models from models.yaml.

    Any model with load_mode == "off" is excluded from the registry entirely.

    Per-model capabilities are propagated into registry metadata for API gating/UI.

    IMPORTANT:
      - meta["capabilities"] is:
          * None  => unspecified (registry treats as allow-all)
          * dict  => explicitly specified per-model caps
    """
    cfg = load_models_config()
    primary_id = cfg.primary_id
    s = get_settings()
    http_timeout = int(getattr(s, "http_client_timeout", 60) or 60)

    multi_enabled = _truthy_env("ENABLE_MULTI_MODELS", default=False)

    # Global load mode override (env wins, then settings)
    global_load_mode = (os.getenv("MODEL_LOAD_MODE") or getattr(s, "model_load_mode", None) or "").strip().lower()

    # If globally off, return an empty registry (unit tests expect this shape).
    if global_load_mode == "off":
        return MultiModelManager(models={}, default_id=primary_id, model_meta={})

    spec_map: Dict[str, ModelSpec] = {sp.id: sp for sp in cfg.models}
    ordered_ids: List[str] = [
        mid for mid in cfg.model_ids if (spec_map.get(mid) and spec_map[mid].load_mode != "off")
    ]

    if not ordered_ids:
        raise AppError(
            code="model_config_invalid",
            message="No enabled models after applying load_mode=off filters",
            status_code=500,
            extra={"primary_id": primary_id, "configured_ids": cfg.model_ids},
        )

    if primary_id in ordered_ids:
        ordered_ids = [primary_id] + [x for x in ordered_ids if x != primary_id]
    else:
        primary_id = ordered_ids[0]

    if not multi_enabled:
        ordered_ids = [primary_id]

    def _need_service_url_for(mid: str) -> bool:
        sp = spec_map.get(mid)
        return bool(sp and sp.backend == "remote")

    if any(_need_service_url_for(mid) for mid in ordered_ids):
        if not getattr(s, "llm_service_url", None):
            raise AppError(
                code="remote_models_require_llm_service_url",
                message="Remote models configured but llm_service_url is not set",
                status_code=500,
                extra={"primary_id": primary_id, "model_ids": ordered_ids},
            )

    # Single model shortcut
    if len(ordered_ids) == 1:
        sp = spec_map.get(primary_id)
        backend = (sp.backend if sp else "local")

        if backend == "remote":
            return _make_http_client(base_url=s.llm_service_url, model_id=primary_id, timeout=http_timeout)

        mgr = ModelManager.from_settings(s)
        mgr.model_id = primary_id
        if sp is not None:
            trc = bool(getattr(sp, "trust_remote_code", False))
            # internal flag for your real ModelManager
            if hasattr(mgr, "_trust_remote_code"):
                mgr._trust_remote_code = trc  # type: ignore[attr-defined]
            # public attribute for test + compatibility
            setattr(mgr, "trust_remote_code", trc)
        return mgr

    models: Dict[str, Any] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    for mid in ordered_ids:
        sp = spec_map.get(mid)
        backend = (sp.backend if sp else "local")
        caps = _caps_meta(sp)

        if backend == "remote":
            models[mid] = _make_http_client(base_url=s.llm_service_url, model_id=mid, timeout=http_timeout)
            meta[mid] = {
                "backend": "http_remote",
                "load_mode": "remote",
                "capabilities": caps,  # None => unspecified; dict => explicit
            }
            continue

        mm = ModelManager.from_settings(s)
        mm.model_id = mid
        if sp is not None:
            trc = bool(getattr(sp, "trust_remote_code", False))
            if hasattr(mm, "_trust_remote_code"):
                mm._trust_remote_code = trc  # type: ignore[attr-defined]
            setattr(mm, "trust_remote_code", trc)

        models[mid] = mm
        meta[mid] = {
            "backend": "local_hf",
            "load_mode": "lazy(default eager-by-lifespan)" if mid == primary_id else "lazy",
            "capabilities": caps,  # None => unspecified; dict => explicit
        }

    return MultiModelManager(models=models, default_id=primary_id, model_meta=meta)