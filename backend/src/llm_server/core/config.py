# llm_server/core/config.py
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from pydantic_settings.sources import SettingsSourceCallable  # type: ignore
except Exception:  # pragma: no cover
    SettingsSourceCallable = Callable[..., Dict[str, Any]]  # type: ignore

try:
    import yaml  # pyyaml
except Exception:  # pragma: no cover
    yaml = None


# =========================
# Path resolution helpers
# =========================
def _app_root() -> Path:
    v = (os.environ.get("APP_ROOT") or "").strip()
    if v:
        return Path(v).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (_app_root() / p).resolve()


def _load_app_yaml(path: str) -> Dict[str, Any]:
    """
    Load YAML config and map it to Settings fields.
    Missing file => {}.
    """
    if yaml is None:
        return {}

    p = _resolve_path(path)
    if not p.exists():
        return {}

    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    def g(*keys, default=None):
        cur: Any = raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    out: Dict[str, Any] = {}

    # service
    if (v := g("service", "name")) is not None:
        out["service_name"] = v
    if (v := g("service", "version")) is not None:
        out["version"] = v
    if (v := g("service", "debug")) is not None:
        out["debug"] = v
    if (v := g("service", "env")) is not None:
        out["env"] = v

    # server
    if (v := g("server", "host")) is not None:
        out["host"] = v
    if (v := g("server", "port")) is not None:
        out["port"] = v

    # api
    if (v := g("api", "cors_allowed_origins")) is not None:
        out["cors_allowed_origins"] = v

    # capabilities (support both old and new shapes)
    # New recommended:
    if (v := g("capabilities", "generate")) is not None:
        out["enable_generate"] = v
    if (v := g("capabilities", "extract")) is not None:
        out["enable_extract"] = v
    # Backward compatible:
    if (v := g("capabilities", "enable_generate")) is not None:
        out["enable_generate"] = v
    if (v := g("capabilities", "enable_extract")) is not None:
        out["enable_extract"] = v

    # model (YAML)
    if (v := g("model", "default_id")) is not None:
        out["model_id"] = v
    if (v := g("model", "allowed_models")) is not None:
        out["allowed_models"] = v
    if (v := g("model", "models_config_path")) is not None:
        out["models_config_path"] = v
    if (v := g("model", "dtype")) is not None:
        out["model_dtype"] = v
    if (v := g("model", "device")) is not None:
        out["model_device"] = v

    # ✅ critical runtime toggles can be expressed in YAML too
    if (v := g("model", "model_load_mode")) is not None:
        out["model_load_mode"] = v
    if (v := g("model", "require_model_ready")) is not None:
        out["require_model_ready"] = v
    if (v := g("model", "token_counting")) is not None:
        out["token_counting"] = v

    # redis
    if (v := g("redis", "enabled")) is not None:
        out["redis_enabled"] = v
    if (v := g("redis", "url")) is not None:
        out["redis_url"] = v

    # http
    if (v := g("http", "llm_service_url")) is not None:
        out["llm_service_url"] = v
    if (v := g("http", "client_timeout_seconds")) is not None:
        out["http_client_timeout"] = v

    # limits
    if (v := g("limits", "rate_limit_rpm", "admin")) is not None:
        out["rate_limit_rpm_admin"] = v
    if (v := g("limits", "rate_limit_rpm", "default")) is not None:
        out["rate_limit_rpm_default"] = v
    if (v := g("limits", "rate_limit_rpm", "free")) is not None:
        out["rate_limit_rpm_free"] = v
    if (v := g("limits", "quota_auto_reset_days")) is not None:
        out["quota_auto_reset_days"] = v

    # cache
    if (v := g("cache", "api_key_cache_ttl_seconds")) is not None:
        out["api_key_cache_ttl_seconds"] = v

    return out


def _truthy(v: Any) -> str:
    return "1" if bool(v) else "0"


def _sync_runtime_env(s: "Settings") -> None:
    """
    Keep runtime env vars coherent with Settings.

    Why: some modules still read os.getenv(...) directly. If Settings are loaded
    from YAML/.env/monkeypatch, we want those os.getenv reads to see the same values.
    """
    os.environ["ENV"] = str(s.env)
    os.environ["DEBUG"] = _truthy(s.debug)

    os.environ["ENABLE_GENERATE"] = _truthy(s.enable_generate)
    os.environ["ENABLE_EXTRACT"] = _truthy(s.enable_extract)

    os.environ["REDIS_ENABLED"] = _truthy(s.redis_enabled)

    os.environ["MODEL_LOAD_MODE"] = str(s.model_load_mode)
    os.environ["REQUIRE_MODEL_READY"] = _truthy(s.require_model_ready)
    os.environ["TOKEN_COUNTING"] = _truthy(s.token_counting)


class Settings(BaseSettings):
    # --- config file path ---
    app_config_path: str = Field(
        default="config/app.yaml",
        validation_alias="APP_CONFIG_PATH",
        description="Path to config/app.yaml (YAML defaults). Env vars override YAML.",
    )

    # --- service info ---
    service_name: str = "LLM Server"
    version: str = "0.1.0"
    debug: bool = False

    # --- server ---
    env: str = Field(default="dev")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # --- database ---
    database_url: str = Field(default="postgresql+asyncpg://llm:llm@postgres:5432/llm", validation_alias="DATABASE_URL")

    # --- CORS ---
    cors_allowed_origins: Any = Field(default_factory=lambda: ["*"])

    # --- capabilities ---
    enable_generate: bool = Field(default=True, validation_alias="ENABLE_GENERATE")
    enable_extract: bool = Field(default=True, validation_alias="ENABLE_EXTRACT")

    # --- model config ---
    model_id: str = Field(default="mistralai/Mistral-7B-v0.1")
    allowed_models: List[str] = Field(default_factory=list)
    models_config_path: Optional[str] = Field(default="config/models.yaml")
    model_dtype: Literal["float16", "bfloat16", "float32"] = Field(default="float16")
    model_device: Optional[str] = Field(default=None)

    # --- runtime model behavior toggles (these match your env vars + tests) ---
    model_load_mode: Literal["off", "lazy", "eager"] = Field(default="lazy", validation_alias="MODEL_LOAD_MODE")
    require_model_ready: bool = Field(default=True, validation_alias="REQUIRE_MODEL_READY")
    token_counting: bool = Field(default=True, validation_alias="TOKEN_COUNTING")

    # --- Redis ---
    redis_url: Optional[str] = Field(default=None, validation_alias="REDIS_URL")
    redis_enabled: bool = Field(default=False, validation_alias="REDIS_ENABLED")

    # --- LLM service ---
    llm_service_url: str = Field(default="http://127.0.0.1:9001")
    http_client_timeout: int = Field(default=60)

    # --- rate limits / quotas ---
    rate_limit_rpm_admin: int = 0
    rate_limit_rpm_default: int = 120
    rate_limit_rpm_free: int = 30
    quota_auto_reset_days: int = 30

    # --- API key cache ---
    api_key_cache_ttl_seconds: int = 10

    @property
    def all_model_ids(self) -> List[str]:
        return self.allowed_models or [self.model_id]

    @field_validator("cors_allowed_origins", mode="after")
    @classmethod
    def normalize_cors_origins(cls, v: Any) -> List[str]:
        if v is None:
            return ["*"]
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return ["*"]
            if s == "*":
                return ["*"]
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
            values = [item.strip() for item in s.split(",") if item.strip()]
            return values or ["*"]
        try:
            return [str(v).strip()] if str(v).strip() else ["*"]
        except Exception:
            return ["*"]

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: SettingsSourceCallable,
        env_settings: SettingsSourceCallable,
        dotenv_settings: SettingsSourceCallable,
        file_secret_settings: SettingsSourceCallable,
    ):
        def yaml_settings() -> Dict[str, Any]:
            path = os.getenv("APP_CONFIG_PATH", "config/app.yaml")
            return _load_app_yaml(path)

        # ✅ IMPORTANT: env must override dotenv, not the other way around.
        # Order: defaults -> YAML -> dotenv -> env -> secrets
        return (init_settings, yaml_settings, dotenv_settings, env_settings, file_secret_settings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    _sync_runtime_env(s)
    return s