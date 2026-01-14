# src/llm_server/core/config.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional, List, Any, Dict

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import SettingsSourceCallable

try:
    import yaml  # pyyaml
except Exception:  # pragma: no cover
    yaml = None


def _load_app_yaml(path: str) -> Dict[str, Any]:
    """
    Load config/app.yaml and flatten it into Settings fields.
    Missing file => {}.
    """
    p = Path(path)
    if not p.exists() or yaml is None:
        return {}

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
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

    # model
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

    # redis
    if (v := g("redis", "enabled")) is not None:
        out["redis_enabled"] = v

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
    database_url: str = Field(default="postgresql+asyncpg://llm:llm@postgres:5432/llm")

    # --- CORS ---
    cors_allowed_origins: Any = Field(default_factory=lambda: ["*"])

    # --- model config ---
    model_id: str = Field(default="mistralai/Mistral-7B-v0.1")
    allowed_models: List[str] = Field(default_factory=list)
    models_config_path: Optional[str] = Field(default="config/models.yaml")
    model_dtype: Literal["float16", "bfloat16", "float32"] = Field(default="float16")
    model_device: Optional[str] = Field(default=None)

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
        # (unchanged)
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
        def yaml_settings(settings: BaseSettings) -> Dict[str, Any]:
            # Use the *init* value for app_config_path if provided, else default.
            path = getattr(settings, "app_config_path", "config/app.yaml")
            return _load_app_yaml(path)

        # Order matters: defaults -> YAML -> env -> dotenv -> secrets
        return (init_settings, yaml_settings, env_settings, dotenv_settings, file_secret_settings)


settings = Settings()