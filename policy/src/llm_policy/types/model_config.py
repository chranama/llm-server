# src/llm_policy/types/models_config.py
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModelDefaults(BaseModel):
    backend: str = "local"
    load_mode: str = "lazy"
    device: str = "auto"
    trust_remote_code: bool = False
    quantization: Any = None
    capabilities: Dict[str, bool] = Field(default_factory=dict)


class ModelEntry(BaseModel):
    id: str

    backend: Optional[str] = None
    load_mode: Optional[str] = None
    dtype: Optional[str] = None
    device: Optional[str] = None
    text_only: Optional[bool] = None
    max_context: Optional[int] = None
    trust_remote_code: Optional[bool] = None
    quantization: Any = None
    capabilities: Dict[str, bool] = Field(default_factory=dict)
    notes: Optional[str] = None

    def effective_capabilities(self, defaults: ModelDefaults) -> Dict[str, bool]:
        # explicit per-model keys override defaults
        out = dict(defaults.capabilities or {})
        for k, v in (self.capabilities or {}).items():
            out[str(k)] = bool(v)
        return out

    def has_capability(self, defaults: ModelDefaults, cap: str) -> bool:
        cap = cap.strip().lower()
        return bool(self.effective_capabilities(defaults).get(cap, False))


class ModelsConfig(BaseModel):
    default_model: str
    defaults: ModelDefaults = Field(default_factory=ModelDefaults)
    models: list[ModelEntry] = Field(default_factory=list)

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def default_entry(self) -> Optional[ModelEntry]:
        return self.get_model(self.default_model)

    def capabilities_for(self, model_id: str) -> Dict[str, bool]:
        m = self.get_model(model_id)
        if m is None:
            return dict(self.defaults.capabilities or {})
        return m.effective_capabilities(self.defaults)

    def is_extract_enabled(self, model_id: str) -> bool:
        return bool(self.capabilities_for(model_id).get("extract", False))