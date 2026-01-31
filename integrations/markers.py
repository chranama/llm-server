# integrations/markers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# -------------------------
# Marker names (string constants)
# -------------------------
MODE_GENERATE_ONLY = "generate_only"
MODE_FULL = "full"

REQUIRES_API_KEY = "requires_api_key"
REQUIRES_REDIS = "requires_redis"
REQUIRES_DB = "requires_db"
REQUIRES_METRICS = "requires_metrics"

# Optional “where am I running”
TARGET_COMPOSE = "compose"
TARGET_HOST = "host"
TARGET_K8S = "k8s"


@dataclass(frozen=True)
class ModelsSnapshot:
    default_model: str
    deployment_capabilities: Dict[str, bool]
    models: List[Dict[str, Any]]

    @classmethod
    def from_json(cls, x: Dict[str, Any]) -> "ModelsSnapshot":
        return cls(
            default_model=str(x.get("default_model") or ""),
            deployment_capabilities=dict(x.get("deployment_capabilities") or {}),
            models=list(x.get("models") or []),
        )

    def deployment_supports(self, cap: str) -> Optional[bool]:
        v = self.deployment_capabilities.get(cap)
        return bool(v) if v is not None else None

    def any_model_supports(self, cap: str) -> bool:
        for m in self.models:
            caps = m.get("capabilities") or {}
            if caps.get(cap) is True:
                return True
        return False

    def all_models_have(self, cap: str, expected: bool) -> bool:
        for m in self.models:
            caps = m.get("capabilities") or {}
            if bool(caps.get(cap)) is not expected:
                return False
        return True


def assert_generate_only(snapshot: ModelsSnapshot) -> None:
    """
    Generate-only contract (based on /v1/models):
      - deployment_capabilities.generate == True
      - deployment_capabilities.extract == False
      - every model has generate=True, extract=False
    """
    dep = snapshot.deployment_capabilities
    gen = dep.get("generate")
    ext = dep.get("extract")

    assert gen is True, f"deployment_capabilities.generate expected True, got {gen}"
    assert ext is False, f"deployment_capabilities.extract expected False, got {ext}"

    assert snapshot.models, "no models returned from /v1/models"
    assert snapshot.all_models_have("generate", True), "expected all models to have generate=true"
    assert snapshot.all_models_have("extract", False), "expected all models to have extract=false"


def assert_full(snapshot: ModelsSnapshot) -> None:
    """
    Full contract (based on your config/models.full.yaml intent):
      - deployment_capabilities.generate == True
      - deployment_capabilities.extract == True
      - at least one model has extract=True
      - any model with extract=True must also have generate=True
    """
    dep = snapshot.deployment_capabilities
    gen = dep.get("generate")
    ext = dep.get("extract")

    assert gen is True, f"deployment_capabilities.generate expected True, got {gen}"
    assert ext is True, f"deployment_capabilities.extract expected True, got {ext}"

    assert snapshot.models, "no models returned from /v1/models"
    assert snapshot.any_model_supports("extract"), "expected at least one model with extract=true"

    bad = []
    for m in snapshot.models:
        caps = m.get("capabilities") or {}
        if caps.get("extract") is True and caps.get("generate") is not True:
            bad.append((m.get("id"), caps))
    assert not bad, f"invalid caps (extract true but generate not true): {bad}"