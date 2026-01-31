# src/llm_policy/io/models_yaml.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from llm_policy.types.models_config import ModelsConfig
from llm_policy.utils.fs import read_yaml, write_yaml


@dataclass(frozen=True)
class PatchResult:
    changed: bool
    warnings: list[str]


def load_models_yaml(path: str) -> ModelsConfig:
    obj = read_yaml(path)
    return ModelsConfig.model_validate(obj)


def patch_models_yaml(
    *,
    path: str,
    model_id: str,
    capability: str,
    enable: bool,
    write: bool = True,
) -> PatchResult:
    """
    Patch models.yaml in-place by setting models[i].capabilities[capability] to enable.

    Design constraints (aligned with your llm_server expectations):
      - capabilities are a mapping of booleans (NOT a list)
      - we never infer capabilities: we write explicit true/false
      - if model is missing, do not silently create it (warn instead)
    """
    cap = capability.strip().lower()
    if not cap:
        return PatchResult(changed=False, warnings=["capability was empty"])

    obj = read_yaml(path)
    warnings: list[str] = []
    changed = False

    if not isinstance(obj, dict):
        return PatchResult(changed=False, warnings=[f"models yaml is not a mapping: {path}"])

    models = obj.get("models")
    if not isinstance(models, list):
        return PatchResult(changed=False, warnings=[f"models list missing or invalid in {path}"])

    target: Optional[Dict[str, Any]] = None
    for m in models:
        if isinstance(m, dict) and m.get("id") == model_id:
            target = m
            break

    if target is None:
        return PatchResult(changed=False, warnings=[f"model id not found: {model_id}"])

    caps = target.get("capabilities")
    if caps is None or not isinstance(caps, dict):
        caps = {}
        target["capabilities"] = caps
        warnings.append("model.capabilities missing or invalid; created a new mapping")

    prev = caps.get(cap)
    newv = bool(enable)
    if prev is None or bool(prev) != newv:
        caps[cap] = newv
        changed = True

    # Also ensure defaults.capabilities includes this key (so it is explicit everywhere)
    defaults = obj.get("defaults")
    if not isinstance(defaults, dict):
        defaults = {}
        obj["defaults"] = defaults
        warnings.append("defaults section missing; created a new mapping")

    d_caps = defaults.get("capabilities")
    if d_caps is None or not isinstance(d_caps, dict):
        d_caps = {}
        defaults["capabilities"] = d_caps
        warnings.append("defaults.capabilities missing or invalid; created a new mapping")

    if cap not in d_caps:
        # keep defaults conservative: do not flip default based on a single model decision
        d_caps[cap] = False
        warnings.append(f"defaults.capabilities.{cap} was missing; set to false explicitly")

    if write and (changed or warnings):
        write_yaml(path, obj)

    return PatchResult(changed=changed, warnings=warnings)