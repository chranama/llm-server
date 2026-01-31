# src/llm_policy/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from llm_policy.types.thresholds import ExtractThresholds
from llm_policy.utils.fs import read_yaml


@dataclass(frozen=True)
class PolicyConfig:
    """
    Policy runtime config.

    - thresholds_root: directory containing threshold profiles (e.g. thresholds/extract/*.yaml)
    """
    thresholds_root: str

    @staticmethod
    def default() -> "PolicyConfig":
        # package-relative: src/llm_policy/thresholds
        here = Path(__file__).resolve().parent
        root = str(here / "thresholds")
        env = os.getenv("LLM_POLICY_THRESHOLDS_ROOT")
        if env and env.strip():
            root = env.strip()
        return PolicyConfig(thresholds_root=root)


def _normalize_profile(profile: Optional[str]) -> str:
    """
    Accept:
      - None -> "extract/default"
      - "extract/sroie"
      - "sroie" (shorthand -> "extract/sroie")
    """
    if not profile or not str(profile).strip():
        return "extract/default"

    p = str(profile).strip().replace("\\", "/").strip("/")
    if "/" not in p:
        p = f"extract/{p}"
    return p


def _load_thresholds_yaml(path: str) -> dict[str, Any]:
    obj = read_yaml(path)
    return obj if isinstance(obj, dict) else {}


def load_extract_thresholds(
    *,
    cfg: Optional[PolicyConfig] = None,
    profile: Optional[str] = None,
) -> Tuple[str, ExtractThresholds]:
    """
    Load ExtractThresholds from:
      <thresholds_root>/<profile>.yaml

    Returns (resolved_profile, thresholds).
    """
    cfg = cfg or PolicyConfig.default()
    resolved = _normalize_profile(profile)

    root = Path(cfg.thresholds_root).resolve()
    yml_path = (root / f"{resolved}.yaml").resolve()

    # Safety: prevent path traversal out of thresholds_root
    if str(yml_path).find(str(root)) != 0:
        raise ValueError("Invalid profile path (path traversal)")

    if not yml_path.exists():
        # fall back to extract/default.yaml if requested profile missing
        fallback = (root / "extract" / "default.yaml").resolve()
        if not fallback.exists():
            raise FileNotFoundError(f"Thresholds file not found: {yml_path} (and no fallback {fallback})")
        obj = _load_thresholds_yaml(str(fallback))
        return "extract/default", ExtractThresholds.model_validate(obj)

    obj = _load_thresholds_yaml(str(yml_path))
    return resolved, ExtractThresholds.model_validate(obj)