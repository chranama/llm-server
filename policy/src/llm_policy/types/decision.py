# policy/src/llm_policy/types/decision.py
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DecisionStatus(str, Enum):
    """
    Minimal tri-state:
      - allow: explicitly pass / enable
      - deny: explicitly fail / disable
      - unknown: policy could not determine (treat as deny for gating)
    """
    allow = "allow"
    deny = "deny"
    unknown = "unknown"


class DecisionReason(BaseModel):
    code: str
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


class DecisionWarning(BaseModel):
    code: str
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """
    Canonical policy output object.

    This should be stable across time because it becomes:
      - a CLI output artifact
      - a record the backend can expose through an admin endpoint
      - something you may persist to disk in policy_out/

    Notes:
      - "status" is the primary truth.
      - "enable_extract" is included because it’s the concrete action you patch into models.yaml.
      - if status != allow, treat enable_extract as False (fail-closed).
    """

    policy: str = Field(..., description="Policy name, e.g. extract_enablement")
    status: DecisionStatus = Field(default=DecisionStatus.unknown)

    # Concrete action flags (policy-specific but useful)
    enable_extract: bool = Field(default=False)

    # Traceability / provenance
    thresholds_profile: Optional[str] = Field(default=None)
    thresholds_version: Optional[str] = Field(default=None)
    eval_run_dir: Optional[str] = Field(default=None)
    eval_task: Optional[str] = Field(default=None)
    eval_run_id: Optional[str] = Field(default=None)
    model_id: Optional[str] = Field(default=None)

    # Human/diagnostic payload
    reasons: List[DecisionReason] = Field(default_factory=list)
    warnings: List[DecisionWarning] = Field(default_factory=list)

    # Useful scalar metrics extracted from summary.json for reporting/debugging.
    # Keep values JSON-serializable.
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # Optional: record contract issues count so we can fail-closed or display “inputs invalid”.
    contract_errors: int = Field(default=0)
    contract_warnings: int = Field(default=0)

    def ok(self) -> bool:
        """
        "ok" is used as CLI exit code / gating boolean.
        Fail-closed:
          - contract errors => not ok
          - unknown => not ok
        """
        if self.contract_errors > 0:
            return False
        return self.status == DecisionStatus.allow

    @classmethod
    def allow_extract(
        cls,
        *,
        policy: str,
        thresholds_profile: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        reasons: Optional[List[DecisionReason]] = None,
        warnings: Optional[List[DecisionWarning]] = None,
        **kwargs: Any,
    ) -> "Decision":
        return cls(
            policy=policy,
            status=DecisionStatus.allow,
            enable_extract=True,
            thresholds_profile=thresholds_profile,
            metrics=metrics or {},
            reasons=reasons or [],
            warnings=warnings or [],
            **kwargs,
        )

    @classmethod
    def deny_extract(
        cls,
        *,
        policy: str,
        thresholds_profile: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        reasons: Optional[List[DecisionReason]] = None,
        warnings: Optional[List[DecisionWarning]] = None,
        **kwargs: Any,
    ) -> "Decision":
        return cls(
            policy=policy,
            status=DecisionStatus.deny,
            enable_extract=False,
            thresholds_profile=thresholds_profile,
            metrics=metrics or {},
            reasons=reasons or [],
            warnings=warnings or [],
            **kwargs,
        )