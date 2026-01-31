# src/llm_policy/types/thresholds.py
from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class ExtractThresholds(BaseModel):
    """
    Thresholds used by extract enablement policy.

    Keep these stable and explicit; avoid "magic" inferred behavior.
    """

    # Hard health gating (catastrophic failures)
    max_error_rate: float = 0.05  # fraction of examples allowed to error (non-2xx)
    max_5xx_rate: float = 0.01    # fraction of examples allowed to be 5xx
    max_transport_error_rate: float = 0.01

    # Quality gating
    min_schema_validity_rate: float = 0.90
    min_required_present_rate: Optional[float] = None
    min_doc_required_exact_match_rate: Optional[float] = None

    # Per-field exact match expectations (optional)
    min_field_exact_match_rate: Dict[str, float] = Field(default_factory=dict)

    # Latency gating (optional)
    max_latency_p95_ms: Optional[float] = None
    max_latency_p99_ms: Optional[float] = None

    # Minimum sample size to treat decision as meaningful
    min_n_total: int = 20


class ThresholdBundle(BaseModel):
    """
    Future-proof wrapper: lets you extend with other policy thresholds later.
    """
    extract: ExtractThresholds = Field(default_factory=ExtractThresholds)