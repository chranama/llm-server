# src/llm_policy/types/eval_artifact.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


# -------------------------
# Contract / issues
# -------------------------


IssueSeverity = Literal["info", "warn", "error"]


class ContractIssue(BaseModel):
    """
    Structured issues found when validating an eval artifact contract.

    Policy should prefer returning a DENY decision with these issues rather than crashing.
    """

    model_config = ConfigDict(extra="ignore")

    severity: IssueSeverity
    code: str
    message: str
    extra: Optional[Dict[str, Any]] = None


# -------------------------
# Summary (summary.json)
# -------------------------


class EvalSummary(BaseModel):
    """
    Typed view over llm_eval summary.json.

    Notes:
    - Tolerant: ignores unknown fields (extra="ignore")
    - Preserves None for "not computed" metrics
    - Provides contract_issues() to support fail-closed policy decisions.
    """

    model_config = ConfigDict(extra="ignore")

    # Versioning (optional, but recommended)
    artifact_version: Optional[str] = None

    # Identity / provenance
    task: str
    run_id: str
    dataset: Optional[str] = None
    split: Optional[str] = None
    schema_id: Optional[str] = None
    base_url: Optional[str] = None
    model_override: Optional[str] = None
    max_examples: Optional[int] = None
    run_dir: Optional[str] = None

    # Core counts
    n_total: int = Field(ge=0)
    n_ok: int = Field(ge=0)

    # Core extraction metric
    schema_validity_rate: Optional[float] = None

    # Repair / cache aggregates (optional)
    n_invalid_initial: Optional[int] = Field(default=None, ge=0)
    n_repair_attempted: Optional[int] = Field(default=None, ge=0)
    n_repair_success: Optional[int] = Field(default=None, ge=0)
    repair_success_rate: Optional[float] = None

    n_cached: Optional[int] = Field(default=None, ge=0)
    cache_hit_rate: Optional[float] = None

    # Error aggregates (optional but very useful)
    error_code_counts: Optional[Dict[str, int]] = None
    status_code_counts: Optional[Dict[str, int]] = None
    error_stage_counts: Optional[Dict[str, int]] = None

    # Field-level / doc-level metrics (optional; task dependent)
    field_exact_match_rate: Optional[Dict[str, float]] = None
    doc_required_exact_match_rate: Optional[float] = None
    required_present_rate: Optional[float] = None

    # Latency
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None

    def contract_issues(self) -> List[ContractIssue]:
        """
        Validate that the summary has enough information for policy decisions.

        This intentionally does NOT raise. Policy should deny with reasons if issues exist.
        """
        issues: List[ContractIssue] = []

        # Required identity
        if not self.task.strip():
            issues.append(
                ContractIssue(
                    severity="error",
                    code="missing_task",
                    message="summary.task is missing/empty",
                )
            )
        if not self.run_id.strip():
            issues.append(
                ContractIssue(
                    severity="error",
                    code="missing_run_id",
                    message="summary.run_id is missing/empty",
                )
            )

        # Required counts
        if self.n_total < 0 or self.n_ok < 0:
            issues.append(
                ContractIssue(
                    severity="error",
                    code="negative_counts",
                    message="summary has negative counts (n_total/n_ok)",
                    extra={"n_total": self.n_total, "n_ok": self.n_ok},
                )
            )

        if self.n_ok > self.n_total:
            issues.append(
                ContractIssue(
                    severity="error",
                    code="inconsistent_counts",
                    message="n_ok cannot exceed n_total",
                    extra={"n_total": self.n_total, "n_ok": self.n_ok},
                )
            )

        if self.n_total == 0:
            issues.append(
                ContractIssue(
                    severity="warn",
                    code="zero_examples",
                    message="n_total == 0; metrics are not meaningful",
                )
            )

        # Core metric presence (for extraction-like tasks)
        if self.schema_validity_rate is None:
            issues.append(
                ContractIssue(
                    severity="warn",
                    code="missing_schema_validity_rate",
                    message="schema_validity_rate is missing; policy may have to rely on n_ok/n_total only",
                )
            )
        else:
            if not (0.0 <= self.schema_validity_rate <= 1.0):
                issues.append(
                    ContractIssue(
                        severity="error",
                        code="invalid_schema_validity_rate",
                        message="schema_validity_rate must be between 0 and 1",
                        extra={"schema_validity_rate": self.schema_validity_rate},
                    )
                )

        # Sanity check: if we have aggregates, they should sum to n_total (soft)
        if self.status_code_counts:
            s = _safe_sum_counts(self.status_code_counts)
            if s is not None and self.n_total and s != self.n_total:
                issues.append(
                    ContractIssue(
                        severity="info",
                        code="status_code_counts_mismatch",
                        message="Sum(status_code_counts) != n_total (can happen if summary is partial)",
                        extra={"sum_status_code_counts": s, "n_total": self.n_total},
                    )
                )

        if self.error_code_counts:
            s = _safe_sum_counts(self.error_code_counts)
            if s is not None and self.n_total and s != self.n_total:
                issues.append(
                    ContractIssue(
                        severity="info",
                        code="error_code_counts_mismatch",
                        message="Sum(error_code_counts) != n_total (can happen if summary is partial)",
                        extra={"sum_error_code_counts": s, "n_total": self.n_total},
                    )
                )

        # Helpful warning for pure-500 runs (like your current case)
        if self.status_code_counts and _count_for_status(self.status_code_counts, "500") == self.n_total and self.n_total:
            issues.append(
                ContractIssue(
                    severity="warn",
                    code="all_500s",
                    message="All requests returned HTTP 500; this is an operational failure, not a model quality signal",
                    extra={"status_code_counts": self.status_code_counts},
                )
            )

        return issues

    def ok_rate(self) -> float:
        if self.n_total <= 0:
            return 0.0
        return float(self.n_ok) / float(self.n_total)

    def operational_failure_rate(self) -> Optional[float]:
        """
        Derived signal: fraction of requests that failed at the server/transport layer.
        Uses status_code_counts if present; otherwise returns None.
        """
        if not self.status_code_counts or self.n_total <= 0:
            return None
        # Treat 0 (transport), 500-599 as operational failures for gating
        failures = 0
        for k, v in self.status_code_counts.items():
            try:
                code = int(k)
            except Exception:
                continue
            if code == 0 or (500 <= code <= 599):
                failures += int(v or 0)
        return float(failures) / float(self.n_total)


# -------------------------
# Per-example rows (results.jsonl)
# -------------------------


class EvalRow(BaseModel):
    """
    Typed view over a single results.jsonl row.

    This must remain tolerant because row shape may vary by task.
    """

    model_config = ConfigDict(extra="ignore")

    doc_id: str
    schema_id: Optional[str] = None

    ok: bool = False
    status_code: Optional[int] = None
    error_code: Optional[str] = None
    error_stage: Optional[str] = None

    latency_ms: Optional[float] = None

    cached: Optional[bool] = None
    repair_attempted: Optional[bool] = None
    model: Optional[str] = None

    expected: Optional[Dict[str, Any]] = None
    predicted: Optional[Dict[str, Any]] = None

    # Task-specific / scoring aides
    field_correct: Optional[Dict[str, Optional[bool]]] = None
    required_present_non_null: Optional[bool] = None
    required_all_correct: Optional[bool] = None

    # Debug payload from client/server
    extra: Optional[Dict[str, Any]] = None

    def is_operational_failure(self) -> bool:
        """
        A conservative classifier used by policy.
        """
        sc = self.status_code
        if sc is None:
            return False
        return sc == 0 or (500 <= sc <= 599)

    def request_id(self) -> Optional[str]:
        """
        Convenience for observability; depends on llm_eval HttpEvalClient capturing request_id in extra.
        """
        if not isinstance(self.extra, dict):
            return None
        rid = self.extra.get("request_id")
        return str(rid) if isinstance(rid, str) and rid.strip() else None


# -------------------------
# Optional wrapper (in-memory)
# -------------------------


@dataclass(frozen=True)
class EvalArtifact:
    """
    Lightweight in-memory bundle.

    Keep this minimal: policy/IO layers can create it after reading from disk.
    """

    summary: EvalSummary
    rows: Optional[List[EvalRow]] = None  # None => summary-only mode

    def contract_issues(self) -> List[ContractIssue]:
        issues = self.summary.contract_issues()

        # Optional consistency check if rows were loaded
        if self.rows is not None:
            if self.summary.n_total and len(self.rows) != self.summary.n_total:
                issues.append(
                    ContractIssue(
                        severity="info",
                        code="results_length_mismatch",
                        message="len(results.jsonl) != summary.n_total",
                        extra={"len_results": len(self.rows), "n_total": self.summary.n_total},
                    )
                )
            # If we can compute n_ok from rows, compare (soft)
            try:
                n_ok_rows = sum(1 for r in self.rows if r.ok)
                if self.summary.n_ok != n_ok_rows:
                    issues.append(
                        ContractIssue(
                            severity="info",
                            code="n_ok_mismatch",
                            message="summary.n_ok != count(rows where ok=true)",
                            extra={"summary_n_ok": self.summary.n_ok, "rows_n_ok": n_ok_rows},
                        )
                    )
            except Exception:
                pass

        return issues


# -------------------------
# Small helpers
# -------------------------


def _safe_sum_counts(counts: Dict[str, int]) -> Optional[int]:
    try:
        return int(sum(int(v or 0) for v in counts.values()))
    except Exception:
        return None


def _count_for_status(counts: Dict[str, int], code_str: str) -> int:
    v = counts.get(code_str)
    try:
        return int(v or 0)
    except Exception:
        return 0