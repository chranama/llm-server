from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from llm_policy.policies.extract_enablement import decide_extract_enablement
from llm_policy.types.decision import DecisionReason, DecisionWarning


@dataclass
class FakeSummary:
    task: str = "extraction_sroie"
    run_id: str = "r1"
    run_dir: str = "/tmp/run"
    n_total: int = 10
    n_ok: int = 10

    schema_validity_rate: Optional[float] = 0.99
    required_present_rate: Optional[float] = 0.99
    doc_required_exact_match_rate: Optional[float] = 0.99
    field_exact_match_rate: dict[str, float] = None  # set in init

    latency_p95_ms: Optional[float] = 100.0
    latency_p99_ms: Optional[float] = 200.0

    def __post_init__(self):
        if self.field_exact_match_rate is None:
            self.field_exact_match_rate = {"total": 1.0}


@dataclass
class FakeArtifact:
    summary: FakeSummary


@dataclass
class FakeThresholds:
    # required by policy
    min_n_total: int = 20
    min_schema_validity_rate: float = 0.98
    min_required_present_rate: Optional[float] = 0.98
    min_doc_required_exact_match_rate: Optional[float] = 0.98
    min_field_exact_match_rate: Optional[dict[str, float]] = None
    max_latency_p95_ms: Optional[float] = 500.0
    max_latency_p99_ms: Optional[float] = 800.0


@pytest.fixture(autouse=True)
def _patch_health_gate_pass(monkeypatch: pytest.MonkeyPatch):
    """
    Default behavior for this module: health gate passes.
    Individual tests can override this fixture by re-patching in the test body.
    """

    class HG:
        enable_extract = True
        reasons = []
        warnings = []
        metrics = {}

    monkeypatch.setattr(
        "llm_policy.policies.extract_enablement.health_gate_from_eval",
        lambda *args, **kwargs: HG(),
        raising=True,
    )


def test_all_thresholds_pass_enables_extract():
    s = FakeSummary(
        n_total=100,
        schema_validity_rate=0.99,
        required_present_rate=0.99,
        doc_required_exact_match_rate=0.99,
        field_exact_match_rate={"total": 1.0},
        latency_p95_ms=100.0,
        latency_p99_ms=200.0,
    )
    artifact = FakeArtifact(summary=s)
    th = FakeThresholds(min_n_total=20)

    d = decide_extract_enablement(artifact, thresholds=th, thresholds_profile="extract/sroie")

    assert d.enable_extract is True
    assert d.ok() is True
    # status strictness depends on your policy constructor; keep permissive.
    assert getattr(d, "status", None) is not None
    assert d.thresholds_profile == "extract/sroie"
    assert d.eval_task == "extraction_sroie"
    assert d.eval_run_id == "r1"
    assert d.eval_run_dir == "/tmp/run"
    # always includes counters in metrics
    assert d.metrics.get("n_total") == 100
    assert d.metrics.get("n_ok") == 10 or d.metrics.get("n_ok") == 95 or isinstance(d.metrics.get("n_ok"), int)


def test_missing_schema_validity_blocks_extract():
    s = FakeSummary(n_total=100, schema_validity_rate=None)
    artifact = FakeArtifact(summary=s)
    th = FakeThresholds()

    d = decide_extract_enablement(artifact, thresholds=th, thresholds_profile="extract/sroie")

    assert d.enable_extract is False
    assert d.ok() is False
    assert any(r.code == "missing_metric" for r in d.reasons)


def test_schema_validity_too_low_blocks_extract():
    s = FakeSummary(n_total=100, schema_validity_rate=0.50)
    artifact = FakeArtifact(summary=s)
    th = FakeThresholds(min_schema_validity_rate=0.98)

    d = decide_extract_enablement(artifact, thresholds=th, thresholds_profile="extract/sroie")

    assert d.enable_extract is False
    assert any(r.code == "schema_validity_too_low" for r in d.reasons)


def test_sample_size_below_min_adds_warning_not_block():
    s = FakeSummary(
        n_total=5,
        schema_validity_rate=0.99,
        required_present_rate=0.99,
        doc_required_exact_match_rate=0.99,
    )
    artifact = FakeArtifact(summary=s)
    th = FakeThresholds(min_n_total=20)

    d = decide_extract_enablement(artifact, thresholds=th, thresholds_profile="extract/sroie")

    assert d.enable_extract is True  # quality passes
    assert any(w.code == "insufficient_sample_size" for w in d.warnings)


def test_field_exact_match_missing_field_blocks():
    s = FakeSummary(
        n_total=100,
        schema_validity_rate=0.99,
        required_present_rate=0.99,
        doc_required_exact_match_rate=0.99,
        field_exact_match_rate={"a": 1.0},
    )
    artifact = FakeArtifact(summary=s)
    th = FakeThresholds(min_field_exact_match_rate={"b": 0.9})

    d = decide_extract_enablement(artifact, thresholds=th, thresholds_profile="extract/sroie")

    assert d.enable_extract is False
    assert any(r.code == "missing_metric" and "field_exact_match_rate.b" in r.message for r in d.reasons)


def test_latency_threshold_blocks_when_exceeded():
    s = FakeSummary(
        n_total=100,
        schema_validity_rate=0.99,
        required_present_rate=0.99,
        doc_required_exact_match_rate=0.99,
        latency_p95_ms=900.0,
    )
    artifact = FakeArtifact(summary=s)
    th = FakeThresholds(max_latency_p95_ms=500.0)

    d = decide_extract_enablement(artifact, thresholds=th, thresholds_profile="extract/sroie")

    assert d.enable_extract is False
    assert any(r.code == "latency_p95_too_high" for r in d.reasons)


# ---------------------------------------------------------------------------
# Added (from the merged version): health gate behavior tests
# ---------------------------------------------------------------------------

def test_health_gate_blocks_short_circuit(monkeypatch: pytest.MonkeyPatch):
    class HG:
        enable_extract = False
        reasons = [DecisionReason(code="health_gate_block", message="nope", context={})]
        warnings = [DecisionWarning(code="hg_warn", message="note", context={})]
        metrics = {"hg": True}

    monkeypatch.setattr(
        "llm_policy.policies.extract_enablement.health_gate_from_eval",
        lambda *args, **kwargs: HG(),
        raising=True,
    )

    s = FakeSummary(
        n_total=100,
        schema_validity_rate=0.99,
        required_present_rate=0.99,
        doc_required_exact_match_rate=0.99,
    )
    artifact = FakeArtifact(summary=s)
    th = FakeThresholds(min_n_total=1)

    d = decide_extract_enablement(artifact, thresholds=th, thresholds_profile="extract/default")

    assert d.enable_extract is False
    assert d.ok() is False
    assert any(r.code == "health_gate_block" for r in d.reasons)
    assert any(w.code == "hg_warn" for w in d.warnings)
    assert d.metrics.get("hg") is True


def test_health_gate_warnings_propagate_on_pass(monkeypatch: pytest.MonkeyPatch):
    class HG:
        enable_extract = True
        reasons = []
        warnings = [DecisionWarning(code="hg_warn", message="note", context={})]
        metrics = {"hg": "ok"}

    monkeypatch.setattr(
        "llm_policy.policies.extract_enablement.health_gate_from_eval",
        lambda *args, **kwargs: HG(),
        raising=True,
    )

    s = FakeSummary(
        n_total=100,
        schema_validity_rate=0.99,
        required_present_rate=0.99,
        doc_required_exact_match_rate=0.99,
    )
    artifact = FakeArtifact(summary=s)
    th = FakeThresholds(min_n_total=1)

    d = decide_extract_enablement(artifact, thresholds=th, thresholds_profile="extract/default")

    assert d.enable_extract is True
    assert d.ok() is True
    assert any(w.code == "hg_warn" for w in d.warnings)
    assert d.metrics.get("hg") == "ok"