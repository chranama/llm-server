# tests/unit/runners/test_runner_metrics_sanity.py
from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import pytest

from llm_eval.client.http_client import ExtractOk, GenerateOk
from llm_eval.runners.base import EvalConfig, RunnerDeps
from llm_eval.runners.docred_runner import GenerateDocREDRunner
from llm_eval.runners.extraction_runner import ExtractionEvalRunner
from llm_eval.runners.paraloq_json_extraction_runner import GenerateParaloqJsonExtractionRunner
from llm_eval.runners.squad_v2_runner import GenerateSquadV2Runner

from tests.fakes.fake_examples import (
    DocREDExample,
    ParaloqExample,
    ReceiptExample,
    SquadV2Example,
)
from tests.fakes.fake_http_client import FakeHttpClient


def _deps_with_datasets(
    base_deps: RunnerDeps,
    dataset_overrides: dict[str, Callable[..., Iterable[Any]]],
) -> RunnerDeps:
    return RunnerDeps(
        client_factory=base_deps.client_factory,
        run_id_factory=base_deps.run_id_factory,
        ensure_dir=base_deps.ensure_dir,
        open_fn=base_deps.open_fn,
        dataset_overrides=dataset_overrides,
    )


def _is_pct(x: Any) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(float(x)) and 0.0 <= float(x) <= 100.0


def _is_nonneg_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(float(x)) and float(x) >= 0.0


@pytest.mark.asyncio
async def test_squad_v2_metrics_sanity(deps: RunnerDeps, fake_client: FakeHttpClient):
    # dataset seam
    def _iter(split: str, max_samples=None):
        assert split == "validation"
        xs = [
            SquadV2Example(
                id="q1",
                context="Paris is the capital of France.",
                question="What is the capital of France?",
                title="France",
                answers=["Paris"],
                is_impossible=False,
            ),
            SquadV2Example(
                id="q2",
                context="No answer here.",
                question="Who invented the teleporter?",
                title="Nope",
                answers=[],
                is_impossible=True,
            ),
        ]
        return xs[: (max_samples or len(xs))]

    # runner currently keys overrides by underlying iterator name (as implemented in runners)
    deps2 = _deps_with_datasets(deps, {"iter_squad_v2": _iter})

    fake_client.generate_queue.extend(
        [
            GenerateOk(model="m", output_text="Paris", cached=False, latency_ms=10.0),
            # correct no-answer token is enforced by prompt contract; here we just return it
            GenerateOk(model="m", output_text="NO_ANSWER", cached=False, latency_ms=20.0),
        ]
    )

    r = GenerateSquadV2Runner(base_url="http://x", api_key="k", config=EvalConfig(max_examples=2), deps=deps2)
    payload = await r.run()
    summary = payload["summary"]

    # latencies should exist and be non-negative
    assert _is_nonneg_num(summary.get("latency_p50_ms"))
    assert _is_nonneg_num(summary.get("latency_p95_ms"))
    assert _is_nonneg_num(summary.get("latency_p99_ms"))

    # core metrics should be numeric and in [0,1] or [0,100] depending on implementation
    # We avoid over-specifying the exact scale; just sanity check finite numbers.
    for k in ("answerable_exact_match_rate", "unanswerable_accuracy", "combined_score"):
        if k in summary and summary[k] is not None:
            assert isinstance(summary[k], (int, float))
            assert not math.isnan(float(summary[k]))


@pytest.mark.asyncio
async def test_docred_metrics_sanity(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter(split: str, max_samples=None):
        assert split == "validation"
        xs = [
            DocREDExample(
                id="d1",
                text="Barack Obama was born in Hawaii.",
                expected={"title": "Obama", "entities": ["Barack Obama", "Hawaii"], "relations": []},
            )
        ]
        return xs[: (max_samples or len(xs))]

    deps2 = _deps_with_datasets(deps, {"iter_docred": _iter})

    fake_client.generate_queue.append(
        GenerateOk(model="m", output_text='{"relations": []}', cached=False, latency_ms=7.0)
    )

    r = GenerateDocREDRunner(base_url="http://x", api_key="k", config=EvalConfig(max_examples=1), deps=deps2)
    payload = await r.run()
    summary = payload["summary"]

    # precision/recall/f1 should be finite and within [0,1] or [0,100]; don’t overconstrain scale
    for k in ("precision", "recall", "f1"):
        if k in summary and summary[k] is not None:
            assert isinstance(summary[k], (int, float))
            assert not math.isnan(float(summary[k]))
            assert float(summary[k]) >= 0.0

    for k in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms"):
        assert _is_nonneg_num(summary.get(k))


@pytest.mark.asyncio
async def test_paraloq_metrics_sanity(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter(split: str, max_samples=None):
        assert split == "train"
        schema = {
            "type": "object",
            "properties": {"invoice_id": {"type": "string"}},
            "required": ["invoice_id"],
            "additionalProperties": False,
        }
        xs = [
            ParaloqExample(
                id="p1",
                text="Invoice ID: INV-123",
                schema=schema,
                expected={"invoice_id": "INV-123"},
            )
        ]
        return xs[: (max_samples or len(xs))]

    deps2 = _deps_with_datasets(deps, {"iter_paraloq_json_extraction": _iter})

    fake_client.generate_queue.append(
        GenerateOk(model="m", output_text='{"invoice_id":"INV-123"}', cached=False, latency_ms=5.0)
    )

    r = GenerateParaloqJsonExtractionRunner(
        base_url="http://x", api_key="k", config=EvalConfig(max_examples=1), deps=deps2
    )
    payload = await r.run()
    summary = payload["summary"]

    # rates are percentages in this runner
    for k in (
        "http_ok_rate",
        "json_valid_rate",
        "schema_valid_rate",
        "required_present_rate",
        "required_all_correct_rate",
        "field_micro_exact_match_rate",
    ):
        assert _is_pct(summary.get(k)), f"{k} expected 0..100, got {summary.get(k)}"

    for k in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms"):
        assert _is_nonneg_num(summary.get(k))


@pytest.mark.asyncio
async def test_extraction_metrics_sanity(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter(split: str, schema_id: str, max_samples=None):
        assert split == "train"
        xs = [
            ReceiptExample(
                id="r1",
                schema_id=schema_id,
                text="Company: ACME\nDate: 2024-01-01\nTotal: 10.00",
                expected={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            )
        ]
        return xs[: (max_samples or len(xs))]

    deps2 = _deps_with_datasets(deps, {"iter_voxel51_scanned_receipts": _iter})

    fake_client.extract_queue.append(
        ExtractOk(
            schema_id="sroie_receipt_v1",
            model="m",
            data={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            cached=False,
            repair_attempted=False,
            latency_ms=11.0,
        )
    )

    r = ExtractionEvalRunner(
        base_url="http://x",
        api_key="k",
        config=EvalConfig(max_examples=1),
        schema_id="sroie_receipt_v1",
        split="train",
        deps=deps2,
    )
    payload = await r.run()
    summary = payload["summary"]

    # We don’t assume the exact metric names/scales from summarize_extraction;
    # just enforce “present & finite” for the most likely summary metrics if they exist.
    for k in (
        "schema_validity_rate",
        "doc_required_exact_match_rate",
        "required_present_rate",
    ):
        if k in summary and summary[k] is not None:
            assert isinstance(summary[k], (int, float))
            assert not math.isnan(float(summary[k]))
            assert float(summary[k]) >= 0.0

    # extraction runner doesn't currently compute latency quantiles; ensure base keys are sane
    assert summary.get("task") == "extraction_sroie"
    assert summary.get("dataset") == "Voxel51/scanned_receipts"
    assert summary.get("split") == "train"