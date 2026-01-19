# tests/unit/runners/test_runner_contracts.py
from __future__ import annotations

from typing import Any, Callable, Iterable

import pytest

from llm_eval.client.http_client import ExtractOk, GenerateErr, GenerateOk
from llm_eval.runners.base import EvalConfig, RunnerDeps
from llm_eval.runners.docred_runner import GenerateDocREDRunner
from llm_eval.runners.extraction_runner import ExtractionEvalRunner
from llm_eval.runners.paraloq_json_extraction_runner import GenerateParaloqJsonExtractionRunner
from llm_eval.runners.squad_v2_runner import GenerateSquadV2Runner

from tests.fakes.fake_examples import DocREDExample, ParaloqExample, ReceiptExample, SquadV2Example
from tests.fakes.fake_http_client import FakeHttpClient


# -------------------------
# Dataset override keys (MUST match runner implementation)
# -------------------------
DOCRED_KEY = "iter_docred"
SQUADV2_KEY = "iter_squad_v2"
PARALOQ_KEY = "iter_paraloq_json_extraction"
RECEIPTS_KEY = "iter_voxel51_scanned_receipts"


def _deps_with_datasets(
    base_deps: RunnerDeps,
    dataset_overrides: dict[str, Callable[..., Iterable[Any]]],
) -> RunnerDeps:
    """
    RunnerDeps is frozen, so build a new one by copying existing deps fields.
    """
    return RunnerDeps(
        client_factory=base_deps.client_factory,
        run_id_factory=base_deps.run_id_factory,
        ensure_dir=base_deps.ensure_dir,
        open_fn=base_deps.open_fn,
        dataset_overrides=dataset_overrides,
    )


@pytest.mark.asyncio
async def test_squad_v2_runner_nested_contract_via_dataset_overrides(deps, fake_client: FakeHttpClient):
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
                context="This context does not contain the answer.",
                question="Who invented the teleporter?",
                title="Nope",
                answers=[],
                is_impossible=True,
            ),
        ]
        return xs[: (max_samples or len(xs))]

    deps2 = _deps_with_datasets(deps, {SQUADV2_KEY: _iter})

    fake_client.generate_queue.extend(
        [
            GenerateOk(model="m", output_text="Paris", cached=False, latency_ms=12.0),
            GenerateErr(status_code=502, error_code="bad_gateway", message="oops", extra=None, latency_ms=34.0),
        ]
    )

    r = GenerateSquadV2Runner(
        base_url="http://x",
        api_key="k",
        config=EvalConfig(max_examples=2),
        deps=deps2,
    )
    payload = await r.run()

    assert isinstance(payload, dict)
    assert set(payload.keys()) >= {"summary", "results", "report_text", "config"}

    summary = payload["summary"]
    results = payload["results"]

    assert summary["task"] == "generate_squad_v2"
    assert summary["run_id"] == "RUNID_TEST_0001"
    assert summary["split"] == "validation"
    assert isinstance(results, list)
    assert len(results) == 2

    err_row = [x for x in results if x.get("status_code") and x.get("status_code") != 200][0]
    assert err_row["error_code"] == "bad_gateway"
    assert err_row["latency_ms"] == 34.0


@pytest.mark.asyncio
async def test_docred_runner_nested_contract_via_dataset_overrides(deps, fake_client: FakeHttpClient):
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

    deps2 = _deps_with_datasets(deps, {DOCRED_KEY: _iter})

    fake_client.generate_queue.append(
        GenerateOk(model="m", output_text='{"relations": []}', cached=False, latency_ms=10.0)
    )

    r = GenerateDocREDRunner(
        base_url="http://x",
        api_key="k",
        config=EvalConfig(max_examples=1),
        deps=deps2,
    )
    payload = await r.run()

    assert set(payload.keys()) >= {"summary", "results", "report_text", "config"}
    assert payload["summary"]["task"] == "generate_docred"
    assert payload["summary"]["run_id"] == "RUNID_TEST_0001"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["model"] == "m"


@pytest.mark.asyncio
async def test_paraloq_runner_nested_contract_via_dataset_overrides(deps, fake_client: FakeHttpClient):
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

    deps2 = _deps_with_datasets(deps, {PARALOQ_KEY: _iter})

    fake_client.generate_queue.append(
        GenerateOk(model="m", output_text='{"invoice_id":"INV-123"}', cached=False, latency_ms=9.0)
    )

    r = GenerateParaloqJsonExtractionRunner(
        base_url="http://x",
        api_key="k",
        config=EvalConfig(max_examples=1),
        deps=deps2,
    )
    payload = await r.run()

    assert payload["summary"]["task"] == "generate_paraloq_json_extraction"
    assert payload["summary"]["run_id"] == "RUNID_TEST_0001"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["model"] == "m"


@pytest.mark.asyncio
async def test_extraction_runner_nested_contract_via_dataset_overrides(deps, fake_client: FakeHttpClient):
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

    deps2 = _deps_with_datasets(deps, {RECEIPTS_KEY: _iter})

    fake_client.extract_queue.append(
        ExtractOk(
            schema_id="sroie_receipt_v1",
            model="m",
            data={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            cached=False,
            repair_attempted=False,
            latency_ms=15.0,
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

    assert payload["summary"]["task"] == "extraction_sroie"
    assert payload["summary"]["run_id"] == "RUNID_TEST_0001"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["model"] == "m"