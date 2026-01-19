# tests/unit/runners/test_runner_http_calls.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Optional

import pytest

from llm_eval.client.http_client import ExtractOk, GenerateOk
from llm_eval.runners.base import EvalConfig, RunnerDeps
from llm_eval.runners.docred_runner import GenerateDocREDRunner
from llm_eval.runners.extraction_runner import ExtractionEvalRunner
from llm_eval.runners.paraloq_json_extraction_runner import GenerateParaloqJsonExtractionRunner
from llm_eval.runners.squad_v2_runner import GenerateSquadV2Runner

from tests.fakes.fake_examples import DocREDExample, ParaloqExample, ReceiptExample, SquadV2Example
from tests.fakes.fake_http_client import FakeHttpClient


def _mk_deps_with_overrides(
    base: RunnerDeps,
    overrides: dict[str, Callable[..., Iterable[Any]]],
) -> RunnerDeps:
    return RunnerDeps(
        client_factory=base.client_factory,
        run_id_factory=base.run_id_factory,
        ensure_dir=base.ensure_dir,
        open_fn=base.open_fn,
        dataset_overrides=overrides,
    )


@pytest.mark.asyncio
async def test_squad_v2_runner_generate_call_shape(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter_squad_v2(*, split: str, max_samples: Optional[int] = None):
        assert split == "validation"
        xs = [
            SquadV2Example(
                id="q1",
                context="Paris is the capital of France.",
                question="What is the capital of France?",
                title="France",
                answers=["Paris"],
                is_impossible=False,
            )
        ]
        return xs[: (max_samples or len(xs))]

    deps2 = _mk_deps_with_overrides(deps, {"iter_squad_v2": _iter_squad_v2})

    fake_client.generate_queue.append(GenerateOk(model="mX", output_text="Paris", cached=False, latency_ms=1.0))

    r = GenerateSquadV2Runner(
        base_url="http://svc",
        api_key="KEY",
        config=EvalConfig(max_examples=1, model_override="mX"),
        deps=deps2,
    )
    await r.run()

    assert len(fake_client.generate_calls) == 1
    call = fake_client.generate_calls[0]

    assert isinstance(call.get("prompt"), str)
    assert call["max_new_tokens"] == 64
    assert call["temperature"] == 0.0
    assert call["model"] == "mX"

    # runner doesn't pass cache for generate (today); ensure we didn't accidentally set it
    assert call.get("cache") is None


@pytest.mark.asyncio
async def test_docred_runner_generate_call_shape(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter_docred(*, split: str, max_samples: Optional[int] = None):
        assert split == "validation"
        xs = [
            DocREDExample(
                id="d1",
                text="Barack Obama was born in Hawaii.",
                expected={"title": "Obama", "entities": ["Barack Obama", "Hawaii"], "relations": []},
            )
        ]
        return xs[: (max_samples or len(xs))]

    deps2 = _mk_deps_with_overrides(deps, {"iter_docred": _iter_docred})

    fake_client.generate_queue.append(GenerateOk(model="mX", output_text='{"relations":[]}', cached=False, latency_ms=1.0))

    r = GenerateDocREDRunner(
        base_url="http://svc",
        api_key="KEY",
        config=EvalConfig(max_examples=1, model_override="mX"),
        deps=deps2,
    )
    await r.run()

    assert len(fake_client.generate_calls) == 1
    call = fake_client.generate_calls[0]

    assert isinstance(call.get("prompt"), str)
    assert call["max_new_tokens"] == 512
    assert call["temperature"] == 0.0
    assert call["model"] == "mX"
    assert call.get("cache") is None


@pytest.mark.asyncio
async def test_paraloq_runner_generate_call_shape(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter_paraloq(*, split: str, max_samples: Optional[int] = None):
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

    deps2 = _mk_deps_with_overrides(deps, {"iter_paraloq_json_extraction": _iter_paraloq})

    fake_client.generate_queue.append(GenerateOk(model="mX", output_text='{"invoice_id":"INV-123"}', cached=False, latency_ms=1.0))

    r = GenerateParaloqJsonExtractionRunner(
        base_url="http://svc",
        api_key="KEY",
        config=EvalConfig(max_examples=1, model_override="mX"),
        deps=deps2,
    )
    await r.run()

    assert len(fake_client.generate_calls) == 1
    call = fake_client.generate_calls[0]

    assert isinstance(call.get("prompt"), str)
    assert call["max_new_tokens"] == 512
    assert call["temperature"] == 0.0
    assert call["model"] == "mX"
    assert call.get("cache") is None


@pytest.mark.asyncio
async def test_extraction_runner_extract_call_shape(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter_receipts(*, split: str, schema_id: str, max_samples: Optional[int] = None):
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

    deps2 = _mk_deps_with_overrides(deps, {"iter_voxel51_scanned_receipts": _iter_receipts})

    fake_client.extract_queue.append(
        ExtractOk(
            schema_id="sroie_receipt_v1",
            model="mX",
            data={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            cached=False,
            repair_attempted=True,
            latency_ms=1.0,
        )
    )

    r = ExtractionEvalRunner(
        base_url="http://svc",
        api_key="KEY",
        config=EvalConfig(max_examples=1, model_override="mX"),
        schema_id="sroie_receipt_v1",
        split="train",
        deps=deps2,
    )
    await r.run()

    assert len(fake_client.extract_calls) == 1
    call = fake_client.extract_calls[0]

    assert call["schema_id"] == "sroie_receipt_v1"
    assert isinstance(call.get("text"), str)
    assert call["model"] == "mX"
    assert call["max_new_tokens"] == 512
    assert call["temperature"] == 0.0

    # runner sets these explicitly
    assert call["cache"] is False
    assert call["repair"] is True