# tests/unit/runners/test_runner_patchability.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional

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


def _mk_deps_with_overrides(
    base: RunnerDeps,
    overrides: dict[str, Callable[..., Iterable[Any]]],
) -> RunnerDeps:
    """
    RunnerDeps is frozen, so create a new one preserving everything
    except dataset_overrides.
    """
    return RunnerDeps(
        client_factory=base.client_factory,
        run_id_factory=base.run_id_factory,
        ensure_dir=base.ensure_dir,
        open_fn=base.open_fn,
        dataset_overrides=overrides,
    )


@pytest.mark.asyncio
async def test_docred_uses_dataset_override(deps: RunnerDeps, fake_client: FakeHttpClient):
    calls: dict[str, Any] = {}

    def _iter_docred(*, split: str, max_samples: Optional[int] = None):
        calls["split"] = split
        calls["max_samples"] = max_samples
        return [
            DocREDExample(
                id="d1",
                text="Barack Obama was born in Hawaii.",
                expected={"title": "Obama", "entities": ["Barack Obama", "Hawaii"], "relations": []},
            )
        ]

    deps2 = _mk_deps_with_overrides(deps, {"iter_docred": _iter_docred})

    # One OK response for the single example
    fake_client.generate_queue.append(
        GenerateOk(model="m", output_text='{"relations": []}', cached=False, latency_ms=1.0)
    )

    r = GenerateDocREDRunner(
        base_url="http://x",
        api_key="k",
        config=EvalConfig(max_examples=1),
        deps=deps2,
    )
    payload = await r.run()

    assert calls["split"] == "validation"
    assert calls["max_samples"] == 1
    assert payload["summary"]["task"] == "generate_docred"
    assert len(payload["results"]) == 1


@pytest.mark.asyncio
async def test_squad_v2_uses_dataset_override(deps: RunnerDeps, fake_client: FakeHttpClient):
    calls: dict[str, Any] = {}

    def _iter_squad_v2(*, split: str, max_samples: Optional[int] = None):
        calls["split"] = split
        calls["max_samples"] = max_samples
        return [
            SquadV2Example(
                id="q1",
                context="Paris is the capital of France.",
                question="What is the capital of France?",
                title="France",
                answers=["Paris"],
                is_impossible=False,
            )
        ]

    deps2 = _mk_deps_with_overrides(deps, {"iter_squad_v2": _iter_squad_v2})

    fake_client.generate_queue.append(
        GenerateOk(model="m", output_text="Paris", cached=False, latency_ms=1.0)
    )

    r = GenerateSquadV2Runner(
        base_url="http://x",
        api_key="k",
        config=EvalConfig(max_examples=1),
        deps=deps2,
    )
    payload = await r.run()

    assert calls["split"] == "validation"
    assert calls["max_samples"] == 1
    assert payload["summary"]["task"] == "generate_squad_v2"
    assert len(payload["results"]) == 1


@pytest.mark.asyncio
async def test_paraloq_uses_dataset_override(deps: RunnerDeps, fake_client: FakeHttpClient):
    calls: dict[str, Any] = {}

    def _iter_paraloq(*, split: str, max_samples: Optional[int] = None):
        calls["split"] = split
        calls["max_samples"] = max_samples
        schema = {
            "type": "object",
            "properties": {"invoice_id": {"type": "string"}},
            "required": ["invoice_id"],
            "additionalProperties": False,
        }
        return [
            ParaloqExample(
                id="p1",
                text="Invoice ID: INV-123",
                schema=schema,
                expected={"invoice_id": "INV-123"},
            )
        ]

    deps2 = _mk_deps_with_overrides(deps, {"iter_paraloq_json_extraction": _iter_paraloq})

    fake_client.generate_queue.append(
        GenerateOk(model="m", output_text='{"invoice_id":"INV-123"}', cached=False, latency_ms=1.0)
    )

    r = GenerateParaloqJsonExtractionRunner(
        base_url="http://x",
        api_key="k",
        config=EvalConfig(max_examples=1),
        deps=deps2,
    )
    payload = await r.run()

    assert calls["split"] == "train"
    assert calls["max_samples"] == 1
    assert payload["summary"]["task"] == "generate_paraloq_json_extraction"
    assert len(payload["results"]) == 1


@pytest.mark.asyncio
async def test_receipts_uses_dataset_override_and_passes_schema_id(deps: RunnerDeps, fake_client: FakeHttpClient):
    calls: dict[str, Any] = {}

    def _iter_receipts(*, split: str, schema_id: str, max_samples: Optional[int] = None):
        calls["split"] = split
        calls["schema_id"] = schema_id
        calls["max_samples"] = max_samples
        return [
            ReceiptExample(
                id="r1",
                schema_id=schema_id,
                text="Company: ACME\nDate: 2024-01-01\nTotal: 10.00",
                expected={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            )
        ]

    deps2 = _mk_deps_with_overrides(deps, {"iter_voxel51_scanned_receipts": _iter_receipts})

    fake_client.extract_queue.append(
        ExtractOk(
            schema_id="sroie_receipt_v1",
            model="m",
            data={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            cached=False,
            repair_attempted=False,
            latency_ms=1.0,
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

    assert calls["split"] == "train"
    assert calls["schema_id"] == "sroie_receipt_v1"
    assert calls["max_samples"] == 1
    assert payload["summary"]["task"] == "extraction_sroie"
    assert len(payload["results"]) == 1