# tests/unit/runners/test_runner_empty_dataset.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Optional

import pytest

from llm_eval.runners.base import EvalConfig, RunnerDeps
from llm_eval.runners.docred_runner import GenerateDocREDRunner
from llm_eval.runners.extraction_runner import ExtractionEvalRunner
from llm_eval.runners.paraloq_json_extraction_runner import GenerateParaloqJsonExtractionRunner
from llm_eval.runners.squad_v2_runner import GenerateSquadV2Runner

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


def _assert_nested_contract(payload: dict[str, Any]) -> None:
    assert isinstance(payload, dict)
    assert isinstance(payload.get("summary"), dict)
    assert isinstance(payload.get("results"), list)
    assert "report_text" in payload
    assert "config" in payload


@pytest.mark.asyncio
async def test_squad_v2_empty_dataset_returns_empty_results(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter_squad_v2(*, split: str, max_samples: Optional[int] = None):
        assert split == "validation"
        return []

    deps2 = _mk_deps_with_overrides(deps, {"iter_squad_v2": _iter_squad_v2})

    r = GenerateSquadV2Runner(
        base_url="http://svc",
        api_key="KEY",
        config=EvalConfig(max_examples=0, model_override="mX"),
        deps=deps2,
    )
    payload = await r.run()
    _assert_nested_contract(payload)

    assert payload["summary"]["task"] == "generate_squad_v2"
    assert payload["summary"]["run_id"] == "RUNID_TEST_0001"
    assert payload["summary"]["split"] == "validation"
    assert payload["results"] == []
    # No HTTP calls should have happened
    assert fake_client.generate_calls == []


@pytest.mark.asyncio
async def test_docred_empty_dataset_returns_empty_results(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter_docred(*, split: str, max_samples: Optional[int] = None):
        assert split == "validation"
        return []

    deps2 = _mk_deps_with_overrides(deps, {"iter_docred": _iter_docred})

    r = GenerateDocREDRunner(
        base_url="http://svc",
        api_key="KEY",
        config=EvalConfig(max_examples=0, model_override="mX"),
        deps=deps2,
    )
    payload = await r.run()
    _assert_nested_contract(payload)

    assert payload["summary"]["task"] == "generate_docred"
    assert payload["summary"]["run_id"] == "RUNID_TEST_0001"
    assert payload["summary"]["split"] == "validation"
    assert payload["results"] == []
    assert fake_client.generate_calls == []


@pytest.mark.asyncio
async def test_paraloq_empty_dataset_returns_empty_results(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter_paraloq(*, split: str, max_samples: Optional[int] = None):
        assert split == "train"
        return []

    deps2 = _mk_deps_with_overrides(deps, {"iter_paraloq_json_extraction": _iter_paraloq})

    r = GenerateParaloqJsonExtractionRunner(
        base_url="http://svc",
        api_key="KEY",
        config=EvalConfig(max_examples=0, model_override="mX"),
        deps=deps2,
    )
    payload = await r.run()
    _assert_nested_contract(payload)

    assert payload["summary"]["task"] == "generate_paraloq_json_extraction"
    assert payload["summary"]["run_id"] == "RUNID_TEST_0001"
    assert payload["summary"]["split"] == "train"
    assert payload["results"] == []
    assert fake_client.generate_calls == []


@pytest.mark.asyncio
async def test_extraction_empty_dataset_returns_empty_results(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter_receipts(*, split: str, schema_id: str, max_samples: Optional[int] = None):
        assert split == "train"
        assert schema_id == "sroie_receipt_v1"
        return []

    deps2 = _mk_deps_with_overrides(deps, {"iter_voxel51_scanned_receipts": _iter_receipts})

    r = ExtractionEvalRunner(
        base_url="http://svc",
        api_key="KEY",
        config=EvalConfig(max_examples=0, model_override="mX"),
        schema_id="sroie_receipt_v1",
        split="train",
        deps=deps2,
    )
    payload = await r.run()
    _assert_nested_contract(payload)

    assert payload["summary"]["task"] == "extraction_sroie"
    assert payload["summary"]["run_id"] == "RUNID_TEST_0001"
    assert payload["summary"]["split"] == "train"
    assert payload["results"] == []
    assert fake_client.extract_calls == []