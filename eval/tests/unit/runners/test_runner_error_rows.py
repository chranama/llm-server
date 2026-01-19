# tests/unit/runners/test_runner_error_rows.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Optional

import pytest

from llm_eval.client.http_client import ExtractErr, GenerateErr
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


def _assert_common_generate_err_row(row: dict[str, Any]) -> None:
    assert isinstance(row.get("status_code"), int)
    assert row["status_code"] != 200
    assert isinstance(row.get("error_code"), str)
    assert row.get("latency_ms") is not None
    assert isinstance(row["latency_ms"], (int, float))
    # model must be None on error rows
    assert row.get("model") is None


@pytest.mark.asyncio
async def test_squad_v2_runner_error_row_includes_required_fields(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter_squad_v2(*, split: str, max_samples: Optional[int] = None):
        assert split == "validation"
        xs = [
            SquadV2Example(
                id="q_err",
                context="Some context.",
                question="Some question?",
                title="T",
                answers=[],
                is_impossible=True,
            )
        ]
        return xs[: (max_samples or len(xs))]

    deps2 = _mk_deps_with_overrides(deps, {"iter_squad_v2": _iter_squad_v2})

    fake_client.generate_queue.append(
        GenerateErr(
            status_code=502,
            error_code="bad_gateway",
            message="upstream fail",
            extra={"detail": "x"},
            latency_ms=33.0,
        )
    )

    r = GenerateSquadV2Runner(
        base_url="http://svc",
        api_key="KEY",
        config=EvalConfig(max_examples=1, model_override="mX"),
        deps=deps2,
    )
    payload = await r.run()

    assert payload["summary"]["task"] == "generate_squad_v2"
    assert len(payload["results"]) == 1
    row = payload["results"][0]
    _assert_common_generate_err_row(row)
    assert row["id"] == "q_err"
    # runner truncates error message to 500 chars; still should be non-empty here
    assert "upstream" in (row.get("error") or "").lower()
    # extra should be dict or None
    assert isinstance(row.get("extra"), (dict, type(None)))


@pytest.mark.asyncio
async def test_docred_runner_error_row_includes_required_fields(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter_docred(*, split: str, max_samples: Optional[int] = None):
        assert split == "validation"
        xs = [
            DocREDExample(
                id="d_err",
                text="Text.",
                expected={"title": "T", "entities": ["A", "B"], "relations": []},
            )
        ]
        return xs[: (max_samples or len(xs))]

    deps2 = _mk_deps_with_overrides(deps, {"iter_docred": _iter_docred})

    fake_client.generate_queue.append(
        GenerateErr(
            status_code=503,
            error_code="service_unavailable",
            message="down",
            extra=None,
            latency_ms=44.0,
        )
    )

    r = GenerateDocREDRunner(
        base_url="http://svc",
        api_key="KEY",
        config=EvalConfig(max_examples=1, model_override="mX"),
        deps=deps2,
    )
    payload = await r.run()

    assert payload["summary"]["task"] == "generate_docred"
    assert len(payload["results"]) == 1
    row = payload["results"][0]
    _assert_common_generate_err_row(row)
    assert row["id"] == "d_err"
    assert row["split"] == "validation"
    assert isinstance(row.get("n_entities"), int)


@pytest.mark.asyncio
async def test_paraloq_runner_error_row_has_schema_and_predicted_text(deps: RunnerDeps, fake_client: FakeHttpClient):
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
                id="p_err",
                text="Invoice ID: INV-123",
                schema=schema,
                expected={"invoice_id": "INV-123"},
            )
        ]
        return xs[: (max_samples or len(xs))]

    deps2 = _mk_deps_with_overrides(deps, {"iter_paraloq_json_extraction": _iter_paraloq})

    fake_client.generate_queue.append(
        GenerateErr(
            status_code=500,
            error_code="server_error",
            message="boom",
            extra={"stage": "generate"},
            latency_ms=55.0,
        )
    )

    r = GenerateParaloqJsonExtractionRunner(
        base_url="http://svc",
        api_key="KEY",
        config=EvalConfig(max_examples=1, model_override="mX"),
        deps=deps2,
    )
    payload = await r.run()

    assert payload["summary"]["task"] == "generate_paraloq_json_extraction"
    assert len(payload["results"]) == 1
    row = payload["results"][0]
    _assert_common_generate_err_row(row)
    assert row["id"] == "p_err"
    # Runner pins schema_id and provides empty predicted_text on error
    assert row.get("schema_id") == "inline_json_schema"
    assert row.get("predicted_text") == ""
    # Should still compute metrics dict even on error (scoring uses empty predicted_text)
    assert isinstance(row.get("metrics"), dict)


@pytest.mark.asyncio
async def test_extraction_runner_error_row_includes_stage_and_latency(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter_receipts(*, split: str, schema_id: str, max_samples: Optional[int] = None):
        assert split == "train"
        xs = [
            ReceiptExample(
                id="r_err",
                schema_id=schema_id,
                text="Company: ACME\nDate: 2024-01-01\nTotal: 10.00",
                expected={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            )
        ]
        return xs[: (max_samples or len(xs))]

    deps2 = _mk_deps_with_overrides(deps, {"iter_voxel51_scanned_receipts": _iter_receipts})

    fake_client.extract_queue.append(
        ExtractErr(
            status_code=422,
            error_code="validation_error",
            message="bad json",
            extra={"stage": "schema_validate"},
            latency_ms=66.0,
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
    payload = await r.run()

    assert payload["summary"]["task"] == "extraction_sroie"
    assert len(payload["results"]) == 1
    row = payload["results"][0]

    # error row invariants
    assert row["doc_id"] == "r_err"
    assert row["schema_id"] == "sroie_receipt_v1"
    assert row["ok"] is False
    assert row["status_code"] == 422
    assert row["error_code"] == "validation_error"
    assert row["latency_ms"] == 66.0
    assert row.get("model") is None

    # stage extracted from resp.extra
    assert row.get("error_stage") == "schema_validate"

    # predicted should be None on error
    assert row.get("predicted") is None

    # extraction runner includes extra dict (or None)
    assert isinstance(row.get("extra"), (dict, type(None)))
    if isinstance(row.get("extra"), dict):
        assert row["extra"].get("stage") == "schema_validate"