# tests/conftest.py
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from llm_eval.runners.base import RunnerDeps
from tests.fakes.fake_http_client import FakeHttpClient


@pytest.fixture
def fake_client() -> FakeHttpClient:
    return FakeHttpClient()


@pytest.fixture
def run_id_factory() -> Callable[[], str]:
    # deterministic, simple default
    return lambda: "RUNID_TEST_0001"


@pytest.fixture
def deps(fake_client: FakeHttpClient, run_id_factory: Callable[[], str]) -> RunnerDeps:
    """
    Default runner deps for unit tests:
      - deterministic run_id
      - fake HTTP client
      - empty dataset_overrides (tests opt-in)
    """
    return RunnerDeps(
        client_factory=lambda base_url, api_key: fake_client,
        run_id_factory=run_id_factory,
    )


@pytest.fixture
def with_dataset_overrides() -> Callable[[RunnerDeps, dict[str, Any]], RunnerDeps]:
    """
    Helper: RunnerDeps is frozen; return a copy with dataset_overrides swapped.
    """
    def _mk(base: RunnerDeps, overrides: dict[str, Any]) -> RunnerDeps:
        return RunnerDeps(
            client_factory=base.client_factory,
            run_id_factory=base.run_id_factory,
            ensure_dir=base.ensure_dir,
            open_fn=base.open_fn,
            dataset_overrides=overrides,
        )

    return _mk