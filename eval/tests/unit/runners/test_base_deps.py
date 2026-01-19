# tests/unit/runners/test_base_deps.py
from __future__ import annotations

from typing import Any, Optional

import pytest

from llm_eval.runners.base import BaseEvalRunner, EvalConfig, RunnerDeps


class _FakeClient:
    """Simple fake to prove DI wiring works."""
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key


class _Runner(BaseEvalRunner):
    task_name = "fake_runner"

    def __init__(self, *, deps: RunnerDeps):
        super().__init__(base_url="http://example", api_key="KEY", config=EvalConfig(), deps=deps)

    async def _run_impl(self) -> Any:
        return {"ok": True}


def test_make_client_uses_injected_client_factory():
    deps = RunnerDeps(client_factory=lambda base_url, api_key: _FakeClient(base_url, api_key))
    r = _Runner(deps=deps)

    c = r.make_client()
    assert isinstance(c, _FakeClient)
    assert c.base_url == "http://example"
    assert c.api_key == "KEY"


def test_new_run_id_uses_injected_run_id_factory():
    deps = RunnerDeps(
        client_factory=lambda base_url, api_key: _FakeClient(base_url, api_key),
        run_id_factory=lambda: "RID123",
    )
    r = _Runner(deps=deps)

    assert r.new_run_id() == "RID123"
    assert r.new_run_id() == "RID123"  # deterministic


def test_ensure_dir_calls_injected_ensure_dir(tmp_path):
    called = {"n": 0, "path": None}

    def _ensure(p: str) -> None:
        called["n"] += 1
        called["path"] = p

    deps = RunnerDeps(
        client_factory=lambda base_url, api_key: _FakeClient(base_url, api_key),
        ensure_dir=_ensure,
    )
    r = _Runner(deps=deps)

    p = str(tmp_path / "x")
    r.ensure_dir(p)
    assert called["n"] == 1
    assert called["path"] == p


def test_open_file_uses_injected_open_fn(tmp_path):
    captured: dict[str, Any] = {}

    class _Sentinel:
        pass

    sentinel = _Sentinel()

    def _open_fn(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    deps = RunnerDeps(
        client_factory=lambda base_url, api_key: _FakeClient(base_url, api_key),
        open_fn=_open_fn,
    )
    r = _Runner(deps=deps)

    f = r.open_file(str(tmp_path / "a.txt"), "w", encoding="utf-8")
    assert f is sentinel
    assert captured["args"][0].endswith("a.txt")
    assert captured["args"][1] == "w"
    assert captured["kwargs"]["encoding"] == "utf-8"


@pytest.mark.asyncio
async def test_run_updates_config_from_args_before_run_impl():
    observed: dict[str, Optional[int | str]] = {"max_examples": None, "model_override": None}

    class _ObservingRunner(BaseEvalRunner):
        task_name = "observing"

        def __init__(self, *, deps: RunnerDeps):
            super().__init__(base_url="http://example", api_key="KEY", config=EvalConfig(), deps=deps)

        async def _run_impl(self) -> Any:
            observed["max_examples"] = self.config.max_examples
            observed["model_override"] = self.config.model_override
            return {"ok": True}

    deps = RunnerDeps(client_factory=lambda base_url, api_key: _FakeClient(base_url, api_key))
    r = _ObservingRunner(deps=deps)

    out = await r.run(max_examples=7, model_override="m1")
    assert out == {"ok": True}
    assert observed["max_examples"] == 7
    assert observed["model_override"] == "m1"


def test_dataset_overrides_default_empty_dict():
    deps = RunnerDeps(client_factory=lambda base_url, api_key: _FakeClient(base_url, api_key))
    assert isinstance(deps.dataset_overrides, dict)
    assert deps.dataset_overrides == {}


def test_dataset_overrides_can_be_injected():
    def _dummy_iter(**kwargs: Any):
        return []

    deps = RunnerDeps(
        client_factory=lambda base_url, api_key: _FakeClient(base_url, api_key),
        dataset_overrides={"x": _dummy_iter},
    )
    assert "x" in deps.dataset_overrides
    assert deps.dataset_overrides["x"] is _dummy_iter

def test_get_dataset_callable_returns_default_when_no_override():
    deps = RunnerDeps(client_factory=lambda base_url, api_key: _FakeClient(base_url, api_key))
    r = _Runner(deps=deps)

    def _default():
        return "DEFAULT"

    got = r.get_dataset_callable("docred", _default)
    assert got is _default
    assert got() == "DEFAULT"


def test_get_dataset_callable_returns_override_when_present():
    def _override():
        return "OVERRIDE"

    deps = RunnerDeps(
        client_factory=lambda base_url, api_key: _FakeClient(base_url, api_key),
        dataset_overrides={"docred": _override},
    )
    r = _Runner(deps=deps)

    def _default():
        return "DEFAULT"

    got = r.get_dataset_callable("docred", _default)
    assert got is _override
    assert got() == "OVERRIDE"