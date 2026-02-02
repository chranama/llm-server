# backend/tests/unit/test_api_resolve_model_unit.py
from __future__ import annotations

import pytest

from llm_server.api import deps
from llm_server.core.errors import AppError


class FakeMultiModelManager:
    def __init__(self, models: dict, default_id: str, default_for_cap: dict | None = None):
        self.models = models
        self.default_id = default_id
        self._default_for_cap = default_for_cap or {}

    def __contains__(self, k: str) -> bool:
        return k in self.models

    def __getitem__(self, k: str):
        return self.models[k]

    def list_models(self):
        return list(self.models.keys())

    def default_for_capability(self, cap: str):
        return self._default_for_cap.get(cap, self.default_id)


def patch_allowed(monkeypatch, allowed: list[str], default_mid: str):
    monkeypatch.setattr(deps, "allowed_model_ids", lambda *args, **kwargs: allowed, raising=True)
    monkeypatch.setattr(deps, "default_model_id_from_settings", lambda *args, **kwargs: default_mid, raising=True)
    monkeypatch.setattr(deps, "MultiModelManager", FakeMultiModelManager, raising=True)


# -----------------------------------------------------------------------------
# MultiModelManager
# -----------------------------------------------------------------------------


def test_resolve_model_multimodel_override_missing(monkeypatch):
    llm = FakeMultiModelManager(models={"m1": object()}, default_id="m1")
    patch_allowed(monkeypatch, ["m1"], "m1")

    with pytest.raises(AppError) as e:
        deps.resolve_model(llm, "nope", capability=None, request=None)

    assert e.value.code == "model_missing"
    assert e.value.status_code == 500


def test_resolve_model_multimodel_no_override_uses_default_id(monkeypatch):
    llm = FakeMultiModelManager(models={"m1": object(), "m2": object()}, default_id="m2")
    patch_allowed(monkeypatch, ["m1", "m2"], "m2")

    mid, _ = deps.resolve_model(llm, None, capability="generate", request=None)
    assert mid == "m2"


def test_resolve_model_multimodel_extract_no_override_uses_default_for_capability(monkeypatch):
    llm = FakeMultiModelManager(
        models={"gen": object(), "ext": object()},
        default_id="gen",
        default_for_cap={"extract": "ext"},
    )
    patch_allowed(monkeypatch, ["gen", "ext"], "gen")

    mid, _ = deps.resolve_model(llm, None, capability="extract", request=None)
    assert mid == "ext"


def test_resolve_model_multimodel_allowed_models_excludes_chosen(monkeypatch):
    llm = FakeMultiModelManager(models={"m1": object()}, default_id="m1")
    patch_allowed(monkeypatch, ["other"], "m1")

    with pytest.raises(AppError) as e:
        deps.resolve_model(llm, None, capability=None, request=None)

    assert e.value.code == "model_not_allowed"
    assert e.value.status_code == 400


# -----------------------------------------------------------------------------
# Dict registry (legacy)
# -----------------------------------------------------------------------------


def test_resolve_model_dict_registry_override_missing(monkeypatch):
    llm = {"a": object()}
    patch_allowed(monkeypatch, ["a", "b"], "a")

    # override is allowed-list OK, but missing from dict -> model_missing (500)
    with pytest.raises(AppError) as e:
        deps.resolve_model(llm, "b", capability=None, request=None)

    assert e.value.code == "model_missing"
    assert e.value.status_code == 500


def test_resolve_model_dict_registry_override_not_allowed(monkeypatch):
    llm = {"a": object()}
    patch_allowed(monkeypatch, ["a"], "a")

    with pytest.raises(AppError) as e:
        deps.resolve_model(llm, "b", capability=None, request=None)

    assert e.value.code == "model_not_allowed"
    assert e.value.status_code == 400


def test_resolve_model_dict_registry_default_fallback_order(monkeypatch):
    llm = {"a": object(), "b": object()}
    patch_allowed(monkeypatch, ["a", "b"], "b")

    mid, _ = deps.resolve_model(llm, None, capability=None, request=None)
    assert mid == "b"


# -----------------------------------------------------------------------------
# Single backend
# -----------------------------------------------------------------------------


def test_resolve_model_single_backend_override_allowed_and_not_allowed(monkeypatch):
    llm = object()
    patch_allowed(monkeypatch, ["m1"], "m1")

    # allowed override
    mid, _ = deps.resolve_model(llm, "m1", capability=None, request=None)
    assert mid == "m1"

    # not allowed override
    with pytest.raises(AppError) as e:
        deps.resolve_model(llm, "m2", capability=None, request=None)

    assert e.value.code == "model_not_allowed"
    assert e.value.status_code == 400