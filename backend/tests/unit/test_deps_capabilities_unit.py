# backend/tests/unit/test_deps_capabilities_unit.py
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_server.api import deps
from llm_server.core.errors import AppError


# ----------------------------
# Helpers: request + settings
# ----------------------------
def _req_with_settings(enable_extract: bool = True, enable_generate: bool = True):
    settings = SimpleNamespace(enable_extract=enable_extract, enable_generate=enable_generate)
    app = SimpleNamespace(state=SimpleNamespace(settings=settings, llm=None))
    return SimpleNamespace(app=app)


# ----------------------------
# Helpers: fake models config
# ----------------------------
class FakeModelsConfig:
    def __init__(self, defaults: dict, models: list):
        self.defaults = defaults
        self.models = models


class FakeModelSpec:
    def __init__(self, id: str, capabilities=None):
        self.id = id
        self.capabilities = capabilities


def _clear_cache_or_fail() -> None:
    if not hasattr(deps, "clear_models_config_cache"):
        raise AssertionError(
            "deps.clear_models_config_cache() is required for these tests. "
            "Expose it (recommended) or update tests to clear the cache another way."
        )
    deps.clear_models_config_cache()


def _install_models_config(monkeypatch: pytest.MonkeyPatch, cfg: FakeModelsConfig) -> None:
    """
    Patch the underlying loader and clear the lru cache so each test is isolated.
    NOTE: We DO NOT patch deployment_capabilities here; we rely on request.app.state.settings
    via deps.settings_from_request() for deployment gating in request-aware tests.
    """
    monkeypatch.setattr(deps, "load_models_config", lambda: cfg, raising=True)
    _clear_cache_or_fail()


# ============================================================
# Policy override behavior (request-aware)
# ============================================================

def test_effective_capabilities_policy_disables_extract(monkeypatch: pytest.MonkeyPatch):
    req = _req_with_settings(enable_extract=True, enable_generate=True)

    # base says extract=True (from models.yaml path)
    monkeypatch.setattr(deps, "_model_capabilities_from_models_yaml", lambda _mid: {"extract": True, "generate": True})
    # policy says extract=False
    monkeypatch.setattr(deps, "policy_capability_overrides", lambda _mid, request: {"extract": False})

    out = deps.effective_capabilities("m1", request=req)
    assert out["extract"] is False
    assert out["generate"] is True


def test_require_capability_policy_denies_extract(monkeypatch: pytest.MonkeyPatch):
    req = _req_with_settings(enable_extract=True, enable_generate=True)

    monkeypatch.setattr(deps, "_model_capabilities_from_models_yaml", lambda _mid: {"extract": True, "generate": True})
    monkeypatch.setattr(deps, "policy_capability_overrides", lambda _mid, request: {"extract": False})

    with pytest.raises(AppError) as ei:
        deps.require_capability("m1", "extract", request=req)

    e = ei.value
    assert e.code == "capability_not_supported"
    assert e.status_code == 400
    assert e.extra and e.extra.get("model_id") == "m1"


def test_deployment_gate_still_wins_over_policy(monkeypatch: pytest.MonkeyPatch):
    # deployment disables extract
    req = _req_with_settings(enable_extract=False, enable_generate=True)

    monkeypatch.setattr(deps, "_model_capabilities_from_models_yaml", lambda _mid: {"extract": True, "generate": True})
    # even if policy tries to allow it
    monkeypatch.setattr(deps, "policy_capability_overrides", lambda _mid, request: {"extract": True})

    with pytest.raises(AppError) as ei:
        deps.require_capability("m1", "extract", request=req)

    e = ei.value
    assert e.code == "capability_disabled"
    assert e.status_code == 501


# ============================================================
# models.yaml behavior (no request -> pure config path)
# ============================================================

def test_models_yaml_unspecified_means_allow_all(monkeypatch: pytest.MonkeyPatch):
    cfg = FakeModelsConfig(defaults={}, models=[FakeModelSpec(id="m1", capabilities=None)])
    _install_models_config(monkeypatch, cfg)

    # no defaults.capabilities and no model.capabilities => model_capabilities returns None
    assert deps.model_capabilities("m1", request=None) is None

    # require_capability must allow when model caps unspecified
    deps.require_capability("m1", "extract", request=None)   # should not raise
    deps.require_capability("m1", "generate", request=None)  # should not raise


def test_defaults_missing_key_defaults_true(monkeypatch: pytest.MonkeyPatch):
    # defaults specify only extract False; generate missing => defaults True
    cfg = FakeModelsConfig(
        defaults={"capabilities": {"extract": False}},
        models=[FakeModelSpec(id="m1", capabilities=None)],
    )
    _install_models_config(monkeypatch, cfg)

    caps = deps.effective_capabilities("m1", request=None)
    assert caps["extract"] is False
    assert caps["generate"] is True


def test_model_overrides_defaults(monkeypatch: pytest.MonkeyPatch):
    # defaults extract True, model extract False => effective False
    cfg = FakeModelsConfig(
        defaults={"capabilities": {"extract": True}},
        models=[FakeModelSpec(id="m1", capabilities={"extract": False})],
    )
    _install_models_config(monkeypatch, cfg)

    # model_capabilities should reflect override
    mc = deps.model_capabilities("m1", request=None)
    assert mc is not None
    assert mc["extract"] is False

    # require_capability should block
    with pytest.raises(AppError) as e:
        deps.require_capability("m1", "extract", request=None)

    assert e.value.code == "capability_not_supported"
    assert e.value.status_code == 400


def test_deployment_disables_capability_overrides_model(monkeypatch: pytest.MonkeyPatch):
    cfg = FakeModelsConfig(
        defaults={"capabilities": {"extract": True}},
        models=[FakeModelSpec(id="m1", capabilities={"extract": True})],
    )
    _install_models_config(monkeypatch, cfg)

    # deployment gate off => 501 regardless of model
    monkeypatch.setattr(
        deps,
        "deployment_capabilities",
        lambda request=None: {"generate": True, "extract": False},
        raising=True,
    )

    with pytest.raises(AppError) as e:
        deps.require_capability("m1", "extract", request=None)

    assert e.value.code == "capability_disabled"
    assert e.value.status_code == 501


def test_clear_models_config_cache_resets(monkeypatch: pytest.MonkeyPatch):
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return FakeModelsConfig(defaults={}, models=[])

    monkeypatch.setattr(deps, "load_models_config", loader, raising=True)

    _clear_cache_or_fail()

    # First call populates cache
    _ = deps._cached_models_config()
    assert calls["n"] == 1

    # Second call should hit cache
    _ = deps._cached_models_config()
    assert calls["n"] == 1

    # Clear cache
    _clear_cache_or_fail()

    # Next call reloads
    _ = deps._cached_models_config()
    assert calls["n"] == 2