# backend/tests/unit/test_llm_build_unit.py
from __future__ import annotations

import types

import pytest

from llm_server.core.errors import AppError
from llm_server.services import llm as llm_mod


class FakeModelsConfig:
    """
    Matches what llm_mod.build_llm_from_settings expects today:
      - primary_id
      - model_ids (ordered)
      - models (list of specs)
      - defaults (kept for completeness)
    """

    def __init__(self, *, primary_id: str, defaults: dict, models: list):
        self.primary_id = primary_id
        self.defaults = defaults
        self.models = models
        self.model_ids = [m.id for m in models]


class FakeModelSpec:
    def __init__(
        self,
        id: str,
        kind: str = "local",
        llm_service_url: str | None = None,
        trust_remote_code: bool | None = None,
        capabilities=None,
        load_mode: str = "lazy",
        backend: str | None = None,
    ):
        self.id = id

        # New builder expects spec.backend (string like "local" / "remote")
        # Preserve your tests' "kind" param but map it.
        self.kind = kind
        self.backend = backend if backend is not None else ("remote" if kind == "remote" else "local")

        self.llm_service_url = llm_service_url
        self.trust_remote_code = trust_remote_code
        self.capabilities = capabilities
        self.load_mode = load_mode


class FakeLocalManager:
    def __init__(self, model_id: str, trust_remote_code: bool | None = None):
        self.model_id = model_id
        self.trust_remote_code = trust_remote_code

    @classmethod
    def from_settings(cls, *args, **kwargs):
        """
        Accept both builder calling conventions:

          - from_settings(spec, settings)   (older style)
          - from_settings(settings)         (newer style)

        We'll derive model_id + trust_remote_code where possible.
        """
        spec = None
        settings = None

        if len(args) == 1:
            # from_settings(settings)
            settings = args[0]
        elif len(args) >= 2:
            # from_settings(spec, settings)
            spec = args[0]
            settings = args[1]

        # Prefer spec.id if present; else fall back to settings.model_id
        model_id = getattr(spec, "id", None) or getattr(settings, "model_id", None) or "unknown"
        trust_remote_code = getattr(spec, "trust_remote_code", None)

        return cls(model_id=model_id, trust_remote_code=trust_remote_code)


class FakeHttpClient:
    def __init__(self, model_id: str, base_url: str):
        self.model_id = model_id
        self.base_url = base_url


def _settings(**kw):
    base = dict(
        env="test",
        model_id="primary",
        model_load_mode="lazy",
        enable_multi_models=True,
        llm_service_url=None,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _mm_models_set(mm) -> set[str]:
    if hasattr(mm, "list_models") and callable(mm.list_models):
        return set(mm.list_models())
    if hasattr(mm, "models"):
        return set(getattr(mm, "models").keys())
    return set()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    # Keep deterministic: each test sets what it needs explicitly.
    for k in ["ENABLE_MULTI_MODELS", "MODEL_LOAD_MODE", "LLM_SERVICE_URL", "ENV"]:
        monkeypatch.delenv(k, raising=False)


def _patch_builder_deps(monkeypatch, cfg: FakeModelsConfig, settings_obj):
    monkeypatch.setattr(llm_mod, "load_models_config", lambda: cfg, raising=True)
    monkeypatch.setattr(llm_mod, "ModelManager", FakeLocalManager, raising=True)
    monkeypatch.setattr(llm_mod, "HttpLLMClient", FakeHttpClient, raising=True)
    monkeypatch.setattr(llm_mod, "get_settings", lambda: settings_obj, raising=True)


def test_enable_multi_models_0_returns_single_backend(monkeypatch):
    cfg = FakeModelsConfig(
        primary_id="primary",
        defaults={},
        models=[FakeModelSpec(id="primary", kind="local")],
    )
    _patch_builder_deps(monkeypatch, cfg, _settings(enable_multi_models=False, model_id="primary"))

    # build_llm_from_settings currently gates multi-model via env
    monkeypatch.setenv("ENABLE_MULTI_MODELS", "0")

    llm = llm_mod.build_llm_from_settings()

    assert isinstance(llm, FakeLocalManager)
    assert llm.model_id == "primary"


def test_enable_multi_models_1_returns_multimodelmanager(monkeypatch):
    cfg = FakeModelsConfig(
        primary_id="primary",
        defaults={},
        models=[
            FakeModelSpec(id="primary", kind="local"),
            FakeModelSpec(id="other", kind="local"),
        ],
    )
    _patch_builder_deps(monkeypatch, cfg, _settings(enable_multi_models=True, model_id="primary"))

    monkeypatch.setenv("ENABLE_MULTI_MODELS", "1")

    llm = llm_mod.build_llm_from_settings()

    from llm_server.services.llm_registry import MultiModelManager

    assert isinstance(llm, MultiModelManager)
    assert llm.default_id == "primary"
    assert _mm_models_set(llm) == {"primary", "other"}

    assert llm["primary"].model_id == "primary"
    assert llm["other"].model_id == "other"


def test_load_mode_off_excludes_models(monkeypatch):
    cfg = FakeModelsConfig(
        primary_id="primary",
        defaults={},
        models=[
            FakeModelSpec(id="primary", kind="local", load_mode="lazy"),
            FakeModelSpec(id="other", kind="local", load_mode="lazy"),
        ],
    )
    _patch_builder_deps(monkeypatch, cfg, _settings(enable_multi_models=True, model_load_mode="off"))

    monkeypatch.setenv("ENABLE_MULTI_MODELS", "1")
    monkeypatch.setenv("MODEL_LOAD_MODE", "off")

    llm = llm_mod.build_llm_from_settings()

    from llm_server.services.llm_registry import MultiModelManager

    assert isinstance(llm, MultiModelManager)
    assert _mm_models_set(llm) == set()


def test_remote_models_require_llm_service_url(monkeypatch):
    cfg = FakeModelsConfig(
        primary_id="primary",
        defaults={},
        models=[FakeModelSpec(id="primary", kind="remote", llm_service_url=None)],
    )
    _patch_builder_deps(monkeypatch, cfg, _settings(enable_multi_models=True, llm_service_url=None))

    monkeypatch.setenv("ENABLE_MULTI_MODELS", "1")

    with pytest.raises(AppError) as e:
        llm_mod.build_llm_from_settings()

    assert e.value.code == "remote_models_require_llm_service_url"


def test_meta_propagation_local_and_remote_backend_types_and_caps_order(monkeypatch):
    cfg = FakeModelsConfig(
        primary_id="primary",
        defaults={},
        models=[
            FakeModelSpec(
                id="primary",
                kind="local",
                capabilities=["extract", "generate"],
                trust_remote_code=True,
            ),
            FakeModelSpec(
                id="r1",
                kind="remote",
                llm_service_url="http://x",
                capabilities=["generate"],
            ),
        ],
    )
    _patch_builder_deps(monkeypatch, cfg, _settings(enable_multi_models=True, llm_service_url="http://base"))

    monkeypatch.setenv("ENABLE_MULTI_MODELS", "1")
    monkeypatch.setenv("LLM_SERVICE_URL", "http://base")

    llm = llm_mod.build_llm_from_settings()

    meta = getattr(llm, "_meta", {})
    assert meta["primary"]["backend"] == "local_hf"
    assert meta["r1"]["backend"] == "http_remote"

    assert meta["primary"]["capabilities"] == ["extract", "generate"]
    assert meta["r1"]["capabilities"] == ["generate"]

    primary_backend = llm["primary"]
    assert getattr(primary_backend, "trust_remote_code", None) is True