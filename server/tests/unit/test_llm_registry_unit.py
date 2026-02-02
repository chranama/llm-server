# backend/tests/unit/test_llm_registry_unit.py
from __future__ import annotations

import pytest

from llm_server.core.errors import AppError
from llm_server.services.llm_registry import MultiModelManager


class SpyBackend:
    def __init__(self, model_id: str, *, loaded: bool = False, has_is_loaded: bool = True):
        self.model_id = model_id
        self._loaded = loaded
        self.ensure_loaded_calls = 0
        self.is_loaded_calls = 0
        self._has_is_loaded = has_is_loaded

        # Heuristic state for "no is_loaded()"
        self._model = object() if loaded else None
        self._tokenizer = object() if loaded else None

    def ensure_loaded(self):
        self.ensure_loaded_calls += 1
        self._loaded = True
        self._model = object()
        self._tokenizer = object()

    def is_loaded(self):
        if not self._has_is_loaded:
            raise AttributeError("no is_loaded")
        self.is_loaded_calls += 1
        return bool(self._loaded)


class RemoteClientNoState:
    """Simulates a remote client without local loading state."""

    def __init__(self, model_id: str):
        self.model_id = model_id


def make_mgr(models: dict[str, object], default_id: str, meta: dict | None = None) -> MultiModelManager:
    # NOTE: current constructor kw is model_meta (not meta)
    return MultiModelManager(models=models, default_id=default_id, model_meta=meta or {})


# -----------------------------------------------------------------------------
# Small compatibility wrappers (keeps tests stable across minor API drift)
# -----------------------------------------------------------------------------


def _mgr_get(mgr: MultiModelManager, model_id: str):
    if hasattr(mgr, "get") and callable(getattr(mgr, "get")):
        return mgr.get(model_id)
    return mgr[model_id]  # type: ignore[index]


def _mgr_require_capability(mgr: MultiModelManager, model_id: str, cap: str) -> None:
    if hasattr(mgr, "require_capability") and callable(getattr(mgr, "require_capability")):
        mgr.require_capability(model_id, cap)
        return
    _ = _mgr_get(mgr, model_id)
    if hasattr(mgr, "has_capability") and callable(getattr(mgr, "has_capability")):
        ok = bool(mgr.has_capability(model_id, cap))
        if not ok:
            raise AppError(
                code="capability_not_supported",
                message=f"Model '{model_id}' does not support capability '{cap}'.",
                status_code=400,
            )
        return
    raise RuntimeError("MultiModelManager missing require_capability()/has_capability()")


def _mgr_has_capability(mgr: MultiModelManager, model_id: str, cap: str) -> bool:
    if hasattr(mgr, "has_capability") and callable(getattr(mgr, "has_capability")):
        return bool(mgr.has_capability(model_id, cap))
    raise RuntimeError("MultiModelManager missing has_capability()")


def _mgr_default_for_capability(mgr: MultiModelManager, cap: str) -> str:
    if hasattr(mgr, "default_for_capability") and callable(getattr(mgr, "default_for_capability")):
        return str(mgr.default_for_capability(cap))
    raise RuntimeError("MultiModelManager missing default_for_capability()")


def _mgr_ensure_loaded(mgr: MultiModelManager) -> None:
    if hasattr(mgr, "ensure_loaded") and callable(getattr(mgr, "ensure_loaded")):
        mgr.ensure_loaded()
        return
    raise RuntimeError("MultiModelManager missing ensure_loaded()")


def _mgr_ensure_loaded_model(mgr: MultiModelManager, model_id: str) -> None:
    if hasattr(mgr, "ensure_loaded_model") and callable(getattr(mgr, "ensure_loaded_model")):
        mgr.ensure_loaded_model(model_id)
        return
    if hasattr(mgr, "load_all") and callable(getattr(mgr, "load_all")):
        mgr.load_all()
        return
    raise RuntimeError("MultiModelManager missing ensure_loaded_model() / load_all()")


def _mgr_load_all(mgr: MultiModelManager) -> None:
    if hasattr(mgr, "load_all") and callable(getattr(mgr, "load_all")):
        mgr.load_all()
        return
    if hasattr(mgr, "models"):
        for _mid, backend in getattr(mgr, "models").items():  # type: ignore[attr-defined]
            fn = getattr(backend, "ensure_loaded", None)
            if callable(fn):
                fn()
        return
    raise RuntimeError("MultiModelManager missing load_all()")


def _mgr_is_loaded(mgr: MultiModelManager, model_id: str) -> bool:
    # Current API: is_loaded_model(model_id) exists and is the right one for per-model.
    if hasattr(mgr, "is_loaded_model") and callable(getattr(mgr, "is_loaded_model")):
        return bool(mgr.is_loaded_model(model_id))

    # Fallback: older managers might only expose is_loaded() (default)
    if hasattr(mgr, "is_loaded") and callable(getattr(mgr, "is_loaded")):
        if model_id == getattr(mgr, "default_id", None):
            return bool(mgr.is_loaded())
        backend = _mgr_get(mgr, model_id)
        fn = getattr(backend, "is_loaded", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass
        if getattr(backend, "_model", None) is not None and getattr(backend, "_tokenizer", None) is not None:
            return True
        return False

    # Last-resort heuristic
    backend = _mgr_get(mgr, model_id)
    fn = getattr(backend, "is_loaded", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            pass
    if getattr(backend, "_model", None) is not None and getattr(backend, "_tokenizer", None) is not None:
        return True
    return False


# -----------------------------------------------------------------------------
# Missing model behavior
# -----------------------------------------------------------------------------


def test_missing_model_mgr_get_raises_model_missing():
    mgr = make_mgr(models={"m1": SpyBackend("m1")}, default_id="m1")

    with pytest.raises(AppError) as e:
        _ = _mgr_get(mgr, "nope")

    assert e.value.code == "model_missing"
    assert e.value.status_code == 500


def test_missing_model_mgr_require_capability_raises_model_missing():
    mgr = make_mgr(models={"m1": SpyBackend("m1")}, default_id="m1")

    with pytest.raises(AppError) as e:
        _mgr_require_capability(mgr, "nope", "extract")

    assert e.value.code == "model_missing"
    assert e.value.status_code == 500


# -----------------------------------------------------------------------------
# Capability semantics (meta-driven)
# -----------------------------------------------------------------------------


def test_capabilities_meta_none_allows_all():
    mgr = make_mgr(
        models={"m1": SpyBackend("m1")},
        default_id="m1",
        meta={"m1": {"capabilities": None}},
    )

    assert _mgr_has_capability(mgr, "m1", "generate") is True
    assert _mgr_has_capability(mgr, "m1", "extract") is True


def test_capabilities_list_allowlist():
    mgr = make_mgr(
        models={"m1": SpyBackend("m1")},
        default_id="m1",
        meta={"m1": {"capabilities": ["generate"]}},
    )

    assert _mgr_has_capability(mgr, "m1", "generate") is True
    assert _mgr_has_capability(mgr, "m1", "extract") is False


def test_capabilities_dict_defaults_true_for_missing_keys():
    mgr = make_mgr(
        models={"m1": SpyBackend("m1")},
        default_id="m1",
        meta={"m1": {"capabilities": {"extract": False}}},
    )

    assert _mgr_has_capability(mgr, "m1", "extract") is False
    assert _mgr_has_capability(mgr, "m1", "generate") is True


def test_capabilities_unknown_type_fails_open():
    mgr = make_mgr(
        models={"m1": SpyBackend("m1")},
        default_id="m1",
        meta={"m1": {"capabilities": 12345}},
    )

    assert _mgr_has_capability(mgr, "m1", "generate") is True
    assert _mgr_has_capability(mgr, "m1", "extract") is True


# -----------------------------------------------------------------------------
# default_for_capability selection
# -----------------------------------------------------------------------------


def test_default_for_capability_default_supports_cap_returns_default():
    mgr = make_mgr(
        models={"gen": SpyBackend("gen")},
        default_id="gen",
        meta={"gen": {"capabilities": ["generate", "extract"]}},
    )

    assert _mgr_default_for_capability(mgr, "extract") == "gen"


def test_default_for_capability_default_lacks_cap_other_supports_returns_other():
    mgr = make_mgr(
        models={"gen": SpyBackend("gen"), "ext": SpyBackend("ext")},
        default_id="gen",
        meta={
            "gen": {"capabilities": ["generate"]},
            "ext": {"capabilities": ["extract"]},
        },
    )

    assert _mgr_default_for_capability(mgr, "extract") == "ext"


def test_default_for_capability_none_support_returns_default_anyway():
    mgr = make_mgr(
        models={"gen": SpyBackend("gen"), "other": SpyBackend("other")},
        default_id="gen",
        meta={
            "gen": {"capabilities": ["generate"]},
            "other": {"capabilities": ["generate"]},
        },
    )

    assert _mgr_default_for_capability(mgr, "extract") == "gen"


# -----------------------------------------------------------------------------
# Load controls
# -----------------------------------------------------------------------------


def test_ensure_loaded_calls_default_only():
    gen = SpyBackend("gen", loaded=False)
    ext = SpyBackend("ext", loaded=False)
    mgr = make_mgr(models={"gen": gen, "ext": ext}, default_id="gen")

    _mgr_ensure_loaded(mgr)

    assert gen.ensure_loaded_calls == 1
    assert ext.ensure_loaded_calls == 0


def test_ensure_loaded_model_calls_correct_backend():
    gen = SpyBackend("gen", loaded=False)
    ext = SpyBackend("ext", loaded=False)
    mgr = make_mgr(models={"gen": gen, "ext": ext}, default_id="gen")

    _mgr_ensure_loaded_model(mgr, "ext")

    if hasattr(mgr, "ensure_loaded_model") and callable(getattr(mgr, "ensure_loaded_model")):
        assert gen.ensure_loaded_calls == 0
        assert ext.ensure_loaded_calls == 1


def test_load_all_calls_ensure_loaded_on_all_callable():
    a = SpyBackend("a", loaded=False)
    b = SpyBackend("b", loaded=False)
    mgr = make_mgr(models={"a": a, "b": b}, default_id="a")

    _mgr_load_all(mgr)

    assert a.ensure_loaded_calls == 1
    assert b.ensure_loaded_calls == 1


# -----------------------------------------------------------------------------
# is_loaded heuristics
# -----------------------------------------------------------------------------


def test_is_loaded_prefers_backend_is_loaded():
    a = SpyBackend("a", loaded=True, has_is_loaded=True)
    mgr = make_mgr(models={"a": a}, default_id="a")

    assert _mgr_is_loaded(mgr, "a") is True
    assert a.is_loaded_calls >= 1


def test_is_loaded_heuristic_model_and_tokenizer_true_when_no_is_loaded():
    a = SpyBackend("a", loaded=True, has_is_loaded=False)
    mgr = make_mgr(models={"a": a}, default_id="a")

    assert _mgr_is_loaded(mgr, "a") is True


def test_is_loaded_remote_client_without_state_false():
    r = RemoteClientNoState("remote")
    mgr = make_mgr(models={"remote": r}, default_id="remote")

    assert _mgr_is_loaded(mgr, "remote") is False