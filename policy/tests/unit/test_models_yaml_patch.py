from __future__ import annotations

from pathlib import Path

import yaml

from llm_policy.io.models_yaml import patch_models_yaml


def _write(p: Path, obj) -> None:
    p.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def _read(p: Path):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def test_patch_models_yaml_sets_capability_and_writes(tmp_path: Path):
    path = tmp_path / "models.yaml"
    _write(
        path,
        {
            "defaults": {"capabilities": {"generate": True}},
            "models": [{"id": "m1", "capabilities": {"generate": True}}],
        },
    )

    res = patch_models_yaml(path=str(path), model_id="m1", capability="extract", enable=True, write=True)
    assert res.changed is True

    obj = _read(path)
    assert obj["models"][0]["capabilities"]["extract"] is True
    # defaults.capabilities.extract should be present and conservative False
    assert obj["defaults"]["capabilities"]["extract"] is False


def test_patch_models_yaml_is_idempotent(tmp_path: Path):
    path = tmp_path / "models.yaml"
    _write(
        path,
        {
            "defaults": {"capabilities": {"extract": False}},
            "models": [{"id": "m1", "capabilities": {"extract": True}}],
        },
    )

    res1 = patch_models_yaml(path=str(path), model_id="m1", capability="extract", enable=True, write=True)
    assert res1.changed is False  # already true, no change

    before = path.read_text(encoding="utf-8")
    res2 = patch_models_yaml(path=str(path), model_id="m1", capability="extract", enable=True, write=True)
    after = path.read_text(encoding="utf-8")
    assert res2.changed is False
    assert after == before  # file unchanged


def test_patch_models_yaml_model_not_found(tmp_path: Path):
    path = tmp_path / "models.yaml"
    _write(
        path,
        {
            "defaults": {"capabilities": {"extract": False}},
            "models": [{"id": "m1", "capabilities": {"extract": False}}],
        },
    )

    res = patch_models_yaml(path=str(path), model_id="does-not-exist", capability="extract", enable=True, write=True)
    assert res.changed is False
    assert any("model id not found" in w for w in res.warnings)

    # file should remain parseable and unchanged (content check is optional)
    obj = _read(path)
    assert obj["models"][0]["id"] == "m1"