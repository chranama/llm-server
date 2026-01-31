from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_policy.cli import main


def _write_summary(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_models_yaml(path: Path) -> None:
    # minimal models.yaml shape that patch_models_yaml expects
    path.write_text(
        "\n".join(
            [
                "defaults:",
                "  capabilities:",
                "    extract: false",
                "models:",
                "  - id: m1",
                "    capabilities:",
                "      extract: false",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_thresholds_root(root: Path) -> None:
    # Create a self-contained thresholds root that load_extract_thresholds can read.
    # CLI expects profile like "extract/default" => <thresholds_root>/extract/default.yaml
    p = root / "extract"
    p.mkdir(parents=True, exist_ok=True)
    (p / "default.yaml").write_text(
        "\n".join(
            [
                "min_n_total: 1",
                "min_schema_validity_rate: 0.95",
                "min_required_present_rate: 0.95",
                "min_doc_required_exact_match_rate: 0.80",
                "min_field_exact_match_rate: {}",
                "max_latency_p95_ms: 5000",
                "max_latency_p99_ms: 8000",
                "",
            ]
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# End-to-end-ish: exercises CLI + loader + thresholds loading
# (self-contained thresholds so it doesn't depend on repo files)
# ---------------------------------------------------------------------------

def test_cli_decide_extract_exit_code_allows(tmp_path: Path):
    thresholds_root = tmp_path / "thresholds"
    _write_thresholds_root(thresholds_root)

    run_dir = tmp_path / "run_allow"
    _write_summary(
        run_dir,
        {
            "task": "extraction_sroie",
            "run_id": "run_allow",
            "run_dir": str(run_dir),
            "n_total": 100,
            "n_ok": 95,
            "schema_validity_rate": 0.99,
            "required_present_rate": 0.99,
            "doc_required_exact_match_rate": 0.95,
            "latency_p95_ms": 100.0,
            "latency_p99_ms": 200.0,
            "field_exact_match_rate": {"total": 1.0},
        },
    )

    rc = main(
        [
            "decide-extract",
            "--run-dir",
            str(run_dir),
            "--threshold-profile",
            "extract/default",
            "--thresholds-root",
            str(thresholds_root),
            "--format",
            "text",
        ]
    )
    assert rc == 0


def test_cli_decide_extract_exit_code_denies(tmp_path: Path):
    thresholds_root = tmp_path / "thresholds"
    _write_thresholds_root(thresholds_root)

    run_dir = tmp_path / "run_deny"
    _write_summary(
        run_dir,
        {
            "task": "extraction_sroie",
            "run_id": "run_deny",
            "run_dir": str(run_dir),
            "n_total": 100,
            "n_ok": 5,
            "schema_validity_rate": 0.10,  # force fail
            "required_present_rate": 0.10,
            "doc_required_exact_match_rate": 0.10,
        },
    )

    rc = main(
        [
            "decide-extract",
            "--run-dir",
            str(run_dir),
            "--threshold-profile",
            "extract/default",
            "--thresholds-root",
            str(thresholds_root),
            "--format",
            "json",
        ]
    )
    assert rc == 2


def test_cli_decide_and_patch_dry_run_does_not_modify_models_yaml(tmp_path: Path):
    thresholds_root = tmp_path / "thresholds"
    _write_thresholds_root(thresholds_root)

    run_dir = tmp_path / "run_patch"
    _write_summary(
        run_dir,
        {
            "task": "extraction_sroie",
            "run_id": "run_patch",
            "run_dir": str(run_dir),
            "n_total": 100,
            "n_ok": 95,
            "schema_validity_rate": 0.99,
            "required_present_rate": 0.99,
            "doc_required_exact_match_rate": 0.95,
        },
    )

    models_yaml = tmp_path / "models.yaml"
    _write_models_yaml(models_yaml)
    before = models_yaml.read_text(encoding="utf-8")

    rc = main(
        [
            "decide-and-patch",
            "--run-dir",
            str(run_dir),
            "--models-yaml",
            str(models_yaml),
            "--model-id",
            "m1",
            "--threshold-profile",
            "extract/default",
            "--thresholds-root",
            str(thresholds_root),
            "--format",
            "text",
            "--dry-run",
        ]
    )
    assert rc == 0

    after = models_yaml.read_text(encoding="utf-8")
    assert after == before, "dry-run should not edit models.yaml"


# ---------------------------------------------------------------------------
# Your original monkeypatch smoke test merged in
# ---------------------------------------------------------------------------

def test_decide_and_patch_smoke(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path):
    class FakeDecision:
        enable_extract = True

        def ok(self) -> bool:
            return True

    class Res:
        changed = True
        warnings = []

    monkeypatch.setattr("llm_policy.cli.load_eval_artifact", lambda run_dir: object(), raising=True)
    monkeypatch.setattr(
        "llm_policy.cli.load_extract_thresholds",
        lambda cfg, profile=None: ("extract/default", {}),
        raising=True,
    )
    monkeypatch.setattr("llm_policy.cli.decide_extract_enablement", lambda *a, **k: FakeDecision(), raising=True)
    monkeypatch.setattr("llm_policy.cli.patch_models_yaml", lambda *a, **k: Res(), raising=True)
    monkeypatch.setattr("llm_policy.cli.render_decision_text", lambda d: "OK\n", raising=True)

    rc = main(
        [
            "decide-and-patch",
            "--run-dir",
            str(tmp_path),
            "--models-yaml",
            str(tmp_path / "models.yaml"),
            "--model-id",
            "m1",
            "--format",
            "text",
            "--dry-run",
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "OK" in out
    assert "patched" in out.lower()