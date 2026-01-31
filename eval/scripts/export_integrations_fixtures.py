#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Reuse your existing dataset iterators
from llm_eval.datasets.voxel51_scanned_receipts import iter_voxel51_scanned_receipts
from llm_eval.datasets.paraloq_json_extraction import iter_paraloq_json_extraction


def require_packages(pkgs: list[str]) -> None:
    missing = []
    for p in pkgs:
        try:
            __import__(p)
        except Exception:
            missing.append(p)

    if missing:
        raise SystemExit(
            "❌ Missing required packages:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nInstall with:\n"
            + "  uv sync --extra datasets\n"
        )

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s if s.endswith("\n") else s + "\n", encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def safe_slug(x: str) -> str:
    # small, filesystem-safe slug
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in x)[:120]


def export_voxel51_raw(
    out_root: Path,
    max_samples: int,
    schema_id: str,
) -> None:
    ds_name = "Voxel51/scanned_receipts"
    base = out_root / "voxel51_scanned_receipts"
    manifest = base / "manifest.jsonl"
    samples_dir = base / "samples"

    n = 0
    for ex in iter_voxel51_scanned_receipts(max_samples=max_samples, schema_id=schema_id):
        n += 1
        sid = safe_slug(ex.id) or f"sample_{n:04d}"
        prefix = f"{n:04d}"

        txt_path = samples_dir / f"{prefix}.txt"
        exp_path = samples_dir / f"{prefix}.expected.json"

        write_text(txt_path, ex.text)
        write_json(exp_path, ex.expected)

        append_jsonl(
            manifest,
            {
                "exported_at": utc_now_iso(),
                "dataset": ds_name,
                "schema_id": ex.schema_id,
                "upstream_id": ex.id,
                "files": {
                    "text": str(txt_path.relative_to(out_root)),
                    "expected": str(exp_path.relative_to(out_root)),
                },
            },
        )
    print(f"[raw] exported voxel51: {n} samples -> {base}")


def export_paraloq_raw(
    out_root: Path,
    max_samples: int,
    split: str,
) -> None:
    ds_name = "paraloq/json_data_extraction"
    base = out_root / "paraloq_json_extraction"
    manifest = base / "manifest.jsonl"
    samples_dir = base / "samples"

    n = 0
    for ex in iter_paraloq_json_extraction(split=split, max_samples=max_samples):
        n += 1
        prefix = f"{n:04d}"

        text_path = samples_dir / f"{prefix}.text.txt"
        schema_path = samples_dir / f"{prefix}.schema.json"
        exp_path = samples_dir / f"{prefix}.expected.json"

        write_text(text_path, ex.text)
        write_json(schema_path, ex.schema)
        write_json(exp_path, ex.expected)

        append_jsonl(
            manifest,
            {
                "exported_at": utc_now_iso(),
                "dataset": ds_name,
                "split": split,
                "upstream_id": ex.id,
                "files": {
                    "text": str(text_path.relative_to(out_root)),
                    "schema": str(schema_path.relative_to(out_root)),
                    "expected": str(exp_path.relative_to(out_root)),
                },
            },
        )
    print(f"[raw] exported paraloq: {n} samples -> {base}")


def promote_to_golden(
    integrations_dir: Path,
    kind: str,  # invoice|ticket|sroie
    name: str,  # invoice_001
    text_path: Path,
    expected_path: Path,
    schema_id: str,
) -> None:
    """
    Promote a raw sample into integrations/data/golden/<kind>/<name>.*
    Creates a contract stub you can refine.
    """
    golden_dir = integrations_dir / "data" / "golden" / kind
    golden_dir.mkdir(parents=True, exist_ok=True)

    out_txt = golden_dir / f"{name}.txt"
    out_expected = golden_dir / f"{name}.expected.json"
    out_contract = golden_dir / f"{name}.contract.yaml"

    # copy content
    write_text(out_txt, text_path.read_text(encoding="utf-8"))
    write_json(out_expected, json.loads(expected_path.read_text(encoding="utf-8")))

    # contract stub: stable “shape” checks, not exact text
    contract = f"""# Contract for {name}
schema_id: {schema_id}
assertions:
  # Keys that must exist (even if null/empty)
  required_keys: []
  # Keys that must be non-empty strings if present
  non_empty_if_present: []
  # Keys that must match a regex (if present)
  regex_if_present: {{}}
notes: "Fill these in once you decide the invariants that matter."
"""
    out_contract.write_text(contract, encoding="utf-8")

    print(f"[golden] promoted -> {golden_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description="Export HF datasets into integrations fixtures (raw + optional golden).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # export-raw
    r = sub.add_parser("export-raw", help="Export raw samples into integrations/data/raw/...")
    r.add_argument("--integrations-dir", default="integrations", help="Path to integrations/ root")
    r.add_argument("--max-voxel51", type=int, default=10)
    r.add_argument("--voxel51-schema-id", default="sroie_receipt_v1")
    r.add_argument("--max-paraloq", type=int, default=10)
    r.add_argument("--paraloq-split", default="train")

    # promote-golden
    g = sub.add_parser("promote-golden", help="Promote a raw sample to integrations/data/golden/")
    g.add_argument("--integrations-dir", default="integrations", help="Path to integrations/ root")
    g.add_argument("--kind", choices=["invoice", "ticket", "sroie"], required=True)
    g.add_argument("--name", required=True, help="e.g. invoice_001")
    g.add_argument("--text", required=True, help="Path to a .txt")
    g.add_argument("--expected", required=True, help="Path to a .expected.json")
    g.add_argument("--schema-id", required=True, help="Schema id your /v1/extract expects (e.g. sroie_receipt_v1)")

    args = p.parse_args()

    integrations_dir = Path(args.integrations_dir).resolve()
    data_raw_root = integrations_dir / "data" / "raw"

    if args.cmd == "export-raw":
        require_packages(["fiftyone"])
        export_voxel51_raw(
            out_root=data_raw_root,
            max_samples=int(args.max_voxel51),
            schema_id=str(args.voxel51_schema_id),
        )
        export_paraloq_raw(
            out_root=data_raw_root,
            max_samples=int(args.max_paraloq),
            split=str(args.paraloq_split),
        )
        return 0

    if args.cmd == "promote-golden":
        promote_to_golden(
            integrations_dir=integrations_dir.resolve(),
            kind=str(args.kind),
            name=str(args.name),
            text_path=Path(args.text).resolve(),
            expected_path=Path(args.expected).resolve(),
            schema_id=str(args.schema_id),
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())