# src/llm_server/eval/datasets/voxel51_scanned_receipts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_SCHEMA_ID = "sroie_receipt_v1"


@dataclass(frozen=True)
class ExtractionExample:
    id: str
    schema_id: str
    text: str
    expected: Dict[str, Any]


def _safe_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    s = str(x).strip()
    return s if s else None


def _bbox_sort_key_det(det) -> tuple[float, float]:
    """
    FiftyOne Detection has bounding_box = [x, y, w, h] in relative coords.
    Sort top-to-bottom then left-to-right.
    """
    bb = getattr(det, "bounding_box", None)
    if isinstance(bb, (list, tuple)) and len(bb) >= 2:
        return (float(bb[1]), float(bb[0]))
    return (1e9, 1e9)


def build_ocr_text_from_sample(sample) -> str:
    """
    Build OCR text from FiftyOne sample fields:
      - prefer text_detections.detections[].label
      - fallback to text_polygons.polylines[].label
    """
    lines: List[str] = []

    td = getattr(sample, "text_detections", None)
    if td is not None and hasattr(td, "detections") and td.detections:
        dets = sorted(td.detections, key=_bbox_sort_key_det)
        for d in dets:
            t = _safe_str(getattr(d, "label", None))
            if t:
                lines.append(t)

    if not lines:
        tp = getattr(sample, "text_polygons", None)
        if tp is not None and hasattr(tp, "polylines") and tp.polylines:
            # Polylines don't have bounding_box; use min y/x over points
            def poly_key(p):
                pts = getattr(p, "points", None)
                if not pts:
                    return (1e9, 1e9)
                xs: List[float] = []
                ys: List[float] = []
                for ring in pts:
                    for xy in ring:
                        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                            xs.append(float(xy[0]))
                            ys.append(float(xy[1]))
                return (min(ys) if ys else 1e9, min(xs) if xs else 1e9)

            polys = sorted(tp.polylines, key=poly_key)
            for p in polys:
                t = _safe_str(getattr(p, "label", None))
                if t:
                    lines.append(t)

    return "\n".join(lines).strip()


def build_expected_from_sample(sample) -> Dict[str, Any]:
    """
    Ground truth lives on the sample directly (per your output):
      company/date/address/total
    """
    return {
        "company": _safe_str(getattr(sample, "company", None)),
        "address": _safe_str(getattr(sample, "address", None)),
        "date": _safe_str(getattr(sample, "date", None)),
        "total": _safe_str(getattr(sample, "total", None)),
    }


def iter_voxel51_scanned_receipts(
    split: str = "train",  # kept for API compatibility; hub dataset isn't split in the same way
    schema_id: str = DEFAULT_SCHEMA_ID,
    max_samples: Optional[int] = None,
) -> Iterator[ExtractionExample]:
    """
    Uses FiftyOne hub dataset, which actually contains OCR + labels.
    """
    from fiftyone.utils.huggingface import load_from_hub  # local import keeps optional dep

    ds = load_from_hub("Voxel51/scanned_receipts")

    n = 0
    for i, sample in enumerate(ds):
        if max_samples is not None and n >= max_samples:
            break

        text = build_ocr_text_from_sample(sample)
        expected = build_expected_from_sample(sample)

        # stable ID: filepath (what you saw in sample print)
        sid = _safe_str(getattr(sample, "filepath", None)) or f"{split}:{i}"

        yield ExtractionExample(
            id=sid,
            schema_id=schema_id,
            text=text,
            expected=expected,
        )
        n += 1