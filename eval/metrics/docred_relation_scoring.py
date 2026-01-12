"""
DocRED relation extraction scoring.

We score predicted vs expected relations as SETS of triples:
  (head, tail, relation_id)

This yields micro precision/recall/F1.

Optionally, you can also score relation_text instead of relation_id,
but relation_id is typically more stable.

Predicted format expectation:
{
  "relations": [
    {"head": <int>, "tail": <int>, "relation_id": "Pxxx", ...},
    ...
  ]
}

If your /generate endpoint returns text, your eval harness should parse it into JSON
before calling these metrics (or you can add a tolerant parser here).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


Triple = Tuple[int, int, str]


def _as_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _as_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    s = str(x).strip()
    return s if s else None


def _extract_triples(obj: Any, *, use_relation_text: bool = False) -> Set[Triple]:
    """
    Extract a set of (head, tail, relation_id_or_text) triples from an object.
    """
    if not isinstance(obj, dict):
        return set()

    rels = obj.get("relations")
    if not isinstance(rels, list):
        return set()

    out: Set[Triple] = set()
    for r in rels:
        if not isinstance(r, dict):
            continue
        h = _as_int(r.get("head"))
        t = _as_int(r.get("tail"))
        key = "relation_text" if use_relation_text else "relation_id"
        rid = _as_str(r.get(key))

        if h is None or t is None or not rid:
            continue

        out.add((h, t, rid))
    return out


def micro_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def score_docred_example(
    expected: Dict[str, Any],
    predicted: Dict[str, Any],
    *,
    use_relation_text: bool = False,
) -> Dict[str, Any]:
    """
    Score one example.
    """
    gold = _extract_triples(expected, use_relation_text=use_relation_text)
    pred = _extract_triples(predicted, use_relation_text=use_relation_text)

    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gold": len(gold),
        "n_pred": len(pred),
        **micro_prf(tp, fp, fn),
    }


def aggregate_docred_scores(per_example: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate micro PRF across examples by summing TP/FP/FN.
    """
    tp = sum(int(x.get("tp", 0)) for x in per_example)
    fp = sum(int(x.get("fp", 0)) for x in per_example)
    fn = sum(int(x.get("fn", 0)) for x in per_example)

    agg = micro_prf(tp, fp, fn)
    agg.update(
        {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_examples": len(per_example),
        }
    )
    return agg


# ---------- Optional: tolerant parsing helpers (if you need them) ----------

def parse_predicted_maybe_json(pred: Any) -> Dict[str, Any]:
    """
    If `pred` is already a dict, return it.
    If it's a string, try JSON-decode it.
    Otherwise, return {}.

    (Use this if /generate returns a JSON string.)
    """
    if isinstance(pred, dict):
        return pred
    if isinstance(pred, str):
        s = pred.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}