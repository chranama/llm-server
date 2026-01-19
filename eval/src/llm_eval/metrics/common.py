# llm_eval/metrics/common.py
from __future__ import annotations

from typing import List, Optional


def quantile(vals: List[float], q: float) -> Optional[float]:
    """
    Compute linear-interpolated quantile.

    q in [0,1]:
      0.0 -> min
      0.5 -> median
      1.0 -> max

    Returns None if vals is empty.
    """
    if not vals:
        return None

    xs = sorted(vals)

    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]

    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)

    if lo == hi:
        return xs[lo]

    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac