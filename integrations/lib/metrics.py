# integrations/lib/metrics.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')
_METRIC_LINE_RE = re.compile(
    r"""
    ^
    (?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)
    (?:\{(?P<labels>[^}]*)\})?
    \s+
    (?P<value>[-+]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?)
    (?:\s+(?P<ts>\d+))?
    \s*$
    """,
    re.VERBOSE,
)


def _unescape_label_value(s: str) -> str:
    # Prometheus text exposition uses Go string escapes.
    # Common ones: \" \\ \n \t
    return (
        s.replace(r"\\", "\\")
        .replace(r"\"", '"')
        .replace(r"\n", "\n")
        .replace(r"\t", "\t")
    )


def _parse_labels(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    raw = raw.strip()
    if not raw:
        return out
    for m in _LABEL_RE.finditer(raw):
        k = m.group(1)
        v = _unescape_label_value(m.group(2))
        out[k] = v
    return out


def _labels_key(labels: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    # Stable, hashable ordering.
    return tuple(sorted(labels.items(), key=lambda kv: kv[0]))


@dataclass(frozen=True)
class Sample:
    name: str
    labels: Tuple[Tuple[str, str], ...]  # canonical
    value: float


@dataclass(frozen=True)
class MetricsSnapshot:
    """
    Parsed Prometheus text format.

    Data model:
      series[name][labels_key] = value
    """
    series: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]]

    def get(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[float]:
        m = self.series.get(name)
        if not m:
            return None
        lk = _labels_key(labels or {})
        return m.get(lk)

    def iter_samples(self, name: str) -> Iterable[Sample]:
        for lk, v in (self.series.get(name) or {}).items():
            yield Sample(name=name, labels=lk, value=v)

    def filter_samples(self, name: str, required: Dict[str, str]) -> List[Sample]:
        req = dict(required)
        out: List[Sample] = []
        for s in self.iter_samples(name):
            ld = dict(s.labels)
            ok = True
            for k, v in req.items():
                if ld.get(k) != v:
                    ok = False
                    break
            if ok:
                out.append(s)
        return out


def parse_prometheus_text(text: str) -> MetricsSnapshot:
    """
    Parses /metrics output in Prometheus text exposition format.

    Ignores:
      - # HELP lines
      - # TYPE lines
      - blank lines
      - unparsable lines (rare; but we fail hard if it looks like a sample line)
    """
    series: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue

        m = _METRIC_LINE_RE.match(line)
        if not m:
            # Be strict: if it's not a comment and not parseable, raise.
            raise ValueError(f"Unparsable metrics line: {raw_line!r}")

        name = m.group("name")
        labels_raw = m.group("labels") or ""
        value = float(m.group("value"))

        labels = _parse_labels(labels_raw)
        lk = _labels_key(labels)

        series.setdefault(name, {})[lk] = value

    return MetricsSnapshot(series=series)


def diff_metric(
    before: MetricsSnapshot,
    after: MetricsSnapshot,
    name: str,
    labels: Optional[Dict[str, str]] = None,
) -> float:
    """
    Returns after - before for a single series (name+labels).
    Missing series is treated as 0.0.
    """
    b = before.get(name, labels) or 0.0
    a = after.get(name, labels) or 0.0
    return a - b


def find_any_increment(
    before: MetricsSnapshot,
    after: MetricsSnapshot,
    name: str,
    *,
    required_labels: Optional[Dict[str, str]] = None,
    min_delta: float = 1.0,
) -> Tuple[bool, str]:
    """
    Checks if ANY series under metric `name` increments by >= min_delta.

    If required_labels is provided, only considers series that match those labels.

    Returns:
      (ok, message)
    """
    req = dict(required_labels or {})
    candidates_after = after.filter_samples(name, req) if req else list(after.iter_samples(name))

    if not candidates_after:
        return False, f"metric '{name}' has no series matching labels={req}"

    best_delta = None
    best_series = None

    for s_after in candidates_after:
        labels = dict(s_after.labels)
        d = diff_metric(before, after, name, labels)
        if best_delta is None or d > best_delta:
            best_delta = d
            best_series = labels

    assert best_delta is not None
    assert best_series is not None

    if best_delta >= min_delta:
        return True, f"metric '{name}' incremented by {best_delta} for labels={best_series}"
    return False, f"metric '{name}' did not increment by >= {min_delta}; best delta={best_delta} labels={best_series}"