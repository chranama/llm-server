# src/llm_server/eval/metrics/extraction_scoring.py
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# -------------------------
# Types
# -------------------------

FieldName = str


@dataclass(frozen=True)
class ExtractAttempt:
    """
    One model/service attempt for a single document.

    You can build these records in your extraction runner from:
      - dataset example (id, expected, schema_id)
      - API response on success OR error payload on failure
      - timing, caching, etc.
    """
    doc_id: str
    schema_id: str

    expected: Dict[str, Any]  # ground truth
    predicted: Optional[Dict[str, Any]]  # None if request failed (422/500/etc.)

    # Service signals (fill what you have)
    ok: bool  # True if response returned schema-valid JSON (HTTP 200)
    status_code: Optional[int] = None  # HTTP status for failures
    error_code: Optional[str] = None  # e.g. "invalid_json", "schema_validation_failed"
    error_stage: Optional[str] = None  # optional: parse/validate/repair_parse/repair_validate

    repair_attempted: bool = False  # from API response on success OR inferred from error path
    cached: bool = False  # from API response
    cache_layer: Optional[str] = None  # "redis" / "db" / None (if you track)

    latency_ms: Optional[float] = None  # end-to-end (client) or server if exposed


# -------------------------
# Normalization utilities
# -------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_CURRENCY_RE = re.compile(r"[€$£¥₹₩₽₺₫₴₦₲₱₡₵₸₮₭₤₳₠₢₣₥₧₯₰₶₷₺₻₼₾₿]")


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def norm_text_basic(s: str) -> str:
    """
    Conservative normalization:
    - strip
    - casefold
    - collapse whitespace
    """
    s = s.strip().casefold()
    s = _WS_RE.sub(" ", s)
    return s


def norm_text_strict(s: str) -> str:
    """
    Stronger normalization for noisy OCR:
    - basic normalization
    - remove most punctuation
    """
    s = norm_text_basic(s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def norm_company(s: str) -> str:
    """
    Company names suffer OCR punctuation + casing.
    Use strict normalization and remove common noise tokens.
    """
    s = norm_text_strict(s)

    # Optionally remove trailing legal suffixes for robustness
    # (keep conservative; don't over-normalize)
    suffixes = ["inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co"]
    toks = s.split()
    # Remove suffix if it is the last token
    if toks and toks[-1] in suffixes:
        toks = toks[:-1]
    return " ".join(toks).strip()


def norm_address(s: str) -> str:
    """
    Address is usually the noisiest; strict normalization helps.
    """
    return norm_text_strict(s)


def _parse_amount_to_float(s: str) -> Optional[float]:
    """
    Parse currency-ish amount strings to float.
    Handles:
      "$1,234.56", "1.234,56", "1234.56", "TOTAL 12.34"
    We keep it best-effort.
    """
    if not isinstance(s, str):
        s = str(s)

    t = s.strip()
    if not t:
        return None

    # Remove currency symbols and spaces
    t = _CURRENCY_RE.sub("", t)
    t = t.replace(" ", "")

    # Keep only digits, separators, minus
    t = re.sub(r"[^0-9,.\-]", "", t)
    if not t or t in {"-", ".", ",", "-.", "-,"}:
        return None

    # Heuristic: if both '.' and ',' exist, decide which is decimal by last separator
    if "." in t and "," in t:
        last_dot = t.rfind(".")
        last_com = t.rfind(",")
        if last_dot > last_com:
            # dot decimal, remove commas
            t = t.replace(",", "")
        else:
            # comma decimal, remove dots, swap comma -> dot
            t = t.replace(".", "")
            t = t.replace(",", ".")
    else:
        # Only one separator type: if comma used, treat as decimal if last group length != 3
        if "," in t and "." not in t:
            parts = t.split(",")
            if len(parts) == 2 and len(parts[1]) in (1, 2):
                t = parts[0].replace(".", "") + "." + parts[1]
            else:
                # commas are thousands separators
                t = t.replace(",", "")
        # If only dots exist, assume dots are decimal/thousands; common case: 1,234.56 already handled

    try:
        return float(t)
    except Exception:
        return None


def norm_total(s: str) -> str:
    """
    Normalize totals by parsing to float when possible; otherwise fall back to strict text.
    Represent floats in a canonical form.
    """
    f = _parse_amount_to_float(s)
    if f is None:
        return norm_text_strict(s)
    # canonical numeric format (avoid scientific notation)
    return f"{f:.2f}"


def norm_date_loose(s: str) -> str:
    """
    Light date normalization:
    - basic normalization
    - remove punctuation
    Keeps digits/letters but standardizes separators.
    This is *not* full date parsing; it's robust across formats.
    """
    t = norm_text_basic(s)
    # replace separators with single dash
    t = re.sub(r"[./\\_]", "-", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# -------------------------
# Field comparators
# -------------------------

def default_field_normalizer(field: FieldName):
    if field == "company":
        return norm_company
    if field == "address":
        return norm_address
    if field == "total":
        return norm_total
    if field == "date":
        return norm_date_loose
    # fallback
    return norm_text_strict


def field_equal(
    field: FieldName,
    pred_val: Any,
    exp_val: Any,
    *,
    allow_none_match: bool = False,
) -> bool:
    """
    Compare field values with normalization.

    allow_none_match:
      - If True: None == None counts as correct.
      - If False: if expected is non-empty, predicted must match it; None is incorrect.
    """
    if _is_empty(exp_val):
        # If ground truth missing/empty, you can choose to ignore or treat as match
        # For SROIE, expected fields are typically present; keep default as:
        return allow_none_match and _is_empty(pred_val)

    if _is_empty(pred_val):
        return False

    p = str(pred_val)
    e = str(exp_val)

    norm = default_field_normalizer(field)
    return norm(p) == norm(e)


# -------------------------
# Core per-document scoring
# -------------------------

@dataclass(frozen=True)
class DocFieldScore:
    doc_id: str
    schema_id: str
    ok: bool
    repair_attempted: bool
    cached: bool
    latency_ms: Optional[float]
    status_code: Optional[int]
    error_code: Optional[str]
    error_stage: Optional[str]
    # field -> correctness (True/False/None if not scorable)
    field_correct: Dict[str, Optional[bool]]
    # convenience
    required_all_correct: Optional[bool]
    required_present_non_null: Optional[bool]


def score_document(
    attempt: ExtractAttempt,
    *,
    fields: Sequence[FieldName],
    required_fields: Sequence[FieldName],
    ignore_if_expected_missing: bool = True,
    allow_none_match_when_expected_missing: bool = False,
) -> DocFieldScore:
    """
    Returns per-document correctness by field + rollups.

    ignore_if_expected_missing:
      If expected[field] is empty/missing, we return None for that field (not scorable).
    """
    field_correct: Dict[str, Optional[bool]] = {}

    if not attempt.ok or attempt.predicted is None:
        # no field scoring possible
        for f in fields:
            field_correct[f] = None
        return DocFieldScore(
            doc_id=attempt.doc_id,
            schema_id=attempt.schema_id,
            ok=False,
            repair_attempted=attempt.repair_attempted,
            cached=attempt.cached,
            latency_ms=attempt.latency_ms,
            status_code=attempt.status_code,
            error_code=attempt.error_code,
            error_stage=attempt.error_stage,
            field_correct=field_correct,
            required_all_correct=None,
            required_present_non_null=None,
        )

    pred = attempt.predicted
    exp = attempt.expected

    for f in fields:
        exp_val = exp.get(f)
        pred_val = pred.get(f)

        if ignore_if_expected_missing and _is_empty(exp_val):
            field_correct[f] = None
        else:
            field_correct[f] = field_equal(
                f,
                pred_val,
                exp_val,
                allow_none_match=allow_none_match_when_expected_missing,
            )

    # Required present rate: required keys exist and are non-null/non-empty
    required_present = True
    for rf in required_fields:
        if _is_empty(pred.get(rf)):
            required_present = False
            break

    # Required correctness: all required fields (that are scorable) are True
    req_scores: List[bool] = []
    for rf in required_fields:
        v = field_correct.get(rf)
        if v is None:
            # If expected missing and you ignored it, don't count it
            continue
        req_scores.append(bool(v))

    required_all_correct: Optional[bool]
    if req_scores:
        required_all_correct = all(req_scores)
    else:
        required_all_correct = None

    return DocFieldScore(
        doc_id=attempt.doc_id,
        schema_id=attempt.schema_id,
        ok=True,
        repair_attempted=attempt.repair_attempted,
        cached=attempt.cached,
        latency_ms=attempt.latency_ms,
        status_code=attempt.status_code,
        error_code=attempt.error_code,
        error_stage=attempt.error_stage,
        field_correct=field_correct,
        required_all_correct=required_all_correct,
        required_present_non_null=required_present,
    )


# -------------------------
# Aggregation helpers
# -------------------------

def _percent(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return 100.0 * (num / den)


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    xs = sorted(values)
    # nearest-rank with interpolation
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


# -------------------------
# Aggregate metrics
# -------------------------

@dataclass(frozen=True)
class ExtractionScoreSummary:
    n_total: int

    # validity
    n_ok: int
    schema_validity_rate: float

    # repair
    n_invalid_initial: int
    n_repair_attempted: int
    n_repair_success: int
    repair_success_rate: float  # among attempted repairs

    # cache
    n_cached: int
    cache_hit_rate: float

    # errors
    error_code_counts: Dict[str, int]
    status_code_counts: Dict[str, int]
    error_stage_counts: Dict[str, int]

    # field scoring
    field_exact_match_rate: Dict[str, float]  # per field
    doc_required_exact_match_rate: Optional[float]  # doc-level: all required correct (among scorable docs)
    required_present_rate: Optional[float]  # required fields non-null

    # latency
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
    latency_p99_ms: Optional[float]


def summarize_extraction(
    attempts: Sequence[ExtractAttempt],
    *,
    fields: Sequence[FieldName],
    required_fields: Sequence[FieldName],
    ignore_if_expected_missing: bool = True,
) -> ExtractionScoreSummary:
    """
    Compute the full summary for a run.
    """
    n_total = len(attempts)
    n_ok = sum(1 for a in attempts if a.ok)

    # Cache stats (only meaningful on ok responses)
    n_cached = sum(1 for a in attempts if a.ok and a.cached)

    # Latencies
    latencies = [float(a.latency_ms) for a in attempts if a.latency_ms is not None]
    p50 = _quantile(latencies, 0.50)
    p95 = _quantile(latencies, 0.95)
    p99 = _quantile(latencies, 0.99)

    # Error counts
    error_code_counts: Dict[str, int] = {}
    status_code_counts: Dict[str, int] = {}
    error_stage_counts: Dict[str, int] = {}

    for a in attempts:
        if a.ok:
            continue
        if a.error_code:
            error_code_counts[a.error_code] = error_code_counts.get(a.error_code, 0) + 1
        if a.status_code is not None:
            k = str(a.status_code)
            status_code_counts[k] = status_code_counts.get(k, 0) + 1
        if a.error_stage:
            error_stage_counts[a.error_stage] = error_stage_counts.get(a.error_stage, 0) + 1

    # Repair success rate:
    # In your current API, "repair_attempted" is returned on success.
    # For failures, you'll only know repair_attempted if you capture it in the runner.
    # We'll define:
    #   - attempted repair = a.repair_attempted is True
    #   - repair success = ok AND repair_attempted
    n_repair_attempted = sum(1 for a in attempts if a.repair_attempted)
    n_repair_success = sum(1 for a in attempts if a.ok and a.repair_attempted)

    # How many were invalid initially? You can only know this if you record it explicitly.
    # In your service, you *do* know internally, but you don't return it.
    # Best-effort proxy:
    #   invalid_initial = repair_success (because it had to be invalid first) + failures with stage parse/validate (if you capture)
    n_invalid_initial = n_repair_success + sum(
        1
        for a in attempts
        if (not a.ok) and (a.error_stage in {"parse", "validate", "repair_parse", "repair_validate"})
    )

    repair_success_rate = (n_repair_success / n_repair_attempted) if n_repair_attempted else 0.0

    # Field scoring
    doc_scores = [
        score_document(
            a,
            fields=fields,
            required_fields=required_fields,
            ignore_if_expected_missing=ignore_if_expected_missing,
        )
        for a in attempts
    ]

    # Per-field exact match: among docs where field_correct is not None
    field_exact_match_rate: Dict[str, float] = {}
    for f in fields:
        den = 0
        num = 0
        for ds in doc_scores:
            v = ds.field_correct.get(f)
            if v is None:
                continue
            den += 1
            if v:
                num += 1
        field_exact_match_rate[f] = _percent(num, den)

    # Document-level required exact match (among docs where required_all_correct is not None)
    doc_req_den = 0
    doc_req_num = 0
    for ds in doc_scores:
        if ds.required_all_correct is None:
            continue
        doc_req_den += 1
        if ds.required_all_correct:
            doc_req_num += 1
    doc_required_exact_match_rate = _percent(doc_req_num, doc_req_den) if doc_req_den else None

    # Required present rate (only among ok docs)
    pres_den = 0
    pres_num = 0
    for ds in doc_scores:
        if not ds.ok:
            continue
        if ds.required_present_non_null is None:
            continue
        pres_den += 1
        if ds.required_present_non_null:
            pres_num += 1
    required_present_rate = _percent(pres_num, pres_den) if pres_den else None

    return ExtractionScoreSummary(
        n_total=n_total,
        n_ok=n_ok,
        schema_validity_rate=_percent(n_ok, n_total),
        n_invalid_initial=n_invalid_initial,
        n_repair_attempted=n_repair_attempted,
        n_repair_success=n_repair_success,
        repair_success_rate=100.0 * repair_success_rate if n_repair_attempted else 0.0,
        n_cached=n_cached,
        cache_hit_rate=_percent(n_cached, n_ok) if n_ok else 0.0,
        error_code_counts=error_code_counts,
        status_code_counts=status_code_counts,
        error_stage_counts=error_stage_counts,
        field_exact_match_rate=field_exact_match_rate,
        doc_required_exact_match_rate=doc_required_exact_match_rate,
        required_present_rate=required_present_rate,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
    )


# -------------------------
# Convenience: pretty printing
# -------------------------

def format_summary(s: ExtractionScoreSummary) -> str:
    lines: List[str] = []
    lines.append(f"n_total={s.n_total}")
    lines.append(f"schema_validity_rate={s.schema_validity_rate:.2f}% ({s.n_ok}/{s.n_total})")
    lines.append(
        f"repair_success_rate={s.repair_success_rate:.2f}% (success={s.n_repair_success}, attempted={s.n_repair_attempted})"
    )
    lines.append(f"cache_hit_rate={s.cache_hit_rate:.2f}% (cached={s.n_cached}/{s.n_ok})")

    if s.required_present_rate is not None:
        lines.append(f"required_present_rate={s.required_present_rate:.2f}%")
    if s.doc_required_exact_match_rate is not None:
        lines.append(f"doc_required_exact_match_rate={s.doc_required_exact_match_rate:.2f}%")

    lines.append("field_exact_match_rate:")
    for k, v in s.field_exact_match_rate.items():
        lines.append(f"  - {k}: {v:.2f}%")

    if s.latency_p50_ms is not None:
        lines.append(f"latency_p50_ms={s.latency_p50_ms:.1f}")
    if s.latency_p95_ms is not None:
        lines.append(f"latency_p95_ms={s.latency_p95_ms:.1f}")
    if s.latency_p99_ms is not None:
        lines.append(f"latency_p99_ms={s.latency_p99_ms:.1f}")

    if s.status_code_counts:
        lines.append(f"status_code_counts={s.status_code_counts}")
    if s.error_code_counts:
        lines.append(f"error_code_counts={s.error_code_counts}")
    if s.error_stage_counts:
        lines.append(f"error_stage_counts={s.error_stage_counts}")

    return "\n".join(lines)