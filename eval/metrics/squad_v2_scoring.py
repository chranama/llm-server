# src/llm_server/eval/metrics/squad_v2_scoring.py
"""
SQuAD v2 scoring for /v1/generate (text-only).

We implement:
- SQuAD-style Exact Match (EM) for answerable questions (max over gold answers)
- SQuAD-style token-overlap F1 for answerable questions (max over gold answers)
- Unanswerable accuracy (prediction == NO_ANSWER_TOKEN)

This is intentionally "API-contract aligned":
- /v1/generate returns a string
- We do NOT use logits or no_answer_probability thresholding
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

_WS_RE = re.compile(r"\s+")


# -----------------------------
# Normalization (SQuAD-style)
# -----------------------------

def _lower(text: str) -> str:
    return text.casefold()


def _remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def _remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def _white_space_fix(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def normalize_answer(text: str) -> str:
    """
    Standard SQuAD normalization:
      - casefold
      - remove punctuation
      - remove articles
      - collapse whitespace
    """
    if text is None:
        return ""
    text = str(text)
    return _white_space_fix(_remove_articles(_remove_punc(_lower(text))))


def is_no_answer(predicted: str, *, no_answer_token: str) -> bool:
    return normalize_answer(predicted) == normalize_answer(no_answer_token)


# -----------------------------
# EM / F1
# -----------------------------

def exact_match(predicted: str, gold: str) -> bool:
    return normalize_answer(predicted) == normalize_answer(gold)


def _tokenize(normed_text: str) -> List[str]:
    if not normed_text:
        return []
    return normed_text.split()


def f1_score(predicted: str, gold: str) -> float:
    """
    Token overlap F1 (SQuAD style).
    """
    pred_tokens = _tokenize(normalize_answer(predicted))
    gold_tokens = _tokenize(normalize_answer(gold))

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    # multiset intersection
    counts: Dict[str, int] = {}
    for t in gold_tokens:
        counts[t] = counts.get(t, 0) + 1

    num_same = 0
    for t in pred_tokens:
        if counts.get(t, 0) > 0:
            num_same += 1
            counts[t] -= 1

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def max_over_gold(predicted: str, gold_answers: List[str], fn) -> float:
    if not gold_answers:
        return 0.0
    return max(fn(predicted, g) for g in gold_answers)


# -----------------------------
# Per-example scoring
# -----------------------------

@dataclass(frozen=True)
class SquadV2ExampleScore:
    gold_has_answer: bool
    pred_no_answer: bool
    exact_match: Optional[bool]          # only for answerable
    f1: Optional[float]                 # only for answerable
    unanswerable_correct: Optional[bool]  # only for unanswerable


def score_squad_v2_example(
    *,
    predicted: str,
    answers: List[str],
    is_impossible: bool,
    no_answer_token: str,
) -> SquadV2ExampleScore:
    pred = (predicted or "").strip()
    pred_no = is_no_answer(pred, no_answer_token=no_answer_token)

    gold_has_answer = (not is_impossible) and len(answers) > 0

    if gold_has_answer:
        em_val = bool(max_over_gold(pred, answers, lambda p, g: 1.0 if exact_match(p, g) else 0.0) > 0.0)
        f1_val = float(max_over_gold(pred, answers, f1_score))
        return SquadV2ExampleScore(
            gold_has_answer=True,
            pred_no_answer=pred_no,
            exact_match=em_val,
            f1=f1_val,
            unanswerable_correct=None,
        )

    # Unanswerable: correct iff model abstains with NO_ANSWER_TOKEN
    return SquadV2ExampleScore(
        gold_has_answer=False,
        pred_no_answer=pred_no,
        exact_match=None,
        f1=None,
        unanswerable_correct=bool(pred_no),
    )


# -----------------------------
# Aggregation
# -----------------------------

def _pct(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return 100.0 * (num / den)


def summarize_squad_v2(scores: Iterable[SquadV2ExampleScore]) -> Dict[str, Any]:
    n_total = 0

    n_answerable = 0
    n_unanswerable = 0

    em_sum = 0.0
    f1_sum = 0.0

    unans_correct = 0.0

    for s in scores:
        n_total += 1
        if s.gold_has_answer:
            n_answerable += 1
            em_sum += 1.0 if s.exact_match else 0.0
            f1_sum += float(s.f1 or 0.0)
        else:
            n_unanswerable += 1
            unans_correct += 1.0 if s.unanswerable_correct else 0.0

    answerable_em = _pct(em_sum, n_answerable)
    answerable_f1 = _pct(f1_sum, n_answerable)
    unanswerable_acc = _pct(unans_correct, n_unanswerable)

    # Optional combined score: mean(answerable_em, unanswerable_acc) if both exist
    parts = [x for x in [answerable_em, unanswerable_acc] if x is not None]
    combined = (sum(parts) / len(parts)) if parts else None

    return {
        "n_total": n_total,
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "answerable_exact_match_rate": answerable_em,
        "answerable_f1_rate": answerable_f1,
        "unanswerable_accuracy": unanswerable_acc,
        "combined_score": combined,
    }