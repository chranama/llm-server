# src/llm_server/eval/datasets/squad_v2.py
"""
SQuAD v2 adapter (text-only)

Hugging Face dataset:
  - https://huggingface.co/datasets/squad_v2
  - (You referenced GEM/squad_v2, but the canonical HF dataset is squad_v2)

Contract:
  Input:
    - context: Wikipedia paragraph (string)
    - question: question about the context (string)
  Output:
    - answers: list[str] of acceptable ground-truth answers (may be empty)
    - is_impossible: bool, True when the question is unanswerable from context

This adapter yields SlotFillingQAExample objects for benchmarking /v1/generate
in a "slot-filling QA" framing (answer span or NO_ANSWER).

Notes:
- SQuAD v2 answers are extractive spans; for unanswerable questions, answers.text == [].
- We do not touch images, layout, OCR, etc. This is pure text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional


@dataclass(frozen=True)
class SlotFillingQAExample:
    id: str
    title: Optional[str]
    context: str
    question: str
    answers: List[str]
    is_impossible: bool


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _normalize_answer_list(raw_answers: Any) -> List[str]:
    """
    raw_answers is typically a dict like:
      {"text": [...], "answer_start": [...]}
    """
    if not isinstance(raw_answers, dict):
        return []

    texts = raw_answers.get("text")
    if not isinstance(texts, list):
        return []

    out: List[str] = []
    for t in texts:
        s = _safe_str(t).strip()
        if s:
            out.append(s)

    # de-duplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for a in out:
        if a not in seen:
            seen.add(a)
            deduped.append(a)
    return deduped


def iter_squad_v2(
    *,
    split: str = "validation",
    max_samples: Optional[int] = None,
    dataset_name: str = "squad_v2",
) -> Iterator[SlotFillingQAExample]:
    """
    Yields SlotFillingQAExample objects.

    Example:
        for ex in iter_squad_v2(split="validation", max_samples=3):
            print(ex.id, ex.is_impossible, ex.answers[:2])
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "The `datasets` package is required to load SQuAD v2.\n"
            "Install it (for eval only) with something like:\n"
            "  uv pip install datasets\n"
            "or add it to an eval extra in pyproject.toml."
        ) from e

    ds = load_dataset(dataset_name, split=split)

    n = 0
    for row in ds:
        if max_samples is not None and n >= max_samples:
            break

        ex_id = _safe_str(row.get("id")).strip() or f"{split}:{n}"
        title = _safe_str(row.get("title")).strip() or None
        context = _safe_str(row.get("context")).strip()
        question = _safe_str(row.get("question")).strip()

        answers = _normalize_answer_list(row.get("answers"))
        is_impossible = bool(row.get("is_impossible")) if "is_impossible" in row else (len(answers) == 0)

        yield SlotFillingQAExample(
            id=ex_id,
            title=title,
            context=context,
            question=question,
            answers=answers,
            is_impossible=is_impossible,
        )
        n += 1


def load_squad_v2(
    *,
    split: str = "validation",
    max_samples: Optional[int] = None,
    dataset_name: str = "squad_v2",
) -> List[SlotFillingQAExample]:
    return list(iter_squad_v2(split=split, max_samples=max_samples, dataset_name=dataset_name))


if __name__ == "__main__":
    # quick smoke test
    for ex in iter_squad_v2(split="validation", max_samples=3):
        print("=" * 80)
        print("ID:", ex.id)
        print("TITLE:", ex.title)
        print("IMPOSSIBLE:", ex.is_impossible)
        print("Q:", ex.question)
        print("A (first 3):", ex.answers[:3])
        print("CONTEXT_HEAD:", repr(ex.context[:200]))