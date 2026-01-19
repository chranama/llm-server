# tests/fakes/fake_examples.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SquadV2Example:
    id: str
    context: str
    question: str
    title: Optional[str]
    answers: List[str]
    is_impossible: bool


@dataclass(frozen=True)
class DocREDExample:
    id: str
    text: str
    expected: Dict[str, Any]


@dataclass(frozen=True)
class ParaloqExample:
    id: str
    text: str
    schema: Dict[str, Any]
    expected: Dict[str, Any]


@dataclass(frozen=True)
class ReceiptExample:
    id: str
    schema_id: str
    text: str
    expected: Dict[str, Any]