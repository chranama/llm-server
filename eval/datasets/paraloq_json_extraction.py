# src/llm_server/eval/datasets/paraloq_json_extraction.py

from datasets import load_dataset
from typing import Iterator, Dict, Any
from dataclasses import dataclass
import json

DATASET_NAME = "paraloq/json_data_extraction"

@dataclass
class ParaloqExample:
    id: str
    text: str
    schema: Dict[str, Any]
    expected: Dict[str, Any]

def iter_paraloq_json_extraction(split: str = "train", max_samples: int | None = None) -> Iterator[ParaloqExample]:
    ds = load_dataset(DATASET_NAME, split=split)

    count = 0
    for i, row in enumerate(ds):
        yield ParaloqExample(
            id=f"{split}:{i}",
            text=row["text"],
            schema=json.loads(row["schema"]) if isinstance(row["schema"], str) else row["schema"],
            expected=json.loads(row["item"]) if isinstance(row["item"], str) else row["item"],
        )

        count += 1
        if max_samples and count >= max_samples:
            break