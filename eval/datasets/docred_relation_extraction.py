"""
DocRED adapter (document-level relation extraction)

HuggingFace dataset: https://huggingface.co/datasets/thunlp/docred

Important:
- The dataset viewer is disabled because it uses a loading script. You must load it with
  trust_remote_code=True.

This adapter builds:
- `text`: plain text document reconstructed from tokenized sentences
- `expected`: a structured target with entities + labeled relations (when available)
- `schema_id`: stable id you can map to a JSON schema (e.g. docred_relations_v1)

We keep this TEXT-ONLY. No OCR, no images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datasets import load_dataset


DEFAULT_SCHEMA_ID = "docred_relations_v1"


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


def _join_tokens(tokens: List[str]) -> str:
    # Simple whitespace join. (You can get fancier with punctuation rules later.)
    return " ".join(t for t in tokens if isinstance(t, str)).strip()


def build_doc_text(sample: Dict[str, Any]) -> str:
    """
    Turn `sents` (list[list[str]]) into a multi-line document.
    """
    sents = sample.get("sents") or []
    lines: List[str] = []
    if isinstance(sents, list):
        for sent in sents:
            if isinstance(sent, list):
                line = _join_tokens(sent)
                if line:
                    lines.append(line)
    return "\n".join(lines).strip()


def build_entities(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert `vertexSet` into a compact entity list.

    vertexSet: list[entity], where entity is list[mention]
      mention has: name, sent_id, pos, type

    We choose a canonical entity name as the most frequent mention surface form,
    falling back to the first mention name.
    """
    vs = sample.get("vertexSet") or []
    entities: List[Dict[str, Any]] = []

    if not isinstance(vs, list):
        return entities

    for ent_idx, ent in enumerate(vs):
        if not isinstance(ent, list) or not ent:
            entities.append({"id": ent_idx, "name": None, "mentions": []})
            continue

        mention_names: List[str] = []
        mentions_out: List[Dict[str, Any]] = []

        for m in ent:
            if not isinstance(m, dict):
                continue
            name = _safe_str(m.get("name"))
            if name:
                mention_names.append(name)

            mentions_out.append(
                {
                    "name": name,
                    "sent_id": m.get("sent_id"),
                    "pos": m.get("pos"),  # keep as-is (usually list[int])
                    "type": _safe_str(m.get("type")),
                }
            )

        # pick canonical name
        canon = None
        if mention_names:
            # most frequent surface form
            freq: Dict[str, int] = {}
            for n in mention_names:
                freq[n] = freq.get(n, 0) + 1
            canon = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        entities.append(
            {
                "id": ent_idx,
                "name": canon,
                "mentions": mentions_out,
            }
        )

    return entities


def build_relations(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert `labels` into a list of relation instances.

    labels fields typically:
      head: int
      tail: int
      relation_id: str
      relation_text: str
      evidence: list[int]
    """
    labels = sample.get("labels")
    if labels is None:
        return []

    rels: List[Dict[str, Any]] = []

    # In many HF conversions, `labels` might be a list[dict] rather than dict-of-lists.
    if isinstance(labels, list):
        for item in labels:
            if not isinstance(item, dict):
                continue
            rels.append(
                {
                    "head": item.get("head"),
                    "tail": item.get("tail"),
                    "relation_id": _safe_str(item.get("relation_id")),
                    "relation_text": _safe_str(item.get("relation_text")),
                    "evidence": item.get("evidence") if isinstance(item.get("evidence"), list) else [],
                }
            )
        return rels

    # Dataset card example shows dict-of-lists
    if isinstance(labels, dict):
        heads = labels.get("head") or []
        tails = labels.get("tail") or []
        rids = labels.get("relation_id") or []
        rtxt = labels.get("relation_text") or []
        evid = labels.get("evidence") or []

        n = min(len(heads), len(tails), len(rids), len(rtxt), len(evid))
        for i in range(n):
            rels.append(
                {
                    "head": heads[i],
                    "tail": tails[i],
                    "relation_id": _safe_str(rids[i]),
                    "relation_text": _safe_str(rtxt[i]),
                    "evidence": evid[i] if isinstance(evid[i], list) else [],
                }
            )
        return rels

    return []


def build_expected(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected target for evaluation.

    Note:
    - Some splits (e.g. test) may not include labels; relations will be [].
    """
    title = _safe_str(sample.get("title"))
    entities = build_entities(sample)
    relations = build_relations(sample)

    return {
        "title": title,
        "entities": entities,       # list of entity clusters
        "relations": relations,     # list of labeled relations (supervised splits)
    }


def iter_docred(
    split: str = "train_annotated",
    schema_id: str = DEFAULT_SCHEMA_ID,
    max_samples: Optional[int] = None,
) -> Iterator[ExtractionExample]:
    """
    Yields ExtractionExample objects suitable for evaluation.

    Splits on HF include (per dataset card):
      train_annotated, train_distant, validation, test
    """
    ds = load_dataset("thunlp/docred", split=split, trust_remote_code=True)

    n = 0
    for i, sample in enumerate(ds):
        if max_samples is not None and n >= max_samples:
            break

        text = build_doc_text(sample)
        expected = build_expected(sample)
        sid = f"{split}:{i}"

        yield ExtractionExample(
            id=sid,
            schema_id=schema_id,
            text=text,
            expected=expected,
        )
        n += 1


def load_docred(
    split: str = "train_annotated",
    schema_id: str = DEFAULT_SCHEMA_ID,
    max_samples: Optional[int] = None,
) -> List[ExtractionExample]:
    return list(iter_docred(split=split, schema_id=schema_id, max_samples=max_samples))


if __name__ == "__main__":
    # quick smoke test
    for ex in iter_docred(max_samples=2):
        print("=" * 80)
        print("ID:", ex.id)
        print("SCHEMA:", ex.schema_id)
        print("TITLE:", ex.expected.get("title"))
        print("TEXT_HEAD:", repr(ex.text[:200]))
        print("N_ENTITIES:", len(ex.expected.get("entities", [])))
        print("N_RELATIONS:", len(ex.expected.get("relations", [])))