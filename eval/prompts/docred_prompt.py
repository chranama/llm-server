# src/llm_server/eval/prompts/docred_prompt.py
"""
DocRED prompt builder for /v1/generate.

We provide the model:
- Document text
- Entity list with stable IDs (matching vertexSet order)

We ask for ONLY JSON:
{
  "relations": [
    {"head": <int>, "tail": <int>, "relation_id": "<str>"}
  ]
}

This aligns with the provided metrics which score sets of triples:
(head, tail, relation_id)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_JSON_BEGIN = "<<<JSON>>>"
_JSON_END = "<<<END>>>"


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def _entity_brief(entity: Dict[str, Any], *, max_mentions: int = 3) -> str:
    """
    Build a concise entity line like:
      [12] Barack Obama | mentions: Barack Obama; Obama
    """
    eid = entity.get("id")
    name = _safe_str(entity.get("name")) or "UNKNOWN"

    mentions = entity.get("mentions") if isinstance(entity.get("mentions"), list) else []
    mnames: List[str] = []
    for m in mentions:
        if not isinstance(m, dict):
            continue
        n = _safe_str(m.get("name"))
        if n:
            mnames.append(n)

    # de-dupe while preserving order
    seen = set()
    uniq: List[str] = []
    for n in mnames:
        if n not in seen:
            seen.add(n)
            uniq.append(n)

    uniq = uniq[:max_mentions]
    if uniq:
        return f"[{eid}] {name} | mentions: " + "; ".join(uniq)
    return f"[{eid}] {name}"


def build_docred_prompt(
    *,
    text: str,
    entities: List[Dict[str, Any]],
    title: Optional[str] = None,
) -> str:
    """
    Build DocRED relation extraction prompt.

    Contract:
    - return ONLY JSON
    - only key is "relations"
    - each relation has head/tail as entity IDs from the provided entity list
    - relation_id is a string (DocRED commonly uses relation IDs like "P17", etc.)
    """
    title_s = _safe_str(title)
    text_s = _safe_str(text)

    ent_lines: List[str] = []
    for e in entities:
        if isinstance(e, dict):
            ent_lines.append(_entity_brief(e))

    ent_block = "\n".join(ent_lines).strip()

    return (
        "You are an information extraction engine.\n"
        "Extract document-level relations between the PROVIDED ENTITIES.\n"
        "Return ONLY a JSON object. No markdown. No commentary.\n\n"
        "Rules:\n"
        "1) head and tail MUST be integers referencing entity IDs from PROVIDED ENTITIES.\n"
        "2) relation_id MUST be a string.\n"
        "3) Output MUST be a single JSON object with exactly one top-level key: relations.\n"
        "4) If no relations can be confidently extracted, return {\"relations\": []}.\n\n"
        f"OUTPUT FORMAT:\n{_JSON_BEGIN}\n{{\"relations\": [{{\"head\": 0, \"tail\": 1, \"relation_id\": \"Pxxx\"}}]}}\n{_JSON_END}\n\n"
        + (f"TITLE:\n{title_s}\n\n" if title_s else "")
        + "PROVIDED ENTITIES (use these IDs):\n"
        + ent_block
        + "\n\nDOCUMENT TEXT:\n"
        + text_s
        + "\n"
    )