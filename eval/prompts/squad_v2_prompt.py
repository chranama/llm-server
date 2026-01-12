# src/llm_server/eval/prompts/squad_v2_prompt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Keep the "contract" explicit and tight.
# We want a stable, machine-checkable output from /v1/generate.
NO_ANSWER_TOKEN = "NO_ANSWER"


@dataclass(frozen=True)
class SquadV2Prompt:
    """
    Build prompts for /v1/generate to behave like an extractive QA engine.

    Output contract:
      - Return EXACTLY ONE line.
      - If answerable: return the shortest answer span string.
      - If unanswerable: return the token NO_ANSWER.
    """
    no_answer_token: str = NO_ANSWER_TOKEN


def build_squad_v2_prompt(
    *,
    context: str,
    question: str,
    title: Optional[str] = None,
    no_answer_token: str = NO_ANSWER_TOKEN,
) -> str:
    title_line = f"TITLE: {title}\n" if title else ""

    return (
        "You are an extractive QA system.\n"
        "Answer the QUESTION using ONLY the CONTEXT.\n"
        "Rules:\n"
        f"- If the answer is not present in the context, output exactly: {no_answer_token}\n"
        "- Otherwise, output ONLY the answer text as it appears in the context.\n"
        "- Output must be EXACTLY ONE LINE. No quotes. No punctuation added. No explanation.\n\n"
        f"{title_line}"
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "ANSWER:\n"
    )