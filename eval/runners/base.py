# src/llm_server/eval/runners/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EvalConfig:
    """
    Shared configuration object for evaluation runs.

    - max_examples: optional cap on number of examples to evaluate
    - model_override: optional model name to pass through to the server
      (you can later wire this into your /v1/generate payload if desired)
    """
    max_examples: Optional[int] = None
    model_override: Optional[str] = None


class BaseEvalRunner(ABC):
    task_name: str = "base"
    """
    Base class for evaluation runners.

    Subclasses implement `_run_impl()`, and callers use `await runner.run(...)`.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        config: Optional[EvalConfig] = None,
    ) -> None:
        # Normalize base_url a bit to avoid double slashes
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.config = config or EvalConfig()

    async def run(
        self,
        max_examples: Optional[int] = None,
        model_override: Optional[str] = None,
    ) -> Any:
        """
        Public entrypoint for running an evaluation.

        CLI or callers can override config per run; the rest of the logic is
        delegated to `_run_impl()`.
        """
        if max_examples is not None:
            self.config.max_examples = max_examples
        if model_override is not None:
            self.config.model_override = model_override

        return await self._run_impl()

    @abstractmethod
    async def _run_impl(self) -> Any:
        """
        Subclasses implement this with their task-specific logic.

        Should return a dict of metrics, e.g. {"task": "...", "accuracy": 0.73}
        """
        raise NotImplementedError