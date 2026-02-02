from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any

@dataclass
class FakeLLM:
    outputs: list[str]
    sleep_s: float = 0.0

    def __post_init__(self):
        self._i = 0

    def generate(self, **kwargs: Any) -> str:
        if self.sleep_s:
            time.sleep(self.sleep_s)

        if self._i >= len(self.outputs):
            return self.outputs[-1] if self.outputs else ""
        out = self.outputs[self._i]
        self._i += 1
        return out

    def is_loaded(self) -> bool:
        return True