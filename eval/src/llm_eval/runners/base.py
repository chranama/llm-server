# llm_eval/runners/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING


if TYPE_CHECKING:
    # Type-only imports so importing llm_eval.runners.* stays light.
    from llm_eval.client.http_client import ExtractErr, ExtractOk, GenerateErr, GenerateOk


@dataclass
class EvalConfig:
    """
    Shared configuration object for evaluation runs.

    - max_examples: optional cap on number of examples to evaluate
    - model_override: optional model name to pass through to the server
    """
    max_examples: Optional[int] = None
    model_override: Optional[str] = None


# -------------------------
# Runtime deps (DI seam)
# -------------------------


class HttpClient(Protocol):
    """
    Structural protocol for the eval HTTP client.

    Runners must NOT depend on httpx directly.
    They only depend on this protocol + typed results.
    """

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        model: Optional[str] = None,
        cache: Optional[bool] = None,
    ) -> "GenerateOk | GenerateErr":
        ...

    async def extract(
        self,
        *,
        schema_id: str,
        text: str,
        model: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        cache: bool = True,
        repair: bool = True,
    ) -> "ExtractOk | ExtractErr":
        ...


ClientFactory = Callable[[str, str], HttpClient]
RunIdFactory = Callable[[], str]
EnsureDirFn = Callable[[str], None]
OpenFn = Callable[..., Any]
DatasetOverrides = dict[str, Any]  # callables keyed by dataset name


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_ensure_dir(path: str) -> None:
    import os
    os.makedirs(path, exist_ok=True)


@dataclass(frozen=True)
class RunnerDeps:
    """
    Injectable runtime dependencies.

    In prod you use defaults.
    In tests you pass fakes to avoid HTTP/filesystem/time nondeterminism.

    dataset_overrides:
      - A dict of callables keyed by dataset name (runner-defined keys).
      - Lets tests inject small fixture iterators without monkeypatching imports.
    """
    client_factory: ClientFactory
    run_id_factory: RunIdFactory = _default_run_id
    ensure_dir: EnsureDirFn = _default_ensure_dir
    open_fn: OpenFn = open  # allows in-memory file capture in tests if you want
    dataset_overrides: DatasetOverrides = field(default_factory=dict)


def default_deps() -> RunnerDeps:
    # Import inside function so llm_eval package import is light-weight
    from llm_eval.client.http_client import HttpEvalClient

    return RunnerDeps(
        client_factory=lambda base_url, api_key: HttpEvalClient(base_url=base_url, api_key=api_key)
    )


# -------------------------
# Base runner
# -------------------------


class BaseEvalRunner(ABC):
    task_name: str = "base"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        config: Optional[EvalConfig] = None,
        *,
        deps: Optional[RunnerDeps] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.config = config or EvalConfig()
        self.deps = deps or default_deps()

    def make_client(self) -> HttpClient:
        return self.deps.client_factory(self.base_url, self.api_key)

    def new_run_id(self) -> str:
        return self.deps.run_id_factory()

    def ensure_dir(self, path: str) -> None:
        self.deps.ensure_dir(path)

    def open_file(self, *args: Any, **kwargs: Any) -> Any:
        return self.deps.open_fn(*args, **kwargs)

    def get_dataset_callable(self, key: str, default: Any) -> Any:
        """
        Returns an override callable if present, else the provided default.
        Intended for dataset iterators like iter_docred, iter_squad_v2, etc.
        """
        overrides = self.deps.dataset_overrides
        if isinstance(overrides, dict) and key in overrides and overrides[key] is not None:
            return overrides[key]
        return default

    async def run(
        self,
        max_examples: Optional[int] = None,
        model_override: Optional[str] = None,
    ) -> Any:
        if max_examples is not None:
            self.config.max_examples = max_examples
        if model_override is not None:
            self.config.model_override = model_override
        return await self._run_impl()

    @abstractmethod
    async def _run_impl(self) -> Any:
        raise NotImplementedError