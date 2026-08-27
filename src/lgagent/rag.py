"""Validated, thread-safe lifecycle for local and remote RAG retrieval."""

from __future__ import annotations

import inspect
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

import yaml

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


class RAGInitializationError(RuntimeError):
    """Raised when retrieval cannot be made ready for search."""


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _reject_lfs_pointer(path: Path, label: str) -> None:
    if not path.is_file():
        return
    with path.open("rb") as stream:
        if stream.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
            raise RAGInitializationError(
                f"{label} is a Git LFS pointer, not usable data: {path}. "
                "Run `git lfs pull` before enabling RAG."
            )


def _validate_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RAGInitializationError(f"{label} does not exist or is not a file: {path}")
    _reject_lfs_pointer(path, label)
    if path.stat().st_size == 0:
        raise RAGInitializationError(f"{label} is empty: {path}")


def _validate_index(path: Path, label: str) -> None:
    if not path.exists():
        raise RAGInitializationError(f"{label} does not exist: {path}")
    if path.is_file():
        _validate_file(path, label)
        return
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    if not files:
        raise RAGInitializationError(f"{label} contains no index files: {path}")
    for candidate in files:
        _reject_lfs_pointer(candidate, label)


@dataclass(frozen=True)
class RetrieverSettings:
    server_root: Path
    parameter_path: Path
    project_root: Path
    mode: str
    top_k: int
    parameters: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        server_root: str | Path,
        *,
        mode: str = "local",
        top_k: int | None = None,
        project_root: str | Path | None = None,
    ) -> "RetrieverSettings":
        root = Path(server_root).expanduser().resolve()
        parameter_path = root / "retriever" / "parameter.yaml"
        if not parameter_path.is_file():
            raise RAGInitializationError(
                f"retriever parameter metadata is missing: {parameter_path}"
            )
        with parameter_path.open("r", encoding="utf-8") as stream:
            parameters = yaml.safe_load(stream) or {}
        if not isinstance(parameters, Mapping):
            raise RAGInitializationError(
                f"retriever parameters must be a mapping: {parameter_path}"
            )
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"local", "remote"}:
            raise RAGInitializationError(
                f"retriever mode must be 'local' or 'remote', got {mode!r}"
            )
        base = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else root.parent
        )
        configured_top_k = parameters.get("top_k", 5) if top_k is None else top_k
        if not isinstance(configured_top_k, int) or configured_top_k <= 0:
            raise RAGInitializationError("retriever top_k must be a positive integer")
        return cls(
            server_root=root,
            parameter_path=parameter_path,
            project_root=base,
            mode=normalized_mode,
            top_k=configured_top_k,
            parameters=dict(parameters),
        )

    @property
    def metadata_path(self) -> Path:
        return self.server_root / "retriever" / "server.yaml"

    @property
    def server_path(self) -> Path:
        return self.server_root / "retriever" / "src" / "retriever.py"

    @property
    def corpus_path(self) -> Path:
        value = self.parameters.get("corpus_path")
        if not value:
            raise RAGInitializationError("retriever corpus_path is required")
        return _resolve_path(str(value), self.project_root)

    @property
    def remote_url(self) -> str:
        value = str(self.parameters.get("retriever_url", "")).strip()
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RAGInitializationError(
                f"retriever_url must be an HTTP(S) URL, got {value!r}"
            )
        return value

    def index_path(self) -> Path:
        backend = str(self.parameters.get("backend", "")).lower()
        backend_configs = self.parameters.get("backend_configs") or {}
        if backend == "bm25":
            value = (backend_configs.get("bm25") or {}).get("save_path")
            label = "BM25 index"
        else:
            index_backend = str(self.parameters.get("index_backend", "faiss")).lower()
            index_configs = self.parameters.get("index_backend_configs") or {}
            config = index_configs.get(index_backend) or {}
            value = config.get("index_path") if index_backend == "faiss" else config.get("uri")
            label = f"{index_backend} index"
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                raise RAGInitializationError(
                    f"{label} is remote and cannot be validated as a local index"
                )
        if not value:
            raise RAGInitializationError(f"{label} path is required")
        return _resolve_path(str(value), self.project_root)

    @property
    def embedding_path(self) -> Path:
        value = self.parameters.get("embedding_path")
        if not value:
            raise RAGInitializationError(
                "retriever embedding_path is required to build a dense index"
            )
        return _resolve_path(str(value), self.project_root)

    def init_arguments(self) -> dict[str, Any]:
        keys = (
            "model_name_or_path",
            "backend_configs",
            "batch_size",
            "gpu_ids",
            "is_multimodal",
            "backend",
            "index_backend",
            "index_backend_configs",
        )
        arguments = {key: self.parameters[key] for key in keys if key in self.parameters}
        arguments["corpus_path"] = str(self.corpus_path)
        return arguments


def build_retriever_metadata(settings: RetrieverSettings) -> None:
    """Build server.yaml from the retriever's registered tools."""
    _validate_file(settings.server_path, "retriever server")
    code = (
        "import runpy, sys; from pathlib import Path; "
        "sys.path.insert(0, str(Path(sys.argv[1]).parent)); "
        "module = runpy.run_path(sys.argv[1]); "
        "module['app'].build(sys.argv[2])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(settings.server_path), str(settings.parameter_path)],
        cwd=settings.project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RAGInitializationError(f"failed to build retriever metadata: {detail}")


def _metadata_is_valid(settings: RetrieverSettings) -> bool:
    if not settings.metadata_path.is_file():
        return False
    try:
        with settings.metadata_path.open("r", encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream) or {}
        tools = metadata.get("tools", {})
        required = {"retriever_init", "retriever_search", "retriever_deploy_search"}
        return required.issubset(tools)
    except (OSError, yaml.YAMLError, AttributeError):
        return False


class RAGRetriever:
    """One search interface with a once-only initialization barrier."""

    def __init__(
        self,
        settings: RetrieverSettings,
        *,
        tool_call: Any = None,
        api_initializer: Callable[..., Any] | None = None,
        api_shutdown: Callable[..., Any] | None = None,
        metadata_builder: Callable[[RetrieverSettings], None] = build_retriever_metadata,
    ) -> None:
        self.settings = settings
        self._tool_call = tool_call
        self._api_initializer = api_initializer
        self._api_shutdown = api_shutdown
        self._metadata_builder = metadata_builder
        self._condition = threading.Condition()
        self._state = "new"
        self._error: BaseException | None = None
        self._search_lock = threading.Lock()
        self._mcp_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lgagent-rag"
        )

    def initialize(self) -> None:
        with self._condition:
            while self._state == "initializing":
                self._condition.wait()
            if self._state == "ready":
                return
            if self._state == "closed":
                raise RAGInitializationError("retriever lifecycle is closed")
            if self._state == "failed":
                assert self._error is not None
                raise RAGInitializationError(str(self._error)) from self._error
            self._state = "initializing"

        try:
            self._initialize_once()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._state = "failed"
                self._condition.notify_all()
            if isinstance(exc, RAGInitializationError):
                raise
            raise RAGInitializationError(f"retriever initialization failed: {exc}") from exc

        with self._condition:
            self._state = "ready"
            self._condition.notify_all()

    def _initialize_once(self) -> None:
        if not _metadata_is_valid(self.settings):
            self._metadata_builder(self.settings)
        if not _metadata_is_valid(self.settings):
            raise RAGInitializationError(
                f"retriever metadata is missing required tools: {self.settings.metadata_path}"
            )

        index_exists = False
        if self.settings.mode == "local":
            _validate_file(self.settings.corpus_path, "retriever corpus")
            index_path = self.settings.index_path()
            index_exists = index_path.exists()
            if index_exists:
                _validate_index(index_path, "retriever index")
        else:
            self.settings.remote_url

        if self._tool_call is None or self._api_initializer is None:
            from ultrarag.api import ToolCall, initialize, shutdown

            self._tool_call = ToolCall
            self._api_initializer = initialize
            self._api_shutdown = shutdown
        self._api_initializer(
            ["retriever"], server_root=str(self.settings.server_root)
        )
        if self.settings.mode == "local":
            self._invoke(
                self._tool_call.retriever.retriever_init,
                **self.settings.init_arguments(),
            )
            if not index_exists:
                self._build_index()
                _validate_index(self.settings.index_path(), "retriever index")

    def _build_index(self) -> None:
        backend = str(self.settings.parameters.get("backend", "")).lower()
        overwrite = bool(self.settings.parameters.get("overwrite", False))
        if backend == "bm25":
            self._invoke(
                self._tool_call.retriever.bm25_index,
                overwrite=overwrite,
            )
            return

        embedding_path = self.settings.embedding_path
        self._invoke(
            self._tool_call.retriever.retriever_embed,
            embedding_path=str(embedding_path),
            overwrite=overwrite,
            is_multimodal=bool(
                self.settings.parameters.get("is_multimodal", False)
            ),
        )
        self._invoke(
            self._tool_call.retriever.retriever_index,
            embedding_path=str(embedding_path),
            overwrite=overwrite,
        )

    def _invoke(self, function: Callable[..., Any], **kwargs: Any) -> Any:
        def call_on_mcp_thread() -> Any:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = function(**kwargs)
            if inspect.isawaitable(result):
                return loop.run_until_complete(result)
            return result

        return self._mcp_executor.submit(call_on_mcp_thread).result()

    def close(self) -> None:
        with self._condition:
            if self._state == "closed":
                return
            state = self._state
            self._state = "closed"
        if state == "ready" and self._api_shutdown is not None:
            self._invoke(self._api_shutdown)

        def close_loop() -> None:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                return
            if not loop.is_running():
                loop.close()

        self._mcp_executor.submit(close_loop).result()
        self._mcp_executor.shutdown(wait=True)

    def search(self, query: str, *, top_k: int | None = None) -> list[str]:
        self.initialize()
        limit = self.settings.top_k if top_k is None else top_k
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("top_k must be a positive integer")
        method = (
            self._tool_call.retriever.retriever_search
            if self.settings.mode == "local"
            else self._tool_call.retriever.retriever_deploy_search
        )
        kwargs: dict[str, Any] = {"query_list": [query], "top_k": limit}
        if self.settings.mode == "remote":
            kwargs["retriever_url"] = self.settings.remote_url
        with self._search_lock:
            result = self._invoke(method, **kwargs)
        passages = (result or {}).get("ret_psg", [])
        if not passages or not isinstance(passages[0], list):
            return []
        return [str(item).strip() for item in passages[0] if item]
