from __future__ import annotations

import asyncio
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lgagent.rag import (
    RAGInitializationError,
    RAGRetriever,
    RetrieverSettings,
    build_retriever_metadata,
)


def write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def write_metadata(path: Path) -> None:
    write_yaml(
        path,
        {
            "path": "retriever.py",
            "tools": {
                "retriever_init": {"input": {}},
                "retriever_search": {"input": {}},
                "retriever_deploy_search": {"input": {}},
            },
        },
    )


class FakeRetrieverTools:
    def __init__(self, events: list[str], index_path: Path | None = None) -> None:
        self.events = events
        self.index_path = index_path
        self.init_calls = 0
        self.local_calls: list[dict[str, object]] = []
        self.remote_calls: list[dict[str, object]] = []
        self.call_threads: list[int] = []

    def retriever_init(self, **kwargs: object) -> None:
        asyncio.get_event_loop()
        self.call_threads.append(threading.get_ident())
        self.init_calls += 1
        self.events.append("retriever_init")

    def retriever_search(self, **kwargs: object) -> dict[str, list[list[str]]]:
        asyncio.get_event_loop()
        self.call_threads.append(threading.get_ident())
        self.local_calls.append(kwargs)
        self.events.append("local_search")
        return {"ret_psg": [[" local result "]]}

    def retriever_embed(self, **kwargs: object) -> None:
        self.events.append("retriever_embed")
        embedding_path = Path(str(kwargs["embedding_path"]))
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        embedding_path.write_bytes(b"embedding")

    def retriever_index(self, **kwargs: object) -> None:
        self.events.append("retriever_index")
        if self.index_path is None:
            raise RuntimeError("fake index_path is not configured")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_bytes(b"index")

    def bm25_index(self, **kwargs: object) -> None:
        self.events.append("bm25_index")
        if self.index_path is None:
            raise RuntimeError("fake index_path is not configured")
        self.index_path.mkdir(parents=True, exist_ok=True)
        (self.index_path / "index.json").write_text("{}", encoding="utf-8")

    def retriever_deploy_search(
        self, **kwargs: object
    ) -> dict[str, list[list[str]]]:
        asyncio.get_event_loop()
        self.call_threads.append(threading.get_ident())
        self.remote_calls.append(kwargs)
        self.events.append("remote_search")
        return {"ret_psg": [[" remote result "]]}


class RAGLifecycleOfflineTest(unittest.TestCase):
    def make_settings(
        self,
        root: Path,
        *,
        mode: str = "local",
        corpus_contents: bytes = b'{"id":"1","contents":"law"}\n',
        index_contents: bytes = b"index",
    ) -> RetrieverSettings:
        server_root = root / "servers"
        retriever_root = server_root / "retriever"
        (retriever_root / "src").mkdir(parents=True)
        (retriever_root / "src" / "retriever.py").write_text(
            "# fake retriever\n", encoding="utf-8"
        )
        (root / "data").mkdir()
        (root / "data" / "corpus.jsonl").write_bytes(corpus_contents)
        (root / "index").mkdir()
        (root / "index" / "index.bin").write_bytes(index_contents)
        write_yaml(
            retriever_root / "parameter.yaml",
            {
                "model_name_or_path": "fake",
                "corpus_path": "data/corpus.jsonl",
                "embedding_path": "embedding/embedding.npy",
                "backend": "sentence_transformers",
                "backend_configs": {"sentence_transformers": {}},
                "index_backend": "faiss",
                "index_backend_configs": {
                    "faiss": {"index_path": "index/index.bin"}
                },
                "batch_size": 1,
                "gpu_ids": None,
                "is_multimodal": False,
                "top_k": 5,
                "retriever_url": "https://retriever.invalid/search",
            },
        )
        return RetrieverSettings.load(
            server_root, mode=mode, project_root=root
        )

    def make_lifecycle(
        self,
        settings: RetrieverSettings,
        events: list[str],
        tools: FakeRetrieverTools,
        *,
        build_delay: float = 0.0,
    ) -> RAGRetriever:
        def build(current: RetrieverSettings) -> None:
            if build_delay:
                time.sleep(build_delay)
            events.append("metadata")
            write_metadata(current.metadata_path)

        def api_initialize(*args: object, **kwargs: object) -> None:
            events.append("mcp_initialize")

        lifecycle = RAGRetriever(
            settings,
            tool_call=SimpleNamespace(retriever=tools),
            api_initializer=api_initialize,
            metadata_builder=build,
        )
        self.addCleanup(lifecycle.close)
        return lifecycle

    def test_local_search_builds_metadata_and_initializes_before_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events: list[str] = []
            settings = self.make_settings(Path(directory))
            tools = FakeRetrieverTools(events)
            lifecycle = self.make_lifecycle(settings, events, tools)

            self.assertEqual(lifecycle.search("question", top_k=3), ["local result"])

            self.assertEqual(
                events,
                ["metadata", "mcp_initialize", "retriever_init", "local_search"],
            )
            self.assertEqual(tools.local_calls[0]["query_list"], ["question"])
            self.assertEqual(tools.local_calls[0]["top_k"], 3)
            self.assertEqual(len(set(tools.call_threads)), 1)

    def test_metadata_builder_resolves_server_sibling_imports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self.make_settings(Path(directory))
            settings.server_path.with_name("helper.py").write_text(
                "from pathlib import Path\n"
                "class App:\n"
                "    def build(self, parameter):\n"
                "        Path(parameter).with_name('server.yaml').write_text("
                "\"tools:\\n  retriever_init: {}\\n"
                "  retriever_search: {}\\n"
                "  retriever_deploy_search: {}\\n\", encoding='utf-8')\n",
                encoding="utf-8",
            )
            settings.server_path.write_text(
                "from helper import App\napp = App()\n", encoding="utf-8"
            )

            build_retriever_metadata(settings)

            self.assertTrue(settings.metadata_path.is_file())

    def test_concurrent_search_initializes_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events: list[str] = []
            settings = self.make_settings(Path(directory))
            tools = FakeRetrieverTools(events)
            lifecycle = self.make_lifecycle(
                settings, events, tools, build_delay=0.03
            )
            threads = [
                threading.Thread(target=lifecycle.search, args=(f"q{index}",))
                for index in range(8)
            ]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(events.count("metadata"), 1)
            self.assertEqual(events.count("mcp_initialize"), 1)
            self.assertEqual(tools.init_calls, 1)
            self.assertEqual(len(tools.local_calls), 8)
            self.assertEqual(len(set(tools.call_threads)), 1)

    def test_corpus_lfs_pointer_is_rejected_before_mcp_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events: list[str] = []
            settings = self.make_settings(
                Path(directory),
                corpus_contents=(
                    b"version https://git-lfs.github.com/spec/v1\n"
                    b"oid sha256:123\nsize 123\n"
                ),
            )
            write_metadata(settings.metadata_path)
            tools = FakeRetrieverTools(events)
            lifecycle = self.make_lifecycle(settings, events, tools)

            with self.assertRaisesRegex(RAGInitializationError, "Git LFS pointer"):
                lifecycle.initialize()

            self.assertNotIn("mcp_initialize", events)
            self.assertEqual(tools.init_calls, 0)

    def test_index_lfs_pointer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events: list[str] = []
            settings = self.make_settings(
                Path(directory),
                index_contents=(
                    b"version https://git-lfs.github.com/spec/v1\n"
                    b"oid sha256:456\nsize 456\n"
                ),
            )
            write_metadata(settings.metadata_path)
            lifecycle = self.make_lifecycle(
                settings, events, FakeRetrieverTools(events)
            )

            with self.assertRaisesRegex(RAGInitializationError, "Git LFS pointer"):
                lifecycle.initialize()

    def test_missing_dense_index_is_built_then_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events: list[str] = []
            settings = self.make_settings(Path(directory))
            settings.index_path().unlink()
            write_metadata(settings.metadata_path)
            tools = FakeRetrieverTools(events, settings.index_path())
            lifecycle = self.make_lifecycle(
                settings, events, tools
            )

            lifecycle.initialize()

            self.assertTrue(settings.index_path().is_file())
            self.assertEqual(events.count("retriever_embed"), 1)
            self.assertEqual(events.count("retriever_index"), 1)
            lifecycle.initialize()
            self.assertEqual(events.count("retriever_embed"), 1)
            self.assertEqual(events.count("retriever_index"), 1)

    def test_remote_and_local_retrievers_share_search_interface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events: list[str] = []
            settings = self.make_settings(Path(directory), mode="remote")
            settings.corpus_path.unlink()
            settings.index_path().unlink()
            write_metadata(settings.metadata_path)
            tools = FakeRetrieverTools(events)
            lifecycle = self.make_lifecycle(settings, events, tools)

            self.assertEqual(lifecycle.search("question"), ["remote result"])
            self.assertEqual(tools.init_calls, 0)
            self.assertEqual(
                tools.remote_calls[0]["retriever_url"],
                "https://retriever.invalid/search",
            )

    def test_ultrarag_api_uses_current_python_for_mcp_subprocess(self) -> None:
        import ultrarag.api as api

        with tempfile.TemporaryDirectory() as directory:
            server_root = Path(directory)
            source = server_root / "retriever" / "src" / "retriever.py"
            source.parent.mkdir(parents=True)
            source.write_text("# fake\n", encoding="utf-8")
            captured: dict[str, object] = {}

            class FakeClient:
                def __init__(self, config: object) -> None:
                    captured["config"] = config

            with patch.object(api, "Client", FakeClient):
                api.initialize(["retriever"], str(server_root))

            config = captured["config"]
            command = config["mcpServers"]["retriever"]["command"]  # type: ignore[index]
            self.assertEqual(command, sys.executable)


if __name__ == "__main__":
    unittest.main()
