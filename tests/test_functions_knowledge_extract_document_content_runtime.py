# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime coverage for document-content extraction helpers."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from world_understanding.functions.knowledge import extract_document_content as docmod


def _install_extract_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    schema: type | None = None,
    extractor_map: dict[str, object] | None = None,
) -> None:
    for name in [
        "nv_ingest_client",
        "nv_ingest_client.primitives",
        "nv_ingest_client.primitives.tasks",
        "nv_ingest_client.primitives.tasks.extract",
    ]:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    extract = sys.modules["nv_ingest_client.primitives.tasks.extract"]
    extract.ExtractTaskSchema = schema or (lambda **_kwargs: None)  # type: ignore[attr-defined]
    extract._DEFAULT_EXTRACTOR_MAP = extractor_map or {"pdf": object()}  # type: ignore[attr-defined]


def _install_client_module(
    monkeypatch: pytest.MonkeyPatch,
    ingestor_cls: type,
    nv_ingest_client_cls: type | None = None,
) -> None:
    client = ModuleType("nv_ingest_client.client")
    client.Ingestor = ingestor_cls  # type: ignore[attr-defined]
    if nv_ingest_client_cls is not None:
        client.NvIngestClient = nv_ingest_client_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nv_ingest_client.client", client)


def _install_setup_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"run_pipeline": [], "clients": []}
    for name in [
        "nv_ingest",
        "nv_ingest.framework",
        "nv_ingest.framework.orchestration",
        "nv_ingest.framework.orchestration.ray",
        "nv_ingest.framework.orchestration.ray.util",
        "nv_ingest.framework.orchestration.ray.util.pipeline",
        "nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners",
        "nv_ingest_api",
        "nv_ingest_api.util",
        "nv_ingest_api.util.message_brokers",
        "nv_ingest_api.util.message_brokers.simple_message_broker",
    ]:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    class PipelineCreationSchema:
        pass

    def run_pipeline(config: object, **kwargs: Any) -> None:
        calls["run_pipeline"].append({"config": config, **kwargs})

    class SimpleClient:
        pass

    class NvIngestClient:
        def __init__(self, **kwargs: Any):
            calls["clients"].append(kwargs)

    runners = sys.modules[
        "nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners"
    ]
    runners.PipelineCreationSchema = PipelineCreationSchema  # type: ignore[attr-defined]
    runners.run_pipeline = run_pipeline  # type: ignore[attr-defined]
    broker = sys.modules["nv_ingest_api.util.message_brokers.simple_message_broker"]
    broker.SimpleClient = SimpleClient  # type: ignore[attr-defined]
    _install_client_module(monkeypatch, type("UnusedIngestor", (), {}), NvIngestClient)
    calls["SimpleClient"] = SimpleClient
    return calls


def test_data_preprocessor_validation_setup_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingSchema:
        calls: list[dict[str, Any]] = []

        def __init__(self, **kwargs: Any):
            self.calls.append(kwargs)

    _install_extract_module(monkeypatch, schema=RecordingSchema)
    docmod.DataPreprocessor._nv_ingest_client = object()

    preprocessor = docmod.DataPreprocessor(
        {"extract_method": "fast", "text_depth": "doc"}
    )

    assert preprocessor.nv_ingest_client is docmod.DataPreprocessor._nv_ingest_client
    assert RecordingSchema.calls[-1] == {
        "document_type": "pdf",
        "extract_method": "fast",
        "text_depth": "doc",
    }

    class BadSchema:
        def __init__(self, **_kwargs: Any):
            raise ValueError("bad config")

    _install_extract_module(monkeypatch, schema=BadSchema)
    with pytest.raises(RuntimeError, match="Invalid DataPreprocessor configuration"):
        docmod.DataPreprocessor({"extract_method": "bad"})

    closed: list[bool] = []
    docmod.DataPreprocessor._nv_ingest_client = SimpleNamespace(
        close=lambda: closed.append(True)
    )
    preprocessor.cleanup()
    assert closed == [True]
    assert docmod.DataPreprocessor._nv_ingest_client is None

    docmod.DataPreprocessor._nv_ingest_client = SimpleNamespace(
        close=lambda: (_ for _ in ()).throw(RuntimeError("close failed"))
    )
    preprocessor.cleanup()

    for name in [
        "nv_ingest_client.primitives.tasks.extract",
        "nv_ingest_client.primitives.tasks",
        "nv_ingest_client.primitives",
        "nv_ingest_client",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    preprocessor._validate_config()


def test_data_preprocessor_setup_uses_fake_nv_ingest_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_extract_module(monkeypatch)
    setup_calls = _install_setup_modules(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "api-key")
    docmod.DataPreprocessor._nv_ingest_client = None

    preprocessor = docmod.DataPreprocessor({"timeout": 11})

    assert preprocessor.nv_ingest_client is docmod.DataPreprocessor._nv_ingest_client
    assert setup_calls["run_pipeline"][0]["block"] is False
    assert setup_calls["run_pipeline"][0]["disable_dynamic_scaling"] is True
    assert setup_calls["run_pipeline"][0]["run_in_subprocess"] is True
    assert (
        setup_calls["clients"][0]["message_client_allocator"]
        is setup_calls["SimpleClient"]
    )
    assert setup_calls["clients"][0]["message_client_kwargs"] == {
        "connection_timeout": 11
    }

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    docmod.DataPreprocessor._nv_ingest_client = None
    with pytest.raises(RuntimeError, match="nv_ingest setup failed"):
        docmod.DataPreprocessor()


def test_validate_file_paths_and_document_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_extract_module(monkeypatch)
    document = tmp_path / "doc.pdf"
    document.write_text("content", encoding="utf-8")

    docmod.DataPreprocessor._validate_file_paths([str(document)])

    for name in [
        "nv_ingest_client.primitives.tasks.extract",
        "nv_ingest_client.primitives.tasks",
        "nv_ingest_client.primitives",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    docmod.DataPreprocessor._validate_file_paths([str(document)])
    _install_extract_module(monkeypatch)

    with pytest.raises(ValueError, match="Invalid file path"):
        no_extension = tmp_path / "doc"
        no_extension.write_text("content", encoding="utf-8")
        docmod.DataPreprocessor._validate_file_paths([str(no_extension)])

    with pytest.raises(ValueError, match="Error checking document type support"):
        docmod.DataPreprocessor._validate_file_paths([str(tmp_path / "missing.pdf")])

    assert docmod.DataPreprocessor.get_document_content(
        {
            "document_type": "structured",
            "metadata": {"table_metadata": {"table_content": "table"}},
        }
    ) == ("structured", "table")
    assert docmod.DataPreprocessor.get_document_content(
        {"document_type": "text", "metadata": {"content": "text"}}
    ) == ("text", "text")
    assert docmod.DataPreprocessor.get_document_content(
        {"document_type": "image", "metadata": {"content": "encoded"}}
    ) == ("image", "encoded")
    assert docmod.DataPreprocessor.get_document_content(
        {
            "document_type": "audio",
            "metadata": {"audio_metadata": {"audio_transcript": "audio"}},
        }
    ) == ("audio", "audio")
    with pytest.raises(NotImplementedError):
        docmod.DataPreprocessor.get_document_content(
            {"document_type": "binary", "metadata": {}}
        )


def test_preprocess_documents_success_retry_and_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_extract_module(monkeypatch)
    document = tmp_path / "doc.pdf"
    document.write_text("content", encoding="utf-8")
    doc = {
        "document_type": "text",
        "metadata": {
            "content": "hello",
            "source_metadata": {"source_name": str(document)},
        },
    }

    class FakeIngestor:
        calls = 0

        def __init__(self, client: object):
            self.client = client

        def files(self, file_paths: list[str]) -> FakeIngestor:
            self.file_paths = file_paths
            return self

        def extract(self, **kwargs: Any) -> FakeIngestor:
            self.extract_kwargs = kwargs
            return self

        def ingest(
            self, **kwargs: Any
        ) -> tuple[list[list[dict[str, Any]]], list[tuple[str, str]]]:
            type(self).calls += 1
            self.ingest_kwargs = kwargs
            return ([[doc]], [])

    _install_client_module(monkeypatch, FakeIngestor)
    preprocessor = docmod.DataPreprocessor.__new__(docmod.DataPreprocessor)
    preprocessor.config = {"extract_text": False, "show_progress": False, "timeout": 7}
    preprocessor.nv_ingest_client = object()

    result = preprocessor.preprocess_documents(
        [str(document)],
        save_content_only=True,
        batch_size=2,
    )
    assert result == {str(document): [{"document_type": "text", "content": "hello"}]}
    assert FakeIngestor.calls == 1

    full_result = preprocessor.preprocess_documents(
        [str(document)],
        save_content_only=False,
        batch_size=1,
    )
    assert full_result == {str(document): [doc]}

    class RetryIngestor(FakeIngestor):
        calls = 0

        def ingest(self, **kwargs: Any):
            type(self).calls += 1
            if type(self).calls == 1:
                return ([], [(f"source:{document}", "failed once")])
            return ([[doc]], [])

    _install_client_module(monkeypatch, RetryIngestor)
    assert preprocessor.preprocess_documents([str(document)], max_retries=2)

    with pytest.raises(ValueError, match="No file paths provided"):
        preprocessor.preprocess_documents([])

    preprocessor.nv_ingest_client = None
    with pytest.raises(RuntimeError, match="No nv_ingest client available"):
        preprocessor.preprocess_documents([str(document)])

    preprocessor.nv_ingest_client = object()

    class EmptyIngestor(FakeIngestor):
        def ingest(self, **_kwargs: Any):
            return ([], [])

    _install_client_module(monkeypatch, EmptyIngestor)
    with pytest.raises(RuntimeError, match="returned no results"):
        preprocessor.preprocess_documents([str(document)])

    class EmptyResultIngestor(FakeIngestor):
        def ingest(self, **_kwargs: Any):
            return ([[]], [])

    _install_client_module(monkeypatch, EmptyResultIngestor)
    with pytest.raises(RuntimeError, match="No result found"):
        preprocessor.preprocess_documents([str(document)])

    class AlwaysFailIngestor(FakeIngestor):
        def ingest(self, **_kwargs: Any):
            return ([], [(f"source:{document}", "still failed")])

    _install_client_module(monkeypatch, AlwaysFailIngestor)
    with pytest.raises(RuntimeError, match="returned failures"):
        preprocessor.preprocess_documents([str(document)], max_retries=1)


def test_extract_document_content_collects_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_extract_module(
        monkeypatch, extractor_map={"pdf": object(), "txt": object()}
    )
    pdf = tmp_path / "doc.pdf"
    txt = tmp_path / "doc.txt"
    ignored = tmp_path / "ignored.bin"
    pdf.write_text("pdf", encoding="utf-8")
    txt.write_text("txt", encoding="utf-8")
    ignored.write_text("ignored", encoding="utf-8")

    class FakePreprocessor:
        instances: list[FakePreprocessor] = []

        def __init__(self):
            self.calls: list[dict[str, Any]] = []
            self.instances.append(self)

        def preprocess_documents(self, file_paths: list[str], **kwargs: Any):
            self.calls.append({"file_paths": file_paths, **kwargs})
            return {"files": file_paths}

    monkeypatch.setattr(docmod, "DataPreprocessor", FakePreprocessor)

    assert docmod.extract_document_content(pdf)["files"] == [str(pdf)]
    assert sorted(docmod.extract_document_content(tmp_path)["files"]) == [
        str(pdf),
        str(txt),
    ]
    assert docmod.extract_document_content([pdf, ignored], batch_size=3)["files"] == [
        str(pdf)
    ]
    assert FakePreprocessor.instances[-1].calls[-1]["batch_size"] == 3

    with pytest.raises(FileNotFoundError):
        docmod.extract_document_content(tmp_path / "missing")
    with pytest.raises(ValueError, match="No document files found"):
        docmod.extract_document_content([ignored])
    with pytest.raises(ValueError, match="Invalid source type"):
        docmod.extract_document_content(object())  # type: ignore[arg-type]


def test_extract_document_content_requires_nv_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(
        sys.modules, "nv_ingest_client.primitives.tasks.extract", raising=False
    )
    with pytest.raises(RuntimeError, match="nv_ingest is required"):
        docmod.extract_document_content("unused.pdf")


def test_split_document_content_by_type(tmp_path: Path) -> None:
    payload = {
        str(tmp_path / "doc.pdf"): [
            {"document_type": "structured", "content": "table"},
            {"document_type": "text", "content": "plain"},
            {
                "document_type": "image",
                "content": base64.b64encode(b"png-bytes").decode("ascii"),
            },
            {"document_type": "image", "content": "not base64"},
            {"document_type": "audio", "content": "ignored"},
        ]
    }
    input_file = tmp_path / "content.json"
    input_file.write_text(json.dumps(payload), encoding="utf-8")

    created = docmod.split_document_content_by_type(
        str(input_file), str(tmp_path / "out")
    )

    paths = created["doc"]
    assert len(paths) == 3
    assert paths[0].read_text(encoding="utf-8") == "table"
    assert paths[1].read_text(encoding="utf-8") == "plain"
    assert paths[2].read_bytes() == b"png-bytes"

    with pytest.raises(ValueError, match="Invalid input file"):
        docmod.split_document_content_by_type(
            str(tmp_path / "missing.json"), str(tmp_path / "out")
        )
    with pytest.raises(ValueError, match="Invalid input file"):
        docmod.split_document_content_by_type(str(tmp_path), str(tmp_path / "out"))
