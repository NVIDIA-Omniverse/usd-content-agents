# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for telemetry exporters and tool wrapper helpers."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image
from rich.console import Console

import world_understanding.telemetry as telemetry
from world_understanding.registry import tool_registry as registry_module
from world_understanding.telemetry.config import (
    ExporterType,
    FileConfig,
    HTTPConfig,
    LangfuseConfig,
    TelemetryConfig,
    TempoConfig,
)
from world_understanding.telemetry.exporters import file as file_exporter_module
from world_understanding.telemetry.exporters import http as http_exporter_module
from world_understanding.telemetry.exporters import langfuse as langfuse_exporter_module
from world_understanding.telemetry.exporters import tempo as tempo_exporter_module
from world_understanding.telemetry.exporters.file import LocalJsonSpanExporter
from world_understanding.telemetry.exporters.http import HTTPJsonExporter
from world_understanding.tools import base as tool_base
from world_understanding.tools.base import (
    Tool,
    ToolInput,
    ToolOutput,
    ToolSpec,
    clear_registry,
    register_tool,
)
from world_understanding.tools.graphics import image_edit as image_edit_module


def _fake_span(parent: Any | None = None) -> Any:
    status_code = SimpleNamespace(name="OK")
    return SimpleNamespace(
        context=SimpleNamespace(trace_id=0xABC, span_id=0x123),
        parent=parent,
        name="span",
        kind=SimpleNamespace(name="SERVER"),
        start_time=1,
        end_time=2,
        attributes={"answer": 42},
        status=SimpleNamespace(status_code=status_code, description=None),
        resource=SimpleNamespace(attributes={"service.name": "svc"}),
        events=[
            SimpleNamespace(
                name="event",
                timestamp=3,
                attributes={"event.attr": "value"},
            )
        ],
        links=[
            SimpleNamespace(
                context=SimpleNamespace(trace_id=0xDEF, span_id=0x456),
                attributes={"link.attr": "value"},
            )
        ],
    )


def test_local_json_span_exporter_serializes_and_handles_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from opentelemetry.sdk.trace.export import SpanExportResult

    output_path = tmp_path / "spans" / "out.jsonl"
    exporter = LocalJsonSpanExporter(str(output_path), append=False)
    assert output_path.read_text(encoding="utf-8") == ""
    assert exporter.export([]) == SpanExportResult.SUCCESS
    assert exporter.export([_fake_span(SimpleNamespace(span_id=0x999))]) == (
        SpanExportResult.SUCCESS
    )

    record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["trace_id"].endswith("0abc")
    assert record["parent_span_id"] == "0000000000000999"
    assert record["attributes"] == {"answer": 42}
    assert record["resource"] == {"service.name": "svc"}
    assert exporter.shutdown() is None
    assert exporter.force_flush() is True

    processor = file_exporter_module.create_file_exporter(str(tmp_path / "ok.jsonl"))
    assert processor is not None
    processor.shutdown()

    def raising_open(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", raising_open)
    assert exporter.export([_fake_span()]) == SpanExportResult.FAILURE

    assert (
        file_exporter_module.create_file_exporter(
            str(tmp_path / "bad.jsonl"),
            max_queue_size=1,
            max_export_batch_size=2,
        )
        is None
    )


def test_http_json_exporter_serializes_posts_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.sdk.trace.export import SpanExportResult

    class FakeResponse:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail

        def raise_for_status(self) -> None:
            if self.fail:
                raise RuntimeError("bad status")

    class FakeSession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}
            self.posts: list[dict[str, Any]] = []
            self.closed = False
            self.fail = False

        def post(self, endpoint: str, data: str, timeout: int) -> FakeResponse:
            self.posts.append({"endpoint": endpoint, "data": data, "timeout": timeout})
            return FakeResponse(self.fail)

        def close(self) -> None:
            self.closed = True

    fake_session = FakeSession()
    monkeypatch.setattr(
        "requests.Session",
        lambda: fake_session,
    )

    exporter = HTTPJsonExporter(
        "https://collector.test/spans",
        headers={"Authorization": "Bearer token"},
        timeout=7,
    )
    assert exporter.export([]) == SpanExportResult.SUCCESS
    assert exporter.export([_fake_span(SimpleNamespace(span_id=0x999))]) == (
        SpanExportResult.SUCCESS
    )
    assert fake_session.headers["Content-Type"] == "application/json"
    assert fake_session.headers["Authorization"] == "Bearer token"
    payload = json.loads(fake_session.posts[0]["data"])
    assert payload["spans"][0]["parent_span_id"] == "0000000000000999"
    assert payload["spans"][0]["events"][0]["name"] == "event"
    assert payload["spans"][0]["links"][0]["span_id"].endswith("0456")
    assert payload["resource"] == {"service.name": "svc"}
    assert exporter._serialize_resource(None) == {}
    assert exporter._serialize_resource(SimpleNamespace(attributes={})) == {}

    fake_session.fail = True
    assert exporter.export([_fake_span()]) == SpanExportResult.FAILURE
    exporter.shutdown()
    assert fake_session.closed is True
    assert exporter.force_flush() is True

    assert (
        http_exporter_module.create_http_exporter(
            "https://collector.test/spans",
            max_queue_size=1,
            max_export_batch_size=2,
        )
        is None
    )


class FakeProvider:
    def __init__(self, fail_shutdown: bool = False) -> None:
        self.processors: list[Any] = []
        self.fail_shutdown = fail_shutdown
        self.shutdown_called = False

    def add_span_processor(self, processor: Any) -> None:
        self.processors.append(processor)

    def shutdown(self) -> None:
        self.shutdown_called = True
        if self.fail_shutdown:
            raise RuntimeError("shutdown failed")


def test_telemetry_reinitialization_exporter_branches_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # BaseSettings intentionally reads OTEL_* values, but this test exercises
    # explicit missing-config branches and must not inherit CI credentials.
    for name in tuple(os.environ):
        if name.startswith("OTEL_"):
            monkeypatch.delenv(name)

    telemetry._initialized = True
    telemetry._tracer_provider = "existing-provider"  # type: ignore[assignment]
    assert telemetry.initialize_telemetry(TelemetryConfig(enabled=True)) == (
        "existing-provider"
    )

    telemetry._initialized = False
    telemetry._tracer_provider = None
    telemetry._shutdown_registered = False
    from opentelemetry import trace

    monkeypatch.setattr(trace, "set_tracer_provider", lambda _provider: None)
    provider = telemetry.initialize_telemetry(
        TelemetryConfig(enabled=True, exporters="none", sample_rate=0.5)
    )
    assert provider is not None
    telemetry.shutdown_telemetry()

    import world_understanding.telemetry.exporters as exporters

    calls: list[tuple[str, dict[str, Any]]] = []

    def factory(name: str):
        def _create(**kwargs: Any) -> str:
            calls.append((name, kwargs))
            return f"{name}-processor"

        return _create

    monkeypatch.setattr(
        exporters, "create_console_exporter", lambda: "console-processor"
    )
    monkeypatch.setattr(exporters, "create_tempo_exporter", factory("tempo"))
    monkeypatch.setattr(exporters, "create_langfuse_exporter", factory("langfuse"))
    monkeypatch.setattr(exporters, "create_http_exporter", factory("http"))
    monkeypatch.setattr(exporters, "create_file_exporter", factory("file"))

    fake_provider = FakeProvider()
    telemetry._add_exporters(
        fake_provider,  # type: ignore[arg-type]
        TelemetryConfig(
            exporters="none,console,tempo,langfuse,http,file",
            tempo=TempoConfig(endpoint="tempo:4317"),
            langfuse=LangfuseConfig(public_key="pub", secret_key="sec"),
            http=HTTPConfig(endpoint="https://collector.test"),
            file=FileConfig(path="/tmp/spans.jsonl", append=False),
        ),
    )
    assert fake_provider.processors == [
        "console-processor",
        "tempo-processor",
        "langfuse-processor",
        "http-processor",
        "file-processor",
    ]
    assert [name for name, _kwargs in calls] == ["tempo", "langfuse", "http", "file"]

    missing_config_provider = FakeProvider()
    telemetry._add_exporters(
        missing_config_provider,  # type: ignore[arg-type]
        TelemetryConfig(exporters="tempo,langfuse,http,file"),
    )
    assert missing_config_provider.processors == []

    monkeypatch.setattr(exporters, "create_http_exporter", lambda **_kwargs: None)
    failed_provider = FakeProvider()
    telemetry._add_exporters(
        failed_provider,  # type: ignore[arg-type]
        TelemetryConfig(
            exporters=ExporterType.HTTP.value,
            http=HTTPConfig(endpoint="https://collector.test"),
        ),
    )
    assert failed_provider.processors == []

    telemetry._initialized = True
    telemetry._tracer_provider = FakeProvider(fail_shutdown=True)  # type: ignore[assignment]
    telemetry.shutdown_telemetry()
    assert telemetry._initialized is False


def test_tempo_and_langfuse_exporter_success_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOTLPSpanExporter:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeBatchSpanProcessor:
        def __init__(self, exporter: Any, **kwargs: Any) -> None:
            self.exporter = exporter
            self.kwargs = kwargs
            self.shutdown_called = False

        def shutdown(self) -> None:
            self.shutdown_called = True

    fake_export_module = types.ModuleType("fake_trace_exporter")
    fake_export_module.OTLPSpanExporter = FakeOTLPSpanExporter
    fake_trace_export = types.ModuleType("fake_sdk_trace_export")
    fake_trace_export.BatchSpanProcessor = FakeBatchSpanProcessor
    fake_grpc = types.ModuleType("grpc")
    fake_grpc.Compression = types.SimpleNamespace(Gzip="gzip")

    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
        fake_export_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        fake_export_module,
    )
    monkeypatch.setitem(
        sys.modules, "opentelemetry.sdk.trace.export", fake_trace_export
    )
    monkeypatch.setitem(sys.modules, "grpc", fake_grpc)

    tempo_processor = tempo_exporter_module.create_tempo_exporter(
        endpoint="tempo:4317",
        compression="gzip",
    )
    assert tempo_processor is not None
    tempo_processor.shutdown()

    langfuse_processor = langfuse_exporter_module.create_langfuse_exporter(
        endpoint="https://langfuse.test",
        public_key="public",
        secret_key="secret",
        environment="ci",
    )
    assert langfuse_processor is not None
    assert (
        langfuse_processor.exporter.kwargs["headers"]["X-Langfuse-Environment"] == "ci"
    )
    langfuse_processor.shutdown()


class NumberInput(ToolInput):
    value: int


class NumberOutput(ToolOutput):
    doubled: int


def test_tool_base_async_validation_display_and_registry_wrapper() -> None:
    original_registry = tool_base.get_tool_registry().copy()
    clear_registry()
    try:
        assert tool_base.get_tool_registry() == {}

        def display(
            _outputs: dict[str, Any], _console: Console, _indent: str = ""
        ) -> None:
            return None

        def raw_tool(inputs: NumberInput) -> dict[str, int]:
            return {"doubled": inputs.value * 2}

        raw_tool._display_function = display  # type: ignore[attr-defined]
        tool = Tool(
            raw_tool,
            ToolSpec(
                name="raw",
                version="1",
                description="raw tool",
                input_model=NumberInput,
                output_model=NumberOutput,
                tags=["math"],
            ),
        )
        assert tool.run({"value": 2}).doubled == 4
        assert tool.validate_output({"doubled": 8}).doubled == 8
        assert tool.get_display_function() is display

        @register_tool(
            name="async_double",
            description="Async double",
            input_model=NumberInput,
            output_model=NumberOutput,
            tags=["math", "async"],
        )
        async def async_double(inputs: NumberInput) -> dict[str, int]:
            return {"doubled": inputs.value * 2}

        registered = tool_base.get_tool("async_double")
        assert registered is not None
        assert asyncio.run(registered.arun({"value": 3})).doubled == 6
        assert asyncio.run(async_double(NumberInput(value=4))) == {"doubled": 8}

        wrapper = registry_module.ToolRegistry()
        assert wrapper.get("async_double") is registered
        assert "async_double" in wrapper.list_tools()
        assert wrapper.list_by_tag("async") == ["async_double"]
        assert wrapper.get_json_schemas()["async_double"]["name"] == "async_double"
        wrapper.register_display("async_double", display)
        assert registry_module.get_tool_registry() is registry_module._registry
    finally:
        clear_registry()
        tool_base.get_tool_registry().update(original_registry)


def test_image_edit_tool_outputs_display_and_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.png"
    Image.new("RGB", (2, 2), "red").save(source)

    def fake_edit(**_kwargs: Any) -> dict[str, Any]:
        return {
            "edited_image": Image.new("RGB", (3, 4), "green"),
            "rescaled_input": Image.new("RGB", (2, 2), "blue"),
            "image_size": (3, 4),
            "execution_time": 1.5,
        }

    monkeypatch.setattr(image_edit_module, "edit_image_with_comfyui", fake_edit)
    result = image_edit_module.image_edit_tool(
        image_edit_module.ImageEditInput(
            image_path=str(source),
            prompt="make green",
            negative_prompt="blur",
            return_rescaled_input=True,
            server_url="http://server",
        )
    )
    assert Path(result.edited_image_path).exists()
    assert Path(result.rescaled_input_path or "").exists()
    assert (result.image_width, result.image_height) == (3, 4)
    assert result.execution_time == 1.5

    rendered = StringIO()
    image_edit_module._display_image_edit_results(
        result.model_dump(),
        Console(file=rendered, force_terminal=False),
    )
    assert "Image Edit Complete" in rendered.getvalue()

    monkeypatch.setattr(
        image_edit_module,
        "edit_image_with_comfyui",
        lambda **_kwargs: {
            "edited_image": None,
            "image_size": (0, 0),
            "execution_time": 0.0,
        },
    )
    with pytest.raises(ValueError, match="No edited image"):
        image_edit_module.image_edit_tool(
            image_edit_module.ImageEditInput(image_path=str(source), prompt="bad")
        )
