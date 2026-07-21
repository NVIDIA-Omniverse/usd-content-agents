# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for telemetry decorator attribute extraction."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.trace import StatusCode

import world_understanding.telemetry.decorators as decorators_module
from world_understanding.telemetry.attributes import GenAIAttributes, MAAttributes


class RecordingSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.status_codes: list[StatusCode] = []
        self.exceptions: list[Exception] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: Any) -> None:
        self.status_codes.append(status.status_code)

    def record_exception(self, exception: Exception) -> None:
        self.exceptions.append(exception)


class RecordingSpanContext:
    def __init__(self, span: RecordingSpan) -> None:
        self.span = span

    def __enter__(self) -> RecordingSpan:
        return self.span

    def __exit__(self, *_args: object) -> None:
        return None


class RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[RecordingSpan] = []

    def start_as_current_span(self, name: str) -> RecordingSpanContext:
        span = RecordingSpan(name)
        self.spans.append(span)
        return RecordingSpanContext(span)


@pytest.fixture
def recording_tracer(monkeypatch: pytest.MonkeyPatch) -> RecordingTracer:
    tracer = RecordingTracer()
    monkeypatch.setattr(
        decorators_module.trace,
        "get_tracer",
        lambda _name: tracer,
    )
    return tracer


@pytest.mark.asyncio
async def test_traced_async_captures_attributes_input_and_output(
    recording_tracer: RecordingTracer,
) -> None:
    @decorators_module.traced(
        "async.capture",
        span_type="task",
        capture_input=True,
        capture_output=True,
        attributes={"custom.key": "custom.value", "custom.payload": {"answer": 42}},
    )
    async def add_values(left: int, *, right: int) -> dict[str, int]:
        return {"total": left + right}

    assert await add_values(2, right=3) == {"total": 5}

    span = recording_tracer.spans[0]
    assert span.name == "async.capture"
    assert span.attributes["observation.type"] == "task"
    assert span.attributes["custom.key"] == "custom.value"
    assert span.attributes["custom.payload"] == "{'answer': 42}"
    assert span.attributes["input.args"] == "(2,)"
    assert span.attributes["input.kwargs"] == "{'right': 3}"
    assert span.attributes["output"] == "{'total': 5}"
    assert span.status_codes == [StatusCode.OK]


@pytest.mark.asyncio
async def test_traced_llm_async_extracts_request_attributes(
    recording_tracer: RecordingTracer,
) -> None:
    @decorators_module.traced_llm(system="nim", operation="completion")
    async def complete(**_kwargs: Any) -> str:
        return "done"

    assert (
        await complete(
            model_id="llama",
            temperature=0.25,
            max_completion_tokens=128,
        )
        == "done"
    )

    attrs = recording_tracer.spans[0].attributes
    assert attrs["observation.type"] == "generation"
    assert attrs[GenAIAttributes.SYSTEM] == "nim"
    assert attrs[GenAIAttributes.OPERATION_NAME] == "completion"
    assert attrs[GenAIAttributes.REQUEST_MODEL] == "llama"
    assert attrs[GenAIAttributes.REQUEST_TEMPERATURE] == 0.25
    assert attrs[GenAIAttributes.REQUEST_MAX_TOKENS] == 128


def test_traced_vlm_extracts_model_backend_and_image_pair_attributes(
    recording_tracer: RecordingTracer,
) -> None:
    class Client:
        _model_name = "self-model"
        backend_name = "local-backend"

    @decorators_module.traced_vlm(system="vlm-system", operation="caption")
    def caption(client: Client, **_kwargs: Any) -> str:
        return "caption"

    assert (
        caption(
            Client(),
            model="kw-model",
            temperature=0.5,
            max_tokens=64,
            image_caption_pairs=[("image-a", "caption-a"), ("image-b", "caption-b")],
        )
        == "caption"
    )

    attrs = recording_tracer.spans[0].attributes
    assert attrs[GenAIAttributes.SYSTEM] == "vlm-system"
    assert attrs[GenAIAttributes.OPERATION_NAME] == "caption"
    assert attrs[GenAIAttributes.REQUEST_MODEL] == "self-model"
    assert attrs[GenAIAttributes.REQUEST_TEMPERATURE] == 0.5
    assert attrs[GenAIAttributes.REQUEST_MAX_TOKENS] == 64
    assert attrs[MAAttributes.VLM_IMAGE_COUNT] == 2
    assert attrs[MAAttributes.VLM_BACKEND] == "local-backend"
