# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for shared agentic workflow primitives."""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from world_understanding.agentic import base_pipeline_executor as executor_module
from world_understanding.agentic import session as session_module
from world_understanding.agentic import tasks as tasks_module
from world_understanding.agentic.base import BaseAgent
from world_understanding.agentic.base_pipeline_executor import (
    BasePipelineExecutor,
    PathEncoder,
)
from world_understanding.agentic.events import (
    CLIEventListener,
    CollectingEventListener,
    EventListener,
    LoggerAsListener,
    NoOpEventListener,
    create_default_listener,
    get_listener,
)
from world_understanding.agentic.session import SessionManager
from world_understanding.agentic.tasks import (
    AgenticLoopTask,
    CallableTask,
    RouterTask,
    Task,
    ToolTask,
)
from world_understanding.utils.credentials import InlineSecretError
from world_understanding.utils.object_store import InMemoryObjectStore
from world_understanding.utils.result_projection import project_result_metadata


def _assert_production_traceback_locals_exclude(
    error: BaseException, sentinel: str
) -> None:
    traceback_frame = error.__traceback__
    production_frames = 0
    while traceback_frame is not None:
        frame = traceback_frame.tb_frame
        if Path(frame.f_code.co_filename).resolve() != Path(__file__).resolve():
            production_frames += 1
            assert sentinel not in repr(frame.f_locals)
        traceback_frame = traceback_frame.tb_next
    assert production_frames > 0


def _traceback_locals_for(
    error: BaseException,
    function_name: str,
) -> list[dict[str, Any]]:
    """Copy locals retained by named frames on an exception traceback."""
    frames: list[dict[str, Any]] = []
    cursor = error.__traceback__
    while cursor is not None:
        if cursor.tb_frame.f_code.co_name == function_name:
            frames.append(dict(cursor.tb_frame.f_locals))
        cursor = cursor.tb_next
    return frames


class RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.calls.append(("info", message))

    def debug(self, message: str) -> None:
        self.calls.append(("debug", message))

    def warning(self, message: str) -> None:
        self.calls.append(("warning", message))

    def error(self, message: str) -> None:
        self.calls.append(("error", message))


class RecordingConsole:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def print(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class FakeAgent:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def arun(
        self,
        task: str,
        context: dict[str, Any],
        object_store: InMemoryObjectStore | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {"task": task, "context": dict(context), "object_store": object_store}
        )
        response = self.responses.pop(0) if self.responses else {}
        context.update(response)
        return context


class FakeInput:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class DumpableOutput:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self.payload


class RecordingSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.exceptions: list[Exception] = []
        self.statuses: list[Any] = []

    def __enter__(self) -> RecordingSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def record_exception(self, error: Exception) -> None:
        self.exceptions.append(error)

    def set_status(self, status: Any) -> None:
        self.statuses.append(status)


class RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[RecordingSpan] = []

    def start_as_current_span(self, name: str) -> RecordingSpan:
        span = RecordingSpan(name)
        self.spans.append(span)
        return span


class FakeTool:
    class Spec:
        input_model = FakeInput

    def __init__(self, output: Any, error: Exception | None = None) -> None:
        self.spec = self.Spec()
        self.output = output
        self.error = error
        self.inputs: list[FakeInput] = []

    async def arun(self, input_obj: FakeInput) -> Any:
        self.inputs.append(input_obj)
        if self.error is not None:
            raise self.error
        return self.output


class ConcreteExecutor(BasePipelineExecutor):
    def __init__(
        self,
        fail_step: str | None = None,
        failure_message: str = "step failed",
    ) -> None:
        self.fail_step = fail_step
        self.failure_message = failure_message
        self.executed: list[str] = []

    def _execute_step(
        self,
        step_name: str,
        context: dict[str, Any],
        object_store: InMemoryObjectStore | None,
    ) -> dict[str, Any]:
        self.executed.append(step_name)
        if step_name == self.fail_step:
            raise ValueError(self.failure_message)
        return {"step": step_name, "working_dir": context["working_dir"]}

    def _get_step_list_key(self) -> str:
        return "steps"

    def _get_required_context_keys(self) -> list[str]:
        return ["steps", "working_dir"]

    def _get_state_file(self, context: dict[str, Any]) -> Path:
        return Path(context["working_dir"]) / ".pipeline_state.json"


class ConcreteAgent(BaseAgent):
    def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        object_store: InMemoryObjectStore | None = None,
    ) -> dict[str, Any]:
        BaseAgent.run(self, task, context, object_store)
        return {"task": task, "context": context or {}, "object_store": object_store}


def test_base_agent_async_delegates_to_sync_run() -> None:
    store = InMemoryObjectStore()
    agent = ConcreteAgent(name="concrete", description="test")

    result = asyncio.run(agent.arun("task", {"value": 1}, store))

    assert result == {"task": "task", "context": {"value": 1}, "object_store": store}


def test_protocol_and_noop_listener_methods_are_callable() -> None:
    EventListener.info(object(), "info")
    EventListener.debug(object(), "debug")
    EventListener.warning(object(), "warning")
    EventListener.error(object(), "error")
    EventListener.event(object(), "event", {"ok": True})

    listener = NoOpEventListener()
    listener.info("info", extra=True)
    listener.debug("debug", extra=True)
    listener.warning("warning", extra=True)
    listener.error("error", extra=True)
    listener.event("event", {"ok": True}, extra=True)


def test_cli_listener_logs_and_renders_supported_events() -> None:
    logger = RecordingLogger()
    console = RecordingConsole()
    listener = CLIEventListener(logger=logger, console=console, show_events=True)

    listener.info("one")
    listener.debug("two")
    listener.warning("three")
    listener.error("four")

    listener.event("step.started", {"step_name": "extract"})
    listener.event("step.completed", {"step_name": "extract"})
    listener.event("step.failed", {"step_name": "extract", "error": "boom"})
    listener.event(
        "pipeline.overview",
        {"steps": ["extract", "render"], "completed_steps": ["extract"]},
    )
    listener.event("pipeline.success", {})
    listener.event("pipeline.failed", {"failed_step": "render", "error": "bad usd"})
    listener.event(
        "pipeline.config.display",
        {
            "config": "config.yaml",
            "skip_steps": ["judge"],
            "only_steps": ["render"],
            "resume": True,
            "dry_run": True,
            "clean": True,
        },
    )
    listener.event(
        "task.progress",
        {"current": 2, "total": 4, "percentage": 50.0, "task_name": "render"},
    )

    assert logger.calls == [
        ("info", "one"),
        ("debug", "two"),
        ("warning", "three"),
        ("error", "four"),
    ]
    assert len(console.calls) >= 9
    assert console.calls[-1][1] == {"end": "\r"}


def test_cli_listener_silent_and_show_events_without_console() -> None:
    silent = CLIEventListener(logger=RecordingLogger())
    silent.event("step.started", {"step_name": "ignored"})

    console_only = CLIEventListener(
        logger=RecordingLogger(), console=RecordingConsole()
    )
    console_only.event("unknown.event", {})

    visible_without_console = CLIEventListener(
        logger=RecordingLogger(), show_events=True
    )
    visible_without_console.event("task.progress", {"current": 1})


def test_collecting_listener_filters_logs_and_events() -> None:
    listener = CollectingEventListener()
    listener.info("info", request_id="1")
    listener.debug("debug")
    listener.warning("warning")
    listener.error("error")
    listener.event("task.progress", {"current": 1})
    listener.event("task.done", {"ok": True})

    assert [log["level"] for log in listener.get_logs()] == [
        "info",
        "debug",
        "warning",
        "error",
    ]
    assert listener.get_logs("warning")[0]["message"] == "warning"
    assert [event["type"] for event in listener.get_events()] == [
        "task.progress",
        "task.done",
    ]
    assert listener.get_events("task.done")[0]["data"] == {"ok": True}


def test_default_and_logger_listeners(caplog: pytest.LogCaptureFixture) -> None:
    custom_logger = logging.getLogger("agentic-test-listener")
    listener = create_default_listener(logger=custom_logger, show_events=True)
    assert isinstance(listener, CLIEventListener)

    default_listener = create_default_listener()
    assert isinstance(default_listener, CLIEventListener)

    provided = object()
    assert get_listener({"event_listener": provided}) is provided

    fallback = get_listener({}, logger_name="agentic-test-fallback")
    assert isinstance(fallback, LoggerAsListener)
    with caplog.at_level(logging.DEBUG, logger="agentic-test-fallback"):
        fallback.info("info")
        fallback.debug("debug")
        fallback.warning("warning")
        fallback.error("error")
        fallback.event("kind", {"payload": True})

    assert "Event: kind - {'payload': True}" in caplog.text


def test_session_manager_creates_loads_updates_and_lists_sessions(
    tmp_path: Path,
) -> None:
    session = SessionManager.create(
        tmp_path,
        project_name="project-a",
        session_id="session-a",
        prefix="run-",
        metadata={"custom": "value"},
    )

    assert session.session_dir == (tmp_path / "run-session-a").resolve()
    assert session.metadata["custom"] == "value"
    assert session.get_subdir("nested/path").is_dir()
    assert session.get_subdir("no-create", create=False) == (
        session.session_dir / "no-create"
    )
    assert not (session.session_dir / "no-create").exists()
    assert session.get_file("output/result.json") == (
        session.session_dir / "output/result.json"
    )

    session.update_metadata(status="complete")
    loaded = SessionManager.from_id("session-a", tmp_path, prefix="run-")
    assert loaded.project_name == "project-a"
    assert loaded.metadata["status"] == "complete"
    assert repr(loaded).startswith("SessionManager(id=session-a")
    assert str(loaded).startswith("Session session-a (project-a)")

    older = tmp_path / "run-session-b"
    older.mkdir()
    (older / ".metadata.json").write_text(
        json.dumps({"project_name": "older", "created_at": "2000-01-01"}),
        encoding="utf-8",
    )
    bad = tmp_path / "run-session-bad"
    bad.mkdir()
    (bad / ".metadata.json").write_text("{not-json", encoding="utf-8")
    (tmp_path / "run-file").write_text("not a dir", encoding="utf-8")
    (tmp_path / "other-session").mkdir()

    sessions = SessionManager.list_sessions(tmp_path, prefix="run-")
    assert [item["session_id"] for item in sessions] == [
        "session-a",
        "session-b",
        "session-bad",
    ]
    assert sessions[-1]["project_name"] == "unknown"
    assert SessionManager.list_sessions(tmp_path / "missing") == []


def test_session_manager_generates_ids_and_handles_metadata_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        "world_understanding.agentic.session.uuid.uuid4",
        lambda: "generated-id",
    )
    generated = SessionManager.create(tmp_path, project_name=None)
    assert generated.session_id == "generated-id"
    assert generated.project_name == "unknown_project"

    bad_dir = tmp_path / ".bad"
    bad_dir.mkdir()
    (bad_dir / ".metadata.json").write_text("{bad-json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        loaded = SessionManager.from_id("bad", tmp_path)
    assert loaded.project_name == "unknown_project"
    assert "Failed to load session metadata" in caplog.text

    with pytest.raises(FileNotFoundError):
        SessionManager.from_id("missing", tmp_path)

    unwritable = SessionManager("x", tmp_path / "not-a-directory" / "session")
    with caplog.at_level(logging.WARNING):
        unwritable.save_metadata()
    assert "Failed to save session metadata" in caplog.text

    secret = "session-save-path-secret-713"
    signed_parent = tmp_path / f"run?X-Amz-Signature={secret}"
    signed_parent.write_text("not a directory", encoding="utf-8")
    signed_session = SessionManager("safe", signed_parent / "session")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        signed_session.save_metadata()
    assert secret not in caplog.text
    assert "<redacted>" in caplog.text


@pytest.mark.parametrize("session_id", ["", ".", "..", "nested/id", r"nested\id"])
def test_session_ids_reject_non_filename_components_across_entrypoints(
    tmp_path: Path,
    session_id: str,
) -> None:
    calls = (
        lambda: SessionManager(session_id, tmp_path / "direct"),
        lambda: SessionManager.create(tmp_path, session_id=session_id),
        lambda: SessionManager.from_id(session_id, tmp_path),
    )

    for call in calls:
        with pytest.raises(
            ValueError,
            match="Session ID must be a single filename component",
        ):
            call()

    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("prefix", ["..", "nested/", "nested\\"])
def test_session_prefixes_reject_non_filename_components_before_io(
    tmp_path: Path,
    prefix: str,
) -> None:
    calls = (
        lambda: SessionManager.create(tmp_path, session_id="safe", prefix=prefix),
        lambda: SessionManager.from_id("safe", tmp_path, prefix=prefix),
        lambda: SessionManager.list_sessions(tmp_path, prefix=prefix),
    )

    for call in calls:
        with pytest.raises(
            ValueError,
            match="Session prefix must be a single filename component",
        ):
            call()

    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "session_id",
    [
        "safe?X-Amz-Signature=session-id-secret-713",
        "safe#access_token=session-id-secret-713",
    ],
)
def test_session_identifiers_reject_relative_bearer_values_before_persistence(
    tmp_path: Path,
    session_id: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "session-id-secret-713"

    with caplog.at_level(logging.DEBUG), pytest.raises(InlineSecretError) as exc_info:
        SessionManager.create(tmp_path, session_id=session_id)

    observable = caplog.text + "".join(traceback.format_exception(exc_info.value))
    assert secret not in observable
    assert not any(tmp_path.iterdir())

    executor = ConcreteExecutor()
    with pytest.raises(InlineSecretError) as pipeline_error:
        executor.run(
            {
                "steps": ["one"],
                "working_dir": tmp_path / "pipeline",
                "session_id": session_id,
            }
        )
    assert secret not in "".join(traceback.format_exception(pipeline_error.value))
    run_frames = _traceback_locals_for(pipeline_error.value, "run")
    assert run_frames
    assert secret not in repr(run_frames)
    assert not _traceback_locals_for(pipeline_error.value, "_run_impl")
    assert not (tmp_path / "pipeline").exists()


def test_session_listing_skips_stat_failures_without_exposing_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "session-list-stat-secret-713"
    inaccessible = tmp_path / f".user:{secret}@host"
    inaccessible.mkdir()
    original_is_dir = Path.is_dir

    def fail_secret_entry(path: Path) -> bool:
        if path == inaccessible:
            raise PermissionError(13, "denied", str(path))
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", fail_secret_entry)
    with caplog.at_level(logging.WARNING):
        sessions = SessionManager.list_sessions(tmp_path)

    assert sessions == []
    assert secret not in caplog.text
    assert "<redacted>" in caplog.text


@pytest.mark.parametrize(
    "error_type",
    [PermissionError, FileNotFoundError],
)
def test_session_listing_severs_sensitive_iterdir_exception_graph(
    error_type: type[OSError],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "session-list-context-secret-713"
    base_path = tmp_path / f"user:{secret}@sessions.example.test"
    base_path.mkdir()
    rejected_error = error_type(
        errno.EACCES,
        f"directory provider echoed {secret}",
        str(base_path),
    )

    def reject_listing(_path: Path) -> Any:
        raise rejected_error

    monkeypatch.setattr(Path, "iterdir", reject_listing)

    with pytest.raises(error_type) as exc_info:
        SessionManager.list_sessions(base_path)

    assert exc_info.value.errno == errno.EACCES
    assert exc_info.value.filename == "<redacted>"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in repr(
        (exc_info.value, exc_info.value.__cause__, exc_info.value.__context__)
    )
    _assert_production_traceback_locals_exclude(exc_info.value, secret)


def test_session_manager_projects_runtime_secrets_from_durable_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "session-metadata-secret-713"
    session = SessionManager.create(
        tmp_path,
        project_name=f"https://user:{secret}@project.example.test/name",
        session_id="safe-session",
        metadata={
            "api_key": secret,
            "artifact": Path(
                f"https://assets.example.test/model.usd?X-Amz-Signature={secret}"
            ),
            "safe": "retained",
        },
    )

    with caplog.at_level(logging.DEBUG):
        session.save_metadata()
        rendered = f"{session!r}\n{session}"

    metadata_file = session.session_dir / ".metadata.json"
    persisted_text = metadata_file.read_text(encoding="utf-8")
    persisted = json.loads(persisted_text)
    assert persisted["api_key"] == "<redacted>"
    assert persisted["artifact"] == "<redacted>"
    assert persisted["project_name"] == "<redacted>"
    assert persisted["safe"] == "retained"
    assert session.metadata["api_key"] == secret

    loaded = SessionManager.from_id("safe-session", tmp_path)
    listed = SessionManager.list_sessions(tmp_path)
    observable = f"{persisted_text}\n{rendered}\n{caplog.text}\n{loaded!r}\n{listed!r}"
    assert secret not in observable
    assert loaded.metadata["api_key"] == "<redacted>"
    assert listed[0]["metadata"]["artifact"] == "<redacted>"


def test_session_manager_projects_benign_nested_metadata_to_json_primitives(
    tmp_path: Path,
) -> None:
    session = SessionManager.create(
        tmp_path,
        session_id="json-safe",
        metadata={
            "features": {"zeta", 2, ("nested", 1)},
            "frozen": frozenset({"beta", 1}),
            "nested": {
                "tuple": (True, Path("artifacts/result.usd")),
                "list": [None, 3.5],
            },
        },
    )

    session.save_metadata()

    persisted = json.loads(
        (session.session_dir / ".metadata.json").read_text(encoding="utf-8")
    )
    assert persisted["features"] == [2, "zeta", ["nested", 1]]
    assert persisted["frozen"] == [1, "beta"]
    assert persisted["nested"] == {
        "tuple": [True, "artifacts/result.usd"],
        "list": [None, 3.5],
    }


def test_session_metadata_serialization_failure_preserves_existing_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = SessionManager.create(tmp_path, session_id="serialize-failure")
    metadata_file = session.session_dir / ".metadata.json"
    previous_payload = b'{"status": "last-valid"}\n'
    metadata_file.write_bytes(previous_payload)

    def fail_serialization(*_args: object, **_kwargs: object) -> str:
        raise TypeError("serialization-value-sentinel")

    monkeypatch.setattr(session_module.json, "dumps", fail_serialization)
    session.metadata["status"] = "new"
    with caplog.at_level(logging.WARNING):
        session.save_metadata()

    assert metadata_file.read_bytes() == previous_payload
    assert list(session.session_dir.glob(f".{metadata_file.name}.*.tmp")) == []
    assert "Failed to save session metadata" in caplog.text
    assert "serialization-value-sentinel" not in caplog.text


def test_session_metadata_projection_failure_preserves_existing_document(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = SessionManager.create(tmp_path, session_id="projection-failure")
    metadata_file = session.session_dir / ".metadata.json"
    previous_payload = b'{"status": "last-valid"}\n'
    metadata_file.write_bytes(previous_payload)

    session.metadata["runtime_only"] = object()
    with caplog.at_level(logging.WARNING):
        session.save_metadata()

    assert metadata_file.read_bytes() == previous_payload
    assert list(session.session_dir.glob(f".{metadata_file.name}.*.tmp")) == []
    assert "Failed to save session metadata" in caplog.text


def test_session_metadata_replace_failure_preserves_existing_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = SessionManager.create(tmp_path, session_id="replace-failure")
    metadata_file = session.session_dir / ".metadata.json"
    previous_payload = b'{"status": "last-valid"}\n'
    metadata_file.write_bytes(previous_payload)

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("replacement-value-sentinel")

    monkeypatch.setattr(session_module.os, "replace", fail_replace)
    session.metadata["status"] = "new"
    with caplog.at_level(logging.WARNING):
        session.save_metadata()

    assert metadata_file.read_bytes() == previous_payload
    assert list(session.session_dir.glob(f".{metadata_file.name}.*.tmp")) == []
    assert "Failed to save session metadata" in caplog.text
    assert "replacement-value-sentinel" not in caplog.text


class SimpleTask(Task):
    def run(
        self, context: dict[str, Any], object_store: InMemoryObjectStore | None = None
    ) -> dict[str, Any]:
        context["ran"] = object_store is not None
        return context


def test_task_arun_and_callable_task() -> None:
    Task.run(object(), {}, None)

    store = InMemoryObjectStore()
    result = asyncio.run(SimpleTask().arun({}, store))
    assert result == {"ran": True}

    def add_value(
        context: dict[str, Any], object_store: InMemoryObjectStore | None
    ) -> dict[str, Any]:
        context["name"] = object_store.get("name") if object_store else "missing"
        return context

    store.set("name", "callable")
    assert CallableTask(add_value, name="adder").run({}, store) == {"name": "callable"}


def test_agentic_loop_task_completion_paths() -> None:
    early = AgenticLoopTask(FakeAgent([{"avg_confidence": 0.9}]))
    assert early.run({})["completion_reason"] == "confidence_threshold_met"

    completed = asyncio.run(
        AgenticLoopTask(FakeAgent([{"completed": True}])).arun(
            {}, InMemoryObjectStore()
        )
    )
    assert completed["completed"] is True
    assert "completion_reason" not in completed

    no_refinement = asyncio.run(
        AgenticLoopTask(FakeAgent([{"needs_refinement": False}])).arun(
            {}, InMemoryObjectStore()
        )
    )
    assert no_refinement["completion_reason"] == "no_refinement_requested"

    maxed = asyncio.run(
        AgenticLoopTask(
            FakeAgent(
                [
                    {"avg_confidence": 0.1, "needs_refinement": True},
                    {"avg_confidence": 0.2, "needs_refinement": True},
                ]
            ),
            max_iterations=2,
        ).arun({}, InMemoryObjectStore())
    )
    assert maxed["completion_reason"] == "max_iterations_reached"
    assert maxed["final_iteration"] == 2


def test_tool_task_resolves_inputs_runs_and_stores_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = FakeTool(DumpableOutput({"answer": 42}))
    monkeypatch.setattr(tasks_module, "get_tool_registry", lambda: {"fake": tool})
    task = ToolTask(
        "fake",
        inputs={
            "static": "value",
            "number": 7,
            "nested": "${source.metadata.count}",
        },
        input_mapping={
            "direct": "source.name",
            "missing_nested": "source.missing.key",
            "referenced": "${settings.mode}",
            "plain": "flat_key",
            "literal": 123,
        },
        output_key="result",
        name="Fake Tool",
    )
    store = InMemoryObjectStore()
    context = {
        "source": {"name": "dataset", "metadata": {"count": 3}},
        "settings": {"mode": "fast"},
        "flat_key": "flat",
    }

    result = task.run(context, store)

    assert result["result"] == {"answer": 42}
    assert result["fake_success"] is True
    assert store.get("result") == {"answer": 42}
    assert tool.inputs[0].kwargs == {
        "static": "value",
        "number": 7,
        "nested": 3,
        "direct": "dataset",
        "missing_nested": None,
        "referenced": "fast",
        "plain": "flat",
        "literal": 123,
    }


def test_tool_task_reference_errors_missing_tool_and_non_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = ToolTask("missing")
    assert task._resolve_value("${}", {}) is None
    assert task._resolve_value("${.bad}", {}) is None
    assert task._resolve_value("${bad.}", {}) is None
    assert task._resolve_value("${bad..path}", {}) is None
    assert task._resolve_value("${missing}", {}) is None
    assert task._resolve_value("plain", {}) == "plain"
    assert task._resolve_value(3, {}) == 3

    missing = asyncio.run(task.arun({}))
    assert missing["error"] == "Tool 'missing' not found"
    assert missing["failed_task"] == "missing"

    tool = FakeTool({"raw": "dict"})
    monkeypatch.setattr(tasks_module, "get_tool_registry", lambda: {"raw": tool})
    raw_task = ToolTask("raw")
    assert asyncio.run(raw_task.arun({}))["raw_result"] == {"raw": "dict"}

    failing = FakeTool({}, error=RuntimeError("tool exploded"))
    monkeypatch.setattr(tasks_module, "get_tool_registry", lambda: {"bad": failing})
    failed = asyncio.run(ToolTask("bad", name="Bad Tool").arun({}))
    assert failed["bad_success"] is False
    assert failed["bad_error"] == "Tool execution failed"
    assert failed["error"] == "Tool execution failed"
    assert failed["failed_task"] == "Bad Tool"


@pytest.mark.parametrize("run_sync", [False, True])
def test_tool_task_keeps_runtime_inputs_out_of_failure_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    run_sync: bool,
) -> None:
    password = "TOOLTASK_INLINE_PASSWORD"
    signature = "TOOLTASK_SIGNED_QUERY_TOKEN"
    api_key = "nvapi-TOOLTASK_INLINE_API_KEY"
    credentialed_source = (
        f"https://runtime-user:{password}@example.test/input.pdf"
        f"?X-Amz-Signature={signature}"
    )
    runtime_inputs = {
        "source": credentialed_source,
        "model": {"api_key": api_key},
    }
    failing = FakeTool(
        {},
        error=RuntimeError(
            f"request failed for {credentialed_source} with api_key={api_key}"
        ),
    )
    monkeypatch.setattr(tasks_module, "get_tool_registry", lambda: {"bad": failing})
    task = ToolTask(
        "bad",
        inputs=runtime_inputs,
        name=f"Bad Tool {credentialed_source}",
    )

    with caplog.at_level(logging.DEBUG, logger=tasks_module.__name__):
        if run_sync:
            result = task.run({})
        else:
            result = asyncio.run(task.arun({}))

    assert failing.inputs[0].kwargs == runtime_inputs
    assert result["bad_error"] == "Tool execution failed"
    assert result["error"] == "Tool execution failed"
    assert result["failed_task"] == task.name
    assert "Resolved 2 tool input field(s)" in caplog.text

    published_result = project_result_metadata(result)
    assert published_result["failed_task"] == "<redacted>"
    observable = f"{caplog.text}\n{published_result!r}"
    for secret in (password, signature, api_key):
        assert secret not in observable


@pytest.mark.parametrize("fails", [False, True])
def test_tool_task_projects_credentialed_dispatch_names_from_context_keys(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fails: bool,
) -> None:
    secret = "tool-dispatch-name-secret-713"
    raw_tool_name = f"https://runtime-user:{secret}@tools.example.test/run"
    tool = FakeTool(
        {"answer": 42},
        error=RuntimeError("backend failed") if fails else None,
    )
    monkeypatch.setattr(
        tasks_module,
        "get_tool_registry",
        lambda: {raw_tool_name: tool},
    )

    with caplog.at_level(logging.INFO, logger=tasks_module.__name__):
        result = asyncio.run(ToolTask(raw_tool_name).arun({}))

    # Registry dispatch and live context keep the exact runtime identifier.
    # Public result projection is the credential boundary.
    assert len(tool.inputs) == 1
    assert result[f"{raw_tool_name}_success"] is not fails
    if fails:
        assert result[f"{raw_tool_name}_error"] == "Tool execution failed"
    else:
        assert result[f"{raw_tool_name}_result"] == {"answer": 42}

    published_result = project_result_metadata(result)
    assert published_result == {}
    observable = f"{published_result!r}\n{caplog.text}"
    assert raw_tool_name not in observable
    assert secret not in observable


def test_router_task_empty_context_and_explicit_tasks() -> None:
    empty = asyncio.run(RouterTask(FakeAgent([])).arun({}))
    assert empty == {"router_results": [], "tasks_completed": 0, "all_success": True}

    store = InMemoryObjectStore()
    agent = FakeAgent([{"success": True, "value": 1}, {"success": False, "value": 2}])
    task = RouterTask(
        agent,
        tasks=[
            {"description": "first", "image_path": "a.png"},
            {"target_color": [1, 2, 3]},
        ],
    )
    result = task.run({}, store)

    assert result["tasks_completed"] == 2
    assert result["all_success"] is False
    assert [call["task"] for call in agent.calls] == ["first", "Task 2"]
    assert agent.calls[0]["context"] == {"image_path": "a.png"}
    assert agent.calls[1]["context"] == {"target_color": [1, 2, 3]}
    assert store.get("task_1_result") == {
        "image_path": "a.png",
        "success": True,
        "value": 1,
    }
    assert store.get("task_2_result") == {
        "target_color": [1, 2, 3],
        "success": False,
        "value": 2,
    }

    context_task = RouterTask(FakeAgent([{"success": True}]))
    context_result = asyncio.run(
        context_task.arun({"router_tasks": [{"description": "from context"}]})
    )
    assert context_result["all_success"] is True


def test_base_pipeline_executor_success_resume_and_checkpoint(tmp_path: Path) -> None:
    executor = ConcreteExecutor()
    work_dir = tmp_path / "work"
    context = {
        "steps": ["prepare", "render"],
        "working_dir": work_dir,
        "project_name": "core",
        "session_id": "s1",
    }

    result = executor.run(context, InMemoryObjectStore())
    assert executor.executed == ["prepare", "render"]
    assert result["pipeline_state"] == "completed"
    assert result["pipeline_results"] == {
        "prepare": {"step": "prepare", "working_dir": work_dir},
        "render": {"step": "render", "working_dir": work_dir},
    }

    state_file = work_dir / ".pipeline_state.json"
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["completed_steps"] == ["prepare", "render"]
    assert saved["step_outputs"]["prepare"]["working_dir"] == str(work_dir)
    assert (
        executor._get_state_lock_file(state_file) == work_dir / ".pipeline_state.lock"
    )

    resume_state = {
        "session_id": "s1",
        "project_name": "core",
        "completed_steps": ["prepare"],
        "failed_steps": [],
        "step_outputs": {"prepare": {"cached": True}},
        "current_step": None,
    }
    state_file.write_text(json.dumps(resume_state), encoding="utf-8")
    resumed_executor = ConcreteExecutor()
    resumed = resumed_executor.run({**context, "resume": True})
    assert resumed_executor.executed == ["render"]
    assert resumed["pipeline_results"]["prepare"] == {"cached": True}


def test_base_pipeline_executor_public_failure_quarantines_secret_result(
    tmp_path: Path,
) -> None:
    secret = "pipeline-result-checkpoint-secret-713"

    class CredentialResultExecutor(ConcreteExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.checkpoint_attempts = 0

        def _execute_step(
            self,
            step_name: str,
            context: dict[str, Any],
            object_store: InMemoryObjectStore | None,
        ) -> dict[str, Any]:
            self.executed.append(step_name)
            return {"api_key": secret}

        def _save_checkpoint(
            self,
            pipeline_state: dict[str, Any],
            state_file: Path,
        ) -> None:
            self.checkpoint_attempts += 1
            super()._save_checkpoint(pipeline_state, state_file)

    executor = CredentialResultExecutor()
    working_dir = tmp_path / "credential-result"
    runtime_config = {"api_key": secret}

    with pytest.raises(InlineSecretError) as exc_info:
        executor.run(
            {
                "steps": ["predict"],
                "working_dir": working_dir,
                "runtime_config": runtime_config,
            }
        )

    assert executor.checkpoint_attempts == 1
    assert str(exc_info.value) == (
        "Pipeline checkpoint state contains inline credentials"
    )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    run_frames = _traceback_locals_for(exc_info.value, "run")
    assert run_frames
    assert secret not in repr(run_frames)
    assert not _traceback_locals_for(exc_info.value, "_run_impl")
    assert not _traceback_locals_for(
        exc_info.value,
        "_execute_step_with_tracing",
    )
    assert not _traceback_locals_for(exc_info.value, "_save_checkpoint")
    assert secret not in "".join(traceback.format_exception(exc_info.value))
    assert not (working_dir / ".pipeline_state.json").exists()


def test_base_pipeline_executor_public_failure_preserves_os_error_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "pipeline-public-os-error-secret-713"
    credentialed_path = tmp_path / f"run?X-Amz-Signature={secret}"
    rejected_error = PermissionError(
        errno.EACCES,
        "Unable to read pipeline checkpoint",
        str(credentialed_path),
    )
    executor = ConcreteExecutor()

    def reject_state_load(
        _context: dict[str, Any],
        _resume: bool = False,
    ) -> dict[str, Any]:
        raise rejected_error

    monkeypatch.setattr(executor, "_initialize_pipeline_state", reject_state_load)

    with pytest.raises(PermissionError) as exc_info:
        executor.run(
            {
                "steps": ["predict"],
                "working_dir": credentialed_path,
                "runtime_config": {"api_key": secret},
            }
        )

    assert exc_info.value.errno == errno.EACCES
    assert exc_info.value.strerror == "Unable to read pipeline checkpoint"
    assert exc_info.value.filename == "<redacted>"
    assert str(exc_info.value) == (
        "[Errno 13] Unable to read pipeline checkpoint: '<redacted>'"
    )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    run_frames = _traceback_locals_for(exc_info.value, "run")
    assert run_frames
    assert secret not in repr(run_frames)
    assert not _traceback_locals_for(exc_info.value, "_run_impl")


def test_base_pipeline_executor_public_failure_drops_opaque_exception_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "opaque-unclassified-pipeline-secret-713"
    executor = ConcreteExecutor()

    def reject_validation(_context: dict[str, Any]) -> None:
        raise TypeError(secret)

    monkeypatch.setattr(executor, "_validate_context", reject_validation)

    with pytest.raises(
        RuntimeError,
        match="^TypeError during pipeline execution$",
    ) as exc_info:
        executor.run(
            {
                "steps": ["predict"],
                "working_dir": tmp_path / "opaque-failure",
                "runtime_config": {"api_key": secret},
            }
        )

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in str(exc_info.value)
    run_frames = _traceback_locals_for(exc_info.value, "run")
    assert run_frames
    assert secret not in repr(run_frames)
    assert not _traceback_locals_for(exc_info.value, "_run_impl")

    def reject_with_unstructured_os_error(_context: dict[str, Any]) -> None:
        raise OSError(secret)

    monkeypatch.setattr(
        executor,
        "_validate_context",
        reject_with_unstructured_os_error,
    )
    with pytest.raises(
        OSError,
        match="^OSError during pipeline operation$",
    ) as os_exc_info:
        executor.run(
            {
                "steps": ["predict"],
                "working_dir": tmp_path / "opaque-os-failure",
                "runtime_config": {"api_key": secret},
            }
        )

    assert os_exc_info.value.__cause__ is None
    assert os_exc_info.value.__context__ is None
    assert secret not in str(os_exc_info.value)
    os_run_frames = _traceback_locals_for(os_exc_info.value, "run")
    assert os_run_frames
    assert secret not in repr(os_run_frames)
    assert not _traceback_locals_for(os_exc_info.value, "_run_impl")

    for opaque_error in (ValueError(secret), RuntimeError(secret)):

        def reject_with_documented_error(
            _context: dict[str, Any],
            *,
            rejected: Exception = opaque_error,
        ) -> None:
            raise rejected

        monkeypatch.setattr(
            executor,
            "_validate_context",
            reject_with_documented_error,
        )
        with pytest.raises(
            RuntimeError,
            match=(f"^{type(opaque_error).__name__} during pipeline execution$"),
        ) as documented_exc_info:
            executor.run(
                {
                    "steps": ["predict"],
                    "working_dir": tmp_path / "opaque-documented-failure",
                    "runtime_config": {"api_key": secret},
                }
            )

        assert documented_exc_info.value.__cause__ is None
        assert documented_exc_info.value.__context__ is None
        assert secret not in str(documented_exc_info.value)
        documented_run_frames = _traceback_locals_for(
            documented_exc_info.value,
            "run",
        )
        assert documented_run_frames
        assert secret not in repr(documented_run_frames)
        assert not _traceback_locals_for(
            documented_exc_info.value,
            "_run_impl",
        )


def test_base_pipeline_executor_preserves_raw_success_identity(tmp_path: Path) -> None:
    secret = "pipeline-runtime-success-secret-713"
    runtime_config = {"api_key": secret}
    step_result = {"api_key": secret, "opaque": object()}

    class RuntimeOnlyResultExecutor(ConcreteExecutor):
        def _execute_step(
            self,
            step_name: str,
            context: dict[str, Any],
            object_store: InMemoryObjectStore | None,
        ) -> dict[str, Any]:
            self.executed.append(step_name)
            return step_result

        def _save_checkpoint(
            self,
            pipeline_state: dict[str, Any],
            state_file: Path,
        ) -> None:
            # This test has no durable boundary: it verifies that the public
            # exception quarantine is transparent to a successful runtime call.
            return None

    executor = RuntimeOnlyResultExecutor()
    context = {
        "steps": ["predict"],
        "working_dir": tmp_path / "runtime-only",
        "runtime_config": runtime_config,
    }

    result = executor.run(context)

    assert result is context
    assert result["runtime_config"] is runtime_config
    assert result["pipeline_results"]["predict"] is step_result
    assert result["pipeline_results"]["predict"]["api_key"] == secret


def test_base_pipeline_executor_checkpoint_persists_benign_paths(
    tmp_path: Path,
) -> None:
    executor = ConcreteExecutor()
    state_file = tmp_path / ".pipeline_state.json"
    benign_path = tmp_path / "artifacts" / "model.usd"
    pipeline_state = {
        "completed_steps": ["prepare"],
        "failed_steps": [],
        "step_outputs": {"prepare": {"artifact_path": benign_path}},
        "current_step": None,
    }

    executor._save_checkpoint(pipeline_state, state_file)

    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["step_outputs"]["prepare"]["artifact_path"] == str(benign_path)


@pytest.mark.parametrize(
    ("secret_state", "secret"),
    [
        (
            {"step_outputs": {"prepare": {"api_key": "api-secret-token-713"}}},
            "api-secret-token-713",
        ),
        (
            {
                "step_outputs": {
                    "prepare": {
                        "artifact_path": (
                            "https://user:userinfo-secret-token-713@"
                            "assets.example.test/model.usd"
                        )
                    }
                }
            },
            "userinfo-secret-token-713",
        ),
        (
            {
                "step_outputs": {
                    "prepare": {
                        "artifact_path": (
                            "https://assets.example.test/model.usd?"
                            "X-Amz-Signature=signed-secret-token-713"
                        )
                    }
                }
            },
            "signed-secret-token-713",
        ),
        (
            {
                "step_outputs": {
                    "prepare": {
                        "artifact_path": Path(
                            "https://assets.example.test/model.usd?"
                            "X-Amz-Signature=normalized-path-secret-token-713"
                        )
                    }
                }
            },
            "normalized-path-secret-token-713",
        ),
    ],
    ids=["api-key", "userinfo", "signed-url", "normalized-path"],
)
def test_base_pipeline_executor_rejects_secret_checkpoint_and_resume_state(
    tmp_path: Path, secret_state: dict[str, Any], secret: str
) -> None:
    executor = ConcreteExecutor()
    state_file = tmp_path / ".pipeline_state.json"
    safe_state = {
        "completed_steps": [],
        "failed_steps": [],
        "step_outputs": {"prepare": {"artifact_path": "/safe/model.usd"}},
        "current_step": None,
    }
    executor._save_checkpoint(safe_state, state_file)
    safe_checkpoint = state_file.read_text(encoding="utf-8")

    with pytest.raises(InlineSecretError) as checkpoint_exc:
        executor._save_checkpoint(secret_state, state_file)

    assert state_file.read_text(encoding="utf-8") == safe_checkpoint
    assert secret not in str(checkpoint_exc.value)
    assert checkpoint_exc.value.__cause__ is None
    assert checkpoint_exc.value.__context__ is None
    checkpoint_frames = _traceback_locals_for(
        checkpoint_exc.value,
        "_save_checkpoint",
    )
    assert checkpoint_frames
    assert secret not in repr(checkpoint_frames)

    resume_state = {
        "completed_steps": [],
        "failed_steps": [],
        "step_outputs": {},
        "current_step": None,
        **secret_state,
    }
    state_file.write_text(
        json.dumps(resume_state, cls=PathEncoder),
        encoding="utf-8",
    )
    with pytest.raises(InlineSecretError) as resume_exc:
        executor._initialize_pipeline_state({"working_dir": tmp_path}, resume=True)

    assert secret not in str(resume_exc.value)
    assert resume_exc.value.__cause__ is None
    assert resume_exc.value.__context__ is None
    resume_frames = _traceback_locals_for(
        resume_exc.value,
        "_initialize_pipeline_state",
    )
    assert resume_frames
    assert secret not in repr(resume_frames)


def test_base_pipeline_executor_rejects_non_object_checkpoint_root(
    tmp_path: Path,
) -> None:
    secret = "checkpoint-root-secret-713"
    state_file = tmp_path / ".pipeline_state.json"
    state_file.write_text(
        json.dumps([{"api_key": secret}]),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="^Pipeline checkpoint root must be a JSON object$",
    ) as exc_info:
        ConcreteExecutor()._initialize_pipeline_state(
            {"working_dir": tmp_path},
            resume=True,
        )

    assert secret not in str(exc_info.value)
    traceback_cursor = exc_info.value.__traceback__
    executor_locals: dict[str, Any] | None = None
    while traceback_cursor is not None:
        if traceback_cursor.tb_frame.f_code.co_name == "_initialize_pipeline_state":
            executor_locals = traceback_cursor.tb_frame.f_locals
            break
        traceback_cursor = traceback_cursor.tb_next
    assert executor_locals is not None
    assert secret not in repr(executor_locals)


@pytest.mark.parametrize(
    "malformed_state",
    [
        None,
        [],
        {},
        {
            "completed_steps": "prepare",
            "failed_steps": [],
            "step_outputs": {},
        },
        {
            "completed_steps": [7],
            "failed_steps": [],
            "step_outputs": {},
        },
        {
            "completed_steps": [],
            "failed_steps": "render",
            "step_outputs": {},
        },
        {
            "completed_steps": [],
            "failed_steps": [7],
            "step_outputs": {},
        },
        {
            "completed_steps": [],
            "failed_steps": [],
            "step_outputs": [],
        },
        {
            "completed_steps": [],
            "failed_steps": [],
            "step_outputs": {"prepare": []},
        },
        {
            "completed_steps": [],
            "failed_steps": [],
            "step_outputs": {},
            "current_step": 7,
        },
    ],
    ids=[
        "null",
        "list",
        "missing-fields",
        "completed-not-list",
        "completed-member-not-string",
        "failed-not-list",
        "failed-member-not-string",
        "outputs-not-mapping",
        "output-value-not-mapping",
        "current-step-not-string",
    ],
)
def test_base_pipeline_executor_rejects_malformed_resume_checkpoint_structure(
    tmp_path: Path,
    malformed_state: object,
) -> None:
    state_file = tmp_path / ".pipeline_state.json"
    state_file.write_text(json.dumps(malformed_state), encoding="utf-8")

    expected_message = (
        r"^Pipeline checkpoint root must be a JSON object$"
        if type(malformed_state) is not dict
        else r"^Invalid pipeline checkpoint structure: "
    )
    with pytest.raises(
        ValueError,
        match=expected_message,
    ) as exc_info:
        ConcreteExecutor()._initialize_pipeline_state(
            {"working_dir": tmp_path},
            resume=True,
        )

    assert str(malformed_state) not in str(exc_info.value)


@pytest.mark.parametrize(
    "working_dir_name",
    [
        "user:{secret}@checkpoint.example.test",
        "run?X-Amz-Signature={secret}",
        "run#access_token={secret}",
    ],
)
def test_base_pipeline_executor_checkpoint_diagnostics_hide_sensitive_paths(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    working_dir_name: str,
) -> None:
    executor = ConcreteExecutor()
    secret = "checkpoint-diagnostic-path-secret-713"
    working_dir = tmp_path / working_dir_name.format(secret=secret)
    state_file = working_dir / ".pipeline_state.json"
    safe_state = {
        "completed_steps": [],
        "failed_steps": [],
        "step_outputs": {},
        "current_step": None,
    }

    with caplog.at_level(logging.DEBUG):
        executor._save_checkpoint(safe_state, state_file)

    state_file.write_text(
        f'{{"project_name":"https://user:{secret}@example.test"',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc_info:
        executor._initialize_pipeline_state(
            {"working_dir": working_dir},
            resume=True,
        )

    observable = "\n".join(
        (
            caplog.text,
            str(exc_info.value),
            "".join(traceback.format_exception(exc_info.value)),
        )
    )
    assert secret not in observable
    assert "Unable to parse pipeline checkpoint: <redacted>" == str(exc_info.value)


def test_base_pipeline_executor_checkpoint_lock_and_write_share_parent_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "active"
    state_file = state_dir / ".pipeline_state.json"
    moved_state_dir = tmp_path / "held"
    real_open_lock = executor_module.open_confined_lock_file

    @contextmanager
    def swap_path_after_lock_open(
        parent_descriptor: int,
        lock_name: str,
        *,
        file_mode: int = 0o600,
    ) -> Iterator[int]:
        with real_open_lock(
            parent_descriptor,
            lock_name,
            file_mode=file_mode,
        ) as lock_descriptor:
            state_dir.rename(moved_state_dir)
            state_dir.mkdir()
            yield lock_descriptor

    monkeypatch.setattr(
        executor_module,
        "open_confined_lock_file",
        swap_path_after_lock_open,
    )
    pipeline_state = {
        "completed_steps": [],
        "failed_steps": [],
        "step_outputs": {},
        "current_step": None,
    }

    ConcreteExecutor()._save_checkpoint(pipeline_state, state_file)

    assert json.loads((moved_state_dir / state_file.name).read_text()) == pipeline_state
    assert not state_file.exists()


def test_base_pipeline_executor_detaches_os_failure_graphs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "pipeline-os-error-secret-713"

    def fail_os(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(errno.EIO, secret)

    def assert_detached(error: BaseException, function_name: str) -> None:
        assert error.__cause__ is None
        assert error.__context__ is None
        assert secret not in str(error)
        frames = _traceback_locals_for(error, function_name)
        assert frames
        assert secret not in repr(frames)

    credentialed_dir = tmp_path / f"run?X-Amz-Signature={secret}"
    credentialed_dir.mkdir()
    with monkeypatch.context() as scoped:
        scoped.setattr(executor_module, "remove_legacy_pipeline_temp", fail_os)
        with pytest.raises(OSError) as exc_info:
            executor_module.remove_legacy_pipeline_temp_with_safe_diagnostics(
                credentialed_dir
            )
        assert_detached(
            exc_info.value,
            "remove_legacy_pipeline_temp_with_safe_diagnostics",
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(executor_module, "remove_confined_tree", fail_os)
        with pytest.raises(OSError) as exc_info:
            ConcreteExecutor()._clean_directories(
                {
                    "working_dir": credentialed_dir,
                    "working_dir_base": tmp_path,
                    "api_key": secret,
                }
            )
        assert_detached(exc_info.value, "_clean_directories")

    state_file = credentialed_dir / ".pipeline_state.json"
    state_file.write_text('{"completed_steps": []}', encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(executor_module, "open", fail_os, raising=False)
        with pytest.raises(OSError) as exc_info:
            ConcreteExecutor()._initialize_pipeline_state(
                {"working_dir": credentialed_dir, "api_key": secret},
                resume=True,
            )
        assert_detached(exc_info.value, "_initialize_pipeline_state")

    with monkeypatch.context() as scoped:
        scoped.setattr(executor_module, "write_bytes_to_confined", fail_os)
        with pytest.raises(OSError) as exc_info:
            ConcreteExecutor()._save_checkpoint(
                {"completed_steps": [], "opaque": secret},
                state_file,
            )
        assert_detached(exc_info.value, "_save_checkpoint")


def test_base_pipeline_executor_validation_filtering_clean_and_failure(
    tmp_path: Path,
) -> None:
    executor = ConcreteExecutor()
    with pytest.raises(ValueError, match="Required context keys missing"):
        executor.run({"steps": ["one"]})
    with pytest.raises(ValueError, match="No steps to run"):
        executor.run({"steps": [], "working_dir": tmp_path / "empty"})

    assert executor._apply_step_filtering(
        ["one", "two", "three"], {"only_steps": ["one", "three"], "skip_steps": ["one"]}
    ) == ["three"]

    executor._clean_directories({})
    with pytest.raises(ValueError, match="dangerous path"):
        executor._clean_directories({"working_dir": Path("/")})
    with pytest.raises(ValueError, match="dangerous path"):
        executor._clean_directories({"working_dir": Path.home()})
    with pytest.raises(ValueError, match="dangerous path"):
        executor._clean_directories({"working_dir": Path("/") / "tmp" / ".."})
    with pytest.raises(ValueError, match="dangerous path"):
        executor._clean_directories({"working_dir": Path.home() / "not-created" / ".."})

    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(Path("/"), target_is_directory=True)
    with pytest.raises(ValueError, match="dangerous path"):
        executor._clean_directories({"working_dir": root_alias})
    with pytest.raises(ValueError, match="dangerous path"):
        executor._clean_directories(
            {"working_dir": Path("/"), "working_dir_base": tmp_path}
        )
    with pytest.raises(ValueError, match="dangerous path"):
        executor._clean_directories(
            {"working_dir": Path.home(), "working_dir_base": tmp_path}
        )
    with pytest.raises(ValueError, match="dangerous path"):
        executor._clean_directories(
            {
                "working_dir": Path.home(),
                "path_resolver": SimpleNamespace(working_dir_base=Path.home().parent),
            }
        )
    with pytest.raises(ValueError, match="child of the cleanup root"):
        executor._clean_directories(
            {"working_dir": tmp_path, "working_dir_base": tmp_path}
        )

    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "old.txt").write_text("old", encoding="utf-8")
    cleaning = ConcreteExecutor()
    cleaning.run(
        {
            "steps": ["one"],
            "working_dir": dirty,
            "working_dir_base": tmp_path,
            "clean": True,
        }
    )
    assert cleaning.executed == ["one"]
    assert dirty.exists()
    assert not (dirty / "old.txt").exists()

    resolver_owned = tmp_path / "resolver-owned"
    resolver_owned.mkdir()
    (resolver_owned / "old.txt").write_text("old", encoding="utf-8")
    executor._clean_directories(
        {
            "working_dir": resolver_owned,
            "path_resolver": SimpleNamespace(working_dir_base=tmp_path),
        }
    )
    assert not resolver_owned.exists()
    assert not (resolver_owned / "old.txt").exists()

    failing = ConcreteExecutor(fail_step="bad")
    with pytest.raises(RuntimeError, match="Pipeline failed at step 'bad'"):
        failing.run({"steps": ["ok", "bad"], "working_dir": tmp_path / "fail"})
    failed_state = json.loads(
        (tmp_path / "fail" / ".pipeline_state.json").read_text(encoding="utf-8")
    )
    assert failed_state["completed_steps"] == ["ok"]
    assert failed_state["failed_steps"] == ["bad"]


def test_base_pipeline_cleanup_rejects_post_validation_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned_root = tmp_path / "owned"
    alias = owned_root / "alias"
    working_dir = alias / "work"
    working_dir.mkdir(parents=True)
    (working_dir / "owned.txt").write_text("owned", encoding="utf-8")

    outside = tmp_path / "outside"
    outside_work = outside / "work"
    outside_work.mkdir(parents=True)
    sentinel = outside_work / "sentinel.txt"
    sentinel.write_text("outside", encoding="utf-8")

    original_confine = executor_module.confined_cleanup_path

    def swap_after_validation(
        target: str | Path,
        allowed_root: str | Path,
    ) -> Path:
        validated = original_confine(target, allowed_root)
        alias.rename(owned_root / "detached")
        alias.symlink_to(outside, target_is_directory=True)
        return validated

    monkeypatch.setattr(
        executor_module,
        "confined_cleanup_path",
        swap_after_validation,
    )

    with pytest.raises(OSError, match="Unable to clean working directory"):
        ConcreteExecutor()._clean_directories(
            {
                "working_dir": working_dir,
                "working_dir_base": owned_root,
            }
        )

    assert sentinel.read_text(encoding="utf-8") == "outside"
    assert (owned_root / "detached" / "work" / "owned.txt").exists()


def test_base_pipeline_executor_projects_step_failures_and_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "pipeline-diagnostic-secret-713"
    step_name = f"https://user:{secret}@steps.example.test/run"
    tracer = RecordingTracer()
    monkeypatch.setattr(executor_module, "_tracer", tracer)
    executor = ConcreteExecutor(
        fail_step=step_name,
        failure_message=f"api_key={secret}",
    )
    monkeypatch.setattr(executor, "_save_checkpoint", lambda *_args: None)

    with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError) as exc_info:
        executor.run(
            {
                "steps": [step_name],
                "working_dir": tmp_path / "failure",
                "project_name": f"https://user:{secret}@project.example.test/run",
                "session_id": "safe-session",
            }
        )

    telemetry = repr(
        [
            (
                span.name,
                span.attributes,
                [str(error) for error in span.exceptions],
                [str(status) for status in span.statuses],
            )
            for span in tracer.spans
        ]
    )
    observable = "\n".join(
        (
            caplog.text,
            telemetry,
            "".join(traceback.format_exception(exc_info.value)),
        )
    )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in repr((exc_info.value.__cause__, exc_info.value.__context__))
    run_frames = _traceback_locals_for(exc_info.value, "run")
    assert run_frames
    assert secret not in repr(run_frames)
    assert not _traceback_locals_for(exc_info.value, "_run_impl")
    assert not _traceback_locals_for(
        exc_info.value,
        "_execute_step_with_tracing",
    )
    assert secret not in observable
    assert "ValueError during step execution" in observable
    assert "<redacted>" in observable


def test_base_pipeline_executor_default_methods_and_lock_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = BasePipelineExecutor()
    with pytest.raises(NotImplementedError):
        base._execute_step("x", {}, None)
    with pytest.raises(NotImplementedError):
        base._get_step_list_key()
    with pytest.raises(NotImplementedError):
        base._get_required_context_keys()
    with pytest.raises(NotImplementedError):
        base._get_state_file({})

    assert json.dumps({"path": tmp_path}, cls=PathEncoder) == (
        '{"path": "' + str(tmp_path) + '"}'
    )
    with pytest.raises(TypeError):
        PathEncoder().default(object())

    class TimeoutLock:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> None:
            raise executor_module.Timeout("locked")

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(executor_module, "FileLock", TimeoutLock)
    concrete = ConcreteExecutor()
    state_file = tmp_path / "locked" / ".pipeline_state.json"
    state_file.parent.mkdir()
    state_file.write_text(json.dumps({"completed_steps": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Failed to resume"):
        concrete._initialize_pipeline_state({"working_dir": state_file.parent}, True)
    monkeypatch.setattr(
        executor_module,
        "_confined_checkpoint_lock",
        lambda *_args, **_kwargs: TimeoutLock(),
    )
    with pytest.raises(RuntimeError, match="Could not save checkpoint"):
        concrete._save_checkpoint({}, state_file)


def test_base_pipeline_executor_new_state_and_completion_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    executor = ConcreteExecutor()
    state = executor._initialize_pipeline_state(
        {"working_dir": tmp_path, "session_id": "sid", "project_name": "proj"},
        resume=False,
    )
    assert state == {
        "session_id": "sid",
        "project_name": "proj",
        "completed_steps": [],
        "failed_steps": [],
        "step_outputs": {},
        "current_step": None,
    }

    with caplog.at_level(logging.INFO, logger=executor_module.__name__):
        executor._log_pipeline_started(
            {"project_name": "proj", "session_id": "sid"}, ["one", "two"]
        )
        executor._log_pipeline_completed(
            {"project_name": "proj", "session_id": "sid"},
            {"completed_steps": ["one"], "failed_steps": ["two"]},
        )
    assert "Starting pipeline: proj" in caplog.text
    assert "Failed steps: 1" in caplog.text
