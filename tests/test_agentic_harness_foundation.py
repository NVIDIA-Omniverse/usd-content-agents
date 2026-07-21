# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared agentic harness foundation."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path

import pytest
from pydantic import BaseModel

from world_understanding.agentic.events import CollectingEventListener
from world_understanding.agentic.harness import (
    HarnessArtifact,
    HarnessDecision,
    HarnessIssue,
    HarnessRefinementAction,
    HarnessRefinementPlan,
    HarnessRunResult,
    JobRuntime,
    LongRunningJobManager,
    RecipeContext,
    RecipeRegistry,
    RecipeSpec,
    TaskSkillSpec,
    artifact,
)
from world_understanding.agentic.harness.jobs import (
    _json_compatible,
    _normalize_result,
    _status_from_result,
)


class EchoInput(BaseModel):
    value: str


class EchoOutput(BaseModel):
    value: str
    artifact_path: Path


class JsonEnum(Enum):
    VALUE = "enum-value"


@dataclass
class JsonDataclass:
    day: date
    path: Path


async def _echo_recipe(inputs: EchoInput, context: RecipeContext) -> EchoOutput:
    context.raise_if_cancelled()
    context.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = context.output_dir / "echo.txt"
    artifact_path.write_text(inputs.value, encoding="utf-8")
    context.emit("echo.progress", {"value": inputs.value})
    return EchoOutput(value=inputs.value, artifact_path=artifact_path)


async def _echo_task(inputs: EchoInput) -> EchoOutput:
    return EchoOutput(value=inputs.value, artifact_path=Path("echo.txt"))


def _wait_for_terminal(
    manager: LongRunningJobManager,
    job_id: str,
    timeout_s: float = 10.0,
    poll_s: float = 0.02,
) -> dict:
    last: dict | None = None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        last = manager.snapshot(job_id)
        if last["status"] in {"succeeded", "failed", "canceled", "blocked"}:
            return last
        time.sleep(poll_s)
    raise AssertionError(f"job {job_id} did not finish: {last}")


def test_shared_contracts_are_domain_neutral_and_json_serializable(
    tmp_path: Path,
) -> None:
    result = HarnessRunResult(
        run_id="run-1",
        recipe_id="fake.evaluate",
        output_dir=tmp_path,
        status="blocked",
        summary="one blocking issue needs another harness turn",
        artifacts=[
            HarnessArtifact(
                path=str(tmp_path / "render.png"),
                kind="image",
                label="diagnostic render",
                metadata={"view": "front"},
            )
        ],
        issues=[
            HarnessIssue(
                severity="warning",
                code="LOW_SCORE",
                message="score below target",
                blocking=True,
                source="fake_judge",
            )
        ],
        decision=HarnessDecision(
            decision="continue",
            reason="refine the candidate and run the judge again",
            issue_codes=["LOW_SCORE"],
            next_actions=["write refinement plan", "execute plan"],
        ),
    )
    dumped = result.model_dump(mode="json")

    assert dumped["output_dir"] == str(tmp_path)
    assert dumped["decision"]["decision"] == "continue"
    assert dumped["artifacts"][0]["metadata"]["view"] == "front"


def test_refinement_plan_round_trips_without_domain_specific_fields() -> None:
    plan = HarnessRefinementPlan(
        goal="Improve the candidate using evidence from the last run.",
        observations=["The first run has one blocker."],
        evidence_paths=["runs/first/judge.json"],
        actions=[
            HarnessRefinementAction(
                action="patch_source",
                rationale="The harness decided a patch is the next bounded action.",
                target_ids=["candidate.py"],
                params={"patch_path": "runs/first/refinement.patch"},
            )
        ],
        stop_conditions=["accept when no blocking issues remain"],
    )
    data = json.loads(plan.model_dump_json())
    loaded = HarnessRefinementPlan.model_validate(data)

    assert loaded.schema_version == "agentic-harness-refinement-plan/v1"
    assert loaded.actions[0].action == "patch_source"
    assert "cad" not in loaded.model_dump_json().lower()
    assert "material" not in loaded.model_dump_json().lower()


def test_contract_schema_helpers_and_artifact_constructor(tmp_path: Path) -> None:
    context = RecipeContext(run_id="run-1", output_dir=tmp_path)
    context.emit("ignored.without.listener", {"ok": True})

    task_spec = TaskSkillSpec(
        id="test.echo.task",
        domain="test",
        name="Echo task",
        description="Echo input",
        when_to_use="tests",
        task=_echo_task,
        input_model=EchoInput,
        output_model=EchoOutput,
    )
    recipe_spec = RecipeSpec(
        id="test.echo.recipe",
        domain="test",
        name="Echo recipe",
        description="Echo input",
        when_to_use="tests",
        recipe=_echo_recipe,
        input_model=EchoInput,
        output_model=EchoOutput,
    )

    assert task_spec.input_schema()["title"] == "EchoInput"
    assert task_spec.output_schema()["title"] == "EchoOutput"
    assert recipe_spec.input_schema()["title"] == "EchoInput"
    assert recipe_spec.output_schema()["title"] == "EchoOutput"
    assert artifact("out.txt", "text", "Output").model_dump() == {
        "path": "out.txt",
        "kind": "text",
        "label": "Output",
    }


def test_long_running_job_result_helpers_normalize_json_edge_cases(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    today = date(2026, 1, 2)

    assert _json_compatible(now) == "2026-01-02T03:04:05+00:00"
    assert _json_compatible(today) == "2026-01-02"
    assert _json_compatible(JsonEnum.VALUE) == "enum-value"
    assert _json_compatible((tmp_path / "artifact.txt", JsonEnum.VALUE)) == [
        str(tmp_path / "artifact.txt"),
        "enum-value",
    ]
    assert _json_compatible(JsonDataclass(today, tmp_path)) == {
        "day": "2026-01-02",
        "path": str(tmp_path),
    }
    assert _json_compatible(object()).startswith("<object object at ")

    assert _normalize_result(None) == {}
    assert _normalize_result("completed") == {"value": "completed"}
    assert _status_from_result({}) == "succeeded"


@pytest.mark.asyncio
async def test_recipe_registry_validates_dispatches_and_emits_events(
    tmp_path: Path,
) -> None:
    listener = CollectingEventListener()
    context = RecipeContext(
        run_id="run-1",
        output_dir=tmp_path / "recipe",
        event_listener=listener,
    )
    registry = RecipeRegistry()
    echo_spec = RecipeSpec(
        id="fake.echo",
        domain="fake",
        name="Echo",
        description="Echo input into an artifact.",
        when_to_use="Use in tests.",
        recipe=_echo_recipe,
        input_model=EchoInput,
        output_model=EchoOutput,
    )
    registry.register(echo_spec)
    assert registry.get("fake.echo") is echo_spec

    output = await registry.call("fake.echo", {"value": "hello"}, context)
    assert isinstance(output, EchoOutput)
    assert output.artifact_path.read_text(encoding="utf-8") == "hello"
    model_output = await registry.call("fake.echo", EchoInput(value="model"), context)

    assert model_output.value == "model"
    assert [event["type"] for event in listener.events] == [
        "recipe.started",
        "echo.progress",
        "recipe.completed",
        "recipe.started",
        "echo.progress",
        "recipe.completed",
    ]

    other = RecipeRegistry()
    other.register_many(
        [
            RecipeSpec(
                id="fake.one",
                domain="fake",
                name="One",
                description="First recipe.",
                when_to_use="Use in tests.",
                recipe=_echo_recipe,
                input_model=EchoInput,
                output_model=EchoOutput,
            ),
            RecipeSpec(
                id="fake.two",
                domain="fake",
                name="Two",
                description="Second recipe.",
                when_to_use="Use in tests.",
                recipe=_echo_recipe,
                input_model=EchoInput,
                output_model=EchoOutput,
            ),
        ]
    )
    assert sorted(other.as_dict()) == ["fake.one", "fake.two"]
    with pytest.raises(KeyError, match="Unknown recipe id 'missing'"):
        await registry.call("missing", {"value": "x"}, context)


def test_long_running_job_manager_runs_persists_replays_and_resumes(
    tmp_path: Path,
) -> None:
    def runner(request: dict, runtime: JobRuntime) -> dict:
        runtime.output_dir.mkdir(parents=True, exist_ok=True)
        runtime.emit("work", "writing artifact", data={"value": request["value"]})
        artifact_path = runtime.output_dir / "artifact.json"
        artifact_path.write_text(
            json.dumps({"value": request["value"]}), encoding="utf-8"
        )
        return {
            "ok": True,
            "value": request["value"],
            "artifact_path": str(artifact_path),
        }

    manager = LongRunningJobManager(tmp_path / "jobs", runner, kind="fake")
    started = manager.start({"value": "abc"})
    done = _wait_for_terminal(manager, started["job_id"])

    assert done["status"] == "succeeded"
    assert done["result"]["value"] == "abc"
    assert Path(done["result"]["artifact_path"]).exists()

    replayed = list(
        manager.event_batches(
            started["job_id"], since=0, follow=False, keepalive_s=0.01
        )
    )
    replayed_events = [event["event"] for batch in replayed for event in batch]
    assert replayed_events[:2] == ["queued", "started"]
    assert "completed" in replayed_events

    reloaded = LongRunningJobManager(tmp_path / "jobs", runner, kind="fake")
    loaded = reloaded.snapshot(started["job_id"])
    assert loaded["status"] == "succeeded"

    resumed = reloaded.resume(started["job_id"])
    resumed_done = _wait_for_terminal(reloaded, resumed["job_id"])
    assert resumed_done["resumed_from"] == started["job_id"]
    assert resumed_done["status"] == "succeeded"


def test_long_running_job_manager_lists_recent_persisted_jobs(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    for job_id, created_at in [
        ("fake_old", "2026-01-01T00:00:00+00:00"),
        ("fake_new", "2026-01-02T00:00:00+00:00"),
    ]:
        job_dir = jobs_root / job_id
        job_dir.mkdir(parents=True)
        (job_dir / "job_state.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "kind": "fake",
                    "status": "succeeded",
                    "created_at": created_at,
                    "updated_at": created_at,
                    "request": {},
                    "result": {"ok": True, "job_id": job_id},
                }
            ),
            encoding="utf-8",
        )

    manager = LongRunningJobManager(
        jobs_root,
        lambda _request, _runtime: {"ok": True},
        kind="fake",
    )

    snapshots = manager.list(limit=1)

    assert [snapshot["job_id"] for snapshot in snapshots] == ["fake_new"]
    assert manager.snapshot("fake_old")["status"] == "succeeded"


def test_long_running_job_manager_cancels_cooperatively(tmp_path: Path) -> None:
    entered = threading.Event()

    def runner(_request: dict, runtime: JobRuntime) -> dict:
        entered.set()
        deadline = time.time() + 2
        while time.time() < deadline:
            runtime.raise_if_cancelled()
            time.sleep(0.01)
        return {"ok": True}

    manager = LongRunningJobManager(tmp_path / "jobs", runner, kind="fake")
    started = manager.start({})
    assert entered.wait(timeout=2)

    cancel_snapshot = manager.cancel(started["job_id"])
    assert cancel_snapshot["status"] == "cancel_requested"

    done = _wait_for_terminal(manager, started["job_id"])
    assert done["status"] == "canceled"
    assert any(event["event"] == "cancel_requested" for event in done["log_tail"])


def test_long_running_job_manager_cancel_returns_terminal_snapshot(
    tmp_path: Path,
) -> None:
    manager = LongRunningJobManager(
        tmp_path / "jobs",
        lambda _request, _runtime: {"ok": True},
        kind="fake",
    )
    started = manager.start({})
    done = _wait_for_terminal(manager, started["job_id"])

    cancel_snapshot = manager.cancel(started["job_id"])

    assert cancel_snapshot["status"] == "succeeded"
    assert cancel_snapshot["job_id"] == done["job_id"]


def test_long_running_job_manager_event_batches_return_after_terminal(
    tmp_path: Path,
) -> None:
    manager = LongRunningJobManager(
        tmp_path / "jobs",
        lambda _request, _runtime: {"ok": True},
        kind="fake",
    )
    started = manager.start({})
    done = _wait_for_terminal(manager, started["job_id"])

    assert (
        list(
            manager.event_batches(
                started["job_id"],
                since=done["event_count"],
                follow=True,
                keepalive_s=0.01,
            )
        )
        == []
    )


def test_long_running_job_manager_event_batches_emit_keepalive_for_quiet_job(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def runner(_request: dict, _runtime: JobRuntime) -> dict:
        entered.set()
        release.wait(timeout=2)
        return {"ok": True}

    manager = LongRunningJobManager(tmp_path / "jobs", runner, kind="fake")
    started = manager.start({})
    assert entered.wait(timeout=2)
    snapshot = manager.snapshot(started["job_id"])

    generator = manager.event_batches(
        started["job_id"],
        since=snapshot["event_count"],
        follow=True,
        keepalive_s=0.01,
    )

    ping_batch = next(generator)
    generator.close()
    release.set()
    done = _wait_for_terminal(manager, started["job_id"])

    assert ping_batch[0]["event"] == "ping"
    assert ping_batch[0]["keepalive"] is True
    assert ping_batch[0]["index"] == snapshot["event_count"]
    assert done["status"] == "succeeded"


def test_long_running_job_manager_event_batches_waits_for_new_events(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    emit_later = threading.Event()
    release = threading.Event()

    def runner(_request: dict, runtime: JobRuntime) -> dict:
        entered.set()
        assert emit_later.wait(timeout=2)
        runtime.emit("work", "later event", data={"value": "after-wait"})
        release.wait(timeout=2)
        return {"ok": True}

    manager = LongRunningJobManager(tmp_path / "jobs", runner, kind="fake")
    started = manager.start({})
    assert entered.wait(timeout=2)
    snapshot = manager.snapshot(started["job_id"])

    batches: list[list[dict]] = []
    errors: list[BaseException] = []

    def consume_next_batch() -> None:
        try:
            batches.append(
                next(
                    manager.event_batches(
                        started["job_id"],
                        since=snapshot["event_count"],
                        follow=True,
                        keepalive_s=2,
                    )
                )
            )
        except BaseException as exc:  # pragma: no cover - test failure capture
            errors.append(exc)

    consumer = threading.Thread(target=consume_next_batch)
    consumer.start()
    time.sleep(0.05)
    emit_later.set()
    consumer.join(timeout=2)
    release.set()
    done = _wait_for_terminal(manager, started["job_id"])

    assert errors == []
    assert batches
    assert batches[0][0]["event"] == "log"
    assert batches[0][0]["value"] == "after-wait"
    assert done["status"] == "succeeded"


def test_long_running_job_manager_rejects_resume_before_cancel_finishes(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def runner(_request: dict, runtime: JobRuntime) -> dict:
        entered.set()
        while not runtime.cancel_requested():
            time.sleep(0.01)
        release.wait(timeout=2)
        runtime.raise_if_cancelled()
        return {"ok": True}

    manager = LongRunningJobManager(tmp_path / "jobs", runner, kind="fake")
    started = manager.start({})
    assert entered.wait(timeout=2)

    cancel_snapshot = manager.cancel(started["job_id"])
    assert cancel_snapshot["status"] == "cancel_requested"
    with pytest.raises(RuntimeError):
        manager.resume(started["job_id"])

    release.set()
    done = _wait_for_terminal(manager, started["job_id"])
    assert done["status"] == "canceled"


def test_long_running_job_manager_handles_asyncio_cancellation(tmp_path: Path) -> None:
    async def runner(_request: dict, _runtime: JobRuntime) -> dict:
        raise asyncio.CancelledError("async recipe canceled")

    manager = LongRunningJobManager(tmp_path / "jobs", runner, kind="fake")
    started = manager.start({})

    done = _wait_for_terminal(manager, started["job_id"])
    assert done["status"] == "canceled"
    assert done["error"] == "async recipe canceled"


def test_long_running_job_manager_handles_custom_awaitable(tmp_path: Path) -> None:
    class CustomAwaitable:
        def __await__(self):
            async def _inner() -> dict:
                return {"ok": True, "value": "awaited"}

            return _inner().__await__()

    def runner(_request: dict, _runtime: JobRuntime) -> CustomAwaitable:
        return CustomAwaitable()

    manager = LongRunningJobManager(tmp_path / "jobs", runner, kind="fake")
    started = manager.start({})

    done = _wait_for_terminal(manager, started["job_id"])
    assert done["status"] == "succeeded"
    assert done["result"]["value"] == "awaited"


def test_long_running_job_manager_marks_result_completed_after_cancel_as_canceled(
    tmp_path: Path,
) -> None:
    def runner(_request: dict, runtime: JobRuntime) -> dict:
        runtime.cancel_event.set()
        return {"ok": True, "value": "late result"}

    manager = LongRunningJobManager(tmp_path / "jobs", runner, kind="fake")
    started = manager.start({})

    done = _wait_for_terminal(manager, started["job_id"])
    assert done["status"] == "canceled"
    assert done["result"]["ok"] is False
    assert done["result"]["canceled_after_result"] is True
    assert done["result"]["runner_result"]["value"] == "late result"


def test_long_running_job_manager_rejects_unsafe_job_ids(tmp_path: Path) -> None:
    manager = LongRunningJobManager(
        tmp_path / "jobs",
        lambda _request, _runtime: {"ok": True},
        kind="fake",
    )

    for job_id in ["", ".", "..", "../other", "nested/job", "\\windows"]:
        with pytest.raises(ValueError):
            manager.snapshot(job_id)
        with pytest.raises(ValueError):
            manager.cancel(job_id)
        with pytest.raises(ValueError):
            manager.resume(job_id)
        with pytest.raises(ValueError):
            list(manager.event_batches(job_id, follow=False))


def test_long_running_job_manager_rejects_unsafe_kinds(tmp_path: Path) -> None:
    def runner(_request: dict, _runtime: JobRuntime) -> dict:
        return {"ok": True}

    for kind in ["", "../escape", "nested/kind", "\\windows", "bad kind", "$bad"]:
        with pytest.raises(ValueError):
            LongRunningJobManager(tmp_path / "jobs", runner, kind=kind)

    manager = LongRunningJobManager(tmp_path / "jobs", runner, kind="safe.kind-1")
    with pytest.raises(ValueError):
        manager.start({}, kind="../escape")

    started = manager.start({}, kind="override_kind.2")
    done = _wait_for_terminal(manager, started["job_id"])
    assert done["status"] == "succeeded"
    assert done["job_id"].startswith("override_kind.2_")


def test_long_running_job_manager_constrains_output_dir_to_job_dir(
    tmp_path: Path,
) -> None:
    def runner(_request: dict, runtime: JobRuntime) -> dict:
        runtime.output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = runtime.output_dir / "artifact.txt"
        artifact_path.write_text("ok", encoding="utf-8")
        return {"ok": True, "artifact_path": str(artifact_path)}

    manager = LongRunningJobManager(tmp_path / "jobs", runner, kind="fake")

    with pytest.raises(ValueError):
        manager.start({"output_dir": tmp_path / "outside"})
    with pytest.raises(ValueError):
        manager.start({"output_dir": "../outside"})
    with pytest.raises(ValueError):
        manager.start({"output_dir": "."})
    unsafe_default_manager = LongRunningJobManager(
        tmp_path / "unsafe-default-jobs",
        runner,
        kind="fake",
        output_dir_name="../outside",
    )
    with pytest.raises(ValueError):
        unsafe_default_manager.start({})

    started = manager.start({"output_dir": "artifacts"})
    done = _wait_for_terminal(manager, started["job_id"])
    assert done["status"] == "succeeded"
    assert Path(done["output_dir"]) == Path(done["job_dir"]) / "artifacts"
    assert Path(done["result"]["artifact_path"]).read_text(encoding="utf-8") == "ok"


def test_long_running_job_manager_reloads_orphaned_job_with_output_subdir(
    tmp_path: Path,
) -> None:
    jobs_root = tmp_path / "jobs"
    job_dir = jobs_root / "fake_orphan"
    job_dir.mkdir(parents=True)
    (job_dir / "job_state.json").write_text(
        json.dumps(
            {
                "job_id": "fake_orphan",
                "kind": "fake",
                "status": "running",
                "request": {},
            }
        ),
        encoding="utf-8",
    )

    manager = LongRunningJobManager(
        jobs_root,
        lambda _request, _runtime: {"ok": True},
        kind="fake",
        output_dir_name="artifact_output",
    )

    loaded = manager.snapshot("fake_orphan")
    assert loaded["status"] == "failed"
    assert loaded["stage"] == "orphaned"
    assert loaded["cancel_requested"] is False
    assert loaded["output_dir"] == str(job_dir / "artifact_output")
    persisted = json.loads((job_dir / "job_state.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["stage"] == "orphaned"


def test_long_running_job_manager_loads_corrupt_persisted_edges(
    tmp_path: Path,
) -> None:
    jobs_root = tmp_path / "jobs"
    job_dir = jobs_root / "fake_bad"
    job_dir.mkdir(parents=True)
    (job_dir / "job_state.json").write_text(
        json.dumps(
            {
                "job_id": "fake_bad",
                "kind": "bad kind",
                "status": "succeeded",
                "output_dir": str(tmp_path / "outside"),
                "request": {},
                "result": {"ok": True},
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "events.jsonl").write_text(
        "\nnot-json\n"
        + json.dumps(
            {
                "index": 0,
                "event": "loaded",
                "job_id": "fake_bad",
                "kind": "fake",
                "status": "succeeded",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manager = LongRunningJobManager(
        jobs_root,
        lambda _request, _runtime: {"ok": True},
        kind="fake",
    )

    loaded = manager.snapshot("fake_bad")

    assert loaded["kind"] == "fake"
    assert loaded["output_dir"] == str(job_dir / "output")
    assert loaded["event_count"] == 1
    assert loaded["log_tail"][0]["event"] == "loaded"
    with pytest.raises(KeyError):
        manager.snapshot("missing_job")


def test_long_running_job_manager_normalizes_pydantic_results(tmp_path: Path) -> None:
    def runner(_request: dict, runtime: JobRuntime) -> HarnessRunResult:
        return HarnessRunResult(
            run_id="run-1",
            recipe_id="fake.recipe",
            output_dir=runtime.output_dir,
            status="succeeded",
            summary="recipe completed",
            artifacts=[
                HarnessArtifact(
                    path=str(runtime.output_dir / "render.png"),
                    kind="image",
                    label="render",
                    metadata={"source_dir": runtime.output_dir},
                )
            ],
        )

    manager = LongRunningJobManager(tmp_path / "jobs", runner, kind="fake")
    started = manager.start({})

    done = _wait_for_terminal(manager, started["job_id"])
    assert done["status"] == "succeeded"
    assert done["result"]["output_dir"] == done["output_dir"]
    assert done["result"]["artifacts"][0]["path"].endswith("render.png")
    assert (
        done["result"]["artifacts"][0]["metadata"]["source_dir"] == done["output_dir"]
    )
    json.dumps(done)


def test_long_running_job_manager_uses_result_status_and_protects_event_fields(
    tmp_path: Path,
) -> None:
    def runner(request: dict, runtime: JobRuntime) -> dict:
        runtime.emit(
            "work",
            "payload tries to overwrite reserved fields",
            data={
                "event": "payload-event",
                "status": "payload-status",
                "data": "payload-data",
                "path": tmp_path,
                "value": request["status"],
            },
        )
        return {
            "status": request["status"],
            "ok": request.get("ok"),
            "summary": "terminal by result status",
        }

    manager = LongRunningJobManager(tmp_path / "jobs", runner, kind="fake")
    blocked = manager.start({"status": "blocked"})
    blocked_done = _wait_for_terminal(manager, blocked["job_id"])
    assert blocked_done["status"] == "blocked"

    log_event = next(
        event for event in blocked_done["log_tail"] if event["event"] == "log"
    )
    assert log_event["status"] == "running"
    assert log_event["event"] == "log"
    assert log_event["value"] == "blocked"
    assert log_event["path"] == str(tmp_path)
    assert log_event["data"]["event"] == "payload-event"
    assert log_event["data"]["status"] == "payload-status"
    assert log_event["data"]["data"] == "payload-data"
    assert log_event["data"]["path"] == str(tmp_path)
    json.dumps(blocked_done)

    failed = manager.start({"status": "failed"})
    failed_done = _wait_for_terminal(manager, failed["job_id"])
    assert failed_done["status"] == "failed"

    contradictory = manager.start({"status": "succeeded", "ok": False})
    contradictory_done = _wait_for_terminal(manager, contradictory["job_id"])
    assert contradictory_done["status"] == "failed"


def test_long_running_job_manager_rejects_unknown_result_status(
    tmp_path: Path,
) -> None:
    def runner(_request: dict, _runtime: JobRuntime) -> dict:
        return {"status": "running", "summary": "not a terminal result"}

    manager = LongRunningJobManager(tmp_path / "jobs", runner, kind="fake")
    started = manager.start({})

    done = _wait_for_terminal(manager, started["job_id"])
    assert done["status"] == "failed"
    assert done["result"]["status"] == "running"


def test_long_running_job_manager_evicts_terminal_jobs_from_memory(
    tmp_path: Path,
) -> None:
    manager = LongRunningJobManager(
        tmp_path / "jobs",
        lambda _request, _runtime: {"ok": True},
        kind="fake",
        max_loaded_jobs=2,
    )

    done_ids = []
    for _ in range(4):
        started = manager.start({})
        done_ids.append(_wait_for_terminal(manager, started["job_id"])["job_id"])

    assert len(manager._jobs) <= 2
    assert manager.snapshot(done_ids[0])["status"] == "succeeded"


@pytest.mark.asyncio
async def test_recipe_registry_emits_cancelled_for_recipe_raised_cancel(
    tmp_path: Path,
) -> None:
    async def cancel_recipe(_inputs: EchoInput, _context: RecipeContext) -> EchoOutput:
        raise asyncio.CancelledError("recipe canceled itself")

    listener = CollectingEventListener()
    context = RecipeContext(
        run_id="run-cancel",
        output_dir=tmp_path / "recipe",
        event_listener=listener,
    )
    registry = RecipeRegistry()
    registry.register(
        RecipeSpec(
            id="fake.cancel",
            domain="fake",
            name="Cancel",
            description="Cancel from inside recipe body.",
            when_to_use="Use in tests.",
            recipe=cancel_recipe,
            input_model=EchoInput,
            output_model=EchoOutput,
        )
    )

    with pytest.raises(asyncio.CancelledError):
        await registry.call("fake.cancel", {"value": "x"}, context)

    assert [event["type"] for event in listener.events] == [
        "recipe.started",
        "recipe.cancelled",
    ]


@pytest.mark.asyncio
async def test_recipe_registry_does_not_duplicate_context_cancel_event(
    tmp_path: Path,
) -> None:
    cancel_checks = 0

    def cancel_after_start() -> bool:
        nonlocal cancel_checks
        cancel_checks += 1
        return cancel_checks > 1

    async def cancel_recipe(_inputs: EchoInput, context: RecipeContext) -> EchoOutput:
        context.raise_if_cancelled()
        raise AssertionError("recipe should have observed cancellation")

    listener = CollectingEventListener()
    context = RecipeContext(
        run_id="run-context-cancel",
        output_dir=tmp_path / "recipe",
        event_listener=listener,
        cancel_checker=cancel_after_start,
    )
    registry = RecipeRegistry()
    registry.register(
        RecipeSpec(
            id="fake.context-cancel",
            domain="fake",
            name="Context Cancel",
            description="Cancel through RecipeContext.",
            when_to_use="Use in tests.",
            recipe=cancel_recipe,
            input_model=EchoInput,
            output_model=EchoOutput,
        )
    )

    with pytest.raises(asyncio.CancelledError):
        await registry.call("fake.context-cancel", {"value": "x"}, context)

    assert [event["type"] for event in listener.events] == [
        "recipe.started",
        "recipe.cancelled",
    ]


@pytest.mark.asyncio
async def test_recipe_registry_keeps_output_when_cancel_arrives_after_return(
    tmp_path: Path,
) -> None:
    cancel_requested = False

    def cancel_checker() -> bool:
        return cancel_requested

    async def finish_recipe(_inputs: EchoInput, context: RecipeContext) -> EchoOutput:
        nonlocal cancel_requested
        artifact_path = context.output_dir / "finished.txt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text("finished", encoding="utf-8")
        cancel_requested = True
        return EchoOutput(value="finished", artifact_path=artifact_path)

    listener = CollectingEventListener()
    context = RecipeContext(
        run_id="run-finish-before-cancel",
        output_dir=tmp_path / "recipe",
        event_listener=listener,
        cancel_checker=cancel_checker,
    )
    registry = RecipeRegistry()
    registry.register(
        RecipeSpec(
            id="fake.finish-before-cancel",
            domain="fake",
            name="Finish Before Cancel",
            description="Return output even if cancellation arrives after return.",
            when_to_use="Use in tests.",
            recipe=finish_recipe,
            input_model=EchoInput,
            output_model=EchoOutput,
        )
    )

    output = await registry.call("fake.finish-before-cancel", {"value": "x"}, context)

    assert isinstance(output, EchoOutput)
    assert output.artifact_path.read_text(encoding="utf-8") == "finished"
    assert [event["type"] for event in listener.events] == [
        "recipe.started",
        "recipe.completed",
    ]
