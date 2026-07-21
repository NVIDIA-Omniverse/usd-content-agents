# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic state transitions for large-scene orchestration."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock
from pydantic import ValidationError

from content_agent_workflows.common.artifacts import (
    artifact_set_digest,
    atomic_write_json,
    load_json,
)

from .gates import validate_handoff
from .models import (
    PHASE_ORDER,
    HandoffValidationReport,
    LargeSceneRun,
    PhaseName,
    PhaseState,
    PhaseTransition,
)


class LargeSceneStateError(RuntimeError):
    """Raised when a requested run-state transition is invalid."""


class HandoffValidationError(LargeSceneStateError):
    """Raised after a failed handoff is durably recorded."""

    def __init__(self, report: HandoffValidationReport) -> None:
        super().__init__("; ".join(report.errors))
        self.report = report


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _run_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _source_scene_dependency_paths(source_scene: str | Path) -> list[Path]:
    source = Path(source_scene).expanduser().resolve()
    try:
        from pxr import Usd  # type: ignore
    except Exception as exc:
        raise ValueError(
            "USD Python bindings are required to digest source scene"
        ) from exc

    stage = Usd.Stage.Open(str(source), Usd.Stage.LoadAll)
    if not stage:
        raise ValueError(f"Could not open source scene: {source}")

    dependencies: set[Path] = {source}
    for layer in stage.GetUsedLayers():
        real_path = getattr(layer, "realPath", "") or ""
        if real_path:
            dependencies.add(Path(real_path).expanduser().resolve())
    return sorted(dependencies, key=str)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with FileLock(str(lock_path)):
        yield


def _source_input_digest(
    source_scene: str | Path,
    request_artifact_paths: list[str | Path],
    requested_tasks: list[str],
    additional_instructions: str | None = None,
) -> str:
    metadata: dict[str, object] = {
        "schema": "content-agent-workflows.large-scene-input.v1",
        "requested_tasks": sorted(requested_tasks),
    }
    if additional_instructions:
        metadata["additional_instructions"] = additional_instructions
    return artifact_set_digest(
        [*_source_scene_dependency_paths(source_scene), *request_artifact_paths],
        metadata=metadata,
    )


def _write_run(path: Path, run: LargeSceneRun) -> LargeSceneRun:
    updated = run.model_copy(update={"revision": run.revision + 1})
    atomic_write_json(path, updated)
    return updated


def load_run_state(path: str | Path) -> LargeSceneRun:
    """Load and validate durable large-scene run state."""

    resolved = _run_path(path)
    try:
        return LargeSceneRun.model_validate(load_json(resolved))
    except (OSError, ValueError, ValidationError) as exc:
        raise LargeSceneStateError(f"Invalid run state at {resolved}: {exc}") from exc


def _transition(
    run: LargeSceneRun,
    phase: PhaseName,
    *,
    from_state: PhaseState,
    to_status: str,
    reason: str,
    actor: str,
    result_path: str | None = None,
    output_digest: str | None = None,
) -> None:
    run.transitions.append(
        PhaseTransition(
            timestamp=_timestamp(),
            phase=phase,
            from_status=from_state.status,
            to_status=to_status,
            reason=reason,
            actor=actor,
            input_digest=from_state.input_digest,
            result_path=result_path
            if result_path is not None
            else from_state.result_path,
            output_digest=output_digest
            if output_digest is not None
            else from_state.output_digest,
        )
    )


def create_run(
    path: str | Path,
    *,
    run_id: str,
    source_scene: str | Path,
    requested_tasks: list[str],
    request_artifact_paths: list[str | Path] | None = None,
    additional_instructions: str | None = None,
    actor: str = "agent",
) -> LargeSceneRun:
    """Create a new run with Workflow 1 ready to begin."""

    resolved_path = _run_path(path)
    source = Path(source_scene).expanduser().resolve()
    request_paths = [
        Path(item).expanduser().resolve() for item in (request_artifact_paths or [])
    ]
    normalized_instructions = (additional_instructions or "").strip() or None
    tasks = list(dict.fromkeys(requested_tasks))
    if not tasks:
        raise LargeSceneStateError("At least one requested task is required")
    if len(tasks) != len(requested_tasks):
        raise LargeSceneStateError("requested_tasks must be unique")
    try:
        input_digest = _source_input_digest(
            source,
            request_paths,
            tasks,
            normalized_instructions,
        )
    except (OSError, ValueError) as exc:
        raise LargeSceneStateError(f"Cannot digest large-scene inputs: {exc}") from exc

    with _exclusive_lock(resolved_path):
        if resolved_path.exists():
            raise LargeSceneStateError(f"Run state already exists: {resolved_path}")
        phases: dict[PhaseName, PhaseState] = {
            "decomposition": PhaseState(status="ready", input_digest=input_digest),
            "asset_task_processing": PhaseState(),
            "collection": PhaseState(),
        }
        run = LargeSceneRun(
            run_id=run_id,
            source_scene=str(source),
            additional_instructions=normalized_instructions,
            request_artifact_paths=[str(item) for item in request_paths],
            requested_tasks=tasks,
            source_input_digest=input_digest,
            phases=phases,
        )
        run.transitions.append(
            PhaseTransition(
                timestamp=_timestamp(),
                phase="decomposition",
                from_status="pending",
                to_status="ready",
                reason="Run created and source inputs digested.",
                actor=actor,
                input_digest=input_digest,
            )
        )
        return _write_run(resolved_path, run)


def _preceding_phase(phase: PhaseName) -> PhaseName | None:
    index = PHASE_ORDER.index(phase)
    return PHASE_ORDER[index - 1] if index else None


def _verify_source_inputs(run: LargeSceneRun) -> str:
    try:
        current_digest = _source_input_digest(
            run.source_scene,
            run.request_artifact_paths,
            run.requested_tasks,
            run.additional_instructions,
        )
    except (OSError, ValueError) as exc:
        raise LargeSceneStateError(f"Cannot verify source inputs: {exc}") from exc
    if current_digest != run.source_input_digest:
        raise LargeSceneStateError(
            "Source scene or request artifacts changed; invalidate from decomposition"
        )
    return current_digest


def _verify_prerequisites(run: LargeSceneRun, phase: PhaseName) -> None:
    state = run.phases[phase]
    current_source_digest = _verify_source_inputs(run)
    if phase == "decomposition":
        if state.input_digest != current_source_digest:
            raise LargeSceneStateError(
                "Source scene or request artifacts changed; invalidate from decomposition"
            )
        return
    previous_phase = _preceding_phase(phase)
    assert previous_phase is not None
    previous = run.phases[previous_phase]
    if previous.status != "completed" or not previous.result_path:
        raise LargeSceneStateError(
            f"Cannot begin {phase}: {previous_phase} is not completed"
        )
    if not previous.output_digest or state.input_digest != previous.output_digest:
        raise LargeSceneStateError(
            f"Cannot begin {phase}: predecessor output digest does not match phase input"
        )
    report = validate_handoff(run, previous_phase, previous.result_path)
    if not report.valid:
        raise LargeSceneStateError(
            f"Cannot begin {phase}: predecessor handoff is stale: "
            + "; ".join(report.errors)
        )
    if report.output_digest != previous.output_digest:
        raise LargeSceneStateError(
            f"Cannot begin {phase}: {previous_phase} output digest changed; "
            f"expected {previous.output_digest}, got {report.output_digest}"
        )


def begin_phase(
    path: str | Path,
    phase: PhaseName,
    *,
    actor: str = "agent",
) -> LargeSceneRun:
    """Begin a ready phase after revalidating its prerequisites."""

    resolved_path = _run_path(path)
    with _exclusive_lock(resolved_path):
        run = load_run_state(resolved_path)
        state = run.phases[phase]
        if state.status != "ready":
            raise LargeSceneStateError(
                f"Cannot begin {phase} from status {state.status}; expected ready"
            )
        _verify_prerequisites(run, phase)
        previous = state.model_copy(deep=True)
        state.status = "running"
        state.error = None
        run.current_phase = phase
        _transition(
            run,
            phase,
            from_state=previous,
            to_status="running",
            reason="Phase prerequisites and input digest validated.",
            actor=actor,
        )
        return _write_run(resolved_path, run)


def validate_phase_handoff(
    path: str | Path,
    phase: PhaseName,
    result_path: str | Path,
) -> HandoffValidationReport:
    """Validate a handoff against current run state without changing it."""

    return validate_handoff(load_run_state(path), phase, result_path)


def complete_phase(
    path: str | Path,
    phase: PhaseName,
    result_path: str | Path,
    *,
    actor: str = "agent",
) -> LargeSceneRun:
    """Complete a running phase and atomically make its successor ready."""

    resolved_path = _run_path(path)
    resolved_result_path = Path(result_path).expanduser().resolve()
    with _exclusive_lock(resolved_path):
        run = load_run_state(resolved_path)
        state = run.phases[phase]
        if state.status != "running":
            raise LargeSceneStateError(
                f"Cannot complete {phase} from status {state.status}; expected running"
            )
        report = validate_handoff(run, phase, resolved_result_path)
        if not report.valid:
            previous = state.model_copy(deep=True)
            state.status = "failed"
            state.result_path = str(resolved_result_path)
            state.output_digest = report.output_digest
            state.error = "; ".join(report.errors)
            run.current_phase = phase
            _transition(
                run,
                phase,
                from_state=previous,
                to_status="failed",
                reason="Phase handoff validation failed.",
                actor=actor,
                result_path=str(resolved_result_path),
                output_digest=report.output_digest,
            )
            _write_run(resolved_path, run)
            raise HandoffValidationError(report)

        previous = state.model_copy(deep=True)
        state.status = "completed"
        state.result_path = str(resolved_result_path)
        state.output_digest = report.output_digest
        state.error = None
        _transition(
            run,
            phase,
            from_state=previous,
            to_status="completed",
            reason="Phase handoff validated.",
            actor=actor,
            result_path=str(resolved_result_path),
            output_digest=report.output_digest,
        )

        phase_index = PHASE_ORDER.index(phase)
        if phase_index + 1 < len(PHASE_ORDER):
            next_phase = PHASE_ORDER[phase_index + 1]
            next_state = run.phases[next_phase]
            previous_next = next_state.model_copy(deep=True)
            next_state.status = "ready"
            next_state.input_digest = report.output_digest
            next_state.result_path = None
            next_state.output_digest = None
            next_state.error = None
            _transition(
                run,
                next_phase,
                from_state=previous_next,
                to_status="ready",
                reason=f"Validated output from {phase} accepted as phase input.",
                actor=actor,
            )
            run.current_phase = next_phase
        else:
            run.current_phase = None
        return _write_run(resolved_path, run)


def fail_phase(
    path: str | Path,
    phase: PhaseName,
    *,
    reason: str,
    actor: str = "agent",
) -> LargeSceneRun:
    """Record an explicit phase execution failure."""

    if not reason.strip():
        raise LargeSceneStateError("A failure reason is required")
    resolved_path = _run_path(path)
    with _exclusive_lock(resolved_path):
        run = load_run_state(resolved_path)
        state = run.phases[phase]
        if state.status not in {"ready", "running"}:
            raise LargeSceneStateError(
                f"Cannot fail {phase} from status {state.status}"
            )
        previous = state.model_copy(deep=True)
        state.status = "failed"
        state.error = reason
        run.current_phase = phase
        _transition(
            run,
            phase,
            from_state=previous,
            to_status="failed",
            reason=reason,
            actor=actor,
        )
        return _write_run(resolved_path, run)


def invalidate_from(
    path: str | Path,
    phase: PhaseName,
    *,
    reason: str,
    actor: str = "agent",
) -> LargeSceneRun:
    """Return to the earliest affected phase and invalidate all successors."""

    if not reason.strip():
        raise LargeSceneStateError("An invalidation reason is required")
    resolved_path = _run_path(path)
    with _exclusive_lock(resolved_path):
        run = load_run_state(resolved_path)
        phase_index = PHASE_ORDER.index(phase)
        previous_phase = _preceding_phase(phase)
        if previous_phase is None:
            try:
                input_digest = _source_input_digest(
                    run.source_scene,
                    run.request_artifact_paths,
                    run.requested_tasks,
                    run.additional_instructions,
                )
            except (OSError, ValueError) as exc:
                raise LargeSceneStateError(
                    f"Cannot refresh source input digest: {exc}"
                ) from exc
            run.source_input_digest = input_digest
        else:
            predecessor = run.phases[previous_phase]
            if predecessor.status != "completed" or not predecessor.output_digest:
                raise LargeSceneStateError(
                    f"Cannot invalidate from {phase}: {previous_phase} is not completed"
                )
            input_digest = predecessor.output_digest

        for index in range(phase_index, len(PHASE_ORDER)):
            affected_phase = PHASE_ORDER[index]
            state = run.phases[affected_phase]
            previous = state.model_copy(deep=True)
            state.status = "ready" if affected_phase == phase else "invalidated"
            state.input_digest = input_digest if affected_phase == phase else None
            state.result_path = None
            state.output_digest = None
            state.error = None
            _transition(
                run,
                affected_phase,
                from_state=previous,
                to_status=state.status,
                reason=reason,
                actor=actor,
            )
        run.current_phase = phase
        return _write_run(resolved_path, run)


def revise_additional_instructions(
    path: str | Path,
    *,
    additional_instructions: str,
    reason: str,
    actor: str = "agent",
) -> LargeSceneRun:
    """Revise downstream guidance while preserving completed decomposition."""

    normalized_instructions = additional_instructions.strip()
    if not normalized_instructions:
        raise LargeSceneStateError("Revised additional instructions are required")
    if not reason.strip():
        raise LargeSceneStateError("A guidance revision reason is required")

    resolved_path = _run_path(path)
    with _exclusive_lock(resolved_path):
        run = load_run_state(resolved_path)
        decomposition = run.phases["decomposition"]
        if decomposition.status != "completed" or not decomposition.output_digest:
            raise LargeSceneStateError(
                "Additional instructions can be revised in place only after "
                "decomposition is completed"
            )
        if run.additional_instructions == normalized_instructions:
            raise LargeSceneStateError("Revised additional instructions are unchanged")

        run.additional_instructions = normalized_instructions
        try:
            run.source_input_digest = _source_input_digest(
                run.source_scene,
                run.request_artifact_paths,
                run.requested_tasks,
                run.additional_instructions,
            )
        except (OSError, ValueError) as exc:
            raise LargeSceneStateError(
                f"Cannot refresh source input digest: {exc}"
            ) from exc
        input_digest = decomposition.output_digest
        for index, affected_phase in enumerate(PHASE_ORDER[1:]):
            state = run.phases[affected_phase]
            previous = state.model_copy(deep=True)
            state.status = "ready" if index == 0 else "invalidated"
            state.input_digest = input_digest if index == 0 else None
            state.result_path = None
            state.output_digest = None
            state.error = None
            _transition(
                run,
                affected_phase,
                from_state=previous,
                to_status=state.status,
                reason=reason,
                actor=actor,
            )
        run.current_phase = "asset_task_processing"
        return _write_run(resolved_path, run)
