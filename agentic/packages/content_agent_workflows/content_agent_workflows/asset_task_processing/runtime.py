# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic runtime primitives for Workflow 2."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from filelock import FileLock
from pydantic import BaseModel, ValidationError

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_json,
    resolve_artifact_path,
    seal_phase_result,
)
from content_agent_workflows.scene_decomposition.manifest import (
    ManifestCatalog,
    SceneDecompositionManifest,
)

from .contracts import (
    AcceptedWaiver,
    AgentPlanPointer,
    AssetTaskInventory,
    AssetTaskResult,
    AssetTaskResultsIndex,
    AssetTaskRunState,
    AssetTaskStateTransition,
    AssetTaskWorkItem,
    AssetTaskWorkItemState,
    DecisionLedgerEntry,
    ProcessingPhaseResult,
    ResultIndexEntry,
    TaskCatalog,
)


class AssetTaskRuntimeError(RuntimeError):
    """Raised when a Workflow 2 state operation is invalid."""


@dataclass(frozen=True)
class ProcessingPaths:
    """Canonical files owned by one Workflow 2 run."""

    output_dir: Path
    inventory: Path
    state: Path
    plan_pointer: Path
    ledger: Path
    results_index: Path
    phase_result: Path
    lock: Path

    @classmethod
    def from_output_dir(cls, output_dir: str | Path) -> ProcessingPaths:
        root = Path(output_dir).expanduser().resolve()
        return cls(
            output_dir=root,
            inventory=root / "asset_task_inventory.json",
            state=root / "asset_task_run_state.json",
            plan_pointer=root / "agent_plan" / "current.json",
            ledger=root / "decision_ledger.jsonl",
            results_index=root / "asset_task_results_index.json",
            phase_result=root / "processing_result.json",
            lock=root / ".asset_task_processing.lock",
        )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@contextmanager
def _exclusive_lock(paths: ProcessingPaths) -> Iterator[None]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(paths.lock)):
        yield


def _load_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        return model_type.model_validate(load_json(path))
    except (OSError, ValueError, ValidationError) as exc:
        raise AssetTaskRuntimeError(
            f"Invalid {model_type.__name__} at {path}: {exc}"
        ) from exc


def _write_state(path: Path, state: AssetTaskRunState) -> AssetTaskRunState:
    updated = state.model_copy(update={"revision": state.revision + 1})
    atomic_write_json(path, updated)
    return updated


def _resolve_reference(reference: str, owner_path: Path) -> Path:
    return resolve_artifact_path(reference, base_dir=owner_path.parent)


def _load_manifests(
    catalog: ManifestCatalog,
    catalog_path: Path,
) -> dict[str, tuple[Path, SceneDecompositionManifest]]:
    result: dict[str, tuple[Path, SceneDecompositionManifest]] = {}
    for entry in catalog.manifests:
        if not entry.finalized:
            raise AssetTaskRuntimeError(
                f"Manifest {entry.manifest_id} is not finalized"
            )
        path = _resolve_reference(entry.path, catalog_path)
        if not path.is_file():
            raise AssetTaskRuntimeError(f"Manifest does not exist: {path}")
        if file_sha256(path) != entry.manifest_digest:
            raise AssetTaskRuntimeError(f"Manifest digest mismatch: {path}")
        result[entry.manifest_id] = (
            path,
            _load_model(path, SceneDecompositionManifest),
        )
    return result


def _task_request_digests(
    task_catalog: TaskCatalog,
    task_catalog_path: Path,
) -> dict[str, str]:
    """Return stable content identities for every task request."""

    digests: dict[str, str] = {}
    for task in task_catalog.tasks:
        request_path = _resolve_reference(task.request_path, task_catalog_path)
        if not request_path.is_file():
            raise AssetTaskRuntimeError(
                f"Task request does not exist for {task.task_id}: {request_path}"
            )
        digests[task.task_id] = file_sha256(request_path)
    return digests


def _load_run(
    output_dir: str | Path,
) -> tuple[
    ProcessingPaths,
    AssetTaskInventory,
    AssetTaskRunState,
    TaskCatalog,
    ManifestCatalog,
]:
    paths = ProcessingPaths.from_output_dir(output_dir)
    state = _load_model(paths.state, AssetTaskRunState)
    inventory_path = Path(state.inventory_path).expanduser().resolve()
    task_catalog_path = Path(state.task_catalog_path).expanduser().resolve()
    manifest_catalog_path = Path(state.manifest_catalog_path).expanduser().resolve()
    inventory = _load_model(inventory_path, AssetTaskInventory)
    task_catalog = _load_model(task_catalog_path, TaskCatalog)
    manifest_catalog = _load_model(manifest_catalog_path, ManifestCatalog)
    if inventory.input_digest != state.input_digest:
        raise AssetTaskRuntimeError("Inventory and run-state input digests differ")
    if inventory.task_request_digests != state.task_request_digests:
        raise AssetTaskRuntimeError(
            "Inventory and run-state task request digests differ"
        )
    if state.task_request_digests:
        current_request_digests = _task_request_digests(task_catalog, task_catalog_path)
        if current_request_digests != state.task_request_digests:
            raise AssetTaskRuntimeError(
                "Task request changed after Workflow 2 preparation; invalidate and "
                "prepare the phase again"
            )
    inventory_ids = {item.work_item_id for item in inventory.work_items}
    state_ids = {item.work_item_id for item in state.work_items}
    if inventory_ids != state_ids:
        raise AssetTaskRuntimeError(
            "Inventory and run state contain different work items"
        )
    return paths, inventory, state, task_catalog, manifest_catalog


def prepare_processing_run(
    *,
    manifest_catalog_path: str | Path,
    task_catalog_path: str | Path,
    output_dir: str | Path,
    input_digest: str,
) -> dict[str, object]:
    """Create an immutable work inventory and empty mutable run state."""

    if not input_digest.strip():
        raise AssetTaskRuntimeError("input_digest is required")
    paths = ProcessingPaths.from_output_dir(output_dir)
    manifest_catalog_file = Path(manifest_catalog_path).expanduser().resolve()
    task_catalog_file = Path(task_catalog_path).expanduser().resolve()
    manifest_catalog = _load_model(manifest_catalog_file, ManifestCatalog)
    task_catalog = _load_model(task_catalog_file, TaskCatalog)
    manifests = _load_manifests(manifest_catalog, manifest_catalog_file)
    task_request_digests = _task_request_digests(task_catalog, task_catalog_file)

    work_items: list[AssetTaskWorkItem] = []
    for task in task_catalog.tasks:
        if task.manifest_id not in manifests:
            raise AssetTaskRuntimeError(
                f"Task {task.task_id} references unknown manifest {task.manifest_id}"
            )
        _manifest_path, manifest = manifests[task.manifest_id]
        for asset in manifest.processable_assets:
            working_usd_path = (
                str(_resolve_reference(asset.working_usd_path, _manifest_path))
                if asset.working_usd_path
                else None
            )
            work_items.append(
                AssetTaskWorkItem(
                    work_item_id=f"{task.task_id}:{task.manifest_id}:{asset.asset_id}",
                    manifest_id=task.manifest_id,
                    asset_id=asset.asset_id,
                    asset_label=asset.label,
                    task_id=task.task_id,
                    required=task.required,
                    original_root_path=asset.original_root_path,
                    working_usd_path=working_usd_path,
                    working_root_path=asset.working_root_path,
                    source_path_prefixes=asset.source_path_prefixes,
                )
            )
    inventory = AssetTaskInventory(
        input_digest=input_digest,
        task_request_digests=task_request_digests,
        work_items=work_items,
    )
    state = AssetTaskRunState(
        input_digest=input_digest,
        inventory_path=str(paths.inventory),
        task_catalog_path=str(task_catalog_file),
        manifest_catalog_path=str(manifest_catalog_file),
        task_request_digests=task_request_digests,
        work_items=[
            AssetTaskWorkItemState(work_item_id=item.work_item_id)
            for item in work_items
        ],
    )

    with _exclusive_lock(paths):
        existing = [
            path
            for path in (
                paths.inventory,
                paths.state,
                paths.results_index,
                paths.ledger,
            )
            if path.exists()
        ]
        if existing:
            raise AssetTaskRuntimeError(
                "Processing run already exists; resume it instead of preparing again: "
                + ", ".join(map(str, existing))
            )
        atomic_write_json(paths.inventory, inventory)
        _write_state(paths.state, state)
        atomic_write_json(paths.results_index, AssetTaskResultsIndex(entries=[]))
        atomic_write_text(paths.ledger, "")

    return {
        "status": "prepared",
        "output_dir": str(paths.output_dir),
        "task_request_digests": task_request_digests,
        "inventory_path": str(paths.inventory),
        "state_path": str(paths.state),
        "results_index_path": str(paths.results_index),
        "decision_ledger_path": str(paths.ledger),
        "work_item_count": len(work_items),
        "required_work_item_count": sum(item.required for item in work_items),
    }


def record_plan(
    output_dir: str | Path,
    plan_file: str | Path,
) -> AgentPlanPointer:
    """Copy an agent-authored plan into the next immutable revision."""

    paths = ProcessingPaths.from_output_dir(output_dir)
    source = Path(plan_file).expanduser().resolve()
    try:
        plan_text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise AssetTaskRuntimeError(f"Cannot read agent plan {source}: {exc}") from exc
    if not plan_text.strip():
        raise AssetTaskRuntimeError("Agent plan must not be empty")

    with _exclusive_lock(paths):
        _load_run(paths.output_dir)
        if paths.plan_pointer.exists():
            pointer = _load_model(paths.plan_pointer, AgentPlanPointer)
            revision = pointer.current_revision + 1
            revision_paths = list(pointer.revision_paths)
        else:
            revision = 1
            revision_paths = []
        destination = paths.plan_pointer.parent / f"revision-{revision:04d}.md"
        if destination.exists():
            raise AssetTaskRuntimeError(f"Plan revision already exists: {destination}")
        atomic_write_text(destination, plan_text.rstrip() + "\n")
        revision_paths.append(str(destination))
        pointer = AgentPlanPointer(
            current_revision=revision,
            current_plan_path=str(destination),
            revision_paths=revision_paths,
        )
        atomic_write_json(paths.plan_pointer, pointer)
        return pointer


def _states_by_id(state: AssetTaskRunState) -> dict[str, AssetTaskWorkItemState]:
    return {item.work_item_id: item for item in state.work_items}


def _inventory_by_id(inventory: AssetTaskInventory) -> dict[str, AssetTaskWorkItem]:
    return {item.work_item_id: item for item in inventory.work_items}


def _dependencies_satisfied(
    item: AssetTaskWorkItem,
    inventory: AssetTaskInventory,
    state: AssetTaskRunState,
    task_catalog: TaskCatalog,
) -> tuple[bool, list[str]]:
    tasks_by_id = {task.task_id: task for task in task_catalog.tasks}
    task = tasks_by_id[item.task_id]
    if not task.depends_on:
        return True, []
    states = _states_by_id(state)
    items_by_id = _inventory_by_id(inventory)
    blockers: list[str] = []
    for dependency_task_id in task.depends_on:
        dependency_task = tasks_by_id[dependency_task_id]
        if dependency_task.manifest_id == item.manifest_id:
            dependency_id = (
                f"{dependency_task.task_id}:{item.manifest_id}:{item.asset_id}"
            )
            dependency_item = items_by_id.get(dependency_id)
        else:
            matches = [
                candidate
                for candidate in inventory.work_items
                if candidate.task_id == dependency_task.task_id
                and candidate.manifest_id == dependency_task.manifest_id
                and candidate.original_root_path == item.original_root_path
            ]
            if len(matches) > 1:
                raise AssetTaskRuntimeError(
                    "Ambiguous dependency work items for "
                    f"{item.work_item_id} -> {dependency_task.task_id}"
                )
            dependency_item = matches[0] if matches else None
        if dependency_item is None:
            raise AssetTaskRuntimeError(
                "Missing dependency work item for "
                f"{item.work_item_id} -> {dependency_task.task_id}"
            )
        dependency_state = states.get(dependency_item.work_item_id)
        if dependency_state is None:
            raise AssetTaskRuntimeError(
                f"Missing state for dependency work item: {dependency_item.work_item_id}"
            )
        if dependency_state.status not in {"completed", "waived"}:
            blockers.append(dependency_item.work_item_id)
    return not blockers, blockers


def processing_status(output_dir: str | Path) -> dict[str, object]:
    """Return aggregate status and currently eligible work items."""

    _paths, inventory, state, task_catalog, _manifest_catalog = _load_run(output_dir)
    states = _states_by_id(state)
    counts = Counter(item.status for item in state.work_items)
    eligible: list[str] = []
    for item in inventory.work_items:
        item_state = states.get(item.work_item_id)
        if item_state is None:
            raise AssetTaskRuntimeError(
                f"Missing state for inventory work item: {item.work_item_id}"
            )
        if item_state.status not in {"planned", "failed", "deferred"}:
            continue
        dependencies_ok, _blockers = _dependencies_satisfied(
            item, inventory, state, task_catalog
        )
        if dependencies_ok:
            eligible.append(item.work_item_id)
    return {
        "input_digest": state.input_digest,
        "revision": state.revision,
        "work_item_count": len(inventory.work_items),
        "status_counts": dict(sorted(counts.items())),
        "eligible_work_item_ids": eligible,
        "accepted_waiver_count": len(state.accepted_waivers),
    }


def get_work_item(
    output_dir: str | Path,
    work_item_id: str,
) -> tuple[AssetTaskWorkItem, AssetTaskWorkItemState]:
    """Read one inventory item and its mutable state."""

    _paths, inventory, state, _tasks, _manifests = _load_run(output_dir)
    item = _inventory_by_id(inventory).get(work_item_id)
    item_state = _states_by_id(state).get(work_item_id)
    if item is None or item_state is None:
        raise AssetTaskRuntimeError(f"Unknown work item: {work_item_id}")
    return item, item_state


def begin_work_item(
    output_dir: str | Path,
    work_item_id: str,
    *,
    actor: str = "agent",
) -> AssetTaskRunState:
    """Claim one eligible work item for the driving agent."""

    paths = ProcessingPaths.from_output_dir(output_dir)
    with _exclusive_lock(paths):
        _paths, inventory, state, task_catalog, _manifest_catalog = _load_run(
            paths.output_dir
        )
        if not paths.plan_pointer.is_file():
            raise AssetTaskRuntimeError("Record an agent plan before beginning work")
        item = _inventory_by_id(inventory).get(work_item_id)
        item_state = _states_by_id(state).get(work_item_id)
        if item is None or item_state is None:
            raise AssetTaskRuntimeError(f"Unknown work item: {work_item_id}")
        if item_state.status not in {"planned", "failed", "deferred"}:
            raise AssetTaskRuntimeError(
                f"Cannot begin {work_item_id} from status {item_state.status}"
            )
        dependencies_ok, blockers = _dependencies_satisfied(
            item, inventory, state, task_catalog
        )
        if not dependencies_ok:
            raise AssetTaskRuntimeError(
                f"Dependencies are incomplete for {work_item_id}: {blockers}"
            )
        previous_status = item_state.status
        item_state.status = "running"
        item_state.attempt_count += 1
        item_state.started_at = _timestamp()
        item_state.completed_at = None
        item_state.result_path = None
        item_state.validation_path = None
        item_state.error = None
        state.transitions.append(
            AssetTaskStateTransition(
                timestamp=item_state.started_at,
                work_item_id=work_item_id,
                from_status=previous_status,
                to_status="running",
                actor=actor,
                reason="Work item claimed after dependency validation.",
                attempt_count=item_state.attempt_count,
            )
        )
        return _write_state(paths.state, state)


def _read_ledger(path: Path) -> list[DecisionLedgerEntry]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AssetTaskRuntimeError(
            f"Cannot read decision ledger {path}: {exc}"
        ) from exc
    entries: list[DecisionLedgerEntry] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entries.append(DecisionLedgerEntry.model_validate_json(line))
        except (ValueError, ValidationError) as exc:
            raise AssetTaskRuntimeError(
                f"Invalid decision ledger line {line_number}: {exc}"
            ) from exc
    return entries


def _paths_match(reference: str | None, expected: Path) -> bool:
    if not reference:
        return False
    return Path(reference).expanduser().resolve() == expected


def _matching_index_entry(
    entry: ResultIndexEntry,
    *,
    result_file: Path,
    validation_file: Path,
) -> bool:
    return (
        entry.status == "completed"
        and _paths_match(entry.result_path, result_file)
        and _paths_match(entry.validation_path, validation_file)
    )


def _complete_state_from_indexed_result(
    state: AssetTaskRunState,
    item_state: AssetTaskWorkItemState,
    *,
    result_file: Path,
    validation_file: Path,
    actor: str,
    reason: str,
) -> None:
    completed_at = _timestamp()
    previous_status = item_state.status
    item_state.status = "completed"
    item_state.completed_at = completed_at
    item_state.result_path = str(result_file)
    item_state.validation_path = str(validation_file)
    item_state.error = None
    state.transitions.append(
        AssetTaskStateTransition(
            timestamp=completed_at,
            work_item_id=item_state.work_item_id,
            from_status=previous_status,
            to_status="completed",
            actor=actor,
            reason=reason,
            attempt_count=item_state.attempt_count,
            result_path=str(result_file),
            validation_path=str(validation_file),
        )
    )


def _recover_indexed_completions(
    paths: ProcessingPaths,
    inventory: AssetTaskInventory,
    state: AssetTaskRunState,
    task_catalog: TaskCatalog,
    *,
    actor: str = "runtime-recovery",
) -> AssetTaskRunState:
    """Repair items committed to durable indexes before the final state write."""

    index = _load_model(paths.results_index, AssetTaskResultsIndex)
    index_by_id = {entry.work_item_id: entry for entry in index.entries}
    ledger_by_id = {entry.work_item_id: entry for entry in _read_ledger(paths.ledger)}
    items_by_id = _inventory_by_id(inventory)
    tasks_by_id = {task.task_id: task for task in task_catalog.tasks}
    recovered = False
    for item_state in state.work_items:
        if item_state.status != "running":
            continue
        item = items_by_id.get(item_state.work_item_id)
        if item is None:
            raise AssetTaskRuntimeError(
                f"Missing inventory item for state: {item_state.work_item_id}"
            )
        task = tasks_by_id.get(item.task_id)
        if task is None:
            raise AssetTaskRuntimeError(
                f"Missing task catalog entry for work item: {item_state.work_item_id}"
            )
        index_entry = index_by_id.get(item_state.work_item_id)
        ledger_entry = ledger_by_id.get(item_state.work_item_id)
        if index_entry is None and ledger_entry is None:
            continue
        if (
            index_entry is None
            or ledger_entry is None
            or index_entry.status != "completed"
            or not index_entry.result_path
            or not index_entry.validation_path
            or ledger_entry.validation_status != "passed"
        ):
            raise AssetTaskRuntimeError(
                "Incomplete indexed completion for running work item "
                f"{item_state.work_item_id}"
            )
        result_file = Path(index_entry.result_path).expanduser().resolve()
        validation_file = Path(index_entry.validation_path).expanduser().resolve()
        result = _load_model(result_file, AssetTaskResult)
        if result.work_item_id != item_state.work_item_id:
            raise AssetTaskRuntimeError(
                "Indexed result identity does not match running work item "
                f"{item_state.work_item_id}"
            )
        if result.domain != task.domain:
            raise AssetTaskRuntimeError(
                f"Indexed result domain does not match task catalog for {item.task_id}"
            )
        if result.original_root_path != item.original_root_path:
            raise AssetTaskRuntimeError(
                "Indexed result original_root_path does not match inventory"
            )
        if item.working_usd_path and (
            not result.working_usd_path
            or Path(result.working_usd_path).expanduser().resolve()
            != Path(item.working_usd_path).expanduser().resolve()
        ):
            raise AssetTaskRuntimeError(
                "Indexed result working_usd_path does not match inventory"
            )
        if result.mapping.unresolved_paths:
            raise AssetTaskRuntimeError(
                f"Indexed result contains unresolved source mappings for {item.task_id}"
            )
        expected_request_digest = state.task_request_digests.get(item.task_id)
        if (
            expected_request_digest
            and result.provenance.task_request_digest != expected_request_digest
        ):
            raise AssetTaskRuntimeError(
                "Indexed result does not cite the frozen task request digest"
            )
        if ledger_entry.task_id != item.task_id or ledger_entry.domain != task.domain:
            raise AssetTaskRuntimeError(
                "Indexed ledger entry task/domain does not match inventory"
            )
        if (
            expected_request_digest
            and ledger_entry.task_request_digest != expected_request_digest
        ):
            raise AssetTaskRuntimeError(
                "Indexed ledger entry does not cite the frozen task request digest"
            )
        if set(ledger_entry.informed_by_results) != set(
            result.provenance.informed_by_results
        ):
            raise AssetTaskRuntimeError("Indexed result and ledger provenance differ")
        states_by_id = _states_by_id(state)
        for prior_id in result.provenance.informed_by_results:
            prior_state = states_by_id.get(prior_id)
            if prior_state is None or prior_state.status != "completed":
                raise AssetTaskRuntimeError(
                    f"Indexed result cites an incomplete prior result: {prior_id}"
                )
        try:
            validation = load_json(validation_file)
        except (OSError, ValueError) as exc:
            raise AssetTaskRuntimeError(
                f"Invalid validation report at {validation_file}: {exc}"
            ) from exc
        if not isinstance(validation, dict) or validation.get("passed") is not True:
            raise AssetTaskRuntimeError(
                f"Indexed validation did not pass for {item_state.work_item_id}"
            )
        _complete_state_from_indexed_result(
            state,
            item_state,
            result_file=result_file,
            validation_file=validation_file,
            actor=actor,
            reason=(
                "Recovered completed work item already present in the decision "
                "ledger and results index."
            ),
        )
        recovered = True
    if recovered:
        return _write_state(paths.state, state)
    return state


def commit_work_item(
    output_dir: str | Path,
    work_item_id: str,
    *,
    result_path: str | Path,
    validation_path: str | Path,
    ledger_entry_path: str | Path,
    actor: str = "agent",
) -> AssetTaskRunState:
    """Validate and atomically index one completed work item."""

    paths = ProcessingPaths.from_output_dir(output_dir)
    result_file = Path(result_path).expanduser().resolve()
    validation_file = Path(validation_path).expanduser().resolve()
    ledger_entry_file = Path(ledger_entry_path).expanduser().resolve()
    result = _load_model(result_file, AssetTaskResult)
    ledger_entry = _load_model(ledger_entry_file, DecisionLedgerEntry)
    try:
        validation = load_json(validation_file)
    except (OSError, ValueError) as exc:
        raise AssetTaskRuntimeError(
            f"Invalid validation report at {validation_file}: {exc}"
        ) from exc
    if not isinstance(validation, dict):
        raise AssetTaskRuntimeError(
            f"Invalid validation report at {validation_file}: expected object"
        )
    if validation.get("passed") is not True:
        raise AssetTaskRuntimeError(f"Validation did not pass for {work_item_id}")

    with _exclusive_lock(paths):
        _paths, inventory, state, task_catalog, _manifest_catalog = _load_run(
            paths.output_dir
        )
        item = _inventory_by_id(inventory).get(work_item_id)
        item_state = _states_by_id(state).get(work_item_id)
        if item is None or item_state is None:
            raise AssetTaskRuntimeError(f"Unknown work item: {work_item_id}")
        if item_state.status not in {"running", "completed"}:
            raise AssetTaskRuntimeError(
                f"Cannot commit {work_item_id} from status {item_state.status}"
            )
        task = next(task for task in task_catalog.tasks if task.task_id == item.task_id)
        if result.work_item_id != work_item_id:
            raise AssetTaskRuntimeError("Result identity does not match work item")
        if result.domain != task.domain:
            raise AssetTaskRuntimeError("Result domain does not match task catalog")
        if result.original_root_path != item.original_root_path:
            raise AssetTaskRuntimeError(
                "Result original_root_path does not match inventory"
            )
        if item.working_usd_path and (
            not result.working_usd_path
            or Path(result.working_usd_path).expanduser().resolve()
            != Path(item.working_usd_path).expanduser().resolve()
        ):
            raise AssetTaskRuntimeError(
                "Result working_usd_path does not match inventory"
            )
        if result.mapping.unresolved_paths:
            raise AssetTaskRuntimeError("Result contains unresolved source mappings")
        expected_request_digest = state.task_request_digests.get(item.task_id)
        if (
            expected_request_digest
            and result.provenance.task_request_digest != expected_request_digest
        ):
            raise AssetTaskRuntimeError(
                "Result does not cite the frozen task request digest"
            )
        for label, output_reference in result.domain_outputs.items():
            output_path = _resolve_reference(output_reference, result_file)
            if not output_path.exists():
                raise AssetTaskRuntimeError(
                    f"Result domain output {label} does not exist: {output_path}"
                )
        if ledger_entry.work_item_id != work_item_id:
            raise AssetTaskRuntimeError(
                "Ledger entry identity does not match work item"
            )
        if ledger_entry.task_id != item.task_id or ledger_entry.domain != task.domain:
            raise AssetTaskRuntimeError(
                "Ledger entry task/domain does not match inventory"
            )
        if ledger_entry.validation_status != "passed":
            raise AssetTaskRuntimeError(
                "Ledger entry does not record passed validation"
            )
        if (
            expected_request_digest
            and ledger_entry.task_request_digest != expected_request_digest
        ):
            raise AssetTaskRuntimeError(
                "Ledger entry does not cite the frozen task request digest"
            )
        if set(ledger_entry.informed_by_results) != set(
            result.provenance.informed_by_results
        ):
            raise AssetTaskRuntimeError("Result and ledger provenance differ")
        pointer = _load_model(paths.plan_pointer, AgentPlanPointer)
        if ledger_entry.agent_plan_revision != result.provenance.agent_plan_revision:
            raise AssetTaskRuntimeError(
                "Ledger entry plan revision does not match result provenance"
            )
        if result.provenance.agent_plan_revision > pointer.current_revision:
            raise AssetTaskRuntimeError(
                "Result references an unavailable plan revision"
            )
        for prior_id in result.provenance.informed_by_results:
            prior_state = _states_by_id(state).get(prior_id)
            if prior_state is None or prior_state.status != "completed":
                raise AssetTaskRuntimeError(
                    f"Result cites an incomplete prior result: {prior_id}"
                )

        index = _load_model(paths.results_index, AssetTaskResultsIndex)
        existing_index_entry = next(
            (entry for entry in index.entries if entry.work_item_id == work_item_id),
            None,
        )
        if existing_index_entry is not None and not _matching_index_entry(
            existing_index_entry,
            result_file=result_file,
            validation_file=validation_file,
        ):
            raise AssetTaskRuntimeError(f"Result index already contains {work_item_id}")
        ledger_entries = _read_ledger(paths.ledger)
        existing_ledger_entry = next(
            (entry for entry in ledger_entries if entry.work_item_id == work_item_id),
            None,
        )
        if existing_ledger_entry is not None and (
            existing_ledger_entry.model_dump(mode="json")
            != ledger_entry.model_dump(mode="json")
        ):
            raise AssetTaskRuntimeError(
                f"Decision ledger already contains {work_item_id}"
            )

        if existing_ledger_entry is None:
            ledger_entries.append(ledger_entry)
            atomic_write_text(
                paths.ledger,
                "".join(entry.model_dump_json() + "\n" for entry in ledger_entries),
            )
        if existing_index_entry is None:
            index.entries.append(
                ResultIndexEntry(
                    work_item_id=work_item_id,
                    status="completed",
                    result_path=str(result_file),
                    validation_path=str(validation_file),
                )
            )
            atomic_write_json(paths.results_index, index)

        if item_state.status == "completed":
            if not (
                _paths_match(item_state.result_path, result_file)
                and _paths_match(item_state.validation_path, validation_file)
            ):
                raise AssetTaskRuntimeError(
                    f"Work item {work_item_id} is already completed differently"
                )
            return state

        _complete_state_from_indexed_result(
            state,
            item_state,
            result_file=result_file,
            validation_file=validation_file,
            actor=actor,
            reason=(
                "Result and domain validation accepted."
                if existing_index_entry is None and existing_ledger_entry is None
                else "Recovered completed work item already present in durable indexes."
            ),
        )
        return _write_state(paths.state, state)


def fail_work_item(
    output_dir: str | Path,
    work_item_id: str,
    *,
    reason: str,
    actor: str = "agent",
) -> AssetTaskRunState:
    """Record a failed attempt while leaving the item retryable."""

    if not reason.strip():
        raise AssetTaskRuntimeError("A failure reason is required")
    paths = ProcessingPaths.from_output_dir(output_dir)
    with _exclusive_lock(paths):
        _paths, _inventory, state, _tasks, _manifests = _load_run(paths.output_dir)
        item_state = _states_by_id(state).get(work_item_id)
        if item_state is None:
            raise AssetTaskRuntimeError(f"Unknown work item: {work_item_id}")
        if item_state.status != "running":
            raise AssetTaskRuntimeError(
                f"Cannot fail {work_item_id} from status {item_state.status}"
            )
        item_state.status = "failed"
        item_state.error = reason
        state.transitions.append(
            AssetTaskStateTransition(
                timestamp=_timestamp(),
                work_item_id=work_item_id,
                from_status="running",
                to_status="failed",
                actor=actor,
                reason=reason,
                attempt_count=item_state.attempt_count,
            )
        )
        return _write_state(paths.state, state)


def waive_work_item(
    output_dir: str | Path,
    work_item_id: str,
    *,
    reason: str,
    accepted_by: str,
    actor: str = "agent",
) -> AssetTaskRunState:
    """Record an explicit accepted waiver for an uncompleted item."""

    if not reason.strip() or not accepted_by.strip():
        raise AssetTaskRuntimeError("Waiver reason and accepted_by are required")
    paths = ProcessingPaths.from_output_dir(output_dir)
    with _exclusive_lock(paths):
        _paths, inventory, state, _tasks, _manifests = _load_run(paths.output_dir)
        item = _inventory_by_id(inventory).get(work_item_id)
        item_state = _states_by_id(state).get(work_item_id)
        if item is None or item_state is None:
            raise AssetTaskRuntimeError(f"Unknown work item: {work_item_id}")
        if item_state.status in {"running", "completed", "waived"}:
            raise AssetTaskRuntimeError(
                f"Cannot waive {work_item_id} from status {item_state.status}"
            )
        waiver = AcceptedWaiver(
            waiver_id=f"waiver-{uuid4()}",
            work_item_id=work_item_id,
            reason=reason,
            accepted_by=accepted_by,
            accepted_at=_timestamp(),
        )
        previous_status = item_state.status
        item_state.status = "waived"
        item_state.waiver_id = waiver.waiver_id
        item_state.completed_at = None
        item_state.error = None
        state.accepted_waivers.append(waiver)
        state.transitions.append(
            AssetTaskStateTransition(
                timestamp=waiver.accepted_at,
                work_item_id=work_item_id,
                from_status=previous_status,
                to_status="waived",
                actor=actor,
                reason=reason,
                attempt_count=item_state.attempt_count,
            )
        )
        index = _load_model(paths.results_index, AssetTaskResultsIndex)
        index.entries.append(
            ResultIndexEntry(work_item_id=work_item_id, status="waived")
        )
        atomic_write_json(paths.results_index, index)
        return _write_state(paths.state, state)


def finalize_processing_run(output_dir: str | Path) -> ProcessingPhaseResult:
    """Seal ``processing_result.json`` after every required item is resolved."""

    paths = ProcessingPaths.from_output_dir(output_dir)
    with _exclusive_lock(paths):
        _paths, inventory, state, task_catalog, manifest_catalog = _load_run(
            paths.output_dir
        )
        state = _recover_indexed_completions(paths, inventory, state, task_catalog)
        pointer = _load_model(paths.plan_pointer, AgentPlanPointer)
        index = _load_model(paths.results_index, AssetTaskResultsIndex)
        states = _states_by_id(state)
        unresolved_required = [
            item.work_item_id
            for item in inventory.work_items
            if item.required
            and states[item.work_item_id].status not in {"completed", "waived"}
        ]
        if unresolved_required:
            raise AssetTaskRuntimeError(
                "Required work items remain unresolved: "
                + ", ".join(unresolved_required[:20])
                + (" ..." if len(unresolved_required) > 20 else "")
            )

        manifest_catalog_path = Path(state.manifest_catalog_path).expanduser().resolve()
        task_catalog_path = Path(state.task_catalog_path).expanduser().resolve()
        artifact_paths: set[Path] = {
            Path(state.task_catalog_path).expanduser().resolve(),
            manifest_catalog_path,
            paths.inventory,
            paths.state,
            paths.plan_pointer,
            paths.ledger,
            paths.results_index,
        }
        for entry in manifest_catalog.manifests:
            artifact_paths.add(_resolve_reference(entry.path, manifest_catalog_path))
        for task in task_catalog.tasks:
            artifact_paths.add(_resolve_reference(task.request_path, task_catalog_path))
        artifact_paths.update(
            Path(reference).expanduser().resolve()
            for reference in pointer.revision_paths
        )
        for ledger_entry in _read_ledger(paths.ledger):
            artifact_paths.update(
                resolve_artifact_path(reference, base_dir=paths.ledger.parent)
                for reference in ledger_entry.artifact_paths
            )
        for entry in index.entries:
            if entry.result_path:
                result_path = Path(entry.result_path).expanduser().resolve()
                artifact_paths.add(result_path)
                result = _load_model(result_path, AssetTaskResult)
                artifact_paths.update(
                    _resolve_reference(reference, result_path)
                    for reference in result.domain_outputs.values()
                )
            if entry.validation_path:
                artifact_paths.add(Path(entry.validation_path).expanduser().resolve())

        required_items = [item for item in inventory.work_items if item.required]
        optional_items = [item for item in inventory.work_items if not item.required]
        completed_required = sum(
            states[item.work_item_id].status == "completed" for item in required_items
        )
        completed_optional = sum(
            states[item.work_item_id].status == "completed" for item in optional_items
        )
        result = ProcessingPhaseResult(
            success=True,
            input_digest=state.input_digest,
            task_catalog_path=state.task_catalog_path,
            manifest_catalog_path=state.manifest_catalog_path,
            asset_task_inventory_path=str(paths.inventory),
            work_item_state_path=str(paths.state),
            agent_plan_pointer_path=str(paths.plan_pointer),
            decision_ledger_path=str(paths.ledger),
            results_index_path=str(paths.results_index),
            task_request_digests=state.task_request_digests,
            required_work_item_count=len(required_items),
            completed_required_count=completed_required,
            optional_work_item_count=len(optional_items),
            completed_optional_count=completed_optional,
            accepted_waivers=state.accepted_waivers,
            artifact_paths=sorted(map(str, artifact_paths)),
            completion_policy_satisfied=True,
        )
        return seal_phase_result(result, paths.phase_result)
