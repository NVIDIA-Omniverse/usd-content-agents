# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic handoff gates for the three large-scene phases."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from content_agent_workflows.asset_task_processing.contracts import (
    AgentPlanPointer,
    AssetTaskInventory,
    AssetTaskResult,
    AssetTaskResultsIndex,
    AssetTaskRunState,
    DecisionLedgerEntry,
    ProcessingPhaseResult,
    TaskCatalog,
)
from content_agent_workflows.common.artifacts import (
    artifact_set_digest,
    file_sha256,
    load_json,
    phase_result_digest,
    resolve_artifact_path,
)
from content_agent_workflows.scene_collection.contracts import (
    CollectionPhaseResult,
    DomainCollectionResult,
)
from content_agent_workflows.scene_decomposition.manifest import (
    DecompositionPhaseResult,
    ManifestCatalog,
    SceneDecompositionManifest,
)

from .models import HandoffValidationReport, LargeSceneRun, PhaseName


def _resolve(path: str | Path, result_path: Path) -> Path:
    return resolve_artifact_path(path, base_dir=result_path.parent)


def _load_model[ModelT: BaseModel](
    path: Path,
    model_type: type[ModelT],
    errors: list[str],
    label: str,
) -> ModelT | None:
    try:
        return model_type.model_validate(load_json(path))
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid {label} at {path}: {exc}")
        return None


def _require_artifact(
    path: str | Path,
    *,
    result_path: Path,
    artifact_paths: set[Path],
    errors: list[str],
    label: str,
) -> Path:
    resolved = _resolve(path, result_path)
    if not resolved.exists():
        errors.append(f"Missing {label}: {resolved}")
    if resolved not in artifact_paths:
        errors.append(f"{label} is not covered by output_digest: {resolved}")
    return resolved


def _report_passed(path: Path, errors: list[str], label: str) -> None:
    try:
        report = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid {label} at {path}: {exc}")
        return
    if report.get("passed") is not True:
        errors.append(f"{label} did not record passed=true: {path}")


def _validate_decomposition(
    result: DecompositionPhaseResult,
    run: LargeSceneRun,
    result_path: Path,
    artifact_paths: set[Path],
    errors: list[str],
    warnings: list[str],
) -> None:
    del warnings
    if not result.manifest_catalog_path:
        errors.append("decomposition_result is missing manifest_catalog_path")
        return

    catalog_path = _require_artifact(
        result.manifest_catalog_path,
        result_path=result_path,
        artifact_paths=artifact_paths,
        errors=errors,
        label="manifest catalog",
    )
    catalog = _load_model(catalog_path, ManifestCatalog, errors, "manifest catalog")
    if catalog is None:
        return

    source_scene = Path(run.source_scene).expanduser().resolve()
    result_source_scene = Path(result.source_scene).expanduser().resolve()
    catalog_source_scene = Path(catalog.original_usd_path).expanduser().resolve()
    if result_source_scene != source_scene:
        errors.append(
            f"decomposition source scene {result_source_scene} does not match {source_scene}"
        )
    if catalog_source_scene != source_scene:
        errors.append(
            f"manifest catalog source scene {catalog_source_scene} does not match {source_scene}"
        )
    try:
        source_identity_digest = artifact_set_digest([source_scene])
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot recompute source identity: {exc}")
    else:
        if result.source_identity_digest != source_identity_digest:
            errors.append("decomposition source_identity_digest is stale")
        if catalog.source_identity_digest != source_identity_digest:
            errors.append("manifest catalog source_identity_digest is stale")

    catalog_ids = [entry.manifest_id for entry in catalog.manifests]
    if len(catalog_ids) != len(set(catalog_ids)):
        errors.append("manifest catalog contains duplicate manifest_id values")
    if not catalog.manifests:
        errors.append("manifest catalog has no finalized views")

    catalog_manifest_paths: set[Path] = set()
    expected_extracts: set[Path] = set()
    for entry in catalog.manifests:
        manifest_path = _require_artifact(
            entry.path,
            result_path=catalog_path,
            artifact_paths=artifact_paths,
            errors=errors,
            label=f"manifest {entry.manifest_id}",
        )
        catalog_manifest_paths.add(manifest_path)
        if not entry.finalized:
            errors.append(f"manifest {entry.manifest_id} is not finalized")
        if manifest_path.is_file():
            try:
                actual_digest = file_sha256(manifest_path)
            except OSError as exc:
                errors.append(f"Cannot digest manifest {manifest_path}: {exc}")
            else:
                if entry.manifest_digest != actual_digest:
                    errors.append(f"manifest digest mismatch: {manifest_path}")

        manifest = _load_model(
            manifest_path,
            SceneDecompositionManifest,
            errors,
            f"manifest {entry.manifest_id}",
        )
        if manifest is None:
            continue
        if Path(manifest.original_usd_path).expanduser().resolve() != source_scene:
            errors.append(
                f"manifest {entry.manifest_id} belongs to another source scene"
            )
        asset_ids = [asset.asset_id for asset in manifest.assets]
        if len(asset_ids) != len(set(asset_ids)):
            errors.append(f"manifest {entry.manifest_id} has duplicate asset_id values")
        original_roots = [asset.original_root_path for asset in manifest.assets]
        if len(original_roots) != len(set(original_roots)):
            errors.append(
                f"manifest {entry.manifest_id} has duplicate original asset roots"
            )
        assets_by_id = {asset.asset_id: asset for asset in manifest.assets}
        assets_by_root = {asset.original_root_path: asset for asset in manifest.assets}
        for group in manifest.instance_groups:
            representative = assets_by_id.get(group.representative_asset_id or "")
            if representative is None or not representative.processable:
                errors.append(
                    f"instance group {entry.manifest_id}:{group.group_id} has no "
                    "processable representative"
                )
            for member_path in group.member_paths:
                member = assets_by_root.get(member_path)
                if (
                    member is not None
                    and member.asset_id != group.representative_asset_id
                    and member.processable
                ):
                    errors.append(
                        f"non-representative instance member is processable: "
                        f"{entry.manifest_id}:{member.asset_id}"
                    )
        for asset in manifest.processable_assets:
            if asset.skip_reason:
                errors.append(
                    f"processable asset {entry.manifest_id}:{asset.asset_id} has a skip_reason"
                )
            if asset.working_usd_path:
                working_path = _resolve(asset.working_usd_path, manifest_path)
                expected_extracts.add(working_path)
                if working_path not in artifact_paths:
                    errors.append(
                        "Extracted working USD is not covered by output_digest: "
                        f"{working_path}"
                    )

    declared_manifest_paths = {
        _resolve(path, result_path) for path in result.manifest_paths
    }
    if declared_manifest_paths != catalog_manifest_paths:
        errors.append(
            "decomposition_result manifest_paths do not match manifest_catalog"
        )

    declared_extracts = {
        _resolve(path, result_path) for path in result.extracted_asset_paths
    }
    if not expected_extracts.issubset(declared_extracts):
        errors.append(
            "decomposition_result extracted_asset_paths omit processable working USDs: "
            f"{sorted(map(str, expected_extracts - declared_extracts))}"
        )
    for extracted_path in declared_extracts:
        if not extracted_path.is_file():
            errors.append(f"Missing extracted asset: {extracted_path}")
            continue
        try:
            is_empty = extracted_path.stat().st_size == 0
        except OSError as exc:
            errors.append(f"Cannot inspect extracted asset {extracted_path}: {exc}")
        else:
            if is_empty:
                errors.append(f"Extracted asset is empty: {extracted_path}")
        if extracted_path not in artifact_paths:
            errors.append(
                f"Extracted asset is not covered by output_digest: {extracted_path}"
            )


def _load_manifest_assets(
    catalog: ManifestCatalog,
    catalog_path: Path,
    artifact_paths: set[Path],
    errors: list[str],
) -> dict[str, set[str]]:
    assets_by_manifest: dict[str, set[str]] = {}
    for entry in catalog.manifests:
        manifest_path = _require_artifact(
            entry.path,
            result_path=catalog_path,
            artifact_paths=artifact_paths,
            errors=errors,
            label=f"manifest {entry.manifest_id}",
        )
        if manifest_path.is_file():
            try:
                manifest_digest = file_sha256(manifest_path)
            except OSError as exc:
                errors.append(f"Cannot digest manifest {manifest_path}: {exc}")
            else:
                if manifest_digest != entry.manifest_digest:
                    errors.append(f"manifest digest mismatch: {manifest_path}")
        manifest = _load_model(
            manifest_path,
            SceneDecompositionManifest,
            errors,
            f"manifest {entry.manifest_id}",
        )
        if manifest is not None:
            assets_by_manifest[entry.manifest_id] = {
                asset.asset_id for asset in manifest.processable_assets
            }
    return assets_by_manifest


def _validate_processing(
    result: ProcessingPhaseResult,
    run: LargeSceneRun,
    result_path: Path,
    artifact_paths: set[Path],
    errors: list[str],
    warnings: list[str],
) -> None:
    del warnings
    core_paths = {
        "task catalog": result.task_catalog_path,
        "manifest catalog": result.manifest_catalog_path,
        "asset-task inventory": result.asset_task_inventory_path,
        "work-item state": result.work_item_state_path,
        "agent plan pointer": result.agent_plan_pointer_path,
        "decision ledger": result.decision_ledger_path,
        "results index": result.results_index_path,
    }
    resolved_core = {
        label: _require_artifact(
            path,
            result_path=result_path,
            artifact_paths=artifact_paths,
            errors=errors,
            label=label,
        )
        for label, path in core_paths.items()
    }
    task_catalog = _load_model(
        resolved_core["task catalog"], TaskCatalog, errors, "task catalog"
    )
    manifest_catalog = _load_model(
        resolved_core["manifest catalog"],
        ManifestCatalog,
        errors,
        "manifest catalog",
    )
    inventory = _load_model(
        resolved_core["asset-task inventory"],
        AssetTaskInventory,
        errors,
        "asset-task inventory",
    )
    work_item_state = _load_model(
        resolved_core["work-item state"],
        AssetTaskRunState,
        errors,
        "work-item state",
    )
    agent_plan = _load_model(
        resolved_core["agent plan pointer"],
        AgentPlanPointer,
        errors,
        "agent plan pointer",
    )
    index = _load_model(
        resolved_core["results index"],
        AssetTaskResultsIndex,
        errors,
        "results index",
    )
    if any(
        item is None
        for item in (
            task_catalog,
            manifest_catalog,
            inventory,
            work_item_state,
            agent_plan,
            index,
        )
    ):
        return

    if inventory.input_digest != result.input_digest:
        errors.append("asset-task inventory input_digest does not match phase input")
    if work_item_state.input_digest != result.input_digest:
        errors.append("work-item state input_digest does not match phase input")
    if inventory.task_request_digests != work_item_state.task_request_digests:
        errors.append("inventory and work-item state task request digests differ")
    if result.task_request_digests != work_item_state.task_request_digests:
        errors.append("processing_result task request digests are stale")
    if run.additional_instructions and not result.task_request_digests:
        errors.append(
            "large-scene additional instructions were not frozen by Workflow 2"
        )
    task_by_id = {task.task_id: task for task in task_catalog.tasks}
    if result.task_request_digests and set(result.task_request_digests) != set(
        task_by_id
    ):
        errors.append("processing_result task request digest set is incomplete")
    missing_requested_tasks = set(run.requested_tasks) - set(task_by_id)
    if missing_requested_tasks:
        errors.append(
            f"task catalog is missing requested tasks: {sorted(missing_requested_tasks)}"
        )
    catalog_manifest_ids = {entry.manifest_id for entry in manifest_catalog.manifests}
    source_scene = Path(run.source_scene).expanduser().resolve()
    if Path(manifest_catalog.original_usd_path).expanduser().resolve() != source_scene:
        errors.append("processing manifest catalog belongs to another source scene")
    try:
        source_identity_digest = artifact_set_digest([source_scene])
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot recompute processing source identity: {exc}")
    else:
        if manifest_catalog.source_identity_digest != source_identity_digest:
            errors.append("processing manifest catalog source identity is stale")
    for task in task_catalog.tasks:
        if task.manifest_id not in catalog_manifest_ids:
            errors.append(
                f"task {task.task_id} references unknown manifest {task.manifest_id}"
            )
        request_path = _require_artifact(
            task.request_path,
            result_path=resolved_core["task catalog"],
            artifact_paths=artifact_paths,
            errors=errors,
            label=f"task request {task.task_id}",
        )
        if request_path.is_file():
            try:
                request_digest = file_sha256(request_path)
            except OSError as exc:
                errors.append(f"Cannot digest task request {request_path}: {exc}")
            else:
                expected_digest = result.task_request_digests.get(task.task_id)
                if expected_digest and request_digest != expected_digest:
                    errors.append(f"task request digest mismatch: {task.task_id}")
                if result.task_request_digests and expected_digest is None:
                    errors.append(
                        f"processing_result lacks task request digest: {task.task_id}"
                    )
            if run.additional_instructions:
                try:
                    task_request = load_json(request_path)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(
                        f"Cannot inspect task request guidance {request_path}: {exc}"
                    )
                else:
                    if (
                        task_request.get("additional_instructions")
                        != run.additional_instructions
                    ):
                        errors.append(
                            "task request does not carry the large-scene additional "
                            f"instructions: {task.task_id}"
                        )

    for plan_path_reference in agent_plan.revision_paths:
        plan_path = _require_artifact(
            plan_path_reference,
            result_path=resolved_core["agent plan pointer"],
            artifact_paths=artifact_paths,
            errors=errors,
            label="agent plan revision",
        )
        if plan_path.is_file():
            try:
                plan_is_empty = plan_path.stat().st_size == 0
            except OSError as exc:
                errors.append(f"Cannot inspect agent plan revision {plan_path}: {exc}")
            else:
                if plan_is_empty:
                    errors.append(f"Agent plan revision is empty: {plan_path}")

    assets_by_manifest = _load_manifest_assets(
        manifest_catalog,
        resolved_core["manifest catalog"],
        artifact_paths,
        errors,
    )
    expected_work_items = {
        f"{task.task_id}:{task.manifest_id}:{asset_id}"
        for task in task_catalog.tasks
        for asset_id in assets_by_manifest.get(task.manifest_id, set())
    }
    actual_work_items = {item.work_item_id for item in inventory.work_items}
    if actual_work_items != expected_work_items:
        missing = sorted(expected_work_items - actual_work_items)
        extra = sorted(actual_work_items - expected_work_items)
        errors.append(
            "asset-task inventory does not match the eligible work matrix; "
            f"missing={missing}, extra={extra}"
        )

    items_by_id = {item.work_item_id: item for item in inventory.work_items}
    states_by_id = {item.work_item_id: item for item in work_item_state.work_items}
    if set(states_by_id) != set(items_by_id):
        errors.append("work-item state does not match the immutable inventory")
    for item in inventory.work_items:
        if item.task_id not in task_by_id:
            errors.append(f"work item {item.work_item_id} references an unknown task")
        elif item.required != task_by_id[item.task_id].required:
            errors.append(
                f"work item required flag disagrees with task: {item.work_item_id}"
            )

    waivers_by_item = {
        waiver.work_item_id: waiver for waiver in result.accepted_waivers
    }
    state_waivers_by_item = {
        waiver.work_item_id: waiver for waiver in work_item_state.accepted_waivers
    }
    if {
        key: value.model_dump(mode="json") for key, value in waivers_by_item.items()
    } != {
        key: value.model_dump(mode="json")
        for key, value in state_waivers_by_item.items()
    }:
        errors.append("processing_result waivers do not match work-item state")
    if len(waivers_by_item) != len(result.accepted_waivers):
        errors.append("accepted waivers contain duplicate work_item_id values")
    waiver_ids = [waiver.waiver_id for waiver in result.accepted_waivers]
    if len(waiver_ids) != len(set(waiver_ids)):
        errors.append("accepted waivers contain duplicate waiver_id values")
    for work_item_id in waivers_by_item:
        if work_item_id not in items_by_id:
            errors.append(f"waiver references unknown work item {work_item_id}")

    index_by_id = {entry.work_item_id: entry for entry in index.entries}
    completed_required = 0
    completed_optional = 0
    required_items = [item for item in inventory.work_items if item.required]
    optional_items = [item for item in inventory.work_items if not item.required]
    for item in inventory.work_items:
        item_state = states_by_id.get(item.work_item_id)
        if item_state is None:
            continue
        entry = index_by_id.get(item.work_item_id)
        waiver = waivers_by_item.get(item.work_item_id)
        if item_state.status == "completed":
            if waiver is not None:
                errors.append(
                    f"completed work item also has a waiver: {item.work_item_id}"
                )
            if item.required:
                completed_required += 1
            else:
                completed_optional += 1
            if entry is None or entry.status != "completed":
                errors.append(
                    f"completed work item is missing a completed index entry: {item.work_item_id}"
                )
        elif item.required:
            if waiver is None:
                errors.append(
                    f"required work item is neither completed nor waived: {item.work_item_id}"
                )
            elif item_state.status != "waived":
                errors.append(
                    f"waived work item state is {item_state.status}: {item.work_item_id}"
                )
            if entry is None or entry.status != "waived":
                errors.append(
                    f"waived work item is missing a waived index entry: {item.work_item_id}"
                )
        if waiver is not None and item_state.waiver_id != waiver.waiver_id:
            errors.append(f"work item waiver_id mismatch: {item.work_item_id}")
        if waiver is None and item_state.waiver_id is not None:
            errors.append(f"work item has an unapproved waiver_id: {item.work_item_id}")

    if result.required_work_item_count != len(required_items):
        errors.append("processing_result required_work_item_count is incorrect")
    if result.optional_work_item_count != len(optional_items):
        errors.append("processing_result optional_work_item_count is incorrect")
    if result.completed_required_count != completed_required:
        errors.append("processing_result completed_required_count is incorrect")
    if result.completed_optional_count != completed_optional:
        errors.append("processing_result completed_optional_count is incorrect")

    completed_result_ids: set[str] = set()
    completed_results: dict[str, AssetTaskResult] = {}
    referenced_plan_revisions: set[int] = set()
    for entry in index.entries:
        item = items_by_id.get(entry.work_item_id)
        if item is None:
            errors.append(
                f"results index references unknown work item {entry.work_item_id}"
            )
            continue
        if entry.status == "waived":
            if entry.work_item_id not in waivers_by_item:
                errors.append(f"index marks unapproved waiver for {entry.work_item_id}")
            item_state = states_by_id.get(entry.work_item_id)
            if item_state is None or item_state.status != "waived":
                errors.append(
                    f"results index waiver disagrees with work-item state: "
                    f"{entry.work_item_id}"
                )
            continue
        item_state = states_by_id.get(entry.work_item_id)
        if item_state is None or item_state.status != "completed":
            errors.append(
                f"results index marks a non-completed work item completed: {entry.work_item_id}"
            )
        if not entry.result_path:
            errors.append(
                f"completed index entry has no result_path: {entry.work_item_id}"
            )
            continue
        work_result_path = _require_artifact(
            entry.result_path,
            result_path=resolved_core["results index"],
            artifact_paths=artifact_paths,
            errors=errors,
            label=f"asset-task result {entry.work_item_id}",
        )
        work_result = _load_model(
            work_result_path,
            AssetTaskResult,
            errors,
            f"asset-task result {entry.work_item_id}",
        )
        if work_result is None:
            continue
        completed_result_ids.add(entry.work_item_id)
        completed_results[entry.work_item_id] = work_result
        if work_result.work_item_id != entry.work_item_id:
            errors.append(f"asset-task result identity mismatch: {entry.work_item_id}")
        if work_result.original_root_path != item.original_root_path:
            errors.append(
                f"asset-task result original root mismatch: {entry.work_item_id}"
            )
        if item.working_usd_path and (
            not work_result.working_usd_path
            or Path(work_result.working_usd_path).expanduser().resolve()
            != Path(item.working_usd_path).expanduser().resolve()
        ):
            errors.append(
                f"asset-task result working USD mismatch: {entry.work_item_id}"
            )
        task = task_by_id.get(work_result.task_id)
        if task is not None and work_result.domain != task.domain:
            errors.append(f"asset-task result domain mismatch: {entry.work_item_id}")
        expected_request_digest = result.task_request_digests.get(work_result.task_id)
        if (
            expected_request_digest
            and work_result.provenance.task_request_digest != expected_request_digest
        ):
            errors.append(
                f"asset-task result request provenance mismatch: {entry.work_item_id}"
            )
        if work_result.mapping.unresolved_paths:
            errors.append(
                f"asset-task result has unresolved source paths: {entry.work_item_id}"
            )
        referenced_plan_revisions.add(work_result.provenance.agent_plan_revision)
        for informed_by in work_result.provenance.informed_by_results:
            informed_entry = index_by_id.get(informed_by)
            if informed_entry is None or informed_entry.status != "completed":
                errors.append(
                    f"{entry.work_item_id} cites an unavailable prior result: {informed_by}"
                )
        for output_label, output_path in work_result.domain_outputs.items():
            _require_artifact(
                output_path,
                result_path=work_result_path,
                artifact_paths=artifact_paths,
                errors=errors,
                label=f"{entry.work_item_id} domain output {output_label}",
            )
        if not entry.validation_path:
            errors.append(
                f"completed index entry has no validation_path: {entry.work_item_id}"
            )
        else:
            validation_path = _require_artifact(
                entry.validation_path,
                result_path=resolved_core["results index"],
                artifact_paths=artifact_paths,
                errors=errors,
                label=f"validation report {entry.work_item_id}",
            )
            if validation_path.is_file():
                _report_passed(
                    validation_path,
                    errors,
                    f"validation report {entry.work_item_id}",
                )

    ledger_ids: set[str] = set()
    ledger_path = resolved_core["decision ledger"]
    if ledger_path.is_file():
        try:
            ledger_lines = ledger_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            errors.append(f"Cannot read decision ledger {ledger_path}: {exc}")
        else:
            for line_number, line in enumerate(ledger_lines, start=1):
                if not line.strip():
                    continue
                try:
                    ledger_entry = DecisionLedgerEntry.model_validate_json(line)
                except (ValueError, ValidationError) as exc:
                    errors.append(
                        f"Invalid decision ledger line {line_number} in {ledger_path}: {exc}"
                    )
                    continue
                if ledger_entry.work_item_id in ledger_ids:
                    errors.append(
                        f"decision ledger has duplicate work item {ledger_entry.work_item_id}"
                    )
                ledger_ids.add(ledger_entry.work_item_id)
                ledger_result = completed_results.get(ledger_entry.work_item_id)
                if ledger_result is None:
                    errors.append(
                        "decision ledger references unknown or incomplete result: "
                        f"{ledger_entry.work_item_id}"
                    )
                else:
                    if (
                        ledger_entry.task_id != ledger_result.task_id
                        or ledger_entry.domain != ledger_result.domain
                    ):
                        errors.append(
                            "decision ledger task/domain mismatch for "
                            f"{ledger_entry.work_item_id}"
                        )
                    if (
                        ledger_entry.agent_plan_revision
                        != ledger_result.provenance.agent_plan_revision
                    ):
                        errors.append(
                            "decision ledger plan revision mismatch for "
                            f"{ledger_entry.work_item_id}"
                        )
                    if set(ledger_entry.informed_by_results) != set(
                        ledger_result.provenance.informed_by_results
                    ):
                        errors.append(
                            "decision ledger provenance mismatch for "
                            f"{ledger_entry.work_item_id}"
                        )
                referenced_plan_revisions.add(ledger_entry.agent_plan_revision)
                for artifact_reference in ledger_entry.artifact_paths:
                    _require_artifact(
                        artifact_reference,
                        result_path=ledger_path,
                        artifact_paths=artifact_paths,
                        errors=errors,
                        label=(f"decision ledger artifact {ledger_entry.work_item_id}"),
                    )
                if ledger_entry.validation_status != "passed":
                    errors.append(
                        f"decision ledger validation failed for {ledger_entry.work_item_id}"
                    )
                expected_request_digest = result.task_request_digests.get(
                    ledger_entry.task_id
                )
                if (
                    expected_request_digest
                    and ledger_entry.task_request_digest != expected_request_digest
                ):
                    errors.append(
                        "decision ledger request provenance mismatch for "
                        f"{ledger_entry.work_item_id}"
                    )
    if not completed_result_ids.issubset(ledger_ids):
        errors.append(
            "decision ledger is missing completed results: "
            f"{sorted(completed_result_ids - ledger_ids)}"
        )
    if (
        referenced_plan_revisions
        and max(referenced_plan_revisions) > agent_plan.current_revision
    ):
        errors.append("result or ledger references an unavailable agent plan revision")


def _validate_collection(
    result: CollectionPhaseResult,
    run: LargeSceneRun,
    result_path: Path,
    artifact_paths: set[Path],
    errors: list[str],
    warnings: list[str],
) -> None:
    del warnings
    processing_state = run.phases["asset_task_processing"]
    if processing_state.result_path:
        processing_result_path = _resolve(processing_state.result_path, result_path)
        processing_result = _load_model(
            processing_result_path,
            ProcessingPhaseResult,
            errors,
            "processing phase result",
        )
        if processing_result is not None and processing_result.task_request_digests:
            _require_artifact(
                processing_result_path,
                result_path=result_path,
                artifact_paths=artifact_paths,
                errors=errors,
                label="processing phase result",
            )
            task_catalog_path = _require_artifact(
                processing_result.task_catalog_path,
                result_path=processing_result_path,
                artifact_paths=artifact_paths,
                errors=errors,
                label="collection task catalog",
            )
            task_catalog = _load_model(
                task_catalog_path,
                TaskCatalog,
                errors,
                "collection task catalog",
            )
            if task_catalog is not None:
                for task in task_catalog.tasks:
                    task_request_path = _require_artifact(
                        task.request_path,
                        result_path=task_catalog_path,
                        artifact_paths=artifact_paths,
                        errors=errors,
                        label=f"collection task request {task.task_id}",
                    )
                    if task_request_path.is_file():
                        expected_digest = processing_result.task_request_digests.get(
                            task.task_id
                        )
                        if expected_digest:
                            try:
                                task_request_digest = file_sha256(task_request_path)
                            except OSError as exc:
                                errors.append(
                                    "Cannot hash collection task request "
                                    f"{task.task_id}: {exc}"
                                )
                            else:
                                if task_request_digest != expected_digest:
                                    errors.append(
                                        "collection task request digest mismatch: "
                                        f"{task.task_id}"
                                    )
    domain_results: list[DomainCollectionResult] = []
    seen_domains: set[str] = set()
    for domain_result_reference in result.domain_result_paths:
        domain_result_path = _require_artifact(
            domain_result_reference,
            result_path=result_path,
            artifact_paths=artifact_paths,
            errors=errors,
            label="domain collection result",
        )
        domain_result = _load_model(
            domain_result_path,
            DomainCollectionResult,
            errors,
            "domain collection result",
        )
        if domain_result is None:
            continue
        domain_results.append(domain_result)
        if domain_result.domain in seen_domains:
            errors.append(f"duplicate domain collection result: {domain_result.domain}")
        seen_domains.add(domain_result.domain)
        try:
            computed_digest = phase_result_digest(
                domain_result, result_path=domain_result_path
            )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(
                f"Cannot recompute domain output digest for {domain_result.domain}: {exc}"
            )
        else:
            if domain_result.output_digest != computed_digest:
                errors.append(
                    f"domain output digest mismatch for {domain_result.domain}"
                )
        domain_artifacts = {
            _resolve(path, domain_result_path) for path in domain_result.artifact_paths
        }
        missing_from_phase = domain_artifacts - artifact_paths
        if missing_from_phase:
            errors.append(
                f"domain {domain_result.domain} artifacts are not covered by collection digest: "
                f"{sorted(map(str, missing_from_phase))}"
            )
        if domain_result.required:
            if domain_result.status != "completed":
                errors.append(
                    f"required domain {domain_result.domain} is {domain_result.status}"
                )
            if not domain_result.validation_passed:
                errors.append(
                    f"required domain {domain_result.domain} did not pass validation"
                )
            if domain_result.unresolved_issues:
                errors.append(
                    f"required domain {domain_result.domain} has unresolved issues"
                )
        for label, path in (
            ("collection report", domain_result.collection_report_path),
            ("harmonization report", domain_result.harmonization_report_path),
            ("validation report", domain_result.validation_report_path),
        ):
            resolved = _require_artifact(
                path,
                result_path=domain_result_path,
                artifact_paths=artifact_paths,
                errors=errors,
                label=f"{domain_result.domain} {label}",
            )
            if label == "validation report" and resolved.is_file():
                _report_passed(
                    resolved,
                    errors,
                    f"{domain_result.domain} validation report",
                )
        for output_path in domain_result.output_paths:
            _require_artifact(
                output_path,
                result_path=domain_result_path,
                artifact_paths=artifact_paths,
                errors=errors,
                label=f"{domain_result.domain} output",
            )

    requested_domains = set(run.requested_tasks)
    required_domains = [domain for domain in domain_results if domain.required]
    required_domain_names = {domain.domain for domain in required_domains}
    missing_domains = sorted(requested_domains - seen_domains)
    if missing_domains:
        errors.append(f"collection is missing requested domains: {missing_domains}")
    if required_domain_names != requested_domains:
        errors.append(
            "required collection domains do not match the run request; "
            f"expected={sorted(requested_domains)}, "
            f"actual={sorted(required_domain_names)}"
        )
    completed_required_domains = [
        domain
        for domain in required_domains
        if domain.status == "completed" and domain.validation_passed
    ]
    if result.required_domain_count != len(requested_domains):
        errors.append("collection_result required_domain_count is incorrect")
    if result.completed_required_domain_count != len(completed_required_domains):
        errors.append("collection_result completed_required_domain_count is incorrect")

    topology_report = _require_artifact(
        result.topology_report_path,
        result_path=result_path,
        artifact_paths=artifact_paths,
        errors=errors,
        label="topology report",
    )
    if topology_report.is_file():
        _report_passed(topology_report, errors, "topology report")
    if result.composition_path:
        _require_artifact(
            result.composition_path,
            result_path=result_path,
            artifact_paths=artifact_paths,
            errors=errors,
            label="composed output",
        )
    if result.cross_domain_validation_path:
        cross_domain_report = _require_artifact(
            result.cross_domain_validation_path,
            result_path=result_path,
            artifact_paths=artifact_paths,
            errors=errors,
            label="cross-domain validation report",
        )
        if cross_domain_report.is_file():
            _report_passed(
                cross_domain_report, errors, "cross-domain validation report"
            )
    if not result.final_output_paths:
        errors.append("collection_result has no final_output_paths")
    for final_output in result.final_output_paths:
        _require_artifact(
            final_output,
            result_path=result_path,
            artifact_paths=artifact_paths,
            errors=errors,
            label="final output",
        )


_RESULT_MODELS: dict[PhaseName, type[BaseModel]] = {
    "decomposition": DecompositionPhaseResult,
    "asset_task_processing": ProcessingPhaseResult,
    "collection": CollectionPhaseResult,
}


def load_phase_result(phase: PhaseName, result_path: str | Path) -> BaseModel:
    """Load one phase result using its phase-owned schema."""

    path = Path(result_path).expanduser().resolve()
    return _RESULT_MODELS[phase].model_validate(load_json(path))


def validate_handoff(
    run: LargeSceneRun,
    phase: PhaseName,
    result_path: str | Path,
) -> HandoffValidationReport:
    """Validate one phase result without mutating run state."""

    resolved_result_path = Path(result_path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    result: BaseModel | None = None
    try:
        result = load_phase_result(phase, resolved_result_path)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid {phase} result at {resolved_result_path}: {exc}")

    output_digest: str | None = None
    computed_output_digest: str | None = None
    input_digest: str | None = None
    artifact_count = 0
    if result is not None:
        input_digest = getattr(result, "input_digest", None)
        output_digest = getattr(result, "output_digest", None)
        artifact_references = getattr(result, "artifact_paths", [])
        artifact_paths = {
            _resolve(path, resolved_result_path) for path in artifact_references
        }
        artifact_count = len(artifact_paths)
        if len(artifact_paths) != len(artifact_references):
            errors.append("phase result artifact_paths contain duplicates")
        if resolved_result_path in artifact_paths:
            errors.append("phase result must not include itself in artifact_paths")
        for artifact_path in artifact_paths:
            if not artifact_path.exists():
                errors.append(f"Missing phase artifact: {artifact_path}")

        expected_input_digest = run.phases[phase].input_digest
        if input_digest != expected_input_digest:
            errors.append(
                f"phase input digest mismatch: expected {expected_input_digest}, got {input_digest}"
            )
        if getattr(result, "success", False) is not True:
            errors.append(f"{phase} result did not report success")
        if getattr(result, "completion_policy_satisfied", False) is not True:
            errors.append(f"{phase} completion policy is not satisfied")
        unresolved = getattr(result, "unresolved_issues", [])
        if unresolved:
            errors.append(f"{phase} has unresolved issues: {unresolved}")
        if not output_digest:
            errors.append(f"{phase} result has no output_digest")
        try:
            computed_output_digest = phase_result_digest(
                result, result_path=resolved_result_path
            )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"Cannot recompute {phase} output digest: {exc}")
        else:
            if output_digest != computed_output_digest:
                errors.append(
                    f"{phase} output digest mismatch: expected {computed_output_digest}, "
                    f"got {output_digest}"
                )

        validator = {
            "decomposition": _validate_decomposition,
            "asset_task_processing": _validate_processing,
            "collection": _validate_collection,
        }[phase]
        validator(
            result,  # type: ignore[arg-type]
            run,
            resolved_result_path,
            artifact_paths,
            errors,
            warnings,
        )

    return HandoffValidationReport(
        phase=phase,
        valid=not errors,
        result_path=str(resolved_result_path),
        input_digest=input_digest,
        output_digest=output_digest,
        computed_output_digest=computed_output_digest,
        artifact_count=artifact_count,
        errors=errors,
        warnings=warnings,
    )
