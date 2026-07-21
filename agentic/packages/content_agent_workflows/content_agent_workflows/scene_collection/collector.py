# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow 3 collection, material projection, composition, and validation."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from content_agent_workflows.asset_task_processing.contracts import (
    AssetTaskInventory,
    AssetTaskResult,
    AssetTaskResultsIndex,
    ProcessingPhaseResult,
    TaskCatalog,
)
from content_agent_workflows.asset_task_processing.material_task import (
    MaterialCandidateEvidence,
    MaterialDecisionPatch,
    MaterialSurvey,
)
from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_json,
    phase_result_digest,
    resolve_artifact_path,
    seal_phase_result,
)
from content_agent_workflows.scene_decomposition.manifest import (
    ManifestCatalog,
    SceneDecompositionManifest,
)

from .contracts import (
    CollectionInputArtifact,
    CollectionInputIndex,
    CollectionPhaseResult,
    CollectionRequest,
    DomainCollectionResult,
    ProjectedMaterialBinding,
)


class CollectionRuntimeError(RuntimeError):
    """Raised when Workflow 3 cannot project or validate an input."""


def _load_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate(load_json(path))
    except (OSError, ValueError, ValidationError) as exc:
        raise CollectionRuntimeError(
            f"Invalid {model.__name__} at {path}: {exc}"
        ) from exc


def _resolved(path: str | Path, *, base: Path | None = None) -> Path:
    return resolve_artifact_path(path, base_dir=base)


def _artifact_sha256(path: Path, label: str) -> str:
    try:
        return file_sha256(path)
    except OSError as exc:
        raise CollectionRuntimeError(f"Cannot hash {label}: {path}") from exc


def _append_unique_artifact(
    artifacts: list[tuple[str, Path]],
    seen_paths: set[Path],
    role: str,
    path: Path,
) -> None:
    resolved = path.expanduser().resolve()
    if resolved in seen_paths:
        return
    seen_paths.add(resolved)
    artifacts.append((role, resolved))


def prepare_collection(request: CollectionRequest) -> CollectionInputIndex:
    """Validate sealed Workflow 2 inputs and persist a stable collection index."""

    output_dir = _resolved(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = atomic_write_json(output_dir / "request.json", request)
    processing_path = _resolved(request.processing_result_path)
    processing = _load_model(processing_path, ProcessingPhaseResult)
    expected_processing_digest = phase_result_digest(
        processing,
        result_path=processing_path,
    )
    if processing.output_digest != expected_processing_digest:
        raise CollectionRuntimeError(
            "Processing phase result seal does not match its artifacts"
        )
    if not processing.success or not processing.completion_policy_satisfied:
        raise CollectionRuntimeError("Processing phase is not complete")
    if processing.output_digest != request.input_digest:
        raise CollectionRuntimeError(
            "Collection input_digest does not match the processing output digest"
        )

    routed_paths = {
        "manifest_catalog": _resolved(request.manifest_catalog_path),
        "task_catalog": _resolved(request.task_catalog_path),
        "asset_task_inventory": _resolved(request.asset_task_inventory_path),
        "results_index": _resolved(request.results_index_path),
    }
    processing_routes = {
        "manifest_catalog": _resolved(processing.manifest_catalog_path),
        "task_catalog": _resolved(processing.task_catalog_path),
        "asset_task_inventory": _resolved(processing.asset_task_inventory_path),
        "results_index": _resolved(processing.results_index_path),
    }
    for role, path in routed_paths.items():
        if path != processing_routes[role]:
            raise CollectionRuntimeError(
                f"Collection {role} does not match the sealed processing result"
            )

    manifest_catalog = _load_model(routed_paths["manifest_catalog"], ManifestCatalog)
    source_scene = _resolved(request.source_scene)
    if _resolved(manifest_catalog.original_usd_path) != source_scene:
        raise CollectionRuntimeError(
            "Collection source scene differs from manifest catalog"
        )
    task_catalog = _load_model(routed_paths["task_catalog"], TaskCatalog)
    inventory = _load_model(routed_paths["asset_task_inventory"], AssetTaskInventory)
    results_index = _load_model(routed_paths["results_index"], AssetTaskResultsIndex)
    if processing.task_request_digests != inventory.task_request_digests:
        raise CollectionRuntimeError(
            "Processing result and inventory task request digests differ"
        )

    domains = {task.domain for task in task_catalog.tasks}
    missing_domains = set(request.requested_domains) - domains
    if missing_domains:
        raise CollectionRuntimeError(
            f"Requested collection domains are absent from task catalog: {sorted(missing_domains)}"
        )
    result_entries = {entry.work_item_id: entry for entry in results_index.entries}
    missing_required = [
        item.work_item_id
        for item in inventory.work_items
        if item.required
        and (
            item.work_item_id not in result_entries
            or result_entries[item.work_item_id].status not in {"completed", "waived"}
        )
    ]
    if missing_required:
        raise CollectionRuntimeError(
            f"Required work items are unavailable for collection: {missing_required}"
        )

    core_artifacts: list[tuple[str, Path]] = []
    seen_artifacts: set[Path] = set()
    for role, path in [
        ("collection_request", request_path),
        ("source_scene", source_scene),
        ("processing_result", processing_path),
        *routed_paths.items(),
    ]:
        _append_unique_artifact(core_artifacts, seen_artifacts, role, path)
    task_request_digests: dict[str, str] = {}
    for task in task_catalog.tasks:
        task_request_path = _resolved(
            task.request_path, base=routed_paths["task_catalog"].parent
        )
        task_request_digest = _artifact_sha256(
            task_request_path, f"task request {task.task_id}"
        )
        expected_digest = processing.task_request_digests.get(task.task_id)
        if expected_digest and task_request_digest != expected_digest:
            raise CollectionRuntimeError(
                f"Task request digest changed before collection: {task.task_id}"
            )
        if processing.task_request_digests and expected_digest is None:
            raise CollectionRuntimeError(
                f"Processing result lacks task request digest: {task.task_id}"
            )
        task_request_digests[task.task_id] = task_request_digest
        _append_unique_artifact(
            core_artifacts,
            seen_artifacts,
            f"task_request:{task.task_id}",
            task_request_path,
        )
    if request.material_library_yaml:
        _append_unique_artifact(
            core_artifacts,
            seen_artifacts,
            "material_library_yaml",
            _resolved(request.material_library_yaml),
        )
    for index, reference in enumerate(processing.artifact_paths, start=1):
        artifact_path = _resolved(reference, base=processing_path.parent)
        _append_unique_artifact(
            core_artifacts,
            seen_artifacts,
            f"processing_artifact:{index:04d}",
            artifact_path,
        )
    artifacts = [
        CollectionInputArtifact(
            role=role,
            path=str(path),
            sha256=_artifact_sha256(path, role),
        )
        for role, path in core_artifacts
    ]
    index = CollectionInputIndex(
        input_digest=request.input_digest,
        source_scene=str(source_scene),
        requested_domains=request.requested_domains,
        task_request_digests=task_request_digests,
        required_work_item_count=sum(item.required for item in inventory.work_items),
        completed_work_item_count=sum(
            entry.status == "completed" for entry in results_index.entries
        ),
        artifacts=artifacts,
    )
    atomic_write_json(output_dir / "collection_input_index.json", index)
    return index


def _candidate_materials(
    decision: MaterialDecisionPatch,
    survey: MaterialSurvey,
) -> dict[str, tuple[str, str]]:
    expected = {candidate.prim_path for candidate in survey.candidates}
    projected: dict[str, tuple[str, str]] = {}
    for assignment in decision.assignments:
        if assignment.coverage_mode == "descendants":
            prefix = assignment.target_prim_path.rstrip("/") + "/"
            covered = {
                path
                for path in expected
                if path == assignment.target_prim_path or path.startswith(prefix)
            }
        else:
            covered = set(assignment.covered_candidate_paths)
        if not covered:
            raise CollectionRuntimeError(
                f"Material assignment covers no surveyed candidates: {assignment.target_prim_path}"
            )
        for candidate_path in covered:
            if candidate_path in projected:
                raise CollectionRuntimeError(
                    f"Material candidate has conflicting assignments: {candidate_path}"
                )
            projected[candidate_path] = (
                assignment.material_name,
                assignment.target_prim_path,
            )
    if set(projected) != expected:
        raise CollectionRuntimeError(
            "Material decision coverage differs from survey; "
            f"missing={sorted(expected - set(projected))}, "
            f"extra={sorted(set(projected) - expected)}"
        )
    return projected


def _surface_candidates_under(stage: Any, root_path: str) -> list[dict[str, Any]]:
    from pxr import Usd, UsdGeom, UsdShade

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise CollectionRuntimeError(f"Original asset root does not exist: {root_path}")
    candidates: list[dict[str, Any]] = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        computed_visible = (
            UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
        )
        mesh = UsdGeom.Mesh(prim)
        face_count = len(mesh.GetFaceVertexCountsAttr().Get() or [])
        covered_faces: set[int] = set()
        material_subsets: list[tuple[Any, int]] = []
        for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim)):
            indices = subset.GetIndicesAttr().Get() or []
            if indices:
                material_subsets.append((subset.GetPrim(), len(indices)))
                covered_faces.update(int(index) for index in indices)
        for subset_prim, subset_face_count in material_subsets:
            material, _relationship = UsdShade.MaterialBindingAPI(
                subset_prim
            ).ComputeBoundMaterial()
            candidates.append(
                {
                    "prim_path": str(subset_prim.GetPath()),
                    "prim_type": subset_prim.GetTypeName(),
                    "face_count": subset_face_count,
                    "computed_visible": computed_visible,
                    "source_material_key": (
                        material.GetPrim().GetName()
                        if material
                        else subset_prim.GetName()
                    ),
                }
            )
        if not material_subsets or len(covered_faces) < face_count:
            material, _relationship = UsdShade.MaterialBindingAPI(
                prim
            ).ComputeBoundMaterial()
            candidates.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "prim_type": prim.GetTypeName(),
                    "face_count": max(face_count - len(covered_faces), 0),
                    "computed_visible": computed_visible,
                    "source_material_key": (
                        material.GetPrim().GetName() if material else None
                    ),
                }
            )
    return candidates


def _relative_path(path: str, root: str) -> str:
    if path == root:
        return ""
    prefix = root.rstrip("/") + "/"
    if not path.startswith(prefix):
        raise CollectionRuntimeError(f"Path {path} is outside representative {root}")
    return path[len(root) :]


def _tail(path: str, length: int) -> tuple[str, ...]:
    parts = tuple(part for part in path.split("/") if part)
    return parts[-length:]


def _map_member_candidates(
    stage: Any,
    representative_root: str,
    member_root: str,
    source_candidates: list[MaterialCandidateEvidence],
    *,
    skip_invisible: bool = False,
) -> tuple[
    list[tuple[MaterialCandidateEvidence, dict[str, Any], str]],
    list[dict[str, Any]],
]:
    member_candidates = _surface_candidates_under(stage, member_root)

    mapped: dict[int, tuple[dict[str, Any], str]] = {}
    used_targets: set[str] = set()
    member_by_path = {
        candidate["prim_path"]: candidate for candidate in member_candidates
    }
    for index, source in enumerate(source_candidates):
        relative = _relative_path(source.prim_path, representative_root)
        exact_path = member_root + relative
        exact = member_by_path.get(exact_path)
        if exact and exact["prim_type"] == source.prim_type:
            mapped[index] = (exact, "exact_relative")
            used_targets.add(exact_path)

    for index, source in enumerate(source_candidates):
        if index in mapped:
            continue
        available = [
            candidate
            for candidate in member_candidates
            if candidate["prim_path"] not in used_targets
            and candidate["prim_type"] == source.prim_type
            and candidate["face_count"] == source.face_count
        ]
        selected = None
        source_relative = _relative_path(source.prim_path, representative_root)
        for length in (4, 3, 2, 1):
            same_tail = [
                candidate
                for candidate in available
                if _tail(_relative_path(candidate["prim_path"], member_root), length)
                == _tail(source_relative, length)
            ]
            if len(same_tail) == 1:
                selected = same_tail[0]
                break
        if selected is not None:
            mapped[index] = (selected, "stable_suffix")
            used_targets.add(selected["prim_path"])

    for index, source in enumerate(source_candidates):
        if index in mapped:
            continue
        source_key = source.bound_material_name
        if not source_key:
            continue
        same_identity = [
            candidate
            for candidate in member_candidates
            if candidate["prim_path"] not in used_targets
            and candidate["prim_type"] == source.prim_type
            and candidate["source_material_key"] == source_key
        ]
        if len(same_identity) == 1:
            selected = same_identity[0]
            mapped[index] = (selected, "equivalent_material_id")
            used_targets.add(selected["prim_path"])

    if len(mapped) != len(source_candidates):
        unresolved = [
            source_candidates[index].prim_path
            for index in range(len(source_candidates))
            if index not in mapped
        ]
        raise CollectionRuntimeError(
            f"Could not project representative candidates to {member_root}: {unresolved}"
        )
    mappings = [
        (source, mapped[index][0], mapped[index][1])
        for index, source in enumerate(source_candidates)
    ]
    unmatched_members = [
        candidate
        for candidate in member_candidates
        if candidate["prim_path"] not in used_targets
        and (not skip_invisible or candidate["computed_visible"])
    ]
    return mappings, unmatched_members


def _authoring_target(stage: Any, instance_target_path: str) -> str:
    prim = stage.GetPrimAtPath(instance_target_path)
    if not prim or not prim.IsValid():
        raise CollectionRuntimeError(
            f"Projected instance target does not exist: {instance_target_path}"
        )
    if not prim.IsInstanceProxy() and not prim.IsInPrototype():
        return instance_target_path
    root_identifier = stage.GetRootLayer().identifier
    for spec in prim.GetPrimStack():
        if spec.layer.identifier == root_identifier:
            return str(spec.path)
    raise CollectionRuntimeError(
        f"No original-layer authoring target for {instance_target_path}"
    )


def _project_material_bindings(
    request: CollectionRequest,
) -> tuple[
    list[ProjectedMaterialBinding],
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
]:
    from pxr import Usd

    processing_dir = _resolved(request.asset_task_inventory_path).parent
    manifest_catalog = _load_model(
        _resolved(request.manifest_catalog_path), ManifestCatalog
    )
    task_catalog = _load_model(_resolved(request.task_catalog_path), TaskCatalog)
    inventory = _load_model(
        _resolved(request.asset_task_inventory_path), AssetTaskInventory
    )
    results_index = _load_model(
        _resolved(request.results_index_path), AssetTaskResultsIndex
    )
    material_tasks = [task for task in task_catalog.tasks if task.domain == "material"]
    if len(material_tasks) != 1:
        raise CollectionRuntimeError(
            f"Expected one material task, found {len(material_tasks)}"
        )
    task = material_tasks[0]
    manifest_entry = next(
        (
            entry
            for entry in manifest_catalog.manifests
            if entry.manifest_id == task.manifest_id
        ),
        None,
    )
    if manifest_entry is None or not manifest_entry.finalized:
        raise CollectionRuntimeError("Material manifest is missing or not finalized")
    manifest = _load_model(
        _resolved(
            manifest_entry.path, base=_resolved(request.manifest_catalog_path).parent
        ),
        SceneDecompositionManifest,
    )
    assets_by_id = {asset.asset_id: asset for asset in manifest.assets}
    assets_by_path = {asset.original_root_path: asset for asset in manifest.assets}
    groups = {group.group_id: group for group in manifest.instance_groups}
    groups_by_representative = {
        group.representative_asset_id: group
        for group in manifest.instance_groups
        if group.representative_asset_id
    }
    index_entries = {entry.work_item_id: entry for entry in results_index.entries}
    stage = Usd.Stage.Open(str(_resolved(request.source_scene)), Usd.Stage.LoadAll)
    if stage is None:
        raise CollectionRuntimeError(
            f"Could not open source scene: {request.source_scene}"
        )

    equivalence_candidates: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for item in inventory.work_items:
        if item.task_id != task.task_id:
            continue
        entry = index_entries.get(item.work_item_id)
        if entry is None or not entry.result_path:
            continue
        result = _load_model(
            _resolved(entry.result_path, base=processing_dir), AssetTaskResult
        )
        decision_reference = result.domain_outputs.get("decision_path")
        survey_reference = result.domain_outputs.get("survey_path")
        if not decision_reference or not survey_reference:
            continue
        decision = _load_model(_resolved(decision_reference), MaterialDecisionPatch)
        survey = _load_model(_resolved(survey_reference), MaterialSurvey)
        candidate_materials = _candidate_materials(decision, survey)
        for candidate in survey.candidates:
            if not candidate.bound_material_name:
                continue
            material_name, _decision_target = candidate_materials[candidate.prim_path]
            equivalence_candidates[candidate.bound_material_name][material_name].add(
                item.work_item_id
            )
    material_equivalences = {
        key: {
            "material_name": next(iter(materials)),
            "evidence_work_item_ids": sorted(next(iter(materials.values()))),
        }
        for key, materials in equivalence_candidates.items()
        if len(materials) == 1
    }

    projected: list[ProjectedMaterialBinding] = []
    direct_member_count = 0
    propagated_member_count = 0
    equivalence_expansion_count = 0
    for item in inventory.work_items:
        if item.task_id != task.task_id:
            continue
        entry = index_entries.get(item.work_item_id)
        if entry is not None and entry.status == "waived":
            continue
        if entry is None or entry.status != "completed" or not entry.result_path:
            raise CollectionRuntimeError(
                f"Missing completed result: {item.work_item_id}"
            )
        result_path = _resolved(entry.result_path, base=processing_dir)
        result = _load_model(result_path, AssetTaskResult)
        if result.work_item_id != item.work_item_id:
            raise CollectionRuntimeError(
                f"Result identity mismatch for {item.work_item_id}"
            )
        if result.mapping.path_space != "original" or result.mapping.unresolved_paths:
            raise CollectionRuntimeError(
                f"Result has unresolved original-path mapping: {item.work_item_id}"
            )
        decision_reference = result.domain_outputs.get("decision_path")
        survey_reference = result.domain_outputs.get("survey_path")
        if not decision_reference or not survey_reference:
            raise CollectionRuntimeError(
                f"Material result lacks decision or survey: {item.work_item_id}"
            )
        decision = _load_model(_resolved(decision_reference), MaterialDecisionPatch)
        survey = _load_model(_resolved(survey_reference), MaterialSurvey)
        if (
            decision.work_item_id != item.work_item_id
            or survey.work_item_id != item.work_item_id
        ):
            raise CollectionRuntimeError(
                f"Material payload identity mismatch: {item.work_item_id}"
            )
        candidate_materials = _candidate_materials(decision, survey)
        representative = assets_by_id.get(item.asset_id)
        if representative is None or not representative.processable:
            raise CollectionRuntimeError(
                f"Inventory asset is not a processable representative: {item.asset_id}"
            )
        if representative.original_root_path != item.original_root_path:
            raise CollectionRuntimeError(
                f"Inventory root differs from manifest for {item.asset_id}"
            )

        members: list[tuple[str, str]]
        group = (
            groups.get(representative.instance_group_id)
            if representative.instance_group_id
            else None
        )
        if group is None:
            group = groups_by_representative.get(representative.asset_id)
        if group and group.representative_asset_id == representative.asset_id:
            members = []
            member_paths = list(
                dict.fromkeys([representative.original_root_path, *group.member_paths])
            )
            for index, member_path in enumerate(member_paths):
                member = assets_by_path.get(member_path)
                member_asset_id = (
                    member.asset_id
                    if member is not None
                    else f"{group.group_id}:member-{index:04d}"
                )
                members.append((member_asset_id, member_path))
        else:
            members = [(representative.asset_id, representative.original_root_path)]

        for member_asset_id, member_root_path in members:
            propagation_basis = (
                "explicit"
                if member_root_path == representative.original_root_path
                else "instance_group"
            )
            if propagation_basis == "explicit":
                direct_member_count += 1
            else:
                propagated_member_count += 1
            candidate_mapping, unmatched_member_candidates = _map_member_candidates(
                stage,
                representative.original_root_path,
                member_root_path,
                survey.candidates,
                skip_invisible=survey.visibility_policy == "visible_only",
            )
            for source_candidate, target_candidate, mapping_method in candidate_mapping:
                material_name, decision_target = candidate_materials[
                    source_candidate.prim_path
                ]
                instance_target_path = target_candidate["prim_path"]
                projected.append(
                    ProjectedMaterialBinding(
                        work_item_id=item.work_item_id,
                        representative_asset_id=representative.asset_id,
                        member_asset_id=member_asset_id,
                        representative_root_path=representative.original_root_path,
                        member_root_path=member_root_path,
                        decision_target_path=decision_target,
                        source_candidate_path=source_candidate.prim_path,
                        instance_target_path=instance_target_path,
                        authoring_target_path=_authoring_target(
                            stage, instance_target_path
                        ),
                        material_name=material_name,
                        propagation_basis=propagation_basis,
                        mapping_method=mapping_method,
                        evidence_work_item_ids=(
                            [item.work_item_id]
                            if mapping_method == "equivalent_material_id"
                            else []
                        ),
                    )
                )
            for target_candidate in unmatched_member_candidates:
                source_key = target_candidate.get("source_material_key")
                equivalence = material_equivalences.get(source_key)
                if equivalence is None:
                    raise CollectionRuntimeError(
                        "Unmatched member material region has no unanimous prior "
                        f"equivalence: {target_candidate['prim_path']} ({source_key})"
                    )
                equivalence_expansion_count += 1
                instance_target_path = target_candidate["prim_path"]
                projected.append(
                    ProjectedMaterialBinding(
                        work_item_id=item.work_item_id,
                        representative_asset_id=representative.asset_id,
                        member_asset_id=member_asset_id,
                        representative_root_path=representative.original_root_path,
                        member_root_path=member_root_path,
                        decision_target_path=f"material-equivalence:{source_key}",
                        source_candidate_path=instance_target_path,
                        instance_target_path=instance_target_path,
                        authoring_target_path=_authoring_target(
                            stage, instance_target_path
                        ),
                        material_name=str(equivalence["material_name"]),
                        propagation_basis=propagation_basis,
                        mapping_method="equivalent_material_id",
                        evidence_work_item_ids=list(
                            equivalence["evidence_work_item_ids"]
                        ),
                    )
                )

    by_authoring_target: dict[str, list[ProjectedMaterialBinding]] = defaultdict(list)
    for binding in projected:
        by_authoring_target[binding.authoring_target_path].append(binding)
    harmonized: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []
    duplicate_merge_count = 0
    for target_path, bindings in sorted(by_authoring_target.items()):
        materials = sorted({binding.material_name for binding in bindings})
        if len(materials) != 1:
            conflicts.append(
                {
                    "authoring_target_path": target_path,
                    "materials": materials,
                    "work_item_ids": sorted(
                        {binding.work_item_id for binding in bindings}
                    ),
                }
            )
            continue
        harmonized[target_path] = materials[0]
        duplicate_merge_count += len(bindings) - 1
    if conflicts:
        raise CollectionRuntimeError(
            f"Material harmonization found {len(conflicts)} conflicting targets"
        )

    projection_summary = {
        "schema_version": "content-agent-workflows.material-projection-report.v1",
        "task_request_digest": inventory.task_request_digests.get(task.task_id),
        "work_item_count": sum(
            item.task_id == task.task_id for item in inventory.work_items
        ),
        "representative_member_count": direct_member_count,
        "propagated_member_count": propagated_member_count,
        "covered_member_count": direct_member_count + propagated_member_count,
        "manifest_asset_count": len(manifest.assets),
        "projected_binding_count": len(projected),
        "authoring_target_count": len(harmonized),
        "material_equivalence_expansion_count": equivalence_expansion_count,
        "material_histogram": dict(sorted(Counter(harmonized.values()).items())),
        "mapping_method_histogram": dict(
            sorted(Counter(binding.mapping_method for binding in projected).items())
        ),
        "propagation_histogram": dict(
            sorted(Counter(binding.propagation_basis for binding in projected).items())
        ),
    }
    harmonization_summary = {
        "schema_version": "content-agent-workflows.material-harmonization-report.v1",
        "task_request_digest": inventory.task_request_digests.get(task.task_id),
        "input_binding_count": len(projected),
        "output_target_count": len(harmonized),
        "duplicate_merge_count": duplicate_merge_count,
        "material_equivalence_expansion_count": equivalence_expansion_count,
        "material_equivalence_keys": {
            key: value
            for key, value in sorted(material_equivalences.items())
            if any(
                binding.mapping_method == "equivalent_material_id"
                and (
                    binding.decision_target_path == f"material-equivalence:{key}"
                    or key in binding.instance_target_path.rsplit("/", 1)[-1]
                )
                for binding in projected
            )
        },
        "conflict_count": 0,
        "conflicts": [],
        "policy": "identical source-target opinions merge; divergent opinions fail closed",
    }
    return projected, harmonized, projection_summary, harmonization_summary


def _write_json_lines(path: Path, records: list[dict[str, Any]]) -> Path:
    return atomic_write_text(
        path,
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
    )


def _validate_material_output(
    *,
    material_layer_path: Path,
    composed_scene_path: Path,
    projected: list[ProjectedMaterialBinding],
    harmonized: dict[str, str],
    material_paths: dict[str, str],
) -> dict[str, Any]:
    from pxr import Sdf, Usd, UsdShade

    layer = None
    stage = None
    errors: list[str] = []
    try:
        layer = Sdf.Layer.FindOrOpen(str(material_layer_path))
        stage = Usd.Stage.Open(str(composed_scene_path), Usd.Stage.LoadAll)
        if layer is None:
            errors.append("Material layer could not be opened")
        if stage is None:
            errors.append("Composed scene could not be opened")

        authored_binding_count = 0
        if layer is not None:
            for target_path, material_name in harmonized.items():
                spec = layer.GetPrimAtPath(target_path)
                relationship = (
                    spec.relationships.get("material:binding") if spec else None
                )
                targets = (
                    [str(path) for path in relationship.targetPathList.explicitItems]
                    if relationship
                    else []
                )
                if material_paths[material_name] not in targets:
                    errors.append(
                        f"Authored binding mismatch at {target_path}: {targets}"
                    )
                    if len(errors) >= 100:
                        break
                else:
                    authored_binding_count += 1

        composed_binding_count = 0
        if stage is not None and len(errors) < 100:
            for binding in projected:
                prim = stage.GetPrimAtPath(binding.instance_target_path)
                if not prim or not prim.IsValid():
                    errors.append(
                        f"Composed target is missing: {binding.instance_target_path}"
                    )
                else:
                    material, _relationship = UsdShade.MaterialBindingAPI(
                        prim
                    ).ComputeBoundMaterial()
                    actual = str(material.GetPath()) if material else None
                    expected = material_paths[binding.material_name]
                    if actual != expected:
                        errors.append(
                            "Composed binding mismatch at "
                            f"{binding.instance_target_path}: "
                            f"expected {expected}, got {actual}"
                        )
                    else:
                        composed_binding_count += 1
                if len(errors) >= 100:
                    break

        return {
            "schema_version": (
                "content-agent-workflows.material-collection-validation.v1"
            ),
            "passed": not errors,
            "expected_authoring_target_count": len(harmonized),
            "authored_binding_count": authored_binding_count,
            "expected_composed_binding_count": len(projected),
            "validated_composed_binding_count": composed_binding_count,
            "errors": errors,
        }
    finally:
        # OpenUSD Python bindings do not expose Stage.Close(); drop references
        # promptly so bulk collection can release file handles between assets.
        stage = None
        layer = None


def _create_composed_scene(
    source_scene: Path,
    material_layer: Path,
    output_path: Path,
) -> Path:
    from pxr import Sdf

    source_layer = Sdf.Layer.FindOrOpen(str(source_scene))
    if source_layer is None:
        raise CollectionRuntimeError(f"Could not open source layer: {source_scene}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=output_path.suffix,
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
    temporary_path.unlink()
    try:
        layer = Sdf.Layer.CreateNew(str(temporary_path))
        if layer is None:
            raise CollectionRuntimeError(
                f"Could not create composed scene: {output_path}"
            )
        layer.subLayerPaths = [
            str(material_layer.resolve()),
            str(source_scene.resolve()),
        ]
        if source_layer.defaultPrim:
            layer.defaultPrim = source_layer.defaultPrim
        for key in ("upAxis", "metersPerUnit"):
            if source_layer.pseudoRoot.HasInfo(key):
                layer.pseudoRoot.SetInfo(key, source_layer.pseudoRoot.GetInfo(key))
        layer.Save()
        del layer
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def _topology_report(source_scene: Path, composed_scene: Path) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    source = Usd.Stage.Open(str(source_scene), Usd.Stage.LoadAll)
    composed = Usd.Stage.Open(str(composed_scene), Usd.Stage.LoadAll)
    if source is None or composed is None:
        return {
            "schema_version": "content-agent-workflows.topology-preservation-report.v1",
            "passed": False,
            "errors": ["Source or composed stage could not be opened"],
        }
    source_prims = {
        str(prim.GetPath()): prim.GetTypeName() for prim in source.TraverseAll()
    }
    composed_prims = {
        str(prim.GetPath()): prim.GetTypeName() for prim in composed.TraverseAll()
    }
    missing = sorted(set(source_prims) - set(composed_prims))
    type_mismatches = sorted(
        path
        for path, type_name in source_prims.items()
        if path in composed_prims and composed_prims[path] != type_name
    )

    def counts(stage: Any) -> dict[str, int]:
        prims = list(stage.TraverseAll())
        return {
            "prim_count": len(prims),
            "mesh_count": sum(prim.IsA(UsdGeom.Mesh) for prim in prims),
            "instance_count": sum(prim.IsInstance() for prim in prims),
            "prototype_count": len(stage.GetPrototypes()),
        }

    source_counts = counts(source)
    composed_counts = counts(composed)
    errors: list[str] = []
    if missing:
        errors.append(f"Missing source paths: {missing[:20]}")
    if type_mismatches:
        errors.append(f"Source type mismatches: {type_mismatches[:20]}")
    for key in ("mesh_count", "instance_count", "prototype_count"):
        if source_counts[key] != composed_counts[key]:
            errors.append(
                f"{key} changed: {source_counts[key]} -> {composed_counts[key]}"
            )
    source_default = (
        str(source.GetDefaultPrim().GetPath()) if source.GetDefaultPrim() else None
    )
    composed_default = (
        str(composed.GetDefaultPrim().GetPath()) if composed.GetDefaultPrim() else None
    )
    if source_default != composed_default:
        errors.append(f"Default prim changed: {source_default} -> {composed_default}")
    return {
        "schema_version": "content-agent-workflows.topology-preservation-report.v1",
        "passed": not errors,
        "source_scene": str(source_scene),
        "composed_scene": str(composed_scene),
        "source_counts": source_counts,
        "composed_counts": composed_counts,
        "missing_source_path_count": len(missing),
        "type_mismatch_count": len(type_mismatches),
        "source_default_prim": source_default,
        "composed_default_prim": composed_default,
        "errors": errors,
    }


def run_collection(request: CollectionRequest) -> CollectionPhaseResult:
    """Run Workflow 3 for the requested material domain and seal its result."""

    output_dir = _resolved(request.output_dir)
    input_index = prepare_collection(request)
    if request.requested_domains != ["material"]:
        raise CollectionRuntimeError(
            "This implementation currently requires requested_domains=['material']"
        )
    if not request.material_library_yaml:
        raise CollectionRuntimeError(
            "Material collection requires material_library_yaml"
        )

    projected, harmonized, collection_report, harmonization_report = (
        _project_material_bindings(request)
    )
    domain_dir = output_dir / "domains" / "material"
    domain_dir.mkdir(parents=True, exist_ok=True)
    projected_path = atomic_write_json(
        domain_dir / "projected_results.json",
        {
            "schema_version": "content-agent-workflows.projected-material-bindings.v1",
            "bindings": [binding.model_dump(mode="json") for binding in projected],
        },
    )
    decisions_path = _write_json_lines(
        domain_dir / "collection_decisions.jsonl",
        [
            {
                "authoring_target_path": path,
                "material_name": material,
                "action": "author",
            }
            for path, material in sorted(harmonized.items())
        ],
    )
    collection_report_path = atomic_write_json(
        domain_dir / "collection_report.json", collection_report
    )
    harmonization_report_path = atomic_write_json(
        domain_dir / "harmonization_report.json", harmonization_report
    )

    try:
        from material_agent.scene.collect import author_projected_material_layer
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise CollectionRuntimeError(
            "Material collection requires the material_agent package"
        ) from exc
    material_layer_path = domain_dir / "material_layer.usda"
    authoring_summary = author_projected_material_layer(
        _resolved(request.source_scene),
        material_layer_path,
        _resolved(request.material_library_yaml),
        harmonized,
    )
    authoring_summary_path = atomic_write_json(
        domain_dir / "authoring_summary.json", authoring_summary
    )

    composition_dir = output_dir / "composition"
    composed_scene_path = _create_composed_scene(
        _resolved(request.source_scene),
        material_layer_path,
        composition_dir / "composed_scene.usda",
    )
    composition_plan_path = atomic_write_json(
        composition_dir / "composition_plan.json",
        {
            "schema_version": "content-agent-workflows.composition-plan.v1",
            "source_scene": str(_resolved(request.source_scene)),
            "domain_layers_strong_to_weak": [str(material_layer_path)],
            "composed_scene": str(composed_scene_path),
        },
    )
    validation = _validate_material_output(
        material_layer_path=material_layer_path,
        composed_scene_path=composed_scene_path,
        projected=projected,
        harmonized=harmonized,
        material_paths=dict(authoring_summary["material_paths"]),
    )
    validation_path = atomic_write_json(
        domain_dir / "validation_report.json", validation
    )
    domain_artifacts = [
        str(projected_path),
        str(decisions_path),
        str(collection_report_path),
        str(harmonization_report_path),
        str(material_layer_path),
        str(authoring_summary_path),
        str(validation_path),
    ]
    domain_result_path = domain_dir / "domain_result.json"
    domain_result = seal_phase_result(
        DomainCollectionResult(
            domain="material",
            status="completed" if validation["passed"] else "failed",
            required=True,
            output_paths=[str(material_layer_path)],
            collection_report_path=str(collection_report_path),
            harmonization_report_path=str(harmonization_report_path),
            validation_report_path=str(validation_path),
            validation_passed=bool(validation["passed"]),
            unresolved_issues=list(validation["errors"]),
            artifact_paths=domain_artifacts,
        ),
        domain_result_path,
    )

    topology = _topology_report(_resolved(request.source_scene), composed_scene_path)
    topology_path = atomic_write_json(
        composition_dir / "topology_report.json", topology
    )
    cross_domain = {
        "schema_version": "content-agent-workflows.cross-domain-validation.v1",
        "passed": bool(validation["passed"] and topology["passed"]),
        "domain_count": 1,
        "checks": ["material_domain_validation", "topology_preservation"],
        "errors": [*validation["errors"], *topology.get("errors", [])],
    }
    cross_domain_path = atomic_write_json(
        composition_dir / "cross_domain_validation.json", cross_domain
    )
    unresolved_path = atomic_write_json(
        output_dir / "unresolved_issues.json",
        {"issues": cross_domain["errors"]},
    )
    phase_artifacts = list(
        dict.fromkeys(
            [
                str(output_dir / "collection_input_index.json"),
                *(artifact.path for artifact in input_index.artifacts),
                *domain_artifacts,
                str(domain_result_path),
                str(composition_plan_path),
                str(composed_scene_path),
                str(topology_path),
                str(cross_domain_path),
                str(unresolved_path),
            ]
        )
    )
    success = bool(
        domain_result.status == "completed"
        and domain_result.validation_passed
        and topology["passed"]
        and cross_domain["passed"]
    )
    return seal_phase_result(
        CollectionPhaseResult(
            success=success,
            input_digest=input_index.input_digest,
            domain_result_paths=[str(domain_result_path)],
            required_domain_count=1,
            completed_required_domain_count=int(
                domain_result.status == "completed" and domain_result.validation_passed
            ),
            composition_path=str(composed_scene_path),
            topology_report_path=str(topology_path),
            cross_domain_validation_path=str(cross_domain_path),
            final_output_paths=[str(composed_scene_path)],
            artifact_paths=phase_artifacts,
            completion_policy_satisfied=success,
            unresolved_issues=list(cross_domain["errors"]),
            error=None if success else "Collection validation failed",
        ),
        output_dir / "collection_result.json",
    )
