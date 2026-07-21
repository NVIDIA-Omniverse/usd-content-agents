# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staged SimReady Foundation profile conformance router."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ValidationError
from world_understanding.utils.usd.package import (
    extract_usdz_members_to_dir,
    safe_usdz_member_parts,
)

from .foundation_runtime import resolve_simready_runtime
from .models import (
    DEFAULT_SIMREADY_PROFILE,
    DEFAULT_SIMREADY_PROFILE_VERSION,
    SimReadyConformanceInput,
    SimReadyConformanceReport,
    SimReadyGraspLinePlan,
    SimReadyGraspPlan,
)

FOUNDATION_SKILL_BY_AREA = {
    "core": "simready-foundation-conform-fet-000-core",
    "minimal": "simready-foundation-conform-fet-001-minimal",
    "rigid_body": "simready-foundation-conform-fet-003-rigid-body-physics",
    "multibody": "simready-foundation-conform-fet-004-simulate-multi-body-physics",
    "grasp": "simready-foundation-conform-fet-005-simulate-grasp-physics",
    "materials": "simready-foundation-conform-fet-006-materials",
    "nonvisual_materials": "simready-foundation-conform-fet-007-nonvisual-materials",
    "robot_core": "simready-foundation-conform-fet-021-robot-core",
    "robot_materials": "simready-foundation-conform-fet-023-robot-materials",
    "base_articulation": "simready-foundation-conform-fet-024-base-articulation",
}

CORE_REQUIREMENTS = {"NP.002", "NP.003", "NP.004", "NP.005", "NP.006", "SR.001"}
MINIMAL_REQUIREMENTS = {
    "AA.001",
    "UN.001",
    "UN.002",
    "UN.003",
    "UN.004",
    "UN.005",
    "UN.006",
    "UN.007",
}
MULTIBODY_REQUIREMENTS = {"RB.MB.001"}
GRASP_REQUIREMENTS = {"GSP.001"}
PHYSICS_MATERIAL_REQUIREMENTS = {"PMT.001"}
PHYSICS_MATERIAL_PURPOSE = "physics"
ATOMIC_ASSET_PATH_REQUIREMENT = "AA.001"
ISAAC_COMPOSITION_REQUIREMENT = "ISA.001"
GATE3A_HYGIENE_REQUIREMENT = "G3A.HYG.001"
GATE3A_HYGIENE_OUTPUT_DIR = "gate3a-hygiene"
GATE3A_HYGIENE_RECEIPT_SCHEMA_VERSION = (
    "content-agent-workflows.simready-gate3a-hygiene-receipt.v1"
)
ISAAC_COMPOSITION_PAYLOAD_DIR = "payloads"
ISAAC_COMPOSITION_OUTPUT_DIR = "conformed"
GSP001_OUTPUT_DIR = "grasp-conformed"
GSP001_RECEIPT_SCHEMA_VERSION = (
    "content-agent-workflows.simready-gsp001-repair-receipt.v1"
)
_GSP001_AUTHORED_PROPERTIES = {
    "curveVertexCounts",
    "extent",
    "points",
    "type",
    "widths",
    "wrap",
}
_ISAAC_COMPOSITION_LAYER_METADATA_EXCLUSIONS = {
    "defaultPrim",
    "subLayers",
    "subLayerOffsets",
}
_REQUIREMENT_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:\.[A-Z][A-Z0-9]*)*\.\d+\b")


@dataclass
class _RepairResult:
    status: str
    passed: bool
    reason: str
    output_path: Path
    package_root: Path | None = None
    report: dict[str, Any] = field(default_factory=dict)
    report_path: Path | None = None


@dataclass(frozen=True)
class _GSP001SourceLineage:
    receipt_path: Path
    receipt_sha256: str


@dataclass(frozen=True)
class _AA001IdentityOpinion:
    layer_path: str
    prim_path: str
    asset_path: str


@dataclass
class _AA001Plan:
    layers: tuple[str, ...]
    dependencies: tuple[dict[str, Any], ...]
    replacements: dict[str, dict[str, str]]
    identity_removals: tuple[_AA001IdentityOpinion, ...]

    @property
    def changed(self) -> bool:
        return bool(self.replacements or self.identity_removals)


def run_simready_profile_conformance(
    params: SimReadyConformanceInput,
) -> SimReadyConformanceReport:
    """Route profile failures to Foundation conformance skills safely."""

    asset_path = _absolute_path(Path(params.asset_path).expanduser())
    output_dir = Path(params.output_dir).expanduser().resolve()
    report_path = (
        Path(params.report_path).expanduser().resolve()
        if params.report_path is not None
        else output_dir / "simready-conform-profile.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not asset_path.exists():
        report = SimReadyConformanceReport(
            input_usd_path=str(asset_path),
            output_usd_path=str(asset_path),
            output_dir=str(output_dir),
            profile=params.profile,
            profile_version=params.profile_version,
            passed=False,
            status="FAIL",
            errors=[f"Asset path does not exist: {asset_path}"],
        )
        return _write_report(report_path, report)

    runtime = resolve_simready_runtime(
        foundation_root=params.foundation_root,
        foundation_spec_root=params.foundation_spec_root,
        install_missing=False,
    )
    foundation_root = (
        Path(runtime.foundation_root).expanduser().resolve()
        if runtime.foundation_root
        else None
    )
    foundation_missing = foundation_root is None or not foundation_root.exists()

    validation_report_path = (
        Path(params.validation_report_path).expanduser().resolve()
        if params.validation_report_path is not None
        else None
    )
    grasp_plan_path = (
        Path(params.grasp_plan_path).expanduser().resolve()
        if params.grasp_plan_path is not None
        else None
    )
    grasp_source_asset_path = (
        _absolute_path(Path(params.source_asset).expanduser())
        if params.source_asset is not None
        else asset_path
    )
    failed_requirements, report_errors = _failed_requirements(validation_report_path)
    requested_requirements = sorted(
        set(params.repair_requirements) | set(failed_requirements)
    )

    try:
        latest_output, latest_package_root, stage_warnings = _stage_input(
            asset_path,
            output_dir,
            force=params.force,
        )
    except (OSError, ValueError) as exc:
        report = SimReadyConformanceReport(
            input_usd_path=str(asset_path),
            output_usd_path=str(asset_path),
            output_dir=str(output_dir),
            profile=params.profile,
            profile_version=params.profile_version,
            validation_report=str(validation_report_path)
            if validation_report_path
            else None,
            failed_requirements=failed_requirements,
            passed=False,
            status="FAIL",
            errors=[
                *report_errors,
                f"Could not stage asset for SimReady conformance: {exc}",
            ],
            next_step="fix-asset-staging",
        )
        return _write_report(report_path, report)
    selected_requirements = sorted(
        _expanded_repair_requirements(
            requested_requirements,
            asset_path=latest_output,
        ),
        key=_repair_order_key,
    )
    steps: list[dict[str, Any]] = []
    blocked: list[str] = []
    repaired: list[str] = []
    skipped: list[str] = []
    repair_reports: dict[str, str] = {}
    warnings: list[str] = list(stage_warnings)
    errors: list[str] = []
    grasp_source_lineage: _GSP001SourceLineage | None = None

    if foundation_missing:
        warnings.append(
            "SimReady Foundation checkout is unavailable; conformance routing "
            "for requirements without deterministic local repairs will be blocked."
        )
    errors.extend(report_errors)

    for requirement in selected_requirements:
        skill = _skill_for_requirement(requirement)
        has_local_repair = _has_local_repair(requirement)
        if skill is None and not has_local_repair:
            skipped.append(requirement)
            steps.append(
                _step(
                    requirement=requirement,
                    status="SKIPPED",
                    passed=True,
                    input_path=latest_output,
                    output_path=latest_output,
                    reason=(
                        "No Foundation conformance route is registered for this "
                        "requirement in the current adapter."
                    ),
                )
            )
            continue

        skill_path = (
            _foundation_skill_path(foundation_root, skill)
            if skill is not None
            else None
        )
        if skill is not None and skill_path is None and not has_local_repair:
            blocked.append(requirement)
            steps.append(
                _step(
                    requirement=requirement,
                    status="BLOCKED",
                    passed=False,
                    input_path=latest_output,
                    output_path=latest_output,
                    upstream_skill=skill,
                    upstream_skill_path=str(skill_path) if skill_path else None,
                    reason="The mapped Foundation conformance skill is unavailable.",
                )
            )
            continue

        step_input = latest_output
        grasp_source_path = grasp_source_asset_path
        repair_result = _repair_requirement(
            requirement=requirement,
            asset_path=step_input,
            package_root=latest_package_root,
            output_dir=output_dir,
            grasp_plan_path=grasp_plan_path,
            grasp_source_asset_path=grasp_source_path,
            grasp_source_lineage=grasp_source_lineage,
            expected_physics_inventory_sha256=(
                params.expected_physics_inventory_sha256
            ),
            source_asset=params.source_asset,
            grasp_prim_path=params.grasp_prim_path,
        )
        if (
            requirement == GATE3A_HYGIENE_REQUIREMENT
            and repair_result.passed
            and repair_result.report_path is not None
        ):
            grasp_source_lineage = _GSP001SourceLineage(
                receipt_path=repair_result.report_path,
                receipt_sha256=_file_sha256(repair_result.report_path),
            )
        latest_output = repair_result.output_path
        if repair_result.package_root is not None:
            latest_package_root = repair_result.package_root
        if repair_result.report_path is not None:
            repair_reports[requirement] = str(repair_result.report_path)
        if repair_result.passed:
            repaired.append(requirement)
        else:
            blocked.append(requirement)
        steps.append(
            _step(
                requirement=requirement,
                status=repair_result.status,
                passed=repair_result.passed,
                input_path=step_input,
                output_path=repair_result.output_path,
                upstream_skill=skill,
                upstream_skill_path=str(skill_path) if skill_path is not None else None,
                reason=repair_result.reason,
            )
        )

    if not selected_requirements and validation_report_path is None:
        warnings.append(
            "No validation report or explicit repair requirements were supplied; "
            "conformance was a staged no-op."
        )

    if params.expected_physics_inventory_sha256 is not None:
        try:
            from .gate3a_hygiene import inspect_gate3a_physics_inventory

            final_inventory = inspect_gate3a_physics_inventory(latest_output)
            if final_inventory.sha256 != params.expected_physics_inventory_sha256:
                errors.append(
                    "Final conformance output changed the expected physics "
                    "inventory: expected "
                    f"{params.expected_physics_inventory_sha256}, received "
                    f"{final_inventory.sha256}."
                )
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"Could not verify final physics inventory: {exc}")

    status = "FAIL" if errors else "BLOCKED" if blocked or skipped else "PASS"
    report = SimReadyConformanceReport(
        input_usd_path=str(asset_path),
        output_usd_path=str(latest_output),
        output_dir=str(output_dir),
        profile=params.profile,
        profile_version=params.profile_version,
        foundation_root=runtime.foundation_root,
        foundation_commit=runtime.foundation_commit,
        foundation_spec_root=runtime.foundation_spec_root,
        validation_report=str(validation_report_path)
        if validation_report_path
        else None,
        failed_requirements=failed_requirements,
        requirements_repaired=sorted(set(repaired)),
        requirements_blocked=sorted(set(blocked)),
        requirements_skipped=sorted(set(skipped)),
        steps=steps,
        reports=repair_reports,
        passed=status == "PASS",
        status=status,
        warnings=_dedupe([*warnings, *runtime.warnings]),
        errors=_dedupe(errors),
        next_step="simready-validate",
    )
    return _write_report(report_path, report)


def _repair_requirement(
    *,
    requirement: str,
    asset_path: Path,
    package_root: Path,
    output_dir: Path,
    grasp_plan_path: Path | None = None,
    grasp_source_asset_path: Path | None = None,
    grasp_source_lineage: _GSP001SourceLineage | None = None,
    expected_physics_inventory_sha256: str | None = None,
    source_asset: str | None = None,
    grasp_prim_path: str | None = None,
) -> _RepairResult:
    if requirement == GATE3A_HYGIENE_REQUIREMENT:
        from .gate3a_hygiene import repair_gate3a_hygiene

        result = repair_gate3a_hygiene(
            asset_path=asset_path,
            package_root=package_root,
            output_dir=output_dir,
            expected_physics_inventory_sha256=(expected_physics_inventory_sha256),
        )
        return _write_repair_result(
            requirement=requirement,
            asset_path=result.output_path,
            output_dir=output_dir,
            status=result.status,
            passed=result.passed,
            reason=result.reason,
            report=result.report,
            package_root=result.package_root,
        )
    if requirement == ATOMIC_ASSET_PATH_REQUIREMENT:
        return _repair_atomic_asset_paths(
            requirement=requirement,
            asset_path=asset_path,
            package_root=package_root,
            output_dir=output_dir,
        )
    if requirement == ISAAC_COMPOSITION_REQUIREMENT:
        return _repair_isaac_composition(
            requirement=requirement,
            asset_path=asset_path,
            package_root=package_root,
            output_dir=output_dir,
        )
    if requirement == "NP.006":
        return _repair_simready_metadata(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            source_asset=source_asset,
        )
    if requirement in {"UN.006", "UN.007"}:
        return _repair_stage_metrics(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            source_asset=source_asset,
        )
    if requirement in {"RB.COL.001", "RB.COL.002"}:
        return _repair_neutral_collision_schema(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
        )
    if requirement == "VM.MAT.001":
        return _repair_missing_visual_material_bindings(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
        )
    if requirement == "GSP.001":
        return _repair_missing_grasp_line(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            grasp_plan_path=grasp_plan_path,
            source_asset_path=grasp_source_asset_path,
            source_lineage=grasp_source_lineage,
            grasp_prim_path=grasp_prim_path,
        )
    if requirement == "PMT.001":
        return _repair_missing_physics_material_bindings(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
        )
    return _blocked_repair_result(
        requirement=requirement,
        asset_path=asset_path,
        output_dir=output_dir,
        reason=(
            "Foundation conformance route resolved. Automated repair is not "
            "implemented by this adapter for this requirement yet."
        ),
    )


def _repair_order_key(requirement: str) -> tuple[int, str]:
    if requirement == "NP.006":
        return (1, requirement)
    if requirement in {"UN.006", "UN.007"}:
        return (2, requirement)
    if requirement.startswith("RB.COL."):
        return (10, requirement)
    if requirement == "VM.MAT.001":
        return (20, requirement)
    if requirement == "PMT.001":
        return (30, requirement)
    if requirement == GATE3A_HYGIENE_REQUIREMENT:
        return (35, requirement)
    if requirement == "GSP.001":
        return (40, requirement)
    if requirement == ATOMIC_ASSET_PATH_REQUIREMENT:
        # Keep source-bound grasp evidence ahead of package-only rewrites, while
        # ensuring ISA.001 authors the final composition from AA-clean inputs.
        return (80, requirement)
    if requirement == ISAAC_COMPOSITION_REQUIREMENT:
        return (90, requirement)
    return (100, requirement)


def _expanded_repair_requirements(
    requirements: list[str], *, asset_path: Path | None = None
) -> set[str]:
    expanded = set(requirements)
    if "PMT.001" not in expanded and asset_path is not None:
        if "GSP.001" in expanded and _has_collision_api_prims(asset_path):
            expanded.add("PMT.001")
        if any(requirement.startswith("RB.COL.") for requirement in expanded):
            if _needs_pmt_after_collision_repair(asset_path):
                expanded.add("PMT.001")
    return expanded


def _has_collision_api_prims(asset_path: Path) -> bool:
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        return False

    stage, _opened_path, _error = _open_stage(asset_path, Usd)
    if stage is None:
        return False
    return any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stage.Traverse())


def _needs_pmt_after_collision_repair(asset_path: Path) -> bool:
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        return False

    stage, _opened_path, _error = _open_stage(asset_path, Usd)
    if stage is None:
        return False
    for prim in stage.Traverse():
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        has_mesh_collision = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        if not has_collision and not has_mesh_collision:
            continue
        if has_mesh_collision and prim.IsA(UsdGeom.Mesh) and not has_collision:
            return True
        invalid_collision = has_collision and not UsdGeom.Gprim(prim)
        invalid_mesh_collision = has_mesh_collision and not prim.IsA(UsdGeom.Mesh)
        if not invalid_collision and not invalid_mesh_collision:
            continue
        descendant_gprims = [
            child
            for child in Usd.PrimRange(prim)
            if child != prim and UsdGeom.Gprim(child)
        ]
        candidates = _collider_migration_candidates(
            descendant_gprims=descendant_gprims,
            requires_mesh=has_mesh_collision,
            UsdGeom=UsdGeom,
        )
        if _collider_migration_targets(prim, candidates):
            return True
    return False


def _open_stage(
    asset_path: Path, Usd: Any
) -> tuple[Any | None, Path | None, str | None]:
    open_path, path_error = _stage_open_path(asset_path)
    if open_path is None:
        return None, None, path_error
    try:
        stage = Usd.Stage.Open(str(open_path))
    except Exception as exc:  # pragma: no cover - OpenUSD exception types vary
        return None, open_path, f"Unable to open staged USD {open_path}: {exc}"
    if stage is None:
        return None, open_path, f"Unable to open staged USD: {open_path}"
    return stage, open_path, None


def _stage_open_path(asset_path: Path) -> tuple[Path | None, str | None]:
    if not asset_path.is_dir():
        return asset_path, None

    candidates = sorted(
        path
        for path in asset_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".usd", ".usda", ".usdc"}
    )
    named_candidates = [
        path for path in candidates if path.stem.lower() == asset_path.name.lower()
    ]
    if len(named_candidates) == 1:
        return named_candidates[0], None
    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return (
            None,
            f"Directory asset has no top-level USD file to open: {asset_path}",
        )
    return (
        None,
        "Directory asset has multiple top-level USD files and no unambiguous "
        f"root matching the directory name: {asset_path}",
    )


def _save_stage_root_layer(stage: Any) -> str | None:
    layer = stage.GetRootLayer()
    try:
        if layer.Save():
            return None
    except Exception as exc:  # pragma: no cover - OpenUSD exception types vary
        return f"Could not save repaired USD layer {layer.identifier}: {exc}"
    return f"Could not save repaired USD layer {layer.identifier}."


def _repair_atomic_asset_paths(
    *, requirement: str, asset_path: Path, package_root: Path, output_dir: Path
) -> _RepairResult:
    """Anchor real dependencies and remove identity-only asset paths atomically."""

    try:
        from pxr import Ar, Sdf, Usd, UsdUtils
    except ImportError as exc:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=f"OpenUSD Python APIs are unavailable: {exc}",
        )

    source_tree: Path | None = None
    source_root: Path | None = None
    extraction_dir: Path | None = None
    build_dir: Path | None = None
    report: dict[str, Any] = {
        "schema_version": "content-agent-workflows.simready-repair.v1",
        "requirement": requirement,
        "asset_path": str(asset_path),
    }
    try:
        source_tree, source_root, asset_root, extraction_dir = _aa001_source_package(
            asset_path=asset_path,
            package_root=package_root,
            output_dir=output_dir,
        )
        source_tree_sha256 = _isa001_tree_sha256(source_tree)
        source_inventory = _aa001_tree_inventory(source_tree)
        source_file_sha256 = _aa001_file_sha256(source_tree)
        source_stage = Usd.Stage.Open(str(source_root), load=Usd.Stage.LoadAll)
        if source_stage is None:
            raise ValueError(f"Unable to open staged USD: {source_root}")
        plan = _aa001_plan(
            stage=source_stage,
            source_root=source_root,
            tree_root=source_tree,
            asset_root=asset_root,
            Ar=Ar,
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        source_fingerprint = _aa001_stage_fingerprint(
            stage=source_stage,
            asset_root=asset_root,
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        source_stage = None
        gc.collect()

        report.update(
            {
                "source_root": str(source_root),
                "source_was_usdz": extraction_dir is not None,
                "source_tree_sha256": source_tree_sha256,
                "layers": list(plan.layers),
                "dependencies": list(plan.dependencies),
            }
        )
        if not plan.changed:
            unchanged_output = asset_path if extraction_dir is not None else source_root
            report.update(
                {
                    "changes": [],
                    "output_root": str(unchanged_output),
                    "output_tree_sha256": source_tree_sha256,
                    "source_stage_fingerprint": source_fingerprint,
                    "output_stage_fingerprint": source_fingerprint,
                    "remaining_findings": [],
                    "reused_output": True,
                }
            )
            return _write_repair_result(
                requirement=requirement,
                asset_path=unchanged_output,
                output_dir=output_dir,
                status="REPAIRED",
                passed=True,
                reason=(
                    "Asset paths already satisfy the deterministic AA.001 "
                    "local contract."
                ),
                report=report,
                package_root=package_root,
            )

        publish_root = output_dir / ISAAC_COMPOSITION_OUTPUT_DIR
        publish_root.mkdir(parents=True, exist_ok=True)
        build_dir = _private_mkdtemp(prefix=".aa001-build-", directory=publish_root)
        shutil.copytree(source_tree, build_dir, dirs_exist_ok=True)
        source_relative_root = source_root.relative_to(source_tree)
        asset_root_relative = asset_root.relative_to(source_tree)
        build_root = build_dir / source_relative_root
        build_asset_root = build_dir / asset_root_relative

        changes = _apply_aa001_plan(
            plan=plan,
            build_tree=build_dir,
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        if _aa001_tree_inventory(build_dir) != source_inventory:
            raise ValueError("AA.001 repair changed the copied package inventory.")
        build_file_sha256 = _aa001_file_sha256(build_dir)
        modified_layers = {
            *plan.replacements,
            *(opinion.layer_path for opinion in plan.identity_removals),
        }
        changed_unowned_files = sorted(
            path
            for path, digest in source_file_sha256.items()
            if path not in modified_layers and build_file_sha256.get(path) != digest
        )
        if changed_unowned_files:
            raise ValueError(
                "AA.001 repair changed bytes outside the authored USD layers: "
                + ", ".join(changed_unowned_files[:5])
            )

        output_stage = Usd.Stage.Open(str(build_root), load=Usd.Stage.LoadAll)
        if output_stage is None:
            raise ValueError(f"Unable to open authored AA.001 USD: {build_root}")
        remaining_plan = _aa001_plan(
            stage=output_stage,
            source_root=build_root,
            tree_root=build_dir,
            asset_root=build_asset_root,
            Ar=Ar,
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        if remaining_plan.changed:
            raise ValueError(
                "Authored AA.001 package still has repairable asset paths."
            )
        output_fingerprint = _aa001_stage_fingerprint(
            stage=output_stage,
            asset_root=build_asset_root,
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        output_stage = None
        gc.collect()
        if output_fingerprint != source_fingerprint:
            raise ValueError(
                "Authored AA.001 package changed composed USD content beyond "
                "anchored paths and stale asset identity metadata."
            )
        if _isa001_tree_sha256(source_tree) != source_tree_sha256:
            raise ValueError(
                "Staged USD package changed while AA.001 output was being built."
            )

        output_tree_sha256 = _isa001_tree_sha256(build_dir)
        final_tree, reused_output = _publish_aa001_tree(
            build_dir=build_dir,
            publish_root=publish_root,
            tree_sha256=output_tree_sha256,
        )
        build_dir = None
        final_root = final_tree / source_relative_root
        report.update(
            {
                "changes": changes,
                "output_root": str(final_root),
                "output_tree_sha256": output_tree_sha256,
                "source_stage_fingerprint": source_fingerprint,
                "output_stage_fingerprint": output_fingerprint,
                "remaining_findings": [],
                "reused_output": reused_output,
            }
        )
        return _write_repair_result(
            requirement=requirement,
            asset_path=final_root,
            output_dir=output_dir,
            status="REPAIRED",
            passed=True,
            reason="Published an atomic deterministic AA.001 asset package.",
            report=report,
            package_root=final_tree,
        )
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        report.update(
            {
                "status": "BLOCKED",
                "reason": str(exc),
                "changes": [],
            }
        )
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="BLOCKED",
            passed=False,
            reason=f"Could not safely author AA.001 asset paths: {exc}",
            report=report,
        )
    finally:
        if build_dir is not None:
            shutil.rmtree(build_dir, ignore_errors=True)
        if extraction_dir is not None:
            shutil.rmtree(extraction_dir, ignore_errors=True)


def _aa001_source_package(
    *, asset_path: Path, package_root: Path, output_dir: Path
) -> tuple[Path, Path, Path, Path | None]:
    if asset_path.suffix.lower() == ".usdz":
        extraction_dir = _private_mkdtemp(prefix=".aa001-source-", directory=output_dir)
        try:
            root_relative = _isa001_usdz_root(asset_path)
            extract_usdz_members_to_dir(
                asset_path,
                extraction_dir,
                allowed_suffixes=None,
                fail_on_filtered_member=True,
            )
            root_path = extraction_dir / root_relative
            if root_path.is_symlink() or not root_path.is_file():
                raise ValueError(
                    f"USDZ package root was not extracted: {root_relative.as_posix()}"
                )
        except BaseException:
            shutil.rmtree(extraction_dir, ignore_errors=True)
            raise
        return extraction_dir, root_path, root_path.parent, extraction_dir

    selected_root, path_error = _stage_open_path(asset_path)
    if selected_root is None:
        raise ValueError(path_error or f"Unable to select a USD root: {asset_path}")
    root_path = _absolute_path(selected_root)
    source_tree = _absolute_path(package_root)
    allowed_roots = (
        _absolute_path(output_dir / "staged"),
        _absolute_path(output_dir / ISAAC_COMPOSITION_OUTPUT_DIR),
        _absolute_path(output_dir / GSP001_OUTPUT_DIR),
        _absolute_path(output_dir / GATE3A_HYGIENE_OUTPUT_DIR),
    )
    if not any(_relative_to(source_tree, root) is not None for root in allowed_roots):
        raise ValueError(
            f"AA.001 package root is outside workflow-owned output: {source_tree}"
        )
    if source_tree.is_symlink() or not source_tree.is_dir():
        raise ValueError(
            f"AA.001 package root is not a regular directory: {source_tree}"
        )
    if _relative_to(root_path, source_tree) is None:
        raise ValueError(
            f"AA.001 repair input is outside its package root: {root_path}"
        )
    if root_path.is_symlink() or not root_path.is_file():
        raise ValueError(f"AA.001 package root is not a regular file: {root_path}")
    return source_tree, root_path, root_path.parent, None


def _aa001_plan(
    *,
    stage: Any,
    source_root: Path,
    tree_root: Path,
    asset_root: Path,
    Ar: Any,
    Sdf: Any,
    UsdUtils: Any,
) -> _AA001Plan:
    source_root = _absolute_path(source_root)
    tree_root = _absolute_path(tree_root)
    asset_root = _absolute_path(asset_root)
    layers: list[tuple[str, Any, Path]] = []
    try:
        dependency_layers, _assets, _unresolved = UsdUtils.ComputeAllDependencies(
            str(source_root)
        )
    except RuntimeError as exc:
        raise ValueError(
            f"AA.001 could not inspect the complete USD layer closure: {exc}"
        ) from exc
    for layer in dependency_layers:
        identifier = str(getattr(layer, "realPath", "") or "")
        if not identifier or identifier.startswith("anon:"):
            raise ValueError(
                "AA.001 dependency closure contains an in-memory layer: "
                f"{getattr(layer, 'identifier', identifier)}"
            )
        layer_path = _absolute_path(Path(identifier))
        relative_asset_path = _relative_to(layer_path, asset_root)
        relative_tree_path = _relative_to(layer_path, tree_root)
        if relative_asset_path is None or relative_tree_path is None:
            raise ValueError(
                f"AA.001 USD layer resolves outside the copied asset root: {layer_path}"
            )
        symlink = _aa001_symlink_component(layer_path, asset_root)
        if symlink is not None:
            raise ValueError(f"AA.001 USD layer uses a symlink: {symlink}")
        if not layer_path.is_file():
            raise ValueError(f"AA.001 USD layer is not a regular file: {layer_path}")
        layers.append((relative_tree_path.as_posix(), layer, layer_path))
    if not any(layer_path == source_root for _, _, layer_path in layers):
        raise ValueError(
            f"AA.001 dependency closure omitted its root layer: {source_root}"
        )
    layers.sort(key=lambda item: item[0])

    actual_by_layer: dict[str, dict[str, set[str]]] = {}
    identities: list[_AA001IdentityOpinion] = []
    layer_paths: dict[str, Path] = {}
    for relative_path, layer, layer_path in layers:
        actual, layer_identities = _aa001_layer_authored_paths(layer, Sdf=Sdf)
        actual_by_layer[relative_path] = actual
        layer_paths[relative_path] = layer_path
        identities.extend(
            _AA001IdentityOpinion(
                layer_path=relative_path,
                prim_path=prim_path,
                asset_path=asset_path,
            )
            for prim_path, asset_path in layer_identities
        )

    actual_texts = {
        authored_path for paths in actual_by_layer.values() for authored_path in paths
    }
    replacements: dict[str, dict[str, str]] = {}
    dependency_records: list[dict[str, Any]] = []
    actual_targets: set[Path] = set()
    for relative_path in sorted(actual_by_layer):
        layer_path = layer_paths[relative_path]
        for authored_path, dependency_types in sorted(
            actual_by_layer[relative_path].items()
        ):
            target, replacement = _aa001_dependency_target(
                authored_path=authored_path,
                layer_path=layer_path,
                asset_root=asset_root,
                allow_missing=False,
                Ar=Ar,
            )
            if target is None:  # pragma: no cover - allow_missing is false
                raise ValueError(f"AA.001 dependency is unresolved: {authored_path}")
            actual_targets.add(target)
            if replacement != authored_path:
                replacements.setdefault(relative_path, {})[authored_path] = replacement
            dependency_records.append(
                {
                    "layer": relative_path,
                    "dependency_types": sorted(dependency_types),
                    "authored_path": authored_path,
                    "output_path": replacement,
                    "resolved_path": target.relative_to(asset_root).as_posix(),
                }
            )

    identity_removals: list[_AA001IdentityOpinion] = []
    for opinion in sorted(
        identities,
        key=lambda item: (item.layer_path, item.prim_path, item.asset_path),
    ):
        if opinion.asset_path in actual_texts:
            raise ValueError(
                "AA.001 assetInfo.identifier overlaps an actual dependency: "
                f"{opinion.asset_path}"
            )
        layer_path = layer_paths[opinion.layer_path]
        target, _replacement = _aa001_dependency_target(
            authored_path=opinion.asset_path,
            layer_path=layer_path,
            asset_root=asset_root,
            allow_missing=True,
            Ar=Ar,
        )
        if target is None:
            identity_removals.append(opinion)
            continue
        if target in actual_targets:
            raise ValueError(
                "AA.001 assetInfo.identifier resolves to an actual dependency: "
                f"{opinion.asset_path}"
            )
        identity_removals.append(opinion)
        dependency_records.append(
            {
                "layer": opinion.layer_path,
                "dependency_types": ["assetInfo.identifier"],
                "authored_path": opinion.asset_path,
                "output_path": None,
                "resolved_path": target.relative_to(asset_root).as_posix(),
            }
        )

    return _AA001Plan(
        layers=tuple(relative_path for relative_path, _, _ in layers),
        dependencies=tuple(dependency_records),
        replacements={
            layer: dict(sorted(paths.items()))
            for layer, paths in sorted(replacements.items())
        },
        identity_removals=tuple(identity_removals),
    )


def _aa001_layer_authored_paths(
    layer: Any, *, Sdf: Any
) -> tuple[dict[str, set[str]], list[tuple[str, str]]]:
    dependencies: dict[str, set[str]] = {}
    identities: list[tuple[str, str]] = []

    def add_dependency(asset_path: str, dependency_type: str) -> None:
        if asset_path:
            dependencies.setdefault(asset_path, set()).add(dependency_type)

    for sublayer_path in layer.subLayerPaths:
        add_dependency(str(sublayer_path), "sublayer")

    def inspect(path: Any) -> None:
        spec = layer.GetObjectAtPath(path)
        if spec is None:
            return
        for key in spec.ListInfoKeys():
            if key == "subLayerOffsets":
                # The offsets are paired with subLayerPaths, which are inspected
                # through the typed layer API above. OpenUSD 25.5 has no Python
                # by-value converter for this metadata field.
                continue
            try:
                value = spec.GetInfo(key)
            except RuntimeError as exc:  # pragma: no cover - USD errors vary
                raise ValueError(
                    f"AA.001 could not inspect {spec.path} metadata {key}: {exc}"
                ) from exc
            if isinstance(spec, Sdf.PrimSpec | Sdf.VariantSpec) and key == "assetInfo":
                asset_info = value if isinstance(value, dict) else {}
                identifier = asset_info.get("identifier")
                if isinstance(identifier, Sdf.AssetPath) and identifier.path:
                    identities.append((str(spec.path), str(identifier.path)))
                for info_key, info_value in asset_info.items():
                    if info_key == "identifier":
                        continue
                    for asset_path, dependency_type in _aa001_paths_in_value(
                        info_value,
                        default_type="asset",
                        Sdf=Sdf,
                    ):
                        add_dependency(asset_path, dependency_type)
                continue
            default_type = _aa001_dependency_type_for_info_key(str(key))
            for asset_path, dependency_type in _aa001_paths_in_value(
                value,
                default_type=default_type,
                Sdf=Sdf,
            ):
                add_dependency(asset_path, dependency_type)
        if isinstance(spec, Sdf.AttributeSpec):
            for time_code in layer.ListTimeSamplesForPath(spec.path):
                value = layer.QueryTimeSample(spec.path, time_code)
                for asset_path, dependency_type in _aa001_paths_in_value(
                    value,
                    default_type="asset",
                    Sdf=Sdf,
                ):
                    add_dependency(asset_path, dependency_type)

    layer.Traverse(Sdf.Path.absoluteRootPath, inspect)
    return dependencies, identities


def _aa001_dependency_type_for_info_key(key: str) -> str:
    key = key.lower()
    if "reference" in key:
        return "reference"
    if "payload" in key:
        return "payload"
    return "asset"


def _aa001_paths_in_value(
    value: Any, *, default_type: str, Sdf: Any
) -> list[tuple[str, str]]:
    if isinstance(value, Sdf.AssetPath):
        return [(str(value.path), default_type)] if value.path else []
    if isinstance(value, Sdf.Reference):
        return [(str(value.assetPath), "reference")] if value.assetPath else []
    if isinstance(value, Sdf.Payload):
        return [(str(value.assetPath), "payload")] if value.assetPath else []
    if isinstance(value, dict):
        return [
            item
            for nested_value in value.values()
            for item in _aa001_paths_in_value(
                nested_value,
                default_type=default_type,
                Sdf=Sdf,
            )
        ]
    get_applied_items = getattr(value, "GetAppliedItems", None)
    if get_applied_items is not None:
        return [
            item
            for nested_value in get_applied_items()
            for item in _aa001_paths_in_value(
                nested_value,
                default_type=default_type,
                Sdf=Sdf,
            )
        ]
    sdf_asset_path_array = getattr(Sdf, "AssetPathArray", None)
    is_asset_path_array = (
        sdf_asset_path_array is not None and isinstance(value, sdf_asset_path_array)
    ) or (
        value.__class__.__module__ == "pxr.Vt"
        and value.__class__.__name__ == "AssetPathArray"
    )
    if isinstance(value, list | tuple) or is_asset_path_array:
        return [
            item
            for nested_value in value
            for item in _aa001_paths_in_value(
                nested_value,
                default_type=default_type,
                Sdf=Sdf,
            )
        ]
    return []


def _aa001_dependency_target(
    *,
    authored_path: str,
    layer_path: Path,
    asset_root: Path,
    allow_missing: bool,
    Ar: Any,
) -> tuple[Path | None, str]:
    text = str(authored_path)
    if not text or text != text.strip():
        raise ValueError(f"AA.001 asset path is empty or ambiguous: {text!r}")
    if (
        Path(text).is_absolute()
        or re.match(r"^[A-Za-z]:[\\/]", text)
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", text)
        or text.startswith(("//", "~"))
        or "\\" in text
        or "[" in text
        or "]" in text
    ):
        raise ValueError(f"AA.001 blocks absolute, URL, or package asset path: {text}")

    layer_path = _absolute_path(layer_path)
    asset_root = _absolute_path(asset_root)
    target = _absolute_path(layer_path.parent / Path(text))
    if _relative_to(target, asset_root) is None:
        raise ValueError(f"AA.001 asset path resolves outside the asset root: {text}")
    symlink = _aa001_symlink_component(target, asset_root)
    if symlink is not None:
        raise ValueError(f"AA.001 asset path uses a symlink: {text} -> {symlink}")
    if target == layer_path:
        if allow_missing:
            replacement = text if text.startswith(("./", "../")) else f"./{text}"
            return target, replacement
        raise ValueError(f"AA.001 asset path is a self-reference: {text}")

    try:
        resolver = Ar.GetResolver()
        identifier = resolver.CreateIdentifier(
            text,
            Ar.ResolvedPath(str(layer_path)),
        )
        resolved = resolver.Resolve(identifier)
        resolved_text = resolved.GetPathString() if resolved else ""
    except Exception as exc:  # pragma: no cover - resolver errors vary by plugin
        raise ValueError(f"AA.001 could not resolve asset path {text}: {exc}") from exc
    if resolved_text:
        if "[" in resolved_text or "]" in resolved_text:
            raise ValueError(f"AA.001 asset path is package-relative: {text}")
        resolved_path = _absolute_path(Path(resolved_text))
        if resolved_path != target:
            raise ValueError(
                "AA.001 blocks resolver search paths: "
                f"{text} resolved to {resolved_path} instead of {target}"
            )
    if not target.is_file() or not resolved_text:
        if allow_missing and not target.exists() and not resolved_text:
            replacement = text if text.startswith(("./", "../")) else f"./{text}"
            return None, replacement
        raise ValueError(f"AA.001 actual dependency is unresolved: {text}")

    replacement = text if text.startswith(("./", "../")) else f"./{text}"
    return target, replacement


def _aa001_symlink_component(path: Path, root: Path) -> Path | None:
    path = _absolute_path(path)
    root = _absolute_path(root)
    relative = _relative_to(path, root)
    if relative is None:
        return None
    if root.is_symlink():
        return root
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return current
    return None


def _apply_aa001_plan(
    *, plan: _AA001Plan, build_tree: Path, Sdf: Any, UsdUtils: Any
) -> list[dict[str, Any]]:
    removals_by_layer: dict[str, list[_AA001IdentityOpinion]] = {}
    for opinion in plan.identity_removals:
        removals_by_layer.setdefault(opinion.layer_path, []).append(opinion)
    modified_layers = sorted({*plan.replacements, *removals_by_layer})
    changes: list[dict[str, Any]] = []
    for relative_path in modified_layers:
        layer_path = build_tree / relative_path
        layer = Sdf.Layer.FindOrOpen(str(layer_path))
        if layer is None:
            raise ValueError(f"Could not open copied AA.001 layer: {layer_path}")
        replacements = plan.replacements.get(relative_path, {})
        if replacements:
            UsdUtils.ModifyAssetPaths(
                layer,
                lambda path: replacements.get(str(path), str(path)),
            )
            changes.extend(
                {
                    "action": "anchor_asset_path",
                    "layer": relative_path,
                    "source_path": source_path,
                    "output_path": output_path,
                }
                for source_path, output_path in sorted(replacements.items())
            )
        for opinion in removals_by_layer.get(relative_path, []):
            identity_spec = layer.GetObjectAtPath(Sdf.Path(opinion.prim_path))
            if not isinstance(identity_spec, Sdf.PrimSpec | Sdf.VariantSpec):
                raise ValueError(
                    "Could not find copied asset identity spec: "
                    f"{relative_path}:{opinion.prim_path}"
                )
            asset_info = identity_spec.GetInfo("assetInfo")
            asset_info = dict(asset_info) if isinstance(asset_info, dict) else {}
            identifier = asset_info.get("identifier")
            identifier_path = (
                str(identifier.path) if isinstance(identifier, Sdf.AssetPath) else ""
            )
            if identifier_path != opinion.asset_path:
                raise ValueError(
                    "Copied asset identity changed before AA.001 repair: "
                    f"{relative_path}:{opinion.prim_path}"
                )
            del asset_info["identifier"]
            if asset_info:
                identity_spec.SetInfo("assetInfo", asset_info)
            else:
                identity_spec.ClearInfo("assetInfo")
            changes.append(
                {
                    "action": "remove_asset_info_identifier",
                    "layer": relative_path,
                    "prim_path": opinion.prim_path,
                    "source_path": opinion.asset_path,
                }
            )
        if not layer.Save():
            raise OSError(f"Could not save copied AA.001 layer: {layer_path}")
    return changes


def _aa001_tree_inventory(root: Path) -> tuple[str, ...]:
    return tuple(
        ("D:" if path.is_dir() else "F:") + path.relative_to(root).as_posix()
        for path in sorted(
            root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
        )
    )


def _aa001_file_sha256(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        result[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return result


def _aa001_stage_fingerprint(
    *, stage: Any, asset_root: Path, Sdf: Any, UsdUtils: Any
) -> str:
    flattened = stage.Flatten()
    flattened.documentation = ""

    def clear_identity(path: Any) -> None:
        spec = flattened.GetObjectAtPath(path)
        if not isinstance(spec, Sdf.PrimSpec) or not spec.HasInfo("assetInfo"):
            return
        asset_info = spec.GetInfo("assetInfo")
        if not isinstance(asset_info, dict) or "identifier" not in asset_info:
            return
        asset_info = dict(asset_info)
        del asset_info["identifier"]
        if asset_info:
            spec.SetInfo("assetInfo", asset_info)
        else:
            spec.ClearInfo("assetInfo")

    flattened.Traverse(Sdf.Path.absoluteRootPath, clear_identity)
    stage_root = Path(stage.GetRootLayer().realPath).parent
    asset_root = _absolute_path(asset_root)

    def canonicalize(asset_path: str) -> str:
        text = str(asset_path)
        if not text:
            return text
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", text) or "[" in text or "]" in text:
            raise ValueError(f"Flattened stage contains non-local asset path: {text}")
        path = Path(text)
        path = path if path.is_absolute() else stage_root / path
        relative = _relative_to(_absolute_path(path), asset_root)
        if relative is None:
            raise ValueError(f"Flattened stage asset path escapes the package: {text}")
        return f"__AA001_PACKAGE__/{relative.as_posix()}"

    UsdUtils.ModifyAssetPaths(flattened, canonicalize)
    return hashlib.sha256(flattened.ExportToString().encode("utf-8")).hexdigest()


def _publish_aa001_tree(
    *, build_dir: Path, publish_root: Path, tree_sha256: str
) -> tuple[Path, bool]:
    return _publish_isa001_tree(
        build_dir=build_dir,
        publish_root=publish_root,
        tree_sha256=tree_sha256,
    )


def _repair_isaac_composition(
    *, requirement: str, asset_path: Path, package_root: Path, output_dir: Path
) -> _RepairResult:
    """Publish a source-preserving deterministic ISA.001 package derivative.

    The main layer retains every source opinion. The required meshes, base, and
    physics layers are structural overlays, not standalone semantic exports.
    The complete package tree is copied without rebasing authored asset paths,
    then verified by dependency closure, composed-stage fingerprint, and tree
    hash before a same-parent atomic rename publishes it.
    """

    try:
        from pxr import Kind, Sdf, Usd, UsdUtils
    except ImportError as exc:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=f"OpenUSD Python APIs are unavailable: {exc}",
        )

    source_tree: Path | None = None
    source_root: Path | None = None
    extraction_dir: Path | None = None
    build_dir: Path | None = None
    report: dict[str, Any] = {
        "schema_version": "content-agent-workflows.simready-repair.v1",
        "requirement": requirement,
        "asset_path": str(asset_path),
    }
    try:
        source_tree, source_root, extraction_dir = _isa001_source_package(
            asset_path=asset_path,
            package_root=package_root,
            output_dir=output_dir,
        )
        report["source_root"] = str(source_root)
        report["source_was_usdz"] = extraction_dir is not None

        stage = Usd.Stage.Open(str(source_root), load=Usd.Stage.LoadAll)
        if stage is None:
            raise ValueError(f"Unable to open staged USD: {source_root}")
        default_prim = stage.GetDefaultPrim()
        compliance_findings = _isa001_compliance_findings(
            stage=stage,
            root_path=source_root,
            Kind=Kind,
            Sdf=Sdf,
            Usd=Usd,
        )
        already_compliant = not compliance_findings
        report["initial_findings"] = compliance_findings

        ignored_identity_paths = _validate_isa001_dependency_closure(
            stage=stage,
            source_root=source_root,
            source_tree=source_tree,
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        report["ignored_unresolved_asset_identity_paths"] = list(ignored_identity_paths)
        source_tree_sha256 = _isa001_tree_sha256(source_tree)
        report["source_tree_sha256"] = source_tree_sha256

        if already_compliant and extraction_dir is None:
            report.update(
                {
                    "changes": [],
                    "output_root": str(source_root),
                    "output_tree_sha256": source_tree_sha256,
                    "reused_output": True,
                }
            )
            return _write_repair_result(
                requirement=requirement,
                asset_path=source_root,
                output_dir=output_dir,
                status="REPAIRED",
                passed=True,
                reason=(
                    "Isaac Sim composition already satisfies the deterministic "
                    "ISA.001 local contract."
                ),
                report=report,
                package_root=source_tree,
            )

        if already_compliant:
            root_prim = default_prim
        else:
            root_prim = _isa001_repair_root_prim(stage)
            _validate_isa001_output_targets(source_root)
        root_path = root_prim.GetPath()
        root_name = root_prim.GetName()
        top_level_paths = tuple(
            prim.GetPath() for prim in stage.GetPseudoRoot().GetAllChildren()
        )
        root_layer = stage.GetRootLayer()
        layer_metadata = _isa001_layer_metadata(root_layer)
        source_fingerprint = _isa001_stage_fingerprint(
            stage=stage,
            root_path=root_path,
            package_root=source_tree,
            UsdUtils=UsdUtils,
        )
        source_relative_root = source_root.relative_to(source_tree)
        # Usd.Stage has no Python Close API. Drop every stage-owned handle before
        # copying or replacing package files so CPython releases the C++ objects.
        del root_layer, root_prim, default_prim, stage

        publish_root = output_dir / ISAAC_COMPOSITION_OUTPUT_DIR
        publish_root.mkdir(parents=True, exist_ok=True)
        build_dir = _private_mkdtemp(prefix=".isa001-build-", directory=publish_root)
        shutil.copytree(source_tree, build_dir, dirs_exist_ok=True)
        build_root = build_dir / source_relative_root

        changes: list[dict[str, Any]] = []
        if not already_compliant:
            changes = _author_isa001_composition(
                build_root=build_root,
                build_tree=build_dir,
                root_path=root_path,
                root_name=root_name,
                top_level_paths=top_level_paths,
                ignored_identity_paths=ignored_identity_paths,
                layer_metadata=layer_metadata,
                Kind=Kind,
                Sdf=Sdf,
            )

        output_stage = Usd.Stage.Open(str(build_root), load=Usd.Stage.LoadAll)
        if output_stage is None:
            raise ValueError(f"Unable to open authored ISA.001 USD: {build_root}")
        remaining_findings = _isa001_compliance_findings(
            stage=output_stage,
            root_path=build_root,
            Kind=Kind,
            Sdf=Sdf,
            Usd=Usd,
        )
        if remaining_findings:
            raise ValueError(
                "Authored ISA.001 package failed local structural verification: "
                + "; ".join(remaining_findings)
            )
        output_fingerprint = _isa001_stage_fingerprint(
            stage=output_stage,
            root_path=root_path,
            package_root=build_dir,
            UsdUtils=UsdUtils,
        )
        if output_fingerprint != source_fingerprint:
            raise ValueError(
                "Authored ISA.001 package changed composed USD content beyond "
                "the required defaultPrim/kind metadata."
            )
        # See the source-stage lifetime note above.
        del output_stage

        if _isa001_tree_sha256(source_tree) != source_tree_sha256:
            raise ValueError(
                "Staged USD package changed while ISA.001 output was being built."
            )
        output_tree_sha256 = _isa001_tree_sha256(build_dir)
        final_tree, reused_output = _publish_isa001_tree(
            build_dir=build_dir,
            publish_root=publish_root,
            tree_sha256=output_tree_sha256,
        )
        build_dir = None
        final_root = final_tree / source_relative_root
        report.update(
            {
                "changes": changes,
                "root_prim": str(root_path),
                "output_root": str(final_root),
                "output_tree_sha256": output_tree_sha256,
                "source_stage_fingerprint": source_fingerprint,
                "output_stage_fingerprint": output_fingerprint,
                "remaining_findings": [],
                "reused_output": reused_output,
            }
        )
        return _write_repair_result(
            requirement=requirement,
            asset_path=final_root,
            output_dir=output_dir,
            status="REPAIRED",
            passed=True,
            reason=(
                "Published an atomic deterministic ISA.001 composition package."
                if changes
                else "Published an extracted ISA.001-compliant USDZ package."
            ),
            report=report,
            package_root=final_tree,
        )
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        report.update(
            {
                "status": "BLOCKED",
                "reason": str(exc),
                "changes": [],
            }
        )
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="BLOCKED",
            passed=False,
            reason=f"Could not safely author ISA.001 composition: {exc}",
            report=report,
        )
    finally:
        if build_dir is not None:
            shutil.rmtree(build_dir, ignore_errors=True)
        if extraction_dir is not None:
            shutil.rmtree(extraction_dir, ignore_errors=True)


def _isa001_source_package(
    *, asset_path: Path, package_root: Path, output_dir: Path
) -> tuple[Path, Path, Path | None]:
    """Return the complete local package tree and selected root layer."""

    if asset_path.suffix.lower() == ".usdz":
        extraction_dir = _private_mkdtemp(
            prefix=".isa001-source-", directory=output_dir
        )
        try:
            root_relative = _isa001_usdz_root(asset_path)
            extract_usdz_members_to_dir(
                asset_path,
                extraction_dir,
                allowed_suffixes=None,
                fail_on_filtered_member=True,
            )
            root_path = extraction_dir / root_relative
            if not root_path.is_file():
                raise ValueError(
                    f"USDZ package root was not extracted: {root_relative.as_posix()}"
                )
        except BaseException:
            shutil.rmtree(extraction_dir, ignore_errors=True)
            raise
        return extraction_dir, root_path, extraction_dir

    selected_root, path_error = _stage_open_path(asset_path)
    if selected_root is None:
        raise ValueError(path_error or f"Unable to select a USD root: {asset_path}")
    staging_root = _absolute_path(package_root)
    if staging_root.is_symlink() or not staging_root.is_dir():
        raise ValueError(
            f"ISA.001 package root is not a regular directory: {staging_root}"
        )
    root_path = _absolute_path(selected_root)
    if _relative_to(root_path, staging_root) is None:
        raise ValueError(
            f"ISA.001 repair input is outside its package root: {root_path}"
        )
    return staging_root, root_path, None


def _isa001_usdz_root(asset_path: Path) -> Path:
    """Return the USDZ entry point, which the format defines as its first file."""

    try:
        with zipfile.ZipFile(asset_path) as archive:
            root_info = next(
                (info for info in archive.infolist() if not info.is_dir()),
                None,
            )
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError(f"Malformed USDZ archive {asset_path}: {exc}") from exc
    if root_info is None:
        raise ValueError(f"USDZ package has no package root: {asset_path}")
    root_parts = safe_usdz_member_parts(root_info.filename)
    if root_parts is None:
        raise ValueError(f"USDZ package root has an unsafe path: {root_info.filename}")
    root_path = Path(*root_parts)
    if root_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError(
            f"USDZ package root is not a supported USD layer: {root_path.as_posix()}"
        )
    return root_path


def _validate_isa001_dependency_closure(
    *, stage: Any, source_root: Path, source_tree: Path, Sdf: Any, UsdUtils: Any
) -> tuple[str, ...]:
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(source_root))
    except Exception as exc:  # pragma: no cover - OpenUSD exception types vary
        raise ValueError(
            f"Could not inspect the USD dependency closure: {exc}"
        ) from exc
    metadata_only_paths = _isa001_metadata_only_asset_identity_paths(
        stage,
        Sdf=Sdf,
    )
    unresolved_paths = tuple(sorted(str(item) for item in unresolved))
    blocking_unresolved = tuple(
        path for path in unresolved_paths if path not in metadata_only_paths
    )
    if blocking_unresolved:
        sample = ", ".join(blocking_unresolved[:5])
        suffix = "" if len(blocking_unresolved) <= 5 else ", ..."
        raise ValueError(f"USD dependency closure is unresolved: {sample}{suffix}")

    dependency_paths: list[Path] = []
    for layer in layers:
        identifier = str(
            getattr(layer, "realPath", "") or getattr(layer, "identifier", "") or ""
        )
        if not identifier or identifier.startswith("anon:"):
            raise ValueError(
                f"USD dependency layer has no stable local path: {identifier or layer}"
            )
        dependency_paths.append(Path(identifier))
    for asset in assets:
        authored_identifier = str(getattr(asset, "path", "") or asset)
        if authored_identifier in metadata_only_paths:
            continue
        identifier = str(
            getattr(asset, "resolvedPath", "") or getattr(asset, "path", "") or asset
        )
        if not identifier:
            continue
        path = Path(identifier)
        if not path.is_absolute():
            raise ValueError(
                "USD dependency has no stable resolved local path: "
                f"{authored_identifier}"
            )
        dependency_paths.append(path)

    source_tree = _absolute_path(source_tree)
    for dependency in dependency_paths:
        dependency = _absolute_path(dependency)
        if _relative_to(dependency, source_tree) is None:
            raise ValueError(
                f"USD dependency resolves outside the staged package: {dependency}"
            )
        if dependency.is_symlink() or not dependency.is_file():
            raise ValueError(
                f"USD dependency is not a regular staged file: {dependency}"
            )
    return tuple(sorted(metadata_only_paths))


def _isa001_metadata_only_asset_identity_paths(stage: Any, *, Sdf: Any) -> set[str]:
    """Return unresolved paths used only as informational asset identity."""

    identity_paths: set[str] = set()
    authored_dependency_paths: set[str] = set()
    for prim in stage.TraverseAll():
        identifier = prim.GetAssetInfo().get("identifier")
        if isinstance(identifier, Sdf.AssetPath) and identifier.path:
            identity_paths.add(str(identifier.path))
        for attribute in prim.GetAttributes():
            value = attribute.Get()
            if isinstance(value, Sdf.AssetPath) and value.path:
                authored_dependency_paths.add(str(value.path))
            elif isinstance(value, list | tuple):
                authored_dependency_paths.update(
                    str(item.path)
                    for item in value
                    if isinstance(item, Sdf.AssetPath) and item.path
                )
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        authored_dependency_paths.update(str(path) for path in layer.subLayerPaths)

        def collect_composition_arcs(path: Any) -> None:
            spec = layer.GetObjectAtPath(path)
            if not isinstance(spec, Sdf.PrimSpec):
                return
            authored_dependency_paths.update(
                str(item.assetPath)
                for item in spec.referenceList.GetAppliedItems()
                if item.assetPath
            )
            authored_dependency_paths.update(
                str(item.assetPath)
                for item in spec.payloadList.GetAppliedItems()
                if item.assetPath
            )

        layer.Traverse(Sdf.Path.absoluteRootPath, collect_composition_arcs)
    return identity_paths - authored_dependency_paths


def _isa001_repair_root_prim(stage: Any) -> Any:
    root_layer = stage.GetRootLayer()
    default_prim = stage.GetDefaultPrim()
    if root_layer.defaultPrim and not default_prim:
        raise ValueError(
            "The authored defaultPrim does not resolve to a composed prim: "
            f"{root_layer.defaultPrim}"
        )
    top_level_prims = list(stage.GetPseudoRoot().GetAllChildren())
    if default_prim:
        root_prim = default_prim
    elif len(top_level_prims) == 1:
        root_prim = top_level_prims[0]
    else:
        raise ValueError(
            "ISA.001 repair requires a valid default prim when the stage has "
            f"multiple composed top-level prims; found {len(top_level_prims)}."
        )
    if root_prim.GetParent() != stage.GetPseudoRoot():
        raise ValueError("defaultPrim must identify a top-level prim for ISA.001.")
    if (
        not root_prim.IsValid()
        or not root_prim.IsActive()
        or not root_prim.IsDefined()
        or root_prim.IsAbstract()
    ):
        raise ValueError(
            f"ISA.001 repair root is not an active defined prim: {root_prim.GetPath()}"
        )
    return root_prim


def _validate_isa001_output_targets(source_root: Path) -> None:
    payload_dir = source_root.parent / ISAAC_COMPOSITION_PAYLOAD_DIR
    if payload_dir.exists() and (payload_dir.is_symlink() or not payload_dir.is_dir()):
        raise ValueError(
            f"ISA.001 payload target is not a regular directory: {payload_dir}"
        )
    stem = source_root.stem
    target_paths = [
        payload_dir / f"{stem}_{suffix}.usd" for suffix in ("meshes", "base", "physics")
    ]
    collisions = [path for path in target_paths if path.exists() or path.is_symlink()]
    if collisions:
        raise ValueError(
            "ISA.001 target layer already exists in a non-compliant package: "
            + ", ".join(str(path) for path in collisions)
        )


def _isa001_layer_metadata(layer: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in layer.pseudoRoot.ListInfoKeys():
        if key in _ISAAC_COMPOSITION_LAYER_METADATA_EXCLUSIONS:
            continue
        metadata[key] = layer.pseudoRoot.GetInfo(key)
    return metadata


def _author_isa001_composition(
    *,
    build_root: Path,
    build_tree: Path,
    root_path: Any,
    root_name: str,
    top_level_paths: tuple[Any, ...],
    ignored_identity_paths: tuple[str, ...],
    layer_metadata: dict[str, Any],
    Kind: Any,
    Sdf: Any,
) -> list[dict[str, Any]]:
    payload_dir = build_root.parent / ISAAC_COMPOSITION_PAYLOAD_DIR
    payload_dir.mkdir(parents=True, exist_ok=True)
    stem = build_root.stem
    meshes_path = payload_dir / f"{stem}_meshes.usd"
    base_path = payload_dir / f"{stem}_base.usd"
    physics_path = payload_dir / f"{stem}_physics.usd"

    # Keep all source opinions in the main layer. Moving an arbitrary root behind
    # a reference can invalidate relationships that target sibling top-level prims.
    # The required layers are therefore structural overlays until a specialized
    # producer can prove that a semantic geometry/physics split is lossless.
    meshes_layer, _meshes_prim = _new_isa001_layer(
        path=meshes_path,
        root_path=root_path,
        root_name=root_name,
        layer_metadata=layer_metadata,
        specifier=Sdf.SpecifierDef,
        Sdf=Sdf,
    )
    _export_isa001_layer(meshes_layer, meshes_path)

    base_layer, base_prim = _new_isa001_layer(
        path=base_path,
        root_path=root_path,
        root_name=root_name,
        layer_metadata=layer_metadata,
        specifier=Sdf.SpecifierDef,
        Sdf=Sdf,
    )
    base_prim.referenceList.prependedItems = [
        Sdf.Reference(f"./{meshes_path.name}", root_path)
    ]
    _export_isa001_layer(base_layer, base_path)

    physics_layer, _physics_prim = _new_isa001_layer(
        path=physics_path,
        root_path=root_path,
        root_name=root_name,
        layer_metadata=layer_metadata,
        specifier=Sdf.SpecifierDef,
        Sdf=Sdf,
    )
    _export_isa001_layer(physics_layer, physics_path)

    main_layer = Sdf.Layer.OpenAsAnonymous(str(build_root))
    if main_layer is None:
        raise ValueError(f"Could not clone source USD layer: {build_root}")
    main_layer.defaultPrim = root_name
    main_prim = main_layer.GetPrimAtPath(root_path)
    if main_prim is None:
        raise ValueError(f"Could not find default prim spec at {root_path}")
    main_prim.SetInfo("kind", Kind.Tokens.component)
    _prepend_isa001_list_item(
        main_prim.referenceList,
        Sdf.Reference(f"./{ISAAC_COMPOSITION_PAYLOAD_DIR}/{base_path.name}", root_path),
    )
    _prepend_isa001_list_item(
        main_prim.payloadList,
        Sdf.Payload(
            f"./{ISAAC_COMPOSITION_PAYLOAD_DIR}/{physics_path.name}", root_path
        ),
    )
    sibling_paths = sorted(
        (path for path in top_level_paths if path != root_path),
        key=str,
    )
    source_format = str(main_layer.GetFileFormat().formatId)
    export_args = {"format": source_format} if source_format in {"usda", "usdc"} else {}
    if not main_layer.Export(str(build_root), args=export_args):
        raise OSError(f"Could not write ISA.001 main layer: {build_root}")

    return [
        {
            "action": "create_isaac_composition_layers",
            "root_prim": str(root_path),
            "main_layer": build_root.relative_to(build_tree).as_posix(),
            "meshes_layer": meshes_path.relative_to(build_tree).as_posix(),
            "base_layer": base_path.relative_to(build_tree).as_posix(),
            "physics_layer": physics_path.relative_to(build_tree).as_posix(),
            "preserved_sibling_top_level_prims": [str(path) for path in sibling_paths],
            "ignored_asset_identity_paths": list(ignored_identity_paths),
        }
    ]


def _prepend_isa001_list_item(list_op: Any, item: Any) -> None:
    if list_op.isExplicit:
        list_op.explicitItems = [item, *list_op.explicitItems]
    else:
        list_op.prependedItems = [item, *list_op.prependedItems]


def _new_isa001_layer(
    *,
    path: Path,
    root_path: Any,
    root_name: str,
    layer_metadata: dict[str, Any],
    specifier: Any,
    Sdf: Any,
) -> tuple[Any, Any]:
    layer = Sdf.Layer.CreateAnonymous(path.name)
    for key, value in layer_metadata.items():
        try:
            layer.pseudoRoot.SetInfo(key, value)
        except Exception as exc:  # pragma: no cover - metadata types vary by USD
            raise ValueError(
                f"Could not preserve root-layer metadata {key!r}: {exc}"
            ) from exc
    layer.defaultPrim = root_name
    prim = Sdf.CreatePrimInLayer(layer, root_path)
    prim.specifier = specifier
    return layer, prim


def _export_isa001_layer(layer: Any, path: Path) -> None:
    if not layer.Export(str(path)):
        raise OSError(f"Could not write ISA.001 layer: {path}")


def _isa001_compliance_findings(
    *, stage: Any, root_path: Path, Kind: Any, Sdf: Any, Usd: Any
) -> list[str]:
    findings: list[str] = []
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        return ["Stage has no valid defaultPrim."]
    prim_path = default_prim.GetPath()
    if Usd.ModelAPI(default_prim).GetKind() != Kind.Tokens.component:
        findings.append("defaultPrim kind is not component.")
    prim_spec = stage.GetRootLayer().GetPrimAtPath(prim_path)
    if prim_spec is None:
        findings.append("Root layer has no direct defaultPrim spec.")
        return findings

    stem = root_path.stem
    payload_dir = root_path.parent / ISAAC_COMPOSITION_PAYLOAD_DIR
    meshes_path = payload_dir / f"{stem}_meshes.usd"
    base_path = payload_dir / f"{stem}_base.usd"
    physics_path = payload_dir / f"{stem}_physics.usd"
    for path in (meshes_path, base_path, physics_path):
        if path.is_symlink() or not path.is_file():
            findings.append(f"Missing regular ISA.001 layer: {path.name}")

    expected_base = f"{ISAAC_COMPOSITION_PAYLOAD_DIR}/{base_path.name}"
    expected_physics = f"{ISAAC_COMPOSITION_PAYLOAD_DIR}/{physics_path.name}"
    references = prim_spec.referenceList.GetAddedOrExplicitItems()
    payloads = prim_spec.payloadList.GetAddedOrExplicitItems()
    if not any(
        _isa001_arc_matches(reference, expected_base, prim_path)
        for reference in references
    ):
        findings.append("defaultPrim lacks the direct _base.usd reference.")
    if not any(
        _isa001_arc_matches(payload, expected_physics, prim_path)
        for payload in payloads
    ):
        findings.append("defaultPrim lacks the direct _physics.usd payload.")

    if base_path.is_file() and not base_path.is_symlink():
        base_layer = Sdf.Layer.FindOrOpen(str(base_path))
        base_prim = base_layer.GetPrimAtPath(prim_path) if base_layer else None
        if base_layer is None or base_layer.defaultPrim != default_prim.GetName():
            findings.append("Base layer has no matching defaultPrim.")
        elif base_prim is None or not any(
            _isa001_arc_matches(reference, meshes_path.name, prim_path)
            for reference in base_prim.referenceList.GetAddedOrExplicitItems()
        ):
            findings.append("Base layer lacks the direct _meshes.usd reference.")

    for label, path in (("meshes", meshes_path), ("physics", physics_path)):
        if not path.is_file() or path.is_symlink():
            continue
        layer_stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
        layer_default = layer_stage.GetDefaultPrim() if layer_stage else None
        if not layer_default or layer_default.GetPath() != prim_path:
            findings.append(f"{label.capitalize()} layer has no matching defaultPrim.")
    return findings


def _isa001_arc_matches(arc: Any, expected_asset_path: str, prim_path: Any) -> bool:
    asset_path = str(arc.assetPath).replace("\\", "/")
    if asset_path.startswith("./"):
        asset_path = asset_path[2:]
    return asset_path == expected_asset_path and str(arc.primPath) in {
        "",
        str(prim_path),
    }


def _isa001_stage_fingerprint(
    *, stage: Any, root_path: Any, package_root: Path, UsdUtils: Any
) -> str:
    flattened = stage.Flatten()
    flattened.documentation = ""
    flattened.defaultPrim = root_path.name
    root_spec = flattened.GetPrimAtPath(root_path)
    if root_spec is None:
        raise ValueError(f"Flattened stage has no repair root: {root_path}")
    if root_spec.HasInfo("kind"):
        root_spec.ClearInfo("kind")
    stage_root = Path(stage.GetRootLayer().realPath).parent

    def canonicalize(asset_path: str) -> str:
        text = str(asset_path).strip()
        if not text:
            return text
        if "://" in text or text.startswith(("anon:", "file:")):
            raise ValueError(f"Flattened stage contains non-local asset path: {text}")
        path = Path(text)
        path = path if path.is_absolute() else stage_root / path
        relative = _relative_to(_absolute_path(path), _absolute_path(package_root))
        if relative is None:
            raise ValueError(f"Flattened stage asset path escapes the package: {text}")
        return f"__ISA001_PACKAGE__/{relative.as_posix()}"

    UsdUtils.ModifyAssetPaths(flattened, canonicalize)
    return hashlib.sha256(flattened.ExportToString().encode("utf-8")).hexdigest()


def _isa001_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Conformance package tree contains a symlink: {path}")
        if path.is_dir():
            digest.update(b"D\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            continue
        if not path.is_file():
            raise ValueError(f"Conformance package tree contains a non-file: {path}")
        digest.update(b"F\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _publish_isa001_tree(
    *, build_dir: Path, publish_root: Path, tree_sha256: str
) -> tuple[Path, bool]:
    """Atomically publish a verified tree from a sibling temporary directory."""

    build_dir = _absolute_path(build_dir)
    publish_root = _absolute_path(publish_root)
    if build_dir.is_symlink() or not build_dir.is_dir():
        raise ValueError(f"ISA.001 build tree is not a regular directory: {build_dir}")
    if publish_root.is_symlink() or not publish_root.is_dir():
        raise ValueError(
            f"ISA.001 publish root is not a regular directory: {publish_root}"
        )
    if build_dir.parent != publish_root:
        raise ValueError(
            "ISA.001 atomic publication requires the build tree to be a direct "
            f"child of the publish root: {build_dir}"
        )
    final_tree = publish_root / tree_sha256
    if final_tree.exists() or final_tree.is_symlink():
        if final_tree.is_symlink() or not final_tree.is_dir():
            raise ValueError(
                f"Conformance content-addressed output is not a directory: {final_tree}"
            )
        if _isa001_tree_sha256(final_tree) != tree_sha256:
            raise ValueError(
                "Existing conformance content-addressed output failed identity check: "
                f"{final_tree}"
            )
        shutil.rmtree(build_dir)
        return final_tree, True
    try:
        # The source and destination share one parent, so the OS rename is an
        # atomic same-filesystem publication. An EEXIST race is verified below.
        build_dir.replace(final_tree)
    except OSError:
        if not final_tree.is_dir() or _isa001_tree_sha256(final_tree) != tree_sha256:
            raise
        shutil.rmtree(build_dir, ignore_errors=True)
        return final_tree, True
    return final_tree, False


def _repair_neutral_collision_schema(
    *, requirement: str, asset_path: Path, output_dir: Path
) -> _RepairResult:
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=f"OpenUSD Python APIs are unavailable: {exc}",
        )

    stage, _opened_path, open_error = _open_stage(asset_path, Usd)
    if stage is None:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=open_error or f"Unable to open staged USD: {asset_path}",
        )

    changes: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for prim in list(stage.Traverse()):
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        has_mesh_collision = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        if not has_collision and not has_mesh_collision:
            continue
        is_gprim = bool(UsdGeom.Gprim(prim))
        is_mesh = prim.IsA(UsdGeom.Mesh)
        if has_mesh_collision and is_mesh and not has_collision:
            UsdPhysics.CollisionAPI.Apply(prim)
            changes.append(
                {
                    "source_prim": str(prim.GetPath()),
                    "source_type": prim.GetTypeName(),
                    "target_meshes": [str(prim.GetPath())],
                    "moved_collision_api": False,
                    "moved_mesh_collision_api": False,
                    "applied_missing_collision_api": True,
                }
            )
            continue
        invalid_collision = has_collision and not is_gprim
        invalid_mesh_collision = has_mesh_collision and not is_mesh
        if not invalid_collision and not invalid_mesh_collision:
            continue

        descendant_gprims = [
            child
            for child in Usd.PrimRange(prim)
            if child != prim and UsdGeom.Gprim(child)
        ]
        candidate_prims = _collider_migration_candidates(
            descendant_gprims=descendant_gprims,
            requires_mesh=has_mesh_collision,
            UsdGeom=UsdGeom,
        )
        if not candidate_prims:
            target_type = "Mesh" if has_mesh_collision else "Gprim"
            blocked.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "type_name": prim.GetTypeName(),
                    "reason": (
                        f"Invalid collider API owner has no descendant {target_type}."
                    ),
                }
            )
            continue

        target_prims = _collider_migration_targets(prim, candidate_prims)
        if not target_prims:
            blocked.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "type_name": prim.GetTypeName(),
                    "descendant_candidates": [
                        str(candidate.GetPath()) for candidate in candidate_prims
                    ],
                    "reason": (
                        "Invalid collider API owner has no unambiguous "
                        "collider-identified descendant target."
                    ),
                }
            )
            continue

        approximation = None
        approximation_authored = False
        collision_enabled_authored = False
        collision_enabled = None
        if has_collision:
            collision_api = UsdPhysics.CollisionAPI(prim)
            enabled_attr = collision_api.GetCollisionEnabledAttr()
            if enabled_attr.HasAuthoredValueOpinion():
                collision_enabled_authored = True
                collision_enabled = enabled_attr.Get()
        if has_mesh_collision:
            approximation_attr = UsdPhysics.MeshCollisionAPI(
                prim
            ).GetApproximationAttr()
            if approximation_attr.HasAuthoredValueOpinion():
                approximation_authored = True
                approximation = approximation_attr.Get()

        physics_material_targets = []
        physics_binding = prim.GetRelationship("material:binding:physics")
        if physics_binding:
            physics_material_targets = list(physics_binding.GetTargets())
        binding_conflicts = []
        if physics_material_targets:
            source_binding_path = (
                str(physics_binding.GetPath()) if physics_binding else None
            )
            source_target_paths = {str(target) for target in physics_material_targets}
            for target_prim in target_prims:
                computed_material, computed_relationship = (
                    _computed_physics_material_binding(target_prim, UsdShade)
                )
                computed_binding_path = (
                    str(computed_relationship.GetPath())
                    if computed_relationship is not None
                    else None
                )
                if (
                    computed_material
                    and computed_binding_path != source_binding_path
                    and str(computed_material.GetPath()) not in source_target_paths
                ):
                    binding_conflicts.append(
                        {
                            "target_path": str(target_prim.GetPath()),
                            "source_targets": [
                                str(target) for target in physics_material_targets
                            ],
                            "target_material": str(computed_material.GetPath()),
                            "target_binding": computed_binding_path,
                        }
                    )
        if binding_conflicts:
            blocked.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "type_name": prim.GetTypeName(),
                    "target_prims": [
                        str(target_prim.GetPath()) for target_prim in target_prims
                    ],
                    "reason": (
                        "Invalid collider owner and selected descendant target "
                        "compute conflicting physics material bindings."
                    ),
                    "binding_conflicts": binding_conflicts,
                }
            )
            continue
        for target_prim in target_prims:
            if has_collision or has_mesh_collision:
                collision_api = UsdPhysics.CollisionAPI.Apply(target_prim)
                enabled_attr = collision_api.GetCollisionEnabledAttr()
                if (
                    collision_enabled_authored
                    and not enabled_attr.HasAuthoredValueOpinion()
                ):
                    collision_api.CreateCollisionEnabledAttr().Set(collision_enabled)
            if has_mesh_collision:
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(target_prim)
                approximation_attr = mesh_collision.GetApproximationAttr()
                if (
                    approximation_authored
                    and not approximation_attr.HasAuthoredValueOpinion()
                ):
                    mesh_collision.CreateApproximationAttr().Set(approximation)
            if physics_material_targets:
                computed_material, computed_relationship = (
                    _computed_physics_material_binding(target_prim, UsdShade)
                )
                computed_binding_path = (
                    str(computed_relationship.GetPath())
                    if computed_relationship is not None
                    else None
                )
                should_author_binding = computed_material is None or (
                    physics_binding is not None
                    and computed_binding_path == str(physics_binding.GetPath())
                )
                if should_author_binding:
                    target_prim.CreateRelationship(
                        "material:binding:physics"
                    ).SetTargets(physics_material_targets)
        if has_collision:
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            prim.RemoveProperty("physics:collisionEnabled")
        if has_mesh_collision:
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            prim.RemoveProperty("physics:approximation")
        if physics_material_targets:
            prim.RemoveProperty("material:binding:physics")
        changes.append(
            {
                "source_prim": str(prim.GetPath()),
                "source_type": prim.GetTypeName(),
                "target_prims": [
                    str(target_prim.GetPath()) for target_prim in target_prims
                ],
                "moved_collision_api": has_collision,
                "moved_mesh_collision_api": has_mesh_collision,
                "moved_collision_enabled": collision_enabled_authored,
                "moved_physics_material_binding": bool(physics_material_targets),
            }
        )

    remaining = _neutral_collision_schema_findings(stage)
    report = {
        "schema_version": "content-agent-workflows.simready-repair.v1",
        "requirement": requirement,
        "asset_path": str(asset_path),
        "changes": changes,
        "blocked": blocked,
        "remaining_findings": remaining,
    }
    if blocked or remaining:
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="BLOCKED",
            passed=False,
            reason=(
                "Could not safely move every neutral collider schema API to a "
                "valid mesh/Gprim owner."
            ),
            report=report,
        )
    if changes:
        save_error = _save_stage_root_layer(stage)
        if save_error:
            report["save_error"] = save_error
            return _write_repair_result(
                requirement=requirement,
                asset_path=asset_path,
                output_dir=output_dir,
                status="BLOCKED",
                passed=False,
                reason=save_error,
                report=report,
            )
    return _write_repair_result(
        requirement=requirement,
        asset_path=asset_path,
        output_dir=output_dir,
        status="REPAIRED",
        passed=True,
        reason=(
            f"Moved neutral collision schema APIs from {len(changes)} invalid "
            "owner prim(s) to descendant Gprim/Mesh prim(s)."
            if changes
            else "Neutral collision schema placement already satisfies local checks."
        ),
        report=report,
    )


def _collider_migration_candidates(
    *, descendant_gprims: list[Any], requires_mesh: bool, UsdGeom: Any
) -> list[Any]:
    if requires_mesh:
        return [prim for prim in descendant_gprims if prim.IsA(UsdGeom.Mesh)]
    analytic_gprims = [prim for prim in descendant_gprims if not prim.IsA(UsdGeom.Mesh)]
    return analytic_gprims or list(descendant_gprims)


def _collider_migration_targets(source_prim: Any, candidates: list[Any]) -> list[Any]:
    source_path = str(source_prim.GetPath())
    explicit = [
        candidate
        for candidate in candidates
        if _path_has_collider_token(
            _relative_prim_path(path=str(candidate.GetPath()), prefix=source_path)
        )
        or _path_has_collider_token(candidate.GetName())
    ]
    if explicit:
        return explicit
    if _path_has_collider_token(source_prim.GetName()):
        return list(candidates)
    source_segments = [segment for segment in source_path.split("/") if segment]
    parent_is_collider_scope = len(source_segments) > 1 and _path_has_collider_token(
        source_segments[-2]
    )
    display_name = source_prim.GetMetadata("displayName")
    if (
        parent_is_collider_scope
        and isinstance(display_name, str)
        and _path_has_collider_token(display_name)
    ):
        return list(candidates)
    return []


def _relative_prim_path(*, path: str, prefix: str) -> str:
    prefix = prefix.rstrip("/")
    if path.startswith(prefix + "/"):
        return path[len(prefix) + 1 :]
    return path


def _path_has_collider_token(value: str) -> bool:
    normalized = value.lower()
    return "collider" in normalized or "collision" in normalized


def _neutral_collision_schema_findings(stage: Any) -> list[dict[str, Any]]:
    from pxr import UsdGeom, UsdPhysics

    findings: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not UsdGeom.Gprim(prim):
            findings.append(
                {
                    "requirement": "RB.COL.001",
                    "prim_path": str(prim.GetPath()),
                    "type_name": prim.GetTypeName(),
                    "reason": "CollisionAPI is applied to a non-Gprim prim.",
                }
            )
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            if not prim.IsA(UsdGeom.Mesh):
                findings.append(
                    {
                        "requirement": "RB.COL.002",
                        "prim_path": str(prim.GetPath()),
                        "type_name": prim.GetTypeName(),
                        "reason": "MeshCollisionAPI is applied to a non-Mesh prim.",
                    }
                )
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                findings.append(
                    {
                        "requirement": "RB.COL.002",
                        "prim_path": str(prim.GetPath()),
                        "type_name": prim.GetTypeName(),
                        "reason": "MeshCollisionAPI is missing paired CollisionAPI.",
                    }
                )
    return findings


def _repair_missing_visual_material_bindings(
    *, requirement: str, asset_path: Path, output_dir: Path
) -> _RepairResult:
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=f"OpenUSD Python APIs are unavailable: {exc}",
        )

    stage, _opened_path, open_error = _open_stage(asset_path, Usd)
    if stage is None:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=open_error or f"Unable to open staged USD: {asset_path}",
        )

    unbound_targets = _unbound_visual_material_targets(stage, UsdGeom, UsdShade)
    if not unbound_targets:
        report: dict[str, Any] = {
            "schema_version": "content-agent-workflows.simready-repair.v1",
            "requirement": requirement,
            "asset_path": str(asset_path),
            "material_path": None,
            "bound_gprims": [],
            "remaining_unbound_gprims": [],
        }
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="REPAIRED",
            passed=True,
            reason=(
                "All renderable Gprims and material-bind subsets already compute "
                "visual material bindings."
            ),
            report=report,
        )

    materials = _visual_materials(stage, UsdPhysics, UsdShade)
    if not materials:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=(
                "Cannot repair visual material bindings because no sourced "
                "visual UsdShade Material with a surface output exists."
            ),
        )
    if len(materials) > 1:
        report = {
            "schema_version": "content-agent-workflows.simready-repair.v1",
            "requirement": requirement,
            "asset_path": str(asset_path),
            "candidate_visual_materials": [
                str(material.GetPath()) for material in materials
            ],
            "unbound_gprims": [str(prim.GetPath()) for prim in unbound_targets],
            "remaining_unbound_gprims": [
                str(prim.GetPath()) for prim in unbound_targets
            ],
        }
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="BLOCKED",
            passed=False,
            reason=(
                "Cannot repair visual material bindings because multiple sourced "
                "visual UsdShade Materials exist and the intended assignment is "
                "ambiguous."
            ),
            report=report,
        )

    material = materials[0]
    bound_paths: list[str] = []
    for prim in unbound_targets:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        bound_paths.append(str(prim.GetPath()))

    save_error = None
    if bound_paths:
        save_error = _save_stage_root_layer(stage)

    remaining_unbound = [
        str(prim.GetPath())
        for prim in _unbound_visual_material_targets(stage, UsdGeom, UsdShade)
    ]
    report = {
        "schema_version": "content-agent-workflows.simready-repair.v1",
        "requirement": requirement,
        "asset_path": str(asset_path),
        "material_path": str(material.GetPath()),
        "bound_gprims": bound_paths,
        "remaining_unbound_gprims": remaining_unbound,
    }
    if save_error:
        report["save_error"] = save_error
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="BLOCKED",
            passed=False,
            reason=save_error,
            report=report,
        )
    if remaining_unbound:
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="BLOCKED",
            passed=False,
            reason="Some renderable Gprims still do not compute a material binding.",
            report=report,
        )
    return _write_repair_result(
        requirement=requirement,
        asset_path=asset_path,
        output_dir=output_dir,
        status="REPAIRED",
        passed=True,
        reason=(
            f"Bound {len(bound_paths)} renderable Gprim(s) to {material.GetPath()}."
        ),
        report=report,
    )


def _repair_missing_physics_material_bindings(
    *, requirement: str, asset_path: Path, output_dir: Path
) -> _RepairResult:
    try:
        from pxr import Usd, UsdPhysics, UsdShade
    except ImportError as exc:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=f"OpenUSD Python APIs are unavailable: {exc}",
        )

    stage, _opened_path, open_error = _open_stage(asset_path, Usd)
    if stage is None:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=open_error or f"Unable to open staged USD: {asset_path}",
        )

    colliders = [
        prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not colliders:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason="Cannot bind physics materials because no CollisionAPI prims exist.",
        )

    initial_findings = _physics_material_binding_findings(stage, UsdPhysics, UsdShade)
    if not initial_findings:
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="REPAIRED",
            passed=True,
            reason="All CollisionAPI prims already have valid physics materials.",
            report={
                "schema_version": "content-agent-workflows.simready-repair.v1",
                "requirement": requirement,
                "asset_path": str(asset_path),
                "bound_colliders": [],
                "replaced_invalid_bindings": [],
                "remaining_findings": [],
            },
        )

    materials = _physics_materials(stage, UsdPhysics, UsdShade)
    if not materials:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=(
                "Cannot repair physics material bindings because no "
                "UsdShade Material with UsdPhysics.MaterialAPI exists."
            ),
        )
    if len(materials) > 1:
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="BLOCKED",
            passed=False,
            reason=(
                "Cannot repair physics material bindings because multiple "
                "UsdPhysics.MaterialAPI materials exist and material identity is "
                "ambiguous."
            ),
            report={
                "schema_version": "content-agent-workflows.simready-repair.v1",
                "requirement": requirement,
                "asset_path": str(asset_path),
                "candidate_physics_materials": [
                    str(material.GetPath()) for material in materials
                ],
                "initial_findings": initial_findings,
            },
        )

    material = materials[0]

    bound_paths: list[str] = []
    replaced_paths: list[str] = []
    for prim in colliders:
        relationship = prim.GetRelationship("material:binding:physics")
        targets = relationship.GetTargets() if relationship else []
        if (
            len(targets) == 1
            and all(
                _is_physics_material_target(stage, target, UsdPhysics, UsdShade)
                for target in targets
            )
        ) or (
            not targets
            and _is_computed_physics_material_binding_valid(
                prim, UsdPhysics=UsdPhysics, UsdShade=UsdShade
            )
        ):
            continue
        prim.CreateRelationship("material:binding:physics").SetTargets(
            [material.GetPath()]
        )
        if targets:
            replaced_paths.append(str(prim.GetPath()))
        else:
            bound_paths.append(str(prim.GetPath()))

    save_error = None
    if bound_paths or replaced_paths:
        save_error = _save_stage_root_layer(stage)

    remaining_findings = _physics_material_binding_findings(stage, UsdPhysics, UsdShade)
    report = {
        "schema_version": "content-agent-workflows.simready-repair.v1",
        "requirement": requirement,
        "asset_path": str(asset_path),
        "physics_material_path": str(material.GetPath()),
        "bound_colliders": bound_paths,
        "replaced_invalid_bindings": replaced_paths,
        "remaining_findings": remaining_findings,
    }
    if save_error:
        report["save_error"] = save_error
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="BLOCKED",
            passed=False,
            reason=save_error,
            report=report,
        )
    if remaining_findings:
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="BLOCKED",
            passed=False,
            reason=(
                "Some CollisionAPI prims still lack a valid "
                "material:binding:physics target."
            ),
            report=report,
        )
    return _write_repair_result(
        requirement=requirement,
        asset_path=asset_path,
        output_dir=output_dir,
        status="REPAIRED",
        passed=True,
        reason=(
            f"Bound {len(bound_paths)} collider prim(s) to physics material "
            f"{material.GetPath()}."
        ),
        report=report,
    )


def _physics_materials(stage: Any, UsdPhysics: Any, UsdShade: Any) -> list[Any]:
    return [
        UsdShade.Material(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdShade.Material) and prim.HasAPI(UsdPhysics.MaterialAPI)
    ]


def _visual_materials(stage: Any, UsdPhysics: Any, UsdShade: Any) -> list[Any]:
    materials: list[Any] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        material = UsdShade.Material(prim)
        has_surface_source = _has_surface_source(material)
        if prim.HasAPI(UsdPhysics.MaterialAPI) and not has_surface_source:
            continue
        if has_surface_source:
            materials.append(material)
    return materials


def _has_surface_source(material: Any) -> bool:
    for render_context in ("", "mdl", "mtlx"):
        try:
            if render_context:
                shader, _source_name, _source_type = material.ComputeSurfaceSource(
                    render_context
                )
            else:
                shader, _source_name, _source_type = material.ComputeSurfaceSource()
        except TypeError:
            continue
        if shader and shader.GetPrim().IsValid():
            return True
    return False


def _renderable_gprims(stage: Any, UsdGeom: Any) -> list[Any]:
    result: list[Any] = []
    for prim in stage.Traverse():
        if not UsdGeom.Gprim(prim):
            continue
        imageable = UsdGeom.Imageable(prim)
        purpose = imageable.ComputePurpose() or "default"
        if purpose not in {"default", "render"}:
            continue
        result.append(prim)
    return result


def _unbound_visual_material_targets(
    stage: Any, UsdGeom: Any, UsdShade: Any
) -> list[Any]:
    result: list[Any] = []
    for prim in _renderable_gprims(stage, UsdGeom):
        if not _has_computed_material_binding(prim, UsdShade):
            result.append(prim)
        for subset_prim in _material_bind_subset_prims(prim, UsdShade):
            if not _has_computed_material_binding(subset_prim, UsdShade):
                result.append(subset_prim)
    return result


def _material_bind_subset_prims(prim: Any, UsdShade: Any) -> list[Any]:
    return [
        subset.GetPrim()
        for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
    ]


def _has_computed_material_binding(prim: Any, UsdShade: Any) -> bool:
    material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
        materialPurpose=UsdShade.Tokens.full
    )
    return bool(material)


def _is_physics_material_target(
    stage: Any, target: Any, UsdPhysics: Any, UsdShade: Any
) -> bool:
    prim = stage.GetPrimAtPath(target)
    return bool(
        prim
        and prim.IsValid()
        and prim.IsA(UsdShade.Material)
        and prim.HasAPI(UsdPhysics.MaterialAPI)
    )


def _computed_physics_material_binding(
    prim: Any, UsdShade: Any
) -> tuple[Any | None, Any | None]:
    material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
        materialPurpose=PHYSICS_MATERIAL_PURPOSE
    )
    if not material:
        return None, relationship
    return material, relationship


def _is_computed_physics_material_binding_valid(
    prim: Any, *, UsdPhysics: Any, UsdShade: Any
) -> bool:
    material, _relationship = _computed_physics_material_binding(prim, UsdShade)
    return bool(material and material.GetPrim().HasAPI(UsdPhysics.MaterialAPI))


def _physics_material_binding_findings(
    stage: Any, UsdPhysics: Any, UsdShade: Any
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        relationship = prim.GetRelationship("material:binding:physics")
        targets = relationship.GetTargets() if relationship else []
        if not targets:
            if _is_computed_physics_material_binding_valid(
                prim, UsdPhysics=UsdPhysics, UsdShade=UsdShade
            ):
                continue
            findings.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "reason": (
                        "CollisionAPI prim lacks computed material:binding:physics."
                    ),
                }
            )
            continue
        if len(targets) > 1:
            findings.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "targets": [str(target) for target in targets],
                    "reason": (
                        "material:binding:physics must target exactly one "
                        "UsdPhysics material."
                    ),
                }
            )
            continue
        invalid_targets = [
            str(target)
            for target in targets
            if not _is_physics_material_target(stage, target, UsdPhysics, UsdShade)
        ]
        if invalid_targets:
            findings.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "invalid_targets": invalid_targets,
                    "reason": (
                        "material:binding:physics target is missing or is not a "
                        "UsdShade Material with UsdPhysics.MaterialAPI."
                    ),
                }
            )
    return findings


def _repair_simready_metadata(
    *,
    requirement: str,
    asset_path: Path,
    output_dir: Path,
    source_asset: str | None,
) -> _RepairResult:
    try:
        from pxr import Usd
    except ImportError as exc:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=f"OpenUSD Python APIs are unavailable: {exc}",
        )

    stage, _opened_path, open_error = _open_stage(asset_path, Usd)
    if stage is None:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=open_error or f"Unable to open staged USD: {asset_path}",
        )

    root_layer = stage.GetRootLayer()
    layer_data = dict(root_layer.customLayerData or {})
    existing = layer_data.get("SimReady_Metadata")
    if existing is None:
        if asset_path.suffix.lower() == ".usdz":
            return _blocked_repair_result(
                requirement=requirement,
                asset_path=asset_path,
                output_dir=output_dir,
                reason=(
                    "Cannot author SimReady metadata directly into a read-only "
                    "USDZ package layer; extract and republish the package first."
                ),
            )
        metadata = {
            "description": "SimReady conformed asset",
            "identifier": asset_path.stem,
            "version": "1.0.0",
        }
        if source_asset:
            metadata["source_asset"] = source_asset
        layer_data["SimReady_Metadata"] = json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        )
        try:
            root_layer.customLayerData = layer_data
            save_error = _save_stage_root_layer(stage)
        except Exception as exc:  # pragma: no cover - OpenUSD exception types vary
            save_error = f"Could not author SimReady metadata: {exc}"
        if save_error:
            return _blocked_repair_result(
                requirement=requirement,
                asset_path=asset_path,
                output_dir=output_dir,
                reason=save_error,
            )
        reason = "Authored portable SimReady metadata in root-layer custom data."
    else:
        reason = "Root-layer custom data already contains SimReady metadata."

    return _write_repair_result(
        requirement=requirement,
        asset_path=asset_path,
        output_dir=output_dir,
        status="REPAIRED",
        passed=True,
        reason=reason,
        report={
            "schema_version": "content-agent-workflows.simready-repair.v1",
            "requirement": requirement,
            "asset_path": str(asset_path),
            "metadata_key": "SimReady_Metadata",
            "source_asset": source_asset,
        },
    )


def _prepend_xform_op(xformable: Any, xform_op: Any) -> None:
    order_attr = xformable.GetXformOpOrderAttr()
    op_name = xform_op.GetOpName()
    op_order = [token for token in order_attr.Get() or [] if token != op_name]
    insert_at = 1 if op_order and str(op_order[0]) == "!resetXformStack!" else 0
    op_order.insert(insert_at, op_name)
    order_attr.Set(op_order)


def _stage_metric_transform_frontiers(
    stage: Any,
    default_prim: Any,
    *,
    Gf: Any,
    UsdGeom: Any,
) -> list[Any]:
    """Return every transform frontier that must receive metric compensation."""

    if default_prim.IsA(UsdGeom.Boundable):
        raise ValueError(
            "geometry-bearing default prim cannot remain an identity stage-metric "
            f"root: {default_prim.GetPath()}"
        )
    default_xformable = UsdGeom.Xformable(default_prim)
    if not Gf.IsClose(
        default_xformable.GetLocalTransformation(),
        Gf.Matrix4d(1.0),
        1e-9,
    ):
        raise ValueError(
            "default prim local transform must remain identity during stage-metric "
            f"normalization: {default_prim.GetPath()}"
        )

    targets: dict[str, Any] = {}

    def add_first_xformable_descendants(parent: Any) -> None:
        for child in parent.GetChildren():
            if child.IsA(UsdGeom.Xformable):
                targets[str(child.GetPath())] = child
            else:
                add_first_xformable_descendants(child)

    top_level_prims = list(stage.GetPseudoRoot().GetChildren())
    if not top_level_prims:
        raise ValueError("stage has no top-level prims")
    for prim in top_level_prims:
        if prim == default_prim:
            continue
        if prim.IsA(UsdGeom.Xformable):
            targets[str(prim.GetPath())] = prim
        else:
            add_first_xformable_descendants(prim)

    add_first_xformable_descendants(default_prim)
    for prim in stage.TraverseAll():
        if prim == default_prim or not prim.IsA(UsdGeom.Xformable):
            continue
        if UsdGeom.Xformable(prim).GetResetXformStack():
            targets[str(prim.GetPath())] = prim

    if not targets:
        raise ValueError(
            "default prim has no safe descendant stage-metric transform frontier"
        )
    for path, prim in targets.items():
        if prim.IsInstance() or prim.IsInstanceProxy() or prim.IsInstanceable():
            raise ValueError(
                f"instance transform frontier is unsafe to normalize: {path}"
            )
        xformable = UsdGeom.Xformable(prim)
        order = list(xformable.GetXformOpOrderAttr().Get() or [])
        reset_indices = [
            index
            for index, token in enumerate(order)
            if str(token) == "!resetXformStack!"
        ]
        if len(reset_indices) > 1 or (reset_indices and reset_indices[0] != 0):
            raise ValueError(
                f"ambiguous reset xform order at transform frontier: {path}"
            )
        if any(
            op.GetAttr().GetNumTimeSamples() for op in xformable.GetOrderedXformOps()
        ):
            raise ValueError(f"time-sampled transform frontier is unsafe: {path}")
    return [targets[path] for path in sorted(targets)]


def _scale_authored_physics_linear_quantities(
    stage: Any,
    *,
    factor: float,
    Gf: Any,
    UsdPhysics: Any,
) -> list[str]:
    scaled_paths: list[str] = []
    for prim in stage.TraverseAll():
        if prim.IsA(UsdPhysics.Scene):
            gravity_attr = UsdPhysics.Scene(prim).GetGravityMagnitudeAttr()
            if gravity_attr.HasAuthoredValueOpinion():
                if gravity_attr.GetNumTimeSamples():
                    raise ValueError(
                        "Cannot safely normalize time-sampled gravity magnitude: "
                        f"{gravity_attr.GetPath()}"
                    )
                gravity_value = gravity_attr.Get()
                if (
                    isinstance(gravity_value, bool)
                    or not isinstance(gravity_value, int | float)
                    or not math.isfinite(float(gravity_value))
                ):
                    raise ValueError(
                        "Cannot normalize invalid gravity magnitude: "
                        f"{gravity_attr.GetPath()}"
                    )
                gravity_attr.Set(float(gravity_value) * factor)
                scaled_paths.append(str(gravity_attr.GetPath()))
        velocity_attr = prim.GetAttribute("physics:velocity")
        if velocity_attr and velocity_attr.HasAuthoredValueOpinion():
            if velocity_attr.GetNumTimeSamples():
                raise ValueError(
                    "Cannot safely normalize time-sampled rigid-body velocity: "
                    f"{velocity_attr.GetPath()}"
                )
            velocity = [float(item) for item in velocity_attr.Get()]
            if len(velocity) != 3 or not all(math.isfinite(item) for item in velocity):
                raise ValueError(
                    f"Cannot normalize invalid rigid-body velocity: {velocity_attr.GetPath()}"
                )
            velocity_attr.Set(Gf.Vec3f(*(item * factor for item in velocity)))
            scaled_paths.append(str(velocity_attr.GetPath()))
        if not prim.IsA(UsdPhysics.Joint):
            continue
        is_prismatic = prim.IsA(UsdPhysics.PrismaticJoint)
        is_distance = prim.IsA(UsdPhysics.DistanceJoint)
        for attr in prim.GetAttributes():
            name = attr.GetName()
            affected = is_prismatic and (
                name in {"physics:lowerLimit", "physics:upperLimit"}
                or name.startswith("drive:linear:physics:target")
                or name == "drive:linear:physics:maxForce"
                or name.startswith("state:linear:physics:")
            )
            affected = affected or (
                is_distance and name in {"physics:minDistance", "physics:maxDistance"}
            )
            if not affected or not attr.HasAuthoredValueOpinion():
                continue
            if attr.GetNumTimeSamples():
                raise ValueError(
                    f"Cannot safely normalize time-sampled physics distance: {attr.GetPath()}"
                )
            value = attr.Get()
            if isinstance(value, int | float):
                values = [float(value)]
                scaled_value: Any = values[0] * factor
            else:
                values = [float(item) for item in value]
                scaled_values = [item * factor for item in values]
                value_type = type(value)
                try:
                    scaled_value = value_type(*scaled_values)
                except TypeError:
                    scaled_value = (
                        Gf.Vec3f(*scaled_values)
                        if attr.GetTypeName().type.typeName == "GfVec3f"
                        else Gf.Vec3d(*scaled_values)
                    )
            if not all(math.isfinite(item) for item in values):
                raise ValueError(
                    f"Cannot normalize non-finite physics distance: {attr.GetPath()}"
                )
            attr.Set(scaled_value)
            scaled_paths.append(str(attr.GetPath()))
    return scaled_paths


def _unsupported_authored_physics_unit_paths(stage: Any) -> list[str]:
    """Find physics values whose unit conversion is not implemented safely."""

    unsupported_names = {
        "physics:centerOfMass",
        "physics:density",
        "physics:diagonalInertia",
    }
    unsupported_suffixes = (
        ":breakForce",
        ":breakTorque",
        ":contactOffset",
        ":restOffset",
    )
    paths: list[str] = []
    for prim in stage.TraverseAll():
        for attr in prim.GetAttributes():
            name = attr.GetName()
            unsupported = name in unsupported_names or name.endswith(
                unsupported_suffixes
            )
            unsupported = unsupported or (
                name.startswith("drive:angular:") and name.endswith(":maxForce")
            )
            if unsupported and attr.HasAuthoredValueOpinion():
                paths.append(str(attr.GetPath()))
    return sorted(paths)


def _rotate_authored_gravity_directions(
    stage: Any,
    *,
    rotation: Any,
    Gf: Any,
    UsdPhysics: Any,
) -> list[str]:
    rotated_paths: list[str] = []
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdPhysics.Scene):
            continue
        attr = UsdPhysics.Scene(prim).GetGravityDirectionAttr()
        if not attr.HasAuthoredValueOpinion():
            continue
        if attr.GetNumTimeSamples():
            raise ValueError(
                f"Cannot safely normalize time-sampled gravity: {attr.GetPath()}"
            )
        value = attr.Get()
        values = [float(item) for item in value]
        if len(values) != 3 or not all(math.isfinite(item) for item in values):
            raise ValueError(f"Cannot normalize invalid gravity: {attr.GetPath()}")
        rotated = rotation.TransformDir(Gf.Vec3d(*values))
        attr.Set(Gf.Vec3f(*(float(item) for item in rotated)))
        rotated_paths.append(str(attr.GetPath()))
    return rotated_paths


def _repair_stage_metrics(
    *,
    requirement: str,
    asset_path: Path,
    output_dir: Path,
    source_asset: str | None,
) -> _RepairResult:
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=f"OpenUSD Python APIs are unavailable: {exc}",
        )

    stage, _opened_path, open_error = _open_stage(asset_path, Usd)
    if stage is None:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=open_error or f"Unable to open staged USD: {asset_path}",
        )
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason="Cannot normalize stage metrics without a default prim.",
        )
    xformable = UsdGeom.Xformable(default_prim)
    if not xformable:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason="Cannot normalize stage metrics because the default prim is not xformable.",
        )

    repair_report: dict[str, Any] = {
        "schema_version": "content-agent-workflows.simready-repair.v1",
        "requirement": requirement,
        "asset_path": str(asset_path),
        "default_prim": str(default_prim.GetPath()),
    }
    try:
        if requirement == "UN.006":
            source_up_axis = UsdGeom.GetStageUpAxis(stage)
            repair_report["source_up_axis"] = str(source_up_axis)
            if source_up_axis == UsdGeom.Tokens.z:
                reason = "Stage is already Z-up."
            elif source_up_axis == UsdGeom.Tokens.y:
                up_axis_rotation = Gf.Rotation(
                    Gf.Vec3d(1.0, 0.0, 0.0),
                    90.0,
                )
                rotation = Gf.Matrix4d(1.0)
                rotation.SetRotate(up_axis_rotation)
                transform_frontiers = _stage_metric_transform_frontiers(
                    stage,
                    default_prim,
                    Gf=Gf,
                    UsdGeom=UsdGeom,
                )
                for frontier in transform_frontiers:
                    frontier_xformable = UsdGeom.Xformable(frontier)
                    up_axis_op = frontier_xformable.AddTransformOp(
                        UsdGeom.XformOp.PrecisionDouble,
                        "simreadyUpAxis",
                    )
                    up_axis_op.Set(rotation)
                    _prepend_xform_op(frontier_xformable, up_axis_op)
                repair_report["transform_frontiers"] = [
                    str(prim.GetPath()) for prim in transform_frontiers
                ]
                repair_report["rotated_gravity_directions"] = (
                    _rotate_authored_gravity_directions(
                        stage,
                        rotation=up_axis_rotation,
                        Gf=Gf,
                        UsdPhysics=UsdPhysics,
                    )
                )
                UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
                reason = "Rotated Y-up content into Z-up coordinates."
            else:
                return _blocked_repair_result(
                    requirement=requirement,
                    asset_path=asset_path,
                    output_dir=output_dir,
                    reason=f"Unsupported source up axis: {source_up_axis}",
                )
            repair_report["target_up_axis"] = "Z"
        else:
            if not stage.HasAuthoredMetadata("metersPerUnit"):
                return _blocked_repair_result(
                    requirement=requirement,
                    asset_path=asset_path,
                    output_dir=output_dir,
                    reason=(
                        "Cannot normalize stage units without authored "
                        "metersPerUnit or explicit owner-approved source-unit "
                        "evidence; the OpenUSD fallback is not source evidence."
                    ),
                )
            source_meters_per_unit = float(stage.GetMetadata("metersPerUnit"))
            repair_report["source_meters_per_unit"] = source_meters_per_unit
            source_suffix = (
                Path(source_asset).suffix.lower() if source_asset is not None else ""
            )
            gltf_meter_native = source_suffix in {".glb", ".gltf"}
            repair_report["source_asset"] = source_asset
            if (
                not math.isfinite(source_meters_per_unit)
                or source_meters_per_unit <= 0.0
            ):
                return _blocked_repair_result(
                    requirement=requirement,
                    asset_path=asset_path,
                    output_dir=output_dir,
                    reason=(
                        "Cannot normalize a stage with non-finite or non-positive "
                        f"metersPerUnit={source_meters_per_unit}."
                    ),
                )
            if gltf_meter_native:
                UsdGeom.SetStageMetersPerUnit(stage, 1.0)
                reason = (
                    "Corrected converter-authored stage units for meter-native glTF "
                    "content without rescaling geometry."
                )
                repair_report["source_format_units"] = "meters"
            elif source_meters_per_unit == 1.0:
                reason = "Stage is already meter-native."
            else:
                transform_frontiers = _stage_metric_transform_frontiers(
                    stage,
                    default_prim,
                    Gf=Gf,
                    UsdGeom=UsdGeom,
                )
                unsupported_physics_paths = _unsupported_authored_physics_unit_paths(
                    stage
                )
                if unsupported_physics_paths:
                    return _blocked_repair_result(
                        requirement=requirement,
                        asset_path=asset_path,
                        output_dir=output_dir,
                        reason=(
                            "Cannot safely normalize stage units while unsupported "
                            "authored physics quantities are present: "
                            f"{unsupported_physics_paths}"
                        ),
                    )
                for frontier in transform_frontiers:
                    frontier_xformable = UsdGeom.Xformable(frontier)
                    meters_per_unit_op = frontier_xformable.AddScaleOp(
                        UsdGeom.XformOp.PrecisionDouble,
                        "simreadyMetersPerUnit",
                    )
                    meters_per_unit_op.Set(
                        Gf.Vec3d(
                            source_meters_per_unit,
                            source_meters_per_unit,
                            source_meters_per_unit,
                        )
                    )
                    _prepend_xform_op(frontier_xformable, meters_per_unit_op)
                repair_report["transform_frontiers"] = [
                    str(prim.GetPath()) for prim in transform_frontiers
                ]
                repair_report["scaled_physics_linear_quantities"] = (
                    _scale_authored_physics_linear_quantities(
                        stage,
                        factor=source_meters_per_unit,
                        Gf=Gf,
                        UsdPhysics=UsdPhysics,
                    )
                )
                UsdGeom.SetStageMetersPerUnit(stage, 1.0)
                reason = "Baked the source linear scale into all transform frontiers."
            repair_report["target_meters_per_unit"] = 1.0
        save_error = _save_stage_root_layer(stage)
        if save_error:
            return _blocked_repair_result(
                requirement=requirement,
                asset_path=asset_path,
                output_dir=output_dir,
                reason=f"Could not normalize stage metrics: {save_error}",
            )
    except Exception as exc:  # pragma: no cover - OpenUSD exception types vary
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=f"Could not normalize stage metrics: {exc}",
        )

    return _write_repair_result(
        requirement=requirement,
        asset_path=asset_path,
        output_dir=output_dir,
        status="REPAIRED",
        passed=True,
        reason=reason,
        report=repair_report,
    )


def _repair_missing_grasp_line(
    *,
    requirement: str,
    asset_path: Path,
    output_dir: Path,
    grasp_plan_path: Path | None,
    source_asset_path: Path | None,
    source_lineage: _GSP001SourceLineage | None,
    grasp_prim_path: str | None = None,
) -> _RepairResult:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt
    except ImportError as exc:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=f"OpenUSD Python APIs are unavailable: {exc}",
        )

    stage, _opened_path, open_error = _open_stage(asset_path, Usd)
    if stage is None:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=open_error or f"Unable to open staged USD: {asset_path}",
        )
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason="Cannot author grasp line because the stage has no default prim.",
        )

    existing = _grasp_identifier_prims(default_prim, Usd)
    valid_existing = [
        prim for prim in existing if _is_valid_grasp_identifier(prim, UsdGeom)
    ]
    invalid_paths = sorted(
        set(str(prim.GetPath()) for prim in existing)
        - set(str(prim.GetPath()) for prim in valid_existing)
    )
    if invalid_paths:
        reason = (
            "Cannot replace or add grasp geometry while conflicting existing "
            "grasp_identifier prims are present."
        )
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="BLOCKED",
            passed=False,
            reason=reason,
            report={
                "schema_version": GSP001_RECEIPT_SCHEMA_VERSION,
                "requirement": requirement,
                "asset_path": str(asset_path),
                "invalid_existing_grasp_lines": invalid_paths,
                "changes": [],
            },
        )

    if existing and grasp_plan_path is None:
        try:
            identity_path = _gsp001_identity_path(source_asset_path or asset_path)
        except ValueError as exc:
            return _blocked_repair_result(
                requirement=requirement,
                asset_path=asset_path,
                output_dir=output_dir,
                reason=f"Cannot establish the existing grasp-line asset identity: {exc}",
            )
        try:
            source_sha256 = _file_sha256(identity_path)
        except OSError as exc:
            return _blocked_repair_result(
                requirement=requirement,
                asset_path=asset_path,
                output_dir=output_dir,
                reason=f"Cannot hash the existing grasp-line asset: {exc}",
            )
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="REPAIRED",
            passed=True,
            reason="A valid grasp_identifier BasisCurves prim already exists.",
            report={
                "schema_version": GSP001_RECEIPT_SCHEMA_VERSION,
                "requirement": requirement,
                "asset_path": str(asset_path),
                "source_asset_path": str(identity_path),
                "source_asset_sha256": source_sha256,
                "existing_grasp_lines": [str(prim.GetPath()) for prim in existing],
                "changes": [],
                "readback_verified": True,
                "unrelated_stage_preserved": True,
                "source_asset_preserved": True,
            },
        )

    if grasp_plan_path is None and grasp_prim_path is not None:
        if asset_path.suffix.lower() == ".usdz":
            return _blocked_repair_result(
                requirement=requirement,
                asset_path=asset_path,
                output_dir=output_dir,
                reason=(
                    "Cannot author an explicit grasp repair directly into a "
                    "read-only USDZ package layer; extract and republish the "
                    "package first."
                ),
            )
        return _repair_missing_grasp_line_from_prim(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            stage=stage,
            default_prim=default_prim,
            grasp_prim_path=grasp_prim_path,
            Gf=Gf,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdShade=UsdShade,
        )

    if grasp_plan_path is None:
        reason = (
            "Cannot author grasp_identifier geometry without an explicit "
            "owner-approved local-coordinate grasp plan."
        )
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="BLOCKED",
            passed=False,
            reason=reason,
            report={
                "schema_version": GSP001_RECEIPT_SCHEMA_VERSION,
                "requirement": requirement,
                "asset_path": str(asset_path),
                "grasp_plan_path": None,
                "changes": [],
            },
        )

    source_tree: Path | None = None
    source_root: Path | None = None
    bound_source_identity_path: Path | None = None
    staged_identity_path: Path | None = None
    extraction_dir: Path | None = None
    build_dir: Path | None = None
    report: dict[str, Any] = {
        "schema_version": GSP001_RECEIPT_SCHEMA_VERSION,
        "requirement": requirement,
        "asset_path": str(asset_path),
        "grasp_plan_path": str(grasp_plan_path),
        "existing_grasp_lines": [str(prim.GetPath()) for prim in valid_existing],
        "changes": [],
    }
    try:
        plan, plan_sha256 = _load_grasp_plan(grasp_plan_path)
        report["grasp_plan_sha256"] = plan_sha256
        report["grasp_plan_schema_version"] = plan.schema_version

        bound_source_identity_path = _gsp001_identity_path(
            source_asset_path or asset_path
        )
        source_tree, source_root, staged_identity_path, extraction_dir = (
            _gsp001_source_package(asset_path=asset_path, output_dir=output_dir)
        )
        source_asset_sha256 = _file_sha256(bound_source_identity_path)
        staged_asset_sha256 = _file_sha256(staged_identity_path)
        source_lineage_report: dict[str, Any] | None = None
        report.update(
            {
                "source_asset_path": str(bound_source_identity_path),
                "source_asset_sha256": source_asset_sha256,
                "staged_asset_path": str(staged_identity_path),
                "staged_asset_sha256": staged_asset_sha256,
                "source_root": str(source_root),
                "source_was_usdz": extraction_dir is not None,
            }
        )
        if plan.source_asset_sha256 != source_asset_sha256:
            raise ValueError(
                "Grasp plan source_asset_sha256 is stale: expected "
                f"{source_asset_sha256}, received {plan.source_asset_sha256}."
            )
        if staged_asset_sha256 != source_asset_sha256:
            source_lineage_report = _validate_gsp001_hygiene_lineage(
                source_lineage=source_lineage,
                source_asset_sha256=source_asset_sha256,
                staged_identity_path=staged_identity_path,
                staged_asset_sha256=staged_asset_sha256,
            )
            report["source_lineage"] = source_lineage_report

        source_stage = Usd.Stage.Open(str(source_root), load=Usd.Stage.LoadAll)
        if source_stage is None:
            raise ValueError(f"Unable to open GSP.001 source USD: {source_root}")
        _validate_isa001_dependency_closure(
            stage=source_stage,
            source_root=source_root,
            source_tree=source_tree,
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        _validate_gsp001_plan(stage=source_stage, plan=plan, Sdf=Sdf)
        source_tree_sha256 = _isa001_tree_sha256(source_tree)
        source_stage_sha256 = _gsp001_stage_fingerprint(
            stage=source_stage,
            package_root=source_tree,
            excluded_prim_paths=(),
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        source_relative_root = source_root.relative_to(source_tree)
        report.update(
            {
                "default_prim_path": plan.default_prim_path,
                "source_tree_sha256": source_tree_sha256,
                "source_stage_sha256": source_stage_sha256,
                "provenance": plan.provenance.model_dump(mode="json"),
            }
        )
        source_stage = None
        gc.collect()

        publish_root = output_dir / GSP001_OUTPUT_DIR
        publish_root.mkdir(parents=True, exist_ok=True)
        build_dir = _private_mkdtemp(prefix=".gsp001-build-", directory=publish_root)
        shutil.copytree(source_tree, build_dir, dirs_exist_ok=True)
        build_root = build_dir / source_relative_root

        build_stage = Usd.Stage.Open(str(build_root), load=Usd.Stage.LoadAll)
        if build_stage is None:
            raise ValueError(f"Unable to open GSP.001 build USD: {build_root}")
        _validate_gsp001_plan(stage=build_stage, plan=plan, Sdf=Sdf)
        authored = _author_gsp001_lines(
            stage=build_stage,
            lines=plan.grasp_lines,
            Gf=Gf,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            Vt=Vt,
        )
        save_error = _save_stage_root_layer(build_stage)
        if save_error:
            raise OSError(save_error)
        build_stage = None
        gc.collect()

        output_stage = Usd.Stage.Open(str(build_root), load=Usd.Stage.LoadAll)
        if output_stage is None:
            raise ValueError(f"Unable to open authored GSP.001 USD: {build_root}")
        _verify_gsp001_readback(
            stage=output_stage,
            lines=plan.grasp_lines,
            Gf=Gf,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            Vt=Vt,
        )
        output_unrelated_stage_sha256 = _gsp001_stage_fingerprint(
            stage=output_stage,
            package_root=build_dir,
            excluded_prim_paths=tuple(line.prim_path for line in plan.grasp_lines),
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        if output_unrelated_stage_sha256 != source_stage_sha256:
            raise ValueError(
                "Authored GSP.001 output contains an unplanned stage change."
            )
        output_stage = None
        gc.collect()

        if _isa001_tree_sha256(source_tree) != source_tree_sha256:
            raise ValueError(
                "Staged USD package changed while GSP.001 output was being built."
            )
        if _file_sha256(bound_source_identity_path) != source_asset_sha256:
            raise ValueError(
                "Source asset changed while GSP.001 output was being built."
            )
        if _file_sha256(staged_identity_path) != staged_asset_sha256:
            raise ValueError(
                "Staged asset changed while GSP.001 output was being built."
            )
        if _file_sha256(grasp_plan_path) != plan_sha256:
            raise ValueError("Grasp plan changed while GSP.001 output was being built.")

        output_asset_sha256 = _file_sha256(build_root)
        output_tree_sha256 = _isa001_tree_sha256(build_dir)
        final_tree, publication_outcome = _publish_gsp001_tree(
            build_dir=build_dir,
            publish_root=publish_root,
            tree_sha256=output_tree_sha256,
        )
        build_dir = None
        final_root = final_tree / source_relative_root
        report.update(
            {
                "changes": authored,
                "output_root": str(final_root),
                "output_asset_sha256": output_asset_sha256,
                "output_tree_sha256": output_tree_sha256,
                "output_unrelated_stage_sha256": output_unrelated_stage_sha256,
                "authored_grasp_lines": [
                    line.model_dump(mode="json") for line in plan.grasp_lines
                ],
                "readback_verified": True,
                "unrelated_stage_preserved": True,
                "source_asset_preserved": True,
                "reused_output": publication_outcome != "published",
                "publication_outcome": publication_outcome,
            }
        )
        return _write_repair_result(
            requirement=requirement,
            asset_path=final_root,
            package_root=final_tree,
            output_dir=output_dir,
            status="REPAIRED",
            passed=True,
            reason="Published evidence-backed GSP.001 grasp lines atomically.",
            report=report,
        )
    except (
        OSError,
        RuntimeError,
        UnicodeError,
        ValidationError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        report["changes"] = []
        return _write_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            status="BLOCKED",
            passed=False,
            reason=f"Could not safely author GSP.001 grasp lines: {exc}",
            report=report,
        )
    finally:
        if build_dir is not None:
            shutil.rmtree(build_dir, ignore_errors=True)
        if extraction_dir is not None:
            shutil.rmtree(extraction_dir, ignore_errors=True)


def _repair_missing_grasp_line_from_prim(
    *,
    requirement: str,
    asset_path: Path,
    output_dir: Path,
    stage: Any,
    default_prim: Any,
    grasp_prim_path: str,
    Gf: Any,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    UsdShade: Any,
) -> _RepairResult:
    grasp_target = stage.GetPrimAtPath(grasp_prim_path)
    if not grasp_target:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=f"Explicit grasp evidence prim does not exist: {grasp_prim_path}",
        )
    bounds = (
        UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        .ComputeWorldBound(grasp_target)
        .ComputeAlignedRange()
    )
    if bounds.IsEmpty():
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=f"Explicit grasp evidence prim has empty bounds: {grasp_prim_path}",
        )

    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    size = bounds.GetSize()
    center = (minimum + maximum) * 0.5
    up_index = 2 if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else 1
    grasp_axis = max(
        (index for index in range(3) if index != up_index),
        key=lambda index: float(size[index]),
    )
    half_length = max(float(size[grasp_axis]) * 0.25, 1e-6)
    first_world = Gf.Vec3d(center)
    second_world = Gf.Vec3d(center)
    first_world[grasp_axis] -= half_length
    second_world[grasp_axis] += half_length
    world_to_root = (
        UsdGeom.XformCache().GetLocalToWorldTransform(default_prim).GetInverse()
    )
    points = [
        Gf.Vec3f(world_to_root.Transform(first_world)),
        Gf.Vec3f(world_to_root.Transform(second_world)),
    ]
    grasp_path = default_prim.GetPath().AppendChild("grasp_identifier_01")
    grasp = UsdGeom.BasisCurves.Define(stage, grasp_path)
    grasp.CreateTypeAttr(UsdGeom.Tokens.linear)
    grasp.CreateCurveVertexCountsAttr([2])
    grasp.CreatePointsAttr(points)
    width = max(float(max(size)) * 0.01, 1e-6)
    grasp.CreateWidthsAttr([width, width])
    looks_path = default_prim.GetPath().AppendChild("Looks")
    stage.DefinePrim(looks_path, "Scope")
    material = UsdShade.Material.Define(
        stage, looks_path.AppendChild("SimReadyGraspIdentifier")
    )
    shader = UsdShade.Shader.Define(
        stage, material.GetPath().AppendChild("PreviewSurface")
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 0.6, 0.0)
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(grasp.GetPrim()).Bind(material)
    save_error = _save_stage_root_layer(stage)
    if save_error:
        return _blocked_repair_result(
            requirement=requirement,
            asset_path=asset_path,
            output_dir=output_dir,
            reason=f"Could not safely author explicit grasp evidence: {save_error}",
        )

    return _write_repair_result(
        requirement=requirement,
        asset_path=asset_path,
        output_dir=output_dir,
        status="REPAIRED",
        passed=True,
        reason="Authored a grasp line from explicit target-prim bounds evidence.",
        report={
            "schema_version": GSP001_RECEIPT_SCHEMA_VERSION,
            "requirement": requirement,
            "asset_path": str(asset_path),
            "grasp_prim_path": grasp_prim_path,
            "grasp_identifier_path": str(grasp_path),
            "grasp_axis": grasp_axis,
            "points": [[float(value) for value in point] for point in points],
            "changes": [str(grasp_path)],
            "readback_verified": True,
        },
    )


def _grasp_identifier_prims(default_prim: Any, Usd: Any) -> list[Any]:
    return sorted(
        (
            prim
            for prim in Usd.PrimRange(default_prim)
            if prim.GetName().startswith("grasp_identifier")
        ),
        key=lambda prim: str(prim.GetPath()),
    )


def _is_valid_grasp_identifier(prim: Any, UsdGeom: Any) -> bool:
    if not prim.IsA(UsdGeom.BasisCurves):
        return False
    points = prim.GetAttribute("points").Get()
    widths = prim.GetAttribute("widths").Get()
    vertex_counts = prim.GetAttribute("curveVertexCounts").Get()
    if (
        points is None
        or widths is None
        or vertex_counts is None
        or len(points) < 2
        or not widths
        or not vertex_counts
        or any(int(count) < 2 for count in vertex_counts)
        or sum(int(count) for count in vertex_counts) != len(points)
    ):
        return False
    return all(
        len(point) == 3 and all(math.isfinite(float(value)) for value in point)
        for point in points
    ) and all(math.isfinite(float(width)) and float(width) > 0.0 for width in widths)


def _load_grasp_plan(path: Path) -> tuple[SimReadyGraspPlan, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Grasp plan is not a regular file: {path}")
    payload_bytes = path.read_bytes()
    payload = json.loads(
        payload_bytes.decode("utf-8"),
        object_pairs_hook=_json_object_without_duplicate_keys,
        parse_constant=_reject_nonfinite_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("Grasp plan must be a JSON object.")
    return SimReadyGraspPlan.model_validate(payload), hashlib.sha256(
        payload_bytes
    ).hexdigest()


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Grasp plan contains duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ValueError(f"Grasp plan contains non-finite JSON number: {value}")


def _gsp001_identity_path(asset_path: Path) -> Path:
    asset_path = _absolute_path(asset_path)
    if asset_path.is_symlink():
        raise ValueError(f"Source asset identity path is a symlink: {asset_path}")
    if asset_path.is_file():
        return asset_path
    selected_root, path_error = _stage_open_path(asset_path)
    if selected_root is None:
        raise ValueError(
            path_error or f"Unable to select a source identity file: {asset_path}"
        )
    selected_root = _absolute_path(selected_root)
    if selected_root.is_symlink() or not selected_root.is_file():
        raise ValueError(
            f"Source asset identity is not a regular file: {selected_root}"
        )
    return selected_root


def _validate_gsp001_hygiene_lineage(
    *,
    source_lineage: _GSP001SourceLineage | None,
    source_asset_sha256: str,
    staged_identity_path: Path,
    staged_asset_sha256: str,
) -> dict[str, Any]:
    if source_lineage is None:
        raise ValueError(
            "Staged GSP.001 asset no longer matches the exact plan-bound source "
            "asset SHA-256 and has no verified Gate 3A hygiene lineage."
        )
    receipt_path = _absolute_path(source_lineage.receipt_path)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError(
            f"Gate 3A hygiene lineage receipt is not a regular file: {receipt_path}"
        )
    if _file_sha256(receipt_path) != source_lineage.receipt_sha256:
        raise ValueError("Gate 3A hygiene lineage receipt changed before GSP.001.")
    payload_bytes = receipt_path.read_bytes()
    payload = json.loads(
        payload_bytes.decode("utf-8"),
        object_pairs_hook=_json_object_without_duplicate_keys,
        parse_constant=_reject_nonfinite_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("Gate 3A hygiene lineage receipt must be a JSON object.")
    required_truths = (
        "passed",
        "source_identity_verified",
        "dependencies_preserved",
        "readback_verified",
        "physics_inventory_preserved",
    )
    if (
        payload.get("schema_version") != GATE3A_HYGIENE_RECEIPT_SCHEMA_VERSION
        or payload.get("requirement") != GATE3A_HYGIENE_REQUIREMENT
        or payload.get("status") != "REPAIRED"
        or any(payload.get(key) is not True for key in required_truths)
    ):
        raise ValueError("Gate 3A hygiene lineage receipt is not a verified repair.")
    receipt_source_sha256 = payload.get("source_asset_sha256") or payload.get(
        "source_root_sha256"
    )
    if receipt_source_sha256 != source_asset_sha256:
        raise ValueError(
            "Gate 3A hygiene lineage does not start at the plan-bound source asset."
        )
    receipt_output_path = _absolute_path(Path(str(payload.get("output_root", ""))))
    if receipt_output_path != _absolute_path(staged_identity_path):
        raise ValueError(
            "Gate 3A hygiene lineage does not end at the staged GSP.001 asset."
        )
    if payload.get("output_root_sha256") != staged_asset_sha256:
        raise ValueError(
            "Gate 3A hygiene lineage output hash does not match the staged GSP.001 "
            "asset."
        )
    return {
        "kind": "G3A.HYG.001-receipt",
        "receipt_path": str(receipt_path),
        "receipt_sha256": source_lineage.receipt_sha256,
        "source_asset_sha256": source_asset_sha256,
        "derivative_asset_sha256": staged_asset_sha256,
    }


def _gsp001_source_package(
    *, asset_path: Path, output_dir: Path
) -> tuple[Path, Path, Path, Path | None]:
    if asset_path.suffix.lower() == ".usdz":
        extraction_dir = _private_mkdtemp(
            prefix=".gsp001-source-", directory=output_dir
        )
        try:
            root_relative = _isa001_usdz_root(asset_path)
            extract_usdz_members_to_dir(
                asset_path,
                extraction_dir,
                allowed_suffixes=None,
                fail_on_filtered_member=True,
            )
            root_path = extraction_dir / root_relative
            if not root_path.is_file():
                raise ValueError(
                    f"USDZ package root was not extracted: {root_relative.as_posix()}"
                )
        except BaseException:
            shutil.rmtree(extraction_dir, ignore_errors=True)
            raise
        return extraction_dir, root_path, asset_path, extraction_dir

    selected_root, path_error = _stage_open_path(asset_path)
    if selected_root is None:
        raise ValueError(path_error or f"Unable to select a USD root: {asset_path}")
    root_path = _absolute_path(selected_root)
    staging_root = _absolute_path(output_dir / "staged")
    staged_relative = _relative_to(root_path, staging_root)
    hygiene_root = _absolute_path(output_dir / GATE3A_HYGIENE_OUTPUT_DIR)
    hygiene_relative = _relative_to(root_path, hygiene_root)
    if staged_relative is not None:
        source_tree = staging_root
    elif hygiene_relative is not None and len(hygiene_relative.parts) >= 2:
        source_tree = hygiene_root / hygiene_relative.parts[0]
    else:
        raise ValueError(
            "GSP.001 repair input is outside workflow-owned staged or Gate 3A "
            f"hygiene output: {root_path}"
        )
    return source_tree, root_path, root_path, None


def _validate_gsp001_plan(*, stage: Any, plan: SimReadyGraspPlan, Sdf: Any) -> None:
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise ValueError("GSP.001 repair requires a valid default prim.")
    default_path = default_prim.GetPath()
    if plan.default_prim_path != str(default_path):
        raise ValueError(
            "Grasp plan default_prim_path is stale: expected "
            f"{default_path}, received {plan.default_prim_path}."
        )

    for line in plan.grasp_lines:
        path = Sdf.Path(line.prim_path)
        if (
            path.isEmpty
            or not path.IsAbsolutePath()
            or not path.IsPrimPath()
            or path.ContainsPrimVariantSelection()
            or str(path) != line.prim_path
        ):
            raise ValueError(
                f"Grasp line prim_path is not a canonical absolute prim path: "
                f"{line.prim_path}"
            )
        if path == default_path or not path.HasPrefix(default_path):
            raise ValueError(
                f"Grasp line prim_path escapes the default prim: {line.prim_path}"
            )
        if not path.name.startswith("grasp_identifier"):
            raise ValueError(
                "Grasp line prim name must start with grasp_identifier: "
                f"{line.prim_path}"
            )
        if stage.GetPrimAtPath(path):
            raise ValueError(
                f"Grasp line conflicts with an existing prim: {line.prim_path}"
            )
        parent = stage.GetPrimAtPath(path.GetParentPath())
        if (
            not parent
            or not parent.IsValid()
            or not parent.IsActive()
            or not parent.IsDefined()
            or parent.IsAbstract()
            or parent.IsInstance()
            or parent.IsInstanceProxy()
        ):
            raise ValueError(
                "Grasp line parent must be an existing editable prim: "
                f"{path.GetParentPath()}"
            )


def _author_gsp001_lines(
    *,
    stage: Any,
    lines: list[SimReadyGraspLinePlan],
    Gf: Any,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    Vt: Any,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for line in lines:
        curve = UsdGeom.BasisCurves.Define(stage, Sdf.Path(line.prim_path))
        if not curve:
            raise ValueError(f"Could not define grasp line: {line.prim_path}")
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curve.CreateCurveVertexCountsAttr(Vt.IntArray([len(line.points)]))
        curve.CreatePointsAttr(
            Vt.Vec3fArray([Gf.Vec3f(*point) for point in line.points])
        )
        curve.CreateWidthsAttr(Vt.FloatArray(line.widths))
        interpolation = (
            UsdGeom.Tokens.constant if len(line.widths) == 1 else UsdGeom.Tokens.vertex
        )
        if not curve.SetWidthsInterpolation(interpolation):
            raise ValueError(
                f"Could not set grasp-line width interpolation: {line.prim_path}"
            )
        extent = UsdGeom.Boundable(curve).ComputeExtent(Usd.TimeCode.Default())
        if extent is None or len(extent) != 2:
            raise ValueError(f"Could not compute grasp-line extent: {line.prim_path}")
        curve.CreateExtentAttr(extent)
        changes.append(
            {
                "prim_path": line.prim_path,
                "prim_type": "BasisCurves",
                "authored_properties": sorted(_GSP001_AUTHORED_PROPERTIES),
                "point_count": len(line.points),
                "width_count": len(line.widths),
                "widths_interpolation": str(interpolation),
            }
        )
    return changes


def _verify_gsp001_readback(
    *,
    stage: Any,
    lines: list[SimReadyGraspLinePlan],
    Gf: Any,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    Vt: Any,
) -> None:
    root_layer = stage.GetRootLayer()
    for line in lines:
        prim = stage.GetPrimAtPath(Sdf.Path(line.prim_path))
        curve = UsdGeom.BasisCurves(prim)
        if not prim or not prim.IsA(UsdGeom.BasisCurves) or not curve:
            raise ValueError(f"Grasp-line readback is missing: {line.prim_path}")
        prim_spec = root_layer.GetPrimAtPath(prim.GetPath())
        if (
            prim_spec is None
            or set(prim_spec.ListInfoKeys()) != {"specifier", "typeName"}
            or prim_spec.specifier != Sdf.SpecifierDef
            or prim_spec.typeName != "BasisCurves"
            or len(prim.GetPrimStack()) != 1
            or prim.GetAppliedSchemas()
            or prim.GetChildren()
        ):
            raise ValueError(
                f"Grasp-line readback has unplanned prim state: {line.prim_path}"
            )
        property_names = {prop.GetName() for prop in prim.GetAuthoredProperties()}
        if property_names != _GSP001_AUTHORED_PROPERTIES:
            raise ValueError(
                f"Grasp-line readback has unplanned properties at {line.prim_path}: "
                f"{sorted(property_names)}"
            )
        for prop in prim.GetAuthoredProperties():
            prop_spec = root_layer.GetPropertyAtPath(prop.GetPath())
            expected_info = {"custom", "default", "typeName", "variability"}
            if prop.GetName() == "widths":
                expected_info.add("interpolation")
            if prop_spec is None or set(prop_spec.ListInfoKeys()) != expected_info:
                raise ValueError(
                    "Grasp-line readback has unplanned property metadata at "
                    f"{prop.GetPath()}"
                )
        expected_points = Vt.Vec3fArray([Gf.Vec3f(*point) for point in line.points])
        expected_widths = Vt.FloatArray(line.widths)
        expected_extent = UsdGeom.Boundable(curve).ComputeExtent(Usd.TimeCode.Default())
        expected_interpolation = (
            UsdGeom.Tokens.constant if len(line.widths) == 1 else UsdGeom.Tokens.vertex
        )
        if curve.GetTypeAttr().Get() != UsdGeom.Tokens.linear:
            raise ValueError(f"Grasp-line type readback failed: {line.prim_path}")
        if curve.GetWrapAttr().Get() != UsdGeom.Tokens.nonperiodic:
            raise ValueError(f"Grasp-line wrap readback failed: {line.prim_path}")
        if list(curve.GetCurveVertexCountsAttr().Get() or []) != [len(line.points)]:
            raise ValueError(
                f"Grasp-line vertex-count readback failed: {line.prim_path}"
            )
        if curve.GetPointsAttr().Get() != expected_points:
            raise ValueError(f"Grasp-line point readback failed: {line.prim_path}")
        if curve.GetWidthsAttr().Get() != expected_widths:
            raise ValueError(f"Grasp-line width readback failed: {line.prim_path}")
        if curve.GetExtentAttr().Get() != expected_extent:
            raise ValueError(f"Grasp-line extent readback failed: {line.prim_path}")
        if curve.GetWidthsInterpolation() != expected_interpolation:
            raise ValueError(
                f"Grasp-line width interpolation readback failed: {line.prim_path}"
            )


def _gsp001_stage_fingerprint(
    *,
    stage: Any,
    package_root: Path,
    excluded_prim_paths: tuple[str, ...],
    Sdf: Any,
    UsdUtils: Any,
) -> str:
    flattened = stage.Flatten()
    flattened.documentation = ""
    namespace_edits = Sdf.BatchNamespaceEdit()
    for prim_path in sorted(
        excluded_prim_paths,
        key=lambda value: (value.count("/"), value),
        reverse=True,
    ):
        path = Sdf.Path(prim_path)
        if flattened.GetPrimAtPath(path) is None:
            raise ValueError(
                f"Flattened GSP.001 output is missing planned prim: {prim_path}"
            )
        namespace_edits.Add(Sdf.NamespaceEdit.Remove(path))
    if excluded_prim_paths and not flattened.Apply(namespace_edits):
        raise ValueError("Could not remove planned prims from the stage fingerprint.")

    stage_root = Path(stage.GetRootLayer().realPath).parent
    package_root = _absolute_path(package_root)

    def canonicalize(asset_path: str) -> str:
        text = str(asset_path).strip()
        if not text:
            return text
        if "://" in text or text.startswith(("anon:", "file:")):
            raise ValueError(f"Flattened stage contains non-local asset path: {text}")
        path = Path(text)
        path = path if path.is_absolute() else stage_root / path
        relative = _relative_to(_absolute_path(path), package_root)
        if relative is None:
            raise ValueError(f"Flattened stage asset path escapes the package: {text}")
        return f"__GSP001_PACKAGE__/{relative.as_posix()}"

    UsdUtils.ModifyAssetPaths(flattened, canonicalize)
    return hashlib.sha256(flattened.ExportToString().encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_mkdtemp(*, prefix: str, directory: Path) -> Path:
    """Create a workflow temporary directory with an explicit owner-only mode."""

    path = Path(tempfile.mkdtemp(prefix=prefix, dir=directory))
    try:
        os.chmod(path, stat.S_IRWXU, follow_symlinks=False)
        mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        if mode != stat.S_IRWXU:
            raise OSError(f"Temporary directory is not mode 0700: {path}")
    except BaseException:
        shutil.rmtree(path, ignore_errors=True)
        raise
    return path


def _publish_gsp001_tree(
    *, build_dir: Path, publish_root: Path, tree_sha256: str
) -> tuple[Path, str]:
    final_tree = publish_root / tree_sha256
    if final_tree.exists() or final_tree.is_symlink():
        if final_tree.is_symlink() or not final_tree.is_dir():
            raise ValueError(
                f"GSP.001 content-addressed output is not a directory: {final_tree}"
            )
        if _isa001_tree_sha256(final_tree) != tree_sha256:
            raise ValueError(
                "Existing GSP.001 content-addressed output failed identity check: "
                f"{final_tree}"
            )
        _remove_gsp001_build_tree(build_dir)
        return final_tree, "cache_hit"
    try:
        build_dir.replace(final_tree)
    except OSError:
        if not final_tree.is_dir() or _isa001_tree_sha256(final_tree) != tree_sha256:
            raise
        _remove_gsp001_build_tree(build_dir)
        return final_tree, "concurrent_reuse"
    return final_tree, "published"


def _remove_gsp001_build_tree(build_dir: Path) -> None:
    def make_owner_writable(path: Path) -> None:
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"GSP.001 build tree contains a symlink: {path}")
        writable_mode = mode | stat.S_IRUSR | stat.S_IWUSR
        if stat.S_ISDIR(mode):
            writable_mode |= stat.S_IXUSR
        os.chmod(path, writable_mode, follow_symlinks=False)

    def raise_walk_error(error: OSError) -> None:
        raise error

    make_owner_writable(build_dir)
    for directory, directory_names, file_names in os.walk(
        build_dir,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        directory_path = Path(directory)
        for name in directory_names:
            make_owner_writable(directory_path / name)
        for name in file_names:
            make_owner_writable(directory_path / name)
    shutil.rmtree(build_dir)


def _blocked_repair_result(
    *, requirement: str, asset_path: Path, output_dir: Path, reason: str
) -> _RepairResult:
    return _write_repair_result(
        requirement=requirement,
        asset_path=asset_path,
        output_dir=output_dir,
        status="BLOCKED",
        passed=False,
        reason=reason,
        report={
            "schema_version": "content-agent-workflows.simready-repair.v1",
            "requirement": requirement,
            "asset_path": str(asset_path),
            "status": "BLOCKED",
            "reason": reason,
        },
    )


def _write_repair_result(
    *,
    requirement: str,
    asset_path: Path,
    output_dir: Path,
    status: str,
    passed: bool,
    reason: str,
    report: dict[str, Any],
    package_root: Path | None = None,
) -> _RepairResult:
    report = {**report, "status": status, "passed": passed, "reason": reason}
    report_path = output_dir / "reports" / f"{requirement.replace('.', '_')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _RepairResult(
        status=status,
        passed=passed,
        reason=reason,
        output_path=asset_path,
        package_root=package_root,
        report=report,
        report_path=report_path,
    )


def _stage_input(
    asset_path: Path, output_dir: Path, *, force: bool = False
) -> tuple[Path, Path, list[str]]:
    if asset_path.is_symlink():
        raise ValueError(f"Input asset root must not be a symlink: {asset_path}")
    if not asset_path.is_file() and not asset_path.is_dir():
        raise ValueError(
            f"Input asset root is not a regular file or directory: {asset_path}"
        )
    staged_dir = output_dir / "staged"
    warnings = _clear_staged_dir(staged_dir, asset_path) if force else []
    if asset_path.is_dir():
        _reject_staging_tree_symlinks(
            asset_path,
            ignored_roots=(output_dir, staged_dir),
        )
        external_errors, dependency_warnings = _directory_external_dependency_findings(
            asset_path
        )
        if external_errors:
            raise ValueError("; ".join(external_errors))
        staged = staged_dir / asset_path.name
        if _absolute_path(staged) == _absolute_path(asset_path):
            return staged, staged, [*warnings, *dependency_warnings]
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_symlink = _aa001_symlink_component(
            _absolute_path(staged), _absolute_path(staged_dir)
        )
        if staged_symlink is not None:
            raise ValueError(
                f"Staged directory target uses a symlink: {staged_symlink}"
            )
        if staged.exists():
            if not staged.is_dir():
                raise ValueError(
                    f"Staged directory target is not a directory: {staged}"
                )
            _reject_staging_tree_symlinks(staged, ignored_roots=())
        shutil.copytree(
            asset_path,
            staged,
            dirs_exist_ok=True,
            symlinks=False,
            copy_function=_copy_staging_file,
            ignore=_stage_copy_ignore(output_dir=output_dir, staged_dir=staged_dir),
        )
        return staged, staged, [*warnings, *dependency_warnings]
    dependencies, dependency_warnings = _usd_dependency_paths(asset_path)
    package_root = _staging_package_root(asset_path, dependencies)
    staged = staged_dir / asset_path.relative_to(package_root)
    if _absolute_path(staged) == _absolute_path(asset_path):
        return staged, staged_dir, warnings
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged_symlink = _aa001_symlink_component(
        _absolute_path(staged), _absolute_path(staged_dir)
    )
    if staged_symlink is not None:
        raise ValueError(f"Staged asset target uses a symlink: {staged_symlink}")
    _copy_staging_file(str(asset_path), str(staged))
    warnings = [
        *warnings,
        *dependency_warnings,
        *_stage_usd_dependencies(
            asset_path=asset_path,
            dependencies=dependencies,
            package_root=package_root,
            staged_dir=staged_dir,
        ),
    ]
    return staged, staged_dir, warnings


def _clear_staged_dir(staged_dir: Path, asset_path: Path) -> list[str]:
    staged_root = _absolute_path(staged_dir)
    asset_resolved = _absolute_path(asset_path)
    if _relative_to(asset_resolved, staged_root) is not None:
        return ["Skipped clearing staged output because it contains the input asset."]
    if not staged_root.exists():
        return []
    shutil.rmtree(staged_root)
    return []


def _stage_usd_dependencies(
    *,
    asset_path: Path,
    dependencies: list[Path],
    package_root: Path,
    staged_dir: Path,
) -> list[str]:
    warnings: list[str] = []
    staged_root = _absolute_path(staged_dir)
    asset_resolved = _absolute_path(asset_path)
    for dependency in dependencies:
        dependency = _absolute_path(dependency)
        if dependency == asset_resolved:
            continue
        relative_dependency = _relative_to(dependency, package_root)
        if relative_dependency is None:
            raise ValueError(
                "Cannot stage USD dependency outside the asset package without "
                f"rewriting the authored path: {dependency}"
            )
        source_symlink = _aa001_symlink_component(dependency, package_root)
        if source_symlink is not None:
            raise ValueError(
                f"Cannot stage USD dependency through a symlink: {source_symlink}"
            )
        target = staged_dir / relative_dependency
        target_resolved = _absolute_path(target)
        if _relative_to(target_resolved, staged_root) is None:
            warnings.append(
                f"Skipped staging dependency with unsafe path: {dependency}"
            )
            continue
        if target_resolved == dependency:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target_symlink = _aa001_symlink_component(target_resolved, staged_root)
        if target_symlink is not None:
            raise ValueError(
                f"Staged USD dependency target uses a symlink: {target_symlink}"
            )
        _copy_staging_file(str(dependency), str(target))
    return warnings


def _copy_staging_file(source: str, destination: str) -> str:
    source_path = Path(source)
    destination_path = Path(destination)
    source_mode = source_path.stat(follow_symlinks=False).st_mode
    if stat.S_ISLNK(source_mode) or not stat.S_ISREG(source_mode):
        raise ValueError(f"Cannot stage non-regular file: {source_path}")
    if destination_path.is_symlink() or any(
        parent.is_symlink() for parent in destination_path.parents
    ):
        raise ValueError(
            f"Staged file target must not be a symlink: {destination_path}"
        )
    return shutil.copy2(
        source,
        destination,
        follow_symlinks=False,
    )


def _reject_staging_tree_symlinks(
    root: Path, *, ignored_roots: tuple[Path, ...]
) -> None:
    ignored = tuple(_absolute_path(path) for path in ignored_roots)

    def is_ignored(path: Path) -> bool:
        absolute = _absolute_path(path)
        return any(_relative_to(absolute, item) is not None for item in ignored)

    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        directory_path = Path(directory)
        retained_directories: list[str] = []
        for name in directory_names:
            candidate = directory_path / name
            if is_ignored(candidate):
                continue
            mode = candidate.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"Directory asset contains a symlink: {candidate}")
            if not stat.S_ISDIR(mode):
                raise ValueError(
                    f"Directory asset contains a non-directory entry: {candidate}"
                )
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in file_names:
            candidate = directory_path / name
            if is_ignored(candidate):
                continue
            mode = candidate.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"Directory asset contains a symlink: {candidate}")
            if not stat.S_ISREG(mode):
                raise ValueError(
                    f"Directory asset contains a non-regular file: {candidate}"
                )


def _directory_external_dependency_findings(
    asset_dir: Path,
) -> tuple[list[str], list[str]]:
    root = _absolute_path(asset_dir)
    errors: list[str] = []
    warnings: list[str] = []
    for asset_path in sorted(root.rglob("*")):
        if not asset_path.is_file() or asset_path.suffix.lower() not in {
            ".usd",
            ".usda",
            ".usdc",
        }:
            continue
        dependencies, dependency_warnings = _usd_dependency_paths(asset_path)
        warnings.extend(dependency_warnings)
        for dependency in dependencies:
            if _relative_to(_absolute_path(dependency), root) is None:
                errors.append(
                    "Directory asset contains a USD dependency outside the "
                    f"directory package: {asset_path} -> {dependency}"
                )
    return errors, warnings


def _staging_package_root(asset_path: Path, dependencies: list[Path]) -> Path:
    root = _absolute_path(asset_path.parent)
    root_limit = _staging_root_limit(asset_path)
    for dependency in dependencies:
        dependency = _absolute_path(dependency)
        if _relative_to(dependency, root_limit) is None:
            continue
        try:
            common = Path(os.path.commonpath([str(root), str(dependency.parent)]))
        except ValueError:
            continue
        if common == Path(common.anchor):
            continue
        if _relative_to(common, root_limit) is None:
            continue
        root = common
    return root


def _staging_root_limit(asset_path: Path) -> Path:
    root = _absolute_path(asset_path.parent)
    parent = _absolute_path(root.parent)
    return parent if parent != root else root


def _usd_dependency_paths(asset_path: Path) -> tuple[list[Path], list[str]]:
    try:
        from pxr import UsdUtils
    except ImportError:
        return [], [
            "OpenUSD Python APIs are unavailable; only the root USD was staged."
        ]
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(asset_path))
    except Exception as exc:  # pragma: no cover - OpenUSD reports vary by plugin
        return [], [f"Could not inspect USD dependencies for staging: {exc}"]

    paths: set[Path] = set()
    for layer in layers:
        source_root = _layer_source_root(layer, asset_path.parent)
        for dependency in _layer_authored_dependencies(layer):
            if path := _dependency_path(dependency, source_root):
                paths.add(path)
    package_root = _staging_package_root(asset_path, list(paths))
    for asset in assets:
        if path := _dependency_path(asset, asset_path.parent):
            paths.add(path)
        elif path := _resolved_asset_dependency_path(asset, package_root):
            paths.add(path)

    warnings: list[str] = []
    if unresolved:
        sample = ", ".join(str(item) for item in unresolved[:5])
        suffix = "" if len(unresolved) <= 5 else ", ..."
        warnings.append(
            f"Some USD dependencies could not be resolved: {sample}{suffix}"
        )
    return sorted(paths), warnings


def _layer_source_root(layer: Any, default: Path) -> Path:
    identifier = str(getattr(layer, "identifier", "") or "").strip()
    if not identifier:
        return default
    path = Path(identifier)
    if not path.is_absolute():
        return default
    return _absolute_path(path.parent)


def _layer_authored_dependencies(layer: Any) -> list[str]:
    dependencies: set[str] = set()
    for method_name in ("GetCompositionAssetDependencies", "GetExternalReferences"):
        method = getattr(layer, method_name, None)
        if method is None:
            continue
        try:
            values = method()
        except Exception:  # pragma: no cover - OpenUSD layer errors vary
            continue
        dependencies.update(str(value) for value in values or [])
    export_to_string = getattr(layer, "ExportToString", None)
    if (
        export_to_string is not None
        and not dependencies
        and _layer_allows_text_export(layer)
    ):
        try:
            text = export_to_string()
        except Exception:  # pragma: no cover - OpenUSD layer errors vary
            text = ""
        dependencies.update(re.findall(r"@([^@\n]+)@", text))
    return sorted(dependencies)


def _layer_allows_text_export(layer: Any) -> bool:
    file_format = None
    get_file_format = getattr(layer, "GetFileFormat", None)
    if get_file_format is not None:
        try:
            file_format = get_file_format()
        except Exception:  # pragma: no cover - OpenUSD layer errors vary
            file_format = None
    format_id = str(
        getattr(file_format, "formatId", "") or getattr(file_format, "id", "") or ""
    ).lower()
    if format_id:
        return format_id == "usda"
    identifier = str(getattr(layer, "identifier", "") or "").strip().lower()
    return identifier.endswith(".usda")


def _dependency_path(value: Any, source_root: Path) -> Path | None:
    text = (
        str(getattr(value, "identifier", "") or getattr(value, "path", "") or value)
        if value is not None
        else ""
    )
    text = text.strip().strip("@")
    if not text or text.startswith(("anon:", "omniverse:", "http://", "https://")):
        return None
    path = Path(text)
    if path.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", text):
        return None
    path = _absolute_path(source_root / path)
    return path if path.is_file() else None


def _resolved_asset_dependency_path(value: Any, package_root: Path) -> Path | None:
    text = (
        str(getattr(value, "resolvedPath", "") or getattr(value, "path", "") or value)
        if value is not None
        else ""
    )
    text = text.strip().strip("@")
    if not text:
        return None
    raw_path = Path(text)
    if not raw_path.is_absolute():
        return None
    path = _absolute_path(raw_path)
    if _relative_to(path, package_root) is None:
        return None
    return path if path.is_file() else None


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _relative_to(path: Path, root: Path) -> Path | None:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _stage_copy_ignore(output_dir: Path, staged_dir: Path) -> Any:
    output_dir = output_dir.resolve()
    staged_dir = staged_dir.resolve()

    def ignore(src: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        src_path = Path(src).resolve()
        for name in names:
            source_candidate = src_path / name
            if source_candidate.is_symlink():
                raise ValueError(
                    f"Directory asset contains a symlink: {source_candidate}"
                )
            candidate = source_candidate.resolve()
            if (
                candidate == output_dir
                or output_dir in candidate.parents
                or candidate == staged_dir
                or staged_dir in candidate.parents
            ):
                ignored.add(name)
        return ignored

    return ignore


def _failed_requirements(
    validation_report_path: Path | None,
) -> tuple[list[str], list[str]]:
    if validation_report_path is None:
        return [], []
    if not validation_report_path.exists():
        return [], [f"Validation report does not exist: {validation_report_path}"]
    try:
        payload = json.loads(validation_report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], [f"Validation report could not be parsed: {exc}"]
    if not isinstance(payload, dict):
        return [], [f"Validation report is not a JSON object: {validation_report_path}"]
    ignored_requirements = _ignored_requirements(payload)
    rerun_reasons = payload.get("rerun_reasons")
    if isinstance(rerun_reasons, list):
        rerun_requirements = {
            requirement
            for item in rerun_reasons
            if (requirement := _parse_requirement(item)) is not None
            and requirement not in ignored_requirements
        }
        return sorted(rerun_requirements), []
    requirements: set[str] = set()
    for issue in payload.get("issues", []):
        if not isinstance(issue, dict):
            continue
        for key in ("requirement_id", "requirement", "code", "message"):
            if requirement := _parse_requirement(issue.get(key)):
                if requirement not in ignored_requirements:
                    requirements.add(requirement)
                break
    feature_results = payload.get("feature_results")
    items = (
        feature_results.values()
        if isinstance(feature_results, dict)
        else feature_results or []
    )
    for feature in items:
        if not isinstance(feature, dict):
            continue
        failing = feature.get("failing_requirements")
        if isinstance(failing, list):
            for item in failing:
                if requirement := _parse_requirement(item):
                    if requirement not in ignored_requirements:
                        requirements.add(requirement)
        elif isinstance(failing, str):
            requirements.update(
                requirement
                for requirement in _parse_requirements(failing)
                if requirement not in ignored_requirements
            )
    return sorted(requirements), []


def _ignored_requirements(payload: dict[str, Any]) -> set[str]:
    ignored: set[str] = set()
    ignored_issues = payload.get("ignored_issues")
    if not isinstance(ignored_issues, list):
        return ignored
    for issue in ignored_issues:
        if not isinstance(issue, dict):
            continue
        for key in ("requirement_id", "requirement", "code", "message"):
            if requirement := _parse_requirement(issue.get(key)):
                ignored.add(requirement)
                break
    return ignored


def _skill_for_requirement(requirement: str) -> str | None:
    if requirement in CORE_REQUIREMENTS:
        return FOUNDATION_SKILL_BY_AREA["core"]
    if requirement in MINIMAL_REQUIREMENTS:
        return FOUNDATION_SKILL_BY_AREA["minimal"]
    if requirement in MULTIBODY_REQUIREMENTS:
        return FOUNDATION_SKILL_BY_AREA["multibody"]
    if requirement in PHYSICS_MATERIAL_REQUIREMENTS:
        return FOUNDATION_SKILL_BY_AREA["nonvisual_materials"]
    if requirement in GRASP_REQUIREMENTS:
        return FOUNDATION_SKILL_BY_AREA["grasp"]
    if requirement.startswith(("RB.", "PC.", "PM.")):
        return FOUNDATION_SKILL_BY_AREA["rigid_body"]
    if requirement.startswith(("VM.", "MAT.", "TEX.")):
        return FOUNDATION_SKILL_BY_AREA["materials"]
    if requirement.startswith(("NVM.", "NM.")):
        return FOUNDATION_SKILL_BY_AREA["nonvisual_materials"]
    if requirement.startswith(("ROBOT.", "RC.")):
        return FOUNDATION_SKILL_BY_AREA["robot_core"]
    if requirement.startswith(("RM.",)):
        return FOUNDATION_SKILL_BY_AREA["robot_materials"]
    if requirement.startswith(("ART.", "PJ.", "DRV.")):
        return FOUNDATION_SKILL_BY_AREA["base_articulation"]
    return None


def _has_local_repair(requirement: str) -> bool:
    return requirement in {
        ATOMIC_ASSET_PATH_REQUIREMENT,
        GATE3A_HYGIENE_REQUIREMENT,
        ISAAC_COMPOSITION_REQUIREMENT,
        "GSP.001",
        "NP.006",
        "PMT.001",
        "RB.COL.001",
        "RB.COL.002",
        "UN.006",
        "UN.007",
        "VM.MAT.001",
    }


def _foundation_skill_path(foundation_root: Path | None, skill: str) -> Path | None:
    if foundation_root is None:
        return None
    for candidate in _foundation_skill_path_candidates(foundation_root, skill):
        if candidate.exists():
            return candidate
    return None


def _foundation_skill_path_candidates(foundation_root: Path, skill: str) -> list[Path]:
    agent_skill = _foundation_agent_skill_name(skill)
    return [
        foundation_root / "skills" / skill / "SKILL.md",
        foundation_root / ".agents" / "skills" / skill / "SKILL.md",
        foundation_root / ".agents" / "skills" / agent_skill / "SKILL.md",
    ]


def _foundation_agent_skill_name(skill: str) -> str:
    prefix = "simready-foundation-conform-fet-"
    if not skill.startswith(prefix):
        return skill
    suffix = skill.removeprefix(prefix)
    number, separator, topic = suffix.partition("-")
    if not separator:
        return f"simready-conform-fet_{number}"
    return f"simready-conform-fet_{number}-{topic}"


def _step(
    *,
    requirement: str,
    status: str,
    passed: bool,
    input_path: Path,
    output_path: Path,
    reason: str,
    upstream_skill: str | None = None,
    upstream_skill_path: str | None = None,
) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "status": status,
        "passed": passed,
        "input_usd_path": str(input_path),
        "output_usd_path": str(output_path),
        "upstream_skill": upstream_skill,
        "upstream_skill_path": upstream_skill_path,
        "reason": reason,
    }


def _parse_requirement(value: Any) -> str | None:
    if value is None:
        return None
    match = _REQUIREMENT_ID_RE.search(str(value))
    return match.group(0) if match else None


def _parse_requirements(value: str) -> list[str]:
    return _REQUIREMENT_ID_RE.findall(value)


def _write_report(
    report_path: Path, report: SimReadyConformanceReport
) -> SimReadyConformanceReport:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.report_path = str(report_path)
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _as_json(model: BaseModel) -> dict[str, Any]:
    return cast(dict[str, Any], model.model_dump(mode="json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Route staged SimReady profile conformance through Foundation."
    )
    parser.add_argument("asset_path", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", default=DEFAULT_SIMREADY_PROFILE)
    parser.add_argument("--profile-version", default=DEFAULT_SIMREADY_PROFILE_VERSION)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument(
        "--grasp-plan",
        type=Path,
        help="Strict evidence-backed JSON plan for GSP.001 grasp-line repair.",
    )
    parser.add_argument("--source-asset")
    parser.add_argument("--grasp-prim", dest="grasp_prim_path")
    parser.add_argument(
        "--expected-physics-inventory-sha256",
        help=(
            "Mandatory trusted Joint Agent physics-inventory fingerprint when "
            "routing G3A.HYG.001."
        ),
    )
    parser.add_argument("--foundation-root", type=Path)
    parser.add_argument("--foundation-spec-root", type=Path)
    parser.add_argument(
        "--repair", dest="repair_requirements", action="append", default=[]
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when conformance is blocked or failed.",
    )
    args = parser.parse_args(argv)

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(args.asset_path),
            output_dir=str(args.output_dir),
            profile=args.profile,
            profile_version=args.profile_version,
            report_path=str(args.report) if args.report is not None else None,
            validation_report_path=str(args.validation_report)
            if args.validation_report is not None
            else None,
            grasp_plan_path=str(args.grasp_plan)
            if args.grasp_plan is not None
            else None,
            source_asset=args.source_asset,
            grasp_prim_path=args.grasp_prim_path,
            expected_physics_inventory_sha256=(args.expected_physics_inventory_sha256),
            foundation_root=str(args.foundation_root)
            if args.foundation_root is not None
            else None,
            foundation_spec_root=str(args.foundation_spec_root)
            if args.foundation_spec_root is not None
            else None,
            repair_requirements=args.repair_requirements,
            force=args.force,
        )
    )
    print(json.dumps(_as_json(report), indent=2, sort_keys=True))
    if report.passed:
        return 0
    if args.strict or report.status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
