# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed composition contract for source CAD/mesh/scene to SimReady USD."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from content_agent_workflows.common.artifacts import file_sha256

CAD_TO_SIMREADY_REQUEST_SCHEMA_VERSION = (
    "content-agent-workflows.cad-to-simready-request.v2"
)
CAD_TO_SIMREADY_RESULT_SCHEMA_VERSION = (
    "content-agent-workflows.cad-to-simready-result.v3"
)
CAD_TO_SIMREADY_FLATTEN_SCHEMA_VERSION = (
    "content-agent-workflows.cad-to-simready-flatten.v1"
)
SIMREADY_PROFILE_VALIDATION_SCHEMA_VERSION = (
    "content-agent-workflows.simready-profile-validation.v3"
)
CAD_TO_SIMREADY_WORKFLOW_SKILL = "content-workflow-cad-to-simready"
CAD_TO_SIMREADY_WORKFLOW_ENTRYPOINT: tuple[str, str] = ("cad-to-simready", "run")

CadToSimReadyStageName = Literal["convert", "material", "physics", "validation"]
CAD_TO_SIMREADY_STAGE_ORDER: tuple[CadToSimReadyStageName, ...] = (
    "convert",
    "material",
    "physics",
    "validation",
)


class CadToSimReadyStep(BaseModel):
    """One ordered domain handoff owned by the composed workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: CadToSimReadyStageName
    name: str = Field(min_length=1)
    required_inputs: tuple[str, ...] = ()
    produced_artifacts: tuple[str, ...] = ()


class CadToSimReadyPreflightStep(BaseModel):
    """One prerequisite gate that must pass before child-agent execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    produced_artifact: str = Field(min_length=1)


CAD_TO_SIMREADY_PREFLIGHT_STEPS: tuple[CadToSimReadyPreflightStep, ...] = (
    CadToSimReadyPreflightStep(
        name="convert-to-usd-preflight",
        produced_artifact="converter_preflight",
    ),
    CadToSimReadyPreflightStep(
        name="physics-runtime-preflight",
        produced_artifact="physics_runtime_preflight",
    ),
    CadToSimReadyPreflightStep(
        name="simready-foundation-preflight",
        produced_artifact="simready_preflight",
    ),
)


CAD_TO_SIMREADY_STEPS: tuple[CadToSimReadyStep, ...] = (
    CadToSimReadyStep(
        stage="convert",
        name="convert-to-usd",
        required_inputs=("source",),
        produced_artifacts=("conversion_report", "converted_usd"),
    ),
    CadToSimReadyStep(
        stage="convert",
        name="canonicalize-usd",
        required_inputs=("converted_usd",),
        produced_artifacts=("canonicalization_report", "canonical_usd"),
    ),
    CadToSimReadyStep(
        stage="material",
        name="assign-materials",
        required_inputs=("canonical_usd",),
        produced_artifacts=("materialized_usd",),
    ),
    CadToSimReadyStep(
        stage="physics",
        name="flatten-for-physics",
        required_inputs=("materialized_usd",),
        produced_artifacts=("flatten_report", "flattened_physics_usd"),
    ),
    CadToSimReadyStep(
        stage="physics",
        name="normalize-units-for-physics",
        required_inputs=("flattened_physics_usd",),
        produced_artifacts=("physics_units_report", "physics_input_usd"),
    ),
    CadToSimReadyStep(
        stage="physics",
        name="apply-physics",
        required_inputs=("physics_input_usd",),
        produced_artifacts=("physics_usd",),
    ),
    CadToSimReadyStep(
        stage="validation",
        name="initial-simready-validation",
        required_inputs=("physics_usd",),
        produced_artifacts=("initial_validation_report",),
    ),
    CadToSimReadyStep(
        stage="validation",
        name="conform-simready-profile",
        required_inputs=("physics_usd", "initial_validation_report", "flatten_report"),
        produced_artifacts=("conformance_report", "final_usd"),
    ),
    CadToSimReadyStep(
        stage="validation",
        name="final-simready-validation",
        required_inputs=("final_usd",),
        produced_artifacts=("final_validation_report",),
    ),
    CadToSimReadyStep(
        stage="validation",
        name="render-final-simready",
        required_inputs=("final_usd", "final_validation_report"),
        produced_artifacts=(
            "final_render_manifest",
            "final_render_receipt",
            "final_render_receipt_checkpoint",
            "final_render_records",
            "final_hero_render",
            "final_multiview_render",
            "final_turntable_render",
        ),
    ),
)


class CadToSimReadyArtifactBinding(BaseModel):
    """Digest-bound identity for one workflow input or output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class CadToSimReadyInvocation(BaseModel):
    """Recorded public workflow or deterministic helper invocation."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    step: str
    argv: list[str]
    exit_code: int


class CadToSimReadyStageRecord(BaseModel):
    """One completed or failed stage with exact artifacts."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "failed", "skipped"]
    invocations: list[CadToSimReadyInvocation] = Field(default_factory=list)
    artifacts: dict[str, CadToSimReadyArtifactBinding] = Field(default_factory=dict)


class CadToSimReadyPreflightRecord(BaseModel):
    """Run-level prerequisite gate completed before any child-agent turn."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "failed", "planned"]
    invocations: list[CadToSimReadyInvocation] = Field(default_factory=list)
    artifacts: dict[str, CadToSimReadyArtifactBinding] = Field(default_factory=dict)


class CadToSimReadyRequest(BaseModel):
    """Frozen inputs and runtime settings for the composed workflow."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = CAD_TO_SIMREADY_REQUEST_SCHEMA_VERSION
    workflow: Literal["cad-to-simready.run"] = "cad-to-simready.run"
    workflow_skill: Literal["content-workflow-cad-to-simready"] = (
        CAD_TO_SIMREADY_WORKFLOW_SKILL
    )
    asset_id: str = Field(min_length=1)
    source: CadToSimReadyArtifactBinding
    source_format: str = Field(min_length=1)
    source_meters_per_unit: float | None = Field(default=None, gt=0)
    requires_cad_converter: bool
    expected_stages: list[CadToSimReadyStageName] = Field(
        default_factory=lambda: list(CAD_TO_SIMREADY_STAGE_ORDER)
    )
    profile: str = Field(min_length=1)
    profile_version: str = Field(min_length=1)
    materials_yaml: CadToSimReadyArtifactBinding
    materials_usd: CadToSimReadyArtifactBinding | None = None
    runner: Literal["codex", "claude"] = "codex"
    model: str | None = None
    model_reasoning_effort: str | None = None
    prompt_mode: Literal["skill-routed"] = "skill-routed"
    max_iterations: int = Field(ge=1)
    install_missing: bool = True
    scene_tool_timeout_seconds: float = Field(default=300.0, gt=0)
    render_width: int = Field(default=768, ge=64)
    render_height: int = Field(default=576, ge=64)
    turntable_frame_count: int = Field(default=24, ge=2)
    turntable_fps: float = Field(default=12.5, gt=0)


class CadToSimReadyResult(BaseModel):
    """Consolidated terminal evidence for the composed workflow."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = CAD_TO_SIMREADY_RESULT_SCHEMA_VERSION
    asset_id: str
    status: Literal["completed", "failed", "blocked", "planned"]
    request: CadToSimReadyArtifactBinding
    preflight: CadToSimReadyPreflightRecord
    expected_stages: list[CadToSimReadyStageName]
    completed_stages: list[CadToSimReadyStageName]
    stages: dict[CadToSimReadyStageName, CadToSimReadyStageRecord]
    profile: str
    profile_version: str
    output_asset: CadToSimReadyArtifactBinding | None = None
    error: str = ""


def source_format(path: Path | str) -> str:
    """Return the canonical source-format token for one source path."""

    resolved = Path(path)
    if resolved.name.lower().endswith(".forge.step"):
        return "step"
    return resolved.suffix.lower().lstrip(".")


def bind_artifact(
    path: Path | str,
    *,
    relative_to: Path | str | None = None,
) -> CadToSimReadyArtifactBinding:
    """Bind one regular file, optionally storing a path relative to the run."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Workflow artifact is missing: {resolved}")
    stored_path = resolved
    if relative_to is not None:
        root = Path(relative_to).expanduser().resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Workflow artifact escapes run directory: {resolved}")
        stored_path = resolved.relative_to(root)
    return CadToSimReadyArtifactBinding(
        path=str(stored_path),
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def flatten_usd_for_physics(
    source: Path | str,
    output: Path | str,
    report: Path | str,
    *,
    source_meters_per_unit: float | None = None,
) -> dict[str, object]:
    """Flatten material output, apply an explicit unit contract, and bind bytes."""

    from pxr import Sdf, Usd, UsdGeom

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    report_path = Path(report).expanduser().resolve()
    if not source_path.is_file():
        raise ValueError(f"materialized USD is missing: {source_path}")
    if source_path == output_path:
        raise ValueError("flattened physics input must not overwrite its source")
    if source_meters_per_unit is not None and (
        not math.isfinite(source_meters_per_unit) or source_meters_per_unit <= 0
    ):
        raise ValueError("source_meters_per_unit must be finite and positive")
    stage = Usd.Stage.Open(str(source_path))
    if stage is None:
        raise ValueError(f"unable to open materialized USD: {source_path}")
    input_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(input_meters_per_unit) or input_meters_per_unit <= 0:
        raise ValueError("input stage metersPerUnit must be finite and positive")
    input_meters_per_unit_authored = bool(UsdGeom.StageHasAuthoredMetersPerUnit(stage))

    # OpenUSD preserves native instances when Flatten() runs. It moves their
    # backing geometry into root-level ``Flattened_Prototype_*`` specs and keeps
    # references from the instance roots. Clearing ``instanceable`` after that
    # operation leaves the mixed topology behind. Some consumers can compose
    # it, but it is not the concrete physics-authoring handoff promised here.
    # Author session-layer overrides first so Flatten() writes ordinary geometry
    # under each instance root without mutating the materialized source layer.
    instance_paths: list[Sdf.Path] = []
    deinstanced_paths: set[Sdf.Path] = set()
    session_layer = stage.GetSessionLayer()
    with Usd.EditContext(stage, session_layer):
        while True:
            pending_prims = [
                prim
                for prim in stage.TraverseAll()
                if (prim.IsInstance() or prim.IsInstanceable())
                and not prim.IsInstanceProxy()
                and prim.GetPath() not in deinstanced_paths
            ]
            if not pending_prims:
                break
            for prim in pending_prims:
                instance_path = prim.GetPath()
                if not prim.SetInstanceable(False):
                    raise ValueError(f"unable to deinstance USD prim: {instance_path}")
                instance_paths.append(instance_path)
                deinstanced_paths.add(instance_path)

    flattened_layer = stage.Flatten()
    flattened = Usd.Stage.Open(flattened_layer)
    if flattened is None:
        raise ValueError(f"unable to compose flattened USD: {source_path}")

    # A material handoff may already contain synthetic prototype specs from an
    # earlier raw Flatten(). All composition arcs are resolved now, so those
    # root specs are unreachable duplicates and must not leak into the final
    # physics-authoring stage.
    removed_flattened_prototype_paths: list[str] = []
    default_prim = flattened.GetDefaultPrim()
    default_prim_path = default_prim.GetPath() if default_prim else Sdf.Path.emptyPath
    for root_spec in list(flattened.GetRootLayer().rootPrims):
        if not root_spec.name.startswith("Flattened_Prototype_"):
            continue
        prototype_path = root_spec.path
        if prototype_path == default_prim_path:
            raise ValueError("flattened USD default prim is a synthetic prototype")
        if not flattened.RemovePrim(prototype_path):
            raise ValueError(f"unable to remove synthetic prototype: {prototype_path}")
        removed_flattened_prototype_paths.append(str(prototype_path))

    removed_prototype_paths = [
        Sdf.Path(path) for path in removed_flattened_prototype_paths
    ]
    stale_relationship_targets: list[str] = []
    for prim in flattened.TraverseAll():
        for relationship in prim.GetRelationships():
            for target_path in relationship.GetTargets():
                if any(
                    target_path.HasPrefix(prototype_path)
                    for prototype_path in removed_prototype_paths
                ):
                    stale_relationship_targets.append(
                        f"{relationship.GetPath()} -> {target_path}"
                    )
    if stale_relationship_targets:
        targets = ", ".join(stale_relationship_targets)
        raise ValueError(
            "flattened USD contains relationships targeting removed prototypes: "
            f"{targets}"
        )

    if source_meters_per_unit is not None:
        UsdGeom.SetStageMetersPerUnit(flattened, source_meters_per_unit)

    remaining_instances = [
        str(prim.GetPath())
        for prim in flattened.TraverseAll()
        if prim.IsInstance() or prim.IsInstanceable() or prim.IsInstanceProxy()
    ]
    if remaining_instances or flattened.GetPrototypes():
        paths = ", ".join(remaining_instances) or "runtime prototypes"
        raise ValueError(f"flattened USD still contains instances: {paths}")

    output_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(flattened))
    default_prim = flattened.GetDefaultPrim()
    if not default_prim:
        raise ValueError(f"flattened USD has no default prim: {source_path}")
    default_prim_path = str(default_prim.GetPath())
    if not default_prim_path.startswith("/"):
        raise ValueError("flattened USD default prim path is invalid")

    renderable_mesh_paths: list[str] = []
    for prim in Usd.PrimRange(default_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if imageable.ComputePurpose() == UsdGeom.Tokens.guide:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or len(points) == 0:
            continue
        if face_vertex_counts is None or len(face_vertex_counts) == 0:
            continue
        if any(count <= 0 for count in face_vertex_counts):
            continue
        if face_vertex_indices is None or sum(face_vertex_counts) != len(
            face_vertex_indices
        ):
            continue
        if any(index < 0 or index >= len(points) for index in face_vertex_indices):
            continue
        renderable_mesh_paths.append(str(prim.GetPath()))
    if not renderable_mesh_paths:
        raise ValueError(
            "flattened USD default prim exposes no visible renderable meshes: "
            f"{default_prim_path}"
        )

    missing_output_dirs: list[Path] = []
    candidate = output_path.parent
    while not candidate.exists():
        missing_output_dirs.append(candidate)
        candidate = candidate.parent
    created_output_dirs: list[Path] = []
    temporary_path: Path | None = None
    existing_output_mode = (
        output_path.stat().st_mode & 0o7777 if output_path.is_file() else None
    )
    try:
        for directory in reversed(missing_output_dirs):
            try:
                directory.mkdir()
            except FileExistsError:
                continue
            created_output_dirs.append(directory)
        temporary_file = tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=".flatten-",
            suffix=output_path.suffix or ".usd",
            delete=False,
        )
        temporary_path = Path(temporary_file.name)
        temporary_file.close()
        temporary_path.unlink()
        if not flattened.GetRootLayer().Export(str(temporary_path)):
            raise RuntimeError("OpenUSD layer export returned false")
        if existing_output_mode is not None:
            temporary_path.chmod(existing_output_mode)
        temporary_path.replace(output_path)
    except Exception as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        for directory in reversed(created_output_dirs):
            try:
                directory.rmdir()
            except OSError:
                break
        raise ValueError(f"unable to export flattened USD: {output_path}") from exc

    payload: dict[str, object] = {
        "schema_version": CAD_TO_SIMREADY_FLATTEN_SCHEMA_VERSION,
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "default_prim_path": default_prim_path,
        "cleared_instanceable_count": len(instance_paths),
        "materialized_instance_paths": [str(path) for path in instance_paths],
        "removed_flattened_prototype_paths": removed_flattened_prototype_paths,
        "renderable_mesh_count": len(renderable_mesh_paths),
        "renderable_mesh_paths": renderable_mesh_paths,
        "input_meters_per_unit": input_meters_per_unit,
        "input_meters_per_unit_authored": input_meters_per_unit_authored,
        "source_meters_per_unit_override": source_meters_per_unit,
        "output_meters_per_unit": output_meters_per_unit,
        "unit_source": (
            "explicit-source-contract"
            if source_meters_per_unit is not None
            else "input-stage-metadata"
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def build_stage_record(
    stage: CadToSimReadyStageName,
    *,
    run_dir: Path,
    invocations: list[CadToSimReadyInvocation],
    artifacts: dict[str, Path],
    accepted_nonzero_steps: frozenset[str] = frozenset(),
) -> CadToSimReadyStageRecord:
    """Build one fail-closed stage record from the workflow contract.

    ``accepted_nonzero_steps`` is reserved for commands whose nonzero status is
    itself a successfully produced domain verdict. The original exit code stays
    in the invocation evidence; callers must verify the verdict artifact before
    opting a step into this set.
    """

    expected_steps = [step for step in CAD_TO_SIMREADY_STEPS if step.stage == stage]
    expected_artifacts = {
        artifact for step in expected_steps for artifact in step.produced_artifacts
    }
    bindings = {
        name: bind_artifact(path, relative_to=run_dir)
        for name, path in artifacts.items()
        if path.is_file()
    }
    if not invocations and not bindings:
        return CadToSimReadyStageRecord(status="skipped")
    complete = bool(
        len(invocations) == len(expected_steps)
        and [invocation.step for invocation in invocations]
        == [step.name for step in expected_steps]
        and all(
            invocation.exit_code == 0 or invocation.step in accepted_nonzero_steps
            for invocation in invocations
        )
        and set(bindings) == expected_artifacts
    )
    return CadToSimReadyStageRecord(
        status="completed" if complete else "failed",
        invocations=invocations,
        artifacts=bindings,
    )


def build_preflight_record(
    *,
    run_dir: Path,
    invocations: list[CadToSimReadyInvocation],
    artifact_paths: MutableMapping[str, Path],
    planned: bool = False,
) -> CadToSimReadyPreflightRecord:
    """Build the fail-closed run-level readiness record."""

    artifacts = {
        step.produced_artifact: bind_artifact(
            artifact_paths[step.produced_artifact], relative_to=run_dir
        )
        for step in CAD_TO_SIMREADY_PREFLIGHT_STEPS
        if step.produced_artifact in artifact_paths
        and artifact_paths[step.produced_artifact].is_file()
    }
    complete = bool(
        len(invocations) == len(CAD_TO_SIMREADY_PREFLIGHT_STEPS)
        and [invocation.step for invocation in invocations]
        == [step.name for step in CAD_TO_SIMREADY_PREFLIGHT_STEPS]
        and all(invocation.exit_code == 0 for invocation in invocations)
        and set(artifacts)
        == {step.produced_artifact for step in CAD_TO_SIMREADY_PREFLIGHT_STEPS}
    )
    return CadToSimReadyPreflightRecord(
        status="planned" if planned else ("completed" if complete else "failed"),
        invocations=invocations,
        artifacts=artifacts,
    )


def execute_cad_to_simready_workflow(
    *,
    request: CadToSimReadyRequest,
    request_path: Path,
    run_dir: Path,
    artifact_paths: MutableMapping[str, Path],
    invoke: Callable[[str], CadToSimReadyInvocation],
    planned: bool = False,
) -> CadToSimReadyResult:
    """Own ordered execution, artifact gates, and stop policy for the workflow."""

    if planned:
        stages = {
            stage: CadToSimReadyStageRecord(status="skipped")
            for stage in CAD_TO_SIMREADY_STAGE_ORDER
        }
        return build_cad_to_simready_result(
            request=request,
            request_path=request_path,
            run_dir=run_dir,
            preflight=build_preflight_record(
                run_dir=run_dir,
                invocations=[],
                artifact_paths=artifact_paths,
                planned=True,
            ),
            stages=stages,
            final_usd=None,
            planned=True,
        )

    preflight_invocations: list[CadToSimReadyInvocation] = []
    for step in CAD_TO_SIMREADY_PREFLIGHT_STEPS:
        invocation = invoke(step.name)
        preflight_invocations.append(invocation)
        artifact = artifact_paths.get(step.produced_artifact)
        if invocation.exit_code != 0 or artifact is None or not artifact.is_file():
            break
    preflight = build_preflight_record(
        run_dir=run_dir,
        invocations=preflight_invocations,
        artifact_paths=artifact_paths,
    )

    stage_invocations: dict[CadToSimReadyStageName, list[CadToSimReadyInvocation]] = {
        stage: [] for stage in CAD_TO_SIMREADY_STAGE_ORDER
    }
    accepted_nonzero_steps: dict[CadToSimReadyStageName, set[str]] = {
        stage: set() for stage in CAD_TO_SIMREADY_STAGE_ORDER
    }
    if preflight.status == "completed":
        for step in CAD_TO_SIMREADY_STEPS:
            if any(
                name not in artifact_paths or not artifact_paths[name].is_file()
                for name in step.required_inputs
            ):
                break
            invocation = invoke(step.name)
            stage_invocations[step.stage].append(invocation)
            missing_outputs = any(
                name not in artifact_paths or not artifact_paths[name].is_file()
                for name in step.produced_artifacts
            )
            validation_report = artifact_paths.get("final_validation_report")
            validation_domain_failure = bool(
                step.name == "final-simready-validation"
                and invocation.exit_code != 0
                and not missing_outputs
                and validation_report is not None
                and _is_completed_profile_failure(
                    validation_report,
                    artifact_paths.get("final_usd"),
                )
            )
            # A verified FAIL report is a completed domain verdict rather than a
            # validator crash, so preserve its exact USD and continue to
            # diagnostic rendering. The benchmark score still fails the asset's
            # SimReady requirement, while workflow execution remains complete.
            if validation_domain_failure:
                accepted_nonzero_steps[step.stage].add(step.name)
            if missing_outputs or (
                invocation.exit_code != 0 and not validation_domain_failure
            ):
                break

    stages = {
        stage: build_stage_record(
            stage,
            run_dir=run_dir,
            invocations=stage_invocations[stage],
            artifacts={
                name: artifact_paths[name]
                for step in CAD_TO_SIMREADY_STEPS
                if step.stage == stage
                for name in step.produced_artifacts
                if name in artifact_paths
            },
            accepted_nonzero_steps=frozenset(accepted_nonzero_steps[stage]),
        )
        for stage in CAD_TO_SIMREADY_STAGE_ORDER
    }
    return build_cad_to_simready_result(
        request=request,
        request_path=request_path,
        run_dir=run_dir,
        preflight=preflight,
        stages=stages,
        final_usd=artifact_paths.get("final_usd"),
    )


def _is_completed_profile_failure(
    report_path: Path,
    asset_path: Path | None,
) -> bool:
    """Return whether a validator artifact records a completed negative verdict."""

    if asset_path is None or not asset_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            return False
        reported_asset = Path(str(report.get("asset_path") or "")).resolve(strict=True)
        expected_asset = asset_path.resolve(strict=True)
        expected_digest = file_sha256(expected_asset)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        report.get("schema_version") == SIMREADY_PROFILE_VALIDATION_SCHEMA_VERSION
        and reported_asset == expected_asset
        and report.get("asset_sha256") == expected_digest
        and report.get("foundation_checkout_verified") is True
        and report.get("validator_runtime_verified") is True
        and isinstance(report.get("profile_name"), str)
        and bool(report["profile_name"])
        and isinstance(report.get("profile_version"), str)
        and bool(report["profile_version"])
        and isinstance(report.get("errors"), list)
        and report.get("passed") is False
        and str(report.get("status") or "").upper() == "FAIL"
    )


def build_cad_to_simready_result(
    *,
    request: CadToSimReadyRequest,
    request_path: Path,
    run_dir: Path,
    preflight: CadToSimReadyPreflightRecord,
    stages: dict[CadToSimReadyStageName, CadToSimReadyStageRecord],
    final_usd: Path | None,
    planned: bool = False,
) -> CadToSimReadyResult:
    """Consolidate stage evidence and enforce terminal completion semantics."""

    completed = [
        stage
        for stage in CAD_TO_SIMREADY_STAGE_ORDER
        if stages[stage].status == "completed"
    ]
    output = (
        bind_artifact(final_usd, relative_to=run_dir)
        if final_usd is not None and final_usd.is_file()
        else None
    )
    if planned:
        status: Literal["completed", "failed", "blocked", "planned"] = "planned"
        error = ""
    elif preflight.status != "completed":
        status = "blocked"
        error = "workflow prerequisites are not ready"
    elif completed == list(CAD_TO_SIMREADY_STAGE_ORDER) and output is not None:
        status = "completed"
        error = ""
    else:
        status = "failed"
        error = "one or more required stages failed"
    return CadToSimReadyResult(
        asset_id=request.asset_id,
        status=status,
        request=bind_artifact(request_path, relative_to=run_dir),
        preflight=preflight,
        expected_stages=list(CAD_TO_SIMREADY_STAGE_ORDER),
        completed_stages=completed,
        stages=stages,
        profile=request.profile,
        profile_version=request.profile_version,
        output_asset=output,
        error=error,
    )
