# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: inspect and prepare UV coordinates for texture generation.

The task preserves existing UVs by default, writes a structured
``uv_report.json`` for every run, and only mutates UVs according to the
configured ``texture.uv_policy``.
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd, UsdGeom
from world_understanding.agentic.tasks import Task
from world_understanding.functions.graphics.so_export import (
    _atomic_output_file,
    _normalize_dependency_roots,
    export_stage_portably,
)
from world_understanding.functions.graphics.uv_generation import (
    ProjectionType,
    generate_atlas_uvs,
    generate_projection_uvs,
)

from texture_agent.functions.uv_generation import (
    DEGENERATE_UV_SPAN,
    UVPreparePolicy,
    UVProjectionMode,
    fix_uv_interpolation,
    generate_uvs_for_stage,
    inspect_uvs_for_stage,
    normalize_uvs,
    repair_degenerate_uvs,
)

logger = logging.getLogger(__name__)


_SO_PROJECTION_TYPES = {
    "planar": ProjectionType.PLANAR,
    "spherical": ProjectionType.SPHERICAL,
    "cylindrical": ProjectionType.CYLINDRICAL,
    "triplanar": ProjectionType.TRIPLANAR,
    "cube": ProjectionType.CUBE,
    "box": ProjectionType.CUBE,
}


class UVPreparationError(ValueError):
    """Raised when UV preparation cannot satisfy the configured policy."""


class UVSceneOptimizerGenerationMode(StrEnum):
    """Scene Optimizer UV generation modes exposed by Texture Agent."""

    PROJECTION = "projection"
    ATLAS = "atlas"


def _scene_optimizer_dependency_roots(
    context: dict[str, Any],
    *,
    usd_path: str | Path,
    working_dir: Path,
) -> tuple[Path, ...]:
    """Return the bounded source/cache roots needed by portable SO export."""
    working_root = working_dir.expanduser().resolve()
    if working_root.parent == working_root:
        raise ValueError(
            "approved_dependency_roots must not contain filesystem roots: "
            f"{working_root}"
        )
    source_candidates = [Path(usd_path).expanduser().resolve().parent]

    source_usd_path = context.get("source_usd_path")
    if isinstance(source_usd_path, str | Path) and str(source_usd_path).strip():
        source_candidates.append(Path(source_usd_path).expanduser().resolve().parent)

    dependency_root = context.get("usd_dependency_root")
    if isinstance(dependency_root, str | Path) and str(dependency_root).strip():
        source_candidates.append(Path(dependency_root).expanduser().resolve())

    # Validate caller/source trust roots before creating ``prepared/`` or
    # flattening the stage. The task owns ``working_root`` and may create it;
    # the worker validates the complete tuple again after that happens.
    source_roots = _normalize_dependency_roots(source_candidates)
    return tuple(dict.fromkeys((working_root, *source_roots)))


def _resolve_uv_policy(texture_config: dict[str, Any]) -> UVPreparePolicy:
    policy_value = texture_config.get(
        "uv_policy", UVPreparePolicy.GENERATE_MISSING.value
    )
    try:
        return UVPreparePolicy(policy_value)
    except ValueError:
        valid = [policy.value for policy in UVPreparePolicy]
        raise ValueError(
            f"Invalid UV policy '{policy_value}'. Valid policies: {valid}"
        ) from None


def _resolve_so_generation_mode(
    texture_config: dict[str, Any],
) -> UVSceneOptimizerGenerationMode:
    mode_value = texture_config.get(
        "uv_generation_mode",
        UVSceneOptimizerGenerationMode.PROJECTION.value,
    )
    try:
        return UVSceneOptimizerGenerationMode(str(mode_value).strip().lower())
    except ValueError:
        valid = [mode.value for mode in UVSceneOptimizerGenerationMode]
        raise ValueError(
            f"Invalid UV generation mode '{mode_value}'. Valid modes: {valid}"
        ) from None


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _collect_uv_target_prim_paths(context: dict[str, Any]) -> tuple[str, ...]:
    texture_config = context.get("texture_config", {})
    target_paths: list[str] = []

    raw_plan = context.get("texture_plan")
    explicit_plan_path = context.get("texture_plan_path")
    if raw_plan is not None or explicit_plan_path is not None:
        from texture_agent.tasks.plan_textures import require_executable_texture_plan

        plan = require_executable_texture_plan(context)
        # Every selected unit is in the immutable operator-approved scope. UV
        # preparation starts from the original source asset on each service
        # execution, so retaining every selected member here also preserves
        # already-accepted units during a targeted texture regeneration.
        for unit in plan.selected_units:
            if not unit.member_prim_paths and not unit.member_subset_paths:
                materials = ", ".join(unit.material_prim_paths)
                raise UVPreparationError(
                    "Selected material has no renderable bound geometry: "
                    f"{materials}. Select a material bound to a renderable prim "
                    "or use an explicit renderable prim scope."
                )
            target_paths.extend(unit.member_prim_paths)
            target_paths.extend(
                str(Sdf.Path(path).GetParentPath()) for path in unit.member_subset_paths
            )
    else:
        explicit_paths = texture_config.get("uv_target_prim_paths")
        if explicit_paths is None:
            explicit_paths = texture_config.get("uv_prim_paths")
        target_paths.extend(_as_string_list(explicit_paths))
        material_textures = context.get("material_textures") or {}
        for spec in material_textures.values():
            if not isinstance(spec, dict):
                continue
            target_paths.extend(_as_string_list(spec.get("prim_paths")))
            prim_path = spec.get("prim_path")
            if prim_path:
                target_paths.append(str(prim_path))

            per_prim = spec.get("per_prim") or {}
            if isinstance(per_prim, dict):
                target_paths.extend(str(path) for path in per_prim if str(path).strip())

    normalized: list[str] = []
    for raw_path in target_paths:
        path = str(raw_path).strip().rstrip("/") or "/"
        sdf_path = Sdf.Path(path)
        if (
            not sdf_path.IsAbsolutePath()
            or sdf_path.IsAbsoluteRootPath()
            or not sdf_path.IsPrimPath()
        ):
            raise ValueError(
                "UV target prim paths must be absolute, non-root USD prim paths: "
                f"{raw_path!r}"
            )
        normalized.append(str(sdf_path))
    return tuple(dict.fromkeys(normalized))


def _flatten_for_uv_preparation(stage: Usd.Stage, source_path: str) -> Usd.Stage:
    """Flatten without adding source-path provenance to stage documentation."""

    flat_layer = stage.Flatten(addSourceFileComment=False)
    flat_stage = Usd.Stage.Open(flat_layer)
    if not flat_stage:
        raise RuntimeError(f"Failed to open flattened stage for: {source_path}")
    return flat_stage


def _active_flattened_mesh_backings(stage: Usd.Stage) -> dict[str, str]:
    """Map composed mesh paths to their active flattened root-layer specs."""

    root_layer = stage.GetRootLayer()
    backings: dict[str, str] = {}
    for composed_mesh in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not composed_mesh.IsA(UsdGeom.Mesh):
            continue
        authoring_mesh = (
            composed_mesh.GetPrimInPrototype()
            if composed_mesh.IsInstanceProxy()
            else composed_mesh
        )
        candidates = {
            str(spec.path)
            for spec in authoring_mesh.GetPrimStack()
            if spec.layer == root_layer
            and stage.GetPrimAtPath(spec.path).IsA(UsdGeom.Mesh)
            and not stage.GetPrimAtPath(spec.path).IsInstanceProxy()
        }
        if len(candidates) != 1:
            raise RuntimeError(
                "Composed mesh did not resolve to one active flattened backing: "
                f"{composed_mesh.GetPath()}"
            )
        backings[str(composed_mesh.GetPath())] = next(iter(candidates))
    return backings


def _copy_scene_optimizer_uvs(
    destination_stage: Usd.Stage,
    optimized_stage: Usd.Stage,
    *,
    destination_backings_by_composed: dict[str, str] | None,
    optimized_backings_by_composed: dict[str, str] | None,
) -> int:
    """Copy only Scene Optimizer's UV opinions into the trusted flat stage.

    The isolated SO worker must export a portable USD, which can re-anchor every
    asset dependency into a sidecar.  SO implementations may also author other
    scene opinions while running a UV operation.  Neither is part of the UV
    task's contract, so downstream tasks consume a fresh portable export of the
    pre-SO flattened stage with only ``primvars:st`` and its indices replaced.
    """

    destination_layer = destination_stage.GetRootLayer()
    optimized_layer = optimized_stage.GetRootLayer()

    if (destination_backings_by_composed is None) != (
        optimized_backings_by_composed is None
    ):
        raise RuntimeError(
            "Scene Optimizer UV path mapping is incomplete between source and output"
        )

    if destination_backings_by_composed is None:
        destination_backings = _active_flattened_mesh_backings(destination_stage)
        optimized_backings = _active_flattened_mesh_backings(optimized_stage)
    else:
        assert optimized_backings_by_composed is not None
        destination_backings = destination_backings_by_composed
        optimized_backings = optimized_backings_by_composed

    if set(destination_backings) != set(optimized_backings):
        raise RuntimeError("Scene Optimizer UV output changed the composed mesh scope")
    path_pairs = list(
        dict.fromkeys(
            (
                Sdf.Path(destination_backings[composed_path]),
                Sdf.Path(optimized_backings[composed_path]),
            )
            for composed_path in sorted(destination_backings)
        )
    )

    copied = 0
    for destination_path, optimized_path in path_pairs:
        destination_mesh = destination_layer.GetPrimAtPath(destination_path)
        optimized_mesh = optimized_layer.GetPrimAtPath(optimized_path)
        if destination_mesh is None or destination_mesh.typeName != "Mesh":
            raise RuntimeError(
                f"Scene Optimizer UV destination is not a mesh: {destination_path}"
            )
        if optimized_mesh is None or optimized_mesh.typeName != "Mesh":
            raise RuntimeError(
                f"Scene Optimizer UV output is missing mesh: {optimized_path}"
            )

        optimized_st_path = optimized_path.AppendProperty("primvars:st")
        if optimized_layer.GetPropertyAtPath(optimized_st_path) is None:
            for property_name in ("primvars:st", "primvars:st:indices"):
                if (
                    destination_layer.GetPropertyAtPath(
                        destination_path.AppendProperty(property_name)
                    )
                    is not None
                ):
                    del destination_mesh.properties[property_name]
            continue

        for property_name in ("primvars:st", "primvars:st:indices"):
            destination_property_path = destination_path.AppendProperty(property_name)
            optimized_property_path = optimized_path.AppendProperty(property_name)
            optimized_property = optimized_layer.GetPropertyAtPath(
                optimized_property_path
            )
            destination_property = destination_layer.GetPropertyAtPath(
                destination_property_path
            )
            if optimized_property is not None:
                if not Sdf.CopySpec(
                    optimized_layer,
                    optimized_property_path,
                    destination_layer,
                    destination_property_path,
                ):
                    raise RuntimeError(
                        "Failed to copy Scene Optimizer UV property: "
                        f"{optimized_property_path} -> {destination_property_path}"
                    )
            elif destination_property is not None:
                del destination_mesh.properties[property_name]
        copied += 1

    return copied


def _map_composed_targets_to_flattened_authoring_paths(
    flat_stage: Usd.Stage,
    target_prim_paths: tuple[str, ...],
) -> dict[str, str]:
    """Resolve composed target meshes to mutable specs in a flattened layer.

    Flattening preserves instance proxies at their composed paths but authors
    their backing geometry under synthetic ``/Flattened_Prototype_*`` roots.
    UV helpers cannot write to the proxies, so scoped preparation must mutate
    those exact backing mesh specs while retaining composed paths in reports.
    """
    targets = tuple(Sdf.Path(path) for path in target_prim_paths)
    matched_targets: set[Sdf.Path] = set()
    selected_meshes: list[Usd.Prim] = []
    for prim in flat_stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        prim_path = prim.GetPath()
        matching = [
            target
            for target in targets
            if prim_path == target or prim_path.HasPrefix(target)
        ]
        if not matching:
            continue
        matched_targets.update(matching)
        selected_meshes.append(prim)

    unresolved = [str(path) for path in targets if path not in matched_targets]
    if unresolved:
        raise UVPreparationError(
            "UV target scope did not resolve to composed renderable meshes: "
            + ", ".join(unresolved)
        )

    instances_by_prototype: dict[str, list[str]] = {}
    for prim in flat_stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsInstance():
            continue
        prototype = prim.GetPrototype()
        if prototype.IsValid():
            instances_by_prototype.setdefault(str(prototype.GetPath()), []).append(
                str(prim.GetPath())
            )

    root_layer = flat_stage.GetRootLayer()
    authoring_paths: dict[str, str] = {}
    for mesh_prim in selected_meshes:
        authoring_prim = mesh_prim
        instance_root = None
        if mesh_prim.IsInstanceProxy():
            authoring_prim = mesh_prim.GetPrimInPrototype()
            cursor = mesh_prim.GetParent()
            while cursor.IsValid() and not cursor.IsInstance():
                cursor = cursor.GetParent()
            instance_root = cursor if cursor.IsValid() else None

        if instance_root is not None:
            prototype = instance_root.GetPrototype()
            instance_paths = instances_by_prototype.get(str(prototype.GetPath()), [])
            if len(instance_paths) != 1:
                raise UVPreparationError(
                    "UV target scope maps to a shared instance prototype used by "
                    f"{len(instance_paths)} composed instances ({', '.join(instance_paths)}); "
                    "refusing a scoped edit whose package writeback is ambiguous"
                )

        candidates = {
            str(spec.path)
            for spec in authoring_prim.GetPrimStack()
            if spec.layer == root_layer
            and flat_stage.GetPrimAtPath(spec.path).IsA(UsdGeom.Mesh)
            and not flat_stage.GetPrimAtPath(spec.path).IsInstanceProxy()
        }
        if len(candidates) != 1:
            raise UVPreparationError(
                "UV target mesh could not be resolved to one flattened root-layer "
                f"authoring path: {mesh_prim.GetPath()}"
            )
        authoring_paths[str(mesh_prim.GetPath())] = next(iter(candidates))

    return authoring_paths


def _resolve_uv_target_scope(
    context: dict[str, Any],
) -> tuple[str, ...] | None:
    texture_config = context.get("texture_config", {})
    uv_scope = str(texture_config.get("uv_scope", "stage")).strip().lower()
    if uv_scope in {"stage", "all", "full_stage"}:
        return None
    if uv_scope not in {"target_prims", "targets", "selected_prims"}:
        raise ValueError(
            "Invalid UV scope "
            f"'{texture_config.get('uv_scope')}'. Valid scopes: "
            "['stage', 'target_prims']"
        )

    target_prim_paths = _collect_uv_target_prim_paths(context)
    if not target_prim_paths:
        raise ValueError(
            "texture.uv_scope='target_prims' requires geometry prim paths via "
            "the executable texture plan, texture.uv_target_prim_paths, or "
            "material_textures.<name>.prim_paths"
        )
    return target_prim_paths


def _resolve_uv_mode(
    texture_config: dict[str, Any],
    *,
    projection_required: bool = True,
    warn_on_scene_optimizer_alias: bool = True,
    allow_scene_optimizer_projection: bool = False,
) -> UVProjectionMode:
    # ``uv_mode`` is kept for backward compatibility; ``uv_projection`` is the
    # v0.4 policy field.
    if not projection_required:
        return UVProjectionMode.BOX

    if "uv_mode" in texture_config:
        uv_mode_str = texture_config["uv_mode"]
    elif allow_scene_optimizer_projection:
        # SO supports projections that the Python fallback does not. Defer
        # validation to the SO projection map and use box only if fallback runs.
        uv_mode_str = UVProjectionMode.BOX.value
    else:
        uv_mode_str = texture_config.get("uv_projection", UVProjectionMode.BOX.value)
    if uv_mode_str in {"cube", "triplanar"}:
        if warn_on_scene_optimizer_alias:
            logger.warning(
                "UV projection '%s' requires Scene Optimizer; using Python "
                "box projection instead",
                uv_mode_str,
            )
        uv_mode_str = UVProjectionMode.BOX.value
    try:
        return UVProjectionMode(uv_mode_str)
    except ValueError:
        valid = [mode.value for mode in UVProjectionMode]
        raise ValueError(
            f"Invalid UV projection mode '{uv_mode_str}'. Valid modes: {valid}"
        ) from None


def _write_uv_report(
    report: dict[str, Any],
    working_dir: Path,
    *,
    input_usd: str,
    prepared_usd: str,
    policy: UVPreparePolicy,
    projection: UVProjectionMode | str,
    actions: dict[str, Any],
) -> Path:
    report_dir = working_dir / "prepared"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "uv_report.json"
    payload = {
        **report,
        "input_usd": str(input_usd),
        "prepared_usd": str(prepared_usd),
        "policy": policy.value,
        "projection": projection.value
        if isinstance(projection, UVProjectionMode)
        else str(projection),
        "actions": actions,
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return report_path


def _diagnostic_summary(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for mesh in report.get("meshes", []):
        for diagnostic in mesh.get("diagnostics", []):
            if diagnostic.get("severity") == "error":
                lines.append(
                    f"{diagnostic.get('code')} at {diagnostic.get('prim_path')}: "
                    f"{diagnostic.get('recommended_action')}"
                )
    return lines


def _mesh_matches_target_paths(
    mesh_report: dict[str, Any],
    target_prim_paths: tuple[str, ...] | None,
) -> bool:
    if not target_prim_paths:
        return True
    prim_path = str(mesh_report.get("prim_path", "")).rstrip("/")
    return any(
        prim_path == target_path or prim_path.startswith(f"{target_path}/")
        for target_path in target_prim_paths
    )


def _preflight_policy_errors(
    report: dict[str, Any],
    policy: UVPreparePolicy,
    target_prim_paths: tuple[str, ...] | None = None,
) -> list[str]:
    errors: list[str] = []
    for mesh in report.get("meshes", []):
        if not _mesh_matches_target_paths(mesh, target_prim_paths):
            continue
        status = mesh.get("status")
        mesh_errors: list[str] = []
        if policy == UVPreparePolicy.VALIDATE and status in {
            "missing",
            "repairable",
            "invalid",
        }:
            mesh_errors = _diagnostic_summary({"meshes": [mesh]})
        elif policy == UVPreparePolicy.PRESERVE_OR_FIX and status in {
            "missing",
            "invalid",
        }:
            mesh_errors = _diagnostic_summary({"meshes": [mesh]})
        elif policy == UVPreparePolicy.GENERATE_MISSING and status == "invalid":
            mesh_errors = _diagnostic_summary({"meshes": [mesh]})
        if (
            not mesh_errors
            and policy == UVPreparePolicy.VALIDATE
            and status == "repairable"
        ):
            mesh_errors = [
                f"UV_NOT_READY at {mesh.get('prim_path')}: "
                f"{mesh.get('recommended_action')}"
            ]
        errors.extend(mesh_errors)
    return errors


def _post_mutation_errors(
    report: dict[str, Any],
    target_prim_paths: tuple[str, ...] | None = None,
) -> list[str]:
    errors: list[str] = []
    for mesh in report.get("meshes", []):
        if not _mesh_matches_target_paths(mesh, target_prim_paths):
            continue
        if mesh.get("status") in {"missing", "invalid"}:
            errors.extend(_diagnostic_summary({"meshes": [mesh]}))
    return errors


def _prepare_with_python_uvs(
    usd_path: str,
    working_dir: Path,
    uv_mode: UVProjectionMode,
    policy: UVPreparePolicy,
    normalize_out_of_range: bool,
    repair_degenerate: bool = True,
    degenerate_min_span: float = DEGENERATE_UV_SPAN,
    target_prim_paths: tuple[str, ...] | None = None,
    extra_actions: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise FileNotFoundError(f"Failed to open: {usd_path}")

    flat_stage = _flatten_for_uv_preparation(stage, usd_path)

    pre_report = inspect_uvs_for_stage(flat_stage)
    actions: dict[str, Any] = {
        "backend": "python",
        "generated": 0,
        "fixed_interpolation": 0,
        "degenerate_repaired": 0,
        "normalized": 0,
    }
    if extra_actions:
        actions.update(extra_actions)
    if target_prim_paths:
        actions["uv_scope"] = "target_prims"
        actions["target_prim_paths"] = list(target_prim_paths)
        authoring_targets_by_composed = (
            _map_composed_targets_to_flattened_authoring_paths(
                flat_stage,
                target_prim_paths,
            )
        )
        authoring_target_paths = tuple(
            dict.fromkeys(authoring_targets_by_composed.values())
        )
        target_kwargs = {"target_prim_paths": authoring_target_paths}
    else:
        actions["uv_scope"] = "stage"
        target_kwargs = {}

    preflight_errors = _preflight_policy_errors(
        pre_report,
        policy,
        target_prim_paths=target_prim_paths,
    )
    if preflight_errors:
        report_path = _write_uv_report(
            pre_report,
            working_dir,
            input_usd=usd_path,
            prepared_usd=usd_path,
            policy=policy,
            projection=uv_mode,
            actions=actions,
        )
        actions["uv_report_path"] = str(report_path)
        raise UVPreparationError(
            "UV preparation failed policy preflight: " + "; ".join(preflight_errors)
        )

    if policy == UVPreparePolicy.FORCE_PROJECTION:
        actions["generated"] = generate_uvs_for_stage(
            flat_stage,
            mode=uv_mode,
            overwrite_existing=True,
            **target_kwargs,
        )
    elif policy == UVPreparePolicy.GENERATE_MISSING:
        actions["generated"] = generate_uvs_for_stage(
            flat_stage,
            mode=uv_mode,
            **target_kwargs,
        )

    if policy in {
        UVPreparePolicy.PRESERVE_OR_FIX,
        UVPreparePolicy.GENERATE_MISSING,
    }:
        actions["fixed_interpolation"] = fix_uv_interpolation(
            flat_stage,
            **target_kwargs,
        )

    if repair_degenerate and policy != UVPreparePolicy.VALIDATE:
        actions["degenerate_repaired"] = repair_degenerate_uvs(
            flat_stage,
            min_span=degenerate_min_span,
            **target_kwargs,
        )

    if normalize_out_of_range and policy != UVPreparePolicy.VALIDATE:
        actions["normalized"] = normalize_uvs(
            flat_stage,
            **target_kwargs,
        )

    total_fixes = (
        int(actions["generated"])
        + int(actions["fixed_interpolation"])
        + int(actions["degenerate_repaired"])
        + int(actions["normalized"])
    )

    prepared_path = Path(usd_path)
    if total_fixes > 0:
        prep_dir = working_dir / "prepared"
        prep_dir.mkdir(parents=True, exist_ok=True)
        prepared_path = prep_dir / "prepared_input.usd"
        flat_stage.GetRootLayer().Export(str(prepared_path))

    post_report = inspect_uvs_for_stage(flat_stage)
    post_errors = _post_mutation_errors(
        post_report,
        target_prim_paths=target_prim_paths,
    )
    report_path = _write_uv_report(
        post_report,
        working_dir,
        input_usd=usd_path,
        prepared_usd=str(prepared_path),
        policy=policy,
        projection=uv_mode,
        actions=actions,
    )
    actions["uv_report_path"] = str(report_path)

    if post_errors:
        raise UVPreparationError(
            "UV preparation left meshes not UV-ready: " + "; ".join(post_errors)
        )

    if total_fixes == 0:
        logger.info("No UV fixes needed")
        return usd_path, actions

    return str(prepared_path), actions


class PrepareUVsTask(Task):
    """Prepare UV coordinates for all meshes in the input USD.

    Detects meshes without UVs, with unsafe interpolation/counts, or with
    out-of-range coordinates. Mutations are controlled by ``texture.uv_policy``
    and are saved to a prepared copy when needed.

    Context keys read:
        usd_path (str): Path to the input USD file.
        source_usd_path (str, optional): Immutable original source path.
        usd_dependency_root (str, optional): Trusted input-bundle root.
        working_dir (str): Working directory.
        texture_config (dict): May contain uv_policy, uv_projection/uv_mode,
            uv_normalize_out_of_range, and uv_backend.

    Context keys written:
        usd_path (str): Updated to point to the prepared USD copy.
        source_usd_path (str): Immutable original input identity, populated only
            when the caller did not already provide one.
        uv_preparation (dict): Summary of fixes applied.
    """

    def __init__(self) -> None:
        self.name = "PrepareUVs"
        self.description = "Generate and fix UV coordinates"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        usd_path = context["usd_path"]
        source_usd_path = context.get("source_usd_path")
        if not isinstance(source_usd_path, str) or not source_usd_path.strip():
            # UV preparation may replace ``usd_path`` with a flattened working
            # copy. Retain the immutable input identity so later tasks can
            # preserve its authored composition while carrying the UV edits
            # forward.
            context["source_usd_path"] = usd_path
        working_dir = Path(context["working_dir"])
        texture_config = context.get("texture_config", {})

        uv_policy = _resolve_uv_policy(texture_config)
        uv_backend = texture_config.get("uv_backend", "python")
        scene_optimizer_requested = uv_backend in ("scene_optimizer", "so")
        so_generation_mode = _resolve_so_generation_mode(texture_config)
        if (
            so_generation_mode == UVSceneOptimizerGenerationMode.ATLAS
            and not scene_optimizer_requested
        ):
            raise ValueError(
                "texture.uv_generation_mode='atlas' requires "
                "texture.uv_backend='scene_optimizer'"
            )
        target_prim_paths = _resolve_uv_target_scope(context)
        use_scene_optimizer = scene_optimizer_requested and uv_policy in {
            UVPreparePolicy.GENERATE_MISSING,
            UVPreparePolicy.FORCE_PROJECTION,
        }
        uv_mode = _resolve_uv_mode(
            texture_config,
            projection_required=uv_policy
            in {
                UVPreparePolicy.GENERATE_MISSING,
                UVPreparePolicy.FORCE_PROJECTION,
            },
            warn_on_scene_optimizer_alias=not use_scene_optimizer,
            allow_scene_optimizer_projection=use_scene_optimizer,
        )
        normalize_out_of_range = bool(
            texture_config.get("uv_normalize_out_of_range", False)
        )
        # CAD conversions routinely emit UVs that sit inside [0, 1] but span a
        # tiny fraction of it, so a generated map samples a few texels and
        # renders flat. Repair those by default; normalization only covers
        # out-of-range UVs and would leave these untouched.
        repair_degenerate = bool(texture_config.get("uv_repair_degenerate", True))
        degenerate_min_span = float(
            texture_config.get("uv_degenerate_min_span", DEGENERATE_UV_SPAN)
        )

        logger.info(
            "Preparing UVs for %s (backend=%s, policy=%s, generation_mode=%s, "
            "projection=%s)",
            usd_path,
            uv_backend,
            uv_policy.value,
            so_generation_mode.value,
            uv_mode.value,
        )

        fallback_uv_mode = uv_mode
        fallback_actions: dict[str, Any] | None = None
        if scene_optimizer_requested and not use_scene_optimizer:
            logger.info(
                "Scene Optimizer UV backend configured but skipped because "
                "uv_policy=%s does not require projection; using Python UV "
                "validation/repair",
                uv_policy.value,
            )
        if use_scene_optimizer:
            approved_dependency_roots = _scene_optimizer_dependency_roots(
                context,
                usd_path=usd_path,
                working_dir=working_dir,
            )
            prep_dir = working_dir / "prepared"
            prep_dir.mkdir(parents=True, exist_ok=True)
            flat_input_path = prep_dir / "prepared_input_flat.usd"
            prepared_path = prep_dir / "prepared_input.usd"

            stage = Usd.Stage.Open(str(usd_path))
            if not stage:
                raise FileNotFoundError(f"Failed to open: {usd_path}")
            flat_stage = _flatten_for_uv_preparation(stage, usd_path)
            authoring_targets_by_composed = (
                _map_composed_targets_to_flattened_authoring_paths(
                    flat_stage,
                    target_prim_paths,
                )
                if target_prim_paths
                else None
            )
            authoring_target_paths = (
                tuple(dict.fromkeys(authoring_targets_by_composed.values()))
                if authoring_targets_by_composed
                else None
            )
            flat_stage.GetRootLayer().Export(str(flat_input_path))

            projection_type: ProjectionType | None = None
            if so_generation_mode == UVSceneOptimizerGenerationMode.PROJECTION:
                so_projection = texture_config.get("uv_projection", uv_mode.value)
                projection_type = _SO_PROJECTION_TYPES.get(str(so_projection))
                if projection_type is None:
                    valid = ", ".join(sorted(_SO_PROJECTION_TYPES))
                    raise ValueError(
                        f"Invalid Scene Optimizer UV projection '{so_projection}'. "
                        f"Valid modes: {valid}"
                    )
            overwrite_existing = uv_policy == UVPreparePolicy.FORCE_PROJECTION or bool(
                texture_config.get("uv_overwrite_existing", False)
            )
            target_paths_list = (
                list(authoring_target_paths) if authoring_target_paths else None
            )

            try:
                common_kwargs = {
                    "paths": target_paths_list,
                    "backend": texture_config.get("uv_so_backend", "local"),
                    "allow_remote_fallback": texture_config.get(
                        "uv_allow_remote_fallback", False
                    ),
                    "overwrite_existing": overwrite_existing,
                    "use_world_space_scales": texture_config.get(
                        "uv_use_world_space_scales", True
                    ),
                    "scale_factor": texture_config.get("uv_scale_factor", 0.01),
                    "scale_units": texture_config.get("uv_scale_units", 0.0),
                    "timeout": texture_config.get("uv_timeout", 600),
                    "approved_dependency_roots": approved_dependency_roots,
                }
                if so_generation_mode == UVSceneOptimizerGenerationMode.ATLAS:
                    result = generate_atlas_uvs(
                        flat_input_path,
                        prepared_path,
                        distortion_threshold=texture_config.get(
                            "uv_atlas_distortion_threshold", 3.0
                        ),
                        enable_atlas_packing=texture_config.get(
                            "uv_atlas_enable_packing", True
                        ),
                        **common_kwargs,
                    )
                else:
                    if projection_type is None:
                        raise AssertionError("Scene Optimizer projection type not set")
                    result = generate_projection_uvs(
                        flat_input_path,
                        prepared_path,
                        projection_type=projection_type,
                        **common_kwargs,
                    )
                so_prepared_stage = Usd.Stage.Open(str(prepared_path))
                if not so_prepared_stage:
                    raise RuntimeError(
                        f"Scene Optimizer UV output could not be opened: {prepared_path}"
                    )
                try:
                    post_so_authoring_targets_by_composed = (
                        _map_composed_targets_to_flattened_authoring_paths(
                            so_prepared_stage,
                            target_prim_paths,
                        )
                        if target_prim_paths
                        else None
                    )
                except UVPreparationError as err:
                    raise RuntimeError(
                        "Scene Optimizer UV output could not preserve the scoped "
                        f"target geometry: {err}"
                    ) from err
                uv_writeback_meshes = _copy_scene_optimizer_uvs(
                    flat_stage,
                    so_prepared_stage,
                    destination_backings_by_composed=authoring_targets_by_composed,
                    optimized_backings_by_composed=(
                        post_so_authoring_targets_by_composed
                    ),
                )
                if not export_stage_portably(
                    flat_stage,
                    prepared_path,
                    approved_dependency_roots=approved_dependency_roots,
                ):
                    raise RuntimeError(
                        "Failed to publish UV-only Scene Optimizer output: "
                        f"{prepared_path}"
                    )
                # The SO output layer was opened before the UV-only portable
                # export atomically replaced it.  Refresh USD's layer registry
                # so downstream repair and reporting consume the new file.
                so_prepared_stage.GetRootLayer().Reload(force=True)
                prepared_stage = Usd.Stage.Open(str(prepared_path))
                if not prepared_stage:
                    raise RuntimeError(
                        "UV-only Scene Optimizer output could not be opened: "
                        f"{prepared_path}"
                    )
            except (OSError, RuntimeError) as err:
                fallback_uv_mode = UVProjectionMode.BOX
                fallback_actions = {
                    "fallback_from": {
                        "backend": "scene_optimizer",
                        "generation_mode": so_generation_mode.value,
                        "error": str(err),
                    }
                }
                logger.warning(
                    "Scene Optimizer UV %s failed (%s); falling back to Python "
                    "box projection",
                    so_generation_mode.value,
                    err,
                )
            else:
                fixed_interp = 0
                if uv_policy in {
                    UVPreparePolicy.PRESERVE_OR_FIX,
                    UVPreparePolicy.GENERATE_MISSING,
                }:
                    fixed_interp = fix_uv_interpolation(
                        prepared_stage,
                        target_prim_paths=authoring_target_paths,
                    )
                degenerate_repaired = (
                    repair_degenerate_uvs(
                        prepared_stage,
                        min_span=degenerate_min_span,
                        target_prim_paths=authoring_target_paths,
                    )
                    if repair_degenerate
                    else 0
                )
                normalized = (
                    normalize_uvs(
                        prepared_stage,
                        target_prim_paths=authoring_target_paths,
                    )
                    if normalize_out_of_range
                    else 0
                )
                with _atomic_output_file(prepared_path) as transaction_output:
                    if not prepared_stage.GetRootLayer().Export(
                        str(transaction_output)
                    ):
                        raise RuntimeError(
                            "Failed to publish prepared Scene Optimizer UV stage: "
                            f"{prepared_path}"
                        )

                actions = {
                    "backend": "scene_optimizer",
                    "generation_mode": so_generation_mode.value,
                    "generated": int(result.get("meshes_with_uvs", 0)),
                    "uv_writeback_meshes": uv_writeback_meshes,
                    "fixed_interpolation": fixed_interp,
                    "degenerate_repaired": degenerate_repaired,
                    "normalized": normalized,
                    "so_result": result,
                }
                if projection_type is not None:
                    actions["projection"] = projection_type.name.lower()
                if target_prim_paths:
                    actions["uv_scope"] = "target_prims"
                    actions["target_prim_paths"] = list(target_prim_paths)
                else:
                    actions["uv_scope"] = "stage"
                post_report = inspect_uvs_for_stage(prepared_stage)
                post_errors = _post_mutation_errors(
                    post_report,
                    target_prim_paths=target_prim_paths,
                )
                report_path = _write_uv_report(
                    post_report,
                    working_dir,
                    input_usd=usd_path,
                    prepared_usd=str(prepared_path),
                    policy=uv_policy,
                    projection=(
                        projection_type.name.lower()
                        if projection_type is not None
                        else so_generation_mode.value
                    ),
                    actions=actions,
                )
                actions["uv_report_path"] = str(report_path)
                if post_errors:
                    raise UVPreparationError(
                        "Scene Optimizer UV preparation left meshes not UV-ready: "
                        + "; ".join(post_errors)
                    )

                context["usd_path"] = str(prepared_path)
                context["uv_preparation"] = actions
                logger.info(
                    "UV preparation complete via Scene Optimizer: %s",
                    prepared_path,
                )
                return context

        prepared_usd_path, summary = _prepare_with_python_uvs(
            usd_path,
            working_dir,
            fallback_uv_mode,
            uv_policy,
            normalize_out_of_range,
            repair_degenerate=repair_degenerate,
            degenerate_min_span=degenerate_min_span,
            target_prim_paths=target_prim_paths,
            extra_actions=fallback_actions,
        )

        if fallback_actions and prepared_usd_path != usd_path:
            # SO may have registered a partially written output layer at the
            # canonical prepared path before Python fallback replaces its file.
            # Refresh that registry entry so subsequent tasks in this process
            # compose the Python output rather than the stale SO stage.
            cached_so_layer = Sdf.Layer.Find(prepared_usd_path)
            if cached_so_layer is not None:
                cached_so_layer.Reload(force=True)

        if prepared_usd_path == usd_path:
            context["uv_preparation"] = summary
            return context

        context["usd_path"] = prepared_usd_path
        context["uv_preparation"] = summary

        logger.info(
            "UV preparation complete: %d generated, %d interp fixed, "
            "%d normalized. Saved: %s",
            summary["generated"],
            summary["fixed_interpolation"],
            summary["normalized"],
            prepared_usd_path,
        )

        return context
