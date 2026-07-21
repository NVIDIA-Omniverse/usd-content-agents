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

from pxr import Usd
from world_understanding.agentic.tasks import Task
from world_understanding.functions.graphics.uv_generation import (
    ProjectionType,
    generate_atlas_uvs,
    generate_projection_uvs,
)

from texture_agent.functions.uv_generation import (
    UVPreparePolicy,
    UVProjectionMode,
    fix_uv_interpolation,
    generate_uvs_for_stage,
    inspect_uvs_for_stage,
    normalize_uvs,
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
        path = str(raw_path).strip()
        if path:
            normalized.append(path.rstrip("/"))
    return tuple(dict.fromkeys(normalized))


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
            "texture.uv_target_prim_paths or material_textures.<name>.prim_paths"
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
    target_prim_paths: tuple[str, ...] | None = None,
    extra_actions: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise FileNotFoundError(f"Failed to open: {usd_path}")

    flat_layer = stage.Flatten()
    flat_stage = Usd.Stage.Open(flat_layer)
    if not flat_stage:
        raise RuntimeError(f"Failed to open flattened stage for: {usd_path}")

    pre_report = inspect_uvs_for_stage(flat_stage)
    actions: dict[str, Any] = {
        "backend": "python",
        "generated": 0,
        "fixed_interpolation": 0,
        "normalized": 0,
    }
    if extra_actions:
        actions.update(extra_actions)
    if target_prim_paths:
        actions["uv_scope"] = "target_prims"
        actions["target_prim_paths"] = list(target_prim_paths)
        target_kwargs = {"target_prim_paths": target_prim_paths}
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

    if normalize_out_of_range and policy != UVPreparePolicy.VALIDATE:
        actions["normalized"] = normalize_uvs(
            flat_stage,
            **target_kwargs,
        )

    total_fixes = (
        int(actions["generated"])
        + int(actions["fixed_interpolation"])
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
        working_dir (str): Working directory.
        texture_config (dict): May contain uv_policy, uv_projection/uv_mode,
            uv_normalize_out_of_range, and uv_backend.

    Context keys written:
        usd_path (str): Updated to point to the prepared USD copy.
        uv_preparation (dict): Summary of fixes applied.
    """

    def __init__(self) -> None:
        self.name = "PrepareUVs"
        self.description = "Generate and fix UV coordinates"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        usd_path = context["usd_path"]
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
            prep_dir = working_dir / "prepared"
            prep_dir.mkdir(parents=True, exist_ok=True)
            flat_input_path = prep_dir / "prepared_input_flat.usd"
            prepared_path = prep_dir / "prepared_input.usd"

            stage = Usd.Stage.Open(str(usd_path))
            if not stage:
                raise FileNotFoundError(f"Failed to open: {usd_path}")
            flat_stage = Usd.Stage.Open(stage.Flatten())
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
            target_paths_list = list(target_prim_paths) if target_prim_paths else None

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
                prepared_stage = Usd.Stage.Open(str(prepared_path))
                if not prepared_stage:
                    raise RuntimeError(
                        f"Scene Optimizer UV output could not be opened: {prepared_path}"
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
                        target_prim_paths=target_prim_paths,
                    )
                normalized = (
                    normalize_uvs(
                        prepared_stage,
                        target_prim_paths=target_prim_paths,
                    )
                    if normalize_out_of_range
                    else 0
                )
                prepared_stage.GetRootLayer().Export(str(prepared_path))

                actions = {
                    "backend": "scene_optimizer",
                    "generation_mode": so_generation_mode.value,
                    "generated": int(result.get("meshes_with_uvs", 0)),
                    "fixed_interpolation": fixed_interp,
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
            target_prim_paths=target_prim_paths,
            extra_actions=fallback_actions,
        )

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
