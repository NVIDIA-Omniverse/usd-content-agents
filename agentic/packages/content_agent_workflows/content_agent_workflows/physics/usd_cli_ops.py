# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-level usd-cli execution helpers for the workflow-owned physics pipeline.

This module deliberately accepts already-decided edits. Component grouping,
topology policy, acceptance criteria, and workflow verdicts remain in
``content_agent_workflows.physics.workflow``.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

from content_agent_workflows.common.artifacts import (
    atomic_write_text,
    contained_regular_file,
    file_sha256,
    read_contained_artifact,
)
from content_agent_workflows.common.usd_cli import validate_ovrtx_probe
from content_agent_workflows.common.usd_cli_session import WorkflowUsdCliSession


class UsdCliPhysicsError(RuntimeError):
    """Raised when a required low-level usd-cli operation cannot be completed."""


_PHYSICS_VISUAL_REVIEW_MAX_FRAMES = 8


def _format_time_code(value: float) -> str:
    normalized = 0.0 if abs(value) < 1e-12 else value
    return str(int(normalized)) if normalized.is_integer() else f"{normalized:.15g}"


def _representative_frame_spec(
    recording: Path,
    *,
    max_frames: int = _PHYSICS_VISUAL_REVIEW_MAX_FRAMES,
) -> str | None:
    """Return evenly spaced authored frames for bounded behavior review."""

    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(str(recording))
        if stage is None:
            return None
        if stage.HasAuthoredTimeCodeRange():
            start = float(stage.GetStartTimeCode())
            end = float(stage.GetEndTimeCode())
        else:
            samples = [
                time_code
                for prim in stage.Traverse(Usd.TraverseInstanceProxies())
                for attribute in prim.GetAttributes()
                for time_code in attribute.GetTimeSamples()
            ]
            if not samples:
                return None
            start = float(min(samples))
            end = float(max(samples))
    except (OSError, RuntimeError, ValueError):
        return None
    if end < start:
        return None
    if start.is_integer() and end.is_integer():
        start_i = int(start)
        end_i = int(end)
        frame_count = end_i - start_i + 1
        if frame_count <= max_frames:
            return f"{start_i}:{end_i}" if frame_count > 1 else str(start_i)
        frames = sorted(
            {
                round(start_i + index * (end_i - start_i) / (max_frames - 1))
                for index in range(max_frames)
            }
        )
        return ",".join(str(frame) for frame in frames)

    if start == end:
        return _format_time_code(start)
    frame_count = min(max_frames, max(2, math.ceil(end - start) + 1))
    frames = [
        start + index * (end - start) / (frame_count - 1)
        for index in range(frame_count)
    ]
    return ",".join(_format_time_code(frame) for frame in frames)


def _write_json(path: Path, payload: object, *, within: Path) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        within=within,
    )


def _create_owned_session(
    *,
    run_root: Path,
    purpose: str,
    input_roots: tuple[Path, ...],
) -> WorkflowUsdCliSession:
    """Create one workflow-owned sidecar for a standalone physics operation."""

    try:
        return WorkflowUsdCliSession.create(
            owner_root=run_root,
            # Physics outputs are explicitly validated by the workflow beneath the
            # run root, so that root is also the sidecar's output-confined project.
            project_dir=run_root,
            identity=f"physics:{purpose}:{run_root}",
            workflow="physics-operations",
            input_roots=input_roots,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise UsdCliPhysicsError(
            f"Could not create the workflow-owned usd-cli physics session: {exc}"
        ) from exc


def _run_json(
    *,
    project_dir: Path,
    session_id: str,
    arguments: list[str],
    timeout_seconds: float = 1800.0,
    allow_not_ok: bool = False,
    workflow_session: WorkflowUsdCliSession | None = None,
) -> dict[str, Any]:
    if workflow_session is None:
        raise UsdCliPhysicsError(
            "Physics usd-cli operations require a workflow session"
        )
    try:
        requested_project = project_dir.expanduser().resolve(strict=True)
        session_project = workflow_session.project_dir.resolve(strict=True)
    except OSError as exc:
        raise UsdCliPhysicsError(
            "Physics usd-cli operations require an existing workflow project"
        ) from exc
    if (
        requested_project != session_project
        or session_id != workflow_session.session_id
    ):
        raise UsdCliPhysicsError(
            "Physics usd-cli command identity does not match the workflow session"
        )
    try:
        if allow_not_ok:
            command_result = workflow_session.run_json_result(
                arguments,
                timeout_seconds=timeout_seconds,
                allow_not_ok=True,
            )
            payload = command_result.payload
            returncode = command_result.returncode
        else:
            payload = workflow_session.run_json(
                arguments,
                timeout_seconds=timeout_seconds,
            )
            returncode = 0
    except RuntimeError as exc:
        raise UsdCliPhysicsError(
            f"usd-cli command could not complete: {arguments!r}: {exc}"
        ) from exc
    payload = dict(payload)
    payload["_workflow_command"] = {
        "argv": [
            str(workflow_session.route.wrapper),
            "--json",
            "--session",
            workflow_session.session_id,
            *arguments,
        ],
        "returncode": returncode,
    }
    return payload


def _prepare_contained_output(run_root: Path, target: Path) -> Path:
    """Create the output parent no-follow and reject an escaping/symlink target."""

    root = run_root.resolve(strict=True)
    absolute = Path(os.path.abspath(target))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise UsdCliPhysicsError(
            f"usd-cli physics output must stay within the workflow run: {absolute}"
        ) from exc
    if relative.name in {"", ".", ".."}:
        raise UsdCliPhysicsError(f"Unsafe usd-cli physics output: {absolute}")
    marker = absolute.parent / ".workflow-owned-output"
    try:
        atomic_write_text(
            marker,
            "content-agent-workflows.physics\n",
            within=root,
        )
    except ValueError as exc:
        raise UsdCliPhysicsError(
            f"Unsafe usd-cli physics output parent: {absolute.parent}: {exc}"
        ) from exc
    if absolute.exists() or absolute.is_symlink():
        try:
            contained_regular_file(root, absolute)
        except ValueError as exc:
            raise UsdCliPhysicsError(
                f"usd-cli physics output target is unsafe: {absolute}: {exc}"
            ) from exc
    return absolute


def _require_ovrtx_probe(payload: dict[str, Any], project_dir: Path) -> None:
    try:
        validate_ovrtx_probe(payload)
    except RuntimeError as exc:
        raise UsdCliPhysicsError(str(exc)) from exc
    render = payload.get("render")
    assert isinstance(render, dict)
    render_path = render.get("path")
    if not isinstance(render_path, str) or not render_path:
        raise UsdCliPhysicsError("OVRTX probe did not return a render path.")
    try:
        safe_render = read_contained_artifact(
            project_dir,
            render_path,
            max_bytes=16 * 1024 * 1024,
            image=True,
        )
    except ValueError as exc:
        raise UsdCliPhysicsError(
            f"OVRTX probe render evidence is unsafe: {exc}"
        ) from exc
    if safe_render.size_bytes != render.get("size_bytes"):
        raise UsdCliPhysicsError(
            "OVRTX probe render size does not match its contained evidence"
        )


def _physics_material_scope_path(source_usd: Path) -> str:
    """Return the source stage's default-prim-owned Looks scope path."""

    from pxr import Usd

    source = source_usd.expanduser().resolve(strict=True)
    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadNone)
    if stage is None:
        raise UsdCliPhysicsError(f"Could not open physics source USD: {source}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise UsdCliPhysicsError(
            "Physics material authoring requires a valid stage default prim."
        )
    return default_prim.GetPath().AppendChild("Looks").pathString


def _reserved_physics_material_paths(
    source_usd: Path,
    scope_path: str,
) -> set[str]:
    """Return occupied direct children that are not reusable physics materials."""

    from pxr import Usd, UsdPhysics

    source = source_usd.expanduser().resolve(strict=True)
    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadNone)
    if stage is None:
        raise UsdCliPhysicsError(f"Could not open physics source USD: {source}")
    scope = stage.GetPrimAtPath(scope_path)
    if not scope.IsValid():
        return set()
    return {
        child.GetPath().pathString
        for child in scope.GetChildren()
        if not child.HasAPI(UsdPhysics.MaterialAPI)
    }


def _requires_physics_material(decisions: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(properties := decision.get("physical_properties"), dict)
        and any(
            properties.get(key) is not None
            for key in ("static_friction", "dynamic_friction", "restitution")
        )
        for decision in decisions
    )


def _deinstance_roots_for_patch(
    source_usd: Path,
    patch: dict[str, Any],
) -> list[str]:
    """Resolve the editable instance roots required by explicit patch targets."""

    from pxr import Sdf, Usd

    source = source_usd.expanduser().resolve(strict=True)
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise UsdCliPhysicsError(f"Could not open physics source USD: {source}")
    # Planning may need to de-instance an outer root temporarily so a nested
    # proxy becomes addressable. Keep those opinions in this stage's anonymous
    # session layer; opening a path can reuse the process-wide cached root layer,
    # which must remain immutable.
    stage.SetEditTarget(stage.GetSessionLayer())
    existing_target_paths = [
        str(operation["path"])
        for operation_name in ("rigid_bodies", "colliders")
        for operation in patch.get(operation_name, [])
        if isinstance(operation, dict) and isinstance(operation.get("path"), str)
    ]
    existing_target_paths.extend(
        str(operation["target_path"])
        for operation in patch.get("bindings", [])
        if isinstance(operation, dict) and isinstance(operation.get("target_path"), str)
    )
    definition_paths = [
        str(scene_path)
        for scene_path in patch.get("scene_paths", [])
        if isinstance(scene_path, str)
    ]
    definition_paths.extend(
        str(operation["path"])
        for operation in patch.get("materials", [])
        if isinstance(operation, dict) and isinstance(operation.get("path"), str)
    )

    def editable_ancestor(target_path: str, *, must_exist: bool) -> Any:
        prim = stage.GetPrimAtPath(target_path)
        if prim.IsValid() or must_exist:
            return prim
        ancestor_path = Sdf.Path(target_path).GetParentPath()
        while ancestor_path != Sdf.Path.emptyPath:
            prim = stage.GetPrimAtPath(ancestor_path)
            if prim.IsValid():
                return prim
            ancestor_path = ancestor_path.GetParentPath()
        return stage.GetPseudoRoot()

    roots: list[str] = []
    targets = list(
        dict.fromkeys(
            [(path, True) for path in existing_target_paths]
            + [(path, False) for path in definition_paths]
        )
    )
    for target_path, must_exist in targets:
        prim = editable_ancestor(target_path, must_exist=must_exist)
        while prim.IsValid() and (prim.IsInstanceProxy() or prim.IsInstance()):
            instance_root = prim if prim.IsInstance() else prim.GetParent()
            while instance_root.IsValid() and not instance_root.IsPseudoRoot():
                if instance_root.IsInstance() and not instance_root.IsInstanceProxy():
                    break
                instance_root = instance_root.GetParent()
            if (
                not instance_root.IsValid()
                or instance_root.IsPseudoRoot()
                or not instance_root.IsInstance()
                or instance_root.IsInstanceProxy()
            ):
                raise UsdCliPhysicsError(
                    f"Physics patch target has no editable instance root: {target_path}"
                )
            root_path = str(instance_root.GetPath())
            if root_path in roots:
                raise UsdCliPhysicsError(
                    "Physics patch target remained an instance proxy after planning "
                    f"de-instancing for {root_path}: {target_path}"
                )
            roots.append(root_path)
            instance_root.SetInstanceable(False)
            prim = editable_ancestor(target_path, must_exist=must_exist)
        if must_exist and not prim.IsValid():
            raise UsdCliPhysicsError(
                f"Physics patch target does not exist in the source stage: {target_path}"
            )
    return roots


def physics_patch_from_workflow_decisions(
    decisions: list[dict[str, Any]],
    *,
    author_rigid_body: bool,
    physics_scene_path: str,
    physics_material_scope_path: str | None = None,
    reserved_material_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Translate workflow decisions into explicit, policy-free usd-cli operations."""

    if not physics_scene_path.startswith("/") or physics_scene_path == "/":
        raise UsdCliPhysicsError(
            "The workflow must provide an absolute physics-scene prim path."
        )
    if physics_material_scope_path is not None and (
        not physics_material_scope_path.startswith("/")
        or physics_material_scope_path == "/"
        or physics_material_scope_path == "/Looks"
        or not physics_material_scope_path.endswith("/Looks")
    ):
        raise UsdCliPhysicsError(
            "The workflow must provide an absolute default-prim Looks scope path."
        )
    operations: dict[str, Any] = {
        "scene_paths": [physics_scene_path],
        "rigid_bodies": [],
        "colliders": [],
        "materials": [],
        "bindings": [],
    }
    used_material_paths: set[str] = set(reserved_material_paths or ())
    for index, decision in enumerate(decisions):
        body_root = decision.get("mass_authoring_path")
        collider_paths = decision.get("collider_paths")
        properties = decision.get("physical_properties")
        if not isinstance(body_root, str) or not body_root:
            raise UsdCliPhysicsError(
                f"Physics decision {index} has no mass_authoring_path."
            )
        if (
            not isinstance(collider_paths, list)
            or not collider_paths
            or not all(isinstance(path, str) and path for path in collider_paths)
        ):
            raise UsdCliPhysicsError(
                f"Physics decision {index} has invalid collider_paths."
            )
        if not isinstance(properties, dict):
            raise UsdCliPhysicsError(
                f"Physics decision {index} has no physical_properties."
            )
        if author_rigid_body and decision.get("component_role", "body") != "unowned_static":
            operations["rigid_bodies"].append(
                {
                    "path": body_root,
                    "density": properties.get("density"),
                    "mass": properties.get("estimated_mass_kg"),
                    **({"mass_properties": decision["mass_properties"]}
                       if decision.get("mass_properties") is not None else {}),
                }
            )
        operations["colliders"].extend(
            {
                "path": path,
                "approximation": decision.get("collision_approximation"),
            }
            for path in collider_paths
        )
        if any(
            properties.get(key) is not None
            for key in ("static_friction", "dynamic_friction", "restitution")
        ):
            if physics_material_scope_path is None:
                raise UsdCliPhysicsError(
                    "Physics-material decisions require the stage default-prim Looks "
                    "scope path."
                )
            component = str(
                decision.get("component_id")
                or decision.get("decision_id")
                or f"component_{index + 1}"
            )
            child_base = "Physics_" + re.sub(r"[^A-Za-z0-9_]", "_", component)
            child_name = child_base
            suffix = 2
            material_path = f"{physics_material_scope_path}/{child_name}"
            while material_path in used_material_paths:
                child_name = f"{child_base}_{suffix}"
                suffix += 1
                material_path = f"{physics_material_scope_path}/{child_name}"
            used_material_paths.add(material_path)
            operations["materials"].append(
                {
                    "path": material_path,
                    "static_friction": properties.get("static_friction"),
                    "dynamic_friction": properties.get("dynamic_friction"),
                    "restitution": properties.get("restitution"),
                }
            )
            operations["bindings"].append(
                {"target_path": body_root, "material_path": material_path}
            )
    return operations


def authored_physics_report(
    usd_cli_result: dict[str, Any],
) -> dict[str, Any]:
    """Normalize the authoritative ``physics.validate`` response for the workflow."""

    validation = usd_cli_result.get("structural_validation")
    if not isinstance(validation, dict):
        raise UsdCliPhysicsError(
            "usd-cli physics execution returned no structural validation response."
        )
    data = validation.get("data")
    checks = data.get("checks") if isinstance(data, dict) else None
    if not isinstance(checks, dict):
        raise UsdCliPhysicsError(
            "usd-cli physics validation returned no structural checks."
        )

    def count(name: str) -> int:
        value = checks.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise UsdCliPhysicsError(
                f"usd-cli physics validation returned an invalid {name!r} count."
            )
        return value

    return {
        "physics_usd": str(usd_cli_result.get("physics_usd") or ""),
        "physics_scene_count": count("scenes"),
        "rigid_body_count": count("rigid_bodies"),
        "enabled_rigid_body_count": count("enabled_rigid_bodies"),
        "collision_count": count("colliders"),
        "inspection_backend": "usd-cli",
        "inspection_command": "physics.validate",
    }


def apply_physics_patch(
    *,
    source_usd: Path,
    output_usd: Path,
    raw_dir: Path,
    workflow_decisions: list[dict[str, Any]],
    author_rigid_body: bool,
    physics_scene_path: str,
    usd_cli_session: WorkflowUsdCliSession | None = None,
    timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    """Apply an already-validated patch, structurally validate, and save a derivative."""

    source = source_usd.expanduser().resolve()
    raw_root = raw_dir.expanduser().resolve(strict=True)
    run_root = raw_root.parent
    output = _prepare_contained_output(
        run_root,
        output_usd.expanduser(),
    )
    if source == output:
        raise UsdCliPhysicsError(
            "usd-cli physics output must not overwrite the source."
        )
    source_digest_before = file_sha256(source)
    physics_material_scope_path = (
        _physics_material_scope_path(source)
        if _requires_physics_material(workflow_decisions)
        else None
    )
    patch = physics_patch_from_workflow_decisions(
        workflow_decisions,
        author_rigid_body=author_rigid_body,
        physics_scene_path=physics_scene_path,
        physics_material_scope_path=physics_material_scope_path,
        reserved_material_paths=(
            _reserved_physics_material_paths(source, physics_material_scope_path)
            if physics_material_scope_path is not None
            else None
        ),
    )
    patch_path = _write_json(
        raw_dir / "workflow_physics_patch.json",
        patch,
        within=raw_dir,
    )
    owned_session = (
        None
        if usd_cli_session is not None
        else _create_owned_session(
            run_root=run_root,
            purpose="apply",
            input_roots=(source,),
        )
    )
    session = usd_cli_session or owned_session
    assert session is not None
    project_dir = session.project_dir
    session_id = session.session_id
    records: list[dict[str, Any]] = []
    deinstance_roots: list[str] = []
    primary_error = False
    try:
        probe_response = _run_json(
            project_dir=project_dir,
            session_id=session_id,
            arguments=[
                "render-probe",
                "--require-engine",
                "ovrtx",
                "--output-dir",
                str(project_dir / "ovrtx_probe"),
            ],
            workflow_session=session,
            timeout_seconds=timeout_seconds,
        )
        _require_ovrtx_probe(probe_response, project_dir)
        records.append(probe_response)
        deinstance_roots = _deinstance_roots_for_patch(source, patch)
        # The finalizer authors from the digest-bound staged source as it
        # exists on disk. When the shared workflow session still holds unsaved
        # exploration edits on that file (the agentic child inspects without
        # saving), a plain `open` is refused by the same-session reload guard;
        # --force-reload discards those scratch edits deliberately.
        open_response = _run_json(
            project_dir=project_dir,
            session_id=session_id,
            arguments=["open", str(source), "--force-reload"],
            workflow_session=session,
            timeout_seconds=timeout_seconds,
        )
        records.append(open_response)
        records.append(
            _run_json(
                project_dir=project_dir,
                session_id=session_id,
                arguments=["checkpoint", "save", "workflow-open", "--full"],
                workflow_session=session,
                timeout_seconds=timeout_seconds,
            )
        )
        for instance_root in deinstance_roots:
            records.append(
                _run_json(
                    project_dir=project_dir,
                    session_id=session_id,
                    arguments=["set", instance_root, "instanceable", "false"],
                    workflow_session=session,
                    timeout_seconds=timeout_seconds,
                )
            )
        apply_response = _run_json(
            project_dir=project_dir,
            session_id=session_id,
            arguments=["physics", "apply", "-f", str(patch_path)],
            workflow_session=session,
            timeout_seconds=timeout_seconds,
        )
        records.append(apply_response)
        validation_response = _run_json(
            project_dir=project_dir,
            session_id=session_id,
            arguments=["physics", "validate"],
            allow_not_ok=True,
            workflow_session=session,
            timeout_seconds=timeout_seconds,
        )
        records.append(validation_response)
        records.append(
            _run_json(
                project_dir=project_dir,
                session_id=session_id,
                arguments=["checkpoint", "save", "workflow-applied", "--full"],
                workflow_session=session,
                timeout_seconds=timeout_seconds,
            )
        )
        save_response = _run_json(
            project_dir=project_dir,
            session_id=session_id,
            arguments=["save", str(output), "--flatten"],
            workflow_session=session,
            timeout_seconds=timeout_seconds,
        )
        records.append(save_response)
        try:
            contained_output = read_contained_artifact(run_root, output)
        except ValueError as exc:
            raise UsdCliPhysicsError(
                f"usd-cli did not create a safe physics derivative: {output}: {exc}"
            ) from exc
        if contained_output.size_bytes <= 0:
            raise UsdCliPhysicsError(
                f"usd-cli created an empty physics derivative: {contained_output.path}"
            )
        source_digest_after = file_sha256(source)
        if source_digest_after != source_digest_before:
            raise UsdCliPhysicsError(
                "usd-cli physics execution changed the immutable source USD."
            )
        command_record_path = _write_json(
            raw_dir / "usd_cli_physics_commands.json",
            {
                "schema_version": (
                    "content-agent-workflows.usd-cli-physics-commands.v1"
                ),
                "scene_backend": "usd-cli",
                "project_dir": str(project_dir),
                "session_id": session_id,
                "reused_workflow_session": usd_cli_session is not None,
                "source_usd": str(source),
                "output_usd": str(output),
                "source_sha256_before": source_digest_before,
                "source_sha256_after": source_digest_after,
                "ovrtx_probe": probe_response,
                "workflow_patch_path": str(patch_path),
                "physics_scene_path": physics_scene_path,
                "deinstanced_roots": deinstance_roots,
                "commands": records,
            },
            within=raw_dir,
        )
        return {
            "scene_backend": "usd-cli",
            "scene_tool_transport": "usd-cli-tel",
            "physics_usd": str(output),
            "low_level_patch_path": str(patch_path),
            "physics_scene_path": physics_scene_path,
            "deinstanced_roots": deinstance_roots,
            "command_record_path": str(command_record_path),
            "apply_response": apply_response,
            "structural_validation": validation_response,
            "save_response": save_response,
            "ovrtx_probe": probe_response,
            "source_sha256_before": source_digest_before,
            "source_sha256_after": source_digest_after,
            "project_dir": str(project_dir),
            "session_id": session_id,
            "reused_workflow_session": usd_cli_session is not None,
        }
    except Exception:
        primary_error = True
        raise
    finally:
        try:
            if owned_session is not None:
                owned_session.close()
        except RuntimeError as exc:
            if not primary_error:
                raise UsdCliPhysicsError(
                    f"Could not close the usd-cli physics session: {exc}"
                ) from exc


def simulate_physics_scene(
    *,
    scene_usd: Path,
    output_dir: Path,
    body_path: str,
    rest_position: list[float],
    world_up: list[float],
    duration_s: float,
    dt: float,
    sample_fps: int,
    body_pattern: str | None = None,
    usd_cli_session: WorkflowUsdCliSession | None = None,
    timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    """Run one workflow-authored scenario through neutral usd-cli simulation.

    Scenario construction, body selection, and result acceptance are deliberately
    caller-owned. This helper only transports the explicit inputs to usd-cli and
    returns its raw simulation facts and artifacts.
    """

    scene = scene_usd.expanduser().resolve(strict=True)
    output = output_dir.expanduser().absolute()
    if usd_cli_session is not None:
        run_root = usd_cli_session.project_dir.resolve(strict=True)
        try:
            output.relative_to(run_root)
        except ValueError as exc:
            raise UsdCliPhysicsError(
                f"usd-cli physics simulation output escapes the workflow run: {output}"
            ) from exc
    else:
        output.mkdir(parents=True, exist_ok=True)
        run_root = output.resolve(strict=True)
    atomic_write_text(
        output / ".workflow-owned",
        "content-agent-workflows.physics-simulation\n",
        within=run_root,
    )
    output = output.resolve(strict=True)
    owned_session = (
        None
        if usd_cli_session is not None
        else _create_owned_session(
            run_root=run_root,
            purpose="simulate",
            input_roots=(scene,),
        )
    )
    session = usd_cli_session or owned_session
    assert session is not None
    primary_error = False
    try:
        _run_json(
            project_dir=session.project_dir,
            session_id=session.session_id,
            arguments=["open", str(scene)],
            workflow_session=session,
            timeout_seconds=timeout_seconds,
        )
        payload = _run_json(
            project_dir=session.project_dir,
            session_id=session.session_id,
            arguments=[
                "physics",
                "simulate",
                "--scene",
                str(scene),
                "--body",
                body_path,
                *(["--body-pattern", body_pattern] if body_pattern is not None else []),
                "--rest-position",
                ",".join(str(float(value)) for value in rest_position),
                "--world-up",
                ",".join(str(float(value)) for value in world_up),
                "--engine",
                "ovphysx",
                "--duration",
                str(float(duration_s)),
                "--dt",
                str(float(dt)),
                "--fps",
                str(int(sample_fps)),
                "--output",
                str(output),
            ],
            workflow_session=session,
            timeout_seconds=timeout_seconds,
        )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise UsdCliPhysicsError(
                "usd-cli physics simulate returned no structured simulation data"
            )
        return dict(data)
    except Exception:
        primary_error = True
        raise
    finally:
        try:
            if owned_session is not None:
                owned_session.close()
        except RuntimeError as exc:
            if not primary_error:
                raise UsdCliPhysicsError(
                    f"Could not close the usd-cli physics session: {exc}"
                ) from exc


def render_physics_frames(
    *,
    recording_usd: Path,
    output_dir: Path,
    raw_dir: Path,
    resolution: str = "640x480",
    focus_prim_path: str | None = None,
    timeout_seconds: float = 1800.0,
    usd_cli_session: WorkflowUsdCliSession | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Render a simulation recording through the required OVRTX usd-cli backend."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    recording = recording_usd.expanduser().resolve()
    raw_root = raw_dir.expanduser().resolve(strict=True)
    run_root = raw_root.parent
    frames_dir = Path(os.path.abspath(output_dir.expanduser()))
    try:
        frames_dir.relative_to(run_root)
        atomic_write_text(
            frames_dir / ".workflow-owned",
            "content-agent-workflows.physics-render\n",
            within=run_root,
        )
    except ValueError as exc:
        raise UsdCliPhysicsError(
            f"Unsafe usd-cli physics render output: {frames_dir}: {exc}"
        ) from exc
    frames_dir = frames_dir.resolve(strict=True)
    owned_session = (
        None
        if usd_cli_session is not None
        else _create_owned_session(
            run_root=run_root,
            purpose="render",
            input_roots=(recording,),
        )
    )
    session = usd_cli_session or owned_session
    assert session is not None
    project_dir = session.project_dir
    session_id = session.session_id
    primary_error = False
    try:
        frame_spec = _representative_frame_spec(recording)
        _run_json(
            project_dir=project_dir,
            session_id=session_id,
            arguments=["open", str(recording)],
            workflow_session=session,
            timeout_seconds=timeout_seconds,
        )
        probe = _run_json(
            project_dir=project_dir,
            session_id=session_id,
            arguments=[
                "render-probe",
                "--require-engine",
                "ovrtx",
                "--output-dir",
                str(project_dir / "ovrtx_probe"),
            ],
            workflow_session=session,
            timeout_seconds=timeout_seconds,
        )
        _require_ovrtx_probe(probe, project_dir)
        response = _run_json(
            project_dir=project_dir,
            session_id=session_id,
            arguments=[
                "render-frames",
                "--scene",
                str(recording),
                "--res",
                resolution,
                "--mode",
                "fast",
                *(["--frames", frame_spec] if frame_spec is not None else []),
                *(["--focus", focus_prim_path] if focus_prim_path is not None else []),
                "--no-animate",
                "-o",
                str(frames_dir),
            ],
            workflow_session=session,
            timeout_seconds=timeout_seconds,
        )
        summary = response.get("summary")
        resolved_renderer = probe.get("resolved_renderer")
        if (
            not isinstance(summary, dict)
            or summary.get("backend") != resolved_renderer
            or resolved_renderer not in {"ovrtx", "remote"}
        ):
            raise UsdCliPhysicsError(
                "usd-cli render-frames did not preserve its probed OVRTX-backed "
                "renderer identity."
            )
        data = response.get("data")
        raw_paths = data.get("frame_paths") if isinstance(data, dict) else None
        if not isinstance(raw_paths, list) or not raw_paths:
            raise UsdCliPhysicsError("usd-cli render-frames returned no frame paths.")
        frame_paths: list[str] = []
        raw_renderer_identities = (
            data.get("renderer_identities") if isinstance(data, dict) else None
        )
        if not isinstance(raw_renderer_identities, list) or len(
            raw_renderer_identities
        ) != len(raw_paths):
            raise UsdCliPhysicsError(
                "usd-cli render-frames did not return aligned renderer identities."
            )
        renderer_identities: list[dict[str, Any]] = []
        if resolved_renderer == "remote":
            probe_backends = probe.get("backends")
            if not isinstance(probe_backends, list):
                raise UsdCliPhysicsError(
                    "Remote OVRTX probe did not return verified backend profiles."
                )
            verified_profiles = {
                (
                    (
                        item["url"].rstrip("/")
                        if isinstance(item.get("url"), str)
                        else item.get("url")
                    ),
                    item.get("engine"),
                    item.get("protocol_version"),
                    item.get("status"),
                )
                for item in probe_backends
                if isinstance(item, dict)
            }
            for identity in raw_renderer_identities:
                if not isinstance(identity, dict):
                    raise UsdCliPhysicsError(
                        "Remote OVRTX frame is missing its renderer identity."
                    )
                profile = (
                    (
                        identity["endpoint"].rstrip("/")
                        if isinstance(identity.get("endpoint"), str)
                        else identity.get("endpoint")
                    ),
                    identity.get("engine"),
                    identity.get("protocol_version"),
                    identity.get("status"),
                )
                if profile not in verified_profiles:
                    raise UsdCliPhysicsError(
                        "Remote OVRTX frame identity does not match a verified "
                        "probe backend."
                    )
                renderer_identities.append(dict(identity))
        else:
            renderer_identities = [
                {
                    "endpoint": "local",
                    "engine": "ovrtx",
                    "protocol_version": None,
                    "status": "ready",
                }
                for _path in raw_paths
            ]
        frames_root = frames_dir.resolve()
        for raw_path in raw_paths:
            if not isinstance(raw_path, str):
                raise UsdCliPhysicsError(
                    "usd-cli render-frames returned a non-string frame path."
                )
            try:
                frame = read_contained_artifact(
                    frames_root,
                    raw_path,
                    max_bytes=128 * 1024 * 1024,
                    image=True,
                )
            except ValueError as exc:
                raise UsdCliPhysicsError(
                    f"usd-cli render frame is unsafe: {raw_path}: {exc}"
                ) from exc
            if frame.size_bytes <= 0:
                raise UsdCliPhysicsError(
                    "usd-cli render frame is not a regular non-empty file: "
                    f"{frame.path}"
                )
            frame_paths.append(str(frame.path))
        record = {
            "schema_version": ("content-agent-workflows.usd-cli-physics-render.v1"),
            "scene_backend": "usd-cli",
            "renderer": "ovrtx",
            "resolved_renderer": resolved_renderer,
            "transport": probe.get("transport"),
            "project_dir": str(project_dir),
            "session_id": session_id,
            "reused_workflow_session": usd_cli_session is not None,
            "recording_usd": str(recording),
            "focus_prim_path": focus_prim_path,
            "frame_spec": frame_spec,
            "timeout_seconds": timeout_seconds,
            "probe": probe,
            "render_response": response,
            "frame_paths": frame_paths,
            "frame_renderer_identities": renderer_identities,
        }
        return frame_paths, record
    except Exception:
        primary_error = True
        raise
    finally:
        try:
            if owned_session is not None:
                owned_session.close()
        except RuntimeError as exc:
            if not primary_error:
                raise UsdCliPhysicsError(
                    f"Could not close the usd-cli physics session: {exc}"
                ) from exc
