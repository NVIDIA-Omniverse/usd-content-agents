# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply physics properties from predictions JSONL to a USD file."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils
from world_understanding.utils.usd.material import ensure_looks_scope
from world_understanding.utils.usd.package import (
    UsdzPackageError,
    extract_usdz_package_for_edit,
)

from physics_agent.functions.mass_scale_quality import (
    VALID_MASS_SCALE_POLICIES,
    has_mass_scale_suspicious_warning,
)
from physics_agent.functions.prediction_schema import unwrap_output_key_payload

logger = logging.getLogger(__name__)


class PhysicsAuthoringError(RuntimeError):
    """Raised when physics schemas cannot be authored safely."""


_USD_LAYER_EXTENSIONS = {".usd", ".usda", ".usdc"}
_USD_EXTENSIONS = _USD_LAYER_EXTENSIONS | {".usdz"}


def load_predictions(jsonl_path: str) -> list[dict]:
    """Load prediction JSONL, rejecting malformed records."""

    predictions = []
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise PhysicsAuthoringError(
                    f"Malformed JSON in predictions file {jsonl_path} at "
                    f"line {lineno}: {e}"
                ) from e
    return predictions


def _create_physics_material(
    stage: Usd.Stage,
    material_path: str,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
) -> UsdShade.Material:
    ensure_looks_scope(stage, material_path)
    material = UsdShade.Material.Define(stage, material_path)
    physics_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_mat.CreateStaticFrictionAttr(static_friction)
    physics_mat.CreateDynamicFrictionAttr(dynamic_friction)
    physics_mat.CreateRestitutionAttr(restitution)
    return material


def _remove_flattened_mass_attributes(
    flattened_layer: Sdf.Layer,
    prim_paths: set[str],
) -> None:
    """Remove mass specs for prims whose suspicious mass was skipped."""

    for prim_path in prim_paths:
        prim_spec = flattened_layer.GetPrimAtPath(prim_path)
        if prim_spec is None:
            continue
        mass_prop = prim_spec.properties.get("physics:mass")
        if mass_prop is not None:
            prim_spec.RemoveProperty(mass_prop)


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _is_bare_mdl_token(path_text: str) -> bool:
    token = path_text.strip("@")
    return (
        bool(token)
        and ":" not in token
        and "/" not in token
        and "\\" not in token
        and Path(token).suffix.lower() == ".mdl"
    )


def _copy_usdz_asset_for_flattened_output(
    asset_path: Sdf.AssetPath,
    extract_dir: Path,
    output: Path,
) -> Sdf.AssetPath:
    """Rewrite a flattened USDZ asset path to a portable sidecar path."""

    path_text = asset_path.path or ""
    resolved_text = asset_path.resolvedPath or ""
    candidates = [text for text in (path_text, resolved_text) if text]

    for candidate_text in candidates:
        candidate = Path(candidate_text)
        if candidate.is_absolute():
            if not _is_relative_to(candidate, extract_dir):
                continue
            rel = candidate.relative_to(extract_dir)
            source = candidate
        else:
            rel = candidate
            if ".." in rel.parts:
                continue
            source = extract_dir / rel

        if not source.is_file():
            continue

        assets_dir = output.parent / f"{output.stem}_assets"
        dest = assets_dir / rel
        extract_root = extract_dir.resolve()
        source_resolved = source.resolve()
        assets_root = assets_dir.resolve()
        dest_resolved = dest.resolve(strict=False)
        if not _is_relative_to(source_resolved, extract_root):
            continue
        if not _is_relative_to(dest_resolved, assets_root):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        return Sdf.AssetPath((Path(assets_dir.name) / rel).as_posix())

    # In Omniverse hosts, bare MDL tokens may resolve during flattening. Keep
    # those as runtime-resolved module names instead of baking host paths.
    if path_text and _is_bare_mdl_token(path_text):
        return Sdf.AssetPath(path_text.strip("@"))

    if candidates:
        logger.warning(
            "Leaving flattened USDZ asset path unchanged because it could not "
            "be resolved under the extracted package: path=%r resolvedPath=%r "
            "package_root=%s",
            path_text,
            resolved_text,
            extract_dir,
        )

    return asset_path


def _rewrite_flattened_usdz_asset_paths(
    flattened_layer: Sdf.Layer,
    extract_dir: Path,
    output: Path,
) -> None:
    """Copy USDZ asset dependencies beside output and rewrite asset paths."""

    def rewrite_value(value: object) -> object:
        if isinstance(value, Sdf.AssetPath):
            return _copy_usdz_asset_for_flattened_output(value, extract_dir, output)
        if isinstance(value, Sdf.AssetPathArray):
            return Sdf.AssetPathArray(
                [
                    _copy_usdz_asset_for_flattened_output(item, extract_dir, output)
                    for item in value
                ]
            )
        return value

    def rewrite_prim(prim_spec: Sdf.PrimSpec) -> None:
        for attr_spec in prim_spec.attributes.values():
            attr_spec.default = rewrite_value(attr_spec.default)
        for child in prim_spec.nameChildren:
            rewrite_prim(child)

    for root_prim in flattened_layer.rootPrims:
        rewrite_prim(root_prim)


def _api_enabled(attr: Usd.Attribute | None) -> bool:
    """Treat an applied physics API as enabled unless explicitly authored false."""

    value = attr.Get() if attr else None
    return value is not False


def _is_enabled_rigid_body(prim: Usd.Prim) -> bool:
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return False
    return _api_enabled(UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr())


def _has_enabled_rigid_body_descendant(prim: Usd.Prim) -> bool:
    for descendant in Usd.PrimRange(prim):
        if descendant == prim:
            continue
        if _is_enabled_rigid_body(descendant):
            return True
    return False


def _has_enabled_rigid_body_ancestor(prim: Usd.Prim) -> bool:
    parent = prim.GetParent()
    while parent and parent.IsValid() and not parent.IsPseudoRoot():
        if _is_enabled_rigid_body(parent):
            return True
        parent = parent.GetParent()
    return False


def _apply_collider_to_prim(
    stage: Usd.Stage,
    prim_path: str,
    physics_props: dict,
    collision_approx: str,
    materials_root: str,
    material_cache: dict[str, UsdShade.Material],
    *,
    preserve_existing_collider: bool = False,
) -> bool:
    """Author per-mesh collider schemas on a single prim."""

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        logger.warning("Prim not found: %s", prim_path)
        return False

    if prim.IsInstanceProxy():
        raise PhysicsAuthoringError(
            f"Prim {prim_path} is an instance proxy and cannot be authored on. "
            "Apply physics to a deinstanced USD instead, for example by "
            "running optimize_usd with enable_deinstance and using the raw "
            "predict output keyed to the optimized USD."
        )

    density = physics_props.get("density", 0.0)
    static_friction = physics_props.get("static_friction", 0.5)
    dynamic_friction = physics_props.get("dynamic_friction", 0.4)
    restitution = physics_props.get("restitution", 0.3)

    if preserve_existing_collider:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise PhysicsAuthoringError(
                f"Prim {prim_path} requested preserve_existing collision mode "
                "but has no existing CollisionAPI."
            )
        collision_enabled = (
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        )
        if collision_enabled is False:
            raise PhysicsAuthoringError(
                f"Prim {prim_path} requested preserve_existing collision mode "
                "but its existing CollisionAPI is disabled."
            )
    else:
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr(True)

    if not preserve_existing_collider and collision_approx != "none":
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr(collision_approx)

    if density > 0:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateDensityAttr(density)

    mat_key = f"sf{static_friction:.2f}_df{dynamic_friction:.2f}_r{restitution:.2f}"
    if mat_key not in material_cache:
        safe_name = mat_key.replace(".", "_").replace("-", "m")
        mat_path = f"{materials_root}/PhysMat_{safe_name}"
        material = _create_physics_material(
            stage, mat_path, static_friction, dynamic_friction, restitution
        )
        material_cache[mat_key] = material
        logger.info(
            "  Created physics material: %s (sf=%.2f df=%.2f r=%.2f)",
            mat_path,
            static_friction,
            dynamic_friction,
            restitution,
        )

    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.Bind(
        material_cache[mat_key],
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )

    logger.info(
        "  Applied collider to %s: density=%.0f friction=%.2f/%.2f restitution=%.2f",
        prim_path,
        density,
        static_friction,
        dynamic_friction,
        restitution,
    )

    return True


def _block_existing_mass(stage: Usd.Stage, prim_path: str) -> bool:
    """Block any pre-existing ``physics:mass`` on a prim."""

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    if not prim.HasAPI(UsdPhysics.MassAPI):
        mass_attr = prim.GetAttribute("physics:mass")
        if not mass_attr or not mass_attr.IsAuthored():
            return False
    mass_attr = UsdPhysics.MassAPI(prim).GetMassAttr()
    if mass_attr.HasAuthoredValueOpinion():
        if prim.IsInstanceProxy():
            raise PhysicsAuthoringError(
                f"Mass authoring prim {prim_path} is an instance proxy "
                "and cannot be authored on. Apply physics to a deinstanced USD."
            )
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
        root_layer = stage.GetRootLayer()
        has_weaker_mass = any(
            spec.layer != root_layer for spec in mass_attr.GetPropertyStack()
        )
        changed = False
        root_prim_spec = root_layer.GetPrimAtPath(prim_path)
        if root_prim_spec is not None:
            mass_prop = root_prim_spec.properties.get("physics:mass")
            if mass_prop is not None:
                root_prim_spec.RemoveProperty(mass_prop)
                changed = True
                mass_attr = UsdPhysics.MassAPI(prim).GetMassAttr()
        if has_weaker_mass:
            mass_attr.Block()
            changed = True
        return changed
    return False


def _apply_predictions_to_stage(
    stage: Usd.Stage,
    stage_path: Path,
    predictions: list[dict],
    collision_approx: str,
    output_key: str,
    mass_scale_policy: str,
    allow_empty_predictions: bool = False,
    author_rigid_body: bool = True,
) -> tuple[int, int, int, set[str], str]:
    """Author physics schemas into ``stage`` and return write statistics."""

    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise PhysicsAuthoringError(
            f"USD stage {stage_path} has no default prim. apply_physics "
            "authors the asset's RigidBodyAPI on the default prim; "
            "set defaultPrim in the source USD."
        )
    if author_rigid_body and not default_prim.IsA(UsdGeom.Xformable):
        raise PhysicsAuthoringError(
            f"default prim {default_prim.GetPath()} ({default_prim.GetTypeName()}) "
            "is not Xformable; RigidBodyAPI requires an Xformable parent. "
            "Set defaultPrim to an Xform that wraps the asset's geometry."
        )

    physics_scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if not physics_scenes:
        scene_path = default_prim.GetPath().AppendChild("PhysicsScene")
        physics_scene = UsdPhysics.Scene.Define(stage, scene_path)
        physics_scene.CreateGravityMagnitudeAttr(9.81)
        up_axis = UsdGeom.GetStageUpAxis(stage)
        if up_axis == UsdGeom.Tokens.y:
            physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, -1, 0))
        else:
            physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        logger.info("Created PhysicsScene at %s", scene_path)

    materials_root = str(default_prim.GetPath()) + "/Looks"

    material_cache: dict[str, UsdShade.Material] = {}
    skipped_mass_paths: set[str] = set()
    applied = 0
    skipped = 0
    aggregated_mass = 0.0
    explicit_body_masses: dict[str, float] = {}
    explicit_mass_keys: set[tuple[str, str]] = set()
    any_suspicious = False

    for pred in predictions:
        prim_path = pred.get("id", "")
        classification = unwrap_output_key_payload(pred.get(output_key, {}), output_key)
        if not isinstance(classification, dict):
            classification = {}
        physics_props = classification.get("physical_properties", {})
        suspicious_mass_scale = has_mass_scale_suspicious_warning(pred)

        if not prim_path:
            logger.warning("Prediction missing 'id', skipping")
            skipped += 1
            continue

        if not physics_props:
            logger.warning("No physical_properties for %s, skipping", prim_path)
            skipped += 1
            continue

        if suspicious_mass_scale:
            if mass_scale_policy == "fail":
                raise PhysicsAuthoringError(
                    f"Prediction for {prim_path} has mass/scale QA warning; "
                    "refusing to author physics because mass_scale_policy='fail'"
                )
            if mass_scale_policy == "warn":
                logger.warning(
                    "Prediction for %s has mass/scale QA warning; including "
                    "predicted mass in the asset aggregate because "
                    "mass_scale_policy='warn'",
                    prim_path,
                )
            any_suspicious = True

        record_collision_approx = classification.get("collision_approximation")
        effective_collision_approx = (
            record_collision_approx
            if isinstance(record_collision_approx, str) and record_collision_approx
            else collision_approx
        )
        if _apply_collider_to_prim(
            stage,
            prim_path,
            physics_props,
            effective_collision_approx,
            materials_root,
            material_cache,
            preserve_existing_collider=classification.get("collision_mode")
            == "preserve_existing",
        ):
            applied += 1
            mass = physics_props.get("estimated_mass_kg", 0.0)
            if mass > 0:
                aggregated_mass += mass
                mass_authoring_path = classification.get("mass_authoring_path")
                if (
                    isinstance(mass_authoring_path, str)
                    and mass_authoring_path
                    and author_rigid_body
                    and not (suspicious_mass_scale and mass_scale_policy == "skip_mass")
                ):
                    should_add_explicit_mass = True
                    component_key = classification.get("component_id") or (
                        classification.get("decision_id")
                    )
                    if isinstance(component_key, str) and component_key:
                        mass_key = (mass_authoring_path, component_key)
                        if mass_key in explicit_mass_keys:
                            should_add_explicit_mass = False
                        else:
                            explicit_mass_keys.add(mass_key)
                            component_mass = classification.get(
                                "component_estimated_mass_kg",
                                mass,
                            )
                            try:
                                component_mass_value = float(component_mass)
                            except (TypeError, ValueError):
                                pass
                            else:
                                if component_mass_value <= 0.0:
                                    should_add_explicit_mass = False
                                else:
                                    mass = component_mass_value
                    if should_add_explicit_mass:
                        explicit_body_masses[mass_authoring_path] = (
                            explicit_body_masses.get(mass_authoring_path, 0.0) + mass
                        )
            if suspicious_mass_scale and mass_scale_policy == "skip_mass":
                mass_paths = [prim_path]
                mass_authoring_path = classification.get("mass_authoring_path")
                if (
                    isinstance(mass_authoring_path, str)
                    and mass_authoring_path
                    and mass_authoring_path not in mass_paths
                ):
                    mass_paths.append(mass_authoring_path)
                for mass_path in mass_paths:
                    if _block_existing_mass(stage, mass_path):
                        skipped_mass_paths.add(mass_path)
                        logger.warning(
                            "  Cleared pre-existing mass on %s "
                            "(mass/scale QA warning, policy=skip_mass)",
                            mass_path,
                        )
        else:
            skipped += 1

    if predictions and applied == 0 and not allow_empty_predictions:
        raise PhysicsAuthoringError(
            f"No physics schemas were applied from {len(predictions)} prediction(s); "
            f"{skipped} were skipped. Check prediction prim paths, "
            f"output_key={output_key!r}, and whether the target USD contains "
            "authorable prims."
        )

    for body_path, body_mass in explicit_body_masses.items():
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim.IsValid():
            raise PhysicsAuthoringError(
                f"Explicit mass authoring prim not found: {body_path}"
            )
        if body_prim.IsInstanceProxy():
            raise PhysicsAuthoringError(
                f"Explicit mass authoring prim {body_path} is an instance proxy "
                "and cannot be authored on. Apply physics to a deinstanced USD."
            )
        if body_prim.IsInstanceable():
            body_prim.SetInstanceable(False)
        UsdPhysics.MassAPI.Apply(body_prim).CreateMassAttr(body_mass)
        logger.info(
            "Authored component mass on body %s: %.6f kg",
            body_path,
            body_mass,
        )

    preserves_existing_articulation = _has_enabled_rigid_body_descendant(
        default_prim
    ) or _has_enabled_rigid_body_ancestor(default_prim)
    if preserves_existing_articulation:
        logger.info(
            "Preserving existing rigid-body hierarchy around %s: "
            "skipping RigidBodyAPI on the default prim because another prim "
            "in the hierarchy already carries enabled RigidBodyAPI",
            default_prim.GetPath(),
        )
    elif not author_rigid_body:
        logger.info(
            "Skipping RigidBodyAPI authoring on %s because author_rigid_body=false",
            default_prim.GetPath(),
        )
    else:
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(default_prim)
        rigid_body.CreateRigidBodyEnabledAttr(True)
        body_mass_api = UsdPhysics.MassAPI.Apply(default_prim)

        skip_aggregate_mass = any_suspicious and mass_scale_policy == "skip_mass"
        if skip_aggregate_mass:
            body_path = str(default_prim.GetPath())
            existing = body_mass_api.GetMassAttr()
            if existing.HasAuthoredValueOpinion():
                existing.Block()
            skipped_mass_paths.add(body_path)
            logger.warning(
                "Skipping aggregate mass on body %s due to one or more "
                "mass/scale QA warnings; engine will derive mass from "
                "per-collider density × volume",
                body_path,
            )
        elif aggregated_mass > 0:
            body_path = str(default_prim.GetPath())
            body_mass = explicit_body_masses.get(body_path, aggregated_mass)
            body_mass_api.CreateMassAttr(body_mass)
            logger.info(
                "Authored aggregated mass on body %s: %.6f kg from %d collider(s)",
                default_prim.GetPath(),
                body_mass,
                applied,
            )

    return (
        applied,
        skipped,
        len(material_cache),
        skipped_mass_paths,
        str(default_prim.GetPath()),
    )


def _export_flattened_stage(
    stage: Usd.Stage,
    output: Path,
    skipped_mass_paths: set[str],
    package_asset_root: Path | None = None,
) -> None:
    # USDZ inputs are flattened from an extracted package. When
    # package_asset_root is provided, package-local assets are copied beside
    # the output and rewritten to portable sidecar references.
    #
    # FIXME(apply_physics-flatten-anchor): the non-package path can still
    # inherit absolute asset paths from stage.Flatten() when dependencies live
    # outside the output directory. For direct CLI working-dir use this points
    # at the user's own checkout, but a redistributable single-file export
    # would need the same copy-and-rewrite treatment or a USDZ package output.
    #
    # Alternative: skip Flatten() entirely for single-layer inputs and
    # use a sublayer/reference structure where the original input
    # layer carries the asset paths (USD resolves them relative to the
    # layer they're authored in, which would be the original input
    # location). That preserves the source's anchor at the cost of
    # losing the single-file output guarantee callers may depend on.
    flattened_layer = stage.Flatten()
    _remove_flattened_mass_attributes(flattened_layer, skipped_mass_paths)
    if package_asset_root is not None:
        _rewrite_flattened_usdz_asset_paths(flattened_layer, package_asset_root, output)
    flattened_layer.Export(str(output))


def _extract_usdz_for_edit(usdz_path: Path, extract_dir: Path) -> Path:
    try:
        return extract_usdz_package_for_edit(usdz_path, extract_dir)
    except UsdzPackageError as exc:
        raise RuntimeError(str(exc)) from exc


def _create_usdz_package(root_layer_path: Path, output: Path) -> None:
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}_",
        suffix=output.suffix,
        dir=output.parent,
        delete=False,
    ) as temp_file:
        temp_output = Path(temp_file.name)
    temp_output.unlink(missing_ok=True)
    try:
        ok = UsdUtils.CreateNewUsdzPackage(str(root_layer_path), str(temp_output))
        if not ok or not temp_output.exists():
            raise RuntimeError(f"Failed to create USDZ package: {output}")
        if not zipfile.is_zipfile(temp_output):
            raise RuntimeError(
                f"CreateNewUsdzPackage wrote a non-ZIP file: {temp_output}"
            )
        temp_output.replace(output)
    finally:
        temp_output.unlink(missing_ok=True)


def _save_stage(stage: Usd.Stage) -> None:
    if not stage.GetRootLayer().Save():
        raise RuntimeError(
            f"Failed to save USD layer: {stage.GetRootLayer().identifier}"
        )


def _open_stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {path}")
    return stage


def _open_and_apply(
    path: Path,
    predictions: list[dict],
    collision_approx: str,
    output_key: str,
    mass_scale_policy: str,
    allow_empty_predictions: bool = False,
    author_rigid_body: bool = True,
) -> tuple[Usd.Stage, int, int, int, set[str], str]:
    stage = _open_stage(path)
    applied, skipped, material_count, skipped_mass_paths, body_path = (
        _apply_predictions_to_stage(
            stage,
            path,
            predictions,
            collision_approx,
            output_key,
            mass_scale_policy,
            allow_empty_predictions=allow_empty_predictions,
            author_rigid_body=author_rigid_body,
        )
    )
    return stage, applied, skipped, material_count, skipped_mass_paths, body_path


def apply_physics(
    usd_path: str,
    predictions_path: str,
    output_path: str,
    collision_approx: str = "convexHull",
    output_key: str = "classification",
    mass_scale_policy: str = "skip_mass",
    allow_empty_predictions: bool = False,
    author_rigid_body: bool = True,
) -> str:
    """Apply physics properties from predictions to a USD file.

    Reads a predictions JSONL file and authors per-mesh colliders on each
    predicted prim, plus one rigid body on the asset's default prim when
    ``author_rigid_body`` is true. USDZ inputs with USDZ outputs preserve
    package structure by editing the extracted root layer and repackaging bundled
    dependencies. USDZ inputs with layer outputs flatten composed geometry while
    copying package-local asset dependencies beside the output and rewriting them
    to relative paths.

    Args:
        usd_path: Path to input USD file. Must have a default prim that
            is ``UsdGeom.Xformable``.
        predictions_path: Path to predictions JSONL from physics-agent predict step.
        output_path: Path for the output USD file.
        collision_approx: Collision approximation — "convexHull", "convexDecomposition",
            "boundingCube", "boundingSphere", "meshSimplification", or "none".
        output_key: Key under which the VLM classification dict is stored in
            each prediction entry. Must match `predict.output_key` from the
            upstream step (defaults to "classification").
        mass_scale_policy: Asset-scoped handling for predictions carrying
            ``quality_warnings[].code == "mass_scale_suspicious"``:
            ``"skip_mass"`` blocks the body's aggregate mass attribute when
            **any** prediction is flagged (engine derives mass from
            per-collider density × volume); ``"warn"`` logs and includes
            the suspicious mass in the aggregate; ``"fail"`` raises before
            writing output.
        allow_empty_predictions: When ``False`` (default), reject an empty
            predictions file instead of authoring a rigid body with no colliders.
        author_rigid_body: When ``False``, author per-target collider/material
            data but do not add a default-prim ``RigidBodyAPI`` or body mass.

    Returns:
        Absolute path to the created USD file.
    """
    if mass_scale_policy not in VALID_MASS_SCALE_POLICIES:
        raise ValueError(
            "mass_scale_policy must be one of "
            f"{sorted(VALID_MASS_SCALE_POLICIES)}, got {mass_scale_policy!r}"
        )
    if not isinstance(allow_empty_predictions, bool):
        raise ValueError(
            "allow_empty_predictions must be a boolean, got "
            f"{type(allow_empty_predictions).__name__}"
        )
    if not isinstance(author_rigid_body, bool):
        raise ValueError(
            "author_rigid_body must be a boolean, got "
            f"{type(author_rigid_body).__name__}"
        )

    source = Path(usd_path).resolve()
    output = Path(output_path).resolve()
    if source == output:
        raise ValueError("apply_physics output_path must differ from usd_path")

    input_suffix = source.suffix.lower()
    output_suffix = output.suffix.lower()
    if input_suffix not in _USD_EXTENSIONS:
        raise ValueError(f"Unsupported input USD extension: {input_suffix}")
    if output_suffix not in _USD_EXTENSIONS:
        raise ValueError(f"Unsupported output USD extension: {output_suffix}")

    predictions = load_predictions(predictions_path)
    logger.info("Loaded %d predictions from %s", len(predictions), predictions_path)
    if not predictions and not allow_empty_predictions:
        raise PhysicsAuthoringError(
            "No predictions were loaded from "
            f"{predictions_path}; refusing to author physics with zero colliders. "
            "Set allow_empty_predictions=true only for workflows that intentionally "
            "permit empty physics authoring."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    output_preexisted = output.exists()
    body_path = ""
    try:
        if output_suffix == ".usdz":
            with tempfile.TemporaryDirectory(prefix="physics_usdz_") as temp_root:
                temp_dir = Path(temp_root)
                if input_suffix == ".usdz":
                    editable_root = _extract_usdz_for_edit(source, temp_dir)
                    (
                        stage,
                        applied,
                        skipped,
                        material_count,
                        _,
                        body_path,
                    ) = _open_and_apply(
                        editable_root,
                        predictions,
                        collision_approx,
                        output_key,
                        mass_scale_policy,
                        allow_empty_predictions=allow_empty_predictions,
                        author_rigid_body=author_rigid_body,
                    )
                    _save_stage(stage)
                    _create_usdz_package(editable_root, output)
                else:
                    logger.warning(
                        "Flattening non-USDZ input %s before packaging %s as USDZ "
                        "because relative USD dependency trees cannot be returned "
                        "alongside a package without rewriting asset paths",
                        source,
                        output,
                    )
                    (
                        stage,
                        applied,
                        skipped,
                        material_count,
                        skipped_mass_paths,
                        body_path,
                    ) = _open_and_apply(
                        source,
                        predictions,
                        collision_approx,
                        output_key,
                        mass_scale_policy,
                        allow_empty_predictions=allow_empty_predictions,
                        author_rigid_body=author_rigid_body,
                    )
                    flattened_root = temp_dir / f"{output.stem}.usda"
                    _export_flattened_stage(stage, flattened_root, skipped_mass_paths)
                    _create_usdz_package(flattened_root, output)

        elif input_suffix == ".usdz":
            with tempfile.TemporaryDirectory(prefix="physics_usdz_") as temp_root:
                temp_dir = Path(temp_root)
                editable_root = _extract_usdz_for_edit(source, temp_dir)
                (
                    stage,
                    applied,
                    skipped,
                    material_count,
                    skipped_mass_paths,
                    body_path,
                ) = _open_and_apply(
                    editable_root,
                    predictions,
                    collision_approx,
                    output_key,
                    mass_scale_policy,
                    allow_empty_predictions=allow_empty_predictions,
                    author_rigid_body=author_rigid_body,
                )
                _export_flattened_stage(
                    stage,
                    output,
                    skipped_mass_paths,
                    package_asset_root=temp_dir,
                )

        elif input_suffix == output_suffix and output.parent == source.parent:
            with tempfile.NamedTemporaryFile(
                prefix=f".{output.stem}_",
                suffix=output.suffix,
                dir=output.parent,
                delete=False,
            ) as temp_file:
                temp_output = Path(temp_file.name)
            temp_output.unlink(missing_ok=True)
            try:
                shutil.copy2(source, temp_output)
                stage, applied, skipped, material_count, _, body_path = _open_and_apply(
                    temp_output,
                    predictions,
                    collision_approx,
                    output_key,
                    mass_scale_policy,
                    allow_empty_predictions=allow_empty_predictions,
                    author_rigid_body=author_rigid_body,
                )
                _save_stage(stage)
                temp_output.replace(output)
            finally:
                temp_output.unlink(missing_ok=True)

        else:
            stage, applied, skipped, material_count, skipped_mass_paths, body_path = (
                _open_and_apply(
                    source,
                    predictions,
                    collision_approx,
                    output_key,
                    mass_scale_policy,
                    allow_empty_predictions=allow_empty_predictions,
                    author_rigid_body=author_rigid_body,
                )
            )
            _export_flattened_stage(stage, output, skipped_mass_paths)

    except Exception:
        if not output_preexisted:
            output.unlink(missing_ok=True)
        raise

    logger.info(
        "Saved %s: %s on %s with %d collider(s), %d skipped, %d materials created",
        output,
        "1 rigid body" if author_rigid_body else "no authored rigid body",
        body_path,
        applied,
        skipped,
        material_count,
    )
    return str(output)
