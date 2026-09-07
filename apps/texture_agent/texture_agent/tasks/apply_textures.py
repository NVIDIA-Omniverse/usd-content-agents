# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: Apply PBR textures to materials in USD.

Supports per-material mode (shared texture) and per-prim mode (unique
texture per geometry prim via material cloning).
"""

from __future__ import annotations

import filecmp
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pxr import Ar, Gf, Pcp, Sdf, Usd, UsdGeom, UsdShade, UsdUtils
from world_understanding.agentic.tasks import Task
from world_understanding.utils.usd.package import (
    UsdzPackageError,
    extract_usdz_package_for_edit,
)

from texture_agent.functions.artifact_manifest import (
    validate_output_texture_portability,
)
from texture_agent.functions.cached_apply import (
    allows_non_executable_cached_apply_plan,
    is_cached_apply_context,
    is_valid_cached_texture_png,
)
from texture_agent.functions.material_discovery import MaterialInfo, PrimTextureUnit
from texture_agent.planning import TexturePlanUnit
from texture_agent.planning.contracts import validate_texture_plan_payload
from texture_agent.tasks.blend_textures import BlendedTextures

logger = logging.getLogger(__name__)


def _cached_blended_texture_set(out_dir: Path, key: str) -> BlendedTextures | None:
    albedo = out_dir / f"{key}_albedo.png"
    normal = out_dir / f"{key}_normal.png"
    orm = out_dir / f"{key}_orm.png"
    if albedo.exists() and normal.exists() and orm.exists():
        return BlendedTextures(albedo=str(albedo), normal=str(normal), orm=str(orm))
    return None


def _load_cached_blended_textures(
    working_dir: Path,
    units: list[PrimTextureUnit],
) -> dict[str, BlendedTextures]:
    out_dir = working_dir / "textures"
    cached: dict[str, BlendedTextures] = {}
    for unit in units:
        textures = _cached_blended_texture_set(out_dir, unit.key)
        if textures:
            cached[unit.key] = textures
    return cached


def _missing_cached_blended_artifacts(
    blended: dict[str, BlendedTextures],
    units: list[PrimTextureUnit],
) -> list[str]:
    """Return missing unit/channel labels for an all-or-nothing cached apply."""
    missing: list[str] = []
    for unit in units:
        textures = blended.get(unit.key)
        if not isinstance(textures, BlendedTextures):
            missing.extend(
                f"{unit.key}:{channel}" for channel in ("albedo", "normal", "orm")
            )
            continue
        for channel in ("albedo", "normal", "orm"):
            value = getattr(textures, channel)
            if not value or not is_valid_cached_texture_png(value):
                missing.append(f"{unit.key}:{channel}")
    return list(dict.fromkeys(missing))


def _accepted_plan_units_by_id(
    context: dict[str, Any],
) -> dict[str, TexturePlanUnit] | None:
    """Validate one accepted plan once and index its immutable units."""

    raw_plan = context.get("texture_plan")
    if raw_plan is None:
        return None
    plan = validate_texture_plan_payload(raw_plan)

    if allows_non_executable_cached_apply_plan(context):
        # The service may hydrate a pre-plan display-key cache with a validated
        # compatibility plan for scope recovery. It is explicitly non-executable,
        # so it cannot supply immutable plan IDs to the legacy cached apply path.
        return None

    units_by_id = {item.unit_id: item for item in plan.selected_units}
    if len(units_by_id) != len(plan.selected_units):
        raise RuntimeError("Texture apply plan contains duplicate unit IDs")
    return units_by_id


def _accepted_plan_unit_for_runtime_unit(
    plan_units_by_id: dict[str, TexturePlanUnit] | None,
    unit: PrimTextureUnit,
) -> TexturePlanUnit | None:
    """Return the exact immutable plan unit for a stable runtime unit key."""

    if plan_units_by_id is None:
        return None
    selected = plan_units_by_id.get(unit.key)
    if selected is None:
        raise RuntimeError(
            f"Texture apply runtime unit {unit.key} does not map to one plan unit"
        )
    return selected


def _planned_binding_target_paths(
    plan_unit: TexturePlanUnit | None,
) -> tuple[str, ...]:
    """Return every exact prim or subset binding target in plan order."""

    if plan_unit is None:
        return ()
    return tuple(
        dict.fromkeys((*plan_unit.member_prim_paths, *plan_unit.member_subset_paths))
    )


def _material_clone_source_path(stage: Usd.Stage, mat: MaterialInfo) -> str:
    """Resolve one composed material to an authorable source namespace.

    Discovery canonicalizes materials below instances to prototype paths. Those
    paths cannot be authored, but an exact composed alias may resolve through a
    unique internal reference to the ordinary source prim. Prefer that source
    without de-instancing; fail closed when no material namespace can safely
    own the clone.
    """

    for candidate_path in dict.fromkeys((mat.prim_path, *mat.material_alias_paths)):
        candidate = stage.GetPrimAtPath(candidate_path)
        if candidate.IsInstanceProxy():
            candidate = _instance_source_prim(stage, candidate) or Usd.Prim()
        if (
            not candidate.IsValid()
            or candidate.IsInstanceProxy()
            or candidate.IsPrototype()
            or candidate.IsInPrototype()
            or not candidate.IsA(UsdShade.Material)
        ):
            continue
        return str(candidate.GetPath())
    raise RuntimeError(
        f"Cannot clone composed material without an authorable source: {mat.prim_path}"
    )


def _clone_material(
    stage: Usd.Stage,
    source_mat_path: str,
    clone_name: str,
) -> str:
    """Clone a material prim (deep copy of entire shader subtree).

    Args:
        stage: The USD stage.
        source_mat_path: Path to the source material prim.
        clone_name: Name for the cloned material.

    Returns:
        Path to the cloned material prim.
    """
    _require_nonvariant_edit_subtree(
        stage.GetPrimAtPath(source_mat_path),
        purpose="material cloning",
    )
    parent_path = str(Sdf.Path(source_mat_path).GetParentPath())
    clone_path = f"{parent_path}/{clone_name}"
    _require_nonvariant_edit(
        stage.GetPrimAtPath(parent_path),
        purpose="material clone parent editing",
    )
    _require_nonvariant_edit_subtree(
        stage.GetPrimAtPath(clone_path),
        purpose="material clone replacement",
    )

    layer = stage.GetRootLayer()
    flattened = stage.Flatten()
    if flattened.GetPrimAtPath(source_mat_path) is None:
        raise RuntimeError(
            f"Cannot clone material without a composed material spec: {source_mat_path}"
        )
    # A composed source can live wholly in a sublayer. Author only the missing
    # parent overs in the editable root layer, while copying the fully composed
    # material subtree from a flattened snapshot into the exact clone path.
    parent_spec = layer.GetPrimAtPath(parent_path)
    if parent_spec is None:
        parent_spec = Sdf.CreatePrimInLayer(layer, parent_path)
        parent_spec.specifier = Sdf.SpecifierOver
    try:
        copied = Sdf.CopySpec(flattened, source_mat_path, layer, clone_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to clone material {source_mat_path} to {clone_path}"
        ) from exc
    if not copied:
        raise RuntimeError(
            f"Failed to clone material {source_mat_path} to {clone_path}"
        )
    clone_prim = stage.GetPrimAtPath(clone_path)
    if not clone_prim.IsValid() or not clone_prim.IsA(UsdShade.Material):
        raise RuntimeError(f"Material clone is missing or invalid: {clone_path}")

    logger.debug("Cloned material: %s -> %s", source_mat_path, clone_path)
    return clone_path


def _bind_material_or_raise(
    stage: Usd.Stage,
    target_prim: Usd.Prim,
    material_path: str,
) -> None:
    """Bind one material and fail closed when USD rejects the authoring call."""

    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid() or not material_prim.IsA(UsdShade.Material):
        raise RuntimeError(f"Material binding target is invalid: {material_path}")
    try:
        bound = UsdShade.MaterialBindingAPI.Apply(target_prim).Bind(
            UsdShade.Material(material_prim)
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to bind material {material_path} to {target_prim.GetPath()}"
        ) from exc
    if not bound:
        raise RuntimeError(
            f"Failed to bind material {material_path} to {target_prim.GetPath()}"
        )


def _set_texture_attr(
    prim: Usd.Prim,
    attr_name: str,
    texture_path: str,
) -> None:
    """Set an asset path attribute on a prim, creating if needed."""
    attr = prim.GetAttribute(attr_name)
    if attr and attr.IsValid():
        attr.Set(Sdf.AssetPath(texture_path))
    else:
        prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(texture_path)
        )


def _prim_crosses_variant_arc(prim: Usd.Prim) -> bool:
    """Return whether a root-layer edit could cross a variant boundary."""
    if not prim.IsValid():
        return False
    if any(
        arc.GetArcType() == Pcp.ArcTypeVariant
        for arc in Usd.PrimCompositionQuery(prim).GetCompositionArcs()
    ):
        return True

    # A dormant variant selection does not expose an ArcTypeVariant in the
    # composition query. Still reject edits below any prim that owns variants:
    # a root-layer opinion at the same path would silently mask that variant
    # when a selection is made later.
    cursor = prim
    while cursor.IsValid() and not cursor.IsPseudoRoot():
        if cursor.GetVariantSets().GetNames():
            return True
        cursor = cursor.GetParent()
    return False


def _require_nonvariant_edit(prim: Usd.Prim, *, purpose: str) -> None:
    """Reject one root-layer write that would leak across variants."""
    if not prim.IsValid():
        return
    if _prim_crosses_variant_arc(prim):
        raise RuntimeError(
            f"{purpose} crosses a variant arc and cannot be authored at "
            "the composed path without leaking between variants: "
            f"{prim.GetPath()}"
        )


def _require_nonvariant_edit_subtree(prim: Usd.Prim, *, purpose: str) -> None:
    """Reject subtree writes that would leak across variant selections."""
    if not prim.IsValid():
        return
    for candidate in Usd.PrimRange(prim):
        _require_nonvariant_edit(candidate, purpose=purpose)


def _instance_source_prim(stage: Usd.Stage, prim: Usd.Prim) -> Usd.Prim | None:
    """Return the authored prim an instance proxy resolves to, if it is unique.

    An instance proxy cannot be authored to, and neither can the prim it maps to
    via ``GetPrimInPrototype`` — USD refuses edits to a prototype. What *is*
    authorable is the prim the instance references, which for a CAD conversion is
    an ordinary namespace such as ``/Body/Prototypes/Body/...``.

    ``None`` is returned when more than one instance shares the source. Authoring
    there would silently apply one instance's textures to all of them, so the
    caller keeps the unauthorable prim and the edit is reported rather than
    applied to the wrong geometry.
    """
    cursor = prim
    while cursor.IsValid() and not cursor.IsInstance():
        cursor = cursor.GetParent()
    if not cursor.IsValid():
        return None

    # Instances sharing a prototype share every edit made through the source, so
    # count them from the prototype rather than by scanning the stage.
    prototype = cursor.GetPrototype()
    instance_count = len(prototype.GetInstances()) if prototype else 0
    if instance_count > 1:
        logger.warning(
            "Not authoring %s through its source: %d instances share it",
            prim.GetPath(),
            instance_count,
        )
        return None

    relative = prim.GetPath().MakeRelativePath(cursor.GetPath())
    root_layer = stage.GetRootLayer()
    query = Usd.PrimCompositionQuery.GetDirectReferences(cursor)
    for arc in query.GetCompositionArcs():
        node = arc.GetTargetNode()
        # Only an internal reference has its source on this stage. An external
        # reference's target path names a prim in the referenced layer, and a
        # same-named prim here would be an unrelated prim that must not be
        # textured in its place.
        if node.layerStack.identifier.rootLayer != root_layer:
            continue
        candidate = stage.GetPrimAtPath(node.path.AppendPath(relative))
        if not candidate.IsValid() or candidate.IsInstanceProxy():
            continue
        return candidate
    return None


def _deinstance_prim(
    stage: Usd.Stage,
    prim: Usd.Prim,
    deinstanced_paths: set[str] | None,
) -> None:
    """Deinstance ``prim`` and record only a successfully authored change."""
    if prim.IsInstanceProxy():
        return
    _require_nonvariant_edit(prim, purpose="Deinstancing")
    path = str(prim.GetPath())
    was_instanceable = prim.IsInstanceable()
    prim.SetInstanceable(False)
    refreshed = stage.GetPrimAtPath(path)
    if (
        deinstanced_paths is not None
        and was_instanceable
        and refreshed.IsValid()
        and not refreshed.IsInstanceable()
    ):
        deinstanced_paths.add(path)


def _editable_prim_for_path(
    stage: Usd.Stage,
    prim_path: str,
    *,
    deinstanced_paths: set[str] | None = None,
) -> Usd.Prim:
    """Return a prim that can accept authored properties at ``prim_path``.

    Prefer the authored source an instance references. ``SetInstanceable(False)``
    was previously called unconditionally, which expands the prototype's
    namespace on the stage that is later exported to packaging, turning one
    composed path into two over a single source spec — the shape the packaging
    guard refuses. See issue #916.

    That resolution only works when the source is reachable on this stage, which
    holds for an internal reference. An external reference composes its source
    from another layer, where there is no prim to author, so de-instancing
    remains the fallback for that case and for a shared source. The hazard is
    narrowed to the cases that need it rather than applied to every asset.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return prim
    if not (prim.IsInstanceProxy() or prim.IsInstance() or prim.IsInstanceable()):
        return prim

    # Only a proxy sits over a prototype, so only a proxy can produce the dual
    # composed path that breaks packaging. An instanceable prim with no
    # reference has no prototype at all and is handled as before.
    if prim.IsInstanceProxy():
        source = _instance_source_prim(stage, prim)
        if source is not None:
            return source

    cursor = prim
    while cursor.IsValid() and not cursor.GetPath().IsAbsoluteRootPath():
        if cursor.IsInstance() or cursor.IsInstanceable():
            logger.debug(
                "De-instancing %s to author %s: no authorable source on this stage",
                cursor.GetPath(),
                prim_path,
            )
            _deinstance_prim(stage, cursor, deinstanced_paths)
            break
        cursor = cursor.GetParent()
    return stage.GetPrimAtPath(prim_path)


def _can_define_parent_scope(stage: Usd.Stage, parent: Usd.Prim) -> bool:
    """Return whether the material parent can be safely authored as a Scope.

    Scope promotion is limited to undefined or typeless material containers.
    Existing typed parents such as Xform or Mesh are preserved because
    Scope.Define would retype the prim and change asset semantics.
    """
    if not parent.IsValid():
        return False

    parent_path = parent.GetPath()
    if parent_path.IsAbsoluteRootPath():
        return False

    if parent.IsInstanceProxy() or parent.IsPrototype() or parent.IsInPrototype():
        return False

    edit_layer = stage.GetEditTarget().GetLayer()
    if edit_layer is not None and not edit_layer.permissionToEdit:
        return False

    if parent.IsDefined():
        if parent.IsA(UsdGeom.Scope):
            return False
        # Do not retype authored Xform, Mesh, or other typed parents. Only
        # promote typeless material containers that already exist as defs.
        return str(parent.GetTypeName()) == ""

    if parent.GetSpecifier() == Sdf.SpecifierOver:
        return False

    return True


def _set_tiledimage_file_input(
    stage: Usd.Stage,
    mat_path: str,
    shader_name: str,
    texture_path: str,
) -> None:
    """Set the concrete tiledimage shader input used by NVCF/OpenPBR."""
    shader_prim = stage.GetPrimAtPath(f"{mat_path}/{shader_name}")
    if not shader_prim.IsValid():
        logger.debug(
            "OpenPBR tiledimage shader not found: %s/%s", mat_path, shader_name
        )
        return

    if not shader_prim.IsA(UsdShade.Shader):
        logger.debug("Prim is not a UsdShade shader: %s", shader_prim.GetPath())
        return

    shader = UsdShade.Shader(shader_prim)
    file_input = shader.GetInput("file")
    if file_input:
        file_input.Set(Sdf.AssetPath(texture_path))
    else:
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(texture_path)
        )


def _author_texture_reference(
    texture_path: str,
    output_usd_path: Path,
    bundle_root: Path | None = None,
) -> str:
    """Return the texture path to author into the output USD layer."""
    if not texture_path or _is_unbundleable_asset_path(texture_path):
        return texture_path
    path = Path(texture_path)
    if not path.is_absolute():
        return texture_path.replace("\\", "/")
    try:
        allowed_root = bundle_root or output_usd_path.parent.parent
        path.resolve().relative_to(allowed_root.resolve())
        return Path(os.path.relpath(path, output_usd_path.parent)).as_posix()
    except (OSError, ValueError):
        return texture_path


def _is_portable_texture_reference(
    raw: str,
    output_usd_path: Path,
    bundle_root: Path | None = None,
) -> bool:
    """Return whether an authored texture ref already resolves inside the run."""
    if not raw or _is_unbundleable_asset_path(raw) or Path(raw).is_absolute():
        return False
    try:
        resolved = (output_usd_path.parent / raw).resolve()
        allowed_root = bundle_root or output_usd_path.parent.parent
        resolved.relative_to(allowed_root.resolve())
    except (OSError, ValueError):
        return False
    return resolved.is_file()


def _canonical_texture_source_root(
    root: Path,
    *,
    label: str,
    require_directory: bool = False,
) -> Path:
    """Resolve one approved source root without ever widening trust to ``/``."""
    lexical = Path(os.path.abspath(root))
    try:
        resolved = lexical.resolve(strict=require_directory)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not an existing directory: {lexical}") from exc
    if resolved == Path(resolved.anchor):
        raise RuntimeError(f"{label} must not be the filesystem root: {lexical}")
    if require_directory and not resolved.is_dir():
        raise RuntimeError(f"{label} is not an existing directory: {lexical}")
    return resolved


def _validated_dependency_root(context: dict[str, Any]) -> Path | None:
    """Return the lexical trusted bundle root after strict validation."""
    value = context.get("usd_dependency_root")
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str | os.PathLike):
        raise RuntimeError("Texture dependency root must be a filesystem path")
    lexical = Path(os.path.abspath(Path(value)))
    _canonical_texture_source_root(
        lexical,
        label="Texture dependency root",
        require_directory=True,
    )
    return lexical


def _validate_authoritative_source_root(
    source_path: Path,
    dependency_root: Path,
) -> None:
    """Require the immutable source, not a generated working copy, in the root."""
    resolved_root = _canonical_texture_source_root(
        dependency_root,
        label="Texture dependency root",
        require_directory=True,
    )
    try:
        Path(source_path).resolve().relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "Authoritative texture source must be inside Texture dependency root: "
            f"{source_path} is outside {dependency_root}"
        ) from exc


def _allowed_texture_source_roots(
    usd_path: str,
    working_dir: Path,
    context: dict[str, Any],
) -> list[Path]:
    roots = [
        _canonical_texture_source_root(
            Path(usd_path).resolve().parent,
            label="Texture USD parent",
        ),
        _canonical_texture_source_root(
            working_dir,
            label="Texture working directory",
        ),
    ]
    dependency_root = _validated_dependency_root(context)
    if dependency_root is not None:
        roots.append(
            _canonical_texture_source_root(
                dependency_root,
                label="Texture dependency root",
                require_directory=True,
            )
        )
    uv_preparation = context.get("uv_preparation")
    if isinstance(uv_preparation, dict):
        report_path = uv_preparation.get("uv_report_path")
        if isinstance(report_path, str) and report_path.strip():
            try:
                payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            input_usd = payload.get("input_usd")
            if isinstance(input_usd, str) and input_usd.strip():
                roots.append(
                    _canonical_texture_source_root(
                        Path(input_usd).resolve().parent,
                        label="Texture UV input parent",
                    )
                )

    deduped: list[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def _is_under_any_root(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.resolve()
    except (OSError, ValueError):
        return False
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except (OSError, ValueError):
            continue
    return False


def _localized_texture_copy_path(candidate: Path, tex_dir: Path) -> Path:
    """Choose a de-duplicated bundle-local path for an existing texture."""
    target = tex_dir / candidate.name
    if not target.exists():
        return target
    try:
        if target.resolve() == candidate.resolve():
            return target
    except (OSError, ValueError):
        pass
    try:
        filecmp.clear_cache()
        if target.stat().st_size == candidate.stat().st_size and filecmp.cmp(
            target,
            candidate,
            shallow=False,
        ):
            return target
    except OSError:
        pass
    digest = hashlib.sha256(str(candidate.resolve()).encode("utf-8")).hexdigest()[:8]
    return tex_dir / f"{candidate.stem}_{digest}{candidate.suffix.lower()}"


def _iter_localizable_stage_texture_references(
    stage: Usd.Stage,
    *,
    usd_path: str,
    working_dir: Path,
    output_usd_path: Path,
    context: dict[str, Any],
) -> Iterator[tuple[Usd.Prim, Usd.Attribute, Path, bool]]:
    """Yield existing local PNG references that the localization pass can edit."""
    bundle_root = working_dir.resolve()
    allowed_roots = _allowed_texture_source_roots(usd_path, working_dir, context)

    # PrimAllPrimsPredicate reaches the authored specs behind an instanceable
    # reference, which the default predicate does not descend into. On an
    # instanced CAD asset the default predicate yields no shaders at all, and
    # this pass only worked because texture application had de-instanced the
    # stage first (issue #916). It yields authored prims, not proxies, so the
    # skip below is retained only as a guard.
    for prim in stage.Traverse(Usd.PrimAllPrimsPredicate):
        if prim.IsInstanceProxy():
            continue
        is_shader = prim.IsA(UsdShade.Shader)
        for attr in prim.GetAttributes():
            value = attr.Get()
            raw: str | None = None
            set_asset = False
            if isinstance(value, Sdf.AssetPath) and value.path:
                raw = value.path
                set_asset = True
            elif isinstance(value, str) and value and is_shader:
                attr_name = attr.GetName()
                if attr_name.startswith("inputs:") and attr_name.endswith("_texture"):
                    raw = value
            if not raw or not raw.lower().endswith(".png"):
                continue
            if _is_unbundleable_asset_path(raw):
                continue
            if _is_portable_texture_reference(raw, output_usd_path, working_dir):
                continue

            candidate = _resolve_layer_anchored_path(
                attr,
                raw,
                Path(usd_path).resolve().parent,
            )
            if candidate is None:
                continue
            try:
                candidate = candidate.resolve()
            except (OSError, ValueError):
                continue
            if not candidate.is_file() or candidate.suffix.lower() != ".png":
                continue
            try:
                candidate.relative_to(bundle_root)
            except ValueError:
                if not _is_under_any_root(candidate, allowed_roots):
                    continue

            yield prim, attr, candidate, set_asset


def _preflight_stage_texture_reference_localization(
    stage: Usd.Stage,
    *,
    usd_path: str,
    working_dir: Path,
    output_usd_path: Path,
    context: dict[str, Any],
) -> None:
    """Reject unsafe stage-wide localization before any material is changed."""
    for (
        prim,
        _attr,
        _candidate,
        _set_asset,
    ) in _iter_localizable_stage_texture_references(
        stage,
        usd_path=usd_path,
        working_dir=working_dir,
        output_usd_path=output_usd_path,
        context=context,
    ):
        _require_nonvariant_edit_subtree(
            prim,
            purpose="Texture reference localization",
        )


def _localize_stage_texture_references(
    stage: Usd.Stage,
    *,
    usd_path: str,
    working_dir: Path,
    output_usd_path: Path,
    context: dict[str, Any],
    deinstanced_paths: set[str] | None = None,
) -> list[str]:
    """Rewrite all local PNG refs to bundle-local sibling-relative paths.

    Scoped texture edits leave unedited materials untouched. For SimReady assets
    those untouched materials can still point back to the original source
    package, which renders locally but fails the downloadable package
    portability check. This pass copies each existing local PNG dependency once
    into ``working_dir/textures`` and authors a path relative to the output USD.
    """
    tex_dir = working_dir / "textures"
    bundle_root = working_dir.resolve()
    localized: list[str] = []

    for prim, attr, candidate, set_asset in _iter_localizable_stage_texture_references(
        stage,
        usd_path=usd_path,
        working_dir=working_dir,
        output_usd_path=output_usd_path,
        context=context,
    ):
        _require_nonvariant_edit_subtree(
            prim,
            purpose="Texture reference localization",
        )

        try:
            candidate.relative_to(bundle_root)
            target = candidate
        except ValueError:
            tex_dir.mkdir(parents=True, exist_ok=True)
            target = _localized_texture_copy_path(candidate, tex_dir)
            try:
                if not target.exists() or target.resolve() != candidate:
                    shutil.copyfile(candidate, target)
            except OSError as err:
                logger.warning(
                    "Failed to localize stage texture %s -> %s: %s",
                    candidate,
                    target,
                    err,
                )
                continue

        authored = _author_texture_reference(str(target), output_usd_path, working_dir)
        try:
            if prim.IsInstance() or prim.IsInstanceable():
                _deinstance_prim(stage, prim, deinstanced_paths)
            if set_asset:
                attr.Set(Sdf.AssetPath(authored))
            else:
                attr.Set(authored)
        except Exception as err:
            logger.warning(
                "Failed to rewrite texture reference %s = %r: %s",
                attr.GetPath(),
                authored,
                err,
            )
            continue
        localized.append(f"{prim.GetPath()}:{attr.GetName()}")

    return localized


# SimReady/OmniPBR MDL texture-input names → channel of the BlendedTextures bundle.
# Keys are lowercased so we can match case-insensitively (e.g. SimReady's
# "ORM_texture" alongside OmniPBR's "ORM_texture" and OmniPBR-derived
# "diffuse_texture").
_MDL_TEXTURE_INPUT_MAP = {
    "diffuse_texture": "albedo",
    "albedo_texture": "albedo",
    "base_color_texture": "albedo",
    "diffuse_color_texture": "albedo",
    "normalmap_texture": "normal",
    "detail_normalmap_texture": "normal",
    "normal_texture": "normal",
    "normal_map_texture": "normal",
    "orm_texture": "orm",
    "reflectionroughness_texture": "roughness",
    "roughness_texture": "roughness",
    "specular_roughness_texture": "roughness",
    "metallic_texture": "metalness",
    "metalness_texture": "metalness",
}

_PREVIEW_SURFACE_TEXTURE_INPUT_MAP = {
    "diffusecolor": "albedo",
    "basecolor": "albedo",
    "normal": "normal",
    "occlusion": "orm",
    "roughness": "roughness",
    "metallic": "metalness",
}
_PREVIEW_SURFACE_SCALAR_CHANNELS = frozenset({"orm", "roughness", "metalness"})
_PACKED_ORM_PREVIEW_OUTPUTS = {
    "orm": "r",
    "roughness": "g",
    "metalness": "b",
}

_AUTHORED_PREVIEW_SHADER_NAMES = {
    "st": "TextureAgentSTReader",
    "albedo": "TextureAgentAlbedoTexture",
    "normal": "TextureAgentNormalTexture",
    "orm": "TextureAgentORMTexture",
    "roughness": "TextureAgentRoughnessTexture",
    "metalness": "TextureAgentMetalnessTexture",
}


def _is_mdl_shader(prim: Usd.Prim) -> bool:
    if not prim.IsA(UsdShade.Shader):
        return False
    attr = prim.GetAttribute("info:mdl:sourceAsset")
    if attr and attr.IsValid() and attr.HasAuthoredValue():
        return True

    # Some Omniverse-authored assets leave the MDL source asset empty while
    # still marking the shader as a sourceAsset implementation with a concrete
    # MDL sub-identifier, commonly "OmniPBR". Treat that as an MDL shader so
    # texture inputs are still rewritten to generated maps.
    implementation_attr = prim.GetAttribute("info:implementationSource")
    implementation_source = (
        implementation_attr.Get()
        if implementation_attr and implementation_attr.IsValid()
        else None
    )
    if str(implementation_source or "") != "sourceAsset":
        return False

    sub_attr = prim.GetAttribute("info:mdl:sourceAsset:subIdentifier")
    return bool(sub_attr and sub_attr.IsValid() and sub_attr.Get())


_UNBUNDLEABLE_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

# Channels whose generated PNG is also referenced via an `Sdf.AssetPath`-typed
# attribute on the Material prim (the OpenPBR write path). USDZ packaging only
# follows asset-typed dependencies; channels in this set are guaranteed to be
# bundled regardless of how the MDL Shader's input is typed. Channels NOT in
# this set (today: only ``orm`` — the packed ORM is not duplicated to an
# OpenPBR Asset attr) cannot be safely written into a string/token-typed MDL
# input, since the packager would rewrite the path but the file would never
# enter the downloaded archive.
_USDZ_BUNDLED_CHANNELS = frozenset({"albedo", "normal", "roughness", "metalness"})

# The MDL `*_texture` input types we know how to round-trip safely. Anything
# else (e.g. AssetArray, StringArray, custom typedefs) is left untouched —
# we'd rather skip a rare schema than emit a corrupted value.
_SUPPORTED_TEXTURE_INPUT_TYPES = frozenset(
    {Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.String, Sdf.ValueTypeNames.Token}
)


def _is_unbundleable_asset_path(path: str) -> bool:
    """A texture path the public bundle cannot resolve at render time.

    Anything carrying a URI scheme (`omniverse://`, `http://`, `https://`, …)
    falls in this bucket — only callers with the matching asset resolver can
    fetch it, and the service's USDZ packager rewrites every `*.png` asset
    path to `../textures/<basename>` regardless of source, which would point
    at a file the bundle does not ship. Local relative or absolute paths are
    left alone — they're either already packageable or were placed there
    intentionally by the asset author.
    """
    if not path:
        return False
    return bool(_UNBUNDLEABLE_SCHEME_RE.match(path))


def _resolve_layer_anchored_path(
    attr: Usd.Attribute,
    raw: str,
    fallback_anchor: Path,
) -> Path | None:
    """Resolve a relative MDL asset path against the layer that authored it.

    Composed USDs can author shader inputs in a referenced or sublayered file,
    where ``@./opacity.png@`` is relative to *that* layer, not the root. Using
    the root USD's directory as the anchor (Codex round-5 finding) silently
    drops legitimate textures from referenced material libraries.

    Resolution order:

    1. Prefer the asset resolver's already-resolved path
       (``Sdf.AssetPath.resolvedPath``) when USD has populated it.
    2. Fall back to anchoring on the strongest authoring layer's directory
       (from the property stack).
    3. Fall back to ``fallback_anchor`` (the root USD's directory) when no
       layer-on-disk anchor is available (anonymous layers, in-memory stages).

    Errors during ``Path.resolve()`` (NUL bytes, invalid UTF-8, …) are caught
    so a malicious USD can't crash apply_textures.
    """
    val = attr.Get()
    if val is None:
        return None
    resolved = getattr(val, "resolvedPath", "") or ""
    if resolved:
        try:
            return Path(resolved).resolve()
        except (OSError, ValueError):
            return None

    anchor = fallback_anchor
    try:
        prop_stack = attr.GetPropertyStack(Usd.TimeCode.Default())
    except Exception:
        prop_stack = []
    if prop_stack:
        layer = prop_stack[0].layer
        layer_path = getattr(layer, "realPath", "") if layer else ""
        if layer_path:
            anchor = Path(layer_path).parent

    try:
        return (anchor / raw).resolve()
    except (OSError, ValueError):
        return None


def _localize_asset(
    candidate: Path,
    allowed_roots: Path | list[Path],
    tex_dir: Path,
    mat_name: str,
    input_name: str,
) -> str | None:
    """Copy an already-resolved local asset into the bundle textures dir.

    Security: USD content can come from untrusted uploads, so we refuse to
    localize anything that resolves outside the upload root or the run-owned
    composition staging tree. Without this scope check a crafted MDL input
    like ``inputs:leak_texture = @/etc/passwd@`` would copy a host file into
    ``working_dir/textures/`` — which the service exposes as a downloadable
    artifact. We additionally require the file to
    carry a (case-insensitive) ``.png`` suffix so a path with no extension or
    a non-PNG image can't slip into the bundle: the service packager and the
    textures-artifact ZIP only handle ``*.png`` (case-sensitive ``endswith``),
    so anything else would be silently dropped or inconsistently rewritten.

    The caller is responsible for resolving the raw asset path (including
    layer-anchored relative resolution); this function only enforces the
    security boundary and the copy.

    Returns the new local path inside ``tex_dir`` on success, or ``None`` if
    the source could not be resolved or copied — caller should fall back to
    clearing.
    """
    try:
        if not candidate.is_file():
            return None
    except (OSError, ValueError):
        return None

    # Reject anything outside the uploaded source or this run's owned
    # workspace. The latter includes dependency files copied into a private
    # composition-preserving staging tree.
    roots = allowed_roots if isinstance(allowed_roots, list) else [allowed_roots]
    if not _is_under_any_root(candidate, roots):
        return None

    # Only PNG (case-insensitive). Non-PNG suffixes would not be rewritten by
    # the service packager (which matches lower-case ``.png``) and would not
    # make it into the textures-artifact ZIP (which globs ``*.png``), so
    # accepting them creates inconsistent bundles.
    if candidate.suffix.lower() != ".png":
        return None

    # Already inside the bundle textures dir → nothing to do.
    try:
        if candidate.parent.samefile(tex_dir):
            return str(candidate)
    except OSError:
        pass

    # Prefix with material+input to avoid collisions across materials sharing
    # a basename (`opacity.png`). Always emit lower-case ``.png`` so the
    # packager's ``endswith(".png")`` match (case-sensitive) succeeds.
    safe_mat = mat_name.replace("/", "_").lstrip("_") or "mat"
    target = tex_dir / f"{safe_mat}__{input_name}.png"

    tex_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not target.exists() or not target.samefile(candidate):
            shutil.copyfile(candidate, target)
    except OSError as err:
        logger.warning(
            "Failed to localize MDL asset %s -> %s: %s", candidate, target, err
        )
        return None
    return str(target)


def _override_mdl_texture_inputs(
    stage: Usd.Stage,
    mat_path: str,
    channel_paths: dict[str, str],
    usd_path: str,
    working_dir: Path,
    output_usd_path: Path,
    *,
    allowed_source_roots: list[Path] | None = None,
) -> tuple[int, list[str], list[str]]:
    """Overwrite MDL shader texture inputs in-place with bundle-local paths.

    SimReady/OmniPBR-style materials carry a child Shader with an `info:mdl:sourceAsset`
    and texture inputs like `inputs:normalmap_texture` / `inputs:ORM_texture`. The
    OpenPBR-style attributes the agent writes on the Material prim are not consumed
    by the MDL shader, so without this override the freshly generated textures are
    silently ignored at render time and the original (often Nucleus-hosted) refs
    survive into the output bundle.

    For unmapped authored `*_texture` inputs (e.g. `opacity_texture`,
    `emissive_color_texture`, `displacement_texture`) the rule is:

    * **URI-scheme paths** (`omniverse://...`, `http(s)://...`) are unbundleable
      — the public bundle's asset resolver cannot satisfy them and the service
      packager's `../textures/<basename>` rewrite would dangle. → cleared.
    * **Local paths that resolve to an existing file on disk** (relative to the
      input USD or absolute) and remain under a normalized trusted source root
      are copied into ``working_dir/textures`` under a
      `<material>__<input>.<ext>` filename so the service packager's rewrite
      step finds them, and the input is rewritten to that local copy. →
      localized.
    * **Local paths that do not resolve** (the asset author's reference is
      already broken) are cleared.

    Clearing an unbundleable path drops back to the MDL's constant default,
    which renders correctly everywhere.

    Returns:
        (overridden_count, cleared_input_names, localized_input_names)
    """
    mat_prim = stage.GetPrimAtPath(mat_path)
    if not mat_prim.IsValid():
        return 0, [], []

    upload_root = Path(usd_path).resolve().parent
    trusted_source_roots = allowed_source_roots or [upload_root, working_dir]
    tex_dir = working_dir / "textures"

    overridden = 0
    cleared: list[str] = []
    localized: list[str] = []
    mat_name = Path(mat_path).name
    for child in mat_prim.GetChildren():
        if not _is_mdl_shader(child):
            continue
        shader = UsdShade.Shader(child)
        for inp in shader.GetInputs():
            base = inp.GetBaseName()
            # MDL shaders can legally author ``inputs:*_texture`` as ``asset``,
            # ``string`` or ``token`` (Codex round-6/7 findings). Read the
            # current value as a plain string regardless, then write back using
            # the input's native type via ``_safe_set_typed_value`` so we
            # don't crash on a string-typed input nor silently leave a
            # Nucleus URL pointing at an unbundleable file.
            type_name = inp.GetTypeName()
            existing = _read_texture_input_string(inp, type_name)
            if existing is None:
                continue
            channel = _MDL_TEXTURE_INPUT_MAP.get(base.lower())

            if channel is not None:
                new_path = channel_paths.get(channel)
                if not new_path:
                    continue
                # Asset-typed mapped inputs always override. String/token
                # mapped inputs only override for channels that already have
                # a parallel Asset-typed dep on the Material — otherwise
                # USDZ packaging won't bundle the file (Codex round-9
                # finding: packed ORM is the canonical un-bundled channel).
                if (
                    type_name != Sdf.ValueTypeNames.Asset
                    and channel not in _USDZ_BUNDLED_CHANNELS
                ):
                    if _safe_set_typed_value(inp, type_name, ""):
                        cleared.append(f"{mat_path}:{base}")
                    continue
                if _safe_set_typed_value(inp, type_name, new_path):
                    overridden += 1
                continue

            if not base.lower().endswith("_texture"):
                continue
            if not existing:
                continue
            if _is_unbundleable_asset_path(existing):
                if _safe_set_typed_value(inp, type_name, ""):
                    cleared.append(f"{mat_path}:{base}")
                continue
            # Localization writes a copy into ``working_dir/textures`` and
            # rewrites the input to point at it. USDZ packaging only follows
            # ``Sdf.AssetPath``-typed dependencies, so localizing a
            # string/token-typed unmapped input would put the path in the
            # USD but never include the file in the downloaded archive
            # (Codex round-8 finding). For string/token unmapped inputs we
            # therefore clear instead — the MDL drops back to its constant
            # default, which renders correctly. Mapped channels above are
            # always safe because the OpenPBR Material attribute references
            # the same generated PNG via an Asset-typed dep that USDZ does
            # bundle.
            if type_name != Sdf.ValueTypeNames.Asset:
                if _safe_set_typed_value(inp, type_name, ""):
                    cleared.append(f"{mat_path}:{base}")
                continue
            candidate = _resolve_layer_anchored_path(
                inp.GetAttr(), existing, upload_root
            )
            copied = (
                _localize_asset(
                    candidate,
                    trusted_source_roots,
                    tex_dir,
                    mat_name,
                    base,
                )
                if candidate is not None
                else None
            )
            if copied is None:
                if _safe_set_typed_value(inp, type_name, ""):
                    cleared.append(f"{mat_path}:{base}")
            else:
                copied_ref = _author_texture_reference(
                    copied, output_usd_path, working_dir
                )
                if _safe_set_typed_value(inp, type_name, copied_ref):
                    localized.append(f"{mat_path}:{base}")

    return overridden, cleared, localized


def _shader_id(shader: UsdShade.Shader) -> str:
    value = shader.GetIdAttr().Get()
    return str(value) if value else ""


def _preview_source_output_name(source_name: object) -> str:
    value = str(source_name or "").strip().lower()
    if value.startswith("outputs:"):
        return value.split(":", 1)[1]
    return value


def _connected_usd_uv_texture_source(
    inp: UsdShade.Input,
) -> tuple[UsdShade.Shader, str] | None:
    connected = inp.GetConnectedSource()
    if connected is None:
        return None
    source, source_name, _source_type = connected
    if not source:
        return None
    prim = source.GetPrim()
    if not prim or not prim.IsValid() or not prim.IsA(UsdShade.Shader):
        return None
    shader = UsdShade.Shader(prim)
    if _shader_id(shader) != "UsdUVTexture":
        return None
    return shader, _preview_source_output_name(source_name)


def _preview_input_has_shader_connection(
    attribute: UsdShade.Input | UsdShade.Output,
) -> bool:
    """Return whether a shading attribute resolves to an authored shader.

    Material and NodeGraph interface inputs commonly forward constants into a
    PreviewSurface. Those connections are not texture coverage and generated
    maps should replace them. Interface inputs or outputs that eventually
    resolve to a shader output remain authored networks and must be preserved.
    """
    visited: set[str] = set()
    current = attribute
    while True:
        connected = current.GetConnectedSource()
        if connected is None:
            return False
        source, source_name, source_type = connected
        if not source:
            return False

        source_prim = source.GetPrim()
        if source_type == UsdShade.AttributeType.Output and source_prim.IsA(
            UsdShade.Shader
        ):
            return True
        if source_type == UsdShade.AttributeType.Input:
            source_attribute = source.GetInput(str(source_name))
        elif source_type == UsdShade.AttributeType.Output:
            source_attribute = source.GetOutput(str(source_name))
        else:
            return True  # pragma: no cover - unfamiliar USD attribute type
        if not source_attribute:
            # Preserve unfamiliar connection shapes conservatively.
            return True
        attribute_path = str(source_attribute.GetAttr().GetPath())
        if attribute_path in visited:
            # Preserve cyclic interface graphs rather than mutating malformed
            # source data while trying to activate generated maps.
            return True
        visited.add(attribute_path)
        current = source_attribute


def _preview_graph_shader(
    stage: Usd.Stage,
    mat_path: str,
    base_name: str,
    shader_id: str,
) -> UsdShade.Shader:
    """Return a reserved shader node without retyping existing material prims."""
    material_prim = stage.GetPrimAtPath(mat_path)
    # There are one more candidate names than direct children, so a free name
    # must exist even when many earlier Texture Agent names are occupied by
    # incompatible prims. This keeps the search bounded without silently
    # emitting a partially connected preview graph.
    for suffix in range(len(material_prim.GetAllChildren()) + 1):
        name = base_name if suffix == 0 else f"{base_name}_{suffix}"
        shader_path = f"{mat_path}/{name}"
        prim = stage.GetPrimAtPath(shader_path)
        if not prim.IsValid():
            shader = UsdShade.Shader.Define(stage, shader_path)
            shader.CreateIdAttr(shader_id)
            return shader
        if (
            not prim.IsActive()
            or not prim.IsDefined()
            or prim.IsAbstract()
            or not prim.IsA(UsdShade.Shader)
        ):
            continue
        shader = UsdShade.Shader(prim)
        if _shader_id(shader) == shader_id:
            return shader

    raise RuntimeError(
        f"Could not reserve a {shader_id} shader below material {mat_path}"
    )


def _configure_preview_uv_texture(
    shader: UsdShade.Shader,
    *,
    texture_path: str,
    st_output: UsdShade.Output,
    source_color_space: str,
    fallback: Gf.Vec4f,
) -> None:
    """Configure one package-local UsdUVTexture node."""
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(texture_path)
    )
    shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
        source_color_space
    )
    shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    shader.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(fallback)
    shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_output)


def _author_usd_preview_texture_graph(
    stage: Usd.Stage,
    mat_path: str,
    preview_shader_channels: list[tuple[UsdShade.Shader, frozenset[str]]],
    channel_paths: dict[str, str],
) -> list[str]:
    """Author missing PreviewSurface texture connections.

    Texture backends are normalized to ``BlendedTextures`` before this point,
    so this graph is shared by simple image generation, projection services,
    and any future backend. Existing connections are preserved; each shader is
    connected only for the generated channels it does not already source.
    """
    needed_channels = {
        channel
        for _preview_shader, channels in preview_shader_channels
        for channel in channels
        if channel_paths.get(channel)
    }
    if not needed_channels:
        return []

    st_reader = _preview_graph_shader(
        stage,
        mat_path,
        _AUTHORED_PREVIEW_SHADER_NAMES["st"],
        "UsdPrimvarReader_float2",
    )
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_reader.CreateInput("fallback", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.0, 0.0))
    st_output = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    texture_outputs: dict[str, UsdShade.Output] = {}
    authored: list[str] = []
    for channel, source_color_space, fallback, output_name, output_type in (
        (
            "albedo",
            "sRGB",
            Gf.Vec4f(0.18, 0.18, 0.18, 1.0),
            "rgb",
            Sdf.ValueTypeNames.Float3,
        ),
        (
            "normal",
            "raw",
            Gf.Vec4f(0.5, 0.5, 1.0, 1.0),
            "rgb",
            Sdf.ValueTypeNames.Float3,
        ),
        (
            "orm",
            "raw",
            Gf.Vec4f(1.0, 1.0, 0.0, 1.0),
            "r",
            Sdf.ValueTypeNames.Float,
        ),
        (
            "roughness",
            "raw",
            Gf.Vec4f(0.5, 0.5, 0.5, 1.0),
            "r",
            Sdf.ValueTypeNames.Float,
        ),
        (
            "metalness",
            "raw",
            Gf.Vec4f(0.0, 0.0, 0.0, 1.0),
            "r",
            Sdf.ValueTypeNames.Float,
        ),
    ):
        if channel not in needed_channels:
            continue
        texture_path = channel_paths.get(channel)
        if not texture_path:
            continue
        texture = _preview_graph_shader(
            stage,
            mat_path,
            _AUTHORED_PREVIEW_SHADER_NAMES[channel],
            "UsdUVTexture",
        )
        _configure_preview_uv_texture(
            texture,
            texture_path=texture_path,
            st_output=st_output,
            source_color_space=source_color_space,
            fallback=fallback,
        )
        if channel == "normal":
            # UsdPreviewSurface expects tangent-space normals in [-1, 1], while
            # normal-map texels are stored in [0, 1].
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(2.0, 2.0, 2.0, 2.0)
            )
            texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(-1.0, -1.0, -1.0, 0.0)
            )
        texture_outputs[channel] = texture.CreateOutput(output_name, output_type)
        authored.append(f"{texture.GetPrim().GetPath()}:file")

    for preview_shader, missing_channels in preview_shader_channels:
        albedo_output = texture_outputs.get("albedo")
        if albedo_output and "albedo" in missing_channels:
            preview_shader.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            ).ConnectToSource(albedo_output)
        normal_output = texture_outputs.get("normal")
        if normal_output and "normal" in missing_channels:
            preview_shader.CreateInput(
                "normal", Sdf.ValueTypeNames.Normal3f
            ).ConnectToSource(normal_output)
        orm_output = texture_outputs.get("orm")
        if orm_output and "orm" in missing_channels:
            preview_shader.CreateInput(
                "occlusion", Sdf.ValueTypeNames.Float
            ).ConnectToSource(orm_output)
        roughness_output = texture_outputs.get("roughness")
        if roughness_output and "roughness" in missing_channels:
            preview_shader.CreateInput(
                "roughness", Sdf.ValueTypeNames.Float
            ).ConnectToSource(roughness_output)
        metalness_output = texture_outputs.get("metalness")
        if metalness_output and "metalness" in missing_channels:
            preview_shader.CreateInput(
                "metallic", Sdf.ValueTypeNames.Float
            ).ConnectToSource(metalness_output)

    return authored


def _shared_preview_texture_uses_packed_orm(
    outputs_by_channel: dict[str, set[str]],
) -> bool:
    if len(outputs_by_channel) <= 1:
        return False
    for channel, expected_output in _PACKED_ORM_PREVIEW_OUTPUTS.items():
        outputs = outputs_by_channel.get(channel)
        if outputs is not None and expected_output not in outputs:
            return False
    return True


def _override_usd_preview_texture_inputs(
    stage: Usd.Stage,
    mat_path: str,
    channel_paths: dict[str, str],
) -> list[str]:
    """Point existing UsdPreviewSurface texture nodes at generated maps.

    Some renderers fall back to the UsdPreviewSurface graph when they cannot
    resolve OmniPBR/MDL. SimReady assets often carry both graphs, so updating
    only OpenPBR/MDL inputs can leave fallback renders showing the original
    texture set. This routine rewrites existing connected UsdUVTexture ``file``
    inputs and authors package-local nodes only for generated channels that are
    still unconnected.
    """
    mat_prim = stage.GetPrimAtPath(mat_path)
    if not mat_prim.IsValid():
        return []

    overridden: list[str] = []
    touched_files: set[str] = set()
    preview_shader_channels: list[tuple[UsdShade.Shader, frozenset[str]]] = []
    material_prefix = mat_path.rstrip("/") + "/"
    for prim in Usd.PrimRange(mat_prim):
        if not prim.IsA(UsdShade.Shader):
            continue
        preview_shader = UsdShade.Shader(prim)
        if _shader_id(preview_shader) != "UsdPreviewSurface":
            continue

        connected_inputs: list[tuple[UsdShade.Input, UsdShade.Shader, str, str]] = []
        channels_by_texture: dict[str, set[str]] = defaultdict(set)
        scalar_outputs_by_texture: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        connected_preview_channels: set[str] = set()
        for inp in preview_shader.GetInputs():
            channel = _PREVIEW_SURFACE_TEXTURE_INPUT_MAP.get(inp.GetBaseName().lower())
            if channel is None:
                continue
            if _preview_input_has_shader_connection(inp):
                connected_preview_channels.add(channel)
            source = _connected_usd_uv_texture_source(inp)
            if source is None:
                continue
            uv_texture, source_output = source
            texture_node_path = str(uv_texture.GetPrim().GetPath())
            if not texture_node_path.startswith(material_prefix):
                continue
            connected_inputs.append((inp, uv_texture, channel, source_output))
            channels_by_texture[texture_node_path].add(channel)
            if channel in _PREVIEW_SURFACE_SCALAR_CHANNELS:
                scalar_outputs_by_texture[texture_node_path][channel].add(source_output)

        for _inp, uv_texture, channel, _source_output in connected_inputs:
            texture_node_path = str(uv_texture.GetPrim().GetPath())
            connected_channels = channels_by_texture.get(texture_node_path, set())
            if len(connected_channels) > 1:
                scalar_outputs = scalar_outputs_by_texture.get(texture_node_path, {})
                if (
                    connected_channels <= _PREVIEW_SURFACE_SCALAR_CHANNELS
                    and _shared_preview_texture_uses_packed_orm(scalar_outputs)
                    and channel_paths.get("orm")
                ):
                    # A shared scalar node is only safe to rewrite to packed ORM
                    # when the existing graph already samples ORM-style channels.
                    channel = "orm"
                else:
                    # Ambiguous shared nodes cannot represent separate generated
                    # maps without changing graph topology; preserve the source.
                    continue
            texture_path = channel_paths.get(channel)
            if not texture_path:
                continue
            file_input = uv_texture.GetInput("file")
            if file_input:
                file_input.Set(Sdf.AssetPath(texture_path))
            else:
                uv_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath(texture_path)
                )
            file_path = str(uv_texture.GetPrim().GetPath())
            record = f"{file_path}:file"
            if record not in touched_files:
                touched_files.add(record)
                overridden.append(record)

        # Preserve authored shader networks, including custom and external
        # sources, while filling any remaining generated channels. Material
        # interface inputs that only forward constants are not texture
        # coverage and are intentionally replaced.
        missing_channels = frozenset(
            channel
            for channel in _PREVIEW_SURFACE_TEXTURE_INPUT_MAP.values()
            if channel_paths.get(channel) and channel not in connected_preview_channels
        )
        if missing_channels:
            preview_shader_channels.append((preview_shader, missing_channels))

    for record in _author_usd_preview_texture_graph(
        stage,
        mat_path,
        preview_shader_channels,
        channel_paths,
    ):
        if record not in touched_files:
            touched_files.add(record)
            overridden.append(record)
    return overridden


def _read_texture_input_string(
    inp: UsdShade.Input, type_name: Sdf.ValueTypeName
) -> str | None:
    """Read an MDL texture input as a plain string, regardless of authored type.

    Returns the value's string form for ``Asset``/``String``/``Token``-typed
    inputs (the only types we know how to safely round-trip), or ``None`` for
    unauthored values, unsupported types, or array variants. ``None`` means
    "skip this input" — neither an override candidate nor a clear/localize
    candidate.
    """
    if type_name not in _SUPPORTED_TEXTURE_INPUT_TYPES:
        return None
    val = inp.Get()
    if val is None:
        return None
    if type_name == Sdf.ValueTypeNames.Asset:
        return val.path if hasattr(val, "path") else str(val)
    return str(val)


def _safe_set_typed_value(
    inp: UsdShade.Input, type_name: Sdf.ValueTypeName, value: str
) -> bool:
    """Write a string back into an MDL texture input using its authored type.

    USD content can come from untrusted uploads; an in-pipeline `Set` raising
    ``pxr.Tf.ErrorException`` would tear down the whole apply_textures step
    instead of skipping a single input. We log and return ``False`` on
    failure so the caller does not record the input in its stat list.

    Only the three texture-input types listed in
    ``_SUPPORTED_TEXTURE_INPUT_TYPES`` are accepted; anything else is a no-op
    and returns ``False``.
    """
    if type_name not in _SUPPORTED_TEXTURE_INPUT_TYPES:
        return False
    try:
        if type_name == Sdf.ValueTypeNames.Asset:
            inp.Set(Sdf.AssetPath(value))
        else:
            inp.Set(value)
        return True
    except Exception as err:
        logger.warning(
            "Failed to set MDL texture input %s = %r: %s",
            inp.GetAttr().GetPath(),
            value,
            err,
        )
        return False


def _apply_pbr_textures(
    stage: Usd.Stage,
    mat_path: str,
    textures: BlendedTextures,
    working_dir: Path,
    key: str,
    usd_path: str,
    output_usd_path: Path,
    *,
    deinstanced_paths: set[str] | None = None,
    allowed_source_roots: list[Path] | None = None,
) -> tuple[int, list[str], list[str], list[str]]:
    """Apply albedo, normal, and ORM textures to a material prim.

    Returns:
        (
            mdl_inputs_overridden,
            mdl_inputs_cleared,
            mdl_inputs_localized,
            preview_texture_inputs_overridden,
        )
    """
    requested_prim = stage.GetPrimAtPath(mat_path)
    _require_nonvariant_edit_subtree(
        requested_prim,
        purpose="Texture application",
    )
    prim = _editable_prim_for_path(
        stage,
        mat_path,
        deinstanced_paths=deinstanced_paths,
    )
    if not prim.IsValid():
        logger.warning("Material prim not found: %s", mat_path)
        return 0, [], [], []
    if (
        prim.IsInstanceProxy()
        or prim.IsInstance()
        or prim.IsInstanceable()
        or prim.IsPrototype()
        or prim.IsInPrototype()
    ):
        logger.warning("Material prim is not editable: %s", mat_path)
        return 0, [], [], []
    _require_nonvariant_edit_subtree(
        prim,
        purpose="Texture application",
    )

    # Follow the resolved prim. Resolution can move off the requested path when
    # that path is an instance proxy, and the shader-level helpers below look
    # their targets up by path; leaving mat_path on the proxy would point every
    # one of them at prims USD refuses to author.
    mat_path = str(prim.GetPath())

    # Ensure the material container is a typed Scope for NVCF traversal and
    # usd-validation-nvidia's Basic TypeChecker.
    parent = prim.GetParent()
    # Scope.Define authors a prim spec, so keep it out of read-only composition
    # contexts such as pseudo-root, instance proxies, and prototype contents.
    if _can_define_parent_scope(stage, parent):
        _require_nonvariant_edit(
            parent,
            purpose="Material scope definition",
        )
        if parent.IsInstanceable():
            _deinstance_prim(stage, parent, deinstanced_paths)
        UsdGeom.Scope.Define(stage, parent.GetPath())

    albedo_ref = _author_texture_reference(
        textures.albedo, output_usd_path, working_dir
    )
    channel_paths: dict[str, str] = {"albedo": albedo_ref}

    # Albedo
    _set_texture_attr(prim, "inputs:base_color_texture_file", albedo_ref)
    _set_tiledimage_file_input(
        stage,
        mat_path,
        "tiledimage_base_color",
        albedo_ref,
    )

    # Normal
    if textures.normal and Path(textures.normal).exists():
        normal_ref = _author_texture_reference(
            textures.normal, output_usd_path, working_dir
        )
        _set_texture_attr(prim, "inputs:geometry_normal_texture_file", normal_ref)
        _set_tiledimage_file_input(
            stage,
            mat_path,
            "tiledimage_geometry_normal",
            normal_ref,
        )
        channel_paths["normal"] = normal_ref

    # ORM → unpack into roughness + metalness (and keep packed for MDL ORM_texture)
    if textures.orm and Path(textures.orm).exists():
        import numpy as np
        from PIL import Image

        channel_paths["orm"] = _author_texture_reference(
            textures.orm, output_usd_path, working_dir
        )

        with Image.open(textures.orm) as orm_img:
            orm_arr = np.array(orm_img)
        tex_dir = working_dir / "textures"

        roughness_arr = orm_arr[:, :, 1]
        roughness_path = tex_dir / f"{key}_roughness.png"
        Image.fromarray(roughness_arr).save(str(roughness_path))
        roughness_ref = _author_texture_reference(
            str(roughness_path), output_usd_path, working_dir
        )
        _set_texture_attr(prim, "inputs:specular_roughness_texture_file", roughness_ref)
        _set_tiledimage_file_input(
            stage,
            mat_path,
            "tiledimage_specular_roughness",
            roughness_ref,
        )
        channel_paths["roughness"] = roughness_ref

        metalness_arr = orm_arr[:, :, 2]
        metalness_path = tex_dir / f"{key}_metalness.png"
        Image.fromarray(metalness_arr).save(str(metalness_path))
        metalness_ref = _author_texture_reference(
            str(metalness_path), output_usd_path, working_dir
        )
        _set_texture_attr(prim, "inputs:base_metalness_texture_file", metalness_ref)
        _set_tiledimage_file_input(
            stage,
            mat_path,
            "tiledimage_base_metalness",
            metalness_ref,
        )
        channel_paths["metalness"] = metalness_ref

    preview_texture_inputs_overridden = _override_usd_preview_texture_inputs(
        stage, mat_path, channel_paths
    )
    mdl_inputs_overridden, mdl_inputs_cleared, mdl_inputs_localized = (
        _override_mdl_texture_inputs(
            stage,
            mat_path,
            channel_paths,
            usd_path,
            working_dir,
            output_usd_path,
            allowed_source_roots=allowed_source_roots,
        )
    )
    return (
        mdl_inputs_overridden,
        mdl_inputs_cleared,
        mdl_inputs_localized,
        preview_texture_inputs_overridden,
    )


def _material_apply_paths(mat: MaterialInfo) -> list[str]:
    """Return material paths that should receive one per-material unit's maps."""
    return sorted({mat.prim_path, *mat.material_alias_paths})


def _require_nonvariant_apply_plan(
    stage: Usd.Stage,
    units_by_material: dict[str, list[PrimTextureUnit]],
    plan_units_by_id: dict[str, TexturePlanUnit] | None,
) -> None:
    """Validate every planned material and binding target before authoring."""
    for mat_units in units_by_material.values():
        mat = mat_units[0].material_info
        plan_unit = (
            _accepted_plan_unit_for_runtime_unit(plan_units_by_id, mat_units[0])
            if len(mat_units) == 1
            else None
        )
        planned_target_paths = _planned_binding_target_paths(plan_unit)
        if (
            len(mat_units) == 1
            and not mat_units[0].prim_path
            and not planned_target_paths
        ):
            for mat_path in _material_apply_paths(mat):
                _require_nonvariant_edit_subtree(
                    stage.GetPrimAtPath(mat_path),
                    purpose="Texture application",
                )
            continue

        source_path = Sdf.Path(_material_clone_source_path(stage, mat))
        _require_nonvariant_edit_subtree(
            stage.GetPrimAtPath(source_path),
            purpose="material cloning",
        )
        _require_nonvariant_edit(
            stage.GetPrimAtPath(source_path.GetParentPath()),
            purpose="material clone parent editing",
        )
        for unit in mat_units:
            selected_plan_unit = _accepted_plan_unit_for_runtime_unit(
                plan_units_by_id, unit
            )
            if selected_plan_unit is not None:
                target_prim_paths = _planned_binding_target_paths(selected_plan_unit)
            elif unit.prim_path:
                target_prim_paths = (unit.prim_path,)
            else:
                target_prim_paths = ()
            if not target_prim_paths:
                raise RuntimeError(
                    f"Texture apply unit {unit.key} has no exact prim target"
                )
            clone_path = source_path.GetParentPath().AppendChild(unit.key)
            _require_nonvariant_edit_subtree(
                stage.GetPrimAtPath(clone_path),
                purpose="material clone replacement",
            )
            for target_prim_path in target_prim_paths:
                target_prim = stage.GetPrimAtPath(target_prim_path)
                if not target_prim.IsValid():
                    if selected_plan_unit is not None:
                        raise RuntimeError(
                            "Texture apply target prim is missing before authoring: "
                            f"{target_prim_path}"
                        )
                    continue
                _require_nonvariant_edit(
                    target_prim,
                    purpose="Material binding",
                )


_USD_LAYER_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})


@dataclass(frozen=True)
class _CompositionSnapshot:
    prim_paths: frozenset[str]
    mesh_paths: frozenset[str]
    instanceable_paths: frozenset[str]
    default_prim_path: str
    stage_metadata: tuple[tuple[str, Any], ...]


def _composition_snapshot(stage: Usd.Stage) -> _CompositionSnapshot:
    """Capture the composed identities that relocation must not discard."""
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    prim_paths: set[str] = set()
    mesh_paths: set[str] = set()
    instanceable_paths: set[str] = set()
    for prim in Usd.PrimRange.Stage(stage, predicate):
        if prim.IsPseudoRoot():
            continue
        path = str(prim.GetPath())
        prim_paths.add(path)
        if prim.IsA(UsdGeom.Mesh):
            mesh_paths.add(path)
        if prim.IsInstanceable():
            instanceable_paths.add(path)
    default_prim = stage.GetDefaultPrim()
    return _CompositionSnapshot(
        prim_paths=frozenset(prim_paths),
        mesh_paths=frozenset(mesh_paths),
        instanceable_paths=frozenset(instanceable_paths),
        default_prim_path=(
            str(default_prim.GetPath())
            if default_prim and default_prim.IsValid()
            else ""
        ),
        stage_metadata=tuple(
            sorted(stage.GetPseudoRoot().GetAllAuthoredMetadata().items())
        ),
    )


def _stage_has_file_backed_composition(stage: Usd.Stage) -> bool:
    """Return whether exporting only the root layer would lose composition."""
    root_layer = stage.GetRootLayer()
    if root_layer.subLayerPaths:
        return True
    try:
        # Internal references are reported as an empty asset identifier. They
        # carry composition semantics but do not require copying another file.
        if any(
            str(dependency).strip()
            for dependency in root_layer.GetCompositionAssetDependencies()
        ):
            return True
    except (AttributeError, RuntimeError):
        pass
    session_layer = stage.GetSessionLayer()
    composed_layers = [
        layer for layer in stage.GetUsedLayers() if layer is not session_layer
    ]
    return len(composed_layers) > 1


def _localize_composed_stage(
    source_path: Path,
    output_dir: Path,
    *,
    dependency_root: Path | None = None,
    root_layer_at_output_root: bool = False,
) -> Path:
    """Copy the exact bounded dependency closure while preserving its layout.

    ``Sdf.Layer.Export`` moves only the root, so its relative arcs dangle.
    ``UsdUtils.LocalizeAsset`` is not suitable here either: the USD 25.05 build
    rewrites dependencies into numbered directories but leaves some paths
    anchored to the package root instead of their authoring layer. Copying the
    computed closure at its source-relative locations preserves every authored
    relative path without flattening variants, payloads, or instances.

    Local files outside the trusted dependency root are rejected, preventing a
    crafted asset path from copying arbitrary host files into a downloadable
    result. The root defaults to the source directory, while callers may name a
    wider input bundle root for legitimate ``../shared`` composition arcs.
    Unresolved runtime URI tokens (for example an MDL sourceAsset) remain
    authored but are not fetched; unresolved USD composition arcs fail closed.
    """
    source_path = Path(os.path.abspath(source_path))
    source_root = Path(os.path.abspath(dependency_root or source_path.parent))
    try:
        resolved_source_root = _canonical_texture_source_root(
            source_root,
            label=(
                "Texture dependency root"
                if dependency_root is not None
                else "Texture source root"
            ),
            require_directory=True,
        )
        resolved_source_path = source_path.resolve(strict=True)
        resolved_source_path.relative_to(resolved_source_root)
        try:
            source_relative_path = source_path.relative_to(source_root)
        except ValueError:
            source_relative_path = source_path.relative_to(resolved_source_root)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "Layered texture input is outside its trusted dependency root: "
            f"{source_path} is outside {source_root}"
        ) from exc
    dependency_layers, dependency_assets, unresolved = UsdUtils.ComputeAllDependencies(
        str(source_path)
    )
    unresolved_layers = sorted(
        value
        for value in (str(item) for item in unresolved)
        if Path(value).suffix.lower() in _USD_LAYER_SUFFIXES
    )
    if unresolved_layers:
        raise RuntimeError(
            "Layered texture input has unresolved USD composition dependencies: "
            + ", ".join(unresolved_layers[:10])
        )

    def _lexical_path(candidate: Path) -> Path:
        """Normalize dot segments without following authored symlinks."""
        return Path(os.path.abspath(candidate))

    def _bounded_dependency_paths(candidate: Path) -> tuple[Path, Path, Path]:
        """Return lexical, canonical, and source-relative dependency paths."""
        lexical = _lexical_path(candidate)
        try:
            resolved = lexical.resolve(strict=True)
            resolved.relative_to(resolved_source_root)
            try:
                relative = lexical.relative_to(source_root)
            except ValueError:
                relative = lexical.relative_to(resolved_source_root)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "Layered texture output has a dependency outside the source "
                f"dependency root: {candidate}"
            ) from exc
        return lexical, resolved, relative

    layer_sources: list[Path] = []
    for layer in dependency_layers:
        identifier = str(layer.realPath or layer.identifier)
        if identifier and not identifier.startswith("anon:"):
            layer_sources.append(Path(identifier))
    layer_sources.append(source_path)

    asset_sources = [
        Path(value)
        for value in (str(asset) for asset in dependency_assets)
        if value and not _is_unbundleable_asset_path(value)
    ]
    source_lexical_path = _lexical_path(source_path)

    def _localized_dependency_path(lexical: Path, relative: Path) -> Path:
        if root_layer_at_output_root and lexical == source_lexical_path:
            suffix = source_path.suffix or ".usd"
            return output_dir / f"source_root{suffix}"
        destination_root = (
            output_dir / "source" if root_layer_at_output_root else output_dir
        )
        return destination_root / relative

    copy_plan: list[tuple[Path, Path, Path, Path]] = []
    for candidate in [*layer_sources, *asset_sources]:
        lexical, resolved, relative = _bounded_dependency_paths(candidate)
        if not resolved.is_file():
            raise RuntimeError(f"Layered USD dependency is not a file: {resolved}")
        copy_plan.append(
            (
                lexical,
                resolved,
                relative,
                _localized_dependency_path(lexical, relative),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[Path, Path] = {}
    for lexical, resolved, _relative, destination in copy_plan:
        if lexical not in copied:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved, destination)
            if destination.suffix.lower() in _USD_LAYER_SUFFIXES:
                destination.chmod(destination.stat().st_mode | stat.S_IWUSR)
            copied[lexical] = destination

    # Relative paths keep the same meaning because the directory structure is
    # unchanged. Re-anchor the uncommon absolute path that still points inside
    # the copied source tree; outside paths were rejected above.
    for original in layer_sources:
        localized_layer_path = copied[_lexical_path(original)]
        localized_layer = Sdf.Layer.FindOrOpen(str(localized_layer_path))
        if localized_layer is None:
            raise RuntimeError(
                f"Failed to open localized USD layer: {localized_layer_path}"
            )

        def _reanchor_absolute_path(
            raw: str,
            *,
            source_anchor: Path = _lexical_path(original).parent,
            authoring_dir: Path = localized_layer_path.parent,
        ) -> str:
            if not raw or _is_unbundleable_asset_path(raw):
                return raw
            raw_path = Path(raw)
            if not raw_path.is_absolute() and not root_layer_at_output_root:
                return raw.replace("\\", "/")
            source_asset = (
                raw_path if raw_path.is_absolute() else source_anchor / raw_path
            )
            try:
                lexical, _resolved, relative = _bounded_dependency_paths(source_asset)
            except RuntimeError:
                return raw
            localized_asset = _localized_dependency_path(lexical, relative)
            if not localized_asset.is_file():
                raise RuntimeError(
                    "Layered USD dependency was not copied into the localized "
                    f"output: {raw}"
                )
            return Path(os.path.relpath(localized_asset, authoring_dir)).as_posix()

        UsdUtils.ModifyAssetPaths(localized_layer, _reanchor_absolute_path)
        if not localized_layer.Save():
            raise RuntimeError(
                f"Failed to save localized USD layer: {localized_layer_path}"
            )

    localized_root = _localized_dependency_path(
        source_lexical_path,
        source_relative_path,
    )
    if not localized_root.is_file():
        raise RuntimeError(
            "USD dependency localization did not produce its root layer: "
            f"{localized_root}"
        )
    return localized_root


def _stage_composed_source(
    source_path: Path,
    staging_dir: Path,
    *,
    dependency_root: Path | None = None,
    root_layer_at_staging_root: bool = False,
) -> Path:
    """Materialize a writable dependency tree below a private staging root."""
    source_dir = staging_dir / "source"
    if source_path.suffix.lower() == ".usdz":
        try:
            return extract_usdz_package_for_edit(source_path, source_dir)
        except (OSError, UsdzPackageError) as exc:
            raise RuntimeError(
                f"Failed to extract layered USDZ for texture editing: {source_path}"
            ) from exc
    if root_layer_at_staging_root:
        return _localize_composed_stage(
            source_path,
            staging_dir,
            dependency_root=dependency_root,
            root_layer_at_output_root=True,
        )
    return _localize_composed_stage(
        source_path,
        source_dir,
        dependency_root=dependency_root,
    )


def _primvar_semantic_signature(primvar: UsdGeom.Primvar) -> tuple[Any, ...] | None:
    """Return the composed UV state that preparation is allowed to change."""
    if not primvar:
        return None
    attribute = primvar.GetAttr()
    return (
        str(attribute.GetTypeName()),
        primvar.GetInterpolation(),
        primvar.GetElementSize(),
        repr(attribute.Get()),
        repr(primvar.GetIndices()) if primvar.IsIndexed() else None,
    )


def _transfer_prepared_uv_deltas(
    prepared_stage: Usd.Stage,
    source_stage: Usd.Stage,
    localized_stage: Usd.Stage,
) -> int:
    """Author only UV changes from a flattened preparation onto composition.

    The prepared stage is a writable flattened view used by UV backends. Its
    full layer must never be placed over the authored source because doing so
    pins variants and materializes instance proxies. Instead, compare the one
    property family those backends may change and author only differing
    ``primvars:st`` opinions at the same composed prim paths. A changed opinion
    that crosses a variant arc or maps only to an instance proxy is not safely
    authorable at that path and fails closed; the UV repair traversal must
    target the owning authored spec instead.
    """
    transferred = 0
    predicate = Usd.PrimIsActive & Usd.PrimIsLoaded
    for prepared_prim in Usd.PrimRange.Stage(prepared_stage, predicate):
        if not prepared_prim.IsA(UsdGeom.Mesh):
            continue
        prepared_uv = UsdGeom.PrimvarsAPI(prepared_prim).GetPrimvar("st")
        if not prepared_uv:
            continue

        prim_path = prepared_prim.GetPath()
        source_prim = source_stage.GetPrimAtPath(prim_path)
        if not source_prim.IsValid():
            raise RuntimeError(
                f"Prepared UV edit has no source-composition prim: {prim_path}"
            )
        source_uv = UsdGeom.PrimvarsAPI(source_prim).GetPrimvar("st")
        prepared_signature = _primvar_semantic_signature(prepared_uv)
        if prepared_signature == _primvar_semantic_signature(source_uv):
            continue
        if _prim_crosses_variant_arc(source_prim):
            raise RuntimeError(
                "Prepared UV edit crosses a variant arc and cannot be authored "
                f"at the composed path without leaking between variants: {prim_path}"
            )

        localized_prim = localized_stage.GetPrimAtPath(prim_path)
        if not localized_prim.IsValid():
            raise RuntimeError(
                f"Prepared UV edit has no localized-composition prim: {prim_path}"
            )
        if localized_prim.IsInstanceProxy():
            raise RuntimeError(
                "Prepared UV edit maps to an instance proxy and cannot be "
                f"authored without breaking instancing: {prim_path}"
            )

        prepared_attr = prepared_uv.GetAttr()
        prepared_value = prepared_attr.Get()
        if prepared_value is None:
            raise RuntimeError(f"Prepared UV edit has no value: {prim_path}")
        localized_uv = UsdGeom.PrimvarsAPI(localized_prim).CreatePrimvar(
            "st",
            prepared_attr.GetTypeName(),
            prepared_uv.GetInterpolation(),
            prepared_uv.GetElementSize(),
        )
        if not localized_uv.Set(prepared_value):
            raise RuntimeError(f"Failed to author prepared UVs at {prim_path}")
        if prepared_uv.IsIndexed():
            if not localized_uv.SetIndices(prepared_uv.GetIndices()):
                raise RuntimeError(
                    f"Failed to author prepared UV indices at {prim_path}"
                )
        else:
            localized_uv.BlockIndices()
        if _primvar_semantic_signature(localized_uv) != prepared_signature:
            raise RuntimeError(f"Prepared UV verification failed at {prim_path}")
        transferred += 1
    return transferred


_APPLY_OUTPUT_CONTEXT_KEYS = (
    "output_usd_paths",
    "output_usdz_path",
    "output_portability",
    "apply_textures_stats",
    "render_output_usd_paths",
    "rendered_image_paths",
    "render_stats",
    "render_diagnostics",
    "render_errors",
    "render_string_texture_localizations",
    "usdz_packaging_failed",
    "usdz_packaging_error",
    "package_diagnostics",
    "usdz_source_portability",
    "usdz_generated_textures_member",
    "usdz_layer_references_localized",
    "usdz_layer_references_localized_count",
    "usdz_layer_references_cleared",
    "usdz_layer_references_cleared_count",
    "usdz_layer_references_missing",
    "usdz_layer_references_missing_count",
    "usdz_mdl_source_assets_cleared",
    "usdz_absolute_dependencies",
    "usdz_absolute_dependency_count",
    "source_usdz_stage_path",
    "source_usdz_extract_root",
    "source_usdz_prepared_edit_layer_paths",
)


def _clear_apply_output_metadata(context: dict[str, Any]) -> None:
    """Forget every prior apply/package result before a rerun can fail."""
    for key in _APPLY_OUTPUT_CONTEXT_KEYS:
        context.pop(key, None)


def _reanchor_stage_metadata_value(
    value: Any,
    *,
    metadata_key: str,
    source_anchor: Path,
    output_path: Path,
) -> Any:
    """Re-anchor every typed asset in one authored stage-metadata value.

    Stage metadata is copied from the localized source layer onto the public
    wrapper.  A relative ``Sdf.AssetPath`` would otherwise change meaning when
    its authoring layer changes.  Only already-localized files below the
    private output tree are accepted; missing, remote, or escaping metadata
    assets fail closed instead of producing a wrapper that depends on the
    original upload.
    """
    if isinstance(value, Sdf.AssetPath):
        raw = value.path
        if not raw:
            return Sdf.AssetPath()
        if _is_unbundleable_asset_path(raw):
            raise RuntimeError(
                f"Stage metadata {metadata_key!r} has an unbundleable asset: {raw}"
            )
        resolved_value = str(value.resolvedPath or "").strip()
        candidate = (
            Path(resolved_value) if resolved_value else source_anchor / Path(raw)
        )
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(output_path.parent.resolve())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Stage metadata {metadata_key!r} has an asset outside the "
                f"localized output tree: {raw}"
            ) from exc
        if not resolved.is_file():
            raise RuntimeError(
                f"Stage metadata {metadata_key!r} asset is not a file: {raw}"
            )
        relative = Path(os.path.relpath(resolved, output_path.parent)).as_posix()
        return Sdf.AssetPath(relative)
    if isinstance(value, Sdf.AssetPathArray):
        return Sdf.AssetPathArray(
            [
                _reanchor_stage_metadata_value(
                    item,
                    metadata_key=metadata_key,
                    source_anchor=source_anchor,
                    output_path=output_path,
                )
                for item in value
            ]
        )
    if isinstance(value, dict):
        return {
            key: _reanchor_stage_metadata_value(
                item,
                metadata_key=metadata_key,
                source_anchor=source_anchor,
                output_path=output_path,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _reanchor_stage_metadata_value(
                item,
                metadata_key=metadata_key,
                source_anchor=source_anchor,
                output_path=output_path,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _reanchor_stage_metadata_value(
                item,
                metadata_key=metadata_key,
                source_anchor=source_anchor,
                output_path=output_path,
            )
            for item in value
        )
    return value


def _stage_metadata_values_match(
    expected: Any,
    observed: Any,
    *,
    allowed_root: Path,
    exact_asset_paths: bool,
) -> bool:
    """Compare metadata exactly except for proven source-asset relocation."""
    if isinstance(expected, Sdf.AssetPath):
        if not isinstance(observed, Sdf.AssetPath):
            return False
        if not expected.path:
            return not observed.path
        if (
            not observed.path
            or Path(observed.path).is_absolute()
            or _is_unbundleable_asset_path(observed.path)
        ):
            return False
        resolved_value = str(observed.resolvedPath or "").strip()
        if not resolved_value:
            return False
        try:
            resolved = Path(resolved_value).resolve(strict=True)
            resolved.relative_to(allowed_root.resolve())
        except (OSError, ValueError):
            return False
        if not resolved.is_file():
            return False
        if exact_asset_paths:
            return bool(expected.path == observed.path)

        expected_resolved_value = str(expected.resolvedPath or "").strip()
        if expected_resolved_value:
            expected_resolved = Path(expected_resolved_value)
            if expected_resolved.is_file():
                return filecmp.cmp(expected_resolved, resolved, shallow=False)

        # A package-backed source asset resolves to ``archive.usdz[member]``
        # and cannot be compared through pathlib after extraction.  Match the
        # exact package member path, not the often-basename-only authored path.
        if not Ar.IsPackageRelativePath(expected_resolved_value):
            return False
        _package_path, member_path = Ar.SplitPackageRelativePathOuter(
            expected_resolved_value
        )
        if not member_path or Ar.IsPackageRelativePath(member_path):
            return False
        expected_parts = Path(member_path.replace("\\", "/")).parts
        observed_parts = resolved.relative_to(allowed_root.resolve()).parts
        return observed_parts[-len(expected_parts) :] == expected_parts
    if isinstance(expected, Sdf.AssetPathArray):
        return (
            isinstance(observed, Sdf.AssetPathArray)
            and len(expected) == len(observed)
            and all(
                _stage_metadata_values_match(
                    expected_item,
                    observed_item,
                    allowed_root=allowed_root,
                    exact_asset_paths=exact_asset_paths,
                )
                for expected_item, observed_item in zip(expected, observed, strict=True)
            )
        )
    if isinstance(expected, dict):
        return (
            isinstance(observed, dict)
            and expected.keys() == observed.keys()
            and all(
                _stage_metadata_values_match(
                    expected[key],
                    observed[key],
                    allowed_root=allowed_root,
                    exact_asset_paths=exact_asset_paths,
                )
                for key in expected
            )
        )
    if isinstance(expected, list | tuple):
        return (
            isinstance(observed, type(expected))
            and len(expected) == len(observed)
            and all(
                _stage_metadata_values_match(
                    expected_item,
                    observed_item,
                    allowed_root=allowed_root,
                    exact_asset_paths=exact_asset_paths,
                )
                for expected_item, observed_item in zip(expected, observed, strict=True)
            )
        )
    return bool(observed == expected)


def _invalidate_previous_output_tree(output_dir: Path) -> None:
    """Remove output that may reference textures overwritten by this run."""
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise RuntimeError(f"Texture output path is not a directory: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def _write_composed_entrypoint(
    composed_root: Path,
    output_path: Path,
    source_snapshot: _CompositionSnapshot,
) -> tuple[tuple[str, Any], ...]:
    """Write the stable public root without moving the localized source root."""
    wrapper_stage = Usd.Stage.CreateNew(str(output_path))
    if not wrapper_stage:
        raise RuntimeError(f"Failed to create textured output layer: {output_path}")
    wrapper = wrapper_stage.GetRootLayer()
    relative_root = Path(os.path.relpath(composed_root, output_path.parent)).as_posix()
    wrapper.subLayerPaths = [relative_root]
    default_prim_path = source_snapshot.default_prim_path
    if default_prim_path:
        default_path = Sdf.Path(default_prim_path)
        wrapper.defaultPrim = (
            default_path.name if default_path.IsRootPrimPath() else str(default_path)
        )
    composed_stage = Usd.Stage.Open(str(composed_root))
    if not composed_stage:
        raise RuntimeError(
            f"Failed to read localized stage metadata from {composed_root}"
        )
    localized_metadata = composed_stage.GetPseudoRoot().GetAllAuthoredMetadata()
    wrapper_metadata: list[tuple[str, Any]] = []
    for key, _expected in source_snapshot.stage_metadata:
        if key not in localized_metadata:
            raise RuntimeError(
                f"Localized source lost authored stage metadata {key!r}: "
                f"{composed_root}"
            )
        value = _reanchor_stage_metadata_value(
            localized_metadata[key],
            metadata_key=key,
            source_anchor=composed_root.parent,
            output_path=output_path,
        )
        if not wrapper_stage.SetMetadata(key, value):
            raise RuntimeError(
                f"Failed to preserve stage metadata {key!r} in {output_path}"
            )
        wrapper_metadata.append((key, value))
    if not wrapper.Save():
        raise RuntimeError(f"Failed to save textured output layer: {output_path}")
    return tuple(wrapper_metadata)


def _publish_output_tree(
    staging_dir: Path,
    output_dir: Path,
    *,
    validate: Callable[[Path], None] | None = None,
) -> None:
    """Atomically replace and validate the output, restoring it on failure.

    The private staging directory and public output directory are siblings.
    Layer and texture paths are authored before publication, so retaining the
    same parent depth is required for those relative paths to keep their
    meaning after the directory rename.
    """
    if staging_dir.parent.resolve() != output_dir.parent.resolve():
        raise RuntimeError(
            "Texture staging and output directories must be siblings: "
            f"{staging_dir} and {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise RuntimeError(f"Texture output path is not a directory: {output_dir}")

    backup_dir: Path | None = None
    if output_dir.exists():
        backup_dir = Path(
            tempfile.mkdtemp(
                dir=output_dir.parent,
                prefix=f".{output_dir.name}.backup-",
            )
        )
        backup_dir.rmdir()
        os.replace(output_dir, backup_dir)

    try:
        os.replace(staging_dir, output_dir)
        if validate is not None:
            validate(output_dir)
    except Exception:
        if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
            output_dir.unlink(missing_ok=True)
        elif output_dir.exists():
            shutil.rmtree(output_dir)
        if backup_dir is not None and backup_dir.exists():
            os.replace(backup_dir, output_dir)
        raise

    if backup_dir is not None:
        try:
            shutil.rmtree(backup_dir)
        except OSError as exc:  # pragma: no cover - cleanup after committed publish
            logger.warning("Failed to remove prior texture output tree: %s", exc)


def _validate_composed_output(
    output_path: Path,
    source_snapshot: _CompositionSnapshot,
    *,
    allowed_root: Path,
    allowed_deinstanced_paths: set[str] | None = None,
    expected_stage_metadata: tuple[tuple[str, Any], ...] | None = None,
) -> None:
    """Fail closed when relocation loses prims, instances, or dependencies."""
    output_stage = Usd.Stage.Open(str(output_path))
    if not output_stage:
        raise RuntimeError(f"Textured output could not be opened: {output_path}")
    output_snapshot = _composition_snapshot(output_stage)

    for label in ("prim_paths", "mesh_paths", "instanceable_paths"):
        expected = getattr(source_snapshot, label)
        observed = getattr(output_snapshot, label)
        missing = sorted(expected - observed)
        if label == "instanceable_paths" and allowed_deinstanced_paths:
            missing = sorted(set(missing) - allowed_deinstanced_paths)
        if missing:
            raise RuntimeError(
                f"Textured output lost composed {label.replace('_', ' ')}: "
                + ", ".join(missing[:10])
            )

    expected_default = source_snapshot.default_prim_path
    if expected_default and output_snapshot.default_prim_path != expected_default:
        raise RuntimeError(
            "Textured output changed the source default prim: "
            f"expected {expected_default}, got "
            f"{output_snapshot.default_prim_path or '<none>'}"
        )
    metadata_expectations = (
        source_snapshot.stage_metadata
        if expected_stage_metadata is None
        else expected_stage_metadata
    )
    for key, expected in metadata_expectations:
        if not output_stage.HasAuthoredMetadata(key):
            raise RuntimeError(f"Textured output lost source stage metadata: {key}")
        observed = output_stage.GetMetadata(key)
        if not _stage_metadata_values_match(
            expected,
            observed,
            allowed_root=allowed_root,
            exact_asset_paths=expected_stage_metadata is not None,
        ):
            raise RuntimeError(
                "Textured output changed source stage metadata: "
                f"{key} expected {expected!r}, got {observed!r}"
            )

    dependency_layers, dependency_assets, unresolved = UsdUtils.ComputeAllDependencies(
        str(output_path)
    )
    allowed_root = allowed_root.resolve()
    escaped: list[str] = []
    for layer in dependency_layers:
        identifier = str(layer.realPath or layer.identifier)
        if not identifier or identifier.startswith("anon:"):
            continue
        try:
            Path(identifier).resolve().relative_to(allowed_root)
        except (OSError, ValueError):
            escaped.append(identifier)
    for asset in dependency_assets:
        value = str(asset)
        if not value or _is_unbundleable_asset_path(value):
            continue
        try:
            Path(value).resolve().relative_to(allowed_root)
        except (OSError, ValueError):
            escaped.append(value)
    if escaped:
        raise RuntimeError(
            "Textured output retains dependencies outside the run directory: "
            + ", ".join(sorted(set(escaped))[:10])
        )

    unresolved_layers = sorted(
        value
        for value in (str(item) for item in unresolved)
        if Path(value).suffix.lower() in _USD_LAYER_SUFFIXES
    )
    if unresolved_layers:
        raise RuntimeError(
            "Textured output has unresolved USD composition dependencies: "
            + ", ".join(unresolved_layers[:10])
        )


def _require_portable_output(portability: dict[str, Any]) -> None:
    """Raise with compact diagnostic codes when an output is not portable."""
    if portability.get("portable", False):
        return
    diagnostics = portability.get("diagnostics") or []
    codes = sorted(
        {
            str(item.get("code"))
            for item in diagnostics
            if isinstance(item, dict) and item.get("code")
        }
    )
    raise RuntimeError(
        "Textured output failed portability validation"
        + (f": {', '.join(codes)}" if codes else "")
    )


class ApplyTexturesTask(Task):
    """Set PBR texture file paths on OpenPBR materials in the USD stage.

    In legacy per-material mode: applies textures directly to shared materials.
    For an immutable plan, clones once and binds only the plan's exact members.
    In per-prim mode: clones materials so each prim gets its own texture,
    then re-binds each geometry prim to its cloned material.

    For materials whose Material prim has an MDL Shader child (SimReady /
    OmniPBR), the task also overrides the well-known MDL texture inputs
    (`diffuse_texture`, `normalmap_texture`, `ORM_texture`,
    `reflectionroughness_texture`, `metallic_texture`, plus aliases) with
    the freshly generated local textures, and clears any unmapped
    `*_texture` input that points at an unbundleable URI (`omniverse://`,
    `http(s)://`, …) so the output USD does not survive into the
    downloaded bundle with refs the asset resolver cannot satisfy. Local
    relative/absolute paths on unmapped inputs are preserved.

    Context keys read:
        usd_path (str): Input USD file path.
        source_usd_path (str, optional): Immutable authored input used to
            preserve composition when ``usd_path`` is a flattened UV-prepared
            working layer.
        blended_textures (dict[str, BlendedTextures]): From BlendTexturesTask.
        prim_texture_units (list[PrimTextureUnit]): From DiscoverMaterialsTask.
        working_dir (str): Working directory.
        usd_dependency_root (str, optional): Trusted input-bundle root for raw
            layered USD files whose relative dependencies are siblings of the
            root layer directory. Defaults to the root layer's parent.
        service_managed_usdz_reconstruction (bool, optional): Internal service
            contract that defers uploaded USDZ reconstruction to packaging.

    Context keys written:
        output_usd_paths (list[str]): Paths to output USD files.
        apply_textures_stats (dict): Summary of MDL-override activity:
            ``applied_count`` (int), ``mdl_inputs_overridden`` (int),
            ``mdl_inputs_cleared`` (list of ``"<mat_path>:<input_name>"``
            strings — unbundleable URI paths or unresolvable local refs that
            were blanked), and ``mdl_inputs_localized`` (list of strings —
            resolvable local refs that were copied into
            ``working_dir/textures`` so the bundle's path-rewrite step
            keeps them packageable). Consumed by the texture-agent service
            to surface a per-step warning in ``/status`` / ``/results``.
            The stats also include ``preview_texture_inputs_overridden``:
            connected `UsdUVTexture` nodes in existing UsdPreviewSurface
            fallback graphs that were pointed at generated maps.
    """

    def __init__(self) -> None:
        self.name = "ApplyTextures"
        self.description = "Apply PBR texture maps to materials in USD"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        """Apply textures and always discard unpublished private staging trees."""
        staging_dirs: list[Path] = []
        try:
            return self._run(context, object_store, staging_dirs)
        finally:
            for staging_dir in staging_dirs:
                shutil.rmtree(staging_dir, ignore_errors=True)

    def _run(
        self,
        context: dict[str, Any],
        object_store: Any,
        staging_dirs: list[Path],
    ) -> dict[str, Any]:
        usd_path = context["usd_path"]
        blended: dict[str, BlendedTextures] = context.get("blended_textures", {})
        units: list[PrimTextureUnit] = context.get("prim_texture_units", [])
        working_dir = Path(context["working_dir"])
        source_value = context.get("source_usd_path")
        source_candidate = (
            Path(source_value)
            if isinstance(source_value, str) and source_value.strip()
            else Path(usd_path)
        )
        dependency_root = _validated_dependency_root(context)
        if dependency_root is not None:
            _validate_authoritative_source_root(source_candidate, dependency_root)
        # Validate every root that can authorize a source texture before stale
        # result metadata or the prior public output tree is changed.
        _allowed_texture_source_roots(usd_path, working_dir, context)
        if Path(os.path.abspath(source_candidate)) != Path(os.path.abspath(usd_path)):
            _allowed_texture_source_roots(str(source_candidate), working_dir, context)
        cached_apply = is_cached_apply_context(context)
        _clear_apply_output_metadata(context)

        if cached_apply:
            cached = _load_cached_blended_textures(working_dir, units)
            blended = {**cached, **blended}
            if cached:
                logger.info(
                    "Loaded %d cached blended texture sets from %s",
                    len(cached),
                    working_dir / "textures",
                )
                context["blended_textures"] = blended
            if not units:
                raise RuntimeError(
                    "Cached apply reconstructed no texture units; refusing to "
                    "produce an untextured output"
                )
            missing = _missing_cached_blended_artifacts(blended, units)
            if missing:
                raise RuntimeError(
                    "Cached apply requires complete albedo, normal, and ORM maps "
                    "for every texture unit; missing: " + ", ".join(missing)
                )
        elif not blended and context.get("resume"):
            blended = _load_cached_blended_textures(working_dir, units)
            if blended:
                logger.info(
                    "Loaded %d cached blended texture sets from %s",
                    len(blended),
                    working_dir / "textures",
                )
                context["blended_textures"] = blended

        if not blended:
            logger.info("No blended textures to apply")
            context["output_usd_paths"] = []
            return context

        working_dir.mkdir(parents=True, exist_ok=True)
        out_dir = working_dir / "output"
        output_usd_path = out_dir / "textured_output.usd"
        # Generated/localized maps live beside ``output`` and may already have
        # replaced files used by a prior result. Keeping or restoring that old
        # USD tree would expose a silently mixed artifact after any failure.
        _invalidate_previous_output_tree(out_dir)

        prepared_path = Path(usd_path)
        prepared_stage = Usd.Stage.Open(str(prepared_path))
        if not prepared_stage:
            raise FileNotFoundError(f"Failed to open USD stage: {usd_path}")
        source_path = prepared_path
        source_stage = prepared_stage
        prepared_from_source = source_candidate.resolve() != prepared_path.resolve()
        service_managed_usdz = bool(context.get("service_managed_usdz_reconstruction"))
        defer_source_usdz = (
            service_managed_usdz and source_candidate.suffix.lower() == ".usdz"
        )
        restore_immutable_source = prepared_from_source and not defer_source_usdz
        if restore_immutable_source:
            candidate_stage = Usd.Stage.Open(str(source_candidate))
            if not candidate_stage:
                raise FileNotFoundError(
                    f"Failed to open original USD stage: {source_candidate}"
                )
            source_path = source_candidate
            source_stage = candidate_stage
        source_snapshot = _composition_snapshot(source_stage)

        stage = source_stage
        edit_output_path = output_usd_path
        localized_root: Path | None = None
        localized_staging_dir: Path | None = None
        task_deinstanced_paths: set[str] = set()
        # The service explicitly reconstructs an uploaded USDZ immediately
        # after this task. Every other layered input, including a direct-CLI
        # USDZ, must carry its dependency tree with the apply artifact itself.
        source_has_file_backed_composition = (
            source_path.suffix.lower() != ".usdz"
            and _stage_has_file_backed_composition(source_stage)
        )
        requires_localization = (
            (source_path.suffix.lower() == ".usdz" and not defer_source_usdz)
            or restore_immutable_source
            or source_has_file_backed_composition
        )
        if requires_localization:
            localized_staging_dir = Path(
                tempfile.mkdtemp(
                    dir=working_dir,
                    prefix=".texture-output-stage-",
                )
            )
            staging_dirs.append(localized_staging_dir)
            try:
                localized_root = _stage_composed_source(
                    source_path,
                    localized_staging_dir,
                    dependency_root=dependency_root,
                    root_layer_at_staging_root=(
                        restore_immutable_source
                        and source_path.suffix.lower() != ".usdz"
                        and not source_has_file_backed_composition
                    ),
                )
            except Exception:
                shutil.rmtree(localized_staging_dir, ignore_errors=True)
                raise
            stage = Usd.Stage.Open(str(localized_root))
            if not stage:
                shutil.rmtree(localized_staging_dir, ignore_errors=True)
                raise RuntimeError(
                    f"Localized layered USD could not be opened: {localized_root}"
                )
            # Prove localization itself preserved composition and instance
            # semantics before texture authoring is allowed to make any
            # intentional, narrowly tracked deinstancing changes.
            try:
                _validate_composed_output(
                    localized_root,
                    source_snapshot,
                    allowed_root=working_dir,
                )
            except Exception:
                shutil.rmtree(localized_staging_dir, ignore_errors=True)
                raise
            if prepared_path.resolve() != source_path.resolve():
                _transfer_prepared_uv_deltas(
                    prepared_stage,
                    source_stage,
                    stage,
                )
            # Texture paths are authored relative to the layer that owns the
            # opinion, not the final canonical wrapper. Keeping the real
            # authoring anchor preserves paths from nested source layers.
            edit_output_path = localized_root
        else:
            out_dir.mkdir(parents=True, exist_ok=True)

        # Group by canonical material path, not display name. Distinct Looks
        # scopes commonly contain materials with the same leaf name, and
        # grouping those together would incorrectly enter the per-prim clone
        # path and apply both units to whichever material appeared first.
        units_by_material: dict[str, list[PrimTextureUnit]] = defaultdict(list)
        for unit in units:
            if unit.key in blended:
                units_by_material[unit.material_info.prim_path].append(unit)
        plan_units_by_id = _accepted_plan_units_by_id(context)
        _require_nonvariant_apply_plan(stage, units_by_material, plan_units_by_id)
        _preflight_stage_texture_reference_localization(
            stage,
            usd_path=str(source_path),
            working_dir=working_dir,
            output_usd_path=edit_output_path,
            context=context,
        )
        allowed_source_roots = _allowed_texture_source_roots(
            str(source_path),
            working_dir,
            context,
        )

        applied_count = 0
        mdl_inputs_overridden = 0
        mdl_inputs_cleared: list[str] = []
        mdl_inputs_localized: list[str] = []
        stage_texture_refs_localized: list[str] = []
        preview_texture_inputs_overridden: list[str] = []

        for _mat_name, mat_units in units_by_material.items():
            mat = mat_units[0].material_info

            plan_unit = (
                _accepted_plan_unit_for_runtime_unit(plan_units_by_id, mat_units[0])
                if len(mat_units) == 1
                else None
            )
            planned_target_paths = _planned_binding_target_paths(plan_unit)
            if (
                len(mat_units) == 1
                and not mat_units[0].prim_path
                and not planned_target_paths
            ):
                # Per-material units without exact binding members author the
                # selected material itself. This covers both legacy runs and
                # explicit unbound-material plans without widening scope.
                unit = mat_units[0]
                apply_paths = _material_apply_paths(mat)
                for mat_path in apply_paths:
                    overridden, cleared, localized, preview_overridden = (
                        _apply_pbr_textures(
                            stage,
                            mat_path,
                            blended[unit.key],
                            working_dir,
                            unit.key,
                            str(source_path),
                            edit_output_path,
                            deinstanced_paths=task_deinstanced_paths,
                            allowed_source_roots=allowed_source_roots,
                        )
                    )
                    mdl_inputs_overridden += overridden
                    mdl_inputs_cleared.extend(cleared)
                    mdl_inputs_localized.extend(localized)
                    preview_texture_inputs_overridden.extend(preview_overridden)
                logger.info(
                    "Applied textures to %s (direct, %d material path%s)",
                    unit.key,
                    len(apply_paths),
                    "" if len(apply_paths) == 1 else "s",
                )
                applied_count += 1

            else:
                # Per-prim mode clones per unit.  A planned per-material unit
                # also clones once, then binds only its exact accepted members;
                # editing the shared source material would spill to non-targets.
                for unit in mat_units:
                    selected_plan_unit = _accepted_plan_unit_for_runtime_unit(
                        plan_units_by_id, unit
                    )
                    if selected_plan_unit is not None:
                        target_prim_paths = _planned_binding_target_paths(
                            selected_plan_unit
                        )
                    elif unit.prim_path:
                        target_prim_paths = (unit.prim_path,)
                    else:  # pragma: no cover - guarded by the direct branch
                        target_prim_paths = ()
                    if not target_prim_paths:
                        raise RuntimeError(
                            f"Texture apply unit {unit.key} has no exact prim target"
                        )
                    existing_target_paths = tuple(
                        path
                        for path in target_prim_paths
                        if stage.GetPrimAtPath(path).IsValid()
                    )
                    if len(existing_target_paths) != len(target_prim_paths):
                        if selected_plan_unit is not None:
                            raise RuntimeError(
                                f"Texture apply unit {unit.key} lost an exact target"
                            )
                        for missing_path in sorted(
                            set(target_prim_paths) - set(existing_target_paths)
                        ):
                            logger.warning(
                                "Prim not found for rebinding: %s", missing_path
                            )
                    target_prim_paths = existing_target_paths
                    if not target_prim_paths:
                        continue
                    clone_name = unit.key
                    clone_source_path = _material_clone_source_path(stage, mat)
                    clone_path = _clone_material(
                        stage,
                        clone_source_path,
                        clone_name,
                    )

                    # Apply textures to the clone
                    overridden, cleared, localized, preview_overridden = (
                        _apply_pbr_textures(
                            stage,
                            clone_path,
                            blended[unit.key],
                            working_dir,
                            unit.key,
                            str(source_path),
                            edit_output_path,
                            deinstanced_paths=task_deinstanced_paths,
                            allowed_source_roots=allowed_source_roots,
                        )
                    )
                    mdl_inputs_overridden += overridden
                    mdl_inputs_cleared.extend(cleared)
                    mdl_inputs_localized.extend(localized)
                    preview_texture_inputs_overridden.extend(preview_overridden)

                    # Re-bind only exact accepted geometry members to the clone.
                    bound_target_count = 0
                    for target_prim_path in target_prim_paths:
                        geom_prim = _editable_prim_for_path(
                            stage,
                            target_prim_path,
                            deinstanced_paths=task_deinstanced_paths,
                        )
                        if not geom_prim.IsValid() or geom_prim.IsInstanceProxy():
                            raise RuntimeError(
                                f"Texture apply unit {unit.key} cannot author exact "
                                f"target {target_prim_path}"
                            )
                        _require_nonvariant_edit(
                            geom_prim,
                            purpose="Material binding",
                        )
                        _bind_material_or_raise(stage, geom_prim, clone_path)
                        bound_target_count += 1
                        logger.info(
                            "Applied textures to %s (cloned, bound %s)",
                            unit.key,
                            target_prim_path,
                        )

                    if bound_target_count != len(target_prim_paths):
                        raise RuntimeError(
                            f"Texture apply unit {unit.key} did not bind every target"
                        )
                    applied_count += 1

        stage_texture_refs_localized = _localize_stage_texture_references(
            stage,
            usd_path=str(source_path),
            working_dir=working_dir,
            output_usd_path=edit_output_path,
            context=context,
            deinstanced_paths=task_deinstanced_paths,
        )

        localized_portability: dict[str, Any] | None = None
        if localized_root is not None and localized_staging_dir is not None:
            staged_output_path = localized_staging_dir / "textured_output.usd"
            try:
                if not stage.GetRootLayer().Save():
                    raise RuntimeError(
                        f"Failed to save localized layered USD: {localized_root}"
                    )
                wrapper_stage_metadata = _write_composed_entrypoint(
                    localized_root,
                    staged_output_path,
                    source_snapshot,
                )
                _validate_composed_output(
                    staged_output_path,
                    source_snapshot,
                    allowed_root=working_dir,
                    allowed_deinstanced_paths=task_deinstanced_paths,
                    expected_stage_metadata=wrapper_stage_metadata,
                )
                localized_portability = validate_output_texture_portability(
                    staged_output_path,
                    bundle_root=working_dir,
                )
                _require_portable_output(localized_portability)

                def _validate_published_output(published_dir: Path) -> None:
                    nonlocal localized_portability
                    published_output = published_dir / "textured_output.usd"
                    _validate_composed_output(
                        published_output,
                        source_snapshot,
                        allowed_root=working_dir,
                        allowed_deinstanced_paths=task_deinstanced_paths,
                        expected_stage_metadata=wrapper_stage_metadata,
                    )
                    localized_portability = validate_output_texture_portability(
                        published_output,
                        bundle_root=working_dir,
                    )
                    _require_portable_output(localized_portability)

                _publish_output_tree(
                    localized_staging_dir,
                    out_dir,
                    validate=_validate_published_output,
                )
                staging_dirs.remove(localized_staging_dir)
            except Exception:
                shutil.rmtree(localized_staging_dir, ignore_errors=True)
                raise
        elif not stage.GetRootLayer().Export(str(output_usd_path)):
            raise RuntimeError(f"Failed to export textured USD: {output_usd_path}")
        logger.info(
            "Applied PBR textures to %d units, saved to %s",
            applied_count,
            output_usd_path,
        )
        if mdl_inputs_overridden:
            logger.info(
                "Overrode %d pre-baked MDL texture inputs with new local textures",
                mdl_inputs_overridden,
            )
        if mdl_inputs_cleared:
            logger.warning(
                "Cleared %d MDL texture inputs that had no matching generated "
                "channel (would have produced broken refs after bundle "
                "rewriting): %s",
                len(mdl_inputs_cleared),
                ", ".join(mdl_inputs_cleared),
            )
        if mdl_inputs_localized:
            logger.info(
                "Localized %d MDL texture inputs (copied existing local refs into "
                "the bundle textures dir): %s",
                len(mdl_inputs_localized),
                ", ".join(mdl_inputs_localized),
            )
        if stage_texture_refs_localized:
            logger.info(
                "Localized %d stage texture references for portable output: %s",
                len(stage_texture_refs_localized),
                ", ".join(stage_texture_refs_localized),
            )
        if preview_texture_inputs_overridden:
            logger.info(
                "Overrode %d UsdPreviewSurface texture inputs with generated maps",
                len(preview_texture_inputs_overridden),
            )

        output_portability = localized_portability or (
            validate_output_texture_portability(output_usd_path)
        )

        context["output_usd_paths"] = [str(output_usd_path)]
        context["output_portability"] = output_portability
        context["apply_textures_stats"] = {
            "applied_count": applied_count,
            "mdl_inputs_overridden": mdl_inputs_overridden,
            "mdl_inputs_cleared": mdl_inputs_cleared,
            "mdl_inputs_localized": mdl_inputs_localized,
            "stage_texture_refs_localized": stage_texture_refs_localized,
            "preview_texture_inputs_overridden": preview_texture_inputs_overridden,
        }
        return context
