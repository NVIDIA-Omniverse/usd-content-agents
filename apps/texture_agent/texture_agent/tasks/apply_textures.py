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
from collections import defaultdict
from pathlib import Path
from typing import Any

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from world_understanding.agentic.tasks import Task

from texture_agent.functions.artifact_manifest import (
    validate_output_texture_portability,
)
from texture_agent.functions.cached_apply import (
    is_cached_apply_context,
    is_valid_cached_texture_png,
)
from texture_agent.functions.material_discovery import MaterialInfo, PrimTextureUnit
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
    parent_path = str(Sdf.Path(source_mat_path).GetParentPath())
    clone_path = f"{parent_path}/{clone_name}"

    layer = stage.GetRootLayer()
    Sdf.CopySpec(layer, source_mat_path, layer, clone_path)

    logger.debug("Cloned material: %s -> %s", source_mat_path, clone_path)
    return clone_path


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


def _editable_prim_for_path(stage: Usd.Stage, prim_path: str) -> Usd.Prim:
    """Return a prim that can accept authored properties at ``prim_path``."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return prim
    if not (prim.IsInstanceProxy() or prim.IsInstance() or prim.IsInstanceable()):
        return prim

    cursor = prim
    while cursor.IsValid() and not cursor.GetPath().IsAbsoluteRootPath():
        if cursor.IsInstance() or cursor.IsInstanceable():
            cursor.SetInstanceable(False)
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
        return parent.GetTypeName() == ""

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


def _author_texture_reference(texture_path: str, output_usd_path: Path) -> str:
    """Return the texture path to author into the output USD layer."""
    if not texture_path or _is_unbundleable_asset_path(texture_path):
        return texture_path
    path = Path(texture_path)
    if not path.is_absolute():
        return texture_path.replace("\\", "/")
    try:
        path.resolve().relative_to(output_usd_path.parent.parent.resolve())
        return Path(os.path.relpath(path, output_usd_path.parent)).as_posix()
    except (OSError, ValueError):
        return texture_path


def _is_portable_texture_reference(raw: str, output_usd_path: Path) -> bool:
    """Return whether an authored texture ref already resolves inside the run."""
    if not raw or _is_unbundleable_asset_path(raw) or Path(raw).is_absolute():
        return False
    try:
        resolved = (output_usd_path.parent / raw).resolve()
        resolved.relative_to(output_usd_path.parent.parent.resolve())
    except (OSError, ValueError):
        return False
    return resolved.is_file()


def _allowed_texture_source_roots(
    usd_path: str,
    working_dir: Path,
    context: dict[str, Any],
) -> list[Path]:
    roots = [Path(usd_path).resolve().parent, working_dir.resolve()]
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
                roots.append(Path(input_usd).resolve().parent)

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


def _localize_stage_texture_references(
    stage: Usd.Stage,
    *,
    usd_path: str,
    working_dir: Path,
    output_usd_path: Path,
    context: dict[str, Any],
) -> list[str]:
    """Rewrite all local PNG refs to bundle-local sibling-relative paths.

    Scoped texture edits leave unedited materials untouched. For SimReady assets
    those untouched materials can still point back to the original source
    package, which renders locally but fails the downloadable package
    portability check. This pass copies each existing local PNG dependency once
    into ``working_dir/textures`` and authors a path relative to the output USD.
    """
    tex_dir = working_dir / "textures"
    bundle_root = output_usd_path.parent.parent.resolve()
    allowed_roots = _allowed_texture_source_roots(usd_path, working_dir, context)
    localized: list[str] = []

    for prim in stage.Traverse():
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
            if _is_portable_texture_reference(raw, output_usd_path):
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
                target = candidate
            except ValueError:
                if not _is_under_any_root(candidate, allowed_roots):
                    continue
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

            authored = _author_texture_reference(str(target), output_usd_path)
            try:
                if prim.IsInstance() or prim.IsInstanceable():
                    prim.SetInstanceable(False)
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
    upload_root: Path,
    tex_dir: Path,
    mat_name: str,
    input_name: str,
) -> str | None:
    """Copy an already-resolved local asset into the bundle textures dir.

    Security: USD content can come from untrusted uploads, so we refuse to
    localize anything that resolves outside the upload root. Without this
    scope check a crafted MDL input like ``inputs:leak_texture = @/etc/passwd@``
    would copy a host file into ``working_dir/textures/`` — which the service
    exposes as a downloadable artifact. We additionally require the file to
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

    # Reject anything outside the upload root — this is the trust boundary in
    # the service path. ``resolve()`` already followed symlinks before we got
    # here, so an in-upload symlink pointing at ``/etc/passwd`` would resolve
    # outside ``upload_root`` and be rejected here.
    try:
        upload_resolved = upload_root.resolve()
    except (OSError, ValueError):
        return None
    try:
        candidate.relative_to(upload_resolved)
    except ValueError:
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
      input USD or absolute) are copied into ``working_dir/textures`` under a
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
                _localize_asset(candidate, upload_root, tex_dir, mat_name, base)
                if candidate is not None
                else None
            )
            if copied is None:
                if _safe_set_typed_value(inp, type_name, ""):
                    cleared.append(f"{mat_path}:{base}")
            else:
                copied_ref = _author_texture_reference(copied, output_usd_path)
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
            if inp.GetConnectedSource() is not None:
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

        # Preserve every authored connection, including custom and external
        # sources, while filling any remaining generated channels. Treating a
        # partially connected shader as all-or-nothing leaves generated maps
        # inactive and prevents later runs from enriching an albedo-only graph.
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
    prim = _editable_prim_for_path(stage, mat_path)
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

    # Ensure the material container is a typed Scope for NVCF traversal and
    # usd-validation-nvidia's Basic TypeChecker.
    parent = prim.GetParent()
    # Scope.Define authors a prim spec, so keep it out of read-only composition
    # contexts such as pseudo-root, instance proxies, and prototype contents.
    if _can_define_parent_scope(stage, parent):
        if parent.IsInstanceable():
            parent.SetInstanceable(False)
        UsdGeom.Scope.Define(stage, parent.GetPath())

    albedo_ref = _author_texture_reference(textures.albedo, output_usd_path)
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
        normal_ref = _author_texture_reference(textures.normal, output_usd_path)
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

        channel_paths["orm"] = _author_texture_reference(textures.orm, output_usd_path)

        with Image.open(textures.orm) as orm_img:
            orm_arr = np.array(orm_img)
        tex_dir = working_dir / "textures"

        roughness_arr = orm_arr[:, :, 1]
        roughness_path = tex_dir / f"{key}_roughness.png"
        Image.fromarray(roughness_arr).save(str(roughness_path))
        roughness_ref = _author_texture_reference(str(roughness_path), output_usd_path)
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
        metalness_ref = _author_texture_reference(str(metalness_path), output_usd_path)
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
            stage, mat_path, channel_paths, usd_path, working_dir, output_usd_path
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


class ApplyTexturesTask(Task):
    """Set PBR texture file paths on OpenPBR materials in the USD stage.

    In per-material mode: applies textures directly to shared materials.
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
        blended_textures (dict[str, BlendedTextures]): From BlendTexturesTask.
        prim_texture_units (list[PrimTextureUnit]): From DiscoverMaterialsTask.
        working_dir (str): Working directory.

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
        usd_path = context["usd_path"]
        blended: dict[str, BlendedTextures] = context.get("blended_textures", {})
        units: list[PrimTextureUnit] = context.get("prim_texture_units", [])
        working_dir = Path(context["working_dir"])
        cached_apply = is_cached_apply_context(context)

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

        out_dir = working_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_usd_path = out_dir / "textured_output.usd"

        stage = Usd.Stage.Open(str(usd_path))
        if not stage:
            raise FileNotFoundError(f"Failed to open USD stage: {usd_path}")

        # Group by canonical material path, not display name. Distinct Looks
        # scopes commonly contain materials with the same leaf name, and
        # grouping those together would incorrectly enter the per-prim clone
        # path and apply both units to whichever material appeared first.
        units_by_material: dict[str, list[PrimTextureUnit]] = defaultdict(list)
        for unit in units:
            if unit.key in blended:
                units_by_material[unit.material_info.prim_path].append(unit)

        applied_count = 0
        mdl_inputs_overridden = 0
        mdl_inputs_cleared: list[str] = []
        mdl_inputs_localized: list[str] = []
        stage_texture_refs_localized: list[str] = []
        preview_texture_inputs_overridden: list[str] = []

        for _mat_name, mat_units in units_by_material.items():
            mat = mat_units[0].material_info

            if len(mat_units) == 1 and not mat_units[0].prim_path:
                # Per-material mode (or single prim): apply directly
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
                            usd_path,
                            output_usd_path,
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
                # Per-prim mode: clone material for each prim
                for unit in mat_units:
                    clone_name = unit.key
                    clone_path = _clone_material(stage, mat.prim_path, clone_name)

                    # Apply textures to the clone
                    overridden, cleared, localized, preview_overridden = (
                        _apply_pbr_textures(
                            stage,
                            clone_path,
                            blended[unit.key],
                            working_dir,
                            unit.key,
                            usd_path,
                            output_usd_path,
                        )
                    )
                    mdl_inputs_overridden += overridden
                    mdl_inputs_cleared.extend(cleared)
                    mdl_inputs_localized.extend(localized)
                    preview_texture_inputs_overridden.extend(preview_overridden)

                    # Re-bind the geometry prim to the cloned material
                    if unit.prim_path:
                        geom_prim = stage.GetPrimAtPath(unit.prim_path)
                        if geom_prim.IsValid():
                            binding_api = UsdShade.MaterialBindingAPI.Apply(geom_prim)
                            cloned_mat = UsdShade.Material(
                                stage.GetPrimAtPath(clone_path)
                            )
                            binding_api.Bind(cloned_mat)
                            logger.info(
                                "Applied textures to %s (cloned, bound %s)",
                                unit.key,
                                unit.prim_path,
                            )
                        else:
                            logger.warning(
                                "Prim not found for rebinding: %s",
                                unit.prim_path,
                            )

                    applied_count += 1

        stage_texture_refs_localized = _localize_stage_texture_references(
            stage,
            usd_path=usd_path,
            working_dir=working_dir,
            output_usd_path=output_usd_path,
            context=context,
        )

        stage.GetRootLayer().Export(str(output_usd_path))
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

        context["output_usd_paths"] = [str(output_usd_path)]
        context["output_portability"] = validate_output_texture_portability(
            output_usd_path
        )
        context["apply_textures_stats"] = {
            "applied_count": applied_count,
            "mdl_inputs_overridden": mdl_inputs_overridden,
            "mdl_inputs_cleared": mdl_inputs_cleared,
            "mdl_inputs_localized": mdl_inputs_localized,
            "stage_texture_refs_localized": stage_texture_refs_localized,
            "preview_texture_inputs_overridden": preview_texture_inputs_overridden,
        }
        return context
