# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate materials.yaml manifest and thumbnails from a USD material library.

This module provides the business logic for the ``material-agent generate-manifest``
CLI command.  The high-level workflow is:

1. **Discover** all Material prims in a USD file.
2. **Render** thumbnails via NVCF cloud rendering (compose a template scene with
   camera/lights/sphere, reference each material, bundle MDL/textures, upload to
   S3, render).
3. Optionally **describe** each material using a Vision-Language Model.
4. **Write** ``materials.yaml`` with name, description, binding path, and icon for
   every material.
"""

from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from world_understanding.utils.credentials import redact_sensitive_config

if TYPE_CHECKING:  # pragma: no cover
    from pxr import Usd

log = logging.getLogger(__name__)

_THUMBNAIL_RENDER_FAILURE_MESSAGE = "Thumbnail render failed"
_S3_CLEANUP_FAILURE_MESSAGE = "S3 cleanup failed"
_DESCRIPTION_GENERATION_FAILURE_MESSAGE = "VLM description generation failed"

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_PACKAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_TEMPLATE = (
    _PACKAGE_DIR / ".." / "data" / "templates" / "thumbnail_template.usd"
)

# Template scene prim paths
_TEMPLATE_CAMERA = "/Root/thumbnail_CAM"
_TEMPLATE_SPHERE = "/Root/Sphere"

# S3 configuration (required for thumbnail rendering; set via environment variables)
_S3_BUCKET = os.environ.get("MATERIAL_S3_BUCKET", "")
_S3_PROFILE = os.environ.get("MATERIAL_S3_PROFILE", "")
_S3_REGION = os.environ.get("MATERIAL_S3_REGION", "us-east-2")

# MDL import regexes
_MDL_RELATIVE_IMPORT_RE = re.compile(
    r"(?:using|import)\s+((?:\.\.::)+[\w:]+)", re.MULTILINE
)
_MDL_SIBLING_IMPORT_RE = re.compile(
    r"(?:using|import)\s+([A-Za-z_]\w*)(?:::\S+)?", re.MULTILINE
)


# ---------------------------------------------------------------------------
# Input / Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GenerateManifestInput:
    """Parameters for manifest generation."""

    usd_file: Path
    output_dir: Path
    image_size: int = 256
    skip_existing: bool = False
    library_path: str | None = None
    template: Path = field(default_factory=lambda: _DEFAULT_TEMPLATE)
    max_workers: int = 4
    skip_descriptions: bool = False
    vlm_backend: str = "nim"
    vlm_model: str | None = "qwen/qwen3.5-397b-a17b"
    vlm_workers: int = 8
    list_materials: bool = False
    verbose: bool = False


@dataclass
class GenerateManifestResult:
    """Result of manifest generation."""

    success: bool
    yaml_path: Path | None = None
    materials_count: int = 0
    thumbnails_count: int = 0
    descriptions_count: int = 0
    error: str | None = None
    material_paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1: Discover Material Prims
# ---------------------------------------------------------------------------


def discover_materials(usd_file: Path) -> list[str]:
    """Return all Material prim paths from a USD file."""
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(usd_file))
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {usd_file}")

    materials: list[str] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            materials.append(prim.GetPath().pathString)

    log.info("Discovered %d materials in %s", len(materials), usd_file.name)
    return materials


def prim_path_to_name(prim_path: str) -> str:
    """Convert a prim path to a human-readable name.

    Example: /World/Looks/Aluminum_Brushed -> Aluminum Brushed
    """
    last_segment = prim_path.rsplit("/", 1)[-1]
    return last_segment.replace("_", " ")


def prim_path_to_filename(prim_path: str) -> str:
    """Convert a prim path to a filename-safe string.

    Example: /World/Looks/Aluminum_Brushed -> Aluminum_Brushed
    """
    return prim_path.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Step 2: Render Thumbnails via NVCF
# ---------------------------------------------------------------------------


def _compose_thumbnail_stage(
    template_path: Path,
    usd_file: Path,
    material_prim_path: str,
) -> Usd.Stage:
    """Compose and flatten a thumbnail stage with a material applied to the sphere.

    Creates an in-memory stage that sublayers the template (camera, lights,
    sphere) and references the given material from the user's USD file,
    binding it to the sphere. The stage is then flattened so all composition
    arcs are resolved into a single layer -- this is required because the
    NVCF renderer cannot access local files referenced by sublayers or prims.

    Args:
        template_path: Path to the thumbnail template USD file.
        usd_file: Path to the user's USD material library.
        material_prim_path: Prim path of the material in the user's USD file.

    Returns:
        A flattened Usd.Stage ready for rendering.
    """
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateInMemory()
    root_layer = stage.GetRootLayer()

    # Sublayer the template to bring in camera, lights, sphere geometry
    root_layer.subLayerPaths.append(str(template_path.resolve()))

    # Define the material prim (must be 'def', not 'over', so it survives Flatten)
    mat_name = material_prim_path.rsplit("/", 1)[-1]
    mat_prim_path = Sdf.Path(f"/Materials/{mat_name}")
    mat_prim = stage.DefinePrim(mat_prim_path, "Material")
    mat_prim.GetReferences().AddReference(
        assetPath=str(usd_file.resolve()),
        primPath=material_prim_path,
    )

    # Bind the material to the sphere
    sphere_prim = stage.GetPrimAtPath(_TEMPLATE_SPHERE)
    if not sphere_prim.IsValid():
        raise RuntimeError(f"Sphere prim not found at {_TEMPLATE_SPHERE}")

    binding_api = UsdShade.MaterialBindingAPI.Apply(sphere_prim)
    material = UsdShade.Material(mat_prim)
    binding_api.Bind(material)

    # Flatten so all composition arcs are resolved into one layer.
    # NVCF cannot follow local sublayer/reference paths; flattening
    # inlines everything except external asset refs (MDL, textures)
    # which get handled by the bundling step later.
    flat_layer = stage.Flatten()
    flat_stage = Usd.Stage.Open(flat_layer)

    # Convert custom MDL shaders to built-in equivalents so the NVCF
    # renderer can resolve them (it cannot load custom MDL modules).
    from world_understanding.utils.usd.material import convert_custom_mdl_to_builtin

    convert_custom_mdl_to_builtin(flat_stage)

    return flat_stage


def _collect_mdl_deps(mdl_path: Path, visited: set[Path] | None = None) -> set[Path]:
    """Recursively collect an MDL file and all its transitive local-import dependencies.

    Handles two forms of local imports:
    - Relative parent imports: ``..::..::Templates::GlassWithVolume``
    - Sibling imports: ``import GlassUtils::func`` (looks for GlassUtils.mdl in same dir)
    """
    if visited is None:
        visited = set()
    if mdl_path in visited or not mdl_path.exists():
        return visited
    visited.add(mdl_path)

    try:
        content = mdl_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return visited

    # Strip single-line (//) and block (/* */) comments so the regexes
    # don't match import-like keywords inside comments or strings.
    content = re.sub(r"//[^\n]*", "", content)
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

    current_dir = mdl_path.parent

    # Relative parent imports: ..::..::Templates::Foo
    for match in _MDL_RELATIVE_IMPORT_RE.finditer(content):
        import_str = match.group(1)
        parts = import_str.split("::")
        base = current_dir
        i = 0
        while i < len(parts) and parts[i] == "..":
            base = base.parent
            i += 1
        module_parts = parts[i:]
        if not module_parts:  # pragma: no cover
            continue
        candidate = base.joinpath(*module_parts[:-1]) / f"{module_parts[-1]}.mdl"
        _collect_mdl_deps(candidate, visited)

    # Sibling imports: import Foo::bar  →  look for Foo.mdl in same directory
    for match in _MDL_SIBLING_IMPORT_RE.finditer(content):
        module_name = match.group(1)
        candidate = current_dir / f"{module_name}.mdl"
        if candidate.exists():
            _collect_mdl_deps(candidate, visited)

    return visited


def _find_mdl_root(mdl_files: list[Path]) -> Path | None:
    """Find the common ancestor root for all MDL files, accounting for ..:: import depth."""
    if not mdl_files:
        return None

    roots: list[Path] = []
    for mdl_path in mdl_files:
        root = mdl_path.parent
        try:
            content = mdl_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            roots.append(root)
            continue
        max_depth = 0
        for match in _MDL_RELATIVE_IMPORT_RE.finditer(content):
            depth = match.group(1).count("..::")
            max_depth = max(max_depth, depth)
        for _ in range(max_depth):
            if root.parent != root:
                root = root.parent
        roots.append(root)

    try:
        return Path(os.path.commonpath([str(r) for r in roots]))
    except ValueError:
        return roots[0]


def _bundle_stage_flat(
    stage: Usd.Stage,
    bundle_dir: Path,
) -> Path:
    """Export stage with all MDL and texture assets in a ZIP bundle.

    MDL files are stored at their correct relative paths (preserving directory
    structure) so that relative MDL imports (e.g. ``..::..::Templates::Foo``)
    and co-located texture subdirectories resolve correctly in the NVCF renderer.

    Returns the path to the created ZIP file.
    """
    import shutil
    import zipfile

    from pxr import Sdf, UsdShade

    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Export stage
    stage_path = bundle_dir / "stage.usda"
    stage.GetRootLayer().Export(str(stage_path))

    # Collect direct MDL files and texture paths from USD shader attributes
    direct_mdl_files: list[Path] = []
    texture_files: dict[str, Path] = {}  # resolved_path -> Path

    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        mdl_attr = prim.GetAttribute("info:mdl:sourceAsset")
        if mdl_attr and mdl_attr.IsValid():
            val = mdl_attr.Get()
            if val and hasattr(val, "path") and val.path:
                mdl_path = Path(val.resolvedPath or val.path)
                if mdl_path.is_absolute() and mdl_path.exists():
                    direct_mdl_files.append(mdl_path)

        for attr in prim.GetAttributes():
            if attr.GetTypeName().type.typeName != "SdfAssetPath":
                continue
            val = attr.Get()
            if not val or not hasattr(val, "path") or not val.path:
                continue
            fpath = Path(val.resolvedPath or val.path)
            if (
                fpath.is_absolute()
                and fpath.exists()
                and fpath.suffix.lower() != ".mdl"
            ):
                texture_files[str(fpath)] = fpath

    # Collect all transitive MDL dependencies
    all_mdl_deps: set[Path] = set()
    for mdl_path in direct_mdl_files:
        _collect_mdl_deps(mdl_path, all_mdl_deps)

    # Determine the MDL root directory (common ancestor accounting for ..:: depth)
    mdl_root = _find_mdl_root(direct_mdl_files)

    # Build a map: original absolute path -> bundle-relative path string
    bundle_path_map: dict[str, str] = {}

    # Copy MDL files preserving structure relative to mdl_root
    for dep in all_mdl_deps:
        if mdl_root is not None:
            try:
                rel = dep.relative_to(mdl_root)
            except ValueError:
                rel = Path(dep.name)
        else:
            rel = Path(dep.name)
        dst = bundle_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(dep), str(dst))
        bundle_path_map[str(dep)] = str(rel)

        # Also copy co-located texture subdirectory (same stem as the MDL file)
        tex_subdir = dep.parent / dep.stem
        if tex_subdir.is_dir():
            dst_subdir = bundle_dir / rel.parent / dep.stem
            shutil.copytree(str(tex_subdir), str(dst_subdir), dirs_exist_ok=True)

    # Copy textures referenced directly in USD attributes (not inside MDL files)
    for tex_str, tex_path in texture_files.items():
        if tex_str in bundle_path_map:
            continue
        if mdl_root is not None:
            try:
                rel = tex_path.relative_to(mdl_root)
                dst = bundle_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(tex_path), str(dst))
                bundle_path_map[tex_str] = str(rel)
                continue
            except ValueError:
                pass
        # Fallback: copy flat to bundle root
        shutil.copy2(str(tex_path), str(bundle_dir / tex_path.name))
        bundle_path_map[tex_str] = tex_path.name

    # Rewrite asset paths in stage to bundle-relative paths
    layer = Sdf.Layer.FindOrOpen(str(stage_path))

    def rewrite_prim(prim_spec: Sdf.PrimSpec) -> None:
        for attr_name in list(prim_spec.attributes.keys()):
            attr_spec = prim_spec.attributes[attr_name]
            value = attr_spec.default
            if not isinstance(value, Sdf.AssetPath):
                continue
            orig = value.path
            if not orig:
                continue
            orig_path = Path(orig)
            if not orig_path.is_absolute():
                continue
            bundle_rel = bundle_path_map.get(str(orig_path))
            if bundle_rel is not None:
                attr_spec.default = Sdf.AssetPath(bundle_rel)
        for child in prim_spec.nameChildren:
            rewrite_prim(child)

    for prim in layer.rootPrims:
        rewrite_prim(prim)
    layer.Save()

    # Create ZIP (walk all files including subdirs)
    zip_path = bundle_dir.parent / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(str(bundle_dir)):
            for fname in files:
                fpath = Path(root) / fname
                arcname = fpath.relative_to(bundle_dir)
                zf.write(str(fpath), str(arcname))

    return zip_path


def _render_one_thumbnail(
    template_path: Path,
    usd_file: Path,
    material_prim_path: str,
    output_path: Path,
    image_size: int,
) -> tuple[str, Path | None, str | None]:
    """Render a single material thumbnail via NVCF.

    Returns (prim_path, thumbnail_path_or_None, error_or_None).
    """
    import tempfile
    import uuid

    from world_understanding.functions.graphics.render_remote import (
        RenderingStatus,
        render_single_camera_from_url,
    )
    from world_understanding.utils.nvcf_utils import s3_uri_to_https_url
    from world_understanding.utils.s3_utils import delete_s3_path, upload_file_to_s3

    try:
        # Compose template + material (flattened)
        stage = _compose_thumbnail_stage(template_path, usd_file, material_prim_path)

        # Bundle stage with all MDL/texture assets flat at root level
        with tempfile.TemporaryDirectory(prefix="nvcf_thumb_") as tmp:
            bundle_dir = Path(tmp) / "bundle"
            zip_path = _bundle_stage_flat(stage, bundle_dir)

            # Upload bundle to S3
            s3_key = f"nvcf-renders/{uuid.uuid4().hex}/bundle.zip"
            s3_uri = upload_file_to_s3(
                file_path=str(zip_path),
                s3_path=f"s3://{_S3_BUCKET}/{s3_key}",
                profile_name=_S3_PROFILE,
            )
            url = s3_uri_to_https_url(s3_uri, _S3_REGION)

        try:
            result = render_single_camera_from_url(
                usd_url=url,
                camera=_TEMPLATE_CAMERA,
                image_width=image_size,
                image_height=image_size,
            )
        finally:
            # Clean up S3 assets
            try:
                delete_s3_path(s3_uri, profile_name=_S3_PROFILE)
            except Exception:
                log.debug(_S3_CLEANUP_FAILURE_MESSAGE)

        # Check render result
        status = result.get("status")
        if status != RenderingStatus.success:
            return material_prim_path, None, _THUMBNAIL_RENDER_FAILURE_MESSAGE

        images = result.get("images", [])
        if not images:
            return material_prim_path, None, "No images returned from NVCF"

        # Save the rendered image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(str(output_path))
        return material_prim_path, output_path, None

    except Exception:
        return material_prim_path, None, _THUMBNAIL_RENDER_FAILURE_MESSAGE


def render_thumbnails(
    usd_file: Path,
    prim_paths: list[str],
    output_dir: Path,
    image_size: int,
    skip_existing: bool,
    template_path: Path,
    max_workers: int,
) -> dict[str, Path]:
    """Render thumbnails via NVCF cloud rendering.

    For each material, composes a stage with the template scene and the
    material, uploads to S3 with bundled assets, and renders via NVCF.

    Returns a mapping of prim_path -> thumbnail_path for successful renders.
    """
    thumbs_dir = output_dir / "thumbs" / f"{image_size}x{image_size}"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    if not template_path.exists():
        log.error("Template USD not found: %s", template_path)
        return {}

    # Filter out existing thumbnails if requested
    to_render: list[str] = []
    already_done: dict[str, Path] = {}
    for pp in prim_paths:
        thumb_path = thumbs_dir / f"{prim_path_to_filename(pp)}.png"
        if skip_existing and thumb_path.exists():
            log.debug("Skipping existing: %s", thumb_path.name)
            already_done[pp] = thumb_path
        else:
            to_render.append(pp)

    if already_done:
        log.info("Skipping %d materials with existing thumbnails", len(already_done))

    if not to_render:
        log.info("All thumbnails already exist, nothing to render")
        return already_done

    log.info(
        "Rendering %d thumbnails via NVCF (workers=%d)", len(to_render), max_workers
    )

    results: dict[str, Path] = dict(already_done)
    failed: list[str] = []
    done_count = 0
    total = len(to_render)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for pp in to_render:
            thumb_path = thumbs_dir / f"{prim_path_to_filename(pp)}.png"
            future = executor.submit(
                _render_one_thumbnail,
                template_path=template_path,
                usd_file=usd_file,
                material_prim_path=pp,
                output_path=thumb_path,
                image_size=image_size,
            )
            futures[future] = pp

        for future in as_completed(futures):
            done_count += 1
            pp, thumb_path, error = future.result()
            name = prim_path_to_filename(pp)

            if error:
                log.warning(
                    "[%d/%d] %s", done_count, total, _THUMBNAIL_RENDER_FAILURE_MESSAGE
                )
                failed.append(pp)
            else:
                results[pp] = thumb_path  # type: ignore[assignment]
                log.info(
                    "[%d/%d] OK %s",
                    done_count,
                    total,
                    redact_sensitive_config(name, _path_context=True),
                )

    if failed:
        log.warning(
            "Failed to render %d/%d thumbnails: %s",
            len(failed),
            len(prim_paths),
            ", ".join(prim_path_to_filename(p) for p in failed),
        )

    return results


# ---------------------------------------------------------------------------
# Step 3: Generate Descriptions via VLM
# ---------------------------------------------------------------------------

DESCRIPTION_PROMPT = """\
Write a concise 1-sentence description of this material's visual appearance.

Rules:
- START the description with the EXACT material name "{name}" — do not paraphrase or shorten it
- Keep it 10-25 words, one sentence only
- Use compound color adjectives (e.g., "silver-gray", "yellow-gold", "reddish-brown")
- Mention: color, surface type, reflectivity level, finish/texture
- Use precise terms: "high/medium/low reflectivity", "glossy/matte/brushed finish", \
"sharp/soft highlights", "directional grain"
- Do NOT invent a different name or omit words from the material name

Examples by category:
- Metal base: "Gold is a warm yellow-gold metallic surface with medium reflectivity, subtle color \
variation, and moderately sharp highlights"
- Metal brushed: "Aluminum Brushed features a silver-gray metallic surface with a fine directional \
grain, softened reflections, and a streaked, brushed appearance"
- Metal matte: "Brass Matte presents a dull yellow-gold metallic surface with low reflectivity, \
soft diffuse reflections, and an even, non-glossy finish"
- Metal polished: "Copper Polished is a bright reddish-brown metallic surface with very high \
reflectivity, mirror-like shine, and crisp, intense highlights"
- Plastic: "Plastic Black is a glossy opaque plastic with smooth reflective finish and sharp highlights"
- Car paint: "Car Paint Blue features deep, dark-blue-colored automotive paint with a smooth, even \
surface, gentle gloss reflections, and a subtle clear coat shimmer"
- Glass: "Glass Clear is a transparent, colorless glass with smooth surfaces, crisp refractions, \
and sharp, bright reflections"

Material name: {name}
Prim name: {prim_name}

Respond with ONLY the description, no prefix or extra text."""

DESCRIPTION_SYSTEM_PROMPT = "You are a material science expert who writes short, technical material descriptions."


def _describe_one(
    vlm: object,
    pp: str,
    thumb_path: Path,
) -> tuple[str, str | None, str | None]:
    """Describe a single material. Returns (prim_path, description, error)."""
    from world_understanding.functions.cv.vlm import generate_vlm_response

    name = prim_path_to_name(pp)
    prim_name = prim_path_to_filename(pp)
    prompt = DESCRIPTION_PROMPT.format(name=name, prim_name=prim_name)
    try:
        result = generate_vlm_response(
            vlm=vlm,
            prompt=prompt,
            system_prompt=DESCRIPTION_SYSTEM_PROMPT,
            images=[str(thumb_path)],
        )
        if "error" in result:
            return pp, None, _DESCRIPTION_GENERATION_FAILURE_MESSAGE
        desc = result["response"].strip().strip('"').strip("'")
        return pp, desc, None
    except Exception:
        return pp, None, _DESCRIPTION_GENERATION_FAILURE_MESSAGE


def generate_descriptions(
    thumbnails: dict[str, Path],
    vlm_backend: str,
    vlm_model: str | None,
    max_workers: int = 8,
) -> dict[str, str]:
    """Generate VLM descriptions for each material thumbnail.

    Uses ThreadPoolExecutor for parallel VLM calls.

    Returns a mapping of prim_path -> description.
    """
    from world_understanding.functions.models.vision_language_models import create_vlm
    from world_understanding.utils.credentials import (
        API_KEY_ENV_VAR_MAP,
        get_env_api_key_for_backend,
    )

    log.info(
        "Generating descriptions via VLM (backend=%s, model=%s, workers=%d)",
        redact_sensitive_config(vlm_backend),
        redact_sensitive_config(vlm_model or "default", _path_context=True),
        max_workers,
    )

    vlm_kwargs: dict = {"backend": vlm_backend}
    if vlm_model:
        vlm_kwargs["model"] = vlm_model

    # Resolve API key from environment based on backend.
    env_vars = API_KEY_ENV_VAR_MAP.get(vlm_backend, ())
    if env_vars:
        api_key = get_env_api_key_for_backend(vlm_backend)
        if not api_key:
            expected = " or ".join(env_vars)
            raise ValueError(f"{expected} not set for {vlm_backend} backend")
        vlm_kwargs["api_key"] = api_key

    provider_setup_failed = False
    try:
        vlm = create_vlm(**vlm_kwargs)
    except Exception:
        provider_setup_failed = True
        vlm = None
    if provider_setup_failed:
        vlm_kwargs = {}
        api_key = None
        raise RuntimeError(_DESCRIPTION_GENERATION_FAILURE_MESSAGE)

    descriptions: dict[str, str] = {}
    failed: list[str] = []
    done_count = 0
    total = len(thumbnails)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_describe_one, vlm, pp, thumb_path): pp
            for pp, thumb_path in thumbnails.items()
        }

        for future in as_completed(futures):
            done_count += 1
            pp, desc, error = future.result()
            name = prim_path_to_name(pp)
            if error:
                log.warning(
                    "[%d/%d] %s",
                    done_count,
                    total,
                    _DESCRIPTION_GENERATION_FAILURE_MESSAGE,
                )
                failed.append(name)
            else:
                descriptions[pp] = desc  # type: ignore[assignment]
                log.info(
                    "[%d/%d] %s: %s",
                    done_count,
                    total,
                    redact_sensitive_config(name, _path_context=True),
                    redact_sensitive_config(desc),
                )

    if failed:
        log.warning("Failed to describe %d/%d materials", len(failed), len(thumbnails))

    return descriptions


# ---------------------------------------------------------------------------
# Step 4: Output materials.yaml
# ---------------------------------------------------------------------------


def write_materials_yaml(
    output_dir: Path,
    usd_file: Path,
    prim_paths: list[str],
    thumbnails: dict[str, Path],
    descriptions: dict[str, str],
    image_size: int,
    library_path: str | None,
) -> Path:
    """Write materials.yaml to the output directory."""
    import yaml

    entries = []
    for pp in prim_paths:
        name = prim_path_to_name(pp)
        filename = prim_path_to_filename(pp)
        icon = f"thumbs/{image_size}x{image_size}/{filename}.png"

        entry: dict = {
            "name": name,
            "description": descriptions.get(pp, ""),
            "binding": pp,
            "icon": icon if pp in thumbnails else "",
        }
        entries.append(entry)

    if library_path:
        lib_path = library_path
    else:
        # Make path relative to the output directory (yaml location)
        lib_path = os.path.relpath(usd_file, output_dir)

    data = {
        "library_path": lib_path,
        "entries": entries,
    }

    yaml_path = output_dir / "materials.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Material Library Configuration\n")
        f.write("# Auto-generated by material-agent generate-manifest\n")
        f.write(f"# Source: {usd_file.name}\n\n")
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )

    log.info("Wrote %s with %d entries", yaml_path, len(entries))
    return yaml_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_generate_manifest(params: GenerateManifestInput) -> GenerateManifestResult:
    """Run the manifest generation workflow.

    This is the main entry point called by the CLI command.
    """
    # Validate USD file
    if not params.usd_file.exists():
        return GenerateManifestResult(
            success=False, error=f"USD file not found: {params.usd_file}"
        )

    # Step 1: Discover materials
    prim_paths = discover_materials(params.usd_file)
    if not prim_paths:
        return GenerateManifestResult(
            success=False, error=f"No materials found in {params.usd_file}"
        )

    # --list-materials: just return the paths
    if params.list_materials:
        return GenerateManifestResult(
            success=True,
            materials_count=len(prim_paths),
            material_paths=prim_paths,
        )

    # Ensure output dir exists
    params.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve template path
    template_path = params.template.resolve()
    if not template_path.exists():
        return GenerateManifestResult(
            success=False, error=f"Template USD not found: {template_path}"
        )
    log.info("Using template: %s", template_path)

    # Step 2: Render thumbnails via NVCF
    start = time.time()
    thumbnails = render_thumbnails(
        usd_file=params.usd_file,
        prim_paths=prim_paths,
        output_dir=params.output_dir,
        image_size=params.image_size,
        skip_existing=params.skip_existing,
        template_path=template_path,
        max_workers=params.max_workers,
    )
    elapsed_thumbs = time.time() - start
    log.info(
        "Thumbnail rendering: %d/%d succeeded in %.1fs",
        len(thumbnails),
        len(prim_paths),
        elapsed_thumbs,
    )

    # Step 3: Generate descriptions
    descriptions: dict[str, str] = {}
    if not params.skip_descriptions and thumbnails:
        start = time.time()
        descriptions = generate_descriptions(
            thumbnails=thumbnails,
            vlm_backend=params.vlm_backend,
            vlm_model=params.vlm_model,
            max_workers=params.vlm_workers,
        )
        elapsed_desc = time.time() - start
        log.info(
            "Description generation: %d/%d succeeded in %.1fs",
            len(descriptions),
            len(thumbnails),
            elapsed_desc,
        )
    elif params.skip_descriptions:
        log.info("Skipping VLM description generation (--skip-descriptions)")

    # Step 4: Write materials.yaml
    yaml_path = write_materials_yaml(
        output_dir=params.output_dir,
        usd_file=params.usd_file,
        prim_paths=prim_paths,
        thumbnails=thumbnails,
        descriptions=descriptions,
        image_size=params.image_size,
        library_path=params.library_path,
    )

    return GenerateManifestResult(
        success=True,
        yaml_path=yaml_path,
        materials_count=len(prim_paths),
        thumbnails_count=len(thumbnails),
        descriptions_count=len(descriptions),
        material_paths=prim_paths,
    )
