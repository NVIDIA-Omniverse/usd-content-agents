# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD Stage utility functions for loading, saving, and manipulating USD stages."""

import base64
import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pxr import Usd  # pragma: no cover

logger = logging.getLogger(__name__)

# Conservative defaults to avoid filesystem NAME_MAX(255) issues once suffixes are added.
MAX_PATH_COMPONENT_LEN = 96
MAX_FILENAME_STEM_LEN = 120
_DATA_URI_MIME_TYPE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$"
)
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")


def is_windows_drive_path(asset_path: str) -> bool:
    """Return true for Windows drive paths, which are not URI schemes."""
    return bool(_WINDOWS_DRIVE_RE.match(asset_path))


def has_uri_scheme(asset_path: str) -> bool:
    """Return true for URI-like USD asset paths, excluding Windows drives."""
    if is_windows_drive_path(asset_path):
        return False
    return bool(_URI_SCHEME_RE.match(asset_path))


def normalize_windows_drive_path(asset_path: str) -> str:
    """Return a USD-friendly spelling of a Windows drive path."""
    return PureWindowsPath(asset_path).as_posix()


def sanitize_name_for_filesystem(name: str) -> str:
    """Sanitize a name to be safe for filesystem use.

    This function converts USD prim paths and other names into filesystem-safe strings
    by replacing problematic characters with underscores.

    Args:
        name: The name to sanitize (e.g., camera path like "/World/Camera_001")

    Returns:
        A sanitized string safe for use in filenames

    Example:
        >>> sanitize_name_for_filesystem("/World/Camera 001")
        'World_Camera_001'
        >>> sanitize_name_for_filesystem("Camera:Main (HD)")
        'Camera_Main__HD_'
    """
    # First, replace forward slashes with underscores
    name = name.replace("/", "_")

    # Then replace any character that isn't word character, hyphen, underscore, or dot
    # This handles spaces, colons, parentheses, and other special characters
    sanitized = re.sub(r"[^\w\-_.]", "_", name)

    # Remove leading/trailing underscores that might result from leading slashes
    sanitized = sanitized.strip("_")

    # Ensure we don't have empty string
    if not sanitized:
        sanitized = "unnamed"

    return sanitized


def shorten_for_filesystem(
    name: str,
    max_len: int = MAX_PATH_COMPONENT_LEN,
    hash_len: int = 8,
) -> str:
    """Sanitize and deterministically shorten a name for filesystem safety.

    This keeps names readable while bounding length and preserving uniqueness
    via a stable hash suffix when truncation is needed.

    Args:
        name: Raw name (prim segment, prim name, etc.)
        max_len: Maximum allowed output length.
        hash_len: Number of hex chars in hash suffix when truncating.

    Returns:
        Filesystem-safe and length-bounded name.
    """
    if max_len <= 0:
        raise ValueError("max_len must be > 0")
    if hash_len <= 0:
        raise ValueError("hash_len must be > 0")

    sanitized = sanitize_name_for_filesystem(name)
    if len(sanitized) <= max_len:
        return sanitized

    # Reserve one char for "_" separator between prefix and hash.
    # If max_len is too small, fall back to hash-only output.
    if max_len <= hash_len + 1:
        digest_size = max(1, (max_len + 1) // 2)
        return hashlib.blake2b(
            sanitized.encode("utf-8"),
            digest_size=digest_size,
        ).hexdigest()[:max_len]

    digest_size = max(4, (hash_len + 1) // 2)
    suffix = hashlib.blake2b(
        sanitized.encode("utf-8"),
        digest_size=digest_size,
    ).hexdigest()[:hash_len]
    prefix_len = max_len - hash_len - 1
    prefix = sanitized[:prefix_len]
    return f"{prefix}_{suffix}"


def create_stage(identifier: str | None = None) -> "Usd.Stage":
    """Create a new USD stage in memory.

    Args:
        identifier: Optional identifier for the stage. If None, creates an anonymous stage.
                   Can be a filename like 'scene.usda' or a descriptive name.

    Returns:
        A new USD Stage object

    Example:
        >>> stage = create_stage("my_scene.usda")
        >>> # Or create anonymous stage
        >>> stage = create_stage()
    """
    from pxr import Usd

    if identifier:
        return Usd.Stage.CreateInMemory(identifier)
    else:
        return Usd.Stage.CreateInMemory()


def create_stage_with_file(file_path: str | Path) -> "Usd.Stage":
    """Create a new USD stage and associate it with a file path.

    This creates a new stage that will be saved to the specified file path.
    The directory will be created if it doesn't exist.

    Args:
        file_path: Path where the stage will be saved

    Returns:
        A new USD Stage object

    Raises:
        OSError: If the directory cannot be created
        RuntimeError: If the stage cannot be created

    Example:
        >>> stage = create_stage_with_file("/tmp/scene.usda")
        >>> # Stage is ready to be edited and saved
    """
    from pxr import Usd

    file_path = Path(file_path)

    # Ensure parent directory exists
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory for {file_path}: {e}")
        raise

    try:
        stage = Usd.Stage.CreateNew(str(file_path))
        logger.info(f"Created new USD stage at {file_path}")
        return stage
    except Exception as e:
        logger.error(f"Failed to create USD stage at {file_path}: {e}")
        raise RuntimeError(f"Failed to create USD stage: {e}") from e


def load_stage(file_path: str | Path) -> "Usd.Stage":
    """Load a USD stage from a file.

    Args:
        file_path: Path to the USD file to load

    Returns:
        The loaded USD Stage object

    Raises:
        FileNotFoundError: If the file doesn't exist
        RuntimeError: If the stage cannot be loaded

    Example:
        >>> stage = load_stage("assets/scene.usda")
        >>> print(stage.GetRootLayer().identifier)
    """
    from pxr import Usd

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"USD file not found: {file_path}")

    try:
        stage = Usd.Stage.Open(str(file_path))
        if not stage:
            raise RuntimeError(f"Failed to open USD stage from {file_path}")
        logger.info(f"Loaded USD stage from {file_path}")
        return stage
    except Exception as e:
        logger.error(f"Failed to load USD stage from {file_path}: {e}")
        raise RuntimeError(f"Failed to load USD stage: {e}") from e


def save_stage(stage: "Usd.Stage", file_path: str | Path | None = None) -> str:
    """Save a USD stage to a file.

    If no file path is provided, saves to the stage's existing file path.
    For in-memory stages without a file path, you must provide one.

    Args:
        stage: The USD Stage to save
        file_path: Optional path where to save the stage. If None, uses the stage's
                  existing file path.

    Returns:
        The path where the stage was saved

    Raises:
        ValueError: If no file path is available
        RuntimeError: If the stage cannot be saved

    Example:
        >>> stage = create_stage()
        >>> # ... modify stage ...
        >>> save_stage(stage, "output/scene.usda")
    """
    if file_path:
        file_path = Path(file_path)
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Export to the new path
        root_layer = stage.GetRootLayer()
        if not root_layer.Export(str(file_path)):
            raise RuntimeError(f"Failed to export stage to {file_path}")
        logger.info(f"Saved USD stage to {file_path}")
        return str(file_path)
    else:
        # Save to existing path
        root_layer = stage.GetRootLayer()
        if root_layer.anonymous:
            raise ValueError(
                "Cannot save in-memory stage without providing a file path"
            )

        # For stages created with CreateNew, we need to call Save()
        stage.Save()
        identifier = str(root_layer.identifier)
        logger.info(f"Saved USD stage to {identifier}")
        return identifier


def export_stage_to_string(stage: "Usd.Stage", add_comment: bool = True) -> str:
    """Export a USD stage to a string in USDA format.

    Args:
        stage: The USD Stage to export
        add_comment: Whether to add a source file comment to the output

    Returns:
        The stage contents as a USDA format string

    Example:
        >>> stage = create_stage()
        >>> # ... modify stage ...
        >>> usda_string = export_stage_to_string(stage)
        >>> print(usda_string)
    """
    root_layer = stage.GetRootLayer()
    if add_comment:
        # ExportToString on the root layer includes more metadata
        return str(root_layer.ExportToString())
    else:
        # Use stage's ExportToString for cleaner output
        return str(stage.ExportToString())


def load_stage_from_string(
    usda_string: str, identifier: str = "string.usda"
) -> "Usd.Stage":
    """Create a USD stage from a USDA format string.

    Args:
        usda_string: USD content in USDA text format
        identifier: Optional identifier for the in-memory stage

    Returns:
        A new USD Stage object loaded from the string

    Raises:
        RuntimeError: If the stage cannot be created from the string

    Example:
        >>> usda_content = '''#usda 1.0
        ... def Xform "World"
        ... {
        ... }
        ... '''
        >>> stage = load_stage_from_string(usda_content)
    """
    from pxr import Sdf, Usd

    try:
        # Create an anonymous layer and import the string content
        layer = Sdf.Layer.CreateAnonymous(
            f"{identifier}.usda" if identifier else ".usda"
        )
        if not layer.ImportFromString(usda_string):
            raise RuntimeError("Failed to import USDA content")

        # Create a stage with this layer as the root
        stage = Usd.Stage.Open(layer)
        if not stage:
            raise RuntimeError("Failed to create stage from layer")

        logger.info(f"Created USD stage from string with identifier: {identifier}")
        return stage
    except Exception as e:
        logger.error(f"Failed to create stage from string: {e}")
        raise RuntimeError(f"Failed to create stage from string: {e}") from e


def duplicate_stage(
    source_stage: "Usd.Stage", identifier: str | None = None
) -> "Usd.Stage":
    """Create a duplicate of an existing USD stage.

    This creates a new in-memory stage with the same content as the source.

    Uses a flattening approach to avoid instance proxy issues by resolving
    all instanced content into concrete, editable prims. Instanceable metadata
    is explicitly removed to ensure the resulting stage has no instance proxies
    that would prevent modification.

    Args:
        source_stage: The stage to duplicate
        identifier: Optional identifier for the new stage

    Returns:
        A new USD Stage object with duplicated content (flattened, de-instanced)

    Example:
        >>> original = load_stage("scene.usda")
        >>> duplicate = duplicate_stage(original, "scene_copy.usda")
    """
    from pxr import Sdf, Usd

    try:
        # Flatten the stage to resolve all composition arcs and instancing
        # into a single layer with concrete, editable prims.
        # Uses binary TransferContent instead of text ExportToString/ImportFromString
        # to avoid the ~4x memory expansion from USDA text serialization.
        flat_layer = source_stage.Flatten()
        new_layer = Sdf.Layer.CreateAnonymous(f"{identifier or 'duplicate'}.usda")
        new_layer.TransferContent(flat_layer)

        new_stage = Usd.Stage.Open(new_layer)
        if not new_stage:
            raise RuntimeError("Failed to create stage from flattened content")

        # IMPORTANT: Disable instanceable metadata on ALL prims to ensure
        # no instance proxies exist. ExportToString() preserves instanceable
        # metadata, which causes USD to recreate instance proxies on load.
        # This prevents "authoring to an instance proxy is not allowed" errors.
        _disable_all_instancing(new_stage)

        logger.info(
            f"Duplicated USD stage with identifier: {identifier or 'anonymous'} (flattened, de-instanced)"
        )
        return new_stage

    except Exception as e:
        logger.error(f"Failed to duplicate stage: {e}")
        raise RuntimeError(f"Failed to duplicate stage: {e}") from e


def _disable_all_instancing(stage: "Usd.Stage") -> int:
    """Disable instanceable metadata on all prims in a stage.

    This function traverses the entire stage and sets instanceable=False on any
    prims that have instanceable metadata authored. This ensures the stage has
    no instance proxies, making all prims directly editable.

    Args:
        stage: The stage to modify in place

    Returns:
        Number of prims that had their instanceable metadata changed
    """

    count = 0
    # Use TraverseAll() to include inactive and abstract prims,
    # and traverse into instance prototypes
    for prim in stage.TraverseAll():
        # Check if prim has instanceable metadata authored (true or false)
        # We need to clear it if set to True
        if prim.HasAuthoredInstanceable() and prim.IsInstanceable():
            prim.SetInstanceable(False)
            count += 1

    if count > 0:
        logger.debug(f"Disabled instancing on {count} prim(s)")

    return count


def create_temp_stage(
    prefix: str = "temp_stage_", suffix: str = ".usda"
) -> tuple["Usd.Stage", str]:
    """Create a temporary USD stage with an associated temporary file.

    The temporary file is created but the stage is not automatically saved.
    You should clean up the file when done.

    Args:
        prefix: Prefix for the temporary filename
        suffix: Suffix for the temporary filename (should include extension)

    Returns:
        Tuple of (stage, temp_file_path)

    Example:
        >>> stage, temp_path = create_temp_stage()
        >>> try:
        ...     # Work with stage
        ...     save_stage(stage)
        ... finally:
        ...     # Clean up
        ...     os.unlink(temp_path)
    """
    from pxr import Usd

    # Create a temporary file
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)  # Close the file descriptor, we'll use USD to write

    try:
        stage = Usd.Stage.CreateNew(temp_path)
        logger.info(f"Created temporary USD stage at {temp_path}")
        return stage, temp_path
    except Exception as e:
        # Clean up the temp file if stage creation fails
        os.unlink(temp_path)
        raise RuntimeError(f"Failed to create temporary stage: {e}") from e


def flatten_stage(
    source_stage: "Usd.Stage",
    output_path: str | Path | None = None,
    add_comment: bool = True,
) -> "Usd.Stage":
    """Flatten a USD stage, resolving all composition arcs into a single layer.

    This is useful for creating a self-contained USD file from a stage that
    may reference external files or use complex composition.

    Args:
        source_stage: The stage to flatten
        output_path: Optional path to save the flattened stage. If None,
                    returns an in-memory stage.
        add_comment: Whether to add source file comments

    Returns:
        A new flattened USD Stage

    Example:
        >>> complex_stage = load_stage("complex_scene.usda")
        >>> flat_stage = flatten_stage(complex_stage, "flat_scene.usda")
    """

    from pxr import Sdf, Usd

    # Flatten resolves all composition arcs and instancing into a single layer.
    # Uses binary TransferContent instead of text ExportToString/ImportFromString
    # to avoid the ~4x memory expansion from USDA text serialization.
    flattened_layer = source_stage.Flatten(addSourceFileComment=add_comment)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export the flattened layer directly to disk (avoids in-memory copy)
        if not flattened_layer.Export(str(output_path)):
            raise RuntimeError(f"Failed to export flattened stage to {output_path}")

        new_stage = Usd.Stage.Open(str(output_path))
        if not new_stage:
            raise RuntimeError(f"Failed to open flattened stage from {output_path}")

        logger.info(f"Created flattened stage at {output_path}")
        return new_stage
    else:
        # Binary transfer to in-memory layer (no text serialization overhead)
        new_layer = Sdf.Layer.CreateAnonymous("flattened.usda")
        new_layer.TransferContent(flattened_layer)

        new_stage = Usd.Stage.Open(new_layer)
        if not new_stage:
            raise RuntimeError("Failed to create stage from flattened content")

        return new_stage


def prepare_stage_for_render(
    source_stage: "Usd.Stage",
    *,
    flatten: bool = True,
    normalize_materials: bool = True,
    add_comment: bool = True,
) -> tuple["Usd.Stage", dict[str, Any]]:
    """Prepare a USD stage for render backends that export one root layer.

    REST-style renderers typically receive a single exported root layer. For
    composed stages, that root layer can omit referenced payloads, sublayers, or
    other composition arcs unless the stage is flattened first. This helper
    centralizes the render-prep behavior shared by agents:

    - optionally flatten the composed stage;
    - preserve stage up axis and meters-per-unit metadata lost by ``Flatten()``;
    - optionally normalize custom/local MDL references for remote renderers;
    - bake session-layer opinions into non-flattened render roots so one-layer
      exports preserve authored overrides.

    Args:
        source_stage: The stage to prepare.
        flatten: Whether to flatten composition arcs into a single layer.
        normalize_materials: Whether to normalize unsupported/local MDL shader
            references using the shared material utility.
        add_comment: Whether ``Flatten()`` should add source-file comments.

    Returns:
        Tuple of ``(prepared_stage, metadata)``. The returned stage is a new
        in-memory stage when ``flatten`` is true. When ``flatten`` is false, the
        returned stage is a render-prepared clone with relative composition arcs
        anchored to ``source_stage``'s layer directories and session-layer
        opinions folded into the returned root layer. Non-flattened clones are
        render intermediates, not portable USD artifacts, because anchored
        composition paths can point at the local host filesystem.
    """

    from pxr import Sdf, Usd, UsdGeom, UsdUtils

    metadata: dict[str, Any] = {
        "flattened": False,
        "material_normalized": False,
    }
    source_root_layer = source_stage.GetRootLayer()
    if source_root_layer.realPath:
        metadata["asset_base_dir"] = str(Path(source_root_layer.realPath).parent)

    prepared_stage = source_stage
    if flatten:
        original_up_axis = UsdGeom.GetStageUpAxis(source_stage)
        original_meters_per_unit = UsdGeom.GetStageMetersPerUnit(source_stage)
        flattened_layer = source_stage.Flatten(addSourceFileComment=add_comment)
        prepared_stage = Usd.Stage.Open(flattened_layer)
        if prepared_stage is None:
            raise RuntimeError("Failed to open flattened USD stage for rendering")

        UsdGeom.SetStageUpAxis(prepared_stage, original_up_axis)
        UsdGeom.SetStageMetersPerUnit(prepared_stage, original_meters_per_unit)
        metadata.update(
            {
                "flattened": True,
                "up_axis": str(original_up_axis),
                "meters_per_unit": float(original_meters_per_unit),
            }
        )
    else:
        _validate_nonflatten_render_clone_dependencies(source_stage)
        render_layer = Sdf.Layer.CreateAnonymous("render-prepared.usda")
        render_layer.TransferContent(source_root_layer)
        if source_root_layer.realPath:
            source_dir = Path(source_root_layer.realPath).parent
            _resolve_render_layer_composition_paths(render_layer, source_dir)
        elif _render_layer_has_relative_composition_paths(render_layer):
            raise RuntimeError(
                "Non-flatten render preparation cannot anchor relative "
                "composition paths because the source root layer has no "
                "filesystem path. Save the source stage, provide a rooted USD "
                "layer, or use flatten=True."
            )
        source_session_layer = source_stage.GetSessionLayer()
        if not source_session_layer.empty:
            render_session_layer = Sdf.Layer.CreateAnonymous(
                "render-prepared-session.usda"
            )
            render_session_layer.TransferContent(source_session_layer)
            if source_session_layer.realPath:
                session_dir = Path(source_session_layer.realPath).parent
                _resolve_render_layer_composition_paths(
                    render_session_layer,
                    session_dir,
                )
            elif _render_layer_has_relative_composition_paths(render_session_layer):
                raise RuntimeError(
                    "Non-flatten render preparation cannot anchor relative "
                    "composition paths because the source session layer has no "
                    "filesystem path. Save the session layer, provide rooted "
                    "USD layers, or use flatten=True."
                )

            combined_layer = Sdf.Layer.CreateAnonymous("render-prepared.usda")
            combined_layer.TransferContent(render_session_layer)
            UsdUtils.StitchLayers(combined_layer, render_layer)
            render_layer = combined_layer

        prepared_stage = Usd.Stage.Open(render_layer)
        if prepared_stage is None:
            raise RuntimeError("Failed to clone USD stage for render preparation")

    if normalize_materials:
        from world_understanding.utils.usd.material import convert_custom_mdl_to_builtin

        convert_custom_mdl_to_builtin(prepared_stage)
        metadata["material_normalized"] = True

    return prepared_stage, metadata


def _validate_nonflatten_render_clone_dependencies(source_stage: "Usd.Stage") -> None:
    """Reject layer state that a non-flattened root-layer clone would drop."""
    root_layer = source_stage.GetRootLayer()
    session_layer = source_stage.GetSessionLayer()
    cloned_layer_ids = {root_layer.identifier, session_layer.identifier}
    for layer in source_stage.GetUsedLayers():
        if layer.identifier in cloned_layer_ids:
            continue
        if layer.anonymous:
            raise RuntimeError(
                "Non-flatten render preparation cannot preserve anonymous "
                f"composed layer {layer.identifier!r}; use flatten=True."
            )
        if layer.dirty:
            raise RuntimeError(
                "Non-flatten render preparation cannot preserve unsaved edits "
                f"in composed layer {layer.identifier!r}; save the layer or use "
                "flatten=True."
            )


def _render_layer_has_relative_composition_paths(layer: Any) -> bool:
    """Return true when a layer has composition paths needing a source anchor."""
    from pxr import Sdf

    if any(_is_relative_render_asset_path(path) for path in layer.subLayerPaths):
        return True

    prim_paths: list[Any] = []
    layer.Traverse("/", prim_paths.append)
    for prim_path in prim_paths:
        prim_spec = layer.GetPrimAtPath(prim_path)
        if not isinstance(prim_spec, Sdf.PrimSpec):
            continue
        if _list_editor_has_relative_render_asset_path(prim_spec.referenceList):
            return True
        if _list_editor_has_relative_render_asset_path(prim_spec.payloadList):
            return True
    return False


def _list_editor_has_relative_render_asset_path(list_editor: Any) -> bool:
    # deletedItems are removals, not dependencies that need anchoring.
    for field in (
        "prependedItems",
        "appendedItems",
        "addedItems",
        "explicitItems",
    ):
        for item in getattr(list_editor, field):
            if _is_relative_render_asset_path(getattr(item, "assetPath", "")):
                return True
    return False


def _is_relative_render_asset_path(asset_path: str) -> bool:
    if not asset_path:
        return False
    if asset_path.startswith("anon:"):
        return False
    if os.path.isabs(asset_path) or is_windows_drive_path(asset_path):
        return False
    return not has_uri_scheme(asset_path)


def _resolve_render_sublayer_path(path: str, source_dir: Path) -> str:
    """Return an absolute sublayer path for anonymous render-prep clones."""
    if not path:
        return path
    if is_windows_drive_path(path):
        return normalize_windows_drive_path(path)
    if path.startswith("anon:") or os.path.isabs(path) or has_uri_scheme(path):
        return path
    return (source_dir / path).resolve().as_posix()


def _resolve_render_layer_composition_paths(layer: Any, source_dir: Path) -> None:
    """Anchor relative composition paths before exporting a cloned root layer."""
    from pxr import Sdf

    if layer.subLayerPaths:
        layer.subLayerPaths = [
            _resolve_render_sublayer_path(path, source_dir)
            for path in layer.subLayerPaths
        ]

    prim_paths: list[Any] = []
    layer.Traverse("/", prim_paths.append)
    for prim_path in prim_paths:
        prim_spec = layer.GetPrimAtPath(prim_path)
        if not isinstance(prim_spec, Sdf.PrimSpec):
            continue
        _resolve_render_asset_list(prim_spec.referenceList, source_dir)
        _resolve_render_asset_list(prim_spec.payloadList, source_dir)


def _resolve_render_asset_list(list_editor: Any, source_dir: Path) -> None:
    """Anchor relative reference or payload list items in an Sdf list editor."""
    for field in (
        "prependedItems",
        "appendedItems",
        "addedItems",
        "explicitItems",
    ):
        items = getattr(list_editor, field)
        if not items:
            continue
        resolved_items = [
            _resolve_render_composition_item(item, source_dir) for item in items
        ]
        setattr(list_editor, field, resolved_items)


def _resolve_render_composition_item(item: Any, source_dir: Path) -> Any:
    """Return a reference/payload item with its asset path anchored if needed."""
    from pxr import Sdf

    asset_path = getattr(item, "assetPath", "")
    resolved_asset_path = _resolve_render_asset_path(asset_path, source_dir)
    if resolved_asset_path == asset_path:
        return item
    if isinstance(item, Sdf.Reference):
        return Sdf.Reference(
            resolved_asset_path,
            item.primPath,
            item.layerOffset,
            item.customData,
        )
    if isinstance(item, Sdf.Payload):
        return Sdf.Payload(resolved_asset_path, item.primPath, item.layerOffset)
    return item


def _resolve_render_asset_path(asset_path: str, source_dir: Path) -> str:
    """Return an absolute USD asset path for a source-relative composition arc."""
    if not asset_path:
        return asset_path
    if is_windows_drive_path(asset_path):
        return normalize_windows_drive_path(asset_path)
    if os.path.isabs(asset_path):
        return Path(asset_path).as_posix()
    if asset_path.startswith("anon:") or has_uri_scheme(asset_path):
        return asset_path
    return (source_dir / asset_path).resolve().as_posix()


def remove_animation(
    stage: "Usd.Stage",
    reference_time: "Usd.TimeCode" = None,
) -> int:
    """Remove all time-sampled attributes, keeping value at reference time.

    This function converts animated (time-sampled) attributes to static values
    by evaluating them at a specific time and removing all time samples.
    This is useful when preparing a stage for rendering systems that use
    time codes for other purposes (e.g., per-prim camera positioning).

    For each animated attribute:
    1. Get value at reference_time (default: stage.GetStartTimeCode() or 0)
    2. Clear all time samples
    3. Set static value

    Note:
        Uses stage.TraverseAll() to include prims inside prototypes.
        Instance proxies are skipped since they are read-only.

    Args:
        stage: The USD stage to modify (in-place)
        reference_time: Time to sample values from. If not specified, uses
            stage.GetStartTimeCode() if set, otherwise Usd.TimeCode(0).

    Returns:
        Number of attributes converted from animated to static

    Example:
        >>> stage = load_stage("animated_scene.usda")
        >>> num_removed = remove_animation(stage)
        >>> print(f"Converted {num_removed} animated attributes to static")
    """
    from pxr import Usd

    if reference_time is None:
        # Use stage start time if available, otherwise fall back to time 0
        start_time = stage.GetStartTimeCode()
        if start_time != Usd.TimeCode.Default():
            reference_time = Usd.TimeCode(start_time)
        else:
            reference_time = Usd.TimeCode(0)

    count = 0
    for prim in stage.TraverseAll():
        # Skip instance proxies - they are read-only
        if prim.IsInstanceProxy():
            continue
        for attr in prim.GetAttributes():
            if attr.GetNumTimeSamples() > 0:
                value = attr.Get(reference_time)
                attr.Clear()
                if value is not None:
                    attr.Set(value)
                count += 1

    if count > 0:
        logger.debug(f"Removed animation from {count} attributes")

    return count


def get_stage_info_from_path(file_path: str | Path) -> dict[str, Any] | None:
    """Get stage information from a USD file path.

    This is a best-effort helper intended for lightweight statistics and UI
    warnings. It returns None if the USD cannot be opened or inspected.

    Args:
        file_path: Path to a USD file.

    Returns:
        Stage info dictionary, or None if the stage cannot be opened.
    """
    from pxr import Usd

    try:
        stage = Usd.Stage.Open(str(file_path))
        if stage is None:
            return None
        return get_stage_info(stage)
    except Exception:
        logger.exception("Failed to get USD stage info")
        return None


def get_stage_info(stage: "Usd.Stage") -> dict[str, Any]:
    """Get information about a USD stage.

    Args:
        stage: The USD Stage to inspect

    Returns:
        Dictionary containing stage information:
        - root_layer_path: Path to the root layer (if any)
        - up_axis: The up axis (Y or Z)
        - meters_per_unit: Scale factor for units
        - start_time_code: Start frame
        - end_time_code: End frame
        - time_codes_per_second: Frame rate
        - prim_count: Total number of prims
        - default_prim: Default prim path (if set)
        - layer_count: Number of layers in the stage

    Example:
        >>> stage = load_stage("scene.usda")
        >>> info = get_stage_info(stage)
        >>> print(f"Stage has {info['prim_count']} prims")
    """
    from pxr import UsdGeom

    root_layer = stage.GetRootLayer()

    # Get stage metrics
    info = {
        "root_layer_path": root_layer.identifier or None,
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "start_time_code": stage.GetStartTimeCode(),
        "end_time_code": stage.GetEndTimeCode(),
        "time_codes_per_second": stage.GetTimeCodesPerSecond(),
        "prim_count": sum(1 for _ in stage.Traverse()),
        "default_prim": None,
        "layer_count": len(stage.GetLayerStack()),
    }

    # Get default prim if set
    if stage.HasDefaultPrim():
        default_prim = stage.GetDefaultPrim()
        if default_prim:
            info["default_prim"] = str(default_prim.GetPath())

    return info


def collect_file_info(file_path: str | Path) -> dict[str, Any]:
    """Collect file-level metadata for a USD file.

    Args:
        file_path: Path to the USD file.

    Returns:
        Dictionary with keys: path, filename, file_size_bytes, format.
    """
    p = Path(file_path)
    suffix = p.suffix.lower()
    fmt = {".usdz": "usdz", ".usda": "usda", ".usdc": "usdc", ".usd": "usd"}.get(
        suffix, "unknown"
    )
    return {
        "path": str(p.resolve()),
        "filename": p.name,
        "file_size_bytes": p.stat().st_size,
        "format": fmt,
    }


def get_scene_extent(stage: "Usd.Stage") -> dict[str, Any] | None:
    """Compute the world-space bounding box of the entire stage.

    Uses the default prim (or pseudo-root if no default) as the root for
    bounding box computation. Returns extents in both scene units and meters.

    Args:
        stage: The USD stage to compute extent for.

    Returns:
        Dictionary with bounding_box (min/max), size_scene_units, size_meters,
        or None if computation fails.
    """
    from pxr import UsdGeom

    from world_understanding.utils.usd.prim import get_bbox_from_prim

    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        root = stage.GetPseudoRoot()
    try:
        bbox = get_bbox_from_prim(root)
        rng = bbox.ComputeAlignedRange()
        mn = rng.GetMin()
        mx = rng.GetMax()
        size = mx - mn
        mpu = UsdGeom.GetStageMetersPerUnit(stage)
        return {
            "bounding_box": {
                "min": [mn[0], mn[1], mn[2]],
                "max": [mx[0], mx[1], mx[2]],
            },
            "size_scene_units": [size[0], size[1], size[2]],
            "size_meters": [size[0] * mpu, size[1] * mpu, size[2] * mpu],
        }
    except Exception as e:
        logger.warning(f"Failed to compute scene extent: {e}")
        return None


def merge_stages(
    base_stage: "Usd.Stage",
    overlay_stage: "Usd.Stage",
    target_path: str = "/",
) -> "Usd.Stage":
    """Merge one USD stage into another.

    This creates a new stage that combines the content of both stages.
    The overlay stage's content is added under the specified target path.

    Args:
        base_stage: The base stage to merge into
        overlay_stage: The stage to merge from
        target_path: Path in the base stage where overlay content will be added

    Returns:
        A new USD Stage containing merged content

    Example:
        >>> base = load_stage("base_scene.usda")
        >>> props = load_stage("props.usda")
        >>> merged = merge_stages(base, props, "/World/Props")
    """
    from pxr import Sdf

    # Create a duplicate of the base stage
    merged_stage = duplicate_stage(base_stage, "merged.usda")

    # Create or get the target prim
    if target_path != "/":
        target_prim = merged_stage.GetPrimAtPath(target_path)
        if not target_prim:
            # Create parent path if needed
            parent_path = Sdf.Path(target_path).GetParentPath()
            if parent_path != "/":
                parent_prim = merged_stage.GetPrimAtPath(parent_path)
                if not parent_prim:
                    merged_stage.DefinePrim(parent_path)
            merged_stage.DefinePrim(target_path)

    # Helper function to copy a prim and its descendants
    def copy_prim_hierarchy(src_prim: Any, dst_path: Any) -> None:
        # Create destination prim
        dst_prim = merged_stage.DefinePrim(str(dst_path), src_prim.GetTypeName())

        # Copy attributes
        for attr in src_prim.GetAttributes():
            if attr.IsAuthored():
                dst_attr = dst_prim.CreateAttribute(
                    attr.GetName(), attr.GetTypeName(), attr.IsCustom()
                )
                # Copy value at default time
                value = attr.Get()
                if value is not None:
                    dst_attr.Set(value)
                # Copy time samples if any
                time_samples = attr.GetTimeSamples()
                for time in time_samples:
                    value = attr.Get(time)
                    if value is not None:
                        dst_attr.Set(value, time)

        # Copy relationships
        for rel in src_prim.GetRelationships():
            if rel.IsAuthored():
                dst_rel = dst_prim.CreateRelationship(rel.GetName(), rel.IsCustom())
                dst_rel.SetTargets(rel.GetTargets())

        # Recursively copy children
        for child in src_prim.GetChildren():
            child_dst_path = dst_path.AppendChild(child.GetName())
            copy_prim_hierarchy(child, child_dst_path)

    # Copy all root prims from overlay stage
    for prim in overlay_stage.GetPseudoRoot().GetChildren():
        if target_path == "/":
            dst_path = prim.GetPath()
        else:
            dst_path = Sdf.Path(target_path).AppendChild(prim.GetName())
        copy_prim_hierarchy(prim, dst_path)

    logger.info(f"Merged stages under path: {target_path}")
    return merged_stage


def create_data_uri_from_file(
    file_path: str | Path,
    *,
    mime_type: str = "model/vnd.usd",
) -> str:
    """Create a data URI from a local file.

    Args:
        file_path: Path to the local file
        mime_type: MIME type to include in the data URI

    Returns:
        Data URI string in format: data:<mime_type>;name=file.ext;base64,<data>

    Raises:
        FileNotFoundError: If the file doesn't exist
        OSError: If the file cannot be read
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not _DATA_URI_MIME_TYPE_RE.fullmatch(mime_type):
        raise ValueError(f"Invalid MIME type for data URI: {mime_type!r}")

    file_name = file_path_obj.name
    file_extension = file_path_obj.suffix

    # Read and encode the file
    try:
        file_data = file_path_obj.read_bytes()
    except OSError as e:
        logger.error("Failed to read file %s: %s", file_path, e)
        raise OSError(f"Failed to read file {file_path}: {e}") from e

    base64_data = base64.b64encode(file_data).decode("utf-8")

    # Create data URI with filename
    data_uri = f"data:{mime_type};name={file_name};base64,{base64_data}"

    logger.info(
        "Created data URI from %s (size: %d bytes, extension: %s)",
        file_path,
        len(file_data),
        file_extension,
    )

    return data_uri
