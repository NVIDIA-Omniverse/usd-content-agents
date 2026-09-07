# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for USD asset paths that may be resolved outside Python.

USD layers, references, and material attributes can be evaluated by native
ArResolver code in renderers. These helpers keep that boundary conservative:
generated output should author local relative paths, not resolver schemes or
host absolute paths that bypass Python-side URL and filesystem checks.
"""

from __future__ import annotations

import os
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import Any

_UDIM_TOKEN = "<UDIM>"
_UDIM_TILE_MIN = 1001
_UDIM_TILE_MAX = 1100


def is_windows_drive_path(path: str) -> bool:
    """Return true for Windows drive-absolute paths such as ``C:/foo``."""
    return (
        len(path) >= 3
        and path[0].isalpha()
        and path[1] == ":"
        and path[2] in {"/", "\\"}
    )


def usd_asset_uri_scheme(path: str) -> str:
    """Return a resolver/URI scheme for a USD asset path, if one is present."""
    if not path or is_windows_drive_path(path):
        return ""

    colon_index = path.find(":")
    if colon_index <= 0:
        return ""

    first_separator = len(path)
    for separator in ("/", "\\"):
        separator_index = path.find(separator)
        if separator_index >= 0:
            first_separator = min(first_separator, separator_index)
    if first_separator < colon_index:
        return ""

    scheme = path[:colon_index]
    if not scheme[0].isalpha():
        return ""
    if not all(char.isalnum() or char in {"+", ".", "-"} for char in scheme):
        return ""
    return scheme.lower()


def is_uri_asset_path(path: str) -> bool:
    """Return true when a USD asset path uses a resolver/URI scheme."""
    return bool(usd_asset_uri_scheme(path))


def is_absolute_asset_path(path: str) -> bool:
    """Return whether an asset path is absolute on POSIX or Windows."""
    return path.startswith("/") or os.path.isabs(path) or is_windows_drive_path(path)


def is_bare_mdl_asset_path(path: object) -> bool:
    """Return whether ``path`` is a renderer-resolved bare MDL module token."""
    token = str(path).strip("@")
    return (
        bool(token)
        and ":" not in token
        and "/" not in token
        and "\\" not in token
        and Path(token).suffix.lower() == ".mdl"
    )


def is_unsafe_resolver_asset_path(path: str) -> bool:
    """Return true for paths that should not be authored into generated USD."""
    return is_uri_asset_path(path) or is_absolute_asset_path(path)


def is_relative_to(path: Path, base: Path) -> bool:
    """Compatibility wrapper for ``Path.is_relative_to``."""
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def expand_udim_asset_pattern(pattern: str) -> tuple[str, ...]:
    """Return OpenUSD's bounded concrete tile names for one UDIM pattern.

    Only a single literal ``<UDIM>`` token in the final path component is
    supported. Keeping expansion exact and bounded avoids treating arbitrary
    resolver expressions as filesystem globs while allowing callers to prove
    that every dependency they accept is a real confined file or package
    member.
    """
    normalized = pattern.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        normalized.count(_UDIM_TOKEN) != 1
        or _UDIM_TOKEN not in path.name
        or _UDIM_TOKEN in path.parent.as_posix()
    ):
        raise ValueError(
            "UDIM asset path must contain exactly one <UDIM> token in its "
            f"filename: {pattern}"
        )
    return tuple(
        normalized.replace(_UDIM_TOKEN, str(tile))
        for tile in range(_UDIM_TILE_MIN, _UDIM_TILE_MAX + 1)
    )


def resolve_layer_udim_asset_paths(
    layer: Any,
    authored_path: str,
    *,
    usd_shade: Any,
) -> tuple[str, tuple[str, ...]]:
    """Resolve one authored UDIM pattern and all of its concrete tiles.

    ``Sdf.ComputeAssetPathRelativeToLayer`` intentionally leaves unresolved
    UDIM patterns unchanged. ``UsdShade.UdimUtils`` is the OpenUSD API that
    preserves the owning-layer anchor for both filesystem layers and USDZ
    members. Callers must still apply their own confinement and file policies
    to the returned identifiers.
    """
    expand_udim_asset_pattern(authored_path)
    try:
        anchored_pattern = str(
            usd_shade.UdimUtils.ResolveUdimPath(authored_path, layer)
        )
        concrete_paths = tuple(
            str(path)
            for path, _tile in usd_shade.UdimUtils.ResolveUdimTilePaths(
                authored_path,
                layer,
            )
            if str(path)
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Unable to resolve authored UDIM asset path: {authored_path}"
        ) from exc
    if not concrete_paths:
        raise ValueError(
            f"Authored UDIM asset path has no concrete tiles: {authored_path}"
        )
    if not anchored_pattern:
        raise ValueError(f"Unable to anchor authored UDIM asset path: {authored_path}")
    return anchored_pattern, concrete_paths


def resolve_relative_asset_path_under_base(path: str, base_dir: Path) -> Path:
    """Resolve a local relative USD asset path and require it to stay in base_dir."""
    if not path:
        raise ValueError("empty asset path")
    if is_uri_asset_path(path):
        raise ValueError(f"resolver URI schemes are not allowed: {path}")
    if is_absolute_asset_path(path):
        raise ValueError(f"absolute asset paths are not allowed: {path}")

    resolved_base = base_dir.resolve()
    resolved_path = (resolved_base / path).resolve()
    if not is_relative_to(resolved_path, resolved_base):
        raise ValueError(f"asset path escapes its source directory: {path}")
    return resolved_path


def require_authored_dependency_file(
    dependency: str,
    *,
    split_identifier: Callable[[str], tuple[str, str | None]],
    layer_identifier: str,
    authored_path: str,
    package_members_cache: dict[Path, frozenset[str]],
) -> None:
    """Require one confined dependency to exist, caching USDZ member indexes."""
    package_path_text, member = split_identifier(dependency)
    try:
        package_path = Path(package_path_text).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            "Authored USD dependency does not resolve to a file: "
            f"layer={layer_identifier!r}, path={authored_path!r}, "
            f"resolved={dependency!r}"
        ) from exc
    if not package_path.is_file():
        raise ValueError(
            "Authored USD dependency does not resolve to a file: "
            f"layer={layer_identifier!r}, path={authored_path!r}, "
            f"resolved={dependency!r}"
        )
    if member is None:
        return

    member_names = package_members_cache.get(package_path)
    if member_names is None:
        try:
            with zipfile.ZipFile(package_path) as package:
                member_names = frozenset(package.namelist())
        except (OSError, zipfile.BadZipFile) as exc:
            raise ValueError(
                "Authored USD dependency package is unreadable: "
                f"layer={layer_identifier!r}, path={authored_path!r}, "
                f"resolved={dependency!r}"
            ) from exc
        package_members_cache[package_path] = member_names
    if member not in member_names:
        raise ValueError(
            "Authored USD dependency package member is missing: "
            f"layer={layer_identifier!r}, path={authored_path!r}, "
            f"resolved={dependency!r}"
        )


def _iter_sdf_asset_paths(value: Any, *, sdf: Any) -> Iterable[str]:
    """Yield authored paths from nested SdfAssetPath-valued metadata."""
    if isinstance(value, sdf.AssetPath):
        if value.path:
            yield str(value.path)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_sdf_asset_paths(item, sdf=sdf)
        return
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        for item in value:
            yield from _iter_sdf_asset_paths(item, sdf=sdf)


def _layer_identifier(layer: Any) -> str:
    identifier = getattr(layer, "identifier", None)
    return str(identifier) if identifier else "<unknown>"


def collect_layer_authored_asset_paths(
    layer: Any,
    *,
    sdf: Any,
) -> tuple[set[str], set[str]]:
    """Collect raw authored asset paths without invoking stage composition.

    ``Sdf.Layer.Traverse`` walks relationship-target and attribute-connection
    list-op nodes as well as concrete specs. OpenUSD 25.05 exposes those target
    paths through traversal while ``GetObjectAtPath`` intentionally returns no
    Python spec for them. Only those validated target nodes are skipped; every
    other missing spec or unreadable metadata value fails closed with layer and
    path context.

    The first result contains every authored asset path. The second contains
    composition dependencies (sublayers, references, and payloads), allowing
    callers to recurse through layers while retaining their own confinement and
    resolver policies.
    """
    layer_identifier = _layer_identifier(layer)
    all_paths = {str(path) for path in layer.subLayerPaths if str(path)}
    composition_paths = set(all_paths)

    def visit(path: Any) -> None:
        spec = layer.GetObjectAtPath(path)
        if spec is None:
            is_target_path = bool(getattr(path, "IsTargetPath", lambda: False)())
            if is_target_path:
                owner_path = path.GetParentPath()
                owner_spec = layer.GetObjectAtPath(owner_path)
                if isinstance(owner_spec, sdf.RelationshipSpec | sdf.AttributeSpec):
                    return
                raise ValueError(
                    "USD layer traversal target has no relationship or attribute "
                    f"owner: layer={layer_identifier!r}, path={path}, "
                    f"owner_path={owner_path}"
                )
            raise ValueError(
                "USD layer traversal returned no spec: "
                f"layer={layer_identifier!r}, path={path}"
            )

        try:
            keys = spec.ListInfoKeys()
        except Exception as exc:
            raise ValueError(
                "Unable to list USD metadata safely: "
                f"layer={layer_identifier!r}, path={path}"
            ) from exc

        for key in keys:
            # OpenUSD 25.05 exposes pseudo-root subLayerOffsets even though its
            # value has no Python converter. It contains only SdfLayerOffset
            # values; subLayerPaths above holds the corresponding asset paths.
            if str(key) == "subLayerOffsets":
                continue
            try:
                value = spec.GetInfo(key)
                authored_paths = tuple(_iter_sdf_asset_paths(value, sdf=sdf))
            except Exception as exc:
                raise ValueError(
                    "Unable to inspect USD metadata safely: "
                    f"layer={layer_identifier!r}, path={path}, key={key!s}"
                ) from exc
            all_paths.update(authored_paths)

    layer.Traverse(sdf.Path.absoluteRootPath, visit)

    composition_getter = getattr(layer, "GetCompositionAssetDependencies", None)
    if composition_getter is None:
        composition_getter = layer.GetExternalReferences
    try:
        composition_dependencies = composition_getter()
    except Exception as exc:
        raise ValueError(
            "Unable to inspect USD composition dependencies safely: "
            f"layer={layer_identifier!r}"
        ) from exc
    for external in composition_dependencies:
        if external:
            authored = str(external)
            all_paths.add(authored)
            composition_paths.add(authored)

    external_getter = getattr(layer, "GetExternalAssetDependencies", None)
    if external_getter is not None:
        try:
            external_dependencies = external_getter()
        except Exception as exc:
            raise ValueError(
                "Unable to inspect USD asset dependencies safely: "
                f"layer={layer_identifier!r}"
            ) from exc
        all_paths.update(str(path) for path in external_dependencies if path)

    return all_paths, composition_paths
