# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity binding for USD assets and their resolved dependency closure."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import Any

ASSET_DEPENDENCY_MANIFEST_SCHEMA = "content-agent-workflows.usd-dependency-manifest.v1"
_REMOTE_ASSET_PREFIXES = ("anon:", "http://", "https://", "omniverse:")


class AssetDependencyIdentityError(RuntimeError):
    """Raised when a complete, stable local USD dependency identity is unavailable."""

    def __init__(
        self,
        message: str,
        *,
        dependency_paths: Iterable[Path] = (),
        path_inventory_complete: bool = False,
    ) -> None:
        super().__init__(message)
        self.dependency_paths = frozenset(
            path.expanduser().resolve() for path in dependency_paths
        )
        self.path_inventory_complete = path_inventory_complete


def build_asset_dependency_manifest(
    asset_path: str | Path,
    *,
    is_runtime_asset_path: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Return a canonical identity for the root and resolved local dependencies.

    An optional runtime predicate excludes resolver-owned paths; their authored
    references remain bound by the root or local layer bytes in the manifest.
    """

    root = Path(asset_path).expanduser().resolve()
    try:
        root_metadata = root.stat()
    except OSError as exc:
        raise AssetDependencyIdentityError(
            f"Could not inspect dependency {root}: {exc}",
            dependency_paths=(root,),
            path_inventory_complete=True,
        ) from exc
    if not stat.S_ISREG(root_metadata.st_mode):
        raise AssetDependencyIdentityError(
            f"USD dependency is not a regular file: {root}",
            dependency_paths=(root,),
            path_inventory_complete=True,
        )
    paths = {root}
    if root.suffix.lower() != ".usdz":
        paths.update(
            _resolved_usd_dependencies(
                root,
                is_runtime_asset_path=is_runtime_asset_path,
            )
        )

    try:
        files = [
            {
                "path": str(path),
                "role": "root" if path == root else "dependency",
                **_stable_file_identity(path),
            }
            for path in sorted(paths, key=lambda candidate: str(candidate))
        ]
    except AssetDependencyIdentityError as exc:
        raise AssetDependencyIdentityError(
            str(exc),
            dependency_paths=paths,
            path_inventory_complete=True,
        ) from exc
    unsigned = {
        "schema": ASSET_DEPENDENCY_MANIFEST_SCHEMA,
        "asset_path": str(root),
        "files": files,
    }
    return {**unsigned, "sha256": _canonical_json_sha256(unsigned)}


def asset_dependency_identity_errors(
    asset_path: str | Path,
    expected_manifest: Any,
) -> list[str]:
    """Return fail-closed errors when the current dependency closure differs."""

    if not isinstance(expected_manifest, dict):
        return ["Asset dependency manifest is missing or is not a JSON object."]
    try:
        current_manifest = build_asset_dependency_manifest(asset_path)
    except AssetDependencyIdentityError as exc:
        return [f"Could not verify asset dependency identity: {exc}"]
    if current_manifest != expected_manifest:
        return [
            "Asset or resolved USD dependency content changed after validation identity "
            "was captured."
        ]
    return []


def asset_dependency_root_sha256(manifest: dict[str, Any]) -> str:
    """Return the uniquely bound root digest from a generated manifest."""

    files = manifest.get("files")
    roots = (
        [
            item
            for item in files
            if isinstance(item, dict) and item.get("role") == "root"
        ]
        if isinstance(files, list)
        else []
    )
    if len(roots) != 1:
        raise AssetDependencyIdentityError(
            "USD dependency manifest does not contain exactly one root file."
        )
    digest = roots[0].get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise AssetDependencyIdentityError(
            "USD dependency manifest root has no valid SHA-256 digest."
        )
    return digest


def _resolved_usd_dependencies(
    root: Path,
    *,
    is_runtime_asset_path: Callable[[str], bool] | None = None,
) -> set[Path]:
    try:
        from pxr import Sdf, UsdUtils
    except ImportError as exc:
        raise AssetDependencyIdentityError(
            "OpenUSD Python APIs are required to resolve dependency identity."
        ) from exc

    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(root))
    except Exception as exc:
        raise AssetDependencyIdentityError(
            f"OpenUSD could not enumerate dependencies for {root}: {exc}"
        ) from exc
    _reject_remote_composition_arcs(layers, Sdf=Sdf)
    dependencies: set[Path] = set()
    for layer in layers:
        identifier = getattr(layer, "identifier", None)
        if _is_runtime_asset(identifier, is_runtime_asset_path):
            continue
        _reject_nonlocal_identifier(identifier)
        candidate = _dependency_path(
            getattr(layer, "realPath", None)
            or getattr(layer, "resolvedPath", None)
            or identifier,
            root.parent,
        )
        if candidate is not None:
            dependencies.add(candidate)
    for asset in assets:
        authored_path = getattr(asset, "path", None) or asset
        if _is_runtime_asset(authored_path, is_runtime_asset_path):
            continue
        _reject_nonlocal_identifier(authored_path)
        candidate = _dependency_path(
            getattr(asset, "resolvedPath", None) or authored_path or asset,
            root.parent,
        )
        if candidate is None:
            raise AssetDependencyIdentityError(
                f"USD dependency has no local resolved path: {asset}"
            )
        dependencies.add(candidate)

    rejected_unresolved = [
        item
        for item in unresolved
        if not _is_runtime_asset(item, is_runtime_asset_path)
    ]
    unresolved_paths: set[Path] = set()
    unresolved_inventory_complete = True
    for item in rejected_unresolved:
        try:
            _reject_nonlocal_identifier(item)
            candidate = _dependency_path(item, root.parent)
        except AssetDependencyIdentityError:
            unresolved_inventory_complete = False
            continue
        if candidate is None:
            unresolved_inventory_complete = False
            continue
        unresolved_paths.add(candidate)
    if rejected_unresolved:
        sample = ", ".join(str(item) for item in rejected_unresolved[:5])
        suffix = "" if len(rejected_unresolved) <= 5 else ", ..."
        raise AssetDependencyIdentityError(
            f"USD contains unresolved dependencies: {sample}{suffix}",
            dependency_paths={root, *dependencies, *unresolved_paths},
            path_inventory_complete=unresolved_inventory_complete,
        )
    return dependencies


def _reject_remote_composition_arcs(layers: Any, *, Sdf: Any) -> None:
    """Reject composition layers that cannot be included in the local identity."""

    for layer in layers:
        for sublayer_path in layer.subLayerPaths:
            _reject_nonlocal_identifier(sublayer_path)

        def reject_prim_arcs(path: Any) -> None:
            spec = layer.GetObjectAtPath(path)
            if not isinstance(spec, Sdf.PrimSpec):
                return
            for list_editor in (spec.referenceList, spec.payloadList):
                for item in list_editor.GetAppliedItems():
                    asset_path = getattr(item, "assetPath", "")
                    if asset_path:
                        _reject_nonlocal_identifier(asset_path)

        layer.Traverse(Sdf.Path.absoluteRootPath, reject_prim_arcs)


def _is_runtime_asset(
    value: Any,
    predicate: Callable[[str], bool] | None,
) -> bool:
    return predicate is not None and predicate(str(value or "").strip().strip("@"))


def _reject_nonlocal_identifier(value: Any) -> None:
    text = str(value or "").strip().strip("@").lower()
    if text.startswith(_REMOTE_ASSET_PREFIXES):
        raise AssetDependencyIdentityError(
            f"Remote USD dependency cannot be content-bound locally: {value}"
        )


def _dependency_path(value: Any, default_root: Path) -> Path | None:
    text = str(value or "").strip().strip("@")
    if not text or text.startswith("anon:"):
        return None
    if text.startswith(_REMOTE_ASSET_PREFIXES[1:]):
        raise AssetDependencyIdentityError(
            f"Remote USD dependency cannot be content-bound locally: {text}"
        )
    if "[" in text or "]" in text:
        # A flattened derivative can retain resolved texture/layer paths into its
        # source USDZ. Bind the complete archive bytes, not an unhashable member
        # locator. Unsupported or unresolved package references still fail closed.
        from pxr import Ar

        if not Ar.IsPackageRelativePath(text):
            raise AssetDependencyIdentityError(f"Malformed packaged dependency: {text}")
        archive, member = Ar.SplitPackageRelativePathOuter(text)
        _reject_nonlocal_identifier(archive)
        member_path = PurePosixPath(member)
        if (
            not archive.lower().endswith(".usdz")
            or not member
            or member_path.is_absolute()
            or ".." in member_path.parts
            or any(c in archive + member for c in "[]\\")
            or ":" in member
        ):
            raise AssetDependencyIdentityError(f"Unsupported packaged dependency: {text}")
        archive_path = Path(archive).expanduser()
        if not archive_path.is_absolute():
            archive_path = default_root / archive_path
        archive_path = archive_path.resolve()
        locator = f"{archive_path}[{member}]"
        if not Ar.GetResolver().Resolve(locator):
            raise AssetDependencyIdentityError(f"Unresolved packaged dependency: {text}")
        return archive_path
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = default_root / path
    return path.resolve()


def _stable_file_identity(path: Path) -> dict[str, Any]:
    try:
        before = path.stat()
    except OSError as exc:
        raise AssetDependencyIdentityError(
            f"Could not inspect dependency {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise AssetDependencyIdentityError(
            f"USD dependency is not a regular file: {path}"
        )

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise AssetDependencyIdentityError(
            f"Could not hash dependency {path}: {exc}"
        ) from exc
    if _stat_identity(before) != _stat_identity(after):
        raise AssetDependencyIdentityError(
            f"USD dependency changed while hashing: {path}"
        )
    return {"sha256": digest.hexdigest(), "size": before.st_size}


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
