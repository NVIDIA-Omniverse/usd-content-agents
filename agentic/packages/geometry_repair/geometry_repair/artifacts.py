# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic artifact and source-preservation helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import SourceArtifact, SourcePackage

_CHUNK_SIZE = 1024 * 1024
_DEPENDENCY_DISCOVERY_FAILURE_PREFIX = "dependency discovery failed:"
_MATERIAL_DEPENDENCY_SUFFIXES = {
    ".bmp",
    ".dds",
    ".exr",
    ".hdr",
    ".jpeg",
    ".jpg",
    ".mdl",
    ".mtl",
    ".png",
    ".tga",
    ".tif",
    ".tiff",
    ".tx",
}


def classify_unresolved_dependencies(
    unresolved: list[str],
) -> tuple[list[str], list[str]]:
    """Split unresolved dependencies into geometry blockers and material debt.

    OpenUSD may report an absolute path, an asset identifier, or an exception
    prefixed with context.  Only known visual-material suffixes are safe to
    defer.  Discovery failures and every other unresolved dependency remain
    geometry blocking.
    """

    material = sorted(
        item
        for item in unresolved
        if not item.startswith(_DEPENDENCY_DISCOVERY_FAILURE_PREFIX)
        and Path(item.rsplit(": ", 1)[-1].strip("@ ")).suffix.lower()
        in _MATERIAL_DEPENDENCY_SUFFIXES
    )
    geometry = sorted(set(unresolved) - set(material))
    return geometry, material


def _file_fingerprint(path: str | Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def file_sha256(path: str | Path) -> str:
    """Return a stable SHA-256 digest for one file."""

    return _file_fingerprint(path)[0]


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Atomically replace a UTF-8 text artifact."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def atomic_write_json(path: str | Path, payload: BaseModel | dict[str, Any]) -> Path:
    """Atomically write a stable JSON document."""

    document = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    return atomic_write_text(
        path,
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def _dependency_paths(source: Path) -> tuple[list[Path], list[str]]:
    if source.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        return [], []
    try:
        from pxr import UsdUtils

        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(source))
        candidates: list[Path] = []
        for layer in layers:
            real_path = str(getattr(layer, "realPath", "") or "")
            if real_path:
                candidates.append(Path(real_path))
        for asset in assets:
            asset_path = str(getattr(asset, "resolvedPath", "") or asset)
            if asset_path:
                candidates.append(Path(asset_path))
        unique = sorted(
            {path.resolve() for path in candidates if path.is_file() and path.resolve() != source},
            key=str,
        )
        return unique, sorted(str(item) for item in unresolved)
    except Exception as exc:
        return [], [f"dependency discovery failed: {type(exc).__name__}: {exc}"]


def preserve_source(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    source_uri: str | None = None,
    source_license: str | None = None,
    source_provenance: dict[str, Any] | None = None,
    dependency_paths: list[str | Path] | None = None,
    unresolved_dependencies: list[str] | None = None,
    create_resolved_snapshot: bool = True,
) -> SourcePackage:
    """Copy an input and resolved dependencies into an immutable job snapshot."""

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Geometry repair source is not a file: {source}")
    source_dir = Path(output_dir).expanduser().resolve() / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    preserved_source = source_dir / source.name
    shutil.copy2(source, preserved_source)
    source_digest = file_sha256(source)
    if file_sha256(preserved_source) != source_digest:
        raise RuntimeError("Preserved geometry source digest does not match the input")
    source_record = SourceArtifact(
        logical_role="source",
        original_path=str(source),
        preserved_path=str(preserved_source),
        sha256=source_digest,
        size_bytes=source.stat().st_size,
    )

    if dependency_paths is None:
        dependencies, unresolved = _dependency_paths(source)
    else:
        dependencies = sorted(
            {
                Path(path).expanduser().resolve()
                for path in dependency_paths
                if Path(path).expanduser().resolve().is_file()
                and Path(path).expanduser().resolve() != source
            },
            key=str,
        )
        unresolved = sorted(set(unresolved_dependencies or []))
    dependency_records: list[SourceArtifact] = []
    dependency_dir = source_dir / "dependencies"
    for dependency in dependencies:
        source_fingerprint = _file_fingerprint(dependency)
        target = dependency_dir / f"{source_fingerprint[0][:12]}_{dependency.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dependency, target)
        preserved_fingerprint = _file_fingerprint(target)
        current_source_fingerprint = _file_fingerprint(dependency)
        if (
            preserved_fingerprint != source_fingerprint
            or current_source_fingerprint != source_fingerprint
        ):
            raise RuntimeError(
                f"Dependency source or preserved copy changed while copying: {dependency}"
            )
        dependency_records.append(
            SourceArtifact(
                logical_role="dependency",
                original_path=str(dependency),
                preserved_path=str(target),
                sha256=preserved_fingerprint[0],
                size_bytes=preserved_fingerprint[1],
            )
        )
    resolved_snapshot_path = None
    resolved_snapshot_sha256 = None
    if create_resolved_snapshot and source.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}:
        try:
            from pxr import Usd

            stage = Usd.Stage.Open(str(source))
            if stage is None:
                raise RuntimeError("Usd.Stage.Open returned None")
            snapshot = source_dir / "resolved_snapshot.usdc"
            flattened = stage.Flatten(addSourceFileComment=False)
            if not flattened.Export(str(snapshot)):
                raise RuntimeError("flattened USD snapshot export failed")
            from .mesh_io import _canonicalize_flattened_prototypes

            _canonicalize_flattened_prototypes(snapshot)
            if file_sha256(source) != source_digest:
                raise RuntimeError("source changed while the resolved snapshot was created")
            resolved_snapshot_path = str(snapshot)
            resolved_snapshot_sha256 = file_sha256(snapshot)
        except Exception as exc:
            unresolved.append(f"resolved snapshot creation failed: {type(exc).__name__}: {exc}")
    unresolved_geometry, unresolved_material = classify_unresolved_dependencies(unresolved)
    manifest_path = source_dir / "source_manifest.json"
    package = SourcePackage(
        source=source_record,
        dependencies=dependency_records,
        unresolved_dependencies=unresolved,
        unresolved_geometry_dependencies=unresolved_geometry,
        unresolved_material_dependencies=unresolved_material,
        resolved_snapshot_path=resolved_snapshot_path,
        resolved_snapshot_sha256=resolved_snapshot_sha256,
        source_uri=source_uri,
        source_license=source_license,
        source_provenance=dict(source_provenance or {}),
        manifest_path=str(manifest_path),
    )
    atomic_write_json(manifest_path, package)
    return package
