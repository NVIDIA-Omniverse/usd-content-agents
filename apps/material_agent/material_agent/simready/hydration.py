# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy hydration for indexed SimReady material selections."""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import shutil
import stat
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from filelock import FileLock, Timeout
from pxr import Sdf, Usd
from world_understanding.utils.usd.asset_paths import (
    is_absolute_asset_path,
    is_relative_to,
    is_uri_asset_path,
    resolve_relative_asset_path_under_base,
)

from material_agent.simready.catalog import SimReadyCatalogError

SIMREADY_DOWNLOAD_TIMEOUT_SECONDS = 300
SIMREADY_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
SIMREADY_LOCK_TIMEOUT_SECONDS = 600
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HydratedSimReadyLibrary:
    """Session-local SimReady material library produced for apply."""

    library_path: Path
    entries: list[dict[str, str]]
    report: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_verified_marker_path(archive_path: Path) -> Path:
    return archive_path.with_suffix(archive_path.suffix + ".verified.json")


def _archive_file_state(archive_path: Path) -> dict[str, int]:
    st = archive_path.stat()
    return {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def _archive_has_verified_marker(
    archive_path: Path,
    marker_path: Path,
    expected_digest: str,
) -> bool:
    if not archive_path.exists() or not marker_path.exists():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if not isinstance(marker, dict):
            return False
        state = _archive_file_state(archive_path)
    except (OSError, json.JSONDecodeError):
        return False
    # Marker reuse trusts the local cache directory. Initial downloads are still
    # SHA-256 verified before the marker is written.
    return (
        marker.get("sha256") == expected_digest
        and marker.get("size") == state["size"]
        and marker.get("mtime_ns") == state["mtime_ns"]
    )


def _write_archive_verified_marker(
    archive_path: Path,
    marker_path: Path,
    expected_digest: str,
) -> None:
    marker = {
        "archive": archive_path.name,
        "sha256": expected_digest,
        **_archive_file_state(archive_path),
    }
    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def _set_response_read_timeout(response: Any, timeout: float) -> None:
    """Best-effort update of the underlying socket timeout for urllib responses."""
    stack = [response]
    seen: set[int] = set()
    for _ in range(8):
        if not stack:
            return
        candidate = stack.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        settimeout = getattr(candidate, "settimeout", None)
        if callable(settimeout):
            settimeout(max(timeout, 0.001))
            return
        for attr_name in ("fp", "raw", "_sock", "sock"):
            stack.append(getattr(candidate, attr_name, None))


def _copy_url_to_path(
    url: str,
    destination: Path,
    *,
    timeout: int = SIMREADY_DOWNLOAD_TIMEOUT_SECONDS,
    max_bytes: int | None = None,
) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"", "file", "http", "https"}:
        raise SimReadyCatalogError(
            f"Unsupported SimReady archive URL scheme: {parsed.scheme}"
        )
    try:
        if parsed.scheme == "file":
            source = Path(unquote(parsed.path))
            if max_bytes is not None and source.stat().st_size > max_bytes:
                raise SimReadyCatalogError(
                    f"SimReady archive exceeds expected size limit: {source.name}"
                )
            shutil.copyfile(source, destination)
            return
        if parsed.scheme == "":
            source = Path(url)
            if source.exists():
                if max_bytes is not None and source.stat().st_size > max_bytes:
                    raise SimReadyCatalogError(
                        f"SimReady archive exceeds expected size limit: {source.name}"
                    )
                shutil.copyfile(source, destination)
                return
        deadline = time.monotonic() + timeout
        bytes_written = 0
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with destination.open("wb") as f:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"download timed out after {timeout} seconds"
                        )
                    _set_response_read_timeout(response, min(timeout, remaining))
                    chunk = response.read(SIMREADY_DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if max_bytes is not None and bytes_written > max_bytes:
                        raise SimReadyCatalogError(
                            "SimReady archive download exceeded expected size limit"
                        )
                    f.write(chunk)
    except (
        OSError,
        TimeoutError,
        urllib.error.URLError,
        ValueError,
        http.client.HTTPException,
    ) as exc:
        raise SimReadyCatalogError(
            f"Failed to download SimReady archive from {url}: {exc}"
        ) from exc


def _with_file_lock(lock_path: Path) -> FileLock:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(lock_path), timeout=SIMREADY_LOCK_TIMEOUT_SECONDS)


def _resolve_manifest_child_path(
    root: Path,
    relative_path: str,
    description: str,
) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise SimReadyCatalogError(
            f"Unsafe {description} in SimReady manifest: {relative_path}"
        )
    root_resolved = root.resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise SimReadyCatalogError(
            f"Unsafe {description} in SimReady manifest: {relative_path}"
        ) from exc
    return candidate


def _ensure_archive(asset: dict[str, Any], archive_dir: Path) -> Path:
    name = str(asset.get("name") or "")
    expected_digest = str(asset.get("sha256") or "")
    url = str(asset.get("url") or "")
    if not name or not expected_digest or not url:
        raise SimReadyCatalogError("SimReady archive metadata is incomplete")
    expected_size = asset.get("size")
    max_bytes: int | None = None
    if expected_size is not None:
        try:
            max_bytes = int(expected_size)
        except (TypeError, ValueError) as exc:
            raise SimReadyCatalogError(
                f"Invalid SimReady archive size for {name}: {expected_size!r}"
            ) from exc
        if max_bytes <= 0:
            raise SimReadyCatalogError(
                f"Invalid SimReady archive size for {name}: {expected_size!r}"
            )

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = _resolve_manifest_child_path(archive_dir, name, "archive name")
    partial_path = archive_path.with_suffix(archive_path.suffix + ".partial")
    verified_marker_path = _archive_verified_marker_path(archive_path)
    lock_path = archive_dir.parent / "locks" / f"{expected_digest}.archive.lock"
    try:
        with _with_file_lock(lock_path):
            if _archive_has_verified_marker(
                archive_path,
                verified_marker_path,
                expected_digest,
            ):
                return archive_path
            if archive_path.exists() and _sha256(archive_path) == expected_digest:
                _write_archive_verified_marker(
                    archive_path,
                    verified_marker_path,
                    expected_digest,
                )
                return archive_path

            verified_marker_path.unlink(missing_ok=True)
            partial_path.unlink(missing_ok=True)
            try:
                _copy_url_to_path(url, partial_path, max_bytes=max_bytes)
            except SimReadyCatalogError:
                partial_path.unlink(missing_ok=True)
                raise
            actual_digest = _sha256(partial_path)
            if actual_digest != expected_digest:
                partial_path.unlink(missing_ok=True)
                raise SimReadyCatalogError(
                    f"Digest mismatch for {name}: expected {expected_digest}, "
                    f"got {actual_digest}"
                )
            partial_path.replace(archive_path)
            _write_archive_verified_marker(
                archive_path,
                verified_marker_path,
                expected_digest,
            )
            return archive_path
    except Timeout as exc:
        raise SimReadyCatalogError(
            f"Timed out waiting for SimReady archive lock: {lock_path}"
        ) from exc


def _safe_extract(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if stat.S_ISLNK(info.external_attr >> 16):
                raise SimReadyCatalogError(
                    f"Unsupported symlink in SimReady archive "
                    f"{zip_path.name}: {info.filename}"
                )
            destination = (target_dir / info.filename).resolve()
            try:
                destination.relative_to(target_root)
            except ValueError as exc:
                raise SimReadyCatalogError(
                    f"Unsafe path in SimReady archive {zip_path.name}: {info.filename}"
                ) from exc
        zf.extractall(target_dir)


def _category_metadata(
    manifest: dict[str, Any],
    category: str,
    *,
    split_archives_enabled: bool,
) -> dict[str, Any]:
    categories = manifest.get("categories")
    if not isinstance(categories, dict):
        raise SimReadyCatalogError("SimReady manifest is missing categories")
    metadata = categories.get(category)
    if not isinstance(metadata, dict):
        raise SimReadyCatalogError(f"SimReady category is not indexed: {category}")
    if metadata.get("requires_split_archive"):
        if split_archives_enabled:
            raise SimReadyCatalogError(
                "Split SimReady archives require multi-file hydration support "
                f"and are not supported yet: {category}"
            )
        raise SimReadyCatalogError(
            f"Split SimReady archives are disabled and not supported yet: {category}"
        )
    return metadata


def _hydrate_category(
    manifest: dict[str, Any],
    category: str,
    cache_dir: Path,
    *,
    split_archives_enabled: bool,
) -> tuple[Path, list[Path]]:
    metadata = _category_metadata(
        manifest,
        category,
        split_archives_enabled=split_archives_enabled,
    )
    archive_files = metadata.get("archive_files")
    if not isinstance(archive_files, list) or not archive_files:
        raise SimReadyCatalogError(f"SimReady category has no archives: {category}")
    if len(archive_files) > 1:
        if split_archives_enabled:
            raise SimReadyCatalogError(
                "Split SimReady archives require multi-file hydration support "
                f"and are not supported yet: {category}"
            )
        raise SimReadyCatalogError(
            f"Split SimReady archives are disabled and not supported yet: {category}"
        )

    release_tag = str(manifest.get("release_tag") or "unknown")
    repository = str(manifest.get("repository") or "unknown").replace("/", "__")
    category_cache = cache_dir / repository / release_tag / category
    archive_dir = category_cache / "archives"
    archive_path = _ensure_archive(archive_files[0], archive_dir)
    digest = str(archive_files[0]["sha256"])
    extracted_dir = category_cache / "extracted" / digest[:16]
    marker_path = extracted_dir / ".simready_extracted.json"
    lock_path = category_cache / "locks" / f"{digest}.extract.lock"
    try:
        with _with_file_lock(lock_path):
            if not marker_path.exists():
                partial_dir = extracted_dir.with_name(extracted_dir.name + ".partial")
                shutil.rmtree(partial_dir, ignore_errors=True)
                _safe_extract(archive_path, partial_dir)
                marker = {
                    "category": category,
                    "archive": archive_path.name,
                    "sha256": digest,
                    "release_tag": release_tag,
                }
                marker_tmp = partial_dir / ".simready_extracted.json"
                marker_tmp.write_text(
                    json.dumps(marker, indent=2) + "\n",
                    encoding="utf-8",
                )
                shutil.rmtree(extracted_dir, ignore_errors=True)
                partial_dir.replace(extracted_dir)
    except Timeout as exc:
        raise SimReadyCatalogError(
            f"Timed out waiting for SimReady extraction lock: {lock_path}"
        ) from exc
    return extracted_dir, [archive_path]


def _create_parent_prims(layer: Sdf.Layer, path: Sdf.Path) -> None:
    parents: list[Sdf.Path] = []
    parent = path.GetParentPath()
    while parent != Sdf.Path.absoluteRootPath:
        parents.append(parent)
        parent = parent.GetParentPath()
    for parent_path in reversed(parents):
        if not layer.GetPrimAtPath(parent_path):
            Sdf.CreatePrimInLayer(layer, parent_path)


def _copy_and_remap_asset_path(
    path: str,
    *,
    source_dir: Path,
    library_dir: Path,
    asset_dir: Path,
    material_binding: str,
    skipped_assets: list[dict[str, str]] | None,
) -> str:
    def skip(reason: str, resolved_path: Path | None = None) -> str:
        logger.warning(
            "Skipping %s SimReady asset path while hydrating %s: %s",
            reason,
            material_binding,
            path,
        )
        if skipped_assets is not None:
            skipped_asset = {
                "material_binding": material_binding,
                "path": path,
                "reason": reason,
                "source_dir": str(source_dir),
            }
            if resolved_path is not None:
                skipped_asset["resolved_path"] = str(resolved_path)
            skipped_assets.append(skipped_asset)
        return ""

    if not path:
        return path
    if is_uri_asset_path(path):
        return skip("external URI")
    if is_absolute_asset_path(path):
        source_path = Path(path).resolve()
        if not is_relative_to(source_path, source_dir.resolve()):
            return skip("out-of-tree absolute path", source_path)
    else:
        try:
            source_path = resolve_relative_asset_path_under_base(path, source_dir)
        except ValueError:
            return skip("unsafe relative path")
    if not source_path.exists():
        return skip("missing file", source_path)

    relative_source_path = source_path.relative_to(source_dir.resolve())
    destination = asset_dir / relative_source_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return str(destination.relative_to(library_dir.resolve())).replace("\\", "/")


def _copy_and_remap_asset_value(
    value: Any,
    *,
    source_dir: Path,
    library_dir: Path,
    asset_dir: Path,
    material_binding: str,
    skipped_assets: list[dict[str, str]] | None,
) -> Any:
    if isinstance(value, Sdf.AssetPath):
        new_path = _copy_and_remap_asset_path(
            value.path,
            source_dir=source_dir,
            library_dir=library_dir,
            asset_dir=asset_dir,
            material_binding=material_binding,
            skipped_assets=skipped_assets,
        )
        if new_path != value.path:
            return Sdf.AssetPath(new_path)
    elif isinstance(value, Sdf.AssetPathArray):
        new_arr = Sdf.AssetPathArray(
            [
                Sdf.AssetPath(
                    _copy_and_remap_asset_path(
                        asset_path.path,
                        source_dir=source_dir,
                        library_dir=library_dir,
                        asset_dir=asset_dir,
                        material_binding=material_binding,
                        skipped_assets=skipped_assets,
                    )
                )
                for asset_path in value
            ]
        )
        if new_arr != value:
            return new_arr
    return value


def _copy_and_remap_asset_paths_in_prim(
    layer: Sdf.Layer,
    prim_path: Sdf.Path,
    *,
    source_dir: Path,
    library_dir: Path,
    asset_dir: Path,
    material_binding: str,
    skipped_assets: list[dict[str, str]] | None,
) -> None:
    prim_spec = layer.GetPrimAtPath(prim_path)
    if not prim_spec:
        return

    for attr_name in list(prim_spec.attributes.keys()):
        attr_spec = prim_spec.attributes[attr_name]
        value = attr_spec.default
        remapped_value = _copy_and_remap_asset_value(
            value,
            source_dir=source_dir,
            library_dir=library_dir,
            asset_dir=asset_dir,
            material_binding=material_binding,
            skipped_assets=skipped_assets,
        )
        if remapped_value != value:
            attr_spec.default = remapped_value

        for time_code in attr_spec.ListTimeSamples():
            sample_value = attr_spec.QueryTimeSample(time_code)
            remapped_sample = _copy_and_remap_asset_value(
                sample_value,
                source_dir=source_dir,
                library_dir=library_dir,
                asset_dir=asset_dir,
                material_binding=material_binding,
                skipped_assets=skipped_assets,
            )
            if remapped_sample != sample_value:
                attr_spec.SetTimeSample(time_code, remapped_sample)

    for child_spec in prim_spec.nameChildren:
        _copy_and_remap_asset_paths_in_prim(
            layer,
            prim_path.AppendChild(child_spec.name),
            source_dir=source_dir,
            library_dir=library_dir,
            asset_dir=asset_dir,
            material_binding=material_binding,
            skipped_assets=skipped_assets,
        )


def _asset_dir_for_material_binding(output_dir: Path, target_path: Sdf.Path) -> Path:
    path_parts = [part for part in str(target_path).strip("/").split("/") if part]
    if path_parts[:2] == ["World", "Looks"]:
        path_parts = path_parts[2:]
    if not path_parts:
        path_parts = [target_path.name or "material"]
    return output_dir.resolve().joinpath("assets", *path_parts)


def _copy_flattened_material(
    *,
    source_file: Path,
    target_layer: Sdf.Layer,
    target_binding: str,
    output_dir: Path,
    skipped_assets: list[dict[str, str]],
) -> None:
    from material_agent.tasks.apply_materials_to_usd import (
        clear_color_space_on_empty_asset_inputs,
    )

    stage = Usd.Stage.Open(str(source_file))
    if stage is None:
        raise SimReadyCatalogError(f"Could not open SimReady material: {source_file}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise SimReadyCatalogError(
            f"SimReady material has no default prim: {source_file}"
        )

    flat_layer = stage.Flatten()
    source_path = default_prim.GetPath()
    target_path = Sdf.Path(target_binding)
    _create_parent_prims(target_layer, target_path)
    if not Sdf.CopySpec(flat_layer, source_path, target_layer, target_path):
        raise SimReadyCatalogError(
            f"Failed to copy SimReady material {source_file} to {target_binding}"
        )
    _copy_and_remap_asset_paths_in_prim(
        target_layer,
        target_path,
        source_dir=source_file.resolve().parent,
        library_dir=output_dir.resolve(),
        asset_dir=_asset_dir_for_material_binding(output_dir, target_path),
        material_binding=target_binding,
        skipped_assets=skipped_assets,
    )
    clear_color_space_on_empty_asset_inputs(target_layer, target_path)


def hydrate_simready_library(
    *,
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    material_names: set[str],
    cache_dir: str | Path,
    output_dir: str | Path,
    split_archives_enabled: bool = False,
    listener: Any | None = None,
) -> HydratedSimReadyLibrary:
    """Hydrate selected SimReady entries into a session-local USD library."""
    selected_entries = [
        entry
        for entry in entries
        if entry.get("name") in material_names and entry.get("simready_source_path")
    ]
    if not selected_entries:
        raise SimReadyCatalogError(
            "No SimReady material entries selected for hydration"
        )

    cache_root = Path(cache_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    library_path = output_root / "simready_material_library.usda"
    if library_path.exists():
        library_path.unlink()

    categories = sorted({str(entry["simready_category"]) for entry in selected_entries})
    if listener is not None:
        listener.info(
            f"Hydrating SimReady material categories: {', '.join(categories)}"
        )
    extracted_by_category: dict[str, Path] = {}
    archive_paths: list[Path] = []
    for category in categories:
        extracted_dir, category_archives = _hydrate_category(
            manifest,
            category,
            cache_root,
            split_archives_enabled=split_archives_enabled,
        )
        extracted_by_category[category] = extracted_dir
        archive_paths.extend(category_archives)

    layer = Sdf.Layer.CreateNew(str(library_path))
    layer.defaultPrim = "World"
    hydrated_entries: list[dict[str, str]] = []
    skipped_assets: list[dict[str, str]] = []
    for entry in selected_entries:
        category = str(entry["simready_category"])
        source_file = _resolve_manifest_child_path(
            extracted_by_category[category],
            str(entry["simready_source_path"]),
            "material source path",
        )
        if not source_file.exists():
            raise SimReadyCatalogError(
                f"SimReady material file not found: {source_file}"
            )
        hydrated_entry = {
            "name": str(entry["name"]),
            "description": str(entry.get("description") or ""),
            "binding": str(entry["binding"]),
        }
        _copy_flattened_material(
            source_file=source_file,
            target_layer=layer,
            target_binding=hydrated_entry["binding"],
            output_dir=library_path.parent,
            skipped_assets=skipped_assets,
        )
        hydrated_entries.append(hydrated_entry)

    layer.Save()
    if skipped_assets and listener is not None:
        listener.warning(
            f"Skipped {len(skipped_assets)} SimReady asset path(s) during hydration; "
            "see simready_hydration_report.json for details"
        )
    report = {
        "release_tag": manifest.get("release_tag"),
        "schema_version": manifest.get("schema_version"),
        "library_path": str(library_path),
        "material_count": len(hydrated_entries),
        "materials": [entry["name"] for entry in hydrated_entries],
        "categories": categories,
        "archives": [str(path) for path in archive_paths],
        "cache_dir": str(cache_root),
        "skipped_asset_count": len(skipped_assets),
        "skipped_assets": skipped_assets,
    }
    (output_root / "simready_hydration_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return HydratedSimReadyLibrary(
        library_path=library_path,
        entries=hydrated_entries,
        report=report,
    )
