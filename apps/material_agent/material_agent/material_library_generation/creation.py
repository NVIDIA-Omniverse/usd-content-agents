# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material creation orchestration and package registration."""

from __future__ import annotations

import json
import os
import shutil
import struct
import threading
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from material_agent.material_library_generation.creation_contract import (
    MATERIAL_CREATION_MANIFEST_NAME,
    MATERIAL_CREATION_SCHEMA_VERSION,
    BackendMaterialResult,
    CreatedMaterial,
    CreatedMaterialListEntry,
    CreateMaterialRequest,
    MaterialArtifactLayout,
    MaterialChannel,
    MaterialChannelArtifact,
    MaterialChannelComponent,
    MaterialChannelSource,
    MaterialColorSpace,
    MaterialComponentProvenance,
    MaterialCreationBackend,
    MaterialCreationDiagnostic,
    MaterialCreationError,
    MaterialCreationErrorCode,
    MaterialCreationProvenance,
    MaterialDegradation,
    MaterialDegradationCode,
    MaterialDiagnosticSeverity,
    NormalConvention,
    ORMPacking,
    PreparedMaterialConditioning,
)
from material_agent.material_library_generation.schema import (
    GeneratedMaterial,
    TextureMapSet,
)
from material_agent.material_library_generation.usd_authoring import (
    MaterialAuthoringError,
    inspect_material_library_authoring,
    require_material_authoring_prerequisites,
    write_material_library_usd,
)
from material_agent.material_profiles import normalize_material_profile


class MaterialCreationBackendRegistry:
    """Run-local registry for material creation backends."""

    def __init__(self) -> None:
        self._backends: dict[str, MaterialCreationBackend] = {}
        self._default_backend_name: str | None = None

    @property
    def default_backend_name(self) -> str | None:
        return self._default_backend_name

    @property
    def backend_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))

    def register(
        self,
        backend: MaterialCreationBackend,
        *,
        make_default: bool = False,
        replace_existing: bool = False,
    ) -> MaterialCreationBackend:
        name = backend.name.strip()
        if not name:
            raise ValueError("material creation backend name must be non-empty")
        if name in self._backends and not replace_existing:
            raise ValueError(f"material creation backend already registered: {name}")
        self._backends[name] = backend
        if make_default or self._default_backend_name is None:
            self._default_backend_name = name
        return backend

    def resolve(self, requested: str) -> MaterialCreationBackend:
        requested = requested.strip()
        if requested == "auto":
            if self._default_backend_name is None:
                raise MaterialCreationError(
                    MaterialCreationErrorCode.BACKEND_UNAVAILABLE,
                    "No default material creation backend is registered.",
                    backend="auto",
                    retryable=False,
                )
            requested = self._default_backend_name
        try:
            return self._backends[requested]
        except KeyError as exc:
            raise MaterialCreationError(
                MaterialCreationErrorCode.BACKEND_UNAVAILABLE,
                f"Material creation backend is not registered: {requested}",
                backend=requested,
                retryable=False,
            ) from exc


def _authoring_creation_error(
    error: MaterialAuthoringError,
    *,
    backend: str,
) -> MaterialCreationError:
    prerequisite = error.code == "OPENPBR_MATERIALX_AUTHORING_UNAVAILABLE"
    return MaterialCreationError(
        (
            MaterialCreationErrorCode.BACKEND_UNAVAILABLE
            if prerequisite
            else MaterialCreationErrorCode.INVALID_OUTPUT
        ),
        str(error),
        backend=backend,
        retryable=False,
        diagnostics=(
            MaterialCreationDiagnostic(
                code=error.code,
                message=str(error),
                severity=MaterialDiagnosticSeverity.ERROR,
                phase="authoring_preflight" if prerequisite else "authoring_validation",
                retryable=False,
                details=error.to_dict()["details"],
            ),
        ),
    )


def create_material_package(
    request: CreateMaterialRequest,
    package_dir: str | Path,
    *,
    registry: MaterialCreationBackendRegistry,
    conditioning: PreparedMaterialConditioning | None = None,
    cancel_event: threading.Event | None = None,
    material_profile: str = "auto",
    overwrite: bool = False,
) -> CreatedMaterial:
    """Run a backend and package one created material for run-local assignment."""

    backend = registry.resolve(request.backend)
    try:
        normalized_material_profile = normalize_material_profile(material_profile)
    except ValueError as exc:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_REQUEST,
            str(exc),
            backend=backend.name,
            retryable=False,
        ) from exc
    if normalized_material_profile == "display_color":
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_REQUEST,
            "material_profile='display_color' is apply-only and cannot author a "
            "created-material package.",
            backend=backend.name,
            retryable=False,
        )
    layout = MaterialArtifactLayout(Path(package_dir), request.recipe.material_id)

    if layout.creation_manifest_path.exists() and not overwrite:
        return _load_cached_created_material(
            layout,
            request,
            backend,
            conditioning=conditioning,
            material_profile=normalized_material_profile,
        )

    try:
        require_material_authoring_prerequisites(normalized_material_profile)
    except MaterialAuthoringError as exc:
        raise _authoring_creation_error(exc, backend=backend.name) from exc

    if layout.package_dir.exists():
        if not overwrite:
            raise MaterialCreationError(
                MaterialCreationErrorCode.INVALID_OUTPUT,
                "Material package already exists without a reusable creation manifest.",
                backend=backend.name,
                retryable=False,
            )
        shutil.rmtree(layout.package_dir)

    created_package_dir = False
    try:
        layout.package_dir.mkdir(parents=True, exist_ok=False)
        created_package_dir = True
        result = backend.create(
            request,
            output_dir=layout.package_dir,
            conditioning=conditioning,
            cancel_event=cancel_event,
        )
        _validate_backend_result(result, request, backend, conditioning=conditioning)
        packaged_artifacts = _packaged_artifacts(result, layout)
        generated = _generated_material(request, packaged_artifacts)

        authoring_evidence: dict[str, Any] = {}
        write_material_library_usd(
            layout.material_usd_path,
            (generated,),
            material_profile=normalized_material_profile,
            authoring_evidence=authoring_evidence,
        )
        material_list_entry = CreatedMaterialListEntry.for_request(
            request,
            creation_manifest=Path(MATERIAL_CREATION_MANIFEST_NAME),
            provenance=result.provenance,
        )
        _write_materials_manifest(
            layout.materials_manifest_path,
            layout.material_usd_path,
            material_list_entry,
        )
        validation = _validate_created_material_package(layout, packaged_artifacts)
        created = CreatedMaterial(
            material_id=request.recipe.material_id,
            material_prim_path=request.recipe.binding,
            material_usd_path=layout.material_usd_path,
            creation_manifest_path=layout.creation_manifest_path,
            texture_artifacts=packaged_artifacts,
            material_list_entry=material_list_entry,
            preview_paths=result.preview_paths,
            validation={
                **validation,
                "cache_hit": False,
                "material_profile": authoring_evidence,
            },
            provenance=result.provenance,
            degradations=result.degradations,
        )
        _write_creation_manifest(
            layout.creation_manifest_path, request, result, created
        )
        return created
    except MaterialCreationError:
        if created_package_dir:
            shutil.rmtree(layout.package_dir, ignore_errors=True)
        raise
    except MaterialAuthoringError as exc:
        if created_package_dir:
            shutil.rmtree(layout.package_dir, ignore_errors=True)
        raise _authoring_creation_error(exc, backend=backend.name) from exc
    except Exception as exc:
        if created_package_dir:
            shutil.rmtree(layout.package_dir, ignore_errors=True)
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_OUTPUT,
            f"Material creation package assembly failed: {exc}",
            backend=backend.name,
            retryable=False,
            diagnostics=(
                MaterialCreationDiagnostic(
                    code="PACKAGE_ASSEMBLY_FAILED",
                    message=str(exc),
                    severity=MaterialDiagnosticSeverity.ERROR,
                    phase="packaging",
                    retryable=False,
                ),
            ),
        ) from exc


def _validate_backend_result(
    result: BackendMaterialResult,
    request: CreateMaterialRequest,
    backend: MaterialCreationBackend,
    *,
    conditioning: PreparedMaterialConditioning | None = None,
) -> None:
    if result.provenance.request_id != request.request_id:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_OUTPUT,
            "Backend result provenance does not match the creation request.",
            backend=backend.name,
            retryable=False,
        )
    if result.provenance.backend != backend.name:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_OUTPUT,
            "Backend result provenance backend does not match the selected backend.",
            backend=backend.name,
            retryable=False,
        )
    if result.provenance.backend_revision != backend.revision:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_OUTPUT,
            "Backend result provenance revision does not match the selected backend.",
            backend=backend.name,
            retryable=False,
        )
    _validate_provenance_cache_key(
        result.provenance,
        request,
        backend,
        conditioning=conditioning,
    )


def _validate_provenance_cache_key(
    provenance: MaterialCreationProvenance,
    request: CreateMaterialRequest,
    backend: MaterialCreationBackend,
    *,
    conditioning: PreparedMaterialConditioning | None,
) -> None:
    expected = MaterialCreationProvenance.for_request(
        request,
        backend=backend.name,
        backend_revision=backend.revision,
        model_revisions=provenance.model_revisions,
        duration_seconds=provenance.duration_seconds,
        conditioning=conditioning,
    )
    if provenance.cache_key != expected.cache_key:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_OUTPUT,
            "Material creation provenance cache key does not match the current "
            "request, backend revision, model revisions, and conditioning.",
            backend=backend.name,
            retryable=False,
        )


def _packaged_artifacts(
    result: BackendMaterialResult,
    layout: MaterialArtifactLayout,
) -> tuple[MaterialChannelArtifact, ...]:
    artifacts = {artifact.channel: artifact for artifact in result.artifacts}
    for channel in (MaterialChannel.ALBEDO, MaterialChannel.ORM):
        artifact = artifacts.get(channel)
        if artifact is None:
            raise MaterialCreationError(
                MaterialCreationErrorCode.PARTIAL_OUTPUT,
                f"Backend output is missing required {channel.value} texture.",
                backend=result.provenance.backend,
                retryable=False,
            )
        _validate_artifact_path(artifact.path, layout.package_dir)

    normal = artifacts.get(MaterialChannel.NORMAL)
    if normal is not None:
        _validate_artifact_path(normal.path, layout.package_dir)
    else:
        if not _has_normal_degradation(result.degradations):
            raise MaterialCreationError(
                MaterialCreationErrorCode.PARTIAL_OUTPUT,
                "Backend output is missing normal texture without degradation.",
                backend=result.provenance.backend,
                retryable=False,
            )
        normal = _write_neutral_normal_fallback(layout)
        artifacts[MaterialChannel.NORMAL] = normal

    ordered = tuple(
        artifacts[channel]
        for channel in (
            MaterialChannel.ALBEDO,
            MaterialChannel.NORMAL,
            MaterialChannel.ORM,
        )
    )
    for artifact in ordered:
        if not artifact.path.is_file():
            raise MaterialCreationError(
                MaterialCreationErrorCode.INVALID_OUTPUT,
                f"Packaged texture does not exist: {artifact.path}",
                backend=result.provenance.backend,
                retryable=False,
            )
    return ordered


def _validate_artifact_path(path: Path, package_dir: Path) -> None:
    try:
        path.resolve().relative_to(package_dir.resolve())
    except ValueError as exc:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_OUTPUT,
            f"Backend artifact path is outside the material package: {path}",
            retryable=False,
        ) from exc


def _has_normal_degradation(degradations: tuple[MaterialDegradation, ...]) -> bool:
    return any(
        degradation.code
        in {
            MaterialDegradationCode.MISSING_NORMAL,
            MaterialDegradationCode.BUMP_TO_NORMAL_FAILED,
        }
        and MaterialChannel.NORMAL in degradation.channels
        for degradation in degradations
    )


def _generated_material(
    request: CreateMaterialRequest,
    artifacts: tuple[MaterialChannelArtifact, ...],
) -> GeneratedMaterial:
    paths = {artifact.channel: artifact.path for artifact in artifacts}
    return GeneratedMaterial(
        recipe=request.recipe,
        textures=TextureMapSet(
            albedo=paths[MaterialChannel.ALBEDO],
            normal=paths[MaterialChannel.NORMAL],
            orm=paths[MaterialChannel.ORM],
        ),
    )


def _write_materials_manifest(
    manifest_path: Path,
    library_path: Path,
    entry: CreatedMaterialListEntry,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "library_path": _relative_path(library_path, manifest_path.parent),
        "entries": [entry.to_dict()],
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def _validate_created_material_package(
    layout: MaterialArtifactLayout,
    artifacts: tuple[MaterialChannelArtifact, ...],
    *,
    material_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    required_paths = [
        layout.material_usd_path,
        layout.materials_manifest_path,
        *material_paths,
        *(artifact.path for artifact in artifacts),
    ]
    for path in required_paths:
        _validate_package_path(path, layout.package_dir)
    missing = [path.as_posix() for path in required_paths if not path.is_file()]
    if missing:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_OUTPUT,
            f"Material package is missing expected files: {missing}",
            retryable=False,
        )
    return {
        "status": "ok",
        "material_library": layout.material_usd_path.name,
        "materials_manifest": layout.materials_manifest_path.name,
        "texture_channels": [artifact.channel.value for artifact in artifacts],
    }


def _write_creation_manifest(
    manifest_path: Path,
    request: CreateMaterialRequest,
    result: BackendMaterialResult,
    created: CreatedMaterial,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    package_dir = manifest_path.parent
    data = {
        "schema_version": MATERIAL_CREATION_SCHEMA_VERSION,
        "request": request.to_dict(),
        "backend_result": _relativize_backend_result(result.to_dict(), package_dir),
        "created_material": _relativize_created_material(
            created.to_dict(), package_dir
        ),
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _load_cached_created_material(
    layout: MaterialArtifactLayout,
    request: CreateMaterialRequest,
    backend: MaterialCreationBackend,
    *,
    conditioning: PreparedMaterialConditioning | None,
    material_profile: str,
) -> CreatedMaterial:
    try:
        data = json.loads(layout.creation_manifest_path.read_text(encoding="utf-8"))
        if data.get("schema_version") != MATERIAL_CREATION_SCHEMA_VERSION:
            raise ValueError("schema_version mismatch")
        created = _created_material_from_dict(
            data["created_material"],
            package_dir=layout.package_dir,
        )
    except Exception as exc:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_OUTPUT,
            f"Existing material creation manifest is not reusable: {exc}",
            backend=backend.name,
            retryable=False,
        ) from exc

    if created.provenance.request_id != request.request_id:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_OUTPUT,
            "Existing material package belongs to a different creation request.",
            backend=backend.name,
            retryable=False,
        )
    if (
        created.provenance.backend != backend.name
        or created.provenance.backend_revision != backend.revision
    ):
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_OUTPUT,
            "Existing material package was produced by a different backend revision.",
            backend=backend.name,
            retryable=False,
        )
    _validate_provenance_cache_key(
        created.provenance,
        request,
        backend,
        conditioning=conditioning,
    )
    _validate_created_material_package(
        layout,
        created.texture_artifacts,
        material_paths=(
            created.material_usd_path,
            created.creation_manifest_path,
            *created.preview_paths,
        ),
    )
    stored_profile = created.validation.get("material_profile")
    if isinstance(stored_profile, dict):
        stored_request = stored_profile.get("requested_profile")
        if stored_request != material_profile:
            raise MaterialCreationError(
                MaterialCreationErrorCode.INVALID_OUTPUT,
                "Existing material package was authored for a different material "
                f"profile: requested={material_profile}, existing={stored_request}.",
                backend=backend.name,
                retryable=False,
            )
    elif material_profile != "auto":
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_OUTPUT,
            "Existing material package has no material-profile evidence and cannot "
            f"satisfy explicit profile {material_profile!r}.",
            backend=backend.name,
            retryable=False,
        )

    generated = _generated_material(request, created.texture_artifacts)
    try:
        authoring_evidence = inspect_material_library_authoring(
            created.material_usd_path,
            (generated,),
            material_profile=material_profile,
        )
    except MaterialAuthoringError as exc:
        raise _authoring_creation_error(exc, backend=backend.name) from exc
    except Exception as exc:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_OUTPUT,
            f"Existing material package authoring inspection failed: {exc}",
            backend=backend.name,
            retryable=False,
            diagnostics=(
                MaterialCreationDiagnostic(
                    code="CACHED_AUTHORING_INSPECTION_FAILED",
                    message=str(exc),
                    severity=MaterialDiagnosticSeverity.ERROR,
                    phase="cache_validation",
                    retryable=False,
                ),
            ),
        ) from exc
    return replace(
        created,
        validation={
            **created.validation,
            "cache_hit": True,
            "material_profile": authoring_evidence,
        },
    )


def _created_material_from_dict(
    data: dict[str, Any],
    *,
    package_dir: Path,
) -> CreatedMaterial:
    provenance = _provenance_from_dict(data["provenance"])
    entry = _material_list_entry_from_dict(data["material_list_entry"])
    degradations = tuple(
        _degradation_from_dict(item) for item in data.get("degradations", ())
    )
    artifacts = tuple(
        _artifact_from_dict(item, package_dir=package_dir)
        for item in data["texture_artifacts"]
    )
    return CreatedMaterial(
        material_id=str(data["material_id"]),
        material_prim_path=str(data["material_prim_path"]),
        material_usd_path=_resolve_package_path(
            str(data["material_usd_path"]),
            package_dir,
        ),
        creation_manifest_path=_resolve_package_path(
            str(data["creation_manifest_path"]),
            package_dir,
        ),
        texture_artifacts=artifacts,
        material_list_entry=entry,
        preview_paths=tuple(
            _resolve_package_path(str(path), package_dir)
            for path in data.get("preview_paths", ())
        ),
        validation=dict(data.get("validation") or {}),
        provenance=provenance,
        degradations=degradations,
    )


def _provenance_from_dict(data: dict[str, Any]) -> MaterialCreationProvenance:
    return MaterialCreationProvenance(
        request_id=str(data["request_id"]),
        cache_key=str(data["cache_key"]),
        backend=str(data["backend"]),
        backend_revision=str(data["backend_revision"]),
        model_revisions=tuple(str(item) for item in data["model_revisions"]),
        recipe_id=str(data["recipe_id"]),
        prompt=str(data["prompt"]),
        seed=int(data["seed"]),
        target_prim_paths=tuple(str(path) for path in data["target_prim_paths"]),
        reference_image_uris=tuple(
            str(uri) for uri in data.get("reference_image_uris", ())
        ),
        duration_seconds=float(data["duration_seconds"]),
        source_usd=str(data["source_usd"]),
        source_usd_sha256=data.get("source_usd_sha256"),
        conditioning_fingerprint=data.get("conditioning_fingerprint"),
    )


def _material_list_entry_from_dict(data: dict[str, Any]) -> CreatedMaterialListEntry:
    return CreatedMaterialListEntry(
        name=str(data["name"]),
        description=str(data["description"]),
        binding=str(data["binding"]),
        generation_id=str(data["generation_id"]),
        creation_request_id=str(data["creation_request_id"]),
        creation_cache_key=str(data["creation_cache_key"]),
        reuse_key=str(data["reuse_key"]),
        target_prim_paths=tuple(str(path) for path in data["target_prim_paths"]),
        creation_manifest=str(data["creation_manifest"]),
        intended_parts=tuple(str(part) for part in data.get("intended_parts", ())),
        source=str(data.get("source", "generated")),
    )


def _artifact_from_dict(
    data: dict[str, Any],
    *,
    package_dir: Path,
) -> MaterialChannelArtifact:
    return MaterialChannelArtifact(
        channel=MaterialChannel(str(data["channel"])),
        path=_resolve_package_path(str(data["path"]), package_dir),
        color_space=MaterialColorSpace(str(data["color_space"])),
        component_provenance=tuple(
            MaterialComponentProvenance(
                component=MaterialChannelComponent(str(component["component"])),
                source=MaterialChannelSource(str(component["source"])),
                source_detail=str(component["source_detail"]),
            )
            for component in data["component_provenance"]
        ),
        packing=ORMPacking(str(data["packing"])) if data.get("packing") else None,
        normal_convention=(
            NormalConvention(str(data["normal_convention"]))
            if data.get("normal_convention")
            else None
        ),
    )


def _degradation_from_dict(data: dict[str, Any]) -> MaterialDegradation:
    return MaterialDegradation(
        code=MaterialDegradationCode(str(data["code"])),
        channels=tuple(MaterialChannel(str(channel)) for channel in data["channels"]),
        message=str(data["message"]),
        fallback=str(data["fallback"]),
    )


def _relativize_created_material(
    data: dict[str, Any],
    package_dir: Path,
) -> dict[str, Any]:
    data = dict(data)
    for key in ("material_usd_path", "creation_manifest_path"):
        data[key] = _relative_or_original(str(data[key]), package_dir)
    data["texture_paths"] = {
        channel: _relative_or_original(str(path), package_dir)
        for channel, path in data["texture_paths"].items()
    }
    data["texture_artifacts"] = [
        _relativize_artifact_dict(item, package_dir)
        for item in data["texture_artifacts"]
    ]
    data["preview_paths"] = [
        _relative_or_original(str(path), package_dir)
        for path in data.get("preview_paths", ())
    ]
    return data


def _relativize_backend_result(
    data: dict[str, Any],
    package_dir: Path,
) -> dict[str, Any]:
    data = dict(data)
    data["artifacts"] = [
        _relativize_artifact_dict(item, package_dir) for item in data["artifacts"]
    ]
    data["preview_paths"] = [
        _relative_or_original(str(path), package_dir)
        for path in data.get("preview_paths", ())
    ]
    return data


def _relativize_artifact_dict(
    data: dict[str, Any],
    package_dir: Path,
) -> dict[str, Any]:
    data = dict(data)
    data["path"] = _relative_or_original(str(data["path"]), package_dir)
    return data


def _relative_or_original(path: str, package_dir: Path) -> str:
    value = Path(path)
    if not value.is_absolute():
        return value.as_posix()
    try:
        return value.resolve().relative_to(package_dir.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _resolve_package_path(path: str, package_dir: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return package_dir / value


def _validate_package_path(path: Path, package_dir: Path) -> None:
    try:
        path.resolve().relative_to(package_dir.resolve())
    except ValueError as exc:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_OUTPUT,
            f"Material package path is outside the package directory: {path}",
            retryable=False,
        ) from exc


def _relative_path(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve()).replace("\\", "/")


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)
    )


def _write_rgb_png(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    scanline = b"\x00" + bytes(rgb)
    path.write_bytes(
        signature
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanline, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _write_neutral_normal_fallback(
    layout: MaterialArtifactLayout,
) -> MaterialChannelArtifact:
    path = layout.texture_path(MaterialChannel.NORMAL)
    _write_rgb_png(path, (128, 128, 255))
    return MaterialChannelArtifact(
        channel=MaterialChannel.NORMAL,
        path=path,
        color_space=MaterialColorSpace.RAW,
        component_provenance=(
            MaterialComponentProvenance(
                component=MaterialChannelComponent.TANGENT_NORMAL,
                source=MaterialChannelSource.NEUTRAL_FALLBACK,
                source_detail="neutral normal fallback written during packaging",
            ),
        ),
        normal_convention=NormalConvention.TANGENT_OPENGL,
    )
