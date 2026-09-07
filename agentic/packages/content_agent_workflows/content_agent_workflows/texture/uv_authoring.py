# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-free, bounded UV inspection and authoring for Texture leaves.

The operation is deliberately independent from the Texture Agent service and
from the compatibility Texture workflow.  An outer coordinator supplies one
typed invocation.  This module inspects or authors only ``primvars:st`` on the
explicit mesh scope, saves the stage, reopens the saved bytes, and records the
exact UV and effective-material identities observed from that reopened stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from world_understanding.utils.usd.package import validate_usdz_package_layout

from content_agent_workflows.asset_composition import (
    ArtifactBinding,
    bind_usd_dependency_closure,
    canonical_asset_digest,
    verify_usd_dependency_closure,
)
from content_agent_workflows.common.artifacts import (
    read_contained_artifact,
)
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding

TEXTURE_UV_LEAF_ID: Literal["texture.uv-prepare.v1"] = "texture.uv-prepare.v1"
TEXTURE_UV_INVOCATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-uv-leaf-invocation.v1"
] = "content-agent-workflows.texture-uv-leaf-invocation.v1"
TEXTURE_UV_READBACK_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-uv-saved-stage-readback.v1"
] = "content-agent-workflows.texture-uv-saved-stage-readback.v1"
TEXTURE_UV_EVIDENCE_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-uv-evidence.v1"
] = "content-agent-workflows.texture-uv-evidence.v1"
TEXTURE_UV_RESULT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-uv-leaf-result.v1"
] = "content-agent-workflows.texture-uv-leaf-result.v1"

TextureUvPolicy = Literal["inspect", "generate_missing"]
TextureUvTargetKind = Literal["mesh", "subset", "descendant_mesh"]
TextureUvMaterialPurpose = Literal["allPurpose", "preview", "full"]
_SUPPORTED_UV_ARRAY_TYPES = {
    "double2[]",
    "float2[]",
    "half2[]",
    "texCoord2d[]",
    "texCoord2f[]",
    "texCoord2h[]",
}
_MAX_INVOCATION_BYTES = 1024 * 1024
_FileIdentity = tuple[int, int]
_DirectoryIdentity = tuple[int, int]
_READ_CHUNK_SIZE = 1024 * 1024
_SUPPORTED_TEXTURE_UV_SOURCE_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
_COMPOSITION_ROOT_METADATA_KEYS = frozenset(
    {"relocates", "subLayerOffsets", "subLayers"}
)


def _stage_metadata_state(layer: Sdf.Layer) -> dict[str, Any]:
    """Return authored stage metadata, excluding composition arcs."""

    return {
        key: layer.pseudoRoot.GetInfo(key)
        for key in layer.pseudoRoot.ListInfoKeys()
        if key not in _COMPOSITION_ROOT_METADATA_KEYS
    }


def _copy_stage_metadata(source: Sdf.Layer, destination: Sdf.Layer) -> None:
    """Copy every authored non-composition root metadata field exactly."""

    for key, value in _stage_metadata_state(source).items():
        destination.pseudoRoot.SetInfo(key, value)


def _require_stage_metadata_preserved(
    source: Sdf.Layer,
    destination: Sdf.Layer,
) -> None:
    """Fail closed when a wrapper or saved overlay changes stage metadata."""

    if _stage_metadata_state(source) != _stage_metadata_state(destination):
        raise ValueError("Texture UV saved stage changed source stage metadata")


def _validate_supported_source_layer(path: str | Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix not in _SUPPORTED_TEXTURE_UV_SOURCE_SUFFIXES:
        raise ValueError(
            "Texture UV source must be a supported USD layer "
            "(.usd, .usda, .usdc, or .usdz)"
        )


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TextureUvLeafInvocation(_FrozenModel):
    """Exact outer-supplied request for one provider-free UV leaf."""

    schema_version: Literal["content-agent-workflows.texture-uv-leaf-invocation.v1"] = (
        TEXTURE_UV_INVOCATION_SCHEMA_VERSION
    )
    leaf_id: Literal["texture.uv-prepare.v1"] = TEXTURE_UV_LEAF_ID
    source: ExecutionArtifactBinding
    source_dependencies: tuple[ArtifactBinding, ...]
    output_dir: str = Field(min_length=1)
    target_prim_paths: tuple[str, ...] = Field(min_length=1, max_length=1024)
    policy: TextureUvPolicy = "inspect"

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        _validate_supported_source_layer(self.source.path)
        if (
            Path(self.source.path).suffix.lower() == ".usdz"
            and self.source_dependencies
        ):
            raise ValueError(
                "Texture UV package source must be self-contained; external "
                "dependency bindings are not supported"
            )
        normalized = tuple(path.strip() for path in self.target_prim_paths)
        valid_paths = []
        for path in normalized:
            valid = Sdf.Path.IsValidPathString(path)
            if not bool(valid):
                valid_paths.append(False)
                continue
            sdf_path = Sdf.Path(path)
            valid_paths.append(
                sdf_path.IsAbsolutePath()
                and sdf_path.IsPrimPath()
                and sdf_path != Sdf.Path.absoluteRootPath
                and str(sdf_path) == path
            )
        if normalized != self.target_prim_paths or not all(valid_paths):
            raise ValueError(
                "Texture UV targets must be normalized absolute prim paths"
            )
        if len(normalized) != len(set(normalized)):
            raise ValueError("Texture UV targets must be unique")
        return self


class TextureUvMeshReadback(_FrozenModel):
    """Exact UV and material identity for one reopened mesh."""

    prim_path: str = Field(min_length=1)
    primvar_source_prim_path: str | None = None
    status: Literal["ready", "missing", "repair_required", "unsupported"]
    interpolation: str
    indexed: bool
    value_count: int = Field(ge=0)
    index_count: int = Field(ge=0)
    expected_element_count: int = Field(ge=0)
    topology_time_sample_count: int = Field(ge=0)
    topology_time_samples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    value_time_sample_count: int = Field(ge=0)
    index_time_sample_count: int = Field(ge=0)
    values_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    indices_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    time_samples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uv_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    effective_material_path: str | None = None
    binding_relationship_path: str | None = None
    material_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reasons: tuple[str, ...] = ()


class TextureUvScopeReadback(_FrozenModel):
    """Exact requested Texture scope and effective material after reopen."""

    requested_prim_path: str = Field(min_length=1)
    scope_prim_path: str = Field(min_length=1)
    uv_mesh_path: str = Field(min_length=1)
    target_kind: TextureUvTargetKind
    material_purpose: TextureUvMaterialPurpose = "allPurpose"
    effective_material_path: str | None = None
    binding_relationship_path: str | None = None
    material_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TextureUvSavedStageReadback(_FrozenModel):
    """Facts re-derived only after reopening the final saved stage bytes."""

    schema_version: Literal[
        "content-agent-workflows.texture-uv-saved-stage-readback.v1"
    ] = TEXTURE_UV_READBACK_SCHEMA_VERSION
    source: ExecutionArtifactBinding
    source_dependencies: tuple[ArtifactBinding, ...]
    saved_stage: ExecutionArtifactBinding
    saved_stage_dependencies: tuple[ArtifactBinding, ...]
    policy: TextureUvPolicy
    authoring_method: Literal["bounded_box_fallback"] | None
    requested_prim_paths: tuple[str, ...]
    resolved_mesh_paths: tuple[str, ...] = Field(min_length=1)
    authored_mesh_paths: tuple[str, ...]
    preserved_mesh_paths: tuple[str, ...]
    mesh_readbacks: tuple[TextureUvMeshReadback, ...] = Field(min_length=1)
    scope_readbacks: tuple[TextureUvScopeReadback, ...] = Field(min_length=1)
    all_uv_ready: bool
    material_identities_unchanged: bool
    preserved_uv_identities_unchanged: bool
    external_dependencies_preserved: bool
    reopened_saved_stage: Literal[True] = True
    readback_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_authoring_method(self) -> Self:
        expected = "bounded_box_fallback" if self.authored_mesh_paths else None
        if self.authoring_method != expected:
            raise ValueError(
                "Texture UV saved-stage readback authoring method does not "
                "match its authored mesh paths"
            )
        return self


class TextureUvEvidence(_FrozenModel):
    """Provider-free facts retained beside one exact saved-stage readback."""

    schema_version: Literal["content-agent-workflows.texture-uv-evidence.v1"] = (
        TEXTURE_UV_EVIDENCE_SCHEMA_VERSION
    )
    source: ExecutionArtifactBinding
    saved_stage: ExecutionArtifactBinding
    policy: TextureUvPolicy
    authoring_method: Literal["bounded_box_fallback"] | None
    requested_prim_paths: tuple[str, ...] = Field(min_length=1)
    resolved_mesh_paths: tuple[str, ...] = Field(min_length=1)
    authored_mesh_paths: tuple[str, ...]
    before_uv_identity_sha256: dict[str, str]
    before_scope_identity_sha256: dict[str, str]
    saved_stage_readback_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_dependencies_preserved: bool
    provider_invoked: Literal[False] = False
    texture_service_constructed: Literal[False] = False
    vlm_assessor_constructed: Literal[False] = False
    image_generator_constructed: Literal[False] = False
    fixed_pipeline_invoked: Literal[False] = False
    nested_coordinator_invoked: Literal[False] = False

    @model_validator(mode="after")
    def validate_authoring_method(self) -> Self:
        expected = "bounded_box_fallback" if self.authored_mesh_paths else None
        if self.authoring_method != expected:
            raise ValueError(
                "Texture UV evidence authoring method does not match its "
                "authored mesh paths"
            )
        return self


class TextureUvLeafResult(_FrozenModel):
    """Native result from one standalone provider-free Texture UV operation."""

    schema_version: Literal["content-agent-workflows.texture-uv-leaf-result.v1"] = (
        TEXTURE_UV_RESULT_SCHEMA_VERSION
    )
    leaf_id: Literal["texture.uv-prepare.v1"] = TEXTURE_UV_LEAF_ID
    invocation: ExecutionArtifactBinding
    native_disposition: Literal["passed", "failed", "not_evaluated"]
    detail: str = Field(min_length=1)
    error: str | None = None
    source: ExecutionArtifactBinding
    output: ExecutionArtifactBinding
    output_dependencies: tuple[ArtifactBinding, ...]
    evidence: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    saved_stage_readbacks: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    provider_invoked: Literal[False] = False
    texture_service_constructed: Literal[False] = False
    vlm_assessor_constructed: Literal[False] = False
    image_generator_constructed: Literal[False] = False
    fixed_pipeline_invoked: Literal[False] = False
    nested_coordinator_invoked: Literal[False] = False

    @model_validator(mode="after")
    def validate_native_terminal_disposition(self) -> Self:
        expected_error = self.detail if self.native_disposition == "failed" else None
        if self.error != expected_error:
            raise ValueError(
                "Texture UV failed disposition must preserve its exact detail as error"
            )
        return self


def _stable_json_bytes(payload: BaseModel | Mapping[str, Any]) -> bytes:
    document = (
        payload.model_dump(mode="json")
        if isinstance(payload, BaseModel)
        else dict(payload)
    )
    return (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _read_regular_fd(
    descriptor: int,
    display_path: Path,
    *,
    max_bytes: int | None = None,
) -> tuple[bytes, _FileIdentity]:
    """Read one stable regular-file generation from an already-held fd."""

    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Texture UV artifact is not a regular file: {display_path}")
    if before.st_nlink != 1:
        raise ValueError(
            f"Texture UV attempt artifact must have one link: {display_path}"
        )
    if max_bytes is not None and before.st_size > max_bytes:
        raise ValueError(
            f"Texture UV artifact exceeds {max_bytes} bytes: {display_path}"
        )
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_SIZE)
        if not chunk:
            break
        size += len(chunk)
        if max_bytes is not None and size > max_bytes:
            raise ValueError(
                f"Texture UV artifact exceeds {max_bytes} bytes: {display_path}"
            )
        chunks.append(chunk)
    after = os.fstat(descriptor)
    stable_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    )
    stable_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    )
    if stable_before != stable_after or size != after.st_size:
        raise ValueError(f"Texture UV artifact changed while read: {display_path}")
    return b"".join(chunks), (after.st_dev, after.st_ino)


@dataclass
class _HeldDirectory:
    """A no-follow attempt-root descriptor held for one whole operation."""

    path: Path
    descriptor: int
    identity: _DirectoryIdentity
    cleanup_callbacks: list[Callable[[], None]] = field(default_factory=list)

    @classmethod
    def open(cls, path: str | Path) -> _HeldDirectory:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute() or candidate.is_symlink():
            raise ValueError("Texture UV attempt root must be canonical and absolute")
        canonical = candidate.resolve(strict=True)
        if candidate != canonical:
            raise ValueError("Texture UV attempt root must be canonical and absolute")
        descriptor = os.open(
            canonical,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    f"Texture UV attempt root is not a directory: {canonical}"
                )
            held = cls(
                path=canonical,
                descriptor=descriptor,
                identity=(metadata.st_dev, metadata.st_ino),
            )
            descriptor = -1
            held.verify_path()
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            else:
                held.close()
            raise
        return held

    def __enter__(self) -> _HeldDirectory:
        return self

    def __exit__(
        self,
        _exc_type: object,
        exc_value: BaseException | None,
        _traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc_value is None:
                raise
            # Preserve the primary operation failure while retaining cleanup
            # evidence for callers that inspect the raised exception.
            if hasattr(exc_value, "add_note"):
                exc_value.add_note(
                    f"Texture UV snapshot cleanup also failed: {cleanup_error}"
                )
            try:
                setattr(exc_value, "texture_uv_cleanup_error", cleanup_error)
            except (AttributeError, TypeError):
                pass

    def close(self) -> None:
        cleanup_error: BaseException | None = None
        while self.cleanup_callbacks:
            callback = self.cleanup_callbacks.pop()
            try:
                callback()
            except BaseException as exc:  # cleanup must still release all handles
                cleanup_error = cleanup_error or exc
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1
        if cleanup_error is not None:
            raise cleanup_error

    def retain_cleanup(self, callback: Callable[[], None]) -> None:
        """Retain one private resource until this held operation exits."""

        if self.descriptor < 0:
            raise ValueError("Texture UV attempt root is already closed")
        self.cleanup_callbacks.append(callback)

    @property
    def descriptor_path(self) -> Path:
        """Linux path that continues to name this exact held directory."""

        if self.descriptor < 0:
            raise ValueError("Texture UV directory descriptor is closed")
        return Path(f"/proc/self/fd/{self.descriptor}")

    def verify_path(self) -> None:
        try:
            metadata = self.path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise ValueError("Texture UV attempt root was renamed or removed") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (
                metadata.st_dev,
                metadata.st_ino,
            )
            != self.identity
        ):
            raise ValueError("Texture UV attempt root identity changed")

    def claim_directory(self, name: str, *, mode: int = 0o700) -> _HeldDirectory:
        """Atomically create and retain one direct child through this descriptor."""

        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Texture UV child directory must be one direct name")
        try:
            os.mkdir(name, mode=mode, dir_fd=self.descriptor)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Texture UV child directory already exists: {self.path / name}"
            ) from exc
        descriptor = -1
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self.descriptor,
            )
            os.fchmod(descriptor, mode)
            os.fsync(self.descriptor)
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    f"Texture UV child is not a directory: {self.path / name}"
                )
            held = _HeldDirectory(
                path=self.path / name,
                descriptor=descriptor,
                identity=(metadata.st_dev, metadata.st_ino),
            )
            descriptor = -1
            try:
                held.verify_path()
            except BaseException:
                held.close()
                raise
            return held
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            raise

    def direct_name(self, path: str | Path) -> str:
        candidate = Path(path).expanduser()
        absolute = candidate if candidate.is_absolute() else self.path / candidate
        if absolute.parent != self.path or absolute.name in {"", ".", ".."}:
            raise ValueError(
                f"Texture UV artifact is outside the workflow run: {absolute}"
            )
        return absolute.name

    def file_identity(self, name: str) -> _FileIdentity:
        metadata = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"Texture UV artifact is not a regular file: {self.path / name}"
            )
        return metadata.st_dev, metadata.st_ino

    def exists(self, name: str) -> bool:
        try:
            os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    def open_read(self, name: str, *, identity: _FileIdentity | None = None) -> int:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=self.descriptor,
            )
        except OSError as exc:
            raise ValueError(
                "Texture UV artifact path must not contain symlinks: "
                f"{self.path / name}"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            observed = (metadata.st_dev, metadata.st_ino)
            if not stat.S_ISREG(metadata.st_mode) or (
                identity is not None and observed != identity
            ):
                raise ValueError(
                    f"Texture UV artifact identity changed: {self.path / name}"
                )
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def read(
        self,
        path: str | Path,
        *,
        max_bytes: int | None = None,
    ) -> tuple[ExecutionArtifactBinding, bytes, _FileIdentity]:
        name = self.direct_name(path)
        descriptor = self.open_read(name)
        try:
            data, identity = _read_regular_fd(
                descriptor,
                self.path / name,
                max_bytes=max_bytes,
            )
        finally:
            os.close(descriptor)
        self.verify_path()
        if self.file_identity(name) != identity:
            raise ValueError(
                f"Texture UV artifact path changed after read: {self.path / name}"
            )
        return (
            ExecutionArtifactBinding(
                path=str(self.path / name),
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
            ),
            data,
            identity,
        )

    def write_bytes(self, name: str, document: bytes) -> _FileIdentity:
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=self.descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            identity = (metadata.st_dev, metadata.st_ino)
            view = memoryview(document)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("Texture UV artifact write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            os.fsync(self.descriptor)
        finally:
            os.close(descriptor)
        return identity

    def write_json(
        self,
        name: str,
        payload: BaseModel | Mapping[str, Any],
    ) -> _FileIdentity:
        return self.write_bytes(name, _stable_json_bytes(payload))

    def binding(
        self,
        name: str,
        *,
        identity: _FileIdentity | None = None,
    ) -> ExecutionArtifactBinding:
        descriptor = self.open_read(name, identity=identity)
        try:
            data, observed_identity = _read_regular_fd(
                descriptor,
                self.path / name,
            )
        finally:
            os.close(descriptor)
        if identity is not None and observed_identity != identity:
            raise ValueError(
                f"Texture UV artifact identity changed: {self.path / name}"
            )
        self.verify_path()
        if self.file_identity(name) != observed_identity:
            raise ValueError(
                f"Texture UV artifact path changed after bind: {self.path / name}"
            )
        return ExecutionArtifactBinding(
            path=str(self.path / name),
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    def verify_binding(
        self,
        binding: ExecutionArtifactBinding,
        *,
        label: str,
    ) -> None:
        observed = self.binding(self.direct_name(binding.path))
        if observed != binding:
            raise ValueError(f"{label} identity changed: {binding.path}")


def _read_bound_bytes(
    binding: ArtifactBinding | ExecutionArtifactBinding,
    *,
    label: str,
) -> bytes:
    """Read exactly one bound external generation through a held descriptor."""

    path = Path(binding.path)
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        data, _identity = _read_regular_fd(descriptor, path)
    finally:
        os.close(descriptor)
    observed = ExecutionArtifactBinding(
        path=str(path),
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )
    if (
        observed.path != binding.path
        or observed.sha256 != binding.sha256
        or observed.size_bytes != binding.size_bytes
    ):
        raise ValueError(f"{label} identity changed: {binding.path}")
    return data


def _verify_bound_generation(
    binding: ArtifactBinding | ExecutionArtifactBinding,
    *,
    label: str,
) -> None:
    """Stream and bind one closure member that USD will not parse as a layer."""

    path = Path(binding.path)
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} is not a single-link regular file: {path}")
        digest = hashlib.sha256()
        size_bytes = 0
        while chunk := os.read(descriptor, _READ_CHUNK_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    )
    stable_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    )
    if (
        stable_before != stable_after
        or size_bytes != binding.size_bytes
        or digest.hexdigest() != binding.sha256
        or str(path) != binding.path
    ):
        raise ValueError(f"{label} identity changed: {binding.path}")


def _write_private_snapshot(
    path: Path,
    document: bytes,
) -> int:
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(document)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Texture UV snapshot write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _snapshot_file_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _snapshot_directory_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_binary_snapshot_layer(
    *,
    snapshot_root: Path,
    snapshot_path: Path,
    original_path: Path,
    document: bytes,
) -> Sdf.Layer:
    """Open binary USD only while its held bytes and namespace stay unchanged."""

    directory_descriptor = os.open(
        snapshot_root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        descriptor = _write_private_snapshot(snapshot_path, document)
        os.fsync(directory_descriptor)
        file_before = _snapshot_file_signature(os.fstat(descriptor))
        directory_before = _snapshot_directory_signature(os.fstat(directory_descriptor))
        layer = Sdf.Layer.OpenAsAnonymous(str(snapshot_path))

        def verify_generation() -> None:
            file_after = _snapshot_file_signature(os.fstat(descriptor))
            directory_after = _snapshot_directory_signature(
                os.fstat(directory_descriptor)
            )
            try:
                path_after = _snapshot_file_signature(
                    snapshot_path.stat(follow_symlinks=False)
                )
            except FileNotFoundError as exc:
                raise ValueError(
                    "Texture UV binary source snapshot path changed during USD open: "
                    f"{original_path}"
                ) from exc
            os.lseek(descriptor, 0, os.SEEK_SET)
            rebound, rebound_identity = _read_regular_fd(descriptor, snapshot_path)
            if (
                layer is None
                or file_before != file_after
                or file_before != path_after
                or directory_before != directory_after
                or rebound_identity != (file_before[0], file_before[1])
                or rebound != document
            ):
                raise ValueError(
                    "Texture UV binary source snapshot namespace or bytes changed "
                    f"during USD open: {original_path}"
                )

        verify_generation()
        try:
            serialized_layer = str(layer.ExportToString())
        except Exception as exc:
            verify_generation()
            raise ValueError(
                "Texture UV binary source snapshot could not be detached: "
                f"{original_path}"
            ) from exc
        verify_generation()
        verification_path = snapshot_path.with_name(
            f"{snapshot_path.stem}-verification{snapshot_path.suffix}"
        )
        verification_descriptor = _write_private_snapshot(verification_path, document)
        try:
            verification_layer = Sdf.Layer.OpenAsAnonymous(str(verification_path))
        finally:
            os.close(verification_descriptor)
            verification_path.unlink(missing_ok=True)
        if verification_layer is None or serialized_layer != str(
            verification_layer.ExportToString()
        ):
            raise ValueError(
                "Texture UV binary source snapshot namespace or bytes changed "
                f"during USD open: {original_path}"
            )
        detached = Sdf.Layer.CreateAnonymous(f"source-{original_path.name}.usda")
        if not detached.ImportFromString(serialized_layer):
            raise ValueError(
                "Texture UV binary source snapshot could not be detached: "
                f"{original_path}"
            )
        return detached
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
        finally:
            os.close(directory_descriptor)


def _open_snapshot_layer(
    *,
    snapshot_root: Path,
    snapshot_path: Path,
    original_path: Path,
    document: bytes,
) -> Sdf.Layer:
    if document.startswith(b"PXR-USDC\x00"):
        return _open_binary_snapshot_layer(
            snapshot_root=snapshot_root,
            snapshot_path=snapshot_path,
            original_path=original_path,
            document=document,
        )
    try:
        text = document.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Texture UV source layer is neither USDA nor USDC: {original_path}"
        ) from exc
    # ``.usd`` is an ambiguous container suffix.  Importing text through an
    # anonymous layer with that suffix lets the registry select the binary
    # crate format on some USD builds.  The detached snapshot is known text,
    # so give the anonymous layer an explicit USDA tag.
    layer = Sdf.Layer.CreateAnonymous(f"source-{original_path.name}.usda")
    if not layer.ImportFromString(text):
        raise ValueError(
            f"Texture UV exact source layer could not be imported: {original_path}"
        )
    return layer


def _resolve_composition_dependency(
    owner_path: Path,
    authored_path: str,
) -> Path:
    if not authored_path or "[" in authored_path or "]" in authored_path:
        raise ValueError(
            "Texture UV exact source snapshot does not support package-relative "
            f"composition paths: {authored_path!r}"
        )
    candidate = Path(authored_path).expanduser()
    if not candidate.is_absolute():
        candidate = owner_path.parent / candidate
    return candidate.resolve()


def _rebind_all_composition_dependency_occurrences(
    layer: Sdf.Layer,
    *,
    owner_path: Path,
    authored_path: str,
    target_identifier: str,
) -> None:
    document = str(layer.ExportToString())
    # Every authored occurrence occupies serialized layer space. This bound keeps a
    # faulty USD implementation from returning success forever while retaining it.
    maximum_updates = len(document) + 1
    for _ in range(maximum_updates):
        dependencies = tuple(layer.GetCompositionAssetDependencies())
        if authored_path not in dependencies:
            break
        if not layer.UpdateCompositionAssetDependency(
            authored_path,
            target_identifier,
        ):
            raise ValueError(
                "Texture UV exact source snapshot could not rebind composition "
                f"dependency {authored_path!r} from {owner_path}"
            )
        updated_document = str(layer.ExportToString())
        if updated_document == document:
            raise ValueError(
                "Texture UV exact source snapshot made no progress rebinding "
                f"composition dependency {authored_path!r} from {owner_path}"
            )
        document = updated_document

    if authored_path in tuple(layer.GetCompositionAssetDependencies()):
        raise ValueError(
            "Texture UV exact source snapshot retained residual composition "
            f"dependency {authored_path!r} from {owner_path}"
        )


@dataclass
class _HeldPackageSnapshot:
    """Owner-private exact package bytes retained for the snapshot lifetime."""

    temporary_directory: tempfile.TemporaryDirectory
    root: Path
    package_path: Path
    directory_descriptor: int
    package_descriptor: int
    directory_signature: tuple[int, ...]
    package_signature: tuple[int, ...]
    source_sha256: str
    source_size_bytes: int
    closed: bool = False

    @classmethod
    def create(
        cls,
        source: ExecutionArtifactBinding,
        document: bytes,
    ) -> _HeldPackageSnapshot:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="texture-uv-package-source-"
        )
        root = Path(temporary_directory.name)
        directory_descriptor = -1
        package_descriptor = -1
        try:
            if os.name != "nt":
                root.chmod(0o700)
            root_metadata = root.lstat()
            if (
                not stat.S_ISDIR(root_metadata.st_mode)
                or root.is_symlink()
                or (
                    os.name != "nt"
                    and (
                        stat.S_IMODE(root_metadata.st_mode) != 0o700
                        or root_metadata.st_uid != os.geteuid()
                    )
                )
            ):
                raise ValueError(
                    "Texture UV package snapshot root is not owner-controlled"
                )
            directory_descriptor = os.open(
                root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            package_path = root / "source.usdz"
            package_descriptor = _write_private_snapshot(package_path, document)
            os.fsync(directory_descriptor)
            held = cls(
                temporary_directory=temporary_directory,
                root=root,
                package_path=package_path,
                directory_descriptor=directory_descriptor,
                package_descriptor=package_descriptor,
                directory_signature=_snapshot_directory_signature(
                    os.fstat(directory_descriptor)
                ),
                package_signature=_snapshot_file_signature(
                    os.fstat(package_descriptor)
                ),
                source_sha256=source.sha256,
                source_size_bytes=source.size_bytes,
            )
            held.verify()
            validate_usdz_package_layout(package_path)
            held.verify()
            return held
        except BaseException:
            if package_descriptor >= 0:
                os.close(package_descriptor)
            if directory_descriptor >= 0:
                os.close(directory_descriptor)
            temporary_directory.cleanup()
            raise

    def verify(self) -> None:
        """Reject namespace, inode, mode, size, or byte drift in the held copy."""

        if self.closed:
            raise ValueError("Texture UV package snapshot is already closed")
        directory_after = _snapshot_directory_signature(
            os.fstat(self.directory_descriptor)
        )
        package_after = _snapshot_file_signature(os.fstat(self.package_descriptor))
        try:
            path_after = _snapshot_file_signature(
                self.package_path.stat(follow_symlinks=False)
            )
        except FileNotFoundError as exc:
            raise ValueError("Texture UV package snapshot path disappeared") from exc
        if (
            directory_after != self.directory_signature
            or package_after != self.package_signature
            or path_after != self.package_signature
            or not stat.S_ISREG(package_after[2])
            or stat.S_IMODE(package_after[2]) != 0o600
            or package_after[6] != 1
        ):
            raise ValueError("Texture UV package snapshot namespace changed")
        os.lseek(self.package_descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        size_bytes = 0
        while chunk := os.read(self.package_descriptor, _READ_CHUNK_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)
        if (
            size_bytes != self.source_size_bytes
            or digest.hexdigest() != self.source_sha256
        ):
            raise ValueError("Texture UV package snapshot bytes changed")

    def close(self) -> None:
        """Release held descriptors and erase the private package namespace."""

        if self.closed:
            return
        self.closed = True
        try:
            os.close(self.package_descriptor)
        finally:
            try:
                os.close(self.directory_descriptor)
            finally:
                self.temporary_directory.cleanup()


@dataclass(frozen=True)
class _FrozenUsdSnapshot:
    """USD graph loaded only from exact bound bytes or one held exact package."""

    source_path: Path
    root_layer: Sdf.Layer
    layers: tuple[Sdf.Layer, ...]
    held_package: _HeldPackageSnapshot | None = None

    def __enter__(self) -> _FrozenUsdSnapshot:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self.held_package is not None:
            self.held_package.close()

    @classmethod
    def open(
        cls,
        source: ExecutionArtifactBinding,
        dependencies: tuple[ArtifactBinding, ...],
    ) -> _FrozenUsdSnapshot:
        source_path = Path(source.path)
        if source_path.suffix.lower() == ".usdz":
            if dependencies:
                raise ValueError(
                    "Texture UV package source must be self-contained; external "
                    "dependency bindings are not supported"
                )
            _verify_bound_generation(source, label="Texture UV package source")
            verify_usd_dependency_closure(source.path, dependencies)
            document = _read_bound_bytes(source, label="Texture UV package source")
            held_package = _HeldPackageSnapshot.create(source, document)
            try:
                _verify_bound_generation(source, label="Texture UV package source")
                verify_usd_dependency_closure(source.path, dependencies)
                held_package.verify()
                package_layer = Sdf.Layer.FindOrOpen(str(held_package.package_path))
                held_package.verify()
                if package_layer is None:
                    raise ValueError(
                        f"Texture UV package source cannot be opened: {source_path}"
                    )
                root_layer = Sdf.Layer.CreateAnonymous(
                    f"source-{source_path.name}.usda"
                )
                root_layer.subLayerPaths = [package_layer.identifier]
                _copy_stage_metadata(package_layer, root_layer)
                _require_stage_metadata_preserved(package_layer, root_layer)
                return cls(
                    source_path=source_path,
                    root_layer=root_layer,
                    layers=(root_layer, package_layer),
                    held_package=held_package,
                )
            except BaseException:
                held_package.close()
                raise

        bindings: tuple[ArtifactBinding | ExecutionArtifactBinding, ...] = (
            source,
            *dependencies,
        )
        if len({binding.path for binding in bindings}) != len(bindings):
            raise ValueError("Texture UV source snapshot paths must be unique")
        layer_bindings = tuple(
            binding
            for binding in bindings
            if Path(binding.path).suffix.lower() in {".usd", ".usda", ".usdc"}
        )
        if not layer_bindings or layer_bindings[0] != source:
            raise ValueError(
                "Texture UV exact source snapshot requires a USD layer source"
            )
        for index, binding in enumerate(bindings):
            if binding not in layer_bindings:
                _verify_bound_generation(
                    binding,
                    label=f"Texture UV source closure artifact {index + 1}",
                )

        opened: dict[Path, Sdf.Layer] = {}
        with tempfile.TemporaryDirectory(prefix="texture-uv-source-") as directory:
            snapshot_root = Path(directory)
            for index, binding in enumerate(layer_bindings):
                original_path = Path(binding.path)
                snapshot_path = snapshot_root / (
                    f"{index:04d}{original_path.suffix.lower()}"
                )
                layer = _open_snapshot_layer(
                    snapshot_root=snapshot_root,
                    snapshot_path=snapshot_path,
                    original_path=original_path,
                    document=_read_bound_bytes(
                        binding,
                        label=f"Texture UV source layer {index + 1}",
                    ),
                )
                opened[original_path] = layer

        for original_path, layer in opened.items():
            for authored_path in tuple(layer.GetCompositionAssetDependencies()):
                # USD reports an empty asset path for internal references and
                # payloads. Those arcs stay within this already-detached layer
                # and therefore have no external bytes to snapshot or rebind.
                if authored_path == "":
                    continue
                resolved = _resolve_composition_dependency(
                    original_path,
                    authored_path,
                )
                target = opened.get(resolved)
                if target is None:
                    raise ValueError(
                        "Texture UV exact source snapshot is missing composition "
                        f"dependency {authored_path!r} from {original_path}"
                    )
                _rebind_all_composition_dependency_occurrences(
                    layer,
                    owner_path=original_path,
                    authored_path=authored_path,
                    target_identifier=target.identifier,
                )

        root_layer = opened[source_path]
        return cls(
            source_path=source_path,
            root_layer=root_layer,
            layers=tuple(opened.values()),
        )

    def open_stage(self) -> Usd.Stage:
        if self.held_package is not None:
            self.held_package.verify()
        stage = Usd.Stage.Open(self.root_layer, load=Usd.Stage.LoadAll)
        if self.held_package is not None:
            self.held_package.verify()
        if stage is None:
            raise ValueError(
                f"Texture UV source stage cannot be opened: {self.source_path}"
            )
        return stage


def _binding(path: str | Path) -> ExecutionArtifactBinding:
    candidate = Path(path).expanduser()
    root = candidate.parent.resolve(strict=True)
    observed = read_contained_artifact(root, root / candidate.name)
    return ExecutionArtifactBinding(
        path=str(observed.path),
        sha256=observed.sha256,
        size_bytes=observed.size_bytes,
    )


def _verify_binding(binding: ExecutionArtifactBinding, *, label: str) -> Path:
    observed = _binding(binding.path)
    if observed != binding:
        raise ValueError(f"{label} identity changed: {binding.path}")
    return Path(observed.path)


def build_texture_uv_leaf_invocation(
    source_path: str | Path,
    *,
    output_dir: str | Path,
    target_prim_paths: tuple[str, ...],
    policy: TextureUvPolicy = "inspect",
) -> TextureUvLeafInvocation:
    """Bind a source and its complete closure without executing the leaf."""

    _validate_supported_source_layer(source_path)
    source = _binding(source_path)
    return TextureUvLeafInvocation(
        source=source,
        source_dependencies=tuple(bind_usd_dependency_closure(source.path)),
        output_dir=str(Path(output_dir).expanduser().resolve()),
        target_prim_paths=target_prim_paths,
        policy=policy,
    )


def _require_exact_package_predecessor_dependency(
    source: ExecutionArtifactBinding,
    output_dependencies: tuple[ArtifactBinding, ...],
) -> None:
    """Require an authored overlay to retain its exact sealed USDZ predecessor."""

    if Path(source.path).suffix.lower() != ".usdz":
        return
    matching = tuple(
        binding for binding in output_dependencies if binding.path == source.path
    )
    if (
        len(matching) != 1
        or matching[0].sha256 != source.sha256
        or matching[0].size_bytes != source.size_bytes
    ):
        raise ValueError(
            "Texture UV authored overlay lost its exact USDZ predecessor binding"
        )


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Sdf.AssetPath):
        # Bind the authored operand because package-local resolved paths include a
        # transient extraction root that changes on every USDZ reopen. Exact source
        # and dependency bindings separately seal the referenced bytes.
        return {"asset_path": value.path or value.resolvedPath}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple) or hasattr(value, "__iter__"):
        try:
            return [_json_value(item) for item in value]
        except TypeError:
            pass
    return str(value)


def _attribute_identity(attribute: Usd.Attribute) -> dict[str, Any]:
    samples = tuple(float(item) for item in attribute.GetTimeSamples())
    return {
        "name": attribute.GetName(),
        "type": str(attribute.GetTypeName()),
        "metadata": _json_value(attribute.GetAllAuthoredMetadata()),
        "default": _json_value(attribute.Get()),
        "connections": tuple(str(path) for path in attribute.GetConnections()),
        "time_samples": tuple(
            (sample, _json_value(attribute.Get(sample))) for sample in samples
        ),
    }


def _material_identity(stage: Usd.Stage, material_path: str | None) -> str:
    if material_path is None:
        return canonical_asset_digest({"material_path": None})
    root = stage.GetPrimAtPath(material_path)
    if not root or not root.IsA(UsdShade.Material):
        raise ValueError(f"Bound Texture material is unavailable: {material_path}")
    connected_prims: dict[str, Usd.Prim] = {}
    pending = [root]
    while pending:
        network_root = pending.pop()
        for prim in Usd.PrimRange(network_root):
            prim_path = str(prim.GetPath())
            if prim_path in connected_prims:
                continue
            connected_prims[prim_path] = prim
            for attribute in prim.GetAttributes():
                for connection in attribute.GetConnections():
                    connected = stage.GetPrimAtPath(connection.GetPrimPath())
                    if not connected:
                        raise ValueError(
                            "Texture material has an unresolved connected source: "
                            f"{connection}"
                        )
                    if str(connected.GetPath()) not in connected_prims:
                        pending.append(connected)

    prims: list[dict[str, Any]] = []
    for prim_path in sorted(connected_prims):
        prim = connected_prims[prim_path]
        prims.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "active": prim.IsActive(),
                "metadata": _json_value(prim.GetAllAuthoredMetadata()),
                "attributes": tuple(
                    _attribute_identity(attribute)
                    for attribute in sorted(
                        prim.GetAttributes(), key=lambda item: item.GetName()
                    )
                    if attribute.HasAuthoredValueOpinion() or attribute.GetConnections()
                ),
                "relationships": tuple(
                    {
                        "name": relationship.GetName(),
                        "metadata": _json_value(relationship.GetAllAuthoredMetadata()),
                        "targets": tuple(
                            str(path) for path in relationship.GetTargets()
                        ),
                    }
                    for relationship in sorted(
                        prim.GetRelationships(), key=lambda item: item.GetName()
                    )
                    if relationship.HasAuthoredTargets()
                ),
            }
        )
    return canonical_asset_digest(
        {
            "material_path": material_path,
            "reachable_shading_prims": tuple(prims),
        }
    )


def _effective_material_identity(
    stage: Usd.Stage,
    prim: Usd.Prim,
    *,
    material_purpose: TextureUvMaterialPurpose = "allPurpose",
) -> tuple[str | None, str | None, str]:
    purpose_token = (
        UsdShade.Tokens.allPurpose
        if material_purpose == "allPurpose"
        else material_purpose
    )
    material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
        purpose_token
    )
    material_path = (
        str(material.GetPath()) if material and material.GetPrim().IsValid() else None
    )
    relationship_path = (
        str(relationship.GetPath()) if relationship and relationship.IsValid() else None
    )
    return (
        material_path,
        relationship_path,
        _material_identity(stage, material_path),
    )


def _effective_material_purposes(
    stage: Usd.Stage,
    prim: Usd.Prim,
) -> tuple[TextureUvMaterialPurpose, ...]:
    """Return distinct effective bindings, including purpose-specific ones."""

    purposes: tuple[TextureUvMaterialPurpose, ...] = (
        "allPurpose",
        "preview",
        "full",
    )
    identities = {
        purpose: _effective_material_identity(
            stage,
            prim,
            material_purpose=purpose,
        )
        for purpose in purposes
    }
    selected: list[TextureUvMaterialPurpose] = []
    seen: set[tuple[str | None, str | None, str]] = set()
    for purpose in purposes:
        identity = identities[purpose]
        if identity[0] is None or identity[1] is None or identity in seen:
            continue
        selected.append(purpose)
        seen.add(identity)
    return tuple(selected)


def _resolve_uv_scope(
    stage: Usd.Stage,
    requested_paths: tuple[str, ...],
) -> tuple[
    tuple[str, ...],
    tuple[
        tuple[
            str,
            str,
            str,
            TextureUvTargetKind,
            TextureUvMaterialPurpose,
        ],
        ...,
    ],
]:
    resolved: list[str] = []
    scope: list[
        tuple[
            str,
            str,
            str,
            TextureUvTargetKind,
            TextureUvMaterialPurpose,
        ]
    ] = []

    def rows_for_mesh(
        requested: str,
        mesh_prim: Usd.Prim,
        mesh_kind: TextureUvTargetKind,
    ) -> tuple[
        tuple[
            Usd.Prim,
            Usd.Prim,
            TextureUvTargetKind,
            TextureUvMaterialPurpose,
        ],
        ...,
    ]:
        mesh_purposes = _effective_material_purposes(stage, mesh_prim)
        rows: list[
            tuple[
                Usd.Prim,
                Usd.Prim,
                TextureUvTargetKind,
                TextureUvMaterialPurpose,
            ]
        ] = [
            (mesh_prim, mesh_prim, mesh_kind, purpose)
            for purpose in (mesh_purposes or ("allPurpose",))
        ]
        for child in mesh_prim.GetChildren():
            if not child.IsA(UsdGeom.Subset):
                continue
            rows.extend(
                (child, mesh_prim, "subset", purpose)
                for purpose in _effective_material_purposes(stage, child)
            )
        return tuple(rows)

    for requested in requested_paths:
        prim = stage.GetPrimAtPath(requested)
        if not prim:
            raise ValueError(f"Texture UV target does not exist: {requested}")
        if prim.IsA(UsdGeom.Subset):
            parent = prim.GetParent()
            if not parent.IsA(UsdGeom.Mesh):
                raise ValueError(f"Texture UV subset has no mesh parent: {requested}")
            candidate_rows: tuple[
                tuple[
                    Usd.Prim,
                    Usd.Prim,
                    TextureUvTargetKind,
                    TextureUvMaterialPurpose,
                ],
                ...,
            ] = tuple(
                (prim, parent, "subset", purpose)
                for purpose in (
                    _effective_material_purposes(stage, prim) or ("allPurpose",)
                )
            )
        elif prim.IsA(UsdGeom.Mesh):
            candidate_rows = rows_for_mesh(requested, prim, "mesh")
        else:
            descendants = tuple(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
            descendant_rows = tuple(
                row
                for item in descendants
                if item.IsA(UsdGeom.Mesh)
                for row in rows_for_mesh(requested, item, "descendant_mesh")
            )
            candidate_rows = descendant_rows
        if not candidate_rows:
            raise ValueError(f"Texture UV target contains no meshes: {requested}")
        for scope_prim, mesh_prim, target_kind, material_purpose in candidate_rows:
            mesh_path = str(mesh_prim.GetPath())
            resolved.append(mesh_path)
            scope.append(
                (
                    requested,
                    str(scope_prim.GetPath()),
                    mesh_path,
                    target_kind,
                    material_purpose,
                )
            )
    return tuple(dict.fromkeys(resolved)), tuple(dict.fromkeys(scope))


def _expected_element_count(mesh: UsdGeom.Mesh, interpolation: str) -> int:
    if interpolation == "constant":
        return 1
    points = mesh.GetPointsAttr().Get() or []
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    if interpolation == "uniform":
        return len(counts)
    if interpolation in {"vertex", "varying"}:
        return len(points)
    if interpolation == "faceVarying":
        return sum(int(count) for count in counts)
    return 0


def _topology_time_sample_payload(mesh: UsdGeom.Mesh) -> dict[str, Any]:
    attributes = (
        mesh.GetPointsAttr(),
        mesh.GetFaceVertexCountsAttr(),
        mesh.GetFaceVertexIndicesAttr(),
        mesh.GetHoleIndicesAttr(),
    )
    return {
        attribute.GetName(): tuple(
            (float(sample), _json_value(attribute.Get(sample)))
            for sample in attribute.GetTimeSamples()
        )
        for attribute in attributes
        if attribute.IsValid() and attribute.GetTimeSamples()
    }


def _numeric_rows(values: Any) -> tuple[tuple[float, ...], ...]:
    if values is None:
        return ()
    rows: list[tuple[float, ...]] = []
    try:
        for value in values:
            try:
                rows.append(tuple(float(component) for component in value))
            except TypeError:
                rows.append((float(value),))
    except (TypeError, ValueError):
        return ()
    return tuple(rows)


def _uv_readback(stage: Usd.Stage, prim_path: str) -> TextureUvMeshReadback:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"Texture UV readback mesh is unavailable: {prim_path}")
    mesh = UsdGeom.Mesh(prim)
    primvar = UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance("st")
    attribute = primvar.GetAttr() if primvar else Usd.Attribute()
    values = primvar.Get() if attribute.IsValid() else None
    value_rows = _numeric_rows(values)
    indexed = bool(primvar and primvar.IsIndexed())
    indices_attribute = primvar.GetIndicesAttr() if primvar else Usd.Attribute()
    indices_raw = primvar.GetIndices() if indexed else None
    indices = (
        tuple(int(item) for item in indices_raw) if indices_raw is not None else ()
    )
    interpolation = str(primvar.GetInterpolation()) if primvar else ""
    value_time_samples = (
        tuple(float(item) for item in attribute.GetTimeSamples())
        if attribute.IsValid()
        else ()
    )
    index_time_samples = (
        tuple(float(item) for item in indices_attribute.GetTimeSamples())
        if indices_attribute.IsValid()
        else ()
    )
    time_sample_payload = {
        "values": tuple(
            (sample, _json_value(attribute.Get(sample)))
            for sample in value_time_samples
        ),
        "indices": tuple(
            (sample, _json_value(indices_attribute.Get(sample)))
            for sample in index_time_samples
        ),
    }
    topology_time_sample_payload = _topology_time_sample_payload(mesh)
    topology_time_samples = tuple(
        sorted(
            {
                float(sample)
                for rows in topology_time_sample_payload.values()
                for sample, _value in rows
            }
        )
    )
    expected = _expected_element_count(mesh, interpolation)
    reasons: list[str] = []
    uv_type = str(attribute.GetTypeName()) if attribute.IsValid() else ""
    status: Literal["ready", "missing", "repair_required", "unsupported"]
    if topology_time_samples:
        status = "unsupported"
        reasons.append("time-varying mesh topology is unsupported")
    elif not primvar or not attribute.IsValid():
        status = "missing"
        reasons.append("primvars:st is missing or empty")
    elif uv_type not in _SUPPORTED_UV_ARRAY_TYPES:
        status = "unsupported"
        reasons.append(f"unsupported primvars:st array type {uv_type or '<empty>'}")
    elif value_rows and any(len(row) != 2 for row in value_rows):
        status = "unsupported"
        reasons.append("primvars:st values must contain exactly two components")
    elif value_time_samples or index_time_samples:
        status = "unsupported"
        reasons.append("time-sampled primvars:st is inspection-only")
    elif not value_rows:
        status = "missing"
        reasons.append("primvars:st is missing or empty")
    elif interpolation not in {"uniform", "vertex", "varying", "faceVarying"}:
        status = "repair_required" if interpolation == "constant" else "unsupported"
        reasons.append(f"unsupported UV interpolation {interpolation or '<empty>'}")
    elif expected <= 0:
        status = "unsupported"
        reasons.append("mesh topology has no supported UV element count")
    elif not all(math.isfinite(component) for row in value_rows for component in row):
        status = "repair_required"
        reasons.append("primvars:st contains non-finite values")
    elif indexed and (
        len(indices) != expected
        or min(indices, default=0) < 0
        or max(indices, default=-1) >= len(value_rows)
    ):
        status = "repair_required"
        reasons.append("primvars:st indices differ from mesh topology")
    elif not indexed and len(value_rows) != expected:
        status = "repair_required"
        reasons.append("primvars:st value count differs from mesh topology")
    else:
        status = "ready"
    material_path, relationship_path, material_digest = _effective_material_identity(
        stage, prim
    )
    values_digest = canonical_asset_digest(value_rows)
    indices_digest = canonical_asset_digest(indices)
    uv_payload = {
        "prim_path": prim_path,
        "primvar_source_prim_path": (
            str(attribute.GetPrim().GetPath()) if attribute.IsValid() else None
        ),
        "status": status,
        "interpolation": interpolation,
        "indexed": indexed,
        "value_count": len(value_rows),
        "index_count": len(indices),
        "expected_element_count": expected,
        "topology_time_sample_count": len(topology_time_samples),
        "topology_time_samples_sha256": canonical_asset_digest(
            topology_time_sample_payload
        ),
        "value_time_sample_count": len(value_time_samples),
        "index_time_sample_count": len(index_time_samples),
        "values_sha256": values_digest,
        "indices_sha256": indices_digest,
        "time_samples_sha256": canonical_asset_digest(time_sample_payload),
    }
    return TextureUvMeshReadback.model_validate(
        {
            **uv_payload,
            "uv_identity_sha256": canonical_asset_digest(uv_payload),
            "effective_material_path": material_path,
            "binding_relationship_path": relationship_path,
            "material_identity_sha256": material_digest,
            "reasons": tuple(reasons),
        }
    )


def _stage_readbacks(
    stage: Usd.Stage,
    mesh_paths: tuple[str, ...],
) -> tuple[TextureUvMeshReadback, ...]:
    return tuple(_uv_readback(stage, path) for path in mesh_paths)


def _scope_readbacks(
    stage: Usd.Stage,
    scope: tuple[
        tuple[
            str,
            str,
            str,
            TextureUvTargetKind,
            TextureUvMaterialPurpose,
        ],
        ...,
    ],
) -> tuple[TextureUvScopeReadback, ...]:
    readbacks: list[TextureUvScopeReadback] = []
    for requested, scope_path, mesh_path, target_kind, material_purpose in scope:
        prim = stage.GetPrimAtPath(scope_path)
        if not prim:
            raise ValueError(f"Texture UV scope disappeared from stage: {scope_path}")
        material_path, relationship_path, material_digest = (
            _effective_material_identity(
                stage,
                prim,
                material_purpose=material_purpose,
            )
        )
        payload = {
            "requested_prim_path": requested,
            "scope_prim_path": scope_path,
            "uv_mesh_path": mesh_path,
            "target_kind": target_kind,
            "material_purpose": material_purpose,
            "effective_material_path": material_path,
            "binding_relationship_path": relationship_path,
            "material_identity_sha256": material_digest,
        }
        readbacks.append(
            TextureUvScopeReadback.model_validate(
                {
                    **payload,
                    "scope_identity_sha256": canonical_asset_digest(payload),
                }
            )
        )
    return tuple(readbacks)


def _bounded_box_fallback_values(
    mesh: UsdGeom.Mesh,
    *,
    margin: float = 0.05,
) -> tuple[tuple[float, float], ...]:
    if _topology_time_sample_payload(mesh):
        raise ValueError(
            f"Texture UV authoring rejects time-varying topology: {mesh.GetPath()}"
        )
    points_raw = mesh.GetPointsAttr().Get()
    indices_raw = mesh.GetFaceVertexIndicesAttr().Get()
    counts_raw = mesh.GetFaceVertexCountsAttr().Get()
    if points_raw is None or indices_raw is None or counts_raw is None:
        raise ValueError(f"Texture UV mesh has incomplete topology: {mesh.GetPath()}")
    points = tuple(
        tuple(float(component) for component in point) for point in points_raw
    )
    indices = tuple(int(item) for item in indices_raw)
    counts = tuple(int(item) for item in counts_raw)
    if (
        not points
        or not indices
        or any(count < 3 for count in counts)
        or sum(counts) != len(indices)
    ):
        raise ValueError(f"Texture UV mesh has invalid topology: {mesh.GetPath()}")
    if min(indices) < 0 or max(indices) >= len(points):
        raise ValueError(f"Texture UV mesh indices are out of range: {mesh.GetPath()}")
    if any(
        len(point) != 3 or any(not math.isfinite(component) for component in point)
        for point in points
    ):
        raise ValueError(f"Texture UV mesh has non-finite points: {mesh.GetPath()}")
    used = tuple(points[index] for index in indices)
    minimum = tuple(min(point[axis] for point in used) for axis in range(3))
    maximum = tuple(max(point[axis] for point in used) for axis in range(3))
    normalization_extent = tuple(
        maximum[axis] - minimum[axis] if maximum[axis] != minimum[axis] else 1.0
        for axis in range(3)
    )

    def normalized(
        point: tuple[float, ...], axes: tuple[int, int]
    ) -> tuple[float, float]:
        return tuple(
            ((point[axis] - minimum[axis]) / normalization_extent[axis])
            * (1.0 - 2.0 * margin)
            + margin
            for axis in axes
        )  # type: ignore[return-value]

    face_points_by_face: list[tuple[tuple[float, ...], ...]] = []
    face_normals: list[tuple[float, float, float]] = []
    cursor = 0
    for count in counts:
        face_points = tuple(points[indices[cursor + offset]] for offset in range(count))
        # Newell's method uses the complete polygon.  In particular it does not
        # collapse a valid face merely because its first three vertices happen
        # to be collinear.
        normal = tuple(
            sum(
                (point[(axis + 1) % 3] - next_point[(axis + 1) % 3])
                * (point[(axis + 2) % 3] + next_point[(axis + 2) % 3])
                for point, next_point in zip(
                    face_points,
                    (*face_points[1:], face_points[0]),
                    strict=True,
                )
            )
            for axis in range(3)
        )
        face_points_by_face.append(face_points)
        face_normals.append(normal)  # type: ignore[arg-type]
        cursor += count

    valid_normals = tuple(
        normal
        for normal in face_normals
        if any(component != 0.0 for component in normal)
    )
    if not valid_normals:
        raise ValueError(
            f"Texture UV mesh has no non-collinear projection basis: {mesh.GetPath()}"
        )
    fallback_normal = max(
        valid_normals,
        key=lambda normal: sum(component * component for component in normal),
    )

    values: list[tuple[float, float]] = []
    for face_points, normal in zip(
        face_points_by_face,
        face_normals,
        strict=True,
    ):
        projection_normal = (
            normal if any(component != 0.0 for component in normal) else fallback_normal
        )
        dominant = max(range(3), key=lambda axis: abs(projection_normal[axis]))
        axes = ((1, 2), (0, 2), (0, 1))[dominant]
        values.extend(normalized(point, axes) for point in face_points)
    return tuple(values)


def _author_uvs(
    stage: Usd.Stage,
    mesh_path: str,
) -> None:
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"Texture UV authoring mesh is unavailable: {mesh_path}")
    values = _bounded_box_fallback_values(UsdGeom.Mesh(prim))
    primvar = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    primvar.GetAttr().Clear()
    indices_attribute = primvar.GetIndicesAttr()
    if indices_attribute.IsValid():
        indices_attribute.Clear()
    primvar.SetInterpolation(UsdGeom.Tokens.faceVarying)
    if primvar.IsIndexed():
        primvar.BlockIndices()
    primvar.Set(Vt.Vec2fArray([Gf.Vec2f(u, v) for u, v in values]))


def _create_source_overlay_stage(
    source_stage: Usd.Stage,
    *,
    frozen_source_layer: Sdf.Layer,
) -> Usd.Stage:
    """Create a stronger authoring layer without flattening source composition."""

    overlay = Sdf.Layer.CreateAnonymous("prepared_texture_uvs.usda")
    overlay.subLayerPaths = [frozen_source_layer.identifier]
    source_root = source_stage.GetRootLayer()
    _copy_stage_metadata(source_root, overlay)
    _require_stage_metadata_preserved(source_root, overlay)
    working_stage = Usd.Stage.Open(overlay, load=Usd.Stage.LoadAll)
    if working_stage is None:
        raise ValueError("Texture UV source-preserving overlay could not be opened")
    working_stage.SetEditTarget(overlay)
    return working_stage


def _export_source_preserving_overlay(
    working_stage: Usd.Stage,
    *,
    source_path: Path,
) -> bytes:
    """Detach an authored overlay, restore its public source arc, and export."""

    overlay = Sdf.Layer.CreateAnonymous("prepared_texture_uvs-export.usda")
    overlay.TransferContent(working_stage.GetRootLayer())
    overlay.subLayerPaths = [source_path.as_posix()]
    _require_stage_metadata_preserved(working_stage.GetRootLayer(), overlay)
    return str(overlay.ExportToString()).encode("utf-8")


def _layer_from_saved_snapshot(document: bytes) -> Sdf.Layer:
    """Open exactly the already-bound authored USDA bytes, never their path."""

    try:
        text = document.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Texture UV authored overlay is not valid UTF-8 USDA") from exc
    layer = Sdf.Layer.CreateAnonymous("prepared_texture_uvs.usda")
    if not layer.ImportFromString(text):
        raise ValueError("Texture UV authored overlay could not be imported")
    return layer


def _stage_from_saved_snapshot(
    document: bytes,
    *,
    source_path: Path,
    frozen_source_layer: Sdf.Layer,
) -> Usd.Stage:
    """Compose exact overlay bytes only with an exact frozen source graph."""

    layer = _layer_from_saved_snapshot(document)
    _require_stage_metadata_preserved(frozen_source_layer, layer)
    dependencies = tuple(layer.GetCompositionAssetDependencies())
    if dependencies != (source_path.as_posix(),):
        raise ValueError(
            "Texture UV authored overlay must have exactly one bound source sublayer"
        )
    if not layer.UpdateCompositionAssetDependency(
        dependencies[0],
        frozen_source_layer.identifier,
    ):
        raise ValueError("Texture UV authored overlay source could not be rebound")
    stage = Usd.Stage.Open(layer, load=Usd.Stage.LoadAll)
    if stage is None:
        raise ValueError("Texture UV final saved stage could not be reopened")
    return stage


def _expected_authored_overlay(
    *,
    source_path: Path,
    source_stage: Usd.Stage,
    frozen_source_layer: Sdf.Layer,
    authored_paths: tuple[str, ...],
) -> bytes:
    working_stage = _create_source_overlay_stage(
        source_stage,
        frozen_source_layer=frozen_source_layer,
    )
    for mesh_path in authored_paths:
        _author_uvs(working_stage, mesh_path)
    return _export_source_preserving_overlay(
        working_stage,
        source_path=source_path,
    )


def _verify_deterministic_uv_overlay(
    *,
    document: bytes,
    source_path: Path,
    source_stage: Usd.Stage,
    frozen_source_layer: Sdf.Layer,
    authored_paths: tuple[str, ...],
) -> Usd.Stage:
    """Prove exact UV values and the complete source-preserving layer delta."""

    expected_document = _expected_authored_overlay(
        source_path=source_path,
        source_stage=source_stage,
        frozen_source_layer=frozen_source_layer,
        authored_paths=authored_paths,
    )
    actual_layer = _layer_from_saved_snapshot(document)
    expected_layer = _layer_from_saved_snapshot(expected_document)
    if actual_layer.ExportToString() != expected_layer.ExportToString():
        raise ValueError(
            "Texture UV authored overlay contains a non-UV or non-deterministic delta"
        )
    output_stage = _stage_from_saved_snapshot(
        document,
        source_path=source_path,
        frozen_source_layer=frozen_source_layer,
    )
    for mesh_path in authored_paths:
        source_prim = source_stage.GetPrimAtPath(mesh_path)
        output_prim = output_stage.GetPrimAtPath(mesh_path)
        expected_values = tuple(
            tuple(float(component) for component in Gf.Vec2f(u, v))
            for u, v in _bounded_box_fallback_values(UsdGeom.Mesh(source_prim))
        )
        primvar = UsdGeom.PrimvarsAPI(output_prim).GetPrimvar("st")
        mismatch_detail = (
            "Texture UV authored overlay does not contain the exact expected "
            f"bounded box fallback values: {mesh_path}"
        )
        if not primvar:
            raise ValueError(mismatch_detail)
        observed_values = tuple(
            tuple(float(component) for component in value)
            for value in (primvar.Get() or ())
        )
        if (
            primvar.GetInterpolation() != UsdGeom.Tokens.faceVarying
            or primvar.IsIndexed()
            or observed_values != expected_values
        ):
            raise ValueError(mismatch_detail)
    return output_stage


def _build_readback(
    *,
    source: ExecutionArtifactBinding,
    source_dependencies: tuple[ArtifactBinding, ...],
    saved_stage: ExecutionArtifactBinding,
    saved_stage_dependencies: tuple[ArtifactBinding, ...],
    invocation: TextureUvLeafInvocation,
    resolved_mesh_paths: tuple[str, ...],
    authored_mesh_paths: tuple[str, ...],
    before: tuple[TextureUvMeshReadback, ...],
    after: tuple[TextureUvMeshReadback, ...],
    before_scope: tuple[TextureUvScopeReadback, ...],
    after_scope: tuple[TextureUvScopeReadback, ...],
) -> TextureUvSavedStageReadback:
    before_by_path = {item.prim_path: item for item in before}
    after_by_path = {item.prim_path: item for item in after}
    preserved = tuple(
        path for path in resolved_mesh_paths if path not in set(authored_mesh_paths)
    )
    before_scope_by_key = {
        (
            item.requested_prim_path,
            item.scope_prim_path,
            item.uv_mesh_path,
            item.material_purpose,
        ): item
        for item in before_scope
    }
    after_scope_by_key = {
        (
            item.requested_prim_path,
            item.scope_prim_path,
            item.uv_mesh_path,
            item.material_purpose,
        ): item
        for item in after_scope
    }
    materials_unchanged = (
        before_scope_by_key.keys() == after_scope_by_key.keys()
        and all(
            before_scope_by_key[key].effective_material_path
            == after_scope_by_key[key].effective_material_path
            and before_scope_by_key[key].binding_relationship_path
            == after_scope_by_key[key].binding_relationship_path
            and before_scope_by_key[key].material_identity_sha256
            == after_scope_by_key[key].material_identity_sha256
            for key in before_scope_by_key
        )
    )
    preserved_unchanged = all(
        before_by_path[mesh_path].uv_identity_sha256
        == after_by_path[mesh_path].uv_identity_sha256
        for mesh_path in preserved
    )
    usd_suffixes = {".usd", ".usda", ".usdc", ".usdz"}
    source_external = sorted(
        (binding.sha256, binding.size_bytes)
        for binding in source_dependencies
        if binding.path != source.path
        and Path(binding.path).suffix.lower() not in usd_suffixes
    )
    saved_external = sorted(
        (binding.sha256, binding.size_bytes)
        for binding in saved_stage_dependencies
        if binding.path != saved_stage.path
        and Path(binding.path).suffix.lower() not in usd_suffixes
    )
    external_dependencies_preserved = all(
        source_external.count(identity) <= saved_external.count(identity)
        for identity in set(source_external)
    )
    payload = {
        "schema_version": TEXTURE_UV_READBACK_SCHEMA_VERSION,
        "source": source.model_dump(mode="json"),
        "source_dependencies": tuple(
            item.model_dump(mode="json") for item in source_dependencies
        ),
        "saved_stage": saved_stage.model_dump(mode="json"),
        "saved_stage_dependencies": tuple(
            item.model_dump(mode="json") for item in saved_stage_dependencies
        ),
        "policy": invocation.policy,
        "authoring_method": ("bounded_box_fallback" if authored_mesh_paths else None),
        "requested_prim_paths": invocation.target_prim_paths,
        "resolved_mesh_paths": resolved_mesh_paths,
        "authored_mesh_paths": authored_mesh_paths,
        "preserved_mesh_paths": preserved,
        "mesh_readbacks": tuple(item.model_dump(mode="json") for item in after),
        "scope_readbacks": tuple(item.model_dump(mode="json") for item in after_scope),
        "all_uv_ready": all(item.status == "ready" for item in after),
        "material_identities_unchanged": materials_unchanged,
        "preserved_uv_identities_unchanged": preserved_unchanged,
        "external_dependencies_preserved": external_dependencies_preserved,
        "reopened_saved_stage": True,
    }
    readback = TextureUvSavedStageReadback.model_validate(
        {
            **payload,
            "readback_identity_sha256": canonical_asset_digest(payload),
        }
    )
    return readback


def _expected_authored_paths(
    invocation: TextureUvLeafInvocation,
    before: tuple[TextureUvMeshReadback, ...],
) -> tuple[str, ...]:
    blockers = tuple(item for item in before if item.status not in {"ready", "missing"})
    if invocation.policy != "generate_missing" or blockers:
        return ()
    return tuple(item.prim_path for item in before if item.status == "missing")


def _reject_variant_composed_authoring(
    stage: Usd.Stage,
    authored_paths: tuple[str, ...],
) -> None:
    def has_variant_composition(prim: Usd.Prim) -> bool:
        pending = [prim.GetPrimIndex().rootNode]
        while pending:
            node = pending.pop()
            if node.path.ContainsPrimVariantSelection():
                return True
            pending.extend(node.children)
        return any(
            spec.path.ContainsPrimVariantSelection() for spec in prim.GetPrimStack()
        )

    variant_paths = tuple(
        path
        for path in authored_paths
        if has_variant_composition(stage.GetPrimAtPath(path))
    )
    if variant_paths:
        raise ValueError(
            "Texture UV authoring rejects variant-composed target meshes: "
            f"{variant_paths}"
        )


def _reject_instance_proxy_authoring(
    stage: Usd.Stage,
    authored_paths: tuple[str, ...],
) -> None:
    proxy_paths = tuple(
        path for path in authored_paths if stage.GetPrimAtPath(path).IsInstanceProxy()
    )
    if proxy_paths:
        raise ValueError(
            "Texture UV authoring rejects read-only instance proxy meshes: "
            f"{proxy_paths}"
        )


def _preflight_bounded_box_fallback_authoring(
    stage: Usd.Stage,
    authored_paths: tuple[str, ...],
) -> None:
    """Validate every selected mesh before authoring any in-memory opinions."""

    for mesh_path in authored_paths:
        prim = stage.GetPrimAtPath(mesh_path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Texture UV authoring mesh is unavailable: {mesh_path}")
        _bounded_box_fallback_values(UsdGeom.Mesh(prim))


def _deterministic_authoring_rejection_detail(
    stage: Usd.Stage,
    authored_paths: tuple[str, ...],
) -> str | None:
    """Return an exact native failure for an expected deterministic rejection."""

    if not authored_paths:
        return None
    try:
        _reject_variant_composed_authoring(stage, authored_paths)
        _reject_instance_proxy_authoring(stage, authored_paths)
        _preflight_bounded_box_fallback_authoring(stage, authored_paths)
    except ValueError as exc:
        return f"Texture UV deterministic authoring preflight rejected: {exc}"
    return None


def _expected_evidence(
    *,
    invocation: TextureUvLeafInvocation,
    saved_stage: ExecutionArtifactBinding,
    mesh_paths: tuple[str, ...],
    authored_paths: tuple[str, ...],
    before: tuple[TextureUvMeshReadback, ...],
    before_scope: tuple[TextureUvScopeReadback, ...],
    readback: TextureUvSavedStageReadback,
) -> TextureUvEvidence:
    return TextureUvEvidence(
        source=invocation.source,
        saved_stage=saved_stage,
        policy=invocation.policy,
        authoring_method=readback.authoring_method,
        requested_prim_paths=invocation.target_prim_paths,
        resolved_mesh_paths=mesh_paths,
        authored_mesh_paths=authored_paths,
        before_uv_identity_sha256={
            item.prim_path: item.uv_identity_sha256 for item in before
        },
        before_scope_identity_sha256={
            "|".join(
                (
                    f"requested={item.requested_prim_path}",
                    f"scope={item.scope_prim_path}",
                    f"mesh={item.uv_mesh_path}",
                    f"purpose={item.material_purpose}",
                )
            ): item.scope_identity_sha256
            for item in before_scope
        },
        saved_stage_readback_identity_sha256=readback.readback_identity_sha256,
        external_dependencies_preserved=readback.external_dependencies_preserved,
    )


def _expected_native_disposition(
    *,
    policy: TextureUvPolicy,
    all_uv_ready: bool,
) -> Literal["passed", "failed", "not_evaluated"]:
    if all_uv_ready:
        return "passed"
    return "failed" if policy == "generate_missing" else "not_evaluated"


def _expected_result_detail(
    *,
    policy: TextureUvPolicy,
    native_disposition: Literal["passed", "failed", "not_evaluated"],
    mesh_readbacks: tuple[TextureUvMeshReadback, ...],
) -> str:
    """Derive the native result detail from saved-stage facts and policy."""

    if native_disposition == "passed":
        return "Saved-stage UV and material identities passed reopened-stage readback."
    blockers = tuple(
        item
        for item in mesh_readbacks
        if item.status in {"repair_required", "unsupported"}
    )
    if not blockers:
        return "UV inspection completed but the selected stage requires authoring."
    blocker_reasons = tuple(
        f"{item.prim_path}: {reason}"
        for item in blockers
        for reason in (item.reasons or (item.status,))
    )
    if policy == "generate_missing":
        return (
            "UV authoring refused this saved stage fail closed; saved-stage "
            f"readback reasons: {'; '.join(blocker_reasons)}."
        )
    blocker_statuses = tuple(f"{item.prim_path}: {item.status}" for item in blockers)
    return (
        "UV inspection completed with non-ready saved-stage readback statuses: "
        f"{'; '.join(blocker_statuses)}; saved-stage readback reasons: "
        f"{'; '.join(blocker_reasons)}."
    )


def _recompute_saved_stage_contract(
    *,
    attempt: _HeldDirectory,
    invocation: TextureUvLeafInvocation,
    result: TextureUvLeafResult,
) -> tuple[TextureUvSavedStageReadback, TextureUvEvidence, str | None]:
    """Re-derive the native contract from exact source and output stage bytes."""

    source_path = _verify_binding(invocation.source, label="Texture UV source")
    current_source_dependencies = tuple(
        bind_usd_dependency_closure(invocation.source.path)
    )
    if current_source_dependencies != invocation.source_dependencies:
        raise ValueError("Texture UV source dependency closure changed")
    verify_usd_dependency_closure(source_path, invocation.source_dependencies)
    source_snapshot = _FrozenUsdSnapshot.open(
        invocation.source,
        current_source_dependencies,
    )
    attempt.retain_cleanup(source_snapshot.close)
    source_stage = source_snapshot.open_stage()
    mesh_paths, scope = _resolve_uv_scope(
        source_stage,
        invocation.target_prim_paths,
    )
    before = _stage_readbacks(source_stage, mesh_paths)
    before_scope = _scope_readbacks(source_stage, scope)
    planned_authored_paths = _expected_authored_paths(invocation, before)
    rejection_detail = _deterministic_authoring_rejection_detail(
        source_stage,
        planned_authored_paths,
    )
    authored_paths = () if rejection_detail is not None else planned_authored_paths
    should_author = bool(authored_paths)
    if should_author != (result.output != invocation.source):
        raise ValueError(
            "Texture UV native output does not match recomputed authoring decision"
        )

    if should_author:
        output_name = attempt.direct_name(result.output.path)
        output_binding, output_bytes, _output_identity = attempt.read(output_name)
        if output_binding != result.output:
            raise ValueError(
                "Texture UV output identity changed from its exact bound bytes"
            )
        output_stage = _verify_deterministic_uv_overlay(
            document=output_bytes,
            source_path=source_path,
            source_stage=source_stage,
            frozen_source_layer=source_snapshot.root_layer,
            authored_paths=authored_paths,
        )
        current_output_dependencies = tuple(
            bind_usd_dependency_closure(result.output.path)
        )
        _require_exact_package_predecessor_dependency(
            invocation.source,
            current_output_dependencies,
        )
    else:
        if result.output != invocation.source:
            raise ValueError("Texture UV non-authoring result changed its output")
        output_stage = source_stage
        current_output_dependencies = current_source_dependencies
    if current_output_dependencies != result.output_dependencies:
        raise ValueError("Texture UV output dependency closure changed")
    verify_usd_dependency_closure(result.output.path, result.output_dependencies)

    after_mesh_paths, after_scope_shape = _resolve_uv_scope(
        output_stage,
        invocation.target_prim_paths,
    )
    if after_mesh_paths != mesh_paths or after_scope_shape != scope:
        raise ValueError("Texture UV saved stage changed requested scope topology")
    after = _stage_readbacks(output_stage, mesh_paths)
    after_scope = _scope_readbacks(output_stage, scope)
    recomputed = _build_readback(
        source=invocation.source,
        source_dependencies=current_source_dependencies,
        saved_stage=result.output,
        saved_stage_dependencies=current_output_dependencies,
        invocation=invocation,
        resolved_mesh_paths=mesh_paths,
        authored_mesh_paths=authored_paths,
        before=before,
        after=after,
        before_scope=before_scope,
        after_scope=after_scope,
    )
    evidence = _expected_evidence(
        invocation=invocation,
        saved_stage=result.output,
        mesh_paths=mesh_paths,
        authored_paths=authored_paths,
        before=before,
        before_scope=before_scope,
        readback=recomputed,
    )
    _verify_binding(invocation.source, label="Texture UV source")
    verify_usd_dependency_closure(source_path, invocation.source_dependencies)
    if should_author:
        attempt.verify_binding(result.output, label="Texture UV output")
    attempt.verify_path()
    return recomputed, evidence, rejection_detail


def run_texture_uv_leaf(
    invocation_path: str | Path,
) -> TextureUvLeafResult:
    """Execute one typed UV leaf and return only after saved-stage readback."""

    invocation_candidate = Path(invocation_path).expanduser()
    if not invocation_candidate.is_absolute():
        invocation_candidate = Path.cwd() / invocation_candidate
    invocation_root = invocation_candidate.parent.resolve(strict=True)
    with _HeldDirectory.open(invocation_root) as attempt:
        invocation_name = attempt.direct_name(invocation_candidate)
        try:
            invocation_binding, invocation_bytes, _invocation_identity = attempt.read(
                invocation_name,
                max_bytes=_MAX_INVOCATION_BYTES,
            )
            invocation_payload = json.loads(invocation_bytes)
            if not isinstance(invocation_payload, dict):
                raise ValueError("invocation JSON must be an object")
            invocation = TextureUvLeafInvocation.model_validate(invocation_payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid Texture UV leaf invocation: {exc}") from exc
        attempt.verify_binding(
            invocation_binding,
            label="Texture UV invocation",
        )
        if invocation.output_dir != str(attempt.path):
            raise ValueError(
                "Texture UV invocation must use its exact held attempt root"
            )

        source_path = _verify_binding(invocation.source, label="Texture UV source")
        verify_usd_dependency_closure(source_path, invocation.source_dependencies)
        names = {
            "output": "prepared_texture_uvs.usda",
            "evidence": "texture_uv_evidence.json",
            "readback": "texture_uv_saved_stage_readback.json",
            "result": "texture_uv_leaf_result.json",
        }
        for name in names.values():
            if attempt.exists(name):
                raise FileExistsError(
                    "Texture UV leaf refuses to replace artifact: "
                    f"{attempt.path / name}"
                )

        source_snapshot = _FrozenUsdSnapshot.open(
            invocation.source,
            invocation.source_dependencies,
        )
        attempt.retain_cleanup(source_snapshot.close)
        source_stage = source_snapshot.open_stage()
        mesh_paths, scope = _resolve_uv_scope(
            source_stage,
            invocation.target_prim_paths,
        )
        before = _stage_readbacks(source_stage, mesh_paths)
        before_scope = _scope_readbacks(source_stage, scope)
        attempt.verify_binding(invocation_binding, label="Texture UV invocation")
        _verify_binding(invocation.source, label="Texture UV source")
        verify_usd_dependency_closure(source_path, invocation.source_dependencies)

        planned_authored_paths = _expected_authored_paths(invocation, before)
        rejection_detail = _deterministic_authoring_rejection_detail(
            source_stage,
            planned_authored_paths,
        )
        authored_paths = () if rejection_detail is not None else planned_authored_paths
        should_author = bool(authored_paths)
        saved_stage_binding = invocation.source
        final_stage = source_stage
        saved_stage_descriptor = -1
        created: dict[str, _FileIdentity] = {}
        try:
            if should_author:
                attempt.verify_binding(
                    invocation_binding,
                    label="Texture UV invocation",
                )
                _verify_binding(invocation.source, label="Texture UV source")
                verify_usd_dependency_closure(
                    source_path,
                    invocation.source_dependencies,
                )
                working_stage = _create_source_overlay_stage(
                    source_stage,
                    frozen_source_layer=source_snapshot.root_layer,
                )
                for mesh_path in authored_paths:
                    _author_uvs(working_stage, mesh_path)
                in_memory_after = _stage_readbacks(working_stage, mesh_paths)
                if any(item.status != "ready" for item in in_memory_after):
                    failed = {
                        item.prim_path: item.status
                        for item in in_memory_after
                        if item.status != "ready"
                    }
                    raise ValueError(
                        f"Texture UV authored stage is not UV-ready: {failed}"
                    )
                exported = _export_source_preserving_overlay(
                    working_stage,
                    source_path=source_path,
                )
                working_stage = None
                created[names["output"]] = attempt.write_bytes(
                    names["output"],
                    exported,
                )
                output_identity = created[names["output"]]
                saved_stage_descriptor = attempt.open_read(
                    names["output"],
                    identity=output_identity,
                )
                saved_stage_bytes, observed_identity = _read_regular_fd(
                    saved_stage_descriptor,
                    attempt.path / names["output"],
                )
                if observed_identity != output_identity:
                    raise ValueError(
                        "Texture UV output identity changed before exact reopen"
                    )
                saved_stage_binding = ExecutionArtifactBinding(
                    path=str(attempt.path / names["output"]),
                    sha256=hashlib.sha256(saved_stage_bytes).hexdigest(),
                    size_bytes=len(saved_stage_bytes),
                )
                final_stage = _verify_deterministic_uv_overlay(
                    document=saved_stage_bytes,
                    source_path=source_path,
                    source_stage=source_stage,
                    frozen_source_layer=source_snapshot.root_layer,
                    authored_paths=authored_paths,
                )
                attempt.verify_path()
                if attempt.file_identity(names["output"]) != output_identity:
                    raise ValueError(
                        "Texture UV output path changed after exact reopen"
                    )

            saved_stage_dependencies = (
                tuple(bind_usd_dependency_closure(saved_stage_binding.path))
                if should_author
                else invocation.source_dependencies
            )
            if should_author:
                _require_exact_package_predecessor_dependency(
                    invocation.source,
                    saved_stage_dependencies,
                )
            after = _stage_readbacks(final_stage, mesh_paths)
            after_scope = _scope_readbacks(final_stage, scope)
            attempt.verify_binding(invocation_binding, label="Texture UV invocation")
            _verify_binding(invocation.source, label="Texture UV source")
            verify_usd_dependency_closure(source_path, invocation.source_dependencies)
            if should_author:
                attempt.verify_binding(
                    saved_stage_binding,
                    label="Texture UV saved stage",
                )
                assert saved_stage_descriptor >= 0
                rebound_bytes, rebound_identity = _read_regular_fd(
                    saved_stage_descriptor,
                    attempt.path / names["output"],
                )
                if (
                    rebound_identity != created[names["output"]]
                    or hashlib.sha256(rebound_bytes).hexdigest()
                    != saved_stage_binding.sha256
                ):
                    raise ValueError(
                        "Texture UV saved-stage snapshot changed during readback"
                    )
            else:
                _verify_binding(saved_stage_binding, label="Texture UV saved stage")
            verify_usd_dependency_closure(
                saved_stage_binding.path,
                saved_stage_dependencies,
            )
            readback = _build_readback(
                source=invocation.source,
                source_dependencies=invocation.source_dependencies,
                saved_stage=saved_stage_binding,
                saved_stage_dependencies=saved_stage_dependencies,
                invocation=invocation,
                resolved_mesh_paths=mesh_paths,
                authored_mesh_paths=authored_paths,
                before=before,
                after=after,
                before_scope=before_scope,
                after_scope=after_scope,
            )
            if not readback.material_identities_unchanged:
                raise ValueError(
                    "Texture UV authoring changed saved-stage material identity"
                )
            if not readback.preserved_uv_identities_unchanged:
                raise ValueError("Texture UV authoring changed a preserved UV identity")
            if not readback.external_dependencies_preserved:
                raise ValueError(
                    "Texture UV saved stage changed an external dependency identity"
                )
            created[names["readback"]] = attempt.write_json(
                names["readback"],
                readback,
            )
            evidence = _expected_evidence(
                invocation=invocation,
                saved_stage=saved_stage_binding,
                mesh_paths=mesh_paths,
                authored_paths=authored_paths,
                before=before,
                before_scope=before_scope,
                readback=readback,
            )
            created[names["evidence"]] = attempt.write_json(
                names["evidence"],
                evidence,
            )
            evidence_binding = attempt.binding(
                names["evidence"],
                identity=created[names["evidence"]],
            )
            readback_binding = attempt.binding(
                names["readback"],
                identity=created[names["readback"]],
            )
            disposition = _expected_native_disposition(
                policy=invocation.policy,
                all_uv_ready=readback.all_uv_ready,
            )
            detail = rejection_detail or _expected_result_detail(
                policy=invocation.policy,
                native_disposition=disposition,
                mesh_readbacks=readback.mesh_readbacks,
            )
            result = TextureUvLeafResult(
                invocation=invocation_binding,
                native_disposition=disposition,
                detail=detail,
                error=detail if disposition == "failed" else None,
                source=invocation.source,
                output=saved_stage_binding,
                output_dependencies=saved_stage_dependencies,
                evidence=(evidence_binding,),
                saved_stage_readbacks=(readback_binding,),
            )
            attempt.verify_binding(invocation_binding, label="Texture UV invocation")
            _verify_binding(invocation.source, label="Texture UV source")
            verify_usd_dependency_closure(source_path, invocation.source_dependencies)
            if should_author:
                attempt.verify_binding(
                    saved_stage_binding,
                    label="Texture UV saved stage",
                )
            else:
                _verify_binding(saved_stage_binding, label="Texture UV saved stage")
            verify_usd_dependency_closure(
                saved_stage_binding.path,
                saved_stage_dependencies,
            )
            attempt.verify_binding(evidence_binding, label="Texture UV evidence")
            attempt.verify_binding(
                readback_binding,
                label="Texture UV saved-stage readback",
            )
            created[names["result"]] = attempt.write_json(names["result"], result)
            attempt.verify_binding(invocation_binding, label="Texture UV invocation")
            _verify_binding(invocation.source, label="Texture UV source")
            verify_usd_dependency_closure(source_path, invocation.source_dependencies)
            if should_author:
                attempt.verify_binding(
                    saved_stage_binding,
                    label="Texture UV saved stage",
                )
            else:
                _verify_binding(saved_stage_binding, label="Texture UV saved stage")
            verify_usd_dependency_closure(
                saved_stage_binding.path,
                saved_stage_dependencies,
            )
            attempt.verify_binding(evidence_binding, label="Texture UV evidence")
            attempt.verify_binding(
                readback_binding,
                label="Texture UV saved-stage readback",
            )
            attempt.binding(
                names["result"],
                identity=created[names["result"]],
            )
            attempt.verify_path()
            return result
        finally:
            if saved_stage_descriptor >= 0:
                os.close(saved_stage_descriptor)


__all__ = [
    "TEXTURE_UV_EVIDENCE_SCHEMA_VERSION",
    "TEXTURE_UV_INVOCATION_SCHEMA_VERSION",
    "TEXTURE_UV_LEAF_ID",
    "TEXTURE_UV_READBACK_SCHEMA_VERSION",
    "TEXTURE_UV_RESULT_SCHEMA_VERSION",
    "TextureUvEvidence",
    "TextureUvLeafInvocation",
    "TextureUvLeafResult",
    "TextureUvMeshReadback",
    "TextureUvSavedStageReadback",
    "TextureUvScopeReadback",
    "build_texture_uv_leaf_invocation",
    "run_texture_uv_leaf",
]
