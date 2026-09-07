# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Digest-bound usd-cli evidence for articulation candidates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import stat
import tempfile
import uuid
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, Self
from urllib.parse import quote

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
    read_contained_artifact,
)
from content_agent_workflows.common.usd_cli_session import (
    WorkflowUsdCliSession,
    direction_angles,
    stage_up_axis_is_y,
)

from .client import CancelChecker
from .models import (
    ArticulationWorkflowRequest,
    ArtifactBinding,
    Stage2CandidateDocument,
)

ARTICULATION_SCENE_EVIDENCE_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-scene-evidence.v1"
] = "content-agent-workflows.articulation-scene-evidence.v1"

_CONFIGURATION_DIGEST_SCHEMA_VERSION = (
    "content-agent-workflows.articulation-scene-evidence-configuration.v1"
)
_COLLECTION_SCHEMA_VERSION = (
    "content-agent-workflows.articulation-scene-evidence-collection.v1"
)
_COLLECTION_ID_PATTERN = r"^[0-9a-f]{32}$"
_COLLECTION_PREFIX = "collection-"
_COLLECTION_MARKER_NAME = "collection.json"
_COLLECTION_ROOT_ARTIFACT_NAMES = frozenset(
    {
        _COLLECTION_MARKER_NAME,
        "session_response.json",
        "scene_snapshot.json",
        "topology_inspection.json",
        "candidate_prim_properties.json",
    }
)
_ALLOWED_RENDER_QUALITIES = frozenset({"interactive", "inspection", "final"})
_MAX_DIRECTIONS = 8
_MAX_FOCUSED_RENDERS = 256
_MAX_RENDER_DIMENSION = 8192
_MAX_SNAPSHOT_PRIMS = 4096
_MAX_ORPHAN_COLLECTIONS = 64
_MAX_ORPHAN_TREE_ENTRIES = 2048
_MAX_EVIDENCE_ROOT_ENTRIES = 512
_RENDER_BUNDLE_ARTIFACT_NAMES = frozenset({"image.png", "response.json", "camera.json"})

_LOGGER = logging.getLogger(__name__)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_unique(field_name: str, values: tuple[str, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique values")


class ArticulationSceneEvidenceIdentity(_StrictFrozenModel):
    """Exact workflow and policy inputs covered by Scene evidence."""

    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asset: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_ids: tuple[str, ...] = ()
    collector_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_candidate_ids(self) -> Self:
        _require_unique("candidate_ids", self.candidate_ids)
        return self


class ArticulationSceneRenderEvidence(_StrictFrozenModel):
    """One focused Scene render and its downloaded provenance artifacts."""

    candidate_id: str = Field(min_length=1)
    focus_prim_path: str = Field(min_length=1)
    direction: str = Field(min_length=1)
    width: int = Field(ge=1, le=_MAX_RENDER_DIMENSION)
    height: int = Field(ge=1, le=_MAX_RENDER_DIMENSION)
    render_quality: Literal["interactive", "inspection", "final"]
    preview_scene_path: str = Field(min_length=1)
    renderer: str = Field(min_length=1)
    ovrtx_render_mode: str = Field(min_length=1)
    ovrtx_num_sensor_updates: int = Field(ge=1)
    active_aov: str = Field(min_length=1)
    render_bundle_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    image_artifact: ArtifactBinding
    response_artifact: ArtifactBinding
    camera_artifact: ArtifactBinding


class ArticulationSceneCandidateEvidence(_StrictFrozenModel):
    """Source-prim focus mapping and renders for one Stage 2 candidate."""

    candidate_id: str = Field(min_length=1)
    fixed_parent_prim: str | None = None
    moving_part_prims: tuple[str, ...] = ()
    inspected_prim_paths: tuple[str, ...] = ()
    renders: tuple[ArticulationSceneRenderEvidence, ...] = ()

    @model_validator(mode="after")
    def _validate_candidate_mapping(self) -> Self:
        _require_unique("moving_part_prims", self.moving_part_prims)
        _require_unique("inspected_prim_paths", self.inspected_prim_paths)
        expected_inspected = tuple(
            dict.fromkeys(
                (
                    *((self.fixed_parent_prim,) if self.fixed_parent_prim else ()),
                    *self.moving_part_prims,
                )
            )
        )
        if self.inspected_prim_paths != expected_inspected:
            raise ValueError(
                "inspected_prim_paths must exactly cover the candidate endpoints"
            )
        render_keys: list[tuple[str, str]] = []
        for render in self.renders:
            if render.candidate_id != self.candidate_id:
                raise ValueError("render candidate_id does not match its candidate")
            if render.focus_prim_path not in self.moving_part_prims:
                raise ValueError("render focus must be one of the moving part prims")
            render_keys.append((render.focus_prim_path, render.direction))
        if len(render_keys) != len(set(render_keys)):
            raise ValueError("candidate renders must have unique focus/direction pairs")
        rendered_focus_paths = {render.focus_prim_path for render in self.renders}
        if set(self.moving_part_prims) != rendered_focus_paths:
            raise ValueError(
                "candidate renders must cover every moving part prim at least once"
            )
        return self


class ArticulationSceneEvidenceResult(_StrictFrozenModel):
    """Typed manifest payload written by the durable articulation workflow."""

    schema_version: Literal[
        "content-agent-workflows.articulation-scene-evidence.v1"
    ] = ARTICULATION_SCENE_EVIDENCE_SCHEMA_VERSION
    identity: ArticulationSceneEvidenceIdentity
    scene_session_id: str = Field(min_length=1)
    scene_workspace_dir: str = Field(min_length=1)
    source_scene_path: str = Field(min_length=1)
    inspection_scene_path: str = Field(min_length=1)
    scene_source_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    inspected_prim_paths: tuple[str, ...] = Field(min_length=1)
    collection_id: str | None = Field(default=None, pattern=_COLLECTION_ID_PATTERN)
    collection_artifact: ArtifactBinding | None = None
    session_response_artifact: ArtifactBinding
    scene_snapshot_artifact: ArtifactBinding
    topology_inspection_artifact: ArtifactBinding
    prim_properties_artifact: ArtifactBinding
    candidates: tuple[ArticulationSceneCandidateEvidence, ...] = ()

    @model_validator(mode="after")
    def _validate_result_scope(self) -> Self:
        _require_unique("inspected_prim_paths", self.inspected_prim_paths)
        if (self.collection_id is None) != (self.collection_artifact is None):
            raise ValueError(
                "Scene collection ID and artifact must be provided together"
            )
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        _require_unique("candidate evidence IDs", candidate_ids)
        if candidate_ids != self.identity.candidate_ids:
            raise ValueError(
                "Scene candidate evidence must preserve exact candidate order"
            )
        inspected = set(self.inspected_prim_paths)
        for candidate in self.candidates:
            if not set(candidate.inspected_prim_paths).issubset(inspected):
                raise ValueError(
                    "candidate endpoint paths must be covered by inspected_prim_paths"
                )
            for render in candidate.renders:
                if (
                    self.collection_id is None
                    and render.render_bundle_sha256 is not None
                ):
                    raise ValueError(
                        "Content-addressed Scene renders require a collection ID"
                    )
                if (
                    self.collection_id is not None
                    and render.render_bundle_sha256 is None
                ):
                    raise ValueError(
                        "Collection-owned Scene renders require bundle digests"
                    )
        return self


class ArticulationSceneEvidenceCollector(Protocol):
    """Source-inspection boundary used by the durable articulation workflow."""

    def configuration_sha256(self, request: ArticulationWorkflowRequest) -> str:
        """Return the stable evidence-policy identity for one request."""

    def collect(
        self,
        request: ArticulationWorkflowRequest,
        *,
        request_sha256: str,
        source_sha256: str,
        source_dependency_bundle_sha256: str,
        candidate_document: Stage2CandidateDocument,
        candidate_document_sha256: str,
        output_dir: Path,
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationSceneEvidenceResult:
        """Capture evidence for the exact source and Stage 2 candidate document."""


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_identity(source_path: Path) -> tuple[str, str]:
    from world_understanding.functions.physics.joint_rigger import (
        identify_usd_artifact,
    )

    try:
        identity = identify_usd_artifact(
            source_path,
            uri=source_path.resolve().as_uri(),
        )
    except Exception as exc:
        raise ValueError(
            f"Cannot establish composed Scene source identity: {exc}"
        ) from exc
    dependency_sha256 = identity.dependency_bundle_sha256
    if dependency_sha256 is None:
        raise ValueError(
            "Composed Scene source identity lacks a dependency bundle digest"
        )
    return identity.root_sha256, dependency_sha256


def _require_source_identity(
    source_path: Path,
    *,
    source_sha256: str,
    source_dependency_bundle_sha256: str,
) -> None:
    observed_source, observed_dependencies = _source_identity(source_path)
    if observed_source != source_sha256:
        raise ValueError("Scene evidence source digest changed before use")
    if observed_dependencies != source_dependency_bundle_sha256:
        raise ValueError("Scene evidence source dependency bundle changed before use")


def _scene_source_digest(source_path: Path) -> str:
    from usd_core.physics_topology import source_digest

    try:
        return source_digest(source_path)
    except Exception as exc:
        raise ValueError(f"Cannot establish usd-cli source digest: {exc}") from exc


def _receipt_checkpoint_sha256_for_retry(evidence_root: Path) -> str | None:
    """Return the sealed receipt checkpoint when a prior attempt left a journal."""

    journal = evidence_root / "raw" / "usd_cli_command_receipts.jsonl"
    if not os.path.lexists(journal):
        return None
    checkpoint = evidence_root / "raw" / "usd_cli_command_receipts.checkpoint.json"
    try:
        return read_contained_artifact(
            evidence_root,
            checkpoint,
            max_bytes=64 * 1024,
            parse_json=True,
        ).sha256
    except ValueError as exc:
        raise ValueError(
            "Prior articulation evidence attempt left a receipt journal without "
            "a valid sealed checkpoint"
        ) from exc


def _artifact_binding(path: Path) -> ArtifactBinding:
    resolved = path.expanduser().resolve()
    return ArtifactBinding(path=str(resolved), sha256=file_sha256(resolved))


def _require_evidence_root_for_write(evidence_root: Path) -> Path:
    raw_path = evidence_root.expanduser()
    if not raw_path.is_absolute():
        raise ValueError("Scene evidence directory must be an absolute path")
    if raw_path.is_symlink():
        raise ValueError("Scene evidence directory must not be a symbolic link")
    resolved_path = raw_path.resolve()
    if resolved_path != raw_path:
        raise ValueError(
            "Scene evidence directory must resolve without traversing symlinks"
        )
    if raw_path.exists() and not raw_path.is_dir():
        raise ValueError("Scene evidence path exists but is not a directory")
    return raw_path


def _fsync_directory_entry(directory: Path, *, label: str) -> None:
    raw_directory = directory.expanduser()
    if not raw_directory.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    if raw_directory.is_symlink() or raw_directory.resolve() != raw_directory:
        raise ValueError(f"{label} path must not traverse symbolic links")
    parent = raw_directory.parent
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(parent, directory_flags)
        try:
            directory_fd = os.open(
                raw_directory.name,
                directory_flags,
                dir_fd=parent_fd,
            )
        except BaseException:
            os.close(parent_fd)
            raise
    except OSError as exc:
        raise ValueError(f"Cannot open {label} for durability: {exc}") from exc
    try:
        parent_metadata = os.fstat(parent_fd)
        directory_metadata = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_metadata.st_mode):
            raise ValueError(f"{label} is not a directory")
        os.fsync(directory_fd)
        os.fsync(parent_fd)
        current_parent_fd = -1
        current_directory_fd = -1
        try:
            try:
                current_parent_fd = os.open(parent, directory_flags)
                current_directory_fd = os.open(raw_directory, directory_flags)
            except OSError as exc:
                raise ValueError(f"Cannot re-open {label} after fsync: {exc}") from exc
            current_parent_metadata = os.fstat(current_parent_fd)
            current_directory_metadata = os.fstat(current_directory_fd)
        finally:
            if current_directory_fd >= 0:
                os.close(current_directory_fd)
            if current_parent_fd >= 0:
                os.close(current_parent_fd)
        if (
            current_parent_metadata.st_dev,
            current_parent_metadata.st_ino,
        ) != (
            parent_metadata.st_dev,
            parent_metadata.st_ino,
        ) or (
            current_directory_metadata.st_dev,
            current_directory_metadata.st_ino,
        ) != (
            directory_metadata.st_dev,
            directory_metadata.st_ino,
        ):
            raise ValueError(f"{label} changed during durability confirmation")
    except OSError as exc:
        raise ValueError(f"Cannot fsync {label}: {exc}") from exc
    finally:
        os.close(directory_fd)
        os.close(parent_fd)


def _prepare_evidence_root_for_write(output_dir: Path) -> Path:
    evidence_root = _require_evidence_root_for_write(output_dir)
    missing_directories: list[Path] = []
    current = evidence_root
    while not current.exists():
        if current.is_symlink():
            raise ValueError(
                "Scene evidence directory path must not traverse symbolic links"
            )
        missing_directories.append(current)
        if current.parent == current:
            raise ValueError("Cannot find an existing Scene evidence ancestor")
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise ValueError("Scene evidence ancestor is not a private directory")
    evidence_root.mkdir(parents=True, exist_ok=True)
    for created_directory in missing_directories:
        _fsync_directory_entry(
            created_directory,
            label="Scene evidence directory",
        )
    evidence_root = _require_evidence_root_for_write(evidence_root)
    if not evidence_root.is_dir():
        raise ValueError("Scene evidence directory is missing")
    return evidence_root


def _require_contained_write_path(
    evidence_root: Path,
    path: Path,
    *,
    label: str,
) -> Path:
    root = _require_evidence_root_for_write(evidence_root)
    if not root.is_dir():
        raise ValueError("Scene evidence directory is missing")

    raw_path = path.expanduser()
    if not raw_path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    try:
        raw_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the Scene evidence directory") from exc
    if raw_path == root:
        raise ValueError(f"{label} must be nested below the evidence directory")
    if raw_path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")

    resolved_path = raw_path.resolve()
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the Scene evidence directory") from exc
    if resolved_path != raw_path:
        raise ValueError(f"{label} path must not traverse symbolic links")
    return raw_path


def _prepare_contained_directory(
    evidence_root: Path,
    path: Path,
    *,
    label: str,
) -> Path:
    directory = _require_contained_write_path(
        evidence_root,
        path,
        label=label,
    )
    parent = directory.parent
    if parent != evidence_root:
        parent = _require_contained_write_path(
            evidence_root,
            parent,
            label=f"{label} parent directory",
        )
        if not parent.is_dir():
            raise ValueError(f"{label} parent directory is missing")
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"{label} exists but is not a directory")
    directory.mkdir(exist_ok=True)
    directory = _require_contained_write_path(
        evidence_root,
        directory,
        label=label,
    )
    if not directory.is_dir():
        raise ValueError(f"{label} is missing")
    return directory


def _prepare_artifact_write_path(
    evidence_root: Path,
    path: Path,
    *,
    label: str,
) -> Path:
    artifact_path = _require_contained_write_path(
        evidence_root,
        path,
        label=label,
    )
    parent = artifact_path.parent
    if parent != evidence_root:
        parent = _require_contained_write_path(
            evidence_root,
            parent,
            label=f"{label} parent directory",
        )
    if not parent.is_dir():
        raise ValueError(f"{label} parent directory is missing")
    if artifact_path.exists() and not artifact_path.is_file():
        raise ValueError(f"{label} exists but is not a file")
    if artifact_path.exists():
        try:
            link_count = artifact_path.stat(follow_symlinks=False).st_nlink
        except OSError as exc:
            raise ValueError(f"Cannot inspect {label}: {exc}") from exc
        if link_count != 1:
            raise ValueError(f"{label} must have exactly one hard link")
    return artifact_path


def _write_evidence_json(
    evidence_root: Path,
    path: Path,
    payload: dict[str, Any],
    *,
    label: str,
) -> Path:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    return _write_evidence_bytes(
        evidence_root,
        path,
        encoded,
        label=label,
    )


def _verify_private_artifact_payload_fd(
    parent_fd: int,
    name: str,
    *,
    expected_payload: bytes,
    label: str,
    expected_signature: tuple[int, ...] | None = None,
) -> os.stat_result:
    """Re-read one private artifact without introducing another fsync window."""

    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        artifact_fd = os.open(name, read_flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"Cannot open {label} for final verification: {exc}") from exc
    try:
        before_metadata = os.fstat(artifact_fd)
        before_signature = _private_file_metadata_signature(
            before_metadata,
            label=label,
        )
        if before_metadata.st_size != len(expected_payload):
            raise ValueError(f"{label} has conflicting bytes")
        os.lseek(artifact_fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(artifact_fd), "rb") as stream:
            observed_payload = stream.read(len(expected_payload) + 1)
        after_signature = _private_file_metadata_signature(
            os.fstat(artifact_fd),
            label=label,
        )
    finally:
        os.close(artifact_fd)
    try:
        current_signature = _private_file_metadata_signature(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False),
            label=label,
        )
    except OSError as exc:
        raise ValueError(f"Cannot re-inspect {label}: {exc}") from exc
    if (
        before_signature != after_signature
        or before_signature != current_signature
        or (expected_signature is not None and before_signature != expected_signature)
    ):
        raise ValueError(f"{label} changed during final verification")
    if observed_payload != expected_payload:
        raise ValueError(f"{label} has conflicting bytes")
    return before_metadata


def _write_evidence_bytes(
    evidence_root: Path,
    path: Path,
    payload: bytes,
    *,
    label: str,
) -> Path:
    artifact_path = _prepare_artifact_write_path(
        evidence_root,
        path,
        label=label,
    )
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(artifact_path.parent, directory_flags)
    except OSError as exc:
        raise ValueError(f"Cannot open {label} parent directory: {exc}") from exc

    def require_current_parent_binding() -> None:
        try:
            current_directory_fd = os.open(artifact_path.parent, directory_flags)
        except OSError as exc:
            raise ValueError(f"Cannot re-open {label} parent directory: {exc}") from exc
        try:
            opened = os.fstat(directory_fd)
            current = os.fstat(current_directory_fd)
        finally:
            os.close(current_directory_fd)
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError(f"{label} parent directory changed during publication")

    temporary_name = f".{artifact_path.name}.{uuid.uuid4().hex}.tmp"
    temporary_fd = -1
    temporary_created = False
    temporary_metadata: os.stat_result | None = None
    previous_metadata: os.stat_result | None = None
    destination_published = False
    publication_complete = False
    try:
        require_current_parent_binding()
        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            existing_fd = os.open(
                artifact_path.name,
                read_flags,
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            existing_fd = -1
        if existing_fd >= 0:
            try:
                existing_metadata = os.fstat(existing_fd)
                existing_signature = _private_file_metadata_signature(
                    existing_metadata,
                    label=label,
                )
                if existing_metadata.st_size != len(payload):
                    raise ValueError(f"{label} already exists with different bytes")
                os.lseek(existing_fd, 0, os.SEEK_SET)
                with os.fdopen(os.dup(existing_fd), "rb") as existing_stream:
                    existing_payload = existing_stream.read(len(payload) + 1)
                os.fsync(existing_fd)
                after_fsync_signature = _private_file_metadata_signature(
                    os.fstat(existing_fd),
                    label=label,
                )
                os.lseek(existing_fd, 0, os.SEEK_SET)
                with os.fdopen(os.dup(existing_fd), "rb") as final_stream:
                    final_payload = final_stream.read(len(payload) + 1)
                final_existing_signature = _private_file_metadata_signature(
                    os.fstat(existing_fd),
                    label=label,
                )
            finally:
                os.close(existing_fd)
            current_existing_signature = _private_file_metadata_signature(
                os.stat(
                    artifact_path.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                ),
                label=label,
            )
            if (
                after_fsync_signature != existing_signature
                or final_existing_signature != existing_signature
                or current_existing_signature != existing_signature
            ):
                raise ValueError(f"{label} changed while being verified")
            if existing_payload != payload or final_payload != payload:
                raise ValueError(f"{label} already exists with different bytes")
            os.fsync(directory_fd)
            require_current_parent_binding()
            _verify_private_artifact_payload_fd(
                directory_fd,
                artifact_path.name,
                expected_payload=payload,
                label=label,
                expected_signature=existing_signature,
            )
            require_current_parent_binding()
            return artifact_path

        write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        write_flags |= getattr(os, "O_NOFOLLOW", 0)
        temporary_fd = os.open(
            temporary_name,
            write_flags,
            0o600,
            dir_fd=directory_fd,
        )
        temporary_created = True
        temporary_metadata = os.fstat(temporary_fd)
        if (
            not stat.S_ISREG(temporary_metadata.st_mode)
            or temporary_metadata.st_nlink != 1
        ):
            raise ValueError(f"{label} temporary file is not private")
        with os.fdopen(temporary_fd, "wb") as stream:
            temporary_fd = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            written_metadata = os.fstat(stream.fileno())
            if (
                (
                    written_metadata.st_dev,
                    written_metadata.st_ino,
                )
                != (
                    temporary_metadata.st_dev,
                    temporary_metadata.st_ino,
                )
                or not stat.S_ISREG(written_metadata.st_mode)
                or written_metadata.st_nlink != 1
            ):
                raise ValueError(f"{label} temporary file is not private")

        # Recheck the destination immediately before publishing. A hard link
        # created after this check is still safe: replace changes the directory
        # entry atomically instead of truncating the linked inode.
        artifact_path = _prepare_artifact_write_path(
            evidence_root,
            artifact_path,
            label=label,
        )
        require_current_parent_binding()
        try:
            previous_metadata = os.stat(
                artifact_path.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            previous_metadata = None
        if previous_metadata is not None:
            if (
                not stat.S_ISREG(previous_metadata.st_mode)
                or previous_metadata.st_nlink != 1
            ):
                raise ValueError(f"{label} must be a private regular file")
            raise ValueError(f"{label} appeared during publication")
        os.replace(
            temporary_name,
            artifact_path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_created = False
        destination_published = True
        require_current_parent_binding()
        published_metadata = os.stat(
            artifact_path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if temporary_metadata is None or (
            published_metadata.st_dev,
            published_metadata.st_ino,
        ) != (
            temporary_metadata.st_dev,
            temporary_metadata.st_ino,
        ):
            raise ValueError(f"{label} publication changed the staged file identity")
        if (
            not stat.S_ISREG(published_metadata.st_mode)
            or published_metadata.st_nlink != 1
        ):
            raise ValueError(f"{label} published file is not private")
        published_signature = _private_file_metadata_signature(
            published_metadata,
            label=label,
        )
        # The atomic replacement is the commit point. Once the staged inode is
        # bound at the destination, preserve it even if a later durability
        # confirmation fails; a crash can then expose only the complete old or
        # complete new file, never a rollback hard-link window.
        publication_complete = True
        os.fsync(directory_fd)
        require_current_parent_binding()
        artifact_path = _prepare_artifact_write_path(
            evidence_root,
            artifact_path,
            label=label,
        )
        require_current_parent_binding()
        current_metadata = _verify_private_artifact_payload_fd(
            directory_fd,
            artifact_path.name,
            expected_payload=payload,
            label=label,
            expected_signature=published_signature,
        )
        if (
            current_metadata.st_dev,
            current_metadata.st_ino,
        ) != (
            temporary_metadata.st_dev,
            temporary_metadata.st_ino,
        ):
            raise ValueError(f"{label} path changed during publication")
        require_current_parent_binding()
    except OSError as exc:
        raise ValueError(f"Cannot write {label}: {exc}") from exc
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_created:
            try:
                cleanup_metadata = os.stat(
                    temporary_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if temporary_metadata is not None and (
                    cleanup_metadata.st_dev,
                    cleanup_metadata.st_ino,
                ) == (
                    temporary_metadata.st_dev,
                    temporary_metadata.st_ino,
                ):
                    os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            except OSError:
                _LOGGER.exception("Failed to clean up private %s staging file", label)
        if destination_published and not publication_complete:
            try:
                destination_metadata: os.stat_result | None
                try:
                    destination_metadata = os.stat(
                        artifact_path.name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    destination_metadata = None
                destination_is_published = (
                    destination_metadata is not None
                    and temporary_metadata is not None
                    and (
                        destination_metadata.st_dev,
                        destination_metadata.st_ino,
                    )
                    == (
                        temporary_metadata.st_dev,
                        temporary_metadata.st_ino,
                    )
                )
                if destination_is_published:
                    os.unlink(artifact_path.name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
            except FileNotFoundError:
                pass
            except (OSError, ValueError):
                _LOGGER.exception(
                    "Failed to roll back incomplete %s publication",
                    label,
                )
        os.close(directory_fd)

    return artifact_path


def _collection_name(collection_id: str) -> str:
    if len(collection_id) != 32 or any(
        character not in "0123456789abcdef" for character in collection_id
    ):
        raise ValueError("Scene evidence collection ID is invalid")
    return f"{_COLLECTION_PREFIX}{collection_id}"


def _collection_marker_payload(collection_id: str) -> dict[str, str]:
    return {
        "schema_version": _COLLECTION_SCHEMA_VERSION,
        "collection_id": collection_id,
    }


def _create_collection_directory(
    evidence_root: Path,
) -> tuple[str, Path, ArtifactBinding]:
    """Create one private collection root and bind it to a random identity."""

    resolved_evidence_root = _require_evidence_root_for_write(evidence_root)
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        evidence_fd = os.open(resolved_evidence_root, directory_flags)
    except OSError as exc:
        raise ValueError(f"Cannot open Scene evidence directory: {exc}") from exc

    collection_id: str | None = None
    collection_name: str | None = None
    collection_created = False
    collection_fd = -1
    collection_metadata: os.stat_result | None = None
    try:
        for _ in range(4):
            candidate_id = uuid.uuid4().hex
            candidate_name = _collection_name(candidate_id)
            try:
                os.mkdir(candidate_name, mode=0o700, dir_fd=evidence_fd)
            except FileExistsError:
                continue
            collection_id = candidate_id
            collection_name = candidate_name
            collection_created = True
            break
        if collection_id is None or collection_name is None:
            raise ValueError("Cannot allocate a unique Scene collection ID")

        collection_metadata = os.stat(
            collection_name,
            dir_fd=evidence_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(collection_metadata.st_mode):
            raise ValueError("Scene collection root is not a directory")
        collection_fd = os.open(
            collection_name,
            directory_flags,
            dir_fd=evidence_fd,
        )
        opened_collection_metadata = os.fstat(collection_fd)
        if (
            opened_collection_metadata.st_dev,
            opened_collection_metadata.st_ino,
        ) != (
            collection_metadata.st_dev,
            collection_metadata.st_ino,
        ):
            raise ValueError("Scene collection root changed during initialization")
        os.fsync(collection_fd)
        os.fsync(evidence_fd)

        collection_path = _require_contained_write_path(
            resolved_evidence_root,
            resolved_evidence_root / collection_name,
            label="Scene collection directory",
        )
        if not collection_path.is_dir():
            raise ValueError("Scene collection directory is missing")
        marker_path = _write_evidence_json(
            resolved_evidence_root,
            collection_path / _COLLECTION_MARKER_NAME,
            _collection_marker_payload(collection_id),
            label="Scene collection marker",
        )
        current_collection_fd = os.open(
            collection_name,
            directory_flags,
            dir_fd=evidence_fd,
        )
        try:
            current_collection_metadata = os.fstat(current_collection_fd)
        finally:
            os.close(current_collection_fd)
        if (
            current_collection_metadata.st_dev,
            current_collection_metadata.st_ino,
        ) != (
            collection_metadata.st_dev,
            collection_metadata.st_ino,
        ):
            raise ValueError("Scene collection root changed during initialization")
        marker_binding = _artifact_binding(marker_path)
        final_collection_fd = os.open(
            collection_name,
            directory_flags,
            dir_fd=evidence_fd,
        )
        try:
            final_collection_metadata = os.fstat(final_collection_fd)
        finally:
            os.close(final_collection_fd)
        if (
            final_collection_metadata.st_dev,
            final_collection_metadata.st_ino,
        ) != (
            collection_metadata.st_dev,
            collection_metadata.st_ino,
        ):
            raise ValueError("Scene collection root changed during initialization")
        return collection_id, collection_path, marker_binding
    except BaseException:
        if (
            collection_created
            and collection_name is not None
            and collection_metadata is not None
        ):
            try:
                current_collection_metadata = os.stat(
                    collection_name,
                    dir_fd=evidence_fd,
                    follow_symlinks=False,
                )
                if (
                    current_collection_metadata.st_dev,
                    current_collection_metadata.st_ino,
                ) != (
                    collection_metadata.st_dev,
                    collection_metadata.st_ino,
                ):
                    raise ValueError(
                        "Refusing to clean up a replaced Scene collection root"
                    )
                if collection_fd >= 0:
                    for name in os.listdir(collection_fd):
                        metadata = os.stat(
                            name,
                            dir_fd=collection_fd,
                            follow_symlinks=False,
                        )
                        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                            os.unlink(name, dir_fd=collection_fd)
                    os.fsync(collection_fd)
                os.rmdir(collection_name, dir_fd=evidence_fd)
                os.fsync(evidence_fd)
            except (OSError, ValueError):
                _LOGGER.exception(
                    "Failed to clean up uninitialized Scene collection %s",
                    collection_name,
                )
        raise
    finally:
        if collection_fd >= 0:
            os.close(collection_fd)
        os.close(evidence_fd)


def _require_downloaded_artifact(
    evidence_root: Path,
    record: dict[str, Any],
    *,
    field_name: str,
    expected_path: Path,
    label: str,
) -> Path:
    raw_value = record.get(field_name)
    if not isinstance(raw_value, str) or not raw_value:
        raise ValueError(f"{label} path is missing from the download record")
    raw_path = Path(raw_value).expanduser()
    if raw_path != expected_path:
        raise ValueError(f"{label} download path does not match the requested path")
    artifact_path = _prepare_artifact_write_path(
        evidence_root,
        raw_path,
        label=label,
    )
    if not artifact_path.is_file():
        raise ValueError(f"{label} download is missing")
    return artifact_path


def _require_string(payload: dict[str, Any], field_name: str, *, label: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is missing {field_name}")
    return value


def _require_finite_matrix4(
    value: object,
    *,
    label: str,
) -> list[list[float]]:
    """Validate and normalize one USD camera transform for durable evidence."""

    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} must be a 4x4 matrix")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"{label} must be a 4x4 matrix")
        normalized_row: list[float] = []
        for entry in row:
            if (
                not isinstance(entry, int | float)
                or isinstance(entry, bool)
                or not math.isfinite(entry)
            ):
                raise ValueError(f"{label} must contain only finite numbers")
            normalized_row.append(float(entry))
        matrix.append(normalized_row)
    return matrix


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _candidate_endpoint_paths(
    candidate_document: Stage2CandidateDocument,
) -> tuple[str, ...]:
    paths: list[str] = []
    for candidate in candidate_document.candidates:
        if candidate.fixed_parent_prim:
            paths.append(candidate.fixed_parent_prim)
        paths.extend(candidate.moving_part_prims)
    return tuple(dict.fromkeys(paths))


def _artifact_bindings(
    result: ArticulationSceneEvidenceResult,
) -> tuple[tuple[str, ArtifactBinding], ...]:
    bindings: list[tuple[str, ArtifactBinding]] = []
    if result.collection_artifact is not None:
        bindings.append(("collection marker", result.collection_artifact))
    bindings.extend(
        [
            ("session response", result.session_response_artifact),
            ("scene snapshot", result.scene_snapshot_artifact),
            ("topology inspection", result.topology_inspection_artifact),
            ("prim properties", result.prim_properties_artifact),
        ]
    )
    for candidate in result.candidates:
        for index, render in enumerate(candidate.renders):
            prefix = f"candidate {candidate.candidate_id} render {index}"
            bindings.extend(
                (
                    (f"{prefix} image", render.image_artifact),
                    (f"{prefix} response", render.response_artifact),
                    (f"{prefix} camera", render.camera_artifact),
                )
            )
    return tuple(bindings)


def articulation_scene_artifact_bindings(
    result: ArticulationSceneEvidenceResult,
) -> tuple[ArtifactBinding, ...]:
    """Return every artifact transitively bound by a Scene manifest."""

    return tuple(binding for _label, binding in _artifact_bindings(result))


def _verify_artifact_binding(
    binding: ArtifactBinding,
    *,
    label: str,
    evidence_root: Path,
) -> Path:
    path, _ = _read_and_verify_artifact_binding(
        binding,
        label=label,
        evidence_root=evidence_root,
        capture_bytes=False,
    )
    return path


def _read_verified_artifact_binding_bytes(
    binding: ArtifactBinding,
    *,
    label: str,
    evidence_root: Path,
) -> tuple[Path, bytes]:
    path, payload = _read_and_verify_artifact_binding(
        binding,
        label=label,
        evidence_root=evidence_root,
        capture_bytes=True,
    )
    assert payload is not None
    return path, payload


def _read_and_verify_artifact_binding(
    binding: ArtifactBinding,
    *,
    label: str,
    evidence_root: Path,
    capture_bytes: bool,
) -> tuple[Path, bytes | None]:
    raw_path = Path(binding.path).expanduser()
    if not raw_path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    if raw_path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    path = raw_path.resolve()
    try:
        relative_path = path.relative_to(evidence_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the Scene evidence directory") from exc
    file_descriptor = -1
    directory_fds: list[int] = []
    try:
        directory_flags = os.O_RDONLY
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        directory_fds.append(os.open(evidence_root, directory_flags))
        for component in relative_path.parts[:-1]:
            directory_fds.append(
                os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_fds[-1],
                )
            )
        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(
            relative_path.name,
            read_flags,
            dir_fd=directory_fds[-1],
        )
        opened_metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        if opened_metadata.st_nlink != 1:
            raise ValueError(f"{label} must have exactly one hard link")
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if capture_bytes else None
        with os.fdopen(os.dup(file_descriptor), "rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
        final_descriptor_metadata = os.fstat(file_descriptor)
        current_path_metadata = os.stat(
            relative_path.name,
            dir_fd=directory_fds[-1],
            follow_symlinks=False,
        )
        metadata_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(final_descriptor_metadata, field) != getattr(opened_metadata, field)
            or getattr(current_path_metadata, field) != getattr(opened_metadata, field)
            for field in metadata_fields
        ):
            raise ValueError(f"{label} changed while being read")
        observed_sha256 = digest.hexdigest()
    except OSError as exc:
        raise ValueError(f"Cannot read {label}: {exc}") from exc
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
    if observed_sha256 != binding.sha256:
        raise ValueError(
            f"{label} digest mismatch: expected {binding.sha256}, got {observed_sha256}"
        )
    payload = b"".join(chunks) if chunks is not None else None
    return path, payload


def _validate_snapshot_payload(
    payload: dict[str, Any],
    *,
    expected_session_id: str,
    expected_source_scene_path: str | Path,
    expected_inspection_scene_path: str | Path,
) -> tuple[str, tuple[str, ...]]:
    """Validate the complete snapshot scope before any focused rendering."""

    if payload.get("session_id") != expected_session_id:
        raise ValueError("Scene snapshot session identity does not match")
    if not _same_path(
        _require_string(payload, "source_scene_path", label="scene snapshot"),
        expected_source_scene_path,
    ):
        raise ValueError("Scene snapshot source path does not match")
    if not _same_path(
        _require_string(payload, "inspection_scene_path", label="scene snapshot"),
        expected_inspection_scene_path,
    ):
        raise ValueError("Scene snapshot inspection path does not match")
    summary = payload.get("summary")
    if not isinstance(summary, dict) or summary.get("truncated") is not False:
        raise ValueError("Scene snapshot must explicitly report untruncated evidence")
    raw_paths = payload.get("paths")
    if not isinstance(raw_paths, list) or not all(
        isinstance(item, str) for item in raw_paths
    ):
        raise ValueError("Scene snapshot evidence must contain string paths")
    root_prim_path = _require_string(
        payload,
        "root_prim_path",
        label="scene snapshot",
    )
    return root_prim_path, tuple(raw_paths)


def _verify_snapshot(
    result: ArticulationSceneEvidenceResult,
    payload: bytes,
) -> tuple[str, tuple[str, ...]]:
    return _validate_snapshot_payload(
        _load_json_bytes(payload, label="scene snapshot"),
        expected_session_id=result.scene_session_id,
        expected_source_scene_path=result.source_scene_path,
        expected_inspection_scene_path=result.inspection_scene_path,
    )


def _validate_workspace_path(
    value: str,
    *,
    expected_session_id: str,
) -> PurePosixPath:
    workspace = PurePosixPath(value)
    if (
        not workspace.is_absolute()
        or ".." in workspace.parts
        or workspace.name != expected_session_id
    ):
        raise ValueError(
            "Scene session workspace is not an absolute session-bound path"
        )
    return workspace


def _validate_session_response_payload(
    payload: dict[str, Any],
    *,
    expected_session_id: str,
    expected_source_scene_path: str | Path,
    expected_inspection_scene_path: str | Path,
    expected_workspace_dir: str | None = None,
) -> str:
    """Validate the public session identity that anchors generated previews."""

    if payload.get("session_id") != expected_session_id:
        raise ValueError("Scene session response identity does not match")
    if payload.get("status") != "ready":
        raise ValueError("Scene session response did not report ready")
    if not _same_path(
        _require_string(
            payload,
            "source_scene_path",
            label="Scene session response",
        ),
        expected_source_scene_path,
    ):
        raise ValueError("Scene session response source path does not match")
    if not _same_path(
        _require_string(
            payload,
            "inspection_scene_path",
            label="Scene session response",
        ),
        expected_inspection_scene_path,
    ):
        raise ValueError("Scene session response inspection path does not match")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Scene session response is missing artifacts")
    workspace_value = _require_string(
        artifacts,
        "workspace_dir",
        label="Scene session artifacts",
    )
    workspace = _validate_workspace_path(
        workspace_value,
        expected_session_id=expected_session_id,
    )
    if expected_workspace_dir is not None and workspace != PurePosixPath(
        expected_workspace_dir
    ):
        raise ValueError("Scene session workspace does not match the manifest")
    return str(workspace)


def _verify_session_response(
    result: ArticulationSceneEvidenceResult,
    payload: bytes,
) -> None:
    _validate_session_response_payload(
        _load_json_bytes(payload, label="Scene session response"),
        expected_session_id=result.scene_session_id,
        expected_source_scene_path=result.source_scene_path,
        expected_inspection_scene_path=result.inspection_scene_path,
        expected_workspace_dir=result.scene_workspace_dir,
    )


def _validate_topology_payload(
    payload: dict[str, Any],
    *,
    expected_asset: str | Path,
    expected_source_digest: str,
) -> None:
    """Require the exact full-source topology response contract."""

    def require_list(
        field_name: str,
        item_type: type[Any],
    ) -> list[Any]:
        value = payload.get(field_name)
        if not isinstance(value, list) or not all(
            isinstance(item, item_type) for item in value
        ):
            raise ValueError(
                f"Topology inspection {field_name} must be a list of "
                f"{item_type.__name__} values"
            )
        return value

    def require_count(field_name: str, expected_count: int) -> None:
        value = payload.get(field_name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value != expected_count
        ):
            raise ValueError(
                f"Topology inspection {field_name} must match its inventory"
            )

    def require_string_list(
        record: dict[str, Any],
        field_name: str,
        *,
        record_label: str,
    ) -> None:
        value = record.get(field_name)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(
                f"Topology inspection {record_label} {field_name} "
                "must be a list of strings"
            )

    if payload.get("schema_version") != "usd-cli.physics-topology.v1":
        raise ValueError("Topology inspection schema version does not match")
    if payload.get("path_space") != "source":
        raise ValueError("Topology inspection path space must be source")
    if "root_prim_path" not in payload or payload.get("root_prim_path") is not None:
        raise ValueError("Topology inspection must be explicitly unscoped")
    if payload.get("source_digest") != expected_source_digest:
        raise ValueError("Topology inspection source digest does not match")
    if not _same_path(
        _require_string(payload, "asset", label="topology inspection"),
        expected_asset,
    ):
        raise ValueError("Topology inspection asset path does not match")

    rigid_body_paths = require_list("rigid_body_paths", str)
    colliders = require_list("colliders", dict)
    joints = require_list("joints", dict)
    require_list("articulation_root_paths", str)
    fixed_to_world_joints = require_list("fixed_to_world_joints", dict)
    findings = require_list("findings", dict)
    require_count("enabled_rigid_body_count", len(rigid_body_paths))
    require_count("enabled_collider_count", len(colliders))

    for collider in colliders:
        if (
            not isinstance(collider.get("prim_path"), str)
            or not isinstance(collider.get("type_name"), str)
            or (
                collider.get("owner_rigid_body_path") is not None
                and not isinstance(collider.get("owner_rigid_body_path"), str)
            )
        ):
            raise ValueError(
                "Topology inspection collider record has invalid field types"
            )
    for record_label, records in (
        ("joint", joints),
        ("fixed_to_world_joint", fixed_to_world_joints),
    ):
        for joint in records:
            if (
                not isinstance(joint.get("prim_path"), str)
                or not isinstance(joint.get("joint_type"), str)
                or not isinstance(joint.get("is_fixed_joint"), bool)
                or not isinstance(joint.get("enabled"), bool)
            ):
                raise ValueError(
                    f"Topology inspection {record_label} record has invalid field types"
                )
            for field_name in (
                "body0_targets",
                "body1_targets",
                "body0_rigid_body_paths",
                "body1_rigid_body_paths",
            ):
                require_string_list(
                    joint,
                    field_name,
                    record_label=record_label,
                )
    for finding in findings:
        if not isinstance(finding.get("code"), str) or not isinstance(
            finding.get("prim_path"), str
        ):
            raise ValueError(
                "Topology inspection finding record has invalid field types"
            )
        require_string_list(
            finding,
            "related_paths",
            record_label="finding",
        )


def _verify_topology(
    result: ArticulationSceneEvidenceResult,
    payload: bytes,
) -> None:
    _validate_topology_payload(
        _load_json_bytes(payload, label="topology inspection"),
        expected_asset=result.identity.source_asset,
        expected_source_digest=result.scene_source_digest,
    )


def _validate_properties_payload(
    payload: dict[str, Any],
    *,
    expected_session_id: str,
    expected_prim_paths: tuple[str, ...],
) -> None:
    """Require complete properties for the exact requested prim sequence."""

    if payload.get("session_id") != expected_session_id:
        raise ValueError("Prim properties session identity does not match")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list) or not all(
        isinstance(item, dict) for item in raw_results
    ):
        raise ValueError("Prim properties evidence must contain object results")
    if any(item.get("truncated") is not False for item in raw_results):
        raise ValueError(
            "Prim properties evidence must explicitly report untruncated results"
        )
    observed_paths = tuple(item.get("prim_path") for item in raw_results)
    if observed_paths != expected_prim_paths:
        raise ValueError(
            "Prim properties evidence does not exactly cover inspected_prim_paths"
        )
    for item in raw_results:
        properties = item.get("properties")
        if not isinstance(properties, dict):
            raise ValueError(
                "Prim properties evidence rows must contain a properties mapping"
            )
        if properties.get("path") != item.get("prim_path"):
            raise ValueError(
                "Prim properties evidence path does not match its requested prim"
            )
        for field_name in ("name", "type_name"):
            if not isinstance(properties.get(field_name), str):
                raise ValueError(
                    f"Prim properties evidence {field_name} must be a string"
                )
        for field_name in ("active", "loaded"):
            if not isinstance(properties.get(field_name), bool):
                raise ValueError(
                    f"Prim properties evidence {field_name} must be a boolean"
                )
        for field_name in ("metadata", "attributes", "relationships"):
            if not isinstance(properties.get(field_name), dict):
                raise ValueError(
                    f"Prim properties evidence {field_name} must be a mapping"
                )
        bounds = properties.get("bounds")
        if "bounds" not in properties or (
            bounds is not None and not isinstance(bounds, dict)
        ):
            raise ValueError(
                "Prim properties evidence bounds must be a mapping or null"
            )


def _verify_properties(
    result: ArticulationSceneEvidenceResult,
    payload: bytes,
) -> None:
    _validate_properties_payload(
        _load_json_bytes(payload, label="prim properties"),
        expected_session_id=result.scene_session_id,
        expected_prim_paths=result.inspected_prim_paths,
    )


def _validate_render_image(
    path: Path,
    *,
    expected_width: int,
    expected_height: int,
) -> None:
    """Decode one review render and require its bound PNG dimensions."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Scene render image is not decodable: {path}") from exc
    _validate_render_image_payload(
        payload,
        source_label=str(path),
        expected_width=expected_width,
        expected_height=expected_height,
    )


def _validate_render_image_payload(
    payload: bytes,
    *,
    source_label: str,
    expected_width: int,
    expected_height: int,
) -> None:
    """Validate immutable PNG bytes before they can become durable evidence."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                image_format = image.format
                image_size = image.size
                if image_format != "PNG":
                    raise ValueError(
                        f"Scene render image format must be PNG, got {image_format!r}"
                    )
                expected_size = (expected_width, expected_height)
                if image_size != expected_size:
                    raise ValueError(
                        "Scene render image dimensions do not match: "
                        f"expected {expected_size}, got {image_size}"
                    )
                image.verify()
        # ``verify`` checks the container structure without decoding pixels.
        # Reopen and force a full decode so a CRC-valid but broken IDAT stream
        # cannot become durable review evidence.
        with Image.open(BytesIO(payload)) as image:
            image.load()
    except ValueError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
        UnidentifiedImageError,
    ) as exc:
        raise ValueError(
            f"Scene render image is not decodable: {source_label}"
        ) from exc


def _validate_render_response_scope(
    response: dict[str, Any],
    *,
    expected_session_id: str,
    expected_workspace_dir: str,
) -> str:
    """Bind one generated preview and its artifacts to the requested session."""

    preview_scene_path = PurePosixPath(
        _require_string(
            response,
            "preview_scene_path",
            label="Scene render response",
        )
    )
    image_path = PurePosixPath(
        _require_string(
            response,
            "image_path",
            label="Scene render response",
        )
    )
    camera_path = PurePosixPath(
        _require_string(
            response,
            "camera_json_path",
            label="Scene render response",
        )
    )
    for label, path in (
        ("preview scene", preview_scene_path),
        ("image", image_path),
        ("camera", camera_path),
    ):
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Scene render {label} path must be an absolute normalized path"
            )

    expected_workspace = _validate_workspace_path(
        expected_workspace_dir,
        expected_session_id=expected_session_id,
    )
    preview_workspace = preview_scene_path.parent.parent
    render_workspace = image_path.parent.parent
    if (
        preview_scene_path.parent.name != "previews"
        or not preview_scene_path.name.startswith("preview-")
        or preview_scene_path.suffix != ".usda"
        or preview_workspace != expected_workspace
    ):
        raise ValueError(
            "Scene render preview scene is not bound to the requested session"
        )
    if (
        image_path.parent.name != "renders"
        or camera_path.parent != image_path.parent
        or render_workspace != preview_workspace
        or image_path.suffix != ".png"
        or camera_path.suffix != ".json"
        or image_path.stem != camera_path.stem
    ):
        raise ValueError(
            "Scene render artifacts are not bound to the preview session workspace"
        )

    encoded_session_id = quote(expected_session_id, safe="")
    expected_route = f"/sessions/{encoded_session_id}/renders"
    for field_name, artifact_path in (
        ("image_url", image_path),
        ("camera_json_url", camera_path),
    ):
        artifact_url = _require_string(
            response,
            field_name,
            label="Scene render response",
        )
        expected_url = f"{expected_route}/{quote(artifact_path.name, safe='')}"
        if artifact_url != expected_url:
            raise ValueError(
                f"Scene render {field_name} is not bound to the requested session"
            )
    return str(preview_scene_path)


def _mock_png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def _is_render_bundle_name(name: str) -> bool:
    prefix = "render-"
    digest = name.removeprefix(prefix)
    return (
        name.startswith(prefix)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _read_private_artifact_payload(
    binding: ArtifactBinding,
    path: Path,
    *,
    label: str,
) -> bytes:
    """Read one binding through a pinned parent and recheck its path identity."""

    if Path(binding.path).expanduser().resolve() != path:
        raise ValueError(f"{label} path changed after binding verification")
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(path.parent, directory_flags)
    except OSError as exc:
        raise ValueError(f"Cannot open {label} parent directory: {exc}") from exc
    try:
        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            artifact_fd = os.open(path.name, read_flags, dir_fd=directory_fd)
        except OSError as exc:
            raise ValueError(f"Cannot open {label}: {exc}") from exc
        try:
            opened_metadata = os.fstat(artifact_fd)
            if not stat.S_ISREG(opened_metadata.st_mode):
                raise ValueError(f"{label} is not a regular file")
            if opened_metadata.st_nlink != 1:
                raise ValueError(f"{label} must have exactly one hard link")
            with os.fdopen(os.dup(artifact_fd), "rb") as stream:
                payload = stream.read()
        finally:
            os.close(artifact_fd)

        current_metadata = os.stat(
            path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            current_metadata.st_dev,
            current_metadata.st_ino,
        ) != (
            opened_metadata.st_dev,
            opened_metadata.st_ino,
        ):
            raise ValueError(f"{label} changed while being read")

        try:
            current_directory_fd = os.open(path.parent, directory_flags)
        except OSError as exc:
            raise ValueError(f"Cannot re-open {label} parent directory: {exc}") from exc
        try:
            opened_directory = os.fstat(directory_fd)
            current_directory = os.fstat(current_directory_fd)
        finally:
            os.close(current_directory_fd)
        if (
            opened_directory.st_dev,
            opened_directory.st_ino,
        ) != (
            current_directory.st_dev,
            current_directory.st_ino,
        ):
            raise ValueError(f"{label} parent directory changed while being read")
    except OSError as exc:
        raise ValueError(f"Cannot read {label}: {exc}") from exc
    finally:
        os.close(directory_fd)

    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != binding.sha256:
        raise ValueError(
            f"{label} digest mismatch: expected {binding.sha256}, got {observed_sha256}"
        )
    return payload


def _read_render_payloads(
    render: ArticulationSceneRenderEvidence,
    *,
    image_path: Path,
    response_path: Path,
    camera_path: Path,
    collection_dir: Path | None,
    candidate_index: int,
) -> tuple[bytes, bytes, bytes]:
    bindings = (
        ("image", render.image_artifact, image_path),
        ("response", render.response_artifact, response_path),
        ("camera", render.camera_artifact, camera_path),
    )
    bundle_digest = render.render_bundle_sha256
    if bundle_digest is None:
        if collection_dir is not None:
            raise ValueError("Collection-owned Scene renders require bundle digests")
        if any(path.parent.name.startswith("render-") for _, _, path in bindings):
            raise ValueError(
                "Content-addressed Scene render paths require render_bundle_sha256"
            )
        payloads = tuple(
            _read_private_artifact_payload(
                binding,
                path,
                label=f"Scene render {label} artifact",
            )
            for label, binding, path in bindings
        )
        return payloads[0], payloads[1], payloads[2]

    expected_bundle_name = f"render-{bundle_digest}"
    parents = {path.parent for _, _, path in bindings}
    if len(parents) != 1:
        raise ValueError("Scene render bundle artifacts must share one directory")
    bundle_path = parents.pop()
    if bundle_path.name != expected_bundle_name:
        raise ValueError(
            "Scene render bundle directory does not match render_bundle_sha256"
        )
    if collection_dir is None:
        raise ValueError("Content-addressed Scene render has no collection")
    expected_candidate_dir = collection_dir / f"candidate-{candidate_index:04d}"
    if bundle_path.parent != expected_candidate_dir:
        raise ValueError(
            "Scene render bundle is outside its collection candidate directory"
        )
    expected_names = {
        "image": "image.png",
        "response": "response.json",
        "camera": "camera.json",
    }
    for label, binding, path in bindings:
        if path.name != expected_names[label]:
            raise ValueError(
                f"Scene render bundle {label} artifact has an invalid name"
            )
        if Path(binding.path).expanduser().resolve() != path:
            raise ValueError(
                f"Scene render {label} path changed after binding verification"
            )
    if bundle_path.is_symlink() or bundle_path.resolve() != bundle_path:
        raise ValueError(
            "Scene render bundle directory must resolve without symbolic links"
        )

    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        bundle_fd = os.open(bundle_path, directory_flags)
    except OSError as exc:
        raise ValueError(f"Cannot open Scene render bundle: {exc}") from exc
    payload_by_label: dict[str, bytes] = {}
    metadata_by_label: dict[str, os.stat_result] = {}

    def metadata_signature(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    try:
        observed_names = set(os.listdir(bundle_fd))
        if observed_names != set(expected_names.values()):
            raise ValueError(
                "Scene render bundle contains unexpected or missing artifacts"
            )
        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        for label, binding, _path in bindings:
            name = expected_names[label]
            try:
                artifact_fd = os.open(name, read_flags, dir_fd=bundle_fd)
            except OSError as exc:
                raise ValueError(
                    f"Cannot open Scene render {label} artifact: {exc}"
                ) from exc
            try:
                opened_metadata = os.fstat(artifact_fd)
                if not stat.S_ISREG(opened_metadata.st_mode):
                    raise ValueError(
                        f"Scene render {label} artifact is not a regular file"
                    )
                if opened_metadata.st_nlink != 1:
                    raise ValueError(
                        f"Scene render {label} artifact must have exactly one hard link"
                    )
                with os.fdopen(os.dup(artifact_fd), "rb") as stream:
                    payload = stream.read()
            finally:
                os.close(artifact_fd)
            current_metadata = os.stat(
                name,
                dir_fd=bundle_fd,
                follow_symlinks=False,
            )
            if (
                current_metadata.st_dev,
                current_metadata.st_ino,
            ) != (
                opened_metadata.st_dev,
                opened_metadata.st_ino,
            ):
                raise ValueError(
                    f"Scene render {label} artifact changed while being read"
                )
            observed_sha256 = hashlib.sha256(payload).hexdigest()
            if observed_sha256 != binding.sha256:
                raise ValueError(
                    f"Scene render {label} digest mismatch: "
                    f"expected {binding.sha256}, got {observed_sha256}"
                )
            payload_by_label[label] = payload
            metadata_by_label[label] = opened_metadata

        # Reopen and reread the complete set only after the first pass has
        # finished. This catches an earlier entry being changed while a later
        # artifact was read, which per-entry identity checks alone cannot see.
        final_metadata_by_label: dict[str, os.stat_result] = {}
        if set(os.listdir(bundle_fd)) != set(expected_names.values()):
            raise ValueError("Scene render bundle changed between read passes")
        for label, binding, _path in bindings:
            name = expected_names[label]
            artifact_fd = os.open(name, read_flags, dir_fd=bundle_fd)
            try:
                final_opened_metadata = os.fstat(artifact_fd)
                if metadata_signature(final_opened_metadata) != metadata_signature(
                    metadata_by_label[label]
                ):
                    raise ValueError(
                        f"Scene render {label} artifact changed between read passes"
                    )
                with os.fdopen(os.dup(artifact_fd), "rb") as stream:
                    final_payload = stream.read()
            finally:
                os.close(artifact_fd)
            if final_payload != payload_by_label[label]:
                raise ValueError(
                    f"Scene render {label} artifact changed between read passes"
                )
            if hashlib.sha256(final_payload).hexdigest() != binding.sha256:
                raise ValueError(
                    f"Scene render {label} digest changed between read passes"
                )
            final_path_metadata = os.stat(
                name,
                dir_fd=bundle_fd,
                follow_symlinks=False,
            )
            if metadata_signature(final_path_metadata) != metadata_signature(
                final_opened_metadata
            ):
                raise ValueError(
                    f"Scene render {label} artifact changed during final read"
                )
            final_metadata_by_label[label] = final_opened_metadata

        if set(os.listdir(bundle_fd)) != set(expected_names.values()):
            raise ValueError("Scene render bundle changed after final read")
        for label, _binding, _path in bindings:
            final_path_metadata = os.stat(
                expected_names[label],
                dir_fd=bundle_fd,
                follow_symlinks=False,
            )
            if metadata_signature(final_path_metadata) != metadata_signature(
                final_metadata_by_label[label]
            ):
                raise ValueError(
                    f"Scene render {label} artifact changed after final read"
                )

        try:
            current_bundle_fd = os.open(bundle_path, directory_flags)
        except OSError as exc:
            raise ValueError(f"Cannot re-open Scene render bundle: {exc}") from exc
        try:
            opened_bundle = os.fstat(bundle_fd)
            current_bundle = os.fstat(current_bundle_fd)
        finally:
            os.close(current_bundle_fd)
        if (
            opened_bundle.st_dev,
            opened_bundle.st_ino,
        ) != (
            current_bundle.st_dev,
            current_bundle.st_ino,
        ):
            raise ValueError("Scene render bundle changed while being read")
    except OSError as exc:
        raise ValueError(f"Cannot read Scene render bundle: {exc}") from exc
    finally:
        os.close(bundle_fd)

    image_payload = payload_by_label["image"]
    response_payload = payload_by_label["response"]
    camera_payload = payload_by_label["camera"]
    observed_bundle_digest = _render_bundle_digest(
        candidate_id=render.candidate_id,
        focus_prim_path=render.focus_prim_path,
        direction=render.direction,
        width=render.width,
        height=render.height,
        render_quality=render.render_quality,
        image_payload=image_payload,
        response_payload=response_payload,
        camera_payload=camera_payload,
    )
    if observed_bundle_digest != bundle_digest:
        raise ValueError(
            "Scene render bundle digest does not match its exact artifacts"
        )
    return image_payload, response_payload, camera_payload


def _verify_render(
    result: ArticulationSceneEvidenceResult,
    render: ArticulationSceneRenderEvidence,
    *,
    image_path: Path,
    response_path: Path,
    camera_path: Path,
    collection_dir: Path | None,
    candidate_index: int,
) -> None:
    image_payload, response_payload, camera_payload = _read_render_payloads(
        render,
        image_path=image_path,
        response_path=response_path,
        camera_path=camera_path,
        collection_dir=collection_dir,
        candidate_index=candidate_index,
    )
    _validate_render_image_payload(
        image_payload,
        source_label=str(image_path),
        expected_width=render.width,
        expected_height=render.height,
    )
    try:
        response = json.loads(response_payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Scene render response is not valid JSON") from exc
    if not isinstance(response, dict):
        raise ValueError("Scene render response must be a mapping")
    if response.get("session_id") != result.scene_session_id:
        raise ValueError("Scene render session identity does not match")
    if response.get("status") != "success":
        raise ValueError("Scene render response does not report success")
    preview_scene_path = _validate_render_response_scope(
        response,
        expected_session_id=result.scene_session_id,
        expected_workspace_dir=result.scene_workspace_dir,
    )
    if not _same_path(
        preview_scene_path,
        render.preview_scene_path,
    ):
        raise ValueError("Scene render preview scene path does not match")
    for field_name, expected in (
        ("renderer", render.renderer),
        ("render_quality", render.render_quality),
        ("ovrtx_render_mode", render.ovrtx_render_mode),
        ("ovrtx_num_sensor_updates", render.ovrtx_num_sensor_updates),
        ("active_aov", render.active_aov),
    ):
        if response.get(field_name) != expected:
            raise ValueError(f"Scene render {field_name} does not match")

    try:
        camera = json.loads(camera_payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Scene camera evidence is not valid JSON") from exc
    if not isinstance(camera, dict):
        raise ValueError("Scene camera evidence must be a mapping")
    _validate_render_camera(
        camera,
        focus_prim_path=render.focus_prim_path,
        direction=render.direction,
        width=render.width,
        height=render.height,
        render_quality=render.render_quality,
        render_mode=render.ovrtx_render_mode,
        sensor_updates=render.ovrtx_num_sensor_updates,
        active_aov=render.active_aov,
    )


def _validate_render_camera(
    camera: dict[str, Any],
    *,
    focus_prim_path: str,
    direction: str,
    width: int,
    height: int,
    render_quality: str,
    render_mode: str,
    sensor_updates: int,
    active_aov: str,
) -> None:
    """Require camera evidence to describe the exact requested render."""

    for field_name, expected in (
        ("direction", direction),
        ("image_width", width),
        ("image_height", height),
        ("render_quality", render_quality),
        ("ovrtx_render_mode", render_mode),
        ("ovrtx_num_sensor_updates", sensor_updates),
        ("active_aov", active_aov),
    ):
        if camera.get(field_name) != expected:
            raise ValueError(f"Scene camera {field_name} does not match")
    camera_state = camera.get("camera_state")
    if not isinstance(camera_state, dict):
        raise ValueError("Scene camera evidence is missing camera_state")
    if camera_state.get("last_framed_prim_path") != focus_prim_path:
        raise ValueError("Scene camera focus path does not match candidate")
    camera_path = camera.get("camera_path")
    if not isinstance(camera_path, str) or not camera_path.startswith("/"):
        raise ValueError("Scene camera evidence is missing an absolute camera_path")
    _require_finite_matrix4(
        camera.get("camera_world_transform"),
        label="Scene camera camera_world_transform",
    )


def _render_bundle_digest(
    *,
    candidate_id: str,
    focus_prim_path: str,
    direction: str,
    width: int,
    height: int,
    render_quality: str,
    image_payload: bytes,
    response_payload: bytes,
    camera_payload: bytes,
) -> str:
    """Bind one immutable render bundle to both identity and exact bytes."""

    identity = {
        "schema_version": (
            "content-agent-workflows.articulation-scene-render-bundle.v1"
        ),
        "candidate_id": candidate_id,
        "focus_prim_path": focus_prim_path,
        "direction": direction,
        "width": width,
        "height": height,
        "render_quality": render_quality,
        "image_sha256": hashlib.sha256(image_payload).hexdigest(),
        "response_sha256": hashlib.sha256(response_payload).hexdigest(),
        "camera_sha256": hashlib.sha256(camera_payload).hexdigest(),
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_render_bundle_fd(
    bundle_fd: int,
    *,
    expected_payloads: dict[str, bytes],
    label: str,
) -> dict[str, os.stat_result]:
    """Verify one pinned bundle directory without following filesystem links."""

    observed_names = _bounded_directory_names(
        bundle_fd,
        max_entries=len(expected_payloads),
        label=label,
    )
    if observed_names != set(expected_payloads):
        raise ValueError(f"{label} contains unexpected or missing artifacts")
    metadata_by_name: dict[str, os.stat_result] = {}
    signature_by_name: dict[str, tuple[int, ...]] = {}
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    for name, expected_payload in expected_payloads.items():
        try:
            artifact_fd = os.open(name, read_flags, dir_fd=bundle_fd)
        except OSError as exc:
            raise ValueError(f"Cannot open {label} artifact {name}: {exc}") from exc
        try:
            metadata = os.fstat(artifact_fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"{label} artifact {name} is not a regular file")
            if metadata.st_nlink != 1:
                raise ValueError(
                    f"{label} artifact {name} must have exactly one hard link"
                )
            before_signature = _private_file_metadata_signature(
                metadata,
                label=f"{label} artifact {name}",
            )
            if metadata.st_size != len(expected_payload):
                raise ValueError(
                    f"{label} content-addressed artifact {name} has conflicting bytes"
                )
            os.lseek(artifact_fd, 0, os.SEEK_SET)
            with os.fdopen(os.dup(artifact_fd), "rb") as stream:
                observed_payload = stream.read(len(expected_payload) + 1)
            os.fsync(artifact_fd)
            after_fsync_signature = _private_file_metadata_signature(
                os.fstat(artifact_fd),
                label=f"{label} artifact {name}",
            )
            os.lseek(artifact_fd, 0, os.SEEK_SET)
            with os.fdopen(os.dup(artifact_fd), "rb") as final_stream:
                final_payload = final_stream.read(len(expected_payload) + 1)
            final_signature = _private_file_metadata_signature(
                os.fstat(artifact_fd),
                label=f"{label} artifact {name}",
            )
        finally:
            os.close(artifact_fd)
        current_signature = _private_file_metadata_signature(
            os.stat(name, dir_fd=bundle_fd, follow_symlinks=False),
            label=f"{label} artifact {name}",
        )
        if (
            after_fsync_signature != before_signature
            or final_signature != before_signature
            or current_signature != before_signature
        ):
            raise ValueError(f"{label} artifact {name} changed during verification")
        if observed_payload != expected_payload or final_payload != expected_payload:
            raise ValueError(
                f"{label} content-addressed artifact {name} has conflicting bytes"
            )
        metadata_by_name[name] = metadata
        signature_by_name[name] = before_signature
    os.fsync(bundle_fd)
    return _verify_render_bundle_payloads_fd(
        bundle_fd,
        expected_payloads=expected_payloads,
        label=label,
        expected_signatures=signature_by_name,
    )


def _verify_render_bundle_payloads_fd(
    bundle_fd: int,
    *,
    expected_payloads: dict[str, bytes],
    label: str,
    expected_signatures: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, os.stat_result]:
    """Perform a final bundle-wide payload pass without another fsync."""

    observed_names = _bounded_directory_names(
        bundle_fd,
        max_entries=len(expected_payloads),
        label=label,
    )
    if observed_names != set(expected_payloads):
        raise ValueError(f"{label} contains unexpected or missing artifacts")
    metadata_by_name: dict[str, os.stat_result] = {}
    for name, expected_payload in expected_payloads.items():
        expected_signature = (
            None if expected_signatures is None else expected_signatures[name]
        )
        metadata_by_name[name] = _verify_private_artifact_payload_fd(
            bundle_fd,
            name,
            expected_payload=expected_payload,
            label=f"{label} artifact {name}",
            expected_signature=expected_signature,
        )
    final_names = _bounded_directory_names(
        bundle_fd,
        max_entries=len(expected_payloads),
        label=label,
    )
    if final_names != observed_names:
        raise ValueError(f"{label} changed during final verification")
    return metadata_by_name


def _publish_render_bundle(
    evidence_root: Path,
    *,
    render_dir: Path,
    staging_dir: Path,
    bundle_digest: str,
    expected_payloads: dict[str, bytes],
) -> tuple[Path, Path, Path]:
    """Atomically publish one immutable image/response/camera directory."""

    label = "Scene render bundle"
    render_dir = _prepare_contained_directory(
        evidence_root,
        render_dir,
        label="Scene candidate render directory",
    )
    staging_dir = _prepare_contained_directory(
        evidence_root,
        staging_dir,
        label="Scene render download staging directory",
    )
    if staging_dir.parent != render_dir:
        raise ValueError("Scene render staging directory changed parents")
    if set(expected_payloads) != {"image.png", "response.json", "camera.json"}:
        raise ValueError("Scene render bundle has an invalid artifact set")

    bundle_name = f"render-{bundle_digest}"
    bundle_path = _require_contained_write_path(
        evidence_root,
        render_dir / bundle_name,
        label=label,
    )
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        render_fd = os.open(render_dir, directory_flags)
    except OSError as exc:
        raise ValueError(f"Cannot open {label} parent directory: {exc}") from exc

    def require_current_render_dir() -> None:
        try:
            current_fd = os.open(render_dir, directory_flags)
        except OSError as exc:
            raise ValueError(f"Cannot re-open {label} parent directory: {exc}") from exc
        try:
            opened = os.fstat(render_fd)
            current = os.fstat(current_fd)
        finally:
            os.close(current_fd)
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError(f"{label} parent directory changed during publication")

    renamed = False
    staging_metadata: os.stat_result | None = None
    staging_file_metadata: dict[str, os.stat_result] = {}
    try:
        require_current_render_dir()
        staging_fd = os.open(staging_dir.name, directory_flags, dir_fd=render_fd)
        try:
            staging_metadata = os.fstat(staging_fd)
            staging_file_metadata = _verify_render_bundle_fd(
                staging_fd,
                expected_payloads=expected_payloads,
                label=f"{label} staging directory",
            )
        finally:
            os.close(staging_fd)

        try:
            existing_fd = os.open(bundle_name, directory_flags, dir_fd=render_fd)
        except FileNotFoundError:
            existing_fd = -1
        except OSError as exc:
            raise ValueError(f"Cannot inspect existing {label}: {exc}") from exc
        if existing_fd >= 0:
            try:
                _verify_render_bundle_fd(
                    existing_fd,
                    expected_payloads=expected_payloads,
                    label=label,
                )
            finally:
                os.close(existing_fd)
        else:
            require_current_render_dir()
            try:
                os.rename(
                    staging_dir.name,
                    bundle_name,
                    src_dir_fd=render_fd,
                    dst_dir_fd=render_fd,
                )
            except OSError as exc:
                # A concurrent identical publisher may have won the race.
                try:
                    existing_fd = os.open(
                        bundle_name,
                        directory_flags,
                        dir_fd=render_fd,
                    )
                except OSError:
                    raise ValueError(f"Cannot publish {label}: {exc}") from exc
                try:
                    _verify_render_bundle_fd(
                        existing_fd,
                        expected_payloads=expected_payloads,
                        label=label,
                    )
                finally:
                    os.close(existing_fd)
            else:
                renamed = True
                require_current_render_dir()
                published_fd = os.open(
                    bundle_name,
                    directory_flags,
                    dir_fd=render_fd,
                )
                try:
                    published_metadata = os.fstat(published_fd)
                    if staging_metadata is None or (
                        published_metadata.st_dev,
                        published_metadata.st_ino,
                    ) != (
                        staging_metadata.st_dev,
                        staging_metadata.st_ino,
                    ):
                        raise ValueError(f"{label} directory identity changed")
                    _verify_render_bundle_fd(
                        published_fd,
                        expected_payloads=expected_payloads,
                        label=label,
                    )
                finally:
                    os.close(published_fd)
                os.fsync(render_fd)
                require_current_render_dir()

        bundle_path = _require_contained_write_path(
            evidence_root,
            bundle_path,
            label=label,
        )
        if not bundle_path.is_dir():
            raise ValueError(f"{label} is missing")
        image_path = _prepare_artifact_write_path(
            evidence_root,
            bundle_path / "image.png",
            label=f"{label} artifact image.png",
        )
        response_path = _prepare_artifact_write_path(
            evidence_root,
            bundle_path / "response.json",
            label=f"{label} artifact response.json",
        )
        camera_path = _prepare_artifact_write_path(
            evidence_root,
            bundle_path / "camera.json",
            label=f"{label} artifact camera.json",
        )
        require_current_render_dir()
        try:
            final_bundle_fd = os.open(
                bundle_name,
                directory_flags,
                dir_fd=render_fd,
            )
        except OSError as exc:
            raise ValueError(f"Cannot re-open {label}: {exc}") from exc
        try:
            _verify_render_bundle_payloads_fd(
                final_bundle_fd,
                expected_payloads=expected_payloads,
                label=label,
            )
        finally:
            os.close(final_bundle_fd)
        require_current_render_dir()
        return image_path, response_path, camera_path
    except Exception:
        if renamed and staging_metadata is not None:
            try:
                published_fd = os.open(
                    bundle_name,
                    directory_flags,
                    dir_fd=render_fd,
                )
                try:
                    published_metadata = os.fstat(published_fd)
                    if (
                        published_metadata.st_dev,
                        published_metadata.st_ino,
                    ) == (
                        staging_metadata.st_dev,
                        staging_metadata.st_ino,
                    ):
                        current_metadata = _verify_render_bundle_fd(
                            published_fd,
                            expected_payloads=expected_payloads,
                            label=label,
                        )
                        if all(
                            (
                                current_metadata[name].st_dev,
                                current_metadata[name].st_ino,
                            )
                            == (
                                staging_file_metadata[name].st_dev,
                                staging_file_metadata[name].st_ino,
                            )
                            for name in expected_payloads
                        ):
                            for name in expected_payloads:
                                os.unlink(name, dir_fd=published_fd)
                            os.fsync(published_fd)
                finally:
                    os.close(published_fd)
                os.rmdir(bundle_name, dir_fd=render_fd)
                os.fsync(render_fd)
            except OSError:
                _LOGGER.exception("Failed to roll back incomplete %s", label)
            except ValueError:
                _LOGGER.exception("Refused to roll back changed %s", label)
        raise
    finally:
        os.close(render_fd)


def _is_candidate_render_directory_name(name: str) -> bool:
    prefix = "candidate-"
    index = name.removeprefix(prefix)
    return bool(
        name.startswith(prefix)
        and len(index) == 4
        and all(character in "0123456789" for character in index)
    )


def _read_private_fd_payload(
    file_descriptor: int,
    *,
    label: str,
    capture_payload: bool = True,
    max_payload_bytes: int | None = None,
) -> tuple[bytes | None, os.stat_result]:
    metadata = os.fstat(file_descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{label} must be a private regular file")
    if not capture_payload:
        return None, metadata
    if max_payload_bytes is not None and metadata.st_size > max_payload_bytes:
        raise ValueError(f"{label} exceeds its maximum size")
    with os.fdopen(os.dup(file_descriptor), "rb") as stream:
        payload = stream.read(
            max_payload_bytes + 1 if max_payload_bytes is not None else -1
        )
    if max_payload_bytes is not None and len(payload) > max_payload_bytes:
        raise ValueError(f"{label} exceeds its maximum size")
    return payload, metadata


@dataclass(frozen=True)
class _OrphanFileSnapshot:
    name: str
    signature: tuple[int, ...]
    sha256: str | None = None


@dataclass(frozen=True)
class _OrphanDirectorySnapshot:
    name: str
    identity: tuple[int, int, int]
    files: tuple[_OrphanFileSnapshot, ...]
    directories: tuple[_OrphanDirectorySnapshot, ...]


def _bounded_directory_names(
    directory_fd: int,
    *,
    max_entries: int,
    label: str,
) -> set[str]:
    names: set[str] = set()
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                if len(names) >= max_entries:
                    raise ValueError(f"{label} contains too many entries")
                names.add(entry.name)
    except OSError as exc:
        raise ValueError(f"Cannot enumerate {label}: {exc}") from exc
    return names


def _directory_identity(
    metadata: os.stat_result, *, label: str
) -> tuple[int, int, int]:
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} is not a directory")
    return metadata.st_dev, metadata.st_ino, metadata.st_mode


def _private_file_metadata_signature(
    metadata: os.stat_result,
    *,
    label: str,
) -> tuple[int, ...]:
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{label} must be a private regular file")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _capture_orphan_file(
    parent_fd: int,
    name: str,
    *,
    label: str,
    hash_payload: bool = False,
    max_payload_bytes: int | None = None,
) -> _OrphanFileSnapshot:
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        artifact_fd = os.open(name, read_flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"Cannot open {label}: {exc}") from exc
    try:
        before = _private_file_metadata_signature(
            os.fstat(artifact_fd),
            label=label,
        )
        payload_sha256: str | None = None
        if hash_payload:
            if max_payload_bytes is None:
                raise ValueError(f"{label} payload bound is missing")
            if before[4] > max_payload_bytes:
                raise ValueError(f"{label} exceeds its maximum size")
            with os.fdopen(os.dup(artifact_fd), "rb") as stream:
                payload_sha256 = hashlib.sha256(
                    stream.read(max_payload_bytes + 1)
                ).hexdigest()
        after = _private_file_metadata_signature(
            os.fstat(artifact_fd),
            label=label,
        )
    finally:
        os.close(artifact_fd)
    current = _private_file_metadata_signature(
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False),
        label=label,
    )
    if before != after or before != current:
        raise ValueError(f"{label} changed during orphan preflight")
    return _OrphanFileSnapshot(
        name=name,
        signature=before,
        sha256=payload_sha256,
    )


def _orphan_root_temporary_artifact(name: str) -> bool:
    for artifact_name in _COLLECTION_ROOT_ARTIFACT_NAMES:
        prefix = f".{artifact_name}."
        if not name.startswith(prefix) or not name.endswith(".tmp"):
            continue
        token = name[len(prefix) : -len(".tmp")]
        return len(token) == 32 and all(
            character in "0123456789abcdef" for character in token
        )
    return False


def _orphan_download_staging_directory(name: str) -> bool:
    prefix = ".focus-"
    if not name.startswith(prefix):
        return False
    remainder = name[len(prefix) :]
    focus_index, separator, remainder = remainder.partition("-view-")
    if (
        not separator
        or len(focus_index) != 3
        or any(character not in "0123456789" for character in focus_index)
        or int(focus_index) >= _MAX_FOCUSED_RENDERS
    ):
        return False
    direction_index, separator, suffix = remainder.partition("-download-")
    return bool(
        separator
        and len(direction_index) == 2
        and all(character in "0123456789" for character in direction_index)
        and int(direction_index) < _MAX_DIRECTIONS
        and 1 <= len(suffix) <= 32
        and all(
            character
            in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in suffix
        )
    )


def _capture_orphan_leaf_directory(
    parent_fd: int,
    name: str,
    *,
    label: str,
    entry_count: list[int],
) -> _OrphanDirectorySnapshot:
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(name, directory_flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"Cannot open {label}: {exc}") from exc
    try:
        identity = _directory_identity(os.fstat(directory_fd), label=label)
        names = _bounded_directory_names(
            directory_fd,
            max_entries=len(_RENDER_BUNDLE_ARTIFACT_NAMES),
            label=label,
        )
        if not names.issubset(_RENDER_BUNDLE_ARTIFACT_NAMES):
            raise ValueError(f"{label} contains an unexpected entry")
        entry_count[0] += 1 + len(names)
        if entry_count[0] > _MAX_ORPHAN_TREE_ENTRIES:
            raise ValueError("Scene orphan collection tree is too large")
        files = tuple(
            _capture_orphan_file(
                directory_fd,
                artifact_name,
                label=f"{label} artifact {artifact_name}",
            )
            for artifact_name in sorted(names)
        )
        final_identity = _directory_identity(os.fstat(directory_fd), label=label)
        if (
            identity != final_identity
            or _bounded_directory_names(
                directory_fd,
                max_entries=len(names),
                label=label,
            )
            != names
        ):
            raise ValueError(f"{label} changed during orphan preflight")
    finally:
        os.close(directory_fd)
    current_identity = _directory_identity(
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False),
        label=label,
    )
    if current_identity != identity:
        raise ValueError(f"{label} changed during orphan preflight")
    return _OrphanDirectorySnapshot(
        name=name,
        identity=identity,
        files=files,
        directories=(),
    )


def _capture_orphan_candidate_directory(
    collection_fd: int,
    name: str,
    *,
    entry_count: list[int],
) -> tuple[_OrphanDirectorySnapshot, int]:
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    label = f"Scene orphan candidate directory {name}"
    try:
        candidate_fd = os.open(name, directory_flags, dir_fd=collection_fd)
    except OSError as exc:
        raise ValueError(f"Cannot open {label}: {exc}") from exc
    staging_count = 0
    try:
        identity = _directory_identity(os.fstat(candidate_fd), label=label)
        remaining_entry_budget = _MAX_ORPHAN_TREE_ENTRIES - entry_count[0]
        names = _bounded_directory_names(
            candidate_fd,
            max_entries=min(
                _MAX_FOCUSED_RENDERS + 1,
                remaining_entry_budget,
            ),
            label=label,
        )
        unexpected = {
            child_name
            for child_name in names
            if not _is_render_bundle_name(child_name)
            and not _orphan_download_staging_directory(child_name)
        }
        if unexpected:
            raise ValueError(f"{label} contains an unexpected entry")
        staging_count = sum(
            _orphan_download_staging_directory(child_name) for child_name in names
        )
        if staging_count > 1:
            raise ValueError(
                "Scene orphan collection contains multiple download staging directories"
            )
        if sum(_is_render_bundle_name(child_name) for child_name in names) > (
            _MAX_FOCUSED_RENDERS
        ):
            raise ValueError(f"{label} contains too many render bundles")
        entry_count[0] += 1
        if entry_count[0] > _MAX_ORPHAN_TREE_ENTRIES:
            raise ValueError("Scene orphan collection tree is too large")
        directories = tuple(
            _capture_orphan_leaf_directory(
                candidate_fd,
                child_name,
                label=(
                    "Scene orphan render download"
                    if _orphan_download_staging_directory(child_name)
                    else "Scene orphan render bundle"
                ),
                entry_count=entry_count,
            )
            for child_name in sorted(names)
        )
        final_identity = _directory_identity(os.fstat(candidate_fd), label=label)
        if (
            identity != final_identity
            or _bounded_directory_names(
                candidate_fd,
                max_entries=len(names),
                label=label,
            )
            != names
        ):
            raise ValueError(f"{label} changed during orphan preflight")
    finally:
        os.close(candidate_fd)
    current_identity = _directory_identity(
        os.stat(name, dir_fd=collection_fd, follow_symlinks=False),
        label=label,
    )
    if current_identity != identity:
        raise ValueError(f"{label} changed during orphan preflight")
    return (
        _OrphanDirectorySnapshot(
            name=name,
            identity=identity,
            files=(),
            directories=directories,
        ),
        staging_count,
    )


def _capture_orphan_collection(
    evidence_fd: int,
    collection_name: str,
    *,
    collection_id: str,
) -> _OrphanDirectorySnapshot:
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    label = f"Scene orphan collection {collection_id}"
    try:
        collection_fd = os.open(
            collection_name,
            directory_flags,
            dir_fd=evidence_fd,
        )
    except OSError as exc:
        raise ValueError(f"Cannot open {label}: {exc}") from exc
    try:
        identity = _directory_identity(os.fstat(collection_fd), label=label)
        names = _bounded_directory_names(
            collection_fd,
            max_entries=min(
                len(_COLLECTION_ROOT_ARTIFACT_NAMES) + 1 + _MAX_FOCUSED_RENDERS,
                _MAX_ORPHAN_TREE_ENTRIES - 1,
            ),
            label=label,
        )
        root_artifact_names = names & _COLLECTION_ROOT_ARTIFACT_NAMES
        root_temporary_names = {
            name for name in names if _orphan_root_temporary_artifact(name)
        }
        candidate_names = {
            name for name in names if _is_candidate_render_directory_name(name)
        }
        unexpected = names - (
            root_artifact_names | root_temporary_names | candidate_names
        )
        if unexpected:
            raise ValueError(
                "Scene orphan collection contains unexpected entries: "
                + ", ".join(sorted(unexpected))
            )
        marker_present = _COLLECTION_MARKER_NAME in root_artifact_names
        marker_temporary_names = {
            name
            for name in root_temporary_names
            if name.startswith(f".{_COLLECTION_MARKER_NAME}.")
        }
        if not marker_present:
            if (
                candidate_names
                or root_artifact_names
                or root_temporary_names != marker_temporary_names
                or len(marker_temporary_names) > 1
            ):
                raise ValueError("Scene orphan collection marker is missing")
        elif marker_temporary_names:
            raise ValueError(
                "Scene orphan collection has a stale marker temporary artifact"
            )
        if len(root_temporary_names) > 1:
            raise ValueError(
                "Scene orphan collection contains multiple temporary artifacts"
            )
        if len(candidate_names) > _MAX_FOCUSED_RENDERS or any(
            int(name.removeprefix("candidate-")) >= _MAX_FOCUSED_RENDERS
            for name in candidate_names
        ):
            raise ValueError("Scene orphan collection has too many candidates")

        entry_count = [1 + len(root_artifact_names) + len(root_temporary_names)]
        expected_marker_payload = (
            json.dumps(
                _collection_marker_payload(collection_id),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8")
        files = tuple(
            _capture_orphan_file(
                collection_fd,
                artifact_name,
                label=f"{label} artifact {artifact_name}",
                hash_payload=artifact_name == _COLLECTION_MARKER_NAME,
                max_payload_bytes=(
                    len(expected_marker_payload)
                    if artifact_name == _COLLECTION_MARKER_NAME
                    else None
                ),
            )
            for artifact_name in sorted(root_artifact_names | root_temporary_names)
        )
        if marker_present:
            marker_file = next(
                file for file in files if file.name == _COLLECTION_MARKER_NAME
            )
            if (
                marker_file.sha256
                != hashlib.sha256(expected_marker_payload).hexdigest()
            ):
                raise ValueError("Scene orphan collection marker identity changed")

        directories: list[_OrphanDirectorySnapshot] = []
        staging_count = 0
        for candidate_name in sorted(candidate_names):
            candidate_snapshot, candidate_staging_count = (
                _capture_orphan_candidate_directory(
                    collection_fd,
                    candidate_name,
                    entry_count=entry_count,
                )
            )
            directories.append(candidate_snapshot)
            staging_count += candidate_staging_count
        if staging_count > 1:
            raise ValueError(
                "Scene orphan collection contains multiple download staging directories"
            )
        if entry_count[0] > _MAX_ORPHAN_TREE_ENTRIES:
            raise ValueError("Scene orphan collection tree is too large")
        final_identity = _directory_identity(os.fstat(collection_fd), label=label)
        if (
            identity != final_identity
            or _bounded_directory_names(
                collection_fd,
                max_entries=len(names),
                label=label,
            )
            != names
        ):
            raise ValueError(f"{label} changed during orphan preflight")
    finally:
        os.close(collection_fd)
    current_identity = _directory_identity(
        os.stat(collection_name, dir_fd=evidence_fd, follow_symlinks=False),
        label=label,
    )
    if current_identity != identity:
        raise ValueError(f"{label} changed during orphan preflight")
    return _OrphanDirectorySnapshot(
        name=collection_name,
        identity=identity,
        files=files,
        directories=tuple(directories),
    )


def _require_directory_snapshot_binding(
    parent_fd: int,
    snapshot: _OrphanDirectorySnapshot,
    *,
    label: str,
) -> None:
    try:
        current = os.stat(
            snapshot.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ValueError(f"Cannot re-open {label}: {exc}") from exc
    if _directory_identity(current, label=label) != snapshot.identity:
        raise ValueError(f"{label} changed during orphan cleanup")


def _delete_orphan_directory_snapshot(
    parent_fd: int,
    snapshot: _OrphanDirectorySnapshot,
    *,
    label: str,
    parent_guard: Callable[[], None],
) -> None:
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_guard()
    _require_directory_snapshot_binding(parent_fd, snapshot, label=label)
    try:
        directory_fd = os.open(snapshot.name, directory_flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"Cannot open {label} for cleanup: {exc}") from exc

    def current_guard() -> None:
        parent_guard()
        _require_directory_snapshot_binding(parent_fd, snapshot, label=label)

    remaining_names = {
        *(file.name for file in snapshot.files),
        *(directory.name for directory in snapshot.directories),
    }
    try:
        if _directory_identity(os.fstat(directory_fd), label=label) != (
            snapshot.identity
        ):
            raise ValueError(f"{label} changed during orphan cleanup")
        if (
            _bounded_directory_names(
                directory_fd,
                max_entries=len(remaining_names),
                label=label,
            )
            != remaining_names
        ):
            raise ValueError(f"{label} changed before orphan cleanup")

        for child in snapshot.directories:
            current_guard()
            if (
                _bounded_directory_names(
                    directory_fd,
                    max_entries=len(remaining_names),
                    label=label,
                )
                != remaining_names
            ):
                raise ValueError(f"{label} changed during orphan cleanup")
            _delete_orphan_directory_snapshot(
                directory_fd,
                child,
                label=f"{label}/{child.name}",
                parent_guard=current_guard,
            )
            remaining_names.remove(child.name)

        ordered_files = sorted(
            snapshot.files,
            key=lambda file: file.name == _COLLECTION_MARKER_NAME,
        )
        for file in ordered_files:
            current_guard()
            if (
                _bounded_directory_names(
                    directory_fd,
                    max_entries=len(remaining_names),
                    label=label,
                )
                != remaining_names
            ):
                raise ValueError(f"{label} changed during orphan cleanup")
            current_file = _capture_orphan_file(
                directory_fd,
                file.name,
                label=f"{label} artifact {file.name}",
                hash_payload=file.sha256 is not None,
                max_payload_bytes=(
                    file.signature[4] if file.sha256 is not None else None
                ),
            )
            if current_file != file:
                raise ValueError(
                    f"{label} artifact {file.name} changed during orphan cleanup"
                )
            if file.name == _COLLECTION_MARKER_NAME and remaining_names != {
                _COLLECTION_MARKER_NAME
            }:
                raise ValueError(
                    "Scene orphan collection marker would not be removed last"
                )
            os.unlink(file.name, dir_fd=directory_fd)
            os.fsync(directory_fd)
            remaining_names.remove(file.name)
    finally:
        os.close(directory_fd)

    current_guard()
    if remaining_names:
        raise ValueError(f"{label} cleanup did not remove every owned entry")
    os.rmdir(snapshot.name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _reclaim_orphaned_collections(evidence_root: Path) -> tuple[str, ...]:
    """Reclaim marker-authenticated partial collections after a durable resume."""

    resolved_evidence_root = _require_evidence_root_for_write(evidence_root)
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        evidence_fd = os.open(resolved_evidence_root, directory_flags)
    except OSError as exc:
        raise ValueError(
            f"Cannot open Scene evidence directory for recovery: {exc}"
        ) from exc
    try:
        evidence_identity = _directory_identity(
            os.fstat(evidence_fd),
            label="Scene evidence directory",
        )

        def evidence_guard() -> None:
            try:
                os.stat(
                    "manifest.json",
                    dir_fd=evidence_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise ValueError(
                    "Refusing orphan recovery while a Scene manifest exists"
                )
            try:
                current_fd = os.open(resolved_evidence_root, directory_flags)
            except OSError as exc:
                raise ValueError(
                    "Cannot re-open Scene evidence directory during recovery"
                ) from exc
            try:
                current_identity = _directory_identity(
                    os.fstat(current_fd),
                    label="Scene evidence directory",
                )
            finally:
                os.close(current_fd)
            if current_identity != evidence_identity:
                raise ValueError(
                    "Scene evidence directory changed during orphan recovery"
                )

        root_names = _bounded_directory_names(
            evidence_fd,
            max_entries=_MAX_EVIDENCE_ROOT_ENTRIES,
            label="Scene evidence directory",
        )
        if "manifest.json" in root_names:
            raise ValueError("Refusing orphan recovery while a Scene manifest exists")
        malformed_collection_names = {
            name
            for name in root_names
            if name.startswith(_COLLECTION_PREFIX)
            and (
                len(name.removeprefix(_COLLECTION_PREFIX)) != 32
                or any(
                    character not in "0123456789abcdef"
                    for character in name.removeprefix(_COLLECTION_PREFIX)
                )
            )
        }
        if malformed_collection_names:
            raise ValueError(
                "Scene evidence directory contains malformed collection entries"
            )
        collection_names = tuple(
            sorted(name for name in root_names if name.startswith(_COLLECTION_PREFIX))
        )
        if len(collection_names) > _MAX_ORPHAN_COLLECTIONS:
            raise ValueError("Scene evidence directory has too many collections")
        snapshots = tuple(
            _capture_orphan_collection(
                evidence_fd,
                collection_name,
                collection_id=collection_name.removeprefix(_COLLECTION_PREFIX),
            )
            for collection_name in collection_names
        )
        evidence_guard()
        if (
            _bounded_directory_names(
                evidence_fd,
                max_entries=len(root_names),
                label="Scene evidence directory",
            )
            != root_names
        ):
            raise ValueError("Scene evidence directory changed during orphan preflight")
        for snapshot in snapshots:
            _delete_orphan_directory_snapshot(
                evidence_fd,
                snapshot,
                label=f"Scene orphan collection {snapshot.name}",
                parent_guard=evidence_guard,
            )
        return tuple(
            snapshot.name.removeprefix(_COLLECTION_PREFIX) for snapshot in snapshots
        )
    except OSError as exc:
        raise ValueError(f"Cannot reclaim Scene orphan collections: {exc}") from exc
    finally:
        os.close(evidence_fd)


def _validate_owned_collection_tree(
    collection_fd: int,
    *,
    collection_id: str,
    expected_marker_sha256: str,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Validate one exact collection tree before any cleanup mutation."""

    expected_marker_payload = (
        json.dumps(
            _collection_marker_payload(collection_id),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    root_names = set(os.listdir(collection_fd))
    unexpected_root_names = {
        name
        for name in root_names
        if name not in _COLLECTION_ROOT_ARTIFACT_NAMES
        and not _is_candidate_render_directory_name(name)
    }
    if unexpected_root_names:
        raise ValueError(
            "Scene collection contains unexpected entries: "
            + ", ".join(sorted(unexpected_root_names))
        )
    if _COLLECTION_MARKER_NAME not in root_names:
        raise ValueError("Scene collection marker is missing")

    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    for name in sorted(root_names & _COLLECTION_ROOT_ARTIFACT_NAMES):
        artifact_fd = os.open(name, read_flags, dir_fd=collection_fd)
        try:
            payload, _ = _read_private_fd_payload(
                artifact_fd,
                label=f"Scene collection artifact {name}",
                capture_payload=name == _COLLECTION_MARKER_NAME,
                max_payload_bytes=(
                    len(expected_marker_payload)
                    if name == _COLLECTION_MARKER_NAME
                    else None
                ),
            )
        finally:
            os.close(artifact_fd)
        if name == _COLLECTION_MARKER_NAME:
            assert payload is not None
            if hashlib.sha256(payload).hexdigest() != expected_marker_sha256:
                raise ValueError("Scene collection marker digest changed")
            try:
                marker = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError("Scene collection marker is not valid JSON") from exc
            if marker != _collection_marker_payload(collection_id):
                raise ValueError("Scene collection marker identity changed")

    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    candidate_names = tuple(
        sorted(name for name in root_names if _is_candidate_render_directory_name(name))
    )
    bundles_by_candidate: dict[str, tuple[str, ...]] = {}
    expected_bundle_artifacts = {"image.png", "response.json", "camera.json"}
    for candidate_name in candidate_names:
        candidate_fd = os.open(candidate_name, directory_flags, dir_fd=collection_fd)
        try:
            bundle_names = tuple(sorted(os.listdir(candidate_fd)))
            if any(not _is_render_bundle_name(name) for name in bundle_names):
                raise ValueError(
                    f"Scene candidate collection {candidate_name} "
                    "contains an unexpected entry"
                )
            for bundle_name in bundle_names:
                bundle_fd = os.open(
                    bundle_name,
                    directory_flags,
                    dir_fd=candidate_fd,
                )
                try:
                    if set(os.listdir(bundle_fd)) != expected_bundle_artifacts:
                        raise ValueError(
                            f"Scene render bundle {candidate_name}/{bundle_name} "
                            "contains unexpected or missing artifacts"
                        )
                    for artifact_name in expected_bundle_artifacts:
                        artifact_fd = os.open(
                            artifact_name,
                            read_flags,
                            dir_fd=bundle_fd,
                        )
                        try:
                            _read_private_fd_payload(
                                artifact_fd,
                                label=(
                                    "Scene render bundle artifact "
                                    f"{candidate_name}/{bundle_name}/{artifact_name}"
                                ),
                                capture_payload=False,
                            )
                        finally:
                            os.close(artifact_fd)
                finally:
                    os.close(bundle_fd)
            bundles_by_candidate[candidate_name] = bundle_names
        finally:
            os.close(candidate_fd)
    return candidate_names, bundles_by_candidate


def _discard_owned_collection(
    evidence_root: Path,
    *,
    collection_id: str,
    expected_marker_sha256: str,
) -> bool:
    """Delete only one marker-authenticated collection-owned subtree."""

    resolved_evidence_root = _require_evidence_root_for_write(evidence_root)
    collection_name = _collection_name(collection_id)
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        evidence_fd = os.open(resolved_evidence_root, directory_flags)
    except OSError as exc:
        raise ValueError(
            f"Cannot open Scene evidence directory for cleanup: {exc}"
        ) from exc
    try:
        try:
            collection_fd = os.open(
                collection_name,
                directory_flags,
                dir_fd=evidence_fd,
            )
        except FileNotFoundError:
            return False
        try:
            collection_metadata = os.fstat(collection_fd)
            candidate_names, bundles_by_candidate = _validate_owned_collection_tree(
                collection_fd,
                collection_id=collection_id,
                expected_marker_sha256=expected_marker_sha256,
            )
            expected_bundle_artifacts = {
                "image.png",
                "response.json",
                "camera.json",
            }
            expected_marker_payload = (
                json.dumps(
                    _collection_marker_payload(collection_id),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                )
                + "\n"
            ).encode("utf-8")
            for candidate_name in candidate_names:
                candidate_fd = os.open(
                    candidate_name,
                    directory_flags,
                    dir_fd=collection_fd,
                )
                try:
                    candidate_metadata = os.fstat(candidate_fd)
                    for bundle_name in bundles_by_candidate[candidate_name]:
                        bundle_fd = os.open(
                            bundle_name,
                            directory_flags,
                            dir_fd=candidate_fd,
                        )
                        try:
                            bundle_metadata = os.fstat(bundle_fd)
                            if set(os.listdir(bundle_fd)) != expected_bundle_artifacts:
                                raise ValueError(
                                    "Scene render bundle changed during cleanup"
                                )
                            for artifact_name in expected_bundle_artifacts:
                                artifact_fd = os.open(
                                    artifact_name,
                                    read_flags,
                                    dir_fd=bundle_fd,
                                )
                                try:
                                    _read_private_fd_payload(
                                        artifact_fd,
                                        label="Scene render bundle cleanup artifact",
                                        capture_payload=False,
                                    )
                                finally:
                                    os.close(artifact_fd)
                                os.unlink(artifact_name, dir_fd=bundle_fd)
                            os.fsync(bundle_fd)
                        finally:
                            os.close(bundle_fd)
                        current_bundle = os.stat(
                            bundle_name,
                            dir_fd=candidate_fd,
                            follow_symlinks=False,
                        )
                        if (
                            current_bundle.st_dev,
                            current_bundle.st_ino,
                        ) != (
                            bundle_metadata.st_dev,
                            bundle_metadata.st_ino,
                        ):
                            raise ValueError(
                                "Scene render bundle changed during cleanup"
                            )
                        os.rmdir(bundle_name, dir_fd=candidate_fd)
                    os.fsync(candidate_fd)
                finally:
                    os.close(candidate_fd)
                current_candidate = os.stat(
                    candidate_name,
                    dir_fd=collection_fd,
                    follow_symlinks=False,
                )
                if (
                    current_candidate.st_dev,
                    current_candidate.st_ino,
                ) != (
                    candidate_metadata.st_dev,
                    candidate_metadata.st_ino,
                ):
                    raise ValueError(
                        "Scene candidate collection changed during cleanup"
                    )
                os.rmdir(candidate_name, dir_fd=collection_fd)

            remaining_root_names = set(os.listdir(collection_fd))
            if (
                _COLLECTION_MARKER_NAME not in remaining_root_names
                or not remaining_root_names.issubset(_COLLECTION_ROOT_ARTIFACT_NAMES)
            ):
                raise ValueError("Scene collection changed during cleanup")
            ordered_root_names = sorted(
                remaining_root_names,
                key=lambda name: name == _COLLECTION_MARKER_NAME,
            )
            for artifact_name in ordered_root_names:
                artifact_fd = os.open(
                    artifact_name,
                    read_flags,
                    dir_fd=collection_fd,
                )
                try:
                    payload, _ = _read_private_fd_payload(
                        artifact_fd,
                        label="Scene collection cleanup artifact",
                        capture_payload=(artifact_name == _COLLECTION_MARKER_NAME),
                        max_payload_bytes=(
                            len(expected_marker_payload)
                            if artifact_name == _COLLECTION_MARKER_NAME
                            else None
                        ),
                    )
                finally:
                    os.close(artifact_fd)
                if artifact_name == _COLLECTION_MARKER_NAME:
                    assert payload is not None
                    if hashlib.sha256(payload).hexdigest() != expected_marker_sha256:
                        raise ValueError("Scene collection marker changed")
                elif _COLLECTION_MARKER_NAME in remaining_root_names:
                    # Keep the ownership marker durable until all other
                    # collection artifacts have been removed successfully.
                    os.unlink(artifact_name, dir_fd=collection_fd)
                    continue
                os.unlink(artifact_name, dir_fd=collection_fd)
            os.fsync(collection_fd)
        finally:
            os.close(collection_fd)

        current_collection = os.stat(
            collection_name,
            dir_fd=evidence_fd,
            follow_symlinks=False,
        )
        if (
            current_collection.st_dev,
            current_collection.st_ino,
        ) != (
            collection_metadata.st_dev,
            collection_metadata.st_ino,
        ):
            raise ValueError("Scene collection root changed during cleanup")
        os.rmdir(collection_name, dir_fd=evidence_fd)
        os.fsync(evidence_fd)
        return True
    except OSError as exc:
        raise ValueError(f"Cannot discard Scene collection: {exc}") from exc
    finally:
        os.close(evidence_fd)


def _collection_has_committed_manifest(
    result: ArticulationSceneEvidenceResult,
    *,
    evidence_root: Path,
) -> bool:
    manifest_path = evidence_root / "manifest.json"
    try:
        manifest_payload = load_json(manifest_path)
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Cannot determine whether the Scene collection is committed"
        ) from exc
    try:
        manifest_result = ArticulationSceneEvidenceResult.model_validate(
            manifest_payload
        )
    except ValueError as exc:
        raise ValueError(
            "Cannot determine whether the Scene collection is committed"
        ) from exc
    return bool(
        result.collection_id is not None
        and manifest_result.collection_id == result.collection_id
    )


@dataclass(frozen=True)
class _CollectionLayoutSnapshot:
    collection_dir: Path
    collection_signature: tuple[int, ...]
    root_names: frozenset[str]
    root_artifact_signatures: tuple[tuple[str, tuple[int, ...]], ...]
    candidate_signatures: tuple[tuple[str, tuple[int, ...]], ...]
    candidate_bundle_names: tuple[tuple[str, frozenset[str]], ...]
    bundle_signatures: tuple[tuple[str, str, tuple[int, ...]], ...]
    bundle_artifact_signatures: tuple[tuple[str, str, str, tuple[int, ...]], ...]


def _directory_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _regular_file_signature(
    file_descriptor: int,
    *,
    label: str,
) -> tuple[int, ...]:
    metadata = os.fstat(file_descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file")
    if metadata.st_nlink != 1:
        raise ValueError(f"{label} must have exactly one hard link")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _capture_collection_layout(
    result: ArticulationSceneEvidenceResult,
    collection_dir: Path,
) -> _CollectionLayoutSnapshot:
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        collection_fd = os.open(collection_dir, directory_flags)
    except OSError as exc:
        raise ValueError(f"Cannot open Scene collection: {exc}") from exc
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    root_artifact_signatures: list[tuple[str, tuple[int, ...]]] = []
    candidate_signatures: list[tuple[str, tuple[int, ...]]] = []
    candidate_bundle_names: list[tuple[str, frozenset[str]]] = []
    bundle_signatures: list[tuple[str, str, tuple[int, ...]]] = []
    bundle_artifact_signatures: list[tuple[str, str, str, tuple[int, ...]]] = []
    try:
        collection_metadata = os.fstat(collection_fd)
        observed_root_names = frozenset(os.listdir(collection_fd))
        expected_root_names = set(_COLLECTION_ROOT_ARTIFACT_NAMES)
        for artifact_name in sorted(_COLLECTION_ROOT_ARTIFACT_NAMES):
            artifact_fd = os.open(
                artifact_name,
                read_flags,
                dir_fd=collection_fd,
            )
            try:
                signature = _regular_file_signature(
                    artifact_fd,
                    label=f"Scene collection artifact {artifact_name}",
                )
            finally:
                os.close(artifact_fd)
            root_artifact_signatures.append((artifact_name, signature))
        for candidate_index, candidate in enumerate(result.candidates):
            if not candidate.renders:
                continue
            candidate_name = f"candidate-{candidate_index:04d}"
            expected_root_names.add(candidate_name)
            expected_bundle_names = frozenset(
                f"render-{render.render_bundle_sha256}" for render in candidate.renders
            )
            try:
                candidate_fd = os.open(
                    candidate_name,
                    directory_flags,
                    dir_fd=collection_fd,
                )
            except OSError as exc:
                raise ValueError(
                    f"Cannot open Scene candidate collection: {exc}"
                ) from exc
            try:
                candidate_metadata = os.fstat(candidate_fd)
                observed_bundle_names = frozenset(os.listdir(candidate_fd))
                if observed_bundle_names != expected_bundle_names:
                    raise ValueError(
                        "Scene candidate collection contains unbound render bundles"
                    )
                for bundle_name in sorted(expected_bundle_names):
                    bundle_fd = os.open(
                        bundle_name,
                        directory_flags,
                        dir_fd=candidate_fd,
                    )
                    try:
                        bundle_metadata = os.fstat(bundle_fd)
                        observed_artifact_names = frozenset(os.listdir(bundle_fd))
                        expected_artifact_names = frozenset(
                            {"image.png", "response.json", "camera.json"}
                        )
                        if observed_artifact_names != expected_artifact_names:
                            raise ValueError(
                                "Scene render bundle contains unbound artifacts"
                            )
                        bundle_signatures.append(
                            (
                                candidate_name,
                                bundle_name,
                                _directory_signature(bundle_metadata),
                            )
                        )
                        for artifact_name in sorted(expected_artifact_names):
                            artifact_fd = os.open(
                                artifact_name,
                                read_flags,
                                dir_fd=bundle_fd,
                            )
                            try:
                                signature = _regular_file_signature(
                                    artifact_fd,
                                    label=(
                                        "Scene render bundle artifact "
                                        f"{candidate_name}/{bundle_name}/"
                                        f"{artifact_name}"
                                    ),
                                )
                            finally:
                                os.close(artifact_fd)
                            bundle_artifact_signatures.append(
                                (
                                    candidate_name,
                                    bundle_name,
                                    artifact_name,
                                    signature,
                                )
                            )
                    finally:
                        os.close(bundle_fd)
            finally:
                os.close(candidate_fd)
            candidate_signatures.append(
                (candidate_name, _directory_signature(candidate_metadata))
            )
            candidate_bundle_names.append((candidate_name, observed_bundle_names))
        if observed_root_names != frozenset(expected_root_names):
            raise ValueError("Scene collection contains unbound artifacts")
        return _CollectionLayoutSnapshot(
            collection_dir=collection_dir,
            collection_signature=_directory_signature(collection_metadata),
            root_names=observed_root_names,
            root_artifact_signatures=tuple(root_artifact_signatures),
            candidate_signatures=tuple(candidate_signatures),
            candidate_bundle_names=tuple(candidate_bundle_names),
            bundle_signatures=tuple(bundle_signatures),
            bundle_artifact_signatures=tuple(bundle_artifact_signatures),
        )
    except OSError as exc:
        raise ValueError(f"Cannot inspect Scene collection: {exc}") from exc
    finally:
        os.close(collection_fd)


def _validate_collection_result_layout(
    result: ArticulationSceneEvidenceResult,
    *,
    evidence_root: Path,
) -> _CollectionLayoutSnapshot | None:
    """Bind a live result to its exact collection-owned filesystem layout."""

    if result.collection_id is None:
        return None
    collection_artifact = result.collection_artifact
    if collection_artifact is None:
        raise ValueError("Scene collection artifact is missing")
    collection_dir = evidence_root / _collection_name(result.collection_id)
    if (
        collection_dir.is_symlink()
        or collection_dir.resolve() != collection_dir
        or not collection_dir.is_dir()
    ):
        raise ValueError("Scene collection directory is invalid")
    expected_marker_path = collection_dir / _COLLECTION_MARKER_NAME
    if Path(collection_artifact.path).expanduser().resolve() != expected_marker_path:
        raise ValueError("Scene collection marker path does not match its ID")

    expected_root_paths = {
        result.session_response_artifact.path: collection_dir / "session_response.json",
        result.scene_snapshot_artifact.path: collection_dir / "scene_snapshot.json",
        result.topology_inspection_artifact.path: collection_dir
        / "topology_inspection.json",
        result.prim_properties_artifact.path: collection_dir
        / "candidate_prim_properties.json",
    }
    for raw_path, expected_path in expected_root_paths.items():
        if Path(raw_path).expanduser().resolve() != expected_path:
            raise ValueError(
                "Scene collection root artifact path does not match its role"
            )

    expected_root_names = set(_COLLECTION_ROOT_ARTIFACT_NAMES)
    for candidate_index, candidate in enumerate(result.candidates):
        if not candidate.renders:
            continue
        candidate_name = f"candidate-{candidate_index:04d}"
        expected_root_names.add(candidate_name)
        candidate_dir = collection_dir / candidate_name
        if (
            candidate_dir.is_symlink()
            or candidate_dir.resolve() != candidate_dir
            or not candidate_dir.is_dir()
        ):
            raise ValueError("Scene candidate collection directory is invalid")
        expected_bundle_names: set[str] = set()
        for render in candidate.renders:
            bundle_digest = render.render_bundle_sha256
            if bundle_digest is None:
                raise ValueError(
                    "Collection-owned Scene render bundle digest is missing"
                )
            bundle_name = f"render-{bundle_digest}"
            expected_bundle_names.add(bundle_name)
            expected_bundle_dir = candidate_dir / bundle_name
            for binding in (
                render.image_artifact,
                render.response_artifact,
                render.camera_artifact,
            ):
                if Path(binding.path).expanduser().resolve().parent != (
                    expected_bundle_dir
                ):
                    raise ValueError(
                        "Scene render artifact is outside its collection bundle"
                    )
        try:
            observed_bundle_names = set(os.listdir(candidate_dir))
        except OSError as exc:
            raise ValueError(
                f"Cannot inspect Scene candidate collection: {exc}"
            ) from exc
        if observed_bundle_names != expected_bundle_names:
            raise ValueError(
                "Scene candidate collection contains unbound render bundles"
            )
    try:
        observed_root_names = set(os.listdir(collection_dir))
    except OSError as exc:
        raise ValueError(f"Cannot inspect Scene collection: {exc}") from exc
    if observed_root_names != expected_root_names:
        raise ValueError("Scene collection contains unbound artifacts")
    return _capture_collection_layout(result, collection_dir)


def verify_articulation_scene_evidence(
    result: ArticulationSceneEvidenceResult,
    *,
    request: ArticulationWorkflowRequest,
    request_sha256: str,
    source_sha256: str,
    source_dependency_bundle_sha256: str,
    candidate_document: Stage2CandidateDocument,
    candidate_document_sha256: str,
    collector_configuration_sha256: str,
    evidence_root: Path,
) -> None:
    """Rehash and validate every artifact and workflow identity claim."""

    raw_evidence_root = evidence_root.expanduser()
    if not raw_evidence_root.is_absolute():
        raise ValueError("Scene evidence directory must be an absolute path")
    if raw_evidence_root.is_symlink():
        raise ValueError("Scene evidence directory must not be a symbolic link")
    if not raw_evidence_root.is_dir():
        raise ValueError("Scene evidence directory is missing")
    resolved_evidence_root = raw_evidence_root.resolve()
    if resolved_evidence_root != raw_evidence_root:
        raise ValueError(
            "Scene evidence directory must resolve without traversing symlinks"
        )
    collection_layout = _validate_collection_result_layout(
        result,
        evidence_root=resolved_evidence_root,
    )
    collection_dir = (
        collection_layout.collection_dir if collection_layout is not None else None
    )

    source_path = Path(request.source_asset).expanduser().resolve()
    expected_identity = ArticulationSceneEvidenceIdentity(
        request_sha256=request_sha256,
        source_asset=str(source_path),
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        candidate_document_sha256=candidate_document_sha256,
        candidate_ids=candidate_document.candidate_ids,
        collector_configuration_sha256=collector_configuration_sha256,
    )
    if result.identity != expected_identity:
        raise ValueError("Scene evidence identity does not match the current workflow")
    for evidence_candidate, source_candidate in zip(
        result.candidates,
        candidate_document.candidates,
        strict=True,
    ):
        if evidence_candidate.fixed_parent_prim != source_candidate.fixed_parent_prim:
            raise ValueError(
                "Scene candidate fixed parent does not match candidate document"
            )
        if evidence_candidate.moving_part_prims != source_candidate.moving_part_prims:
            raise ValueError(
                "Scene candidate moving prims do not match candidate document"
            )
        expected_candidate_paths = tuple(
            dict.fromkeys(
                (
                    *(
                        (source_candidate.fixed_parent_prim,)
                        if source_candidate.fixed_parent_prim
                        else ()
                    ),
                    *source_candidate.moving_part_prims,
                )
            )
        )
        if evidence_candidate.inspected_prim_paths != expected_candidate_paths:
            raise ValueError(
                "Scene candidate inspected paths do not match candidate document"
            )
    if not _same_path(result.source_scene_path, source_path):
        raise ValueError("Scene evidence source scene path does not match")
    if not _same_path(result.inspection_scene_path, source_path):
        raise ValueError("Scene evidence inspection scene path does not match")
    _require_source_identity(
        source_path,
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
    )
    if result.scene_source_digest != _scene_source_digest(source_path):
        raise ValueError(
            "Scene source digest does not match the current composed source"
        )

    bindings = _artifact_bindings(result)
    resolved_paths: dict[Path, str] = {}
    verified_paths: dict[str, Path] = {}
    verified_payloads: dict[str, bytes] = {}
    root_payload_labels = {
        "collection marker",
        "session response",
        "scene snapshot",
        "topology inspection",
        "prim properties",
    }
    for label, binding in bindings:
        if label in root_payload_labels:
            path, payload = _read_verified_artifact_binding_bytes(
                binding,
                label=label,
                evidence_root=resolved_evidence_root,
            )
            verified_payloads[label] = payload
        else:
            path = _verify_artifact_binding(
                binding,
                label=label,
                evidence_root=resolved_evidence_root,
            )
        if path in resolved_paths:
            raise ValueError(
                f"Scene evidence artifacts alias one path: "
                f"{resolved_paths[path]} and {label}"
            )
        resolved_paths[path] = label
        verified_paths[label] = path

    if collection_dir is not None:
        collection_id = result.collection_id
        if collection_id is None:
            raise ValueError("Scene collection identity is missing")
        marker_payload = _load_json_bytes(
            verified_payloads["collection marker"],
            label="Scene collection marker",
        )
        if marker_payload != _collection_marker_payload(collection_id):
            raise ValueError("Scene collection marker identity does not match")

    _verify_session_response(result, verified_payloads["session response"])
    snapshot_root_prim_path, snapshot_paths = _verify_snapshot(
        result,
        verified_payloads["scene snapshot"],
    )
    expected_inspected_prim_paths = _candidate_endpoint_paths(candidate_document) or (
        snapshot_root_prim_path,
    )
    if result.inspected_prim_paths != expected_inspected_prim_paths:
        raise ValueError("Scene inspected paths do not match candidate document scope")
    missing_snapshot_paths = tuple(
        prim_path
        for prim_path in expected_inspected_prim_paths
        if prim_path not in snapshot_paths
    )
    if missing_snapshot_paths:
        raise ValueError(
            "Scene snapshot does not cover required inspected prim paths: "
            f"{', '.join(missing_snapshot_paths)}"
        )
    _verify_topology(result, verified_payloads["topology inspection"])
    _verify_properties(result, verified_payloads["prim properties"])
    for candidate_index, candidate in enumerate(result.candidates):
        for index, render in enumerate(candidate.renders):
            prefix = f"candidate {candidate.candidate_id} render {index}"
            _verify_render(
                result,
                render,
                image_path=verified_paths[f"{prefix} image"],
                response_path=verified_paths[f"{prefix} response"],
                camera_path=verified_paths[f"{prefix} camera"],
                collection_dir=collection_dir,
                candidate_index=candidate_index,
            )
    if collection_layout is not None:
        final_collection_layout = _capture_collection_layout(
            result,
            collection_layout.collection_dir,
        )
        if final_collection_layout != collection_layout:
            raise ValueError(
                "Scene collection layout changed during evidence verification"
            )


@dataclass(frozen=True)
class _UncommittedSceneCollection:
    """One live result identity authorized for lifecycle cleanup."""

    result: ArticulationSceneEvidenceResult
    evidence_root: Path
    collection_id: str
    marker_sha256: str


class LiveUsdCliArticulationEvidenceCollector:
    """Capture source-bound inspection and focused usd-cli/OVRTX renders."""

    def __init__(
        self,
        *,
        evidence_policy_id: str,
        directions: Sequence[str] = ("+x-y+z",),
        width: int = 1024,
        height: int = 768,
        render_quality: Literal["interactive", "inspection", "final"] = "inspection",
        max_snapshot_prims: int = _MAX_SNAPSHOT_PRIMS,
        timeout: float = 300.0,
    ) -> None:
        if (
            not isinstance(evidence_policy_id, str)
            or not evidence_policy_id.strip()
            or evidence_policy_id != evidence_policy_id.strip()
        ):
            raise ValueError("evidence_policy_id must be a stable non-empty identifier")
        normalized_directions = tuple(
            str(direction).strip() for direction in directions
        )
        if (
            not normalized_directions
            or any(not direction for direction in normalized_directions)
            or len(normalized_directions) > _MAX_DIRECTIONS
        ):
            raise ValueError(
                f"directions must contain between 1 and {_MAX_DIRECTIONS} values"
            )
        _require_unique("directions", normalized_directions)
        if not 1 <= width <= _MAX_RENDER_DIMENSION:
            raise ValueError("width is outside the Scene render bounds")
        if not 1 <= height <= _MAX_RENDER_DIMENSION:
            raise ValueError("height is outside the Scene render bounds")
        if render_quality not in _ALLOWED_RENDER_QUALITIES:
            raise ValueError(f"Unsupported render_quality: {render_quality}")
        if not 1 <= max_snapshot_prims <= _MAX_SNAPSHOT_PRIMS:
            raise ValueError(
                f"max_snapshot_prims must be between 1 and {_MAX_SNAPSHOT_PRIMS}"
            )
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.evidence_policy_id = evidence_policy_id
        self.directions = normalized_directions
        self.width = width
        self.height = height
        self.render_quality = render_quality
        self.max_snapshot_prims = max_snapshot_prims
        self.timeout = timeout
        self._uncommitted_collections: dict[int, _UncommittedSceneCollection] = {}

    def reclaim_orphaned_collections(
        self,
        *,
        evidence_root: Path,
    ) -> tuple[str, ...]:
        """Remove canonical uncommitted collections during a durable resume."""

        return _reclaim_orphaned_collections(evidence_root)

    def _owned_uncommitted_collection(
        self,
        result: ArticulationSceneEvidenceResult,
    ) -> _UncommittedSceneCollection:
        ownership = self._uncommitted_collections.get(id(result))
        if ownership is None or ownership.result is not result:
            raise ValueError(
                "Scene evidence result is not owned by this collector invocation"
            )
        return ownership

    def discard_uncommitted_collection(
        self,
        result: ArticulationSceneEvidenceResult,
        *,
        evidence_root: Path,
    ) -> bool:
        """Consume one invocation-scoped capability and remove its collection."""

        ownership = self._owned_uncommitted_collection(result)
        resolved_evidence_root = _require_evidence_root_for_write(evidence_root)
        if resolved_evidence_root != ownership.evidence_root:
            raise ValueError(
                "Scene evidence root differs from the collecting invocation"
            )
        if _collection_has_committed_manifest(
            ownership.result,
            evidence_root=ownership.evidence_root,
        ):
            raise ValueError(
                "Refusing to discard a Scene collection referenced by the "
                "durable manifest"
            )
        removed = _discard_owned_collection(
            ownership.evidence_root,
            collection_id=ownership.collection_id,
            expected_marker_sha256=ownership.marker_sha256,
        )
        self._uncommitted_collections.pop(id(result), None)
        return removed

    def release_committed_collection(
        self,
        result: ArticulationSceneEvidenceResult,
        *,
        manifest_path: Path,
    ) -> None:
        """Consume one capability only after its exact manifest is durable."""

        ownership = self._owned_uncommitted_collection(result)
        expected_manifest_path = ownership.evidence_root / "manifest.json"
        if Path(manifest_path).expanduser().resolve() != expected_manifest_path:
            raise ValueError(
                "Scene manifest path differs from the collecting invocation"
            )
        try:
            manifest_payload = load_json(expected_manifest_path)
        except (OSError, ValueError) as exc:
            raise ValueError(
                "Cannot verify the committed Scene evidence manifest"
            ) from exc
        if manifest_payload != result.model_dump(mode="json"):
            raise ValueError(
                "Committed Scene evidence manifest differs from the "
                "collecting invocation"
            )
        self._uncommitted_collections.pop(id(result), None)

    def configuration_sha256(self, request: ArticulationWorkflowRequest) -> str:
        return _canonical_sha256(
            {
                "schema_version": _CONFIGURATION_DIGEST_SCHEMA_VERSION,
                "evidence_policy_id": self.evidence_policy_id,
                "target_runtime": request.target_runtime,
                "directions": self.directions,
                "width": self.width,
                "height": self.height,
                "render_quality": self.render_quality,
                "max_snapshot_prims": self.max_snapshot_prims,
            }
        )

    @staticmethod
    def _check_cancelled(cancel_checker: CancelChecker | None) -> None:
        if cancel_checker is not None and cancel_checker():
            raise asyncio.CancelledError

    def _render_candidate(
        self,
        *,
        session: WorkflowUsdCliSession,
        candidate_id: str,
        candidate_index: int,
        focus_prim_path: str,
        focus_index: int,
        direction: str,
        direction_index: int,
        evidence_root: Path,
        collection_dir: Path,
        expected_workspace_dir: str,
        up_axis_y: bool,
    ) -> ArticulationSceneRenderEvidence:
        render_dir = _prepare_contained_directory(
            evidence_root,
            collection_dir / f"candidate-{candidate_index:04d}",
            label="usd-cli candidate render directory",
        )
        _fsync_directory_entry(render_dir, label="usd-cli candidate render directory")
        name = f"focus-{focus_index:03d}-view-{direction_index:02d}"
        azimuth, elevation = direction_angles(
            direction,
            up_axis_y=up_axis_y,
        )
        orbit_response = session.run_json(
            [
                "camera",
                "orbit",
                focus_prim_path,
                "--az",
                str(azimuth),
                "--el",
                str(elevation),
            ]
        )
        orbit_summary = orbit_response.get("summary")
        if not isinstance(orbit_summary, dict):
            raise ValueError("usd-cli camera orbit returned no summary")
        camera_prim_path = orbit_summary.get("camera")
        if not isinstance(camera_prim_path, str) or not camera_prim_path.startswith(
            "/"
        ):
            raise ValueError(
                "usd-cli camera orbit did not report an absolute camera path"
            )
        with tempfile.TemporaryDirectory(
            dir=render_dir,
            prefix=f".{name}-render-",
        ) as staging_dir_value:
            staging_dir = _prepare_contained_directory(
                evidence_root,
                Path(staging_dir_value),
                label="usd-cli render staging directory",
            )
            staged_image_path = staging_dir / "image.png"
            staged_response_path = staging_dir / "response.json"
            staged_camera_path = staging_dir / "camera.json"
            raw_response = session.run_json(
                [
                    "render",
                    "--res",
                    f"{self.width}x{self.height}",
                    "--mode",
                    "quality" if self.render_quality == "final" else "fast",
                    "--camera",
                    camera_prim_path,
                    "--output",
                    str(staged_image_path),
                ],
                timeout_seconds=self.timeout,
            )
            summary = raw_response.get("summary", {})
            if not isinstance(summary, dict):
                raise ValueError("usd-cli articulation render returned no summary")
            backend = str(summary.get("backend") or "")
            if backend not in {"ovrtx", "remote"}:
                raise ValueError(
                    f"usd-cli articulation evidence did not use OVRTX: {backend!r}"
                )
            if not staged_image_path.is_file():
                raise ValueError("usd-cli articulation render did not produce an image")
            render_mode = summary.get("ovrtx_render_mode")
            sensor_updates = summary.get("ovrtx_num_sensor_updates")
            active_aov = summary.get("active_aov")
            if not isinstance(render_mode, str) or not render_mode:
                raise ValueError(
                    "usd-cli articulation render did not report its effective "
                    "OVRTX render mode"
                )
            if (
                not isinstance(sensor_updates, int)
                or isinstance(sensor_updates, bool)
                or sensor_updates < 1
            ):
                raise ValueError(
                    "usd-cli articulation render did not report its effective "
                    "OVRTX sensor-update count"
                )
            if not isinstance(active_aov, str) or not active_aov:
                raise ValueError(
                    "usd-cli articulation render did not report its active OVRTX AOV"
                )
            if summary.get("camera") != camera_prim_path:
                raise ValueError(
                    "usd-cli articulation render did not use the orbit camera"
                )
            camera_world_transform = _require_finite_matrix4(
                summary.get("camera_world_transform"),
                label="usd-cli render camera_world_transform",
            )
            workspace = PurePosixPath(expected_workspace_dir)
            preview_scene_path = workspace / "previews" / f"preview-{name}.usda"
            workspace_image_path = workspace / "renders" / f"{name}.png"
            workspace_camera_path = workspace / "renders" / f"{name}.json"
            encoded_session_id = quote(session.session_id, safe="")
            route = f"/sessions/{encoded_session_id}/renders"
            response = {
                "session_id": session.session_id,
                "status": "success",
                "preview_scene_path": str(preview_scene_path),
                "image_path": str(workspace_image_path),
                "camera_json_path": str(workspace_camera_path),
                "image_url": f"{route}/{quote(workspace_image_path.name, safe='')}",
                "camera_json_url": (
                    f"{route}/{quote(workspace_camera_path.name, safe='')}"
                ),
                # ``remote`` is the OVRTX transport, not a distinct renderer.
                "renderer": "ovrtx",
                "render_transport": backend,
                "render_quality": self.render_quality,
                "ovrtx_render_mode": render_mode,
                "ovrtx_num_sensor_updates": sensor_updates,
                "active_aov": active_aov,
                "usd_cli_response": raw_response,
            }
            camera = {
                "camera_path": camera_prim_path,
                "camera_world_transform": camera_world_transform,
                "direction": direction,
                "image_width": self.width,
                "image_height": self.height,
                "render_quality": self.render_quality,
                "ovrtx_render_mode": render_mode,
                "ovrtx_num_sensor_updates": sensor_updates,
                "active_aov": active_aov,
                "camera_state": {
                    "last_framed_prim_path": focus_prim_path,
                    "usd_cli_summary": raw_response.get("summary", {}),
                },
            }
            atomic_write_json(staged_response_path, response)
            atomic_write_json(staged_camera_path, camera)
            staged_image_bytes = staged_image_path.read_bytes()
            staged_response_bytes = staged_response_path.read_bytes()
            staged_camera_bytes = staged_camera_path.read_bytes()
            _validate_render_image_payload(
                staged_image_bytes,
                source_label=str(staged_image_path),
                expected_width=self.width,
                expected_height=self.height,
            )
            bundle_digest = _render_bundle_digest(
                candidate_id=candidate_id,
                focus_prim_path=focus_prim_path,
                direction=direction,
                width=self.width,
                height=self.height,
                render_quality=self.render_quality,
                image_payload=staged_image_bytes,
                response_payload=staged_response_bytes,
                camera_payload=staged_camera_bytes,
            )
            image_path, response_path, camera_path = _publish_render_bundle(
                evidence_root,
                render_dir=render_dir,
                staging_dir=staging_dir,
                bundle_digest=bundle_digest,
                expected_payloads={
                    "image.png": staged_image_bytes,
                    "response.json": staged_response_bytes,
                    "camera.json": staged_camera_bytes,
                },
            )
        return ArticulationSceneRenderEvidence(
            candidate_id=candidate_id,
            focus_prim_path=focus_prim_path,
            direction=direction,
            width=self.width,
            height=self.height,
            render_quality=self.render_quality,
            preview_scene_path=str(preview_scene_path),
            renderer="ovrtx",
            ovrtx_render_mode=render_mode,
            ovrtx_num_sensor_updates=sensor_updates,
            active_aov=active_aov,
            render_bundle_sha256=bundle_digest,
            image_artifact=ArtifactBinding(
                path=str(image_path),
                sha256=hashlib.sha256(staged_image_bytes).hexdigest(),
            ),
            response_artifact=ArtifactBinding(
                path=str(response_path),
                sha256=hashlib.sha256(staged_response_bytes).hexdigest(),
            ),
            camera_artifact=ArtifactBinding(
                path=str(camera_path),
                sha256=hashlib.sha256(staged_camera_bytes).hexdigest(),
            ),
        )

    def collect(
        self,
        request: ArticulationWorkflowRequest,
        *,
        request_sha256: str,
        source_sha256: str,
        source_dependency_bundle_sha256: str,
        candidate_document: Stage2CandidateDocument,
        candidate_document_sha256: str,
        output_dir: Path,
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationSceneEvidenceResult:
        """Collect one isolated transaction and discard its subtree on failure."""

        for candidate in candidate_document.candidates:
            _require_unique(
                f"candidate {candidate.candidate_id} moving_part_prims",
                candidate.moving_part_prims,
            )
        focused_render_count = sum(
            len(candidate.moving_part_prims)
            for candidate in candidate_document.candidates
        ) * len(self.directions)
        if focused_render_count > _MAX_FOCUSED_RENDERS:
            raise ValueError(
                "Scene focused render workload "
                f"{focused_render_count} exceeds limit {_MAX_FOCUSED_RENDERS}"
            )
        resolved_evidence_dir = _prepare_evidence_root_for_write(output_dir)
        collection_id, collection_dir, collection_artifact = (
            _create_collection_directory(resolved_evidence_dir)
        )
        try:
            result = self._collect(
                request,
                request_sha256=request_sha256,
                source_sha256=source_sha256,
                source_dependency_bundle_sha256=(source_dependency_bundle_sha256),
                candidate_document=candidate_document,
                candidate_document_sha256=candidate_document_sha256,
                output_dir=resolved_evidence_dir,
                collection_id=collection_id,
                collection_dir=collection_dir,
                collection_artifact=collection_artifact,
                cancel_checker=cancel_checker,
            )
            self._uncommitted_collections[id(result)] = _UncommittedSceneCollection(
                result=result,
                evidence_root=resolved_evidence_dir,
                collection_id=collection_id,
                marker_sha256=collection_artifact.sha256,
            )
            return result
        except BaseException as exc:
            try:
                _discard_owned_collection(
                    resolved_evidence_dir,
                    collection_id=collection_id,
                    expected_marker_sha256=collection_artifact.sha256,
                )
            except Exception as cleanup_error:
                exc.add_note(
                    "Could not discard failed Scene collection "
                    f"{collection_id}: {cleanup_error}"
                )
                _LOGGER.exception(
                    "Failed to discard Scene collection %s",
                    collection_id,
                )
            raise

    def _collect(
        self,
        request: ArticulationWorkflowRequest,
        *,
        request_sha256: str,
        source_sha256: str,
        source_dependency_bundle_sha256: str,
        candidate_document: Stage2CandidateDocument,
        candidate_document_sha256: str,
        output_dir: Path,
        collection_id: str,
        collection_dir: Path,
        collection_artifact: ArtifactBinding,
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationSceneEvidenceResult:
        source_path = Path(request.source_asset).expanduser().resolve()
        focused_render_count = sum(
            len(candidate.moving_part_prims)
            for candidate in candidate_document.candidates
        ) * len(self.directions)
        if focused_render_count > _MAX_FOCUSED_RENDERS:
            raise ValueError(
                "Scene focused render workload "
                f"{focused_render_count} exceeds limit {_MAX_FOCUSED_RENDERS}"
            )
        resolved_evidence_dir = _prepare_evidence_root_for_write(output_dir)
        resolved_collection_dir = _require_contained_write_path(
            resolved_evidence_dir,
            collection_dir,
            label="Scene collection directory",
        )
        if (
            resolved_collection_dir.name != _collection_name(collection_id)
            or not resolved_collection_dir.is_dir()
        ):
            raise ValueError("Scene collection directory identity does not match")
        configuration_sha256 = self.configuration_sha256(request)
        identity = ArticulationSceneEvidenceIdentity(
            request_sha256=request_sha256,
            source_asset=str(source_path),
            source_sha256=source_sha256,
            source_dependency_bundle_sha256=source_dependency_bundle_sha256,
            candidate_document_sha256=candidate_document_sha256,
            candidate_ids=candidate_document.candidate_ids,
            collector_configuration_sha256=configuration_sha256,
        )
        _require_source_identity(
            source_path,
            source_sha256=source_sha256,
            source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        )
        scene_source_digest = _scene_source_digest(source_path)
        up_axis_y = stage_up_axis_is_y(source_path)
        self._check_cancelled(cancel_checker)
        session = WorkflowUsdCliSession.create(
            owner_root=resolved_evidence_dir,
            project_dir=resolved_evidence_dir,
            identity=f"{request_sha256}:{candidate_document_sha256}",
            workflow="articulation-evidence",
            input_roots=(source_path,),
            receipt_checkpoint_sha256=_receipt_checkpoint_sha256_for_retry(
                resolved_evidence_dir
            ),
        )
        session_id = session.session_id
        scene_workspace_dir = str(session.project_dir / session_id)
        session_response = {
            "session_id": session_id,
            "status": "ready",
            "source_scene_path": str(source_path),
            "inspection_scene_path": str(source_path),
            "artifacts": {"workspace_dir": scene_workspace_dir},
            "scene_tool": "usd-cli",
        }

        try:
            session.require_ovrtx(resolved_evidence_dir / "ovrtx_probe")
            session.open(source_path)
            _validate_session_response_payload(
                session_response,
                expected_session_id=session_id,
                expected_source_scene_path=source_path,
                expected_inspection_scene_path=source_path,
            )

            self._check_cancelled(cancel_checker)
            topology_response = session.run_json(
                ["physics", "topology"], timeout_seconds=self.timeout
            )
            topology = topology_response.get("data")
            if not isinstance(topology, dict):
                raise ValueError("usd-cli topology response is missing structured data")
            try:
                _validate_topology_payload(
                    topology,
                    expected_asset=source_path,
                    expected_source_digest=scene_source_digest,
                )
            except ValueError as exc:
                raise ValueError(
                    f"usd-cli-observed topology does not match workflow identity: {exc}"
                ) from exc

            snapshot_response = session.run_json(
                ["snapshot", "--bounds", "--properties"],
                timeout_seconds=self.timeout,
            )
            snapshot_data = snapshot_response.get("data")
            if not isinstance(snapshot_data, dict):
                raise ValueError("usd-cli snapshot is missing structured data")
            nodes = snapshot_data.get("nodes")
            if not isinstance(nodes, list) or not all(
                isinstance(node, dict) and isinstance(node.get("path"), str)
                for node in nodes
            ):
                raise ValueError("usd-cli snapshot nodes are malformed")
            if len(nodes) > self.max_snapshot_prims:
                raise ValueError("usd-cli snapshot exceeds the configured prim limit")
            snapshot_paths = tuple(str(node["path"]) for node in nodes)
            snapshot = {
                "session_id": session_id,
                "source_scene_path": str(source_path),
                "inspection_scene_path": str(source_path),
                "root_prim_path": snapshot_paths[0] if snapshot_paths else "/",
                "paths": list(snapshot_paths),
                "summary": {"truncated": False, "prim_count": len(snapshot_paths)},
                "usd_cli_response": snapshot_response,
            }
            endpoint_paths = _candidate_endpoint_paths(candidate_document)
            root_prim_path, snapshot_paths = _validate_snapshot_payload(
                snapshot,
                expected_session_id=session_id,
                expected_source_scene_path=source_path,
                expected_inspection_scene_path=source_path,
            )
            inspected_prim_paths = endpoint_paths or (root_prim_path,)
            missing_snapshot_paths = tuple(
                prim_path
                for prim_path in inspected_prim_paths
                if prim_path not in snapshot_paths
            )
            if missing_snapshot_paths:
                raise ValueError(
                    "Scene snapshot does not cover required inspected prim paths: "
                    f"{', '.join(missing_snapshot_paths)}"
                )
            property_results: list[dict[str, Any]] = []
            for prim_path in inspected_prim_paths:
                property_response = session.run_json(
                    ["properties", prim_path], timeout_seconds=self.timeout
                )
                raw_properties = property_response.get("data")
                if not isinstance(raw_properties, dict):
                    raise ValueError(f"usd-cli properties are missing for {prim_path}")
                property_results.append(
                    {
                        "prim_path": prim_path,
                        "truncated": False,
                        "properties": {
                            "path": prim_path,
                            "name": str(raw_properties.get("name") or ""),
                            "type_name": str(raw_properties.get("type") or ""),
                            "active": bool(raw_properties.get("active", True)),
                            "loaded": bool(raw_properties.get("loaded", True)),
                            "metadata": raw_properties.get("metadata") or {},
                            "attributes": raw_properties.get("attributes") or {},
                            "relationships": raw_properties.get("relationships") or {},
                            "bounds": raw_properties.get("bounds"),
                        },
                        "usd_cli_response": property_response,
                    }
                )
            properties = {"session_id": session_id, "results": property_results}
            _validate_properties_payload(
                properties,
                expected_session_id=session_id,
                expected_prim_paths=inspected_prim_paths,
            )

            session_response_path = _write_evidence_json(
                resolved_evidence_dir,
                resolved_collection_dir / "session_response.json",
                session_response,
                label="Scene session response artifact",
            )
            snapshot_path = _write_evidence_json(
                resolved_evidence_dir,
                resolved_collection_dir / "scene_snapshot.json",
                snapshot,
                label="Scene scene snapshot artifact",
            )
            topology_path = _write_evidence_json(
                resolved_evidence_dir,
                resolved_collection_dir / "topology_inspection.json",
                topology,
                label="Scene topology inspection artifact",
            )
            properties_path = _write_evidence_json(
                resolved_evidence_dir,
                resolved_collection_dir / "candidate_prim_properties.json",
                properties,
                label="Scene prim properties artifact",
            )

            candidate_evidence: list[ArticulationSceneCandidateEvidence] = []
            for candidate_index, candidate in enumerate(candidate_document.candidates):
                candidate_inspected_paths = tuple(
                    dict.fromkeys(
                        (
                            *(
                                (candidate.fixed_parent_prim,)
                                if candidate.fixed_parent_prim
                                else ()
                            ),
                            *candidate.moving_part_prims,
                        )
                    )
                )
                renders: list[ArticulationSceneRenderEvidence] = []
                for focus_index, focus_path in enumerate(candidate.moving_part_prims):
                    for direction_index, direction in enumerate(self.directions):
                        self._check_cancelled(cancel_checker)
                        renders.append(
                            self._render_candidate(
                                session=session,
                                candidate_id=candidate.candidate_id,
                                candidate_index=candidate_index,
                                focus_prim_path=focus_path,
                                focus_index=focus_index,
                                direction=direction,
                                direction_index=direction_index,
                                evidence_root=resolved_evidence_dir,
                                collection_dir=resolved_collection_dir,
                                expected_workspace_dir=scene_workspace_dir,
                                up_axis_y=up_axis_y,
                            )
                        )
                candidate_evidence.append(
                    ArticulationSceneCandidateEvidence(
                        candidate_id=candidate.candidate_id,
                        fixed_parent_prim=candidate.fixed_parent_prim,
                        moving_part_prims=candidate.moving_part_prims,
                        inspected_prim_paths=candidate_inspected_paths,
                        renders=tuple(renders),
                    )
                )

            result = ArticulationSceneEvidenceResult(
                identity=identity,
                scene_session_id=session_id,
                scene_workspace_dir=scene_workspace_dir,
                source_scene_path=str(source_path),
                inspection_scene_path=str(source_path),
                scene_source_digest=scene_source_digest,
                inspected_prim_paths=inspected_prim_paths,
                collection_id=collection_id,
                collection_artifact=collection_artifact,
                session_response_artifact=_artifact_binding(session_response_path),
                scene_snapshot_artifact=_artifact_binding(snapshot_path),
                topology_inspection_artifact=_artifact_binding(topology_path),
                prim_properties_artifact=_artifact_binding(properties_path),
                candidates=tuple(candidate_evidence),
            )
        finally:
            try:
                session.close()
            except Exception:
                _LOGGER.warning(
                    "Failed to close usd-cli articulation evidence session %s.",
                    session_id,
                    exc_info=True,
                )

        _require_source_identity(
            source_path,
            source_sha256=source_sha256,
            source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        )
        if _scene_source_digest(source_path) != scene_source_digest:
            raise ValueError(
                "Source digest changed before usd-cli evidence checkpointing"
            )
        verify_articulation_scene_evidence(
            result,
            request=request,
            request_sha256=request_sha256,
            source_sha256=source_sha256,
            source_dependency_bundle_sha256=source_dependency_bundle_sha256,
            candidate_document=candidate_document,
            candidate_document_sha256=candidate_document_sha256,
            collector_configuration_sha256=configuration_sha256,
            evidence_root=resolved_evidence_dir,
        )
        return result


class MockArticulationSceneEvidenceCall(_StrictFrozenModel):
    """One recorded mock collection request for workflow assertions."""

    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_ids: tuple[str, ...] = ()
    output_dir: str = Field(min_length=1)


class MockArticulationSceneEvidenceCollector:
    """Credential-free evidence collector for durable workflow tests."""

    def __init__(self, *, evidence_policy_id: str = "mock-v1") -> None:
        self.evidence_policy_id = evidence_policy_id
        self.calls: list[MockArticulationSceneEvidenceCall] = []

    @property
    def call_count(self) -> int:
        """Return the number of recorded collection attempts."""

        return len(self.calls)

    def configuration_sha256(self, request: ArticulationWorkflowRequest) -> str:
        return _canonical_sha256(
            {
                "schema_version": _CONFIGURATION_DIGEST_SCHEMA_VERSION,
                "adapter": "mock",
                "evidence_policy_id": self.evidence_policy_id,
                "target_runtime": request.target_runtime,
            }
        )

    def collect(
        self,
        request: ArticulationWorkflowRequest,
        *,
        request_sha256: str,
        source_sha256: str,
        source_dependency_bundle_sha256: str,
        candidate_document: Stage2CandidateDocument,
        candidate_document_sha256: str,
        output_dir: Path,
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationSceneEvidenceResult:
        if cancel_checker is not None and cancel_checker():
            raise asyncio.CancelledError
        source_path = Path(request.source_asset).expanduser().resolve()
        resolved_evidence_dir = _prepare_evidence_root_for_write(output_dir)
        self.calls.append(
            MockArticulationSceneEvidenceCall(
                request_sha256=request_sha256,
                source_sha256=source_sha256,
                source_dependency_bundle_sha256=(source_dependency_bundle_sha256),
                candidate_document_sha256=candidate_document_sha256,
                candidate_ids=candidate_document.candidate_ids,
                output_dir=str(resolved_evidence_dir),
            )
        )
        session_id = "mock-articulation-scene-session"
        mock_workspace = PurePosixPath("/mock-scene") / session_id
        endpoint_paths = _candidate_endpoint_paths(candidate_document)
        inspected_prim_paths = endpoint_paths or ("/",)
        session_response = {
            "session_id": session_id,
            "status": "ready",
            "source_scene_path": str(source_path),
            "inspection_scene_path": str(source_path),
            "artifacts": {"workspace_dir": str(mock_workspace)},
        }
        snapshot = {
            "session_id": session_id,
            "root_prim_path": "/",
            "source_scene_path": str(source_path),
            "inspection_scene_path": str(source_path),
            "paths": list(inspected_prim_paths),
            "summary": {"mock": True, "truncated": False},
        }
        topology: dict[str, Any] = {
            "schema_version": "usd-cli.physics-topology.v1",
            "asset": str(source_path),
            "source_digest": _scene_source_digest(source_path),
            "path_space": "source",
            "root_prim_path": None,
            "rigid_body_paths": [],
            "enabled_rigid_body_count": 0,
            "colliders": [],
            "enabled_collider_count": 0,
            "joints": [],
            "articulation_root_paths": [],
            "fixed_to_world_joints": [],
            "findings": [],
        }
        properties = {
            "session_id": session_id,
            "results": [
                {
                    "session_id": session_id,
                    "prim_path": path,
                    "properties": {
                        "path": path,
                        "name": PurePosixPath(path).name,
                        "type_name": "Xform",
                        "active": True,
                        "loaded": True,
                        "metadata": {},
                        "attributes": {},
                        "relationships": {},
                        "bounds": None,
                    },
                    "truncated": False,
                }
                for path in inspected_prim_paths
            ],
        }
        session_response_path = _write_evidence_json(
            resolved_evidence_dir,
            resolved_evidence_dir / "session_response.json",
            session_response,
            label="Scene session response artifact",
        )
        snapshot_path = _write_evidence_json(
            resolved_evidence_dir,
            resolved_evidence_dir / "scene_snapshot.json",
            snapshot,
            label="Scene scene snapshot artifact",
        )
        topology_path = _write_evidence_json(
            resolved_evidence_dir,
            resolved_evidence_dir / "topology_inspection.json",
            topology,
            label="Scene topology inspection artifact",
        )
        properties_path = _write_evidence_json(
            resolved_evidence_dir,
            resolved_evidence_dir / "candidate_prim_properties.json",
            properties,
            label="Scene prim properties artifact",
        )
        candidates: list[ArticulationSceneCandidateEvidence] = []
        for candidate_index, candidate in enumerate(candidate_document.candidates):
            candidate_paths = tuple(
                dict.fromkeys(
                    (
                        *(
                            (candidate.fixed_parent_prim,)
                            if candidate.fixed_parent_prim
                            else ()
                        ),
                        *candidate.moving_part_prims,
                    )
                )
            )
            renders: list[ArticulationSceneRenderEvidence] = []
            for focus_index, focus_path in enumerate(candidate.moving_part_prims):
                render_dir = _prepare_contained_directory(
                    resolved_evidence_dir,
                    resolved_evidence_dir / f"candidate-{candidate_index:04d}",
                    label="Scene candidate render directory",
                )
                name = f"focus-{focus_index:03d}-view-00"
                preview_scene_path = (
                    mock_workspace
                    / "previews"
                    / f"preview-{candidate_index:04d}-{focus_index:03d}.usda"
                )
                service_image_path = mock_workspace / "renders" / f"{name}.png"
                service_camera_path = mock_workspace / "renders" / f"{name}.json"
                image_path = _write_evidence_bytes(
                    resolved_evidence_dir,
                    render_dir / f"{name}.png",
                    _mock_png_bytes(),
                    label="Scene render image artifact",
                )
                response_path = _write_evidence_json(
                    resolved_evidence_dir,
                    render_dir / f"{name}_response.json",
                    {
                        "session_id": session_id,
                        "status": "success",
                        "preview_scene_path": str(preview_scene_path),
                        "image_path": str(service_image_path),
                        "image_url": (
                            f"/sessions/{session_id}/renders/{service_image_path.name}"
                        ),
                        "camera_json_path": str(service_camera_path),
                        "camera_json_url": (
                            f"/sessions/{session_id}/renders/{service_camera_path.name}"
                        ),
                        "renderer": "mock",
                        "render_quality": "inspection",
                        "ovrtx_render_mode": "mock",
                        "ovrtx_num_sensor_updates": 1,
                        "active_aov": "LdrColor",
                    },
                    label="Scene render response artifact",
                )
                camera_path = _write_evidence_json(
                    resolved_evidence_dir,
                    render_dir / f"{name}_camera.json",
                    {
                        "camera_path": "/mock-scene/Cameras/Main",
                        "camera_world_transform": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        "camera_state": {
                            "last_framed_prim_path": focus_path,
                        },
                        "direction": "+x-y+z",
                        "image_width": 64,
                        "image_height": 64,
                        "render_quality": "inspection",
                        "ovrtx_render_mode": "mock",
                        "ovrtx_num_sensor_updates": 1,
                        "active_aov": "LdrColor",
                    },
                    label="Scene render camera artifact",
                )
                renders.append(
                    ArticulationSceneRenderEvidence(
                        candidate_id=candidate.candidate_id,
                        focus_prim_path=focus_path,
                        direction="+x-y+z",
                        width=64,
                        height=64,
                        render_quality="inspection",
                        preview_scene_path=str(preview_scene_path),
                        renderer="mock",
                        ovrtx_render_mode="mock",
                        ovrtx_num_sensor_updates=1,
                        active_aov="LdrColor",
                        image_artifact=_artifact_binding(image_path),
                        response_artifact=_artifact_binding(response_path),
                        camera_artifact=_artifact_binding(camera_path),
                    )
                )
            candidates.append(
                ArticulationSceneCandidateEvidence(
                    candidate_id=candidate.candidate_id,
                    fixed_parent_prim=candidate.fixed_parent_prim,
                    moving_part_prims=candidate.moving_part_prims,
                    inspected_prim_paths=candidate_paths,
                    renders=tuple(renders),
                )
            )

        result = ArticulationSceneEvidenceResult(
            identity=ArticulationSceneEvidenceIdentity(
                request_sha256=request_sha256,
                source_asset=str(source_path),
                source_sha256=source_sha256,
                source_dependency_bundle_sha256=source_dependency_bundle_sha256,
                candidate_document_sha256=candidate_document_sha256,
                candidate_ids=candidate_document.candidate_ids,
                collector_configuration_sha256=self.configuration_sha256(request),
            ),
            scene_session_id=session_id,
            scene_workspace_dir=str(mock_workspace),
            source_scene_path=str(source_path),
            inspection_scene_path=str(source_path),
            scene_source_digest=_scene_source_digest(source_path),
            inspected_prim_paths=inspected_prim_paths,
            session_response_artifact=_artifact_binding(session_response_path),
            scene_snapshot_artifact=_artifact_binding(snapshot_path),
            topology_inspection_artifact=_artifact_binding(topology_path),
            prim_properties_artifact=_artifact_binding(properties_path),
            candidates=tuple(candidates),
        )
        verify_articulation_scene_evidence(
            result,
            request=request,
            request_sha256=request_sha256,
            source_sha256=source_sha256,
            source_dependency_bundle_sha256=source_dependency_bundle_sha256,
            candidate_document=candidate_document,
            candidate_document_sha256=candidate_document_sha256,
            collector_configuration_sha256=self.configuration_sha256(request),
            evidence_root=resolved_evidence_dir,
        )
        return result


__all__ = [
    "ARTICULATION_SCENE_EVIDENCE_SCHEMA_VERSION",
    "ArticulationSceneCandidateEvidence",
    "ArticulationSceneEvidenceCollector",
    "ArticulationSceneEvidenceIdentity",
    "ArticulationSceneEvidenceResult",
    "ArticulationSceneRenderEvidence",
    "LiveUsdCliArticulationEvidenceCollector",
    "MockArticulationSceneEvidenceCall",
    "MockArticulationSceneEvidenceCollector",
    "articulation_scene_artifact_bindings",
    "verify_articulation_scene_evidence",
]
