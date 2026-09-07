# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed artifact, source, and durable job storage."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import threading
import uuid
import zipfile
from base64 import b64encode
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import UploadFile
from geometry_authoring_contracts.source_dependencies import (
    DEPENDENCY_REPRESENTATION_ROLES,
    PROCESSABLE_REPRESENTATION_ROLES,
)

from .config import Settings
from .models import (
    ArtifactRecord,
    ArtifactRole,
    JobKind,
    JobRecord,
    ServiceError,
    SourceRecord,
)
from .source_dependencies import (
    SourceDependencyError,
    validate_3mf_package_dependencies,
    validate_materialized_source_dependencies,
)

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_ID_RE = re.compile(
    r"^sha256:(?P<digest>[0-9a-f]{64})(?::(?P<reference>[0-9a-f]{64}))?$"
)
_SOURCE_ID_RE = re.compile(r"^src_[0-9a-f]{64}$")
_JOB_ID_RE = re.compile(r"^job_[0-9a-f]{32}$")
_JOB_OWNER_KEY = "_owner_instance_id"

GEOMETRY_SOURCE_EXTENSIONS = {
    ".3mf",
    ".brep",
    ".glb",
    ".gltf",
    ".iges",
    ".igs",
    ".obj",
    ".ply",
    ".step",
    ".stl",
    ".stp",
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
    ".zip",
}
REFERENCE_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}
_ARCHIVE_EXTENSIONS = {".3mf", ".usdz", ".zip"}
_BUNDLE_MEMBER_EXTENSIONS = (
    GEOMETRY_SOURCE_EXTENSIONS
    | REFERENCE_IMAGE_EXTENSIONS
    | {
        ".bin",
        ".exr",
        ".hdr",
        ".json",
        ".mtl",
        ".model",
        ".rels",
        ".tif",
        ".tiff",
        ".txt",
        ".xml",
    }
) - {".zip"}
_ARCHIVE_ENTRYPOINT_EXTENSIONS = GEOMETRY_SOURCE_EXTENSIONS - {".zip"}


class StorageError(ValueError):
    """Typed storage validation failure suitable for an HTTP error response."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _archive_member_suffix(name: str) -> str:
    path = PurePosixPath(name)
    return ".rels" if path.name.casefold() == ".rels" else path.suffix.lower()


def utc_now() -> datetime:
    return datetime.now(UTC)


def _normalized_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("job-owner lease timestamps must include a UTC offset")
    return value.astimezone(UTC)


class WorkspaceStorage:
    """Stores bytes by digest and exposes only opaque IDs to API callers."""

    def __init__(self, settings: Settings) -> None:
        raw_root = Path(settings.workspace_root).expanduser()
        if raw_root.exists() and raw_root.is_symlink():
            raise RuntimeError("Geometry Agent workspace root must not be a symlink")
        self.root = raw_root.resolve()
        self.settings = settings
        self.artifacts_dir = self.root / "artifacts" / "sha256"
        self.sources_dir = self.root / "records" / "sources"
        self.jobs_dir = self.root / "records" / "jobs"
        self.job_owners_dir = self.root / "records" / "job-owners"
        self.executions_dir = self.root / "executions"
        self.tmp_dir = self.root / "tmp"
        self._lock = threading.RLock()

    def initialize(self) -> None:
        for path in (
            self.artifacts_dir,
            self.sources_dir,
            self.jobs_dir,
            self.job_owners_dir,
            self.executions_dir,
            self.tmp_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                raise RuntimeError(
                    f"Workspace storage directory must not be a symlink: {path}"
                )

    async def store_upload(
        self,
        upload: UploadFile,
        *,
        role: ArtifactRole,
        expected_sha256: str | None,
        archive_entrypoint: str | None,
    ) -> SourceRecord:
        self.initialize()
        filename = _validate_filename(upload.filename)
        suffix = Path(filename).suffix.lower()
        _validate_extension(suffix, role)
        expected = _normalize_expected_digest(expected_sha256)
        temp_path = self.tmp_dir / f"{uuid.uuid4().hex}{suffix}"
        digest = hashlib.sha256()
        size = 0
        try:
            with temp_path.open("xb") as stream:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    upload_limit = (
                        self.settings.max_reference_image_bytes
                        if role == "reference_image"
                        else self.settings.max_upload_bytes
                    )
                    if size > upload_limit:
                        raise StorageError(
                            "upload_too_large",
                            "Uploaded artifact exceeds the configured byte limit.",
                            status_code=413,
                        )
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            actual = digest.hexdigest()
            if expected is not None and actual != expected:
                raise StorageError(
                    "digest_mismatch",
                    "Uploaded artifact does not match X-Content-SHA256.",
                    status_code=422,
                )
            selected_entrypoint = None
            if suffix in _ARCHIVE_EXTENSIONS:
                selected_entrypoint = self._validate_archive(
                    temp_path,
                    suffix=suffix,
                    requested_entrypoint=archive_entrypoint,
                )
            elif archive_entrypoint is not None:
                raise StorageError(
                    "unexpected_archive_entrypoint",
                    "archive_entrypoint is valid only for ZIP-based source containers.",
                    status_code=422,
                )
            if role == "geometry_source":
                try:
                    if suffix in _ARCHIVE_EXTENSIONS:
                        self._validate_archive_dependency_closure(
                            temp_path,
                            suffix=suffix,
                            selected_entrypoint=selected_entrypoint,
                        )
                    else:
                        validate_materialized_source_dependencies(
                            temp_path,
                            package_root=temp_path.parent,
                            package_files=(temp_path,),
                        )
                except SourceDependencyError as exc:
                    raise StorageError(
                        "unbound_source_dependency",
                        (
                            "Geometry source dependencies must be self-contained or "
                            "bound inside one validated source package."
                        ),
                        status_code=422,
                    ) from exc
            artifact = self._commit_temp_artifact(
                temp_path,
                digest=actual,
                filename=filename,
                media_type=upload.content_type,
                size_bytes=size,
            )
            return self._create_source_record(
                artifact,
                role=role,
                archive_entrypoint=selected_entrypoint,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def get_source(self, source_id: str) -> SourceRecord:
        if not _SOURCE_ID_RE.fullmatch(source_id):
            raise StorageError(
                "source_not_found", "Source artifact was not found.", status_code=404
            )
        path = self.sources_dir / f"{source_id}.json"
        try:
            payload = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise StorageError(
                "source_not_found",
                "Source artifact was not found.",
                status_code=404,
            ) from exc
        return SourceRecord.model_validate_json(payload)

    def get_artifact(self, artifact_id: str) -> tuple[ArtifactRecord, Path]:
        """Resolve and re-verify one content-addressed artifact for download."""

        match = _ARTIFACT_ID_RE.fullmatch(artifact_id)
        if match is None:
            raise StorageError(
                "artifact_not_found", "Artifact was not found.", status_code=404
            )
        digest = match.group("digest")
        reference_digest = match.group("reference")
        blob = self._artifact_blob(digest)
        metadata_path = (
            blob.parent / "references" / f"{reference_digest}.json"
            if reference_digest is not None
            else blob.parent / "artifact.json"
        )
        try:
            record = ArtifactRecord.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise StorageError(
                "artifact_not_found", "Artifact was not found.", status_code=404
            ) from exc
        if (
            record.artifact_id != artifact_id
            or record.sha256 != digest
            or blob.stat().st_size != record.size_bytes
            or _sha256_file(blob) != digest
        ):
            raise RuntimeError(f"Corrupt content-addressed artifact: {digest}")
        return record, blob

    def inline_source_artifact(self, source: SourceRecord) -> dict[str, Any]:
        """Return one bounded digest-verified artifact for a provider request."""

        if source.role != "reference_image":
            raise StorageError(
                "invalid_reference_image",
                "Only reference_image sources can be bound into authoring requests.",
                status_code=422,
            )
        if source.artifact.size_bytes > self.settings.max_reference_image_bytes:
            raise StorageError(
                "reference_image_too_large",
                "Reference image exceeds the provider request byte limit.",
                status_code=422,
            )
        blob = self._verified_artifact_blob(source.artifact)
        content = blob.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if len(content) != source.artifact.size_bytes or not hmac.compare_digest(
            actual, source.artifact.sha256
        ):
            raise RuntimeError(
                f"Corrupt content-addressed reference image: {source.artifact.sha256}"
            )
        return {
            "filename": source.artifact.filename,
            "media_type": source.artifact.media_type or "application/octet-stream",
            "sha256": source.artifact.sha256,
            "size_bytes": source.artifact.size_bytes,
            "content_base64": b64encode(content).decode("ascii"),
        }

    def materialize_source(self, source: SourceRecord, destination: Path) -> Path:
        """Copy a registered source into a job-owned directory without executing it."""

        destination = self._execution_subdir(destination)
        destination.mkdir(parents=True, exist_ok=False)
        blob = self._verified_artifact_blob(source.artifact)
        suffix = Path(source.artifact.filename).suffix.lower()
        if source.archive_entrypoint is not None:
            if suffix not in _ARCHIVE_EXTENSIONS:
                raise StorageError(
                    "unexpected_archive_entrypoint",
                    "The source record assigns an entrypoint to a non-archive artifact.",
                    status_code=422,
                )
            self._extract_archive(blob, destination)
            entrypoint = (destination / source.archive_entrypoint).resolve()
            if (
                not entrypoint.is_relative_to(destination.resolve())
                or not entrypoint.is_file()
            ):
                raise StorageError(
                    "archive_entrypoint_missing",
                    "The registered archive entrypoint is unavailable.",
                    status_code=422,
                )
            return entrypoint
        if suffix == ".zip":
            raise StorageError(
                "archive_entrypoint_missing",
                "The source bundle has no geometry entrypoint.",
                status_code=422,
            )
        output = destination / source.artifact.filename
        shutil.copyfile(blob, output)
        return output

    def materialize_source_handoff(
        self,
        source: SourceRecord,
        destination: Path,
    ) -> tuple[Path, Path | None]:
        """Materialize one source and its optional canonical bundle manifest."""

        if source.source_bundle_manifest is None:
            return self.materialize_source(source, destination), None
        destination = self._execution_subdir(destination)
        destination.mkdir(parents=True, exist_ok=False)
        records = source.source_bundle_artifacts
        if not records:
            raise StorageError(
                "source_bundle_incomplete",
                "The registered source bundle has no materialized artifacts.",
                status_code=500,
            )
        selected_path: Path | None = None
        for artifact in records:
            target = destination / (artifact.bundle_member or artifact.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise StorageError(
                    "source_bundle_name_collision",
                    "The registered source bundle contains duplicate relative paths.",
                    status_code=500,
                )
            shutil.copyfile(self._verified_artifact_blob(artifact), target)
            if artifact == source.artifact:
                selected_path = target
        if selected_path is None:
            raise StorageError(
                "source_bundle_incomplete",
                "The selected source representation is missing from its bundle.",
                status_code=500,
            )
        manifest = destination / source.source_bundle_manifest.filename
        shutil.copyfile(
            self._verified_artifact_blob(source.source_bundle_manifest),
            manifest,
        )
        return selected_path, manifest

    def materialized_source_sha256(self, path: Path) -> str:
        """Digest one regular source materialized inside this execution workspace."""

        if path.is_symlink():
            raise StorageError(
                "invalid_materialized_source",
                "Materialized geometry source must not be a symbolic link.",
                status_code=500,
            )
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
        if (
            not resolved.is_relative_to(self.executions_dir)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise StorageError(
                "invalid_materialized_source",
                "Materialized geometry source is outside its execution workspace.",
                status_code=500,
            )
        return _sha256_file(resolved)

    def store_generated_file(
        self,
        path: Path,
        *,
        expected_filename: str | None = None,
        expected_sha256: str | None = None,
        expected_size_bytes: int | None = None,
        media_type: str | None = None,
        bundle_member: str | None = None,
    ) -> ArtifactRecord:
        """Publish a workflow-owned file as a content-addressed artifact."""

        if path.is_symlink():
            raise StorageError(
                "invalid_generated_artifact",
                "Workflow artifacts must not be symbolic links.",
                status_code=500,
            )
        resolved = path.resolve(strict=True)
        file_stat = resolved.stat()
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_nlink != 1
            or not resolved.is_relative_to(self.executions_dir)
        ):
            raise StorageError(
                "invalid_generated_artifact",
                "Workflow artifacts must be single-link regular files in the job workspace.",
                status_code=500,
            )
        if expected_filename is not None and resolved.name != expected_filename:
            raise StorageError(
                "provider_artifact_name_mismatch",
                "Provider artifact path differs from its declared filename.",
                status_code=500,
            )
        temp_path = self.tmp_dir / f"{uuid.uuid4().hex}.generated"
        digest = hashlib.sha256()
        size = 0
        try:
            with resolved.open("rb") as source, temp_path.open("xb") as target:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.settings.max_generated_artifact_bytes:
                        raise StorageError(
                            "generated_artifact_too_large",
                            "A generated artifact exceeds the configured byte limit.",
                            status_code=500,
                        )
                    digest.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            actual_digest = digest.hexdigest()
            if expected_size_bytes is not None and size != expected_size_bytes:
                raise StorageError(
                    "provider_artifact_size_mismatch",
                    "Provider artifact differs from its declared size.",
                    status_code=500,
                )
            if expected_sha256 is not None and not hmac.compare_digest(
                actual_digest, expected_sha256
            ):
                raise StorageError(
                    "provider_artifact_digest_mismatch",
                    "Provider artifact differs from its declared SHA-256 digest.",
                    status_code=500,
                )
            return self._commit_temp_artifact(
                temp_path,
                digest=actual_digest,
                filename=resolved.name,
                bundle_member=bundle_member,
                media_type=media_type,
                size_bytes=size,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def create_source_from_artifact(self, artifact: ArtifactRecord) -> SourceRecord:
        """Register a provider-produced geometry artifact as a runnable source."""

        _validate_extension(Path(artifact.filename).suffix.lower(), "geometry_source")
        self._validate_provider_dependency_closure((("render_geometry", artifact),))
        return self._create_source_record(
            artifact,
            role="geometry_source",
            archive_entrypoint=None,
        )

    def create_source_from_bundle(
        self,
        artifact: ArtifactRecord,
        *,
        manifest: ArtifactRecord,
        representation_artifacts: list[tuple[str, ArtifactRecord]],
        provenance_artifacts: list[ArtifactRecord] | None = None,
        representation_id: str,
    ) -> SourceRecord:
        """Register a provider-produced canonical source bundle."""

        _validate_extension(Path(artifact.filename).suffix.lower(), "geometry_source")
        artifacts = [item for _role, item in representation_artifacts]
        persisted_provenance = provenance_artifacts or []
        if artifact not in artifacts:
            raise StorageError(
                "source_bundle_incomplete",
                "The selected source representation is missing from its bundle.",
                status_code=500,
            )
        self._validate_provider_dependency_closure(tuple(representation_artifacts))
        bundle_members = [
            item.bundle_member or item.filename
            for item in (*artifacts, *persisted_provenance)
        ]
        if len(bundle_members) != len(set(bundle_members)):
            raise StorageError(
                "source_bundle_name_collision",
                "The registered source bundle contains duplicate relative paths.",
                status_code=500,
            )
        return self._create_source_record(
            artifact,
            role="geometry_source",
            archive_entrypoint=None,
            source_bundle_manifest=manifest,
            source_bundle_artifacts=[*artifacts, *persisted_provenance],
            source_representation_id=representation_id,
        )

    def execution_dir(self, job_id: str) -> Path:
        if not _JOB_ID_RE.fullmatch(job_id):
            raise ValueError(f"Invalid job id: {job_id!r}")
        path = self.executions_dir / job_id
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _execution_subdir(self, path: Path) -> Path:
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(self.executions_dir):
            raise StorageError(
                "invalid_workspace_destination",
                "Job destination must remain inside the execution workspace.",
                status_code=500,
            )
        return resolved

    def _artifact_blob(self, digest: str) -> Path:
        if not _DIGEST_RE.fullmatch(digest):
            raise StorageError(
                "artifact_not_found", "Artifact was not found.", status_code=404
            )
        path = self.artifacts_dir / digest[:2] / digest / "blob"
        if not path.is_file():
            raise StorageError(
                "artifact_not_found", "Artifact was not found.", status_code=404
            )
        return path

    def _verified_artifact_blob(self, artifact: ArtifactRecord) -> Path:
        """Resolve one record and verify its immutable blob before materialization."""

        persisted, blob = self.get_artifact(artifact.artifact_id)
        if persisted != artifact:
            raise RuntimeError(
                f"Corrupt content-addressed artifact metadata: {artifact.artifact_id}"
            )
        return blob

    def _commit_temp_artifact(
        self,
        temp_path: Path,
        *,
        digest: str,
        filename: str,
        bundle_member: str | None = None,
        media_type: str | None,
        size_bytes: int,
    ) -> ArtifactRecord:
        artifact_dir = self.artifacts_dir / digest[:2] / digest
        blob = artifact_dir / "blob"
        reference_payload = {
            "sha256": digest,
            "filename": filename,
            "bundle_member": bundle_member,
            "media_type": media_type,
            "size_bytes": size_bytes,
        }
        reference_digest = hashlib.sha256(
            json.dumps(
                reference_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        record = ArtifactRecord(
            artifact_id=f"sha256:{digest}:{reference_digest}",
            sha256=digest,
            filename=filename,
            bundle_member=bundle_member,
            media_type=media_type,
            size_bytes=size_bytes,
        )
        with self._lock:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            if blob.exists():
                if blob.stat().st_size != size_bytes or _sha256_file(blob) != digest:
                    raise RuntimeError(f"Corrupt content-addressed artifact: {digest}")
            else:
                os.replace(temp_path, blob)
                blob.chmod(0o400)
            metadata_path = artifact_dir / "references" / f"{reference_digest}.json"
            metadata_path.parent.mkdir(mode=0o700, exist_ok=True)
            if metadata_path.exists():
                persisted = ArtifactRecord.model_validate_json(
                    metadata_path.read_text(encoding="utf-8")
                )
                if persisted != record:
                    raise RuntimeError(
                        f"Corrupt content-addressed artifact metadata: {reference_digest}"
                    )
            else:
                _atomic_write_json(metadata_path, record.model_dump(mode="json"))
        return record

    def _create_source_record(
        self,
        artifact: ArtifactRecord,
        *,
        role: ArtifactRole,
        archive_entrypoint: str | None,
        source_bundle_manifest: ArtifactRecord | None = None,
        source_bundle_artifacts: list[ArtifactRecord] | None = None,
        source_representation_id: str | None = None,
    ) -> SourceRecord:
        bundle_artifacts = source_bundle_artifacts or []
        identity_fields: dict[str, Any] = {
            "artifact_id": artifact.artifact_id,
            "archive_entrypoint": archive_entrypoint,
            "source_bundle_manifest": (
                source_bundle_manifest.artifact_id
                if source_bundle_manifest is not None
                else None
            ),
            "filename": artifact.filename,
            "role": role,
        }
        if bundle_artifacts:
            identity_fields["source_bundle_artifacts"] = sorted(
                item.artifact_id for item in bundle_artifacts
            )
        if source_representation_id is not None:
            identity_fields["source_representation_id"] = source_representation_id
        identity = json.dumps(
            identity_fields,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        source_id = f"src_{hashlib.sha256(identity).hexdigest()}"
        record = SourceRecord(
            source_id=source_id,
            role=role,
            artifact=artifact,
            archive_entrypoint=archive_entrypoint,
            source_bundle_manifest=source_bundle_manifest,
            source_bundle_artifacts=bundle_artifacts,
            source_representation_id=source_representation_id,
            created_at=utc_now(),
        )
        path = self.sources_dir / f"{source_id}.json"
        with self._lock:
            if path.exists():
                return SourceRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            _atomic_write_json(path, record.model_dump(mode="json"))
        return record

    def _validate_archive(
        self,
        path: Path,
        *,
        suffix: str,
        requested_entrypoint: str | None,
    ) -> str | None:
        try:
            archive = zipfile.ZipFile(path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise StorageError(
                "invalid_archive",
                f"{suffix} uploads must be valid ZIP containers.",
                status_code=422,
            ) from exc
        with archive:
            files = self._validated_members(archive)
        candidates = [
            name
            for name in files
            if _archive_member_suffix(name) in _ARCHIVE_ENTRYPOINT_EXTENSIONS
        ]
        if requested_entrypoint is not None:
            normalized = _validate_member_name(requested_entrypoint)
            if normalized not in candidates:
                raise StorageError(
                    "invalid_archive_entrypoint",
                    "archive_entrypoint must name a supported geometry file in the bundle.",
                    status_code=422,
                )
            return normalized
        if suffix != ".zip":
            return None
        if len(candidates) != 1:
            raise StorageError(
                "archive_entrypoint_required",
                "A .zip source bundle must contain exactly one geometry file or declare archive_entrypoint.",
                status_code=422,
            )
        return candidates[0]

    def _validated_members(self, archive: zipfile.ZipFile) -> list[str]:
        infos = archive.infolist()
        if len(infos) > self.settings.max_archive_members:
            raise StorageError(
                "archive_member_limit_exceeded",
                "Archive contains too many members.",
                status_code=422,
            )
        names: set[str] = set()
        files: list[str] = []
        unpacked = 0
        compressed = 0
        for info in infos:
            name = _validate_member_name(info.filename)
            if name in names:
                raise StorageError(
                    "duplicate_archive_member",
                    "Archive contains duplicate member names.",
                    status_code=422,
                )
            names.add(name)
            if info.flag_bits & 0x1:
                raise StorageError(
                    "encrypted_archive_unsupported",
                    "Encrypted archive members are not supported.",
                    status_code=422,
                )
            mode = (info.external_attr >> 16) & 0o170000
            if stat.S_ISLNK(mode):
                raise StorageError(
                    "archive_link_forbidden",
                    "Archive links are not allowed.",
                    status_code=422,
                )
            if info.is_dir():
                continue
            suffix = _archive_member_suffix(name)
            if suffix not in _BUNDLE_MEMBER_EXTENSIONS:
                raise StorageError(
                    "archive_member_type_forbidden",
                    f"Archive member type is not allowed: {suffix or '<none>'}",
                    status_code=422,
                )
            unpacked += info.file_size
            compressed += info.compress_size
            files.append(name)
        if unpacked > self.settings.max_archive_unpacked_bytes:
            raise StorageError(
                "archive_unpacked_limit_exceeded",
                "Archive exceeds the configured unpacked byte limit.",
                status_code=422,
            )
        if compressed == 0 and unpacked > 0:
            ratio = float("inf")
        else:
            ratio = unpacked / max(compressed, 1)
        if ratio > self.settings.max_archive_compression_ratio:
            raise StorageError(
                "archive_compression_ratio_exceeded",
                "Archive compression ratio exceeds the configured safety limit.",
                status_code=422,
            )
        return files

    def _extract_archive(self, blob: Path, destination: Path) -> None:
        with zipfile.ZipFile(blob) as archive:
            self._validated_members(archive)
            for info in archive.infolist():
                name = _validate_member_name(info.filename)
                target = (destination / name).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise StorageError(
                        "archive_traversal_forbidden",
                        "Archive member escapes the extraction directory.",
                        status_code=422,
                    )
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)

    def _validate_archive_dependency_closure(
        self,
        archive_path: Path,
        *,
        suffix: str,
        selected_entrypoint: str | None,
    ) -> None:
        validation_root = self.tmp_dir / f"{uuid.uuid4().hex}.dependency-check"
        validation_root.mkdir(mode=0o700)
        try:
            self._extract_archive(archive_path, validation_root)
            package_files = tuple(
                path for path in validation_root.rglob("*") if path.is_file()
            )
            if selected_entrypoint is not None:
                source_path = validation_root / selected_entrypoint
                validate_materialized_source_dependencies(
                    source_path,
                    package_root=validation_root,
                    package_files=package_files,
                )
                return
            if suffix == ".3mf":
                validate_3mf_package_dependencies(validation_root, package_files)
                return
            if suffix == ".usdz":
                with zipfile.ZipFile(archive_path) as archive:
                    ordered_files = self._validated_members(archive)
                if not ordered_files or _archive_member_suffix(
                    ordered_files[0]
                ) not in {".usd", ".usda", ".usdc"}:
                    raise SourceDependencyError(
                        "USDZ package must begin with its root USD layer"
                    )
                validate_materialized_source_dependencies(
                    validation_root / ordered_files[0],
                    package_root=validation_root,
                    package_files=package_files,
                )
                return
            raise SourceDependencyError("source package has no geometry entrypoint")
        finally:
            shutil.rmtree(validation_root, ignore_errors=True)

    def _validate_provider_dependency_closure(
        self,
        representation_artifacts: tuple[tuple[str, ArtifactRecord], ...],
    ) -> None:
        """Validate runnable roots without interpreting opaque provider artifacts."""

        validation_root = self.tmp_dir / f"{uuid.uuid4().hex}.provider-dependency-check"
        validation_root.mkdir(mode=0o700)
        materialized: dict[str, Path] = {}
        try:
            for role, artifact in representation_artifacts:
                if role not in DEPENDENCY_REPRESENTATION_ROLES:
                    continue
                member = _validate_member_name(
                    artifact.bundle_member or artifact.filename
                )
                target = validation_root / member
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise StorageError(
                        "source_bundle_name_collision",
                        "The provider source bundle contains duplicate relative paths.",
                        status_code=500,
                    )
                shutil.copyfile(self._verified_artifact_blob(artifact), target)
                materialized[artifact.artifact_id] = target

            package_files = tuple(materialized.values())
            for role, artifact in representation_artifacts:
                if role not in PROCESSABLE_REPRESENTATION_ROLES:
                    continue
                source_path = materialized.get(artifact.artifact_id)
                if source_path is None:
                    raise StorageError(
                        "source_bundle_incomplete",
                        "A runnable provider representation is absent from its dependency closure.",
                        status_code=500,
                    )
                suffix = source_path.suffix.lower()
                if suffix in {".3mf", ".usdz"}:
                    self._validate_archive(
                        source_path,
                        suffix=suffix,
                        requested_entrypoint=None,
                    )
                    self._validate_archive_dependency_closure(
                        source_path,
                        suffix=suffix,
                        selected_entrypoint=None,
                    )
                else:
                    validate_materialized_source_dependencies(
                        source_path,
                        package_root=validation_root,
                        package_files=package_files,
                    )
        except SourceDependencyError as exc:
            raise StorageError(
                "unbound_source_dependency",
                (
                    "Provider geometry dependencies must be self-contained or "
                    "bound inside the immutable source package."
                ),
                status_code=422,
            ) from exc
        finally:
            shutil.rmtree(validation_root, ignore_errors=True)


class JobStore:
    """Atomic JSON job records that survive process restarts."""

    def __init__(
        self,
        storage: WorkspaceStorage,
        *,
        owner_instance_id: str,
        owner_lease_timeout_seconds: float = 60.0,
    ) -> None:
        self.storage = storage
        self.owner_instance_id = owner_instance_id
        self.owner_lease_timeout_seconds = owner_lease_timeout_seconds
        self._lock = threading.RLock()
        self.heartbeat()

    def create(
        self,
        kind: JobKind,
        *,
        provider_id: str | None = None,
        source_id: str | None = None,
    ) -> JobRecord:
        now = utc_now()
        record = JobRecord(
            job_id=f"job_{uuid.uuid4().hex}",
            kind=kind,
            status="queued",
            created_at=now,
            updated_at=now,
            provider_id=provider_id,
            source_id=source_id,
        )
        self._write(record)
        return record

    def get(self, job_id: str) -> JobRecord:
        if not _JOB_ID_RE.fullmatch(job_id):
            raise StorageError(
                "job_not_found", "Geometry job was not found.", status_code=404
            )
        path = self.storage.jobs_dir / f"{job_id}.json"
        try:
            job, _owner_instance_id = self._read(path)
            return job
        except FileNotFoundError as exc:
            raise StorageError(
                "job_not_found",
                "Geometry job was not found.",
                status_code=404,
            ) from exc

    def running(self, job: JobRecord) -> JobRecord:
        return self._replace(job, status="running")

    def succeed(self, job: JobRecord, result: dict[str, Any]) -> JobRecord:
        return self._replace(job, status="succeeded", result=result, error=None)

    def fail(
        self,
        job: JobRecord,
        *,
        code: str,
        message: str,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> JobRecord:
        return self._replace(
            job,
            status="failed",
            result=result,
            error=ServiceError(
                code=code,
                message=message,
                retryable=retryable,
                details=details or {},
            ),
        )

    def heartbeat(self, *, observed_at: datetime | None = None) -> None:
        """Renew this service instance's durable ownership lease."""

        self.storage.initialize()
        timestamp = _normalized_utc(observed_at or utc_now())
        payload = {
            "owner_instance_id": self.owner_instance_id,
            "observed_at": timestamp.isoformat(),
        }
        with self._lock:
            _atomic_write_json(self._owner_lease_path(self.owner_instance_id), payload)

    def recover_interrupted(
        self,
        *,
        now: datetime | None = None,
        include_current_owner: bool = True,
    ) -> int:
        recovery_time = _normalized_utc(now or utc_now())
        active_owners = self._active_owner_ids(recovery_time)
        recovered = 0
        for path in self.storage.jobs_dir.glob("job_*.json"):
            try:
                job, owner_instance_id = self._read(path)
            except Exception:
                continue
            if job.status not in {"queued", "running"}:
                continue
            if owner_instance_id == self.owner_instance_id:
                if not include_current_owner:
                    continue
            elif owner_instance_id in active_owners:
                continue
            self.fail(
                job,
                code="service_restarted",
                message="The service restarted before this synchronous job completed.",
                retryable=True,
            )
            recovered += 1
        return recovered

    def _active_owner_ids(self, now: datetime) -> set[str]:
        cutoff = now - timedelta(seconds=self.owner_lease_timeout_seconds)
        active: set[str] = set()
        for path in self.storage.job_owners_dir.glob("owner_*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                owner_id = payload["owner_instance_id"]
                observed_at = datetime.fromisoformat(payload["observed_at"])
                if not isinstance(owner_id, str):
                    continue
                if _normalized_utc(observed_at) >= cutoff:
                    active.add(owner_id)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return active

    def _owner_lease_path(self, owner_instance_id: str) -> Path:
        digest = hashlib.sha256(owner_instance_id.encode("utf-8")).hexdigest()
        return self.storage.job_owners_dir / f"owner_{digest}.json"

    def _replace(self, job: JobRecord, **updates: Any) -> JobRecord:
        updated = job.model_copy(update={"updated_at": utc_now(), **updates})
        self._write(updated)
        return updated

    def _write(self, record: JobRecord) -> None:
        self.storage.initialize()
        path = self.storage.jobs_dir / f"{record.job_id}.json"
        payload = record.model_dump(mode="json")
        payload[_JOB_OWNER_KEY] = self.owner_instance_id
        with self._lock:
            _atomic_write_json(path, payload)

    @staticmethod
    def _read(path: Path) -> tuple[JobRecord, str | None]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Geometry job record must be a JSON object")
        owner_instance_id = payload.pop(_JOB_OWNER_KEY, None)
        if owner_instance_id is not None and not isinstance(owner_instance_id, str):
            raise ValueError("Geometry job owner must be a string")
        return JobRecord.model_validate(payload), owner_instance_id


def _validate_filename(raw: str | None) -> str:
    if not raw or raw in {".", ".."}:
        raise StorageError(
            "invalid_filename", "Upload filename is required.", status_code=422
        )
    if "/" in raw or "\\" in raw or "\x00" in raw or Path(raw).name != raw:
        raise StorageError(
            "unsafe_filename",
            "Upload filename must be a plain filename without path components.",
            status_code=422,
        )
    if any(ord(character) < 32 for character in raw):
        raise StorageError(
            "unsafe_filename",
            "Upload filename contains control characters.",
            status_code=422,
        )
    return raw


def _validate_extension(suffix: str, role: ArtifactRole) -> None:
    allowed = (
        GEOMETRY_SOURCE_EXTENSIONS
        if role == "geometry_source"
        else REFERENCE_IMAGE_EXTENSIONS
    )
    if suffix not in allowed:
        raise StorageError(
            "source_type_forbidden",
            f"File type is not allowed for {role}: {suffix or '<none>'}",
            status_code=415,
        )


def _normalize_expected_digest(raw: str | None) -> str | None:
    if raw is None:
        return None
    normalized = raw.removeprefix("sha256:").lower()
    if not _DIGEST_RE.fullmatch(normalized):
        raise StorageError(
            "invalid_expected_digest",
            "X-Content-SHA256 must contain a 64-character SHA-256 digest.",
            status_code=422,
        )
    return normalized


def _validate_member_name(raw: str) -> str:
    if not raw or "\x00" in raw or "\\" in raw:
        raise StorageError(
            "archive_traversal_forbidden",
            "Archive contains an unsafe member name.",
            status_code=422,
        )
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise StorageError(
            "archive_traversal_forbidden",
            "Archive contains a path traversal member.",
            status_code=422,
        )
    return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
