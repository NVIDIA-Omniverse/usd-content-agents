# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import io
import json
import logging
import mimetypes
import os
import random
import stat
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aioboto3
from aiobotocore.config import AioConfig
from botocore.exceptions import ClientError
from cachetools import TTLCache
from world_understanding.utils.artifacts import (
    confined_atomic_writer,
    is_pipeline_temp_path,
    iter_open_regular_files,
    open_confined_directory,
    open_regular_file_no_follow,
    prune_confined_snapshot,
    remove_confined_tree,
    validated_artifact_relative_key,
    validated_s3_object_suffix,
)
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)

from .base import (
    METADATA_KEY,
    CompletedSessionSnapshot,
    SessionGeneration,
    SessionGenerationConflictError,
    SessionGenerationOwnershipError,
    SessionNotCompletedError,
    SessionStore,
)
from .config import StorageConfig

logger = logging.getLogger(__name__)

# Cache key for sessions list (single key since we only cache one list per store)
_SESSIONS_CACHE_KEY = "sessions"
_SESSION_PROTOCOL = "physics-session-generation.v1"
_MANIFEST_PROTOCOL = "physics-session-artifact-manifest.v1"
_GENERATION_ROOT = ".generations"
_GENERATION_HISTORY_KEY = "generation_history"
_LOCAL_SNAPSHOT_DIRECTORY = ".pipeline_temp"
_LOCAL_SNAPSHOT_MARKER = "s3-snapshot-generation.json"
_LOCAL_SNAPSHOT_MARKER_PROTOCOL = "physics-local-s3-snapshot.v3"
_LOCAL_SNAPSHOT_MARKER_PROTOCOL_V2 = "physics-local-s3-snapshot.v2"
_MAX_CAS_ATTEMPTS = 16
_GENERATION_LEASE_SECONDS = 300
_GENERATION_LEASE_RENEWAL_SECONDS = 60
_CONDITIONAL_WRITE_ERRORS = {
    "409",
    "412",
    "ConditionalRequestConflict",
    "PreconditionFailed",
}


def _validated_snapshot_prefix(value: object) -> str:
    """Return one canonical logical snapshot prefix."""
    if not isinstance(value, str):
        raise ValueError("Snapshot prefix must be a string")
    if value == "":
        return value
    validated_artifact_relative_key(value[:-1] if value.endswith("/") else value)
    return value


def _normalize_snapshot_prefixes(prefixes: Sequence[str]) -> tuple[str, ...]:
    """Deduplicate snapshot coverage and remove prefixes narrowed by a parent."""
    normalized: list[str] = []
    for raw_prefix in prefixes:
        prefix = _validated_snapshot_prefix(raw_prefix)
        if any(prefix.startswith(existing) for existing in normalized):
            continue
        normalized = [
            existing for existing in normalized if not existing.startswith(prefix)
        ]
        normalized.append(prefix)
    return tuple(normalized)


def _snapshot_prefix_intersections(
    requested_prefixes: Sequence[str],
    complete_prefixes: Sequence[str],
) -> tuple[str, ...]:
    """Return only requested regions known to be complete in the manifest."""
    intersections: list[str] = []
    for requested in requested_prefixes:
        for complete in complete_prefixes:
            if requested.startswith(complete):
                intersections.append(requested)
            elif complete.startswith(requested):
                intersections.append(complete)
    return _normalize_snapshot_prefixes(intersections)


def _updated_snapshot_prefixes(
    previous_prefixes: Sequence[str],
    selected_prefixes: Sequence[str],
) -> tuple[str, ...]:
    """Replace overlapping coverage while retaining disjoint complete regions."""
    selected = _normalize_snapshot_prefixes(selected_prefixes)
    retained = [
        previous
        for previous in _normalize_snapshot_prefixes(previous_prefixes)
        if not any(
            previous.startswith(current) or current.startswith(previous)
            for current in selected
        )
    ]
    return _normalize_snapshot_prefixes((*retained, *selected))


@dataclass(frozen=True)
class _LocalSnapshotMarker:
    """Local ownership plus the publication against which it was reconciled."""

    protocol: str
    local_generation: SessionGeneration
    publication_generation: SessionGeneration | None
    prefixes: tuple[str, ...]


def _read_local_snapshot_marker(
    root_descriptor: int,
) -> _LocalSnapshotMarker | None:
    """Read local ownership and its independently pinned publication."""
    try:
        directory_descriptor = os.open(
            _LOCAL_SNAPSHOT_DIRECTORY,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
    except OSError:
        return None
    try:
        marker_descriptor = os.open(
            _LOCAL_SNAPSHOT_MARKER,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_descriptor,
        )
        try:
            if not stat.S_ISREG(os.fstat(marker_descriptor).st_mode):
                return None
            with os.fdopen(marker_descriptor, "rb") as source:
                marker_descriptor = -1
                raw_marker = source.read(4097)
        finally:
            if marker_descriptor >= 0:
                os.close(marker_descriptor)
    except OSError:
        return None
    finally:
        os.close(directory_descriptor)
    if len(raw_marker) > 4096:
        return None
    try:
        marker = json.loads(raw_marker)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    protocol = marker.get("protocol") if isinstance(marker, dict) else None
    generation = marker.get("generation") if isinstance(marker, dict) else None
    owner_id = marker.get("owner_id") if isinstance(marker, dict) else None
    raw_prefixes = marker.get("prefixes") if isinstance(marker, dict) else None
    if (
        not isinstance(marker, dict)
        or protocol
        not in {
            _LOCAL_SNAPSHOT_MARKER_PROTOCOL,
            _LOCAL_SNAPSHOT_MARKER_PROTOCOL_V2,
        }
        or not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 1
        or not isinstance(owner_id, str)
        or not owner_id
        or not isinstance(raw_prefixes, list)
    ):
        return None
    try:
        prefixes = _normalize_snapshot_prefixes(raw_prefixes)
    except ValueError:
        return None
    local_generation = SessionGeneration(generation, owner_id)
    if protocol == _LOCAL_SNAPSHOT_MARKER_PROTOCOL_V2:
        publication_generation: SessionGeneration | None = local_generation
    else:
        published_generation = marker.get("publication_generation")
        published_owner_id = marker.get("publication_owner_id")
        if published_generation is None and published_owner_id is None:
            publication_generation = None
        elif (
            isinstance(published_generation, int)
            and not isinstance(published_generation, bool)
            and published_generation >= 1
            and isinstance(published_owner_id, str)
            and published_owner_id
        ):
            publication_generation = SessionGeneration(
                published_generation,
                published_owner_id,
            )
        else:
            return None
    return _LocalSnapshotMarker(
        protocol,
        local_generation,
        publication_generation,
        prefixes,
    )


def _write_local_snapshot_generation(
    root_descriptor: int,
    generation: SessionGeneration,
    prefixes: Sequence[str],
    *,
    publication_generation: SessionGeneration | None,
) -> None:
    """Atomically record local ownership and its pinned publication."""
    marker = json.dumps(
        {
            "protocol": _LOCAL_SNAPSHOT_MARKER_PROTOCOL,
            "generation": generation.generation,
            "owner_id": generation.owner_id,
            "publication_generation": (
                publication_generation.generation
                if publication_generation is not None
                else None
            ),
            "publication_owner_id": (
                publication_generation.owner_id
                if publication_generation is not None
                else None
            ),
            "prefixes": list(_normalize_snapshot_prefixes(prefixes)),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        os.mkdir(_LOCAL_SNAPSHOT_DIRECTORY, mode=0o700, dir_fd=root_descriptor)
    except FileExistsError:
        pass
    directory_descriptor = os.open(
        _LOCAL_SNAPSHOT_DIRECTORY,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=root_descriptor,
    )
    temporary_name = f".{_LOCAL_SNAPSHOT_MARKER}.{uuid.uuid4().hex}.tmp"
    try:
        marker_descriptor = os.open(
            temporary_name,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        try:
            with os.fdopen(marker_descriptor, "wb") as destination:
                marker_descriptor = -1
                destination.write(marker)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(
                temporary_name,
                _LOCAL_SNAPSHOT_MARKER,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
            )
            os.fsync(directory_descriptor)
        finally:
            if marker_descriptor >= 0:
                os.close(marker_descriptor)
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
    finally:
        os.close(directory_descriptor)


_ACTIVE_GENERATIONS: ContextVar[dict[tuple[object, str], SessionGeneration] | None] = (
    ContextVar(
        "physics_s3_active_generations",
        default=None,
    )
)

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from types_aiobotocore_s3 import S3Client  # type: ignore[import-untyped]


class S3SessionStore(SessionStore):
    """S3-compatible session storage backend (works with AWS S3, MinIO, etc.)."""

    # Default cache TTL in seconds

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        use_path_style: bool = True,
        create_bucket_if_missing: bool = True,
        presign_by_default: bool = True,
        sessions_cache_ttl: int = StorageConfig().s3_sessions_cache_ttl,
        generation_retention: int = StorageConfig().s3_generation_retention,
    ) -> None:
        if not bucket:
            raise ValueError("bucket is required for S3SessionStore")
        if generation_retention < 1:
            raise ValueError("generation_retention must be at least 1")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._region = region
        self._endpoint_url = endpoint_url
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token
        self._use_path_style = use_path_style
        self._create_bucket_if_missing = create_bucket_if_missing
        self.presign_by_default = presign_by_default
        self._generation_retention = generation_retention
        self._session = aioboto3.Session()
        self._bucket_ensured = False
        self._conditional_writes_verified = False

        # TTL cache for sessions list (maxsize=1 since we only cache one list)
        # TTLCache automatically expires entries after ttl seconds
        ttl = sessions_cache_ttl
        self._sessions_cache: TTLCache[str, list[str]] = TTLCache(maxsize=1, ttl=ttl)
        self._cache_lock = asyncio.Lock()

    @property
    def kind(self) -> str:
        return "s3"

    async def verify_capabilities(self) -> None:
        """Eagerly run the bucket and conditional-write startup checks."""
        async with self._client():
            pass

    @classmethod
    def from_config(cls, config: StorageConfig) -> S3SessionStore:
        """Create an S3SessionStore from a StorageConfig.

        Args:
            config: StorageConfig with S3 settings

        Returns:
            Configured S3SessionStore instance

        Raises:
            ValueError: If config.s3_bucket is not set
        """
        if not config.s3_bucket:
            raise ValueError(
                "s3_bucket is required in StorageConfig for S3SessionStore"
            )

        return cls(
            bucket=config.s3_bucket,
            prefix=config.s3_prefix,
            region=config.s3_region,
            endpoint_url=config.s3_endpoint_url,
            access_key_id=config.s3_access_key_id,
            secret_access_key=config.s3_secret_access_key,
            session_token=config.s3_session_token,
            use_path_style=config.s3_use_path_style,
            create_bucket_if_missing=config.s3_create_bucket,
            presign_by_default=config.s3_presign,
            sessions_cache_ttl=config.s3_sessions_cache_ttl,
            generation_retention=config.s3_generation_retention,
        )

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[S3Client]:
        """Get an async S3 client."""
        cfg = AioConfig(
            s3={"addressing_style": "path" if self._use_path_style else "virtual"}
        )
        async with self._session.client(
            "s3",
            region_name=self._region,
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._access_key_id,
            aws_secret_access_key=self._secret_access_key,
            aws_session_token=self._session_token,
            config=cfg,
        ) as client:
            if self._create_bucket_if_missing and not self._bucket_ensured:
                await self._ensure_bucket(client)
                self._bucket_ensured = True
            if not self._conditional_writes_verified:
                await self._verify_conditional_writes(client)
                self._conditional_writes_verified = True
            yield client

    async def _ensure_bucket(self, client: S3Client) -> None:
        """Ensure the bucket exists, creating it if necessary."""
        try:
            await client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchBucket"):
                await client.create_bucket(Bucket=self.bucket)
            else:
                raise

    async def _verify_conditional_writes(self, client: S3Client) -> None:
        """Fail closed unless the configured backend enforces S3 CAS headers."""
        capability_key = (
            f"{self.prefix}/.capability/conditional-{uuid.uuid4().hex}"
            if self.prefix
            else f".capability/conditional-{uuid.uuid4().hex}"
        )
        try:
            created = await client.put_object(
                Bucket=self.bucket,
                Key=capability_key,
                Body=b"probe",
                IfNoneMatch="*",
            )
            etag = created.get("ETag")
            if not isinstance(etag, str) or not etag:
                raise RuntimeError(
                    "S3 endpoint did not return an ETag for conditional-write probe"
                )

            for kwargs in (
                {"IfNoneMatch": "*"},
                {"IfMatch": '"physics-agent-invalid-etag"'},
            ):
                try:
                    await client.put_object(
                        Bucket=self.bucket,
                        Key=capability_key,
                        Body=b"must-not-overwrite",
                        **kwargs,
                    )
                except ClientError as error:
                    if self._is_conditional_conflict(error):
                        continue
                    raise
                raise RuntimeError(
                    "S3 endpoint does not enforce PutObject IfMatch/IfNoneMatch; "
                    "generation-aware multi-replica storage is unsafe"
                )
        finally:
            try:
                await client.delete_object(Bucket=self.bucket, Key=capability_key)
            except Exception:  # noqa: BLE001 - best-effort probe cleanup
                logger.warning("Could not remove S3 conditional-write probe object")

    @staticmethod
    async def _cas_retry_delay(attempt: int) -> None:
        delay = min(0.25, 0.005 * (2**attempt))
        await asyncio.sleep(delay * (0.5 + random.random()))

    def _key(self, session_id: str, key: str) -> str:
        base = (
            f"{self.prefix}/sessions/{session_id}"
            if self.prefix
            else f"sessions/{session_id}"
        )
        return f"{base}/{key}".lstrip("/")

    @staticmethod
    def _is_not_found(error: ClientError) -> bool:
        return error.response.get("Error", {}).get("Code", "") in {
            "404",
            "NoSuchKey",
        }

    @staticmethod
    def _is_conditional_conflict(error: ClientError) -> bool:
        return (
            error.response.get("Error", {}).get("Code", "") in _CONDITIONAL_WRITE_ERRORS
        )

    @staticmethod
    def _generation_state(metadata: Mapping[str, Any]) -> str:
        status = metadata.get("status")
        if status in {"pending", "running", "cancelling"}:
            return "active"
        if status in {"completed", "failed", "cancelled"}:
            return "terminal"
        if status == "ready":
            return "idle"
        return "idle" if not metadata else "starting"

    @staticmethod
    def _lease_expiry() -> str:
        return (
            datetime.now(UTC) + timedelta(seconds=_GENERATION_LEASE_SECONDS)
        ).isoformat()

    @staticmethod
    def _lease_is_expired(envelope: Mapping[str, Any]) -> bool:
        raw_expiry = envelope.get("generation_lease_expires_at")
        if not isinstance(raw_expiry, str):
            return True
        try:
            expiry = datetime.fromisoformat(raw_expiry)
        except ValueError:
            return True
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        return expiry <= datetime.now(UTC)

    @staticmethod
    def _lease_needs_renewal(envelope: Mapping[str, Any]) -> bool:
        raw_expiry = envelope.get("generation_lease_expires_at")
        if not isinstance(raw_expiry, str):
            return True
        try:
            expiry = datetime.fromisoformat(raw_expiry)
        except ValueError:
            return True
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        return expiry <= datetime.now(UTC) + timedelta(
            seconds=_GENERATION_LEASE_RENEWAL_SECONDS
        )

    @staticmethod
    def _generation_prefix(generation: SessionGeneration) -> str:
        return f"{_GENERATION_ROOT}/{generation.generation}-{generation.owner_id}"

    def _active_generation(self, session_id: str) -> SessionGeneration | None:
        return (_ACTIVE_GENERATIONS.get() or {}).get((self, session_id))

    def _activate_generation(
        self, session_id: str, generation: SessionGeneration
    ) -> None:
        active = dict(_ACTIVE_GENERATIONS.get() or {})
        active[(self, session_id)] = generation
        _ACTIVE_GENERATIONS.set(active)

    @staticmethod
    def _new_envelope(
        generation: SessionGeneration,
        *,
        metadata: Mapping[str, Any] | None = None,
        publication: Mapping[str, Any] | None = None,
        history: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        return {
            "protocol": _SESSION_PROTOCOL,
            "generation": generation.generation,
            "owner_id": generation.owner_id,
            "generation_state": "starting",
            "generation_started_at": now,
            "generation_lease_expires_at": S3SessionStore._lease_expiry(),
            "metadata": dict(metadata or {}),
            "artifact_publication": dict(publication) if publication else None,
            "cancellation_requested": False,
            _GENERATION_HISTORY_KEY: [dict(item) for item in history],
            "deleted": False,
        }

    @staticmethod
    def _decode_envelope(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid S3 session document")
        if payload.get("protocol") == _SESSION_PROTOCOL:
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                raise RuntimeError("Invalid generation-aware session metadata")
            return payload

        # Upgrade legacy canonical session.json documents in memory. The next
        # begin_generation CAS persists the generation-aware envelope.
        return {
            "protocol": _SESSION_PROTOCOL,
            "generation": 0,
            "owner_id": "",
            "generation_state": S3SessionStore._generation_state(payload),
            "generation_started_at": payload.get("created_at"),
            "metadata": payload,
            "artifact_publication": None,
            "cancellation_requested": False,
            _GENERATION_HISTORY_KEY: [],
            "deleted": False,
            "legacy": True,
        }

    async def _read_envelope(
        self, client: S3Client, session_id: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            response = await client.get_object(
                Bucket=self.bucket,
                Key=self._key(session_id, METADATA_KEY),
            )
        except ClientError as error:
            if self._is_not_found(error):
                return None, None
            raise
        body = await response["Body"].read()
        return self._decode_envelope(json.loads(body)), response.get("ETag")

    async def _write_envelope(
        self,
        client: S3Client,
        session_id: str,
        envelope: Mapping[str, Any],
        *,
        etag: str | None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": self._key(session_id, METADATA_KEY),
            "Body": json.dumps(envelope, separators=(",", ":")).encode("utf-8"),
            "ContentType": "application/json",
        }
        if etag is None:
            kwargs["IfNoneMatch"] = "*"
        else:
            kwargs["IfMatch"] = etag
        await client.put_object(**kwargs)

    async def _write_generation_descriptor(
        self,
        client: S3Client,
        session_id: str,
        generation: SessionGeneration,
        *,
        base_publication: Mapping[str, Any] | None,
    ) -> None:
        descriptor = {
            "protocol": _SESSION_PROTOCOL,
            "generation": generation.generation,
            "owner_id": generation.owner_id,
            "started_at": datetime.now(UTC).isoformat(),
            "base_publication": (
                dict(base_publication) if base_publication is not None else None
            ),
        }
        await client.put_object(
            Bucket=self.bucket,
            Key=self._key(
                session_id,
                f"{self._generation_prefix(generation)}/generation.json",
            ),
            Body=json.dumps(descriptor, separators=(",", ":")).encode("utf-8"),
            ContentType="application/json",
            IfNoneMatch="*",
        )

    async def _best_effort_write_generation_descriptor(
        self,
        client: S3Client,
        session_id: str,
        generation: SessionGeneration,
        *,
        base_publication: Mapping[str, Any] | None,
    ) -> None:
        try:
            await self._write_generation_descriptor(
                client,
                session_id,
                generation,
                base_publication=base_publication,
            )
        except Exception:  # noqa: BLE001 - descriptor is diagnostic only
            log_durable_failure(
                logger,
                "physics_s3_generation_descriptor_failed",
                phase=FailurePhase.PERSISTENCE_VERIFICATION,
                retryable=True,
            )

    async def _snapshot_legacy_publication(
        self,
        client: S3Client,
        session_id: str,
        generation: SessionGeneration,
    ) -> dict[str, Any]:
        """Copy one terminal legacy session into an immutable base snapshot.

        A generation-aware manifest disables canonical-key fallback. Preserve
        every readable legacy artifact before the first envelope CAS so a
        partial rerun cannot make disjoint outputs disappear.
        """
        generation_prefix = self._generation_prefix(generation)
        migration_id = uuid.uuid4().hex
        migration_prefix = f"{generation_prefix}/legacy/{migration_id}"
        session_prefix = self._key(session_id, "")
        artifacts: dict[str, str] = {}
        paginator = client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=session_prefix,
        ):
            for item in page.get("Contents", []):
                physical_key = item.get("Key")
                if not isinstance(physical_key, str):
                    continue
                relative_key = physical_key[len(session_prefix) :]
                if (
                    not relative_key
                    or relative_key == METADATA_KEY
                    or relative_key.startswith(".")
                    or is_pipeline_temp_path(relative_key)
                ):
                    continue
                try:
                    relative_key = validated_artifact_relative_key(relative_key)
                except ValueError as error:
                    raise RuntimeError(
                        "Invalid canonical legacy session artifact"
                    ) from error
                migrated_key = f"{migration_prefix}/artifacts/{relative_key}"
                await client.copy_object(
                    Bucket=self.bucket,
                    Key=self._key(session_id, migrated_key),
                    CopySource={"Bucket": self.bucket, "Key": physical_key},
                )
                artifacts[relative_key] = migrated_key

        publication_id = uuid.uuid4().hex
        publication_prefix = f"{generation_prefix}/publications/{publication_id}"
        manifest_key = f"{publication_prefix}/manifest.json"
        manifest = {
            "protocol": _MANIFEST_PROTOCOL,
            "generation": generation.generation,
            "owner_id": generation.owner_id,
            "publication_id": publication_id,
            "created_at": datetime.now(UTC).isoformat(),
            "complete": True,
            "artifacts": artifacts,
            "snapshot_prefixes": [""],
        }
        await client.put_object(
            Bucket=self.bucket,
            Key=self._key(session_id, manifest_key),
            Body=json.dumps(manifest, separators=(",", ":")).encode("utf-8"),
            ContentType="application/json",
            IfNoneMatch="*",
        )
        return {
            "generation": generation.generation,
            "owner_id": generation.owner_id,
            "generation_prefix": generation_prefix,
            "manifest_key": manifest_key,
            "published_at": datetime.now(UTC).isoformat(),
        }

    async def init_session(self, session_id: str) -> None:
        generation = SessionGeneration(generation=1, owner_id=uuid.uuid4().hex)
        envelope = self._new_envelope(generation)
        async with self._client() as client:
            try:
                await self._write_envelope(
                    client,
                    session_id,
                    envelope,
                    etag=None,
                )
            except ClientError as error:
                if self._is_conditional_conflict(error):
                    raise SessionGenerationConflictError(
                        f"Session already exists: {session_id}"
                    ) from error
                raise
            await self._best_effort_write_generation_descriptor(
                client,
                session_id,
                generation,
                base_publication=None,
            )
        self._activate_generation(session_id, generation)

        async with self._cache_lock:
            if _SESSIONS_CACHE_KEY in self._sessions_cache:
                cached = self._sessions_cache[_SESSIONS_CACHE_KEY]
                if session_id not in cached:
                    cached.append(session_id)

    async def begin_generation(self, session_id: str) -> SessionGeneration:
        """Atomically fence one new run or rerun across all service replicas."""
        async with self._client() as client:
            for attempt in range(_MAX_CAS_ATTEMPTS):
                envelope, etag = await self._read_envelope(client, session_id)
                if envelope is None or etag is None or envelope.get("deleted") is True:
                    raise FileNotFoundError(session_id)
                generation_is_active = envelope.get("generation_state") in {
                    "starting",
                    "active",
                }
                # Pre-protocol workers do not renew a lease and still write the
                # canonical session document unconditionally. During a rolling
                # upgrade, never treat their missing lease as abandoned: wait
                # for the legacy worker to publish a terminal state before a
                # generation-aware replica can take ownership.
                legacy_generation_is_active = bool(
                    envelope.get("legacy") and generation_is_active
                )
                if legacy_generation_is_active or (
                    generation_is_active and not self._lease_is_expired(envelope)
                ):
                    raise SessionGenerationConflictError(
                        f"A session generation is already active for {session_id}"
                    )

                current_generation = int(envelope.get("generation") or 0)
                current_owner = str(envelope.get("owner_id") or "")
                generation = SessionGeneration(
                    generation=current_generation + 1,
                    owner_id=uuid.uuid4().hex,
                )
                raw_publication = envelope.get("artifact_publication")
                publication = (
                    dict(raw_publication)
                    if isinstance(raw_publication, Mapping)
                    else None
                )
                if envelope.get("legacy"):
                    publication = await self._snapshot_legacy_publication(
                        client,
                        session_id,
                        generation,
                    )
                published_artifacts = (
                    await self._artifacts_for_publication(
                        client,
                        session_id,
                        publication,
                    )
                    if isinstance(publication, Mapping)
                    else {}
                )
                history = list(envelope.get(_GENERATION_HISTORY_KEY) or [])
                if current_generation > 0 and current_owner:
                    history.append(
                        {
                            "generation": current_generation,
                            "owner_id": current_owner,
                            "state": (
                                "abandoned"
                                if generation_is_active
                                else envelope.get("generation_state")
                            ),
                            "started_at": envelope.get("generation_started_at"),
                            "status": envelope.get("metadata", {}).get("status"),
                            "publication": publication,
                        }
                    )
                history = history[-self._generation_retention :]
                replacement = self._new_envelope(
                    generation,
                    metadata=envelope["metadata"],
                    publication=publication,
                    history=history,
                )
                try:
                    await self._write_envelope(
                        client,
                        session_id,
                        replacement,
                        etag=etag,
                    )
                except ClientError as error:
                    if self._is_conditional_conflict(error):
                        await self._cas_retry_delay(attempt)
                        continue
                    raise
                self._activate_generation(session_id, generation)
                await self._best_effort_write_generation_descriptor(
                    client,
                    session_id,
                    generation,
                    base_publication=publication,
                )
                protected_generations = self._referenced_generation_numbers(
                    published_artifacts
                )
                published_generation = (
                    publication.get("generation")
                    if isinstance(publication, Mapping)
                    else None
                )
                if isinstance(published_generation, int):
                    protected_generations.add(published_generation)
                await self._best_effort_generation_gc(
                    client,
                    session_id,
                    current_generation=generation.generation,
                    protected_generations=protected_generations,
                )
                return generation
        raise SessionGenerationConflictError(
            f"Session generation remained contended for {session_id}"
        )

    async def adopt_local_generation(
        self,
        session_id: str,
        local_session_dir: str,
        generation: SessionGeneration,
    ) -> None:
        """Transfer proven local base coverage to a newly claimed generation.

        Reruns deliberately reuse cached intermediates. Coverage transfers only
        when the local marker matches the immutable publication retained as the
        new generation's base. A stale replica therefore starts with no trusted
        regions, while an empty working directory is safe to own in full.
        """
        async with self._client() as client:
            envelope, _etag = await self._read_envelope(client, session_id)
        if (
            envelope is None
            or envelope.get("deleted") is True
            or envelope.get("generation") != generation.generation
            or envelope.get("owner_id") != generation.owner_id
        ):
            raise SessionGenerationOwnershipError(
                f"Stale generation cannot adopt local cache for {session_id}"
            )
        publication = envelope.get("artifact_publication")
        base_generation: SessionGeneration | None = None
        if isinstance(publication, Mapping):
            published_generation = publication.get("generation")
            published_owner = publication.get("owner_id")
            if (
                isinstance(published_generation, int)
                and not isinstance(published_generation, bool)
                and published_generation >= 1
                and isinstance(published_owner, str)
                and published_owner
            ):
                base_generation = SessionGeneration(
                    published_generation,
                    published_owner,
                )
        predecessor_generation: SessionGeneration | None = None
        history = envelope.get(_GENERATION_HISTORY_KEY)
        if isinstance(history, list) and history:
            predecessor = history[-1]
            predecessor_number = (
                predecessor.get("generation")
                if isinstance(predecessor, Mapping)
                else None
            )
            predecessor_owner = (
                predecessor.get("owner_id")
                if isinstance(predecessor, Mapping)
                else None
            )
            if (
                isinstance(predecessor_number, int)
                and not isinstance(predecessor_number, bool)
                and predecessor_number == generation.generation - 1
                and isinstance(predecessor_owner, str)
                and predecessor_owner
                and predecessor.get("publication") == publication
            ):
                predecessor_generation = SessionGeneration(
                    predecessor_number,
                    predecessor_owner,
                )

        with open_confined_directory(
            local_session_dir,
            create=True,
        ) as root_descriptor:
            local_marker = _read_local_snapshot_marker(root_descriptor)
            artifacts = iter_open_regular_files(root_descriptor)
            try:
                local_is_empty = next(artifacts, None) is None
            finally:
                artifacts.close()
            reconciled_prefixes: tuple[str, ...]
            if local_is_empty:
                reconciled_prefixes = ("",)
            elif (
                local_marker is not None
                and local_marker.local_generation
                in {base_generation, predecessor_generation}
                and (
                    local_marker.publication_generation == base_generation
                    or (
                        local_marker.protocol == _LOCAL_SNAPSHOT_MARKER_PROTOCOL_V2
                        and local_marker.local_generation == predecessor_generation
                    )
                )
            ):
                reconciled_prefixes = local_marker.prefixes
            else:
                reconciled_prefixes = ()
            _write_local_snapshot_generation(
                root_descriptor,
                generation,
                reconciled_prefixes,
                publication_generation=base_generation,
            )

    async def owns_active_generation(self, session_id: str) -> bool:
        """Return whether this task still owns the session generation it started."""
        generation = self._active_generation(session_id)
        if generation is None:
            return True
        async with self._client() as client:
            for attempt in range(_MAX_CAS_ATTEMPTS):
                envelope, etag = await self._read_envelope(client, session_id)
                owns = bool(
                    envelope
                    and not envelope.get("deleted")
                    and envelope.get("generation") == generation.generation
                    and envelope.get("owner_id") == generation.owner_id
                )
                if not owns or envelope is None or etag is None:
                    return False
                if not self._lease_needs_renewal(envelope):
                    return True
                replacement = dict(envelope)
                replacement["generation_lease_expires_at"] = self._lease_expiry()
                try:
                    await self._write_envelope(
                        client,
                        session_id,
                        replacement,
                        etag=etag,
                    )
                except ClientError as error:
                    if self._is_conditional_conflict(error):
                        await self._cas_retry_delay(attempt)
                        continue
                    raise
                return True
        raise RuntimeError(f"Generation lease remained contended: {session_id}")

    async def request_generation_cancellation(
        self,
        session_id: str,
        *,
        update_status: bool,
    ) -> bool:
        """Atomically bind a cancellation request to the current generation."""
        async with self._client() as client:
            for attempt in range(_MAX_CAS_ATTEMPTS):
                envelope, etag = await self._read_envelope(client, session_id)
                if envelope is None or envelope.get("deleted") is True:
                    return False
                if int(envelope.get("generation") or 0) != 0:
                    break

                # Legacy workers poll the canonical .cancel object. Preserve
                # that contract until their session reaches a terminal state,
                # and keep the legacy session.json shape so an old worker can
                # continue reading it during a rolling deployment.
                if envelope.get("generation_state") not in {"starting", "active"}:
                    await client.delete_object(
                        Bucket=self.bucket,
                        Key=self._key(session_id, ".cancel"),
                    )
                    return False
                await client.put_object(
                    Bucket=self.bucket,
                    Key=self._key(session_id, ".cancel"),
                    Body=b"",
                )
                if not update_status:
                    return True

                metadata = deepcopy(envelope["metadata"])
                metadata["status"] = "cancelling"
                metadata["updated_at"] = datetime.now(UTC).isoformat()
                try:
                    await client.put_object(
                        Bucket=self.bucket,
                        Key=self._key(session_id, METADATA_KEY),
                        Body=json.dumps(metadata, separators=(",", ":")).encode(
                            "utf-8"
                        ),
                        ContentType="application/json",
                        IfMatch=etag,
                    )
                except ClientError as error:
                    if self._is_conditional_conflict(error):
                        await self._cas_retry_delay(attempt)
                        continue
                    raise
                return True
            else:
                raise RuntimeError(
                    f"Legacy cancellation remained contended: {session_id}"
                )
        if (
            envelope is None
            or envelope.get("deleted") is True
            or int(envelope.get("generation") or 0) == 0
        ):
            return False
        current_generation = SessionGeneration(
            generation=int(envelope["generation"]),
            owner_id=str(envelope["owner_id"]),
        )
        bound_generation = self._active_generation(session_id)
        if bound_generation is not None and bound_generation != current_generation:
            return False
        if bound_generation is None:
            self._activate_generation(session_id, current_generation)
        try:
            return await self._set_generation_cancellation(
                session_id,
                requested=True,
                update_status=update_status,
                require_active=True,
            )
        except SessionGenerationOwnershipError:
            return False

    async def _set_generation_cancellation(
        self,
        session_id: str,
        *,
        requested: bool,
        update_status: bool = False,
        require_active: bool = False,
    ) -> bool:
        target_generation = self._active_generation(session_id)
        async with self._client() as client:
            for attempt in range(_MAX_CAS_ATTEMPTS):
                envelope, etag = await self._read_envelope(client, session_id)
                if envelope is None or etag is None or envelope.get("deleted") is True:
                    return False
                if int(envelope.get("generation") or 0) == 0:
                    return False
                current_generation = SessionGeneration(
                    generation=int(envelope["generation"]),
                    owner_id=str(envelope["owner_id"]),
                )
                if target_generation is None:
                    target_generation = current_generation
                elif current_generation != target_generation:
                    if self._active_generation(session_id) is not None:
                        raise SessionGenerationOwnershipError(
                            f"Stale generation cannot cancel session {session_id}"
                        )
                    return False
                metadata = envelope["metadata"]
                if require_active and metadata.get("status") not in {
                    "pending",
                    "running",
                    "cancelling",
                }:
                    return False
                replacement = dict(envelope)
                replacement["cancellation_requested"] = requested
                if update_status and requested:
                    replacement_metadata = deepcopy(metadata)
                    replacement_metadata["status"] = "cancelling"
                    replacement_metadata["updated_at"] = datetime.now(UTC).isoformat()
                    replacement["metadata"] = replacement_metadata
                    replacement["generation_state"] = "active"
                try:
                    await self._write_envelope(
                        client,
                        session_id,
                        replacement,
                        etag=etag,
                    )
                except ClientError as error:
                    if self._is_conditional_conflict(error):
                        await self._cas_retry_delay(attempt)
                        continue
                    raise
                return True
        raise RuntimeError(f"Cancellation remained contended: {session_id}")

    async def delete_session(self, session_id: str) -> None:
        await self._delete_session(session_id, require_terminal=False)

    async def delete_session_if_terminal(self, session_id: str) -> bool:
        """CAS-delete only while the current generation remains terminal."""
        return await self._delete_session(session_id, require_terminal=True)

    async def _delete_session(
        self,
        session_id: str,
        *,
        require_terminal: bool,
    ) -> bool:
        target_generation: SessionGeneration | None = None
        async with self._client() as client:
            for attempt in range(_MAX_CAS_ATTEMPTS):
                envelope, etag = await self._read_envelope(client, session_id)
                if envelope is None:
                    return False
                observed_generation = SessionGeneration(
                    generation=int(envelope.get("generation") or 0),
                    owner_id=str(envelope.get("owner_id") or ""),
                )
                if target_generation is None:
                    target_generation = observed_generation
                elif observed_generation != target_generation:
                    if require_terminal:
                        return False
                    raise SessionGenerationConflictError(
                        f"Session generation changed during deletion: {session_id}"
                    )
                generation = self._active_generation(session_id)
                if generation is not None and (
                    envelope.get("generation") != generation.generation
                    or envelope.get("owner_id") != generation.owner_id
                ):
                    raise SessionGenerationOwnershipError(
                        f"Stale generation cannot delete session {session_id}"
                    )
                if require_terminal and envelope.get("generation_state") not in {
                    "terminal",
                    "idle",
                }:
                    return False
                tombstone = dict(envelope)
                tombstone["deleted"] = True
                tombstone["generation_state"] = "deleted"
                try:
                    await self._write_envelope(
                        client,
                        session_id,
                        tombstone,
                        etag=etag,
                    )
                except ClientError as error:
                    if self._is_conditional_conflict(error):
                        await self._cas_retry_delay(attempt)
                        continue
                    raise
                break
            else:
                raise RuntimeError(f"Could not fence deletion for {session_id}")

            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self.bucket, Prefix=self._key(session_id, "")
            ):
                for obj in page.get("Contents", []):
                    await client.delete_object(Bucket=self.bucket, Key=obj["Key"])

        # Update cache: remove session from cached list if it exists
        async with self._cache_lock:
            if _SESSIONS_CACHE_KEY in self._sessions_cache:
                cached = self._sessions_cache[_SESSIONS_CACHE_KEY]
                if session_id in cached:
                    cached.remove(session_id)
        return True

    async def list_sessions(self, use_cache: bool = True) -> list[str]:
        """List all session IDs in the S3 bucket.

        Lists all unique session IDs by finding common prefixes under
        the sessions/ directory in the bucket. Results are cached using
        TTLCache for performance (default TTL: 30 seconds).

        Args:
            use_cache: If True, return cached results if available.
                       Set to False to force a refresh from S3.

        Returns:
            List of session IDs
        """
        # Check cache first (with lock to prevent race conditions)
        async with self._cache_lock:
            if use_cache and _SESSIONS_CACHE_KEY in self._sessions_cache:
                # Return a copy to prevent external modification
                return list(self._sessions_cache[_SESSIONS_CACHE_KEY])

        # Fetch from S3
        sessions: list[str] = []
        sessions_prefix = f"{self.prefix}/sessions/" if self.prefix else "sessions/"

        async with self._client() as client:
            paginator = client.get_paginator("list_objects_v2")
            # Use Delimiter to get "directories" (common prefixes)
            async for page in paginator.paginate(
                Bucket=self.bucket,
                Prefix=sessions_prefix,
                Delimiter="/",
            ):
                # Common prefixes are the session "directories"
                for prefix_info in page.get("CommonPrefixes", []):
                    prefix_path = prefix_info.get("Prefix", "")
                    # Extract session ID from prefix path
                    # e.g., "my-prefix/sessions/abc123/" -> "abc123"
                    session_id = prefix_path.rstrip("/").split("/")[-1]
                    if session_id:
                        sessions.append(session_id)

        # Update cache
        async with self._cache_lock:
            self._sessions_cache[_SESSIONS_CACHE_KEY] = sessions

        return list(sessions)

    def invalidate_sessions_cache(self) -> None:
        """Invalidate the sessions cache.

        Call this when you know the session list has changed externally
        (e.g., another process created/deleted sessions) to ensure the
        next list_sessions() call fetches fresh data from S3.

        Note: For internal create/delete operations, the cache is
        automatically updated, so invalidation is not needed.
        """
        self._sessions_cache.clear()

    async def _resolved_control_key(self, session_id: str, key: str) -> str:
        generation = self._active_generation(session_id)
        if generation is None:
            async with self._client() as client:
                envelope, _etag = await self._read_envelope(client, session_id)
            if envelope is None:
                return key
            if envelope.get("deleted") is True:
                raise FileNotFoundError(session_id)
            raw_generation = envelope.get("generation")
            raw_owner = envelope.get("owner_id")
            if raw_generation == 0 and not raw_owner:
                return key
            if not isinstance(raw_generation, int) or not isinstance(raw_owner, str):
                raise RuntimeError("Invalid current session generation")
            generation = SessionGeneration(raw_generation, raw_owner)
        return f"{self._generation_prefix(generation)}/control/{key}"

    async def _published_artifacts(
        self, client: S3Client, session_id: str
    ) -> dict[str, str] | None:
        snapshot = await self._published_snapshot(client, session_id)
        return snapshot[0] if snapshot is not None else None

    async def _published_snapshot(
        self,
        client: S3Client,
        session_id: str,
    ) -> tuple[dict[str, str], tuple[str, ...], SessionGeneration] | None:
        """Load the artifact map, complete prefixes, and publication owner."""
        envelope, _etag = await self._read_envelope(client, session_id)
        if envelope is None or envelope.get("deleted") is True:
            return None
        publication = envelope.get("artifact_publication")
        if publication is None:
            return None
        if not isinstance(publication, Mapping):
            raise RuntimeError("Invalid session artifact publication")
        snapshot = await self._snapshot_for_publication(
            client,
            session_id,
            publication,
        )
        generation = publication.get("generation")
        owner_id = publication.get("owner_id")
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 1
            or not isinstance(owner_id, str)
            or not owner_id
        ):
            raise RuntimeError("Invalid session artifact publication identity")
        return *snapshot, SessionGeneration(generation, owner_id)

    async def _artifacts_for_publication(
        self,
        client: S3Client,
        session_id: str,
        publication: Mapping[str, Any],
    ) -> dict[str, str]:
        snapshot = await self._snapshot_for_publication(
            client,
            session_id,
            publication,
        )
        return snapshot[0]

    async def _snapshot_for_publication(
        self,
        client: S3Client,
        session_id: str,
        publication: Mapping[str, Any],
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        """Load and validate one immutable publication manifest."""
        manifest_key = publication.get("manifest_key")
        if not isinstance(manifest_key, str):
            raise RuntimeError("Invalid session artifact manifest pointer")
        try:
            validated_artifact_relative_key(manifest_key)
        except ValueError as error:
            raise RuntimeError("Invalid session artifact manifest pointer") from error
        try:
            response = await client.get_object(
                Bucket=self.bucket,
                Key=self._key(session_id, manifest_key),
            )
        except ClientError as error:
            if self._is_not_found(error):
                raise RuntimeError("Published session manifest is missing") from error
            raise
        manifest = json.loads(await response["Body"].read())
        publication_generation = publication.get("generation")
        publication_owner = publication.get("owner_id")
        if (
            not isinstance(manifest, dict)
            or manifest.get("protocol") != _MANIFEST_PROTOCOL
            or manifest.get("complete") is not True
            or manifest.get("generation") != publication_generation
            or manifest.get("owner_id") != publication_owner
        ):
            raise RuntimeError("Invalid published session manifest")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise RuntimeError("Invalid published session artifact map")
        generation_prefix = publication.get("generation_prefix")
        expected_prefix = (
            f"{_GENERATION_ROOT}/{publication_generation}-{publication_owner}"
        )
        if generation_prefix != expected_prefix:
            raise RuntimeError("Invalid published generation prefix")
        validated: dict[str, str] = {}
        for logical_key, physical_key in artifacts.items():
            try:
                logical_key = validated_artifact_relative_key(logical_key)
                physical_key = validated_artifact_relative_key(physical_key)
            except ValueError as error:
                raise RuntimeError(
                    "Invalid published session artifact entry"
                ) from error
            if not physical_key.startswith(f"{_GENERATION_ROOT}/"):
                raise RuntimeError("Invalid published session artifact entry")
            validated[logical_key] = physical_key
        raw_snapshot_prefixes = manifest.get("snapshot_prefixes", [])
        if not isinstance(raw_snapshot_prefixes, list):
            raise RuntimeError("Invalid published session snapshot prefixes")
        try:
            snapshot_prefixes = _normalize_snapshot_prefixes(
                raw_snapshot_prefixes,
            )
        except ValueError as error:
            raise RuntimeError("Invalid published session snapshot prefixes") from error
        return validated, snapshot_prefixes

    @staticmethod
    def _referenced_generation_numbers(artifacts: Mapping[str, str]) -> set[int]:
        generations: set[int] = set()
        for physical_key in artifacts.values():
            parts = physical_key.split("/", 2)
            if len(parts) < 2 or parts[0] != _GENERATION_ROOT:
                raise RuntimeError("Invalid generation-qualified artifact key")
            generation_text = parts[1].split("-", 1)[0]
            try:
                generations.add(int(generation_text))
            except ValueError as error:
                raise RuntimeError(
                    "Invalid generation-qualified artifact key"
                ) from error
        return generations

    async def _resolve_read_key(
        self, client: S3Client, session_id: str, key: str
    ) -> str:
        if key.startswith("."):
            return await self._resolved_control_key(session_id, key)
        artifacts = await self._published_artifacts(client, session_id)
        if artifacts is None:
            return key
        try:
            return artifacts[key]
        except KeyError:
            raise FileNotFoundError(key) from None

    async def _reject_canonical_generation_artifact_write(
        self,
        client: S3Client,
        session_id: str,
    ) -> None:
        """Keep generation-aware session artifacts on immutable publish paths."""
        envelope, _etag = await self._read_envelope(client, session_id)
        if envelope is not None and int(envelope.get("generation") or 0) > 0:
            raise SessionGenerationOwnershipError(
                "Generation-aware S3 artifacts must be published with sync_from_local"
            )

    async def put_bytes(
        self, session_id: str, key: str, data: bytes, content_type: str | None = None
    ) -> None:
        if is_pipeline_temp_path(key):
            raise ValueError("Artifact path is reserved")
        if key == ".cancel" and await self._set_generation_cancellation(
            session_id,
            requested=True,
        ):
            return
        storage_key = (
            await self._resolved_control_key(session_id, key)
            if key.startswith(".")
            else key
        )
        async with self._client() as client:
            if not key.startswith("."):
                await self._reject_canonical_generation_artifact_write(
                    client,
                    session_id,
                )
            kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Key": self._key(session_id, storage_key),
                "Body": data,
            }
            if content_type:
                kwargs["ContentType"] = content_type
            await client.put_object(**kwargs)

    async def put_bytes_if_absent(
        self, session_id: str, key: str, data: bytes, content_type: str | None = None
    ) -> bool:
        """Atomically publish a claim using S3 conditional-write semantics."""
        if is_pipeline_temp_path(key):
            raise ValueError("Artifact path is reserved")
        storage_key = (
            await self._resolved_control_key(session_id, key)
            if key.startswith(".")
            else key
        )
        async with self._client() as client:
            if not key.startswith("."):
                await self._reject_canonical_generation_artifact_write(
                    client,
                    session_id,
                )
            kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Key": self._key(session_id, storage_key),
                "Body": data,
                "IfNoneMatch": "*",
            }
            if content_type:
                kwargs["ContentType"] = content_type
            try:
                await client.put_object(**kwargs)
            except ClientError as error:
                if self._is_conditional_conflict(error):
                    return False
                raise
        return True

    async def put_file(
        self, session_id: str, key: str, file_path: str, content_type: str | None = None
    ) -> None:
        if is_pipeline_temp_path(key):
            raise ValueError("Artifact path is reserved")
        with open_regular_file_no_follow(file_path) as (source, _metadata):
            async with self._client() as client:
                await self._reject_canonical_generation_artifact_write(
                    client,
                    session_id,
                )
                extra: dict[str, Any] = {}
                if content_type:
                    extra["ContentType"] = content_type
                await client.upload_fileobj(
                    source,
                    self.bucket,
                    self._key(session_id, key),
                    ExtraArgs=extra,
                )

    async def delete_key(self, session_id: str, key: str) -> None:
        """Delete one session artifact."""
        if is_pipeline_temp_path(key):
            raise ValueError("Artifact path is reserved")
        if key == ".cancel" and await self._set_generation_cancellation(
            session_id,
            requested=False,
        ):
            return
        storage_key = (
            await self._resolved_control_key(session_id, key)
            if key.startswith(".")
            else key
        )
        async with self._client() as client:
            if not key.startswith("."):
                await self._reject_canonical_generation_artifact_write(
                    client,
                    session_id,
                )
            await client.delete_object(
                Bucket=self.bucket, Key=self._key(session_id, storage_key)
            )

    async def open_read(self, session_id: str, key: str) -> io.BytesIO:
        if is_pipeline_temp_path(key):
            raise FileNotFoundError(key)
        if key == METADATA_KEY:
            metadata = await self.get_json(session_id, key)
            if metadata is None:
                raise FileNotFoundError(key)
            return io.BytesIO(json.dumps(metadata).encode("utf-8"))
        async with self._client() as client:
            if key == ".cancel":
                envelope, _etag = await self._read_envelope(client, session_id)
                if envelope is not None and int(envelope.get("generation") or 0) > 0:
                    generation = self._active_generation(session_id)
                    if generation is not None and (
                        envelope.get("generation") != generation.generation
                        or envelope.get("owner_id") != generation.owner_id
                    ):
                        raise FileNotFoundError(key)
                    if envelope.get("cancellation_requested") is True:
                        return io.BytesIO(b"")
                    raise FileNotFoundError(key)
            resolved_key = await self._resolve_read_key(client, session_id, key)
            try:
                response = await client.get_object(
                    Bucket=self.bucket,
                    Key=self._key(session_id, resolved_key),
                )
            except ClientError as error:
                if self._is_not_found(error):
                    raise FileNotFoundError(key) from None
                raise
            return io.BytesIO(await response["Body"].read())

    async def exists(self, session_id: str, key: str) -> bool:
        if is_pipeline_temp_path(key):
            return False
        if key == METADATA_KEY:
            return await self.get_json(session_id, key) is not None
        async with self._client() as client:
            if key == ".cancel":
                envelope, _etag = await self._read_envelope(client, session_id)
                if envelope is not None and int(envelope.get("generation") or 0) > 0:
                    generation = self._active_generation(session_id)
                    return bool(
                        envelope.get("cancellation_requested") is True
                        and (
                            generation is None
                            or (
                                envelope.get("generation") == generation.generation
                                and envelope.get("owner_id") == generation.owner_id
                            )
                        )
                    )
            try:
                resolved_key = await self._resolve_read_key(client, session_id, key)
                await client.head_object(
                    Bucket=self.bucket,
                    Key=self._key(session_id, resolved_key),
                )
                return True
            except FileNotFoundError:
                return False
            except ClientError as error:
                if self._is_not_found(error):
                    return False
                raise

    async def list_keys(self, session_id: str, prefix: str = "") -> list[str]:
        if is_pipeline_temp_path(prefix):
            return []
        async with self._client() as client:
            artifacts = await self._published_artifacts(client, session_id)
            if artifacts is not None:
                return sorted(key for key in artifacts if key.startswith(prefix))

            out: list[str] = []
            pfx = self._key(session_id, prefix)
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self.bucket, Prefix=pfx):
                for obj in page.get("Contents", []):
                    full = obj["Key"]
                    base_len = len(self._key(session_id, ""))
                    rel = full[base_len:].lstrip("/")
                    if is_pipeline_temp_path(rel) or rel.startswith(
                        f"{_GENERATION_ROOT}/"
                    ):
                        continue
                    out.append(rel)
        return out

    async def put_json(self, session_id: str, key: str, obj: dict) -> None:
        if key == METADATA_KEY:
            async with self._client() as client:
                for attempt in range(_MAX_CAS_ATTEMPTS):
                    envelope, etag = await self._read_envelope(client, session_id)
                    if envelope is None:
                        generation = SessionGeneration(
                            generation=1,
                            owner_id=uuid.uuid4().hex,
                        )
                        replacement = self._new_envelope(
                            generation,
                            metadata=obj,
                        )
                        replacement["generation_state"] = self._generation_state(obj)
                        try:
                            await self._write_envelope(
                                client,
                                session_id,
                                replacement,
                                etag=None,
                            )
                        except ClientError as error:
                            if self._is_conditional_conflict(error):
                                await self._cas_retry_delay(attempt)
                                continue
                            raise
                        self._activate_generation(session_id, generation)
                        await self._best_effort_write_generation_descriptor(
                            client,
                            session_id,
                            generation,
                            base_publication=None,
                        )
                        return
                    generation = self._active_generation(session_id)
                    if generation is not None and (
                        envelope.get("generation") != generation.generation
                        or envelope.get("owner_id") != generation.owner_id
                    ):
                        raise SessionGenerationOwnershipError(
                            f"Stale generation cannot update {session_id}"
                        )
                    replacement = dict(envelope)
                    replacement.pop("legacy", None)
                    replacement["metadata"] = deepcopy(obj)
                    replacement["generation_state"] = self._generation_state(obj)
                    replacement["generation_lease_expires_at"] = self._lease_expiry()
                    try:
                        await self._write_envelope(
                            client,
                            session_id,
                            replacement,
                            etag=etag,
                        )
                    except ClientError as error:
                        if self._is_conditional_conflict(error):
                            await self._cas_retry_delay(attempt)
                            continue
                        raise
                    return
            raise RuntimeError(f"Session metadata remained contended: {session_id}")
        await self.put_bytes(
            session_id, key, json.dumps(obj).encode("utf-8"), "application/json"
        )

    async def update_json(
        self,
        session_id: str,
        key: str,
        updater: Callable[[dict], dict | None],
    ) -> dict | None:
        if key != METADATA_KEY:
            current = await self.get_json(session_id, key)
            if current is None:
                return None
            updated = updater(dict(current))
            if updated is not None:
                await self.put_json(session_id, key, updated)
                return updated
            return current

        async with self._client() as client:
            for attempt in range(_MAX_CAS_ATTEMPTS):
                envelope, etag = await self._read_envelope(client, session_id)
                if envelope is None or etag is None or envelope.get("deleted") is True:
                    return None
                generation = self._active_generation(session_id)
                if generation is not None and (
                    envelope.get("generation") != generation.generation
                    or envelope.get("owner_id") != generation.owner_id
                ):
                    raise SessionGenerationOwnershipError(
                        f"Stale generation cannot update {session_id}"
                    )
                updated = updater(deepcopy(envelope["metadata"]))
                if updated is None:
                    return deepcopy(envelope["metadata"])
                replacement = dict(envelope)
                replacement.pop("legacy", None)
                replacement["metadata"] = deepcopy(updated)
                replacement["generation_state"] = self._generation_state(updated)
                replacement["generation_lease_expires_at"] = self._lease_expiry()
                try:
                    await self._write_envelope(
                        client,
                        session_id,
                        replacement,
                        etag=etag,
                    )
                except ClientError as error:
                    if self._is_conditional_conflict(error):
                        await self._cas_retry_delay(attempt)
                        continue
                    raise
                return updated
        raise RuntimeError(f"Session metadata remained contended: {session_id}")

    async def update_json_if_not_cancelled(
        self,
        session_id: str,
        key: str,
        updater: Callable[[dict], dict | None],
    ) -> dict | None:
        """CAS-update metadata only while this generation is not cancelled."""
        if key != METADATA_KEY:
            raise ValueError(
                "Cancellation arbitration is only valid for session metadata"
            )

        async with self._client() as client:
            for attempt in range(_MAX_CAS_ATTEMPTS):
                envelope, etag = await self._read_envelope(client, session_id)
                if envelope is None or etag is None or envelope.get("deleted") is True:
                    return None
                generation = self._active_generation(session_id)
                if generation is not None and (
                    envelope.get("generation") != generation.generation
                    or envelope.get("owner_id") != generation.owner_id
                ):
                    raise SessionGenerationOwnershipError(
                        f"Stale generation cannot update {session_id}"
                    )
                if envelope.get("cancellation_requested") is True:
                    return None

                updated = updater(deepcopy(envelope["metadata"]))
                if updated is None:
                    return deepcopy(envelope["metadata"])
                replacement = dict(envelope)
                replacement.pop("legacy", None)
                replacement["metadata"] = deepcopy(updated)
                replacement["generation_state"] = self._generation_state(updated)
                replacement["generation_lease_expires_at"] = self._lease_expiry()
                try:
                    await self._write_envelope(
                        client,
                        session_id,
                        replacement,
                        etag=etag,
                    )
                except ClientError as error:
                    if self._is_conditional_conflict(error):
                        await self._cas_retry_delay(attempt)
                        continue
                    raise
                return updated
        raise RuntimeError(f"Session metadata remained contended: {session_id}")

    async def get_json(self, session_id: str, key: str) -> dict | None:
        if is_pipeline_temp_path(key):
            return None
        if key == METADATA_KEY:
            async with self._client() as client:
                envelope, _etag = await self._read_envelope(client, session_id)
            if envelope is None or envelope.get("deleted") is True:
                return None
            return deepcopy(envelope["metadata"])
        try:
            stream = await self.open_read(session_id, key)
        except FileNotFoundError:
            return None
        return json.loads(stream.read())

    async def append_event(self, session_id: str, event: dict) -> None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        control_root = await self._resolved_control_key(session_id, "events")
        event_key = f"{control_root}/{ts}-{uuid.uuid4().hex[:8]}.json"
        async with self._client() as client:
            await client.put_object(
                Bucket=self.bucket,
                Key=self._key(session_id, event_key),
                Body=json.dumps(event).encode("utf-8"),
                ContentType="application/json",
                IfNoneMatch="*",
            )

    async def get_event_log(self, session_id: str) -> list[dict]:
        control_root = await self._resolved_control_key(session_id, "events")
        events: list[dict] = []
        async with self._client() as client:
            paginator = client.get_paginator("list_objects_v2")
            event_keys: list[str] = []
            async for page in paginator.paginate(
                Bucket=self.bucket,
                Prefix=self._key(session_id, f"{control_root}/"),
            ):
                event_keys.extend(obj["Key"] for obj in page.get("Contents", []))
            for physical_key in sorted(event_keys):
                response = await client.get_object(Bucket=self.bucket, Key=physical_key)
                events.append(json.loads(await response["Body"].read()))
        return events

    async def make_public_url(
        self, session_id: str, key: str, expires_seconds: int = 3600
    ) -> str | None:
        if is_pipeline_temp_path(key) or not self.presign_by_default:
            return None
        async with self._client() as client:
            try:
                resolved_key = await self._resolve_read_key(client, session_id, key)
            except FileNotFoundError:
                return None
            return await client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket,
                    "Key": self._key(session_id, resolved_key),
                },
                ExpiresIn=expires_seconds,
            )

    async def sync_from_local(
        self,
        session_id: str,
        local_session_dir: str,
        prefix: str | Sequence[str] = "",
    ) -> int:
        """Publish one immutable, generation-qualified artifact snapshot.

        Every selected file is durable before the manifest pointer moves. The
        pointer and owner identity live in the same conditional S3 document, so
        a superseded worker cannot expose its late completion.

        Args:
            session_id: Session identifier
            local_session_dir: Path to local session directory
            prefix: Optional prefix or prefixes published as one generation.

        Returns:
            Number of files synced
        """
        generation = self._active_generation(session_id)
        if generation is None:
            raise SessionGenerationOwnershipError(
                f"No generation owner is bound for publication of {session_id}"
            )
        prefixes = (prefix,) if isinstance(prefix, str) else tuple(prefix)
        if not prefixes:
            return 0

        def selected(relative_key: str) -> bool:
            return any(relative_key.startswith(item) for item in prefixes)

        generation_prefix = self._generation_prefix(generation)
        uploaded_artifacts: dict[str, str] = {}
        upload_id = uuid.uuid4().hex
        upload_prefix = f"{generation_prefix}/uploads/{upload_id}"
        with open_confined_directory(local_session_dir) as source_descriptor:
            async with self._client() as client:
                for selected_prefix in dict.fromkeys(prefixes):
                    for artifact in iter_open_regular_files(
                        source_descriptor,
                        prefix=selected_prefix,
                    ):
                        if artifact.relative_key in uploaded_artifacts:
                            continue
                        physical_key = (
                            f"{upload_prefix}/artifacts/{artifact.relative_key}"
                        )
                        content_type, _ = mimetypes.guess_type(artifact.relative_key)
                        extra: dict[str, Any] = {}
                        if content_type:
                            extra["ContentType"] = content_type
                        await client.upload_fileobj(
                            artifact.stream,
                            self.bucket,
                            self._key(session_id, physical_key),
                            ExtraArgs=extra,
                        )
                        uploaded_artifacts[artifact.relative_key] = physical_key

                for attempt in range(_MAX_CAS_ATTEMPTS):
                    envelope, etag = await self._read_envelope(client, session_id)
                    if envelope is None or etag is None:
                        raise SessionGenerationOwnershipError(
                            f"Session disappeared during publication: {session_id}"
                        )
                    if (
                        envelope.get("deleted") is True
                        or envelope.get("generation") != generation.generation
                        or envelope.get("owner_id") != generation.owner_id
                    ):
                        raise SessionGenerationOwnershipError(
                            f"Stale generation cannot publish {session_id}"
                        )
                    previous_publication = envelope.get("artifact_publication")
                    previous_snapshot = (
                        await self._snapshot_for_publication(
                            client,
                            session_id,
                            previous_publication,
                        )
                        if isinstance(previous_publication, Mapping)
                        else ({}, ())
                    )
                    previous_artifacts, previous_snapshot_prefixes = previous_snapshot
                    # A selected prefix is a complete replacement snapshot:
                    # omitted keys beneath it must disappear. Unselected
                    # artifact families remain part of the session snapshot.
                    # When they came from an older generation, copy their
                    # immutable bytes into this generation rather than pinning
                    # old generations beyond the configured retention bound.
                    carried_artifacts: dict[str, str] = {}
                    for (
                        logical_key,
                        previous_physical_key,
                    ) in previous_artifacts.items():
                        if selected(logical_key):
                            continue
                        if (
                            isinstance(previous_publication, Mapping)
                            and previous_publication.get("generation")
                            == generation.generation
                        ):
                            carried_artifacts[logical_key] = previous_physical_key
                            continue
                        physical_key = f"{upload_prefix}/artifacts/{logical_key}"
                        await client.copy_object(
                            Bucket=self.bucket,
                            Key=self._key(session_id, physical_key),
                            CopySource={
                                "Bucket": self.bucket,
                                "Key": self._key(
                                    session_id,
                                    previous_physical_key,
                                ),
                            },
                        )
                        carried_artifacts[logical_key] = physical_key
                    artifacts = {**carried_artifacts, **uploaded_artifacts}
                    snapshot_prefixes = _updated_snapshot_prefixes(
                        previous_snapshot_prefixes,
                        prefixes,
                    )
                    publication_id = uuid.uuid4().hex
                    publication_prefix = (
                        f"{generation_prefix}/publications/{publication_id}"
                    )
                    manifest_key = f"{publication_prefix}/manifest.json"
                    manifest = {
                        "protocol": _MANIFEST_PROTOCOL,
                        "generation": generation.generation,
                        "owner_id": generation.owner_id,
                        "publication_id": publication_id,
                        "created_at": datetime.now(UTC).isoformat(),
                        "complete": True,
                        "artifacts": artifacts,
                        "snapshot_prefixes": list(snapshot_prefixes),
                    }
                    await client.put_object(
                        Bucket=self.bucket,
                        Key=self._key(session_id, manifest_key),
                        Body=json.dumps(manifest, separators=(",", ":")).encode(
                            "utf-8"
                        ),
                        ContentType="application/json",
                        IfNoneMatch="*",
                    )
                    publication = {
                        "generation": generation.generation,
                        "owner_id": generation.owner_id,
                        "generation_prefix": generation_prefix,
                        "manifest_key": manifest_key,
                        "published_at": datetime.now(UTC).isoformat(),
                    }
                    replacement = dict(envelope)
                    replacement["artifact_publication"] = publication
                    try:
                        await self._write_envelope(
                            client,
                            session_id,
                            replacement,
                            etag=etag,
                        )
                    except ClientError as error:
                        if self._is_conditional_conflict(error):
                            await self._cas_retry_delay(attempt)
                            continue
                        raise
                    protected_generations = self._referenced_generation_numbers(
                        artifacts
                    )
                    protected_generations.add(generation.generation)
                    await self._best_effort_generation_gc(
                        client,
                        session_id,
                        current_generation=generation.generation,
                        protected_generations=protected_generations,
                    )
                    try:
                        local_marker = _read_local_snapshot_marker(
                            source_descriptor,
                        )
                        if (
                            local_marker is not None
                            and local_marker.local_generation == generation
                        ):
                            reconciled_prefixes = local_marker.prefixes
                        elif generation.generation == 1:
                            # A new session directory is wholly owned by its
                            # first generation, including local-only pipeline
                            # intermediates beside a narrowly published file.
                            reconciled_prefixes = ("",)
                        else:
                            # A reused rerun directory may still contain bytes
                            # from its predecessor. Only the region just
                            # published is proven to belong to this generation.
                            reconciled_prefixes = ()
                        _write_local_snapshot_generation(
                            source_descriptor,
                            generation,
                            (*reconciled_prefixes, *prefixes),
                            publication_generation=generation,
                        )
                    except (OSError, RuntimeError, ValueError):
                        # A missing marker makes later hydration clear the
                        # requested local region, which is conservative and
                        # does not invalidate the already committed snapshot.
                        logger.warning(
                            "Could not record local S3 snapshot generation for %s",
                            session_id,
                            exc_info=True,
                        )
                    return len(uploaded_artifacts)
                raise RuntimeError(
                    f"Artifact publication remained contended: {session_id}"
                )

    async def _best_effort_generation_gc(
        self,
        client: S3Client,
        session_id: str,
        *,
        current_generation: int,
        protected_generations: set[int],
    ) -> None:
        """Bound diagnostic generations without invalidating a committed run."""
        try:
            await self._garbage_collect_generations(
                client,
                session_id,
                current_generation=current_generation,
                protected_generations=protected_generations,
            )
        except Exception:  # noqa: BLE001 - GC is non-critical after a CAS commit
            log_durable_failure(
                logger,
                "physics_s3_generation_gc_failed",
                phase=FailurePhase.ROLLBACK,
                retryable=True,
            )

    async def _garbage_collect_generations(
        self,
        client: S3Client,
        session_id: str,
        *,
        current_generation: int,
        protected_generations: set[int],
    ) -> None:
        """Retain a bounded set of completed, failed, and incomplete generations."""
        envelope, _etag = await self._read_envelope(client, session_id)
        if (
            envelope is None
            or envelope.get("deleted") is True
            or envelope.get("generation") != current_generation
        ):
            # A delayed GC caller must never prune the base publication
            # inherited by a newer generation.
            return

        effective_protected = set(protected_generations)
        publication = envelope.get("artifact_publication")
        if isinstance(publication, Mapping):
            artifacts, _prefixes = await self._snapshot_for_publication(
                client,
                session_id,
                publication,
            )
            effective_protected.update(self._referenced_generation_numbers(artifacts))
            published_generation = publication.get("generation")
            if isinstance(published_generation, int) and not isinstance(
                published_generation, bool
            ):
                effective_protected.add(published_generation)

        prefix = self._key(session_id, f"{_GENERATION_ROOT}/")
        paginator = client.get_paginator("list_objects_v2")
        objects_by_generation: dict[int, list[str]] = {}
        async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                physical_key = obj.get("Key")
                if not isinstance(physical_key, str):
                    continue
                relative = physical_key[len(prefix) :]
                generation_text = relative.split("-", 1)[0]
                try:
                    generation_number = int(generation_text)
                except ValueError:
                    continue
                objects_by_generation.setdefault(generation_number, []).append(
                    physical_key
                )
        retained = sorted(objects_by_generation, reverse=True)[
            : self._generation_retention
        ]
        retained_set = set(retained) | {current_generation} | effective_protected
        for generation_number, physical_keys in objects_by_generation.items():
            if (
                generation_number in retained_set
                or generation_number >= current_generation
            ):
                continue
            for physical_key in physical_keys:
                await client.delete_object(Bucket=self.bucket, Key=physical_key)

    async def sync_to_local(
        self,
        session_id: str,
        local_session_dir: str,
        prefix: str | Sequence[str] = "",
    ) -> int:
        """Sync files from S3 to local session directory.

        Published files replace replica-local cache entries. A local path may
        belong to an older immutable generation, so existence alone is not a
        valid cache hit after the manifest pointer moves.

        Args:
            session_id: Session identifier
            local_session_dir: Path to local session directory
            prefix: Optional prefix or prefixes to filter keys.

        Returns:
            Number of files downloaded
        """
        prefixes = tuple(
            dict.fromkeys((prefix,) if isinstance(prefix, str) else prefix)
        )
        if not prefixes:
            return 0

        async with self._client() as client:
            snapshot = await self._published_snapshot(client, session_id)
            count, _artifact_keys = await self._sync_snapshot_to_local(
                client,
                session_id,
                local_session_dir,
                prefixes,
                snapshot,
            )
        return count

    async def sync_completed_publication_to_local(
        self,
        session_id: str,
        local_session_dir: str,
        prefix: str | Sequence[str] = "",
    ) -> CompletedSessionSnapshot:
        """Hydrate one publication whose metadata is completed in the same view."""
        prefixes = tuple(
            dict.fromkeys((prefix,) if isinstance(prefix, str) else prefix)
        )
        async with self._client() as client:
            envelope, _etag = await self._read_envelope(client, session_id)
            if envelope is None or envelope.get("deleted") is True:
                raise FileNotFoundError(session_id)
            metadata = envelope.get("metadata")
            if not isinstance(metadata, dict):
                raise RuntimeError("Invalid generation-aware session metadata")
            if (
                metadata.get("status") != "completed"
                or envelope.get("generation_state") != "terminal"
            ):
                raise SessionNotCompletedError(session_id)

            publication = envelope.get("artifact_publication")
            snapshot: tuple[dict[str, str], tuple[str, ...], SessionGeneration] | None
            if publication is None:
                if int(envelope.get("generation") or 0) != 0:
                    raise RuntimeError(
                        "Completed generation has no artifact publication"
                    )
                # Legacy canonical artifacts never change once a generation-
                # aware rerun begins, so listing them after this envelope read
                # still represents the completed legacy publication.
                snapshot = None
            elif isinstance(publication, Mapping):
                artifacts, complete_prefixes = await self._snapshot_for_publication(
                    client,
                    session_id,
                    publication,
                )
                generation = publication.get("generation")
                owner_id = publication.get("owner_id")
                if (
                    not isinstance(generation, int)
                    or isinstance(generation, bool)
                    or generation < 1
                    or not isinstance(owner_id, str)
                    or not owner_id
                ):
                    raise RuntimeError("Invalid session artifact publication identity")
                snapshot = (
                    artifacts,
                    complete_prefixes,
                    SessionGeneration(generation, owner_id),
                )
            else:
                raise RuntimeError("Invalid session artifact publication")

            count, artifact_keys = await self._sync_snapshot_to_local(
                client,
                session_id,
                local_session_dir,
                prefixes,
                snapshot,
            )
        return CompletedSessionSnapshot(
            metadata=deepcopy(metadata),
            artifact_keys=artifact_keys,
            downloaded_count=count,
        )

    async def _sync_snapshot_to_local(
        self,
        client: S3Client,
        session_id: str,
        local_session_dir: str,
        prefixes: tuple[str, ...],
        snapshot: tuple[
            dict[str, str],
            tuple[str, ...],
            SessionGeneration,
        ]
        | None,
    ) -> tuple[int, tuple[str, ...]]:
        """Hydrate an already-selected immutable snapshot without re-reading it."""
        if not prefixes:
            return 0, ()

        def selected(relative_key: str) -> bool:
            return any(relative_key.startswith(item) for item in prefixes)

        publication_generation: SessionGeneration | None = None
        if snapshot is None:
            # Legacy sessions remain readable until their next rerun.
            complete_prefixes: tuple[str, ...] = ()
            remote_files: list[tuple[str, str]] = []
            paginator = client.get_paginator("list_objects_v2")
            session_prefix = self._key(session_id, "")
            seen_physical_keys: set[str] = set()
            for selected_prefix in prefixes:
                async for page in paginator.paginate(
                    Bucket=self.bucket,
                    Prefix=self._key(session_id, selected_prefix),
                ):
                    for obj in page.get("Contents", []):
                        physical_key = obj.get("Key")
                        relative_key = validated_s3_object_suffix(
                            physical_key,
                            session_prefix,
                        )
                        if (
                            not relative_key
                            or not selected(relative_key)
                            or is_pipeline_temp_path(relative_key)
                            or relative_key.startswith(f"{_GENERATION_ROOT}/")
                            or physical_key in seen_physical_keys
                        ):
                            continue
                        seen_physical_keys.add(physical_key)
                        remote_files.append((physical_key, relative_key))
        else:
            artifacts, complete_prefixes, publication_generation = snapshot
            remote_files = [
                (self._key(session_id, physical_key), logical_key)
                for logical_key, physical_key in artifacts.items()
                if selected(logical_key)
            ]

        count = 0
        remote_relative_keys = {
            relative_key for _physical_key, relative_key in remote_files
        }
        with open_confined_directory(
            local_session_dir,
            create=True,
        ) as destination_descriptor:
            local_marker = _read_local_snapshot_marker(destination_descriptor)
            reconciled_prefixes = (
                local_marker.prefixes
                if publication_generation is not None
                and local_marker is not None
                and local_marker.publication_generation == publication_generation
                else ()
            )
            local_generation = (
                local_marker.local_generation
                if publication_generation is not None
                and local_marker is not None
                and local_marker.publication_generation == publication_generation
                else publication_generation
            )
            if publication_generation is not None:
                # Unmarked bytes, or bytes owned by a superseded run, must
                # not be combined with a newer partial manifest. A marker
                # is prefix-scoped because service hydration is normally
                # incremental; reconciling input/ must not bless an old
                # cache/ owned by another generation.
                for selected_prefix in prefixes:
                    if any(
                        selected_prefix.startswith(reconciled)
                        for reconciled in reconciled_prefixes
                    ):
                        continue
                    prune_confined_snapshot(
                        destination_descriptor,
                        selected_prefix,
                        set(),
                    )
            for s3_key, relative_key in remote_files:
                with confined_atomic_writer(
                    destination_descriptor,
                    relative_key,
                    overwrite=True,
                ) as destination:
                    if destination.stream is None:
                        continue
                    await client.download_fileobj(
                        self.bucket,
                        s3_key,
                        destination.stream,
                    )
                if destination.published:
                    count += 1
            for selected_prefix in _snapshot_prefix_intersections(
                prefixes,
                complete_prefixes,
            ):
                prune_confined_snapshot(
                    destination_descriptor,
                    selected_prefix,
                    remote_relative_keys,
                )
            if publication_generation is not None:
                try:
                    _write_local_snapshot_generation(
                        destination_descriptor,
                        local_generation,
                        (*reconciled_prefixes, *prefixes),
                        publication_generation=publication_generation,
                    )
                except (OSError, RuntimeError, ValueError):
                    # An absent marker makes the next hydration clear its
                    # requested region again, which remains fail-closed.
                    logger.warning(
                        "Could not record hydrated S3 snapshot generation for %s",
                        session_id,
                        exc_info=True,
                    )
        return count, tuple(sorted(remote_relative_keys))

    async def cleanup_stale_local_sessions(
        self,
        local_storage_path: str,
        max_age_hours: float = 24.0,
        skip_session_ids: set[str] | None = None,
    ) -> int:
        """Clean up stale local caches without republishing unowned bytes.

        Workers publish before terminal metadata. A background replica cannot
        prove that an old local cache belongs to the current generation, so it
        must never sync that cache during cleanup; doing so could replace a
        newer run. Terminal caches are simply removed after the age check.

        Args:
            local_storage_path: Root path where local sessions are stored
            max_age_hours: Maximum age in hours before cleanup (default: 24)
            skip_session_ids: Session IDs that must not be removed locally

        Returns:
            Number of sessions cleaned up
        """
        local_root = Path(local_storage_path)
        if not local_root.exists():
            return 0

        cutoff_time = datetime.now(UTC) - timedelta(hours=max_age_hours)
        cleaned_count = 0
        skip_session_ids = skip_session_ids or set()

        for session_dir in local_root.iterdir():
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name
            if session_id in skip_session_ids:
                continue

            # Determine last update time using the store API
            last_updated = await self._get_session_last_updated(session_id)

            if last_updated is None:
                # No metadata, use directory mtime (as UTC)
                last_updated = datetime.fromtimestamp(
                    session_dir.stat().st_mtime, tz=UTC
                )

            # Check if session is stale
            if last_updated >= cutoff_time:
                continue  # Session is still fresh

            # Session is stale. If its owner never committed this generation's
            # publication, preserve the only local input/diagnostic copy rather
            # than silently deleting it from the final surviving replica.
            async with self._client() as client:
                envelope, _etag = await self._read_envelope(client, session_id)
            if envelope is not None and int(envelope.get("generation") or 0) > 0:
                publication = envelope.get("artifact_publication")
                publication_generation = (
                    publication.get("generation")
                    if isinstance(publication, Mapping)
                    else None
                )
                metadata = envelope.get("metadata")
                terminal_status = (
                    metadata.get("status") if isinstance(metadata, Mapping) else None
                )
                if terminal_status in {
                    "completed",
                    "failed",
                    "cancelled",
                } and publication_generation != envelope.get("generation"):
                    logger.warning(
                        "Preserving unpublished local cache for session %s",
                        session_id[:8],
                    )
                    continue
            try:
                # Remove local directory
                remove_confined_tree(session_dir, local_root)
                logger.info(f"Removed local cache for session {session_id[:8]}")

                cleaned_count += 1

            except Exception:
                log_durable_failure(
                    logger,
                    "stale_session_cleanup_failed",
                    phase=FailurePhase.ROLLBACK,
                    retryable=True,
                )

        if cleaned_count > 0:
            logger.info(
                f"Cleaned up {cleaned_count} stale local sessions "
                f"(older than {max_age_hours}h)"
            )

        return cleaned_count

    async def _get_session_last_updated(self, session_id: str) -> datetime | None:
        """Get the last updated time from session metadata using the store API.

        Retrieves metadata through get_json to ensure consistent access patterns.

        Args:
            session_id: Session identifier

        Returns:
            Last updated datetime or None if not available
        """
        try:
            metadata = await self.get_json(session_id, METADATA_KEY)
            if metadata:
                updated_at = metadata.get("updated_at")
                if updated_at:
                    return datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except Exception:
            pass

        return None
