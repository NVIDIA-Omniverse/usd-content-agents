# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import io
import json
import logging
import mimetypes
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
    remove_confined_tree,
    validated_s3_object_suffix,
)
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)

from .base import METADATA_KEY, SessionStore
from .config import StorageConfig

logger = logging.getLogger(__name__)

# Cache key for sessions list (single key since we only cache one list per store)
_SESSIONS_CACHE_KEY = "sessions"

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
    ) -> None:
        if not bucket:
            raise ValueError("bucket is required for S3SessionStore")
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
        self._session = aioboto3.Session()
        self._bucket_ensured = False

        # TTL cache for sessions list (maxsize=1 since we only cache one list)
        # TTLCache automatically expires entries after ttl seconds
        ttl = sessions_cache_ttl
        self._sessions_cache: TTLCache[str, list[str]] = TTLCache(maxsize=1, ttl=ttl)
        self._cache_lock = asyncio.Lock()

    @property
    def kind(self) -> str:
        return "s3"

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

    def _key(self, session_id: str, key: str) -> str:
        base = (
            f"{self.prefix}/sessions/{session_id}"
            if self.prefix
            else f"sessions/{session_id}"
        )
        return f"{base}/{key}".lstrip("/")

    async def init_session(self, session_id: str) -> None:
        # Update cache: add session to cached list if it exists
        async with self._cache_lock:
            if _SESSIONS_CACHE_KEY in self._sessions_cache:
                cached = self._sessions_cache[_SESSIONS_CACHE_KEY]
                if session_id not in cached:
                    cached.append(session_id)

    async def delete_session(self, session_id: str) -> None:
        async with self._client() as client:
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

    async def put_bytes(
        self, session_id: str, key: str, data: bytes, content_type: str | None = None
    ) -> None:
        if is_pipeline_temp_path(key):
            raise ValueError("Artifact path is reserved")
        async with self._client() as client:
            kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Key": self._key(session_id, key),
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
        async with self._client() as client:
            kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Key": self._key(session_id, key),
                "Body": data,
                "IfNoneMatch": "*",
            }
            if content_type:
                kwargs["ContentType"] = content_type
            try:
                await client.put_object(**kwargs)
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in {
                    "412",
                    "PreconditionFailed",
                    "ConditionalRequestConflict",
                }:
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
        async with self._client() as client:
            await client.delete_object(
                Bucket=self.bucket, Key=self._key(session_id, key)
            )

    async def open_read(self, session_id: str, key: str) -> io.BytesIO:
        if is_pipeline_temp_path(key):
            raise FileNotFoundError(key)
        async with self._client() as client:
            response = await client.get_object(
                Bucket=self.bucket, Key=self._key(session_id, key)
            )
            body = await response["Body"].read()
            return io.BytesIO(body)

    async def exists(self, session_id: str, key: str) -> bool:
        if is_pipeline_temp_path(key):
            return False
        async with self._client() as client:
            try:
                await client.head_object(
                    Bucket=self.bucket, Key=self._key(session_id, key)
                )
                return True
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("404", "NoSuchKey"):
                    return False
                raise

    async def list_keys(self, session_id: str, prefix: str = "") -> list[str]:
        if is_pipeline_temp_path(prefix):
            return []
        out: list[str] = []
        pfx = self._key(session_id, prefix)
        async with self._client() as client:
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self.bucket, Prefix=pfx):
                for obj in page.get("Contents", []):
                    full = obj["Key"]
                    base_len = len(self._key(session_id, ""))
                    rel = full[base_len:].lstrip("/")
                    if is_pipeline_temp_path(rel):
                        continue
                    out.append(rel)
        return out

    async def put_json(self, session_id: str, key: str, obj: dict) -> None:
        await self.put_bytes(
            session_id, key, json.dumps(obj).encode("utf-8"), "application/json"
        )

    async def get_json(self, session_id: str, key: str) -> dict | None:
        if is_pipeline_temp_path(key):
            return None
        async with self._client() as client:
            try:
                response = await client.get_object(
                    Bucket=self.bucket, Key=self._key(session_id, key)
                )
                body = await response["Body"].read()
                return json.loads(body)
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("404", "NoSuchKey"):
                    return None
                raise

    async def append_event(self, session_id: str, event: dict) -> None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        event_key = f"events/{ts}-{uuid.uuid4().hex[:8]}.json"
        await self.put_bytes(
            session_id, event_key, json.dumps(event).encode("utf-8"), "application/json"
        )

    async def get_event_log(self, session_id: str) -> list[dict]:
        keys = await self.list_keys(session_id, prefix="events/")
        if not keys:
            return []
        events: list[dict] = []
        async with self._client() as client:
            for key in sorted(keys):
                response = await client.get_object(
                    Bucket=self.bucket, Key=self._key(session_id, key)
                )
                body = await response["Body"].read()
                events.append(json.loads(body))
        return events

    async def make_public_url(
        self, session_id: str, key: str, expires_seconds: int = 3600
    ) -> str | None:
        if is_pipeline_temp_path(key) or not self.presign_by_default:
            return None
        async with self._client() as client:
            return await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": self._key(session_id, key)},
                ExpiresIn=expires_seconds,
            )

    async def sync_from_local(
        self, session_id: str, local_session_dir: str, prefix: str = ""
    ) -> int:
        """Sync files from local session directory to S3.

        Only uploads files that don't already exist on S3,
        preserving the directory structure. Uses a single LIST call
        to determine existing keys instead of per-file HEAD checks.

        Args:
            session_id: Session identifier
            local_session_dir: Path to local session directory
            prefix: Optional prefix to filter files (e.g., "output/")

        Returns:
            Number of files synced
        """
        try:
            with open_confined_directory(local_session_dir) as source_descriptor:
                count = 0
                async with self._client() as client:
                    existing_keys: set[str] = set()
                    s3_prefix = self._key(session_id, prefix)
                    paginator = client.get_paginator("list_objects_v2")
                    async for page in paginator.paginate(
                        Bucket=self.bucket,
                        Prefix=s3_prefix,
                    ):
                        for obj in page.get("Contents", []):
                            existing_keys.add(obj["Key"])

                    for artifact in iter_open_regular_files(
                        source_descriptor,
                        prefix=prefix,
                    ):
                        s3_key = self._key(session_id, artifact.relative_key)
                        if s3_key in existing_keys:
                            continue
                        content_type, _ = mimetypes.guess_type(artifact.relative_key)
                        extra: dict[str, Any] = {}
                        if content_type:
                            extra["ContentType"] = content_type
                        await client.upload_fileobj(
                            artifact.stream,
                            self.bucket,
                            s3_key,
                            ExtraArgs=extra,
                        )
                        count += 1
                return count
        except FileNotFoundError:
            return 0

    async def sync_to_local(
        self, session_id: str, local_session_dir: str, prefix: str = ""
    ) -> int:
        """Sync files from S3 to local session directory.

        Only downloads files that don't already exist locally.

        Args:
            session_id: Session identifier
            local_session_dir: Path to local session directory
            prefix: Optional prefix to filter keys (e.g., "input/")

        Returns:
            Number of files downloaded
        """
        remote_files: list[tuple[str, str]] = []
        async with self._client() as client:
            paginator = client.get_paginator("list_objects_v2")
            s3_prefix = self._key(session_id, prefix)
            session_prefix = self._key(session_id, "")
            async for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
                for obj in page.get("Contents", []):
                    s3_key = obj.get("Key")
                    relative_key = validated_s3_object_suffix(
                        s3_key,
                        session_prefix,
                    )
                    if prefix and not relative_key.startswith(prefix):
                        continue
                    remote_files.append((s3_key, relative_key))

            count = 0
            with open_confined_directory(
                local_session_dir,
                create=True,
            ) as destination_descriptor:
                for s3_key, relative_key in remote_files:
                    with confined_atomic_writer(
                        destination_descriptor,
                        relative_key,
                        overwrite=False,
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
        return count

    async def cleanup_stale_local_sessions(
        self,
        local_storage_path: str,
        max_age_hours: float = 24.0,
        skip_session_ids: set[str] | None = None,
    ) -> int:
        """Clean up stale local session directories by syncing to S3 and removing.

        Iterates through local session directories, checks when they were last
        updated (via metadata.json updated_at or directory mtime), and if older
        than max_age_hours:
        1. Syncs all files to S3
        2. Deletes the local directory

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

            # Session is stale - sync to S3 and remove locally
            try:
                # Sync all files to S3
                synced = await self.sync_from_local(session_id, str(session_dir))
                logger.info(
                    f"Synced {synced} files for stale session {session_id[:8]} "
                    f"(last updated: {last_updated})"
                )

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
