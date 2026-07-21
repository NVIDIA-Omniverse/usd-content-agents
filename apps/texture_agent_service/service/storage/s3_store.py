# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import json
import logging
import mimetypes
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, BinaryIO

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, ProfileNotFound

from .base import EVENT_LOG_KEY, METADATA_KEY, SessionStore
from .config import StorageConfig

logger = logging.getLogger(__name__)

_SESSIONS_CACHE_KEY = "sessions"
_SESSION_INDEX_KEY = "sessions-index.json"
_SESSION_METADATA_CACHE_KEY = "session-metadata"
_SESSION_INDEX_RETRY_ATTEMPTS = 20
_OBJECT_CAS_RETRY_ATTEMPTS = 20
_SYNC_UPLOAD_MAX_WORKERS = 32
_SESSION_INDEX_BASE_BACKOFF_SECONDS = 0.05
_SESSION_INDEX_MAX_BACKOFF_SECONDS = 1.0
_CONDITIONAL_WRITE_CONFLICT_CODES = frozenset(
    {"409", "412", "ConditionalRequestConflict", "PreconditionFailed"}
)


class _StreamingBodyReader(io.RawIOBase):
    """File-like wrapper that closes the boto3 response body after streaming."""

    def __init__(self, body: Any) -> None:
        super().__init__()
        self._body = body

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self._body.read()
        return self._body.read(size)

    def close(self) -> None:
        try:
            self._body.close()
        finally:
            super().close()


class S3SessionStore(SessionStore):
    """S3-compatible session storage backend for multi-instance deployments."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
        profile: str | None = None,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        use_path_style: bool = True,
        create_bucket_if_missing: bool = False,
        presign_by_default: bool = True,
        sessions_cache_ttl: int = 5,
        max_pool_connections: int = 64,
    ) -> None:
        if not bucket:
            raise ValueError("bucket is required for S3SessionStore")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._region = region
        self._profile = profile
        self._endpoint_url = endpoint_url
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token.strip() if session_token else None
        if not self._session_token:
            self._session_token = None
        self._use_path_style = use_path_style
        self._create_bucket_if_missing = create_bucket_if_missing
        self.presign_by_default = presign_by_default
        self._sessions_cache_ttl = max(0, sessions_cache_ttl)
        self._max_pool_connections = max(10, max_pool_connections)
        self._sessions_cache: dict[str, tuple[float, list[str]]] = {}
        self._session_metadata_cache: dict[str, tuple[float, list[dict]]] = {}
        self._bucket_ensured = False
        self._client = None
        self._client_lock = threading.Lock()
        self._cache_lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            "S3SessionStore("
            f"bucket={self.bucket!r}, "
            f"prefix={self.prefix!r}, "
            f"region={self._region!r}, "
            f"endpoint_url={self._endpoint_url!r}, "
            f"use_path_style={self._use_path_style!r}, "
            f"create_bucket_if_missing={self._create_bucket_if_missing!r}, "
            f"presign_by_default={self.presign_by_default!r}, "
            f"sessions_cache_ttl={self._sessions_cache_ttl!r}, "
            f"max_pool_connections={self._max_pool_connections!r})"
        )

    @property
    def kind(self) -> str:
        return "s3"

    @classmethod
    def from_config(cls, config: StorageConfig) -> S3SessionStore:
        if not config.s3_bucket:
            raise ValueError("s3_bucket is required in StorageConfig")
        return cls(
            bucket=config.s3_bucket,
            prefix=config.s3_prefix,
            region=config.s3_region,
            profile=config.s3_profile,
            endpoint_url=config.s3_endpoint_url,
            access_key_id=config.s3_access_key_id,
            secret_access_key=config.s3_secret_access_key,
            session_token=config.s3_session_token,
            use_path_style=config.s3_use_path_style,
            create_bucket_if_missing=config.s3_create_bucket,
            presign_by_default=config.s3_presign,
            sessions_cache_ttl=config.s3_sessions_cache_ttl,
            max_pool_connections=config.s3_max_pool_connections,
        )

    def _get_client(self):
        with self._client_lock:
            if self._client is None:
                cfg = BotoConfig(
                    max_pool_connections=self._max_pool_connections,
                    s3={
                        "addressing_style": (
                            "path" if self._use_path_style else "virtual"
                        )
                    },
                )
                client_kwargs = {
                    "region_name": self._region,
                    "endpoint_url": self._endpoint_url,
                    "aws_access_key_id": self._access_key_id,
                    "aws_secret_access_key": self._secret_access_key,
                    "aws_session_token": self._session_token,
                    "config": cfg,
                }
                has_explicit_credentials = bool(
                    self._access_key_id
                    or self._secret_access_key
                    or self._session_token
                )
                if self._profile and not has_explicit_credentials:
                    try:
                        session = boto3.session.Session(profile_name=self._profile)
                        self._client = session.client("s3", **client_kwargs)
                    except ProfileNotFound:
                        logger.warning(
                            "AWS profile %s was not found; falling back to the "
                            "default boto3 credential chain",
                            self._profile,
                        )
                        self._client = boto3.client("s3", **client_kwargs)
                else:
                    self._client = boto3.client("s3", **client_kwargs)
            if self._create_bucket_if_missing and not self._bucket_ensured:
                self._ensure_bucket()
                self._bucket_ensured = True
        return self._client

    def _ensure_bucket(self) -> None:
        client = self._client
        if client is None:
            raise RuntimeError("S3 client was not initialized")
        try:
            client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code not in ("404", "NoSuchBucket"):
                raise
            kwargs: dict[str, Any] = {"Bucket": self.bucket}
            if self._region and self._region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self._region
                }
            try:
                client.create_bucket(**kwargs)
            except ClientError as create_error:
                create_code = create_error.response.get("Error", {}).get("Code", "")
                if create_code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                    logger.info(
                        "S3 bucket %s appeared while ensuring it exists; continuing",
                        self.bucket,
                    )
                    return
                raise

    @staticmethod
    def _is_not_found(error: ClientError) -> bool:
        error_code = error.response.get("Error", {}).get("Code", "")
        return error_code in ("404", "NoSuchKey", "NotFound")

    def _base_key(self, session_id: str) -> str:
        base = (
            f"{self.prefix}/sessions/{session_id}"
            if self.prefix
            else f"sessions/{session_id}"
        )
        return base.strip("/")

    def _key(self, session_id: str, key: str) -> str:
        base = self._base_key(session_id)
        clean_key = key.strip("/")
        return f"{base}/{clean_key}" if clean_key else f"{base}/"

    def _prefix_key(self, session_id: str, prefix: str) -> str:
        key = self._key(session_id, prefix)
        if prefix and prefix.endswith("/") and not key.endswith("/"):
            key = f"{key}/"
        return key

    @staticmethod
    def _prefix_matches(rel_path: str, prefix: str) -> bool:
        clean_prefix = prefix.lstrip("/")
        if not clean_prefix:
            return True
        if clean_prefix.endswith("/"):
            return rel_path.startswith(clean_prefix)
        return rel_path == clean_prefix or rel_path.startswith(f"{clean_prefix}/")

    def _sessions_prefix(self) -> str:
        return f"{self.prefix}/sessions/" if self.prefix else "sessions/"

    def _session_index_key(self) -> str:
        key = (
            f"{self.prefix}/{_SESSION_INDEX_KEY}" if self.prefix else _SESSION_INDEX_KEY
        )
        return key.strip("/")

    @staticmethod
    def _index_retry_sleep(attempt: int) -> None:
        delay = min(
            _SESSION_INDEX_MAX_BACKOFF_SECONDS,
            _SESSION_INDEX_BASE_BACKOFF_SECONDS * (2**attempt),
        )
        time.sleep(delay + random.uniform(0, _SESSION_INDEX_BASE_BACKOFF_SECONDS))

    @staticmethod
    def _summary_from_metadata(session_id: str, metadata: dict) -> dict:
        return {
            "session_id": metadata.get("session_id") or session_id,
            "status": metadata.get("status", "unknown"),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
            "elapsed_seconds": metadata.get("elapsed_seconds", 0),
            "ttl_expires_at": metadata.get("ttl_expires_at"),
            "config": metadata.get("config") or {},
        }

    def _read_session_index_with_etag(self) -> tuple[dict[str, dict], str | None]:
        try:
            response = self._get_client().get_object(
                Bucket=self.bucket,
                Key=self._session_index_key(),
            )
        except ClientError as e:
            if self._is_not_found(e):
                return {}, None
            raise

        body = response["Body"]
        try:
            payload = json.loads(body.read())
        finally:
            body.close()

        sessions = payload.get("sessions") if isinstance(payload, dict) else None
        if not isinstance(sessions, dict):
            return {}, response.get("ETag")
        return (
            {
                session_id: metadata
                for session_id, metadata in sessions.items()
                if isinstance(session_id, str)
                and not session_id.startswith("_")
                and isinstance(metadata, dict)
            },
            response.get("ETag"),
        )

    def _read_session_index(self) -> dict[str, dict]:
        index, _etag = self._read_session_index_with_etag()
        return index

    def _write_session_index(
        self,
        sessions: dict[str, dict],
        etag: str | None = None,
    ) -> bool:
        payload = {
            "updated_at": time.time(),
            "sessions": sessions,
        }
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": self._session_index_key(),
            "Body": json.dumps(payload, indent=2).encode("utf-8"),
            "ContentType": "application/json",
        }
        if etag is None:
            kwargs["IfNoneMatch"] = "*"
        else:
            kwargs["IfMatch"] = etag
        try:
            self._get_client().put_object(**kwargs)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in _CONDITIONAL_WRITE_CONFLICT_CODES:
                return False
            raise

    def _cache_session_metadata(self, metadata: list[dict]) -> None:
        now = time.monotonic()
        session_ids = sorted(
            {
                str(item["session_id"])
                for item in metadata
                if isinstance(item.get("session_id"), str)
            }
        )
        with self._cache_lock:
            self._session_metadata_cache[_SESSION_METADATA_CACHE_KEY] = (
                now,
                [dict(item) for item in metadata],
            )
            self._sessions_cache[_SESSIONS_CACHE_KEY] = (now, session_ids)

    def _list_session_prefixes(self) -> list[str]:
        client = self._get_client()
        sessions: list[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=self._sessions_prefix(),
            Delimiter="/",
        ):
            for prefix_info in page.get("CommonPrefixes", []):
                prefix_path = prefix_info.get("Prefix", "")
                session_id = prefix_path.rstrip("/").split("/")[-1]
                if session_id and not session_id.startswith("_"):
                    sessions.append(session_id)
        return sorted(set(sessions))

    def init_session(self, session_id: str) -> None:
        with self._cache_lock:
            cached = self._sessions_cache.get(_SESSIONS_CACHE_KEY)
            if cached is not None and session_id not in cached[1]:
                sessions = sorted({*cached[1], session_id})
                self._sessions_cache[_SESSIONS_CACHE_KEY] = (cached[0], sessions)

    def delete_session(self, session_id: str) -> None:
        client = self._get_client()
        paginator = client.get_paginator("list_objects_v2")
        prefix = self._key(session_id, "")
        errors: list[dict[str, Any]] = []
        metadata_key = self._key(session_id, METADATA_KEY)
        metadata_objects: list[dict[str, str]] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            metadata_objects.extend(
                obj for obj in objects if obj["Key"] == metadata_key
            )
            artifact_objects = [obj for obj in objects if obj["Key"] != metadata_key]
            if artifact_objects:
                response = client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": artifact_objects},
                )
                errors.extend(response.get("Errors", []))
        if errors:
            raise RuntimeError(
                f"Failed to delete {len(errors)} S3 object(s) for {session_id}"
            )
        if metadata_objects:
            response = client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": metadata_objects},
            )
            errors.extend(response.get("Errors", []))
        if errors:
            raise RuntimeError(
                f"Failed to delete {len(errors)} S3 object(s) for {session_id}"
            )
        self._remove_session_from_index(session_id)
        self.invalidate_sessions_cache()

    def delete_key(self, session_id: str, key: str) -> None:
        self._get_client().delete_object(
            Bucket=self.bucket,
            Key=self._key(session_id, key),
        )

    def list_sessions(self, use_cache: bool = True) -> list[str]:
        now = time.monotonic()
        with self._cache_lock:
            cached = self._sessions_cache.get(_SESSIONS_CACHE_KEY)
            if (
                use_cache
                and cached is not None
                and self._sessions_cache_ttl > 0
                and now - cached[0] <= self._sessions_cache_ttl
            ):
                return list(cached[1])

        index = self._read_session_index()
        sessions = sorted({*index, *self._list_session_prefixes()})
        with self._cache_lock:
            self._sessions_cache[_SESSIONS_CACHE_KEY] = (now, sessions)
        return list(sessions)

    def list_session_metadata(self, use_cache: bool = True) -> list[dict]:
        now = time.monotonic()
        with self._cache_lock:
            cached = self._session_metadata_cache.get(_SESSION_METADATA_CACHE_KEY)
            if (
                use_cache
                and cached is not None
                and self._sessions_cache_ttl > 0
                and now - cached[0] <= self._sessions_cache_ttl
            ):
                return [dict(item) for item in cached[1]]

        index = self._read_session_index()
        metadata_by_id = {session_id: dict(item) for session_id, item in index.items()}
        # Keep the compact index as a best-effort cache, not the source of
        # truth. Prefix discovery preserves legacy/unindexed session IDs
        # without issuing one GetObject per missing prefix on the hot list API.
        for session_id in self._list_session_prefixes():
            if session_id in metadata_by_id:
                continue
            metadata_by_id[session_id] = {"session_id": session_id, "status": "unknown"}
        metadata = list(metadata_by_id.values())
        self._cache_session_metadata(metadata)
        return [dict(item) for item in metadata]

    def update_session_index(self, session_id: str, metadata: dict) -> None:
        for attempt in range(_SESSION_INDEX_RETRY_ATTEMPTS):
            try:
                index, etag = self._read_session_index_with_etag()
                index[session_id] = self._summary_from_metadata(session_id, metadata)
                if not self._write_session_index(index, etag):
                    self._index_retry_sleep(attempt)
                    continue
                self.invalidate_sessions_cache()
                return
            except Exception as exc:
                if attempt == _SESSION_INDEX_RETRY_ATTEMPTS - 1:
                    logger.warning(
                        "Failed to update S3 session index for %s: %s",
                        session_id,
                        exc,
                    )
                    return
                self._index_retry_sleep(attempt)

    def _remove_session_from_index(self, session_id: str) -> None:
        for attempt in range(_SESSION_INDEX_RETRY_ATTEMPTS):
            try:
                index, etag = self._read_session_index_with_etag()
                if session_id not in index:
                    return
                index.pop(session_id, None)
                if not self._write_session_index(index, etag):
                    self._index_retry_sleep(attempt)
                    continue
                self.invalidate_sessions_cache()
                return
            except Exception as exc:
                if attempt == _SESSION_INDEX_RETRY_ATTEMPTS - 1:
                    logger.warning(
                        "Failed to remove %s from S3 session index: %s",
                        session_id,
                        exc,
                    )
                    return
                self._index_retry_sleep(attempt)

    def invalidate_sessions_cache(self) -> None:
        with self._cache_lock:
            self._sessions_cache.clear()
            self._session_metadata_cache.clear()

    def put_bytes(
        self,
        session_id: str,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": self._key(session_id, key),
            "Body": data,
        }
        if content_type:
            kwargs["ContentType"] = content_type
        self._get_client().put_object(**kwargs)

    def put_file(
        self,
        session_id: str,
        key: str,
        file_path: str,
        content_type: str | None = None,
    ) -> None:
        extra: dict[str, Any] = {}
        if content_type:
            extra["ContentType"] = content_type
        self._get_client().upload_file(
            file_path,
            self.bucket,
            self._key(session_id, key),
            ExtraArgs=extra,
        )

    def open_read(self, session_id: str, key: str) -> BinaryIO:
        response = self._get_client().get_object(
            Bucket=self.bucket,
            Key=self._key(session_id, key),
        )
        return _StreamingBodyReader(response["Body"])

    def exists(self, session_id: str, key: str) -> bool:
        try:
            self._get_client().head_object(
                Bucket=self.bucket,
                Key=self._key(session_id, key),
            )
            return True
        except ClientError as e:
            if self._is_not_found(e):
                return False
            raise

    def list_keys(self, session_id: str, prefix: str = "") -> list[str]:
        client = self._get_client()
        keys: list[str] = []
        base = self._key(session_id, "")
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=self._prefix_key(session_id, prefix),
        ):
            for obj in page.get("Contents", []):
                full_key = obj["Key"]
                rel_path = full_key[len(base) :].lstrip("/")
                if rel_path and self._prefix_matches(rel_path, prefix):
                    keys.append(rel_path)
        return sorted(keys)

    def put_json(self, session_id: str, key: str, obj: dict) -> None:
        self.put_bytes(
            session_id,
            key,
            json.dumps(obj, indent=2).encode("utf-8"),
            "application/json",
        )

    def put_json_if_absent(self, session_id: str, key: str, obj: dict) -> bool:
        try:
            self._get_client().put_object(
                Bucket=self.bucket,
                Key=self._key(session_id, key),
                Body=json.dumps(obj, indent=2).encode("utf-8"),
                ContentType="application/json",
                IfNoneMatch="*",
            )
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in _CONDITIONAL_WRITE_CONFLICT_CODES:
                return False
            raise

    def get_json(self, session_id: str, key: str) -> dict | None:
        try:
            stream = self.open_read(session_id, key)
        except ClientError as e:
            if self._is_not_found(e):
                return None
            raise
        try:
            return json.loads(stream.read())
        finally:
            stream.close()

    def _get_json_with_etag(
        self,
        session_id: str,
        key: str,
    ) -> tuple[dict | None, str | None]:
        try:
            response = self._get_client().get_object(
                Bucket=self.bucket,
                Key=self._key(session_id, key),
            )
        except ClientError as e:
            if self._is_not_found(e):
                return None, None
            raise

        body = response["Body"]
        try:
            return json.loads(body.read()), response.get("ETag")
        finally:
            body.close()

    def _get_bytes_with_etag(
        self,
        session_id: str,
        key: str,
    ) -> tuple[bytes | None, str | None]:
        try:
            response = self._get_client().get_object(
                Bucket=self.bucket,
                Key=self._key(session_id, key),
            )
        except ClientError as e:
            if self._is_not_found(e):
                return None, None
            raise

        body = response["Body"]
        try:
            return body.read(), response.get("ETag")
        finally:
            body.close()

    def update_json(
        self,
        session_id: str,
        key: str,
        updater: Callable[[dict], dict | None],
    ) -> dict | None:
        for attempt in range(_OBJECT_CAS_RETRY_ATTEMPTS):
            current, etag = self._get_json_with_etag(session_id, key)
            if current is None:
                return None
            updated = updater(dict(current))
            if updated is None:
                return current

            kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Key": self._key(session_id, key),
                "Body": json.dumps(updated, indent=2).encode("utf-8"),
                "ContentType": "application/json",
            }
            if etag is not None:
                kwargs["IfMatch"] = etag
            try:
                self._get_client().put_object(**kwargs)
                return updated
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in _CONDITIONAL_WRITE_CONFLICT_CODES:
                    self._index_retry_sleep(attempt)
                    continue
                raise
        raise RuntimeError(f"Failed to update {key} for {session_id}: CAS exhausted")

    def delete_json_if_match(
        self,
        session_id: str,
        key: str,
        predicate: Callable[[dict], bool],
    ) -> bool:
        for attempt in range(_OBJECT_CAS_RETRY_ATTEMPTS):
            current, etag = self._get_json_with_etag(session_id, key)
            if current is None or not predicate(dict(current)):
                return False
            if etag is None:
                return False
            try:
                self._get_client().delete_object(
                    Bucket=self.bucket,
                    Key=self._key(session_id, key),
                    IfMatch=etag,
                )
                return True
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in _CONDITIONAL_WRITE_CONFLICT_CODES:
                    self._index_retry_sleep(attempt)
                    continue
                if self._is_not_found(e):
                    return False
                raise
        raise RuntimeError(f"Failed to delete {key} for {session_id}: CAS exhausted")

    def _append_bytes(
        self,
        session_id: str,
        key: str,
        data: bytes,
        content_type: str,
    ) -> None:
        for attempt in range(_OBJECT_CAS_RETRY_ATTEMPTS):
            current, etag = self._get_bytes_with_etag(session_id, key)
            updated = (current or b"") + data
            kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Key": self._key(session_id, key),
                "Body": updated,
                "ContentType": content_type,
            }
            if etag is None:
                kwargs["IfNoneMatch"] = "*"
            else:
                kwargs["IfMatch"] = etag
            try:
                self._get_client().put_object(**kwargs)
                return
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in _CONDITIONAL_WRITE_CONFLICT_CODES:
                    self._index_retry_sleep(attempt)
                    continue
                raise
        raise RuntimeError(f"Failed to append {key} for {session_id}: CAS exhausted")

    def append_event(self, session_id: str, event: dict) -> None:
        # Fallback for callers that cannot flush a local event_log.jsonl file.
        # SessionManager uses put_file() on a throttled cadence for normal
        # worker progress so shared reads remain a single GetObject.
        self._append_bytes(
            session_id,
            EVENT_LOG_KEY,
            json.dumps(event).encode("utf-8") + b"\n",
            "application/x-ndjson",
        )

    @staticmethod
    def _dedupe_events(events: list[dict]) -> list[dict]:
        deduped: list[dict] = []
        seen: set[str] = set()
        for event in events:
            marker = json.dumps(event, sort_keys=True, default=str)
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(event)
        return deduped

    def get_event_log(self, session_id: str) -> list[dict]:
        events: list[dict] = []
        loaded_compact_log = False
        try:
            stream = self.open_read(session_id, EVENT_LOG_KEY)
        except ClientError as e:
            if not self._is_not_found(e):
                raise
        else:
            try:
                events.extend(
                    json.loads(line)
                    for line in stream.read().decode("utf-8").splitlines()
                    if line
                )
                loaded_compact_log = True
            finally:
                stream.close()

        if not loaded_compact_log:
            keys = self.list_keys(session_id, prefix="events/")
            for key in sorted(keys):
                stream = self.open_read(session_id, key)
                try:
                    data = stream.read().decode("utf-8")
                    if key.endswith(".jsonl"):
                        events.extend(
                            json.loads(line) for line in data.splitlines() if line
                        )
                    else:
                        events.append(json.loads(data))
                finally:
                    stream.close()
        return self._dedupe_events(events)

    def make_public_url(
        self,
        session_id: str,
        key: str,
        expires_seconds: int = 3600,
    ) -> str | None:
        if not self.presign_by_default:
            return None
        return self._get_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": self._key(session_id, key)},
            ExpiresIn=expires_seconds,
        )

    def sync_from_local(
        self,
        session_id: str,
        local_session_dir: str,
        prefix: str = "",
    ) -> int:
        local_dir = Path(local_session_dir)
        if not local_dir.exists():
            return 0

        client = self._get_client()
        if prefix:
            prefix_path = local_dir / prefix.strip("/")
            if prefix_path.is_file():
                candidates = (prefix_path,)
            elif prefix_path.is_dir():
                candidates = prefix_path.rglob("*")
            else:
                return 0
        else:
            candidates = local_dir.rglob("*")

        uploads: list[tuple[Path, str, dict[str, Any]]] = []
        for file_path in candidates:
            if not file_path.is_file():
                continue
            rel_path = str(file_path.relative_to(local_dir))
            if not self._prefix_matches(rel_path, prefix):  # pragma: no cover
                continue
            content_type, _ = mimetypes.guess_type(str(file_path))
            extra: dict[str, Any] = {}
            if content_type:
                extra["ContentType"] = content_type
            uploads.append((file_path, rel_path, extra))

        if not uploads:
            return 0

        def _upload(item: tuple[Path, str, dict[str, Any]]) -> None:
            file_path, rel_path, extra = item
            client.upload_file(
                str(file_path),
                self.bucket,
                self._key(session_id, rel_path),
                ExtraArgs=extra,
            )

        max_workers = min(
            len(uploads),
            self._max_pool_connections,
            _SYNC_UPLOAD_MAX_WORKERS,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_upload, item) for item in uploads]
            for future in as_completed(futures):
                future.result()
        return len(uploads)

    def sync_to_local(
        self,
        session_id: str,
        local_session_dir: str,
        prefix: str = "",
    ) -> int:
        local_dir = Path(local_session_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        base = self._key(session_id, "")
        client = self._get_client()
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=self._prefix_key(session_id, prefix),
        ):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                rel_path = s3_key[len(base) :].lstrip("/")
                if not self._prefix_matches(rel_path, prefix):  # pragma: no cover
                    continue
                local_path = local_dir / rel_path
                resolved_local_path = local_path.resolve()
                if not resolved_local_path.is_relative_to(local_dir.resolve()):
                    logger.warning(
                        "Skipping S3 object outside session directory: %s",
                        s3_key,
                    )
                    continue
                local_path.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(self.bucket, s3_key, str(local_path))
                count += 1
        return count
