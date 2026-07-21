# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for shared texture session storage hardening."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError, ProfileNotFound
from filelock import Timeout

from ...service.runtime import bus as bus_module
from ...service.runtime.bus import EventBus
from ...service.runtime.events import ProgressEvent, StepState
from ...service.session.manager import (
    _CLEANUP_LOCK_KEY,
    _MAINTENANCE_SESSION_ID,
    SessionManager,
)
from ...service.storage import (
    CANCEL_KEY,
    EVENT_LOG_KEY,
    METADATA_KEY,
    WORKER_RESERVATION_KEY,
    LocalSessionStore,
    S3SessionStore,
)
from ...service.storage import s3_store as s3_store_module
from ...service.storage.config import StorageConfig
from ...service.workers.executor import _sync_prefix_to_store


class _FakeS3Body:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            data = self._data
            self._data = b""
            return data
        data = self._data[:size]
        self._data = self._data[size:]
        return data

    def close(self) -> None:
        self.closed = True


def test_storage_config_reads_env_at_instantiation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TA_STORAGE_KIND", "s3")
    monkeypatch.setenv("TA_STORAGE_S3_BUCKET", "bucket")

    config = StorageConfig()

    assert config.kind == "s3"
    assert config.s3_bucket == "bucket"


def test_storage_config_honors_wu_s3_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TA_STORAGE_S3_BUCKET", raising=False)
    monkeypatch.delenv("TA_STORAGE_S3_REGION", raising=False)
    monkeypatch.delenv("TA_STORAGE_S3_PROFILE", raising=False)
    monkeypatch.setenv("WU_S3_BUCKET", "wu-bucket")
    monkeypatch.setenv("WU_S3_REGION", "us-east-2")
    monkeypatch.setenv("WU_S3_PROFILE", "wu-profile")

    config = StorageConfig()

    assert config.s3_bucket == "wu-bucket"
    assert config.s3_region == "us-east-2"
    assert config.s3_profile == "wu-profile"


def test_storage_config_normalizes_empty_s3_session_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TA_STORAGE_S3_SESSION_TOKEN", "")

    config = StorageConfig()

    assert config.s3_session_token is None


def test_storage_config_repr_hides_s3_credentials() -> None:
    config = StorageConfig(
        s3_access_key_id="ACCESS",
        s3_secret_access_key="SECRET",
        s3_session_token="TOKEN",
    )

    config_repr = repr(config)

    assert "ACCESS" not in config_repr
    assert "SECRET" not in config_repr
    assert "TOKEN" not in config_repr


def test_s3_session_store_repr_hides_credentials() -> None:
    store = S3SessionStore(
        bucket="bucket",
        prefix="texture",
        access_key_id="ACCESS",
        secret_access_key="SECRET",
        session_token="TOKEN",
    )

    store_repr = repr(store)

    assert "S3SessionStore" in store_repr
    assert "bucket" in store_repr
    assert "ACCESS" not in store_repr
    assert "SECRET" not in store_repr
    assert "TOKEN" not in store_repr


def test_s3_session_store_explicit_credentials_ignore_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    fake_client = object()

    class UnexpectedProfileSession:
        def __init__(self, profile_name: str) -> None:
            raise AssertionError("explicit credentials should bypass profile session")

    def fake_client_factory(service_name: str, **kwargs: Any) -> object:
        calls["service_name"] = service_name
        calls["client_kwargs"] = kwargs
        return fake_client

    monkeypatch.setattr(
        s3_store_module.boto3.session,
        "Session",
        UnexpectedProfileSession,
    )
    monkeypatch.setattr(s3_store_module.boto3, "client", fake_client_factory)

    store = S3SessionStore(
        bucket="bucket",
        region="us-west-2",
        profile="dev-profile",
        access_key_id="EXPLICIT_ACCESS",
        secret_access_key="EXPLICIT_SECRET",
        session_token="EXPLICIT_TOKEN",
    )

    assert store._get_client() is fake_client
    assert calls["service_name"] == "s3"
    assert calls["client_kwargs"]["region_name"] == "us-west-2"
    assert calls["client_kwargs"]["aws_access_key_id"] == "EXPLICIT_ACCESS"
    assert calls["client_kwargs"]["aws_secret_access_key"] == "EXPLICIT_SECRET"
    assert calls["client_kwargs"]["aws_session_token"] == "EXPLICIT_TOKEN"


def test_s3_session_store_normalizes_empty_session_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    fake_client = object()

    def fake_client_factory(service_name: str, **kwargs: Any) -> object:
        calls["service_name"] = service_name
        calls["client_kwargs"] = kwargs
        return fake_client

    monkeypatch.setattr(s3_store_module.boto3, "client", fake_client_factory)

    store = S3SessionStore(
        bucket="bucket",
        access_key_id="EXPLICIT_ACCESS",
        secret_access_key="EXPLICIT_SECRET",
        session_token="",
    )

    assert store._get_client() is fake_client
    assert calls["service_name"] == "s3"
    assert calls["client_kwargs"]["aws_session_token"] is None


def test_s3_session_store_missing_profile_falls_back_to_default_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    fake_client = object()

    class MissingProfileSession:
        def __init__(self, profile_name: str) -> None:
            calls["profile_name"] = profile_name
            raise ProfileNotFound(profile=profile_name)

    def fake_client_factory(service_name: str, **kwargs: Any) -> object:
        calls["service_name"] = service_name
        calls["client_kwargs"] = kwargs
        return fake_client

    monkeypatch.setattr(
        s3_store_module.boto3.session,
        "Session",
        MissingProfileSession,
    )
    monkeypatch.setattr(s3_store_module.boto3, "client", fake_client_factory)

    store = S3SessionStore(
        bucket="bucket",
        region="us-west-2",
        profile="missing-profile",
    )

    assert store._get_client() is fake_client
    assert calls["profile_name"] == "missing-profile"
    assert calls["service_name"] == "s3"
    assert calls["client_kwargs"]["region_name"] == "us-west-2"
    assert calls["client_kwargs"]["aws_access_key_id"] is None


def test_s3_client_initialization_is_thread_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"client": 0, "head_bucket": 0}

    class FakeClient:
        def head_bucket(self, **kwargs: Any) -> None:
            calls["head_bucket"] += 1

    def fake_client(*args: Any, **kwargs: Any) -> FakeClient:
        calls["client"] += 1
        return FakeClient()

    monkeypatch.setattr(s3_store_module.boto3, "client", fake_client)
    store = S3SessionStore(bucket="bucket", create_bucket_if_missing=True)

    with ThreadPoolExecutor(max_workers=8) as executor:
        clients = list(executor.map(lambda _: store._get_client(), range(32)))

    assert len({id(client) for client in clients}) == 1
    assert calls == {"client": 1, "head_bucket": 1}


def test_s3_bucket_create_tolerates_concurrent_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"head_bucket": 0, "create_bucket": 0}

    class FakeClient:
        def head_bucket(self, **kwargs: Any) -> None:
            calls["head_bucket"] += 1
            raise ClientError(
                {"Error": {"Code": "NoSuchBucket"}},
                "HeadBucket",
            )

        def create_bucket(self, **kwargs: Any) -> None:
            calls["create_bucket"] += 1
            raise ClientError(
                {"Error": {"Code": "BucketAlreadyOwnedByYou"}},
                "CreateBucket",
            )

    monkeypatch.setattr(s3_store_module.boto3, "client", lambda *a, **kw: FakeClient())
    store = S3SessionStore(bucket="bucket", create_bucket_if_missing=True)

    store._get_client()

    assert calls == {"head_bucket": 1, "create_bucket": 1}


def test_s3_client_uses_configured_connection_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeClient:
        pass

    def fake_client(*args: Any, **kwargs: Any) -> FakeClient:
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(s3_store_module.boto3, "client", fake_client)
    store = S3SessionStore(bucket="bucket", max_pool_connections=48)

    store._get_client()

    assert captured["config"].max_pool_connections == 48


def test_s3_open_read_closes_streaming_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _FakeS3Body(b'{"ok": true}')

    class FakeClient:
        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            return {"Body": body}

    monkeypatch.setattr(s3_store_module.boto3, "client", lambda *a, **kw: FakeClient())
    store = S3SessionStore(bucket="bucket")

    stream = store.open_read("session", "key.json")
    assert stream.read() == b'{"ok": true}'
    assert body.closed is False
    stream.close()
    assert body.closed is True


def test_local_store_prefix_matching_uses_directory_boundaries(tmp_path: Path) -> None:
    store = LocalSessionStore(str(tmp_path))

    store.put_bytes("session", "cache/output.usd", b"usd")
    store.put_bytes("session", "cache_discovery/legacy.json", b"json")

    assert store.list_keys("session", prefix="cache") == ["cache/output.usd"]
    assert store.list_keys("session", prefix="cache/") == ["cache/output.usd"]
    assert store.sync_to_local("session", str(tmp_path / "local"), prefix="cache") == 1
    assert not (tmp_path / "local" / "cache_discovery").exists()


def test_local_store_rejects_keys_outside_session_dir(tmp_path: Path) -> None:
    store = LocalSessionStore(str(tmp_path))
    store.init_session("session")

    with pytest.raises(ValueError, match="Invalid key"):
        store.put_bytes("session", "../escape.txt", b"nope")
    with pytest.raises(ValueError, match="Invalid key"):
        store.open_read("session", "../escape.txt")

    assert not (tmp_path / "escape.txt").exists()


def test_s3_open_read_uses_boto_read_all_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StrictBody:
        def __init__(self) -> None:
            self.read_args: list[Any] = []

        def read(self, amt: Any = None) -> bytes:
            self.read_args.append(amt)
            return b"payload"

        def close(self) -> None:
            return None

    body = StrictBody()

    class FakeClient:
        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            return {"Body": body}

    monkeypatch.setattr(s3_store_module.boto3, "client", lambda *a, **kw: FakeClient())
    store = S3SessionStore(bucket="bucket")

    assert store.open_read("session", "key.bin").read() == b"payload"
    assert body.read_args == [None]


def test_s3_event_log_appends_to_compact_jsonl_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}
            self.etags: dict[str, str] = {}
            self.put_keys: list[str] = []
            self.put_kwargs: list[dict[str, Any]] = []
            self.get_keys: list[str] = []

        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            key = kwargs["Key"]
            self.get_keys.append(key)
            if key not in self.objects:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey"}},
                    "GetObject",
                )
            return {"Body": _FakeS3Body(self.objects[key]), "ETag": self.etags[key]}

        def put_object(self, **kwargs: Any) -> None:
            key = kwargs["Key"]
            if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
                raise ClientError(
                    {"Error": {"Code": "PreconditionFailed"}},
                    "PutObject",
                )
            if "IfMatch" in kwargs and kwargs["IfMatch"] != self.etags.get(key):
                raise ClientError(
                    {"Error": {"Code": "PreconditionFailed"}},
                    "PutObject",
                )
            self.put_keys.append(key)
            self.put_kwargs.append(kwargs)
            self.objects[key] = kwargs["Body"]
            self.etags[key] = f'"etag-{len(self.put_keys)}"'

        def get_paginator(self, name: str) -> Any:
            assert name == "list_objects_v2"

            class FakePaginator:
                def __init__(self, objects: dict[str, bytes]) -> None:
                    self.objects = objects

                def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
                    prefix = kwargs["Prefix"]
                    return [
                        {
                            "Contents": [
                                {"Key": key}
                                for key in sorted(self.objects)
                                if key.startswith(prefix)
                            ]
                        }
                    ]

            return FakePaginator(self.objects)

    client = FakeClient()
    monkeypatch.setattr(s3_store_module.boto3, "client", lambda *a, **kw: client)
    store = S3SessionStore(bucket="bucket")

    store.append_event("session", {"step": "prepare_uvs", "state": "running"})
    store.append_event("session", {"step": "prepare_uvs", "state": "completed"})

    assert client.put_keys == [
        "sessions/session/event_log.jsonl",
        "sessions/session/event_log.jsonl",
    ]
    assert client.put_kwargs[0]["IfNoneMatch"] == "*"
    assert client.put_kwargs[1]["IfMatch"] == '"etag-1"'
    assert store.get_event_log("session") == [
        {"step": "prepare_uvs", "state": "running"},
        {"step": "prepare_uvs", "state": "completed"},
    ]
    assert list(client.objects) == ["sessions/session/event_log.jsonl"]


def test_s3_get_event_log_skips_legacy_listing_when_compact_log_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["Key"] == "sessions/session/event_log.jsonl"
            return {
                "Body": _FakeS3Body(
                    b'{"step":"prepare_uvs","state":"running"}\n'
                    b'{"step":"prepare_uvs","state":"completed"}\n'
                )
            }

        def get_paginator(self, name: str) -> Any:
            raise AssertionError("legacy event prefixes should not be listed")

    monkeypatch.setattr(s3_store_module.boto3, "client", lambda *a, **kw: FakeClient())
    store = S3SessionStore(bucket="bucket")

    assert store.get_event_log("session") == [
        {"step": "prepare_uvs", "state": "running"},
        {"step": "prepare_uvs", "state": "completed"},
    ]


def test_s3_session_listing_uses_compact_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}
            self.get_keys: list[str] = []

        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            key = kwargs["Key"]
            self.get_keys.append(key)
            if key not in self.objects:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey"}},
                    "GetObject",
                )
            return {"Body": _FakeS3Body(self.objects[key])}

        def put_object(self, **kwargs: Any) -> None:
            self.objects[kwargs["Key"]] = kwargs["Body"]

        def get_paginator(self, name: str) -> Any:
            assert name == "list_objects_v2"

            class FakePaginator:
                def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
                    return [{"CommonPrefixes": []}]

            return FakePaginator()

    client = FakeClient()
    monkeypatch.setattr(s3_store_module.boto3, "client", lambda *a, **kw: client)
    store = S3SessionStore(bucket="bucket")
    store.update_session_index(
        "session-a",
        {
            "session_id": "session-a",
            "status": "completed",
            "created_at": "2026-05-22T00:00:00+00:00",
            "updated_at": "2026-05-22T00:01:00+00:00",
            "config": {"original_filename": "a.usda"},
        },
    )
    client.get_keys.clear()

    rows = store.list_session_metadata(use_cache=False)

    assert rows == [
        {
            "session_id": "session-a",
            "status": "completed",
            "created_at": "2026-05-22T00:00:00+00:00",
            "updated_at": "2026-05-22T00:01:00+00:00",
            "elapsed_seconds": 0,
            "ttl_expires_at": None,
            "config": {"original_filename": "a.usda"},
        }
    ]
    assert client.get_keys == ["sessions-index.json"]
    assert all(not key.endswith("/session.json") for key in client.get_keys)


def test_s3_session_listing_marks_unindexed_prefixes_unknown_without_gets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.get_keys: list[str] = []
            self.objects = {
                "sessions-index.json": json.dumps(
                    {
                        "sessions": {
                            "indexed": {
                                "session_id": "indexed",
                                "status": "completed",
                            }
                        }
                    }
                ).encode("utf-8"),
                "sessions/legacy/session.json": json.dumps(
                    {
                        "session_id": "legacy",
                        "status": "failed",
                        "created_at": "2026-05-21T00:00:00+00:00",
                        "updated_at": "2026-05-21T01:00:00+00:00",
                        "ttl_expires_at": "2026-05-22T00:00:00+00:00",
                        "config": {"original_filename": "legacy.usda"},
                    }
                ).encode("utf-8"),
            }

        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            key = kwargs["Key"]
            self.get_keys.append(key)
            if key not in self.objects:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey"}},
                    "GetObject",
                )
            return {"Body": _FakeS3Body(self.objects[key]), "ETag": '"index"'}

        def get_paginator(self, name: str) -> Any:
            assert name == "list_objects_v2"

            class FakePaginator:
                def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
                    return [
                        {
                            "CommonPrefixes": [
                                {"Prefix": "sessions/indexed/"},
                                {"Prefix": "sessions/legacy/"},
                                {"Prefix": "sessions/_maintenance/"},
                            ]
                        }
                    ]

            return FakePaginator()

    client = FakeClient()
    monkeypatch.setattr(s3_store_module.boto3, "client", lambda *a, **kw: client)
    store = S3SessionStore(bucket="bucket")

    rows = store.list_session_metadata(use_cache=False)

    assert {row["session_id"]: row["status"] for row in rows} == {
        "indexed": "completed",
        "legacy": "unknown",
    }
    legacy_row = next(row for row in rows if row["session_id"] == "legacy")
    assert legacy_row == {"session_id": "legacy", "status": "unknown"}
    assert client.get_keys == ["sessions-index.json"]


def test_s3_update_json_uses_etag_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.put_kwargs: dict[str, Any] = {}

        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "Body": _FakeS3Body(b'{"session_id": "s", "status": "pending"}'),
                "ETag": '"etag-1"',
            }

        def put_object(self, **kwargs: Any) -> None:
            self.put_kwargs = kwargs

    client = FakeClient()
    monkeypatch.setattr(s3_store_module.boto3, "client", lambda *a, **kw: client)
    store = S3SessionStore(bucket="bucket")

    updated = store.update_json(
        "s",
        METADATA_KEY,
        lambda metadata: {**metadata, "status": "running"},
    )

    assert updated is not None
    assert updated["status"] == "running"
    assert client.put_kwargs["IfMatch"] == '"etag-1"'


def test_s3_update_json_retries_more_than_five_cas_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.put_attempts = 0
            self.body = b'{"session_id": "s", "status": "pending"}'

        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            return {"Body": _FakeS3Body(self.body), "ETag": '"etag-1"'}

        def put_object(self, **kwargs: Any) -> None:
            self.put_attempts += 1
            if self.put_attempts <= 6:
                raise ClientError(
                    {"Error": {"Code": "PreconditionFailed"}},
                    "PutObject",
                )
            self.body = kwargs["Body"]

    client = FakeClient()
    monkeypatch.setattr(s3_store_module.boto3, "client", lambda *a, **kw: client)
    monkeypatch.setattr(s3_store_module.time, "sleep", lambda seconds: None)
    store = S3SessionStore(bucket="bucket")

    updated = store.update_json(
        "s",
        METADATA_KEY,
        lambda metadata: {**metadata, "status": "running"},
    )

    assert updated is not None
    assert updated["status"] == "running"
    assert client.put_attempts == 7


def test_s3_delete_json_if_match_uses_etag_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.delete_kwargs: dict[str, Any] = {}

        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "Body": _FakeS3Body(b'{"owner_token": "owner-1"}'),
                "ETag": '"etag-1"',
            }

        def delete_object(self, **kwargs: Any) -> None:
            self.delete_kwargs = kwargs

    client = FakeClient()
    monkeypatch.setattr(s3_store_module.boto3, "client", lambda *a, **kw: client)
    store = S3SessionStore(bucket="bucket")

    deleted = store.delete_json_if_match(
        "s",
        WORKER_RESERVATION_KEY,
        lambda marker: marker.get("owner_token") == "owner-1",
    )

    assert deleted is True
    assert client.delete_kwargs["IfMatch"] == '"etag-1"'


def test_botocore_s3_put_object_model_supports_preconditions() -> None:
    import botocore.session

    service_model = botocore.session.get_session().get_service_model("s3")
    put_shape = service_model.operation_model("PutObject").input_shape
    delete_shape = service_model.operation_model("DeleteObject").input_shape

    assert "IfMatch" in put_shape.members
    assert "IfNoneMatch" in put_shape.members
    assert "IfMatch" in delete_shape.members


def test_s3_delete_session_raises_on_partial_delete_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey"}},
                "GetObject",
            )

        def get_paginator(self, name: str) -> Any:
            assert name == "list_objects_v2"

            class FakePaginator:
                def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
                    return [
                        {
                            "Contents": [
                                {"Key": "sessions/s/session.json"},
                                {"Key": "sessions/s/cache/output.usd"},
                            ]
                        }
                    ]

            return FakePaginator()

        def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "Errors": [
                    {
                        "Key": "sessions/s/cache/output.usd",
                        "Code": "AccessDenied",
                    }
                ]
            }

    monkeypatch.setattr(s3_store_module.boto3, "client", lambda *a, **kw: FakeClient())
    store = S3SessionStore(bucket="bucket")

    with pytest.raises(RuntimeError, match="Failed to delete 1 S3 object"):
        store.delete_session("s")


def test_s3_delete_session_keeps_metadata_and_index_on_partial_delete_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.objects = {
                "sessions-index.json": json.dumps(
                    {"sessions": {"s": {"session_id": "s", "status": "completed"}}}
                ).encode(),
                "sessions/s/session.json": b'{"session_id": "s"}',
                "sessions/s/cache/output.usd": b"usd",
            }
            self.index_updated = False

        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            key = kwargs["Key"]
            if key not in self.objects:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey"}},
                    "GetObject",
                )
            return {"Body": _FakeS3Body(self.objects[key]), "ETag": '"etag"'}

        def put_object(self, **kwargs: Any) -> dict[str, Any]:
            self.index_updated = kwargs["Key"] == "sessions-index.json"
            self.objects[kwargs["Key"]] = kwargs["Body"]
            return {}

        def get_paginator(self, name: str) -> Any:
            assert name == "list_objects_v2"
            client = self

            class FakePaginator:
                def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
                    prefix = kwargs["Prefix"]
                    return [
                        {
                            "Contents": [
                                {"Key": key}
                                for key in sorted(client.objects)
                                if key.startswith(prefix)
                            ]
                        }
                    ]

            return FakePaginator()

        def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
            errors = []
            for obj in kwargs["Delete"]["Objects"]:
                key = obj["Key"]
                if key == "sessions/s/cache/output.usd":
                    errors.append({"Key": key, "Code": "AccessDenied"})
                    continue
                self.objects.pop(key, None)
            return {"Errors": errors} if errors else {}

    client = FakeClient()
    monkeypatch.setattr(s3_store_module.boto3, "client", lambda *a, **kw: client)
    store = S3SessionStore(bucket="bucket")

    with pytest.raises(RuntimeError, match="Failed to delete 1 S3 object"):
        store.delete_session("s")

    assert "sessions/s/session.json" in client.objects
    assert "sessions/s/cache/output.usd" in client.objects
    assert "sessions-index.json" in client.objects
    assert (
        json.loads(client.objects["sessions-index.json"])["sessions"]["s"]["session_id"]
        == "s"
    )
    assert client.index_updated is False


def test_s3_sync_to_local_skips_objects_that_escape_session_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.downloads: list[tuple[str, str]] = []

        def get_paginator(self, name: str) -> Any:
            assert name == "list_objects_v2"

            class FakePaginator:
                def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
                    return [
                        {
                            "Contents": [
                                {"Key": "sessions/s/../escape.txt"},
                                {"Key": "sessions/s/cache/output.usdz"},
                            ]
                        }
                    ]

            return FakePaginator()

        def download_file(self, bucket: str, key: str, filename: str) -> None:
            self.downloads.append((key, filename))
            Path(filename).write_text("downloaded", encoding="utf-8")

    client = FakeClient()
    monkeypatch.setattr(s3_store_module.boto3, "client", lambda *a, **kw: client)
    store = S3SessionStore(bucket="bucket")
    local_dir = tmp_path / "local"

    assert store.sync_to_local("s", str(local_dir)) == 1

    assert client.downloads == [
        ("sessions/s/cache/output.usdz", str(local_dir / "cache" / "output.usdz"))
    ]
    assert not (tmp_path / "escape.txt").exists()
    assert (local_dir / "cache" / "output.usdz").read_text(encoding="utf-8") == (
        "downloaded"
    )


def test_s3_sync_from_local_uploads_files_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.uploads: list[tuple[str, str, dict[str, Any]]] = []
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def upload_file(
            self,
            filename: str,
            bucket: str,
            key: str,
            **kwargs: Any,
        ) -> None:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.02)
                with self.lock:
                    self.uploads.append((filename, key, kwargs["ExtraArgs"]))
            finally:
                with self.lock:
                    self.active -= 1

    local_dir = tmp_path / "local"
    textures_dir = local_dir / "cache" / "textures"
    textures_dir.mkdir(parents=True)
    for idx in range(8):
        (textures_dir / f"texture_{idx}.png").write_bytes(b"png")

    client = FakeClient()
    monkeypatch.setattr(s3_store_module.boto3, "client", lambda *a, **kw: client)
    store = S3SessionStore(bucket="bucket", max_pool_connections=4)

    assert store.sync_from_local("s", str(local_dir), "cache/textures/") == 8

    assert client.max_active > 1
    assert sorted(key for _, key, _ in client.uploads) == [
        f"sessions/s/cache/textures/texture_{idx}.png" for idx in range(8)
    ]


def test_shared_manager_update_session_uses_store_update_json(
    tmp_path: Path,
) -> None:
    class CasAssertingStore(LocalSessionStore):
        def __init__(self, root_dir: str) -> None:
            super().__init__(root_dir)
            self.update_json_calls: list[tuple[str, str]] = []

        def update_json(
            self,
            session_id: str,
            key: str,
            updater: Any,
        ) -> dict | None:
            self.update_json_calls.append((session_id, key))
            return super().update_json(session_id, key, updater)

    store = CasAssertingStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=store)
    session_id = "cas-update-session"

    manager.create_session(session_id)
    store.update_json_calls.clear()
    manager.update_session(session_id, {"status": "running"})
    manager.update_step_progress(
        session_id,
        "generate_textures",
        {"percent": 10},
    )
    manager.mark_step_completed(
        session_id,
        "generate_textures",
        {"textures_generated": 2},
    )

    metadata = manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "running"
    assert metadata["completed_steps"][0]["stats"] == {"textures_generated": 2}
    assert store.update_json_calls == [
        (session_id, METADATA_KEY),
        (session_id, METADATA_KEY),
        (session_id, METADATA_KEY),
    ]


def test_shared_manager_update_session_does_not_hydrate_before_cas(
    tmp_path: Path,
) -> None:
    class NoPreHydrateStore(LocalSessionStore):
        def __init__(self, root_dir: str) -> None:
            super().__init__(root_dir)
            self.in_update_json = False

        def get_json(self, session_id: str, key: str) -> dict | None:
            if key == METADATA_KEY and not self.in_update_json:
                raise AssertionError("metadata should not hydrate before CAS update")
            return super().get_json(session_id, key)

        def update_json(
            self,
            session_id: str,
            key: str,
            updater: Any,
        ) -> dict | None:
            self.in_update_json = True
            try:
                return super().update_json(session_id, key, updater)
            finally:
                self.in_update_json = False

    store = NoPreHydrateStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=store)
    session_id = "no-pre-hydrate"
    store.init_session(session_id)
    store.put_json(
        session_id,
        METADATA_KEY,
        {
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "pending",
        },
    )

    manager.update_session(session_id, {"status": "running"})

    local_metadata = manager.get_session_dir(session_id) / METADATA_KEY
    assert local_metadata.is_file()
    assert json.loads(local_metadata.read_text(encoding="utf-8"))["status"] == "running"


def test_shared_manager_update_missing_session_does_not_create_session_dir(
    tmp_path: Path,
) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=shared_store)

    manager.update_session("missing-shared", {"status": "cancelled"})

    assert not manager.get_session_dir("missing-shared").exists()


def test_update_session_propagates_shared_store_cas_exhaustion(
    tmp_path: Path,
) -> None:
    class CasExhaustingStore(LocalSessionStore):
        def update_json(
            self,
            session_id: str,
            key: str,
            updater: Any,
        ) -> dict | None:
            raise RuntimeError("CAS exhausted")

    shared_store = CasExhaustingStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=shared_store)
    session_id = "cas-exhausted"
    manager.create_session(session_id)

    with pytest.raises(RuntimeError, match="CAS exhausted"):
        manager.update_session(session_id, {"status": "running"})

    assert manager.get_session_metadata(session_id)["status"] == "pending"


def test_shared_session_event_log_merges_local_and_shared_events(
    tmp_path: Path,
) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=shared_store)
    session_id = "shared-events"
    local_event = {
        "timestamp": "2026-05-22T02:00:00+00:00",
        "step": "prepare_uvs",
        "state": "running",
    }
    remote_event = {
        "timestamp": "2026-05-22T02:00:01+00:00",
        "step": "prepare_uvs",
        "state": "completed",
    }

    manager.create_session(session_id)
    manager.append_event(session_id, local_event)
    shared_store.append_event(session_id, remote_event)

    assert manager.get_event_log(session_id) == [local_event, remote_event]


def test_shared_session_event_log_merge_preserves_timestamp_order(
    tmp_path: Path,
) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=shared_store)
    session_id = "shared-events-ordered"
    local_event = {
        "timestamp": "2026-05-22T02:00:00+00:00",
        "step": "prepare_uvs",
        "state": "running",
    }
    remote_event = {
        "timestamp": "2026-05-22T02:00:01+00:00",
        "step": "prepare_uvs",
        "state": "completed",
    }

    manager.create_session(session_id)
    shared_store.append_event(session_id, remote_event)
    (manager.get_session_dir(session_id) / EVENT_LOG_KEY).write_text(
        json.dumps(local_event) + "\n",
        encoding="utf-8",
    )

    assert manager.get_event_log(session_id) == [local_event, remote_event]


def test_append_event_serializes_local_event_log_writes(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path, ttl_hours=1)
    session_id = "locked-event-log"
    manager.create_session(session_id)

    def append_event(index: int) -> None:
        manager.append_event(session_id, {"event_index": index})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append_event, range(100)))

    events = manager.get_event_log(session_id)

    assert len(events) == 100
    assert {event["event_index"] for event in events} == set(range(100))
    assert (tmp_path / ".locks").is_dir()
    assert not (manager.get_session_dir(session_id) / "event_log.jsonl.lock").exists()


def test_shared_event_log_flushes_compact_file_on_throttled_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingStore(LocalSessionStore):
        def __init__(self, root_dir: str) -> None:
            super().__init__(root_dir)
            self.put_file_keys: list[str] = []

        def put_file(
            self,
            session_id: str,
            key: str,
            file_path: str,
            content_type: str | None = None,
        ) -> None:
            self.put_file_keys.append(key)
            super().put_file(session_id, key, file_path, content_type)

    shared_store = RecordingStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=shared_store)
    session_id = "shared-event-flush"
    clock = iter([0.0, 0.5, 1.0, 2.1])
    monkeypatch.setattr(
        "apps.texture_agent_service.service.session.manager.time.monotonic",
        lambda: next(clock),
    )

    manager.create_session(session_id)
    manager.append_event(session_id, {"step": "prepare_uvs", "state": "running"})
    manager.append_event(session_id, {"step": "prepare_uvs", "state": "running"})
    manager.append_event(session_id, {"step": "prepare_uvs", "state": "running"})
    manager.append_event(session_id, {"step": "prepare_uvs", "state": "running"})

    assert shared_store.put_file_keys == [EVENT_LOG_KEY, EVENT_LOG_KEY]
    assert shared_store.get_event_log(session_id) == [
        {"step": "prepare_uvs", "state": "running"},
        {"step": "prepare_uvs", "state": "running"},
        {"step": "prepare_uvs", "state": "running"},
        {"step": "prepare_uvs", "state": "running"},
    ]


def test_get_session_metadata_retries_transient_store_client_errors(
    tmp_path: Path,
) -> None:
    class FlakyMetadataStore(LocalSessionStore):
        def __init__(self, root_dir: str) -> None:
            super().__init__(root_dir)
            self.failures_remaining = 2

        def get_json(self, session_id: str, key: str) -> dict | None:
            if key == METADATA_KEY and self.failures_remaining > 0:
                self.failures_remaining -= 1
                raise ClientError(
                    {"Error": {"Code": "SlowDown"}},
                    "GetObject",
                )
            return super().get_json(session_id, key)

    store = FlakyMetadataStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=store)

    manager.create_session("flaky-metadata")

    metadata = manager.get_session_metadata("flaky-metadata")
    assert metadata is not None
    assert metadata["session_id"] == "flaky-metadata"
    assert store.failures_remaining == 0


def test_shared_worker_reservation_is_atomic_and_owner_scoped(
    tmp_path: Path,
) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    pod_a = SessionManager(tmp_path / "pod_a", ttl_hours=1, store=shared_store)
    pod_b = SessionManager(tmp_path / "pod_b", ttl_hours=1, store=shared_store)
    session_id = "atomic-reservation"

    pod_a.create_session(session_id)
    worker_lock = pod_a.acquire_worker_lock(session_id, timeout=0)
    marker = shared_store.get_json(session_id, WORKER_RESERVATION_KEY)
    assert isinstance(marker, dict)
    owner_token = marker["owner_token"]

    with pytest.raises(Timeout):
        pod_b.acquire_worker_lock(session_id, timeout=0)
    assert (
        shared_store.get_json(session_id, WORKER_RESERVATION_KEY)["owner_token"]
        == owner_token
    )

    shared_store.delete_key(session_id, WORKER_RESERVATION_KEY)
    shared_store.put_json(
        session_id,
        WORKER_RESERVATION_KEY,
        {"owner_token": "peer-token", "created_at": datetime.now(UTC).isoformat()},
    )
    pod_a.release_worker_lock(worker_lock, session_id)
    assert (
        shared_store.get_json(session_id, WORKER_RESERVATION_KEY)["owner_token"]
        == "peer-token"
    )


def test_shared_worker_heartbeat_is_owner_scoped(tmp_path: Path) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=shared_store)
    session_id = "owner-heartbeat"

    manager.create_session(session_id)
    worker_lock = manager.acquire_worker_lock(session_id, timeout=0)
    owner_token = manager.get_worker_reservation_owner_token(session_id)
    assert owner_token is not None
    shared_store.put_json(
        session_id,
        WORKER_RESERVATION_KEY,
        {
            "owner_token": "peer-token",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
    )

    try:
        assert manager.heartbeat_worker(session_id, owner_token=owner_token) is False
        marker = shared_store.get_json(session_id, WORKER_RESERVATION_KEY)
        assert marker is not None
        assert marker["owner_token"] == "peer-token"
        assert marker["updated_at"] == "2026-01-01T00:00:00+00:00"
    finally:
        manager.release_worker_lock(worker_lock, session_id)


def test_shared_worker_reservation_prefers_marker_heartbeat(
    tmp_path: Path,
) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    pod_a = SessionManager(tmp_path / "pod_a", ttl_hours=1, store=shared_store)
    pod_b = SessionManager(tmp_path / "pod_b", ttl_hours=1, store=shared_store)
    session_id = "fresh-marker-stale-metadata"

    pod_a.create_session(session_id)
    worker_lock = pod_a.acquire_worker_lock(session_id, timeout=0)
    try:
        stale_timestamp = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
        metadata = pod_a.get_session_metadata(session_id)
        assert metadata is not None
        metadata.update(
            {
                "status": "running",
                "updated_at": stale_timestamp,
                "ttl_expires_at": stale_timestamp,
            }
        )
        shared_store.put_json(session_id, METADATA_KEY, metadata)

        assert pod_b.is_worker_active(session_id) is True
        assert shared_store.get_json(session_id, WORKER_RESERVATION_KEY) is not None
    finally:
        pod_a.release_worker_lock(worker_lock, session_id)


def test_stale_shared_worker_reservation_clear_is_owner_scoped(
    tmp_path: Path,
) -> None:
    class OwnerScopedStore(LocalSessionStore):
        def __init__(self, root_dir: str) -> None:
            super().__init__(root_dir)
            self.delete_matches: list[tuple[str, str | None, bool]] = []

        def delete_key(self, session_id: str, key: str) -> None:
            if key == WORKER_RESERVATION_KEY:
                raise AssertionError("worker reservation delete must be owner-scoped")
            super().delete_key(session_id, key)

        def delete_json_if_match(
            self,
            session_id: str,
            key: str,
            predicate: Any,
        ) -> bool:
            current = self.get_json(session_id, key)
            matched = current is not None and predicate(dict(current))
            owner_token = (
                current.get("owner_token") if isinstance(current, dict) else None
            )
            self.delete_matches.append((key, owner_token, matched))
            if not matched:
                return False
            LocalSessionStore.delete_key(self, session_id, key)
            return True

    shared_store = OwnerScopedStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=shared_store)
    session_id = "stale-owner-scoped"
    stale_timestamp = (datetime.now(UTC) - timedelta(hours=3)).isoformat()

    manager.create_session(session_id)
    metadata = manager.get_session_metadata(session_id)
    assert metadata is not None
    metadata.update(
        {
            "status": "running",
            "updated_at": stale_timestamp,
            "ttl_expires_at": stale_timestamp,
        }
    )
    shared_store.put_json(session_id, METADATA_KEY, metadata)
    shared_store.put_json(
        session_id,
        WORKER_RESERVATION_KEY,
        {
            "owner_token": "stale-owner",
            "created_at": stale_timestamp,
            "updated_at": stale_timestamp,
        },
    )

    assert manager.is_worker_active(session_id) is False
    assert shared_store.get_json(session_id, WORKER_RESERVATION_KEY) is None
    assert shared_store.delete_matches == [
        (WORKER_RESERVATION_KEY, "stale-owner", True)
    ]


def test_stale_shared_cleanup_lock_clear_is_owner_scoped(
    tmp_path: Path,
) -> None:
    class OwnerScopedStore(LocalSessionStore):
        def __init__(self, root_dir: str) -> None:
            super().__init__(root_dir)
            self.delete_matches: list[tuple[str, str | None, bool]] = []

        def delete_key(self, session_id: str, key: str) -> None:
            if key == _CLEANUP_LOCK_KEY:
                raise AssertionError("cleanup lock delete must be owner-scoped")
            super().delete_key(session_id, key)

        def delete_json_if_match(
            self,
            session_id: str,
            key: str,
            predicate: Any,
        ) -> bool:
            current = self.get_json(session_id, key)
            matched = current is not None and predicate(dict(current))
            owner_token = (
                current.get("owner_token") if isinstance(current, dict) else None
            )
            self.delete_matches.append((key, owner_token, matched))
            if not matched:
                return False
            LocalSessionStore.delete_key(self, session_id, key)
            return True

    shared_store = OwnerScopedStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=shared_store)
    stale_timestamp = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
    shared_store.put_json(
        _MAINTENANCE_SESSION_ID,
        _CLEANUP_LOCK_KEY,
        {
            "owner_token": "stale-cleanup-owner",
            "created_at": stale_timestamp,
            "updated_at": stale_timestamp,
        },
    )

    owner_token = manager._acquire_shared_cleanup_lock()

    assert owner_token is not None
    assert shared_store.delete_matches == [
        (_CLEANUP_LOCK_KEY, "stale-cleanup-owner", True)
    ]
    marker = shared_store.get_json(_MAINTENANCE_SESSION_ID, _CLEANUP_LOCK_KEY)
    assert marker is not None
    assert marker["owner_token"] == owner_token


def test_request_cancellation_retries_shared_marker_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FlakyCancelStore(LocalSessionStore):
        def __init__(self, root_dir: str) -> None:
            super().__init__(root_dir)
            self.failures_remaining = 2

        def put_bytes(
            self,
            session_id: str,
            key: str,
            data: bytes,
            content_type: str | None = None,
        ) -> None:
            if key == CANCEL_KEY and self.failures_remaining > 0:
                self.failures_remaining -= 1
                raise OSError("transient put failure")
            super().put_bytes(session_id, key, data, content_type)

    shared_store = FlakyCancelStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=shared_store)
    session_id = "cancel-retry"
    sleeps: list[float] = []
    monkeypatch.setattr(
        "apps.texture_agent_service.service.session.manager.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    manager.create_session(session_id)
    manager.request_cancellation(session_id)

    assert shared_store.failures_remaining == 0
    assert sleeps == [0.05, 0.1]
    assert shared_store.exists(session_id, CANCEL_KEY) is True
    assert manager.is_cancelled(session_id) is True
    assert manager.get_session_metadata(session_id)["status"] == "cancelling"


def test_request_cancellation_surfaces_shared_marker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCancelStore(LocalSessionStore):
        def put_bytes(
            self,
            session_id: str,
            key: str,
            data: bytes,
            content_type: str | None = None,
        ) -> None:
            if key == CANCEL_KEY:
                raise OSError("persistent put failure")
            super().put_bytes(session_id, key, data, content_type)

    shared_store = FailingCancelStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=shared_store)
    session_id = "cancel-fails"
    sleeps: list[float] = []
    monkeypatch.setattr(
        "apps.texture_agent_service.service.session.manager.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    manager.create_session(session_id)
    with pytest.raises(RuntimeError, match="shared cancellation marker"):
        manager.request_cancellation(session_id)

    assert sleeps == pytest.approx([0.05, 0.1, 0.15, 0.2])
    assert shared_store.exists(session_id, CANCEL_KEY) is False
    assert manager.is_cancelled(session_id) is False
    assert manager.get_session_metadata(session_id)["status"] == "pending"


async def test_event_bus_heartbeats_shared_worker_reservation(
    tmp_path: Path,
) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=shared_store)
    session_id = "reservation-heartbeat"
    manager.create_session(session_id)
    worker_lock = manager.acquire_worker_lock(session_id, timeout=0)
    old_timestamp = (datetime.now(UTC) - timedelta(minutes=20)).isoformat()
    marker = shared_store.get_json(session_id, WORKER_RESERVATION_KEY)
    assert marker is not None
    marker["updated_at"] = old_timestamp
    shared_store.put_json(session_id, WORKER_RESERVATION_KEY, marker)

    bus = EventBus(session_manager=manager)
    try:
        await bus.emit(
            ProgressEvent(
                session_id=session_id,
                step="prepare_uvs",
                state=StepState.RUNNING,
            )
        )
        updated_marker = shared_store.get_json(session_id, WORKER_RESERVATION_KEY)
        assert updated_marker is not None
        assert updated_marker["updated_at"] > old_timestamp
    finally:
        manager.release_worker_lock(worker_lock, session_id)


async def test_event_bus_deduplicates_concurrent_shared_worker_heartbeats(
    tmp_path: Path,
) -> None:
    class CountingManager(SessionManager):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.heartbeat_calls = 0

        def heartbeat_worker(
            self,
            session_id: str,
            owner_token: str | None = None,
        ) -> bool:
            self.heartbeat_calls += 1
            return True

    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    manager = CountingManager(tmp_path / "pod", ttl_hours=1, store=shared_store)
    session_id = "dedupe-heartbeat"
    manager.create_session(session_id)
    worker_lock = manager.acquire_worker_lock(session_id, timeout=0)
    bus = EventBus(session_manager=manager)

    try:
        await asyncio.gather(
            *(bus._heartbeat_worker_if_due(session_id) for _ in range(8))
        )

        assert manager.heartbeat_calls == 1
    finally:
        manager.release_worker_lock(worker_lock, session_id)


async def test_event_bus_persists_live_progress_to_shared_metadata(
    tmp_path: Path,
) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    worker_manager = SessionManager(
        tmp_path / "worker", ttl_hours=1, store=shared_store
    )
    remote_manager = SessionManager(
        tmp_path / "remote", ttl_hours=1, store=shared_store
    )
    session_id = "remote-progress"
    worker_manager.create_session(session_id)
    bus = EventBus(session_manager=worker_manager)

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.RUNNING,
            current=1,
            total=4,
            percent=25,
            message="Generating texture",
        )
    )

    remote_metadata = remote_manager.get_session_metadata(session_id)
    assert remote_metadata is not None
    assert remote_metadata["status"] == "running"
    assert remote_metadata["current_step"]["name"] == "generate_textures"
    assert remote_metadata["current_step"]["progress"]["percent"] == 25
    assert remote_metadata["overall_progress"]["percent"] == 33

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.COMPLETED,
            percent=100,
            extra={"textures_generated": 2},
        )
    )

    remote_metadata = remote_manager.get_session_metadata(session_id)
    assert remote_metadata is not None
    assert remote_metadata["current_step"] is None
    assert remote_metadata["overall_progress"]["percent"] == 75
    assert remote_metadata["completed_steps"][0]["stats"] == {"textures_generated": 2}


async def test_event_bus_debounces_shared_live_progress_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bus_module, "_LIVE_METADATA_FLUSH_INTERVAL_SECONDS", 1.0)

    class CountingManager(SessionManager):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.live_updates: list[dict[str, Any]] = []
            self.exists_checks = 0

        def session_exists(self, session_id: str) -> bool:
            self.exists_checks += 1
            time.sleep(0.01)
            return super().session_exists(session_id)

        def update_session(
            self,
            session_id: str,
            updates: dict[str, Any],
            *,
            update_index: bool = True,
        ) -> None:
            if not update_index:
                self.live_updates.append(updates)
            super().update_session(
                session_id,
                updates,
                update_index=update_index,
            )

    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    manager = CountingManager(tmp_path / "worker", ttl_hours=1, store=shared_store)
    session_id = "debounced-progress"
    manager.create_session(session_id)
    bus = EventBus(session_manager=manager)

    for percent in (10, 20, 30):
        await bus.emit(
            ProgressEvent(
                session_id=session_id,
                step="generate_textures",
                state=StepState.RUNNING,
                current=percent,
                total=100,
                percent=percent,
            )
        )

    assert len(manager.live_updates) == 1
    assert manager.exists_checks == 1
    assert manager.live_updates[0]["current_step"]["progress"]["percent"] == 10

    deadline = asyncio.get_running_loop().time() + 2.0
    while len(manager.live_updates) < 2:
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail("timed out waiting for debounced live metadata flush")
        await asyncio.sleep(0.01)

    assert len(manager.live_updates) == 2
    assert manager.live_updates[-1]["current_step"]["progress"]["percent"] == 30


async def test_event_bus_forces_shared_live_metadata_on_step_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bus_module, "_LIVE_METADATA_FLUSH_INTERVAL_SECONDS", 60.0)

    class CountingManager(SessionManager):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.live_update_count = 0

        def update_session(
            self,
            session_id: str,
            updates: dict[str, Any],
            *,
            update_index: bool = True,
        ) -> None:
            if not update_index:
                self.live_update_count += 1
            super().update_session(
                session_id,
                updates,
                update_index=update_index,
            )

    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    manager = CountingManager(tmp_path / "worker", ttl_hours=1, store=shared_store)
    session_id = "force-completion"
    manager.create_session(session_id)
    bus = EventBus(session_manager=manager)

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.RUNNING,
            percent=10,
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.RUNNING,
            percent=20,
        )
    )

    assert manager.live_update_count == 1

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.COMPLETED,
            percent=100,
            extra={"textures_generated": 2},
        )
    )

    metadata = manager.get_session_metadata(session_id)
    assert metadata is not None
    assert manager.live_update_count == 2
    assert metadata["current_step"] is None
    assert metadata["overall_progress"]["percent"] == 75


async def test_event_bus_keeps_emit_nonfatal_on_shared_exists_error(
    tmp_path: Path,
) -> None:
    class FlakyExistsManager(SessionManager):
        def session_exists(self, session_id: str) -> bool:
            raise RuntimeError("s3 is temporarily unavailable")

    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    manager = FlakyExistsManager(tmp_path / "worker", ttl_hours=1, store=shared_store)
    session_id = "exists-transient-error"
    manager.create_session(session_id)
    bus = EventBus(session_manager=manager)

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.RUNNING,
            percent=10,
        )
    )

    event = await asyncio.wait_for(bus.get_queue(session_id).get(), timeout=1)
    assert event.step == "generate_textures"
    assert bus.get_snapshot(session_id)["status"] == "running"


async def test_final_step_completion_waits_for_pipeline_completed_event(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=1)
    session_id = "sync-before-complete"
    manager.create_session(session_id)
    bus = EventBus(session_manager=manager)

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="render",
            state=StepState.RUNNING,
            percent=99,
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="render",
            state=StepState.COMPLETED,
            percent=100,
        )
    )

    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "running"
    assert snapshot["overall_progress"]["percent"] == 100
    assert manager.get_session_metadata(session_id)["status"] == "running"

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="render",
            state=StepState.COMPLETED,
            percent=100,
            extra={"pipeline_completed": True},
        )
    )

    assert manager.get_session_metadata(session_id)["status"] == "completed"


async def test_event_bus_does_not_persist_non_terminal_over_terminal_status(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=1)
    session_id = "terminal-cancel-race"
    manager.create_session(session_id)
    manager.update_session(session_id, {"status": "completed"})
    bus = EventBus(session_manager=manager)

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="render",
            state=StepState.CANCELLING,
            message="Pipeline cancellation requested",
        )
    )

    assert manager.get_session_metadata(session_id)["status"] == "completed"


def test_cancel_poll_survives_shared_store_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FlakyCancelStore(LocalSessionStore):
        def exists(self, session_id: str, key: str) -> bool:
            if key == CANCEL_KEY:
                raise RuntimeError("temporary S3 failure")
            return super().exists(session_id, key)

    store = FlakyCancelStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=store)
    session_id = "cancel-fallback"

    manager.create_session(session_id)
    assert manager.is_cancelled(session_id) is False
    assert "Failed to check shared cancellation marker" in caplog.text

    (manager.get_session_dir(session_id) / CANCEL_KEY).write_text("", encoding="utf-8")
    assert manager.is_cancelled(session_id) is True


def test_remote_running_session_blocks_ttl_cleanup(tmp_path: Path) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    pod_a = SessionManager(tmp_path / "pod_a", ttl_hours=1, store=shared_store)
    pod_b = SessionManager(tmp_path / "pod_b", ttl_hours=1, store=shared_store)
    session_id = "remote-running"
    pod_a.create_session(session_id)
    pod_a.update_session(
        session_id,
        {
            "status": "running",
            "ttl_expires_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        },
    )

    assert not (pod_b.get_session_dir(session_id) / "session.json").exists()
    assert pod_b.is_worker_active(session_id) is True
    assert pod_b.cleanup_expired_sessions() == []
    assert pod_a.session_exists(session_id) is True


def test_hydrated_remote_running_session_blocks_ttl_cleanup(tmp_path: Path) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    pod_a = SessionManager(tmp_path / "pod_a", ttl_hours=1, store=shared_store)
    pod_b = SessionManager(tmp_path / "pod_b", ttl_hours=1, store=shared_store)
    session_id = "hydrated-remote-running"
    pod_a.create_session(session_id)

    pod_b.sync_from_store(session_id)
    assert (pod_b.get_session_dir(session_id) / METADATA_KEY).is_file()

    pod_a.update_session(
        session_id,
        {
            "status": "running",
            "ttl_expires_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        },
    )

    assert pod_b.is_worker_active(session_id) is True
    assert pod_b.cleanup_expired_sessions() == []
    assert pod_a.session_exists(session_id) is True


def test_stale_remote_running_session_can_expire(tmp_path: Path) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    pod_a = SessionManager(tmp_path / "pod_a", ttl_hours=1, store=shared_store)
    pod_b = SessionManager(tmp_path / "pod_b", ttl_hours=1, store=shared_store)
    session_id = "stale-remote-running"
    pod_a.create_session(session_id)

    old_timestamp = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
    metadata = pod_a.get_session_metadata(session_id)
    assert metadata is not None
    metadata.update(
        {
            "status": "running",
            "updated_at": old_timestamp,
            "ttl_expires_at": old_timestamp,
        }
    )
    pod_a.store.put_json(session_id, METADATA_KEY, metadata)

    assert pod_b.is_worker_active(session_id) is False
    assert pod_b.cleanup_expired_sessions() == [session_id]
    assert pod_a.session_exists(session_id) is False


def test_cleanup_expired_sessions_heartbeats_shared_cleanup_lock(
    tmp_path: Path,
) -> None:
    class RecordingStore(LocalSessionStore):
        def __init__(self, root_dir: str) -> None:
            super().__init__(root_dir)
            self.cleanup_heartbeats = 0

        def update_json(
            self,
            session_id: str,
            key: str,
            updater: Any,
        ) -> dict | None:
            if session_id == _MAINTENANCE_SESSION_ID and key == _CLEANUP_LOCK_KEY:
                self.cleanup_heartbeats += 1
            return super().update_json(session_id, key, updater)

    shared_store = RecordingStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=shared_store)
    expired_at = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    session_ids = ["cleanup-heartbeat-a", "cleanup-heartbeat-b"]
    for session_id in session_ids:
        manager.create_session(session_id)
        manager.update_session(
            session_id,
            {
                "status": "completed",
                "ttl_expires_at": expired_at,
            },
        )

    assert manager.cleanup_expired_sessions() == session_ids
    assert shared_store.cleanup_heartbeats >= len(session_ids)


def test_stale_remote_running_session_uses_heartbeat_before_ttl(tmp_path: Path) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    pod_a = SessionManager(tmp_path / "pod_a", ttl_hours=24, store=shared_store)
    pod_b = SessionManager(tmp_path / "pod_b", ttl_hours=24, store=shared_store)
    session_id = "stale-heartbeat-before-ttl"
    pod_a.create_session(session_id)

    old_timestamp = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
    future_ttl = (datetime.now(UTC) + timedelta(hours=21)).isoformat()
    metadata = pod_a.get_session_metadata(session_id)
    assert metadata is not None
    metadata.update(
        {
            "status": "running",
            "updated_at": old_timestamp,
            "ttl_expires_at": future_ttl,
        }
    )
    pod_a.store.put_json(session_id, METADATA_KEY, metadata)

    assert pod_b.is_worker_active(session_id) is False


async def test_event_bus_checks_shared_s3_session_exists_outside_lock() -> None:
    class AssertingManager:
        def __init__(self) -> None:
            self.store = type("Store", (), {"kind": "s3"})()
            self.bus: EventBus | None = None
            self.session_exists_calls = 0

        def uses_shared_store(self) -> bool:
            return True

        def session_exists(self, session_id: str) -> bool:
            assert self.bus is not None
            assert self.bus._lock.locked() is False
            self.session_exists_calls += 1
            return True

        def update_session(self, session_id: str, updates: dict[str, Any]) -> None:
            return None

        def append_event(self, session_id: str, event: dict[str, Any]) -> None:
            return None

    manager = AssertingManager()
    bus = EventBus(session_manager=manager)
    manager.bus = bus

    await bus.emit(
        ProgressEvent(
            session_id="lock-free-exists",
            step="prepare_uvs",
            state=StepState.RUNNING,
        )
    )

    assert manager.session_exists_calls == 1


async def test_event_bus_prunes_sessions_deleted_by_another_instance(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = "remote-deleted"
    manager.create_session(session_id)
    bus = EventBus(session_manager=manager)
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.RUNNING,
        )
    )
    assert bus.get_snapshot(session_id) is not None

    manager.delete_session(session_id)
    assert await bus.cleanup_orphaned_sessions() == [session_id]
    assert bus.get_snapshot(session_id) is None


async def test_sync_prefix_to_store_raises_after_retries() -> None:
    class FailingManager:
        def __init__(self) -> None:
            self.calls = 0

        def sync_to_store(self, session_id: str, prefix: str) -> int:
            self.calls += 1
            raise RuntimeError("s3 unavailable")

    manager = FailingManager()

    with pytest.raises(RuntimeError, match="Failed to sync cache/output/"):
        await _sync_prefix_to_store(
            manager,
            "session",
            "cache/output/",
            attempts=2,
        )

    assert manager.calls == 2
