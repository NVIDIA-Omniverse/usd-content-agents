# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import AsyncIterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

import pytest
from botocore.exceptions import ClientError

from ...service.storage import config as storage_config_mod
from ...service.storage import s3_store as s3_store_mod
from ...service.storage.base import (
    METADATA_KEY,
    CompletedSessionSnapshot,
    SessionGeneration,
    SessionGenerationConflictError,
    SessionGenerationOwnershipError,
    SessionNotCompletedError,
)
from ...service.storage.config import StorageConfig
from ...service.storage.s3_store import S3SessionStore
from ...service.workers.predict_executor import detect_predict_mode


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _Paginator:
    def __init__(self, client: _FakeS3Client) -> None:
        self.client = client

    async def paginate(
        self,
        *,
        Bucket: str,  # noqa: N803 - boto-style fake signature
        Prefix: str,  # noqa: N803 - boto-style fake signature
        Delimiter: str | None = None,  # noqa: N803 - boto-style fake signature
    ) -> AsyncIterator[dict]:  # noqa: N803 - boto-style fake signature
        assert Bucket == self.client.bucket
        keys = sorted(key for key in self.client.objects if key.startswith(Prefix))
        if Delimiter:
            common = set()
            for key in keys:
                tail = key[len(Prefix) :]
                if Delimiter in tail:
                    common.add(Prefix + tail.split(Delimiter, 1)[0] + Delimiter)
            yield {"CommonPrefixes": [{"Prefix": item} for item in sorted(common)]}
        else:
            yield {"Contents": [{"Key": key} for key in keys]}


class _FakeS3Client:
    def __init__(self, bucket: str = "bucket") -> None:
        self.bucket = bucket
        self.objects: dict[str, bytes] = {}
        self.uploads: list[tuple[str, dict]] = []
        self.downloads: list[str] = []
        self.deleted: list[str] = []
        self.bucket_exists = True
        self.created_bucket = False
        self.head_bucket_error: ClientError | None = None
        self.raise_on_download: Exception | None = None
        self.raise_on_get: Exception | None = None
        self.ignore_conditions = False

    def _etag(self, key: str) -> str:
        return f'"{hashlib.sha256(self.objects[key]).hexdigest()}"'

    async def head_bucket(self, *, Bucket: str) -> None:  # noqa: N803
        assert Bucket == self.bucket
        if self.head_bucket_error is not None:
            raise self.head_bucket_error
        if not self.bucket_exists:
            raise ClientError({"Error": {"Code": "NoSuchBucket"}}, "HeadBucket")

    async def create_bucket(self, *, Bucket: str) -> None:  # noqa: N803
        assert Bucket == self.bucket
        self.created_bucket = True
        self.bucket_exists = True

    def get_paginator(self, name: str) -> _Paginator:
        assert name == "list_objects_v2"
        return _Paginator(self)

    async def put_object(self, **kwargs) -> dict:
        if (
            not self.ignore_conditions
            and kwargs.get("IfNoneMatch") == "*"
            and kwargs["Key"] in self.objects
        ):
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed"}},
                "PutObject",
            )
        if (
            not self.ignore_conditions
            and "IfMatch" in kwargs
            and (
                kwargs["Key"] not in self.objects
                or kwargs["IfMatch"] != self._etag(kwargs["Key"])
            )
        ):
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed"}},
                "PutObject",
            )
        self.objects[kwargs["Key"]] = kwargs["Body"]
        self.uploads.append((kwargs["Key"], kwargs))
        return {"ETag": self._etag(kwargs["Key"])}

    async def upload_file(
        self,
        file_path: str,
        bucket: str,
        key: str,
        *,
        ExtraArgs: dict,  # noqa: N803 - boto-style fake signature
    ) -> None:  # noqa: N803
        assert bucket == self.bucket
        self.objects[key] = Path(file_path).read_bytes()
        self.uploads.append((key, ExtraArgs))

    async def upload_fileobj(
        self,
        file_obj: BinaryIO,
        bucket: str,
        key: str,
        *,
        ExtraArgs: dict,  # noqa: N803 - boto-style fake signature
    ) -> None:
        assert bucket == self.bucket
        self.objects[key] = file_obj.read()
        self.uploads.append((key, ExtraArgs))

    async def copy_object(
        self,
        *,
        Bucket: str,  # noqa: N803 - boto-style fake signature
        Key: str,  # noqa: N803 - boto-style fake signature
        CopySource: dict[str, str],  # noqa: N803 - boto-style fake signature
    ) -> None:
        assert Bucket == self.bucket
        assert CopySource["Bucket"] == self.bucket
        self.objects[Key] = self.objects[CopySource["Key"]]

    async def get_object(self, *, Bucket: str, Key: str) -> dict:  # noqa: N803
        assert Bucket == self.bucket
        if self.raise_on_get is not None:
            raise self.raise_on_get
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": _Body(self.objects[Key]), "ETag": self._etag(Key)}

    async def head_object(self, *, Bucket: str, Key: str) -> None:  # noqa: N803
        assert Bucket == self.bucket
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    async def delete_object(self, *, Bucket: str, Key: str) -> None:  # noqa: N803
        assert Bucket == self.bucket
        self.deleted.append(Key)
        self.objects.pop(Key, None)

    async def download_file(self, bucket: str, key: str, local_path: str) -> None:
        assert bucket == self.bucket
        if self.raise_on_download is not None:
            raise self.raise_on_download
        Path(local_path).write_bytes(self.objects[key])
        self.downloads.append(key)

    async def download_fileobj(
        self,
        bucket: str,
        key: str,
        file_obj: BinaryIO,
    ) -> None:
        assert bucket == self.bucket
        if self.raise_on_download is not None:
            raise self.raise_on_download
        file_obj.write(self.objects[key])
        self.downloads.append(key)

    async def generate_presigned_url(
        self,
        operation: str,
        *,
        Params: dict,  # noqa: N803 - boto-style fake signature
        ExpiresIn: int,  # noqa: N803 - boto-style fake signature
    ) -> str:  # noqa: N803
        assert operation == "get_object"
        return (
            f"https://example.test/{Params['Bucket']}/{Params['Key']}?exp={ExpiresIn}"
        )


class _ClientContext:
    def __init__(self, client: _FakeS3Client) -> None:
        self.client = client

    async def __aenter__(self) -> _FakeS3Client:
        return self.client

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeSession:
    def __init__(self, client: _FakeS3Client) -> None:
        self.client_obj = client
        self.calls: list[dict] = []

    def client(self, service_name: str, **kwargs) -> _ClientContext:
        assert service_name == "s3"
        self.calls.append(kwargs)
        return _ClientContext(self.client_obj)


def _store(client: _FakeS3Client | None = None, **kwargs) -> S3SessionStore:
    store = S3SessionStore(bucket="bucket", prefix="wu", **kwargs)
    store._session = _FakeSession(client or _FakeS3Client())  # type: ignore[assignment]
    # Most tests target session semantics directly. Capability probing has
    # dedicated coverage below and would otherwise add unrelated scratch keys.
    store._conditional_writes_verified = True
    return store


def _read_local_marker(path: Path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return s3_store_mod._read_local_snapshot_marker(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.asyncio
async def test_reserved_s3_keys_are_invisible_and_writes_fail_loudly() -> None:
    client = _FakeS3Client()
    store = _store(client)
    client.objects[store._key("s1", "output/result.json")] = b"{}"
    client.objects[store._key("s1", "cache/.pipeline_temp/config.json")] = b"{}"
    client.objects[store._key("s1", r"cache\.pipeline_temp\windows.json")] = b"{}"

    assert await store.list_keys("s1") == ["output/result.json"]
    assert await store.list_keys("s1", r"cache\.pipeline_temp") == []
    assert not await store.exists("s1", "cache/.pipeline_temp/config.json")
    assert await store.get_json("s1", "cache/.pipeline_temp/config.json") is None
    assert await store.make_public_url("s1", "cache/.pipeline_temp/config.json") is None
    with pytest.raises(FileNotFoundError):
        await store.open_read("s1", r"cache\.pipeline_temp\windows.json")
    with pytest.raises(ValueError, match="reserved"):
        await store.put_bytes("s1", "cache/.pipeline_temp/new.json", b"{}")
    with pytest.raises(ValueError, match="reserved"):
        await store.delete_key("s1", "cache/.pipeline_temp/config.json")

    await store.put_bytes("s1", ".cancel", b"")
    await store.delete_key("s1", ".cancel")
    assert not await store.exists("s1", ".cancel")


@pytest.mark.asyncio
async def test_s3_store_init_from_config_and_client_setup() -> None:
    with pytest.raises(ValueError, match="bucket"):
        S3SessionStore(bucket="")
    with pytest.raises(ValueError, match="s3_bucket"):
        S3SessionStore.from_config(StorageConfig(kind="s3", s3_bucket=""))

    client = _FakeS3Client()
    client.bucket_exists = False
    store = _store(
        client,
        region="us-east-1",
        endpoint_url="http://minio",
        access_key_id="ak",
        secret_access_key="sk",
        session_token="tok",
        use_path_style=False,
    )

    async with store._client() as yielded:
        assert yielded is client
    async with store._client():
        pass

    assert client.created_bucket is True
    assert store.kind == "s3"
    assert store._key("s1", "a/b.txt") == "wu/sessions/s1/a/b.txt"
    assert len(store._session.calls) == 2  # type: ignore[attr-defined]

    cfg = StorageConfig(
        kind="s3",
        s3_bucket="bucket",
        s3_prefix="prefix",
        s3_region="us-west-2",
        s3_endpoint_url="http://endpoint",
        s3_access_key_id="key",
        s3_secret_access_key="secret",
        s3_session_token="token",
        s3_use_path_style=False,
        s3_create_bucket=False,
        s3_presign=False,
        s3_sessions_cache_ttl=3,
        s3_generation_retention=7,
    )
    from_cfg = S3SessionStore.from_config(cfg)
    assert from_cfg.bucket == "bucket"
    assert from_cfg.prefix == "prefix"
    assert from_cfg.presign_by_default is False
    assert from_cfg._generation_retention == 7


def test_storage_config_empty_integer_environment_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PA_STORAGE_S3_GENERATION_RETENTION", "")
    monkeypatch.setenv("PA_STORAGE_S3_SESSIONS_CACHE_TTL", "")
    assert storage_config_mod._env_int("PA_STORAGE_S3_GENERATION_RETENTION", 5) == 5
    assert storage_config_mod._env_int("PA_STORAGE_S3_SESSIONS_CACHE_TTL", 5) == 5


@pytest.mark.asyncio
async def test_s3_generation_retention_fails_closed() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        S3SessionStore(bucket="bucket", generation_retention=0)


def test_s3_generation_state_and_lease_edge_shapes() -> None:
    assert S3SessionStore._generation_state({"status": "ready"}) == "idle"
    assert S3SessionStore._lease_is_expired({}) is True
    assert (
        S3SessionStore._lease_is_expired({"generation_lease_expires_at": "not-a-date"})
        is True
    )
    assert (
        S3SessionStore._lease_is_expired(
            {
                "generation_lease_expires_at": (
                    datetime.now() + timedelta(minutes=5)
                ).isoformat()
            }
        )
        is False
    )
    assert S3SessionStore._lease_needs_renewal({}) is True
    assert (
        S3SessionStore._lease_needs_renewal(
            {"generation_lease_expires_at": "not-a-date"}
        )
        is True
    )
    assert (
        S3SessionStore._lease_needs_renewal(
            {
                "generation_lease_expires_at": (
                    datetime.now() + timedelta(minutes=10)
                ).isoformat()
            }
        )
        is False
    )


def test_local_snapshot_marker_protocol_compatibility_and_validation(
    tmp_path: Path,
) -> None:
    marker_dir = tmp_path / ".pipeline_temp"
    marker_dir.mkdir()
    marker_path = marker_dir / "s3-snapshot-generation.json"

    marker_path.write_text(
        json.dumps(
            {
                "protocol": "physics-local-s3-snapshot.v2",
                "generation": 2,
                "owner_id": "owner-2",
                "prefixes": ["cache/", "cache/dataset/"],
            }
        ),
        encoding="utf-8",
    )
    marker = _read_local_marker(tmp_path)
    assert marker is not None
    assert marker.protocol == "physics-local-s3-snapshot.v2"
    assert marker.local_generation == marker.publication_generation
    assert marker.prefixes == ("cache/",)

    marker_path.write_text(
        json.dumps(
            {
                "protocol": "physics-local-s3-snapshot.v3",
                "generation": 3,
                "owner_id": "owner-3",
                "publication_generation": None,
                "publication_owner_id": None,
                "prefixes": [""],
            }
        ),
        encoding="utf-8",
    )
    marker = _read_local_marker(tmp_path)
    assert marker is not None
    assert marker.publication_generation is None

    invalid_markers = [
        b"[]",
        b"not-json",
        json.dumps(
            {
                "protocol": "physics-local-s3-snapshot.v3",
                "generation": 3,
                "owner_id": "owner-3",
                "publication_generation": 2,
                "publication_owner_id": None,
                "prefixes": [],
            }
        ).encode(),
        json.dumps(
            {
                "protocol": "physics-local-s3-snapshot.v3",
                "generation": 3,
                "owner_id": "owner-3",
                "publication_generation": 2,
                "publication_owner_id": "owner-2",
                "prefixes": ["../unsafe"],
            }
        ).encode(),
        b"{" + (b" " * 4096) + b"}",
    ]
    for raw_marker in invalid_markers:
        marker_path.write_bytes(raw_marker)
        assert _read_local_marker(tmp_path) is None

    marker_path.unlink()
    assert _read_local_marker(tmp_path) is None
    marker_path.mkdir()
    assert _read_local_marker(tmp_path) is None
    marker_path.rmdir()
    marker_dir.rmdir()
    assert _read_local_marker(tmp_path) is None


def test_local_snapshot_marker_closes_descriptor_when_fdopen_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)

    def fail_fdopen(*_args, **_kwargs):
        raise OSError("fdopen failed")

    monkeypatch.setattr(s3_store_mod.os, "fdopen", fail_fdopen)
    try:
        with pytest.raises(OSError, match="fdopen failed"):
            s3_store_mod._write_local_snapshot_generation(
                root_descriptor,
                SessionGeneration(1, "owner"),
                ("cache/",),
                publication_generation=None,
            )
    finally:
        os.close(root_descriptor)


@pytest.mark.asyncio
async def test_s3_conditional_write_probe_rejects_unsafe_endpoint() -> None:
    client = _FakeS3Client()
    client.ignore_conditions = True
    store = _store(client)

    with pytest.raises(RuntimeError, match="does not enforce"):
        await store._verify_conditional_writes(client)

    assert not any("/.capability/" in key for key in client.objects)


@pytest.mark.asyncio
async def test_s3_capability_verification_and_cleanup_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _FakeS3Client()
    store = _store(client)
    store._conditional_writes_verified = False

    await store.verify_capabilities()

    assert store._conditional_writes_verified is True

    async def fail_delete(*, Bucket: str, Key: str) -> None:  # noqa: N803
        raise OSError("cleanup unavailable")

    monkeypatch.setattr(client, "delete_object", fail_delete)
    with caplog.at_level("WARNING"):
        await store._verify_conditional_writes(client)
    assert "Could not remove S3 conditional-write probe object" in caplog.text


@pytest.mark.asyncio
async def test_s3_store_bucket_errors_are_propagated() -> None:
    client = _FakeS3Client()
    client.head_bucket_error = ClientError({"Error": {"Code": "403"}}, "HeadBucket")
    store = _store(client)

    with pytest.raises(ClientError):
        async with store._client():
            pass


@pytest.mark.asyncio
async def test_s3_init_updates_list_cache_when_descriptor_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _store()
    store._sessions_cache[s3_store_mod._SESSIONS_CACHE_KEY] = []

    async def fail_descriptor(*_args, **_kwargs) -> None:
        raise OSError("descriptor unavailable")

    monkeypatch.setattr(store, "_write_generation_descriptor", fail_descriptor)
    with caplog.at_level("ERROR"):
        await store.init_session("cached-session")

    assert store._sessions_cache[s3_store_mod._SESSIONS_CACHE_KEY] == ["cached-session"]
    assert "physics_s3_generation_descriptor_failed" in caplog.text


@pytest.mark.asyncio
async def test_s3_store_crud_cache_and_events(tmp_path: Path) -> None:
    client = _FakeS3Client()
    store = _store(client, presign_by_default=True)

    await store.put_bytes("s1", "a.txt", b"hello", content_type="text/plain")
    assert await store.put_bytes_if_absent("s1", "claim", b"first") is True
    assert await store.put_bytes_if_absent("s1", "claim", b"second") is False
    assert (
        await store.put_bytes_if_absent(
            "s1",
            "typed-claim",
            b"typed",
            content_type="application/octet-stream",
        )
        is True
    )
    assert client.objects["wu/sessions/s1/claim"] == b"first"

    src = tmp_path / "data.bin"
    src.write_bytes(b"file")
    await store.put_file("s1", "nested/data.bin", str(src), content_type="app/test")
    assert client.objects["wu/sessions/s1/nested/data.bin"] == b"file"

    await store.put_json("s1", METADATA_KEY, {"id": "s1", "n": 1})
    assert await store.get_json("s1", METADATA_KEY) == {"id": "s1", "n": 1}
    assert await store.get_json("s1", "missing.json") is None

    stream = await store.open_read("s1", "a.txt")
    assert stream.read() == b"hello"
    assert await store.exists("s1", "a.txt") is True
    assert await store.exists("s1", "missing.txt") is False

    with pytest.raises(ClientError):
        client.raise_on_get = ClientError({"Error": {"Code": "AccessDenied"}}, "Get")
        await store.get_json("s1", METADATA_KEY)
    client.raise_on_get = None

    await store.append_event("s1", {"type": "a"})
    await store.append_event("s1", {"type": "b"})
    events = await store.get_event_log("s1")
    assert [event["type"] for event in events] == ["a", "b"]
    assert await store.get_event_log("empty") == []

    assert "a.txt" in await store.list_keys("s1")
    assert await store.make_public_url("s1", "a.txt", expires_seconds=7) == (
        "https://example.test/bucket/wu/sessions/s1/a.txt?exp=7"
    )

    no_presign = _store(_FakeS3Client(), presign_by_default=False)
    assert await no_presign.make_public_url("s1", "a.txt") is None


@pytest.mark.asyncio
async def test_s3_put_file_holds_source_across_leaf_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeS3Client()
    store = _store(client)
    source = tmp_path / "source.bin"
    moved_source = tmp_path / "source-held.bin"
    outside = tmp_path / "outside.bin"
    source.write_bytes(b"original")
    outside.write_bytes(b"outside")

    async def swap_then_upload(
        file_obj: BinaryIO,
        bucket: str,
        key: str,
        *,
        ExtraArgs: dict,  # noqa: N803 - boto-style fake signature
    ) -> None:
        assert bucket == client.bucket
        source.rename(moved_source)
        source.symlink_to(outside)
        client.objects[key] = file_obj.read()
        client.uploads.append((key, ExtraArgs))

    monkeypatch.setattr(client, "upload_fileobj", swap_then_upload)

    await store.put_file("s1", "upload.bin", str(source))

    assert client.objects[store._key("s1", "upload.bin")] == b"original"
    assert moved_source.read_bytes() == b"original"
    assert outside.read_bytes() == b"outside"


@pytest.mark.asyncio
async def test_s3_store_session_listing_cache_and_delete() -> None:
    client = _FakeS3Client()
    store = _store(client)
    client.objects.update(
        {
            "wu/sessions/s1/session.json": b"{}",
            "wu/sessions/s2/session.json": b"{}",
            "wu/other/file.txt": b"x",
        }
    )

    assert await store.list_sessions(use_cache=False) == ["s1", "s2"]
    client.objects["wu/sessions/s3/session.json"] = b"{}"
    assert await store.list_sessions() == ["s1", "s2"]

    with pytest.raises(SessionGenerationConflictError):
        await store.init_session("s3")
    assert await store.list_sessions() == ["s1", "s2"]

    store.invalidate_sessions_cache()
    assert await store.list_sessions() == ["s1", "s2", "s3"]

    await store.delete_session("s2")
    assert "wu/sessions/s2/session.json" in client.deleted
    assert await store.list_sessions() == ["s1", "s3"]


@pytest.mark.asyncio
async def test_s3_store_sync_from_and_to_local(tmp_path: Path) -> None:
    client = _FakeS3Client()
    store = _store(client)
    await store.init_session("s1")
    await store.put_json("s1", METADATA_KEY, {"status": "running"})

    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        await store.sync_from_local("s1", str(missing))

    empty = tmp_path / "empty"
    empty.mkdir()
    assert await store.sync_from_local("s1", str(empty)) == 0

    local = tmp_path / "local"
    (local / "input").mkdir(parents=True)
    (local / "input" / "config.yaml").write_text("project: {}\n", encoding="utf-8")
    (local / "cache").mkdir(parents=True)
    (local / "cache" / "a.txt").write_text("a", encoding="utf-8")
    (local / "cache" / "b.json").write_text("{}", encoding="utf-8")
    pipeline_temp = local / "cache" / "nested" / ".pipeline_temp"
    pipeline_temp.mkdir(parents=True)
    (pipeline_temp / "config.yaml").write_text(
        "api_key: sentinel",
        encoding="utf-8",
    )
    (local / "skip.txt").write_text("skip", encoding="utf-8")
    assert await store.sync_from_local("s1", str(local), prefix="input/") == 1
    assert await store.sync_from_local("s1", str(local), prefix="cache/") == 2
    assert await store.list_keys("s1") == [
        "cache/a.txt",
        "cache/b.json",
        "input/config.yaml",
    ]
    assert (await store.open_read("s1", "cache/a.txt")).read() == b"a"
    assert await store.exists("s1", "cache/b.json")
    assert not await store.exists("s1", "skip.txt")
    envelope = json.loads(client.objects[store._key("s1", METADATA_KEY)])
    manifest = json.loads(
        client.objects[
            store._key("s1", envelope["artifact_publication"]["manifest_key"])
        ]
    )
    assert manifest["snapshot_prefixes"] == ["input/", "cache/"]
    assert all(
        "/.generations/" in key
        for key in client.objects
        if not key.endswith("/session.json")
    )

    target = tmp_path / "target"
    assert await store.sync_to_local("s1", str(target), prefix="cache/") == 2
    assert (target / "cache" / "a.txt").read_bytes() == b"a"
    assert (target / "cache" / "b.json").read_text(encoding="utf-8") == "{}"
    assert not (target / "cache" / "nested" / ".pipeline_temp" / "remote.yaml").exists()
    (target / "cache" / "a.txt").write_text("stale", encoding="utf-8")
    (target / "cache" / "omitted.txt").write_text("obsolete", encoding="utf-8")
    assert await store.sync_to_local("s1", str(target), prefix="cache/") == 2
    assert (target / "cache" / "a.txt").read_bytes() == b"a"
    assert not (target / "cache" / "omitted.txt").exists()

    selected_target = tmp_path / "selected-target"
    assert (
        await store.sync_to_local(
            "s1",
            str(selected_target),
            prefix=("input/", "cache/a.txt"),
        )
        == 2
    )
    assert (selected_target / "input" / "config.yaml").exists()
    assert (selected_target / "cache" / "a.txt").exists()
    assert not (selected_target / "cache" / "b.json").exists()

    (selected_target / "unselected.txt").write_text("keep", encoding="utf-8")
    (selected_target / "cache" / "obsolete.txt").write_text(
        "remove",
        encoding="utf-8",
    )
    await store.sync_to_local(
        "s1",
        str(selected_target),
        prefix=("input/", "cache/a.txt"),
    )
    assert (selected_target / "unselected.txt").read_text(encoding="utf-8") == "keep"
    assert (selected_target / "cache" / "obsolete.txt").exists()

    direct_file = tmp_path / "direct.txt"
    direct_file.write_text("direct", encoding="utf-8")
    with pytest.raises(SessionGenerationOwnershipError, match="sync_from_local"):
        await store.put_bytes("s1", "cache/direct.txt", b"direct")
    with pytest.raises(SessionGenerationOwnershipError, match="sync_from_local"):
        await store.put_file("s1", "cache/direct.txt", str(direct_file))
    with pytest.raises(SessionGenerationOwnershipError, match="sync_from_local"):
        await store.delete_key("s1", "cache/a.txt")


@pytest.mark.asyncio
async def test_completed_snapshot_binds_status_to_one_publication(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    owner = _store(client)
    session_id = "completed-source"
    await owner.init_session(session_id)
    await owner.put_json(session_id, METADATA_KEY, {"status": "running"})
    source = tmp_path / "source"
    output = source / "cache" / "physics" / "scene_physics.usda"
    output.parent.mkdir(parents=True)
    output.write_text("completed-publication", encoding="utf-8")
    await owner.sync_from_local(
        session_id,
        str(source),
        prefix="cache/physics/",
    )
    await owner.put_json(session_id, METADATA_KEY, {"status": "completed"})

    target = tmp_path / "target"
    snapshot = await owner.sync_completed_publication_to_local(
        session_id,
        str(target),
        prefix="cache/physics/",
    )
    assert snapshot == CompletedSessionSnapshot(
        metadata={"status": "completed"},
        artifact_keys=("cache/physics/scene_physics.usda",),
        downloaded_count=1,
    )
    assert (target / "cache" / "physics" / "scene_physics.usda").read_text(
        encoding="utf-8"
    ) == "completed-publication"

    successor = _store(client)
    await successor.begin_generation(session_id)
    # begin_generation intentionally retains the previous completed metadata,
    # but the same envelope says the successor generation is starting. A
    # status-only read would accept it and then hydrate the wrong publication.
    with pytest.raises(SessionNotCompletedError):
        await owner.sync_completed_publication_to_local(
            session_id,
            str(tmp_path / "rejected"),
            prefix="cache/physics/",
        )


@pytest.mark.asyncio
async def test_completed_snapshot_handles_legacy_empty_and_missing_publications(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    legacy = _store(client)
    legacy_session_id = "legacy-completed-source"
    client.objects[legacy._key(legacy_session_id, METADATA_KEY)] = json.dumps(
        {"status": "completed"}
    ).encode()
    legacy_key = legacy._key(
        legacy_session_id,
        "cache/physics/scene_physics.usda",
    )
    client.objects[legacy_key] = b"legacy-completed"

    legacy_target = tmp_path / "legacy-target"
    legacy_snapshot = await legacy.sync_completed_publication_to_local(
        legacy_session_id,
        str(legacy_target),
        prefix="cache/physics/",
    )
    assert legacy_snapshot.downloaded_count == 1
    assert legacy_snapshot.artifact_keys == ("cache/physics/scene_physics.usda",)
    assert (
        legacy_target / "cache" / "physics" / "scene_physics.usda"
    ).read_bytes() == b"legacy-completed"

    empty_snapshot = await legacy.sync_completed_publication_to_local(
        legacy_session_id,
        str(tmp_path / "empty-target"),
        prefix=(),
    )
    assert empty_snapshot.downloaded_count == 0
    assert empty_snapshot.artifact_keys == ()

    generation = _store(client)
    generation_session_id = "completed-without-publication"
    await generation.init_session(generation_session_id)
    await generation.put_json(
        generation_session_id,
        METADATA_KEY,
        {"status": "completed"},
    )
    with pytest.raises(RuntimeError, match="no artifact publication"):
        await generation.sync_completed_publication_to_local(
            generation_session_id,
            str(tmp_path / "missing-publication"),
            prefix="cache/physics/",
        )


@pytest.mark.asyncio
async def test_hydration_prunes_only_manifest_declared_complete_prefixes(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    store = _store(client)
    session_id = "partial-dataset"
    session_dir = tmp_path / "session"
    dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    render = session_dir / "cache" / "dataset" / "usd" / "renders" / "view.png"
    render.parent.mkdir(parents=True)
    dataset.write_text('{"media":{"images":[{"path":"usd/renders/view.png"}]}}\n')
    render.write_bytes(b"local-only-render")
    assert detect_predict_mode(session_dir=session_dir, dataset_path=None)[0] == (
        "dataset_only"
    )

    await store.init_session(session_id)
    await store.put_json(session_id, METADATA_KEY, {"status": "running"})
    assert (
        await store.sync_from_local(
            session_id,
            str(session_dir),
            prefix="cache/dataset/dataset.jsonl",
        )
        == 1
    )
    envelope = json.loads(client.objects[store._key(session_id, METADATA_KEY)])
    manifest = json.loads(
        client.objects[
            store._key(
                session_id,
                envelope["artifact_publication"]["manifest_key"],
            )
        ]
    )
    assert manifest["snapshot_prefixes"] == ["cache/dataset/dataset.jsonl"]

    dataset.write_text("stale\n", encoding="utf-8")
    assert (
        await store.sync_to_local(
            session_id,
            str(session_dir),
            prefix="cache/dataset/",
        )
        == 1
    )
    assert dataset.read_text(encoding="utf-8").startswith('{"media"')
    assert render.read_bytes() == b"local-only-render"
    assert detect_predict_mode(session_dir=session_dir, dataset_path=None)[0] == (
        "dataset_only"
    )

    await store.put_json(session_id, METADATA_KEY, {"status": "completed"})
    successor = _store(client)
    generation = await successor.begin_generation(session_id)
    assert generation.generation == 2
    successor_dir = tmp_path / "successor"
    await successor.adopt_local_generation(
        session_id,
        str(successor_dir),
        generation,
    )
    await successor.put_json(session_id, METADATA_KEY, {"status": "running"})
    successor_dataset = successor_dir / "cache" / "dataset" / "dataset.jsonl"
    successor_render = (
        successor_dir / "cache" / "dataset" / "usd" / "renders" / "view.png"
    )
    successor_dataset.parent.mkdir(parents=True)
    successor_render.parent.mkdir(parents=True)
    successor_dataset.write_text(
        '{"media":{"images":[{"path":"usd/renders/view.png"}]}}\n',
        encoding="utf-8",
    )
    successor_render.write_bytes(b"generation-2-render")
    assert (
        await successor.sync_from_local(
            session_id,
            str(successor_dir),
            prefix="cache/dataset/dataset.jsonl",
        )
        == 1
    )
    assert (
        await successor.sync_to_local(
            session_id,
            str(successor_dir),
            prefix="cache/dataset/",
        )
        == 1
    )
    assert successor_render.read_bytes() == b"generation-2-render"

    # Reconciling an unrelated prefix first must not bless the whole local
    # directory as generation 2.
    assert (
        await store.sync_to_local(
            session_id,
            str(session_dir),
            prefix="input/",
        )
        == 0
    )
    assert render.read_bytes() == b"local-only-render"

    # The original replica's render belongs to generation 1. Hydrating the
    # generation-2 JSONL must not pair it with stale, same-named local images.
    assert (
        await store.sync_to_local(
            session_id,
            str(session_dir),
            prefix="cache/dataset/",
        )
        == 1
    )
    assert not render.exists()
    assert detect_predict_mode(session_dir=session_dir, dataset_path=None) == (
        "full_predict",
        None,
    )


@pytest.mark.asyncio
async def test_rerun_adopts_current_base_local_only_intermediates(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    store = _store(client)
    session_id = "same-replica-rerun"
    session_dir = tmp_path / "session"
    dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    render = session_dir / "cache" / "dataset" / "usd" / "renders" / "view.png"
    render.parent.mkdir(parents=True)
    dataset.write_text('{"media":{"images":[{"path":"usd/renders/view.png"}]}}\n')
    render.write_bytes(b"generation-1-render")

    await store.init_session(session_id)
    await store.put_json(session_id, METADATA_KEY, {"status": "running"})
    await store.sync_from_local(
        session_id,
        str(session_dir),
        prefix="cache/dataset/dataset.jsonl",
    )
    await store.put_json(session_id, METADATA_KEY, {"status": "completed"})

    generation = await store.begin_generation(session_id)
    await store.adopt_local_generation(
        session_id,
        str(session_dir),
        generation,
    )
    # The successor still reads generation 1's retained publication until it
    # commits its first snapshot. Publication provenance must therefore remain
    # distinct from the generation that owns local-only intermediates.
    assert (
        await store.sync_to_local(
            session_id,
            str(session_dir),
            prefix="cache/dataset/",
        )
        == 1
    )
    assert render.read_bytes() == b"generation-1-render"
    await store.put_json(session_id, METADATA_KEY, {"status": "running"})
    dataset.write_text(
        '{"media":{"images":[{"path":"usd/renders/view.png"}]}}\n',
        encoding="utf-8",
    )
    render.write_bytes(b"generation-2-render")
    await store.sync_from_local(
        session_id,
        str(session_dir),
        prefix="cache/dataset/dataset.jsonl",
    )

    assert (
        await store.sync_to_local(
            session_id,
            str(session_dir),
            prefix="cache/dataset/",
        )
        == 1
    )
    assert render.read_bytes() == b"generation-2-render"
    assert detect_predict_mode(session_dir=session_dir, dataset_path=None)[0] == (
        "dataset_only"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_marker_protocol", ["v3", "v2"])
async def test_retry_adopts_unpublished_predecessor_local_coverage(
    tmp_path: Path,
    failed_marker_protocol: str,
) -> None:
    client = _FakeS3Client()
    store = _store(client)
    session_id = "same-replica-startup-retry"
    session_dir = tmp_path / "session"
    dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    render = session_dir / "cache" / "dataset" / "usd" / "renders" / "view.png"
    render.parent.mkdir(parents=True)
    dataset.write_text("{}\n", encoding="utf-8")
    render.write_bytes(b"generation-1-render")

    await store.init_session(session_id)
    await store.put_json(session_id, METADATA_KEY, {"status": "running"})
    await store.sync_from_local(
        session_id,
        str(session_dir),
        prefix="cache/dataset/dataset.jsonl",
    )
    await store.put_json(session_id, METADATA_KEY, {"status": "completed"})

    failed_generation = await store.begin_generation(session_id)
    await store.adopt_local_generation(
        session_id,
        str(session_dir),
        failed_generation,
    )
    if failed_marker_protocol == "v2":
        marker = _read_local_marker(session_dir)
        assert marker is not None
        marker_path = session_dir / ".pipeline_temp" / "s3-snapshot-generation.json"
        marker_path.write_text(
            json.dumps(
                {
                    "protocol": "physics-local-s3-snapshot.v2",
                    "generation": failed_generation.generation,
                    "owner_id": failed_generation.owner_id,
                    "prefixes": list(marker.prefixes),
                }
            ),
            encoding="utf-8",
        )
    # Simulate route startup rollback: the generation becomes terminal without
    # advancing the immutable artifact publication.
    await store.put_json(session_id, METADATA_KEY, {"status": "completed"})

    retry_generation = await store.begin_generation(session_id)
    await store.adopt_local_generation(
        session_id,
        str(session_dir),
        retry_generation,
    )
    assert (
        await store.sync_to_local(
            session_id,
            str(session_dir),
            prefix="cache/dataset/",
        )
        == 1
    )
    assert render.read_bytes() == b"generation-1-render"


@pytest.mark.asyncio
async def test_adoption_rejects_unproven_nonempty_cache_and_renews_lease(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    owner = _store(client)
    session_id = "adoption-fail-closed"
    published_dir = tmp_path / "published"
    published_file = published_dir / "cache" / "dataset" / "dataset.jsonl"
    published_file.parent.mkdir(parents=True)
    published_file.write_text("{}\n", encoding="utf-8")

    assert await _store(client).owns_active_generation(session_id) is True
    await owner.init_session(session_id)
    assert await owner.owns_active_generation(session_id) is True
    await owner.put_json(session_id, METADATA_KEY, {"status": "running"})
    await owner.sync_from_local(
        session_id,
        str(published_dir),
        prefix="cache/dataset/dataset.jsonl",
    )
    await owner.put_json(session_id, METADATA_KEY, {"status": "completed"})

    rerun = _store(client)
    generation = await rerun.begin_generation(session_id)
    stale_dir = tmp_path / "stale"
    (stale_dir / "cache").mkdir(parents=True)
    (stale_dir / "cache" / "unknown.bin").write_bytes(b"unproven")
    await rerun.adopt_local_generation(session_id, str(stale_dir), generation)
    marker = _read_local_marker(stale_dir)
    assert marker is not None
    assert marker.prefixes == ()

    envelope_key = rerun._key(session_id, METADATA_KEY)
    envelope = json.loads(client.objects[envelope_key])
    envelope["generation_lease_expires_at"] = datetime.now(UTC).isoformat()
    client.objects[envelope_key] = json.dumps(envelope).encode()
    assert await rerun.owns_active_generation(session_id) is True
    renewed = json.loads(client.objects[envelope_key])
    assert (
        renewed["generation_lease_expires_at"]
        != envelope["generation_lease_expires_at"]
    )


@pytest.mark.asyncio
async def test_generation_cancellation_and_delete_edge_states() -> None:
    client = _FakeS3Client()
    external = _store(client)
    assert not await external.request_generation_cancellation(
        "missing",
        update_status=True,
    )
    assert not await external._set_generation_cancellation(
        "missing",
        requested=True,
    )
    assert not await external.delete_session_if_terminal("missing")

    legacy_id = "legacy-terminal"
    legacy_key = external._key(legacy_id, METADATA_KEY)
    client.objects[legacy_key] = json.dumps({"status": "completed"}).encode()
    client.objects[external._key(legacy_id, ".cancel")] = b""
    assert not await external.request_generation_cancellation(
        legacy_id,
        update_status=True,
    )
    assert external._key(legacy_id, ".cancel") not in client.objects

    active_legacy_id = "legacy-active"
    active_legacy_key = external._key(active_legacy_id, METADATA_KEY)
    client.objects[active_legacy_key] = json.dumps({"status": "running"}).encode()
    assert await external.request_generation_cancellation(
        active_legacy_id,
        update_status=False,
    )
    assert external._key(active_legacy_id, ".cancel") in client.objects

    owner = _store(client)
    session_id = "generation-terminal"
    await owner.init_session(session_id)
    await owner.put_json(session_id, METADATA_KEY, {"status": "completed"})
    assert not await _store(client).request_generation_cancellation(
        session_id,
        update_status=True,
    )

    envelope_key = owner._key(session_id, METADATA_KEY)
    envelope = json.loads(client.objects[envelope_key])
    envelope["generation"] = 0
    envelope["owner_id"] = ""
    client.objects[envelope_key] = json.dumps(envelope).encode()
    assert not await _store(client)._set_generation_cancellation(
        session_id,
        requested=True,
    )


@pytest.mark.asyncio
async def test_generation_control_resolution_and_json_update_paths() -> None:
    client = _FakeS3Client()
    owner = _store(client)
    session_id = "control-resolution"
    await owner.init_session(session_id)
    await owner.put_json(
        session_id,
        METADATA_KEY,
        {"status": "running", "value": 1},
    )
    assert (
        json.loads((await owner.open_read(session_id, METADATA_KEY)).read())["value"]
        == 1
    )
    assert await owner.exists(session_id, METADATA_KEY)
    assert (await owner.update_json(session_id, METADATA_KEY, lambda _current: None))[
        "value"
    ] == 1
    assert (
        await owner.update_json(
            session_id,
            METADATA_KEY,
            lambda current: {**current, "value": 2},
        )
    )["value"] == 2
    assert (
        await owner.update_json_if_not_cancelled(
            session_id,
            METADATA_KEY,
            lambda _current: None,
        )
    )["value"] == 2
    assert (
        await owner.update_json_if_not_cancelled(
            session_id,
            METADATA_KEY,
            lambda current: {**current, "value": 3},
        )
    )["value"] == 3
    await owner.request_generation_cancellation(session_id, update_status=False)
    assert (await owner.open_read(session_id, ".cancel")).read() == b""

    reader = _store(client)
    resolved = await reader._resolved_control_key(session_id, ".claim")
    assert resolved.startswith(".generations/1-")
    assert resolved.endswith("/control/.claim")

    legacy_id = "legacy-control"
    client.objects[reader._key(legacy_id, METADATA_KEY)] = json.dumps(
        {"status": "completed"}
    ).encode()
    assert await reader._resolved_control_key(legacy_id, ".claim") == ".claim"

    deleted_id = "deleted-control"
    deleted = owner._new_envelope(SessionGeneration(1, "owner-deleted"))
    deleted["deleted"] = True
    client.objects[reader._key(deleted_id, METADATA_KEY)] = json.dumps(deleted).encode()
    with pytest.raises(FileNotFoundError):
        await reader._resolved_control_key(deleted_id, ".claim")

    invalid_id = "invalid-control"
    invalid = owner._new_envelope(SessionGeneration(1, "owner-valid"))
    invalid["generation"] = "invalid"
    client.objects[reader._key(invalid_id, METADATA_KEY)] = json.dumps(invalid).encode()
    with pytest.raises(RuntimeError, match="Invalid current session generation"):
        await reader._resolved_control_key(invalid_id, ".claim")

    legacy_json_id = "legacy-json"
    client.objects[reader._key(legacy_json_id, "custom.json")] = b'{"value":1}'
    assert await reader.update_json(
        legacy_json_id,
        "custom.json",
        lambda current: {**current, "value": 2},
    ) == {"value": 2}
    assert await reader.update_json(
        legacy_json_id,
        "custom.json",
        lambda _current: None,
    ) == {"value": 2}
    assert (
        await reader.update_json("missing-json", "custom.json", lambda value: value)
        is None
    )
    assert (
        await reader.update_json("missing-json", METADATA_KEY, lambda value: value)
        is None
    )
    assert (
        await reader.update_json_if_not_cancelled(
            "missing-json",
            METADATA_KEY,
            lambda value: value,
        )
        is None
    )
    with pytest.raises(FileNotFoundError):
        await reader.open_read("missing-json", METADATA_KEY)
    publication_generation = SessionGeneration(1, "owner-publication")
    publication_prefix = reader._generation_prefix(publication_generation)
    manifest_key = f"{publication_prefix}/publications/test/manifest.json"
    publication_envelope = reader._new_envelope(publication_generation)
    publication_envelope["artifact_publication"] = {
        "generation": publication_generation.generation,
        "owner_id": publication_generation.owner_id,
        "generation_prefix": publication_prefix,
        "manifest_key": manifest_key,
    }
    client.objects[reader._key("missing-json", METADATA_KEY)] = json.dumps(
        publication_envelope
    ).encode()
    client.objects[reader._key("missing-json", manifest_key)] = json.dumps(
        {
            "protocol": s3_store_mod._MANIFEST_PROTOCOL,
            "complete": True,
            "generation": publication_generation.generation,
            "owner_id": publication_generation.owner_id,
            "artifacts": {},
            "snapshot_prefixes": [],
        }
    ).encode()
    assert await reader.make_public_url("missing-json", "missing.bin") is None
    with pytest.raises(
        SessionGenerationOwnershipError,
        match="No generation owner is bound",
    ):
        await reader.sync_from_local(
            session_id,
            "/unused",
            prefix=(),
        )
    assert (
        await reader.sync_to_local(
            session_id,
            "/unused",
            prefix=(),
        )
        == 0
    )


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        "/absolute.txt",
        ".",
        "..",
        "cache/./file.txt",
        "cache/../file.txt",
        "cache\\file.txt",
        "cache//file.txt",
        "cache/",
        "C:/drive.txt",
        ".pipeline_temp/config.yaml",
    ],
)
@pytest.mark.asyncio
async def test_s3_sync_rejects_unsafe_object_suffixes(
    tmp_path: Path,
    suffix: str,
) -> None:
    client = _FakeS3Client()
    store = _store(client)
    client.objects[f"{store._key('s1', '')}{suffix}"] = b"malicious"
    target = tmp_path / "target"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="S3 object key"):
        await store.sync_to_local("s1", str(target))

    assert client.downloads == []
    assert not target.exists()
    assert outside.read_bytes() == b"outside"


@pytest.mark.asyncio
async def test_generation_cancellation_terminal_delete_and_empty_publish_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeS3Client()
    store = _store(client)

    class ChangedEnvelope(dict):
        generation_reads = 0

        def get(self, key, default=None):
            if key == "generation":
                self.generation_reads += 1
                return 1 if self.generation_reads == 1 else 0
            return super().get(key, default)

    async def changed_envelope(*_args, **_kwargs):
        return ChangedEnvelope(deleted=False), "etag"

    with monkeypatch.context() as patch:
        patch.setattr(store, "_read_envelope", changed_envelope)
        assert not await store.request_generation_cancellation(
            "idle",
            update_status=False,
        )

    active = "active-edge"
    await store.init_session(active)
    generation = store._active_generation(active)
    assert generation is not None
    assert not await store.delete_session_if_terminal(active)
    assert await store.sync_from_local(active, str(tmp_path), prefix=()) == 0

    unbound = _store(client)
    assert await unbound._set_generation_cancellation(active, requested=False)

    reader = _store(client)
    calls = 0

    def changing_active_generation(_session_id: str):
        nonlocal calls
        calls += 1
        return generation if calls == 1 else None

    monkeypatch.setattr(reader, "_active_generation", changing_active_generation)
    replacement = store._new_envelope(
        SessionGeneration(generation.generation + 1, "replacement-owner")
    )
    client.objects[store._key(active, METADATA_KEY)] = json.dumps(replacement).encode()
    assert not await reader._set_generation_cancellation(active, requested=True)


@pytest.mark.asyncio
async def test_sync_to_local_skips_unavailable_atomic_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeS3Client()
    store = _store(client)
    session_id = "unavailable-writer"
    client.objects[store._key(session_id, "cache/a.txt")] = b"a"

    @contextmanager
    def unavailable_writer(*_args, **_kwargs):
        yield SimpleNamespace(stream=None, published=False)

    monkeypatch.setattr(
        s3_store_mod,
        "confined_atomic_writer",
        unavailable_writer,
    )
    assert (
        await store.sync_to_local(
            session_id,
            str(tmp_path / "destination"),
            prefix="cache/",
        )
        == 0
    )


@pytest.mark.asyncio
async def test_s3_store_rejects_symlink_alias_to_pipeline_temp(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    store = _store(client)
    await store.init_session("s1")
    local = tmp_path / "local"
    secret_path = local / ".pipeline_temp" / "config.yaml"
    secret_path.parent.mkdir(parents=True)
    secret_path.write_text("api_key: sentinel", encoding="utf-8")
    alias = local / "cache" / "export.yaml"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(secret_path)

    with pytest.raises(
        RuntimeError,
        match="symlinked session artifact",
    ):
        await store.sync_from_local("s1", str(local), prefix="cache/")

    assert "wu/sessions/s1/cache/export.yaml" not in client.objects


@pytest.mark.asyncio
async def test_s3_store_rejects_directory_symlink_alias_to_pipeline_temp(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    store = _store(client)
    await store.init_session("s1")
    local = tmp_path / "local"
    secret_dir = local / ".pipeline_temp" / "nested"
    secret_dir.mkdir(parents=True)
    (secret_dir / "config.yaml").write_text(
        "api_key: sentinel",
        encoding="utf-8",
    )
    alias = local / "cache" / "export"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(secret_dir, target_is_directory=True)

    with pytest.raises(
        RuntimeError,
        match="symlinked session artifact",
    ):
        await store.sync_from_local("s1", str(local), prefix="cache/")

    assert not any("/artifacts/" in key for key in client.objects)


@pytest.mark.asyncio
async def test_s3_store_cleanup_stale_local_sessions(tmp_path: Path) -> None:
    client = _FakeS3Client()
    store = _store(client)
    assert await store.cleanup_stale_local_sessions(str(tmp_path / "missing")) == 0

    fresh = tmp_path / "fresh"
    stale = tmp_path / "stale"
    stale_by_mtime = tmp_path / "stale-by-mtime"
    (tmp_path / "not-a-session.txt").write_text("x", encoding="utf-8")
    file_path = stale / "out.txt"
    file_path.parent.mkdir()
    file_path.write_text("x", encoding="utf-8")
    stale_by_mtime.mkdir()
    fresh.mkdir()

    now = datetime.now(UTC)
    await store.put_json(
        "fresh",
        METADATA_KEY,
        {"updated_at": (now - timedelta(minutes=5)).isoformat()},
    )
    await store.put_json(
        "stale",
        METADATA_KEY,
        {"updated_at": (now - timedelta(hours=4)).isoformat()},
    )
    old_ts = (now - timedelta(hours=5)).timestamp()
    os.utime(stale_by_mtime, (old_ts, old_ts))

    assert (
        await store.cleanup_stale_local_sessions(
            str(tmp_path), max_age_hours=1, skip_session_ids={"stale-by-mtime"}
        )
        == 1
    )
    assert fresh.exists()
    assert not stale.exists()
    assert stale_by_mtime.exists()

    async def fail_get_json(*_args, **_kwargs):
        raise RuntimeError("metadata failed")

    store.get_json = fail_get_json  # type: ignore[method-assign]
    assert await store._get_session_last_updated("broken") is None


@pytest.mark.asyncio
async def test_s3_store_cleanup_never_republishes_unowned_local_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(_FakeS3Client())
    stale = tmp_path / "stale"
    stale.mkdir()
    old_ts = (datetime.now(UTC) - timedelta(hours=5)).timestamp()
    os.utime(stale, (old_ts, old_ts))

    async def fail_sync(*_args, **_kwargs) -> int:
        raise AssertionError("cleanup must not publish an unowned local cache")

    monkeypatch.setattr(store, "sync_from_local", fail_sync)

    assert await store.cleanup_stale_local_sessions(str(tmp_path), max_age_hours=1) == 1
    assert not stale.exists()


@pytest.mark.asyncio
async def test_s3_store_cleanup_preserves_unpublished_failed_generation(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    store = _store(client)
    session_id = "failed-local-only"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    (session_dir / "diagnostic.txt").write_text("partial", encoding="utf-8")
    old_ts = (datetime.now(UTC) - timedelta(hours=5)).timestamp()
    os.utime(session_dir, (old_ts, old_ts))

    await store.init_session(session_id)
    await store.put_json(session_id, METADATA_KEY, {"status": "failed"})

    assert (
        await store.cleanup_stale_local_sessions(
            str(tmp_path),
            max_age_hours=1,
        )
        == 0
    )
    assert (session_dir / "diagnostic.txt").read_text(encoding="utf-8") == "partial"


@pytest.mark.asyncio
async def test_s3_gc_legacy_hydration_and_cleanup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _FakeS3Client()
    store = _store(client, generation_retention=1)
    session_id = "legacy-hydration"
    client.objects[store._key(session_id, "cache/a.txt")] = b"a"
    client.objects[store._key(session_id, ".generations/2-owner/artifact")] = b"x"
    destination = tmp_path / "hydrated"
    assert (
        await store.sync_to_local(
            session_id,
            str(destination),
            prefix=("cache/", "cache/a"),
        )
        == 1
    )
    assert (destination / "cache" / "a.txt").read_bytes() == b"a"

    gc_prefix = store._key(session_id, ".generations/")
    client.objects[f"{gc_prefix}bad-owner/artifact"] = b"bad"
    client.objects[f"{gc_prefix}1-owner/artifact"] = b"old"
    client.objects[f"{gc_prefix}2-owner/artifact"] = b"current"
    original_paginator = client.get_paginator

    class _MalformedPaginator(_Paginator):
        async def paginate(self, **kwargs):
            yield {"Contents": [{"Key": None}]}
            async for page in super().paginate(**kwargs):
                yield page

    monkeypatch.setattr(
        client,
        "get_paginator",
        lambda name: _MalformedPaginator(client),
    )
    await store.init_session(session_id)
    await store.put_json(session_id, METADATA_KEY, {"status": "completed"})
    await store.begin_generation(session_id)
    await store._garbage_collect_generations(
        client,
        session_id,
        current_generation=2,
        protected_generations=set(),
    )
    assert f"{gc_prefix}1-owner/artifact" in client.deleted
    monkeypatch.setattr(client, "get_paginator", original_paginator)

    async def fail_gc(*_args, **_kwargs) -> None:
        raise OSError("gc unavailable")

    monkeypatch.setattr(store, "_garbage_collect_generations", fail_gc)
    with caplog.at_level("ERROR"):
        await store._best_effort_generation_gc(
            client,
            session_id,
            current_generation=2,
            protected_generations=set(),
        )
    assert "physics_s3_generation_gc_failed" in caplog.text

    stale = tmp_path / "cleanup-failure"
    stale.mkdir()
    old_ts = (datetime.now(UTC) - timedelta(hours=5)).timestamp()
    os.utime(stale, (old_ts, old_ts))

    def fail_remove(*_args, **_kwargs) -> None:
        raise OSError("remove unavailable")

    monkeypatch.setattr(s3_store_mod, "remove_confined_tree", fail_remove)
    with caplog.at_level("ERROR"):
        assert (
            await store.cleanup_stale_local_sessions(
                str(tmp_path),
                max_age_hours=1,
                skip_session_ids={"hydrated"},
            )
            == 0
        )
    assert "stale_session_cleanup_failed" in caplog.text


@pytest.mark.asyncio
async def test_snapshot_marker_write_failures_remain_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _FakeS3Client()
    publisher = _store(client)
    session_id = "marker-write-failure"
    source = tmp_path / "source"
    artifact = source / "cache" / "a.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"a")
    await publisher.init_session(session_id)
    await publisher.put_json(session_id, METADATA_KEY, {"status": "running"})

    original_write = s3_store_mod._write_local_snapshot_generation

    def fail_marker(*_args, **_kwargs) -> None:
        raise OSError("marker unavailable")

    monkeypatch.setattr(s3_store_mod, "_write_local_snapshot_generation", fail_marker)
    with caplog.at_level("WARNING"):
        assert (
            await publisher.sync_from_local(
                session_id,
                str(source),
                prefix=("cache/", "cache/a"),
            )
            == 1
        )
    assert "Could not record local S3 snapshot generation" in caplog.text

    destination = tmp_path / "destination"
    with caplog.at_level("WARNING"):
        assert (
            await _store(client).sync_to_local(
                session_id,
                str(destination),
                prefix="cache/",
            )
            == 1
        )
    assert "Could not record hydrated S3 snapshot generation" in caplog.text
    monkeypatch.setattr(
        s3_store_mod,
        "_write_local_snapshot_generation",
        original_write,
    )


@pytest.mark.asyncio
async def test_cross_replica_rerun_fences_stale_completion_cancel_and_delete(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    first = _store(client)
    second = _store(client)
    external = _store(client)
    session_id = "race"

    first_dir = tmp_path / "first"
    (first_dir / "output").mkdir(parents=True)
    (first_dir / "output" / "result.json").write_text(
        '{"generation": 1}', encoding="utf-8"
    )
    await first.init_session(session_id)
    await first.put_json(session_id, METADATA_KEY, {"status": "running"})
    await first.sync_from_local(session_id, str(first_dir), prefix="output/")
    await first.put_json(session_id, METADATA_KEY, {"status": "completed"})
    await first.put_bytes(session_id, ".cancel", b"old")

    generation = await second.begin_generation(session_id)
    assert generation.generation == 2
    await second.put_json(session_id, METADATA_KEY, {"status": "running"})
    assert not await second.exists(session_id, ".cancel")
    assert not await first.owns_active_generation(session_id)

    # A cancellation request with no worker context targets exactly the current
    # generation; the prior generation's marker remains isolated for diagnosis.
    assert not await first.request_generation_cancellation(
        session_id,
        update_status=False,
    )
    assert await external.request_generation_cancellation(
        session_id,
        update_status=False,
    )
    assert await second.exists(session_id, ".cancel")
    await second.delete_key(session_id, ".cancel")

    second_dir = tmp_path / "second"
    (second_dir / "output").mkdir(parents=True)
    (second_dir / "output" / "result.json").write_text(
        '{"generation": 2}', encoding="utf-8"
    )
    with pytest.raises(SessionGenerationOwnershipError, match="Stale generation"):
        await first.sync_from_local(session_id, str(first_dir), prefix="output/")

    await second.sync_from_local(session_id, str(second_dir), prefix="output/")
    await second.put_json(session_id, METADATA_KEY, {"status": "completed"})
    assert (await external.open_read(session_id, "output/result.json")).read() == (
        b'{"generation": 2}'
    )

    with pytest.raises(SessionGenerationOwnershipError, match="Stale generation"):
        await first.delete_session(session_id)
    await external.delete_session(session_id)
    assert await external.get_json(session_id, METADATA_KEY) is None
    assert not any(f"/sessions/{session_id}/" in key for key in client.objects)


@pytest.mark.asyncio
async def test_expired_generation_lease_allows_fenced_takeover() -> None:
    client = _FakeS3Client()
    owner = _store(client)
    contender = _store(client)
    session_id = "expired-owner"

    await owner.init_session(session_id)
    await owner.put_json(session_id, METADATA_KEY, {"status": "running"})
    with pytest.raises(SessionGenerationConflictError):
        await contender.begin_generation(session_id)

    envelope_key = owner._key(session_id, METADATA_KEY)
    envelope = json.loads(client.objects[envelope_key])
    envelope["generation_lease_expires_at"] = (
        datetime.now(UTC) - timedelta(seconds=1)
    ).isoformat()
    client.objects[envelope_key] = json.dumps(envelope).encode("utf-8")

    generation = await contender.begin_generation(session_id)
    assert generation.generation == 2
    assert not await owner.owns_active_generation(session_id)

    replacement = json.loads(client.objects[envelope_key])
    assert replacement["generation_history"][-1]["state"] == "abandoned"


@pytest.mark.asyncio
async def test_active_legacy_session_blocks_takeover_and_keeps_cancellation_shape() -> (
    None
):
    client = _FakeS3Client()
    store = _store(client)
    session_id = "legacy-active"
    envelope_key = store._key(session_id, METADATA_KEY)
    client.objects[envelope_key] = json.dumps(
        {
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "running",
        }
    ).encode("utf-8")

    with pytest.raises(SessionGenerationConflictError, match="already active"):
        await store.begin_generation(session_id)

    assert await store.request_generation_cancellation(
        session_id,
        update_status=True,
    )
    legacy_document = json.loads(client.objects[envelope_key])
    assert legacy_document["status"] == "cancelling"
    assert "protocol" not in legacy_document
    assert client.objects[store._key(session_id, ".cancel")] == b""


@pytest.mark.asyncio
async def test_terminal_legacy_session_can_enter_generation_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeS3Client()
    store = _store(client)
    reader = _store(client)
    session_id = "legacy-terminal"
    envelope_key = store._key(session_id, METADATA_KEY)
    client.objects[envelope_key] = json.dumps(
        {
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "completed",
        }
    ).encode("utf-8")
    client.objects[store._key(session_id, "cache/physics/scene_physics.usda")] = (
        b"legacy-physics"
    )
    client.objects[store._key(session_id, "cache/predictions/old.jsonl")] = (
        b"legacy-prediction"
    )

    original_paginator = client.get_paginator

    class LegacyPaginator(_Paginator):
        async def paginate(self, **kwargs):
            yield {"Contents": [{"Key": None}]}
            async for page in super().paginate(**kwargs):
                yield page

    monkeypatch.setattr(
        client,
        "get_paginator",
        lambda name: LegacyPaginator(client),
    )

    generation = await store.begin_generation(session_id)
    monkeypatch.setattr(client, "get_paginator", original_paginator)
    assert generation.generation == 1
    upgraded_document = json.loads(client.objects[envelope_key])
    assert upgraded_document["protocol"] == "physics-session-generation.v1"
    assert upgraded_document["generation"] == 1
    assert upgraded_document["artifact_publication"]["generation"] == 1
    assert (
        await reader.open_read(session_id, "cache/physics/scene_physics.usda")
    ).read() == b"legacy-physics"

    rerun_dir = tmp_path / "legacy-rerun"
    (rerun_dir / "cache" / "predictions").mkdir(parents=True)
    (rerun_dir / "cache" / "predictions" / "new.jsonl").write_bytes(b"new-prediction")
    await store.sync_from_local(
        session_id,
        str(rerun_dir),
        prefix="cache/predictions/",
    )
    assert (
        await reader.open_read(session_id, "cache/physics/scene_physics.usda")
    ).read() == b"legacy-physics"
    with pytest.raises(FileNotFoundError):
        await reader.open_read(session_id, "cache/predictions/old.jsonl")
    assert (
        await reader.open_read(session_id, "cache/predictions/new.jsonl")
    ).read() == b"new-prediction"


@pytest.mark.asyncio
async def test_stale_generation_gc_cannot_delete_current_base_publication(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    first = _store(client, generation_retention=1)
    session_id = "stale-gc"
    first_dir = tmp_path / "first"
    (first_dir / "output").mkdir(parents=True)
    (first_dir / "output" / "result.txt").write_text("one", encoding="utf-8")
    await first.init_session(session_id)
    await first.sync_from_local(session_id, str(first_dir), prefix="output/")
    await first.put_json(session_id, METADATA_KEY, {"status": "completed"})

    second = _store(client, generation_retention=1)
    await second.begin_generation(session_id)
    second_dir = tmp_path / "second"
    (second_dir / "output").mkdir(parents=True)
    (second_dir / "output" / "result.txt").write_text("two", encoding="utf-8")
    await second.sync_from_local(session_id, str(second_dir), prefix="output/")
    await second.put_json(session_id, METADATA_KEY, {"status": "completed"})

    third = _store(client, generation_retention=1)
    await third.begin_generation(session_id)
    generation_two_keys = {
        key for key in client.objects if f"sessions/{session_id}/.generations/2-" in key
    }
    assert generation_two_keys

    # A generation-1 GC that resumes after generation 3 claimed the session
    # must not delete generation 2, which the current envelope still exposes.
    await first._garbage_collect_generations(
        client,
        session_id,
        current_generation=1,
        protected_generations={1},
    )
    assert generation_two_keys <= client.objects.keys()


@pytest.mark.asyncio
async def test_reader_keeps_complete_old_manifest_until_new_generation_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeS3Client()
    writer = _store(client, generation_retention=1)
    reader = _store(client)
    session_id = "visibility"
    old_dir = tmp_path / "old"
    (old_dir / "output").mkdir(parents=True)
    (old_dir / "output" / "a.txt").write_text("old-a", encoding="utf-8")
    (old_dir / "output" / "b.txt").write_text("old-b", encoding="utf-8")
    await writer.init_session(session_id)
    await writer.put_json(session_id, METADATA_KEY, {"status": "running"})
    await writer.sync_from_local(session_id, str(old_dir), prefix="output/")
    await writer.put_json(session_id, METADATA_KEY, {"status": "completed"})

    next_writer = _store(client, generation_retention=1)
    await next_writer.begin_generation(session_id)
    await next_writer.put_json(session_id, METADATA_KEY, {"status": "running"})
    new_dir = tmp_path / "new"
    (new_dir / "output").mkdir(parents=True)
    (new_dir / "output" / "a.txt").write_text("new-a", encoding="utf-8")
    (new_dir / "output" / "b.txt").write_text("new-b", encoding="utf-8")

    first_upload_done = asyncio.Event()
    release_upload = asyncio.Event()
    original_upload = client.upload_fileobj
    upload_count = 0

    async def blocked_upload(*args, **kwargs) -> None:
        nonlocal upload_count
        await original_upload(*args, **kwargs)
        upload_count += 1
        if upload_count == 1:
            first_upload_done.set()
            await release_upload.wait()

    monkeypatch.setattr(client, "upload_fileobj", blocked_upload)
    publishing = asyncio.create_task(
        next_writer.sync_from_local(session_id, str(new_dir), prefix="output/")
    )
    await first_upload_done.wait()

    assert await reader.list_keys(session_id, prefix="output/") == [
        "output/a.txt",
        "output/b.txt",
    ]
    assert (await reader.open_read(session_id, "output/a.txt")).read() == b"old-a"
    assert (await reader.open_read(session_id, "output/b.txt")).read() == b"old-b"

    release_upload.set()
    assert await publishing == 2
    assert (await reader.open_read(session_id, "output/a.txt")).read() == b"new-a"

    await next_writer.put_json(session_id, METADATA_KEY, {"status": "completed"})
    empty_writer = _store(client, generation_retention=1)
    await empty_writer.begin_generation(session_id)
    await empty_writer.put_json(session_id, METADATA_KEY, {"status": "running"})
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert (
        await empty_writer.sync_from_local(
            session_id,
            str(empty_dir),
            prefix="output/",
        )
        == 0
    )
    assert await reader.list_keys(session_id, prefix="output/") == []
    with pytest.raises(FileNotFoundError):
        await reader.open_read(session_id, "output/a.txt")

    await empty_writer.put_json(session_id, METADATA_KEY, {"status": "completed"})
    fourth_writer = _store(client, generation_retention=1)
    await fourth_writer.begin_generation(session_id)
    # The live empty manifest is hosted by generation 3. Starting generation 4
    # must protect that manifest without retaining generation 2's stale output.
    assert any("/.generations/3-" in key for key in client.objects)
    assert not any("/.generations/2-" in key for key in client.objects)
    assert await reader.list_keys(session_id, prefix="output/") == []


@pytest.mark.asyncio
async def test_partial_rerun_copies_unselected_artifacts_into_new_generation(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    first = _store(client, generation_retention=1)
    reader = _store(client)
    session_id = "partial-rerun"
    first_dir = tmp_path / "first"
    (first_dir / "input").mkdir(parents=True)
    (first_dir / "input" / "config.yaml").write_text("run: 1\n", encoding="utf-8")
    (first_dir / "cache" / "physics").mkdir(parents=True)
    (first_dir / "cache" / "physics" / "scene_physics.usda").write_text(
        "old-physics",
        encoding="utf-8",
    )
    (first_dir / "cache" / "predictions").mkdir(parents=True)
    (first_dir / "cache" / "predictions" / "obsolete.jsonl").write_text(
        "old-prediction",
        encoding="utf-8",
    )
    await first.init_session(session_id)
    await first.put_json(session_id, METADATA_KEY, {"status": "running"})
    await first.sync_from_local(
        session_id,
        str(first_dir),
        prefix=("input/", "cache/physics/", "cache/predictions/"),
    )
    await first.put_json(session_id, METADATA_KEY, {"status": "completed"})

    second = _store(client, generation_retention=1)
    generation = await second.begin_generation(session_id)
    assert generation.generation == 2
    await second.put_json(session_id, METADATA_KEY, {"status": "running"})
    second_dir = tmp_path / "second"
    (second_dir / "input").mkdir(parents=True)
    (second_dir / "input" / "config.yaml").write_text("run: 2\n", encoding="utf-8")
    (second_dir / "cache" / "predictions").mkdir(parents=True)
    (second_dir / "cache" / "predictions" / "predictions.jsonl").write_text(
        "new-prediction",
        encoding="utf-8",
    )
    assert (
        await second.sync_from_local(
            session_id,
            str(second_dir),
            prefix=("input/", "cache/predictions/"),
        )
        == 2
    )

    assert await reader.list_keys(session_id, prefix="cache/physics/") == [
        "cache/physics/scene_physics.usda"
    ]
    physics_stream = await reader.open_read(
        session_id,
        "cache/physics/scene_physics.usda",
    )
    assert physics_stream.read() == b"old-physics"
    assert await reader.list_keys(session_id, prefix="cache/predictions/") == [
        "cache/predictions/predictions.jsonl"
    ]
    envelope = json.loads(client.objects[second._key(session_id, METADATA_KEY)])
    manifest = json.loads(
        client.objects[
            second._key(
                session_id,
                envelope["artifact_publication"]["manifest_key"],
            )
        ]
    )
    assert manifest["artifacts"]["cache/physics/scene_physics.usda"].startswith(
        ".generations/2-"
    )

    await second.put_json(session_id, METADATA_KEY, {"status": "completed"})
    third = _store(client, generation_retention=1)
    await third.begin_generation(session_id)
    assert not any("/.generations/1-" in key for key in client.objects)
    assert (
        await reader.open_read(session_id, "cache/physics/scene_physics.usda")
    ).read() == b"old-physics"


@pytest.mark.asyncio
async def test_terminal_cleanup_cannot_delete_a_concurrently_started_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeS3Client()
    original_put = client.put_object
    cleanup_cas_started = asyncio.Event()
    release_cleanup_cas = asyncio.Event()

    owner = _store(client)
    session_id = "cleanup-race"
    await owner.init_session(session_id)
    await owner.put_json(session_id, METADATA_KEY, {"status": "completed"})

    async def block_cleanup_tombstone(**kwargs) -> None:
        body = json.loads(kwargs["Body"])
        if kwargs["Key"].endswith(f"/{METADATA_KEY}") and body.get("deleted") is True:
            cleanup_cas_started.set()
            await release_cleanup_cas.wait()
        await original_put(**kwargs)

    monkeypatch.setattr(client, "put_object", block_cleanup_tombstone)
    cleaner = _store(client)
    cleanup = asyncio.create_task(cleaner.delete_session_if_terminal(session_id))
    await cleanup_cas_started.wait()

    rerun = _store(client)
    generation = await rerun.begin_generation(session_id)
    assert generation.generation == 2
    await rerun.put_json(session_id, METADATA_KEY, {"status": "running"})

    release_cleanup_cas.set()
    assert await cleanup is False
    assert await cleaner.get_json(session_id, METADATA_KEY) == {"status": "running"}
    assert any(f"/sessions/{session_id}/" in key for key in client.objects)


@pytest.mark.asyncio
async def test_explicit_delete_cannot_retry_into_a_newer_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeS3Client()
    original_put = client.put_object
    delete_cas_started = asyncio.Event()
    release_delete_cas = asyncio.Event()
    owner = _store(client)
    session_id = "delete-race"
    await owner.init_session(session_id)
    await owner.put_json(session_id, METADATA_KEY, {"status": "completed"})

    async def block_delete_tombstone(**kwargs) -> None:
        body = json.loads(kwargs["Body"])
        if kwargs["Key"].endswith(f"/{METADATA_KEY}") and body.get("deleted"):
            delete_cas_started.set()
            await release_delete_cas.wait()
        await original_put(**kwargs)

    monkeypatch.setattr(client, "put_object", block_delete_tombstone)
    cleaner = _store(client)
    deleting = asyncio.create_task(cleaner.delete_session(session_id))
    await delete_cas_started.wait()

    rerun = _store(client)
    await rerun.begin_generation(session_id)
    await rerun.put_json(session_id, METADATA_KEY, {"status": "running"})
    release_delete_cas.set()

    with pytest.raises(SessionGenerationConflictError, match="changed during deletion"):
        await deleting
    assert await rerun.get_json(session_id, METADATA_KEY) == {"status": "running"}


@pytest.mark.asyncio
async def test_delayed_cancellation_cannot_cross_into_a_new_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeS3Client()
    original_put = client.put_object
    cancellation_cas_started = asyncio.Event()
    release_cancellation_cas = asyncio.Event()

    owner = _store(client)
    session_id = "cancel-race"
    await owner.init_session(session_id)
    await owner.put_json(session_id, METADATA_KEY, {"status": "running"})

    async def block_cancellation_cas(**kwargs) -> None:
        body = json.loads(kwargs["Body"])
        if (
            kwargs["Key"].endswith(f"/{METADATA_KEY}")
            and body.get("cancellation_requested") is True
        ):
            cancellation_cas_started.set()
            await release_cancellation_cas.wait()
        await original_put(**kwargs)

    monkeypatch.setattr(client, "put_object", block_cancellation_cas)
    external = _store(client)
    cancellation = asyncio.create_task(
        external.request_generation_cancellation(
            session_id,
            update_status=True,
        )
    )
    await cancellation_cas_started.wait()

    await owner.put_json(session_id, METADATA_KEY, {"status": "completed"})
    rerun = _store(client)
    await rerun.begin_generation(session_id)
    await rerun.put_json(session_id, METADATA_KEY, {"status": "running"})

    release_cancellation_cas.set()
    assert await cancellation is False
    assert not await rerun.exists(session_id, ".cancel")
    assert await external.get_json(session_id, METADATA_KEY) == {"status": "running"}


@pytest.mark.asyncio
async def test_cancellation_atomically_beats_terminal_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeS3Client()
    original_put = client.put_object
    terminal_cas_started = asyncio.Event()
    release_terminal_cas = asyncio.Event()
    owner = _store(client)
    external = _store(client)
    session_id = "terminal-cancel-race"

    await owner.init_session(session_id)
    await owner.put_json(session_id, METADATA_KEY, {"status": "running"})

    async def block_terminal_cas(**kwargs) -> None:
        body = json.loads(kwargs["Body"])
        if (
            kwargs["Key"].endswith(f"/{METADATA_KEY}")
            and body.get("metadata", {}).get("status") == "completed"
        ):
            terminal_cas_started.set()
            await release_terminal_cas.wait()
        await original_put(**kwargs)

    monkeypatch.setattr(client, "put_object", block_terminal_cas)
    terminal = asyncio.create_task(
        owner.update_json_if_not_cancelled(
            session_id,
            METADATA_KEY,
            lambda metadata: {**metadata, "status": "completed"},
        )
    )
    await terminal_cas_started.wait()
    assert await external.request_generation_cancellation(
        session_id,
        update_status=True,
    )
    release_terminal_cas.set()

    assert await terminal is None
    metadata = await external.get_json(session_id, METADATA_KEY)
    assert metadata is not None
    assert metadata["status"] == "cancelling"
    assert metadata["updated_at"]


@pytest.mark.asyncio
async def test_concurrent_rerun_claim_has_one_owner_and_retention_bounds_incomplete_runs(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    initial = _store(client, generation_retention=2)
    session_id = "retention"
    output = tmp_path / "output"
    output.mkdir()
    (output / "result.txt").write_text("one", encoding="utf-8")
    await initial.init_session(session_id)
    await initial.put_json(session_id, METADATA_KEY, {"status": "running"})
    await initial.sync_from_local(session_id, str(tmp_path), prefix="output/")
    await initial.put_json(session_id, METADATA_KEY, {"status": "completed"})

    contenders = [
        _store(client, generation_retention=2),
        _store(client, generation_retention=2),
    ]

    async def claim(store: S3SessionStore):
        try:
            return await store.begin_generation(session_id)
        except SessionGenerationConflictError:
            return None

    claims = await asyncio.gather(*(claim(store) for store in contenders))
    assert sum(claim is not None for claim in claims) == 1
    winner_index = 0 if claims[0] is not None else 1
    winner = contenders[winner_index]
    winning_claim = claims[winner_index]
    assert winning_claim is not None
    winner._activate_generation(session_id, winning_claim)
    await winner.put_json(session_id, METADATA_KEY, {"status": "failed"})

    # Generation 2 has a durable descriptor but no manifest: it is retained as
    # an incomplete/failed diagnostic generation until the bounded GC window
    # advances past it.
    third = _store(client, generation_retention=2)
    await third.begin_generation(session_id)
    await third.put_json(session_id, METADATA_KEY, {"status": "running"})
    (output / "result.txt").write_text("three", encoding="utf-8")
    await third.sync_from_local(session_id, str(tmp_path), prefix="output/")
    await third.put_json(session_id, METADATA_KEY, {"status": "completed"})
    assert any("/.generations/2-" in key for key in client.objects)
    assert not any("/.generations/1-" in key for key in client.objects)

    fourth = _store(client, generation_retention=2)
    await fourth.begin_generation(session_id)
    await fourth.put_json(session_id, METADATA_KEY, {"status": "running"})
    (output / "result.txt").write_text("four", encoding="utf-8")
    await fourth.sync_from_local(session_id, str(tmp_path), prefix="output/")
    assert not any("/.generations/2-" in key for key in client.objects)
