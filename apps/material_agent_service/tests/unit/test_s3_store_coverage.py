# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the S3 session store using an in-memory async client."""
# ruff: noqa: N801, N803, N818

from __future__ import annotations

import hashlib
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO

import pytest
from botocore.exceptions import ClientError

from ...service.storage.base import JsonPreconditionError
from ...service.storage.config import StorageConfig
from ...service.storage.s3_store import S3SessionStore


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code}}, "operation")


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0
        self.read_sizes: list[int | None] = []
        self.closed = False

    async def read(self, amount: int | None = None) -> bytes:
        self.read_sizes.append(amount)
        if amount is None:
            chunk = self._data[self._offset :]
            self._offset = len(self._data)
            return chunk
        chunk = self._data[self._offset : self._offset + amount]
        self._offset += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


class _Paginator:
    def __init__(self, client: _FakeS3Client) -> None:
        self.client = client

    async def paginate(self, *, Bucket: str, Prefix: str, Delimiter: str | None = None):
        assert Bucket == self.client.bucket
        if Delimiter == "/":
            common = set()
            for key in self.client.objects:
                if not key.startswith(Prefix):
                    continue
                rest = key[len(Prefix) :]
                session = rest.split("/", 1)[0]
                if session:
                    common.add(f"{Prefix}{session}/")
            yield {"CommonPrefixes": [{"Prefix": value} for value in sorted(common)]}
            return

        yield {
            "Contents": [
                {"Key": key}
                for key in sorted(self.client.objects)
                if key.startswith(Prefix)
            ]
        }


class _FakeS3Client:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self, bucket: str = "bucket") -> None:
        self.bucket = bucket
        self.objects: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}
        self.deleted: list[str] = []
        self.uploads: list[tuple[str, str, dict[str, Any]]] = []
        self.downloads: list[tuple[str, str]] = []
        self.head_bucket_error: ClientError | None = None
        self.created_buckets: list[str] = []
        self.head_object_errors: dict[str, BaseException] = {}
        self.get_object_errors: dict[str, BaseException] = {}
        self.put_object_error: ClientError | None = None
        self.omit_get_etag: set[str] = set()
        self.omit_put_etag = False
        self.last_body: _Body | None = None

    async def head_bucket(self, *, Bucket: str) -> None:
        assert Bucket == self.bucket
        if self.head_bucket_error is not None:
            raise self.head_bucket_error

    async def create_bucket(self, *, Bucket: str) -> None:
        self.created_buckets.append(Bucket)

    def get_paginator(self, name: str) -> _Paginator:
        assert name == "list_objects_v2"
        return _Paginator(self)

    async def delete_object(self, *, Bucket: str, Key: str) -> None:
        assert Bucket == self.bucket
        self.deleted.append(Key)
        self.objects.pop(Key, None)

    async def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str | None = None,
        IfMatch: str | None = None,
        IfNoneMatch: str | None = None,
    ) -> dict[str, str]:
        assert Bucket == self.bucket
        if self.put_object_error is not None:
            raise self.put_object_error
        current_etag = self._etag(self.objects[Key]) if Key in self.objects else None
        if IfNoneMatch == "*" and current_etag is not None:
            raise _client_error("PreconditionFailed")
        if IfMatch is not None and current_etag != IfMatch:
            raise _client_error("PreconditionFailed")
        self.objects[Key] = Body
        if ContentType:
            self.content_types[Key] = ContentType
        return {} if self.omit_put_etag else {"ETag": self._etag(Body)}

    @staticmethod
    def _etag(data: bytes) -> str:
        return f'"{hashlib.sha256(data).hexdigest()}"'

    async def upload_file(
        self,
        file_path: str,
        bucket: str,
        key: str,
        *,
        ExtraArgs: dict[str, Any],
    ) -> None:
        assert bucket == self.bucket
        self.objects[key] = Path(file_path).read_bytes()
        self.uploads.append((file_path, key, ExtraArgs))

    async def upload_fileobj(
        self,
        file_obj: BinaryIO,
        bucket: str,
        key: str,
        *,
        ExtraArgs: dict[str, Any],
    ) -> None:
        assert bucket == self.bucket
        self.objects[key] = file_obj.read()
        self.uploads.append((f"fd:{file_obj.fileno()}", key, ExtraArgs))

    async def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        assert Bucket == self.bucket
        if Key in self.get_object_errors:
            raise self.get_object_errors[Key]
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey(Key)
        self.last_body = _Body(self.objects[Key])
        response: dict[str, Any] = {"Body": self.last_body}
        if Key not in self.omit_get_etag:
            response["ETag"] = self._etag(self.objects[Key])
        return response

    async def head_object(self, *, Bucket: str, Key: str) -> dict[str, str]:
        assert Bucket == self.bucket
        if Key in self.head_object_errors:
            raise self.head_object_errors[Key]
        if Key not in self.objects:
            raise _client_error("404")
        return {"ETag": self._etag(self.objects[Key])}

    async def generate_presigned_url(
        self,
        operation: str,
        *,
        Params: dict[str, str],
        ExpiresIn: int,
    ) -> str:
        return f"{operation}:{Params['Bucket']}:{Params['Key']}:{ExpiresIn}"

    async def download_file(self, bucket: str, key: str, dest_path: str) -> None:
        assert bucket == self.bucket
        Path(dest_path).write_bytes(self.objects[key])
        self.downloads.append((key, dest_path))

    async def download_fileobj(
        self,
        bucket: str,
        key: str,
        file_obj: BinaryIO,
    ) -> None:
        assert bucket == self.bucket
        file_obj.write(self.objects[key])
        self.downloads.append((key, f"fd:{file_obj.fileno()}"))


def _store_with_client(
    client: _FakeS3Client,
    *,
    prefix: str = "tenant",
    presign_by_default: bool = True,
) -> S3SessionStore:
    store = S3SessionStore(
        bucket=client.bucket,
        prefix=prefix,
        create_bucket_if_missing=False,
        presign_by_default=presign_by_default,
    )

    @asynccontextmanager
    async def fake_client() -> AsyncIterator[_FakeS3Client]:
        yield client

    store._client = fake_client  # type: ignore[method-assign]
    return store


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reserved_s3_keys_are_invisible_and_writes_fail_loudly() -> None:
    client = _FakeS3Client()
    store = _store_with_client(client)
    client.objects[store._key("s1", "output/result.json")] = b"{}"
    client.objects[store._key("s1", "cache/.pipeline_temp/config.json")] = b"{}"
    client.objects[store._key("s1", r"cache\.pipeline_temp\windows.json")] = b"{}"

    assert await store.list_keys("s1") == ["output/result.json"]
    assert await store.list_keys("s1", r"cache\.pipeline_temp") == []
    assert not await store.exists("s1", "cache/.pipeline_temp/config.json")
    assert await store.get_json("s1", "cache/.pipeline_temp/config.json") is None
    assert await store.get_json_batch(
        ["s1", "s2"], "cache/.pipeline_temp/config.json"
    ) == [None, None]
    assert await store.make_public_url("s1", "cache/.pipeline_temp/config.json") is None
    with pytest.raises(FileNotFoundError):
        await store.open_read("s1", r"cache\.pipeline_temp\windows.json")
    with pytest.raises(ValueError, match="reserved"):
        await store.put_bytes("s1", "cache/.pipeline_temp/new.json", b"{}")


@pytest.mark.unit
def test_s3_store_constructor_and_from_config_validation() -> None:
    with pytest.raises(ValueError, match="bucket is required"):
        S3SessionStore(bucket="")

    with pytest.raises(ValueError, match="s3_bucket is required"):
        S3SessionStore.from_config(StorageConfig(kind="s3", s3_bucket=None))

    store = S3SessionStore.from_config(
        StorageConfig(
            kind="s3",
            s3_bucket="bucket",
            s3_prefix="prefix",
            s3_region="us-west-2",
            s3_endpoint_url="http://minio:9000",
            s3_access_key_id="access",
            s3_secret_access_key="secret",
            s3_session_token="token",
            s3_use_path_style=False,
            s3_create_bucket=False,
            s3_presign=False,
            s3_sessions_cache_ttl=5,
        )
    )
    assert store.kind == "s3"
    assert store._key("sid", "session.json") == "prefix/sessions/sid/session.json"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_s3_client_context_ensures_bucket_once() -> None:
    client = _FakeS3Client()
    client.head_bucket_error = _client_error("NoSuchBucket")
    store = S3SessionStore(bucket="bucket", create_bucket_if_missing=True)

    class _Session:
        def client(self, *_args: Any, **_kwargs: Any):
            @asynccontextmanager
            async def context():
                yield client

            return context()

    store._session = _Session()

    async with store._client() as yielded:
        assert yielded is client
    async with store._client() as yielded:
        assert yielded is client

    assert client.created_buckets == ["bucket"]

    client.head_bucket_error = _client_error("403")
    failing_store = S3SessionStore(bucket="bucket", create_bucket_if_missing=True)
    failing_store._session = _Session()
    failing_store._bucket_ensured = False
    with pytest.raises(ClientError):
        async with failing_store._client():
            pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_s3_session_listing_cache_and_delete() -> None:
    client = _FakeS3Client()
    store = _store_with_client(client)
    client.objects.update(
        {
            store._key("alpha", "session.json"): b"{}",
            store._key("beta", "session.json"): b"{}",
        }
    )

    assert await store.list_sessions() == ["alpha", "beta"]
    client.objects[store._key("gamma", "session.json")] = b"{}"
    assert await store.list_sessions() == ["alpha", "beta"]
    assert await store.list_sessions(use_cache=False) == ["alpha", "beta", "gamma"]

    await store.init_session("delta")
    assert await store.list_sessions() == ["alpha", "beta", "gamma", "delta"]

    await store.delete_session("beta")
    assert store._key("beta", "session.json") in client.deleted
    assert await store.list_sessions() == ["alpha", "gamma", "delta"]
    store.invalidate_sessions_cache()
    assert await store.list_sessions() == ["alpha", "gamma"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_s3_object_json_event_and_url_operations(tmp_path: Path) -> None:
    client = _FakeS3Client()
    store = _store_with_client(client)

    await store.put_bytes("sid", "raw.bin", b"raw", content_type="application/bin")
    assert client.objects[store._key("sid", "raw.bin")] == b"raw"
    assert client.content_types[store._key("sid", "raw.bin")] == "application/bin"

    source = tmp_path / "upload.txt"
    source.write_text("hello")
    await store.put_file("sid", "upload.txt", str(source), content_type="text/plain")
    assert client.objects[store._key("sid", "upload.txt")] == b"hello"
    assert client.uploads[-1][2] == {"ContentType": "text/plain"}

    reader = await store.open_read("sid", "upload.txt")
    assert reader.read() == b"hello"
    chunks = [
        chunk
        async for chunk in store.iter_read(
            "sid",
            "upload.txt",
            chunk_size=2,
        )
    ]
    assert chunks == [b"he", b"ll", b"o"]
    assert client.last_body is not None
    assert client.last_body.read_sizes == [2, 2, 2, 2]
    assert client.last_body.closed is True
    assert await store.exists("sid", "upload.txt") is True
    assert await store.exists("sid", "missing.txt") is False

    client.head_object_errors[store._key("sid", "forbidden.txt")] = _client_error("403")
    with pytest.raises(ClientError):
        await store.exists("sid", "forbidden.txt")

    assert await store.list_keys("sid") == ["raw.bin", "upload.txt"]

    await store.put_json("sid", "session.json", {"path": tmp_path / "asset.usd"})
    assert await store.get_json("sid", "session.json") == {
        "path": str(tmp_path / "asset.usd")
    }
    assert await store.get_json("sid", "missing.json") is None

    client.get_object_errors[store._key("bad", "session.json")] = RuntimeError("boom")
    assert await store.get_json_batch(["sid", "missing", "bad"], "session.json") == [
        {"path": str(tmp_path / "asset.usd")},
        None,
        None,
    ]

    await store.append_event("sid", {"event": "second"})
    await store.append_event("sid", {"event": "first"})
    assert [item["event"] for item in await store.get_event_log("sid")] == [
        "second",
        "first",
    ]
    assert await store.get_event_log("empty") == []

    assert await store.make_public_url("sid", "upload.txt", expires_seconds=7) == (
        f"get_object:bucket:{store._key('sid', 'upload.txt')}:7"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_s3_put_file_holds_source_across_leaf_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeS3Client()
    store = _store_with_client(client)
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
        ExtraArgs: dict[str, Any],  # noqa: N803 - boto-style fake signature
    ) -> None:
        assert bucket == client.bucket
        source.rename(moved_source)
        source.symlink_to(outside)
        client.objects[key] = file_obj.read()
        client.uploads.append((f"fd:{file_obj.fileno()}", key, ExtraArgs))

    monkeypatch.setattr(client, "upload_fileobj", swap_then_upload)

    await store.put_file("sid", "upload.bin", str(source))

    assert client.objects[store._key("sid", "upload.bin")] == b"original"
    assert moved_source.read_bytes() == b"original"
    assert outside.read_bytes() == b"outside"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_s3_batch_failure_uses_value_free_phase_telemetry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _FakeS3Client()
    store = _store_with_client(client)
    sentinel_key = "access_token=batch-key-sentinel-727"
    sentinel_error = "batch-backend-error-sentinel-727"
    client.get_object_errors[store._key("sid", sentinel_key)] = RuntimeError(
        sentinel_error
    )

    assert await store.get_json_batch(["sid"], sentinel_key) == [None]
    assert "code=material_s3_batch_read_failed" in caplog.text
    assert "phase=persistence_verification" in caplog.text
    assert sentinel_key not in caplog.text
    assert sentinel_error not in caplog.text
    await store.delete_file("sid", "raw.bin")
    assert store._key("sid", "raw.bin") in client.deleted
    assert not await store.exists("sid", "raw.bin")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_s3_json_compare_and_swap_uses_etag_preconditions() -> None:
    client = _FakeS3Client()
    store = _store_with_client(client)

    missing = await store.get_json_versioned("sid", "session.json")
    assert missing.value is None
    assert missing.version is None

    first_version = await store.replace_json_if_version(
        "sid", "session.json", {"winner": 0}, None
    )
    first = await store.get_json_versioned("sid", "session.json")
    assert first.value == {"winner": 0}
    assert first.version == first_version

    with pytest.raises(JsonPreconditionError):
        await store.replace_json_if_version(
            "sid", "session.json", {"winner": "stale-create"}, None
        )

    second_version = await store.replace_json_if_version(
        "sid", "session.json", {"winner": 1}, first_version
    )
    assert second_version != first_version

    with pytest.raises(JsonPreconditionError):
        await store.replace_json_if_version(
            "sid", "session.json", {"winner": "stale-update"}, first_version
        )
    assert await store.get_json("sid", "session.json") == {"winner": 1}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_s3_versioned_json_error_contracts() -> None:
    client = _FakeS3Client()
    store = _store_with_client(client)
    key = store._key("sid", "session.json")

    client.get_object_errors[key] = _client_error("404")
    assert (await store.get_json_versioned("sid", "session.json")).value is None
    client.get_object_errors[key] = _client_error("403")
    with pytest.raises(ClientError):
        await store.get_json_versioned("sid", "session.json")

    client.get_object_errors.pop(key)
    client.objects[key] = b"{}"
    client.omit_get_etag.add(key)
    with pytest.raises(RuntimeError, match="did not return an ETag"):
        await store.get_json_versioned("sid", "session.json")
    client.omit_get_etag.clear()

    current = await store.get_json_versioned("sid", "session.json")
    assert current.version is not None
    client.put_object_error = _client_error("ConditionalRequestConflict")
    with pytest.raises(JsonPreconditionError):
        await store.replace_json_if_version(
            "sid", "session.json", {"value": 1}, current.version
        )
    client.put_object_error = _client_error("NoSuchKey")
    with pytest.raises(JsonPreconditionError):
        await store.replace_json_if_version(
            "sid", "session.json", {"value": 1}, current.version
        )

    client.objects.pop(key)
    client.put_object_error = _client_error("404")
    with pytest.raises(JsonPreconditionError):
        await store.replace_json_if_version(
            "sid", "session.json", {"value": 1}, current.version
        )

    client.objects[key] = b'{"recreated": true}'
    with pytest.raises(JsonPreconditionError):
        await store.replace_json_if_version(
            "sid", "session.json", {"value": 1}, current.version
        )
    client.objects.pop(key)

    client.head_object_errors[key] = client.exceptions.NoSuchKey(key)
    with pytest.raises(JsonPreconditionError):
        await store.replace_json_if_version(
            "sid", "session.json", {"value": 1}, current.version
        )

    client.head_object_errors[key] = _client_error("NoSuchBucket")
    with pytest.raises(ClientError) as verify_error:
        await store.replace_json_if_version(
            "sid", "session.json", {"value": 1}, current.version
        )
    assert verify_error.value.response["Error"]["Code"] == "NoSuchBucket"

    client.head_object_errors.clear()
    client.objects[key] = b"{}"
    client.put_object_error = _client_error("404")
    with pytest.raises(ClientError):
        await store.replace_json_if_version(
            "sid", "session.json", {"value": 1}, current.version
        )

    client.put_object_error = _client_error("NoSuchKey")
    with pytest.raises(ClientError):
        await store.replace_json_if_version("sid", "session.json", {"value": 1}, None)

    client.put_object_error = _client_error("403")
    with pytest.raises(ClientError):
        await store.replace_json_if_version(
            "sid", "session.json", {"value": 1}, current.version
        )

    client.put_object_error = None
    client.omit_put_etag = True
    with pytest.raises(RuntimeError, match="did not return an ETag"):
        await store.replace_json_if_version(
            "sid", "session.json", {"value": 1}, current.version
        )
    no_presign = _store_with_client(client, presign_by_default=False)
    assert await no_presign.make_public_url("sid", "upload.txt") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_s3_sync_from_local_and_to_local(tmp_path: Path) -> None:
    client = _FakeS3Client()
    store = _store_with_client(client)

    missing = tmp_path / "missing"
    assert await store.sync_from_local("sid", str(missing)) == 0

    local_dir = tmp_path / "local"
    (local_dir / "input").mkdir(parents=True)
    (local_dir / "input" / "a.txt").write_text("a")
    pipeline_temp = local_dir / "input" / "nested" / ".pipeline_temp"
    pipeline_temp.mkdir(parents=True)
    (pipeline_temp / "config.yaml").write_text("api_key: sentinel")
    (local_dir / "skip.bin").write_bytes(b"skip")
    client.objects[store._key("sid", "input/existing.txt")] = b"existing"
    (local_dir / "input" / "existing.txt").write_text("existing")

    assert await store.sync_from_local("sid", str(local_dir), prefix="input/") == 1
    assert client.objects[store._key("sid", "input/a.txt")] == b"a"
    assert store._key("sid", "input/nested/.pipeline_temp/config.yaml") not in (
        client.objects
    )
    assert client.uploads[-1][2] == {"ContentType": "text/plain"}

    client.head_object_errors[store._key("sid", "input/error.txt")] = _client_error(
        "500"
    )
    (local_dir / "input" / "error.txt").write_text("error")
    with pytest.raises(ClientError):
        await store.sync_from_local("sid", str(local_dir), prefix="input/error")

    download_dir = tmp_path / "download"
    (download_dir / "input").mkdir(parents=True)
    (download_dir / "input" / "existing.txt").write_text("local")
    assert await store.sync_to_local("sid", str(download_dir), prefix="input/") == 1
    assert (download_dir / "input" / "a.txt").read_text() == "a"
    assert not (
        download_dir / "input" / "nested" / ".pipeline_temp" / "remote.yaml"
    ).exists()


@pytest.mark.unit
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
    store = _store_with_client(client)
    client.objects[f"{store._key('sid', '')}{suffix}"] = b"malicious"
    target = tmp_path / "target"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="S3 object key"):
        await store.sync_to_local("sid", str(target))

    assert client.downloads == []
    assert not target.exists()
    assert outside.read_bytes() == b"outside"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_s3_sync_rejects_symlink_alias_to_pipeline_temp(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    store = _store_with_client(client)
    local_dir = tmp_path / "local"
    secret_path = local_dir / ".pipeline_temp" / "config.yaml"
    secret_path.parent.mkdir(parents=True)
    secret_path.write_text("api_key: sentinel", encoding="utf-8")
    alias = local_dir / "input" / "export.yaml"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(secret_path)

    with pytest.raises(
        RuntimeError,
        match="symlinked session artifact",
    ):
        await store.sync_from_local("sid", str(local_dir), prefix="input/")

    assert store._key("sid", "input/export.yaml") not in client.objects


@pytest.mark.unit
@pytest.mark.asyncio
async def test_s3_cleanup_stale_local_sessions(tmp_path: Path) -> None:
    client = _FakeS3Client()
    store = _store_with_client(client)

    missing_root = tmp_path / "missing-root"
    assert await store.cleanup_stale_local_sessions(str(missing_root)) == 0

    local_root = tmp_path / "sessions"
    stale = local_root / "stale"
    fresh = local_root / "fresh"
    no_metadata = local_root / "no-metadata"
    for path in (stale, fresh, no_metadata):
        path.mkdir(parents=True)
        (path / "file.txt").write_text(path.name)
    (local_root / "not-a-dir").write_text("skip")

    old = datetime.now(UTC) - timedelta(hours=48)
    recent = datetime.now(UTC)
    os.utime(no_metadata, (old.timestamp(), old.timestamp()))

    async def fake_last_updated(session_id: str):
        if session_id == "stale":
            return old
        if session_id == "fresh":
            return recent
        return None

    synced: list[str] = []

    async def fake_sync(session_id: str, _local_dir: str, prefix: str = "") -> int:
        synced.append(session_id)
        return 3

    store._get_session_last_updated = fake_last_updated  # type: ignore[method-assign]
    store.sync_from_local = fake_sync  # type: ignore[method-assign]
    assert (
        await store.cleanup_stale_local_sessions(str(local_root), max_age_hours=24) == 2
    )
    assert sorted(synced) == ["no-metadata", "stale"]
    assert not stale.exists()
    assert fresh.exists()
    assert not no_metadata.exists()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_s3_get_session_last_updated_paths() -> None:
    client = _FakeS3Client()
    store = _store_with_client(client)
    await store.put_json("sid", "session.json", {"updated_at": "2026-01-02T03:04:05Z"})
    assert await store._get_session_last_updated("sid") == datetime(
        2026,
        1,
        2,
        3,
        4,
        5,
        tzinfo=UTC,
    )
    assert await store._get_session_last_updated("missing") is None

    client.objects[store._key("bad", "session.json")] = b'{"updated_at": "not-date"}'
    assert await store._get_session_last_updated("bad") is None
