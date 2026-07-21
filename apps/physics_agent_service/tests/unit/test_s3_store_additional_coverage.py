# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO

import pytest
from botocore.exceptions import ClientError

from ...service.storage.base import METADATA_KEY
from ...service.storage.config import StorageConfig
from ...service.storage.s3_store import S3SessionStore


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

    async def put_object(self, **kwargs) -> None:
        if kwargs.get("IfNoneMatch") == "*" and kwargs["Key"] in self.objects:
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed"}},
                "PutObject",
            )
        self.objects[kwargs["Key"]] = kwargs["Body"]
        self.uploads.append((kwargs["Key"], kwargs))

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

    async def get_object(self, *, Bucket: str, Key: str) -> dict:  # noqa: N803
        assert Bucket == self.bucket
        if self.raise_on_get is not None:
            raise self.raise_on_get
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": _Body(self.objects[Key])}

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
    return store


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
    )
    from_cfg = S3SessionStore.from_config(cfg)
    assert from_cfg.bucket == "bucket"
    assert from_cfg.prefix == "prefix"
    assert from_cfg.presign_by_default is False


@pytest.mark.asyncio
async def test_s3_store_bucket_errors_are_propagated() -> None:
    client = _FakeS3Client()
    client.head_bucket_error = ClientError({"Error": {"Code": "403"}}, "HeadBucket")
    store = _store(client)

    with pytest.raises(ClientError):
        async with store._client():
            pass


@pytest.mark.asyncio
async def test_s3_store_crud_cache_and_events(tmp_path: Path) -> None:
    client = _FakeS3Client()
    store = _store(client, presign_by_default=True)

    await store.put_bytes("s1", "a.txt", b"hello", content_type="text/plain")
    assert await store.put_bytes_if_absent("s1", "claim", b"first") is True
    assert await store.put_bytes_if_absent("s1", "claim", b"second") is False
    assert client.objects["wu/sessions/s1/claim"] == b"first"
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

    src = tmp_path / "data.bin"
    src.write_bytes(b"file")
    await store.put_file("s1", "nested/data.bin", str(src), content_type="app/test")
    assert client.objects["wu/sessions/s1/nested/data.bin"] == b"file"

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

    await store.init_session("s3")
    assert await store.list_sessions() == ["s1", "s2", "s3"]

    store.invalidate_sessions_cache()
    assert await store.list_sessions() == ["s1", "s2", "s3"]

    await store.delete_session("s2")
    assert "wu/sessions/s2/session.json" in client.deleted
    assert await store.list_sessions() == ["s1", "s3"]


@pytest.mark.asyncio
async def test_s3_store_sync_from_and_to_local(tmp_path: Path) -> None:
    client = _FakeS3Client()
    store = _store(client)

    missing = tmp_path / "missing"
    assert await store.sync_from_local("s1", str(missing)) == 0

    empty = tmp_path / "empty"
    empty.mkdir()
    assert await store.sync_from_local("s1", str(empty)) == 0

    local = tmp_path / "local"
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
    client.objects["wu/sessions/s1/cache/a.txt"] = b"old"

    assert await store.sync_from_local("s1", str(local), prefix="cache/") == 1
    assert "wu/sessions/s1/cache/b.json" in client.objects
    assert (
        "wu/sessions/s1/cache/nested/.pipeline_temp/config.yaml" not in client.objects
    )
    assert "wu/sessions/s1/skip.txt" not in client.objects

    target = tmp_path / "target"
    assert await store.sync_to_local("s1", str(target), prefix="cache/") == 2
    assert (target / "cache" / "a.txt").read_bytes() == b"old"
    assert (target / "cache" / "b.json").read_text(encoding="utf-8") == "{}"
    assert not (target / "cache" / "nested" / ".pipeline_temp" / "remote.yaml").exists()
    assert await store.sync_to_local("s1", str(target), prefix="cache/") == 0

    class LoosePaginator:
        async def paginate(self, **_kwargs):
            yield {"Contents": [{"Key": "wu/sessions/s1/other/file.txt"}]}

    client.get_paginator = lambda _name: LoosePaginator()  # type: ignore[method-assign]
    assert (
        await store.sync_to_local("s1", str(tmp_path / "loose"), prefix="cache/") == 0
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
async def test_s3_store_rejects_symlink_alias_to_pipeline_temp(
    tmp_path: Path,
) -> None:
    client = _FakeS3Client()
    store = _store(client)
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

    assert not client.objects


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
async def test_s3_store_cleanup_logs_and_continues_on_sync_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(_FakeS3Client())
    stale = tmp_path / "stale"
    stale.mkdir()
    old_ts = (datetime.now(UTC) - timedelta(hours=5)).timestamp()
    os.utime(stale, (old_ts, old_ts))

    async def fail_sync(*_args, **_kwargs) -> int:
        raise RuntimeError("sync failed")

    monkeypatch.setattr(store, "sync_from_local", fail_sync)

    assert await store.cleanup_stale_local_sessions(str(tmp_path), max_age_hours=1) == 0
    assert stale.exists()
