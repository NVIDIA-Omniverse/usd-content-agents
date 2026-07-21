# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from ...service.storage import METADATA_KEY, S3SessionStore
from ...service.storage import s3_store as s3_store_module
from ...service.storage.s3_store import _StreamingBodyReader


class _Body:
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


def _client_error(code: str, operation: str = "S3") -> ClientError:
    return ClientError({"Error": {"Code": code}}, operation)


class _FakePaginator:
    def __init__(self, client: _FakeS3Client) -> None:
        self._client = client

    def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
        prefix = kwargs["Prefix"]
        keys = [key for key in sorted(self._client.objects) if key.startswith(prefix)]
        if kwargs.get("Delimiter") == "/":
            common_prefixes = set()
            for key in keys:
                remainder = key[len(prefix) :]
                if "/" not in remainder:
                    continue
                common_prefixes.add(f"{prefix}{remainder.split('/', 1)[0]}/")
            return [
                {
                    "CommonPrefixes": [
                        {"Prefix": item} for item in sorted(common_prefixes)
                    ]
                }
            ]
        return [{"Contents": [{"Key": key} for key in keys]}]


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.etags: dict[str, str] = {}
        self.put_calls: list[dict[str, Any]] = []
        self.uploads: list[tuple[str, str, dict[str, Any]]] = []
        self.downloads: list[tuple[str, str]] = []
        self.deleted: list[str] = []

    def _set_object(self, key: str, data: bytes) -> None:
        self.objects[key] = data
        self.etags[key] = f'"etag-{len(self.etags) + 1}"'

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        if key not in self.objects:
            raise _client_error("NoSuchKey", "GetObject")
        response: dict[str, Any] = {"Body": _Body(self.objects[key])}
        if key in self.etags:
            response["ETag"] = self.etags[key]
        return response

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
            raise _client_error("PreconditionFailed", "PutObject")
        if "IfMatch" in kwargs and kwargs["IfMatch"] != self.etags.get(key):
            raise _client_error("PreconditionFailed", "PutObject")
        body = kwargs.get("Body", b"")
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._set_object(key, body)
        self.put_calls.append(dict(kwargs))
        return {}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs["Key"] not in self.objects:
            raise _client_error("NoSuchKey", "HeadObject")
        return {}

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        if "IfMatch" in kwargs and kwargs["IfMatch"] != self.etags.get(key):
            raise _client_error("PreconditionFailed", "DeleteObject")
        self.objects.pop(key, None)
        self.etags.pop(key, None)
        self.deleted.append(key)
        return {}

    def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
        for item in kwargs["Delete"]["Objects"]:
            key = item["Key"]
            self.objects.pop(key, None)
            self.etags.pop(key, None)
            self.deleted.append(key)
        return {}

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        **kwargs: Any,
    ) -> None:
        self._set_object(key, Path(filename).read_bytes())
        self.uploads.append((filename, key, kwargs.get("ExtraArgs", {})))

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        if key not in self.objects:
            raise _client_error("NoSuchKey", "DownloadFile")
        Path(filename).write_bytes(self.objects[key])
        self.downloads.append((key, filename))

    def generate_presigned_url(self, operation: str, **kwargs: Any) -> str:
        return f"https://s3.example.test/{kwargs['Params']['Key']}?expires={kwargs['ExpiresIn']}"

    def get_paginator(self, name: str) -> _FakePaginator:
        assert name == "list_objects_v2"
        return _FakePaginator(self)


def _store(client: _FakeS3Client, **kwargs: Any) -> S3SessionStore:
    store = S3SessionStore(bucket="bucket", prefix="root", **kwargs)
    store._client = client
    return store


def test_s3_store_object_operations_and_cache_paths(tmp_path: Path) -> None:
    client = _FakeS3Client()
    store = _store(client, sessions_cache_ttl=60)
    session_id = "session-a"

    assert store.kind == "s3"
    client._set_object("root/sessions-index.json", b'{"sessions": []}')
    index, etag = store._read_session_index_with_etag()
    assert index == {}
    assert etag == '"etag-1"'

    assert store._write_session_index({"existing": {"session_id": "existing"}}, etag)
    assert store._prefix_key(session_id, "cache/") == "root/sessions/session-a/cache/"
    assert store._prefix_matches("cache/output.usdz", "cache")

    store._sessions_cache[s3_store_module._SESSIONS_CACHE_KEY] = (
        s3_store_module.time.monotonic(),
        ["old"],
    )
    store.init_session(session_id)
    assert store.list_sessions() == ["old", session_id]
    assert store.list_sessions() == ["old", session_id]

    store.put_bytes(session_id, "cache/data.bin", b"data", "application/octet-stream")
    local_file = tmp_path / "local.txt"
    local_file.write_text("hello", encoding="utf-8")
    store.put_file(session_id, "cache/local.txt", str(local_file), "text/plain")

    assert store.exists(session_id, "cache/data.bin") is True
    assert store.exists(session_id, "cache/missing.bin") is False
    assert store.list_keys(session_id, "cache") == [
        "cache/data.bin",
        "cache/local.txt",
    ]
    assert client.uploads[0][2] == {"ContentType": "text/plain"}
    store.delete_key(session_id, "cache/local.txt")
    assert not store.exists(session_id, "cache/local.txt")

    assert store.put_json_if_absent(session_id, "metadata.json", {"status": "new"})
    assert not store.put_json_if_absent(session_id, "metadata.json", {"status": "old"})
    assert store.get_json(session_id, "metadata.json") == {"status": "new"}
    assert store.get_json(session_id, "missing.json") is None
    store.put_json(session_id, "extra.json", {"answer": 42})
    assert store.get_json(session_id, "extra.json") == {"answer": 42}

    assert store.update_json(session_id, "missing.json", lambda obj: obj) is None
    current = store.update_json(session_id, "metadata.json", lambda obj: None)
    assert current == {"status": "new"}
    updated = store.update_json(
        session_id,
        "metadata.json",
        lambda obj: {**obj, "status": "updated"},
    )
    assert updated == {"status": "updated"}

    assert (
        store.delete_json_if_match(
            session_id,
            "metadata.json",
            lambda obj: obj.get("status") == "different",
        )
        is False
    )
    client.etags.pop(store._key(session_id, "metadata.json"))
    assert (
        store.delete_json_if_match(
            session_id,
            "metadata.json",
            lambda obj: obj.get("status") == "updated",
        )
        is False
    )

    assert store.make_public_url(session_id, "cache/data.bin", expires_seconds=5) == (
        "https://s3.example.test/root/sessions/session-a/cache/data.bin?expires=5"
    )
    no_presign = _store(client, presign_by_default=False)
    assert no_presign.make_public_url(session_id, "cache/data.bin") is None

    client._set_object(store._key("legacy-prefix", METADATA_KEY), b"{}")
    assert "legacy-prefix" in store.list_sessions(use_cache=False)
    metadata_rows = store.list_session_metadata()
    assert store.list_session_metadata()[0] == metadata_rows[0]


def test_s3_store_deletes_metadata_after_artifacts_and_updates_index() -> None:
    client = _FakeS3Client()
    store = _store(client)
    session_id = "delete-me"
    client._set_object(
        "root/sessions-index.json",
        json.dumps({"sessions": {session_id: {"session_id": session_id}}}).encode(),
    )
    client._set_object(
        store._key(session_id, METADATA_KEY), b'{"session_id": "delete-me"}'
    )
    client._set_object(store._key(session_id, "cache/output.usdz"), b"usdz")

    store.delete_session(session_id)

    assert store._key(session_id, "cache/output.usdz") in client.deleted
    assert store._key(session_id, METADATA_KEY) in client.deleted
    assert session_id not in store._read_session_index()


def test_s3_store_event_log_legacy_fallback_and_local_sync(tmp_path: Path) -> None:
    client = _FakeS3Client()
    store = _store(client, max_pool_connections=2)
    session_id = "sync-session"
    client._set_object(store._key(session_id, "events/one.json"), b'{"event": 1}')
    client._set_object(
        store._key(session_id, "events/two.jsonl"),
        b'{"event": 2}\n{"event": 1}\n',
    )

    assert store.get_event_log(session_id) == [{"event": 1}, {"event": 2}]

    source = tmp_path / "source"
    (source / "cache").mkdir(parents=True)
    (source / "cache" / "texture.png").write_bytes(b"png")
    (source / "skip.txt").write_text("skip", encoding="utf-8")
    (source / "subdir").mkdir()
    assert store.sync_from_local("other", str(tmp_path / "missing")) == 0
    assert store.sync_from_local("other", str(source), prefix="missing/") == 0
    assert store.sync_from_local("other", str(source), prefix="subdir/") == 0
    assert store.sync_from_local("other", str(source)) == 2
    assert store.sync_from_local("other", str(source), prefix="cache/texture.png") == 1
    assert client.uploads[-1][1] == store._key("other", "cache/texture.png")

    target = tmp_path / "target"
    client._set_object(store._key("other", "../escape.txt"), b"escape")
    assert store.sync_to_local("other", str(target), prefix="cache/") == 1
    assert (target / "cache" / "texture.png").read_bytes() == b"png"


def test_s3_store_profile_session_success_and_stream_read_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    client = _FakeS3Client()

    class FakeSession:
        def __init__(self, profile_name: str) -> None:
            calls["profile_name"] = profile_name

        def client(self, service_name: str, **kwargs: Any) -> _FakeS3Client:
            calls["service_name"] = service_name
            calls["kwargs"] = kwargs
            return client

    monkeypatch.setattr(s3_store_module.boto3.session, "Session", FakeSession)
    store = S3SessionStore(bucket="bucket", profile="profile-a", region="us-west-2")

    assert store._get_client() is client
    assert calls["profile_name"] == "profile-a"
    assert calls["service_name"] == "s3"

    client._set_object("sessions/s/blob.bin", b"abcdef")
    stream = store.open_read("s", "blob.bin")
    try:
        assert stream.readable() is True
        assert stream.read(2) == b"ab"
        assert stream.read() == b"cdef"
    finally:
        stream.close()

    body = _Body(b"z")
    reader = _StreamingBodyReader(body)
    reader.close()
    assert body.closed is True


def test_s3_store_index_retry_and_remove_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeS3Client()
    store = _store(client)
    sleeps: list[int] = []
    monkeypatch.setattr(
        store, "_index_retry_sleep", lambda attempt: sleeps.append(attempt)
    )

    write_results = iter([False, True])

    def flaky_write(index: dict[str, Any], etag: str | None) -> bool:
        return next(write_results)

    monkeypatch.setattr(store, "_write_session_index", flaky_write)
    store.update_session_index("retry-session", {"status": "pending"})

    assert sleeps == [0]

    client._set_object(
        "root/sessions-index.json",
        json.dumps({"sessions": {"missing": {"session_id": "missing"}}}).encode(),
    )
    store._remove_session_from_index("absent")

    sleeps.clear()
    remove_results = iter([False, True])
    monkeypatch.setattr(
        store,
        "_write_session_index",
        lambda index, etag: next(remove_results),
    )
    store._remove_session_from_index("missing")

    assert sleeps == [0]
