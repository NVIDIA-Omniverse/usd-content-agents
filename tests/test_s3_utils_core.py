# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for S3 utility orchestration with fake boto clients."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from world_understanding.utils import s3_utils


class FakePaginator:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return self.pages


class FakeS3Client:
    def __init__(self, pages: list[dict[str, Any]] | None = None) -> None:
        self.paginator = FakePaginator(pages or [])
        self.uploads: list[dict[str, Any]] = []
        self.downloads: list[dict[str, Any]] = []
        self.deleted_objects: list[dict[str, Any]] = []
        self.deleted_batches: list[dict[str, Any]] = []
        self.heads: list[dict[str, Any]] = []
        self.presigned: list[dict[str, Any]] = []

    def get_paginator(self, name: str) -> FakePaginator:
        assert name == "list_objects_v2"
        return self.paginator

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict[str, Any],
        Callback: Any,
    ) -> None:
        self.uploads.append(
            {
                "filename": filename,
                "bucket": bucket,
                "key": key,
                "extra_args": ExtraArgs,
                "callback": Callback,
            }
        )

    def head_object(self, **kwargs: Any) -> None:
        self.heads.append(kwargs)

    def download_file(
        self, bucket: str, key: str, filename: str, Callback: Any = None
    ) -> None:
        Path(filename).write_text(f"{bucket}/{key}", encoding="utf-8")
        self.downloads.append(
            {"bucket": bucket, "key": key, "filename": filename, "callback": Callback}
        )

    def delete_object(self, **kwargs: Any) -> None:
        self.deleted_objects.append(kwargs)

    def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
        self.deleted_batches.append(kwargs)
        objects = kwargs["Delete"]["Objects"]
        response: dict[str, Any] = {"Deleted": objects[:1]}
        if len(objects) > 1:
            response["Errors"] = [
                {"Key": objects[1]["Key"], "Code": "Denied", "Message": "no"}
            ]
        return response

    def generate_presigned_url(self, operation: str, **kwargs: Any) -> str:
        self.presigned.append({"operation": operation, **kwargs})
        return "https://signed.example/object"


def test_list_upload_download_exists_and_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeS3Client(
        [
            {},
            {
                "Contents": [
                    {"Key": "prefix/"},
                    {"Key": "prefix/a.txt"},
                    {"Key": "prefix/nested/b.txt"},
                ]
            },
        ]
    )
    monkeypatch.setattr(s3_utils, "_create_s3_client", lambda profile_name=None: client)

    listed = s3_utils.list_s3_folder("s3://bucket/prefix")
    assert listed == ["s3://bucket/prefix/a.txt", "s3://bucket/prefix/nested/b.txt"]
    assert client.paginator.calls[0] == {"Bucket": "bucket", "Prefix": "prefix/"}

    source = tmp_path / "file.txt"
    source.write_text("hello", encoding="utf-8")
    callback = object()
    uploaded = s3_utils.upload_file_to_s3(
        source, "bucket/path/file.txt", callback=callback
    )
    assert uploaded == "s3://bucket/path/file.txt"
    assert client.uploads[0]["extra_args"]["ContentType"] == "text/plain"
    assert client.uploads[0]["callback"] is callback

    explicit = s3_utils.upload_file_to_s3(
        source,
        "path/file.bin",
        bucket_name="bucket",
        extra_args={"ContentType": "application/custom"},
    )
    assert explicit == "s3://bucket/path/file.bin"
    assert client.uploads[1]["extra_args"]["ContentType"] == "application/custom"

    assert s3_utils.check_s3_file_exists("s3://bucket/path/file.txt") is True
    assert client.heads[-1] == {"Bucket": "bucket", "Key": "path/file.txt"}

    local_path = tmp_path / "nested" / "download.txt"
    result_path = s3_utils.download_file_from_s3(
        "path/file.txt", local_path, bucket_name="bucket", callback=callback
    )
    assert result_path == str(local_path)
    assert local_path.read_text(encoding="utf-8") == "bucket/path/file.txt"
    assert client.downloads[-1]["callback"] is callback

    assert (
        s3_utils.get_s3_file_url("s3://bucket/path/file.txt", use_public_url=True)
        == "https://bucket.s3.amazonaws.com/path/file.txt"
    )
    assert s3_utils.get_s3_file_url("s3://bucket/path/file.txt") == (
        "https://signed.example/object"
    )
    assert client.presigned[-1]["operation"] == "get_object"
    assert client.presigned[-1]["Params"] == {
        "Bucket": "bucket",
        "Key": "path/file.txt",
    }


def test_upload_directory_variants_and_parse_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    (root / "nested").mkdir(parents=True)
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "nested" / "b.txt").write_text("b", encoding="utf-8")
    (root / "nested" / "c.bin").write_bytes(b"c")

    uploaded_calls: list[str] = []

    def fake_upload(file_path: Path, s3_path: str, **_kwargs: Any) -> str:
        uploaded_calls.append(s3_path)
        if str(file_path).endswith("b.txt"):
            raise RuntimeError("skip this file")
        return s3_path

    monkeypatch.setattr(s3_utils, "upload_file_to_s3", fake_upload)
    recursive = s3_utils.upload_directory_to_s3(
        root, "s3://bucket/prefix", file_pattern="*.txt"
    )
    assert recursive == ["s3://bucket/prefix/a.txt"]
    assert "s3://bucket/prefix/nested/b.txt" in uploaded_calls

    uploaded_calls.clear()
    recursive_all = s3_utils.upload_directory_to_s3(root, "s3://bucket/prefix")
    assert "s3://bucket/prefix/a.txt" in recursive_all
    assert "s3://bucket/prefix/nested/c.bin" in recursive_all

    uploaded_calls.clear()
    nonrecursive = s3_utils.upload_directory_to_s3(
        root, "prefix", bucket_name="bucket", recursive=False
    )
    assert nonrecursive == ["s3://bucket/prefix/a.txt"]
    assert uploaded_calls == ["s3://bucket/prefix/a.txt"]

    uploaded_calls.clear()
    nonrecursive_pattern = s3_utils.upload_directory_to_s3(
        root, "s3://bucket/prefix", recursive=False, file_pattern="*.txt"
    )
    assert nonrecursive_pattern == ["s3://bucket/prefix/a.txt"]

    with pytest.raises(FileNotFoundError, match="Directory not found"):
        s3_utils.upload_directory_to_s3(root / "missing", "s3://bucket/prefix")
    with pytest.raises(ValueError, match="not a directory"):
        s3_utils.upload_directory_to_s3(root / "a.txt", "s3://bucket/prefix")

    assert s3_utils._parse_s3_path("bucket/key") == ("bucket", "key")
    assert s3_utils._parse_s3_path("key", bucket_name="bucket") == ("bucket", "key")


def test_delete_s3_path_single_and_recursive(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeS3Client(
        [
            {},
            {"Contents": []},
            {
                "Contents": [
                    {"Key": "folder/a.txt"},
                    {"Key": "folder/b.txt"},
                ]
            },
        ]
    )
    monkeypatch.setattr(s3_utils, "_create_s3_client", lambda profile_name=None: client)

    assert s3_utils.delete_s3_path("s3://bucket/file.txt") is True
    assert client.deleted_objects == [{"Bucket": "bucket", "Key": "file.txt"}]

    result = s3_utils.delete_s3_path("s3://bucket/folder", recursive=True, max_keys=2)
    assert result == {"deleted": 1, "failed": 1}
    assert client.paginator.calls[-1] == {
        "Bucket": "bucket",
        "Prefix": "folder/",
        "PaginationConfig": {"PageSize": 2},
    }
    assert client.deleted_batches[-1]["Delete"]["Objects"] == [
        {"Key": "folder/a.txt"},
        {"Key": "folder/b.txt"},
    ]

    empty_client = FakeS3Client([{}])
    monkeypatch.setattr(
        s3_utils, "_create_s3_client", lambda profile_name=None: empty_client
    )
    assert s3_utils.delete_s3_path("s3://bucket/empty/", recursive=True) == {
        "deleted": 0,
        "failed": 0,
    }
