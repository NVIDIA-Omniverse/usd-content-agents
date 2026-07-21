# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for storage backends and StorageConfig.

Tests:
- LocalSessionStore CRUD operations
- StorageConfig env var parsing with JA_ prefix
"""

import os
from pathlib import Path

import pytest

from ...service.storage import LocalSessionStore, StorageConfig
from ...service.storage import local_store as local_store_module
from ...service.storage.base import METADATA_KEY


@pytest.mark.unit
class TestLocalSessionStoreCRUD:
    """Test LocalSessionStore lifecycle and data operations."""

    @pytest.mark.asyncio
    async def test_init_session_creates_directory(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        assert (tmp_path / "s1").is_dir()

    @pytest.mark.asyncio
    async def test_init_session_rejects_nested_identifier(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))

        with pytest.raises(ValueError, match="storage child"):
            await store.init_session("nested/session")

        assert not (tmp_path / "nested").exists()

    @pytest.mark.asyncio
    async def test_delete_session_removes_directory(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        await store.put_bytes("s1", "file.txt", b"data")
        await store.delete_session("s1")
        assert not (tmp_path / "s1").exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session_is_noop(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.delete_session("nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_put_and_get_json(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        data = {"key": "value", "nested": {"a": 1}}
        await store.put_json("s1", METADATA_KEY, data)
        result = await store.get_json("s1", METADATA_KEY)
        assert result == data
        assert (tmp_path / "s1" / METADATA_KEY).stat().st_mode & 0o777 == 0o600

    @pytest.mark.asyncio
    async def test_get_json_returns_none_for_missing(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        result = await store.get_json("s1", "nonexistent.json")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_json_returns_none_for_malformed_json(
        self, tmp_path: Path
    ) -> None:
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        await store.put_bytes("s1", "malformed.json", b"{")

        assert await store.get_json("s1", "malformed.json") is None

    @pytest.mark.asyncio
    async def test_put_and_read_bytes(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        await store.put_bytes("s1", "data.bin", b"hello bytes")
        stream = await store.open_read("s1", "data.bin")
        assert stream.read() == b"hello bytes"
        stream.close()

    @pytest.mark.asyncio
    async def test_atomic_create_and_single_key_delete(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        claim_path = tmp_path / "s1" / "claim"

        assert await store.put_bytes_if_absent("s1", "claim", b"run-a") is True
        assert await store.put_bytes_if_absent("s1", "claim", b"run-b") is False
        assert claim_path.read_bytes() == b"run-a"
        assert list(claim_path.parent.iterdir()) == [claim_path]
        stream = await store.open_read("s1", "claim")
        assert stream.read() == b"run-a"
        stream.close()
        await store.delete_key("s1", "claim")
        assert await store.exists("s1", "claim") is False

    @pytest.mark.asyncio
    async def test_put_bytes_if_absent_publishes_only_complete_file(
        self, tmp_path, monkeypatch
    ):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        claim_path = tmp_path / "s1" / ".active_run"
        payload = b"complete-run-claim"
        real_write = local_store_module.os.write
        real_link = local_store_module.os.link
        write_count = 0

        def write_one_byte(descriptor, buffer):
            nonlocal write_count
            assert not claim_path.exists()
            write_count += 1
            return real_write(descriptor, buffer[:1])

        def publish(
            source,
            destination,
            *,
            src_dir_fd,
            dst_dir_fd,
            follow_symlinks,
        ):
            source_descriptor = os.open(source, os.O_RDONLY, dir_fd=src_dir_fd)
            try:
                assert os.read(source_descriptor, len(payload) + 1) == payload
            finally:
                os.close(source_descriptor)
            assert destination == claim_path.name
            assert not claim_path.exists()
            return real_link(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )

        monkeypatch.setattr(local_store_module.os, "write", write_one_byte)
        monkeypatch.setattr(local_store_module.os, "link", publish)

        assert await store.put_bytes_if_absent("s1", ".active_run", payload) is True
        assert write_count == len(payload)
        assert claim_path.read_bytes() == payload
        assert list(claim_path.parent.iterdir()) == [claim_path]

    @pytest.mark.asyncio
    async def test_put_bytes_if_absent_cleans_resources_after_write_failure(
        self, tmp_path, monkeypatch
    ):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        written_descriptors = []

        def fail_write(descriptor, _buffer):
            written_descriptors.append(descriptor)
            raise OSError("injected write failure")

        monkeypatch.setattr(local_store_module.os, "write", fail_write)

        with pytest.raises(OSError, match="injected write failure"):
            await store.put_bytes_if_absent("s1", ".active_run", b"run-a")

        assert written_descriptors
        for descriptor in written_descriptors:
            with pytest.raises(OSError):
                os.fstat(descriptor)
        assert list((tmp_path / "s1").iterdir()) == []

    @pytest.mark.asyncio
    async def test_exists(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        assert not await store.exists("s1", "file.txt")
        await store.put_bytes("s1", "file.txt", b"x")
        assert await store.exists("s1", "file.txt")

    @pytest.mark.asyncio
    async def test_list_keys(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        await store.put_bytes("s1", "a.txt", b"a")
        await store.put_bytes("s1", "sub/b.txt", b"b")
        keys = await store.list_keys("s1")
        assert sorted(keys) == ["a.txt", "sub/b.txt"]

    @pytest.mark.asyncio
    async def test_list_keys_with_prefix(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        await store.put_bytes("s1", "cache/a.txt", b"a")
        await store.put_bytes("s1", "cache/b.txt", b"b")
        await store.put_bytes("s1", "other.txt", b"c")
        keys = await store.list_keys("s1", prefix="cache/")
        assert sorted(keys) == ["cache/a.txt", "cache/b.txt"]

    @pytest.mark.asyncio
    async def test_list_sessions(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        await store.put_json("s1", METADATA_KEY, {"id": "s1"})
        await store.init_session("s2")
        await store.put_json("s2", METADATA_KEY, {"id": "s2"})
        sessions = await store.list_sessions()
        assert sorted(sessions) == ["s1", "s2"]

    @pytest.mark.asyncio
    async def test_list_sessions_excludes_dirs_without_metadata(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("valid")
        await store.put_json("valid", METADATA_KEY, {"id": "valid"})
        await store.init_session("empty")
        sessions = await store.list_sessions()
        assert sessions == ["valid"]

    @pytest.mark.asyncio
    async def test_append_and_get_event_log(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        await store.append_event("s1", {"type": "start"})
        await store.append_event("s1", {"type": "progress", "pct": 50})
        events = await store.get_event_log("s1")
        assert len(events) == 2
        assert events[0]["type"] == "start"
        assert events[1]["pct"] == 50

    @pytest.mark.asyncio
    async def test_get_event_log_empty(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        events = await store.get_event_log("s1")
        assert events == []

    @pytest.mark.asyncio
    async def test_get_event_log_returns_empty_for_partial_json(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        await store.put_bytes("s1", "events.jsonl", b'{"type": "partial"')

        assert await store.get_event_log("s1") == []

    @pytest.mark.asyncio
    async def test_put_file(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        src = tmp_path / "source.txt"
        src.write_text("file content")
        await store.put_file("s1", "copied.txt", str(src))
        stream = await store.open_read("s1", "copied.txt")
        assert stream.read() == b"file content"
        stream.close()

    @pytest.mark.asyncio
    async def test_make_public_url_returns_none(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        url = await store.make_public_url("s1", "file.txt")
        assert url is None

    @pytest.mark.asyncio
    async def test_sync_from_local_is_noop(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        count = await store.sync_from_local("s1", str(tmp_path))
        assert count == 0

    @pytest.mark.asyncio
    async def test_invalidate_sessions_cache_is_noop(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        store.invalidate_sessions_cache()  # Should not raise

    @pytest.mark.asyncio
    async def test_kind_is_local(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        assert store.kind == "local"

    @pytest.mark.asyncio
    async def test_nested_key_creates_parent_dirs(self, tmp_path):
        store = LocalSessionStore(root_dir=str(tmp_path))
        await store.init_session("s1")
        await store.put_bytes("s1", "deep/nested/file.txt", b"deep")
        assert await store.exists("s1", "deep/nested/file.txt")

    @pytest.mark.asyncio
    async def test_reserved_paths_are_invisible_and_writes_fail_loudly(
        self, tmp_path: Path
    ) -> None:
        store = LocalSessionStore(root_dir=str(tmp_path))
        session_dir = store._session_dir("s1")
        reserved = session_dir / "cache" / ".pipeline_temp" / "config.json"
        reserved.parent.mkdir(parents=True)
        reserved.write_text('{"api_key": "sentinel"}', encoding="utf-8")
        alias = session_dir / "cache" / "alias.json"
        alias.symlink_to(reserved)

        assert await store.list_keys("s1") == []
        assert await store.list_keys("s1", r"cache\.pipeline_temp") == []
        assert not await store.exists("s1", "cache/.pipeline_temp/config.json")
        assert not await store.exists("s1", "cache/alias.json")
        assert await store.get_json("s1", "cache/.pipeline_temp/config.json") is None
        with pytest.raises(FileNotFoundError):
            await store.open_read("s1", r"cache\.pipeline_temp\config.json")
        with pytest.raises(ValueError, match="reserved"):
            await store.put_bytes(
                "s1",
                "cache/.pipeline_temp/new.json",
                b"{}",
            )

        safe_target = session_dir / "cache" / "public.json"
        safe_target.write_bytes(b"public")
        safe_alias = session_dir / "cache" / "public-alias.json"
        safe_alias.symlink_to(safe_target)
        with pytest.raises(FileNotFoundError):
            await store.open_read("s1", "cache/public-alias.json")

        outside = tmp_path / "outside.json"
        outside.write_bytes(b"outside")
        outside_alias = session_dir / "cache" / "outside-alias.json"
        outside_alias.symlink_to(outside)
        with pytest.raises(FileNotFoundError):
            await store.open_read("s1", "cache/outside-alias.json")


@pytest.mark.unit
class TestStorageConfig:
    """Test StorageConfig construction and defaults.

    Note: StorageConfig is a dataclass whose defaults call os.getenv() at
    class definition time. For runtime override, fields are passed explicitly
    (as ServiceConfig.build_session_store() does). We test both the defaults
    and explicit construction.
    """

    def test_defaults(self):
        """Test default values (as evaluated at import time)."""
        cfg = StorageConfig()
        # kind defaults to "local" (unless JA_STORAGE_KIND was set before import)
        assert cfg.kind in ("local", "s3")
        assert cfg.s3_prefix == "" or isinstance(cfg.s3_prefix, str)
        assert isinstance(cfg.s3_use_path_style, bool)
        assert isinstance(cfg.s3_create_bucket, bool)
        assert isinstance(cfg.s3_presign, bool)
        assert isinstance(cfg.s3_sessions_cache_ttl, int)

    def test_explicit_construction(self):
        """Test that all fields can be set explicitly."""
        cfg = StorageConfig(
            kind="s3",
            s3_bucket="my-bucket",
            s3_prefix="wu/asset/prod",
            s3_region="eu-west-1",
            s3_endpoint_url="http://localhost:9000",
            s3_access_key_id="AKID",
            s3_secret_access_key="SECRET",
            s3_session_token="TOKEN",
            s3_use_path_style=False,
            s3_create_bucket=True,
            s3_presign=False,
            s3_sessions_cache_ttl=30,
        )
        assert cfg.kind == "s3"
        assert cfg.s3_bucket == "my-bucket"
        assert cfg.s3_prefix == "wu/asset/prod"
        assert cfg.s3_region == "eu-west-1"
        assert cfg.s3_endpoint_url == "http://localhost:9000"
        assert cfg.s3_use_path_style is False
        assert cfg.s3_create_bucket is True
        assert cfg.s3_presign is False
        assert cfg.s3_sessions_cache_ttl == 30

    def test_local_root_default(self):
        """Test that local_root has a sensible default."""
        cfg = StorageConfig()
        assert isinstance(cfg.local_root, str)
        assert len(cfg.local_root) > 0

    def test_env_vars_use_ja_prefix(self):
        """Verify the config source code reads JA_STORAGE_ prefixed env vars."""
        import inspect

        source = inspect.getsource(StorageConfig)
        assert "JA_STORAGE_KIND" in source
        assert "JA_STORAGE_S3_BUCKET" in source
        assert "JA_STORAGE_S3_PREFIX" in source
        # Ensure we're NOT using the unprefixed STORAGE_ form
        assert 'os.getenv("STORAGE_KIND"' not in source
