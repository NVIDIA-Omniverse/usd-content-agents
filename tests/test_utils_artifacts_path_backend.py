# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The confined-artifact backend used where ``openat`` is unavailable.

The descriptor implementation is the production path on POSIX. This backend
takes over where ``os.open`` accepts no ``dir_fd``, and it is plain filesystem
work, so these tests select it explicitly through
``_SUPPORTS_DIRECTORY_DESCRIPTORS`` and run everywhere. Otherwise the whole
backend would only ever execute on a host CI does not run.
"""

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from world_understanding.utils import artifacts


@pytest.fixture(autouse=True)
def _select_path_backend(monkeypatch):
    monkeypatch.setattr(artifacts, "_SUPPORTS_DIRECTORY_DESCRIPTORS", False)
    monkeypatch.setattr(artifacts, "_SUPPORTS_WINDOWS_HANDLE_CONFINEMENT", False)


def _reparse_metadata(mode: int = stat.S_IFDIR | 0o755) -> SimpleNamespace:
    """Stat-like result carrying the Windows reparse-point attribute."""
    return SimpleNamespace(
        st_mode=mode,
        st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
    )


class TestConfinedDirectory:
    def test_opens_an_existing_directory(self, tmp_path):
        with artifacts.open_confined_directory(tmp_path) as root:
            assert isinstance(root, artifacts._PathConfinedDirectory)
            assert root.path == tmp_path

    def test_creates_missing_parents_when_asked(self, tmp_path):
        target = tmp_path / "runs" / "session"
        with artifacts.open_confined_directory(target, create=True) as root:
            assert root.path.is_dir()
        assert target.is_dir()

    def test_missing_directory_reports_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            with artifacts.open_confined_directory(tmp_path / "absent"):
                raise AssertionError("should not have opened")

    def test_existing_non_directory_reports_not_a_directory(self, tmp_path):
        regular = tmp_path / "state.json"
        regular.write_bytes(b"{}")
        with pytest.raises(NotADirectoryError):
            with artifacts.open_confined_directory(regular):
                raise AssertionError("should not have opened")

    def test_creating_over_an_existing_file_reports_not_a_directory(self, tmp_path):
        """``exist_ok`` tolerates a directory, so a file still raises.

        The create path has to swallow that ``FileExistsError`` and let the
        directory check produce the error a descriptor open would report,
        rather than surfacing the raw mkdir failure.
        """
        regular = tmp_path / "state.json"
        regular.write_bytes(b"{}")
        with pytest.raises(NotADirectoryError):
            with artifacts.open_confined_directory(regular, create=True):
                raise AssertionError("should not have opened")

    def test_reparse_point_component_is_refused(self, tmp_path, monkeypatch):
        """A swapped component must fail closed, not be followed."""
        monkeypatch.setattr(Path, "lstat", lambda _self: _reparse_metadata())
        with pytest.raises(artifacts.ArtifactPathError, match="reparsed"):
            with artifacts.open_confined_directory(tmp_path):
                raise AssertionError("should not have opened")

    def test_uninspectable_component_is_refused(self, tmp_path, monkeypatch):
        def deny(_self):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "lstat", deny)
        with pytest.raises(artifacts.ArtifactPathError, match="Cannot inspect"):
            with artifacts.open_confined_directory(tmp_path):
                raise AssertionError("should not have opened")

    def test_is_not_a_file_descriptor(self, tmp_path):
        """Misuse must fail loudly rather than act on an unrelated descriptor."""
        with artifacts.open_confined_directory(tmp_path) as root:
            with pytest.raises(TypeError, match="not a file descriptor"):
                os.fstat(root)

    def test_opens_and_creates_a_nested_directory(self, tmp_path):
        with artifacts.open_confined_directory(tmp_path) as root:
            with artifacts.open_confined_directory_at(
                root,
                "runs/session",
                create=True,
                mode=0o700,
            ) as nested:
                assert isinstance(nested, artifacts._PathConfinedDirectory)
                assert nested.path == tmp_path / "runs" / "session"

    def test_nested_directory_rejects_a_regular_file(self, tmp_path):
        (tmp_path / "runs").write_bytes(b"not a directory")
        with artifacts.open_confined_directory(tmp_path) as root:
            with pytest.raises(NotADirectoryError):
                with artifacts.open_confined_directory_at(
                    root,
                    "runs/session",
                    create=True,
                ):
                    raise AssertionError("should not have opened")

    def test_nested_directory_rejects_a_reparse_point(self, tmp_path, monkeypatch):
        target = tmp_path / "runs"
        target.mkdir()
        real_lstat = Path.lstat

        def reparse_target(path):
            if path == target:
                return _reparse_metadata()
            return real_lstat(path)

        monkeypatch.setattr(Path, "lstat", reparse_target)
        with artifacts.open_confined_directory(tmp_path) as root:
            with pytest.raises(artifacts.ArtifactPathError, match="reparsed"):
                with artifacts.open_confined_directory_at(root, "runs/session"):
                    raise AssertionError("should not have opened")


class TestConfinedLockFile:
    def test_creates_and_holds_a_regular_lock_file(self, tmp_path):
        with artifacts.open_confined_directory(tmp_path) as root:
            with artifacts.open_confined_lock_file(root, "state.lock") as descriptor:
                assert stat.S_ISREG(os.fstat(descriptor).st_mode)
                os.write(descriptor, b"held")
        assert (tmp_path / "state.lock").read_bytes() == b"held"

    def test_creates_missing_key_parents(self, tmp_path):
        with artifacts.open_confined_directory(tmp_path) as root:
            with artifacts.open_confined_lock_file(root, "nested/state.lock") as fd:
                assert fd >= 0
        assert (tmp_path / "nested" / "state.lock").is_file()

    def test_exclusive_create_rejects_an_existing_lock_file(self, tmp_path):
        with artifacts.open_confined_directory(tmp_path) as root:
            with artifacts.open_confined_lock_file(
                root,
                "nested/state.lock",
                exclusive_create=True,
            ) as descriptor:
                os.write(descriptor, b"held")
                with pytest.raises(FileExistsError):
                    with artifacts.open_confined_lock_file(
                        root,
                        "nested/state.lock",
                        exclusive_create=True,
                    ):
                        raise AssertionError("existing lock file must not be reopened")

        assert (tmp_path / "nested" / "state.lock").read_bytes() == b"held"

    def test_rejects_a_traversing_key(self, tmp_path):
        with artifacts.open_confined_directory(tmp_path) as root:
            with pytest.raises(ValueError):
                with artifacts.open_confined_lock_file(root, "../escape.lock"):
                    raise AssertionError("should not have opened")


class TestConfinedRead:
    def test_opens_a_regular_file_and_preserves_bytes(self, tmp_path):
        payload = b"alpha\n\x00\xff"
        (tmp_path / "artifact.bin").write_bytes(payload)

        with artifacts.open_confined_directory(tmp_path) as root:
            with artifacts.open_confined_regular_file(
                root,
                "artifact.bin",
            ) as (stream, metadata):
                assert stat.S_ISREG(metadata.st_mode)
                assert stream.read() == payload

    def test_rejects_a_reparse_point_target(self, tmp_path, monkeypatch):
        target = tmp_path / "artifact.bin"
        target.write_bytes(b"outside")
        real_lstat = Path.lstat

        def reparse_target(path):
            if path == target:
                return _reparse_metadata(mode=stat.S_IFREG | 0o600)
            return real_lstat(path)

        monkeypatch.setattr(Path, "lstat", reparse_target)
        with artifacts.open_confined_directory(tmp_path) as root:
            with pytest.raises(artifacts.ArtifactPathError, match="reparsed"):
                with artifacts.open_confined_regular_file(root, "artifact.bin"):
                    raise AssertionError("should not have opened")


class TestConfinedWrite:
    def test_publishes_bytes(self, tmp_path):
        with artifacts.open_confined_directory(tmp_path) as root:
            assert artifacts.write_bytes_to_confined(root, "result.bin", b"payload")
        assert (tmp_path / "result.bin").read_bytes() == b"payload"

    def test_writes_bytes_verbatim_without_newline_translation(self, tmp_path):
        """Artifacts are bytes; the platform must not rewrite line endings.

        The Windows CRT opens a descriptor in text-translation mode unless the
        binary flag is set, which turns every ``\\n`` into ``\\r\\n`` and every
        ``\\r\\n`` into ``\\r\\r\\n``. That silently corrupts any artifact whose
        bytes are not plain LF text, and this backend is the one that runs
        where descriptors are unavailable.
        """
        payload = b"alpha\nbeta\r\ngamma\r\n\ndelta\x00\xff"
        with artifacts.open_confined_directory(tmp_path) as root:
            assert artifacts.write_bytes_to_confined(root, "result.bin", payload)
        written = (tmp_path / "result.bin").read_bytes()
        assert written == payload, (
            f"artifact bytes were rewritten: {written!r} != {payload!r}"
        )

    def test_replaces_an_existing_artifact_when_overwriting(self, tmp_path):
        (tmp_path / "result.bin").write_bytes(b"stale")
        with artifacts.open_confined_directory(tmp_path) as root:
            assert artifacts.write_bytes_to_confined(root, "result.bin", b"fresh")
        assert (tmp_path / "result.bin").read_bytes() == b"fresh"

    def test_publishes_without_overwrite_when_the_target_is_absent(self, tmp_path):
        """The no-clobber publish must still publish when nothing is there.

        Refusing to clobber is only half the contract; the ordinary case has
        to succeed and report that it did.
        """
        with artifacts.open_confined_directory(tmp_path) as root:
            published = artifacts.write_bytes_to_confined(
                root, "result.bin", b"first writer", overwrite=False
            )
        assert published is True
        assert (tmp_path / "result.bin").read_bytes() == b"first writer"
        leftovers = [q.name for q in tmp_path.iterdir() if q.name.endswith(".tmp")]
        assert leftovers == [], f"transaction files left behind: {leftovers}"

    def test_does_not_clobber_a_writer_that_wins_the_race(self, tmp_path, monkeypatch):
        """Losing the existence race must not destroy the other writer.

        The existence check and the publish are two steps. If another writer
        creates the target in between, an unconditional replace would delete
        their file and still report success. Simulate that by making the
        pre-check see nothing while the file is really there.
        """
        target = tmp_path / "result.bin"
        target.write_bytes(b"written by the other process")

        real_exists = Path.exists

        def _blind_to_the_target(self, *args, **kwargs):
            if self == target:
                return False
            return real_exists(self, *args, **kwargs)

        monkeypatch.setattr(Path, "exists", _blind_to_the_target)
        with artifacts.open_confined_directory(tmp_path) as root:
            published = artifacts.write_bytes_to_confined(
                root, "result.bin", b"mine", overwrite=False
            )
        assert published is False
        assert target.read_bytes() == b"written by the other process"
        leftovers = [q.name for q in tmp_path.iterdir() if q.name.endswith(".tmp")]
        assert leftovers == [], f"transaction files left behind: {leftovers}"

    def test_refuses_to_replace_when_not_overwriting(self, tmp_path):
        (tmp_path / "result.bin").write_bytes(b"original")
        with artifacts.open_confined_directory(tmp_path) as root:
            assert not artifacts.write_bytes_to_confined(
                root, "result.bin", b"replacement", overwrite=False
            )
        assert (tmp_path / "result.bin").read_bytes() == b"original"

    def test_publish_is_atomic_and_leaves_no_transaction_behind(
        self, tmp_path, monkeypatch
    ):
        """A failed write must not leave a partial file or a stray transaction."""
        real_write = os.write

        def fail_after_first(descriptor, data):
            real_write(descriptor, data[:1])
            raise OSError("device full")

        with artifacts.open_confined_directory(tmp_path) as root:
            monkeypatch.setattr(artifacts.os, "write", fail_after_first)
            with pytest.raises(OSError, match="device full"):
                artifacts.write_bytes_to_confined(root, "result.bin", b"payload")

        assert not (tmp_path / "result.bin").exists()
        assert list(tmp_path.iterdir()) == []

    def test_writes_a_nested_key(self, tmp_path):
        with artifacts.open_confined_directory(tmp_path) as root:
            assert artifacts.write_bytes_to_confined(root, "a/b/c.bin", b"deep")
        assert (tmp_path / "a" / "b" / "c.bin").read_bytes() == b"deep"


class TestConfinedAppend:
    def test_appends_bytes_verbatim_and_creates_parents(self, tmp_path):
        first = b"alpha\n"
        second = b"beta\r\n\x00\xff"
        with artifacts.open_confined_directory(tmp_path) as root:
            artifacts.append_bytes_to_confined(root, "trace/events.jsonl", first)
            artifacts.append_bytes_to_confined(root, "trace/events.jsonl", second)

        assert (tmp_path / "trace" / "events.jsonl").read_bytes() == first + second

    def test_rejects_a_reparse_point_target(self, tmp_path, monkeypatch):
        target = tmp_path / "events.jsonl"
        target.write_bytes(b"outside")
        real_lstat = Path.lstat

        def reparse_target(path):
            if path == target:
                return _reparse_metadata(mode=stat.S_IFREG | 0o600)
            return real_lstat(path)

        monkeypatch.setattr(Path, "lstat", reparse_target)
        with artifacts.open_confined_directory(tmp_path) as root:
            with pytest.raises(artifacts.ArtifactPathError, match="reparsed"):
                artifacts.append_bytes_to_confined(root, "events.jsonl", b"new")

        assert target.read_bytes() == b"outside"


class TestConfinedDelete:
    def test_deletes_a_regular_file_and_is_idempotent(self, tmp_path):
        target = tmp_path / "nested" / "stale.bin"
        target.parent.mkdir()
        target.write_bytes(b"stale")

        with artifacts.open_confined_directory(tmp_path) as root:
            assert artifacts.delete_confined_file(root, "nested/stale.bin")
            assert not artifacts.delete_confined_file(root, "nested/stale.bin")

        assert not target.exists()

    def test_refuses_a_reparse_point_target(self, tmp_path, monkeypatch):
        target = tmp_path / "stale.bin"
        target.write_bytes(b"outside")
        real_lstat = Path.lstat

        def reparse_target(path):
            if path == target:
                return _reparse_metadata(mode=stat.S_IFREG | 0o600)
            return real_lstat(path)

        monkeypatch.setattr(Path, "lstat", reparse_target)
        with artifacts.open_confined_directory(tmp_path) as root:
            with pytest.raises(artifacts.ArtifactPathError, match="reparsed"):
                artifacts.delete_confined_file(root, "stale.bin")

        assert target.read_bytes() == b"outside"


class TestConfinedTreeRemoval:
    def test_removes_an_owned_tree(self, tmp_path):
        working = tmp_path / "session"
        (working / "nested").mkdir(parents=True)
        (working / "nested" / "artifact.bin").write_bytes(b"x")

        assert artifacts.remove_confined_tree(working, tmp_path)
        assert not working.exists()

    def test_missing_tree_reports_false(self, tmp_path):
        assert not artifacts.remove_confined_tree(tmp_path / "absent", tmp_path)

    def test_refuses_a_non_directory_target(self, tmp_path):
        regular = tmp_path / "session"
        regular.write_bytes(b"not a directory")
        with pytest.raises(ValueError, match="must be a directory"):
            artifacts.remove_confined_tree(regular, tmp_path)

    def test_refuses_a_reparse_point_target(self, tmp_path, monkeypatch):
        working = tmp_path / "session"
        working.mkdir()
        monkeypatch.setattr(Path, "lstat", lambda _self: _reparse_metadata())
        with pytest.raises(ValueError, match="cannot be a symlink"):
            artifacts.remove_confined_tree(working, tmp_path)

    def test_refuses_a_target_outside_the_cleanup_root(self, tmp_path):
        outside = tmp_path.parent / "elsewhere"
        with pytest.raises(ValueError, match="outside the configured cleanup root"):
            artifacts.remove_confined_tree(outside, tmp_path / "root")
