# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-handle confinement coverage for Windows artifact operations."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from world_understanding.utils import artifacts
from world_understanding.utils.file_locking import exclusive_descriptor_lock

pytestmark = pytest.mark.skipif(
    os.name != "nt",
    reason="requires native Windows handle APIs",
)


def test_windows_confined_directory_close_attempts_every_owned_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    last_error = 0

    def close_handle(handle: int) -> bool:
        nonlocal last_error
        calls.append(handle)
        if handle in {30, 10}:
            last_error = 6 if handle == 30 else 5
            return False
        return True

    monkeypatch.setattr(artifacts, "_close_handle", close_handle)
    monkeypatch.setattr(artifacts.ctypes, "get_last_error", lambda: last_error)
    directory = artifacts._WindowsConfinedDirectory(tmp_path, (10, 20, 30))

    with pytest.raises(OSError) as exc_info:
        directory.close()

    assert exc_info.value.winerror == 6
    assert calls == [30, 20, 10]
    assert directory._owned_handles == ()
    directory.close()


def test_windows_handle_backend_pins_directory_and_supports_core_io(
    tmp_path: Path,
) -> None:
    payload = b"alpha\n\x1abeta\r\n"
    with artifacts.open_confined_directory(tmp_path) as root:
        assert isinstance(root, artifacts._WindowsConfinedDirectory)
        assert root._owned_handles
        with artifacts.open_confined_directory(tmp_path) as reopened:
            assert artifacts.confined_directory_identity(
                root
            ) == artifacts.confined_directory_identity(reopened)
        with artifacts.open_confined_directory_at(
            root,
            "runs/session",
            create=True,
        ) as nested:
            assert isinstance(nested, artifacts._WindowsConfinedDirectory)
            assert nested.path == tmp_path / "runs" / "session"

        assert artifacts.write_bytes_to_confined(root, "runs/result.bin", payload)
        artifacts.append_bytes_to_confined(root, "runs/result.bin", b"tail")
        with artifacts.open_confined_regular_file(
            root,
            "runs/result.bin",
        ) as (stream, metadata):
            assert metadata.st_nlink == 1
            assert stream.read() == payload + b"tail"
        assert artifacts.delete_confined_file(root, "runs/result.bin")
        assert not artifacts.delete_confined_file(root, "runs/result.bin")


def test_windows_confined_directory_identity_rejects_non_native_handle(
    tmp_path: Path,
) -> None:
    regular = tmp_path / "regular.bin"
    regular.write_bytes(b"payload")
    descriptor = os.open(regular, os.O_RDONLY)
    try:
        with pytest.raises(
            artifacts.ArtifactPathError,
            match="held Windows directory handle",
        ):
            artifacts.confined_directory_identity(descriptor)
    finally:
        os.close(descriptor)


def test_windows_confined_directory_identity_rejects_closed_handle(
    tmp_path: Path,
) -> None:
    with artifacts.open_confined_directory(tmp_path) as root:
        artifacts.confined_directory_identity(root)

    with pytest.raises(OSError):
        artifacts.confined_directory_identity(root)


@pytest.mark.parametrize(
    "key",
    [
        "file:stream",
        "result?.json",
        'quote"name.json',
        "pipe|name.json",
        "star*.json",
        "<bad>.json",
        "control\x1f.json",
        "CON",
        "CONIN$.json",
        "COM¹.txt",
        "trailing.",
        "trailing ",
    ],
)
def test_windows_handle_backend_rejects_aliased_components(
    tmp_path: Path,
    key: str,
) -> None:
    with artifacts.open_confined_directory(tmp_path) as root:
        with pytest.raises(ValueError, match="non-canonical Windows"):
            artifacts.write_bytes_to_confined(root, key, b"payload")


def test_windows_handle_backend_preserves_valid_spaced_device_prefix(
    tmp_path: Path,
) -> None:
    key = "CON .json/result.json"
    with artifacts.open_confined_directory(tmp_path) as root:
        assert artifacts.write_bytes_to_confined(root, key, b"payload")
        with artifacts.open_confined_regular_file(root, key) as (stream, _metadata):
            assert stream.read() == b"payload"

    assert (tmp_path / key).read_bytes() == b"payload"


def test_windows_handle_backend_rejects_reparse_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Windows symlink creation is unavailable: {exc}")

    with artifacts.open_confined_directory(tmp_path) as root:
        with pytest.raises(artifacts.ArtifactPathError, match="reparsed"):
            with artifacts.open_confined_directory_at(root, "link"):
                raise AssertionError("should not have opened")


def test_windows_handle_backend_atomic_publish_contract(tmp_path: Path) -> None:
    target = tmp_path / "result.bin"
    target.write_bytes(b"first")

    with artifacts.open_confined_directory(tmp_path) as root:
        assert not artifacts.write_bytes_to_confined(
            root,
            "result.bin",
            b"not published",
            overwrite=False,
        )
        assert artifacts.write_bytes_to_confined(
            root,
            "result.bin",
            b"replacement",
        )

    assert target.read_bytes() == b"replacement"
    assert not list(tmp_path.glob("*.tmp"))


def test_windows_streaming_no_clobber_validates_race_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(b"losing payload")
    target = tmp_path / "result.bin"
    real_set_file_name = artifacts._windows_set_file_name
    real_try_open = artifacts._windows_try_open_existing_regular_file
    validation_calls = 0

    def install_winner_then_publish(
        descriptor: int,
        destination: Path,
        *,
        overwrite: bool,
    ) -> None:
        destination.write_bytes(b"winning payload")
        real_set_file_name(descriptor, destination, overwrite=overwrite)

    def record_winner_validation(parent, leaf_name: str) -> int | None:
        nonlocal validation_calls
        validation_calls += 1
        return real_try_open(parent, leaf_name)

    monkeypatch.setattr(
        artifacts,
        "_windows_set_file_name",
        install_winner_then_publish,
    )
    monkeypatch.setattr(
        artifacts,
        "_windows_try_open_existing_regular_file",
        record_winner_validation,
    )

    with source_path.open("rb") as source:
        source_metadata = os.fstat(source.fileno())
        with artifacts.open_confined_directory(tmp_path) as root:
            assert not artifacts.copy_open_file_to_confined(
                root,
                "result.bin",
                source,
                source_metadata,
                overwrite=False,
            )

    assert validation_calls == 2
    assert target.read_bytes() == b"winning payload"
    assert not list(tmp_path.glob("*.tmp"))


def test_windows_handle_backend_live_writer_truncates_and_streams(
    tmp_path: Path,
) -> None:
    target = tmp_path / "logs" / "child.log"
    target.parent.mkdir()
    target.write_bytes(b"stale content")

    with artifacts.open_confined_directory(tmp_path) as root:
        stream = artifacts.open_confined_binary_writer(root, "logs/child.log")
    try:
        stream.write(b"live output")
        stream.flush()
        os.fsync(stream.fileno())
        assert target.read_bytes() == b"live output"
    finally:
        stream.close()


def test_windows_handle_backend_live_writer_rejects_hardlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "child.log"
    target.write_bytes(b"payload")
    os.link(target, tmp_path / "alias.log")

    with artifacts.open_confined_directory(tmp_path) as root:
        with pytest.raises(artifacts.ArtifactPathError, match="single-link"):
            artifacts.open_confined_binary_writer(root, "child.log")


def test_windows_handle_backend_failed_publish_removes_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = artifacts.os.write

    def fail_after_first(descriptor: int, data: bytes) -> int:
        real_write(descriptor, data[:1])
        raise OSError("device full")

    with artifacts.open_confined_directory(tmp_path) as root:
        monkeypatch.setattr(artifacts.os, "write", fail_after_first)
        with pytest.raises(OSError, match="device full"):
            artifacts.write_bytes_to_confined(root, "result.bin", b"payload")

    assert list(tmp_path.iterdir()) == []


def test_windows_handle_backend_rejects_reparse_file(tmp_path: Path) -> None:
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link = tmp_path / "link.bin"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Windows symlink creation is unavailable: {exc}")

    with artifacts.open_confined_directory(tmp_path) as root:
        with pytest.raises(artifacts.ArtifactPathError, match="reparsed"):
            with artifacts.open_confined_regular_file(root, "link.bin"):
                raise AssertionError("should not have opened")
        with pytest.raises(artifacts.ArtifactPathError, match="reparsed"):
            artifacts.delete_confined_file(root, "link.bin")

    assert outside.read_bytes() == b"outside"


def test_windows_handle_backend_rejects_multiply_linked_file(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"payload")
    os.link(target, tmp_path / "alias.bin")

    with artifacts.open_confined_directory(tmp_path) as root:
        with pytest.raises(artifacts.ArtifactPathError, match="single-link"):
            with artifacts.open_confined_regular_file(root, "target.bin"):
                raise AssertionError("should not have opened")


def test_windows_handle_backend_can_read_caller_owned_hardlink(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"payload")
    os.link(target, tmp_path / "alias.bin")

    with artifacts.open_confined_directory(tmp_path) as root:
        with artifacts.open_confined_regular_file(
            root,
            "target.bin",
            allow_hardlinks=True,
        ) as (stream, metadata):
            assert stream.read() == b"payload"
            assert metadata.st_nlink == 2


def test_windows_handle_backend_opens_lock_descriptor(tmp_path: Path) -> None:
    with artifacts.open_confined_directory(tmp_path) as root:
        with artifacts.open_confined_lock_file(root, "locks/run.lock") as descriptor:
            with exclusive_descriptor_lock(descriptor):
                os.write(descriptor, b"held")

    assert (tmp_path / "locks" / "run.lock").read_bytes() == b"held"


def test_windows_handle_backend_exclusively_creates_lock_descriptor(
    tmp_path: Path,
) -> None:
    with artifacts.open_confined_directory(tmp_path) as root:
        with artifacts.open_confined_lock_file(
            root,
            "locks/run.lock",
            exclusive_create=True,
        ) as descriptor:
            os.write(descriptor, b"held")
            with pytest.raises(FileExistsError):
                with artifacts.open_confined_lock_file(
                    root,
                    "locks/run.lock",
                    exclusive_create=True,
                ):
                    raise AssertionError("existing lock file must not be reopened")

    assert (tmp_path / "locks" / "run.lock").read_bytes() == b"held"


def test_windows_handle_backend_preserves_unicode_components(tmp_path: Path) -> None:
    relative_key = "輸出/scene-😀.bin"
    with artifacts.open_confined_directory(tmp_path) as root:
        assert artifacts.write_bytes_to_confined(root, relative_key, b"payload")

    assert (tmp_path / "輸出" / "scene-😀.bin").read_bytes() == b"payload"


def test_windows_handle_backend_falls_back_when_ancestor_open_is_denied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_ancestor_open(*args: object, **kwargs: object) -> int:
        raise PermissionError("sandbox may traverse but not open this ancestor")

    monkeypatch.setattr(
        artifacts,
        "_windows_open_relative_handle",
        deny_ancestor_open,
    )

    with artifacts.open_confined_directory(tmp_path) as root:
        assert root.path == tmp_path
        assert len(root._owned_handles) == 1


def test_windows_handle_backend_fallback_rejects_redirected_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_ancestor_open(*args: object, **kwargs: object) -> int:
        raise PermissionError("sandbox may traverse but not open this ancestor")

    monkeypatch.setattr(
        artifacts,
        "_windows_open_relative_handle",
        deny_ancestor_open,
    )
    monkeypatch.setattr(
        artifacts,
        "_windows_normalized_handle_path",
        lambda _handle: os.path.normcase(os.fspath(tmp_path.parent)),
    )

    with pytest.raises(artifacts.ArtifactPathError, match="reparse point"):
        with artifacts.open_confined_directory(tmp_path):
            raise AssertionError("should not have opened")
