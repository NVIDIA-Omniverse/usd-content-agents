# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Path-backed portable USD publication for hosts without ``dir_fd`` support."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from world_understanding.functions.graphics.so_export import _DirectoryHandle


@pytest.fixture
def path_backend(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    from world_understanding.functions.graphics import so_export

    monkeypatch.setattr(so_export, "_SUPPORTS_DIRECTORY_DESCRIPTORS", False)
    monkeypatch.setattr(so_export, "_HOST_IS_NATIVE_WINDOWS", False)
    return so_export


def _owned_sidecar(
    so_export: Any,
    path: Path,
    payload_name: str,
    payload: bytes,
) -> None:
    path.mkdir()
    (path / so_export.PORTABLE_SIDECAR_MARKER_NAME).write_bytes(
        so_export.PORTABLE_SIDECAR_MARKER_BYTES
    )
    (path / payload_name).write_bytes(payload)


def test_path_directory_create_and_leaf_validation(
    tmp_path: Path,
    path_backend,
) -> None:
    missing = tmp_path / "new" / "child"
    with pytest.raises(FileNotFoundError):
        path_backend._open_directory_nofollow(missing, create=False)

    directory = path_backend._open_directory_nofollow(missing, create=True)
    assert isinstance(directory, path_backend._PathDirectory)
    assert os.path.samestat(path_backend._directory_metadata(directory), missing.stat())
    for unsafe_name in (
        "../escape",
        "nested/escape",
        "nested\\escape",
        "stream:result.usda",
        "CON.usda",
        "CON .usda",
        "com¹.log",
        "lPt9.backup.usda",
        "trailing.",
        "trailing ",
        "wild*.usda",
        "control\x01.usda",
    ):
        with pytest.raises(RuntimeError, match="Unsafe USD export transaction entry"):
            path_backend._path_child(directory, unsafe_name)
    path_backend._close_directory(directory)


def test_native_windows_dispatches_to_handle_backend(
    tmp_path: Path,
    path_backend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_parent = tmp_path / "not-created"
    monkeypatch.setattr(path_backend, "_HOST_IS_NATIVE_WINDOWS", True)
    native_directory = object()
    monkeypatch.setattr(
        path_backend,
        "_windows_open_absolute_directory",
        lambda path, *, create: native_directory,
    )

    opened = path_backend._open_directory_nofollow(output_parent, create=True)

    assert opened is native_directory
    assert not output_parent.exists()


@pytest.mark.parametrize(
    "failure_site",
    ("transaction", "parent_before_clear", "parent_after_clear"),
)
def test_path_cleanup_metadata_races_preserve_transaction(
    tmp_path: Path,
    path_backend,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    parent = path_backend._open_directory_nofollow(tmp_path, create=False)
    transaction_name, transaction, transaction_path = (
        path_backend._create_transaction_directory(parent, ".metadata_race_")
    )
    original_directory_metadata = path_backend._directory_metadata
    original_entry_metadata = path_backend._entry_metadata
    parent_inspections = 0

    def inspect_directory(directory: _DirectoryHandle) -> os.stat_result:
        if failure_site == "transaction" and directory is transaction:
            raise path_backend._OutputCommitRaceError("simulated transaction race")
        return cast(os.stat_result, original_directory_metadata(directory))

    def inspect_entry(
        directory: _DirectoryHandle,
        name: str,
    ) -> os.stat_result | None:
        nonlocal parent_inspections
        if directory is parent and name == transaction_name:
            parent_inspections += 1
            expected_inspection = 1 if failure_site == "parent_before_clear" else 2
            if parent_inspections == expected_inspection:
                raise path_backend._OutputCommitRaceError("simulated parent race")
        return cast(os.stat_result | None, original_entry_metadata(directory, name))

    monkeypatch.setattr(path_backend, "_directory_metadata", inspect_directory)
    monkeypatch.setattr(path_backend, "_entry_metadata", inspect_entry)

    assert (
        path_backend._cleanup_transaction_directory(
            parent,
            transaction_name,
            transaction,
        )
        is False
    )
    assert transaction_path.exists()


def test_path_directory_rejects_reparse_points(
    tmp_path: Path,
    path_backend,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks require host privileges: {exc}")
    with pytest.raises(RuntimeError, match="symlink or reparse point"):
        path_backend._open_directory_nofollow(alias / "output", create=True)
    assert not (outside / "output").exists()


def test_path_rename_noreplace_preserves_existing_destination(
    tmp_path: Path,
    path_backend,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"source")
    destination.write_bytes(b"destination")
    directory = path_backend._open_directory_nofollow(tmp_path, create=False)

    with pytest.raises(FileExistsError):
        path_backend._rename_noreplace(
            directory,
            source.name,
            directory,
            destination.name,
        )

    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"destination"


def test_windows_path_transaction_inherits_destination_acl(
    tmp_path: Path,
    path_backend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = path_backend._open_directory_nofollow(tmp_path, create=False)
    original_mkdir = Path.mkdir
    observed_modes: list[int] = []

    def record_mkdir(
        path: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        observed_modes.append(mode)
        original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(path_backend, "_PATH_TRANSACTION_DIRECTORY_MODE", None)
    monkeypatch.setattr(Path, "mkdir", record_mkdir)

    transaction_name, transaction, transaction_path = (
        path_backend._create_transaction_directory(parent, ".acl_")
    )

    assert observed_modes == [0o777]
    assert path_backend._cleanup_transaction_directory(
        parent,
        transaction_name,
        transaction,
    )
    assert not transaction_path.exists()


def test_path_transaction_retries_name_collision(
    tmp_path: Path,
    path_backend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = path_backend._open_directory_nofollow(tmp_path, create=False)
    (tmp_path / ".retry_collision").mkdir()
    tokens = iter(("collision", "fresh"))
    monkeypatch.setattr(path_backend.secrets, "token_hex", lambda _size: next(tokens))

    transaction_name, transaction, transaction_path = (
        path_backend._create_transaction_directory(parent, ".retry_")
    )

    assert transaction_name == ".retry_fresh"
    assert transaction_path == tmp_path / transaction_name
    assert path_backend._cleanup_transaction_directory(
        parent,
        transaction_name,
        transaction,
    )


def test_path_commit_replaces_owned_bundle_and_reports_metadata(
    tmp_path: Path,
    path_backend,
) -> None:
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    transaction_output = transaction / "result.usdc"
    transaction_output.write_bytes(b"new-root")
    transaction_sidecar = transaction / "result.usdc_assets"
    _owned_sidecar(path_backend, transaction_sidecar, "new.png", b"new-texture")

    output = tmp_path / "result.usdc"
    output.write_bytes(b"old-root")
    sidecar = tmp_path / "result.usdc_assets"
    _owned_sidecar(path_backend, sidecar, "old.png", b"old-texture")
    finalized: list[tuple[os.stat_result, os.stat_result, os.stat_result | None]] = []

    path_backend._commit_export(
        transaction_output,
        transaction_sidecar,
        output,
        sidecar,
        post_commit=lambda parent, root, assets: finalized.append(
            (parent, root, assets)
        ),
    )

    assert output.read_bytes() == b"new-root"
    assert (sidecar / "new.png").read_bytes() == b"new-texture"
    assert not (sidecar / "old.png").exists()
    assert len(finalized) == 1
    assert os.path.samestat(finalized[0][0], tmp_path.stat())
    assert os.path.samestat(finalized[0][1], output.stat())
    assert finalized[0][2] is not None
    assert os.path.samestat(finalized[0][2], sidecar.stat())


def test_path_commit_rolls_back_pair_when_post_commit_fails(
    tmp_path: Path,
    path_backend,
) -> None:
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    transaction_output = transaction / "result.usdc"
    transaction_output.write_bytes(b"new-root")
    transaction_sidecar = transaction / "result.usdc_assets"
    _owned_sidecar(path_backend, transaction_sidecar, "new.png", b"new-texture")

    output = tmp_path / "result.usdc"
    output.write_bytes(b"old-root")
    sidecar = tmp_path / "result.usdc_assets"
    _owned_sidecar(path_backend, sidecar, "old.png", b"old-texture")

    def fail_after_commit(*_args: object) -> None:
        raise OSError("simulated finalizer failure")

    with pytest.raises(OSError, match="simulated finalizer failure"):
        path_backend._commit_export(
            transaction_output,
            transaction_sidecar,
            output,
            sidecar,
            post_commit=fail_after_commit,
        )

    assert output.read_bytes() == b"old-root"
    assert (sidecar / "old.png").read_bytes() == b"old-texture"
    assert transaction_output.read_bytes() == b"new-root"
    assert (transaction_sidecar / "new.png").read_bytes() == b"new-texture"


def test_path_commit_rejects_foreign_sidecar_without_publication(
    tmp_path: Path,
    path_backend,
) -> None:
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    transaction_output = transaction / "result.usdc"
    transaction_output.write_bytes(b"new-root")
    transaction_sidecar = transaction / "result.usdc_assets"
    _owned_sidecar(path_backend, transaction_sidecar, "new.png", b"new-texture")

    output = tmp_path / "result.usdc"
    sidecar = tmp_path / "result.usdc_assets"
    sidecar.mkdir()
    (sidecar / "keep.txt").write_bytes(b"foreign")

    with pytest.raises(RuntimeError, match="non-exporter-owned USD sidecar"):
        path_backend._commit_export(
            transaction_output,
            transaction_sidecar,
            output,
            sidecar,
        )

    assert not output.exists()
    assert (sidecar / "keep.txt").read_bytes() == b"foreign"
    assert transaction_output.read_bytes() == b"new-root"
    assert (transaction_sidecar / "new.png").read_bytes() == b"new-texture"


def test_path_owned_sidecar_propagates_identity_race(
    tmp_path: Path,
    path_backend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "result.usdc_assets"
    _owned_sidecar(path_backend, sidecar, "texture.png", b"texture")
    parent = path_backend._open_directory_nofollow(tmp_path, create=False)
    monkeypatch.setattr(os, "fstat", lambda _descriptor: tmp_path.parent.stat())

    with pytest.raises(
        path_backend._OutputCommitRaceError,
        match="sidecar marker changed",
    ):
        path_backend._require_owned_sidecar_at(
            parent,
            sidecar.name,
            sidecar,
        )


def test_path_commit_rolls_back_interrupted_rename(
    tmp_path: Path,
    path_backend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    transaction_output = transaction / "result.usdc"
    transaction_output.write_bytes(b"new-root")
    transaction_sidecar = transaction / "result.usdc_assets"
    _owned_sidecar(path_backend, transaction_sidecar, "new.png", b"new-texture")

    output = tmp_path / "result.usdc"
    output.write_bytes(b"old-root")
    sidecar = tmp_path / "result.usdc_assets"
    _owned_sidecar(path_backend, sidecar, "old.png", b"old-texture")
    original_rename = os.rename
    interrupted = False

    def interrupt_new_output(source: os.PathLike[str], destination: os.PathLike[str]):
        nonlocal interrupted
        original_rename(source, destination)
        if not interrupted and Path(source) == transaction_output:
            interrupted = True
            raise OSError("simulated interrupted rename")

    monkeypatch.setattr(os, "rename", interrupt_new_output)

    with pytest.raises(OSError, match="simulated interrupted rename"):
        path_backend._commit_export(
            transaction_output,
            transaction_sidecar,
            output,
            sidecar,
        )

    assert interrupted is True
    assert output.read_bytes() == b"old-root"
    assert (sidecar / "old.png").read_bytes() == b"old-texture"
    assert transaction_output.read_bytes() == b"new-root"
    assert (transaction_sidecar / "new.png").read_bytes() == b"new-texture"


def test_path_cleanup_removes_nested_and_read_only_entries(
    tmp_path: Path,
    path_backend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = path_backend._open_directory_nofollow(tmp_path, create=False)
    transaction_name, transaction, transaction_path = (
        path_backend._create_transaction_directory(parent, ".cleanup_")
    )
    nested = transaction_path / "nested"
    nested.mkdir()
    read_only = nested / "marker"
    read_only.write_bytes(b"owned")
    original_unlink = Path.unlink
    original_scandir = os.scandir
    denied_once = False

    class WindowsDirEntry:
        def __init__(self, entry: os.DirEntry[str]) -> None:
            self.name = entry.name
            self.path = entry.path

        def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
            raise AssertionError("path cleanup must not use DirEntry.stat on Windows")

    def windows_scandir(path: os.PathLike[str]):
        with original_scandir(path) as entries:
            return [WindowsDirEntry(entry) for entry in entries]

    def deny_first_unlink(path: Path, *args, **kwargs) -> None:
        nonlocal denied_once
        if path == read_only and not denied_once:
            denied_once = True
            raise PermissionError("simulated Windows read-only attribute")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_first_unlink)
    monkeypatch.setattr(os, "scandir", windows_scandir)

    assert path_backend._cleanup_transaction_directory(
        parent,
        transaction_name,
        transaction,
    )
    assert denied_once is True
    assert not transaction_path.exists()


def test_path_clear_reports_directory_scan_failure(
    tmp_path: Path,
    path_backend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = path_backend._open_directory_nofollow(tmp_path, create=False)

    def fail_scandir(_path: os.PathLike[str]):
        raise OSError("simulated scan failure")

    monkeypatch.setattr(os, "scandir", fail_scandir)

    assert path_backend._clear_directory_contents(directory) is False


def test_path_clear_reports_entry_lstat_failure(
    tmp_path: Path,
    path_backend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = tmp_path / "lstat_failure"
    case.mkdir()
    payload = case / "payload"
    payload.write_bytes(b"owned")
    directory = path_backend._open_directory_nofollow(case, create=False)
    original_lstat = Path.lstat

    def fail_payload_lstat(path: Path):
        if path == payload:
            raise OSError("simulated lstat failure")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_payload_lstat)

    assert path_backend._clear_directory_contents(directory) is False


def test_path_clear_rejects_nested_reparse_point(
    tmp_path: Path,
    path_backend,
) -> None:
    case = tmp_path / "reparse_entry"
    case.mkdir()
    outside = tmp_path / "outside_payload"
    outside.write_bytes(b"foreign")
    alias = case / "alias"
    try:
        alias.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file symlinks require host privileges: {exc}")
    directory = path_backend._open_directory_nofollow(case, create=False)

    assert path_backend._clear_directory_contents(directory) is False
    assert outside.read_bytes() == b"foreign"


@pytest.mark.parametrize(
    "failure_mode",
    ("open", "opened_identity", "recursive", "post_identity", "rmdir"),
)
def test_path_clear_reports_nested_directory_failures(
    tmp_path: Path,
    path_backend,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    case = tmp_path / failure_mode
    case.mkdir()
    nested = case / "nested"
    nested.mkdir()
    directory = path_backend._open_directory_nofollow(case, create=False)

    if failure_mode == "open":
        original_open = path_backend._open_directory_nofollow

        def fail_nested_open(path: Path, *, create: bool):
            if Path(path) == nested:
                raise OSError("simulated child open failure")
            return original_open(path, create=create)

        monkeypatch.setattr(path_backend, "_open_directory_nofollow", fail_nested_open)
    elif failure_mode == "opened_identity":
        original_metadata = path_backend._directory_metadata

        def wrong_child_metadata(opened):
            if (
                isinstance(opened, path_backend._PathDirectory)
                and opened.path == nested
            ):
                return tmp_path.stat()
            return original_metadata(opened)

        monkeypatch.setattr(path_backend, "_directory_metadata", wrong_child_metadata)
    elif failure_mode == "recursive":
        original_clear = path_backend._clear_directory_contents

        def fail_recursive_clear(opened):
            if (
                isinstance(opened, path_backend._PathDirectory)
                and opened.path == nested
            ):
                return False
            return original_clear(opened)

        monkeypatch.setattr(
            path_backend,
            "_clear_directory_contents",
            fail_recursive_clear,
        )
    elif failure_mode == "post_identity":
        original_entry_metadata = path_backend._entry_metadata

        def missing_nested_entry(opened, name: str):
            if (
                isinstance(opened, path_backend._PathDirectory)
                and opened.path == case
                and name == nested.name
            ):
                return None
            return original_entry_metadata(opened, name)

        monkeypatch.setattr(path_backend, "_entry_metadata", missing_nested_entry)
    else:
        original_rmdir = Path.rmdir

        def fail_nested_rmdir(path: Path) -> None:
            if path == nested:
                raise OSError("simulated rmdir failure")
            original_rmdir(path)

        monkeypatch.setattr(Path, "rmdir", fail_nested_rmdir)

    assert path_backend._clear_directory_contents(directory) is False
    assert nested.exists()


@pytest.mark.parametrize("failure_mode", ("identity", "readonly", "unlink"))
def test_path_clear_reports_regular_file_failures(
    tmp_path: Path,
    path_backend,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    case = tmp_path / failure_mode
    case.mkdir()
    payload = case / "payload"
    payload.write_bytes(b"owned")
    directory = path_backend._open_directory_nofollow(case, create=False)

    if failure_mode == "identity":
        original_entry_metadata = path_backend._entry_metadata

        def missing_payload(opened, name: str):
            if (
                isinstance(opened, path_backend._PathDirectory)
                and opened.path == case
                and name == payload.name
            ):
                return None
            return original_entry_metadata(opened, name)

        monkeypatch.setattr(path_backend, "_entry_metadata", missing_payload)
    elif failure_mode == "readonly":
        monkeypatch.setattr(
            Path,
            "unlink",
            lambda _path: (_ for _ in ()).throw(PermissionError("read only")),
        )
        monkeypatch.setattr(
            Path,
            "chmod",
            lambda _path, _mode: (_ for _ in ()).throw(OSError("chmod failed")),
        )
    else:
        monkeypatch.setattr(
            Path,
            "unlink",
            lambda _path: (_ for _ in ()).throw(OSError("unlink failed")),
        )

    assert path_backend._clear_directory_contents(directory) is False
    assert payload.exists()


def test_path_atomic_output_clears_owned_sidecar(
    tmp_path: Path,
    path_backend,
) -> None:
    output = tmp_path / "result.usdz"
    output.write_bytes(b"old-root")
    sidecar = tmp_path / path_backend.portable_sidecar_name(output)
    _owned_sidecar(path_backend, sidecar, "stale.png", b"stale")

    with path_backend._atomic_output_file(
        output,
        clear_portable_sidecar=True,
    ) as transaction_output:
        assert isinstance(transaction_output, Path)
        assert str(transaction_output).startswith(str(tmp_path))
        transaction_output.write_bytes(b"new-root")

    assert output.read_bytes() == b"new-root"
    assert not sidecar.exists()
    assert not list(tmp_path.glob(".usd_output_*"))


def test_path_portable_export_localizes_and_replaces_dependencies(
    tmp_path: Path,
    path_backend,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Ar, Sdf, Usd, UsdShade

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    texture = source_dir / "albedo.png"
    texture.write_bytes(b"path-backend-texture")
    source = source_dir / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("albedo.png")
    )
    stage.GetRootLayer().Save()

    output = tmp_path / "delivery" / "result.usdc"
    for _ in range(2):
        assert path_backend.export_stage_portably(
            stage,
            output,
            approved_dependency_roots=[source_dir],
        )

    sidecar = output.parent / path_backend.portable_sidecar_name(output)
    delivered = Usd.Stage.Open(str(output))
    assert delivered is not None
    asset = delivered.GetPrimAtPath("/World/Shader").GetAttribute("inputs:file").Get()
    assert asset.path.startswith(f"{sidecar.name}/")
    resolved = Ar.GetResolver().Resolve(asset.resolvedPath)
    opened = Ar.GetResolver().OpenAsset(resolved)
    assert opened is not None
    assert bytes(opened.GetBuffer()) == b"path-backend-texture"
    assert not list(output.parent.glob(".result_export_*"))
