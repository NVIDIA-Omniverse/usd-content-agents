# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene Optimizer export must not empty a layered asset (issue #963).

``stage.GetRootLayer().Export(path)`` writes only the root layer and does not
re-anchor relative composition arcs. The optimizer writes to the working
directory rather than beside the source, so a layered asset's arcs dangle. USD
drops an unresolved sublayer with a warning, so the export reports success and
the stage composes to nothing.

Flattening fixes that but collapses authored composition such as variant sets,
so it is applied only when the stage is actually layered. Both halves are pinned
here.
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:  # keep pxr optional at collection time
    from pxr import Usd

_POSIX_DIRFD_ONLY = pytest.mark.skipif(
    os.name != "posix",
    reason="injects failures into the POSIX dir_fd implementation",
)


@pytest.fixture(autouse=True)
def _skip_unavailable_windows_symlink_fixtures(
    monkeypatch: pytest.MonkeyPatch,
):
    if os.name != "nt":
        yield
        return

    original_symlink_to = Path.symlink_to

    def symlink_to(path: Path, *args: Any, **kwargs: Any) -> None:
        try:
            original_symlink_to(path, *args, **kwargs)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314 and not path.name.endswith(
                "current"
            ):
                pytest.skip(f"Windows symlink creation is unavailable: {exc}")
            raise

    monkeypatch.setattr(Path, "symlink_to", symlink_to)
    yield


@pytest.mark.parametrize(
    ("asset_path", "expected"),
    [
        ("OmniPBR.mdl", True),
        ("@OmniPBR.MDL@", True),
        ("./OmniPBR.mdl", False),
        ("materials/OmniPBR.mdl", False),
        ("https://example.invalid/OmniPBR.mdl", False),
        (r"C:\\materials\\OmniPBR.mdl", False),
        ("OmniPBR.png", False),
        ("", False),
    ],
)
def test_bare_mdl_token_detection(asset_path: str, expected: bool) -> None:
    from world_understanding.functions.graphics.so_export import is_bare_mdl_token

    assert is_bare_mdl_token(asset_path) is expected


@pytest.mark.parametrize(
    ("asset_path", "expected"),
    [
        ("OmniPBR.mdl", True),
        ("https://example.invalid/runtime.png", True),
        ("https:/example.invalid/runtime.png", True),
        ("omniverse://server/material.mdl", True),
        ("file:///tmp/material.mdl", False),
        (r"C:\\materials\\OmniPBR.mdl", False),
        ("textures/albedo.png", False),
    ],
)
def test_runtime_resolved_asset_path_detection(
    asset_path: str,
    expected: bool,
) -> None:
    from world_understanding.functions.graphics.so_export import (
        is_runtime_resolved_asset_path,
    )

    assert is_runtime_resolved_asset_path(asset_path) is expected


@pytest.mark.skipif(os.name != "nt", reason="exercises Windows native handles")
def test_windows_native_handles_publish_without_clobber_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.graphics import so_export

    with pytest.raises(FileNotFoundError):
        so_export._open_directory_nofollow(tmp_path / "missing", create=False)

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"source")
    destination.write_bytes(b"destination")
    parent = so_export._open_directory_nofollow(tmp_path, create=False)
    transaction = None
    try:
        with pytest.raises(FileExistsError):
            so_export._rename_noreplace(
                parent,
                source.name,
                parent,
                destination.name,
            )
        assert source.read_bytes() == b"source"
        assert destination.read_bytes() == b"destination"

        (tmp_path / ".txn_collision").mkdir()
        tokens = iter(("collision", "fresh"))
        monkeypatch.setattr(so_export.secrets, "token_hex", lambda _size: next(tokens))
        name, transaction, transaction_path = so_export._create_transaction_directory(
            parent,
            ".txn_",
        )
        assert name == ".txn_fresh"
        nested = transaction_path / "nested"
        nested.mkdir()
        (nested / "artifact.bin").write_bytes(b"artifact")
        assert so_export._cleanup_transaction_directory(
            parent,
            name,
            transaction,
        )
    finally:
        if transaction is not None:
            so_export._close_directory(transaction)
        so_export._close_directory(parent)

    assert not (tmp_path / ".txn_fresh").exists()


@pytest.mark.skipif(os.name != "nt", reason="exercises Windows native handles")
def test_windows_directory_close_attempts_every_owned_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.graphics import so_export

    calls: list[int] = []
    last_error = 0

    def close_handle(handle: int) -> bool:
        nonlocal last_error
        calls.append(handle)
        if handle in {30, 10}:
            last_error = 6 if handle == 30 else 5
            return False
        return True

    monkeypatch.setattr(so_export, "_WINDOWS_CLOSE_HANDLE", close_handle)
    monkeypatch.setattr(so_export.ctypes, "get_last_error", lambda: last_error)
    directory = so_export._WindowsDirectory(tmp_path, (10, 20, 30))

    with pytest.raises(OSError) as exc_info:
        directory.close()

    assert exc_info.value.winerror == 6
    assert calls == [30, 20, 10]
    assert directory._owned_handles == ()
    directory.close()


@pytest.mark.skipif(os.name != "nt", reason="exercises Windows native handles")
def test_windows_native_delete_entry_removes_read_only_file(tmp_path: Path) -> None:
    from world_understanding.functions.graphics import so_export

    target = tmp_path / "read-only.bin"
    target.write_bytes(b"owned")
    target.chmod(0o444)
    parent = so_export._open_directory_nofollow(tmp_path, create=False)
    try:
        so_export._delete_entry(parent, target.name, directory=False)
    finally:
        so_export._close_directory(parent)
        if target.exists():
            target.chmod(stat.S_IWRITE)

    assert not target.exists()


@pytest.mark.parametrize(
    "failure_site",
    ["entry_metadata", "child_open", "child_metadata", "file_recheck", "file_delete"],
)
@pytest.mark.skipif(os.name != "nt", reason="exercises Windows native cleanup")
def test_windows_native_cleanup_treats_reparse_rejection_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    from world_understanding.functions.graphics import so_export

    root = tmp_path / "root"
    root.mkdir()
    entry = root / "entry"
    if failure_site in {"child_open", "child_metadata"}:
        entry.mkdir()
    else:
        entry.write_bytes(b"entry")
    expected = entry.lstat()
    root_descriptor = so_export._WindowsDirectory(root, (10,))
    monkeypatch.setattr(so_export, "_WINDOWS_CLOSE_HANDLE", lambda _handle: True)

    metadata_reads = 0

    def entry_metadata(_descriptor, _name: str):
        nonlocal metadata_reads
        metadata_reads += 1
        if failure_site == "entry_metadata" or (
            failure_site == "file_recheck" and metadata_reads == 2
        ):
            raise RuntimeError("simulated reparse rejection")
        return expected

    def reject_reparse(*_args, **_kwargs):
        raise RuntimeError("simulated reparse rejection")

    monkeypatch.setattr(so_export, "_entry_metadata", entry_metadata)
    if failure_site == "child_open":
        monkeypatch.setattr(so_export, "_open_child_directory", reject_reparse)
    elif failure_site == "child_metadata":
        child = so_export._WindowsDirectory(entry, (20,))
        monkeypatch.setattr(
            so_export,
            "_open_child_directory",
            lambda *_args, **_kwargs: child,
        )
        monkeypatch.setattr(
            so_export,
            "_directory_metadata",
            reject_reparse,
        )
    elif failure_site == "file_delete":
        monkeypatch.setattr(so_export, "_delete_entry", reject_reparse)

    try:
        assert so_export._clear_directory_contents(root_descriptor) is False
    finally:
        root_descriptor.close()


@pytest.mark.skipif(os.name != "nt", reason="exercises restricted Windows ancestors")
def test_windows_native_handles_pin_exact_directory_when_ancestor_open_is_denied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.graphics import so_export

    target = tmp_path / "runtime"
    original_open_relative = so_export._windows_open_relative_handle
    denied = False

    def deny_first_ancestor(*args: Any, **kwargs: Any) -> int:
        nonlocal denied
        if not denied:
            denied = True
            raise PermissionError(13, "Access is denied")
        return original_open_relative(*args, **kwargs)

    monkeypatch.setattr(
        so_export,
        "_windows_open_relative_handle",
        deny_first_ancestor,
    )
    descriptor = so_export._open_directory_nofollow(target, create=True)
    try:
        assert descriptor.path == target.resolve()
        assert target.is_dir()
    finally:
        so_export._close_directory(descriptor)

    assert denied is True


@_POSIX_DIRFD_ONLY
def test_open_directory_nofollow_handles_missing_and_concurrent_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.graphics import so_export

    missing = tmp_path / "missing" / "child"
    with pytest.raises(RuntimeError) as exc_info:
        so_export._open_directory_nofollow(missing, create=False)
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)

    original_mkdir = os.mkdir
    raced = False

    def simulate_concurrent_creation(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal raced
        original_mkdir(path, mode=mode, dir_fd=dir_fd)
        if not raced:
            raced = True
            raise FileExistsError

    monkeypatch.setattr(os, "mkdir", simulate_concurrent_creation)
    descriptor = so_export._open_directory_nofollow(missing, create=True)
    os.close(descriptor)

    assert raced is True
    assert missing.is_dir()


@_POSIX_DIRFD_ONLY
def test_rename_noreplace_preserves_existing_destination(tmp_path: Path) -> None:
    from world_understanding.functions.graphics import so_export

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"source")
    destination.write_bytes(b"destination")
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(OSError) as exc_info:
            so_export._rename_noreplace(
                descriptor,
                source.name,
                descriptor,
                destination.name,
            )
    finally:
        os.close(descriptor)

    assert exc_info.value.errno == errno.EEXIST
    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"destination"


@_POSIX_DIRFD_ONLY
def test_create_transaction_directory_retries_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.graphics import so_export

    (tmp_path / ".txn_collision").mkdir()
    tokens = iter(("collision", "fresh"))
    monkeypatch.setattr(so_export.secrets, "token_hex", lambda _size: next(tokens))
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    transaction_descriptor = -1
    try:
        name, transaction_descriptor, transaction_path = (
            so_export._create_transaction_directory(parent_descriptor, ".txn_")
        )
        assert name == ".txn_fresh"
        assert transaction_path.samefile(tmp_path / name)
    finally:
        if transaction_descriptor >= 0:
            os.close(transaction_descriptor)
        os.close(parent_descriptor)


@pytest.mark.parametrize(
    "failure_mode",
    [
        "scandir",
        "entry_stat",
        "child_open",
        "child_opened_inode",
        "child_recursive",
        "child_recheck",
        "child_rmdir",
        "file_recheck",
        "file_unlink",
    ],
)
@_POSIX_DIRFD_ONLY
def test_clear_directory_contents_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    from world_understanding.functions.graphics import so_export

    root = tmp_path / "root"
    root.mkdir()
    root_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    original_scandir = os.scandir
    original_open = os.open
    original_fstat = os.fstat
    original_entry_metadata = so_export._entry_metadata
    original_clear = so_export._clear_directory_contents

    if failure_mode == "scandir":

        def fail_scandir(path: int):
            if path == root_descriptor:
                raise OSError("simulated scandir failure")
            return original_scandir(path)

        monkeypatch.setattr(os, "scandir", fail_scandir)
    elif failure_mode == "entry_stat":
        (root / "entry").write_bytes(b"entry")

        class BrokenEntry:
            name = "entry"

            @staticmethod
            def stat(*, follow_symlinks: bool) -> os.stat_result:
                del follow_symlinks
                raise OSError("simulated stat failure")

        monkeypatch.setattr(
            os,
            "scandir",
            lambda path: [BrokenEntry()]
            if path == root_descriptor
            else original_scandir(path),
        )
    elif failure_mode in {
        "child_open",
        "child_opened_inode",
        "child_recursive",
        "child_recheck",
        "child_rmdir",
    }:
        (root / "child").mkdir()
        child_descriptor: dict[str, int] = {}

        def track_child_open(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if (
                failure_mode == "child_open"
                and path == "child"
                and dir_fd == root_descriptor
            ):
                raise OSError("simulated child open failure")
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
            if path == "child" and dir_fd == root_descriptor:
                child_descriptor["value"] = descriptor
            return descriptor

        monkeypatch.setattr(os, "open", track_child_open)

        if failure_mode == "child_opened_inode":

            def mismatch_child_fstat(descriptor: int):
                metadata = original_fstat(descriptor)
                if descriptor == child_descriptor.get("value"):
                    return SimpleNamespace(
                        st_dev=metadata.st_dev,
                        st_ino=metadata.st_ino + 1,
                        st_mode=metadata.st_mode,
                    )
                return metadata

            monkeypatch.setattr(os, "fstat", mismatch_child_fstat)
        elif failure_mode == "child_recursive":

            def fail_recursive_clear(descriptor: int) -> bool:
                if descriptor == root_descriptor:
                    return original_clear(descriptor)
                return False

            monkeypatch.setattr(
                so_export,
                "_clear_directory_contents",
                fail_recursive_clear,
            )
        elif failure_mode == "child_recheck":
            monkeypatch.setattr(
                so_export,
                "_entry_metadata",
                lambda descriptor, name: None
                if descriptor == root_descriptor and name == "child"
                else original_entry_metadata(descriptor, name),
            )
        elif failure_mode == "child_rmdir":
            original_rmdir = os.rmdir

            def fail_child_rmdir(
                path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                *,
                dir_fd: int | None = None,
            ) -> None:
                if path == "child" and dir_fd == root_descriptor:
                    raise OSError("simulated child rmdir failure")
                original_rmdir(path, dir_fd=dir_fd)

            monkeypatch.setattr(os, "rmdir", fail_child_rmdir)
    else:
        (root / "file").write_bytes(b"file")
        if failure_mode == "file_recheck":
            monkeypatch.setattr(
                so_export,
                "_entry_metadata",
                lambda descriptor, name: None
                if descriptor == root_descriptor and name == "file"
                else original_entry_metadata(descriptor, name),
            )
        else:
            original_unlink = os.unlink

            def fail_file_unlink(
                path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                *,
                dir_fd: int | None = None,
            ) -> None:
                if path == "file" and dir_fd == root_descriptor:
                    raise OSError("simulated file unlink failure")
                original_unlink(path, dir_fd=dir_fd)

            monkeypatch.setattr(os, "unlink", fail_file_unlink)

    try:
        assert so_export._clear_directory_contents(root_descriptor) is False
    finally:
        os.close(root_descriptor)


@pytest.mark.parametrize("failure_mode", ["clear", "recheck", "rmdir"])
@_POSIX_DIRFD_ONLY
def test_cleanup_transaction_directory_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    from world_understanding.functions.graphics import so_export

    transaction_name = "transaction"
    transaction = tmp_path / transaction_name
    transaction.mkdir()
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    transaction_descriptor = os.open(transaction, os.O_RDONLY | os.O_DIRECTORY)
    original_entry_metadata = so_export._entry_metadata

    if failure_mode == "clear":
        monkeypatch.setattr(
            so_export,
            "_clear_directory_contents",
            lambda _descriptor: False,
        )
    elif failure_mode == "recheck":
        metadata_reads = 0

        def disappear_after_clear(descriptor: int, name: str):
            nonlocal metadata_reads
            if descriptor == parent_descriptor and name == transaction_name:
                metadata_reads += 1
                if metadata_reads == 2:
                    return None
            return original_entry_metadata(descriptor, name)

        monkeypatch.setattr(
            so_export,
            "_entry_metadata",
            disappear_after_clear,
        )
    else:
        original_rmdir = os.rmdir

        def fail_transaction_rmdir(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            *,
            dir_fd: int | None = None,
        ) -> None:
            if path == transaction_name and dir_fd == parent_descriptor:
                raise OSError("simulated transaction rmdir failure")
            original_rmdir(path, dir_fd=dir_fd)

        monkeypatch.setattr(os, "rmdir", fail_transaction_rmdir)

    try:
        assert (
            so_export._cleanup_transaction_directory(
                parent_descriptor,
                transaction_name,
                transaction_descriptor,
            )
            is False
        )
    finally:
        os.close(transaction_descriptor)
        os.close(parent_descriptor)


@_POSIX_DIRFD_ONLY
def test_commit_export_closes_owned_transaction_fd_on_destination_open_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.graphics import so_export

    transaction = tmp_path / "transaction"
    transaction.mkdir()
    transaction_output = transaction / "result.usdc"
    transaction_output.write_bytes(b"new-root")
    output = tmp_path / "destination" / "result.usdc"
    sidecar = output.parent / so_export.portable_sidecar_name(output)
    opened_transaction_descriptor = -1
    original_open_directory = so_export._open_directory_nofollow

    def fail_destination_open(path: Path, *, create: bool) -> int:
        nonlocal opened_transaction_descriptor
        if path == output.parent:
            raise RuntimeError("simulated destination open failure")
        opened_transaction_descriptor = original_open_directory(path, create=create)
        return opened_transaction_descriptor

    monkeypatch.setattr(
        so_export,
        "_open_directory_nofollow",
        fail_destination_open,
    )

    with pytest.raises(RuntimeError, match="destination open failure"):
        so_export._commit_export(
            transaction_output,
            transaction / sidecar.name,
            output,
            sidecar,
        )

    assert opened_transaction_descriptor >= 0
    with pytest.raises(OSError) as exc_info:
        os.fstat(opened_transaction_descriptor)
    assert exc_info.value.errno == errno.EBADF


def _export_as_worker_does(stage: Usd.Stage, output: Path) -> int:
    """Export through the production decision so a regression reaches these tests."""
    from world_understanding.functions.graphics.so_worker import export_layer_for

    assert export_layer_for(stage).Export(str(output))
    return len(
        [
            layer
            for layer in stage.GetUsedLayers()
            if layer is not stage.GetSessionLayer()
        ]
    )


def _mesh_paths(usd_path: Path) -> list[str]:
    """Hold the stage: Open(x).Traverse() releases it and raises on expired prims."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None
    return sorted(
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)
    )


def test_layered_asset_keeps_its_geometry(tmp_path: Path) -> None:
    """Geometry behind a relative sublayer must survive the export."""
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    source = tmp_path / "source"
    (source / "0").mkdir(parents=True)
    geometry_stage = Usd.Stage.CreateNew(str(source / "0" / "Body.usda"))
    mesh = UsdGeom.Mesh.Define(geometry_stage, "/World/Box")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 1, 0), Gf.Vec3f(0, 1, 0)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    geometry_stage.GetRootLayer().Save()

    root = source / "scene.usda"
    root_stage = Usd.Stage.CreateNew(str(root))
    root_stage.GetRootLayer().subLayerPaths.append("0/Body.usda")
    root_stage.GetRootLayer().Save()

    assert _mesh_paths(root) == ["/World/Box"], "fixture must start with geometry"

    stage = Usd.Stage.Open(str(root))
    output = tmp_path / "work" / "scene_optimized.usd"
    output.parent.mkdir(parents=True)
    used = _export_as_worker_does(stage, output)

    assert used > 1, "fixture must be layered or it does not exercise the branch"
    assert _mesh_paths(output) == ["/World/Box"], (
        "geometry was lost: the relative sublayer did not survive export"
    )


def test_single_layer_asset_keeps_its_variant_sets(tmp_path: Path) -> None:
    """A single-layer stage must not be flattened.

    Flattening collapses variant sets to the current selection, so a stage with
    no dangling arc to repair must take the root-layer export and keep them.
    """
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    source = tmp_path / "source"
    source.mkdir()
    root = source / "scene.usda"
    stage = Usd.Stage.CreateNew(str(root))
    part = stage.DefinePrim("/World/Part", "Xform")
    variants = part.GetVariantSets().AddVariantSet("lod")
    for name in ("high", "low"):
        variants.AddVariant(name)
        variants.SetVariantSelection(name)
        with variants.GetVariantEditContext():
            UsdGeom.Mesh.Define(
                stage, f"/World/Part/{name}"
            ).CreateFaceVertexCountsAttr([4])
    variants.SetVariantSelection("high")
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(root))
    output = tmp_path / "work" / "scene_optimized.usd"
    output.parent.mkdir(parents=True)
    used = _export_as_worker_does(reopened, output)

    assert used == 1, "fixture must be single-layer to exercise this branch"

    # Hold the stage; Open(x).GetPrimAtPath(y) releases it and expires the prim.
    exported = Usd.Stage.Open(str(output))
    part_out = exported.GetPrimAtPath("/World/Part")
    assert list(part_out.GetVariantSets().GetNames()) == ["lod"], (
        "variant sets were collapsed; a single-layer stage must not be flattened"
    )


def _make_layered_textured_package(tmp_path: Path) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils

    source = tmp_path / "source"
    (source / "0").mkdir(parents=True)
    (source / "textures").mkdir()
    (source / "textures" / "albedo.png").write_bytes(b"portable-texture")

    geometry_stage = Usd.Stage.CreateNew(str(source / "0" / "Body.usda"))
    mesh = UsdGeom.Mesh.Define(geometry_stage, "/World/Box")
    mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    shader = UsdShade.Shader.Define(geometry_stage, "/World/Shader")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../textures/albedo.png")
    )
    geometry_stage.GetRootLayer().Save()

    root = source / "scene.usda"
    root_stage = Usd.Stage.CreateNew(str(root))
    root_stage.GetRootLayer().subLayerPaths.append("0/Body.usda")
    root_stage.GetRootLayer().Save()

    package = tmp_path / "vehicle.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(root), str(package))
    return package


def _shader_asset(usd_path: Path, input_name: str = "file"):
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None
    shader = stage.GetPrimAtPath("/World/Shader")
    value = shader.GetAttribute(f"inputs:{input_name}").Get()
    return stage, value


def test_layered_textured_usdz_export_is_portable_after_delivery_move(
    tmp_path: Path,
) -> None:
    """A packaged texture must survive source deletion and artifact movement."""
    pytest.importorskip("pxr")
    from pxr import Ar, Sdf, Usd, UsdGeom, UsdUtils

    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    package = _make_layered_textured_package(tmp_path)
    source_stage = Usd.Stage.Open(str(package))
    assert source_stage is not None

    work = tmp_path / "work"
    output = work / "vehicle_optimized.usdc"
    assert export_stage_portably(
        source_stage,
        output,
        approved_dependency_roots=[tmp_path],
    )
    sidecar = work / "vehicle_optimized.usdc_assets"
    assert sidecar.is_dir()
    assert (sidecar / ".usd_portable_sidecar").is_file()
    assert [
        path.name
        for path in sidecar.rglob("*")
        if path.is_file() and path.name != ".usd_portable_sidecar"
    ] == ["albedo.png"]
    assert export_stage_portably(
        source_stage,
        output,
        approved_dependency_roots=[tmp_path],
    ), "a retry must replace the prior root and sidecar together"

    package.unlink()
    shutil.rmtree(tmp_path / "source")
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    delivered_output = Path(shutil.move(str(output), delivery / output.name))
    delivered_sidecar = Path(shutil.move(str(sidecar), delivery / sidecar.name))

    delivered_stage, asset = _shader_asset(delivered_output)
    assert any(prim.IsA(UsdGeom.Mesh) for prim in delivered_stage.Traverse())
    assert asset.path.startswith(f"{delivered_sidecar.name}/")
    assert not Path(asset.path).is_absolute()
    assert asset.resolvedPath

    resolved = Ar.GetResolver().Resolve(asset.resolvedPath)
    resolver_asset = Ar.GetResolver().OpenAsset(resolved)
    assert resolver_asset is not None
    assert bytes(resolver_asset.GetBuffer()) == b"portable-texture"

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
        Sdf.AssetPath(str(delivered_output))
    )
    assert len(layers) == 1
    assert unresolved == []
    assert len(assets) == 1
    assert Path(assets[0]).resolve().is_relative_to(delivered_sidecar.resolve())


def test_portable_exports_with_same_stem_keep_distinct_sidecars(
    tmp_path: Path,
) -> None:
    """Different output extensions must never replace each other's assets."""
    pytest.importorskip("pxr")
    from pxr import Ar, Sdf, Usd, UsdShade

    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
        portable_sidecar_name,
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    texture = source_dir / "albedo.png"
    texture.write_bytes(b"same-stem-texture")
    source = source_dir / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("albedo.png")
    )
    stage.GetRootLayer().Save()

    output_dir = tmp_path / "delivery"
    outputs = [output_dir / "model.usda", output_dir / "model.usdc"]
    for output in outputs:
        assert export_stage_portably(
            stage,
            output,
            approved_dependency_roots=[source_dir],
        )

    sidecars = [output.parent / portable_sidecar_name(output) for output in outputs]
    assert [sidecar.name for sidecar in sidecars] == [
        "model.usda_assets",
        "model.usdc_assets",
    ]
    assert all(sidecar.is_dir() for sidecar in sidecars)
    shutil.rmtree(source_dir)

    for output, sidecar in zip(outputs, sidecars, strict=True):
        _, asset = _shader_asset(output)
        assert asset.path.startswith(f"{sidecar.name}/")
        resolver_asset = Ar.GetResolver().OpenAsset(
            Ar.GetResolver().Resolve(asset.resolvedPath)
        )
        assert resolver_asset is not None
        assert bytes(resolver_asset.GetBuffer()) == b"same-stem-texture"


def test_portable_export_validation_checks_outer_package_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pxr")
    from pxr import UsdUtils

    from world_understanding.functions.graphics import so_export

    output = tmp_path / "result.usdc"
    output.write_bytes(b"output")
    sidecar = tmp_path / "result_assets"
    sidecar.mkdir()
    package = sidecar / "dependency.usdz"
    package.write_bytes(b"package")
    package_asset = f"{sidecar.name}/{package.name}[0/albedo.png]"
    assert so_export._outer_asset_identifier(package_asset) == (
        f"{sidecar.name}/{package.name}"
    )

    class OutputLayer:
        realPath = str(output)
        resolvedPath = ""
        identifier = str(output)

    monkeypatch.setattr(
        UsdUtils,
        "ComputeAllDependencies",
        lambda *_args, **_kwargs: ([OutputLayer()], [package_asset], []),
    )

    so_export._validate_portable_export(output, sidecar, lambda _path: False)

    monkeypatch.setattr(
        UsdUtils,
        "ComputeAllDependencies",
        lambda *_args, **_kwargs: ([OutputLayer()], [package.as_uri()], []),
    )
    so_export._validate_portable_export(output, sidecar, lambda _path: False)

    monkeypatch.setattr(
        UsdUtils,
        "ComputeAllDependencies",
        lambda *_args, **_kwargs: ([OutputLayer()], ["../outside.png"], []),
    )
    with pytest.raises(RuntimeError, match="outside its sidecar"):
        so_export._validate_portable_export(output, sidecar, lambda _path: False)

    monkeypatch.setattr(
        UsdUtils,
        "ComputeAllDependencies",
        lambda *_args, **_kwargs: (
            [OutputLayer()],
            ["https://example.invalid/albedo.png"],
            [],
        ),
    )
    with pytest.raises(RuntimeError, match="outside its sidecar"):
        so_export._validate_portable_export(output, sidecar, lambda _path: False)


@pytest.mark.parametrize("layer_kind", ["anonymous", "external"])
def test_portable_export_validation_rejects_external_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layer_kind: str,
) -> None:
    pytest.importorskip("pxr")
    from pxr import UsdUtils

    from world_understanding.functions.graphics import so_export

    output = tmp_path / "result.usdc"
    output.write_bytes(b"output")
    sidecar = tmp_path / "result_assets"
    sidecar.mkdir()

    class DependencyLayer:
        realPath = "" if layer_kind == "anonymous" else str(tmp_path / "external.usda")
        resolvedPath = ""
        identifier = ""

    monkeypatch.setattr(
        UsdUtils,
        "ComputeAllDependencies",
        lambda *_args, **_kwargs: ([DependencyLayer()], [], []),
    )

    expected = "<anonymous>" if layer_kind == "anonymous" else "external.usda"
    with pytest.raises(RuntimeError, match=expected):
        so_export._validate_portable_export(output, sidecar, lambda _path: False)


def test_portable_export_dependency_path_defensive_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Ar

    from world_understanding.functions.graphics import so_export

    network_file = so_export._filesystem_dependency_path(
        "file://asset-server/share/albedo.png"
    )
    assert network_file == Path("//asset-server/share/albedo.png").resolve()

    # Non-file resolver URIs are resolver-owned and do not authorize host paths.
    so_export._require_approved_dependency(
        "https://example.invalid/albedo.png",
        (tmp_path,),
    )

    # Some OpenUSD wrappers return a scalar rather than the usual pair. Ensure
    # the defensive parser terminates instead of looping on the same value.
    package_path = "dependency.usdz[layer.usda]"
    monkeypatch.setattr(Ar, "IsPackageRelativePath", lambda _path: True)
    monkeypatch.setattr(Ar, "SplitPackageRelativePathInner", lambda _path: package_path)
    assert so_export._asset_basename(package_path) == package_path


def test_portable_export_cleanup_removes_files_and_directories(tmp_path: Path) -> None:
    from world_understanding.functions.graphics.so_export import _remove_path

    regular_file = tmp_path / "stale.usdc"
    regular_file.write_bytes(b"stale")
    stale_directory = tmp_path / "stale_assets"
    stale_directory.mkdir()
    (stale_directory / "texture.png").write_bytes(b"stale")

    _remove_path(regular_file)
    _remove_path(stale_directory)

    assert not regular_file.exists()
    assert not stale_directory.exists()


def test_portable_export_commit_finalizes_the_pinned_destination(
    tmp_path: Path,
) -> None:
    from world_understanding.functions.graphics import so_export

    transaction_dir = tmp_path / "transaction"
    transaction_dir.mkdir()
    transaction_output = transaction_dir / "result.usdc"
    transaction_output.write_bytes(b"new-root")
    transaction_sidecar = transaction_dir / "result_assets"
    output = tmp_path / "result.usdc"
    sidecar = tmp_path / "result_assets"
    finalized: list[tuple[os.stat_result, os.stat_result, os.stat_result | None]] = []

    so_export._commit_export(
        transaction_output,
        transaction_sidecar,
        output,
        sidecar,
        post_commit=lambda destination, committed_output, committed_sidecar: (
            finalized.append((destination, committed_output, committed_sidecar))
        ),
    )

    assert output.read_bytes() == b"new-root"
    assert len(finalized) == 1
    destination_stat, output_stat, sidecar_stat = finalized[0]
    assert os.path.samestat(destination_stat, tmp_path.stat())
    assert os.path.samestat(output_stat, output.stat())
    assert sidecar_stat is None


def test_portable_export_commit_restores_prior_pair_on_root_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.graphics import so_export

    transaction_dir = tmp_path / "transaction"
    transaction_dir.mkdir()
    transaction_output = transaction_dir / "result.usdc"
    transaction_output.write_bytes(b"new-root")
    transaction_sidecar = transaction_dir / "result_assets"
    transaction_sidecar.mkdir()
    (transaction_sidecar / so_export.PORTABLE_SIDECAR_MARKER_NAME).write_bytes(
        so_export.PORTABLE_SIDECAR_MARKER_BYTES
    )
    (transaction_sidecar / "new.png").write_bytes(b"new-texture")

    output = tmp_path / "result.usdc"
    output.write_bytes(b"prior-root")
    sidecar = tmp_path / "result_assets"
    sidecar.mkdir()
    (sidecar / ".usd_portable_sidecar").write_bytes(
        b"world-understanding portable USD sidecar v1\n"
    )
    (sidecar / "prior.png").write_bytes(b"prior-texture")

    original_rename = so_export._rename_noreplace

    def fail_root_commit(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        if source_name == transaction_output.name and destination_name == output.name:
            raise OSError("simulated root replace failure")
        original_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(so_export, "_rename_noreplace", fail_root_commit)

    with pytest.raises(OSError, match="simulated root replace failure"):
        so_export._commit_export(
            transaction_output,
            transaction_sidecar,
            output,
            sidecar,
        )

    assert output.read_bytes() == b"prior-root"
    assert (sidecar / "prior.png").read_bytes() == b"prior-texture"
    assert not (sidecar / "new.png").exists()
    assert transaction_output.read_bytes() == b"new-root"


def test_portable_export_commit_no_clobber_rejects_concurrent_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.graphics import so_export

    transaction_dir = tmp_path / "transaction"
    transaction_dir.mkdir()
    transaction_output = transaction_dir / "result.usdc"
    transaction_output.write_bytes(b"new-root")
    transaction_sidecar = transaction_dir / "result_assets"

    output = tmp_path / "result.usdc"
    sidecar = tmp_path / "result_assets"
    original_require_replaceable_output = so_export._require_replaceable_output
    injected = False

    def inject_after_destination_check(
        directory_descriptor: int,
        name: str,
        path: Path,
    ) -> os.stat_result | None:
        nonlocal injected
        metadata = original_require_replaceable_output(
            directory_descriptor,
            name,
            path,
        )
        if path == output and metadata is None and not injected:
            injected = True
            output.write_bytes(b"concurrent-root")
        return metadata

    monkeypatch.setattr(
        so_export,
        "_require_replaceable_output",
        inject_after_destination_check,
    )

    with pytest.raises(so_export._OutputCommitRaceError):
        so_export._commit_export(
            transaction_output,
            transaction_sidecar,
            output,
            sidecar,
            overwrite=False,
        )

    assert injected is True
    assert output.read_bytes() == b"concurrent-root"
    assert transaction_output.read_bytes() == b"new-root"


@pytest.mark.parametrize("interrupt_point", ["rename_return", "metadata_read"])
def test_portable_export_interrupt_after_root_rename_restores_prior_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_point: str,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from world_understanding.functions.graphics import so_export

    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().Save()

    output = tmp_path / "result.usdc"
    output.write_bytes(b"prior-root")
    sidecar = tmp_path / so_export.portable_sidecar_name(output)
    sidecar.mkdir()
    (sidecar / so_export.PORTABLE_SIDECAR_MARKER_NAME).write_bytes(
        so_export.PORTABLE_SIDECAR_MARKER_BYTES
    )
    (sidecar / "prior.png").write_bytes(b"prior-texture")

    original_rename = so_export._rename_noreplace
    original_entry_metadata = so_export._entry_metadata
    interrupted = False
    root_destination_descriptor: int | None = None

    def interrupt_after_root_move(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal interrupted, root_destination_descriptor
        original_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )
        if (
            not interrupted
            and source_name == output.name
            and destination_name == output.name
        ):
            root_destination_descriptor = destination_descriptor
            if interrupt_point == "rename_return":
                interrupted = True
                raise KeyboardInterrupt("simulated post-rename interrupt")

    def interrupt_during_post_rename_metadata(
        directory_descriptor: int,
        name: str,
    ) -> os.stat_result | None:
        nonlocal interrupted
        if (
            interrupt_point == "metadata_read"
            and not interrupted
            and directory_descriptor == root_destination_descriptor
            and name == output.name
        ):
            interrupted = True
            raise KeyboardInterrupt("simulated post-rename interrupt")
        return original_entry_metadata(directory_descriptor, name)

    monkeypatch.setattr(so_export, "_rename_noreplace", interrupt_after_root_move)
    monkeypatch.setattr(
        so_export,
        "_entry_metadata",
        interrupt_during_post_rename_metadata,
    )

    with pytest.raises(KeyboardInterrupt, match="post-rename interrupt"):
        so_export.export_stage_portably(
            stage,
            output,
            approved_dependency_roots=[tmp_path],
        )

    assert interrupted is True
    assert output.read_bytes() == b"prior-root"
    assert (sidecar / "prior.png").read_bytes() == b"prior-texture"
    assert not list(tmp_path.glob(".result_export_*"))


@pytest.mark.parametrize("interrupt_point", ["rename_return", "metadata_read"])
def test_atomic_output_interrupt_after_root_rename_restores_prior_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_point: str,
) -> None:
    from world_understanding.functions.graphics import so_export

    output = tmp_path / "result.usdc"
    output.write_bytes(b"prior-root")
    original_rename = so_export._rename_noreplace
    original_entry_metadata = so_export._entry_metadata
    interrupted = False
    root_destination_descriptor: int | None = None

    def interrupt_after_root_move(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal interrupted, root_destination_descriptor
        original_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )
        if (
            not interrupted
            and source_name == output.name
            and destination_name == output.name
        ):
            root_destination_descriptor = destination_descriptor
            if interrupt_point == "rename_return":
                interrupted = True
                raise KeyboardInterrupt("simulated post-rename interrupt")

    def interrupt_during_post_rename_metadata(
        directory_descriptor: int,
        name: str,
    ) -> os.stat_result | None:
        nonlocal interrupted
        if (
            interrupt_point == "metadata_read"
            and not interrupted
            and directory_descriptor == root_destination_descriptor
            and name == output.name
        ):
            interrupted = True
            raise KeyboardInterrupt("simulated post-rename interrupt")
        return original_entry_metadata(directory_descriptor, name)

    monkeypatch.setattr(so_export, "_rename_noreplace", interrupt_after_root_move)
    monkeypatch.setattr(
        so_export,
        "_entry_metadata",
        interrupt_during_post_rename_metadata,
    )

    with pytest.raises(KeyboardInterrupt, match="post-rename interrupt"):
        with so_export._atomic_output_file(output) as transaction_output:
            transaction_output.write_bytes(b"new-root")

    assert interrupted is True
    assert output.read_bytes() == b"prior-root"
    assert not list(tmp_path.glob(".usd_output_*"))


def test_atomic_output_preserves_transaction_when_post_rename_state_is_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.graphics import so_export

    output = tmp_path / "result.usdc"
    output.write_bytes(b"prior-root")
    original_rename = so_export._rename_noreplace
    original_entry_metadata = so_export._entry_metadata
    root_destination_descriptor: int | None = None

    def mark_root_move(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal root_destination_descriptor
        original_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )
        if source_name == output.name and destination_name == output.name:
            root_destination_descriptor = destination_descriptor

    def fail_public_root_metadata(
        directory_descriptor: int,
        name: str,
    ) -> os.stat_result | None:
        if directory_descriptor == root_destination_descriptor and name == output.name:
            raise OSError("persistent metadata failure")
        return original_entry_metadata(directory_descriptor, name)

    monkeypatch.setattr(so_export, "_rename_noreplace", mark_root_move)
    monkeypatch.setattr(so_export, "_entry_metadata", fail_public_root_metadata)

    with pytest.raises(so_export._OutputCommitRaceError, match="preserved"):
        with so_export._atomic_output_file(output) as transaction_output:
            transaction_output.write_bytes(b"new-root")

    assert output.read_bytes() == b"new-root"
    transactions = list(tmp_path.glob(".usd_output_*"))
    assert len(transactions) == 1
    backup = next(transactions[0].glob(".previous_output_*"))
    assert backup.read_bytes() == b"prior-root"


def test_atomic_output_default_retains_owned_portable_sidecar(tmp_path: Path) -> None:
    from world_understanding.functions.graphics import so_export

    output = tmp_path / "result.usdc"
    output.write_bytes(b"prior-root")
    sidecar = tmp_path / so_export.portable_sidecar_name(output)
    sidecar.mkdir()
    (sidecar / so_export.PORTABLE_SIDECAR_MARKER_NAME).write_bytes(
        so_export.PORTABLE_SIDECAR_MARKER_BYTES
    )
    (sidecar / "texture.png").write_bytes(b"texture")

    with so_export._atomic_output_file(output) as transaction_output:
        transaction_output.write_bytes(b"new-root")

    assert output.read_bytes() == b"new-root"
    assert (sidecar / "texture.png").read_bytes() == b"texture"


def test_atomic_output_sidecar_clear_restores_prior_pair_on_root_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.graphics import so_export

    output = tmp_path / "result.usdc"
    output.write_bytes(b"prior-root")
    sidecar = tmp_path / so_export.portable_sidecar_name(output)
    sidecar.mkdir()
    (sidecar / so_export.PORTABLE_SIDECAR_MARKER_NAME).write_bytes(
        so_export.PORTABLE_SIDECAR_MARKER_BYTES
    )
    (sidecar / "texture.png").write_bytes(b"prior-texture")
    original_move = so_export._move_checked_entry

    def fail_new_root(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
        expected: os.stat_result,
        description: str,
    ) -> os.stat_result:
        if description == "new USD output":
            raise OSError("simulated standalone root failure")
        return original_move(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
            expected,
            description,
        )

    monkeypatch.setattr(so_export, "_move_checked_entry", fail_new_root)

    with pytest.raises(OSError, match="standalone root failure"):
        with so_export._atomic_output_file(
            output,
            clear_portable_sidecar=True,
        ) as transaction_output:
            transaction_output.write_bytes(b"new-root")

    assert output.read_bytes() == b"prior-root"
    assert (sidecar / "texture.png").read_bytes() == b"prior-texture"
    assert not list(tmp_path.glob(".usd_output_*"))


@pytest.mark.parametrize(
    "interrupt_description",
    [
        "existing USD sidecar",
        "existing USD output",
        "new USD sidecar",
        "new USD output",
    ],
)
def test_portable_export_caller_boundary_interrupt_restores_prior_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_description: str,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    from world_understanding.functions.graphics import so_export

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "albedo.png").write_bytes(b"new-texture")
    source = source_dir / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("albedo.png")
    )
    stage.GetRootLayer().Save()

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output = output_dir / "result.usdc"
    output.write_bytes(b"prior-root")
    sidecar = output_dir / so_export.portable_sidecar_name(output)
    sidecar.mkdir()
    (sidecar / so_export.PORTABLE_SIDECAR_MARKER_NAME).write_bytes(
        so_export.PORTABLE_SIDECAR_MARKER_BYTES
    )
    (sidecar / "prior.png").write_bytes(b"prior-texture")

    original_move = so_export._move_checked_entry
    interrupted = False

    def interrupt_after_move(*args, **kwargs):
        nonlocal interrupted
        moved = original_move(*args, **kwargs)
        description = args[5]
        if not interrupted and description == interrupt_description:
            interrupted = True
            raise KeyboardInterrupt("simulated caller-boundary interrupt")
        return moved

    monkeypatch.setattr(so_export, "_move_checked_entry", interrupt_after_move)

    with pytest.raises(KeyboardInterrupt, match="caller-boundary interrupt"):
        so_export.export_stage_portably(
            stage,
            output,
            approved_dependency_roots=[source_dir],
        )

    assert interrupted is True
    assert output.read_bytes() == b"prior-root"
    assert (sidecar / "prior.png").read_bytes() == b"prior-texture"
    assert not list(output_dir.glob(".result_export_*"))


def test_portable_export_commit_rejects_output_directory_without_mutation(
    tmp_path: Path,
) -> None:
    from world_understanding.functions.graphics.so_export import _commit_export

    transaction_dir = tmp_path / "transaction"
    transaction_dir.mkdir()
    transaction_output = transaction_dir / "result.usdc"
    transaction_output.write_bytes(b"new-root")
    transaction_sidecar = transaction_dir / "result.usdc_assets"
    transaction_sidecar.mkdir()
    (transaction_sidecar / "new.png").write_bytes(b"new-texture")

    output = tmp_path / "result.usdc"
    output.mkdir()
    (output / "keep.txt").write_text("must-survive", encoding="utf-8")
    sidecar = tmp_path / "result.usdc_assets"
    sidecar.mkdir()
    (sidecar / ".usd_portable_sidecar").write_bytes(
        b"world-understanding portable USD sidecar v1\n"
    )
    (sidecar / "prior.png").write_bytes(b"prior-texture")

    with pytest.raises(RuntimeError, match="non-file USD output"):
        _commit_export(transaction_output, transaction_sidecar, output, sidecar)

    assert (output / "keep.txt").read_text(encoding="utf-8") == "must-survive"
    assert (sidecar / "prior.png").read_bytes() == b"prior-texture"
    assert transaction_output.read_bytes() == b"new-root"
    assert (transaction_sidecar / "new.png").read_bytes() == b"new-texture"


@_POSIX_DIRFD_ONLY
def test_transaction_fd_path_and_cleanup_preserve_swapped_root(
    tmp_path: Path,
) -> None:
    from world_understanding.functions.graphics import so_export

    parent = tmp_path / "parent"
    parent.mkdir()
    parent_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    transaction_descriptor = -1
    try:
        name, transaction_descriptor, transaction_path = (
            so_export._create_transaction_directory(parent_descriptor, ".txn_")
        )
        original = parent / name
        moved_original = parent / "moved-original"
        original.rename(moved_original)
        replacement = parent / name
        replacement.mkdir()
        (replacement / "must-survive.txt").write_text("replacement", encoding="utf-8")

        (transaction_path / "written-through-fd.txt").write_text(
            "original",
            encoding="utf-8",
        )

        assert (moved_original / "written-through-fd.txt").read_text(
            encoding="utf-8"
        ) == "original"
        assert not (replacement / "written-through-fd.txt").exists()
        assert (
            so_export._cleanup_transaction_directory(
                parent_descriptor,
                name,
                transaction_descriptor,
            )
            is False
        )
        assert (replacement / "must-survive.txt").read_text(encoding="utf-8") == (
            "replacement"
        )
        assert (moved_original / "written-through-fd.txt").exists()
    finally:
        if transaction_descriptor >= 0:
            os.close(transaction_descriptor)
        os.close(parent_descriptor)


def test_portable_export_preserves_transaction_on_output_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from world_understanding.functions.graphics import so_export

    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().Save()

    work = tmp_path / "work"
    work.mkdir()
    output = work / "result.usdc"
    output.write_bytes(b"prevalidated-output")
    displaced = tmp_path / "displaced-output.usdc"
    original_rename = so_export._rename_noreplace
    swapped = False

    def swap_output_before_backup(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal swapped
        if (
            not swapped
            and source_name == output.name
            and destination_name.startswith(".previous_output_")
        ):
            swapped = True
            output.rename(displaced)
            output.write_bytes(b"swapped-output")
            original_rename(
                source_descriptor,
                source_name,
                destination_descriptor,
                destination_name,
            )
            output.write_bytes(b"late-output")
            return
        original_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(so_export, "_rename_noreplace", swap_output_before_backup)

    with pytest.raises(so_export._OutputCommitRaceError, match="preserved"):
        so_export.export_stage_portably(
            stage,
            output,
            approved_dependency_roots=[tmp_path],
        )

    assert swapped is True
    assert displaced.read_bytes() == b"prevalidated-output"
    assert output.read_bytes() == b"late-output"
    transactions = list(work.glob(".result_export_*"))
    assert len(transactions) == 1
    backup = next(transactions[0].glob(".previous_output_*"))
    assert backup.read_bytes() == b"swapped-output"
    assert (transactions[0] / output.name).is_file()


def test_portable_export_preserves_transaction_on_sidecar_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from world_understanding.functions.graphics import so_export

    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().Save()

    work = tmp_path / "work"
    work.mkdir()
    output = work / "result.usdc"
    sidecar = work / so_export.portable_sidecar_name(output)
    sidecar.mkdir()
    (sidecar / so_export.PORTABLE_SIDECAR_MARKER_NAME).write_bytes(
        so_export.PORTABLE_SIDECAR_MARKER_BYTES
    )
    (sidecar / "prevalidated.txt").write_text("owned", encoding="utf-8")
    displaced = tmp_path / "displaced-sidecar"
    original_rename = so_export._rename_noreplace
    swapped = False

    def swap_sidecar_before_backup(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal swapped
        if (
            not swapped
            and source_name == sidecar.name
            and destination_name.startswith(".previous_sidecar_")
        ):
            swapped = True
            sidecar.rename(displaced)
            sidecar.mkdir()
            (sidecar / "swapped.txt").write_text("swapped", encoding="utf-8")
            original_rename(
                source_descriptor,
                source_name,
                destination_descriptor,
                destination_name,
            )
            sidecar.mkdir()
            (sidecar / "late.txt").write_text("late", encoding="utf-8")
            return
        original_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(so_export, "_rename_noreplace", swap_sidecar_before_backup)

    with pytest.raises(so_export._OutputCommitRaceError, match="preserved"):
        so_export.export_stage_portably(
            stage,
            output,
            approved_dependency_roots=[tmp_path],
        )

    assert swapped is True
    assert (displaced / "prevalidated.txt").read_text(encoding="utf-8") == "owned"
    assert (sidecar / "late.txt").read_text(encoding="utf-8") == "late"
    transactions = list(work.glob(".result_export_*"))
    assert len(transactions) == 1
    backup = next(transactions[0].glob(".previous_sidecar_*"))
    assert (backup / "swapped.txt").read_text(encoding="utf-8") == "swapped"
    assert (transactions[0] / output.name).is_file()


@pytest.mark.parametrize("symlink_kind", ["leaf", "parent"])
def test_portable_export_rejects_output_symlink_escape_without_publication(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
        portable_sidecar_name,
    )

    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().Save()

    outside = tmp_path / "outside"
    outside.mkdir()
    requested_parent = tmp_path / "requested"
    if symlink_kind == "leaf":
        requested_parent.mkdir()
        outside_output = outside / "sentinel.usdc"
        outside_output.write_bytes(b"outside-must-survive")
        output = requested_parent / "result.usdc"
        output.symlink_to(outside_output)
        message = "reparse point" if os.name == "nt" else "symlink USD output"
    else:
        outside_output = outside / "result.usdc"
        outside_output.write_bytes(b"outside-must-survive")
        requested_parent.symlink_to(outside, target_is_directory=True)
        output = requested_parent / "result.usdc"
        message = (
            "reparse point" if os.name == "nt" else "symlink or non-directory ancestor"
        )

    with pytest.raises(RuntimeError, match=message):
        export_stage_portably(
            stage,
            output,
            approved_dependency_roots=[tmp_path],
        )

    assert outside_output.read_bytes() == b"outside-must-survive"
    assert requested_parent.is_symlink() is (symlink_kind == "parent")
    if symlink_kind == "leaf":
        assert output.is_symlink()
        assert not (requested_parent / portable_sidecar_name(output)).exists()
    assert not (outside / portable_sidecar_name(output)).exists()


def test_portable_export_preserves_unresolved_bare_mdl_runtime_token(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("mdl", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("OmniPBR.mdl")
    )
    stage.GetRootLayer().Save()

    output = tmp_path / "work" / "result.usdc"
    stale_sidecar = output.parent / "result.usdc_assets"
    stale_sidecar.mkdir(parents=True)
    (stale_sidecar / ".usd_portable_sidecar").write_bytes(
        b"world-understanding portable USD sidecar v1\n"
    )
    (stale_sidecar / "stale.png").write_bytes(b"stale")
    assert export_stage_portably(
        stage,
        output,
        approved_dependency_roots=[tmp_path],
    )
    source.unlink()

    _, asset = _shader_asset(output, "mdl")
    assert asset.path == "OmniPBR.mdl"
    assert not stale_sidecar.exists()


def test_portable_export_discovery_ignores_resolved_bare_mdl_runtime_token(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Ar, Sdf, Usd, UsdShade

    from world_understanding.functions.graphics import so_export

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "OmniPBR.mdl").write_text("mdl 1.7;\n", encoding="utf-8")

    source = source_dir / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("mdl", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("OmniPBR.mdl")
    )
    stage.GetRootLayer().Save()

    context = Ar.DefaultResolverContext([str(runtime_dir)])
    with Ar.ResolverContextBinder(context):
        assert Ar.GetResolver().Resolve("OmniPBR.mdl")
        expanded = so_export._discover_expanded_dependencies(
            stage,
            so_export.is_bare_mdl_token,
            lambda path: so_export._require_approved_dependency(
                path,
                (source_dir.resolve(),),
            ),
        )

    assert expanded == {}


@pytest.mark.parametrize(
    "foreign_kind",
    [
        "file",
        "missing-marker",
        "directory-marker",
        pytest.param(
            "fifo-marker",
            marks=pytest.mark.skipif(
                not hasattr(os, "mkfifo"),
                reason="host has no FIFO filesystem primitive",
            ),
        ),
        "symlink-marker",
        "invalid-marker",
    ],
)
def test_portable_export_refuses_to_replace_unowned_sidecar(
    tmp_path: Path,
    foreign_kind: str,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("mdl", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("OmniPBR.mdl")
    )
    stage.GetRootLayer().Save()

    output = tmp_path / "work" / "result.usdc"
    output.parent.mkdir()
    sidecar = output.parent / "result.usdc_assets"
    if foreign_kind == "file":
        sidecar.write_bytes(b"foreign-sidecar")
    else:
        sidecar.mkdir()
        (sidecar / "keep.txt").write_text("foreign-data", encoding="utf-8")
        marker = sidecar / ".usd_portable_sidecar"
        if foreign_kind == "directory-marker":
            marker.mkdir()
        elif foreign_kind == "fifo-marker":
            os.mkfifo(marker)
        elif foreign_kind == "symlink-marker":
            marker_target = tmp_path / "foreign-marker"
            marker_target.write_bytes(b"world-understanding portable USD sidecar v1\n")
            marker.symlink_to(marker_target)
        elif foreign_kind == "invalid-marker":
            marker.write_bytes(b"not-an-exporter-marker")

    with pytest.raises(RuntimeError, match="non-exporter-owned USD sidecar"):
        export_stage_portably(
            stage,
            output,
            approved_dependency_roots=[tmp_path],
        )

    assert sidecar.exists()
    if sidecar.is_dir():
        assert (sidecar / "keep.txt").read_text(encoding="utf-8") == "foreign-data"
    else:
        assert sidecar.read_bytes() == b"foreign-sidecar"
    assert not output.exists()


def test_portable_export_localizes_all_udim_tiles(tmp_path: Path) -> None:
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade, UsdUtils

    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    source_dir = tmp_path / "source"
    texture_dir = source_dir / "textures"
    texture_dir.mkdir(parents=True)
    (texture_dir / "albedo.1001.png").write_bytes(b"tile-1001")
    (texture_dir / "albedo.1002.png").write_bytes(b"tile-1002")
    source = source_dir / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/albedo.<UDIM>.png")
    )
    stage.GetRootLayer().Save()

    output = tmp_path / "work" / "result.usdc"
    assert export_stage_portably(
        stage,
        output,
        approved_dependency_roots=[source_dir],
    )
    shutil.rmtree(source_dir)

    _, asset = _shader_asset(output)
    assert asset.path.startswith("result.usdc_assets/")
    assert "<UDIM>" in asset.path
    _, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(output)))
    assert unresolved == []
    assert sorted(Path(path).read_bytes() for path in assets) == [
        b"tile-1001",
        b"tile-1002",
    ]


def test_portable_export_accepts_dependencies_across_one_session_root(
    tmp_path: Path,
) -> None:
    """Input and sibling cache directories can share one approved session root."""
    pytest.importorskip("pxr")
    from pxr import Ar, Sdf, Usd, UsdShade

    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    session = tmp_path / "session"
    input_dir = session / "input"
    shared_dir = session / "shared"
    input_dir.mkdir(parents=True)
    shared_dir.mkdir()
    (shared_dir / "albedo.png").write_bytes(b"session-sibling-texture")

    source = input_dir / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../shared/albedo.png")
    )
    stage.GetRootLayer().Save()

    output = session / "cache" / "result.usdc"
    assert export_stage_portably(
        stage,
        output,
        approved_dependency_roots=[session],
    )
    shutil.rmtree(input_dir)
    shutil.rmtree(shared_dir)

    _, asset = _shader_asset(output)
    assert asset.resolvedPath
    resolved = Ar.GetResolver().Resolve(asset.resolvedPath)
    resolver_asset = Ar.GetResolver().OpenAsset(resolved)
    assert resolver_asset is not None
    assert bytes(resolver_asset.GetBuffer()) == b"session-sibling-texture"


def test_portable_export_rejects_dependency_outside_approved_root(
    tmp_path: Path,
) -> None:
    """A readable host file must never become a downloadable sidecar artifact."""
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "host-data.txt"
    secret.write_bytes(b"must-not-publish")

    source = approved / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(secret)))
    stage.GetRootLayer().Save()

    output = tmp_path / "work" / "result.usdc"
    with pytest.raises(RuntimeError, match="outside approved roots"):
        export_stage_portably(
            stage,
            output,
            approved_dependency_roots=[approved],
        )

    assert not output.exists()
    assert not (output.parent / "result.usdc_assets").exists()


def test_portable_export_validates_roots_before_creating_output_tree(
    tmp_path: Path,
) -> None:
    """An invalid trust policy must fail without creating delivery paths."""
    pytest.importorskip("pxr")
    from pxr import Usd

    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    output = tmp_path / "new" / "nested" / "result.usdc"

    with pytest.raises(ValueError, match="must not contain filesystem roots"):
        export_stage_portably(
            Usd.Stage.CreateInMemory(),
            output,
            approved_dependency_roots=[Path(tmp_path.anchor)],
        )

    assert not output.parent.exists()


def test_portable_export_rejects_filesystem_root_approval() -> None:
    from world_understanding.functions.graphics.so_export import (
        _normalize_dependency_roots,
    )

    filesystem_root = Path(Path.cwd().anchor)
    with pytest.raises(ValueError, match="must not contain filesystem roots"):
        _normalize_dependency_roots([filesystem_root])


def test_portable_export_rejects_symlink_escape_from_approved_root(
    tmp_path: Path,
) -> None:
    """An in-root symlink must not authorize its out-of-root target."""
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "host-data.txt"
    secret.write_bytes(b"must-not-publish-through-symlink")
    escaped_link = approved / "apparently-safe.txt"
    escaped_link.symlink_to(secret)

    source = approved / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(escaped_link))
    )
    stage.GetRootLayer().Save()

    output = tmp_path / "work" / "result.usdc"
    with pytest.raises(RuntimeError, match="outside approved roots"):
        export_stage_portably(
            stage,
            output,
            approved_dependency_roots=[approved],
        )

    assert not output.exists()
    assert not (output.parent / "result.usdc_assets").exists()


def test_portable_export_fails_closed_for_unresolved_non_runtime_asset(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("missing.png")
    )
    stage.GetRootLayer().Save()

    output = tmp_path / "work" / "result.usdc"
    with pytest.raises(RuntimeError, match="unresolved asset dependencies"):
        export_stage_portably(
            stage,
            output,
            approved_dependency_roots=[tmp_path],
        )

    assert not output.exists()
    assert not (output.parent / "result.usdc_assets").exists()


def test_portable_export_localizes_unloaded_payload_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Ar, Sdf, Usd, UsdGeom, UsdShade, UsdUtils

    from world_understanding.functions.graphics import so_export

    copied_asset_paths: list[str] = []
    original_copy_resolved_asset = so_export._copy_resolved_asset

    def record_copy_resolved_asset(
        resolver: Any,
        resolved_path: Any,
        destination: Path,
    ) -> None:
        copied_asset_paths.append(so_export._resolved_path_string(resolved_path))
        original_copy_resolved_asset(resolver, resolved_path, destination)

    monkeypatch.setattr(
        so_export,
        "_copy_resolved_asset",
        record_copy_resolved_asset,
    )

    source_dir = tmp_path / "source"
    texture_dir = source_dir / "textures"
    texture_dir.mkdir(parents=True)
    (texture_dir / "albedo.png").write_bytes(b"nested-payload-texture")
    payload = source_dir / "payload.usda"
    payload_stage = Usd.Stage.CreateNew(str(payload))
    payload_root = payload_stage.DefinePrim("/Payload", "Xform")
    UsdGeom.Mesh.Define(payload_stage, "/Payload/Mesh")
    shader = UsdShade.Shader.Define(payload_stage, "/Payload/Shader")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/albedo.png")
    )
    payload_stage.SetDefaultPrim(payload_root)
    payload_stage.GetRootLayer().Save()

    source = source_dir / "source.usda"
    root_stage = Usd.Stage.CreateNew(str(source))
    root_stage.DefinePrim("/World", "Xform")
    for name in ("A", "B"):
        instance = root_stage.DefinePrim(f"/World/{name}", "Xform")
        instance.GetPayloads().AddPayload("payload.usda")
        instance.SetInstanceable(True)
    root_stage.GetRootLayer().Save()

    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadNone)
    assert stage is not None
    assert (
        len(
            [
                layer
                for layer in stage.GetUsedLayers()
                if layer is not stage.GetSessionLayer()
            ]
        )
        == 1
    )

    output = tmp_path / "work" / "result.usdc"
    assert so_export.export_stage_portably(
        stage,
        output,
        approved_dependency_roots=[source_dir],
    )
    shutil.rmtree(source_dir)

    delivery = tmp_path / "delivery"
    delivery.mkdir()
    delivered_output = Path(shutil.move(str(output), delivery / output.name))
    sidecar = output.parent / "result.usdc_assets"
    delivered_sidecar = Path(shutil.move(str(sidecar), delivery / sidecar.name))

    delivered = Usd.Stage.Open(str(delivered_output))
    assert delivered is not None
    assert delivered.GetPrimAtPath("/World/A").IsInstance()
    assert delivered.GetPrimAtPath("/World/B").IsInstance()
    assert delivered.GetPrimAtPath("/World/A/Mesh").IsInstanceProxy()
    assert delivered.GetPrimAtPath("/World/A/Mesh").IsA(UsdGeom.Mesh)
    nested_asset = (
        delivered.GetPrimAtPath("/World/A/Shader").GetAttribute("inputs:file").Get()
    )
    assert nested_asset.resolvedPath
    resolved = Ar.GetResolver().Resolve(nested_asset.resolvedPath)
    resolver_asset = Ar.GetResolver().OpenAsset(resolved)
    assert resolver_asset is not None
    assert bytes(resolver_asset.GetBuffer()) == b"nested-payload-texture"

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
        Sdf.AssetPath(str(delivered_output))
    )
    assert unresolved == []
    assert len(layers) == 2
    assert len(assets) == 1
    assert Path(assets[0]).resolve().is_relative_to(delivered_sidecar.resolve())
    assert all(
        Path(layer.realPath or layer.identifier).resolve() == delivered_output.resolve()
        or Path(layer.realPath or layer.identifier)
        .resolve()
        .is_relative_to(delivered_sidecar.resolve())
        for layer in layers
    )
    assert {
        Path(so_export._outer_asset_identifier(path)).suffix.lower()
        for path in copied_asset_paths
    } == {".png"}


def test_portable_export_fails_closed_for_unresolved_uri(tmp_path: Path) -> None:
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("https://example.invalid/texture.png")
    )
    stage.GetRootLayer().Save()

    output = tmp_path / "work" / "result.usdc"
    with pytest.raises(RuntimeError, match="unresolved asset dependencies"):
        export_stage_portably(
            stage,
            output,
            approved_dependency_roots=[tmp_path],
        )

    assert not output.exists()
    assert not (output.parent / "result.usdc_assets").exists()
