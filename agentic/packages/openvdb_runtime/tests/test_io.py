# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from openvdb_runtime import (
    ExecutionLimits,
    InvalidGeometryError,
    ResourceLimitError,
    read_all,
    read_grid,
    write,
)


def test_read_grid_uses_official_module(fake_openvdb, tmp_path: Path):
    source = tmp_path / "asset.vdb"
    source.write_bytes(b"vdb")

    grid = read_grid(source, "density", trusted_artifact=True)

    assert grid.label == "read"
    call = next(call for call in fake_openvdb.calls if call[0] == "read")
    assert call[1] == (str(source), "density")


def test_read_all_normalizes_contents(fake_openvdb, tmp_path: Path):
    source = tmp_path / "asset.vdb"
    source.write_bytes(b"vdb")

    contents = read_all(source, trusted_artifact=True)

    assert len(contents.grids) == 1
    assert contents.metadata == {"creator": "test"}


@pytest.mark.parametrize(
    ("metadata", "limits", "message"),
    [
        ({"a": 1, "b": 2}, ExecutionLimits(max_metadata_entries=1), "entry limit"),
        ({"value": "12345678"}, ExecutionLimits(max_metadata_bytes=8), "byte limit"),
    ],
)
def test_read_all_bounds_returned_file_metadata(
    fake_openvdb,
    tmp_path: Path,
    metadata: dict[str, object],
    limits: ExecutionLimits,
    message: str,
) -> None:
    source = tmp_path / "asset.vdb"
    source.write_bytes(b"vdb")
    fake_openvdb.readAll = lambda *_args: ([fake_openvdb.FloatGrid()], metadata)

    with pytest.raises(ResourceLimitError, match=message):
        read_all(source, trusted_artifact=True, limits=limits)


@pytest.mark.parametrize("metadata", [["not", "a", "mapping"], {"value": float("nan")}])
def test_read_all_rejects_malformed_returned_file_metadata(
    fake_openvdb,
    tmp_path: Path,
    metadata: object,
) -> None:
    source = tmp_path / "asset.vdb"
    source.write_bytes(b"vdb")
    fake_openvdb.readAll = lambda *_args: ([fake_openvdb.FloatGrid()], metadata)

    with pytest.raises(InvalidGeometryError, match="file metadata is malformed"):
        read_all(source, trusted_artifact=True)


def test_write_passes_metadata_and_returns_path(fake_openvdb, tmp_path: Path):
    grid = fake_openvdb.FloatGrid()
    destination = tmp_path / "asset.vdb"

    output = write(destination, grid, metadata={"units": "meter"})

    assert output == destination
    assert destination.read_bytes() == b"vdb"
    call = next(call for call in fake_openvdb.calls if call[0] == "write")
    assert call[1][1:] == (grid,)
    assert Path(call[1][0]).parent.parent == tmp_path
    assert call[2] == {"metadata": {"units": "meter"}}


def test_write_uses_private_inode_verified_staging(
    fake_openvdb, monkeypatch, tmp_path: Path
) -> None:
    destination = tmp_path / "asset.vdb"
    observed: dict[str, object] = {}

    def inspect_staging(path, *_args, **_kwargs):
        temporary = Path(path)
        observed["exists"] = temporary.exists()
        observed["regular"] = temporary.is_file()
        observed["parent_mode"] = stat.S_IMODE(temporary.parent.stat().st_mode)
        observed["parent_parent"] = temporary.parent.parent
        temporary.write_bytes(b"vdb")

    monkeypatch.setattr(fake_openvdb, "write", inspect_staging)

    write(destination, fake_openvdb.FloatGrid())

    parent_mode = observed.pop("parent_mode")
    assert isinstance(parent_mode, int)
    assert parent_mode & 0o700 == 0o700
    assert parent_mode & 0o077 == 0
    assert observed == {
        "exists": True,
        "regular": True,
        "parent_parent": tmp_path,
    }


def test_write_rejects_native_replacement_of_secure_staging_inode(
    fake_openvdb, monkeypatch, tmp_path: Path
) -> None:
    destination = tmp_path / "asset.vdb"

    def replace_staging(path, *_args, **_kwargs):
        temporary = Path(path)
        temporary.unlink()
        temporary.write_bytes(b"replacement")

    monkeypatch.setattr(fake_openvdb, "write", replace_staging)

    with pytest.raises(InvalidGeometryError, match="replaced its secure staging file"):
        write(destination, fake_openvdb.FloatGrid())

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_io_rejects_invalid_names_and_byte_paths(fake_openvdb):
    with pytest.raises(ValueError, match="grid_name"):
        read_grid("asset.vdb", "", trusted_artifact=True)
    with pytest.raises(TypeError, match="not bytes"):
        read_all(b"asset.vdb", trusted_artifact=True)


@pytest.mark.parametrize("operation", ["read_grid", "read_all", "write"])
def test_io_enforces_loaded_and_written_grid_limits(fake_openvdb, tmp_path: Path, operation):
    source = tmp_path / "asset.vdb"
    source.write_bytes(b"vdb")
    grid = fake_openvdb.FloatGrid(active_voxels=9)
    fake_openvdb.read = lambda *_args: grid
    fake_openvdb.readAll = lambda *_args: ([grid], {})
    limits = ExecutionLimits(max_active_voxels=8)

    with pytest.raises(ResourceLimitError, match="active voxels"):
        if operation == "read_grid":
            read_grid(source, "density", trusted_artifact=True, limits=limits)
        elif operation == "read_all":
            read_all(source, trusted_artifact=True, limits=limits)
        else:
            write(source, grid, limits=limits)


@pytest.mark.parametrize("operation", ["read_grid", "read_all"])
def test_read_rejects_oversized_file_before_native_load(
    fake_openvdb, tmp_path: Path, operation: str
) -> None:
    source = tmp_path / "oversized.vdb"
    source.write_bytes(b"123456789")
    limits = ExecutionLimits(max_file_bytes=8)

    with pytest.raises(ResourceLimitError, match="VDB input is 9 bytes"):
        if operation == "read_grid":
            read_grid(source, "density", trusted_artifact=True, limits=limits)
        else:
            read_all(source, trusted_artifact=True, limits=limits)

    assert not any(call[0] in {"read", "readAll"} for call in fake_openvdb.calls)


def test_read_rejects_non_regular_input_before_native_load(fake_openvdb, tmp_path: Path) -> None:
    with pytest.raises(InvalidGeometryError, match="regular file"):
        read_all(tmp_path, trusted_artifact=True)

    assert not any(call[0] == "readAll" for call in fake_openvdb.calls)


def test_read_all_enforces_field_count_and_aggregate_voxel_limits(
    fake_openvdb, tmp_path: Path
) -> None:
    source = tmp_path / "fields.vdb"
    source.write_bytes(b"vdb")
    grids = [fake_openvdb.FloatGrid(active_voxels=4) for _ in range(2)]
    fake_openvdb.readAllGridMetadata = lambda *_args: grids
    fake_openvdb.readAll = lambda *_args: (grids, {})

    with pytest.raises(ResourceLimitError, match="2 fields"):
        read_all(source, trusted_artifact=True, limits=ExecutionLimits(max_fields=1))
    with pytest.raises(ResourceLimitError, match="aggregate active voxels"):
        read_all(source, trusted_artifact=True, limits=ExecutionLimits(max_active_voxels=7))


def test_read_preflights_declared_voxels_before_loading_native_tree(
    fake_openvdb, tmp_path: Path
) -> None:
    source = tmp_path / "large.vdb"
    source.write_bytes(b"vdb")
    fake_openvdb.readGridMetadata = lambda *_args: fake_openvdb.FloatGrid(active_voxels=9)

    with pytest.raises(ResourceLimitError, match="9 active voxels"):
        read_grid(
            source,
            "density",
            trusted_artifact=True,
            limits=ExecutionLimits(max_active_voxels=8),
        )

    assert not any(call[0] == "read" for call in fake_openvdb.calls)


def test_read_all_preflights_declared_memory_before_loading_native_trees(
    fake_openvdb, tmp_path: Path
) -> None:
    source = tmp_path / "large.vdb"
    source.write_bytes(b"vdb")
    fake_openvdb.readAllGridMetadata = lambda *_args: [fake_openvdb.FloatGrid(memory_bytes=9)]

    with pytest.raises(ResourceLimitError, match="9 bytes"):
        read_all(
            source,
            trusted_artifact=True,
            limits=ExecutionLimits(max_field_memory_bytes=8),
        )

    assert not any(call[0] == "read_all" for call in fake_openvdb.calls)


def test_read_all_enforces_actual_aggregate_memory_after_loading(
    fake_openvdb, tmp_path: Path
) -> None:
    source = tmp_path / "underreported.vdb"
    source.write_bytes(b"vdb")
    declared = [fake_openvdb.FloatGrid(memory_bytes=1) for _ in range(2)]
    loaded = [fake_openvdb.FloatGrid(memory_bytes=5) for _ in range(2)]
    fake_openvdb.readAllGridMetadata = lambda *_args: declared
    fake_openvdb.readAll = lambda *_args: (loaded, {})

    with pytest.raises(ResourceLimitError, match="10 aggregate bytes"):
        read_all(
            source,
            trusted_artifact=True,
            limits=ExecutionLimits(
                max_field_memory_bytes=8,
                max_total_field_memory_bytes=9,
            ),
        )


def test_write_rejects_field_count_before_inspecting_grids(fake_openvdb) -> None:
    class UntouchedGrid:
        def activeVoxelCount(self) -> int:
            raise AssertionError("write field should not be inspected after count rejection")

    with pytest.raises(ResourceLimitError, match="2 fields"):
        write(
            "asset.vdb",
            [UntouchedGrid(), UntouchedGrid()],
            limits=ExecutionLimits(max_fields=1),
        )

    assert not any(call[0] == "write" for call in fake_openvdb.calls)


def test_write_preflights_per_field_and_aggregate_memory(fake_openvdb, tmp_path: Path) -> None:
    destination = tmp_path / "asset.vdb"

    with pytest.raises(ResourceLimitError, match="9 bytes"):
        write(
            destination,
            fake_openvdb.FloatGrid(memory_bytes=9),
            limits=ExecutionLimits(max_field_memory_bytes=8),
        )
    with pytest.raises(ResourceLimitError, match="aggregate bytes"):
        write(
            destination,
            [
                fake_openvdb.FloatGrid(memory_bytes=5),
                fake_openvdb.FloatGrid(memory_bytes=5),
            ],
            limits=ExecutionLimits(max_total_field_memory_bytes=9),
        )

    assert not any(call[0] == "write" for call in fake_openvdb.calls)


def test_write_rejects_aggregate_active_voxels_before_native_call(
    fake_openvdb, tmp_path: Path
) -> None:
    class SingleInspectionGrid(fake_openvdb.FloatGrid):
        active_voxel_inspections = 0

        def activeVoxelCount(self) -> int:
            self.active_voxel_inspections += 1
            if self.active_voxel_inspections > 1:
                raise AssertionError("active voxel count was inspected more than once")
            return super().activeVoxelCount()

    destination = tmp_path / "asset.vdb"
    grids = [SingleInspectionGrid(active_voxels=4) for _ in range(2)]

    with pytest.raises(ResourceLimitError, match="8 aggregate active voxels"):
        write(
            destination,
            grids,
            limits=ExecutionLimits(max_active_voxels=7),
        )

    assert [grid.active_voxel_inspections for grid in grids] == [1, 1]
    assert not any(call[0] == "write" for call in fake_openvdb.calls)
    assert not destination.exists()


def test_write_inspects_each_grid_memory_once(fake_openvdb, tmp_path: Path) -> None:
    class CountingGrid(fake_openvdb.FloatGrid):
        memory_inspections = 0

        def memUsage(self) -> int:
            self.memory_inspections += 1
            return super().memUsage()

    grid = CountingGrid()

    write(tmp_path / "asset.vdb", grid)

    assert grid.memory_inspections == 1


def test_write_bounds_metadata_before_native_call(fake_openvdb, tmp_path: Path) -> None:
    destination = tmp_path / "asset.vdb"
    grid = fake_openvdb.FloatGrid()

    with pytest.raises(ResourceLimitError, match="entry limit"):
        write(
            destination,
            grid,
            metadata={"a": 1, "b": 2},
            limits=ExecutionLimits(max_metadata_entries=1),
        )
    with pytest.raises(ResourceLimitError, match="byte limit"):
        write(
            destination,
            grid,
            metadata={"value": "12345678"},
            limits=ExecutionLimits(max_metadata_bytes=8),
        )

    assert not any(call[0] == "write" for call in fake_openvdb.calls)


def test_oversized_write_is_atomic_and_preserves_existing_output(
    fake_openvdb, monkeypatch, tmp_path: Path
) -> None:
    destination = tmp_path / "asset.vdb"
    destination.write_bytes(b"existing")

    def oversized_write(path, *_args, **_kwargs):
        Path(path).write_bytes(b"four")

    monkeypatch.setattr(fake_openvdb, "write", oversized_write)

    with pytest.raises(ResourceLimitError, match="VDB output is 4 bytes"):
        write(
            destination,
            fake_openvdb.FloatGrid(),
            limits=ExecutionLimits(max_file_bytes=3),
        )

    assert destination.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [destination]


def test_read_requires_explicit_trusted_artifact_admission(fake_openvdb, tmp_path: Path) -> None:
    source = tmp_path / "unadmitted.vdb"
    source.write_bytes(b"vdb")

    with pytest.raises(TypeError, match="trusted_artifact"):
        read_all(source)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="trusted_artifact=True"):
        read_grid(source, "density", trusted_artifact=False)

    assert not any(call[0] in {"read", "readAll"} for call in fake_openvdb.calls)
