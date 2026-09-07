# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from pxr import Usd, UsdGeom

from geometry_repair import RepairRequest, run_geometry_repair
from geometry_repair.mesh_io import external_mesh_to_usd_stage, load_meshes


def _duplicate_face_obj(path: Path) -> Path:
    path.write_text(
        """o box
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
f 1 3 2
""",
        encoding="utf-8",
    )
    return path


def test_external_mesh_working_copy_applies_explicit_scale(tmp_path: Path) -> None:
    source = _duplicate_face_obj(tmp_path / "source.obj")
    target = external_mesh_to_usd_stage(
        source,
        tmp_path / "working.usda",
        meters_per_unit=0.001,
        up_axis="Z",
    )

    stage = Usd.Stage.Open(str(target))
    assert stage is not None
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert str(UsdGeom.GetStageUpAxis(stage)) == "Z"
    assert stage.GetDefaultPrim().GetCustomDataByKey("geometryRepairSourcePath") == str(
        source.resolve()
    )
    meshes, _metadata = load_meshes(target)
    assert len(meshes) == 1
    assert meshes[0].world_vertices_m.max(axis=0).tolist() == pytest.approx([0.001, 0.001, 0.001])


def test_external_mesh_working_copy_maps_x_up_to_valid_usd_axis(tmp_path: Path) -> None:
    source = _duplicate_face_obj(tmp_path / "source.obj")
    target = external_mesh_to_usd_stage(
        source,
        tmp_path / "working.usda",
        meters_per_unit=1.0,
        up_axis="X",
    )

    stage = Usd.Stage.Open(str(target))
    assert stage is not None
    assert str(UsdGeom.GetStageUpAxis(stage)) == "Z"
    assert stage.GetDefaultPrim().GetCustomDataByKey("geometryRepairSourceUpAxis") == "X"
    meshes, _metadata = load_meshes(target)
    assert len(meshes) == 1
    assert meshes[0].world_vertices_m.min(axis=0).tolist() == pytest.approx([-1.0, 0.0, 0.0])
    assert meshes[0].world_vertices_m.max(axis=0).tolist() == pytest.approx([0.0, 1.0, 1.0])


def test_raw_mesh_repair_requires_frame_then_routes_real_worker(tmp_path: Path) -> None:
    source = _duplicate_face_obj(tmp_path / "source.obj")
    blocked = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "blocked",
            profile="static_environment",
            mode="auto",
            enabled_workers=["trimesh_conservative_cleanup"],
        )
    )
    blocked_plan = json.loads(Path(blocked.repair_plan_path).read_text(encoding="utf-8"))
    assert Path(blocked.normalized_source_path).suffix == ".obj"
    assert [item["worker"] for item in blocked_plan["operations"]] == ["noop"]

    repaired = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repaired",
            profile="static_environment",
            mode="auto",
            enabled_workers=["trimesh_conservative_cleanup"],
            source_meters_per_unit=1.0,
            source_up_axis="Z",
        )
    )
    repaired_plan = json.loads(Path(repaired.repair_plan_path).read_text(encoding="utf-8"))
    assert Path(repaired.normalized_source_path).suffix == ".usd"
    assert [item["worker"] for item in repaired_plan["operations"]] == [
        "noop",
        "trimesh_conservative_cleanup",
    ]
    assert any(
        item.operation.worker == "trimesh_conservative_cleanup" for item in repaired.attempts
    )


@pytest.mark.parametrize("feature", ["animations", "skins"])
def test_external_mesh_normalization_refuses_dynamic_gltf(
    tmp_path: Path,
    feature: str,
) -> None:
    source = tmp_path / "dynamic.gltf"
    source.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "scenes": [{"nodes": []}],
                "scene": 0,
                feature: [{}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=feature):
        external_mesh_to_usd_stage(
            source,
            tmp_path / "working.usda",
            meters_per_unit=1.0,
            up_axis="Y",
        )


def test_external_mesh_normalization_refuses_dynamic_glb(tmp_path: Path) -> None:
    source = tmp_path / "dynamic.glb"
    encoded = json.dumps(
        {"asset": {"version": "2.0"}, "animations": [{}]},
        separators=(",", ":"),
    ).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    total_length = 12 + 8 + len(encoded)
    source.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
    )

    with pytest.raises(ValueError, match="animations"):
        external_mesh_to_usd_stage(
            source,
            tmp_path / "working.usda",
            meters_per_unit=1.0,
            up_axis="Y",
        )
