#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a VoMP-format fixture and author its rigid-body mass properties."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from physics_agent.integrations.vomp import apply_vomp_mass_properties


def _write_input_usd(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Object")
    cube = UsdGeom.Cube.Define(stage, "/World/Object/Geometry")
    cube.CreateSizeAttr(1.0)
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.GetRootLayer().Save()


def _write_vomp_npz(path: Path) -> None:
    coordinates = np.asarray(
        [
            (x, y, z)
            for x in (-0.25, 0.25)
            for y in (-0.25, 0.25)
            for z in (-0.25, 0.25)
        ],
        dtype=np.float32,
    )
    dtype = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("youngs_modulus", "<f4"),
        ("poissons_ratio", "<f4"),
        ("density", "<f4"),
        ("segment_id", "<U32"),
    ]
    voxel_data: np.ndarray = np.zeros(len(coordinates), dtype=dtype)
    voxel_data["x"] = coordinates[:, 0]
    voxel_data["y"] = coordinates[:, 1]
    voxel_data["z"] = coordinates[:, 2]
    voxel_data["density"] = np.where(coordinates[:, 0] < 0.0, 800.0, 1200.0)
    voxel_data["youngs_modulus"] = 2.0e9
    voxel_data["poissons_ratio"] = 0.3
    voxel_data["segment_id"] = "voxel_material"
    np.savez_compressed(path, voxel_data=voxel_data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/vomp_rigid_body"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_usd = output_dir / "object.usda"
    vomp_npz = output_dir / "object_materials.npz"
    output_usd = output_dir / "object_vomp_physics.usda"
    _write_input_usd(input_usd)
    _write_vomp_npz(vomp_npz)

    result = apply_vomp_mass_properties(
        input_usd,
        vomp_npz,
        output_usd,
        target_prim_path="/World/Object",
        voxel_size_m=0.5,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
    )
    properties = result.mass_properties
    print(
        "\n".join(
            [
                f"Input USD: {input_usd}",
                f"VoMP NPZ: {vomp_npz}",
                f"Mass: {properties.mass_kg:.6f} kg",
                f"Center of mass (local m): {properties.center_of_mass_local_m}",
                f"Principal inertia (kg m^2): {properties.diagonal_inertia_kg_m2}",
                f"Physics Agent USD: {result.output_usd_path}",
                f"Provenance: {result.provenance_path}",
            ]
        )
    )


if __name__ == "__main__":
    main()
