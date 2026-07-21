# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import texture_agent.functions.material_discovery as md
from texture_agent.functions.material_discovery import MaterialInfo

pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdShade  # noqa: E402


def test_material_discovery_private_coercion_edges() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Steel")
    prim = material.GetPrim()
    prim.CreateAttribute("inputs:unset_color", Sdf.ValueTypeNames.Color3f)
    prim.CreateAttribute("inputs:unset_asset", Sdf.ValueTypeNames.Asset)
    prim.CreateAttribute("inputs:empty_asset", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("@@")
    )
    prim.CreateAttribute("inputs:asset", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/a.png")
    )

    assert md._read_color3f(prim, "inputs:unset_color") is None
    assert md._coerce_color3f(None) is None
    assert md._coerce_color3f("x") is None
    assert md._coerce_float(None) is None
    assert md._coerce_float(object()) is None
    assert md._read_asset_path(prim, "inputs:missing") is None
    assert md._read_asset_path(prim, "inputs:unset_asset") is None
    assert md._read_asset_path(prim, "inputs:empty_asset") is None
    assert md._read_asset_path(prim, "inputs:asset") == "textures/a.png"
    assert md._coerce_texture_path("textures/b.png") == "textures/b.png"


def test_expand_to_prim_units_uses_full_path_when_leaf_names_collide() -> None:
    material = MaterialInfo(
        prim_path="/World/Looks/Steel",
        name="Steel",
        bound_prim_paths=["/World/A/Mesh", "/World/B/Mesh"],
    )

    units = md.expand_to_prim_units(
        [material],
        {"Steel": {"prompt": "brushed steel"}},
        mode="per_prim",
    )

    assert [unit.key for unit in units] == [
        "Steel__World_A_Mesh",
        "Steel__World_B_Mesh",
    ]
