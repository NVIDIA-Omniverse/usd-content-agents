# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for USD dependency identity runtime-asset exclusions."""

from pathlib import Path

import pytest
from world_understanding.functions.graphics.so_export import (
    is_runtime_resolved_asset_path,
)

from content_agent_workflows.simready.asset_identity import (
    AssetDependencyIdentityError,
    build_asset_dependency_manifest,
)


def test_dependency_manifest_excludes_only_supported_runtime_assets(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr")
    root = tmp_path / "asset.usda"
    root.write_text(
        '#usda 1.0\ndef Xform "World"\n{\n'
        "    custom asset mdl = @OmniPBR.mdl@\n"
        "    custom asset texture = "
        "@omniverse://assets.example.test/textures/albedo.png@\n}\n",
        encoding="utf-8",
    )

    with pytest.raises(AssetDependencyIdentityError, match="OmniPBR.mdl"):
        build_asset_dependency_manifest(root)

    manifest = build_asset_dependency_manifest(
        root,
        is_runtime_asset_path=is_runtime_resolved_asset_path,
    )
    assert [(item["role"], item["path"]) for item in manifest["files"]] == [
        ("root", str(root.resolve()))
    ]

    root.write_text(
        '#usda 1.0\ndef Xform "World"\n{\n'
        "    custom asset mdl = @OmniPBR.mdl@\n"
        "    custom asset missing = @textures/missing.png@\n}\n",
        encoding="utf-8",
    )
    with pytest.raises(
        AssetDependencyIdentityError,
        match="textures/missing.png",
    ) as exc_info:
        build_asset_dependency_manifest(
            root,
            is_runtime_asset_path=is_runtime_resolved_asset_path,
        )
    assert exc_info.value.path_inventory_complete is True
    assert (tmp_path / "textures" / "missing.png").resolve() in (
        exc_info.value.dependency_paths
    )


@pytest.mark.parametrize("arc_kind", ["sublayer", "reference", "payload"])
def test_dependency_manifest_rejects_remote_composition_arcs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arc_kind: str,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Sdf, UsdUtils

    root = tmp_path / "asset.usda"
    layer = Sdf.Layer.CreateNew(str(root))
    remote_layer = "omniverse://assets.example.test/library/body.usda"
    if arc_kind == "sublayer":
        layer.subLayerPaths = [remote_layer]
    else:
        prim = Sdf.CreatePrimInLayer(layer, "/World")
        prim.specifier = Sdf.SpecifierDef
        list_editor = (
            prim.referenceList if arc_kind == "reference" else prim.payloadList
        )
        item = (
            Sdf.Reference(remote_layer)
            if arc_kind == "reference"
            else Sdf.Payload(remote_layer)
        )
        list_editor.prependedItems = [item]
    layer.Save()

    monkeypatch.setattr(
        UsdUtils,
        "ComputeAllDependencies",
        lambda _root: ([layer], [], [remote_layer]),
    )

    with pytest.raises(
        AssetDependencyIdentityError,
        match="Remote USD dependency cannot be content-bound locally",
    ):
        build_asset_dependency_manifest(
            root,
            is_runtime_asset_path=is_runtime_resolved_asset_path,
        )
