# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from pxr import Sdf, Usd, UsdShade

from content_agent_workflows.material_assignment import manifest as manifest_module
from content_agent_workflows.material_assignment.manifest import (
    load_material_manifest,
)


def _write_library(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    UsdShade.Material.Define(stage, "/World/Looks/Red")
    UsdShade.Material.Define(stage, "/World/Looks/Metal")
    stage.GetRootLayer().Save()


def _write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(
        yaml.safe_dump({"library_path": "materials.usda", "entries": entries}),
        encoding="utf-8",
    )


def test_manifest_resolves_library_and_preserves_authoritative_fields(
    tmp_path: Path,
) -> None:
    library = tmp_path / "materials.usda"
    manifest_path = tmp_path / "materials.yaml"
    _write_library(library)
    _write_manifest(
        manifest_path,
        [
            {
                "name": "Paint Red",
                "description": "Authoritative red coating",
                "binding": "/World/Looks/Red",
                "tags": ["authored", "red"],
            },
            {
                "name": "Brushed Metal",
                "description": "Brushed aluminum",
                "binding": "/World/Looks/Metal",
            },
        ],
    )

    manifest = load_material_manifest(manifest_path)
    palette = manifest.as_palette()

    assert manifest.library_path == library.resolve()
    assert manifest.by_name["Paint Red"].description == "Authoritative red coating"
    assert manifest.by_name["Paint Red"].tags == ("authored", "red")
    assert manifest.by_name["Brushed Metal"].tags == ("metal", "brushed")
    assert palette["materials"][0]["material_path"] == "/World/Looks/Red"
    assert palette["schema_version"] == "content-agents.material-palette.v1"


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (
            [
                {"name": "Same", "binding": "/World/Looks/Red"},
                {"name": "Same", "binding": "/World/Looks/Metal"},
            ],
            "Duplicate material manifest name",
        ),
        (
            [
                {"name": "First", "binding": "/World/Looks/Red"},
                {"name": "Second", "binding": "/World/Looks/Red"},
            ],
            "Duplicate material manifest binding",
        ),
        (
            [{"name": "Relative", "binding": "World/Looks/Red"}],
            "absolute USD prim path",
        ),
        (
            [{"name": "Missing", "binding": "/World/Looks/DoesNotExist"}],
            "must name UsdShade.Material",
        ),
    ],
)
def test_manifest_rejects_invalid_identity_or_binding(
    tmp_path: Path,
    entries: list[dict[str, object]],
    message: str,
) -> None:
    _write_library(tmp_path / "materials.usda")
    manifest_path = tmp_path / "materials.yaml"
    _write_manifest(manifest_path, entries)

    with pytest.raises(ValueError, match=message):
        load_material_manifest(manifest_path)


def test_manifest_rejects_non_material_prim(tmp_path: Path) -> None:
    library = tmp_path / "materials.usda"
    stage = Usd.Stage.CreateNew(str(library))
    stage.DefinePrim("/World/Looks/NotMaterial", "Scope")
    stage.GetRootLayer().Save()
    manifest_path = tmp_path / "materials.yaml"
    _write_manifest(
        manifest_path,
        [{"name": "Wrong type", "binding": "/World/Looks/NotMaterial"}],
    )

    with pytest.raises(ValueError, match="must name UsdShade.Material"):
        load_material_manifest(manifest_path)


def test_manifest_accepts_legacy_prim_with_authored_material_terminal(
    tmp_path: Path,
) -> None:
    library = tmp_path / "materials.usda"
    stage = Usd.Stage.CreateNew(str(library))
    legacy = stage.DefinePrim("/World/Looks/Cardboard", "Xform")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Cardboard/PreviewSurface")
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    terminal = legacy.CreateAttribute("outputs:surface", Sdf.ValueTypeNames.Token)
    terminal.SetConnections([shader_output.GetAttr().GetPath()])
    stage.GetRootLayer().Save()
    manifest_path = tmp_path / "materials.yaml"
    _write_manifest(
        manifest_path,
        [{"name": "Cardboard", "binding": "/World/Looks/Cardboard"}],
    )

    manifest = load_material_manifest(manifest_path)

    assert manifest.by_name["Cardboard"].binding_path == "/World/Looks/Cardboard"


def test_manifest_rejects_oversized_yaml_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "materials.yaml"
    manifest_path.write_bytes(b"x" * 65)
    monkeypatch.setattr(manifest_module, "MAX_MATERIAL_MANIFEST_BYTES", 64)

    with pytest.raises(ValueError, match="64-byte limit"):
        load_material_manifest(
            manifest_path,
            validate_material_prims=False,
        )
