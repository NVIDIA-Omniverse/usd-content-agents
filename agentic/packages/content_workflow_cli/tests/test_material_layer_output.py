# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real USD coverage for material assignment's optional layer output."""

import json
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from content_workflow_cli import runner as mat_runner
from content_workflow_cli.cli import build_parser
from content_workflow_cli.material_layer_output import publish_material_layer
from content_workflow_cli.runner import (
    MaterialAssignConfig,
    _restore_materialized_output,
    _validate_config,
    usd_cli_output_usd_path,
)
from content_workflow_cli.trace import TraceWriter


@pytest.mark.parametrize("shape", ["cube", "nested_sphere", "composed_cube"])
@pytest.mark.parametrize("suffix", ["usda", "usdc"])
@pytest.mark.parametrize("reject_staged", [False, True])
def test_publish_material_layer_sublayers_unchanged_source(
    tmp_path: Path, shape: str, suffix: str, reject_staged: bool
) -> None:
    source = tmp_path / "source.usda"
    geometry = tmp_path / "geometry.usda"
    stage = Usd.Stage.CreateNew(str(geometry if shape == "composed_cube" else source))
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    if shape in {"cube", "composed_cube"}:
        target = "/Asset/Body"
        UsdGeom.Cube.Define(stage, target)
    else:
        UsdGeom.Xform.Define(stage, "/Asset/Assembly")
        target = "/Asset/Assembly/Part"
        UsdGeom.Sphere.Define(stage, target)
    stage.GetRootLayer().Save()
    if shape == "composed_cube":
        source_stage = Usd.Stage.CreateNew(str(source))
        source_stage.GetRootLayer().subLayerPaths.append(geometry.name)
        source_stage.GetRootLayer().defaultPrim = "Asset"
        source_stage.GetRootLayer().Save()
    original_bytes = source.read_bytes()

    library = tmp_path / "library.usda"
    library_stage = Usd.Stage.CreateNew(str(library))
    material_path = "/World/Looks/TestMaterial"
    UsdShade.Material.Define(library_stage, material_path)
    shader = UsdShade.Shader.Define(library_stage, material_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.2, 0.4, 0.6)
    )
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    UsdShade.Material(
        library_stage.GetPrimAtPath(material_path)
    ).CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    library_stage.GetRootLayer().Save()

    output = tmp_path / f"material_layer.{suffix}"
    previous_bytes = b""
    if reject_staged:
        previous_stage = Usd.Stage.CreateNew(str(output))
        UsdGeom.Cube.Define(previous_stage, "/Previous")
        previous_stage.GetRootLayer().Save()
        previous_bytes = output.read_bytes()

    def validate_output(path: Path) -> None:
        mat_runner._validate_materialized_usd_output(path)
        if reject_staged:
            raise RuntimeError("Rejected staged layer")

    def publish() -> None:
        publish_material_layer(
            input_usd_path=source,
            materials_usd_path=library,
            output_usd_path=output,
            assignments=[
                {
                    "coverage_status": "material_assignment",
                    "material_name": "TestMaterial",
                    "material_path": material_path,
                    "source_prim_paths": [target],
                }
            ],
            source_aliases={},
            candidate_path_space="source",
            publish_output=mat_runner._publish_materialized_output,
            validate_output=validate_output,
        )

    if reject_staged:
        with pytest.raises(RuntimeError, match="Rejected staged layer"):
            publish()
        assert output.read_bytes() == previous_bytes
        assert source.read_bytes() == original_bytes
        return
    publish()

    assert source.read_bytes() == original_bytes
    root_layer = Sdf.Layer.FindOrOpen(str(output))
    assert root_layer.GetFileFormat().formatId == suffix
    assert str(source) in root_layer.subLayerPaths
    assert root_layer.GetPrimAtPath(target).specifier == Sdf.SpecifierOver
    assert not root_layer.GetPrimAtPath(target).attributes
    composed = Usd.Stage.Open(str(output))
    material, _ = UsdShade.MaterialBindingAPI(
        composed.GetPrimAtPath(target)
    ).ComputeBoundMaterial()
    assert material and material.GetPath() == Sdf.Path("/Asset/Looks/TestMaterial")


def test_restore_materialized_output_publishes_layer(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.Cube.Define(stage, "/Asset/Body")
    stage.GetRootLayer().Save()
    library = tmp_path / "library.usda"
    library_stage = Usd.Stage.CreateNew(str(library))
    UsdShade.Material.Define(library_stage, "/Looks/Paint")
    library_stage.GetRootLayer().Save()

    run_dir = tmp_path / "run"
    flat_output = usd_cli_output_usd_path(run_dir)
    flat_output.parent.mkdir(parents=True)
    stage.GetRootLayer().Export(str(flat_output))
    (run_dir / "raw").mkdir()
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "assignments": [
                    {
                        "coverage_status": "material_assignment",
                        "material_name": "Paint",
                        "material_path": "/Looks/Paint",
                        "source_prim_paths": ["/Asset/Body"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "raw" / "visible_candidate_prims.json").write_text(
        json.dumps({"path_space": "source", "candidates": []}), encoding="utf-8"
    )
    output = tmp_path / "published.usda"
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=source,
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=library,
        output_usd_path=output,
        layer_only=True,
    )
    result = _restore_materialized_output(
        config=config,
        run_dir=run_dir,
        preflight_packet=None,
        trace_writer=TraceWriter(run_dir),
    )
    assert result.output_path == output
    assert str(source) in Sdf.Layer.FindOrOpen(str(output)).subLayerPaths
    publication = json.loads(
        (run_dir / "raw" / "material_output_publication.json").read_text()
    )
    assert publication["layer_only"] is True


def test_layer_only_requires_output_path(tmp_path: Path) -> None:
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "source.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "library.usda",
        layer_only=True,
    )
    with pytest.raises(ValueError, match="--layer-only requires --output-usd"):
        _validate_config(config)

    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "source.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "library.usda",
        output_usd_path=tmp_path / "layer.usdz",
        layer_only=True,
    )
    with pytest.raises(ValueError, match="must use .usd, .usda, or .usdc"):
        _validate_config(config)


def test_material_assign_parser_accepts_layer_output(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "materials",
            "assign",
            "--usd",
            str(tmp_path / "source.usda"),
            "--materials-yaml",
            str(tmp_path / "materials.yaml"),
            "--output-usd",
            str(tmp_path / "layer.usda"),
            "--layer-only",
        ]
    )
    assert args.layer_only is True
    assert args.output_usd == tmp_path / "layer.usda"
