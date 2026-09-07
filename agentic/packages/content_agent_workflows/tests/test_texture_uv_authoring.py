# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from world_understanding.utils.usd.package import write_usdz_package_from_directory

import content_agent_workflows.texture.uv_authoring as uv_authoring
from content_agent_workflows.common.artifacts import atomic_write_json
from content_agent_workflows.texture import (
    TextureUvLeafInvocation,
    build_texture_uv_leaf_invocation,
    run_texture_uv_leaf,
)


def _build_stage(
    source: Path,
    *,
    authored_uvs: bool = False,
    indexed_uvs: bool = False,
    inherited_uvs: bool = False,
    time_sampled_uvs: bool = False,
    external_texture: Path | None = None,
) -> None:
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 1.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    if authored_uvs or indexed_uvs or inherited_uvs or time_sampled_uvs:
        owner = root.GetPrim() if inherited_uvs else mesh.GetPrim()
        primvar = UsdGeom.PrimvarsAPI(owner).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            (UsdGeom.Tokens.constant if inherited_uvs else UsdGeom.Tokens.faceVarying),
        )
        values = Vt.Vec2fArray(
            [Gf.Vec2f(0.5, 0.5)]
            if inherited_uvs
            else [
                Gf.Vec2f(0.0, 0.0),
                Gf.Vec2f(1.0, 0.0),
                Gf.Vec2f(1.0, 1.0),
                Gf.Vec2f(0.0, 1.0),
            ]
        )
        if time_sampled_uvs:
            primvar.Set(values, Usd.TimeCode(1.0))
            primvar.Set(values, Usd.TimeCode(2.0))
        else:
            primvar.Set(values)
        if indexed_uvs:
            primvar.Set(Vt.Vec2fArray(values[:2]))
            primvar.SetIndices(Vt.IntArray([0, 1, 1, 0]))
    material = UsdShade.Material.Define(stage, "/World/Looks/Material")
    surface = UsdShade.Shader.Define(stage, "/World/Looks/Material/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    if external_texture is not None:
        texture = UsdShade.Shader.Define(stage, "/World/Looks/Material/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        relative = external_texture.relative_to(source.parent).as_posix()
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(relative)
        )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    assert stage.GetRootLayer().Save()


def _write_uv_invocation(
    source: Path,
    output_dir: Path,
    *,
    policy: str,
) -> Path:
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=output_dir,
        target_prim_paths=("/World/Mesh",),
        policy=policy,  # type: ignore[arg-type]
    )
    path = output_dir / "invocation.json"
    atomic_write_json(path, invocation)
    return path


def _assert_sealed_preflight_rejection(
    result: Any,
    output_dir: Path,
    *,
    message: str,
) -> None:
    assert result.native_disposition == "failed"
    assert result.detail.startswith(
        "Texture UV deterministic authoring preflight rejected: "
    )
    assert message in result.detail
    assert result.error == result.detail
    assert result.output == result.source
    assert not (output_dir / "prepared_texture_uvs.usda").exists()
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "invocation.json",
        "texture_uv_evidence.json",
        "texture_uv_leaf_result.json",
        "texture_uv_saved_stage_readback.json",
    ]


def test_provider_free_uv_authoring_reopens_stage_and_preserves_external_dependency(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "separate-leaf-attempt"
    texture_dir = source_dir / "textures"
    texture_dir.mkdir(parents=True)
    output_dir.mkdir()
    texture_path = texture_dir / "albedo.png"
    texture_path.write_bytes(b"digest-bound-external-texture")
    source = source_dir / "asset.usda"
    _build_stage(source, external_texture=texture_path)

    invocation_path = _write_uv_invocation(
        source,
        output_dir,
        policy="generate_missing",
    )
    result = run_texture_uv_leaf(invocation_path)

    assert result.native_disposition == "passed"
    assert result.provider_invoked is False
    assert result.texture_service_constructed is False
    assert result.vlm_assessor_constructed is False
    assert result.image_generator_constructed is False
    assert result.fixed_pipeline_invoked is False
    assert result.nested_coordinator_invoked is False
    readback = json.loads(
        Path(result.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )
    evidence = json.loads(Path(result.evidence[0].path).read_text(encoding="utf-8"))
    assert "projection" not in json.loads(invocation_path.read_text(encoding="utf-8"))
    assert "projection" not in readback
    assert "projection" not in evidence
    assert readback["authoring_method"] == "bounded_box_fallback"
    assert evidence["authoring_method"] == "bounded_box_fallback"
    assert readback["reopened_saved_stage"] is True
    assert readback["authored_mesh_paths"] == ["/World/Mesh"]
    assert readback["all_uv_ready"] is True
    assert readback["material_identities_unchanged"] is True
    assert readback["external_dependencies_preserved"] is True
    texture_digests = [
        dependency["sha256"]
        for dependency in readback["source_dependencies"]
        if dependency["path"].endswith("albedo.png")
    ]
    assert len(texture_digests) == 1
    assert texture_digests[0] in {
        item["sha256"] for item in readback["saved_stage_dependencies"]
    }
    reopened = Usd.Stage.Open(result.output.path)
    primvar = UsdGeom.PrimvarsAPI(
        reopened.GetPrimAtPath("/World/Mesh")
    ).FindPrimvarWithInheritance("st")
    assert primvar.GetInterpolation() == UsdGeom.Tokens.faceVarying
    assert len(primvar.Get()) == 4


@pytest.mark.parametrize(
    "target",
    ("/", "/World//Mesh", "/../World", "/World.attr", "/World{variant=x}"),
)
def test_uv_invocation_rejects_noncanonical_or_nonprim_targets(
    tmp_path: Path,
    target: str,
) -> None:
    source = tmp_path / "source.usda"
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    _build_stage(source)
    valid = build_texture_uv_leaf_invocation(
        source,
        output_dir=output_dir,
        target_prim_paths=("/World/Mesh",),
    )

    with pytest.raises(ValueError, match="normalized absolute prim paths"):
        TextureUvLeafInvocation.model_validate(
            {
                **valid.model_dump(mode="python"),
                "target_prim_paths": (target,),
            }
        )


def test_uv_authoring_uses_full_face_for_projection_basis(tmp_path: Path) -> None:
    source = tmp_path / "collinear-prefix.usda"
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(2.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    assert stage.GetRootLayer().Save()

    result = run_texture_uv_leaf(
        _write_uv_invocation(
            source,
            output_dir,
            policy="generate_missing",
        )
    )

    assert result.native_disposition == "passed"
    reopened = Usd.Stage.Open(result.output.path)
    values = (
        UsdGeom.PrimvarsAPI(reopened.GetPrimAtPath("/World/Mesh"))
        .GetPrimvar("st")
        .Get()
    )
    assert len({(float(value[0]), float(value[1])) for value in values}) > 2


def test_uv_authoring_uses_mesh_basis_for_degenerate_face(tmp_path: Path) -> None:
    source = tmp_path / "mixed-degenerate.usda"
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    _build_stage(source)
    stage = Usd.Stage.Open(str(source))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Mesh"))
    mesh.GetPointsAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(2.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
    )
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3, 3]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 0, 1, 3]))
    assert stage.GetRootLayer().Save()

    result = run_texture_uv_leaf(
        _write_uv_invocation(source, output_dir, policy="generate_missing")
    )

    assert result.native_disposition == "passed"
    reopened = Usd.Stage.Open(result.output.path)
    values = (
        UsdGeom.PrimvarsAPI(reopened.GetPrimAtPath("/World/Mesh"))
        .GetPrimvar("st")
        .Get()
    )
    assert len(values) == 6
    assert len({(float(value[0]), float(value[1])) for value in values[:3]}) == 3


def test_uv_authoring_rejects_mesh_with_only_degenerate_faces(
    tmp_path: Path,
) -> None:
    source = tmp_path / "all-degenerate.usda"
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    _build_stage(source)
    stage = Usd.Stage.Open(str(source))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Mesh"))
    mesh.GetPointsAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(2.0, 0.0, 0.0),
                Gf.Vec3f(3.0, 0.0, 0.0),
            ]
        )
    )
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3, 3]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 1, 2, 3]))
    assert stage.GetRootLayer().Save()

    result = run_texture_uv_leaf(
        _write_uv_invocation(source, output_dir, policy="generate_missing")
    )

    _assert_sealed_preflight_rejection(
        result,
        output_dir,
        message="no non-collinear projection basis",
    )


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_uv_authoring_rejects_nonfinite_points_during_preflight(
    tmp_path: Path,
    nonfinite: float,
) -> None:
    source = tmp_path / "nonfinite.usda"
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    _build_stage(source)
    stage = Usd.Stage.Open(str(source))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Mesh"))
    mesh.GetPointsAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(nonfinite, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 1.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
    )
    assert stage.GetRootLayer().Save()

    result = run_texture_uv_leaf(
        _write_uv_invocation(
            source,
            output_dir,
            policy="generate_missing",
        )
    )

    _assert_sealed_preflight_rejection(
        result,
        output_dir,
        message="non-finite points",
    )


def test_uv_overlay_preserves_authored_root_timing_metadata(tmp_path: Path) -> None:
    source = tmp_path / "animated-source.usda"
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    _build_stage(source)
    stage = Usd.Stage.Open(str(source))
    root = stage.GetRootLayer()
    root.startTimeCode = 101.0
    root.endTimeCode = 181.0
    root.timeCodesPerSecond = 60.0
    root.framesPerSecond = 30.0
    assert root.Save()

    result = run_texture_uv_leaf(
        _write_uv_invocation(source, output_dir, policy="generate_missing")
    )

    saved = Sdf.Layer.FindOrOpen(result.output.path)
    assert saved is not None
    assert saved.HasStartTimeCode() and saved.startTimeCode == 101.0
    assert saved.HasEndTimeCode() and saved.endTimeCode == 181.0
    assert saved.HasTimeCodesPerSecond() and saved.timeCodesPerSecond == 60.0
    assert saved.HasFramesPerSecond() and saved.framesPerSecond == 30.0


def test_uv_leaf_imports_ascii_usd_through_explicit_usda_snapshot(
    tmp_path: Path,
) -> None:
    usda_source = tmp_path / "source.usda"
    source = tmp_path / "source.usd"
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    _build_stage(usda_source)
    source.write_bytes(usda_source.read_bytes())

    result = run_texture_uv_leaf(
        _write_uv_invocation(source, output_dir, policy="generate_missing")
    )

    assert result.native_disposition == "passed"
    assert result.output.path.endswith("prepared_texture_uvs.usda")


def test_uv_invocation_rejects_non_usd_source_before_opening(
    tmp_path: Path,
) -> None:
    source = tmp_path / "not-usd.png"
    source.write_bytes(b"must-not-be-opened-as-a-layer")
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    message = r"supported USD layer \(\.usd, \.usda, \.usdc, or \.usdz\)"

    with pytest.raises(ValueError, match=message):
        build_texture_uv_leaf_invocation(
            source,
            output_dir=output_dir,
            target_prim_paths=("/World/Mesh",),
            policy="inspect",
        )

    with pytest.raises(ValueError, match=message):
        TextureUvLeafInvocation.model_validate(
            {
                "source": {
                    "path": str(source.resolve()),
                    "sha256": "0" * 64,
                    "size_bytes": source.stat().st_size,
                },
                "source_dependencies": [],
                "output_dir": str(output_dir.resolve()),
                "target_prim_paths": ["/World/Mesh"],
                "policy": "inspect",
            }
        )


@pytest.mark.parametrize(
    ("authored_uvs", "policy", "expects_overlay"),
    [(True, "inspect", False), (False, "generate_missing", True)],
)
def test_uv_leaf_accepts_exact_self_contained_package_predecessor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authored_uvs: bool,
    policy: str,
    expects_overlay: bool,
) -> None:
    package_source = tmp_path / "package-source"
    package_source.mkdir()
    root_layer = package_source / "asset.usda"
    _build_stage(root_layer, authored_uvs=authored_uvs)
    package_stage = Usd.Stage.Open(str(root_layer))
    assert package_stage is not None
    UsdGeom.SetStageUpAxis(package_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(package_stage, 2.5)
    package_stage.GetRootLayer().documentation = "sealed articulation predecessor"
    package_stage.GetRootLayer().framePrecision = 4
    assert package_stage.GetRootLayer().Save()
    source = tmp_path / "articulation-published.usdz"
    write_usdz_package_from_directory(
        package_source,
        Path("asset.usda"),
        source,
    )
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    observed_snapshot_roots: list[Path] = []
    original_create = uv_authoring._HeldPackageSnapshot.create.__func__

    def capture_snapshot_root(
        cls: type[uv_authoring._HeldPackageSnapshot],
        source_binding: Any,
        document: bytes,
    ) -> uv_authoring._HeldPackageSnapshot:
        held = original_create(cls, source_binding, document)
        observed_snapshot_roots.append(held.root)
        return held

    monkeypatch.setattr(
        uv_authoring._HeldPackageSnapshot,
        "create",
        classmethod(capture_snapshot_root),
    )
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=output_dir,
        target_prim_paths=("/World/Mesh",),
        policy=policy,  # type: ignore[arg-type]
    )
    assert invocation.source.path == str(source.resolve())
    assert invocation.source_dependencies == ()
    invocation_path = output_dir / "invocation.json"
    atomic_write_json(invocation_path, invocation)

    result = run_texture_uv_leaf(invocation_path)

    assert result.native_disposition == "passed"
    assert observed_snapshot_roots
    assert all(not path.exists() for path in observed_snapshot_roots)
    reopened = Usd.Stage.Open(result.output.path, load=Usd.Stage.LoadAll)
    assert reopened is not None
    assert UsdGeom.GetStageUpAxis(reopened) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(reopened) == 2.5
    assert reopened.GetRootLayer().documentation == "sealed articulation predecessor"
    assert reopened.GetRootLayer().framePrecision == 4
    if not expects_overlay:
        assert result.output == invocation.source
        assert result.output_dependencies == ()
        return
    assert result.output.path == str(output_dir / "prepared_texture_uvs.usda")
    assert len(result.output_dependencies) == 1
    predecessor = result.output_dependencies[0]
    assert predecessor.path == invocation.source.path
    assert predecessor.sha256 == invocation.source.sha256
    assert predecessor.size_bytes == invocation.source.size_bytes
    saved_layer = Sdf.Layer.FindOrOpen(result.output.path)
    assert saved_layer is not None
    assert saved_layer.subLayerPaths == [invocation.source.path]
    primvar = UsdGeom.PrimvarsAPI(reopened.GetPrimAtPath("/World/Mesh")).GetPrimvar(
        "st"
    )
    assert primvar and len(primvar.Get()) == 4


def test_uv_leaf_rejects_package_with_external_dependency_bindings(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external.usda"
    _build_stage(external, authored_uvs=True)
    package_source = tmp_path / "package-source"
    package_source.mkdir()
    root_layer = package_source / "asset.usda"
    _build_stage(root_layer)
    layer = Sdf.Layer.FindOrOpen(str(root_layer))
    assert layer is not None
    layer.subLayerPaths = [str(external.resolve())]
    assert layer.Save()
    source = tmp_path / "not-self-contained.usdz"
    write_usdz_package_from_directory(
        package_source,
        Path("asset.usda"),
        source,
    )
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    with pytest.raises(ValueError, match="package source must be self-contained"):
        build_texture_uv_leaf_invocation(
            source,
            output_dir=output_dir,
            target_prim_paths=("/World/Mesh",),
            policy="inspect",
        )
    source_binding = uv_authoring._binding(source)
    external_dependencies = tuple(uv_authoring.bind_usd_dependency_closure(source))
    assert tuple(binding.path for binding in external_dependencies) == (
        str(external.resolve()),
    )
    with pytest.raises(ValueError, match="package source must be self-contained"):
        TextureUvLeafInvocation(
            source=source_binding,
            source_dependencies=external_dependencies,
            output_dir=str(output_dir.resolve()),
            target_prim_paths=("/World/Mesh",),
            policy="inspect",
        )

    assert not tuple(output_dir.iterdir())


def test_uv_authoring_missing_reopened_primvar_uses_explicit_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    _build_stage(source)
    invocation_path = _write_uv_invocation(
        source,
        output_dir,
        policy="generate_missing",
    )
    real_open = uv_authoring._stage_from_saved_snapshot

    def remove_reopened_st(document: bytes, **kwargs: object) -> Usd.Stage:
        stage = real_open(document, **kwargs)  # type: ignore[arg-type]
        stage.SetEditTarget(stage.GetRootLayer())
        prim = stage.GetPrimAtPath("/World/Mesh")
        prim.RemoveProperty("primvars:st")
        prim.RemoveProperty("primvars:st:indices")
        assert not UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        return stage

    monkeypatch.setattr(
        uv_authoring,
        "_stage_from_saved_snapshot",
        remove_reopened_st,
    )

    with pytest.raises(ValueError) as failure:
        run_texture_uv_leaf(invocation_path)

    assert str(failure.value) == (
        "Texture UV authored overlay does not contain the exact expected "
        "bounded box fallback values: /World/Mesh"
    )
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "invocation.json",
        "prepared_texture_uvs.usda",
    ]


def test_generate_missing_detects_but_does_not_replace_inherited_primvar(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "leaf"
    source_dir.mkdir()
    output_dir.mkdir()
    source = source_dir / "inherited.usda"
    _build_stage(source, inherited_uvs=True)

    result = run_texture_uv_leaf(
        _write_uv_invocation(source, output_dir, policy="generate_missing")
    )

    readback = json.loads(
        Path(result.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )
    evidence = json.loads(Path(result.evidence[0].path).read_text(encoding="utf-8"))
    assert readback["authored_mesh_paths"] == []
    assert readback["authoring_method"] is None
    assert evidence["authoring_method"] is None
    assert readback["preserved_mesh_paths"] == ["/World/Mesh"]
    assert readback["preserved_uv_identities_unchanged"] is True
    assert readback["mesh_readbacks"][0]["status"] == "repair_required"
    assert readback["mesh_readbacks"][0]["interpolation"] == "constant"
    assert readback["mesh_readbacks"][0]["value_count"] == 1
    assert readback["mesh_readbacks"][0]["primvar_source_prim_path"] == "/World"
    assert result.native_disposition == "failed"
    assert result.detail == (
        "UV authoring refused this saved stage fail closed; saved-stage readback "
        "reasons: /World/Mesh: unsupported UV interpolation constant."
    )
    assert result.error == result.detail
    assert result.output == result.source
    assert not (output_dir / "prepared_texture_uvs.usda").exists()


def test_uv_inspection_preserves_nonpassing_native_disposition(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "leaf"
    source_dir.mkdir()
    output_dir.mkdir()
    source = source_dir / "missing.usda"
    _build_stage(source)

    result = run_texture_uv_leaf(
        _write_uv_invocation(source, output_dir, policy="inspect")
    )

    assert result.native_disposition == "not_evaluated"
    assert result.error is None
    assert result.detail == (
        "UV inspection completed but the selected stage requires authoring."
    )
    assert result.output == result.source
    readback = json.loads(
        Path(result.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )
    evidence = json.loads(Path(result.evidence[0].path).read_text(encoding="utf-8"))
    assert readback["authoring_method"] is None
    assert evidence["authoring_method"] is None
    assert readback["all_uv_ready"] is False
    assert readback["mesh_readbacks"][0]["status"] == "missing"


def test_uv_inspection_reports_repair_required_saved_stage_reasons(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "leaf"
    source_dir.mkdir()
    output_dir.mkdir()
    source = source_dir / "inherited.usda"
    _build_stage(source, inherited_uvs=True)

    result = run_texture_uv_leaf(
        _write_uv_invocation(source, output_dir, policy="inspect")
    )

    assert result.native_disposition == "not_evaluated"
    assert result.error is None
    assert result.detail == (
        "UV inspection completed with non-ready saved-stage readback statuses: "
        "/World/Mesh: repair_required; saved-stage readback reasons: "
        "/World/Mesh: unsupported UV interpolation constant."
    )
    readback = json.loads(
        Path(result.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )
    assert readback["mesh_readbacks"][0]["status"] == "repair_required"
    assert result.output == result.source


def test_generate_missing_preserves_ready_indexed_uvs(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "leaf"
    source_dir.mkdir()
    output_dir.mkdir()
    source = source_dir / "indexed.usda"
    _build_stage(source, indexed_uvs=True)

    result = run_texture_uv_leaf(
        _write_uv_invocation(source, output_dir, policy="generate_missing")
    )

    readback = json.loads(
        Path(result.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )
    evidence = json.loads(Path(result.evidence[0].path).read_text(encoding="utf-8"))
    mesh = readback["mesh_readbacks"][0]
    assert result.native_disposition == "passed"
    assert result.output == result.source
    assert readback["authoring_method"] is None
    assert evidence["authoring_method"] is None
    assert mesh["status"] == "ready"
    assert mesh["indexed"] is True
    assert mesh["value_count"] == 2
    assert mesh["index_count"] == 4
    assert not (output_dir / "prepared_texture_uvs.usda").exists()


def test_generate_missing_does_not_replace_time_sampled_uvs(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "leaf"
    source_dir.mkdir()
    output_dir.mkdir()
    source = source_dir / "animated.usda"
    _build_stage(source, time_sampled_uvs=True)

    result = run_texture_uv_leaf(
        _write_uv_invocation(source, output_dir, policy="generate_missing")
    )

    readback = json.loads(
        Path(result.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )
    mesh = readback["mesh_readbacks"][0]
    assert result.native_disposition == "failed"
    assert result.detail == (
        "UV authoring refused this saved stage fail closed; saved-stage readback "
        "reasons: /World/Mesh: time-sampled primvars:st is inspection-only."
    )
    assert result.error == result.detail
    assert result.output == result.source
    assert mesh["status"] == "unsupported"
    assert mesh["value_time_sample_count"] == 2
    assert not (output_dir / "prepared_texture_uvs.usda").exists()


def test_uv_authoring_preserves_subset_specific_material_identity(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "leaf"
    source_dir.mkdir()
    output_dir.mkdir()
    source = source_dir / "subset.usda"
    _build_stage(source)
    stage = Usd.Stage.Open(str(source))
    subset = UsdGeom.Subset.Define(stage, "/World/Mesh/Subset")
    subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
    subset.CreateIndicesAttr(Vt.IntArray([0]))
    subset_material = UsdShade.Material.Define(
        stage,
        "/World/Looks/SubsetMaterial",
    )
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(subset_material)
    assert stage.GetRootLayer().Save()
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=output_dir,
        target_prim_paths=("/World/Mesh/Subset",),
        policy="generate_missing",
    )
    invocation_path = output_dir / "invocation.json"
    atomic_write_json(invocation_path, invocation)

    result = run_texture_uv_leaf(invocation_path)

    readback = json.loads(
        Path(result.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )
    assert result.native_disposition == "passed"
    assert readback["authored_mesh_paths"] == ["/World/Mesh"]
    assert readback["material_identities_unchanged"] is True
    assert readback["scope_readbacks"] == [
        {
            "binding_relationship_path": "/World/Mesh/Subset.material:binding",
            "effective_material_path": "/World/Looks/SubsetMaterial",
            "material_identity_sha256": readback["scope_readbacks"][0][
                "material_identity_sha256"
            ],
            "material_purpose": "allPurpose",
            "requested_prim_path": "/World/Mesh/Subset",
            "scope_identity_sha256": readback["scope_readbacks"][0][
                "scope_identity_sha256"
            ],
            "scope_prim_path": "/World/Mesh/Subset",
            "target_kind": "subset",
            "uv_mesh_path": "/World/Mesh",
        }
    ]


def test_uv_leaf_refuses_artifact_collision_before_authoring(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "leaf"
    source_dir.mkdir()
    output_dir.mkdir()
    source = source_dir / "missing.usda"
    _build_stage(source)
    invocation_path = _write_uv_invocation(
        source,
        output_dir,
        policy="generate_missing",
    )
    collision = output_dir / "texture_uv_evidence.json"
    collision.write_bytes(b"preserve-me")

    with pytest.raises(FileExistsError, match="refuses to replace artifact"):
        run_texture_uv_leaf(invocation_path)

    assert collision.read_bytes() == b"preserve-me"
    assert not (output_dir / "prepared_texture_uvs.usda").exists()


def test_uv_leaf_detects_source_change_during_stage_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "leaf"
    source_dir.mkdir()
    output_dir.mkdir()
    source = source_dir / "ready.usda"
    _build_stage(source, authored_uvs=True)
    invocation_path = _write_uv_invocation(source, output_dir, policy="inspect")
    original_scope_readbacks = uv_authoring._scope_readbacks
    changed = False

    def mutate_after_readback(
        stage: Usd.Stage,
        scope: tuple[tuple[str, str, str, object], ...],
    ) -> object:
        nonlocal changed
        readbacks = original_scope_readbacks(stage, scope)  # type: ignore[arg-type]
        if not changed:
            source.write_text(
                source.read_text(encoding="utf-8") + "\n# concurrent change\n",
                encoding="utf-8",
            )
            changed = True
        return readbacks

    monkeypatch.setattr(uv_authoring, "_scope_readbacks", mutate_after_readback)

    with pytest.raises(ValueError, match="source identity changed"):
        run_texture_uv_leaf(invocation_path)

    assert tuple(output_dir.iterdir()) == (invocation_path,)


@pytest.mark.parametrize("swap_target", ["source", "dependency"])
def test_uv_leaf_consumes_frozen_bytes_across_path_swap_and_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_target: str,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "leaf"
    source_dir.mkdir()
    output_dir.mkdir()
    dependency = source_dir / "mesh.usda"
    _build_stage(dependency)
    source = source_dir / "root.usda"
    root_layer = Sdf.Layer.CreateNew(str(source))
    root_layer.subLayerPaths = [dependency.name]
    assert root_layer.Save()
    invocation_path = _write_uv_invocation(
        source,
        output_dir,
        policy="generate_missing",
    )

    target = source if swap_target == "source" else dependency
    competitor = source_dir / f"competitor-{swap_target}.usda"
    _build_stage(competitor, authored_uvs=True)
    competitor_stage = Usd.Stage.Open(str(competitor))
    UsdGeom.Xform.Define(competitor_stage, "/Injected")
    assert competitor_stage.GetRootLayer().Save()
    held_original = source_dir / f"held-{swap_target}.usda"
    original_open_stage = uv_authoring._FrozenUsdSnapshot.open_stage
    swapped = False

    def swap_during_open(
        snapshot: uv_authoring._FrozenUsdSnapshot,
    ) -> Usd.Stage:
        nonlocal swapped
        if swapped:
            return original_open_stage(snapshot)
        swapped = True
        target.rename(held_original)
        competitor.rename(target)
        try:
            return original_open_stage(snapshot)
        finally:
            target.rename(competitor)
            held_original.rename(target)

    monkeypatch.setattr(
        uv_authoring._FrozenUsdSnapshot,
        "open_stage",
        swap_during_open,
    )

    result = run_texture_uv_leaf(invocation_path)

    assert swapped is True
    assert result.native_disposition == "passed"
    assert result.output != result.source
    reopened = Usd.Stage.Open(result.output.path)
    assert not reopened.GetPrimAtPath("/Injected")
    assert UsdGeom.PrimvarsAPI(reopened.GetPrimAtPath("/World/Mesh")).GetPrimvar("st")


def test_uv_leaf_rejects_binary_private_snapshot_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usdc"
    competitor = tmp_path / "competitor.usdc"
    _build_stage(source)
    _build_stage(competitor, authored_uvs=True)
    competitor_stage = Usd.Stage.Open(str(competitor))
    UsdGeom.Xform.Define(competitor_stage, "/Injected")
    assert competitor_stage.GetRootLayer().Save()
    replacement_bytes = competitor.read_bytes()
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    invocation_path = _write_uv_invocation(
        source,
        output_dir,
        policy="generate_missing",
    )
    real_open = uv_authoring.Sdf.Layer.OpenAsAnonymous
    substituted = False

    def substitute_snapshot(path: str) -> Sdf.Layer:
        nonlocal substituted
        snapshot_path = Path(path)
        if substituted or snapshot_path.suffix != ".usdc":
            return real_open(path)
        substituted = True
        held_path = snapshot_path.with_name("held-source.usdc")
        snapshot_path.replace(held_path)
        snapshot_path.write_bytes(replacement_bytes)
        try:
            return real_open(path)
        finally:
            snapshot_path.unlink()
            held_path.replace(snapshot_path)

    monkeypatch.setattr(
        uv_authoring.Sdf.Layer,
        "OpenAsAnonymous",
        substitute_snapshot,
    )

    with pytest.raises(ValueError, match="namespace or bytes changed"):
        run_texture_uv_leaf(invocation_path)

    assert substituted is True
    assert tuple(output_dir.iterdir()) == (invocation_path,)


def test_uv_leaf_rejects_binary_snapshot_layer_content_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usdc"
    competitor = tmp_path / "competitor.usdc"
    _build_stage(source)
    _build_stage(competitor, authored_uvs=True)
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    invocation_path = _write_uv_invocation(
        source, output_dir, policy="generate_missing"
    )
    real_open = uv_authoring.Sdf.Layer.OpenAsAnonymous
    mismatched = False

    def mismatched_open(path: str) -> Sdf.Layer:
        nonlocal mismatched
        if not mismatched and Path(path).suffix == ".usdc":
            mismatched = True
            return real_open(str(competitor))
        return real_open(path)

    monkeypatch.setattr(uv_authoring.Sdf.Layer, "OpenAsAnonymous", mismatched_open)

    with pytest.raises(ValueError, match="namespace or bytes changed"):
        run_texture_uv_leaf(invocation_path)

    assert mismatched is True


def test_binary_snapshot_write_failure_closes_directory_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = uv_authoring.os.open
    directory_descriptors: list[int] = []

    def tracked_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == tmp_path:
            directory_descriptors.append(descriptor)
        return descriptor

    def fail_snapshot_write(_path: Path, _document: bytes) -> int:
        raise OSError("snapshot write failed")

    monkeypatch.setattr(uv_authoring.os, "open", tracked_open)
    monkeypatch.setattr(
        uv_authoring,
        "_write_private_snapshot",
        fail_snapshot_write,
    )

    with pytest.raises(OSError, match="snapshot write failed"):
        uv_authoring._open_binary_snapshot_layer(
            snapshot_root=tmp_path,
            snapshot_path=tmp_path / "snapshot.usdc",
            original_path=tmp_path / "source.usdc",
            document=b"PXR-USDC\x00unread",
        )

    assert len(directory_descriptors) == 1
    with pytest.raises(OSError):
        uv_authoring.os.fstat(directory_descriptors[0])


def test_held_directory_open_closes_descriptor_when_fstat_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = uv_authoring.os.open
    real_fstat = uv_authoring.os.fstat
    opened: list[int] = []

    def tracked_open(*args: Any, **kwargs: Any) -> int:
        descriptor = real_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def fail_fstat(descriptor: int) -> os.stat_result:
        if descriptor in opened:
            raise OSError("fstat failed")
        return real_fstat(descriptor)

    monkeypatch.setattr(uv_authoring.os, "open", tracked_open)
    monkeypatch.setattr(uv_authoring.os, "fstat", fail_fstat)

    with pytest.raises(OSError, match="fstat failed"):
        uv_authoring._HeldDirectory.open(tmp_path)

    assert len(opened) == 1
    with pytest.raises(OSError):
        real_fstat(opened[0])


def test_held_directory_cleanup_preserves_primary_failure(tmp_path: Path) -> None:
    held = uv_authoring._HeldDirectory.open(tmp_path)
    descriptor = held.descriptor

    def fail_cleanup() -> None:
        raise RuntimeError("secondary cleanup failure")

    with pytest.raises(ValueError, match="primary operation failure") as raised:
        with held as root:
            root.retain_cleanup(fail_cleanup)
            raise ValueError("primary operation failure")

    assert isinstance(
        getattr(raised.value, "texture_uv_cleanup_error", None),
        RuntimeError,
    )
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_held_directory_claim_closes_transferred_descriptor_on_verify_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = uv_authoring.os.open
    real_fstat = uv_authoring.os.fstat

    with uv_authoring._HeldDirectory.open(tmp_path) as root:
        child_descriptors: list[int] = []

        def tracked_open(*args: Any, **kwargs: Any) -> int:
            descriptor = real_open(*args, **kwargs)
            if kwargs.get("dir_fd") == root.descriptor:
                child_descriptors.append(descriptor)
            return descriptor

        def fail_child_verify(held: uv_authoring._HeldDirectory) -> None:
            if held.path.name == "child":
                raise ValueError("child identity changed")

        monkeypatch.setattr(uv_authoring.os, "open", tracked_open)
        monkeypatch.setattr(
            uv_authoring._HeldDirectory,
            "verify_path",
            fail_child_verify,
        )

        with pytest.raises(ValueError, match="child identity changed"):
            root.claim_directory("child")

        assert len(child_descriptors) == 1
        with pytest.raises(OSError):
            real_fstat(child_descriptors[0])


@pytest.mark.parametrize("operation", ["open_read", "write_bytes"])
def test_held_directory_file_operation_closes_descriptor_when_fstat_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    real_open = uv_authoring.os.open
    real_fstat = uv_authoring.os.fstat
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")

    with uv_authoring._HeldDirectory.open(tmp_path) as root:
        file_descriptors: list[int] = []

        def tracked_open(*args: Any, **kwargs: Any) -> int:
            descriptor = real_open(*args, **kwargs)
            if kwargs.get("dir_fd") == root.descriptor:
                file_descriptors.append(descriptor)
            return descriptor

        def fail_file_fstat(descriptor: int) -> os.stat_result:
            if descriptor in file_descriptors:
                raise OSError("file fstat failed")
            return real_fstat(descriptor)

        monkeypatch.setattr(uv_authoring.os, "open", tracked_open)
        monkeypatch.setattr(uv_authoring.os, "fstat", fail_file_fstat)

        with pytest.raises(OSError, match="file fstat failed"):
            if operation == "open_read":
                root.open_read(source.name)
            else:
                root.write_bytes("output.bin", b"output")

        assert len(file_descriptors) == 1
        with pytest.raises(OSError):
            real_fstat(file_descriptors[0])


def test_binary_snapshot_layers_detach_from_private_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "leaf"
    source_dir.mkdir()
    output_dir.mkdir()
    dependency = source_dir / "mesh.usdc"
    _build_stage(dependency)
    source = source_dir / "root.usdc"
    root_layer = Sdf.Layer.CreateNew(str(source))
    root_layer.subLayerPaths = [dependency.name]
    assert root_layer.Save()
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=output_dir,
        target_prim_paths=("/World/Mesh",),
        policy="inspect",
    )

    snapshot = uv_authoring._FrozenUsdSnapshot.open(
        invocation.source,
        invocation.source_dependencies,
    )

    assert all(layer.anonymous for layer in snapshot.layers)
    assert all(not layer.realPath for layer in snapshot.layers)
    assert all(layer.GetFileFormat().formatId == "usda" for layer in snapshot.layers)
    reopened = snapshot.open_stage()
    assert reopened.GetPrimAtPath("/World/Mesh").IsA(UsdGeom.Mesh)


@pytest.mark.parametrize("arc_kind", ["reference", "payload"])
def test_internal_composition_arc_survives_snapshot_and_uv_authoring(
    tmp_path: Path,
    arc_kind: str,
) -> None:
    source = tmp_path / f"internal-{arc_kind}.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.Xform.Define(stage, "/Template")
    mesh = UsdGeom.Mesh.Define(stage, "/Template/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    UsdGeom.Xform.Define(stage, "/World")
    asset = UsdGeom.Xform.Define(stage, "/World/Asset").GetPrim()
    if arc_kind == "reference":
        assert asset.GetReferences().AddInternalReference("/Template")
    else:
        assert asset.GetPayloads().AddInternalPayload("/Template")
    assert stage.GetRootLayer().Save()
    assert tuple(stage.GetRootLayer().GetCompositionAssetDependencies()) == ("",)

    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=output_dir,
        target_prim_paths=("/World/Asset/Mesh",),
        policy="generate_missing",
    )
    snapshot = uv_authoring._FrozenUsdSnapshot.open(
        invocation.source,
        invocation.source_dependencies,
    )
    frozen_stage = snapshot.open_stage()
    assert frozen_stage.GetPrimAtPath("/World/Asset/Mesh").IsA(UsdGeom.Mesh)

    invocation_path = output_dir / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    result = run_texture_uv_leaf(invocation_path)

    assert result.native_disposition == "passed"
    readback = json.loads(
        Path(result.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )
    assert readback["authored_mesh_paths"] == ["/World/Asset/Mesh"]
    assert readback["all_uv_ready"] is True
    reopened = Usd.Stage.Open(result.output.path, load=Usd.Stage.LoadAll)
    reopened_asset = reopened.GetPrimAtPath("/World/Asset")
    if arc_kind == "reference":
        assert reopened_asset.HasAuthoredReferences()
    else:
        assert reopened_asset.HasAuthoredPayloads()
    primvar = UsdGeom.PrimvarsAPI(
        reopened.GetPrimAtPath("/World/Asset/Mesh")
    ).GetPrimvar("st")
    assert primvar
    assert len(primvar.Get()) == 3


def test_snapshot_rejects_unbound_external_composition_dependency(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "dependency.usda"
    _build_stage(dependency)
    source = tmp_path / "source.usda"
    layer = Sdf.Layer.CreateNew(str(source))
    layer.subLayerPaths = [dependency.name]
    assert layer.Save()

    source_binding = uv_authoring._binding(source)
    with pytest.raises(ValueError, match="missing composition dependency"):
        uv_authoring._FrozenUsdSnapshot.open(source_binding, ())


@pytest.mark.parametrize(
    "authored_path",
    ["archive.usdz[layer.usda]", "[layer.usda]"],
)
def test_snapshot_rejects_package_relative_composition_dependency(
    tmp_path: Path,
    authored_path: str,
) -> None:
    source = tmp_path / "source.usda"
    layer = Sdf.Layer.CreateNew(str(source))
    layer.subLayerPaths = [authored_path]
    assert layer.Save()

    source_binding = uv_authoring._binding(source)
    with pytest.raises(ValueError, match="package-relative composition paths"):
        uv_authoring._FrozenUsdSnapshot.open(source_binding, ())


def test_reused_binary_dependency_rebinds_every_composition_occurrence(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "dependency.usdc"
    dependency_stage = Usd.Stage.CreateNew(str(dependency))
    UsdGeom.Xform.Define(dependency_stage, "/Reusable")
    dependency_mesh = UsdGeom.Mesh.Define(
        dependency_stage,
        "/Reusable/Mesh",
    )
    dependency_mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
    )
    dependency_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    dependency_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    assert dependency_stage.GetRootLayer().Save()

    source = tmp_path / "source.usdc"
    source_stage = Usd.Stage.CreateNew(str(source))
    source_stage.GetRootLayer().subLayerPaths = [dependency.name]
    UsdGeom.Xform.Define(source_stage, "/World")
    referenced = UsdGeom.Xform.Define(
        source_stage,
        "/World/Referenced",
    ).GetPrim()
    referenced.GetReferences().AddReference(dependency.name, "/Reusable")
    assert source_stage.GetRootLayer().Save()

    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    expected_meshes = ("/Reusable/Mesh", "/World/Referenced/Mesh")
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=output_dir,
        target_prim_paths=expected_meshes,
        policy="generate_missing",
    )
    snapshot = uv_authoring._FrozenUsdSnapshot.open(
        invocation.source,
        invocation.source_dependencies,
    )

    frozen_stage = snapshot.open_stage()
    for path in expected_meshes:
        frozen_prim = frozen_stage.GetPrimAtPath(path)
        assert frozen_prim, path
        assert frozen_prim.IsA(UsdGeom.Mesh)

    invocation_path = output_dir / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    result = run_texture_uv_leaf(invocation_path)

    readback = json.loads(
        Path(result.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )
    assert result.native_disposition == "passed"
    assert tuple(readback["authored_mesh_paths"]) == expected_meshes
    assert readback["all_uv_ready"] is True
    reopened = Usd.Stage.Open(result.output.path)
    for path in expected_meshes:
        primvar = UsdGeom.PrimvarsAPI(reopened.GetPrimAtPath(path)).GetPrimvar("st")
        assert primvar
        assert len(primvar.Get()) == 3


@pytest.mark.parametrize(
    ("reports_progress", "message"),
    [(False, "made no progress"), (True, "retained residual")],
)
def test_composition_dependency_rebind_fails_closed_when_incomplete(
    tmp_path: Path,
    reports_progress: bool,
    message: str,
) -> None:
    class IncompleteLayer:
        def __init__(self) -> None:
            self.document = "layer"

        def GetCompositionAssetDependencies(self) -> tuple[str, ...]:
            return ("dependency.usdc",)

        def UpdateCompositionAssetDependency(
            self,
            _authored_path: str,
            _target_identifier: str,
        ) -> bool:
            if reports_progress:
                self.document += "!"
            return True

        def ExportToString(self) -> str:
            return self.document

    layer = cast(Any, IncompleteLayer())

    with pytest.raises(ValueError, match=message):
        uv_authoring._rebind_all_composition_dependency_occurrences(
            layer,
            owner_path=tmp_path / "source.usdc",
            authored_path="dependency.usdc",
            target_identifier="anon:dependency",
        )


def _build_instance_stage(
    source: Path,
    prototype_source: Path,
    *,
    authored_uvs: bool,
) -> None:
    prototype_stage = Usd.Stage.CreateNew(str(prototype_source))
    UsdGeom.Xform.Define(prototype_stage, "/Prototype")
    mesh = UsdGeom.Mesh.Define(prototype_stage, "/Prototype/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    if authored_uvs:
        primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        )
        primvar.Set(
            Vt.Vec2fArray(
                [
                    Gf.Vec2f(0.0, 0.0),
                    Gf.Vec2f(1.0, 0.0),
                    Gf.Vec2f(0.0, 1.0),
                ]
            )
        )
    assert prototype_stage.GetRootLayer().Save()
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.Xform.Define(stage, "/World")
    instance = UsdGeom.Xform.Define(stage, "/World/Instance").GetPrim()
    instance.GetReferences().AddReference(prototype_source.name, "/Prototype")
    instance.SetInstanceable(True)
    assert stage.GetRootLayer().Save()


@pytest.mark.parametrize("target_path", ["/World/Instance/Mesh", "/World"])
@pytest.mark.parametrize("policy", ["inspect", "generate_missing"])
def test_uv_leaf_reads_ready_instance_proxy_scope_without_authoring(
    tmp_path: Path,
    target_path: str,
    policy: str,
) -> None:
    prototype_source = tmp_path / "prototype.usda"
    source = tmp_path / "instance.usda"
    _build_instance_stage(source, prototype_source, authored_uvs=True)
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=output_dir,
        target_prim_paths=(target_path,),
        policy=policy,  # type: ignore[arg-type]
    )
    invocation_path = output_dir / "invocation.json"
    atomic_write_json(invocation_path, invocation)

    result = run_texture_uv_leaf(invocation_path)

    readback = json.loads(
        Path(result.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )
    evidence = json.loads(Path(result.evidence[0].path).read_text(encoding="utf-8"))
    assert result.native_disposition == "passed"
    assert result.output == result.source
    assert readback["resolved_mesh_paths"] == ["/World/Instance/Mesh"]
    assert readback["mesh_readbacks"][0]["status"] == "ready"
    assert readback["authored_mesh_paths"] == []
    assert readback["authoring_method"] is None
    assert evidence["authoring_method"] is None
    assert not (output_dir / "prepared_texture_uvs.usda").exists()


@pytest.mark.parametrize("target_path", ["/World/Instance/Mesh", "/World"])
def test_uv_leaf_rejects_missing_instance_proxy_before_authoring(
    tmp_path: Path,
    target_path: str,
) -> None:
    prototype_source = tmp_path / "prototype.usda"
    source = tmp_path / "instance.usda"
    _build_instance_stage(source, prototype_source, authored_uvs=False)
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=output_dir,
        target_prim_paths=(target_path,),
        policy="generate_missing",
    )
    invocation_path = output_dir / "invocation.json"
    atomic_write_json(invocation_path, invocation)

    result = run_texture_uv_leaf(invocation_path)

    _assert_sealed_preflight_rejection(
        result,
        output_dir,
        message="read-only instance proxy",
    )


@pytest.mark.parametrize("face_vertex_count", [2, 0, -1])
def test_uv_leaf_rejects_invalid_face_counts_before_any_authoring(
    tmp_path: Path,
    face_vertex_count: int,
) -> None:
    source = tmp_path / "invalid-face.usda"
    stage = Usd.Stage.CreateNew(str(source))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([face_vertex_count]))
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray(list(range(max(face_vertex_count, 0))))
    )
    assert stage.GetRootLayer().Save()
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    invocation_path = _write_uv_invocation(
        source,
        output_dir,
        policy="generate_missing",
    )

    result = run_texture_uv_leaf(invocation_path)

    _assert_sealed_preflight_rejection(
        result,
        output_dir,
        message="invalid topology",
    )


def test_uv_leaf_does_not_clobber_or_unlink_concurrent_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "leaf"
    source_dir.mkdir()
    output_dir.mkdir()
    source = source_dir / "missing.usda"
    _build_stage(source)
    invocation_path = _write_uv_invocation(
        source,
        output_dir,
        policy="generate_missing",
    )
    concurrent = output_dir / "texture_uv_evidence.json"
    original_build_readback = uv_authoring._build_readback

    def race_after_precheck(**kwargs: object) -> object:
        readback = original_build_readback(**kwargs)  # type: ignore[arg-type]
        concurrent.write_bytes(b"concurrent-owner")
        return readback

    monkeypatch.setattr(uv_authoring, "_build_readback", race_after_precheck)

    with pytest.raises(FileExistsError):
        run_texture_uv_leaf(invocation_path)

    assert concurrent.read_bytes() == b"concurrent-owner"
    assert (output_dir / "prepared_texture_uvs.usda").is_file()
    assert (output_dir / "texture_uv_saved_stage_readback.json").is_file()
    assert not (output_dir / "texture_uv_leaf_result.json").exists()


def test_uv_leaf_does_not_clobber_concurrent_saved_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "leaf"
    source_dir.mkdir()
    output_dir.mkdir()
    source = source_dir / "missing.usda"
    _build_stage(source)
    invocation_path = _write_uv_invocation(
        source,
        output_dir,
        policy="generate_missing",
    )
    concurrent = output_dir / "prepared_texture_uvs.usda"
    original_write = uv_authoring._HeldDirectory.write_bytes

    def race_saved_stage(
        held: uv_authoring._HeldDirectory,
        name: str,
        document: bytes,
    ) -> tuple[int, int]:
        if name == concurrent.name:
            concurrent.write_bytes(b"concurrent-saved-stage")
        return original_write(held, name, document)

    monkeypatch.setattr(
        uv_authoring._HeldDirectory,
        "write_bytes",
        race_saved_stage,
    )

    with pytest.raises(FileExistsError):
        run_texture_uv_leaf(invocation_path)

    assert concurrent.read_bytes() == b"concurrent-saved-stage"
    assert not (output_dir / "texture_uv_saved_stage_readback.json").exists()
    assert not (output_dir / "texture_uv_evidence.json").exists()
    assert not (output_dir / "texture_uv_leaf_result.json").exists()


def test_material_identity_includes_connected_shader_outside_material_tree(
    tmp_path: Path,
) -> None:
    source = tmp_path / "shared-network.usda"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first_output.mkdir()
    second_output.mkdir()
    _build_stage(source, authored_uvs=True)
    stage = Usd.Stage.Open(str(source))
    material = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/Material"))
    shared = UsdShade.Shader.Define(stage, "/World/Shared")
    shared.CreateIdAttr("UsdPreviewSurface")
    shared.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
    shared.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.GetSurfaceOutput().ConnectToSource(shared.ConnectableAPI(), "surface")
    assert stage.GetRootLayer().Save()

    first = run_texture_uv_leaf(
        _write_uv_invocation(source, first_output, policy="inspect")
    )
    first_readback = json.loads(
        Path(first.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )
    stage = Usd.Stage.Open(str(source))
    UsdShade.Shader(stage.GetPrimAtPath("/World/Shared")).GetInput("roughness").Set(0.8)
    assert stage.GetRootLayer().Save()
    second = run_texture_uv_leaf(
        _write_uv_invocation(source, second_output, policy="inspect")
    )
    second_readback = json.loads(
        Path(second.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )

    assert (
        first_readback["scope_readbacks"][0]["material_identity_sha256"]
        != second_readback["scope_readbacks"][0]["material_identity_sha256"]
    )


def test_source_overlay_preserves_unrelated_variants(tmp_path: Path) -> None:
    source = tmp_path / "variant-source.usda"
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    _build_stage(source)
    stage = Usd.Stage.Open(str(source))
    variant_root = UsdGeom.Xform.Define(stage, "/World/VariantAsset").GetPrim()
    variants = variant_root.GetVariantSets().AddVariantSet("model")
    for name, marker in (("A", "alpha"), ("B", "beta")):
        variants.AddVariant(name)
        variants.SetVariantSelection(name)
        with variants.GetVariantEditContext():
            variant_root.CreateAttribute(
                "variantMarker",
                Sdf.ValueTypeNames.String,
            ).Set(marker)
    variants.SetVariantSelection("A")
    assert stage.GetRootLayer().Save()

    result = run_texture_uv_leaf(
        _write_uv_invocation(source, output_dir, policy="generate_missing")
    )

    assert result.native_disposition == "passed"
    saved = Usd.Stage.Open(result.output.path)
    saved_variants = saved.GetPrimAtPath("/World/VariantAsset").GetVariantSet("model")
    assert saved_variants.GetVariantNames() == ["A", "B"]
    for name, marker in (("A", "alpha"), ("B", "beta")):
        assert saved_variants.SetVariantSelection(name)
        assert (
            saved.GetPrimAtPath("/World/VariantAsset")
            .GetAttribute("variantMarker")
            .Get()
            == marker
        )
    assert "subLayers" in Path(result.output.path).read_text(encoding="utf-8")


def test_animated_topology_is_inspected_but_never_authored(tmp_path: Path) -> None:
    source = tmp_path / "animated-topology.usda"
    inspect_dir = tmp_path / "inspect"
    inspect_dir.mkdir()
    _build_stage(source)
    stage = Usd.Stage.Open(str(source))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Mesh"))
    points = mesh.GetPointsAttr().Get()
    mesh.GetPointsAttr().Set(points, Usd.TimeCode(1.0))
    animated = Vt.Vec3fArray(points)
    animated[1] = Gf.Vec3f(2.0, 0.0, 0.0)
    mesh.GetPointsAttr().Set(animated, Usd.TimeCode(2.0))
    assert stage.GetRootLayer().Save()

    inspected = run_texture_uv_leaf(
        _write_uv_invocation(source, inspect_dir, policy="inspect")
    )
    readback = json.loads(
        Path(inspected.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )
    assert inspected.native_disposition == "not_evaluated"
    assert inspected.detail == (
        "UV inspection completed with non-ready saved-stage readback statuses: "
        "/World/Mesh: unsupported; saved-stage readback reasons: "
        "/World/Mesh: time-varying mesh topology is unsupported."
    )
    assert readback["mesh_readbacks"][0]["status"] == "unsupported"
    assert readback["mesh_readbacks"][0]["topology_time_sample_count"] == 2


@pytest.mark.parametrize(
    ("type_name", "values"),
    [
        (
            Sdf.ValueTypeNames.Float3Array,
            Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)] * 4),
        ),
        (Sdf.ValueTypeNames.FloatArray, Vt.FloatArray([0.0] * 4)),
    ],
)
def test_wrong_uv_type_or_row_arity_never_reports_ready(
    tmp_path: Path,
    type_name: object,
    values: object,
) -> None:
    source = tmp_path / "wrong-uv-type.usda"
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    _build_stage(source)
    stage = Usd.Stage.Open(str(source))
    primvar = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Mesh")).CreatePrimvar(
        "st",
        type_name,
        UsdGeom.Tokens.faceVarying,
    )
    primvar.Set(values)
    assert stage.GetRootLayer().Save()

    result = run_texture_uv_leaf(
        _write_uv_invocation(source, output_dir, policy="generate_missing")
    )

    readback = json.loads(
        Path(result.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )
    assert result.native_disposition == "failed"
    assert result.error == result.detail
    assert result.output == result.source
    assert readback["mesh_readbacks"][0]["status"] == "unsupported"


def test_material_identity_includes_property_metadata(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    texture = source_dir / "albedo.png"
    texture.write_bytes(b"texture")
    source = source_dir / "material-metadata.usda"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    _build_stage(source, authored_uvs=True, external_texture=texture)

    first = run_texture_uv_leaf(
        _write_uv_invocation(source, first_dir, policy="inspect")
    )
    first_readback = json.loads(
        Path(first.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )
    stage = Usd.Stage.Open(str(source))
    file_attribute = (
        UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/Material/Texture"))
        .GetInput("file")
        .GetAttr()
    )
    file_attribute.SetColorSpace("sRGB")
    assert stage.GetRootLayer().Save()
    second = run_texture_uv_leaf(
        _write_uv_invocation(source, second_dir, policy="inspect")
    )
    second_readback = json.loads(
        Path(second.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )

    assert (
        first_readback["scope_readbacks"][0]["material_identity_sha256"]
        != second_readback["scope_readbacks"][0]["material_identity_sha256"]
    )


def test_mesh_selection_includes_material_bound_subset_scope(tmp_path: Path) -> None:
    source = tmp_path / "mesh-subset.usda"
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    _build_stage(source)
    stage = Usd.Stage.Open(str(source))
    subset = UsdGeom.Subset.Define(stage, "/World/Mesh/Subset")
    subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
    subset.CreateIndicesAttr(Vt.IntArray([0]))
    subset_material = UsdShade.Material.Define(
        stage,
        "/World/Looks/SubsetMaterial",
    )
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(subset_material)
    assert stage.GetRootLayer().Save()

    result = run_texture_uv_leaf(
        _write_uv_invocation(source, output_dir, policy="generate_missing")
    )
    readback = json.loads(
        Path(result.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )

    assert [item["scope_prim_path"] for item in readback["scope_readbacks"]] == [
        "/World/Mesh",
        "/World/Mesh/Subset",
    ]
    assert readback["scope_readbacks"][1]["binding_relationship_path"] == (
        "/World/Mesh/Subset.material:binding"
    )
    assert readback["material_identities_unchanged"] is True


def test_mesh_selection_includes_ancestor_collection_preview_subset_binding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mesh-collection-subset.usda"
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    _build_stage(source)
    stage = Usd.Stage.Open(str(source))
    mesh = stage.GetPrimAtPath("/World/Mesh")
    UsdShade.MaterialBindingAPI(mesh).UnbindAllBindings()
    subset = UsdGeom.Subset.Define(stage, "/World/Mesh/Subset")
    subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
    subset.CreateIndicesAttr(Vt.IntArray([0]))
    preview_material = UsdShade.Material.Define(
        stage,
        "/World/Looks/PreviewSubsetMaterial",
    )
    collection = Usd.CollectionAPI.Apply(mesh, "previewSubset")
    collection.CreateIncludesRel().SetTargets([subset.GetPath()])
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
        collection,
        preview_material,
        bindingName="previewSubset",
        materialPurpose=UsdShade.Tokens.preview,
    )
    assert stage.GetRootLayer().Save()

    result = run_texture_uv_leaf(
        _write_uv_invocation(source, output_dir, policy="generate_missing")
    )
    readback = json.loads(
        Path(result.saved_stage_readbacks[0].path).read_text(encoding="utf-8")
    )

    subset_rows = [
        item
        for item in readback["scope_readbacks"]
        if item["scope_prim_path"] == "/World/Mesh/Subset"
    ]
    assert subset_rows == [
        {
            "binding_relationship_path": (
                "/World/Mesh.material:binding:collection:preview:previewSubset"
            ),
            "effective_material_path": "/World/Looks/PreviewSubsetMaterial",
            "material_identity_sha256": subset_rows[0]["material_identity_sha256"],
            "material_purpose": "preview",
            "requested_prim_path": "/World/Mesh",
            "scope_identity_sha256": subset_rows[0]["scope_identity_sha256"],
            "scope_prim_path": "/World/Mesh/Subset",
            "target_kind": "subset",
            "uv_mesh_path": "/World/Mesh",
        }
    ]
    assert readback["material_identities_unchanged"] is True


def test_uv_authoring_rejects_variant_composed_target_without_leaking(
    tmp_path: Path,
) -> None:
    source = tmp_path / "variant-target.usda"
    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/World")
    variants = root.GetPrim().GetVariantSets().AddVariantSet("model")
    for selection, scale in (("A", 1.0), ("B", 2.0)):
        variants.AddVariant(selection)
        variants.SetVariantSelection(selection)
        with variants.GetVariantEditContext():
            mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
            mesh.CreatePointsAttr(
                Vt.Vec3fArray(
                    [
                        Gf.Vec3f(0.0, 0.0, 0.0),
                        Gf.Vec3f(scale, 0.0, 0.0),
                        Gf.Vec3f(0.0, scale, 0.0),
                    ]
                )
            )
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    variants.SetVariantSelection("A")
    assert stage.GetRootLayer().Save()

    invocation_path = _write_uv_invocation(
        source,
        output_dir,
        policy="generate_missing",
    )
    result = run_texture_uv_leaf(invocation_path)

    reopened = Usd.Stage.Open(str(source))
    reopened_variants = (
        reopened.GetPrimAtPath("/World").GetVariantSets().GetVariantSet("model")
    )
    for selection in ("A", "B"):
        reopened_variants.SetVariantSelection(selection)
        primvar = UsdGeom.PrimvarsAPI(reopened.GetPrimAtPath("/World/Mesh")).GetPrimvar(
            "st"
        )
        assert not primvar
    _assert_sealed_preflight_rejection(
        result,
        output_dir,
        message="variant-composed target meshes",
    )


@pytest.mark.parametrize("arc_kind", ["reference", "payload"])
def test_uv_authoring_rejects_ancestor_variant_selected_composition_arc(
    tmp_path: Path,
    arc_kind: str,
) -> None:
    def build_model(path: Path, *, vertex_count: int) -> None:
        model_stage = Usd.Stage.CreateNew(str(path))
        model = UsdGeom.Xform.Define(model_stage, "/Model")
        model_stage.SetDefaultPrim(model.GetPrim())
        mesh = UsdGeom.Mesh.Define(model_stage, "/Model/Mesh")
        points = [
            Gf.Vec3f(float(index % 2), float(index // 2), 0.0)
            for index in range(vertex_count)
        ]
        mesh.CreatePointsAttr(Vt.Vec3fArray(points))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([vertex_count]))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(range(vertex_count)))
        assert model_stage.GetRootLayer().Save()

    quad = tmp_path / "quad.usda"
    triangle = tmp_path / "triangle.usda"
    build_model(quad, vertex_count=4)
    build_model(triangle, vertex_count=3)
    source = tmp_path / f"variant-{arc_kind}.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.Xform.Define(stage, "/World")
    asset = UsdGeom.Xform.Define(stage, "/World/Asset")
    variants = asset.GetPrim().GetVariantSets().AddVariantSet("model")
    for selection, model_path in (("A", quad), ("B", triangle)):
        variants.AddVariant(selection)
        variants.SetVariantSelection(selection)
        with variants.GetVariantEditContext():
            if arc_kind == "reference":
                asset.GetPrim().GetReferences().AddReference(str(model_path))
            else:
                asset.GetPrim().GetPayloads().AddPayload(str(model_path))
    variants.SetVariantSelection("A")
    assert stage.GetRootLayer().Save()

    output_dir = tmp_path / "leaf"
    output_dir.mkdir()
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=output_dir,
        target_prim_paths=("/World/Asset/Mesh",),
        policy="generate_missing",
    )
    invocation_path = output_dir / "invocation.json"
    atomic_write_json(invocation_path, invocation)

    result = run_texture_uv_leaf(invocation_path)

    reopened = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    reopened_variants = (
        reopened.GetPrimAtPath("/World/Asset").GetVariantSets().GetVariantSet("model")
    )
    for selection, expected_count in (("A", 4), ("B", 3)):
        reopened_variants.SetVariantSelection(selection)
        mesh = reopened.GetPrimAtPath("/World/Asset/Mesh")
        assert len(UsdGeom.Mesh(mesh).GetPointsAttr().Get()) == expected_count
        assert not UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    _assert_sealed_preflight_rejection(
        result,
        output_dir,
        message="variant-composed target meshes",
    )


def test_leaf_rejects_attempt_root_replacement_without_path_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    attempt_root = tmp_path / "attempt"
    moved_root = tmp_path / "moved-attempt"
    attempt_root.mkdir()
    _build_stage(source)
    invocation_path = _write_uv_invocation(
        source,
        attempt_root,
        policy="generate_missing",
    )
    real_build = uv_authoring._build_readback
    replaced = False

    def replace_root(**kwargs: object) -> object:
        nonlocal replaced
        readback = real_build(**kwargs)  # type: ignore[arg-type]
        if not replaced:
            attempt_root.rename(moved_root)
            attempt_root.mkdir()
            (attempt_root / "competitor.txt").write_text("preserve", encoding="utf-8")
            replaced = True
        return readback

    monkeypatch.setattr(uv_authoring, "_build_readback", replace_root)
    with pytest.raises(ValueError, match="attempt root identity changed"):
        run_texture_uv_leaf(invocation_path)

    assert (attempt_root / "competitor.txt").read_text(encoding="utf-8") == "preserve"
    assert tuple(attempt_root.iterdir()) == (attempt_root / "competitor.txt",)
    assert sorted(path.name for path in moved_root.iterdir()) == [
        "invocation.json",
        "prepared_texture_uvs.usda",
        "texture_uv_evidence.json",
        "texture_uv_saved_stage_readback.json",
    ]


def test_leaf_rejects_post_open_output_replacement_without_unlinking_competitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    _build_stage(source)
    invocation_path = _write_uv_invocation(
        source,
        attempt_root,
        policy="generate_missing",
    )
    output_path = attempt_root / "prepared_texture_uvs.usda"
    real_open = uv_authoring._stage_from_saved_snapshot

    replaced = False

    def replace_after_open(document: bytes, **kwargs: object) -> Usd.Stage:
        nonlocal replaced
        stage = real_open(document, **kwargs)  # type: ignore[arg-type]
        if not replaced:
            output_path.unlink()
            output_path.write_bytes(b"competitor-output")
            replaced = True
        return stage

    monkeypatch.setattr(
        uv_authoring,
        "_stage_from_saved_snapshot",
        replace_after_open,
    )
    with pytest.raises(ValueError, match="output path changed after exact reopen"):
        run_texture_uv_leaf(invocation_path)

    assert output_path.read_bytes() == b"competitor-output"
    assert sorted(path.name for path in attempt_root.iterdir()) == [
        "invocation.json",
        "prepared_texture_uvs.usda",
    ]
