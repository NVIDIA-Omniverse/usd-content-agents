# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for fail-closed raw Sdf layer asset dependency discovery."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pytest

from world_understanding.utils.usd.asset_paths import (
    collect_layer_authored_asset_paths,
    expand_udim_asset_pattern,
    is_bare_mdl_asset_path,
    require_authored_dependency_file,
    resolve_layer_udim_asset_paths,
)


def test_expand_udim_asset_pattern_is_exact_and_bounded() -> None:
    candidates = expand_udim_asset_pattern("textures/albedo.<UDIM>.png")

    assert len(candidates) == 100
    assert candidates[0] == "textures/albedo.1001.png"
    assert candidates[-1] == "textures/albedo.1100.png"
    assert "textures/albedo.1000.png" not in candidates
    assert "textures/albedo.1101.png" not in candidates


@pytest.mark.parametrize(
    "pattern",
    (
        "textures/albedo.png",
        "textures/<UDIM>/albedo.png",
        "textures/albedo.<UDIM>.<UDIM>.png",
    ),
)
def test_expand_udim_asset_pattern_rejects_ambiguous_patterns(pattern: str) -> None:
    with pytest.raises(ValueError, match="exactly one <UDIM> token"):
        expand_udim_asset_pattern(pattern)


def test_resolve_layer_udim_asset_paths_uses_openusd_layer_anchor(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    textures = tmp_path / "textures"
    textures.mkdir()
    (textures / "albedo.1001.png").write_bytes(b"tile-1001")
    source = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(source))
    prim = stage.DefinePrim("/World")
    prim.CreateAttribute("inputs:file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/albedo.<UDIM>.png")
    )
    stage.GetRootLayer().Save()

    pattern, concrete = resolve_layer_udim_asset_paths(
        stage.GetRootLayer(),
        "textures/albedo.<UDIM>.png",
        usd_shade=UsdShade,
    )

    assert pattern == str(textures / "albedo.<UDIM>.png")
    assert concrete == (str(textures / "albedo.1001.png"),)


def test_resolve_layer_udim_asset_paths_rejects_missing_tiles(tmp_path: Path) -> None:
    from pxr import Usd, UsdShade

    source = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(source))
    stage.DefinePrim("/World")
    stage.GetRootLayer().Save()

    with pytest.raises(ValueError, match="has no concrete tiles"):
        resolve_layer_udim_asset_paths(
            stage.GetRootLayer(),
            "textures/albedo.<UDIM>.png",
            usd_shade=UsdShade,
        )


class _FailingUdimUtils:
    @staticmethod
    def ResolveUdimPath(_path: str, _layer: Any) -> str:  # noqa: N802
        raise RuntimeError("resolver failure")


class _FailingUsdShade:
    UdimUtils = _FailingUdimUtils


def test_resolve_layer_udim_asset_paths_reports_resolver_failure() -> None:
    with pytest.raises(ValueError, match="Unable to resolve authored UDIM"):
        resolve_layer_udim_asset_paths(
            object(),
            "textures/albedo.<UDIM>.png",
            usd_shade=_FailingUsdShade,
        )


class _UnanchoredUdimUtils:
    @staticmethod
    def ResolveUdimPath(_path: str, _layer: Any) -> str:  # noqa: N802
        return ""

    @staticmethod
    def ResolveUdimTilePaths(  # noqa: N802
        _path: str,
        _layer: Any,
    ) -> tuple[tuple[str, str], ...]:
        return (("textures/albedo.1001.png", "1001"),)


class _UnanchoredUsdShade:
    UdimUtils = _UnanchoredUdimUtils


def test_resolve_layer_udim_asset_paths_rejects_empty_anchor() -> None:
    with pytest.raises(ValueError, match="Unable to anchor authored UDIM"):
        resolve_layer_udim_asset_paths(
            object(),
            "textures/albedo.<UDIM>.png",
            usd_shade=_UnanchoredUsdShade,
        )


def _write_default_prim_layer(path: Path, prim_name: str) -> None:
    from pxr import Usd, UsdGeom

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    prim = UsdGeom.Xform.Define(stage, f"/{prim_name}").GetPrim()
    stage.SetDefaultPrim(prim)
    stage.GetRootLayer().Save()


def test_layer_authored_asset_paths_preserves_post_articulation_closure(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    dependencies = {
        "layers/base.usda": "BaseLayer",
        "references/visual.usda": "Visual",
        "payloads/body.usda": "Body",
        "clips/anim.usda": "Animation",
        "clips/manifest.usda": "Manifest",
    }
    for relative_path, prim_name in dependencies.items():
        _write_default_prim_layer(tmp_path / relative_path, prim_name)
    texture_path = tmp_path / "textures" / "albedo.png"
    texture_path.parent.mkdir()
    texture_path.write_bytes(b"png")

    root_path = tmp_path / "root.usda"
    stage = Usd.Stage.CreateNew(str(root_path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    stage.GetRootLayer().subLayerPaths.append("layers/base.usda")
    base = UsdGeom.Cube.Define(stage, "/World/Base").GetPrim()
    door = UsdGeom.Cube.Define(stage, "/World/Door").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    UsdPhysics.RigidBodyAPI.Apply(door)
    UsdPhysics.ArticulationRootAPI.Apply(world)
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joints/Hinge")
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([door.GetPath()])
    referenced = UsdGeom.Xform.Define(stage, "/World/Referenced").GetPrim()
    referenced.GetReferences().AddReference("references/visual.usda")
    payload = UsdGeom.Xform.Define(stage, "/World/Payload").GetPrim()
    payload.GetPayloads().AddPayload("payloads/body.usda")
    animated = UsdGeom.Xform.Define(stage, "/World/Animated").GetPrim()
    clips = Usd.ClipsAPI(animated)
    clips.SetClipAssetPaths([Sdf.AssetPath("clips/anim.usda")])
    clips.SetClipManifestAssetPath(Sdf.AssetPath("clips/manifest.usda"))
    world.CreateAttribute("inputs:albedo", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/albedo.png")
    )
    source = world.CreateAttribute("outputs:source", Sdf.ValueTypeNames.Float)
    sink = world.CreateAttribute("inputs:sink", Sdf.ValueTypeNames.Float)
    sink.AddConnection(source.GetPath())
    stage.GetRootLayer().Save()

    layer = Sdf.Layer.FindOrOpen(str(root_path))
    assert layer is not None
    null_target_paths: list[Any] = []

    def record_null_target(path: Any) -> None:
        if layer.GetObjectAtPath(path) is None:
            null_target_paths.append(path)

    layer.Traverse(Sdf.Path.absoluteRootPath, record_null_target)
    assert null_target_paths
    assert all(path.IsTargetPath() for path in null_target_paths)
    target_owner_specs = [
        layer.GetObjectAtPath(path.GetParentPath()) for path in null_target_paths
    ]
    assert all(
        isinstance(spec, Sdf.RelationshipSpec | Sdf.AttributeSpec)
        for spec in target_owner_specs
    )
    assert any(isinstance(spec, Sdf.RelationshipSpec) for spec in target_owner_specs)
    assert any(isinstance(spec, Sdf.AttributeSpec) for spec in target_owner_specs)

    all_paths, composition_paths = collect_layer_authored_asset_paths(
        layer,
        sdf=Sdf,
    )

    assert composition_paths == {
        "layers/base.usda",
        "references/visual.usda",
        "payloads/body.usda",
    }
    assert all_paths == composition_paths | {
        "clips/anim.usda",
        "clips/manifest.usda",
        "textures/albedo.png",
    }


class _BrokenTraversalPath:
    def IsTargetPath(self) -> bool:  # noqa: N802 - mirrors Sdf.Path
        return False

    def __str__(self) -> str:
        return "/World/Broken"


class _BrokenTraversalLayer:
    identifier = "broken.usda"
    subLayerPaths: tuple[str, ...] = ()

    def Traverse(self, _root: Any, visit: Any) -> None:  # noqa: N802
        visit(_BrokenTraversalPath())

    def GetObjectAtPath(self, _path: Any) -> None:  # noqa: N802
        return None


def test_layer_authored_asset_paths_rejects_unexpected_missing_spec() -> None:
    from pxr import Sdf

    with pytest.raises(
        ValueError,
        match=r"layer='broken\.usda'.*path=/World/Broken",
    ):
        collect_layer_authored_asset_paths(_BrokenTraversalLayer(), sdf=Sdf)


class _UnreadableMetadataSpec:
    def ListInfoKeys(self) -> list[str]:  # noqa: N802
        return ["clips"]

    def GetInfo(self, _key: str) -> None:  # noqa: N802
        raise RuntimeError("missing Python converter")


class _UnreadableMetadataLayer:
    identifier = "malformed.usda"
    subLayerPaths: tuple[str, ...] = ()

    def Traverse(self, _root: Any, visit: Any) -> None:  # noqa: N802
        from pxr import Sdf

        visit(Sdf.Path("/World/Asset"))

    def GetObjectAtPath(self, _path: Any) -> _UnreadableMetadataSpec:  # noqa: N802
        return _UnreadableMetadataSpec()


def test_layer_authored_asset_paths_reports_unreadable_metadata_context() -> None:
    from pxr import Sdf

    with pytest.raises(
        ValueError,
        match=(r"layer='malformed\.usda'.*path=/World/Asset.*key=clips"),
    ):
        collect_layer_authored_asset_paths(_UnreadableMetadataLayer(), sdf=Sdf)


class _LegacyCompositionLayer:
    identifier = "legacy.usda"
    subLayerPaths: tuple[str, ...] = ()

    def Traverse(self, _root: Any, _visit: Any) -> None:  # noqa: N802
        return None

    def GetExternalReferences(self) -> tuple[str, ...]:  # noqa: N802
        return ("legacy-reference.usda",)


def test_layer_authored_asset_paths_supports_legacy_composition_api() -> None:
    from pxr import Sdf

    all_paths, composition_paths = collect_layer_authored_asset_paths(
        _LegacyCompositionLayer(),
        sdf=Sdf,
    )

    assert all_paths == {"legacy-reference.usda"}
    assert composition_paths == {"legacy-reference.usda"}


@pytest.mark.parametrize(
    ("path", "expected"),
    (
        ("OmniPBR.mdl", True),
        ("Other.mdl", True),
        ("materials/OmniPBR.mdl", False),
        ("omniverse://library/OmniPBR.mdl", False),
        ("OmniPBR.png", False),
    ),
)
def test_bare_mdl_asset_path_is_narrow(path: str, expected: bool) -> None:
    assert is_bare_mdl_asset_path(path) is expected


def test_require_authored_dependency_file_caches_package_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_path = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package_path, "w") as package:
        package.writestr("0/a.png", b"a")
        package.writestr("0/b.png", b"b")

    from world_understanding.utils.usd import asset_paths

    opened = 0
    real_zip_file = zipfile.ZipFile

    def counting_zip_file(*args: Any, **kwargs: Any) -> zipfile.ZipFile:
        nonlocal opened
        opened += 1
        return real_zip_file(*args, **kwargs)

    monkeypatch.setattr(asset_paths.zipfile, "ZipFile", counting_zip_file)
    cache: dict[Path, frozenset[str]] = {}

    def split_identifier(value: str) -> tuple[str, str]:
        outer, member = value[:-1].split("[", 1)
        return outer, member

    for member in ("0/a.png", "0/b.png"):
        require_authored_dependency_file(
            f"{package_path}[{member}]",
            split_identifier=split_identifier,
            layer_identifier="scene.usda",
            authored_path=member,
            package_members_cache=cache,
        )

    assert opened == 1
    assert cache == {package_path: frozenset({"0/a.png", "0/b.png"})}
