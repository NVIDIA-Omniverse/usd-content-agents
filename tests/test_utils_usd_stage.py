# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for USD Stage utility functions."""

import os
from pathlib import Path

import pytest

# Skip all tests in this module if pxr is not installed
pxr = pytest.importorskip("pxr")

from pxr import Sdf, Usd, UsdGeom  # noqa: E402

from world_understanding.utils.usd.stage import (  # noqa: E402
    MAX_PATH_COMPONENT_LEN,
    _disable_all_instancing,
    _is_relative_render_asset_path,
    _render_layer_has_relative_composition_paths,
    _resolve_render_asset_path,
    _resolve_render_composition_item,
    _resolve_render_layer_composition_paths,
    _resolve_render_sublayer_path,
    create_stage,
    create_stage_with_file,
    create_temp_stage,
    duplicate_stage,
    export_stage_to_string,
    flatten_stage,
    get_stage_info,
    get_stage_info_from_path,
    has_uri_scheme,
    is_windows_drive_path,
    load_stage,
    load_stage_from_string,
    merge_stages,
    normalize_windows_drive_path,
    prepare_stage_for_render,
    remove_animation,
    save_stage,
    shorten_for_filesystem,
)


def test_usd_asset_uri_scheme_detection_excludes_windows_paths() -> None:
    assert has_uri_scheme("https://example.com/scene.usda")
    assert has_uri_scheme("s3://bucket/scene.usda")
    assert has_uri_scheme("omniverse://server/scene.usda")
    assert has_uri_scheme("data:application/octet-stream;base64,AAAA")
    assert has_uri_scheme("file:/tmp/scene.usda")

    assert not has_uri_scheme("./scene.usda")
    assert not has_uri_scheme("../scene.usda")
    assert not has_uri_scheme("/tmp/scene.usda")
    assert not has_uri_scheme("C:/assets/scene.usda")
    assert not has_uri_scheme(r"C:\assets\scene.usda")

    assert is_windows_drive_path("C:/assets/scene.usda")
    assert is_windows_drive_path(r"C:\assets\scene.usda")
    assert not is_windows_drive_path("s3://bucket/scene.usda")
    assert normalize_windows_drive_path(r"C:\assets\scene.usda") == (
        "C:/assets/scene.usda"
    )
    assert (
        _resolve_render_sublayer_path(
            r"C:\assets\sublayer.usda",
            Path("/fallback"),
        )
        == "C:/assets/sublayer.usda"
    )


def test_sanitize_name_for_filesystem():
    """Test the sanitize_name_for_filesystem function."""
    from world_understanding.utils.usd.stage import (
        sanitize_name_for_filesystem,
    )

    # Test basic path sanitization
    assert sanitize_name_for_filesystem("/World/Camera_001") == "World_Camera_001"

    # Test with spaces
    assert sanitize_name_for_filesystem("/World/Camera 001") == "World_Camera_001"

    # Test with special characters
    assert sanitize_name_for_filesystem("Camera:Main (HD)") == "Camera_Main__HD"

    # Test with multiple slashes
    assert (
        sanitize_name_for_filesystem("/World/Lights/Key Light")
        == "World_Lights_Key_Light"
    )

    # Test empty string
    assert sanitize_name_for_filesystem("") == "unnamed"

    # Test string that becomes empty after sanitization
    assert sanitize_name_for_filesystem("///") == "unnamed"

    # Test with dots and hyphens (should be preserved)
    assert sanitize_name_for_filesystem("camera-main.v2") == "camera-main.v2"


def test_shorten_for_filesystem_leaves_short_names_unchanged():
    """Short names should only be sanitized, not truncated."""
    assert shorten_for_filesystem("short_name", max_len=32) == "short_name"
    assert shorten_for_filesystem("Camera:Main (HD)", max_len=64) == "Camera_Main__HD"


def test_shorten_for_filesystem_is_deterministic_and_bounded():
    """Long names should be stably truncated with hash suffix and length cap."""
    long_name = "part_" + ("abc123XYZ" * 50)
    shortened_once = shorten_for_filesystem(long_name, max_len=MAX_PATH_COMPONENT_LEN)
    shortened_twice = shorten_for_filesystem(long_name, max_len=MAX_PATH_COMPONENT_LEN)

    assert shortened_once == shortened_twice
    assert len(shortened_once) <= MAX_PATH_COMPONENT_LEN
    assert "_" in shortened_once


def test_shorten_for_filesystem_distinguishes_colliding_prefixes():
    """Different long names with same prefix should produce different outputs."""
    shared_prefix = "same_prefix_" + ("x" * 200)
    long_name_a = shared_prefix + "_A"
    long_name_b = shared_prefix + "_B"

    shortened_a = shorten_for_filesystem(long_name_a, max_len=64)
    shortened_b = shorten_for_filesystem(long_name_b, max_len=64)

    assert shortened_a != shortened_b
    assert len(shortened_a) <= 64
    assert len(shortened_b) <= 64


def test_shorten_for_filesystem_hash_only_when_budget_is_tiny():
    shortened = shorten_for_filesystem("very-long-name", max_len=4, hash_len=8)

    assert len(shortened) == 4
    assert "_" not in shortened


def test_disable_all_instancing_clears_authored_instanceable_prims():
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/Instanced", "Xform")
    prim.SetInstanceable(True)

    assert _disable_all_instancing(stage) == 1
    assert prim.HasAuthoredInstanceable()
    assert not prim.IsInstanceable()


def test_create_stage():
    """Test creating a stage in memory."""
    # Test with identifier
    stage = create_stage("test.usda")
    assert stage is not None
    assert isinstance(stage, Usd.Stage)

    # Test without identifier
    stage = create_stage()
    assert stage is not None
    assert isinstance(stage, Usd.Stage)


def test_create_stage_with_file(tmp_path):
    """Test creating a stage with an associated file."""
    test_file = tmp_path / "test.usda"
    stage = create_stage_with_file(test_file)

    assert stage is not None
    assert isinstance(stage, Usd.Stage)
    assert os.path.exists(test_file)


def test_load_stage(tmp_path):
    """Test loading a stage from file."""
    # Create a test file with a real USD stage
    test_file = tmp_path / "test.usda"
    test_stage = Usd.Stage.CreateNew(str(test_file))
    test_stage.Save()

    stage = load_stage(test_file)
    assert stage is not None
    assert isinstance(stage, Usd.Stage)

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        load_stage(tmp_path / "nonexistent.usda")


def test_save_stage(tmp_path):
    """Test saving a stage to file."""
    # Create a stage with a file
    existing_path = tmp_path / "existing.usda"
    stage = Usd.Stage.CreateNew(str(existing_path))
    stage.Save()

    # Test saving to new path
    new_path = tmp_path / "new.usda"
    result = save_stage(stage, new_path)
    assert result == str(new_path)
    assert os.path.exists(new_path)

    # Test saving to existing path
    result = save_stage(stage)
    assert Path(result) == existing_path

    # Test in-memory stage without path
    # Note: In-memory stages have anonymous identifiers like "anon:0x..."
    # The save_stage function should raise an error for anonymous layers
    in_memory_stage = Usd.Stage.CreateInMemory()
    # Check if the identifier is anonymous
    assert in_memory_stage.GetRootLayer().anonymous
    with pytest.raises(ValueError):
        save_stage(in_memory_stage)


def test_export_stage_to_string():
    """Test exporting stage to string."""
    stage = Usd.Stage.CreateInMemory()
    # Add a simple prim to make the stage non-empty
    stage.DefinePrim("/TestPrim", "Xform")

    # Test with comment
    result = export_stage_to_string(stage, add_comment=True)
    assert result is not None
    assert "#usda 1.0" in result
    assert "TestPrim" in result

    # Test without comment
    result = export_stage_to_string(stage, add_comment=False)
    assert result is not None
    assert "#usda 1.0" in result
    assert "TestPrim" in result


def test_load_stage_from_string():
    """Test creating stage from string."""
    usda_content = '#usda 1.0\ndef Xform "Test" {}'

    stage = load_stage_from_string(usda_content, "test.usda")

    assert stage is not None
    assert isinstance(stage, Usd.Stage)
    # Verify the content was loaded
    test_prim = stage.GetPrimAtPath("/Test")
    assert test_prim.IsValid()
    assert test_prim.GetTypeName() == "Xform"


def test_duplicate_stage():
    """Test duplicating a stage."""
    source_stage = Usd.Stage.CreateInMemory()
    source_stage.DefinePrim("/SourcePrim", "Xform")

    # Duplicate with identifier
    dup_stage = duplicate_stage(source_stage, "duplicate.usda")
    assert dup_stage is not None
    assert isinstance(dup_stage, Usd.Stage)
    # Verify the content was duplicated
    dup_prim = dup_stage.GetPrimAtPath("/SourcePrim")
    assert dup_prim.IsValid()

    # Duplicate without identifier
    dup_stage = duplicate_stage(source_stage)
    assert dup_stage is not None
    assert isinstance(dup_stage, Usd.Stage)
    dup_prim = dup_stage.GetPrimAtPath("/SourcePrim")
    assert dup_prim.IsValid()


def test_create_temp_stage():
    """Test creating temporary stage."""
    stage, temp_path = create_temp_stage()

    assert stage is not None
    assert isinstance(stage, Usd.Stage)
    assert os.path.exists(temp_path)
    assert temp_path.endswith(".usda")

    # Clean up
    os.unlink(temp_path)


def test_get_stage_info(tmp_path):
    """Test getting stage information."""
    # Create a stage with some properties
    test_file = tmp_path / "test.usda"
    stage = Usd.Stage.CreateNew(str(test_file))

    # Set up stage properties
    stage.SetStartTimeCode(1.0)
    stage.SetEndTimeCode(100.0)
    stage.SetTimeCodesPerSecond(24.0)

    # Add some prims
    world_prim = stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Child1", "Xform")
    stage.DefinePrim("/World/Child2", "Xform")
    stage.SetDefaultPrim(world_prim)

    # Set up axis and units
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)

    info = get_stage_info(stage)

    assert Path(info["root_layer_path"]) == test_file
    assert info["up_axis"] == "Y"
    assert info["meters_per_unit"] == 0.01
    assert info["start_time_code"] == 1.0
    assert info["end_time_code"] == 100.0
    assert info["time_codes_per_second"] == 24.0
    assert info["prim_count"] == 3  # World, Child1, Child2
    assert info["default_prim"] == "/World"
    assert info["layer_count"] >= 1


def test_get_stage_info_from_path(tmp_path):
    """Test getting stage info from a USD file path."""
    test_file = tmp_path / "count_test.usda"
    stage = Usd.Stage.CreateNew(str(test_file))

    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Child", "Xform")
    stage.Save()

    info = get_stage_info_from_path(test_file)
    assert info is not None
    assert info["prim_count"] == 2
    assert get_stage_info_from_path(tmp_path / "missing.usda") is None


def test_get_stage_info_from_path_returns_none_when_open_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Usd.Stage, "Open", lambda _path: None)

    assert get_stage_info_from_path("unopenable.usda") is None


def test_flatten_stage(tmp_path):
    """Test flattening a stage."""
    # Create a source stage with some hierarchy
    source_stage = Usd.Stage.CreateInMemory()
    source_stage.DefinePrim("/World", "Xform")
    source_stage.DefinePrim("/World/Child", "Xform")

    # Add a sublayer to test flattening
    sublayer = Sdf.Layer.CreateAnonymous()
    source_stage.GetRootLayer().subLayerPaths.append(sublayer.identifier)

    # Test with output path
    output_path = tmp_path / "flattened.usda"
    result = flatten_stage(source_stage, output_path, add_comment=True)
    assert result is not None
    assert isinstance(result, Usd.Stage)
    assert os.path.exists(output_path)

    # Test without output path (in-memory)
    result = flatten_stage(source_stage)
    assert result is not None
    assert isinstance(result, Usd.Stage)
    # Verify the prims exist in the flattened stage
    assert result.GetPrimAtPath("/World").IsValid()
    assert result.GetPrimAtPath("/World/Child").IsValid()


def test_prepare_stage_for_render_flattens_preserves_metadata_and_normalizes_mdl(
    tmp_path: Path,
) -> None:
    """Render prep should produce a self-contained stage without losing metadata."""
    from pxr import UsdShade

    sublayer_path = tmp_path / "geometry.usda"
    sublayer_stage = Usd.Stage.CreateNew(str(sublayer_path))
    sublayer_stage.DefinePrim("/World", "Xform")
    sublayer_stage.DefinePrim("/World/Cube", "Cube")
    shader = UsdShade.Shader.Define(sublayer_stage, "/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./Material/OmniPBR.mdl"))
    sublayer_stage.Save()

    root_path = tmp_path / "root.usda"
    root_stage = Usd.Stage.CreateNew(str(root_path))
    root_stage.GetRootLayer().subLayerPaths.append(str(sublayer_path))
    UsdGeom.SetStageUpAxis(root_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(root_stage, 0.5)

    prepared_stage, metadata = prepare_stage_for_render(root_stage)

    assert prepared_stage is not root_stage
    assert prepared_stage.GetPrimAtPath("/World/Cube").IsValid()
    assert UsdGeom.GetStageUpAxis(prepared_stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(prepared_stage) == 0.5
    mdl_attr = prepared_stage.GetPrimAtPath("/Shader").GetAttribute(
        "info:mdl:sourceAsset"
    )
    assert mdl_attr.Get() == Sdf.AssetPath("OmniPBR.mdl")
    assert metadata == {
        "flattened": True,
        "material_normalized": True,
        "asset_base_dir": str(tmp_path),
        "up_axis": "Z",
        "meters_per_unit": 0.5,
    }


def test_prepare_stage_for_render_can_skip_flatten_and_normalization() -> None:
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World", "Xform")

    prepared_stage, metadata = prepare_stage_for_render(
        stage,
        flatten=False,
        normalize_materials=False,
    )

    assert prepared_stage is not stage
    assert prepared_stage.GetPrimAtPath("/World").IsValid()
    assert metadata == {
        "flattened": False,
        "material_normalized": False,
    }


def test_prepare_stage_for_render_nonflatten_requires_anchor() -> None:
    source_stage = Usd.Stage.CreateInMemory()
    source_stage.DefinePrim("/World", "Xform")
    source_stage.DefinePrim("/World/Reference", "Xform").GetReferences().AddReference(
        "./geometry.usda"
    )

    with pytest.raises(RuntimeError, match="source root layer has no filesystem path"):
        prepare_stage_for_render(
            source_stage,
            flatten=False,
            normalize_materials=False,
        )


def test_prepare_stage_for_render_normalizes_without_mutating_source() -> None:
    from pxr import UsdShade

    source_stage = Usd.Stage.CreateInMemory()
    source_stage.DefinePrim("/World", "Xform")
    shader = UsdShade.Shader.Define(source_stage, "/Shader")
    mdl_attr = shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    )
    mdl_attr.Set(Sdf.AssetPath("./Material/OmniPBR.mdl"))

    prepared_stage, metadata = prepare_stage_for_render(
        source_stage,
        flatten=False,
        normalize_materials=True,
    )

    assert prepared_stage is not source_stage
    assert mdl_attr.Get() == Sdf.AssetPath("./Material/OmniPBR.mdl")
    prepared_attr = prepared_stage.GetPrimAtPath("/Shader").GetAttribute(
        "info:mdl:sourceAsset"
    )
    assert prepared_attr.Get() == Sdf.AssetPath("OmniPBR.mdl")
    assert metadata == {
        "flattened": False,
        "material_normalized": True,
    }


def test_prepare_stage_for_render_nonflatten_preserves_session_layer() -> None:
    source_stage = Usd.Stage.CreateInMemory()
    source_prim = source_stage.DefinePrim("/World", "Xform")
    source_prim.CreateAttribute("user:sessionFlag", Sdf.ValueTypeNames.Bool).Set(False)
    source_stage.DefinePrim("/World/Child", "Xform")
    with Usd.EditContext(source_stage, source_stage.GetSessionLayer()):
        session_prim = source_stage.OverridePrim("/World")
        session_attr = session_prim.CreateAttribute(
            "user:sessionFlag",
            Sdf.ValueTypeNames.Bool,
        )
        session_attr.Set(True)

    prepared_stage, metadata = prepare_stage_for_render(
        source_stage,
        flatten=False,
        normalize_materials=True,
    )

    prepared_attr = prepared_stage.GetPrimAtPath("/World").GetAttribute(
        "user:sessionFlag"
    )
    assert prepared_attr.Get() is True
    assert prepared_stage.GetPrimAtPath("/World/Child").IsValid()
    assert prepared_stage.GetSessionLayer().empty

    exported_root_stage = Usd.Stage.Open(prepared_stage.GetRootLayer())
    assert (
        exported_root_stage.GetPrimAtPath("/World")
        .GetAttribute("user:sessionFlag")
        .Get()
        is True
    )
    assert exported_root_stage.GetPrimAtPath("/World/Child").IsValid()
    assert metadata == {
        "flattened": False,
        "material_normalized": True,
    }


def test_prepare_stage_for_render_nonflatten_requires_session_anchor() -> None:
    source_stage = Usd.Stage.CreateInMemory()
    source_stage.DefinePrim("/World", "Xform")
    with Usd.EditContext(source_stage, source_stage.GetSessionLayer()):
        source_stage.OverridePrim(
            "/World/SessionReference"
        ).GetReferences().AddReference("./session_geometry.usda")

    with pytest.raises(
        RuntimeError, match="source session layer has no filesystem path"
    ):
        prepare_stage_for_render(
            source_stage,
            flatten=False,
            normalize_materials=False,
        )


def test_prepare_stage_for_render_nonflatten_rejects_dirty_sublayers(
    tmp_path: Path,
) -> None:
    sublayer_path = tmp_path / "sublayer.usda"
    sublayer = Sdf.Layer.CreateNew(str(sublayer_path))
    sublayer.Save()

    root_path = tmp_path / "root.usda"
    root_layer = Sdf.Layer.CreateNew(str(root_path))
    root_layer.subLayerPaths = ["./sublayer.usda"]
    root_layer.Save()

    source_stage = Usd.Stage.Open(str(root_path))
    assert source_stage is not None
    loaded_sublayer = Sdf.Layer.FindOrOpen(str(sublayer_path))
    assert loaded_sublayer is not None
    with Usd.EditContext(source_stage, loaded_sublayer):
        source_stage.DefinePrim("/UnsavedSublayerEdit", "Xform")
    assert loaded_sublayer.dirty

    with pytest.raises(RuntimeError, match="unsaved edits"):
        prepare_stage_for_render(
            source_stage,
            flatten=False,
            normalize_materials=True,
        )


def test_prepare_stage_for_render_anchors_nonflattened_relative_arcs(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "geometry.usda"
    reference_stage = Usd.Stage.CreateNew(str(reference_path))
    reference_stage.DefinePrim("/ReferencedRoot", "Xform")
    reference_stage.DefinePrim("/ReferencedRoot/ReferencedChild", "Cube")
    reference_stage.Save()

    payload_path = tmp_path / "payload.usda"
    payload_stage = Usd.Stage.CreateNew(str(payload_path))
    payload_stage.DefinePrim("/PayloadRoot", "Xform")
    payload_stage.DefinePrim("/PayloadRoot/PayloadChild", "Sphere")
    payload_stage.Save()

    root_path = tmp_path / "root.usda"
    root_stage = Usd.Stage.CreateNew(str(root_path))
    root_stage.DefinePrim("/World", "Xform")
    root_stage.DefinePrim("/World/Reference", "Xform").GetReferences().AddReference(
        "./geometry.usda",
        "/ReferencedRoot",
    )
    root_stage.DefinePrim("/World/Payload", "Xform").GetPayloads().AddPayload(
        "./payload.usda",
        "/PayloadRoot",
    )
    variant_prim = root_stage.DefinePrim("/World/Variant", "Xform")
    variant_set = variant_prim.GetVariantSets().AddVariantSet("model")
    variant_set.AddVariant("referenced")
    variant_set.SetVariantSelection("referenced")
    with variant_set.GetVariantEditContext():
        variant_prim.GetReferences().AddReference(
            "./geometry.usda",
            "/ReferencedRoot",
        )
    root_stage.DefinePrim("/World/DeleteReference", "Xform")
    delete_spec = root_stage.GetRootLayer().GetPrimAtPath("/World/DeleteReference")
    delete_spec.referenceList.deletedItems = [Sdf.Reference("./deleted.usda")]
    root_stage.Save()

    prepared_stage, metadata = prepare_stage_for_render(
        root_stage,
        flatten=False,
        normalize_materials=True,
    )

    assert prepared_stage.GetPrimAtPath("/World/Reference/ReferencedChild").IsValid()
    assert prepared_stage.GetPrimAtPath("/World/Payload/PayloadChild").IsValid()
    assert prepared_stage.GetPrimAtPath("/World/Variant/ReferencedChild").IsValid()
    prepared_delete_spec = prepared_stage.GetRootLayer().GetPrimAtPath(
        "/World/DeleteReference"
    )
    assert [
        item.assetPath for item in prepared_delete_spec.referenceList.deletedItems
    ] == ["./deleted.usda"]
    assert metadata == {
        "flattened": False,
        "material_normalized": True,
        "asset_base_dir": str(tmp_path),
    }

    relocated_path = tmp_path / "render-output" / "root_converted.usda"
    relocated_path.parent.mkdir()
    prepared_stage.GetRootLayer().Export(str(relocated_path))
    relocated_stage = Usd.Stage.Open(str(relocated_path))

    assert relocated_stage is not None
    assert relocated_stage.GetPrimAtPath("/World/Reference/ReferencedChild").IsValid()
    assert relocated_stage.GetPrimAtPath("/World/Payload/PayloadChild").IsValid()
    assert relocated_stage.GetPrimAtPath("/World/Variant/ReferencedChild").IsValid()


def test_prepare_stage_for_render_anchors_without_material_normalization(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "geometry.usda"
    reference_stage = Usd.Stage.CreateNew(str(reference_path))
    reference_stage.DefinePrim("/ReferencedRoot", "Xform")
    reference_stage.Save()

    root_path = tmp_path / "root.usda"
    root_stage = Usd.Stage.CreateNew(str(root_path))
    root_stage.DefinePrim("/World", "Xform")
    root_stage.DefinePrim("/World/Reference", "Xform").GetReferences().AddReference(
        "./geometry.usda",
        "/ReferencedRoot",
    )
    root_stage.Save()

    prepared_stage, metadata = prepare_stage_for_render(
        root_stage,
        flatten=False,
        normalize_materials=False,
    )

    prepared_spec = prepared_stage.GetRootLayer().GetPrimAtPath("/World/Reference")
    assert [item.assetPath for item in prepared_spec.referenceList.prependedItems] == [
        reference_path.as_posix()
    ]
    assert metadata == {
        "flattened": False,
        "material_normalized": False,
        "asset_base_dir": str(tmp_path),
    }


def test_prepare_stage_for_render_anchors_file_backed_session_layer(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "session_ref.usda"
    reference_stage = Usd.Stage.CreateNew(str(reference_path))
    reference_stage.DefinePrim("/SessionRoot", "Xform")
    reference_stage.Save()

    root_path = tmp_path / "root.usda"
    root_layer = Sdf.Layer.CreateNew(str(root_path))
    root_layer.Save()
    session_layer = Sdf.Layer.CreateNew(str(tmp_path / "session.usda"))
    session_layer.Save()
    source_stage = Usd.Stage.Open(root_layer, session_layer)
    with Usd.EditContext(source_stage, source_stage.GetSessionLayer()):
        source_stage.OverridePrim(
            "/World/SessionReference"
        ).GetReferences().AddReference(
            "./session_ref.usda",
            "/SessionRoot",
        )
    source_stage.GetSessionLayer().Save()

    prepared_stage, metadata = prepare_stage_for_render(
        source_stage,
        flatten=False,
        normalize_materials=False,
    )

    prepared_spec = prepared_stage.GetRootLayer().GetPrimAtPath(
        "/World/SessionReference"
    )
    assert [item.assetPath for item in prepared_spec.referenceList.prependedItems] == [
        reference_path.as_posix()
    ]
    assert metadata == {
        "flattened": False,
        "material_normalized": False,
        "asset_base_dir": str(tmp_path),
    }


def test_render_layer_relative_composition_path_helpers(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    layer = Sdf.Layer.CreateAnonymous("composition.usda")
    layer.subLayerPaths.append("./sub.usda")
    stage = Usd.Stage.Open(layer)
    stage.DefinePrim("/World/Reference", "Xform").GetReferences().AddReference(
        "./ref.usda"
    )
    stage.DefinePrim("/World/Payload", "Xform").GetPayloads().AddPayload(
        "./payload.usda"
    )

    assert _render_layer_has_relative_composition_paths(layer)
    _resolve_render_layer_composition_paths(layer, source_dir)

    assert layer.subLayerPaths == [(source_dir / "sub.usda").resolve().as_posix()]
    ref_spec = layer.GetPrimAtPath("/World/Reference")
    payload_spec = layer.GetPrimAtPath("/World/Payload")
    assert [item.assetPath for item in ref_spec.referenceList.prependedItems] == [
        (source_dir / "ref.usda").resolve().as_posix()
    ]
    assert [item.assetPath for item in payload_spec.payloadList.prependedItems] == [
        (source_dir / "payload.usda").resolve().as_posix()
    ]


def test_render_layer_detects_relative_payload_without_sublayers() -> None:
    layer = Sdf.Layer.CreateAnonymous("payload-only.usda")
    stage = Usd.Stage.Open(layer)
    stage.DefinePrim("/World/Payload", "Xform").GetPayloads().AddPayload("payload.usda")

    assert _render_layer_has_relative_composition_paths(layer)


def test_render_asset_path_helpers_cover_non_relative_inputs(tmp_path: Path) -> None:
    absolute = (tmp_path / "abs.usda").resolve().as_posix()

    assert not _is_relative_render_asset_path("")
    assert not _is_relative_render_asset_path("anon:layer")
    assert not _is_relative_render_asset_path(absolute)
    assert not _is_relative_render_asset_path("C:/assets/scene.usda")
    assert not _is_relative_render_asset_path("s3://bucket/scene.usda")
    assert _is_relative_render_asset_path("relative.usda")

    assert _resolve_render_sublayer_path("", tmp_path) == ""
    assert _resolve_render_sublayer_path("anon:layer", tmp_path) == "anon:layer"
    assert _resolve_render_sublayer_path(absolute, tmp_path) == absolute
    assert (
        _resolve_render_sublayer_path("s3://bucket/scene.usda", tmp_path)
        == "s3://bucket/scene.usda"
    )

    assert _resolve_render_asset_path("", tmp_path) == ""
    assert _resolve_render_asset_path("C:/assets/scene.usda", tmp_path) == (
        "C:/assets/scene.usda"
    )
    assert _resolve_render_asset_path(absolute, tmp_path) == absolute
    assert _resolve_render_asset_path("anon:layer", tmp_path) == "anon:layer"
    assert (
        _resolve_render_asset_path("s3://bucket/scene.usda", tmp_path)
        == "s3://bucket/scene.usda"
    )


def test_resolve_render_composition_item_preserves_or_rewrites_by_type(
    tmp_path: Path,
) -> None:
    absolute_reference = Sdf.Reference((tmp_path / "abs.usda").resolve().as_posix())
    assert _resolve_render_composition_item(absolute_reference, tmp_path) is (
        absolute_reference
    )

    payload = Sdf.Payload("payload.usda", "/PayloadRoot")
    resolved_payload = _resolve_render_composition_item(payload, tmp_path)

    assert isinstance(resolved_payload, Sdf.Payload)
    assert (
        resolved_payload.assetPath == (tmp_path / "payload.usda").resolve().as_posix()
    )
    assert resolved_payload.primPath == Sdf.Path("/PayloadRoot")

    class UnknownCompositionItem:
        assetPath = "relative.usda"

    unknown = UnknownCompositionItem()
    assert _resolve_render_composition_item(unknown, tmp_path) is unknown


def test_merge_stages():
    """Test merging stages."""
    # Create base stage
    base_stage = Usd.Stage.CreateInMemory()
    base_stage.DefinePrim("/World", "Xform")
    base_stage.DefinePrim("/World/Base", "Xform")

    # Create overlay stage
    overlay_stage = Usd.Stage.CreateInMemory()
    overlay_stage.DefinePrim("/Overlay", "Xform")
    overlay_stage.DefinePrim("/Overlay/Child", "Xform")

    # Test merge
    result = merge_stages(base_stage, overlay_stage, "/World/Merged")
    assert result is not None
    assert isinstance(result, Usd.Stage)

    # Verify base content is preserved
    assert result.GetPrimAtPath("/World").IsValid()
    assert result.GetPrimAtPath("/World/Base").IsValid()

    # Verify overlay content is merged under the specified path
    assert result.GetPrimAtPath("/World/Merged").IsValid()
    # The merge should have created the parent prim for the overlay content
    merged_prim = result.GetPrimAtPath("/World/Merged")
    assert merged_prim.IsValid()


def test_merge_stages_copies_authored_data_and_creates_nested_target():
    base_stage = Usd.Stage.CreateInMemory()
    overlay_stage = Usd.Stage.CreateInMemory()
    root = overlay_stage.DefinePrim("/Overlay", "Xform")
    child = overlay_stage.DefinePrim("/Overlay/Child", "Xform")
    attr = root.CreateAttribute("user:label", Sdf.ValueTypeNames.String)
    attr.Set("default")
    attr.Set("animated", Usd.TimeCode(1))
    root.CreateRelationship("user:child").SetTargets([child.GetPath()])

    merged = merge_stages(base_stage, overlay_stage, "/World/Nested")
    copied = merged.GetPrimAtPath("/World/Nested/Overlay")

    assert merged.GetPrimAtPath("/World").IsValid()
    assert merged.GetPrimAtPath("/World/Nested").IsValid()
    copied_attr = copied.GetAttribute("user:label")
    assert copied_attr.Get() == "default"
    assert copied_attr.Get(Usd.TimeCode(1)) == "animated"
    assert copied_attr.GetTimeSamples() == [1.0]
    assert copied.GetRelationship("user:child").GetTargets() == [
        Sdf.Path("/Overlay/Child")
    ]
    assert merged.GetPrimAtPath("/World/Nested/Overlay/Child").IsValid()


def test_merge_stages_can_copy_overlay_at_root():
    base_stage = Usd.Stage.CreateInMemory()
    overlay_stage = Usd.Stage.CreateInMemory()
    overlay_stage.DefinePrim("/Overlay", "Xform")

    merged = merge_stages(base_stage, overlay_stage, "/")

    assert merged.GetPrimAtPath("/Overlay").IsValid()


def test_remove_animation_basic():
    """Test basic functionality of remove_animation."""
    stage = Usd.Stage.CreateInMemory()

    # Create an Xform prim with animated translate attribute
    xform = UsdGeom.Xform.Define(stage, "/AnimatedPrim")
    translate_op = xform.AddTranslateOp()

    # Set time-sampled values
    translate_op.Set((0.0, 0.0, 0.0), Usd.TimeCode(0))
    translate_op.Set((10.0, 10.0, 10.0), Usd.TimeCode(24))
    translate_op.Set((20.0, 20.0, 20.0), Usd.TimeCode(48))

    # Verify animation exists
    attr = xform.GetPrim().GetAttribute("xformOp:translate")
    assert attr.GetNumTimeSamples() == 3

    # Remove animation (defaults to time 0 since no start time code set)
    num_removed = remove_animation(stage)

    # Verify animation was removed
    assert num_removed == 1
    assert attr.GetNumTimeSamples() == 0

    # Verify static value is set (sampled from time 0)
    value = attr.Get()
    assert value is not None
    assert value == (0.0, 0.0, 0.0)


def test_remove_animation_custom_reference_time():
    """Test remove_animation with custom reference time."""
    stage = Usd.Stage.CreateInMemory()

    # Create an Xform prim with animated translate attribute
    xform = UsdGeom.Xform.Define(stage, "/AnimatedPrim")
    translate_op = xform.AddTranslateOp()

    # Set time-sampled values
    translate_op.Set((0.0, 0.0, 0.0), Usd.TimeCode(0))
    translate_op.Set((10.0, 10.0, 10.0), Usd.TimeCode(24))
    translate_op.Set((20.0, 20.0, 20.0), Usd.TimeCode(48))

    # Remove animation, sampling at time 24
    num_removed = remove_animation(stage, reference_time=Usd.TimeCode(24))

    # Verify animation was removed and value is from time 24
    attr = xform.GetPrim().GetAttribute("xformOp:translate")
    assert num_removed == 1
    assert attr.GetNumTimeSamples() == 0
    value = attr.Get()
    assert value == (10.0, 10.0, 10.0)


def test_remove_animation_default_time_code():
    """Test that remove_animation uses stage.GetStartTimeCode() or time 0 by default."""
    stage = Usd.Stage.CreateInMemory()

    # Set stage start time code to test that it's used as the default
    stage.SetStartTimeCode(24)

    # Create an Xform prim with animated translate attribute
    xform = UsdGeom.Xform.Define(stage, "/AnimatedPrim")
    translate_op = xform.AddTranslateOp()

    # Set time-sampled values at different times
    translate_op.Set((5.0, 5.0, 5.0), Usd.TimeCode(0))
    translate_op.Set((15.0, 15.0, 15.0), Usd.TimeCode(24))

    # Remove animation without specifying reference_time (uses stage start time)
    num_removed = remove_animation(stage)

    # Verify animation was removed
    attr = xform.GetPrim().GetAttribute("xformOp:translate")
    assert num_removed == 1
    assert attr.GetNumTimeSamples() == 0
    # Should use value from stage start time (24), which is (15, 15, 15)
    value = attr.Get()
    assert value is not None
    assert value == (15.0, 15.0, 15.0)


def test_remove_animation_skips_instance_proxy_like_prims():
    class ProxyPrim:
        def IsInstanceProxy(self):
            return True

        def GetAttributes(self):
            raise AssertionError("instance proxies should be skipped")

    class FakeStage:
        def GetStartTimeCode(self):
            return Usd.TimeCode.Default()

        def TraverseAll(self):
            return [ProxyPrim()]

    assert remove_animation(FakeStage()) == 0


def test_remove_animation_multiple_attributes():
    """Test remove_animation with multiple animated attributes."""
    stage = Usd.Stage.CreateInMemory()

    # Create multiple prims with animation
    xform1 = UsdGeom.Xform.Define(stage, "/Prim1")
    translate_op1 = xform1.AddTranslateOp()
    translate_op1.Set((0.0, 0.0, 0.0), Usd.TimeCode(0))
    translate_op1.Set((10.0, 10.0, 10.0), Usd.TimeCode(24))

    xform2 = UsdGeom.Xform.Define(stage, "/Prim2")
    translate_op2 = xform2.AddTranslateOp()
    translate_op2.Set((5.0, 5.0, 5.0), Usd.TimeCode(0))
    translate_op2.Set((15.0, 15.0, 15.0), Usd.TimeCode(24))

    rotate_op2 = xform2.AddRotateXYZOp()
    rotate_op2.Set((0.0, 0.0, 0.0), Usd.TimeCode(0))
    rotate_op2.Set((90.0, 0.0, 0.0), Usd.TimeCode(24))

    # Remove all animation
    num_removed = remove_animation(stage)

    # Verify all animations were removed (3 animated attributes)
    assert num_removed == 3

    # Verify all attributes are now static
    attr1 = xform1.GetPrim().GetAttribute("xformOp:translate")
    attr2 = xform2.GetPrim().GetAttribute("xformOp:translate")
    attr3 = xform2.GetPrim().GetAttribute("xformOp:rotateXYZ")
    assert attr1.GetNumTimeSamples() == 0
    assert attr2.GetNumTimeSamples() == 0
    assert attr3.GetNumTimeSamples() == 0


def test_remove_animation_no_animation():
    """Test remove_animation when stage has no animation."""
    stage = Usd.Stage.CreateInMemory()

    # Create a static prim with no animation
    xform = UsdGeom.Xform.Define(stage, "/StaticPrim")
    translate_op = xform.AddTranslateOp()
    translate_op.Set((5.0, 5.0, 5.0))  # Static value, no time code

    # Remove animation (should find nothing)
    num_removed = remove_animation(stage)

    # Verify no attributes were modified
    assert num_removed == 0

    # Verify static value is preserved
    attr = xform.GetPrim().GetAttribute("xformOp:translate")
    assert attr.Get() == (5.0, 5.0, 5.0)


def test_remove_animation_returns_correct_count():
    """Test that remove_animation returns accurate count of modified attributes."""
    stage = Usd.Stage.CreateInMemory()

    # Create prims: 2 animated, 1 static
    xform1 = UsdGeom.Xform.Define(stage, "/Animated1")
    op1 = xform1.AddTranslateOp()
    op1.Set((0.0, 0.0, 0.0), Usd.TimeCode(0))
    op1.Set((10.0, 10.0, 10.0), Usd.TimeCode(24))

    xform2 = UsdGeom.Xform.Define(stage, "/Animated2")
    op2 = xform2.AddScaleOp()
    op2.Set((1.0, 1.0, 1.0), Usd.TimeCode(0))
    op2.Set((2.0, 2.0, 2.0), Usd.TimeCode(24))

    xform3 = UsdGeom.Xform.Define(stage, "/Static")
    op3 = xform3.AddTranslateOp()
    op3.Set((5.0, 5.0, 5.0))  # Static, no time samples

    # Remove animation
    num_removed = remove_animation(stage)

    # Should only count the 2 animated attributes
    assert num_removed == 2
