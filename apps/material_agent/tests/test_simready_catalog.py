# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for lightweight SimReady material catalog selection."""

import hashlib
import http.client
import stat
import urllib.error
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from material_agent.simready import (
    SimReadyCatalogError,
    build_material_entries,
    is_simready_library_id,
    load_default_manifest,
    load_manifest,
    parse_simready_library_id,
)
from material_agent.simready.catalog import category_names
from material_agent.simready.hydration import hydrate_simready_library


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_default_manifest_counts_match_v0_2_0_release() -> None:
    manifest = load_default_manifest()

    assert manifest["release_tag"] == "v0.2.0"
    assert manifest["generated_at"] == "2026-06-13T00:00:00Z"
    assert len(manifest["materials"]) == 1705
    assert len(manifest["libraries"]["simready-light"]["material_ids"]) == 265


def test_simready_light_is_curated_material_subset() -> None:
    manifest = load_default_manifest()

    entries = build_material_entries(manifest, "simready-light")
    categories = {entry["simready_category"] for entry in entries}

    assert 200 <= len(entries) <= 300
    assert "Leather" not in categories
    assert {"Metal", "Plastic", "Glass", "Paint"}.issubset(categories)


def test_simready_full_index_excludes_split_archives_by_default() -> None:
    manifest = load_default_manifest()

    entries = build_material_entries(manifest, "simready-full")
    categories = {entry["simready_category"] for entry in entries}

    assert len(entries) == 1705 - 176
    assert "Leather" not in categories

    entries_with_opt_in = build_material_entries(
        manifest,
        "simready-full",
        split_archives_enabled=True,
    )
    assert len(entries_with_opt_in) == len(entries)
    assert "Leather" not in {
        entry["simready_category"] for entry in entries_with_opt_in
    }


def test_simready_category_selection_is_case_insensitive() -> None:
    manifest = load_default_manifest()

    entries = build_material_entries(manifest, "simready-category:metal")

    assert len(entries) == 609
    assert {entry["simready_category"] for entry in entries} == {"Metal"}


def test_split_category_is_rejected_until_hydration_supports_it() -> None:
    manifest = load_default_manifest()

    with pytest.raises(SimReadyCatalogError, match="not supported"):
        build_material_entries(manifest, "simready-category:Leather")

    with pytest.raises(SimReadyCatalogError, match="not supported"):
        build_material_entries(
            manifest,
            "simready-category:Leather",
            split_archives_enabled=True,
        )


def test_hydrate_split_category_rejects_enabled_multi_file_archives(
    tmp_path: Path,
) -> None:
    entry = {
        "name": "Leather Test",
        "description": "Synthetic split archive material.",
        "binding": "/World/Looks/Leather_Test",
        "simready_category": "Leather",
        "simready_source_path": "Materials/Leather/Leather_Test.usda",
    }
    manifest = {
        "schema_version": 1,
        "repository": "example/simready",
        "release_tag": "v-test",
        "categories": {
            "Leather": {
                "archive_files": [
                    {"name": "Leather.z01", "url": "file:///missing", "sha256": "0"},
                    {"name": "Leather.zip", "url": "file:///missing", "sha256": "0"},
                ],
                "material_count": 1,
                "requires_split_archive": True,
            }
        },
    }

    with pytest.raises(SimReadyCatalogError, match="multi-file hydration"):
        hydrate_simready_library(
            manifest=manifest,
            entries=[entry],
            material_names={"Leather Test"},
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "out",
            split_archives_enabled=True,
        )


def test_allowed_categories_filter_simready_views() -> None:
    manifest = load_default_manifest()

    entries = build_material_entries(
        manifest,
        "simready-light",
        allowed_categories={"Plastic"},
    )

    assert len(entries) == 45
    assert {entry["simready_category"] for entry in entries} == {"Plastic"}


def test_parse_simready_category_requires_category() -> None:
    with pytest.raises(SimReadyCatalogError, match="missing"):
        parse_simready_library_id("simready-category:")


def test_simready_id_predicate_custom_manifest_and_category_names(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "simready.yaml"
    manifest_path.write_text(
        """
categories:
  Plastic: {}
  Metal: {}
materials: []
""".strip(),
        encoding="utf-8",
    )
    bad_manifest_path = tmp_path / "bad.yaml"
    bad_manifest_path.write_text("- not\n- a mapping\n", encoding="utf-8")

    assert is_simready_library_id(None) is False
    assert is_simready_library_id(" simready-light ") is True
    assert is_simready_library_id("simready-category:Plastic") is True
    assert is_simready_library_id("custom") is False
    assert category_names(load_manifest(manifest_path)) == ["Metal", "Plastic"]
    with pytest.raises(SimReadyCatalogError, match="not a mapping"):
        load_manifest(bad_manifest_path)


def test_manifest_schema_validation_rejects_bad_category_metadata() -> None:
    manifest = {"categories": {"Metal": []}, "materials": []}

    with pytest.raises(SimReadyCatalogError, match="Category metadata"):
        build_material_entries(manifest, "simready-full")


def test_manifest_schema_validation_rejects_bad_material_entries() -> None:
    manifest = {
        "categories": {"Metal": {"requires_split_archive": False}},
        "materials": [{"name": "Incomplete"}],
    }

    with pytest.raises(SimReadyCatalogError, match="missing required keys"):
        build_material_entries(manifest, "simready-full")


def test_hydrate_simready_library_uses_selected_material_only(tmp_path: Path) -> None:
    from pxr import Usd

    archive_path = tmp_path / "Plastic.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(
            "Materials/Plastic/Plastic_Test.usda",
            """#usda 1.0
(
    defaultPrim = "Plastic_Test"
)

def Material "Plastic_Test"
{
    color3f inputs:base_color = (0.5, 0.5, 0.5)
    asset inputs:base_color_texture_file = @./textures/plastic_test.png@
}
""",
        )
        zf.writestr("Materials/Plastic/textures/plastic_test.png", b"texture")

    entry = {
        "name": "Plastic Test",
        "description": "Synthetic SimReady plastic material.",
        "binding": "/World/Looks/Plastic_Test",
        "simready_category": "Plastic",
        "simready_source_path": "Materials/Plastic/Plastic_Test.usda",
    }
    manifest = {
        "schema_version": 1,
        "repository": "example/simready",
        "release_tag": "v-test",
        "categories": {
            "Plastic": {
                "archive_files": [
                    {
                        "name": archive_path.name,
                        "url": archive_path.resolve().as_uri(),
                        "sha256": _sha256(archive_path),
                        "size": archive_path.stat().st_size,
                    }
                ],
                "material_count": 1,
                "requires_split_archive": False,
            }
        },
    }

    hydrated = hydrate_simready_library(
        manifest=manifest,
        entries=[entry],
        material_names={"Plastic Test"},
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )

    assert hydrated.library_path.exists()
    assert hydrated.entries == [
        {
            "name": "Plastic Test",
            "description": "Synthetic SimReady plastic material.",
            "binding": "/World/Looks/Plastic_Test",
        }
    ]
    assert hydrated.report["categories"] == ["Plastic"]
    stage = Usd.Stage.Open(str(hydrated.library_path))
    assert stage is not None
    assert stage.GetPrimAtPath("/World/Looks/Plastic_Test").IsValid()
    assert (
        tmp_path / "out" / "assets" / "Plastic_Test" / "textures" / "plastic_test.png"
    ).exists()


def test_hydrate_simready_library_scopes_assets_by_binding_path(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "Plastic.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        for group, payload in (("Alpha", b"alpha"), ("Beta", b"beta")):
            zf.writestr(
                f"Materials/Plastic/{group}/Shared.usda",
                """#usda 1.0
(
    defaultPrim = "Shared"
)

def Material "Shared"
{
    asset inputs:base_color_texture_file = @./textures/albedo.png@
}
""",
            )
            zf.writestr(
                f"Materials/Plastic/{group}/textures/albedo.png",
                payload,
            )

    entries = [
        {
            "name": "Plastic Alpha",
            "description": "Alpha material.",
            "binding": "/World/Looks/Alpha/Shared",
            "simready_category": "Plastic",
            "simready_source_path": "Materials/Plastic/Alpha/Shared.usda",
        },
        {
            "name": "Plastic Beta",
            "description": "Beta material.",
            "binding": "/World/Looks/Beta/Shared",
            "simready_category": "Plastic",
            "simready_source_path": "Materials/Plastic/Beta/Shared.usda",
        },
    ]
    manifest = {
        "schema_version": 1,
        "repository": "example/simready",
        "release_tag": "v-test",
        "categories": {
            "Plastic": {
                "archive_files": [
                    {
                        "name": archive_path.name,
                        "url": archive_path.resolve().as_uri(),
                        "sha256": _sha256(archive_path),
                        "size": archive_path.stat().st_size,
                    }
                ],
                "material_count": 2,
                "requires_split_archive": False,
            }
        },
    }

    hydrated = hydrate_simready_library(
        manifest=manifest,
        entries=entries,
        material_names={"Plastic Alpha", "Plastic Beta"},
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )

    alpha_texture = (
        tmp_path / "out" / "assets" / "Alpha" / "Shared" / "textures" / "albedo.png"
    )
    beta_texture = (
        tmp_path / "out" / "assets" / "Beta" / "Shared" / "textures" / "albedo.png"
    )
    assert alpha_texture.read_bytes() == b"alpha"
    assert beta_texture.read_bytes() == b"beta"

    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(hydrated.library_path))
    assert layer is not None
    alpha_prim = layer.GetPrimAtPath("/World/Looks/Alpha/Shared")
    beta_prim = layer.GetPrimAtPath("/World/Looks/Beta/Shared")
    assert alpha_prim.attributes["inputs:base_color_texture_file"].default.path == (
        "assets/Alpha/Shared/textures/albedo.png"
    )
    assert beta_prim.attributes["inputs:base_color_texture_file"].default.path == (
        "assets/Beta/Shared/textures/albedo.png"
    )


def test_hydrate_simready_library_remaps_time_sampled_asset_paths(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    source_file = tmp_path / "Animated.usdc"
    source_stage = Usd.Stage.CreateNew(str(source_file))
    material = UsdShade.Material.Define(source_stage, "/Animated")
    source_stage.SetDefaultPrim(material.GetPrim())
    texture_input = material.CreateInput(
        "base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    )
    texture_input.Set(Sdf.AssetPath("./textures/default.png"))
    texture_input.GetAttr().Set(Sdf.AssetPath("./textures/frame_001.png"), 1.0)
    source_stage.GetRootLayer().Save()

    archive_path = tmp_path / "Plastic.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.write(source_file, "Materials/Plastic/Animated.usdc")
        zf.writestr("Materials/Plastic/textures/default.png", b"default")
        zf.writestr("Materials/Plastic/textures/frame_001.png", b"frame")

    entry = {
        "name": "Plastic Animated",
        "description": "Animated texture material.",
        "binding": "/World/Looks/Plastic_Animated",
        "simready_category": "Plastic",
        "simready_source_path": "Materials/Plastic/Animated.usdc",
    }
    manifest = {
        "schema_version": 1,
        "repository": "example/simready",
        "release_tag": "v-test",
        "categories": {
            "Plastic": {
                "archive_files": [
                    {
                        "name": archive_path.name,
                        "url": archive_path.resolve().as_uri(),
                        "sha256": _sha256(archive_path),
                        "size": archive_path.stat().st_size,
                    }
                ],
                "material_count": 1,
                "requires_split_archive": False,
            }
        },
    }

    hydrated = hydrate_simready_library(
        manifest=manifest,
        entries=[entry],
        material_names={"Plastic Animated"},
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )

    assert (
        tmp_path / "out" / "assets" / "Plastic_Animated" / "textures" / "default.png"
    ).exists()
    assert (
        tmp_path / "out" / "assets" / "Plastic_Animated" / "textures" / "frame_001.png"
    ).exists()

    layer = Sdf.Layer.FindOrOpen(str(hydrated.library_path))
    assert layer is not None
    hydrated_prim = layer.GetPrimAtPath("/World/Looks/Plastic_Animated")
    hydrated_attr = hydrated_prim.attributes["inputs:base_color_texture_file"]
    assert hydrated_attr.default.path == (
        "assets/Plastic_Animated/textures/default.png"
    )
    assert hydrated_attr.QueryTimeSample(1.0).path == (
        "assets/Plastic_Animated/textures/frame_001.png"
    )


def test_hydrate_simready_library_reports_skipped_asset_paths(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    archive_path = tmp_path / "Plastic.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(
            "Materials/Plastic/Plastic_Test.usda",
            """#usda 1.0
(
    defaultPrim = "Plastic_Test"
)

def Material "Plastic_Test"
{
    asset inputs:base_color_texture_file = @../shared/shared.png@
}
""",
        )
        zf.writestr("Materials/shared/shared.png", b"shared")

    entry = {
        "name": "Plastic Test",
        "description": "Synthetic SimReady plastic material.",
        "binding": "/World/Looks/Plastic_Test",
        "simready_category": "Plastic",
        "simready_source_path": "Materials/Plastic/Plastic_Test.usda",
    }
    manifest = {
        "schema_version": 1,
        "repository": "example/simready",
        "release_tag": "v-test",
        "categories": {
            "Plastic": {
                "archive_files": [
                    {
                        "name": archive_path.name,
                        "url": archive_path.resolve().as_uri(),
                        "sha256": _sha256(archive_path),
                        "size": archive_path.stat().st_size,
                    }
                ],
                "material_count": 1,
                "requires_split_archive": False,
            }
        },
    }

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "simready_material_library.usda").write_text(
        "stale", encoding="utf-8"
    )
    listener = SimpleNamespace(info=lambda *_args: None, warning=lambda *_args: None)

    with caplog.at_level("WARNING", logger="material_agent.simready.hydration"):
        hydrated = hydrate_simready_library(
            manifest=manifest,
            entries=[entry],
            material_names={"Plastic Test"},
            cache_dir=tmp_path / "cache",
            output_dir=output_dir,
            listener=listener,
        )

    assert hydrated.report["skipped_asset_count"] == 1
    skipped_asset = hydrated.report["skipped_assets"][0]
    assert skipped_asset["material_binding"] == "/World/Looks/Plastic_Test"
    assert skipped_asset["path"].endswith("/Materials/shared/shared.png")
    assert skipped_asset["reason"] == "out-of-tree absolute path"
    assert "Skipping out-of-tree absolute path SimReady asset path" in caplog.text


def test_hydration_marker_and_response_timeout_helpers(tmp_path: Path) -> None:
    import material_agent.simready.hydration as hydration

    archive_path = tmp_path / "Plastic.zip"
    archive_path.write_bytes(b"archive")
    marker_path = archive_path.with_suffix(".zip.verified.json")

    marker_path.write_text("[]", encoding="utf-8")
    assert (
        hydration._archive_has_verified_marker(archive_path, marker_path, "digest")
        is False
    )

    marker_path.write_text("{", encoding="utf-8")
    assert (
        hydration._archive_has_verified_marker(archive_path, marker_path, "digest")
        is False
    )

    timeouts: list[float] = []

    class Socket:
        def settimeout(self, timeout: float) -> None:
            timeouts.append(timeout)

    hydration._set_response_read_timeout(SimpleNamespace(fp=Socket()), 3.0)
    assert timeouts == [3.0]

    cyclic = SimpleNamespace()
    cyclic.raw = cyclic
    hydration._set_response_read_timeout(cyclic, 3.0)
    hydration._set_response_read_timeout(SimpleNamespace(), 3.0)


def test_copy_url_to_path_local_path_and_successful_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import material_agent.simready.hydration as hydration

    source = tmp_path / "source.zip"
    source.write_bytes(b"local")
    destination = tmp_path / "local-copy.zip"

    hydration._copy_url_to_path(str(source), destination, max_bytes=10)
    assert destination.read_bytes() == b"local"

    class Response:
        def __init__(self) -> None:
            self._chunks = [b"http", b""]

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def settimeout(self, _timeout: float) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return self._chunks.pop(0)

    monkeypatch.setattr(hydration.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        hydration.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    http_destination = tmp_path / "http-copy.zip"

    hydration._copy_url_to_path(
        "https://example.invalid/Plastic.zip",
        http_destination,
        timeout=5,
    )

    assert http_destination.read_bytes() == b"http"


def test_ensure_archive_reuses_matching_archive_without_marker(tmp_path: Path) -> None:
    import material_agent.simready.hydration as hydration

    archive_dir = tmp_path / "cache" / "repo" / "v-test" / "Plastic" / "archives"
    archive_dir.mkdir(parents=True)
    archive_path = archive_dir / "Plastic.zip"
    archive_path.write_bytes(b"already cached")
    expected_digest = _sha256(archive_path)

    result = hydration._ensure_archive(
        {
            "name": "Plastic.zip",
            "url": "https://example.invalid/Plastic.zip",
            "sha256": expected_digest,
        },
        archive_dir,
    )

    assert result == archive_path
    assert archive_path.with_suffix(".zip.verified.json").exists()


def test_hydrate_category_rejects_enabled_multi_archive_without_category_flag(
    tmp_path: Path,
) -> None:
    import material_agent.simready.hydration as hydration

    manifest = {
        "repository": "example/simready",
        "release_tag": "v-test",
        "categories": {
            "Plastic": {
                "archive_files": [
                    {"name": "Plastic.z01", "url": "file:///missing", "sha256": "0"},
                    {"name": "Plastic.zip", "url": "file:///missing", "sha256": "0"},
                ],
                "requires_split_archive": False,
            }
        },
    }

    with pytest.raises(SimReadyCatalogError, match="multi-file hydration"):
        hydration._hydrate_category(
            manifest,
            "Plastic",
            tmp_path / "cache",
            split_archives_enabled=True,
        )


def test_asset_path_remap_helper_edges(tmp_path: Path) -> None:
    from pxr import Sdf

    import material_agent.simready.hydration as hydration

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "tex.png").write_bytes(b"texture")
    library_dir = tmp_path / "library"
    asset_dir = library_dir / "assets" / "Material"
    skipped: list[dict[str, str]] = []

    assert (
        hydration._copy_and_remap_asset_path(
            "",
            source_dir=source_dir,
            library_dir=library_dir,
            asset_dir=asset_dir,
            material_binding="/World/Looks/Material",
            skipped_assets=skipped,
        )
        == ""
    )
    assert (
        hydration._copy_and_remap_asset_path(
            "https://example.invalid/tex.png",
            source_dir=source_dir,
            library_dir=library_dir,
            asset_dir=asset_dir,
            material_binding="/World/Looks/Material",
            skipped_assets=skipped,
        )
        == ""
    )
    assert (
        hydration._copy_and_remap_asset_path(
            "../outside.png",
            source_dir=source_dir,
            library_dir=library_dir,
            asset_dir=asset_dir,
            material_binding="/World/Looks/Material",
            skipped_assets=skipped,
        )
        == ""
    )
    assert (
        hydration._copy_and_remap_asset_path(
            "missing.png",
            source_dir=source_dir,
            library_dir=library_dir,
            asset_dir=asset_dir,
            material_binding="/World/Looks/Material",
            skipped_assets=skipped,
        )
        == ""
    )

    remapped_array = hydration._copy_and_remap_asset_value(
        Sdf.AssetPathArray([Sdf.AssetPath("tex.png")]),
        source_dir=source_dir,
        library_dir=library_dir,
        asset_dir=asset_dir,
        material_binding="/World/Looks/Material",
        skipped_assets=skipped,
    )

    assert remapped_array[0].path == "assets/Material/tex.png"
    assert (asset_dir / "tex.png").read_bytes() == b"texture"

    layer = Sdf.Layer.CreateAnonymous()
    hydration._copy_and_remap_asset_paths_in_prim(
        layer,
        Sdf.Path("/Missing"),
        source_dir=source_dir,
        library_dir=library_dir,
        asset_dir=asset_dir,
        material_binding="/World/Looks/Material",
        skipped_assets=skipped,
    )

    parent = Sdf.CreatePrimInLayer(layer, "/Parent")
    child = Sdf.CreatePrimInLayer(layer, "/Parent/Child")
    attr = Sdf.AttributeSpec(child, "inputs:file", Sdf.ValueTypeNames.Asset)
    attr.default = Sdf.AssetPath("tex.png")
    hydration._copy_and_remap_asset_paths_in_prim(
        layer,
        Sdf.Path("/Parent"),
        source_dir=source_dir,
        library_dir=library_dir,
        asset_dir=asset_dir,
        material_binding="/World/Looks/Material",
        skipped_assets=skipped,
    )

    assert parent.nameChildren["Child"].attributes["inputs:file"].default.path == (
        "assets/Material/tex.png"
    )

    class EmptyTargetPath:
        name = ""

        def __str__(self) -> str:
            return ""

    assert hydration._asset_dir_for_material_binding(
        library_dir, EmptyTargetPath()
    ) == (library_dir.resolve() / "assets" / "material")


def test_ensure_archive_reuses_verified_marker_without_rehashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import material_agent.simready.hydration as hydration

    source_path = tmp_path / "source" / "Plastic.zip"
    source_path.parent.mkdir()
    source_path.write_bytes(b"verified archive")
    expected_digest = _sha256(source_path)
    archive_dir = tmp_path / "cache" / "repo" / "v-test" / "Plastic" / "archives"
    asset = {
        "name": "Plastic.zip",
        "url": source_path.resolve().as_uri(),
        "sha256": expected_digest,
    }

    archive_path = hydration._ensure_archive(asset, archive_dir)
    assert archive_path.exists()
    assert archive_path.with_suffix(".zip.verified.json").exists()

    def fail_sha256(_path: Path) -> str:
        pytest.fail("verified archive should not be rehashed")

    monkeypatch.setattr(hydration, "_sha256", fail_sha256)

    assert hydration._ensure_archive(asset, archive_dir) == archive_path


def test_ensure_archive_rejects_downloaded_digest_mismatch(
    tmp_path: Path,
) -> None:
    import material_agent.simready.hydration as hydration

    source_path = tmp_path / "source" / "Plastic.zip"
    source_path.parent.mkdir()
    with zipfile.ZipFile(source_path, "w") as zf:
        zf.writestr("Materials/Plastic/Plastic_Test.usda", "#usda 1.0\n")
    archive_dir = tmp_path / "cache" / "repo" / "v-test" / "Plastic" / "archives"
    archive_path = archive_dir / "Plastic.zip"
    asset = {
        "name": "Plastic.zip",
        "url": source_path.resolve().as_uri(),
        "sha256": "0" * 64,
        "size": source_path.stat().st_size,
    }

    with pytest.raises(SimReadyCatalogError, match="Digest mismatch"):
        hydration._ensure_archive(asset, archive_dir)

    assert not archive_path.exists()
    assert not archive_path.with_suffix(".zip.partial").exists()
    assert not archive_path.with_suffix(".zip.verified.json").exists()


def test_ensure_archive_rejects_manifest_archive_name_traversal(
    tmp_path: Path,
) -> None:
    import material_agent.simready.hydration as hydration

    source_path = tmp_path / "source" / "Plastic.zip"
    source_path.parent.mkdir()
    source_path.write_bytes(b"archive")
    archive_dir = tmp_path / "cache" / "repo" / "v-test" / "Plastic" / "archives"
    asset = {
        "name": "../Plastic.zip",
        "url": source_path.resolve().as_uri(),
        "sha256": _sha256(source_path),
    }

    with pytest.raises(SimReadyCatalogError, match="Unsafe archive name"):
        hydration._ensure_archive(asset, archive_dir)

    assert not (archive_dir.parent / "Plastic.zip").exists()


def test_copy_url_to_path_enforces_total_download_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import material_agent.simready.hydration as hydration

    read_timeouts: list[float] = []

    class SlowResponse:
        def __enter__(self) -> "SlowResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def settimeout(self, timeout: float) -> None:
            read_timeouts.append(timeout)

        def read(self, _size: int) -> bytes:
            return b"x"

    times = iter([0.0, 0.0, 301.0])
    monkeypatch.setattr(hydration.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        hydration.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: SlowResponse(),
    )

    with pytest.raises(SimReadyCatalogError, match="timed out"):
        hydration._copy_url_to_path(
            "https://example.invalid/Plastic.zip",
            tmp_path / "Plastic.zip",
        )
    assert read_timeouts == [300.0]


def test_copy_url_to_path_enforces_max_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import material_agent.simready.hydration as hydration

    class LargeResponse:
        def __enter__(self) -> "LargeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def settimeout(self, _timeout: float) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return b"oversized"

    monkeypatch.setattr(
        hydration.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: LargeResponse(),
    )

    with pytest.raises(SimReadyCatalogError, match="size limit"):
        hydration._copy_url_to_path(
            "https://example.invalid/Plastic.zip",
            tmp_path / "Plastic.zip",
            max_bytes=4,
        )


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.invalid/Plastic.zip",
        "local:/tmp/Plastic.zip",
    ],
)
def test_copy_url_to_path_rejects_unsupported_scheme(
    tmp_path: Path,
    url: str,
) -> None:
    import material_agent.simready.hydration as hydration

    with pytest.raises(SimReadyCatalogError, match="Unsupported.*URL scheme"):
        hydration._copy_url_to_path(url, tmp_path / "x")


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    import material_agent.simready.hydration as hydration

    archive_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("../escape.txt", b"owned")

    with pytest.raises(SimReadyCatalogError, match="Unsafe path"):
        hydration._safe_extract(archive_path, tmp_path / "out")

    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_rejects_symlinks(tmp_path: Path) -> None:
    import material_agent.simready.hydration as hydration

    archive_path = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("Materials/Plastic/link.png")
    info.create_system = 3
    info.external_attr = stat.S_IFLNK << 16
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(info, "../shared/texture.png")

    with pytest.raises(SimReadyCatalogError, match="Unsupported symlink"):
        hydration._safe_extract(archive_path, tmp_path / "out")


def test_hydrate_simready_library_rejects_manifest_source_path_traversal(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "Plastic.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("Materials/Plastic/Plastic_Test.usda", "#usda 1.0\n")

    entry = {
        "name": "Plastic Test",
        "description": "Synthetic SimReady plastic material.",
        "binding": "/World/Looks/Plastic_Test",
        "simready_category": "Plastic",
        "simready_source_path": "../escape.usda",
    }
    manifest = {
        "schema_version": 1,
        "repository": "example/simready",
        "release_tag": "v-test",
        "categories": {
            "Plastic": {
                "archive_files": [
                    {
                        "name": archive_path.name,
                        "url": archive_path.resolve().as_uri(),
                        "sha256": _sha256(archive_path),
                        "size": archive_path.stat().st_size,
                    }
                ],
                "material_count": 1,
                "requires_split_archive": False,
            }
        },
    }

    with pytest.raises(SimReadyCatalogError, match="Unsafe material source path"):
        hydrate_simready_library(
            manifest=manifest,
            entries=[entry],
            material_names={"Plastic Test"},
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "out",
        )


@pytest.mark.parametrize(
    "download_error",
    [
        urllib.error.URLError("network unavailable"),
        http.client.IncompleteRead(b"partial"),
        ValueError("unknown url type"),
    ],
)
def test_hydrate_simready_library_wraps_download_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    download_error: Exception,
) -> None:
    import material_agent.simready.hydration as hydration

    def raise_url_error(*_args: object, **_kwargs: object) -> None:
        raise download_error

    monkeypatch.setattr(hydration.urllib.request, "urlopen", raise_url_error)
    entry = {
        "name": "Plastic Test",
        "description": "Synthetic SimReady plastic material.",
        "binding": "/World/Looks/Plastic_Test",
        "simready_category": "Plastic",
        "simready_source_path": "Materials/Plastic/Plastic_Test.usda",
    }
    manifest = {
        "schema_version": 1,
        "repository": "example/simready",
        "release_tag": "v-test",
        "categories": {
            "Plastic": {
                "archive_files": [
                    {
                        "name": "Plastic.zip",
                        "url": "https://example.invalid/Plastic.zip",
                        "sha256": "0" * 64,
                        "size": 1,
                    }
                ],
                "material_count": 1,
                "requires_split_archive": False,
            }
        },
    }

    with pytest.raises(SimReadyCatalogError, match="Failed to download"):
        hydrate_simready_library(
            manifest=manifest,
            entries=[entry],
            material_names={"Plastic Test"},
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "out",
        )


def test_archive_lock_timeout_preserves_partial_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import material_agent.simready.hydration as hydration

    class TimeoutLock:
        def __enter__(self) -> None:
            raise hydration.Timeout("locked")

        def __exit__(self, *_args: object) -> None:
            return None

    archive_dir = tmp_path / "cache" / "repo" / "v-test" / "Plastic" / "archives"
    archive_dir.mkdir(parents=True)
    partial_path = archive_dir / "Plastic.zip.partial"
    partial_path.write_bytes(b"in-flight download")

    monkeypatch.setattr(hydration, "_with_file_lock", lambda _lock_path: TimeoutLock())

    with pytest.raises(SimReadyCatalogError, match="archive lock"):
        hydration._ensure_archive(
            {
                "name": "Plastic.zip",
                "url": "https://example.invalid/Plastic.zip",
                "sha256": "0" * 64,
            },
            archive_dir,
        )

    assert partial_path.read_bytes() == b"in-flight download"
