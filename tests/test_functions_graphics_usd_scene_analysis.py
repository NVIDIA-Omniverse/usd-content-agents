# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for usd_scene_analysis.detect_objects."""

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

from world_understanding.functions.graphics.usd_scene_analysis import (
    CandidateFeatures,
    _bound_material_identity,
    _build_mesh_ancestry_cache,
    _build_subtree_refs_cache,
    _canonical_instance_context_name,
    _classify_candidate,
    _compute_sibling_homogeneity_map,
    _find_content_root,
    _instance_base_name,
    _name_pattern_group_key,
    _resolve_overlaps,
    _semantic_instance_context,
    _surface_identity_group_key,
    detect_objects,
)
from world_understanding.utils.usd.composition import collect_composition_arcs
from world_understanding.utils.usd.prim import collect_mesh_geometry_stats


class _InvalidBoundMaterial:
    def GetPrim(self) -> Usd.Prim:
        return Usd.Prim()


class TestDetectObjects:
    """Integration tests for detect_objects."""

    def _make_stage_with_meshes(self):
        """Create an in-memory stage with Xforms and Meshes."""
        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())

        # A few objects with meshes
        UsdGeom.Xform.Define(stage, "/World/Car")
        mesh = UsdGeom.Mesh.Define(stage, "/World/Car/Body")
        mesh.GetPointsAttr().Set(
            Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
        )
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))

        UsdGeom.Xform.Define(stage, "/World/Tree")
        mesh2 = UsdGeom.Mesh.Define(stage, "/World/Tree/Trunk")
        mesh2.GetPointsAttr().Set(Vt.Vec3fArray([(2, 0, 0), (3, 0, 0), (3, 2, 0)]))
        mesh2.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]))
        mesh2.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2]))

        return stage

    def test_returns_two_lists(self):
        """detect_objects returns (objects, instance_groups) tuple."""
        stage = self._make_stage_with_meshes()
        comp = collect_composition_arcs(stage)
        geom = collect_mesh_geometry_stats(stage)

        objects, instance_groups = detect_objects(stage, comp, geom)

        assert isinstance(objects, list)
        assert isinstance(instance_groups, list)

    def test_detects_objects_from_simple_scene(self):
        """Objects are detected from a simple hierarchy."""
        stage = self._make_stage_with_meshes()
        comp = collect_composition_arcs(stage)
        geom = collect_mesh_geometry_stats(stage)

        objects, _ = detect_objects(stage, comp, geom)

        # Should find at least some objects
        assert len(objects) >= 1

        # Each object has required keys
        for obj in objects:
            assert "path" in obj
            assert "name" in obj
            assert "source_classification" in obj

    def test_empty_stage_returns_empty(self):
        """Empty stage returns no objects."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

        comp = collect_composition_arcs(stage)
        geom = collect_mesh_geometry_stats(stage)

        objects, instance_groups = detect_objects(stage, comp, geom)
        assert isinstance(objects, list)
        assert isinstance(instance_groups, list)

    def test_generic_numbered_nodes_group_by_semantic_parent(self):
        """Generic node#### names should not group unrelated parent assets."""
        forklift_key = _name_pattern_group_key(
            "/Store/Meshes/Arrangement_1/STORE/FORKLIFT__MFE_000007669/node1480",
            "node",
        )
        forklift_clone_key = _name_pattern_group_key(
            "/Store/Meshes/Arrangement_1/STORE/FORKLIFT__MFE_000007669_5/node1531",
            "node",
        )
        rack_key = _name_pattern_group_key(
            "/Store/Meshes/Arrangement_1/STORE/STORAGE_RACK__EQP_102351/node1471",
            "node",
        )

        assert forklift_key == forklift_clone_key
        assert forklift_key != rack_key


class _FakeRange:
    def GetMin(self) -> tuple[float, float, float]:
        return (0.0, 1.0, 2.0)

    def GetMax(self) -> tuple[float, float, float]:
        return (3.0, 4.0, 5.0)


class _FakeBBox:
    def ComputeAlignedRange(self) -> _FakeRange:
        return _FakeRange()


def _define_object(stage: Usd.Stage, path: str, mesh_name: str = "Mesh") -> None:
    UsdGeom.Xform.Define(stage, path)
    UsdGeom.Mesh.Define(stage, f"{path}/{mesh_name}")


def _define_materialized_quad(
    stage: Usd.Stage,
    path: str,
    material_path: str,
    *,
    two_faces: bool = False,
) -> None:
    UsdGeom.Xform.Define(stage, path)
    mesh = UsdGeom.Mesh.Define(stage, f"{path}/Body")
    mesh.GetPointsAttr().Set(
        Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    )
    if two_faces:
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3, 3]))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 0, 2, 3]))
    else:
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))

    material = UsdShade.Material.Define(stage, material_path)
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)


def _bind_material_subset(
    stage: Usd.Stage,
    mesh_path: str,
    subset_name: str,
    material_path: str,
    indices: list[int],
) -> None:
    material = UsdShade.Material.Define(stage, material_path)
    subset = UsdGeom.Subset.Define(stage, f"{mesh_path}/{subset_name}")
    subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
    subset.CreateFamilyNameAttr("materialBind")
    subset.CreateIndicesAttr(Vt.IntArray(indices))
    UsdShade.MaterialBindingAPI(subset.GetPrim()).Bind(material)


def test_scene_analysis_small_helper_branches() -> None:
    assert _canonical_instance_context_name("FORKLIFT__MFE_000007669_5") == (
        "FORKLIFT__MFE_000007669"
    )
    assert _canonical_instance_context_name("node_12") == "node_12"
    assert _semantic_instance_context("/World/Group_1/Node_2/Mesh001") is None
    assert _semantic_instance_context("/World/STORE/Widget/Mesh001") == "Widget"
    assert _name_pattern_group_key("/World/Car/body001", "body") == ("body", "Car")

    assert _instance_base_name("Bolt") is None
    assert _instance_base_name("Bolt__I12") == "Bolt"
    assert _instance_base_name("Bolt__12") == "Bolt"
    assert _instance_base_name("Bolt_12") == "Bolt"
    assert _instance_base_name("123") is None
    assert _surface_identity_group_key(Usd.Stage.CreateInMemory(), "/Missing") == ()
    assert _bound_material_identity(_InvalidBoundMaterial()) is None

    assert (
        _classify_candidate(
            CandidateFeatures(path="/A", subtree_ref_diversity=1, direct_ref_reuse=30),
            threshold=20,
        )
        == "building_block"
    )
    assert (
        _classify_candidate(
            CandidateFeatures(path="/A", rel_depth=1, direct_ref_reuse=0),
            threshold=20,
        )
        == "category"
    )
    assert (
        _classify_candidate(
            CandidateFeatures(path="/A", subtree_ref_diversity=2),
            threshold=20,
        )
        == "object_root"
    )
    assert (
        _classify_candidate(
            CandidateFeatures(path="/A", direct_ref_reuse=1),
            threshold=20,
        )
        == "object_root"
    )
    assert (
        _classify_candidate(
            CandidateFeatures(path="/A", has_skel_root=True),
            threshold=20,
        )
        == "object_root"
    )
    assert (
        _classify_candidate(
            CandidateFeatures(path="/A", has_mesh_descendants=True),
            threshold=20,
        )
        == "object_root"
    )
    assert (
        _classify_candidate(
            CandidateFeatures(path="/A", child_count=3, rel_depth=2),
            threshold=20,
        )
        == "category"
    )
    assert _classify_candidate(CandidateFeatures(path="/A"), threshold=20) == (
        "component"
    )

    resolved = _resolve_overlaps(
        {
            "/World/Block": "building_block",
            "/World/Block/Child": "object_root",
            "/World/Root": "object_root",
            "/World/Root/Child": "object_root",
        }
    )
    assert resolved["/World/Block/Child"] == "component"
    assert resolved["/World/Root/Child"] == "component"
    assert resolved["/World/Root"] == "object_root"


def test_scene_analysis_tree_cache_helpers() -> None:
    empty = Usd.Stage.CreateInMemory()
    leaf = UsdGeom.Xform.Define(empty, "/Leaf").GetPrim()
    assert _find_content_root(leaf, max_depth=0) == leaf
    assert _find_content_root(leaf) == leaf

    wide = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(wide, "/Root").GetPrim()
    for idx in range(6):
        UsdGeom.Xform.Define(wide, f"/Root/Child{idx}")
    assert _find_content_root(root) == root

    split = Usd.Stage.CreateInMemory()
    split_root = UsdGeom.Xform.Define(split, "/Root").GetPrim()
    for parent in ("A", "B"):
        UsdGeom.Xform.Define(split, f"/Root/{parent}")
        UsdGeom.Xform.Define(split, f"/Root/{parent}/One")
        UsdGeom.Xform.Define(split, f"/Root/{parent}/Two")
    assert _find_content_root(split_root) == split_root

    concentrated = Usd.Stage.CreateInMemory()
    concentrated_root = UsdGeom.Xform.Define(concentrated, "/Root").GetPrim()
    UsdGeom.Xform.Define(concentrated, "/Root/Wrapper")
    UsdGeom.Xform.Define(concentrated, "/Root/Wrapper/ContentA")
    UsdGeom.Xform.Define(concentrated, "/Root/Wrapper/ContentB")
    assert str(_find_content_root(concentrated_root).GetPath()) == "/Root/Wrapper"

    # A thin sibling assembly (single-child wrapper chain) that holds mesh
    # geometry must stop the descent, even when the dominant sibling has far
    # more grandchildren.
    thin_sibling = Usd.Stage.CreateInMemory()
    thin_root = UsdGeom.Xform.Define(thin_sibling, "/World").GetPrim()
    UsdGeom.Xform.Define(thin_sibling, "/World/Asset")
    UsdGeom.Xform.Define(thin_sibling, "/World/Asset/Body")
    for idx in range(4):
        UsdGeom.Mesh.Define(thin_sibling, f"/World/Asset/Body/Mesh{idx}")
    UsdGeom.Xform.Define(thin_sibling, "/World/Asset/Lift")
    UsdGeom.Xform.Define(thin_sibling, "/World/Asset/Lift/Top")
    UsdGeom.Mesh.Define(thin_sibling, "/World/Asset/Lift/Top/Mesh")
    assert str(_find_content_root(thin_root).GetPath()) == "/World/Asset"

    # The same thin-sibling-assembly shape, but rooted directly at the stage
    # pseudo-root (no default prim) instead of a named "/World" prim. The
    # pseudo-root has no usable path identity ("/"); returning it here (as
    # opposed to descending into the dominant asset) would make
    # detect_objects build a "//" prefix that matches no real prim path,
    # silently dropping every object in the scene.
    pseudo_root_stage = Usd.Stage.CreateInMemory()
    pseudo_root = pseudo_root_stage.GetPseudoRoot()
    UsdGeom.Xform.Define(pseudo_root_stage, "/Asset")
    UsdGeom.Xform.Define(pseudo_root_stage, "/Asset/Body")
    for idx in range(4):
        UsdGeom.Mesh.Define(pseudo_root_stage, f"/Asset/Body/Mesh{idx}")
    UsdGeom.Xform.Define(pseudo_root_stage, "/Asset/Lift")
    UsdGeom.Xform.Define(pseudo_root_stage, "/Asset/Lift/Top")
    UsdGeom.Mesh.Define(pseudo_root_stage, "/Asset/Lift/Top/Mesh")
    found_root = _find_content_root(pseudo_root)
    assert not found_root.IsPseudoRoot()
    assert str(found_root.GetPath()) == "/Asset"

    # Two top-level children with significant (but neither mesh-bearing nor
    # single-dominant) grandchild counts. Both are meaningful top-level
    # assemblies, so the pseudo-root itself must be kept as the content
    # root -- descending into only the busiest child ("/Dominant") would
    # silently drop every object under "/Minor". This is safe because
    # `_content_root_prefix()` makes `detect_objects` handle a "/" content
    # root correctly instead of building a "//" prefix that matches nothing.
    split_pseudo_root_stage = Usd.Stage.CreateInMemory()
    split_pseudo_root = split_pseudo_root_stage.GetPseudoRoot()
    UsdGeom.Xform.Define(split_pseudo_root_stage, "/Dominant")
    for idx in range(5):
        UsdGeom.Xform.Define(split_pseudo_root_stage, f"/Dominant/Child{idx}")
    UsdGeom.Xform.Define(split_pseudo_root_stage, "/Minor")
    for idx in range(2):
        UsdGeom.Xform.Define(split_pseudo_root_stage, f"/Minor/Child{idx}")
    split_found_root = _find_content_root(split_pseudo_root)
    assert split_found_root.IsPseudoRoot()

    mesh_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(mesh_stage, "/World")
    UsdGeom.Xform.Define(mesh_stage, "/World/Group")
    UsdGeom.Mesh.Define(mesh_stage, "/World/Group/Mesh")
    assert _build_mesh_ancestry_cache(mesh_stage.GetPrimAtPath("/World")) == {
        "/World",
        "/World/Group",
    }

    refs_cache = _build_subtree_refs_cache(
        {
            "/World/A": ["a.usd"],
            "/World/A/B": ["b.usd"],
            "/Other": ["ignored.usd"],
        },
        "/World",
    )
    assert refs_cache["/World"] == {"a.usd", "b.usd"}
    assert refs_cache["/World/A"] == {"a.usd", "b.usd"}

    # root_path "/" (pseudo-root) is a real, reachable value here too (see
    # the detect_objects pseudo-root regression tests): ancestor-path
    # building must terminate at "/" itself rather than an empty string, and
    # the prefix filter must not become "//" (which would match nothing).
    pseudo_refs_cache = _build_subtree_refs_cache(
        {
            "/World/A": ["a.usd"],
            "/World/A/B": ["b.usd"],
        },
        "/",
    )
    assert pseudo_refs_cache["/"] == {"a.usd", "b.usd"}
    assert pseudo_refs_cache["/World"] == {"a.usd", "b.usd"}
    assert pseudo_refs_cache["/World/A"] == {"a.usd", "b.usd"}

    parent = mesh_stage.GetPrimAtPath("/World")
    assert _compute_sibling_homogeneity_map(leaf, {}) == {}
    homo = _compute_sibling_homogeneity_map(
        parent,
        {"/World/Group": ["same.usd"]},
    )
    assert homo["/World/Group"] == 1.0


def test_surface_identity_uses_material_paths_and_subsets() -> None:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdGeom.Scope.Define(stage, "/World/AltLooks")

    _define_materialized_quad(stage, "/World/A", "/World/Looks/Shared", two_faces=True)
    _define_materialized_quad(stage, "/World/B", "/World/AltLooks/Shared")
    _bind_material_subset(
        stage,
        "/World/A/Body",
        "SubsetA",
        "/World/Looks/SubsetMat",
        [0],
    )

    key_a = _surface_identity_group_key(stage, "/World/A")
    key_b = _surface_identity_group_key(stage, "/World/B")

    assert key_a != key_b
    assert any(surface[2] == "/World/Looks/Shared" for surface in key_a)
    assert any(surface[2] == "/World/AltLooks/Shared" for surface in key_b)
    assert any(surface[2] == "/World/Looks/SubsetMat" for surface in key_a)


def _define_shaded_material(
    stage: Usd.Stage, material_path: str, diffuse_color: tuple[float, float, float]
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse_color)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def test_bound_material_identity_root_relative_path_requires_matching_appearance() -> (
    None
):
    """Root-relative material identity must not collapse differing appearances.

    Two per-asset private material copies with the same relative path (e.g.
    both named "Looks/Copper") are only the same *identity* -- and therefore
    safe to fold into one duplicate-detection representative -- when their
    authored shader appearance also matches. A shared relative path alone is
    not sufficient, or two structurally identical assets with differently
    colored private materials would incorrectly collapse into one.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/AssetX")
    UsdGeom.Xform.Define(stage, "/World/AssetX/Looks")
    UsdGeom.Xform.Define(stage, "/World/AssetY")
    UsdGeom.Xform.Define(stage, "/World/AssetY/Looks")
    UsdGeom.Xform.Define(stage, "/World/AssetZ")
    UsdGeom.Xform.Define(stage, "/World/AssetZ/Looks")

    same_as_x = _define_shaded_material(
        stage, "/World/AssetY/Looks/Copper", (0.8, 0.4, 0.1)
    )
    different_from_x = _define_shaded_material(
        stage, "/World/AssetY/Looks/Copper2", (0.1, 0.1, 0.8)
    )
    copper_x = _define_shaded_material(
        stage, "/World/AssetX/Looks/Copper", (0.8, 0.4, 0.1)
    )
    # Same relative path as copper_x ("Looks/Copper") but a different
    # authored color: this is the specific case the fingerprint targets.
    # Without it, a purely path-based identity would incorrectly collapse
    # this with copper_x even though the two materials look nothing alike.
    same_path_different_color = _define_shaded_material(
        stage, "/World/AssetZ/Looks/Copper", (0.1, 0.1, 0.8)
    )

    identity_x = _bound_material_identity(copper_x, "/World/AssetX")
    identity_y_same_appearance = _bound_material_identity(same_as_x, "/World/AssetY")
    identity_y_different_appearance = _bound_material_identity(
        different_from_x, "/World/AssetY"
    )
    identity_z_same_path_different_color = _bound_material_identity(
        same_path_different_color, "/World/AssetZ"
    )

    # Same relative path ("Looks/Copper") and same authored diffuseColor:
    # these are genuinely the same material copy, so identity must match.
    assert identity_x == identity_y_same_appearance
    # Different relative path (Copper vs Copper2): never equal regardless of
    # appearance.
    assert identity_x != identity_y_different_appearance
    # Same relative path ("Looks/Copper") but different authored diffuseColor:
    # must not collapse just because the path matches.
    assert identity_x != identity_z_same_path_different_color

    # Outside root_path, behavior is unchanged: identity is the absolute path,
    # so even genuinely identical appearances at different absolute paths
    # remain distinct (existing, well-tested behavior for non-per-asset
    # material scopes).
    assert _bound_material_identity(copper_x) == "/World/AssetX/Looks/Copper"
    assert _bound_material_identity(same_as_x) == "/World/AssetY/Looks/Copper"


def _define_textured_material(
    stage: Usd.Stage, material_path: str, texture_file: str
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    texture = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_file)
    texture_output = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
    diffuse_input.ConnectToSource(texture_output)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def test_bound_material_identity_root_relative_path_distinguishes_connected_textures() -> (
    None
):
    """Root-relative identity must not collapse materials with different textures.

    Regression test for a gap where the appearance fingerprint only looked at
    directly authored (unconnected) shader input values. A connected texture
    reader feeding `diffuseColor` leaves that input's own `Get()` at None, so
    without following the connection into the upstream `UsdUVTexture` shader
    the fingerprint was empty for both materials regardless of which texture
    file was actually bound -- silently collapsing two differently-textured
    private material copies that share a relative path.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/AssetX")
    UsdGeom.Xform.Define(stage, "/World/AssetX/Looks")
    UsdGeom.Xform.Define(stage, "/World/AssetY")
    UsdGeom.Xform.Define(stage, "/World/AssetY/Looks")
    UsdGeom.Xform.Define(stage, "/World/AssetZ")
    UsdGeom.Xform.Define(stage, "/World/AssetZ/Looks")

    copper_x = _define_textured_material(
        stage, "/World/AssetX/Looks/Copper", "textures/copper.png"
    )
    same_texture_y = _define_textured_material(
        stage, "/World/AssetY/Looks/Copper", "textures/copper.png"
    )
    # Same relative path ("Looks/Copper") but a different bound texture: this
    # must not collapse with copper_x even though neither shader has a
    # directly authored diffuseColor value.
    different_texture_z = _define_textured_material(
        stage, "/World/AssetZ/Looks/Copper", "textures/rust.png"
    )

    identity_x = _bound_material_identity(copper_x, "/World/AssetX")
    identity_y_same_texture = _bound_material_identity(same_texture_y, "/World/AssetY")
    identity_z_different_texture = _bound_material_identity(
        different_texture_z, "/World/AssetZ"
    )

    assert identity_x == identity_y_same_texture
    assert identity_x != identity_z_different_texture


def _define_dual_textured_material(
    stage: Usd.Stage,
    material_path: str,
    *,
    diffuse_texture: str,
    emissive_texture: str,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    emissive_input = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)

    diffuse_tex = UsdShade.Shader.Define(stage, f"{material_path}/DiffuseTex")
    diffuse_tex.CreateIdAttr("UsdUVTexture")
    diffuse_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(diffuse_texture)
    diffuse_input.ConnectToSource(
        diffuse_tex.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
    )

    emissive_tex = UsdShade.Shader.Define(stage, f"{material_path}/EmissiveTex")
    emissive_tex.CreateIdAttr("UsdUVTexture")
    emissive_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(emissive_texture)
    emissive_input.ConnectToSource(
        emissive_tex.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
    )

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def test_bound_material_identity_distinguishes_swapped_texture_wiring() -> None:
    """Swapping which texture feeds which input must change the fingerprint.

    Regression test: nesting a connected input's upstream fingerprint into
    the parent's flat, sorted value list would let two networks using the
    exact same two textures, but wired to opposite inputs (texture A feeds
    diffuseColor and B feeds emissiveColor, vs. the reverse), collapse to an
    identical fingerprint -- even though the two materials look nothing
    alike. Nesting each connected input's fingerprint as its own value
    (rather than flattening it into the shared list) must keep these
    distinct.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/AssetX")
    UsdGeom.Xform.Define(stage, "/World/AssetX/Looks")
    UsdGeom.Xform.Define(stage, "/World/AssetY")
    UsdGeom.Xform.Define(stage, "/World/AssetY/Looks")

    straight = _define_dual_textured_material(
        stage,
        "/World/AssetX/Looks/Copper",
        diffuse_texture="textures/a.png",
        emissive_texture="textures/b.png",
    )
    swapped = _define_dual_textured_material(
        stage,
        "/World/AssetY/Looks/Copper",
        diffuse_texture="textures/b.png",
        emissive_texture="textures/a.png",
    )

    identity_straight = _bound_material_identity(straight, "/World/AssetX")
    identity_swapped = _bound_material_identity(swapped, "/World/AssetY")

    assert identity_straight != identity_swapped


def test_detect_objects_exercises_grouping_and_source_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    _define_object(stage, "/World/Category/ObjA")
    _define_object(stage, "/World/Category/ObjB")
    _define_object(stage, "/World/LeafCat", "MeshA")
    UsdGeom.Mesh.Define(stage, "/World/LeafCat/MeshB")
    UsdGeom.Mesh.Define(stage, "/World/LeafCat/MeshC")
    _define_object(stage, "/World/NoBBox")
    _define_object(stage, "/World/NameGroup1/Bolt")
    _define_object(stage, "/World/NameGroup2/Bolt")
    _define_object(stage, "/World/Wheels/Wheel__I1")
    _define_object(stage, "/World/Wheels/Wheel__I2")
    _define_object(stage, "/World/FingerA")
    _define_object(stage, "/World/FingerB")
    _define_object(stage, "/World/FingerC")
    _define_object(stage, "/World/FingerD")
    _define_object(stage, "/World/FingerE")
    _define_object(stage, "/World/UniqueMixed")
    _define_object(stage, "/World/UniqueFile")
    _define_object(stage, "/World/Mixed")
    UsdGeom.Xform.Define(stage, "/World/EmptyGroup")
    UsdGeom.Xform.Define(stage, "/World/EmptyGroup/Child")
    stage.DefinePrim("/World/Looks", "Material")
    stage.DefinePrim("/World/Looks/Shader", "Shader")
    _define_object(stage, "/World/SinglePattern/Thing_1")
    _define_object(stage, "/World/Wheels/Wheel__I3")

    UsdGeom.Xform.Define(stage, "/Proto")
    UsdGeom.Mesh.Define(stage, "/Proto/Mesh")
    for name in ("Inst1", "Inst2"):
        prim = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        prim.GetReferences().AddInternalReference("/Proto")
        prim.SetInstanceable(True)
    UsdGeom.Xform.Define(stage, "/ProtoSolo")
    UsdGeom.Mesh.Define(stage, "/ProtoSolo/Mesh")
    solo_inst = UsdGeom.Xform.Define(stage, "/World/InstSingle").GetPrim()
    solo_inst.GetReferences().AddInternalReference("/ProtoSolo")
    solo_inst.SetInstanceable(True)

    ref_path = tmp_path / "asset.usda"
    ref_stage = Usd.Stage.CreateNew(str(ref_path))
    asset = UsdGeom.Xform.Define(ref_stage, "/Asset")
    ref_stage.SetDefaultPrim(asset.GetPrim())
    UsdGeom.Mesh.Define(ref_stage, "/Asset/Mesh")
    ref_stage.GetRootLayer().Save()
    for name in ("RefA", "RefB"):
        prim = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        prim.GetReferences().AddReference(str(ref_path), "/Asset")
    solo_ref_path = tmp_path / "solo.usda"
    solo_stage = Usd.Stage.CreateNew(str(solo_ref_path))
    solo_asset = UsdGeom.Xform.Define(solo_stage, "/Asset")
    solo_stage.SetDefaultPrim(solo_asset.GetPrim())
    UsdGeom.Mesh.Define(solo_stage, "/Asset/Mesh")
    solo_stage.GetRootLayer().Save()
    UsdGeom.Xform.Define(
        stage, "/World/RefSolo"
    ).GetPrim().GetReferences().AddReference(str(solo_ref_path), "/Asset")

    composition_data = {
        "sub_usd_files": [
            {
                "asset_path": "shared.usd",
                "reference_count": 1,
                "referencing_prims": ["/World/Category/ObjA", "/World/Category/ObjB"],
            },
            {
                "asset_path": "block.usd",
                "reference_count": 50,
                "referencing_prims": ["/World/BlockA", "/World/BlockB"],
            },
            {
                "asset_path": "fp_a.usd",
                "reference_count": 1,
                "referencing_prims": [
                    "/World/FingerA",
                    "/World/FingerB",
                    "/World/FingerC",
                    "/World/FingerD",
                    "/World/FingerE",
                    "/World/Mixed",
                ],
            },
            {
                "asset_path": "fp_b.usd",
                "reference_count": 1,
                "referencing_prims": [
                    "/World/FingerA",
                    "/World/FingerB",
                    "/World/FingerC",
                    "/World/FingerD",
                    "/World/FingerE",
                    "/World/Mixed",
                ],
            },
            {
                "asset_path": "unique_a.usd",
                "reference_count": 1,
                "referencing_prims": ["/World/UniqueMixed"],
            },
            {
                "asset_path": "unique_b.usd",
                "reference_count": 1,
                "referencing_prims": ["/World/UniqueMixed"],
            },
            {
                "asset_path": "unique_file.usd",
                "reference_count": 1,
                "referencing_prims": ["/World/UniqueFile"],
            },
        ]
    }

    def fake_stats(stage, path, skip_geometry=False):
        vertices = 7 if "Wheel" in path else 5
        mesh_count = 2 if path.endswith(("FingerC", "FingerD")) else 1
        if path.endswith("FingerE"):
            mesh_count = 3
        if path.endswith("Wheel__I3"):
            vertices = 9
        return {
            "mesh_count": mesh_count,
            "vertex_count": vertices,
            "face_count": 2,
            "prim_type_breakdown": {"Mesh": 1},
        }

    def fake_bbox(prim):
        if "NoBBox" in str(prim.GetPath()):
            raise RuntimeError("bbox failed")
        return _FakeBBox()

    monkeypatch.setattr(
        "world_understanding.utils.usd.prim.get_subtree_geometry_stats",
        fake_stats,
    )
    monkeypatch.setattr(
        "world_understanding.utils.usd.prim.get_bbox_from_prim", fake_bbox
    )

    objects, instance_groups = detect_objects(
        stage,
        composition_data,
        {},
        skip_geometry=True,
        building_block_min_reuse=2,
    )

    by_path = {obj["path"]: obj for obj in objects}
    assert by_path["/World/Category/ObjA"]["source_classification"] == "FILE"
    assert by_path["/World/FingerA"]["source_classification"] == "MIXED"
    assert by_path["/World/NameGroup1/Bolt"]["source_classification"] == "INLINE"
    assert by_path["/World/NoBBox"]["bounding_box"] is None
    assert by_path["/World/Category/ObjA"]["parent_group"] == "Category"

    group_names = {group["group_name"] for group in instance_groups}
    assert "Bolt" in group_names
    assert any(name.startswith("Wheel") for name in group_names)
    assert {"fp_a_1m", "fp_a_2m"}.issubset(group_names)
    assert any(group["source_file"] == str(ref_path) for group in instance_groups)
    assert any(group["source_file"] == "/__Prototype_1" for group in instance_groups)


def test_detect_objects_keeps_non_equivalent_numbered_components_separate() -> None:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")

    _define_materialized_quad(stage, "/World/I12", "/World/Looks/Silver")
    _define_materialized_quad(stage, "/World/I9", "/World/Looks/Black_Plastic")
    _define_materialized_quad(stage, "/World/V1018", "/World/Looks/Black_Plastic")
    _define_materialized_quad(
        stage,
        "/World/V520",
        "/World/Looks/Black_Plastic",
        two_faces=True,
    )

    composition_data = collect_composition_arcs(stage)
    geometry_data = collect_mesh_geometry_stats(stage)

    objects, instance_groups = detect_objects(stage, composition_data, geometry_data)

    object_paths = {obj["path"] for obj in objects}
    assert {"/World/I12", "/World/I9", "/World/V1018", "/World/V520"}.issubset(
        object_paths
    )

    grouped_paths = [set(group["member_paths"]) for group in instance_groups]
    assert not any(
        {"/World/I12", "/World/I9"}.issubset(paths) for paths in grouped_paths
    )
    assert not any(
        {"/World/V1018", "/World/V520"}.issubset(paths) for paths in grouped_paths
    )


def test_detect_objects_handles_pseudo_root_content_root() -> None:
    """detect_objects must not silently drop everything when the content
    root ends up being the stage pseudo-root.

    Regression test: a stage with no default prim and more than 5 top-level
    children hits `_find_content_root`'s `len(children) > 5: return prim`
    early return, which (unlike the two middle branches) is not guarded
    against the pseudo-root. `scene_root_path` then becomes "/", and without
    a fix, `detect_objects` builds a "//" prefix that matches no real prim
    path, so every prim is skipped and zero objects are ever found.
    """
    stage = Usd.Stage.CreateInMemory()
    for idx in range(6):
        _define_object(stage, f"/Asset{idx}")

    composition_data = collect_composition_arcs(stage)
    geometry_data = collect_mesh_geometry_stats(stage)

    objects, _instance_groups = detect_objects(stage, composition_data, geometry_data)

    object_paths = {obj["path"] for obj in objects}
    assert object_paths == {f"/Asset{idx}" for idx in range(6)}


def test_detect_objects_keeps_all_pseudo_root_siblings_in_scope() -> None:
    """Multiple significant top-level assemblies under the pseudo-root must
    all stay in scope, not just the busiest one.

    Regression test: with no default prim and two meaningful top-level
    assemblies (three vs. two mesh-bearing children), `_find_content_root`
    used to special-case the pseudo-root out of its sibling-distribution
    checks, so it fell through to `best_gc_count >= len(children)` and
    descended into only the busiest assembly -- silently omitting every
    object under the other one. Since `detect_objects` now handles a "/"
    content root correctly, keeping the pseudo-root as the content root in
    this case is both safe and necessary to keep every sibling's objects.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/AssetA")
    for idx in range(3):
        _define_object(stage, f"/AssetA/Part{idx}")
    UsdGeom.Xform.Define(stage, "/AssetB")
    for idx in range(2):
        _define_object(stage, f"/AssetB/Part{idx}")

    composition_data = collect_composition_arcs(stage)
    geometry_data = collect_mesh_geometry_stats(stage)

    objects, _instance_groups = detect_objects(stage, composition_data, geometry_data)

    object_paths = {obj["path"] for obj in objects}
    assert {f"/AssetA/Part{idx}" for idx in range(3)}.issubset(object_paths)
    assert {f"/AssetB/Part{idx}" for idx in range(2)}.issubset(object_paths)


def test_detect_objects_keeps_divergent_shared_source_meshes_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    object_paths = [f"/World/mesh_{index}" for index in range(4)]
    for path in object_paths:
        _define_object(stage, path)

    composition_data = {
        "sub_usd_files": [
            {
                "asset_path": "part.usda",
                "reference_count": 1,
                "referencing_prims": object_paths,
            }
        ]
    }

    def fake_stats(
        stage: Usd.Stage, path: str, skip_geometry: bool = False
    ) -> dict[str, object]:
        del stage, skip_geometry
        index = object_paths.index(path)
        return {
            "mesh_count": index + 1,
            "vertex_count": (index + 1) * 10,
            "face_count": (index + 1) * 5,
            "prim_type_breakdown": {"Mesh": index + 1},
        }

    monkeypatch.setattr(
        "world_understanding.utils.usd.prim.get_subtree_geometry_stats",
        fake_stats,
    )

    objects, instance_groups = detect_objects(stage, composition_data, {})

    by_path = {obj["path"]: obj for obj in objects}
    assert set(object_paths).issubset(by_path)
    assert all(by_path[path]["instance_group"] is None for path in object_paths)
    grouped_paths = [set(group["member_paths"]) for group in instance_groups]
    assert not any(len(set(object_paths) & paths) > 1 for paths in grouped_paths)
