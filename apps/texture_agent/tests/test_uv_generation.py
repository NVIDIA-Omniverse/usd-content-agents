# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for UV generation functions."""

from __future__ import annotations

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from texture_agent.functions.uv_generation import (
    UVProjectionMode,
    fix_uv_interpolation,
    generate_box_uvs,
    generate_uvs_for_mesh,
    generate_uvs_for_stage,
    inspect_uvs_for_stage,
    normalize_uvs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_quad_mesh(
    stage: Usd.Stage,
    path: str = "/World/Mesh",
) -> UsdGeom.Mesh:
    """Create a simple quad mesh (a single face with 4 vertices in XY plane)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = [
        Gf.Vec3f(0, 0, 0),
        Gf.Vec3f(1, 0, 0),
        Gf.Vec3f(1, 1, 0),
        Gf.Vec3f(0, 1, 0),
    ]
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    return mesh


def _create_cube_mesh(
    stage: Usd.Stage,
    path: str = "/World/Cube",
) -> UsdGeom.Mesh:
    """Create a simple cube mesh (6 faces, 8 vertices)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = [
        Gf.Vec3f(0, 0, 0),
        Gf.Vec3f(1, 0, 0),
        Gf.Vec3f(1, 1, 0),
        Gf.Vec3f(0, 1, 0),
        Gf.Vec3f(0, 0, 1),
        Gf.Vec3f(1, 0, 1),
        Gf.Vec3f(1, 1, 1),
        Gf.Vec3f(0, 1, 1),
    ]
    mesh.GetPointsAttr().Set(points)
    # 6 quad faces
    mesh.GetFaceVertexCountsAttr().Set([4, 4, 4, 4, 4, 4])
    mesh.GetFaceVertexIndicesAttr().Set(
        [
            0,
            1,
            2,
            3,  # front  (Z=0)
            4,
            5,
            6,
            7,  # back   (Z=1)
            0,
            1,
            5,
            4,  # bottom (Y=0)
            2,
            3,
            7,
            6,  # top    (Y=1)
            0,
            3,
            7,
            4,  # left   (X=0)
            1,
            2,
            6,
            5,  # right  (X=1)
        ]
    )
    return mesh


# ---------------------------------------------------------------------------
# Tests for generate_box_uvs
# ---------------------------------------------------------------------------


class TestGenerateBoxUvs:
    """Tests for the generate_box_uvs function."""

    def test_quad_mesh_produces_correct_shape(self) -> None:
        """Box UV output should have one UV pair per face-vertex index."""
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            dtype=np.float32,
        )
        fvi = np.array([0, 1, 2, 3])
        fvc = np.array([4])

        uvs = generate_box_uvs(points, fvi, fvc)

        assert uvs.shape == (4, 2)

    def test_uvs_within_margin_range(self) -> None:
        """All generated UVs should be within [margin, 1-margin]."""
        margin = 0.05
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            dtype=np.float32,
        )
        fvi = np.array([0, 1, 2, 3])
        fvc = np.array([4])

        uvs = generate_box_uvs(points, fvi, fvc, margin=margin)

        assert uvs.min() >= margin - 1e-6
        assert uvs.max() <= 1.0 - margin + 1e-6

    def test_cube_mesh_all_faces_get_uvs(self) -> None:
        """A cube with 6 quad faces (24 face-vertices) should produce 24 UVs."""
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )
        fvi = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 0, 3, 7, 4, 1, 2, 6, 5]
        )
        fvc = np.array([4, 4, 4, 4, 4, 4])

        uvs = generate_box_uvs(points, fvi, fvc)

        assert uvs.shape == (24, 2)
        assert np.all(uvs >= 0.0)
        assert np.all(uvs <= 1.0)

    def test_custom_margin(self) -> None:
        """UVs should respect a custom margin value."""
        margin = 0.1
        points = np.array(
            [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
            dtype=np.float32,
        )
        fvi = np.array([0, 1, 2, 3])
        fvc = np.array([4])

        uvs = generate_box_uvs(points, fvi, fvc, margin=margin)

        # Corner vertices should map to margin and 1-margin
        assert uvs.min() >= margin - 1e-6
        assert uvs.max() <= 1.0 - margin + 1e-6


# ---------------------------------------------------------------------------
# Tests for fix_uv_interpolation
# ---------------------------------------------------------------------------


class TestFixUvInterpolation:
    """Tests for the fix_uv_interpolation function."""

    def test_fixes_compatible_constant_interpolation(self) -> None:
        """Constant UV interpolation should be fixed only when counts match."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage)
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "constant")
        st.Set(
            Vt.Vec2fArray(
                [Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]
            )
        )

        count = fix_uv_interpolation(stage)

        assert count == 1
        updated = api.GetPrimvar("st")
        assert updated.GetInterpolation() == "faceVarying"

    def test_skips_incompatible_constant_interpolation(self) -> None:
        """A single constant UV value is not silently treated as face-varying."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage)
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "constant")
        st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0)]))

        count = fix_uv_interpolation(stage)

        assert count == 0
        updated = api.GetPrimvar("st")
        assert updated.GetInterpolation() == "constant"

    def test_skips_face_varying_interpolation(self) -> None:
        """Meshes already using 'faceVarying' should not be touched."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage)
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
        st.Set(
            Vt.Vec2fArray(
                [Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]
            )
        )

        count = fix_uv_interpolation(stage)

        assert count == 0

    def test_no_meshes_returns_zero(self) -> None:
        """An empty stage should return 0."""
        stage = Usd.Stage.CreateInMemory()
        count = fix_uv_interpolation(stage)
        assert count == 0


# ---------------------------------------------------------------------------
# Tests for normalize_uvs
# ---------------------------------------------------------------------------


class TestNormalizeUvs:
    """Tests for the normalize_uvs function."""

    def test_out_of_range_uvs_are_normalized(self) -> None:
        """UVs outside [0, 1] should be normalized to [margin, 1-margin]."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage)
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
        # UVs far outside [0, 1]
        st.Set(
            Vt.Vec2fArray(
                [
                    Gf.Vec2f(-1.0, -2.0),
                    Gf.Vec2f(3.0, -2.0),
                    Gf.Vec2f(3.0, 4.0),
                    Gf.Vec2f(-1.0, 4.0),
                ]
            )
        )

        margin = 0.025
        count = normalize_uvs(stage, margin=margin)

        assert count == 1
        updated_uvs = np.array(api.GetPrimvar("st").Get())
        assert updated_uvs.min() >= margin - 1e-6
        assert updated_uvs.max() <= 1.0 - margin + 1e-6

    def test_in_range_uvs_are_not_modified(self) -> None:
        """UVs already in [0, 1] should not be changed."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage)
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
        original = Vt.Vec2fArray(
            [
                Gf.Vec2f(0.1, 0.1),
                Gf.Vec2f(0.9, 0.1),
                Gf.Vec2f(0.9, 0.9),
                Gf.Vec2f(0.1, 0.9),
            ]
        )
        st.Set(original)

        count = normalize_uvs(stage)

        assert count == 0
        # Values should remain unchanged
        result = np.array(api.GetPrimvar("st").Get())
        expected = np.array(original)
        np.testing.assert_allclose(result, expected)

    def test_partially_out_of_range(self) -> None:
        """If any UV component exceeds [0, 1], normalization should apply."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage)
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
        # u goes to 1.5 -> out of range
        st.Set(
            Vt.Vec2fArray(
                [
                    Gf.Vec2f(0.0, 0.0),
                    Gf.Vec2f(1.5, 0.0),
                    Gf.Vec2f(1.5, 0.5),
                    Gf.Vec2f(0.0, 0.5),
                ]
            )
        )

        count = normalize_uvs(stage)

        assert count == 1


# ---------------------------------------------------------------------------
# Tests for generate_uvs_for_stage
# ---------------------------------------------------------------------------


class TestGenerateUvsForStage:
    """Tests for the generate_uvs_for_stage function."""

    def test_generates_uvs_for_meshes_without_uvs(self) -> None:
        """All meshes lacking UVs should receive new ones."""
        stage = Usd.Stage.CreateInMemory()
        _create_quad_mesh(stage, "/World/MeshA")
        _create_quad_mesh(stage, "/World/MeshB")

        count = generate_uvs_for_stage(stage)

        assert count == 2
        for path in ["/World/MeshA", "/World/MeshB"]:
            prim = stage.GetPrimAtPath(path)
            api = UsdGeom.PrimvarsAPI(prim)
            st = api.GetPrimvar("st")
            assert st.IsDefined()
            uvs = st.Get()
            assert len(uvs) == 4

    def test_skips_meshes_with_existing_uvs(self) -> None:
        """Meshes that already have UVs should not be overwritten."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage, "/World/HasUVs")
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
        original_uvs = Vt.Vec2fArray(
            [
                Gf.Vec2f(0.1, 0.2),
                Gf.Vec2f(0.3, 0.4),
                Gf.Vec2f(0.5, 0.6),
                Gf.Vec2f(0.7, 0.8),
            ]
        )
        st.Set(original_uvs)

        # Also add a mesh without UVs
        _create_quad_mesh(stage, "/World/NoUVs")

        count = generate_uvs_for_stage(stage)

        # Only the mesh without UVs should get new ones
        assert count == 1
        # Verify original UVs are untouched
        result_uvs = np.array(
            UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/HasUVs"))
            .GetPrimvar("st")
            .Get()
        )
        np.testing.assert_allclose(result_uvs, np.array(original_uvs))

    def test_target_prim_paths_limit_generation(self) -> None:
        """Only targeted mesh prims should be mutated when a target scope is set."""
        stage = Usd.Stage.CreateInMemory()
        target_mesh = _create_quad_mesh(stage, "/World/TargetMesh")
        other_mesh = _create_quad_mesh(stage, "/World/OtherMesh")
        target_uvs = Vt.Vec2fArray([Gf.Vec2f(0.2, 0.2)] * 4)
        other_uvs = Vt.Vec2fArray([Gf.Vec2f(0.8, 0.8)] * 4)
        UsdGeom.PrimvarsAPI(target_mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
        ).Set(target_uvs)
        UsdGeom.PrimvarsAPI(other_mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
        ).Set(other_uvs)

        count = generate_uvs_for_stage(
            stage,
            overwrite_existing=True,
            target_prim_paths=("/World/TargetMesh",),
        )

        assert count == 1
        updated_target = np.array(
            UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/TargetMesh"))
            .GetPrimvar("st")
            .Get()
        )
        updated_other = np.array(
            UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/OtherMesh"))
            .GetPrimvar("st")
            .Get()
        )
        assert not np.allclose(updated_target, np.array(target_uvs))
        np.testing.assert_allclose(updated_other, np.array(other_uvs))

    def test_target_generation_deinstances_only_target_mesh(self) -> None:
        """Targeted UV generation should not touch unrelated instanceable meshes."""
        stage = Usd.Stage.CreateInMemory()
        target_mesh = _create_quad_mesh(stage, "/World/TargetMesh")
        other_mesh = _create_quad_mesh(stage, "/World/OtherMesh")
        target_mesh.GetPrim().SetInstanceable(True)
        other_mesh.GetPrim().SetInstanceable(True)

        count = generate_uvs_for_stage(
            stage,
            target_prim_paths=("/World/TargetMesh",),
        )

        assert count == 1
        assert target_mesh.GetPrim().IsInstanceable() is False
        assert other_mesh.GetPrim().IsInstanceable() is True

    def test_planar_mode(self) -> None:
        """Planar projection mode should also generate valid UVs."""
        stage = Usd.Stage.CreateInMemory()
        _create_quad_mesh(stage)

        count = generate_uvs_for_stage(stage, mode=UVProjectionMode.PLANAR)

        assert count == 1
        prim = stage.GetPrimAtPath("/World/Mesh")
        api = UsdGeom.PrimvarsAPI(prim)
        st = api.GetPrimvar("st")
        assert st.IsDefined()
        uvs = np.array(st.Get())
        assert uvs.shape == (4, 2)
        assert np.all(uvs >= 0.0)
        assert np.all(uvs <= 1.0)

    def test_empty_stage_returns_zero(self) -> None:
        """An empty stage should return 0."""
        stage = Usd.Stage.CreateInMemory()
        count = generate_uvs_for_stage(stage)
        assert count == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for UV generation."""

    def test_mesh_with_no_points(self) -> None:
        """A mesh prim with no points should be skipped."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Mesh.Define(stage, "/World/Empty")

        count = generate_uvs_for_stage(stage)

        assert count == 0

    def test_mesh_with_empty_topology(self) -> None:
        """A mesh with points but no face indices should be skipped."""
        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/World/NoFaces")
        mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0)])
        mesh.GetFaceVertexCountsAttr().Set([])
        mesh.GetFaceVertexIndicesAttr().Set([])

        count = generate_uvs_for_stage(stage)

        assert count == 0

    def test_instance_proxy_skipped(self) -> None:
        """Instance proxy prims should be skipped during UV generation."""
        stage = Usd.Stage.CreateInMemory()

        # Create a prototype mesh inside a scope we will reference
        stage.DefinePrim("/Prototypes")
        proto_mesh = UsdGeom.Mesh.Define(stage, "/Prototypes/Mesh")
        proto_mesh.GetPointsAttr().Set(
            [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 1, 0)]
        )
        proto_mesh.GetFaceVertexCountsAttr().Set([3])
        proto_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])

        # Create an instance that references the prototype
        instance = stage.DefinePrim("/World/Instance")
        instance.GetReferences().AddInternalReference("/Prototypes")
        instance.SetInstanceable(True)

        # generate_uvs_for_stage should skip instance proxies.
        # The prototype mesh itself is traversable and not an instance proxy,
        # so it should be processed.
        count = generate_uvs_for_stage(stage)

        # The prototype mesh at /Prototypes/Mesh should get UVs
        proto_prim = stage.GetPrimAtPath("/Prototypes/Mesh")
        api = UsdGeom.PrimvarsAPI(proto_prim)
        st = api.GetPrimvar("st")
        assert st.IsDefined()
        assert count == 1

    def test_generate_uvs_for_mesh_returns_false_for_existing_uvs(
        self,
    ) -> None:
        """generate_uvs_for_mesh should return False if UVs already exist."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage)
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
        st.Set(
            Vt.Vec2fArray(
                [
                    Gf.Vec2f(0, 0),
                    Gf.Vec2f(1, 0),
                    Gf.Vec2f(1, 1),
                    Gf.Vec2f(0, 1),
                ]
            )
        )

        result = generate_uvs_for_mesh(mesh.GetPrim())

        assert result is False

    def test_generate_uvs_for_mesh_can_force_projection(self) -> None:
        """force projection should overwrite existing UVs only when requested."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage)
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
        original_uvs = Vt.Vec2fArray(
            [
                Gf.Vec2f(0.2, 0.2),
                Gf.Vec2f(0.2, 0.2),
                Gf.Vec2f(0.2, 0.2),
                Gf.Vec2f(0.2, 0.2),
            ]
        )
        st.Set(original_uvs)

        result = generate_uvs_for_mesh(mesh.GetPrim(), overwrite_existing=True)

        assert result is True
        updated = np.array(api.GetPrimvar("st").Get())
        assert not np.allclose(updated, np.array(original_uvs))

    def test_generate_uvs_for_mesh_overwrites_indexed_primvar(self) -> None:
        """force projection should clear stale indices and author face-varying UVs."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage)
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "constant")
        st.Set(Vt.Vec2fArray([Gf.Vec2f(0.2, 0.2)]))
        st.SetIndices(Vt.IntArray([0, 0, 0, 0]))

        result = generate_uvs_for_mesh(mesh.GetPrim(), overwrite_existing=True)

        assert result is True
        updated = api.GetPrimvar("st")
        assert updated.GetInterpolation() == "faceVarying"
        assert len(updated.Get()) == 4
        assert not updated.IsIndexed()

    def test_inspect_uvs_reports_missing_and_diagnostics(self) -> None:
        """UV inspection should return structured diagnostics for missing UVs."""
        stage = Usd.Stage.CreateInMemory()
        _create_quad_mesh(stage, "/World/NoUVs")

        report = inspect_uvs_for_stage(stage)

        assert report["schema_version"] == "texture-agent-uv-report.v1"
        assert report["summary"]["missing"] == 1
        mesh = report["meshes"][0]
        assert mesh["status"] == "missing"
        assert mesh["diagnostics"][0]["code"] == "UV_MISSING_ST"

    def test_inspect_uvs_reports_indexed_uvs(self) -> None:
        """Indexed UV primvars should report index/value counts."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage)
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
        st.Set(
            Vt.Vec2fArray(
                [Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]
            )
        )
        st.SetIndices(Vt.IntArray([0, 1, 2, 3]))

        report = inspect_uvs_for_stage(stage)

        mesh_report = report["meshes"][0]
        assert mesh_report["status"] == "valid"
        assert mesh_report["indexed"] is True
        assert mesh_report["value_count"] == 4
        assert mesh_report["index_count"] == 4

    def test_inspect_uvs_reports_bad_index_count(self) -> None:
        """Indexed UVs with out-of-range indices should be invalid."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage)
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
        st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0)]))
        st.SetIndices(Vt.IntArray([0, 1, 2, 3]))

        report = inspect_uvs_for_stage(stage)

        mesh_report = report["meshes"][0]
        assert mesh_report["status"] == "invalid"
        assert "UV_BAD_INDEX_COUNT" in mesh_report["issues"]

    def test_inspect_uvs_reports_bad_face_varying_value_count(self) -> None:
        """Unindexed face-varying UV counts must match face-vertex counts."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage)
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
        st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1)]))

        report = inspect_uvs_for_stage(stage)

        mesh_report = report["meshes"][0]
        assert mesh_report["status"] == "invalid"
        assert "UV_BAD_VALUE_COUNT" in mesh_report["issues"]

    def test_inspect_uvs_reports_non_finite_values(self) -> None:
        """NaN or infinite UV coordinates should be blocking diagnostics."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage)
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
        st.Set(
            Vt.Vec2fArray(
                [
                    Gf.Vec2f(0, 0),
                    Gf.Vec2f(float("nan"), 0),
                    Gf.Vec2f(1, 1),
                    Gf.Vec2f(0, 1),
                ]
            )
        )

        report = inspect_uvs_for_stage(stage)

        mesh_report = report["meshes"][0]
        assert mesh_report["status"] == "invalid"
        assert "UV_NAN_INF" in mesh_report["issues"]

    def test_inspect_uvs_reports_incompatible_constant_as_invalid(self) -> None:
        """Unsafe constant interpolation should be a blocking diagnostic."""
        stage = Usd.Stage.CreateInMemory()
        mesh = _create_quad_mesh(stage)
        api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "constant")
        st.Set(Vt.Vec2fArray([Gf.Vec2f(0.5, 0.5)]))

        report = inspect_uvs_for_stage(stage)

        mesh_report = report["meshes"][0]
        assert mesh_report["status"] == "invalid"
        assert mesh_report["diagnostics"][0]["code"] == "UV_BAD_INTERPOLATION"

    def test_fix_uv_interpolation_skips_instance_proxy(self) -> None:
        """fix_uv_interpolation should skip instance proxy prims."""
        stage = Usd.Stage.CreateInMemory()

        # Create prototype with constant interpolation UV
        proto_mesh = UsdGeom.Mesh.Define(stage, "/Prototypes/Mesh")
        proto_mesh.GetPointsAttr().Set(
            [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 1, 0)]
        )
        proto_mesh.GetFaceVertexCountsAttr().Set([3])
        proto_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
        api = UsdGeom.PrimvarsAPI(proto_mesh.GetPrim())
        st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "constant")
        st.Set(
            Vt.Vec2fArray([Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(0.0, 1.0)])
        )

        # Create an instance
        instance = stage.DefinePrim("/World/Instance")
        instance.GetReferences().AddInternalReference("/Prototypes")
        instance.SetInstanceable(True)

        # Should fix the prototype but not the instance proxy
        count = fix_uv_interpolation(stage)

        assert count == 1

    def test_fix_uv_interpolation_deinstances_only_target_mesh(self) -> None:
        """Scoped interpolation repair should not de-instance unrelated meshes."""
        stage = Usd.Stage.CreateInMemory()
        target_mesh = _create_quad_mesh(stage, "/World/TargetMesh")
        other_mesh = _create_quad_mesh(stage, "/World/OtherMesh")
        for mesh in (target_mesh, other_mesh):
            api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
            st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "constant")
            st.Set(
                Vt.Vec2fArray(
                    [
                        Gf.Vec2f(0.0, 0.0),
                        Gf.Vec2f(1.0, 0.0),
                        Gf.Vec2f(1.0, 1.0),
                        Gf.Vec2f(0.0, 1.0),
                    ]
                )
            )
            mesh.GetPrim().SetInstanceable(True)

        count = fix_uv_interpolation(
            stage,
            target_prim_paths=("/World/TargetMesh",),
        )

        target_st = UsdGeom.PrimvarsAPI(target_mesh.GetPrim()).GetPrimvar("st")
        other_st = UsdGeom.PrimvarsAPI(other_mesh.GetPrim()).GetPrimvar("st")
        assert count == 1
        assert target_st.GetInterpolation() == "faceVarying"
        assert other_st.GetInterpolation() == "constant"
        assert target_mesh.GetPrim().IsInstanceable() is False
        assert other_mesh.GetPrim().IsInstanceable() is True

    def test_normalize_uvs_skips_mesh_without_st(self) -> None:
        """normalize_uvs should skip meshes that have no 'st' primvar."""
        stage = Usd.Stage.CreateInMemory()
        _create_quad_mesh(stage)

        count = normalize_uvs(stage)

        assert count == 0

    def test_normalize_uvs_deinstances_only_target_mesh(self) -> None:
        """Scoped UV normalization should not de-instance unrelated meshes."""
        stage = Usd.Stage.CreateInMemory()
        target_mesh = _create_quad_mesh(stage, "/World/TargetMesh")
        other_mesh = _create_quad_mesh(stage, "/World/OtherMesh")
        for mesh in (target_mesh, other_mesh):
            api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
            st = api.CreatePrimvar(
                "st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                "faceVarying",
            )
            st.Set(Vt.Vec2fArray([Gf.Vec2f(-1.0, 2.0)] * 4))
            mesh.GetPrim().SetInstanceable(True)

        count = normalize_uvs(
            stage,
            target_prim_paths=("/World/TargetMesh",),
        )

        assert count == 1
        assert target_mesh.GetPrim().IsInstanceable() is False
        assert other_mesh.GetPrim().IsInstanceable() is True
