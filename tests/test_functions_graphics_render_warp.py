# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Warp rendering backend.

Unit tests run without GPU/warp. Integration tests require warp + CUDA GPU.
"""

import math
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from world_understanding.functions.graphics import render_warp
from world_understanding.functions.graphics.render_warp import (
    _extract_meshes,
    _gf_matrix_to_transform_7f,
    _import_warp,
    _setup_render_context,
    _triangulate,
    _unpack_color_image,
    _unpack_depth_image,
    _unpack_normal_image,
)

# ---------------------------------------------------------------------------
# Unit tests (no GPU required)
# ---------------------------------------------------------------------------


class _FakeWarpArray:
    def __init__(self, data):
        self.data = np.asarray(data)

    def numpy(self):
        return self.data

    def reshape(self, shape):
        return self

    def zero_(self):
        self.data[...] = 0

    def fill_(self, value):
        self.data[...] = value


class _FakeWarp:
    bool = "bool"
    float32 = "float32"
    int32 = "int32"
    uint32 = "uint32"
    uint64 = "uint64"
    vec2f = "vec2f"
    vec3f = "vec3f"
    vec3i = "vec3i"
    vec4f = "vec4f"
    transformf = "transformf"
    transform = "transform"

    def __init__(self):
        self.mesh_ids = 0

    def array(self, data, **kwargs):
        return _FakeWarpArray(data)

    def empty(self, shape, **kwargs):
        return _FakeWarpArray(np.empty(shape, dtype=object))

    def zeros(self, shape, **kwargs):
        if isinstance(shape, tuple):
            return _FakeWarpArray(np.zeros(shape))
        return _FakeWarpArray(np.zeros((shape,)))

    def Mesh(self, **kwargs):
        self.mesh_ids += 1
        return SimpleNamespace(id=self.mesh_ids, kwargs=kwargs)

    def init(self):
        return None

    def synchronize_device(self, device):
        return None


class TestTriangulate:
    """Test _triangulate() for polygon → triangle conversion."""

    def test_single_triangle(self):
        """A single triangle (3 verts) passes through unchanged."""
        fvc = np.array([3])
        fvi = np.array([0, 1, 2])
        result = _triangulate(fvc, fvi)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_single_quad(self):
        """A quad (4 verts) produces 2 triangles via fan."""
        fvc = np.array([4])
        fvi = np.array([0, 1, 2, 3])
        result = _triangulate(fvc, fvi)
        # Fan from v0: (0,1,2) and (0,2,3)
        np.testing.assert_array_equal(result, [0, 1, 2, 0, 2, 3])

    def test_mixed_faces(self):
        """Mixed tri + quad produces correct triangle count."""
        fvc = np.array([3, 4])
        fvi = np.array([0, 1, 2, 3, 4, 5, 6])
        result = _triangulate(fvc, fvi)
        # tri: (0,1,2), quad fan: (3,4,5), (3,5,6)
        assert len(result) == 9  # 3 triangles * 3 verts

    def test_pentagon(self):
        """A pentagon (5 verts) produces 3 triangles."""
        fvc = np.array([5])
        fvi = np.array([0, 1, 2, 3, 4])
        result = _triangulate(fvc, fvi)
        # Fan from v0: (0,1,2), (0,2,3), (0,3,4)
        expected = [0, 1, 2, 0, 2, 3, 0, 3, 4]
        np.testing.assert_array_equal(result, expected)

    def test_empty_input(self):
        """Empty input produces empty output."""
        fvc = np.array([], dtype=np.int32)
        fvi = np.array([], dtype=np.int32)
        result = _triangulate(fvc, fvi)
        assert len(result) == 0

    def test_dtype_is_int32(self):
        """Result dtype should be int32."""
        fvc = np.array([3])
        fvi = np.array([0, 1, 2])
        result = _triangulate(fvc, fvi)
        assert result.dtype == np.int32


class TestGfMatrixToTransform7f:
    """Test _gf_matrix_to_transform_7f() transform conversion."""

    def test_identity_matrix(self):
        """Identity matrix → position (0,0,0) + identity quaternion (0,0,0,1)."""
        from pxr import Gf

        m = Gf.Matrix4d(1.0)
        result = _gf_matrix_to_transform_7f(m)
        assert len(result) == 7
        # Position should be origin
        assert abs(result[0]) < 1e-6
        assert abs(result[1]) < 1e-6
        assert abs(result[2]) < 1e-6
        # Quaternion should be identity (0, 0, 0, 1)
        assert abs(result[3]) < 1e-6  # qx
        assert abs(result[4]) < 1e-6  # qy
        assert abs(result[5]) < 1e-6  # qz
        assert abs(result[6] - 1.0) < 1e-6  # qw

    def test_translation_only(self):
        """Pure translation matrix extracts correct position."""
        from pxr import Gf

        m = Gf.Matrix4d(1.0)
        m.SetTranslateOnly(Gf.Vec3d(1.0, 2.0, 3.0))
        result = _gf_matrix_to_transform_7f(m)
        assert abs(result[0] - 1.0) < 1e-6
        assert abs(result[1] - 2.0) < 1e-6
        assert abs(result[2] - 3.0) < 1e-6

    def test_quaternion_is_normalized(self):
        """Result quaternion should be unit length."""
        from pxr import Gf

        # Create a rotation matrix (90 deg around Y)
        m = Gf.Matrix4d(1.0)
        m.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90))
        result = _gf_matrix_to_transform_7f(m)
        qx, qy, qz, qw = result[3], result[4], result[5], result[6]
        length = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        assert abs(length - 1.0) < 1e-5


class TestUnpackColorImage:
    """Test _unpack_color_image() uint32 → RGBA conversion."""

    def test_red_pixel(self):
        """Pure red pixel (R=255) unpacks correctly."""
        # Pack: R in bits 0-7, G in 8-15, B in 16-23, A in 24-31
        red_packed = np.array(
            [[[[255 | (0 << 8) | (0 << 16) | (255 << 24)]]]],
            dtype=np.uint32,
        )
        result = _unpack_color_image(red_packed, 0, 0)
        assert result.shape == (1, 1, 4)
        assert result[0, 0, 0] == 255  # R
        assert result[0, 0, 1] == 0  # G
        assert result[0, 0, 2] == 0  # B
        assert result[0, 0, 3] == 255  # A (forced to 255)

    def test_green_pixel(self):
        """Pure green pixel (G=255) unpacks correctly."""
        green_packed = np.array(
            [[[[0 | (255 << 8) | (0 << 16) | (255 << 24)]]]],
            dtype=np.uint32,
        )
        result = _unpack_color_image(green_packed, 0, 0)
        assert result[0, 0, 0] == 0  # R
        assert result[0, 0, 1] == 255  # G
        assert result[0, 0, 2] == 0  # B

    def test_blue_pixel(self):
        """Pure blue pixel (B=255) unpacks correctly."""
        blue_packed = np.array(
            [[[[0 | (0 << 8) | (255 << 16) | (255 << 24)]]]],
            dtype=np.uint32,
        )
        result = _unpack_color_image(blue_packed, 0, 0)
        assert result[0, 0, 0] == 0  # R
        assert result[0, 0, 1] == 0  # G
        assert result[0, 0, 2] == 255  # B

    def test_white_pixel(self):
        """White pixel (all 255) unpacks correctly."""
        white_packed = np.array(
            [[[[255 | (255 << 8) | (255 << 16) | (255 << 24)]]]],
            dtype=np.uint32,
        )
        result = _unpack_color_image(white_packed, 0, 0)
        assert result[0, 0, 0] == 255
        assert result[0, 0, 1] == 255
        assert result[0, 0, 2] == 255
        assert result[0, 0, 3] == 255

    def test_alpha_always_255(self):
        """Alpha channel is always forced to 255 regardless of packed value."""
        # Pack with alpha = 0 (should still output 255)
        packed = np.array(
            [[[[128 | (64 << 8) | (32 << 16) | (0 << 24)]]]],
            dtype=np.uint32,
        )
        result = _unpack_color_image(packed, 0, 0)
        assert result[0, 0, 3] == 255  # Alpha forced to 255

    def test_multi_camera(self):
        """Multiple cameras are indexed correctly."""
        # 1 world, 2 cameras, 1x1 pixels
        packed = np.array(
            [
                [
                    [[255]],  # cam 0: red channel only
                    [[255 << 8]],  # cam 1: green channel only
                ]
            ],
            dtype=np.uint32,
        )
        result0 = _unpack_color_image(packed, 0, 0)
        result1 = _unpack_color_image(packed, 0, 1)
        assert result0[0, 0, 0] == 255  # cam 0 = red
        assert result1[0, 0, 1] == 255  # cam 1 = green


class TestUnpackDepthImage:
    """Test _unpack_depth_image() extracts correct slice."""

    def test_extracts_correct_camera(self):
        """Depth extraction selects the right world/camera slice."""
        depth = np.zeros((1, 2, 4, 4), dtype=np.float32)
        depth[0, 0] = 1.0
        depth[0, 1] = 2.0

        result0 = _unpack_depth_image(depth, 0, 0)
        result1 = _unpack_depth_image(depth, 0, 1)
        assert result0.shape == (4, 4)
        np.testing.assert_allclose(result0, 1.0)
        np.testing.assert_allclose(result1, 2.0)

    def test_returns_copy(self):
        """Result should be a copy, not a view."""
        depth = np.ones((1, 1, 2, 2), dtype=np.float32)
        result = _unpack_depth_image(depth, 0, 0)
        result[0, 0] = 99.0
        assert depth[0, 0, 0, 0] == 1.0  # Original unchanged


class TestUnpackNormalImage:
    """Test _unpack_normal_image() extracts correct slice."""

    def test_returns_copy(self):
        normal = np.ones((1, 1, 2, 2, 3), dtype=np.float32)
        result = _unpack_normal_image(normal, 0, 0)
        assert result.shape == (2, 2, 3)
        result[0, 0, 0] = 9.0
        assert normal[0, 0, 0, 0, 0] == 1.0


def test_gf_matrix_zero_quaternion_falls_back_to_identity(monkeypatch):
    from pxr import Gf

    class ZeroQuat:
        def GetReal(self):
            return 0.0

        def GetImaginary(self):
            return [0.0, 0.0, 0.0]

    class ZeroRotation:
        def GetQuat(self):
            return ZeroQuat()

    class ZeroTransform:
        def GetRotation(self):
            return ZeroRotation()

    class ZeroMatrix:
        def ExtractTranslation(self):
            return [1.0, 2.0, 3.0]

    monkeypatch.setattr(Gf, "Transform", lambda matrix: ZeroTransform())

    assert _gf_matrix_to_transform_7f(ZeroMatrix()) == [
        1.0,
        2.0,
        3.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def test_import_warp_uses_legacy_render_shape_type(monkeypatch):
    fake_warp = types.ModuleType("warp")
    fake_newton = types.ModuleType("newton")
    fake_src = types.ModuleType("newton._src")
    fake_sensors = types.ModuleType("newton._src.sensors")
    fake_raytrace = types.ModuleType("newton._src.sensors.warp_raytrace")
    fake_raytrace_types = types.ModuleType("newton._src.sensors.warp_raytrace.types")

    class MeshValue:
        def __int__(self):
            return 42

    fake_raytrace.RenderContext = object
    fake_raytrace.RenderShapeType = SimpleNamespace(MESH=MeshValue())
    fake_raytrace_types.RenderLightType = SimpleNamespace(DIRECTIONAL=3)
    fake_newton._src = fake_src
    fake_src.sensors = fake_sensors
    fake_sensors.warp_raytrace = fake_raytrace
    fake_raytrace.types = fake_raytrace_types

    for name, module in {
        "warp": fake_warp,
        "newton": fake_newton,
        "newton._src": fake_src,
        "newton._src.sensors": fake_sensors,
        "newton._src.sensors.warp_raytrace": fake_raytrace,
        "newton._src.sensors.warp_raytrace.types": fake_raytrace_types,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    wp, render_context, mesh_shape_type_int, render_light_type = _import_warp()

    assert wp is fake_warp
    assert render_context is object
    assert mesh_shape_type_int == 42
    assert render_light_type.DIRECTIONAL == 3


def test_import_warp_uses_geotype_when_render_shape_type_is_missing(monkeypatch):
    fake_warp = types.ModuleType("warp")
    fake_newton = types.ModuleType("newton")
    fake_src = types.ModuleType("newton._src")
    fake_sensors = types.ModuleType("newton._src.sensors")
    fake_raytrace = types.ModuleType("newton._src.sensors.warp_raytrace")
    fake_raytrace_types = types.ModuleType("newton._src.sensors.warp_raytrace.types")
    fake_geometry = types.ModuleType("newton._src.geometry")

    class MeshValue:
        def __int__(self):
            return 99

    fake_raytrace.RenderContext = object
    fake_raytrace_types.RenderLightType = SimpleNamespace(DIRECTIONAL=7)
    fake_geometry.GeoType = SimpleNamespace(MESH=MeshValue())
    fake_newton._src = fake_src
    fake_src.sensors = fake_sensors
    fake_sensors.warp_raytrace = fake_raytrace
    fake_raytrace.types = fake_raytrace_types

    for name, module in {
        "warp": fake_warp,
        "newton": fake_newton,
        "newton._src": fake_src,
        "newton._src.sensors": fake_sensors,
        "newton._src.sensors.warp_raytrace": fake_raytrace,
        "newton._src.sensors.warp_raytrace.types": fake_raytrace_types,
        "newton._src.geometry": fake_geometry,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    wp, render_context, mesh_shape_type_int, render_light_type = _import_warp()

    assert wp is fake_warp
    assert render_context is object
    assert mesh_shape_type_int == 99
    assert render_light_type.DIRECTIONAL == 7


def test_extract_meshes_skips_non_renderable_meshes(monkeypatch):
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Group")

    guide_mesh = UsdGeom.Mesh.Define(stage, "/Guide")
    guide_mesh.GetPointsAttr().Set(
        [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)]
    )
    guide_mesh.GetFaceVertexCountsAttr().Set([3])
    guide_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    UsdGeom.Imageable(guide_mesh.GetPrim()).CreatePurposeAttr().Set(
        UsdGeom.Tokens.guide
    )

    UsdGeom.Mesh.Define(stage, "/MissingAttrs")

    empty_mesh = UsdGeom.Mesh.Define(stage, "/Empty")
    empty_mesh.GetPointsAttr().Set([])
    empty_mesh.GetFaceVertexCountsAttr().Set([])
    empty_mesh.GetFaceVertexIndicesAttr().Set([])

    line_mesh = UsdGeom.Mesh.Define(stage, "/Line")
    line_mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0)])
    line_mesh.GetFaceVertexCountsAttr().Set([2])
    line_mesh.GetFaceVertexIndicesAttr().Set([0, 1])

    valid_mesh = UsdGeom.Mesh.Define(stage, "/Valid")
    valid_mesh.GetPointsAttr().Set(
        [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)]
    )
    valid_mesh.GetFaceVertexCountsAttr().Set([3])
    valid_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    valid_mesh.GetDisplayColorAttr().Set([(0.25, 0.5, 0.75)])

    monkeypatch.setattr(
        render_warp, "_import_warp", lambda: (_FakeWarp(), None, None, None)
    )

    render_meshes, mesh_prims = _extract_meshes(stage, Usd.TimeCode.Default(), "cpu")
    assert len(render_meshes) == 1
    assert [str(prim.GetPath()) for prim in mesh_prims] == ["/Valid"]
    assert render_warp._get_display_color(
        valid_mesh.GetPrim(), Usd.TimeCode.Default()
    ) == (0.75, 1.0, 1.0, 1.0)
    assert render_warp._get_display_color(
        stage.GetPrimAtPath("/MissingAttrs"), Usd.TimeCode.Default()
    ) == (0.8, 0.8, 0.8, 1.0)


def test_setup_lights_uses_existing_distant_light(monkeypatch):
    from pxr import Usd, UsdLux

    stage = Usd.Stage.CreateInMemory()
    UsdLux.DistantLight.Define(stage, "/Key")
    ctx = SimpleNamespace()

    monkeypatch.setattr(
        render_warp,
        "_import_warp",
        lambda: (_FakeWarp(), None, None, SimpleNamespace(DIRECTIONAL=5)),
    )

    render_warp._setup_lights(stage, ctx, Usd.TimeCode.Default(), "cpu")

    assert ctx.lights_active.data.tolist() == [True]
    assert ctx.lights_type.data.tolist() == [5]
    assert ctx.lights_orientation.data.shape == (1, 3)


def test_setup_lights_uses_default_light_rig(monkeypatch):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/NonLight")
    ctx = SimpleNamespace()

    monkeypatch.setattr(
        render_warp,
        "_import_warp",
        lambda: (_FakeWarp(), None, None, SimpleNamespace(DIRECTIONAL=5)),
    )

    render_warp._setup_lights(stage, ctx, Usd.TimeCode.Default(), "cpu")

    assert ctx.lights_active.data.tolist() == [True, True, True]
    assert ctx.lights_type.data.tolist() == [5, 5, 5]
    assert ctx.lights_orientation.data.shape == (3, 3)


def test_setup_render_context_legacy_api(monkeypatch):
    from pxr import Gf, Usd, UsdGeom

    class LegacyRenderContext:
        class Options:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def __init__(self, *, world_count, options, device):
            self.world_count = world_count
            self.options = options
            self.device = device
            self.bounds_computed = False
            self.utils = SimpleNamespace(compute_mesh_bounds=self.compute_mesh_bounds)

        def compute_mesh_bounds(self):
            self.bounds_computed = True

    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    render_mesh = render_warp._RenderMesh(
        warp_mesh=SimpleNamespace(id=77),
        vertices=np.zeros((3, 3), dtype=np.float32),
        indices=np.array([0, 1, 2], dtype=np.int32),
    )

    monkeypatch.setattr(
        render_warp,
        "_import_warp",
        lambda: (_FakeWarp(), LegacyRenderContext, 9, None),
    )

    ctx = _setup_render_context(
        [render_mesh],
        [mesh.GetPrim()],
        Usd.TimeCode.Default(),
        device="cpu",
        enable_shadows=False,
        enable_backface_culling=False,
    )

    assert ctx.options.kwargs["enable_shadows"] is False
    assert ctx.options.kwargs["enable_backface_culling"] is False
    assert ctx.bounds_computed is True
    assert ctx.mesh_ids.data.tolist() == [77]
    assert ctx.shape_count_total == 1
    assert ctx.shape_count_enabled == 1
    assert ctx.shape_colors.data.tolist() == [[0.8, 0.8, 0.8, 1.0]]


def test_newton_model_render_context_paths(monkeypatch):
    from pxr import Gf, Usd, UsdGeom

    class FakeRenderContext:
        class Config:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        def __init__(self, *, world_count, config, device):
            self.world_count = world_count
            self.config = config
            self.device = device
            self.utils = SimpleNamespace()

        def init_from_model(self, model, load_textures):
            model.load_textures = load_textures

    class FakeWarpWithTransform(_FakeWarp):
        def __init__(self):
            super().__init__()
            self.transform = lambda position, rotation: tuple(position) + tuple(
                rotation
            )

    class FakeModel:
        def __init__(self, count: int):
            self.shape_flags = _FakeWarpArray(np.zeros(count, dtype=np.int32))

        def state(self):
            return SimpleNamespace(name="state")

    class FakeModelBuilder:
        class ShapeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def __init__(self):
            self.shapes = []

        def add_shape_mesh(self, **kwargs):
            self.shapes.append(kwargs)

        def finalize(self, device):
            model = FakeModel(len(self.shapes))
            model.device = device
            return model

    fake_newton = types.ModuleType("newton")
    fake_newton.ModelBuilder = FakeModelBuilder
    fake_newton.Mesh = lambda vertices, indices, **kwargs: SimpleNamespace(
        vertices=vertices, indices=indices, kwargs=kwargs
    )
    fake_geometry = types.ModuleType("newton.geometry")
    fake_src = types.ModuleType("newton._src")
    fake_src_geometry = types.ModuleType("newton._src.geometry")
    fake_src_geometry.ShapeFlags = SimpleNamespace(VISIBLE=1)

    def fake_build_bvh_shape(model, state):
        model.bvh_built_for = state

    fake_geometry.build_bvh_shape = fake_build_bvh_shape
    fake_newton._src = fake_src
    fake_src.geometry = fake_src_geometry
    monkeypatch.setitem(sys.modules, "newton", fake_newton)
    monkeypatch.setitem(sys.modules, "newton.geometry", fake_geometry)
    monkeypatch.setitem(sys.modules, "newton._src", fake_src)
    monkeypatch.setitem(sys.modules, "newton._src.geometry", fake_src_geometry)
    monkeypatch.setattr(
        render_warp,
        "_import_warp",
        lambda: (FakeWarpWithTransform(), FakeRenderContext, None, None),
    )

    stage = Usd.Stage.CreateInMemory()
    visible_mesh = UsdGeom.Mesh.Define(stage, "/Visible")
    hidden_mesh = UsdGeom.Mesh.Define(stage, "/Hidden")
    for mesh in (visible_mesh, hidden_mesh):
        mesh.GetPointsAttr().Set(
            [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)]
        )
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    UsdGeom.Imageable(hidden_mesh.GetPrim()).CreateVisibilityAttr().Set(
        UsdGeom.Tokens.invisible
    )

    render_meshes = [
        render_warp._RenderMesh(
            warp_mesh=SimpleNamespace(id=index),
            vertices=np.zeros((3, 3), dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
        )
        for index in (1, 2)
    ]
    ctx = _setup_render_context(
        warp_meshes=render_meshes,
        mesh_prims=[visible_mesh.GetPrim(), hidden_mesh.GetPrim()],
        time_code=Usd.TimeCode.Default(),
        device="cpu",
        enable_shadows=False,
        enable_backface_culling=False,
        color_boost=1.0,
    )

    assert ctx._wu_render_config.enable_global_world is True
    assert ctx._wu_render_model.load_textures is False
    assert ctx._wu_render_model.bvh_built_for is ctx._wu_render_state
    assert ctx._wu_render_model.shape_flags.data.tolist() == [1, 0]
    assert ctx.shape_colors.data.shape == (2, 3)
    assert (
        render_warp._update_render_context_for_frame(
            ctx,
            mesh_prims=[visible_mesh.GetPrim(), hidden_mesh.GetPrim()],
            time_code=Usd.TimeCode.Default(),
            device="cpu",
            color_boost=1.0,
        )
        == 1
    )


def test_newton_1_4_render_context_contract(monkeypatch):
    class FakeRenderContext:
        class Config:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        def __init__(self, *, world_count, device):
            self.world_count = world_count
            self.device = device
            self.render_calls = []

        def render(self, *args, **kwargs):
            self.render_calls.append((args, kwargs))

    class FakeUtils:
        def __init__(self, render_context, render_config):
            self.render_context = render_context
            self.render_config = render_config

    fake_raytrace = types.ModuleType("newton._src.sensors.warp_raytrace")
    fake_raytrace.Utils = FakeUtils
    monkeypatch.setitem(sys.modules, "newton._src.sensors.warp_raytrace", fake_raytrace)
    monkeypatch.setattr(
        render_warp,
        "_import_warp",
        lambda: (_FakeWarp(), FakeRenderContext, None, None),
    )
    monkeypatch.setattr(
        render_warp,
        "_setup_newton_model_render_context",
        lambda *, ctx, **kwargs: ctx,
    )

    ctx = _setup_render_context(
        warp_meshes=[],
        mesh_prims=[],
        time_code=None,
        device="cpu",
        enable_shadows=False,
        enable_backface_culling=False,
    )

    assert ctx.world_count == 1
    assert ctx.device == "cpu"
    assert ctx.utils.render_context is ctx
    assert ctx.utils.render_config is ctx._wu_render_config
    assert ctx._wu_render_config.enable_shadows is False
    assert ctx._wu_render_config.enable_backface_culling is False
    assert ctx._wu_render_config_on_render is True

    model = object()
    state = object()
    ctx._wu_render_model = model
    ctx._wu_render_state = state
    render_warp._render_context_render(ctx, color_image="color")

    assert ctx.render_calls == [
        (
            (model, state),
            {"color_image": "color", "config": ctx._wu_render_config},
        )
    ]


def test_update_render_context_legacy_api(monkeypatch):
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    visible_mesh = UsdGeom.Mesh.Define(stage, "/Visible")
    hidden_mesh = UsdGeom.Mesh.Define(stage, "/Hidden")
    for mesh in (visible_mesh, hidden_mesh):
        mesh.GetPointsAttr().Set(
            [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)]
        )
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    UsdGeom.Imageable(hidden_mesh.GetPrim()).CreateVisibilityAttr().Set(
        UsdGeom.Tokens.invisible
    )

    monkeypatch.setattr(
        render_warp, "_import_warp", lambda: (_FakeWarp(), None, None, None)
    )
    ctx = SimpleNamespace()

    visible_count = render_warp._update_render_context_for_frame(
        ctx,
        mesh_prims=[visible_mesh.GetPrim(), hidden_mesh.GetPrim()],
        time_code=Usd.TimeCode.Default(),
        device="cpu",
        color_boost=1.0,
    )

    assert visible_count == 1
    assert ctx.shape_enabled.data.tolist() == [0]
    assert ctx.shape_count_enabled == 1
    assert ctx.bvh_shapes is None
    assert ctx.shape_colors.data.tolist() == [
        [0.8, 0.8, 0.8, 1.0],
        [0.8, 0.8, 0.8, 1.0],
    ]


def test_render_context_render_without_newton_model():
    calls = []
    ctx = SimpleNamespace(render=lambda **kwargs: calls.append(kwargs))

    render_warp._render_context_render(ctx, color_image="color")

    assert calls == [{"color_image": "color"}]


def test_render_context_render_with_newton_model():
    calls = []
    model = object()
    state = object()
    ctx = SimpleNamespace(
        _wu_render_model=model,
        _wu_render_state=state,
        render=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    render_warp._render_context_render(ctx, color_image="color")

    assert calls == [((model, state), {"color_image": "color"})]


def test_clear_render_outputs_zeros_present_outputs():
    color = _FakeWarpArray(np.ones((1, 1), dtype=np.float32))
    depth = _FakeWarpArray(np.ones((1, 1), dtype=np.float32))

    render_warp._clear_render_outputs(
        {"color_image": color, "depth_image": depth, "normal_image": None}
    )

    assert color.data.sum() == 0
    assert depth.data.sum() == 0


def test_camera_helpers_fallbacks():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.GetFocalLengthAttr().Set(0.0)
    camera.GetVerticalApertureAttr().Set(24.0)

    assert render_warp._compute_camera_fov(
        stage, "/Camera", Usd.TimeCode.Default()
    ) == pytest.approx(math.radians(45.0))
    assert render_warp._get_camera_transforms(
        stage, ["/Missing"], Usd.TimeCode.Default()
    ) == [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

    camera.GetFocalLengthAttr().Set(50.0)
    assert render_warp._compute_camera_fov(
        stage, "/Camera", Usd.TimeCode.Default()
    ) == pytest.approx(2.0 * math.atan(24.0 / 100.0))
    assert (
        len(
            render_warp._get_camera_transforms(
                stage, ["/Camera"], Usd.TimeCode.Default()
            )
        )
        == 1
    )


def test_render_all_cameras_with_fake_warp_no_meshes(monkeypatch):
    from pxr import Usd

    monkeypatch.setattr(
        render_warp, "_import_warp", lambda: (_FakeWarp(), None, None, None)
    )
    monkeypatch.setattr(render_warp, "_extract_meshes", lambda *args: ([], []))

    result = render_warp.render_all_cameras(
        stage=Usd.Stage.CreateInMemory(),
        image_width=1,
        image_height=1,
        cameras=["/Camera"],
        frames="0",
        device="cpu",
    )

    assert result["successful_cameras"] == 0
    assert result["failed_cameras"] == 1
    assert result["results"][0]["error"] == "No meshes found in stage"


def test_render_all_cameras_with_fake_warp_normal_sensor(monkeypatch):
    from pxr import Usd

    class FakeContext:
        def __init__(self):
            self.utils = SimpleNamespace(
                compute_pinhole_camera_rays=lambda *args: "camera-rays",
                create_color_image_output=lambda width, height, num_cameras: (
                    _FakeWarpArray(
                        np.zeros((1, num_cameras, height, width), dtype=np.uint32)
                    )
                ),
                create_depth_image_output=lambda width, height, num_cameras: (
                    _FakeWarpArray(
                        np.zeros((1, num_cameras, height, width), dtype=np.float32)
                    )
                ),
                create_normal_image_output=lambda width, height, num_cameras: (
                    _FakeWarpArray(
                        np.zeros((1, num_cameras, height, width, 3), dtype=np.float32)
                    )
                ),
            )

    def fake_render(ctx, **kwargs):
        kwargs["color_image"].data[...] = 1 | (2 << 8) | (3 << 16)
        kwargs["normal_image"].data[...] = [0.0, 1.0, 0.0]

    monkeypatch.setattr(
        render_warp, "_import_warp", lambda: (_FakeWarp(), None, None, None)
    )
    monkeypatch.setattr(
        render_warp,
        "_extract_meshes",
        lambda stage, time_code, device: ([object()], [object()]),
    )
    monkeypatch.setattr(
        render_warp, "_setup_render_context", lambda **kwargs: FakeContext()
    )
    monkeypatch.setattr(render_warp, "_setup_lights", lambda *args: None)
    monkeypatch.setattr(render_warp, "_compute_camera_fov", lambda *args: 0.5)
    monkeypatch.setattr(
        render_warp,
        "_get_camera_transforms",
        lambda stage, cameras, time_code: [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in cameras
        ],
    )
    monkeypatch.setattr(
        render_warp, "_update_render_context_for_frame", lambda *args, **kwargs: 1
    )
    monkeypatch.setattr(render_warp, "_render_context_render", fake_render)

    result = render_warp.render_all_cameras(
        stage=Usd.Stage.CreateInMemory(),
        image_width=2,
        image_height=2,
        cameras=None,
        frames="0",
        sensors=["normal"],
        device="cpu",
    )

    assert result["successful_cameras"] == 1
    assert result["failed_cameras"] == 0
    assert result["results"][0]["camera"] == "/Camera"
    assert result["results"][0]["images"][0].size == (2, 2)
    np.testing.assert_allclose(
        result["results"][0]["sensors"]["normal"][0], np.full((2, 2, 3), [0, 1, 0])
    )

    empty_frames = render_warp.render_all_cameras(
        stage=Usd.Stage.CreateInMemory(),
        image_width=1,
        image_height=1,
        cameras=["/Camera"],
        frames=",",
        device="cpu",
    )
    assert empty_frames["successful_cameras"] == 0
    assert empty_frames["failed_cameras"] == 1
    assert empty_frames["results"][0]["error"] == "No images produced"


def test_render_all_cameras_with_fake_warp_depth_and_hidden_frame(monkeypatch):
    from pxr import Usd

    class FakeContext:
        def __init__(self):
            self.utils = SimpleNamespace(
                compute_pinhole_camera_rays=lambda *args: "camera-rays",
                create_color_image_output=lambda width, height, num_cameras: (
                    _FakeWarpArray(np.full((1, num_cameras, height, width), 255))
                ),
                create_depth_image_output=lambda width, height, num_cameras: (
                    _FakeWarpArray(np.ones((1, num_cameras, height, width)))
                ),
                create_normal_image_output=lambda width, height, num_cameras: (
                    _FakeWarpArray(np.ones((1, num_cameras, height, width, 3)))
                ),
            )

    monkeypatch.setattr(
        render_warp, "_import_warp", lambda: (_FakeWarp(), None, None, None)
    )
    monkeypatch.setattr(
        render_warp,
        "_extract_meshes",
        lambda stage, time_code, device: ([object()], [object()]),
    )
    monkeypatch.setattr(
        render_warp, "_setup_render_context", lambda **kwargs: FakeContext()
    )
    monkeypatch.setattr(render_warp, "_setup_lights", lambda *args: None)
    monkeypatch.setattr(render_warp, "_compute_camera_fov", lambda *args: 0.5)
    monkeypatch.setattr(
        render_warp,
        "_get_camera_transforms",
        lambda stage, cameras, time_code: [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in cameras
        ],
    )
    monkeypatch.setattr(
        render_warp, "_update_render_context_for_frame", lambda *args, **kwargs: 0
    )
    monkeypatch.setattr(
        render_warp,
        "_render_context_render",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("hidden frames should not render")
        ),
    )

    result = render_warp.render_all_cameras(
        stage=Usd.Stage.CreateInMemory(),
        image_width=2,
        image_height=2,
        cameras=["/Camera"],
        frames="0",
        sensors=["depth"],
        device="cpu",
    )

    assert result["successful_cameras"] == 1
    assert np.asarray(result["results"][0]["images"][0])[:, :, :3].sum() == 0
    assert result["results"][0]["sensors"]["depth"][0].shape == (2, 2)


class TestWarpBackendSensorSupport:
    """Test sensor capability methods without requiring warp."""

    def test_supported_sensor_modes_class_var(self):
        from world_understanding.functions.graphics.rendering import (
            WarpRenderingBackend,
        )

        assert "depth" in WarpRenderingBackend.SUPPORTED_SENSOR_MODES
        assert "normal" in WarpRenderingBackend.SUPPORTED_SENSOR_MODES

    def test_supports_sensors_returns_true(self):
        from world_understanding.functions.graphics.rendering import (
            WarpRenderingBackend,
        )

        backend = WarpRenderingBackend()
        assert backend.supports_sensors() is True

    def test_get_supported_sensor_modes(self):
        from world_understanding.functions.graphics.rendering import (
            WarpRenderingBackend,
        )

        backend = WarpRenderingBackend()
        modes = backend.get_supported_sensor_modes()
        assert "depth" in modes
        assert "normal" in modes
        # Should return a copy
        modes.append("fake")
        assert "fake" not in backend.get_supported_sensor_modes()


class TestWarpBackendInit:
    """Test WarpRenderingBackend initialization."""

    def test_default_params(self):
        from world_understanding.functions.graphics.rendering import (
            WarpRenderingBackend,
        )

        backend = WarpRenderingBackend()
        assert backend._device == "cuda:0"
        assert backend._color_boost == 3.0
        assert backend._enable_shadows is True
        assert backend._enable_backface_culling is True

    def test_custom_params(self):
        from world_understanding.functions.graphics.rendering import (
            WarpRenderingBackend,
        )

        backend = WarpRenderingBackend(
            device="cuda:1",
            color_boost=2.0,
            enable_shadows=False,
            enable_backface_culling=False,
        )
        assert backend._device == "cuda:1"
        assert backend._color_boost == 2.0
        assert backend._enable_shadows is False
        assert backend._enable_backface_culling is False


def test_warp_backend_backface_culling_back_mode(monkeypatch):
    from pxr import Usd

    from world_understanding.functions.graphics.rendering import WarpRenderingBackend

    captured: dict[str, object] = {}

    def fake_render_all_cameras(**kwargs):
        captured.update(kwargs)
        return {"successful_cameras": 1}

    monkeypatch.setattr(render_warp, "render_all_cameras", fake_render_all_cameras)

    backend = WarpRenderingBackend()
    result = backend.render(
        stage=Usd.Stage.CreateInMemory(),
        cameras=["/Camera"],
        image_width=1,
        image_height=1,
        cull_style="back",
    )

    assert result["successful_cameras"] == 1
    assert captured["enable_backface_culling"] is True


# ---------------------------------------------------------------------------
# Integration tests (require CUDA GPU + warp + Newton warp_raytrace)
# ---------------------------------------------------------------------------


def test_import_warp_supports_installed_newton_api_without_cuda():
    pytest.importorskip("warp")
    pytest.importorskip("newton")

    wp, render_context, mesh_shape_type_int, render_light_type = _import_warp()

    assert wp is not None
    assert hasattr(render_context, "Config") or hasattr(render_context, "Options")
    assert isinstance(mesh_shape_type_int, int)
    assert render_light_type is not None


def test_setup_render_context_supports_newton_model_api_without_cuda():
    pytest.importorskip("warp")
    pytest.importorskip("newton")

    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/Tri")
    mesh.GetPointsAttr().Set(
        [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    )
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    mesh.GetDisplayColorAttr().Set([(0.4, 0.5, 0.6)])

    render_meshes, mesh_prims = _extract_meshes(stage, Usd.TimeCode(0), "cpu")
    ctx = _setup_render_context(
        render_meshes,
        mesh_prims,
        Usd.TimeCode(0),
        device="cpu",
    )

    if not hasattr(ctx.utils, "compute_mesh_bounds"):
        assert hasattr(ctx, "_wu_render_model")
        assert ctx._wu_render_config.enable_global_world is True
        assert ctx._wu_render_model.bvh_shape_count_enabled == 1
        assert ctx._wu_render_model.bvh_shapes is not None


_has_warp = False
try:
    import warp as wp

    wp.init()
    # Check we have a CUDA device
    if wp.is_cuda_available():
        from newton._src.sensors.warp_raytrace import RenderContext  # noqa: F401

        _has_warp = True
except (ImportError, RuntimeError):
    pass

requires_warp = pytest.mark.skipif(not _has_warp, reason="warp + CUDA not available")


@pytest.fixture
def simple_usd_stage_with_mesh():
    """Create a simple USD stage with a triangulated mesh and camera."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    # Create a simple quad mesh (2 triangles)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Quad")
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(-1, -1, 0),
            Gf.Vec3f(1, -1, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(-1, 1, 0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    mesh.GetDisplayColorAttr().Set([(0.8, 0.2, 0.2)])

    # Camera looking at the quad
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.GetFocalLengthAttr().Set(50.0)
    camera.GetVerticalApertureAttr().Set(24.0)
    camera.GetHorizontalApertureAttr().Set(36.0)

    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))

    return stage


@requires_warp
class TestWarpIntegrationSingleCamera:
    """Integration test: single camera rendering with Warp."""

    def test_render_single_camera(self, simple_usd_stage_with_mesh):
        from world_understanding.functions.graphics.render_warp import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=simple_usd_stage_with_mesh,
            image_width=64,
            image_height=64,
            cameras=["/Camera"],
            frames="0",
        )

        assert result["total_cameras"] == 1
        assert result["successful_cameras"] == 1
        assert result["failed_cameras"] == 0
        assert len(result["results"]) == 1
        assert result["results"][0]["frame_count"] == 1
        assert len(result["results"][0]["images"]) == 1

        # Check image dimensions
        img = result["results"][0]["images"][0]
        assert img.size == (64, 64)
        assert np.asarray(img)[:, :, :3].sum() > 0

    def test_render_with_depth_sensor(self, simple_usd_stage_with_mesh):
        from world_understanding.functions.graphics.render_warp import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=simple_usd_stage_with_mesh,
            image_width=64,
            image_height=64,
            cameras=["/Camera"],
            frames="0",
            sensors=["depth"],
        )

        assert result["successful_cameras"] == 1
        sensors = result["results"][0]["sensors"]
        assert "depth" in sensors
        assert 0 in sensors["depth"]
        depth_arr = sensors["depth"][0]
        assert depth_arr.shape == (64, 64)

    def test_render_no_meshes_returns_error(self):
        """Rendering a stage with no meshes should report failure."""
        from pxr import Gf, Usd, UsdGeom

        from world_understanding.functions.graphics.render_warp import (
            render_all_cameras,
        )

        stage = Usd.Stage.CreateInMemory()
        camera = UsdGeom.Camera.Define(stage, "/Camera")
        camera.GetFocalLengthAttr().Set(50.0)
        camera.GetVerticalApertureAttr().Set(24.0)
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))

        result = render_all_cameras(
            stage=stage,
            image_width=64,
            image_height=64,
            cameras=["/Camera"],
            frames="0",
        )

        assert result["successful_cameras"] == 0
        assert result["failed_cameras"] == 1
        assert "error" in result["results"][0]


@requires_warp
class TestWarpIntegrationMultiFrame:
    """Integration test: multi-frame rendering with visibility."""

    def test_render_multiple_frames(self, simple_usd_stage_with_mesh):
        from world_understanding.functions.graphics.render_warp import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=simple_usd_stage_with_mesh,
            image_width=64,
            image_height=64,
            cameras=["/Camera"],
            frames="0:2",
        )

        assert result["successful_cameras"] == 1
        assert result["results"][0]["frame_count"] == 3  # frames 0, 1, 2

    def test_all_hidden_frame_clears_previous_pixels(
        self, simple_usd_stage_with_mesh, monkeypatch
    ):
        from pxr import Usd, UsdGeom

        from world_understanding.functions.graphics import render_warp

        mesh_prim = simple_usd_stage_with_mesh.GetPrimAtPath("/World/Quad")
        visibility_attr = UsdGeom.Imageable(mesh_prim).CreateVisibilityAttr()
        visibility_attr.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(0))
        visibility_attr.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(1))

        def _fill_visible_frame(_ctx, **render_kwargs):
            render_kwargs["color_image"].fill_(255)

        monkeypatch.setattr(render_warp, "_render_context_render", _fill_visible_frame)

        result = render_warp.render_all_cameras(
            stage=simple_usd_stage_with_mesh,
            image_width=64,
            image_height=64,
            cameras=["/Camera"],
            frames="0,1",
        )

        assert result["successful_cameras"] == 1
        images = result["results"][0]["images"]
        assert len(images) == 2
        assert np.asarray(images[0])[:, :, :3].sum() > 0
        assert np.asarray(images[1])[:, :, :3].sum() == 0


@requires_warp
class TestWarpIntegrationBackend:
    """Integration test: WarpRenderingBackend class."""

    def test_backend_render(self, simple_usd_stage_with_mesh):
        from world_understanding.functions.graphics.rendering import (
            WarpRenderingBackend,
        )

        backend = WarpRenderingBackend()
        result = backend.render(
            stage=simple_usd_stage_with_mesh,
            cameras=["/Camera"],
            image_width=64,
            image_height=64,
            frames="0",
        )

        assert result["total_cameras"] == 1
        assert result["successful_cameras"] == 1
        assert len(result["results"][0]["images"]) == 1

    def test_backend_cull_style_none(self, simple_usd_stage_with_mesh):
        """cull_style='none' should disable backface culling."""
        from world_understanding.functions.graphics.rendering import (
            WarpRenderingBackend,
        )

        backend = WarpRenderingBackend()
        # Should not raise
        result = backend.render(
            stage=simple_usd_stage_with_mesh,
            cameras=["/Camera"],
            image_width=64,
            cull_style="none",
            frames="0",
        )
        assert result["successful_cameras"] == 1
