# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional smoke test for the repository OpenVDB tools API v2."""

from __future__ import annotations

import sys

import numpy as np
import openvdb


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    triangles = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.int32,
    )
    return points, triangles


def _unaligned_array(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    item_count = int(np.prod(shape, dtype=np.int64))
    storage = np.zeros(item_count * dtype.itemsize + 1, dtype=np.uint8)
    result = np.ndarray(shape, dtype=dtype, buffer=storage, offset=1)
    assert result.flags.c_contiguous and not result.flags.aligned
    return result


def _assert_mesh_arrays(mesh: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    points, triangles, quads = mesh
    assert points.ndim == 2 and points.shape[1] == 3 and points.dtype == np.float32
    assert triangles.ndim == 2 and triangles.shape[1] == 3
    assert triangles.dtype == np.int32
    assert quads.ndim == 2 and quads.shape[1] == 4 and quads.dtype == np.int32
    assert points.flags.writeable and triangles.flags.writeable and quads.flags.writeable
    assert len(points) > 0 and len(triangles) + len(quads) > 0


def main() -> None:
    if sys.flags.optimize:
        raise RuntimeError("native functional smoke requires Python assertions")
    tools = openvdb.tools
    assert tools.API_VERSION == 3
    expected_tools = {
        "active_value_mask",
        "csg_difference",
        "csg_intersection",
        "csg_union",
        "extract_enclosed_region",
        "level_set_mean",
        "level_set_normalize",
        "level_set_offset",
        "level_set_rebuild",
        "mesh_to_level_set",
        "mesh_to_unsigned_distance_field",
        "resample_to_match",
        "sample_gradients",
        "sample_values",
        "scalar_mean",
        "topology_to_level_set",
        "volume_to_mesh",
    }
    assert expected_tools.issubset(dir(tools))

    sphere = openvdb.createLevelSetSphere(1.0, (0, 0, 0), voxelSize=0.2, halfWidth=3.0)
    sphere_count = sphere.activeVoxelCount()
    other_sphere = openvdb.createLevelSetSphere(1.0, (0.5, 0, 0), voxelSize=0.2, halfWidth=3.0)
    csg_points = np.array([[-0.8, 0, 0], [0.8, 0, 0], [1.3, 0, 0]], dtype=np.float32)
    union = tools.csg_union(sphere, other_sphere, thread_count=2)
    intersection = tools.csg_intersection(sphere, other_sphere, thread_count=2)
    difference = tools.csg_difference(sphere, other_sphere, thread_count=2)
    np.testing.assert_array_less(tools.sample_values(union, csg_points), 0)
    assert tuple(tools.sample_values(intersection, csg_points) < 0) == (False, True, False)
    assert tuple(tools.sample_values(difference, csg_points) < 0) == (True, False, False)
    try:
        tools.csg_union(
            sphere,
            other_sphere,
            max_active_voxels=sphere_count + other_sphere.activeVoxelCount() - 1,
        )
    except ValueError as error:
        assert "max_active_voxels" in str(error)
    else:
        raise AssertionError("csg_union ignored a caller active-voxel ceiling")

    wider_sphere = openvdb.createLevelSetSphere(1.0, (0, 0, 0), voxelSize=0.2, halfWidth=5.0)
    try:
        tools.csg_union(sphere, wider_sphere)
    except ValueError as error:
        assert "matching level-set background" in str(error)
    else:
        raise AssertionError("csg_union accepted mismatched level-set background widths")

    surface_point = np.array([[1, 0, 0]], dtype=np.float32)
    mean = tools.level_set_mean(sphere, width=1, iterations=1, thread_count=2)
    try:
        tools.level_set_mean(sphere, width=256, iterations=1024)
    except ValueError as error:
        assert "aggregate work" in str(error)
    else:
        raise AssertionError("level_set_mean accepted excessive aggregate filter work")
    tiled_level_set = openvdb.FloatGrid(background=0.6)
    tiled_level_set.transform = openvdb.createLinearTransform(0.2)
    tiled_level_set.gridClass = "level set"
    tiled_level_set.fill((0, 0, 0), (7, 7, 7), 0.5, active=True)
    for label, operation in (
        ("mean", lambda: tools.level_set_mean(tiled_level_set)),
        ("normalize", lambda: tools.level_set_normalize(tiled_level_set)),
        ("offset", lambda: tools.level_set_offset(tiled_level_set, 0.1)),
    ):
        try:
            operation()
        except ValueError as error:
            assert "active tiles" in str(error)
        else:
            raise AssertionError(f"level-set {label} accepted an active tile")
    offset = tools.level_set_offset(sphere, 0.1, thread_count=2)
    integer_offset = tools.level_set_offset(sphere, 0, thread_count=2)
    numpy_offset = tools.level_set_offset(sphere, np.float32(0.0), thread_count=np.int64(1))
    assert abs(float(tools.sample_values(mean, surface_point)[0])) > 0.01
    np.testing.assert_allclose(tools.sample_values(offset, surface_point), [0.1], atol=0.02)
    np.testing.assert_allclose(tools.sample_values(integer_offset, surface_point), [0], atol=0.02)
    np.testing.assert_allclose(tools.sample_values(numpy_offset, surface_point), [0], atol=0.02)

    distorted = sphere.deepCopy()
    distorted.mapOn(lambda value: value * 2)
    distorted.background *= 2
    normalized = tools.level_set_normalize(distorted, thread_count=2)
    distorted_gradient = tools.sample_gradients(distorted, surface_point)[0, 0]
    normalized_gradient = tools.sample_gradients(normalized, surface_point)[0, 0]
    assert abs(normalized_gradient - 1) < abs(distorted_gradient - 1)

    for result in (union, intersection, difference, mean, offset, normalized):
        assert result.gridClass == "level set"
        assert not result.sharesWith(sphere)
    assert sphere.activeVoxelCount() == sphere_count

    world_points = np.array([[1, 0, 0], [0, 0, 0], [2, 0, 0]], dtype=np.float32)
    for interpolation in ("nearest", "linear", "quadratic"):
        values = tools.sample_values(
            sphere, world_points, interpolation=interpolation, thread_count=2
        )
        assert values.shape == (3,) and values.dtype == np.float32
        assert np.isfinite(values).all() and values.flags.writeable
    gradients = tools.sample_gradients(
        sphere, world_points, interpolation="quadratic", thread_count=2
    )
    assert gradients.shape == (3, 3) and gradients.dtype == np.float32
    np.testing.assert_allclose(gradients[0], [1, 0, 0], atol=0.15)
    empty = np.empty((0, 3), dtype=np.float32)
    assert tools.sample_values(sphere, empty).shape == (0,)
    assert tools.sample_gradients(sphere, empty).shape == (0, 3)

    invalid_transform = openvdb.createLinearTransform(
        matrix=[
            [0.2, 0, 0, 0],
            [0, 0.2, 0, 0],
            [0, 0, 0.2, 0],
            [float("nan"), 0, 0, 1],
        ]
    )
    invalid_grid = sphere.deepCopy()
    invalid_grid.transform = invalid_transform
    for label, operation in (
        ("level-set offset", lambda: tools.level_set_offset(invalid_grid, 0.1)),
        ("sampling", lambda: tools.sample_values(invalid_grid, world_points)),
        ("meshing", lambda: tools.volume_to_mesh(invalid_grid)),
        ("resampling source", lambda: tools.resample_to_match(invalid_grid, sphere)),
        ("resampling reference", lambda: tools.resample_to_match(sphere, invalid_grid)),
    ):
        try:
            operation()
        except ValueError as error:
            assert "transform" in str(error)
        else:
            raise AssertionError(f"{label} accepted a non-finite transform")

    nonlinear_grid = sphere.deepCopy()
    nonlinear_grid.transform = openvdb.createFrustumTransform(
        (0, 0, 0), (10, 10, 10), 1.0, 10.0, 1.0
    )
    try:
        tools.level_set_offset(nonlinear_grid, 0.1)
    except ValueError as error:
        assert "linear transform" in str(error)
    else:
        raise AssertionError("level_set_offset accepted a nonlinear transform")

    huge_translation_grid = sphere.deepCopy()
    huge_translation_grid.transform = openvdb.createLinearTransform(
        matrix=[
            [0.2, 0, 0, 0],
            [0, 0.2, 0, 0],
            [0, 0, 0.2, 0],
            [1e15, 0, 0, 1],
        ]
    )
    try:
        tools.volume_to_mesh(huge_translation_grid)
    except ValueError as error:
        assert "float32 world coordinates" in str(error)
    else:
        raise AssertionError("volume_to_mesh accepted a numerically unstable transform")

    far_sample = np.full((1, 3), np.finfo(np.float32).max, dtype=np.float32)
    try:
        tools.sample_values(sphere, far_sample)
    except ValueError as error:
        assert "int32 index range" in str(error)
    else:
        raise AssertionError("sample_values accepted out-of-range world coordinates")

    edge_topology = openvdb.BoolGrid()
    edge_topology.getAccessor().setValueOn((np.iinfo(np.int32).max, 0, 0), True)
    try:
        tools.topology_to_level_set(edge_topology, dilation=1)
    except ValueError as error:
        assert "int32 index boundary" in str(error)
    else:
        raise AssertionError("topology_to_level_set accepted an overflowing index domain")

    invalid_level_set = sphere.deepCopy()
    invalid_level_set.getAccessor().setValueOn((0, 0, 0), float("nan"))
    try:
        tools.level_set_mean(invalid_level_set)
    except ValueError as error:
        assert "finite stored values" in str(error)
    else:
        raise AssertionError("level_set_mean accepted a NaN level-set value")

    invalid_inactive_level_set = sphere.deepCopy()
    invalid_inactive_level_set.getAccessor().setValueOff((5, 0, 0), float("nan"))
    try:
        tools.level_set_normalize(invalid_inactive_level_set)
    except ValueError as error:
        assert "finite stored values" in str(error)
    else:
        raise AssertionError("level_set_normalize accepted a stored inactive NaN value")

    value_grid = openvdb.FloatGrid()
    value_grid.fill((0, 0, 0), (15, 15, 15), 2.0, active=True)
    value_grid.fill((20, 20, 20), (20, 20, 20), float("nan"), active=True)
    mask = tools.active_value_mask(value_grid, min_value=2.0, max_value=2.0, thread_count=2)
    assert isinstance(mask, openvdb.BoolGrid)
    assert mask.activeVoxelCount() == 16**3
    assert tools.active_value_mask(value_grid, thread_count=2).activeVoxelCount() == 16**3
    assert value_grid.activeVoxelCount() == 16**3 + 1
    assert tools.active_value_mask(value_grid, min_value=1e300).activeVoxelCount() == 0

    surface_mask = tools.active_value_mask(sphere, min_value=-0.11, max_value=0.11, thread_count=2)
    topology_level_set = tools.topology_to_level_set(
        surface_mask,
        half_width=3,
        closing_steps=0,
        dilation=0,
        smoothing_steps=0,
        thread_count=2,
    )
    assert topology_level_set.gridClass == "level set"
    assert topology_level_set.activeVoxelCount() > 0
    enclosed = tools.extract_enclosed_region(sphere, thread_count=2)
    assert isinstance(enclosed, openvdb.BoolGrid) and enclosed.activeVoxelCount() > 0

    reference = openvdb.FloatGrid()
    reference.transform = openvdb.createLinearTransform(0.1)
    resampled = tools.resample_to_match(sphere, reference, interpolation="linear", thread_count=2)
    np.testing.assert_allclose(resampled.transform.voxelSize(), [0.1, 0.1, 0.1])
    np.testing.assert_allclose(sphere.transform.voxelSize(), [0.2, 0.2, 0.2])
    assert resampled.gridClass == "level set" and not resampled.sharesWith(sphere)
    assert resampled.activeVoxelCount() > 0
    np.testing.assert_allclose(tools.sample_values(resampled, surface_point), [0], atol=0.03)

    dense_source = openvdb.FloatGrid()
    dense_source.fill((-1, -1, -1), (1, 1, 1), 1.0, active=True)
    tiny_reference = openvdb.FloatGrid()
    tiny_reference.transform = openvdb.createLinearTransform(0.001)
    try:
        tools.resample_to_match(dense_source, tiny_reference, interpolation="nearest")
    except ValueError as error:
        assert "dense voxel budget" in str(error)
    else:
        raise AssertionError("resample_to_match accepted an excessive output domain")

    inactive_source = openvdb.FloatGrid(background=0.0)
    inactive_source.getAccessor().setValueOff((0, 0, 0), 1.0)
    assert inactive_source.activeVoxelCount() == 0
    try:
        tools.resample_to_match(inactive_source, tiny_reference, interpolation="nearest")
    except ValueError as error:
        assert "dense voxel budget" in str(error)
    else:
        raise AssertionError("resample_to_match ignored an inactive stored leaf")

    try:
        tools.level_set_offset(sphere, 1e10)
    except ValueError as error:
        assert "supported offset" in str(error)
    else:
        raise AssertionError("level_set_offset accepted an excessive distance")

    mesh_points, mesh_triangles = _tetrahedron()
    far_mesh_points = mesh_points * np.float32(2048) + np.float32(1e10)
    for operation in (
        tools.mesh_to_level_set,
        tools.mesh_to_unsigned_distance_field,
    ):
        try:
            operation(far_mesh_points, mesh_triangles, voxel_size=1.0)
        except ValueError as error:
            assert "int32 index range" in str(error)
        else:
            raise AssertionError(f"{operation.__name__} accepted out-of-range mesh coordinates")

    tiny_mesh_points = mesh_points * np.float32(1e-5)
    for operation in (
        tools.mesh_to_level_set,
        tools.mesh_to_unsigned_distance_field,
    ):
        try:
            operation(tiny_mesh_points, mesh_triangles, voxel_size=1e-5)
        except ValueError as error:
            assert "voxel_size" in str(error)
        else:
            raise AssertionError(f"{operation.__name__} accepted a nearly singular transform")

    mesh_level_set = tools.mesh_to_level_set(
        mesh_points,
        mesh_triangles,
        voxel_size=0.1,
        half_width=3.0,
        thread_count=2,
    )
    assert mesh_level_set.gridClass == "level set"
    for label, kwargs in (
        ("vertices", {"max_vertices": len(mesh_points) - 1}),
        ("faces", {"max_faces": len(mesh_triangles) - 1}),
        ("voxels", {"max_active_voxels": 1}),
    ):
        try:
            tools.mesh_to_level_set(mesh_points, mesh_triangles, voxel_size=0.1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"mesh_to_level_set ignored max_{label}")
    unsigned = tools.mesh_to_unsigned_distance_field(
        mesh_points,
        mesh_triangles,
        voxel_size=0.1,
        half_width=3.0,
        thread_count=2,
    )
    assert unsigned.gridClass == "unknown" and unsigned.background > 0
    smoothed = tools.scalar_mean(unsigned, width=1, iterations=1, thread_count=2)
    assert smoothed.gridClass == "unknown" and not smoothed.sharesWith(unsigned)
    expensive_scalar = openvdb.FloatGrid(background=0.0)
    expensive_scalar.fill((0, 0, 0), (99, 99, 99), 1.0, active=True)
    try:
        tools.scalar_mean(expensive_scalar, width=256, iterations=1024)
    except ValueError as error:
        assert "aggregate work" in str(error)
    else:
        raise AssertionError("scalar_mean accepted excessive aggregate filter work")
    tiled_scalar = openvdb.FloatGrid(background=0.0)
    tiled_scalar.fill((0, 0, 0), (7, 7, 7), 1.0, active=True)
    tiled_smoothed = tools.scalar_mean(tiled_scalar, width=1, iterations=1, thread_count=1)
    assert tiled_smoothed.getAccessor().getValue((0, 0, 0)) < 1.0
    try:
        tools.scalar_mean(sphere)
    except ValueError as error:
        assert "level_set_mean" in str(error)
    else:
        raise AssertionError("scalar_mean accepted a GRID_LEVEL_SET input")
    rebuilt = tools.level_set_rebuild(
        unsigned,
        isovalue=0.15,
        exterior_width=3.0,
        interior_width=3.0,
        thread_count=2,
    )
    assert rebuilt.gridClass == "level set" and rebuilt.activeVoxelCount() > 0

    _assert_mesh_arrays(
        tools.volume_to_mesh(
            unsigned,
            isovalue=0.15,
            adaptivity=0.0,
            relax_disoriented_triangles=False,
            thread_count=2,
        )
    )
    sphere_mesh = tools.volume_to_mesh(sphere, isovalue=0, thread_count=2)
    _assert_mesh_arrays(sphere_mesh)
    mesh_vertex_count = len(sphere_mesh[0])
    mesh_face_count = len(sphere_mesh[1]) + len(sphere_mesh[2])
    for label, kwargs in (
        ("vertices", {"max_vertices": mesh_vertex_count - 1}),
        ("faces", {"max_faces": mesh_face_count - 1}),
        ("active voxels", {"max_active_voxels": sphere_count - 1}),
    ):
        try:
            tools.volume_to_mesh(sphere, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"volume_to_mesh ignored its {label} ceiling")

    wrong_dtype_operations = (
        (
            "sample_values points",
            lambda: tools.sample_values(sphere, world_points.astype(np.float64)),
        ),
        (
            "mesh_to_level_set points",
            lambda: tools.mesh_to_level_set(
                mesh_points.astype(np.float64), mesh_triangles, voxel_size=0.1
            ),
        ),
        (
            "mesh_to_level_set triangles",
            lambda: tools.mesh_to_level_set(
                mesh_points, mesh_triangles.astype(np.float64) + 0.25, voxel_size=0.1
            ),
        ),
    )
    for label, operation in wrong_dtype_operations:
        try:
            operation()
        except TypeError:
            pass
        else:
            raise AssertionError(f"{label} accepted an implicitly convertible dtype")

    unaligned_points = _unaligned_array(mesh_points.shape, np.dtype(np.float32))
    unaligned_points[:] = mesh_points
    unaligned_triangles = _unaligned_array(mesh_triangles.shape, np.dtype(np.int32))
    unaligned_triangles[:] = mesh_triangles
    for label, operation in (
        (
            "mesh points",
            lambda: tools.mesh_to_level_set(unaligned_points, mesh_triangles, voxel_size=0.1),
        ),
        (
            "mesh triangles",
            lambda: tools.mesh_to_level_set(mesh_points, unaligned_triangles, voxel_size=0.1),
        ),
        (
            "sample points",
            lambda: tools.sample_values(sphere, unaligned_points),
        ),
    ):
        try:
            operation()
        except ValueError as error:
            assert "aligned" in str(error)
        else:
            raise AssertionError(f"native tools accepted unaligned {label}")

    repeated_triangles = mesh_triangles.copy()
    repeated_triangles[0] = (0, 0, 1)
    try:
        tools.mesh_to_level_set(mesh_points, repeated_triangles, voxel_size=0.1)
    except ValueError as error:
        assert "repeat" in str(error)
    else:
        raise AssertionError("mesh_to_level_set accepted a repeated-index face")

    try:
        tools.sample_values(sphere, world_points, thread_count=33)
    except ValueError:
        pass
    else:
        raise AssertionError("sample_values accepted an excessive thread count")
    strict_numeric_operations = (
        (
            "thread_count",
            lambda value: tools.sample_values(sphere, world_points, thread_count=value),
        ),
        (
            "filter width",
            lambda value: tools.level_set_mean(sphere, width=value),
        ),
        (
            "topology step",
            lambda value: tools.topology_to_level_set(surface_mask, dilation=value),
        ),
        (
            "offset distance",
            lambda value: tools.level_set_offset(sphere, value),
        ),
        (
            "rebuild isovalue",
            lambda value: tools.level_set_rebuild(sphere, isovalue=value),
        ),
        (
            "rebuild exterior width",
            lambda value: tools.level_set_rebuild(sphere, exterior_width=value),
        ),
        (
            "rebuild interior width",
            lambda value: tools.level_set_rebuild(sphere, interior_width=value),
        ),
        (
            "level-set voxel size",
            lambda value: tools.mesh_to_level_set(mesh_points, mesh_triangles, voxel_size=value),
        ),
        (
            "unsigned-field half width",
            lambda value: tools.mesh_to_unsigned_distance_field(
                mesh_points, mesh_triangles, voxel_size=0.1, half_width=value
            ),
        ),
        (
            "active-mask minimum",
            lambda value: tools.active_value_mask(sphere, min_value=value),
        ),
        (
            "meshing isovalue",
            lambda value: tools.volume_to_mesh(sphere, isovalue=value),
        ),
        (
            "meshing adaptivity",
            lambda value: tools.volume_to_mesh(sphere, adaptivity=value),
        ),
    )
    for boolean_name, boolean_value in (
        ("Python bool", True),
        ("NumPy bool scalar", np.bool_(True)),
    ):
        for control_name, operation in strict_numeric_operations:
            try:
                operation(boolean_value)
            except TypeError:
                pass
            else:
                raise AssertionError(f"native tools accepted {boolean_name} for {control_name}")

    strict_integer_operations = (
        (
            "thread_count",
            lambda value: tools.sample_values(sphere, world_points, thread_count=value),
        ),
        ("filter width", lambda value: tools.level_set_mean(sphere, width=value)),
        ("topology step", lambda value: tools.topology_to_level_set(surface_mask, dilation=value)),
    )
    for control_name, operation in strict_integer_operations:
        try:
            operation(np.float32(1.0))
        except TypeError:
            pass
        else:
            raise AssertionError(f"native tools truncated a NumPy float for {control_name}")

    for array_value in (np.array(1.0), np.array([1.0]), np.array(True), np.array([True])):
        for control_name, operation in strict_numeric_operations:
            try:
                operation(array_value)
            except TypeError:
                pass
            else:
                raise AssertionError(f"native tools accepted an ndarray for {control_name}")

    for label, value in (
        ("integer true", 1),
        ("integer false", 0),
        ("NumPy bool scalar", np.bool_(True)),
    ):
        try:
            tools.volume_to_mesh(sphere, relax_disoriented_triangles=value)
        except TypeError:
            pass
        else:
            raise AssertionError(f"native tools accepted {label} for relax_disoriented_triangles")


if __name__ == "__main__":
    main()
