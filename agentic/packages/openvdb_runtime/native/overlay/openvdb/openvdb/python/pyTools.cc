// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetRebuild.h>
#include <openvdb/tools/LevelSetTracker.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/TopologyToLevelSet.h>
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/tools/VolumeToMesh.h>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace wu_openvdb_python {

struct StrictInt {
    int value = 0;
    operator int() const noexcept { return value; }
};

struct StrictFloat {
    float value = 0.0f;
    operator float() const noexcept { return value; }
};

struct StrictDouble {
    double value = 0.0;
    operator double() const noexcept { return value; }
};

} // namespace wu_openvdb_python

namespace nanobind::detail {

inline bool
isBooleanScalar(PyObject* source) noexcept
{
    if (PyBool_Check(source)) return true;
    if (PyLong_CheckExact(source) || PyFloat_CheckExact(source)) return false;

    PyObject* numpy = PyImport_ImportModule("numpy");
    if (numpy == nullptr) {
        PyErr_Clear();
        return true;
    }
    PyObject* numpyBool = PyObject_GetAttrString(numpy, "bool_");
    Py_DECREF(numpy);
    if (numpyBool == nullptr) {
        PyErr_Clear();
        return true;
    }
    const int result = PyObject_IsInstance(source, numpyBool);
    Py_DECREF(numpyBool);
    if (result < 0) {
        PyErr_Clear();
        return true;
    }
    return result == 1;
}

inline bool
isNumpyArray(PyObject* source) noexcept
{
    PyObject* numpy = PyImport_ImportModule("numpy");
    if (numpy == nullptr) {
        PyErr_Clear();
        return false;
    }
    PyObject* ndarray = PyObject_GetAttrString(numpy, "ndarray");
    Py_DECREF(numpy);
    if (ndarray == nullptr) {
        PyErr_Clear();
        return false;
    }
    const int result = PyObject_IsInstance(source, ndarray);
    Py_DECREF(ndarray);
    if (result < 0) {
        PyErr_Clear();
        return true;
    }
    return result == 1;
}

template<>
struct type_caster<wu_openvdb_python::StrictInt> {
    NB_TYPE_CASTER(wu_openvdb_python::StrictInt, const_name("int"))

    bool from_python(handle src, std::uint8_t flags, cleanup_list*) noexcept
    {
        if (isBooleanScalar(src.ptr()) || isNumpyArray(src.ptr()) || !PyIndex_Check(src.ptr())) {
            return false;
        }
        std::int32_t integer = 0;
        if (!load_i32(src.ptr(), flags, &integer)) return false;
        value.value = integer;
        return true;
    }

    static handle from_cpp(
        wu_openvdb_python::StrictInt src, rv_policy, cleanup_list*) noexcept
    {
        return PyLong_FromLong(src.value);
    }
};

template<>
struct type_caster<wu_openvdb_python::StrictFloat> {
    NB_TYPE_CASTER(wu_openvdb_python::StrictFloat, const_name("float"))

    bool from_python(handle src, std::uint8_t flags, cleanup_list*) noexcept
    {
        if (isBooleanScalar(src.ptr()) || isNumpyArray(src.ptr())) return false;
        return load_f32(src.ptr(), flags, &value.value);
    }

    static handle from_cpp(
        wu_openvdb_python::StrictFloat src, rv_policy, cleanup_list*) noexcept
    {
        return PyFloat_FromDouble(src.value);
    }
};

template<>
struct type_caster<wu_openvdb_python::StrictDouble> {
    NB_TYPE_CASTER(wu_openvdb_python::StrictDouble, const_name("float"))

    bool from_python(handle src, std::uint8_t flags, cleanup_list*) noexcept
    {
        if (isBooleanScalar(src.ptr()) || isNumpyArray(src.ptr())) return false;
        return load_f64(src.ptr(), flags, &value.value);
    }

    static handle from_cpp(
        wu_openvdb_python::StrictDouble src, rv_policy, cleanup_list*) noexcept
    {
        return PyFloat_FromDouble(src.value);
    }
};

} // namespace nanobind::detail

namespace nb = nanobind;

namespace {

using namespace openvdb::OPENVDB_VERSION_NAME;
using wu_openvdb_python::StrictDouble;
using wu_openvdb_python::StrictFloat;
using wu_openvdb_python::StrictInt;

using PointArray =
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using TriangleArray =
    nb::ndarray<const std::int32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using MeshData = std::tuple<std::vector<Vec3s>, std::vector<Vec3I>, std::vector<Vec4I>>;

constexpr int MAX_THREAD_COUNT = 32;
constexpr int MAX_FILTER_WIDTH = 256;
constexpr int MAX_FILTER_ITERATIONS = 1024;
constexpr int MAX_TOPOLOGY_STEPS = 1024;
constexpr float MAX_BAND_WIDTH = 1024.0f;
constexpr std::uint64_t MAX_INPUT_POINTS = 10'000'000;
constexpr std::uint64_t MAX_INPUT_FACES = 20'000'000;
constexpr std::uint64_t MAX_ACTIVE_VOXELS = 50'000'000;
constexpr std::uint64_t MAX_MESH_INPUT_ACTIVE_VOXELS = 10'000'000;
constexpr std::uint64_t MAX_GRID_MEMORY_BYTES = 1'073'741'824;
constexpr std::uint64_t MAX_FILTER_WORK = 1'000'000'000;
constexpr std::int64_t MAX_VOXEL_EXTENT = 1'000'000;

static_assert(sizeof(Vec3s) == 3 * sizeof(float), "Vec3s must be tightly packed");
static_assert(sizeof(Vec3I) == 3 * sizeof(std::int32_t), "Vec3I must be tightly packed");
static_assert(sizeof(Vec4I) == 4 * sizeof(std::int32_t), "Vec4I must be tightly packed");

enum class Interpolation { Nearest, Linear, Quadratic };

void
validateThreadCount(int threadCount)
{
    if (threadCount < 1 || threadCount > MAX_THREAD_COUNT) {
        std::ostringstream message;
        message << "thread_count must be between 1 and " << MAX_THREAD_COUNT;
        throw nb::value_error(message.str().c_str());
    }
}

std::uint64_t
validateResourceLimit(int value, std::uint64_t maximum, const char* argument)
{
    if (value < 1 || static_cast<std::uint64_t>(value) > maximum) {
        std::ostringstream message;
        message << argument << " must be between 1 and " << maximum;
        throw nb::value_error(message.str().c_str());
    }
    return static_cast<std::uint64_t>(value);
}

void
validateDenseDimensions(
    const std::int64_t dimensions[3],
    const char* argument,
    bool requireDenseBound,
    std::uint64_t maxDenseVoxels = MAX_ACTIVE_VOXELS)
{
    std::uint64_t denseVoxelCount = 1;
    for (int axis = 0; axis < 3; ++axis) {
        if (dimensions[axis] <= 0 || dimensions[axis] > MAX_VOXEL_EXTENT) {
            std::ostringstream message;
            message << argument << " exceeds the supported voxel extent";
            throw nb::value_error(message.str().c_str());
        }
        if (requireDenseBound) {
            const auto dimension = static_cast<std::uint64_t>(dimensions[axis]);
            if (denseVoxelCount > maxDenseVoxels / dimension) {
                std::ostringstream message;
                message << argument << " exceeds the supported dense voxel budget";
                throw nb::value_error(message.str().c_str());
            }
            denseVoxelCount *= dimension;
        }
    }
}

struct WideCoordBBox {
    bool isEmpty = true;
    std::int64_t minimum[3] = {
        std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::max(),
    };
    std::int64_t maximum[3] = {
        std::numeric_limits<std::int64_t>::lowest(),
        std::numeric_limits<std::int64_t>::lowest(),
        std::numeric_limits<std::int64_t>::lowest(),
    };

    void include(const CoordBBox& bbox)
    {
        if (bbox.empty()) return;
        isEmpty = false;
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] = std::min(minimum[axis], static_cast<std::int64_t>(bbox.min()[axis]));
            maximum[axis] = std::max(maximum[axis], static_cast<std::int64_t>(bbox.max()[axis]));
        }
    }

    void includeLeaf(const Coord& origin, std::int64_t dimension)
    {
        isEmpty = false;
        for (int axis = 0; axis < 3; ++axis) {
            const std::int64_t lower = origin[axis];
            const std::int64_t upper = lower + dimension - 1;
            minimum[axis] = std::min(minimum[axis], lower);
            maximum[axis] = std::max(maximum[axis], upper);
        }
    }
};

template<typename GridType>
WideCoordBBox
storedTreeBoundingBox(const GridType& grid)
{
    WideCoordBBox result;
    for (auto leaf = grid.tree().cbeginLeaf(); leaf; ++leaf) {
        result.includeLeaf(leaf->origin(), static_cast<std::int64_t>(leaf->dim()));
    }
    auto value = grid.cbeginValueAll();
    value.setMaxDepth(value.getLeafDepth() - 1);
    for (; value; ++value) {
        if (!value.isTileValue()) continue;
        if (!value.isValueOn() && *value == grid.background()) continue;
        CoordBBox bbox;
        value.getBoundingBox(bbox);
        result.include(bbox);
    }
    return result;
}

template<typename GridType>
void
validateGridDomain(
    const GridType& grid,
    std::int64_t margin,
    const char* argument,
    bool requireDenseBound = false)
{
    if (grid.memUsage() > MAX_GRID_MEMORY_BYTES) {
        std::ostringstream message;
        message << argument << " exceeds the supported in-memory grid budget";
        throw nb::value_error(message.str().c_str());
    }
    if (grid.activeVoxelCount() > MAX_ACTIVE_VOXELS) {
        std::ostringstream message;
        message << argument << " exceeds the supported active-voxel budget";
        throw nb::value_error(message.str().c_str());
    }

    const WideCoordBBox bbox = storedTreeBoundingBox(grid);
    if (bbox.isEmpty) return;
    const std::int64_t minCoord = std::numeric_limits<std::int32_t>::lowest();
    const std::int64_t maxCoord = std::numeric_limits<std::int32_t>::max();
    std::int64_t dimensions[3];
    for (int axis = 0; axis < 3; ++axis) {
        const std::int64_t minimum = bbox.minimum[axis];
        const std::int64_t maximum = bbox.maximum[axis];
        if (minimum < minCoord + margin || maximum > maxCoord - margin) {
            std::ostringstream message;
            message << argument << " is too close to the int32 index boundary";
            throw nb::value_error(message.str().c_str());
        }
        dimensions[axis] = maximum - minimum + 1 + 2 * margin;
    }
    validateDenseDimensions(dimensions, argument, requireDenseBound);
}

void
validateGradientGridInputBudget(const FloatGrid& grid)
{
    if (grid.memUsage() > MAX_GRID_MEMORY_BYTES / 3) {
        throw nb::value_error(
            "grid gradient domain exceeds the supported derived gradient-grid memory budget");
    }
}

void
validateFiniteTransform(const GridBase& grid, const char* argument)
{
    const math::Transform& transform = grid.transform();
    if (!transform.isLinear()) {
        std::ostringstream message;
        message << argument << " must use a linear transform";
        throw nb::value_error(message.str().c_str());
    }

    math::AffineMap::Ptr affine;
    try {
        affine = transform.baseMap()->getAffineMap();
    } catch (const Exception&) {
        std::ostringstream message;
        message << argument << " must use a finite, invertible transform";
        throw nb::value_error(message.str().c_str());
    }
    const double determinant = affine->determinant();
    if (!affine->getMat4().isFinite() || !std::isfinite(determinant) || determinant == 0.0) {
        std::ostringstream message;
        message << argument << " must use a finite, invertible transform";
        throw nb::value_error(message.str().c_str());
    }

    Vec3d voxelSize;
    try {
        voxelSize = transform.voxelSize();
    } catch (const Exception&) {
        std::ostringstream message;
        message << argument << " must use a nondegenerate transform";
        throw nb::value_error(message.str().c_str());
    }
    if (!voxelSize.isFinite() || voxelSize.x() <= 0.0 || voxelSize.y() <= 0.0 ||
        voxelSize.z() <= 0.0) {
        std::ostringstream message;
        message << argument << " must use a nondegenerate transform";
        throw nb::value_error(message.str().c_str());
    }

    const Vec3d basis[] = {
        Vec3d(0.0, 0.0, 0.0),
        Vec3d(1.0, 0.0, 0.0),
        Vec3d(0.0, 1.0, 0.0),
        Vec3d(0.0, 0.0, 1.0),
    };
    Vec3s floatWorldPoints[4];
    for (std::size_t index = 0; index < 4; ++index) {
        const Vec3d& indexPoint = basis[index];
        Vec3d worldPoint, roundTrip;
        try {
            worldPoint = transform.indexToWorld(indexPoint);
            roundTrip = transform.worldToIndex(worldPoint);
        } catch (const Exception&) {
            std::ostringstream message;
            message << argument << " must use a numerically stable transform";
            throw nb::value_error(message.str().c_str());
        }
        if (!worldPoint.isFinite() || !roundTrip.isFinite() ||
            !roundTrip.eq(indexPoint, 1.0e-9)) {
            std::ostringstream message;
            message << argument << " must use a numerically stable transform";
            throw nb::value_error(message.str().c_str());
        }
        floatWorldPoints[index] = Vec3s(worldPoint);
        if (!floatWorldPoints[index].isFinite()) {
            std::ostringstream message;
            message << argument << " must map to finite float32 world coordinates";
            throw nb::value_error(message.str().c_str());
        }
    }
    const Vec3s xAxis = floatWorldPoints[1] - floatWorldPoints[0];
    const Vec3s yAxis = floatWorldPoints[2] - floatWorldPoints[0];
    const Vec3s zAxis = floatWorldPoints[3] - floatWorldPoints[0];
    const float floatDeterminant = xAxis.dot(yAxis.cross(zAxis));
    if (!std::isfinite(floatDeterminant) || floatDeterminant == 0.0f) {
        std::ostringstream message;
        message << argument << " must preserve voxel resolution in float32 world coordinates";
        throw nb::value_error(message.str().c_str());
    }
}

void
validateUniformTransform(const GridBase& grid, const char* argument)
{
    validateFiniteTransform(grid, argument);
    if (!grid.transform().hasUniformScale()) {
        std::ostringstream message;
        message << argument << " must use a uniform-scale transform";
        throw nb::value_error(message.str().c_str());
    }
}

void
validateScalarFloatGrid(const FloatGrid& grid, const char* argument)
{
    validateFiniteTransform(grid, argument);
    validateGridDomain(grid, 0, argument);
    if (grid.getGridClass() == GRID_STAGGERED) {
        std::ostringstream message;
        message << argument << " must be a scalar FloatGrid, not a staggered grid";
        throw nb::value_error(message.str().c_str());
    }
    if (!std::isfinite(grid.background())) {
        std::ostringstream message;
        message << argument << " must have a finite background value";
        throw nb::value_error(message.str().c_str());
    }
}

void
validateFiniteStoredValues(const FloatGrid& grid, const char* argument)
{
    for (FloatGrid::ValueAllCIter value = grid.cbeginValueAll(); value; ++value) {
        if (!std::isfinite(*value)) {
            std::ostringstream message;
            message << argument << " must contain only finite stored values";
            throw nb::value_error(message.str().c_str());
        }
    }
}

void
validateLevelSet(const FloatGrid& grid, const char* argument)
{
    if (grid.getGridClass() != GRID_LEVEL_SET) {
        std::ostringstream message;
        message << argument << " must have gridClass == GRID_LEVEL_SET";
        throw nb::value_error(message.str().c_str());
    }
    if (grid.tree().hasActiveTiles()) {
        std::ostringstream message;
        message << argument << " must not contain active tiles";
        throw nb::value_error(message.str().c_str());
    }
    validateUniformTransform(grid, argument);
    validateGridDomain(grid, 0, argument);
    if (!std::isfinite(grid.background()) || grid.background() <= 0.0f) {
        std::ostringstream message;
        message << argument << " must have a finite, positive background value";
        throw nb::value_error(message.str().c_str());
    }
    const double backgroundVoxels =
        static_cast<double>(grid.background()) / grid.voxelSize().x();
    if (!std::isfinite(backgroundVoxels) || backgroundVoxels > MAX_BAND_WIDTH) {
        std::ostringstream message;
        message << argument << " background exceeds the supported narrow-band width";
        throw nb::value_error(message.str().c_str());
    }
    validateFiniteStoredValues(grid, argument);
    const double valueLimit = static_cast<double>(grid.background()) * (1.0 + 1.0e-5);
    for (FloatGrid::ValueAllCIter value = grid.cbeginValueAll(); value; ++value) {
        if (std::abs(*value) > valueLimit) {
            std::ostringstream message;
            message << argument << " must keep stored values inside its narrow-band background";
            throw nb::value_error(message.str().c_str());
        }
    }
}

void
validateProducedFloatGrid(const FloatGrid& grid, const char* argument)
{
    if (grid.getGridClass() == GRID_LEVEL_SET) {
        validateLevelSet(grid, argument);
    } else {
        validateScalarFloatGrid(grid, argument);
        validateFiniteStoredValues(grid, argument);
    }
}

void
validateFilterControls(int width, int iterations)
{
    if (width < 1 || width > MAX_FILTER_WIDTH) {
        std::ostringstream message;
        message << "width must be between 1 and " << MAX_FILTER_WIDTH;
        throw nb::value_error(message.str().c_str());
    }
    if (iterations < 1 || iterations > MAX_FILTER_ITERATIONS) {
        std::ostringstream message;
        message << "iterations must be between 1 and " << MAX_FILTER_ITERATIONS;
        throw nb::value_error(message.str().c_str());
    }
}

void
validateFilterWork(const FloatGrid& grid, int width, int iterations)
{
    const std::uint64_t kernelWidth = static_cast<std::uint64_t>(2 * width + 1);
    const std::uint64_t passCount = 3 * static_cast<std::uint64_t>(iterations);
    const std::uint64_t workPerVoxel = kernelWidth * passCount;
    const std::uint64_t activeVoxels = grid.activeVoxelCount();
    if (activeVoxels > 0 && workPerVoxel > MAX_FILTER_WORK / activeVoxels) {
        throw nb::value_error("filter request exceeds the supported aggregate work budget");
    }
}

void
validateBandWidth(float width, const char* argument)
{
    if (!std::isfinite(width) || width <= 0.0f || width > MAX_BAND_WIDTH) {
        std::ostringstream message;
        message << argument << " must be finite and between 0 and " << MAX_BAND_WIDTH;
        throw nb::value_error(message.str().c_str());
    }
}

void
validateFloatBandProduct(const Vec3d& voxelSize, double width, const char* argument)
{
    const double maximumVoxelSize = std::max({voxelSize.x(), voxelSize.y(), voxelSize.z()});
    const double background = maximumVoxelSize * width;
    if (!std::isfinite(background) || background > std::numeric_limits<float>::max()) {
        std::ostringstream message;
        message << argument << " would exceed the finite float32 background range";
        throw nb::value_error(message.str().c_str());
    }
}

void
validateTopologyStep(int value, const char* argument)
{
    if (value < 0 || value > MAX_TOPOLOGY_STEPS) {
        std::ostringstream message;
        message << argument << " must be between 0 and " << MAX_TOPOLOGY_STEPS;
        throw nb::value_error(message.str().c_str());
    }
}

Interpolation
parseInterpolation(const std::string& interpolation)
{
    if (interpolation == "nearest") return Interpolation::Nearest;
    if (interpolation == "linear") return Interpolation::Linear;
    if (interpolation == "quadratic") return Interpolation::Quadratic;
    throw nb::value_error("interpolation must be one of: nearest, linear, quadratic");
}

void
validateCsgInputs(
    const FloatGrid& a,
    const FloatGrid& b,
    int threadCount,
    std::uint64_t maxActiveVoxels)
{
    validateThreadCount(threadCount);
    validateLevelSet(a, "a");
    validateLevelSet(b, "b");
    if (a.transform() != b.transform()) {
        throw nb::value_error("a and b must use identical transforms");
    }
    if (a.background() != b.background()) {
        throw nb::value_error("a and b must use matching level-set background widths");
    }
    if (a.activeVoxelCount() > maxActiveVoxels ||
        b.activeVoxelCount() > maxActiveVoxels - a.activeVoxelCount()) {
        throw nb::value_error("combined CSG inputs exceed max_active_voxels");
    }
}

template<typename Operation>
FloatGrid::Ptr
runCsgCopy(
    const FloatGrid& a,
    const FloatGrid& b,
    int threadCount,
    std::uint64_t maxActiveVoxels,
    Operation&& operation)
{
    validateCsgInputs(a, b, threadCount, maxActiveVoxels);
    FloatGrid::Ptr aSnapshot = a.deepCopy();
    FloatGrid::Ptr bSnapshot = b.deepCopy();
    {
        nb::gil_scoped_release release;
        tbb::global_control concurrency(
            tbb::global_control::max_allowed_parallelism, threadCount);
        tbb::task_arena arena(threadCount);
        arena.execute([&]() { operation(*aSnapshot, *bSnapshot); });
    }
    validateLevelSet(*aSnapshot, "CSG result");
    return aSnapshot;
}

FloatGrid::Ptr
csgUnion(
    const FloatGrid& a,
    const FloatGrid& b,
    StrictInt threadCount,
    StrictInt maxActiveVoxels)
{
    const std::uint64_t limit =
        validateResourceLimit(maxActiveVoxels, MAX_ACTIVE_VOXELS, "max_active_voxels");
    return runCsgCopy(a, b, threadCount, limit, [](FloatGrid& lhs, FloatGrid& rhs) {
        tools::csgUnion(lhs, rhs);
    });
}

FloatGrid::Ptr
csgIntersection(
    const FloatGrid& a,
    const FloatGrid& b,
    StrictInt threadCount,
    StrictInt maxActiveVoxels)
{
    const std::uint64_t limit =
        validateResourceLimit(maxActiveVoxels, MAX_ACTIVE_VOXELS, "max_active_voxels");
    return runCsgCopy(a, b, threadCount, limit, [](FloatGrid& lhs, FloatGrid& rhs) {
        tools::csgIntersection(lhs, rhs);
    });
}

FloatGrid::Ptr
csgDifference(
    const FloatGrid& a,
    const FloatGrid& b,
    StrictInt threadCount,
    StrictInt maxActiveVoxels)
{
    const std::uint64_t limit =
        validateResourceLimit(maxActiveVoxels, MAX_ACTIVE_VOXELS, "max_active_voxels");
    return runCsgCopy(a, b, threadCount, limit, [](FloatGrid& lhs, FloatGrid& rhs) {
        tools::csgDifference(lhs, rhs);
    });
}

FloatGrid::Ptr
levelSetMean(
    const FloatGrid& grid,
    StrictInt width,
    StrictInt iterations,
    StrictInt threadCount)
{
    validateThreadCount(threadCount);
    validateLevelSet(grid, "grid");
    validateFilterControls(width, iterations);
    validateFilterWork(grid, width, iterations);
    validateGridDomain(
        grid,
        static_cast<std::int64_t>(width) * iterations,
        "grid filter domain");

    FloatGrid::Ptr result = grid.deepCopy();
    {
        nb::gil_scoped_release release;
        tbb::global_control concurrency(
            tbb::global_control::max_allowed_parallelism, threadCount);
        tbb::task_arena arena(threadCount);
        arena.execute([&]() {
            tools::LevelSetFilter<FloatGrid> filter(*result);
            for (int iteration = 0; iteration < iterations; ++iteration) {
                filter.mean(width);
            }
        });
    }
    validateLevelSet(*result, "level-set mean result");
    return result;
}

FloatGrid::Ptr
levelSetOffset(const FloatGrid& grid, StrictFloat distance, StrictInt threadCount)
{
    validateThreadCount(threadCount);
    validateLevelSet(grid, "grid");
    if (!std::isfinite(distance.value)) throw nb::value_error("distance must be finite");
    const double distanceVoxels = std::abs(distance.value) / grid.voxelSize().x();
    if (!std::isfinite(distanceVoxels) || distanceVoxels > MAX_BAND_WIDTH) {
        throw nb::value_error("distance exceeds the supported offset in voxels");
    }
    validateGridDomain(
        grid,
        static_cast<std::int64_t>(std::ceil(distanceVoxels)) + 4,
        "grid offset domain",
        true);

    FloatGrid::Ptr result = grid.deepCopy();
    {
        nb::gil_scoped_release release;
        tbb::global_control concurrency(
            tbb::global_control::max_allowed_parallelism, threadCount);
        tbb::task_arena arena(threadCount);
        arena.execute([&]() {
            tools::LevelSetFilter<FloatGrid> filter(*result);
            filter.offset(distance.value);
        });
    }
    validateLevelSet(*result, "level-set offset result");
    return result;
}

FloatGrid::Ptr
scalarMean(
    const FloatGrid& grid,
    StrictInt width,
    StrictInt iterations,
    StrictInt threadCount)
{
    validateThreadCount(threadCount);
    validateScalarFloatGrid(grid, "grid");
    validateFiniteStoredValues(grid, "grid");
    if (grid.getGridClass() == GRID_LEVEL_SET) {
        throw nb::value_error(
            "scalar_mean does not accept GRID_LEVEL_SET; use level_set_mean instead");
    }
    validateFilterControls(width, iterations);
    validateFilterWork(grid, width, iterations);
    validateGridDomain(
        grid,
        static_cast<std::int64_t>(width) * iterations,
        "grid filter domain");

    FloatGrid::Ptr result = grid.deepCopy();
    {
        nb::gil_scoped_release release;
        tbb::global_control concurrency(
            tbb::global_control::max_allowed_parallelism, threadCount);
        tbb::task_arena arena(threadCount);
        arena.execute([&]() {
            tools::Filter<FloatGrid> filter(*result);
            filter.setProcessTiles(true);
            filter.mean(width, iterations);
        });
    }
    validateProducedFloatGrid(*result, "scalar mean result");
    return result;
}

FloatGrid::Ptr
levelSetNormalize(const FloatGrid& grid, StrictInt threadCount)
{
    validateThreadCount(threadCount);
    validateLevelSet(grid, "grid");
    validateGridDomain(grid, 4, "grid normalization domain");

    FloatGrid::Ptr result = grid.deepCopy();
    {
        nb::gil_scoped_release release;
        tbb::global_control concurrency(
            tbb::global_control::max_allowed_parallelism, threadCount);
        tbb::task_arena arena(threadCount);
        arena.execute([&]() {
            tools::LevelSetTracker<FloatGrid> tracker(*result);
            tracker.normalize();
        });
    }
    validateLevelSet(*result, "level-set normalization result");
    return result;
}

FloatGrid::Ptr
levelSetRebuild(
    const FloatGrid& grid,
    StrictFloat isovalue,
    StrictFloat exteriorWidth,
    StrictFloat interiorWidth,
    StrictInt threadCount)
{
    validateThreadCount(threadCount);
    validateScalarFloatGrid(grid, "grid");
    validateFiniteStoredValues(grid, "grid");
    validateUniformTransform(grid, "grid");
    if (!std::isfinite(isovalue.value)) throw nb::value_error("isovalue must be finite");
    validateBandWidth(exteriorWidth.value, "exterior_width");
    validateBandWidth(interiorWidth.value, "interior_width");
    validateFloatBandProduct(
        grid.voxelSize(),
        std::max(exteriorWidth.value, interiorWidth.value),
        "rebuild band");
    const std::int64_t expansion = static_cast<std::int64_t>(
        std::ceil(std::max(exteriorWidth.value, interiorWidth.value))) + 4;
    validateGridDomain(grid, expansion, "grid rebuild domain", true);

    FloatGrid::Ptr snapshot = grid.deepCopy();
    FloatGrid::Ptr result;
    {
        nb::gil_scoped_release release;
        tbb::global_control concurrency(
            tbb::global_control::max_allowed_parallelism, threadCount);
        tbb::task_arena arena(threadCount);
        result = arena.execute([&]() {
            return tools::levelSetRebuild(
                *snapshot, isovalue.value, exteriorWidth.value, interiorWidth.value);
        });
    }
    validateLevelSet(*result, "level-set rebuild result");
    return result;
}

template<typename Sampler>
FloatGrid::Ptr
resampleWith(const FloatGrid& grid, const math::Transform& referenceTransform)
{
    FloatGrid::Ptr result = grid.copyWithNewTree();
    result->setTransform(referenceTransform.copy());
    tools::resampleToMatch<Sampler>(grid, *result);
    return result;
}

void
validateResampleDomain(
    const FloatGrid& grid, const FloatGrid& referenceGrid, Interpolation interpolation)
{
    constexpr std::int64_t safetyMargin = 4;
    const std::int64_t samplerRadius = interpolation == Interpolation::Nearest ? 0 : 1;
    validateGridDomain(grid, safetyMargin, "grid resampling domain");
    const WideCoordBBox bbox = storedTreeBoundingBox(grid);
    if (bbox.isEmpty) return;

    const double minCoord =
        static_cast<double>(std::numeric_limits<std::int32_t>::lowest()) + safetyMargin;
    const double maxCoord =
        static_cast<double>(std::numeric_limits<std::int32_t>::max()) - safetyMargin;
    Vec3d minimum(std::numeric_limits<double>::infinity());
    Vec3d maximum(-std::numeric_limits<double>::infinity());
    for (int x = 0; x < 2; ++x) {
        for (int y = 0; y < 2; ++y) {
            for (int z = 0; z < 2; ++z) {
                const Vec3d sourceIndex(
                    x ? static_cast<double>(bbox.maximum[0]) + 1.0 : bbox.minimum[0],
                    y ? static_cast<double>(bbox.maximum[1]) + 1.0 : bbox.minimum[1],
                    z ? static_cast<double>(bbox.maximum[2]) + 1.0 : bbox.minimum[2]);
                const Vec3d worldPoint = grid.indexToWorld(sourceIndex);
                const Vec3d targetIndex = referenceGrid.worldToIndex(worldPoint);
                if (!worldPoint.isFinite() || !targetIndex.isFinite() ||
                    targetIndex.x() < minCoord || targetIndex.x() > maxCoord ||
                    targetIndex.y() < minCoord || targetIndex.y() > maxCoord ||
                    targetIndex.z() < minCoord || targetIndex.z() > maxCoord) {
                    throw nb::value_error(
                        "resampled grid maps outside the supported int32 index range");
                }
                minimum = math::minComponent(minimum, targetIndex);
                maximum = math::maxComponent(maximum, targetIndex);
            }
        }
    }
    std::int64_t dimensions[3];
    for (int axis = 0; axis < 3; ++axis) {
        const std::int64_t outputMinimum =
            static_cast<std::int64_t>(std::floor(minimum[axis])) - samplerRadius;
        const std::int64_t outputMaximum =
            static_cast<std::int64_t>(std::ceil(maximum[axis])) + samplerRadius;
        if (outputMinimum < std::numeric_limits<std::int32_t>::lowest() + safetyMargin ||
            outputMaximum > std::numeric_limits<std::int32_t>::max() - safetyMargin) {
            throw nb::value_error(
                "resampled grid maps outside the supported int32 index range");
        }
        dimensions[axis] = outputMaximum - outputMinimum + 1 + 2 * safetyMargin;
    }
    validateDenseDimensions(dimensions, "resampled grid domain", true);
}

FloatGrid::Ptr
resampleToMatch(
    const FloatGrid& grid,
    const FloatGrid& referenceGrid,
    const std::string& interpolationName,
    StrictInt threadCount)
{
    validateThreadCount(threadCount);
    validateScalarFloatGrid(grid, "grid");
    validateScalarFloatGrid(referenceGrid, "reference_grid");
    validateFiniteStoredValues(grid, "grid");
    const Interpolation interpolation = parseInterpolation(interpolationName);
    validateResampleDomain(grid, referenceGrid, interpolation);

    FloatGrid::Ptr gridSnapshot = grid.deepCopy();
    math::Transform::Ptr referenceTransform = referenceGrid.transform().copy();
    FloatGrid::Ptr result;
    {
        nb::gil_scoped_release release;
        tbb::global_control concurrency(
            tbb::global_control::max_allowed_parallelism, threadCount);
        tbb::task_arena arena(threadCount);
        result = arena.execute([&]() {
            switch (interpolation) {
            case Interpolation::Nearest:
                return resampleWith<tools::PointSampler>(*gridSnapshot, *referenceTransform);
            case Interpolation::Linear:
                return resampleWith<tools::BoxSampler>(*gridSnapshot, *referenceTransform);
            case Interpolation::Quadratic:
                return resampleWith<tools::QuadraticSampler>(*gridSnapshot, *referenceTransform);
            }
            throw nb::value_error("unsupported interpolation");
        });
    }
    validateProducedFloatGrid(*result, "resampling result");
    return result;
}

struct TriangleMesh {
    std::vector<Vec3s> points;
    std::vector<Vec3I> triangles;
};

template<typename Scalar, typename Array>
void
validateAlignedArray(const Array& array, const char* argument)
{
    const auto address = reinterpret_cast<std::uintptr_t>(array.data());
    if (address % alignof(Scalar) != 0) {
        std::ostringstream message;
        message << argument << " must use an aligned data buffer";
        throw nb::value_error(message.str().c_str());
    }
}

TriangleMesh
copyTriangleMesh(
    PointArray pointsArray,
    TriangleArray trianglesArray,
    std::uint64_t maxVertices,
    std::uint64_t maxFaces)
{
    if (pointsArray.shape(0) == 0) throw nb::value_error("points must not be empty");
    if (trianglesArray.shape(0) == 0) throw nb::value_error("triangles must not be empty");
    if (pointsArray.shape(0) > maxVertices) {
        throw nb::value_error("points exceed max_vertices");
    }
    if (trianglesArray.shape(0) > maxFaces) {
        throw nb::value_error("triangles exceed max_faces");
    }
    validateAlignedArray<float>(pointsArray, "points");
    validateAlignedArray<std::int32_t>(trianglesArray, "triangles");

    TriangleMesh mesh;
    mesh.points.resize(pointsArray.shape(0));
    for (std::size_t point = 0; point < mesh.points.size(); ++point) {
        const float x = pointsArray(point, 0);
        const float y = pointsArray(point, 1);
        const float z = pointsArray(point, 2);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            throw nb::value_error("points must contain only finite values");
        }
        mesh.points[point] = Vec3s(x, y, z);
    }

    mesh.triangles.resize(trianglesArray.shape(0));
    const auto pointCount = static_cast<std::int64_t>(mesh.points.size());
    for (std::size_t face = 0; face < mesh.triangles.size(); ++face) {
        const std::int32_t i = trianglesArray(face, 0);
        const std::int32_t j = trianglesArray(face, 1);
        const std::int32_t k = trianglesArray(face, 2);
        if (i < 0 || j < 0 || k < 0 || i >= pointCount || j >= pointCount || k >= pointCount) {
            throw nb::value_error("triangles contain an out-of-range vertex index");
        }
        if (i == j || j == k || k == i) {
            throw nb::value_error("triangles must not repeat a vertex index");
        }
        mesh.triangles[face] = Vec3I(i, j, k);
    }
    return mesh;
}

void
validateMeshVoxelDomain(
    const TriangleMesh& mesh,
    double voxelSize,
    float halfWidth,
    std::uint64_t maxActiveVoxels)
{
    const std::int64_t margin = static_cast<std::int64_t>(std::ceil(halfWidth)) + 4;
    const double minCoord =
        static_cast<double>(std::numeric_limits<std::int32_t>::lowest()) + margin;
    const double maxCoord =
        static_cast<double>(std::numeric_limits<std::int32_t>::max()) - margin;
    Vec3d minimum(std::numeric_limits<double>::infinity());
    Vec3d maximum(-std::numeric_limits<double>::infinity());
    for (const Vec3s& point : mesh.points) {
        const Vec3d indexPoint = Vec3d(point) / voxelSize;
        if (!indexPoint.isFinite() || indexPoint.x() < minCoord || indexPoint.x() > maxCoord ||
            indexPoint.y() < minCoord || indexPoint.y() > maxCoord ||
            indexPoint.z() < minCoord || indexPoint.z() > maxCoord) {
            throw nb::value_error("mesh points map outside the supported int32 index range");
        }
        minimum = math::minComponent(minimum, indexPoint);
        maximum = math::maxComponent(maximum, indexPoint);
    }
    std::int64_t dimensions[3];
    for (int axis = 0; axis < 3; ++axis) {
        dimensions[axis] = static_cast<std::int64_t>(std::ceil(maximum[axis])) -
            static_cast<std::int64_t>(std::floor(minimum[axis])) + 1 + 2 * margin;
    }
    validateDenseDimensions(dimensions, "mesh voxel domain", true, maxActiveVoxels);
}

FloatGrid::Ptr
meshToLevelSet(
    PointArray pointsArray,
    TriangleArray trianglesArray,
    StrictDouble voxelSize,
    StrictFloat halfWidth,
    StrictInt threadCount,
    StrictInt maxVertices,
    StrictInt maxFaces,
    StrictInt maxActiveVoxels)
{
    validateThreadCount(threadCount);
    const std::uint64_t vertexLimit =
        validateResourceLimit(maxVertices, MAX_INPUT_POINTS, "max_vertices");
    const std::uint64_t faceLimit =
        validateResourceLimit(maxFaces, MAX_INPUT_FACES, "max_faces");
    const std::uint64_t voxelLimit =
        validateResourceLimit(maxActiveVoxels, MAX_ACTIVE_VOXELS, "max_active_voxels");
    if (!std::isfinite(voxelSize.value) || voxelSize.value <= 0.0) {
        throw nb::value_error("voxel_size must be finite and greater than zero");
    }
    validateBandWidth(halfWidth.value, "half_width");
    validateFloatBandProduct(
        Vec3d(voxelSize.value), halfWidth.value, "mesh level-set band");
    TriangleMesh mesh = copyTriangleMesh(pointsArray, trianglesArray, vertexLimit, faceLimit);
    validateMeshVoxelDomain(mesh, voxelSize.value, halfWidth.value, voxelLimit);
    math::Transform::Ptr transform;
    try {
        transform = math::Transform::createLinearTransform(voxelSize.value);
    } catch (const Exception&) {
        throw nb::value_error("voxel_size does not define a stable OpenVDB transform");
    }

    FloatGrid::Ptr result;
    {
        nb::gil_scoped_release release;
        tbb::global_control concurrency(
            tbb::global_control::max_allowed_parallelism, threadCount);
        tbb::task_arena arena(threadCount);
        result = arena.execute([&]() {
            return tools::meshToLevelSet<FloatGrid>(
                *transform, mesh.points, mesh.triangles, halfWidth.value);
        });
    }
    validateLevelSet(*result, "mesh level-set result");
    return result;
}

FloatGrid::Ptr
meshToUnsignedDistanceField(
    PointArray pointsArray,
    TriangleArray trianglesArray,
    StrictDouble voxelSize,
    StrictFloat halfWidth,
    StrictInt threadCount,
    StrictInt maxVertices,
    StrictInt maxFaces,
    StrictInt maxActiveVoxels)
{
    validateThreadCount(threadCount);
    const std::uint64_t vertexLimit =
        validateResourceLimit(maxVertices, MAX_INPUT_POINTS, "max_vertices");
    const std::uint64_t faceLimit =
        validateResourceLimit(maxFaces, MAX_INPUT_FACES, "max_faces");
    const std::uint64_t voxelLimit =
        validateResourceLimit(maxActiveVoxels, MAX_ACTIVE_VOXELS, "max_active_voxels");
    if (!std::isfinite(voxelSize.value) || voxelSize.value <= 0.0) {
        throw nb::value_error("voxel_size must be finite and greater than zero");
    }
    validateBandWidth(halfWidth.value, "half_width");
    validateFloatBandProduct(
        Vec3d(voxelSize.value), halfWidth.value, "mesh unsigned-distance band");
    TriangleMesh mesh = copyTriangleMesh(pointsArray, trianglesArray, vertexLimit, faceLimit);
    validateMeshVoxelDomain(mesh, voxelSize.value, halfWidth.value, voxelLimit);
    math::Transform::Ptr transform;
    try {
        transform = math::Transform::createLinearTransform(voxelSize.value);
    } catch (const Exception&) {
        throw nb::value_error("voxel_size does not define a stable OpenVDB transform");
    }

    FloatGrid::Ptr result;
    {
        nb::gil_scoped_release release;
        tbb::global_control concurrency(
            tbb::global_control::max_allowed_parallelism, threadCount);
        tbb::task_arena arena(threadCount);
        result = arena.execute([&]() {
            const std::vector<Vec4I> quads;
            return tools::meshToUnsignedDistanceField<FloatGrid>(
                *transform, mesh.points, mesh.triangles, quads, halfWidth.value);
        });
    }
    validateProducedFloatGrid(*result, "mesh unsigned-distance result");
    return result;
}

std::vector<Vec3d>
copyWorldPoints(PointArray pointsArray)
{
    if (pointsArray.shape(0) > MAX_INPUT_POINTS) {
        throw nb::value_error("points exceed the supported sample limit");
    }
    validateAlignedArray<float>(pointsArray, "points");
    std::vector<Vec3d> points(pointsArray.shape(0));
    for (std::size_t point = 0; point < points.size(); ++point) {
        const float x = pointsArray(point, 0);
        const float y = pointsArray(point, 1);
        const float z = pointsArray(point, 2);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            throw nb::value_error("points must contain only finite values");
        }
        points[point] = Vec3d(x, y, z);
    }
    return points;
}

void
validateWorldPoints(const FloatGrid& grid, const std::vector<Vec3d>& points)
{
    constexpr double margin = 4.0;
    constexpr double minIndex =
        static_cast<double>(std::numeric_limits<std::int32_t>::lowest()) + margin;
    constexpr double maxIndex =
        static_cast<double>(std::numeric_limits<std::int32_t>::max()) - margin;
    for (const Vec3d& worldPoint : points) {
        const Vec3d indexPoint = grid.worldToIndex(worldPoint);
        if (!indexPoint.isFinite() || indexPoint.x() < minIndex || indexPoint.x() > maxIndex ||
            indexPoint.y() < minIndex || indexPoint.y() > maxIndex ||
            indexPoint.z() < minIndex || indexPoint.z() > maxIndex) {
            throw nb::value_error("points map outside the supported int32 index range");
        }
    }
}

template<typename Sampler>
std::vector<float>
sampleValuesWith(const FloatGrid& grid, const std::vector<Vec3d>& points)
{
    std::vector<float> values(points.size());
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, points.size()), [&](const auto& range) {
        FloatGrid::ConstAccessor accessor = grid.getConstAccessor();
        tools::GridSampler<FloatGrid::ConstAccessor, Sampler> sampler(
            accessor, grid.transform());
        for (std::size_t index = range.begin(); index != range.end(); ++index) {
            values[index] = sampler.wsSample(points[index]);
        }
    });
    return values;
}

template<typename Sampler>
std::vector<Vec3s>
sampleGradientsWith(const FloatGrid& grid, const std::vector<Vec3d>& points)
{
    Vec3SGrid::Ptr gradientGrid = tools::gradient(grid, true);
    validateGridDomain(*gradientGrid, 0, "derived gradient grid");
    std::vector<Vec3s> gradients(points.size());
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, points.size()), [&](const auto& range) {
        Vec3SGrid::ConstAccessor accessor = gradientGrid->getConstAccessor();
        tools::GridSampler<Vec3SGrid::ConstAccessor, Sampler> sampler(
            accessor, gradientGrid->transform());
        for (std::size_t index = range.begin(); index != range.end(); ++index) {
            gradients[index] = sampler.wsSample(points[index]);
        }
    });
    return gradients;
}

nb::ndarray<nb::numpy, float>
ownedFloatArray(std::vector<float>&& data)
{
    auto valuesStorage = std::make_unique<std::vector<float>>(std::move(data));
    auto* values = valuesStorage.get();
    nb::capsule owner(values, [](void* pointer) noexcept {
        delete static_cast<std::vector<float>*>(pointer);
    });
    valuesStorage.release();
    return nb::ndarray<nb::numpy, float>(values->data(), {values->size()}, owner);
}

nb::ndarray<nb::numpy, float>
ownedVec3Array(std::vector<Vec3s>&& data)
{
    auto valuesStorage = std::make_unique<std::vector<Vec3s>>(std::move(data));
    auto* values = valuesStorage.get();
    nb::capsule owner(values, [](void* pointer) noexcept {
        delete static_cast<std::vector<Vec3s>*>(pointer);
    });
    valuesStorage.release();
    return nb::ndarray<nb::numpy, float>(
        values->data(), {values->size(), std::size_t(3)}, owner, {3, 1});
}

nb::ndarray<nb::numpy, float>
sampleValues(
    const FloatGrid& grid,
    PointArray pointsArray,
    const std::string& interpolationName,
    StrictInt threadCount)
{
    validateThreadCount(threadCount);
    validateScalarFloatGrid(grid, "grid");
    validateFiniteStoredValues(grid, "grid");
    const Interpolation interpolation = parseInterpolation(interpolationName);
    const std::vector<Vec3d> points = copyWorldPoints(pointsArray);
    FloatGrid::Ptr snapshot = grid.deepCopy();
    validateGridDomain(*snapshot, 1, "grid sampling domain");
    validateWorldPoints(*snapshot, points);
    std::vector<float> values;
    {
        nb::gil_scoped_release release;
        tbb::global_control concurrency(
            tbb::global_control::max_allowed_parallelism, threadCount);
        tbb::task_arena arena(threadCount);
        values = arena.execute([&]() {
            switch (interpolation) {
            case Interpolation::Nearest:
                return sampleValuesWith<tools::PointSampler>(*snapshot, points);
            case Interpolation::Linear:
                return sampleValuesWith<tools::BoxSampler>(*snapshot, points);
            case Interpolation::Quadratic:
                return sampleValuesWith<tools::QuadraticSampler>(*snapshot, points);
            }
            throw nb::value_error("unsupported interpolation");
        });
    }
    for (const float value : values) {
        if (!std::isfinite(value)) throw nb::value_error("sampled values must be finite");
    }
    return ownedFloatArray(std::move(values));
}

nb::ndarray<nb::numpy, float>
sampleGradients(
    const FloatGrid& grid,
    PointArray pointsArray,
    const std::string& interpolationName,
    StrictInt threadCount)
{
    validateThreadCount(threadCount);
    validateScalarFloatGrid(grid, "grid");
    validateFiniteStoredValues(grid, "grid");
    const Interpolation interpolation = parseInterpolation(interpolationName);
    const std::vector<Vec3d> points = copyWorldPoints(pointsArray);
    FloatGrid::Ptr snapshot = grid.deepCopy();
    validateGridDomain(*snapshot, 1, "grid gradient domain");
    validateGradientGridInputBudget(*snapshot);
    validateWorldPoints(*snapshot, points);
    std::vector<Vec3s> gradients;
    {
        nb::gil_scoped_release release;
        tbb::global_control concurrency(
            tbb::global_control::max_allowed_parallelism, threadCount);
        tbb::task_arena arena(threadCount);
        gradients = arena.execute([&]() {
            switch (interpolation) {
            case Interpolation::Nearest:
                return sampleGradientsWith<tools::PointSampler>(*snapshot, points);
            case Interpolation::Linear:
                return sampleGradientsWith<tools::BoxSampler>(*snapshot, points);
            case Interpolation::Quadratic:
                return sampleGradientsWith<tools::QuadraticSampler>(*snapshot, points);
            }
            throw nb::value_error("unsupported interpolation");
        });
    }
    for (const Vec3s& value : gradients) {
        if (!std::isfinite(value.x()) || !std::isfinite(value.y()) || !std::isfinite(value.z())) {
            throw nb::value_error("sampled gradients must be finite");
        }
    }
    return ownedVec3Array(std::move(gradients));
}

struct ActiveValueMaskOp {
    std::optional<double> minValue;
    std::optional<double> maxValue;

    void operator()(const FloatGrid::ValueOnCIter& iter, BoolGrid::Accessor& accessor) const
    {
        const float value = *iter;
        if (std::isnan(value)) return;
        if ((minValue && value < *minValue) || (maxValue && value > *maxValue)) return;
        if (iter.isVoxelValue()) {
            accessor.setValueOn(iter.getCoord(), true);
        } else {
            CoordBBox bbox;
            iter.getBoundingBox(bbox);
            accessor.getTree()->fill(bbox, true, true);
        }
    }
};

BoolGrid::Ptr
activeValueMask(
    const FloatGrid& grid,
    std::optional<StrictDouble> minValueArgument,
    std::optional<StrictDouble> maxValueArgument,
    StrictInt threadCount)
{
    validateThreadCount(threadCount);
    validateScalarFloatGrid(grid, "grid");
    const std::optional<double> minValue = minValueArgument
        ? std::optional<double>(minValueArgument->value)
        : std::nullopt;
    const std::optional<double> maxValue = maxValueArgument
        ? std::optional<double>(maxValueArgument->value)
        : std::nullopt;
    if (minValue && !std::isfinite(*minValue)) {
        throw nb::value_error("min_value must be finite when supplied");
    }
    if (maxValue && !std::isfinite(*maxValue)) {
        throw nb::value_error("max_value must be finite when supplied");
    }
    if (minValue && maxValue && *minValue > *maxValue) {
        throw nb::value_error("min_value must not exceed max_value");
    }

    FloatGrid::Ptr snapshot = grid.deepCopy();
    BoolGrid::Ptr result(new BoolGrid(static_cast<const GridBase&>(*snapshot)));
    result->setGridClass(GRID_UNKNOWN);
    ActiveValueMaskOp operation{minValue, maxValue};
    {
        nb::gil_scoped_release release;
        tbb::global_control concurrency(
            tbb::global_control::max_allowed_parallelism, threadCount);
        tbb::task_arena arena(threadCount);
        arena.execute([&]() {
            tools::transformValues(snapshot->cbeginValueOn(), *result, operation, true, true);
        });
    }
    validateGridDomain(*result, 0, "active-value mask result");
    return result;
}

template<typename GridType>
FloatGrid::Ptr
topologyToLevelSetImpl(
    const GridType& grid,
    int halfWidth,
    int closingSteps,
    int dilation,
    int smoothingSteps,
    int threadCount)
{
    validateThreadCount(threadCount);
    validateUniformTransform(grid, "grid");
    if (halfWidth < 1 || halfWidth > static_cast<int>(MAX_BAND_WIDTH)) {
        std::ostringstream message;
        message << "half_width must be between 1 and " << static_cast<int>(MAX_BAND_WIDTH);
        throw nb::value_error(message.str().c_str());
    }
    validateFloatBandProduct(grid.voxelSize(), halfWidth, "topology level-set band");
    validateTopologyStep(closingSteps, "closing_steps");
    validateTopologyStep(dilation, "dilation");
    validateTopologyStep(smoothingSteps, "smoothing_steps");
    const std::int64_t expansion = static_cast<std::int64_t>(halfWidth) + closingSteps +
        dilation + smoothingSteps + 4;
    validateGridDomain(grid, expansion, "grid topology", true);

    typename GridType::Ptr snapshot = grid.deepCopy();
    FloatGrid::Ptr result;
    {
        nb::gil_scoped_release release;
        tbb::global_control concurrency(
            tbb::global_control::max_allowed_parallelism, threadCount);
        tbb::task_arena arena(threadCount);
        result = arena.execute([&]() {
            return tools::topologyToLevelSet(
                *snapshot, halfWidth, closingSteps, dilation, smoothingSteps);
        });
    }
    validateLevelSet(*result, "topology level-set result");
    return result;
}

FloatGrid::Ptr
topologyToLevelSetFloat(
    const FloatGrid& grid,
    StrictInt halfWidth,
    StrictInt closingSteps,
    StrictInt dilation,
    StrictInt smoothingSteps,
    StrictInt threadCount)
{
    validateScalarFloatGrid(grid, "grid");
    return topologyToLevelSetImpl(
        grid, halfWidth, closingSteps, dilation, smoothingSteps, threadCount);
}

FloatGrid::Ptr
topologyToLevelSetBool(
    const BoolGrid& grid,
    StrictInt halfWidth,
    StrictInt closingSteps,
    StrictInt dilation,
    StrictInt smoothingSteps,
    StrictInt threadCount)
{
    return topologyToLevelSetImpl(
        grid, halfWidth, closingSteps, dilation, smoothingSteps, threadCount);
}

BoolGrid::Ptr
extractEnclosedRegion(const FloatGrid& grid, StrictInt threadCount)
{
    validateThreadCount(threadCount);
    validateLevelSet(grid, "grid");
    validateGridDomain(grid, 1, "grid enclosed-region domain", true);
    FloatGrid::Ptr snapshot = grid.deepCopy();
    BoolGrid::Ptr result;
    {
        nb::gil_scoped_release release;
        tbb::global_control concurrency(
            tbb::global_control::max_allowed_parallelism, threadCount);
        tbb::task_arena arena(threadCount);
        result = arena.execute([&]() { return tools::extractEnclosedRegion(*snapshot); });
    }
    validateGridDomain(*result, 0, "enclosed-region result");
    return result;
}

MeshData
extractMesh(
    const FloatGrid& grid,
    double isovalue,
    double adaptivity,
    bool relaxDisorientedTriangles,
    int threadCount,
    std::uint64_t maxVertices,
    std::uint64_t maxFaces,
    std::uint64_t maxActiveVoxels)
{
    validateThreadCount(threadCount);
    validateScalarFloatGrid(grid, "grid");
    validateFiniteStoredValues(grid, "grid");
    validateGridDomain(grid, 2, "grid meshing domain");
    if (grid.activeVoxelCount() > maxActiveVoxels ||
        grid.activeVoxelCount() > MAX_MESH_INPUT_ACTIVE_VOXELS) {
        throw nb::value_error("grid exceeds max_active_voxels for volume meshing");
    }
    if (!std::isfinite(isovalue)) throw nb::value_error("isovalue must be finite");
    if (!std::isfinite(adaptivity) || adaptivity < 0.0 || adaptivity > 1.0) {
        throw nb::value_error("adaptivity must be finite and between 0 and 1");
    }

    FloatGrid::Ptr snapshot = grid.deepCopy();
    nb::gil_scoped_release release;
    tbb::global_control concurrency(tbb::global_control::max_allowed_parallelism, threadCount);
    tbb::task_arena arena(threadCount);
    return arena.execute([&]() {
        MeshData mesh;
        tools::VolumeToMesh mesher(
            isovalue,
            adaptivity,
            relaxDisorientedTriangles,
            static_cast<std::size_t>(maxVertices),
            static_cast<std::size_t>(maxFaces));
        mesher(*snapshot);

        auto& points = std::get<0>(mesh);
        auto& triangles = std::get<1>(mesh);
        auto& quads = std::get<2>(mesh);
        points.resize(mesher.pointListSize());
        if (!points.empty()) {
            std::copy_n(mesher.pointList().get(), points.size(), points.begin());
        }

        std::size_t triangleCount = 0, quadCount = 0;
        for (std::size_t pool = 0; pool < mesher.polygonPoolListSize(); ++pool) {
            const tools::PolygonPool& polygons = mesher.polygonPoolList()[pool];
            triangleCount += polygons.numTriangles();
            quadCount += polygons.numQuads();
        }
        if (triangleCount > maxFaces || quadCount > maxFaces - triangleCount) {
            throw nb::value_error("VolumeToMesh polygon output exceeds max_faces");
        }
        triangles.resize(triangleCount);
        quads.resize(quadCount);
        std::size_t triangleIndex = 0, quadIndex = 0;
        for (std::size_t pool = 0; pool < mesher.polygonPoolListSize(); ++pool) {
            const tools::PolygonPool& polygons = mesher.polygonPoolList()[pool];
            for (std::size_t index = 0; index < polygons.numTriangles(); ++index) {
                triangles[triangleIndex++] = polygons.triangle(index);
            }
            for (std::size_t index = 0; index < polygons.numQuads(); ++index) {
                quads[quadIndex++] = polygons.quad(index);
            }
        }
        return mesh;
    });
}

void
validateExtractedMesh(
    const MeshData& mesh, std::uint64_t maxVertices, std::uint64_t maxFaces)
{
    const auto& points = std::get<0>(mesh);
    const auto& triangles = std::get<1>(mesh);
    const auto& quads = std::get<2>(mesh);
    if (points.size() > maxVertices) {
        throw nb::value_error("extracted mesh exceeds max_vertices");
    }
    if (triangles.size() > maxFaces || quads.size() > maxFaces - triangles.size()) {
        throw nb::value_error("extracted mesh exceeds max_faces");
    }
    for (const Vec3s& point : points) {
        if (!point.isFinite()) {
            throw nb::value_error("extracted mesh points must be finite float32 values");
        }
    }
    const auto pointCount = static_cast<std::int64_t>(points.size());
    for (const Vec3I& face : triangles) {
        const std::int32_t i = face.x(), j = face.y(), k = face.z();
        if (i < 0 || j < 0 || k < 0 || i >= pointCount || j >= pointCount ||
            k >= pointCount || i == j || j == k || k == i) {
            throw nb::value_error("extracted mesh contains an invalid triangle");
        }
    }
    for (const Vec4I& face : quads) {
        const std::int32_t i = face.x(), j = face.y(), k = face.z(), l = face.w();
        if (i < 0 || j < 0 || k < 0 || l < 0 || i >= pointCount || j >= pointCount ||
            k >= pointCount || l >= pointCount || i == j || i == k || i == l || j == k ||
            j == l || k == l) {
            throw nb::value_error("extracted mesh contains an invalid quad");
        }
    }
}

nb::tuple
volumeToMesh(
    const FloatGrid& grid,
    StrictDouble isovalue,
    StrictDouble adaptivity,
    bool relaxDisorientedTriangles,
    StrictInt threadCount,
    StrictInt maxVertices,
    StrictInt maxFaces,
    StrictInt maxActiveVoxels)
{
    const std::uint64_t vertexLimit =
        validateResourceLimit(maxVertices, MAX_INPUT_POINTS, "max_vertices");
    const std::uint64_t faceLimit =
        validateResourceLimit(maxFaces, MAX_INPUT_FACES, "max_faces");
    const std::uint64_t voxelLimit =
        validateResourceLimit(maxActiveVoxels, MAX_ACTIVE_VOXELS, "max_active_voxels");
    MeshData mesh = extractMesh(
        grid,
        isovalue.value,
        adaptivity.value,
        relaxDisorientedTriangles,
        threadCount,
        vertexLimit,
        faceLimit,
        voxelLimit);
    validateExtractedMesh(mesh, vertexLimit, faceLimit);

    auto pointsStorage = std::make_unique<std::vector<Vec3s>>(std::move(std::get<0>(mesh)));
    auto trianglesStorage =
        std::make_unique<std::vector<Vec3I>>(std::move(std::get<1>(mesh)));
    auto quadsStorage = std::make_unique<std::vector<Vec4I>>(std::move(std::get<2>(mesh)));
    auto* points = pointsStorage.get();
    auto* triangles = trianglesStorage.get();
    auto* quads = quadsStorage.get();

    nb::capsule pointsOwner(points, [](void* pointer) noexcept {
        delete static_cast<std::vector<Vec3s>*>(pointer);
    });
    pointsStorage.release();
    nb::capsule trianglesOwner(triangles, [](void* pointer) noexcept {
        delete static_cast<std::vector<Vec3I>*>(pointer);
    });
    trianglesStorage.release();
    nb::capsule quadsOwner(quads, [](void* pointer) noexcept {
        delete static_cast<std::vector<Vec4I>*>(pointer);
    });
    quadsStorage.release();

    nb::ndarray<nb::numpy, float> pointArray(
        points->data(), {points->size(), std::size_t(3)}, pointsOwner, {3, 1});
    nb::ndarray<nb::numpy, std::int32_t> triangleArray(
        triangles->data(), {triangles->size(), std::size_t(3)}, trianglesOwner, {3, 1});
    nb::ndarray<nb::numpy, std::int32_t> quadArray(
        quads->data(), {quads->size(), std::size_t(4)}, quadsOwner, {4, 1});

    return nb::make_tuple(pointArray, triangleArray, quadArray);
}

} // namespace

void
exportTools(nb::module_ module)
{
    nb::module_ toolsModule = module.def_submodule("tools", "Bounded OpenVDB geometry tools.");
    toolsModule.attr("API_VERSION") = 3;

    toolsModule.def(
        "csg_union",
        &csgUnion,
        nb::arg("a"),
        nb::arg("b"),
        nb::kw_only(),
        nb::arg("thread_count") = 1,
        nb::arg("max_active_voxels") = static_cast<int>(MAX_ACTIVE_VOXELS),
        "Return the level-set union of two aligned FloatGrids without modifying either input.");
    toolsModule.def(
        "csg_intersection",
        &csgIntersection,
        nb::arg("a"),
        nb::arg("b"),
        nb::kw_only(),
        nb::arg("thread_count") = 1,
        nb::arg("max_active_voxels") = static_cast<int>(MAX_ACTIVE_VOXELS),
        "Return the level-set intersection of two aligned FloatGrids without modifying either input.");
    toolsModule.def(
        "csg_difference",
        &csgDifference,
        nb::arg("a"),
        nb::arg("b"),
        nb::kw_only(),
        nb::arg("thread_count") = 1,
        nb::arg("max_active_voxels") = static_cast<int>(MAX_ACTIVE_VOXELS),
        "Return the level-set difference a minus b without modifying either input.");
    toolsModule.def(
        "level_set_mean",
        &levelSetMean,
        nb::arg("grid"),
        nb::kw_only(),
        nb::arg("width") = 1,
        nb::arg("iterations") = 1,
        nb::arg("thread_count") = 1,
        "Return a mean-filtered deep copy of a level-set FloatGrid.");
    toolsModule.def(
        "level_set_offset",
        &levelSetOffset,
        nb::arg("grid"),
        nb::arg("distance"),
        nb::kw_only(),
        nb::arg("thread_count") = 1,
        "Return an offset deep copy of a level-set FloatGrid. Distance is in world units.");
    toolsModule.def(
        "scalar_mean",
        &scalarMean,
        nb::arg("grid"),
        nb::kw_only(),
        nb::arg("width") = 1,
        nb::arg("iterations") = 1,
        nb::arg("thread_count") = 1,
        "Return a mean-filtered deep copy of a scalar FloatGrid.");
    toolsModule.def(
        "level_set_normalize",
        &levelSetNormalize,
        nb::arg("grid"),
        nb::kw_only(),
        nb::arg("thread_count") = 1,
        "Return a normalized deep copy of a level-set FloatGrid.");
    toolsModule.def(
        "level_set_rebuild",
        &levelSetRebuild,
        nb::arg("grid"),
        nb::kw_only(),
        nb::arg("isovalue") = 0.0f,
        nb::arg("exterior_width") = float(openvdb::LEVEL_SET_HALF_WIDTH),
        nb::arg("interior_width") = float(openvdb::LEVEL_SET_HALF_WIDTH),
        nb::arg("thread_count") = 1,
        "Rebuild a scalar FloatGrid as a narrow-band level set.");
    toolsModule.def(
        "resample_to_match",
        &resampleToMatch,
        nb::arg("grid"),
        nb::arg("reference_grid"),
        nb::kw_only(),
        nb::arg("interpolation") = "quadratic",
        nb::arg("thread_count") = 1,
        "Return a copy of a FloatGrid resampled into a reference FloatGrid's index space.");
    toolsModule.def(
        "mesh_to_level_set",
        &meshToLevelSet,
        nb::arg("points").noconvert(),
        nb::arg("triangles").noconvert(),
        nb::kw_only(),
        nb::arg("voxel_size"),
        nb::arg("half_width") = float(openvdb::LEVEL_SET_HALF_WIDTH),
        nb::arg("thread_count") = 1,
        nb::arg("max_vertices") = static_cast<int>(MAX_INPUT_POINTS),
        nb::arg("max_faces") = static_cast<int>(MAX_INPUT_FACES),
        nb::arg("max_active_voxels") = static_cast<int>(MAX_ACTIVE_VOXELS),
        "Convert float32 points and int32 triangles to a level-set FloatGrid.");
    toolsModule.def(
        "mesh_to_unsigned_distance_field",
        &meshToUnsignedDistanceField,
        nb::arg("points").noconvert(),
        nb::arg("triangles").noconvert(),
        nb::kw_only(),
        nb::arg("voxel_size"),
        nb::arg("half_width") = float(openvdb::LEVEL_SET_HALF_WIDTH),
        nb::arg("thread_count") = 1,
        nb::arg("max_vertices") = static_cast<int>(MAX_INPUT_POINTS),
        nb::arg("max_faces") = static_cast<int>(MAX_INPUT_FACES),
        nb::arg("max_active_voxels") = static_cast<int>(MAX_ACTIVE_VOXELS),
        "Convert float32 points and int32 triangles to an unsigned-distance FloatGrid.");
    toolsModule.def(
        "sample_values",
        &sampleValues,
        nb::arg("grid"),
        nb::arg("points").noconvert(),
        nb::kw_only(),
        nb::arg("interpolation") = "quadratic",
        nb::arg("thread_count") = 1,
        "Sample scalar FloatGrid values at world-space float32 points.");
    toolsModule.def(
        "sample_gradients",
        &sampleGradients,
        nb::arg("grid"),
        nb::arg("points").noconvert(),
        nb::kw_only(),
        nb::arg("interpolation") = "quadratic",
        nb::arg("thread_count") = 1,
        "Sample transform-aware FloatGrid gradients in world coordinates per world unit.");
    toolsModule.def(
        "active_value_mask",
        &activeValueMask,
        nb::arg("grid"),
        nb::kw_only(),
        nb::arg("min_value") = nb::none(),
        nb::arg("max_value") = nb::none(),
        nb::arg("thread_count") = 1,
        "Return a BoolGrid mask for active values inside optional inclusive bounds.");
    toolsModule.def(
        "topology_to_level_set",
        &topologyToLevelSetFloat,
        nb::arg("grid"),
        nb::kw_only(),
        nb::arg("half_width") = 3,
        nb::arg("closing_steps") = 0,
        nb::arg("dilation") = 0,
        nb::arg("smoothing_steps") = 0,
        nb::arg("thread_count") = 1,
        "Convert a FloatGrid's active topology to a level-set FloatGrid.");
    toolsModule.def(
        "topology_to_level_set",
        &topologyToLevelSetBool,
        nb::arg("grid"),
        nb::kw_only(),
        nb::arg("half_width") = 3,
        nb::arg("closing_steps") = 0,
        nb::arg("dilation") = 0,
        nb::arg("smoothing_steps") = 0,
        nb::arg("thread_count") = 1,
        "Convert a BoolGrid's active topology to a level-set FloatGrid.");
    toolsModule.def(
        "extract_enclosed_region",
        &extractEnclosedRegion,
        nb::arg("grid"),
        nb::kw_only(),
        nb::arg("thread_count") = 1,
        "Return a BoolGrid containing the interior and enclosed regions of a level set.");
    toolsModule.def(
        "volume_to_mesh",
        &volumeToMesh,
        nb::arg("grid"),
        nb::kw_only(),
        nb::arg("isovalue") = 0.0,
        nb::arg("adaptivity") = 0.0,
        nb::arg("relax_disoriented_triangles") = true,
        nb::arg("thread_count") = 1,
        nb::arg("max_vertices") = static_cast<int>(MAX_INPUT_POINTS),
        nb::arg("max_faces") = static_cast<int>(MAX_INPUT_FACES),
        nb::arg("max_active_voxels") = static_cast<int>(MAX_ACTIVE_VOXELS),
        "Convert a scalar FloatGrid to owned float32 points and int32 polygon arrays.");
}
