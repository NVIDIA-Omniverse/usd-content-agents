// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <pmp/algorithms/hole_filling.h>
#include <pmp/surface_mesh.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

using Json = nlohmann::json;
using Triangle = std::array<std::size_t, 3>;
using Point = std::array<double, 3>;

constexpr const char* kCapabilitiesSchema = "geometry-repair.hard-mesh-capabilities.v1";
constexpr const char* kProtocol = "geometry-repair.hard-mesh-json.v1";
constexpr const char* kRequestSchema = "geometry-repair.hard-mesh-request.v1";
constexpr const char* kReportSchema = "geometry-repair.hard-mesh-report.v1";
constexpr const char* kWorker = "pmp_patch";
constexpr const char* kImplementationVersion = "geometry-repair-pmp-adapter-1";
constexpr const char* kBuildId =
    "pmp-library:2a2ad502743724ba90e09af816364c84032f9015";
constexpr const char* kPmpCommit = "2a2ad502743724ba90e09af816364c84032f9015";
constexpr const char* kPmpTreeSha256 =
    "ec134774c578b2ddd97b620daaa73ee1839ffb78b9d3355050443d8bdb6e5508";

struct Arguments {
    bool capabilities = false;
    std::filesystem::path request;
    std::filesystem::path result;
};

struct SourceMesh {
    pmp::SurfaceMesh mesh;
    std::vector<Point> points;
    std::vector<Triangle> triangles;
    pmp::FaceProperty<int> source_face_ids;
};

struct FillOutput {
    std::vector<Point> points;
    std::vector<Triangle> triangles;
    std::vector<std::size_t> generated_face_ids;
    std::vector<Triangle> generated_triangles;
    double maximum_envelope_ratio = 0.0;
};

std::string sha256_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("could not open file for SHA-256: " + path.string());
    EVP_MD_CTX* raw_context = EVP_MD_CTX_new();
    if (!raw_context)
        throw std::runtime_error("EVP_MD_CTX_new failed");
    const auto free_context = [](EVP_MD_CTX* context) { EVP_MD_CTX_free(context); };
    std::unique_ptr<EVP_MD_CTX, decltype(free_context)> context(raw_context, free_context);
    if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("EVP_DigestInit_ex failed");
    std::array<char, 64 * 1024> buffer{};
    while (stream)
    {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = stream.gcount();
        if (count > 0 && EVP_DigestUpdate(context.get(), buffer.data(), count) != 1)
            throw std::runtime_error("EVP_DigestUpdate failed");
    }
    if (!stream.eof())
        throw std::runtime_error("failed while hashing file: " + path.string());
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1)
        throw std::runtime_error("EVP_DigestFinal_ex failed");
    std::ostringstream encoded;
    encoded << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index)
        encoded << std::setw(2) << static_cast<unsigned int>(digest[index]);
    return encoded.str();
}

std::string sha256_text(const std::string& content)
{
    EVP_MD_CTX* raw_context = EVP_MD_CTX_new();
    if (!raw_context)
        throw std::runtime_error("EVP_MD_CTX_new failed");
    const auto free_context = [](EVP_MD_CTX* context) { EVP_MD_CTX_free(context); };
    std::unique_ptr<EVP_MD_CTX, decltype(free_context)> context(raw_context, free_context);
    if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(context.get(), content.data(), content.size()) != 1)
        throw std::runtime_error("failed while hashing captured source bytes");
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1)
        throw std::runtime_error("EVP_DigestFinal_ex failed");
    std::ostringstream encoded;
    encoded << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index)
        encoded << std::setw(2) << static_cast<unsigned int>(digest[index]);
    return encoded.str();
}

std::filesystem::path executable_path(const char* argv0)
{
    std::error_code error;
    auto path = std::filesystem::canonical("/proc/self/exe", error);
    if (!error)
        return path;
    return std::filesystem::canonical(argv0);
}

std::string read_bounded_text(const std::filesystem::path& path, std::uintmax_t maximum_bytes)
{
    const auto size = std::filesystem::file_size(path);
    if (size == 0 || size > maximum_bytes)
        throw std::runtime_error("input document is empty or exceeds its byte limit");
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("could not open input document: " + path.string());
    auto content =
        std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    if (content.empty() || content.size() > maximum_bytes)
        throw std::runtime_error("input document changed outside its byte limit while reading");
    return content;
}

void write_atomic(const std::filesystem::path& path, const std::string& content)
{
    std::filesystem::create_directories(path.parent_path());
    auto template_path = path.parent_path() / (path.filename().string() + ".tmp.XXXXXX");
    auto template_text = template_path.string();
    std::vector<char> template_buffer(template_text.begin(), template_text.end());
    template_buffer.push_back('\0');

    const int descriptor = ::mkstemp(template_buffer.data());
    if (descriptor < 0)
        throw std::runtime_error("could not securely create temporary output");
    const std::filesystem::path temporary(template_buffer.data());
    std::FILE* stream = ::fdopen(descriptor, "wb");
    if (!stream)
    {
        ::close(descriptor);
        std::filesystem::remove(temporary);
        throw std::runtime_error("could not open securely created temporary output");
    }

    const auto written = std::fwrite(content.data(), 1, content.size(), stream);
    const bool flushed = std::fflush(stream) == 0;
    const bool synced = flushed && ::fsync(::fileno(stream)) == 0;
    const bool closed = std::fclose(stream) == 0;
    if (written != content.size() || !flushed || !synced || !closed)
    {
        std::filesystem::remove(temporary);
        throw std::runtime_error("failed while writing securely created temporary output");
    }

    try
    {
        std::filesystem::rename(temporary, path);
    }
    catch (...)
    {
        std::filesystem::remove(temporary);
        throw;
    }
}

Arguments parse_arguments(int argc, char** argv)
{
    Arguments arguments;
    for (int index = 1; index < argc; ++index)
    {
        const std::string token(argv[index]);
        if (token == "--capabilities-json")
        {
            arguments.capabilities = true;
        }
        else if (token == "--request" || token == "--result")
        {
            if (++index >= argc)
                throw std::runtime_error("missing value for " + token);
            if (token == "--request")
                arguments.request = argv[index];
            else
                arguments.result = argv[index];
        }
        else
        {
            throw std::runtime_error("unknown argument: " + token);
        }
    }
    if (!arguments.capabilities && (arguments.request.empty() || arguments.result.empty()))
        throw std::runtime_error("--request and --result are required");
    return arguments;
}

Json capability_document(const std::string& executable_sha256)
{
    return {
        {"schema_version", kCapabilitiesSchema},
        {"protocol_version", kProtocol},
        {"worker", kWorker},
        {"implementation_version", kImplementationVersion},
        {"build_id", kBuildId},
        {"operations", {"pmp_fill_classified_hole", "pmp_remesh_generated_patch"}},
        {"capabilities",
         {"changed_region_report",
          "deterministic_single_thread",
          "freeze_vertices",
          "generated_face_classification",
          "independently_approved_executable_digest",
          "pinned_backend_source",
          "protect_edges",
          "source_face_correspondence"}},
        {"deterministic", true},
        {"backend_source_commit", kPmpCommit},
        {"backend_source_tree_sha256", kPmpTreeSha256},
        {"executable_sha256", executable_sha256},
    };
}

Json base_report(
    const Json& request,
    const std::string& request_sha256,
    const std::string& executable_sha256)
{
    return {
        {"schema_version", kReportSchema},
        {"protocol_version", kProtocol},
        {"status", "failed"},
        {"request_sha256", request_sha256},
        {"worker", kWorker},
        {"operation", request.value("operation", "unknown")},
        {"implementation_version", kImplementationVersion},
        {"build_id", kBuildId},
        {"backend_source_commit", kPmpCommit},
        {"backend_source_tree_sha256", kPmpTreeSha256},
        {"executable_sha256", executable_sha256},
        {"source_sha256", request.value("source_sha256", std::string(64, '0'))},
        {"changed", false},
        {"changed_face_ids", Json::array()},
        {"generated_face_ids", Json::array()},
        {"invariants_satisfied", Json::array()},
        {"fragments", Json::array()},
        {"intersection_curves", Json::array()},
        {"fragment_selection_performed", false},
        {"deleted_fragment_ids", Json::array()},
        {"warnings", Json::array()},
        {"failures", Json::array()},
    };
}

void validate_request_identity(
    const Json& request,
    const std::filesystem::path& source,
    const std::string& captured_source_sha256)
{
    if (request.at("schema_version") != kRequestSchema ||
        request.at("protocol_version") != kProtocol || request.at("worker") != kWorker ||
        request.at("implementation_build_id") != kBuildId)
        throw std::runtime_error("request protocol or implementation identity mismatch");
    if (request.at("source_path").get<std::string>() != source.string())
        throw std::runtime_error("source path normalization mismatch");
    const auto output = std::filesystem::path(request.at("output_path").get<std::string>());
    if (std::filesystem::weakly_canonical(source) ==
        std::filesystem::weakly_canonical(output))
        throw std::runtime_error("output path must not overwrite the immutable source");
    if (captured_source_sha256 != request.at("source_sha256").get<std::string>())
        throw std::runtime_error("source SHA-256 does not match immutable request");
    if (request.at("deterministic_seed").get<std::int64_t>() < 0)
        throw std::runtime_error("deterministic seed must be non-negative");
}

std::size_t parse_positive_obj_index(const std::string& token, std::size_t vertex_count)
{
    if (token.find('/') != std::string::npos)
        throw std::runtime_error("neutral OBJ cannot contain texture or normal indices");
    std::size_t consumed = 0;
    const long long value = std::stoll(token, &consumed);
    if (consumed != token.size() || value <= 0 || static_cast<std::size_t>(value) > vertex_count)
        throw std::runtime_error("neutral OBJ face index is invalid");
    return static_cast<std::size_t>(value - 1);
}

SourceMesh read_source_mesh(const std::string& source_text)
{
    SourceMesh source;
    std::istringstream stream(source_text);
    std::string line;
    while (std::getline(stream, line))
    {
        std::istringstream values(line);
        std::string record;
        values >> record;
        if (record.empty() || record == "#")
            continue;
        if (record == "v")
        {
            Point point{};
            if (!(values >> point[0] >> point[1] >> point[2]) ||
                !std::isfinite(point[0]) || !std::isfinite(point[1]) ||
                !std::isfinite(point[2]))
                throw std::runtime_error("neutral OBJ contains an invalid vertex");
            std::string trailing;
            if (values >> trailing)
                throw std::runtime_error("neutral OBJ vertex has trailing fields");
            source.points.push_back(point);
        }
        else if (record == "f")
        {
            std::array<std::string, 3> tokens{};
            std::string trailing;
            if (!(values >> tokens[0] >> tokens[1] >> tokens[2]) || values >> trailing)
                throw std::runtime_error("neutral OBJ must contain triangles only");
            source.triangles.push_back(
                {parse_positive_obj_index(tokens[0], source.points.size()),
                 parse_positive_obj_index(tokens[1], source.points.size()),
                 parse_positive_obj_index(tokens[2], source.points.size())});
        }
        else
        {
            throw std::runtime_error("neutral OBJ contains unsupported record: " + record);
        }
    }
    if (source.points.empty() || source.triangles.empty())
        throw std::runtime_error("neutral OBJ contains no triangle mesh");
    std::vector<pmp::Vertex> vertices;
    vertices.reserve(source.points.size());
    for (const auto& point : source.points)
        vertices.push_back(source.mesh.add_vertex(pmp::Point(
            static_cast<pmp::Scalar>(point[0]), static_cast<pmp::Scalar>(point[1]),
            static_cast<pmp::Scalar>(point[2]))));
    source.source_face_ids = source.mesh.add_face_property<int>("f:source_id", -1);
    for (std::size_t index = 0; index < source.triangles.size(); ++index)
    {
        const auto& triangle = source.triangles[index];
        if (triangle[0] == triangle[1] || triangle[1] == triangle[2] ||
            triangle[0] == triangle[2])
            throw std::runtime_error("neutral OBJ contains a repeated triangle vertex");
        const auto& first = source.points[triangle[0]];
        const auto& second = source.points[triangle[1]];
        const auto& third = source.points[triangle[2]];
        const Point first_edge = {
            second[0] - first[0], second[1] - first[1], second[2] - first[2]};
        const Point second_edge = {
            third[0] - first[0], third[1] - first[1], third[2] - first[2]};
        const Point normal = {
            first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
            first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
            first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]};
        const double squared_area =
            normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        if (!std::isfinite(squared_area) || squared_area <= 0.0)
            throw std::runtime_error("neutral OBJ contains a degenerate triangle");
        auto face = source.mesh.add_triangle(
            vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]);
        source.source_face_ids[face] = static_cast<int>(index);
    }
    return source;
}

Point subtract(const Point& left, const Point& right)
{
    return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

Point cross(const Point& left, const Point& right)
{
    return {left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0]};
}

double dot(const Point& left, const Point& right)
{
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

double length(const Point& value)
{
    return std::sqrt(dot(value, value));
}

Point add(const Point& left, const Point& right)
{
    return {left[0] + right[0], left[1] + right[1], left[2] + right[2]};
}

Point scale(const Point& value, double factor)
{
    return {value[0] * factor, value[1] * factor, value[2] * factor};
}

double bbox_diagonal(const std::vector<Point>& points)
{
    Point minimum = points.front();
    Point maximum = points.front();
    for (const auto& point : points)
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], point[axis]);
            maximum[axis] = std::max(maximum[axis], point[axis]);
        }
    return length(subtract(maximum, minimum));
}

std::vector<std::size_t> unique_indices(const Json& values, std::size_t maximum, const char* label)
{
    if (!values.is_array() || values.empty())
        throw std::runtime_error(std::string(label) + " must be a non-empty array");
    std::vector<std::size_t> result;
    std::set<std::size_t> unique;
    for (const auto& value : values)
    {
        const auto index = value.get<std::size_t>();
        if (index >= maximum || !unique.insert(index).second)
            throw std::runtime_error(std::string(label) + " contains an invalid or duplicate ID");
        result.push_back(index);
    }
    return result;
}

std::set<std::pair<std::size_t, std::size_t>> protected_edges(
    const Json& values,
    std::size_t maximum)
{
    if (!values.is_array() || values.empty())
        throw std::runtime_error("protected edges must be a non-empty array");
    std::set<std::pair<std::size_t, std::size_t>> edges;
    for (const auto& value : values)
    {
        if (!value.is_array() || value.size() != 2)
            throw std::runtime_error("protected edge must contain exactly two vertex IDs");
        auto first = value[0].get<std::size_t>();
        auto second = value[1].get<std::size_t>();
        if (first >= maximum || second >= maximum || first == second)
            throw std::runtime_error("protected edge contains an invalid vertex ID");
        if (first > second)
            std::swap(first, second);
        if (!edges.emplace(first, second).second)
            throw std::runtime_error("protected edges contain a duplicate pair");
    }
    return edges;
}

double bounded_number(
    const Json& parameters,
    const char* name,
    double minimum,
    double maximum,
    bool minimum_inclusive)
{
    const auto value = parameters.at(name).get<double>();
    const bool below = minimum_inclusive ? value < minimum : value <= minimum;
    if (!std::isfinite(value) || below || value > maximum)
        throw std::runtime_error(std::string(name) + " is outside the native policy bound");
    return value;
}

void validate_parameter_bounds(const Json& parameters)
{
    if (parameters.at("intent_evidence_id").get<std::string>().empty())
        throw std::runtime_error("intent_evidence_id must be non-empty");
    const auto& loop = parameters.at("boundary_loop_vertex_ids");
    const auto& frozen = parameters.at("frozen_boundary_vertex_ids");
    const auto& edges = parameters.at("protected_edge_vertex_pairs");
    if (!loop.is_array() || loop.size() < 5 || loop.size() > 4096 || !frozen.is_array() ||
        frozen.size() < loop.size() || frozen.size() > 4096 || !edges.is_array() ||
        edges.size() < loop.size() || edges.size() > 4096)
        throw std::runtime_error("boundary, frozen-vertex, or protected-edge count is out of bounds");
    bounded_number(parameters, "max_loop_perimeter_ratio", 0.0, 0.5, false);
    bounded_number(parameters, "max_patch_area_ratio", 0.0, 0.1, false);
    bounded_number(parameters, "max_nonplanarity_ratio", 0.0, 0.02, true);
    bounded_number(parameters, "max_boundary_turn_radians", 0.0, std::numbers::pi, false);
    bounded_number(parameters, "max_envelope_ratio", 0.0, 0.005, false);
    bounded_number(parameters, "timeout_s", 1.0, 300.0, true);
    const auto maximum_new_vertices = parameters.at("max_new_vertices").get<std::size_t>();
    if (maximum_new_vertices == 0 || maximum_new_vertices > 100000)
        throw std::runtime_error("max_new_vertices is outside the native policy bound");
    const auto seed = parameters.at("deterministic_seed").get<std::int64_t>();
    if (seed < 0 || seed > std::numeric_limits<std::int32_t>::max())
        throw std::runtime_error("deterministic_seed is outside the native policy bound");
}

bool cyclic_equal(const std::vector<std::size_t>& left, const std::vector<std::size_t>& right)
{
    if (left.size() != right.size())
        return false;
    for (std::size_t offset = 0; offset < right.size(); ++offset)
    {
        bool equal = true;
        for (std::size_t index = 0; index < left.size(); ++index)
            equal = equal && left[index] == right[(index + offset) % right.size()];
        if (equal)
            return true;
    }
    return false;
}

pmp::Halfedge validate_and_find_boundary(
    SourceMesh& source,
    const std::vector<std::size_t>& loop,
    const std::vector<std::size_t>& frozen,
    const std::set<std::pair<std::size_t, std::size_t>>& protected_pairs)
{
    if (loop.size() < 5)
        throw std::runtime_error("PMP production path requires a classified 5+ vertex hole");
    const std::set<std::size_t> frozen_set(frozen.begin(), frozen.end());
    for (const auto vertex : loop)
        if (!frozen_set.contains(vertex))
            throw std::runtime_error("every boundary vertex must be frozen");
    for (std::size_t index = 0; index < loop.size(); ++index)
    {
        auto first = loop[index];
        auto second = loop[(index + 1) % loop.size()];
        if (first > second)
            std::swap(first, second);
        if (!protected_pairs.contains({first, second}))
            throw std::runtime_error("every boundary edge must be protected");
    }
    for (const auto& [first, second] : protected_pairs)
        if (!source.mesh.find_edge(pmp::Vertex(first), pmp::Vertex(second)).is_valid())
            throw std::runtime_error("a protected edge does not exist in the source mesh");

    auto halfedge = source.mesh.find_halfedge(pmp::Vertex(loop[0]), pmp::Vertex(loop[1]));
    if (!halfedge.is_valid() || !source.mesh.is_boundary(halfedge))
        halfedge = source.mesh.find_halfedge(pmp::Vertex(loop[1]), pmp::Vertex(loop[0]));
    if (!halfedge.is_valid() || !source.mesh.is_boundary(halfedge))
        throw std::runtime_error("classified loop does not identify a PMP boundary halfedge");
    std::vector<std::size_t> component;
    auto current = halfedge;
    do
    {
        component.push_back(source.mesh.from_vertex(current).idx());
        current = source.mesh.next_halfedge(current);
        if (component.size() > loop.size())
            throw std::runtime_error("classified loop is not the complete boundary component");
    } while (current != halfedge);
    auto reversed_loop = loop;
    std::reverse(reversed_loop.begin(), reversed_loop.end());
    if (!cyclic_equal(component, loop) && !cyclic_equal(component, reversed_loop))
        throw std::runtime_error("classified IDs do not match the complete boundary component");
    return halfedge;
}

std::pair<Point, Point> validate_geometric_limits(
    const SourceMesh& source,
    const std::vector<std::size_t>& loop,
    const Json& parameters,
    double diagonal)
{
    if (!std::isfinite(diagonal) || diagonal <= 1e-12)
        throw std::runtime_error("source mesh has no finite repair scale");
    std::vector<Point> boundary;
    boundary.reserve(loop.size());
    for (const auto vertex : loop)
        boundary.push_back(source.points[vertex]);
    double perimeter = 0.0;
    Point area_vector{0.0, 0.0, 0.0};
    Point centroid{0.0, 0.0, 0.0};
    for (std::size_t index = 0; index < boundary.size(); ++index)
    {
        const auto& current = boundary[index];
        const auto& next = boundary[(index + 1) % boundary.size()];
        perimeter += length(subtract(next, current));
        area_vector = add(area_vector, cross(current, next));
        centroid = add(centroid, current);
    }
    centroid = scale(centroid, 1.0 / static_cast<double>(boundary.size()));
    const double area = 0.5 * length(area_vector);
    if (perimeter / diagonal > parameters.at("max_loop_perimeter_ratio").get<double>())
        throw std::runtime_error("classified loop exceeds max_loop_perimeter_ratio");
    if (area <= 1e-12 || area / (diagonal * diagonal) >
                            parameters.at("max_patch_area_ratio").get<double>())
        throw std::runtime_error("classified loop exceeds max_patch_area_ratio or has zero area");
    const double area_length = length(area_vector);
    if (area_length <= 1e-12)
        throw std::runtime_error("classified loop has no stable normal");
    const Point normal = scale(area_vector, 1.0 / area_length);
    double maximum_nonplanarity = 0.0;
    for (const auto& point : boundary)
        maximum_nonplanarity = std::max(
            maximum_nonplanarity, std::abs(dot(subtract(point, centroid), normal)) / diagonal);
    if (maximum_nonplanarity > parameters.at("max_nonplanarity_ratio").get<double>())
        throw std::runtime_error("classified loop exceeds max_nonplanarity_ratio");
    const double maximum_turn = parameters.at("max_boundary_turn_radians").get<double>();
    for (std::size_t index = 0; index < boundary.size(); ++index)
    {
        const auto incoming = subtract(boundary[(index + boundary.size() - 1) % boundary.size()],
                                       boundary[index]);
        const auto outgoing = subtract(boundary[(index + 1) % boundary.size()], boundary[index]);
        const double denominator = length(incoming) * length(outgoing);
        if (denominator <= diagonal * diagonal * 1e-24)
            throw std::runtime_error("classified loop contains a zero-length edge");
        const double cosine = std::clamp(dot(incoming, outgoing) / denominator, -1.0, 1.0);
        if (std::acos(cosine) > maximum_turn)
            throw std::runtime_error("classified loop exceeds max_boundary_turn_radians");
    }
    return {centroid, normal};
}

std::vector<std::size_t> face_vertices(const pmp::SurfaceMesh& mesh, pmp::Face face)
{
    std::vector<std::size_t> vertices;
    for (const auto vertex : mesh.vertices(face))
        vertices.push_back(vertex.idx());
    return vertices;
}

FillOutput run_fill(SourceMesh& source, const Json& parameters)
{
    validate_parameter_bounds(parameters);
    const auto loop = unique_indices(
        parameters.at("boundary_loop_vertex_ids"), source.points.size(), "boundary loop");
    const auto frozen = unique_indices(
        parameters.at("frozen_boundary_vertex_ids"), source.points.size(), "frozen vertices");
    const auto protected_pairs =
        protected_edges(parameters.at("protected_edge_vertex_pairs"), source.points.size());
    if (parameters.at("region_intent") != "classified_accidental_hole")
        throw std::runtime_error("PMP requires classified_accidental_hole intent");
    const auto maximum_new_vertices = parameters.at("max_new_vertices").get<std::size_t>();
    const auto diagonal = bbox_diagonal(source.points);
    const auto [centroid, normal] =
        validate_geometric_limits(source, loop, parameters, diagonal);
    const auto boundary = validate_and_find_boundary(source, loop, frozen, protected_pairs);

    std::vector<pmp::Point> original_points;
    original_points.reserve(source.points.size());
    for (std::size_t index = 0; index < source.points.size(); ++index)
        original_points.push_back(source.mesh.position(pmp::Vertex(index)));
    pmp::fill_hole(source.mesh, boundary);

    for (std::size_t index = 0; index < source.points.size(); ++index)
    {
        const auto point = source.mesh.position(pmp::Vertex(index));
        const auto& original = original_points[index];
        if (point != original)
            throw std::runtime_error("PMP moved a source vertex outside the generated patch");
    }
    for (const auto& [first, second] : protected_pairs)
        if (!source.mesh.find_edge(pmp::Vertex(first), pmp::Vertex(second)).is_valid())
            throw std::runtime_error("PMP removed a protected edge");

    std::vector<bool> found_source_faces(source.triangles.size(), false);
    std::vector<pmp::Face> generated_faces;
    for (const auto face : source.mesh.faces())
    {
        const int source_id = source.source_face_ids[face];
        if (source_id < 0)
        {
            generated_faces.push_back(face);
            continue;
        }
        const auto index = static_cast<std::size_t>(source_id);
        if (index >= source.triangles.size() || found_source_faces[index])
            throw std::runtime_error("PMP duplicated or corrupted a source-face identity");
        const auto vertices = face_vertices(source.mesh, face);
        const std::vector<std::size_t> expected(
            source.triangles[index].begin(), source.triangles[index].end());
        if (!cyclic_equal(vertices, expected))
            throw std::runtime_error("PMP changed a source face outside the generated patch");
        found_source_faces[index] = true;
    }
    if (std::find(found_source_faces.begin(), found_source_faces.end(), false) !=
        found_source_faces.end())
        throw std::runtime_error("PMP removed a source face outside the generated patch");
    if (generated_faces.empty())
        throw std::runtime_error("PMP produced no generated patch faces");

    FillOutput output;
    std::unordered_map<std::size_t, std::size_t> vertex_remap;
    for (const auto vertex : source.mesh.vertices())
    {
        const auto point = source.mesh.position(vertex);
        const auto output_index = output.points.size();
        if (vertex.idx() < source.points.size() && output_index != vertex.idx())
            throw std::runtime_error("PMP invalidated a source vertex handle");
        vertex_remap.emplace(vertex.idx(), output_index);
        output.points.push_back({point[0], point[1], point[2]});
    }
    if (output.points.size() - source.points.size() > maximum_new_vertices)
        throw std::runtime_error("PMP exceeded max_new_vertices");
    output.triangles = source.triangles;
    for (const auto face : generated_faces)
    {
        const auto vertices = face_vertices(source.mesh, face);
        if (vertices.size() < 3)
            throw std::runtime_error("PMP generated a degenerate patch face");
        for (std::size_t index = 1; index + 1 < vertices.size(); ++index)
        {
            const Triangle triangle = {
                vertex_remap.at(vertices[0]),
                vertex_remap.at(vertices[index]),
                vertex_remap.at(vertices[index + 1]),
            };
            output.generated_face_ids.push_back(output.triangles.size());
            output.generated_triangles.push_back(triangle);
            output.triangles.push_back(triangle);
        }
    }
    double patch_area = 0.0;
    double maximum_envelope = 0.0;
    for (const auto& triangle : output.generated_triangles)
    {
        const auto& first = output.points[triangle[0]];
        const auto& second = output.points[triangle[1]];
        const auto& third = output.points[triangle[2]];
        patch_area += 0.5 * length(cross(subtract(second, first), subtract(third, first)));
        for (const auto vertex : triangle)
            maximum_envelope = std::max(
                maximum_envelope,
                std::abs(dot(subtract(output.points[vertex], centroid), normal)) / diagonal);
    }
    if (patch_area / (diagonal * diagonal) >
        parameters.at("max_patch_area_ratio").get<double>())
        throw std::runtime_error("generated patch exceeds max_patch_area_ratio");
    output.maximum_envelope_ratio = maximum_envelope;
    if (maximum_envelope > parameters.at("max_envelope_ratio").get<double>())
        throw std::runtime_error("generated patch exceeds max_envelope_ratio");
    return output;
}

std::string obj_document(
    const std::vector<Point>& points,
    const std::vector<Triangle>& triangles,
    bool generated_region)
{
    std::ostringstream stream;
    stream << (generated_region ? "# generated PMP patch v1\n" : "# PMP candidate v1\n")
           << std::setprecision(10);
    for (const auto& point : points)
        stream << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
    for (const auto& triangle : triangles)
        stream << "f " << triangle[0] + 1 << ' ' << triangle[1] + 1 << ' '
               << triangle[2] + 1 << '\n';
    return stream.str();
}

std::string patch_obj_document(
    const std::vector<Point>& points,
    const std::vector<Triangle>& triangles)
{
    std::set<std::size_t> referenced;
    for (const auto& triangle : triangles)
        referenced.insert(triangle.begin(), triangle.end());
    std::unordered_map<std::size_t, std::size_t> remap;
    std::vector<Point> patch_points;
    for (const auto source_index : referenced)
    {
        remap.emplace(source_index, patch_points.size());
        patch_points.push_back(points.at(source_index));
    }
    std::vector<Triangle> patch_triangles;
    patch_triangles.reserve(triangles.size());
    for (const auto& triangle : triangles)
        patch_triangles.push_back(
            {remap.at(triangle[0]), remap.at(triangle[1]), remap.at(triangle[2])});
    return obj_document(patch_points, patch_triangles, true);
}

void write_success(
    Json& report,
    const Json& request,
    const FillOutput& output,
    const std::filesystem::path& result_path)
{
    const auto output_path = std::filesystem::path(request.at("output_path").get<std::string>());
    const auto evidence_root = result_path.parent_path();
    const auto changed_region = evidence_root / "pmp_changed_region.obj";
    const auto correspondence = evidence_root / "pmp_correspondence.json";
    const auto attribute_transfer = evidence_root / "pmp_attribute_transfer.json";
    write_atomic(output_path, obj_document(output.points, output.triangles, false));
    write_atomic(changed_region, patch_obj_document(output.points, output.generated_triangles));

    Json source_faces = Json::array();
    const auto source_face_count = output.triangles.size() - output.generated_face_ids.size();
    for (std::size_t index = 0; index < source_face_count; ++index)
        source_faces.push_back({{"output_face_id", index}, {"source_face_id", index}});
    const Json correspondence_document = {
        {"schema_version", "geometry-repair.pmp-correspondence.v1"},
        {"source_face_count", source_face_count},
        {"output_face_count", output.triangles.size()},
        {"source_faces", source_faces},
        {"generated_face_ids", output.generated_face_ids},
    };
    write_atomic(correspondence, correspondence_document.dump(2) + "\n");
    const Json transfer_document = {
        {"schema_version", "geometry-repair.pmp-attribute-transfer.v1"},
        {"policy", "preserve_source_faces_generated_patch_inherits_mesh_binding"},
        {"source_face_attributes_preserved", true},
        {"generated_face_attributes", "whole_mesh_binding_only"},
    };
    write_atomic(attribute_transfer, transfer_document.dump(2) + "\n");

    report["status"] = "success";
    report["output_path"] = output_path.string();
    report["output_sha256"] = sha256_file(output_path);
    report["changed"] = true;
    report["changed_face_ids"] = output.generated_face_ids;
    report["generated_face_ids"] = output.generated_face_ids;
    report["changed_region_path"] = changed_region.string();
    report["changed_region_sha256"] = sha256_file(changed_region);
    report["correspondence_path"] = correspondence.string();
    report["correspondence_sha256"] = sha256_file(correspondence);
    report["correspondence_coverage_ratio"] = 1.0;
    report["attribute_transfer_path"] = attribute_transfer.string();
    report["attribute_transfer_sha256"] = sha256_file(attribute_transfer);
    report["preserved_frozen_vertices"] = true;
    report["preserved_protected_edges"] = true;
    report["maximum_envelope_ratio"] = output.maximum_envelope_ratio;
    report["invariants_satisfied"] = {
        "classified_complete_boundary",
        "deterministic_single_thread",
        "generated_patch_only",
        "protected_edges",
        "source_faces_identity",
        "source_vertices_frozen",
    };
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto arguments = parse_arguments(argc, argv);
        const auto binary_sha256 = sha256_file(executable_path(argv[0]));
        if (arguments.capabilities)
        {
            std::cout << capability_document(binary_sha256).dump() << '\n';
            return 0;
        }
        const auto request_text = read_bounded_text(arguments.request, 1024 * 1024);
        const auto request = Json::parse(request_text);
        auto report = base_report(request, sha256_file(arguments.request), binary_sha256);
        try
        {
            const auto source = std::filesystem::path(request.at("source_path").get<std::string>());
            const auto source_text = read_bounded_text(source, 512ULL * 1024ULL * 1024ULL);
            const auto captured_source_sha256 = sha256_text(source_text);
            validate_request_identity(request, source, captured_source_sha256);
            const auto operation = request.at("operation").get<std::string>();
            if (operation == "pmp_remesh_generated_patch")
            {
                report["status"] = "refused";
                report["failures"] = {
                    "generated-patch remeshing is unavailable until frozen-region correspondence "
                    "is independently validated",
                };
            }
            else if (operation != "pmp_fill_classified_hole")
            {
                report["status"] = "refused";
                report["failures"] = {"unsupported PMP operation"};
            }
            else
            {
                auto source_mesh = read_source_mesh(source_text);
                const auto output = run_fill(source_mesh, request.at("parameters"));
                write_success(report, request, output, arguments.result);
            }
        }
        catch (const pmp::InvalidInputException& error)
        {
            report["status"] = "refused";
            report["failures"] = {std::string("PMP rejected the classified hole: ") + error.what()};
        }
        catch (const std::exception& error)
        {
            report["status"] = "failed";
            report["failures"] = {error.what()};
        }
        write_atomic(arguments.result, report.dump(2) + "\n");
        return report.at("status") == "failed" ? 1 : 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "geometry_repair_pmp_patch: " << error.what() << '\n';
        return 2;
    }
}
