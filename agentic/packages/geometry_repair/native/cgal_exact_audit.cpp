// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Evaluation-only wrapper. The linked CGAL Polygon Mesh Processing package is
// GPL-3.0-or-later; do not ship this binary in production images.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "geometry-repair-cgal-exact-audit " << CGAL_VERSION_STR << '\n';
        return 0;
    }
    if (argc != 2) {
        std::cerr << "usage: geometry_repair_cgal_exact_audit INPUT\n";
        return 2;
    }
    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Point = Kernel::Point_3;
    std::vector<Point> points;
    std::vector<std::vector<std::size_t>> polygons;
    if (!CGAL::IO::read_polygon_soup(argv[1], points, polygons)) {
        std::cerr << "could not read polygon soup\n";
        return 2;
    }
    for (const auto& polygon : polygons) {
        if (polygon.size() != 3) {
            std::cerr << "polygon soup must contain triangles only\n";
            return 2;
        }
        if (polygon[0] >= points.size() || polygon[1] >= points.size() ||
            polygon[2] >= points.size() || polygon[0] == polygon[1] ||
            polygon[1] == polygon[2] || polygon[0] == polygon[2]) {
            std::cerr << "polygon soup contains an invalid triangle\n";
            return 2;
        }
    }
    std::vector<std::pair<std::size_t, std::size_t>> intersections;
    CGAL::Polygon_mesh_processing::triangle_soup_self_intersections(
        points,
        polygons,
        std::back_inserter(intersections));
    std::cout << "{\"cgal_version\":\"" << CGAL_VERSION_STR
              << "\",\"point_count\":" << points.size() << ",\"face_count\":"
              << polygons.size() << ",\"self_intersection_pair_count\":"
              << intersections.size() << "}\n";
    return 0;
}
