cmake_minimum_required(VERSION 3.20)

project(geometry_repair_native LANGUAGES CXX)

option(
  GEOMETRY_REPAIR_BUILD_GPL_EVALUATORS
  "Build GPL evaluation-only binaries that are never shipped in production images"
  OFF
)
option(
  GEOMETRY_REPAIR_BUILD_PMP
  "Build the pinned PMP classified-hole adapter"
  OFF
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(
  GEOMETRY_REPAIR_NLOHMANN_JSON_INCLUDE_DIR
  ""
  CACHE PATH
  "Pinned nlohmann/json single_include directory"
)

if(GEOMETRY_REPAIR_BUILD_PMP)
  set(
    GEOMETRY_REPAIR_PMP_SOURCE_DIR
    ""
    CACHE PATH
    "Pinned pmp-library source checkout"
  )
  if(NOT EXISTS "${GEOMETRY_REPAIR_PMP_SOURCE_DIR}/src/pmp/surface_mesh.h")
    message(FATAL_ERROR "GEOMETRY_REPAIR_PMP_SOURCE_DIR is not a pmp-library checkout")
  endif()
  if(NOT EXISTS "${GEOMETRY_REPAIR_NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR "GEOMETRY_REPAIR_NLOHMANN_JSON_INCLUDE_DIR is not a json checkout")
  endif()

  set(PMP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(PMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(PMP_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(PMP_BUILD_VIEWERS OFF CACHE BOOL "" FORCE)
  set(PMP_BUILD_POLYGONAL OFF CACHE BOOL "" FORCE)
  set(PMP_BUILD_REGRESSIONS OFF CACHE BOOL "" FORCE)
  set(PMP_INSTALL OFF CACHE BOOL "" FORCE)
  set(PMP_STRICT_COMPILATION OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(CMAKE_DISABLE_FIND_PACKAGE_OpenMP TRUE)
  add_subdirectory(
    "${GEOMETRY_REPAIR_PMP_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/pmp-library"
    EXCLUDE_FROM_ALL
  )

  find_package(OpenSSL REQUIRED COMPONENTS Crypto)
  add_executable(geometry_repair_pmp_patch pmp_patch.cpp)
  target_include_directories(
    geometry_repair_pmp_patch
    PRIVATE "${GEOMETRY_REPAIR_NLOHMANN_JSON_INCLUDE_DIR}"
  )
  target_link_libraries(geometry_repair_pmp_patch PRIVATE pmp OpenSSL::Crypto)
  target_compile_features(geometry_repair_pmp_patch PRIVATE cxx_std_20)
  target_compile_options(
    geometry_repair_pmp_patch
    PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
  install(TARGETS geometry_repair_pmp_patch RUNTIME DESTINATION bin)
endif()

if(GEOMETRY_REPAIR_BUILD_GPL_EVALUATORS)
  find_package(CGAL REQUIRED)
  add_executable(geometry_repair_cgal_exact_audit cgal_exact_audit.cpp)
  target_link_libraries(geometry_repair_cgal_exact_audit PRIVATE CGAL::CGAL)
  target_compile_options(
    geometry_repair_cgal_exact_audit
    PRIVATE -Wall -Wextra -Wpedantic
  )
endif()
