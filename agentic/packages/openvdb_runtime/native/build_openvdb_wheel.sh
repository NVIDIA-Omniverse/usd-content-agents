#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="${SCRIPT_DIR}/source-lock.json"
LICENSE_POLICY_FILE="${SCRIPT_DIR}/license-policy.json"
LICENSE_VERIFIER="${SCRIPT_DIR}/verify_wheel_license_policy.py"
RELEASE_CANDIDATE_GENERATOR="${SCRIPT_DIR}/generate_release_candidate.py"
WHEEL_CANONICALIZER="${SCRIPT_DIR}/canonicalize_wheel.py"
WHEEL_CANONICALIZER_SHA256="e96d5fbccade606dfa0d5cfe18018c6f6ba4df8a9c16424b446e39ef01a06d67"
RELEASE_LOCK_FILE="${SCRIPT_DIR}/release-lock.json"
OVERLAY_DIR="${SCRIPT_DIR}/overlay"
PATCH_DIR="${SCRIPT_DIR}/patches"
NATIVE_PATCH_DIR="${SCRIPT_DIR}/native-patches"
LICENSE_DIR="${SCRIPT_DIR}/licenses"

BUILD_ROOT="${OPENVDB_RUNTIME_BUILD_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/world-understanding/openvdb13-wheel}"
CACHE_DIR="${OPENVDB_RUNTIME_SOURCE_CACHE:-${BUILD_ROOT}/cache}"
SOURCE_DIR="${BUILD_ROOT}/source"
NATIVE_SOURCE_ROOT="${BUILD_ROOT}/native-sources"
NATIVE_BUILD_ROOT="${BUILD_ROOT}/native-build"
NATIVE_PREFIX="${BUILD_ROOT}/native-prefix"
CMAKE_BUILD_DIR="${BUILD_ROOT}/cmake-build"
RAW_WHEEL_DIR="${BUILD_ROOT}/wheel-raw"
REPAIRED_WHEEL_DIR="${BUILD_ROOT}/wheel-repaired"
TOOL_ENV="${BUILD_ROOT}/build-tools"
SMOKE_ENV="${BUILD_ROOT}/smoke-env"
NANOBIND_WHEEL_DIR="${BUILD_ROOT}/nanobind-wheel"
NATIVE_PROVENANCE="${BUILD_ROOT}/native-dependency-provenance.json"
BOOTSTRAP_ROOT="${BUILD_ROOT}/locked-bootstrap"
BUILD_TOOL_WHEEL_DIR="${BUILD_ROOT}/build-tool-wheelhouse"
OUTPUT_DIR="${OPENVDB_RUNTIME_WHEEL_DIR:-${SCRIPT_DIR}/../dist-native}"
JOBS="${OPENVDB_RUNTIME_BUILD_JOBS:-2}"

MODE="release"
if [[ "${1:-}" == "--prepare-only" ]]; then
  MODE="prepare"
elif [[ "${1:-}" == "--candidate" ]]; then
  MODE="candidate"
elif [[ $# -ne 0 ]]; then
  printf 'usage: %s [--prepare-only|--candidate]\n' "$0" >&2
  exit 2
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'missing required command: %s\n' "$1" >&2
    exit 2
  fi
}

reject_ambient_toolchain_environment() {
  local variable
  local -a forbidden=(
    AR AS CC CPP CXX F77 F90 FC LD NM OBJCOPY OBJDUMP RANLIB READELF STRIP
    ARCHFLAGS ASMFLAGS CFLAGS CPPFLAGS CXXFLAGS FCFLAGS FFLAGS LDFLAGS
    CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH INCLUDE INCLUDE_PATH
    OBJC_INCLUDE_PATH
    LIBRARY_PATH LD_AUDIT LD_DEBUG LD_LIBRARY_PATH LD_PRELOAD LD_PROFILE
    LD_RUN_PATH
    COMPILER_PATH GCC_COMPARE_DEBUG GCC_EXEC_PREFIX GCC_SPECS
    COLLECT_GCC_OPTIONS COLLECT_LTO_WRAPPER CCC_OVERRIDE_OPTIONS
    DEPENDENCIES_OUTPUT SUNPRO_DEPENDENCIES
    CMAKE_ARGS CMAKE_BUILD_PARALLEL_LEVEL CMAKE_GENERATOR
    CMAKE_GENERATOR_INSTANCE CMAKE_GENERATOR_PLATFORM CMAKE_GENERATOR_TOOLSET
    CMAKE_INCLUDE_PATH CMAKE_LIBRARY_PATH CMAKE_PREFIX_PATH CMAKE_PROGRAM_PATH
    CMAKE_TOOLCHAIN_FILE CMAKE_PROJECT_INCLUDE CMAKE_PROJECT_TOP_LEVEL_INCLUDES
    CMAKE_FIND_ROOT_PATH
    SKBUILD_BUILD_DIR SKBUILD_CMAKE_ARGS
    PKG_CONFIG PKG_CONFIG_PATH PKG_CONFIG_LIBDIR PKG_CONFIG_SYSROOT_DIR
    PKG_CONFIG_SYSTEM_INCLUDE_PATH PKG_CONFIG_SYSTEM_LIBRARY_PATH
    MAKEFLAGS MFLAGS NINJAFLAGS DESTDIR
    CONDA_PREFIX CONDA_BUILD_SYSROOT VIRTUAL_ENV OPENVDB_RUNTIME_PYTHON
    PYTHONHOME PYTHONPATH PYTHONUSERBASE _PYTHON_SYSCONFIGDATA_NAME
    BASH_ENV ENV SOURCE_DATE_EPOCH ZERO_AR_DATE
  )

  for variable in "${forbidden[@]}"; do
    if [[ -v ${variable} ]]; then
      printf 'ambient build environment variable is forbidden: %s\n' \
        "${variable}" >&2
      exit 2
    fi
  done
  while IFS= read -r variable; do
    case "${variable}" in
      CCACHE_*|CMAKE_*|DISTCC_*|DYLD_*|ICECC_*|LD_*|NIX_*|PIP_*|PKG_CONFIG_*|SCCACHE_*|SKBUILD_*|UV_*|VCPKG_*)
        printf 'ambient build environment variable is forbidden: %s\n' \
          "${variable}" >&2
        exit 2
        ;;
    esac
  done < <(compgen -e)
}

reject_ambient_toolchain_environment
umask 022
export LANG=C
export LC_ALL=C
export TZ=UTC

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

if [[ "$(sha256_file "${WHEEL_CANONICALIZER}")" != "${WHEEL_CANONICALIZER_SHA256}" ]]; then
  printf 'native wheel canonicalizer digest mismatch\n' >&2
  exit 3
fi

download_locked_file() {
  local label="$1"
  local url="$2"
  local expected_sha256="$3"
  local destination="$4"

  if [[ ! -f "${destination}" ]]; then
    local temporary="${destination}.part.$$"
    if ! curl --fail --location --proto '=https' --tlsv1.2 \
      --output "${temporary}" "${url}"; then
      rm -f -- "${temporary}"
      exit 3
    fi
    if [[ "$(sha256_file "${temporary}")" != "${expected_sha256}" ]]; then
      rm -f -- "${temporary}"
      printf 'downloaded %s digest mismatch\n' "${label}" >&2
      exit 3
    fi
    mv -- "${temporary}" "${destination}"
  fi
  if [[ "$(sha256_file "${destination}")" != "${expected_sha256}" ]]; then
    printf '%s digest mismatch: %s\n' "${label}" "${destination}" >&2
    exit 3
  fi
}

bootstrap_locked_build_runtime() {
  local python_archive="${CACHE_DIR}/cpython-3.12.13-20260602-x86_64.tar.gz"
  local uv_wheel="${CACHE_DIR}/uv-0.11.19-x86_64.whl"
  local python_root="${BOOTSTRAP_ROOT}/python"
  local uv_root="${BOOTSTRAP_ROOT}/uv"

  download_locked_file \
    "CPython 3.12.13 build runtime" \
    "https://releases.astral.sh/github/python-build-standalone/releases/download/20260602/cpython-3.12.13%2B20260602-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz" \
    "191b5188b42886fb8a14968d714571e8f3d1cef92ac7ad4c7e24cc4d0929b194" \
    "${python_archive}"
  download_locked_file \
    "uv 0.11.19 build frontend" \
    "https://files.pythonhosted.org/packages/5a/74/2bd8b51e1d76210fd424ae55ec3f34ded5a10eeff3dd38aeb03c816a0af2/uv-0.11.19-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "731d9fab8db5d41590af64236d03f8069c8da665fd0f9493b85985f19c86cd90" \
    "${uv_wheel}"

  rm -rf -- "${BOOTSTRAP_ROOT}"
  mkdir -p -- "${python_root}" "${uv_root}"
  tar --extract --gzip --file "${python_archive}" --directory "${python_root}" \
    --strip-components 1 --no-same-owner
  unzip -q "${uv_wheel}" -d "${uv_root}"
  chmod -R go-w -- "${BOOTSTRAP_ROOT}"

  PYTHON="${python_root}/bin/python3.12"
  UV_BIN="${uv_root}/uv-0.11.19.data/scripts/uv"
  if [[ "$(sha256_file "${PYTHON}")" != \
      "d713474fd44339f8bd8150edc7b324d0be494dbf2998a1de04a7b0ecdbc1c2ed" ]] || \
    [[ "$(sha256_file "${python_root}/lib/libpython3.12.so.1.0")" != \
      "8083793fd24e3ca7d04c5e857bc2d4c0115513a4fa6c24b9fbcf496c2e78ef80" ]]; then
    printf 'locked CPython extraction identity mismatch\n' >&2
    exit 3
  fi
  if [[ "$(sha256_file "${UV_BIN}")" != \
      "a00d3a24514fc0403fc232c9c99bf5e542657c38f4ed941e0611731e4cff268b" ]] || \
    [[ "$("${UV_BIN}" --version)" != "uv 0.11.19 (x86_64-unknown-linux-gnu)" ]]; then
    printf 'locked uv extraction identity mismatch\n' >&2
    exit 3
  fi
  if [[ "$("${PYTHON}" -I -S -c 'import sys; print(sys.version.split()[0])')" != "3.12.13" ]]; then
    printf 'locked CPython version mismatch\n' >&2
    exit 3
  fi
}

verify_locked_source_tree() {
  local component_id="$1"
  local source_root="$2"
  "${PYTHON}" - "${LOCK_FILE}" "${SCRIPT_DIR}" "${component_id}" "${source_root}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

lock = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
native_dir = pathlib.Path(sys.argv[2])
component_id = sys.argv[3]
source_root = pathlib.Path(sys.argv[4])

sources = [lock["source"], *lock["native_dependencies"]]
sources.extend(
    lock["build_inputs"][name]
    for name in ("nanobind", "robin-map")
)
source = next((item for item in sources if item["id"] == component_id), None)
if source is None:
    raise SystemExit(f"unknown locked source id: {component_id}")

entries = []
for path in source_root.rglob("*"):
    if path.is_dir():
        continue
    relative = path.relative_to(source_root).as_posix().encode("utf-8")
    if component_id == "c-blosc" and relative == b"internal-complibs/zstd-1.5.6/LICENSE":
        # c-blosc omits zstd's root license file. The exact v1.5.6 license is
        # injected for provenance after extraction and excluded from its Git tree.
        continue
    if path.is_symlink():
        contents = os.readlink(path).encode("utf-8")
        mode = b"120000"
    else:
        contents = path.read_bytes()
        mode = b"100755" if path.stat().st_mode & 0o111 else b"100644"
    header = b"blob " + str(len(contents)).encode("ascii") + b"\0"
    blob_sha1 = hashlib.sha1(header + contents).hexdigest().encode("ascii")
    entries.append((relative, mode + b" blob " + blob_sha1 + b"\t" + relative + b"\0"))
inventory = hashlib.sha256(b"".join(entry for _, entry in sorted(entries))).hexdigest()
if inventory != source["git_tree_inventory_sha256"]:
    raise SystemExit(f"extracted {component_id} Git-tree inventory digest mismatch")

licensed = [source, *source.get("vendored_components", [])]
for item in licensed:
    for source_file in item.get("source_files", []):
        path = source_root / source_file["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_file["sha256"]:
            raise SystemExit(
                f"vendored source digest mismatch for {item['id']}: {source_file['path']}"
            )
    for license_file in item["license_files"]:
        if item is source and component_id == "openvdb":
            redistribution = source_root / license_file["redistribution_path"]
        else:
            redistribution = native_dir / license_file["redistribution_path"]
        if hashlib.sha256(redistribution.read_bytes()).hexdigest() != license_file["sha256"]:
            raise SystemExit(f"redistribution license digest mismatch for {item['id']}")
        source_path = license_file.get("source_path")
        if source_path is not None:
            path = source_root / source_path
            expected_source_sha256 = license_file.get("source_sha256", license_file["sha256"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected_source_sha256:
                raise SystemExit(f"source license digest mismatch for {item['id']}: {source_path}")
PY
}

prepare_locked_source() {
  local component_id="$1"
  local url="$2"
  local archive_sha256="$3"
  local destination="$4"
  local override_archive="${5:-}"
  local archive_path="${override_archive:-${CACHE_DIR}/${component_id}-${archive_sha256}.tar.gz}"

  if [[ -z "${override_archive}" ]]; then
    download_locked_file "${component_id} source archive" "${url}" \
      "${archive_sha256}" "${archive_path}"
  elif [[ ! -f "${archive_path}" ]] || \
    [[ "$(sha256_file "${archive_path}")" != "${archive_sha256}" ]]; then
    printf '%s source archive is missing or has the wrong digest: %s\n' \
      "${component_id}" "${archive_path}" >&2
    exit 3
  fi

  rm -rf -- "${destination}"
  mkdir -p -- "${destination}"
  tar --extract --gzip --file "${archive_path}" --directory "${destination}" \
    --strip-components 1 --no-same-owner
  if [[ "${component_id}" == "c-blosc" ]]; then
    cp -- "${LICENSE_DIR}/zstd-BSD-3-Clause.txt" \
      "${destination}/internal-complibs/zstd-1.5.6/LICENSE"
  fi
  verify_locked_source_tree "${component_id}" "${destination}"
}

for command in curl git sha256sum tar unzip; do
  require_command "${command}"
done
mkdir -p -- "${CACHE_DIR}" "${BUILD_ROOT}"
bootstrap_locked_build_runtime

if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'OPENVDB_RUNTIME_BUILD_JOBS must be a positive integer\n' >&2
  exit 2
fi

"${PYTHON}" - "${LOCK_FILE}" "${SCRIPT_DIR}" <<'PY'
import hashlib
import json
import pathlib
import re
import subprocess
import sys

lock_path = pathlib.Path(sys.argv[1])
native_dir = pathlib.Path(sys.argv[2])
lock = json.loads(lock_path.read_text(encoding="utf-8"))

if lock.get("schema") != "world-understanding.openvdb-native-source-lock.v1":
    raise SystemExit("unexpected OpenVDB source-lock schema")
if lock["source"]["version"] != "13.0.0":
    raise SystemExit("source lock must select OpenVDB 13.0.0")
if lock["source"]["commit"] != "7c03e1f084873cd1b3422c7ff7aec6ee681b3b38":
    raise SystemExit("source lock contains an unapproved OpenVDB commit")
if lock["distribution"]["python_version"] != "3.12":
    raise SystemExit("native wheel must target CPython 3.12")
if lock["build_tools"].get("cmake") != "3.31.6":
    raise SystemExit("native wheel must use source-locked CMake 3.31.6")
if lock["build_tools"].get("nanobind") != lock["build_inputs"]["nanobind"]["version"]:
    raise SystemExit("nanobind build-tool and source versions differ")
expected_bootstrap = {
    "architecture": "x86_64",
    "python": {
        "implementation": "CPython",
        "version": "3.12.13",
        "build": "20260602",
        "filename": "cpython-3.12.13+20260602-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
        "url": "https://releases.astral.sh/github/python-build-standalone/releases/download/20260602/cpython-3.12.13%2B20260602-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
        "sha256": "191b5188b42886fb8a14968d714571e8f3d1cef92ac7ad4c7e24cc4d0929b194",
        "executable_path": "bin/python3.12",
        "executable_sha256": "d713474fd44339f8bd8150edc7b324d0be494dbf2998a1de04a7b0ecdbc1c2ed",
        "libpython_path": "lib/libpython3.12.so.1.0",
        "libpython_sha256": "8083793fd24e3ca7d04c5e857bc2d4c0115513a4fa6c24b9fbcf496c2e78ef80",
    },
    "uv": {
        "version": "0.11.19",
        "filename": "uv-0.11.19-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "url": "https://files.pythonhosted.org/packages/5a/74/2bd8b51e1d76210fd424ae55ec3f34ded5a10eeff3dd38aeb03c816a0af2/uv-0.11.19-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "sha256": "731d9fab8db5d41590af64236d03f8069c8da665fd0f9493b85985f19c86cd90",
        "executable_path": "uv-0.11.19.data/scripts/uv",
        "executable_sha256": "a00d3a24514fc0403fc232c9c99bf5e542657c38f4ed941e0611731e4cff268b",
    },
}
if lock.get("bootstrap") != expected_bootstrap:
    raise SystemExit("source lock does not select the approved Python/uv bootstrap")
expected_build_tool_wheels = {
    (
        "cmake",
        "3.31.6",
        "cmake-3.31.6-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "1c8b05df0602365da91ee6a3336fe57525b137706c4ab5675498f662ae1dbcec",
    ),
    (
        "scikit-build-core",
        "0.11.6",
        "scikit_build_core-0.11.6-py3-none-any.whl",
        "ce6d8fe64e6b4c759ea0fb95d2f8a68f60d2df31c2989838633b8ec930736360",
    ),
    (
        "auditwheel",
        "6.4.2",
        "auditwheel-6.4.2-py3-none-any.whl",
        "4302ae79dcff242e799a37173cfeeae727d0924843eca4b3f622d3bcb28de2db",
    ),
    (
        "packaging",
        "25.0",
        "packaging-25.0-py3-none-any.whl",
        "29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484",
    ),
    (
        "pathspec",
        "0.12.1",
        "pathspec-0.12.1-py3-none-any.whl",
        "a0d503e138a4c123b27490a4f7beda6a01c6f288df0e4a8b79c7eb0dc7b4cc08",
    ),
    (
        "pyelftools",
        "0.32",
        "pyelftools-0.32-py3-none-any.whl",
        "013df952a006db5e138b1edf6d8a68ecc50630adbd0d83a2d41e7f846163d738",
    ),
}
actual_build_tool_wheels = {
    (item.get("name"), item.get("version"), item.get("filename"), item.get("sha256"))
    for item in lock.get("build_tool_wheels", [])
    if isinstance(item, dict) and set(item) == {"name", "version", "filename", "url", "sha256"}
}
if actual_build_tool_wheels != expected_build_tool_wheels or len(
    lock.get("build_tool_wheels", [])
) != len(expected_build_tool_wheels):
    raise SystemExit("source lock does not select the exact approved build-tool wheels")
if any(
    not item["url"].startswith("https://files.pythonhosted.org/")
    for item in lock["build_tool_wheels"]
):
    raise SystemExit("build-tool wheels must use exact files.pythonhosted.org URLs")
toolchain = lock.get("toolchain")
if not isinstance(toolchain, dict) or toolchain.get("architecture") != "x86_64":
    raise SystemExit("source lock must bind the approved x86_64 native toolchain")
if toolchain.get("compiler_family") != "GNU" or toolchain.get("compiler_version") != "11.4.0":
    raise SystemExit("source lock must bind the approved GNU compiler identity")
build_recipe = lock.get("build_recipe")
if not isinstance(build_recipe, dict) or build_recipe.get("path") != "build_openvdb_wheel.sh":
    raise SystemExit("source lock does not bind the native build recipe")
recipe_path = native_dir / build_recipe["path"]
if hashlib.sha256(recipe_path.read_bytes()).hexdigest() != build_recipe.get("sha256"):
    raise SystemExit("native build recipe digest mismatch")

expected_sources = {
    "openvdb": ("13.0.0", "7c03e1f084873cd1b3422c7ff7aec6ee681b3b38"),
    "onetbb": ("2022.2.0", "06ce6212da6710f4bb2d20a1904b018aa44069bf"),
    "c-blosc": ("1.21.6", "616f4b7343a8479f7e71dd3d7025bd92c9a6bbd0"),
    "zlib": ("1.3.1", "51b7f2abdade71cd9bb0e7a373ef2610ec6f9daf"),
    "nanobind": ("2.13.0", "e2dc00f7a34f935c6cf91948776d59c4709e9fe6"),
    "robin-map": ("1.4.0", "4ec1bf19c6a96125ea22062f38c2cf5b958e448e"),
}
sources = [lock["source"], *lock["native_dependencies"]]
sources.extend(lock["build_inputs"][name] for name in ("nanobind", "robin-map"))
if {item["id"] for item in sources} != set(expected_sources):
    raise SystemExit("source lock contains a missing or unknown compiled source")
for item in sources:
    if (item["version"], item["commit"]) != expected_sources[item["id"]]:
        raise SystemExit(f"unapproved source identity for {item['id']}")
    if not item.get("archive_sha256") or not item.get("git_tree_inventory_sha256"):
        raise SystemExit(f"source lock is missing archive evidence for {item['id']}")
    if not item.get("license_files"):
        raise SystemExit(f"source lock is missing license evidence for {item['id']}")

expected_vendored = {
    "openvdb": {"openexr-half"},
    "c-blosc": {
        "bitshuffle",
        "fastlz",
        "libdivsufsort-lite",
        "lz4",
        "zlib-ng",
        "zstandard",
    },
}
for source_id, expected_ids in expected_vendored.items():
    containing_source = next(item for item in sources if item["id"] == source_id)
    vendored = containing_source.get("vendored_components", [])
    if {item["id"] for item in vendored} != expected_ids:
        raise SystemExit(f"{source_id} vendored component inventory mismatch")
    for item in vendored:
        if item["containing_source_id"] != source_id or not item.get("license_files"):
            raise SystemExit(f"invalid vendored provenance for {item['id']}")
if lock["disabled_native_dependencies"].get("snappy") is None:
    raise SystemExit("source lock must explicitly disable Snappy")
onetbb = next(item for item in sources if item["id"] == "onetbb")
excluded_features = onetbb.get("excluded_source_features", [])
if len(excluded_features) != 1:
    raise SystemExit("oneTBB must record exactly one toolchain-excluded source feature")
excluded = excluded_features[0]
if (
    excluded.get("id") != "gcc-libstdcxx-bug-62258-workaround"
    or excluded.get("source_path") != "src/tbb/exception.cpp"
    or excluded.get("compiled") is not False
    or excluded.get("required_absent_symbol") != "__cxa_get_globals"
):
    raise SystemExit("oneTBB GCC ABI workaround exclusion is not locked")

listed_patches = [item["path"] for item in lock["patches"]]
disk_patches = [
    path.relative_to(native_dir).as_posix()
    for path in sorted((native_dir / "patches").glob("*.patch"))
]
if listed_patches != sorted(listed_patches) or listed_patches != disk_patches:
    raise SystemExit("source-lock patch inventory does not match numbered patch files")
for item in lock["patches"]:
    path = native_dir / item["path"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != item["sha256"]:
        raise SystemExit(f"patch digest mismatch: {item['path']}")

listed_native_patches = sorted(
    patch["path"]
    for source in sources
    for patch in source.get("patches", [])
)
disk_native_patches = [
    path.relative_to(native_dir).as_posix()
    for path in sorted((native_dir / "native-patches").rglob("*.patch"))
]
if listed_native_patches != disk_native_patches:
    raise SystemExit("source-lock native patch inventory does not match disk")
for source in sources:
    for patch in source.get("patches", []):
        path = native_dir / patch["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != patch["sha256"]:
            raise SystemExit(f"native patch digest mismatch: {patch['path']}")

listed_redistribution_files = [item["path"] for item in lock["redistribution_files"]]
disk_redistribution_files = ["SOURCE_PROVENANCE.md", "THIRD_PARTY_NOTICES.md"] + [
    path.relative_to(native_dir).as_posix()
    for path in sorted((native_dir / "licenses").glob("*"))
]
if listed_redistribution_files != disk_redistribution_files:
    raise SystemExit("source-lock redistribution inventory does not match disk")
for item in lock["redistribution_files"]:
    path = native_dir / item["path"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != item["sha256"]:
        raise SystemExit(f"redistribution file digest mismatch: {item['path']}")

license_admission = lock.get("license_admission")
if not isinstance(license_admission, dict) or set(license_admission) != {"policy", "verifier"}:
    raise SystemExit("source lock must bind the native license policy and verifier")
expected_admission_files = {
    "policy": (
        "license-policy.json",
        "world-understanding.sdf-native-license-policy.v2",
        "schema",
    ),
    "verifier": (
        "verify_wheel_license_policy.py",
        "world-understanding.sdf-native-license-admission.v1",
        "report_schema",
    ),
}
for kind, (expected_path, expected_schema, schema_key) in expected_admission_files.items():
    item = license_admission.get(kind)
    required_keys = {"path", "sha256", schema_key}
    if not isinstance(item, dict) or set(item) != required_keys:
        raise SystemExit(f"invalid source-lock native license {kind} binding")
    if item["path"] != expected_path or item[schema_key] != expected_schema:
        raise SystemExit(f"unexpected source-lock native license {kind} identity")
    path = native_dir / item["path"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != item["sha256"]:
        raise SystemExit(f"native license {kind} digest mismatch: {item['path']}")

overlay_root = native_dir / lock["overlay"]["path"]
digest = hashlib.sha256()
for path in sorted(candidate for candidate in overlay_root.rglob("*") if candidate.is_file()):
    digest.update(path.relative_to(overlay_root).as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
if digest.hexdigest() != lock["overlay"]["inventory_sha256"]:
    raise SystemExit("OpenVDB source overlay digest mismatch")
PY

mapfile -t LOCK_VALUES < <(
  "${PYTHON}" - "${LOCK_FILE}" <<'PY'
import json
import pathlib
import sys

lock = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(lock["source"]["archive_url"])
print(lock["source"]["archive_sha256"])
print(lock["source"]["source_date_epoch"])
print(lock["distribution"]["version"])
print(lock["wheel_policy"]["linux"])
for item in lock["patches"]:
    print(item["path"])
PY
)
ARCHIVE_URL="${LOCK_VALUES[0]}"
ARCHIVE_SHA256="${LOCK_VALUES[1]}"
SOURCE_DATE_EPOCH="${LOCK_VALUES[2]}"
DISTRIBUTION_VERSION="${LOCK_VALUES[3]}"
LOCKED_LINUX_POLICY="${LOCK_VALUES[4]}"
PATCH_PATHS=("${LOCK_VALUES[@]:5}")

mkdir -p -- "${CACHE_DIR}" "${BUILD_ROOT}"
declare -A SOURCE_URLS SOURCE_SHA256S
while IFS=$'\t' read -r component_id url archive_sha256; do
  SOURCE_URLS["${component_id}"]="${url}"
  SOURCE_SHA256S["${component_id}"]="${archive_sha256}"
done < <(
  "${PYTHON}" - "${LOCK_FILE}" <<'PY'
import json
import pathlib
import sys

lock = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
sources = [lock["source"], *lock["native_dependencies"]]
sources.extend(lock["build_inputs"][name] for name in ("nanobind", "robin-map"))
for source in sources:
    print(source["id"], source["archive_url"], source["archive_sha256"], sep="\t")
PY
)

rm -rf -- \
  "${SOURCE_DIR}" \
  "${NATIVE_SOURCE_ROOT}" \
  "${NATIVE_BUILD_ROOT}" \
  "${NATIVE_PREFIX}" \
  "${CMAKE_BUILD_DIR}" \
  "${RAW_WHEEL_DIR}" \
  "${REPAIRED_WHEEL_DIR}" \
  "${NANOBIND_WHEEL_DIR}"
mkdir -p -- \
  "${NATIVE_SOURCE_ROOT}" \
  "${NATIVE_BUILD_ROOT}" \
  "${NATIVE_PREFIX}" \
  "${RAW_WHEEL_DIR}" \
  "${REPAIRED_WHEEL_DIR}" \
  "${NANOBIND_WHEEL_DIR}"

prepare_locked_source \
  openvdb "${ARCHIVE_URL}" "${ARCHIVE_SHA256}" "${SOURCE_DIR}" \
  "${OPENVDB_RUNTIME_SOURCE_ARCHIVE:-}"
for component_id in onetbb c-blosc zlib nanobind robin-map; do
  prepare_locked_source \
    "${component_id}" \
    "${SOURCE_URLS[${component_id}]}" \
    "${SOURCE_SHA256S[${component_id}]}" \
    "${NATIVE_SOURCE_ROOT}/${component_id}"
done

while IFS=$'\t' read -r component_id patch_path; do
  if [[ -z "${component_id}" ]]; then
    continue
  fi
  component_source="${NATIVE_SOURCE_ROOT}/${component_id}"
  git -C "${component_source}" apply --check "${SCRIPT_DIR}/${patch_path}"
  git -C "${component_source}" apply "${SCRIPT_DIR}/${patch_path}"
  git -C "${component_source}" apply --reverse --check \
    "${SCRIPT_DIR}/${patch_path}"
done < <(
  "${PYTHON}" - "${LOCK_FILE}" <<'PY'
import json
import pathlib
import sys

lock = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
sources = [*lock["native_dependencies"]]
sources.extend(lock["build_inputs"][name] for name in ("nanobind", "robin-map"))
for source in sources:
    for patch in source.get("patches", []):
        print(source["id"], patch["path"], sep="\t")
PY
)

if [[ -e "${NATIVE_SOURCE_ROOT}/c-blosc/internal-complibs/snappy-1.1.1" ]]; then
  printf 'locked c-blosc source unexpectedly contains Snappy\n' >&2
  exit 3
fi
mkdir -p -- "${NATIVE_SOURCE_ROOT}/nanobind/ext/robin_map"
cp -a -- "${NATIVE_SOURCE_ROOT}/robin-map/." \
  "${NATIVE_SOURCE_ROOT}/nanobind/ext/robin_map/"

cp -a -- "${OVERLAY_DIR}/." "${SOURCE_DIR}/"
for patch_path in "${PATCH_PATHS[@]}"; do
  git -C "${SOURCE_DIR}" apply --check "${SCRIPT_DIR}/${patch_path}"
  git -C "${SOURCE_DIR}" apply "${SCRIPT_DIR}/${patch_path}"
done

cp -- "${LOCK_FILE}" "${SOURCE_DIR}/WORLD_UNDERSTANDING_SOURCE_LOCK.json"
cp -- "${SCRIPT_DIR}/SOURCE_PROVENANCE.md" "${SOURCE_DIR}/SOURCE_PROVENANCE.md"
cp -- "${SCRIPT_DIR}/THIRD_PARTY_NOTICES.md" "${SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
mkdir -p -- "${SOURCE_DIR}/LICENSES"
cp -- "${LICENSE_DIR}/"* "${SOURCE_DIR}/LICENSES/"

"${PYTHON}" - "${SOURCE_DIR}/pyproject.toml" "${DISTRIBUTION_VERSION}" <<'PY'
import pathlib
import sys
import tomllib

metadata = tomllib.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if metadata["project"]["version"] != sys.argv[2]:
    raise SystemExit("patched OpenVDB distribution version does not match source lock")
if metadata["tool"]["scikit-build"]["cmake"]["define"]["OPENVDB_BUILD_PYTHON_MODULE"] != "ON":
    raise SystemExit("official OpenVDB Python module is not enabled")
PY

if [[ "${MODE}" == "prepare" ]]; then
  printf 'Prepared verified OpenVDB source at %s\n' "${SOURCE_DIR}"
  exit 0
fi

for command in ldd; do
  require_command "${command}"
done

"${PYTHON}" - <<'PY'
import platform
import sys

if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 12):
    raise SystemExit("OpenVDB wheel build requires CPython 3.12")
if sys.platform != "linux":
    raise SystemExit("this wheel repair path currently supports Linux only")
PY

case "$(uname -m)" in
  x86_64) BUILD_ARCH=x86_64 ;;
  *) printf 'no locked native toolchain for architecture: %s\n' "$(uname -m)" >&2; exit 4 ;;
esac

mapfile -t TOOLCHAIN_PATHS < <(
  "${PYTHON}" - "${LOCK_FILE}" "${BUILD_ARCH}" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

lock = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
architecture = sys.argv[2]
expected = {
    "architecture": "x86_64",
    "target": "x86_64-linux-gnu",
    "compiler_family": "GNU",
    "compiler_version": "11.4.0",
    "tools": {
        "cc": {"path": "/usr/bin/x86_64-linux-gnu-gcc-11", "sha256": "821af3c74506283c179ca413bb33e6b528805a4dd8a5c09df125e5ad560a9e89", "version": "x86_64-linux-gnu-gcc-11 (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0"},
        "cxx": {"path": "/usr/bin/x86_64-linux-gnu-g++-11", "sha256": "2360901d864cf10bfd6296e261cb2c14053552a80377761ab07146ec9ec9a2c0", "version": "x86_64-linux-gnu-g++-11 (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0"},
        "ld": {"path": "/usr/bin/x86_64-linux-gnu-ld.bfd", "sha256": "58937fc20c21e147883b4fdaa0fc7438a8e8f2bb886cfcaa4896100ca91139e7", "version": "GNU ld (GNU Binutils for Ubuntu) 2.38"},
        "ar": {"path": "/usr/bin/x86_64-linux-gnu-ar", "sha256": "5b394f8752fde1776992d405d76034d5017b24b9b922bedaeeaff8cc344ef044", "version": "GNU ar (GNU Binutils for Ubuntu) 2.38"},
        "ranlib": {"path": "/usr/bin/x86_64-linux-gnu-ranlib", "sha256": "95e470c76f1fb0383ef5c10cfdf10127aefab07dad924fe94933ec8be8a5a9fb", "version": "GNU ranlib (GNU Binutils for Ubuntu) 2.38"},
        "as": {"path": "/usr/bin/x86_64-linux-gnu-as", "sha256": "4e6b50c3faaa834150db32be778fd7d9440a4e1f5fa8beb8a72277b12159d689", "version": "GNU assembler (GNU Binutils for Ubuntu) 2.38"},
        "nm": {"path": "/usr/bin/x86_64-linux-gnu-nm", "sha256": "ed6565c132d6f20019ff3f050043dc1b2b8bc2e8b3b94c1ffe3f34436431a5a4", "version": "GNU nm (GNU Binutils for Ubuntu) 2.38"},
        "strip": {"path": "/usr/bin/x86_64-linux-gnu-strip", "sha256": "001150abd7bbfaf605f2b091a2a3db7d00aa05aba1b6c62dc7cbb416336e23c7", "version": "GNU strip (GNU Binutils for Ubuntu) 2.38"},
        "objcopy": {"path": "/usr/bin/x86_64-linux-gnu-objcopy", "sha256": "876729646822863167b0a2cb03e5a1052d82755824026a664673ddb54f05b54f", "version": "GNU objcopy (GNU Binutils for Ubuntu) 2.38"},
        "objdump": {"path": "/usr/bin/x86_64-linux-gnu-objdump", "sha256": "1eaaef2e7f57c4c7f69115c495e2466f5a8c8e5f3bc42221d092382f30f9d4cd", "version": "GNU objdump (GNU Binutils for Ubuntu) 2.38"},
        "readelf": {"path": "/usr/bin/x86_64-linux-gnu-readelf", "sha256": "04db0000749aff89e4af21429340b00b536fc6f80e811c872c006507881a5560", "version": "GNU readelf (GNU Binutils for Ubuntu) 2.38"},
        "strings": {"path": "/usr/bin/x86_64-linux-gnu-strings", "sha256": "2f72e4d1091897cfec64708064307bed19e2d79ab9aa0816022bcc8368482353", "version": "GNU strings (GNU Binutils for Ubuntu) 2.38"},
        "ninja": {"path": "/usr/bin/ninja", "sha256": "a9ddfbf7e365a2e5026ed44f9173e498e37d8b71ec5e259b2c2a684f16cfa862", "version": "1.10.1"},
        "patchelf": {"path": "/usr/bin/patchelf", "sha256": "ebc7bde09fdc923acdfade62f49c32b7c32de3d64f83aea14fc38143ecf77c25", "version": "patchelf 0.14.3"},
        "gcc_ar": {"path": "/usr/bin/x86_64-linux-gnu-gcc-ar-11", "sha256": "e1697276051ba32ecc4b715875f7dd232491a282d274d97ce32a826a0325c5d3", "version": "GNU ar (GNU Binutils for Ubuntu) 2.38"},
        "gcc_ranlib": {"path": "/usr/bin/x86_64-linux-gnu-gcc-ranlib-11", "sha256": "da8f6901ca3ed58b7abef4aea8ffee05faad80b6061bee81c7dff5abcf392777", "version": "GNU ranlib (GNU Binutils for Ubuntu) 2.38"},
    },
    "compiler_components": {
        "cc1": {"path": "/usr/lib/gcc/x86_64-linux-gnu/11/cc1", "sha256": "31c2233432d9105001eea158b799f7d403dc5a1944c712283dba251f3ab8eb43"},
        "cc1plus": {"path": "/usr/lib/gcc/x86_64-linux-gnu/11/cc1plus", "sha256": "283421000e15a9152de4affe89ac2e528f3e0bcf2038b245171a17d8a912ace9"},
        "collect2": {"path": "/usr/lib/gcc/x86_64-linux-gnu/11/collect2", "sha256": "74da9a263cce4582f98c7995d04fa52110c8366d65cba1c5bc976a00f1b3f14b"},
        "liblto_plugin": {"path": "/usr/lib/gcc/x86_64-linux-gnu/11/liblto_plugin.so", "sha256": "fa42ac5108eaf6d91d2299ac8387278ac4add3d8f60131cf14267730867188dd"},
        "lto_wrapper": {"path": "/usr/lib/gcc/x86_64-linux-gnu/11/lto-wrapper", "sha256": "eedb0194034b36c3d22f2912eaeac300253b4be3cdba8e3d8999699f129da9f4"},
    },
}
toolchain = lock.get("toolchain")
if architecture != expected["architecture"] or toolchain != expected:
    raise SystemExit("source lock does not select the approved deterministic GNU toolchain")

def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

for name, tool in toolchain["tools"].items():
    path = pathlib.Path(tool["path"])
    if not path.is_file() or path.stat().st_mode & 0o022:
        raise SystemExit(f"locked tool is missing or writable by group/other: {name}")
    if digest(path) != tool["sha256"]:
        raise SystemExit(f"locked tool digest mismatch: {name}")
    version = subprocess.check_output(
        [path, "--version"], stderr=subprocess.STDOUT, text=True
    ).splitlines()[0]
    if version != tool["version"]:
        raise SystemExit(f"locked tool version mismatch: {name}")
for name, component in toolchain["compiler_components"].items():
    path = pathlib.Path(component["path"])
    if not path.is_file() or path.stat().st_mode & 0o022:
        raise SystemExit(f"locked compiler component is missing or writable: {name}")
    if digest(path) != component["sha256"]:
        raise SystemExit(f"locked compiler component digest mismatch: {name}")

cc = toolchain["tools"]["cc"]["path"]
cxx = toolchain["tools"]["cxx"]["path"]
if subprocess.check_output([cc, "-dumpfullversion"], text=True).strip() != "11.4.0":
    raise SystemExit("locked C compiler full version mismatch")
if subprocess.check_output([cxx, "-dumpfullversion"], text=True).strip() != "11.4.0":
    raise SystemExit("locked C++ compiler full version mismatch")
if subprocess.check_output([cc, "-dumpmachine"], text=True).strip() != toolchain["target"]:
    raise SystemExit("locked compiler target mismatch")
if pathlib.Path(subprocess.check_output([cc, "-print-prog-name=cc1"], text=True).strip()) != pathlib.Path(toolchain["compiler_components"]["cc1"]["path"]):
    raise SystemExit("locked C compiler resolves an unexpected cc1")
if pathlib.Path(subprocess.check_output([cxx, "-print-prog-name=cc1plus"], text=True).strip()) != pathlib.Path(toolchain["compiler_components"]["cc1plus"]["path"]):
    raise SystemExit("locked C++ compiler resolves an unexpected cc1plus")
if pathlib.Path(subprocess.check_output([cc, "-print-prog-name=collect2"], text=True).strip()) != pathlib.Path(toolchain["compiler_components"]["collect2"]["path"]):
    raise SystemExit("locked compiler resolves an unexpected collect2")
if pathlib.Path(subprocess.check_output([cc, "-print-file-name=liblto_plugin.so"], text=True).strip()) != pathlib.Path(toolchain["compiler_components"]["liblto_plugin"]["path"]):
    raise SystemExit("locked compiler resolves an unexpected LTO plugin")
if pathlib.Path(subprocess.check_output([cc, "-print-prog-name=lto-wrapper"], text=True).strip()) != pathlib.Path(toolchain["compiler_components"]["lto_wrapper"]["path"]):
    raise SystemExit("locked compiler resolves an unexpected LTO wrapper")

for name in ("cc", "cxx", "ld", "ar", "ranlib", "as", "nm", "strip", "objcopy", "objdump", "readelf", "strings", "ninja", "patchelf", "gcc_ar", "gcc_ranlib"):
    print(toolchain["tools"][name]["path"])
PY
)
LOCKED_CC="${TOOLCHAIN_PATHS[0]}"
LOCKED_CXX="${TOOLCHAIN_PATHS[1]}"
LOCKED_LD="${TOOLCHAIN_PATHS[2]}"
LOCKED_AR="${TOOLCHAIN_PATHS[3]}"
LOCKED_RANLIB="${TOOLCHAIN_PATHS[4]}"
LOCKED_AS="${TOOLCHAIN_PATHS[5]}"
LOCKED_NM="${TOOLCHAIN_PATHS[6]}"
LOCKED_STRIP="${TOOLCHAIN_PATHS[7]}"
LOCKED_OBJCOPY="${TOOLCHAIN_PATHS[8]}"
LOCKED_OBJDUMP="${TOOLCHAIN_PATHS[9]}"
LOCKED_READELF="${TOOLCHAIN_PATHS[10]}"
LOCKED_STRINGS="${TOOLCHAIN_PATHS[11]}"
LOCKED_NINJA="${TOOLCHAIN_PATHS[12]}"
LOCKED_PATCHELF="${TOOLCHAIN_PATHS[13]}"
LOCKED_GCC_AR="${TOOLCHAIN_PATHS[14]}"
LOCKED_GCC_RANLIB="${TOOLCHAIN_PATHS[15]}"

PYTHON_BUILD_PREFIX="$("${PYTHON}" -c 'import sys; print(sys.base_prefix)')"
export CMAKE_GENERATOR=Ninja
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"
export SKBUILD_BUILD_DIR="${CMAKE_BUILD_DIR}"
export SOURCE_DATE_EPOCH
export PYTHONHASHSEED=0
PREFIX_MAP_FLAGS="-ffile-prefix-map=${BUILD_ROOT}=. -fdebug-prefix-map=${BUILD_ROOT}=. -fmacro-prefix-map=${BUILD_ROOT}=."
PINNED_BINUTILS_PREFIX="-B/usr/bin/x86_64-linux-gnu-"
export ASMFLAGS="${PINNED_BINUTILS_PREFIX}"
export CFLAGS="${PINNED_BINUTILS_PREFIX} ${PREFIX_MAP_FLAGS} -fno-record-gcc-switches"
export CXXFLAGS="${PINNED_BINUTILS_PREFIX} ${PREFIX_MAP_FLAGS} -fno-record-gcc-switches"
export CPPFLAGS=""
export LDFLAGS="${PINNED_BINUTILS_PREFIX} -fuse-ld=bfd -Wl,--build-id=sha1"
export CC="${LOCKED_CC}"
export CXX="${LOCKED_CXX}"
export LD="${LOCKED_LD}"
export AR="${LOCKED_AR}"
export RANLIB="${LOCKED_RANLIB}"
export AS="${LOCKED_AS}"
export NM="${LOCKED_NM}"
export STRIP="${LOCKED_STRIP}"
export OBJCOPY="${LOCKED_OBJCOPY}"
export OBJDUMP="${LOCKED_OBJDUMP}"
export READELF="${LOCKED_READELF}"

rm -rf -- "${TOOL_ENV}" "${SMOKE_ENV}" "${BUILD_TOOL_WHEEL_DIR}"
mkdir -p -- "${BUILD_TOOL_WHEEL_DIR}"
while IFS=$'\t' read -r tool_name tool_url tool_sha256 tool_filename; do
  download_locked_file \
    "${tool_name} build-tool wheel" \
    "${tool_url}" \
    "${tool_sha256}" \
    "${BUILD_TOOL_WHEEL_DIR}/${tool_filename}"
done < <(
  "${PYTHON}" - "${LOCK_FILE}" <<'PY'
import json
import pathlib
import sys

lock = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in sorted(lock["build_tool_wheels"], key=lambda value: value["filename"]):
    print(item["name"], item["url"], item["sha256"], item["filename"], sep="\t")
PY
)
"${UV_BIN}" venv --python "${PYTHON}" --no-python-downloads --no-project --clear "${TOOL_ENV}"
TOOL_PYTHON="${TOOL_ENV}/bin/python"
mapfile -t LOCKED_BUILD_TOOL_WHEELS < <(
  find "${BUILD_TOOL_WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' -print | sort
)
if [[ "${#LOCKED_BUILD_TOOL_WHEELS[@]}" -ne 6 ]]; then
  printf 'expected six exact build-tool wheels, found %s\n' \
    "${#LOCKED_BUILD_TOOL_WHEELS[@]}" >&2
  exit 3
fi
"${UV_BIN}" pip install --python "${TOOL_PYTHON}" --offline --no-index \
  --no-cache --no-deps "${LOCKED_BUILD_TOOL_WHEELS[@]}"
"${TOOL_PYTHON}" - "${LOCK_FILE}" <<'PY'
import importlib.metadata
import json
import pathlib
import sys

lock = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "cmake": lock["build_tools"]["cmake"],
    "scikit-build-core": lock["build_tools"]["scikit_build_core"],
    "auditwheel": lock["build_tools"]["auditwheel"],
    "packaging": lock["build_tools"]["packaging"],
    "pathspec": lock["build_tools"]["pathspec"],
    "pyelftools": lock["build_tools"]["pyelftools"],
}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"installed build-tool versions differ from source lock: {actual!r}")
PY
CMAKE_BIN="${TOOL_ENV}/bin/cmake"
export PATH="${TOOL_ENV}/bin:/usr/bin:/bin"
LOCKED_CMAKE_TOOLCHAIN_ARGS=(
  "-DCMAKE_C_COMPILER=${LOCKED_CC}"
  "-DCMAKE_CXX_COMPILER=${LOCKED_CXX}"
  "-DCMAKE_ASM_COMPILER=${LOCKED_CC}"
  "-DCMAKE_LINKER=${LOCKED_LD}"
  "-DCMAKE_AR=${LOCKED_AR}"
  "-DCMAKE_RANLIB=${LOCKED_RANLIB}"
  "-DCMAKE_C_COMPILER_AR=${LOCKED_GCC_AR}"
  "-DCMAKE_CXX_COMPILER_AR=${LOCKED_GCC_AR}"
  "-DCMAKE_C_COMPILER_RANLIB=${LOCKED_GCC_RANLIB}"
  "-DCMAKE_CXX_COMPILER_RANLIB=${LOCKED_GCC_RANLIB}"
  "-DCMAKE_NM=${LOCKED_NM}"
  "-DCMAKE_STRIP=${LOCKED_STRIP}"
  "-DCMAKE_OBJCOPY=${LOCKED_OBJCOPY}"
  "-DCMAKE_OBJDUMP=${LOCKED_OBJDUMP}"
  "-DCMAKE_READELF=${LOCKED_READELF}"
  "-DCMAKE_MAKE_PROGRAM=${LOCKED_NINJA}"
  "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE"
  "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE"
  "-DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON"
  "-DCMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY=ON"
)
export CMAKE_ARGS="${LOCKED_CMAKE_TOOLCHAIN_ARGS[*]}"
if [[ "$("${CMAKE_BIN}" --version | awk 'NR == 1 {print $3}')" != "3.31.6" ]]; then
  printf 'OpenVDB wheel build requires CMake 3.31.6\n' >&2
  exit 2
fi

"${UV_BIN}" build --python "${TOOL_PYTHON}" --no-python-downloads --no-sources \
  --no-build-isolation --wheel --out-dir "${NANOBIND_WHEEL_DIR}" \
  "${NATIVE_SOURCE_ROOT}/nanobind"
mapfile -t NANOBIND_WHEELS < <(
  find "${NANOBIND_WHEEL_DIR}" -maxdepth 1 -type f -name 'nanobind-2.13.0-*.whl' -print
)
if [[ "${#NANOBIND_WHEELS[@]}" -ne 1 ]]; then
  printf 'expected one source-built nanobind wheel, found %s\n' \
    "${#NANOBIND_WHEELS[@]}" >&2
  exit 4
fi
NANOBIND_WHEEL="${NANOBIND_WHEELS[0]}"
"${UV_BIN}" pip install --python "${TOOL_PYTHON}" --no-cache --no-deps "${NANOBIND_WHEEL}"
if [[ "$("${TOOL_PYTHON}" -c 'import nanobind; print(nanobind.__version__)')" != "2.13.0" ]]; then
  printf 'source-built nanobind version mismatch\n' >&2
  exit 4
fi

ZLIB_SOURCE="${NATIVE_SOURCE_ROOT}/zlib"
ZLIB_BUILD="${NATIVE_BUILD_ROOT}/zlib"
"${CMAKE_BIN}" -S "${ZLIB_SOURCE}" -B "${ZLIB_BUILD}" -G Ninja \
  "${LOCKED_CMAKE_TOOLCHAIN_ARGS[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${NATIVE_PREFIX}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DINSTALL_INC_DIR="${NATIVE_PREFIX}/include" \
  -DINSTALL_LIB_DIR="${NATIVE_PREFIX}/lib" \
  -DSKIP_INSTALL_FILES=ON \
  -DZLIB_BUILD_EXAMPLES=OFF
"${CMAKE_BIN}" --build "${ZLIB_BUILD}" --parallel "${JOBS}"
"${CMAKE_BIN}" --install "${ZLIB_BUILD}"
rm -f -- "${NATIVE_PREFIX}"/lib/libz.so "${NATIVE_PREFIX}"/lib/libz.so.*
if [[ ! -f "${NATIVE_PREFIX}/lib/libz.a" ]]; then
  printf 'source-built static zlib archive is missing\n' >&2
  exit 4
fi

export ZLIB_ROOT="${NATIVE_PREFIX}"
BLOSC_SOURCE="${NATIVE_SOURCE_ROOT}/c-blosc"
BLOSC_BUILD="${NATIVE_BUILD_ROOT}/c-blosc"
"${CMAKE_BIN}" -S "${BLOSC_SOURCE}" -B "${BLOSC_BUILD}" -G Ninja \
  "${LOCKED_CMAKE_TOOLCHAIN_ARGS[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${NATIVE_PREFIX}" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_BENCHMARKS=OFF \
  -DBUILD_FUZZERS=OFF \
  -DBUILD_SHARED=OFF \
  -DBUILD_STATIC=ON \
  -DBUILD_TESTS=OFF \
  -DDEACTIVATE_LZ4=OFF \
  -DDEACTIVATE_SNAPPY=ON \
  -DDEACTIVATE_ZLIB=OFF \
  -DDEACTIVATE_ZSTD=OFF \
  -DPREFER_EXTERNAL_LZ4=OFF \
  -DPREFER_EXTERNAL_ZLIB=ON \
  -DPREFER_EXTERNAL_ZSTD=OFF \
  -DZLIB_INCLUDE_DIR="${NATIVE_PREFIX}/include" \
  -DZLIB_LIBRARY="${NATIVE_PREFIX}/lib/libz.a"
"${CMAKE_BIN}" --build "${BLOSC_BUILD}" --parallel "${JOBS}"
"${CMAKE_BIN}" --install "${BLOSC_BUILD}"
if [[ ! -f "${NATIVE_PREFIX}/lib/libblosc.a" ]]; then
  printf 'source-built static c-blosc archive is missing\n' >&2
  exit 4
fi
BLOSC_ARCHIVE_MEMBERS="$("${LOCKED_AR}" t "${NATIVE_PREFIX}/lib/libblosc.a")"
if ! grep -qx 'lz4.c.o' <<<"${BLOSC_ARCHIVE_MEMBERS}" || \
  ! grep -qx 'zstd_common.c.o' <<<"${BLOSC_ARCHIVE_MEMBERS}" || \
  ! grep -qx 'divsufsort.c.o' <<<"${BLOSC_ARCHIVE_MEMBERS}"; then
  printf 'source-built c-blosc archive is missing its locked bundled codecs\n' >&2
  exit 4
fi
if grep -Eq '(^|/)(snappy|adler32|deflate|inflate)[.]c[.]o$' \
  <<<"${BLOSC_ARCHIVE_MEMBERS}"; then
  printf 'c-blosc compiled Snappy or its vendored zlib instead of the locked external zlib\n' >&2
  exit 4
fi

TBB_SOURCE="${NATIVE_SOURCE_ROOT}/onetbb"
TBB_BUILD="${NATIVE_BUILD_ROOT}/onetbb"
"${CMAKE_BIN}" -S "${TBB_SOURCE}" -B "${TBB_BUILD}" -G Ninja \
  "${LOCKED_CMAKE_TOOLCHAIN_ARGS[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${NATIVE_PREFIX}" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DBUILD_SHARED_LIBS=ON \
  -DTBB4PY_BUILD=OFF \
  -DTBBMALLOC_BUILD=OFF \
  -DTBBMALLOC_PROXY_BUILD=OFF \
  -DTBB_CPF=OFF \
  -DTBB_DISABLE_HWLOC_AUTOMATIC_SEARCH=ON \
  -DTBB_ENABLE_IPO=OFF \
  -DTBB_ENABLE_ITT_NOTIFY=OFF \
  -DTBB_ENABLE_OPTIONAL_DYNAMIC_LOADING=OFF \
  -DTBB_EXAMPLES=OFF \
  -DTBB_FIND_PACKAGE=OFF \
  -DTBB_INSTALL=ON \
  -DTBB_STRICT=OFF \
  -DTBB_TEST=OFF
"${PYTHON}" - "${TBB_BUILD}/CMakeCache.txt" "${TBB_BUILD}/build.ninja" \
  "${LOCKED_CXX}" "${LOCKED_NINJA}" <<'PY'
import pathlib
import sys

cache_path = pathlib.Path(sys.argv[1])
ninja_path = pathlib.Path(sys.argv[2])
expected_cxx = pathlib.Path(sys.argv[3]).resolve()
expected_ninja = pathlib.Path(sys.argv[4]).resolve()
values = {}
for line in cache_path.read_text(encoding="utf-8").splitlines():
    if not line or line.startswith(("//", "#")) or "=" not in line or ":" not in line:
        continue
    key_type, value = line.split("=", 1)
    key, _ = key_type.split(":", 1)
    values[key] = value

for key in ("TBB_ENABLE_ITT_NOTIFY", "TBB_ENABLE_OPTIONAL_DYNAMIC_LOADING"):
    if values.get(key) != "OFF":
        raise SystemExit(f"oneTBB optional runtime feature was not disabled: {key}")
if pathlib.Path(values.get("CMAKE_CXX_COMPILER", "missing")).resolve() != expected_cxx:
    raise SystemExit("oneTBB resolved an unapproved C++ compiler")
if pathlib.Path(values.get("CMAKE_MAKE_PROGRAM", "missing")).resolve() != expected_ninja:
    raise SystemExit("oneTBB resolved an unapproved build program")

ninja = ninja_path.read_text(encoding="utf-8")
if "__TBB_USE_ITT_NOTIFY" in ninja:
    raise SystemExit("oneTBB compile graph still enables ITT notification loading")
for definition in ("__TBB_DYNAMIC_LOAD_ENABLED=0", "__TBB_WEAK_SYMBOLS_PRESENT=0"):
    if definition not in ninja:
        raise SystemExit(f"oneTBB compile graph is missing {definition}")
PY
"${CMAKE_BIN}" --build "${TBB_BUILD}" --parallel "${JOBS}"
"${CMAKE_BIN}" --install "${TBB_BUILD}"
if [[ ! -f "${NATIVE_PREFIX}/lib/libtbb.so.12.16" ]] || \
  [[ "$("${LOCKED_READELF}" -d "${NATIVE_PREFIX}/lib/libtbb.so.12.16" | \
    awk '/SONAME/ {print $NF}')" != "[libtbb.so.12]" ]]; then
  printf 'source-built oneTBB artifact identity mismatch\n' >&2
  exit 4
fi
"${PYTHON}" - "${NATIVE_PREFIX}/lib/libtbb.so.12.16" \
  "${LOCKED_READELF}" \
  "${TBB_BUILD}/src/tbb/CMakeFiles/tbb.dir/exception.cpp.o" <<'PY'
import pathlib
import re
import subprocess
import sys

binary = pathlib.Path(sys.argv[1])
readelf = pathlib.Path(sys.argv[2])
exception_object = pathlib.Path(sys.argv[3])
data = binary.read_bytes()
for probe in (
    b"libittnotify.so",
    b"libtbbbind.so.3",
    b"libtbbbind_2_0.so.3",
    b"libtbbbind_2_5.so.3",
    b"libtbbmalloc.so.2",
    b"libiomp5.so",
    b"libirml.so.1",
    b"libtcm.so.1",
    b"INTEL_ITTNOTIFY_GROUPS",
    b"INTEL_LIBITTNOTIFY64",
):
    if probe in data:
        raise SystemExit(f"oneTBB binary retains optional runtime probe: {probe!r}")

symbols = subprocess.check_output([readelf, "-Ws", binary], text=True)
if re.search(r"\bUND\b.*\b(?:dlopen|dlsym|dlclose|dlerror)(?:@|\b)", symbols):
    raise SystemExit("oneTBB binary retains a dynamic-loader symbol")
for artifact in (exception_object, binary):
    artifact_symbols = subprocess.check_output([readelf, "-Ws", artifact], text=True)
    if re.search(r"\b__cxa_get_globals(?:@|\b)", artifact_symbols):
        raise SystemExit(
            "oneTBB compiled the excluded GCC libstdc++ bug 62258 ABI workaround"
        )
PY
if find "${NATIVE_PREFIX}/lib" -maxdepth 1 -type f -name '*.so*' \
  ! -name 'libtbb.so.12.16' -print -quit | grep -q .; then
  printf 'private native prefix contains an unexpected shared library\n' >&2
  exit 4
fi

"${PYTHON}" - \
  "${LOCK_FILE}" \
  "${NATIVE_PREFIX}" \
  "${NANOBIND_WHEEL}" \
  "${LOCKED_READELF}" \
  "${NATIVE_PROVENANCE}" <<'PY'
import hashlib
import json
import pathlib
import re
import subprocess
import sys

lock_path, prefix, nanobind_wheel, readelf, output = map(
    pathlib.Path, sys.argv[1:]
)
lock = json.loads(lock_path.read_text(encoding="utf-8"))

def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build_id(path: pathlib.Path) -> str:
    notes = subprocess.check_output([readelf, "-n", path], text=True)
    match = re.search(r"Build ID: ([0-9a-f]+)", notes)
    if match is None:
        raise SystemExit(f"ELF is missing a GNU build ID: {path}")
    return match.group(1)

artifacts = {
    "onetbb": [{
        "path": "lib/libtbb.so.12.16",
        "sha256": digest(prefix / "lib/libtbb.so.12.16"),
        "gnu_build_id": build_id(prefix / "lib/libtbb.so.12.16"),
        "linkage": "shared",
        "soname": "libtbb.so.12",
    }],
    "c-blosc": [{
        "path": "lib/libblosc.a",
        "sha256": digest(prefix / "lib/libblosc.a"),
        "linkage": "static",
    }],
    "zlib": [{
        "path": "lib/libz.a",
        "sha256": digest(prefix / "lib/libz.a"),
        "linkage": "static",
    }],
    "nanobind": [{
        "path": nanobind_wheel.name,
        "sha256": digest(nanobind_wheel),
        "linkage": "compiled_in",
    }],
}

sources = [lock["source"], *lock["native_dependencies"]]
sources.extend(lock["build_inputs"][name] for name in ("nanobind", "robin-map"))
components = []
for source in sources:
    component = {
        "id": source["id"],
        "version": source["version"],
        "relationship": source["relationship"],
        "license": source["license"],
        "license_files": source.get("license_files", []),
        "artifacts": artifacts.get(source["id"], []),
    }
    if "commit" in source:
        component["source"] = {
            key: source[key]
            for key in (
                "repository",
                "commit",
                "archive_sha256",
                "git_tree_inventory_sha256",
            )
        }
        if source.get("patches"):
            component["source"]["patches"] = source["patches"]
    if source.get("excluded_source_features"):
        component["excluded_source_features"] = source["excluded_source_features"]
    components.append(component)
    for vendored in source.get("vendored_components", []):
        compiled_into = {
            "c-blosc": "lib/libblosc.a",
            "openvdb": "lib/libopenvdb.so.13.0.0",
        }[vendored["containing_source_id"]]
        vendored_component = {
            "id": vendored["id"],
            "version": vendored["version"],
            "relationship": "compiled_in",
            "license": vendored["license"],
            "license_files": vendored["license_files"],
            "containing_source_id": vendored["containing_source_id"],
            "containing_path": vendored["containing_path"],
            "artifacts": [{"compiled_into": compiled_into}],
        }
        if vendored.get("source_files"):
            vendored_component["source_files"] = vendored["source_files"]
        components.append(vendored_component)

document = {
    "schema": "world-understanding.openvdb-native-dependency-provenance.v1",
    "components": sorted(components, key=lambda item: item["id"]),
    "source_lock_sha256": digest(lock_path),
}
output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
cp -- "${NATIVE_PROVENANCE}" "${SOURCE_DIR}/NATIVE_DEPENDENCY_PROVENANCE.json"

# OpenVDB's install step rewrites its build-only RPATH. A GNU build ID is
# calculated before that rewrite and would otherwise retain build-root entropy.
export LDFLAGS="${PINNED_BINUTILS_PREFIX} -fuse-ld=bfd -Wl,--build-id=none"

unset CONDA_PREFIX
export CMAKE_PREFIX_PATH="${NATIVE_PREFIX}"
export TBB_ROOT="${NATIVE_PREFIX}"
export TBB_INCLUDEDIR="${NATIVE_PREFIX}/include"
export TBB_LIBRARYDIR="${NATIVE_PREFIX}/lib"
export Blosc_ROOT="${NATIVE_PREFIX}"
export BLOSC_INCLUDEDIR="${NATIVE_PREFIX}/include"
export BLOSC_LIBRARYDIR="${NATIVE_PREFIX}/lib"
export ZLIB_ROOT="${NATIVE_PREFIX}"

"${UV_BIN}" build --python "${TOOL_PYTHON}" --no-python-downloads --no-sources \
  --no-build-isolation --wheel --out-dir "${RAW_WHEEL_DIR}" "${SOURCE_DIR}"
"${PYTHON}" - "${CMAKE_BUILD_DIR}/CMakeCache.txt" "${NATIVE_PREFIX}" \
  "${LOCKED_CC}" "${LOCKED_CXX}" "${LOCKED_NINJA}" <<'PY'
import pathlib
import sys

cache_path = pathlib.Path(sys.argv[1])
prefix = pathlib.Path(sys.argv[2]).resolve()
expected_cc = pathlib.Path(sys.argv[3]).resolve()
expected_cxx = pathlib.Path(sys.argv[4]).resolve()
expected_ninja = pathlib.Path(sys.argv[5]).resolve()
values = {}
for line in cache_path.read_text(encoding="utf-8").splitlines():
    if not line or line.startswith(("//", "#")) or "=" not in line or ":" not in line:
        continue
    key_type, value = line.split("=", 1)
    key, _ = key_type.split(":", 1)
    values[key] = value

expected_flags = {
    "BLOSC_USE_STATIC_LIBS": "ON",
    "DISABLE_CMAKE_SEARCH_PATHS": "ON",
    "USE_PKGCONFIG": "OFF",
    "ZLIB_USE_STATIC_LIBS": "ON",
}
for key, expected in expected_flags.items():
    if values.get(key) != expected:
        raise SystemExit(f"OpenVDB CMake cache did not lock {key}={expected}")

expected_tools = {
    "CMAKE_C_COMPILER": expected_cc,
    "CMAKE_CXX_COMPILER": expected_cxx,
    "CMAKE_MAKE_PROGRAM": expected_ninja,
}
for key, expected in expected_tools.items():
    actual = pathlib.Path(values.get(key, "missing")).resolve()
    if actual != expected:
        raise SystemExit(f"OpenVDB resolved an unapproved build tool for {key}: {actual}")

resolved = {
    "Tbb_INCLUDE_DIR": prefix / "include",
    "Tbb_tbb_LIBRARY_RELEASE": prefix / "lib/libtbb.so.12.16",
    "Blosc_INCLUDE_DIR": prefix / "include",
    "Blosc_LIBRARY_RELEASE": prefix / "lib/libblosc.a",
    "ZLIB_INCLUDE_DIR": prefix / "include",
}
zlib_library = values.get("ZLIB_LIBRARY_RELEASE", values.get("ZLIB_LIBRARY"))
if zlib_library is None:
    raise SystemExit("OpenVDB CMake cache has no resolved zlib library")
resolved["ZLIB_LIBRARY"] = prefix / "lib/libz.a"
values["ZLIB_LIBRARY"] = zlib_library
for key, expected in resolved.items():
    actual = pathlib.Path(values.get(key, "missing")).resolve()
    if actual != expected.resolve():
        raise SystemExit(f"OpenVDB resolved {key} outside the private source-built prefix: {actual}")
PY
mapfile -t RAW_WHEELS < <(find "${RAW_WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' -print)
if [[ "${#RAW_WHEELS[@]}" -ne 1 ]]; then
  printf 'expected one raw wheel, found %s\n' "${#RAW_WHEELS[@]}" >&2
  exit 4
fi
RAW_WHEEL="${RAW_WHEELS[0]}"
"${TOOL_PYTHON}" - "${RAW_WHEEL}" "${BUILD_ROOT}/raw-wheel-native-closure.json" <<'PY'
import io
import json
import pathlib
import re
import sys
import zipfile

from elftools.elf.elffile import ELFFile

wheel = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
records = []
with zipfile.ZipFile(wheel) as archive:
    for name in sorted(archive.namelist()):
        data = archive.read(name)
        if not data.startswith(b"\x7fELF"):
            continue
        dynamic = ELFFile(io.BytesIO(data)).get_section_by_name(".dynamic")
        if dynamic is None:
            raise SystemExit(f"raw wheel ELF has no dynamic section: {name}")
        needed = sorted(
            str(tag.needed)
            for tag in dynamic.iter_tags()
            if tag.entry.d_tag == "DT_NEEDED"
        )
        records.append({"member": name, "needed": needed})

needed = {item for record in records for item in record["needed"]}
forbidden = sorted(
    item for item in needed
    if re.match(r"^lib(?:blosc|lz4|snappy|zstd|z)[.]so(?:[.]|$)", item)
)
if forbidden:
    raise SystemExit(f"static native dependency leaked into DT_NEEDED: {forbidden}")
if "libtbb.so.12" not in needed:
    raise SystemExit("raw wheel does not link the source-built oneTBB ABI")
if any(re.match(r"^libtbb[.]so", pathlib.PurePosixPath(row["member"]).name) for row in records):
    raise SystemExit("raw wheel unexpectedly bundles oneTBB before auditwheel repair")
output.write_text(
    json.dumps({"schema": "world-understanding.openvdb-raw-elf-closure.v1", "elf": records},
               indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
LD_LIBRARY_PATH="${NATIVE_PREFIX}/lib" \
  "${TOOL_ENV}/bin/auditwheel" show "${RAW_WHEEL}" \
  >"${BUILD_ROOT}/auditwheel-show-raw.txt"

case "${BUILD_ARCH}" in
  x86_64) DEFAULT_POLICY="${LOCKED_LINUX_POLICY}_x86_64" ;;
  aarch64) DEFAULT_POLICY="${LOCKED_LINUX_POLICY}_aarch64" ;;
esac
WHEEL_POLICY="${OPENVDB_RUNTIME_AUDITWHEEL_PLAT:-${DEFAULT_POLICY}}"
LD_LIBRARY_PATH="${NATIVE_PREFIX}/lib" \
  "${TOOL_ENV}/bin/auditwheel" repair --plat "${WHEEL_POLICY}" \
  --wheel-dir "${REPAIRED_WHEEL_DIR}" "${RAW_WHEEL}"
mapfile -t REPAIRED_WHEELS < <(
  find "${REPAIRED_WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' -print
)
if [[ "${#REPAIRED_WHEELS[@]}" -ne 1 ]]; then
  printf 'expected one repaired wheel, found %s\n' "${#REPAIRED_WHEELS[@]}" >&2
  exit 4
fi
FINAL_WHEEL="${REPAIRED_WHEELS[0]}"
REPAIRED=true
"${TOOL_PYTHON}" "${WHEEL_CANONICALIZER}" "${FINAL_WHEEL}" "${SOURCE_DATE_EPOCH}"

"${TOOL_PYTHON}" - \
  "${FINAL_WHEEL}" \
  "${DISTRIBUTION_VERSION}" \
  "${LOCK_FILE}" \
  "${NATIVE_PROVENANCE}" <<'PY'
import email
import hashlib
import io
import json
import pathlib
import re
import sys
import zipfile

from elftools.elf.elffile import ELFFile

wheel = pathlib.Path(sys.argv[1])
expected_version = sys.argv[2]
expected_source_lock = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
expected_provenance = pathlib.Path(sys.argv[4]).read_bytes()
provenance = json.loads(expected_provenance)
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
    metadata = email.message_from_bytes(archive.read(metadata_name))
    if metadata["Name"] != "openvdb" or metadata["Version"] != expected_version:
        raise SystemExit("wheel distribution identity mismatch")
    if metadata.get_all("Requires-Dist", []):
        raise SystemExit("native OpenVDB wheel must not declare Python dependencies")
    modules = [
        name for name in names
        if re.fullmatch(r"openvdb/lib/openvdb[^/]*\.so", name)
    ]
    if len(modules) != 1:
        raise SystemExit(f"expected one official OpenVDB extension, found {modules}")
    versioned_libraries = [
        name for name in names
        if name == "openvdb/lib/libopenvdb.so.13.0.0"
    ]
    if len(versioned_libraries) != 1:
        raise SystemExit("wheel is missing the exact OpenVDB 13 shared library")
    for member in [*modules, *versioned_libraries]:
        elf = ELFFile(io.BytesIO(archive.read(member)))
        if elf.get_section_by_name(".note.gnu.build-id") is not None:
            raise SystemExit(
                f"reproducible OpenVDB runtime member contains a GNU build ID: {member}"
            )
    if "openvdb/_source_lock.json" not in names:
        raise SystemExit("wheel is missing its embedded source lock")
    required_license_files = {
        "bitshuffle-MIT.txt",
        "c-blosc-BSD-3-Clause.txt",
        "fastlz-MIT.txt",
        "libdivsufsort-lite-MIT.txt",
        "lz4-BSD-2-Clause.txt",
        "nanobind-BSD-3-Clause.txt",
        "oneTBB-Apache-2.0.txt",
        "oneTBB-third-party-programs.txt",
        "openexr-half-BSD-3-Clause.txt",
        "robin-map-MIT.txt",
        "zlib-ng-Zlib.txt",
        "zlib.txt",
        "zstd-BSD-3-Clause.txt",
    }
    packaged_license_files = {pathlib.PurePosixPath(name).name for name in names}
    missing_licenses = required_license_files - packaged_license_files
    if missing_licenses:
        raise SystemExit(f"wheel is missing native dependency licenses: {sorted(missing_licenses)}")
    provenance_names = [
        name for name in names
        if pathlib.PurePosixPath(name).name == "NATIVE_DEPENDENCY_PROVENANCE.json"
    ]
    if len(provenance_names) != 1 or archive.read(provenance_names[0]) != expected_provenance:
        raise SystemExit("wheel native dependency provenance mismatch")
    tbb_members = [
        name for name in names
        if re.fullmatch(r"openvdb[.]libs/libtbb-[0-9a-f]{8}[.]so[.]12(?:[.][0-9]+)?", name)
    ]
    if len(tbb_members) != 1:
        raise SystemExit(f"repaired wheel must contain exactly one source-built oneTBB: {tbb_members}")
    tbb_elf = ELFFile(io.BytesIO(archive.read(tbb_members[0])))
    build_id_section = tbb_elf.get_section_by_name(".note.gnu.build-id")
    build_ids = [] if build_id_section is None else [
        note["n_desc"] for note in build_id_section.iter_notes()
        if note["n_type"] == "NT_GNU_BUILD_ID"
    ]
    expected_build_id = next(
        component for component in provenance["components"] if component["id"] == "onetbb"
    )["artifacts"][0]["gnu_build_id"]
    if build_ids != [expected_build_id]:
        raise SystemExit("repaired oneTBB GNU build ID does not match source-built provenance")
    embedded = json.loads(archive.read("openvdb/_source_lock.json"))
    if embedded != expected_source_lock:
        raise SystemExit("embedded OpenVDB source lock does not match the build input")
    if embedded["source"]["commit"] != "7c03e1f084873cd1b3422c7ff7aec6ee681b3b38":
        raise SystemExit("embedded OpenVDB source identity mismatch")
    if any("vdb_print" in name or name.startswith("nanovdb/") for name in names):
        raise SystemExit("wheel contains an unapproved OpenVDB binary or NanoVDB payload")
PY

"${UV_BIN}" venv --python "${PYTHON}" --no-python-downloads --no-project --clear "${SMOKE_ENV}"
SMOKE_PYTHON="${SMOKE_ENV}/bin/python"
"${UV_BIN}" pip install --python "${SMOKE_PYTHON}" --no-cache --no-deps "${FINAL_WHEEL}"
NATIVE_MODULE_PATH="$(
  env -u LD_LIBRARY_PATH -u PYTHONPATH "${SMOKE_PYTHON}" - <<'PY'
import importlib
import sys

if "numpy" in sys.modules:
    raise SystemExit("NumPy was loaded before the dependency-free native smoke")
import openvdb

if "numpy" in sys.modules:
    raise SystemExit("the official OpenVDB wrapper imported NumPy eagerly")
if tuple(openvdb.LIBRARY_VERSION) != (13, 0, 0):
    raise SystemExit(f"unexpected OpenVDB library version: {openvdb.LIBRARY_VERSION!r}")
if not hasattr(openvdb, "tools"):
    raise SystemExit("OpenVDB tools overlay was not exported")
if openvdb.tools.API_VERSION != 3:
    raise SystemExit(f"unexpected OpenVDB tools API version: {openvdb.tools.API_VERSION!r}")
grid = openvdb.FloatGrid()
grid.name = "native-build-smoke"
if grid.name != "native-build-smoke":
    raise SystemExit("OpenVDB grid smoke failed")
native = importlib.import_module("openvdb.lib.openvdb")
print(native.__file__)
PY
)"
env -u LD_LIBRARY_PATH ldd "${NATIVE_MODULE_PATH}" >"${BUILD_ROOT}/native-module-ldd.txt"
FORBIDDEN_RUNTIME_PREFIXES=(
  "${SOURCE_DIR}"
  "${NATIVE_SOURCE_ROOT}"
  "${NATIVE_BUILD_ROOT}"
  "${NATIVE_PREFIX}"
  "${CMAKE_BUILD_DIR}"
  "${TOOL_ENV}"
)
if [[ "${PYTHON_BUILD_PREFIX}" != "/usr" && "${PYTHON_BUILD_PREFIX}" != "/usr/local" ]]; then
  FORBIDDEN_RUNTIME_PREFIXES+=("${PYTHON_BUILD_PREFIX}")
fi
for forbidden_path in "${FORBIDDEN_RUNTIME_PREFIXES[@]}"; do
  if grep -F "${forbidden_path}" "${BUILD_ROOT}/native-module-ldd.txt" >/dev/null; then
    printf 'native wheel resolves a library from a build-only path: %s\n' \
      "${forbidden_path}" >&2
    exit 5
  fi
done

if [[ "${REPAIRED}" == "true" ]]; then
  LICENSE_ARTIFACT_MODE="repaired"
else
  LICENSE_ARTIFACT_MODE="raw"
fi
PREPROMOTION_REPORT="${BUILD_ROOT}/native-prepromotion-verification.json"
"${TOOL_PYTHON}" "${LICENSE_VERIFIER}" \
  --prepromotion-only \
  --policy "${LICENSE_POLICY_FILE}" \
  --source-lock "${LOCK_FILE}" \
  --provenance "${NATIVE_PROVENANCE}" \
  --wheel "${FINAL_WHEEL}" \
  --ldd "${BUILD_ROOT}/native-module-ldd.txt" \
  --artifact-mode "${LICENSE_ARTIFACT_MODE}" \
  --output "${PREPROMOTION_REPORT}"

RELEASE_CANDIDATE_RECORD="${BUILD_ROOT}/native-release-candidate.json"
"${TOOL_PYTHON}" "${RELEASE_CANDIDATE_GENERATOR}" \
  --architecture "${BUILD_ARCH}" \
  --policy "${LICENSE_POLICY_FILE}" \
  --verifier "${LICENSE_VERIFIER}" \
  --source-lock "${LOCK_FILE}" \
  --provenance "${NATIVE_PROVENANCE}" \
  --prepromotion-report "${PREPROMOTION_REPORT}" \
  --wheel "${FINAL_WHEEL}" \
  --ldd "${BUILD_ROOT}/native-module-ldd.txt" \
  --artifact-mode "${LICENSE_ARTIFACT_MODE}" \
  >"${RELEASE_CANDIDATE_RECORD}"

if [[ "${MODE}" == "candidate" ]]; then
  mkdir -p -- "${OUTPUT_DIR}"
  cp -- "${FINAL_WHEEL}" "${OUTPUT_DIR}/$(basename -- "${FINAL_WHEEL}")"
  cp -- "${PREPROMOTION_REPORT}" \
    "${OUTPUT_DIR}/openvdb-native-prepromotion-verification.json"
  cp -- "${RELEASE_CANDIDATE_RECORD}" \
    "${OUTPUT_DIR}/openvdb-native-release-candidate.json"
  cp -- "${NATIVE_PROVENANCE}" \
    "${OUTPUT_DIR}/openvdb-native-dependency-provenance.json"
  cp -- "${BUILD_ROOT}/native-module-ldd.txt" \
    "${OUTPUT_DIR}/openvdb-native-ldd-closure.txt"
  printf 'Built an unadmitted OpenVDB 13 release candidate:\n  %s\n' \
    "${OUTPUT_DIR}/$(basename -- "${FINAL_WHEEL}")"
  printf 'Candidate review record:\n  %s\n' \
    "${OUTPUT_DIR}/openvdb-native-release-candidate.json"
  exit 0
fi

LICENSE_ADMISSION_REPORT="${BUILD_ROOT}/native-license-admission.json"
"${TOOL_PYTHON}" "${LICENSE_VERIFIER}" \
  --policy "${LICENSE_POLICY_FILE}" \
  --release-lock "${RELEASE_LOCK_FILE}" \
  --prepromotion-report "${PREPROMOTION_REPORT}" \
  --source-lock "${LOCK_FILE}" \
  --provenance "${NATIVE_PROVENANCE}" \
  --wheel "${FINAL_WHEEL}" \
  --ldd "${BUILD_ROOT}/native-module-ldd.txt" \
  --artifact-mode "${LICENSE_ARTIFACT_MODE}" \
  --output "${LICENSE_ADMISSION_REPORT}"

mkdir -p -- "${OUTPUT_DIR}"
PUBLISHED_WHEEL="${OUTPUT_DIR}/$(basename -- "${FINAL_WHEEL}")"
PUBLISHED_LICENSE_REPORT="${OUTPUT_DIR}/openvdb-native-license-admission.json"
PUBLISHED_PREPROMOTION_REPORT="${OUTPUT_DIR}/openvdb-native-prepromotion-verification.json"
PUBLISHED_SOURCE_PROVENANCE="${OUTPUT_DIR}/openvdb-native-dependency-provenance.json"
PUBLISHED_LDD_CLOSURE="${OUTPUT_DIR}/openvdb-native-ldd-closure.txt"
PUBLISHED_RELEASE_LOCK="${OUTPUT_DIR}/openvdb-native-release-lock.json"
cp -- "${FINAL_WHEEL}" "${PUBLISHED_WHEEL}"
cp -- "${LICENSE_ADMISSION_REPORT}" "${PUBLISHED_LICENSE_REPORT}"
cp -- "${PREPROMOTION_REPORT}" "${PUBLISHED_PREPROMOTION_REPORT}"
cp -- "${NATIVE_PROVENANCE}" "${PUBLISHED_SOURCE_PROVENANCE}"
cp -- "${BUILD_ROOT}/native-module-ldd.txt" "${PUBLISHED_LDD_CLOSURE}"
cp -- "${RELEASE_LOCK_FILE}" "${PUBLISHED_RELEASE_LOCK}"
"${TOOL_ENV}/bin/auditwheel" show "${PUBLISHED_WHEEL}" >"${BUILD_ROOT}/auditwheel-show-final.txt"

"${PYTHON}" - \
  "${LOCK_FILE}" \
  "${PUBLISHED_WHEEL}" \
  "${OUTPUT_DIR}/openvdb-native-build-manifest.json" \
  "${BUILD_ROOT}/native-module-ldd.txt" \
  "${BUILD_ROOT}/auditwheel-show-final.txt" \
  "${PUBLISHED_LICENSE_REPORT}" \
  "${PUBLISHED_SOURCE_PROVENANCE}" \
  "${BUILD_ROOT}/raw-wheel-native-closure.json" \
  "${NATIVE_MODULE_PATH}" \
  "${REPAIRED}" \
  "${WHEEL_POLICY}" \
  "$(uname -m)" \
  "${LOCKED_CXX}" \
  "${CMAKE_BIN}" \
  "${LOCKED_NINJA}" \
  "${UV_BIN}" <<'PY'
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import zipfile

(
    lock_path,
    wheel_path,
    output_path,
    ldd_path,
    audit_path,
    admission_path,
    provenance_path,
    raw_closure_path,
    module_path,
) = map(
    pathlib.Path, sys.argv[1:10]
)
repaired = sys.argv[10]
wheel_policy = sys.argv[11]
architecture = sys.argv[12]
compiler_path, cmake_path, ninja_path, uv_path = map(pathlib.Path, sys.argv[13:17])

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

with zipfile.ZipFile(wheel_path) as archive:
    native_members = {
        name: digest(archive.read(name))
        for name in sorted(archive.namelist())
        if name.endswith(".so") or ".so." in name
    }

def first_line(*command: object) -> str:
    return subprocess.check_output(command, text=True).splitlines()[0]

source_lock = json.loads(lock_path.read_text(encoding="utf-8"))
license_admission = json.loads(admission_path.read_text(encoding="utf-8"))
dependency_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
if license_admission.get("schema") != source_lock["license_admission"]["verifier"]["report_schema"]:
    raise SystemExit("native license admission report schema mismatch")
if license_admission.get("admission", {}).get("passed") is not True:
    raise SystemExit("native license admission report did not pass")
if license_admission.get("wheel", {}).get("sha256") != digest(wheel_path.read_bytes()):
    raise SystemExit("native license admission report wheel digest mismatch")
if (
    license_admission.get("policy", {}).get("sha256")
    != source_lock["license_admission"]["policy"]["sha256"]
):
    raise SystemExit("native license admission report policy digest mismatch")

manifest = {
    "schema": "world-understanding.openvdb-native-build.v1",
    "source_lock": source_lock,
    "source_lock_sha256": digest(lock_path.read_bytes()),
    "license_admission": license_admission,
    "native_dependency_provenance": dependency_provenance,
    "wheel": {
        "filename": wheel_path.name,
        "sha256": digest(wheel_path.read_bytes()),
        "auditwheel_repaired": repaired == "true",
        "requested_platform_policy": wheel_policy,
        "architecture": architecture,
        "native_members": native_members,
    },
    "verification": {
        "library_version": [13, 0, 0],
        "native_module_path": str(module_path),
        "dynamic_dependencies": ldd_path.read_text(encoding="utf-8").splitlines(),
        "raw_wheel_elf_closure": json.loads(raw_closure_path.read_text(encoding="utf-8")),
        "auditwheel": audit_path.read_text(encoding="utf-8").splitlines(),
    },
    "toolchain": {
        "python": first_line(sys.executable, "--version"),
        "compiler": first_line(compiler_path, "--version"),
        "cmake": first_line(cmake_path, "--version"),
        "ninja": first_line(ninja_path, "--version"),
        "uv": first_line(uv_path, "--version"),
        "platform": platform.platform(),
    },
}
output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

printf 'Built and verified OpenVDB 13 in-process Python wheel:\n  %s\n' "${PUBLISHED_WHEEL}"
printf 'Build manifest:\n  %s\n' "${OUTPUT_DIR}/openvdb-native-build-manifest.json"
printf 'Native license admission:\n  %s\n' "${PUBLISHED_LICENSE_REPORT}"
printf 'Native prepromotion verification:\n  %s\n' "${PUBLISHED_PREPROMOTION_REPORT}"
printf 'Native dependency provenance:\n  %s\n' "${PUBLISHED_SOURCE_PROVENANCE}"
