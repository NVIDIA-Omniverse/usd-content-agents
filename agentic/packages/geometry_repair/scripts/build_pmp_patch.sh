#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PMP_REPOSITORY="${GEOMETRY_REPAIR_PMP_REPOSITORY:-https://github.com/pmp-library/pmp-library.git}"
PMP_COMMIT="2a2ad502743724ba90e09af816364c84032f9015"
PMP_TREE_INVENTORY_SHA256="ec134774c578b2ddd97b620daaa73ee1839ffb78b9d3355050443d8bdb6e5508"
JSON_REPOSITORY="${GEOMETRY_REPAIR_JSON_REPOSITORY:-https://github.com/nlohmann/json.git}"
JSON_COMMIT="9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03"
JSON_TREE_INVENTORY_SHA256="2a9391f4fb9775e87ae994395672831d29d8496026a536031f6deab453cd07a8"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CACHE_ROOT="${GEOMETRY_REPAIR_PMP_BUILD_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/geometry-repair/pmp-patch}"
SOURCE_ROOT="${CACHE_ROOT}/sources"
BUILD_DIR="${CACHE_ROOT}/build"
INSTALL_PREFIX="${GEOMETRY_REPAIR_PMP_INSTALL_PREFIX:-${CACHE_ROOT}/install}"
PMP_SOURCE="${GEOMETRY_REPAIR_PMP_SOURCE_DIR:-${SOURCE_ROOT}/pmp-library}"
JSON_SOURCE="${GEOMETRY_REPAIR_JSON_SOURCE_DIR:-${SOURCE_ROOT}/nlohmann-json}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'missing required command: %s\n' "$1" >&2
    exit 2
  fi
}

clone_at_commit() {
  local repository="$1"
  local commit="$2"
  local destination="$3"
  local freshly_cloned="false"
  if [[ ! -d "${destination}/.git" ]]; then
    if [[ -e "${destination}" ]]; then
      printf 'refusing to replace non-Git source path: %s\n' "${destination}" >&2
      exit 3
    fi
    git clone --filter=blob:none --no-checkout "${repository}" "${destination}"
    freshly_cloned="true"
  fi
  if [[ "$(git -C "${destination}" remote get-url origin)" != "${repository}" ]]; then
    printf 'source remote mismatch for %s\n' "${destination}" >&2
    exit 3
  fi
  if [[ "${freshly_cloned}" == "false" ]] && \
    [[ -n "$(git -C "${destination}" status --porcelain --untracked-files=all)" ]]; then
    printf 'refusing to modify dirty third-party checkout: %s\n' "${destination}" >&2
    exit 3
  fi
  if [[ "${freshly_cloned}" == "false" ]] && \
    [[ "$(git -C "${destination}" rev-parse HEAD)" == "${commit}" ]]; then
    return
  fi
  if git -C "${destination}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "${destination}" checkout --detach "${commit}"
    return
  fi
  git -C "${destination}" fetch origin "${commit}"
  git -C "${destination}" checkout --detach "${commit}"
}

verify_tree_inventory() {
  local checkout="$1"
  local expected="$2"
  local actual
  actual="$(git -C "${checkout}" ls-tree -r -z --full-tree HEAD | sha256sum | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    printf 'source-tree inventory mismatch for %s: expected %s, got %s\n' \
      "${checkout}" "${expected}" "${actual}" >&2
    exit 3
  fi
}

for command in c++ cmake git ninja openssl python3 sha256sum; do
  require_command "${command}"
done

mkdir -p -- "${SOURCE_ROOT}" "${CACHE_ROOT}"
clone_at_commit "${PMP_REPOSITORY}" "${PMP_COMMIT}" "${PMP_SOURCE}"
clone_at_commit "${JSON_REPOSITORY}" "${JSON_COMMIT}" "${JSON_SOURCE}"
verify_tree_inventory "${PMP_SOURCE}" "${PMP_TREE_INVENTORY_SHA256}"
verify_tree_inventory "${JSON_SOURCE}" "${JSON_TREE_INVENTORY_SHA256}"

rm -rf -- "${BUILD_DIR}" "${INSTALL_PREFIX}"
mkdir -p -- "${BUILD_DIR}" "${INSTALL_PREFIX}"
export SOURCE_DATE_EPOCH=1778716800
REPRODUCIBLE_FLAGS="-ffile-prefix-map=${CACHE_ROOT}=. -fdebug-prefix-map=${CACHE_ROOT}=."
cmake \
  -S "${PACKAGE_DIR}/native" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG ${REPRODUCIBLE_FLAGS}" \
  -DGEOMETRY_REPAIR_BUILD_GPL_EVALUATORS=OFF \
  -DGEOMETRY_REPAIR_BUILD_PMP=ON \
  -DGEOMETRY_REPAIR_PMP_SOURCE_DIR="${PMP_SOURCE}" \
  -DGEOMETRY_REPAIR_NLOHMANN_JSON_INCLUDE_DIR="${JSON_SOURCE}/single_include"
cmake --build "${BUILD_DIR}" --target geometry_repair_pmp_patch --parallel 2
cmake --install "${BUILD_DIR}"

BINARY="${INSTALL_PREFIX}/bin/geometry_repair_pmp_patch"
BINARY_SHA256="$(sha256sum "${BINARY}" | awk '{print $1}')"
NOTICE_DIR="${INSTALL_PREFIX}/share/licenses/geometry-repair-pmp-patch"
mkdir -p -- "${NOTICE_DIR}/eigen"
cp "${PMP_SOURCE}/LICENSE.txt" "${NOTICE_DIR}/PMP-LICENSE.txt"
cp "${JSON_SOURCE}/LICENSE.MIT" "${NOTICE_DIR}/NLOHMANN-JSON-LICENSE.MIT"
cp "${PMP_SOURCE}"/external/eigen-3.4.0/COPYING.* "${NOTICE_DIR}/eigen/"
if [[ -f /usr/share/doc/libssl-dev/copyright ]]; then
  cp /usr/share/doc/libssl-dev/copyright "${NOTICE_DIR}/OPENSSL-copyright"
fi

ENVIRONMENT_FILE="${INSTALL_PREFIX}/pmp_patch.env"
cat >"${ENVIRONMENT_FILE}" <<EOF
export GEOMETRY_REPAIR_PMP_EXECUTABLE='${BINARY}'
export GEOMETRY_REPAIR_PMP_EXECUTABLE_SHA256='${BINARY_SHA256}'
EOF

python3 - "${INSTALL_PREFIX}/build_manifest.json" <<PY
import json
import pathlib
import platform
import subprocess
import sys


def package_version(name: str) -> str | None:
    try:
        return subprocess.check_output(
            ["dpkg-query", "-W", "-f=\${Version}", name],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

manifest = {
    "schema_version": "geometry-repair.pmp-native-build.v1",
    "pmp": {
        "repository": "${PMP_REPOSITORY}",
        "commit": "${PMP_COMMIT}",
        "tree_inventory_sha256": "${PMP_TREE_INVENTORY_SHA256}",
        "license": "MIT",
    },
    "nlohmann_json": {
        "repository": "${JSON_REPOSITORY}",
        "commit": "${JSON_COMMIT}",
        "tree_inventory_sha256": "${JSON_TREE_INVENTORY_SHA256}",
        "license": "MIT",
    },
    "openssl": {
        "version": subprocess.check_output(["openssl", "version"], text=True).strip(),
        "libssl_dev_package_version": package_version("libssl-dev"),
        "license": "Apache-2.0",
    },
    "toolchain": {
        "cmake": subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0],
        "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "platform": platform.platform(),
        "source_date_epoch": 1778716800,
    },
    "executable": {
        "path": "${BINARY}",
        "sha256": "${BINARY_SHA256}",
    },
}
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"${BINARY}" --capabilities-json
printf '\nPMP adapter built outside the repository. Activate with:\n  source %q\n' \
  "${ENVIRONMENT_FILE}"
