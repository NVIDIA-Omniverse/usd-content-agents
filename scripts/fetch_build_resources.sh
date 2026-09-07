#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fetch build resources that are not on public PyPI.
#
# Downloads the public `scene_optimizer_core_usd_25.11_py_3.12` zip from the
# NVIDIA-Omniverse/usd-optimize GitHub release and unpacks it into
# `.build-resources/scene_optimizer_core/`.
# The Dockerfiles mount that directory into the image and point
# `WU_SO_PACKAGE_DIR` at it so the `optimize_usd` step has a working local
# backend. If the directory is absent, the Dockerfiles skip the setup and
# `optimize_usd` falls back to the NVCF cloud backend (or fails if neither is
# configured).
#
# Idempotent: exits early if the unpacked package is already present.
#
# Override the default package version/platform by setting `SO_CORE_URL`:
#   SO_CORE_URL=https://.../scene_optimizer_core_*.zip ./scripts/fetch_build_resources.sh
# Override architecture detection by setting `SO_CORE_ARCH` to `x86_64` or
# `aarch64`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_RESOURCES="${SO_CORE_BUILD_RESOURCES:-$REPO_ROOT/.build-resources}"
PKG_DIR="$BUILD_RESOURCES/scene_optimizer_core"

SO_CORE_ARCH="${SO_CORE_ARCH:-$(uname -m)}"
# Scene Optimizer Core publishes a windows-x86_64 package beside the manylinux
# ones. Detect a Windows shell (Git Bash, MSYS2, Cygwin) so the release asset
# matches the host instead of resolving to a Linux package that cannot load.
SO_CORE_OS="${SO_CORE_OS:-$(uname -s)}"
case "$SO_CORE_OS" in
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        SO_CORE_HOST_OS="windows"
        ;;
    *)
        SO_CORE_HOST_OS="linux"
        ;;
esac

case "$SO_CORE_ARCH" in
    x86_64|amd64)
        if [[ "$SO_CORE_HOST_OS" == "windows" ]]; then
            SO_CORE_PLATFORM="windows-x86_64"
        else
            SO_CORE_PLATFORM="manylinux_2_35_x86_64"
        fi
        ;;
    aarch64|arm64)
        if [[ "$SO_CORE_HOST_OS" == "windows" ]]; then
            # The override has to be honoured here too, or the message below
            # advertises a way out that this branch never reaches.
            if [[ -z "${SO_CORE_URL:-}" ]]; then
                echo "ERROR: Scene Optimizer Core publishes no windows-aarch64 package" >&2
                echo "       Set SO_CORE_URL to an explicit package URL to override." >&2
                exit 1
            fi
            SO_CORE_PLATFORM="custom"
        else
            SO_CORE_PLATFORM="manylinux_2_35_aarch64"
        fi
        ;;
    *)
        if [[ -z "${SO_CORE_URL:-}" ]]; then
            echo "ERROR: unsupported Scene Optimizer Core architecture: $SO_CORE_ARCH" >&2
            echo "       Set SO_CORE_URL to an explicit package URL to override." >&2
            exit 1
        fi
        SO_CORE_PLATFORM="custom"
        ;;
esac

# Scene Optimizer Core v1.0.3.
DEFAULT_URL="https://github.com/NVIDIA-Omniverse/usd-optimize/releases/download/v1.0.3/scene_optimizer_core_usd_25.11_py_3.12%401.0.3.1-0-3.506.5ccdcb0b.gl.${SO_CORE_PLATFORM}.release.zip"
URL="${SO_CORE_URL:-$DEFAULT_URL}"
URL_SHA256="$(printf "%s" "$URL" | sha256sum | awk '{print $1}')"
# curl uses exponential backoff when no explicit --retry-delay is supplied.
# The overrides keep failure-path tests fast without weakening production defaults.
SO_CORE_CURL_RETRIES="${SO_CORE_CURL_RETRIES:-5}"
SO_CORE_CURL_CONNECT_TIMEOUT="${SO_CORE_CURL_CONNECT_TIMEOUT:-15}"
SO_CORE_CURL_RETRY_MAX_TIME="${SO_CORE_CURL_RETRY_MAX_TIME:-120}"
SO_CORE_CURL_MAX_TIME="${SO_CORE_CURL_MAX_TIME:-600}"
SO_CORE_CURL_SPEED_LIMIT="${SO_CORE_CURL_SPEED_LIMIT:-1024}"
SO_CORE_CURL_SPEED_TIME="${SO_CORE_CURL_SPEED_TIME:-60}"

validate_nonnegative_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $name must be a non-negative integer, got: $value" >&2
        exit 1
    fi
}

validate_positive_integer() {
    local name="$1"
    local value="$2"
    validate_nonnegative_integer "$name" "$value"
    if [[ "$value" == "0" ]]; then
        echo "ERROR: $name must be greater than zero" >&2
        exit 1
    fi
}

validate_nonnegative_integer SO_CORE_CURL_RETRIES "$SO_CORE_CURL_RETRIES"
validate_positive_integer SO_CORE_CURL_CONNECT_TIMEOUT "$SO_CORE_CURL_CONNECT_TIMEOUT"
validate_positive_integer SO_CORE_CURL_RETRY_MAX_TIME "$SO_CORE_CURL_RETRY_MAX_TIME"
validate_positive_integer SO_CORE_CURL_MAX_TIME "$SO_CORE_CURL_MAX_TIME"
validate_positive_integer SO_CORE_CURL_SPEED_LIMIT "$SO_CORE_CURL_SPEED_LIMIT"
validate_positive_integer SO_CORE_CURL_SPEED_TIME "$SO_CORE_CURL_SPEED_TIME"

if [[ "${SO_CORE_PRINT_URL_ONLY:-}" == "1" || "${SO_CORE_PRINT_URL_ONLY:-}" == "true" ]]; then
    echo "$URL"
    exit 0
fi

mkdir -p "$BUILD_RESOURCES"

has_expected_layout_at() {
    local dir="$1"
    [[ -d "$dir/python" && -d "$dir/lib" && -d "$dir/extraLibs" && -d "$dir/usdpy" ]]
}

has_expected_layout() {
    has_expected_layout_at "$PKG_DIR"
}

package_matches_platform() {
    local marker="$PKG_DIR/.so_core_platform"
    [[ -f "$marker" ]] || return 1
    grep -Fxq "platform=$SO_CORE_PLATFORM" "$marker" \
        && grep -Fxq "url_sha256=$URL_SHA256" "$marker"
}

# Skip if already unpacked. The `usdpy` dir is a good sentinel — it only
# exists in the public package layout (vs. legacy internal bundle layouts).
if has_expected_layout; then
    if package_matches_platform; then
        echo "scene_optimizer_core already unpacked at $PKG_DIR for $SO_CORE_PLATFORM — skipping fetch"
        exit 0
    fi
    echo "Existing scene_optimizer_core at $PKG_DIR does not match $SO_CORE_PLATFORM; refetching"
fi

TMP_DIR=""
BACKUP_DIR=""
cleanup() {
    if [[ -n "${TMP_DIR:-}" ]]; then
        rm -rf "$TMP_DIR"
    fi
}
trap cleanup EXIT

TMP_DIR="$(mktemp -d "$BUILD_RESOURCES/scene_optimizer_core.tmp.XXXXXX")"
UNPACK_DIR="$TMP_DIR/scene_optimizer_core"
ZIP_PATH="$TMP_DIR/scene_optimizer_core.zip"
echo "Fetching Scene Optimizer Core from:"
echo "  $URL"
CURL_HELP_ALL="$(curl --help all 2>/dev/null || true)"
CURL_RETRY_ARGS=(--retry "$SO_CORE_CURL_RETRIES")
if grep -Fq -- "--retry-connrefused" <<< "$CURL_HELP_ALL"; then
    CURL_RETRY_ARGS+=(--retry-connrefused)
fi
if grep -Fq -- "--retry-all-errors" <<< "$CURL_HELP_ALL"; then
    CURL_RETRY_ARGS+=(--retry-all-errors)
else
    echo "curl does not support --retry-all-errors; using compatible retries" >&2
fi
curl \
    --fail \
    --location \
    --silent \
    --show-error \
    "${CURL_RETRY_ARGS[@]}" \
    --connect-timeout "$SO_CORE_CURL_CONNECT_TIMEOUT" \
    --retry-max-time "$SO_CORE_CURL_RETRY_MAX_TIME" \
    --max-time "$SO_CORE_CURL_MAX_TIME" \
    --speed-limit "$SO_CORE_CURL_SPEED_LIMIT" \
    --speed-time "$SO_CORE_CURL_SPEED_TIME" \
    --output "$ZIP_PATH" \
    "$URL"

echo "Unpacking into temporary directory ..."
mkdir -p "$UNPACK_DIR"
unzip -q "$ZIP_PATH" -d "$UNPACK_DIR"
rm -f "$ZIP_PATH"

# Sanity check: make sure the expected layout is in place.
for sub in python lib extraLibs usdpy; do
    if [[ ! -d "$UNPACK_DIR/$sub" ]]; then
        echo "ERROR: unpacked package missing expected subdir: $UNPACK_DIR/$sub" >&2
        echo "       (check that SO_CORE_URL points at a Scene Optimizer Core zip)" >&2
        exit 1
    fi
done

{
    echo "platform=$SO_CORE_PLATFORM"
    echo "url_sha256=$URL_SHA256"
} > "$UNPACK_DIR/.so_core_platform"

if ! has_expected_layout_at "$UNPACK_DIR"; then
    echo "ERROR: unpacked package failed final layout validation: $UNPACK_DIR" >&2
    exit 1
fi

echo "Installing into $PKG_DIR ..."
if [[ -e "$PKG_DIR" ]]; then
    BACKUP_DIR="$(mktemp -d "$BUILD_RESOURCES/scene_optimizer_core.backup.XXXXXX")"
    rmdir "$BACKUP_DIR"
    if ! mv "$PKG_DIR" "$BACKUP_DIR"; then
        echo "ERROR: failed to move existing Scene Optimizer Core package aside" >&2
        BACKUP_DIR=""
        exit 1
    fi
fi

# Renaming a freshly unpacked tree can lose a race with an on-access scanner
# still holding the new binaries, which surfaces as a transient EACCES. Retry a
# few times before treating the install as failed.
install_unpacked_package() {
    local attempt
    for attempt in 1 2 3 4 5; do
        # mv moves the source INTO the destination when the destination is an
        # existing directory. A retry after a partial install would then create
        # $PKG_DIR/scene_optimizer_core, report success, and leave the platform
        # marker unfindable.
        if [[ -n "$PKG_DIR" && -e "$PKG_DIR" ]]; then
            rm -rf "$PKG_DIR"
        fi
        if mv "$UNPACK_DIR" "$PKG_DIR"; then
            return 0
        fi
        if [[ "$attempt" -lt 5 ]]; then
            echo "Install attempt $attempt failed; retrying ..." >&2
            sleep "$attempt"
        fi
    done
    return 1
}

if ! install_unpacked_package; then
    echo "ERROR: failed to install Scene Optimizer Core into $PKG_DIR" >&2
    if [[ -n "${BACKUP_DIR:-}" && -e "$BACKUP_DIR" ]]; then
        if mv "$BACKUP_DIR" "$PKG_DIR"; then
            BACKUP_DIR=""
        else
            echo "ERROR: failed to restore previous Scene Optimizer Core package" >&2
            echo "       backup preserved at: $BACKUP_DIR" >&2
            BACKUP_DIR=""
        fi
    fi
    exit 1
fi

if [[ -n "${BACKUP_DIR:-}" ]]; then
    rm -rf "$BACKUP_DIR"
    BACKUP_DIR=""
fi

du -sh "$PKG_DIR"
echo "Done. You can now run: docker compose build"
