#!/usr/bin/env bash
# Pinned permissive repair dependency only. Build does not approve task results.
set -euo pipefail
RERUN_ROOT="$(cd "$1" && pwd)"
GEO_SOURCE="$RERUN_ROOT/resources/geogram-source"
GEO_BUILD="$RERUN_ROOT/resources/geogram-build"
GEO_OUT="$RERUN_ROOT/resources/geogram"
GEO_COMMIT=c8529bb00838186938ab31d96008a59b6a892dee
export TMPDIR="$RERUN_ROOT/tmp"
export CMAKE_BUILD_PARALLEL_LEVEL=2
if [[ ! -d "$GEO_SOURCE/.git" ]]; then
  git clone --no-checkout https://github.com/BrunoLevy/geogram.git "$GEO_SOURCE"
  git -C "$GEO_SOURCE" checkout --detach "$GEO_COMMIT"
  git -C "$GEO_SOURCE" submodule update --init --recursive
fi
[[ "$(git -C "$GEO_SOURCE" rev-parse HEAD)" == "$GEO_COMMIT" ]]
PATCH_FILE="$RERUN_ROOT/repo/agentic/packages/geometry_repair/native/geogram-1.10.0-no-tbb.patch"
if git -C "$GEO_SOURCE" apply --check "$PATCH_FILE"; then
 git -C "$GEO_SOURCE" apply "$PATCH_FILE"
else
 git -C "$GEO_SOURCE" apply --reverse --check "$PATCH_FILE"
fi
cmake -S "$GEO_SOURCE" -B "$GEO_BUILD" -DCMAKE_BUILD_TYPE=Release \
 -DVORPALINE_PLATFORM=Linux64-gcc-dynamic -DCMAKE_INSTALL_PREFIX="$GEO_OUT" \
 -DGEOGRAM_WITH_GRAPHICS=OFF -DGEOGRAM_WITH_LUA=OFF -DGEOGRAM_WITH_TETGEN=OFF \
 -DGEOGRAM_WITH_TRIANGLE=OFF -DGEOGRAM_WITH_HLBFGS=OFF \
 -DGEOGRAM_WITH_LEGACY_NUMERICS=OFF -DGEOGRAM_WITH_TBB=OFF
cmake --build "$GEO_BUILD" --target vorpalite --parallel 2
mkdir -p "$GEO_OUT/bin" "$GEO_OUT/lib" "$GEO_OUT/share/licenses"
cp "$GEO_BUILD/bin/vorpalite" "$GEO_OUT/bin/vorpalite"
cp -a "$GEO_BUILD/lib/"libgeogram.so* "$GEO_OUT/lib/"
cp "$GEO_SOURCE/LICENSE" "$GEO_OUT/share/licenses/GEOGRAM_LICENSE"
cp "$RERUN_ROOT/repo/agentic/packages/geometry_repair/geometry_repair/GEOGRAM_THIRD_PARTY_NOTICES.md" "$GEO_OUT/share/licenses/"
git -C "$GEO_SOURCE" submodule status --recursive > "$GEO_OUT/share/licenses/submodule_commits.txt"
printf '%s\n' "$GEO_COMMIT" > "$GEO_OUT/share/licenses/source_commit.txt"
LD_LIBRARY_PATH="$GEO_OUT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$GEO_OUT/bin/vorpalite" --version
sha256sum "$GEO_OUT/bin/vorpalite" > "$RERUN_ROOT/environment/evidence/geogram_executable_unapproved.sha256"
printf '%s\n' 'Build only. Root must verify source/patch/submodules and approve this exact binary SHA before setting GEOMETRY_REPAIR_GEOGRAM_EXECUTABLE_SHA256.'
