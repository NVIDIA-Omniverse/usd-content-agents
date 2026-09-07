#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
source_scene="$repo_root/.data/examples/scene-material-pass/mini_workcell.usda"
if [[ ! -f "$source_scene" ]]; then
  echo "Missing $source_scene" >&2
  echo "Run agentic/examples/scene/material-pass/fetch.sh first." >&2
  exit 1
fi

cd "$repo_root/agentic"
exec content-workflow-cli scene run \
  --usd "$source_scene" \
  --task material \
  --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --additional-instructions-file examples/scene/material-pass/material_guidance.md \
  "$@"
