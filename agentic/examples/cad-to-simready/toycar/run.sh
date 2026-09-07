#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
source_asset="$repo_root/.data/examples/toycar/ToyCar.glb"
if [[ ! -f "$source_asset" ]]; then
  echo "Missing $source_asset" >&2
  echo "Run agentic/examples/cad-to-simready/toycar/fetch.sh first." >&2
  exit 1
fi

cd "$repo_root/agentic"
exec content-workflow-cli cad-to-simready run "$source_asset" \
  --asset-id toycar \
  --output-dir ../.local-runs/content-workflow-cli/cad-to-simready-toycar \
  --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml \
  "$@"
