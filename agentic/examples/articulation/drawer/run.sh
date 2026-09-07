#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
output_dir="${CONTENT_WORKFLOW_OUTPUT_DIR:-$repo_root/runs/articulation-mini-drawer}"
if [[ "$output_dir" != /* ]]; then
  output_dir="$repo_root/$output_dir"
fi

cd "$repo_root"

exec content-workflow-cli asset run \
  --usd "$repo_root/agentic/examples/articulation/drawer/drawer.usda" \
  --output-dir "$output_dir" \
  --prompt "Use the sole long-running coordinator to author exactly one prismatic joint that moves /MiniCabinet/Drawer along the Y axis relative to /MiniCabinet/Frame. Use deterministic scene evidence by default; do not select an external articulation proposal provider. Do not author masses or colliders, and preserve all non-target topology." \
  --required-leaf articulation.author.v1 \
  --required-leaf articulation.evidence.v1 \
  --required-leaf articulation.preparation-publisher.v1 \
  --required-leaf articulation.publish.v1 \
  --required-leaf articulation.review.v1 \
  --required-leaf validation.canonical-ovrtx-evidence.v1 \
  --required-terminal-leaf articulation.publish.v1 \
  --required-terminal-leaf validation.canonical-ovrtx-evidence.v1 \
  --required-leaf-dependency articulation.author.v1=articulation.preparation-publisher.v1 \
  --required-leaf-dependency articulation.evidence.v1=articulation.author.v1 \
  --required-leaf-dependency articulation.review.v1=articulation.evidence.v1 \
  --required-leaf-dependency articulation.publish.v1=articulation.review.v1 \
  --required-leaf-dependency validation.canonical-ovrtx-evidence.v1=articulation.author.v1 \
  --exact-leaf-scope \
  --runner codex \
  "$@"
