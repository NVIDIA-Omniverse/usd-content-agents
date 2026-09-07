#!/usr/bin/env bash
set -euo pipefail

cd "$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/agentic"

asset=../apps/usd_cli/sample_assets/simready/Cube/cube.usda
output_dir="${CONTENT_WORKFLOW_OUTPUT_DIR:-../.local-runs/content-workflow-cli/simready-cube}"
mkdir -p "$output_dir"

content-workflow-cli simready validate-profile "$asset" \
  --profile Package \
  --report "$output_dir/validation.json" --strict
exec content-workflow-cli simready conform-profile "$asset" \
  --output-dir "$output_dir/conformed" \
  --profile Package \
  --validation-report "$output_dir/validation.json" --strict "$@"
