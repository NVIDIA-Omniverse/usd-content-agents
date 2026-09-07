#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
output_dir="${CONTENT_WORKFLOW_OUTPUT_DIR:-$repo_root/runs/validation-smoke-bracket}"
if [[ "$output_dir" != /* ]]; then
  output_dir="$repo_root/$output_dir"
fi
model="${CONTENT_WORKFLOW_MODEL:-gpt-5.6-sol}"
render_backend="${CONTENT_WORKFLOW_RENDER_BACKEND:-ovrtx}"
case "$render_backend" in
  ovrtx | remote) ;;
  *)
    echo "CONTENT_WORKFLOW_RENDER_BACKEND must be 'ovrtx' or 'remote'." >&2
    exit 2
    ;;
esac

cd "$repo_root/agentic"

exec content-workflow-cli validate run \
  --usd "$repo_root/agentic/examples/geometry/quickstart/smoke_bracket.usda" \
  --task "Validate static USD integrity and current OVRTX rendering for release readiness. Do not modify the asset; report evidence, limitations, and recommended actions." \
  --output-dir "$output_dir" \
  --runner codex \
  --model "$model" \
  --render-backend "$render_backend" \
  "$@"
