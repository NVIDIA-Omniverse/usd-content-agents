#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
grip_prim=/RootNode/Geometry/M_AluminumStepLadder_B01_Plastic2
requested_appearance="smooth high-visibility safety orange rubberized plastic with an even single-color matte finish"
output_dir="${CONTENT_WORKFLOW_OUTPUT_DIR:-$repo_root/runs/texture-ladder}"
if [[ "$output_dir" != /* ]]; then
  output_dir="$repo_root/$output_dir"
fi

cd "$repo_root/agentic"

set +e
content-workflow-cli texture run \
  --usd ../apps/texture_agent/data/examples/ladder/sources/usd/ladder_uv_ready.usd \
  --prompt "$requested_appearance" \
  --prim-path "$grip_prim" \
  --unit-action "$grip_prim=generate" \
  --unit-appearance "$grip_prim=$requested_appearance" \
  --output-dir "$output_dir" \
  --runner codex \
  "$@"
status=$?
set -e

if [[ $status -eq 3 && -f "$output_dir/texture_companion_generation_handoff.json" ]]; then
  printf '%s\n' \
    "Texture paused for the outer coding-agent companion." \
    "Read $output_dir/texture_companion_generation_handoff.json," \
    "write the exact requested result manifest, then resume with:" \
    "content-workflow-cli texture resume --run-dir \"$output_dir\"" >&2
fi

exit "$status"
