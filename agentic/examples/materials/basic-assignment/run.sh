#!/usr/bin/env bash
set -euo pipefail

cd "$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/agentic"

exec content-workflow-cli materials assign \
  --usd ../apps/material_agent/data/examples/ladder/sources/usd/ladder.usd \
  --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg \
  --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg \
  --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --additional-instructions-file examples/materials/basic-assignment/material_guidance.md \
  "$@"
