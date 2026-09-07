#!/usr/bin/env bash
set -euo pipefail

cd "$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/agentic"

exec content-workflow-cli mesh-segmentation run \
  --asset examples/mesh-segmentation/basic/fused_cart.usda \
  --target-prim /FusedCart/Geometry \
  --target-semantic-part body \
  --target-semantic-part wheel \
  --allow-codex-configured-auth \
  --no-memory \
  "$@"
