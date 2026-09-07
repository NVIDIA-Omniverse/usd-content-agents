#!/usr/bin/env bash
set -euo pipefail

cd "$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/agentic"

exec content-workflow-cli physics apply \
  --usd ../apps/physics_agent/data/examples/Lightbulb01/light_bulb_01.usda \
  --output-dir ../.local-runs/content-workflow-cli/physics-lightbulb \
  "$@"
