#!/usr/bin/env bash
set -euo pipefail

cd "$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/agentic"

exec content-workflow-cli physics apply \
  --usd ../apps/physics_agent/data/examples/Tire_B01/tire.usdc \
  --scenario examples/physics/tire-bounce/scenario.yaml \
  --behavior-prompt "Show the tire's drop, visible airborne first rebound, natural tipping motion, and final settling behavior." \
  --additional-instructions-file examples/physics/tire-bounce/physics_guidance.md \
  --refine \
  --tune-engine ovphysx \
  --optimizer botorch \
  --max-trials 30 \
  --max-iterations 12 \
  --collision-approximation convexDecomposition \
  --no-optimize \
  --simulation-engine ovphysx \
  --duration-s 3.0 \
  --sample-fps 30 \
  --drop-height-m 1.0 \
  --child-timeout 7200 \
  --output-dir ../.local-runs/content-workflow-cli/physics-tire-bounce-refine \
  "$@"
