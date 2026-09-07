#!/usr/bin/env bash
set -euo pipefail

cd "$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/agentic"

exec content-workflow-cli physics apply \
  --usd ../apps/physics_agent/data/examples/Container_Gray_C04/container.usdc \
  --scenario examples/physics/container-slide/scenario.yaml \
  --behavior-prompt "Make this closed plastic warehouse container slide realistically across a flat dry industrial floor after a gentle horizontal push, decelerating smoothly and coming naturally to rest while its lid and body remain together." \
  --additional-instructions-file examples/physics/container-slide/physics_guidance.md \
  --refine \
  --tune-engine ovphysx \
  --optimizer botorch \
  --max-trials 8 \
  --max-iterations 3 \
  --collision-approximation convexHull \
  --no-optimize \
  --simulation-engine ovphysx \
  --duration-s 3.0 \
  --sample-fps 24 \
  --child-timeout 7200 \
  --output-dir ../.local-runs/content-workflow-cli/physics-container-slide-refine \
  "$@"
