#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

cd "$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/agentic"

exec content-workflow-cli physics apply \
  --usd ../apps/physics_agent/data/examples/Lightbulb01/light_bulb_01.usda \
  --output-dir ../.local-runs/content-workflow-cli/physics-lightbulb \
  "$@"
