#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Set up the host content-workflow-cli CLI and Content Workbench sidecar.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKIP_BUILD_RESOURCES=0
RECREATE_VENV=0

usage() {
    cat <<'EOF'
Usage: scripts/setup_content_agent.sh [options]

Options:
  --skip-build-resources     Do not fetch Scene Optimizer build resources
  --recreate-venv            Delete and recreate .venv with Python 3.12
  -h, --help                 Show this help

Notes:
  - Run from Linux, WSL2, or macOS.
  - Local Workbench rendering requires a host with an NVIDIA GPU runtime.
  - Native macOS can run content-workflow-cli against a remote Workbench URL, but local
    OvRTX rendering requires a Linux NVIDIA GPU host.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build-resources)
            SKIP_BUILD_RESOURCES=1
            shift
            ;;
        --recreate-venv)
            RECREATE_VENV=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $1" >&2
        exit 1
    fi
}

require_command uv
require_command node
require_command npm

cd "$REPO_ROOT"

if [[ -d "$REPO_ROOT/.venv" && "$RECREATE_VENV" -eq 0 ]]; then
    echo "Reusing existing .venv. Pass --recreate-venv to rebuild it with Python 3.12."
elif [[ -d "$REPO_ROOT/.venv" ]]; then
    rm -rf "$REPO_ROOT/.venv"
    uv venv --python=3.12
else
    uv venv --python=3.12
fi
uv pip install --python "$REPO_ROOT/.venv/bin/python" -e "$REPO_ROOT/apps/material_agent[all]"
uv pip install --python "$REPO_ROOT/.venv/bin/python" -e "$REPO_ROOT/agentic/packages/content_workflow_cli"
npm ci --prefix "$REPO_ROOT/agentic/packages/content_workflow_cli"
WORKBENCH_INSTALLED=0
if [[ "$(uname -s)" == "Linux" ]]; then
    WORKBENCH_INSTALLED=1
    uv pip install --python "$REPO_ROOT/.venv/bin/python" -e "$REPO_ROOT/agentic/packages/content_workbench"
else
    echo "Skipping local Workbench install: host OvRTX rendering requires Linux with an NVIDIA GPU runtime."
fi

if [[ "$SKIP_BUILD_RESOURCES" -eq 0 ]]; then
    "$REPO_ROOT/scripts/fetch_build_resources.sh"
fi

cat <<EOF

content-workflow-cli setup complete.

Activate the environment:
  source "$REPO_ROOT/.venv/bin/activate"

Verify Codex auth, using ChatGPT/OAuth if that is your normal Codex login:
  content-workflow-cli auth login

For headless hosts:
  content-workflow-cli auth login --device-code
EOF

if [[ "$WORKBENCH_INSTALLED" -eq 1 ]]; then
    cat <<EOF
Start the local Workbench sidecar manually on a Linux NVIDIA GPU host:
  content-workbench

Or let content-workflow-cli start it for localhost runs:
  content-workflow-cli materials assign ...

Check Workbench:
  curl http://127.0.0.1:8088/healthz
EOF
else
    cat <<EOF
Run content-workflow-cli against an existing Workbench endpoint:
  content-workflow-cli materials assign ... --workbench-url http://workbench-host:8088

Local Workbench rendering requires a Linux NVIDIA GPU host.
EOF
fi
