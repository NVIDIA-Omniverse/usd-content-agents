#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Set up the host content-workflow-cli CLI and canonical usd-cli scene tool.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKIP_BUILD_RESOURCES=0
INSTALL_LIVE_VIEW=0
RECREATE_VENV=0
INSTALL_CHILD_RUNNERS=1

usage() {
    cat <<'EOF'
Usage: scripts/setup_content_agent.sh [options]

Options:
  --skip-build-resources     Do not fetch Scene Optimizer build resources
  --without-child-runners    Skip Node/npm SDK setup for direct-only workflows
  --recreate-venv            Delete and recreate .venv with Python 3.12
  -h, --help                 Show this help
EOF
    if [[ -x "$REPO_ROOT/agentic/packages/content_workflow_viewer/scripts/setup.sh" ]]; then
        cat <<'EOF'
  --live-view                Install the optional OvRTX/WebRTC viewer
EOF
    fi
    cat <<'EOF'

Notes:
  - Invoke this script from the repository root on Linux or WSL2.
  - Native Windows execution is unsupported in the 0.6 release. On a Windows
    host, run the supported workflow inside WSL2; scripts/setup_content_agent.ps1
    is retained for development-only setup and future qualification.
  - This installs material-agent directly and installs physics-agent,
    joint-agent, and texture-agent through content-agent-workflows. Using those
    fixed-pipeline interfaces remains opt-in; validation-agent is not installed.
  - Native B-rep repair is not included. CAD providers should also return a
    validated USD or mesh representation for the Geometry workflow.
  - Node.js and npm are required only for workflows that launch Codex or Claude.
  - jq is also required for those child-runner workflows.
  - Local OVRTX rendering requires a native Linux NVIDIA GPU/Vulkan host. It is
    unavailable under WSL2; configure remote OVRTX there instead.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build-resources)
            SKIP_BUILD_RESOURCES=1
            shift
            ;;
        --live-view)
            INSTALL_LIVE_VIEW=1
            shift
            ;;
        --without-child-runners)
            INSTALL_CHILD_RUNNERS=0
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

# Reject missing prerequisites before setup can create or modify the environment.
require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $1" >&2
        exit 1
    fi
}

# Fail before setup mutates the environment when build-resource extraction is unavailable.
require_build_resource_unzip() {
    if ! command -v unzip >/dev/null 2>&1; then
        echo "ERROR: required command not found: unzip" >&2
        echo "       Install it on Ubuntu/WSL2 with:" >&2
        echo "         sudo apt-get update && sudo apt-get install -y unzip" >&2
        echo "       Or rerun with --skip-build-resources when Scene Optimizer" >&2
        echo "       build resources are not needed." >&2
        exit 1
    fi
}

# Enforce the child-runner Node.js baseline after the executable preflight.
require_node_20() {
    local reported major
    reported="$(node --version 2>/dev/null || true)"
    major="${reported#v}"
    major="${major%%.*}"
    if [[ ! "$major" =~ ^[0-9]+$ ]] || (( major < 20 )); then
        echo "ERROR: Node.js 20 or newer is required for the Codex and Claude child" >&2
        echo "       runners (found: ${reported:-unknown})." >&2
        echo "       Install Node 20+, or rerun with --without-child-runners to set" >&2
        echo "       up direct Python workflows only." >&2
        exit 1
    fi
}

require_command uv
if [[ "$SKIP_BUILD_RESOURCES" -eq 0 ]]; then
    require_build_resource_unzip
fi
if [[ "$INSTALL_CHILD_RUNNERS" -eq 1 ]]; then
    require_command node
    require_command npm
    require_command jq
    require_node_20
    if [[ "$(uname -s)" == "Linux" ]]; then
        require_command bwrap
        bwrap_probe_executable="$(type -P true || true)"
        if [[ -z "$bwrap_probe_executable" || ! -x "$bwrap_probe_executable" ]]; then
            echo "ERROR: unable to resolve an external true executable for the bwrap smoke test." >&2
            exit 1
        fi
        if ! bwrap --ro-bind / / --dev /dev --proc /proc -- "$bwrap_probe_executable"; then
            echo "ERROR: bwrap cannot create an unprivileged user-namespace sandbox." >&2
            echo "Enable unprivileged user namespaces and allow bubblewrap in the host/container policy." >&2
            exit 1
        fi
    fi
fi

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
uv pip install --python "$REPO_ROOT/.venv/bin/python" \
    -e "$REPO_ROOT/apps/usd_cli[cli,server]" \
    --overrides "apps/usd_cli/requirements/usd-exchange-override.txt"
uv pip install --python "$REPO_ROOT/.venv/bin/python" \
    -e "$REPO_ROOT/agentic/packages/content_workflow_cli" \
    --overrides "apps/usd_cli/requirements/usd-exchange-override.txt"
if [[ "$INSTALL_CHILD_RUNNERS" -eq 1 ]]; then
    npm ci --prefix "$REPO_ROOT/agentic/packages/content_workflow_cli"
else
    echo "Skipping Node SDK setup. Child-agent Material, Physics, Scene, and composed-asset runs will not be available."
fi
if [[ "$(uname -s)" == "Linux" && "$INSTALL_LIVE_VIEW" -eq 1 ]]; then
    viewer_setup="$REPO_ROOT/agentic/packages/content_workflow_viewer/scripts/setup.sh"
    if [[ ! -x "$viewer_setup" ]]; then
        echo "ERROR: optional live-view package is not available in this checkout." >&2
        exit 1
    fi
    "$viewer_setup"
    viewer_python="$REPO_ROOT/agentic/packages/content_workflow_viewer/.venv/bin/python"
    viewer_frontend="$REPO_ROOT/agentic/packages/content_workflow_viewer/frontend/dist/index.html"
    "$viewer_python" -c \
        'import fastapi, moderngl, numpy, ovstage, ovstream, PIL, uvicorn, warp; import content_workflow_viewer'
    if [[ ! -f "$viewer_frontend" ]]; then
        echo "ERROR: live-view setup did not produce frontend/dist/index.html." >&2
        exit 1
    fi
fi

if [[ "$SKIP_BUILD_RESOURCES" -eq 0 ]]; then
    "$REPO_ROOT/scripts/fetch_build_resources.sh"
fi

cat <<EOF

content-workflow-cli setup complete.

Activate the environment:
  source "$REPO_ROOT/.venv/bin/activate"
EOF

cat <<EOF
Inspect or run the deterministic Geometry workflow (no model login required):
  content-workflow-cli geometry run --help

Provision OVRTX before accepted Geometry evidence. See:
  $REPO_ROOT/agentic/docs/geometry_quickstart.md
EOF

if [[ "$INSTALL_CHILD_RUNNERS" -eq 1 ]]; then
    cat <<'EOF'
For model-backed workflows, verify Codex auth using ChatGPT/OAuth if that is
your normal Codex login:
  content-workflow-cli auth login
  content-workflow-cli auth status --sandbox-smoke

On Linux, `auth status --sandbox-smoke` proves both the direct Codex CLI and
the product SDK bridge can execute a workspace-write command. Codex and Claude
both require `bubblewrap` (`bwrap`) plus unprivileged user namespaces; Codex
has no unconfined fallback.

For headless hosts:
  content-workflow-cli auth login --device-code
EOF
else
    cat <<EOF
This environment is configured for direct Python workflows only. Supported
examples include conversion, Texture, Validation, and SimReady commands. Rerun
without --without-child-runners before launching Articulation or another Codex-
or Claude-authored workflow.
EOF
fi

cat <<EOF
Write user-facing run artifacts under:
  $REPO_ROOT/runs/

material-agent, physics-agent, joint-agent, and texture-agent are installed.
Their standalone fixed-pipeline interfaces remain opt-in. Add the Warp
dependency closure for supported local WSL2 fixed-pipeline rendering:
  uv pip install -e ".[warp]"

The general setup does not install the standalone Validation CLI. Install it
separately when that fixed-pipeline interface is required:
  uv pip install -e apps/validation_agent
EOF

if [[ "$INSTALL_LIVE_VIEW" -eq 1 ]]; then
    cat <<EOF
After installing the optional viewer with --live-view, run a workflow with:
  content-workflow-cli materials assign ... --live-view
EOF
fi
