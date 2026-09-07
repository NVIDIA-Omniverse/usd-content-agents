---
name: quickstart
description: Set up and route NVIDIA Content Agents through the default Agentic Content Workflow from the repository root. Use when the user asks for /quickstart, first-run setup, an unqualified supported content task, or prerequisite checks. Route an explicitly requested fixed pipeline CLI, REST, config, benchmark, or deployment through the fixed-pipeline umbrella without treating it as the default.
version: "0.2.4"
author: NVIDIA Omniverse
tags:
  - content-agents
  - quickstart
  - agentic
  - routing
tools:
  - Shell
  - Filesystem
  - curl
compatibility: Requires a native Linux or WSL2 repository checkout with Python 3.12 and uv. Native Windows execution is unsupported in the 0.6 release. Child-agent workflows additionally require Node.js, npm, and runner credentials; each capability requires only its documented services.
metadata:
  author: NVIDIA Omniverse
  tags:
    - content-agents
    - quickstart
    - agentic
    - routing
---

# Content Agents Quickstart

## When to Use

- Set up Content Agents for the first time from the repository root.
- Choose the supported workflow for a user request.
- Check agent, usd-cli, model, render, or service prerequisites.
- Route an explicit fixed pipeline request without changing its meaning.

## Limitations

- Agentic workflows support native Linux and WSL2. Native Windows execution is
  unsupported in the 0.6 release.
- Local usd-cli OVRTX rendering supports compatible native Linux NVIDIA
  RTX/Vulkan hosts. WSL2 cannot run local OVRTX; Agentic rendering there uses a
  remote OVRTX service configured through usd-cli.
- Keep secrets in `.env` or the environment. Never print or commit them.
- Never silently fall back to fixed pipeline when agentic setup is blocked.

## Routing

`/quickstart` means Agentic by default. Never select or start a fixed pipeline
unless the user explicitly asks for that interface.

Use the agentic Content Workflow for supported, unqualified asset requests.
The established application workflows under `apps/` are the fixed pipeline.
Invoke `$fixed-pipeline` only when the user explicitly requests it, names one
of its runtime commands or services, asks for its YAML configuration, or needs
its benchmark/dataset compatibility. The umbrella loads the narrowest bundled
reference; those references are not independently discoverable skills.

| Intent | Route |
|---|---|
| Convert an asset to USD | `content-workflow-cli convert-to-usd` |
| Assign or refine materials | `content-workflow-material` / `content-workflow-cli materials assign` |
| Generate scoped textures | `content-workflow-texture`; use focused `content-workflow-cli texture prepare` through `texture publish` operations |
| Author and validate physics | `content-workflow-physics` / `content-workflow-cli physics apply` |
| Segment a fused mesh | `content-workflow-mesh-segmentation` / `content-workflow-cli mesh-segmentation run` |
| Validate against a prompt | `content-workflow-cli validate run` |
| Validate/conform SimReady profile | `content-workflow-simready` / `content-workflow-cli simready ...` |
| Process a composed scene | `content-workflow-cli scene run` |
| Compose Joint through validation | `content-workflow-asset` / `content-workflow-cli asset run` |
| Explicit `material-agent` or Material REST | `$fixed-pipeline`; load `material-agent-cli` or `material-agent-client` reference |
| Explicit `physics-agent` or Physics REST | `$fixed-pipeline`; load `physics-agent-cli` or `physics-agent-client` reference |
| Explicit fixed pipeline Joint/Texture/Validation CLI | `$fixed-pipeline`; load the corresponding CLI reference |
| Start a fixed pipeline local service | `$fixed-pipeline`; load the corresponding deployment reference |

### Routing Examples

| Request | Required mode |
|---|---|
| “Assign materials to this asset.” | Agentic |
| “Use the agentic workflow for this asset.” | Agentic |
| “Inspect the scene and refine the material result.” | Agentic |
| “Use the fixed pipeline material pipeline.” | Fixed pipeline |
| “Run `material-agent` with this YAML.” | Fixed pipeline |
| “Submit this asset to the Material REST service.” | Fixed pipeline |
| “Run the material benchmark dataset.” | Fixed pipeline |
| “Compare agentic and fixed pipeline material results.” | Both, separately labeled |
| Agentic prerequisite is unavailable | Stop with remediation; no fallback |

## Instructions

1. Stay at the repository root.
2. Check the supported platform and required commands.
3. Run the root setup script.
4. Activate `.venv` and verify the selected runner's authentication.
5. Run `content-workflow-cli preflight --help` and the command-specific
   preflight or `--help` before an expensive workflow.
6. Use root `runs/<descriptive-name>` for output unless the user supplied a
   path.
7. Report the selected mode, prerequisites, command, output directory, and
   recovery command.

## Prerequisites

- Native Linux or WSL2 for Agentic workflows. On a Windows host, run the
  workflow inside WSL2.
- Python 3.12 and `uv`. Node.js and npm are additionally required only for
  workflows that launch Codex or Claude child agents; direct-only setup can use
  `--without-child-runners`.
- On Linux/WSL2, default setup also requires `unzip` for Scene Optimizer build
  resources. Install it with `sudo apt-get update && sudo apt-get install -y
  unzip`, or use `--skip-build-resources` when those resources are not needed.
- On Linux, a C++17 compiler, CMake, Ninja, and outbound HTTPS for workflows
  that need the fresh exact-OCP/Geometry provider build. It downloads
  approximately 55 MB of pinned source archives and performs a substantial
  native compilation.
- Codex authentication or Claude credentials.
- The installed usd-cli for workflows that inspect or author USD, plus a local
  or configured remote OVRTX renderer when visual evidence is required.

## Setup

On Linux/WSL2:

```bash
uname -s
command -v uv
uv python find 3.12

# Required for default build-resource setup; omit with --skip-build-resources:
command -v unzip

# Required only for workflows that launch a Codex or Claude child agent:
command -v node
command -v npm

./scripts/setup_content_agent.sh
source .venv/bin/activate
content-workflow-cli auth status --sandbox-smoke
content-workflow-cli --help
```

Native Windows is not a supported 0.6 setup path. On a Windows host, enter WSL2
and use the Linux commands above.

Verify the installed low-level tool. Before an expensive workflow, let the
workflow run its own OVRTX readiness probe, or run the same probe explicitly.
On native Linux:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
usd-cli --version
usd-cli render-probe --require-engine ovrtx \
  --output-dir .local-runs/usd-cli-readiness
```

To use a local OVRTX runtime when one is not already provisioned, explicitly
allow the one-time download before the daemon starts, then run the same probe:

On native Linux:

```bash
export WU_OVRTX_AUTO_PROVISION=1
usd-cli server stop  # required after changing the variable for a running daemon
usd-cli render-probe --require-engine ovrtx \
  --output-dir .local-runs/usd-cli-readiness
```

The initial probe deliberately exits after reporting `auto-install STARTED`;
rerun that exact command while it reports `auto-install in progress`. The
workflow readiness check polls those transitional states within its configured
timeout. For Geometry's reviewed hash-locked provision-only path, see
`agentic/docs/geometry_quickstart.md`.

The workflow manages one usd-cli sidecar for its session. Do not manually start
a second sidecar for the same run. Docker, Compose, container-GPU, and service
port checks belong to the selected `$fixed-pipeline` deployment reference; do
not require them for an Agentic workflow that does not use those services.

For direct conversion, Texture, Validation, or SimReady work that will not
launch Codex or Claude, omit Node/npm SDK setup. On Linux/WSL2:

```bash
./scripts/setup_content_agent.sh --without-child-runners
```

If authentication is missing:

```bash
content-workflow-cli auth login
```

For an existing remote OVRTX service, configure usd-cli before launching the
workflow:

```bash
usd-cli remote configure https://gpu-host.example.test:8000
```

Do not pass a backend URL to `content-workflow-cli`; backend selection remains
a low-level usd-cli configuration detail.

## First Workflow

Run commands from the repository root. The ladder example is the first public
smoke test, not the boundary of the agentic product:

```bash
content-workflow-cli materials assign \
  --usd apps/material_agent/data/examples/ladder/sources/usd/ladder.usd \
  --reference-image apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg \
  --reference-image apps/material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg \
  --materials-yaml apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --output-dir runs/ladder-agentic
```

Use `content-workflow-cli --help` and the routing table for other supported
capabilities.

## Output Format

Return:

- workflow: `agentic` or `fixed-pipeline`;
- setup/preflight status;
- exact command;
- root-relative run directory;
- canonical output and evidence summary when available;
- unresolved prerequisites or decisions;
- exact resume or safe-restart command.

## Troubleshooting

| Symptom | Action |
|---|---|
| `content-workflow-cli` missing | Run `./scripts/setup_content_agent.sh` on native Linux/WSL2, activate `.venv`, and retry. |
| Authentication missing | Run `content-workflow-cli auth login`; do not request secrets in chat. |
| Login exists but the auth probe says the model requires a newer Codex | Upgrade the Codex app/CLI, then rerun `content-workflow-cli auth status`. |
| usd-cli unavailable | Rerun setup and activate `.venv`; for rendering, provision local OVRTX only on native Linux, or configure a remote OVRTX service on WSL2, then rerun `usd-cli render-probe --require-engine ovrtx`. |
| Native Windows or macOS workflow error | Run the workflow on native Linux or inside WSL2; do not fall back to fixed pipeline automatically. |
| Explicit fixed pipeline request | Invoke `$fixed-pipeline`, load its named CLI, client, or deployment reference, and preserve the requested interface. |
