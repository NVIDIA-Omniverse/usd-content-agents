# Content Agents Agent Guide

This file provides public, repo-local guidance for coding agents working in
`NVIDIA-Omniverse/usd-content-agents`.

USD Content Agents is a reference implementation intended to be read, forked,
and adapted, not a stable SDK or deploy-as-is product. Treat its interfaces,
skills, and artifact contracts as release-specific.

## Start Here

Stay at the repository root. The agentic Content Workflow is the default
for supported asset tasks.

```bash
./scripts/setup_content_agent.sh
source .venv/bin/activate
content-workflow-cli auth status
content-workflow-cli --help
```

Native Windows execution is unsupported in the 0.6 release. On a Windows host,
run the supported workflow inside WSL2 and use the Linux setup commands above.

Python 3.12 and `uv` are required. Node.js 20+ and `npm` are also required for
workflows that launch Codex or Claude Code; direct-only workflows may use
`--without-child-runners` on Linux/WSL2.

Clone inside the WSL2 Linux filesystem so the checked-in workflow skill
symlinks materialize correctly. If a path such as
`.agents/skills/content-workflow-material` is a plain text file instead of a
directory, stop and clone again inside WSL2. Do not continue with a partial
skill inventory.

Keep API keys in `.env` or environment variables. Never print, commit, or paste
secrets.

## Default Routing Rule

- Route supported, unqualified conversion, material, texture, physics,
  segmentation, validation, articulation, composed-asset, and large-scene
  requests through `content-workflow-cli` and the root `content-workflow-*`
  skills.
- Never silently fall back to fixed pipeline when an agentic prerequisite is
  missing. Report the blocker and remediation.
- Write user-facing run artifacts under root `runs/` unless the user specifies
  another output directory.

| User goal | Default root skill or command |
|---|---|
| General setup and routing | `.codex/skills/quickstart` or `.claude/skills/quickstart` |
| Prepared agentic workflow | `content-workflow-cli` |
| Material authoring | `content-workflow-material` |
| Texture authoring | `content-workflow-texture` and its focused `texture prepare` through `texture publish` operations |
| Physics authoring | `content-workflow-physics` |
| usd-cli operations | `usd-cli` |
| Conversion | `content-workflow-convert-to-usd` |
| Mesh segmentation | `content-workflow-mesh-segmentation` |
| SimReady profile work | `content-workflow-simready` |
| Large scenes | `content-workflow-large-scene` and its phase skills |
| Prompt validation | `content-workflow-cli validate run` |
| Reviewed articulation | `content-workflow-cli articulation run` |
| Composed asset | `content-workflow-asset` |

## Explicit Fixed Pipeline

The established application workflows under `apps/` are the fixed pipeline.
Use them only when the user explicitly asks for fixed pipeline or names an app
CLI, YAML configuration, Python API, benchmark, REST service, or deployment.

Invoke the single root `$fixed-pipeline` skill and let it load the one relevant
reference from `.agents/skills/fixed-pipeline/references/`. Do not present those
nested references as independent skills and do not route an unqualified task
to `material-agent`, `physics-agent`, `joint-agent`, `texture-agent`, or
`validation-agent`.

## Skill Organization

- `.agents/skills/` is the canonical root discovery tree.
- `.codex/skills/` and `.claude/skills/` are compatibility mirrors of that
  tree.
- Use Agentic skills through the root `.agents/skills/` discovery tree; their
  implementation location is not a separate user entrypoint.
- `fixed-pipeline` is the only discoverable fixed-pipeline skill; detailed
  procedures live beneath its `references/` directory.
- Use the owning workflow's render method, or `usd-cli` for an explicit
  low-level local or remote OVRTX render.
- `agentic/` is an implementation workspace, not a second user entrypoint.

## First Workflow

Before running this smoke test, complete the applicable local or remote OVRTX
setup and successful readiness probe in the root
[Quick Start](README.md#2-quick-start).

```bash
content-workflow-cli materials assign \
  --usd apps/material_agent/data/examples/ladder/sources/usd/ladder.usd \
  --reference-image apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg \
  --reference-image apps/material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg \
  --materials-yaml apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --output-dir runs/ladder-agentic
```

This ladder run is the first smoke test, not the boundary of the product. Use
`content-workflow-cli --help` for the complete supported command surface and
`MIGRATING_TO_0_6.md` for the agentic/fixed pipeline capability matrix.

## Platform

Agentic Content Workflows support native Linux and WSL2. Native Windows
execution is unsupported in the 0.6 release. On a Windows host, run the
supported workflow inside WSL2.

Agentic rendering under WSL2 uses remote OVRTX. Fixed pipelines run under WSL2
with the `warp` rendering backend. Native Windows fixed-pipeline execution and
local OVRTX rendering under WSL2 are not supported.

WSL2 cannot run local OVRTX because it does not expose the required Vulkan
driver. Remote OVRTX packages the composed scene as USDZ so the renderer does
not need the workflow host's asset paths.

## Repository Map

- `.agents/skills/` — canonical root discovery tree.
- `agentic/packages/` — Content Workflow implementations.
- `apps/usd_cli/` — the low-level USD CLI and sidecar implementation.
- `.agents/skills/fixed-pipeline/references/` — non-discoverable fixed-pipeline procedures.
- `apps/*_agent/` — explicit fixed-pipeline config-driven applications.
- `apps/*_agent_service/` — explicit fixed-pipeline REST services.
- `scripts/setup_content_agent.sh` — Linux/WSL2 Agentic setup.
- `scripts/setup_content_agent.ps1` — development helper; not a supported 0.6
  native-Windows release path.
- `runs/` — ignored root output location.

## Validation

```bash
python3 -m pytest tests/test_public_quickstart_regression.py tests/test_packaging_requests_declared.py tests/test_pyproject_fallback_version.py -q
./scripts/sync_agent_skills.sh --check
```

## Safety

- Do not mutate source USD files unless the user explicitly requests it.
- Preserve run evidence and partial outputs on failure or cancellation.
- Use the workflow's reported resume command; otherwise provide a safe restart
  with a new run directory.
- Treat commands that delete run directories as destructive.
