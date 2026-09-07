# Content Agents Claude Code Guide

This file provides public, repo-local guidance for Claude Code working in
`NVIDIA-Omniverse/usd-content-agents`.

USD Content Agents is a reference implementation intended to be read, forked,
and adapted, not a stable SDK or deploy-as-is product. Treat its interfaces,
skills, and artifact contracts as release-specific.

## Start Here

Start Claude Code from the repository root. The agentic Content Workflow is the
default for supported tasks.

```bash
./scripts/setup_content_agent.sh
source .venv/bin/activate
content-workflow-cli --help
```

Native Windows execution is unsupported in the 0.6 release. On a Windows host,
run the supported workflow inside WSL2 and use the Linux setup commands above.

Python 3.12 and `uv` are required. Node.js 20+ and `npm` are also required for
workflows that launch Claude Code; direct-only workflows may use
`--without-child-runners` on Linux/WSL2.

Clone inside the WSL2 Linux filesystem so the checked-in workflow skill
symlinks materialize correctly. If a canonical skill path is a plain text file
instead of a directory, stop and clone again inside WSL2.

Keep `ANTHROPIC_API_KEY` and other provider credentials in the environment or
`.env`; never print or commit them.

## Default Routing Rule

- Use root `.claude/skills/content-workflow-cli` and the
  `content-workflow-*` skills for supported, unqualified asset requests.
- Never silently fall back to fixed pipeline when agentic setup is blocked.
- Write outputs under root `runs/` unless the user supplies another path.

## Explicit Fixed Pipeline

The established application workflows under `apps/` are the fixed pipeline.
Use them only when the user explicitly asks for fixed pipeline or names an app
CLI, YAML configuration, Python API, benchmark, REST service, or deployment.

Invoke the single root `$fixed-pipeline` skill. It selects one relevant nested
reference from `.agents/skills/fixed-pipeline/references/`; those references
are not independent skills. Do not route an unqualified task to
`material-agent`, `physics-agent`, `joint-agent`, `texture-agent`, or
`validation-agent`.

## Skill Organization

- `.agents/skills/` is the canonical root discovery tree;
  `.claude/skills/` is its compatibility mirror.
- Use Agentic skills through the root `.agents/skills/` discovery tree; their
  implementation location is not a separate user entrypoint.
- `fixed-pipeline` is the sole discoverable fixed-pipeline skill.
- Use the owning workflow's render method, or `usd-cli` for an explicit
  low-level local or remote OVRTX render.
- `agentic/` is an implementation workspace, not a separate user entrypoint.

## First Workflow

Before running this smoke test, complete the applicable local or remote OVRTX
setup and successful readiness probe in the root
[Quick Start](README.md#2-quick-start).

```bash
content-workflow-cli materials assign \
  --runner claude \
  --usd apps/material_agent/data/examples/ladder/sources/usd/ladder.usd \
  --reference-image apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg \
  --reference-image apps/material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg \
  --materials-yaml apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --output-dir runs/ladder-agentic-claude
```

Use `content-workflow-cli --help` for conversion, texture, physics,
segmentation, validation, large-scene, articulation, and composed-asset
commands. See `MIGRATING_TO_0_6.md` for the capability matrix.

## Validation

Run the public quickstart and packaging regression checks before declaring
related documentation or skill changes ready:

```bash
python3 -m pytest tests/test_public_quickstart_regression.py tests/test_packaging_requests_declared.py tests/test_pyproject_fallback_version.py -q
```

## Platform And Safety

- Agentic Content Workflows support native Linux and WSL2. Native Windows
  execution is unsupported in the 0.6 release. On a Windows host, run the
  supported workflow inside WSL2.
- Agentic rendering under WSL2 uses remote OVRTX. Fixed pipelines run under
  WSL2 with the `warp` rendering backend. Native Windows fixed-pipeline
  execution and local OVRTX rendering under WSL2 are not supported.
- Claude child execution requires its documented OS sandbox prerequisites.
- WSL2 cannot run local OVRTX because it does not expose the required Vulkan
  driver.
- Do not mutate source USD files without explicit user approval.
- Preserve partial runs and report the exact resume or safe-restart command.
- Keep credentials and generated run artifacts out of Git.
