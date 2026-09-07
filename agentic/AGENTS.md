# Agentic Workflow Compatibility Guide

The repository root is the canonical entrypoint for Content Agent workflows.
This directory contains implementation packages and compatibility links for
existing nested-workspace users.

## Start From The Repository Root

```bash
./scripts/setup_content_agent.sh
source .venv/bin/activate
content-workflow-cli auth status
content-workflow-cli --help
```

Do not require users to enter `agentic/`. Release-approved workflow skills are
discoverable under root `.agents/skills`; root `.codex/skills` and
`.claude/skills` expose the same set.

## Routing

- The established application workflows under `apps/` are the fixed pipeline.
  Invoke the root `$fixed-pipeline` umbrella only when the user explicitly
  requests one of those interfaces.
- Use `content-workflow-cli` and the `content-workflow-*` skills by default for
  supported conversion, material, texture, physics, segmentation, validation,
  large-scene, articulation, and composed-asset tasks.
- Use `material-agent`, `physics-agent`, `joint-agent`, `texture-agent`, their
  YAML configs, benchmarks, or REST services only when the user explicitly
  requests that fixed pipeline interface.
- Never silently fall back to fixed pipeline when an agentic prerequisite is
  missing.
- Write run artifacts under root `runs/` unless the user provides another
  output directory.

See the root `README.md` for first-run examples and
`../MIGRATING_TO_0_6.md` for the canonical capability and recovery matrix.

## Skill Organization

- Root `.agents/skills/` is the canonical discovery surface.
- Agentic implementations remain in this directory under `.agents/skills/` and
  are promoted to root discovery.
- Fixed-pipeline procedures remain nested under the root
  `.agents/skills/fixed-pipeline/references/` umbrella and must not be invoked
  as independent skills.
- Use the owning workflow's render method, or `usd-cli` for an explicit
  low-level local or remote OVRTX render.
