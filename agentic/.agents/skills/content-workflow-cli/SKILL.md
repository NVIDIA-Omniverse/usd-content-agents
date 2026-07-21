---
name: content-workflow-cli
description: Use when the user wants the public batch shortcut for a prepared agentic asset workflow, including single-asset material assignment and three-phase large-scene execution or resume through content-workflow-cli.
---

# content-workflow-cli

Use this skill when the user wants the batch shortcut for a prepared agentic
asset workflow.

`content-workflow-cli` is a launcher. It is not the source of truth for
Workbench API mechanics or workflow methodology.

## Current Primary Workflow

```bash
content-workflow-cli materials assign \
  --usd path/to/asset.usdc \
  --reference-image path/to/reference.png \
  --additional-instructions-file path/to/material-guidance.md \
  --optimizer-selection agent \
  --materials-yaml path/to/materials.yaml \
  --output-dir agentic/runs/example
```

Use `--optimizer-selection agent` when optimizer behavior should be chosen from
the asset rather than fixed by the launcher. The wrapper first provides an
unoptimized Workbench inspection to the child agent, validates the resulting
`raw/optimizer_decision.json`, and only then creates the material-authoring
session with the selected optimization, prototype flattening, deinstancing,
splitting, and deduplication settings. Fixed optimizer flags remain available
for reproducible diagnostics and explicit user overrides, but do not combine
them with agent selection.

Optimizer decisions are task-scoped. `materials assign` selects settings against
visible material coverage, independent appearance authoring, and source-path
mapping. `physics apply` runs a separate unoptimized inspection and selects
settings against component membership, rigid-body and joint topology,
collider/helper roles, legal authoring targets, and runtime behavior. Each
`optimizer_decision.json` declares `task`; the wrapper rejects a decision for a
different operation.

Use `--additional-instructions` for short inline guidance or
`--additional-instructions-file` for durable multi-line policy. The wrapper
stores the normalized text in `request.json`, includes it in the child-agent
task, and reuses it during bounded VQA refinement. Provide task-wide guidance
once; do not generate per-prim prompt copies.

## Large-Scene Workflow

Use the public scene launcher rather than invoking the internal transition tool
or agent runtime directly:

```bash
content-workflow-cli scene run \
  --usd path/to/scene.usd \
  --task material \
  --materials-yaml path/to/materials.yaml \
  --reference-dir path/to/references \
  --reference-image path/to/accepted-render.png \
  --additional-instructions-file path/to/material-guidance.md
```

The wrapper writes a resolved request and `large_scene_run.json`, launches one
long-running agent from the isolated `agentic/` workspace, and returns success
only after decomposition, asset-task processing, and collection complete their
handoff gates. Resume a prepared or interrupted run with:

```bash
content-workflow-cli scene resume --run-dir agentic/runs/RUN_ID
```

`content-workflow-large-scene` is an internal transition utility used by the
umbrella skill, tests, and recovery. Do not expose it as the user launch command.

## Wrapper Responsibilities

- parse workflow inputs;
- create a run directory;
- write `request.json`;
- start or connect to Workbench;
- launch Codex or Claude Code;
- pass a compact structured task request;
- validate required workflow artifacts;
- require terminal phase validation for a large-scene run;
- preserve logs, traces, and partial outputs.

## Agent Responsibilities

The launched agent should use the same skills available in interactive mode,
especially:

- `content-workbench`
- `content-workflow-material`
- `content-workflow-large-scene`

## Rule

If a task needs custom reasoning, inspection, or recovery, prefer an interactive
long-running agent session from `agentic/` using the Workbench and workflow
skills directly.
