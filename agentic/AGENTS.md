# Agentic Workflow Agent Guide

This directory is the public Agentic Workflow preview workspace. Use it when a
coding agent should operate a USD asset through Content Workbench instead of
running the older config-driven app CLIs.

## Start Here

1. Read `README.md` in this directory.
2. If `content-workflow-cli` is missing, run from this directory:

   ```bash
   ../scripts/setup_content_agent.sh
   ```

3. Make sure the repository environment is active:

   ```bash
   source ../.venv/bin/activate
   ```

4. Keep secrets in environment variables or `.env`; never print or commit them.

## Skill Routing

The public Agentic Workflow skills live under `.agents/skills/`. The
`.codex/skills` and `.claude/skills` paths are compatibility mirrors.

Use these skills for public single-asset workflows:

- `content-workbench`: Content Workbench scene/session/render/pick/edit APIs.
- `content-workflow-cli`: Public batch launcher guidance for prepared runs.
- `content-workflow-convert-to-usd`: Convert supported source assets to USD.
- `content-workflow-material`: Assign materials and run visual review.
- `content-workflow-physics`: Author physics schema and validation evidence.
- `content-workflow-simready`: Run SimReady profile checks and evidence export.

Use these skills for public large-scene workflows:

- `content-workflow-large-scene`: Coordinate decomposition, asset-task
  processing, collection, handoff gates, and recovery.
- `content-workflow-scene-decomposition`: Build processable representatives and
  original-topology mappings.
- `content-workflow-asset-task-processing`: Run per-representative domain tasks
  such as material or physics.
- `content-workflow-scene-collection`: Project validated task results back onto
  the original scene topology.

For one-line samples covering every public skill, use the "Public Skill Quick
Samples" table in `README.md`.

## First Command

From this directory, run the shipped ladder example:

```bash
content-workflow-cli materials assign \
  --usd ../apps/material_agent/data/examples/ladder/sources/usd/ladder.usd \
  --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg \
  --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg \
  --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --output-dir runs/content-workflow-cli/ladder-codex
```

For a large scene, use the batch scene launcher rather than invoking transition
helpers directly:

```bash
content-workflow-cli scene run \
  --usd path/to/scene.usd \
  --task material \
  --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --reference-dir path/to/references \
  --additional-instructions-file path/to/material-guidance.md \
  --output-dir runs/scene-material
```

For physics authoring, run the shipped Lightbulb01 example:

```bash
content-workflow-cli physics apply \
  --usd ../apps/physics_agent/data/examples/Lightbulb01/light_bulb_01.usda \
  --output-dir runs/content-workflow-cli/lightbulb-physics
```

## Safety

- Treat Workbench as a trusted local sidecar. Do not expose it to shared
  networks without an operator-controlled proxy.
- Do not mutate source USD files unless the user explicitly requests it.
- Write run outputs under `runs/`.
- Inspect `assignments.json`, `visual_quality_assessment.json`, final renders,
  and `trace/operation_trace.md` before claiming success.
- For large scenes, inspect `large_scene_run.json`, phase outputs, and terminal
  validation before reporting completion.
- For physics runs, inspect `physics_assignments.json`,
  `physics_behavior_assessment.json`, `validation_evidence.json`, and runtime
  validation artifacts before reporting completion.
