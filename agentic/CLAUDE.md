# Agentic Workflow Claude Code Guide

This directory is the public Agentic Workflow preview workspace for Claude Code.
Use it when Claude should operate USD assets through Content Workbench and the
Agentic Workflow skills.

## Start Here

1. Read `README.md` in this directory.
2. If setup has not run yet, run this command from this directory:

   ```bash
   ../scripts/setup_content_agent.sh
   ```

3. Activate the repository environment:

   ```bash
   source ../.venv/bin/activate
   ```

4. Export Claude credentials through the Claude Agent SDK environment, such as
   `ANTHROPIC_API_KEY`.

## Skill Routing

Use these public skills under `.claude/skills/`:

- `content-workbench`
- `content-workflow-cli`
- `content-workflow-convert-to-usd`
- `content-workflow-material`
- `content-workflow-physics`
- `content-workflow-simready`
- `content-workflow-large-scene`
- `content-workflow-scene-decomposition`
- `content-workflow-asset-task-processing`
- `content-workflow-scene-collection`

For one-line samples covering every public skill, use the "Public Skill Quick
Samples" table in `README.md`.

## First Command

From this directory:

```bash
content-workflow-cli materials assign \
  --usd ../apps/material_agent/data/examples/ladder/sources/usd/ladder.usd \
  --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg \
  --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg \
  --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --runner claude \
  --output-dir runs/content-workflow-cli/ladder-claude
```

For a large scene:

```bash
content-workflow-cli scene run \
  --usd path/to/scene.usd \
  --task material \
  --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --reference-dir path/to/references \
  --additional-instructions-file path/to/material-guidance.md \
  --runner claude \
  --output-dir runs/scene-material-claude
```

For physics authoring:

```bash
content-workflow-cli physics apply \
  --usd ../apps/physics_agent/data/examples/Lightbulb01/light_bulb_01.usda \
  --runner claude \
  --output-dir runs/content-workflow-cli/lightbulb-physics-claude
```

## Safety

- Keep credentials out of prompts, run requests, and commits.
- Use `--workbench-url` only for a Workbench host that can read the same asset
  and material-library paths.
- Do not edit source USD files directly unless the user asks for source
  mutation.
- Summarize final render paths, material assignments, VQA status, and trace
  artifacts when reporting completion.
- For large scenes, summarize `large_scene_run.json`, completed phase outputs,
  collection output, and terminal validation status.
- For physics runs, summarize the authored physics USD,
  `physics_assignments.json`, `physics_behavior_assessment.json`,
  `validation_evidence.json`, and runtime validation status.
