# Agentic Asset Workflow — Research Preview

This directory contains the public Agentic Workflow Research Preview for USD asset
authoring. It lets a long-running coding agent, such as Codex or Claude Code,
drive Content Workbench to inspect a scene, render and pick visual evidence,
apply material or physics edits, validate results, and write reviewable run
artifacts.

Use this page when you want the agent-driven Workbench workflow. Use the
per-agent CLIs under `../apps/` when you want the older config-driven Material,
Physics, or Texture pipelines.

## 0. Quick Start

From this directory, set up the environment once:

```bash
../scripts/setup_content_agent.sh
source ../.venv/bin/activate
content-workflow-cli auth status
```

If you are using the default Codex runner and `auth status` reports missing
credentials, run `content-workflow-cli auth login`. For Claude runs, export
`ANTHROPIC_API_KEY` and pass `--runner claude` in the relevant examples below.

Then run the ladder material-assignment example:

```bash
content-workflow-cli materials assign \
  --usd ../apps/material_agent/data/examples/ladder/sources/usd/ladder.usd \
  --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg \
  --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg \
  --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --output-dir runs/content-workflow-cli/ladder-codex
```

Review the result:

```bash
sed -n '1,120p' runs/content-workflow-cli/ladder-codex/final_summary.md
ls runs/content-workflow-cli/ladder-codex/final_renders
```

## 1. Two Ways To Run

Choose the path based on how much control you want during the run.

### 1.1 CLI-Based Batch Mode

Use this path when you want the quickest reproducible run. From this directory,
run
`content-workflow-cli materials assign` for one USD asset, or
`content-workflow-cli scene run` for a larger composed scene that needs
decomposition, per-asset work, and collection back onto the original topology.
Both commands prepare inputs, start or connect to Workbench, launch the
selected child agent, validate expected artifacts, and record a trace.

### 1.2 Skill-Based Interactive Mode

Use this path when you want an iterative coding-agent session. Start Codex or
Claude from this directory and ask it to use the relevant checked-in workflow
skills. The public skills are `content-workbench`,
`content-workflow-cli`, `content-workflow-convert-to-usd`,
`content-workflow-material`, `content-workflow-physics`,
`content-workflow-simready`, and the large-scene orchestration skills
`content-workflow-large-scene`,
`content-workflow-scene-decomposition`,
`content-workflow-asset-task-processing`, and
`content-workflow-scene-collection`.

The current public-ready surface includes source-to-USD conversion,
single-asset material and physics authoring, large-scene material processing,
and SimReady profile validation. The Workbench-driven authoring paths follow
this loop:

```text
USD asset or composed USD scene + reference evidence and task policy
  -> Codex or Claude child agent
  -> Content Workbench scene APIs
  -> material or physics edits, review, and optional scene collection
  -> authored USD, evidence, JSON artifacts, trace, and summary
```

## 2. Requirements

- Python 3.12 and `uv`
- Node.js and `npm`
- Codex authentication or Claude Agent SDK credentials
- Linux with an NVIDIA GPU runtime for local Content Workbench rendering, or a
  remote Workbench endpoint passed with `--workbench-url`

Workbench receives host file paths. For remote Workbench runs, the USD asset,
reference files, and material library must be readable by the Workbench host at
the paths passed to the CLI.

## 3. Setup

From this directory:

```bash
../scripts/setup_content_agent.sh
source ../.venv/bin/activate
```

Authenticate the runner you plan to use:

```bash
# Default Codex runner
content-workflow-cli auth login
content-workflow-cli auth status

# Or Claude runner
export ANTHROPIC_API_KEY=...
```

The setup script fetches Workbench build resources by default. When targeting a
remote Workbench and local Scene Optimizer resources are not needed, run it
with `--skip-build-resources`. If you skipped resources but later need them,
run:

```bash
../scripts/fetch_build_resources.sh
```

## 4. Public Skill Quick Samples

These are first-touch samples for the public user-facing workflow entry points.
User-facing batch workflows should start with `content-workflow-cli`; the
large-scene phase skills are loaded by the child agent after `scene run`
creates durable state.

- [4.1 Convert To USD](#41-convert-to-usd)
- [4.2 Material Assignment](#42-material-assignment)
- [4.3 Physics Authoring](#43-physics-authoring)
- [4.4 SimReady Validation](#44-simready-validation)
- [4.5 Large Scene Orchestration](#45-large-scene-orchestration)
- [4.6 Content Workflow CLI](#46-content-workflow-cli)
- [4.7 Content Workbench](#47-content-workbench)

### 4.1 Convert To USD

Use `content-workflow-convert-to-usd` when a source asset needs a USD output
before Workbench-driven authoring.

```bash
content-workflow-cli convert-to-usd path/to/source.step \
  --output-format usdc \
  --output-dir runs/convert-source
```

See `packages/content_workflow_cli/README.md` and
`.agents/skills/content-workflow-convert-to-usd/SKILL.md`.

### 4.2 Material Assignment

Use `content-workflow-material` for single-asset material assignment from
reference images and a material library.

```bash
content-workflow-cli materials assign \
  --usd ../apps/material_agent/data/examples/ladder/sources/usd/ladder.usd \
  --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg \
  --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --output-dir runs/content-workflow-cli/ladder-codex
```

See `.agents/skills/content-workflow-material/SKILL.md`.

### 4.3 Physics Authoring

Use `content-workflow-physics` to author physics properties and collect
validation evidence for an asset.

```bash
content-workflow-cli physics apply \
  --usd ../apps/physics_agent/data/examples/Lightbulb01/light_bulb_01.usda \
  --output-dir runs/content-workflow-cli/lightbulb-physics
```

See `.agents/skills/content-workflow-physics/SKILL.md`.

### 4.4 SimReady Validation

Use `content-workflow-simready` to check a staged USD against the expected
SimReady profile.

```bash
content-workflow-cli simready validate-profile \
  runs/content-workflow-cli/lightbulb-physics/physics.usda \
  --report runs/simready/lightbulb-profile.json
```

See `.agents/skills/content-workflow-simready/SKILL.md`.

### 4.5 Large Scene Orchestration

Use `content-workflow-large-scene` through the public `scene run` launcher when
a composed scene needs decomposition, per-asset processing, and collection.

```bash
content-workflow-cli scene run \
  --usd path/to/scene.usd \
  --task material \
  --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --reference-dir path/to/references \
  --output-dir runs/scene-material
```

See `.agents/skills/content-workflow-large-scene/SKILL.md`.

### 4.6 Content Workflow CLI

Use `content-workflow-cli` for repeatable batch runs and help for the public
workflow commands.

```bash
content-workflow-cli --help
content-workflow-cli materials assign --help
content-workflow-cli scene run --help
```

See `packages/content_workflow_cli/README.md`.

### 4.7 Content Workbench

Use `content-workbench` in interactive agent sessions when the agent needs to
open a scene, inspect prims, render evidence, pick from images, or apply edits
through Workbench APIs.

```text
Use the content-workbench skill to load
../apps/material_agent/data/examples/ladder/sources/usd/ladder.usd,
render preview views, and write evidence under runs/workbench-ladder.
```

See `packages/content_workbench/README.md` and
`packages/content_workbench/docs/agent_api.md`.

## 5. CLI Quick Start

### 5.1 Single Asset

Run the shipped ladder example with the public material library:

```bash
content-workflow-cli materials assign \
  --usd ../apps/material_agent/data/examples/ladder/sources/usd/ladder.usd \
  --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg \
  --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg \
  --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --output-dir runs/content-workflow-cli/ladder-codex
```

To use Claude:

```bash
content-workflow-cli materials assign \
  --usd ../apps/material_agent/data/examples/ladder/sources/usd/ladder.usd \
  --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg \
  --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg \
  --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --runner claude \
  --output-dir runs/content-workflow-cli/ladder-claude
```

For an already-running or remote Workbench service:

```bash
content-workflow-cli materials assign \
  --usd /shared/assets/my_asset.usdc \
  --reference-image /shared/assets/reference_front.png \
  --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --workbench-url http://workbench-host:8088 \
  --output-dir runs/content-workflow-cli/my-asset
```

### 5.2 Large Scene

Use `scene run` when a composed USD scene is too large, repetitive, or
instanced to process as one monolithic asset. The CLI creates one durable run,
launches a long-running child agent, and requires all three handoff gates to
pass: scene decomposition, asset-task processing, and collection back onto the
original topology.

```bash
content-workflow-cli scene run \
  --usd path/to/scene.usd \
  --task material \
  --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --reference-dir path/to/references \
  --additional-instructions-file path/to/material-guidance.md \
  --output-dir runs/scene-material
```

Resume a prepared or interrupted large-scene run without changing the frozen
request:

```bash
content-workflow-cli scene resume --run-dir runs/scene-material
```

The public batch entrypoint is `content-workflow-cli scene run`. The
`content-workflow-large-scene` command family is for the running agent's
state transitions and recovery, not the normal user launcher.

### 5.3 Physics

Run the shipped Lightbulb01 example when you want Workbench-backed physics
authoring and validation evidence:

```bash
content-workflow-cli physics apply \
  --usd ../apps/physics_agent/data/examples/Lightbulb01/light_bulb_01.usda \
  --output-dir runs/content-workflow-cli/lightbulb-physics
```

By default the physics workflow runs runtime validation. Add `--no-simulation`
only when you need schema authoring artifacts on a host where runtime
validation is unavailable.

## 6. Skill-Based Interactive Mode

Start your coding agent from this directory when you want it to use the
Agentic Workflow skills directly:

```bash
codex
# or
claude
```

A useful first prompt is:

```text
Use the content-workbench and content-workflow-material skills.
Assign materials for ../apps/material_agent/data/examples/ladder/sources/usd/ladder.usd
using ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg,
../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg,
and ../apps/material_agent/data/materials/material_libs_default/materials.yaml.
Write artifacts under runs/ladder-interactive.
```

For a large scene, use the same workspace and ask for the large-scene workflow:

```text
Use the content-workflow-large-scene, content-workbench, and
content-workflow-material skills. Process path/to/scene.usd for material
assignment using path/to/references and
../apps/material_agent/data/materials/material_libs_default/materials.yaml.
Write artifacts under runs/scene-interactive.
```

## 7. Reviewing Results

Every run writes a self-contained result folder at the `--output-dir` path.
Start there and look for the summary, rendered or validation evidence, any
authored USD output, and the final status. The exact file layout varies by
workflow, but the result folder should answer these questions:

- What did the agent change?
- What evidence supports the result?
- Did any checks remain unresolved?
- What logs or traces are available if you need to debug or resume?

For the ladder quick start:

```bash
ls runs/content-workflow-cli/ladder-codex
sed -n '1,120p' runs/content-workflow-cli/ladder-codex/final_summary.md
ls runs/content-workflow-cli/ladder-codex/final_renders
```

For physics runs, review the authored USD output and validation status in the
run folder. For large-scene runs, start with the run summary and final
collected scene before drilling into phase details.

Unresolved visual or physical quality is a workflow outcome, not always a
process failure. For example, a run may finish while recording that a part
cannot be separated with the available mesh/material granularity.

## 8. Reference

- `packages/content_workflow_cli/README.md`: CLI reference after public
  staging replacement.
- `packages/content_workbench/README.md`: Workbench service reference.
- `packages/content_workbench/docs/agent_api.md`: REST API guide also exposed
  by a running Workbench at `/agent-api`.
