# Agentic Content Workflow

This is the detailed reference for the Agentic backend of the
[USD Content Agents](../README.md) reference implementation. Start every
user-facing workflow from the repository root. The `agentic/` directory
contains implementation details; it is not a separate product entry point, a
stable SDK, or a directory users need to work from.

The interactive coding-agent experience and `content-workflow-cli` batch
commands use the same workflows, tools, run-state contracts, and evidence
model. They differ only in how a run is initiated and controlled.

## 1. Overview and Scope

Use the Agentic backend when the next operation depends on discovered scene
structure, visual evidence, validation results, or earlier attempts. A coding
agent plans the work, invokes typed scene operations, reviews the resulting
evidence, and makes bounded refinements. Each workflow writes a durable run
directory containing the inputs, decisions, authored USD, evidence, status,
and summary appropriate to that task.

Do not silently switch to a fixed pipeline when an Agentic prerequisite is
missing. Report the missing prerequisite and remediation instead. Use the
[root documentation](../README.md) for the fixed-pipeline backend only when the
request explicitly requires an established Material, Physics, Joint, Texture,
or Validation app CLI, YAML contract, Python API, REST service, benchmark, or
deployment.

## 2. Quick Start

Clone the repository and open its root in Codex or Claude Code:

```bash
git clone https://github.com/NVIDIA-Omniverse/usd-content-agents.git
cd usd-content-agents
```

Native Windows execution is unsupported in the 0.6 release. On a Windows host,
run the supported workflow inside WSL2 and clone into the WSL2 Linux filesystem
so the checked-in skill symlinks materialize correctly.

Then describe the asset outcome you want:

```text
Take /absolute/path/to/my_asset.usd and assign materials based on this reference
image: /absolute/path/to/reference.png.
```

The coding agent discovers the checked-in setup and workflow skills, installs
what the selected workflow requires, and reports any missing runtime or
credential with the command needed to resolve it. Personal skill installation
is not required inside the checkout.

## 3. Choose an Entry Point

| Entry point | Use it when | Backend and outputs |
|---|---|---|
| [Interactive coding agent](#31-interactive-coding-agent) | You want to describe an outcome, review intermediate evidence, and refine the request conversationally. | Agentic backend with durable run artifacts |
| [Batch CLI](#32-batch-cli) | You want a repeatable command with explicit inputs, output directory, and recovery behavior. | The same Agentic backend and durable run artifacts |

### 3.1 Interactive Coding Agent

Start Codex or Claude Code from the repository root and describe the desired
result in natural language. The coding agent routes the request to the
appropriate checked-in workflow and capability skills.

You can begin with the prompt in [Quick Start](#2-quick-start). For subsequent
tasks, specify the source asset, reference evidence, desired result, and output
directory when those details matter. The agent will inspect the asset and
explain material prerequisites or blockers before starting expensive work.

### 3.2 Batch CLI

Prepare the Agentic environment from the repository root:

```bash
cp .env_example .env
./scripts/setup_content_agent.sh
source .venv/bin/activate

content-workflow-cli auth login
content-workflow-cli auth status
content-workflow-cli --help
```

On a Windows host, enter WSL2 and use the Linux setup commands above.

On Linux/WSL2, use `--skip-build-resources` when local Scene Optimizer resources
are not needed. Use `--without-child-runners` only for commands that do not
launch Codex or Claude Code. Run the relevant subcommand with `--help` before
automating it.

## 4. Requirements and Configuration

Agentic workflows support native Linux and WSL2. Their shared requirements are
Python 3.12 and [`uv`](https://docs.astral.sh/uv/). Additional integration
requirements may include:

- Node.js 20+ and `npm` when launching a child coding agent;
- Codex authentication or Claude credentials; and
- a local NVIDIA GPU runtime or the applicable remote service.

Native Windows and native macOS are not supported release or runtime targets
for 0.6. `usd-cli` receives host file paths, so inputs must be readable from the
workflow session's configured allowed roots. Local OVRTX rendering supports
compatible NVIDIA RTX/Vulkan hosts on native Linux. WSL2 cannot run local OVRTX;
Agentic rendering there uses remote OVRTX.

Configure only the providers needed by the selected workflow:

| Provider | Environment variable |
|---|---|
| NVIDIA hosted models | `NVIDIA_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` |
| NVCF-hosted functions | `NGC_API_KEY` plus the selected function IDs |
| Remote rendering | `RENDER_ENDPOINT` |
| Remote Scene Optimizer | `OPTIMIZER_ENDPOINT` |

Never commit `.env` or paste credentials into prompts. Child-agent workflows
fail closed when their required sandbox and process-isolation controls are not
available. See the [Content Workflow CLI reference](packages/content_workflow_cli/README.md)
for the complete platform and child-runner security requirements.

## 5. Workflow Catalog

The table below matches the Agentic capability surface advertised by the root
README. Interactive and batch routes share the same owning workflow unless a
route is explicitly identified as interactive-only or compatibility-only.

| Goal | Default Agentic route |
|---|---|
| Generate auditable geometry | `geometry-agent generate` through an explicitly configured external authoring provider |
| Segment a fused mesh into semantic parts | `content-workflow-cli mesh-segmentation run` |
| Convert a supported source asset to USD | `content-workflow-cli convert-to-usd` |
| Assign materials with iterative visual review | `content-workflow-cli materials assign` or the interactive `content-workflow-material` skill |
| Generate and apply scoped textures | Interactive `content-workflow-texture` with focused `texture prepare` through `texture publish` operations |
| Author physics and collect behavior evidence | `content-workflow-cli physics apply` |
| Infer, review, and author articulation | `content-workflow-cli articulation run` |
| Validate content against a prompt and evidence | `content-workflow-cli validate run` |
| Validate or conform to the SimReady profile | `content-workflow-cli simready ...` |
| Process a large composed scene | `content-workflow-cli scene run` |
| Coordinate one asset through Joint, Material, Texture, Physics, and Validation | `content-workflow-cli asset run` |

### 5.1 Geometry Agent

Geometry Agent covers three related capabilities:

- **External authoring:** use the authenticated Geometry Agent service/CLI to
  generate, revise, or export geometry through one explicitly configured
  provider. The public runtime accepts typed receipts and immutable
  `geometry.source.v1` bundles; it never executes provider-native source.
- **Geometry handoff:** prepare, optimize, repair, validate, and render an
  existing artifact or source bundle with `content-workflow-cli geometry run`.
- **Mesh segmentation:** use the interactive
  `content-workflow-mesh-segmentation` skill or its batch launcher to turn a
  fused mesh into named, independently reviewable parts.

For text or image generation, configure the service with an external provider,
then use the credential-safe CLI:

```bash
geometry-agent providers
geometry-agent generate \
  --provider build123d-http \
  --prompt "A 60 mm mounting bracket with two M5 clearance holes" \
  --format step --format usdc \
  --target-profile geometry-agent.insertion-or-fixture-asset.v1 \
  --output runs/my-bracket/generation.json
```

The included Build123d connector talks only to an isolated remote worker.
Optional `forgecad-http` delegates to a separately operated, explicitly
authorized worker without adding its SDK or runtime as a dependency. Onshape
FeatureScript authoring uses the official Onshape Labs MCP in the user's client;
the exported file then enters Geometry Agent through normal upload. An optional
local export-only helper can snapshot an Onshape workspace and produce an
immutable source bundle using API credentials configured outside the agent
conversation; those credentials never enter the service. Other systems can
implement the public `GeometryAuthoringProvider` contract.

For mesh segmentation, the CLI freezes the source and references into a new
run and launches a fresh child-agent session:

```bash
content-workflow-cli mesh-segmentation run \
  --asset path/to/fused_asset.usd \
  --target-prim /World/FusedMesh \
  --target-semantic-part wheel \
  --reference-dir path/to/references \
  --output-dir runs/mesh-segmentation-wheel
```

Repeat `--target-semantic-part` for multiple known targets. Use `--dry-run` to
inspect the frozen request and staged evidence before launching the child.

### 5.2 Conversion and Materials

Convert a supported source asset when a workflow requires USD input:

```bash
content-workflow-cli convert-to-usd path/to/source.step \
  --output-format usdc \
  --converter-timeout 600
```

The converter timeout defaults to 120 seconds and must be a positive finite
number. Durable runs freeze it in their request and reject a different value on
`--resume`.

Assign materials from reference images and a material library:

```bash
content-workflow-cli materials assign \
  --usd path/to/asset.usdc \
  --reference-image path/to/reference.png \
  --materials-yaml apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --output-dir runs/materials-my-asset
```

Use `content-workflow-cli scene run --task material` instead when a composed
scene needs decomposition, per-asset processing, and collection back onto the
original topology.

### 5.3 Textures

For new Texture work, start an interactive coding-agent session and use
`content-workflow-texture`. The outer reasoner selects the necessary focused
operations from `texture prepare` through `texture publish`, preserves explicit
material or prim scope, and owns semantic review.

`content-workflow-cli texture run` now defaults to one bounded skill-routed
attempt with provider-neutral preparation, a plan-only coding-agent child, and
matched OVRTX review. Generated units pause for an exact coding-agent companion
image handoff when the outer coordinator exposes image generation; `--runner`
selects the plan/review child and does not add that outer capability. A Claude
coordinator without an image generator must instead select a service using both
`--texture-backend <name>` and `--texture-agent-url <url>` on a fresh run.
Preserve and apply-provided units do not construct a generation provider, VLM
client, or Texture service. Use `--execution-mode fixed` only for the legacy
service-backed compatibility launcher. See the
[CLI reference](packages/content_workflow_cli/README.md) for scope, handoff,
resume, and compatibility arguments.

### 5.4 Physics

Author physics properties and collect runtime evidence:

```bash
content-workflow-cli physics apply \
  --usd apps/physics_agent/data/examples/Lightbulb01/light_bulb_01.usda \
  --output-dir runs/lightbulb-physics
```

Runtime validation is enabled by default. `--no-simulation` retains
schema-authoring artifacts but produces a conditional, non-passing result; it
does not satisfy the workflow's required runtime and visual evidence.

### 5.5 Articulation

Infer joint candidates and bind authoring to explicit review decisions:

```bash
content-workflow-cli articulation run \
  --usd path/to/asset.usdz \
  --joint-config path/to/joint.yaml \
  --intent "Present candidates for review, then author approved joints." \
  --output-dir runs/articulation
```

Inspect the reported candidates and scene evidence, then submit one
decision for every review-required candidate with
`content-workflow-cli articulation review`. Resume an interrupted run with
`content-workflow-cli articulation resume --run-dir runs/articulation`.

### 5.6 Validation and SimReady

Validate an asset against a task and optional reference evidence without
modifying the asset:

```bash
content-workflow-cli validate run \
  --usd path/to/asset.usdc \
  --task "Validate that the asset renders and matches the reference." \
  --reference-image path/to/reference.png \
  --output-dir runs/validation \
  --runner codex \
  --model gpt-5.6-sol \
  --render-backend remote
```

Validation does not select a planning provider or rendering backend by default.
Choose `--runner` and `--model` explicitly. Claude also requires an explicit
`--claude-execution-mode`; pass `--render-backend` only when a selected check
requires current rendering.

When an interactive outer reasoner owns the selected checks, use the focused
`content-workflow-cli validate prepare` path. Its related operations include
`ingest-verified-operation-result` for trusted external results and
`review-assessment` for the final evidence-bound semantic review.

Validate the current USD against the SimReady profile:

```bash
content-workflow-cli simready validate-profile path/to/asset.usda \
  --report runs/simready/profile.json
```

Run `simready conform-profile` only when the validation report contains failed
requirements, then validate the returned `output_usd_path` again. Use
`validate resume` or the relevant `simready` subcommand help for recovery and
strict-exit behavior.

### 5.7 Scene and Composed Asset Workflows

Process a composed scene through one durable scene-level run:

```bash
content-workflow-cli scene run \
  --usd path/to/scene.usd \
  --task material \
  --materials-yaml path/to/materials.yaml \
  --reference-dir path/to/references \
  --output-dir runs/scene-material
```

Use the composed asset workflow when Joint, Material, Texture, Physics, and
Validation must operate on the same asset in order:

```bash
content-workflow-cli asset run \
  --usd path/to/asset.usdz \
  --prompt "Review joints, assign materials, add physics, validate, and package the asset." \
  --joint-config path/to/joint.yaml \
  --materials-yaml path/to/materials.yaml \
  --output-dir runs/composed-asset
```

The asset workflow pauses at the Joint review gate. Use `asset review` with
one decision per required candidate, and `asset resume` after an interruption.

## 6. Review and Resume

Every workflow writes a self-contained run folder under the supplied
`--output-dir`. Start with its final summary, authored USD, render or simulation
evidence, validation result, and current status. The exact filenames vary by
workflow, but the run should answer:

- What inputs and request were frozen?
- What did the agent change?
- What evidence supports the result?
- Which checks passed, failed, or remain unresolved?
- How can the run be resumed or diagnosed?

Use the workflow-specific `resume` command only with the original run
directory. Durable requests, checkpoints, and accepted artifacts are
digest-bound; changing source inputs or policy generally requires a new run.
Preserve failed and cancelled runs as diagnostic records.

Unresolved visual or physical quality can be an honest workflow outcome rather
than a process error. Read the final status and evidence before deciding
whether to refine the request, repair an input, or start a new run.

## 7. Architecture and Reference

The Agentic backend uses these terms consistently:

- **Coding-agent runtime:** interprets intent, plans work, reviews evidence,
  and decides whether to refine or stop.
- **Skill:** owns capability policy, evidence requirements, and decision
  procedures.
- **Workflow:** owns run state, budgets, checkpoints, artifacts, recovery, and
  completion criteria.
- **Typed tool:** performs a deterministic operation such as inspection,
  conversion, authoring, rendering, simulation, validation, or export.

The layering is workflow-first: `content-workflow-cli` may launch the coding
agent, the selected `content-workflow-*` skill owns workflow policy, and
`usd-cli` plus the relevant typed libraries perform scene operations.
The fixed pipeline is a separate explicitly selected backend, not a fallback.

The public skills are `content-workflow-geometry`,
`geometry-source-intake`, `geometry-evidence-review`,
`content-workflow-geometry-repair`, `product-image-decomposer`,
`content-articulation-authoring`,
`content-articulation-inspection`, `content-articulation-proposal`,
`content-articulation-review`, `content-texture-candidate`,
`content-texture-publish`, `content-texture-quality`, `content-texture-scope`,
`usd-cli`, `content-workflow-articulation`, `content-workflow-cli`,
`content-workflow-asset`,
`content-workflow-convert-to-usd`,
`content-workflow-mesh-segmentation`, `content-workflow-material`,
`content-workflow-validation`, `content-workflow-physics`,
`content-workflow-physics-external-tuning`, `content-sdf-operations`,
`content-workflow-simready`, `content-workflow-large-scene`,
`content-workflow-scene-decomposition`,
`content-workflow-asset-task-processing`, `content-workflow-scene-collection`,
`content-workflow-texture`, and `image-generation`.

Further reference:

- [Content Workflow CLI](packages/content_workflow_cli/README.md)
- [`usd-cli`](../apps/usd_cli/README.md)
- [Geometry authoring connectors](packages/geometry_authoring_connectors/README.md)
- [Geometry Agent service](../apps/geometry_agent_service/README.md)
- [SDF tools](packages/sdf_tools/README.md)
- [Validation capability matrix](docs/validation_capability_matrix.md)
