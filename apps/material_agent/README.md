# Material Agent

A Vision-Language Model (VLM) based system for intelligent material assignment to 3D-rendered object parts. The Material Agent analyzes visual characteristics of object components and assigns appropriate materials from a material library, enabling automated material selection for 3D modeling and rendering workflows.

> **Content Agents 0.6:** this package is the fixed pipeline, config-driven material
> workflow. For an unqualified material-authoring task, start from the repository
> root with the default agentic Content Workflow described in the root
> [README](../../README.md#choose-an-execution-mode). Use `material-agent` when
> you explicitly need fixed pipeline steps, YAML configuration, benchmarking,
> Python APIs, or the matching REST contract.

## Overview

The Material Agent addresses a fundamental challenge in 3D content creation: accurately identifying object parts and assigning suitable materials based on visual analysis. By leveraging Vision-Language Models, it can:

- **Identify object parts** from multi-view 3D renders
- **Understand visual characteristics** through geometric and structural analysis
- **Select appropriate materials** from a provided material library
- **Provide reasoning** for material choices based on functional requirements
- **Orchestrate complete workflows** through a unified pipeline command

### Key Features

- VLM-powered material assignment with multi-view analysis
- Pipeline orchestration with automatic data flow between steps
- Material library matching with fuzzy validation
- Scene pipeline for large multi-asset USD files
- Specification evidence that can corroborate visual material predictions or
  flag conflicts for review without overriding the visual result
- Batch processing with parallel execution
- Checkpointing and resume from failures
- USD instance handling for cost savings and consistency
- Optional image-based prim clustering for large scenes with repeated parts
- Representation-preserving rendered material refinement and judged variation
  sets, with Texture Variation used only for textured sources

Technical specification text and converted PDF pages stay outside the visual
model prompt. After the visual material label is selected, extracted material
claims can corroborate it or flag a conflict for review; they cannot introduce
or replace the visual label.

## Prefer the REST service?

This README covers the explicit fixed pipeline `material-agent` CLI described in the
root [execution-mode guide](../../README.md#choose-an-execution-mode). If you'd
rather drive the same fixed pipeline over HTTP with session management and
progress streaming, see [`../material_agent_service/`](../material_agent_service/)
— it brings up with a single `docker compose up`.

## Optional Prim Clustering

For large scenes with many visually repeated prims, the pipeline can cluster
rendered prim-only images before VLM prediction, predict only representative
prims, then expand those predictions back to cluster members. Keep it disabled
for small scenes or assets where small visual differences require separate
material decisions.

Enable it in unified config by adding a `cluster_prims` step before `predict`
and `expand_cluster_predictions` after `predict`, or use the REST service
fields documented in `apps/material_agent_service/docs/api.md`. The public
embedding default is NVIDIA NIM
`nvidia/llama-nemotron-embed-vl-1b-v2`; hosted use requires `NVIDIA_API_KEY`,
and local Docker deployments can route the same model to the optional
embedding sidecar. Use `max_cluster_size` to cap how many prims can inherit one
representative prediction.

## Installation

From the repository root:

```bash
# Install the core library first
uv pip install -e .

# Install the material agent
uv pip install -e apps/material_agent
```

For development:

```bash
uv pip install -e "apps/material_agent[dev]"
```

### Scene Optimizer (required for the `optimize_usd` pipeline step)

The first pipeline step, `optimize_usd`, runs a local Scene Optimizer
subprocess. Before the first CLI run, fetch the public Scene Optimizer
Core package (~332 MB, one-time):

```bash
./scripts/fetch_build_resources.sh
```

The agent auto-discovers the unpacked package at
`.build-resources/scene_optimizer_core/` when invoked from the repo root.
To point at a different location, set `WU_SO_PACKAGE_DIR`.

(Users of the `material_agent_service` docker-compose stack don't need to
do this separately — the Dockerfile runs the fetch during `docker compose
build`.)

**Escape hatches** (only when you really don't want the local fetch):

- `optimize_usd.enabled: false` in your config — skips the step entirely.
  Use for fast evaluation or when working with already-flat USD assets.
- `optimize_usd.backend: remote` + `NVCF_OPTIMIZER_FUNCTION_ID` +
  `NGC_API_KEY` — runs optimization on a remote NVCF function instead.

## Environment Setup

Copy `.env_example` to `.env` and add your VLM provider API key:

```bash
cp .env_example .env
```

You only need a key for the backend you plan to use:

| Backend | Environment Variable | Provider |
|---------|---------------------|----------|
| `nim` | `NVIDIA_API_KEY` | [NVIDIA NIM](https://build.nvidia.com/) |
| `openai` | `OPENAI_API_KEY` | [OpenAI](https://platform.openai.com/) |
| `anthropic` | `ANTHROPIC_API_KEY` | [Anthropic](https://console.anthropic.com/) |
| `gemini` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | [Google Gemini](https://aistudio.google.com/) |

The shipped `configs/unified_example.yaml` defaults to NVIDIA NIM for both
VLM and LLM parsing (`predict.vlm.backend: nim` and
`predict.llm.backend: nim`), so the unedited example requires
`NVIDIA_API_KEY`. To use another provider, set matching backend/model
overrides in `.env` or edit those YAML fields directly:

```bash
MA_VLM_BACKEND=openai
MA_VLM_MODEL=example-vlm-model
MA_LLM_BACKEND=openai
MA_LLM_MODEL=example-vlm-model
```

### Rendering Backends

Material preview, per-prim dataset, and final-render steps use the same backend
names and fail with a configuration error for any unknown value:

| Backend | Semantics |
|---|---|
| `remote` | Render through an HTTP service. Configure `RENDER_ENDPOINT`, or `NVCF_RENDER_FUNCTION_ID` for NVCF; the bundled `material_agent_service` exposes its OVRTX sidecar at `http://localhost:8001`. |
| `ovrtx` | Render locally with the isolated OVRTX RTX subprocess. Requires a compatible NVIDIA GPU and driver. |
| `warp` | Render locally with the optional CUDA/Warp backend. Install the root `warp` extra. |
| `mock` | Produce deterministic CPU-only images for simulation and CI. These are not production visual evidence. |

Set the name under `steps.render_preview.renderer.backend` or
`steps.build_dataset_usd.renderer.backend`; the final render uses
`steps.render.backend`. The shipped `unified_example.yaml` uses `remote`, so
either `RENDER_ENDPOINT` or `NVCF_RENDER_FUNCTION_ID` must be configured for an
unedited run. For an NVIDIA Cloud Functions renderer, the function ID supplies
the endpoint;
set `NGC_API_KEY` when that endpoint requires NVCF bearer authentication.

## Quick Start

### Run the Example Pipeline

The fixed-pipeline Material Agent CLI runs on Linux or under WSL2 on Windows.
Native Windows fixed-pipeline execution is not supported; see the repository
[platform policy](../../README.md#platform-support).

On native Linux, the unedited example uses remote rendering:

```bash
source .venv/bin/activate
# The unedited config uses backend: nim and requires NVIDIA_API_KEY.
# Configure RENDER_ENDPOINT, or NVCF_RENDER_FUNCTION_ID when using NVCF.
material-agent run apps/material_agent/configs/unified_example.yaml
```

On WSL2, install and initialize Warp, copy the example beside the original, and
set both `steps.build_dataset_usd.renderer.backend` and
`steps.render.backend` to `warp` in the copy before running it:

```bash
source .venv/bin/activate
uv pip install -e ".[warp]" -e apps/material_agent
python -c "import newton; import warp as wp; wp.init()"
CONFIG_DIR=apps/material_agent/configs
cp "$CONFIG_DIR/unified_example.yaml" "$CONFIG_DIR/unified_example_wsl2.yaml"
# Edit both renderer fields listed above to warp.
material-agent run "$CONFIG_DIR/unified_example_wsl2.yaml"
```

### Refine a Material or Create Variations

Material Agent can run a fresh bounded optimizer sweep over source-anchored
color, roughness, and metallic response without changing representation.
The user-facing `goal` contains only `appearance_prompt`; the configured model
infers bounded internal PBR controls and records them as output evidence.
Scalar-PBR sources emit scalar-PBR outputs with no textures. Textured sources
retain raw Texture Variation maps and emit textured outputs. The workflow
renders and judges only the sweep winner, then uses VLM feedback for the next
attempt. Variation sets reuse that core and carry approved evidence between
slots. Input is an existing `source` material state plus one `goal`; generation
recipes are not part of the refinement contract. A source can also identify an
existing OpenPBR MaterialX or UsdPreviewSurface material with `source_usd` and
`material_prim_path`. That path clones and edits the authored graph without
mutating the source or invoking material-graph generation. These public
examples use `nim` with `moonshotai/kimi-k3`; their scalar inputs require no
Texture Variation endpoint:

```bash
material-agent refine-material \
  apps/material_agent/configs/material_refinement_example.yaml

material-agent refine-material \
  apps/material_agent/configs/material_refinement_preview_surface_graph_example.yaml

material-agent refine-material \
  apps/material_agent/configs/material_refinement_openpbr_example.yaml

material-agent optimize-variations \
  apps/material_agent/configs/material_variation_example.yaml
```

The default `optimization.name: auto` resolves to BoTorch and never silently
falls back to random search. Install `material-agent[refinement]` before
running these workflows. Random search remains available only when explicitly
selected with `optimization.name: random`. See the
[refinement and variation guide](docs/material_refinement_and_variation.md) for
the Python API, output evidence, cancellation behavior, and publication rules.

### Try with Public SimReady Assets

Once the shipped ladder example runs, you can try the same pipeline on
assets from NVIDIA's public SimReady catalogs. Download a curated prop
and run the pipeline on it with the dataset's thumbnail as the reference
image:

```bash
# Example: steel rolling scaffold from the HuggingFace SimReady Warehouse
pip install -U huggingface_hub
hf download --repo-type dataset nvidia/PhysicalAI-SimReady-Warehouse-01 \
  --include "Props/general/SM_SteelRollingScaffold_A01_01/*" \
  --local-dir ~/content-agents-data/simready/hf/

# Copy the example config, then edit input.usd_path / reference_images
# to point at the downloaded prop (absolute paths — ~ is not expanded
# by the config loader). Because this copy lives at the repository root,
# also set materials.path to the repository-root-relative manifest below:
cp apps/material_agent/configs/unified_example.yaml my_simready_scaffold.yaml
# materials.path: apps/material_agent/data/materials/material_libs_default/materials.yaml
# ...edit, then:
material-agent run my_simready_scaffold.yaml
```

Four curated assets (HF: scaffold, cleaning trolley; GitHub
`NVIDIA/simready-foundation`: electricians toolbox, UR10 robot arm) are
documented in
[`../../.agents/skills/fixed-pipeline/references/material-agent-cli/references/simready-quickstart.md`](../../.agents/skills/fixed-pipeline/references/material-agent-cli/references/simready-quickstart.md),
including the VLM key pre-flight probe and the `skip_instances: false`
caveat required for the UR10.

### Create a Configuration

```bash
# Interactive configuration wizard
material-agent configure my_pipeline.yaml

# With a materials manifest
material-agent configure my_pipeline.yaml -m data/materials/materials.yaml

# With reference images
material-agent configure my_pipeline.yaml -m materials.yaml -r ref1.jpg -r ref2.jpg
```

## CLI Reference

### Pipeline Command (Fixed-Pipeline Workflow)

```bash
# Run complete end-to-end pipeline
material-agent run CONFIG

# Skip specific steps
material-agent run CONFIG --skip build_dataset_usd

# Run only specific steps
material-agent run CONFIG --only predict,apply

# Resume from checkpoint after failure
material-agent run CONFIG --resume

# Dry run to see execution plan
material-agent run CONFIG --dry-run

# Verbose logging
material-agent run CONFIG -v
```

### Individual Step Commands

```bash
material-agent predict CONFIG              # VLM prediction only
material-agent apply CONFIG                # Apply materials to USD
material-agent benchmark CONFIG            # Benchmark with LLM-judge scoring
material-agent evaluate CONFIG             # Evaluate existing predictions
material-agent build-dataset usd CONFIG    # Build dataset from USD
material-agent build-dataset prepare-dataset CONFIG  # Prepare dataset for VLM
material-agent configure CONFIG            # Interactive config creation
material-agent refine-material CONFIG      # Refine one rendered material
material-agent optimize-variations CONFIG  # Create a judged variation set
material-agent generate-manifest USD OUT   # Generate materials.yaml from USD
```

### Scene Pipeline (Large Multi-Asset Scenes)

```bash
# Full end-to-end
material-agent scene run CONFIG --workers 4 -v

# Individual steps
material-agent scene analyze CONFIG -v
material-agent scene extract CONFIG -v
material-agent scene run-agent CONFIG --workers 4 -v
material-agent scene collect CONFIG -v
```

Scene extraction produces standalone per-asset USDs. The optional
`scene.extract.flatten` setting defaults to `true` and must remain `true`;
non-flattened extraction is rejected because USD population masks are not
persisted in exported root layers.

## Pipeline Steps

1. `optimize_usd` -- Flatten/deinstance USD via scene optimizer
2. `render_preview` -- Lightweight whole-scene preview rendering
3. `generate_reference_image` -- Generate photorealistic reference images
4. `build_dataset_usd` -- Render prim views for VLM input
5. `build_dataset_prepare_dataset` -- Prepare dataset entries with material specs
6. `predict` -- VLM inference for material assignment
7. `validate_predictions` -- Validate/repair predicted material names
8. `harmonize_predictions` -- Resolve conflicts for instanced parts
9. `apply` -- Apply predicted materials to USD
10. `render` -- Render final output

## Configuration

Pipeline configs are YAML files under `configs/`. Use `unified_example.yaml` as a template:

```yaml
project:
  name: "my_asset"

input:
  usd_path: "path/to/asset.usd"

materials:
  path: "path/to/materials.yaml"

steps:
  predict:
    vlm:
      backend: nim                  # or: openai, anthropic, gemini
      model: moonshotai/kimi-k3     # model from your chosen provider

  render:
    backend: ovrtx
```

Paths in config files are relative to the config file's directory.

## Documentation

- **[API Reference](docs/api.md)** -- Python API reference
