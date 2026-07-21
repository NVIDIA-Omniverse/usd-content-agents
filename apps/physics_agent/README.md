# Physics Agent

VLM-based classification agent for 3D assets that identifies component types, surface materials, and physical properties for physics simulation.

## Overview

Physics Agent processes USD files to classify components using Vision-Language Models. Given rendered views of an asset, it predicts:

- **Component type** (e.g., wheel, chassis, sensor housing)
- **Surface material** (e.g., rubber, steel, plastic)
- **Physical properties** (e.g., mass estimate, friction class, rigidity)

Results are structured for downstream physics simulation pipelines.

## Prefer the REST service?

This README covers the `physics-agent` CLI (Option B in the root [README](../../README.md#three-ways-to-use-content-agents)). If you'd rather drive the same pipeline over HTTP with session management and progress streaming, see [`../physics_agent_service/`](../physics_agent_service/) — it brings up with a single `docker compose up`.

## Installation

From the repository root:

```bash
uv pip install -e .
uv pip install -e apps/physics_agent

# Optional: production auto-tuning dependencies (BoTorch / OvPhysX path)
uv pip install -e "apps/physics_agent[tuning]"
```

## Rendering

Physics whole-asset preview and per-prim dataset steps use the same backend
names and fail with a configuration error for any unknown value:

| Backend | Semantics |
|---|---|
| `remote` | Render through an HTTP service. Set `RENDER_ENDPOINT`; the bundled `physics_agent_service` points this at its OVRTX sidecar. |
| `ovrtx` | Render locally with the isolated OVRTX RTX subprocess. Requires a compatible NVIDIA GPU and driver. |
| `warp` | Render locally with the optional CUDA/Warp backend. Install the root `warp` extra. |
| `mock` | Produce deterministic CPU-only images for simulation and CI. These are not production visual evidence. |

Set the same name at `steps.identify_asset.renderer.backend` and
`steps.build_dataset_usd.renderer.backend`. The shipped `lightbulb.yaml` uses
`ovrtx` for both steps.

## Quick Start

```bash
source .venv/bin/activate
physics-agent run apps/physics_agent/configs/lightbulb.yaml
```

## Where outputs land

Every run writes into a single **working directory** placed next to the config file. By default the directory is `.{session_id}` (a hidden folder), where `session_id` comes from `project.session_id` in the config. To override the directory name, set `project.working_dir` to a simple child path of the config directory — e.g. `working_dir: ".my_run"` — without `..` segments.

> ⚠️ The working directory must be a dedicated, pipeline-owned directory. `physics-agent run --clean` recursively deletes the resolved `working_dir` before the run; a `..` segment or a path that escapes the config directory can wipe unrelated files. Never point `working_dir` at a directory that holds files you care about.

For the bundled `lightbulb.yaml` (`session_id: lightbulb`), the working directory is `apps/physics_agent/configs/.lightbulb/`, with this layout:

```text
apps/physics_agent/configs/.lightbulb/
├── identification/                              # whole-asset preview + identification
├── dataset/
│   └── usd/                                     # per-prim renders
├── predictions/
│   ├── predictions.jsonl                        # VLM classifications (component, material, physics)
│   └── report.html                              # HTML report
└── physics/
    └── light_bulb_01_physics.usda               # simulation-ready USD (apply_physics output)
```

The simulation-ready USD is at `<working_dir>/physics/<input-stem>_physics<output-ext>`, where `<input-stem>` is the input USD filename without its extension. The output extension preserves `.usd`, `.usda`, and `.usdc` inputs; `.usdz` inputs default to `.usda` so Omniverse MDL shader references can remain as runtime-resolved asset paths instead of being bundled into a new USDZ package. Package-local asset dependencies from the source USDZ are copied beside the USDA output and rewritten to relative paths when referenced. For the bundled `lightbulb.yaml`, `light_bulb_01.usdz` produces `light_bulb_01_physics.usda`. This default applies to unified pipeline autowiring; lower-level `apply_physics` calls with an explicit `.usdz` output path still write USDZ when the host can resolve every referenced asset. It is the input USD with `UsdPhysics.RigidBodyAPI` / `CollisionAPI` / `MassAPI` / `MaterialAPI` schemas applied to each predicted prim. Under the default `mass_scale_policy: skip_mass`, scale-driven mass estimates omit `MassAPI.mass` while still authoring density and collision/material properties.

## CLI Reference

```bash
# Run complete pipeline
physics-agent run CONFIG

# Pipeline options
physics-agent run CONFIG --skip build_dataset_usd    # skip a step
physics-agent run CONFIG --only predict              # run specific steps
physics-agent run CONFIG --resume                    # resume from checkpoint
physics-agent run CONFIG --dry-run                   # show execution plan
physics-agent run CONFIG --clean                     # wipe working dir first
physics-agent run CONFIG -v                          # verbose logging

# Individual commands
physics-agent predict CONFIG                         # VLM prediction only
physics-agent build-dataset usd CONFIG               # Build dataset from USD
physics-agent build-dataset prepare-dataset CONFIG   # Prepare dataset for VLM

# Physics auto-tuning over an apply_physics output USD
physics-agent tune apps/physics_agent/configs/tuning/drop_settle.yaml \
  --physics-usd path/to/asset_physics.usda \
  --engine ovphysx \
  --optimizer auto

# Prompt-authored scenario; no YAML required
physics-agent tune \
  --user-prompt "make this object bouncy" \
  --physics-usd path/to/asset_physics.usda \
  --engine ovphysx

# Iterative tune -> judge -> scenario refinement
physics-agent refine apps/physics_agent/configs/tuning/drop_settle.yaml \
  --physics-usd path/to/asset_physics.usda \
  --user-prompt "match this observed motion" \
  --max-iterations 3
```

## Configuration

Pipeline configs are YAML files under `configs/`. Use `lightbulb.yaml` as a reference:

```yaml
project:
  name: "my_asset"

input:
  usd_path: "path/to/asset.usd"

predict:
  vlm:
    backend: nim                    # or: openai, anthropic, gemini
    model: qwen/qwen3.5-397b-a17b
```

Paths in config files are relative to the config file's directory.

## Documentation

- **[API Reference](docs/api.md)** -- Python API reference
- **[Auto-Tuning Guide](docs/tuning.md)** -- architecture, extension points, CLI modes, examples, and service integration status
