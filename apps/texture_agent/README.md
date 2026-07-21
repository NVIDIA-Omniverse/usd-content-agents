# Texture Agent

AI-driven texture generation and application for USD assets with OpenPBR,
MaterialX, and MDL-style material metadata.

## Overview

The Texture Agent takes a USD file with materials already assigned (e.g., output of the Material Agent) and fills empty texture slots with AI-generated texture maps -- transforming flat, constant-color surfaces into visually rich textured ones.

### Key Features

- Material texture generation for OpenPBR, MaterialX, and MDL-style metadata
  (albedo, roughness, metalness, normal)
- Per-material or per-prim texture modes
- Texture blending and compositing
- Multiple generation backends
- UV readiness reporting with policy-controlled projection UV generation
- Optional projection/reference-image backend path through the Texture Variation
  API service contract

For complete bucket examples on a public asset, including asset download,
request/config fields, pass criteria, reference render/texture evidence, and
local visual evidence sheet generation, see [`examples/`](examples/). Public
0.5 ships the simple image-gen baseline and the service-backend API contract;
it does not ship a managed Step1X runtime package or UV-aware Step1X evidence
workflow.

## Backend Choices

- `simple_image_gen`: lightweight text-to-texture generation through the
  configured image-generation provider. This is the default CLI/service path
  and is the fastest way to verify the Texture Agent pipeline. The public
  default NIM route is text-only and rejects reference, turntable, or
  multi-view conditioning before provider launch; use a backend that advertises
  the requested capability when media conditioning is required.
- `service`: Texture Variation API-compatible backend path for projection,
  reference-image, and UV-aware texture editing. A separately deployed
  Step1X-compatible backend can use this path, but this public release does not
  include a managed Step1X runtime, downloader/setup package, model
  checkpoints, or bucket validation package.

Use the simple image-gen bucket example for a baseline result. For service
backend runs, provide an explicit material/prim scope, point at the deployed
Texture Variation API endpoint, and validate emitted UV/backend diagnostics;
arbitrary USDs may require asset-specific UV or backend configuration.

## Prefer the REST service?

This README covers the `texture-agent` CLI (Option B in the root
[README](../../README.md#three-ways-to-use-content-agents)). If you'd rather drive the same
pipeline over HTTP with session management and progress streaming, see
[`../texture_agent_service/`](../texture_agent_service/). Its default Compose
stack starts the Texture Agent service only; optional service-backend runs
target an operator-provided Texture Variation API endpoint.

## Installation

From the repository root:

```bash
uv pip install -e .
uv pip install -e apps/texture_agent
```

## Rendering

Two pipeline steps render USD views through the shared rendering-backend
contract:

- `render_previews` — renders the current state of each material for VLM-based prompt generation.
- `render` — final render of the textured output.

Both default to `backend: remote`. Texture Agent supports the `remote`, `ovrtx`,
and `mock` USD rendering backends on these steps. The canonical `warp` backend
is not supported here because it does not preserve the textured material
evidence these tasks produce. Options:

- **Point at a running OVRTX rendering API** — export `RENDER_ENDPOINT=http://localhost:8001` (the port exposed by the bundled `material_agent_service` / `physics_agent_service` sidecars, or a standalone OVRTX deployment).
- **Use an NVCF-hosted function** — set `NVCF_RENDER_FUNCTION_ID` and `NGC_API_KEY` instead of `RENDER_ENDPOINT`.
- **Render with local OVRTX** — set `backend: ovrtx` and provide the local OVRTX runtime/GPU prerequisites.
- **Run a CPU-only rendering smoke test** — set `backend: mock`. Mock images are deterministic placeholders for pipeline testing and are not production visual evidence of texture quality.
- **Skip the rendering steps** — use `--skip render_previews,render`, or disable them in the config's `steps.render_previews.enabled` / `steps.render.enabled`. Texture generation and application still run; you just don't get previews or a final composite.

## Quick Start

```bash
source .venv/bin/activate
texture-agent run apps/texture_agent/configs/texture_example.yaml
```

## CLI Reference

```bash
# Run complete pipeline
texture-agent run CONFIG

# Pipeline options
texture-agent run CONFIG --skip render_previews                      # skip a step
texture-agent run CONFIG --only generate_textures,apply_textures     # run specific steps
texture-agent run CONFIG --resume                                    # reuse existing artifacts
texture-agent run CONFIG --session-id previous-run                   # reuse a session directory
texture-agent run CONFIG --dry-run                                   # show execution plan
texture-agent run CONFIG --verbose                                   # verbose logging

# Individual commands
texture-agent discover CONFIG        # Discover materials in the scene
texture-agent generate CONFIG        # Generate textures only
texture-agent apply CONFIG           # Apply textures to USD only
```

To resume after a partial local run, use the same config and either rerun
`texture-agent run CONFIG --resume` or split the last stages explicitly:
`texture-agent generate CONFIG` writes generated/blended texture artifacts,
and `texture-agent apply CONFIG` reloads those artifacts from the config's
working directory before writing the textured USD.

## Pipeline Steps

1. `prepare_uvs` -- Inspect UVs, preserve valid UVs, and optionally prepare missing UVs
2. `discover_materials` -- Discover and catalog materials in the scene
3. `render_previews` -- Render preview images of the current state
4. `generate_textures` -- Generate texture images via the configured backend
5. `blend_textures` -- Blend generated textures (e.g., albedo compositing)
6. `apply_textures` -- Apply generated textures to USD materials
7. `render` -- Render final output

## Configuration

Pipeline configs are YAML files under `configs/`. Key settings:

```yaml
project:
  name: "my_textured_asset"

input:
  usd_path: "path/to/materialized_asset.usd"

texture:
  backend: simple_image_gen     # or: service
  mode: per_material            # or: per_prim
```

UV behavior is controlled under `texture`:

```yaml
texture:
  uv_policy: generate_missing       # validate, preserve_or_fix, generate_missing, force_projection
  uv_scope: stage                   # stage or target_prims
  uv_generation_mode: projection    # SO only: projection or explicit atlas
  uv_projection: box                # box or planar for Python projection
  uv_normalize_out_of_range: false  # preserve tiled UVs by default
  uv_rebake_source_albedo: false    # opt-in source-albedo rebake for scoped service img2img
```

`prepare_uvs` writes `prepared/uv_report.json` in the run directory so missing,
invalid, repaired, generated, and out-of-range UV conditions are inspectable.
Use `uv_scope: target_prims` when forced projection should apply only to
geometry prims listed in `material_textures.<name>.prim_paths` or
`texture.uv_target_prim_paths`.
Scene Optimizer atlas unwrap is opt-in with `uv_backend: scene_optimizer` and
`uv_generation_mode: atlas`; authored UVs are still preserved unless overwrite
is explicitly requested.
For trim-sheet assets, combine that scoped UV mode with
`uv_rebake_source_albedo: true` when the service backend should preserve the
original albedo appearance after target UVs were regenerated.

By default, `material_textures` is also a strict processing scope: materials
not listed there are skipped. Set `auto_prompt.enabled: true` to generate
prompts for discovered materials that are missing explicit specs.

For the Step1X service backend, transparent overlay, decal, label, and sticker
targets are rejected before backend dispatch by default. This avoids sending
thin overlay geometry to Step1X. Set
`texture.custom_parameters.allow_step1x_overlay_targets: true` only when that is
the intended target and the asset has been validated for it.

### Conservative/AOI Detail Policy

Use `texture.detail_policy: surface_only` for AOI, CAD, PCB, and SimReady-style
assets where traces, vias, pads, labels, holes, seams, components, or other
semantic details already exist as modeled geometry. The policy keeps prompts
limited to subtle material variation such as roughness, gloss, low-frequency
color noise, dust, fingerprints, tiny scratches, and mild wear. It applies to
both `simple_image_gen` and `service` backends: simple image generation receives
a sanitized plain-material prompt, service backends also receive policy metadata,
and the selected policy is recorded in the artifact manifest.

```yaml
texture:
  backend: simple_image_gen
  detail_policy: surface_only

material_textures:
  Plastic_Green:
    prompt: "matte green solder mask with subtle roughness variation"
    opacity: 0.45
```

The policy can also be set per material or per prim. For strict AOI runs,
combine `surface_only` with explicit material prompts, lower opacity/strength,
`auto_prompt.enabled: false`, and scoped targets such as
`texture.uv_scope: target_prims` or service `strict_scope: true`.

For one-off CLI runs, override the config globally with
`texture-agent run config.yaml --detail-policy surface_only`.

### Projection/Reference Backend

`texture.uv_projection` controls UV generation for meshes that need UVs. It is
not the same thing as the projection/reference-image texture editing backend.
For projection texture editing, keep UV preparation enabled and route texture
generation through a Texture Variation API-compatible service:

```yaml
texture:
  backend: service
  endpoint: "http://REPLACE_WITH_TEXTURE_VARIATION_ENDPOINT"
  engine: "YOUR_ENGINE_OR_MODEL"
  size: 1024
  workers: 1
  seed: 11631
  strength: 0.85
  strict_scope: true
  reference_image_uris:
    - "file:///absolute/path/reference.png"
  multiview_image_uris: []
  capabilities:
    image_conditioning: true
    multiview: false
    normal_map: true
    orm: true
    masks: true
    coverage: true
    geometry_output: "none"
  custom_parameters:
    run_label: "manual-projection-run"

material_textures:
  Aluminum_Matte:
    prompt: "matte aluminum with light scuffs"
    opacity: 0.85
    reference_image_uris:
      - "file:///absolute/path/aluminum_detail.png"

auto_prompt:
  enabled: false
```

The backend receives the UV-prepared USD, selected material/prim target scope,
prompt/reference conditioning, seed/size/strength, and capability hints. It must
return at least an albedo map. Missing optional normal or ORM maps are recorded
as diagnostics; roughness/metalness maps are packed into ORM when available.
Unsupported conditioning, low coverage, ignored replacement geometry, blank
maps, portability failures, and missing required albedo are surfaced as
structured diagnostics instead of silent success.

Service/client runs use the same fields through `POST /pipeline`:

```bash
BASE_URL="http://localhost:8001"
TEXTURE_VARIATION_ENDPOINT="http://REPLACE_WITH_TEXTURE_VARIATION_ENDPOINT"
BACKEND_ENGINE="YOUR_ENGINE_OR_MODEL"
MATERIAL_TEXTURES='{"Aluminum_Matte":{"prompt":"matte aluminum with light scuffs","opacity":0.85}}'

curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@apps/texture_agent/data/examples/ladder/sources/usd/ladder.usd" \
  -F "auto_prompt_enabled=false" \
  -F "texture_backend=service" \
  -F "texture_endpoint=$TEXTURE_VARIATION_ENDPOINT" \
  -F "backend_engine=$BACKEND_ENGINE" \
  -F 'backend_custom_parameters_json={"run_label":"manual-projection-run"}' \
  -F 'reference_image_uris_json=["file:///absolute/path/reference.png"]' \
  -F "seed=11631" \
  -F "strength=0.85" \
  -F "strict_scope=true" \
  -F "material_textures_json=$MATERIAL_TEXTURES"
```

Or use the bundled Python client:

```python
from apps.texture_agent_service.client.client import TextureAgentClient

client = TextureAgentClient("http://localhost:8001")
session_id, status = client.run_and_monitor(
    usd_path="apps/texture_agent/data/examples/ladder/sources/usd/ladder.usd",
    material_textures={
        "Aluminum_Matte": {
            "prompt": "matte aluminum with light scuffs",
            "opacity": 0.85,
        }
    },
    auto_prompt_enabled=False,
    texture_backend="service",
    texture_endpoint="http://REPLACE_WITH_TEXTURE_VARIATION_ENDPOINT",
    backend_engine="YOUR_ENGINE_OR_MODEL",
    backend_custom_parameters={"run_label": "manual-projection-run"},
    reference_image_uris=["file:///absolute/path/reference.png"],
    seed=11631,
    strength=0.85,
    strict_scope=True,
)
```

For a real backend, provide the deployed Texture Variation API endpoint and
engine/model name. Configure backend credentials on the service or host
environment, make reference-image URIs reachable from the process that calls
the backend, and enable rendering only when a render endpoint is available. The
`fake_projection` backend is test-only and is exercised by the smoke command
below.

### Reviewable Beta Runs

For inspectable review runs, use an explicit `material_textures` scope,
keep `auto_prompt.enabled: false` unless you want discovered materials to be
added, and inspect these outputs before treating the run as successful:

- `prepared/uv_report.json`
- Generated and blended PNG maps in the run directory
- Textured output USD or package
- Service `/pipeline/{session_id}/results` stats when using the REST service
- Final or close-up render when rendering is enabled
- `usd-validation-nvidia` schema (`Basic`, `Layer`, `Layout`, `Other`),
  and `Material` rule coverage for textured USD outputs
- UV readiness in `prepared/uv_report.json`

Real backend runs are useful evidence, but repeatable fake-backend smoke tests
should remain the CI gate for CLI/service parity.

Run the deterministic projection-backend smoke tests locally with:

```bash
source .venv/bin/activate
PYTHONPATH="$PWD:$PWD/apps/texture_agent:$PWD/apps/texture_agent_service" \
  pytest --no-cov \
    apps/texture_agent/tests/test_issue116_projection_backend_smoke.py \
    apps/texture_agent_service/tests/unit/test_issue116_projection_backend_smoke.py
```

### Texture Modes

- **`per_material`** (default) -- One texture set per material, shared across all geometry referencing it.
- **`per_prim`** -- Clones materials per geometry prim, allowing unique textures on each mesh.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Service backend is unreachable | `texture.endpoint` or `texture_endpoint` is missing, or the service is down. | Check the endpoint URL and service health; reduce workers to 1 for fragile endpoints. |
| Backend reports unsupported conditioning | The backend capability metadata says reference images or multi-view inputs are unsupported. | Retry with text-only conditioning or choose a backend that supports the requested media. |
| Projection backend reports missing albedo | The backend did not return the required base-color map. | Treat the run as failed; retry with a compatible backend or inspect `artifacts_manifest.json` diagnostics. |
| Optional maps are missing | Backend returned albedo only, or roughness/metalness without packed ORM. | Texture Agent records degraded channels and synthesizes neutral normal or packs ORM when possible. |
| Low target coverage is reported | Backend coverage metadata says the selected material/prim was not well covered. | Inspect coverage/mask artifacts, use stricter target scope, or provide clearer reference images. |
| Reference image rejected before backend call | URI scheme is unsupported or a local/file URI is missing where the pipeline runs. | Use a local path, `file://`, `http(s)://`, `s3://`, `omni://`, or `omniverse://`; verify local files exist in the CLI or service container. |
| Output is not portable | Texture references are absolute, remote, missing, or outside the output package. | Inspect manifest portability diagnostics and keep `output/` with the generated `textures/` directory. |
| Texture changes apply too broadly | Auto-prompting or non-strict scope submitted additional materials. | Set `auto_prompt.enabled: false` or `auto_prompt_enabled=false`, set `texture.strict_scope: true`, and list only the intended material keys. |
