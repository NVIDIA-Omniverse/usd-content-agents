# Material Refinement and Variation

> Compatibility workflow: new agentic and fixed integrations should plan and
> judge outside the authoring core, then publish through
> `MaterialAuthoringRequest` and `author_material_package`. See the
> [material authoring architecture](internal/material_authoring_architecture.md).
> The commands in this guide remain supported while callers migrate.

Material Agent can refine one material toward a requested appearance or create
a set of visually distinct variations. Both paths use the same one-material
workflow:

1. Materialize the supplied source state or clone its authored USD material
   graph as a canonical UV-mapped mesh swatch without changing representation.
2. Build initial source-anchored bounds for base color, roughness, and metallic
   response.
3. Ask the configured model to infer normalized internal PBR controls from the
   appearance prompt. Expand the source-local search space with a bounded window
   around that semantic target. These controls are evidence, not user input.
4. Evaluate the inferred target. Scalar-PBR refinement authors that analytically
   known target directly; variation slots with an existing diversity constraint
   use the configured optimizer to search for a distinct scalar result.
   Textured-PBR trials run the fresh `run_tuning` budget through the configured
   optimizer, call the Texture Variation API, and retain raw maps.
5. Score each candidate from its authored scalar values or measured maps.
6. Render only the best valid trial through Material Agent's canonical swatch.
7. Ask the configured Material Agent VLM to approve the render or provide
   feedback, scoring only properties that the current representation can edit.
8. On rejection, translate the visual feedback into a normalized target, expand
   the active bounds with a target-centered bounded window, revise the generation
   prompt, and begin another fresh sweep through `run_refinement`.

Variation-set orchestration invokes that exact workflow once per requested
slot. Approved feature measurements and renders become diversity evidence for
the next slot. The outer layer does not own another optimizer or generator, and
every selected slot keeps the source representation.

## Representation Preservation

Refinement does not implicitly convert between scalar and textured materials:

- `source` without `textures` is `scalar_pbr`. Candidate and final USD files
  contain scalar PBR inputs and no texture files or texture shader nodes.
- `source.textures` is `textured_pbr`. It requires an explicit albedo, normal,
  and ORM set; candidate and final packages remain textured.
- `source.source_usd` plus `source.material_prim_path` selects an existing
  OpenPBR MaterialX, scalar OmniPBR MDL, or UsdPreviewSurface graph. Its
  profile, scalar and optical values, and supported texture graph are detected
  from USD.
  Every candidate copies that graph and changes only the selected inputs or
  copied texture assets.
- Source-backed candidates preserve shader IDs and connection topology. The
  source USD is hashed before and after authoring and is never modified.
- `source.material_profile` must match the detected graph profile. From-scratch
  scalar and texture-map inputs continue to use the material library writer.

`source` describes the existing material. `goal` contains exactly one user
input, `appearance_prompt`; refinement rejects explicit color, roughness,
metallic, weight, and reference-image fields. It does not accept a generation
recipe or a second appearance prompt.

```yaml
goal:
  appearance_prompt: >-
    Satin cobalt-blue painted steel with soft broad highlights and a physically
    plausible dielectric response.
```

Scalar input:

```yaml
source:
  material_id: cobalt_painted_steel
  name: Cobalt Painted Steel
  material_profile: preview_surface
  base_color: [0.10, 0.24, 0.68]
  roughness: 0.40
  metallic: 0.0
```

Textured input:

```yaml
source:
  material_id: cobalt_painted_steel
  name: Cobalt Painted Steel
  material_profile: preview_surface
  base_color: [0.10, 0.24, 0.68]
  roughness: 0.40
  metallic: 0.0
  textures:
    albedo: ./input/albedo.png
    normal: ./input/normal.png
    orm: ./input/orm.png
```

Existing OpenPBR input:

```yaml
source:
  material_id: aluminum_openpbr
  name: Aluminum OpenPBR
  source_usd: ../data/materials/material_libs_default/materials_libs_v2.usd
  material_prim_path: /World/Looks/Aluminum
  material_profile: openpbr_materialx
```

Existing PreviewSurface input:

```yaml
source:
  material_id: neutral_painted_steel
  name: Neutral Painted Steel
  source_usd: ../data/materials/material_refinement_examples/preview_surface.usda
  material_prim_path: /World/Looks/NeutralPaintedSteel
  material_profile: preview_surface
```

Existing scalar OmniPBR MDL input:

```yaml
source:
  material_id: neutral_satin_polymer_mdl
  name: Neutral Satin Polymer MDL
  source_usd: ../data/materials/material_refinement_examples/omnipbr_mdl.usda
  material_prim_path: /World/Looks/NeutralSatinPolymer
  material_profile: omnipbr_mdl
```

Graph-backed textured refinement currently requires a discoverable albedo,
normal, and shared ORM graph. Unsupported partial or custom texture graphs fail
before optimization instead of disconnecting or replacing their nodes. MDL
source refinement is intentionally narrower: it accepts only scalar
`OmniPBR.mdl:OmniPBR` graphs with authored color, roughness, and metallic
constants. Texture-backed OmniPBR and arbitrary MDL modules fail closed.

## Texture Variation Boundary

Candidate generation uses the existing Texture Variation API:

```text
POST   /v1/texture-variation-assets?filename={filename}
GET    /v1/texture-variation-assets/{asset_id}/{filename}
POST   /v1/texture-variations
GET    /v1/texture-variations/{job_id}
GET    /v1/texture-variations/{job_id}/artifacts/{artifact_path}
DELETE /v1/texture-variations/{job_id}
```

Before submission, Material Agent packages the source swatch and its USD
dependencies as USDZ, uploads that package and any reference images, and sends
the returned service-owned HTTP(S) URIs in the job request. Completed jobs
publish generated artifacts through the same endpoint origin. No shared host
filesystem or container mount is assumed. Uploads and downloads are bounded by
`variation.max_artifact_bytes` on the client; the bundled service also applies
a bounded upload limit and rejects unsafe filenames and artifact paths.

Textured trials vary six standardized controls: `base_color_r`,
`base_color_g`, `base_color_b`, `roughness`, `metallic`, and
`configuration.strength`. Scalar trials vary only the five PBR controls and do
not provision or call a Texture Variation service. Initial RGB and PBR bounds
are centered on source values or measured source maps. The configured model may
infer any normalized PBR target in `[0, 1]` from the appearance prompt. The same
`search` deltas define a target-centered proposal that is merged into the active
space before tuning and after each visual rejection. This permits explicit
appearance requests such as metallic-to-dielectric paint while keeping the
source shader graph and scalar or textured representation intact. Texture-
strength bounds do not expand. `goal.json` always contains only
`appearance_prompt`; inferred targets, raw model responses, and every bound
revision are written separately to `target_inference.json` and the corresponding
attempt evidence. They are never accepted as goal input fields.

The trial-specific RGB/PBR values are included in the Texture Variation prompt.
After the service returns, Material Agent validates and packages the generated
maps without replacing their channel values. The objective measures albedo in
linear light after decoding its sRGB image samples and reads roughness/metallic from
the generated ORM map, so the optimizer scores backend output rather than its
own requested controls. Seed is owned by the workflow. Engine-specific settings
may be supplied under `variation.custom_parameters`, but Material Agent does not
present those as portable controls or assume that every backend supports them.

The REST service must return same-origin HTTP(S) albedo, normal, and ORM
artifacts. Missing channels fail the trial. A configured endpoint is required
unless Python callers inject a `TextureVariationGenerator` implementation;
injected in-process generators may return local artifacts directly.

The REST adapter in this compatibility workflow and Texture Agent's typed REST
client are not yet one implementation. Upload and artifact routes remain a
shared Texture Variation service contract; client convergence is deferred to a
provider-neutral transport in `texture_gen_service_common` so Material Agent
does not depend on Texture Agent. See the internal material-authoring
architecture for the recorded ownership decision and migration criteria.

## CLI

The checked-in examples use scalar PBR and therefore need no Texture Variation
endpoint. For textured input, configure `source.textures` and a reachable
`variation.endpoint`. Run:

```bash
material-agent refine-material \
  apps/material_agent/configs/material_refinement_example.yaml \
  --output-dir "$(mktemp -d)/material-refinement"

material-agent refine-material \
  apps/material_agent/configs/material_refinement_preview_surface_graph_example.yaml \
  --output-dir "$(mktemp -d)/material-refinement-preview-surface"

material-agent refine-material \
  apps/material_agent/configs/material_refinement_openpbr_example.yaml \
  --output-dir "$(mktemp -d)/material-refinement-openpbr"

material-agent refine-material \
  apps/material_agent/configs/material_refinement_omnipbr_mdl_example.yaml \
  --output-dir "$(mktemp -d)/material-refinement-omnipbr-mdl"

material-agent optimize-variations \
  apps/material_agent/configs/material_variation_example.yaml \
  --output-dir "$(mktemp -d)/material-variations"
```

## Python

```python
from pathlib import Path

from material_agent.api import create_material_variations, refine_material

refined = refine_material(
    Path("apps/material_agent/configs/material_refinement_example.yaml"),
    output_dir=Path("/tmp/material-refinement"),
)
variations = create_material_variations(
    Path("apps/material_agent/configs/material_variation_example.yaml"),
    output_dir=Path("/tmp/material-variations"),
)
```

Runnable Python entry points are in
[`examples/material_refinement.py`](../examples/material_refinement.py) and
[`examples/material_variation_optimization.py`](../examples/material_variation_optimization.py).

## Evidence and Publication

One-material output contains copied goal evidence, the representation-preserved
source package, source swatch, JSONL trial history, winner-only render, VLM
verdict, selected material package, and summary. Textured runs additionally
retain raw service maps and validated candidate maps. Scalar runs contain no
texture artifacts in their published material package.

The canonical swatch uses deterministic dense tessellation. The visual rubric
excludes fixed fixture properties such as swatch geometry, camera, and lighting.
Every refinement iteration receives a fresh visual verdict.

That VLM verdict is diagnostic-grade selection inside this compatibility loop.
It is not the agentic material-authoring workflow's acceptance decision. A fixed
or composed workflow consuming the package must make its own outer disposition;
`completed` here means only that the compatibility loop published its selected
candidate.

A completed variation set additionally publishes `final/material_library.usda`,
`final/materials.yaml`, `final/material_variation_plan.yaml`, and
`material_variation_manifest.json`. Incomplete and cancelled sets retain slot
evidence and the top-level manifest but do not publish a partial final library.

The default `optimization.name: auto` resolves to BoTorch and never silently
falls back to random search. Install `material-agent[refinement]` before
running refinement or variation workflows. Random search remains available
only when explicitly selected with `optimization.name: random`.
