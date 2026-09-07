# Provider, Render, And Handoff

Read this reference when external authoring, render evidence, or exported
representations affect acceptance.

## Authoring Provider Contract

The public boundary is `GeometryAuthoringProvider`. A provider reports
capabilities and performs bounded `generate`, `revise`, or `export` operations.
Each operation returns a typed receipt bound to the exact request digest.

An accepted `geometry.source.v1` bundle contains:

- stable provider ID and version;
- immutable source revision;
- units, axes, and handedness;
- content-addressed representations with explicit roles;
- optional semantic parts and scalar parameters;
- typed semantic parameter definitions with bounds, step, choices, role, and
  declared effects when the provider supports them;
- request and parent-bundle provenance;
- an affirmative rights assertion for the requested use.

Provider-native source is opaque. The Geometry workflow may retain it as an
inert `native_source` representation, but must never import, evaluate, or
execute it. Provider credentials stay in administrator configuration and must
not appear in requests, manifests, logs, or artifact URIs.

## Reference Providers

| Provider | Public operation | Boundary |
| --- | --- | --- |
| Build123d worker | Text/image generation and bounded revision over HTTP | Worker is independently operated. It returns exported artifacts and typed JSON; the service never executes returned Python. |
| Onshape Labs FeatureScript MCP | Agent-assisted parametric FeatureScript authoring in Onshape | The user's MCP client owns OAuth. Use manual export or the local API-key export helper; neither MCP nor API credentials enter the Geometry Agent service. |
| ForgeCAD authoring worker | Generation and revision through `forgecad-http` | Optional worker requiring explicit automated-use authorization and a rights assertion. The Geometry Agent never invokes a ForgeCAD runtime or CLI. |
| ForgeCAD artifact adapter | Import existing STEP, mesh, or USD exports and optional inert source | Artifact-only. It has no installer, runtime import, or command execution. Operators remain responsible for external licensing and export. |
| Custom provider | Any declared capability | Must implement the same request, receipt, source-bundle, containment, digest, rights, and failure contracts. |

Query capabilities before submitting work. Do not infer support from an
installed package. If a provider cannot perform an operation, preserve the
typed failure. Select another provider only when the user explicitly chooses
it; never install or invoke a local provider runtime as fallback.

For parameterized work, require `semantic_parameters`,
`parameter_definitions`, and `parameter_families` in the selected provider's
feature list. Read the definitions from the returned source bundle. Never
invent a parameter name, clamp a value, or reinterpret a unit. Use
`geometry-agent family` so every named row starts from the same immutable base;
do not chain one variant from another.

The family result must show a distinct source bundle for every successful row,
the exact requested values, and zero failed variants before it is described as
complete. Visual similarity alone does not prove a parameter was applied, and
different metadata alone does not prove geometric variation. Send resulting
sources through the ordinary Geometry workflow and OVRTX evidence path when
geometry readiness or visible variation is part of acceptance.

## Geometry Agent CLI

Set `GEOMETRY_AGENT_SERVICE_URL` to the service URL and provide the API key in
`GEOMETRY_AGENT_SERVICE_API_KEY`. The client rejects credential-bearing URLs,
redirects, unbounded responses, digest mismatches, and non-loopback HTTP.

```bash
geometry-agent providers
geometry-agent generate \
  --provider build123d-http \
  --prompt "A 60 mm mounting bracket with two M5 clearance holes" \
  --format step --format usdc \
  --target-profile geometry-agent.insertion-or-fixture-asset.v1 \
  --output result.json
geometry-agent run SOURCE_ID --output run.json
geometry-agent download ARTIFACT_ID model.usdc
```

Use `geometry-agent revise SOURCE_ID --provider PROVIDER_ID` for a provider-
owned immutable revision. The selectable delegated generation IDs are
`build123d-http` and `forgecad-http` when registered. Use the official Onshape
Labs MCP for FeatureScript authoring, then export manually or run
`geometry-agent export-onshape` with locally configured API credentials.
Use `geometry-agent import-forgecad` only for existing exported artifacts.

Use `geometry-agent family SOURCE_ID --provider PROVIDER_ID --variants FILE`
for semantic variants and `geometry-agent export SOURCE_ID --provider
PROVIDER_ID --format FORMAT` for providers with a distinct export operation.

## Render Evidence

Final evidence uses
`content_agent_workflows.geometry.render_geometry_evidence`, which delegates
to the shared OVRTX renderer and binds rendering and image validation to the
exact USD digest. Direct scene-tool renders remain diagnostics unless the
caller separately establishes equivalent renderer identity and digest binding.

For the bundled remote API, require a no-redirect `GET /health` response with
`service=ovrtx-rendering-api`, `renderer=ovrtx`, `status=healthy`, and
`gpu_initialized=true`. Use HTTPS and endpoint-scoped authentication for
non-loopback hosts. Never serialize credentials in evidence.

Reject or rerender before geometry critique when the object is blank, tiny,
clipped, hidden, framed from the wrong side, or rendered by an unproven
backend. Use matching lighting and cameras for comparisons.

Useful diagnostic views include material, depth, normal, wire, kind, and part
views. Openings and contact features need section-like or close-up evidence;
GLB or USD loadability alone does not prove they are unobstructed.

## Handoff Rules

Preserve the source bundle, provider receipt, selected representation, and all
artifact digests. Report:

- provider identity, operation, request digest, and source revision;
- source bundle ID, rights, coordinates, parts, parameters, and provenance;
- selected USD plus source and representation roles;
- conversion, optimization, repair, and validation reports;
- optional STEP, GLB, URDF, MJCF, and collision representations with accurate
  fidelity labels;
- optional accepted OVRTX evidence;
- degraded, skipped, unavailable, or failed outputs with reasons.

An optional export failure must not erase an accepted geometry candidate. A
tessellated representation must never be labeled exact B-rep. Geometry
acceptance must not be promoted to final material, physics, articulation,
runtime, manufacturing, or SimReady proof.
