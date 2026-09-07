---
name: content-workflow-geometry
description: Complete a provider-neutral Geometry handoff from CAD, mesh, generated, or existing USD. Use when a workflow needs source lineage, prepared and optimized geometry, OVRTX and USD evidence, semantic parts or parameters, and content_agents_manifest.json without claiming final physics, runtime behavior, or SimReady status.
metadata:
  author: NVIDIA Omniverse
---

# Content Workflow Geometry

Own Geometry run state, evidence, artifacts, retry policy, and completion. Keep
external authoring systems behind the public `GeometryAuthoringProvider`
contract. Preserve the exact `geometry.source.v1` bundle, source revision,
semantic parts, parameters, coordinate system, rights, and provenance.
Ownership is compositional: this workflow owns run state, evidence acceptance,
artifact lifecycle, and completion.

## Capability Map

| Stage | Skill or operation | Responsibility |
| --- | --- | --- |
| Source admission | `geometry-source-intake` | Admit one immutable source bundle or existing artifact without losing fidelity |
| External authoring | Geometry Agent service/CLI | Ask one explicitly selected provider to generate, revise, or export a source bundle |
| Scene operations | `usd-cli` | Inspect, optimize, validate, and export through typed scene tools |
| Final visual evidence | `render_geometry_evidence` | Produce digest-bound OVRTX evidence |
| Evidence decision | `geometry-evidence-review` | Accept, refine, repair, request evidence, or block |

Use `content-workflow-geometry-repair` when imported geometry is defective. The
repair workflow owns its repair certificate; this workflow decides whether the
result can enter the handoff.

## Invariants

- Treat provider-native source as opaque provenance. Never import, evaluate, or
  execute it in the Geometry workflow.
- Select providers explicitly. Installed modules and environment variables do
  not authorize a provider or operation.
- Query provider capabilities before authoring. Honor the user's exact provider
  choice; an unavailable or insufficient provider is a typed block, not
  permission to install software or choose a fallback.
- Bind every representation to byte size and SHA-256. A provider receipt is
  not evidence for bytes whose identity does not match.
- Prefer `render_geometry`, then `design_exchange`, `collision_candidate`, and
  `reference`. `native_source` is not processable geometry.
- Preserve source units, axes, part hierarchy, semantic parameters, rights,
  and upstream edit URI.
- A normalized copy is not optimized geometry. An unavailable required
  optimizer cannot produce a passing optimization claim.
- Geometry does not claim final materials, physics, articulation, runtime
  behavior, manufacturing validity, or SimReady status.

## Workflow

1. Freeze the request, provider choice, input/reference digests, target
   profile, requested formats, rights policy, budgets, and completion gates.
2. If generation or revision is required, invoke the Geometry Agent service or
   a `GeometryAuthoringProvider`. Require a successful typed receipt whose
   provider identity, operation, and request digest match the request. Store the
   returned source bundle and all representations immutably.
3. Apply `geometry-source-intake` to the source bundle manifest or existing
   artifact. Do not duplicate converter, parser, or provider logic in the
   reasoning loop.
4. If intake or evidence identifies defective imported geometry, run
   `content-workflow-geometry-repair` with frozen protected features, profile,
   worker policy, and budgets. Preserve the original source and repair lineage.
5. Invoke optimization with `skip`, `preserve_correspondence`, or
   `runtime_efficiency`. Keep source mapping and representation roles.
6. Run the shared USD and Geometry validators required by the request. Keep
   source-format, topology, generic USD, and workflow reports separate. Retain
   unavailable checks as `not_evaluated`.
7. When visual evidence is required, call
   `content_agent_workflows.geometry.render_geometry_evidence`. Accept final
   visual evidence only from OVRTX with renderer identity and readiness bound
   to the exact USD digest. Use other renderers only for diagnostics.
8. Apply `geometry-evidence-review`. A deterministic failure cannot be
   overridden by visual opinion. On `refine`, send a bounded revision request
   to the same or explicitly selected provider, with the exact parent bundle.
   On `repair`, run only the selected typed repair route. Append evidence; do
   not overwrite prior candidates.
9. Leave runtime validation at `skip` for geometry-only handoff. Use authored
   physics only when an upstream stage supplied it. A temporary loadability
   proxy proves loadability only.
10. Run formal SimReady validation only when requested. Treat a blocked or
    unavailable validator as `not_evaluated`.
11. Write `geometry_validation_evidence.json`, the evidence bundle, and
    `content_agents_manifest.json` with exact artifact identities and claim
    scope.
12. Finish as `completed`, `conditional`, `blocked`, `failed`, or `cancelled`.
    Mark `completed` only when every requested Geometry gate is evaluated and
    accepted.

## Provider Routes

- **Build123d reference worker:** generation/revision happens in an isolated,
  independently operated worker. The public workflow receives only a typed
  receipt and exported artifacts.
- **Onshape Labs FeatureScript MCP:** connect the user's MCP-compatible client to
  `https://fs-mcp.labs.onshape.app/mcp` over HTTP and let the user complete the
  Onshape OAuth flow. Discover the current tool list, read the FeatureScript
  notes, and test code before creating geometry. The MCP does not currently
  export geometry. Prefer manual export; when the user asks for an automated
  continuation, ask them to configure API credentials in their local secret
  environment without pasting values into chat, then use the export-only helper.
  Never place those credentials in arguments, service configuration, logs, or
  artifacts.
- **ForgeCAD authoring worker:** use `forgecad-http` only when it is registered
  with an explicit automated-use authorization and rights assertion. Never run
  an installed ForgeCAD package or CLI from this workflow.
- **ForgeCAD artifact intake:** import existing exports and optional inert
  source provenance only. This is not generation or revision.
- **Other providers:** implement the same public protocol and pass capability,
  digest, containment, rights, and response validation.

See `references/onshape-featurescript-mcp.md` for the Onshape authoring handoff
and `references/backend-render-handoff.md` for other providers and rendering.

## Policy Surface

```text
source_authoring_mode:
  auto | shared_conversion | opaque_import | parametric_recovery |
  direct_preserve

optimization_policy:
  skip | preserve_correspondence | runtime_efficiency

repair_mode:
  off | diagnose | auto

runtime_validation_mode:
  skip | authored_physics | temporary_loadability_proxy

simready_mode:
  skip | validate | validate_and_route_conformance
```

## Required Outputs

- immutable `geometry.source.v1` manifest and provider receipt when externally
  authored;
- prepared and final geometry USD paths and SHA-256 identities;
- source fidelity, conversion, optimization, and validation reports;
- semantic parts, parameters, coordinate system, rights, and provenance;
- optional digest-bound OVRTX views and render report;
- optional runtime and formal SimReady reports with explicit claim scope;
- `geometry_validation_evidence.json`;
- `geometry_evidence_bundle.json`;
- `content_agents_manifest.json`.

Keep compatibility fields such as `mesh_usd` and `brep_usd`, but never label a
tessellated mesh as B-rep. Never reinterpret Geometry acceptance as downstream
material, physics, articulation, runtime, or SimReady proof.

## Progressive References

- `references/routing-and-fidelity.md`: source categories and fidelity rules.
- `references/evidence-contract.md`: status semantics and claims.
- `references/downstream-delegation.md`: downstream ownership.
- `references/backend-render-handoff.md`: provider, source-bundle, and OVRTX
  handoff.
