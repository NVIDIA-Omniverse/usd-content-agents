---
name: content-workflow-texture
description: Use one outer Codex or Claude reasoner to select and invoke focused, digest-bound Texture capabilities without a nested Texture workflow controller.
version: "0.5.1"
author: NVIDIA Omniverse
tags:
  - content-agents
  - texture
  - workflow
  - validation
tools:
  - Shell
  - Filesystem
  - Python
compatibility: Requires repository-root setup and Python >=3.12; agentic/ remains an internal implementation workspace and does not require changing directories; fixed-pipeline service, config, and benchmark paths remain available explicitly.
metadata:
  author: NVIDIA Omniverse
  tags:
    - content-agents
    - texture
    - workflow
    - validation
---

# Texture Workflow Method

The one outer Codex or Claude reasoner selects focused Texture skills directly.
There is no Texture-local selector, planner agent, or hidden workflow controller.
Deterministic code only prepares typed evidence, invokes an explicitly selected
leaf, validates identities and scope, records the outer decision, and publishes
the exact accepted candidate.

Read [the capability matrix](references/capability-matrix.md) before choosing
operations.

## When to Use

Use for bounded texture generation, reference-faithful texture work, candidate
review, or Texture publication. Choose only the operations required by the
current task.

## Limitations

- Candidate execution has two explicit, mutually exclusive modes: invoke one
  selected generator provider, or deterministically apply outer-provided
  generated images. There is no automatic selection or fallback.
- Advisory Texture service planning and domain VLM critique are optional leaves.
- Deterministic measurements cannot replace semantic visual review.
- The provider-free UV leaf supports only `inspect` and `generate_missing`.
  Its source must be one `.usd`, `.usda`, or `.usdc` layer, or one
  self-contained `.usdz` package. A package with external dependency bindings
  is rejected before execution. When `generate_missing` authors an overlay over
  a package, the output closure retains the exact sealed USDZ predecessor path,
  digest, and size.
  Generate-missing uses one bounded box fallback for meshes with no authored UVs;
  it is not a caller-selectable projection or atlas implementation. Canonical
  full projection and atlas preparation remain on the existing Scene Optimizer
  Texture prepare path.
- Map-set expansion and broader quality-control capabilities remain #360 scope.
- Architecture readiness is not asset qualification. Do not claim #1080,
  #1092, or #1098 from this workflow change.

## Prerequisites

- A frozen local USD source and explicit material or prim scope.
- A dedicated new run directory.
- Explicit reference roles and files when the request uses references.
- The installed usd-cli and a reachable local or remote OVRTX renderer for
  required render evidence.
- Explicit provider configuration only for selected proposal, generator, or
  critique leaves.

## Instructions

1. As the outer reasoner, freeze the task-scoped operation selection. `inspect`
   is always requested. If generation or publication is selected, evidence,
   outer review, and publication safety gates cannot be omitted.
2. When the selected scope needs UV inspection or a conservative missing-UV
   fallback, author one `texture-uv-leaf-invocation.v1` inside the current leaf
   attempt and run `texture agentic-leaf uv-prepare`. Select only `inspect` or
   `generate_missing`; there is no projection argument. Generate-missing may
   author only absent UVs through the bounded box fallback. This leaf constructs
   no Texture service, VLM assessor, image generator, fixed pipeline, nested
   coordinator, or Scene Optimizer. For mutation it saves and reopens the stage,
   then binds exact UV and purpose-specific effective-material identities from
   that reopened stage. The leaf consumes a frozen exact source-layer closure,
   rejects variant-composed mutation, and permits only its deterministic UV
   overlay delta. A failed native result seals the current attempt. In graph
   v2, call `fail_leaf` with that exact result error, invocation path, and result
   path, then call `recover_leaf` before creating a fresh attempt; never delete
   or reuse failed artifacts. A passing result calls `complete_leaf` with only
   the exact invocation and result paths. An `inspect` result may be
   `not_evaluated`; it may call `complete_leaf` only when its selected graph
   node is optional and no selected leaf depends on it. The six-leaf Texture
   chain always selects `generate_missing`, because `texture.prepare.v1`
   requires an exact passing UV result. If an inspect-only graph cannot satisfy
   that dependency, start a fresh graph and attempt instead of beginning or
   relabeling the inspect invocation. Do not
   pass compatibility projection, publication, evidence-index, operation-index,
   or saved-stage paths to graph v2: the registered Texture runtime projector
   derives and verifies its receipt categories without executing the leaf or
   mutating graph state. `project_texture_uv_verified_operation` is a
   standalone compatibility/audit projector; its indexes are its own
   projection and publication artifacts, not graph-v2 runtime requirements.
   A self-contained `.usdz` source has no external source-dependency bindings;
   when `generate_missing` authors an overlay, require its output dependency
   closure to retain that exact sealed USDZ predecessor.
3. Invoke `content-texture-scope` and run `texture prepare`. When this operation
   is selected in the asset graph, author one direct
   `texture-focused-leaf-invocation.v1` in the active attempt and run
   `texture agentic-leaf prepare`; bind the exact passing UV result and use its
   output plus dependency closure as the preparation request source. The native
   output root is the attempt-local `native/` directory. Inspect the typed
   source, dependency closure, material/UV facts, references, initial OVRTX
   images, and capability constraints yourself. Project only the resulting
   exact native packet, terminal receipt, and saved-stage readback.
4. If and only if `propose=requested`, run `texture propose`. Treat its service
   plan as advisory; your canonical targets, semantics, and generator inputs do
   not have to equal it.
5. Author `texture-outer-capability-plan.v1`, binding the preparation, stable
   units, exact references, requested appearance, generator inputs, acceptance
   criteria, preservation policy, and stop policy.
6. If and only if `generate=requested`, invoke `content-texture-candidate` and
   choose exactly one mode. The public Texture graph exposes only
   `texture agentic-leaf apply-provided`, with one exact outer-generated albedo
   artifact per prepared unit and proposal/critique recorded as
   `not_requested`. Provider generation remains an explicitly selected focused
   command outside this provider-free graph chain. References remain references
   and cannot be converted into candidates. Stop after the digest-bound
   generation packet.
7. If candidate mutation occurred, invoke `content-texture-quality` and run
   `texture agentic-leaf evidence` when graph-selected. Deterministic code
   rechecks scope and captures matched source/candidate OVRTX views; semantic
   assessment remains `not_evaluated`.
8. If and only if `critique=requested`, run `texture critique` with an explicit
   VLM backend and model. Its findings are advisory and provenance-distinct.
9. Directly inspect every reference, outer-provided candidate image, and matched
   OVRTX view. Author one typed accept, reject, or revise disposition per exact
   unit, then run `texture agentic-leaf review` to validate and record it when
   graph-selected. A reject or revise disposition is a truthful `failed`
   review result whose exact error retains every nonaccepted unit, disposition,
   and rationale. Call `fail_leaf` with that exact error, then `recover_leaf`
   before a fresh review/candidate attempt. It cannot satisfy publication.
10. Only after every unit is accepted, invoke `content-texture-publish` and run
   `texture publish`. Deterministic finalization rechecks all identities,
   source closure, references, generator result, evidence, candidate bytes,
   scope, non-target preservation, and saved-stage readback. Graph-selected
   publication uses `texture agentic-leaf publish` and writes the published
   asset only below that active attempt's `native/` root.
11. Stop. Do not automatically choose another operation. The outer reasoner
    decides whether a rejected/revise result warrants a new bounded attempt.
    For every focused graph-v2 leaf, use the same terminal API rule as the UV
    leaf: `complete_leaf` receives only the exact invocation and result paths;
    `fail_leaf` additionally receives the exact native error as its reason.
    Never forward standalone publication fields into either graph-v2 call.

## Optional Status Rules

- Unselected proposal, generator, or critique leaves are `not_requested`.
- Selected leaves not yet invoked are `not_evaluated`.
- Neither state means pass and neither may satisfy a mandatory gate.
- A completed optional leaf records its exact provider/capability and artifact.

## Command Reference

The canonical focused surface is:

```text
content-workflow-cli texture prepare
content-workflow-cli texture propose
content-workflow-cli texture generate
content-workflow-cli texture apply-provided
content-workflow-cli texture evidence
content-workflow-cli texture critique
content-workflow-cli texture review
content-workflow-cli texture publish
content-workflow-cli texture agentic-leaf uv-prepare
content-workflow-cli texture agentic-leaf prepare
content-workflow-cli texture agentic-leaf apply-provided
content-workflow-cli texture agentic-leaf evidence
content-workflow-cli texture agentic-leaf review
content-workflow-cli texture agentic-leaf publish
```

Each command performs only the named operation. It never selects or invokes the
next command. The `texture agentic-leaf` family cannot enter `texture run`,
`texture resume`, `texture _agent-step`, or a fixed pipeline. Those remain
legacy Agentic compatibility commands. Texture Agent service/config execution
and benchmark entry points remain explicit fixed-pipeline surfaces.

## Resume and Identity

Every downstream packet binds its exact inputs. Before resuming, revalidate the
request, source and dependency closure, scope-plan digest, reference roles and
bytes, outer plan, provider result, candidate, evidence, and review. Drift
requires a fresh run directory and identity; never relabel an earlier attempt.

## Output Format

The focused operation family writes:

- `capability_request.json`
- `texture_preparation.json` plus inspection and initial renders
- optional `texture_provider_proposal.json`
- outer-authored plan input
- `texture_generation.json`
- `texture_candidate_evidence.json`
- optional `texture_critique.json`
- `texture_outer_review.json`
- `publication_validation.json`
- `texture_publication_receipt.json`
- `texture_uv_evidence.json`, `texture_uv_saved_stage_readback.json`, and
  `texture_uv_leaf_result.json` for the provider-free UV leaf
- `texture_uv_verified_operation_projection.json` and
  `texture_uv_asset_leaf_publication.json` only when the standalone
  compatibility/audit projector is explicitly requested; graph v2 uses the
  registered runtime projector and does not consume these indexes
- one operation-specific focused invocation, native packet, saved-stage
  readback, result, terminal receipt, verified projection, and generic
  publication for each selected `prepare`, `apply-provided`, `evidence`,
  `review`, or `publish` graph leaf

The Texture asset-leaf catalog registers the explicit dependency chain
`texture.uv-prepare.v1` -> `texture.prepare.v1` ->
`texture.apply-provided.v1` -> `texture.evidence.v1` ->
`texture.review.v1` -> `texture.publish.v1`. Every focused leaf writes only
below its own active attempt, reopens and revalidates the exact stage, and emits
a native disposition that the non-executing projector must recompute. Proposal,
provider generation, and critique are deliberately not graph leaves in this
provider-free chain.

The final receipt binds references and any outer-provided candidate images
through plan, generation, evidence, review, resume inputs, and publication.

## Troubleshooting

- If a provider is contacted during preparation or evidence collection, stop;
  the wrong capability boundary was used.
- If optional status is ambiguous or reported as pass without an invocation,
  reject the packet.
- If reference, outer-provided image, source, plan, candidate, or evidence bytes
  drift, start a new attempt.
- If the outer review omits a reference or matched image, fail closed.
