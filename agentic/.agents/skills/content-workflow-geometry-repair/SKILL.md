---
name: content-workflow-geometry-repair
description: Complete a source-preserving Geometry repair workflow for imported CAD, mesh, USD, URDF, or MJCF geometry with bounded topology work, collision separation, and explicit evidence. Use when a user or parent workflow needs a durable certified, conditional, rejected, blocked, failed, or cancelled geometry-repair outcome rather than one isolated diagnosis or edit.
metadata:
  author: NVIDIA Omniverse
---

# Content Workflow Geometry Repair

Own the repair run: source snapshot, attempts, budgets, evidence, artifacts,
certificate, and completion state. Preserve the source, measure before
mutation, and let deterministic validators decide whether a candidate
advances. Use only typed operations allowed by the checked-in worker policy.
Ownership is compositional: this workflow owns run state, evidence acceptance,
artifact lifecycle, and completion.

## Capability Map

- Apply `geometry-source-intake` before diagnosis. Admit its immutable source
  identity, units, provenance, representation roles, and unresolved
  dependencies.
- Use `usd-cli` for typed scene inspection, picking, reversible
  previews, diagnostic rendering, and export operations.
- Use Geometry `render_geometry_evidence` through
  `render_usd_visual_evidence` for batch/final OVRTX evidence bound to the exact
  source USD digest.
- Apply `geometry-evidence-review` after every candidate that could advance.
- Invoke deterministic repair, re-diagnosis, correspondence, collision, and
  runtime modules through their typed contracts. Skills do not replace those
  validators.

## When to Use

- Diagnose or repair an imported CAD, mesh, USD, URDF, or MJCF asset.
- Preserve attributed topology while fixing classified local defects.
- Audit, retain, or regenerate render-part-specific collision geometry.
- Localize external USD dependencies into a portable, hash-bound package.
- Produce geometry-scoped evidence for rigid, articulated, or contact-rich
  downstream workflows.

## Limitations

- Do not invent geometry, units, part intent, materials, joints, physical
  properties, task paths, or clearances.
- Do not flatten a readable USD stage merely to normalize it.
- Do not interpret an unavailable worker, fallback copy, unevaluated predicate,
  or inferred feature as a pass.
- Do not reinterpret `geometry_repair.<profile>` evidence as final Articulation,
  Physics, Runtime Validation, or SimReady certification.

## Prerequisites

- Confirm the requested profile and whether mutation is authorized. Profile
  inference may route work but cannot produce a certified outcome.
- Preserve the source package and retain source/dependency digests.
- For production use, require source URI, source license, and source provenance.
  Missing rights evidence must remain conditional.
- Provide approved dependency roots or an exact remap manifest when the source
  package is not self-contained; never authorize basename or glob matching.
- Supply explicit protected features and task geometry when known.
- Use the shared conversion workflow for URDF/MJCF and a public
  `GeometryAuthoringProvider` when an upstream authoring system must export
  exact design geometry. Never execute provider-native source during repair.
- Use the `content-sdf-operations` capability contract for any direct field
  operation. Geometry Repair may select only a driver admitted by checked-in
  `sdf_tools` policy and separately behaviorally qualified by checked-in
  Geometry Repair policy; agent input cannot install or register an
  implementation.
- Optional NVIDIA first-party repair runtimes must remain out-of-process,
  digest-pinned, explicitly enabled by administrators, and independently
  validated. Their absence is `not_evaluated`, never a pass or an implicit
  fallback.

## Workflow

1. Freeze the source digest, profile, mutation authorization, protected
   features, worker allow-list, budgets, deterministic seed, and completion
   criteria. Apply `geometry-source-intake`; do not repair during intake.
2. For USD, inventory dependencies, authored hierarchy, transforms, prim roles,
   materials, subsets, primvars, collision state, and native primitives before
   extraction or repair. Localize only source-relative or caller-approved
   dependencies; keep unresolved or remote dependencies explicit.
   Run the independent official source-format validator first: OpenUSD
   `UsdUtils.ComplianceChecker` for USD, or Khronos `gltf_validator` for
   GLTF/GLB. A missing validator is `not_evaluated`, not pass.
3. Run the scalable non-mutating audit. Preserve its full four-state result and
   resource evidence; never collapse a resource skip into pass or failure.
4. Declare source-backed protected openings, cavities, clearances, handles,
   mating surfaces, material boundaries, thin features, and interfaces. Treat
   detected candidates as preservation evidence, never mutation authority.
5. Assign every issue and proposed operation to `render`, `collision`,
   `brep_source`, or `helper`. An agent may propose typed repair intent to rank
   approved candidates, but cannot add defects, target unknown prims, cross a
   representation boundary, enable workers, or change any gate.
6. Require source-part, face/corner, attribute, and prim-path correspondence for
   every topology-changing candidate. Refuse ambiguous categorical transfer.
7. Pair supplied collision prims to render parts before regenerating anything.
   Preserve every source part that meets fidelity, protected-space, task,
   complexity, and runtime requirements; require review for ambiguous or
   unmatched parts.
8. Run the public Geometry workflow with the confined source or source bundle,
   profile, mode, protected features, budgets, deterministic seed, and approved
   worker policy. For example, use `content-workflow-cli geometry run` with
   `--repair-mode diagnose|auto` and `--repair-profile PROFILE`. Use `ovphysx`
   only when authoritative collision runtime evidence is required; `fake` is
   test-only.
9. Route from the diagnosed defect to the least destructive available worker.
   Keep candidate and shadow adapters out of production authority until their
   audited implementation and registry state are enabled. When an adapter uses
   a native executable, that executable must also be audited and enabled. A
   reconstructive SDF candidate must bind one explicit driver admitted by
   `sdf_tools` and qualified by Geometry Repair, then preserve its identity
   evidence; never switch backends silently after an unavailable result.
10. Re-diagnose each candidate against the immutable source and apply
   `geometry-evidence-review`. Retain every attempt and do not let a repair
   operation approve its own result. Then retain
   the repair certificate, attempt ledger, geometry evidence, correspondence,
   advanced-profile evidence when requested, and handoff manifest.
11. If accepted repair changed render geometry, require digest-bound OVRTX views
   from Geometry `render_geometry_evidence` and a recorded human review.
   usd-cli previews alone are insufficient. Do not certify changed render
   geometry from numeric drift checks alone.
12. Finish with one explicit outcome: `certified`, `conditional`, `rejected`,
   `blocked`, `failed`, or `cancelled`. The parent workflow may consume the
   result but cannot broaden its geometry-only claim scope.

## Reference Routing

- Read [USD intake and scalable audit](references/usd-intake-and-scalable-audit.md)
  for any USD source, unresolved composition, prim-role question, dense mesh, or
  intersection audit that may hit a resource limit.
- Read [dependency localization](references/dependency-localization.md) whenever
  an asset uses external layers, payloads, references, textures, remote
  identifiers, or caller-supplied remaps.
- Read [protected geometry and correspondence](references/protected-geometry-and-correspondence.md)
  before filling, deleting, moving, fusing, reconstructing, simplifying, or
  transferring attributed topology.
- Read [role-aware repair](references/role-aware-repair.md) before proposing a
  repair intent, generating collision from an invalid render source, or
  interpreting role-scoped certificate results.
- Read [source collision audit](references/source-collision-audit.md) when source
  collision prims exist or a collider may be regenerated or simplified.
- Read [hard-local worker policy](references/hard-local-worker-policy.md) when a
  classified local defect needs PMP, WildMeshing, MCUT, Geogram, or bounded
  local-hole routing, or when a native adapter is unavailable.
- Read [hard-mesh reconstruction](references/hard-mesh-reconstruction.md) before
  any reconstructive Trimesh/SDF route or shadow exact evaluator. Load
  `content-sdf-operations` before invoking field operations directly.
- Read [profiles and certificates](references/profiles-and-certificates.md) when
  selecting a profile, interpreting an outcome, or handing work downstream.
- Read [articulated-rigid geometry](references/articulated-rigid.md) only for
  caller-supplied semantic links and joint-sweep geometry.
- Read [contact-rich geometry](references/contact-rich.md) only for
  caller-supplied insertion, mating, path, clearance, stop, or SDF geometry.

## Output Format

Return one geometry-scoped outcome:

- `certified`: every mandatory gate for an initial geometry profile passed;
- `conditional`: named review, missing evidence, or downstream work remains;
- `rejected`: no bounded candidate satisfied validity and source fidelity;
- `blocked`: required input, authority, dependency, or evidence is unavailable;
- `failed`: the workflow encountered an unrecoverable execution failure; and
- `cancelled`: the caller's cancellation request stopped the workflow.

Always state `claim_scope=geometry_repair.<profile>`, identify unevaluated
checks, and list downstream owners.

## Troubleshooting

- If a dependency is remote, unsafe, or outside the allowlist, record it and use
  the isolated root-layer inventory; do not fetch it implicitly.
- If localization is incomplete, continue geometry-only work when composition
  permits, but do not claim material fidelity or package portability.
- If an audit exhausts a budget, preserve `skipped_resource_limit` or
  `indeterminate` and increase a declared budget only with operator approval.
- If attributes cannot be mapped exactly, keep the source or refuse the edit.
- If a native hard-mesh executable is absent or drifts from its pinned build,
  retain `unavailable`; do not substitute an unreviewed binary.
- If an SDF backend is absent, not admitted by `sdf_tools`, not qualified for
  Geometry Repair, missing a required capability, or fails its identity or
  license policy, retain `unavailable`; do not install a plugin or select
  another backend implicitly.
- If an optional first-party B-rep validator or healer is unavailable,
  unpinned, or cannot validate both source and candidate, leave B-rep
  validity/fidelity unevaluated and non-certifying.
- If advanced-profile geometry passes, keep the repair outcome conditional
  until downstream articulation, physics, runtime, and SimReady work completes.
