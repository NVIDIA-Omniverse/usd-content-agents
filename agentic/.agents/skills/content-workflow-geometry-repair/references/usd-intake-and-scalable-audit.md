# USD Intake And Scalable Audit

Read this reference for USD sources, unresolved composition arcs, prim-role
questions, dense meshes, or intersection checks that may exhaust resources.

## Intake Contract

Call `geometry_repair.inventory_usd_stage` before mutation. It is non-mutating
and returns `UsdIntakeReport` with:

- a stable source digest and a second digest check proving whether the source
  changed during intake;
- allowlisted local dependency hashes and explicit states for missing, remote,
  dynamic, oversized, or outside-root dependencies;
- composed hierarchy, local/world transforms, purpose, visibility, kind,
  variants, materials, subsets, primvars, collision API state, and native
  primitive parameters;
- evidence-based roles: `render`, `source_collision`,
  `disabled_visual_collision_api`, `helper`, `guide`, or `unknown`.

Remote identifiers are recorded and never fetched. If a sublayer, reference, or
payload is unsafe or unresolved, intake opens an isolated root-layer view,
marks `composition_complete=false`, and does not traverse the unsafe arc.
Dependency hashing defaults to the source directory; add trusted roots through
`RepairRequest.dependency_roots`.

`run_geometry_repair` executes this intake for USD sources and writes
`usd_intake.json`. A complete intake is provenance and routing evidence, not a
material or visual-fidelity certificate.

## Scalable Audit Contract

Call `geometry_repair.audit_triangle_mesh(vertices, triangles, part_ids=...,
budget=ScalableAuditBudget(...))` for a direct array audit. Normal repair jobs
run the same audit through diagnosis and retain the full report path in
`Diagnosis.scalable_audit_path` and `RepairResult.scalable_audit_path`.

Interpret each predicate independently:

- `evaluated_pass`: every non-adjacent broad-phase candidate was tested and no
  witness was found;
- `evaluated_fail`: at least one concrete witness was found, even if a later
  resource limit stopped exhaustive counting;
- `skipped_resource_limit`: no conclusive witness was found before a declared
  broad-pair, exact-test, wall-time, or memory limit stopped evaluation;
- `indeterminate`: malformed or numerically unsuitable input prevented a
  decision.

Retain `ScalableMeshAuditReport`; its resource counters, completion state, and
within-part/cross-part findings are authoritative. The compatibility projection
maps both skipped and indeterminate states to legacy `not_evaluated`, never to
failure or pass.

## Operator Sequence

1. Inventory USD and resolve role-bearing dependencies before triangulation.
2. Audit render parts separately from source collision and helper geometry.
3. Supply one `part_id` per triangle when part ownership is known.
4. Record explicit broad-pair, exact-test, wall-time, memory, and chunk budgets.
5. Stop automatic mutation when a mandatory predicate is not evaluated.
6. Raise a budget only after reviewing the previous report and expected cost.

Do not use elapsed time alone as correctness evidence. Exhaustiveness is proven
by the report's completion and resource counters.
