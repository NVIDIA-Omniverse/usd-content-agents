# Source Collision Audit

Read this reference when a USD source already contains collision prims or when
collision geometry may be regenerated, decomposed, simplified, or replaced.

## Preserve Before Replacing

Treat the source collider as a candidate with provenance. A large primitive or
hull count is an advisory complexity signal, not a geometry defect. Preserve a
source proxy when it passes the required fidelity, protected-space, task, and
runtime gates.

Call `audit_source_collision_proxy(render_path, collision_path, ...)` to produce
`SourceCollisionAudit`. Its deterministic evidence includes:

- render-surface coverage, surface gap, and overreach;
- occupancy-based false-positive and false-negative ratios when evaluable;
- the requested occupancy face-point workload, its explicit budget, and a
  truthful `not_evaluated` result when that budget would be exceeded;
- render and collision part/triangle/vertex counts;
- primitive, hull, and convexity complexity evidence;
- explicit protected-feature probes and named task gates;
- blocking failures, unevaluated required checks, advisory warnings, and
  reasons that justify selective regeneration.

Use `evaluate_source_collision_gates` when metrics already come from a trusted
measurement pipeline. Do not synthesize a pass from absent occupancy or task
evidence.

## Decision Rules

1. Pair each role-identified source collider to a render part using explicit
   metadata first, then deterministic path convention, then a bounded uniquely
   measured fallback. Ambiguous or unmatched assignments require review.
2. Evaluate every required opening, cavity, handle gap, support, containment,
   insertion, and clearance probe in collision space.
3. Preserve each passing source part, even if CoACD or primitive fitting could
   reduce its count.
4. Regenerate only a named failing part or an explicit optimization target.
5. Compare the replacement with the source proxy under the same gates.
6. Require the target-runtime cook/contact or drop/settle evidence separately.

For `rigid_pick_place`, the collision builder writes a collision-only source
snapshot and runs the audit in an isolated subprocess under the remaining
memory budget and dedicated `source_collision_audit_wall_time_s` limit. For
compound assets it retains the typed pairing and per-render-part audit, reuses
passing parts, and generates candidates only for measured failing parts.
`review_required` is terminal for automatic replacement: retain the snapshot
and evidence, author no replacement, and return a conditional handoff.

Every generated part evaluates materialized source, primitive, convex-hull,
and, when needed, CoACD candidates under fixed volume, occupancy,
surface-distance, protected-feature, hull, and runtime budgets. Keep each
candidate USD, SHA-256, hard-gate result, score, and deterministic selection in
`candidate_search.json`. A measured gate failure is not conditional. Missing
probe evidence may select only a conditional candidate when no fully evidenced
candidate passes. That outcome is review-required and cannot be certified or
accepted automatically. The selected composed collider still requires real
runtime cook/drop/settle evidence for production certification.

Count only generated collision prims against the generated-hull budget and
reserve at least one slot for every remaining render part. An over-complex
single convex hull may be simplified only as a collision-role candidate using
the approved bounded working-copy backend; recompute convexity and rerun every
hard gate against the unsimplified source before selection. Exact-volume
pre-screening may reject an impossible primitive early, but it cannot accept a
candidate or replace the full gate ledger.

CoACD search is adaptive: start coarse, retain every failed candidate as
evidence, and escalate only to the caller-declared hull budget. The default is
128 hulls and the public maximum is 256. Request the higher budget only when a
default-budget candidate fails fixed fidelity gates and the target runtime can
accept the added complexity. Never increase a fidelity threshold to make a
candidate pass.

## Outcomes

- `pass`: all blocking source-collision gates evaluated and passed;
- `conditional`: advisory concerns or explicitly review-required evidence gaps
  remain; the handoff cannot be certified or accepted automatically;
- `fail`: a blocking measured gate failed;
- `not_evaluated`: at least one required blocking gate lacked evidence.

Collision audit is geometry evidence. Physics properties, collision filtering,
solver tuning, and full task/runtime behavior remain downstream-owned.
