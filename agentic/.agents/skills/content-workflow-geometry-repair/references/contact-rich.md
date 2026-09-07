# Contact-Rich Geometry Evidence

Read this reference for insertion, mating, connector, keyed, lead-in, or seated
contact tasks under the `contact_rich` profile.

## Ownership

Geometry Repair owns metric-space evidence for:

- a caller-declared contact axis and approach direction;
- centerline path occupancy and clearance against receiver collision geometry;
- a caller-declared seated-stop support point;
- minimum protected-feature preservation;
- SDF voxel-size adequacy when the selected collision representation is SDF.

Geometry Repair does not infer the task path, tune contact materials or solver
settings, author joints, execute insertion control, or claim dynamic seating.
Physics, Articulation, and Runtime Validation own those results.

## Required Inputs

Embed an `AdvancedProfileRequest(profile="contact_rich", contact_probes=[...])`
in the parent repair request. Each `ContactRichProbeInput` identifies the moving
and receiver parts and supplies:

- exact `receiver_collision_paths` for the receiver's collision prims;
- world-space axis origin and nonzero axis;
- `along_axis` or `against_axis` approach direction;
- an ordered world-space path from approach to seated pose;
- axis and angular tolerances;
- a measured moving-envelope radius and required radial clearance;
- a receiver stop-surface probe point and tolerance;
- the minimum protected feature size;
- `convex` or `sdf` collision representation;
- SDF voxel size and required voxels across the minimum feature when using SDF;
- deterministic path sample step and bound;
- source or measured evidence for the task intent.

Never fabricate missing axis, origin, path, stop, or resolution values. Missing
axis/origin/path yields `not_evaluated` evidence and a `conditional` report,
not a generic geometry failure or inferred replacement path.
Missing `receiver_collision_paths` also yields `not_evaluated`. Duplicate,
unresolved, multiply resolved, empty, or non-collision prim paths fail the
receiver-selection check. Never query the complete collision artifact merely
because `receiver_part_id` is present.

## Evaluation

After candidate acceptance, call
`geometry_repair.advanced_profiles.evaluate_advanced_profile` with the accepted
render/collision artifacts and embedded request.

For each probe:

1. Resolve every declared receiver path exactly once and require its prim role
   to be collision. Build the query union in memory from only those meshes;
   do not write a filtered temporary asset and do not include moving-part or
   unrelated colliders.
2. Check every path control point against the declared axis tolerance.
3. Check every nonzero segment against the declared approach direction and
   angular tolerance.
4. Require strict monotonic progress along the approach direction.
5. Deterministically sample the complete path. If the sample bound truncates
   it, return `indeterminate`.
6. Require every center sample to remain outside the selected watertight
   receiver solids.
7. Require minimum surface distance to the selected receiver union to be at
   least moving-envelope radius plus
   required radial clearance.
8. Require the seated-stop probe point to lie within its surface tolerance.
9. For SDF, require
   `voxel_size <= minimum_protected_feature / required_voxels_across_feature`.
   Mark this check non-required for a declared convex representation.

The centerline-plus-envelope probe is a bounded geometry test. It is not a
full six-degree-of-freedom motion planner and does not prove force closure,
frictional insertion, thread engagement, latching, or solver convergence.

## Evidence Rules

- `pass`: all required axis, approach, path, clearance, stop, and applicable
  resolution checks completed and passed.
- `fail`: an evaluated metric violated its caller-supplied threshold.
- `not_evaluated`: required caller geometry, receiver selection, accepted
  collision artifact, or SDF resolution was absent.
- `indeterminate`: occupancy, surface distance, or bounded sampling could not
  decide the supplied query.

An inferred task hypothesis alone cannot set `certification_eligible`. Require
at least one `source_fact` or `measured_fact` for every probe. Preserve failed,
not-evaluated, and indeterminate checks exactly; do not collapse them into a
single false geometry failure.

## Output And Handoff

Persist one `AdvancedProfileEvidenceReport` with
`write_advanced_profile_report`, containing the request, per-probe
checks, status, disposition, blockers, and downstream owners. Reference it from
the parent repair certificate after candidate acceptance.

When the accepted candidate has no collision artifact, call
`write_collision_unavailable_advanced_profile_report`. It emits the same report
shape with `status=not_evaluated` and `disposition=conditional`. Never pass the
render artifact to the contact evaluator as a collision substitute.

Route the accepted collision and evidence to Runtime Validation for the actual
approach-to-seat trajectory and disturbance tests. Route physical contact
properties to Physics and any constrained degrees of freedom to Articulation.
Do not call geometry-only contact evidence final runtime or SimReady proof.
