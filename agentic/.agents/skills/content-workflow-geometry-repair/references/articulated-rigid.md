# Articulated-Rigid Geometry Evidence

Read this reference when the requested repair profile is `articulated_rigid`,
the source contains multiple rigid links, or a caller asks whether repaired
collision geometry can move through supplied joint limits.

## Ownership

Geometry Repair owns:

- explicit semantic part/link hypotheses with provenance and confidence;
- render/collision prim mapping for each proposed link;
- finite, indexed, watertight, consistently wound, convex per-link collision
  checks;
- deterministic sampled collision checks over caller-supplied joint geometry;
- source-relative geometry fidelity and protected-feature evidence.

Geometry Repair does not infer or author joints, limits, drives, collision
filters, mass, inertia, friction, controls, or runtime stability. Route those
to Articulation, Physics, and Runtime Validation after geometry evidence is
complete.

## Required Inputs

Embed an `AdvancedProfileRequest(profile="articulated_rigid", ...)` in the
parent repair request. Provide:

- `SemanticLinkHypothesis` records with source paths, provenance, confidence,
  and evidence kind;
- one `LinkGeometryMapping` per link, listing exact render and collision prim
  paths;
- `JointSweepInput` records supplied by the caller, including parent, child,
  moving subtree, type, world-space axis and origin, limits, reference value,
  units, and sample count;
- `AdjacentLinkExclusion` records for intentionally skipped adjacent pairs,
  including reason and source evidence.

Do not translate a semantic link hypothesis into a joint. Missing axis,
origin, limits, reference value, or units is valid request state: retain the
request and emit `not_evaluated` swept-motion evidence with a `conditional`
disposition.

## Evaluation

After the parent orchestrator accepts a repaired candidate:

1. Load the accepted render and collision artifacts.
2. Call `geometry_repair.advanced_profiles.evaluate_advanced_profile`.
3. Require exact link-ID parity between semantic hypotheses and geometry
   mappings. Reject shared prim ownership across links.
4. Validate every mapped collision part independently. Do not treat a render
   mesh as a collider substitute.
5. Sample each supplied joint from lower to upper limit, inclusive, relative
   to the supplied reference value.
6. Transform only the caller-declared moving links. Check them against all
   stationary links except explicitly listed adjacent exclusions.
7. Treat a triangle-pair budget exhaustion as `indeterminate`, never pass.

Sampled swept geometry proves only that the tested collision surfaces do not
intersect at the sampled poses. It does not prove continuous collision
freedom, actuation, dynamic stability, contact behavior, or runtime collision
filtering.

## Evidence Rules

- `pass`: every required deterministic geometry check completed and passed.
- `fail`: an evaluated mapping, collider, or sampled pose violated a gate.
- `not_evaluated`: required caller geometry was absent or no joint was supplied.
- `indeterminate`: a supplied query could not be decided within its backend or
  resource bound.

Set `certification_eligible` only when all required checks pass and every
semantic, mapping, joint, and exclusion claim has at least one `source_fact` or
`measured_fact`. Inferred hypotheses may coexist with authoritative evidence,
but cannot be the sole basis for eligibility.

## Output And Handoff

Persist the complete `AdvancedProfileEvidenceReport` with
`write_advanced_profile_report` as one JSON artifact and
reference it from the parent repair certificate. Preserve its embedded request,
link checks, sampled values, collision events, exclusion metadata, blockers,
and downstream owners.

Send the accepted render/collision pair and report to:

- Articulation for joint schema, limits, drives, and collision-filter authoring;
- Physics for mass and contact properties;
- Runtime Validation for target-runtime actuation, limit, contact, and
  stability evidence.

Do not upgrade the geometry-only report to final articulated or SimReady
certification.
