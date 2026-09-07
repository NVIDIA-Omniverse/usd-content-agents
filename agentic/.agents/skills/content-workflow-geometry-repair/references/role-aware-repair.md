# Role-Aware Repair

## Representation Authority

- `render` owns visible topology, part paths, UVs, normals, material subsets,
  and visual fidelity.
- `collision` owns task collision geometry, protected negative space,
  approximation error, complexity, cooking, and runtime geometry evidence.
- `brep_source` owns exact CAD/B-rep validity and source topology.
- `helper` owns guide geometry and profile-specific geometric evidence that is
  neither render nor collision.

Do not use success in one role to clear a blocker in another. The certificate
derives one profile outcome from the required role states.

Scope every `ProtectedFeature` with `affected_roles`. Detector-derived
features are render evidence by default. Promote one into collision authority
only from a caller-confirmed task requirement, authored semantic/source fact,
or equivalent recorded evidence; visual topology alone is insufficient.

## Repair Intent

`RepairIntent` is ranking input, not mutation or validation authority. Require:

- a known target role and source prim path;
- measured issue IDs from the diagnosis;
- explicit expected topology effects;
- only workers enabled by the request and registry;
- existing evidence paths and named protected features.

Reject unknown defects, prims, features, workers, cross-role operations, and
agent-proposed thresholds. Deterministic diagnosis, budgets, worker policy,
fidelity, correspondence, protected-feature probes, and runtime checks remain
authoritative.

## Collision-Only Reconstruction

Use `sdf_collision_rebuild` only when all of these hold:

1. The profile requires collision and the render-derived collision working
   source is not a measured positive watertight solid.
2. `allow_reconstructive=true` and the worker is explicitly enabled.
3. No approved source collision, primitive, hull, or less destructive route
   already satisfies the fixed collision gates.
4. Collision-role openings, cavities, handles, clearances, mating surfaces,
   and interfaces have deterministic probes or remain explicitly unmeasured.

The native helper reads a neutral world-space triangle copy. It must never
author render USD. Require matching render SHA-256 before and after every
attempt. Try the bounded collision grid ladder as immutable candidates; a fine
grid may preserve an open surface as a thin double shell while a coarser grid
can close a discretization-scale defect. Reject area, positive-volume,
winding, watertightness, resource, feature-resolution, or p99 drift failures.

Measure primitives, convex hulls, and CoACD against the selected reconstructed
collision reference. Preserve every failed-grid report. A generated collision
reference remains conditional until independent task evidence or recorded
human review accepts it. It does not repair or certify defective render
geometry.

## Attributed Local Render Surgery

Prefer source-face deletion or reorientation with exact face/corner mapping.
For a classified accidental hole:

- freeze all source vertices and protect every boundary edge;
- retain every source face in original order;
- classify every appended face as generated;
- permit generated faces over source boundary vertices only;
- extend continuous or categorical topology attributes only when all declared
  boundary contributors agree;
- reauthor geometric normals explicitly;
- append material-subset membership only when boundary signatures agree.

Refuse time-sampled topology, UV seams with conflicting corner values,
conflicting uniform attributes, conflicting subset membership, new patch
vertices without exact interpolation evidence, moved frozen vertices, removed
protected edges, or changes outside the measured region. Require source
correspondence, protected-feature comparison, fidelity, and OVRTX review after
the worker completes.

## Outcome Reading

- `pass`: the role's required deterministic evidence passed.
- `conditional`: generated geometry, visual review, task evidence, or another
  named check remains.
- `fail`: a required gate failed; another role cannot mask it.
- `not_evaluated`: the role applies but lacks authority or evidence.
- `not_applicable`: the role is not required and has no relevant source data.
