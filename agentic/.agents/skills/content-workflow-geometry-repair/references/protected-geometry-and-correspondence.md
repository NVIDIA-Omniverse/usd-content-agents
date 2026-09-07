# Protected Geometry And Correspondence

Read this reference before any operation can add, delete, move, merge, split,
reconstruct, simplify, or reassign source geometry.

## Protected Geometry

Explicit `ProtectedFeature` records are authoritative operator/task inputs.
Use `negative_space_path` probes for openings, cavities, handle gaps, and
clearances. Use `surface_support` probes for rims, tines, wires, mating faces,
and required support surfaces. Set `affected_roles` deliberately: a render
detail is not automatically a collision affordance, and a collision-only
clearance does not authorize render mutation.

`detect_protected_feature_candidates` derives deterministic source evidence for
openings, cavities, handles, wires, rims, and tines. It analyzes a
position-welded copy so UV/material property seams do not masquerade as physical
holes. Every detected candidate has `mutation_authorized=false`:

- `protect` means the measured confidence met the configured protection
  threshold;
- `review_only` means intent is uncertain and cannot authorize an edit.

Use `compare_protected_feature_candidates` before and after a candidate edit.
A regression blocks the candidate; an unevaluated or low-confidence comparison
keeps the result conditional. The orchestrator runs detection before planning,
writes `protected_feature_candidates.json`, compares every changed candidate,
and keeps detector-derived hypotheses render-scoped. Mesh topology can show a
boundary, cavity, wire, or tine, but cannot establish that it belongs in a
rigid collision proxy. Collision validation accepts only features whose
`affected_roles` include `collision`; pass known task protections explicitly or
promote them from authoritative source/task evidence. Role-excluded feature
names remain visible in `CollisionReport.role_excluded_protected_features`.
Inferred candidates still have `mutation_authorized=false`.

## Correspondence

Use `CorrespondenceEvidence` for topology-changing workers. It records source
and output digests plus vertex, corner, face, part, prim-path, changed-region,
generated-region, and attribute-transfer evidence.

Apply these transfer rules:

- Map UVs and other face-varying values by source face-corner or by an explicit
  source face plus barycentric coordinates, never nearest source vertex.
- Interpolate continuous data only through an explicit mapping.
- Transfer categorical face data only when every contributing source face
  agrees. Otherwise split the region or refuse it.
- Keep material, part, UV, hard-normal, primitive-interface, and protected
  boundaries locked by default.
- For a bounded generated patch, extend uniform, indexed face-varying, and
  subset data only when every declared boundary contributor agrees. Recompute
  generated face normals explicitly and persist the patch source-face/corner
  map. Refuse any conflict.
- For face deletion, duplicate/degenerate cleanup, or winding repair, reauthor
  by exact source face/corner IDs while retaining the source point array and
  prim path.
- Mark generated patches and reauthored normals, tangents, or UVs explicitly;
  do not call them source preservation.

`identity_correspondence`, `build_mesh_correspondence`,
`refused_correspondence`,
`resolve_categorical_assignment`, and `transfer_face_varying_values` are
implemented and tested. `write_correspondence_evidence` writes a finalized,
hashed report atomically.

The orchestrator writes typed `CorrespondenceEvidence` for every mesh candidate
and rejects explicit refusal or missing required evidence. Identity candidates
retain exact source counts and attributes; changed candidates use measured
mesh correspondence. The accepted attempt report is copied to
`final/geometry_correspondence.json` and referenced by the certificate and
handoff. Refuse a topology-changing attributed edit whenever the report cannot
establish the required coverage or transfer.

Geogram may mutate attributed geometry only when its complete output can be
recovered as a unique exact subset/reorientation of source triangles. Split,
merged, moved, generated, or duplicate-ambiguous triangles remain unavailable.

## Stop Conditions

- A required protected feature regressed or was not evaluated.
- Source and candidate part identities are ambiguous.
- A generated face has no explicit assignment policy.
- A categorical assignment has conflicting source values.
- An indexed attribute or face-varying primvar lacks exact corner coverage.
- A worker reports completion without verifiable correspondence artifacts.
