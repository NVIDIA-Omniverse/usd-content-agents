# Geometry Repair

`geometry-repair` is the deterministic core for source-preserving diagnosis and
bounded repair of imported CAD, mesh, and USD assets. It preserves the source
package, records USD dependency and representation-role evidence, diagnoses
before mutation, runs typed workers under resource limits, compares candidates
with the immutable source, and emits a geometry-scoped outcome.

The claim scope is `geometry_repair.<profile>`. It does not certify final
materials, mass/inertia, friction, joints, drives, controls, runtime task
behavior, or complete SimReady conformance.

## Capability Status

| Capability | Status |
| --- | --- |
| Independent source-format validation | OpenUSD sources run `UsdUtils.ComplianceChecker`; GLTF/GLB sources run the official Khronos validator when `gltf_validator` is installed, otherwise the check remains explicitly `not_evaluated` and caps the outcome at conditional |
| USD dependency intake and localization | Integrated into `run_geometry_repair`; exact remaps or approved roots produce a hash-bound portable bundle, while unresolved paths block material-fidelity claims |
| Scalable self-intersection/coplanar audit | Integrated into diagnosis with explicit pass/fail/skip/indeterminate and resource evidence |
| Conservative mesh/USD repair, collision generation, runtime probes | Integrated through the approved worker registry and profile gates |
| Native B-rep inspection and healing | Not included in the public runtime; supply an authorized backend separately or provide validated USD/mesh geometry from the authoring provider |
| Detected negative-space/thin-feature candidates | Integrated before planning; changed render candidates are compared, while collision gates require role-scoped task/source authority |
| Corner/face/attribute correspondence contract | Exact USD reauthoring preserves indexed UVs, normals, uniform data, subsets, and prim paths through bounded deletion, winding repair, and agreeing generated patches; ambiguity refuses |
| Role-aware diagnosis and repair intent | Diagnosis, plans, certificates, manifests, and results separate `render`, `collision`, `brep_source`, and `helper` evidence. Agent-proposed intent may rank only measured, policy-approved operations and cannot alter gates |
| Source and compound collision audit | Deterministic metadata/path/measured part pairing drives per-part audit, reuse, regeneration, or review-required refusal |
| Adaptive collision planning | Every source/primitive/hull/CoACD option has a materialized hash-bound ledger and fixed fidelity, complexity, protected-feature, and runtime gates. Generated-hull accounting is part-aware; dense primitive fits and bounded reduced-hull candidates still require the complete critic. |
| Collision-only SDF reconstruction | Reconstructive rigid jobs may derive a bounded collision reference through an admitted SDF backend without rewriting attributed render USD. Every grid attempt records backend identity/resource/drift evidence and render digest identity; generated results remain conditional |
| Manifold seam analysis | Enabled, pinned, isolated, and integrated as non-mutating merge-vector analysis only |
| PMP classified-hole repair | Registry-enabled for explicit 5+ vertex accidental-hole intent; frozen source-boundary vertices may carry indexed UVs, face-varying normals, uniform data, material subsets, and bindings when exact boundary agreement succeeds |
| WildMeshing/MCUT hard-local adapters | Typed and refusal-tested, but shadow-only with no production registry authority |
| SDF reconstructive route | Backend-neutral `sdf_tools` operations with fixed resource bounds, signed default routing, exact backend evidence, and conditional-only reconstruction claims. The current qualified driver is source-locked OpenVDB 13 |
| External mesh working layers | OBJ/STL/PLY/GLB/glTF/3MF/OFF may enter repair only with authoritative units/up-axis (glTF supplies its specified meter/Y-up frame); X-up is mapped to valid Z-up USD, while animation, skin, and morph-target semantics fail closed pending a preserving converter |
| Articulated/contact-rich geometry evidence | Source-authored USD joint/link extraction, integrity-bound per-link/per-probe artifacts, sampled motion, and an authoritative fixed-property `ovphysx` insertion canary; final physical tuning, controls, and complete-profile certification remain downstream |

The integrated Phase 0–3 contracts and complete package suite are green. The
capability table above states the public acceptance boundary; executable proof
and regression coverage live in `agentic/packages/geometry_repair/tests/`.

## Profiles

- `visual_only`: finite render geometry with preserved source identity;
  intentional open sheets are allowed.
- `static_environment`: separate static triangle collision plus authoritative
  `ovphysx` cook-and-contact evidence for production certification.
- `rigid_pick_place`: watertight render bodies, bounded primitive/single-hull or
  pinned CoACD collision, protected-feature probes, and authoritative multi-pose
  `ovphysx` cook/drop/settle evidence.
- `articulated_rigid`: per-link render/collision validation and deterministic
  source-authored or caller-supplied sampled joint sweeps. Source extraction
  refuses missing joints, body mappings, render geometry, or collision geometry.
  The repair outcome remains conditional pending final Articulation, Physics,
  and Runtime Validation.
- `contact_rich`: caller-supplied axis, path, clearance, seated-stop, and SDF
  resolution geometry probes. A fixed-property `ovphysx` approach-to-seat
  canary can validate contact geometry, but the repair outcome remains
  conditional pending final friction, force, controller, and task validation.
- `deformable_or_cae`: reserved and fail-closed until its element-quality and
  solver validators exist.

## CLI

Use the CAD Agent and shared Geometry workflow for product operation. The
package CLI remains useful for deterministic development:

```bash
geometry-repair imported.usd \
  --out output/imported_repair \
  --profile rigid_pick_place \
  --production-use \
  --source-uri s3://approved-bucket/assets/imported.usd \
  --source-license LicenseRef-Company-Approved \
  --source-provenance-json source_provenance.json \
  --protected-features-json protected_features.json \
  --dependency-root /approved/assets \
  --dependency-remap-manifest exact_dependency_remaps.json \
  --budgets-json repair_budgets.json \
  --seed 17 \
  --collision-runtime-engine ovphysx
```

For a directly repaired polygonal mesh, also provide
`--source-meters-per-unit` and `--source-up-axis`. Geometry Repair preserves a
raw diagnostic copy and rejects physical-profile repair when that frame is
unknown; it never guesses OBJ/STL/PLY/OFF units. Dynamic GLTF/GLB inputs must
first use a converter that preserves animation, skin, and morph-target
semantics.

Use `--diagnose-only` to prohibit mutation. `fake` runtime evidence tests the
wiring only and cannot grant production certification.

For production use, source URI, license, and provenance are mandatory. An
inferred rather than caller-confirmed profile cannot certify. If an accepted
repair changes render geometry, the result remains conditional until accepted
OVRTX evidence and human visual review are recorded.

Adaptive CoACD search starts coarse and escalates only after a measured
failure. The default generated-collision budget is 128 hulls; callers may
explicitly request up to 256 when the target runtime accepts the added cost.
All fidelity, protected-feature, per-hull, and runtime gates remain fixed.

When `allow_reconstructive=true`, a rigid collision planner may invoke
`sdf_collision_rebuild` for a render-derived working source that is not a
measured solid. The collision-specific grid ladder defaults to a maximum of
128 cells and may descend to 32 to close only discretization-scale defects.
Every resolution is independently bounded and recorded. Render USD must retain
the same SHA-256 before and after every invocation. Collision success remains
conditional and never clears a render-role defect.

## Python Entry Points

```python
from geometry_repair import (
    RepairRequest,
    ScalableAuditBudget,
    audit_triangle_mesh,
    inventory_usd_stage,
    run_geometry_repair,
)
```

- `inventory_usd_stage(...)` inventories a USD stage without fetching remote
  dependencies or changing the source.
- `localize_usd_dependencies(...)` copies only approved dependencies, rewrites
  portable paths, and records source/localized SHA-256 evidence.
- `audit_triangle_mesh(...)` returns the complete resource-bounded predicate
  report for array-level use.
- `run_geometry_repair(RepairRequest(...))` executes the integrated profile
  workflow and writes canonical artifacts.

Specialized implemented APIs live in:

- `geometry_repair.protected_features` for the integrated protection-candidate
  and before/after probe models;
- `geometry_repair.correspondence` for integrated identity/refusal/measured
  correspondence and exact corner/attribute transfer;
- `geometry_repair.roles` and `geometry_repair.repair_intent` for
  representation-scoped evidence and fail-closed proposal validation;
- `geometry_repair.collision_audit` for integrated rigid source-collider
  evaluation;
- `geometry_repair.collision_planner` and `geometry_repair.compound_collision`
  for typed candidate ledgers and deterministic per-part pairing;
- `geometry_repair.dependency_localization` for exact remaps and portable
  dependency bundles;
- `geometry_repair.hard_mesh_policy` for typed native protocols and route
  selection;
- `geometry_repair.advanced_profiles` for articulated/contact-rich geometry
  reports;
- `geometry_repair.usd_articulation` for source-authoritative USD joint/link
  extraction and world-preserving nested-rigid-body normalization;
- `geometry_repair.runtime` for authoritative static and contact-insertion
  runtime canaries.

These module APIs are intentionally not all re-exported from the package root.
Only the call sites and registry described above grant automatic orchestration
or worker authority.

## Artifacts

An integrated run may emit:

- immutable source/dependency hashes and a resolved working snapshot;
- `source_format_validation.json` from an independent authoritative validator,
  or an explicit `not_evaluated` status when that validator is unavailable;
- `usd_intake.json` for USD dependency, role, hierarchy, material, primvar, and
  collision-state evidence;
- `dependency_bundle/dependency_localization.json` with exact remap, copied-file
  hash, rewrite, unresolved-path, and material-fidelity evidence;
- `diagnosis.json` plus the full scalable-audit report;
- `protected_feature_candidates.json` and per-changed-attempt comparison
  evidence;
- `repair_plan.json` and an immutable per-attempt ledger;
- source-relative fidelity and typed per-attempt/accepted-path correspondence;
- optional Manifold seam analysis, source-collision audit, compound part audit,
  per-part collision candidate-search ledgers, and collision-only SDF backend
  reports;
- separate `final/render.usd`, `final/collision.usda`, and composed
  `final/asset.usda`;
- `advanced_profile_evidence.json` for requested articulated/contact geometry;
- `repair_certificate.json`, geometry validation evidence, and
  `content_agents_manifest.json`.

Never discard a full scalable-audit, correspondence, collision-audit, or
advanced-profile report in favor of a lossy summary field.

`ProtectedFeature.affected_roles` names the representations that the feature
may gate. Explicit caller records default to render and collision for backward
compatibility. Detector-derived candidates are render-scoped because mesh
topology alone cannot prove that a visible opening, wire, or detail is required
in a rigid collision proxy. Pass collision affordances explicitly or promote
them from source/task evidence; role-excluded names remain in the collision
report instead of silently constraining reconstruction resolution.

`final/asset.usda` uses typeless reference roots beneath one default component
prim so the referenced render/collision default-prim schemas remain intact.
When no replacement collision layer exists, source colliders remain active.
When a validated replacement exists, source colliders are explicitly disabled
and the replacement stays active. Final package validation records and checks
all three active-collider counts.

## Native Worker Policy

`geometry_repair/worker_registry.json` is executable authority. Documentation,
an installed library, or an agent suggestion cannot enable a worker. Missing,
disabled, unpinned, license-unapproved, capability-drifted, or shadow-only
workers must remain unavailable.

The base `geometry-repair` distribution depends on the backend-neutral
`sdf-tools` contract. Install `geometry-repair[openvdb]` when the current
OpenVDB driver is required; higher-level content-agent applications select that
driver explicitly. A future admitted driver can satisfy the same neutral
workers without adding OpenVDB to the base dependency graph.
`geometry_repair/sdf_backend_qualifications.json` separately binds production
repair authority to benchmarked driver identity and capabilities. Adding a
driver to `sdf_tools` alone does not qualify it to publish repaired geometry.

The PMP builder reuses a clean checkout already at the approved commit without
contacting its remote. CI caches that pinned source tree. Offline or mirrored
PMP builds may provide clean Git checkouts with
`GEOMETRY_REPAIR_PMP_SOURCE_DIR` and `GEOMETRY_REPAIR_JSON_SOURCE_DIR`, and may
replace the corresponding origins with `GEOMETRY_REPAIR_PMP_REPOSITORY` and
`GEOMETRY_REPAIR_JSON_REPOSITORY`. Commit, origin, cleanliness, and complete
Git tree-inventory verification remain mandatory; these controls cannot select
different source. The OpenVDB wheel builder instead verifies the immutable
archive and source-tree digests recorded in
`openvdb_runtime/native/source-lock.json`; offline builds may provide only the
matching archive through `OPENVDB_RUNTIME_SOURCE_ARCHIVE`.

PMP is production-routable only for one explicitly classified 5+ vertex
accidental hole. Build the pinned executable outside the repository with
`scripts/build_pmp_patch.sh`, then provide both
`GEOMETRY_REPAIR_PMP_EXECUTABLE` and
`GEOMETRY_REPAIR_PMP_EXECUTABLE_SHA256`. Discovery independently hashes the
binary and rejects a missing or mismatched digest. This authority does not
extend to generic remeshing, inferred hole intent, or reconstructive repair.

The current `openvdb` driver requires the `openvdb==13.0.0+wu.3` wheel built from
OpenVDB 13.0.0 commit
`7c03e1f084873cd1b3422c7ff7aec6ee681b3b38`. `openvdb_runtime` validates ABI
13, installed and source-lock distribution identities, the source commit, and
loaded-extension SHA-256 evidence before use. Content code calls `sdf_tools`;
the admitted driver invokes OpenVDB's nanobind module and bounded tools overlay
in-process inside Geometry Repair's existing generic isolated worker. There is
no OpenVDB helper, daemon, or OpenVDB-specific subprocess. A successful
reconstruction remains conditional until independent visual/task review.

Geogram production routing likewise requires the pinned vorpalite 1.10.0
executable plus an approved digest supplied through
`GEOMETRY_REPAIR_GEOGRAM_EXECUTABLE_SHA256` or the read-only absolute file
named by `GEOMETRY_REPAIR_GEOGRAM_EXECUTABLE_SHA256_FILE`. The worker hashes
the discovered executable before checking its version and records that digest
in every admitted worker result.

`sdf_collision_rebuild` uses the same admitted backend as `sdf_rebuild` but
has only collision-role authority. It consumes a neutral world-space mesh,
applies the collision-specific resolution ladder, and emits a separate
collision layer. It cannot bypass render-worker UV/material refusal and cannot
author the render source.

WildMeshing and MCUT remain typed shadow adapters without production registry
authority. Do not report their protocol tests as production repair success.

See the capability table above for the current acceptance boundary and
`agentic/packages/geometry_repair/tests/` for executable proof.
