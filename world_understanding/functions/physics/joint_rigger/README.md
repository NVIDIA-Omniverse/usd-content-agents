# Joint Rigger v1 contract and facade

`world_understanding.functions.physics.joint_rigger` is the shared boundary for
structured USD articulation authoring. The current increment implements the
WP-R0/WP-R1 contract, offline reference oracle, artifact policy, execution
facade, WP-R2 owned topology author, and WP-R3 evidence-backed physics-schema
boundary. A consuming app translates its evidence into these models and either
calls an owned entry point or supplies a different concrete authoring backend.

The package deliberately separates evidence from execution. A rigged reference
may produce a reviewed golden request offline, but it is never a hidden input to
a production authoring run.

## Public API

The public package exports the version constants, all v1 contract models,
canonical JSON/hash helpers, the reference oracle, artifact targets, backend
protocol, facade, owned topology backend, and topology validation helpers.

The WP-R3 public surface also exports:

- `author_joint_rig_with_physics()` and
  `validate_authored_joint_rig_with_physics()` for the atomic combined path;
- `author_physics_schemas()`, `validate_authored_physics_schemas()`, and
  `validate_physics_plan_evidence()` for direct schema authoring and validation;
- `JointRiggerStageSnapshot`, `JointRiggerPhysicsSchemaSnapshot`,
  `capture_joint_rigger_stage_snapshot()`,
  `capture_joint_rigger_physics_schema_snapshot()`, and
  `validate_joint_rigger_stage_preservation()` for structural readback; and
- `physics_schema_counts()` for deterministic owned-schema inventories.

`JointRiggerInputV2` and `JointRiggerPlanV2` extend the same strict boundary for
multi-root forests and aggregate rigid links. V2 requests carry every graph
component root plus an exact source-to-authored mapping for each rigid-link
member. Aggregate authoring is available only through a backend that explicitly
declares both V2 and aggregate support; reopened outputs must match the sealed
pre-move member structure and world transforms.

Importing the contracts and facade does not require an app package or eagerly
import OpenUSD. Calling the USD identity or oracle functions does require the
project's USD dependency.

Use the oracle only while preparing fixtures or reviewed plans:

```python
from world_understanding.functions.physics.joint_rigger import (
    extract_reference_input,
    write_reference_input,
)

request = extract_reference_input(
    "fixtures/drawer-source.usda",
    "fixtures/drawer-rigged-reference.usda",
    source_uri="fixture://drawer/source",
    reference_uri="fixture://drawer/rigged-reference",
    joint_paths=("/World/Joints/drawer",),
)
write_reference_input("fixtures/drawer-joint-rigger-input-v1.json", request)
```

Run a reviewed request through an app- or integration-owned backend:

```python
from world_understanding.functions.physics.joint_rigger import (
    JointRiggerArtifactTargets,
    author_joint_rig,
)

targets = JointRiggerArtifactTargets(
    output_path="artifacts/rigged.usda",
    diagnostics_path="artifacts/joint-rigger-diagnostics.json",
    result_path="artifacts/joint-rigger-result.json",
)
result = author_joint_rig(request, backend, targets)
```

`backend` must implement the `JointRiggerBackend` protocol described below.

Author a reviewed topology plan with the owned WP-R2 backend:

```python
from world_understanding.functions.physics.joint_rigger import (
    author_joint_topology,
)

result = author_joint_topology(
    request,
    source_usd_path="fixtures/drawer-source.usda",
    artifact_targets=targets,
)
```

The owned entry point supports raw `.usd`, `.usda`, and `.usdc` roots and
preserves the source layer format. Source roots must be regular files; symlinked
source paths are rejected. A root with relative composition dependencies must
publish beside its source so those dependencies remain valid. USDZ, cross-format
output, root relocation with dependencies, legacy `component_name`, existing
joint graphs, and WP-R3 physics-schema fields fail closed before publication.

## Versioned v1 contract

All contract models are strict, frozen Pydantic models: unknown fields and
coercion are rejected. The input's only conflict policy is `"error"`.

The on-wire schema literals are
`world-understanding-joint-rigger-input-v1`,
`world-understanding-joint-rigger-plan-v1`,
`world-understanding-joint-rigger-diagnostics-v1`, and
`world-understanding-joint-rigger-result-v1`.

| Contract | Represented facts |
| --- | --- |
| `JointRiggerInputV1` | Source artifact identity, one versioned plan, optional explicit legacy compatibility input |
| `JointTopologyV1` | Stable joint ID, `revolute`, `prismatic`, or `spherical` type, distinct absolute `body0`/`body1` paths, and a normalized stage-frame axis where required. Oracle-generated IDs are absolute joint prim paths. |
| `JointPlanV1` | Required topology plus optional, provenance-backed limit, anchor, joint friction, drive, state, or mimic facts |
| `JointFrictionV1` | One finite, nonnegative PhysX joint-friction coefficient and its independent provenance |
| `RigidBodyPlanV1` | Exact rigid-body prim, optional SI mass/center/inertia, and exact source-backed collider prims |
| `MassPropertiesV1` | Positive SI mass and diagonal inertia, optional body-local SI center of mass and principal axes, plus one evidence receipt |
| `ColliderPlanV1` | Collision-owner target, explicit `PhysicsMeshCollisionAPI` presence, and optional `none`, `convexHull`, `convexDecomposition`, or `sdf` approximation |
| `ArticulationRootPlanV1` | One explicit articulation-root prim |
| `FieldProvenanceV1` | Evidence source, artifact identity, prim, properties, derivation, and human-readable evidence |
| `JointRiggerDiagnosticsV1` | Per-field `accepted`, `ignored`, `defaulted`, `rejected`, or `unresolved` decisions with stable reasons |
| `JointRiggerResultV1` | Status, canonical input/plan hashes, exact output identity, and diagnostics |

`mesh_collision_api=True` distinguishes a bare applied
`PhysicsMeshCollisionAPI` from no API. For compatibility with earlier v1
payloads, a non-null `mesh_approximation` also implies API presence; a
redundant explicit `True` is normalized away so both inputs have the same
canonical bytes and hash.

`JointStateV1` and `JointMimicV1` are versioned model surfaces, but the current
paired-reference oracle rejects state and mimic schemas because it cannot yet
prove replay for them. Likewise, a backend must reject any otherwise valid v1
field that it does not implement; model validity is not backend capability.
`JointFrictionV1` is a joint-level fact rather than a drive field. It is valid
for revolute and prismatic joints with or without a drive, but v1 rejects it on
spherical or mimic joints. An absent `joint_friction` is omitted from canonical
JSON, preserving the canonical bytes and hash of existing v1 plans.
Collider ownership follows the nearest planned rigid-body ancestor, and scalar
mimic facts cannot reference a spherical joint without an explicit degree-of-
freedom contract. Backend report JSON also rejects duplicate object keys before
the original bytes can be sealed for publication.

`canonical_json()` removes absent optional fields, sorts object keys, relies on
the models' canonical ordering for set-like collections, rejects non-finite
JSON numbers, and emits compact UTF-8 JSON. `canonical_sha256()` hashes those
exact bytes. Persist canonical requests with `write_reference_input()` rather
than a generic JSON dumper.

### Units and frames

The contract normalizes only fields whose names declare that normalization.
Other control values intentionally remain native to their OpenUSD or PhysX
schema so the contract does not invent a unit system.

| Field | v1 meaning |
| --- | --- |
| `topology.axis_stage` | Unitless, normalized signed direction in the stage frame. The oracle derives it from `physics:axis`, both local joint rotations, and endpoint world transforms. Spherical joints have no axis field. |
| `anchor.position_stage` | Position in the stage frame, expressed in the stage's length units. The oracle requires `localPos0` and `localPos1` to resolve to the same position. |
| Revolute `limit` | Degrees, matching the USD revolute-joint schema. |
| Prismatic `limit` | Meters. The oracle converts authored stage units using `metersPerUnit`. |
| `mass.mass_kg` | Kilograms. The oracle converts using `kilogramsPerUnit`. |
| `mass.center_of_mass_m` | Optional body-local position in meters. Body-level evidence converts with `metersPerUnit`. For descendant evidence, the exact equation is `R * (center_stage * metersPerUnit) + translation_stage * metersPerUnit`; there is no SI-to-stage conversion. |
| `mass.diagonal_inertia_kg_m2` | Kilogram-meter squared. The oracle converts both mass and length scales. |
| `mass.principal_axes` | Unit quaternion in real-first `(w, x, y, z)` order. Body-level and lifted evidence use the same deterministic equivalent sign. Lifted inertia axes are rotated into the owner frame; diagonal eigenvalues are unchanged by the rigid frame change. |
| `joint_friction.coefficient` | Unitless native `physxJoint:jointFriction` coefficient. It is not scaled by stage units and is independent of drive damping. |
| Drive values | Native values of the matching `PhysicsDriveAPI:angular` or `PhysicsDriveAPI:linear` attributes; optional `max_joint_velocity` remains the authored `PhysxJointAPI` value. No additional SI conversion is implied. |
| State and mimic values | Native values of their source schemas. Position, velocity, offset, frequency, and related scalars have no independent v1 unit tag or conversion. |

Source and reference stages must agree on effective `upAxis`, `metersPerUnit`,
and `kilogramsPerUnit`. Selected endpoint transforms, collider geometry, and
collider transform chains must also agree at default time; these fields are not
silently reframed or reshaped.

## Offline reference oracle

`extract_reference_input()` compares a pre-authoring source USD with a rigged
reference USD and emits only replayable, evidence-backed v1 facts. It supports
revolute, prismatic, and spherical topology; scalar revolute/prismatic limits;
complete anchors and drives; independent revolute/prismatic PhysX joint
friction; rigid bodies; complete mass/inertia; collision and mesh-collision
schemas; and one articulation root.

Complete authored mass facts on the rigid-body owner remain body-local. Any
additional descendant MassAPI evidence is a conflict. If the owner has no
MassAPI evidence, the oracle may instead consume exactly one nearest-owner
descendant collider that authors all four of
`physics:mass`, `physics:centerOfMass`, `physics:diagonalInertia`, and
`physics:principalAxes`. It lifts the center and inertia frame through the
static descendant-to-owner rigid transform, performs the length conversion
without an SI-to-stage round trip, and records the descendant path, all four
source properties, transform receipt, and unit equations in provenance. Exact
descendant mass facts already present in both source and reference remain
replay-preserved and are not duplicated on the owner. Differing source evidence
fails closed instead of being replaced or combined with reference-only facts.

The oracle fails closed with `JointRiggerContractError.code` and `.detail`.
Important restrictions include:

- Selected joint prims must be absent from the source and present, active, and
  defined in the reference. Omitted reference joint types require an explicit
  `allowed_omitted_joint_types` policy.
- Required endpoints, type, and axis must be complete and non-contradictory.
  Fields represented as static in v1 may not be time-sampled.
- Optional fields remain absent unless accepted evidence exists. A partial
  schema, property without its API, unsupported approximation, negative or
  non-finite joint friction, non-finite value, or unrepresented relationship is
  an error, not a default. Unknown `PhysxJointAPI` instances and
  `physxJoint:*` properties remain unsupported.
- Descendant mass lifting rejects non-collider contributors, instances and
  instance proxies, time-varying or connected transforms, scale, shear,
  reflection, reset transform stacks below the rigid-body owner, incomplete
  values, body-plus-descendant evidence, and multiple descendant contributors.
  Composed and raw mass-property and transform time samples, splines,
  connections, and ancestor value-clip metadata are inspected. It never chooses
  a contributor, duplicates replay-preserved mass, or combines mass records.
  Density and geometry/bounds-derived defaults remain unsupported.
- Collider geometry and its full transform chain must already exist
  compatibly in the source. Exact Xform instance-root colliders additionally
  require paired source/reference instance roots, identical composed prototype
  structure and geometry, and an explicit supported reference approximation.
  Non-instance or bare-API Xforms fail closed. The oracle does not manufacture
  collision meshes.
- Physics/PhysX APIs, attributes, and relationships anywhere in a selected
  body's replay subtree (or on its ancestors) that v1 does not model must
  already have exact source/reference parity, including applied-schema order.
  Physics-purpose material
  bindings, their collection definitions, and physics facts on bound material
  prims are included even when the material lives outside that subtree.
  Reference-only or contradictory unmodeled behavior is rejected instead of
  being dropped.
- Instance-proxy physics, ambiguous articulation roots, unsupported joint
  schemas, finite break thresholds, spherical scalar controls or authored
  spherical frame rotations, and unsupported nondefault schema behavior are
  rejected.
- Root and dependency-closure identities are checked before return, including
  mutation during extraction. Local USD/USDZ dependencies and package members
  contribute to the artifact identity; resolver-backed assets are hashed in
  bounded chunks rather than copied into memory as one buffer.
- Before OpenUSD composes a local stage, every authored local locator in the
  text or binary layer closure is descriptor-copied into a private mirrored
  filesystem projection. OpenUSD composes and inventories only that retained
  projection; devices and FIFOs fail before composition, and a later live-path
  swap cannot enter the read. The original files and stable symlink chains are
  rechecked against the projected identities before the operation returns.
  Every confirmed local locator is rewritten to an absolute path inside that
  projection. Unanchored locators that the active resolver maps to local files
  are descriptor-copied there; unresolved local-looking locators are redirected
  to absent private paths, while remote resolver identifiers remain resolver-owned.
- OpenUSD layers are process-global. Identity reads never force-reload a cached
  layer because that could discard a concurrent unsaved edit. A dirty cache, or
  a clean cached layer that no longer matches a fresh disk read, fails closed
  with `artifact_dependency_cache_dirty` or
  `artifact_dependency_cache_stale`; release the cached stage/layer and retry,
  or retry in a fresh process.

These restrictions make the oracle suitable for golden-fixture preparation,
not for repairing arbitrary production assets.

## Backend and facade obligations

A conforming backend provides:

```python
class JointRiggerBackend(Protocol):
    def probe(self, request: JointRiggerInputV1) -> None: ...

    def author(
        self,
        request: JointRiggerInputV1,
        artifact_targets: JointRiggerArtifactTargets,
    ) -> JointRiggerResultV1: ...
```

The obligations are strict:

- `probe()` performs dependency, version, API-shape, and supported-field checks
  without writing artifacts. Missing dependencies and incompatible APIs must
  fail before authoring with the typed facade errors. A backend whose
  `author()` repeats all probe checks inside the same protected resource
  lifecycle may set the exact class marker `author_runs_probe_checks = True`;
  the facade still validates both methods but skips the immediately redundant
  standalone call.
- `author()` writes only the physical staging `output_path`, `sidecar_path`,
  `diagnostics_path`, and `result_path` it receives. Authored filesystem
  references must derive from the immutable `publication_*` paths, never
  staging names. A local output URI must identify `publication_output_path`;
  an intentionally nonlocal logical URI is also permitted and remains bound
  to the actual generated bytes by the required hashes.
- Wrappers that must read local files to construct the request use
  `author_joint_rig_from_factory()`. The facade captures the final targets
  before invoking the factory, then carries that exact reservation through the
  same authoring and promotion transaction.
- Physical staging deliberately uses two ownership shapes. Diagnostics, result,
  and sidecar paths are absent children of unpredictable private owner
  directories whose no-follow descriptors remain held through cleanup. The
  generated USD root instead remains an unpredictable absent sibling of its
  final root so relative USD dependencies keep their publication semantics.
  The default facade path does not cleanup-own that sibling until successful
  no-follow validation binds it. Owned backends may instead use the
  facade-provided created-file binder while the creator descriptor is still
  live, binding the exact newly created inode before any fallible post-copy
  work. The combined topology-and-physics backend first validates its frozen
  private source and then uses this immediate binding for the generated root
  and both reports, so every post-copy failure remains descriptor-cleanable.
  A bound descriptor remains live until an actual commit or cleanup is
  recorded. If authoring fails before either binding path and the backend-known
  root name exists, cleanup preserves the exact private name and reports the
  preservation instead of risking deletion of a replacement; no final
  artifact bundle is published.
- A publishable call returns `status="succeeded"`, writes one regular generated
  root and both valid v1 reports, and returns the exact persisted result. Each
  JSON report is read through a non-symlink regular-file descriptor and is
  limited to 64 MiB. Accepted report bytes are copied to facade-private,
  read-only inodes; held descriptors are revalidated and copied into fresh
  promoter-owned target inodes, which are sealed read-only before exposure.
  Later writes or pathname replacement of backend-known reports therefore
  cannot change what is published.
- `input_sha256` and `plan_sha256` must match the request. The output identity
  must bind its reported URI, root bytes, and complete dependency bundle.
- Diagnostics must account for every fact present in the plan. Each planned
  fact must be `accepted` with provenance exactly matching that fact; missing,
  duplicate, unexpected, rejected, or provenance-mismatched decisions prevent
  publication. Joint decisions use relative paths such as
  `topology.body0`, `limit.lower`, and `drive.stiffness`. Body decisions use
  top-level paths such as `rigid_bodies[/World/Base].rigid_body`,
  `.mass.mass_kg`, `.mass.center_of_mass_m`, and
  `.colliders[/World/Base/Collision].mesh_approximation`; the articulation
  decision is `articulation_root`. Diagnosed defaults for absent fields are
  narrowly allowlisted. Legacy compatibility remains explicit as described
  below.

The facade validates local input roots and dependency closures, protects every
input locator from output aliasing, detects input mutation, and validates the
persisted reports and generated dependency layout. Targets are normalized once
to absolute lexical paths; caller targets may not alias or nest each other.
Owned raw-source sealing copies every accepted root into a Linux memfd with
write, grow, shrink, and further-seal operations disabled before the descriptor
crosses a trust boundary. Root size has no product-level byte ceiling; hosted
callers may enforce transport budgets, while local callers remain bounded by
available system resources. Auxiliary dependencies retain fixed,
non-request-tunable ceilings of 16 MiB per file, 256 unique files, and 4,096
captured references. A dependency-limit failure closes every earlier snapshot
and does not allocate or copy the offending memfd.

Hosted admission and deployment limits remain separate from this reusable core
contract. Joint Agent Service rejects inputs above `JA_MAX_UPLOAD_SIZE_MB`
(500 MiB by default) before pipeline execution and logs the accepted byte size;
its shipped Helm and Docker Compose configurations also set container memory
limits. Operators must tune upload size and request concurrency to their memory
budget and monitor container memory pressure. Direct library and local CLI
callers intentionally have no product root-size ceiling and own cancellation and
process-resource limits at their orchestration boundary.

Publication is a rollback-capable, same-filesystem transaction. Reports and an
optional composition sidecar are promoted first and the generated root is the
commit point promoted last. An incomplete result, backend exception, identity
mismatch, or promotion failure does not publish a partial bundle; an existing
complete bundle is restored when rollback succeeds. Consumers should therefore
treat the root's presence as the bundle commit marker. Publications acquire
nonblocking advisory locks directly on held physical parent-directory
descriptors, deduplicated and acquired in deterministic device/inode order.
This transaction protects against process interruption and concurrent writers;
it is not a storage power-loss durability protocol, and does not claim that
directory-entry updates survive a host or filesystem crash without recovery.
Symlink and bind-mount aliases of one parent therefore contend without any
replaceable lock-file entry. All publications under one physical parent
serialize, even when their final filenames are disjoint; publications whose
physical parents are disjoint remain concurrent. Kernel locks are released on
descriptor close or process exit, and publication creates no persistent lock
files. Backup, promotion, rollback, and private-report cleanup use held
parent-directory descriptors and revalidate their physical identities;
duplicate physical targets are rejected before publication. Descriptor-bound
staging reservations also capture each final target's parent inode before any
backend or wrapper input read runs. An existing target is retained through a
no-follow descriptor with its full mutation-sensitive stat state and a
streaming file, symlink-payload, or content-plus-physical-tree digest. Those
states are rechecked under the publication locks before and after backend
prebackup validation and immediately before each backup. Backups receive a
fresh post-rename baseline, remain fingerprinted through the root commit gate,
and are deleted after commit only while unchanged. A target created, removed,
replaced, or modified in place after reservation therefore fails as concurrent
drift; the caller must start a fresh transaction rather than letting a stale
bundle legitimize the new entry. Descriptor-bound report and generated-root
promotion copies exact validated bytes into fresh targets under the same
transaction locks. The facade holds the generated root
through a read-only descriptor, removes its write bits, and rechecks its stable
SHA-256 before promotion, so a backend-held writer cannot mutate the committed
root inode. A composition sidecar is first deep-copied into a facade-owned tree
with distinct inodes, sealed without write bits, and bound to a retained
directory descriptor plus a versioned exact-tree digest. The promoter then
copies that descriptor tree recursively with fd-relative, no-follow,
nonblocking reads into its own private target-parent tree; it verifies both
trees and the already-exposed evidence again at the root commit boundary.
Every caller-visible artifact-tree walk uses the same fixed trust-boundary
ceilings: at most 64 levels, 100,000 entries, and 8 GiB of aggregate regular
file or link-payload bytes. Capture, exact-tree hashing, detached copying,
mount validation, and descriptor-relative cleanup fail closed with a stable
limit error. Directory names are collected through bounded descriptor-relative
scans before sorting, and mutation-detection rescans stop at the original
bounded count. Existing targets are checked against those ceilings before any
backup or cleanup quarantine rename, so an oversized tree cannot trigger a
partially destructive traversal. The ceilings are intentionally not
request-tunable.

Composition validation after sealing is authoritative: the root and optional
sidecar are projected from their retained descriptors under the exact
publication basenames before dependency inspection. For no-sidecar outputs,
URI and symlink locators are rejected before the normalized inventory is
captured. Dependency structure is captured on both sides of opening a
read-only, non-symlink descriptor for every external backing file; identity
content hashes are then derived only from those retained descriptors, never
from a later live-path read. Package-relative dependencies require a sidecar
and are rejected in no-sidecar mode. The locked precommit callback rechecks
request inputs and all sealed artifacts after evidence promotion immediately
before the root replacement. When the promoter reaches this gate it invokes
the callback exactly once under all target locks, and never after the root
rename. Descriptor-backed directory sources remain caller-owned after
successful promotion so their creator can clean the private source tree. That
atomic replacement is the commit point; there are no fallible final-root
reopens after it. Subsequent staged-source and backup cleanup failures are
reported as committed cleanup outcomes rather than rolling the root marker
back. Regular-file opens and projection copies use
no-follow, nonblocking descriptor reads, so a raced FIFO or pathname
substitution fails closed instead of hanging or changing the proven closure.
The owned topology backend binds the exact root inode and parent before its
final validation, proves that validation did not change the retained stat state
or SHA-256, rewrites final-relative dependency locators in place on that same
inode, then reopens it read-only and copies from the retained descriptor. A
stable pathname replacement before binding is therefore validated and rejected,
while a replacement or in-place mutation during validation, rebinding, or copy
fails the retained-state check.
Referenced MDL and MaterialX documents receive an additional bounded recursive
closure check: local imports, includes, and texture/resource paths must resolve
to regular files inside the declared sidecar (or already be represented in a
no-sidecar identity). The no-sidecar precomposition projection adds each
reachable local descendant as an `opaque_asset` identity record and retains it
through descriptor-sealed publication validation. Absolute paths, URIs,
unbounded traversal, patterns, DTDs/entities,
and unapproved MDL runtime modules fail closed. The MDL core modules,
`::nvidia::core_definitions`, and the exact Kit-provided `::OmniPBR` module are
the explicit runtime-module boundary. Relative modules, including packaged
OmniPBR sibling chains, must remain in the artifact. Bounded MDL `using ...
import ...` clauses are parsed by their source module, and sibling resources
may traverse parent segments only when the normalized target remains inside
the already bound artifact root.
Private sidecar cleanup is relative to a held parent descriptor. The same
fd-relative, no-follow, inode-bound deletion rules apply to private input
projections, so a raced child symlink is unlinked without chmod or traversal of
its external target. These
mechanisms use POSIX `flock` on the supported Linux, Linux-container, and WSL2
runtime targets; native Windows is not a supported runtime target. Backend
staging reservations retain both their original parent descriptors and the
exact owner/root descriptors after ownership is established. Cleanup checks
each distinct displaced or recreated lexical parent once, never reuses retired
device/inode authority, and reports renamed or multiply linked residuals
instead of guessing ownership. If
cleanup alone fails after the bundle has committed, the facade raises
`JointRiggerPostCommitCleanupError` with `committed=True` and the
`committed_result`; callers must not treat that outcome as an unpublished run.

The concurrent-writer contract covers coordinated publications and mutations
of backend-visible or original names; cleanup of those known names remains
descriptor-relative, identity-checked, and mount-checked. An uncoordinated
principal with parent-directory mutation rights that enumerates and deliberately
targets a freshly generated 128-bit private quarantine name is outside this
contract. This includes same-UID and in-process code. The exclusion does not
relax any identity, mount, or known-name guarantee.

## Legacy `component_name` compatibility

There is no implicit label fallback. A component-name-only backend may receive
`LegacyComponentNameCompatibilityV1` only when the caller explicitly adds each
`prim_path`, `component_name`, and `source_field` assignment to the request.

The result diagnostics must contain one decision for every assignment. A value
copied from an authored `component_name` is `accepted` with provenance; a value
derived from a role is `defaulted` with
`reason_code="legacy_component_name_compatibility"`. When compatibility was
not requested, exactly one top-level ignored or rejected
`legacy_component_names` decision proves that no fallback occurred.

## Opt-in Joint Agent Stage 2 bridge

The Joint Agent currently exposes a transitional, direct opt-in helper:

```python
from joint_agent.functions.joint_rigger_core_bridge import (
    author_stage2_candidate_edges_via_core,
)

result = author_stage2_candidate_edges_via_core(
    input_usd_path="source.usda",
    articulation_candidates_path="articulation-candidates.json",
    artifact_targets=targets,
)
```

This adapter reuses the existing candidate-edge authorer behind the shared
facade. It accepts ready Stage 2 topology and source-backed scalar limits only,
records the existing body1-origin anchor behavior as a diagnosed default, and
rejects body physics, articulation root, anchor, drive, state, mimic, and legacy
`component_name` inputs. Ready spherical candidates are also rejected because
the existing authorer writes local frame orientation that v1 cannot represent.
It does not consume predictions or silently fall back to the legacy
wheel/component label path.

Stage 2 JSON is bound once into a sealed Linux descriptor before hashing or
parsing and is bounded to 64 MiB. Its provenance freezes the captured resolved
path, parsing and v0 authoring consume private copies of the same sealed bytes,
and the descriptor plus configured-path authority remain live through the
locked inner publication. A candidate-path FIFO, alias retarget, replacement,
or collision with a final target therefore cannot block, change the authored
plan, or destroy the captured candidate.

The via-core wrapper uses the facade request-factory entry point: final targets
are captured once before Stage 2 source/candidate validation or request
construction, and that same reservation remains authoritative through
publication. Direct backend calls likewise snapshot their target entries before
dependency scanning or probe consumption.

Callers must choose this helper directly; existing Joint Agent pipeline and
service schemas are unchanged. USDZ-to-raw-USD runs also require the facade
sidecar target named `<output-stem>_assets` beside the output.

Facade-driven calls reuse one exact-object probe proof between `probe()` and
`author()`. Direct calls to a shared `Stage2CandidateEdgesBackend` that have no
matching proof safely repeat preflight under the backend lock, so those direct
fallback validations are serialized. Abandoned proofs hold only weak request
references and the pending-proof table is bounded.

## Owned WP-R2 topology author

`OwnedTopologyBackend` and `author_joint_topology()` implement app-independent
topology authoring from an explicit `JointRiggerInputV1`. They author only
revolute, prismatic, and spherical joints under
`<defaultPrim>/Joints/<sanitized_joint_id>_<topology_sha12>`. The backend
preserves exact body relationships, reconstructs local frames from the signed
stage-frame axis, converts prismatic limits from contract meters to stage units,
and writes only complete source-backed drive opinions plus independently
source-backed scalar joint friction. Passive friction does not manufacture a
`PhysicsDriveAPI`. An omitted anchor is diagnosed as a body1-world-origin
default; it is never presented as accepted source evidence.

The backend snapshots the source root and every local dependency into sealed
Linux descriptors before OpenUSD composition. Authoring occurs in a private
projection derived from those descriptors, and the projection is validated,
restored to final-output-relative dependency paths, frozen, and copied into the
facade staging transaction. Source/dependency mutation and lexical symlink
retargeting during authoring therefore cannot change the bytes used to build the
output. Final output, diagnostics, and result paths are checked against that
bound local closure even when the request uses a logical or remote artifact URI.
The owned backend performs its complete probe checks inside that authoring
projection, so a facade-driven run binds, hashes, and projects the source closure
only once. Calling `OwnedTopologyBackend.probe()` directly remains a complete,
non-writing standalone probe.
The original source hierarchy, transforms, applied schemas, and authored
PrimSpec/property-spec metadata, values, time samples, relationship targets,
list operations, and property set are checked before publication. The only
ignored source-layer scaffold is an opinion-free ancestor `over` created solely
to contain a planned joint. Ambiguous endpoints, time-varying transform chains,
singular transforms, existing joints, and unsupported fields fail with
machine-readable contract reasons. Composition-aware hidden-joint discovery is
bounded to 1,000,000 prim visits and 1,024 inactive-ancestor activation rounds;
these fixed trust-boundary ceilings are intentionally not request-tunable.

`validate_joint_topology_plan()` performs the non-mutating preflight against an
open stage. `validate_authored_joint_topology()` verifies a topology-only output
against the exact plan and optional diagnostics. Joined WP-R2/WP-R3 artifacts
use the combined validator exported by the WP-R3 package increment rather than
weakening this strict topology-only boundary.

## Owned topology and physics-schema boundary

The shared package now implements owned WP-R2 topology authoring and the
evidence-backed WP-R3 schema subset. `author_joint_rig_with_physics()` composes
both phases in one facade transaction; semantic joint IDs remain stable in the
result while `authored_prim_path` identifies each deterministic USD prim. The
physics phase emits the exact leaf decisions required by the facade, with the
plan provenance repeated on every accepted mass, center-of-mass, inertia,
collider, state, friction, drive, and mimic fact. Direct and combined
physics-schema authoring emit an
independent `joint_friction.coefficient` decision and author `PhysxJointAPI`
exactly once for the union of planned max-velocity and friction opinions.

`validate_physics_plan_evidence()` is a pure, stage-free readiness check used by
the combined backend probe. It rejects incomplete or template-default physics
facts before a staging root is allocated. The probe also checks pre-R2 topology,
but intentionally defers R3 stage compatibility until deterministic joint prims
exist inside the authoring transaction. Direct `author_physics_schemas()` calls
additionally snapshot the active edit layer and restore it after any apply or
post-validation failure. A planned prim may carry one sole, non-empty explicit
`apiSchemas` opinion containing foreign schemas such as `IsaacLinkAPI` and an
ordered subset of its exact planned schemas. A planned descendant collider may
also preserve one source-owned `PhysicsMassAPI` with positive finite static
float `physics:mass` and/or `physics:density` defaults; the complete planned
rigid-body mass and inertia remain authoritative. The author converts only this
validated single-contributor opinion to an equivalent prepend before adding
missing owned schemas, then proves that the raw composed token sequence and
registered applied schemas are unchanged. Unplanned R3-family tokens, blocked,
connected, sampled, unsupported, multiply-authored, or ambiguous source mass
facts, and any composition drift fail closed with exact edit-layer rollback.
Owned attributes with authored connections, including an explicit empty
connection list, fail closed and connection state is retained
in structural snapshots. Instance-proxy value-clip coverage uses one shared
proxy-expanded stage traversal bounded to 1,000,000 visited prims and 16,384
retained covered paths. The downstream value-clip audit is separately bounded
to 65,536 aggregate ancestor visits. These fixed trust-boundary ceilings are
not request-tunable. Articulation-root discovery has an independent shared
composed/prototype budget of 1,000,000 prim visits and 16,384 retained root
paths; its inactive-subtree scan retains the shared topology validator's fixed
budget. Structural and physics-schema snapshots, plus public schema counting,
stream composed prims under a separate fixed 1,000,000-visit ceiling. Existing
joint discovery shares one 1,000,000-visit budget across the proxy-expanded
stage and every prototype and retains at most 16,384 joint paths, including
inactive-subtree matches. Ordinary OpenUSD failures during read-only preflight
and validation are normalized to stable contract errors.

Mesh colliders preserve all three v1 representations: no
`PhysicsMeshCollisionAPI`, a bare explicitly represented API, or an API implied
by an explicit approximation. The author never applies the mesh API for the
first state and never invents an approximation for the second. Non-Mesh GPrims
reject both mesh-schema representations. The owned R3 author also accepts an
exact Xform instance root as a collision owner when its plan explicitly carries
`PhysicsMeshCollisionAPI` and one of the supported approximations, including
`none`. Non-instance Xforms and bare-API Xform plans fail closed. The offline
reference oracle admits the same narrow Xform instance-root surface only when
paired composed prototype geometry and transforms match exactly; it does not
infer plans from instance proxies or unowned PhysX schemas.

## Current non-claims

The transitional bridge still delegates to the existing app authorer and is
not a generic shared authoring implementation.

It also does not expose a new service/client surface, certify arbitrary assets,
or claim Gate 3 readiness. Gate 3A schema checks, Foundation Gate 3B execution,
joined Gate 2/3 identities, residual ownership, dynamic limits, contact,
containment, stable simulation, no-explosion behavior, and independent human QA
remain separate evidence and release-gate work.
