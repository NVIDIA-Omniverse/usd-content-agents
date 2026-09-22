# Unscored source-preserving drawer workflow

This isolated branch is a follow-up to the frozen pilot, not a change to its
scored code or results. It does not contain the direct Astra baseline solution.

The new explicit `lossless_gltf` source-authoring mode admits static glTF2
triangle meshes by copying their source vertex and index arrays into USD at
meter scale. Original zero-area faces remain in the render representation.
Unsupported animation, skinning, compressed/morphed geometry, required
extensions or unsupported attributes fail explicitly. Materials are translated
to USDPreviewSurface; exact shader equivalence is not claimed. Dependencies
are contained and hashed, and a source receipt records each mesh's arrays.

`--render-topology-policy preserve_source` is an explicit visual handoff policy
available only with that intake, `--optimization-policy skip`, no render repair,
and no stage-metric rewrite. It compares the complete prepared source mesh
inventory, vertices, face arrays, default world transforms, units and axes with
the saved handoff. Time-sampled geometry or transforms are rejected at this static
boundary. Nonfinite/malformed meshes, changed source geometry, any
rigid body or any collider fail that visual-only boundary. The full original
strict topology report is retained. This policy does not establish collision
cooking, contact, articulation, load holding or physical task acceptance.

The default topology policy remains strict. Downstream Joint, Physics and
Validation must still execute and honor their own failed/conditional gates.
Warnings from shared USD validation remain warnings; this change does not
silence them or relabel a conditional overall handoff as a completed one.

Example Geometry call after the normal repository setup:

```sh
content-workflow-cli geometry run /path/to/original/drawer.gltf \
  --output-dir runs/drawer-geometry-fresh \
  --source-authoring-mode lossless_gltf \
  --render-topology-policy preserve_source \
  --optimization-policy skip
```

OVRTX evidence remains enabled by default. A CPU-only preflight may explicitly
use `--no-render-evidence`; it does not count as final visual evidence.

The protected-feature detector separately uses explicit analysis-to-source
vertex IDs. Its previous exact-position lookup could raise a KeyError after
floating-point averaging during seam welding. Neither mapping operation edits
the source mesh. The focused regression file is
`tests/test_drawer_capstone_geometry.py`; it contains source-fidelity positives,
negative mutations/physics misuse, strict-default behavior and the nonidentical
average witness. Synthetic tests are not the physical drawer demonstration.

## Subsequent unscored Physics extension

The native topology-only Joint handoff can require a guarded Physics endpoint
promotion before its component inventory separates moving and static parts.
If a one-shot decision cannot be rebased after that split, retain the failed
attempt and use its exact prepared derivative in a fresh native Physics run.
Do not relabel the failed attempt as accepted.

The resolved Physics decision now carries the authoritative inspected component
role. `unowned_static` components keep their collision/material operations and
receive no new rigid-body operation. Zero mass on an actual body retains the
existing USD automatic-mass behavior; zero alone does not mean static.

An optional `mass_properties` decision record accepts body-local center of mass,
positive principal moments of inertia, and a normalized `(w,x,y,z)` principal-axis
quaternion. Values use USD stage units and describe estimates supplied by the
reasoner, not measured ground truth. Invalid records fail before batch mutation.
An explicit inertia record cannot be carried across merged bodies or a changed
body-root path. This guard relies on the existing topology-plan operations,
which do not edit transforms; any future operation that changes a body frame
must reauthor or transform those values as well. The defaults remain scalar-only.

`tests/test_drawer_capstone_physics.py` checks target resolution through native
saved MassAPI readback, static preservation, invalid vectors and unchanged
scalar defaults. These synthetic checks do not establish drawer contact or
payload acceptance. Independent task execution remains required. The completed
capstone uses the separately versioned, unscored `source_clear_v1` fixture
described below, with the original controller and acceptance thresholds.

For a model-authored Physics decision, provide the intended moving/static roles,
mass assumptions, body-local frame and required collision openings as explicit
guidance. For example, on an already prepared Joint output with a valid component
inventory:

```sh
content-workflow-cli physics apply \
  --usd /path/to/native-prepared-source.usdz \
  --output-dir runs/drawer-physics-fresh \
  --runner codex --model gpt-6-astra --model-reasoning-effort ultra \
  --no-optimize --collision-approximation convexDecomposition \
  --simulation-engine ovphysx --duration-s 3 --dt 0.004166666666666667 \
  --runtime-placement-mode mounted --drop-height-m 0 \
  --additional-instructions-file physics-guidance.md
```

The optional record belongs alongside `physical_properties` in a native V2
component target decision, rather than inside its scalar dictionary. This is a
synthetic example for a body with metre/kilogram units, not drawer ground truth:

```json
{
  "physical_properties": {
    "estimated_mass_kg": 2.0,
    "density": 500.0,
    "static_friction": 0.6,
    "dynamic_friction": 0.6,
    "restitution": 0.0
  },
  "mass_properties": {
    "center_of_mass": [0.02, 0.03, 0.04],
    "diagonal_inertia": [0.1, 0.2, 0.25],
    "principal_axes": [1.0, 0.0, 0.0, 0.0]
  }
}
```

Use the actual inspected component/target IDs, source digest and body frame in
the surrounding decision. Do not assign these synthetic numbers to arbitrary
assets. The component role is derived from inspection and cannot be changed by
the decision. Explicit vectors for a static component are rejected.

When a native adapter rejects external texture locators, the supported
`usd-cli open` followed by `usd-cli save /path/to/source.usdz --flatten` can
produce a self-contained input. Verify source arrays, transforms and existing
joint/body opinions after packaging, publish fresh preparation evidence, and
retain the failed attempt. Packaging does not add physical validity. A successful
three-second mounted smoke test likewise does not demonstrate opening, closing
or free-payload retention; those require independent task execution.

## Mounted validation and packaged identities

Use `--runtime-placement-mode mounted` for an existing mounted mechanism. The
default `drop` behavior is unchanged. A zero drop height alone still requests
ground placement in drop mode; it does not preserve a constrained body's pose.
Mounted mode keeps all authored transforms and joint frames, including empty
world-anchor targets. It requires `metersPerUnit=1` and an omitted or zero drop
gap, and fails explicitly for unsupported units rather than partially scaling
a mechanism. Gravity remains 9.81 m/s² and initial velocities are reset as in
the existing smoke policy. Settling, finite/bounded motion, pose continuity and
the existing synthetic ground/penetration checks remain active. Mounted joints
can resist gravity, so free-fall displacement is not required.

After a harness defect is fixed, the supported deterministic executor can replay
an unchanged native V2 decision patch with `physics apply --direct-executor
--decision-patch /path/to/original-patch.json`, the same inspected source and the
explicit mounted mode. Retain the original failed run and verify the patch and
source hashes. Reapplying a decision does not substitute for actual solver or
visual validation.

A flattened USDA may retain a texture locator into a local USDZ. Dependency
identity now binds that locator through the complete archive bytes. Archive
changes invalidate the identity. Missing or unresolved members, remote locators,
path traversal, malformed syntax, non-USDZ containers and nested package
locators fail closed. This change does not alter the asset or extract textures.

An unchanged-patch replay also requires the exact inspected source identity.
The native source digest includes resolved layer identifiers; moving identical
bytes to a new path can therefore require a fresh inspection and decision.
If the purpose is only to revalidate the exact previously authored asset after
a harness correction, use the exported `validate_physics_runtime` API with that
asset and its bounded, original dependency root. Retain its actual native
`ValidationEvidence` and runtime report. This does not manufacture a successful
top-level authoring workflow or a model behavior assessment.

## Explicit collision cooking controls

The configured collision approximation is the initial fallback. An explicit
component decision may choose a representation appropriate to its geometry and
mobility: for example, actual triangle surfaces (`none`) for a static cabinet
and convex decomposition for its moving concave drawer. Later refinement pins
the actual per-component choices; coarsening a collider or changing filtering
to evade a contact failure is not permitted. A later, separately declared Joint
repair changes which actual static actor the endpoint resolves to; its effective
connected-actor exclusion and ideal-rail scope are disclosed below. It is a fresh
native attempt, not a silent amendment to an earlier failed asset.

For a mesh decision using `convexDecomposition`, the optional record below
exposes bounded native cooking controls. It is a configuration example, not
evidence that this setting succeeds for an arbitrary asset:

```json
{
  "collision_approximation": "convexDecomposition",
  "convex_decomposition": {
    "shrink_wrap": true,
    "error_percentage": 1.0,
    "hull_vertex_limit": 64,
    "max_convex_hulls": 64,
    "voxel_resolution": 1000000
  }
}
```

The record is allowed alongside the scalar physical properties and optional
mass-property vectors in the native target-ID decision. Omitting it leaves
existing cooking attributes unchanged. Providing it authors all five values,
using native 0.4.13 defaults for omitted fields. Workflow resource bounds are
error 0–100%, hulls 1–256 and voxels 10,000–4,000,000. Hull vertices are restricted
to 8–64 because the OvPhysX 0.4.13 decomposition backend rejects 128 even though
the USD schema accepts an integer property. The earlier 4–255 authoring bound
was too broad; schema readback did not establish backend support. Existing
failed assets and their receipts remain unchanged. Other resource bounds are
not claims about every backend setting being independently qualified.
Invalid types, unknown fields,
incompatible approximations, non-mesh targets and conflicting existing attribute
types fail before batch mutation. Rebased decisions with different cooking
options cannot silently merge.

The source render mesh stays unchanged. The saved PhysX API and values must
still be exercised by the solver: inspect cooked floor height, cavity clearance,
initial contacts, constraint behavior and the actual free-payload task. A
synthetic cooking witness or successful schema readback is insufficient to
accept the original drawer.

## Completed outcome and limits

The separate unscored capstone completed native Geometry, Joint, Physics and
Validation, together with five independent loaded-drawer trials. Geometry04
retains a **conditional** source-preserving handoff because original topology
and UV warnings remain. Native Joint03 is accepted; Physics10 passes its actual
mounted runtime and eight-frame Astra Ultra OVRTX review. Native Validation01
ends `completed / accept / pass`: its four required gates pass (static,
runtime, visual quality and package integrity). Its standalone
`cross_stage_integrity` check remains `not_evaluated`; no broader integrity
claim is inferred from that result.

The native Physics, Validation and independent task evidence bind the same
original authored USD SHA256:

```text
0ef7038845d569a2f502f38af9fb5b1e4e99cf06c5f4e04035e4c259b1b07156
```

The measured implementation is commit
`e640b8d6aa667745830db5e9bf87fd1b4cd763cb`, based on public commit
`a96faf9cb2f5c1f655fe0d60c0ccf57e3477b1aa`. Its integrated regression run passed
311 tests, with 15 skipped and 7 warnings. Later documentation edits do not
change which implementation produced the evidence. See the
[capstone evidence and reproduction guide](../../docs/experiments/astra-drawer-capstone-2026-09-21/README.md)
for source provenance, exact receipts, prior failures, task traces and declared
reproduction limits.

Independent readback preserves the original five glTF meshes, all 26,406
triangles, world transforms, meter units and Y-up. The moving upper drawer keeps
the original native estimate of approximately 6.16 kg and its explicit COM and
inertia; these are thin-panel estimates, not measured material properties. All
physics materials use friction 0.6 and restitution 0. The evaluated dynamic cooking
settings are shrink wrapping, 0.1% error, 64 vertices per hull, 128 hulls and
4,000,000 voxels. The earlier JSON example is not the measured configuration.
Actual initialized CPU queries, source comparisons and native logs supplement
schema readback. Only the exact registry-startup message documented as nonfatal
by installed OvPhysX 0.4.13 is scoped as nonfatal; all other startup, service,
thread and CPU-fallback warnings remain retained.

Joint03 starts afresh from the joint-free source-preserving Geometry output. It
changes the fixed endpoint from `/Asset/drawer_cabinet_0`, a parent Xform without
a native actor, to `/Asset/drawer_cabinet_0/Primitive_0`, the actual static cabinet
Mesh. The existing `collisionEnabled=false` value, identity joint frames, Z axis
and 0–0.30 m limits remain. Resolving the actual connected actor makes that
existing exclusion effective. No additional collision group, filtered pair or
disabled collider is introduced. This deliberately models an **ideal rail**;
it does not validate realistic cabinet-to-drawer clearance or rail friction.
The earlier Xform-endpoint assets and failed trials remain unchanged.

The original task payload spawn intersected the visible source floor by about
8 mm. Before the final asset trials, the separately frozen `source_clear_v1`
fixture moved the payload center to Y=1.0529997730255127 m, providing 10 mm of
source-floor clearance. Only that initial Y and the protocol ID differ from the
original specification. Controller, forces, seeds, contact, penetration,
retention and settling limits are unchanged; the original invalid spawn is
retained. Source-clearance checks and 16 positive/negative qualification tests
passed before the five trials.

All five seeds (11, 23, 47, 83, 131) pass on CPU PhysX with a free 0.5 kg payload
and a 40 N force cap. The drawer opens approximately 24.3 cm and returns closed;
worst closed-hold error is 0.1584 mm, worst closed-hold speed is 0.000934 m/s,
and maximum reported penetration is 0.01860 mm. Each payload remains retained
with 2,280 contact samples. An independent audit checks all 12,300 finite,
contiguous recorded samples, seeded inputs, controller forces, reports and
unchanged source/fixture hashes. These are five bounded simulation trials, not
a real-world reliability estimate or a test of every joint limit and load.

Native Validation's physical-behavior check consumes the genuine mounted-rest
runtime and visual-review evidence. The five loaded-drawer trials are a separate
required acceptance condition on the same asset hash; their success is not
inferred from a schema check or from adding arbitrary task JSON to Validation.
A later portable USDZ convenience derivative would have a new identity and
would not automatically inherit this acceptance.

This outcome required substantial isolated development: source-preserving
Geometry and vertex-mapping fixes, authoritative static roles and typed mass
properties, mounted validation, packaged-dependency identity, typed cooking
controls and corrected backend bounds, a strict native Physics evidence
consumer for Validation, and the fresh static-Mesh endpoint route. Failed
setup, authoring, runtime, visual-review and task attempts are retained. None of
these changes amend the frozen paired pilot or demonstrate that its pinned
baseline completed this task. Physics03's original root console log was
overwritten by the Physics07 launcher; the original per-run asset, decisions,
runtime, typed review and failure receipts survive, but that console is not
reconstructed. Conditional Geometry warnings, estimated physical properties,
ideal-joint semantics and the stated CPU/five-seed scope remain part of the
result; this is not a universal SimReady claim.
