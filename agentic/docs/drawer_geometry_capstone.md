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
the saved handoff. Nonfinite/malformed meshes, changed source geometry, any
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
payload acceptance. The original frozen independent evaluator remains required.

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
to evade a contact failure is not permitted.

For a mesh decision using `convexDecomposition`, the optional record below
exposes bounded native cooking controls. It is a configuration example, not
evidence that this setting succeeds for an arbitrary asset:

```json
{
  "collision_approximation": "convexDecomposition",
  "convex_decomposition": {
    "shrink_wrap": true,
    "error_percentage": 1.0,
    "hull_vertex_limit": 128,
    "max_convex_hulls": 64,
    "voxel_resolution": 1000000
  }
}
```

The record is allowed alongside the scalar physical properties and optional
mass-property vectors in the native target-ID decision. Omitting it leaves
existing cooking attributes unchanged. Providing it authors all five values,
using native0.4.13 defaults for omitted fields. Workflow resource bounds are
error0–100%, vertices4–255, hulls1–256 and voxels10,000–4,000,000; these are not
claims about the full native schema's limits. Invalid types, unknown fields,
incompatible approximations, non-mesh targets and conflicting existing attribute
types fail before batch mutation. Rebased decisions with different cooking
options cannot silently merge.

The source render mesh stays unchanged. The saved PhysX API and values must
still be exercised by the solver: inspect cooked floor height, cavity clearance,
initial contacts, constraint behavior and the actual free-payload task. A
synthetic cooking witness or successful schema readback is insufficient to
accept the original drawer.
