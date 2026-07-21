# Physics Workflow Policy

The physics workflow is agent-driven. Do not run the fixed
`apps/physics_agent` pipeline as the workflow engine.

Use Workbench `inspect-components` and `inspect-topology` to inspect logical
components, source/inspection path mapping, bounds, existing materials, collider
ownership, rigid bodies, joints, and articulation structure. Visual geometry,
collider geometry, and helpers are distinct roles. The agent infers physics
properties from geometry, visual material evidence, part function, names,
references, and user intent.

## Property Contract

For each accepted physics decision, record:

- target runtime and validation profile;
- runtime and source prim paths;
- component type and component name;
- inferred physical material;
- density in kg/m3;
- estimated mass in kg;
- static friction;
- dynamic friction;
- restitution;
- collision approximation;
- rigid body grouping intent;
- confidence;
- rationale;
- quality warnings.

Use conservative property ranges:

- density: positive, normally below 50000 kg/m3;
- mass: positive, scene-scale plausible, normally below 1000000 kg;
- static friction: 0.0 to 10.0, with common material estimates usually 0.0 to
  1.5;
- dynamic friction: 0.0 to 10.0 and not greater than static friction;
- restitution: 0.0 to 1.0.

Mass estimates must account for fill factor. Do not treat hollow shells, sheet
metal, tubes, frames, or thin covers as solid bounding boxes. Record a
`mass_scale_suspicious` warning when geometry scale or mass plausibility is
uncertain.

## Authoring Policy

The durable output is a USD/USDZ with physics schema applied. Prefer one rigid
body on a common Xformable ancestor for a single multi-mesh asset, with
colliders on mesh leaves. Preserve existing articulated rigid-body hierarchies
and joints instead of adding nested or parent rigid bodies.

Author at least:

- `UsdPhysics.Scene` when missing;
- `UsdPhysics.CollisionAPI` on collider prims;
- `UsdPhysics.MaterialAPI` and physics material bindings for friction and
  restitution;
- `UsdPhysics.MassAPI` for density and/or mass when plausible;
- `UsdPhysics.RigidBodyAPI` only at the correct body root.

Collision approximation should be explicit. Use `convexHull` as the conservative
default, `convexDecomposition` when concavity materially affects behavior, and
simple bounding approximations only when they are intentional workflow tradeoffs.

Use `preserve_existing` when the component already has enabled collider paths;
do not add collision schemas to its visual evidence. Use `author_on_targets`
only with explicit visible-geometry targets when the component has no existing
colliders. Author component mass on `body_root_path`, not independently on each
render or helper mesh.

Topology repair is a separate, explicit decision. `mobility_intent=preserve` is
the default and forbids removing bodies or joints. Use a digest-bound topology
plan only when user or workflow context resolves the asset as `movable` or
`static`. The topology-plan allowlist may ensure or remove `RigidBodyAPI` and
remove fixed joints; it must not remove non-fixed joints, alter articulations,
delete colliders, reparent prims, or mutate the source asset.

## Coverage Policy

Every logical component must be covered exactly once by an accepted decision or
an `unresolved_components` record with a specific reason. Visual evidence may
drive material/property reasoning without becoming an authoring target. Helper
prims, joints, and scopes are never independent physics candidates. Existing
collision geometry is an authoring target for properties but not a second
semantic component.

Optimized Workbench sessions should make decisions in runtime/inspection space
and record source expansions for durable restore/export.
