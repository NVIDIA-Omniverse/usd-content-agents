# Physics Workflow Policy

The physics workflow is agent-driven. Do not run the fixed
`apps/physics_agent` pipeline as the workflow engine.

Use usd-cli to inspect authored scene state, bounds,
existing materials, collider ownership, rigid bodies, joints, and articulation
structure. Visual geometry, collider geometry, and helpers are distinct roles.
The workflow agent—not the scene backend—groups logical components and infers
physics properties from geometry, visual material evidence, part function,
names, references, and user intent.

Plan topology through the existing workflow helpers. Apply an accepted,
digest-bound plan with usd-cli when it supports the
required primitives, otherwise retain the guarded workflow helper.

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
- exception: a component whose `component_role` is `unowned_static` (a floor,
  wall, or fixture the asset merely rests on) takes zero density and zero
  estimated mass, with real friction and restitution from its material, so
  fixture volume never inflates the simulated body's mass;
- static friction: 0.0 to 10.0, with common material estimates usually 0.0 to
  1.5;
- dynamic friction: 0.0 to 10.0 and not greater than static friction;
- restitution: 0.0 to 1.0.

Mass estimates must account for fill factor. Do not treat hollow shells, sheet
metal, tubes, frames, or thin covers as solid bounding boxes. Record
`{"code": "mass_scale_suspicious", "severity": "warning", "message": "..."}`
when geometry scale or mass plausibility is uncertain.

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
delete colliders, reparent prims, or mutate the source asset. Apply topology
plans through usd-cli or the retained workflow
`apply-topology-plan` helper. A missing usd-cli primitive does not move topology
policy into usd-cli or deprecate the workflow.

## Coverage Policy

Every logical component must be covered exactly once by an accepted decision or
an `unresolved_components` record with a specific reason. Visual evidence may
drive material/property reasoning without becoming an authoring target. Helper
prims, joints, and scopes are never independent physics candidates. Existing
collision geometry is an authoring target for properties but not a second
semantic component.

Scene optimization and correspondence-aware restoration are workflow-owned.
When an optimized inspection representation is used, make decisions in
inspection space, record source expansions, reject ambiguous mappings, and
author accepted decisions onto a source derivative through the configured
scene backend.
