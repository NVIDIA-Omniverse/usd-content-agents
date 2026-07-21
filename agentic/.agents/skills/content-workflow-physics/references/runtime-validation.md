# Physics Runtime Validation

Runtime validation is the core review mechanism for physics authoring. It plays
the same role that final rendering and visual quality assessment play in the
material assignment workflow.

For the agentic Workbench workflow, runtime validation has two layers:

- solver-backed evidence from ovphysx or another selected runtime;
- visual behavior review over frames rendered from the generated
  `recording.usda`.

The solver-backed layer is authoritative for hard failures. Visual review may
turn an otherwise passing runtime result into a conditional result, but it must
not override solver load failures, non-finite trajectories, missing rigid
bodies, bounded-motion failures, initial-pose discontinuities, excessive ground
penetration, missing gravity response, or explicit body-count mismatches.

## Validation Tiers

T1 basic stability:

- USD opens;
- expected physics schemas are present;
- rigid bodies have colliders;
- property ranges are valid;
- dynamic friction is not greater than static friction;
- mass and scale are plausible;
- the asset can be loaded into the selected simulator when available;
- trajectory values are finite;
- the body does not explode, tunnel dramatically, or leave the scene.

T2 simulation match:

- run a deterministic scenario such as drop/settle, slide, bounce, tip, or
  freeform user-described behavior;
- compare trajectory metrics against expected behavior;
- record repair hints for mass, friction, restitution, collider approximation,
  or body grouping.

T3 real comparison:

- compare simulated behavior against user-supplied reference video, reference
  images, measured data, or textual behavior requirements;
- keep programmatic trajectory metrics authoritative when they directly measure
  the requested behavior, and use visual/VLM review only as supporting evidence.

## Runtime Contract

Prefer a real solver such as ovphysx for runtime validation. A simulator
operation should return:

- trajectory samples: time, pose, velocity;
- final pose and final velocity;
- body count;
- simulated duration;
- solver step count;
- simulator status and diagnostics.

The workflow should write a derivative simulation scene instead of mutating the
authored asset. Scenario construction must honor stage units and up-axis, add a
ground plane when needed, isolate the body under test, and preserve enough camera
or recording evidence for audit.

In the Workbench workflow, this solver interaction belongs behind the Workbench
runtime-validation operation. ovphysx may run in an isolated daemon because of
OpenUSD-version constraints, but that daemon is not an agent-facing API. The
agent should call Workbench, then inspect returned reports, trajectories,
recordings, failures, warnings, and repair hints.

## Metrics

Useful deterministic metrics include:

- settle distance;
- settle time;
- maximum linear speed;
- maximum angular speed;
- first bounce height;
- fell-over status;
- penetration or below-ground depth when available;
- finite trajectory and finite final pose;
- expected body count.
- first-step displacement versus ballistic displacement and solver tolerance;
- transformed final body-bounds penetration below the ground plane;
- early downward position or velocity response to gravity.

Visual review should inspect rendered simulation frames for parts separating,
no visible motion under gravity, implausible bounce or sliding, obvious
interpenetration/tunneling, stale or blank renders, and mismatches between the
trajectory metrics and visible behavior.

Validation failures should produce targeted repair hints:

- increase or decrease mass/density;
- adjust static or dynamic friction;
- adjust restitution;
- change collision approximation;
- move rigid body authoring to a common ancestor;
- preserve or repair articulation instead of adding a parent body;
- inspect scene units or scale before trusting mass.
