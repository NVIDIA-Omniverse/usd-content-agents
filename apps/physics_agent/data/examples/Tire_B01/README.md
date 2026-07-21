# Tire_B01 example asset

`tire.usdc` (1.4 MB, USDC binary) is a materialized variant of the
**Isaac SimReady `Tire_B01` wheel-assembly tire**, shipped as the
canonical input for the `physics-agent refine` bounce-flow demo at
`apps/physics_agent/configs/tire_bounce.yaml`.

## Provenance

- **Source asset**: Isaac SimReady `Tire_B01/sm_wheelAssembly_tire_b01_01.usd`
  (NVIDIA Omniverse Isaac SimReady assets — refer to your Isaac
  SimReady distribution for license terms).
- **Materialization pipeline**: `material-agent run` was invoked on the
  source SimReady asset using the in-tree
  `apps/material_agent/configs/unified_example.yaml`-derived recipe.
  The pipeline ran `optimize_usd` (with `enable_deinstance: true` to
  strip the SimReady instance-proxy wrapper), `build_dataset_usd`,
  `build_dataset_prepare_dataset`, `predict`, and `apply` (flattened
  output) to produce the shipped `tire.usdc`. The VLM backend used
  for the predict step is unspecified here — operators who reproduce
  this asset should consult `apps/material_agent/configs/` for the
  current public backend selection.
- **Materialization date**: 2026-05-13.
- **Reference video**: `reference_media/tire_bounce_reference.mov` is a
  2.4 s, 720x960, 29.58 fps phone capture of a physical tire drop/bounce.
  It is bundled as optional visual judge evidence for the refine demo and
  can be passed to `physics-agent refine` with `--reference-video`.
- **Bundled textures**: `Textures/` ships three PBR maps
  (`t_rubber_new_a01_tile_alb.png`, `_nor.png`, `_orm.png`), downscaled
  from the SimReady 4K sources to 1K so the example asset stays under
  ~1.5 MB total. The USDC references them via relative paths
  (`./Textures/...`), which resolve correctly *only while `tire.usdc`
  lives next to its `Textures/` sibling*. Keep them together when
  copying the asset elsewhere. (Downstream USD exports — e.g.
  `apply_physics`'s `tire_physics.usdc` working file under
  `.tire_bounce/physics/` — go through `Usd.Stage.Flatten()`, which
  resolves every `@./...@` asset path to its current absolute
  `resolvedPath` and serialises those absolute strings into the new
  layer. The resulting derivative carries texture paths anchored at
  this checkout's filesystem location, so it is gitignored and not
  portable across machines. A principled fix — re-anchoring texture
  asset paths relative to the new layer's parent after flatten AND
  copying the referenced texture tree alongside the output — is
  deferred; see the ``FIXME(apply_physics-flatten-anchor)`` at
  ``apps/physics_agent/physics_agent/functions/apply_physics.py:_export_flattened_stage``
  for the full mechanism + the two-step contract a future fix has to
  honor.)
- **Pre-authored physics schemas**: the USDC has `UsdPhysicsRigidBodyAPI`,
  `UsdPhysicsMassAPI`, `UsdPhysicsCollisionAPI`,
  `UsdPhysicsMeshCollisionAPI` (with `approximation =
  convexDecomposition`), and a `PhysicsMaterialAPI`-bound material
  graph baked in. `physics-agent run apps/physics_agent/configs/tire_bounce.yaml`
  re-runs the public NIM/qwen classification on this shape and
  rewrites the physics values; the committed schemas exist mainly so
  the bare-USDC opens with sensible defaults for visual inspection.

The shipped USDC does NOT depend on the upstream SimReady tree at
runtime — `apps/physics_agent/configs/tire_bounce.yaml` points
directly at this file.

## Why ship a materialized asset

The raw SimReady tire is an instance-proxy wrapper that requires
`material-agent` to run before physics classification produces useful
material/density predictions. Shipping the materialized result lets
the `physics-agent` quickstart run end-to-end on a public install
without first wiring a material-agent pipeline. Re-deriving the asset from
the upstream SimReady source is outside this public example's scope.

## Working with this file

```bash
# Inspect the USD tree
wu print-usd apps/physics_agent/data/examples/Tire_B01/tire.usdc

# Re-render a preview (PBR textures load from ./Textures/)
wu render apps/physics_agent/data/examples/Tire_B01/tire.usdc \
  --output /tmp/tire_preview.png

# Run the physics agent classification + apply_physics step on it
physics-agent run apps/physics_agent/configs/tire_bounce.yaml

# Iteratively tune the bounce parameters with a small text-only smoke run.
# Requires physics_agent[tuning] plus a simulation backend such as OvPhysX.
physics-agent refine apps/physics_agent/configs/tuning/tire_b01_drop_settle.yaml \
  --physics-usd apps/physics_agent/configs/.tire_bounce/physics/tire_physics.usdc \
  --user-prompt "make this object bouncy" \
  --output-dir /tmp/tire_bouncy \
  --engine ovphysx --optimizer random \
  --max-trials 4 --max-iterations 3 --score-threshold 0.7

# Run the full reference-video tire-bounce refine example. This uses the
# bundled phone video as visual judge evidence; do not pass
# --reference-video-description unless you intentionally want to override
# frame-based interpretation. Configure the judge backend and credentials
# for your environment before running this example.
physics-agent refine apps/physics_agent/configs/tuning/tire_b01_drop_settle.yaml \
  --physics-usd apps/physics_agent/configs/.tire_bounce/physics/tire_physics.usdc \
  --user-prompt "Match the bounce behavior shown in the reference video." \
  --reference-video apps/physics_agent/data/examples/Tire_B01/reference_media/tire_bounce_reference.mov \
  --reference-video-frames 32 \
  --judge-reference-frames 32 \
  --judge-generated-frames 32 \
  --output-dir /tmp/tire_bouncy_refvideo \
  --engine ovphysx --optimizer botorch \
  --max-trials 30 --max-iterations 12 --score-threshold 0.9 \
  --seed 42
```

## Known caveats

- **Collision approximation**: the tire is a torus. The shipped
  `tire_bounce.yaml` config authors `collision_approx: convexDecomposition`
  so the rim hole is preserved
  in the physics collider. A `convexHull` approximation would fill the
  hole and produce a disc-shaped collider — fine for a drop test on
  flat ground but wrong for downstream multi-body / friction contact
  studies.
- **Camera framing**: the `tire_b01_drop_settle.yaml` refine scenario
  sets `camera_ground_bias_fraction: 0.75` so the recorded mp4 keeps
  both the falling tire and the ground in frame for the full
  `drop_height_m: 1.0` trajectory.
