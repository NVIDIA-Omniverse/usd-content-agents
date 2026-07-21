# Container_Gray_C04 example asset

`container.usdc` (1.1 MB, USDC binary) is a self-contained materialized variant
of the NVIDIA Isaac SimReady `Container_Gray_C04` warehouse container. It is the
public input for a two-stage Material Agent and Physics Agent refinement QA flow.

## Provenance

- **Source asset**:
  `s3://omniverse-content-production/Assets/Isaac/6.0/Isaac/SimReady/Industrial/Warehouse/Containers/Container_Gray_C04`
  (NVIDIA Omniverse Isaac SimReady assets - refer to your Isaac SimReady
  distribution for license terms).
- **Source USD**: `sm_container_gray_c04_01.usd`.
- **Materialization date**: 2026-07-09.
- **Normalization**: the source lid and body were deinstanced, normalized into
  one `/RootNode` rigid body with two child `convexHull` colliders, and stripped
  of the redundant fixed joint. Material bindings were repaired and the source
  component masses were preserved, for a total authored mass of 6.631465 kg.
- **Runtime dependencies**: none. The shipped USDC has no external layer,
  texture, or material references and does not require access to the source S3
  tree.

## Run The Example

```bash
# Inspect or render the public input.
wu print-usd apps/physics_agent/data/examples/Container_Gray_C04/container.usdc
wu render apps/physics_agent/data/examples/Container_Gray_C04/container.usdc \
  --output /tmp/container_preview.png

# Assign blue molded-plastic materials using the public NVIDIA NIM backend.
material-agent run apps/material_agent/configs/container_blue.yaml --clean

# Refine the blue container's slide behavior with real OvPhysX trials.
physics-agent refine apps/physics_agent/configs/tuning/container_c04_slide.yaml \
  --physics-usd apps/material_agent/configs/.container_blue/output/output.usd \
  --user-prompt "Make this closed blue plastic warehouse container slide realistically across a flat dry industrial floor after a gentle horizontal push, decelerating smoothly and coming naturally to rest while its lid and body remain together." \
  --output-dir /tmp/container_blue_refine \
  --engine ovphysx --optimizer botorch \
  --max-trials 8 --max-iterations 3 --score-threshold 0.9 \
  --seed 42
```

The Material Agent output is written under
`apps/material_agent/configs/.container_blue/`. The generated material and
refinement outputs are intentionally not checked in.

## QA Acceptance

- Both visible parts receive the `Plastic Dark Blue` library material.
- Refine terminates with `approved` and a judge score of at least `0.90`.
- The container slides in +X, settles without bouncing or reversing, and does
  not fall over or penetrate the ground.
- The lid and body remain together as one rigid assembly.

The reference run reached `0.98` on iteration 2. Optimizer parameters and exact
scores can vary by model and backend. Inspect the final scenario and tuned
parameters as well as the aggregate judge score because refine may expand the
parameter set or adjust scenario fields between iterations.
