# VoMP Rigid-Body and Volume-Deformable CLI Guide

Physics Agent integrates the official
[VoMP](https://github.com/nv-tlabs/VoMP) package to infer a complete density
field from calibrated OVRTX renders. The field can drive either strict
rigid-body mass properties or a separate Newton-first volume-deformable
contract. The rigid workflow preserves existing geometry, colliders, and
physics-material opinions; the deformable workflow preserves visual geometry
and adds generated tetrahedral simulation geometry.

```text
USD + target prim -> OVRTX renders -> VoMP -> MassAPI or volume-deformable USD
```

There is no Blender dependency or fallback in this path.

## Rigid-Body One-Command Run

Prepare an official VoMP checkout at the supported commit with its own Python
3.10 environment and downloaded weights. Then run:

```bash
physics-agent run-vomp \
  path/to/simulation_ready.usda \
  path/to/object_vomp.usda \
  --target-prim /World/Object \
  --vomp-root path/to/VoMP
```

The default contract is pinned to VoMP commit
`ac6826f25ca5cfa122eb7bacb74be91b5a9eadf4` and exact SHA-256 digests for
`weights/inference.json`, the geometry checkpoint, MatVAE checkpoint, and
normalization parameters. These pins identify the files distributed by the
official
[NVIDIA VoMP model repository](https://huggingface.co/nvidia/PhysicalAI-Simulation-VoMP-Model)
and referenced by the pinned checkout's `weights/inference.json`. A dirty
checkout, a mismatched artifact, an incompatible worker protocol, or an
incomplete field fails before USD is published. VoMP uses its own interpreter
because its Python 3.10, Torch, CUDA, spconv, and xFormers stack is intentionally
not installed into Physics Agent's Python 3.12 environment.

To intentionally use a different attested artifact, repeat
`--expected-artifact-sha256 KEY=SHA256` for each changed pin. Valid keys are
`config`, `geometry_checkpoint_dir`, `matvae_checkpoint_dir`, and
`normalization_params_path`; omitted keys retain the checked-in defaults.

The command writes:

- the output USD with VoMP-authored `MassAPI` values;
- `<output>.vomp_mass_properties.json` with render, model, and integration
  provenance;
- an evidence directory containing the isolated render USD, metric PLY, OVRTX
  images/calibration, VoMP NPZ, worker manifest, and worker log.

OVRTX output is not guaranteed to be pixel-deterministic. Camera sampling and
model seeds are fixed, and every run records the exact ordered render digest so
results are auditable instead of being presented as bitwise reproducible.

## Volume-Deformable One-Command Run

`run-vomp-deformable` uses the same attested OVRTX and official VoMP runtime,
then maps the complete voxel field to the AOUSD draft volume-deformable profile
consumed by Newton 1.4:

```bash
physics-agent run-vomp-deformable \
  path/to/object.usda \
  path/to/object_vomp_deformable.usda \
  --target-prim /World/Object \
  --vomp-root path/to/VoMP
```

The adapter creates a globally conforming `UsdGeom.TetMesh` using six
positive-volume tetrahedra per occupied voxel. It preserves spatial density as
per-point `physics:masses` using the tetrahedral FEM lumping rule. It authors
the draft `PhysicsBodyAPI`, `PhysicsDeformableBodyAPI`,
`PhysicsVolumeDeformableSimAPI`, and `PhysicsVolumeDeformableMaterialAPI`
tokens with canonical `physics:` attributes. The pseudo-base body token is
explicit because the draft APIs are unregistered. It never adds `RigidBodyAPI`
or `MassAPI`.

Float32 voxel centers from official VoMP output are snapped to the validated
declared pitch before shared corners are built; the small snap error is retained
in provenance.

The authoring contract is pinned in provenance to
[OpenUSD proposal #111](https://github.com/PixarAnimationStudios/OpenUSD-proposals/pull/111)
at commit `61d83b54b7efbe97ad2f480de885255cd1e593be`. Newton 1.4.0 implements
an earlier experimental subset based on proposal commit
`5d89c0ed46a26de92f4d3fefef3bfad6500c07ce`; this adapter authors only names
whose meaning is shared by both revisions and exercises them with Newton's real
USD importer. `newton-usd-schemas` 0.4.1 is recorded separately as Newton's
runtime schema plugin; it does not register the draft deformable APIs, which
are intentionally authored as raw applied-schema tokens. This is a draft
interchange contract, not a claim of compatibility with the separate
OmniPhysics/PhysX deformable schema.

Newton 1.4 consumes one effective elastic material for the generated volume.
Uniform Young's modulus and Poisson ratio are authored directly through USD
float encoding. Values outside Newton 1.4's unclamped Poisson range
`[-0.999, 0.499]` fail instead of changing silently, and the adapter verifies
that no stronger ancestor binding overrides its generated material. A spatially
heterogeneous elastic field fails by default. To intentionally publish a
conditional volume-weighted approximation, pass:

```bash
--material-reduction homogeneous-volume-average
```

The provenance records the original field statistics, Voigt/Reuss bounds,
reduction policy, mass/center of mass, generated topology digest, and explicit
limitations. `--max-deformable-voxels` defaults to 65,536 and fails rather than
silently coarsening or subsampling the field. The inference-side
`--max-complete-voxels` cap is bounded to that topology cap before the VoMP
runtime starts, so oversized fields fail before material inference instead of
after it.

## Unified Pipeline

`vomp_mass` is a concrete opt-in step after `apply_physics`. When both steps
run, its input is automatically wired to the simulation-ready USD, preserving
the collider and physics material authored by the normal pipeline.

```yaml
project:
  name: gear-vomp
input:
  usd_path: gear.usda
steps:
  build_dataset_usd:
    enabled: true
  build_dataset_prepare_dataset:
    enabled: true
  predict:
    enabled: true
  apply_physics:
    enabled: true
  vomp_mass:
    enabled: true
    target_prim: /World/Gear
    runtime_root: /opt/VoMP
    render:
      num_views: 150
      image_width: 512
      image_height: 512
      render_mode: rt2
      num_sensor_updates: 32
      material_target: auto
```

Run it with `physics-agent run config.yaml`. `vomp_mass` may also run by itself
when the input USD is already deinstanced and simulation-ready.
Relative `render.ovrtx_venv_dir` paths are resolved from the YAML file's
directory.

## Advanced NPZ Adapter

`apply-vomp` remains available for an already-produced, complete VoMP NPZ:

```bash
physics-agent apply-vomp \
  object.usda materials.npz object_vomp.usda \
  --target-prim /World/Object \
  --voxel-size-m 0.003125 \
  --coordinate-unit-meters 1.0 \
  --complete-voxel-field
```

It accepts VoMP `save_materials()` structured `voxel_data` with fields `x`,
`y`, `z`, `density`, `youngs_modulus`, `poissons_ratio`, and `segment_id`, or
direct arrays named `voxel_coords_world`, `density`, `youngs_modulus`, and
`poisson_ratio`. The structured field uses upstream's plural
`poissons_ratio`; the direct array uses singular `poisson_ratio`.
Query-point/mesh-vertex results are rejected because they have no integration
volume. The explicit completeness flag is required because raw upstream NPZ
files do not record whether `max_voxels` subsampled the field. Pass the exact
upstream voxel pitch to `--voxel-size-m`; inferred mass scales with its cube.

The runnable synthetic example is:

```bash
python apps/physics_agent/examples/vomp_rigid_body/apply_precomputed_vomp.py
```

The real external-library example is:

```bash
python apps/physics_agent/examples/vomp_rigid_body/run_vomp.py \
  object_physics.usda object_vomp.usda \
  --target-prim /World/Object \
  --vomp-root /opt/VoMP
```

For an existing complete NPZ, the deformable sibling is:

```bash
physics-agent apply-vomp-deformable \
  object.usda materials.npz object_vomp_deformable.usda \
  --target-prim /World/Object \
  --voxel-size-m 0.003125 \
  --coordinate-unit-meters 1.0 \
  --complete-voxel-field
```

## Limits

- Inputs and outputs are `.usd`, `.usda`, or `.usdc`, not USDZ packages.
- The input stage must explicitly author both `metersPerUnit` and
  `kilogramsPerUnit`.
- The target must be static, visible, deinstanced, watertight, and backed by
  polygon meshes with `subdivisionScheme = "none"`. Visible cubes, spheres,
  point instancers, curves, and other non-mesh geometry are rejected. Tessellate
  evaluated surfaces before running VoMP.
- The target world transform must be rigid; scale, shear, and reflection are
  rejected for mass authoring.
- Volume-deformable v1 requires `metersPerUnit = 1` and
  `kilogramsPerUnit = 1`, because Newton 1.4 does not convert non-SI stage
  units. Its target must be a deinstanced `UsdGeom.Xform` with no rigid-body,
  mass, disabled/kinematic/asleep body state, simulation-owner routing,
  existing collision, or independent deformable contract.
- Generated tetrahedra are simulation geometry with `purpose = guide`. The
  adapter does not author visual embedding/skinning, attachments, constraints,
  damping, or runtime trajectory validation; the original visual geometry will
  not automatically follow the simulated particles.
- Point3f encoding is checked for per-tetrahedron volume and center-of-mass
  distortion. Large local coordinates fail with guidance to rebase the target.
- PhysX uses an OmniPhysics-specific deformable schema. This v1 authorer emits
  only the Newton/AOUSD draft profile; a PhysX emitter requires a separate
  explicit mapping and validation harness.
- The output is a flattened USD snapshot. Composition arcs and instancing are
  collapsed. Resolvable local dependencies are copied into
  `<output-filename>_assets/` and rewritten relative to the output; move that
  exporter-owned sidecar together with the root layer to keep the result
  relocatable. Do not rely on the output retaining the input's reference
  structure.
- `run-vomp` replaces `<work-dir>/evidence` on each run. Keep unrelated files
  outside that subtree.
- The integration authors mass properties only. It does not infer collision
  geometry, friction, restitution, damping, joints, or trajectories.
- Run normal `apply_physics` first when the asset still needs colliders or
  physics materials.
- Blank or near-blank OVRTX frames fail the run before VoMP inference.
