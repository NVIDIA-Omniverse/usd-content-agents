# Unscored source-preserving drawer Geometry route

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
