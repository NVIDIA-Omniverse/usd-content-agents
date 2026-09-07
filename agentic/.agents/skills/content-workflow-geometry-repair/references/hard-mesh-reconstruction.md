# Hard-Mesh Reconstruction

## Route Order

1. Preserve and diagnose the immutable source.
2. Apply conservative duplicate, degenerate, or winding cleanup.
3. Close only triangular or planar quadrilateral manifold loops with
   `trimesh_bounded_hole_fill` when reconstruction is explicitly allowed.
4. Apply approved Geogram local repair for diagnosed local topology defects.
5. Use the policy-enabled reconstructive SDF route only when local routes
   cannot produce a valid candidate and `allow_reconstructive=true`.
   `sdf_rebuild` is the canonical worker. Legacy serialized requests using
   `openvdb_rebuild` normalize to that worker before policy admission.
6. Reject when every bounded candidate fails topology, fidelity,
   correspondence, protected-feature, or profile gates.

Do not select route strength from visual intuition. The checked-in route policy
orders immutable candidates using observed benchmark results.

## Bounded Local Closure

The local hole worker accepts only closed manifold loops with three vertices or
planar loops with four vertices. It caps loop count, perimeter relative to bbox,
and total added area. It preserves all existing vertices/faces and refuses
instances. Attributed meshes are accepted only when uniform values, indexed
face-varying values, and subset membership agree across every contributing
boundary face; generated normals and persisted source-corner evidence are
explicit. Any boundary conflict refuses the patch.

Treat its output as reconstructive: source-relative fidelity may pass, but the
new surface still needs ground-truth or visual review.

## SDF Production Boundary

- Call semantic operations through `sdf_tools`, bind one explicit backend for
  each qualified route, and retain `SdfSession.backend_info` as execution
  evidence. Do not import a native-library facade from workflow code, accept a
  caller-supplied module path, install a plugin, or fall back to another backend
  implicitly.
- The current admitted production driver is `openvdb`. Require the
  source-locked OpenVDB 13.0.0 wheel from commit
  `7c03e1f084873cd1b3422c7ff7aec6ee681b3b38`, distribution
  `openvdb==13.0.0+wu.3`, ABI 13, and matching source-lock and loaded-extension
  identity evidence. The admitted driver calls OpenVDB's nanobind Python module
  and the repository's bounded tools overlay in-process. In Geometry Repair
  that call runs inside the existing generic isolated worker; do not introduce
  an SDF- or OpenVDB-specific helper, daemon, transport, or subprocess.
- Require the structured backend license manifest and verified native closure.
  The SDF delta must not introduce or bundle an LGPL component; a top-level
  OpenVDB license declaration is not sufficient admission evidence.
- Use signed reconstruction for automatic routing. Unsigned offset mode can
  create a second surface and is evaluation-only unless explicitly planned.
- Reconstruct each semantic USD mesh independently and retain its prim path.
- Refuse instance proxies, face material subsets, UVs, and face-varying data
  until an exact remapper exists.
- Derive voxel size from the bounded grid and reject grids that undersample a
  required feature.
- Try the checked-in high-to-low resolution ladder as separate candidates.
  Each attempt starts from the accepted pre-reconstruction checkpoint and is
  compared with the immutable normalized source.
- Apply only the checked-in single level-set mean-filter step. Treat stronger
  smoothing as a separate reconstructive experiment; never increase it to make
  a visual review pass without re-running every fidelity and feature gate.
- Reject coarse native outputs whose area indicates a thin double shell or
  vanished surface before expensive exact validation.
- Never convert an SDF operation return into acceptance. Require re-diagnosis,
  bidirectional surface coverage, part correspondence, six-view silhouettes,
  protected probes, and the reconstructive drift band.

An SDF-reconstructed candidate that passes every measured gate remains `conditional`
until a trusted ground truth or a recorded human visual/task review accepts the
generated surface.

## Exact And Research Evaluators

- CGAL Polygon Mesh Processing is GPL in the evaluated distro package. Build
  the exact-intersection helper only with the explicit local evaluation CMake
  flag. Its output is shadow evidence and cannot certify, route, mutate, or ship
  in production images.
- CelloCut was shadow-evaluated and was not promoted. Its upstream GPU
  decimator crashed at both tested resolutions; the evaluation-only CPU
  substitution still needed a separate simplification pass and produced only
  conditional known-ground-truth fidelity. Its linked CGAL boundary is also
  outside the approved production distribution. Do not route production work
  to it or use its non-commercial benchmark data for NVIDIA product evaluation.
- PaMO remains disabled until an explicit AGPL service and distribution design
  is approved. Do not execute or ship it merely because its source is public.

Use `phase2_candidate_decisions.json` and `worker_registry.json` as the policy
authority. Documentation or an agent suggestion cannot enable a worker.

## Review Checklist

- Compare source, known ground truth when available, and repaired OVRTX six-view
  images.
- Check for invented closures, fused parts, lost openings, floaters, scale
  drift, faceting, and semantic-part damage.
- Confirm every required opening/cavity/clearance probe and per-part
  correspondence record.
- Separate geometry acceptance from collision/runtime and complete profile
  outcome. A collision failure does not erase a valid geometry measurement,
  and valid geometry does not imply SimReady completion.
