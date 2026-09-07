# Iterative VQA Refinement

Use iterative refinement when final renders reveal material defects that can be
fixed through usd-cli's low-level material operations.

## Loop

1. Read current assignments and VQA artifacts.
2. Inspect final render records and affected images.
3. Compare the reference and result as images before consulting the intended
   family labels. Check the largest visible surfaces for dominant brightness,
   saturation, metalness, transparency, and finish differences. A family-name
   match does not excuse a visible palette or material-class mismatch; for
   example, dark navy versus bright cyan or galvanized gray versus glossy
   white is a fixable defect.
4. Identify fixable material mismatches.
   If dark geometry merges into the background, compare one temporary
   usd-cli OVRTX dome-fill render before deciding that the material itself is
   wrong. Keep the relit view diagnostic-only and restore the default rig for
   canonical evidence.
5. Use usd-cli to inspect affected prims or objects and
   isolate the affected geometry.
6. Apply a checkpointed, targeted material correction; undo or discard a bad
   correction without mutating the source.
7. Re-render affected views through OVRTX, optionally retaining the prior
   render for comparison.
8. Update decision patch, assignments, and VQA artifacts.
9. Stop when VQA passes or remaining issues are explicitly unfixable.

## Fixable Issues

- missing visible family assignment;
- high-contrast color mismatch;
- obvious material class mismatch;
- over-broad assignment;
- missed labels, accents, lenses, inserts, or hardware;
- stale selection/isolation artifact in final evidence.

## Non-Fixable Issues

Record an unresolved issue when:

- usd-cli cannot isolate the target geometry;
- source/inspection mapping remains ambiguous after backend resolution;
- no available material proxy improves the render;
- the reference is contradictory or insufficient;
- the task requires geometry, articulation, or physics changes outside material
  assignment.
