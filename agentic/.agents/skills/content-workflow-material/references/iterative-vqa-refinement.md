# Iterative VQA Refinement

Use iterative refinement when final renders reveal material defects that can be
fixed through available Workbench material edits.

## Loop

1. Read current assignments and VQA artifacts.
2. Inspect final render records and affected images.
3. Identify fixable material mismatches.
4. Use Workbench to inspect affected prims or objects.
5. Apply targeted material corrections.
6. Re-render affected views.
7. Update decision patch, assignments, and VQA artifacts.
8. Stop when VQA passes or remaining issues are explicitly unfixable.

## Fixable Issues

- missing visible family assignment;
- high-contrast color mismatch;
- obvious material class mismatch;
- over-broad assignment;
- missed labels, accents, lenses, inserts, or hardware;
- stale selection/isolation artifact in final evidence.

## Non-Fixable Issues

Record an unresolved issue when:

- Workbench cannot isolate the target geometry;
- source/inspection mapping is ambiguous;
- no available material proxy improves the render;
- the reference is contradictory or insufficient;
- the task requires geometry, articulation, or physics changes outside material
  assignment.
