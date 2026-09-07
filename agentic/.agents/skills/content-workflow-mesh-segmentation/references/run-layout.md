# Run Directory Layout

Write this exact structure. The launcher's gates and terminal validation locate
evidence by these paths, so an invented alternative makes correct work
unreadable and fails the run.

```text
part_plan.json
segments.json
prepare/  fragments/  state/
PART/                                     one directory per semantic part
  initializer_decision.json
  initializer_decision_validation.json
  falsification_plan.json
  falsification_evidence.json
  falsification_review.json
  falsification_validation.json
  part_completion.json
  rev-NNN/face_labels.u32le
  rev-NNN/edit_manifest.json
  rev-NNN/renders/                        diagnostic, flat, and neutral views
  rev-NNN/selected-only/segmented.usdc    export bound to this revision's labels
  rev-NNN/selected-only/part-only.usdc    isolated stage rendered for review
  rev-NNN/selected-only/*_manifest.json   export and isolated-stage provenance
  rev-NNN/selected-only/renders/VIEW.png  this part alone, one file per camera
  rev-NNN/selected-only/renders/render_manifest.json
final/                                    the request's required artifacts
```

`prepare/` holds the frozen inputs `prepare_mesh.py` writes, under exactly these
names: `neutral.usdc`, `topology.json`, `all_faces_candidate.u32le`, and
`degenerate_face_ids.u32le`. The names are fixed, not conventional — the
launcher lists them as required artifacts, and `validate_falsification_review.py`
reads `prepare/neutral.usdc` to recompute the unselected frontier for itself
rather than trusting the audit's own count. Renaming or removing any of them
fails the run; it does not fall back to a differently-named mesh.

The root `segments.json` is the manifest for the complete final partition. Its
configured IDs must exactly match the IDs present in `state/final_labels.u32le`;
prune superseded or emptied IDs before final export. Per-revision candidate
exports instead use the manifest beside their own selected-only evidence.

`PART` folds to the part's semantic name: lowercase, with each run of
non-alphanumeric characters collapsed to `_`, so `Car Chassis` becomes
`car_chassis`. An optional `NN_` ordering prefix is allowed, as is collecting
the directories under `parts/`. Record the exact semantic name in every
artifact's `semantic_part` field; only the directory spelling may differ.

`rev-NNN/selected-only/renders/` holds renders of the part *by itself* with every
other segment hidden. A whole-object render with the target tinted does not
belong there: anyone auditing that directory expects the part in isolation, and
a tinted whole-object image hides both missing and extra geometry. Its render
manifest must cite the `part-only.usdc` beside it. Keeping both under `rev-NNN`
lets a later monotonic extension preserve the earlier lock's evidence while the
launcher binds each reviewed render to the exact revision by path and digest.

Do not invent parallel per-part directories such as `candidate-vN`,
`inspect-*`, or `review-*` variants, and never write run files outside the run
directory, including scratch inputs such as edit or click batches.
