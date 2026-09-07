# Falsification and Locking

Use this as the canonical post-initialization review and locking procedure.

Treat every initializer output as a provisional hypothesis. Image masks,
direct picks, and deterministic scripts may propose `rev-000`; none may approve
it.

## 1. Write the falsification plan

Before reviewing `rev-000`, write `PART/falsification_plan.json`:

```json
{
  "schema_version": "mesh-segmentation-falsification-plan.v1",
  "semantic_part": "target_part",
  "segment_id": 1,
  "expected_instances": ["instance-a", "instance-b"],
  "confusers": [
    {
      "name": "neighbor_part",
      "visual_test": "highlight must stop before the neighboring surface"
    }
  ],
  "falsifiers": [
    {
      "id": "neighbor-leak",
      "claim": "any highlighted neighbor surface rejects the candidate"
    },
    {
      "id": "missing-instance",
      "claim": "any expected instance without complete visible coverage rejects the candidate"
    }
  ],
  "construction_view_ids": ["seed-view-a", "seed-view-b"],
  "held_out_view_ids": ["opposing-review", "focused-boundary-review"]
}
```

Use target semantics and nearby geometry to state observable failures. Do not
encode asset coordinates, remembered IDs, or category-specific geometry
predicates.

## 2. Collect signed falsification evidence

Render focused neutral, flat-label, face-ID, and fragment-ID evidence. Ensure
that the target and its closest confusers occupy enough pixels to inspect.
Reserve at least one useful view that was not used to construct or last edit
the candidate.

Also export the exact candidate partition, create a selected-only stage with
`build_selected_only_stage.py`, and render only the active segment from every
held-out view. This is not the earlier candidate-ring visualization: it removes
occluding residual geometry so embedded caps, internal plates, floaters, and
other selected non-target surfaces cannot hide in a gray scene.

Use selected-only evidence symmetrically. Search for extra selected surfaces,
but also ask whether every isolated instance forms a coherent, semantically
complete object. Reject truncated shells, discontinuous outer contours,
implausible openings, and unexplained jagged cuts. Connected-component count
proves presence only; it never proves completeness.

For repeated instances, compare their selected-only silhouettes, shells,
openings, and visible surface structure against one another. Geometry present
or missing in only one side or subset of otherwise repeated instances is a
falsification target, not evidence of correctness. Inspect it in the matching
flat-label and ID views.

Treat every visible selected-only boundary as a hypothesis. Classify it as:

- an intended semantic boundary;
- an object silhouette or occlusion;
- an unexplained boundary caused by missing faces.

The third classification rejects the candidate. For every expected instance,
inspect at least two selected-only views, including one opposing or
occlusion-revealing view. In each view, inspect at least one actual adjacent
unselected fragment from the deterministic frontier audit. If any checked
fragment belongs to the target, include its whole frozen fragment, create a new
revision, and repeat the review.

When the candidate is a union of complete disconnected source components, the
deterministic frontier audit reports zero adjacent unselected fragments and
there is nothing to challenge. Do not fabricate a frontier and do not report the
part as blocked. Record
`"adjacent_unselected_frontier_fragment_ids_checked": []` and
`"frontier_decision": "no_adjacent_unselected_frontier"` in each completeness
view. The validator accepts this only when the digest-bound audit confirms an
empty frontier; every other completeness requirement still applies.

Batch-click:

- every expected instance from at least two useful views;
- three positive challenges per instance, covering `target_interior`,
  `target_extent`, and `opposing_or_occluded_surface`;
- every named visible confuser as negative evidence;
- a close positive/negative pair straddling at least one visible semantic
  boundary of every expected instance.

Do not satisfy this gate with easy target-center and confuser-center clicks.
Those only prove that an obvious target face is selected and an obvious
confuser face is not. The purpose is to challenge the current candidate at
places where it could plausibly be wrong.

Add these fields to the click events:

- `coverage_role` on positive events, using one of the three roles above;
- `confuser_id` on negative events, exactly matching a named confuser in the
  falsification plan;
- `boundary_pair_id` on one positive and one negative event placed immediately
  across the same visible boundary, in the same view and for the same expected
  instance.

The paired pixels must be within 4% of the larger image dimension. Use focused
views so this is a meaningful local test rather than two distant easy samples.
Every held-out view must contribute both an accepted positive and an accepted
negative click. If a proposed held-out view cannot show both, replace it before
review.

Resolve the batch with `pick_face_evidence.py`. Accepted evidence must contain
both polarities. A click rejected for a miss or fragment boundary is not
evidence; move it and resolve again.

Run `compare_face_labels.py` against the immutable state from before the active
part:

Every accepted lock is face-immutable in every run mode. If later evidence
suggests that a locked part owns a false-positive fragment, preserve the lock,
record the conflict or unresolved boundary, and continue without changing its
faces. Lock correction is unsupported until a launcher-owned transactional
validator can bind every affected revision to the final state.

```bash
python "$MESH_TOOL_DIR/compare_face_labels.py" \
  --source-usd INPUT.usd \
  --target /World/FusedMesh \
  --fragment-labels RUN/fragments/fragment_ids.u32le \
  --parent-labels RUN/state/PARENT-BEFORE-PART.u32le \
  --candidate-labels RUN/PART/rev-NNN/face_labels.u32le \
  --evidence RUN/PART/falsification_evidence.json \
  --active-segment-id PART_ID \
  --allow-noop \
  --output RUN/PART/rev-NNN/falsification_label_comparison.json
```

The candidate fails if it omits a positive face or includes a negative face.
Do not override this result with visual confidence. If one frozen fragment
contains both valid positive and negative evidence, the representation is too
coarse: regenerate finer fragments before any part is locked, or report the
limitation after locking has begun.

## 3. Diagnose before editing

For each failed falsifier, classify the problem:

- coherent false positive;
- coherent false negative;
- missed repeated instance;
- boundary-local error;
- occlusion or insufficient view;
- fragment-boundary limitation.

Write the diagnosis before changing labels. Change the camera or evidence when
the failure is observational. Batch include or exclude fragments only when the
whole fragment decision is supported. Preserve every rejected candidate.

Do not require an edit merely to increment the revision. A clean `rev-000` may
be accepted, but only after the same falsification procedure.

## 4. Build and review the mandatory revision-consistency evidence

Do this for the exact final candidate after the last edit and before writing
`falsification_review.json`. The consistency gate is mandatory even for a
single expected instance. It makes small omissions and protrusions comparable
at a stable scale instead of relying on a downscaled full-scene contact sheet.

Use the final revision's neutral, flat-label, and selected-only render
manifests. They must expose the same review view IDs. Under the final revision,
write `revision_consistency_regions.json` with this schema:

```json
{
  "schema_version": "mesh-segmentation-consistency-regions.v1",
  "instances": [
    {
      "instance_id": "instance-a",
      "views": [
        {
          "view_id": "primary-review",
          "surface_role": "primary_surface",
          "bbox": [120, 84, 356, 332]
        },
        {
          "view_id": "opposing-review",
          "surface_role": "opposing_or_occlusion_revealing",
          "bbox": [91, 102, 344, 361]
        }
      ]
    }
  ]
}
```

List every expected instance exactly once and in the same order as
`falsification_plan.json`. Each instance needs at least two useful crops and
must include both `primary_surface` and
`opposing_or_occlusion_revealing`. Every `view_id` must exist in all three
render manifests. Bounding boxes are integer `[x0, y0, x1, y1]` coordinates in
the corresponding source images.

Build the standardized crop sheet and the two-view full-context triptych
pages:

```bash
python "$MESH_TOOL_DIR/build_revision_consistency_sheet.py" \
  --neutral-manifest RUN/PART/rev-NNN/renders/neutral/render_manifest.json \
  --flat-label-manifest RUN/PART/rev-NNN/renders/render_manifest.json \
  --selected-only-manifest RUN/PART/rev-NNN/selected-only/renders/render_manifest.json \
  --regions RUN/PART/rev-NNN/revision_consistency_regions.json \
  --output-dir RUN/PART/rev-NNN/revision-consistency \
  --context-views-per-page 2
```

This writes a `mesh-segmentation-consistency-sheet.v1` manifest at
`revision-consistency/manifest.json`, its `comparison_sheet.png`, standardized
flat-label/selected-only crops, and `context_sheet_*.png` pages. Inspect every
instance crop pair and every full-context page. Compare repeated instances
against every peer. For a single instance, compare its primary and opposing
views. Any unique gap, protrusion, unexplained boundary, or context outlier
rejects the candidate and requires another edit, revision, rerender, and sheet.

After a clean inspection, write
`RUN/PART/rev-NNN/revision_consistency_review.json`:

```json
{
  "schema_version": "mesh-segmentation-revision-consistency-review.v1",
  "status": "passed",
  "semantic_part": "target_part",
  "segment_id": 1,
  "revision": "rev-NNN",
  "candidate_labels": "face_labels.u32le",
  "candidate_labels_sha256": "SHA256_OF_CANDIDATE_LABELS",
  "neutral_render_manifest": "renders/neutral/render_manifest.json",
  "neutral_render_manifest_sha256": "SHA256_OF_NEUTRAL_MANIFEST",
  "flat_label_render_manifest": "renders/render_manifest.json",
  "flat_label_render_manifest_sha256": "SHA256_OF_FLAT_LABEL_MANIFEST",
  "selected_only_render_manifest": "selected-only/renders/render_manifest.json",
  "selected_only_render_manifest_sha256": "SHA256_OF_SELECTED_ONLY_MANIFEST",
  "comparison_sheet": "revision-consistency/comparison_sheet.png",
  "comparison_sheet_sha256": "SHA256_OF_COMPARISON_SHEET",
  "consistency_sheet_manifest": "revision-consistency/manifest.json",
  "comparison_channels": ["flat_label", "selected_only"],
  "context_comparison_channels": ["neutral", "flat_label", "selected_only"],
  "context_comparison_sheets": [
    {
      "view_ids": ["primary-review", "opposing-review"],
      "image": "revision-consistency/context_sheet_00.png",
      "image_sha256": "SHA256_OF_CONTEXT_SHEET"
    }
  ],
  "context_outlier_regions": [],
  "comparison_mode": "cross_view_outlier",
  "expected_instances": ["instance-a"],
  "views_compared": ["primary-review", "opposing-review"],
  "outlier_regions": [],
  "instance_comparisons": [
    {
      "instance_id": "instance-a",
      "status": "consistent",
      "peer_instance_ids": [],
      "unique_gap_regions": [],
      "unique_protrusion_regions": [],
      "unique_boundary_regions": [],
      "assessment": "primary and opposing evidence show one complete target shell"
    }
  ]
}
```

Replace every `SHA256_OF_*` placeholder with the lowercase SHA-256 digest of
the exact referenced file. Artifact paths are resolved relative to the JSON
file that names them; absolute paths are also accepted when they remain inside
`RUN`. Do not use symlinks. Copy `context_comparison_sheets` from the generated
sheet manifest without reordering, including every context page and digest.
`views_compared` must be the concatenation of those page `view_ids`.

For repeated instances, use
`comparison_mode: "repeated_instance_outlier"`. Preserve expected-instance
order in `instance_comparisons`, and set each `peer_instance_ids` to every
other expected instance in plan order. A passing review requires empty
`context_outlier_regions`, `outlier_regions`, `unique_gap_regions`,
`unique_protrusion_regions`, and `unique_boundary_regions`. Do not write
`status: "passed"` while any such issue remains.

The consistency review, candidate labels, comparison sheet, generated sheet
manifest, and generated crops must all remain under the final revision except
for the referenced run-contained render manifests. The validator binds their
paths and digests to the same candidate and render evidence used by the
falsification review.

## 5. Record the completion review

After the last edit, rerender the exact latest revision. Write
`PART/falsification_review.json`:

```json
{
  "schema_version": "mesh-segmentation-falsification-review.v1",
  "semantic_part": "target_part",
  "segment_id": 1,
  "revision": "rev-001",
  "status": "accepted",
  "falsification_plan": "falsification_plan.json",
  "candidate_labels": "rev-001/face_labels.u32le",
  "signed_evidence": "falsification_evidence.json",
  "label_comparison": "rev-001/falsification_label_comparison.json",
  "frontier_audit": "review/frontier/frontier_audit.json",
  "latest_render_manifest": "rev-001/renders/render_manifest.json",
  "id_buffer_manifests": ["rev-001/id-buffers/manifest.json"],
  "selected_only_export_manifest": "rev-001/selected-only/export_manifest.json",
  "selected_only_stage_manifest": "rev-001/selected-only/stage_manifest.json",
  "selected_only_render_manifest": "rev-001/selected-only/renders/render_manifest.json",
  "revision_consistency_review": "rev-001/revision_consistency_review.json",
  "selected_only_review": "passed",
  "repeated_instance_consistency_checked": ["instance-a", "instance-b"],
  "instance_completeness_reviews": [
    {
      "instance_id": "instance-a",
      "status": "passed",
      "views": [
        {
          "view_id": "primary-review",
          "surface_role": "primary_surface",
          "selected_only_render": "rev-001/selected-only/renders/primary-review.png",
          "flat_label_render": "rev-001/renders/primary-review.png",
          "coherent_semantic_shell": true,
          "outer_contour_continuous": true,
          "semantic_openings_plausible": true,
          "unexplained_boundaries": [],
          "boundary_classifications": [
            {
              "region_id": "outer-silhouette",
              "classification": "silhouette_or_occlusion",
              "rationale": "the boundary is the visible exterior silhouette"
            }
          ],
          "adjacent_unselected_frontier_fragment_ids_checked": [42],
          "frontier_decision": "confirmed_non_target",
          "assessment": "the isolated shell is complete in this view"
        },
        {
          "view_id": "opposing-review",
          "surface_role": "opposing_or_occluded_surface",
          "selected_only_render": "rev-001/selected-only/renders/opposing-review.png",
          "flat_label_render": "rev-001/renders/opposing-review.png",
          "coherent_semantic_shell": true,
          "outer_contour_continuous": true,
          "semantic_openings_plausible": true,
          "unexplained_boundaries": [],
          "boundary_classifications": [
            {
              "region_id": "semantic-opening",
              "classification": "intended_semantic_boundary",
              "rationale": "the opening matches the target part semantics"
            }
          ],
          "adjacent_unselected_frontier_fragment_ids_checked": [57],
          "frontier_decision": "confirmed_non_target",
          "assessment": "the opposing shell has no truncated or jagged cut"
        }
      ]
    }
  ],
  "construction_view_ids": ["seed-view-a", "seed-view-b"],
  "held_out_view_ids": ["opposing-review"],
  "expected_instances_checked": ["instance-a", "instance-b"],
  "confusers_checked": ["neighbor_part"],
  "falsifiers": [
    {
      "id": "neighbor-leak",
      "status": "passed",
      "evidence": ["rev-001/renders/opposing-review.png"],
      "rationale": "the highlighted boundary stops before the neighbor"
    }
  ],
  "false_positive_review": "passed",
  "false_negative_review": "passed",
  "actionable_issues": [],
  "unresolved_uncertainties": []
}
```

When the deterministic frontier audit contains no unselected fragments because
the selected target is a complete disconnected component, use `[]` for
`adjacent_unselected_frontier_fragment_ids_checked` and
`"not_applicable_complete_disconnected_component"` for `frontier_decision` in
each completeness view. Do not fabricate a neighboring fragment. A nonempty
frontier still requires at least one checked fragment per view and
`"confirmed_non_target"`.

Run:

```bash
python "$MESH_TOOL_DIR/validate_falsification_review.py" \
  --run-dir RUN \
  --review RUN/PART/falsification_review.json \
  --output RUN/PART/falsification_validation.json
```

This command is the only supported way to create
`PART/falsification_validation.json`. Never hand-write that file. Do not edit
the review, consistency review, candidate labels, or any bound evidence after
the command passes; any such change makes the validation digest stale and
requires deleting and regenerating the validation.

Lock only when validation passes. `part_completion.json` must name the same
final revision and record both falsification artifact paths and the validation
digest. A prose claim such as `false_positive_review: passed` is not sufficient
without the signed-evidence, per-instance completeness, frontier, and
candidate-digest checks.

When memory is enabled, record the validated-lock observation immediately
after writing `part_completion.json` and before promoting terminal labels.
Every requested target needs this timely agent-authored record before
`state/final_labels.u32le` is first created.
