# Per-Part Initializer Routing

Use this as the canonical initializer-selection procedure.

Use this procedure once for every unlocked semantic part. The route decision is
semantic and visual; the artifact validation is deterministic.

## 0. Resolve Image Generation

Read `request.json -> runtime.image_generation` and use the `image-generation`
skill for every generated overlay.

- `world_understanding_backend`: invoke the shared skill's explicit-backend
  adapter with the recorded backend, model, optional base URL, and API-key
  environment-variable name. `generate_semantic_overlays.py` is a compatibility
  wrapper around that adapter for the multi-view mesh probe.
- `coding_agent_companion`: invoke the image-generation or image-editing
  capability accompanying the coding-agent session. Treat the recorded model
  as a preference only when that capability supports model selection.

Give both modes the same semantic-edit prompt and conditioning render. Preserve
every raw output and its `agentic-image-generation-result.v1` manifest before
registration. The companion path is not permission to call an unrecorded
network endpoint or invent a provider.

If neither path can produce conditioned semantic overlays, skip the image probe
for that part. Write `PART/image_generation_warning.json`:

```json
{
  "schema_version": "mesh-segmentation-image-generation-warning.v1",
  "severity": "warning",
  "code": "image_generation_unavailable",
  "semantic_part": "vision_glass",
  "reason": "No companion image-generation or image-editing tool is available.",
  "fallback": "direct_agentic_selection"
}
```

Then write an initializer decision with:

- `probe_status: "skipped_image_generation_unavailable"`;
- `decision: "direct_agentic_selection"`;
- an empty `probe_view_ids` list;
- every standard assessment field set to `"unavailable"`;
- `evidence.image_generation_warning` pointing to the warning artifact.

Validate the decision normally. Do not fabricate masks or block the workflow.
Include the warning in the final report.

## 1. Create a Three-View Mask Probe

Render these three representative cube-corner views of the neutral source:

- `+x-y+z`
- `-x+y+z`
- `+x+y-z`

Store them under `PART/initializer-probe/neutral`. Generate semantic overlays
with the configured image-editing backend, preserve the exact prompts, and
affine-register every overlay. Retry only failed registrations and preserve all
attempts.

Render closest-visible face and fragment ID buffers for the same three saved
cameras. Run `build_deterministic_fragment_union.py` with
`--expected-view-count 3` and `--erosion-pixels 2`, writing to
`PART/initializer-probe/projection`. This is a diagnostic projection only. Do
not apply its edit or call it `rev-000`.

Inspect all three forms of evidence together:

1. the registered semantic masks over the neutral renders;
2. the cyan surviving pixels after 2 px erosion;
3. the magenta whole-fragment projections produced by those pixels.

Assess:

- **semantic quality**: whether the image model selected the requested part
  rather than visually similar neighbors;
- **mask scale**: whether each important visible instance retains a meaningful
  interior after erosion;
- **cross-view consistency**: whether the same visible logical surfaces receive
  compatible interpretations;
- **mesh compatibility**: whether the selected fragments stay within semantic
  boundaries instead of amplifying a few mask pixels into large leaks;
- **confuser leakage**: whether likely neighboring parts are selected.

A visually good mask can still be a poor mesh initializer. Always inspect the
fragment projection before choosing the mask route.

## 2. Record and Validate the Decision

Write `PART/initializer_decision.json`:

```json
{
  "schema_version": "mesh-segmentation-initializer-decision.v1",
  "semantic_part": "vision_glass",
  "segment_id": 3,
  "status": "accepted",
  "probe_status": "completed",
  "decision": "direct_agentic_selection",
  "probe_view_ids": [
    "plus_xminus_yplus_z",
    "minus_xplus_yplus_z",
    "plus_xplus_yminus_z"
  ],
  "evidence": {
    "registration_manifests": ["initializer-probe/registered/manifest.json"],
    "projection_manifest": "initializer-probe/projection/manifest.json",
    "registered_masks": [
      "initializer-probe/registered/plus_xminus_yplus_z_aligned_mask.png",
      "initializer-probe/registered/minus_xplus_yplus_z_aligned_mask.png",
      "initializer-probe/registered/plus_xplus_yminus_z_aligned_mask.png"
    ],
    "chosen_pixel_overlays": [
      "initializer-probe/projection/views/plus_xminus_yplus_z/chosen_pixels.png",
      "initializer-probe/projection/views/minus_xplus_yplus_z/chosen_pixels.png",
      "initializer-probe/projection/views/plus_xplus_yminus_z/chosen_pixels.png"
    ],
    "fragment_projections": [
      "initializer-probe/projection/views/plus_xminus_yplus_z/selected_fragments.png",
      "initializer-probe/projection/views/minus_xplus_yplus_z/selected_fragments.png",
      "initializer-probe/projection/views/plus_xplus_yminus_z/selected_fragments.png"
    ]
  },
  "assessment": {
    "semantic_quality": "mixed",
    "mask_scale": "marginal",
    "eroded_interior_support": "marginal",
    "cross_view_consistency": "mixed",
    "mesh_projection_quality": "leaky",
    "confuser_leakage": "major",
    "rationale": "The masks identify some panes, but tiny errors select large non-glass fragments."
  },
  "next_step": "direct_signed_fragment_selection"
}
```

Allowed decisions:

- `image_mask_seed`
- `direct_agentic_selection`

Validate it:

```bash
python "$MESH_TOOL_DIR/validate_initializer_decision.py" \
  --run-dir RUN \
  --decision RUN/PART/initializer_decision.json \
  --output RUN/PART/initializer_decision_validation.json
```

Do not continue until validation passes.

Choose `image_mask_seed` only when the masks are semantically useful, survive
erosion, agree across the probe views, and project to acceptably bounded
fragments. Choose `direct_agentic_selection` whenever masks are too small,
hallucinated, inconsistent, or geometrically amplified. Part size or category
alone never determines the route.

## 3A. Image-Mask Seed Route

Render the other five fixed cube-corner views and generate/register their
overlays. Reuse the three accepted probe registrations; do not regenerate them.
Render ID buffers for all eight saved cameras in one call.

Run `build_deterministic_fragment_union.py` with all accepted registration
manifests, the eight-view ID-buffer manifest, `--expected-view-count 8`, and
`--erosion-pixels 2`.

The script:

1. erodes each mask by exactly 2 px;
2. selects closest-visible fragments touched by surviving positive pixels;
3. forms each per-view fragment set independently;
4. takes their exact set union;
5. removes only already locked fragments.

Do not use negative mask pixels, support ratios, vetoes, or cross-view voting.
Apply the generated edit unchanged as `rev-000`, then verify byte identity
against `expected_face_labels.u32le`.

## 3B. Direct Agentic Selection Route

Discard the mask projection as label evidence. It may explain why the route was
chosen, but no fragment ID from it may be copied into `rev-000`.

Work like a careful DCC user:

1. enumerate the visible logical instances or surfaces belonging to the part;
2. add focused, opposing, grazing, or close cameras that make those instances
   and their confusers unambiguous;
3. batch-click unmistakable positive fragment interiors and nearby negative
   confusers using closest-hit picking;
4. inspect every returned whole fragment in matching fragment-ID and neutral
   views;
5. include only confirmed positive fragments and record the negative fragments
   as exclusions/guards;
6. apply the direct include batch to the immutable parent labels as `rev-000`.

Do not automatically expand clicks, fit an annulus, use remembered coordinates,
or convert the rejected mask proposal into labels. If a selected fragment
crosses a semantic boundary, report the representation problem or regenerate a
finer fragment map before any part is locked.

## 4. Common Refinement

Both routes enter the same canonical fragment review after `rev-000`. Search
false positives and false negatives separately, use matching neutral/label/ID
evidence, batch supported fragment edits, and lock the part only after a
post-edit completion review finds no actionable issue.
