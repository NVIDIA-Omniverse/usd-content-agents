# Pipeline Steps Reference

The material agent unified pipeline recognizes the step order from
`apps/material_agent/material_agent/config/schema.py` `STEP_ORDER`. Steps run in
that order when enabled in the config, unless `--skip` or `--only` narrows the
execution plan.

## Step Execution Order

```text
validate_input -> optimize_usd -> render_preview -> identify_asset -> generate_reference_image -> build_dataset_usd -> build_dataset_pdf_vectorstore -> build_dataset_prepare_dataset -> cluster_prims -> predict -> expand_cluster_predictions -> benchmark -> validate_predictions -> harmonize_predictions -> restore_usd -> apply -> evaluate -> refine -> validate_output -> render
```

### Mutually Exclusive Steps

- **predict** and **benchmark** cannot both run
- **apply** and **refine** cannot both run (`refine` includes its own apply loop)

## Step Details

### 1. validate_input (optional)

Runs USD validation before any material processing to establish a baseline of
existing issues. `validate_output` can reuse this baseline when checking for
regressions later.

**Key config options:**
```yaml
steps:
  validate_input:
    enabled: true
    on_failure: warn         # warn, block, or fix
    validation_config:
      categories: ["Basic", "Layer", "Layout", "Other", "Material"]
      stage_timeout: 180.0
```

**Output:** `{working_dir}/validation/input/` -- baseline validation report

### 2. optimize_usd

Flattens and deinstances USD files via the configured scene optimizer service. This ensures prims are individually addressable for rendering and material assignment.

**Default config:**
```yaml
steps:
  optimize_usd:
    enabled: false
```

**Output:** `{working_dir}/optimized/` -- optimized USD file

### 3. render_preview (optional)

Renders lightweight whole-scene preview images. These previews are used as conditioning input for the `generate_reference_image` step. This step is only needed when using AI-generated reference images instead of user-supplied photos.

**Key config options:**
```yaml
steps:
  render_preview:
    enabled: true
    cameras: ["+x+y+z"]             # Camera angles for preview
    should_reset_materials: true     # Strip materials for a clean preview
    use_lights: false                # Disable scene lights
    image_width: 512
    image_height: 512
    background_color: [1.0, 1.0, 1.0]
    camera_margin: 1.0
```

**Output:** `{working_dir}/preview/` -- preview images, auto-wired to `generate_reference_image`

### 4. identify_asset (optional)

Uses preview images and asset metadata to classify the overall object and
derive prompt context for downstream steps such as generated reference images.

**Requires:** `render_preview` for useful image-based identification. Without
preview images, the step records a conservative unknown-asset fallback.

**Key config options:**
```yaml
steps:
  identify_asset:
    enabled: true
    vlm:
      backend: nim
      model: google/gemma-4-31b-it
```

**Output:** `{working_dir}/identification/identification.json` plus an
`image_gen_prompt` value for downstream reference-image generation.

### 5. generate_reference_image (optional)

Generates photorealistic reference images from the scene previews and a user text prompt using an image generation model. The generated images are automatically injected into `build_dataset_prepare_dataset` as reference images for VLM context.

**Requires:** `render_preview` step to run first (provides preview images as conditioning input). If no explicit prompt is configured, run `identify_asset` first so it can provide an `image_gen_prompt`.

**Key config options:**
```yaml
steps:
  generate_reference_image:
    enabled: true
    image_gen:
      backend: gemini                                  # Public image-gen backend (Google)
      model: gemini-3-pro-image-preview                # Text → pixels; needs GOOGLE_API_KEY
    prompt: "Describe desired materials, e.g. aluminium frame with blue plastic tray"
    num_images: 1                    # Number of reference images to generate
    reference_images: []             # Optional existing images to condition on
```

**Output:** `{working_dir}/generated_refs/` -- generated reference images, auto-injected into dataset preparation

### 6. build_dataset_usd

Renders multi-view images of each prim in the USD scene. These images are used as VLM input for material prediction.

**Key config options:**
```yaml
steps:
  build_dataset_usd:
    renderer:
      backend: remote              # remote, ovrtx, or warp
      image_width: 512
      image_height: 512
      should_assign_random_colors: true
      rendering_modes:
        prim_only:                # Isolated prim view
          margin: 1.2
          cameras: ["+x+y+z", "-x-y-z"]
          camera_focus_mode: prim
        composition:              # Prim in context of full scene
          margin: 6.0
          cameras: ["+x", "+y", "+z"]
          camera_focus_mode: stage
    prim_filters:
      types: ["UsdGeom.Mesh", "UsdGeom.Cube", "UsdGeom.Cylinder",
              "UsdGeom.Capsule", "UsdGeom.Sphere", "UsdGeom.Cone"]
      skip_instances: true
    extract_material_bindings: true
    extract_hierarchy: true
    batch_size: 16
    # Conservative for local OVRTX sidecars; raise only when the render
    # endpoint fronts multiple independent rendering service instances.
    num_workers: 1
    max_concurrent_requests: 1
```

**Output:** `{working_dir}/dataset/usd/` -- rendered images and dataset manifest

### 7. build_dataset_pdf_vectorstore

Builds a searchable vector store from PDF documents (e.g., technical
specifications). Retrieved specification evidence is kept separate from visual
prediction and used afterward to corroborate the visual result or flag a
conflict for review.

**Key config options:**
```yaml
steps:
  build_dataset_pdf_vectorstore:
    enabled: false  # Disabled by default
```

Requires `input.reference_pdfs` to be set in the config. Source PDFs are processed into text/image chunks and indexed.

**Output:** `{working_dir}/vectorstore/` -- FAISS vector store

### 8. build_dataset_prepare_dataset

Prepares the final dataset for VLM inference by combining rendered images,
material lists, and visual prompts. Optional retrieved specification evidence
is stored separately for post-visual reconciliation rather than added to the
visual prompt or media.

**Key config options:**
```yaml
steps:
  build_dataset_prepare_dataset:
    enabled: true
    include_ground_truth: false      # true for benchmarking
    include_prim_path_context: true
    prompts:
      vlm_system: "System prompt with {materials_list} placeholder"
      vlm_user: "User prompt with {context} placeholder"
```

**Output:** `{working_dir}/dataset/dataset.jsonl` -- VLM-ready dataset

### 9. cluster_prims (optional)

Clusters visually similar prims before prediction so the VLM can score only
cluster representatives. Use this for large repeated assemblies where many
prims are near-duplicates.

**Key config options:**
```yaml
steps:
  cluster_prims:
    enabled: true
    embedding_service: nim
    embedding_model: nvidia/llama-nemotron-embed-vl-1b-v2
    min_prims_to_activate: 20
    max_cluster_size: 32
    report:
      enabled: true
```

**Output:** `{working_dir}/clusters/` -- clustering report and representative mapping

### 10. predict

Runs VLM inference on the prepared dataset to predict material assignments for each prim.

**Key config options:**
```yaml
steps:
  predict:
    vlm:
      backend: nim                   # nim (default), openai, anthropic, gemini
      model: google/gemma-4-31b-it
      temperature: 1.0
      max_completion_tokens: 24576
    max_workers: 16                 # Parallel inference workers
    llm:                            # Optional LLM for post-processing
      backend: nim
      model: google/gemma-4-31b-it
    report:
      image_max_size: 256
```

**Output:** `{working_dir}/predictions/predictions.jsonl` and `report.html`

After prediction, inspect the JSONL output and surface every record where
`materials.evidence_reconciliation.review_required` is `true`. For example:

```bash
WORKING_DIR=/path/to/working_dir
jq -c 'select(.materials.evidence_reconciliation.review_required == true) |
  {id, visual_material: .materials.evidence_reconciliation.visual_material,
   conflicting_spec_materials: .materials.evidence_reconciliation.conflicting_spec_materials}' \
  "$WORKING_DIR/predictions/predictions.jsonl"
```

Treat each emitted record as a human-review item. The specification claims do
not replace the fixed visual material label.

### 11. expand_cluster_predictions (optional)

Expands representative-only predictions from `cluster_prims` back onto all
cluster members so downstream steps see per-prim predictions again.

**Default config:**
```yaml
steps:
  expand_cluster_predictions:
    enabled: true
```

**Output:** Updated `{working_dir}/predictions/predictions.jsonl`

### 12. benchmark (alternative to predict)

Runs VLM prediction and LLM-judge evaluation in one pass. This step is mutually
exclusive with `predict`.

**Key config options:**
```yaml
steps:
  benchmark:
    enabled: true
    vlm: { backend: nim, model: google/gemma-4-31b-it }
    llm_judge: { backend: nim, model: meta/llama-4-maverick-17b-128e-instruct }
    max_workers: 64
```

**Output:** `{working_dir}/predictions/` -- predictions plus evaluation artifacts

### 13. validate_predictions (optional)

Validates predicted material names against the manifest, using fuzzy matching
and optional LLM repair to recover near-matches or malformed names.

**Typical config:**
```yaml
steps:
  validate_predictions:
    enabled: true
    llm:
      backend: nim
      model: google/gemma-4-31b-it
```

**Output:** Updated prediction records for downstream `harmonize_predictions`,
`apply`, `evaluate`, or `refine`.

### 14. harmonize_predictions (optional)

Resolves conflicting predictions across instanced or repeated parts. This step
helps keep visually identical hardware consistent across the asset.

**Typical config:**
```yaml
steps:
  harmonize_predictions:
    enabled: true
    llm:
      backend: nim
      model: google/gemma-4-31b-it
```

**Output:** Harmonized predictions for downstream material application.

### 15. restore_usd (optional)

Remaps predictions from optimized USD prim paths back to original USD prim
paths. Use this before `apply` or `refine` when optimization changed the prim
structure and materials must be written back onto the original asset.

**Output:** `{working_dir}/restored/restored_predictions.jsonl`

### 16. apply

Applies predicted materials to the USD file by creating material bindings for each prim.

**Key config options:**
```yaml
steps:
  apply:
    layer_only: false     # true to create a separate layer file
    flatten_output: false # true to flatten output USD
```

**Output:** `{output.usd_path}` or `{working_dir}/output/output.usd`

### 17. evaluate (optional)

Scores an existing predictions file against ground truth using an LLM judge.
Unlike `benchmark`, this evaluates already-generated predictions instead of
running fresh VLM inference.

**Key config options:**
```yaml
steps:
  evaluate:
    enabled: true
    llm_judge:
      backend: nim
      model: meta/llama-4-maverick-17b-128e-instruct
    success_threshold: 4.0
    generate_html_report: true
```

**Output:** `{working_dir}/evaluation/` -- judge scores and evaluation report

### 18. refine (alternative to apply)

Runs the iterative predict-apply-render-judge loop until the judge approves or
the iteration limit is reached. This is the schema step and CLI command name;
older `assign` docs refer to this same workflow.

**Key config options:**
```yaml
steps:
  refine:
    enabled: true
    max_iterations: 5
    vlm: {}
    llm_judge: {}
    apply:
      allow_empty_predictions: false
      fail_on_unknown_material: false
```

**Output:** `{working_dir}/iterations/` -- per-iteration predictions, renders, and judge decisions

### 19. validate_output (optional)

Validates the final materialized USD and compares it against the input
baseline. Use this to catch regressions introduced by `apply` or `refine`.

**Key config options:**
```yaml
steps:
  validate_output:
    enabled: true
    on_failure: warn         # warn or block
    validation_config:
      categories: ["Basic", "Layer", "Layout", "Other", "Material"]
      stage_timeout: 180.0
```

**Output:** `{working_dir}/validation/output/` -- post-run validation report

### 20. render

Renders the final USD file with applied materials to produce output images.

**Key config options:**
```yaml
steps:
  render:
    enabled: true
    backend: remote
    image_width: 1024
    image_height: 1024
    camera_corners: ["+x+y+z", "-x-y-z"]
    camera_margin: 1.0
    background_color: [1.0, 1.0, 1.0]
```

**Output:** `{working_dir}/renders/` -- rendered images

## Auto-Derived Directory Structure

All paths are automatically created under the working directory:

```text
.{session_id}/
  validation/
    input/              # validate_input output
    output/             # validate_output output
  optimized/          # optimize_usd output
  preview/            # render_preview output
  identification/     # identify_asset output
  generated_refs/     # generate_reference_image output
  dataset/
    usd/              # build_dataset_usd output
    dataset.jsonl     # prepare_dataset output
  vectorstore/        # pdf_vectorstore output
  clusters/           # cluster_prims output
  predictions/        # predict/benchmark output
  evaluation/         # evaluate output
  iterations/         # refine output
  restored/           # restore_usd output
  renders/            # render output
  output/
    output.usd        # final output USD
```
