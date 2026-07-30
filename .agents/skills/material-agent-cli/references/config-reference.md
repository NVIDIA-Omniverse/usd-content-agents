# Configuration Reference

Material Agent uses unified YAML configuration files. All paths are resolved relative to the config file's directory.

## Top-Level Structure

```yaml
project:        # Required -- Project metadata
input:          # Required -- Input files
output:         # Optional -- Output options
materials:      # Required for predict/apply -- Material definitions
steps:          # Optional -- Per-step configuration
advanced:       # Optional -- Advanced settings
```

## project (required)

```yaml
project:
  name: "my_pipeline"           # Required. Project identifier
  session_id: "my_session"      # Optional. Defaults to auto-generated ID
  working_dir: ".my_pipeline"   # Optional. Auto-derived as .{session_id}/
  description: "Description"    # Optional
```

## input (required)

```yaml
input:
  usd_path: "path/to/input.usd"           # Required. Input USD file
  reference_images:                         # Optional. Reference images for VLM context
    - path/to/reference1.jpg
    - path/to/reference2.jpg
  reference_pdfs:                           # Optional. PDFs for specification-evidence retrieval
    - path/to/specs.pdf
```

## output (optional)

```yaml
output:
  usd_path: "path/to/output.usd"  # Optional. Auto-derived if omitted
  layer_only: false                 # Create separate layer instead of full USD
  flatten_output: false             # Flatten output USD
```

## materials

Two modes are supported:

### External file (recommended)
```yaml
materials:
  path: "path/to/materials.yaml"
```

The external file format:
```yaml
library_path: "path/to/material_libs.usd"
entries:
  - name: "Aluminum Polished"
    description: "A polished aluminum for structural parts"
    binding: "/World/metal_library/Looks/Aluminum_Polished"
  - name: "Polycarbonate Blue"
    description: "A blue polycarbonate"
    binding: "/World/plastic_library/Looks/Polycarbonate_Blue"
```

### Inline
```yaml
materials:
  library_path: "path/to/material_libs.usd"
  entries:
    - name: "Aluminum Polished"
      description: "A polished aluminum for structural parts"
      binding: "/World/metal_library/Looks/Aluminum_Polished"
```

## steps

Each step can be enabled/disabled and configured. Steps are implicitly enabled if they have any config beyond `enabled`. See references/pipeline-steps.md for step-specific options.

**Important:** Step configs must NOT contain path keys like `usd_path`, `output_dir`, `dataset`, or `predictions_path`. All paths are auto-derived.

```yaml
steps:
  optimize_usd:
    enabled: true

  # Optional: generate reference images from text instead of supplying photos
  render_preview:
    enabled: false               # Enable for AI-generated references
    cameras: ["+x+y+z"]
    should_reset_materials: true
    image_width: 512
    image_height: 512
    background_color: [1.0, 1.0, 1.0]

  generate_reference_image:
    enabled: false               # Enable for AI-generated references
    image_gen:
      backend: gemini                                # Public Google image-gen backend
      model: gemini-3-pro-image-preview              # Needs GOOGLE_API_KEY
    prompt: ""                   # Required when enabled: describe desired materials
    num_images: 1
    reference_images: []         # Optional existing images to condition on

  build_dataset_usd:
    renderer:
      backend: remote
      image_width: 512
      image_height: 512

  build_dataset_pdf_vectorstore:
    enabled: false

  build_dataset_prepare_dataset:
    enabled: true

  predict:
    vlm:
      backend: nim
      model: google/gemma-4-31b-it

  apply: {}

  render:
    enabled: true
    image_width: 1024
    image_height: 1024
```

## advanced (optional)

```yaml
advanced:
  keep_temp_files: true    # Keep intermediate files after pipeline completion
  log_level: INFO          # Logging level
```

## Path Resolution Rules

1. All relative paths are resolved relative to the config file's parent directory
2. Working directory is auto-derived as `.{session_id}/` next to the config
   file, unless `project.working_dir` overrides it
3. Output USD path is auto-derived from `input.usd_path` if not specified
4. Each step's input/output paths are auto-wired based on the working directory structure
5. External materials file path is resolved relative to the config file

## Step Enablement Logic

A step runs if ALL of these are true:
1. It has configuration in the `steps` section (implicitly enabled), OR `enabled: true` is set
2. It is NOT in the `--skip` list
3. If `--only` is specified, it IS in that list
4. It is not `enabled: false`

## Config Template

See references/config-template.yaml for a complete, ready-to-adapt config template with all steps and options. When creating a new config, copy this template and adjust the `project`, `input`, `materials`, and `steps.predict.vlm` sections to match the user's asset and requirements.
