---
name: material-agent-cli
description: Run the direct Material Agent CLI for config-driven VLM material assignment pipelines. Use when the user wants to run material-agent commands directly, edit or execute a material-agent config YAML, resume a fixed pipeline, benchmark material predictions, build datasets from USD files, apply predicted materials, configure a material run, or try a SimReady demo.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - material-agent
  - cli
  - usd
  - vlm
tools:
  - Shell
  - Filesystem
  - Python
  - wu
compatibility: Requires the material-agent CLI, a repo Python environment, provider credentials for the selected VLM/LLM/image-generation backends, a render endpoint for remote rendering configs, and a materials manifest with USD material bindings.
---

# Material Agent CLI

The Material Agent assigns materials to 3D object parts by rendering USD prims,
asking a VLM to choose from a constrained material library, and optionally
applying the predictions back to a USD layer.

## When to Use

- Use when the user asks to run `material-agent` directly from the command
  line.
- Use when the user wants a config-driven Material Agent run, to resume a failed
  material pipeline, or to apply existing predictions to USD.
- Use when the user wants to iteratively refine material assignments with a
  predict/apply/render/judge loop.
- Use when the user wants to benchmark material predictions or build VLM-ready
  datasets from USD renders.
- Use when the user wants a public SimReady demo asset with minimal local data
  setup.
- For an interactive Content Workbench asset workflow with a Codex or Claude
  runner, start the agent from `agentic/` and use its `content-workflow-cli`
  skill.
- Use service or Docker deploy skills instead when the user wants to operate
  the REST service rather than the local CLI.

## Limitations

- Keep secrets out of chat and commits. Tell the user to set provider keys in
  their local environment or repo-root `.env`; never ask them to paste keys.
- The CLI needs a valid config YAML, a readable USD input, and a materials
  manifest containing material names, descriptions, and USD bindings.
- Config paths such as `input.usd_path`, `input.reference_images`, and
  `materials.path` resolve relative to the config file, not the current shell
  directory.
- Step configs must not contain path keys such as `usd_path`, `output_dir`,
  `dataset`, or `predictions_path`; the executor wires paths from the project
  and input sections.
- Remote rendering or optimization configs need deployed services. For a local
  OVRTX Docker sidecar, use `RENDER_ENDPOINT=http://localhost:8001` and keep
  render concurrency conservative unless the endpoint fronts multiple service
  instances.
- Generated reference images are optional and need a configured image
  generation backend plus its required key or endpoint.

## Prerequisites

- Locate the CLI and checkout with the bounded discovery sequence below. If a
  checkout is used, activate its Python environment before running commands.
- Set the VLM/LLM provider key required by the selected backend. Public
  defaults usually use `NVIDIA_API_KEY`; other supported backends can use
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`.
- Set `RENDER_ENDPOINT` and `OPTIMIZER_ENDPOINT` only when the config uses
  remote rendering or optimization services.
- Set `WU_S3_BUCKET`, `WU_S3_PROFILE`, `WU_S3_REGION`, and standard AWS
  credentials only when the run uploads assets to S3.
- Prepare a materials manifest YAML. The default library lives under
  `apps/material_agent/data/materials/material_libs_default/`.

### Locate the CLI and Checkout

Use only known candidates; do not crawl the filesystem or assume a host-specific
checkout path.

1. Run `command -v material-agent` (or `Get-Command material-agent` in a native
   Windows convenience shell). If it succeeds, record the executable path.
2. Ask Git for the checkout containing the current directory with
   `git rev-parse --show-toplevel`. Accept it only when
   `apps/material_agent/pyproject.toml` exists below that root.
3. If the agent host exposes workspace roots, or the user supplied a checkout
   path, validate only those finite candidates with
   `git -C <candidate> rev-parse --show-toplevel` and the same repository marker.
4. If neither the CLI nor a valid checkout is found, stop and report that PATH,
   the current Git root, and the available workspace/user candidates were
   checked. Ask for the checkout path or installation instructions. Do not run
   an unbounded search such as `find /`.

When a valid checkout is selected, it takes precedence over the executable
recorded from `PATH`. Change to the checkout root and activate that checkout's
`.venv` with the current shell's activation command (`source
.venv/bin/activate` on POSIX shells, `.\.venv\Scripts\Activate.ps1` in
PowerShell, or `.venv\Scripts\activate.bat` in `cmd.exe`). Stop and report
the failure if activation does not succeed. Then re-run `command -v
material-agent` or `Get-Command material-agent`, normalize the result, and
require it to resolve below the selected checkout's `.venv` executable
directory. Stop and report a missing or mismatched executable; do not fall back
to the previously recorded installed CLI after selecting a checkout.

If no checkout is selected, retain the installed CLI recorded from `PATH` and
use user-provided config and asset paths instead of assuming the repository
examples are available.

## Instructions

1. Locate the CLI and checkout as described above. When using a checkout, start
   from its root, activate `.venv`, re-resolve `material-agent`, and verify that
   its normalized path is below that checkout's `.venv` executable directory.
   Stop and report any activation, missing-executable, or path-mismatch failure.
2. Choose a config. For a first local run, use
   `apps/material_agent/configs/unified_example.yaml`. For a user asset, copy
   that config or use `material-agent configure`.
3. Verify the config points at the input USD, optional reference images, and
   materials manifest. Keep relative paths relative to the config file.
4. Run a dry run before a new or heavily edited config.
5. Run the full pipeline, or use `--only`, `--skip`, and `--resume` to control
   execution.
6. Inspect the working directory and report the key artifacts from the output
   format. When predictions are present, also surface every record where
   `materials.evidence_reconciliation.review_required` is `true`; do not treat
   a successful pipeline exit as approval of those material assignments.

```bash
source .venv/bin/activate
material-agent run apps/material_agent/configs/unified_example.yaml --dry-run
material-agent run apps/material_agent/configs/unified_example.yaml
```

### Primary Command

```bash
material-agent run <config.yaml> [OPTIONS]
```

| Option | Description |
|---|---|
| `--skip <steps>` | Comma-separated steps to skip. |
| `--only <steps>` | Comma-separated steps to run exclusively. |
| `--session-id <id>` | Reuse or override the session ID. |
| `--resume` | Continue from the last successful checkpoint. |
| `--dry-run` | Show the pipeline plan without executing. |
| `--clean` | Delete the working directory before starting. |
| `--verbose`, `-v` | Enable debug logging. |
| `--log-file <path>` | Write logs to a file. |
| `--log-level <level>` | Override the default `INFO` log level. |

### Other Commands

| Command | Description |
|---|---|
| `material-agent configure <output.yaml>` | Interactive config creation wizard. |
| `material-agent predict <config.yaml>` | Run VLM prediction only. |
| `material-agent apply <config.yaml>` | Apply predictions to USD only. |
| `material-agent refine <config.yaml>` | Iteratively refine materials with predict/apply/render/judge. |
| `material-agent benchmark <config.yaml>` | Predict and evaluate with LLM-judge scoring. |
| `material-agent evaluate <config.yaml> [predictions.jsonl]` | Evaluate existing predictions. |
| `material-agent build-dataset usd <config.yaml>` | Build dataset images from USD renders. |
| `material-agent build-dataset pdf_vectorstore <config.yaml>` | Build a specification-evidence vector store from PDFs. |
| `material-agent build-dataset prepare-dataset <config.yaml>` | Prepare VLM dataset records. |

See `references/commands.md` for full command options.

### Pipeline Steps

The unified config schema recognizes these steps in execution order:

1. `validate_input` - establish an optional USD validation baseline before any
   material processing.
2. `optimize_usd` - flatten, split, deduplicate, or deinstance USD through the
   configured optimizer.
3. `render_preview` - render lightweight scene previews for reference-image
   generation.
4. `identify_asset` - classify the overall asset and derive prompt context from
   previews.
5. `generate_reference_image` - generate optional photorealistic reference
   images from previews and text prompts.
6. `build_dataset_usd` - render prim-level VLM input images.
7. `build_dataset_pdf_vectorstore` - index optional PDF specification evidence.
8. `build_dataset_prepare_dataset` - assemble rendered images and visual
   prompts while retaining optional specification evidence separately.
9. `cluster_prims` - group visually similar prims before prediction.
10. `predict` - run VLM material assignment.
11. `expand_cluster_predictions` - expand cluster-level predictions back to
    member prims.
12. `benchmark` - run prediction plus LLM-judge evaluation; mutually exclusive
    with `predict` in one run.
13. `validate_predictions` - validate or repair predicted material names.
14. `harmonize_predictions` - resolve conflicts for instanced or repeated
    parts.
15. `restore_usd` - remap predictions from optimized paths back to original
    paths before application or refinement.
16. `apply` - apply predictions to USD.
17. `evaluate` - score existing predictions against ground truth with an LLM
    judge.
18. `refine` - run the iterative predict/apply/render/judge loop; mutually
    exclusive with `apply` in one run.
19. `validate_output` - compare the materialized output against the input
    baseline.
20. `render` - render final output images when enabled.

See `references/pipeline-steps.md` for configuration details, outputs, and
step-specific caveats.

### Common Workflows

```bash
# Run the configured pipeline.
material-agent run apps/material_agent/configs/unified_example.yaml

# Run only prediction, application, and final render.
material-agent run apps/material_agent/configs/unified_example.yaml --only predict,apply,render

# Skip optimization when the USD is already prepared.
material-agent run apps/material_agent/configs/unified_example.yaml --skip optimize_usd

# Preview what the pipeline will do.
material-agent run apps/material_agent/configs/unified_example.yaml --dry-run

# Resume after a failed step.
material-agent run apps/material_agent/configs/unified_example.yaml --resume

# Create a config with a materials manifest and reference image.
material-agent configure my_pipeline.yaml -m materials/my_materials.yaml -r reference.jpg

# Benchmark a configured dataset.
material-agent benchmark configs/benchmark.yaml -d dataset.jsonl -o results/
```

### Generated Reference Images

Enable `render_preview` and `generate_reference_image` when the user has no
reference photos and wants a text-described target appearance. The generated
image is injected into the dataset for the prediction step.

```yaml
steps:
  render_preview:
    enabled: true
    cameras: ["+x+y+z"]
  generate_reference_image:
    enabled: true
    prompt: "aluminum frame with a blue plastic tray"
    num_images: 1
```

### SimReady Demo

When the user asks to "try material agent" or run a public demo asset, follow
`references/simready-quickstart.md`. It covers downloading curated SimReady
assets, writing a config that uses the shipped default material library, and
running the pipeline end to end. For UR10 assets, keep
`prim_filters.skip_instances: false` so the agent sees meshes.

### Config Authoring

Prefer copying `apps/material_agent/configs/unified_example.yaml` for new
configs. Adapt only the user-specific fields:

- `project.name` and `project.session_id`
- `input.usd_path`
- `input.reference_images` when reference photos are available
- `materials.path`
- `steps.predict.vlm.model`
- `steps.render.enabled`

Use `references/config-template.yaml` for a complete ready-to-adapt template
and `references/config-reference.md` for the full schema. Keep the prompts,
renderer settings, and prediction settings unless the user explicitly asks to
change them.

### Local OpenAI-Compatible VLM Limits

For a local OpenAI-compatible endpoint, set `steps.predict.vlm.backend` to
`openai`, configure its model and base URL, and use `max_tokens` as the Material
Agent config field. The OpenAI backend normalizes that value to the request
field required by the selected model family; do not copy a provider-specific
`max_completion_tokens` field into a generic local-model config or set both
fields.

Before running, inspect the local server's model metadata, launch arguments, or
operator documentation for both its context-window and maximum-output limits.
Not every `/v1/models` response advertises those limits. Set `max_tokens` no
higher than the server's advertised output cap and leave enough context for the
prompt, material list, and images. Never assume a fixed cap from an example
config. Keep any endpoint credential in the local environment; never ask the
user to paste it.

### Finding Materials

A materials file is a YAML manifest with a `library_path` pointing at the USD
material library and `entries` listing available materials:

```yaml
library_path: "path/to/material_libs.usd"
entries:
  - name: "Aluminum Polished"
    description: "A polished aluminum for structural parts"
    binding: "/World/metal_library/Looks/Aluminum_Polished"
```

Use the default library first, ask for a user-provided manifest second, and
create a new manifest only when the user has a material USD and wants that
library enumerated.

## Output Format

Report these items after a run or handoff:

- Command executed and whether it was full pipeline, `--only`, `--skip`,
  `--resume`, or `--dry-run`.
- Config path and session ID.
- Working directory, usually `.<session_id>/` next to the config unless
  `project.working_dir` overrides it.
- Key artifacts when present:
  - `validation/input/` for pre-run USD validation reports.
  - `dataset/usd/` for rendered prim images and manifests.
  - `dataset/dataset.jsonl` for VLM-ready records.
  - `clusters/` for clustering reports and representative mappings.
  - `generated_refs/` for generated reference images.
  - `predictions/predictions.jsonl` and `report.html` for VLM output.
    Report the prim ID, visual material, and conflicting specification claims
    for every prediction whose
    `materials.evidence_reconciliation.review_required` value is `true`.
  - `evaluation/` for LLM-judge scoring outputs.
  - `iterations/` for iterative refinement artifacts.
  - `restored/restored_predictions.jsonl` when restore/remap ran.
  - `output/output.usd` or the configured `output.usd_path`.
  - `validation/output/` for post-run USD validation reports.
  - `renders/` for final render images.
- Any missing credentials, service endpoints, invalid material bindings, or
  config-path issues.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `material-agent` or the checkout cannot be found | The CLI is not on PATH and the current/known workspace roots are not a valid checkout. | Follow the bounded discovery sequence, report the candidates checked, and ask for a checkout path or installation instructions; do not search the whole filesystem. |
| API key required | The selected VLM, LLM, or image-generation backend has no credential. | Set the required key locally or in `.env`; do not paste it into chat. |
| Local OpenAI-compatible server rejects the token limit | The copied config requests more output than the selected server/model allows, or uses a provider-specific token field. | Inspect the server's advertised model/output limits, use `steps.predict.vlm.max_tokens`, and lower it below the server cap while preserving room for prompt and image context. |
| Pipeline fails midway | A step failed after writing partial artifacts. | Re-run with `--resume`; use `--clean` only when the user wants to discard prior artifacts. |
| Forbidden path key in step config | Step configs contain path keys that the executor owns. | Remove `usd_path`, `output_dir`, `dataset`, or `predictions_path` from step configs. |
| Relative paths resolve unexpectedly | Config paths resolve from the config file directory. | Rewrite paths relative to the config file or make them absolute. |
| No meshes found for a SimReady UR10 asset | Instance filtering hid the geometry. | Set `prim_filters.skip_instances: false`. |
| Remote rendering fails or stalls | `RENDER_ENDPOINT` is missing, unhealthy, or over-concurrent. | Check endpoint health and keep local OVRTX worker/request concurrency at 1. |
| Material names do not match the library | Predictions chose names outside the manifest. | Keep `validate_predictions` enabled and verify the manifest entries are descriptive. |
| Prediction requires evidence review | Untrusted specification claims conflict with the fixed visual material label. | Report the prim ID, visual material, and `conflicting_spec_materials`; keep the visual label unchanged until a human resolves the conflict. |
