# Validation Agent

Validation Agent is the CLI-first release gate for generated 3D content. It
runs prompt-driven or config-driven checks over USD, image, video, render, and
physics-evidence artifacts, then writes stable request, plan, and result JSON
reports with per-template verdicts.

Validation Agent V1 is scoped to local CLI and Python contracts for release
0.5. It does not ship a REST service, OpenAPI spec, or hosted deployment
surface in this release.

## Overview

Validation Agent answers questions such as:

- Does this generated USD render successfully?
- Does the rendered asset look like the prompt or reference evidence?
- Does this USD have sane authored physics?
- Does an existing behavior/refine artifact show an approved rollout?

The V1 template allowlist is:

- `render_valid`: render evidence preflight and artifact detection.
- `look_right`: prompt/reference visual validation over current and reference
  evidence.
- `physics_sane`: deterministic USD physics authoring sanity checks.
- `physical_behavior`: evidence-backed behavior validation from existing
  rollout, video, simulation, or Physics Agent refine artifacts.

## Installation

From the repository root:

```bash
uv pip install -e .
uv pip install -e apps/validation_agent
```

## Runtime Dependencies

`physics_sane` and `physical_behavior` can run from local evidence without a
VLM. Validation Agent runtime USD rendering supports this capability-restricted
subset of the shared rendering contract:

- `remote`: set `RENDER_ENDPOINT` for an OVRTX-compatible REST service, or set
  `NVCF_RENDER_FUNCTION_ID` plus `NGC_API_KEY` for NVCF rendering.
- `ovrtx`: render locally through the isolated OVRTX environment. This requires
  a supported Linux RTX GPU runtime but no REST endpoint.
- A VLM or final-judge key when live `look_right` judging is enabled. Configure
  this through the request `policy.look_right_vlm` or
  `policy.look_right_llm_judge`.

When required renderer or VLM dependencies are unavailable, Validation Agent
reports structured warning or failure issues instead of silently passing.
Canonical shared backends that this surface cannot run, currently `warp` and
`mock`, return a structured `render.renderer_unavailable` warning. An unknown
backend name is a configuration error: it returns `render.backend_unknown` as a
structured failure and makes the validation verdict fail.

## Quick Start

For prompt-driven validation, pass a task and one or more inputs:

```bash
validation-agent validate \
  --task "Validate that this generated asset renders successfully." \
  --template render_valid \
  --render-backend remote \
  --render-view corner \
  --output-dir .validation-runs/my_asset \
  /absolute/path/to/generated_asset.usd
```

For the hello-world path, run the checked-in behavior-evidence config. It does
not need a renderer or VLM key:

```bash
validation-agent run \
  apps/validation_agent/examples/configs/steel_scaffold_behavior_refine_summary.yaml
```

The examples under [`examples/`](examples/) cover three realistic release
cases without requiring users to run Material, Physics, or Texture first:

- Electrician's toolbox visual/reference validation from public SimReady USD
  and thumbnail evidence.
- Steel rolling scaffold known-negative public PhysX source-asset audit for
  `physics_sane`.
- Steel rolling scaffold behavior/refine-summary hello-world evidence for
  `physical_behavior`.

## CLI Reference

```bash
# Prompt-driven direct input path
validation-agent validate --task "Validate render evidence" INPUT
validation-agent validate --task "Validate physics authoring" --template physics_sane INPUT

# Runtime render options for USD visual checks
validation-agent validate --task "Validate this asset" \
  --template render_valid \
  --render-backend remote --render-view corner \
  --image-width 512 --image-height 512 \
  INPUT.usd

# Config-driven path
validation-agent run CONFIG
validation-agent run CONFIG --dry-run
validation-agent run CONFIG --output-dir artifacts/validation
validation-agent run CONFIG --template render_valid --template look_right
validation-agent run CONFIG --focus-prim /World/Object
validation-agent run CONFIG --fail-on-warn
validation-agent run CONFIG --format json
```

Use config-driven `validation-agent run CONFIG` for live `look_right`
reference judging, because the VLM judge policy lives in the request config.

## Configuration

Both `validate --task ... INPUT...` and `run CONFIG` converge into the same
stable `world_understanding.validation.ValidationRequest` schema before
templates execute. A visual validation config:

```yaml
task_description: Validate that the rendered asset matches the reference.
inputs:
  - output.usd
requested_templates:
  - render_valid
  - look_right
render:
  backend: remote
  image_width: 512
  image_height: 512
  views:
    - corner
policy:
  visual_evidence_mode: canonical_usd
  look_right_vlm:
    backend: nim
    model: qwen/qwen3.5-397b-a17b
  reference_image_paths:
    - reference.png
project:
  working_dir: .validation-runs/example
```

`render.backend` is the authoritative V1 selector. The legacy free-form
`policy.render_backend` value remains a compatibility input when
`render.backend` is omitted. If both are present they must identify the same
backend; conflicting values are rejected instead of silently selecting one.
Backend spelling is trimmed and matched case-insensitively for compatibility,
and an omitted or blank value defaults to `remote`.

Paths in configs are resolved relative to the config file directory unless they
are absolute. Direct `validate` input paths and `--reference-image` paths are
resolved relative to the current working directory. Persisted direct-run
requests preserve the user-provided path strings and the resolved output
directory.

## Outputs

Every run writes:

- `validation_request.json`: effective request after CLI overrides.
- `validation_plan.json`: resolved inputs and ordered template plan.
- `validation_result.json`: final verdict, per-template statuses, issues,
  metrics, evidence, artifact paths, and recommended action.

By default, artifacts go under `.validation-runs/validation-agent` unless the
config or `--output-dir` chooses a different working directory. Example configs
write under `apps/validation_agent/examples/runs/`, which is ignored by git.

## Verdicts And Exit Codes

Top-level verdicts are:

- `pass`: selected templates passed.
- `warn`: validation completed with non-blocking issues.
- `fail`: blocking validation issue.
- `needs_refinement`: the asset or behavior needs another iteration.
- `planned`: dry-run output with no executed templates.

Template statuses are `passed`, `warn`, `failed`, `needs_refinement`,
`skipped`, and `error`.

Exit codes are CI-friendly:

- `0` for `pass`, `planned`, and `warn` by default.
- `1` for `fail` and `needs_refinement`.
- `1` for `warn` when `--fail-on-warn` is set.
- `2` for CLI/config/runtime setup errors.

Release-gate configs can make unavailable dependencies blocking:

```yaml
policy:
  gate_policy:
    dependency_unavailable: block
```

When this gate trips, issues such as `render.renderer_unavailable` or
`visual.judge_unavailable` add a top-level
`validation.dependency_unavailable` failure so missing credentials or render
services cannot be mistaken for passing validation.
