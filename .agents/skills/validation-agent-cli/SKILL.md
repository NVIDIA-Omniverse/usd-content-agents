---
name: validation-agent-cli
description: Run the Validation Agent Research Preview CLI for config-driven or prompt-driven validation of generated 3D content. Use when the user asks for validation-agent, run validation agent, validate asset, validate generated USD/image/video evidence, look_right, physics_sane, physical_behavior, render_valid, release validation checks, or Validation Agent V1 CLI artifacts.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - validation-agent
  - cli
  - usd
  - qa
tools:
  - Shell
  - Filesystem
  - Python
  - wu
compatibility: Requires the repo Python environment, the validation-agent package, local validation inputs, and optional render or VLM credentials for live visual validation.
---

# Validation Agent CLI

Validation Agent is the research-preview, CLI-first release gate for generated
3D content. It runs prompt-driven or config-driven checks over USD, image,
video, render, and physics-evidence artifacts, then writes structured request,
plan, and result JSON.

## When to Use

- Use when the user asks to run `validation-agent` directly.
- Use when the user wants to validate a generated asset, USD renderability,
  prompt/reference visual match, authored physics sanity, or behavior evidence.
- Use when the user mentions `render_valid`, `look_right`, `physics_sane`, or
  `physical_behavior`.
- Use when the user needs the structured Validation Agent artifacts:
  `validation_request.json`, `validation_plan.json`, and
  `validation_result.json`.
- Use `joint-agent-validation` instead when the user explicitly asks for Joint
  Agent Gate 3A Isaac Sim Asset Validator or Gate 3B SimReady Foundation runs.
- Use a future service/deploy skill instead if product scope adds REST,
  OpenAPI, or hosted service surfaces. Validation Agent V1 for release 0.5 is
  Research Preview and CLI/Python-contract only.

## Limitations

- Keep secrets out of chat and commits. Tell the user to set provider keys in
  their local environment or repo-root `.env`; never ask them to paste keys.
- Validation Agent V1 Research Preview does not run a REST service, publish an
  OpenAPI surface, or host a deployment endpoint.
- Validation Agent templates are not the Joint Agent Gate 3A/3B validator
  profiles. Route those requests to `joint-agent-validation`.
- `physical_behavior` consumes existing rollout, video, simulation, or Physics
  Agent refine evidence; it does not run simulation or tune/refine loops.
- Live `look_right` judging needs a configured VLM or final-judge policy.
  Without it, the template reports structured skipped or unavailable status.
- USD-only visual checks need a renderer for canonical runtime evidence.
  Missing render dependencies can warn or fail depending on gate policy.
- Config paths resolve relative to the config file. Direct `validate` input and
  `--reference-image` paths resolve relative to the current working directory.

## Prerequisites

- Activate the repo Python environment before running commands.
- Install the repo and standalone app package:

```bash
uv pip install -e .
uv pip install -e apps/validation_agent
```

- Confirm `validation-agent` is on `PATH`.
- Choose `--render-backend remote` for a REST renderer or
  `--render-backend ovrtx` for local isolated OVRTX rendering. For `remote`, set
  either `RENDER_ENDPOINT` for an OVRTX-compatible service or
  `NVCF_RENDER_FUNCTION_ID` plus `NGC_API_KEY`. Local `ovrtx` requires a
  supported Linux RTX GPU runtime and no REST endpoint.
- For live `look_right`, configure `policy.look_right_vlm` or
  `policy.look_right_llm_judge` in the request config and set the selected
  provider credential locally.
- Read `apps/validation_agent/README.md` for CLI details and
  `apps/validation_agent/examples/README.md` for the release examples before
  writing long commands or docs. When changing request or result contracts,
  inspect the shipped schema code under `world_understanding/validation/` and
  the focused validation-agent tests directly.

## Instructions

1. Start from the repo root and activate `.venv`.
2. Decide whether the user needs config-driven repeatability or prompt-driven
   direct input.
3. For config-driven runs, choose or create a Validation Agent V1 request YAML
   and run `validation-agent run CONFIG`.
4. For prompt-driven runs, pass `--task`, one or more inputs, and optional
   `--template`, render, reference-image, and output-dir overrides.
5. Prefer the shipped examples under `apps/validation_agent/examples/configs/`
   for release-scope demos and documentation.
6. Inspect `validation_result.json` and report verdict, template statuses,
   issue codes, artifact paths, and missing dependency notes.

```bash
source .venv/bin/activate
validation-agent run apps/validation_agent/examples/configs/steel_scaffold_behavior_refine_summary.yaml
validation-agent validate --task "Validate that this asset looks correct" output.usd
```

## Command Reference

### Config-Driven Runs

```bash
validation-agent run CONFIG [OPTIONS]
```

| Option | Description |
|---|---|
| `--dry-run` | Write a planned result without executing templates. |
| `--output-dir <dir>` | Override the artifact directory. |
| `--template <name>` | Select one template; repeat for multiple templates. |
| `--focus-prim <path>` | Focus validation on one USD prim. |
| `--fail-on-warn` | Return exit code 1 for warning verdicts. |
| `--format json` | Print JSON summary to stdout. |

Use `validation-agent run CONFIG` for live `look_right` reference judging
because VLM and final-judge policy live in the request config.

### Prompt-Driven Runs

```bash
validation-agent validate --task "Validate render evidence" INPUT
validation-agent validate --task "Validate physics authoring" --template physics_sane INPUT
validation-agent validate --task "Validate this asset" \
  --template render_valid \
  --render-backend remote --render-view corner \
  --image-width 512 --image-height 512 \
  --reference-image reference.png \
  --output-dir .validation-runs/my_asset \
  INPUT.usd
```

Use `--reference-image` for reference visual evidence. Positional image inputs
are treated as current asset or render evidence.

## Templates

| Template | Use For | Dependency Notes |
|---|---|---|
| `render_valid` | Render evidence preflight and artifact detection. | Supports `remote` REST rendering and local `ovrtx`; other shared backends report structured unavailability. |
| `look_right` | Prompt/reference visual validation over current and reference evidence. | Needs visual evidence plus live VLM/final-judge config or a precomputed response. |
| `physics_sane` | Deterministic USD physics authoring sanity checks. | Runs locally from USD physics schemas; no VLM required. |
| `physical_behavior` | Evidence-backed behavior validation from existing rollout, video, simulation, or refine artifacts. | Consumes existing evidence and can pass, warn, fail, refine, or skip. |

## Common Workflows

```bash
# Repeatable behavior evidence smoke, checked in and quick to run.
validation-agent run apps/validation_agent/examples/configs/steel_scaffold_behavior_refine_summary.yaml

# Known-negative public source-asset physics audit.
validation-agent run apps/validation_agent/examples/configs/steel_scaffold_known_negative_physics.yaml

# Visual/reference check after downloading the public SimReady toolbox fixture.
validation-agent run apps/validation_agent/examples/configs/electricians_toolbox_visual.yaml

# Render preflight directly from a generated USD.
validation-agent validate \
  --task "Validate that the generated asset renders successfully." \
  --template render_valid \
  --render-backend remote --render-view corner \
  generated_asset.usd
```

The three release examples are documented in
`apps/validation_agent/examples/README.md`. Point users there for fixture
downloads, expected outcomes, and per-example setup instead of duplicating the
long walkthrough in the skill.

## Output Format

Report these items after a run or handoff:

- Command executed and whether it was `run`, `validate`, or `--dry-run`.
- Config path or direct input paths, selected templates, output directory, and
  whether `--fail-on-warn` was used.
- Top-level verdict and exit-code meaning.
- Per-template statuses for `render_valid`, `look_right`, `physics_sane`, and
  `physical_behavior` when present.
- Key artifacts:
  - `validation_request.json` for the effective request.
  - `validation_plan.json` for resolved inputs and ordered template plan.
  - `validation_result.json` for final verdict, issues, metrics, evidence,
    artifact paths, and recommended action.
- Any missing renderer, VLM/final-judge, fixture, or evidence dependency.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `validation-agent` is not found | The app package is not installed in the active environment. | Run `uv pip install -e . -e apps/validation_agent` from the repo root. |
| Config-relative files are missing | Paths inside CONFIG resolve from the config file directory. | Rewrite paths relative to CONFIG or make them absolute. |
| Direct reference image is ignored | Reference evidence was passed as a positional input. | Use `--reference-image`; positional images are current evidence. |
| Renderer unavailable | The selected canonical backend is unsupported by Validation Agent, or its runtime dependency is unavailable. | Use `remote` with `RENDER_ENDPOINT`, use local `ovrtx` on a supported Linux RTX host, or run non-visual templates. |
| `render.backend_unknown` | The configured name is not part of the shared USD rendering contract. | Choose `remote` or `ovrtx`; do not treat the failed result as missing infrastructure. |
| `look_right` skips or reports judge unavailable | No live VLM/final-judge config or provider credential is available. | Use config-driven `run` with `policy.look_right_vlm` or `policy.look_right_llm_judge` and local credentials. |
| Warning verdict exits 0 | Warnings are non-blocking by default. | Add `--fail-on-warn` for CI gates that should fail on `warn`. |
| `physical_behavior` skips | Existing behavior, video, rollout, simulation, or refine evidence is missing. | Provide `physical_behavior_evidence`, sampled frames, rollout USD/video, or a Physics Agent refine summary/output dir in policy. |
