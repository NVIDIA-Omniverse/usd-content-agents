---
name: joint-agent-cli
description: "Fixed pipeline reference: Run the Joint Agent CLI for config-driven articulated-component classification, Stage 2 candidates, and USD package publication. Use when the user explicitly requests joint-agent, its YAML/config pipeline, benchmark-compatible artifacts, or fixed pipeline package output; use the reviewed agentic articulation workflow for unqualified tasks."
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - joint-agent
  - cli
  - articulation
  - usd
tools:
  - Shell
  - Filesystem
  - Python
compatibility: Requires Linux or WSL2, the repo Python environment, the joint-agent package, a supported public model backend, a renderer, and a USD-family input. On Windows the fixed pipeline runs under WSL2 with the Warp backend only; native Windows execution and local OVRTX under WSL2 are unsupported.
---

# Fixed Pipeline Reference: Joint Agent CLI

Joint Agent 0.5 is a Research Preview for classifying articulated components,
inferring structured joint candidates, and publishing a self-contained USD
package with joint topology and evidence.

## When to Use

- Use when the user asks to run the local `joint-agent` CLI.
- Use for articulated props, robot arms, or other assets whose components and
  possible joints need analysis.
- Use when the user wants Stage 2 articulation candidates or Joint Rigger
  package output.
- For an unqualified articulation task, use `content-workflow-cli articulation`
  or the root `content-workflow-asset` skill instead.
- Use `joint-agent-client` for an already-running REST service.
- Use `joint-agent-validation` after package publication when the user asks for
  Gate 3A or Gate 3B validation.

## Limitations

- Joint Agent 0.5 is a Research Preview. Model-inferred topology can be wrong.
- Successful `owned_core` authoring guarantees package generation and graph
  readback for the accepted structured input; it does not guarantee simulation
  readiness.
- Outputs may lack complete rigid-body, mass/collider, articulation-root,
  joint-state, or drive/mimic schemas and may fail Gate 3 validation.
- The 0.5 owned bridge supports revolute and prismatic candidates. Spherical
  candidates fail closed, while empty or all-unready inputs complete without a
  generated package.
- Keep API keys in the local environment or `.env`. Never print or commit them.
- Input assets and external dependencies must remain available until the final
  self-contained package is published.

## Prerequisites

Activate the repo environment and install the CLI:

```bash
source .venv/bin/activate
uv pip install -e ".[warp]" -e apps/joint_agent
```

The root `warp` extra supplies the Warp/Newton renderer required by the WSL2
path; native Linux runs may instead select the configured remote renderer.

Configure one supported public model backend in `.env`:

| Backend | Credential |
|---|---|
| NVIDIA NIM | `NVIDIA_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |

Configure the selected renderer in the YAML file. For a remote renderer, set
`RENDER_ENDPOINT` or the public endpoint settings required by that backend.

1. **optimize_usd** — Flatten/deinstance USD via NVCF scene optimizer
2. **identify_asset** — Render whole-scene preview and identify the overall asset via VLM
3. **analyze_structure** — LLM hierarchy analysis for segment/joint naming
4. **build_dataset_usd** — Render per-prim views for VLM input
5. **build_dataset_prepare_dataset** — Prepare dataset entries with classification specs
6. **predict** — VLM inference for per-component material classification
7. **restore_usd** — Map predictions back from optimized USD paths to the original asset paths using optimizer metadata
8. **infer_articulation_candidates** — Produce the accepted Stage 2 candidate document when enabled
9. **apply_joint_rigger** — Optionally author topology through `owned_core`; disabled by default

## Instructions

1. Start from the repository root.
2. Copy `apps/joint_agent/configs/byoa_joint_rigger.yaml` to a new local config.
3. Set `input.usd_path` to the user's asset and adjust asset-specific prompts.
4. For an authored output, explicitly enable
   `steps.infer_articulation_candidates`, or provide an existing accepted Stage
   2 document through `articulation_candidates_path`.
5. Explicitly set `steps.apply_joint_rigger.adapter: owned_core`, use a `.usdz`
   output, and keep `apply_masses` and `apply_collision` false. Enabled CLI/YAML
   steps must name the adapter explicitly; there is no implicit default.
6. Run `--dry-run` first to verify paths and planned steps.
7. Run the pipeline and preserve the working directory.
8. Report the final USD/USDZ path, candidate artifact, diagnostics, and any
   skipped or failed step.
9. When simulation readiness matters, invoke `joint-agent-validation` on the
   final package and keep its raw reports.

## Command Reference

Run the fixed pipeline:

```bash
joint-agent run CONFIG.yaml
```

Useful controls:

| Option | Meaning |
|---|---|
| `--dry-run` | Print the planned steps without executing them. |
| `--resume` | Continue from the last successful checkpoint. |
| `--clean` | Remove the configured working directory before starting. |
| `--only STEP[,STEP]` | Run only selected pipeline steps. |
| `--skip STEP[,STEP]` | Skip selected pipeline steps. |
| `--session-id ID` | Override the configured session identifier. |
| `--verbose` | Enable detailed logs. |

Other commands:

```bash
joint-agent analyze CONFIG.yaml
joint-agent predict CONFIG.yaml
joint-agent build-dataset usd CONFIG.yaml
joint-agent build-dataset prepare-dataset CONFIG.yaml
joint-agent validate-rigged-reference REFERENCE.usdz CANDIDATES.json
```

## Common Workflows

### Prepare a bring-your-own-asset run

```bash
cp apps/joint_agent/configs/byoa_joint_rigger.yaml my_joint_asset.yaml
# Edit input.usd_path and the asset-specific prompts.
joint-agent run my_joint_asset.yaml --dry-run
joint-agent run my_joint_asset.yaml
```

Resume after a transient model or renderer failure:

```bash
joint-agent run my_joint_asset.yaml --resume
```

## Apply the built-in topology-only Joint Rigger

Start from `apps/joint_agent/configs/byoa_joint_rigger.yaml`. It enables fresh
Stage 2 candidate inference and configures `adapter: owned_core`, the generated
`articulation_candidates_path`, a `.usdz` output path, and both
`apply_masses: false` and `apply_collision: false`. Package authoring remains
disabled until the user reviews the candidates and explicitly enables
`steps.apply_joint_rigger`. The adapter consumes the Stage 2 document directly
and does not require predictions or the optional external `usd_joint_rigger`
package. This Research Preview does not author the remaining
simulation-readiness schemas.

For a fresh unified-config run, explicitly enable
`steps.infer_articulation_candidates` so that accepted Stage 2 input exists.
For a package-only rerun, point `articulation_candidates_path` at an existing
accepted Stage 2 document. An enabled CLI/YAML `apply_joint_rigger` step must
set `adapter` explicitly; use `owned_core` for the built-in Research Preview.
The REST service has a separate request contract and selects `owned_core` when
its adapter field is omitted.

Publish topology from an accepted Stage 2 document:

```yaml
steps:
  apply_joint_rigger:
    enabled: true
    adapter: owned_core
    articulation_candidates_path: path/to/articulation_candidates.json
    output_usd_path: path/to/rigged.usdz
    diagnostics_path: path/to/joint_rigger_diagnostics.json
    validation_path: path/to/joint_rigger_validation.json
    apply_masses: false
    apply_collision: false
```

## Output Format

Return:

- config and input asset paths;
- session working directory;
- Stage 2 candidate JSON path and candidate count;
- final published USD/USDZ path;
- Joint Rigger diagnostics and readback status;
- each skipped, blocked, or failed step;
- an explicit Research Preview limitation when simulation readiness has not
  been validated.

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| Model credential error | The selected backend key is missing. | Set the matching public credential in `.env` and rerun with `--resume`. |
| Renderer connection failure | The configured renderer is unavailable. | Start the local renderer or correct the public endpoint. |
| Input file not found | Config paths resolve from the config location. | Use a correct relative path or an absolute path. |
| Candidate output is empty | The asset may not contain supported articulated evidence or prompts are too generic. | Review renders, component predictions, and asset-specific prompts. |
| Package opens but Gate 3 fails | Package integrity and simulation readiness are separate contracts. | Use `joint-agent-validation`, preserve findings, and report the 0.5 limitation. |
