---
name: physics-agent-cli
description: Run the Physics Agent CLI for VLM-based physics property classification, tuning, and refine-loop workflows for 3D assets. Use when the user wants to run the physics-agent CLI directly, classify asset components, identify asset type, build datasets from USD files, predict component material/type/physics properties, author simulation-ready USD physics schemas, tune authored physics parameters, or run iterative refine.
version: "0.1.2"
author: NVIDIA Content Agents
tags:
  - content-agents
  - physics-agent
  - cli
  - usd
  - vlm
tools:
  - Shell
  - Filesystem
  - Python
  - wu
compatibility: Requires Linux, a Linux container, or WSL2; the physics-agent CLI; a repo Python environment; provider credentials for the selected VLM backend; a render endpoint for remote rendering configs; and a scene optimizer endpoint or local scene optimizer when optimize_usd is enabled.
---

# Physics Agent CLI

The Physics Agent renders USD components, identifies the asset, predicts
material/component/physics properties for each part, and can author `UsdPhysics`
schemas into a simulation-ready output USD.

## When to Use

- Use when the user asks to run `physics-agent` directly from the command line.
- Use when the user wants component classification for a USD asset.
- Use when the user wants to author rigid-body, collision, mass, and physics
  material schemas from VLM predictions.
- Use when the user needs the instanced-USD/deinstance path for writable
  physics authoring.

Route requests outside the local CLI to the exact sibling skill:

| User intent | Required sibling skill |
|---|---|
| Call an already-running Physics Agent REST API, monitor its sessions, or download service artifacts | `physics-agent-client` |
| Build, start, stop, or configure the local Physics Agent Docker Compose service | `deploy-physics-agent-docker` |
| Provision Brev-hosted render/VLM dependencies or run the Brev hybrid Physics Agent workflow | `deploy-physics-agent-brev` |
| Assign or refine visual/PBR materials rather than physical properties | `material-agent-cli` |

## Limitations

- Keep secrets out of chat and commits. Tell the user to set provider keys in
  their local environment or repo-root `.env`; never ask them to paste keys.
- Official runtime targets are Linux, Linux containers, and WSL2. Native
  Windows shell execution is not supported; direct the user to WSL2 or Linux,
  not to a sibling skill that does not support the requested runtime.
- Config paths such as `input.usd_path` resolve relative to the config file,
  not the current shell directory.
- Instanced USD descendants are instance proxies and cannot be authored on
  directly. Use `optimize_usd` with deinstance enabled before `apply_physics`
  when predictions target instance proxies.
- `restore_usd` maps predictions back to original paths for reporting; when
  optimization runs, `apply_physics` authors onto the optimized/deinstanced USD
  using raw prediction paths.
- The classification pipeline does not apply visual/PBR materials or render a
  final beauty image. It stops after physics authoring unless the user invokes
  other tooling.
- Suspicious scale-driven mass estimates follow the configured
  `apply_physics.mass_scale_policy`; the default skips explicit mass while
  still authoring density, collision, and physics material properties.

## Prerequisites

- Activate the repo Python environment before running commands.
- Confirm `physics-agent` is installed and on `PATH`.
- Set the VLM provider key required by the selected backend. Public defaults
  usually use `NVIDIA_API_KEY`; other supported backends can use
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`.
- Set `RENDER_ENDPOINT` and `OPTIMIZER_ENDPOINT` only when the config uses
  remote rendering or optimization services. For a local OVRTX Docker sidecar,
  use `RENDER_ENDPOINT=http://localhost:8001`.
- Set `WU_S3_BUCKET`, `WU_S3_PROFILE`, `WU_S3_REGION`, and standard AWS
  credentials only when the run uploads assets to S3.

## Instructions

1. Start from the repo root and activate `.venv`.
2. Choose a config. For a first local run, use
   `apps/physics_agent/configs/lightbulb.yaml`.
3. Verify `input.usd_path`, optimizer settings, render backend, and VLM model.
4. For instanced assets, enable deinstance under
   `steps.optimize_usd.scene_optimizer_settings` and keep `restore_usd`
   enabled.
5. Run a dry run before a new or heavily edited config.
6. Run the full pipeline in the foreground, or use `--only`, `--skip`, and
   `--resume` to control execution.
7. Wait for the command to reach a terminal outcome before returning.
8. Report the command, config, session, terminal outcome, and artifacts using
   the output format.

```bash
# Run from Linux, a Linux container, or WSL2.
source .venv/bin/activate
physics-agent run apps/physics_agent/configs/lightbulb.yaml --dry-run
physics-agent run apps/physics_agent/configs/lightbulb.yaml
```

### Required Execution Lifecycle

- Treat every required `physics-agent` invocation, including `run`, `predict`,
  `tune`, `refine`, and dataset commands, as foreground work. Wait for the
  process to exit before reporting completion or returning from a one-shot or
  headless agent run.
- Do not append `&`, use `nohup`, create an ordinary detached shell job, or
  return while the required process is still tied to the current execution
  session. Session teardown can kill that process before it produces results.
- Do not promise to resume, poll, or wake up later through a mechanism that was
  not actually created or cannot fire after the current agent run ends.
- The only exception is a durable external monitor or handoff that the
  execution surface actually provides and that survives session teardown.
  Before returning, report its job or monitor ID, owner, log location, and
  exact status command. A local PID or shell job is not a durable handoff.
- If the foreground run cannot finish and no durable external monitor exists,
  report that constraint and the exact foreground command instead of starting
  work that will be abandoned.

### Primary Command

```bash
physics-agent run <config.yaml> [OPTIONS]
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
| `physics-agent run <config.yaml>` | Execute the unified multi-step pipeline. This is the primary command. |
| `physics-agent predict <config.yaml>` | Run the direct VLM prediction API on a prepared dataset. |
| `physics-agent tune [SCENARIO.yaml]` | Tune authored physics parameters against a simulator. |
| `physics-agent refine <SCENARIO.yaml>` | Iteratively run `tune`, judge the result, and refine the scenario. |
| `physics-agent pipeline <config.yaml>` | Deprecated alias for `run`; mention only for legacy reproduction. |
| `physics-agent build-dataset usd <config.yaml>` | Build dataset images from USD renders. |
| `physics-agent build-dataset prepare-dataset <config.yaml>` | Prepare VLM dataset records. |

### Predict Command

Use `predict` when dataset rendering and preparation have already happened and
the user only needs VLM inference. It calls the prediction API directly instead
of routing through the unified pipeline.

```bash
physics-agent predict <config.yaml> [OPTIONS]
```

| Option | Description |
|---|---|
| `--dataset <path>`, `-d <path>` | Override the prepared `dataset.jsonl` path from config. |
| `--output <dir>`, `-o <dir>` | Override the prediction output directory. |
| `--resume` | Resume from existing predictions. |
| `--verbose`, `-v` | Enable debug logging. |
| `--log-file <path>` | Write logs to a file. |
| `--log-level <level>` | Override the default `INFO` log level. |

`physics-agent run <config.yaml> --only predict` remains supported when the
prediction step should execute inside the pipeline checkpoint system.

### Tune Command

Use `tune` after `apply_physics` has authored a simulation-ready USD. It
patches candidate physics parameters, scores each candidate with the selected
simulation backend, and writes the best parameters plus a tuned USD.

```bash
physics-agent tune [SCENARIO.yaml] [OPTIONS]
```

Supply either `SCENARIO.yaml`, `--user-prompt`, or both. When both are present,
the explicit scenario YAML wins on field conflicts. `--physics-usd` is required
unless the scenario defines `physics_usd:`.

| Option | Description |
|---|---|
| `--user-prompt <text>` | Author a scenario from natural language. |
| `--physics-usd <path>` | Physics-authored USD to tune. |
| `--reference-image <path>` | Add judge image evidence; can repeat. |
| `--reference-video <path>` | Add judge video evidence; can repeat. |
| `--reference-description <text>` | Description for a reference image; can repeat. |
| `--reference-video-description <text>` | Description for a reference video; can repeat. |
| `--engine ovphysx\|newton\|fake` | Simulation backend. Default is `ovphysx`. |
| `--optimizer auto\|botorch\|random\|cma-es` | Optimizer. Default is `auto`. |
| `--output-dir <dir>`, `-o <dir>` | Destination for tune artifacts. |
| `--max-trials <n>` | Number of optimizer trials. |
| `--seed <n>` | Seed for optimizer and backend when supported. |
| `--judge/--no-judge` | Enable or disable the final VLM-as-judge pass. |
| `--judge-max-iterations <n>` | Metadata/pass-through only for single-shot `tune`; use `refine` for real iteration. |
| `--judge-max-tokens <n>` | Override judge response length. |
| `--judge-temperature <value>` | Override judge temperature. |
| `--verbose`, `-v` | Enable debug logging. |
| `--log-file <path>` | Write logs to a file. |
| `--log-level <level>` | Override the default `INFO` log level. |

For production OvPhysX + BoTorch tuning, install the `tuning` extra and provide
an OvPhysX daemon environment. The default daemon venv is
`~/.cache/wu/ovphysx_venv`; override it with `WU_OVPHYSX_VENV_DIR`. For Newton,
install the `newton` extra and select `--engine newton`. For tests or a local
smoke check, use `--engine fake --optimizer random`.

```bash
physics-agent tune apps/physics_agent/configs/tuning/drop_settle.yaml \
  --physics-usd output/physics/asset_physics.usda \
  --engine ovphysx --optimizer auto --output-dir output/tune

physics-agent tune --user-prompt "make this object bouncy" \
  --physics-usd output/physics/asset_physics.usda \
  --engine ovphysx --optimizer random
```

### Refine Command

Use `refine` when a single tune pass is not enough and the judge should be
allowed to rewrite the scenario for additional iterations. Each iteration runs
`tune`, asks the judge whether the result meets the threshold, and when needed
asks the configured chat backend to refine the next scenario.

```bash
physics-agent refine <SCENARIO.yaml> [OPTIONS]
```

The CLI requires both `--physics-usd` and `--user-prompt`.

| Option | Description |
|---|---|
| `--physics-usd <path>` | Required physics-authored USD to tune. |
| `--user-prompt <text>` | Required natural-language target for refinement. |
| `--reference-image <path>` | Add judge image evidence; can repeat. |
| `--reference-video <path>` | Add judge video evidence; can repeat. |
| `--reference-description <text>` | Description for a reference image; can repeat. |
| `--reference-video-description <text>` | Description for a reference video; can repeat. |
| `--no-visual-evidence` | Judge without generated/reference media. |
| `--output-dir <dir>`, `-o <dir>` | Destination for per-iteration artifacts. |
| `--engine ovphysx\|newton\|fake` | Simulation backend passed through to `tune`. |
| `--optimizer auto\|botorch\|random\|cma-es` | Optimizer passed through to `tune`. |
| `--max-trials <n>` | Tune trials per iteration. |
| `--max-iterations <n>` | Hard cap on tune/judge/refine iterations. |
| `--score-threshold <value>` | Combined score above which the judge approves. |
| `--judge-max-tokens <n>` | Override judge response length. |
| `--judge-temperature <value>` | Override judge temperature. |
| `--seed <n>` | Seed forwarded to each tune iteration. |
| `--chat-backend <name>` | Backend for scenario refinement and judge calls. |
| `--chat-model <name>` | Chat model identifier for the selected backend. |
| `--llm-timeout-seconds <seconds>` | Deadline for each judge/refine LLM call; `0` disables. |
| `--verbose`, `-v` | Enable debug logging. |
| `--log-file <path>` | Write logs to a file. |
| `--log-level <level>` | Override the default `INFO` log level. |

The public default chat backend is `gemini`, which reads `GOOGLE_API_KEY` or
`GEMINI_API_KEY`. Internal installs can route through another registered backend
with `--chat-backend` and `--chat-model`.

```bash
physics-agent refine apps/physics_agent/configs/tuning/drop_settle.yaml \
  --physics-usd output/physics/asset_physics.usda \
  --user-prompt "make it bouncy" \
  --output-dir output/refine \
  --engine ovphysx --optimizer random \
  --max-trials 4 --max-iterations 3 --score-threshold 0.7
```

### Pipeline Steps

1. `optimize_usd` - flatten, deinstance, split, or deduplicate USD through the
   configured optimizer.
2. `identify_asset` - render whole-scene previews and identify the overall
   asset type with a VLM.
3. `build_dataset_usd` - render per-prim VLM input images.
4. `build_dataset_prepare_dataset` - assemble classification specs and images.
5. `predict` - predict per-component material, component type, and physical
   properties.
6. `restore_usd` - remap predictions from optimized paths to original paths
   for reporting and artifacts.
7. `apply_physics` - author `RigidBodyAPI`, `CollisionAPI`,
   `MeshCollisionAPI`, `MassAPI`, and physics-purpose material bindings where
   appropriate.

### Common Workflows

```bash
# Run prediction only.
physics-agent run apps/physics_agent/configs/lightbulb.yaml --only predict

# Run direct prediction on an already prepared dataset.
physics-agent predict apps/physics_agent/configs/lightbulb.yaml \
  --dataset apps/physics_agent/configs/.lightbulb/dataset/dataset.jsonl

# Skip optimization for a simple, already prepared USD.
physics-agent run apps/physics_agent/configs/lightbulb.yaml --skip optimize_usd

# Resume after a failed step.
physics-agent run apps/physics_agent/configs/lightbulb.yaml --resume

# Tune a physics-authored USD.
physics-agent tune apps/physics_agent/configs/tuning/drop_settle.yaml \
  --physics-usd apps/physics_agent/configs/.lightbulb/physics/light_bulb_01_physics.usdz \
  --engine ovphysx --optimizer auto

# Run iterative tuning and scenario refinement.
physics-agent refine apps/physics_agent/configs/tuning/drop_settle.yaml \
  --physics-usd apps/physics_agent/configs/.lightbulb/physics/light_bulb_01_physics.usdz \
  --user-prompt "make it settle without bouncing" \
  --max-iterations 3
```

### Instanced USD and Instance Proxies

Enable deinstance when an asset is instanced or `apply_physics` raises a
`PhysicsAuthoringError` mentioning an instance proxy:

```yaml
steps:
  optimize_usd:
    enabled: true
    scene_optimizer_settings:
      enable_deinstance: true
      enable_split_meshes: false
      enable_deduplicate: false
    flatten_prototypes: false
  restore_usd:
    enabled: true
```

If the instanced asset also needs separate component classification from one
combined mesh, enable both `enable_deinstance: true` and
`enable_split_meshes: true`.

### Service Boundary

This skill runs the local CLI. Do not construct REST requests or manage service
containers from it. Use `physics-agent-client` for `/pipeline`, `/predict`,
`/tune`, and `/refine` requests; use `deploy-physics-agent-docker` or
`deploy-physics-agent-brev` for the matching deployment workflow. A local CLI
config may consume remote render or optimizer endpoints without becoming a
service-client or deployment workflow.

### Config Authoring

Prefer copying `apps/physics_agent/configs/lightbulb.yaml` for new configs.
Adapt only the user-specific fields:

- `project.name` and `project.session_id`
- `input.usd_path`
- `steps.optimize_usd.scene_optimizer_settings`
- `steps.predict.vlm.model`

Keep renderer, prompt, and physics authoring settings unless the user
explicitly asks to change them.

## Output Format

Report these items for every execution path:

- Command executed and whether it was full pipeline, `--only`, `--skip`,
  `--resume`, or `--dry-run`.
- Config path and session ID, when created.
- Working directory, usually `.<session_id>/` next to the config unless
  `project.working_dir` overrides it.
- Key artifacts when present:
  - `predictions/predictions.jsonl` with `id`, `classification`, reasoning,
    and optional `quality_warnings`.
  - `restored_predictions.jsonl` or restored prediction artifacts when
    `restore_usd` ran.
  - `physics/<stem>_physics.<derived extension>` when `apply_physics` ran;
    USDZ inputs default to USDA output unless a lower-level apply call used an
    explicit `.usdz` output path.
  - Optimized/deinstanced USD artifacts when `optimize_usd` ran.
  - Tune artifacts: `best_params.json`, `tune_results.json`,
    `history.jsonl`, `report.md`, and `tuned_physics.usda`.
  - Refine artifacts: `iter_<N>/` directories, `final/`, and
    `refine_summary.json`.
- Any missing credentials, service endpoints, optimizer failures, empty
  predictions, tuning/refinement failures, or instance-proxy authoring errors.

### Foreground Terminal Run

- Report the Physics Agent CLI terminal outcome (`completed`, `failed`, or
  `interrupted`) and its exit code. A started process or partial log is not
  evidence of completion.

### Durable External Handoff

- Report the handoff-creation command's terminal outcome and exit code. These
  describe only whether handoff creation succeeded, not the external Physics
  Agent job's eventual outcome.
- Report the real external job ID and, when distinct, the monitor ID, plus the
  owner, log location, exact status command, and currently observed external
  job status. Label that status as non-terminal unless the external monitor
  itself supplies terminal evidence.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| API key required | The selected VLM backend has no credential. | Set the required key locally or in `.env`; do not paste it into chat. |
| Pipeline fails midway | A step failed after writing partial artifacts. | Re-run with `--resume`; use `--clean` only when the user wants to discard prior artifacts. |
| `PhysicsAuthoringError` mentions an instance proxy | The prediction path targets a non-writable instance proxy. | Enable `optimize_usd` with deinstance and keep `restore_usd` enabled. |
| One combined mesh needs per-component physics | The asset needs splitting before rendering/prediction. | Enable `enable_split_meshes` along with deinstance when needed. |
| Relative paths resolve unexpectedly | Config paths resolve from the config file directory. | Rewrite paths relative to the config file or make them absolute. |
| Empty predictions are rejected | The pipeline produced zero records or all prediction calls failed. | Check dataset renders, VLM credentials, and `predict.allow_empty_predictions` before opting into empty outputs. |
| Mass values look suspicious | Scene scale may make inferred mass unreliable. | Review `quality_warnings` and the configured `apply_physics.mass_scale_policy`. |
| BoTorch or OvPhysX is unavailable | The `tuning` extra or isolated OvPhysX daemon environment is missing. | Install `apps/physics_agent[tuning]` and set up `WU_OVPHYSX_VENV_DIR`, or smoke-test with `--engine fake --optimizer random`. |
| Newton tuning fails at startup | The `newton` extra or a compatible GPU/runtime is missing. | Install `apps/physics_agent[newton]`; set `PA_NEWTON_DEVICE=cpu` only for CPU-capable checks. |
| `refine` fails before the first iteration | The selected chat backend has no usable key or model. | Set the backend key locally, or pass a registered `--chat-backend` and `--chat-model`. |
