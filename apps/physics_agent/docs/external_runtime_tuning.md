# External Runtime Tuning and Local Refinement

This guide defines the public contract for trusted, local bring-your-own-runtime
(BYOR) tuning and refinement. The feature ships in Physics Agent's Python API
and CLI, but it deliberately does not run through Physics Agent Service.

## Choose The Tuning Surface

| Use case | Built-in `tune` / `refine` | External `tune-external` / `refine-external` |
|---|---|---|
| Simulator | Physics Agent's OvPhysX or Newton backend | A trusted customer-owned runtime such as IsaacLab |
| Input contract | Physics USD plus a built-in scenario or prompt | Runtime config plus an adapter that returns one scalar objective |
| Parameters | Physics Agent's supported USD/backend parameters | Customer-declared numeric parameters |
| Execution | Local Python API, CLI, and Physics Agent Service | Local Python API and CLI only |
| Best fit | Tune a simulation-ready asset in a supported scenario | Tune a repository-specific robot, task, controller, or environment |

Use the built-in path when Physics Agent can construct and score the scenario.
Use BYOR when the customer runtime must own simulation setup, parameter
application, and scoring. BYOR is not a more general service endpoint.

## Ownership

Physics Agent owns qualification, optimizer proposals, candidate selection, VLM
judging, constrained local refinement, winning-recording playback, process
lifecycle, and result artifacts.
The customer adapter owns simulator setup, parameter application, task
execution, the scalar objective calculation, optional diagnostic metrics,
qualification capture, and exact rollout recording.

The external command runs locally with the calling process's operating-system
identity. It is a trusted-code boundary, not a sandbox for uploaded
repositories. Physics Agent Service and public NVCF deployments do not expose
this capability.

## Protocol

Physics Agent invokes one fresh process per trial:

```text
<python> <python_args...> <script> --request REQUEST --result RESULT <extra_args...>
```

Every request fixes the task, full parameter vector, seed, output paths, scalar
objective identity, and opaque simulator configuration. A successful adapter
result must return:

- `status: ok` and `success: true`;
- the same objective name, unit, and direction plus one finite scalar `value`;
- exact `metadata.applied_params` readback;
- artifact paths contained beneath the supplied artifacts directory.

`metrics` is optional diagnostic data. Physics Agent persists it but never uses
it to calculate the objective or choose a winner.

An optimization request has this shape. Qualification requests replace
`recording` and `publish_artifacts` with the `evidence` PNG-frame manifest
contract.

```json
{
  "schema_version": 1,
  "purpose": "optimization",
  "task": "customer_task",
  "params": {"restitution": 0.6},
  "seed": 1000,
  "output_dir": "/absolute/trial/directory",
  "artifacts_dir": "/absolute/trial/directory/artifacts",
  "trial": {"customer_runtime_config": "opaque JSON"},
  "objective": {
    "name": "bounce_height_error",
    "unit": "m",
    "direction": "minimize"
  },
  "recording": {
    "required": true,
    "artifact_name": "recording_usd",
    "media_type": "model/vnd.usd",
    "fps": 30.0,
    "camera": {
      "position": [1.5, 1.5, 1.0],
      "target": [0.0, 0.0, 0.2]
    }
  },
  "publish_artifacts": ["tuned_task_config"]
}
```

The adapter writes the result path supplied on its command line:

```json
{
  "status": "ok",
  "success": true,
  "objective": {
    "name": "bounce_height_error",
    "unit": "m",
    "direction": "minimize",
    "value": 0.03
  },
  "metrics": {"measured_bounce_height": 0.52},
  "metadata": {
    "applied_params": {"restitution": 0.6}
  },
  "artifacts": {
    "recording_usd": "/absolute/trial/directory/artifacts/recording.usd",
    "tuned_task_config": "/absolute/trial/directory/artifacts/task.json"
  }
}
```

This is the complete top-level config shape; replace the runtime paths and
adapter-owned `trial` values for the customer repository:

```yaml
schema_version: 1
task: customer_bounce_task
runtime:
  python: /absolute/customer/env/bin/python
  script: /absolute/customer/repo/physics_agent_adapter.py
  cwd: /absolute/customer/repo
  timeout_s: 900
  pass_env: [CUSTOMER_RUNTIME_TOKEN]
  fingerprint_paths:
    - /absolute/customer/repo
    - /absolute/customer/repo/physics_agent_adapter.py
  trial:
    target_bounce_height_m: 0.55
parameters:
  restitution: {min: 0.0, max: 1.0}
objective:
  name: bounce_height_error
  unit: m
  direction: minimize
  failure_penalty: 1.0e12
optimizer:
  name: botorch
  max_trials: 30
  seed: 42
  replicas: 1
  replica_seed: 1000
qualification:
  nominal_params: {restitution: 0.5}
  seed: 1000
publish_artifacts:
  - tuned_task_config
evidence:
  artifact_name: frames
  renderer: isaac_sim_kit_rtx
  media_type: application/json
  width: 960
  height: 720
  fps: 30
  min_frames: 16
  require_motion: true
  camera:
    position: [2.4, -2.4, 1.6]
    target: [0.0, 0.0, 0.45]
  recording_artifact_name: recording_usd
  playback_renderer: ovrtx
  max_duration_seconds: 4.0
  num_sensor_updates: 32
  render_mode: rt2
```

Winner rendering writes a `render_response_metadata.json` sidecar beside the
rendered frames: the backend's non-image response fields (renderer identity
included), so the frames' render provenance travels with them. Downstream
evidence publication digest-binds this sidecar alongside the frames and
requires the OVRTX renderer identity it records.

The adapter defines how the task becomes that scalar value. For example, it may
return `abs(measured_bounce_height - target_bounce_height)`. `success` reports
simulator/task feasibility. A `success: false` trial receives the configured
failure penalty and cannot be selected. Transport and contract failures are
also persisted as failed replicas.

## Evidence

External tuning requires an `evidence` config. The qualification request carries
a PNG-frame manifest contract: renderer identity, camera, resolution, frame
rate, and minimum frame count. Physics Agent decodes each declared PNG, checks
visible content and motion, and independently digest-binds the manifest and
every frame.

Every successful optimization request carries a required `recording` contract.
The adapter must persist the exact evaluated rollout as a
time-sampled USD, including every relevant body's translation and orientation.
Candidate trials do not capture qualification frames. After selection, Physics
Agent can post-render the winning fixed-seed recording directly into PNG frames
with the canonical `ovrtx` or `remote` shared renderer. This does not execute
the customer simulator or rescore the candidate. Video encoding is not part of
the winner-evidence path.

BYOR v1 requires that recording to be a self-contained, single-file USD. It
must not depend on external sublayers, references, payloads, textures, or other
files needed to open and render the rollout. Physics Agent copies the selected
recording byte-for-byte to `best_recording.usd`; it does not flatten composition
arcs or collect and rewrite dependencies. Physics Agent discovers dependencies
for validation and rejects a recording with external layers, references,
payloads, asset dependencies, or unresolved paths before that trial can become
a winner.

The qualification digest binds the exact frame-manifest bytes, PNG bytes, and
declared image metadata. Replacing any reviewed frame or the manifest blocks
the approved phase.

## Qualification

1. Fingerprint the declared runtime files, passed environment values, and the
   runtime Python environment. The Python facet records the interpreter,
   package names and versions, direct-install metadata, import paths, `.pth`
   content, and a content digest of every file in each installed distribution.
   Unreadable, unmanifested, or ordinarily missing package files fail
   qualification closed. Missing paths projected through package symlinks are
   bound into the digest as explicit missing state.
2. Run one nominal parameter trial and render its qualification evidence.
3. Publish `qualification.json`, `qualification_frames.json`, and the
   digest-bound `qualification_frames/*.png` files.
4. Stop and require the exact qualification digest from the caller.
5. Restore and revalidate that artifact before any optimization.

Approval binds:

- runtime executable, adapter, working directory, trial setup, and environment;
- task identity;
- parameter names and numeric types plus the full nominal parameter vector;
- scalar objective name, unit, and direction;
- names of adapter outputs eligible for winner publication;
- qualification seed and tolerance;
- renderer, camera, frame contract, and exact nominal result.

`qualification.json` stores the approved runtime contract beside its digest so
the approval evidence remains inspectable. Environment variable values are
hashed, not stored. Runtime configs and adapter requests must still refer to
secrets through environment variables rather than embedding secret values.
Only names listed in `runtime.pass_env` are forwarded in addition to the small
runtime allowlist. Set those variables in the local process environment; do not
place credentials in `runtime.trial`, command arguments, or checked-in configs.

Approval intentionally does not bind active parameter subsets, search bounds,
or optimizer settings. Those are search decisions, not facts established by the
nominal trial. Changing the objective or its target requires changing the
adapter/trial setup and qualifying again.

## Fixed Tuning

`tune-external` evaluates candidates with fixed replica seeds. It averages each
candidate's scalar objective across successful replicas, converts maximize
objectives to the optimizer's lower-is-better convention, and selects the best
feasible candidate. The first fixed-seed replica of that candidate is the
deterministic playback source. No selected-result or final-validation simulation
is run.

A completed run publishes:

- `run_spec.json`, containing the effective fixed values, active search bounds,
  objective, optimizer settings, replica seeds, and qualification digest;
- `history.jsonl`, with a common trial core plus BYOR replica diagnostics;
- `best_params.json`, using the standard `best_score` plus `params` schema;
- `best_recording.usd`, a byte-for-byte copy of the selected replica's exact,
  self-contained rollout;
- `external_tune_results.json`, including the selected trial, replica, seed,
  source recording, and aggregate-versus-replica objective provenance;
- `outputs/`, containing only adapter artifacts named in `publish_artifacts`;
- optional `render/*.png` direct renders of `best_recording.usd`.

Raw adapter requests, subprocess logs, and other trial internals remain beneath
`qualification/` and `optimization/`; they are diagnostic state, not public
deliverables. A declared publish artifact is required on every successful trial
and must remain inside that trial's artifact directory.

Winner rendering is opt-in for fixed tuning with `--render-winning-trial`. When
enabled, `render/*.png` contains direct renders of the selected recording. A
shared-renderer availability failure is reported in `render_error` and does not
invalidate successful optimization. A recording-integrity failure, including a
recording that changes after scoring, is terminal because the selected evidence
can no longer be trusted.

## Local Iterative Refinement

`refine-external` is a local Python/API/CLI workflow. It is intentionally not a
service endpoint. After one qualification approval, each iteration:

1. Runs the full tuning budget under the current active search.
2. Records every successful rollout as time-sampled USD.
3. Renders the winning recording directly into PNG frames without
   rerunning the simulator or encoding MP4.
4. Passes those exact PNGs to the VLM judge for comparison with the user goal and
   optional reference media.
5. If the judge continues, constrains the refiner to changing only the active
   parameter subset and search bounds. The adapter-owned objective stays fixed.

When the active parameter subset changes, parameters omitted from the next
search retain the previous iteration winner's values. The previous winner is
not rerun or inserted into the next optimization history.

`final/` is a curated terminal snapshot. It contains `run_spec.json`,
`search.json`, a portable scoring-only `history.jsonl`, `best_params.json`,
`objective.json`, `external_tune_results.json`, `judge.json`,
`refine_request.json`, `result.json`, `best_recording.usd`, `render/*.png`,
copied `reference_media/` when supplied, declared `outputs/`, and a SHA-256
`manifest.json`. Paths in the terminal judge
and request artifacts are relative to this bundle. It does not copy raw trial
directories or subprocess logs. There is no selected-result simulation replay,
media resampling, or final render rerun. It does not package undeclared USD
dependencies; it rejects the recording instead of collecting or rewriting them.

Refinement requires rendered PNG evidence for every verdict. A renderer failure
therefore fails refinement closed before the VLM judge runs. Terminal result
semantics are:

| `status` | `termination_reason` | `validated` | Meaning |
|---|---|---|---|
| `awaiting_approval` | `awaiting_approval` | `false` | Qualification passed; inspect the report and PNG frames |
| `completed` | `approved` | `true` | The VLM approved the terminal iteration |
| `completed` | `max_iterations` | `false` | A final bundle exists, but the VLM requested another iteration |
| `cancelled` | `cancelled` | `false` | The caller cancelled |
| `qualification_failed` | `qualification_failed` | `false` | The nominal qualification did not produce an approvable result |
| `failed` | `error` | `false` | No valid terminal result was published |

Changing customer code, task setup, parameter exposure, or objective calculation
remains a coding-agent workflow. Such a change alters the runtime fingerprint
and requires a new qualification before local refinement resumes.

## CLI And Python API

The first call performs qualification and stops. Review the report and
digest-bound PNG frame manifest, then repeat the same command with the exact
digest:

```bash
physics-agent tune-external runtime.yaml --output-dir output/external
physics-agent tune-external runtime.yaml --output-dir output/external \
  --approve-qualification sha256:<digest> \
  --render-winning-trial

physics-agent refine-external runtime.yaml --output-dir output/external-refine \
  --user-prompt "match the requested behavior"
physics-agent refine-external runtime.yaml --output-dir output/external-refine \
  --user-prompt "match the requested behavior" \
  --approve-qualification sha256:<digest>
```

The equivalent Python entry points are `run_external_tune()` with
`ExternalTuneInput` and `run_external_refine()` with `ExternalRefineInput`,
exported from `physics_agent.api`.

For agentic workflows, `content-workflow-cli physics
refine-external` (in `agentic/`) runs the same qualification-gated loop with a
coding agent replacing the built-in VLM judge and LLM refiner, driven by the
`content-workflow-physics-external-tuning` skill. The engine contract in this
document is unchanged by that surface. Async variants are also available. See the
[Python API guide](api.md) and the
[experimental pinned IsaacLab walkthrough](../examples/byor_isaaclab/README.md).
The walkthrough is a source-checkout example and is not installed in the
Physics Agent wheel.

## Process Safety

Each simulator execution starts a new process session. Timeout, cancellation,
or log overflow terminates the complete process group. Result, log, and
artifact byte limits are enforced, and declared artifacts may not escape the
trial artifacts directory.

## Service Boundary

`tune-external` and `refine-external` are local Python/API/CLI workflows. Physics
Agent Service exposes neither `/external-tune` nor `/external-refine`; its
`/tune` and `/refine` endpoints remain limited to the built-in Physics Agent
runtime. Customer repositories, executable paths, and adapter scripts are never
accepted by the service.
