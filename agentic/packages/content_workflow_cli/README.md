# Content Agents

`content-workflow-cli` runs agentic asset workflows against local workflow
packages and, when needed, a Content Workbench sidecar.

The CLI launches a child agent through either the TypeScript Codex SDK bridge
or the TypeScript Claude Agent SDK bridge, records Workbench API operations, and
writes trace artifacts that can be inspected after the run. The trace captures
observable API calls, renders, pixel picks, material decisions, and validation
artifacts; it is not model chain-of-thought.

## Quickstart

Workflow commands that launch child coding agents require a Linux or WSL2 host.
When using the local Workbench renderer, run them on a Linux host with an NVIDIA
GPU. macOS and native Windows users must launch these workflows on a Linux/WSL2
host, for example through a remote shell. Pointing a native client at a remote
`--workbench-url` is not sufficient: the wrapper sends host file paths to
Workbench rather than uploading local assets, so the workflow host and
Workbench host must have the same asset and material-library paths mounted.

Child execution fails closed unless its security controls are available. Codex
runs with `approval_policy="never"` and `sandbox_mode="workspace-write"`; the
wrapper does not permit unattended children to select an unconfined mode.
Claude requires the Claude Code OS sandbox. Linux and WSL2 hosts must provide
the `libseccomp` runtime, and Claude additionally requires `bubblewrap` (`bwrap`),
`socat`, and unprivileged user namespaces. See [Child Runner Security
Requirements](#child-runner-security-requirements) before starting a workflow.

1. Install the CLI and Node SDK dependencies from the repository root:

   ```bash
   ./scripts/setup_content_agent.sh
   ```

2. Authenticate one child runner:

   ```bash
   # Default Codex runner.
   content-workflow-cli auth login
   content-workflow-cli auth status

   # Or Claude runner.
   export ANTHROPIC_API_KEY=...
   ```

3. Fetch Workbench build resources for optimized inspection sessions:

   ```bash
   ./scripts/fetch_build_resources.sh
   ```

4. Convert a non-USD source asset when needed:

   ```bash
   content-workflow-cli preflight convert-to-usd path/to/source.urdf
   content-workflow-cli convert-to-usd path/to/source.urdf path/to/source.usda
   ```

   When the output path is omitted, choose `usd`, `usda`, `usdc`, or `usdz`
   with `--output-format`:

   ```bash
   content-workflow-cli convert-to-usd path/to/source.urdf --output-format usdc
   ```

5. Run SimReady Foundation preflight, staged profile conformance routing, and
   formal profile validation when you want profile evidence for a staged USD:

   ```bash
   content-workflow-cli preflight simready-foundation
   content-workflow-cli simready conform-profile path/to/asset.usda --output-dir path/to/simready-conform
   content-workflow-cli simready validate-profile path/to/asset.usda --report path/to/simready-profile.json
   ```

   Validation findings are non-blocking by default after a usable USD exists.
   Add `--strict` when a failed profile should produce a non-zero CLI result.
   Gate 3A hygiene additionally requires `--repair G3A.HYG.001` and the trusted
   pre-hygiene Joint Agent fingerprint via
   `--expected-physics-inventory-sha256 SHA256`.

6. Run material assignment on your own USD asset and reference image set:

   ```bash
   content-workflow-cli materials assign \
     --usd path/to/asset.usdc \
     --reference-image path/to/reference_front.png \
     --reference-image path/to/reference_back.png \
     --reference path/to/asset_spec.pdf \
     --additional-instructions-file path/to/material-guidance.md \
     --materials-yaml apps/material_agent/data/materials/material_libs_default/materials.yaml \
     --output-dir agentic/runs/content-workflow-cli/example-codex
   ```

   Large scenes use the same public launcher with a scene-level command:

   ```bash
   content-workflow-cli scene run \
     --usd path/to/scene.usd \
     --task material \
     --materials-yaml path/to/materials.yaml \
     --reference-dir path/to/references \
     --additional-instructions-file path/to/material-guidance.md \
     --output-dir agentic/runs/content-workflow-cli/scene-material
   ```

   Resume an interrupted run with
   `content-workflow-cli scene resume --run-dir agentic/runs/content-workflow-cli/scene-material`.

   To use Claude instead:

   ```bash
   content-workflow-cli materials assign \
     --usd path/to/asset.usdc \
     --reference-image path/to/reference_front.png \
     --reference-image path/to/reference_back.png \
     --materials-yaml apps/material_agent/data/materials/material_libs_default/materials.yaml \
     --runner claude \
     --output-dir agentic/runs/content-workflow-cli/example-claude
   ```

   For physics authoring on the public Lightbulb01 example:

   ```bash
   content-workflow-cli physics apply \
     --usd apps/physics_agent/data/examples/Lightbulb01/light_bulb_01.usda \
     --output-dir agentic/runs/content-workflow-cli/lightbulb-physics
   ```

   The physics workflow runs runtime validation by default. Add
   `--no-simulation` only when the host cannot run runtime validation and you
   only need schema-authoring artifacts.

7. Inspect the outputs:

   ```bash
   ls agentic/runs/content-workflow-cli/example-codex/final_renders
   sed -n '1,120p' agentic/runs/content-workflow-cli/example-codex/child-final.md
   content-workflow-cli trace build --run-dir agentic/runs/content-workflow-cli/example-codex
   ```

Successful runs produce `assignments.json`,
`visual_quality_assessment.json`, `api_operation_counts.json`, final PNG
renders, and trace artifacts under `trace/`.

Generic `--reference` paths are accepted in addition to `--reference-image`.
Image references are attached as images. Non-image references such as PDFs or
spec documents are persisted as `reference_files` and passed to the child agent
as local evidence paths to inspect.

Use `--additional-instructions` or `--additional-instructions-file` for one
task-wide user policy. The CLI stores the normalized text in `request.json` and
reuses it during VQA refinement without expanding it into per-prim prompts.

## Setup Notes

Use the setup script on Linux or WSL2:

```bash
./scripts/setup_content_agent.sh
```

Manual setup is equivalent to:

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e agentic/packages/content_workflow_cli
npm ci --prefix agentic/packages/content_workflow_cli
uv pip install -e agentic/packages/content_workbench
```

The default Codex runner reuses local Codex authentication. For ChatGPT/OAuth,
run `content-workflow-cli auth login` once on the host that launches
`content-workflow-cli`; no OpenAI API key is required by the wrapper.

The Claude runner has two execution modes, selected with `--claude-execution-mode`:

- `sdk` (default): launches the child agent through the Claude Agent SDK over
  Node (`claude_bridge.mjs`). This requires `ANTHROPIC_API_KEY`, or the
  provider-specific environment required by the Claude SDK, such as Bedrock,
  Vertex AI, Claude Platform on AWS, or Azure Foundry.
- `cli`: spawns the local `claude` CLI binary directly, with no Node/SDK
  dependency. It reuses whatever authentication that binary already has,
  including an OAuth session from `claude login` — useful when you want to
  use a Claude subscription instead of billing API-key usage. Set
  `CONTENT_AGENTS_CLAUDE_CLI_PATH` to point at a specific `claude` executable
  if it is not the first one on `PATH`. `--claude-max-turns` and the
  `settings` key of `--claude-config-json`/`--claude-config-file` are not
  supported in this mode (the claude CLI's print mode has no equivalents);
  `env` and `maxBudgetUsd` overrides are still applied.

`content-workflow-cli auth login` and `content-workflow-cli auth status` are
Codex-only helpers and are not used by `--runner claude` in either execution
mode. Keep Claude secrets in environment variables; run configuration JSON is
persisted in the run directory.

### Child Runner Security Requirements

All child-agent workflow commands require Linux or WSL2 and the `libseccomp`
runtime. Each fresh provider turn is owned by a Linux subreaper. A
`no_new_privs`/libseccomp control-plane guard prevents the child tree from
signaling or manipulating that reaper. The reaper waits for and kills every
adopted descendant, including detached sessions, before trusted artifact
processing resumes. Child turns fail closed on other platforms until an
equivalent descendant-ownership mechanism is available.

The control-plane guard complements, but does not replace, the provider's
native sandbox:

- Codex runs with `approval_policy="never"` and
  `sandbox_mode="workspace-write"`. Only the run directory is writable, and
  the wrapper rejects unconfined sandbox modes.
- Both Claude execution modes require the Claude Code OS sandbox and fail
  closed when it is unavailable. Install `bubblewrap` (`bwrap`) and `socat`,
  and enable unprivileged user namespaces. Only the run directory is added as
  a writable Claude workspace; external USD, material, and non-image reference
  inputs are read through sandboxed Bash.

Provider binaries remain trusted. Model-executed commands remain confined by
the native Codex or Claude sandbox. Treat additional instructions, instruction
files, Workbench exposure, and the repository checkout as trusted inputs.

## Material Library Input

Pass the YAML manifest with `--materials-yaml`. The manifest must contain a
top-level `library_path`, resolved relative to the YAML file. Pass
`--materials-usd` only when overriding the manifest's library USD path.

By default the workflow treats existing material bindings and display colors as
untrusted source setup metadata. They are redacted from material surveys and are
not used as hints unless a scoped appearance-evidence policy opts them in.
Pass `--respect-existing-material-bindings` only for preservation-first runs.

For an already-running or remote Workbench endpoint:

The endpoint must advertise the current run directory as its only permitted
output root. Missing, broader, or multiple roots are rejected; start or restart
a dedicated Workbench sidecar with the exact run root. Cross-UID restores on a
shared mount briefly use a randomized sticky staging directory beneath a
non-listable run root, then reseal it. If the CLI is killed uncatchably or the
host is lost during that short lease, remove the abandoned run directory rather
than reusing it.

```bash
content-workflow-cli materials assign \
  --usd path/to/asset.usdc \
  --reference-image path/to/reference.png \
  --materials-yaml apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --workbench-url http://workbench-host:8088
```

## Runner Configuration

Supported runner names are exactly `codex` and `claude`.

Codex accepts additional config through `--codex-config-json` or
`--codex-config-file`. Prefer provider auth helpers or environment variables
over embedding secrets in JSON, because run requests are persisted in the run
directory.

Claude accepts documented SDK option overrides through `--claude-config-json` or
`--claude-config-file`. Supported top-level keys are `env`, `maxBudgetUsd`, and
`settings`; other keys are rejected before `request.json` is written. Claude
`settings` must be an object.

Set `CONTENT_AGENTS_ALLOW_FALLBACK_SUCCESS=1` only when automation should treat
deterministic fallback artifact recovery as success after a child-agent failure.
The older `CONTENT_AGENTS_DISABLE_FALLBACK_SUCCESS=0` compatibility form is
still honored, but `CONTENT_AGENTS_ALLOW_FALLBACK_SUCCESS` takes precedence if
both variables are set.

`--vqa-refinement-max-iterations` bounds VQA review/refinement to 3 total
iterations by default, counting the initial child final review as iteration 1.
When canonical wrapper-validated artifacts still contain unresolved VQA or
final-review issues, the wrapper launches issue-local repair turns in the same
run directory and Workbench session. A repair turn should inspect the mismatch,
pick problematic pixels to identify bindable prims, patch only those prims or
their exact group, and rerender only affected views. The loop stops on success,
convergence, systematic material/picking/granularity limits, or the configured
maximum. Use `--no-vqa-refinement` to keep only the initial review.

## Trace Outputs

Every run creates:

- `request.json`
- `agent_prompt.md`
- `child-output.log`
- `child-final.md`
- `assignments.json`
- `visual_quality_assessment.json`
- `api_operation_counts.json`
- `final_summary.md`
- `trace/events.jsonl`
- `trace/operation_trace.json`
- `trace/operation_trace.md`
- `trace/run_retrospective.json`
- `trace/replay_manifest.json`

`trace/run_retrospective.json` summarizes what went well, what did not, whether
repository patches were detected, and whether the child process generated
one-off helper code during the run.

The wrapper deterministically finalizes `raw/material_decision_patch.json` into
canonical material-assignment artifacts. If VQA repair runs, the wrapper also
writes `raw/vqa_refinement_history.json` with the initial assessment, compact
repair attempts, issue signatures, convergence state, systematic give-up state,
and stop reason.
