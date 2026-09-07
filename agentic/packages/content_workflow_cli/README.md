# Content Agents

`content-workflow-cli` runs agentic asset workflows against local workflow
packages and uses `usd-cli` for supported low-level stateful USD operations.

In Content Agents 0.6 this is the default launcher for supported tasks. Invoke
it from the repository root; `agentic/` is the implementation workspace, not a
required user working directory. Default run artifacts belong under root
`runs/`.

For agent-authored workflows, the CLI launches a child agent through either the
TypeScript Codex SDK bridge or the TypeScript Claude Agent SDK bridge. For
deterministic service-backed workflows such as Texture generation, it calls the
reusable workflow package directly. Both paths write reviewable API-visible
artifacts; they do not capture model chain-of-thought.

## Quickstart

Agentic workflow commands support native Linux and WSL2. Native Windows
execution is unsupported in the 0.6 release. On a Windows host, run the
supported workflow inside WSL2; Agentic rendering there uses remote OVRTX.
Native macOS child execution is also unsupported. The workflow and its
configured `usd-cli`/OVRTX environment must be able to read the asset and
material-library paths. Unsupported native hosts are rejected by an import-safe
console prefilter; `--dry-run` remains portable.

Both `--runner codex` and `--runner claude` reject native-Windows execution
during preflight, before provider startup. Neither runner has a supported
native-Windows sandbox configuration in 0.6, and there is no unconfined
fallback.

Before launching a workflow, configure either a local OVRTX runtime or a remote
GPU service, then run the structured readiness probe. To opt into the local
one-time download (about 2.5 GB), set the flag before the daemon starts. On
native Linux:

```bash
export WU_OVRTX_AUTO_PROVISION=1
usd-cli server stop  # required if a daemon was already running
usd-cli render-probe --require-engine ovrtx
```

The first probe starts the background install and exits nonzero; rerun the exact
probe while it reports `auto-install in progress`. Content workflows wait for
those two transitional states up to their configured readiness timeout. For the
reviewed, hash-locked Geometry-only provisioner, see
[`geometry_quickstart.md`](../../docs/geometry_quickstart.md).

For a host without a supported local physics runtime, point the package-owned
`usd-cli` at the remote GPU service using the connection details supplied by
its operator, then run the structured readiness probes. On Linux/WSL2:

```bash
export USD_CLI_RENDER_RENDERER=remote
export USD_CLI_RENDER_REMOTE_URL=https://gpu-host.example.com
usd-cli server stop
usd-cli render-probe --require-engine ovrtx
content-workflow-cli preflight physics-runtime
```

The endpoint must be the `usd-cli` OVRTX protocol service. Its public `/live`
response identifies `engine=ovrtx` and `protocol_version`; `/ready` returns HTTP
200 after warm-up, and `/health` returns a JSON readiness payload. Access
control and deployment settings are operator-owned; operators running the
in-tree adapter should follow its
[deployment guide](../../../apps/usd_cli/apps/ovrtx_rendering_api/README.md).
A plain-text Prometheus `/health` response belongs to a different service and
is not compatible. Material and physics run through the local workflow
packages. The remote client uploads the packaged USD dependency closure, so the
remote service does not require the workflow host's filesystem mounts. The old
`CONTENT_AGENTS_*_BASE_URL` variables are not part of the 0.6 CAD-to-SimReady
contract.

Child execution fails closed unless its security controls are available. Codex
runs with `approval_policy="never"` and `sandbox_mode="workspace-write"`; the
wrapper does not permit unattended children to select an unconfined mode.
Claude requires the Claude Code OS sandbox. Linux and WSL2 hosts must provide
the `libseccomp` runtime, `bubblewrap` (`bwrap`), and unprivileged user
namespaces; Claude additionally requires `socat`. `content-workflow-cli auth
status` verifies that both the direct Codex CLI and Codex SDK bridge can run a
workspace-write command. See [Child Runner Security
Requirements](#child-runner-security-requirements) before starting a workflow.

1. Install the CLI and Node SDK dependencies from the repository root:

   ```bash
   ./scripts/setup_content_agent.sh
   ```

   For direct conversion, Texture, Validation, SimReady, or reviewed
   Articulation commands that do not launch a child agent, use
   `./scripts/setup_content_agent.sh --without-child-runners` to omit Node/npm
   SDK setup.

2. Authenticate one child runner:

   ```bash
   # Default Codex runner.
   content-workflow-cli auth login
   content-workflow-cli auth status --sandbox-smoke

   # Or Claude runner.
   export ANTHROPIC_API_KEY=...
   ```

3. Fetch the Scene Optimizer build resources used by optimized workflows:

   ```bash
   ./scripts/fetch_build_resources.sh
   ```

4. Convert a non-USD source asset when needed:

   ```bash
   content-workflow-cli preflight convert-to-usd path/to/source.urdf
   content-workflow-cli convert-to-usd path/to/source.urdf path/to/source.usda
   ```

   Converter execution is bounded to 120 seconds by default. Select a larger
   positive finite budget for supported large inputs with
   `--converter-timeout SECONDS`. Durable `--output-dir` runs record the exact
   value, and `--resume` rejects timeout drift.

   When the output path is omitted, choose `usd`, `usda`, `usdc`, or `usdz`
   with `--output-format`:

   ```bash
   content-workflow-cli convert-to-usd path/to/source.urdf --output-format usdc \
     --converter-timeout 600
   ```

5. Run SimReady Foundation preflight and validate the current staged USD.
   Only when that report contains failed requirements, conform from the failed
   report and revalidate the returned `output_usd_path`:

   ```bash
   content-workflow-cli preflight simready-foundation
   content-workflow-cli simready validate-profile path/to/asset.usda \
     --report path/to/prior-validation.json

   # Run only when prior-validation.json reports failed requirements.
   content-workflow-cli simready conform-profile path/to/asset.usda \
     --output-dir path/to/simready-conform \
     --validation-report path/to/prior-validation.json \
     --report path/to/simready-conformance.json

   # Read output_usd_path from simready-conformance.json.
   content-workflow-cli simready validate-profile CONFORMED_OUTPUT.usd \
     --report path/to/simready-profile.json

   # Run only when requested, using the latest statically validated asset.
   content-workflow-cli simready validate-runtime RUNTIME_ASSET.usd \
     --output-dir path/to/simready-runtime \
     --sr-specs path/to/sr_specs \
     --engines-toml path/to/engines.toml \
     --benchmark-executable path/to/benchmark-venv/bin/simready-benchmark
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
     --output-dir runs/content-workflow-cli/example-codex
   ```

   Large scenes use the same public launcher with a scene-level command:

   ```bash
   content-workflow-cli scene run \
     --usd path/to/scene.usd \
     --task material \
     --materials-yaml path/to/materials.yaml \
     --reference-dir path/to/references \
     --additional-instructions-file path/to/material-guidance.md \
     --output-dir runs/content-workflow-cli/scene-material
   ```

   Resume an interrupted run with
   `content-workflow-cli scene resume --run-dir runs/content-workflow-cli/scene-material`.

   To use Claude instead:

   ```bash
   content-workflow-cli materials assign \
     --usd path/to/asset.usdc \
     --reference-image path/to/reference_front.png \
     --reference-image path/to/reference_back.png \
     --materials-yaml apps/material_agent/data/materials/material_libs_default/materials.yaml \
     --runner claude \
     --output-dir runs/content-workflow-cli/example-claude
   ```

   For physics authoring on the public Lightbulb01 example:

   ```bash
   content-workflow-cli physics apply \
     --usd apps/physics_agent/data/examples/Lightbulb01/light_bulb_01.usda \
     --output-dir runs/content-workflow-cli/lightbulb-physics
   ```

   The physics workflow runs runtime validation by default. `--no-simulation`
   retains schema-authoring artifacts but produces a conditional, non-passing
   result; it does not satisfy required runtime or visual evidence.

   The workflow owns policy, completion criteria, and lifecycle. `usd-cli`
   exposes only the stateful scene-operation primitives it invokes.

7. From a coding-agent session with image generation, optionally verify the
   default one-attempt skill-routed Texture workflow on the shipped UV-ready
   ladder USD. It does not require a Texture service URL, generation provider,
   or separate VLM provider because the outer coordinator supplies the one
   companion image-generation attempt.

   ```bash
   ALUMINUM_PRIM=/RootNode/Geometry/M_AluminumStepLadder_B01_Aluminum

   content-workflow-cli texture run \
     --usd apps/texture_agent/data/examples/ladder/sources/usd/ladder_uv_ready.usd \
     --prompt "Generate mottled galvanized metal for the selected aluminum only; preserve everything else." \
     --prim-path "$ALUMINUM_PRIM" \
     --unit-action "$ALUMINUM_PRIM=generate" \
     --unit-appearance "$ALUMINUM_PRIM=Mottled galvanized metal with restrained industrial roughness" \
     --output-dir runs/texture-ladder
   ```

   When the accepted plan selects generation, the command writes
   `texture_companion_generation_handoff.json` and exits `3`. The outer coding
   agent performs exactly the requested image attempt and writes the bound
   result manifest at the path named by that handoff. Resume the same frozen
   run without repeating its asset, prompt, or scope:

   ```bash
   content-workflow-cli texture resume \
     --run-dir runs/texture-ladder
   ```

   `--runner` selects the plan/review child only; it does not add image
   generation to the outer coordinator. A Claude coordinator without an image
   generator must start a fresh run with both `--texture-backend <name>` and
   `--texture-agent-url <url>` so the accepted generate action uses the selected
   Texture service. The backend flag or service URL alone is insufficient.

   Use focused operations selected by `content-workflow-texture` for custom
   sequencing. Use `--execution-mode fixed` only for the legacy service-backed
   compatibility workflow.

8. Run one composed single-asset prompt when every supported domain is needed:

   ```bash
   content-workflow-cli asset run \
     --usd path/to/file-cabinet.usdz \
     --prompt "Prepare the geometry, review six drawer joints, apply painted metal with light wear, add physics, validate, and package the asset." \
     --joint-config path/to/joint.yaml \
     --materials-yaml path/to/materials.yaml \
     --materials-usd path/to/material-library.usd \
     --physics-validation-mode schema-readback \
     --output-dir runs/file-cabinet-composed
   ```

   `--materials-usd` may be omitted only when `--materials-yaml` contains a
   top-level `library_path`. The resolved library is frozen by path and digest
   before any stage runs. Runs created before that identity was recorded must
   be restarted rather than resumed with mutable material-library bytes.
   Runtime qualification remains the default. Use `schema-readback` for a
   topology-and-authoring demonstration when multi-body runtime evidence is not
   yet available; the report preserves that gap and must not claim dynamics.

   The run pauses at the Joint review gate. Continue with exact decisions:

   ```bash
   content-workflow-cli asset review \
     --run-dir runs/file-cabinet-composed \
     --decisions-json path/to/joint-decisions.json \
     --reviewer asset-owner
   ```

   Resume after interruption:

   ```bash
   content-workflow-cli asset resume \
     --run-dir runs/file-cabinet-composed
   ```

   Failed or cancelled stages require recovery:

   ```bash
   content-workflow-cli asset resume \
     --run-dir runs/file-cabinet-composed \
     --recover "reason"
   ```

   Completed stages are digest-verified
   and reused; final success requires a canonical USDZ and combined report.
   Configuration and reference files are digest-bound and rechecked on resume.
   External USD layers and resources in each accepted handoff are also bound
   and rechecked; package-local USDZ members remain covered by the package hash.
   If no external reference is supplied, the launcher freezes the prompt as
   `inputs/prompt-reference.md` so Material receives a deterministic text
   reference; resume fails if those generated bytes change.
   New runs start with deterministic Geometry preparation. The accepted output
   is a binary Z-up, meter-scale `.usdc` whose manifest, validation evidence,
   optimization status, dependencies, and requested OVRTX evidence are
   digest-bound before the Joint stage starts. Rejected Geometry never falls
   through to the original source. Existing durable runs created before the
   Geometry stage retain their original six-stage order when resumed.

9. Follow the reviewed articulation workflow
   [walkthrough](examples/articulation/README.md).

10. Inspect the outputs:

   ```bash
   ls runs/content-workflow-cli/example-codex/final_renders
   sed -n '1,120p' runs/content-workflow-cli/example-codex/child-final.md
   content-workflow-cli trace build --run-dir runs/content-workflow-cli/example-codex
   ```

Successful child-agent material runs produce `assignments.json`,
`visual_quality_assessment.json`, `api_operation_counts.json`, final PNG
renders, and trace artifacts under `trace/`.

Finalized Texture runs write their own typed plan, checkpoint, progress,
validation, and final summary under `--output-dir`. Runs that reach validation
also preserve the portable output USD and paired validation renders.

Articulation runs expose their complete artifact index with `--json`. Only
`completed` means the accepted joint IDs exactly match self-contained USDZ
readback. `needs_review` and `completed` exit with status 0;
`conditional`, `cancelled`, and `failed` exit with status 1.
`cancelled` and `failed` checkpoints are terminal diagnostic records:
preserve the run, correct the underlying problem, and use a new run directory
instead of expecting `resume` to repeat a backend phase.

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

On a Windows host, enter WSL2 and use the Linux setup command above. Native
Windows is not a supported 0.6 workflow execution path.

Manual Linux/WSL2 setup is equivalent to:

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e agentic/packages/content_workflow_cli \
  --overrides apps/usd_cli/requirements/usd-exchange-override.txt
npm ci --prefix agentic/packages/content_workflow_cli
```

Live articulation also needs a copied
`apps/joint_agent/configs/byoa_joint_rigger.yaml` configured for the model and
render endpoints available on the host. Keep provider credentials in
environment variables; do not write them into the copied YAML or run
artifacts.

The default Codex runner reuses local Codex authentication. For ChatGPT/OAuth,
run `content-workflow-cli auth login` once on the host that launches
`content-workflow-cli`; no OpenAI API key is required by the wrapper.

The Claude runner has two execution modes, selected with `--claude-execution-mode`:

- `sdk` (default): launches the child agent through the Claude Agent SDK over
  Node (`claude_bridge.mjs`). This requires `ANTHROPIC_API_KEY`, or the
  provider-specific environment required by the Claude SDK, such as Bedrock,
  Vertex AI, Claude Platform on AWS, or Azure Foundry.
- `cli`: spawns the local `claude` CLI binary directly, with no Claude SDK
  dependency. Node.js 20 or newer is still required for the parent-owned
  pre-tool command policy hook. The runner reuses whatever authentication that
  binary has, including an OAuth session from `claude login` — useful when you
  want to use a Claude subscription instead of billing API-key usage. Set
  `CONTENT_AGENTS_CLAUDE_CLI_PATH` to point at a specific `claude` executable
  if it is not the first one on `PATH`. SDK `--claude-max-turns` values must be
  greater than zero. `--claude-max-turns` and the
  `settings` key of `--claude-config-json`/`--claude-config-file` are not
  supported in this mode (the claude CLI's print mode has no equivalents);
  `env` and `maxBudgetUsd` overrides are still applied.

`content-workflow-cli auth login` and `content-workflow-cli auth status` are
Codex-only helpers and are not used by `--runner claude` in either execution
mode. Keep Claude secrets in environment variables; run configuration JSON is
persisted in the run directory.

### Child Runner Security Requirements

On Linux and WSL2, child-agent workflow commands require the `libseccomp`
runtime. Each fresh provider turn is owned by a Linux subreaper. A
`no_new_privs`/libseccomp control-plane guard prevents the child tree from
signaling or manipulating that reaper. Before provider startup, the runner waits
for the reaper to confirm its signal handlers and supervisor state, then
explicitly releases the provider. Startup failure or a readiness timeout fails
the turn before the provider exists; the readiness bound defaults to 5 seconds
and can be changed with
`CONTENT_AGENTS_SUPERVISOR_READINESS_TIMEOUT_SECONDS`. The reaper waits for and
kills every adopted descendant, including detached sessions, before trusted
artifact processing resumes. On native Windows development runs only, a
kill-on-close Windows Job Object owns each provider tree before execution
begins; startup and cleanup fail closed if that ownership cannot be
established. Native macOS child execution remains unsupported.

The control-plane guard complements, but does not replace, the provider's
native sandbox:

- Codex runs with `approval_policy="never"` and
  `sandbox_mode="workspace-write"`. Only the run directory is writable, and
  the wrapper rejects unconfined sandbox modes.
- Codex workspace-write and both Claude execution modes fail closed when their
  Linux sandbox is unavailable. Install `bubblewrap` (`bwrap`) and enable
  unprivileged user namespaces; Claude additionally requires `socat`.
  Only the run directory is added as a writable Claude workspace; external USD,
  material, and non-image reference inputs are read through sandboxed Bash.

Provider binaries remain trusted. Model-executed commands remain confined by
the native Codex or Claude sandbox. Treat additional instructions, instruction
files, `usd-cli` access, and the repository checkout as trusted inputs.

## Validation

Turn a prompt into one evidence-backed report. Validation inspects an asset and
never modifies it:

```bash
content-workflow-cli validate run \
  --usd path/to/asset.usdc \
  --task "Validate that this renders successfully and looks like the reference. Do not modify the asset; save a report with evidence and recommended actions." \
  --reference-image path/to/reference.png \
  --output-dir runs/validation-example \
  --runner codex \
  --model gpt-5.6-sol \
  --render-backend remote
```

Validation chooses no planning provider, model, provider execution mode, or
rendering backend by default. Name the runner and model explicitly; Claude also
requires `--claude-execution-mode sdk|cli`. The explicit runner authors only a
preparation-bound check-selection proposal; trusted outer code accepts and
executes the exact checks. Before launch, the parent freezes the complete
runner/model/execution-mode and output-path descriptor; acceptance rejects any
child rewrite by digest. The outer code then publishes
`validation_request.json`, `validation_plan.json`, `validation_result.json`,
`validation_evidence.json`, and `final_summary.json` into the run directory.

Exit status is `0` for pass, `1` for fail (or warn with `--fail-on-warn`), `2`
for a configuration or identity error, and `3` for a cancelled run. Use `--json`
to print the result payload instead of the summary.

Default agentic coordinator runs are not resumable. Preserve an interrupted
run as evidence and start a fresh directory. If `validate resume` is pointed at
one, it writes the digest-bound `validation_safe_restart.json` disposition,
reports `safe_restart_required`, and invokes no legacy executor or nested
agent. Resume continues in place only for a run originally started with
`validate run --direct-executor` (or the implicit composed-asset compatibility
route):

```bash
content-workflow-cli validate resume --output-dir runs/validation-example
```

Resume replays the published request, so a changed prompt, asset, or reference
fails closed rather than silently revalidating something else. An accepted
`render_valid` result is reused and only `look_right` executes.

For source custody, a bare unresolved `OmniPBR.mdl` is the one canonical public
renderer-runtime module declaration. Validation binds it as an external source
dependency without copying or injecting MDL bytes; the selected OVRTX runtime
must still resolve it. Missing custom MDL modules and package-relative forms
such as `./OmniPBR.mdl` remain unresolved and fail before child launch.

A run interrupted mid-check leaves a live claim, and resume refuses to run
beside it because it cannot tell a crashed runner from a concurrent one. Once
the previous runner is confirmed gone, release the claim explicitly with
`--recover-orphaned-claims`. Do not pass that flag while another runner may
still be working.

### Validation command families

Choose one command family and a fresh run directory:

| Command family | Commands | Contract |
| --- | --- | --- |
| Agentic coordinator (default) | `validate run` with explicit `--runner` and `--model` (plus Claude `--claude-execution-mode`), then `collect-evidence`, `assess`, `review-assessment` | One child proposes only an exact check plan; trusted outer code executes it. No provider runner, model, provider execution mode, or render backend is selected by default. |
| All-in-one execute compatibility | `validate run --direct-executor`, `validate resume` | Retains the legacy ordered-template and embedded evidence behavior; resume applies only to this path. |
| Focused execute | `validate prepare`, `check`, `finalize`, `collect-evidence`, `assess`, `review-assessment` | Executes only explicitly outer-selected checks and preserves unselected states. |
| Provided | `validate ingest-verified-operation-result`, `collect-evidence`, `assess`, `review-assessment` | Re-verifies an existing domain result and executes no check, renderer, simulator, judge, or provider. |

The separate `validation-agent` CLI remains the fixed-pipeline config/Python
surface. It is not an automatic fallback for any command family above.

Use `validate prepare`, `check`, `finalize`, `collect-evidence`, `assess`, and
`review-assessment` when one outer Codex or Claude reasoner explicitly chooses
the focused checks and owns semantic/visual judgment. Deterministic Validation
only expands that frozen request, executes one selected check per call, binds
evidence, and finalizes receipts. Unselected checks remain `not_requested` or
`not_evaluated`; optional `look_right` critique is never canonical authority.

When a domain-owned verifier/projector already produced a native digest-bound
result, follow the
[verified-operation ingress protocol](../content_agent_workflows/docs/verified_operation_ingress.md#explicit-modes).

Ingress re-reads the exact source, output, dependencies, report/payload,
artifacts, and component identities. It never calls a provider, renderer,
simulator, judge, or domain workflow. Execute and provided artifacts cannot be
mixed, path-only evidence is not canonical, and each imported native operation
retains its own terminal disposition.

Canonical post-mutation visual evidence is a separate public leaf. Follow the
[canonical post-mutation visual-evidence method](../content_agent_workflows/docs/verified_operation_ingress.md#canonical-post-mutation-visual-evidence),
which requires distinct source and post-mutation USD bindings.

It uses the package-owned usd-cli with an explicit local or remote OVRTX backend
and binds the exact USD dependency closure, image bytes, render responses,
camera records, usd-cli journal/checkpoint/source revision, render metadata,
and tool/backend identity. A completed render is evidence production, not
semantic visual acceptance; the outer organizer reviews the exact digests. See
`../content_agent_workflows/docs/verified_operation_ingress.md`.

## Material Library Input

Pass the YAML manifest with `--materials-yaml`. The manifest must contain a
top-level `library_path`, resolved relative to the YAML file. Pass
`--materials-usd` only when overriding the manifest's library USD path.

By default the workflow treats existing material bindings and display colors as
untrusted source setup metadata. They are redacted from material surveys and are
not used as hints unless a scoped appearance-evidence policy opts them in.
Pass `--respect-existing-material-bindings` only for preservation-first runs.

For remote rendering, configure the workflow's OVRTX environment and use paths
that the workflow host can read. The workflow, rather than `usd-cli`, owns the
run directory, policy, and recovery behavior.

## Texture Generation

The canonical agentic Texture surface consists of independently selected,
typed operations:

```text
texture prepare -> [propose] -> (generate | apply-provided) -> evidence -> [critique] -> review -> publish
```

One outer Codex or Claude reasoner chooses each required operation directly;
the CLI does not select or chain the next operation. `prepare` inspects the
frozen USD, dependency closure, explicit scope, material/UV facts, references,
and initial OVRTX images without constructing a Texture service or VLM.
`propose` and `critique` are optional explicit provider leaves. For candidate
mutation, the outer reasoner chooses exactly one explicit mode: `generate` with
a concrete named provider adapter, or `apply-provided` with one ordered,
digest-bound outer-generated albedo PNG per prepared unit. `apply-provided`
performs deterministic USD authoring and does not call a generator or provider.
References remain references; neither mode falls back to the other. Typed
Python generator adapters are replaceable, while the CLI exposes only concrete
named adapters. `evidence` performs static checks and captures matched OVRTX
images without semantic scoring. `review` records the outer multimodal decision,
and `publish` rechecks all identities, including supplied-image bytes and
producer provenance, plus saved-stage readback before copying the exact accepted
candidate.

The initial `texture prepare` call freezes every operation as `requested` or
`not_requested`; selected operations not yet called are `not_evaluated`.
Neither non-success state is treated as passing. Every provider-facing focused
command requires explicit provider configuration; there is no implicit NIM,
Gemma, NVCF, or Texture-service choice. See the public
`content-workflow-texture` capability matrix for command ownership and output
contracts.

### Skill-routed one-attempt Texture workflow

`content-workflow-cli texture run` defaults to `--execution-mode skill-routed`.
It performs deterministic provider-neutral preparation, launches one plan-only
Codex child by default (or Claude with `--runner claude`), freezes the exact
plan, and executes only selected actions. A Texture service URL and `--vlm-*`
provider settings are not required in this mode. The separate multimodal review
is another coding-agent child over exact current-run OVRTX images; its policy
identity records the actual runner and model rather than an unused VLM provider.

When the accepted plan selects `coding_agent_companion` generation, the command
writes `texture_companion_generation_handoff.json` and exits `3` with
`awaiting_companion_generation`. The outer coding agent performs exactly one
image-generation attempt for each listed unit and writes the bound result
manifest at the path named by that handoff. Then continue the frozen run:

```bash
content-workflow-cli texture resume --run-dir runs/texture-ladder
```

`--runner` controls only the plan and review children. It does not supply the
outer companion image generator. A Claude coordinator without image generation
must start a fresh run with both `--texture-backend <name>` and
`--texture-agent-url <url>`; the backend flag or service URL alone does not
select a usable service generation path.

Resume revalidates the frozen request, preparation, plan, companion result, and
all terminal nested artifact bytes before applying or replaying anything. It
does not fall back to a Texture service or another generation provider.

The standalone skill-routed Texture executor currently owns one generated
candidate and one separate reviewed decision. Its `--max-vqa-iterations` value
therefore defaults to `0`, and an explicit nonzero value fails before the
expensive child, provider, or OVRTX phases instead of implying an unavailable
refinement retry. Embedded Asset Texture execution and the explicit fixed
compatibility workflow retain their bounded refinement budget and default to
`2`.

### Legacy Agentic compatibility workflow

`content-workflow-cli texture run --execution-mode fixed` launches the legacy
Agentic bounded workflow for an existing USD through the reusable
`content_agent_workflows.texture` package. It connects directly to Texture
Agent and invokes `usd-cli` only for supported low-level scene operations; it
does not launch a child coding agent or copy workflow policy into the wrapper.

A fresh run requires `--usd`, `--prompt`, `--output-dir`, and exactly one scope
type: one or more `--material-path` values, or one or more `--prim-path`
values. Do not combine the two types. The output path must not already exist.
The immutable plan resolves a material scope to member geometry, constraining
UV preparation and later mutations to selected members.

Also provide `--texture-agent-url`, `--vlm-backend`, and `--vlm-model`, or set
their corresponding `CONTENT_TEXTURE_*` configuration. The legacy launcher
does not silently choose a Texture service, NIM, Gemma, NVCF, or another
semantic provider.

Configure paired source/output visual validation with `--vlm-backend` and
`--vlm-model`. `--vlm-base-url` selects a compatible custom endpoint, while
`--vlm-api-key-env` names the environment variable holding the credential; the
secret value is not persisted. Supported public backend names are `nim`,
`openai`, `anthropic`, and `gemini`. Custom Anthropic and Gemini base URLs
require an explicit endpoint-scoped `--vlm-api-key-env`; provider-default
hosted credentials are not forwarded.

Optional `--texture-backend`, `--texture-endpoint`, and `--backend-engine`
values are forwarded as typed Texture Agent planning hints.
`--max-vqa-iterations` bounds targeted validation/refinement. Each workflow
run owns one confined `usd-cli` session for low-level scene operations.

Use `--texture-agent-token-env` to name an environment variable containing an
optional Texture Agent bearer token. `--texture-timeout` bounds service
requests, `--texture-poll-interval` controls job polling, and
`--scene-tool-timeout` bounds `usd-cli` operations. Credential values remain
runtime-only.

`texture resume --run-dir` reconstructs the original asset, prompt, explicit
scope, and workflow limits from the durable `request.json` and checkpoint. It
must use the same VLM backend, model, and base URL because they define
validation-policy identity. Texture Agent connection settings,
credential environment-variable names and token values, timeouts, and poll
intervals remain runtime-only and may be supplied again. Resume fails closed if
durable identity or accepted artifacts changed.

During setup, an interrupt aborts immediately. It preserves an existing resume
checkpoint but does not create resumable state before one exists. Once
workflow execution is active and the checkpoint exists, the first interrupt
requests cooperative cancellation and a second force-aborts while preserving
the latest checkpoint.

Only a final Texture `pass` exits 0. A `conditional` result exits 3, and a
cooperative cancellation exits 130. Add `--json` to either Texture command for
the typed machine-readable result.

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
run directory and `usd-cli` session. A repair turn should inspect the mismatch,
pick problematic pixels to identify bindable prims, patch only those prims or
their exact group, and rerender only affected views. The loop stops on success,
convergence, systematic material/picking/granularity limits, or the configured
maximum. Use `--no-vqa-refinement` to keep only the initial review.

## Child-Agent Material Trace Outputs

Child-agent material workflows create:

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
