# Migrating to Content Agents 0.6

Content Agents 0.6 uses the repository root as its single entrypoint and makes
the agentic Content Workflow the default for supported tasks. The established
application workflows under `apps/` are the **fixed pipeline**; their
CLIs, YAML configs, Python APIs, benchmarks, REST services, and deployments
remain supported when explicitly selected.

## What Changes

- Start coding agents and run setup commands from the repository root.
- Use `content-workflow-cli` and root `content-workflow-*` skills by default.
- Write new workflow artifacts under root `runs/`.
- Choose a fixed pipeline app CLI, YAML config, Python API, benchmark, or REST service
  only when that fixed pipeline interface is explicitly required.
- A missing agentic prerequisite is an error with remediation, not permission
  to fall back silently to fixed pipeline.

## Fixed-Pipeline Skill Consolidation

Before 0.6, detailed app, service-client, deployment, benchmark, and maintenance
skills were independently discoverable. In 0.6, invoke the single
`$fixed-pipeline` umbrella when one of those interfaces is explicitly required.
It selects and loads the narrowest retained reference from
`.agents/skills/fixed-pipeline/references/<name>/reference.md`; nested references
are not independently discoverable or invocable skills.

Update saved prompts and coding-agent automation to invoke `$fixed-pipeline`
and name the required runtime interface, for example `material-agent`, the
Material REST client, or the Material Docker deployment. Runtime commands,
service routes, YAML formats, Python APIs, benchmarks, and stable compatibility
flags are not renamed by this skill reorganization.

External workflow skills that intentionally use an app or REST contract should
resolve the v0.6 source of truth at
`.agents/skills/fixed-pipeline/references/<name>/reference.md`. A consumer that also
supports 0.5 may fall back to `.agents/skills/<name>/SKILL.md` only when the
nested path is absent. It must keep the selected operation explicitly labeled
as fixed pipeline, avoid scanning for similarly named skills, and fail closed
if neither exact path exists. This lets an agentic outer workflow, such as
CAD-to-SimReady, safely compose Material, Physics, and optional Texture REST
clients without accidentally selecting this repository's default agentic
workflow.

Every agentic skill available in a checkout is now discoverable from root
`.agents/skills`, so users and coding agents no longer need to enter
`agentic/`. Public release staging still excludes skills and fixed-pipeline
references that are internal-only.

## Retired WU CLI Skill Aliases

The root skill aliases `flatten-usd`, `image-gen`, `optimize`, `print-usd`,
`render-usd`, and `vision` are no longer independently discoverable in 0.6.
This removes their automatic agent routing; it does not remove the corresponding `wu` commands.
Saved prompts should use these replacements:

| Former skill alias | 0.6 route |
|---|---|
| `$flatten-usd` | Explicitly request `wu flatten-usd`; use `$content-workflow-convert-to-usd` when the actual task is format conversion. |
| `$image-gen` | Use `$image-generation` inside an agentic workflow, or explicitly request `wu image-gen` for the standalone CLI. |
| `$optimize` | Explicitly request `wu optimize`; it remains a standalone utility rather than an auto-routed content workflow. |
| `$print-usd` | Use `$usd-cli` for workflow inspection, or explicitly request `wu print-usd` for terminal output. |
| `$render-usd` | Use the owning workflow's render method, or use `$usd-cli` for an explicit low-level local or remote OVRTX render. The `wu render-usd` command remains available only when requested directly. |
| `$vision` | Use the coding agent's available image-inspection capability, or explicitly request `wu vision` for a configured VLM backend. |

External automation that needs one of the retired aliases should call the
named `wu` command explicitly and must not scan for a similarly named
replacement skill.

Bundled fixed-pipeline procedures retain audited frontmatter in
`reference.md`, but they are not skills. Codex skill discovery is based on the
exact `SKILL.md` filename; repository validation rejects that reserved filename
inside the umbrella's `references/` tree. Retained `agents/openai.yaml` files
are presentation metadata for the archived procedure definitions; without a
sibling `SKILL.md`, they are not an independent discovery or invocation entrypoint.

## Capability And Recovery Matrix

All agentic rows use the root CLI and root skill discovery tree. “Restart”
means preserve the existing run as evidence and choose a fresh `runs/<name>`;
it never means delete or reuse a partial directory.

The prerequisite column describes the selected route's tools and providers.
Agentic routes support native Linux and WSL2. Native Windows execution is
unsupported in the 0.6 release. On a Windows host, run the supported workflow
inside WSL2; Agentic rendering there uses remote OVRTX.

| Task | Default agentic CLI / skill | Selected route prerequisites | Recovery | Explicit fixed pipeline interfaces |
|---|---|---|---|---|
| Source conversion | `convert-to-usd` / `content-workflow-convert-to-usd` | Selected converter | Rerun conversion; use a new artifact directory when retaining reports | External/manual conversion tooling |
| Material authoring | `materials assign` / `content-workflow-material` | Child runner, usd-cli, and a compatible local or configured remote renderer | Safe restart; in-place material resume is not release-qualified | `material-agent` CLI/YAML/Python, Material REST, material benchmarks |
| Texture generation | One-attempt `texture run` or focused `texture prepare` through `texture publish` operations / `content-workflow-texture` | Coding-agent companion generation when selected, usd-cli, and the selected image/render providers | Resume the frozen one-attempt run after its exact companion handoff, or revalidate focused packet identity; drift requires a fresh run | `texture run --execution-mode fixed`, `texture-agent` CLI/YAML/Python, and Texture REST |
| Physics authoring | `physics apply` / `content-workflow-physics` | Child runner, usd-cli, and the selected simulation/runtime backend | Safe restart; preserve the previous trace and evidence | `physics-agent` CLI/YAML/Python, Physics REST, physics benchmarks |
| Mesh segmentation | `mesh-segmentation run` / `content-workflow-mesh-segmentation` | Child runner, usd-cli, and required render/image providers | Safe restart; completed runs can seed an explicit continuation | No equivalent fixed pipeline app |
| Validation | Agentic `validate run` with explicit runner/model and Claude execution mode, focused `validate prepare` through `review-assessment`, or provided `validate ingest-verified-operation-result` through `review-assessment` / `content-workflow-validation` | No provider runner, model, provider execution mode, or render backend is selected by default; provided mode executes neither | Use a fresh directory for coordinator, focused, or provided retries; `validate resume` is only for legacy `validate run --direct-executor` compatibility runs | `validation-agent` CLI/config/Python |
| SimReady profile work | `simready ...` / `content-workflow-simready` | Managed Foundation tooling | Rerun the failed preflight, conformance, or validation command | Agent-specific fixed pipeline validation commands |
| Large composed scenes | `scene run` / `content-workflow-large-scene` | Child runner and shared-path usd-cli | `scene resume --run-dir ...` | Scripted per-app pipelines |
| Reviewed articulation | `articulation run` / `content-workflow-articulation` | Joint and usd-cli dependencies | Review and resume while eligible; restart terminal failed/cancelled runs | `joint-agent` CLI/YAML/Python, Joint REST, joint benchmarks |
| Joint-to-validation composition | `asset run` / `content-workflow-asset` | Child runner plus all selected domain services | `asset resume --run-dir ...`; failed stages require `--recover` | Manually composed app pipelines |

`texture run` defaults to one bounded skill-routed attempt. It needs no Texture
service or separate VLM provider when the outer coordinator exposes image
generation; generated units pause for an exact coding-agent companion image
result before `texture resume`. `--runner` selects only the plan/review child.
A Claude coordinator without an image generator must start fresh with both
`--texture-backend <name>` and `--texture-agent-url <url>`. New work that needs
custom sequencing should use the focused operations selected by
`content-workflow-texture`. `texture run --execution-mode fixed` is the legacy
service-backed compatibility launcher.

Fixed pipeline commands keep their existing behavior. The name does not promise
bitwise-deterministic model output. REST is an interface choice inside the fixed
pipeline, not an alternate name for the agentic Content Workflow.

`content-workflow-cli physics apply --direct-executor` selects a lower-level
direct executor inside the agentic Physics command. It is not the fixed pipeline
`apps/physics_agent` workflow. The legacy
`--deterministic-workflow` spelling remains an alias for compatibility.

## Root Quickstart

```bash
./scripts/setup_content_agent.sh
source .venv/bin/activate
content-workflow-cli auth status
content-workflow-cli --help
```

On a Windows host, enter WSL2 and use the Linux setup commands above. The
PowerShell setup helper is not a supported 0.6 release path.

Do not `cd agentic`. That directory remains an implementation workspace; the
root skill tree exposes every agentic skill available in the checkout.

The full setup installs Node/npm SDK dependencies for workflows that launch a
Codex or Claude child agent. Direct conversion, Texture, Validation, and
SimReady users can omit those SDK dependencies:

```bash
./scripts/setup_content_agent.sh --without-child-runners
```

Run the selected command's `--help`, plus `content-workflow-cli auth status`
for child-agent workflows, before an expensive operation. Conversion and
SimReady also expose dedicated `content-workflow-cli preflight ...` commands.

### Existing Runs And Memory

The root-path change does not move or reinterpret existing data. Keep an
in-progress `agentic/runs/...` directory in place and pass that exact path to a
documented `resume` command; workflows without resume support should preserve
the old directory as evidence and start a new root `runs/<name>` directory.

Existing `agentic/runs/.memory` data is not copied automatically. To retain it,
set `WU_AGENT_MEMORY_ROOT` to that existing directory before running the 0.6
workflow. New installations default to root `runs/.memory`.

## Platform Support

Agentic Content Workflows are supported on native Linux and WSL2. Native
Windows execution is unsupported in the 0.6 release. On a Windows host, run the
supported workflow inside WSL2. Local usd-cli OVRTX rendering is supported only
on compatible native Linux NVIDIA RTX/Vulkan hosts; Agentic rendering under
WSL2 uses remote OVRTX.

Fixed pipelines are supported on native Linux and WSL2, with WSL2 rendering
limited to the `warp` backend. Native Windows fixed-pipeline execution and local
OVRTX rendering under WSL2 are not supported.

Direct REST clients may run on another operating system, but the fixed pipeline
service and rendering stack still runs on its documented Linux host. A remote
usd-cli packages and uploads the composed scene as USDZ, so the remote OVRTX
renderer does not need to resolve the workflow host's asset paths.

## Run Result Contract

Every workflow handoff must identify the selected mode, status, run directory,
canonical output, evidence/validation summary, unresolved decisions, whether
the source was mutated, and an exact resume or safe-restart command. A workflow
must not claim resume support beyond the matrix above.

## Existing Automation

Existing commands that explicitly invoke `material-agent`, `physics-agent`,
`joint-agent`, `texture-agent`, `validation-agent`, their configuration files,
or their REST services remain fixed pipeline and are not reinterpreted as agentic.
