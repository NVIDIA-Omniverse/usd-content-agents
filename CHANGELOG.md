# Content Agents 0.5.0 (17 Jul 2026)

Content Agents 0.5 adds the new Joint Agent Research Preview and expands the
suite with asset-specific material creation, REST-accessible Physics
refinement, reference-conditioned Texture editing, stronger Validation, and
adaptive workflow review and repair.

**At a glance**

**Material Agent — Beta**

- Creates new, asset-specific textured materials from reference images and
  optional text guidance, then assigns them through a temporary USD library.
- Adds selectable SimReady catalogs with 265 curated or 1,529 currently usable
  materials, plus clearer per-part assignment evidence.
- Generated-material mode is single-asset and does not include full native
  OpenPBR/MaterialX authoring.

**Physics Agent — Beta**

- Exposes iterative simulation refinement through REST: run trials, compare
  against text or visual goals, revise, and repeat.
- Returns the tuned USD, a recording when produced, and evidence, with improved
  friction handling, failure reporting, and ARM64 support.

**Joint Agent — Research Preview**

- Adds revolute joints for rotating parts and prismatic joints for sliding
  parts, including accepted connections, axes, and limits. Aggregate or
  multi-root assets also receive exact link membership and articulation roots.
- All 17 qualification packages passed the selected static SimReady checks,
  covering 78 joints and 119/119 Foundation results.
- Adds articulation structure but does not create missing rigid bodies, masses,
  colliders, or drives, or prove dynamic behavior.

**Texture Agent — Research Preview**

- Adds reference-conditioned, UV-aware editing for selected materials or parts
  through an operator-provided Texture Variation service.
- Supports bounded planning and targeted retries while preserving accepted
  textures and modeled surface detail.
- The default remains text-to-texture generation; a managed Step1X runtime is
  not included.

**Validation Agent — Research Preview**

- Adds direct local OVRTX rendering and more consistent, fail-closed execution
  across supported render and model providers.
- Continues to validate rendering, visual appearance, authored physics, and
  behavior evidence through CLI and Python.

**Agentic Workflow — Research Preview**

- Uses a persistent Workbench session to inspect intermediate renders, correct
  fine part-level or cross-view inconsistencies, and review Material results
  again, with up to three passes by default.
- Supports durable decomposition, per-asset processing, collection, and
  recovery for larger Material scenes.
- Does not yet provide agentic Texture or Joint workflows or orchestration
  across all five agents.

**Agent details**

**Material Agent — Beta**

Material Agent can now create new, asset-specific textured materials from
reference images, with optional text guidance. It generates the required
texture maps, packages them into a temporary USD material library, and uses the
new materials during assignment.

Users can also select PhysicalAI SimReady material libraries without
downloading the entire collection. The release offers a curated catalog of 265
indexed materials and a full catalog with 1,529 currently usable entries.

More detailed evidence shows which parts were prepared, predicted, assigned a
fallback, successfully bound, or left unbound. Approved assignments are also
preserved when an asset moves from Material processing into Physics processing.

Technical specification text and converted PDF pages are now kept outside the
visual model input. Material Agent selects the image-based material label first;
extracted specification claims can then corroborate that label or flag a
conflict for review, but cannot introduce or replace the visual result.

Generated-material mode is opt-in, limited to single assets, and requires
reference imagery and a configured image-generation backend. Full native
OpenPBR/MaterialX authoring is not included in the default release path.

**Physics Agent — Beta**

The simulation-tuning workflow is now available as a complete iterative
refinement loop through REST. Applications can run simulation trials, compare
the results with text, image, or video goals, revise the simulation scenario,
and repeat until the result is accepted or the configured limit is reached.
They can then retrieve the tuned USD, any produced recording, and supporting
visual evidence.

The release also improves reference and generated-frame controls, physical
friction constraints, visual judging, durable failure reporting, and ARM64
support for Physics/OvPhysX tuning.

**Joint Agent — Research Preview**

Joint Agent adds articulation structure to prepared multipart USD assets. It
identifies possible supporting and moving parts and proposes revolute joints
for rotating parts such as doors, lids, wheels, and knobs, or prismatic joints
for sliding parts such as drawers, rails, and telescoping components.

After review, Joint Agent can author the accepted connections, axes, and
movement limits into a self-contained USDZ. For assets requiring aggregate
links or multiple articulation roots, it also authors their exact rigid-link
membership and articulation roots. Package readback verifies that the
published asset contains the approved joint graph. Joint Agent does not create
missing rigid bodies, masses, colliders, drives, or joint states.

For the 17 prepared assets used to qualify this release, all 17 exact packages
passed the selected static Isaac Sim Asset Validator and SimReady Foundation
profiles. The packages contained 78 authored joints, all 119 pinned Foundation
feature checks passed, and no published package differed from its accepted
joint graph.

These results demonstrate static package and SimReady-profile conformance for
the qualified assets. They do not measure inference accuracy on arbitrary
assets or prove dynamically correct motion, contact, travel limits, or
long-term simulation stability.

**Texture Agent — Research Preview**

Building on its text-to-texture and UV-preparation capabilities, Texture Agent
now supports reference-conditioned, UV-aware editing through an
operator-provided Texture Variation API. Users can target specific materials
or parts, preview a bounded processing plan, and retry individual unsuccessful
texture units while retaining accepted results.

Stricter targeting and a surface-only policy help preserve modeled detail in
CAD, PCB, and industrial assets. Downloaded USDZ packages also retain their
active UV-addressable PBR texture graphs.

The release provides the external-service adapter and API contract. A managed
Step1X runtime, model checkpoints, and setup package are not included; the
default path remains lightweight text-to-texture generation.

**Validation Agent — Research Preview**

Validation Agent checks render validity, visual appearance, authored physics,
and physical-behavior evidence. Version 0.5 adds direct local OVRTX rendering,
consistent rendering-backend selection, normalized results across supported
model providers, and clear failure when a requested capability is unavailable.

Validation remains available through CLI and Python as a reproducible
release-gating tool.

**Agentic Workflow — Research Preview**

Fixed pipelines remain the reproducible default. The Agentic Workflow research
preview adds an adaptive mode that can inspect scene structure, intermediate
renders, Validation findings, and previous attempts before deciding what to do
next.

For Material workflows, it can compare rendered results with references,
identify the affected parts, correct only those assignments, rerender the
relevant views, and review the result again. Single-asset workflows support up
to three review and targeted-repair passes by default.

Because one session retains earlier decisions while inspecting each new render,
it can refine fine part colors, logos, and symmetric assignments that a single
fixed pass may miss.

For larger Material scenes, the workflow manages decomposition, per-asset
processing, and collection back onto the original topology. Durable state
allows interrupted or partially failed work to revisit the affected phase
without discarding unrelated completed results.

The preview also supports source conversion, Physics authoring with runtime
Validation, and selected SimReady checks. It does not yet provide agentic
Texture or Joint workflows or general orchestration across all five agents.

**Platform improvements**

Content Agents 0.5 also strengthens provider selection, credential and artifact
isolation, cancellation correctness, service health and capacity reporting,
configuration diagnostics, container security, and ARM64 support for
Physics/OvPhysX and OVRTX.

## Added

- Added opt-in Material Agent generation of asset-specific textured USD
  material libraries from reference images and optional text guidance, with
  fail-closed backend preflight and single-asset scope.
- Added selectable PhysicalAI SimReady material catalogs with lazy category
  hydration, including curated and full-catalog options.
- Added a Texture Variation service integration for bounded,
  reference-conditioned, UV-aware editing of explicitly selected materials and
  prims, with targeted regeneration and generated-artifact manifests.
- Added the Physics Agent iterative refine REST workflow with progress events,
  cancellation, artifact downloads, visual evidence, and configurable
  reference and generated-frame sampling.
- Added direct local OVRTX rendering for Validation Agent workflows alongside
  the existing remote-render path.
- Added the Agentic Workflow Research Preview for adaptive visual review, targeted
  Material repair, durable large-scene processing, source conversion, Physics
  runtime Validation, and selected SimReady checks.
- Added an owned structured Joint Rigger core that authors articulation
  topology, source-backed limits, diagnostics, and immutable result records
  from Joint Agent Stage 2 inputs.
- Added self-contained USDZ publication with dependency-closure checks,
  package readback, atomic replacement, and provenance identities.
- Added a release-gate workflow for isolated authoring, pinned Isaac Sim Asset
  Validator and SimReady Foundation execution, evidence sealing, publication,
  and terminal machine attestation.
- Added a public Joint Agent validation skill for optional Gate 3A Isaac Sim
  Asset Validator and Gate 3B SimReady Foundation checks on USD/USDZ output.
- Exposed the built-in Joint Rigger through the opt-in Joint Agent service, with
  self-contained USDZ downloads through local and S3 artifact paths.
- Added evidence-backed static conformance plans for stage metadata, mesh
  topology, rigid-body hygiene, physics materials, collider approximation,
  grasp evidence, and collision filtering.

## Changed

- Agentic Workflow child execution now requires Linux or WSL2 and fails closed
  unless per-run sandboxing and descendant supervision are available. Codex
  uses `workspace-write`; Claude additionally requires `bubblewrap`, `socat`,
  and unprivileged user namespaces.
- Material assignment now reports prim-level preparation, prediction, fallback,
  binding, and unbound evidence. Strict policy fails closed unless every target
  is release-ready, while partial policy preserves diagnostics for inspection.
- Generated Material workflows validate conditioning provenance, supported PBR
  profiles, reference-image memory bounds, texture packages, and backend
  preflight evidence.
- Texture workflows now provide clearer planning contracts, safer request
  concurrency, scoped UV projection, stronger large-input guards, targeted
  regeneration, and more reliable generated-texture packaging.
- Physics tune/refine workflows use consistent binary USD artifacts, stronger
  topology and scene construction, clearer freeform scoring, and improved
  generated visual evidence.
- Pipeline status, regeneration, artifact publication, and progress events now
  use stronger ownership and freshness checks to prevent stale workers or
  artifacts from becoming authoritative.
- Rendering paths have more robust bounded retries, material-target handling,
  and generated-texture visibility, while persistent OVRTX daemons have bounded
  render-count and memory-resource limits.
- Public source-release evidence now covers every shipped Python and npm project,
  wheel-declared and legacy bundled legal files, and integrity-pinned
  platform-specific npm dependencies. Production containers and dynamically
  bootstrapped runtimes remain subject to their separate SBOM and compliance
  gates.
- Unified Texture, Validation, and `wu render-usd` backend selectors around a
  shared canonical registry with explicit capability subsets, fail-closed
  unknown-versus-unavailable handling, and structured Validation results. The
  typed `render.backend` selector is now authoritative, and conflicting legacy
  `policy.render_backend` values fail instead of silently taking precedence.
- Enabled CLI/YAML Joint Rigger configurations now require an explicit adapter;
  the REST service retains its separate built-in omission default.
- Pipeline creation and regeneration now return a run-generation token, and
  cancellation requires that exact token so delayed requests cannot cancel a
  successor run in the same session.
- Connected the public NIM, OpenAI, Anthropic, and Gemini model registries to
  runtime selection so documented provider choices resolve consistently.
- Made service health and session-capacity reporting reflect the same limits
  enforced when work is admitted, including degraded and saturated states.
- Added versioned, machine-validated Material and Physics skill evaluation
  suites with portable harness metadata and deterministic fixtures.
- Added architecture-specific ARM64 runtime locks and Docker selection for
  Physics tuning and isolated OvPhysX execution while preserving the reviewed
  x86_64 dependency paths and reporting runtime provisioning through health.
- Made the public Physics Agent CLI and service-client skills safe for headless
  execution and aligned their generated requests with the supported service
  contracts.
- Made the public Content Workflow demo self-contained with shipped inputs and
  durable output paths, and resolved Workbench endpoints from each frozen run
  request instead of hardcoded local ports.
- Standardized agent CLI YAML ingestion, validated step filters before pipeline
  work, and aligned public help and examples with configurations that ship in
  the release.
- Reported ignored configuration typos in closed nested behavior mappings with
  safe dotted-path suggestions while leaving provider and backend extension
  dictionaries open.
- Added Newton 1.4 Warp renderer compatibility while preserving the existing
  renderer contract for supported USD render paths.
- Made Physics tuning enforce ordered physical friction bounds during
  optimization and scenario validation.
- Recognized Joint Agent structure-LLM configuration aliases in CLI and unified
  config validation so supported public model settings are accepted
  consistently.
- Bounded Material CLI checkout discovery, clarified local OpenAI-compatible
  output-token configuration, and expanded Material client/API compatibility
  for optional email fallback, clustering controls, and initial `layer_only`
  requests.
- Consolidated structured JSON normalization used by Validation workflows so
  fenced, provider-wrapped, and directly parsed results follow one contract.
- Joint Rigger source snapshots now scale to large referenced assets without a
  fixed package-size cutoff while preserving bounded resource diagnostics.
- Joint Agent prop inference now completes built-in role labels and reconciles
  link directions with authored joint axes before publication.
- The offline release-readiness corpus now exercises 17 articulated-prop
  references with one artificial structured input contract per asset.
- The frozen candidate passes Gate 3A for 17/17 packages and Gate 3B for 17/17
  packages, including all 119 pinned Foundation feature results and 78 authored
  joints with no accepted-graph mismatch.
- Expanded the Agentic Workflow research preview's SimReady conformance and repair
  handling for imported source assets, stage metrics, collider ownership,
  material bindings, packaged inputs, and explicit grasp evidence.
- Made Scene Optimizer selection task-driven and preserved accepted material
  assignments through the material-to-physics handoff.

## Fixed

- Kept geometrically different meshes from the same source file in separate
  large-scene instance groups so collection no longer projects one topology
  onto incompatible members.
- Made Agentic Workflow authentication status verify that the stored Codex
  login can invoke a model instead of reporting credential presence alone.
- Preserved usable materialized USD output with structured unbound-prim
  warnings when partial coverage remains, and recorded durable USD success or
  failure in assignment, summary, and trace artifacts to prevent false-success
  reports.
- Confined Agentic Workflow child agents, Workbench outputs, and run artifacts
  to each run directory. Runs now reject tampered or out-of-scope paths,
  hostile configuration, unsafe files, name-swap races, and surviving child
  processes before trusted publication or recovery.
- Made required physical-behavior validation fail when supplied evidence
  explicitly rejects the behavior, while optional profiles continue to report
  a warning.
- Fixed Material Agent prompt templates containing literal braces, finalizer
  path aliases and source fanout, missing binding diagnostics, and terminal
  result/event ordering.
- Fixed large Material Agent prim inputs exceeding the credential scanner's
  aggregate traversal limit by scanning each prim record before accumulation,
  while preserving credential rejection for the prepared dataset.
- Fixed Texture Agent provider forwarding, image-response extraction, CAD-scale
  input guards, projection scoping, and generated-asset localization.
- Fixed Physics refine result persistence, tuning dependency availability,
  visual sampling, progress reporting, and artifact links so results advertise
  only files that were produced and published.
- Fixed rendering failures involving missing AOV output, nested texture graphs,
  stage-relative package members, collision-safe texture mirroring, and finite
  render-slot waits.
- Fixed workflow package boundaries, dependency findings in LangChain Core,
  LangSmith, and the Anthropic integration, and public release metadata drift.
- Upgraded the isolated OvPhysX runtime to remove its vulnerable bundled Python
  interpreter and upgraded the locked CPU-only Physics optimization runtime to
  Torch 2.12.1 while preserving tuning behavior.
- Hardened Material Agent material-library prompt boundaries, removed untrusted
  USD path text from self-evaluation prompts, and isolated retrieved
  specification text and converted PDF pages from visual inference. Extracted
  material claims now only corroborate the selected visual label or flag a
  review-required conflict; they cannot introduce or replace that label.
- Made client-controlled S3 inputs fail closed unless the requested source is
  authorized for the owning service session.
- Capped OpenAI-family VLM output tokens consistently across supported runtime
  adapters to prevent provider-limit failures.
- Kept Material, Physics, and Joint runtime credentials in isolated in-memory
  configuration, rejected live secrets at durable artifact and diagnostic
  boundaries, and excluded legacy pipeline handoff paths from storage surfaces.
- Added selected-step credential preflight and normalized provider
  authentication failures to value-free diagnostics.
- Confined local artifact, session, and checkpoint operations to
  descriptor-validated roots with locked atomic rewrites, and hardened
  configuration, optimizer, and service-recovery failure handling.
- Parsed numeric environment overrides with bounded, value-free fallback
  diagnostics so malformed deployment settings cannot abort agent startup.
- Published best-effort failed Texture Agent manifests after partial pipeline
  failures without masking the original error or exposing provider details.
- Restored deterministic Material Agent simulation against the JSON material
  prompt contract while preserving legacy mock prompt compatibility.
- Kept prediction-optional Joint Rigger defaults path-free and made public
  pipeline and result string representations secret-safe without changing their
  programmatic fields.
- Published the Texture Agent service's first progress heartbeat without waiting
  for downstream subscribers, avoiding false startup stalls.
- Rejected wildcard CORS origins when credentials are enabled across public
  agent services.
- Sanitized internal paths, hostnames, and storage descriptors from public
  service responses while preserving structured diagnostics.
- Honored the configured VLM endpoint during Physics judge visual evidence
  generation instead of falling back to the default endpoint.
- Persisted terminal Physics pipeline execution failures so failed service runs
  retain durable status, events, and artifact metadata.
- Stabilized public third-party notice generation with an artifact-hashed
  full-environment lock, retaining the reviewed `pypdfium2` 5.12.0 release
  while preventing transitive resolver drift.
- Fixed Texture material-preview and final-output rendering to honor the
  configured backend, and fixed remote `wu render-usd --all-cameras` runs to
  enumerate every authored camera.
- Moved the public Material, Physics, Texture, and OVRTX service images to
  Ubuntu 24.04 distro Python in isolated virtual environments, refreshed
  inherited OS packages, kept pip above the image-scan floor, and removed build
  caches to clear container-scan findings. The shipped Step1X adapter image now
  enforces the same pip scan floor while retaining its operator-mounted runtime
  cache; its managed runtime remains outside the public 0.5 scope.
- Raised Pillow to 12.3.0 in the root and isolated OVRTX runtime locks to clear
  the current affected advisory ranges.
- Raised the LangChain Core dependency floor and root lock to 1.4.9 so fresh
  and lowest-resolution installs remain above the reviewed affected range.
- Made managed OVRTX cache reuse depend on the complete immutable runtime-lock
  identity and fail closed on partial or stale managed environments.
- Aligned Joint Agent service runtime and API version reporting at 0.5.0,
  hardened non-root container permissions, and prevented stale regenerated
  Joint Rigger artifacts from being published.
- Limited the 0.5 built-in service path to revolute and prismatic candidates.
  Zero-ready candidate sets complete without a package, mixed sets publish the
  ready subset with deferred-candidate warnings, and unsupported spherical
  input fails closed.
- Preserved source stage metadata after Scene Optimizer runs, restarted stale
  Workbench sessions between assets, and retried isolated OVRTX rendering after
  native child-process failures.
- Added ARM64 OVRTX runtime resolution alongside x86_64 and synchronized the
  third-party notice integrity counts for both platform artifacts.
- Replaced the Texture image-generation health probe's curl dependency with a
  Python standard-library readiness check.
- Fixed Texture Agent prompt-limit failures in simple image generation and
  preserved active, UV-addressable generated PBR texture graphs in downloaded
  USDZ packages, including the default apply-only CLI and regeneration flows.
- Aligned public agent and service documentation with the 0.5 scope and version,
  repaired broken local links, and completed the agent-skill matrices and app
  index.

# Content Agents 0.4.3 (30 May 2026)

Public Content Agents release notes.

This update follows the v0.4.2 public release and covers the delta since
v0.4.2.

## Changed

- Scene Optimizer Core build resources now come from the public
  `NVIDIA-Omniverse/usd-optimize` GitHub release `v1.0.3`.
- Release notice dependency resolution now follows the pinned release
  `pyproject.toml` metadata so notice validation stays aligned with the
  shipped dependency set.
- Release metadata was bumped to `0.4.3` across package, service, Helm, and
  public changelog surfaces.
- Scene Optimizer build-resource tests now cover the public GitHub release URL
  shape for both supported Linux bundle platforms.

## Fixed

- Material and Physics service containers can now run local Scene Optimizer
  from the copied bundle as non-root service users.
- Public release skill hygiene metadata was corrected so the public skill gate
  passes for the published skill set.

# Content Agents 0.4.2 (27 May 2026)

Public Content Agents release notes.

This update follows the v0.3.10 public release and covers the delta from
v0.3.10 to v0.4.2.

## Added

- Added Validation Agent.
- Added Physics Agent tuning and refinement workflows, including service APIs.
- Expanded Material Agent scene and material pipeline support.
- Updated OVRTX rendering runtime and service integration.
- Added Texture Agent Service shared/S3 storage and multi-instance support.
- Reorganized agent skills into `.agents/skills`, with `.claude/skills` and
  `.codex/skills` symlinked to the shared skill tree.
- Added deployment collection and Brev deployment assets.
- Refreshed docs, changelog, third-party notices, and tests.

# Content Agents 0.3.10 (30 Apr 2026)

Bug-fix release addressing issues found after the 0.3.9 public release:
public Quick Start regressions, Scene Optimizer subprocess setup,
texture-agent portability and error reporting, texture-service API
consistency, and release packaging/docs polish.

## Added

- KUKA arm row added to the SimReady teaser GIF grid in `README.md`
  and `README_PUBLIC.md`, showing the asset progressing from gray input
  through Material Agent, Texture Agent (rusty), and Physics Agent drop
  simulation.
- `texture-agent` service now exposes a configurable
  `failure_threshold` for image generation and blend. Per-material
  failures now propagate as structured records on SSE events,
  `/pipeline/{id}/status`, `/pipeline/{id}/results`, `/event-log`, and
  persisted session metadata.
- `texture-agent run` supports `--resume` and `--session-id`, and the
  `apply` command can reuse generated artifacts from the configured
  working directory.
- Codex-compatible skills were added alongside the existing Claude
  workflows for common Content Agents tasks, including agent CLIs,
  service clients, deployment, USD utilities, and review helpers.

## Changed

- `material-agent`, `texture-agent`, `physics-agent`, and `joint-agent`
  load `.env` earlier during package import so CLI runs consistently
  honor repo-local environment configuration.
- README and public documentation now scope texture-agent CLI
  capabilities to implemented controls and document the staged
  `discover`, `generate`, and `apply` commands.
- README teaser media under `assets/images/**` is now whitelisted for
  public release packaging.
- OVRTX rendering and material-agent scene helper paths were hardened
  with broader regression coverage.

## Fixed

- Scene Optimizer subprocesses now find the correct Python/libpython
  setup across uv-managed Python installs and isolated worker
  environments. The public material-agent service image installs a
  Python 3.12 worker interpreter when the Scene Optimizer bundle is
  present.
- Public Quick Start paths now work with only `NVIDIA_API_KEY` in the
  repo-root `.env`: physics-agent `lightbulb.yaml` uses local OVRTX for
  `identify_asset`, agent-service Docker Compose files no longer
  clobber API keys with empty substitutions, and README commands pass
  `--env-file .env` so backend/model overrides are honored.
- Model credential routing now uses a shared credential-resolution path
  for chat, VLM, and image-generation models. Public docs and examples
  list the supported key variables, and local NIM base URL overrides can
  be configured without relying on material-agent-specific environment
  names in other services.
- The public texture-agent example config uses the NIM image-generation
  backend by default instead of requiring OpenAI credentials.
- Texture-agent now overrides pre-baked MDL `*_texture` inputs on
  SimReady/OmniPBR-style materials, clears unbundleable URI references,
  and localizes Asset-typed local PNG inputs so downloaded USD/USDZ
  outputs render with the generated textures.
- The shipped ladder texture fixture now uses portable material
  references generated by material-agent rather than unreachable
  SimReady/Nucleus MDL bindings.
- Texture-agent cached texture reuse validates complete PBR sets before
  treating cached files as hits, so partial outputs from failed
  generations are regenerated.
- Texture generation no longer silently completes when per-material
  image generation or blend calls fail. Failures are attributed to the
  failing step and exposed in API/SSE status instead of surfacing later
  as a generic missing-output symptom.
- `texture-agent-service` submit/regenerate endpoints now reject invalid
  input with 4xx responses before job creation.
- `texture-agent-service` API docs, OpenAPI schema, request validation,
  artifact/status errors, cancellation cleanup, and runtime session
  lifecycle handling were hardened to match the public API contract.
- `texture-agent-service` keeps per-session API state consistent across
  `/pipeline`, `/status`, `/results`, `/sessions`, deletion,
  cancellation, and event-log/SSE views.
- Customer-facing texture-service surfaces redact session-storage paths
  and NVCF function-invocation URLs from errors, failed-step stats, and
  session views.
- The texture-agent materials artifact endpoint returns stable material
  metadata from discovery output instead of incomplete or mismatched
  records.
- `physics-agent-service` `/pipeline/upload-usd` persists upload/init
  outcomes into session state so status and result endpoints report the
  initialized USD state reliably.
- `texture-agent run --only/--skip` validates explicit empty filters,
  unknown step names, and mutually exclusive filter combinations before
  scheduling pipeline steps.
- Unsupported image-conditioning warnings from texture generation are
  deduplicated so users see a single actionable warning per run.

# Content Agents 0.3.9 (28 Apr 2026)

Bug-fix release addressing 10 issues filed against the 0.3.8 public release,
plus a new SimReady teaser pipeline and texture-agent improvements.

## Added

- SimReady teaser GIF grid in `README.md` and `README_PUBLIC.md` with four
  animated teasers showing each asset progressing from gray input through
  Material Agent material assignment, Texture Agent rusty texture pass, and
  Physics Agent drop simulation.
- `texture-agent` CLI now auto-loads a project-local `.env`, so the documented
  Quick Start runs zero-edit when keys live in a `.env` file alongside the
  config.
- `texture-agent` `apply_textures` step now also writes concrete OpenPBR
  `tiledimage_*` shader inputs alongside the existing abstract inputs.
- `texture-agent` `prepare_uvs` step has a Python-only fallback when the Scene
  Optimizer UV path is unavailable.
- README adds a Use a Coding Agent section, launch examples for common coding
  agents, follow-up prompts, and a Bring Your Own Asset walkthrough.

## Changed

- README marks Material Agent and Physics Agent as beta to reflect their
  current public readiness.
- `ovrtx-rendering-api` extracts ZIP payloads, including `.usdz` packages, into
  a working directory before re-export so relative texture references resolve
  as real files.
- `README_PUBLIC.md` adds system requirements and authenticated NGC login
  guidance.
- `README_PUBLIC.md` adds output-location guidance for generated artifacts.
- `apps/texture_agent_service/docs/api.md` examples were updated to match live
  status and results response formats.

## Fixed

- Public Quick Start now succeeds when the source is a ZIP download without a
  `.git` directory.
- Editable installs for the agent services now support the documented client
  imports from any current working directory.
- The texture-agent-service Dockerfile no longer references an unreachable
  internal `--extra-index-url`.
- Public material-agent and physics-agent service Docker Compose files no
  longer reference internal-only inference environment variables.
- `texture-agent run` exits non-zero when every per-unit texture generation
  request fails instead of reporting success.
- The texture-agent public example config uses the public NIM image-generation
  default.
- Packages whose source imports `requests` declare it as a direct dependency.
