---
name: fixed-pipeline
description: "Use when the user explicitly requests a fixed-pipeline Content Agent CLI, service client, deployment, benchmark, or maintenance workflow under apps/: material-agent-cli, material-agent-client, physics-agent-cli, physics-agent-client, joint-agent-cli, joint-agent-client, joint-agent-validation, texture-agent-cli, texture-agent-client, validation-agent-cli; deploy-collection, deploy-material-agent-docker, deploy-material-agent-brev, deploy-physics-agent-docker, deploy-physics-agent-brev, deploy-texture-agent-docker, deploy-texture-agent-brev, deploy-ovrtx-docker, deploy-embeddings-brev, deploy-image-gen-brev, or deploy-qwen-vlm-brev."
version: "0.1.4"
author: NVIDIA Omniverse
tags:
  - content-agents
  - fixed-pipeline
  - routing
  - compatibility
tools:
  - Shell
  - Filesystem
  - Python
compatibility: Requires the repository checkout and the prerequisites documented by the selected fixed-pipeline reference; individual app and deployment requirements vary.
metadata:
  author: NVIDIA Omniverse
  version: "0.1.4"
  tags:
    - content-agents
    - fixed-pipeline
    - routing
    - compatibility
---

# Fixed Pipeline

## Purpose

Use this umbrella only for an explicitly selected fixed-pipeline interface. For
an unqualified asset task, use the matching top-level agentic
`content-workflow-*` skill instead.

Fixed pipelines support Linux and Windows. On Windows, run them under WSL2 with
the `warp` rendering backend only. Native Windows fixed-pipeline execution and
local OVRTX rendering under WSL2 are unsupported. Deployment references retain
their own narrower host requirements and do not expand this runtime policy.

## When to Use

- The user says "fixed pipeline" or names an existing app command.
- The user asks for an `apps/*_agent` YAML/config workflow, Python API,
  benchmark, or REST service.
- The user explicitly asks for one of the retained Docker, Brev, Helm, NVCF,
  or collection deployment procedures.
- Existing automation already depends on an app command or service contract.

## Instructions

1. Confirm the request explicitly selects the fixed pipeline. Do not infer it
   from a generic material, texture, physics, articulation, validation, scene,
   or conversion request.
2. Select the narrowest reference below and read its `reference.md` before
   taking action. These are bundled references, not separately invocable
   skills.
3. Preserve the named runtime command, YAML format, Python API, benchmark, or
   REST contract. Moving the guidance under this umbrella does not rename
   runtime interfaces.
4. If a required prerequisite is missing, report remediation. Do not silently
   reroute the work to or from the agentic workflow.
5. Report the selected workflow as `fixed-pipeline`, plus the command, output,
   status, and exact resume or safe-restart action.

### App CLIs and service clients

| Explicit request | Read this reference |
|---|---|
| Material CLI, YAML, dataset, benchmark, or resume | [`material-agent-cli`](references/material-agent-cli/reference.md) |
| Material REST client | [`material-agent-client`](references/material-agent-client/reference.md) |
| Physics CLI, YAML, tuning, benchmark, or refine | [`physics-agent-cli`](references/physics-agent-cli/reference.md) |
| Physics REST client | [`physics-agent-client`](references/physics-agent-client/reference.md) |
| Joint CLI or YAML | [`joint-agent-cli`](references/joint-agent-cli/reference.md) |
| Joint REST client | [`joint-agent-client`](references/joint-agent-client/reference.md) |
| Joint Gate 3 validation | [`joint-agent-validation`](references/joint-agent-validation/reference.md) |
| Texture CLI or YAML | [`texture-agent-cli`](references/texture-agent-cli/reference.md) |
| Texture REST client | [`texture-agent-client`](references/texture-agent-client/reference.md) |
| Validation Agent CLI | [`validation-agent-cli`](references/validation-agent-cli/reference.md) |

### Deployment references

| Explicit request | Read this reference |
|---|---|
| Full service collection | [`deploy-collection`](references/deploy-collection/reference.md) |
| Material Docker | [`deploy-material-agent-docker`](references/deploy-material-agent-docker/reference.md) |
| Physics Docker | [`deploy-physics-agent-docker`](references/deploy-physics-agent-docker/reference.md) |
| Texture Docker | [`deploy-texture-agent-docker`](references/deploy-texture-agent-docker/reference.md) |
| Standalone OVRTX Docker | [`deploy-ovrtx-docker`](references/deploy-ovrtx-docker/reference.md) |
| Material on Brev | [`deploy-material-agent-brev`](references/deploy-material-agent-brev/reference.md) |
| Physics on Brev | [`deploy-physics-agent-brev`](references/deploy-physics-agent-brev/reference.md) |
| Texture on Brev | [`deploy-texture-agent-brev`](references/deploy-texture-agent-brev/reference.md) |
| Qwen VLM on Brev | [`deploy-qwen-vlm-brev`](references/deploy-qwen-vlm-brev/reference.md) |
| Image generation on Brev | [`deploy-image-gen-brev`](references/deploy-image-gen-brev/reference.md) |
| Embeddings on Brev | [`deploy-embeddings-brev`](references/deploy-embeddings-brev/reference.md) |

In internal checkouts, an explicitly named operational reference may also be
available at `references/<requested-name>/reference.md`. Check that exact bounded
path rather than scanning or loading every reference.

### Downstream workflow consumers

An external agentic orchestrator may explicitly compose one of these stable
service contracts without changing its own workflow family. Resolve the
selected v0.6 reference from
`.agents/skills/fixed-pipeline/references/<name>/reference.md`, keep the operation
labeled `fixed-pipeline`, and invoke only the named app or REST interface. In
particular, CAD-to-SimReady consumers may sequence `material-agent-client`,
then `physics-agent-client`, and optionally `texture-agent-client`; deployment
may select the matching Docker references above. Do not reinterpret that
explicit service sequence as permission to invoke an upstream agentic workflow
or to fall back between workflow families.

Consumers that intentionally support a pre-0.6 checkout may use the former
`.agents/skills/<name>/SKILL.md` path only when the v0.6 nested path is absent.
Fail closed when neither exact path exists, and never discover a replacement by
scanning unrelated skill directories.

## Examples

- "Run the fixed-pipeline Material Agent from this YAML" routes to
  `material-agent-cli`.
- "Deploy the Texture Agent service with Docker" routes to
  `deploy-texture-agent-docker`.
- "Convert this asset to SimReady" is unqualified and routes to the matching
  agentic `content-workflow-*` skill, not this umbrella.

## Prerequisites

- Start from the repository root.
- Read the selected reference's prerequisites before running commands.
- Keep credentials in `.env` or environment variables; never print or commit
  them.
- Verify the named CLI, service, renderer, model endpoint, container runtime,
  or cloud environment required by that reference.

## Output Format

Report:

- workflow: `fixed-pipeline`;
- selected reference and runtime interface;
- command or API operation performed;
- status and output/artifact location;
- unresolved prerequisites or decisions;
- exact resume, retry, teardown, or safe-restart command.

## Limitations

- This umbrella is not the default route for unqualified Content Agent tasks.
- Nested references are not independently discoverable or invocable skills.
- A reference's retained `agents/openai.yaml` is archived presentation metadata;
  without a sibling `SKILL.md`, it is not a discovery or invocation entrypoint.
- "Fixed pipeline" describes the established application control flow; it does
  not promise bitwise-deterministic model output.
- Do not load all references into context. Read only the selected workflow and
  any sibling reference it explicitly requires.

## Troubleshooting

| Problem | Action |
|---|---|
| Request is unqualified | Use the matching top-level agentic skill. |
| Named reference is absent | Report that the workflow is unavailable in this checkout; do not invent its procedure. |
| Reference points to a sibling workflow | Load that sibling from this umbrella's `references/` directory. |
| Runtime prerequisite is missing | Stop with the selected reference's remediation; do not switch workflow families silently. |
