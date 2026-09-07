# Geometry Agent Quickstart

Geometry Agent accepts existing CAD, mesh, and USD artifacts or an immutable
`geometry.source.v1` bundle from an explicitly selected external authoring
provider. It converts and prepares geometry, applies requested optimization and
repair policies, runs deterministic validation, renders final evidence with
OVRTX, and publishes digest-bound handoff manifests.

Geometry Agent does not execute provider-native source. It also does not claim
final materials, physics, articulation, manufacturing readiness, or SimReady
conformance; those claims remain with their owning workflows and validators.

## Install

From the repository root, use Python 3.12 and `uv`:

```bash
cp .env_example .env
./scripts/setup_content_agent.sh
source .venv/bin/activate
content-workflow-cli geometry run --help
geometry-agent --help
```

For a Geometry-only environment, sync the public service project directly:

```bash
uv sync --project apps/geometry_agent_service --extra dev
```

Native B-rep inspection and healing are not included in the public runtime.
For STEP, IGES, or BREP authoring, configure an authorized provider that also
returns validated USD or mesh geometry with explicit units and axes.

Local accepted visual evidence requires a supported NVIDIA RTX/Vulkan runtime.
Provision the isolated, hash-locked OVRTX environment once:

```bash
export WU_OVRTX_VENV_DIR="$HOME/.cache/wu/ovrtx_venv"
python -m world_understanding.functions.graphics.render_ovrtx \
  --provision-only \
  --ovrtx-venv-dir "$WU_OVRTX_VENV_DIR"
```

Do not install OVRTX into the main workflow environment. A remote OVRTX service
may be used when the local host has no supported GPU.

## Existing Geometry

Run the checked-in smoke asset:

```bash
content-workflow-cli geometry run \
  agentic/examples/geometry/quickstart/smoke_bracket.usda \
  --output-dir runs/geometry-smoke-001
```

The default run preserves correspondence during optimization and requests a
six-view path-traced OVRTX render. Runtime and SimReady validation remain
skipped unless explicitly requested.

Run a CAD or mesh source with acceptance context:

```bash
content-workflow-cli geometry run path/to/bracket.step \
  --prompt-file path/to/geometry_requirements.md \
  --reference-image path/to/reference.png \
  --install-missing-converters \
  --repair-mode diagnose \
  --repair-profile rigid_pick_place \
  --output-dir runs/bracket-geometry-001
```

Use `--dry-run --json` to freeze and inspect a request without executing it.
Each output directory is create-only; use a new directory for every attempt.

## External Authoring

Text and image authoring happen outside the Geometry workflow. The public
service sends bounded requests to one configured provider and accepts only a
typed receipt plus content-addressed exported artifacts.

Run `geometry-agent providers` and select the provider ID named by the user.
The service never installs a provider runtime and never falls back to another
provider. Build123d and ForgeCAD authoring workers are separate deployments and
are not Python dependencies of this repository. Onshape authoring uses its
official user-authorized MCP and then the normal file-upload path.

Any authorized worker that implements the bounded delegated protocol can be
registered without adding its SDK to Geometry Agent:

```bash
export GEOMETRY_AGENT_SERVICE_DELEGATED_AUTHORING_PROVIDER_ID="authorized-authoring-http"
export GEOMETRY_AGENT_SERVICE_DELEGATED_AUTHORING_PROVIDER_LABEL="Authorized authoring worker"
export GEOMETRY_AGENT_SERVICE_DELEGATED_AUTHORING_ENDPOINT_URL="https://worker.example/api/authoring"
export GEOMETRY_AGENT_SERVICE_DELEGATED_AUTHORING_BEARER_TOKEN="replace-with-worker-token"
export GEOMETRY_AGENT_SERVICE_DELEGATED_AUTHORING_RIGHTS_ASSERTION="Authorized project output"
export GEOMETRY_AGENT_SERVICE_DELEGATED_AUTHORING_SUPPORTED_FORMATS='["step","stl","usd","usda"]'
```

The endpoint receives
`geometry-authoring-connectors.authoring-request.v1` and returns a bounded
`geometry.source.v1` wire bundle. Provider-native source may be retained as an
inert artifact, but Geometry Agent never imports or executes it.

Start the authenticated service with an isolated Build123d worker:

```bash
export GEOMETRY_AGENT_SERVICE_API_KEY="replace-with-a-secret"
export GEOMETRY_AGENT_SERVICE_WORKSPACE_ROOT="$PWD/runs/geometry-service"
export GEOMETRY_AGENT_SERVICE_BUILD123D_ENDPOINT_URL="https://worker.example/v1/author"
export GEOMETRY_AGENT_SERVICE_BUILD123D_BEARER_TOKEN="replace-with-worker-token"
export GEOMETRY_AGENT_SERVICE_BUILD123D_RIGHTS_ASSERTION="Authorized project output"
# This JSON list must match the formats emitted by that worker deployment.
export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTED_FORMATS='["step","stl"]'
# Select `remote` when the service uses a configured render endpoint.
export GEOMETRY_AGENT_SERVICE_RENDER_BACKEND="remote"
export GEOMETRY_AGENT_SERVICE_RENDER_REMOTE_BASE_URL="https://render.example/v1"
export GEOMETRY_AGENT_SERVICE_RENDER_REMOTE_API_KEY="replace-with-render-key"

uv run --project apps/geometry_agent_service geometry-agent-service \
  --host 127.0.0.1 --port 8776
```

In another shell, generate from text or images and immediately run Geometry:

```bash
export GEOMETRY_AGENT_SERVICE_URL="http://127.0.0.1:8776"
export GEOMETRY_AGENT_SERVICE_API_KEY="replace-with-a-secret"

uv run --project apps/geometry_agent_service geometry-agent generate \
  --provider build123d-http \
  --prompt "A 60 mm mounting bracket with two M5 clearance holes" \
  --image path/to/reference.png \
  --parameter width_mm=60 \
  --format step \
  --run --render-evidence \
  --output runs/generated-bracket-001.json
```

Revise a provider-produced source with the same provider:

```bash
uv run --project apps/geometry_agent_service geometry-agent revise \
  src_REPLACE_WITH_SOURCE_ID \
  --provider build123d-http \
  --prompt "Increase width to 75 mm and preserve both mounting holes" \
  --parameter width_mm=75 \
  --format step \
  --output runs/revised-bracket-001.json
```

The service rejects revisions of plain uploads and cross-provider revision
claims. It requires the prior immutable provider bundle and preserves its
lineage in the next receipt.

## Onshape

Connect an MCP-compatible client to the official Onshape Labs FeatureScript MCP:

```text
URL: https://fs-mcp.labs.onshape.app/mcp
Transport: HTTP
```

Complete the Onshape OAuth flow in that client. Ask the agent to create and test
a parameterized FeatureScript feature. The MCP currently does not export files.
The user can export manually, or configure a personal API key locally for the
export-only CLI. Never paste credentials into chat or pass them as command
arguments. Use a local secret manager to expose `ONSHAPE_API_KEY` and
`ONSHAPE_API_SECRET` only to the CLI process:

```bash
geometry-agent export-onshape \
  --document-id DOCUMENT_ID \
  --workspace-id WORKSPACE_ID \
  --snapshot-name "Geometry Agent export YYYY-MM-DD" \
  --element-id ELEMENT_ID \
  --element-kind partstudio \
  --format step \
  --rights-assertion "User-authorized Onshape export" \
  --output-dir runs/onshape-step
```

The workspace form creates a named immutable version before export. Use
`--version-id VERSION_ID` instead of `--workspace-id` and `--snapshot-name` when
one already exists. Export glTF from the same version when the public Geometry
workflow needs a renderable mesh without an exact B-rep backend. The helper
signs requests in memory and emits a digest-bound `geometry.source.json` without
credentials. Neither the MCP OAuth session nor the API key enters Geometry
Agent service or Helm configuration.

## ForgeCAD

When an operator has authorization for automated ForgeCAD use, configure a
separate worker and affirm that authorization before selecting `forgecad-http`:

```bash
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_ENDPOINT_URL="https://worker.example/forgecad"
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_BEARER_TOKEN="replace-with-worker-token"
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_RIGHTS_ASSERTION="Authorized project output"
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTOMATED_USE_AUTHORIZED=true

geometry-agent generate \
  --provider forgecad-http \
  --prompt "A 75 mm slotted motor mount with four M4 holes" \
  --format step \
  --output runs/forgecad-generation-001.json
```

The repository does not install or invoke ForgeCAD. The public-kit license must
not be treated as authorization for a separately governed CLI, package,
backend, hosted service, or automated use. Without an authorized worker, use
only existing exports and inert provenance:

```bash
uv run --project apps/geometry_agent_service geometry-agent import-forgecad \
  --export path/to/object.step \
  --forge-source path/to/object.forge.js \
  --rights-assertion "Authorized project export" \
  --output-dir runs/forgecad-import-001
```

The artifact connector never imports, installs, invokes, or evaluates ForgeCAD.

## Source Bundles

Run Geometry directly from any conforming provider bundle:

```bash
content-workflow-cli geometry run \
  --source-manifest path/to/geometry.source.json \
  --output-dir runs/provider-handoff-001
```

The workflow verifies the manifest and selected representation through held,
no-follow file descriptors, rechecks byte size and SHA-256 identity, and
retains provider, revision, parts, semantic parameters, rights, and provenance
in the final handoff evidence.

## Result Contract

A completed run writes the frozen request and terminal result plus the selected
artifacts:

```text
geometry_request.json
geometry_workflow_result.json
content_agents_manifest.json
geometry_validation_evidence.json
geometry_evidence_bundle.json
geometry_render_evidence_*.json
*.usdc
*.png
```

The CLI exits `0` for a ready handoff, `3` for a conditional handoff requiring
review, `1` for rejection, and `2` for request or execution failure. Missing
optimization, rendering, runtime, or SimReady capabilities are never converted
into a pass.

For policy and routing details, read
[`../.agents/skills/content-workflow-geometry/SKILL.md`](../.agents/skills/content-workflow-geometry/SKILL.md),
[`../packages/geometry_authoring_contracts/README.md`](../packages/geometry_authoring_contracts/README.md),
and
[`../packages/geometry_authoring_connectors/README.md`](../packages/geometry_authoring_connectors/README.md).
