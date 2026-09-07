# Geometry Agent Service

Authenticated public service for geometry source intake, authoring-provider
orchestration, geometry workflow execution, and evidence handoff.

The service exposes only content-addressed artifact and source IDs. Clients
cannot provide server-local paths or arbitrary URLs. Uploaded Python,
JavaScript, shell source, and other executable authoring formats are rejected.
Archive uploads are bounded and checked for traversal, links, encryption,
duplicate members, unsupported member types, expansion size, and compression
ratio before they are stored or extracted. Dependency-bearing formats are also
fail-closed: upload glTF, OBJ, PLY, USD, USDZ, and 3MF with every referenced
buffer, texture, material, or layer in one validated source package. A direct
single-file upload is accepted only when it is self-contained; remote,
absolute, traversal, executable OBJ, and missing dependency references are
rejected before registration.

## API

- `POST /api/geometry/sources` uploads a geometry source or reference image.
- `POST /api/geometry/generations` submits prompt/image intent to one explicitly
  selected registered authoring provider.
- `POST /api/geometry/revisions` submits a bounded change request against an
  immutable source bundle produced by the same provider.
- `POST /api/geometry/families` materializes up to 64 validated semantic
  variants from one immutable provider revision.
- `POST /api/geometry/provider-exports` binds and exports a provider-owned
  immutable revision that has not yet been registered with Geometry Agent.
- `POST /api/geometry/exports` requests a separate export operation from a
  provider that advertises it.
- `POST /api/geometry/runs` runs the public geometry workflow for a registered
  geometry source.
- `GET /api/geometry/jobs/{job_id}` reads the durable JSON job record.
- `GET /api/geometry/providers` lists registered provider capabilities.
- `GET /api/health` and `GET /api/info` are non-sensitive probe endpoints.

The initial implementation executes jobs synchronously but persists every
queued, running, and terminal state atomically. Responses use the same job
contract required by a future asynchronous executor.

Geometry preparation and USD validation work without a simulation runtime.
Runtime validation is disabled by default because the public service image does
not bundle OvPhysX. A deployment that supplies a supported runtime may set
`GEOMETRY_AGENT_SERVICE_DEFAULT_RUNTIME_ENGINE` and request
`run_runtime_validation=true`; the service rejects that request when no engine
is configured instead of failing later inside the workflow.

Authoring providers implement
`geometry_authoring_contracts.GeometryAuthoringProvider`
and are registered explicitly through
`geometry_agent_service.main.authoring_providers` during
deployment startup. No provider is selected as a fallback. A request for an
unregistered provider returns a durable failed job with
`authoring_provider_unavailable`.

## Run locally

Python 3.12 and the repository's `content-agent-workflows` package are required.
Native B-rep inspection and healing are not included. CAD authoring providers
should return validated USD or mesh geometry with explicit units and axes.

```bash
uv sync --project apps/geometry_agent_service
```

Subsequent local launches can remain frozen to that environment:

```bash
export GEOMETRY_AGENT_SERVICE_API_KEY="replace-with-a-secret"
export GEOMETRY_AGENT_SERVICE_WORKSPACE_ROOT="$PWD/.geometry-agent-workspace"
# Use `remote` with a configured render endpoint.
export GEOMETRY_AGENT_SERVICE_RENDER_BACKEND="remote"
export GEOMETRY_AGENT_SERVICE_RENDER_REMOTE_BASE_URL="https://render.example/v1"
export GEOMETRY_AGENT_SERVICE_RENDER_REMOTE_API_KEY="replace-with-render-key"
# Only for an operator-controlled, unauthenticated sidecar. Never enable this
# for a public or otherwise untrusted endpoint.
# export GEOMETRY_AGENT_SERVICE_RENDER_REMOTE_ALLOW_UNAUTHENTICATED_IDENTITY=true
# Register an independently deployed Build123d worker.
# export GEOMETRY_AGENT_SERVICE_BUILD123D_ENDPOINT_URL="https://worker.example/v1/author"
# export GEOMETRY_AGENT_SERVICE_BUILD123D_BEARER_TOKEN="replace-with-worker-token"
# export GEOMETRY_AGENT_SERVICE_BUILD123D_RIGHTS_ASSERTION="Authorized provider output"
# Advertise only formats and operations the configured Build123d worker supports.
# Pydantic Settings parses complex environment values as JSON.
# export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTED_FORMATS='["step","stl"]'
# export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_TEXT=true
# export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_IMAGE=false
# export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_REVISION=true
# export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_EXPORT=true
# export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_SEMANTIC_PARAMETERS=true
# export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_PARAMETER_DEFINITIONS=true
# export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_SEMANTIC_PARTS=true
# export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_PROVIDER_ASSERTIONS=true
# export GEOMETRY_AGENT_SERVICE_BUILD123D_MAX_FAMILY_VARIANTS=64
# Optional delegated authoring workers use separate provider IDs. Each requires
# a worker endpoint, bearer secret, and an operator-supplied rights assertion.
# export GEOMETRY_AGENT_SERVICE_DELEGATED_AUTHORING_PROVIDER_ID="authorized-authoring-http"
# export GEOMETRY_AGENT_SERVICE_DELEGATED_AUTHORING_PROVIDER_LABEL="Authorized authoring worker"
# export GEOMETRY_AGENT_SERVICE_DELEGATED_AUTHORING_ENDPOINT_URL="https://worker.example/api/authoring"
# export GEOMETRY_AGENT_SERVICE_DELEGATED_AUTHORING_BEARER_TOKEN="replace-with-worker-token"
# export GEOMETRY_AGENT_SERVICE_DELEGATED_AUTHORING_RIGHTS_ASSERTION="Authorized project output"
# ForgeCAD also requires GEOMETRY_AGENT_SERVICE_FORGECAD_AUTOMATED_USE_AUTHORIZED=true.
# Its defaults are text-to-STEP generation only. Enable richer worker claims only
# after qualifying the exact deployment, for example:
# export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_REVISION=true
# export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_EXPORT=true
# export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_SEMANTIC_PARAMETERS=true
# export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_PARAMETER_DEFINITIONS=true
# export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_SEMANTIC_PARTS=true
# export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_PROVIDER_ASSERTIONS=true
# export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_MAX_FAMILY_VARIANTS=16
# export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_RETURNS_NATIVE_SOURCE=true
# Bound these to the qualified worker latency and its licensed execution queue.
# export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_CONNECT_TIMEOUT_SECONDS=10
# export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_READ_TIMEOUT_SECONDS=900
uv run --project apps/geometry_agent_service --frozen geometry-agent-service \
  --host 127.0.0.1 --port 8776
```

The public 0.6 package and service image deliberately do not include a CAD or
mesh converter. USD-family inputs, including provider-produced USD, can continue
through validation and OVRTX evidence. STEP, IGES, and mesh provider outputs are
retained as immutable source handoffs, but a request to run them fails closed as
converter-unavailable. Conversion requires a separately qualified deployment
whose runtime, license, notices, SBOM, and image scan are reviewed independently.

```bash
curl http://127.0.0.1:8776/api/health
curl -H "X-Geometry-Agent-Service-Key: $GEOMETRY_AGENT_SERVICE_API_KEY" \
  http://127.0.0.1:8776/api/geometry/providers
curl -H "X-Geometry-Agent-Service-Key: $GEOMETRY_AGENT_SERVICE_API_KEY" \
  -H "X-Content-SHA256: $(sha256sum asset.usda | cut -d' ' -f1)" \
  -F role=geometry_source -F file=@asset.usda \
  http://127.0.0.1:8776/api/geometry/sources
```

The service refuses to start without an API key, including on loopback. This
guard also applies when the application is launched directly through Uvicorn.

## Container

Build from the repository root:

```bash
docker build -f apps/geometry_agent_service/Dockerfile .
docker run --rm -p 8776:8776 \
  -e GEOMETRY_AGENT_SERVICE_API_KEY="replace-with-a-secret" \
  geometry-agent-service
```

The image copies the public geometry workflow package and this service only. It
does not bundle an authoring implementation or a CAD/mesh converter; providers
and any qualified conversion runtime are separate deployment components.

The Helm chart exposes the same generic boundary under
`delegatedAuthoring`. When enabled, set an explicit provider ID, the complete
worker endpoint, a rights assertion, and an existing Kubernetes secret holding
the bearer token. The worker endpoint accepts the bounded request and returns
the digest-bound wire bundle described by
`geometry_authoring_connectors`; Geometry Agent never imports the worker's
runtime.

The default one-replica chart uses a pod-local workspace. Multiple replicas
must set `workspace.kind=pvc` and name an existing shared claim that supports
the deployment's concurrent access pattern; the chart rejects configurations
that would make jobs or content-addressed artifacts pod-local.

## Reference connectors

- Build123d generation, revision, and explicit export use an
  administrator-configured remote worker. The service does not execute
  generated Python.
- `forgecad-http` is an optional delegated worker. It uses the same bounded
  request and digest-bound artifact protocol and does not add its provider SDK
  or runtime to this service. Registration fails closed unless the operator
  affirms authorization for automated use.
- Onshape FeatureScript authoring uses the official Onshape Labs MCP in the
  user's MCP client. The user can export a supported file manually or use the
  local `geometry-agent export-onshape` helper after configuring API credentials
  outside the conversation. No Onshape credential enters this service or its
  deployment configuration.
- `geometry-agent import-forgecad` validates existing ForgeCAD exports and may
  retain `.forge.js` only as inert provenance. It never invokes ForgeCAD.

Query `geometry-agent providers` before generation. If the exact provider ID
requested by the user is absent or lacks the needed modality/format, report it
as unavailable; do not install a provider, switch providers, or invoke a local
CLI as an implicit fallback. External products and their outputs remain subject
to their respective licenses and account terms.

Build123d capabilities are not inferred from the package name or probed from an
untrusted endpoint. The operator must configure the exact worker route, formats,
modalities, operations, semantic outputs, and family limit. Defaults are
deliberately conservative: text input, STEP output, no image conditioning, and
generation only. Revision, export, semantic metadata, provider assertions, and
parameter families remain disabled until explicitly enabled. Geometry Agent
never installs Build123d locally. The configured endpoint is the full authoring
route, for example `http://127.0.0.1:8765/v1/author`.

After `geometry-agent providers` reports `build123d-http`, generate one STEP
handoff with:

```bash
geometry-agent generate \
  --provider build123d-http \
  --prompt "Create a 60 mm square mounting plate with four 5 mm corner holes." \
  --format step \
  --output build123d-result.json
```

The public 0.6 service retains this STEP result but does not convert or render
it. To run the downstream Geometry workflow, submit an existing USD-family
asset or request a provider-produced USD-family representation that the exact
provider advertises. The command fails before contacting the worker if the
requested modality, format, or operation is not enabled. See the
[external Build123d integration guide](docs/build123d_external_integration_evaluation.md)
for the worker boundary, evaluated handoff, and deployment checks.

For the official Onshape Labs FeatureScript MCP workflow and evaluated handoff,
see the
[Onshape integration guide](docs/onshape_external_integration_evaluation.md).

For a licensed external ForgeCAD deployment, including fail-closed capability
configuration, semantic parameter families, exact STEP handoff, and OVRTX
qualification evidence, see the
[ForgeCAD integration guide](docs/forgecad_external_integration_evaluation.md).

## Semantic parameter families

The provider must first return a source bundle with parameter definitions.
Create a JSON document such as:

```json
{
  "variants": [
    {
      "variant_id": "compact-ansi",
      "parameter_overrides": {
        "key_spacing_mm": {"value": 18.0, "unit": "mm"},
        "layout": "ansi"
      }
    },
    {
      "variant_id": "wide-iso",
      "parameter_overrides": {
        "key_spacing_mm": {"value": 20.0, "unit": "mm"},
        "layout": "iso"
      }
    }
  ]
}
```

Then materialize both rows from the same base source:

```bash
geometry-agent family SOURCE_ID \
  --provider build123d-http \
  --variants variants.json \
  --format step \
  --output family-result.json
```

The same command works with `forgecad-http` when the
selected worker advertises `parameter_families`. Geometry Agent validates names,
types, units, bounds, steps, choices, lineage, and returned values. It does not
guess provider controls or silently switch providers.

Providers with a separate export operation use the same command. For example:

```bash
geometry-agent export SOURCE_ID \
  --provider build123d-http \
  --format step \
  --output export-result.json
```

Replace the provider with `forgecad-http` when that provider owns the registered
source and advertises `export`. For Onshape, author through the official MCP and
either export manually or run the local export-only helper:

```bash
geometry-agent export-onshape \
  --document-id DOCUMENT_ID \
  --workspace-id WORKSPACE_ID \
  --snapshot-name "Geometry handoff" \
  --element-id ELEMENT_ID \
  --element-kind partstudio \
  --format gltf \
  --rights-assertion "Authorized to export this Onshape element." \
  --output-dir runs/onshape-source
```

The command reads `ONSHAPE_API_KEY` and `ONSHAPE_API_SECRET` from the local
execution environment. Do not paste them into chat or pass them on the command
line. An existing immutable version can be selected with `--version-id` instead
of `--workspace-id` and `--snapshot-name`.

Run `geometry-agent --help` for the complete service and local connector CLI.
