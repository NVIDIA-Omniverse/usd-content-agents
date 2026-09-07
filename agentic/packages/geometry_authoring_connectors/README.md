# Geometry Authoring Connectors

Fail-closed reference connectors for the provider-neutral contract in
`geometry_authoring_contracts`.

The package provides external authoring integration without embedding an authoring
engine in the Geometry Agent:

- `Build123dHttpConnector` calls one administrator-configured worker endpoint
  for generation, revision, and explicit export.
- `DelegatedAuthoringHttpConnector` calls an explicitly selected external
  authoring worker without importing that provider's SDK or runtime. The
  provider ID included for this route is `forgecad-http`.
- `ForgeCadArtifactAdapter` imports inert `.forge.js` provenance and existing
  STEP/mesh/USD exports. It never invokes ForgeCAD.

All connectors return a materialized `geometry.source.v1` bundle with SHA-256,
size, source revision, units, axes, provider identity, and artifact roles. Provider
selection is explicit. There is no automatic fallback.

The delegated worker protocol is intentionally small. The configured endpoint
accepts one bounded
`geometry-authoring-connectors.authoring-request.v1` JSON document and returns
one `geometry.source.v1` JSON bundle whose artifacts are base64, size, and
SHA-256 bound. The service materializes those bytes; it never executes a
returned native-source artifact. Capabilities and supported formats are
operator configuration and are enforced before the HTTP request. They are not
inferred from a provider name. The Build123d connector defaults to text plus
STEP generation only and does not advertise image conditioning, revision,
export, semantic metadata, assertions, or parameter families unless an
operator explicitly enables each supported capability.

Request parameter values remain a scalar map for simple workers. Optional
`parameter_units` binds units by parameter name without changing that map's
shape.

The source bundle may also return semantic part hierarchy, rich parameter
definitions, provider verification assertions, forward axis, and handedness.
Requested semantic values must be present unchanged in the returned bundle.
This prevents a worker from claiming that it produced a family variant while
silently ignoring the requested controls.

## External Worker Runner

`ExternalAuthoringWorkerRunner` is the common framework-neutral core for
Build123d, authorized ForgeCAD, and custom workers.
`Build123dWorkerRunner` remains a compatibility specialization. A backend
returns `ExternalAuthoringWorkerResult`, which includes:

- provider version and immutable source revision;
- units, up/forward axes, and handedness;
- workspace-contained artifacts and their roles;
- optional semantic parts and transforms;
- optional rich semantic parameter definitions; and
- optional provider-authored assertions and metrics.

`CommandAuthoringExecutionBackend` is a reference adapter for an independently
isolated deployment. Its executable and base argv are fixed by the operator.
For each request it appends only these fixed-shape arguments:

```text
--request /worker/workspace/authoring-request.json
--output-dir /worker/workspace/output
--result /worker/workspace/authoring-result.json
```

The command writes provider artifacts under the output directory and writes one
JSON `ExternalAuthoringWorkerResult` to the result path. It is invoked without a
shell, inherits no ambient Geometry Agent environment, has a bounded timeout,
and never exposes provider stdout or stderr through a public error. Provider
credentials may be supplied explicitly by the worker deployment; they must not
be written into the result, artifact paths, source metadata, or logs.

The fixed-command backend requires POSIX process-group and no-follow file
semantics. It terminates the complete provider process group after success,
failure, or timeout so descendants cannot outlive one request.

This adapter does not create a sandbox. Construct it only with
`isolation_kind="container"` or `"sandboxed-process"` after the deployment
actually enforces that boundary.

### Provider implementations

- A Build123d command may generate or revise provider-owned Python, execute it
  in its isolated environment, explicitly re-export an immutable revision as
  STEP/mesh/USD, and describe the model's meaningful controls. Build123d is not
  imported by this package.
- A ForgeCAD command may generate, revise, and re-export its immutable revision
  only when the operator has the required installation and automated-use
  rights. No request can turn on that integration; service registration still
  requires explicit authorization.

All implementations expose the same HTTP wire document. A deployment may use
FastAPI, another authenticated service framework, or an existing job system to
call `runner.handle(payload)`; authentication, admission control, rate limits,
and network policy belong to that deployment.

A production worker image must also contain the native libraries required by
its selected Build123d/OCP wheel. For the supported Debian bookworm family,
`libgl1` and `libxrender1` are required for a headless import of Build123d
0.10.0. Keep those provider-runtime dependencies in the worker image; do not
add them or Build123d itself to Geometry Agent.

## Trust Boundaries

### Build123d

The Geometry Agent never evaluates Python. The HTTP connector sends only a bounded
typed intent request and receives bounded, digest-bound artifacts.

`Build123dWorkerRunner` uses the common framework-neutral core. A deployment
must provide a `Build123dExecutionBackend` whose
`isolation_kind` is `container` or `sandboxed-process`. The interface deliberately
does not include a local evaluator or process launcher. Returned paths must remain
inside the runner-owned temporary workspace and must be single-link regular files.

The deterministic backend under `tests/trusted_fixture_backend.py` is fixture-only.
It writes checked-in constant output and is rejected unless the runner is created
with `allow_trusted_test_fixture=True`.

### Onshape

Onshape FeatureScript authoring uses the official Onshape Labs MCP at
`https://fs-mcp.labs.onshape.app/mcp`. The user's MCP client owns the Onshape
OAuth session; this package and Geometry Agent receive no account credentials.
The evaluated MCP tool surface does not export files. The user can export from
Onshape manually, or use this package's narrow API-key connector to create an
immutable version and export STEP, glTF, or OBJ. The connector cannot author or
revise geometry.

The companion `geometry-agent export-onshape` command reads
`ONSHAPE_API_KEY` and `ONSHAPE_API_SECRET` only from its local process
environment. Credentials are not command arguments, bundle fields, service
settings, or Helm values. A mutable workspace export requires an explicit
snapshot name; the command creates a version before exporting so the resulting
`geometry.source.v1` bundle has an immutable source identity. Configure secrets
through a local secret manager or environment outside the agent conversation.

Use STEP when exact CAD exchange is required. Use glTF for public USD conversion
and OVRTX evidence when an authorized B-rep conversion backend is unavailable.
Mesh export does not by itself establish watertightness or simulation readiness.

Official references:

- <https://www.onshape.com/en/features/onshape-labs>
- <https://onshape-public.github.io/docs/auth/apikeys/>
- <https://onshape-public.github.io/docs/api-adv/translation/>

### ForgeCAD

The public adapter contains no ForgeCAD import, installer, command invocation, or
source evaluator. A `.forge.js` file is retained only as inert UTF-8 provenance and
must be accompanied by existing geometry. Supported geometry is checked for a
format-specific envelope before materialization. An optional strict manifest binds
every artifact by filename, role, media type, size, and SHA-256.

ForgeCAD availability and licensing remain the operator's responsibility. This
Apache-2.0 connector package does not grant rights to ForgeCAD or any other external
provider.

An operator with authorization for automated ForgeCAD use may expose a separate
worker through `forgecad-http`. Registration requires both a rights assertion
and an affirmative automated-use authorization setting. That worker may
generate, revise, and explicitly export provider-owned revisions. This route
does not change the artifact adapter: `ForgeCadArtifactAdapter` remains
import-only, and this package still contains no ForgeCAD dependency or command
invocation.

ForgeCAD capabilities fail closed at the service boundary. The default is
text-to-STEP generation only; revision, export, semantic parameters and
definitions, parts, provider assertions, parameter families, native source,
additional formats, and longer timeouts must match a qualified worker and be
enabled explicitly. A worker must also queue provider execution according to
its authorized concurrency. Geometry Agent may request family variants in
parallel, but it cannot infer or expand an external product's licensed seat
count.

### External licensing

- Build123d is available under Apache-2.0, but the selected worker and any other
  components it uses remain the operator's responsibility:
  <https://github.com/gumyr/build123d>.
- ForgeCAD's public kit is MIT licensed, while its CLI, package, hosted service,
  and automated/backend use may be governed separately. Do not infer runtime or
  automation rights from the public-kit license:
  <https://github.com/ForgeCAD/forgecad-public-kit>.
- Onshape Labs access is user/account authorized through the official MCP.

## Transport Policy

- Remote endpoints require HTTPS; HTTP is accepted only for loopback workers.
- URLs cannot contain credentials, queries, or fragments.
- OAuth, API-key, and bearer credentials are added only as request headers and
  never enter configuration digests, artifact metadata, or error messages.
- Redirect following is disabled.
- Request, response, artifact, poll-count, and timeout bounds are explicit.
- Content length, streamed byte count, base64, size, and SHA-256 are verified.
- Output files are created exclusively and are rolled back if publication fails.
- Provider bodies and exception details are not exposed in typed public errors.
- JSON parameter or provenance sidecars are accepted only as non-runnable
  `application/json`, with a 4-MiB limit plus standard finite JSON, unique
  object keys, nesting, key-length, and value-count checks.

## Integration

The connector classes structurally implement the generation, revision, export, and
capability surface expected by `geometry_authoring_contracts`.
Registration and secret resolution belong to the service layer; this package reads
no environment variables and contains no credentials.

The package dependencies are limited to the provider-neutral contracts,
Pydantic, and Requests. Build123d, ForgeCAD, and Onshape client libraries are
not dependencies of this package or the Geometry Agent service image.

Run focused checks from this directory:

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

## License

This package is licensed under Apache-2.0. External authoring systems and their
outputs remain subject to their respective terms.
