# External ForgeCAD Integration Guide and Evaluation

## Outcome

Geometry Agent can use an authorized ForgeCAD deployment through the public,
provider-neutral authoring boundary without installing ForgeCAD or receiving its
license material. The evaluated worker generated parameterized keyboard and
wall-outlet families, revised and re-exported immutable sources, returned exact
STEP plus secondary GLB, and completed STEP-to-USD validation and OVRTX visual
evidence.

This is a qualified integration path for two deterministic reference families,
not a claim that the worker accepts arbitrary prompts. The final handoffs are
`conditional`, not SimReady-certified: physics, materials, articulation,
runtime behavior, and formal SimReady validation were outside this test.

This document records the historical qualification environment. The public 0.6
Geometry package and service image do not ship the evaluated CAD/mesh converter
because its embedded Python runtime does not meet the release image scan policy.
ForgeCAD STEP and GLB remain immutable provider handoffs; reproducing the
conversion and visual evidence requires a separately qualified converter
deployment or a provider-produced USD-family representation.

## Licensing and Deployment Boundary

ForgeCAD is an optional third-party provider. Geometry Agent does not install,
import, invoke, redistribute, or grant rights to the ForgeCAD runtime. An
operator who chooses this provider must deploy it separately and confirm that
their installation and automated use are authorized.

The [ForgeCAD public kit](https://github.com/ForgeCAD/forgecad-public-kit) is
published under MIT terms. The ForgeCAD CLI, package, and hosted services have
separate [license terms](https://forgecad.io/license) and
[automation guidance](https://forgecad.io/docs/ai-usage). Users must evaluate
those provider terms for their deployment.

The qualification topology was:

```text
Geometry Agent                 Authorized provider deployment
--------------                 ------------------------------
provider ID + capabilities --> authenticated fixed endpoint
bounded authoring request  --> queued ForgeCAD 0.13 execution
immutable source bundle    <-- STEP + GLB + bounded metadata
STEP -> USD -> validation
             -> OVRTX evidence
```

The provider deployment applied these controls:

- fixed authenticated endpoint and fixed executable selected by the operator;
- one licensed execution at a time, with bounded queue and read timeout;
- no shell, request-selected command, endpoint discovery, or request-selected
  URL;
- disabled container network, read-only root filesystem, non-root user, all
  capabilities dropped, and no-new-privileges;
- bounded CPU, memory, process count, output size, workspace, and execution
  time;
- provider credentials and license files mounted read-only only into the
  provider container; and
- provider-native source retained as inert provenance and never executed by
  Geometry Agent.

The reference worker, its two model templates, provider cache, and generated
artifacts are evaluation infrastructure and are not included in this
repository or the Geometry Agent service image.

## Evaluated Objects

### Parameterized Keyboard

Prompt:

> Create a parameterized compact mechanical keyboard with a rigid hollow
> rounded case, switch plate, separate dished keycaps including a centered wide
> spacebar, cable recess, and rear typing feet.

The source exposed nine semantic controls: key columns, key rows, pitch, gap,
bezel, case height, keycap height, corner radius, and typing angle. It retained
five semantic groups: assembly, case, switch plate, keycaps, and rear feet.

Three family rows exercised materially different geometry:

| Variant | Key changes |
| --- | --- |
| `compact-48` | 12 columns, 4 rows, 0.6 mm gap, 4 mm bezel, 2 degree angle |
| `wide-108` | 18 columns, 6 rows, 20 mm pitch, 12 mm bezel, 8 degree angle |
| `low-profile` | 12 mm case, 6 mm keycaps, 4 mm corners, flat typing angle |

An independent revision increased key pitch to 19.5 mm and bezel to 10 mm. A
4 mm key-gap family request was rejected by Geometry Agent before provider
execution.

### Parameterized Wall Outlet

Prompt:

> Create a parameterized North American duplex wall outlet with a rounded
> insulating face, true blade and ground openings, mounting yoke, center screw,
> rear body, line terminals, and ground terminal.

The source exposed seven semantic controls: NEMA 5-20 mode, face width, face
height, body depth, receptacle spacing, face corner radius, and socket count.
It retained seven semantic groups: assembly, insulating body, mounting yoke,
front face, receptacles, center screw, and terminals.

Three family rows exercised:

| Variant | Key changes |
| --- | --- |
| `nema-5-20` | NEMA 5-20 openings, 36 mm face, 31 mm body depth |
| `single-socket` | One receptacle, 66 mm face height, 23 mm body depth |
| `wide-deep` | 42 x 82 mm face, 38 mm body, 7 mm corners, 36 mm spacing |

An independent revision selected NEMA 5-20 and a 34 mm rear body. A 60 mm face
width was rejected before provider execution.

## Test Matrix

| Check | Result | Evidence |
| --- | --- | --- |
| Provider registration | Pass | Disabled by default; requires endpoint, bearer secret, rights assertion, and explicit automated-use authorization |
| Capability discovery | Pass | Qualified deployment explicitly exposed text, revision, export, semantic parameters and parts, assertions, native provenance, and a three-variant bound |
| Helm parity | Pass | Chart defaults fail closed and expose every ForgeCAD capability and timeout setting |
| Text generation | Pass | Keyboard and outlet returned immutable `geometry.source.v1` bundles |
| Semantic parameters | Pass | 9 keyboard and 7 outlet controls with types, units, bounds, steps, effects, and affected parts |
| Parameter families | Pass | 3/3 keyboard and 3/3 outlet variants; every row had a distinct revision |
| Invalid admission | Pass | Invalid families failed as `parameter_family_invalid` without creating provider state |
| Revision | Pass | Both objects produced a new immutable source revision |
| Explicit export | Pass | Two repeated STEP and GLB exports retained each original source revision |
| Deterministic replay | Pass | Repeated STEP bytes matched; repeated GLB bytes matched |
| Source selection | Pass | Exact root STEP `representation-003` selected; GLB retained as secondary render geometry |
| Native integrity | Pass | Keyboard 7/7 and outlet 11/11 verifications; exact collision checks found zero collisions and skipped zero candidate pairs |
| STEP-to-USD | Pass | Pinned `usd-convert-cad==0.2.0` converted the authoritative STEP |
| USD preflight | Pass | All four retained cases had zero boundary, non-manifold, degenerate, duplicate, and non-triangular faces |
| Axis semantics | Pass | `forward_axis=-Y` produced axis-correct semantic OVRTX views rather than fixed `+Y` labels |
| OVRTX evidence | Pass | Four six-view path-traced packets, 64 sensor updates, verified settings and renderer identity, digest-bound images, and passing render audits |
| Provider isolation | Pass | ForgeCAD runtime, API key, and license were absent from Geometry Agent |
| Runtime/physics/material/SimReady | Not evaluated | Downstream scope |

## Retained Results

Final source results:

| Object | Selected source | Parts | Parameters | Assertions |
| --- | --- | ---: | ---: | ---: |
| Keyboard | `representation-003`, STEP | 5 | 9 | 4 |
| Outlet | `representation-003`, STEP | 7 | 7 | 4 |

Final Geometry and OVRTX results:

| Case | Meshes | Triangles | USD/render result |
| --- | ---: | ---: | --- |
| Keyboard default | 73 | 33,616 | Conditional handoff; checks and OVRTX audit passed |
| Keyboard `wide-108` | 105 | 49,112 | Conditional handoff; checks and OVRTX audit passed |
| Outlet default | 9 | 5,362 | Conditional handoff; checks and OVRTX audit passed |
| Outlet `single-socket` | 7 | 4,416 | Conditional handoff; checks and OVRTX audit passed |

Every accepted image binding retained the complete source USD digest and
verified executed settings. Generated files remain in the operator-selected
workspace; they are intentionally not tracked in source control.

Manual review confirmed that:

- keyboard dimensions, row/column topology, feet, cable recess, and corner
  changes are visible and coherent;
- the front view does not show the rear cable recess, while the back view does;
- outlet receptacle openings remain open and correctly positioned;
- the single-socket topology removes the second receptacle and moves the center
  fastener to a supported location;
- the center screw has a seated head and connected shank rather than a floating
  disk; and
- no case is clipped and no unexplained detached artifact is visible.

## Performance Observations

The evaluated licensed execution was serialized. A cold keyboard generation or
revision took about 73-75 seconds because it included model execution, exact
integrity inspection, STEP/GLB export, and artifact validation. A verified
immutable cache replay took about 1.3-2.0 seconds. The unchanged outlet family
replayed three cached variants in 3.8 seconds total, while three newly changed
keyboard variants took 224 seconds.

The cache is keyed by provider revision, requested formats, worker
implementation digest, and selected model digest. Every hit rechecks regular
file type, size, and SHA-256. It does not trust a filename or revision string
alone. This also makes repeated export byte-stable despite a timestamp written
by the native STEP exporter.

## Defects Found and Corrected

### Geometry Agent changes in this repository

1. ForgeCAD service and Helm defaults previously implied broad provider
   capabilities. The defaults now expose text-to-STEP only. Revision, export,
   semantic parameters, definitions, parts, assertions, native provenance,
   family size, formats, and timeouts require explicit operator configuration.
2. The generic delegated connector also defaulted rich semantic claims on.
   Those claims now default off.
3. Provider parameter metadata arrived as a JSON sidecar, while supporting
   assets previously rejected all JSON. Geometry Agent now accepts only bounded
   `application/json`: at most 4 MiB, standard finite UTF-8 JSON, object/array
   root, unique object keys, at most 32 levels and 100,000 values, and bounded
   key lengths.
   Executable native-source types remain a separate inert provenance role.
4. Six-view evidence used fixed axis labels and ignored the retained provider
   forward axis. A valid `-Y`-forward source was therefore shown with front/back
   and left/right reversed. Camera planning now derives semantic directions
   from retained up, forward, and handedness metadata, records the exact axis
   plan, and fails accepted evidence when retained coordinates cannot be
   resolved.
5. Stage metric normalization now retains the transformed canonical forward
   axis so later rendering remains correct after an up-axis conversion.

### Reference worker and model corrections outside this repository

- selected exact root STEP before secondary GLB and bound the selected
  representation ID through the handoff;
- serialized licensed execution and added bounded cold-run timeouts;
- added immutable export caching and model-scoped implementation identity;
- corrected source coordinates from the assumed `+Y` to the modeled `-Y`;
- replaced a fixed component limit with a topology-derived bound;
- rejected stale immutable revisions after model implementation changes;
- removed silent corner-radius saturation and tested minimum and maximum
  parameter boundaries;
- made the outlet fastener topology-aware, with a face bore, body pilot, shank,
  seated head, and explicit seating/clearance checks; and
- added connector/match and physical-component contracts so the current native
  mechanical-integrity reports pass as complete assemblies, rather than merely
  passing individual assertions.

## Deployment and Reproduction

Use a clean checkout, Python 3.12, `uv`, Docker, an OVRTX configuration, and an
authorized ForgeCAD deployment. No step below depends on a machine-specific
checkout path.

Build the public contract and connector wheels:

```bash
mkdir -p dist/provider-wheels
uv build --wheel --project agentic/packages/geometry_authoring_contracts \
  --out-dir dist/provider-wheels
uv build --wheel --project agentic/packages/geometry_authoring_connectors \
  --out-dir dist/provider-wheels
```

In a separate provider project, implement an authenticated fixed endpoint that
accepts `ExternalAuthoringWorkerRequest` and returns
`ExternalAuthoringWorkerResult`. Keep all native model evaluation, credentials,
license material, caches, and provider source interpretation inside that
deployment. The worker result must bind every artifact by size and SHA-256 and
must declare the actual coordinate system.

Configure Geometry Agent only after qualifying the worker:

```bash
export GEOMETRY_AGENT_SERVICE_API_KEY="replace-with-service-secret"
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_ENDPOINT_URL="https://provider.example/v1/author"
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_BEARER_TOKEN="replace-with-provider-secret"
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_RIGHTS_ASSERTION="Authorized provider output"
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTOMATED_USE_AUTHORIZED=true
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTED_FORMATS='["step","glb"]'
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_TEXT=true
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_REVISION=true
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_EXPORT=true
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_SEMANTIC_PARAMETERS=true
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_PARAMETER_DEFINITIONS=true
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_SEMANTIC_PARTS=true
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_PROVIDER_ASSERTIONS=true
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_MAX_FAMILY_VARIANTS=3
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_RETURNS_NATIVE_SOURCE=true
export GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_READ_TIMEOUT_SECONDS=900

uv run --project apps/geometry_agent_service --frozen --no-sync \
  geometry-agent-service --host 127.0.0.1 --port 8776
```

Use `geometry_agent_client.GeometryAgentClient` to call `providers`,
`generate`, `family`, `revise`, `export`, and `run`. For this matrix, request
STEP and GLB; require the selected representation to be exact STEP with role
`design_exchange`; request a `six_view` OVRTX render; and disable runtime
validation so the result is not confused with SimReady certification.

Run the affected repository checks:

```bash
uv run --project agentic/packages/geometry_authoring_connectors \
  --frozen --extra dev pytest -q -o addopts='' \
  agentic/packages/geometry_authoring_connectors/tests
uv run --project apps/geometry_agent_service --frozen --no-sync \
  pytest -q -o addopts='' apps/geometry_agent_service/tests
uv run --project apps/geometry_agent_service --frozen --no-sync \
  pytest -q -o addopts='' \
  agentic/packages/content_agent_workflows/tests/test_geometry_shared_boundaries.py \
  agentic/packages/content_agent_workflows/tests/test_geometry_workflow.py \
  agentic/packages/content_agent_workflows/tests/test_geometry_audit_artifact_paths.py
```

## Remaining Gaps

### Provider Product and Protocol

- The evaluated worker routes two known templates. General prompt-to-ForgeCAD
  synthesis, image-conditioned authoring, and arbitrary edit quality are not
  demonstrated.
- Capabilities are explicitly configured by an operator rather than fetched
  from an authenticated worker capability handshake. Result validation catches
  many overclaims, but discovery can still be wrong when deployment settings
  are wrong.
- ForgeCAD 0.13 did not expose `forgecad check params` in the tested runtime,
  although the public kit documentation listed that command. This evaluation
  used focused `forgecad run -p` boundary cases instead.
- A single revision capability still covers semantic-parameter and free-form
  revisions. A later contract should advertise those separately.

### Geometry Handoff

- Semantic groups bind the root STEP representation, not stable STEP subshape
  identities. Fine-grained part bindings need a provider representation that
  preserves subshape identity or separate per-part design-exchange artifacts.
- Exact STEP conversion does not retain the authored ForgeCAD material styling.
  The secondary GLB preserves visual styling but is not authoritative geometry.
- Scene Optimizer was unavailable in this portable run. The workflow therefore
  retained a normalized copy and correctly reported a conditional handoff.
- Native provider assertions are evidence, not authority. Geometry Agent still
  performs its independent USD and render checks.

### Downstream Scope

The keyboard is not a simulated input device, and the outlet is not an
electrical model. Collision authoring, mass and friction, key travel and joints,
materials and labels, electrical serviceability, runtime task behavior, and
formal SimReady qualification remain downstream work.
