# External Build123d Integration Guide and Evaluation

## Outcome

Geometry Agent supports a rich Build123d authoring product through the public
provider boundary without installing or importing Build123d in Geometry Agent.
The evaluated provider generated parameterized keyboard and outlet families,
preserved revision lineage and semantic parts, exported STEP and STL, and
completed STEP-to-USD validation plus OVRTX visual evidence. The resulting
Geometry handoffs are conditional, not SimReady-certified: runtime, physics,
materials, articulation, and formal SimReady validation were intentionally not
requested. A second clean replay regenerated both objects and repeated the
complete provider-to-Geometry workflow successfully.

This document records that historical qualification run. The public 0.6
Geometry package and service image now defer CAD/mesh conversion because the
evaluated converter's embedded Python runtime does not meet the release image
scan policy. Provider STEP and STL remain valid immutable handoffs, but replaying
the conversion requires a separately qualified converter deployment or a
provider-produced USD-family representation.

## External-User Topology

The test used only the public `geometry-authoring-contracts` and
`geometry-authoring-connectors` wheels in a separately created provider
project. The provider runtime used Build123d 0.10.0 in its own Docker image;
the Geometry Agent process and image contained neither Build123d nor CadQuery.

The provider deployment applied these controls:

- authenticated fixed endpoint `/v1/author`;
- no provider endpoint discovery or URL supplied by a request;
- one fixed command selected by the operator, with no shell;
- Docker network disabled and root filesystem read-only;
- non-root UID/GID, all capabilities dropped, and no-new-privileges;
- bounded CPU, memory, process count, timeout, state, and workspace;
- digest- and size-bound results returned as `geometry.source.v1`; and
- provider-native `model.py` retained as inert provenance, never executed by
  Geometry Agent.

The Build123d worker image required Debian `libgl1` and `libxrender1` for its
headless native import. Those libraries and Build123d remain provider-owned
dependencies and are not added to Geometry Agent.

## Evaluated Objects

### Parameterized Keyboard

Prompt:

> Create a parameterized 65 percent mechanical keyboard with a rigid rounded
> case, switch plate, separate dished keycaps, and rear typing feet.

The provider returned eight semantic controls: layout, key pitch, key gap,
bezel, case height, keycap height, case corner radius, and typing angle. The
default model has 68 solids and root plus case, switch-plate, keycap, and
rear-feet semantic bindings.

Three family rows exercised a compact 65-percent layout, a 75-percent layout,
and a TKL layout with a wider bezel. Each produced a distinct immutable source
revision. An out-of-range key gap was rejected by Geometry Agent before a
provider call.

### Parameterized Outlet

Prompt:

> Create a parameterized North American NEMA 5-15 duplex wall outlet with a
> rounded face, true blade and ground openings, mounting yoke, center screw,
> and side terminals.

The provider returned seven semantic controls: NEMA 5-15/5-20 standard, face
width and height, body depth, receptacle spacing, corner radius, and socket
count. The default model has 10 solids and root plus insulating-body,
mounting-yoke, front-face, receptacle, center-screw, and terminal bindings.

Three family rows exercised a NEMA 5-20 duplex, a single receptacle, and a
wider/deeper duplex body. Each produced a distinct valid source revision.

## Test Matrix

| Check | Result | Evidence |
| --- | --- | --- |
| Provider discovery | Pass | Explicit capabilities exposed only for configured formats and features |
| Helm capability parity | Pass | Deployed settings preserve the same fail-closed defaults and explicit overrides |
| Text generation | Pass | Keyboard and outlet returned immutable bundles |
| Semantic parameters | Pass | 8 keyboard and 7 outlet controls with types, units, ranges, choices, groups, effects, and affected parts |
| Parameter families | Pass | 3/3 keyboard and 3/3 outlet variants |
| Invalid family admission | Pass | Out-of-range value rejected as `parameter_family_invalid` before provider execution |
| Semantic parts | Pass | Root and grouped per-part STEP bindings retained |
| Provider assertions | Pass | Native shape validity and complete parameter-state assertions |
| Explicit export | Pass | STEP plus STL export retained the immutable source revision |
| Mixed-format source selection | Pass | Exact root STEP selected; STL retained as secondary render geometry |
| CAD-to-USD conversion | Pass | Pinned `usd-convert-cad==0.2.0` converted provider STEP |
| USD mesh preflight | Pass | Keyboard 69,642 triangles; outlet 3,544 triangles; no boundary, over-connected, duplicate, degenerate, or non-triangular faces |
| OVRTX evidence | Pass | Six individual OVRTX views plus derived presentation grid; render audit passed |
| Service image boundary | Pass | Build123d and CadQuery were not importable in the Geometry Agent image |
| Runtime/physics/material/SimReady | Not evaluated | Outside this geometry-authoring integration test |

An earlier mixed-format keyboard run selected the provider STL and failed the
250,000-triangle preflight with 401,674 triangles. The correction selects the
exact root STEP, which converted to 69,642 triangles and passed without
relaxing the shared quality budget.

Final retained evidence from the current provider implementation:

| Object | Generation job | Geometry job | Selected source | OVRTX presentation |
| --- | --- | --- | --- | --- |
| Keyboard | `job_c5cdbb680488403eaa324e5478ccb0e3` | `job_22bc9d0505a74526b45dea465337e902` | `representation-002`, STEP | `executions/job_22bc9d0505a74526b45dea465337e902/output/render_evidence/geometry_six_view.png` |
| Outlet | `job_3071ac978b5c4c04837146e93b7fe853` | `job_0c9eabb2c60741b6be5fbef34a8149ce` | `representation-002`, STEP | `executions/job_0c9eabb2c60741b6be5fbef34a8149ce/output/render_evidence/geometry_six_view.png` |

Both geometry jobs succeeded with `handoff_ready="conditional"`; all evaluated
geometry, USD, OVRTX, and deterministic audit checks passed. Conditional status
is due to the intentionally skipped runtime/SimReady checks and unavailable
Scene Optimizer in this final portable run, not a failed geometry check.

Clean replay evidence:

| Object | Selected source | Result |
| --- | --- | --- |
| Keyboard | Exact root STEP | Geometry workflow succeeded and returned a conditional handoff |
| Outlet | Exact root STEP | Geometry workflow succeeded and returned a conditional handoff |

The replay also verified reference-bound artifact IDs in the form
`sha256:<content-digest>:<reference-digest>` throughout source publication and
workflow evidence.

## Deployment and Verification

Use Python 3.12, `uv`, Docker, and a working OVRTX configuration. Start from a
clean checkout of the target branch; no instruction below depends on a
machine-specific source path.

Build the public provider wheels:

```bash
mkdir -p dist/provider-wheels
uv build --wheel --project agentic/packages/geometry_authoring_contracts \
  --out-dir dist/provider-wheels
uv build --wheel --project agentic/packages/geometry_authoring_connectors \
  --out-dir dist/provider-wheels
```

In a separate provider project, install those wheels and Build123d. Implement a
fixed authenticated `/v1/author` endpoint using `Build123dWorkerRunner` and a
`CommandAuthoringExecutionBackend`. The command must return a bounded
`ExternalAuthoringWorkerResult`; keep model evaluation, native state, and
Build123d entirely inside that trusted provider deployment.

Prepare and launch Geometry Agent:

```bash
OCP_DIRECT_ALLOW_NETWORK_SOURCE_FETCH=1 \
  CMAKE_BUILD_PARALLEL_LEVEL=2 \
  uv sync --project apps/geometry_agent_service --extra dev

export GEOMETRY_AGENT_SERVICE_API_KEY="replace-with-service-secret"
export GEOMETRY_AGENT_SERVICE_WORKSPACE_ROOT="$PWD/.geometry-agent-workspace"
export GEOMETRY_AGENT_SERVICE_BUILD123D_ENDPOINT_URL="http://127.0.0.1:8765/v1/author"
export GEOMETRY_AGENT_SERVICE_BUILD123D_BEARER_TOKEN="replace-with-worker-secret"
export GEOMETRY_AGENT_SERVICE_BUILD123D_RIGHTS_ASSERTION="Authorized provider output"
export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTED_FORMATS='["step","stl"]'
export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_REVISION=true
export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_EXPORT=true
export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_SEMANTIC_PARAMETERS=true
export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_PARAMETER_DEFINITIONS=true
export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_SEMANTIC_PARTS=true
export GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_PROVIDER_ASSERTIONS=true
export GEOMETRY_AGENT_SERVICE_BUILD123D_MAX_FAMILY_VARIANTS=16

uv run --project apps/geometry_agent_service --frozen --no-sync \
  geometry-agent-service --host 127.0.0.1 --port 8776
```

Use `geometry_agent_client.GeometryAgentClient` to call `generate`, `family`,
`export`, and `run`, polling each returned job with `wait`. Request both `step`
and `stl`; confirm `selected_representation_id` points to the root
`design_exchange`, then run with `render_evidence=true`,
`render_preset="six_view"`, and runtime validation disabled for this focused
geometry test.

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

## Integration Requirements Verified

1. Build123d previously advertised broad formats and export behavior by
   default. Defaults now fail closed at text-to-STEP generation; every richer
   capability is an explicit operator setting, and the Helm chart exposes the
   same settings rather than restoring the old broad defaults at deployment.
2. The historical evaluation used `usd-convert-cad==0.2.0` for CAD/mesh-to-USD
   handoff. Public 0.6 deliberately excludes that distribution and its embedded
   runtime from the Geometry package, lock, and service image. Native provider
   outputs remain immutable handoffs, and downstream execution fails closed
   unless an independently qualified converter or provider-produced USD is
   available.
3. Provider publication selected lossy render geometry before exact design
   exchange. It now prefers exact design and, when available, the unique
   representation bound to a root semantic part.
4. The service stored the selected representation ID but did not pass it into
   the shared Geometry workflow. Mixed bundles could therefore be re-selected
   differently and rejected as ambiguous. Runs now forward the exact ID.
5. Six-view audit applied 3-by-2 framing analysis to one accepted square OVRTX
   view, producing false clipping failures. Individual-image health and
   presentation-grid framing are now separate checks.

The external evaluation provider also exposed and corrected two provider-side
bugs: a grounding-hole cylinder used the wrong axial alignment, and immutable
revision IDs did not include the modeling implementation fingerprint. These
are examples of checks a production provider must own; provider-native modeling
code remains outside Geometry Agent.

## Remaining Gaps

### Provider Product Gaps

- The evaluation worker routes two known object families. It proves the public
  contract, not general prompt-to-Build123d synthesis quality.
- Provider implementation identity is bound into the revision, but byte-level
  deterministic re-export is not yet achieved. Two generation calls for the
  same keyboard semantic revision produced STEP digests `c71f51fc...` and
  `668fbca7...`; Build123d/OpenCascade serialization can vary even when geometry
  state does not. A production provider must cache immutable revision exports
  or canonicalize them and bind a geometry-level digest before claiming
  deterministic re-export. The public boundary verifies each returned byte set
  and its lineage but cannot prove how a remote provider recreated it.
- Capabilities are operator configuration, not an authenticated capability
  handshake. Incorrect deployment configuration fails at request/result
  validation, but discovery can still overstate a misconfigured worker.
- One revision capability flag covers both semantic-parameter revisions and
  free-form text/image revisions. A later contract should advertise those
  independently.

### Geometry Handoff Gaps

- Semantic parts bind representation files and optional USD root prims; STEP
  subshape identities are not represented. This test works around that limit
  with separate per-part STEP files.
- Bundles without one unique root semantic-part binding still use deterministic
  provider order within the highest-fidelity role. A future schema should carry
  an explicit primary representation ID.
- The STEP-converted keyboard passes edge-manifold preflight, but shared USD
  validation reports repeated non-manifold-vertex warnings on the dished
  keycaps. This needs a focused tessellation/validator investigation before a
  production keyboard claim.
- `usd-convert-cad` occupied about 133 MiB and carried an embedded Python runtime
  in the evaluated environment. It is not shipped in public 0.6. A future
  converter sidecar or dedicated CAD image profile needs independent runtime,
  license, notice, SBOM, and image-scan qualification before release.
- Scene Optimizer is optional and was not available in the final portable
  rerun, so the handoff correctly reports `conditional` with a normalized copy.
  A separate run with the approved local Scene Optimizer resource completed and
  proved exact geometry and surface correspondence with zero measured drift.

### Downstream Scope

The geometry-only OVRTX renders intentionally have neutral materials and no key
legends. Physics properties, key joints and response curves, electrical outlet
serviceability, collision authoring, materials, labels, runtime simulation, and
formal SimReady conformance belong to their respective downstream workflows and
remain unproven here.

## Generated Evidence

Provider state, service job records, exported geometry, USD handoffs, and OVRTX
renders are generated outputs and are intentionally not tracked in this
repository. Store them in an operator-selected workspace and retain the source
bundle, job record, validation evidence, and six-view render together when the
result is used as qualification evidence.
