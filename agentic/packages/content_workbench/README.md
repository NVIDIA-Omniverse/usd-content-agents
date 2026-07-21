# Content Workbench

Local scene workbench service for agents operating on USD content.

New users evaluating the research-preview agentic asset workflow should start
from [`../../README.md`](../../README.md). This README is the
Workbench service reference: API surface, sidecar behavior, configuration, and
tests. The preview workflow uses Workbench through skills under
`agentic/.agents/skills`, not by asking users to memorize these endpoints.

This implementation provides the stateful control plane, USD grounding layer,
and a service-owned OvRTX still-render slice:

- create and close local scene sessions;
- load local `.usd`, `.usda`, `.usdc`, and `.usdz` files;
- query hierarchy, prim properties, bounds, material bindings, and offline
  diagnostics;
- maintain scene inspection state for selection, hide/show, isolate, active
  AOV, and material overrides;
- export a viewer-owned preview layer that sublayers the source USD and authors
  stronger session opinions for material and visibility changes;
- bind preview edits either to generated `UsdPreviewSurface` materials or to
  actual material prims from a USD material library;
- maintain an agent-controllable camera with frame/orbit/pan/dolly/set-camera
  operations;
- pick viewport pixels with native OvRTX pick queries;
- render native OvRTX selection outlines for selected mesh prims;
- render the current preview scene through the workbench's own `ovrtx`
  renderer, returning a PNG and camera metadata.

Rendering, pixel picking, and selection outlines are owned by this service and
serialized through one Workbench OvRTX worker. Each native operation runs in a
bounded child process, and the service does not call the existing repository
rendering CLI.

Each renderer child configures OvRTX environment variables such as
`OVRTX_RENDER_MODE`, `OVRTX_BIN_PATH`, and `LD_LIBRARY_PATH`, then exits after
the operation so native renderer memory cannot accumulate in the long-lived
Workbench process. Run Workbench as its own local sidecar service.

The first render or pick provisions the reviewed, hash-locked OvRTX runtime at
`~/.cache/wu/ovrtx_venv` (override with `WU_OVRTX_VENV_DIR`). The `uv`
executable must be available for this one-time setup. Set
`WU_OVRTX_AUTO_PROVISION=0` to require a pre-provisioned runtime; Workbench
fails closed when the exact managed runtime is unavailable.

Session artifacts are written under `CONTENT_WORKBENCH_WORKSPACE_DIR` when set.
The legacy aliases `SCENE_INSPECTOR_WORKSPACE_DIR` and `RSI_WORKSPACE_DIR` are
still honored for transition but should not be used by new code. If none are
set, the service uses a per-user runtime directory:
`$XDG_RUNTIME_DIR/content-workbench` when available, otherwise
`<tempdir>/content-workbench-<uid>`.

Set `CONTENT_WORKBENCH_OUTPUT_ROOTS` to a comma-separated list of canonical
directories to confine caller-selected material-apply and scene-restore output
paths. The Workbench rejects outputs outside those roots before material
authoring begins. `content-workflow-cli` starts its sidecar with exactly the
current run directory as the sole output root and rejects an already-running
service whose health metadata does not advertise that exact confinement.

## Run

From the repository root:

```bash
PYTHONPATH=agentic/packages/content_workbench \
  python -m uvicorn content_workbench.main:app \
  --host 127.0.0.1 \
  --port 8088
```

Health check:

```bash
curl http://127.0.0.1:8088/healthz
```

Canonical agent API docs:

```bash
curl http://127.0.0.1:8088/agent-api
curl http://127.0.0.1:8088/agent-api.json
curl http://127.0.0.1:8088/openapi.json
```

The canonical repository copy is
`agentic/packages/content_workbench/docs/agent_api.md`. Wrappers such as
`content-workflow-cli` should pass only the workbench endpoint; child
agents can discover the API from `/agent-api` and `/openapi.json`.

`content-workbench` is model-agnostic. The companion
`content-workflow-cli` CLI can drive the service with either supported
child runner, `codex` or `claude`, through TypeScript SDK bridges. Claude
support is configured in `agentic/packages/content_workflow_cli`; the
Workbench API and sidecar lifecycle are unchanged.

Create a session:

```bash
curl -X POST http://127.0.0.1:8088/sessions \
  -H 'content-type: application/json' \
  -d '{"scene_path":"/path/to/scene.usd"}'
```

Query material binding:

```bash
curl 'http://127.0.0.1:8088/sessions/<session_id>/material-binding?prim_path=/RootNode'
```

Move the agent viewport:

```bash
curl -X POST http://127.0.0.1:8088/sessions/<session_id>/commands \
  -H 'content-type: application/json' \
  -d '{"command":"frame","payload":{"prim_path":"/RootNode/Geometry"}}'

curl -X POST http://127.0.0.1:8088/sessions/<session_id>/commands \
  -H 'content-type: application/json' \
  -d '{"command":"orbit","payload":{"yaw_delta_degrees":20,"pitch_delta_degrees":-8}}'

curl -X POST http://127.0.0.1:8088/sessions/<session_id>/commands \
  -H 'content-type: application/json' \
  -d '{"command":"dolly","payload":{"amount":-0.5}}'

curl http://127.0.0.1:8088/sessions/<session_id>/camera
```

Pick the current viewport and update selection:

```bash
curl -X POST http://127.0.0.1:8088/sessions/<session_id>/pick \
  -H 'content-type: application/json' \
  -d '{"x":420,"y":240,"width":1024,"height":768,"update_selection":true}'
```

Preview a material override and render it:

```bash
curl -X POST http://127.0.0.1:8088/sessions/<session_id>/commands \
  -H 'content-type: application/json' \
  -d '{
    "command": "material_override",
    "payload": {
      "prim_path": "/RootNode/Geometry",
      "unbind_existing": true,
      "material": {
        "display_name": "BrightGray",
        "diffuse_color": [0.72, 0.72, 0.72],
        "roughness": 0.45
      }
    }
  }'

curl -X POST http://127.0.0.1:8088/sessions/<session_id>/render \
  -H 'content-type: application/json' \
  -d '{"width":1024,"height":768,"render_quality":"inspection"}'
```

The render response includes service-local `image_path` and an `image_url`.
Fetch `image_url` from the Workbench endpoint when a caller should not read the
service filesystem directly.

Render requests default to the studio HDRI rig at intensity `600.0`, with no
plain dome or distant light. Workbench deactivates lights authored by the
inspected scene in its transient viewer layer, so still renders, frame renders,
and picks use only the default HDRI or an explicitly requested Workbench light
rig without modifying the source USD.

`material_override.payload.material` must be a JSON object. Bare string
material names are rejected so callers do not get an accidental gray fallback.

Use `render_quality: "final"` for final verification renders. The final preset
requests OvRTX `rt2` mode with 256 sensor updates so normal workflows stay on
one renderer backend across evidence renders, final renders, picks, and outline
renders. Callers can override the update count with `ovrtx_num_sensor_updates`;
render-mode overrides should be reserved for targeted diagnostics.
Render, pick, and screenshot dimensions must be between 1 and 8192 pixels per
axis.

Bind an actual material from a USD material library:

```bash
curl -X POST http://127.0.0.1:8088/sessions/<session_id>/commands \
  -H 'content-type: application/json' \
  -d '{
    "command": "material_override",
    "payload": {
      "prim_path": "/World/AGV__U3A__16/Lift/AGVtop",
      "unbind_existing": true,
      "material": {
        "source": "material_library",
        "library_path": "/path/to/materials_libs_v2.usd",
        "material_path": "/World/Looks/Steel_Painted_Orange",
        "material_name": "Steel Painted Orange"
      }
    }
  }'
```

Material library paths are trusted local file references. The Workbench resolves
them on the host and sublayers the referenced USD into the session preview
stage. When `CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS` is unset, library paths
are limited to the loaded source scene's directory. Set
`CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS` to a comma-separated list of allowed
host directories when agents need to bind materials from another local library.
`content-workflow-cli` injects this allowlist only when it starts the
Workbench sidecar itself; already-running or remote Workbench services must be
started with the material-library directory explicitly allowlisted.
Expose the service only to callers that are allowed to read those files or place
it behind an operator-controlled allowlist/proxy.

## Configuration

Prefer `CONTENT_WORKBENCH_HOST` and `CONTENT_WORKBENCH_PORT` when launching the
service through `python -m content_workbench.main`. Legacy `SCENE_INSPECTOR_*`
and `RSI_*` host/port variables are still accepted after the preferred names,
in that order.

Browser CORS defaults to localhost origins only. CORS is not authentication:
non-browser clients can still call the API if they can reach the port. Set
`CONTENT_WORKBENCH_CORS_ORIGINS` to a comma-separated list of explicit origins,
or `CONTENT_WORKBENCH_CORS_ORIGIN_REGEX` to replace the default localhost regex,
only for trusted local networks or behind an authenticated proxy.

## Test

```bash
PYTHONPATH=agentic/packages/content_workbench \
  pytest agentic/packages/content_workbench/tests -q --no-cov
```
