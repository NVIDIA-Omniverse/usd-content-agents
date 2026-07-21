# Workbench Agent API Reference

Workbench operations are rooted at `/sessions`. The `/agent/*` namespace is
reserved for discovery and compact agent-facing metadata, not session
operations.

## Discovery

```text
GET /agent-api
GET /agent-api.json
GET /openapi.json
GET /agent/capabilities
GET /agent/openapi.json
GET /agent/tool-manifest
```

## Core Operations

```text
POST   /sessions
GET    /sessions/{session_id}
DELETE /sessions/{session_id}
POST   /sessions/{session_id}/scene
POST   /sessions/{session_id}/scene/optimize
POST   /sessions/{session_id}/scene/restore
POST   /sessions/{session_id}/scene/snapshot
GET    /sessions/{session_id}/optimization
POST   /sessions/{session_id}/paths/translate
POST   /sessions/{session_id}/paths/translate:batch
POST   /sessions/{session_id}/render
POST   /sessions/{session_id}/pick
POST   /sessions/{session_id}/commands
GET    /sessions/{session_id}/authoring/material-assignments
POST   /sessions/{session_id}/authoring/material-assignments:apply
POST   /sessions/{session_id}/render-frames
POST   /sessions/{session_id}/physics/inspect-mesh-candidates
POST   /sessions/{session_id}/physics/inspect-components
POST   /sessions/{session_id}/physics/inspect-topology
POST   /sessions/{session_id}/physics/apply-topology-plan
POST   /sessions/{session_id}/physics/apply-schema
POST   /sessions/{session_id}/physics/validate-runtime
```

Future stable scene/object/edit resources should stay under
`/sessions/{session_id}/...`.

## Current Material Override Command

Use current `material_override` commands only as the compatibility mechanism for
material edits until edit transactions are available:

```json
{
  "command": "material_override",
  "payload": {
    "prim_path": "/World/Mesh",
    "space": "inspection",
    "unbind_existing": true,
    "material": {
      "source": "material_library",
      "library_path": "/path/to/materials.usd",
      "material_path": "/World/Looks/Rubber_Black",
      "material_name": "Rubber Black"
    }
  }
}
```

## Restore And Export

Use Workbench restore/export as the atomic finalization operation for accepted
edits:

```json
{
  "output_usd_path": "/absolute/path/to/asset_restored.usda",
  "output_mode": "layer",
  "material_profile": "preview_surface",
  "overwrite": false,
  "include_preview_artifact": true
}
```

Call `POST /sessions/{session_id}/scene/restore`. In the preview build,
material overrides are projected to source-space USD outputs through the durable
material apply path. View-only edits remain in the returned preview artifact and
are reported as warnings.

## Physics Operations

Physics workflows should use Workbench physics operations rather than importing
fixed-pipeline helpers or solver daemons directly:

```text
POST /sessions/{session_id}/physics/inspect-components
POST /sessions/{session_id}/physics/inspect-topology
POST /sessions/{session_id}/physics/apply-topology-plan
POST /sessions/{session_id}/physics/apply-schema
POST /sessions/{session_id}/physics/validate-runtime
```

Use logical components for new workflows. `inspect-mesh-candidates` is a V1
compatibility operation and must not be promoted to one physics decision per
mesh. Component inspection separates visual evidence, collider targets, helper
geometry, rigid-body roots, and joints. Topology plans are digest-bound,
intent-gated derivative mutations; preserve topology when mobility intent is
ambiguous.

`apply-schema` defaults to authoring a dynamic default-prim rigid body. Pass
`author_rigid_body=false` only when a prior topology plan has an explicit static
mobility intent and the desired result is collider/material authoring without a
dynamic rigid body.

`validate-runtime` may use ovphysx internally. The solver process and any
OpenUSD-version isolation are Workbench implementation details; the agent should
consume returned artifacts, metrics, failures, and repair hints.
Agent-supplied output paths for apply-topology-plan, apply-schema,
validate-runtime, and render-frames are optional; when present, they must resolve
inside the Workbench session workspace. Prefer omitting them and using returned
artifact paths or URLs.

Render frame sequences, including time-sampled USD recordings, through the
general Workbench frame sequence API:

```text
POST /sessions/{session_id}/render-frames
```

## Artifact Discipline

Every workflow should preserve:

- request JSON;
- scene/session metadata;
- render artifacts and camera JSON;
- pick results;
- edit or command records;
- validation reports;
- restore/export outputs;
- trace/events where available.
