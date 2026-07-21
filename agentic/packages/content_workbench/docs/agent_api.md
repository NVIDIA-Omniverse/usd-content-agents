# Content Workbench Agent API

Canonical agent-facing document for the Content Workbench service.

Agents should consume this document from one of these stable locations:

- repository file: `agentic/packages/content_workbench/docs/agent_api.md`
- running service: `GET /agent-api`
- machine-readable discovery: `GET /agent-api.json`
- OpenAPI schema: `GET /openapi.json`

Wrappers such as `content-workflow-cli` should pass the workbench endpoint, not a
custom documentation path. A child agent can fetch:

```text
<workbench-endpoint>/agent-api
<workbench-endpoint>/openapi.json
```

`/agent` is reserved for discovery documents and compact agent-facing metadata.
Scene, render, authoring, physics, and restore operations use the canonical
`/sessions/{session_id}/...` API surface.
The superseded `/agent/sessions/*` draft routes are not compatibility aliases;
callers must use the canonical session routes.

## Service Role

Content Workbench is a stateful scene and asset state service for agents
working on local USD content. It is not an agent runner and does not decide
materials, geometry edits, physics labels, or task strategy.

The workbench owns:

- USD scene/session loading;
- hierarchy, property, bounds, diagnostics, and material-binding queries;
- persistent camera, selection, visibility, isolation, and viewport state;
- native OvRTX rendering;
- viewport pixel picking;
- selection highlighting in rendered views;
- non-destructive material and visibility edits;
- durable material apply to caller-selected output USD or USDZ files;
- physics candidate inspection, USD physics schema apply, runtime validation,
  and general frame-sequence render operations;
- one serialized Workbench-owned OvRTX render/pick path.

Session artifacts are written under `CONTENT_WORKBENCH_WORKSPACE_DIR` when
configured. The legacy aliases `SCENE_INSPECTOR_WORKSPACE_DIR` and
`RSI_WORKSPACE_DIR` are still honored for transition but should not be used by
new code. If none are configured, the service uses
`$XDG_RUNTIME_DIR/content-workbench` when available, otherwise a per-user temp
directory named `content-workbench-<uid>`.

The calling agent owns:

- task planning;
- semantic decisions;
- reference-image interpretation;
- material/geometry/physics/layout choices;
- requested output paths;
- final reports and acceptance criteria.

Security model: Content Workbench is a trusted local sidecar. It has no
authentication layer, CORS is not access control, and callers that can reach the
port can request local scene loads and renders. Bind it to localhost by default
or place it behind an authenticated operator-controlled proxy before exposing it
to any shared network.

## Base Workflow

1. Create a session with `POST /sessions`.
2. Snapshot broad scene state with
   `POST /sessions/{session_id}/scene/snapshot`.
3. Query targeted prims with `/tree`, `/properties`, `/material-binding`, and
   `/diagnostics`.
4. Move the camera with `POST /sessions/{session_id}/commands`.
5. Render visual evidence with `POST /sessions/{session_id}/render`.
6. Use `POST /sessions/{session_id}/pick` to ground pixels to exact prim paths.
7. Use `material_override`, visibility, selection, or isolation commands
   to test hypotheses.
8. Render verification views.
9. For accepted edits, call
   `POST /sessions/{session_id}/scene/restore`.
10. Record artifacts, render paths, and acceptance decisions in the calling run.

## Session Endpoints

### Create Session

```http
POST /sessions
content-type: application/json

{
  "scene_path": "/absolute/path/to/asset.usdz",
  "width": 1280,
  "height": 720,
  "optimize": false,
  "clear_materials": false,
  "optimization_config": {}
}
```

`scene_path` accepts local `.usd`, `.usda`, `.usdc`, and `.usdz` assets.
Set `clear_materials` to `true` to create a session-owned inspection layer that
blocks existing material bindings, display colors, and display opacity. This is
the default used by `content-workflow-cli materials assign` unless the
caller explicitly asks to respect existing material bindings.

Response includes:

- `session_id`
- `status`
- `scene_path`
- `source_scene_path`
- `inspection_scene_path`
- `root_prim_path`
- `view`
- `artifacts`
- `optimization`

### Query Session

```http
GET /sessions/{session_id}
DELETE /sessions/{session_id}
```

### Load Or Reload Scene

```http
POST /sessions/{session_id}/scene
content-type: application/json

{
  "scene_path": "/absolute/path/to/asset.usdz",
  "optimize": false,
  "clear_materials": false,
  "optimization_config": {}
}
```

Set `optimize` to `true` to run Scene Optimizer before inspection. In optimized
sessions, `source_scene_path` remains the original USD and
`inspection_scene_path` is the optimized USD used for hierarchy queries, picking,
camera operations, and rendering.

## Scene Optimizer Options

Workbench optimization uses the shared
`world_understanding.agentic.usd_tasks.optimize_usd.OptimizeUSDTask` code path.
The API fields are a thin normalization layer over `optimization_config`, then
the resolved config is passed to `OptimizeUSDTask` with:

- `input_usd_path`: original source USD.
- `output_usd_path`: session workspace optimized USD.
- `optimization_config`: merged optimizer settings.

When `flatten_prototypes` is enabled, temporary flattened input files are written
inside the session optimizer workspace, not beside the source USD. This keeps
read-only source workspaces usable.

First-class request fields:

- `optimizer_backend`: `local` or `remote`; omitted values fall back to the
  optimizer task default, and Workbench currently defaults missing backend to
  `local` before calling the task.
- `flatten_prototypes`: forwards to `optimization_config.flatten_prototypes`.
- `enable_deinstance`: forwards to
  `optimization_config.scene_optimizer_settings.enable_deinstance`.
- `enable_split`: forwards to
  `optimization_config.scene_optimizer_settings.enable_split_meshes`.
- `enable_deduplicate`: forwards to
  `optimization_config.scene_optimizer_settings.enable_deduplicate`.
- `optimization_config`: advanced pass-through for the shared optimizer task.

Explicit first-class fields override matching values inside
`optimization_config`. The raw config remains available for nested SO settings
such as `deduplicate.tolerance`, `deduplicate.ignore_attributes`,
`split_meshes.paths`, `deinstance.prim_paths`, `extract_geom_subset_indices`,
`stage_timeout`, and remote backend settings.

At least one of deinstance, split, or deduplicate must be enabled when
`optimize` is true. This prevents agents from paying the optimization cost while
requesting no SO operation.

Recommended material/visual inspection preset:

```json
{
  "optimize": true,
  "optimizer_backend": "local",
  "flatten_prototypes": true,
  "enable_deinstance": true,
  "enable_split": true,
  "enable_deduplicate": true,
  "optimization_config": {
    "scene_optimizer_settings": {
      "extract_geom_subset_indices": true
    }
  }
}
```

Use this preset when the agent needs to inspect repeated industrial or robot
assets, assign materials at a meaningful mesh/subset granularity, and recover
edits back to original source paths.

Conservative source-like preset:

```json
{
  "optimize": true,
  "optimizer_backend": "local",
  "flatten_prototypes": true,
  "enable_deinstance": true,
  "enable_split": false,
  "enable_deduplicate": false
}
```

Use this when the asset relies heavily on instances/prototypes but the agent
needs to preserve a hierarchy close to the original source.

Skip optimization when exact authored composition is the main thing being
inspected, when the asset is already small and easy to traverse, or when
temporary path remapping ambiguity would be worse than the optimizer benefit.

## Scene Optimizer Coordinate Spaces

Optimized sessions expose two coordinate spaces:

- `source`: prim paths in the original USD.
- `inspection`: prim paths in the optimized USD.

Agents normally navigate and pick in `inspection` space, then recover targets to
`source` space before recording or exporting non-destructive edits.

### Optimization State

```http
GET /sessions/{session_id}/optimization
```

Returns optimizer status, source/inspection scene paths, metadata path,
operations executed, and correspondence-map summary counts.

### Translate Paths

```http
POST /sessions/{session_id}/paths/translate
content-type: application/json

{
  "prim_path": "/World/OptimizedMesh",
  "source_space": "inspection",
  "target_space": "source"
}
```

The response always includes both `source_paths` and `inspection_paths`.
`ambiguous: true` means one path maps to multiple targets, commonly because SO
split or deduplicated geometry.

When `ambiguous` is true, agents should inspect the returned candidate paths
before recording a final decision. For material workflows, `material_override`
commands can be issued in either coordinate space; the workbench uses the
optimizer correspondence map to render the optimized inspection scene while
storing the source-space target paths.

For many paths, use the batch form:

```http
POST /sessions/{session_id}/paths/translate:batch
content-type: application/json

{
  "requests": [
    {
      "prim_path": "/World/OptimizedMeshA",
      "source_space": "inspection",
      "target_space": "source"
    },
    {
      "prim_path": "/World/OptimizedMeshB",
      "source_space": "inspection",
      "target_space": "source"
    }
  ]
}
```

The response contains `results`, a list of the same records returned by the
single-path translation endpoint.

## Scene Queries

### Snapshot

```http
POST /sessions/{session_id}/scene/snapshot
content-type: application/json

{
  "root_prim_path": "/World",
  "include_properties": true,
  "include_material_bindings": true,
  "include_path_translations": true,
  "include_candidate_hints": true,
  "max_prims": 4096
}
```

Returns a one-call snapshot for initial agent inspection:

- flattened `paths` and `nodes`;
- per-prim `properties`;
- per-prim `material_bindings`;
- inspection-to-source `path_translations`;
- visible/renderable material `candidates` as Workbench hints;
- compact `summary` counts.

`max_prims` is a shared hard limit for hierarchy paths and additional candidate
hint paths. When either portion is capped, `summary.truncated` is `true`;
`summary.candidate_hints_truncated` identifies candidate-hint truncation
specifically.

Use this endpoint for broad first-pass inspection instead of recursive
client-side tree walking plus many local glue scripts. Use the lower-level tree,
properties, material-binding, and path-translation endpoints for targeted
follow-up after the snapshot.

### Tree

```http
GET /sessions/{session_id}/tree
GET /sessions/{session_id}/tree?prim_path=/World/Asset
```

Returns direct children for lazy hierarchy traversal.

### Properties

```http
GET /sessions/{session_id}/properties?prim_path=/World/Asset/Mesh
```

Returns type, metadata, attributes, relationships, and bounds.

For many prims, use the batch form instead of issuing one request per path:

```http
POST /sessions/{session_id}/properties:batch
content-type: application/json

{
  "prim_paths": [
    "/World/Asset/MeshA",
    "/World/Asset/MeshB"
  ]
}
```

The response contains `results`, a list of the same records returned by the
single-prim properties endpoint.

### Material Binding

```http
GET /sessions/{session_id}/material-binding?prim_path=/World/Asset/Mesh
```

Returns direct/inherited material binding state plus any session material
override.

For many prims, use the batch form:

```http
POST /sessions/{session_id}/material-binding:batch
content-type: application/json

{
  "prim_paths": [
    "/World/Asset/MeshA",
    "/World/Asset/MeshB"
  ]
}
```

The response contains `results`, a list of the same records returned by the
single-prim material-binding endpoint.

### Material Assignments

```http
GET /sessions/{session_id}/authoring/material-assignments
```

Returns the current Workbench material assignment state. These assignments are
stored in source-space paths and include translated inspection-space paths when
the session was optimized.

### Diagnostics

```http
GET /sessions/{session_id}/diagnostics
```

Returns offline diagnostics for unresolved remote dependencies where detected.

## Physics Authoring Operations

Workbench owns the mechanics for inspecting logical components and authored
topology, applying accepted topology and schema plans, and validating authored
physics in a runtime simulation. The calling agent still owns semantic
reasoning, property decisions, and mobility intent.

### Inspect Components

```http
POST /sessions/{session_id}/physics/inspect-components
content-type: application/json

{
  "usd_path": "/absolute/path/to/asset.usdz",
  "root_prim_path": null,
  "path_space": "source"
}
```

The response includes a source digest and logical components. Each component
separates `visual_evidence_paths`, `collider_paths`, `helper_paths`,
`rigid_body_paths`, and `joint_paths`, plus material evidence, bounds, and
topology findings. New workflows must use this endpoint. The
`inspect-mesh-candidates` endpoint remains available only for V1 callers.

### Inspect Topology

```http
POST /sessions/{session_id}/physics/inspect-topology
content-type: application/json

{
  "usd_path": "/absolute/path/to/asset.usdz",
  "root_prim_path": null,
  "path_space": "source"
}
```

The response reports enabled rigid bodies, enabled colliders and their nearest
body owner, joints and body targets, articulation roots, fixed-to-world joints,
and deterministic findings, including redundant fixed joints whose targets
resolve to the same rigid body. Inspection does not choose mobility intent.

### Apply Topology Plan

```http
POST /sessions/{session_id}/physics/apply-topology-plan
content-type: application/json

{
  "schema_version": "content-workflows.physics-topology-plan.v1",
  "input_usd_path": "/absolute/path/to/asset.usda",
  "expected_source_digest": "sha256:...",
  "mobility_intent": "movable",
  "operations": [
    {"op": "remove_rigid_body_api", "prim_path": "/Asset/Body/Inner"},
    {"op": "remove_fixed_joint", "prim_path": "/Asset/Body/RootJoint"},
    {"op": "ensure_rigid_body_api", "prim_path": "/Asset/Body"}
  ],
  "invariants": {
    "enabled_collider_count": 32,
    "reject_articulation_changes": true
  }
}
```

If `output_usd_path` is omitted, Workbench writes `physics/prepared.usda` under
the session workspace. The V1 allowlist is limited to the three operations
shown above. `preserve` forbids removals, fixed-joint removal requires
`movable`, and articulation changes, non-fixed joint removal, collider-count
changes, stale digests, and in-place output are rejected.

`expected_source_digest` is intended for same-run plan/apply flows. It includes
resolved USD layer identity, so regenerate topology plans after relocating an
asset tree to a different workspace or machine.

### Apply Physics Schema

```http
POST /sessions/{session_id}/physics/apply-schema
content-type: application/json

{
  "usd_path": "/absolute/path/to/asset.usdz",
  "predictions_jsonl_path": "/absolute/path/to/physics_predictions.jsonl",
  "decision_patch_path": "/absolute/path/to/physics_decision_patch.json",
  "collision_approximation": "convexHull",
  "output_key": "classification",
  "author_rigid_body": true
}
```

The response reports authored physics scene paths, rigid bodies, collision
schemas, physics materials, and the output USD path. If `output_usd_path` is
omitted, Workbench writes the authored USD under the session workspace. If it is
provided, it must resolve inside the session workspace. V2 decision patches
target inspected collider roles, keep visual/helper paths as evidence, and
author component mass at the declared body root. Set `author_rigid_body=false`
only for an explicitly static mobility intent; Workbench then authors colliders
and materials without adding a dynamic default-prim rigid body.

### Validate Runtime Behavior

```http
POST /sessions/{session_id}/physics/validate-runtime
content-type: application/json

{
  "physics_usd_path": "/absolute/path/to/asset_physics.usda",
  "engine": "ovphysx",
  "duration_s": 1.0,
  "dt": 0.0041666667,
  "sample_fps": 30,
  "drop_height_m": 0.05,
  "acceptance": {
    "detect_initial_pose_discontinuity": true,
    "max_ground_penetration_m": 0.005,
    "require_gravity_response": true,
    "expected_body_count": 1
  }
}
```

The response includes simulation artifacts such as a drop-settle validation
scene, trajectory JSONL, time-sampled USD recording, runtime report, failures,
warnings, and summary metrics. Hard metrics include first-step displacement
versus a ballistic bound, initial-pose discontinuity, transformed final-bounds
ground penetration, gravity response, and loaded body count. If `output_dir` is
omitted, Workbench writes runtime artifacts under the session workspace. If it
is provided, it must resolve inside the session workspace.

## Frame Sequence Render Operations

```http
POST /sessions/{session_id}/render-frames
content-type: application/json

{
  "scene_path": "/absolute/path/to/run/runtime/recording.usda",
  "camera_path": "+x+y+z",
  "width": 512,
  "height": 512,
  "make_mp4": true
}
```

Use this endpoint when any workflow needs an ordered image sequence. If
`scene_path` is omitted, Workbench renders the current session preview scene.
If `scene_path` points at a time-sampled USD recording, Workbench renders that
recording. `camera_path` can be an authored camera path such as
`/Cameras/plus_xplus_yplus_z` or a friendly direction token such as `+x+y+z`.
Physics validation recordings are one producer, but the API is not
physics-specific.
If `output_dir` is omitted, frame artifacts are written under the session
workspace and the response includes download URLs. If it is provided, it must
resolve inside the session workspace.

## Camera And View Commands

Commands use:

```http
POST /sessions/{session_id}/commands
content-type: application/json

{
  "command": "<name>",
  "payload": {}
}
```

Supported navigation and state commands:

- `pick`
- `focus`
- `frame`
- `orbit`
- `pan`
- `dolly`
- `set_camera`
- `select`
- `hide`
- `show`
- `isolate`
- `clear_isolation`
- `material_override`
- `clear_material_override`
- `clear_visual_overrides`
- `reset_view`
- `change_aov`

Examples:

```json
{
  "command": "frame",
  "payload": {
    "prim_path": "/World/Asset",
    "direction": "+x-y+z",
    "margin": 1.2
  }
}
```

```json
{
  "command": "orbit",
  "payload": {
    "yaw_delta_degrees": 20,
    "pitch_delta_degrees": -8
  }
}
```

```json
{
  "command": "isolate",
  "payload": {
    "paths": ["/World/Asset/Subtree"]
  }
}
```

## Camera State

```http
GET /sessions/{session_id}/camera
POST /sessions/{session_id}/camera
```

Camera state fields:

- `target`: world-space look target
- `distance`: orbit distance
- `yaw_degrees`
- `pitch_degrees`
- `focal_length`
- `horizontal_aperture`
- `last_framed_prim_path`

## Rendering

```http
POST /sessions/{session_id}/render
content-type: application/json

{
  "width": 1024,
  "height": 768,
  "use_session_camera": true,
  "direction": null,
  "focus": null,
  "margin": 1.25,
  "hdri_light": 600.0,
  "dome_light": null,
  "distant_light": null,
  "render_quality": "inspection",
  "ovrtx_render_mode": null,
  "ovrtx_num_sensor_updates": null,
  "save_camera_json": true
}
```

For fixed evidence renders, set `use_session_camera` to `false` and provide
`direction` and optionally `focus`.

By default, Workbench uses its studio HDRI rig at intensity `600.0`, with no
plain dome or distant light. Set `hdri_light` to another number to adjust the
studio rig, or to `null` to disable it. `dome_light` and `distant_light` add
plain fill and directional key lights only when explicitly set. If the default
HDRI asset cannot be resolved in a deployment, Workbench logs a warning and
falls back to the older synthetic dome/key rig instead of failing the render.
All source-authored lights are deactivated in the transient viewer layer for
still renders, frame renders, and picks. Only the requested Workbench light rig
contributes to visual inspection; the source USD is never modified.

Render quality presets:

- `interactive`: `rt2`, 16 sensor updates for fast camera iteration.
- `inspection`: `rt2`, 64 sensor updates for normal evidence renders.
- `final`: `rt2`, 256 sensor updates for final verification.

`ovrtx_render_mode` and `ovrtx_num_sensor_updates` are explicit overrides.
Normal agentic asset workflows should leave `ovrtx_render_mode` unset so the
session stays on the RT2 backend for evidence renders, final renders, picks,
and outline renders. Render-mode overrides are intended for targeted
diagnostics. When `save_camera_json` is true, the saved JSON records the
effective camera, render mode, update count, active AOV, and requested lighting
values.

Common directions:

- `+x-y+z`
- `+z`
- `-z`
- `+x`
- `-x`
- `+y`
- `-y`

Response includes:

- `preview_scene_path`
- `image_path`
- `image_url`
- `camera_json_path`
- `camera_json_url`
- `renderer`
- `render_quality`
- effective `ovrtx_render_mode`
- effective `ovrtx_num_sensor_updates`
- `elapsed_seconds`

`image_path` and `camera_json_path` are service-local filesystem paths. Agents
that should not read service-local filesystem paths directly should fetch
`image_url` and `camera_json_url` from the Workbench endpoint:

```http
GET /sessions/{session_id}/renders/{filename}
```

For ordered frame sequences such as turntables, use the frame-render endpoint
instead of issuing repeated single-frame render requests:

```http
POST /sessions/{session_id}/render-frames
content-type: application/json

{
  "width": 512,
  "height": 384,
  "frames": "0:1",
  "directions": [
    "+1.0000x+0.0000y+0.6200z",
    "+0.8660x+0.5000y+0.6200z"
  ],
  "render_quality": "interactive",
  "ovrtx_num_sensor_updates": null,
  "save_camera_json": false
}
```

`frames` follows the OvRTX renderer convention (`"0"`, `"0:11"`, or
`"0,3,6"`). When `directions` is provided, its length must match the selected
frame count. The response contains ordered `frame_urls` and optional
`camera_json_urls`.

## Pixel Picking

```http
POST /sessions/{session_id}/pick
content-type: application/json

{
  "x": 420,
  "y": 240,
  "width": 1024,
  "height": 768,
  "update_selection": true,
  "mode": "replace",
  "ovrtx_num_sensor_updates": 1
}
```

Response includes the picked `prim_paths` and updated `selected_prims`.

## Session Material Override

Material edits are session-layer edits. They do not mutate the source USD scene.

`payload.material` must be a JSON object. Bare string material names are
rejected so agents do not silently get a default gray material.

In optimized sessions, `material_override` accepts either
`"space": "inspection"` or `"space": "source"`. The workbench stores recovered
source-space targets and translates them back to inspection-space bindings for
verification renders.

Use this form to bind an existing material from a USD material library:

```json
{
  "command": "material_override",
  "payload": {
    "prim_path": "/World/Asset/Mesh",
    "space": "inspection",
    "unbind_existing": true,
    "material": {
      "source": "material_library",
      "library_path": "/absolute/path/to/materials_libs_v2.usd",
      "material_path": "/World/Looks/Steel_Painted_Orange",
      "material_name": "Steel Painted Orange"
    }
  }
}
```

If `material_path` is omitted, the workbench derives
`/World/Looks/<safe_material_name>` from `material_name`.

`library_path` is a trusted local file reference. Workbench resolves it on the
service host and sublayers the USD file into the preview stage. When
`CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS` is unset, library paths are limited
to the loaded source scene's directory. Operators can set
`CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS` to a comma-separated list of allowed
host directories when agents need to bind materials from another local library.
`content-workflow-cli` injects this allowlist only for Workbench sidecars
that it starts; reused or remote Workbench endpoints must already include the
material-library directory in their service environment.
Remote deployments should restrict callers or enforce their own
material-library allowlist before forwarding override commands.

Use this form only for quick generated materials:

```json
{
  "command": "material_override",
  "payload": {
    "prim_path": "/World/Asset/Mesh",
    "space": "inspection",
    "unbind_existing": true,
    "material": {
      "display_name": "Debug Gray",
      "diffuse_color": [0.72, 0.72, 0.72],
      "roughness": 0.45,
      "metallic": 0.0
    }
  }
}
```

Clear commands:

```json
{ "command": "clear_material_override", "payload": { "prim_path": "/World/Asset/Mesh" } }
{ "command": "clear_visual_overrides", "payload": {} }
```

## Scene Restore And Export

Workbench owns restore/export from the current session state. Agents should ask
Workbench to project accepted edits back to source space instead of manually
reversing optimizer path maps in prompts.

```http
POST /sessions/{session_id}/scene/restore
content-type: application/json

{
  "output_usd_path": "/absolute/path/to/asset_restored.usda",
  "output_mode": "layer",
  "material_profile": "preview_surface",
  "overwrite": false,
  "include_preview_artifact": true
}
```

In this preview build, material overrides are restored to source-space USD
outputs through the durable material apply path. View-only edits such as
hide/isolate are included in the returned `preview_scene_path` and reported as
warnings until generic edit transactions land.

The response includes:

- `status`
- `source_scene_path`
- `inspection_scene_path`
- `preview_scene_path`
- `output_usd_path`
- `restored_edit_count`
- `unresolved_mappings`
- `warnings`
- `material_apply` when material edits were projected to source space.

## Durable Material Apply

Workbench owns durable material authoring. Agents should preview and verify
material assignments in Workbench, then prefer scene restore above. The
material apply endpoint remains the lower-level compatibility API used by
restore and by existing wrapper code. The source asset is never edited in
place.

When `CONTENT_WORKBENCH_OUTPUT_ROOTS` is set, every caller-selected material
apply or scene restore path must resolve inside one of its comma-separated
directories. `content-workflow-cli` requires the sole configured root to be
the current run directory before exposing Workbench to a child agent.

```http
POST /sessions/{session_id}/authoring/material-assignments:apply
content-type: application/json

{
  "output_usd_path": "/absolute/path/to/asset_materials.usdz",
  "output_mode": "layer",
  "material_profile": "preview_surface",
  "overwrite": false
}
```

`output_usd_path` accepts `.usd`, `.usda`, `.usdc`, or `.usdz`. For `.usdz`
targets, Workbench writes the material-apply result to a temporary USD layer,
packages it into a validated USDZ archive, and replaces the requested output
only after packaging succeeds.

`output_mode` values:

- `layer`: write a material layer that composes over the input USD. This is the
  default and is recommended for agent workflows.
- `composed`: write a composed output stage that sublayers the input USD and
  authors materials/bindings in the output layer.
- `flattened`: write a flattened, self-contained output after material apply.

Durable apply currently requires assignments backed by a local USD material
library. Preview-only generated colors are useful for inspection but are not
valid durable material assignments.

The response includes:

- `status`
- `input_usd_path`
- `output_usd_path`
- `output_mode`
- `material_profile`
- `assignments_path`
- `predictions_path`
- `material_library_path`
- `materials_applied`
- `assignment_stats`
- `applied_assignment_count`
- `skipped_assignment_count`
- `warnings`

`assignments_path` and `predictions_path` point at session-workspace
intermediates used to call the material apply task. They are useful for
inspection during the session, but they are removed when the session closes.
Only the requested `output_usd_path` is durable.

## Screenshot Endpoint

```http
GET /sessions/{session_id}/screenshot?width=1024&height=768
```

Returns a PNG rendered from the current session state.
Render, pick, and screenshot dimensions are capped at 8192 pixels per axis.

## Agent Rules

- Prefer exact USD queries and pixel picking over guessing from hierarchy names.
- Use multiple camera views for material or geometry decisions.
- Use `isolate`, `select`, and `frame` for focused visual evidence.
- Use material-library bindings when a library is provided.
- Apply accepted edits through `/sessions/{session_id}/scene/restore`;
  use `/authoring/material-assignments:apply` only when a material-specific
  workflow needs the lower-level compatibility path. Do not edit source USD
  files directly.
- Treat `preview_scene_path` as the non-destructive composed result for the
  current session state.
- Record evidence render paths and picked prim paths in the calling agent's
  output artifacts.
