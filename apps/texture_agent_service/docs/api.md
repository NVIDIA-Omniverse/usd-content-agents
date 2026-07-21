# Texture Agent Service API Reference

REST API for AI-driven texture generation on USD materials. Upload a materialized USD file (or reference one by S3 URI), run the texture pipeline end-to-end, stream real-time progress via Server-Sent Events, and download the generated PBR texture set and textured USDZ output.

**Base URL:** `http://localhost:8001`
**Interactive docs:** `GET /docs` (Swagger UI), `GET /redoc`
**OpenAPI:** [`../openapi.yaml`](../openapi.yaml) — snapshot of the live spec produced by FastAPI from the route definitions. When in doubt, the runtime spec at `/openapi.json` is authoritative.

---

## Table of Contents

- [Authentication](#authentication)
- [Root endpoints](#root-endpoints)
- [Pipeline](#pipeline)
- [Sessions](#sessions)
- [Artifacts](#artifacts)
- [Server-Sent Events (SSE)](#server-sent-events-sse)
- [Pipeline steps](#pipeline-steps)
- [Configuration](#configuration)

---

## Authentication

No authentication is required. The service accepts all origins via permissive CORS.

---

## Root endpoints

### `GET /health`

Health check.

**Response** `200`

```json
{
  "status": "healthy",
  "service": "Texture Agent Service",
  "version": "0.5.0",
  "image_gen_backend": "nim",
  "active_backend_key_configured": true,
  "nvidia_api_key_configured": true,
  "max_active_sessions": 4
}
```

### `GET /api`

Returns service info and a map of available endpoints (same catalog as in this document).

---

## Pipeline

### `POST /pipeline/upload-usd`

Upload a USD asset and create a new session without starting a pipeline. Use this when you want to stage a file first, inspect the generated `session_id`, and then call `POST /pipeline` against the same session.

**Request** `multipart/form-data` — supply exactly one of:

| Field | Type | Description |
|-------|------|-------------|
| `usd_file` | file | USD asset (`.usd`, `.usda`, `.usdc`, `.usdz`) uploaded directly. |
| `s3_uri` | string | `s3://bucket/key/path.usd` in a bucket allowed by `TA_S3_ALLOWED_BUCKETS`; the service fetches the file server-side. |

**Response** `201` — `SessionCreated`

```json
{
  "session_id": "11ea5cb5-35aa-491d-9440-dabae87a8f0c",
  "status": "ready",
  "message": "USD uploaded successfully",
  "estimated_duration_minutes": 0
}
```

For S3 uploads, `message` is `USD downloaded from S3 successfully (<size>MB)`.
An empty allowlist or a bucket that is not an exact match returns the same
generic HTTP `403` before any S3 request and does not reveal whether the object
exists. An authorized bucket also returns `403` if S3 access is denied.

### `POST /pipeline`

Create a session and kick off the texture pipeline in one call.

**Request** `multipart/form-data` — supply the USD via one of `usd_file` / `s3_uri` / `session_id`:

| Field | Type | Description |
|-------|------|-------------|
| `usd_file` | file | USD uploaded directly (same rules as `/pipeline/upload-usd`). |
| `reference_image_file` | file | Optional global reference image upload for projection-backend conditioning. Saved into the session and appended to `reference_image_uris`. |
| `s3_uri` | string | `s3://...` reference in a bucket allowed by `TA_S3_ALLOWED_BUCKETS`; the service fetches it. |
| `session_id` | string | Reuse a session previously created via `/pipeline/upload-usd`. |
| `material_textures_json` | string | JSON map `{material_name: {prompt, opacity, detail_policy, per_prim, reference_image_uris, turntable_video_uri, multiview_image_uris}}`. `prompt` is required and must be non-empty, `opacity` is optional and must be between `0.0` and `1.0`, `detail_policy` is optional and must be `default` or `surface_only`, and unknown fields are rejected. Materials not listed get auto-generated prompts via the configured LLM unless `auto_prompt_enabled=false`. |
| `user_prompt` | string | Optional aesthetic direction, e.g. `"weathered mossy patina"`. Used by the LLM auto-prompt step. |
| `auto_prompt_enabled` | boolean | Optional. Defaults to `true` for legacy service behavior. Set `false` to process only materials listed in `material_textures_json`. |
| `texture_backend` | string | Optional texture backend override. Use `service` to route generation through a projection backend instead of the configured default backend. |
| `texture_endpoint` | string | Optional projection backend endpoint. Required when the effective texture backend is `service`. |
| `backend_engine` | string | Optional projection backend engine/model hint. |
| `backend_custom_parameters_json` | string | Optional JSON object passed through to the projection backend, e.g. `{"variant":"success_full_pbr"}` for fake-backend tests. |
| `detail_policy` | string | Optional global texture detail policy. Use `surface_only` for AOI/CAD/PCB assets where traces, vias, labels, seams, holes, components, or other semantic details already exist as geometry. Per-material and per-prim values in `material_textures_json` override this value. |
| `reference_image_uris_json` | string | Optional global JSON list of reference image URIs. Merged with `reference_image_file` and per-material `reference_image_uris`. |
| `turntable_video_uri` | string | Optional global turntable video URI. |
| `multiview_image_uris_json` | string | Optional global JSON list of multi-view image URIs. Merged with per-material `multiview_image_uris`. |
| `seed` | integer | Optional texture backend seed. |
| `strength` | number | Optional edit strength in `[0.0, 1.0]`. |
| `strict_scope` | boolean | Optional selected-scope enforcement flag for projection backend requests. |
| `plan_only` | boolean | Run discovery and persist `texture_plan.json` without invoking prompt, image-generation, application, or render backends. |
| `discovery_mode` | string | Planning scope: `effective_bound` (default), `explicit`, or compatibility-only `all_authored`. |
| `unit_mode` | string | Planning units: `per_material`, `per_group`, or explicitly bounded `per_prim`. |
| `explicit_material_paths_json` | string | JSON list of absolute material prim paths used with explicit discovery. |
| `explicit_prim_paths_json` | string | JSON list of absolute geometry/subset paths used with explicit discovery. |
| `operator_override_cap` | integer | Intentional cap override above the effective default and no greater than 64; recorded in the plan and manifest. |

Plan-only and normal runs use the same planning contract. The generic default
is 32 generation units; UV-aware/service and Step1X plans default to 16.
Requests above that effective default require an explicit override. Plans above
the hard maximum of 64 are rejected before backend work with consolidation and
scope-narrowing guidance. Fetch the validated plan at
`GET /pipeline/{session_id}/plan`; `GET /pipeline/{session_id}/status` embeds a
compact decision, count, limit, and plan-URL summary once planning completes.

**Example (curl)**

```bash
curl -X POST http://localhost:8001/pipeline \
  -F "usd_file=@/path/to/ladder.usd" \
  -F "user_prompt=rusty look" \
  -F "auto_prompt_enabled=false" \
  -F 'material_textures_json={"Steel_Carbon":{"prompt":"heavy patchy rust","opacity":0.85}}'
```

Projection backend runs can be started through the same endpoint by selecting the
`service` texture backend and providing a backend endpoint. The service builds
the Issue #116 texture variation request with a `target`, merged conditioning,
and backend capabilities for each material or per-prim unit:

```bash
curl -X POST http://localhost:8001/pipeline \
  -F "usd_file=@/path/to/ladder.usd" \
  -F "reference_image_file=@/path/to/reference.png" \
  -F "auto_prompt_enabled=false" \
  -F "texture_backend=service" \
  -F "texture_endpoint=http://localhost:8011" \
  -F "backend_engine=fake_projection" \
  -F "reference_image_uris_json=[\"file:///refs/ladder.png\"]" \
  -F "multiview_image_uris_json=[\"file:///refs/view0.png\"]" \
  -F 'backend_custom_parameters_json={"variant":"success_full_pbr"}' \
  -F 'material_textures_json={"Aluminum_Matte":{"prompt":"matte aluminum","reference_image_uris":["file:///refs/aluminum.png"]}}'
```

`per_prim` may override individual prim paths under a material. Each per-prim
entry must include `prompt`, `opacity`, `detail_policy`, or a combination of
those fields, and uses the same opacity bounds and detail-policy values.
Providing any `per_prim` override automatically runs the texture pipeline in
per-prim mode for that request:

```json
{
  "Steel_Carbon": {
    "prompt": "aged steel",
    "opacity": 0.85,
    "detail_policy": "surface_only",
    "per_prim": {
      "/World/Ladder/Rung_01": {"prompt": "fresh scrape marks"},
      "/World/Ladder/Rung_02": {"opacity": 0.65},
      "/World/Ladder/Rung_03": {"detail_policy": "default"}
    }
  }
}
```

`detail_policy=surface_only` rewrites simple image-generation prompts into
plain-material surface prompts and passes policy metadata to projection service
backends.
Use it with explicit prompts, lower opacity/strength, `auto_prompt_enabled=false`,
and `strict_scope=true` when generated material texture should not bake modeled
details such as PCB traces, vias, pads, labels, holes, seams, component outlines,
or markings into the texture maps.

`reference_image_uris`, `turntable_video_uri`, and `multiview_image_uris` may be
provided globally through multipart fields or per material in
`material_textures_json`; per-material lists are appended after global lists.
Local paths and `file://` URIs must resolve inside the service container before
the projection backend is called.

**Response** `202` — `SessionCreated`

```json
{
  "session_id": "11ea5cb5-35aa-491d-9440-dabae87a8f0c",
  "status": "pending",
  "message": "Pipeline queued for execution",
  "estimated_duration_minutes": 10
}
```

### `GET /pipeline/{session_id}/status`

Pipeline state with per-step progress.

**Response** `200` — `PipelineStatus`

```json
{
  "session_id": "11ea5cb5-...",
  "status": "running",
  "current_step": {
    "name": "generate_textures",
    "display_name": "Generating PBR Textures",
    "started_at": "2026-04-21T19:14:23Z",
    "progress": {"current": 3, "total": 8, "percent": 38, "message": "Aluminum_Brushed"},
    "elapsed_seconds": 42
  },
  "completed_steps": [
    {
      "name": "prepare_uvs",
      "display_name": "Preparing UV Coordinates",
      "started_at": "2026-04-21T19:13:00Z",
      "completed_at": "2026-04-21T19:13:01Z",
      "duration_seconds": 1,
      "stats": {}
    },
    {
      "name": "discover_materials",
      "display_name": "Discovering Materials",
      "started_at": "2026-04-21T19:13:01Z",
      "completed_at": "2026-04-21T19:13:01Z",
      "duration_seconds": 0,
      "stats": {}
    },
    {
      "name": "generate_prompts",
      "display_name": "Generating Texture Prompts",
      "started_at": "2026-04-21T19:13:01Z",
      "completed_at": "2026-04-21T19:13:19Z",
      "duration_seconds": 18,
      "stats": {}
    }
  ],
  "overall_progress": {
    "current_step": 4,
    "total_steps": 9,
    "percent": 45,
    "estimated_remaining_seconds": 95
  },
  "preview_images": [],
  "can_cancel": true,
  "elapsed_seconds": 100,
  "created_at": "2026-04-21T19:13:00Z",
  "updated_at": "2026-04-21T19:14:40Z"
}
```

Overall `status` values: `pending | running | completed | failed | cancelled | cancelling`. The `cancelling` state is returned by `POST /pipeline/{session_id}/cancel` and held until the worker reaches the next cancellation checkpoint, after which the status flips to `cancelled`. If a synchronous worker step does not stop within `TA_CANCEL_DRAIN_TIMEOUT_SECONDS`, the session flips to `failed` and a stalled-worker guard blocks deletion until the worker thread finishes.

### `GET /pipeline/{session_id}/results`

Final results and download URLs. Returns `202` while the pipeline is still pending, running, or cancelling; call `/status` first or subscribe to `/events`.

**Response** `200` — `PipelineResults`

```json
{
  "session_id": "11ea5cb5-...",
  "status": "completed",
  "stats": {
    "materials_found": 12,
    "textures_generated": 12,
    "output_usd_count": 1,
    "renders_count": 2
  },
  "download_urls": {
    "materials": "/artifacts/11ea5cb5-.../materials",
    "manifest": "/artifacts/11ea5cb5-.../manifest",
    "textures": "/artifacts/11ea5cb5-.../textures",
    "output": "/artifacts/11ea5cb5-.../output",
    "renders": "/artifacts/11ea5cb5-.../renders"
  },
  "duration_seconds": 142,
  "completed_at": "2026-04-21T19:18:45Z"
}
```

Each artifact is fetched via the matching `/artifacts/{session_id}/{key}` endpoint — see [Artifacts](#artifacts) for the response media types.

For release validation, clients should compare `/results.stats` with the CLI smoke
outputs for the same fixture and material scope. On the ladder beta fixture with
only `Aluminum_Matte` requested and `auto_prompt_enabled=false`, the expected
repeatable smoke shape is four discovered materials, one generated texture set,
one output USD, and zero renders when rendering is disabled. Omitting
`auto_prompt_enabled` keeps legacy auto-prompting enabled and may generate
textures for additional discovered materials. Real backend runs should
additionally record the backend/model, texture size, UV summary, map dimensions,
output package status, and render evidence when rendering is enabled.

Projection backend runs add these status/result stats when available:
`projection_backend_units`, `projection_backend_map_counts`,
`projection_backend_metadata`, `projection_backend_diagnostics`, and
`projection_backend_warnings`. Final results also expose package status, render
availability, and `manifest_url` whenever the artifact manifest has been written.

### `GET /pipeline/{session_id}/events`

Server-Sent Events stream of pipeline progress. See [Server-Sent Events](#server-sent-events-sse).

### `GET /pipeline/{session_id}/event-log`

Full buffered event log (for replay / debugging). Useful on a completed or failed pipeline.

**Response** `200` — list of SSE-shaped events.

### `POST /pipeline/{session_id}/cancel`

Request cancellation. The worker stops at the next cancellation checkpoint, or — if a step is mid-flight — when asyncio cancellation propagates to the next `await` point. Poll `GET /status` to observe the eventual terminal state: normally `cancelled`, or `failed` if the in-flight synchronous worker step exceeds `TA_CANCEL_DRAIN_TIMEOUT_SECONDS`. In the timeout case, DELETE may continue returning `409` until the stalled worker marker clears after the thread exits.

**Response** `200`

```json
{
  "session_id": "11ea5cb5-35aa-491d-9440-dabae87a8f0c",
  "status": "cancelling",
  "message": "Pipeline cancellation requested"
}
```

### `POST /pipeline/{session_id}/regenerate`

Re-run a subset of steps on an existing session — useful when tweaking prompts without re-uploading.

**Request** `application/json`

```json
{
  "steps": ["generate_textures", "blend_textures", "apply_textures"],
  "texture_unit_ids": ["tu_0123456789abcdef0123"],
  "material_textures": {
    "Steel_Carbon": {"prompt": "fresh polished steel", "opacity": 0.75}
  }
}
```

`material_textures` follows the same validated shape as `material_textures_json`
on `POST /pipeline`: material keys must be non-empty, material prompts are
required, opacity is bounded to `0.0` through `1.0`, `per_prim` entries may
override prompt and/or opacity, and unknown fields are rejected. A nested
`per_prim` override promotes the regenerated run to per-prim texture mode;
material-only overrides preserve the session's existing texture mode.

When `generate_textures` is selected, `texture_unit_ids` may name the exact
approved `tu_<20 hex>` units to regenerate from the persisted Texture Plan.
Unknown, duplicate, or non-canonical IDs are rejected. Omit the field to
regenerate every selected unit. Accepted artifacts for all non-requested units
remain unchanged, and a failed targeted retry preserves the previously accepted
artifact while reporting the failed latest attempt.

**Response** `202` — `SessionCreated` (same session_id; new pipeline run).

---

## Sessions

### `GET /sessions`

List known sessions. Includes state, creation time, and basic metadata.

### `GET /sessions/{session_id}`

Session details (status, timestamps, artifact availability).

### `DELETE /sessions/{session_id}`

Remove a session and all associated artifacts from storage.

**Response** `204` — session and stored artifacts removed.

**Error** `404` — JSON response when the session does not exist.

**Error** `409` — JSON response when a live pipeline job is still active or a worker lock shows artifact writes are still in progress. Cancel the pipeline and wait for the worker to stop before deleting the session. Persisted `cancelling` metadata without a live worker lock can still be deleted, which lets restarted services clean up stale session artifacts.

---

## Artifacts

Artifact endpoints are scoped to a `session_id` and only succeed once the corresponding pipeline step has completed.

Unlike the stale pre-0.3.6 contract, these endpoints **return downloadable payloads**, not list-style metadata. Use `GET /pipeline/{session_id}/results` to enumerate available artifact URLs.

Artifact routes intentionally use per-kind media types:

| Endpoint | Success media type | Payload |
|----------|--------------------|---------|
| `GET /artifacts/{session_id}/materials` | `application/json` | Discovered material metadata |
| `GET /artifacts/{session_id}/manifest` | `application/json` | Schema-versioned artifact manifest |
| `GET /artifacts/{session_id}/textures` | `application/zip` | ZIP containing generated textures under `textures/` |
| `GET /artifacts/{session_id}/textures/{filename}` | `image/png` | Single texture image |
| `GET /artifacts/{session_id}/output` | `model/vnd.usdz+zip` | Self-contained textured USDZ |
| `GET /artifacts/{session_id}/renders` | `application/zip` | ZIP containing final rendered images under `renders/` |
| `GET /artifacts/{session_id}/renders/{filename}` | `image/png` | Single render image |
| `GET /artifacts/{session_id}/preview/{filename}` | `image/png` | Single material preview image |

Error responses, including missing sessions or unavailable artifacts, are JSON.
When S3-backed shared session storage is configured with presigning enabled,
single-file artifact routes may return a redirect to a presigned object URL.

### `GET /artifacts/{session_id}/materials`

Discovered-material metadata from the `discover_materials` step.

**Response** `200` — `application/json`, list of `MaterialInfo` records.

### `GET /artifacts/{session_id}/manifest`

Run artifact manifest with schema version `texture-agent-artifacts.v1`. Includes
UV report summary, generated and blended maps, output/package status, render
paths, backend metadata, warnings, errors, and structured package diagnostics
such as `PACKAGE_MISSING_ARTIFACT`.

**Response** `200` — `application/json`.

### `GET /artifacts/{session_id}/textures`

All generated texture files bundled as a ZIP.

**Response** `200` — `application/zip` with a top-level `textures/` folder.

### `GET /artifacts/{session_id}/textures/{filename}`

Single texture image (PNG).

**Response** `200` — `image/png`.

### `GET /artifacts/{session_id}/output`

Textured output asset as a **self-contained USDZ** (USD + embedded textures). Clients should save with a `.usdz` extension regardless of `Content-Disposition`.

**Response** `200` — `model/vnd.usdz+zip`.

### `GET /artifacts/{session_id}/renders`

Rendered preview images (final textured asset) as a ZIP.

**Response** `200` — `application/zip` with a top-level `renders/` folder.

### `GET /artifacts/{session_id}/renders/{filename}`

Single render image.

**Response** `200` — `image/png`.

### `GET /artifacts/{session_id}/preview/{filename}`

Material preview image from the optional `render_previews` step.

**Response** `200` — `image/png`.

---

## Server-Sent Events (SSE)

### `GET /pipeline/{session_id}/events`

Stream real-time events for the pipeline. Standard SSE format:

```
event: step_started
data: {"step": "generate_textures", "started_at": "2026-04-21T19:14:23Z"}

event: step_progress
data: {"step": "generate_textures", "current": 2, "total": 8, "message": "Aluminum_Matte"}

event: step_completed
data: {"step": "apply_textures", "duration_seconds": 1.2, "stats": {"units": 8}}

event: pipeline_completed
data: {"status": "completed", "output_usd_url": "/artifacts/.../output"}
```

Clients should reconnect on disconnect; the `event-log` endpoint lets you replay missed events. In multi-instance deployments, this route returns `503` if the session is running on another instance and the current instance has no local event queue for it.

---

## Pipeline steps

The texture pipeline runs these steps, in order:

1. `prepare_uvs` — Prepare UV coordinates for geometry.
2. `discover_materials` — Discover and catalog materials in the scene.
3. `generate_prompts` — Auto-generate per-material texture prompts via the configured LLM for any material not covered in `material_textures_json`. Falls back to a templated prompt (`"{user_prompt}, applied to {material_name}"`) when the LLM is unavailable.
4. `render_previews` — Render preview images of the current scene (opt-in; disabled by default).
5. `generate_textures` — Generate material textures via the configured texture backend: the default simple image-gen path, or a Texture Variation API service backend such as Step1X. On the simple image-gen path, normal/roughness passes condition on the albedo for coherence except on backends where conditioning is not supported (see note below).
6. `blend_textures` — Composite generated maps at the per-material opacity.
7. `apply_textures` — Attach the textures to the USD materials and write the output USD(Z).
8. `render` — Render the final textured asset (opt-in; disabled by default).

**Image-gen conditioning note.** The cloud `nim` image-gen backend is text-only and drops reference images, so `generate_textures` produces text-conditioned normal/roughness without albedo guidance when that backend is active. For tightly-coupled PBR sets switch `TA_IMAGE_GEN_BACKEND` to `gemini` or `openai`, or run `--profile image-gen` in docker compose to route through the local FLUX.2 NIM sidecar (which does expose `images.edit`). The pipeline logs a one-line warning at the start of `generate_textures` whenever the active backend lacks conditioning support.

### Texture modes

- `per_material` (default) — one texture set per material, shared across every geometry referencing it.
- `per_prim` — clones materials per geometry prim so each mesh gets unique textures.

---

## Configuration

Environment variables read at startup (prefix `TA_`) plus render endpoint
settings used by final render tasks. For Docker Compose, place
Compose-interpolated `TA_*` overrides in the shell or the repo-root `.env`
passed with `--env-file .env`; service-local `apps/texture_agent_service/.env`
is useful for provider keys and other `env_file` values, but explicit Compose
`environment:` entries win. Operator-mounted Step1X-compatible runtime and
OVRTX deployment settings are documented in the service README and Step1X
service README rather than duplicated in this service API table.

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | — | Auth for `nim` image-gen and `nim` chat (build.nvidia.com). |
| `OPENAI_API_KEY` | — | Auth for hosted `openai` image-gen. Not forwarded to local/custom sidecars. |
| `GOOGLE_API_KEY` | — | Auth for `gemini` image-gen. |
| `TA_TEXTURE_BACKEND` | `simple_image_gen` | Texture generation backend. Use `service` for a Texture Variation API backend such as Step1X. In canonical sidecar deployments, requests with `texture_backend=simple_image_gen` route to the simple Texture Variation sidecar when `TA_SIMPLE_TEXTURE_ENDPOINT` is set. |
| `TA_TEXTURE_ENDPOINT` | — | Default Texture Variation API endpoint when `TA_TEXTURE_BACKEND=service`, typically Step1X. |
| `TA_BACKEND_ENGINE` | — | Texture Variation API engine/model hint, for example `step1x`. |
| `TA_SIMPLE_TEXTURE_ENDPOINT` | — | Optional simple Texture Variation API sidecar endpoint. When set, `texture_backend=simple_image_gen` requests use this endpoint and keep stage-level simple defaults. |
| `TA_SIMPLE_BACKEND_ENGINE` | `simple_image_gen` | Engine/model hint used for the simple Texture Variation sidecar. |
| `TA_SIMPLE_TEXTURE_WORKERS` | — | Optional worker count override for simple sidecar requests. |
| `TA_SIMPLE_TEXTURE_JOB_TIMEOUT_SEC` | `3600` | Optional per-material wait timeout for simple sidecar requests. |
| `TA_SIMPLE_UV_SCOPE` | `stage` | UV scope used for simple sidecar requests unless the request overrides `uv_scope`. |
| `TA_SIMPLE_UV_REBAKE_SOURCE_ALBEDO` | `false` | Source albedo rebake default for simple sidecar requests. |
| `TA_SIMPLE_UV_REBAKE_SIZE` | — | Optional source texture rebake resolution override for simple sidecar requests. |
| `TA_IMAGE_GEN_BACKEND` | `nim` | `nim` / `gemini` / `openai`. |
| `TA_IMAGE_GEN_MODEL` | (backend default) | Override the image-gen model. |
| `TA_IMAGE_GEN_BASE_URL` | — | Override image-gen base URL; used by the multi-gpu overlay to route at the local FLUX sidecar. |
| `TA_IMAGE_GEN_API_KEY` | — | Endpoint-specific image-gen key. The multi-gpu overlay sets `not-used` for the local FLUX sidecar and overrides service-local env files. |
| `TA_LLM_BACKEND` | `nim` | LLM backend for auto-prompt generation. |
| `TA_LLM_MODEL` | `qwen/qwen3.5-32b-instruct` | LLM model. |
| `TA_LLM_BASE_URL` | — | Override LLM base URL; set by the overlay when running `--profile llm`. |
| `TA_TEXTURE_SIZE` | `1024` | Output texture resolution. |
| `TA_TEXTURE_WORKERS` | `4` | Parallel texture generation workers. |
| `TA_TEXTURE_JOB_TIMEOUT_SEC` | `3600` | Per-material service backend wait timeout; raise for slow Step1X GPU jobs. |
| `TA_AUTO_PROMPT_MAX_GENERATED_MATERIALS` | `64` | Maximum missing materials auto-prompt may select before requiring explicit scope; `0` disables the guard. |
| `TA_TEXTURE_PLAN_DEFAULT_CAP` | `32` | Generic/simple-image-gen planning default from `texture-agent-plan.v1`. |
| `TA_TEXTURE_PLAN_UV_AWARE_DEFAULT_CAP` | `16` | UV-aware/service and Step1X planning default from `texture-agent-plan.v1`. |
| `TA_TEXTURE_PLAN_HARD_CAP` | `64` | Immutable planning maximum; plans above it require consolidation or narrower scope before backend work. |
| `TA_MAX_TEXTURE_UNITS` | `64` | Compatibility executor guard from issue #463; keep aligned with the planning hard cap. It is not the normal selection default. |
| `TA_BLEND_OPACITY` | `0.85` | Default per-material blend opacity. |
| `TA_UV_POLICY` | `generate_missing` | Default UV policy for service-created pipelines. |
| `TA_UV_SCOPE` | `stage` | Default UV preparation scope; use `target_prims` only with explicit prim targets. |
| `TA_UV_BACKEND` | `python` | Default UV preparation backend. |
| `TA_UV_PROJECTION` | `box` | Projection mode used when UVs are generated. |
| `TA_UV_OVERWRITE_EXISTING` | `false` | Whether generated UVs may overwrite existing UV primvars. |
| `TA_UV_REBAKE_SOURCE_ALBEDO` | `false` | Rebake source albedo when scoped projection UVs are generated. |
| `TA_UV_REBAKE_SIZE` | — | Optional source texture rebake resolution for scoped UV projection. |
| `TA_UV_NORMALIZE_OUT_OF_RANGE` | `false` | Whether out-of-range UVs are normalized during UV prep. |
| `TA_RENDER_ENABLED` | `false` | Enable the final render step for service-created pipelines. |
| `RENDER_ENDPOINT` | — | Remote renderer endpoint used by final render tasks; the Step1X-compatible compose overlay sets `http://ovrtx-rendering-api:8000`. |
| `TA_RENDER_PREVIEWS_ENABLED` | `false` | Enable material preview render step. |
| `TA_RENDER_IMAGE_WIDTH` / `TA_RENDER_IMAGE_HEIGHT` | `1024` / `1024` | Final render dimensions. |
| `TA_RENDER_PREVIEW_IMAGE_WIDTH` / `TA_RENDER_PREVIEW_IMAGE_HEIGHT` | `512` / `512` | Material preview render dimensions. |
| `TA_SESSION_STORAGE_PATH` | `/var/texture-agent/sessions` | Session storage root. |
| `TA_SESSION_TTL_HOURS` | `24` | Session expiry. |
| `TA_S3_ALLOWED_BUCKETS` | empty | Comma- or whitespace-separated exact bucket names allowed for client-supplied `s3_uri` inputs. Empty rejects all such inputs. |
| `TA_STORAGE_KIND` | `local` | Session store backend (`local` or `s3`). |
| `TA_STORAGE_S3_BUCKET` | `WU_S3_BUCKET` | S3 bucket for shared sessions. |
| `TA_STORAGE_S3_PREFIX` | — | Prefix for shared session objects. |
| `TA_STORAGE_S3_REGION` | `WU_S3_REGION` | S3 region. |
| `TA_STORAGE_S3_PROFILE` | `WU_S3_PROFILE` | Optional AWS profile for local/dev runs. |
| `TA_STORAGE_S3_ENDPOINT_URL` | — | Optional S3-compatible endpoint URL. |
| `TA_STORAGE_S3_PRESIGN` | `true` | Return presigned artifact URLs when possible. |
| `TA_STORAGE_S3_MAX_POOL_CONNECTIONS` | `64` | S3 client connection pool size. |
| `TA_MAX_ACTIVE_SESSIONS` | `4` | Max concurrent pipelines, read when the registry starts. Invalid or negative values fall back to `4`; `0` permits no active executions. |
| `TA_CANCEL_DRAIN_TIMEOUT_SECONDS` | `30.0` | Seconds cancellation waits for a synchronous worker thread to stop before marking the session failed with a stalled-worker deletion guard. |
| `TA_MAX_UPLOAD_SIZE_MB` | `500` | Max upload size for `/pipeline/upload-usd`. |

### Client-supplied S3 authorization

`TA_S3_ALLOWED_BUCKETS` controls inbound S3 references and is separate from
`TA_STORAGE_S3_BUCKET`, which configures service-managed session storage. It is
deliberately fail-closed: an unset or empty value, or a bucket that is not an
exact match, is rejected before any S3 operation with HTTP `403` and the
generic detail `S3 URI is not permitted by the service's configured bucket
allowlist`. The same response is used in both cases and does not reveal whether
an object exists.

Upgrade deployments that already accept `s3_uri` by setting this variable
before rollout, or migrate callers to direct uploads. Otherwise all S3 URI
requests begin returning `403`. Restart the service after changing the value.
The Helm chart exposes the same setting as `textureAgent.s3AllowedBuckets`.

Entries are bucket names only (for example, `content-agent-intake`), not
`s3://` URIs, wildcards, or key prefixes. The application policy does not
restrict key prefixes, so use a dedicated intake bucket that contains no
unrelated or sensitive objects. Restrict the service IAM role to only the
required bucket and prefixes as defense in depth; IAM scoping does not replace
request authorization.

For multi-instance deployments, use `TA_STORAGE_KIND=s3` and configure the S3
bucket, prefix, region, and credentials before increasing replicas. Local
storage is single-instance only. The Helm chart exposes the same settings under
`sessionStorage.*`; keep `replicaCount: 1` unless `sessionStorage.kind` is `s3`.
If both explicit S3 credentials and `TA_STORAGE_S3_PROFILE` are configured, the
explicit credentials take precedence; if the named profile is unavailable, the
service falls back to the default boto3 credential chain.

See the [Texture Agent service README](../README.md) for Docker Compose service
startup and the [Step1X service README](../../texture_gen_step1x_service/README.md)
for the operator-mounted runtime contract.
