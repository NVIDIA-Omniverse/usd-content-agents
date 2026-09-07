# OVRTX Rendering API Reference

USD rendering service using the OVRTX local RTX renderer. Drop-in replacement for the Kit-based rendering API: identical request body and V1 image response format.

**Base URL:** `http://localhost:8001`
**Interactive docs:** `GET /docs` (Swagger UI)
**OpenAPI:** [`../openapi.yaml`](../openapi.yaml)

---

## Table of Contents

- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [`GET /live`](#get-live)
  - [`GET /health`](#get-health)
  - [`POST /render`](#post-render)
  - [`POST /render/upload`](#post-renderupload)
- [Data Models](#data-models)
- [Error Handling](#error-handling)

---

## Authentication

No authentication. The service is intended to run on a trusted internal network (e.g., as a sidecar to a material/physics/texture agent service).

---

## Endpoints

### `GET /live`

Package-owned `usd-cli` protocol identity for the standalone single-GPU
service. A compatible image returns HTTP 200 with exact protocol v3 and no
optional transport features, so clients use the bounded gzip/none multipart
floor:

```json
{
  "status": "alive",
  "protocol_version": 3,
  "engine": "ovrtx",
  "renderer": "ovrtx",
  "max_body_bytes": 73400320,
  "max_scene_bytes": 587202560,
  "features": []
}
```

Standalone and dispatcher modes implement this transport and report the same
package-owned protocol version.

### `GET /health`

Health check including GPU initialization state.

**Response** `200`

```json
{
  "status": "healthy",
  "service": "ovrtx-rendering-api",
  "version": "0.1.0",
  "renderer": "ovrtx",
  "protocol_version": 3,
  "gpu_initialized": true,
  "daemon_pid": 123,
  "daemon_completed_renders": 12,
  "daemon_rss_bytes": 8589934592,
  "daemon_recycle_count": 1,
  "daemon_last_recycle_reason": "completed_render_limit",
  "daemon_pending_recycle_reason": null
}
```

The `gpu_initialized` flag is `false` until the renderer finishes its cold-start
GPU warm-up. In practice this commonly takes around 5 minutes, so readiness
checks should tolerate `false` during that window.

When `OVRTX_GPU_WORKERS` enables the in-container multi-GPU dispatcher, the
same endpoint also reports aggregate capacity and per-worker state:

```json
{
  "status": "healthy",
  "service": "ovrtx-rendering-api",
  "version": "0.1.0",
  "renderer": "ovrtx",
  "protocol_version": 3,
  "gpu_initialized": true,
  "renderer_initialized": true,
  "daemon_running": true,
  "ready_workers": 2,
  "total_workers": 2,
  "workers": [
    {
      "gpu": "0",
      "port": 8100,
      "ready": true,
      "busy": false,
      "in_flight": 0,
      "status": "healthy",
      "renderer_initialized": true,
      "daemon_running": true,
      "restart_count": 0,
      "last_error": null
    }
  ]
}
```

`gpu_initialized=true` means at least one worker is ready. Use
`ready_workers == total_workers` when an orchestrator needs full configured
capacity before sending production traffic.

Dispatcher mode expects a single parent uvicorn process. Running the parent
with uvicorn's `--workers N` makes each parent process try to create private
workers on the same port range and is unsupported.

### `POST /render`

Render a USD file and return base64-encoded images for each (frame, camera, sensor) tuple.

**Request body** -- `application/json` -- [`RenderRequest`](#renderrequest)

```json
{
  "url": "file:///data/scene.usd",
  "force_render": true,
  "render_settings": {
    "camera_paths": ["/Camera"],
    "frame_range": {"start": 0, "end": 0},
    "camera_parameters": {"width": 1024, "height": 1024},
    "sensors": ["rgb"],
    "apply_background_mask": false
  }
}
```

**Response** `200` -- [`RenderResponse`](#renderresponse)

The response structure is `images[frame_number][camera_path][sensor_name] = base64_string`. A successful render always returns `status: "success"`; failures return `status: "exception"` with an `error` message and an empty `images` map.

```json
{
  "status": "success",
  "error": null,
  "images": {
    "0": {
      "/Camera": {
        "rgb": "iVBORw0KGgoAAAANSUhEUg..."
      }
    }
  }
}
```

**Notes**

- The endpoint is a synchronous Python function (not `async def`). A single
  OVRTX worker serializes renders internally; dispatcher mode runs one
  single-flight worker per GPU behind the public endpoint.
- For one standalone OVRTX worker, callers should use
  `max_concurrent_requests: 1` and
  `WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS=1`. Increase concurrency only to the
  number of ready dispatcher workers reported by `/health`.
- The `url` field accepts `file://`, `http://`/`https://`, and `s3://` schemes. S3 URLs require both AWS credentials and an exact bucket match in `OVRTX_S3_ALLOWED_BUCKETS`; empty rejects all S3 intake before AWS access.
- Large `frame_range` requests are run sequentially inside one worker. For
  parallelism on a multi-GPU host, set `OVRTX_GPU_WORKERS` to a worker count
  (`2`) or explicit GPU id list (`0,1`). Leave it unset for legacy
  single-worker behavior.

Client-supplied S3 URLs are fail-closed. Configure
`OVRTX_S3_ALLOWED_BUCKETS` with comma- or whitespace-separated exact bucket
names to opt in; an empty or unset value rejects all S3 intake before the
renderer constructs an AWS session or uses service credentials.

### `POST /render/upload`

The package-owned `usd-cli` remote protocol-v3 path. The multipart request has
one binary USDZ `file` part and one `params` JSON string. Parameters bind one to
eight camera prim paths, bounded image dimensions and frames, `fast` or
`quality` intent, `none` or `gzip` compression, and optional exact camera
definitions. Unknown fields and unsafe or oversized uploads are rejected.
The ASGI receive layer rejects the whole multipart request before form parsing
once it exceeds the advertised `max_body_bytes` plus 1 MiB reserved for bounded
parameters and framing; the endpoint separately enforces `max_body_bytes` on
the file part itself.
Frames are finite USD time codes. Their values, order, and multiplicity are
preserved through the OVRTX backend call and on the corresponding response
items.

`mode` preserves the package-owned wire contract, but the standalone service's
`OVRTX_RENDER_MODE` instance policy is authoritative. The response always
reports the exact OVRTX mode that executed, so a client never has to infer
whether its `fast` or `quality` intent was overridden by deployment policy.

Each response item contains the camera, optional requested frame, PNG bytes as
base64, and the exact OVRTX render mode, sensor-update count, and active AOV
reported by the renderer. Missing image coverage or execution metadata fails
closed. The legacy JSON `/render` contract remains unchanged.

---

## Data Models

### `RenderRequest`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | string | -- | USD asset URL (`file://`, `http://`, `s3://`). |
| `force_render` | bool | `true` | If `true`, re-render even when cached. |
| `render_settings` | [`RenderSettings`](#rendersettings) | -- | Render parameters. |

### `RenderSettings`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `camera_paths` | `list[str]` | `["/Camera"]` | USD prim paths of cameras to render. |
| `frame_range` | [`FrameRange`](#framerange) | `{start: 0, end: 0}` | Inclusive frame range (both ends). |
| `camera_parameters` | [`CameraParameters`](#cameraparameters) | `{width: 1024, height: 1024}` | Per-camera image resolution. |
| `sensors` | `list[str] \| null` | `null` (= `rgb` only) | Sensor outputs, e.g. `["rgb", "depth", "instance_id"]`. |
| `apply_background_mask` | bool | `false` | If `true`, apply dome-light background masking. |
| `material_target` | `"auto" \| "display_color" \| "preview_surface" \| "openpbr_materialx" \| "omnipbr_mdl" \| null` | `null` | Explicit render material target. `auto` preserves authored/native materials; use `preview_surface` only to request render-export PreviewSurface fallbacks. |

### `FrameRange`

| Field | Type | Default |
|-------|------|---------|
| `start` | int | `0` |
| `end` | int | `0` |

### `CameraParameters`

| Field | Type | Default |
|-------|------|---------|
| `width` | int | `1024` |
| `height` | int | `1024` |

### `RenderResponse`

| Field | Type | Description |
|-------|------|-------------|
| `status` | `"success" \| "exception"` | Overall result. |
| `error` | `string \| null` | Error message if `status == "exception"`. |
| `images` | nested map | `images[frame][camera][sensor] = base64 string` (PNG). |
| `error_code` | `string \| null` | Stable error identifier; `incomplete_render_output` for missing coverage. |
| `retryable` | `bool` | Whether the caller may retry with bounded backoff. |
| `requested_output_count` | `int \| null` | Requested camera/frame color-output count. |
| `output_count` | `int \| null` | Non-empty requested color outputs returned. |
| `missing_output_count` | `int \| null` | Requested color outputs omitted. |
| `missing_camera_count` | `int \| null` | Requested cameras affected by omissions. |

### `HealthResponse`

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `healthy`, `initializing`, or `unhealthy`. |
| `service` | string | Service name. |
| `version` | string | Service API version. |
| `renderer` | string | Renderer backend name (`ovrtx`). |
| `protocol_version` | `int` | Package-owned protocol version in standalone and dispatcher modes. |
| `gpu_initialized` | bool | Single-worker readiness, or at least one ready worker in dispatcher mode. |
| `renderer_initialized` | bool | Renderer initialization state. |
| `daemon_running` | bool | OVRTX daemon process state. |
| `daemon_pid` | `int \| null` | Current isolated renderer process ID. |
| `daemon_completed_renders` | `int \| null` | Render commands completed by the current daemon generation. |
| `daemon_rss_bytes` | `int \| null` | Current or last observed daemon resident memory in bytes. |
| `daemon_recycle_count` | `int \| null` | Successful bounded-lifetime recycles since service startup. |
| `daemon_last_recycle_reason` | `string \| null` | Most recent successful recycle reason: `completed_render_limit` or `rss_limit`. |
| `daemon_pending_recycle_reason` | `string \| null` | Guard that will recycle the daemon before the next render. |
| `ready_workers` | `int \| null` | Dispatcher mode only: ready worker count. |
| `total_workers` | `int \| null` | Dispatcher mode only: configured worker count. |
| `workers` | `list[object] \| null` | Dispatcher mode only: per-worker health and queue state. |

---

## Error Handling

Most compatibility errors are returned in the response body with
`status: "exception"` and HTTP 200. Missing camera/frame outputs are different:
the service retries once after single-flight daemon recovery, then returns HTTP
503 with `error_code: "incomplete_render_output"` and `retryable: true` if
coverage is still incomplete. Clients should retry 503 responses with bounded
backoff and must verify that every requested camera/frame has a non-empty color
payload.

Common error cases:

- USD file not found or not a valid USD stage
- Camera path does not exist on the stage
- GPU initialization failure (check `/health` — `gpu_initialized` will be `false`)
- OVRTX daemon crash (container will self-restart; client should retry)

## Daemon lifetime configuration

The native OVRTX/Vulkan process is deliberately persistent to amortize GPU
startup, but it is recycled before the next render after either configured
guard trips. Recycling is single-flight and never kills an active render.

| Variable | Default | Description |
|---|---:|---|
| `OVRTX_DAEMON_MAX_RENDERS` | `64` | Completed commands allowed in one daemon generation; `0` disables. |
| `OVRTX_DAEMON_MAX_RSS_BYTES` | `25769803776` | Resident-byte threshold (24 GiB); `0` disables. RSS inspection is Linux-only, with the count guard providing the portable bound. |

The request after a threshold is recorded incurs one cold start. Raise a guard
only when measured scene requirements justify the additional retained memory;
disable guards only for controlled diagnostics.
