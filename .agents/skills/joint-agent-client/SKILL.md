---
name: joint-agent-client
description: Make requests to the Joint Agent REST API service for VLM-based joint and articulation classification of 3D USD assets. Use when the user wants to call the Joint Agent service API, upload or reference a USD file, run the REST classification pipeline, monitor status, download predictions/report/dataset artifacts, or write a client script.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - joint-agent
  - rest-api
  - client
  - usd
tools:
  - Shell
  - Filesystem
  - Python
  - curl
  - jq
compatibility: Requires a running Joint Agent REST service, optional bearer-token authentication, curl or the bundled Python client, and service-side VLM, rendering, S3, and storage dependencies configured for the requested route.
---

# Joint Agent Client

Use the Joint Agent REST API to classify articulated USD components, generate
`predictions.jsonl`, and download the HTML report and dataset artifacts. Joint
Agent 0.5 is a Research Preview.

## When to Use

- Use when the user asks for the Joint Agent service, API, REST workflow,
  Python client, or curl examples.
- Use when the user wants to upload a USD file, submit an S3 URI, run the full
  joint/articulation classification pipeline, check status, stream events, or
  download artifacts.
- Use `joint-agent-cli` instead when the user wants the local CLI directly.
- Use `joint-agent-validation` after downloading a published USD/USDZ when the
  user asks for Gate 3A or Gate 3B validation.

## Limitations

- Keep credentials out of chat and commits. Use service-side configuration or
  `JOINT_AGENT_TOKEN`; never ask the user to paste secrets.
- This skill calls an already running service. It does not build or start the
  service container.
- Joint Rigger is a Research Preview. Both contract-derived paths author
  accepted joint topology and source-backed limits. For aggregate or multi-root
  contracts, V2 additionally authors exact aggregate rigid-link membership and
  articulation roots; ordinary one-root existing-link contracts retain V1. The
  owned path does not author rigid bodies, masses, colliders, drives, joint
  state, or mimic schemas, and does not prove simulation readiness.
- The 0.5 service generates revolute and prismatic candidates for `owned_core`.
  Empty or all-unready candidate sets complete without a generated package.
- Use `remote` rendering for deployed service runs unless the service owner asks
  for `warp` or `ovrtx`.

## Prerequisites

- Joint Agent service base URL, usually `http://localhost:8000`.
- `curl` and `jq` for shell examples, or Python plus
  `apps/joint_agent_service/client/client.py`.
- Optional bearer token supplied as `Authorization: Bearer TOKEN` or through
  `JOINT_AGENT_TOKEN`.
- A local USD file, an uploaded session, or an S3 URI readable by the service.

## Instructions

1. Confirm the service is reachable with `GET /health`.
2. Choose direct `POST /pipeline`, two-step upload, or S3 URI input.
3. Use exactly one input source: `usd_file`, `session_id`, or `s3_uri`.
4. Pass `render_backend` only when you need to override the service default.
5. To opt into Joint Rigger, pass `apply_joint_rigger=true`. Omit the adapter to
   select built-in `owned_core`; do not request mass or collision authoring.
6. Monitor with `GET /pipeline/{id}/events` for SSE or
   `GET /pipeline/{id}/status` for polling.
7. Download predictions, report, dataset, and any requested Joint Rigger
   artifacts after status is `completed`.

## Python Client

```python
from apps.joint_agent_service.client.client import JointAgentClient

client = JointAgentClient(base_url="http://localhost:8000")

session_id, status = client.run_and_monitor(
    usd_path="/path/to/scene.usdz",
    user_prompt="Focus on identifying articulated robot components",
    render_backend="remote",
    apply_joint_rigger=True,
)

print(session_id, status)
```

Use S3 input mode for large files when the service can download the asset:

```python
session_id, status = client.run_and_monitor(
    s3_uri="s3://your-bucket/path/to/scene.usdz",
    render_backend="remote",
)
```

## curl Workflow

```bash
BASE_URL="http://localhost:8000"

curl -fsS "$BASE_URL/health" | jq .

SESSION=$(curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@scene.usd" \
  -F "render_backend=remote" \
  -F "apply_joint_rigger=true" | jq -r .session_id)

curl -fsS "$BASE_URL/pipeline/$SESSION/status" | jq .
curl -N "$BASE_URL/pipeline/$SESSION/events"

curl -fL -o predictions.jsonl "$BASE_URL/artifacts/$SESSION/predictions"
curl -fL -o report.html "$BASE_URL/artifacts/$SESSION/report"
curl -fL -o dataset.jsonl "$BASE_URL/artifacts/$SESSION/dataset"
curl -fL -OJ \
  "$BASE_URL/artifacts/$SESSION/joint-rigger-output"
```

## Endpoint Reference

| Area | Endpoints |
|---|---|
| Health/API | `GET /health`, `GET /api`, `GET /` |
| Pipeline | `POST /pipeline/upload-usd`, `POST /pipeline`, `GET /pipeline/{id}/status`, `GET /pipeline/{id}/results`, `GET /pipeline/{id}/events`, `POST /pipeline/{id}/cancel?run_id={run_id}`, `POST /pipeline/{id}/regenerate`, `GET /pipeline/{id}/event-log` |
| Artifacts | `GET /artifacts/{id}/predictions`, `/report`, `/dataset`, `/joint-rigger-output`, `/joint-rigger-diagnostics`, `/joint-rigger-validation` |
| Sessions | `GET /sessions`, `GET /sessions/{id}`, `DELETE /sessions/{id}` |

## Key Pipeline Parameters

`POST /pipeline` accepts multipart form data. Exactly one of `usd_file`,
`session_id`, or `s3_uri` must be provided.

| Parameter | Required | Description |
|---|---|---|
| `usd_file` | Conditional | USD input file. |
| `session_id` | Conditional | Existing session from `/pipeline/upload-usd`. |
| `s3_uri` | Conditional | Service-side S3 input. |
| `user_prompt` | No | Custom classification guidance. |
| `render_backend` | No | Backend name forwarded unchanged; validated by the server against the current canonical registry. |
| `apply_joint_rigger` | No | Enables the opt-in Research Preview apply step. |
| `joint_rigger_adapter` | No | Defaults to built-in `owned_core` when the apply step is enabled. |
| `joint_rigger_apply_masses` | No | Must remain `false` for `owned_core`. |
| `joint_rigger_apply_collision` | No | Must remain `false` for `owned_core`. |

Status values are `pending`, `running`, `completed`, `failed`, `cancelled`,
and `cancelling`.

Create and regenerate responses include a 32-character `run_id`. Cancellation
must send that exact generation token; stale tokens fail with HTTP 409 instead
of cancelling a successor run.

## Output Format

Return a concise summary with:

- Service base URL and authentication mode, without printing tokens.
- Session ID, input mode, current status, and progress source.
- Render backend used.
- Artifact URLs or downloaded paths for predictions, report, dataset, and any
  published Joint Rigger output and diagnostics.
- Any blocker such as missing service credentials, upload size, S3 access,
  renderer warm-up, or non-terminal status.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Connection refused | Service is not running or base URL is wrong. | Start the Joint Agent service or correct `BASE_URL`. |
| Unauthorized | Bearer token is missing or wrong. | Set `JOINT_AGENT_TOKEN` locally or pass the correct header. |
| `413` upload response | Input exceeds the service upload limit. | Increase `JA_MAX_UPLOAD_SIZE_MB`, use a smaller file, or submit an S3 URI. |
| Results return before completion | Pipeline is still running. | Poll `/status` until `completed` or stream `/events`. |
| Empty or implausible predictions | The asset may not contain articulated parts or the prompt is too broad. | Add a focused `user_prompt` and review the rendered dataset/report. |
