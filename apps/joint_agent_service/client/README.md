### Joint Agent Service Python Client (CLI + API)

Minimal Python client to start the Joint Agent pipeline and monitor progress via SSE (with polling fallback).

Supports two input modes:
- **File upload** — upload a local USD file over HTTP
- **S3 reference** — pass an S3 URI and the service downloads it server-side (better for large files)

#### Requirements
- Python 3.12+
- requests

Install:
```bash
uv pip install requests   # preferred if uv is available
# or
pip install requests
```

#### CLI Usage
From repo root:
```bash
# Local file upload
python apps/joint_agent_service/client/client.py /path/to/scene.usdz

# S3 URI (service downloads server-side — no local file needed)
python apps/joint_agent_service/client/client.py \
  --s3-uri s3://your-bucket/path/to/asset.usdz
```

Auth (Bearer token):
- Flag: `--token "$YOUR_TOKEN"`
- Or env: `export JOINT_AGENT_TOKEN="$YOUR_TOKEN"`

Examples:
```bash
# Simple — local USD file
python apps/joint_agent_service/client/client.py /path/to/scene.usdz

# S3 URI — large asset, no upload needed
python apps/joint_agent_service/client/client.py \
  --s3-uri s3://your-bucket/path/to/asset.usdz

# With user prompt
python apps/joint_agent_service/client/client.py \
  --prompt "Identify electronic components" \
  /path/to/scene.usdz

# Choose rendering backend (remote, warp, ovrtx, or CPU-only mock)
python apps/joint_agent_service/client/client.py \
  --render-backend remote \
  /path/to/scene.usdz

# Upload USD first, then start pipeline (two-step)
python apps/joint_agent_service/client/client.py \
  --upload-first \
  /path/to/scene.usdz

# Opt in to the built-in topology-only Joint Rigger (owned_core is the default)
python apps/joint_agent_service/client/client.py \
  --apply-joint-rigger \
  /path/to/scene.usdz

# With token and custom base URL
python apps/joint_agent_service/client/client.py \
  --base-url http://localhost:8000 \
  --token "$TOKEN" \
  /path/to/scene.usdz
```

Exit behavior:
- Streams live progress (SSE) and prints updates like: `[build_dataset_usd] running overall=45%`.
- Falls back to status polling if SSE is unavailable.
- Prints artifact URLs on completion.

#### Programmatic Use

```python
from apps.joint_agent_service.client.client import JointAgentClient

client = JointAgentClient(base_url="http://localhost:8000")
session_id, status = client.run_and_monitor(
    usd_path="/path/to/scene.usdz",
    user_prompt="Focus on identifying furniture parts",
    render_backend="warp",  # or "remote", "ovrtx", "mock"
)
print(session_id, status)
```

**S3 URI (large assets):**
```python
from apps.joint_agent_service.client.client import JointAgentClient

client = JointAgentClient(base_url="http://localhost:8000")
session_id, status = client.run_and_monitor(
    s3_uri="s3://your-bucket/path/to/asset.usdz",
    user_prompt="Classify robot components",
)
print(session_id, status)
```

**Two-step (upload/download first, then run):**
```python
client = JointAgentClient(base_url="http://localhost:8000")

# Upload from local file
session_id = client.upload_usd("/path/to/scene.usdz")
# Or download from S3
session_id = client.upload_usd(s3_uri="s3://bucket/path/robot.usdz")

# Start pipeline with existing session
session_id = client.start_pipeline(session_id=session_id)
```

To request the Research Preview Joint Rigger programmatically:

```python
session_id = client.start_pipeline(
    session_id=session_id,
    apply_joint_rigger=True,
)
```

An enabled request defaults to `owned_core`, produces `rigged.usdz`, and does
not require the optional external `usd_joint_rigger` package. The owned adapter
authors topology only; requests that set mass or collision authoring to `true`
are rejected.

The Python client retains the `run_id` returned by create or regenerate and
uses it when `client.cancel(session_id)` is called. Raw HTTP callers must pass
that exact token as the cancellation query parameter; stale tokens return 409.

This omission default is specific to the REST service. Enabled Joint Agent
CLI/YAML configurations must set `steps.apply_joint_rigger.adapter` explicitly.

#### Endpoints

Key endpoints the client uses:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/pipeline` | POST | Start pipeline (USD file, S3 URI, or session_id + optional prompt) |
| `/pipeline/upload-usd` | POST | Upload USD file or provide S3 URI, returns session_id |
| `/pipeline/{session_id}/status` | GET | Poll pipeline status |
| `/pipeline/{session_id}/events` | GET | SSE stream (progress/done/ping) |
| `/pipeline/{session_id}/results` | GET | Final results |
| `/pipeline/{session_id}/cancel?run_id={run_id}` | POST | Cancel one exact run generation |
| `/pipeline/{session_id}/regenerate` | POST | Re-run specific steps |
| `/artifacts/{session_id}/predictions` | GET | Download predictions JSONL |
| `/artifacts/{session_id}/report` | GET | Download HTML report |
| `/artifacts/{session_id}/dataset` | GET | Download dataset JSONL |
| `/artifacts/{session_id}/joint-rigger-output` | GET | Download generated `rigged.usdz` (legacy `rigged.usd` is still readable) |
| `/artifacts/{session_id}/joint-rigger-diagnostics` | GET | Download Joint Rigger diagnostics |
| `/artifacts/{session_id}/joint-rigger-validation` | GET | Download Joint Rigger result/validation artifact |
