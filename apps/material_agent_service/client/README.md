### Material Agent Service Python Client (CLI + API)

Minimal Python client to start the Material Agent pipeline and monitor progress via SSE (with polling fallback).

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
python apps/material_agent_service/client/client.py \
  --base-url http://localhost:8000 \
  --email user@example.com \
  --upload-first \
  --prompt "Metal frames should be aluminum" \
  --generate-ref-prompt "Brushed aluminum frame with matte black plastic steps" \
  --coverage-policy strict \
  --ref /path/to/ref1.png --ref /path/to/ref2.jpg \
  --ref-desc "Top view" --ref-desc "Side detail" \
  /path/to/scene.usd
```

Auth (Bearer token):
- Flag: `--token "$YOUR_TOKEN"`
- Or env: `export MATERIAL_AGENT_TOKEN="$YOUR_TOKEN"`

Examples:
```bash
# Simple
python apps/material_agent_service/client/client.py \
  --email user@example.com \
  /path/to/scene.usd

# With token and prompt
python apps/material_agent_service/client/client.py \
  --base-url http://localhost:8000 \
  --token "$TOKEN" \
  --email user@example.com \
  --prompt "Prefer matte plastics" \
  /path/to/scene.usd

# With custom materials (overrides server defaults)
python apps/material_agent_service/client/client.py \
  --email user@example.com \
  --materials-zip /path/to/custom_materials.zip \
  /path/to/scene.usd

# Generate an AI reference image before running the pipeline
python apps/material_agent_service/client/client.py \
  --email user@example.com \
  --upload-first \
  --generate-ref-prompt "Satin red painted metal body with black rubber wheels" \
  /path/to/scene.usd

# Enable image-based prim clustering for a larger repeated scene
python apps/material_agent_service/client/client.py \
  --email user@example.com \
  --enable-prim-clustering \
  --cluster-min-prims 50 \
  --no-cluster-report \
  /path/to/large_scene.usd

# Run the large-scene workflow with its currently supported coverage policy
python apps/material_agent_service/client/client.py \
  --email user@example.com \
  --large-scene \
  --coverage-policy allow_partial \
  /path/to/large_scene.usd
```

Custom Materials ZIP:
- Use `--materials-zip` to provide a ZIP file with custom materials
- ZIP must contain: `materials.yaml` (service format) + USD library file
- Icons in `thumbs/` are optional (for UI previews only)
- Overrides server default materials for this pipeline run only

Generated materials:
- Use `--enable-material-generation` with at least one reference image or
  `--generate-ref-prompt`.
- The service uses deployment-time `MA_IMAGE_GEN_*` settings for generated
  material textures.

Exit behavior:
- Streams live progress (SSE) and prints updates like: `[render] running overall=87%`.
- Falls back to status polling if SSE is unavailable.
- Prints artifact URLs on completion.
- Single-asset runs default to strict material coverage qualification. Use
  `--coverage-policy allow_partial` only when retaining incomplete artifacts for
  inspection; the returned status includes exact coverage counts, ratios,
  missing prim IDs, warnings, and a machine-readable readiness grade.
- When `--large-scene` is used without `--coverage-policy`, the client derives
  `allow_partial` and prints a notice. Large-scene runs report `not_evaluated`
  until scene-wide prim binding evidence is qualified. Passing explicit
  `--coverage-policy strict` is still rejected by the service so unsupported
  strict qualification cannot silently proceed.

#### Programmatic Use
```python
from apps.material_agent_service.client.client import MaterialAgentClient

client = MaterialAgentClient(base_url="http://localhost:8000", token="YOUR_TOKEN")
session_id, results = client.run_and_monitor(
    usd_path="/path/to/scene.usd",
    reference_images=["/path/to/ref.png"],
    reference_descriptions=["Front view"],
    user_prompt="Use stainless steel for rollers",
    generated_reference_prompt=(
        "Satin red painted metal body with black rubber wheels"
    ),
    camera_views="+x+y+z,-x-y-z",
    upload_first=True,
    materials_zip_path="/path/to/custom_materials.zip",  # Optional
    enable_material_generation=True,
    material_generation_guidance="Prioritize the orange enclosure and dark controls",
    material_generation_texture_size=1024,
    enable_prim_clustering=True,
    cluster_min_prims=50,
    cluster_max_size=25,
    cluster_similarity_threshold_low=0.98,
    cluster_similarity_threshold_medium=0.95,
    cluster_similarity_threshold_high=0.90,
    cluster_report=False,
)
print(session_id, results)
```

`get_results(session_id)` preserves normal HTTP error handling and therefore
raises `requests.HTTPError` when the service returns `202` while a run or its
terminal diagnostics are still finalizing. Use `run_and_monitor(...)`, or poll
`get_status(session_id)` until it is terminal before retrying `get_results`.

See `apps/material_agent_service/examples/regenerate_client_usage.py` for a
focused follow-up example that calls `client.regenerate(...)` and
`client.get_event_log(session_id)` after the initial pipeline run.

Key endpoints the client uses:
- POST `/pipeline` (start)
- POST `/pipeline/upload-usd` (optional pre-upload)
- POST `/pipeline/{session_id}/generate-reference-image` (returns an explicit `reference_id`)
- GET `/assets/{session_id}/generated-ref/{reference_id}` (generated reference image)
- GET `/pipeline/{session_id}/events` (SSE)
- GET `/pipeline/{session_id}/status` (polling)
- GET `/pipeline/{session_id}/results` (final)
- POST `/pipeline/{session_id}/cancel` (cancel)
- POST `/pipeline/{session_id}/regenerate` (re-run steps from cache)
- GET `/pipeline/{session_id}/event-log` (persisted event history)
