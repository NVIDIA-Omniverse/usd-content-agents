### Physics Agent Service Python Client (CLI + API)

Python client for the Physics Agent REST service. It supports the main
`/pipeline` workflow plus first-class helpers for `/predict`, `/tune`, and the
server-configured `/refine` loop.

Supports:

- **Pipeline** — upload or reference a raw USD, run classify/apply, and produce
  a simulation-ready `output_usd`.
- **Predict-only** — run VLM prediction without applying physics schemas.
- **Tune** — tune authored physics parameters from a physics USD or completed
  pipeline `source_session_id`.
- **Refine** — run the iterative tune-judge-scenario-refine loop from a physics
  USD or completed pipeline `source_session_id`.

#### Requirements

- Python 3.12+
- `requests`

```bash
uv pip install requests
```

#### CLI Usage

The CLI remains focused on the `/pipeline` workflow:

```bash
python apps/physics_agent_service/client/client.py /path/to/scene.usdz

python apps/physics_agent_service/client/client.py \
  --s3-uri s3://your-bucket/path/to/scene.usdz
```

Auth:

- CLI flag: `--token "$YOUR_TOKEN"`
- Env var: `export PHYSICS_AGENT_TOKEN="$YOUR_TOKEN"`

The historical `client_v2.py` entry point is retained as a compatibility shim
that delegates to `client.py`.

#### Programmatic Use

Pipeline:

```python
from apps.physics_agent_service.client.client import PhysicsAgentClient

client = PhysicsAgentClient(base_url="http://localhost:8000")
session_id, status = client.run_and_monitor(
    usd_path="/path/to/scene.usdz",
    user_prompt="Focus on identifying furniture parts",
    render_backend="remote",
)

output_usd = client.download_output_usd(session_id)
```

Prediction-only:

```python
predict_session = client.start_predict(
    session_id=session_id,
    user_prompt="Only return predictions; do not apply physics",
)
predict_status = client.get_predict_status(predict_session)
predict_results = client.get_predict_results(predict_session)
```

Tune a completed pipeline output:

```python
tune_session = client.start_tune(
    source_session_id=session_id,
    scenario_yaml_path="apps/physics_agent/configs/tuning/drop_settle.yaml",
    optimizer="botorch",
    seed=42,
)
tune_results = client.get_tune_results(tune_session)
best_params = client.download_tune_artifact(tune_session, "best_params.json")
```

Refine a completed pipeline output:

```python
refine_session = client.start_refine(
    source_session_id=session_id,
    scenario_yaml_path="apps/physics_agent/configs/tuning/drop_settle.yaml",
    user_prompt="make the object settle on the target surface",
    optimizer="botorch",
    score_threshold=0.9,
    seed=42,
)
refine_results = client.get_refine_results(refine_session)
tuned_usd = client.download_refine_artifact(
    refine_session,
    "final/tuned_physics.usd",
)
```

#### Endpoint Families

| Area | Client helpers |
|------|----------------|
| Pipeline | `start_pipeline`, `run_and_monitor`, `get_status`, `get_results`, `stream_events`, `cancel`, `regenerate` |
| Predict | `start_predict`, `get_predict_status`, `get_predict_results`, `stream_predict_events`, `cancel_predict` |
| Tune | `start_tune`, `get_tune_status`, `get_tune_results`, `stream_tune_events`, `cancel_tune`, `download_tune_artifact` |
| Refine | `start_refine`, `get_refine_status`, `get_refine_results`, `stream_refine_events`, `cancel_refine`, `download_refine_artifact` |
| Pipeline artifacts | `download_predictions`, `download_report`, `download_dataset`, `download_output_usd` |

`get_status`, `get_results`, `stream_events`, and `cancel` default to
`family="pipeline"` for backward compatibility and also accept
`family="predict"`, `family="tune"`, or `family="refine"`.

#### Route Shape

Do not route tune/refine through `/pipeline`. Use `/pipeline` to produce the
physics-authored `output_usd`, then call `/tune` or `/refine` with
`source_session_id`.
