# Material Agent Python API

This module provides programmatic access to Material Agent functionality. All commands available in the CLI are also available as Python functions.

## Quick Start

The API offers **two usage patterns** - choose based on your needs:

### Pattern 1: Convenience Functions (Simplest)

```python
from material_agent.api import benchmark
from pathlib import Path

# Minimal usage - just pass config
result = benchmark(Path("config.yaml"))

# With optional overrides
result = benchmark(Path("config.yaml"), verbose=True, resume=True)

# With dict config
result = benchmark({
    "model": {"service": "openai", "name": "example-vlm-model"},
    "dataset_path": "data.jsonl"
})

if result.success:
    print(f"FCS: {result.metrics.functional_correctness_score}")
```

### Pattern 2: Full Input Classes (Maximum Control)

```python
from material_agent.api import run_benchmark, BenchmarkInput
from pathlib import Path

# Explicit parameter specification
params = BenchmarkInput(
    config=Path("config.yaml"),
    dataset_override=Path("data.jsonl"),
    resume=True,
    verbose=True
)

result = run_benchmark(params)
if result.success:
    print(f"FCS: {result.metrics.functional_correctness_score}")
    print(f"Success rate: {result.metrics.success_rate}%")
else:
    print(f"Error: {result.error}")
```

**When to use each pattern:**
- **Convenience functions**: Quick scripts, notebooks, simple use cases
- **Input classes**: Web services, complex logic, when you need type safety

## ⚠️ Important: Required Config Fields

**API parameters have defaults, but config contents don't!**

While you only need to pass `config` to the API functions, **the config itself has required fields**:

```python
from material_agent.api import predict

# ✓ API parameter has default
result = predict(config)  # Only 1 required param

# ❌ But config MUST contain required fields!
config = {
    "vlm": {                              # REQUIRED
        "backend": "openai", # REQUIRED
        "model": "example-vlm-model"                  # REQUIRED
    },
    "dataset": "data.jsonl"               # REQUIRED
}
```

**What's REQUIRED in configs:**
- `predict`: VLM backend, VLM model, dataset path
- `benchmark`: VLM, LLM, Judge configs, dataset path
- `apply`: Input USD, predictions, output USD, materials library
- `pipeline`: Project name, input/output USD paths, materials, step configs

See the checked-in [`configs/`](../configs/) examples for complete, runnable
configurations covering these required fields.

### Config Builders (Recommended)

Use config builders to avoid missing required fields:

```python
from material_agent.api import build_predict_config, predict

# Builder guides you through required fields
config = build_predict_config(
    vlm_backend="openai",  # Required param
    vlm_model="example-vlm-model",                  # Required param
    dataset_path="data.jsonl",           # Required param
)

result = predict(config)
```

Available builders:
- `build_predict_config()` - For predictions
- `build_benchmark_config()` - For benchmarks
- `build_apply_config()` - For applying materials
- `build_unified_pipeline_config()` - For full unified pipelines
- `build_cluster_prims_config()` - For opt-in image-based prim clustering
- `build_vlm_config()` - For VLM model configs
- `get_required_fields(api_name)` - List required fields

### Enabling Prim Clustering

Prim clustering is an opt-in pipeline optimization for large scenes with many
visually repeated prims. It clusters rendered prim-only images, predicts
materials for cluster representatives, then expands predictions to cluster
members.

```python
from material_agent.api import build_unified_pipeline_config, pipeline

config = build_unified_pipeline_config(
    project_name="large_scene",
    input_usd_path="scene.usd",
    materials_library_path="materials/materials_libs_v2.usd",
    materials_entries=[
        {"name": "Aluminum", "prim_path": "/Materials/Aluminum"},
        {"name": "Rubber", "prim_path": "/Materials/Rubber"},
    ],
    enable_prim_clustering=True,
    cluster_prims_config={
        "min_prims_to_activate": 50,
        "max_cluster_size": 25,
        "embedding_service": "nim",
        "embedding_model": "nvidia/llama-nemotron-embed-vl-1b-v2",
    },
)

result = pipeline(config)
```

Hosted NVIDIA image embeddings require `NVIDIA_API_KEY`. For self-hosted NIM,
use the same `embedding_model="nvidia/llama-nemotron-embed-vl-1b-v2"` and pass
the NIM `base_url`. `max_cluster_size` caps how many prims can inherit one
representative prediction; oversized clusters are split before prediction.

## Configuration Flexibility

All API functions accept configuration in two formats:

### 1. Config File (Path)

Traditional approach using YAML files:

```python
from pathlib import Path

params = BenchmarkInput(
    config=Path("config.yaml"),  # Path to YAML file
    verbose=True
)
```

### 2. In-Memory Config (Dict)

Dynamic approach for programmatic usage:

```python
config_dict = {
    "model": {
        "service": "openai",
        "name": "example-vlm-model",
        "api_key_env": "${OPENAI_API_KEY}"
    },
    "dataset_path": "data.jsonl",
    "output_dir": "output/"
}

params = BenchmarkInput(
    config=config_dict,  # Dictionary with config contents
    verbose=True
)
```

**Benefits of dict configs:**
- Build configs dynamically at runtime
- No need to write temporary YAML files
- Easier testing and experimentation
- Programmatic config generation

**Note:** Relative paths in dict configs are resolved relative to the current working directory, whereas file-based configs resolve paths relative to the config file location.

## Available APIs

All APIs have both convenience functions and full Input classes.

### Benchmark

Evaluate Material Agent performance on a dataset.

**Minimal usage:**
```python
from material_agent.api import benchmark

# Just pass config
result = benchmark(Path("config.yaml"))

# With optional parameters
result = benchmark(Path("config.yaml"), verbose=True, resume=True)
```

**Full control:**
```python
from material_agent.api import run_benchmark, BenchmarkInput

params = BenchmarkInput(
    config=Path("config.yaml"),
    dataset_override=Path("data.jsonl"),
    output_dir_override=Path("output/"),
    resume=False,
    stream_predictions=True,
    verbose=True
)

result = run_benchmark(params)
```

### Predict

Run material predictions without evaluation.

**Minimal:**
```python
from material_agent.api import predict

result = predict(Path("config.yaml"))
```

**Full:**
```python
from material_agent.api import run_predict, PredictInput

params = PredictInput(
    config=Path("config.yaml"),
    resume=False,
    verbose=True
)

result = run_predict(params)
if result.success:
    print(f"Predictions: {result.predictions_path}")
    print(f"Report: {result.report_path}")
```

### Evaluate

Evaluate existing predictions using an LLM judge.

**Minimal:**
```python
from material_agent.api import evaluate

result = evaluate(Path("evaluate.yaml"))
```

**Full:**
```python
from material_agent.api import run_evaluate, EvaluateInput

result = run_evaluate(EvaluateInput(
    config=Path("evaluate.yaml"),
    predictions_override=Path("predictions.jsonl"),
    verbose=True
))
```

### Apply

Apply predicted materials to a USD file.

**Minimal:**
```python
from material_agent.api import apply

result = apply(Path("config.yaml"))
```

**Full:**
```python
from material_agent.api import run_apply, ApplyInput

result = run_apply(ApplyInput(
    config=Path("config.yaml"),
    input_usd_override=Path("input.usd"),
    predictions_override=Path("predictions.jsonl"),
    output_usd_override=Path("output.usd"),
    layer_only=False,
    render_enabled=True,
    verbose=True
))
```

### Pipeline

Execute a multi-step material agent pipeline using the unified pipeline system.

**Note:** This uses the modern unified pipeline system with auto-wired configurations and standardized project structure.

**Minimal:**
```python
from material_agent.api import pipeline

result = pipeline(Path("unified_config.yaml"))
```

**Full:**
```python
from material_agent.api import run_pipeline, PipelineInput

result = run_pipeline(PipelineInput(
    config=Path("unified_config.yaml"),
    skip_steps=["build_dataset_usd"],
    only_steps=[],
    resume=False,
    dry_run=False,
    clean=False,
    verbose=True
))

if result.success:
    print(f"Completed steps: {result.completed_steps}")
```

### Large-Scene Pipeline

Run the scene-level workflow that analyzes a large USD scene, extracts
sub-assets, runs per-asset material pipelines, and composes one materialized
scene output.

The input is one composed USD stage with a valid default root prim
(`defaultPrim`). Large-scene mode is not modeled as a collection of independent
USD files. Service uploads currently accept one USD-family root stage; use USDZ
for self-contained scenes that need dependencies packaged with the upload.
Direct Python calls resolve USD dependencies from the local filesystem normally.

**Minimal:**
```python
from pathlib import Path
from material_agent.api import scene_pipeline

result = scene_pipeline(Path("scene_config.yaml"), max_workers=2)
if result.success:
    print(result.output_usd_path)
```

**Full:**
```python
from pathlib import Path
from material_agent.api import ScenePipelineInput, run_scene_pipeline

params = ScenePipelineInput(
    config=Path("scene_config.yaml"),
    assets=["BuildingA", "/World/Warehouse/Rack_01"],
    max_workers=2,
    predict_max_workers=8,
    skip_existing=True,
    resume=True,
    no_render=False,
)

result = run_scene_pipeline(params)
if result.success:
    print(f"Sub-assets completed: {result.completed_assets}")
    print(f"Payload groups completed: {result.completed_payloads}")
    if result.validation_report:
        print(f"Validation errors: {len(result.validation_report['errors'])}")
else:
    print(f"Scene pipeline failed: {result.error}")
```

The large-scene API also accepts in-memory config dictionaries. Dict configs
resolve relative paths from `config_base_dir` when provided.

```python
from pathlib import Path
from material_agent.api import ScenePipelineInput, run_scene_pipeline

config = {
    "project": {"name": "warehouse"},
    "input": {"usd_path": "warehouse.usda"},
    "materials": {
        "library_path": "materials/materials.usda",
        "entries": [
            {
                "name": "Brushed Steel",
                "description": "Satin silver metal",
                "binding": "/World/Looks/BrushedSteel",
            }
        ],
    },
    "steps": {
        "predict": {
            "vlm": {"backend": "openai", "model": "example-vlm-model"},
            "max_workers": 8,
        }
    },
}

result = run_scene_pipeline(
    ScenePipelineInput(
        config=config,
        config_base_dir=Path("/data/warehouse"),
        max_workers=2,
        no_render=True,
    )
)
```

### Build Dataset - USD

Build dataset from USD files.

**Minimal:**
```python
from material_agent.api import build_dataset_usd, BuildDatasetUsdInput

result = build_dataset_usd(BuildDatasetUsdInput(config=Path("data_prep.yaml")))
```

**Full:**
```python
from material_agent.api import build_dataset_usd, BuildDatasetUsdInput

result = build_dataset_usd(BuildDatasetUsdInput(
    config=Path("data_prep.yaml"),
    source_override=Path("models/"),
    output_dir_override=Path("dataset/"),
    extract_metadata=False,
    verbose=True
))
```

### Build Dataset - PDF VectorStore

Build vector store from PDF documents.

**Minimal:**
```python
from material_agent.api import build_dataset_pdf_vectorstore, BuildDatasetPdfVectorstoreInput

result = build_dataset_pdf_vectorstore(BuildDatasetPdfVectorstoreInput(
    config=Path("pdf_config.yaml")
))
```

### Build Dataset - Prepare Dataset

Prepare dataset with CMF specifications.

**Minimal:**
```python
from material_agent.api import build_dataset_prepare_dataset, BuildDatasetPrepareDatasetInput

result = build_dataset_prepare_dataset(BuildDatasetPrepareDatasetInput(
    config=Path("prepare_config.yaml")
))
```

### Refine

Refine materials with iterative refinement.

**Minimal:**
```python
from material_agent.api import refine

result = refine(Path("iterative_apply.yaml"))
```

**Full:**
```python
from material_agent.api import run_refine, RefineInput

result = run_refine(RefineInput(
    config=Path("iterative_apply.yaml"),
    max_iterations_override=5,
    verbose=True
))

if result.success:
    print(f"Iterations: {result.iteration_count}")
    print(f"Final score: {result.final_judge_score}")
```

### Configure

Create a new pipeline configuration file.

**Minimal:**
```python
from material_agent.api import configure

result = configure(Path("my_pipeline.yaml"))
```

**Full:**
```python
from material_agent.api import run_configure, ConfigureInput

result = run_configure(ConfigureInput(
    output_config_path=Path("my_pipeline.yaml"),
    force=False,
    verbose=True
))
```

## Error Handling

All API functions return result objects with a `success` field:

```python
result = run_benchmark(params)

if result.success:
    # Process successful result
    print(f"Metrics: {result.metrics}")
else:
    # Handle error
    print(f"Error: {result.error}")
```

## Type Safety

All API functions use dataclasses for inputs and outputs, providing:
- IDE autocomplete
- Type checking with mypy
- Validation at runtime

```python
# This will raise a TypeError at runtime
params = BenchmarkInput(
    config="not_a_path",  # Should be Path or dict
)

# This will be caught by mypy during development
params = BenchmarkInput(
    config=Path("config.yaml"),
    invalid_field="value",  # Error: unexpected keyword argument
)
```

## Integration Examples

### Using in a Script

```python
#!/usr/bin/env python3
from pathlib import Path
from material_agent.api import run_pipeline, PipelineInput

def main():
    # Using config file
    params = PipelineInput(
        config=Path("config.yaml"),
        only_steps=["predict", "apply"],
        verbose=True
    )
    
    result = run_pipeline(params)
    
    if result.success:
        print("Pipeline completed successfully!")
        return 0
    else:
        print(f"Pipeline failed: {result.error}")
        return 1

if __name__ == "__main__":
    exit(main())
```

### Dynamic Config Generation

```python
#!/usr/bin/env python3
import os
from material_agent.api import run_benchmark, BenchmarkInput

def run_benchmark_dynamic(model_name: str, dataset_path: str):
    """Run benchmark with dynamically generated config."""
    
    # Build config dictionary dynamically
    config = {
        "model": {
            "service": "openai",
            "name": model_name,
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        "judge": {
            "service": "openai",
            "name": "example-vlm-model",
        },
        "dataset_path": dataset_path,
        "output_dir": f"output/{model_name}",
    }
    
    # Run benchmark with in-memory config
    params = BenchmarkInput(config=config, verbose=True)
    result = run_benchmark(params)
    
    return result

# Run benchmarks for multiple models
for model in ["example-vlm-model", "example-vlm-model-small", "claude-3-5-sonnet"]:
    result = run_benchmark_dynamic(model, "data/benchmark.jsonl")
    print(f"{model}: FCS = {result.metrics.functional_correctness_score}")
```

### Using in a Web Service

```python
from fastapi import FastAPI, HTTPException
from material_agent.api import run_benchmark, BenchmarkInput
from pathlib import Path
from pydantic import BaseModel

app = FastAPI()

class BenchmarkRequest(BaseModel):
    model_name: str
    dataset_path: str
    output_dir: str = "output"

@app.post("/benchmark")
async def benchmark_endpoint(request: BenchmarkRequest):
    # Build config dict from request
    config_dict = {
        "model": {
            "service": "openai",
            "name": request.model_name,
        },
        "dataset_path": request.dataset_path,
        "output_dir": request.output_dir,
    }
    
    params = BenchmarkInput(config=config_dict, verbose=False)
    result = run_benchmark(params)
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    
    return {
        "success": True,
        "metrics": result.metrics.to_dict(),
        "evaluation_path": str(result.evaluation_path)
    }
```

### Using in Tests

```python
import pytest
from pathlib import Path
from material_agent.api import run_benchmark, BenchmarkInput

def test_benchmark_with_file():
    """Test with config file."""
    params = BenchmarkInput(
        config=Path("tests/fixtures/config.yaml"),
        verbose=False
    )
    
    result = run_benchmark(params)
    
    assert result.success
    assert result.metrics.functional_correctness_score > 0

def test_benchmark_with_dict():
    """Test with in-memory config."""
    config = {
        "model": {"service": "openai", "name": "example-vlm-model"},
        "dataset_path": "tests/fixtures/data.jsonl",
        "output_dir": "tests/output",
    }
    
    params = BenchmarkInput(config=config, verbose=False)
    result = run_benchmark(params)
    
    assert result.success
    assert result.metrics is not None
```

## Architecture

The API layer separates concerns:

- **CLI**: User interface, output formatting, interactive prompts
- **API**: Pure business logic, reusable, testable
- **Workflows**: Task orchestration (used by both CLI and API)

This allows the same functionality to be accessed via:
- Command-line interface
- Python API
- Web services
- Testing frameworks
- CI/CD pipelines
