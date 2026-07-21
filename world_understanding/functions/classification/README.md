# Generic Classification Module

This module provides generic VLM-based object classification that works with any class labels.

## Overview

The classification module was extracted from `material_agent` to enable reuse across different classification tasks. While `material_agent` classifies objects into materials (steel, rubber, plastic), the same core logic can classify:

- **Vehicle types** (sedan, SUV, truck, van)
- **Fabric patterns** (solid, striped, checkered, floral)
- **Defect types** (scratch, dent, corrosion, crack)
- **Animal species** (dog, cat, bird, fish)
- **Any custom classification task**

## Key Functions

### `classify_object()`

Classify a single object using VLM.

```python
from world_understanding.functions.classification import classify_object
from world_understanding.functions.models import create_vlm, create_chat_model

vlm = create_vlm(backend="nim", model="meta/llama-4-maverick-17b")
llm = create_chat_model(service="nim")

result = classify_object(
    vlm=vlm,
    text="This is a vehicle. Types: sedan, SUV, truck",
    images=["vehicle.jpg"],
    llm=llm,
    output_key="vehicle_type",  # Customize output key
    temperature=0.3
)

print(result["vehicle_type"])  # e.g., "sedan"
```

### `batch_classify_objects()`

Batch classification with parallel/sequential processing.

```python
from world_understanding.functions.classification import batch_classify_objects

entries = [
    {"id": "obj_001", "text": "...", "images": ["img1.jpg"]},
    {"id": "obj_002", "text": "...", "images": ["img2.jpg"]},
]

results = batch_classify_objects(
    vlm=vlm,
    entries=entries,
    llm=llm,
    output_key="class",
    max_workers=4  # Parallel processing
)
```

## Parameters

### `output_key`

The `output_key` parameter customizes the output field name:

```python
# Material classification (material_agent)
result = classify_object(..., output_key="material")
# Returns: {"material": "steel", "original_response": "..."}

# Vehicle classification
result = classify_object(..., output_key="vehicle_type")
# Returns: {"vehicle_type": "sedan", "original_response": "..."}

# Generic classification (default)
result = classify_object(..., output_key="class")
# Returns: {"class": "category_a", "original_response": "..."}
```

## Backward Compatibility

The `material_agent` API remains **completely unchanged**:

```python
# material_agent still works identically
from material_agent.functions.inference import assign_material

result = assign_material(vlm, text, images, llm)
# Returns: {"material": "steel", "original_response": "..."}
```

Internally, `assign_material()` is now a thin wrapper:

```python
def assign_material(...):
    return classify_object(..., output_key="material")
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Generic Classification (world_understanding)            │
│   functions/classification/                             │
│     - classify_object(output_key="class")               │
│     - batch_classify_objects(output_key="class")        │
└────────────────────┬────────────────────────────────────┘
                     │ Used by
┌────────────────────┼────────────────────────────────────┐
│ Domain Applications                                      │
│   material_agent: output_key="material"                 │
│   vehicle_agent: output_key="vehicle_type"              │
│   fabric_agent: output_key="pattern"                    │
└─────────────────────────────────────────────────────────┘
```

## Files

- **`inference.py`** - Core VLM classification logic
- **`types.py`** - Dataclasses (ClassificationEntry, ClassificationResult, etc.)
- **`__init__.py`** - Public API exports

## Examples

See `/examples/classification/`:
- `vehicle_classification.py` - Classify vehicle types
- More examples coming soon

## Design Principles

1. **Generic** - Works with any class labels
2. **Stateless** - Pure functions, no side effects
3. **Composable** - Can be used standalone or in pipelines
4. **Backward Compatible** - material_agent API unchanged
5. **Performant** - Supports parallel batch processing

## Testing

```bash
# Test generic classification
pytest tests/functions/classification/

# Test material_agent backward compatibility
pytest apps/material_agent/tests/test_functions_inference.py
```

## Related Documentation

- **material_agent**: `/apps/material_agent/README.md`
- **Project overview**: `../../../README.md`
