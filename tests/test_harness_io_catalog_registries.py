# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for small harness IO/catalog and model registry helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from world_understanding.agentic.harness.catalog import HarnessCatalog, call_task
from world_understanding.agentic.harness.contracts import TaskSkillSpec
from world_understanding.agentic.harness.io import read_json, write_json


class TaskInput(BaseModel):
    value: int


class TaskOutput(BaseModel):
    doubled: int


async def _double_task(inputs: TaskInput) -> TaskOutput:
    return TaskOutput(doubled=inputs.value * 2)


def _task_spec(task_id: str = "double") -> TaskSkillSpec[TaskInput, TaskOutput]:
    return TaskSkillSpec(
        id=task_id,
        domain="test",
        name="Double",
        description="Double a number",
        when_to_use="tests",
        task=_double_task,
        input_model=TaskInput,
        output_model=TaskOutput,
        tags=["math"],
    )


def test_harness_json_io_round_trip_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.json"
    returned = write_json(path, {"b": 2, "a": object()})
    assert returned == path
    assert path.read_text(encoding="utf-8").startswith('{\n  "a":')
    assert read_json(path)["b"] == 2

    list_path = tmp_path / "list.json"
    list_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected JSON object"):
        read_json(list_path)


def test_harness_catalog_register_get_and_call() -> None:
    catalog = HarnessCatalog()
    spec = _task_spec()
    catalog.register(spec)
    assert catalog.get("double") is spec
    assert catalog.as_dict() == {"double": spec}
    assert asyncio.run(catalog.call("double", {"value": 3})).doubled == 6
    assert (
        asyncio.run(call_task(catalog.as_dict(), "double", TaskInput(value=4))).doubled
        == 8
    )

    with pytest.raises(ValueError, match="already registered"):
        catalog.register(spec)
    with pytest.raises(KeyError, match="Unknown task id 'missing'"):
        catalog.get("missing")
    with pytest.raises(KeyError, match="Available tasks: double"):
        asyncio.run(call_task(catalog.as_dict(), "missing", {"value": 1}))

    other = HarnessCatalog()
    other.register_many([_task_spec("one"), _task_spec("two")])
    assert sorted(other.as_dict()) == ["one", "two"]
