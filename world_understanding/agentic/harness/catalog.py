# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic harness task catalog utilities."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel

from world_understanding.agentic.harness.contracts import TaskSkillSpec


class HarnessCatalog:
    """Mutable registry of harness-visible task specs.

    The catalog is intentionally small. It validates task inputs and dispatches
    a selected callable, but it does not plan, loop, or decide quality.
    """

    def __init__(self) -> None:
        self._specs: dict[str, TaskSkillSpec[Any, Any]] = {}

    def register(self, spec: TaskSkillSpec[Any, Any]) -> None:
        """Register one task spec."""
        if spec.id in self._specs:
            raise ValueError(f"Task spec already registered: {spec.id}")
        self._specs[spec.id] = spec

    def register_many(self, specs: list[TaskSkillSpec[Any, Any]]) -> None:
        """Register multiple task specs."""
        for spec in specs:
            self.register(spec)

    def as_dict(self) -> dict[str, TaskSkillSpec[Any, Any]]:
        """Return registered specs keyed by id."""
        return dict(self._specs)

    def get(self, task_id: str) -> TaskSkillSpec[Any, Any]:
        """Return one spec or raise a harness-readable error."""
        try:
            return self._specs[task_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._specs))
            raise KeyError(
                f"Unknown task id '{task_id}'. Available tasks: {available}"
            ) from exc

    async def call(
        self,
        task_id: str,
        inputs: BaseModel | dict[str, Any],
    ) -> BaseModel:
        """Validate inputs and call a registered task."""
        return await call_task(self.as_dict(), task_id, inputs)


async def call_task(
    catalog: dict[str, TaskSkillSpec[Any, Any]],
    task_id: str,
    inputs: BaseModel | dict[str, Any],
) -> BaseModel:
    """Validate inputs and call one task from a catalog dictionary."""
    if task_id not in catalog:
        available = ", ".join(sorted(catalog))
        raise KeyError(f"Unknown task id '{task_id}'. Available tasks: {available}")

    spec = catalog[task_id]
    if isinstance(inputs, spec.input_model):
        validated = inputs
    else:
        validated = spec.input_model.model_validate(inputs)
    return cast(BaseModel, await spec.task(validated))
