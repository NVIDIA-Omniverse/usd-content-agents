# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Domain-neutral recipe registry for bounded harness task execution."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, cast

from pydantic import BaseModel

from world_understanding.agentic.harness.contracts import (
    RecipeContext,
    RecipeSpec,
)


class RecipeRegistry:
    """Mutable registry of deterministic harness recipes.

    Recipes are bounded executors. They validate inputs, run deterministic or
    externally-owned work, write artifacts, and return structured results. They
    do not plan strategy or decide multi-round iteration.
    """

    def __init__(self) -> None:
        self._specs: dict[str, RecipeSpec[Any, Any]] = {}

    def register(self, spec: RecipeSpec[Any, Any]) -> None:
        """Register one recipe spec."""
        if spec.id in self._specs:
            raise ValueError(f"Recipe already registered: {spec.id}")
        self._specs[spec.id] = spec

    def register_many(self, specs: list[RecipeSpec[Any, Any]]) -> None:
        """Register multiple recipe specs."""
        for spec in specs:
            self.register(spec)

    def as_dict(self) -> dict[str, RecipeSpec[Any, Any]]:
        """Return registered recipe specs keyed by id."""
        return dict(self._specs)

    def get(self, recipe_id: str) -> RecipeSpec[Any, Any]:
        """Return one recipe spec or raise a harness-readable error."""
        try:
            return self._specs[recipe_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._specs))
            raise KeyError(
                f"Unknown recipe id '{recipe_id}'. Available recipes: {available}"
            ) from exc

    async def call(
        self,
        recipe_id: str,
        inputs: BaseModel | dict[str, Any],
        context: RecipeContext,
    ) -> BaseModel:
        """Validate inputs and call a registered recipe."""
        return await call_recipe(self.as_dict(), recipe_id, inputs, context)


async def call_recipe(
    registry: dict[str, RecipeSpec[Any, Any]],
    recipe_id: str,
    inputs: BaseModel | dict[str, Any],
    context: RecipeContext,
) -> BaseModel:
    """Validate inputs and call one recipe from a registry dictionary."""
    if recipe_id not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(
            f"Unknown recipe id '{recipe_id}'. Available recipes: {available}"
        )

    spec = registry[recipe_id]
    if isinstance(inputs, spec.input_model):
        validated = inputs
    else:
        validated = spec.input_model.model_validate(inputs)

    context.raise_if_cancelled()
    context.emit(
        "recipe.started",
        {"recipe_id": recipe_id, "run_id": context.run_id},
    )

    try:
        output = spec.recipe(validated, context)
        if inspect.isawaitable(output):
            output = await output
        validated_output = (
            output
            if isinstance(output, spec.output_model)
            else spec.output_model.model_validate(output)
        )
        context.emit(
            "recipe.completed",
            {"recipe_id": recipe_id, "run_id": context.run_id},
        )
        return cast(BaseModel, validated_output)
    except asyncio.CancelledError:
        context.emit_cancelled({"recipe_id": recipe_id})
        raise
    except Exception as exc:
        context.emit(
            "recipe.failed",
            {
                "recipe_id": recipe_id,
                "run_id": context.run_id,
                "error": str(exc),
            },
        )
        raise
