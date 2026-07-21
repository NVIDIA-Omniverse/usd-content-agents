# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NAT runtime loader for direct workflow execution from config files."""

import asyncio
from pathlib import Path
from typing import Any, overload

try:
    from nat.runtime.loader import load_workflow
except ImportError as e:
    raise ImportError(
        "NAT runtime is not installed. Please install it with [nat] extras."
    ) from e


class NATWorkflow:
    """NAT workflow wrapper to avoid reloading config files."""

    def __init__(self, config_path: str | Path):
        """Initialize the workflow.

        Args:
            config_path: Path to the NAT config file (YAML/JSON)
        """
        self.config_path = str(config_path)
        self.workflow: Any | None = None
        self.workflow_manager: Any | None = None

    async def load_workflow(self) -> Any:
        """Load the workflow once for reuse.

        Returns:
            The loaded workflow instance

        Raises:
            Exception: If workflow loading fails
        """
        if self.workflow is None:
            try:
                self.workflow_manager = load_workflow(self.config_path)
                self.workflow = await self.workflow_manager.__aenter__()
            except Exception as e:
                raise Exception(f"Failed to load NAT workflow: {e}") from e
        return self.workflow

    async def close_workflow(self) -> None:
        """Close the workflow and clean up resources."""
        if self.workflow and self.workflow_manager:
            try:
                await self.workflow_manager.__aexit__(None, None, None)
            except (RuntimeError, GeneratorExit):
                # These errors during cleanup can be safely ignored
                pass
            finally:
                self.workflow = None
                self.workflow_manager = None

    async def query(self, question: str) -> str:
        """Query the cached workflow.

        Args:
            question: The question to ask the workflow

        Returns:
            The workflow's response as a string

        Raises:
            Exception: If workflow execution fails
        """
        if self.workflow is None:
            await self.load_workflow()

        if self.workflow is None:
            raise Exception("Failed to load workflow")

        try:
            async with self.workflow.run(question) as runner:
                result = await runner.result(to_type=str)
            return str(result)  # Ensure we return a string
        except Exception as e:
            raise Exception(f"Error querying NAT workflow: {e}") from e

    async def __aenter__(self) -> "NATWorkflow":
        """Context manager entry."""
        await self.load_workflow()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        await self.close_workflow()


@overload  # pragma: no cover - static typing stub.
async def query_workflow(
    config_path: str | Path, question: str
) -> str:  # pragma: no cover - static typing stub.
    """Query with a single question."""
    ...


@overload  # pragma: no cover - static typing stub.
async def query_workflow(
    config_path: str | Path, question: list[str]
) -> list[str]:  # pragma: no cover - static typing stub.
    """Query with multiple questions."""
    ...


async def query_workflow(
    config_path: str | Path, question: str | list[str]
) -> str | list[str]:
    """Query the NAT workflow with one or more questions.

    This is a convenience function that handles workflow lifecycle automatically.
    For better performance with multiple queries, consider using NATWorkflow directly
    with a context manager to reuse the loaded workflow.

    Args:
        config_path: Path to the NAT config file (YAML/JSON)
        question: A single question string or a list of questions

    Returns:
        For single question: The workflow's response as a string
        For multiple questions: A list of responses

    Raises:
        Exception: If workflow loading or execution fails

    Examples:
        # Single question
        response = await query_workflow("config.yaml", "What is AI?")

        # Multiple questions
        responses = await query_workflow("config.yaml", [
            "What is AI?",
            "How does machine learning work?",
            "What are neural networks?"
        ])
    """
    async with NATWorkflow(config_path) as workflow:
        if isinstance(question, str):
            return await workflow.query(question)
        else:
            results = []
            for q in question:
                result = await workflow.query(q)
                results.append(result)
            return results


def validate_nat_config(config_path: str | Path) -> bool:
    """Validate if a NAT config file can be loaded.

    Args:
        config_path: Path to the NAT config file

    Returns:
        True if config is valid, False otherwise
    """
    try:
        # Try to load the workflow without running it
        asyncio.run(_validate_config(str(config_path)))
        return True
    except Exception:
        return False


async def _validate_config(config_path: str) -> None:
    """Internal helper to validate config asynchronously."""
    async with load_workflow(config_path):
        pass  # Just loading is enough to validate
