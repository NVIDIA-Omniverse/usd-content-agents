# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for release-forward-ported shared pipeline hardening."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

import pytest

import world_understanding.agentic.base_pipeline_executor as executor_module
import world_understanding.agentic.usd_tasks.optimize_usd as optimize_module
import world_understanding.utils.artifacts as artifacts
import world_understanding.utils.credentials as credentials
import world_understanding.utils.result_projection as result_projection


def test_public_pipeline_diagnostic_helpers_project_values() -> None:
    assert executor_module.safe_diagnostic_text("render") == "render"
    assert executor_module.safe_diagnostic_steps(["render", "publish"]) == [
        "render",
        "publish",
    ]


def test_confined_checkpoint_lock_retries_before_acquiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def fake_lock_file(*args: Any, **kwargs: Any) -> Iterator[int]:
        del args, kwargs
        yield 10

    calls: list[str] = []
    sleeps: list[float] = []

    @contextmanager
    def contended_once(descriptor: int) -> Iterator[None]:
        calls.append(f"acquire:{descriptor}")
        if len([call for call in calls if call.startswith("acquire")]) == 1:
            raise BlockingIOError
        try:
            yield
        finally:
            calls.append(f"release:{descriptor}")

    monotonic_values = iter((10.0, 10.05))
    monkeypatch.setattr(executor_module, "open_confined_lock_file", fake_lock_file)
    monkeypatch.setattr(executor_module, "exclusive_descriptor_lock", contended_once)
    monkeypatch.setattr(
        executor_module.time, "monotonic", lambda: next(monotonic_values)
    )
    monkeypatch.setattr(executor_module.time, "sleep", sleeps.append)

    with executor_module._confined_checkpoint_lock(1, "state.lock", timeout=0.1):
        pass

    assert sleeps == [pytest.approx(0.01)]
    # Retry the contended descriptor, then hold and release exactly once. The
    # operating-system lock primitives are covered by the file_locking tests.
    assert calls == ["acquire:10", "acquire:10", "release:10"]


def test_pipeline_cleanup_rejects_shallow_relative_target() -> None:
    with pytest.raises(ValueError, match="Working directory path too shallow: work"):
        executor_module.BasePipelineExecutor()._clean_directories(
            {"working_dir": "work"}
        )


def test_result_projection_preserves_paths_and_rejects_unsupported_inputs() -> None:
    result_path = Path("outputs/result.json")

    assert result_projection.project_result_metadata({"path": result_path}) == {
        "path": result_path
    }
    assert (
        result_projection.retain_safe_result_path("outputs/result.json") == result_path
    )
    assert result_projection.retain_safe_result_path(object()) is None
    assert result_projection.retain_safe_result_text(42) is None


def test_delete_missing_file_in_existing_directory_returns_false(
    tmp_path: Path,
) -> None:
    with artifacts.open_confined_directory(tmp_path) as root_descriptor:
        assert not artifacts.delete_confined_file(
            root_descriptor,
            "missing.bin",
            missing_ok=True,
        )


def test_credential_uri_scanners_ignore_values_without_authority_or_user() -> None:
    assert not credentials._is_url_with_inline_secret("///path")
    assert not credentials._has_userinfo_without_authority(":secret@host")


def test_sync_optimizer_reconstructs_safe_async_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = optimize_module.OptimizeUSDTask()

    async def fail_safely(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        raise optimize_module._SafeOptimizationError("safe optimization failure")

    monkeypatch.setattr(task, "arun", fail_safely)

    with pytest.raises(RuntimeError, match="safe optimization failure") as exc_info:
        task.run({})

    assert exc_info.value.__cause__ is None
