# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional tests for material_agent.batch_processor."""

from __future__ import annotations

import errno
import logging
from pathlib import Path
from typing import Any

import pytest

import material_agent.batch_processor as batch_processor
from material_agent.batch_processor import process_usd_batch


async def _never_called(context: dict[str, Any]) -> dict[str, Any]:
    raise AssertionError("workflow_runner should not be called")


@pytest.mark.asyncio
async def test_process_usd_batch_raises_for_missing_directory(tmp_path: Path) -> None:
    sentinel = "api_key=missing-usd-directory-713"
    with pytest.raises(RuntimeError, match="^USD directory not found$") as exc:
        await process_usd_batch(
            tmp_path / sentinel,
            tmp_path / "out",
            workflow_runner=_never_called,
        )
    assert sentinel not in str(exc.value)


@pytest.mark.asyncio
async def test_process_usd_batch_raises_when_no_usd_files_found(tmp_path: Path) -> None:
    sentinel = "api_key=empty-usd-directory-713"
    usd_dir = tmp_path / sentinel
    usd_dir.mkdir()
    (usd_dir / "note.txt").write_text("not a usd")

    async def workflow_runner(context: dict[str, object]) -> dict[str, object]:
        raise AssertionError("workflow_runner should not be called")

    with pytest.raises(RuntimeError, match="^No USD files found in directory$") as exc:
        await process_usd_batch(usd_dir, tmp_path / "out", workflow_runner)
    assert sentinel not in str(exc.value)


@pytest.mark.asyncio
async def test_process_usd_batch_projects_source_inspection_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "never-return-batch-source-inspection-path-713"
    usd_dir = tmp_path / f"user:{sentinel}@assets.example.test"

    def raise_name_too_long(path: Path) -> bool:
        raise OSError(errno.ENAMETOOLONG, "File name too long", str(path))

    monkeypatch.setattr(Path, "exists", raise_name_too_long)

    with pytest.raises(OSError) as exc_info:
        await process_usd_batch(usd_dir, tmp_path / "out", _never_called)

    assert exc_info.value.filename == "<redacted>"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_process_usd_batch_replaces_enumeration_and_output_setup_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_dir = tmp_path / "usd"
    usd_dir.mkdir()
    (usd_dir / "scene.usd").write_text("usd", encoding="utf-8")
    sentinel = "never-return-batch-preloop-error-713"

    def fail_enumeration(path: Path, pattern: str):
        raise OSError(errno.ENAMETOOLONG, sentinel, str(path))

    monkeypatch.setattr(Path, "rglob", fail_enumeration)
    with pytest.raises(
        RuntimeError,
        match="^Unable to inspect USD directory$",
    ) as inspection_error:
        await process_usd_batch(usd_dir, tmp_path / "out", _never_called)
    assert inspection_error.value.__context__ is None
    assert sentinel not in repr(inspection_error.value)

    monkeypatch.undo()
    output_dir = tmp_path / f"api_key={sentinel}"
    created_paths: list[Path] = []

    def fail_output_setup(value: Path, **kwargs: object) -> Path:
        created_paths.append(value)
        raise OSError(errno.ENAMETOOLONG, sentinel, str(value))

    monkeypatch.setattr(
        batch_processor,
        "create_directory_with_safe_diagnostics",
        fail_output_setup,
    )
    with pytest.raises(
        RuntimeError,
        match="^Unable to create batch output directory$",
    ) as output_error:
        await process_usd_batch(usd_dir, output_dir, _never_called)
    assert created_paths == [output_dir]
    assert output_error.value.__context__ is None
    assert sentinel not in repr(output_error.value)


@pytest.mark.asyncio
async def test_process_usd_batch_aggregates_success_error_and_exception(
    tmp_path: Path,
) -> None:
    usd_dir = tmp_path / "usd"
    usd_dir.mkdir()
    for name in ["alpha.usd", "beta.usda", "gamma.usdc"]:
        (usd_dir / name).write_text("usd")

    calls: list[dict[str, object]] = []
    base_context = {"shared": "value"}

    async def workflow_runner(context: dict[str, object]) -> dict[str, object]:
        calls.append(dict(context))
        stem = Path(context["source_override"]).stem
        if stem == "alpha":
            return {"dataset_path": "dataset.jsonl", "num_prims": 4, "num_images": 2}
        if stem == "beta":
            return {"error": "bad input"}
        raise ValueError("boom")

    result = await process_usd_batch(
        usd_dir,
        tmp_path / "batch_out",
        workflow_runner,
        base_context=base_context,
    )

    assert result["num_files_processed"] == 1
    assert result["num_files_failed"] == 2
    assert result["total_files"] == 3
    assert set(result["results"]) == {"alpha", "beta", "gamma"}

    alpha = result["results"]["alpha"]
    assert alpha["status"] == "success"
    assert alpha["dataset_path"] == "dataset.jsonl"
    assert alpha["num_prims"] == 4
    assert alpha["num_images"] == 2

    beta = result["results"]["beta"]
    assert beta["status"] == "failed"
    assert beta["error"] == "USD file processing failed"

    gamma = result["results"]["gamma"]
    assert gamma["status"] == "failed"
    assert gamma["error"] == "USD file processing failed"

    assert base_context == {"shared": "value"}
    assert {Path(call["source_override"]).stem for call in calls} == {
        "alpha",
        "beta",
        "gamma",
    }
    assert {Path(call["output_dir_override"]).name for call in calls} == {
        "alpha",
        "beta",
        "gamma",
    }
    assert all(call["shared"] == "value" for call in calls)


@pytest.mark.asyncio
async def test_process_usd_batch_keeps_raw_io_but_sanitizes_diagnostics(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sentinels = {
        "config": "api_key=batch-config-secret-713",
        "source": "api_key=batch-source-secret-713",
        "output": "api_key=batch-output-secret-713",
        "file_error": "api_key=batch-file-error-secret-713",
        "file_exception": "token=batch-file-exception-secret-713",
        "result_error": "opaque-batch-result-error-secret-713",
        "exception": "opaque-batch-exception-secret-713",
    }
    usd_dir = tmp_path / sentinels["source"]
    usd_dir.mkdir()
    output_dir = tmp_path / sentinels["output"]
    filenames = (
        "ok.usd",
        f"{sentinels['file_error']}.usda",
        f"{sentinels['file_exception']}.usdc",
    )
    for filename in filenames:
        (usd_dir / filename).write_text("usd", encoding="utf-8")

    config_path = tmp_path / sentinels["config"] / "config.yaml"
    calls: list[dict[str, object]] = []
    raw_dataset_path = tmp_path / "api_key=data-plane-output-713" / "dataset.jsonl"

    async def workflow_runner(context: dict[str, object]) -> dict[str, object]:
        calls.append(dict(context))
        filename = Path(context["source_override"]).name
        if filename == "ok.usd":
            return {"dataset_path": str(raw_dataset_path)}
        if sentinels["file_error"] in filename:
            return {"error": sentinels["result_error"]}
        raise RuntimeError(sentinels["exception"])

    with caplog.at_level(logging.INFO):
        result = await process_usd_batch(
            usd_dir,
            output_dir,
            workflow_runner,
            base_context={"config_path": config_path},
        )

    assert result["num_files_processed"] == 1
    assert result["num_files_failed"] == 2
    assert result["output_dir"] == "<redacted>"
    assert result["results"]["ok"]["dataset_path"] == "<redacted>"

    failures = [
        item for item in result["results"].values() if item["status"] == "failed"
    ]
    assert len(failures) == 2
    assert all(item["error"] == "USD file processing failed" for item in failures)

    result_observable = repr(result)
    log_observable = caplog.text
    for key in (
        "config",
        "source",
        "output",
        "file_error",
        "file_exception",
        "result_error",
        "exception",
    ):
        assert sentinels[key] not in result_observable
        assert sentinels[key] not in log_observable
    assert "api_key=data-plane-output-713" not in result_observable

    assert len(calls) == 3
    assert all(call["config_path"] == config_path for call in calls)
    assert {Path(call["source_override"]).name for call in calls} == set(filenames)
    assert all(Path(call["output_dir_override"]).parent == output_dir for call in calls)


@pytest.mark.asyncio
async def test_process_usd_batch_redacted_identifier_cannot_overwrite_safe_stem(
    tmp_path: Path,
) -> None:
    usd_dir = tmp_path / "usd"
    usd_dir.mkdir()
    sensitive_file = usd_dir / "api_key=identifier-secret-713.usd"
    safe_file = usd_dir / "file_1.usda"
    sensitive_file.write_text("usd", encoding="utf-8")
    safe_file.write_text("usd", encoding="utf-8")

    async def workflow_runner(context: dict[str, object]) -> dict[str, object]:
        if Path(context["source_override"]) == sensitive_file:
            return {"error": "expected failure"}
        return {"dataset_path": "dataset.jsonl"}

    result = await process_usd_batch(
        usd_dir,
        tmp_path / "out",
        workflow_runner,
    )

    assert len(result["results"]) == 2
    assert set(result["results"]) == {"file_1", "file_1_2"}
    assert result["num_files_processed"] == 1
    assert result["num_files_failed"] == 1


@pytest.mark.asyncio
async def test_process_usd_batch_raises_when_all_files_fail(tmp_path: Path) -> None:
    usd_dir = tmp_path / "usd"
    usd_dir.mkdir()
    (usd_dir / "broken.usd").write_text("usd")

    async def workflow_runner(context: dict[str, object]) -> dict[str, object]:
        return {"error": "still broken"}

    with pytest.raises(RuntimeError, match="All 1 USD files failed to process"):
        await process_usd_batch(usd_dir, tmp_path / "out", workflow_runner)
