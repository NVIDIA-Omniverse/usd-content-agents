# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from world_understanding.agentic.usd_tasks import validate_usd as vu


class _Listener:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)


def _valid_result(*, issues: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    issues = issues or []
    return {
        "status": "success",
        "summary": {
            "is_valid": not issues,
            "total_issues": len(issues),
            "failures": 0,
            "warnings": len(issues),
            "errors": 0,
        },
        "issues": issues,
        "validation_time": 1.25,
        "categories_checked": ["composition"],
        "fixes": [{"rule": "fixed"}],
    }


def _issue(rule: str = "Rule", at: str = "/World") -> dict[str, str]:
    return {
        "rule": rule,
        "severity": "warning",
        "message": f"{rule} message",
        "at": at,
    }


def test_import_error_helpers_and_fixed_paths(tmp_path: Path) -> None:
    named = ImportError("missing", name="usd_validation_nvidia.core")
    assert vu._is_usd_validator_import_error(named) is True
    assert "usd-validation-nvidia" in vu._validator_import_error_message(named)

    string_match = ImportError("cannot import usd_validation_nvidia")
    assert vu._is_usd_validator_import_error(string_match) is True

    outer = ImportError("outer")
    outer.__cause__ = ImportError("inner", name="usd_validation_nvidia")
    assert vu._is_usd_validator_import_error(outer) is True
    assert (
        vu._is_usd_validator_import_error(ImportError("other", name="other")) is False
    )

    assert (
        vu._fixed_usd_path(Path("asset.usdz"), tmp_path)
        == tmp_path / "fixed_asset.usda"
    )
    assert (
        vu._fixed_usd_path(Path("asset.usd"), tmp_path) == tmp_path / "fixed_asset.usd"
    )


def test_mark_validation_skipped_warn_and_block() -> None:
    listener = _Listener()
    context: dict[str, Any] = {}

    assert vu._mark_validation_skipped(
        context,
        listener,
        "warn",
        ImportError("usd_validation_nvidia missing"),
    )
    assert context["validation_skipped"] is True
    assert context["validation_success"] is False
    assert listener.warnings

    blocked_context: dict[str, Any] = {}
    assert (
        vu._mark_validation_skipped(
            blocked_context,
            listener,
            "block",
            ImportError("usd_validation_nvidia missing"),
        )
        is False
    )
    assert blocked_context["validation_success"] is False
    assert listener.errors


def test_run_validation_success_logs_issue_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.graphics import validate_usd as graphics_validate

    listener = _Listener()

    def fake_validate_usd(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["input_path"] == Path("scene.usd")
        assert kwargs["categories"] == ["composition"]
        assert kwargs["fix"] is True
        assert kwargs["output_path"] == Path("fixed.usda")
        assert kwargs["stage_timeout"] == 9
        return _valid_result(issues=[_issue("RuleA"), _issue("RuleA"), _issue("RuleB")])

    monkeypatch.setattr(graphics_validate, "validate_usd", fake_validate_usd)

    result = asyncio.run(
        vu._run_validation(
            Path("scene.usd"),
            {"categories": ["composition"], "stage_timeout": 9},
            listener,
            "input",
            fix=True,
            output_path=Path("fixed.usda"),
        )
    )

    assert result["summary"]["total_issues"] == 3
    assert any("Top issues by rule" in message for message in listener.infos)


def test_run_validation_raises_on_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from world_understanding.functions.graphics import validate_usd as graphics_validate

    monkeypatch.setattr(
        graphics_validate,
        "validate_usd",
        lambda **kwargs: {"status": "error", "error": "bad stage"},
    )

    with pytest.raises(
        RuntimeError, match="Validation of output USD failed: bad stage"
    ):
        asyncio.run(vu._run_validation(Path("out.usd"), {}, _Listener(), "output"))


def test_compare_and_log_issues() -> None:
    listener = _Listener()

    new_issues = vu._compare_issues(
        baseline_issues=[_issue("RuleA", "/A")],
        current_issues=[
            _issue("RuleA", "/A"),
            _issue("RuleA", "/A"),
            _issue("RuleB", "/B"),
        ],
        listener=listener,
    )
    vu._log_issues(new_issues + [_issue("RuleC", "/C")], listener, limit=1)

    assert [issue["rule"] for issue in new_issues] == ["RuleA", "RuleB"]
    assert any("Baseline comparison" in message for message in listener.infos)
    assert any("and 2 more" in message for message in listener.warnings)


def test_validate_usd_task_requires_inputs_and_valid_modes() -> None:
    task = vu.ValidateUSDTask()

    with pytest.raises(ValueError, match="input_usd_path is required"):
        asyncio.run(task.arun({}))

    with pytest.raises(ValueError, match="Invalid on_failure mode"):
        asyncio.run(task.arun({"input_usd_path": "in.usd", "on_failure": "bad"}))


def test_validate_usd_task_valid_asset_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _Listener()
    monkeypatch.setattr(vu, "get_listener", lambda context: listener)

    async def fake_run_validation(**kwargs: Any) -> dict[str, Any]:
        return _valid_result()

    monkeypatch.setattr(vu, "_run_validation", fake_run_validation)
    context = vu.ValidateUSDTask().run(
        {
            "input_usd_path": "in.usd",
            "validation_config": {"categories": ["composition"]},
            "output_dir": str(tmp_path),
        }
    )

    report = json.loads(
        (tmp_path / "validation_report.json").read_text(encoding="utf-8")
    )
    assert context["validation_success"] is True
    assert context["validation_is_valid"] is True
    assert report["input_usd"] == "in.usd"
    assert report["categories_checked"] == ["composition"]
    assert any("Pre-validation step completed" in message for message in listener.infos)


def test_validate_usd_task_warns_on_invalid_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _Listener()
    monkeypatch.setattr(vu, "get_listener", lambda context: listener)

    async def fake_run_validation(**kwargs: Any) -> dict[str, Any]:
        return _valid_result(issues=[_issue()])

    monkeypatch.setattr(vu, "_run_validation", fake_run_validation)

    context = asyncio.run(
        vu.ValidateUSDTask().arun(
            {
                "input_usd_path": "in.usd",
                "output_dir": str(tmp_path),
                "on_failure": "warn",
            }
        )
    )

    assert context["validation_success"] is True
    assert context["validation_is_valid"] is False
    assert any("Continuing" in message for message in listener.warnings)


def test_validate_usd_task_blocks_on_invalid_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vu, "get_listener", lambda context: _Listener())

    async def fake_run_validation(**kwargs: Any) -> dict[str, Any]:
        return _valid_result(issues=[_issue()])

    monkeypatch.setattr(vu, "_run_validation", fake_run_validation)
    context = {"input_usd_path": "in.usd", "on_failure": "block"}

    with pytest.raises(RuntimeError, match="Pipeline blocked"):
        asyncio.run(vu.ValidateUSDTask().arun(context))

    assert context["validation_success"] is False
    assert "Pipeline blocked" in context["validation_error"]


def test_validate_usd_task_fix_mode_updates_input_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _Listener()
    monkeypatch.setattr(vu, "get_listener", lambda context: listener)
    fixed_path = tmp_path / "fixed_asset.usda"
    fixed_path.write_text("old", encoding="utf-8")
    calls: list[Path] = []

    async def fake_run_validation(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["usd_path"])
        if kwargs.get("fix"):
            kwargs["output_path"].write_text("fixed", encoding="utf-8")
            return _valid_result(issues=[_issue()])
        return _valid_result()

    monkeypatch.setattr(vu, "_run_validation", fake_run_validation)

    context = asyncio.run(
        vu.ValidateUSDTask().arun(
            {
                "input_usd_path": "asset.usdz",
                "output_dir": str(tmp_path),
                "on_failure": "fix",
            }
        )
    )

    assert calls == [Path("asset.usdz"), fixed_path]
    assert context["validation_fixed_usd_path"] == str(fixed_path)
    assert context["input_usd_path"] == str(fixed_path)
    assert json.loads(
        (tmp_path / "validation_report.json").read_text(encoding="utf-8")
    )["fixed_usd_path"] == str(fixed_path)


@pytest.mark.parametrize(
    ("context", "message"),
    [
        (
            {"input_usd_path": "in.usd", "on_failure": "fix"},
            "no output_dir configured",
        ),
        (
            {"input_usd_path": "in.usd", "on_failure": "fix", "output_dir": "."},
            "validator did not write",
        ),
    ],
)
def test_validate_usd_task_fix_mode_errors(
    context: dict[str, Any],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vu, "get_listener", lambda context: _Listener())

    async def fake_run_validation(**kwargs: Any) -> dict[str, Any]:
        return _valid_result(issues=[_issue()])

    monkeypatch.setattr(vu, "_run_validation", fake_run_validation)

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(vu.ValidateUSDTask().arun(context))


def test_validate_usd_task_fix_mode_rejects_still_invalid_fixed_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vu, "get_listener", lambda context: _Listener())

    async def fake_run_validation(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("fix"):
            kwargs["output_path"].write_text("fixed", encoding="utf-8")
        return _valid_result(issues=[_issue()])

    monkeypatch.setattr(vu, "_run_validation", fake_run_validation)

    with pytest.raises(RuntimeError, match="fixed USD still has"):
        asyncio.run(
            vu.ValidateUSDTask().arun(
                {
                    "input_usd_path": "asset.usd",
                    "output_dir": str(tmp_path),
                    "on_failure": "fix",
                }
            )
        )


def test_validate_usd_task_handles_optional_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _Listener()
    monkeypatch.setattr(vu, "get_listener", lambda context: listener)

    async def fake_run_validation(**kwargs: Any) -> dict[str, Any]:
        raise ImportError("missing", name="usd_validation_nvidia")

    monkeypatch.setattr(vu, "_run_validation", fake_run_validation)

    context = asyncio.run(vu.ValidateUSDTask().arun({"input_usd_path": "in.usd"}))
    assert context["validation_skipped"] is True

    blocking_context = {"input_usd_path": "in.usd", "on_failure": "block"}
    with pytest.raises(ImportError):
        asyncio.run(vu.ValidateUSDTask().arun(blocking_context))
    assert blocking_context["validation_success"] is False


def test_validate_usd_task_records_non_optional_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vu, "get_listener", lambda context: _Listener())

    async def fake_run_validation(**kwargs: Any) -> dict[str, Any]:
        raise ImportError("other", name="other")

    monkeypatch.setattr(vu, "_run_validation", fake_run_validation)
    context = {"input_usd_path": "in.usd"}

    with pytest.raises(ImportError):
        asyncio.run(vu.ValidateUSDTask().arun(context))

    assert context["validation_success"] is False
    assert context["validation_error"] == "other"


def test_validate_output_task_requires_inputs_and_modes() -> None:
    task = vu.ValidateOutputUSDTask()

    with pytest.raises(ValueError, match="input_usd_path is required"):
        asyncio.run(task.arun({}))

    with pytest.raises(ValueError, match="Invalid on_failure mode"):
        asyncio.run(task.arun({"input_usd_path": "out.usd", "on_failure": "fix"}))


def test_validate_output_task_run_sync_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    task = vu.ValidateOutputUSDTask()

    async def fake_arun(
        context: dict[str, Any], object_store: Any = None
    ) -> dict[str, Any]:
        context["ran"] = True
        return context

    monkeypatch.setattr(task, "arun", fake_arun)

    assert task.run({"input_usd_path": "out.usd"})["ran"] is True


def test_validate_output_task_uses_cached_baseline_and_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _Listener()
    monkeypatch.setattr(vu, "get_listener", lambda context: listener)

    async def fake_run_validation(**kwargs: Any) -> dict[str, Any]:
        return _valid_result()

    monkeypatch.setattr(vu, "_run_validation", fake_run_validation)
    context = asyncio.run(
        vu.ValidateOutputUSDTask().arun(
            {
                "input_usd_path": "out.usd",
                "original_usd_path": "original.usd",
                "baseline_validation": _valid_result(issues=[_issue("Existing")]),
                "validation_config": {"categories": ["composition"]},
                "output_dir": str(tmp_path),
            }
        )
    )

    report = json.loads(
        (tmp_path / "validation_report.json").read_text(encoding="utf-8")
    )
    assert context["validation_success"] is True
    assert context["validation_regression"] is False
    assert report["regression"] is False
    assert report["input_issues"][0]["rule"] == "Existing"


def test_validate_output_task_warns_on_regression_with_original_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _Listener()
    monkeypatch.setattr(vu, "get_listener", lambda context: listener)
    calls: list[str] = []

    async def fake_run_validation(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["label"])
        if "baseline" in kwargs["label"]:
            return _valid_result(issues=[_issue("Existing")])
        return _valid_result(issues=[_issue("Existing"), _issue("New")])

    monkeypatch.setattr(vu, "_run_validation", fake_run_validation)

    context = asyncio.run(
        vu.ValidateOutputUSDTask().arun(
            {
                "input_usd_path": "out.usd",
                "original_usd_path": "original.usd",
                "on_failure": "warn",
            }
        )
    )

    assert calls == ["input (baseline)", "output"]
    assert context["validation_regression"] is True
    assert [issue["rule"] for issue in context["validation_new_issues"]] == ["New"]
    assert any("REGRESSION" in message for message in listener.warnings)


def test_validate_output_task_without_baseline_only_validates_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _Listener()
    monkeypatch.setattr(vu, "get_listener", lambda context: listener)

    async def fake_run_validation(**kwargs: Any) -> dict[str, Any]:
        return _valid_result(issues=[_issue("Output")])

    monkeypatch.setattr(vu, "_run_validation", fake_run_validation)
    context = asyncio.run(
        vu.ValidateOutputUSDTask().arun({"input_usd_path": "out.usd"})
    )

    assert context["validation_success"] is True
    assert "validation_regression" not in context
    assert any("Cannot compare" in message for message in listener.warnings)


def test_validate_output_task_blocks_on_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vu, "get_listener", lambda context: _Listener())

    async def fake_run_validation(**kwargs: Any) -> dict[str, Any]:
        return _valid_result(issues=[_issue("New")])

    monkeypatch.setattr(vu, "_run_validation", fake_run_validation)
    context = {
        "input_usd_path": "out.usd",
        "baseline_validation": _valid_result(),
        "on_failure": "block",
    }

    with pytest.raises(RuntimeError, match="Output validation regression"):
        asyncio.run(vu.ValidateOutputUSDTask().arun(context))

    assert context["validation_success"] is False
    assert "Output validation regression" in context["validation_error"]


def test_validate_output_task_handles_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vu, "get_listener", lambda context: _Listener())

    async def optional_error(**kwargs: Any) -> dict[str, Any]:
        raise ImportError("missing", name="usd_validation_nvidia")

    monkeypatch.setattr(vu, "_run_validation", optional_error)
    context = asyncio.run(
        vu.ValidateOutputUSDTask().arun({"input_usd_path": "out.usd"})
    )
    assert context["validation_skipped"] is True

    async def other_error(**kwargs: Any) -> dict[str, Any]:
        raise ImportError("other", name="other")

    monkeypatch.setattr(vu, "_run_validation", other_error)
    context = {"input_usd_path": "out.usd"}
    with pytest.raises(ImportError):
        asyncio.run(vu.ValidateOutputUSDTask().arun(context))
    assert context["validation_success"] is False
