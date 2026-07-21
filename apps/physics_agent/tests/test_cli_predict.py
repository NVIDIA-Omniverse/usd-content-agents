# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the first-class `physics-agent predict` CLI workflow.

Covers the issue's acceptance criteria:

* `physics-agent predict CONFIG` calls the prediction API directly
  (`run_predict`), NOT through `run --only predict`.
* `physics-agent run CONFIG --only predict` keeps working as a
  compatibility path.
* `PredictOutput` preserves `predictions_path`, `predictions_count`,
  `failed_count`, and `token_stats` (regression test for schema stability).
* The predict path does not import anything under `physics_agent.tuning`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from typer.testing import CliRunner

from physics_agent.api import PredictInput, PredictOutput
from physics_agent.cli import app


def _write_minimal_config(tmp_path: Path) -> Path:
    """Write a minimal unified pipeline config for CLI smoke tests."""
    config = {
        "project": {
            "name": "predict_cli_test",
            "session_id": "predict_cli_test",
            "working_dir": str(tmp_path / "wd"),
        },
        "input": {"usd_path": str(tmp_path / "scene.usda")},
        "steps": {
            "predict": {"enabled": True},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_predict_cli_calls_run_predict_directly(tmp_path, monkeypatch):
    """`physics-agent predict CONFIG` must invoke `run_predict` directly."""
    config_path = _write_minimal_config(tmp_path)

    captured = {"called_with": None, "run_pipeline_called": False}

    def fake_run_predict(params: PredictInput) -> PredictOutput:
        captured["called_with"] = params
        return PredictOutput(
            success=True,
            predictions_path=tmp_path / "predictions.jsonl",
            predictions_count=7,
            failed_count=1,
            token_stats={"prompt_tokens": 100, "completion_tokens": 50},
        )

    def fake_run_pipeline(*args, **kwargs):
        # If the predict CLI ever routes back through run_pipeline, fail loudly.
        captured["run_pipeline_called"] = True
        raise AssertionError(
            "physics-agent predict must NOT route through run_pipeline; "
            "it should call run_predict directly"
        )

    # Patch the API symbols the CLI imports lazily.
    import physics_agent.api as api_module

    monkeypatch.setattr(api_module, "run_predict", fake_run_predict, raising=True)
    monkeypatch.setattr(api_module, "run_pipeline", fake_run_pipeline, raising=True)

    runner = CliRunner()
    result = runner.invoke(app, ["predict", str(config_path)])

    assert result.exit_code == 0, result.output
    assert captured["run_pipeline_called"] is False
    assert captured["called_with"] is not None
    assert isinstance(captured["called_with"], PredictInput)
    assert Path(str(captured["called_with"].config)) == config_path


def test_predict_cli_passes_overrides(tmp_path, monkeypatch):
    """--dataset / --output / --resume / --verbose flow into PredictInput."""
    config_path = _write_minimal_config(tmp_path)
    dataset_path = tmp_path / "external_dataset.jsonl"
    dataset_path.write_text('{"id":"/p","type":"Mesh","images":{}}\n')
    out_dir = tmp_path / "predict_out"

    captured: dict[str, PredictInput] = {}

    def fake_run_predict(params: PredictInput) -> PredictOutput:
        captured["params"] = params
        return PredictOutput(success=True, predictions_count=0)

    import physics_agent.api as api_module

    monkeypatch.setattr(api_module, "run_predict", fake_run_predict, raising=True)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "predict",
            str(config_path),
            "--dataset",
            str(dataset_path),
            "--output",
            str(out_dir),
            "--resume",
        ],
    )
    assert result.exit_code == 0, result.output

    p = captured["params"]
    assert p.dataset_override == dataset_path
    assert p.output_dir_override == out_dir
    assert p.resume is True


def test_predict_cli_propagates_failure(tmp_path, monkeypatch):
    """Predict failure exits non-zero with the error surfaced."""
    config_path = _write_minimal_config(tmp_path)

    def fake_run_predict(_params: PredictInput) -> PredictOutput:
        return PredictOutput(success=False, error="vlm endpoint refused")

    import physics_agent.api as api_module

    monkeypatch.setattr(api_module, "run_predict", fake_run_predict, raising=True)

    runner = CliRunner()
    result = runner.invoke(app, ["predict", str(config_path)])
    assert result.exit_code != 0


def test_predict_cli_missing_config_exits(tmp_path):
    """Missing config path fails fast with exit code 1, no API call."""
    runner = CliRunner()
    result = runner.invoke(app, ["predict", str(tmp_path / "does_not_exist.yaml")])
    assert result.exit_code == 1


def test_run_only_predict_compat_path_still_works(tmp_path, monkeypatch):
    """`physics-agent run CONFIG --only predict` must keep working.

    Routes through run_pipeline (the unified pipeline executor) — that's the
    legacy compatibility path. We confirm the CLI calls run_pipeline with
    only_steps=['predict'].
    """
    monkeypatch.setenv("NVIDIA_API_KEY", "test-provider-key")
    config_path = _write_minimal_config(tmp_path)

    captured: dict[str, object] = {}

    class _FakePipelineOutput:
        success = True
        error = None
        step_results = {"predict": {"predictions_count": 3}}
        completed_steps = ["predict"]

    def fake_run_pipeline(params):
        captured["params"] = params
        return _FakePipelineOutput()

    import physics_agent.api as api_module

    monkeypatch.setattr(api_module, "run_pipeline", fake_run_pipeline, raising=True)

    runner = CliRunner()
    result = runner.invoke(app, ["run", str(config_path), "--only", "predict"])
    assert result.exit_code == 0, result.output

    params = captured["params"]
    assert params.only_steps == ["predict"]


def test_predict_output_field_stability():
    """Regression: PredictOutput must keep its sticky public fields with
    stable defaults and types.

    Old callers and the REST layer rely on these being present, named
    exactly this, and defaulting exactly this. Adding new optional fields
    is fine; renaming, removing, or changing defaults of these is a
    breaking change the issue explicitly forbids.
    """
    # Field presence + values
    out = PredictOutput(
        success=True,
        predictions_path=Path("/tmp/predictions.jsonl"),
        predictions_count=42,
        failed_count=3,
        token_stats={"prompt_tokens": 1234, "completion_tokens": 567},
    )
    for field_name in (
        "success",
        "error",
        "predictions_path",
        "predictions_count",
        "failed_count",
        "token_stats",
    ):
        assert hasattr(out, field_name), f"PredictOutput missing field: {field_name}"

    assert out.predictions_count == 42
    assert out.failed_count == 3
    assert out.token_stats["prompt_tokens"] == 1234
    assert out.predictions_path == Path("/tmp/predictions.jsonl")

    # Default-value lock — these must not drift, otherwise old callers that
    # construct PredictOutput(success=False) silently change behaviour.
    default_failure = PredictOutput(success=False)
    assert default_failure.success is False
    assert default_failure.error is None
    assert default_failure.predictions_path is None
    assert default_failure.predictions_count == 0
    assert default_failure.failed_count == 0
    assert default_failure.token_stats == {}


_FORBIDDEN_IMPORT_PATTERNS = (
    # Match `import X`, `from X import ...`, and `import X as ...` for the
    # banned modules. Comments and free-form strings (e.g. docstrings that
    # explain *why* the dependency is banned) are intentionally tolerated.
    "import physics_agent.tuning",
    "from physics_agent.tuning",
    "import botorch",
    "from botorch",
    "import ovphysx",
    "from ovphysx",
)


def _assert_no_forbidden_imports(source: str, label: str) -> None:
    for line in source.splitlines():
        # Strip comments to avoid flagging "# do not import physics_agent.tuning"
        code = line.split("#", 1)[0].strip()
        for pattern in _FORBIDDEN_IMPORT_PATTERNS:
            assert pattern not in code, (
                f"{label} must not depend on {pattern!r} — found in: {line!r}"
            )


def test_predict_path_does_not_import_tuning():
    """The predict path must not pull in physics_agent.tuning, BoTorch, or OvPhysX.

    Per the issue: "Do not make prediction depend on BoTorch or OvPhysX."

    We import the module fresh (so sys.modules registers anything it
    transitively pulls in) AND we statically scan the file for forbidden
    import statements. The static check is what fails the build for new
    accidental imports; the runtime import just confirms the module loads
    without error.
    """
    import importlib

    sys.modules.pop("physics_agent.api.predict", None)
    importlib.import_module("physics_agent.api.predict")

    import physics_agent.api.predict as predict_module

    src = Path(predict_module.__file__).read_text()
    _assert_no_forbidden_imports(src, "physics_agent.api.predict")


def test_unified_config_unwraps_to_same_predict_paths(tmp_path):
    """Direct predict and `run --only predict` must hit the same paths.

    Both entry points should resolve `dataset` and `output_dir` to the same
    locations under `{working_dir}/`. This locks the artifact-path parity
    that issue #42 requires (predictions.jsonl + report.html land in the
    same place for both CLIs, so /artifacts and pipeline UIs see them).
    """
    working_dir = tmp_path / "wd"
    config = {
        "project": {
            "name": "parity_test",
            "session_id": "parity_test",
            "working_dir": str(working_dir),
        },
        "input": {"usd_path": str(tmp_path / "scene.usda")},
        "steps": {
            "predict": {
                "enabled": True,
                "vlm": {"backend": "nim", "model": "test/model"},
            },
        },
    }
    config_path = tmp_path / "unified.yaml"
    config_path.write_text(yaml.safe_dump(config))

    # Drive PredictConfigTask directly so we can compare the resolved
    # dataset_path / output_dir to what UnifiedPipelineConfigTask wires up
    # for the predict step. We need a real dataset.jsonl on disk so the
    # task doesn't bail with "Dataset file not found".
    dataset_dir = working_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "dataset.jsonl").write_text(
        '{"id":"/p","type":"Mesh","images":{}}\n'
    )

    from physics_agent.tasks.config_predict import PredictConfigTask

    task = PredictConfigTask()
    ctx = {"config_path": str(config_path)}
    out = task.run(ctx)

    # Direct predict path must auto-wire to {working_dir}/predictions/
    # (matches ProjectPathResolver.get_predictions_dir() used by --only predict).
    expected_dataset = (working_dir / "dataset" / "dataset.jsonl").resolve()
    expected_output_dir = (working_dir / "predictions").resolve()
    assert Path(out["dataset_path"]).resolve() == expected_dataset
    assert Path(out["output_dir"]).resolve() == expected_output_dir


def test_predict_executor_does_not_import_tuning():
    """Same independence guarantee for the service-side executor.

    Reads the executor source via the filesystem rather than `find_spec`:
    the service package isn't always on `sys.path` when `physics_agent`
    tests run, so importlib lookup is unreliable here.
    """
    repo_root = Path(__file__).resolve().parents[3]
    executor_path = (
        repo_root
        / "apps"
        / "physics_agent_service"
        / "service"
        / "workers"
        / "predict_executor.py"
    )
    assert executor_path.exists(), f"predict_executor.py missing at {executor_path}"
    src = executor_path.read_text()
    _assert_no_forbidden_imports(src, "service.workers.predict_executor")


def test_predict_runtime_import_does_not_pull_tuning(tmp_path):
    """Runtime check: importing the predict path in a clean subprocess
    must not transitively load `physics_agent.tuning`, `botorch`, or
    `ovphysx`.

    The static-scan tests above only catch lexical imports in the predict
    files themselves. This subprocess test catches transitive imports via
    `physics_agent.api/__init__.py` (which eagerly imports many subpackages)
    and any dynamic imports anyone might add later.
    """
    import subprocess
    import sys as _sys

    script = tmp_path / "check.py"
    script.write_text(
        # Import the public predict surface AND the cli's lazy import path.
        "import sys\n"
        "import physics_agent.api  # noqa: F401\n"
        "from physics_agent.api import PredictInput, run_predict, arun_predict  # noqa: F401\n"
        "import physics_agent.api.predict  # noqa: F401\n"
        "leaked = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.startswith('physics_agent.tuning')\n"
        "    or m == 'botorch' or m.startswith('botorch.')\n"
        "    or m == 'ovphysx' or m.startswith('ovphysx.')\n"
        ")\n"
        "if leaked:\n"
        "    raise SystemExit(\n"
        "        'Predict path leaked tuning/BoTorch/OvPhysX imports: ' + repr(leaked)\n"
        "    )\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [_sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout
