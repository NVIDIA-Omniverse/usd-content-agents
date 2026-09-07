# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI surface tests for `physics-agent tune`.

The Acceptance Criteria require:
* `--help` exists.
* `--optimizer auto` resolves to BoTorch (or fails clearly when missing).
* `--optimizer random` and `--optimizer cma-es` remain available.
* Missing-BoTorch + missing-OvPhysX both surface the install hint and exit non-zero.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import click
import pytest
from typer.main import get_command
from typer.testing import CliRunner

from physics_agent.cli import app
from physics_agent.tuning.errors import TuningError

runner = CliRunner()


def _command_options(command_name: str) -> set[str]:
    command = get_command(app)
    assert isinstance(command, click.Group)
    return {
        option
        for parameter in command.commands[command_name].params
        if isinstance(parameter, click.Option)
        for option in parameter.opts
    }


@pytest.fixture(autouse=True)
def _isolate_cli_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI setup from replacing pytest's capture handlers."""
    monkeypatch.setattr(
        "physics_agent.cli.setup_logging",
        lambda **_kwargs: logging.getLogger("physics_agent.tests.tuning_cli"),
    )


def _write_scenario(tmp_path: Path) -> Path:
    p = tmp_path / "scenario.yaml"
    p.write_text(
        """
name: drop_settle
parameters:
  - name: mass_scale
    min: 0.5
    max: 2.0
  - name: static_friction
    min: 0.05
    max: 1.0
"""
    )
    return p


def _write_physics_usd(tmp_path: Path) -> Path:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    p = tmp_path / "physics.usda"
    stage = Usd.Stage.CreateNew(str(p))
    body = UsdGeom.Xform.Define(stage, "/Body")
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.0)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mat = UsdShade.Material.Define(stage, "/Mat")
    mat_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mat_api.CreateStaticFrictionAttr(0.4)
    mat_api.CreateDynamicFrictionAttr(0.3)
    mat_api.CreateRestitutionAttr(0.2)
    stage.SetDefaultPrim(body.GetPrim())
    stage.GetRootLayer().Save()
    return p


def test_tune_help_command_works() -> None:
    result = runner.invoke(app, ["tune", "--help"])
    assert result.exit_code == 0
    assert "tune" in result.stdout.lower()
    # All four optimizer names must appear in the help text.
    for name in ("auto", "botorch", "random", "cma-es"):
        assert name in result.stdout


def test_tune_random_optimizer_smoke(tmp_path: Path) -> None:
    """End-to-end CLI smoke with the fake backend + random optimizer."""
    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)
    out = tmp_path / "tune_out"
    result = runner.invoke(
        app,
        [
            "tune",
            str(sc),
            "--physics-usd",
            str(physics),
            "--engine",
            "fake",
            "--optimizer",
            "random",
            "--max-trials",
            "3",
            "--output-dir",
            str(out),
            "--seed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (out / "best_params.json").exists()
    assert (out / "history.jsonl").exists()
    assert (out / "tune_results.json").exists()
    assert (out / "report.md").exists()
    assert (out / "tuned_physics.usd").exists()


def test_tune_botorch_missing_exits_with_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning import optimizers

    monkeypatch.setattr(optimizers, "is_botorch_available", lambda: False)
    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)
    result = runner.invoke(
        app,
        [
            "tune",
            str(sc),
            "--physics-usd",
            str(physics),
            "--engine",
            "fake",
            "--optimizer",
            "auto",
            "--max-trials",
            "3",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    # Substring match — Rich may insert ANSI codes around the message but the
    # text content is preserved.
    assert "BoTorch optimizer requires the tuning extra" in result.stdout


def test_tune_explicit_botorch_missing_exits_with_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning import optimizers

    monkeypatch.setattr(optimizers, "is_botorch_available", lambda: False)
    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)
    result = runner.invoke(
        app,
        [
            "tune",
            str(sc),
            "--physics-usd",
            str(physics),
            "--engine",
            "fake",
            "--optimizer",
            "botorch",
            "--max-trials",
            "3",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "BoTorch optimizer requires the tuning extra" in result.stdout


@pytest.mark.parametrize("error_type", (ValueError, TuningError))
def test_tune_preserves_optimizer_resolution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    from physics_agent import tuning

    message = "Requested optimizer is unavailable; install its optional dependency"

    def fail_tune(_params: Any) -> None:
        raise error_type(message)

    monkeypatch.setattr(tuning, "run_tune", fail_tune)
    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)
    result = runner.invoke(
        app,
        [
            "tune",
            str(sc),
            "--physics-usd",
            str(physics),
            "--engine",
            "fake",
            "--optimizer",
            "random",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert message in result.stdout
    assert "Physics tuning failed" not in result.stdout


def test_tune_ovphysx_missing_exits_with_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning import backend as backend_mod
    from physics_agent.tuning.errors import OvPhysXUnavailableError

    monkeypatch.setattr(
        backend_mod,
        "load_ovphysx_backend",
        lambda: (_ for _ in ()).throw(OvPhysXUnavailableError()),
    )
    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)
    result = runner.invoke(
        app,
        [
            "tune",
            str(sc),
            "--physics-usd",
            str(physics),
            "--engine",
            "ovphysx",
            "--optimizer",
            "random",
            "--max-trials",
            "3",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "OvPhysX backend requires the tuning extra" in result.stdout


def test_tune_missing_scenario_file_errors_clearly(tmp_path: Path) -> None:
    physics = _write_physics_usd(tmp_path)
    result = runner.invoke(
        app,
        [
            "tune",
            str(tmp_path / "nope.yaml"),
            "--physics-usd",
            str(physics),
            "--engine",
            "fake",
            "--optimizer",
            "random",
        ],
    )
    assert result.exit_code != 0
    assert "Scenario file not found" in result.stdout


def test_tune_missing_physics_usd_errors_clearly(tmp_path: Path) -> None:
    sc = _write_scenario(tmp_path)
    result = runner.invoke(
        app,
        [
            "tune",
            str(sc),
            "--physics-usd",
            str(tmp_path / "missing.usda"),
            "--engine",
            "fake",
            "--optimizer",
            "random",
        ],
    )
    assert result.exit_code != 0
    assert "physics USD not found" in result.stdout


def test_tune_rejects_unsupported_reference_image_extension(tmp_path: Path) -> None:
    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)
    bad_ref = tmp_path / "reference.gif"
    bad_ref.write_bytes(b"not supported")

    result = runner.invoke(
        app,
        [
            "tune",
            str(sc),
            "--physics-usd",
            str(physics),
            "--engine",
            "fake",
            "--optimizer",
            "random",
            "--reference-image",
            str(bad_ref),
        ],
    )

    assert result.exit_code != 0
    assert "unsupported extension" in result.stdout
    assert "--reference-image" in result.stdout


def test_tune_default_physics_from_scenario_yaml(tmp_path: Path) -> None:
    """When `physics_usd` is set in the scenario YAML, --physics-usd is optional."""
    physics = _write_physics_usd(tmp_path)
    sc = tmp_path / "scenario.yaml"
    sc.write_text(
        f"""
name: drop_settle
physics_usd: {physics}
parameters:
  - name: mass_scale
    min: 0.5
    max: 2.0
"""
    )
    result = runner.invoke(
        app,
        [
            "tune",
            str(sc),
            "--engine",
            "fake",
            "--optimizer",
            "random",
            "--max-trials",
            "2",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_refine_rejects_removed_reference_video_option(tmp_path: Path) -> None:
    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"video")

    result = runner.invoke(
        app,
        [
            "refine",
            str(sc),
            "--physics-usd",
            str(physics),
            "--user-prompt",
            "make it bouncy",
            "--engine",
            "ovphysx",
            "--optimizer",
            "random",
            "--max-trials",
            "1",
            "--reference-video",
            str(video),
        ],
    )

    assert result.exit_code != 0
    assert "No such option" in result.output
    help_result = runner.invoke(app, ["refine", "--help"])
    assert help_result.exit_code == 0
    assert "--reference-video" not in help_result.output


def test_refine_fake_requires_explicit_text_only_mode(tmp_path: Path) -> None:
    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)

    result = runner.invoke(
        app,
        [
            "refine",
            str(sc),
            "--physics-usd",
            str(physics),
            "--user-prompt",
            "make it bouncy",
            "--engine",
            "fake",
            "--optimizer",
            "random",
        ],
    )

    assert result.exit_code == 2
    assert "--engine fake cannot produce the recording USD" in result.stdout
    assert "--no-visual-evidence" in result.stdout


def test_refine_rejects_unknown_optimizer_before_model_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.models import chat_models

    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)
    create_chat_model = Mock()
    monkeypatch.setattr(chat_models, "create_chat_model", create_chat_model)

    result = runner.invoke(
        app,
        [
            "refine",
            str(sc),
            "--physics-usd",
            str(physics),
            "--user-prompt",
            "make it bouncy",
            "--optimizer",
            "botroch",
        ],
    )

    assert result.exit_code == 2
    assert "Unknown optimizer 'botroch'" in result.stdout
    assert "Supported:" in result.stdout
    create_chat_model.assert_not_called()


def test_refine_cli_builds_and_passes_vlm_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.models import chat_models, vision_language_models
    from world_understanding.functions.models.backends import registry
    from world_understanding.utils import credentials

    import physics_agent.api.refine as refine_api

    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)
    out = tmp_path / "refine_out"
    built_chat = object()
    built_vlm = object()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(registry, "list_chat_backends", lambda: ["gemini"])
    monkeypatch.setattr(registry, "list_vlm_backends", lambda: ["gemini"])
    monkeypatch.setattr(
        credentials,
        "get_env_api_key_for_backend",
        lambda backend: "fake-key",
    )

    def fake_create_chat_model(**kwargs: Any) -> object:
        captured["chat_kwargs"] = kwargs
        return built_chat

    def fake_create_vlm(backend: str, **kwargs: Any) -> object:
        captured["vlm_backend"] = backend
        captured["vlm_kwargs"] = kwargs
        return built_vlm

    def fake_run_refine(params: refine_api.RefineInput) -> refine_api.RefineOutput:
        captured["refine_params"] = params
        return refine_api.RefineOutput(
            success=True,
            output_dir=Path(params.output_dir),
            iterations=[],
            iteration_count=0,
            final_iteration=0,
            final_dir=Path(params.output_dir),
            termination_reason="approved",
            user_prompt=params.user_prompt,
        )

    monkeypatch.setattr(chat_models, "create_chat_model", fake_create_chat_model)
    monkeypatch.setattr(vision_language_models, "create_vlm", fake_create_vlm)
    monkeypatch.setattr(refine_api, "run_refine", fake_run_refine)

    result = runner.invoke(
        app,
        [
            "refine",
            str(sc),
            "--physics-usd",
            str(physics),
            "--user-prompt",
            "make it bouncy",
            "--engine",
            "ovphysx",
            "--optimizer",
            "random",
            "--max-trials",
            "1",
            "--max-iterations",
            "1",
            "--history-window",
            "7",
            "--visual-evidence-timeout-seconds",
            "321",
            "--output-dir",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["chat_kwargs"]["backend"] == "gemini"
    assert captured["vlm_backend"] == "gemini"
    assert captured["vlm_kwargs"]["model"] == "gemini-3-pro-preview"
    assert "reasoning_effort" not in captured["vlm_kwargs"]
    assert captured["refine_params"].chat_model is built_chat
    assert captured["refine_params"].vlm_model is built_vlm
    assert captured["refine_params"].visual_evidence_enabled is True
    assert captured["refine_params"].visual_evidence_timeout_seconds == 321
    assert captured["refine_params"].history_window == 7


def test_refine_cli_no_visual_evidence_disables_judge_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.models import chat_models, vision_language_models
    from world_understanding.functions.models.backends import registry
    from world_understanding.utils import credentials

    import physics_agent.api.refine as refine_api

    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)
    out = tmp_path / "refine_out"
    captured: dict[str, Any] = {}

    monkeypatch.setattr(registry, "list_chat_backends", lambda: ["gemini"])
    monkeypatch.setattr(registry, "list_vlm_backends", lambda: ["gemini"])
    monkeypatch.setattr(
        credentials,
        "get_env_api_key_for_backend",
        lambda backend: "fake-key",
    )
    monkeypatch.setattr(chat_models, "create_chat_model", lambda **kwargs: object())
    monkeypatch.setattr(
        vision_language_models,
        "create_vlm",
        lambda **kwargs: object(),
    )

    def fake_run_refine(params: refine_api.RefineInput) -> refine_api.RefineOutput:
        captured["refine_params"] = params
        return refine_api.RefineOutput(
            success=True,
            output_dir=Path(params.output_dir),
            iterations=[],
            iteration_count=0,
            final_iteration=0,
            final_dir=Path(params.output_dir),
            termination_reason="approved",
            user_prompt=params.user_prompt,
        )

    monkeypatch.setattr(refine_api, "run_refine", fake_run_refine)

    result = runner.invoke(
        app,
        [
            "refine",
            str(sc),
            "--physics-usd",
            str(physics),
            "--user-prompt",
            "make it bouncy",
            "--engine",
            "fake",
            "--optimizer",
            "random",
            "--max-trials",
            "1",
            "--max-iterations",
            "1",
            "--no-visual-evidence",
            "--output-dir",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["refine_params"].visual_evidence_enabled is False
    assert captured["refine_params"].history_window == 20


@pytest.mark.parametrize("error_type", (ValueError, TuningError))
def test_refine_preserves_optimizer_resolution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    from world_understanding.functions.models import chat_models, vision_language_models
    from world_understanding.functions.models.backends import registry
    from world_understanding.utils import credentials

    import physics_agent.api.refine as refine_api

    message = "Requested optimizer is unavailable; install its optional dependency"

    monkeypatch.setattr(registry, "list_chat_backends", lambda: ["gemini"])
    monkeypatch.setattr(registry, "list_vlm_backends", lambda: ["gemini"])
    monkeypatch.setattr(
        credentials,
        "get_env_api_key_for_backend",
        lambda _backend: "fake-key",
    )
    monkeypatch.setattr(
        credentials,
        "apply_vlm_nim_env_override",
        lambda config: config,
    )
    monkeypatch.setattr(chat_models, "create_chat_model", lambda **_kwargs: object())
    monkeypatch.setattr(
        vision_language_models,
        "create_vlm",
        lambda **_kwargs: object(),
    )

    def fail_refine(_params: Any) -> None:
        raise error_type(message)

    monkeypatch.setattr(refine_api, "run_refine", fail_refine)
    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)
    result = runner.invoke(
        app,
        [
            "refine",
            str(sc),
            "--physics-usd",
            str(physics),
            "--user-prompt",
            "make it bouncy",
            "--engine",
            "fake",
            "--optimizer",
            "random",
            "--no-visual-evidence",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert message in result.stdout
    assert "Physics refinement failed" not in result.stdout


def test_refine_cli_passes_reasoning_effort_to_reasoning_vlm_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.agentic import config as agentic_config
    from world_understanding.functions.models import chat_models, vision_language_models
    from world_understanding.functions.models.backends import registry
    from world_understanding.utils import credentials

    import physics_agent.api.refine as refine_api
    from physics_agent.api.defaults import DEFAULT_VLM_REASONING_EFFORT

    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)
    built_chat = object()
    built_vlm = object()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(registry, "list_chat_backends", lambda: ["openai"])
    monkeypatch.setattr(registry, "list_vlm_backends", lambda: ["openai"])
    monkeypatch.setattr(
        credentials,
        "get_env_api_key_for_backend",
        lambda backend: "chat-key",
    )
    monkeypatch.setattr(
        agentic_config,
        "get_api_key_for_model_config",
        lambda backend, config, model_type: "vlm-key",
    )

    def fake_create_chat_model(**kwargs: Any) -> object:
        captured["chat_kwargs"] = kwargs
        return built_chat

    def fake_create_vlm(backend: str, **kwargs: Any) -> object:
        captured["vlm_backend"] = backend
        captured["vlm_kwargs"] = kwargs
        return built_vlm

    def fake_run_refine(params: refine_api.RefineInput) -> refine_api.RefineOutput:
        captured["refine_params"] = params
        return refine_api.RefineOutput(
            success=True,
            output_dir=Path(params.output_dir),
            iterations=[],
            iteration_count=0,
            final_iteration=0,
            final_dir=Path(params.output_dir),
            termination_reason="approved",
            user_prompt=params.user_prompt,
        )

    monkeypatch.setattr(chat_models, "create_chat_model", fake_create_chat_model)
    monkeypatch.setattr(vision_language_models, "create_vlm", fake_create_vlm)
    monkeypatch.setattr(refine_api, "run_refine", fake_run_refine)

    result = runner.invoke(
        app,
        [
            "refine",
            str(sc),
            "--physics-usd",
            str(physics),
            "--user-prompt",
            "make it bouncy",
            "--engine",
            "ovphysx",
            "--optimizer",
            "random",
            "--max-trials",
            "1",
            "--max-iterations",
            "1",
            "--output-dir",
            str(tmp_path / "refine_out"),
            "--chat-backend",
            "openai",
            "--chat-model",
            "gpt-5",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["chat_kwargs"]["backend"] == "openai"
    assert captured["vlm_backend"] == "openai"
    assert captured["vlm_kwargs"]["reasoning_effort"] == DEFAULT_VLM_REASONING_EFFORT
    assert captured["refine_params"].chat_model is built_chat
    assert captured["refine_params"].vlm_model is built_vlm


def test_refine_cli_rejects_vlm_nim_env_backend_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.agentic import config as agentic_config
    from world_understanding.functions.models import chat_models, vision_language_models
    from world_understanding.functions.models.backends import registry
    from world_understanding.utils import credentials

    import physics_agent.api.refine as refine_api

    for env_var in (
        "WU_VLM_NIM_BASE_URL",
        "PA_VLM_NIM_BASE_URL",
        "TA_VLM_NIM_BASE_URL",
        "MA_VLM_NIM_BASE_URL",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("PA_VLM_NIM_BASE_URL", "http://localhost:9000/v1")

    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)
    built_chat = object()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(registry, "list_chat_backends", lambda: ["gemini"])
    monkeypatch.setattr(registry, "list_vlm_backends", lambda: ["gemini", "nim"])
    monkeypatch.setattr(
        credentials,
        "get_env_api_key_for_backend",
        lambda backend: "chat-key",
    )
    monkeypatch.setattr(
        agentic_config,
        "get_api_key_for_model_config",
        lambda backend, config, model_type: "vlm-key",
    )

    def fake_create_chat_model(**kwargs: Any) -> object:
        captured["chat_kwargs"] = kwargs
        return built_chat

    def fake_create_vlm(backend: str, **kwargs: Any) -> object:
        captured["vlm_backend"] = backend
        captured["vlm_kwargs"] = kwargs
        raise AssertionError("VLM should not be constructed after backend override")

    def fake_run_refine(params: refine_api.RefineInput) -> refine_api.RefineOutput:
        captured["refine_params"] = params
        return refine_api.RefineOutput(
            success=True,
            output_dir=Path(params.output_dir),
            iterations=[],
            iteration_count=0,
            final_iteration=0,
            final_dir=Path(params.output_dir),
            termination_reason="approved",
            user_prompt=params.user_prompt,
        )

    monkeypatch.setattr(chat_models, "create_chat_model", fake_create_chat_model)
    monkeypatch.setattr(vision_language_models, "create_vlm", fake_create_vlm)
    monkeypatch.setattr(refine_api, "run_refine", fake_run_refine)

    result = runner.invoke(
        app,
        [
            "refine",
            str(sc),
            "--physics-usd",
            str(physics),
            "--user-prompt",
            "make it bouncy",
            "--engine",
            "ovphysx",
            "--optimizer",
            "random",
            "--max-trials",
            "1",
            "--max-iterations",
            "1",
            "--output-dir",
            str(tmp_path / "refine_out"),
        ],
    )

    assert result.exit_code != 0
    assert "VLM judge backend would be overridden" in result.stdout
    assert captured["chat_kwargs"]["backend"] == "gemini"
    assert "vlm_backend" not in captured
    assert "refine_params" not in captured


def test_tune_external_help_command_works() -> None:
    result = runner.invoke(
        app,
        ["tune-external", "--help"],
        terminal_width=240,
        color=False,
    )

    assert result.exit_code == 0
    assert "qualification" in result.stdout.lower()
    options = _command_options("tune-external")
    assert "--approve-qualification" in options
    assert "--render-winning-trial" in options


def test_tune_external_stops_after_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tuning import external
    from physics_agent.tuning.external import ExternalTuneOutput

    secret = "cli-tune-secret"
    config = tmp_path / f"runtime?token={secret}.json"
    config.write_text("{}", encoding="utf-8")
    qualification_dir = tmp_path / f"run?token={secret}"
    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(
        external,
        "run_external_tune",
        lambda _params: ExternalTuneOutput(
            success=True,
            status="awaiting_approval",
            output_dir=qualification_dir,
            qualification_digest=digest,
            qualification_path=qualification_dir / "qualification.json",
        ),
    )

    result = runner.invoke(
        app,
        [
            "tune-external",
            str(config),
            "--output-dir",
            str(tmp_path / "run"),
        ],
    )

    assert result.exit_code == 0
    assert "review required" in result.stdout.lower()
    assert digest in result.stdout
    assert secret not in result.stdout


def test_tune_external_displays_completed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tuning import ReplicaRecord, TrialRecord, external
    from physics_agent.tuning.external import ExternalTuneOutput

    config = tmp_path / "runtime.json"
    config.write_text("{}", encoding="utf-8")
    replica = ReplicaRecord(seed=1, objective_value=0.1, success=True)
    candidate = TrialRecord(
        trial_index=0,
        params={"gain": 0.75},
        score=0.1,
        objective_value=0.1,
        replicas=[replica],
    )
    captured: dict[str, Any] = {}
    approval_digest = "sha256:" + "c" * 64
    qualification_frames = tmp_path / "run" / "qualification_frames.json"
    render_frame = tmp_path / "run" / "render" / "frame.png"

    def fake_external_tune(params):
        captured["params"] = params
        return ExternalTuneOutput(
            success=True,
            status="completed",
            output_dir=tmp_path / "run",
            optimizer_used="botorch",
            best_params={"gain": 0.75},
            best_objective=0.1,
            n_trials=1,
            history=[candidate],
            rendered_frames=[render_frame],
            render_error="RendererWarning",
            artifacts={"qualification_frames": qualification_frames},
        )

    monkeypatch.setattr(
        external,
        "run_external_tune",
        fake_external_tune,
    )

    result = runner.invoke(
        app,
        [
            "tune-external",
            str(config),
            "--approve-qualification",
            approval_digest,
            "--render-winning-trial",
        ],
    )

    assert result.exit_code == 0
    assert "completed" in result.stdout
    assert "botorch" in result.stdout
    assert "best.gain" in result.stdout
    assert "Qualification frames" in result.stdout
    assert "Best-trial render" in result.stdout
    assert "RendererWarning" in result.stdout
    assert captured["params"].render_winning_trial is True
    assert captured["params"].approval_digest == approval_digest


def test_tune_external_failure_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tuning import external
    from physics_agent.tuning.external import ExternalTuneOutput

    config = tmp_path / "runtime.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        external,
        "run_external_tune",
        lambda _params: ExternalTuneOutput(
            success=False,
            status="qualification_failed",
            error="adapter failed",
        ),
    )

    result = runner.invoke(app, ["tune-external", str(config)])

    assert result.exit_code == 1
    assert "adapter failed" in result.stdout


def test_tune_external_rejects_file_output_directory(tmp_path: Path) -> None:
    config = tmp_path / "runtime.json"
    config.write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    output.write_text("not a directory", encoding="utf-8")

    result = runner.invoke(
        app,
        ["tune-external", str(config), "--output-dir", str(output)],
    )

    assert result.exit_code == 1
    assert "--output-dir must be a directory" in result.stdout


def test_tune_external_runtime_exception_is_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tuning import external

    secret = "tune-runtime-secret"
    config = tmp_path / "runtime.json"
    config.write_text("{}", encoding="utf-8")

    def fail_external_tune(_params):
        raise RuntimeError(f"adapter exposed {secret}")

    monkeypatch.setattr(external, "run_external_tune", fail_external_tune)

    result = runner.invoke(app, ["tune-external", str(config)])

    assert result.exit_code == 1
    assert "External tuning failed" in result.stdout
    assert secret not in result.stdout


def test_refine_external_help_command_works() -> None:
    result = runner.invoke(
        app,
        ["refine-external", "--help"],
        terminal_width=240,
        color=False,
    )

    assert result.exit_code == 0
    options = _command_options("refine-external")
    assert "--user-prompt" in options
    assert "--approve-qualification" in options
    assert "--score-threshold" in options


def test_refine_external_qualification_does_not_build_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent import cli
    from physics_agent.tuning import external
    from physics_agent.tuning.external import ExternalRefineOutput

    secret = "cli-refine-secret"
    config = tmp_path / f"runtime?token={secret}.json"
    config.write_text("{}", encoding="utf-8")
    qualification_dir = tmp_path / f"run?token={secret}"
    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(
        cli,
        "_build_external_refine_models",
        lambda **_kwargs: pytest.fail("qualification must not build models"),
    )
    monkeypatch.setattr(
        external,
        "run_external_refine",
        lambda _params: ExternalRefineOutput(
            success=True,
            status="awaiting_approval",
            termination_reason="awaiting_approval",
            output_dir=qualification_dir,
            qualification_digest=digest,
            qualification_path=qualification_dir / "qualification.json",
        ),
    )

    result = runner.invoke(
        app,
        [
            "refine-external",
            str(config),
            "--user-prompt",
            "make it bounce",
            "--output-dir",
            str(tmp_path / "run"),
        ],
    )

    assert result.exit_code == 0
    assert "review required" in result.stdout.lower()
    assert digest in result.stdout
    assert secret not in result.stdout


@pytest.mark.parametrize(
    ("termination_reason", "validated", "expected_validation"),
    [
        ("approved", True, "yes"),
        ("max_iterations", False, "no"),
    ],
)
def test_refine_external_displays_completed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termination_reason: str,
    validated: bool,
    expected_validation: str,
) -> None:
    from physics_agent import cli
    from physics_agent.tuning import external
    from physics_agent.tuning.external import ExternalRefineOutput

    config = tmp_path / "runtime.json"
    config.write_text("{}", encoding="utf-8")
    final_dir = tmp_path / "run" / "final"
    (final_dir / "render").mkdir(parents=True)
    captured: dict[str, Any] = {}
    approval_digest = "sha256:" + "b" * 64
    monkeypatch.setattr(
        cli,
        "_build_external_refine_models",
        lambda **_kwargs: ("chat", "vlm"),
    )

    def fake_external_refine(params):
        captured["params"] = params
        return ExternalRefineOutput(
            success=True,
            status="completed",
            termination_reason=termination_reason,
            validated=validated,
            output_dir=tmp_path / "run",
            final_best_params={"restitution": 0.8},
            final_objective=0.1,
            final_optimizer_loss=0.1,
            final_dir=final_dir,
        )

    monkeypatch.setattr(
        external,
        "run_external_refine",
        fake_external_refine,
    )

    result = runner.invoke(
        app,
        [
            "refine-external",
            str(config),
            "--user-prompt",
            "make it bounce",
            "--approve-qualification",
            approval_digest,
        ],
    )

    assert result.exit_code == 0
    assert termination_reason in result.stdout
    assert "VLM validated" in result.stdout
    validation_line = next(
        line for line in result.stdout.splitlines() if "VLM validated" in line
    )
    assert expected_validation in validation_line
    assert "best.restitution" in result.stdout
    assert "Best-trial render" in result.stdout
    assert captured["params"].approval_digest == approval_digest


def _stub_external_model_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chat_backends: list[str],
    vlm_backends: list[str],
    api_key: str | None = None,
    requires_key: bool = False,
    override_backend: str | None = None,
    resolved_key: str | None = None,
    supports_reasoning: bool = True,
) -> dict[str, Any]:
    from world_understanding.agentic import config as agentic_config
    from world_understanding.functions.models import (
        chat_models,
        vision_language_models,
    )
    from world_understanding.functions.models.backends import registry
    from world_understanding.utils import credentials

    from physics_agent.tuning import visual_evidence

    captured: dict[str, Any] = {}
    monkeypatch.setattr(registry, "list_chat_backends", lambda: chat_backends)
    monkeypatch.setattr(registry, "list_vlm_backends", lambda: vlm_backends)

    def apply_override(config: dict[str, Any]) -> dict[str, Any]:
        return {
            **config,
            **({"backend": override_backend} if override_backend is not None else {}),
        }

    monkeypatch.setattr(credentials, "apply_llm_nim_env_override", apply_override)
    monkeypatch.setattr(credentials, "apply_vlm_nim_env_override", apply_override)

    def resolve_key(_backend: str, _config: dict[str, Any], _model_type: str) -> str:
        if resolved_key is not None:
            return resolved_key
        if api_key is not None:
            return api_key
        if requires_key:
            raise ValueError("API key is required")
        return ""

    monkeypatch.setattr(
        agentic_config,
        "get_api_key_for_model_config",
        resolve_key,
    )
    monkeypatch.setattr(
        visual_evidence,
        "backend_supports_reasoning_effort",
        lambda _backend, _model=None: supports_reasoning,
    )

    def create_chat_model(**kwargs: Any) -> str:
        captured["chat"] = kwargs
        return "chat-model"

    def create_vlm(**kwargs: Any) -> str:
        captured["vlm"] = kwargs
        return "vlm-model"

    monkeypatch.setattr(chat_models, "create_chat_model", create_chat_model)
    monkeypatch.setattr(vision_language_models, "create_vlm", create_vlm)
    return captured


def test_build_external_refine_models_validates_registered_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent import cli

    _stub_external_model_modules(
        monkeypatch,
        chat_backends=[],
        vlm_backends=["mock"],
    )
    with pytest.raises(ValueError, match="chat backend"):
        cli._build_external_refine_models(
            backend="mock",
            model="model",
        )

    _stub_external_model_modules(
        monkeypatch,
        chat_backends=["mock"],
        vlm_backends=[],
    )
    with pytest.raises(ValueError, match="not registered as a VLM"):
        cli._build_external_refine_models(
            backend="mock",
            model="model",
        )


def test_build_external_refine_models_requires_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent import cli

    _stub_external_model_modules(
        monkeypatch,
        chat_backends=["mock"],
        vlm_backends=["mock"],
        requires_key=True,
    )

    with pytest.raises(ValueError, match="API key"):
        cli._build_external_refine_models(
            backend="mock",
            model="model",
        )


def test_build_external_refine_models_rejects_backend_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent import cli

    _stub_external_model_modules(
        monkeypatch,
        chat_backends=["mock"],
        vlm_backends=["mock"],
        override_backend="other",
    )

    with pytest.raises(ValueError, match="different backend"):
        cli._build_external_refine_models(
            backend="mock",
            model="model",
        )


def test_build_external_refine_models_constructs_chat_and_vlm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent import cli

    captured = _stub_external_model_modules(
        monkeypatch,
        chat_backends=["mock"],
        vlm_backends=["mock"],
        api_key="environment-key",
        resolved_key="resolved-key",
        supports_reasoning=False,
    )

    chat_model, vlm_model = cli._build_external_refine_models(
        backend="mock",
        model="model",
    )

    assert (chat_model, vlm_model) == ("chat-model", "vlm-model")
    assert captured["chat"]["api_key"] == "resolved-key"
    assert captured["vlm"]["api_key"] == "resolved-key"
    assert "max_tokens" not in captured["vlm"]
    assert "temperature" not in captured["vlm"]
    assert "reasoning_effort" not in captured["vlm"]


def test_build_external_refine_models_routes_local_nim_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.models import (
        chat_models,
        vision_language_models,
    )
    from world_understanding.functions.models.backends import registry

    from physics_agent import cli
    from physics_agent.tuning import visual_evidence

    for name in (
        "WU_LLM_NIM_BASE_URL",
        "PA_LLM_NIM_BASE_URL",
        "TA_LLM_NIM_BASE_URL",
        "MA_LLM_NIM_BASE_URL",
        "WU_VLM_NIM_BASE_URL",
        "PA_VLM_NIM_BASE_URL",
        "TA_VLM_NIM_BASE_URL",
        "MA_VLM_NIM_BASE_URL",
        "WU_NIM_API_KEY",
        "PA_NIM_API_KEY",
        "TA_NIM_API_KEY",
        "MA_NIM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PA_LLM_NIM_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("PA_VLM_NIM_BASE_URL", "http://127.0.0.1:8002/v1")
    monkeypatch.setenv("WU_NIM_API_KEY", "not-used")
    monkeypatch.setattr(registry, "list_chat_backends", lambda: ["nim"])
    monkeypatch.setattr(registry, "list_vlm_backends", lambda: ["nim"])
    monkeypatch.setattr(
        visual_evidence,
        "backend_supports_reasoning_effort",
        lambda _backend, _model=None: False,
    )
    captured: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(
        chat_models,
        "create_chat_model",
        lambda **kwargs: captured.setdefault("chat", kwargs),
    )
    monkeypatch.setattr(
        vision_language_models,
        "create_vlm",
        lambda **kwargs: captured.setdefault("vlm", kwargs),
    )

    cli._build_external_refine_models(backend="nim", model="model")

    assert captured["chat"]["base_url"] == "http://127.0.0.1:8001/v1"
    assert captured["vlm"]["base_url"] == "http://127.0.0.1:8002/v1"
    assert captured["chat"]["api_key"] == "not-used"
    assert captured["vlm"]["api_key"] == "not-used"


def test_external_commands_reject_missing_config(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    tune_result = runner.invoke(app, ["tune-external", str(missing)])
    refine_result = runner.invoke(
        app,
        [
            "refine-external",
            str(missing),
            "--user-prompt",
            "goal",
        ],
    )

    assert tune_result.exit_code == 1
    assert "Configuration file not found" in tune_result.stdout
    assert refine_result.exit_code == 1
    assert "Configuration file not found" in refine_result.stdout


def test_refine_external_validates_output_prompt_and_descriptions(
    tmp_path: Path,
) -> None:
    config = tmp_path / "runtime.json"
    config.write_text("{}", encoding="utf-8")
    output_file = tmp_path / "output"
    output_file.write_text("file", encoding="utf-8")

    output_result = runner.invoke(
        app,
        [
            "refine-external",
            str(config),
            "--user-prompt",
            "goal",
            "--output-dir",
            str(output_file),
        ],
    )
    empty_prompt = runner.invoke(
        app,
        [
            "refine-external",
            str(config),
            "--user-prompt",
            " ",
        ],
    )
    image_description = runner.invoke(
        app,
        [
            "refine-external",
            str(config),
            "--user-prompt",
            "goal",
            "--reference-description",
            "image",
        ],
    )
    unsupported_video_option = runner.invoke(
        app,
        [
            "refine-external",
            str(config),
            "--user-prompt",
            "goal",
            "--reference-video-description",
            "video",
        ],
    )

    assert output_result.exit_code == 1
    assert "must be a directory" in output_result.stdout
    assert empty_prompt.exit_code == 1
    assert "must not be empty" in empty_prompt.stdout
    assert image_description.exit_code == 1
    assert "per --reference-image" in image_description.stdout
    assert unsupported_video_option.exit_code == 2
    assert "No such option" in unsupported_video_option.output
    help_result = runner.invoke(app, ["refine-external", "--help"])
    assert help_result.exit_code == 0
    assert "--reference-video-description" not in help_result.output


def test_refine_external_failure_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tuning import external
    from physics_agent.tuning.external import ExternalRefineOutput

    config = tmp_path / "runtime.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        external,
        "run_external_refine",
        lambda _params: ExternalRefineOutput(
            success=False,
            status="failed",
            error="judge failed",
        ),
    )

    result = runner.invoke(
        app,
        [
            "refine-external",
            str(config),
            "--user-prompt",
            "goal",
        ],
    )

    assert result.exit_code == 1
    assert "judge failed" in result.stdout


def test_refine_external_model_exception_is_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent import cli

    secret = "refine-model-secret"
    config = tmp_path / "runtime.json"
    config.write_text("{}", encoding="utf-8")

    def fail_model_build(**_kwargs):
        raise RuntimeError(f"model setup exposed {secret}")

    monkeypatch.setattr(cli, "_build_external_refine_models", fail_model_build)

    result = runner.invoke(
        app,
        [
            "refine-external",
            str(config),
            "--user-prompt",
            "goal",
            "--approve-qualification",
            "sha256:" + "a" * 64,
        ],
    )

    assert result.exit_code == 1
    assert "Could not build refine models" in result.stdout
    assert secret not in result.stdout


def test_refine_external_runtime_exception_is_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tuning import external

    secret = "refine-runtime-secret"
    config = tmp_path / "runtime.json"
    config.write_text("{}", encoding="utf-8")

    def fail_external_refine(_params):
        raise RuntimeError(f"adapter exposed {secret}")

    monkeypatch.setattr(external, "run_external_refine", fail_external_refine)

    result = runner.invoke(
        app,
        [
            "refine-external",
            str(config),
            "--user-prompt",
            "goal",
        ],
    )

    assert result.exit_code == 1
    assert "External refinement failed" in result.stdout
    assert secret not in result.stdout
