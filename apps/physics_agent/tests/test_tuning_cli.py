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

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from physics_agent.cli import app

runner = CliRunner()


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


def test_refine_rejects_unsupported_reference_video_extension(tmp_path: Path) -> None:
    sc = _write_scenario(tmp_path)
    physics = _write_physics_usd(tmp_path)
    bad_ref = tmp_path / "reference.txt"
    bad_ref.write_text("not a video", encoding="utf-8")

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
            "--reference-video",
            str(bad_ref),
        ],
    )

    assert result.exit_code != 0
    assert "unsupported extension" in result.stdout
    assert "--reference-video" in result.stdout


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
            "fake",
            "--optimizer",
            "random",
            "--max-trials",
            "1",
            "--max-iterations",
            "1",
            "--history-window",
            "7",
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
            "fake",
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
            "fake",
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
