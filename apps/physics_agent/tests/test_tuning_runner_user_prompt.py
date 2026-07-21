# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the Part-1.1 wiring inside the tuning runner.

These tests cover the seam where ``_do_run_tune`` dispatches between the
explicit-YAML path, the ``user_prompt`` interpreter path, and the
end-of-tune judge step. They monkeypatch the *imported bindings* of
``infer_scenario_from_prompt`` and ``run_tune_judge`` (both of which the
runner pulls in lazily via ``from … import …`` inside the function body),
plus ``_resolve_judge_vlm_lazy`` so we never touch a real model provider.

The most spec-critical case is
``test_no_judge_byte_identical_baseline_keys``: with ``enable_judge=False``
and ``user_prompt=None`` the tune_results.json and report.md must omit
every Part-1.1 field — that's the hard backwards-compat guarantee from
closed issue #51.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from physics_agent.tuning import TuneInput, TuningError, run_tune
from physics_agent.tuning.artifacts import (
    ARTIFACT_REPORT,
    ARTIFACT_RESULTS,
    write_report_md,
)
from physics_agent.tuning.scenario import parse_scenario
from physics_agent.tuning.types import Scenario, TrialRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scenario_dict() -> dict[str, Any]:
    return {
        "name": "drop_settle",
        "metric": "settle_distance",
        "target": {"drop_height_m": 0.5, "duration_s": 2.0},
        "parameters": [
            {"name": "mass_scale", "min": 0.5, "max": 2.0},
        ],
    }


def _freeform_scenario_dict() -> dict[str, Any]:
    return {
        "name": "freeform",
        "metric": "judge_score",
        "target": {
            "description": "realistic generated rollout",
            "duration_s": 1.0,
            "observations": ["the object behaves realistically"],
        },
        "parameters": [
            {"name": "restitution", "min": 0.0, "max": 1.0},
        ],
    }


def _physics_usd(tmp_path: Path) -> Path:
    """Minimal physics-authored USD for the runner. Mirrors the helper in
    ``test_tuning_runner.py`` to keep these integration tests self-contained.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    p = tmp_path / "physics.usda"
    stage = Usd.Stage.CreateNew(str(p))
    body = UsdGeom.Xform.Define(stage, "/Body")
    mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass_api.CreateMassAttr(2.0)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

    mat = UsdShade.Material.Define(stage, "/Mat")
    mat_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mat_api.CreateStaticFrictionAttr(0.4)
    mat_api.CreateDynamicFrictionAttr(0.3)
    mat_api.CreateRestitutionAttr(0.2)

    stage.SetDefaultPrim(body.GetPrim())
    stage.GetRootLayer().Save()
    return p


def _patch_judge_chat_model_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the judge VLM resolver to return None.

    The test environment may or may not have a working
    ``world_understanding.functions.models.vision_language_models.create_vlm``;
    pinning the resolver to None makes judge-bearing tests deterministic
    (judge runs in programmatic-only mode with ``llm_unavailable=True``).
    """
    monkeypatch.setattr(
        "physics_agent.tuning.runner._resolve_judge_vlm_lazy",
        lambda: None,
    )


def test_default_judge_vlm_setup_failure_uses_value_free_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "physics-judge-setup-api-key-sentinel"

    def _raise_provider_error() -> None:
        raise RuntimeError(f"provider rejected api_key={sentinel}")

    monkeypatch.setattr(
        "physics_agent.tuning.runner.resolve_default_judge_vlm",
        _raise_provider_error,
    )
    from physics_agent.tuning import runner as runner_module

    with caplog.at_level("WARNING", logger=runner_module.__name__):
        assert runner_module._resolve_judge_vlm_lazy() is None

    assert sentinel not in caplog.text
    assert caplog.messages == [
        "Failed to instantiate default judge VLM; judge will be marked unavailable."
    ]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_rejects_both_scenario_and_user_prompt_unset(
    tmp_path: Path,
) -> None:
    """At least one of scenario / user_prompt must be supplied."""
    with pytest.raises(ValueError, match="scenario.*user_prompt|user_prompt.*scenario"):
        run_tune(
            TuneInput(
                physics_usd=_physics_usd(tmp_path),
                output_dir=tmp_path / "out",
                engine="fake",
                optimizer="random",
                max_trials=2,
                enable_judge=False,
            )
        )


def test_validation_rejects_judge_max_iterations_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="judge_max_iterations"):
        run_tune(
            TuneInput(
                scenario=_scenario_dict(),
                physics_usd=_physics_usd(tmp_path),
                output_dir=tmp_path / "out",
                engine="fake",
                optimizer="random",
                max_trials=2,
                judge_max_iterations=0,
                enable_judge=False,
            )
        )


def test_validation_rejects_judge_max_tokens_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="judge_max_tokens"):
        run_tune(
            TuneInput(
                scenario=_scenario_dict(),
                physics_usd=_physics_usd(tmp_path),
                output_dir=tmp_path / "out",
                engine="fake",
                optimizer="random",
                max_trials=2,
                judge_max_tokens=0,
                enable_judge=False,
            )
        )


def test_validation_rejects_judge_temperature_negative(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="judge_temperature"):
        run_tune(
            TuneInput(
                scenario=_scenario_dict(),
                physics_usd=_physics_usd(tmp_path),
                output_dir=tmp_path / "out",
                engine="fake",
                optimizer="random",
                max_trials=2,
                judge_temperature=-0.1,
                enable_judge=False,
            )
        )


# ---------------------------------------------------------------------------
# Scenario resolution dispatch
# ---------------------------------------------------------------------------


def test_scenario_only_skips_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit-YAML path must NOT invoke the NL interpreter."""

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "interpreter must not be invoked when only scenario is supplied"
        )

    monkeypatch.setattr(
        "physics_agent.tasks.interpret_user_prompt_tuning.infer_scenario_from_prompt",
        _explode,
    )
    _patch_judge_chat_model_to_none(monkeypatch)

    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=_physics_usd(tmp_path),
            output_dir=tmp_path / "out",
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=False,
        )
    )
    assert result.success
    # No interpreter cache file written.
    assert not (tmp_path / "out" / "inferred_scenario.json").exists()


def test_user_prompt_only_invokes_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When only ``user_prompt`` is set the runner calls the interpreter and
    receives a Scenario back; subsequent optimizer trials use that scenario.
    """
    captured: dict[str, Any] = {}

    def _fake_infer(
        user_prompt: str,
        *,
        scenario_override: dict[str, Any] | None = None,
        chat_model: Any = None,
        audit_dir: Path | None = None,
        physics_usd: Path | None = None,
        backend_name: str | None = None,
        supported_param_keys: tuple[str, ...] | None = None,
    ) -> Scenario:
        captured["user_prompt"] = user_prompt
        captured["scenario_override"] = scenario_override
        captured["audit_dir"] = audit_dir
        captured["physics_usd"] = physics_usd
        captured["backend_name"] = backend_name
        captured["supported_param_keys"] = supported_param_keys
        # Author a minimal Scenario via the existing parser so it's
        # guaranteed structurally valid.
        return parse_scenario(_scenario_dict())

    monkeypatch.setattr(
        "physics_agent.tasks.interpret_user_prompt_tuning.infer_scenario_from_prompt",
        _fake_infer,
    )
    _patch_judge_chat_model_to_none(monkeypatch)

    physics_usd = _physics_usd(tmp_path)
    out = tmp_path / "out"
    result = run_tune(
        TuneInput(
            user_prompt="make this object bouncy",
            physics_usd=physics_usd,
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=False,
        )
    )
    assert result.success
    assert captured["user_prompt"] == "make this object bouncy"
    assert captured["scenario_override"] is None
    assert captured["audit_dir"] == out
    assert captured["physics_usd"] == physics_usd
    assert captured["backend_name"] == "fake"
    assert set(captured["supported_param_keys"]) == {
        "mass_scale",
        "static_friction",
        "dynamic_friction",
        "restitution",
    }


def test_backend_shutdown_runs_when_binding_resolution_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backend resources must be released if parameter binding fails before
    the trial loop starts."""
    events: list[str] = []

    class _Backend:
        name = "fake"

        def warmup(self) -> None:
            events.append("warmup")

        def tuning_capabilities(self) -> tuple[Any, ...]:
            return ()

        def evaluate(self, *a: Any, **kw: Any) -> dict[str, Any]:
            raise AssertionError("evaluate should not run after binding failure")

        def shutdown(self) -> None:
            events.append("shutdown")

    import physics_agent.tuning.runner as runner_mod

    monkeypatch.setattr(runner_mod, "get_backend", lambda engine: _Backend())
    _patch_judge_chat_model_to_none(monkeypatch)

    with pytest.raises(TuningError, match="does not support tunable parameter"):
        run_tune(
            TuneInput(
                scenario=_scenario_dict(),
                physics_usd=_physics_usd(tmp_path),
                output_dir=tmp_path / "out",
                engine="fake",
                optimizer="random",
                max_trials=2,
                enable_judge=False,
            )
        )

    assert events == ["warmup", "shutdown"]


def test_both_supplied_passes_yaml_as_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both YAML + user_prompt are supplied, the YAML must be passed
    to the interpreter as ``scenario_override`` so explicit fields win."""
    captured: dict[str, Any] = {}

    def _fake_infer(user_prompt: str, **kwargs: Any) -> Scenario:
        captured["user_prompt"] = user_prompt
        captured["scenario_override"] = kwargs.get("scenario_override")
        return parse_scenario(_scenario_dict())

    monkeypatch.setattr(
        "physics_agent.tasks.interpret_user_prompt_tuning.infer_scenario_from_prompt",
        _fake_infer,
    )
    _patch_judge_chat_model_to_none(monkeypatch)

    explicit = {
        "name": "drop_settle",
        "metric": "settle_distance",
        "target": {"drop_height_m": 1.0, "duration_s": 3.0},
        "parameters": [{"name": "restitution", "min": 0.7, "max": 1.0}],
    }

    run_tune(
        TuneInput(
            scenario=explicit,
            user_prompt="make it bouncy",
            physics_usd=_physics_usd(tmp_path),
            output_dir=tmp_path / "out",
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=False,
        )
    )
    assert captured["scenario_override"] == explicit


def test_user_prompt_yaml_path_loaded_as_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the explicit scenario is a Path, the runner must read+parse the
    YAML and pass the resulting dict as scenario_override to the interpreter.
    """
    captured: dict[str, Any] = {}

    def _fake_infer(user_prompt: str, **kwargs: Any) -> Scenario:
        captured["scenario_override"] = kwargs.get("scenario_override")
        return parse_scenario(_scenario_dict())

    monkeypatch.setattr(
        "physics_agent.tasks.interpret_user_prompt_tuning.infer_scenario_from_prompt",
        _fake_infer,
    )
    _patch_judge_chat_model_to_none(monkeypatch)

    yaml_path = tmp_path / "scenario.yaml"
    yaml_path.write_text(
        "name: drop_settle\n"
        "metric: settle_distance\n"
        "target:\n  drop_height_m: 0.7\n  duration_s: 2.5\n"
        "parameters:\n  - name: mass_scale\n    min: 0.6\n    max: 1.4\n",
        encoding="utf-8",
    )

    run_tune(
        TuneInput(
            scenario=yaml_path,
            user_prompt="make it bouncy",
            physics_usd=_physics_usd(tmp_path),
            output_dir=tmp_path / "out",
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=False,
        )
    )
    override = captured["scenario_override"]
    assert override is not None
    assert override["name"] == "drop_settle"
    assert override["target"]["drop_height_m"] == 0.7


def test_backend_warmup_runs_before_llm_interpreter_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 14 (Codex CX P2#3): on the user_prompt path the backend's
    ``warmup`` must run BEFORE the NL interpreter. A box without the
    ovphysx daemon venv would otherwise burn a paid LLM request before
    discovering that the local precondition is missing."""
    call_order: list[str] = []

    class _RecordingBackend:
        name = "fake"

        def warmup(self) -> None:
            call_order.append("warmup")

        def evaluate(self, *a: Any, **kw: Any) -> dict[str, Any]:
            return {"score": 0.5}

        def shutdown(self) -> None:
            pass

    def _fake_infer(user_prompt: str, **kwargs: Any) -> Scenario:
        call_order.append("interpreter")
        return parse_scenario(_scenario_dict())

    monkeypatch.setattr(
        "physics_agent.tasks.interpret_user_prompt_tuning.infer_scenario_from_prompt",
        _fake_infer,
    )
    _patch_judge_chat_model_to_none(monkeypatch)
    import physics_agent.tuning.runner as runner_mod

    monkeypatch.setattr(runner_mod, "get_backend", lambda engine: _RecordingBackend())

    run_tune(
        TuneInput(
            user_prompt="make it bouncy",
            physics_usd=_physics_usd(tmp_path),
            output_dir=tmp_path / "out",
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=False,
        )
    )
    assert "warmup" in call_order, call_order
    assert "interpreter" in call_order, call_order
    assert call_order.index("warmup") < call_order.index("interpreter"), call_order


def test_newton_restitution_override_rejected_before_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit scenario fields win over the LLM, so unsupported Newton params
    in an override can be rejected before backend warmup or the interpreter."""

    def _unexpected_infer(*args: Any, **kwargs: Any) -> Scenario:
        raise AssertionError("interpreter should not be called")

    monkeypatch.setattr(
        "physics_agent.tasks.interpret_user_prompt_tuning.infer_scenario_from_prompt",
        _unexpected_infer,
    )
    import physics_agent.tuning.runner as runner_mod

    get_backend_calls = 0

    def _unexpected_get_backend(engine: str) -> Any:
        nonlocal get_backend_calls
        get_backend_calls += 1
        raise AssertionError("backend should not be constructed")

    monkeypatch.setattr(runner_mod, "get_backend", _unexpected_get_backend)

    with pytest.raises(TuningError, match="does not support tuning restitution"):
        run_tune(
            TuneInput(
                user_prompt="make it bouncy",
                scenario={
                    "name": "drop_settle",
                    "metric": "max_bounce_height",
                    "target": {"drop_height_m": 0.5, "duration_s": 1.0},
                    "parameters": [
                        {"name": "restitution", "min": 0.0, "max": 1.0},
                    ],
                },
                physics_usd=_physics_usd(tmp_path),
                output_dir=tmp_path / "out",
                engine="newton",
                optimizer="random",
                max_trials=3,
                enable_judge=False,
            )
        )

    assert get_backend_calls == 0


def test_newton_static_friction_override_rejected_before_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Static friction is explicit in the override, so reject it before the LLM."""

    def _unexpected_infer(*args: Any, **kwargs: Any) -> Scenario:
        raise AssertionError("interpreter should not be called")

    monkeypatch.setattr(
        "physics_agent.tasks.interpret_user_prompt_tuning.infer_scenario_from_prompt",
        _unexpected_infer,
    )
    import physics_agent.tuning.runner as runner_mod

    get_backend_calls = 0

    def _unexpected_get_backend(engine: str) -> Any:
        nonlocal get_backend_calls
        get_backend_calls += 1
        raise AssertionError("backend should not be constructed")

    monkeypatch.setattr(runner_mod, "get_backend", _unexpected_get_backend)

    with pytest.raises(TuningError, match="static_friction-only trials"):
        run_tune(
            TuneInput(
                user_prompt="make it stick aggressively",
                scenario={
                    "parameters": [
                        {"name": "static_friction", "min": 0.05, "max": 1.0},
                    ],
                },
                physics_usd=_physics_usd(tmp_path),
                output_dir=tmp_path / "out",
                engine="newton",
                optimizer="random",
                max_trials=3,
                enable_judge=False,
            )
        )

    assert get_backend_calls == 0


def test_user_prompt_yaml_override_anchors_relative_paths_to_yaml_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 13 (CodeRabbit thread #3): paths inside the override YAML
    must resolve relative to the YAML file's directory, not the runner's
    CWD. Otherwise programmatic / NL-driven scenarios that reference
    assets via relative paths inside the YAML would break depending on
    where the worker was launched."""
    captured: dict[str, Any] = {}

    def _fake_infer(user_prompt: str, **kwargs: Any) -> Scenario:
        captured["scenario_override"] = kwargs.get("scenario_override")
        return parse_scenario(_scenario_dict())

    monkeypatch.setattr(
        "physics_agent.tasks.interpret_user_prompt_tuning.infer_scenario_from_prompt",
        _fake_infer,
    )
    _patch_judge_chat_model_to_none(monkeypatch)

    # Asset file referenced by a relative path in the scenario YAML.
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    asset_path = asset_dir / "surface.usda"
    asset_path.write_text("#usda 1.0\n", encoding="utf-8")

    scenario_dir = tmp_path / "configs"
    scenario_dir.mkdir()
    yaml_path = scenario_dir / "scenario.yaml"
    # ``surface_usd`` ends in ``_usd``; the anchor pass must rewrite it.
    # ``description`` does NOT end in a recognised path suffix, so a
    # ``./not-a-path`` value here must pass through unchanged.
    yaml_path.write_text(
        "name: drop_settle\n"
        "metric: settle_distance\n"
        "target:\n"
        "  drop_height_m: 0.7\n"
        "  surface_usd: ../assets/surface.usda\n"
        "  description: ./not-a-path\n"
        "parameters:\n  - name: mass_scale\n    min: 0.6\n    max: 1.4\n",
        encoding="utf-8",
    )

    run_tune(
        TuneInput(
            scenario=yaml_path,
            user_prompt="make it bouncy",
            physics_usd=_physics_usd(tmp_path),
            output_dir=tmp_path / "out",
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=False,
        )
    )
    override = captured["scenario_override"]
    assert override is not None
    # Path-shaped key was anchored against the YAML's directory.
    resolved_surface = override["target"]["surface_usd"]
    assert Path(resolved_surface).is_absolute(), resolved_surface
    assert Path(resolved_surface) == asset_path.resolve(), resolved_surface
    # Non-path-shaped key with a path-like value is left alone.
    assert override["target"]["description"] == "./not-a-path"


# ---------------------------------------------------------------------------
# Byte-identical baseline (the hard spec guarantee)
# ---------------------------------------------------------------------------


def test_no_judge_byte_identical_baseline_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``enable_judge=False`` + ``user_prompt=None`` produces no Part-1.1
    fields anywhere in the artifact set. This is the spec's hard
    backwards-compat guarantee from closed issue #51.
    """
    out = tmp_path / "out"
    run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=False,
        )
    )

    tr = json.loads((out / ARTIFACT_RESULTS).read_text())
    assert "user_prompt" not in tr, (
        "tune_results.json must NOT contain user_prompt when "
        "enable_judge=False AND user_prompt is unset"
    )
    assert "judge" not in tr, (
        "tune_results.json must NOT contain judge when enable_judge=False"
    )

    report = (out / ARTIFACT_REPORT).read_text()
    assert "## User prompt" not in report
    assert "## Judge verdict" not in report

    # No interpreter or judge side-effect dirs/files.
    assert not (out / "inferred_scenario.json").exists()
    assert not (out / "judge_cache").exists()


# ---------------------------------------------------------------------------
# Judge wiring
# ---------------------------------------------------------------------------


def test_judge_runs_by_default_and_persists_to_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default ``enable_judge=True`` causes the judge to run and adds the
    judge fields to tune_results.json + a Judge verdict section to report.md.
    The judge degrades to programmatic-only when the VLM resolves to None.
    """
    _patch_judge_chat_model_to_none(monkeypatch)

    out = tmp_path / "out"
    run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            # enable_judge=True is the default — explicit here for clarity.
            enable_judge=True,
        )
    )

    tr = json.loads((out / ARTIFACT_RESULTS).read_text())
    assert "judge" in tr, "judge field missing from tune_results.json"
    judge = tr["judge"]
    # Codex round 9: when the LLM is unavailable the runner records
    # status='degraded' rather than 'completed' so consumers don't
    # trust a programmatic-only "approve" as a real verdict.
    assert judge["enabled"] is True
    assert judge["status"] == "degraded"
    assert judge["decision"] in {"approve", "continue"}
    assert 0.0 <= judge["score"] <= 1.0
    assert judge.get("llm_unavailable") is True
    assert judge["llm_score"] == pytest.approx(judge["programmatic_score"])

    report = (out / ARTIFACT_REPORT).read_text()
    assert "## Judge verdict" in report
    assert "Decision: `" in report

    # No judge_cache directory — caching was removed; rerun-determinism
    # comes from the daemon's seed contract, not a persisted LLM cache.
    assert not (out / "judge_cache").exists()


def test_tune_reference_media_fail_closed_when_judge_vlm_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tuning.visual_evidence import JudgeVisualEvidence

    _patch_judge_chat_model_to_none(monkeypatch)
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"fake image bytes")

    def _fake_prepare_visual_evidence(**_kwargs: Any) -> JudgeVisualEvidence:
        return JudgeVisualEvidence(
            reference_image_caption_pairs=(("synthetic", reference),)
        )

    monkeypatch.setattr(
        "physics_agent.tuning.runner._prepare_visual_evidence_for_judge",
        _fake_prepare_visual_evidence,
    )

    out = tmp_path / "out"
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=True,
            reference_images=[reference],
        )
    )

    assert result.success is False
    assert result.error == (
        "Judge VLM unavailable with reference media; refusing to fall back "
        "to programmatic-only verdict."
    )
    tr = json.loads((out / ARTIFACT_RESULTS).read_text())
    assert tr["judge"]["status"] == "degraded"
    assert tr["judge"]["llm_unavailable"] is True


def test_tune_visual_comparison_in_artifact_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tasks.judge_tune import JudgeResult
    from physics_agent.tuning.artifacts import ARTIFACT_VISUAL_COMPARISON
    from physics_agent.tuning.visual_evidence import JudgeVisualEvidence

    def _fake_prepare_visual_evidence(**kwargs: Any) -> JudgeVisualEvidence:
        comparison = Path(kwargs["output_dir"]) / ARTIFACT_VISUAL_COMPARISON
        comparison.write_bytes(b"\x89PNG\r\n\x1a\nfake\n")
        return JudgeVisualEvidence(comparison_image_path=comparison)

    def _fake_judge(*args: Any, **kwargs: Any) -> JudgeResult:
        return JudgeResult(
            decision="approve",
            score=1.0,
            programmatic_score=1.0,
            llm_score=1.0,
            reasoning="synthetic approve",
            iterations=1,
            llm_unavailable=False,
        )

    monkeypatch.setattr(
        "physics_agent.tuning.runner._prepare_visual_evidence_for_judge",
        _fake_prepare_visual_evidence,
    )
    monkeypatch.setattr(
        "physics_agent.tasks.judge_tune.run_tune_judge",
        _fake_judge,
    )
    _patch_judge_chat_model_to_none(monkeypatch)
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"fake image bytes")

    out = tmp_path / "out"
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=True,
            reference_images=[reference],
        )
    )

    assert result.success is True
    assert result.artifacts[ARTIFACT_VISUAL_COMPARISON] == (
        out / ARTIFACT_VISUAL_COMPARISON
    )


def test_freeform_tune_prepares_generated_visual_evidence_without_reference_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tasks.judge_tune import JudgeResult
    from physics_agent.tuning.visual_evidence import JudgeVisualEvidence

    frame = tmp_path / "generated.png"
    frame.write_bytes(b"fake generated image bytes")
    prep_calls: list[dict[str, Any]] = []
    judge_visual_evidence: list[JudgeVisualEvidence | None] = []

    def _fake_prepare_visual_evidence(**kwargs: Any) -> JudgeVisualEvidence:
        prep_calls.append(kwargs)
        assert kwargs["include_generated_without_reference"] is True
        return JudgeVisualEvidence(generated_image_paths=(frame,))

    def _fake_judge(*args: Any, **kwargs: Any) -> JudgeResult:
        judge_visual_evidence.append(kwargs["visual_evidence"])
        return JudgeResult(
            decision="approve",
            score=1.0,
            programmatic_score=1.0,
            llm_score=1.0,
            reasoning="generated frames looked realistic",
            iterations=1,
            llm_unavailable=False,
        )

    monkeypatch.setattr(
        "physics_agent.tuning.runner._prepare_visual_evidence_for_judge",
        _fake_prepare_visual_evidence,
    )
    monkeypatch.setattr(
        "physics_agent.tasks.judge_tune.run_tune_judge",
        _fake_judge,
    )
    _patch_judge_chat_model_to_none(monkeypatch)

    out = tmp_path / "out"
    result = run_tune(
        TuneInput(
            scenario=_freeform_scenario_dict(),
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=True,
        )
    )

    assert result.success is True
    assert len(prep_calls) == 1
    assert judge_visual_evidence[0] is not None
    assert judge_visual_evidence[0].generated_image_paths == (frame,)


def test_user_prompt_persisted_in_tune_results_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When user_prompt is set, it appears verbatim in tune_results.json and
    in a fenced block inside report.md."""

    def _fake_infer(user_prompt: str, **kwargs: Any) -> Scenario:
        return parse_scenario(_scenario_dict())

    monkeypatch.setattr(
        "physics_agent.tasks.interpret_user_prompt_tuning.infer_scenario_from_prompt",
        _fake_infer,
    )
    _patch_judge_chat_model_to_none(monkeypatch)

    prompt = "make this object bouncy"
    out = tmp_path / "out"
    run_tune(
        TuneInput(
            user_prompt=prompt,
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=False,
        )
    )

    tr = json.loads((out / ARTIFACT_RESULTS).read_text())
    assert tr["user_prompt"] == prompt

    report = (out / ARTIFACT_REPORT).read_text()
    assert "## User prompt" in report
    assert prompt in report


def test_report_visual_evidence_captions_are_markdown_safe(tmp_path: Path) -> None:
    scenario = parse_scenario(_scenario_dict())
    malicious_caption = (
        "Reference Image 1: target\n## Forged verdict\n`code` [link](javascript:bad)"
    )
    history = [
        TrialRecord(
            trial_index=0,
            params={"mass_scale": 1.0},
            score=0.0,
            backend_metrics={"settle_distance": 0.0},
            duration_seconds=0.0,
            failed=False,
        )
    ]

    write_report_md(
        tmp_path,
        scenario=scenario,
        optimizer_used="random",
        engine_used="fake",
        best_params={"mass_scale": 1.0},
        best_score=0.0,
        history=history,
        cancelled=False,
        judge_result={
            "decision": "approve",
            "score": 1.0,
            "iterations": 1,
            "reasoning": "ok",
            "extra": {
                "visual_evidence": {
                    "reference_images": [
                        {"caption": malicious_caption, "path": "ref.png"}
                    ],
                    "generated_images": [],
                    "comparison_image": None,
                    "reference_error": None,
                    "generated_error": None,
                    "comparison_error": None,
                }
            },
        },
    )

    report = (tmp_path / ARTIFACT_REPORT).read_text(encoding="utf-8")
    assert "\n## Forged verdict" not in report
    evidence_line = next(
        line for line in report.splitlines() if "Forged verdict" in line
    )
    assert evidence_line.startswith("  - `ref.png` - `` ")
    assert evidence_line.endswith(" ``")


def test_judge_failure_is_non_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A JudgeError must not abort the tune — artifacts are still written
    without a judge section."""
    from physics_agent.tasks.judge_tune import JudgeError

    sentinel = "physics-judge-api-key-sentinel"

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise JudgeError(f"provider rejected api_key={sentinel}")

    monkeypatch.setattr(
        "physics_agent.tasks.judge_tune.run_tune_judge",
        _explode,
    )
    _patch_judge_chat_model_to_none(monkeypatch)

    events: list[tuple[str, dict[str, Any]]] = []

    class _Listener:
        def event(self, event_type: str, data: dict[str, Any]) -> None:
            events.append((event_type, data))

    out = tmp_path / "out"
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=True,
            event_listener=_Listener(),
        )
    )
    assert result.success
    tr = json.loads((out / ARTIFACT_RESULTS).read_text())
    # Codex round 3: durable failure status — when enable_judge=True we
    # always persist the attempt outcome so consumers can distinguish
    # "judge disabled" (no key) from "judge attempted but failed".
    assert "judge" in tr
    assert tr["judge"]["enabled"] is True
    assert tr["judge"]["status"] == "failed"
    assert tr["judge"]["error_type"] == "JudgeError"
    assert "decision" not in tr["judge"]  # no verdict on failure
    report = (out / ARTIFACT_REPORT).read_text()
    # The "## Judge verdict" section in report.md is gated on a verdict
    # being present. Failed-but-attempted states intentionally do NOT
    # render a Judge-verdict section to avoid showing a partial verdict.
    assert "## Judge verdict" not in report
    failed_events = [
        data for event_type, data in events if event_type == "tune.judge.failed"
    ]
    assert failed_events == [
        {"error": "Judge execution failed.", "error_type": "JudgeError"}
    ]
    published = json.dumps(tr) + report + repr(events) + caplog.text
    assert sentinel not in published


# ---------------------------------------------------------------------------
# Refine-loop forward-compat plumbing
# ---------------------------------------------------------------------------


def test_judge_refine_skipped_event_when_max_iter_gt_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the judge returns ``continue`` and judge_max_iterations > 1,
    the runner emits ``tune.judge.refine_skipped`` documenting that v1.1
    runs single-iteration judging only. judge_max_iterations is plumbing
    for the multi-iteration loop in a follow-up.
    """
    from physics_agent.tasks.judge_tune import JudgeResult

    judge_kwargs: list[dict[str, Any]] = []

    def _fake_judge(*args: Any, **kwargs: Any) -> JudgeResult:
        judge_kwargs.append(kwargs)
        return JudgeResult(
            decision="continue",
            score=0.4,
            programmatic_score=0.4,
            llm_score=0.4,
            reasoning="synthetic continue",
            iterations=kwargs.get("iteration", 1),
            llm_unavailable=True,
        )

    # Replace the imported binding so the runner's lazy import picks it up.
    monkeypatch.setattr(
        "physics_agent.tasks.judge_tune.run_tune_judge",
        _fake_judge,
    )
    _patch_judge_chat_model_to_none(monkeypatch)

    events: list[tuple[str, dict[str, Any]]] = []

    class _Listener:
        def event(self, event_type: str, data: dict[str, Any]) -> None:
            events.append((event_type, data))

    out = tmp_path / "out"
    run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=True,
            judge_max_iterations=3,
            judge_max_tokens=777,
            judge_temperature=0.25,
            event_listener=_Listener(),
        )
    )

    assert judge_kwargs[0]["judge_max_tokens"] == 777
    assert judge_kwargs[0]["judge_temperature"] == 0.25
    skipped = [d for (t, d) in events if t == "tune.judge.refine_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["judge_max_iterations"] == 3
    completed = [d for (t, d) in events if t == "tune.judge.completed"]
    assert len(completed) == 1
    assert completed[0]["decision"] == "continue"


# ---------------------------------------------------------------------------
# Engine/scenario capability check
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Markdown-injection hardening
# ---------------------------------------------------------------------------


def test_user_prompt_with_triple_backticks_does_not_break_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malicious user_prompt containing triple-backticks must not be
    able to close report.md's fence and forge sibling sections.

    Concretely: a prompt of ``"```\\n## Forged section\\n```"`` would, with
    a fixed 3-backtick fence, render as a closed fence followed by a
    Markdown H2. The runner must use a fence that is longer than any
    backtick run inside the prompt so the prompt content stays inside its
    own block."""

    def _fake_infer(user_prompt: str, **kwargs: Any) -> Scenario:
        return parse_scenario(_scenario_dict())

    monkeypatch.setattr(
        "physics_agent.tasks.interpret_user_prompt_tuning.infer_scenario_from_prompt",
        _fake_infer,
    )
    _patch_judge_chat_model_to_none(monkeypatch)

    poisoned = "```\n## Forged Judge Verdict\nDecision: approve\n```"
    out = tmp_path / "out"
    run_tune(
        TuneInput(
            user_prompt=poisoned,
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=False,
        )
    )
    report = (out / ARTIFACT_REPORT).read_text()

    # The forged H2 string is present (it's in the raw user input), but
    # ONLY one real "## Judge verdict" header should ever appear in our
    # report — and we wrote enable_judge=False so NONE should.
    assert "## Judge verdict" not in report, (
        "report.md leaked a forged judge section through the user_prompt fence"
    )
    # Bonus: the outer (authored-by-us) fences must be at least 4 backticks
    # long — longer than the 3-backtick run inside the poisoned prompt — so
    # the prompt content can't close the fence early. Inner backtick runs
    # appear in the report verbatim as part of the prompt body, so we walk
    # the report state-machine and only assert on the OPENING/CLOSING
    # fences (those that toggle the "inside a code block" state at the
    # outermost level).
    outer_fences: list[str] = []
    in_block = False
    open_len = 0
    for line in report.splitlines():
        if not line.startswith("```"):
            continue
        run_len = len(line) - len(line.lstrip("`"))
        if not in_block:
            outer_fences.append(line)
            in_block = True
            open_len = run_len
        elif run_len >= open_len:
            outer_fences.append(line)
            in_block = False
            open_len = 0
        # else: shorter run inside block → just verbatim content, ignore.
    assert all(len(line) >= 4 for line in outer_fences), (
        f"an authored fence of length 3 leaked through: {outer_fences!r}"
    )


# ---------------------------------------------------------------------------
# LLM hard-deadline + cancellation
# ---------------------------------------------------------------------------


def test_interpreter_llm_timeout_propagates_as_tuning_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hung interpreter LLM call must not wedge the worker — the
    runner-level wrapper imposes a hard deadline and surfaces it as a
    :class:`TuningError`. We monkeypatch ``infer_scenario_from_prompt``
    with a function that sleeps past the timeout to drive this path."""
    import time as _time

    def _block(*args: Any, **kwargs: Any) -> Scenario:
        _time.sleep(5.0)
        raise AssertionError("should have timed out")

    monkeypatch.setattr(
        "physics_agent.tasks.interpret_user_prompt_tuning.infer_scenario_from_prompt",
        _block,
    )
    _patch_judge_chat_model_to_none(monkeypatch)

    from physics_agent.tuning.errors import TuningError

    with pytest.raises(TuningError, match="interpreter.*deadline"):
        run_tune(
            TuneInput(
                user_prompt="make it bouncy",
                physics_usd=_physics_usd(tmp_path),
                output_dir=tmp_path / "out",
                engine="fake",
                optimizer="random",
                max_trials=2,
                enable_judge=False,
                llm_timeout_seconds=0.5,
            )
        )


def test_judge_llm_timeout_is_non_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hung judge VLM call is non-fatal: tune artifacts are written
    with a persisted failed judge section, and a tune.judge.failed event
    is emitted with error_type=_LLMTimeoutError."""
    import time as _time

    def _block(*args: Any, **kwargs: Any) -> Any:
        _time.sleep(5.0)
        raise AssertionError("should have timed out")

    monkeypatch.setattr(
        "physics_agent.tasks.judge_tune.run_tune_judge",
        _block,
    )
    _patch_judge_chat_model_to_none(monkeypatch)

    events: list[tuple[str, dict[str, Any]]] = []

    class _Listener:
        def event(self, event_type: str, data: dict[str, Any]) -> None:
            events.append((event_type, data))

    start = _time.monotonic()
    out = tmp_path / "out"
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=True,
            llm_timeout_seconds=0.5,
            event_listener=_Listener(),
        )
    )
    assert _time.monotonic() - start < 2.0
    assert result.success
    tr = json.loads((out / ARTIFACT_RESULTS).read_text())
    # Codex round 3: durable failure status — judge timeout writes a
    # status=failed block instead of leaving the key absent.
    assert "judge" in tr
    assert tr["judge"]["enabled"] is True
    assert tr["judge"]["status"] == "failed"
    assert tr["judge"]["error_type"] == "_LLMTimeoutError"
    assert "decision" not in tr["judge"]
    failed_events = [d for (t, d) in events if t == "tune.judge.failed"]
    assert len(failed_events) == 1
    assert failed_events[0]["error_type"] == "_LLMTimeoutError"
    assert failed_events[0]["error"] == "Judge execution failed."
    assert "deadline" not in caplog.text


def test_reference_media_judge_timeout_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When reference media is supplied, judge timeout must fail closed.

    Reference-media judging is VLM-dependent: a timeout means we cannot
    compare the candidate against the user-provided visual target, so the
    runner must not accept a programmatic-only fallback verdict.
    """
    import time as _time

    from physics_agent.tuning.visual_evidence import JudgeVisualEvidence

    def _block(*args: Any, **kwargs: Any) -> Any:
        _time.sleep(5.0)
        raise AssertionError("should have timed out")

    monkeypatch.setattr(
        "physics_agent.tasks.judge_tune.run_tune_judge",
        _block,
    )
    _patch_judge_chat_model_to_none(monkeypatch)
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"fake image bytes")

    def _fake_prepare_visual_evidence(**_kwargs: Any) -> JudgeVisualEvidence:
        return JudgeVisualEvidence(
            reference_image_caption_pairs=(("synthetic", reference),)
        )

    monkeypatch.setattr(
        "physics_agent.tuning.runner._prepare_visual_evidence_for_judge",
        _fake_prepare_visual_evidence,
    )

    start = _time.monotonic()
    out = tmp_path / "out"
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=True,
            reference_images=[reference],
            llm_timeout_seconds=0.5,
        )
    )

    assert _time.monotonic() - start < 2.0
    assert result.success is False
    assert result.error == (
        "Judge VLM unavailable with reference media; refusing to fall back "
        "to programmatic-only verdict."
    )
    tr = json.loads((out / ARTIFACT_RESULTS).read_text())
    assert tr["judge"]["enabled"] is True
    assert tr["judge"]["status"] == "failed"
    assert tr["judge"]["error_type"] == "_LLMTimeoutError"


def test_reference_media_visual_evidence_timeout_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung render/prep step must not bypass the judge deadline."""
    import time as _time

    def _block_render(*args: Any, **kwargs: Any) -> tuple[list[Path], str | None]:
        _time.sleep(5.0)
        return [], None

    def _should_not_call_judge(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("judge should not run after visual prep timeout")

    monkeypatch.setattr(
        "physics_agent.tuning.runner._render_best_trial_for_visual_judge",
        _block_render,
    )
    monkeypatch.setattr(
        "physics_agent.tasks.judge_tune.run_tune_judge",
        _should_not_call_judge,
    )
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"fake image bytes")

    start = _time.monotonic()
    out = tmp_path / "out"
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=True,
            reference_images=[reference],
            llm_timeout_seconds=0.2,
        )
    )

    assert _time.monotonic() - start < 2.0
    assert result.success is False
    assert result.error == (
        "Judge VLM unavailable with reference media; refusing to fall back "
        "to programmatic-only verdict."
    )
    tr = json.loads((out / ARTIFACT_RESULTS).read_text())
    assert tr["judge"]["enabled"] is True
    assert tr["judge"]["status"] == "failed"
    assert tr["judge"]["error_type"] == "_LLMTimeoutError"


@pytest.mark.parametrize("error_field", ["reference_error", "generated_error"])
def test_reference_media_visual_evidence_error_fails_closed_before_vlm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_field: str,
) -> None:
    from physics_agent.tuning.visual_evidence import JudgeVisualEvidence

    def _fake_prepare_visual_evidence(**_kwargs: Any) -> JudgeVisualEvidence:
        return JudgeVisualEvidence(**{error_field: "synthetic"})

    def _should_not_setup_vlm() -> Any:
        raise AssertionError("VLM setup should not run after evidence prep error")

    def _should_not_call_judge(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("judge should not run after evidence prep error")

    monkeypatch.setattr(
        "physics_agent.tuning.runner._prepare_visual_evidence_for_judge",
        _fake_prepare_visual_evidence,
    )
    monkeypatch.setattr(
        "physics_agent.tuning.runner._resolve_judge_vlm_lazy",
        _should_not_setup_vlm,
    )
    monkeypatch.setattr(
        "physics_agent.tasks.judge_tune.run_tune_judge",
        _should_not_call_judge,
    )
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"fake image bytes")

    out = tmp_path / "out"
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=True,
            reference_images=[reference],
        )
    )

    assert result.success is False
    assert result.error == (
        "Visual judge evidence preparation failed with reference media; "
        "refusing to fall back to programmatic-only verdict."
    )
    tr = json.loads((out / ARTIFACT_RESULTS).read_text())
    assert tr["judge"]["enabled"] is True
    assert tr["judge"]["status"] == "failed"
    assert tr["judge"]["error_type"] == "JudgeError"


def test_reference_media_all_failed_trials_reports_trial_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tuning import backend as backend_mod
    from physics_agent.tuning.visual_evidence import JudgeVisualEvidence

    class BrokenBackend:
        name = "fake"

        def evaluate(
            self,
            params: dict[str, float],
            scenario: Any,
            physics_usd: Path,
            *,
            seed: int,
        ) -> dict[str, Any]:
            return {"score": "not a number"}

    original_get_backend = backend_mod.get_backend

    def fake_get_backend(engine: str):
        if engine == "fake":
            return BrokenBackend()
        return original_get_backend(engine)

    def _fake_prepare_visual_evidence(**_kwargs: Any) -> JudgeVisualEvidence:
        return JudgeVisualEvidence(generated_error="no successful trial to render")

    monkeypatch.setattr("physics_agent.tuning.runner.get_backend", fake_get_backend)
    monkeypatch.setattr(
        "physics_agent.tuning.runner._prepare_visual_evidence_for_judge",
        _fake_prepare_visual_evidence,
    )
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"fake image bytes")

    out = tmp_path / "out"
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=True,
            reference_images=[reference],
        )
    )

    assert result.success is False
    assert result.error is not None
    assert "non-numeric score" in result.error
    assert "Visual judge evidence" not in result.error
    tr = json.loads((out / ARTIFACT_RESULTS).read_text())
    assert tr["judge"]["enabled"] is True
    assert tr["judge"]["status"] == "failed"


def test_reference_media_judge_vlm_setup_timeout_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default VLM construction is also bounded before visual judging."""
    import time as _time

    from physics_agent.tuning.visual_evidence import JudgeVisualEvidence

    def _fake_prepare_visual_evidence(**kwargs: Any) -> JudgeVisualEvidence:
        return JudgeVisualEvidence(
            reference_image_caption_pairs=(("synthetic", reference),)
        )

    def _block_vlm_setup() -> None:
        _time.sleep(5.0)

    def _should_not_call_judge(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("judge should not run after VLM setup timeout")

    monkeypatch.setattr(
        "physics_agent.tuning.runner._prepare_visual_evidence_for_judge",
        _fake_prepare_visual_evidence,
    )
    monkeypatch.setattr(
        "physics_agent.tuning.runner._resolve_judge_vlm_lazy",
        _block_vlm_setup,
    )
    monkeypatch.setattr(
        "physics_agent.tasks.judge_tune.run_tune_judge",
        _should_not_call_judge,
    )
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"fake image bytes")

    start = _time.monotonic()
    out = tmp_path / "out"
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=_physics_usd(tmp_path),
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            enable_judge=True,
            reference_images=[reference],
            llm_timeout_seconds=0.2,
        )
    )

    assert _time.monotonic() - start < 2.0
    assert result.success is False
    assert result.error == (
        "Judge VLM unavailable with reference media; refusing to fall back "
        "to programmatic-only verdict."
    )
    tr = json.loads((out / ARTIFACT_RESULTS).read_text())
    assert tr["judge"]["enabled"] is True
    assert tr["judge"]["status"] == "failed"
    assert tr["judge"]["error_type"] == "_LLMTimeoutError"


# ---------------------------------------------------------------------------
# inferred_scenario.json — write-only audit, no read-back
# ---------------------------------------------------------------------------


def test_interpreter_invoked_every_run_no_cache_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With caching removed, the runner invokes the interpreter on every
    tune call regardless of whether ``inferred_scenario.json`` already
    exists in ``output_dir``. The file is a write-only audit record,
    never a cache key — rerun-determinism is owned by the daemon's
    seed contract, not a persisted LLM-response cache.
    """
    call_count = {"n": 0}

    def _counting_infer(*args: Any, **kwargs: Any) -> Scenario:
        call_count["n"] += 1
        # Real interpreter writes the audit record on success; emulate
        # the minimal current shape (no schema_version / no override
        # canonicalization — those existed solely for cache validation).
        scenario_dict = _scenario_dict()
        audit_dir = kwargs.get("audit_dir")
        if audit_dir is not None:
            payload = {
                "_meta": {
                    "user_prompt": kwargs.get("user_prompt") or args[0],
                    "model": "stub-model",
                    "merged_from_explicit": bool(kwargs.get("scenario_override")),
                },
                **scenario_dict,
            }
            audit_path = Path(audit_dir) / "inferred_scenario.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(payload), encoding="utf-8")
        return parse_scenario(scenario_dict)

    monkeypatch.setattr(
        "physics_agent.tasks.interpret_user_prompt_tuning.infer_scenario_from_prompt",
        _counting_infer,
    )
    _patch_judge_chat_model_to_none(monkeypatch)

    out = tmp_path / "out"
    physics = _physics_usd(tmp_path)
    common = {
        "user_prompt": "make it bouncy",
        "physics_usd": physics,
        "output_dir": out,
        "engine": "fake",
        "optimizer": "random",
        "max_trials": 2,
        "enable_judge": False,
    }
    run_tune(TuneInput(**common))
    assert call_count["n"] == 1
    assert (out / "inferred_scenario.json").exists()

    # Second run with the same args: the interpreter MUST be invoked
    # again. Caching was removed; only the audit file lingers.
    result = run_tune(TuneInput(**common))
    assert result.success
    assert call_count["n"] == 2
