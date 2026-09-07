# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner -> prompt wiring for the tuning gate and collider pin.

The prompt builders are covered directly elsewhere, but a builder-only test
cannot see the defect that actually shipped: a value that the builder renders
correctly and the runner never passes. Deleting
``revalidation_max_penetration_m=config.revalidation_max_penetration_m`` or
``collision_approximation=config.collision_approximation`` from the tuning call
site reproduces the original miss -- the agent believes the limit is 0.005 m, or
silently degrades the collider -- with every builder-level test still green.

These tests drive ``_run_physics_agentic_tuning_phase`` with a stubbed broker and
capture the kwargs the runner actually hands the builder.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from content_agent_workflows.common.usd_cli import UsdCliPackageRoute
from content_agent_workflows.common.usd_cli_session import WorkflowUsdCliSession

from content_workflow_cli import runner
from content_workflow_cli.runner import PhysicsApplyConfig, TraceWriter


class _StopAfterPrompt(Exception):
    """Sentinel: the prompt kwargs are captured, nothing past this point matters."""


class _FakeBroker:
    """Stands in for PhysicsTuningBroker; the loop only needs a URL and stop."""

    url = "http://127.0.0.1:37597"

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def close(self) -> None:
        pass


def _fake_usd_cli_session(run_dir: Path) -> WorkflowUsdCliSession:
    """A typed session is enough where the test stubs all scene operations."""

    route = UsdCliPackageRoute(
        wrapper=Path("/bin/false"),
        target=Path("/bin/false"),
        source_root=run_dir,
        source_revision="test",
    )
    return WorkflowUsdCliSession(
        project_dir=run_dir,
        session_id="workflow-physics-test",
        route=route,
        workflow="physics",
    )


def _prepare_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    physics_usd = run_dir / "physics.usdc"
    physics_usd.write_bytes(b"usd")
    (run_dir / "raw" / "physics_finalize_result_1.json").write_text(
        json.dumps({"physics_usd_path": str(physics_usd)}),
        encoding="utf-8",
    )
    return run_dir


def _captured_prompt_kwargs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **config_overrides: Any,
) -> dict[str, Any]:
    run_dir = _prepare_run_dir(tmp_path)
    captured: dict[str, Any] = {}

    def fake_builder(**kwargs: Any) -> str:
        captured.update(kwargs)
        # The wiring is the whole subject of this test; abort the run here so we
        # do not need a live Workbench, broker ledger, or child agent.
        raise _StopAfterPrompt

    from content_workflow_cli import tuning_broker

    monkeypatch.setattr(tuning_broker, "PhysicsTuningBroker", _FakeBroker)
    monkeypatch.setattr(runner, "build_physics_tuning_session_prompt", fake_builder)
    if config_overrides.get("vomp_mass") is not None:
        monkeypatch.setattr(
            runner,
            "_capture_physics_vomp_tuning_baseline",
            lambda **_kwargs: SimpleNamespace(
                canonical_result_path=run_dir / "raw" / "physics_vomp_result.json"
            ),
        )

    config = PhysicsApplyConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        **config_overrides,
    )

    with pytest.raises(_StopAfterPrompt):
        runner._run_physics_agentic_tuning_phase(
            config=config,
            run_dir=run_dir,
            session_id="session-1",
            trace_writer=TraceWriter(run_dir),
        )
    return captured


def test_runner_passes_the_penetration_override_to_the_tuning_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original defect: the flag reached the wrapper's revalidation but not
    the agent's own prompt, so the agent judged against the unconfigured
    default and stopped."""

    captured = _captured_prompt_kwargs(
        tmp_path,
        monkeypatch,
        revalidation_max_penetration_m=0.05,
    )
    assert captured["revalidation_max_penetration_m"] == 0.05


def test_runner_passes_the_collider_pin_to_the_tuning_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`revise_patch` may rewrite the collision approximation; the pin only works
    if the authored value actually reaches the prompt."""

    captured = _captured_prompt_kwargs(
        tmp_path,
        monkeypatch,
        collision_approximation="convexDecomposition",
    )

    assert captured["collision_approximation"] == "convexDecomposition"


def test_runner_protects_vomp_mass_properties_in_the_tuning_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.physics import PhysicsVompMassConfig

    captured = _captured_prompt_kwargs(
        tmp_path,
        monkeypatch,
        vomp_mass=PhysicsVompMassConfig(runtime_root=tmp_path / "VoMP"),
    )

    assert captured["protected_parameters"] == ["mass_scale"]
    assert captured["allow_revise_patch"] is False


def test_apply_phase_gate_honours_the_penetration_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physics_packet_writer: Any,
) -> None:
    """The refinement child is told not to treat a deeper-than-default rest as a
    hard failure when it is inside the configured limit. If the override does not
    also reach the apply-phase runtime gate, the wrapper records an unresolved
    failure for exactly the rest depth the child was told to accept, and the run
    exits clean with a hard failure in its canonical evidence."""

    from content_agent_workflows.physics import PhysicsApplyWorkflowInput

    run_dir = _prepare_run_dir(tmp_path)
    captured: dict[str, Any] = {}

    def fake_apply(params: PhysicsApplyWorkflowInput) -> Any:
        captured["max_ground_penetration_m"] = params.max_ground_penetration_m
        raise _StopAfterPrompt

    monkeypatch.setattr(
        "content_agent_workflows.physics.run_physics_apply_workflow",
        fake_apply,
    )

    config = PhysicsApplyConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        revalidation_max_penetration_m=0.05,
    )
    config.usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    packet = physics_packet_writer(run_dir, config.usd_path)
    inspection_pin = runner._physics_inspection_pin_from_packet(config, packet)

    with pytest.raises(_StopAfterPrompt):
        runner._finalize_physics_once(
            config=config,
            run_dir=run_dir,
            inspection_pin=inspection_pin,
            session_id="session-1",
            iteration=1,
            trace_writer=TraceWriter(run_dir),
            usd_cli_session=_fake_usd_cli_session(run_dir),
            run_state={},
        )

    assert captured["max_ground_penetration_m"] == 0.05


def test_runner_mirrors_the_revalidation_drop_setup_into_the_tuning_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent's evidence must measure the wrapper's experiment, so the whole
    drop-settle setup -- not just the threshold -- has to reach the prompt."""

    captured = _captured_prompt_kwargs(
        tmp_path,
        monkeypatch,
        simulation_duration_s=2.0,
        simulation_dt=1.0 / 120.0,
        simulation_sample_fps=60,
        drop_height_m=0.9,
    )

    assert captured["revalidation_duration_s"] == 2.0
    assert captured["revalidation_dt"] == 1.0 / 120.0
    assert captured["revalidation_sample_fps"] == 60
    assert captured["revalidation_drop_height_m"] == 0.9


def test_tuning_prompt_pin_prefers_the_patch_authored_collider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The patch's per-decision collision approximation wins at authoring time
    (apply_physics prefers the record's classification), so pinning the config
    default would order a compliant child to rewrite a correctly authored finer
    shape back to the default -- collider degradation by instruction."""

    run_dir = tmp_path / "run"

    def write_patch() -> None:
        (run_dir / "raw" / "physics_decision_patch.json").write_text(
            json.dumps(
                {
                    "decisions": [
                        {"collision_approximation": "convexDecomposition"},
                    ]
                }
            ),
            encoding="utf-8",
        )

    captured: dict[str, Any] = {}

    def fake_builder(**kwargs: Any) -> str:
        captured.update(kwargs)
        raise _StopAfterPrompt

    from content_workflow_cli import tuning_broker

    monkeypatch.setattr(tuning_broker, "PhysicsTuningBroker", _FakeBroker)
    monkeypatch.setattr(runner, "build_physics_tuning_session_prompt", fake_builder)

    _prepare_run_dir(tmp_path)
    write_patch()
    config = PhysicsApplyConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        collision_approximation="convexHull",
    )
    with pytest.raises(_StopAfterPrompt):
        runner._run_physics_agentic_tuning_phase(
            config=config,
            run_dir=run_dir,
            session_id="session-1",
            trace_writer=TraceWriter(run_dir),
        )

    assert captured["collision_approximation"] == "convexDecomposition"


def test_authored_collider_helper_reports_mixed_patch_values(tmp_path: Path) -> None:
    """Multi-component patches may legitimately mix shapes; the pin must state
    all of them rather than pretending one value covers the asset."""

    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "physics_decision_patch.json").write_text(
        json.dumps(
            {
                "decisions": [
                    {"collision_approximation": "convexDecomposition"},
                    {"collision_approximation": "convexHull"},
                ]
            }
        ),
        encoding="utf-8",
    )
    config = PhysicsApplyConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        collision_approximation="convexHull",
    )

    pinned = runner._authored_collision_approximation(tmp_path, config)

    assert pinned == "convexDecomposition, convexHull"


def test_authored_collider_helper_falls_back_to_the_config_value(
    tmp_path: Path,
) -> None:
    """With no patch on disk the apply phase authors with the config value, so
    that is the correct pin."""

    config = PhysicsApplyConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        collision_approximation="convexHull",
    )

    assert runner._authored_collision_approximation(tmp_path, config) == "convexHull"


def test_validate_physics_config_rejects_an_out_of_bound_penetration_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Workbench acceptance schema caps the field at 1.0 m. A unit slip like
    `--revalidation-max-penetration-m 50` used to reach only the in-process
    tuning revalidation; it now also travels to the remote apply-phase gate,
    where it would fail as an opaque 422 mid-finalization. Reject it up front."""

    usd = tmp_path / "asset.usda"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setattr(runner, "usd_cli_source_distributed", lambda _root: True)

    with pytest.raises(ValueError, match="at most 1.0"):
        runner._validate_physics_config(
            PhysicsApplyConfig(
                repo_root=tmp_path,
                usd_path=usd,
                output_dir=tmp_path / "physics-run",
                simulation_engine="fake",
                revalidation_max_penetration_m=50.0,
            )
        )


def test_validate_physics_config_rejects_mass_scale_for_vomp_tuning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from content_agent_workflows.physics import PhysicsVompMassConfig

    usd = tmp_path / "asset.usda"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    runtime_root = tmp_path / "VoMP"
    runtime_root.mkdir()
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        """name: drop_settle
metric: settle_distance
parameters:
  - name: mass_scale
    min: 0.5
    max: 2.0
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "usd_cli_source_distributed", lambda _root: True)

    with pytest.raises(ValueError, match="remove mass_scale"):
        runner._validate_physics_config(
            PhysicsApplyConfig(
                repo_root=tmp_path,
                usd_path=usd,
                simulation_engine="fake",
                tune=True,
                behavior_prompt="settle",
                scenario_path=scenario,
                vomp_mass=PhysicsVompMassConfig(runtime_root=runtime_root),
            )
        )


def test_apply_workflow_input_rejects_an_out_of_bound_penetration_limit() -> None:
    """PhysicsApplyWorkflowInput mirrors the Workbench bound so direct callers
    fail at construction instead of as a remote 422."""

    from content_agent_workflows.physics import PhysicsApplyWorkflowInput

    with pytest.raises(ValueError, match="less_than_equal|1.0|1\\.0"):
        PhysicsApplyWorkflowInput(
            usd_path=Path("asset.usda"),
            output_dir=Path("out"),
            max_ground_penetration_m=50.0,
        )


def test_authored_collider_helper_does_not_pin_a_zero_decision_patch(
    tmp_path: Path,
) -> None:
    """An empty decisions list takes the workflow's preserve branch: nothing is
    authored, so pinning the config default would order refinement to overwrite
    an existing collider that legitimately uses another shape."""

    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "physics_decision_patch.json").write_text(
        json.dumps({"decisions": []}),
        encoding="utf-8",
    )
    config = PhysicsApplyConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        collision_approximation="convexHull",
    )

    assert runner._authored_collision_approximation(tmp_path, config) is None


def test_refinement_pin_is_the_iteration_one_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physics_packet_writer: Any,
) -> None:
    """Recomputing the pin from the patch each iteration launders degradation:
    a child that coarsens the collider on iteration 1 rewrites the patch, and
    iteration 2 would then present the degraded shape as the authored one. The
    pin must stay the iteration-1 baseline for the whole loop."""

    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    patch_path = run_dir / "raw" / "physics_decision_patch.json"
    patch_path.write_text(
        json.dumps({"decisions": [{"collision_approximation": "convexHull"}]}),
        encoding="utf-8",
    )
    source_usd = tmp_path / "asset.usda"
    inspection_usd = run_dir / "raw" / "physics_inspection_scene.usda"
    source_usd.write_text("#usda 1.0\n", encoding="utf-8")
    inspection_usd.write_text(
        '#usda 1.0\ndef Xform "World" {}\n',
        encoding="utf-8",
    )

    pins: dict[int, Any] = {}
    prompt_assets: dict[int, Path] = {}
    previous_assessments: dict[int, Path | None] = {}

    def fake_finalize(**kwargs: Any) -> dict[str, Any]:
        from content_agent_workflows.common import physics_validation_evidence

        payload = physics_validation_evidence(
            asset="asset.usda",
            target_runtime="fake",
            physics_properties_status="pass",
            runtime_loadability_status="not_evaluated",
            no_explosions_status="not_evaluated",
        ).model_dump(mode="json")
        evidence = run_dir / "raw" / "validation_evidence.json"
        evidence.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "validation_evidence_path": str(evidence),
            "simulation_report_path": None,
            "rendered_frames": [],
        }

    def fake_prompt(**kwargs: Any) -> str:
        pins[kwargs["iteration"]] = kwargs["collision_approximation"]
        prompt_assets[kwargs["iteration"]] = kwargs["usd_path"]
        previous_assessments[kwargs["iteration"]] = kwargs["previous_assessment_path"]
        return "prompt"

    def fake_child(**kwargs: Any) -> int:
        Path(kwargs["child_output_path"]).write_text("ok\n", encoding="utf-8")
        Path(kwargs["child_final_path"]).write_text("done\n", encoding="utf-8")
        if "physics_visual_review_1_" in str(kwargs["child_output_path"]):
            # The degenerate optimum: the first child coarsens the collider.
            patch_path.write_text(
                json.dumps(
                    {"decisions": [{"collision_approximation": "boundingSphere"}]}
                ),
                encoding="utf-8",
            )
        return 0

    monkeypatch.setattr(runner, "_finalize_physics_once", fake_finalize)
    monkeypatch.setattr(
        runner,
        "_write_physics_visual_issue_packet",
        lambda **_kwargs: run_dir / "raw" / "issue_packet.json",
    )
    monkeypatch.setattr(runner, "build_physics_visual_refinement_prompt", fake_prompt)
    monkeypatch.setattr(runner, "_run_child_agent", fake_child)
    config = PhysicsApplyConfig(
        repo_root=tmp_path,
        usd_path=source_usd,
        collision_approximation="convexHull",
        vqa_refinement_max_iterations=2,
    )
    staged_source = runner._stage_usd_cli_input_tree(
        label="source",
        source_usd_path=source_usd,
        run_dir=run_dir,
    )
    packet = physics_packet_writer(
        run_dir,
        source_usd,
        inspection_usd=inspection_usd,
    )
    inspection_pin = runner._physics_inspection_pin_from_packet(config, packet)

    runner._run_physics_visual_refinement_loop(
        config=config,
        run_dir=run_dir,
        inspection_pin=inspection_pin,
        session_id="session-1",
        trace_writer=TraceWriter(run_dir),
        scene_service=None,
        usd_cli_session=_fake_usd_cli_session(run_dir),
        run_state={},
    )

    assert pins[1] == "convexHull"
    # Without the baseline capture this reads the mutated patch and reports
    # the degraded shape as authored.
    assert pins[2] == "convexHull"
    staged_usd = staged_source.staged_usd_path.resolve()
    assert prompt_assets == {1: staged_usd, 2: staged_usd}
    assert previous_assessments == {
        1: None,
        2: run_dir / "physics_behavior_assessment.json",
    }


def test_final_visual_patch_change_is_refinalized_before_tuning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physics_packet_writer: Any,
) -> None:
    from content_agent_workflows.common import physics_validation_evidence

    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    patch_path = run_dir / "raw" / "physics_decision_patch.json"
    patch_path.write_text(
        json.dumps({"decisions": [{"collision_approximation": "convexHull"}]}),
        encoding="utf-8",
    )
    source_usd = tmp_path / "asset.usda"
    source_usd.write_text("#usda 1.0\n", encoding="utf-8")
    finalize_iterations: list[int] = []

    def fake_finalize(**kwargs: Any) -> dict[str, Any]:
        iteration = kwargs["iteration"]
        finalize_iterations.append(iteration)
        evidence_path = run_dir / "raw" / f"validation_evidence_{iteration}.json"
        evidence = physics_validation_evidence(
            asset="asset.usda",
            target_runtime="fake",
            physics_properties_status="pass",
            runtime_loadability_status="not_evaluated",
            no_explosions_status="not_evaluated",
        )
        evidence_path.write_text(
            json.dumps(evidence.model_dump(mode="json")),
            encoding="utf-8",
        )
        physics_usd = run_dir / f"physics_{iteration}.usdc"
        physics_usd.write_text("#usda 1.0\n", encoding="utf-8")
        record = {
            "physics_usd_path": str(physics_usd),
            "validation_evidence_path": str(evidence_path),
            "simulation_report_path": None,
            "rendered_frames": [],
        }
        (run_dir / "raw" / f"physics_finalize_result_{iteration}.json").write_text(
            json.dumps(record),
            encoding="utf-8",
        )
        return record

    def fake_child(**kwargs: Any) -> int:
        Path(kwargs["child_output_path"]).write_text("ok\n", encoding="utf-8")
        Path(kwargs["child_final_path"]).write_text("done\n", encoding="utf-8")
        patch_path.write_text(
            json.dumps(
                {"decisions": [{"collision_approximation": "convexDecomposition"}]}
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(runner, "_finalize_physics_once", fake_finalize)
    monkeypatch.setattr(
        runner,
        "_write_physics_visual_issue_packet",
        lambda **_kwargs: run_dir / "raw" / "issue_packet.json",
    )
    monkeypatch.setattr(
        runner,
        "build_physics_visual_refinement_prompt",
        lambda **_kwargs: "prompt",
    )
    monkeypatch.setattr(runner, "_run_child_agent", fake_child)
    config = PhysicsApplyConfig(
        repo_root=tmp_path,
        usd_path=source_usd,
        collision_approximation="convexHull",
        vqa_refinement_max_iterations=1,
    )
    runner._stage_usd_cli_input_tree(
        label="source",
        source_usd_path=source_usd,
        run_dir=run_dir,
    )
    packet = physics_packet_writer(run_dir, source_usd)
    inspection_pin = runner._physics_inspection_pin_from_packet(config, packet)

    returncode, session_id = runner._run_physics_visual_refinement_loop(
        config=config,
        run_dir=run_dir,
        inspection_pin=inspection_pin,
        session_id="session-1",
        trace_writer=TraceWriter(run_dir),
        scene_service=None,
        usd_cli_session=_fake_usd_cli_session(run_dir),
        run_state={},
    )

    latest_record, latest_path = runner._latest_physics_finalize_record(run_dir)
    history = json.loads(
        (run_dir / "raw" / "physics_visual_validation_history.json").read_text(
            encoding="utf-8"
        )
    )
    assert returncode == 0
    assert session_id is None
    assert finalize_iterations == [1, 2]
    assert latest_path.name == "physics_finalize_result_2.json"
    assert latest_record["physics_usd_path"].endswith("physics_2.usdc")
    assert history["iterations"][-1]["final_refinalized"] is True


def test_visual_review_blocks_when_finalized_vomp_artifacts_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physics_packet_writer: Any,
) -> None:
    from types import SimpleNamespace

    from content_agent_workflows.common import physics_validation_evidence
    from content_agent_workflows.physics import PhysicsVompMassConfig

    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    patch_path = raw_dir / "physics_decision_patch.json"
    patch_path.write_text(
        json.dumps({"decisions": [{"collision_approximation": "convexHull"}]}),
        encoding="utf-8",
    )
    physics_usd = run_dir / "physics.usdc"
    source_usd = tmp_path / "asset.usda"
    source_usd.write_text("#usda 1.0\n", encoding="utf-8")

    def fake_finalize(**_kwargs: Any) -> dict[str, Any]:
        physics_usd.write_bytes(b"wrapper-attested")
        evidence_path = raw_dir / "validation_evidence.json"
        evidence = physics_validation_evidence(
            asset=str(physics_usd),
            target_runtime="fake",
            physics_properties_status="pass",
            runtime_loadability_status="not_evaluated",
            no_explosions_status="not_evaluated",
        )
        evidence_path.write_text(
            json.dumps(evidence.model_dump(mode="json")),
            encoding="utf-8",
        )
        return {
            "physics_usd_path": str(physics_usd),
            "vomp_result_path": str(raw_dir / "vomp" / "physics_vomp_result.json"),
            "validation_evidence_path": str(evidence_path),
            "simulation_report_path": None,
            "rendered_frames": [],
        }

    def fake_child(**kwargs: Any) -> int:
        Path(kwargs["child_output_path"]).write_text("ok\n", encoding="utf-8")
        Path(kwargs["child_final_path"]).write_text("done\n", encoding="utf-8")
        physics_usd.write_bytes(b"child-mutated")
        return 0

    monkeypatch.setattr(runner, "_finalize_physics_once", fake_finalize)
    monkeypatch.setattr(
        runner,
        "_capture_physics_vomp_tuning_baseline",
        lambda **kwargs: SimpleNamespace(
            result_path=raw_dir / "vomp" / "physics_vomp_result.json",
            physics_usd_bytes=Path(kwargs["physics_usd"]).read_bytes(),
        ),
    )
    restored: list[Path] = []

    def fake_restore(baseline: SimpleNamespace) -> None:
        physics_usd.write_bytes(baseline.physics_usd_bytes)
        restored.append(physics_usd)

    monkeypatch.setattr(
        runner,
        "_restore_physics_vomp_tuning_baseline",
        fake_restore,
    )
    monkeypatch.setattr(
        runner,
        "_write_physics_visual_issue_packet",
        lambda **_kwargs: raw_dir / "issue_packet.json",
    )
    monkeypatch.setattr(
        runner,
        "build_physics_visual_refinement_prompt",
        lambda **_kwargs: "prompt",
    )
    monkeypatch.setattr(runner, "_run_child_agent", fake_child)
    config = PhysicsApplyConfig(
        repo_root=tmp_path,
        usd_path=source_usd,
        vqa_refinement_max_iterations=1,
        vomp_mass=PhysicsVompMassConfig(runtime_root=tmp_path / "VoMP"),
    )
    runner._stage_usd_cli_input_tree(
        label="source",
        source_usd_path=source_usd,
        run_dir=run_dir,
    )
    packet = physics_packet_writer(run_dir, source_usd)
    inspection_pin = runner._physics_inspection_pin_from_packet(config, packet)

    returncode, _session_id = runner._run_physics_visual_refinement_loop(
        config=config,
        run_dir=run_dir,
        inspection_pin=inspection_pin,
        session_id="session-1",
        trace_writer=TraceWriter(run_dir),
        scene_service=None,
        usd_cli_session=_fake_usd_cli_session(run_dir),
        run_state={},
    )

    history = json.loads(
        (raw_dir / "physics_visual_validation_history.json").read_text(encoding="utf-8")
    )
    assert returncode == 2
    assert history["status"] == "vomp_attestation_changed"
    assert history["iterations"][-1]["vomp_attestation_changed"] is True
    assert history["iterations"][-1]["vomp_artifacts_restored"] is True
    assert physics_usd.read_bytes() == b"wrapper-attested"
    assert restored == [physics_usd]
