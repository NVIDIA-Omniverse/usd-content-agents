# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for validation scaffold helper branches."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import world_understanding.agentic.validation_scaffold as vs
from world_understanding.functions.physics.physical_behavior_evidence import (
    PhysicalBehaviorEvidence,
    PhysicalBehaviorEvidenceResolution,
)
from world_understanding.utils.input_resolver import InputInventory


def _inventory(tmp_path: Path) -> InputInventory:
    return InputInventory(
        items=(),
        usd_paths=(tmp_path / "asset.usda",),
        image_paths=(tmp_path / "render.png",),
        video_paths=(tmp_path / "rollout.mp4",),
        render_bundle_dirs=(),
        render_bundle_image_paths=(tmp_path / "bundle.png",),
        focus_prim_paths=("/World/Cube",),
        working_dir=tmp_path,
    )


def _context(tmp_path: Path) -> vs.DraftValidationContext:
    inventory = _inventory(tmp_path)
    request = vs.DraftValidationRequest(
        task_description="Validate asset behavior.",
        inputs=(),
        working_dir=tmp_path,
        base_dir=tmp_path,
        policy={},
    )
    plan = vs.DraftValidationPlan(
        steps=(),
        input_inventory=inventory,
        reasoning_summary="test",
    )
    return vs.DraftValidationContext(
        request=request,
        plan=plan,
        input_inventory=inventory,
        working_dir=tmp_path,
    )


def _context_with_policy(
    tmp_path: Path,
    policy: dict[str, object],
) -> vs.DraftValidationContext:
    context = _context(tmp_path)
    request = vs.DraftValidationRequest(
        task_description=context.request.task_description,
        inputs=context.request.inputs,
        working_dir=context.request.working_dir,
        base_dir=context.request.base_dir,
        policy=policy,
    )
    return vs.DraftValidationContext(
        request=request,
        plan=context.plan,
        input_inventory=context.input_inventory,
        working_dir=context.working_dir,
    )


def test_registry_fake_template_and_policy_value_helpers(tmp_path: Path) -> None:
    registry = vs.create_default_scaffold_registry()
    assert registry.names() == (
        "look_right",
        "render_valid",
        "physics_sane",
        "physical_behavior",
    )

    result = vs._FakeTemplate(
        name="look_right",
        issue_code="fake.issue",
        message="fake scaffold result",
    ).run(_context(tmp_path))

    assert result.status == "skipped"
    assert result.metrics["usd_path_count"] == 1
    assert result.metrics["image_path_count"] == 2
    assert result.issues[0].code == "fake.issue"

    assert vs._policy_value_present(["", None, "value"]) is True
    assert vs._policy_value_present(["", None, []]) is False
    assert vs._policy_value_present({"enabled": False}) is True
    assert vs._policy_value_present(0) is True


def test_look_right_template_handles_empty_live_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_plan = SimpleNamespace(
        issues=(),
        ready_for_judge=True,
        image_caption_pairs=(),
        evidence_images=(),
        to_dict=lambda: {"ready_for_judge": True},
    )
    fake_invocation = SimpleNamespace(
        raw_response=None,
        backend_name="fake",
        model_name="fake-vlm",
        to_dict=lambda: {"raw_response": None},
    )
    monkeypatch.setattr(
        vs,
        "build_look_right_judge_plan",
        lambda *args, **kwargs: fake_plan,
    )
    monkeypatch.setattr(
        vs,
        "_invoke_live_look_right_judge",
        lambda plan, config: (fake_invocation, None),
    )

    result = vs._LookRightTemplate().run(
        _context_with_policy(
            tmp_path,
            {"look_right_vlm": {"backend": "fake"}},
        )
    )

    assert result.status == "skipped"
    assert result.issues[0].code == vs.VISUAL_JUDGE_UNAVAILABLE
    assert result.metrics["issue_count"] == 1


def test_behavior_evidence_policy_and_refine_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = {
        "physical_behavior_evidence": "direct.json",
        "behavior_evidence": {"path": "mapped.mp4", "kind": "video"},
        "time_sampled_usd_paths": ["motion.usda"],
        "animation_usd_paths": ("anim.usda",),
        "behavior_video_paths": ["behavior.mp4"],
        "video_paths": "clip.mov",
        "simulation_json_paths": ["sim.json"],
        "trajectory_metrics_paths": ["history.jsonl"],
        "sampled_video_frame_paths": "frame.png",
    }

    policy_specs = vs._policy_physical_behavior_evidence(policy)
    assert {
        "path": "motion.usda",
        "kind": "time_sampled_usd",
        "role": "time_sampled_usd",
    } in policy_specs
    assert vs._normalize_evidence_policy_value("one.mp4") == ["one.mp4"]
    assert vs._normalize_evidence_policy_value({"path": "two.mp4"}) == [
        {"path": "two.mp4"}
    ]
    assert vs._normalize_evidence_policy_value(("a", "b")) == ["a", "b"]
    assert vs._normalize_evidence_policy_value(7) == [7]
    assert vs._sampled_frame_evidence(policy, required=True)[0]["required"] is True

    monkeypatch.chdir(tmp_path)
    assert vs._resolve_policy_path("relative/path", None).is_absolute()
    missing_specs = vs._discover_refine_output_evidence(
        tmp_path / "missing-refine",
        required=True,
    )
    assert missing_specs == [
        {
            "path": tmp_path / "missing-refine" / "refine_summary.json",
            "kind": "simulation_json",
            "role": "refine_summary",
            "required": True,
        }
    ]
    assert vs._iter_refine_dirs(tmp_path / "missing-refine") == ()

    refine_dir = tmp_path / "refine"
    final_dir = refine_dir / "final"
    render_dir = final_dir / "render"
    render_dir.mkdir(parents=True)
    for filename in (
        "history.jsonl",
        "judge_result.json",
        "refine_result.json",
        "tune_results.json",
    ):
        (final_dir / filename).write_text("{}", encoding="utf-8")
    (render_dir / "rollout.mp4").write_bytes(b"video")
    (render_dir / "frame.png").write_bytes(b"image")

    final_specs = vs._discover_refine_output_evidence(refine_dir, required=False)
    roles = {spec["role"] for spec in final_specs}
    assert {
        "final_trial_history",
        "final_judge_result",
        "final_refine_result",
        "final_tune_results",
        "rendered_rollout",
        "sampled_frame",
    } <= roles

    deduped = vs._dedupe_evidence_specs(
        [
            "plain",
            "plain",
            {"path": "p", "kind": "video", "role": "r"},
            {"path": "p", "kind": "video", "role": "r"},
        ]
    )
    assert deduped == ("plain", {"path": "p", "kind": "video", "role": "r"})


def test_behavior_summary_and_semantic_helpers(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert vs._load_json_mapping(bad_json) is None
    assert vs._load_json_mapping(list_json) is None
    assert (
        vs._physical_behavior_refine_summary_results(
            (
                PhysicalBehaviorEvidence(
                    original=str(bad_json),
                    path=str(bad_json),
                    kind="simulation_json",
                    exists=True,
                    required=False,
                ),
            )
        )
        == ()
    )

    summary_path = tmp_path / "refine_summary.json"
    payload = {
        "final_iteration": 2,
        "iterations": [
            {"iteration": 1, "judge_llm_unavailable": True},
            {
                "iteration": 2,
                "judge_decision": "continue",
                "judge_score": "0.4",
                "judge_reasoning": "try again",
                "refine_llm_unavailable": True,
            },
        ],
    }
    summary = vs._summarize_refine_summary(summary_path, payload)
    assert summary["final_iteration"] == 2
    assert summary["status"] == "needs_refinement"
    assert summary["judge_llm_unavailable_count"] == 1
    assert summary["refine_llm_unavailable_count"] == 1
    inferred_iteration = vs._summarize_refine_summary(
        summary_path,
        {"iterations": [{"iteration": 3, "judge_decision": "approve"}]},
    )
    assert inferred_iteration["final_iteration"] == 3
    assert (
        vs._final_refine_record({"final_iteration": 99}, payload["iterations"])
        == (payload["iterations"][-1])
    )

    assert (
        vs._behavior_status_from_refine_summary(
            termination_reason="degraded",
            judge_decision=None,
            error=None,
            cancelled=False,
        )
        == "warn"
    )
    assert (
        vs._behavior_status_from_refine_summary(
            termination_reason="approved",
            judge_decision=None,
            error=None,
            cancelled=False,
        )
        == "passed"
    )
    assert (
        vs._behavior_status_from_refine_summary(
            termination_reason="max_iterations",
            judge_decision=None,
            error=None,
            cancelled=False,
        )
        == "needs_refinement"
    )
    assert (
        vs._behavior_status_from_refine_summary(
            termination_reason=None,
            judge_decision="approve",
            error=None,
            cancelled=False,
        )
        == "passed"
    )
    assert (
        vs._behavior_status_from_refine_summary(
            termination_reason=None,
            judge_decision="continue",
            error=None,
            cancelled=False,
        )
        == "needs_refinement"
    )
    assert (
        vs._behavior_status_from_refine_summary(
            termination_reason=None,
            judge_decision=None,
            error=None,
            cancelled=False,
        )
        == "warn"
    )

    empty_resolution = PhysicalBehaviorEvidenceResolution()
    assert vs._physical_behavior_semantic_result(
        empty_resolution,
        refine_summary_results=(),
        behavior_evidence_required=False,
    ) == ("skipped", (), {"reason": "no_behavior_evidence"})

    evidence = PhysicalBehaviorEvidence(
        original="rollout.mp4",
        path=str(tmp_path / "rollout.mp4"),
        kind="video",
        exists=True,
        required=False,
    )
    status, issues, chosen = vs._physical_behavior_semantic_result(
        PhysicalBehaviorEvidenceResolution(evidence=(evidence,)),
        refine_summary_results=(
            {
                "summary": {
                    "status": "unknown",
                    "judge_llm_unavailable_count": 1,
                    "refine_llm_unavailable_count": 2,
                }
            },
        ),
        behavior_evidence_required=False,
    )
    assert status == "warn"
    assert chosen["status"] == "unknown"
    assert {issue.code for issue in issues} == {
        vs.BEHAVIOR_JUDGE_UNAVAILABLE,
        vs.BEHAVIOR_REFINER_UNAVAILABLE,
    }
    assert vs._coerce_behavior_status("unknown") == "warn"
    assert vs._choose_behavior_summary(()) == {
        "status": "warn",
        "reason": "summary_missing",
    }


def test_live_look_right_model_config_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        vs._look_right_vlm_available(
            {"look_right_vlm_available": False},
            raw_judge_response="response",
            live_judge_config={"backend": "fake"},
        )
        is False
    )
    assert (
        vs._look_right_vlm_available(
            {},
            raw_judge_response=None,
            live_judge_config={"backend": "fake"},
        )
        is True
    )
    assert (
        vs._look_right_live_judge_config({"look_right_vlm": {"enabled": False}}) is None
    )
    live_config = vs._look_right_live_judge_config(
        {
            "vlm": {"backend": "nested"},
            "look_right_vlm_model": "model",
            "look_right_vlm_api_key": "secret",
        }
    )
    assert live_config == {"backend": "nested", "model": "model", "api_key": "secret"}
    assert vs._look_right_final_judge_config({"llm_judge": {"enabled": False}}) is None

    invocation, issue = vs._invoke_live_look_right_judge(object(), None)
    assert invocation is None
    assert issue is not None
    with pytest.raises(ValueError, match="no judge"):
        vs._UnavailableTextJudge(ValueError("no judge")).invoke()

    monkeypatch.setattr(
        vs, "get_nim_api_key_for_base_url", lambda base_url, key: "nim-key"
    )
    monkeypatch.setattr(
        vs,
        "get_openai_api_key_for_base_url",
        lambda base_url, key: "openai-key",
    )
    monkeypatch.setattr(
        vs,
        "get_env_api_key_for_backend",
        lambda backend, key: f"{backend}-env-key",
    )
    assert vs._resolve_live_vlm_api_key(
        "nim", base_url="url", explicit_api_key=None
    ) == ("nim-key")
    assert (
        vs._resolve_live_vlm_api_key(
            "openai",
            base_url="url",
            explicit_api_key=None,
        )
        == "openai-key"
    )
    assert (
        vs._resolve_live_vlm_api_key(
            "anthropic",
            base_url=None,
            explicit_api_key=None,
        )
        == "anthropic-env-key"
    )
    assert (
        vs._resolve_live_vlm_api_key(
            "custom",
            base_url=None,
            explicit_api_key="  explicit  ",
        )
        == "explicit"
    )
    assert vs._config_string({"backend": "   "}, "backend") is None

    created_chat: dict[str, object] = {}
    created_vlm: dict[str, object] = {}
    monkeypatch.setattr(
        vs,
        "create_chat_model",
        lambda **kwargs: created_chat.setdefault("kwargs", kwargs) or object(),
    )
    monkeypatch.setattr(
        vs,
        "create_vlm",
        lambda **kwargs: created_vlm.setdefault("kwargs", kwargs) or object(),
    )
    vs._create_live_look_right_llm(
        {
            "backend": "custom",
            "api_key": " explicit ",
            "base_url": "https://example.invalid",
            "temperature": 0.2,
            "max_tokens": 10,
        }
    )
    vs._create_live_look_right_vlm(
        {
            "backend": "custom",
            "api_key": " explicit ",
            "base_url": "https://example.invalid",
        }
    )
    assert created_chat["kwargs"]["api_key"] == "explicit"
    assert "temperature" not in created_chat["kwargs"]
    assert created_vlm["kwargs"]["api_key"] == "explicit"


def test_adapter_mapping_and_render_response_helpers() -> None:
    assert vs._render_response_camera_names("not-mapping") == ()
    assert vs._render_response_camera_names({"results": "bad"}) == ()
    assert vs._render_response_camera_names(
        {"results": ["bad", {"camera_path": "CamA"}, {"camera_name": "CamA"}]}
    ) == ("CamA",)

    focused = vs._optional_focused_image_paths(
        {
            "focused_image_paths": {
                1: ["ignored.png"],
                "/World/A": "a.png",
                "/World/B": 7,
                "/World/C": ["c.png", object()],
            }
        }
    )
    assert focused == {"/World/A": ("a.png",), "/World/C": ("c.png",)}
    assert vs._focused_image_path_values(focused) == ("a.png", "c.png")

    adapter_result = {"metrics": {}, "issues": [], "status": "pass", "evidence": {}}
    template_results = [
        vs.DraftTemplateResult(template_name="other", status="passed"),
        vs.DraftTemplateResult(
            template_name="render_valid",
            status="passed",
            metadata={"adapter_result": adapter_result},
        ),
    ]
    assert (
        vs._previous_adapter_result(template_results, "render_valid") is adapter_result
    )
    assert (
        vs._previous_adapter_result(
            [vs.DraftTemplateResult(template_name="other", status="passed")],
            "render_valid",
        )
        is None
    )
    sequence_results = [
        vs.DraftTemplateResult(
            template_name="render_valid",
            status="passed",
            metadata={"adapter_results": ["bad", adapter_result]},
        )
    ]
    assert (
        vs._previous_adapter_result(sequence_results, "render_valid") is adapter_result
    )

    draft = vs._draft_result_from_adapter_result(
        template_name="render_valid",
        adapter_result={
            "metrics": {"m": 1},
            "issues": [{"code": "adapter.issue", "severity": "warning"}],
            "status": "pass",
            "evidence": {"usd_path": "asset.usda"},
        },
        status="passed",
        metadata={
            "runtime_render": {
                "status": "failed",
                "issues": [
                    {
                        "code": "render.failed",
                        "severity": "fail",
                        "message": "render failed",
                    }
                ],
            }
        },
    )
    assert draft.status == "failed"
    assert draft.metadata["adapter_result"]["status"] == "fail"
    assert len(draft.issues) == 2

    assert vs._adapter_status_from_template_status("passed") == "pass"
    assert vs._adapter_status_from_template_status("failed") == "fail"
    assert vs._adapter_status_from_template_status("warn") == "warn"
    assert vs._aggregate_adapter_statuses([{"status": "error"}]) == "error"
    assert vs._aggregate_adapter_statuses([{"verdict": "fail"}]) == "failed"
    assert vs._aggregate_adapter_statuses([{"verdict": "warn"}]) == "warn"
    assert vs._aggregate_adapter_statuses([{"status": "skipped"}]) == "skipped"
    assert vs._aggregate_adapter_statuses([{"status": "pass"}]) == "passed"


def test_issue_status_and_optional_helpers() -> None:
    assert (
        vs._look_right_issue_severity(
            SimpleNamespace(severity="info", code="visual.other")
        )
        == "info"
    )
    assert (
        vs._look_right_issue_severity(
            SimpleNamespace(severity="error", code=vs.VISUAL_EVIDENCE_MISSING)
        )
        == "warn"
    )
    assert vs._issues_from_look_right_judgment({"issue_codes": "not-a-list"}) == ()
    assert (
        vs._issues_from_look_right_judgment(
            {"issue_codes": ["a", "", 1], "verdict": "fail"}
        )[0].severity
        == "fail"
    )
    assert (
        vs._issues_from_look_right_judgment(
            {"issue_codes": ["a"], "verdict": "needs_refinement"}
        )[0].severity
        == "warn"
    )
    assert (
        vs._issues_from_look_right_judgment({"issue_codes": ["a"], "verdict": "pass"})[
            0
        ].severity
        == "info"
    )
    assert (
        vs._look_right_unready_status(
            (
                vs.DraftValidationIssue(
                    code="visual.blocking",
                    severity="fail",
                    message="blocked",
                ),
            )
        )
        == "failed"
    )
    assert vs._look_right_judgment_status("pass") == "passed"
    assert vs._look_right_judgment_status("fail") == "failed"
    assert vs._look_right_judgment_status("needs_refinement") == "needs_refinement"
    assert vs._look_right_judgment_status("warn") == "warn"
    assert vs._look_right_judgment_status("other") == "error"
    assert vs._issue_severity("info") == "info"
    assert vs._render_adapter_status({"status": "unexpected"}) == "error"

    assert vs._optional_path_sequence({"x": 1}, "x") is None
    assert vs._optional_string_sequence({"x": "one"}, "x") == ("one",)
    assert vs._optional_string_sequence({"x": 1}, "x") is None
    assert vs._optional_sequence({"x": "one"}, "x") == ("one",)
    assert vs._optional_sequence({"x": 1}, "x") is None
    assert vs._optional_float({"x": 1.5}, "x", 0.0) == 1.5
    assert vs._optional_int({"x": 2}, "x", 0) == 2

    report = {"asset_validator_report": {"asset.usda": {"ok": True}, "default": False}}
    assert vs._asset_validator_report(report, "asset.usda") == {"ok": True}
    assert (
        vs._asset_validator_report(report, "other.usda")
        == report["asset_validator_report"]
    )
    assert (
        vs._asset_validator_report({"asset_validator_report": "bad"}, "asset.usda")
        is None
    )
    assert vs._dedupe_preserve_order(["a", "a", "b"]) == ("a", "b")
