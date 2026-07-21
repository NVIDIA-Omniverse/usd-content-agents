# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image as PILImage
from PIL import ImageDraw

from world_understanding.agentic.validation_scaffold import (
    MAX_BEHAVIOR_RENDER_EVIDENCE_FILES,
    DraftTemplateResult,
    DraftValidationContext,
    DraftValidationError,
    TemplateRegistry,
    create_draft_validation_request,
    run_validation_scaffold,
)
from world_understanding.utils.token_tracking import TokenUsage


class _FakeLookRightVLM:
    backend_name = "fake"
    model_name = "fake-vlm"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []
        self.last_token_usage = TokenUsage(
            input_tokens=12,
            output_tokens=8,
            total_tokens=20,
            model_name=self.model_name,
            invocation_type="vlm",
        )

    def generate_with_image_caption_pairs(
        self,
        *,
        image_caption_pairs: list[tuple[str, str]],
        final_prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs: object,
    ) -> str:
        self.calls.append(
            {
                "image_caption_pairs": image_caption_pairs,
                "final_prompt": final_prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "kwargs": kwargs,
            }
        )
        return self.response


class _FakeLookRightLLM:
    backend_name = "fake"
    model_name = "fake-llm"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []
        self.last_token_usage = TokenUsage(
            input_tokens=5,
            output_tokens=4,
            total_tokens=9,
            model_name=self.model_name,
            invocation_type="llm",
        )

    def invoke(self, messages: list[object], **kwargs: object) -> object:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return type("Response", (), {"content": self.response})()


@pytest.fixture(autouse=True)
def _clear_runtime_render_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)


def _write_valid_image(path: Path) -> Path:
    image = _valid_image()
    image.save(path)
    return path


def _valid_image() -> PILImage.Image:
    image = PILImage.new("RGB", (64, 64), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 31, 31), fill=(255, 0, 0))
    draw.rectangle((32, 0, 63, 31), fill=(0, 255, 0))
    draw.rectangle((0, 32, 31, 63), fill=(0, 0, 255))
    draw.rectangle((32, 32, 63, 63), fill=(255, 255, 0))
    return image


def test_dry_run_writes_plan_without_inventing_focus_prims(tmp_path: Path) -> None:
    image_path = tmp_path / "render.png"
    image_path.write_bytes(b"not decoded by scaffold")
    working_dir = tmp_path / "run"
    request = create_draft_validation_request(
        task_description="Check that the asset looks right.",
        inputs=("render.png",),
        base_dir=tmp_path,
        working_dir=working_dir,
        dry_run=True,
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "planned"
    assert (working_dir / "plan.json").is_file()
    assert not (working_dir / "validation_result.json").exists()
    plan_data = json.loads((working_dir / "plan.json").read_text(encoding="utf-8"))
    assert [step["template_name"] for step in plan_data["steps"]] == [
        "render_valid",
        "look_right",
    ]
    assert plan_data["input_inventory"]["focus_prim_paths"] == []


def test_execution_runs_adapter_templates_in_profile_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    image_path = _write_valid_image(tmp_path / "render.png")
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    working_dir = tmp_path / "run"

    def fake_physics_sane_adapter(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "template": "physics_sane",
            "status": "completed",
            "verdict": "pass",
            "passed": True,
            "issues": [],
            "metrics": {"opened": True, "physics_expected": True},
            "evidence": {"usd_path": str(usd_path)},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.run_physics_sane_adapter",
        fake_physics_sane_adapter,
    )

    request = create_draft_validation_request(
        task_description="Validate visual quality and rigid body physics.",
        inputs=(usd_path, image_path),
        working_dir=working_dir,
        focus_prim_paths=("/World/Asset",),
        policy={"expect_physics": True},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "warn"
    assert [template.template_name for template in result.template_results] == [
        "physics_sane",
        "render_valid",
        "look_right",
    ]
    assert {issue.code for issue in result.issues} == {"visual.judge_unavailable"}
    result_data = json.loads(
        (working_dir / "validation_result.json").read_text(encoding="utf-8")
    )
    assert result_data["plan"]["input_inventory"]["focus_prim_paths"] == [
        "/World/Asset"
    ]
    assert result_data["metrics"]["physics_sane"]["usd_path_count"] == 1
    assert result_data["metrics"]["render_valid"]["image_count"] == 1
    assert result_data["template_results"][0]["metadata"]["adapter"] == "physics_sane"
    assert result_data["template_results"][1]["metadata"]["adapter"] == "render_valid"


def test_render_valid_template_uses_real_image_adapter(tmp_path: Path) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    working_dir = tmp_path / "run"
    request = create_draft_validation_request(
        task_description="Check render evidence.",
        inputs=(image_path,),
        working_dir=working_dir,
        requested_templates=("render_valid",),
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "pass"
    assert result.template_results[0].status == "passed"
    assert result.template_results[0].issues == ()
    assert result.metrics["render_valid"]["image_count"] == 1
    result_data = json.loads(
        (working_dir / "validation_result.json").read_text(encoding="utf-8")
    )
    assert result_data["evidence"]["render_valid"]["image_paths"] == [str(image_path)]
    assert result_data["template_results"][0]["metadata"]["adapter"] == "render_valid"


def test_render_valid_template_error_material_detection_is_policy_gated(
    tmp_path: Path,
) -> None:
    image = PILImage.new("RGB", (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 8, 55, 55], fill=(255, 0, 0))
    image_path = tmp_path / "red_asset.png"
    image.save(image_path)

    default_request = create_draft_validation_request(
        task_description="Check render evidence.",
        inputs=(image_path,),
        working_dir=tmp_path / "default",
        requested_templates=("render_valid",),
    )
    default_result = run_validation_scaffold(default_request)

    opt_in_request = create_draft_validation_request(
        task_description="Check render evidence.",
        inputs=(image_path,),
        working_dir=tmp_path / "opt_in",
        requested_templates=("render_valid",),
        policy={"render_detect_error_material_artifacts": True},
    )
    opt_in_result = run_validation_scaffold(opt_in_request)

    assert default_result.template_results[0].status == "passed"
    assert opt_in_result.template_results[0].status == "failed"
    assert opt_in_result.template_results[0].issues[0].code == (
        "render.suspected_error_material"
    )


def test_render_valid_template_rejects_invalid_error_material_policy(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    request = create_draft_validation_request(
        task_description="Check render evidence.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid",),
        policy={"render_detect_error_material_artifacts": 1},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "fail"
    assert result.template_results[0].status == "error"
    assert result.template_results[0].issues[0].code == "agent.template_error"
    assert "policy.render_detect_error_material_artifacts must be a bool" in (
        result.template_results[0].issues[0].message
    )
    assert "got 1" in result.template_results[0].issues[0].message


def test_look_right_template_consumes_render_valid_handoff_and_response(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    reference_path = _write_valid_image(tmp_path / "reference.png")
    working_dir = tmp_path / "run"
    request = create_draft_validation_request(
        task_description="Validate that the asset looks like a yellow toolbox.",
        inputs=(image_path,),
        working_dir=working_dir,
        requested_templates=("render_valid", "look_right"),
        policy={
            "reference_image_paths": [reference_path],
            "look_right_response": """
Critique: The render matches the requested toolbox silhouette and colors.
Score: 9
Decision: PASS
Issue Codes: none
""",
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "pass"
    assert [template.status for template in result.template_results] == [
        "passed",
        "passed",
    ]
    look_right = result.template_results[1]
    assert look_right.metadata["result_helper"] == "normalize_look_right_judgment"
    assert look_right.metadata["final_judge_method"] == "parser"
    assert look_right.metrics["ready_for_judge"] is True
    assert look_right.metrics["final_judge_method"] == "parser"
    assert look_right.metrics["llm_final_judge_invoked"] is False
    assert look_right.metrics["judgment_verdict"] == "pass"
    assert look_right.metrics["judgment_score"] == 0.9
    assert look_right.evidence["render_valid_handoff"]["status"] == "pass"
    assert look_right.evidence["judge_plan"]["ready_for_judge"] is True
    assert look_right.evidence["image_caption_pairs"] == [
        {"caption": "Reference Image 1:", "path": str(reference_path)},
        {"caption": "Current Asset Evidence - View 1:", "path": str(image_path)},
    ]

    result_data = json.loads(
        (working_dir / "validation_result.json").read_text(encoding="utf-8")
    )
    assert result_data["evidence"]["look_right"]["judgment"]["verdict"] == "pass"


def test_look_right_template_invokes_live_vlm_executor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    reference_path = _write_valid_image(tmp_path / "reference.png")
    fake_vlm = _FakeLookRightVLM(
        """
Critique: The render matches the requested toolbox evidence.
Score: 8
Decision: PASS
Issue Codes: none
"""
    )
    created_kwargs: dict[str, object] = {}

    def fake_create_vlm(**kwargs: object) -> _FakeLookRightVLM:
        created_kwargs.update(kwargs)
        return fake_vlm

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.create_vlm",
        fake_create_vlm,
    )
    request = create_draft_validation_request(
        task_description="Validate that the asset looks like a toolbox.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
        policy={
            "reference_image_paths": [reference_path],
            "look_right_vlm": {
                "backend": "fake",
                "model": "fake-vlm",
                "generation_kwargs": {"seed": 7},
            },
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "pass"
    look_right = result.template_results[1]
    assert look_right.status == "passed"
    assert look_right.metadata["vlm_invoked"] is True
    assert look_right.metadata["precomputed_response"] is False
    assert look_right.metrics["vlm_invoked"] is True
    assert look_right.metrics["judge_backend"] == "fake"
    assert look_right.metrics["judge_model"] == "fake-vlm"
    assert look_right.metrics["judgment_verdict"] == "pass"
    assert look_right.metrics["judgment_score"] == 0.8
    assert created_kwargs == {"backend": "fake", "model": "fake-vlm"}
    assert fake_vlm.calls[0]["image_caption_pairs"] == [
        ("Reference Image 1:", str(reference_path)),
        ("Current Asset Evidence - View 1:", str(image_path)),
    ]
    assert fake_vlm.calls[0]["kwargs"] == {"seed": 7}
    invocation = look_right.evidence["judge_invocation"]
    assert invocation["backend_name"] == "fake"
    assert invocation["model_name"] == "fake-vlm"
    assert invocation["token_usage"]["total_tokens"] == 20
    assert look_right.evidence["judgment"]["raw_response"].startswith("Critique:")


def test_look_right_template_uses_llm_final_judge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    fake_llm = _FakeLookRightLLM(
        """
{"decision": "needs_refinement", "score": 0.8,
 "issue_codes": ["visual.low_confidence"],
 "rationale": "The VLM response explicitly refuses PASS."}
"""
    )
    created_kwargs: dict[str, object] = {}

    def fake_create_chat_model(**kwargs: object) -> _FakeLookRightLLM:
        created_kwargs.update(kwargs)
        return fake_llm

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.create_chat_model",
        fake_create_chat_model,
    )
    request = create_draft_validation_request(
        task_description="Validate that the rendered asset looks correct.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
        policy={
            "look_right_response": """
Critique: The render still has issues.
Score: 9
Decision: I do not think this should PASS
Issue Codes: none
""",
            "look_right_llm_judge": {
                "backend": "fake",
                "model": "fake-llm",
                "temperature": 0,
                "max_tokens": 256,
            },
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "needs_refinement"
    look_right = result.template_results[1]
    assert look_right.status == "needs_refinement"
    assert look_right.metrics["final_judge_method"] == "llm"
    assert look_right.metrics["llm_final_judge_invoked"] is True
    assert look_right.metrics["final_judge_backend"] == "fake"
    assert look_right.metrics["final_judge_model"] == "fake-llm"
    assert look_right.evidence["judgment"]["verdict"] == "needs_refinement"
    assert look_right.evidence["final_judge"]["token_usage"]["total_tokens"] == 9
    assert created_kwargs == {"backend": "fake", "model": "fake-llm"}
    assert fake_llm.calls[0]["kwargs"] == {"temperature": 0.0, "max_tokens": 256}


def test_look_right_template_keeps_blocking_visual_defects_out_of_pass(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    reference_path = _write_valid_image(tmp_path / "reference.png")
    request = create_draft_validation_request(
        task_description="Validate that the asset matches the reference toolbox.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
        policy={
            "reference_image_paths": [reference_path],
            "look_right_response": """
Critique: The corner view shows the body material mismatches the reference.
Score: 9
Decision: PASS
Issue Codes: visual.reference_mismatch
Evidence Notes: The visible corner view supports the defect finding.
""",
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "needs_refinement"
    look_right = result.template_results[1]
    assert look_right.status == "needs_refinement"
    assert look_right.metrics["judgment_verdict"] == "needs_refinement"
    assert look_right.metrics["judgment_score"] == 0.9
    assert [issue.code for issue in look_right.issues] == ["visual.reference_mismatch"]


def test_look_right_template_warns_on_malformed_live_vlm_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    fake_vlm = _FakeLookRightVLM("I cannot provide the requested structured output.")

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.create_vlm",
        lambda **_: fake_vlm,
    )
    request = create_draft_validation_request(
        task_description="Validate that the rendered asset looks correct.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
        policy={"look_right_vlm": {"backend": "fake"}},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "warn"
    look_right = result.template_results[1]
    assert look_right.status == "warn"
    assert look_right.metadata["vlm_invoked"] is True
    assert look_right.metrics["judgment_verdict"] == "warn"
    assert look_right.metrics["judgment_score"] is None
    assert [issue.code for issue in look_right.issues] == ["visual.low_confidence"]
    assert look_right.evidence["judgment"]["raw_response"] == (
        "I cannot provide the requested structured output."
    )


def test_look_right_template_skips_when_live_vlm_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    nested_api_key = "nested-test-api-key"
    nested_token = "nested-test-token"

    def fake_create_vlm(**_: object) -> object:
        raise RuntimeError(f"service unavailable for {nested_api_key}")

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.create_vlm",
        fake_create_vlm,
    )
    request = create_draft_validation_request(
        task_description="Validate that the rendered asset looks correct.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
        policy={
            "look_right_vlm": {
                "backend": "fake",
                "api_key": "test-key",
                "generation_kwargs": {
                    "seed": 7,
                    "api_key": nested_api_key,
                    "messages": [{"token": nested_token}, {"safe": "visible"}],
                },
            }
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "warn"
    look_right = result.template_results[1]
    assert look_right.status == "skipped"
    assert [issue.code for issue in look_right.issues] == ["visual.judge_unavailable"]
    assert look_right.issues[0].details == {
        "configured": True,
        "model_config": {
            "backend": "fake",
            "api_key": "<redacted>",
            "generation_kwargs": {
                "seed": 7,
                "api_key": "<redacted>",
                "messages": [
                    {"token": "<redacted>"},
                    {"safe": "visible"},
                ],
            },
        },
        "error_type": "RuntimeError",
        "error": "service unavailable for <redacted>",
    }
    serialized_details = json.dumps(look_right.issues[0].details)
    assert nested_api_key not in serialized_details
    assert nested_token not in serialized_details
    assert look_right.metadata["vlm_invoked"] is False
    assert "judgment" not in look_right.evidence


def test_look_right_template_preflights_policy_render_and_focus_images(
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    render_path = _write_valid_image(tmp_path / "render.png")
    focus_path = _write_valid_image(tmp_path / "focus_handle.png")
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    request = create_draft_validation_request(
        task_description="Validate that the asset render and handle close-up look correct.",
        inputs=(usd_path,),
        working_dir=tmp_path / "run",
        focus_prim_paths=("/World/Handle",),
        requested_templates=("render_valid", "look_right"),
        policy={
            "render_image_paths": [render_path],
            "focused_image_paths": {"/World/Handle": [focus_path]},
            "look_right_response": """
Critique: The whole asset render and handle close-up match the request.
Score: 8
Decision: PASS
Issue Codes: none
""",
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "pass"
    render_valid = result.template_results[0]
    assert render_valid.status == "passed"
    assert render_valid.metrics["image_count"] == 2
    assert render_valid.evidence["image_paths"] == [
        str(render_path),
        str(focus_path),
    ]

    look_right = result.template_results[1]
    assert look_right.status == "passed"
    assert look_right.evidence["render_valid_handoff"]["status"] == "pass"
    assert look_right.evidence["judge_plan"]["ready_for_judge"] is True
    assert look_right.evidence["image_caption_pairs"] == [
        {"caption": "Current Render Output - View 1:", "path": str(render_path)},
        {
            "caption": "Focused Asset Evidence - /World/Handle - View 1:",
            "path": str(focus_path),
        },
    ]


def test_usd_visual_templates_render_runtime_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        calls.append(
            {
                "usd_paths": tuple(usd_paths),
                "working_dir": working_dir,
                "policy": policy,
            }
        )
        render_dir = Path(working_dir) / "renders" / "asset"
        render_dir.mkdir(parents=True)
        render_path = _write_valid_image(render_dir / "asset_front_0000.png")
        return {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(render_path)],
            "render_response": {
                "backend": "remote",
                "status": "completed",
                "results": [
                    {
                        "camera": "front",
                        "camera_path": "/ValidationAgentCameras/front",
                        "images": [str(render_path)],
                        "frame_count": 1,
                        "status": "success",
                    }
                ],
            },
            "render_output_dir": str(render_dir),
            "issues": [],
            "metadata": {"image_count": 1, "views": ["front"]},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )
    request = create_draft_validation_request(
        task_description="Validate that the generated toolbox looks correct.",
        inputs=(usd_path,),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
        policy={
            "expected_cameras": ["front"],
            "look_right_response": """
Critique: The rendered toolbox matches the requested visual evidence.
Score: 8
Decision: PASS
Issue Codes: none
""",
        },
    )

    result = run_validation_scaffold(request)

    assert calls[0]["usd_paths"] == (usd_path.resolve(),)
    assert result.verdict == "pass"
    render_valid = result.template_results[0]
    assert render_valid.status == "passed"
    assert render_valid.metrics["image_count"] == 1
    assert render_valid.metadata["runtime_render"]["status"] == "completed"

    rendered_path = str(tmp_path / "run" / "renders" / "asset" / "asset_front_0000.png")
    assert render_valid.evidence["image_paths"] == [rendered_path]
    look_right = result.template_results[1]
    assert look_right.status == "passed"
    assert look_right.evidence["image_caption_pairs"] == [
        {"caption": "Current Render Output - View 1:", "path": rendered_path},
    ]


def test_canonical_usd_visual_evidence_ignores_supplied_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    caller_image_path = _write_valid_image(tmp_path / "caller_supplied.png")
    caller_render_path = _write_valid_image(tmp_path / "caller_render.png")
    caller_focus_path = _write_valid_image(tmp_path / "caller_focus.png")
    caller_frame_path = _write_valid_image(tmp_path / "caller_frame.png")
    caller_animation_path = _write_valid_image(tmp_path / "caller_animation.png")
    reference_path = _write_valid_image(tmp_path / "reference.png")
    calls: list[dict[str, object]] = []

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        calls.append(
            {
                "usd_paths": tuple(usd_paths),
                "working_dir": working_dir,
                "policy": policy,
            }
        )
        render_dir = Path(working_dir) / "renders" / "asset"
        render_dir.mkdir(parents=True)
        render_path = _write_valid_image(render_dir / "asset_corner_0000.png")
        return {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(render_path)],
            "render_response": {
                "backend": "remote",
                "status": "completed",
                "results": [
                    {
                        "camera": "corner",
                        "camera_label": "corner",
                        "camera_path": "/ValidationAgentCameras/corner",
                        "images": [str(render_path)],
                        "frame_count": 1,
                        "status": "success",
                    }
                ],
            },
            "render_output_dir": str(render_dir),
            "issues": [],
            "metadata": {"image_count": 1, "views": ["corner"]},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )
    request = create_draft_validation_request(
        task_description="Validate the USD artifact visual identity.",
        inputs=(usd_path, caller_image_path),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
        policy={
            "canonical_visual_evidence": True,
            "render_image_paths": [caller_render_path],
            "focused_image_paths": {"/World/Handle": [caller_focus_path]},
            "sampled_video_frame_paths": [caller_frame_path],
            "animation_frame_paths": [caller_animation_path],
            "frame_ids": [0],
            "reference_image_paths": [reference_path],
            "look_right_response": """
Critique: The canonical USD render matches the request.
Score: 8
Decision: PASS
Issue Codes: none
""",
        },
    )

    result = run_validation_scaffold(request)

    assert len(calls) == 1
    assert calls[0]["usd_paths"] == (usd_path.resolve(),)
    assert "runtime_render" not in result.request.policy
    assert result.request.policy["render_image_paths"] == [caller_render_path]
    rendered_path = str(
        tmp_path / "run" / "renders" / "asset" / "asset_corner_0000.png"
    )
    render_valid = result.template_results[0]
    assert render_valid.status == "passed"
    assert render_valid.evidence["image_paths"] == [rendered_path]
    assert render_valid.evidence["animation_frame_paths"] == []
    assert str(caller_image_path) not in render_valid.evidence["image_paths"]
    assert str(caller_render_path) not in render_valid.evidence["image_paths"]
    assert str(caller_focus_path) not in render_valid.evidence["image_paths"]

    look_right = result.template_results[1]
    assert look_right.status == "passed"
    assert look_right.metrics["evidence_mode"] == "canonical_usd"
    assert look_right.evidence["image_caption_pairs"] == [
        {"caption": "Reference Image 1:", "path": str(reference_path)},
        {"caption": "Current Render Output - View 1:", "path": rendered_path},
    ]
    look_right_paths = {
        pair["path"] for pair in look_right.evidence["image_caption_pairs"]
    }
    assert str(caller_image_path) not in look_right_paths
    assert str(caller_render_path) not in look_right_paths
    assert str(caller_focus_path) not in look_right_paths
    assert str(caller_frame_path) not in look_right_paths
    assert str(reference_path) in look_right_paths


def test_canonical_usd_visual_evidence_clears_supplied_render_paths_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    caller_render_path = _write_valid_image(tmp_path / "caller_render.png")

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        return {
            "status": "unavailable",
            "backend": "remote",
            "image_paths": [],
            "render_response": None,
            "render_output_dir": None,
            "issues": [
                {
                    "code": "render.renderer_unavailable",
                    "severity": "warn",
                    "message": "Renderer unavailable.",
                    "details": {},
                }
            ],
            "metadata": {"base_url_configured": False},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )
    request = create_draft_validation_request(
        task_description="Validate the USD artifact visual identity.",
        inputs=(usd_path,),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
        policy={
            "canonical_visual_evidence": True,
            "render_image_paths": [caller_render_path],
            "render_response": {
                "backend": "remote",
                "status": "completed",
                "results": [
                    {
                        "camera": "stale",
                        "images": [str(caller_render_path)],
                        "status": "success",
                    }
                ],
            },
            "render_output_dir": str(tmp_path),
            "look_right_response": """
Critique: This response must not be used without canonical render evidence.
Score: 9
Decision: PASS
Issue Codes: none
""",
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "fail"
    render_valid = result.template_results[0]
    assert render_valid.status == "failed"
    assert render_valid.evidence["image_paths"] == []
    assert {issue.code for issue in render_valid.issues} == {
        "render.canonical_runtime_render_missing",
        "render.evidence_missing",
        "render.renderer_unavailable",
    }

    look_right = result.template_results[1]
    assert look_right.status == "failed"
    assert look_right.metrics["evidence_mode"] == "canonical_usd"
    assert look_right.evidence["image_caption_pairs"] == []
    assert {issue.code for issue in look_right.issues} == {
        "visual.evidence_missing",
        "visual.render_preflight_failed",
    }


def test_canonical_usd_visual_evidence_fails_closed_when_renderer_returns_no_images_with_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    stale_render_path = _write_valid_image(tmp_path / "stale_render.png")

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        return {
            "status": "completed",
            "backend": "remote",
            "image_paths": [],
            "render_response": {
                "backend": "remote",
                "status": "completed",
                "results": [
                    {
                        "camera": "corner",
                        "images": [str(stale_render_path)],
                        "status": "success",
                    }
                ],
            },
            "render_output_dir": str(tmp_path),
            "issues": [],
            "metadata": {"image_count": 0},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )
    request = create_draft_validation_request(
        task_description="Validate the USD artifact visual identity.",
        inputs=(usd_path,),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
        policy={
            "canonical_visual_evidence": True,
            "look_right_response": """
Critique: This response must not be used without canonical render evidence.
Score: 9
Decision: PASS
Issue Codes: none
""",
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "fail"
    render_valid = result.template_results[0]
    assert render_valid.status == "failed"
    assert render_valid.evidence["image_paths"] == []
    assert render_valid.evidence["render_response_images"] == []
    assert {issue.code for issue in render_valid.issues} == {
        "render.canonical_runtime_render_missing",
        "render.evidence_missing",
    }

    look_right = result.template_results[1]
    assert look_right.status == "failed"
    assert look_right.metrics["evidence_mode"] == "canonical_usd"
    assert look_right.evidence["image_caption_pairs"] == []
    assert {issue.code for issue in look_right.issues} == {
        "visual.evidence_missing",
        "visual.render_preflight_failed",
    }


def test_canonical_usd_visual_evidence_with_runtime_render_disabled_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    caller_image_path = _write_valid_image(tmp_path / "caller_supplied.png")
    caller_render_path = _write_valid_image(tmp_path / "caller_render.png")
    caller_animation_path = _write_valid_image(tmp_path / "caller_animation.png")
    calls: list[object] = []

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        calls.append((usd_paths, working_dir, policy))
        raise AssertionError("runtime renderer should be disabled")

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )
    caplog.set_level("WARNING")
    request = create_draft_validation_request(
        task_description="Validate the USD artifact visual identity.",
        inputs=(usd_path, caller_image_path),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
        policy={
            "canonical_visual_evidence": True,
            "runtime_render_usd": False,
            "render_image_paths": [caller_render_path],
            "animation_frame_paths": [caller_animation_path],
            "frame_ids": [0],
            "look_right_response": """
Critique: The canonical USD render matches the request.
Score: 8
Decision: PASS
Issue Codes: none
""",
        },
    )

    result = run_validation_scaffold(request)

    assert calls == []
    assert "requires runtime rendering" in caplog.text
    assert result.verdict == "fail"
    render_valid = result.template_results[0]
    assert render_valid.status == "failed"
    assert render_valid.evidence["image_paths"] == []
    assert render_valid.evidence["animation_frame_paths"] == []
    assert {issue.code for issue in render_valid.issues} == {
        "render.canonical_runtime_render_disabled",
        "render.evidence_missing",
    }

    look_right = result.template_results[1]
    assert look_right.status == "failed"
    assert look_right.metrics["evidence_mode"] == "canonical_usd"
    assert look_right.evidence["image_caption_pairs"] == []
    assert {issue.code for issue in look_right.issues} == {
        "visual.evidence_missing",
        "visual.render_preflight_failed",
    }


@pytest.mark.parametrize("mode", ("canonical_usd", "canonical-usd"))
def test_visual_evidence_mode_enables_canonical_usd_rendering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    caller_image_path = _write_valid_image(tmp_path / "caller_supplied.png")
    caller_render_path = _write_valid_image(tmp_path / "caller_render.png")
    calls: list[dict[str, object]] = []

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        calls.append(
            {
                "usd_paths": tuple(usd_paths),
                "working_dir": working_dir,
                "policy": policy,
            }
        )
        render_dir = Path(working_dir) / "renders" / "asset"
        render_dir.mkdir(parents=True)
        render_path = _write_valid_image(render_dir / "asset_corner_0000.png")
        return {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(render_path)],
            "render_response": {
                "backend": "remote",
                "status": "completed",
                "results": [
                    {
                        "camera": "corner",
                        "camera_path": "/ValidationAgentCameras/corner",
                        "images": [str(render_path)],
                        "frame_count": 1,
                        "status": "success",
                    }
                ],
            },
            "render_output_dir": str(render_dir),
            "issues": [],
            "metadata": {"image_count": 1, "views": ["corner"]},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )
    request = create_draft_validation_request(
        task_description="Validate the USD artifact visual identity.",
        inputs=(usd_path, caller_image_path),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
        policy={
            "visual_evidence_mode": mode,
            "render_image_paths": [caller_render_path],
            "look_right_response": """
Critique: The canonical USD render matches the request.
Score: 8
Decision: PASS
Issue Codes: none
""",
        },
    )

    result = run_validation_scaffold(request)

    rendered_path = str(
        tmp_path / "run" / "renders" / "asset" / "asset_corner_0000.png"
    )
    assert len(calls) == 1
    render_valid = result.template_results[0]
    assert render_valid.status == "passed"
    assert render_valid.evidence["image_paths"] == [rendered_path]
    assert str(caller_image_path) not in render_valid.evidence["image_paths"]
    assert str(caller_render_path) not in render_valid.evidence["image_paths"]

    look_right = result.template_results[1]
    assert look_right.status == "passed"
    assert look_right.metrics["evidence_mode"] == "canonical_usd"
    assert look_right.evidence["image_caption_pairs"] == [
        {"caption": "Current Render Output - View 1:", "path": rendered_path},
    ]


@pytest.mark.parametrize("mode", ("usd", "runtime_usd", "provided_visual_evidence"))
def test_visual_evidence_mode_unknown_values_keep_provided_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    caller_image_path = _write_valid_image(tmp_path / "caller_supplied.png")
    caller_render_path = _write_valid_image(tmp_path / "caller_render.png")
    calls: list[object] = []

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        calls.append((usd_paths, working_dir, policy))
        raise AssertionError("unknown visual_evidence_mode should not force rendering")

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )
    request = create_draft_validation_request(
        task_description="Validate the USD artifact visual identity.",
        inputs=(usd_path, caller_image_path),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
        policy={
            "visual_evidence_mode": mode,
            "render_image_paths": [caller_render_path],
            "look_right_response": """
Critique: The provided images match the request.
Score: 8
Decision: PASS
Issue Codes: none
""",
        },
    )

    result = run_validation_scaffold(request)

    assert calls == []
    render_valid = result.template_results[0]
    assert render_valid.status == "passed"
    assert render_valid.evidence["image_paths"] == [
        str(caller_image_path),
        str(caller_render_path),
    ]
    look_right = result.template_results[1]
    assert look_right.status == "passed"
    assert look_right.metrics["evidence_mode"] == "provided_visual_evidence"
    assert {pair["path"] for pair in look_right.evidence["image_caption_pairs"]} == {
        str(caller_image_path),
        str(caller_render_path),
    }


def test_canonical_usd_visual_evidence_without_usd_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "caller_supplied.png")
    calls: list[object] = []

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        calls.append((usd_paths, working_dir, policy))
        raise AssertionError("canonical mode without USD input should not render")

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )
    caplog.set_level("WARNING")
    request = create_draft_validation_request(
        task_description="Validate the supplied image.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
        policy={
            "canonical_visual_evidence": True,
            "look_right_response": """
Critique: The provided image matches the request.
Score: 8
Decision: PASS
Issue Codes: none
""",
        },
    )

    result = run_validation_scaffold(request)

    assert calls == []
    assert "requires a USD input" in caplog.text
    assert result.verdict == "fail"
    render_valid = result.template_results[0]
    assert render_valid.status == "failed"
    assert render_valid.evidence["image_paths"] == []
    assert {issue.code for issue in render_valid.issues} == {
        "render.canonical_usd_input_missing",
        "render.evidence_missing",
    }
    look_right = result.template_results[1]
    assert look_right.status == "failed"
    assert look_right.metrics["evidence_mode"] == "canonical_usd"
    assert look_right.evidence["image_caption_pairs"] == []
    assert {issue.code for issue in look_right.issues} == {
        "visual.evidence_missing",
        "visual.render_preflight_failed",
    }


def test_canonical_runtime_render_response_scopes_multi_asset_expected_cameras(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_usd = tmp_path / "first.usda"
    second_usd = tmp_path / "second.usda"
    first_usd.write_text("#usda 1.0\n", encoding="utf-8")
    second_usd.write_text("#usda 1.0\n", encoding="utf-8")

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        render_dir = Path(working_dir) / "renders"
        render_dir.mkdir(parents=True)
        render_paths = [
            _write_valid_image(render_dir / "first_front.png"),
            _write_valid_image(render_dir / "second_front.png"),
        ]
        return {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(path) for path in render_paths],
            "render_response": {
                "backend": "remote",
                "status": "completed",
                "results": [
                    {
                        "camera": "first_asset:front",
                        "camera_label": "front",
                        "camera_path": "/ValidationAgentCameras/front",
                        "images": [str(render_paths[0])],
                        "frame_count": 1,
                        "status": "success",
                    },
                    {
                        "camera": "second_asset:front",
                        "camera_label": "front",
                        "camera_path": "/ValidationAgentCameras/front",
                        "images": [str(render_paths[1])],
                        "frame_count": 1,
                        "status": "success",
                    },
                ],
            },
            "render_output_dir": str(render_dir),
            "issues": [],
            "metadata": {"image_count": 2, "views": ["front"]},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )
    request = create_draft_validation_request(
        task_description="Validate both generated assets.",
        inputs=(first_usd, second_usd),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid",),
        policy={"canonical_visual_evidence": True, "expected_cameras": ["front"]},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "pass"
    assert result.template_results[0].status == "passed"
    assert result.template_results[0].metadata["expected_cameras"] == (
        "first_asset:front",
        "second_asset:front",
    )
    assert "runtime_render" not in result.request.policy


def test_usd_runtime_render_unavailable_is_structured_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        return {
            "status": "unavailable",
            "backend": "remote",
            "image_paths": [],
            "render_response": None,
            "render_output_dir": None,
            "issues": [
                {
                    "code": "render.renderer_unavailable",
                    "severity": "warn",
                    "message": "Set RENDER_ENDPOINT before rendering USD inputs.",
                    "details": {"required_env": ["RENDER_ENDPOINT"]},
                }
            ],
            "metadata": {"base_url_configured": False},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )
    request = create_draft_validation_request(
        task_description="Validate that the generated asset looks correct.",
        inputs=(usd_path,),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "warn"
    render_valid = result.template_results[0]
    assert render_valid.status == "skipped"
    assert [issue.code for issue in render_valid.issues] == [
        "render.evidence_missing",
        "render.renderer_unavailable",
    ]
    assert render_valid.metrics["issue_count"] == 2

    look_right = result.template_results[1]
    assert look_right.status == "skipped"
    assert {issue.code for issue in look_right.issues} == {
        "visual.evidence_missing",
        "visual.judge_unavailable",
        "visual.render_preflight_unavailable",
    }


def test_look_right_only_propagates_runtime_render_unavailable_issue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        return {
            "status": "unavailable",
            "backend": "remote",
            "image_paths": [],
            "render_response": None,
            "render_output_dir": None,
            "issues": [
                {
                    "code": "render.renderer_unavailable",
                    "severity": "warn",
                    "message": "Set RENDER_ENDPOINT before rendering USD inputs.",
                    "details": {"required_env": ["RENDER_ENDPOINT"]},
                }
            ],
            "metadata": {"base_url_configured": False},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )
    request = create_draft_validation_request(
        task_description="Validate that the generated asset looks correct.",
        inputs=(usd_path,),
        working_dir=tmp_path / "run",
        requested_templates=("look_right",),
        policy={
            "look_right_response": "\n".join(
                (
                    "Critique: This should not be used without current evidence.",
                    "Score: 9",
                    "Decision: PASS",
                    "Issue Codes: none",
                )
            )
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "warn"
    look_right = result.template_results[0]
    assert look_right.status == "skipped"
    assert {issue.code for issue in look_right.issues} == {
        "render.renderer_unavailable",
        "visual.evidence_missing",
    }
    assert {issue.code for issue in result.issues} == {
        "render.renderer_unavailable",
        "visual.evidence_missing",
    }


def test_look_right_template_skips_when_vlm_unavailable(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    working_dir = tmp_path / "run"
    request = create_draft_validation_request(
        task_description="Validate that the rendered asset looks correct.",
        inputs=(image_path,),
        working_dir=working_dir,
        requested_templates=("render_valid", "look_right"),
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "warn"
    assert result.template_results[0].status == "passed"
    look_right = result.template_results[1]
    assert look_right.status == "skipped"
    assert [issue.code for issue in look_right.issues] == ["visual.judge_unavailable"]
    assert look_right.metrics["ready_for_judge"] is False
    assert look_right.metrics["evidence_image_count"] == 1
    assert look_right.evidence["render_valid_handoff"]["status"] == "pass"
    assert look_right.evidence["judge_plan"]["ready_for_judge"] is False


def test_look_right_template_ignores_malformed_runtime_render_issues(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    request = create_draft_validation_request(
        task_description="Validate that the rendered asset looks correct.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("look_right",),
        policy={
            "look_right_response": "\n".join(
                (
                    "Critique: The supplied image looks consistent.",
                    "Score: 9",
                    "Decision: PASS",
                    "Issue Codes: none",
                )
            ),
            "runtime_render": {"issues": {"not": "a sequence"}},
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "pass"
    assert result.template_results[0].issues == ()


def test_look_right_template_skips_when_visual_evidence_missing(
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    request = create_draft_validation_request(
        task_description="Validate that the asset looks correct.",
        inputs=(usd_path,),
        working_dir=tmp_path / "run",
        requested_templates=("look_right",),
        policy={"runtime_render_usd": False},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "warn"
    look_right = result.template_results[0]
    assert look_right.status == "skipped"
    assert {issue.code for issue in look_right.issues} == {
        "visual.evidence_missing",
        "visual.judge_unavailable",
    }
    assert all(issue.severity == "warn" for issue in look_right.issues)
    assert look_right.metrics["evidence_image_count"] == 0


def test_look_right_template_blocks_when_render_valid_handoff_fails(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"not a decodable image")
    request = create_draft_validation_request(
        task_description="Validate that the rendered asset looks correct.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid", "look_right"),
        policy={
            "look_right_response": """
Critique: This response must not be used because render preflight failed.
Score: 9
Decision: PASS
Issue Codes: none
""",
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "fail"
    assert result.template_results[0].status == "failed"
    look_right = result.template_results[1]
    assert look_right.status == "failed"
    assert [issue.code for issue in look_right.issues] == [
        "visual.render_preflight_failed"
    ]
    assert "judgment" not in look_right.evidence
    assert look_right.evidence["render_valid_handoff"]["status"] == "fail"


def test_render_valid_template_uses_render_bundle_images(tmp_path: Path) -> None:
    render_bundle_dir = tmp_path / "renders"
    render_bundle_dir.mkdir()
    image_path = _write_valid_image(render_bundle_dir / "camera_a.png")
    working_dir = tmp_path / "run"
    request = create_draft_validation_request(
        task_description="Check render bundle evidence.",
        inputs=(render_bundle_dir,),
        working_dir=working_dir,
        requested_templates=("render_valid",),
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "pass"
    assert result.metrics["render_valid"]["image_count"] == 1
    result_data = json.loads(
        (working_dir / "validation_result.json").read_text(encoding="utf-8")
    )
    assert result_data["evidence"]["render_valid"]["image_paths"] == [str(image_path)]


def test_render_response_policy_with_in_memory_image_writes_artifacts(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    working_dir = tmp_path / "run"
    request = create_draft_validation_request(
        task_description="Check render response evidence.",
        inputs=(image_path,),
        working_dir=working_dir,
        requested_templates=("render_valid",),
        policy={
            "render_response": {
                "results": [
                    {
                        "camera": "/CameraA",
                        "frames": [0],
                        "images": [_valid_image()],
                    }
                ]
            },
            "expected_cameras": ["/CameraA"],
            "expected_frames": [0],
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "pass"
    result_data = json.loads(
        (working_dir / "validation_result.json").read_text(encoding="utf-8")
    )
    policy_image = result_data["request"]["policy"]["render_response"]["results"][0][
        "images"
    ][0]
    assert "PIL.Image.Image" in policy_image
    assert result_data["metrics"]["render_valid"]["render_response_present"]


def test_physics_sane_template_uses_real_adapter_for_resolved_usd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def fake_physics_sane_adapter(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append({"args": args, "kwargs": kwargs})
        return {
            "template": "physics_sane",
            "status": "completed",
            "verdict": "warn",
            "passed": True,
            "issues": [
                {
                    "code": "physics.mass_scale_suspicious",
                    "severity": "warn",
                    "message": "Mass scale looks high.",
                    "subject": str(usd_path),
                    "details": {"mass": 900.0},
                }
            ],
            "metrics": {"opened": True, "physics_expected": True},
            "evidence": {"usd_path": str(usd_path)},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.run_physics_sane_adapter",
        fake_physics_sane_adapter,
    )
    request = create_draft_validation_request(
        task_description="Validate collision physics.",
        inputs=(usd_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physics_sane",),
        policy={"expect_physics": True},
    )

    result = run_validation_scaffold(request)

    assert calls[0]["args"] == (usd_path.resolve(),)
    assert calls[0]["kwargs"]["task_description"] == "Validate collision physics."
    assert calls[0]["kwargs"]["policy"] == {"expect_physics": True}
    assert result.verdict == "warn"
    assert result.template_results[0].status == "warn"
    assert result.issues[0].code == "physics.mass_scale_suspicious"
    assert result.metrics["physics_sane"]["usd_path_count"] == 1


def test_requested_physics_sane_without_usd_skips_with_issue(tmp_path: Path) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    working_dir = tmp_path / "run"
    request = create_draft_validation_request(
        task_description="Force physics check without USD.",
        inputs=(image_path,),
        working_dir=working_dir,
        requested_templates=("physics_sane",),
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "warn"
    assert result.template_results[0].status == "skipped"
    assert result.issues[0].code == "physics_sane.evidence_missing"
    result_data = json.loads(
        (working_dir / "validation_result.json").read_text(encoding="utf-8")
    )
    assert result_data["evidence"]["physics_sane"]["usd_paths"] == []


def test_behavior_video_selects_behavior_path_not_look_right(tmp_path: Path) -> None:
    usd_path = tmp_path / "asset.usda"
    video_path = tmp_path / "behavior.mp4"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    video_path.write_bytes(b"not decoded by scaffold")
    working_dir = tmp_path / "run"
    request = create_draft_validation_request(
        task_description="Validate that the asset rolls smoothly and stays upright.",
        inputs=(usd_path, video_path),
        working_dir=working_dir,
    )

    result = run_validation_scaffold(request)

    assert [template.template_name for template in result.template_results] == [
        "render_valid",
        "physical_behavior",
    ]
    result_data = json.loads(
        (working_dir / "validation_result.json").read_text(encoding="utf-8")
    )
    assert result_data["metrics"]["physical_behavior"]["video_path_count"] == 1


def test_video_only_input_selects_behavior_only(tmp_path: Path) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    working_dir = tmp_path / "run"
    request = create_draft_validation_request(
        task_description="Validate rollout behavior.",
        inputs=(video_path,),
        working_dir=working_dir,
    )

    result = run_validation_scaffold(request)

    assert [template.template_name for template in result.template_results] == [
        "physical_behavior"
    ]
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "warn"
    assert physical_behavior.metrics["behavior_evidence_required"] is False


def test_policy_refine_output_selects_behavior_template(tmp_path: Path) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    refine_dir = _write_physical_behavior_refine_output(
        tmp_path / "refine",
        termination_reason="approved",
        judge_decision="approve",
        judge_score=0.9,
    )
    request = create_draft_validation_request(
        task_description="Validate rollout behavior from policy artifacts.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        policy={"physical_behavior_refine_output_dir": str(refine_dir)},
    )

    result = run_validation_scaffold(request)

    assert [template.template_name for template in result.template_results] == [
        "render_valid",
        "physical_behavior",
    ]
    assert result.template_results[1].status == "passed"


def test_policy_refine_output_file_is_treated_as_summary_path(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    refine_dir = _write_physical_behavior_refine_output(
        tmp_path / "refine",
        termination_reason="approved",
        judge_decision="approve",
        judge_score=0.9,
    )
    request = create_draft_validation_request(
        task_description="Validate rollout behavior from policy artifacts.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={
            "physical_behavior_refine_output_dir": str(
                refine_dir / "refine_summary.json"
            )
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "pass"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "passed"
    assert physical_behavior.metrics["refine_summary_count"] == 1


def test_refine_summary_path_with_judge_payload_keeps_execution_status(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    judge_path = tmp_path / "judge_result.json"
    judge_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "decision": "continue",
                "score": 0.31,
                "reasoning": "needs another scenario refinement",
            }
        ),
        encoding="utf-8",
    )
    request = create_draft_validation_request(
        task_description="Validate rollout behavior from an explicit judge result.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={"refine_summary_path": str(judge_path)},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "needs_refinement"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "needs_refinement"
    assert physical_behavior.metrics["behavior_summary_kind"] == "judge_result"
    assert physical_behavior.metrics["execution_status"] == "completed"
    assert "termination_reason" not in physical_behavior.metrics


def test_sampled_frames_preserve_visual_template_selection(tmp_path: Path) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    frame_path = _write_valid_image(tmp_path / "frame_0001.png")
    request = create_draft_validation_request(
        task_description="Validate visual evidence with sampled rollout frames.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        dry_run=True,
        policy={"sampled_video_frame_paths": [str(frame_path)]},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "planned"
    assert [step.template_name for step in result.plan.steps] == [
        "render_valid",
        "look_right",
    ]


def test_physical_behavior_refine_summary_approved_passes(tmp_path: Path) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    refine_dir = _write_physical_behavior_refine_output(
        tmp_path / "refine",
        termination_reason="approved",
        judge_decision="approve",
        judge_score=0.93,
    )
    working_dir = tmp_path / "run"
    request = create_draft_validation_request(
        task_description="Validate that the scaffold rolls without tipping.",
        inputs=(video_path,),
        working_dir=working_dir,
        requested_templates=("physical_behavior",),
        policy={"physical_behavior_refine_output_dir": str(refine_dir)},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "pass"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "passed"
    assert physical_behavior.metrics["termination_reason"] == "approved"
    assert physical_behavior.metrics["judge_decision"] == "approve"
    assert physical_behavior.metrics["video_path_count"] == 2
    assert physical_behavior.metrics["sampled_frame_path_count"] == 1
    assert physical_behavior.metrics["trajectory_metrics_count"] == 1
    assert physical_behavior.evidence["behavior_summary"]["judge_score"] == 0.93


def test_physical_behavior_prefers_loop_summary_over_iter_judge(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    refine_dir = _write_physical_behavior_refine_output(
        tmp_path / "refine",
        termination_reason="approved",
        judge_decision="approve",
        judge_score=0.91,
    )
    (refine_dir / "iter_1" / "judge_result.json").write_text(
        json.dumps(
            {
                "decision": "continue",
                "score": 0.35,
                "reasoning": "first attempt needs refinement",
                "iterations": 1,
            }
        ),
        encoding="utf-8",
    )
    request = create_draft_validation_request(
        task_description="Validate rollout behavior.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={"physical_behavior_refine_output_dir": str(refine_dir)},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "pass"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "passed"
    assert physical_behavior.metrics["termination_reason"] == "approved"
    assert physical_behavior.metrics["judge_decision"] == "approve"


def test_physical_behavior_preserves_rollout_video_when_frames_are_capped(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    refine_dir = _write_physical_behavior_refine_output(
        tmp_path / "refine",
        termination_reason="approved",
        judge_decision="approve",
        judge_score=0.91,
    )
    render_dir = refine_dir / "iter_1" / "render"
    for index in range(MAX_BEHAVIOR_RENDER_EVIDENCE_FILES + 10):
        _write_valid_image(render_dir / f"frame_{index:04d}.png")
    request = create_draft_validation_request(
        task_description="Validate rollout behavior.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={"physical_behavior_refine_output_dir": str(refine_dir)},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "pass"
    physical_behavior = result.template_results[0]
    assert physical_behavior.metrics["video_path_count"] == 2
    # The fixture contributes one rendered rollout video; the remaining
    # per-directory cap is available for sampled frames.
    assert (
        physical_behavior.metrics["sampled_frame_path_count"]
        == MAX_BEHAVIOR_RENDER_EVIDENCE_FILES - 1
    )


def test_physical_behavior_refine_summary_continue_needs_refinement(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    refine_dir = _write_physical_behavior_refine_output(
        tmp_path / "refine",
        termination_reason="max_iterations",
        judge_decision="continue",
        judge_score=0.42,
    )
    request = create_draft_validation_request(
        task_description="Validate that the scaffold rolls without tipping.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={"physical_behavior_refine_output_dir": str(refine_dir)},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "needs_refinement"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "needs_refinement"
    assert physical_behavior.issues[0].code == "physics.behavior_needs_refinement"
    assert physical_behavior.metrics["termination_reason"] == "max_iterations"
    assert physical_behavior.metrics["judge_decision"] == "continue"


def test_physical_behavior_required_continue_summary_fails(tmp_path: Path) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    refine_dir = _write_physical_behavior_refine_output(
        tmp_path / "refine",
        termination_reason="max_iterations",
        judge_decision="continue",
        judge_score=0.3,
    )
    request = create_draft_validation_request(
        task_description="Reject a rollout that tips over.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={
            "behavior_evidence_required": True,
            "physical_behavior_refine_output_dir": str(refine_dir),
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "fail"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "failed"
    assert physical_behavior.issues[0].code == "physics.behavior_needs_refinement"
    assert physical_behavior.issues[0].severity == "fail"
    assert physical_behavior.metrics["termination_reason"] == "max_iterations"
    assert physical_behavior.metrics["judge_decision"] == "continue"


def test_physical_behavior_required_rejected_summary_fails(tmp_path: Path) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    refine_dir = _write_physical_behavior_refine_output(
        tmp_path / "refine",
        termination_reason="max_iterations_reached",
        judge_decision="reject",
        judge_score=0.3,
    )
    request = create_draft_validation_request(
        task_description="Reject a rollout that tips over.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={
            "behavior_evidence_required": True,
            "physical_behavior_refine_output_dir": str(refine_dir),
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "fail"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "failed"
    assert physical_behavior.issues[0].code == "physics.behavior_rejected"
    assert physical_behavior.issues[0].severity == "fail"
    assert physical_behavior.metrics["termination_reason"] == "max_iterations_reached"
    assert physical_behavior.metrics["judge_decision"] == "reject"


def test_physical_behavior_completed_judge_continue_needs_refinement(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    iter_dir = tmp_path / "refine" / "iter_1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "tune_results.json").write_text(
        json.dumps(
            {
                "judge": {
                    "status": "completed",
                    "decision": "continue",
                    "score": 0.31,
                    "reasoning": "needs another scenario refinement",
                }
            }
        ),
        encoding="utf-8",
    )
    request = create_draft_validation_request(
        task_description="Validate rollout behavior.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={"physical_behavior_refine_output_dir": str(tmp_path / "refine")},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "needs_refinement"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "needs_refinement"
    assert physical_behavior.issues[0].code == "physics.behavior_needs_refinement"
    assert physical_behavior.metrics["execution_status"] == "completed"
    assert "termination_reason" not in physical_behavior.metrics
    assert physical_behavior.metrics["judge_decision"] == "continue"


def test_physical_behavior_without_loop_summary_uses_latest_iter_judge(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    refine_dir = tmp_path / "refine"
    for iteration, decision, score in (
        (1, "continue", 0.31),
        (2, "approve", 0.88),
    ):
        iter_dir = refine_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True)
        (iter_dir / "judge_result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "decision": decision,
                    "score": score,
                    "reasoning": f"iteration {iteration}",
                }
            ),
            encoding="utf-8",
        )
    request = create_draft_validation_request(
        task_description="Validate rollout behavior.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={"physical_behavior_refine_output_dir": str(refine_dir)},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "pass"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "passed"
    assert physical_behavior.metrics["iteration"] == 2
    assert physical_behavior.metrics["judge_decision"] == "approve"


def test_physical_behavior_completed_judge_without_decision_warns(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    iter_dir = tmp_path / "refine" / "iter_1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "tune_results.json").write_text(
        json.dumps({"judge": {"status": "completed", "score": 0.31}}),
        encoding="utf-8",
    )
    request = create_draft_validation_request(
        task_description="Validate rollout behavior.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={"physical_behavior_refine_output_dir": str(tmp_path / "refine")},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "warn"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "warn"
    assert physical_behavior.metrics["execution_status"] == "completed"
    assert "termination_reason" not in physical_behavior.metrics
    assert "judge_decision" not in physical_behavior.metrics


def test_physical_behavior_degraded_judge_continue_warns(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    iter_dir = tmp_path / "refine" / "iter_1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "tune_results.json").write_text(
        json.dumps(
            {
                "judge": {
                    "status": "degraded",
                    "decision": "continue",
                    "score": 0.31,
                }
            }
        ),
        encoding="utf-8",
    )
    request = create_draft_validation_request(
        task_description="Validate rollout behavior.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={"physical_behavior_refine_output_dir": str(tmp_path / "refine")},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "warn"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "warn"
    assert physical_behavior.metrics["execution_status"] == "degraded"
    assert "termination_reason" not in physical_behavior.metrics
    assert physical_behavior.metrics["judge_decision"] == "continue"


def test_physical_behavior_refine_summary_error_fails(tmp_path: Path) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    refine_dir = _write_physical_behavior_refine_output(
        tmp_path / "refine",
        termination_reason="error",
        judge_decision="continue",
        judge_score=0.1,
    )
    request = create_draft_validation_request(
        task_description="Validate rollout behavior.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={"physical_behavior_refine_output_dir": str(refine_dir)},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "fail"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "failed"
    assert physical_behavior.issues[0].code == "physics.behavior_refine_loop_failed"


def test_physical_behavior_required_missing_evidence_fails(tmp_path: Path) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    request = create_draft_validation_request(
        task_description="Force behavior validation without behavior evidence.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={"behavior_evidence_required": True},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "fail"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "failed"
    assert physical_behavior.issues[0].code == "physics.behavior_evidence_missing"
    assert physical_behavior.metrics["behavior_evidence_required"] is True


def test_physical_behavior_global_required_fails_without_available_evidence(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    missing_video = tmp_path / "missing_behavior.mp4"
    request = create_draft_validation_request(
        task_description="Force behavior validation with optional missing item.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={
            "behavior_evidence_required": True,
            "physical_behavior_evidence": [
                {
                    "path": str(missing_video),
                    "kind": "video",
                    "required": False,
                }
            ],
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "fail"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "failed"
    assert any(
        issue.code == "physics.behavior_evidence_missing" and issue.severity == "fail"
        for issue in physical_behavior.issues
    )


def test_physical_behavior_required_empty_refine_summary_fails(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    refine_dir = tmp_path / "refine"
    refine_dir.mkdir()
    (refine_dir / "refine_summary.json").write_text("{}", encoding="utf-8")
    request = create_draft_validation_request(
        task_description="Require behavior validation from refine output.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={
            "behavior_evidence_required": True,
            "physical_behavior_refine_output_dir": str(refine_dir),
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "fail"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "failed"
    assert physical_behavior.metrics["refine_summary_count"] == 0
    assert physical_behavior.issues[0].code == "physics.behavior_judge_unavailable"


def test_physical_behavior_required_empty_tune_judge_fails(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    iter_dir = tmp_path / "refine" / "iter_1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "tune_results.json").write_text(
        json.dumps({"judge": {}}),
        encoding="utf-8",
    )
    request = create_draft_validation_request(
        task_description="Require behavior validation from tune output.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={
            "behavior_evidence_required": True,
            "physical_behavior_refine_output_dir": str(tmp_path / "refine"),
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "fail"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "failed"
    assert physical_behavior.metrics["refine_summary_count"] == 0
    assert physical_behavior.issues[0].code == "physics.behavior_judge_unavailable"


def test_physical_behavior_required_empty_judge_result_fails(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    iter_dir = tmp_path / "refine" / "iter_1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "judge_result.json").write_text("{}", encoding="utf-8")
    request = create_draft_validation_request(
        task_description="Require behavior validation from judge output.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={
            "behavior_evidence_required": True,
            "physical_behavior_refine_output_dir": str(tmp_path / "refine"),
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "fail"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "failed"
    assert physical_behavior.metrics["refine_summary_count"] == 0
    assert physical_behavior.issues[0].code == "physics.behavior_judge_unavailable"


def test_physical_behavior_required_error_only_judge_result_fails_as_judge(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    iter_dir = tmp_path / "refine" / "iter_1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "judge_result.json").write_text(
        json.dumps({"status": "error", "error": "timeout"}),
        encoding="utf-8",
    )
    request = create_draft_validation_request(
        task_description="Require behavior validation from judge output.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={
            "behavior_evidence_required": True,
            "physical_behavior_refine_output_dir": str(tmp_path / "refine"),
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "fail"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "failed"
    assert physical_behavior.metrics["refine_summary_count"] == 1
    assert physical_behavior.metrics["execution_status"] == "error"
    assert physical_behavior.metrics["error"] == "timeout"
    assert physical_behavior.issues[0].code == "physics.behavior_refine_loop_failed"


def test_physical_behavior_required_item_without_summary_fails(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    request = create_draft_validation_request(
        task_description="Require behavior evidence and judge output.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={
            "physical_behavior_evidence": [
                {
                    "path": str(video_path),
                    "kind": "video",
                    "required": True,
                }
            ]
        },
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "fail"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "failed"
    assert physical_behavior.issues[0].code == "physics.behavior_judge_unavailable"
    assert physical_behavior.issues[0].details["required_evidence_count"] == 1


def test_physical_behavior_approved_with_unavailable_judge_warns(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    refine_dir = _write_physical_behavior_refine_output(
        tmp_path / "refine",
        termination_reason="approved",
        judge_decision="approve",
        judge_score=0.8,
        judge_llm_unavailable=True,
    )
    request = create_draft_validation_request(
        task_description="Validate rollout behavior.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={"physical_behavior_refine_output_dir": str(refine_dir)},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "warn"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "warn"
    assert {issue.code for issue in physical_behavior.issues} == {
        "physics.behavior_judge_unavailable"
    }
    assert physical_behavior.metrics["judge_llm_unavailable_count"] == 1


def test_physical_behavior_approved_with_unavailable_refiner_warns(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    refine_dir = _write_physical_behavior_refine_output(
        tmp_path / "refine",
        termination_reason="approved",
        judge_decision="approve",
        judge_score=0.8,
        refine_llm_unavailable=True,
    )
    request = create_draft_validation_request(
        task_description="Validate rollout behavior.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={"physical_behavior_refine_output_dir": str(refine_dir)},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "warn"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "warn"
    assert {issue.code for issue in physical_behavior.issues} == {
        "physics.behavior_refiner_unavailable"
    }
    assert physical_behavior.metrics["refine_llm_unavailable_count"] == 1


def test_physical_behavior_refine_summary_non_sequence_iterations(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    refine_dir = tmp_path / "refine"
    refine_dir.mkdir()
    (refine_dir / "refine_summary.json").write_text(
        json.dumps(
            {
                "termination_reason": "approved",
                "iterations": 42,
            }
        ),
        encoding="utf-8",
    )
    request = create_draft_validation_request(
        task_description="Validate rollout behavior.",
        inputs=(video_path,),
        working_dir=tmp_path / "run",
        requested_templates=("physical_behavior",),
        policy={"physical_behavior_refine_output_dir": str(refine_dir)},
    )

    result = run_validation_scaffold(request)

    assert result.verdict == "pass"
    physical_behavior = result.template_results[0]
    assert physical_behavior.status == "passed"
    assert physical_behavior.metrics["iteration_count"] == 0


def test_unknown_requested_template_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "render.png"
    image_path.write_bytes(b"not decoded by scaffold")
    request = create_draft_validation_request(
        task_description="Check render evidence.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("bogus",),
    )

    with pytest.raises(DraftValidationError, match="Unknown validation template"):
        run_validation_scaffold(request)


def test_unregistered_requested_template_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "render.png"
    image_path.write_bytes(b"not decoded by scaffold")

    class RenderValidOnly:
        name = "render_valid"

        def run(self, context: DraftValidationContext) -> DraftTemplateResult:
            return DraftTemplateResult(template_name=self.name, status="passed")

    request = create_draft_validation_request(
        task_description="Check render evidence.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("look_right",),
    )

    with pytest.raises(DraftValidationError, match="not registered"):
        run_validation_scaffold(
            request,
            registry=TemplateRegistry((RenderValidOnly(),)),
        )


def _write_physical_behavior_refine_output(
    output_dir: Path,
    *,
    termination_reason: str,
    judge_decision: str,
    judge_score: float,
    judge_llm_unavailable: bool = False,
    refine_llm_unavailable: bool = False,
) -> Path:
    iter_dir = output_dir / "iter_1"
    render_dir = iter_dir / "render"
    render_dir.mkdir(parents=True)
    (iter_dir / "scenario.yaml").write_text("name: drop_settle\n", encoding="utf-8")
    (iter_dir / "history.jsonl").write_text(
        json.dumps(
            {
                "trial_index": 0,
                "score": 0.1,
                "params": {"restitution": 0.7},
                "failed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (iter_dir / "judge_result.json").write_text(
        json.dumps(
            {
                "decision": judge_decision,
                "score": judge_score,
                "reasoning": "fixture judge result",
                "iterations": 1,
                "llm_unavailable": judge_llm_unavailable,
            }
        ),
        encoding="utf-8",
    )
    (iter_dir / "refine_result.json").write_text(
        json.dumps(
            {
                "llm_unavailable": refine_llm_unavailable,
                "reasoning": "fixture refine result",
            }
        ),
        encoding="utf-8",
    )
    (render_dir / "render.mp4").write_bytes(b"not decoded by scaffold")
    _write_valid_image(render_dir / "frame_0001.png")
    (output_dir / "refine_summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "termination_reason": termination_reason,
                "final_iteration": 1,
                "final_dir": str(output_dir / "final"),
                "user_prompt": "validate rollout behavior",
                "iterations": [
                    {
                        "iteration": 1,
                        "iteration_dir": str(iter_dir),
                        "scenario_yaml_path": str(iter_dir / "scenario.yaml"),
                        "tune_output_dir": str(iter_dir),
                        "best_params": {"restitution": 0.7},
                        "best_score": 0.1,
                        "n_trials": 1,
                        "judge_decision": judge_decision,
                        "judge_score": judge_score,
                        "judge_reasoning": "fixture judge result",
                        "judge_llm_unavailable": judge_llm_unavailable,
                        "refine_llm_unavailable": refine_llm_unavailable,
                        "refine_reasoning": "fixture refine result",
                        "metric_name": "settle_distance",
                        "metric_value": 0.1,
                        "cancelled": False,
                        "error": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return output_dir


def test_template_registry_rejects_unknown_names() -> None:
    class UnknownTemplate:
        name = "bogus"

        def run(self, context: DraftValidationContext) -> DraftTemplateResult:
            raise AssertionError("should not run")

    with pytest.raises(DraftValidationError, match="Unknown validation template"):
        TemplateRegistry((UnknownTemplate(),))


def test_template_failures_become_fail_verdict(tmp_path: Path) -> None:
    image_path = tmp_path / "render.png"
    image_path.write_bytes(b"not decoded by scaffold")

    class ExplodingRenderValid:
        name = "render_valid"

        def run(self, context: DraftValidationContext) -> DraftTemplateResult:
            raise RuntimeError("boom")

    request = create_draft_validation_request(
        task_description="Check render evidence.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid",),
    )

    result = run_validation_scaffold(
        request,
        registry=TemplateRegistry((ExplodingRenderValid(),)),
    )

    assert result.verdict == "fail"
    assert result.template_results[0].status == "error"
    assert result.issues[0].code == "agent.template_error"
