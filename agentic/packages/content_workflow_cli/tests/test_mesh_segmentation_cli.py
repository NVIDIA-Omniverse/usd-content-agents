# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fresh-session mesh-segmentation launcher."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from content_agent_workflows.common import memory as memory_module
from content_agent_workflows.common import memory_cli
from content_agent_workflows.common.memory import (
    AgentMemory,
    MemoryInteraction,
    MemoryOutcome,
    MemorySearchQuery,
    RememberRequest,
)
from content_agent_workflows.mesh_segmentation_contract import (
    CONTINUATION_ARTIFACT,
    required_mesh_segmentation_artifacts,
)
from content_agent_workflows.mesh_segmentation_contract import (
    REQUIRED_FINAL_ARTIFACTS as CONTRACT_REQUIRED_FINAL_ARTIFACTS,
)
from content_agent_workflows.mesh_segmentation_contract import (
    REQUIRED_RECOGNITION_FINAL_ARTIFACTS as CONTRACT_REQUIRED_RECOGNITION_ARTIFACTS,
)
from content_agent_workflows.mesh_segmentation_contract import (
    REQUIRED_TARGETED_FINAL_ARTIFACTS as CONTRACT_REQUIRED_TARGETED_ARTIFACTS,
)
from PIL import Image

from content_workflow_cli import mesh_segmentation_runner
from content_workflow_cli.cli import build_parser, main
from content_workflow_cli.memory_broker import (
    MEMORY_ORIGIN_AGENT_TAG,
    MEMORY_ORIGIN_LAUNCHER_TAG,
    AgentMemoryBroker,
)
from content_workflow_cli.mesh_segmentation_runner import (
    CODEX_EXECUTION_CONTAINER,
    DEFAULT_MESH_SEGMENTATION_CASE_TIMEOUT_SECONDS,
    DEFAULT_MESH_SEGMENTATION_CHILD_TIMEOUT_SECONDS,
    REQUIRED_FINAL_ARTIFACTS,
    REQUIRED_RECOGNITION_FINAL_ARTIFACTS,
    REQUIRED_TARGETED_FINAL_ARTIFACTS,
    MeshSegmentationConfig,
    _build_codex_container_command,
    _make_recognition_sequential_watchdog,
    _validate_terminal_artifacts,
    run_mesh_segmentation,
    validate_recognition_part_locks,
)
from content_workflow_cli.mesh_segmentation_vqa import (
    _resolve_run_artifact,
    validate_targeted_selection_evidence,
)
from content_workflow_cli.runner import _child_workflow_name, _codex_bridge_env
from content_workflow_cli.trace import UnsafeRunArtifactError

PATCH_ROLES = (
    "front",
    "back",
    "front_oblique",
    "back_oblique",
    "upper_grazing",
    "lower_grazing",
)
CANONICAL_ROLES = (
    "plus_x",
    "minus_x",
    "plus_xminus_yplus_z",
    "minus_xminus_yplus_z",
    "plus_xminus_yminus_z",
    "minus_xminus_yminus_z",
)
CANONICAL_PROVIDER_CLI_ARGS = (
    "--codex-responses-url",
    "https://provider.example/v1/responses",
    "--codex-api-key-env",
    "TEST_PROVIDER_API_KEY",
    "--image-gen-backend",
    "openai_compatible",
    "--image-gen-model",
    "provider/image-edit-model",
    "--image-gen-base-url",
    "https://provider.example/v1",
    "--image-gen-api-key-env",
    "TEST_PROVIDER_API_KEY",
)
CANONICAL_PROVIDER_CONFIG = {
    "codex_responses_url": "https://provider.example/v1/responses",
    "codex_api_key_env": "TEST_PROVIDER_API_KEY",
    "image_gen_backend": "openai_compatible",
    "image_gen_model": "provider/image-edit-model",
    "image_gen_base_url": "https://provider.example/v1",
    "image_gen_api_key_env": "TEST_PROVIDER_API_KEY",
}


def test_mesh_segmentation_runner_uses_shared_artifact_contract() -> None:
    assert REQUIRED_FINAL_ARTIFACTS is CONTRACT_REQUIRED_FINAL_ARTIFACTS
    assert (
        REQUIRED_RECOGNITION_FINAL_ARTIFACTS is CONTRACT_REQUIRED_RECOGNITION_ARTIFACTS
    )
    assert REQUIRED_TARGETED_FINAL_ARTIFACTS is CONTRACT_REQUIRED_TARGETED_ARTIFACTS
    assert required_mesh_segmentation_artifacts("targeted", resumable=True) == (
        *REQUIRED_FINAL_ARTIFACTS,
        *REQUIRED_TARGETED_FINAL_ARTIFACTS,
        CONTINUATION_ARTIFACT,
    )
    with pytest.raises(ValueError, match="unsupported mesh-segmentation workflow mode"):
        required_mesh_segmentation_artifacts("unsupported")  # type: ignore[arg-type]


def test_small_mesh_defaults_allow_the_acceptance_fixture_to_finish(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The 36-face fixture must not be killed by the former one-hour turn."""

    monkeypatch.delenv("CONTENT_AGENT_MESH_SEGMENTATION_TIMEOUT", raising=False)
    args = build_parser().parse_args(
        ["mesh-segmentation", "run", "--asset", "fused_cart.usda"]
    )
    config = MeshSegmentationConfig(
        repo_root=Path.cwd(),
        asset_path=Path("fused_cart.usda"),
        reference_images=[],
    )

    assert DEFAULT_MESH_SEGMENTATION_CHILD_TIMEOUT_SECONDS == 7200.0
    assert DEFAULT_MESH_SEGMENTATION_CASE_TIMEOUT_SECONDS == 21600.0
    assert args.child_timeout == config.child_timeout_seconds == 7200.0
    assert args.case_timeout == config.case_timeout_seconds == 21600.0

    with pytest.raises(SystemExit) as help_exit:
        build_parser().parse_args(["mesh-segmentation", "run", "--help"])
    assert help_exit.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "default 7200-second turn budget" in help_text
    assert "default bounds the whole case to 21600 seconds" in help_text


def test_parent_progress_advances_from_running_through_real_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tracker = mesh_segmentation_runner.MeshSegmentationProgressTracker(
        run_dir,
        mesh_segmentation_runner.TraceWriter(run_dir),
    )

    tracker.start()
    (run_dir / "part_plan.json").write_text("{}\n", encoding="utf-8")
    tracker.poll()
    prepare = run_dir / "prepare"
    prepare.mkdir()
    (prepare / "topology.json").write_text("{}\n", encoding="utf-8")
    tracker.poll()
    part = run_dir / "Body"
    part.mkdir()
    (part / "initializer_decision.json").write_text("{}\n", encoding="utf-8")
    tracker.poll()
    (part / "part_completion.json").write_text("{}\n", encoding="utf-8")
    tracker.poll()
    final = run_dir / "final"
    final.mkdir()
    (final / "report.md").write_text("done\n", encoding="utf-8")
    tracker.poll()
    tracker.poll()

    events = [
        json.loads(line)
        for line in (run_dir / "trace" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    progress_events = [
        event for event in events if event["event_type"] == "mesh_segmentation_progress"
    ]
    assert [event["data"]["stage"] for event in progress_events] == [
        "child_running",
        "planning",
        "preparing",
        "segmenting",
        "validating",
        "finalizing",
    ]
    assert all("%" not in event["summary"] for event in progress_events)


@pytest.mark.skipif(os.name == "nt", reason="directory symlinks require privileges")
def test_parent_progress_accepts_a_symlinked_run_directory(tmp_path: Path) -> None:
    real_run_dir = tmp_path / "real-run"
    real_run_dir.mkdir()
    (real_run_dir / "part_plan.json").write_text("{}\n", encoding="utf-8")
    part_dir = real_run_dir / "Body"
    part_dir.mkdir()
    (part_dir / "initializer_decision.json").write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "run-alias"
    run_dir.symlink_to(real_run_dir, target_is_directory=True)

    assert mesh_segmentation_runner._regular_progress_artifact(
        run_dir, "part_plan.json"
    ) == (run_dir / "part_plan.json")
    stage, artifacts = mesh_segmentation_runner._observe_mesh_segmentation_progress(
        run_dir
    )
    assert stage == "segmenting"
    assert artifacts == [run_dir / "Body" / "initializer_decision.json"]


def test_mesh_segmentation_allows_explicit_configured_codex_auth(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    config = MeshSegmentationConfig(
        repo_root=paths["repo_root"],
        asset_path=paths["asset"],
        reference_images=[paths["references"] / "view_A.png"],
        allow_codex_configured_auth=True,
    )

    mesh_segmentation_runner._validate_config(config)


def test_targeted_mesh_segmentation_does_not_require_reference_images(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    config = MeshSegmentationConfig(
        repo_root=paths["repo_root"],
        asset_path=paths["asset"],
        reference_images=[],
        target_semantic_parts=["body", "wheel"],
        allow_codex_configured_auth=True,
    )

    mesh_segmentation_runner._validate_config(config)


@pytest.mark.parametrize(
    ("override", "expected_flag"),
    [
        ({"codex_base_url": "https://provider.example/v1"}, "--codex-base-url"),
        (
            {"codex_responses_url": "https://provider.example/v1/responses"},
            "--codex-responses-url",
        ),
        ({"codex_api_key_env": "TEST_PROVIDER_API_KEY"}, "--codex-api-key-env"),
        (
            {"codex_config": {"model_provider": "proxy"}},
            "--codex-config-json/--codex-config-file",
        ),
    ],
)
def test_mesh_segmentation_rejects_configured_auth_overrides(
    tmp_path: Path,
    override: dict[str, object],
    expected_flag: str,
) -> None:
    config = MeshSegmentationConfig(
        repo_root=tmp_path,
        asset_path=tmp_path / "asset.usdc",
        reference_images=[],
        allow_codex_configured_auth=True,
        **override,
    )

    with pytest.raises(ValueError, match="cannot be combined") as exc_info:
        mesh_segmentation_runner._validate_config(config)
    assert expected_flag in str(exc_info.value)


@pytest.fixture(autouse=True)
def _workflow_usd_cli_session(monkeypatch: pytest.MonkeyPatch) -> None:
    def start_parent_usd_cli_capability(**kwargs: object) -> SimpleNamespace:
        run_dir = Path(kwargs["run_dir"])
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        identity_path = raw_dir / "parent_usd_cli_session_test.json"
        identity_path.write_text("{}\n", encoding="utf-8")
        readiness_path = raw_dir / "ovrtx_probe.json"
        readiness_path.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            identity_path=identity_path,
            identity_sha256="a" * 64,
            server_url="http://127.0.0.1:43210",
            session=SimpleNamespace(session_id="mesh-workflow-usd-cli-session"),
            readiness=SimpleNamespace(artifact_path=readiness_path),
        )

    monkeypatch.setattr(
        mesh_segmentation_runner,
        "start_parent_usd_cli_capability",
        start_parent_usd_cli_capability,
    )
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "stop_parent_usd_cli_capability",
        lambda _capability: None,
    )


def test_mesh_segmentation_allows_companion_image_generation_without_endpoint(
    tmp_path: Path,
) -> None:
    asset_path = tmp_path / "asset.usdc"
    asset_path.write_bytes(b"usd")
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"png")
    skill_path = (
        tmp_path
        / "agentic"
        / ".agents"
        / "skills"
        / "content-workflow-mesh-segmentation"
        / "SKILL.md"
    )
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\nname: content-workflow-mesh-segmentation\n---\n",
        encoding="utf-8",
    )
    config = MeshSegmentationConfig(
        repo_root=tmp_path,
        asset_path=asset_path,
        reference_images=[reference_path],
        codex_responses_url="https://provider.example/v1/responses",
        codex_api_key_env="TEST_PROVIDER_API_KEY",
    )
    mesh_segmentation_runner._validate_config(config)


def test_mesh_segmentation_routes_shared_child_runtime(tmp_path: Path) -> None:
    config = MeshSegmentationConfig(
        repo_root=tmp_path,
        asset_path=tmp_path / "asset.usdc",
        reference_images=[],
    )

    assert _child_workflow_name(config) == "mesh-segmentation.run"


def test_mesh_segmentation_sdk_request_rejects_child_writable_helper_trust(
    tmp_path: Path,
) -> None:
    scripts_dir = tmp_path / "run" / "scripts"
    scripts_dir.mkdir(parents=True)
    helper = scripts_dir / "prepare_mesh.py"
    helper.write_text("print('prepare')\n", encoding="utf-8")
    config = MeshSegmentationConfig(
        repo_root=tmp_path,
        asset_path=tmp_path / "asset.usdc",
        reference_images=[],
        agent_cwd=tmp_path / "run",
        target_semantic_parts=["wheel"],
    )

    request = mesh_segmentation_runner._build_codex_sdk_request(
        config=config,
        prompt="segment",
        run_dir=tmp_path / "run",
        child_final_path=tmp_path / "run" / "child-final.md",
    )

    assert request["trusted_script_digests"] == {}
    assert "terminal_promotion_guard" not in request


def test_mesh_segmentation_external_image_endpoint_requires_provider_pair(
    tmp_path: Path,
) -> None:
    config = MeshSegmentationConfig(
        repo_root=tmp_path,
        asset_path=tmp_path / "asset.usdc",
        reference_images=[],
        codex_responses_url="https://provider.example/v1/responses",
        codex_api_key_env="TEST_PROVIDER_API_KEY",
        image_gen_base_url="https://provider.example/v1",
    )
    with pytest.raises(ValueError, match="require --image-gen-backend"):
        mesh_segmentation_runner._validate_config(config)


def test_mesh_segmentation_rejects_unknown_image_backend(tmp_path: Path) -> None:
    config = MeshSegmentationConfig(
        repo_root=tmp_path,
        asset_path=tmp_path / "asset.usdc",
        reference_images=[],
        codex_responses_url="https://provider.example/v1/responses",
        codex_api_key_env="TEST_PROVIDER_API_KEY",
        image_gen_backend="gemni",
    )

    with pytest.raises(ValueError, match="Unsupported --image-gen-backend"):
        mesh_segmentation_runner._validate_config(config)


def test_mesh_segmentation_rejects_a_negative_case_timeout(tmp_path: Path) -> None:
    """`--case-timeout -1` must not read as "disable the cap".

    The loop clamps the budget with `max(0.0, ...)`, so a mistyped negative
    became the documented zero and quietly expanded a bounded case to
    iteration_budget x child_timeout. Only an explicit zero disables it.
    """

    paths = _write_inputs(tmp_path)
    base = {
        "repo_root": paths["repo_root"],
        "asset_path": paths["asset"],
        "reference_images": [paths["references"] / "view_A.png"],
        **CANONICAL_PROVIDER_CONFIG,
    }

    with pytest.raises(ValueError, match="--case-timeout must be zero"):
        mesh_segmentation_runner._validate_config(
            MeshSegmentationConfig(**base, case_timeout_seconds=-1.0)
        )

    # Zero still disables the cap, as the flag help promises.
    mesh_segmentation_runner._validate_config(
        MeshSegmentationConfig(**base, case_timeout_seconds=0.0)
    )


def test_mesh_segmentation_rejects_semantic_part_paths(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    config = MeshSegmentationConfig(
        repo_root=paths["repo_root"],
        asset_path=paths["asset"],
        reference_images=[paths["references"] / "view_A.png"],
        target_semantic_parts=["wheels/tire"],
        **CANONICAL_PROVIDER_CONFIG,
    )

    with pytest.raises(ValueError, match="plain semantic names"):
        mesh_segmentation_runner._validate_config(config)


@pytest.mark.parametrize(
    "part_name",
    sorted(mesh_segmentation_runner.RESERVED_TARGET_SEMANTIC_PART_NAMES)
    + ["Final", "OTHER"],
)
def test_mesh_segmentation_rejects_reserved_semantic_part_names(
    tmp_path: Path,
    part_name: str,
) -> None:
    paths = _write_inputs(tmp_path)
    config = MeshSegmentationConfig(
        repo_root=paths["repo_root"],
        asset_path=paths["asset"],
        reference_images=[paths["references"] / "view_A.png"],
        target_semantic_parts=[part_name],
        **CANONICAL_PROVIDER_CONFIG,
    )

    with pytest.raises(ValueError, match="reserved run artifact names"):
        mesh_segmentation_runner._validate_config(config)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    repo_root = tmp_path / "repo"
    skill_root = (
        repo_root
        / "agentic"
        / ".agents"
        / "skills"
        / "content-workflow-mesh-segmentation"
    )
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\nname: content-workflow-mesh-segmentation\n---\n",
        encoding="utf-8",
    )
    (repo_root / ".env").write_text(
        "TEST_PROVIDER_API_KEY=test-provider-secret\n",
        encoding="utf-8",
    )
    usd_cli_skill = repo_root / "agentic" / ".agents" / "skills" / "usd-cli"
    usd_cli_skill.mkdir(parents=True)
    (usd_cli_skill / "SKILL.md").write_text(
        "---\nname: usd-cli\n---\n",
        encoding="utf-8",
    )

    asset = tmp_path / "Body.usdc"
    asset.write_bytes(b"fresh fused mesh")
    references = tmp_path / "references"
    references.mkdir()
    (references / "view_b.png").write_bytes(b"b")
    (references / "view_A.png").write_bytes(b"a")
    return {
        "repo_root": repo_root,
        "asset": asset,
        "references": references,
    }


def test_mesh_segmentation_default_run_directory_is_at_repo_root(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    config = MeshSegmentationConfig(
        repo_root=paths["repo_root"],
        asset_path=paths["asset"],
        reference_images=[paths["references"] / "view_A.png"],
        run_id="root-mesh",
        **CANONICAL_PROVIDER_CONFIG,
        dry_run=True,
    )

    run_id, run_dir = mesh_segmentation_runner._prepare_run_dir(config)

    assert run_id == "root-mesh"
    assert run_dir == paths["repo_root"] / "runs" / "root-mesh"


def _write_valid_completed_continuation_run(
    root: Path,
    *,
    asset: Path,
) -> Path:
    source_run = root / "completed-source-run"
    source_run.mkdir()
    asset_digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    (source_run / "terminal_validation.json").write_text(
        json.dumps({"valid": True}),
        encoding="utf-8",
    )
    final_validation = source_run / "validation" / "final_validation.json"
    final_validation.parent.mkdir()
    final_validation.write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    export_manifest = source_run / "final" / "export_manifest.json"
    export_manifest.parent.mkdir()
    export_manifest.write_text(
        json.dumps(
            {
                "source_sha256": asset_digest,
                "exact_source_face_coverage": True,
            }
        ),
        encoding="utf-8",
    )
    state_labels = source_run / "state" / "final_labels.u32le"
    state_labels.parent.mkdir()
    state_labels.write_bytes(
        b"\x01\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00"
    )
    fragment_labels = source_run / "fragments" / "fragment_ids.u32le"
    fragment_labels.parent.mkdir()
    fragment_labels.write_bytes(
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00"
    )
    fixed_artifacts = (
        "prepare/neutral.usdc",
        "prepare/topology.json",
        "prepare/all_faces_candidate.u32le",
        "prepare/degenerate_face_ids.u32le",
        "fragments/fragment_ids.npy",
        "fragments/fragment_adjacency.npy",
        "fragments/fragment_statistics.npz",
        "fragments/fragment_colors.npy",
        "fragments/fragments.usdc",
        "fragments/fragment_manifest.json",
    )
    for relative in fixed_artifacts:
        path = source_run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    (source_run / "segments.json").write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-segments.v1",
                "segments": [
                    {"segment_id": 0, "name": "other"},
                    {"segment_id": 1, "name": "tire"},
                    {"segment_id": 2, "name": "body"},
                ],
            }
        ),
        encoding="utf-8",
    )
    # This is the canonical skill shape: the visual inventory does not need
    # synthetic `parts`, status, or completion_record fields.
    (source_run / "part_plan.json").write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-part-plan.v1",
                "segments": [
                    {
                        "segment_id": 1,
                        "semantic_name": "tire",
                        "expected_instances": "one",
                        "likely_confusers": ["body"],
                        "visual_description": "rubber shell",
                    },
                    {
                        "segment_id": 2,
                        "semantic_name": "body",
                        "expected_instances": "one",
                        "likely_confusers": [],
                        "visual_description": "broad residual",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    for segment_id, name in ((1, "tire"), (2, "body")):
        completion = source_run / name / "part_completion.json"
        completion.parent.mkdir()
        completion.write_text(
            json.dumps(
                {
                    "schema_version": "mesh-segmentation-part-completion.v1",
                    "status": "locked",
                    "semantic_part": name,
                    "segment_id": segment_id,
                    "locked_face_count": 2,
                    "final_revision": "rev-000",
                }
            ),
            encoding="utf-8",
        )
    return source_run


def _write_recognition_lock_evidence(
    run_dir: Path,
    *,
    revision: str,
    target_name: str,
) -> dict[str, object]:
    revision_dir = run_dir / "hypotheses" / revision
    focused_dir = revision_dir / "focused_renders"
    canonical_dir = revision_dir / "canonical_renders"
    focused_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    selection_events = revision_dir / "selection_events.json"
    selection_events.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "operation": "lock_candidate_component_at_pick",
                        "target_patch_id": "000",
                    },
                    *[
                        {
                            "operation": operation,
                            "target_patch_id": "000",
                            "probe_passed": True,
                            "ray_direction": direction,
                        }
                        for operation in (
                            "verify_target_at_pick",
                            "verify_other_at_pick",
                        )
                        for direction in ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )
    semantic_completion = revision_dir / "semantic_completion.json"
    semantic_completion.write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    evidence = {
        "selection_events": [str(selection_events.relative_to(run_dir))],
        "focused_render_dir": str(focused_dir.relative_to(run_dir)),
        "canonical_render_dir": str(canonical_dir.relative_to(run_dir)),
        "semantic_completion": str(semantic_completion.relative_to(run_dir)),
        "diagnostic_color": [0.95, 0.08, 0.62],
    }
    if target_name != "body":
        evidence["semantic_overlay_evidence"] = _write_part_semantic_overlay_evidence(
            run_dir,
            revision=revision,
            target_name=target_name,
        )
    return evidence


def _write_part_semantic_overlay_evidence(
    run_dir: Path,
    *,
    revision: str,
    target_name: str,
) -> dict[str, str]:
    root = run_dir / "hypotheses" / revision / "semantic_overlay_evidence"
    overlay_dir = root / "semantic_overlays"
    registration_dir = root / "registered_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    registration_dir.mkdir(parents=True, exist_ok=True)
    overlay_records = []
    registration_records = []
    for role in CANONICAL_ROLES:
        source = overlay_dir / f"{role}_source.png"
        raw = overlay_dir / f"{role}_raw_overlay.png"
        aligned = registration_dir / f"{role}_aligned_overlay.png"
        mask = registration_dir / f"{role}_aligned_mask.png"
        blend = registration_dir / f"{role}_alignment_blend.png"
        Image.new("RGB", (32, 32), (128, 128, 128)).save(source)
        Image.new("RGB", (32, 32), (255, 0, 255)).save(raw)
        Image.new("RGB", (32, 32), (255, 0, 255)).save(aligned)
        Image.new("L", (32, 32), 255).save(mask)
        Image.new("RGB", (32, 32), (192, 64, 192)).save(blend)
        overlay_record = {
            "role": role,
            "source_render": str(source),
            "source_render_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "raw_overlay": str(raw),
            "raw_overlay_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        }
        overlay_records.append(overlay_record)
        registration_records.append(
            {
                "role": role,
                "accepted": True,
                "aligned_overlay": str(aligned),
                "aligned_overlay_sha256": hashlib.sha256(
                    aligned.read_bytes()
                ).hexdigest(),
                "aligned_mask": str(mask),
                "aligned_mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
                "alignment_blend": str(blend),
                "alignment_blend_sha256": hashlib.sha256(
                    blend.read_bytes()
                ).hexdigest(),
                "silhouette_iou": 1.0,
                "edge_f_score_2px": 1.0,
                "affine_plausibility": {"accepted": True},
                "semantic_pixel_count": 1,
            }
        )
    overlay_manifest = overlay_dir / "manifest.json"
    overlay_manifest.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.mesh-semantic-overlays.v1",
                "target_semantic_part": target_name,
                "records": overlay_records,
            }
        ),
        encoding="utf-8",
    )
    registration_manifest = registration_dir / "manifest.json"
    registration_manifest.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agents.mesh-semantic-overlay-registration.v1"
                ),
                "target_semantic_part": target_name,
                "overlay_manifest_sha256": hashlib.sha256(
                    overlay_manifest.read_bytes()
                ).hexdigest(),
                "accepted_view_count": len(CANONICAL_ROLES),
                "records": registration_records,
            }
        ),
        encoding="utf-8",
    )
    projection_manifest = root / "projection_manifest.json"
    projection_manifest.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.mesh-mask-face-union.v1",
                "target_semantic_part": target_name,
                "registration_manifest_sha256": hashlib.sha256(
                    registration_manifest.read_bytes()
                ).hexdigest(),
                "accepted_roles": list(CANONICAL_ROLES),
                "algorithm": {
                    "name": "registered_mask_nearest_visible_face_union_v1",
                    "pixel_policy": "every_foreground_mask_pixel",
                    "visibility_policy": "closest_ray_intersection_only",
                    "multi_view_reduction": "set_union",
                    "mesh_adjacency_expansion": "none",
                    "mask_morphology": "none",
                    "minimum_view_support": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "overlay_manifest": str(overlay_manifest.relative_to(run_dir)),
        "registration_manifest": str(registration_manifest.relative_to(run_dir)),
        "projection_manifest": str(projection_manifest.relative_to(run_dir)),
    }


def test_distinctive_lock_requires_semantic_overlay_evidence(tmp_path: Path) -> None:
    errors = mesh_segmentation_runner._validate_part_semantic_overlay_evidence(
        tmp_path,
        part={"name": "glass", "role": "distinctive"},
        lock={},
    )

    assert any("lacks semantic_overlay_evidence" in error for error in errors)


def test_distinctive_lock_rejects_single_view_semantic_evidence(
    tmp_path: Path,
) -> None:
    evidence = _write_part_semantic_overlay_evidence(
        tmp_path,
        revision="rev-glass",
        target_name="glass",
    )
    registration_path = tmp_path / evidence["registration_manifest"]
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    for record in registration["records"][1:]:
        record["accepted"] = False
    registration["accepted_view_count"] = 1
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    projection_path = tmp_path / evidence["projection_manifest"]
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    projection["registration_manifest_sha256"] = hashlib.sha256(
        registration_path.read_bytes()
    ).hexdigest()
    projection["accepted_roles"] = ["plus_x"]
    projection_path.write_text(json.dumps(projection), encoding="utf-8")

    errors = mesh_segmentation_runner._validate_part_semantic_overlay_evidence(
        tmp_path,
        part={"name": "glass", "role": "distinctive"},
        lock={"semantic_overlay_evidence": evidence},
    )

    assert any("Fewer than four" in error for error in errors)
    assert any("both asset sides" in error for error in errors)


def test_distinctive_lock_rejects_malformed_overlay_record_without_crashing(
    tmp_path: Path,
) -> None:
    evidence = _write_part_semantic_overlay_evidence(
        tmp_path,
        revision="rev-glass",
        target_name="glass",
    )
    overlay_path = tmp_path / evidence["overlay_manifest"]
    overlays = json.loads(overlay_path.read_text(encoding="utf-8"))
    overlays["records"].append(None)
    overlay_path.write_text(json.dumps(overlays), encoding="utf-8")

    registration_path = tmp_path / evidence["registration_manifest"]
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    registration["overlay_manifest_sha256"] = hashlib.sha256(
        overlay_path.read_bytes()
    ).hexdigest()
    registration_path.write_text(json.dumps(registration), encoding="utf-8")

    projection_path = tmp_path / evidence["projection_manifest"]
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    projection["registration_manifest_sha256"] = hashlib.sha256(
        registration_path.read_bytes()
    ).hexdigest()
    projection_path.write_text(json.dumps(projection), encoding="utf-8")

    errors = mesh_segmentation_runner._validate_part_semantic_overlay_evidence(
        tmp_path,
        part={"name": "glass", "role": "distinctive"},
        lock={"semantic_overlay_evidence": evidence},
    )

    assert any("overlay records must all be objects" in error for error in errors)


def _write_valid_recognition_lock_history(run_dir: Path) -> None:
    (run_dir / "hypotheses" / "rev-light" / "focused_renders").mkdir(
        parents=True,
        exist_ok=True,
    )
    (run_dir / "hypotheses" / "rev-body" / "focused_renders").mkdir(
        parents=True,
        exist_ok=True,
    )
    light_validation = (
        run_dir
        / "hypotheses"
        / "rev-light"
        / "focused_renders"
        / "render_validation.json"
    )
    body_validation = (
        run_dir
        / "hypotheses"
        / "rev-body"
        / "focused_renders"
        / "render_validation.json"
    )
    light_validation.write_text("{}\n", encoding="utf-8")
    body_validation.write_text("{}\n", encoding="utf-8")

    light_labels = b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    body_labels = b"\x01\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00"
    light_path = run_dir / "hypotheses" / "rev-light" / "face_labels.u32le"
    body_path = run_dir / "hypotheses" / "rev-body" / "face_labels.u32le"
    light_path.write_bytes(light_labels)
    body_path.write_bytes(body_labels)
    final_labels = run_dir / "final" / "face_labels.u32le"
    final_labels.parent.mkdir(parents=True, exist_ok=True)
    final_labels.write_bytes(body_labels)

    (run_dir / "part_work_queue.json").write_text(
        json.dumps(
            {
                "mode": "recognition",
                "other_segment_id": 0,
                "active_part": None,
                "parts": [
                    {
                        "name": "marker_light",
                        "segment_id": 1,
                        "role": "distinctive",
                        "estimated_scope": "tiny",
                        "status": "locked",
                    },
                    {
                        "name": "body",
                        "segment_id": 2,
                        "role": "body_fallback",
                        "estimated_scope": "body",
                        "status": "locked",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    locks = [
        {
            "part_name": "marker_light",
            "segment_id": 1,
            "order": 0,
            "face_labels": str(light_path.relative_to(run_dir)),
            "face_labels_sha256": hashlib.sha256(light_labels).hexdigest(),
            "accepted_face_count": 1,
            "unresolved_issues": [],
            "validation_artifacts": [str(light_validation.relative_to(run_dir))],
            **_write_recognition_lock_evidence(
                run_dir,
                revision="rev-light",
                target_name="marker_light",
            ),
        },
        {
            "part_name": "body",
            "segment_id": 2,
            "order": 1,
            "face_labels": str(body_path.relative_to(run_dir)),
            "face_labels_sha256": hashlib.sha256(body_labels).hexdigest(),
            "accepted_face_count": 2,
            "unresolved_issues": [],
            "validation_artifacts": [str(body_validation.relative_to(run_dir))],
            **_write_recognition_lock_evidence(
                run_dir,
                revision="rev-body",
                target_name="body",
            ),
        },
    ]
    (run_dir / "part_lock_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-part-lock-manifest.v1",
                "locks": locks,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "part_review_manifest.json").write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "part_name": lock["part_name"],
                        "lock_fingerprint": hashlib.sha256(
                            json.dumps(lock, sort_keys=True).encode("utf-8")
                        ).hexdigest(),
                        "face_labels_sha256": lock["face_labels_sha256"],
                        "status": "passed",
                    }
                    for lock in locks
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "residual_resolution.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "live_sequential_gate.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "approved_count": 2,
                "approved_parts": ["marker_light", "body"],
            }
        ),
        encoding="utf-8",
    )
    fragment_path = run_dir / "fragments" / "fragment_ids.u32le"
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    if not fragment_path.is_file():
        fragment_path.write_bytes(b"\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00")
    _write_valid_direct_initializer(
        run_dir,
        part_name="marker_light",
        segment_id=1,
        parent_labels=b"\x00" * 12,
        revision_labels=light_labels,
    )
    _write_valid_direct_initializer(
        run_dir,
        part_name="body",
        segment_id=2,
        parent_labels=light_labels,
        revision_labels=body_labels,
    )


def _write_valid_direct_initializer(
    run_dir: Path,
    *,
    part_name: str,
    segment_id: int,
    parent_labels: bytes,
    revision_labels: bytes,
    part_dir: Path | None = None,
) -> None:
    # `part_dir` lets a test materialize the part under a slugged or nested
    # directory while the artifacts keep the real semantic name.
    part_dir = part_dir if part_dir is not None else run_dir / part_name
    revision_dir = part_dir / "rev-000"
    revision_dir.mkdir(parents=True, exist_ok=True)
    warning_path = part_dir / "image_generation_warning.json"
    warning_path.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-image-generation-warning.v1",
                "severity": "warning",
                "code": "image_generation_unavailable",
                "semantic_part": part_name,
                "fallback": "direct_agentic_selection",
                "reason": "test fixture",
            }
        ),
        encoding="utf-8",
    )
    decision_path = part_dir / "initializer_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-initializer-decision.v1",
                "status": "accepted",
                "semantic_part": part_name,
                "segment_id": segment_id,
                "decision": "direct_agentic_selection",
                "next_step": "direct_signed_fragment_selection",
                "probe_status": "skipped_image_generation_unavailable",
                "probe_view_ids": [],
                "assessment": {
                    "semantic_quality": "unavailable",
                    "mask_scale": "unavailable",
                    "eroded_interior_support": "unavailable",
                    "cross_view_consistency": "unavailable",
                    "mesh_projection_quality": "unavailable",
                    "confuser_leakage": "unavailable",
                    "rationale": "test fixture",
                },
                "evidence": {
                    "image_generation_warning": "image_generation_warning.json"
                },
            }
        ),
        encoding="utf-8",
    )
    validation_path = part_dir / "initializer_decision_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "mesh-segmentation-initializer-decision-validation.v1"
                ),
                "status": "passed",
                "semantic_part": part_name,
                "segment_id": segment_id,
                "decision": "direct_agentic_selection",
                "next_step": "direct_signed_fragment_selection",
                "probe_status": "skipped_image_generation_unavailable",
                "probe_view_ids": [],
                "decision_path": str(decision_path),
                "decision_sha256": hashlib.sha256(
                    decision_path.read_bytes()
                ).hexdigest(),
                "artifacts": [
                    {
                        "role": "image_generation_warning",
                        "path": str(warning_path),
                        "sha256": hashlib.sha256(warning_path.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    parent_path = part_dir / "parent_labels.u32le"
    parent_path.write_bytes(parent_labels)
    edits_path = revision_dir / "edits.json"
    edits_path.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-fragment-edits.v1",
                "edits": [],
            }
        ),
        encoding="utf-8",
    )
    revision_path = revision_dir / "face_labels.u32le"
    revision_path.write_bytes(revision_labels)
    fragment_path = run_dir / "fragments" / "fragment_ids.u32le"
    revision_manifest_path = revision_dir / "edit_manifest.json"
    revision_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-fragment-edit-batch.v1",
                "status": "passed",
                "active_segment_id": segment_id,
                "semantic_decision_unit": "immutable_fragment",
                "face_labels": str(revision_path),
                "face_labels_sha256": hashlib.sha256(revision_labels).hexdigest(),
                "parent_labels": str(parent_path),
                "parent_labels_sha256": hashlib.sha256(parent_labels).hexdigest(),
                "fragment_labels": str(fragment_path),
                "fragment_labels_sha256": hashlib.sha256(
                    fragment_path.read_bytes()
                ).hexdigest(),
                "edits": str(edits_path),
                "edits_sha256": hashlib.sha256(edits_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    review_path = part_dir / "falsification_review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-falsification-review.v1",
                "status": "accepted",
                "semantic_part": part_name,
                "revision": "rev-000",
                "false_positive_review": "passed",
                "false_negative_review": "passed",
                "actionable_issues": [],
            }
        ),
        encoding="utf-8",
    )
    falsification_path = part_dir / "falsification_validation.json"
    falsification_path.write_text(
        json.dumps(
            {
                "schema_version": ("mesh-segmentation-falsification-validation.v1"),
                "status": "passed",
                "semantic_part": part_name,
                "review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (part_dir / "part_completion.json").write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-part-completion.v1",
                "status": "locked",
                "semantic_part": part_name,
                "segment_id": segment_id,
                "final_revision": "rev-000",
                "falsification_review": "falsification_review.json",
                "falsification_validation": "falsification_validation.json",
                "falsification_validation_sha256": hashlib.sha256(
                    falsification_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def _write_valid_mask_initializer(run_dir: Path) -> Path:
    part_name = "tire"
    segment_id = 1
    part_dir = run_dir / part_name
    union_dir = part_dir / "initializer-seed" / "union"
    id_dir = part_dir / "initializer-seed" / "id-buffers"
    revision_dir = part_dir / "rev-000"
    for path in (union_dir, id_dir, revision_dir):
        path.mkdir(parents=True, exist_ok=True)

    fragment_path = run_dir / "fragments" / "fragment_ids.u32le"
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_labels = np.asarray([0, 0, 1, 2], dtype="<u4")
    fragment_labels.tofile(fragment_path)
    parent_path = run_dir / "state" / "labels-000.u32le"
    parent_path.parent.mkdir(parents=True, exist_ok=True)
    parent_labels = np.asarray([0, 0, 0, 2], dtype="<u4")
    parent_labels.tofile(parent_path)

    decision_evidence_path = part_dir / "initializer-probe" / "probe.json"
    decision_evidence_path.parent.mkdir(parents=True, exist_ok=True)
    decision_evidence_path.write_text('{"status":"passed"}\n', encoding="utf-8")
    decision_path = part_dir / "initializer_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-initializer-decision.v1",
                "status": "accepted",
                "semantic_part": part_name,
                "segment_id": segment_id,
                "decision": "image_mask_seed",
                "next_step": "eight_view_registered_mask_seed",
                "probe_status": "completed",
                "probe_view_ids": ["a", "b", "c"],
                "assessment": {},
                "evidence": {},
            }
        ),
        encoding="utf-8",
    )
    (part_dir / "initializer_decision_validation.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    "mesh-segmentation-initializer-decision-validation.v1"
                ),
                "status": "passed",
                "semantic_part": part_name,
                "segment_id": segment_id,
                "decision": "image_mask_seed",
                "decision_sha256": hashlib.sha256(
                    decision_path.read_bytes()
                ).hexdigest(),
                "artifacts": [
                    {
                        "role": "probe",
                        "path": str(decision_evidence_path),
                        "sha256": hashlib.sha256(
                            decision_evidence_path.read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    view_ids = (
        "plus_xminus_yplus_z",
        "minus_xminus_yplus_z",
        "plus_xplus_yplus_z",
        "minus_xplus_yplus_z",
        "plus_xminus_yminus_z",
        "minus_xminus_yminus_z",
        "plus_xplus_yminus_z",
        "minus_xplus_yminus_z",
    )
    registration_manifest_path = (
        part_dir / "initializer-seed" / "registration_manifest.json"
    )
    registration_manifest_path.write_text(
        '{"schema_version":"mesh-segmentation-semantic-registration.v1"}\n',
        encoding="utf-8",
    )
    view_records: list[dict[str, object]] = []
    id_records: list[dict[str, object]] = []
    per_view_ids: list[np.ndarray] = []
    for index, view_id in enumerate(view_ids):
        view_dir = union_dir / "views" / view_id
        view_dir.mkdir(parents=True)
        selected = np.asarray([index % 2], dtype=np.int64)
        per_view_ids.append(selected)
        selected_npy = view_dir / "selected_fragment_ids.npy"
        np.save(selected_npy, selected, allow_pickle=False)
        selected_raw = view_dir / "selected_fragment_ids.u32le"
        selected.astype("<u4").tofile(selected_raw)

        mask = np.ones((9, 9), dtype=np.uint8) * 255
        mask_path = view_dir / "registered_mask.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        eroded = np.zeros((9, 9), dtype=np.uint8)
        eroded[2:7, 2:7] = 255
        eroded_path = view_dir / "eroded_mask.png"
        Image.fromarray(eroded, mode="L").save(eroded_path)
        fragment_buffer = np.full((9, 9), index % 2, dtype=np.int32)
        fragment_buffer_path = id_dir / f"{view_id}_fragment_ids.npy"
        np.save(fragment_buffer_path, fragment_buffer, allow_pickle=False)
        source_path = view_dir / "source.png"
        chosen_path = view_dir / "chosen.png"
        projection_path = view_dir / "projection.png"
        for image_path in (source_path, chosen_path, projection_path):
            Image.new("RGB", (9, 9), (128, 128, 128)).save(image_path)

        view_records.append(
            {
                "view_id": view_id,
                "registration_manifest": str(registration_manifest_path),
                "registration_manifest_sha256": hashlib.sha256(
                    registration_manifest_path.read_bytes()
                ).hexdigest(),
                "source_render": str(source_path),
                "source_render_sha256": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "registered_mask": str(mask_path),
                "registered_mask_sha256": hashlib.sha256(
                    mask_path.read_bytes()
                ).hexdigest(),
                "fragment_id_buffer": str(fragment_buffer_path),
                "fragment_id_buffer_sha256": hashlib.sha256(
                    fragment_buffer_path.read_bytes()
                ).hexdigest(),
                "erosion_pixels": 2,
                "eroded_positive_pixel_count": 25,
                "chosen_visible_pixel_count": 25,
                "selected_fragment_count": 1,
                "selected_fragment_ids_npy": str(selected_npy),
                "selected_fragment_ids_npy_sha256": hashlib.sha256(
                    selected_npy.read_bytes()
                ).hexdigest(),
                "selected_fragment_ids_raw": str(selected_raw),
                "selected_fragment_ids_raw_sha256": hashlib.sha256(
                    selected_raw.read_bytes()
                ).hexdigest(),
                "eroded_mask": str(eroded_path),
                "eroded_mask_sha256": hashlib.sha256(
                    eroded_path.read_bytes()
                ).hexdigest(),
                "chosen_pixel_overlay": str(chosen_path),
                "chosen_pixel_overlay_sha256": hashlib.sha256(
                    chosen_path.read_bytes()
                ).hexdigest(),
                "selected_fragment_projection": str(projection_path),
                "selected_fragment_projection_sha256": hashlib.sha256(
                    projection_path.read_bytes()
                ).hexdigest(),
            }
        )
        id_records.append(
            {
                "name": view_id,
                "closest_visible_hit_only": True,
                "channels": {
                    "fragment_ids": {
                        "raw": str(fragment_buffer_path),
                        "raw_sha256": hashlib.sha256(
                            fragment_buffer_path.read_bytes()
                        ).hexdigest(),
                    }
                },
            }
        )
    id_manifest_path = id_dir / "manifest.json"
    id_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-id-buffers.v1",
                "views": id_records,
            }
        ),
        encoding="utf-8",
    )

    raw_union = np.unique(np.concatenate(per_view_ids))
    raw_union_path = union_dir / "raw_union_fragment_ids.npy"
    np.save(raw_union_path, raw_union, allow_pickle=False)
    union_npy_path = union_dir / "union_fragment_ids.npy"
    np.save(union_npy_path, raw_union, allow_pickle=False)
    union_raw_path = union_dir / "union_fragment_ids.u32le"
    raw_union.astype("<u4").tofile(union_raw_path)
    expected_labels = np.asarray([1, 1, 1, 2], dtype="<u4")
    expected_path = union_dir / "expected_face_labels.u32le"
    expected_labels.tofile(expected_path)
    edits_path = union_dir / "edits.json"
    edits_path.write_text('{"edits":[]}\n', encoding="utf-8")
    validation_path = union_dir / "validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "mesh-segmentation-deterministic-fragment-union-validation.v1"
                ),
                "status": "passed",
                "exactly_expected_views_verified": True,
                "closest_visible_buffers_verified": True,
                "per_view_projection_independence_verified": True,
                "two_pixel_erosion_verified": True,
                "set_union_replay_verified": True,
                "negative_pixels_used": False,
                "pixel_ratios_used": False,
                "cross_view_voting_used": False,
                "cross_view_negative_veto_used": False,
                "agent_confirmation_before_rev_000": False,
            }
        ),
        encoding="utf-8",
    )
    union_manifest_path = union_dir / "manifest.json"
    union_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": ("mesh-segmentation-deterministic-fragment-union.v1"),
                "method": (
                    "eight_view_two_pixel_eroded_closest_visible_fragment_set_union"
                ),
                "view_count": 8,
                "view_order": list(view_ids),
                "erosion_pixels": 2,
                "cross_view_reduction": "exact_set_union",
                "negative_pixels_used": False,
                "pixel_ratios_used": False,
                "cross_view_voting_used": False,
                "cross_view_negative_veto_used": False,
                "active_segment_id": segment_id,
                "background_segment_id": 0,
                "views": view_records,
                "id_buffer_manifest": str(id_manifest_path),
                "id_buffer_manifest_sha256": hashlib.sha256(
                    id_manifest_path.read_bytes()
                ).hexdigest(),
                "raw_union_fragment_ids": str(raw_union_path),
                "raw_union_fragment_ids_sha256": hashlib.sha256(
                    raw_union_path.read_bytes()
                ).hexdigest(),
                "raw_union_fragment_count": 2,
                "fragment_labels": str(fragment_path),
                "fragment_labels_sha256": hashlib.sha256(
                    fragment_path.read_bytes()
                ).hexdigest(),
                "parent_labels": str(parent_path),
                "parent_labels_sha256": hashlib.sha256(
                    parent_path.read_bytes()
                ).hexdigest(),
                "union_fragment_ids_npy": str(union_npy_path),
                "union_fragment_ids_npy_sha256": hashlib.sha256(
                    union_npy_path.read_bytes()
                ).hexdigest(),
                "union_fragment_ids_raw": str(union_raw_path),
                "union_fragment_ids_raw_sha256": hashlib.sha256(
                    union_raw_path.read_bytes()
                ).hexdigest(),
                "union_fragment_count": 2,
                "expected_face_labels": str(expected_path),
                "expected_face_labels_sha256": hashlib.sha256(
                    expected_path.read_bytes()
                ).hexdigest(),
                "validation": str(validation_path),
                "validation_sha256": hashlib.sha256(
                    validation_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    revision_path = revision_dir / "face_labels.u32le"
    expected_labels.tofile(revision_path)
    revision_manifest_path = revision_dir / "edit_manifest.json"
    revision_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-fragment-edit-batch.v1",
                "status": "passed",
                "active_segment_id": segment_id,
                "semantic_decision_unit": "immutable_fragment",
                "face_labels": str(revision_path),
                "face_labels_sha256": hashlib.sha256(
                    revision_path.read_bytes()
                ).hexdigest(),
                "parent_labels": str(parent_path),
                "parent_labels_sha256": hashlib.sha256(
                    parent_path.read_bytes()
                ).hexdigest(),
                "fragment_labels": str(fragment_path),
                "fragment_labels_sha256": hashlib.sha256(
                    fragment_path.read_bytes()
                ).hexdigest(),
                "edits": str(edits_path),
                "edits_sha256": hashlib.sha256(edits_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return union_manifest_path


def _write_valid_deterministic_initialization(
    run_dir: Path,
    *,
    first_labels: str = "hypotheses/rev-light/face_labels.u32le",
) -> None:
    sample_dir = run_dir / "initialization" / "coarse_samples"
    semantic_dir = run_dir / "initialization" / "semantic_overlays"
    registered_dir = run_dir / "initialization" / "registered_overlays"
    result_dir = run_dir / "initialization" / "result"
    neutral_dir = run_dir / "source" / "neutral_canonical_renders"
    sample_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    registered_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    neutral_dir.mkdir(parents=True, exist_ok=True)
    views = []
    depth_evidence = []
    for role_index, role in enumerate(CANONICAL_ROLES):
        render_path = neutral_dir / f"{role}.png"
        camera_path = neutral_dir / f"{role}_camera.json"
        depth_path = neutral_dir / f"{role}_linear_depth.npy"
        overlay_path = sample_dir / f"{role}_coarse_samples.png"
        Image.new("RGB", (240, 240), (128, 128, 128)).save(render_path)
        Image.new("RGB", (240, 240), (128, 128, 128)).save(overlay_path)
        camera_path.write_text(
            json.dumps({"image_width": 240, "image_height": 240}),
            encoding="utf-8",
        )
        depth_path.write_bytes(b"test-linear-depth")
        depth_evidence.append(
            {
                "role": role,
                "camera": str(camera_path),
                "camera_sha256": hashlib.sha256(camera_path.read_bytes()).hexdigest(),
                "linear_depth": str(depth_path),
                "linear_depth_sha256": hashlib.sha256(
                    depth_path.read_bytes()
                ).hexdigest(),
            }
        )
        views.append(
            {
                "role": role,
                "role_index": role_index,
                "render": str(render_path),
                "render_sha256": hashlib.sha256(render_path.read_bytes()).hexdigest(),
                "camera": str(camera_path),
                "camera_sha256": hashlib.sha256(camera_path.read_bytes()).hexdigest(),
                "image_width": 240,
                "image_height": 240,
                "overlay": str(overlay_path),
                "overlay_sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
            }
        )
    sample_manifest = {
        "schema_version": "content-agents.mesh-coarse-samples.v1",
        "source_face_count": 3,
        "topology_digest": "sha256:test-topology",
        "sampling_algorithm": "canonical_24x24_cell_centers",
        "grid_columns": 24,
        "grid_rows": 24,
        "canonical_roles": list(CANONICAL_ROLES),
        "views": views,
        "sample_count": 2,
        "eligible_sample_count": 2,
        "samples": [
            {
                "sample_id": "plus_x:r00:c00",
                "role": "plus_x",
                "pixel": [5.0, 5.0],
                "hit_face_id": 0,
                "eligible_for_seed": True,
            },
            {
                "sample_id": "plus_x:r00:c01",
                "role": "plus_x",
                "pixel": [15.0, 5.0],
                "hit_face_id": 1,
                "eligible_for_seed": True,
            },
        ],
    }
    sample_path = sample_dir / "manifest.json"
    sample_path.write_text(
        json.dumps(sample_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    overlay_records = []
    registration_records = []
    for role in CANONICAL_ROLES:
        render_path = neutral_dir / f"{role}.png"
        raw_path = semantic_dir / f"{role}_raw_overlay.png"
        aligned_path = registered_dir / f"{role}_aligned_overlay.png"
        mask_path = registered_dir / f"{role}_aligned_mask.png"
        blend_path = registered_dir / f"{role}_alignment_blend.png"
        Image.new("RGB", (240, 240), (255, 0, 255)).save(raw_path)
        Image.new("RGB", (240, 240), (255, 0, 255)).save(aligned_path)
        Image.new("L", (240, 240), 255).save(mask_path)
        Image.new("RGB", (240, 240), (192, 64, 192)).save(blend_path)
        overlay_record = {
            "role": role,
            "source_render": str(render_path),
            "source_render_sha256": hashlib.sha256(
                render_path.read_bytes()
            ).hexdigest(),
            "raw_overlay": str(raw_path),
            "raw_overlay_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "source_size": [240, 240],
            "generated_size": [240, 240],
        }
        overlay_records.append(overlay_record)
        registration_records.append(
            {
                **overlay_record,
                "aligned_overlay": str(aligned_path),
                "aligned_overlay_sha256": hashlib.sha256(
                    aligned_path.read_bytes()
                ).hexdigest(),
                "aligned_mask": str(mask_path),
                "aligned_mask_sha256": hashlib.sha256(
                    mask_path.read_bytes()
                ).hexdigest(),
                "alignment_blend": str(blend_path),
                "alignment_blend_sha256": hashlib.sha256(
                    blend_path.read_bytes()
                ).hexdigest(),
                "source_to_generated_affine": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                "silhouette_iou": 1.0,
                "edge_f_score_2px": 1.0,
                "affine_plausibility": {"accepted": True},
                "semantic_pixel_count": 1,
                "accepted": True,
            }
        )
    overlay_manifest = {
        "schema_version": "content-agents.mesh-semantic-overlays.v1",
        "target_semantic_part": "marker_light",
        "excluded_semantic_parts": ["body"],
        "prompt": "test",
        "backend": "openai",
        "model": "gpt-image-1",
        "records": overlay_records,
    }
    overlay_path = semantic_dir / "manifest.json"
    overlay_path.write_text(
        json.dumps(overlay_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    registration_manifest = {
        "schema_version": ("content-agents.mesh-semantic-overlay-registration.v1"),
        "target_semantic_part": "marker_light",
        "overlay_manifest": str(overlay_path),
        "overlay_manifest_sha256": hashlib.sha256(
            overlay_path.read_bytes()
        ).hexdigest(),
        "accepted_view_count": len(CANONICAL_ROLES),
        "records": registration_records,
    }
    registration_path = registered_dir / "manifest.json"
    registration_path.write_text(
        json.dumps(registration_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    first_path = run_dir / first_labels
    initial_path = result_dir / "initial_face_labels.u32le"
    initial_path.write_bytes(first_path.read_bytes())
    union_path = result_dir / "union_face_ids.u32le"
    union_path.write_bytes(b"\x00\x00\x00\x00")
    (result_dir / "diagnostic.usdc").write_bytes(b"diagnostic")
    projection_views = []
    for role_index, role in enumerate(CANONICAL_ROLES):
        face_path = result_dir / f"{role}_visible_face_ids.u32le"
        face_path.write_bytes(b"\x00\x00\x00\x00" if role_index == 0 else b"")
        projection_overlay = result_dir / f"{role}_projection_pixels.png"
        Image.new("RGB", (240, 240), (192, 64, 192)).save(projection_overlay)
        mask_path = registered_dir / f"{role}_aligned_mask.png"
        source_render = neutral_dir / f"{role}.png"
        camera_path = neutral_dir / f"{role}_camera.json"
        projection_views.append(
            {
                "role": role,
                "mask": str(mask_path),
                "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                "source_render": str(source_render),
                "source_render_sha256": hashlib.sha256(
                    source_render.read_bytes()
                ).hexdigest(),
                "camera": str(camera_path),
                "camera_sha256": hashlib.sha256(camera_path.read_bytes()).hexdigest(),
                "visible_face_ids": str(face_path),
                "visible_face_ids_sha256": hashlib.sha256(
                    face_path.read_bytes()
                ).hexdigest(),
                "pixel_projection_overlay": str(projection_overlay),
                "pixel_projection_overlay_sha256": hashlib.sha256(
                    projection_overlay.read_bytes()
                ).hexdigest(),
            }
        )
    projection = {
        "schema_version": "content-agents.mesh-mask-face-union.v1",
        "topology_digest": "sha256:test-topology",
        "source_face_count": 3,
        "target_semantic_part": "marker_light",
        "segment_id": 1,
        "registration_manifest_sha256": hashlib.sha256(
            registration_path.read_bytes()
        ).hexdigest(),
        "accepted_roles": list(CANONICAL_ROLES),
        "selected_face_count": 1,
        "union_face_ids_sha256": hashlib.sha256(union_path.read_bytes()).hexdigest(),
        "initial_face_labels_sha256": hashlib.sha256(
            initial_path.read_bytes()
        ).hexdigest(),
        "algorithm": {
            "name": "registered_mask_nearest_visible_face_union_v1",
            "pixel_policy": "every_foreground_mask_pixel",
            "visibility_policy": "closest_ray_intersection_only",
            "multi_view_reduction": "set_union",
            "mesh_adjacency_expansion": "none",
            "mask_morphology": "none",
            "minimum_view_support": 1,
        },
        "views": projection_views,
    }
    projection_path = result_dir / "projection_manifest.json"
    projection_path.write_text(
        json.dumps(projection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = {
        "schema_version": "content-agents.mesh-mask-face-union-validation.v1",
        "status": "passed",
        "topology_digest": "sha256:test-topology",
        "projection_manifest_sha256": hashlib.sha256(
            projection_path.read_bytes()
        ).hexdigest(),
        "initial_face_labels_sha256": hashlib.sha256(
            initial_path.read_bytes()
        ).hexdigest(),
        "exact_union_verified": True,
    }
    (result_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "initialization_gate.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agents.mesh-segmentation-initialization-gate.v1"
                ),
                "status": "accepted",
                "first_nonzero_hypothesis": first_labels,
            }
        ),
        encoding="utf-8",
    )


def test_mesh_segmentation_dry_run_stages_blind_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "fresh-run"
    monkeypatch.setenv(
        "CONTENT_AGENT_CODEX_BASE_URL",
        "https://shared-other-workflow-provider.example/v1",
    )
    (paths["references"] / "notes.md").write_text(
        "Reference annotations are not image inputs.\n",
        encoding="utf-8",
    )
    (paths["repo_root"] / "agentic" / "runs" / ".memory").mkdir(parents=True)

    with pytest.warns(RuntimeWarning, match="WU_AGENT_MEMORY_ROOT"):
        exit_code = main(
            [
                "mesh-segmentation",
                "run",
                "--asset",
                str(paths["asset"]),
                "--reference-dir",
                str(paths["references"]),
                "--repo-root",
                str(paths["repo_root"]),
                "--output-dir",
                str(run_dir),
                "--iteration-budget",
                "7",
                *CANONICAL_PROVIDER_CLI_ARGS,
                "--dry-run",
            ]
        )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["schema_version"] == ("content-agents.mesh-segmentation-request.v3")
    assert request["workflow"] == "mesh-segmentation.run"
    assert request["workflow_mode"] == "recognition"
    assert request["constraints"]["iteration_budget"] == 7
    assert request["constraints"]["base_evidence_view_policy"] == (
        "exact_eight_cube_corners"
    )
    assert request["runtime"]["codex_execution_mode"] == "host"
    assert request["runtime"]["codex_container_image"] is None
    memory_root = paths["repo_root"] / "runs" / ".memory"
    assert request["runtime"]["agent_memory"] == {
        "enabled": True,
        "required_capture": True,
        "run_id": request["run_id"],
        "root": str(memory_root),
        "broker_url": None,
        "access": "launcher_broker",
        "scope": "current_run",
        "context_limit": 12,
    }
    assert (memory_root / request["run_id"] / "manifest.json").is_file()
    assert request["isolation"] == {
        "fresh_child_thread": True,
        "conversation_context_inherited": False,
        "prior_run_access_allowed": False,
        "working_directory": str(run_dir),
        "permitted_evidence_root": str(run_dir),
        "permitted_memory_root": None,
        "memory_access": "launcher_broker",
    }
    staged_asset = Path(request["inputs"]["asset"])
    assert staged_asset == run_dir / "inputs" / "source" / "Body.usdc"
    assert staged_asset.read_bytes() == paths["asset"].read_bytes()
    assert paths["asset"].as_posix() not in json.dumps(request)
    assert [Path(path).name for path in request["inputs"]["reference_images"]] == [
        "01_view_A.png",
        "02_view_b.png",
    ]
    assert request["input_staging"]["asset"]["sha256"]
    assert request["required_skills"] == [
        "content-workflow-mesh-segmentation",
        "image-generation",
        "usd-cli",
    ]
    assert request["inputs"]["target_semantic_parts"] == []
    assert request["constraints"]["part_recognition_required"] is True
    assert request["constraints"]["recognition_queue_order"] == (
        "largest_reliable_distinctive_to_small_body_last"
    )
    assert request["constraints"]["unstarted_parts_must_have_zero_faces"] is True
    assert request["constraints"]["locked_revisions_are_immutable"] is True
    assert (
        request["constraints"][
            "locked_fragment_reassignment_requires_transactional_supersession"
        ]
        is False
    )
    assert request["required_final_artifacts"] == [
        *REQUIRED_FINAL_ARTIFACTS,
        *REQUIRED_RECOGNITION_FINAL_ARTIFACTS,
    ]
    assert request["constraints"]["require_part_plan"] is True
    assert request["constraints"]["require_pixel_to_face_seed_events"] is True
    assert request["constraints"]["require_signed_fragment_evidence"] is True
    assert request["constraints"]["require_hypothesis_card"] is False
    assert request["constraints"]["anti_oscillation_revision_limit"] is None
    assert request["constraints"]["require_held_out_visual_validation"] is True
    assert request["constraints"]["residual_body_last"] is True
    assert request["constraints"]["segmentation_strategy"] == (
        "immutable_fragment_direct_visual_include_exclude"
    )
    assert request["constraints"]["oversegmentation_strategy"] == (
        "boundary_safe_fast_mesh_superfacets_v1"
    )
    assert request["constraints"]["semantic_decision_unit"] == "immutable_fragment"
    assert request["constraints"]["triangle_level_semantic_edits_allowed"] is False
    assert request["constraints"]["require_fragment_atomicity"] is True
    assert request["constraints"]["initialization_strategy"] == (
        "per_part_registered_mask_probe_router_v1"
    )
    assert request["constraints"]["initialization_marker_policy"] == (
        "three_view_mask_probe_then_recorded_route"
    )
    assert request["constraints"]["initialization_multi_view_reduction"] == (
        "mask_route_exact_union_direct_route_none"
    )
    assert request["constraints"]["initialization_view_policy"] == (
        "three_view_probe_then_eight_mask_or_adaptive_direct"
    )
    assert request["constraints"]["initialization_visibility_policy"] == (
        "closest_ray_intersection_only"
    )
    assert request["constraints"]["initialization_mesh_expansion"] == (
        "none_direct_fragment_decisions"
    )
    assert request["constraints"]["image_generation_allowed"] is True
    assert request["constraints"]["semantic_mask_projection_allowed"] is True
    assert request["constraints"]["pixel_or_view_voting_allowed"] is False
    assert request["constraints"]["auxiliary_render_channels"] == [
        "normal",
        "linear_depth",
        "closest_visible_face_id",
        "closest_visible_fragment_id",
        "flat_binary_label",
    ]
    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert "`content-workflow-mesh-segmentation`" in prompt
    assert "`image-generation`" in prompt
    assert "Observation-memory contract:" in prompt
    assert "content-agent-memory" in prompt
    assert "--broker-url <launcher-memory-broker-url>" in prompt
    assert "must never be read or" in prompt
    assert "Memory is required for this run" in prompt
    assert "do not query it then" in prompt
    assert "--workflow mesh-segmentation --target EXACT_PART --limit 4" in (
        " ".join(prompt.split())
    )
    assert "Memory is a checkpoint, not a second decision-maker" in prompt
    assert "Do not search memory before every revision or lock" in prompt
    assert "Memory capture is post-decision bookkeeping" in prompt
    assert "launcher automatically records an evidence-only checkpoint" in prompt
    assert "Final export is forbidden" in prompt
    assert "a first-pass revision is not an export-ready result" in prompt
    assert "do not rewrite the decision or any cited evidence" in (
        " ".join(prompt.split())
    )
    assert "delete the stale validation, rerun the validator" in (
        " ".join(prompt.split())
    )
    assert "A memory summary is not semantic evidence" in prompt
    assert "An ambiguous reinterpretation is not new evidence" in prompt
    assert "keep the earlier candidate unchanged" in " ".join(prompt.split())
    normalized_prompt = " ".join(prompt.split())
    assert "bounded uncertainty pass" in prompt
    assert "at most three weak" in " ".join(prompt.split())
    assert "inspect its candidate labels and frontier artifact" in normalized_prompt
    assert "memory search must be the first action" in normalized_prompt
    assert "search both affected parts first" in normalized_prompt
    assert "selected-only geometry alone is not enough" in normalized_prompt
    assert "at most one new revision per revisited part" in normalized_prompt
    assert "explicit count or repeated collection" in normalized_prompt
    assert "never one entry called `four wings`" in normalized_prompt
    assert "instance-count mismatch cannot pass" in normalized_prompt
    assert "nearby_similar_scale_component_ids" in normalized_prompt
    assert "State history is append-only" in prompt
    assert "Never rewrite an existing `state/labels-NNN.u32le`" in normalized_prompt
    assert "Do not rebuild or reapply unchanged later parts" in prompt
    assert "append only the changed part revisions" in normalized_prompt
    assert "terminal promotion marker, not a rolling working alias" in (
        normalized_prompt
    )
    assert "launcher validates the full lock corpus" in normalized_prompt
    assert "only the launcher may write `state/final_labels.u32le`" in (
        normalized_prompt
    )
    assert "component_assignment_review.json" not in normalized_prompt
    assert "global partition checkpoint" not in normalized_prompt
    assert "Memory never replaces or reduces this evidence set" in prompt
    assert all(
        camera in prompt
        for camera in (
            "+x-y+z",
            "-x-y+z",
            "+x+y+z",
            "-x+y+z",
            "+x-y-z",
            "-x-y-z",
            "+x+y-z",
            "-x+y-z",
        )
    )
    assert "fresh, blind mesh-segmentation run" in prompt
    assert "Do not inspect parent directories, sibling runs" in prompt
    assert "prior experiments" in prompt
    assert "complete semantic queue" in prompt
    assert "exact case-sensitive string" not in prompt
    assert "three representative cube-corner views" in prompt
    assert "`PART/initializer_decision.json`" in prompt
    assert "oversegment_mesh.py" in prompt
    assert "fragment IDs" in prompt
    assert "Treat every rev-000 as provisional" in prompt
    assert "transient capacity error" in prompt
    assert "do not rewrite that part's `initializer_decision.json`" in normalized_prompt
    normalized_prompt = " ".join(prompt.split())
    assert "Never use a `rev-000-applied` output" in normalized_prompt
    assert (
        "initializer decisions, rev-000 creation, refinement, and locking are sequential"
        in normalized_prompt
    )
    assert (
        "Retry while a changed camera, evidence source, or edit still produces"
        in prompt
    )
    assert "Do not defer an unstarted part" in prompt
    assert "A token seed does not count" in prompt
    assert "Only validated locks reserve faces" in prompt
    assert "A validated lock, its faces" in normalized_prompt
    assert "transactional supersession" not in normalized_prompt
    assert "Every accepted lock is immutable" in normalized_prompt
    assert "build_topology_component_evidence.py" in prompt
    assert (
        "root `segments.json` is the manifest for the complete final partition"
        in prompt
    )
    assert "prune superseded or emptied IDs" in prompt
    assert "--appearance-dir inputs/references" in prompt
    assert "isolated cards" in prompt
    assert "global component-to-semantics assignment" in normalized_prompt
    assert "challenge the plan's assembly granularity" in normalized_prompt
    assert "a local seam are supporting clues, not sufficient evidence" in (
        normalized_prompt
    )
    assert "repeatable closed boundary around a coherent 3D subassembly" in (
        normalized_prompt
    )
    assert "PART/rev-NNN/selected-only/part-only.usdc" in prompt
    assert "PART/rev-NNN/selected-only/renders/" in prompt
    assert "review render manifest's `scene`" in prompt
    assert "defer the part and continue with another part" in prompt
    assert "Always export the best complete face partition" in prompt
    assert "render" in prompt
    assert "exported USD from held-out views" in prompt
    assert "affine-register them" in prompt
    assert "Do not apply this three-view probe" in prompt
    assert "generate_semantic_overlays.py" in prompt
    assert "project_semantic_mask_votes_to_faces.py" not in prompt
    assert paths["asset"].as_posix() not in prompt
    assert (
        run_dir
        / ".agents"
        / "skills"
        / "content-workflow-mesh-segmentation"
        / "SKILL.md"
    ).is_file()
    assert not (run_dir / "scripts").exists()
    assert "`scripts/" not in prompt
    normalized_paths_prompt = prompt.replace("\\", "/")
    assert "/scripts/apply_fragment_edits.py" in normalized_paths_prompt
    assert "/scripts/build_topology_component_evidence.py" in normalized_paths_prompt
    assert "/scripts/validate_falsification_review.py" in normalized_paths_prompt
    assert "Controlled JSON artifact writes:" in prompt
    assert "A rejected write command is not evidence" in prompt


def test_parent_terminal_promotion_rejects_minimal_completion_records(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    state_dir = run_dir / "state"
    revision_dir = run_dir / "body" / "rev-000"
    state_dir.mkdir(parents=True)
    revision_dir.mkdir(parents=True)
    labels = np.asarray([1, 1], dtype="<u4").tobytes()
    (state_dir / "labels-001.u32le").write_bytes(labels)
    (revision_dir / "face_labels.u32le").write_bytes(labels)
    (run_dir / "body" / "part_completion.json").write_text(
        json.dumps({"status": "locked", "semantic_part": "body"}),
        encoding="utf-8",
    )

    assert (
        mesh_segmentation_runner._parent_terminal_promotion_source(
            run_dir,
            target_parts=["body"],
        )
        is None
    )


def test_parent_terminal_promotion_uses_latest_lock_bound_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    state_dir = run_dir / "state"
    revision_dir = run_dir / "body" / "rev-000"
    state_dir.mkdir(parents=True)
    revision_dir.mkdir(parents=True)
    labels = np.asarray([1, 0, 1], dtype="<u4").tobytes()
    candidate_path = revision_dir / "face_labels.u32le"
    candidate_path.write_bytes(labels)
    digest = hashlib.sha256(labels).hexdigest()
    (run_dir / "body" / "falsification_validation.json").write_text(
        json.dumps(
            {
                "candidate_labels": str(candidate_path),
                "candidate_labels_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    source = state_dir / "labels-009.u32le"
    source.write_bytes(labels)
    (state_dir / "labels-008.u32le").write_bytes(
        np.asarray([0, 0, 0], dtype="<u4").tobytes()
    )
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_validate_terminal_artifacts",
        lambda *args, **kwargs: {"valid": True},
    )

    promotion = mesh_segmentation_runner._parent_terminal_promotion_source(
        run_dir,
        target_parts=["body"],
    )

    assert promotion == (labels, source, digest)


def test_parent_terminal_promotion_requires_final_target_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True)
    body_labels = np.asarray([1, 0, 1], dtype="<u4").tobytes()
    wheel_labels = np.asarray([1, 2, 1], dtype="<u4").tobytes()
    for target, labels in (("body", body_labels), ("wheel", wheel_labels)):
        part_dir = run_dir / target
        revision_dir = part_dir / "rev-000"
        revision_dir.mkdir(parents=True)
        candidate_path = revision_dir / "face_labels.u32le"
        candidate_path.write_bytes(labels)
        (part_dir / "falsification_validation.json").write_text(
            json.dumps(
                {
                    "candidate_labels": str(candidate_path),
                    "candidate_labels_sha256": hashlib.sha256(labels).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
    (state_dir / "labels-009.u32le").write_bytes(body_labels)
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_validate_terminal_artifacts",
        lambda *args, **kwargs: {"valid": True},
    )

    assert (
        mesh_segmentation_runner._parent_terminal_promotion_source(
            run_dir,
            target_parts=["body", "wheel"],
        )
        is None
    )


def test_parent_terminal_promotion_allows_lock_order_to_differ_from_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True)
    wheel_labels = np.asarray([0, 2, 0], dtype="<u4").tobytes()
    body_labels = np.asarray([1, 2, 1], dtype="<u4").tobytes()
    for target, labels in (("body", body_labels), ("wheel", wheel_labels)):
        part_dir = run_dir / target
        revision_dir = part_dir / "rev-000"
        revision_dir.mkdir(parents=True)
        candidate_path = revision_dir / "face_labels.u32le"
        candidate_path.write_bytes(labels)
        (part_dir / "falsification_validation.json").write_text(
            json.dumps(
                {
                    "candidate_labels": str(candidate_path),
                    "candidate_labels_sha256": hashlib.sha256(labels).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
    wheel_state = state_dir / "labels-001.u32le"
    wheel_state.write_bytes(wheel_labels)
    body_state = state_dir / "labels-002.u32le"
    body_state.write_bytes(body_labels)
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_validate_terminal_artifacts",
        lambda *args, **kwargs: {"valid": True},
    )

    promotion = mesh_segmentation_runner._parent_terminal_promotion_source(
        run_dir,
        target_parts=["body", "wheel"],
    )

    assert promotion == (
        body_labels,
        body_state,
        hashlib.sha256(body_labels).hexdigest(),
    )


def test_mesh_segmentation_rejects_asset_bundle_symlink_before_use(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_inputs(tmp_path)
    bundle = tmp_path / "asset-bundle"
    bundle.mkdir()
    asset = bundle / paths["asset"].name
    paths["asset"].replace(asset)
    paths["asset"] = asset
    outside = tmp_path / "outside.usdc"
    outside.write_bytes(b"outside")
    (paths["asset"].parent / "escape.usdc").symlink_to(outside)

    assert (
        main(
            [
                "mesh-segmentation",
                "run",
                "--asset",
                str(paths["asset"]),
                "--asset-root",
                str(paths["asset"].parent),
                "--reference-dir",
                str(paths["references"]),
                "--repo-root",
                str(paths["repo_root"]),
                "--output-dir",
                str(tmp_path / "fresh-run"),
                *CANONICAL_PROVIDER_CLI_ARGS,
                "--dry-run",
            ]
        )
        == 2
    )
    assert "symlinks are not allowed" in capsys.readouterr().err


def test_mesh_segmentation_can_disable_memory_for_controlled_baseline(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "no-memory-run"

    exit_code = main(
        [
            "mesh-segmentation",
            "run",
            "--asset",
            str(paths["asset"]),
            "--reference-dir",
            str(paths["references"]),
            "--repo-root",
            str(paths["repo_root"]),
            "--output-dir",
            str(run_dir),
            *CANONICAL_PROVIDER_CLI_ARGS,
            "--no-memory",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["constraints"]["require_agent_memory"] is False
    assert request["runtime"]["agent_memory"] == {
        "enabled": False,
        "required_capture": False,
        "run_id": None,
        "root": None,
        "broker_url": None,
        "access": None,
        "scope": None,
        "context_limit": None,
    }
    assert request["isolation"]["permitted_memory_root"] is None
    assert request["isolation"]["memory_access"] is None
    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert "Memory is disabled for this run" in prompt


def test_terminal_validation_requires_durable_mesh_memory_observation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "workflow-run"
    run_dir.mkdir()
    memory_root = tmp_path / "memory-runs"
    run_id = "memory-contract"
    memory = AgentMemory(run_id=run_id, memory_root=memory_root)
    request = {
        "runtime": {
            "agent_memory": {
                "enabled": True,
                "run_id": run_id,
                "root": str(memory_root),
            }
        }
    }

    missing = _validate_terminal_artifacts(
        run_dir,
        required_artifacts=[],
        authoritative_request=request,
    )
    assert missing["valid"] is True
    assert missing["agent_memory"]["status"] == "warning"
    assert any(
        "no mesh-segmentation observations" in error
        for error in missing["agent_memory_warnings"]
    )

    memory.remember(
        RememberRequest(
            workflow="mesh-segmentation",
            phase="part-planning",
            interaction=MemoryInteraction(operation="plan_targeted_segmentation"),
            outcome=MemoryOutcome(
                classification="matched",
                summary="The exact target vocabulary is preserved in the plan.",
            ),
        )
    )
    planning_only = _validate_terminal_artifacts(
        run_dir,
        required_artifacts=[],
        authoritative_request=request,
    )
    assert planning_only["valid"] is True
    assert planning_only["agent_memory"]["status"] == "passed"

    memory.remember(
        RememberRequest(
            workflow="mesh-segmentation",
            phase="part-refinement",
            interaction=MemoryInteraction(operation="review_mesh_segment_revision"),
            outcome=MemoryOutcome(
                classification="contradicted",
                summary="The enclosure revision omits visible cage spokes.",
            ),
            tags=("part:enclosure",),
        )
    )
    recorded = _validate_terminal_artifacts(
        run_dir,
        required_artifacts=[],
        authoritative_request=request,
    )
    assert recorded["valid"] is True
    assert recorded["agent_memory"]["status"] == "passed"
    assert recorded["agent_memory"]["observation_count"] == 2
    assert recorded["agent_memory"]["operational_observation_count"] == 2


def test_current_memory_contract_does_not_count_launcher_checkpoint_as_agent_use(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "workflow-run"
    run_dir.mkdir()
    memory_root = tmp_path / "memory-runs"
    run_id = "origin-contract"
    memory = AgentMemory(run_id=run_id, memory_root=memory_root)
    memory.remember(
        RememberRequest(
            workflow="mesh-segmentation",
            phase="revision_review",
            interaction=MemoryInteraction(operation="checkpoint_reviewed_candidate"),
            outcome=MemoryOutcome(
                classification="not_checked",
                summary="Launcher preserved exact revision evidence.",
            ),
            tags=(MEMORY_ORIGIN_LAUNCHER_TAG,),
        )
    )
    request = {
        "schema_version": mesh_segmentation_runner.REQUEST_SCHEMA_VERSION,
        "inputs": {"target_semantic_parts": []},
        "isolation": {"working_directory": str(run_dir)},
        "runtime": {
            "agent_memory": {
                "enabled": True,
                "run_id": run_id,
                "root": str(memory_root),
            }
        },
    }

    launcher_only, warnings = mesh_segmentation_runner._validate_agent_memory(request)

    assert launcher_only["status"] == "warning"
    assert launcher_only["agent_observation_count"] == 0
    assert launcher_only["launcher_observation_count"] == 1
    assert any("no agent-authored observation" in warning for warning in warnings)


def _write_revision_checkpoint_evidence(
    revision_dir: Path,
    selected_dir: Path,
    frontier: Path,
    *,
    evidence_time_ns: int | None = None,
) -> None:
    labels = revision_dir / "face_labels.u32le"
    selected_stage = revision_dir / "selected-only" / "part-only.usdc"
    selected_stage.parent.mkdir(parents=True, exist_ok=True)
    selected_stage.write_bytes(b"selected-stage:" + labels.read_bytes())
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_image = selected_dir / "front.png"
    Image.new("RGB", (4, 4), "white").save(selected_image)
    render_manifest = selected_dir / "render_manifest.json"
    render_manifest.write_text(
        json.dumps(
            {
                "scene": str(selected_stage),
                "scene_sha256": hashlib.sha256(selected_stage.read_bytes()).hexdigest(),
                "renders": [
                    {
                        "name": "front",
                        "image": str(selected_image),
                        "image_sha256": hashlib.sha256(
                            selected_image.read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    frontier.parent.mkdir(parents=True, exist_ok=True)
    frontier.write_text(
        json.dumps(
            {
                "face_labels": str(labels),
                "face_labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    if evidence_time_ns is not None:
        for path in (render_manifest, frontier):
            os.utime(path, ns=(evidence_time_ns, evidence_time_ns))


def test_memory_checkpoint_watchdog_requires_launcher_broker(tmp_path: Path) -> None:
    memory = AgentMemory(run_id="broker-required", memory_root=tmp_path / "memory")

    with pytest.raises(ValueError, match="launcher-owned broker"):
        mesh_segmentation_runner._make_agent_memory_checkpoint_watchdog(
            tmp_path / "run",
            target_parts=("Plug",),
            memory=memory,
        )


@pytest.mark.parametrize(
    ("selected_relative", "frontier_relative"),
    [
        ("review/selected-only/renders", "review/frontier"),
        ("selected-only/renders", "frontier-final"),
        ("rev-000/selected-only/renders", "review"),
        ("rev-000/selected-only/renders", "review/frontier"),
        ("rev-000/selected-only/renders", "review/frontier-rev000"),
        ("rev-000/selected-only/renders", "rev-000/frontier-audit"),
        ("rev-000/selected-only/renders", "rev-000/frontier"),
        ("review/selected-only/renders-rev-000", "review/frontier-rev000"),
    ],
)
def test_memory_checkpoint_watchdog_records_reviewed_revision_once(
    tmp_path: Path,
    selected_relative: str,
    frontier_relative: str,
) -> None:
    run_dir = tmp_path / "workflow-run"
    revision_dir = run_dir / "Plug" / "rev-000"
    selected_dir = run_dir / "Plug" / selected_relative
    frontier_dir = run_dir / "Plug" / frontier_relative
    revision_dir.mkdir(parents=True)
    selected_dir.mkdir(parents=True)
    frontier_dir.mkdir(parents=True)
    labels = revision_dir / "face_labels.u32le"
    memory = AgentMemory(run_id="checkpoint-run", memory_root=run_dir / ".memory")
    watchdog = mesh_segmentation_runner._make_agent_memory_checkpoint_watchdog(
        run_dir,
        target_parts=("Plug",),
        memory=memory,
        remember=memory.remember,
        target_prim_path="/Root/CustomMesh",
    )
    assert watchdog is not None
    # A revision directory is observable before its labels have been written.
    assert watchdog() is None
    labels.write_bytes(b"\0\0\0\0")
    time.sleep(0.01)
    frontier = frontier_dir / "frontier_audit.json"
    _write_revision_checkpoint_evidence(revision_dir, selected_dir, frontier)

    assert watchdog() is None
    assert watchdog() is None
    cards = memory.search(
        MemorySearchQuery(
            workflow="mesh-segmentation",
            target="Plug",
        )
    )
    assert len(cards) == 1
    assert cards[0].operation == "checkpoint_reviewed_candidate"
    assert cards[0].targets == ("Plug", "/Root/CustomMesh")
    assert set(cards[0].artifact_roles) == {
        "candidate_labels",
        "frontier_audit",
        "selected_only_render_manifest",
        "selected_only_review",
    }

    revision_one = run_dir / "Plug" / "rev-001"
    revision_one.mkdir()
    (revision_one / "face_labels.u32le").write_bytes(b"\1\0\0\0")
    assert watchdog() is None
    assert len(memory.search(MemorySearchQuery(target="Plug"))) == 1

    time.sleep(0.01)
    if selected_relative.startswith("rev-"):
        selected_dir = revision_one / "selected-only" / "renders"
        selected_dir.mkdir(parents=True)
    elif selected_relative.startswith("review/selected-only/renders-rev-"):
        selected_dir = run_dir / "Plug" / "review/selected-only/renders-rev-001"
        selected_dir.mkdir(parents=True)
    if frontier_relative.startswith("review/frontier-rev"):
        frontier_dir = run_dir / "Plug" / "review" / "frontier-rev001"
        frontier_dir.mkdir()
        frontier = frontier_dir / "frontier_audit.json"
    elif frontier_relative.startswith("rev-"):
        frontier_dir = revision_one / "frontier-audit"
        frontier_dir.mkdir()
        frontier = frontier_dir / "frontier_audit.json"
    _write_revision_checkpoint_evidence(revision_one, selected_dir, frontier)
    assert watchdog() is None
    assert len(memory.search(MemorySearchQuery(target="Plug"))) == 2


def test_memory_checkpoint_watchdog_discovers_recognition_queue_parts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "workflow-run"
    revision_dir = run_dir / "Plug" / "rev-000"
    selected_dir = revision_dir / "selected-only" / "renders"
    frontier_dir = revision_dir / "frontier-audit"
    selected_dir.mkdir(parents=True)
    frontier_dir.mkdir(parents=True)
    (run_dir / "part_work_queue.json").write_text(
        json.dumps({"parts": [{"name": "Plug"}]}),
        encoding="utf-8",
    )
    labels = revision_dir / "face_labels.u32le"
    labels.write_bytes(b"\0\0\0\0")
    labels_time_ns = time.time_ns()
    os.utime(labels, ns=(labels_time_ns, labels_time_ns))
    frontier_audit = frontier_dir / "frontier_audit.json"
    evidence_time_ns = labels_time_ns + 1_000_000_000
    _write_revision_checkpoint_evidence(
        revision_dir,
        selected_dir,
        frontier_audit,
        evidence_time_ns=evidence_time_ns,
    )

    memory = AgentMemory(
        run_id="recognition-checkpoint", memory_root=run_dir / ".memory"
    )
    watchdog = mesh_segmentation_runner._make_agent_memory_checkpoint_watchdog(
        run_dir,
        target_parts=(),
        memory=memory,
        remember=memory.remember,
    )

    assert watchdog is not None
    assert watchdog() is None
    assert len(memory.search(MemorySearchQuery(target="Plug"))) == 1


def test_memory_checkpoint_watchdog_captures_every_ready_revision(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "workflow-run"
    base_time_ns = time.time_ns()
    for index in range(2):
        revision_dir = run_dir / "Plug" / f"rev-{index:03d}"
        selected_dir = revision_dir / "selected-only" / "renders"
        frontier_dir = revision_dir / "frontier-audit"
        selected_dir.mkdir(parents=True)
        frontier_dir.mkdir()
        labels = revision_dir / "face_labels.u32le"
        labels.write_bytes(index.to_bytes(4, "little"))
        os.utime(labels, ns=(base_time_ns + index, base_time_ns + index))
        frontier = frontier_dir / "frontier_audit.json"
        evidence_time_ns = base_time_ns + 1_000_000_000 + index
        _write_revision_checkpoint_evidence(
            revision_dir,
            selected_dir,
            frontier,
            evidence_time_ns=evidence_time_ns,
        )

    memory = AgentMemory(run_id="all-revisions", memory_root=tmp_path / "memory")
    watchdog = mesh_segmentation_runner._make_agent_memory_checkpoint_watchdog(
        run_dir,
        target_parts=("Plug",),
        memory=memory,
        remember=memory.remember,
    )

    assert watchdog is not None
    assert watchdog() is None
    assert memory.count_observations() == 2


def test_memory_checkpoint_watchdog_captures_rewritten_revision_digest(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "workflow-run"
    revision_dir = run_dir / "Plug" / "rev-000"
    selected_dir = revision_dir / "selected-only" / "renders"
    frontier = revision_dir / "frontier-audit" / "frontier_audit.json"
    revision_dir.mkdir(parents=True)
    labels = revision_dir / "face_labels.u32le"
    labels.write_bytes(b"\0\0\0\0")
    base_time_ns = time.time_ns()
    os.utime(labels, ns=(base_time_ns, base_time_ns))
    _write_revision_checkpoint_evidence(
        revision_dir,
        selected_dir,
        frontier,
        evidence_time_ns=base_time_ns + 1_000_000_000,
    )

    memory = AgentMemory(run_id="rewritten-revision", memory_root=tmp_path / "memory")
    watchdog = mesh_segmentation_runner._make_agent_memory_checkpoint_watchdog(
        run_dir,
        target_parts=("Plug",),
        memory=memory,
        remember=memory.remember,
    )

    assert watchdog is not None
    assert watchdog() is None
    assert memory.count_observations() == 1

    labels.write_bytes(b"\1\0\0\0")
    os.utime(
        labels,
        ns=(base_time_ns + 2_000_000_000, base_time_ns + 2_000_000_000),
    )
    _write_revision_checkpoint_evidence(
        revision_dir,
        selected_dir,
        frontier,
        evidence_time_ns=base_time_ns + 3_000_000_000,
    )

    assert watchdog() is None
    assert memory.count_observations() == 2


def test_memory_checkpoint_watchdog_rejects_symlinked_child_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "workflow-run"
    revision_dir = run_dir / "Plug" / "rev-000"
    selected_dir = revision_dir / "selected-only" / "renders"
    frontier_dir = revision_dir / "frontier-audit"
    selected_dir.mkdir(parents=True)
    frontier_dir.mkdir(parents=True)
    outside = tmp_path / "outside-labels.u32le"
    outside.write_bytes(b"host-secret")
    labels = revision_dir / "face_labels.u32le"
    labels.symlink_to(outside)
    labels_time_ns = time.time_ns()
    os.utime(outside, ns=(labels_time_ns, labels_time_ns))
    frontier_audit = frontier_dir / "frontier_audit.json"
    evidence_time_ns = labels_time_ns + 1_000_000_000
    _write_revision_checkpoint_evidence(
        revision_dir,
        selected_dir,
        frontier_audit,
        evidence_time_ns=evidence_time_ns,
    )

    memory = AgentMemory(run_id="safe-checkpoint", memory_root=tmp_path / "memory")
    broker = AgentMemoryBroker(run_dir=run_dir, memory=memory)
    broker.start()
    try:
        watchdog = mesh_segmentation_runner._make_agent_memory_checkpoint_watchdog(
            run_dir,
            target_parts=("Plug",),
            memory=memory,
            remember=broker.remember,
        )
        assert watchdog is not None
        assert watchdog() is None
        assert memory.count_observations() == 0
        assert outside.read_bytes() == b"host-secret"
    finally:
        broker.close()


def test_memory_checkpoint_watchdog_does_not_cross_bind_shared_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "workflow-run"
    part_dir = run_dir / "Plug"
    revision_zero = part_dir / "rev-000"
    revision_one = part_dir / "rev-001"
    revision_zero.mkdir(parents=True)
    revision_one.mkdir()
    labels_zero = revision_zero / "face_labels.u32le"
    labels_one = revision_one / "face_labels.u32le"
    labels_zero.write_bytes(b"\0\0\0\0")
    labels_one.write_bytes(b"\1\0\0\0")
    labels_time_ns = time.time_ns()
    for path in (labels_zero, labels_one):
        os.utime(path, ns=(labels_time_ns, labels_time_ns))
    selected_dir = part_dir / "review" / "selected-only" / "renders"
    frontier = part_dir / "review" / "frontier_audit.json"
    _write_revision_checkpoint_evidence(
        revision_one,
        selected_dir,
        frontier,
        evidence_time_ns=labels_time_ns + 1_000_000_000,
    )

    memory = AgentMemory(run_id="bound-revision", memory_root=tmp_path / "memory")
    watchdog = mesh_segmentation_runner._make_agent_memory_checkpoint_watchdog(
        run_dir,
        target_parts=("Plug",),
        memory=memory,
        remember=memory.remember,
    )

    assert watchdog is not None
    assert watchdog() is None
    cards = memory.search(MemorySearchQuery(target="Plug"))
    assert len(cards) == 1
    assert "rev-001" in cards[0].summary


def test_agent_memory_rejects_late_observations_after_final_labels_is_touched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "workflow-run"
    (run_dir / "state").mkdir(parents=True)
    memory_root = run_dir / ".memory"
    run_id = "timely-memory-contract"
    memory = AgentMemory(run_id=run_id, memory_root=memory_root)
    request = {
        "schema_version": mesh_segmentation_runner.REQUEST_SCHEMA_VERSION,
        "inputs": {"target_semantic_parts": ["Plug", "Wire"]},
        "isolation": {"working_directory": str(run_dir)},
        "runtime": {
            "agent_memory": {
                "enabled": True,
                "run_id": run_id,
                "root": str(memory_root),
            }
        },
    }

    def remember_part(part: str) -> None:
        memory.remember(
            RememberRequest(
                workflow="mesh-segmentation",
                phase="revision_review",
                interaction=MemoryInteraction(
                    operation="review_mesh_segment_revision",
                    target_object_ids=(part,),
                ),
                outcome=MemoryOutcome(
                    classification="ambiguous",
                    summary=f"{part} still needs completeness review.",
                ),
                tags=(MEMORY_ORIGIN_AGENT_TAG,),
            )
        )

    monkeypatch.setattr(
        memory_module,
        "_utc_now",
        lambda: "2026-08-11T12:00:00+00:00",
    )
    remember_part("Plug")
    final_labels = run_dir / "state" / "final_labels.u32le"
    final_labels.write_bytes(b"\0\0\0\0")
    promotion_time = datetime(2026, 8, 11, 12, 1, tzinfo=UTC).timestamp()
    os.utime(final_labels, (promotion_time, promotion_time))
    (run_dir / "raw").mkdir()
    (run_dir / "raw" / "final_labels_promotion.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    mesh_segmentation_runner.FINAL_LABELS_PROMOTION_SCHEMA_VERSION
                ),
                "observed_at": "2026-08-11T12:01:00+00:00",
                "final_labels": str(final_labels),
                "final_labels_sha256": hashlib.sha256(
                    final_labels.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        memory_module,
        "_utc_now",
        lambda: "2026-08-11T12:02:00+00:00",
    )
    remember_part("Wire")

    promoted_after_memory = datetime(2026, 8, 11, 12, 3, tzinfo=UTC).timestamp()
    os.utime(final_labels, (promoted_after_memory, promoted_after_memory))
    late_summary, late_errors = mesh_segmentation_runner._validate_agent_memory(request)
    assert late_summary["status"] == "warning"
    assert set(late_summary["timely_target_observation_ids"]) == {"Plug"}
    assert any("target part 'Wire'" in error for error in late_errors)


def test_agent_memory_validates_parts_discovered_by_recognition_queue(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "workflow-run"
    (run_dir / "state").mkdir(parents=True)
    (run_dir / "raw").mkdir()
    (run_dir / "part_work_queue.json").write_text(
        json.dumps({"parts": [{"name": "Plug"}, {"name": "Wire"}]}),
        encoding="utf-8",
    )
    memory_root = tmp_path / "memory-runs"
    run_id = "recognition-memory-contract"
    memory = AgentMemory(run_id=run_id, memory_root=memory_root)
    memory.remember(
        RememberRequest(
            workflow="mesh-segmentation",
            phase="revision_review",
            interaction=MemoryInteraction(
                operation="review_mesh_segment_revision",
                target_object_ids=("Plug",),
            ),
            outcome=MemoryOutcome(
                classification="matched",
                summary="Plug was reviewed before terminal promotion.",
            ),
            tags=(MEMORY_ORIGIN_AGENT_TAG,),
        )
    )
    final_labels = run_dir / "state" / "final_labels.u32le"
    final_labels.write_bytes(b"\0\0\0\0")
    (run_dir / "raw" / "final_labels_promotion.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    mesh_segmentation_runner.FINAL_LABELS_PROMOTION_SCHEMA_VERSION
                ),
                "observed_at": datetime.now(UTC).isoformat(),
                "final_labels": str(final_labels),
                "final_labels_sha256": hashlib.sha256(
                    final_labels.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    request = {
        "schema_version": mesh_segmentation_runner.REQUEST_SCHEMA_VERSION,
        "inputs": {"target_semantic_parts": []},
        "isolation": {"working_directory": str(run_dir)},
        "runtime": {
            "agent_memory": {
                "enabled": True,
                "run_id": run_id,
                "root": str(memory_root),
            }
        },
    }

    summary, warnings = mesh_segmentation_runner._validate_agent_memory(request)

    assert summary["status"] == "warning"
    assert set(summary["timely_target_observation_ids"]) == {"Plug"}
    assert any("target part 'Wire'" in warning for warning in warnings)


def test_agent_memory_accepts_observations_before_refreshed_copy2_promotion(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "workflow-run"
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True)
    numbered_labels = state_dir / "labels-002.u32le"
    numbered_labels.write_bytes(b"\0\0\0\0")
    old_time = time.time() - 60
    os.utime(numbered_labels, (old_time, old_time))

    memory_root = run_dir / ".memory"
    run_id = "copy2-promotion-memory-contract"
    memory = AgentMemory(run_id=run_id, memory_root=memory_root)
    request = {
        "inputs": {"target_semantic_parts": ["Plug", "Wire"]},
        "isolation": {"working_directory": str(run_dir)},
        "runtime": {
            "agent_memory": {
                "enabled": True,
                "run_id": run_id,
                "root": str(memory_root),
            }
        },
    }
    for part in ("Plug", "Wire"):
        memory.remember(
            RememberRequest(
                workflow="mesh-segmentation",
                phase="revision_review",
                interaction=MemoryInteraction(
                    operation="review_mesh_segment_revision",
                    target_object_ids=(part,),
                ),
                outcome=MemoryOutcome(
                    classification="ambiguous",
                    summary=f"{part} still needs completeness review.",
                ),
            )
        )

    final_labels = state_dir / "final_labels.u32le"
    shutil.copy2(numbered_labels, final_labels)
    assert final_labels.stat().st_mtime == numbered_labels.stat().st_mtime
    os.utime(final_labels)
    assert final_labels.stat().st_mtime > numbered_labels.stat().st_mtime

    summary, errors = mesh_segmentation_runner._validate_agent_memory(request)

    assert errors == []
    assert summary["status"] == "passed"
    assert set(summary["timely_target_observation_ids"]) == {"Plug", "Wire"}


def test_agent_memory_target_validation_is_not_limited_to_newest_fifty(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "workflow-run"
    (run_dir / "state").mkdir(parents=True)
    memory_root = run_dir / ".memory"
    memory = AgentMemory(run_id="unbounded-target-check", memory_root=memory_root)
    memory.remember(
        RememberRequest(
            workflow="mesh-segmentation",
            phase="revision_review",
            interaction=MemoryInteraction(
                operation="review_mesh_segment_revision",
                target_object_ids=("Plug",),
            ),
            outcome=MemoryOutcome(
                classification="matched",
                summary="Plug was reviewed before final promotion.",
            ),
        )
    )
    final_labels = run_dir / "state" / "final_labels.u32le"
    final_labels.write_bytes(b"\0\0\0\0")
    for index in range(55):
        memory.remember(
            RememberRequest(
                workflow="mesh-segmentation",
                phase="revision_review",
                interaction=MemoryInteraction(
                    operation="review_mesh_segment_revision",
                    target_object_ids=(f"Other-{index}",),
                ),
                outcome=MemoryOutcome(
                    classification="not_checked",
                    summary=f"Later unrelated observation {index}.",
                ),
            )
        )
    request = {
        "inputs": {"target_semantic_parts": ["Plug"]},
        "isolation": {"working_directory": str(run_dir)},
        "runtime": {
            "agent_memory": {
                "enabled": True,
                "run_id": "unbounded-target-check",
                "root": str(memory_root),
            }
        },
    }

    summary, warnings = mesh_segmentation_runner._validate_agent_memory(request)

    assert warnings == []
    assert summary["status"] == "passed"
    assert summary["observation_count"] == 56
    assert set(summary["timely_target_observation_ids"]) == {"Plug"}


def test_mesh_segmentation_dry_run_records_companion_image_fallback(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "companion-run"

    exit_code = main(
        [
            "mesh-segmentation",
            "run",
            "--asset",
            str(paths["asset"]),
            "--reference-dir",
            str(paths["references"]),
            "--repo-root",
            str(paths["repo_root"]),
            "--output-dir",
            str(run_dir),
            "--model",
            "provider/coding-agent-model",
            "--codex-responses-url",
            "https://provider.example/v1/responses",
            "--codex-api-key-env",
            "TEST_PROVIDER_API_KEY",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["runtime"]["image_generation"] == {
        "mode": "coding_agent_companion",
        "backend": None,
        "model": None,
        "base_url": None,
        "transport": "coding_agent_companion_tool",
        "api_key_env": None,
        "fallback": "warn_and_use_direct_agentic_selection",
        "warning_artifact": "PART/image_generation_warning.json",
    }
    assert request["constraints"]["image_generation_mode"] == ("coding_agent_companion")
    assert request["constraints"]["image_generation_unavailable_policy"] == (
        "warn_and_use_direct_agentic_selection"
    )
    assert request["constraints"]["initialization_strategy"] == (
        "per_part_companion_mask_probe_or_warned_direct_v1"
    )
    assert request["constraints"]["initialization_marker_policy"] == (
        "try_companion_three_view_probe_else_warned_direct"
    )
    assert request["constraints"]["initialization_view_policy"] == (
        "companion_three_view_probe_if_available_else_adaptive_direct"
    )
    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert "accompanying this coding-agent session" in prompt
    assert "image_generation_warning.json" in prompt
    assert "do not fabricate mask evidence" in " ".join(prompt.split())


def test_mesh_segmentation_selects_explicit_registered_image_backend(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "registered-backend-run"

    exit_code = main(
        [
            "mesh-segmentation",
            "run",
            "--asset",
            str(paths["asset"]),
            "--reference-dir",
            str(paths["references"]),
            "--repo-root",
            str(paths["repo_root"]),
            "--output-dir",
            str(run_dir),
            "--model",
            "provider/coding-agent-model",
            "--codex-responses-url",
            "https://provider.example/v1/responses",
            "--codex-api-key-env",
            "TEST_PROVIDER_API_KEY",
            "--image-gen-backend",
            "gemini",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["runtime"]["image_generation"] == {
        "mode": "world_understanding_backend",
        "backend": "gemini",
        "model": None,
        "base_url": None,
        "transport": "world_understanding_image_generation_model",
        "api_key_env": None,
        "fallback": "warn_and_use_direct_agentic_selection",
        "warning_artifact": "PART/image_generation_warning.json",
    }


def test_mesh_segmentation_dry_run_records_provider_and_id_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "fragment-run"
    provider_secret = "nvidia-test-secret-must-not-be-recorded"
    (paths["repo_root"] / ".env").write_text(
        f"TEST_PROVIDER_API_KEY={provider_secret}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TEST_PROVIDER_API_KEY", raising=False)

    exit_code = main(
        [
            "mesh-segmentation",
            "run",
            "--workflow-skill",
            "content-workflow-mesh-segmentation",
            "--asset",
            str(paths["asset"]),
            "--reference-image",
            str(paths["references"] / "view_A.png"),
            "--target-semantic-part",
            "tire",
            "--repo-root",
            str(paths["repo_root"]),
            "--output-dir",
            str(run_dir),
            "--model",
            "provider/coding-agent-model",
            "--model-reasoning-effort",
            "high",
            "--codex-responses-url",
            "https://provider.example/v1/responses",
            "--codex-api-key-env",
            "TEST_PROVIDER_API_KEY",
            "--codex-execution-mode",
            "host",
            "--image-gen-backend",
            "openai_compatible",
            "--image-gen-model",
            "provider/image-edit-model",
            "--image-gen-base-url",
            "https://provider.example/v1",
            "--image-gen-api-key-env",
            "TEST_PROVIDER_API_KEY",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["required_skills"] == [
        "content-workflow-mesh-segmentation",
        "image-generation",
        "usd-cli",
    ]
    assert request["runtime"]["model"] == "provider/coding-agent-model"
    assert request["runtime"]["model_reasoning_effort"] == "high"
    assert request["runtime"]["codex_execution_mode"] == "host"
    assert request["runtime"]["codex_responses_url"] == (
        "https://provider.example/v1/responses"
    )
    assert request["runtime"]["codex_sdk_base_url"] == ("https://provider.example/v1")
    assert request["runtime"]["codex_api_key_env"] == "TEST_PROVIDER_API_KEY"
    assert request["runtime"]["codex_auth_mode"] == "explicit_provider"
    assert request["runtime"]["image_generation"] == {
        "mode": "world_understanding_backend",
        "backend": "openai_compatible",
        "model": "provider/image-edit-model",
        "base_url": "https://provider.example/v1",
        "transport": "world_understanding_image_generation_model",
        "api_key_env": "TEST_PROVIDER_API_KEY",
        "fallback": "warn_and_use_direct_agentic_selection",
        "warning_artifact": "PART/image_generation_warning.json",
    }
    constraints = request["constraints"]
    assert constraints["image_generation_allowed"] is True
    assert constraints["semantic_masks_are_authoritative"] is False
    assert constraints["pixel_or_view_voting_allowed"] is False
    assert constraints["initialization_strategy"] == (
        "per_part_registered_mask_probe_router_v1"
    )
    assert constraints["initialization_multi_view_reduction"] == (
        "mask_route_exact_union_direct_route_none"
    )
    assert constraints["initialization_view_policy"] == (
        "three_view_probe_then_eight_mask_or_adaptive_direct"
    )
    assert constraints["require_initializer_decision"] is True
    assert constraints["initializer_decisions"] == [
        "image_mask_seed",
        "direct_agentic_selection",
    ]
    assert constraints["initializer_revision_is_provisional"] is True
    assert constraints["require_falsification_plan"] is True
    assert constraints["require_signed_confuser_evidence"] is True
    assert constraints["require_per_instance_selected_only_completeness"] is True
    assert constraints["require_unexplained_selected_boundary_zero"] is True
    assert constraints["require_adjacent_unselected_frontier_challenge"] is True
    assert constraints["require_falsification_validation"] is True
    assert constraints["require_false_positive_review"] is True
    assert constraints["require_false_negative_review"] is True
    assert (
        constraints["locked_fragment_reassignment_requires_transactional_supersession"]
        is False
    )
    assert constraints["anti_oscillation_revision_limit"] is None
    assert constraints["anti_oscillation_policy"] == (
        "progress_sensitive_retry_else_defer_continue_and_revisit"
    )
    assert "closest_visible_face_id" in constraints["auxiliary_render_channels"]
    assert "closest_visible_fragment_id" in constraints["auxiliary_render_channels"]
    assert "TEST_PROVIDER_API_KEY" in json.dumps(request)
    assert provider_secret not in json.dumps(request)
    assert os.environ["TEST_PROVIDER_API_KEY"] == provider_secret
    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    normalized_prompt = " ".join(prompt.split())
    assert provider_secret not in prompt
    assert "`content-workflow-mesh-segmentation`" in prompt
    assert "`image-generation`" in prompt
    assert "three representative cube-corner views" in prompt
    assert "`PART/initializer_decision.json`" in prompt
    assert "validate_initializer_decision.py`" in prompt
    assert "`PART/initializer_decision_validation.json`" in prompt
    assert "never substitutes for decision validation" in prompt
    assert "`image_mask_seed`" in prompt
    assert "`direct_agentic_selection`" in prompt
    assert "discard every mask-derived" in prompt.lower()
    assert "fragment id" in prompt.lower()
    assert "exact eight-view set" in prompt
    assert "false positives and false negatives" in prompt
    assert "`instance_completeness_reviews`" in prompt
    assert "Connected-component count proves instance presence" in prompt
    assert "collective phrase cannot stand in for its instances" in prompt
    assert "instance-count mismatch cannot pass" in normalized_prompt
    assert "nearby_similar_scale_component_ids" in prompt
    assert "unexplained jagged cuts" in prompt
    assert "adjacent unselected" in prompt
    assert "frontier fragment" in prompt
    assert "Treat every rev-000 as provisional" in prompt
    assert "signed negative" in prompt
    assert "`PART/falsification_review.json`" in prompt
    assert "`PART/falsification_validation.json`" in prompt
    assert (
        "Retry while a changed camera, evidence source, or edit still produces"
        in prompt
    )
    assert "Do not defer an unstarted part" in prompt
    assert "Treat every requested target as a positive assertion" in prompt
    assert "audit every still-unlocked topology component" in normalized_prompt
    assert "A token seed does not count" in prompt
    assert "Only validated locks reserve faces" in prompt
    assert "A validated lock, its faces" in normalized_prompt
    assert "transactional supersession" not in normalized_prompt
    assert "immutable in every run mode" in normalized_prompt
    assert "build_topology_component_evidence.py" in prompt
    assert "--appearance-dir inputs/references" in prompt
    assert "isolated cards" in prompt
    assert "global component-to-semantics assignment" in normalized_prompt
    assert "defer the part and continue with another part" in prompt
    assert "Always export the best complete face partition" in prompt
    assert "Never start Docker" in prompt
    assert not (run_dir / "scripts").exists()
    assert "`scripts/" not in prompt
    normalized_paths_prompt = prompt.replace("\\", "/")
    assert "/scripts/build_topology_component_evidence.py" in normalized_paths_prompt
    assert "/scripts/build_deterministic_fragment_union.py" in normalized_paths_prompt
    assert "/scripts/validate_initializer_decision.py" in normalized_paths_prompt
    assert "/scripts/validate_falsification_review.py" in normalized_paths_prompt


def test_mesh_segmentation_dry_run_records_configured_codex_auth(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "configured-auth-run"

    exit_code = main(
        [
            "mesh-segmentation",
            "run",
            "--asset",
            str(paths["asset"]),
            "--reference-image",
            str(paths["references"] / "view_A.png"),
            "--repo-root",
            str(paths["repo_root"]),
            "--output-dir",
            str(run_dir),
            "--allow-codex-configured-auth",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["runtime"]["codex_auth_mode"] == "configured_sdk"
    assert request["runtime"]["codex_responses_url"] is None
    assert request["runtime"]["codex_api_key_env"] is None


def test_mesh_segmentation_cli_rejects_configured_auth_with_codex_base_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "redirected-configured-auth-run"
    monkeypatch.delenv("CONTENT_AGENT_CODEX_RESPONSES_URL", raising=False)
    monkeypatch.delenv("CONTENT_AGENT_CODEX_API_KEY_ENV", raising=False)

    exit_code = main(
        [
            "mesh-segmentation",
            "run",
            "--asset",
            str(paths["asset"]),
            "--reference-image",
            str(paths["references"] / "view_A.png"),
            "--repo-root",
            str(paths["repo_root"]),
            "--output-dir",
            str(run_dir),
            "--allow-codex-configured-auth",
            "--codex-base-url",
            "https://redirect.example/v1",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    error = capsys.readouterr().err
    assert "--allow-codex-configured-auth cannot be combined with" in error
    assert "--codex-base-url" in error
    assert not run_dir.exists()


def test_mesh_segmentation_allows_explicit_unsafe_host_child(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "unsafe-host-run"

    exit_code = main(
        [
            "mesh-segmentation",
            "run",
            "--asset",
            str(paths["asset"]),
            "--reference-dir",
            str(paths["references"]),
            "--repo-root",
            str(paths["repo_root"]),
            "--output-dir",
            str(run_dir),
            "--codex-execution-mode",
            "host",
            "--codex-sandbox-mode",
            "danger-full-access",
            "--allow-unsafe-host-child",
            *CANONICAL_PROVIDER_CLI_ARGS,
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["runtime"]["codex_execution_mode"] == "host"
    assert request["runtime"]["allow_unsafe_host_child"] is True
    sdk_prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert "Do not inspect parent directories, sibling runs" in sdk_prompt


def test_mesh_segmentation_accepts_unsafe_sandbox_env_for_host_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "unsafe-host-env-run"
    monkeypatch.setenv(
        "CONTENT_AGENT_CODEX_SANDBOX_MODE",
        "danger-full-access",
    )

    arguments = [
        "mesh-segmentation",
        "run",
        "--asset",
        str(paths["asset"]),
        "--reference-dir",
        str(paths["references"]),
        "--repo-root",
        str(paths["repo_root"]),
        "--output-dir",
        str(run_dir),
        "--allow-unsafe-host-child",
        *CANONICAL_PROVIDER_CLI_ARGS,
        "--dry-run",
    ]
    parsed = build_parser().parse_args(arguments)
    exit_code = main(arguments)

    assert exit_code == 0
    assert parsed.codex_sandbox_mode == "danger-full-access"


def test_mesh_segmentation_rejects_unacknowledged_unsafe_host_child(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="requires --allow-unsafe-host-child"):
        run_mesh_segmentation(
            MeshSegmentationConfig(
                repo_root=paths["repo_root"],
                asset_path=paths["asset"],
                reference_images=[paths["references"] / "view_A.png"],
                codex_sandbox_mode="danger-full-access",
                **CANONICAL_PROVIDER_CONFIG,
                dry_run=True,
            )
        )


def test_completed_run_stages_as_locked_continuation(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    source_run = _write_valid_completed_continuation_run(
        tmp_path,
        asset=paths["asset"],
    )
    output_dir = tmp_path / "continued-run"

    result = run_mesh_segmentation(
        MeshSegmentationConfig(
            repo_root=paths["repo_root"],
            asset_path=paths["asset"],
            reference_images=[paths["references"] / "view_A.png"],
            output_dir=output_dir,
            continue_from_run=source_run,
            **CANONICAL_PROVIDER_CONFIG,
            dry_run=True,
        )
    )

    assert result.returncode == 0
    manifest = json.loads(
        (output_dir / "continuation_seed" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_mode"] == "validated_final"
    assert manifest["locked_segment_ids"] == [1, 2]
    assert [
        (record["segment_id"], record["name"], record["face_count"])
        for record in manifest["locked_segments"]
    ] == [(1, "tire", 2), (2, "body", 2)]
    assert manifest["in_progress_part"] is None
    assert (output_dir / "state" / "labels-parent.u32le").read_bytes() == (
        source_run / "state" / "final_labels.u32le"
    ).read_bytes()
    request = json.loads((output_dir / "request.json").read_text(encoding="utf-8"))
    assert "continuation_seed/manifest.json" in request["required_final_artifacts"]
    assert (
        request["constraints"][
            "locked_fragment_reassignment_requires_transactional_supersession"
        ]
        is False
    )
    prompt = (output_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert "transactional supersession" not in prompt
    assert "Every accepted lock is immutable" in prompt

    final_labels = output_dir / "state" / "final_labels.u32le"
    final_labels.write_bytes(
        (output_dir / "state" / "labels-parent.u32le").read_bytes()
    )
    assert mesh_segmentation_runner._validate_continuation_seed_locks(output_dir) == []
    final_labels.write_bytes(b"\x00\x00\x00\x00" + final_labels.read_bytes()[4:])
    assert any(
        "changed continuation-locked faces" in error
        for error in mesh_segmentation_runner._validate_continuation_seed_locks(
            output_dir
        )
    )
    (output_dir / "continuation_seed" / "manifest.json").unlink()
    terminal = _validate_terminal_artifacts(
        output_dir,
        required_artifacts=["continuation_seed/manifest.json"],
        authoritative_request=request,
    )
    assert terminal["valid"] is False
    assert terminal["missing_artifacts"] == ["continuation_seed/manifest.json"]


@pytest.mark.parametrize(
    ("relative_path", "payload"),
    [
        ("terminal_validation.json", {"valid": False}),
        ("validation/final_validation.json", {"status": "failed"}),
    ],
)
def test_continuation_source_mode_rejects_unvalidated_runs(
    tmp_path: Path,
    relative_path: str,
    payload: dict[str, object],
) -> None:
    paths = _write_inputs(tmp_path)
    source_run = _write_valid_completed_continuation_run(
        tmp_path,
        asset=paths["asset"],
    )
    (source_run / relative_path).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Continuation requires"):
        mesh_segmentation_runner._continuation_source_mode(source_run)


@pytest.mark.parametrize(
    ("manifest_update", "expected_error"),
    [
        (
            {"exact_source_face_coverage": False},
            "lacks exact source-face coverage",
        ),
        (
            {"source_sha256": "sha256:different-source"},
            "different source asset",
        ),
    ],
)
def test_continuation_seed_rejects_untrusted_export_manifest(
    tmp_path: Path,
    manifest_update: dict[str, object],
    expected_error: str,
) -> None:
    paths = _write_inputs(tmp_path)
    source_run = _write_valid_completed_continuation_run(
        tmp_path,
        asset=paths["asset"],
    )
    manifest_path = source_run / "final" / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(manifest_update)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run_dir = tmp_path / "continued-run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match=expected_error):
        mesh_segmentation_runner._stage_continuation_seed(
            source_run=source_run,
            run_dir=run_dir,
            staged_asset=paths["asset"],
        )


def test_continuation_seed_accepts_the_documented_nested_part_layout(
    tmp_path: Path,
) -> None:
    """Continuation must resolve parts the way terminal validation does.

    `parts/NN_<slug>/` is an accepted layout, so a run using it passed every
    terminal contract and then failed `--continue-from-run` as missing its
    completion, because staging looked only for `<exact semantic name>/`.
    """

    paths = _write_inputs(tmp_path)
    source_run = _write_valid_completed_continuation_run(
        tmp_path,
        asset=paths["asset"],
    )
    nested = source_run / "parts"
    nested.mkdir()
    for index, name in enumerate(("tire", "body"), 1):
        part_dir = source_run / name
        # A real locked part carries its rev-000 labels; the shared fixture
        # only writes the completion record, and `_part_directories` needs a
        # marker to recognise a directory as a part at all.
        (part_dir / "rev-000").mkdir()
        (part_dir / "rev-000" / "face_labels.u32le").write_bytes(b"\x00" * 4)
        part_dir.rename(nested / f"{index:02d}_{name}")
    assert not (source_run / "tire").exists()

    run_dir = tmp_path / "continued-run"
    run_dir.mkdir()

    seed = mesh_segmentation_runner._stage_continuation_seed(
        source_run=source_run,
        run_dir=run_dir,
        staged_asset=paths["asset"],
    )

    locked = {entry["name"] for entry in seed["locked_segments"]}  # type: ignore[union-attr]
    assert locked == {"tire", "body"}


def test_mesh_segmentation_launches_confined_fresh_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "fresh-run"
    captured: dict[str, object] = {}

    def fake_child(**kwargs: object) -> int:
        captured.update(kwargs)
        Path(str(kwargs["child_output_path"])).write_text(
            "fresh child\n", encoding="utf-8"
        )
        Path(str(kwargs["child_final_path"])).write_text("complete\n", encoding="utf-8")
        for relative in REQUIRED_FINAL_ARTIFACTS:
            artifact = run_dir / relative
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"artifact")
        fragment_labels = b"\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00"
        final_labels = b"\x01\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00"
        (run_dir / "fragments" / "fragment_ids.u32le").write_bytes(fragment_labels)
        (run_dir / "state" / "final_labels.u32le").write_bytes(final_labels)
        (run_dir / "final" / "export_manifest.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "exact_source_face_coverage": True,
                    "semantic_decision_unit": "immutable_fragment",
                    "face_labels_sha256": hashlib.sha256(final_labels).hexdigest(),
                    "output_usd_sha256": hashlib.sha256(
                        (run_dir / "final" / "segmented.usdc").read_bytes()
                    ).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        _write_valid_recognition_lock_history(run_dir)
        _write_valid_deterministic_initialization(run_dir)
        return 0

    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_run_child_agent",
        fake_child,
    )

    result = run_mesh_segmentation(
        MeshSegmentationConfig(
            repo_root=paths["repo_root"],
            asset_path=paths["asset"],
            reference_images=[
                paths["references"] / "view_A.png",
                paths["references"] / "view_b.png",
            ],
            output_dir=run_dir,
            memory_enabled=False,
            **CANONICAL_PROVIDER_CONFIG,
        )
    )

    assert result.returncode == 0
    assert result.completed is True
    child_config = captured["config"]
    assert child_config.agent_cwd == run_dir  # type: ignore[union-attr]
    assert child_config.asset_path == (  # type: ignore[union-attr]
        run_dir / "inputs" / "source" / "Body.usdc"
    )
    assert all(
        path.is_relative_to(run_dir)
        for path in child_config.reference_images  # type: ignore[union-attr]
    )
    assert captured["bridge_artifact_prefix"] == "mesh_segmentation"
    trusted_digests = captured["trusted_script_digests"]
    assert isinstance(trusted_digests, dict)
    assert trusted_digests
    assert all(not Path(path).is_relative_to(run_dir) for path in trusted_digests)
    assert all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in trusted_digests.items()
    )
    terminal = json.loads(
        result.terminal_validation_path.read_text(encoding="utf-8")  # type: ignore[union-attr]
    )
    assert terminal["valid"] is True
    assert terminal["missing_artifacts"] == []


def test_mesh_segmentation_routes_child_memory_through_launcher_broker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "memory-broker-run"
    captured: dict[str, object] = {}

    def fake_child(**kwargs: object) -> int:
        captured.update(kwargs)
        request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
        memory_config = request["runtime"]["agent_memory"]
        assert memory_config["access"] == "launcher_broker"
        assert memory_config["broker_url"].startswith("http://127.0.0.1:")
        evidence = run_dir / "review.txt"
        evidence.write_text("held-out blade-tip review", encoding="utf-8")
        observation = run_dir / "observation.json"
        observation.write_text(
            json.dumps(
                {
                    "workflow": "mesh-segmentation",
                    "phase": "revision_review",
                    "interaction": {
                        "operation": "review_mesh_segment_revision",
                        "target_object_ids": ["fan blades"],
                    },
                    "outcome": {
                        "classification": "ambiguous",
                        "summary": "Blade tips still need another view.",
                    },
                    "artifacts": [
                        {
                            "path": str(evidence),
                            "role": "selected_only_review",
                            "media_type": "text/plain",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert (
            memory_cli.main(
                [
                    "--run-id",
                    request["run_id"],
                    "--broker-url",
                    memory_config["broker_url"],
                    "record",
                    "--input",
                    str(observation),
                ]
            )
            == 0
        )
        Path(str(kwargs["child_output_path"])).write_text(
            "memory recorded\n", encoding="utf-8"
        )
        Path(str(kwargs["child_final_path"])).write_text(
            "incomplete\n", encoding="utf-8"
        )
        return 1

    monkeypatch.setattr(mesh_segmentation_runner, "_run_child_agent", fake_child)

    result = run_mesh_segmentation(
        MeshSegmentationConfig(
            repo_root=paths["repo_root"],
            asset_path=paths["asset"],
            reference_images=[paths["references"] / "view_A.png"],
            output_dir=run_dir,
            iteration_budget=1,
            **CANONICAL_PROVIDER_CONFIG,
        )
    )

    assert result.returncode == 1
    terminal = json.loads(
        result.terminal_validation_path.read_text(encoding="utf-8")  # type: ignore[union-attr]
    )
    assert terminal["agent_memory"]["status"] == "passed"
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    memory_root = Path(request["runtime"]["agent_memory"]["root"])
    assert memory_root == paths["repo_root"] / "runs" / ".memory"
    assert not memory_root.is_relative_to(run_dir)
    assert "--broker-url http://127.0.0.1:" in str(captured["prompt"])


def test_initializer_watchdog_replays_and_seals_mask_route(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_valid_mask_initializer(run_dir)
    watchdog = mesh_segmentation_runner._make_initialization_watchdog(run_dir)

    assert watchdog() is None
    assert watchdog() is None

    gate_path = run_dir / "tire" / "initializer_runtime_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "accepted"
    assert gate["decision"] == "image_mask_seed"
    assert (
        "tire/initializer-seed/union/expected_face_labels.u32le"
        in (gate["sealed_digests"])
    )

    expected_path = (
        run_dir / "tire" / "initializer-seed" / "union" / "expected_face_labels.u32le"
    )
    expected_path.write_bytes(b"\x00" * len(expected_path.read_bytes()))
    # The part is not locked, so a seed that no longer replays returns to the
    # waiting state for the agent to repair instead of destroying the run.
    assert watchdog() is None
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "waiting_for_rev_000"
    assert gate["problems"]


def test_initializer_rejects_mask_ids_that_do_not_replay(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    union_manifest_path = _write_valid_mask_initializer(run_dir)
    manifest = json.loads(union_manifest_path.read_text(encoding="utf-8"))
    first_view = manifest["views"][0]
    mask_path = Path(first_view["registered_mask"])
    Image.new("L", (9, 9), 0).save(mask_path)
    first_view["registered_mask_sha256"] = hashlib.sha256(
        mask_path.read_bytes()
    ).hexdigest()
    union_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors, _, _ = mesh_segmentation_runner.validate_part_initialization(
        run_dir,
        "tire",
    )

    assert any("selected IDs do not replay" in error for error in errors)


def test_recognition_lock_validator_rejects_future_part_labels(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_valid_recognition_lock_history(run_dir)
    first_lock = run_dir / "hypotheses" / "rev-light" / "face_labels.u32le"
    simultaneous_labels = b"\x01\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00"
    first_lock.write_bytes(simultaneous_labels)
    manifest_path = run_dir / "part_lock_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["locks"][0]["face_labels_sha256"] = hashlib.sha256(
        simultaneous_labels
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = validate_recognition_part_locks(run_dir)

    assert any("unstarted segment IDs [2]" in error for error in errors)


def test_recognition_lock_validator_reports_short_later_labels(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_valid_recognition_lock_history(run_dir)
    body_path = run_dir / "hypotheses" / "rev-body" / "face_labels.u32le"
    short_labels = b"\x01\x00\x00\x00\x02\x00\x00\x00"
    body_path.write_bytes(short_labels)

    manifest_path = run_dir / "part_lock_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body_lock = manifest["locks"][1]
    body_lock["face_labels_sha256"] = hashlib.sha256(short_labels).hexdigest()
    body_lock["accepted_face_count"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    review_path = run_dir / "part_review_manifest.json"
    reviews = json.loads(review_path.read_text(encoding="utf-8"))
    reviews["reviews"][1]["lock_fingerprint"] = hashlib.sha256(
        json.dumps(body_lock, sort_keys=True).encode("utf-8")
    ).hexdigest()
    reviews["reviews"][1]["face_labels_sha256"] = body_lock["face_labels_sha256"]
    review_path.write_text(json.dumps(reviews), encoding="utf-8")

    errors = validate_recognition_part_locks(run_dir)

    assert any("body lock changed the source face count" in error for error in errors)


def test_recognition_lock_validator_rejects_body_before_distinctive_parts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_valid_recognition_lock_history(run_dir)
    queue_path = run_dir / "part_work_queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["parts"].reverse()
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    errors = validate_recognition_part_locks(run_dir)

    assert any("body_fallback, last" in error for error in errors)


def test_recognition_lock_validator_reports_invalid_segment_id(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_valid_recognition_lock_history(run_dir)
    queue_path = run_dir / "part_work_queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["parts"][0]["segment_id"] = None
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    errors = validate_recognition_part_locks(run_dir)

    assert any("invalid segment_id" in error for error in errors)


def test_recognition_lock_validator_rejects_reserved_part_name(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_valid_recognition_lock_history(run_dir)
    queue_path = run_dir / "part_work_queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["parts"][0]["name"] = "Final"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    errors = validate_recognition_part_locks(run_dir)

    assert any("reserved run artifact name" in error for error in errors)


def test_recognition_lock_validator_rejects_artifacts_outside_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_valid_recognition_lock_history(run_dir)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "labels.u32le").write_bytes(b"\x01\x00\x00\x00")
    (run_dir / "escape").symlink_to(outside_dir, target_is_directory=True)
    manifest_path = run_dir / "part_lock_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["locks"][0]["validation_artifacts"] = ["/etc/hostname"]
    manifest["locks"][0]["face_labels"] = "escape/labels.u32le"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = validate_recognition_part_locks(run_dir)

    assert any("missing or unsafe validation artifact" in error for error in errors)
    assert any("lacks a safe face_labels path" in error for error in errors)


def _write_live_recognition_queue(run_dir: Path) -> Path:
    queue_path = run_dir / "part_work_queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        json.dumps(
            {
                "other_segment_id": 0,
                "active_part": "marker_light",
                "parts": [
                    {
                        "name": "marker_light",
                        "segment_id": 1,
                        "role": "distinctive",
                        "estimated_scope": "tiny",
                        "status": "active",
                    },
                    {
                        "name": "body",
                        "segment_id": 2,
                        "role": "body_fallback",
                        "estimated_scope": "body",
                        "status": "unstarted",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return queue_path


def test_live_recognition_gate_rejects_reserved_part_name(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    queue_path = _write_live_recognition_queue(run_dir)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["parts"][0]["name"] = "final"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    watchdog = _make_recognition_sequential_watchdog(run_dir)

    assert watchdog() is None
    gate = json.loads(
        (run_dir / "live_sequential_gate.json").read_text(encoding="utf-8")
    )
    assert gate["status"] == "waiting_for_valid_queue"
    assert "reserved run artifact name" in gate["problems"][0]


@pytest.mark.parametrize("label_root", ["hypotheses", "edits"])
def test_live_recognition_gate_rejects_next_part_before_prior_lock(
    tmp_path: Path,
    label_root: str,
) -> None:
    run_dir = tmp_path / "run"
    gate = _make_recognition_sequential_watchdog(run_dir)
    _write_live_recognition_queue(run_dir)
    assert gate() is None

    rev_light = run_dir / label_root / "rev-light"
    rev_light.mkdir(parents=True)
    rev_light.joinpath("face_labels.u32le").write_bytes(
        b"\x01\x00\x00\x00\x00\x00\x00\x00"
    )
    assert gate() is None

    rev_body = run_dir / label_root / "rev-body"
    rev_body.mkdir()
    rev_body.joinpath("face_labels.u32le").write_bytes(
        b"\x01\x00\x00\x00\x02\x00\x00\x00"
    )

    failure = gate()

    assert failure is not None
    assert failure.fatal is True
    assert "later part was labeled" in failure.reason
    live_gate = json.loads(
        (run_dir / "live_sequential_gate.json").read_text(encoding="utf-8")
    )
    assert live_gate["status"] == "rejected"


def test_live_recognition_gate_waits_for_canonical_queue_before_labels(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    gate = _make_recognition_sequential_watchdog(run_dir)
    (run_dir / "part_work_queue.json").write_text(
        json.dumps({"other_segment_id": 0, "queue": []}),
        encoding="utf-8",
    )

    assert gate() is None
    live_gate = json.loads(
        (run_dir / "live_sequential_gate.json").read_text(encoding="utf-8")
    )
    assert live_gate["status"] == "waiting_for_valid_queue"
    assert live_gate["authorized_segment_ids"] == [0]
    if os.name != "nt":
        assert (run_dir / "live_sequential_gate.json").stat().st_mode & 0o777 == 0o644

    revision = run_dir / "hypotheses" / "rev-first"
    revision.mkdir(parents=True)
    revision.joinpath("face_labels.u32le").write_bytes(b"\x01\x00\x00\x00")

    failure = gate()

    assert failure is not None
    assert failure.fatal is True
    assert "non-empty `parts` list" in failure.reason


def test_live_recognition_gate_accepts_part_name_queue_alias(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    gate = _make_recognition_sequential_watchdog(run_dir)
    queue_path = _write_live_recognition_queue(run_dir)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    for part in queue["parts"]:
        part["part_name"] = part.pop("name")
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    assert gate() is None

    live_gate = json.loads(
        (run_dir / "live_sequential_gate.json").read_text(encoding="utf-8")
    )
    assert live_gate["authorized_segment_ids"] == [0, 1]
    assert live_gate["next_part"] == "marker_light"


def test_live_recognition_gate_requires_large_to_small_distinctive_order(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    gate = _make_recognition_sequential_watchdog(run_dir)
    queue_path = _write_live_recognition_queue(run_dir)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["parts"] = [
        {
            "name": "small_marker",
            "segment_id": 1,
            "role": "distinctive",
            "estimated_scope": "tiny",
            "status": "active",
        },
        {
            "name": "large_foreground",
            "segment_id": 2,
            "role": "distinctive",
            "estimated_scope": "large",
            "status": "unstarted",
        },
        {
            "name": "body",
            "segment_id": 3,
            "role": "body_fallback",
            "estimated_scope": "body",
            "status": "unstarted",
        },
    ]
    queue["active_part"] = "small_marker"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    assert gate() is None
    live_gate = json.loads(
        (run_dir / "live_sequential_gate.json").read_text(encoding="utf-8")
    )
    assert live_gate["status"] == "waiting_for_valid_queue"
    assert any("large-to-small" in problem for problem in live_gate["problems"])
    assert live_gate["authorized_segment_ids"] == [0]

    queue["parts"][0], queue["parts"][1] = queue["parts"][1], queue["parts"][0]
    queue["parts"][0]["segment_id"] = 1
    queue["parts"][0]["status"] = "active"
    queue["parts"][1]["segment_id"] = 2
    queue["parts"][1]["status"] = "unstarted"
    queue["active_part"] = "large_foreground"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    assert gate() is None
    live_gate = json.loads(
        (run_dir / "live_sequential_gate.json").read_text(encoding="utf-8")
    )
    assert live_gate["status"] == "approved_for_next_part"
    assert live_gate["authorized_segment_ids"] == [0, 1]
    assert live_gate["next_part"] == "large_foreground"


def test_live_recognition_gate_approves_each_valid_lock_sequentially(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"

    def pass_review(
        part: dict[str, object],
        lock: dict[str, object],
        fingerprint: str,
    ) -> tuple[bool, list[str], dict[str, object]]:
        return (
            True,
            [],
            {
                "part_name": part["name"],
                "lock_fingerprint": fingerprint,
                "face_labels_sha256": lock["face_labels_sha256"],
                "status": "passed",
            },
        )

    gate = _make_recognition_sequential_watchdog(
        run_dir,
        review_candidate=pass_review,
    )
    queue_path = _write_live_recognition_queue(run_dir)
    assert gate() is None

    rev_light = run_dir / "hypotheses" / "rev-light"
    rev_light.mkdir(parents=True)
    light_labels = b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    light_path = rev_light / "face_labels.u32le"
    light_path.write_bytes(light_labels)
    light_validation = rev_light / "render_validation.json"
    light_validation.write_text("{}\n", encoding="utf-8")
    manifest_path = run_dir / "part_lock_manifest.json"
    manifest = {
        "locks": [
            {
                "part_name": "marker_light",
                "segment_id": 1,
                "order": 0,
                "face_labels": str(light_path.relative_to(run_dir)),
                "face_labels_sha256": hashlib.sha256(light_labels).hexdigest(),
                "accepted_face_count": 1,
                "unresolved_issues": [],
                "validation_artifacts": [str(light_validation.relative_to(run_dir))],
                **_write_recognition_lock_evidence(
                    run_dir,
                    revision="rev-light",
                    target_name="marker_light",
                ),
            }
        ]
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["parts"][0]["status"] = "locked"
    queue["active_part"] = None
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    for _ in range(100):
        assert gate() is None
        live_gate = json.loads(
            (run_dir / "live_sequential_gate.json").read_text(encoding="utf-8")
        )
        if live_gate["approved_parts"] == ["marker_light"]:
            break
        time.sleep(0.01)
    assert live_gate["approved_parts"] == ["marker_light"]
    assert live_gate["next_part"] == "body"

    # The child has to observe the approval before it can rewrite active_part.
    # The trusted gate must preserve this handoff state without authorizing a
    # body label while the queue still names the just-approved part.
    queue["active_part"] = "marker_light"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    assert gate() is None
    live_gate = json.loads(
        (run_dir / "live_sequential_gate.json").read_text(encoding="utf-8")
    )
    assert live_gate["approved_parts"] == ["marker_light"]
    assert live_gate["next_part"] == "body"

    queue["active_part"] = "body"
    queue["parts"][1]["status"] = "active"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    body_labels = b"\x01\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00"
    rev_body = run_dir / "hypotheses" / "rev-body"
    rev_body.mkdir()
    body_path = rev_body / "face_labels.u32le"
    body_path.write_bytes(body_labels)
    assert gate() is None

    body_validation = rev_body / "render_validation.json"
    body_validation.write_text("{}\n", encoding="utf-8")
    manifest["locks"].append(
        {
            "part_name": "body",
            "segment_id": 2,
            "order": 1,
            "face_labels": str(body_path.relative_to(run_dir)),
            "face_labels_sha256": hashlib.sha256(body_labels).hexdigest(),
            "accepted_face_count": 2,
            "unresolved_issues": [],
            "validation_artifacts": [str(body_validation.relative_to(run_dir))],
            **_write_recognition_lock_evidence(
                run_dir,
                revision="rev-body",
                target_name="body",
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    queue["parts"][1]["status"] = "locked"
    queue["active_part"] = None
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    for _ in range(100):
        assert gate() is None
        live_gate = json.loads(
            (run_dir / "live_sequential_gate.json").read_text(encoding="utf-8")
        )
        if live_gate["status"] == "complete":
            break
        time.sleep(0.01)
    assert live_gate["status"] == "complete"
    assert live_gate["approved_parts"] == ["marker_light", "body"]


def test_live_recognition_gate_keeps_part_active_after_failed_independent_review(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"

    def fail_review(
        part: dict[str, object],
        lock: dict[str, object],
        fingerprint: str,
    ) -> tuple[bool, list[str], dict[str, object]]:
        problem = "Visible false-positive diagnostic color crosses into the hub"
        return (
            False,
            [problem],
            {
                "part_name": part["name"],
                "lock_fingerprint": fingerprint,
                "face_labels_sha256": lock["face_labels_sha256"],
                "status": "failed",
                "errors": [problem],
            },
        )

    gate = _make_recognition_sequential_watchdog(
        run_dir,
        review_candidate=fail_review,
    )
    queue_path = _write_live_recognition_queue(run_dir)
    assert gate() is None
    revision = run_dir / "hypotheses" / "rev-light"
    revision.mkdir(parents=True)
    labels = b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    labels_path = revision / "face_labels.u32le"
    labels_path.write_bytes(labels)
    validation = revision / "render_validation.json"
    validation.write_text("{}\n", encoding="utf-8")
    lock = {
        "part_name": "marker_light",
        "segment_id": 1,
        "order": 0,
        "face_labels": str(labels_path.relative_to(run_dir)),
        "face_labels_sha256": hashlib.sha256(labels).hexdigest(),
        "accepted_face_count": 1,
        "unresolved_issues": [],
        "validation_artifacts": [str(validation.relative_to(run_dir))],
        **_write_recognition_lock_evidence(
            run_dir,
            revision="rev-light",
            target_name="marker_light",
        ),
    }
    (run_dir / "part_lock_manifest.json").write_text(
        json.dumps({"locks": [lock]}),
        encoding="utf-8",
    )
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["parts"][0]["status"] = "locked"
    queue["active_part"] = None
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    for _ in range(100):
        assert gate() is None
        live_gate = json.loads(
            (run_dir / "live_sequential_gate.json").read_text(encoding="utf-8")
        )
        if live_gate["status"] == "independent_review_failed":
            break
        time.sleep(0.01)

    assert live_gate["approved_count"] == 0
    assert live_gate["next_part"] == "marker_light"
    assert any("false-positive" in problem for problem in live_gate["problems"])


def test_mesh_segmentation_rejects_existing_output_directory(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "existing"
    run_dir.mkdir()

    with pytest.raises(FileExistsError, match="requires a new output directory"):
        run_mesh_segmentation(
            MeshSegmentationConfig(
                repo_root=paths["repo_root"],
                asset_path=paths["asset"],
                reference_images=[paths["references"] / "view_A.png"],
                output_dir=run_dir,
                **CANONICAL_PROVIDER_CONFIG,
                dry_run=True,
            )
        )


def test_stage_inputs_enforces_expected_publisher_digests(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    reference = paths["references"] / "view_A.png"
    asset_sha256 = hashlib.sha256(paths["asset"].read_bytes()).hexdigest()
    reference_sha256 = hashlib.sha256(reference.read_bytes()).hexdigest()
    run_dir = tmp_path / "run"
    for relative in ("inputs/source", "inputs/references"):
        (run_dir / relative).mkdir(parents=True)

    staged_asset, staged_references, staging = mesh_segmentation_runner._stage_inputs(
        MeshSegmentationConfig(
            repo_root=paths["repo_root"],
            asset_path=paths["asset"],
            reference_images=[reference],
            expected_asset_sha256=asset_sha256,
            expected_reference_sha256=[reference_sha256],
        ),
        run_dir,
    )

    assert hashlib.sha256(staged_asset.read_bytes()).hexdigest() == asset_sha256
    assert [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in staged_references
    ] == [reference_sha256]
    assert staging["asset"]["expected_sha256"] == asset_sha256
    assert staging["reference_images"][0]["expected_sha256"] == reference_sha256


def test_stage_inputs_rejects_bundle_links_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    bundle = tmp_path / "asset-bundle"
    bundle.mkdir()
    asset = bundle / paths["asset"].name
    paths["asset"].replace(asset)
    outside = tmp_path / "outside.usdc"
    outside.write_bytes(b"outside")
    (bundle / "escape.usdc").symlink_to(outside)
    run_dir = tmp_path / "run"
    for relative in ("inputs/source", "inputs/references"):
        (run_dir / relative).mkdir(parents=True)

    def unexpected_hash(_path):
        raise AssertionError("unsafe staged links must be rejected before hashing")

    monkeypatch.setattr(mesh_segmentation_runner, "_sha256_file", unexpected_hash)

    with pytest.raises(UnsafeRunArtifactError, match="symlinks are not allowed"):
        mesh_segmentation_runner._stage_inputs(
            MeshSegmentationConfig(
                repo_root=paths["repo_root"],
                asset_path=asset,
                asset_root=bundle,
                reference_images=[],
            ),
            run_dir,
        )


def test_stage_inputs_rejects_changed_publisher_input(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    reference = paths["references"] / "view_A.png"
    run_dir = tmp_path / "run"
    for relative in ("inputs/source", "inputs/references"):
        (run_dir / relative).mkdir(parents=True)

    with pytest.raises(ValueError, match="expected-asset-sha256"):
        mesh_segmentation_runner._stage_inputs(
            MeshSegmentationConfig(
                repo_root=paths["repo_root"],
                asset_path=paths["asset"],
                reference_images=[reference],
                expected_asset_sha256="0" * 64,
                expected_reference_sha256=[
                    hashlib.sha256(reference.read_bytes()).hexdigest()
                ],
            ),
            run_dir,
        )


def test_stage_inputs_rejects_changed_publisher_reference(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    reference = paths["references"] / "view_A.png"
    run_dir = tmp_path / "run"
    for relative in ("inputs/source", "inputs/references"):
        (run_dir / relative).mkdir(parents=True)

    with pytest.raises(ValueError, match="expected-reference-sha256"):
        mesh_segmentation_runner._stage_inputs(
            MeshSegmentationConfig(
                repo_root=paths["repo_root"],
                asset_path=paths["asset"],
                reference_images=[reference],
                expected_asset_sha256=hashlib.sha256(
                    paths["asset"].read_bytes()
                ).hexdigest(),
                expected_reference_sha256=["0" * 64],
            ),
            run_dir,
        )


def test_mesh_segmentation_rejects_long_run_id_before_creating_output(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "must-not-exist"
    config = MeshSegmentationConfig(
        repo_root=paths["repo_root"],
        asset_path=paths["asset"],
        reference_images=[paths["references"] / "view_A.png"],
        output_dir=run_dir,
        run_id="a" * 129,
        **CANONICAL_PROVIDER_CONFIG,
        dry_run=True,
    )

    with pytest.raises(ValueError, match="at most 128"):
        mesh_segmentation_runner._prepare_run_dir(config)

    assert not run_dir.exists()


def test_mesh_segmentation_rejects_memory_inside_output_before_creation(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="outside the child-writable"):
        run_mesh_segmentation(
            MeshSegmentationConfig(
                repo_root=paths["repo_root"],
                asset_path=paths["asset"],
                reference_images=[paths["references"] / "view_A.png"],
                output_dir=run_dir,
                memory_root=run_dir / ".memory",
                **CANONICAL_PROVIDER_CONFIG,
                dry_run=True,
            )
        )

    assert not run_dir.exists()


def test_mesh_segmentation_rejects_existing_memory_before_output_creation(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "must-not-exist"
    memory_root = tmp_path / "memory"
    run_id = "existing-memory-run"
    AgentMemory(run_id=run_id, memory_root=memory_root)

    with pytest.raises(FileExistsError, match="requires a new memory run"):
        run_mesh_segmentation(
            MeshSegmentationConfig(
                repo_root=paths["repo_root"],
                asset_path=paths["asset"],
                reference_images=[paths["references"] / "view_A.png"],
                output_dir=run_dir,
                run_id=run_id,
                memory_root=memory_root,
                **CANONICAL_PROVIDER_CONFIG,
                dry_run=True,
            )
        )

    assert not run_dir.exists()


def test_codex_bridge_maps_named_provider_key_without_recording_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    monkeypatch.setenv("TEST_PROVIDER_API_KEY", "provider-only-secret")
    config = MeshSegmentationConfig(
        repo_root=paths["repo_root"],
        asset_path=paths["asset"],
        reference_images=[paths["references"] / "view_A.png"],
        codex_api_key_env="TEST_PROVIDER_API_KEY",
    )

    child_env = _codex_bridge_env(config)

    assert child_env["OPENAI_API_KEY"] == "provider-only-secret"
    assert config.codex_api_key_env == "TEST_PROVIDER_API_KEY"


def test_codex_container_command_mounts_only_run_as_writable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-must-not-enter-command")
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "fresh-run"
    run_dir.mkdir()
    request_path = run_dir / "raw" / "mesh_segmentation_request.json"
    request_path.parent.mkdir()
    trusted_root = tmp_path / "parent-tools"
    trusted_root.mkdir()
    trusted_script = trusted_root / "prepare_mesh.py"
    trusted_script.write_text("print('trusted')\n", encoding="utf-8")

    command = _build_codex_container_command(
        config=MeshSegmentationConfig(
            repo_root=paths["repo_root"],
            asset_path=paths["asset"],
            reference_images=[paths["references"] / "view_A.png"],
            codex_execution_mode=CODEX_EXECUTION_CONTAINER,
        ),
        run_dir=run_dir,
        sdk_request_path=request_path,
        bridge_path=tmp_path / "codex_sdk_bridge.mjs",
        node_modules=tmp_path / "node_modules",
        auth_path=tmp_path / "auth.json",
        container_name="content-mesh-segmentation-test",
        host_uid=1000,
        host_gid=1000,
        container_environment={
            "CONTENT_WORKFLOW_PARENT_USD_CLI_SESSION_IDENTITY": str(
                run_dir / "raw" / "parent-session.json"
            ),
            "CONTENT_WORKFLOW_USD_CLI_SERVER_URL": "http://127.0.0.1:43210",
            "OPENAI_API_KEY": "test-secret-must-not-enter-command",
            "USD_CLI_ATTACHED_PROJECT_DIR": str(run_dir),
            "USD_CLI_LOCAL_GPU_FORBIDDEN": "1",
            "USD_CLI_NO_DAEMON": "1",
        },
        trusted_script_digests={
            str(trusted_script): hashlib.sha256(trusted_script.read_bytes()).hexdigest()
        },
    )

    assert "--privileged" not in command
    assert "--cap-add" not in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[command.index("--user") + 1] == "1000:1000"
    assert "--gpus" not in command
    assert "/bin/chown" not in command
    assert command[command.index("--network") + 1] == "host"
    mount_specs = [
        command[index + 1] for index, value in enumerate(command) if value == "--mount"
    ]
    writable = [spec for spec in mount_specs if "readonly" not in spec]
    assert writable == [f"type=bind,src={run_dir},dst={run_dir}"]
    assert (
        f"type=bind,src={run_dir / '.runtime'},dst={run_dir / '.runtime'},readonly"
        in mount_specs
    )
    assert f"type=bind,src={trusted_root},dst={trusted_root},readonly" in mount_specs
    assert all(
        "readonly" in spec for spec in mount_specs if f"src={run_dir}," not in spec
    )
    assert f"PYTHONPATH={run_dir / '.runtime' / 'site-packages'}" in command
    assert "OPENAI_API_KEY" in command
    assert "CONTENT_WORKFLOW_PARENT_USD_CLI_SESSION_IDENTITY" in command
    assert "CONTENT_WORKFLOW_USD_CLI_SERVER_URL" in command
    assert "USD_CLI_ATTACHED_PROJECT_DIR" in command
    assert "USD_CLI_LOCAL_GPU_FORBIDDEN" in command
    assert "USD_CLI_NO_DAEMON" in command
    assert "test-secret-must-not-enter-command" not in command


def test_targeted_mode_skips_recognition_and_requires_pick_evidence(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    run_dir = tmp_path / "targeted-run"

    exit_code = main(
        [
            "mesh-segmentation",
            "run",
            "--asset",
            str(paths["asset"]),
            "--target-prim",
            "/World/FusedMesh",
            "--target-semantic-part",
            "tire",
            "--target-semantic-part",
            "window",
            "--reference-dir",
            str(paths["references"]),
            "--repo-root",
            str(paths["repo_root"]),
            "--output-dir",
            str(run_dir),
            *CANONICAL_PROVIDER_CLI_ARGS,
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["inputs"]["target_semantic_parts"] == ["tire", "window"]
    assert request["workflow_mode"] == "targeted"
    assert request["constraints"]["part_recognition_required"] is False
    assert request["constraints"]["segmentation_strategy"] == (
        "immutable_fragment_direct_visual_include_exclude"
    )
    assert request["constraints"]["require_pixel_to_face_seed_events"] is True
    assert request["constraints"]["require_signed_fragment_evidence"] is True
    assert request["constraints"]["require_hypothesis_card"] is False
    assert request["constraints"]["require_held_out_visual_validation"] is True
    assert request["constraints"]["output_partition"] == "target_parts_plus_other"
    assert request["required_final_artifacts"] == [
        *REQUIRED_FINAL_ARTIFACTS,
        *REQUIRED_TARGETED_FINAL_ARTIFACTS,
    ]
    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert "The user-provided semantic target vocabulary is exactly" in prompt
    assert "`tire`, `window`" in prompt
    # The completion and target-evidence contracts must render literally: an
    # unescaped brace here silently became a format field once already.
    assert '`{"targets": [...]}`' in prompt
    # The run layout is a contract, not a suggestion: agents that invent their
    # own structure make correct work unreadable to every consumer.
    assert "rev-NNN/selected-only/renders/VIEW.png" in prompt
    assert "rev-NNN/face_labels.u32le" in prompt
    assert "candidate-vN" in prompt
    assert "this part alone" in prompt or "part *by itself*" in prompt
    assert "mesh-segmentation-part-completion.v1" in prompt
    assert "falsification_validation_sha256" in prompt
    assert "Skip part recognition" in prompt
    assert "do not add, rename, split, or merge non-residual" in prompt
    assert "exact case-sensitive string" in prompt
    assert "never lowercase, slug, number, or nest target directories" in prompt
    assert "broad provided body categories last" in prompt
    assert "plus `other` only as the residual" in prompt
    assert "three representative cube-corner views" in prompt
    assert "`PART/initializer_decision.json`" in prompt
    assert "project_semantic_mask_votes_to_faces.py" not in prompt
    assert "generate_semantic_overlays.py" in prompt
    assert (
        "Produce every path listed in the frozen request's "
        "`required_final_artifacts` array"
    ) in " ".join(prompt.split())


def test_terminal_rejects_invalid_export_manifest(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    for relative in [*REQUIRED_FINAL_ARTIFACTS, *REQUIRED_TARGETED_FINAL_ARTIFACTS]:
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")

    events_path = run_dir / "hypotheses" / "rev-001" / "selection_events.json"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "operation": "lock_candidate_component_at_pick",
                        "candidate_component_id": 7,
                        "polarity": "positive",
                        "ray_direction": [1.0, 0.0, 0.0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "final" / "target_evidence.json").write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target_semantic_part": "tire",
                        "status": "locked",
                        "selection_events": [
                            "hypotheses/rev-001/selection_events.json"
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = _validate_terminal_artifacts(
        run_dir,
        required_artifacts=[
            *REQUIRED_FINAL_ARTIFACTS,
            *REQUIRED_TARGETED_FINAL_ARTIFACTS,
        ],
    )

    assert result["valid"] is False
    errors = result["semantic_validation_errors"]
    assert any("Could not read final/export_manifest.json" in value for value in errors)


def test_terminal_uses_authoritative_request_for_target_gates(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "request.json").write_text(
        json.dumps(
            {
                "required_skills": [],
                "inputs": {"target_semantic_parts": []},
            }
        ),
        encoding="utf-8",
    )
    authoritative_request: dict[str, object] = {
        "required_skills": ["content-workflow-mesh-segmentation"],
        "inputs": {"target_semantic_parts": ["tire"]},
    }

    result = _validate_terminal_artifacts(
        run_dir,
        required_artifacts=[],
        authoritative_request=authoritative_request,
    )

    assert result["valid"] is False
    assert any(
        "tire/initializer_decision.json" in value
        for value in result["semantic_validation_errors"]
    )


def test_terminal_rejects_stale_exported_usd(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    state_labels = run_dir / "state" / "final_labels.u32le"
    final_labels = run_dir / "final" / "face_labels.u32le"
    output_usd = run_dir / "final" / "segmented.usdc"
    export_manifest = run_dir / "final" / "export_manifest.json"
    state_labels.parent.mkdir(parents=True)
    final_labels.parent.mkdir(parents=True)
    labels = b"\x01\x00\x00\x00"
    state_labels.write_bytes(labels)
    final_labels.write_bytes(labels)
    output_usd.write_bytes(b"stale-usd")
    export_manifest.write_text(
        json.dumps(
            {
                "status": "passed",
                "exact_source_face_coverage": True,
                "semantic_decision_unit": "immutable_fragment",
                "face_labels_sha256": hashlib.sha256(labels).hexdigest(),
                "output_usd_sha256": hashlib.sha256(b"new-usd").hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    result = _validate_terminal_artifacts(
        run_dir,
        required_artifacts=[
            "state/final_labels.u32le",
            "final/face_labels.u32le",
            "final/segmented.usdc",
            "final/export_manifest.json",
        ],
        authoritative_request={"required_skills": []},
    )

    assert result["valid"] is False
    assert any(
        "output USD digest must match" in value
        for value in result["semantic_validation_errors"]
    )


def test_terminal_rejects_mismatched_final_face_labels(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    state_labels = run_dir / "state" / "final_labels.u32le"
    final_labels = run_dir / "final" / "face_labels.u32le"
    export_manifest = run_dir / "final" / "export_manifest.json"
    state_labels.parent.mkdir(parents=True)
    final_labels.parent.mkdir(parents=True)
    state_labels.write_bytes(b"\x01\x00\x00\x00")
    final_labels.write_bytes(b"\x02\x00\x00\x00")
    export_manifest.write_text(
        json.dumps(
            {
                "status": "passed",
                "exact_source_face_coverage": True,
                "semantic_decision_unit": "immutable_fragment",
                "face_labels_sha256": hashlib.sha256(
                    state_labels.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    result = _validate_terminal_artifacts(
        run_dir,
        required_artifacts=[
            "state/final_labels.u32le",
            "final/face_labels.u32le",
            "final/export_manifest.json",
        ],
    )

    assert result["valid"] is False
    assert any(
        "Final face labels must exactly match" in value
        for value in result["semantic_validation_errors"]
    )


def test_recognition_terminal_requires_falsification_for_queue_parts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_valid_recognition_lock_history(run_dir)
    (run_dir / "marker_light" / "falsification_validation.json").unlink()

    result = _validate_terminal_artifacts(
        run_dir,
        required_artifacts=["part_lock_manifest.json"],
        authoritative_request={
            "required_skills": ["content-workflow-mesh-segmentation"],
            "inputs": {"target_semantic_parts": []},
        },
    )

    assert result["valid"] is False
    assert any(
        "marker_light/falsification_validation.json" in value
        for value in result["semantic_validation_errors"]
    )


def test_recognition_terminal_rejects_reserved_queue_part_name(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_valid_recognition_lock_history(run_dir)
    queue_path = run_dir / "part_work_queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["parts"][0]["name"] = "Final"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    result = _validate_terminal_artifacts(
        run_dir,
        required_artifacts=["part_lock_manifest.json"],
        authoritative_request={
            "required_skills": ["content-workflow-mesh-segmentation"],
            "inputs": {"target_semantic_parts": []},
        },
    )

    assert result["valid"] is False
    assert any(
        "reserved run artifact name" in value
        for value in result["semantic_validation_errors"]
    )


def test_targeted_selection_evidence_rejects_empty_targets(tmp_path: Path) -> None:
    evidence_path = tmp_path / "run" / "final" / "target_evidence.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text('{"targets": []}\n', encoding="utf-8")

    assert validate_targeted_selection_evidence(tmp_path / "run") == [
        "Targeted selection evidence must contain at least one target"
    ]


def test_run_artifact_resolution_rejects_ambiguous_fallback(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    for revision in ("rev-001", "rev-002"):
        evidence = run_dir / "hypotheses" / revision / "selection_events.json"
        evidence.parent.mkdir(parents=True)
        evidence.write_text('{"events": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_run_artifact(run_dir, "selection_events.json")


def test_targeted_selection_evidence_must_match_requested_targets(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    evidence_path = run_dir / "final" / "target_evidence.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target_semantic_part": "window",
                        "status": "locked",
                        "selection_events": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    # The child-controlled on-disk request agrees with the bad evidence. The
    # launcher-owned request remains authoritative.
    (run_dir / "request.json").write_text(
        json.dumps({"inputs": {"target_semantic_parts": ["window"]}}),
        encoding="utf-8",
    )

    errors = validate_targeted_selection_evidence(
        run_dir,
        authoritative_request={"inputs": {"target_semantic_parts": ["tire"]}},
    )

    assert any("names do not match request.json" in error for error in errors)


def _direct_initializer_at(
    run_dir: Path,
    *,
    part_name: str,
    part_dir: Path,
) -> None:
    """Stage one valid direct-route part in an explicit directory."""

    fragment_path = run_dir / "fragments" / "fragment_ids.u32le"
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_path.write_bytes(b"\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00")
    _write_valid_direct_initializer(
        run_dir,
        part_name=part_name,
        segment_id=1,
        parent_labels=b"\x00" * 12,
        revision_labels=b"\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00",
        part_dir=part_dir,
    )


def test_initializer_gate_accepts_slugged_part_directory(tmp_path: Path) -> None:
    """A directory that folds to the semantic name must still be validated."""

    run_dir = tmp_path / "run"
    _direct_initializer_at(
        run_dir,
        part_name="Support Frame",
        part_dir=run_dir / "support_frame",
    )

    watchdog = mesh_segmentation_runner._make_initialization_watchdog(run_dir)
    assert watchdog() is None
    assert watchdog() is None

    gate = json.loads(
        (run_dir / "support_frame" / "initializer_runtime_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert gate["status"] == "accepted"
    assert gate["semantic_part"] == "Support Frame"


def test_initializer_watchdog_exposes_only_live_observed_lock(tmp_path: Path) -> None:
    """Failure receipts must not infer launcher provenance from disk alone."""

    run_dir = tmp_path / "run"
    _direct_initializer_at(
        run_dir,
        part_name="Tire",
        part_dir=run_dir / "Tire",
    )
    watchdog = mesh_segmentation_runner._make_initialization_watchdog(
        run_dir,
        expected_parts=["Tire", "Window"],
    )

    assert watchdog() is None
    assert watchdog() is None
    observed = mesh_segmentation_runner._watchdog_observed_lock_seals(watchdog)
    gate = json.loads(
        (run_dir / "Tire" / "initializer_runtime_gate.json").read_text(encoding="utf-8")
    )
    assert set(observed) == {"tire"}
    assert observed["tire"] == gate["sealed_digests"]


def test_initializer_gate_enforces_parts_subdirectory_layout(
    tmp_path: Path,
) -> None:
    """A `parts/`-nested layout must be gated, not silently skipped."""

    run_dir = tmp_path / "run"
    part_dir = run_dir / "parts" / "08_Car_Chassis"
    _direct_initializer_at(run_dir, part_name="Car Chassis", part_dir=part_dir)

    assert mesh_segmentation_runner._part_directories(run_dir) == {
        "car_chassis": part_dir.resolve()
    }

    watchdog = mesh_segmentation_runner._make_initialization_watchdog(run_dir)
    assert watchdog() is None
    assert watchdog() is None
    gate = json.loads(
        (part_dir / "initializer_runtime_gate.json").read_text(encoding="utf-8")
    )
    assert gate["status"] == "accepted"

    # The gate must still catch tampering in the nested layout. It records the
    # violation rather than killing the run; terminal validation enforces it.
    revision_path = part_dir / "rev-000" / "face_labels.u32le"
    revision_path.write_bytes(b"\x00" * len(revision_path.read_bytes()))
    assert watchdog() is None
    gate = json.loads(
        (part_dir / "initializer_runtime_gate.json").read_text(encoding="utf-8")
    )
    assert gate["status"] == "rejected"
    assert any(
        "Locked part artifacts changed and no longer validate" in problem
        for problem in gate["problems"]
    )


def test_initializer_gate_rejects_decision_naming_another_part(
    tmp_path: Path,
) -> None:
    """Folding tolerates spelling, not a decision bound to a different part."""

    run_dir = tmp_path / "run"
    part_dir = run_dir / "parts" / "01_tire"
    _direct_initializer_at(run_dir, part_name="Tire", part_dir=part_dir)
    decision_path = part_dir / "initializer_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["semantic_part"] = "Mirror"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    problems, _, _ = mesh_segmentation_runner.validate_part_initialization(
        run_dir,
        "Tire",
        part_dir=part_dir,
    )
    assert any(
        "semantic_part does not match its directory" in problem for problem in problems
    )


def test_normalize_part_key_folds_layout_spellings() -> None:
    normalize = mesh_segmentation_runner._normalize_part_key
    assert normalize("Car Chassis") == "car_chassis"
    assert normalize("08_Car_Chassis") == "car_chassis"
    assert normalize("Support Frame") == normalize("support_frame")
    assert normalize("Fan Blades") == normalize("03-fan-blades")
    assert normalize("Tire") != normalize("Tires")


def test_initializer_gate_reseals_revised_provisional_rev_000(
    tmp_path: Path,
) -> None:
    """rev-000 is provisional: revising it must re-seal, not kill the run."""

    run_dir = tmp_path / "run"
    part_dir = run_dir / "tire"
    _direct_initializer_at(run_dir, part_name="Tire", part_dir=part_dir)
    # The fixture writes part_completion.json for every part. Remove it before
    # the gate ever observes this part, so the test really exercises the
    # provisional case: once the gate has seen a lock it remembers it, and
    # deleting the file afterwards no longer reopens the reseal path. The
    # locked case is test_initializer_gate_refuses_to_reseal_after_lock.
    (part_dir / "part_completion.json").unlink()
    watchdog = mesh_segmentation_runner._make_initialization_watchdog(run_dir)
    assert watchdog() is None
    assert watchdog() is None
    gate_path = part_dir / "initializer_runtime_gate.json"
    assert json.loads(gate_path.read_text(encoding="utf-8"))["status"] == "accepted"

    # Re-initialize the part with a genuinely different rev-000, exactly as a
    # correcting agent would.
    _write_valid_direct_initializer(
        run_dir,
        part_name="Tire",
        segment_id=1,
        parent_labels=b"\x00" * 12,
        revision_labels=b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        part_dir=part_dir,
    )
    (part_dir / "part_completion.json").unlink()
    assert watchdog() is None
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "accepted"
    assert gate["reseal_count"] == 1
    assert gate["superseded_artifacts"]


def test_initializer_gate_still_rejects_invalid_reinitialization(
    tmp_path: Path,
) -> None:
    """Re-sealing is not a bypass: the replacement must pass validation."""

    run_dir = tmp_path / "run"
    part_dir = run_dir / "tire"
    _direct_initializer_at(run_dir, part_name="Tire", part_dir=part_dir)
    watchdog = mesh_segmentation_runner._make_initialization_watchdog(run_dir)
    assert watchdog() is None
    assert watchdog() is None

    decision_path = part_dir / "initializer_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["status"] = "draft"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    assert watchdog() is None
    gate = json.loads(
        (part_dir / "initializer_runtime_gate.json").read_text(encoding="utf-8")
    )
    assert gate["status"] == "rejected"
    assert any(
        "Locked part artifacts changed and no longer validate" in problem
        for problem in gate["problems"]
    )


def test_initializer_gate_returns_to_waiting_when_part_is_rebuilt(
    tmp_path: Path,
) -> None:
    """A wiped part must return to waiting rather than kill the run.

    Only before it locks: rebuilding is legitimate while rev-000 is
    provisional, whereas losing sealed artifacts after a lock is backfill by
    another route and is covered by
    test_locked_part_losing_sealed_artifacts_is_fatal.
    """

    run_dir = tmp_path / "run"
    part_dir = run_dir / "tire"
    _direct_initializer_at(run_dir, part_name="Tire", part_dir=part_dir)
    (part_dir / "part_completion.json").unlink()
    watchdog = mesh_segmentation_runner._make_initialization_watchdog(run_dir)
    assert watchdog() is None
    assert watchdog() is None

    (part_dir / "rev-000" / "face_labels.u32le").unlink()
    assert watchdog() is None
    gate = json.loads(
        (part_dir / "initializer_runtime_gate.json").read_text(encoding="utf-8")
    )
    assert gate["status"] == "waiting_for_rev_000"


def _write_target_evidence(run_dir: Path, *, status: str = "passed") -> None:
    part_dir = run_dir / "tire"
    part_dir.mkdir(parents=True, exist_ok=True)
    (part_dir / "falsification_validation.json").write_text(
        json.dumps(
            {
                "schema_version": ("mesh-segmentation-falsification-validation.v1"),
                "status": status,
                "semantic_part": "Tire",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "final").mkdir(parents=True, exist_ok=True)
    (run_dir / "final" / "target_evidence.json").write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target_semantic_part": "Tire",
                        "status": "locked",
                        "falsification_validation": (
                            "tire/falsification_validation.json"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_targeted_evidence_accepts_falsification_binding(tmp_path: Path) -> None:
    """targets bind to falsification validation, not v4 patch-pick events."""

    from content_workflow_cli.mesh_segmentation_vqa import (
        validate_targeted_selection_evidence,
    )

    run_dir = tmp_path / "run"
    _write_target_evidence(run_dir)
    errors = validate_targeted_selection_evidence(
        run_dir,
        authoritative_request={"inputs": {"target_semantic_parts": ["Tire"]}},
    )
    assert errors == []


def test_targeted_evidence_rejects_failing_validation(tmp_path: Path) -> None:
    """The fragment path is a binding, not a waiver."""

    from content_workflow_cli.mesh_segmentation_vqa import (
        validate_targeted_selection_evidence,
    )

    run_dir = tmp_path / "run"
    _write_target_evidence(run_dir, status="blocked")
    errors = validate_targeted_selection_evidence(
        run_dir,
        authoritative_request={"inputs": {"target_semantic_parts": ["Tire"]}},
    )
    assert any("falsification validation did not pass" in e for e in errors)


def test_watchdog_ignores_non_target_scratch_directories(tmp_path: Path) -> None:
    """A rejected experiment directory must not gate a targeted run."""

    run_dir = tmp_path / "run"
    _direct_initializer_at(run_dir, part_name="Tire", part_dir=run_dir / "tire")
    _direct_initializer_at(
        run_dir,
        part_name="rejected-fine-region-tire",
        part_dir=run_dir / "rejected-fine-region-tire",
    )
    watchdog = mesh_segmentation_runner._make_initialization_watchdog(
        run_dir, expected_parts=["Tire"]
    )
    assert watchdog() is None
    assert watchdog() is None
    assert (run_dir / "tire" / "initializer_runtime_gate.json").is_file()
    # The scratch directory is never gated at all.
    assert not (
        run_dir / "rejected-fine-region-tire" / "initializer_runtime_gate.json"
    ).is_file()


def _continuation_config(tmp_path: Path, targets: list[str]) -> object:
    from content_workflow_cli.mesh_segmentation_runner import MeshSegmentationConfig

    return MeshSegmentationConfig(
        repo_root=tmp_path,
        asset_path=tmp_path / "asset.usdc",
        reference_images=[],
        output_dir=tmp_path / "run",
        target_semantic_parts=targets,
    )


def test_outstanding_work_reports_unlocked_targets(tmp_path: Path) -> None:
    """A run with unlocked targets must be reported as unfinished."""

    run_dir = tmp_path / "run"
    _direct_initializer_at(run_dir, part_name="Tire", part_dir=run_dir / "tire")
    config = _continuation_config(tmp_path, ["Tire", "Window"])
    outstanding = mesh_segmentation_runner._outstanding_work(run_dir, config=config)
    assert outstanding is not None
    # Tire is locked by the fixture's part_completion.json; Window never started.
    assert "1 of 2 target parts unlocked" in outstanding.description
    assert "Window" in outstanding.description
    assert outstanding.unlocked_parts == 1


@pytest.mark.parametrize("status", ["provisional", "deferred", "failed"])
def test_outstanding_work_does_not_treat_nonlocked_completion_as_done(
    tmp_path: Path, status: str
) -> None:
    """An honest provisional result must earn a focused continuation turn."""

    run_dir = tmp_path / "run"
    part_dir = run_dir / "tire"
    _direct_initializer_at(run_dir, part_name="Tire", part_dir=part_dir)
    completion_path = part_dir / "part_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["status"] = status
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    config = _continuation_config(tmp_path, ["Tire"])

    outstanding = mesh_segmentation_runner._outstanding_work(run_dir, config=config)

    assert outstanding is not None
    assert outstanding.unlocked_parts == 1
    assert "Tire" in outstanding.description


def test_outstanding_work_is_none_when_everything_is_present(tmp_path: Path) -> None:
    """The loop must stop once nothing is outstanding."""

    run_dir = tmp_path / "run"
    _direct_initializer_at(run_dir, part_name="Tire", part_dir=run_dir / "tire")
    config = _continuation_config(tmp_path, ["Tire"])
    for name in (
        *mesh_segmentation_runner.REQUIRED_FINAL_ARTIFACTS,
        *mesh_segmentation_runner.REQUIRED_TARGETED_FINAL_ARTIFACTS,
    ):
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    assert mesh_segmentation_runner._outstanding_work(run_dir, config=config) is None


def test_continuation_prompt_states_resume_contract(tmp_path: Path) -> None:
    """A resuming turn must be told what exists and what it may not touch."""

    run_dir = tmp_path / "run"
    _direct_initializer_at(run_dir, part_name="Tire", part_dir=run_dir / "tire")
    config = _continuation_config(tmp_path, ["Tire", "Window"])
    prompt = mesh_segmentation_runner._continuation_prompt(
        "ORIGINAL PROMPT BODY", run_dir, attempt=2, config=config
    )
    assert prompt.startswith("Continuation turn 2.")
    assert "Window" in prompt
    assert "immutable" in prompt
    assert "part_completion.json says provisional or deferred" in prompt
    assert "part_deferral.json does not finish a requested target" in prompt
    assert "audit every still-unlocked topology component" in prompt
    assert "transactional supersession" not in prompt
    assert "face-immutable in every run mode" in prompt
    assert "never extend, reassign, or otherwise change" in prompt
    # The frozen prompt must still be delivered verbatim.
    assert prompt.endswith("ORIGINAL PROMPT BODY")


def test_continuation_prompt_keeps_recognition_locks_face_immutable(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _continuation_config(tmp_path, [])

    prompt = mesh_segmentation_runner._continuation_prompt(
        "ORIGINAL PROMPT BODY", run_dir, attempt=2, config=config
    )

    assert "transactional supersession" not in prompt
    assert "face-immutable in every run mode" in prompt
    assert "never extend, reassign, or otherwise change" in prompt


def test_continuation_seed_disables_targeted_lock_supersession(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = MeshSegmentationConfig(
        repo_root=tmp_path,
        asset_path=tmp_path / "asset.usdc",
        reference_images=[],
        output_dir=run_dir,
        target_semantic_parts=["Tire", "Window"],
        continue_from_run=tmp_path / "source-run",
    )

    prompt = mesh_segmentation_runner._continuation_prompt(
        "ORIGINAL PROMPT BODY", run_dir, attempt=2, config=config
    )
    request = mesh_segmentation_runner._build_request(
        config,
        run_id="seeded-targeted",
        run_dir=run_dir,
        staging={},
    )

    assert "transactional supersession" not in prompt
    assert "face-immutable in every run mode" in prompt
    assert (
        request["constraints"][
            "locked_fragment_reassignment_requires_transactional_supersession"
        ]
        is False
    )


def test_outstanding_work_records_comparable_counts(tmp_path: Path) -> None:
    """The loop needs a measure that distinguishes progress from spinning."""

    run_dir = tmp_path / "run"
    _direct_initializer_at(run_dir, part_name="Tire", part_dir=run_dir / "tire")
    config = _continuation_config(tmp_path, ["Tire", "Window", "Rim"])
    outstanding = mesh_segmentation_runner._outstanding_work(run_dir, config=config)
    assert outstanding is not None
    first = outstanding.counts
    # Tire is locked by the fixture; two targets remain.
    assert first[0] == 2

    _direct_initializer_at(run_dir, part_name="Window", part_dir=run_dir / "window")
    outstanding = mesh_segmentation_runner._outstanding_work(run_dir, config=config)
    assert outstanding is not None
    second = outstanding.counts
    assert second[0] == 1
    assert second != first

    # Re-measuring without any change must report the identical counts, which
    # is what the loop uses to stop.
    outstanding = mesh_segmentation_runner._outstanding_work(run_dir, config=config)
    assert outstanding is not None
    assert outstanding.counts == second


def _terminal_deferral_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, str]:
    """Stage the non-semantic bindings for a terminal-deferral receipt."""

    run_dir = tmp_path / "run"
    for name in (
        *mesh_segmentation_runner.REQUIRED_FINAL_ARTIFACTS,
        *mesh_segmentation_runner.REQUIRED_TARGETED_FINAL_ARTIFACTS,
    ):
        if name == "state/final_labels.u32le":
            continue
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (run_dir / "raw").mkdir(exist_ok=True)

    staged_asset = run_dir / "inputs" / "source" / "asset.usdc"
    staged_asset.parent.mkdir(parents=True)
    staged_asset.write_bytes(b"frozen source")
    staged_asset_sha256 = hashlib.sha256(staged_asset.read_bytes()).hexdigest()

    locked = run_dir / "Body"
    locked.mkdir()
    (locked / "part_completion.json").write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-part-completion.v1",
                "status": "locked",
                "semantic_part": "Body",
                "segment_id": 1,
            }
        ),
        encoding="utf-8",
    )
    decision_path = locked / "initializer_decision.json"
    decision_path.write_text(
        json.dumps({"semantic_part": "Body", "segment_id": 1}),
        encoding="utf-8",
    )
    observed_seals = {
        decision_path.relative_to(run_dir).as_posix(): hashlib.sha256(
            decision_path.read_bytes()
        ).hexdigest()
    }
    (locked / "initializer_runtime_gate.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    mesh_segmentation_runner.INITIALIZER_RUNTIME_GATE_SCHEMA_VERSION
                ),
                "status": "accepted",
                "locked_observed": True,
                "sealed_digests": observed_seals,
            }
        ),
        encoding="utf-8",
    )

    deferred = run_dir / "Arm"
    deferred.mkdir()
    # Part-directory discovery intentionally ignores arbitrary scratch
    # directories. A genuinely attempted deferral has an initializer record.
    (deferred / "initializer_decision.json").write_text(
        json.dumps({"semantic_part": "Arm", "segment_id": 2}),
        encoding="utf-8",
    )
    (deferred / "part_deferral.json").write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-part-deferral.v1",
                "status": "deferred",
                "semantic_part": "Arm",
                "segment_id": 2,
                "faces_reserved": False,
                "revisit_required": False,
                "revisit_completed": True,
                "reason": "No unlocked face remains.",
                "reason_code": (
                    "no_consistent_unlocked_component_after_exhaustive_audit"
                ),
                # Deliberately hostile child statistics. The launcher must not
                # consume this claim when deciding whether retries can stop.
                "total_unlocked_face_count": 0,
                "component_dispositions": [],
            }
        ),
        encoding="utf-8",
    )

    fragment_ids = run_dir / "fragments" / "fragment_ids.u32le"
    fragment_ids.write_bytes(np.asarray([7, 8], dtype="<u4").tobytes())
    topology_digest = "sha256:" + "a" * 64
    target_prim_path = "/World/Mesh"
    (run_dir / "prepare" / "topology.json").write_text(
        json.dumps(
            {
                "source_face_count": 2,
                "source_sha256": staged_asset_sha256,
                "topology_digest": topology_digest,
                "target_prim_path": target_prim_path,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "fragments" / "fragment_manifest.json").write_text(
        json.dumps(
            {
                "source_face_count": 2,
                "source_sha256": staged_asset_sha256,
                "topology_digest": topology_digest,
                "target_prim_path": target_prim_path,
                "fragment_ids": str(fragment_ids),
                "fragment_ids_sha256": hashlib.sha256(
                    fragment_ids.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    latest_state = run_dir / "state" / "labels-001.u32le"
    latest_state.parent.mkdir(exist_ok=True)
    payload = np.asarray([1, 1], dtype="<u4").tobytes()
    latest_state.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_parent_terminal_promotion_source",
        lambda *args, **kwargs: (payload, latest_state, digest),
    )
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_launcher_source_topology",
        lambda *args, **kwargs: (2, topology_digest, target_prim_path),
    )
    return run_dir, staged_asset, staged_asset_sha256


def _terminal_deferral_observed_lock_seals(
    run_dir: Path,
) -> dict[str, dict[str, str]]:
    gate = json.loads(
        (run_dir / "Body" / "initializer_runtime_gate.json").read_text(encoding="utf-8")
    )
    return {"body": dict(gate["sealed_digests"])}


def test_terminal_deferral_failure_binds_launcher_verified_face_universe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, staged_asset, staged_asset_sha256 = _terminal_deferral_fixture(
        tmp_path, monkeypatch
    )

    receipt = mesh_segmentation_runner._terminal_deferral_failure_source(
        run_dir,
        target_parts=["Body", "Arm"],
        staged_asset=staged_asset,
        staged_asset_sha256=staged_asset_sha256,
        target_prim_path=None,
        launcher_observed_lock_seals=_terminal_deferral_observed_lock_seals(run_dir),
    )

    assert receipt is not None
    assert receipt["status"] == "failed"
    assert receipt["receipt_owner"] == "launcher"
    assert receipt["face_universe"]["unlocked_face_count"] == 0
    assert receipt["publication"]["partial_labels_published"] is False
    assert receipt["terminal_contract"]["terminal_validation_waived"] is False


def test_terminal_deferral_failure_ignores_child_zero_face_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hostile deferral cannot hide a mechanically unlocked source face."""

    run_dir, staged_asset, staged_asset_sha256 = _terminal_deferral_fixture(
        tmp_path, monkeypatch
    )
    latest_state = run_dir / "state" / "labels-001.u32le"
    payload = np.asarray([1, 0], dtype="<u4").tobytes()
    latest_state.write_bytes(payload)
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_parent_terminal_promotion_source",
        lambda *args, **kwargs: (
            payload,
            latest_state,
            hashlib.sha256(payload).hexdigest(),
        ),
    )

    assert (
        mesh_segmentation_runner._terminal_deferral_failure_source(
            run_dir,
            target_parts=["Body", "Arm"],
            staged_asset=staged_asset,
            staged_asset_sha256=staged_asset_sha256,
            target_prim_path=None,
            launcher_observed_lock_seals=_terminal_deferral_observed_lock_seals(
                run_dir
            ),
        )
        is None
    )


@pytest.mark.parametrize("attack", ["symlink", "hardlink"])
def test_terminal_deferral_failure_rejects_linked_child_deferral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    run_dir, staged_asset, staged_asset_sha256 = _terminal_deferral_fixture(
        tmp_path, monkeypatch
    )
    deferral = run_dir / "Arm" / "part_deferral.json"
    outside = tmp_path / "outside-deferral.json"
    outside.write_bytes(deferral.read_bytes())
    deferral.unlink()
    if attack == "symlink":
        try:
            deferral.symlink_to(outside)
        except OSError as exc:
            if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
                pytest.skip("Windows symlink privilege is unavailable")
            raise
    else:
        os.link(outside, deferral)

    assert (
        mesh_segmentation_runner._terminal_deferral_failure_source(
            run_dir,
            target_parts=["Body", "Arm"],
            staged_asset=staged_asset,
            staged_asset_sha256=staged_asset_sha256,
            target_prim_path=None,
            launcher_observed_lock_seals=_terminal_deferral_observed_lock_seals(
                run_dir
            ),
        )
        is None
    )


def test_terminal_deferral_failure_requires_launcher_observed_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, staged_asset, staged_asset_sha256 = _terminal_deferral_fixture(
        tmp_path, monkeypatch
    )
    gate_path = run_dir / "Body" / "initializer_runtime_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["locked_observed"] = False
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    assert (
        mesh_segmentation_runner._terminal_deferral_failure_source(
            run_dir,
            target_parts=["Body", "Arm"],
            staged_asset=staged_asset,
            staged_asset_sha256=staged_asset_sha256,
            target_prim_path=None,
            launcher_observed_lock_seals=_terminal_deferral_observed_lock_seals(
                run_dir
            ),
        )
        is None
    )


def test_terminal_deferral_failure_rejects_child_forged_lock_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid-looking on-disk gate is not live launcher provenance."""

    run_dir, staged_asset, staged_asset_sha256 = _terminal_deferral_fixture(
        tmp_path, monkeypatch
    )

    assert (
        mesh_segmentation_runner._terminal_deferral_failure_source(
            run_dir,
            target_parts=["Body", "Arm"],
            staged_asset=staged_asset,
            staged_asset_sha256=staged_asset_sha256,
            target_prim_path=None,
            launcher_observed_lock_seals={},
        )
        is None
    )


def test_terminal_deferral_failure_rejects_coherent_post_poll_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child cannot backfill a new corpus after the watcher's last poll."""

    run_dir, staged_asset, staged_asset_sha256 = _terminal_deferral_fixture(
        tmp_path, monkeypatch
    )
    observed_seals = _terminal_deferral_observed_lock_seals(run_dir)

    decision_path = run_dir / "Body" / "initializer_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["backfilled_after_launcher_poll"] = True
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    rewritten_digest = hashlib.sha256(decision_path.read_bytes()).hexdigest()

    # Keep the child-writable gate internally coherent with the rewritten
    # corpus. Only the watcher's frozen in-memory seal can detect this.
    gate_path = run_dir / "Body" / "initializer_runtime_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["sealed_digests"] = {
        decision_path.relative_to(run_dir).as_posix(): rewritten_digest
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    assert gate["sealed_digests"] != observed_seals["body"]

    assert (
        mesh_segmentation_runner._terminal_deferral_failure_source(
            run_dir,
            target_parts=["Body", "Arm"],
            staged_asset=staged_asset,
            staged_asset_sha256=staged_asset_sha256,
            target_prim_path=None,
            launcher_observed_lock_seals=observed_seals,
        )
        is None
    )


def test_terminal_deferral_stops_one_turn_without_publishing_partial_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    staged_asset = run_dir / "asset.usdc"
    staged_asset.write_bytes(b"asset")
    config = _loop_config(
        tmp_path,
        asset_path=staged_asset,
        output_dir=run_dir,
        target_semantic_parts=("Body", "Arm"),
    )
    calls = 0

    def fake_child(**kwargs: object) -> int:
        nonlocal calls
        calls += 1
        final_labels = run_dir / "state" / "final_labels.u32le"
        final_labels.parent.mkdir()
        final_labels.write_bytes(np.asarray([1, 1], dtype="<u4").tobytes())
        (run_dir / "raw" / "terminal_deferral_failure.json").write_text(
            '{"receipt_owner":"child"}\n', encoding="utf-8"
        )
        return 0

    launcher_receipt = {
        "schema_version": (
            mesh_segmentation_runner.TERMINAL_DEFERRAL_FAILURE_SCHEMA_VERSION
        ),
        "status": "failed",
        "receipt_owner": "launcher",
        "deferred_targets": [{"semantic_part": "Arm"}],
    }
    monkeypatch.setattr(mesh_segmentation_runner, "_run_child_agent", fake_child)
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_record_turn_usage",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_parent_terminal_promotion_source",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_terminal_deferral_failure_source",
        lambda *args, **kwargs: launcher_receipt,
    )
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_outstanding_work",
        lambda *args, **kwargs: mesh_segmentation_runner.OutstandingWork(
            description="one part remains",
            unlocked_parts=1,
            missing_artifacts=1,
        ),
    )

    result = mesh_segmentation_runner._run_isolated_child_agent(
        config=config,
        prompt="PROMPT",
        run_dir=run_dir,
        child_output_path=run_dir / "child-output.log",
        child_final_path=run_dir / "child-final.txt",
        prompt_image_inputs=[],
        bridge_artifact_prefix="mesh_segmentation",
        enable_workflow_watchdog=False,
    )

    assert result == 0
    assert calls == 1
    assert not (run_dir / "state" / "final_labels.u32le").exists()
    receipt = json.loads(
        (run_dir / "raw" / "terminal_deferral_failure.json").read_text(encoding="utf-8")
    )
    assert receipt["receipt_owner"] == "launcher"


def test_idle_continuation_threshold_tolerates_late_finishers(tmp_path: Path) -> None:
    """One idle turn must never end the loop.

    A recorded run reported the identical outstanding set for four consecutive
    turns and then locked every part on the fifth. The threshold has to sit
    well above that pattern or the loop fails runs that would have succeeded.
    """

    assert mesh_segmentation_runner._MAX_IDLE_CONTINUATION_TURNS > 4


def test_live_recognition_gate_approves_without_a_reviewer(tmp_path: Path) -> None:
    """Production wiring passes no reviewer, and must still be able to approve.

    Every other gate test injects its own `review_candidate`, so none covered
    how `_run_isolated_child_agent` actually builds this watchdog. With the
    independent reviewer retired the default is `None`, and the review branch
    used to raise, leaving `approved_count` at zero forever: recognition runs
    (`--no-input-vocabulary`) stalled on part 1 and failed terminal validation.
    """

    run_dir = tmp_path / "run"
    # No review_candidate: exactly what the launcher wires in production.
    gate = _make_recognition_sequential_watchdog(run_dir)
    queue_path = _write_live_recognition_queue(run_dir)
    assert gate() is None

    rev_light = run_dir / "hypotheses" / "rev-light"
    rev_light.mkdir(parents=True)
    light_labels = b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    light_path = rev_light / "face_labels.u32le"
    light_path.write_bytes(light_labels)
    light_validation = rev_light / "render_validation.json"
    light_validation.write_text("{}\n", encoding="utf-8")
    manifest = {
        "locks": [
            {
                "part_name": "marker_light",
                "segment_id": 1,
                "order": 0,
                "face_labels": str(light_path.relative_to(run_dir)),
                "face_labels_sha256": hashlib.sha256(light_labels).hexdigest(),
                "accepted_face_count": 1,
                "unresolved_issues": [],
                "validation_artifacts": [str(light_validation.relative_to(run_dir))],
                **_write_recognition_lock_evidence(
                    run_dir,
                    revision="rev-light",
                    target_name="marker_light",
                ),
            }
        ]
    }
    (run_dir / "part_lock_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["parts"][0]["status"] = "locked"
    queue["active_part"] = None
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    # Deliberately no polling. With no reviewer there is no review thread --
    # `start_review` is the only place this watchdog starts one, and it is
    # unreachable on this path -- so approval is synchronous inside `gate()`.
    # The sibling test polls because it injects a reviewer and does spawn a
    # thread. Asserting synchronously here is load-bearing: it fails if async
    # review is ever reintroduced into the reviewer-less path, which polling
    # would silently tolerate.
    threads_before = threading.active_count()
    assert gate() is None
    assert threading.active_count() == threads_before
    live_gate = json.loads(
        (run_dir / "live_sequential_gate.json").read_text(encoding="utf-8")
    )
    assert live_gate["approved_parts"] == ["marker_light"]
    assert live_gate["next_part"] == "body"

    # Terminal validation binds each lock to a passing review record, so the
    # gate must still write one -- labelled for the check that actually ran.
    reviews = json.loads(
        (run_dir / "part_review_manifest.json").read_text(encoding="utf-8")
    )["reviews"]
    assert [review["status"] for review in reviews] == ["passed"]
    assert reviews[0]["review_kind"] == "structural_sequential_gate"
    assert reviews[0]["part_name"] == "marker_light"
    assert reviews[0]["face_labels_sha256"] == hashlib.sha256(light_labels).hexdigest()


def test_outstanding_work_tracks_recognition_artifacts(tmp_path: Path) -> None:
    """A recognition run is not finished just because the base artifacts exist.

    `_outstanding_work` gates whether the loop grants another turn, while
    terminal validation independently requires the recognition artifacts. When
    only the targeted set was consulted, a recognition run was declared done as
    soon as the shared base artifacts landed, so the loop stopped early and the
    run then failed validation for files it was never given a turn to write.
    """

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in mesh_segmentation_runner.REQUIRED_FINAL_ARTIFACTS:
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    config = mesh_segmentation_runner.MeshSegmentationConfig(
        asset_path=tmp_path / "asset.usdc",
        output_dir=run_dir,
        repo_root=tmp_path,
        reference_images=(),
        target_semantic_parts=(),
    )

    outstanding = mesh_segmentation_runner._outstanding_work(run_dir, config=config)
    assert outstanding is not None
    assert "required artifacts missing" in outstanding.description

    for name in mesh_segmentation_runner.REQUIRED_RECOGNITION_FINAL_ARTIFACTS:
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    assert mesh_segmentation_runner._outstanding_work(run_dir, config=config) is None


def test_outstanding_work_counts_recognition_lock_progress(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _continuation_config(tmp_path, [])
    queue_path = run_dir / "part_work_queue.json"
    queue_path.write_text(
        json.dumps({"mode": "recognition", "parts": []}) + "\n",
        encoding="utf-8",
    )

    initial = mesh_segmentation_runner._outstanding_work(run_dir, config=config)
    assert initial is not None
    queue_path.write_text(
        json.dumps(
            {
                "mode": "recognition",
                "parts": [{"name": "marker_light", "status": "locked"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    progressed = mesh_segmentation_runner._outstanding_work(run_dir, config=config)

    assert progressed is not None
    assert progressed.progress_units == 1
    assert progressed.counts != initial.counts
    assert "1 recognition parts durably locked" in progressed.description


def test_outstanding_progress_accepts_terminal_artifact_completion() -> None:
    before = mesh_segmentation_runner.OutstandingWork(
        description="recognition artifacts remain",
        unlocked_parts=0,
        missing_artifacts=2,
        progress_units=0,
    )

    assert mesh_segmentation_runner._outstanding_made_durable_progress(before, None)


def test_targets_colliding_after_directory_folding_are_rejected(tmp_path: Path) -> None:
    """Two legal targets must not fold to one directory key.

    Gating keys parts by `_normalize_part_key`, where `a-b` and `a_b` become
    one key. `setdefault` then dropped the second from initializer gating while
    `_part_directories` aliased their evidence -- a requested target silently
    disappearing mid-run. Casefold uniqueness does not catch it, because the
    two names genuinely differ.
    """

    paths = _write_inputs(tmp_path)

    exit_code = main(
        [
            "mesh-segmentation",
            "run",
            "--asset",
            str(paths["asset"]),
            "--reference-dir",
            str(paths["references"]),
            "--repo-root",
            str(paths["repo_root"]),
            "--output-dir",
            str(tmp_path / "run"),
            "--target-semantic-part",
            "a-b",
            "--target-semantic-part",
            "a_b",
            *CANONICAL_PROVIDER_CLI_ARGS,
            "--dry-run",
        ]
    )

    assert exit_code != 0


def test_distinct_targets_that_do_not_collide_are_accepted(tmp_path: Path) -> None:
    """The guard must not reject an ordinary vocabulary."""

    paths = _write_inputs(tmp_path)

    exit_code = main(
        [
            "mesh-segmentation",
            "run",
            "--asset",
            str(paths["asset"]),
            "--reference-dir",
            str(paths["references"]),
            "--repo-root",
            str(paths["repo_root"]),
            "--output-dir",
            str(tmp_path / "run"),
            "--target-semantic-part",
            "Car Chassis",
            "--target-semantic-part",
            "Tire",
            *CANONICAL_PROVIDER_CLI_ARGS,
            "--dry-run",
        ]
    )

    assert exit_code == 0


def test_initializer_gate_refuses_to_reseal_after_lock(tmp_path: Path) -> None:
    """Rewriting sealed evidence after a part locks is backfill, and is fatal.

    Terminal validation only re-runs these same checks, so a corpus rewritten
    to be internally consistent passes it. This live gate is the only thing
    that can tell "observed while the work happened" from "assembled at the
    end", which is why -- unlike a pre-lock failure -- it cannot be advisory.
    """

    run_dir = tmp_path / "run"
    part_dir = run_dir / "support_frame"
    _direct_initializer_at(run_dir, part_name="Support Frame", part_dir=part_dir)

    watchdog = mesh_segmentation_runner._make_initialization_watchdog(run_dir)
    # The gate observes a stable identity on one pass and seals it on the next.
    assert watchdog() is None
    assert watchdog() is None
    gate_path = part_dir / "initializer_runtime_gate.json"
    assert json.loads(gate_path.read_text(encoding="utf-8"))["status"] == "accepted"

    # The fixture already wrote part_completion.json, so the part is locked
    # and its evidence sealed.
    assert (part_dir / "part_completion.json").is_file()
    assert watchdog() is None

    # Rewrite the sealed initializer *consistently* -- different bytes, but
    # decision, validation and rev-000 all regenerated together so the corpus
    # still validates cleanly. That is precisely what makes it undetectable
    # downstream and detectable only here.
    _write_valid_direct_initializer(
        run_dir,
        part_name="Support Frame",
        segment_id=1,
        parent_labels=b"\x00" * 12,
        revision_labels=b"\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00",
        part_dir=part_dir,
    )

    failure = watchdog()

    assert failure is not None
    assert failure.fatal
    assert "resealed initializer evidence" in str(failure)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "rejected"


def test_initializer_gate_tolerates_an_unchanged_recheck_after_lock(
    tmp_path: Path,
) -> None:
    """Re-validating identical sealed evidence is not backfill."""

    run_dir = tmp_path / "run"
    part_dir = run_dir / "support_frame"
    _direct_initializer_at(run_dir, part_name="Support Frame", part_dir=part_dir)

    watchdog = mesh_segmentation_runner._make_initialization_watchdog(run_dir)
    assert watchdog() is None
    assert watchdog() is None
    assert (part_dir / "part_completion.json").is_file()

    sealed = json.loads(
        (part_dir / "initializer_runtime_gate.json").read_text(encoding="utf-8")
    )

    for _ in range(3):
        assert watchdog() is None
    gate = json.loads(
        (part_dir / "initializer_runtime_gate.json").read_text(encoding="utf-8")
    )
    assert gate["status"] == "accepted"
    # Terminal validation compares these to decide the gate is stale. Asserting
    # only the status let a rewrite drop them, which made every locked part
    # look stale and silently skipped the gate-vs-initializer cross-check.
    for field in (
        "semantic_part",
        "decision",
        "decision_sha256",
        "segment_id",
        "rev_000_face_labels_sha256",
    ):
        assert gate.get(field) == sealed.get(field), field


def test_deleting_part_completion_does_not_reopen_the_reseal_path(
    tmp_path: Path,
) -> None:
    """Locking is remembered, not re-read from a file the child controls.

    Reading `part_completion.json`'s current presence let a child delete it,
    reseal the sealed evidence as though the part were still provisional, and
    recreate the record -- walking straight through the anti-backfill guard
    while leaving terminal validation a consistent corpus.
    """

    run_dir = tmp_path / "run"
    part_dir = run_dir / "tire"
    _direct_initializer_at(run_dir, part_name="Tire", part_dir=part_dir)
    watchdog = mesh_segmentation_runner._make_initialization_watchdog(run_dir)
    assert watchdog() is None
    assert watchdog() is None
    assert (part_dir / "part_completion.json").is_file()

    # Delete the lock record, then rewrite the sealed evidence consistently.
    (part_dir / "part_completion.json").unlink()
    _write_valid_direct_initializer(
        run_dir,
        part_name="Tire",
        segment_id=1,
        parent_labels=b"\x00" * 12,
        revision_labels=b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        part_dir=part_dir,
    )

    failure = watchdog()

    assert failure is not None
    assert failure.fatal
    assert "resealed initializer evidence" in str(failure)


def test_record_turn_usage_accumulates_across_turns(tmp_path: Path) -> None:
    """The launcher side of the cumulative-usage contract.

    The adapter is tested against this artifact, but nothing proved the runner
    writes it -- so the two halves could drift while both suites stayed green.
    """

    run_dir = tmp_path / "run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    totals: dict[str, int] = {}

    for turn, (inp, out) in enumerate([(100, 10), (250, 25)], start=1):
        (raw / "mesh_segmentation_result.json").write_text(
            json.dumps(
                {
                    "usage": {"input_tokens": inp, "output_tokens": out},
                    "items": [
                        {"type": "command_execution", "exit_code": 0},
                        {"type": "agent_message"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        mesh_segmentation_runner._record_turn_usage(
            run_dir,
            bridge_artifact_prefix="mesh_segmentation",
            attempt=turn,
            totals=totals,
        )

    payload = json.loads(
        (raw / "mesh_segmentation_usage_total.json").read_text(encoding="utf-8")
    )
    assert payload["input_tokens"] == 350
    assert payload["output_tokens"] == 35
    assert payload["total_tokens"] == 385
    assert payload["command_calls"] == 2
    assert payload["model_turn_count"] == 2
    assert payload["child_turn_count"] == 2
    # Each turn's evidence survives the next turn overwriting the live file.
    assert (raw / "mesh_segmentation_result.turn-1.json").is_file()
    assert (raw / "mesh_segmentation_result.turn-2.json").is_file()


def _loop_config(tmp_path: Path, **overrides) -> object:
    base = {
        "asset_path": tmp_path / "asset.usdc",
        "output_dir": tmp_path / "run",
        "repo_root": tmp_path,
        "reference_images": (),
        "child_timeout_seconds": 7200.0,
        "case_timeout_seconds": 21600.0,
        "iteration_budget": 12,
    }
    base.update(overrides)
    return mesh_segmentation_runner.MeshSegmentationConfig(**base)


def _drive_loop(monkeypatch, tmp_path: Path, config, *, clock: list[float]):
    """Run the real continuation loop, recording the config of every turn."""

    seen: list[float] = []

    def fake_child(*, config, **kwargs):  # noqa: A002 - mirrors the real signature
        seen.append(config.child_timeout_seconds)
        return 0

    monkeypatch.setattr(mesh_segmentation_runner, "_run_child_agent", fake_child)
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_record_turn_usage",
        lambda *args, **kwargs: None,
    )
    # Always outstanding, so only the budget or the iteration cap can stop it.
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_outstanding_work",
        lambda *args, **kwargs: mesh_segmentation_runner.OutstandingWork(
            description="1 of 2 target parts unlocked (Tire)",
            unlocked_parts=1,
            missing_artifacts=0,
        ),
    )
    ticks = iter(clock)
    monkeypatch.setattr(
        mesh_segmentation_runner.time, "monotonic", lambda: next(ticks, clock[-1])
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    mesh_segmentation_runner._run_isolated_child_agent(
        config=config,
        prompt="PROMPT",
        run_dir=run_dir,
        child_output_path=run_dir / "child-output.log",
        child_final_path=run_dir / "child-final.txt",
        prompt_image_inputs=[],
        bridge_artifact_prefix="mesh_segmentation",
        enable_workflow_watchdog=False,
    )
    return seen


def test_child_exit_flushes_memory_without_accepting_invalid_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "raw").mkdir()
    config = _loop_config(
        tmp_path,
        output_dir=run_dir,
        target_semantic_parts=("Plug",),
        iteration_budget=1,
    )
    memory = AgentMemory(run_id="exit-flush", memory_root=tmp_path / "memory")

    def fake_child(**kwargs: object) -> int:
        revision_dir = run_dir / "Plug" / "rev-000"
        selected_dir = revision_dir / "selected-only" / "renders"
        frontier_dir = revision_dir / "frontier-audit"
        selected_dir.mkdir(parents=True)
        frontier_dir.mkdir()
        labels = revision_dir / "face_labels.u32le"
        labels.write_bytes(b"\0\0\0\0")
        labels_time_ns = time.time_ns()
        os.utime(labels, ns=(labels_time_ns, labels_time_ns))
        frontier = frontier_dir / "frontier_audit.json"
        evidence_time_ns = labels_time_ns + 1_000_000_000
        _write_revision_checkpoint_evidence(
            revision_dir,
            selected_dir,
            frontier,
            evidence_time_ns=evidence_time_ns,
        )
        final_labels = run_dir / "state" / "final_labels.u32le"
        final_labels.parent.mkdir()
        final_labels.write_bytes(b"\0\0\0\0")
        return 0

    monkeypatch.setattr(mesh_segmentation_runner, "_run_child_agent", fake_child)
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_make_initialization_watchdog",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_record_turn_usage",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_outstanding_work",
        lambda *args, **kwargs: None,
    )

    result = mesh_segmentation_runner._run_isolated_child_agent(
        config=config,
        prompt="PROMPT",
        run_dir=run_dir,
        child_output_path=run_dir / "child-output.log",
        child_final_path=run_dir / "child-final.txt",
        prompt_image_inputs=[],
        bridge_artifact_prefix="mesh_segmentation",
        agent_memory=memory,
        agent_memory_remember=memory.remember,
    )

    assert result == 0
    assert memory.count_observations() == 1
    assert not (run_dir / "state" / "final_labels.u32le").exists()
    assert not (run_dir / "raw" / "final_labels_promotion.json").exists()


def test_continuation_loop_clamps_the_turn_that_would_overrun(
    tmp_path: Path, monkeypatch
) -> None:
    """A turn starting under the budget must not run a full timeout past it.

    Driven through the real loop: asserting on `dataclasses.replace` proved
    nothing, because deleting the clamp entirely left that test green.
    """

    config = _loop_config(tmp_path)
    # start, turn 1 at 0s, turn 2 at 20000s of a 21600s budget.
    timings = [0.0, 0.0, 20000.0, 21600.0]

    seen = _drive_loop(monkeypatch, tmp_path, config, clock=timings)

    assert seen[0] == 7200.0, "a turn with the whole budget left is not clamped"
    assert seen[1] == 1600.0, "the second turn must be clamped to what remains"


def test_continuation_loop_stops_once_the_case_budget_is_spent(
    tmp_path: Path, monkeypatch
) -> None:
    """The loop must stop granting turns, well short of the iteration budget."""

    config = _loop_config(tmp_path)
    timings = [0.0, 0.0, 21600.0, 21600.0, 21600.0]

    seen = _drive_loop(monkeypatch, tmp_path, config, clock=timings)

    assert len(seen) == 1, seen


def test_continuation_loop_reports_a_case_that_never_got_a_turn(
    tmp_path: Path, monkeypatch
) -> None:
    """Exhausting the budget before the first turn must not crash the launcher.

    `returncode` was bound only inside the loop, so breaking on attempt 1
    raised `UnboundLocalError` from the launcher itself -- losing the
    diagnostic the break had just printed.
    """

    config = _loop_config(tmp_path)
    # The budget is already spent when the loop first checks it.
    seen = _drive_loop(monkeypatch, tmp_path, config, clock=[0.0, 21600.0])

    assert seen == [], "no turn should have run"


def test_continuation_loop_runs_the_iteration_budget_when_uncapped(
    tmp_path: Path, monkeypatch
) -> None:
    """case_timeout_seconds=0 disables the cap, as the flag help promises."""

    config = _loop_config(tmp_path, case_timeout_seconds=0.0, iteration_budget=3)

    seen = _drive_loop(monkeypatch, tmp_path, config, clock=[0.0])

    assert len(seen) == 3
    assert set(seen) == {7200.0}, "an uncapped run must not clamp its turns"


def test_timed_out_turn_with_locked_progress_resumes_with_remaining_case_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timed-out child may resume only from a durable target lock advance."""

    config = _loop_config(
        tmp_path,
        target_semantic_parts=("Tire", "Window"),
        iteration_budget=2,
    )
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    locked = 0
    seen_timeouts: list[float] = []
    seen_prompts: list[str] = []

    def fake_outstanding(*args, **kwargs):
        if locked == 2:
            return None
        return mesh_segmentation_runner.OutstandingWork(
            description=f"{2 - locked} target parts remain",
            unlocked_parts=2 - locked,
            missing_artifacts=0,
        )

    def fake_child(*, config, prompt, child_output_path, **kwargs):  # noqa: A002
        nonlocal locked
        seen_timeouts.append(config.child_timeout_seconds)
        seen_prompts.append(prompt)
        child_output_path.write_text(f"turn {len(seen_prompts)}\n", encoding="utf-8")
        locked += 1
        if locked == 1:
            raise TimeoutError("child turn exceeded its deadline")
        return 0

    monkeypatch.setattr(mesh_segmentation_runner, "_outstanding_work", fake_outstanding)
    monkeypatch.setattr(mesh_segmentation_runner, "_run_child_agent", fake_child)
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_parent_terminal_promotion_source",
        lambda *args, **kwargs: None,
    )
    ticks = iter((0.0, 0.0, 20000.0))
    monkeypatch.setattr(
        mesh_segmentation_runner.time,
        "monotonic",
        lambda: next(ticks, 20000.0),
    )

    result = mesh_segmentation_runner._run_isolated_child_agent(
        config=config,
        prompt="PROMPT",
        run_dir=run_dir,
        child_output_path=run_dir / "child-output.log",
        child_final_path=run_dir / "child-final.txt",
        prompt_image_inputs=[],
        bridge_artifact_prefix="mesh_segmentation",
        enable_workflow_watchdog=False,
    )

    assert result == 0
    assert seen_timeouts == [7200.0, 1600.0]
    assert seen_prompts[1].startswith("Continuation turn 2.")
    assert (run_dir / "child-output.turn-1.log").read_text(encoding="utf-8") == (
        "turn 1\n"
    )
    timeout = json.loads(
        (run_dir / "raw" / "child_turn_timeout.turn-1.json").read_text(encoding="utf-8")
    )
    assert timeout["locked_target_count_before"] == 0
    assert timeout["locked_target_count_after"] == 1
    assert timeout["semantic_progress_detected"] is True


def test_timed_out_recognition_turn_resumes_after_durable_lock_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recognition locks count even though that mode has no requested targets."""

    config = _loop_config(
        tmp_path,
        target_semantic_parts=(),
        iteration_budget=2,
    )
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    locked = 0
    seen_prompts: list[str] = []

    def fake_outstanding(*args, **kwargs):
        if locked == 2:
            return None
        return mesh_segmentation_runner.OutstandingWork(
            description="recognition work remains",
            unlocked_parts=0,
            missing_artifacts=1,
            progress_units=locked,
        )

    def fake_child(*, prompt, child_output_path, **kwargs):
        nonlocal locked
        seen_prompts.append(prompt)
        child_output_path.write_text(f"turn {len(seen_prompts)}\n", encoding="utf-8")
        locked += 1
        if locked == 1:
            raise TimeoutError("recognition turn exceeded its deadline")
        return 0

    monkeypatch.setattr(mesh_segmentation_runner, "_outstanding_work", fake_outstanding)
    monkeypatch.setattr(mesh_segmentation_runner, "_run_child_agent", fake_child)
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_parent_terminal_promotion_source",
        lambda *args, **kwargs: None,
    )
    ticks = iter((0.0, 0.0, 20000.0))
    monkeypatch.setattr(
        mesh_segmentation_runner.time,
        "monotonic",
        lambda: next(ticks, 20000.0),
    )

    result = mesh_segmentation_runner._run_isolated_child_agent(
        config=config,
        prompt="PROMPT",
        run_dir=run_dir,
        child_output_path=run_dir / "child-output.log",
        child_final_path=run_dir / "child-final.txt",
        prompt_image_inputs=[],
        bridge_artifact_prefix="mesh_segmentation",
        enable_workflow_watchdog=False,
    )

    assert result == 0
    assert len(seen_prompts) == 2
    assert seen_prompts[1].startswith("Continuation turn 2.")
    timeout = json.loads(
        (run_dir / "raw" / "child_turn_timeout.turn-1.json").read_text(encoding="utf-8")
    )
    assert timeout["locked_target_count_before"] == 0
    assert timeout["locked_target_count_after"] == 1
    assert timeout["semantic_progress_detected"] is True


def test_zero_progress_timeout_stops_without_starting_another_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout cannot spend another turn on setup-only or idle activity."""

    config = _loop_config(
        tmp_path,
        target_semantic_parts=("Tire",),
        case_timeout_seconds=0.0,
    )
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    calls = 0

    def fake_child(*, child_output_path, **kwargs):
        nonlocal calls
        calls += 1
        child_output_path.write_text("timed out\n", encoding="utf-8")
        (run_dir / "part_plan.json").write_text("{}\n", encoding="utf-8")
        raise TimeoutError("child turn exceeded its deadline")

    monkeypatch.setattr(mesh_segmentation_runner, "_run_child_agent", fake_child)
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_parent_terminal_promotion_source",
        lambda *args, **kwargs: None,
    )

    progress_tracker = mesh_segmentation_runner.MeshSegmentationProgressTracker(
        run_dir,
        mesh_segmentation_runner.TraceWriter(run_dir),
    )
    with pytest.raises(TimeoutError, match="exceeded its deadline"):
        mesh_segmentation_runner._run_isolated_child_agent(
            config=config,
            prompt="PROMPT",
            run_dir=run_dir,
            child_output_path=run_dir / "child-output.log",
            child_final_path=run_dir / "child-final.txt",
            prompt_image_inputs=[],
            bridge_artifact_prefix="mesh_segmentation",
            enable_workflow_watchdog=False,
            progress_tracker=progress_tracker,
        )

    assert calls == 1
    timeout = json.loads(
        (run_dir / "raw" / "child_turn_timeout.turn-1.json").read_text(encoding="utf-8")
    )
    assert timeout["locked_target_count_before"] == 0
    assert timeout["locked_target_count_after"] == 0
    assert timeout["semantic_progress_detected"] is False
    assert timeout["last_observed_stage"] == "planning"
    assert timeout["last_observed_progress"] == (
        "Mesh-segmentation advanced to planning."
    )


def test_continuation_loop_never_trusts_child_writable_script_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _loop_config(tmp_path, case_timeout_seconds=0.0, iteration_budget=2)
    run_dir = tmp_path / "run"
    scripts_dir = run_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    trusted_script = scripts_dir / "prepare_mesh.py"
    trusted_script.write_text("print('trusted')\n", encoding="utf-8")
    observed: list[dict[str, str]] = []

    def fake_child(**kwargs: object) -> int:
        observed.append(dict(kwargs["trusted_script_digests"]))  # type: ignore[arg-type]
        if len(observed) == 1:
            trusted_script.write_text("print('changed')\n", encoding="utf-8")
            (scripts_dir / "shadow.py").write_text("VALUE = 1\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(mesh_segmentation_runner, "_run_child_agent", fake_child)
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_record_turn_usage",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        mesh_segmentation_runner,
        "_outstanding_work",
        lambda *args, **kwargs: mesh_segmentation_runner.OutstandingWork(
            description="one part remains",
            unlocked_parts=1,
            missing_artifacts=0,
        ),
    )

    mesh_segmentation_runner._run_isolated_child_agent(
        config=config,
        prompt="PROMPT",
        run_dir=run_dir,
        child_output_path=run_dir / "child-output.log",
        child_final_path=run_dir / "child-final.txt",
        prompt_image_inputs=[],
        bridge_artifact_prefix="mesh_segmentation",
        enable_workflow_watchdog=False,
    )

    assert observed == [{}, {}]


def test_lock_memory_survives_a_launcher_restart(tmp_path: Path) -> None:
    """A restart must not forget that a part locked.

    Held only in process memory, the guard could be walked past by deleting
    the completion record, waiting for a restart, and then resealing as though
    the part were still provisional.
    """

    run_dir = tmp_path / "run"
    part_dir = run_dir / "tire"
    _direct_initializer_at(run_dir, part_name="Tire", part_dir=part_dir)
    # Real lifecycle order: a part seals as soon as its decision and rev-000
    # agree, and only locks later when falsification completes. The fixture
    # writes part_completion.json up front, which is the one ordering where
    # the initial seal happens to record the lock -- and so the one ordering
    # in which this test could pass while the guard never fired in a real run.
    (part_dir / "part_completion.json").unlink()

    first = mesh_segmentation_runner._make_initialization_watchdog(run_dir)
    assert first() is None
    assert first() is None
    gate = json.loads(
        (part_dir / "initializer_runtime_gate.json").read_text(encoding="utf-8")
    )
    assert gate["locked_observed"] is False, "not locked yet"

    (part_dir / "part_completion.json").write_text(
        json.dumps({"semantic_part": "Tire"}), encoding="utf-8"
    )
    assert first() is None
    gate = json.loads(
        (part_dir / "initializer_runtime_gate.json").read_text(encoding="utf-8")
    )
    assert gate["locked_observed"] is True, "the lock must be recorded when it lands"

    # Child deletes the lock record; the launcher then restarts.
    (part_dir / "part_completion.json").unlink()
    restarted = mesh_segmentation_runner._make_initialization_watchdog(run_dir)
    assert restarted() is None

    _write_valid_direct_initializer(
        run_dir,
        part_name="Tire",
        segment_id=1,
        parent_labels=b"\x00" * 12,
        revision_labels=b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        part_dir=part_dir,
    )

    failure = restarted()

    assert failure is not None
    assert failure.fatal


def test_locked_part_losing_sealed_artifacts_is_fatal(tmp_path: Path) -> None:
    """Deleting sealed evidence must not reopen the provisional path.

    Dropping the seal when an artifact vanishes let a child remove one file,
    recreate a self-consistent corpus, and be accepted through the
    initial-seal path -- backfill by another route.
    """

    run_dir = tmp_path / "run"
    part_dir = run_dir / "tire"
    _direct_initializer_at(run_dir, part_name="Tire", part_dir=part_dir)
    watchdog = mesh_segmentation_runner._make_initialization_watchdog(run_dir)
    assert watchdog() is None
    assert watchdog() is None
    assert (part_dir / "part_completion.json").is_file()

    (part_dir / "rev-000" / "face_labels.u32le").unlink()

    failure = watchdog()

    assert failure is not None
    assert failure.fatal
    assert "lost sealed initializer artifacts" in str(failure)


def test_legacy_schema_acceptance_is_announced_once(capsys) -> None:
    """A compat branch that never says anything outlives its purpose.

    Accepting both spellings silently means nobody learns that something is
    still emitting the retired one, and removing the branch later breaks those
    producers without warning.
    """

    mesh_segmentation_runner._WARNED_LEGACY_SCHEMAS.clear()
    expected = "mesh-segmentation-initializer-decision.v1"
    legacy = "mesh-segmentation-v5-initializer-decision.v1"

    assert mesh_segmentation_runner._schema_matches(expected, expected)
    assert capsys.readouterr().err == "", "the current spelling must be silent"

    assert mesh_segmentation_runner._schema_matches(legacy, expected)
    first = capsys.readouterr().err
    assert legacy in first
    assert "scheduled for removal" in first

    # Once per schema, not once per artifact: a run reads many.
    assert mesh_segmentation_runner._schema_matches(legacy, expected)
    assert capsys.readouterr().err == ""

    assert not mesh_segmentation_runner._schema_matches("something-else.v1", expected)


def test_legacy_schema_tolerance_covers_only_renamed_schemas() -> None:
    """The shim must not accept spellings no producer ever emitted.

    Deriving the legacy name mechanically from any `mesh-segmentation-` schema
    also accepted `mesh-segmentation-v5-fragment-evidence.v1` and friends --
    schemas that never carried the infix -- widening the compatibility surface
    this branch promises to remove later.
    """

    mesh_segmentation_runner._WARNED_LEGACY_SCHEMAS.clear()

    for stem in ("initializer-decision", "falsification-review", "consistency-sheet"):
        expected = f"mesh-segmentation-{stem}.v1"
        assert mesh_segmentation_runner._schema_matches(
            f"mesh-segmentation-v5-{stem}.v1", expected
        ), stem

    # Prefixed schemas keep the same rule.
    assert mesh_segmentation_runner._schema_matches(
        "content-agents.mesh-segmentation-v5-initializer-runtime-gate.v1",
        "content-agents.mesh-segmentation-initializer-runtime-gate.v1",
    )

    for stem in ("fragment-evidence", "export", "comparison"):
        expected = f"mesh-segmentation-{stem}.v1"
        assert not mesh_segmentation_runner._schema_matches(
            f"mesh-segmentation-v5-{stem}.v1", expected
        ), stem
        assert mesh_segmentation_runner._schema_matches(expected, expected), stem


def test_skill_and_launcher_tolerate_the_same_schemas() -> None:
    """An asymmetry here is the exact bug this shim exists to remove.

    A spelling one side accepts and the other rejects makes an artifact
    readable by the launcher and unreadable by the tooling that re-validates
    it.
    """

    import importlib

    scripts_dir = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "content-workflow-mesh-segmentation"
        / "scripts"
    )
    sys.path.insert(0, str(scripts_dir))
    try:
        mesh_geometry = importlib.import_module("mesh_geometry")
    finally:
        sys.path.remove(str(scripts_dir))

    assert (
        mesh_geometry.RENAMED_SCHEMA_STEMS
        == mesh_segmentation_runner.RENAMED_SCHEMA_STEMS
    )

    mesh_segmentation_runner._WARNED_LEGACY_SCHEMAS.clear()
    for stem in sorted(mesh_geometry.RENAMED_SCHEMA_STEMS) + ["fragment-evidence"]:
        expected = f"mesh-segmentation-{stem}.v1"
        legacy = f"mesh-segmentation-v5-{stem}.v1"
        assert mesh_geometry.schema_matches(
            legacy, expected
        ) == mesh_segmentation_runner._schema_matches(legacy, expected), stem


def test_recording_the_lock_preserves_the_gate_identity(tmp_path: Path) -> None:
    """Persisting the lock must not rewrite the gate into a stub.

    `_write_initializer_runtime_gate` rebuilds the whole document from its
    metadata, so writing only the part name dropped decision, decision_sha256,
    segment_id and rev_000_face_labels_sha256 -- the exact fields terminal
    validation compares to decide the gate is stale. Every locked part then
    looked stale, took the watchdog-race path, and had its gate silently
    rewritten, so the cross-check never ran.

    This needs the real ordering: seal first, lock later. The shared fixture
    writes part_completion.json up front, where this branch never fires.
    """

    run_dir = tmp_path / "run"
    part_dir = run_dir / "tire"
    _direct_initializer_at(run_dir, part_name="Tire", part_dir=part_dir)
    (part_dir / "part_completion.json").unlink()

    watchdog = mesh_segmentation_runner._make_initialization_watchdog(run_dir)
    assert watchdog() is None
    assert watchdog() is None
    gate_path = part_dir / "initializer_runtime_gate.json"
    sealed = json.loads(gate_path.read_text(encoding="utf-8"))
    assert sealed["status"] == "accepted"

    (part_dir / "part_completion.json").write_text(
        json.dumps({"semantic_part": "Tire"}), encoding="utf-8"
    )
    assert watchdog() is None

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["locked_observed"] is True
    for field in (
        "semantic_part",
        "decision",
        "decision_sha256",
        "segment_id",
        "rev_000_face_labels_sha256",
    ):
        assert gate.get(field) == sealed.get(field), field
