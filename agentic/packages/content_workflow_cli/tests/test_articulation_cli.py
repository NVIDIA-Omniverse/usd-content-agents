# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the public articulation launcher."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from content_agent_workflows.articulation import (
    ArticulationRunState,
    ArticulationWorkflowError,
    ArticulationWorkflowRequest,
    ArtifactBinding,
    EmbeddedArticulationCapabilityLimits,
    StandaloneArticulationPreparation,
)
from content_agent_workflows.articulation.workflow import _source_identity
from content_agent_workflows.asset_composition import AssetCompositionStateError
from content_agent_workflows.common.domain_execution import (
    DomainExecutionContext,
    EmbeddedStageBinding,
    ExecutionArtifactBinding,
)
from content_agent_workflows.common.embedded_domain_decision import (
    ProducerIdentity,
    ProviderNeutralEvidenceRecord,
    canonical_json_digest,
)
from pydantic import SecretStr

import content_workflow_cli.runner as shared_runner
from content_workflow_cli import articulation_runner
from content_workflow_cli.cli import main
from content_workflow_cli.trace import UnsafeRunArtifactError

REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_ARTICULATION_SOURCE = (
    REPO_ROOT
    / "apps/usd_cli/internal/example_tasks/task-06-joints-drawer-caster/assets/input_unrigged.usda"
)
REAL_INPUT_CHILD_MODEL = "issue-1313-request-smoke-model"
REAL_INPUT_REASONING_EFFORT = "high"
REAL_INPUT_TRANSPORT_STOP = 73


@pytest.fixture(autouse=True)
def _parent_usd_cli_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    def start(**kwargs: object) -> SimpleNamespace:
        run_dir = Path(kwargs["run_dir"])
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        identity_path = raw_dir / "parent_usd_cli_session_test.json"
        identity_path.write_text(
            json.dumps({"run_dir": str(run_dir.resolve()), "launch_id": "test"}) + "\n",
            encoding="utf-8",
        )
        readiness_path = raw_dir / "ovrtx_probe.json"
        readiness_path.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            identity_path=identity_path,
            identity_sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest(),
            server_url="http://127.0.0.1:43210",
            session=SimpleNamespace(session_id="articulation-parent-session"),
            readiness=SimpleNamespace(artifact_path=readiness_path),
        )

    monkeypatch.setattr(articulation_runner, "start_parent_usd_cli_capability", start)
    monkeypatch.setattr(
        articulation_runner,
        "stop_parent_usd_cli_capability",
        lambda _capability: None,
    )
    monkeypatch.setattr(
        shared_runner.ParentUsdCliSessionIdentity,
        "model_validate_json",
        classmethod(
            lambda _cls, _payload: SimpleNamespace(
                run_dir=json.loads(_payload)["run_dir"],
                launch_id=json.loads(_payload)["launch_id"],
            )
        ),
    )
    monkeypatch.setattr(
        shared_runner,
        "verify_live_parent_usd_cli_daemon",
        lambda identity: shared_runner.ParentUsdCliConnection(
            server_url="http://127.0.0.1:43210",
            session_id="articulation-parent-session",
            authentication_token=SecretStr("parent-token"),
        ),
    )


def _result(status: str = "needs_review") -> SimpleNamespace:
    payload = {
        "status": status,
        "output_dir": "/tmp/articulation-run",
        "candidate_document_path": "/tmp/articulation-run/articulation_candidates.json",
        "scene_evidence_path": ("/tmp/articulation-run/scene_evidence/manifest.json"),
        "review_receipt_path": None,
        "approved_candidate_document_path": None,
        "output_asset_path": None,
        "diagnostics_path": None,
        "validation_result_path": None,
        "final_summary_path": "/tmp/articulation-run/final_summary.json",
        "review_required_candidate_ids": ("candidate_0001",),
        "message": f"simulated {status} outcome",
    }
    return SimpleNamespace(
        **payload,
        model_dump=lambda **_kwargs: payload,
    )


def _write_source_and_config(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "cabinet.usdz"
    source.write_bytes(b"fixture")
    config = tmp_path / "joint.yaml"
    config.write_text("project:\n  name: fixture\n", encoding="utf-8")
    return source, config


def _assert_bound_json_artifact(
    identity: dict[str, Any],
    expected_path: Path,
) -> dict[str, Any]:
    resolved = expected_path.resolve()
    payload = resolved.read_bytes()
    assert identity == {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    return json.loads(payload)


def _write_checkpointed_request(
    request: ArticulationWorkflowRequest,
    *,
    scene_evidence: bool,
    phase: str = "authoring",
    mode: Literal["batch", "interactive"] = "batch",
) -> None:
    run_dir = request.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "request.json"
    request_path.write_text(
        request.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    evidence_binding: ArtifactBinding | None = None
    if scene_evidence:
        evidence_path = run_dir / "scene_evidence" / "manifest.json"
        evidence_path.parent.mkdir(parents=True)
        evidence_path.write_text('{"evidence": "checkpointed"}\n', encoding="utf-8")
        evidence_binding = ArtifactBinding(
            path=str(evidence_path.resolve()),
            sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        )
    state = ArticulationRunState(
        mode=mode,
        phase=phase,
        request=ArtifactBinding(
            path=str(request_path.resolve()),
            sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
        ),
        source_asset=request.source_asset,
        source_sha256="1" * 64,
        source_dependency_bundle_sha256="2" * 64,
        backend_configuration_sha256="3" * 64,
        scene_evidence_configuration_sha256="4" * 64,
        scene_evidence=evidence_binding,
    )
    (run_dir / "checkpoint.json").write_text(
        state.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def test_articulation_run_cli_builds_public_request(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run(config: Any) -> SimpleNamespace:
        captured["config"] = config
        captured["request"], _ = articulation_runner._build_request(config)
        return _result()

    monkeypatch.setattr(
        "content_workflow_cli.cli.run_articulation_workflow",
        fake_run,
    )

    code = main(
        [
            "articulation",
            "run",
            "--usd",
            str(source),
            "--joint-config",
            str(joint_config),
            "--output-dir",
            str(tmp_path / "run"),
            "--intent",
            "Identify six drawers and write approved prismatic joints.",
            "--review-policy",
            "all",
            "--allowed-motion-type",
            "prismatic",
            "--allowed-motion-type",
            "revolute",
            "--allowed-motion-type",
            "prismatic",
            "--expected-candidate-count",
            "6",
            "--session-id",
            "cabinet-e2e",
            "--child-timeout",
            "0",
            "--json",
        ]
    )

    assert code == 0
    config = captured["config"]
    assert config.source_asset == source
    assert config.joint_config == joint_config
    assert config.output_dir == tmp_path / "run"
    assert config.review_policy == "all"
    assert config.allowed_motion_types == ("prismatic", "revolute")
    assert captured["request"].allowed_motion_types == ("prismatic", "revolute")
    assert config.expected_candidate_count == 6
    assert config.joint_session_id == "cabinet-e2e"
    assert config.child_timeout_seconds == 0
    assert json.loads(capsys.readouterr().out)["status"] == "needs_review"


def test_articulation_run_cli_accepts_zero_expected_candidates(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run(config: Any) -> SimpleNamespace:
        captured["config"] = config
        captured["request"], _ = articulation_runner._build_request(config)
        return _result("not_articulated")

    monkeypatch.setattr(
        "content_workflow_cli.cli.run_articulation_workflow",
        fake_run,
    )

    code = main(
        [
            "articulation",
            "run",
            "--usd",
            str(source),
            "--joint-config",
            str(joint_config),
            "--output-dir",
            str(tmp_path / "run"),
            "--intent",
            "Verify that this asset has no articulation candidates.",
            "--expected-candidate-count",
            "0",
            "--json",
        ]
    )

    assert code == 0
    assert captured["config"].expected_candidate_count == 0
    assert captured["request"].expected_candidate_count == 0


def test_articulation_run_cli_accepts_embedded_run_state(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    embedded_run_state = tmp_path / "asset-run" / "asset_run.json"
    captured: dict[str, Any] = {}

    def fake_run(config: Any) -> SimpleNamespace:
        captured["config"] = config
        return _result()

    monkeypatch.setattr(
        "content_workflow_cli.cli.run_articulation_workflow",
        fake_run,
    )

    code = main(
        [
            "articulation",
            "run",
            "--usd",
            str(source),
            "--joint-config",
            str(joint_config),
            "--output-dir",
            str(tmp_path / "run"),
            "--intent",
            "Identify reviewed drawer joints.",
            "--embedded-run-state",
            str(embedded_run_state),
            "--json",
        ]
    )

    assert code == 0
    assert captured["config"].embedded_run_state == embedded_run_state


def test_articulation_run_cli_accepts_provider_neutral_preparation_without_config(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, _joint_config = _write_source_and_config(tmp_path)
    embedded_run_state = tmp_path / "asset-run" / "asset_run.json"
    preparation = tmp_path / "articulation-preparation.json"
    preparation.write_text("{}\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_run(config: Any) -> SimpleNamespace:
        captured["config"] = config
        return _result("awaiting_decision")

    monkeypatch.setattr(
        "content_workflow_cli.cli.run_articulation_workflow",
        fake_run,
    )

    code = main(
        [
            "articulation",
            "run",
            "--usd",
            str(source),
            "--embedded-preparation",
            str(preparation),
            "--output-dir",
            str(tmp_path / "run"),
            "--intent",
            "Author the outer-owned canonical graph.",
            "--embedded-run-state",
            str(embedded_run_state),
            "--json",
        ]
    )

    assert code == 0
    assert captured["config"].joint_config is None
    assert captured["config"].embedded_preparation == preparation


def test_articulation_run_cli_treats_not_articulated_as_success(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    monkeypatch.setattr(
        "content_workflow_cli.cli.run_articulation_workflow",
        lambda _config: _result("not_articulated"),
    )

    code = main(
        [
            "articulation",
            "run",
            "--usd",
            str(source),
            "--joint-config",
            str(joint_config),
            "--output-dir",
            str(tmp_path / "run"),
            "--intent",
            "Confirm whether this asset is articulated.",
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "articulation not_articulated" in output
    assert "simulated not_articulated outcome" in output


def test_embedded_articulation_request_injects_exact_context(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    embedded_run_state = tmp_path / "asset-run" / "asset_run.json"
    outer_binding = ExecutionArtifactBinding(
        path=str(tmp_path / "outer.json"),
        sha256="1" * 64,
        size_bytes=23,
    )
    source_binding = ExecutionArtifactBinding(
        path=str(source.resolve()),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        size_bytes=source.stat().st_size,
    )
    context = DomainExecutionContext(
        domain="articulation",
        mode="embedded",
        reasoning_loop_owner="asset_coordinator",
        embedded_stage=EmbeddedStageBinding(
            outer_run_id="asset-run",
            outer_request=outer_binding,
            stage="articulation",
            stage_attempt=3,
            coordinator_plan=outer_binding,
            input_asset=source_binding,
            domain_run_root=str((tmp_path / "run").resolve()),
        ),
    )
    captured: dict[str, Any] = {}

    def fake_context(
        state_path: Path,
        *,
        domain: str,
        input_asset: Path,
        output_dir: Path,
    ) -> DomainExecutionContext:
        captured.update(
            state_path=state_path,
            domain=domain,
            input_asset=input_asset,
            output_dir=output_dir,
        )
        return context

    monkeypatch.setattr(
        articulation_runner,
        "build_embedded_domain_execution_context",
        fake_context,
    )
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "run",
        joint_config=joint_config,
        intent="Identify reviewed drawer joints.",
        embedded_run_state=embedded_run_state,
    )

    request, normalized = articulation_runner._build_request(config)

    assert normalized.embedded_run_state == embedded_run_state.resolve()
    assert captured == {
        "state_path": embedded_run_state.resolve(),
        "domain": "articulation",
        "input_asset": source.resolve(),
        "output_dir": (tmp_path / "run").resolve(),
    }
    assert request.execution_context == context


def test_provider_neutral_preparation_builds_standalone_child_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "cabinet.usda"
    source.write_text(
        '#usda 1.0\ndef Xform "Cabinet" {}\n',
        encoding="utf-8",
    )
    source_sha256, dependency_sha256 = _source_identity(str(source))
    capabilities = EmbeddedArticulationCapabilityLimits(
        canonical_output_evidence_required=True
    )
    provider = ProducerIdentity(
        producer_id="standalone-cli-inspection",
        role="evidence_provider",
        implementation="deterministic-cli-test",
        implementation_digest="1" * 64,
    )

    def record(
        evidence_id: str,
        evidence_type: Literal["inspection", "capability", "render"],
        facts: dict[str, Any],
    ) -> ProviderNeutralEvidenceRecord:
        return ProviderNeutralEvidenceRecord(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            status="available",
            summary=f"Exact {evidence_id}.",
            facts=facts,
        )

    preparation = StandaloneArticulationPreparation(
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=dependency_sha256,
        configuration_sha256=canonical_json_digest({"inspection": "cli-test"}),
        evidence_provider=provider,
        source_hierarchy=record(
            "source-hierarchy-inspection", "inspection", {"root": "/Cabinet"}
        ),
        source_members=record(
            "joint-source-member-inspection",
            "inspection",
            {"source_member_prims": ["/Cabinet"]},
        ),
        authoritative_owners=record(
            "joint-authoritative-owner-inspection",
            "inspection",
            {"authoritative_owner_prims": ["/Cabinet"]},
        ),
        capabilities=record(
            "joint-authoring-capabilities",
            "capability",
            capabilities.model_dump(mode="json"),
        ),
        renders=record("joint-render-inspection", "render", {"diagnostic_only": True}),
        scene=record("joint-scene-inspection", "inspection", {"scene_tool": "usd-cli"}),
    )
    preparation_path = tmp_path / "standalone-preparation.json"
    preparation_path.write_text(
        preparation.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "run",
        joint_config=None,
        embedded_preparation=preparation_path,
        intent="Author a provider-neutral standalone graph.",
    )

    request, normalized = articulation_runner._build_request(config)

    assert request.execution_context == DomainExecutionContext(
        domain="articulation",
        mode="standalone",
        reasoning_loop_owner="domain_child_agent",
    )
    assert normalized.joint_config is None
    assert preparation.proposal_status == "not_requested"
    assert preparation.proposal is None
    assert capabilities.provider_proposals_authority is False

    def forbidden_unselected_client(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("provider-neutral articulation invoked an unselected client")

    monkeypatch.setattr(
        articulation_runner,
        "JointAgentLocalClient",
        forbidden_unselected_client,
    )
    import joint_agent.api as joint_api
    from content_agent_workflows.articulation import proposal_provider

    monkeypatch.setattr(joint_api, "pipeline", forbidden_unselected_client)
    monkeypatch.setattr(
        proposal_provider.ArtifactJsonArticulationProposalProvider,
        "__init__",
        forbidden_unselected_client,
    )
    monkeypatch.setattr(
        proposal_provider.HttpJsonArticulationProposalProvider,
        "__init__",
        forbidden_unselected_client,
    )
    monkeypatch.setattr(
        "requests.sessions.Session.request",
        forbidden_unselected_client,
    )

    def provider_neutral_controller(
        selected_request: ArticulationWorkflowRequest,
        **kwargs: Any,
    ) -> Any:
        assert kwargs["client"].__class__.__name__ == "JointAgentGraphAuthoringClient"
        assert kwargs["scene_evidence_collector"] is None
        assert kwargs["preparation"] == preparation
        return articulation_runner._standalone_articulation_controller(
            selected_request,
            **kwargs,
        )

    result = articulation_runner._execute_articulation_controller(
        request,
        normalized,
        controller=provider_neutral_controller,
    )
    assert result.status == "awaiting_decision"
    identity = json.loads(
        (config.output_dir / "standalone_articulation_identity.json").read_text(
            encoding="utf-8"
        )
    )
    assert identity["default_provider_backend"] is None


def test_standalone_parent_does_not_relaunch_child_at_post_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request = SimpleNamespace(
        execution_context=SimpleNamespace(mode="standalone"),
    )
    config = SimpleNamespace(output_dir=run_dir)
    expected = _result("awaiting_post_review")

    monkeypatch.setattr(
        articulation_runner,
        "_write_articulation_agent_launcher",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        articulation_runner,
        "_write_articulation_observation_if_needed",
        lambda *_args, **_kwargs: None,
    )

    def execute(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        (run_dir / "checkpoint.json").write_text("{}\n", encoding="utf-8")
        return expected

    monkeypatch.setattr(
        articulation_runner,
        "_execute_articulation_controller",
        execute,
    )
    monkeypatch.setattr(
        articulation_runner.ArticulationRunState,
        "model_validate_json",
        classmethod(
            lambda _cls, _payload: SimpleNamespace(phase="awaiting_post_review")
        ),
    )

    result = articulation_runner._run_skill_routed_articulation_request_locked(
        request,  # type: ignore[arg-type]
        config,  # type: ignore[arg-type]
    )

    assert result is expected


def test_embedded_articulation_request_rejects_explicit_fixed_mode(
    tmp_path: Path,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "run",
        joint_config=joint_config,
        intent="Identify reviewed drawer joints.",
        embedded_run_state=tmp_path / "asset-run" / "asset_run.json",
        execution_mode=articulation_runner.ARTICULATION_EXECUTION_FIXED,
    )

    with pytest.raises(
        ValueError,
        match="Embedded Articulation execution cannot use the fixed compatibility mode",
    ):
        articulation_runner._build_request(config)


def test_articulation_agent_launcher_round_trips_tuple_fields(tmp_path: Path) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=run_dir,
        joint_config=joint_config,
        intent="Identify reviewed drawer joints.",
        allowed_motion_types=("prismatic",),
    )

    request, config = articulation_runner._build_request(config)
    articulation_runner._write_articulation_agent_launcher(
        run_dir,
        config,
        request=request,
    )
    loaded = articulation_runner._load_articulation_agent_launcher(
        run_dir,
        request=request,
    )

    assert loaded.allowed_motion_types == ("prismatic",)
    assert isinstance(loaded.allowed_motion_types, tuple)
    policy_path = articulation_runner._articulation_agent_launcher_policy_path(run_dir)
    assert policy_path.parent == run_dir.parent
    assert not policy_path.is_relative_to(run_dir)


def test_articulation_legacy_launchers_remain_loadable(tmp_path: Path) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request, config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=run_dir,
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
        )
    )
    launcher_path = articulation_runner._write_articulation_agent_launcher(
        run_dir,
        config,
        request=request,
    )
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    launcher["schema_version"] = (
        articulation_runner.ARTICULATION_LEGACY_AGENT_LAUNCHER_SCHEMA_VERSION
    )
    launcher["config"].pop("embedded_preparation")
    launcher_path.write_text(
        json.dumps(launcher, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    policy_path = articulation_runner._articulation_agent_launcher_policy_path(run_dir)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["launcher_sha256"] = hashlib.sha256(launcher_path.read_bytes()).hexdigest()
    policy_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = dict(request.metadata)
    legacy_metadata = dict(
        metadata[articulation_runner.ARTICULATION_LAUNCHER_METADATA_KEY]
    )
    legacy_metadata["schema_version"] = (
        articulation_runner.ARTICULATION_LEGACY_LAUNCHER_SCHEMA_VERSION
    )
    legacy_metadata.pop("embedded_preparation")
    legacy_metadata.pop("embedded_preparation_sha256")
    metadata[articulation_runner.ARTICULATION_LAUNCHER_METADATA_KEY] = legacy_metadata
    legacy_request = request.model_copy(update={"metadata": metadata})

    loaded_agent = articulation_runner._load_articulation_agent_launcher(
        run_dir,
        request=request,
    )
    loaded_metadata = articulation_runner._load_launcher_config(legacy_request)

    assert loaded_agent.joint_config == joint_config.resolve()
    assert loaded_agent.embedded_preparation is None
    assert loaded_metadata.joint_config == joint_config.resolve()
    assert loaded_metadata.embedded_preparation is None


def test_articulation_agent_launcher_uses_confined_reader_without_o_nofollow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = tmp_path / "launcher.json"
    launcher.write_text("{}\n", encoding="utf-8")
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    assert (
        articulation_runner._read_articulation_agent_file(
            launcher,
            label="Articulation agent launcher",
            max_bytes=1024,
        )
        == launcher.read_bytes()
    )


def test_articulation_agent_launcher_accepts_workflow_normalized_request(
    tmp_path: Path,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request, config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=run_dir,
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
        )
    )
    articulation_runner._write_articulation_agent_launcher(
        run_dir,
        config,
        request=request,
    )
    normalized = request.model_copy(
        update={
            "metadata": {
                **request.metadata,
                "content_agent_workflows.scene_evidence_required": True,
            }
        }
    )

    loaded = articulation_runner._load_articulation_agent_launcher(
        run_dir,
        request=normalized,
    )

    assert loaded == config


def test_articulation_agent_launcher_rejects_child_writable_tampering(
    tmp_path: Path,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request, config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=run_dir,
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
        )
    )
    launcher_path = articulation_runner._write_articulation_agent_launcher(
        run_dir,
        config,
        request=request,
    )
    payload = json.loads(launcher_path.read_text(encoding="utf-8"))
    payload["config"]["child_timeout_seconds"] = 0.5
    launcher_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="parent-owned policy"):
        articulation_runner._load_articulation_agent_launcher(
            run_dir,
            request=request,
        )


def test_articulation_agent_launcher_rejects_prepared_request_tampering(
    tmp_path: Path,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request, config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=run_dir,
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
        )
    )
    articulation_runner._write_articulation_agent_launcher(
        run_dir,
        config,
        request=request,
    )
    tampered_request = request.model_copy(
        update={"intent": "Ignore the bound request."}
    )

    with pytest.raises(ValueError, match="request digest"):
        articulation_runner._load_articulation_agent_launcher(
            run_dir,
            request=tampered_request,
        )


def test_articulation_run_rejects_mismatched_prepared_request_without_checkpoint(
    tmp_path: Path,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    request, config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=run_dir,
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
        )
    )
    run_dir.mkdir()
    tampered_request = request.model_copy(
        update={"intent": "Ignore the bound request."}
    )
    (run_dir / "articulation_agent_request.json").write_text(
        tampered_request.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="differs from the prepared child request"):
        articulation_runner._run_request(request, config)

    assert not articulation_runner._articulation_agent_launcher_policy_path(
        run_dir
    ).exists()


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("runner", "shell"),
        ("codex_sandbox_mode", "danger-full-access"),
        ("claude_permission_mode", "dontAsk"),
        ("claude_execution_mode", "subprocess"),
        ("child_timeout_seconds", -1.0),
        ("max_candidate_count", 257),
        ("dry_run", "false"),
    ],
)
def test_articulation_agent_launcher_rejects_parent_digest_bound_invalid_scalars(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request, config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=run_dir,
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
        )
    )
    launcher_path = articulation_runner._write_articulation_agent_launcher(
        run_dir,
        config,
        request=request,
    )
    payload = json.loads(launcher_path.read_text(encoding="utf-8"))
    payload["config"][field_name] = invalid_value
    launcher_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    policy_path = articulation_runner._articulation_agent_launcher_policy_path(run_dir)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["launcher_sha256"] = hashlib.sha256(launcher_path.read_bytes()).hexdigest()
    policy_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid Articulation agent launcher config"):
        articulation_runner._load_articulation_agent_launcher(
            run_dir,
            request=request,
        )


@pytest.mark.parametrize(
    "invalid_timeout",
    [True, "0", None, float("nan"), -1.0],
)
def test_articulation_stored_launcher_rejects_timeout_before_coercion(
    tmp_path: Path,
    invalid_timeout: object,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    request, _ = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=tmp_path / "run",
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
        )
    )
    metadata = dict(request.metadata)
    launcher_metadata = dict(
        metadata[articulation_runner.ARTICULATION_LAUNCHER_METADATA_KEY]
    )
    launcher_metadata["child_timeout_seconds"] = invalid_timeout
    metadata[articulation_runner.ARTICULATION_LAUNCHER_METADATA_KEY] = launcher_metadata
    tampered_request = request.model_copy(update={"metadata": metadata})

    with pytest.raises(ValueError, match="child_timeout_seconds"):
        articulation_runner._load_launcher_config(tampered_request)


def test_articulation_agent_config_rejects_inline_secret_before_persistence(
    tmp_path: Path,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    secret = "inline-articulation-secret-must-not-persist"
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "run",
        joint_config=joint_config,
        intent="Identify reviewed drawer joints.",
        codex_config={"provider": {"api_key": secret}},
    )

    with pytest.raises(ValueError, match="contains inline credential") as exc_info:
        articulation_runner._build_request(config)

    assert secret not in str(exc_info.value)
    assert not config.output_dir.exists()


def test_articulation_skill_routed_launcher_uses_one_child_and_embedded_uses_none(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    standalone_config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "standalone-run",
        joint_config=joint_config,
        intent="Identify reviewed drawer joints.",
        execution_mode=articulation_runner.ARTICULATION_EXECUTION_SKILL_ROUTED,
    )
    standalone_request, standalone_config = articulation_runner._build_request(
        standalone_config
    )
    calls: list[str] = []

    def launch_child(
        _request: ArticulationWorkflowRequest,
        _config: articulation_runner.ArticulationRunConfig,
    ) -> SimpleNamespace:
        calls.append("child")
        return _result()

    monkeypatch.setattr(
        articulation_runner,
        "_launch_articulation_child",
        launch_child,
    )
    monkeypatch.setattr(
        articulation_runner,
        "_fixed_articulation_controller",
        lambda *_args, **_kwargs: pytest.fail("default used fixed controller"),
    )
    result = articulation_runner._run_request(
        standalone_request,
        standalone_config,
    )
    assert result.status == "needs_review"
    assert calls == ["child"]

    embedded_state = tmp_path / "asset-run" / "asset_run.json"
    context = DomainExecutionContext(
        domain="articulation",
        mode="embedded",
        reasoning_loop_owner="asset_coordinator",
        embedded_stage=EmbeddedStageBinding(
            outer_run_id="asset-run",
            outer_request=ExecutionArtifactBinding(
                path=str(tmp_path / "outer.json"),
                sha256="1" * 64,
                size_bytes=23,
            ),
            stage="articulation",
            stage_attempt=1,
            coordinator_plan=ExecutionArtifactBinding(
                path=str(tmp_path / "plan.json"),
                sha256="2" * 64,
                size_bytes=19,
            ),
            input_asset=ExecutionArtifactBinding(
                path=str(source.resolve()),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                size_bytes=source.stat().st_size,
            ),
            domain_run_root=str((tmp_path / "embedded-run").resolve()),
        ),
    )
    monkeypatch.setattr(
        articulation_runner,
        "build_embedded_domain_execution_context",
        lambda *_args, **_kwargs: context,
    )
    embedded_config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "embedded-run",
        joint_config=joint_config,
        intent="Identify reviewed drawer joints.",
        embedded_run_state=embedded_state,
    )
    embedded_request, embedded_config = articulation_runner._build_request(
        embedded_config
    )

    def execute_embedded(
        _request: ArticulationWorkflowRequest,
        _config: articulation_runner.ArticulationRunConfig,
        *,
        controller: Any,
    ) -> SimpleNamespace:
        assert controller.func is articulation_runner._embedded_articulation_controller
        assert controller.keywords == {"config": _config}
        calls.append("embedded-step")
        return _result()

    monkeypatch.setattr(
        articulation_runner,
        "_execute_articulation_controller",
        execute_embedded,
    )
    monkeypatch.setattr(
        articulation_runner,
        "_write_articulation_observation_if_needed",
        lambda *_args, **_kwargs: None,
    )
    result = articulation_runner._run_request(embedded_request, embedded_config)
    assert result.status == "needs_review"
    assert calls == ["child", "embedded-step"]


def test_articulation_skill_routed_run_rejects_concurrent_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "run",
        joint_config=joint_config,
        intent="Identify reviewed drawer joints.",
    )
    request, config = articulation_runner._build_request(config)
    config.output_dir.mkdir()
    real_lock = articulation_runner._skill_routed_run_lock

    def no_wait_lock(run_dir: Path, *, domain: Literal["articulation", "texture"]):
        return real_lock(run_dir, domain=domain, timeout_seconds=0)

    monkeypatch.setattr(articulation_runner, "_skill_routed_run_lock", no_wait_lock)
    monkeypatch.setattr(
        articulation_runner,
        "_launch_articulation_child",
        lambda *_args, **_kwargs: pytest.fail("concurrent parent launched a child"),
    )

    with real_lock(config.output_dir, domain="articulation"):
        with pytest.raises(RuntimeError, match="another articulation skill-routed"):
            articulation_runner._run_request(request, config)


def test_articulation_parent_releases_lease_for_child_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    request, config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=tmp_path / "run",
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
        )
    )
    expected = _result("completed")
    child_acquired_lease = False

    monkeypatch.setattr(
        articulation_runner,
        "_run_skill_routed_articulation_request_locked",
        lambda *_args, **_kwargs: (
            articulation_runner._ARTICULATION_CHILD_LAUNCH_REQUIRED
        ),
    )

    def launch_child(
        _request: ArticulationWorkflowRequest,
        child_config: articulation_runner.ArticulationRunConfig,
    ) -> SimpleNamespace:
        nonlocal child_acquired_lease
        with articulation_runner._skill_routed_run_lock(
            child_config.output_dir,
            domain="articulation",
            timeout_seconds=0,
        ):
            child_acquired_lease = True
        return expected

    monkeypatch.setattr(
        articulation_runner,
        "_launch_articulation_child",
        launch_child,
    )

    assert articulation_runner._run_request(request, config) is expected
    assert child_acquired_lease is True
    assert not list(tmp_path.glob(".*skill-routed.lock"))


def test_articulation_parent_rechecks_after_waiting_for_child_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    request, config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=tmp_path / "run",
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
        )
    )
    config.output_dir.mkdir()
    expected = _result("completed")
    launch_started = Event()
    release_launch = Event()
    completed = False
    launch_count = 0
    results: list[SimpleNamespace] = []
    errors: list[BaseException] = []

    def advance(*_args: Any, **_kwargs: Any) -> Any:
        return (
            expected
            if completed
            else articulation_runner._ARTICULATION_CHILD_LAUNCH_REQUIRED
        )

    def launch_child(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        nonlocal completed, launch_count
        launch_count += 1
        launch_started.set()
        assert release_launch.wait(timeout=5)
        completed = True
        return expected

    def complete() -> None:
        try:
            results.append(
                articulation_runner._complete_articulation_child_launch(
                    articulation_runner._ARTICULATION_CHILD_LAUNCH_REQUIRED,
                    request,
                    config,
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    monkeypatch.setattr(
        articulation_runner,
        "_run_skill_routed_articulation_request_locked",
        advance,
    )
    monkeypatch.setattr(articulation_runner, "_launch_articulation_child", launch_child)

    first = Thread(target=complete)
    second = Thread(target=complete)
    first.start()
    assert launch_started.wait(timeout=5)
    second.start()
    release_launch.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert results == [expected, expected]
    assert launch_count == 1


def test_skill_routed_run_lock_rejects_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    outside_target = tmp_path / "outside"
    outside_target.mkdir()
    sentinel = outside_target / "sentinel.txt"
    sentinel.write_text("untouched\n", encoding="utf-8")
    run_dir.symlink_to(outside_target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="run lock safely"):
        with articulation_runner._skill_routed_run_lock(
            run_dir,
            domain="articulation",
            timeout_seconds=0,
        ):
            pytest.fail("unsafe lock was acquired")

    assert run_dir.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "untouched\n"


def test_skill_routed_child_session_lock_rejects_symlink(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside_target = tmp_path / "outside.lock"
    outside_target.write_text("untouched\n", encoding="utf-8")
    lock_path = tmp_path / ".run.articulation-child-session.lock"
    lock_path.symlink_to(outside_target)

    with pytest.raises(RuntimeError, match="child-session lock safely"):
        with shared_runner._skill_routed_child_session_lock(
            run_dir,
            domain="articulation",
            timeout_seconds=0,
        ):
            pytest.fail("unsafe child-session lock was acquired")

    assert lock_path.is_symlink()
    assert outside_target.read_text(encoding="utf-8") == "untouched\n"


@pytest.mark.skipif(
    os.name == "nt",
    reason="native Windows uses confined handles instead of O_NOFOLLOW",
)
def test_skill_routed_run_lock_requires_o_nofollow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.delattr(os, "O_NOFOLLOW")

    with pytest.raises(RuntimeError, match="requires O_NOFOLLOW support"):
        with articulation_runner._skill_routed_run_lock(
            run_dir,
            domain="articulation",
            timeout_seconds=0,
        ):
            pytest.fail("lock acquired without O_NOFOLLOW support")


def test_articulation_waiter_revalidates_run_before_parent_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "run",
        joint_config=joint_config,
        intent="Identify reviewed drawer joints.",
        dry_run=True,
    )
    request, config = articulation_runner._build_request(config)
    config.output_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    prechecked = Event()
    precheck_count = 0
    real_reject = articulation_runner._reject_unsafe_run_links

    def track_precheck(run_dir: Path, *, allow_missing: bool = False) -> None:
        nonlocal precheck_count
        real_reject(run_dir, allow_missing=allow_missing)
        precheck_count += 1
        if precheck_count == 2:
            prechecked.set()

    monkeypatch.setattr(
        articulation_runner,
        "_reject_unsafe_run_links",
        track_precheck,
    )
    outcomes: list[object] = []

    def run_waiter() -> None:
        try:
            outcomes.append(articulation_runner._run_request(request, config))
        except Exception as exc:  # noqa: BLE001 - assert the cross-thread failure
            outcomes.append(exc)

    with articulation_runner._skill_routed_run_lock(
        config.output_dir,
        domain="articulation",
    ):
        waiter = Thread(target=run_waiter)
        waiter.start()
        assert prechecked.wait(timeout=5)
        (config.output_dir / "prompts").symlink_to(
            outside,
            target_is_directory=True,
        )
    waiter.join(timeout=5)

    assert not waiter.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], UnsafeRunArtifactError)
    assert not (outside / "articulation_skill_routed.md").exists()


def test_articulation_child_prompt_is_compact_and_routes_atomic_skills(
    tmp_path: Path,
) -> None:
    request = ArticulationWorkflowRequest(
        source_asset="fixture.usda",
        output_dir=tmp_path / "run",
        intent="Identify reviewed drawer joints.",
    )

    prompt = articulation_runner._build_articulation_agent_prompt(request)

    assert len(prompt) < 4000
    for skill_name in (
        "content-articulation-inspection",
        "content-articulation-proposal",
        "content-articulation-review",
        "content-articulation-authoring",
    ):
        assert skill_name in prompt
    assert "articulation _agent-prepare" in prompt
    assert "articulation _agent-apply" in prompt
    assert "run_batch_articulation_workflow" not in prompt


def test_articulation_child_runtime_has_shared_launcher_identity(
    tmp_path: Path,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    request, config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=tmp_path / "run",
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
        )
    )
    request.output_dir.mkdir(parents=True)
    (request.output_dir / "articulation_agent_request.json").write_text(
        request.model_dump_json() + "\n",
        encoding="utf-8",
    )
    capability_inventory, domain_policy_bounds = (
        articulation_runner._prepare_articulation_child_launch_contract(request)
    )

    child_config = articulation_runner._articulation_child_runtime_config(
        config,
        capability_inventory=capability_inventory,
        domain_policy_bounds=domain_policy_bounds,
    )

    assert shared_runner._child_workflow_name(child_config) == "articulation.author"


def test_standalone_articulation_prompt_forbids_unbound_semantic_inference(
    tmp_path: Path,
) -> None:
    request = ArticulationWorkflowRequest(
        source_asset="fixture.usda",
        output_dir=tmp_path / "run",
        intent="Fail closed when an articulation axis is unresolved.",
        metadata={
            "content_agent_workflows.domain_execution_context": (
                DomainExecutionContext(
                    domain="articulation",
                    mode="standalone",
                    reasoning_loop_owner="domain_child_agent",
                ).model_dump(mode="json")
            )
        },
    )

    prompt = articulation_runner._build_articulation_agent_prompt(request)
    compact_prompt = " ".join(prompt.split())

    assert "Only observation-bound evidence is semantic authority" in compact_prompt
    assert "transforms, bounds, and authored properties" in compact_prompt
    assert "states the derivation" in compact_prompt
    assert "top-level `evidence_requirements` to a non-empty list" in compact_prompt
    assert "every `candidate_decisions` entry" in compact_prompt
    assert "`canonical_graph.groups`" in compact_prompt
    assert "`canonical_graph.memberships`" in compact_prompt
    assert "`canonical_graph.joints`" in compact_prompt
    assert "`canonical_graph.rigid_link_operations`" in compact_prompt
    assert "before the first `_agent-apply` call" in compact_prompt
    assert "Do not use web search" in compact_prompt
    assert "unbound geometry" in compact_prompt
    assert "labels alone" in compact_prompt
    assert "non-success cap receipt without source mutation" in compact_prompt
    assert "stop and return that status for the outer coordinator" in compact_prompt
    assert "The child must not render, bind output evidence" in compact_prompt
    assert "canonical OVRTX evidence, post-review, and terminal finalization" in (
        compact_prompt
    )
    assert "produce and bind current-run canonical OVRTX" not in compact_prompt
    assert "StandaloneArticulationPostReviewPatch" not in compact_prompt
    assert "_agent-finalize" not in compact_prompt


@pytest.mark.parametrize(
    "child_status", ["completed", "awaiting_post_review", "not_articulated"]
)
def test_articulation_child_result_is_reverified_from_checkpoint(
    tmp_path: Path,
    monkeypatch: Any,
    child_status: str,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request, config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=run_dir,
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
        )
    )
    verified = _result(status=child_status)
    calls: list[str] = []

    def run_child_agent(**_kwargs: Any) -> int:
        (run_dir / "final_summary.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "raw").mkdir(exist_ok=True)
        (run_dir / "raw" / "articulation_child_final.json").write_text(
            json.dumps({"status": child_status}) + "\n",
            encoding="utf-8",
        )
        calls.append("child")
        return 0

    def reverify(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        calls.append("reverify")
        return verified

    monkeypatch.setattr(articulation_runner, "run_child_agent", run_child_agent)
    monkeypatch.setattr(
        articulation_runner,
        "_execute_articulation_controller",
        reverify,
    )

    assert articulation_runner._launch_articulation_child(request, config) is verified
    assert calls == ["child", "reverify"]


def test_articulation_child_launch_directories_are_private_under_umask_0002(
    tmp_path: Path,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    request, config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=tmp_path / "run",
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
            dry_run=True,
        )
    )

    previous_umask = os.umask(0o002)
    try:
        result = articulation_runner._run_request(request, config)
    finally:
        os.umask(previous_umask)

    assert result.status == "conditional"
    for private_dir in (
        config.output_dir,
        config.output_dir / "raw",
        config.output_dir / "prompts",
    ):
        assert private_dir.stat().st_mode & 0o777 == 0o700


def test_articulation_child_releases_parent_capability_when_prompt_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request, config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=run_dir,
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
        )
    )
    released: list[object] = []
    monkeypatch.setattr(
        articulation_runner,
        "stop_parent_usd_cli_capability",
        released.append,
    )
    monkeypatch.setattr(
        articulation_runner,
        "parent_usd_cli_prompt_contract",
        lambda _capability: (_ for _ in ()).throw(RuntimeError("prompt failed")),
    )

    with pytest.raises(RuntimeError, match="prompt failed"):
        articulation_runner._launch_articulation_child(request, config)

    assert len(released) == 1


def test_articulation_child_does_not_retry_provider_after_blocked_exit(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request, config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=run_dir,
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
        )
    )

    def run_child_agent(**_kwargs: Any) -> int:
        (run_dir / "raw").mkdir(exist_ok=True)
        (run_dir / "raw" / "articulation_child_final.json").write_text(
            '{"status":"blocked","final_summary_path":null}\n',
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(articulation_runner, "run_child_agent", run_child_agent)
    monkeypatch.setattr(
        articulation_runner,
        "_execute_articulation_controller",
        lambda *_args, **_kwargs: pytest.fail("blocked child retried the provider"),
    )

    with pytest.raises(RuntimeError, match="without a reportable checkpoint"):
        articulation_runner._launch_articulation_child(request, config)

    events = [
        json.loads(line)
        for line in (run_dir / "trace" / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event_type"] == "workflow_failed"
    assert events[-1]["data"]["child_status"] == "blocked"


def test_articulation_domain_credential_names_are_removed_from_child_contract(
    tmp_path: Path,
) -> None:
    joint_config = tmp_path / "joint.yaml"
    joint_config.write_text(
        "provider:\n  api_key_env: JOINT_DOMAIN_TOKEN\n"
        "  nested: ${JOINT_SECONDARY_TOKEN}\n",
        encoding="utf-8",
    )

    assert articulation_runner._articulation_forbidden_environment_names(
        joint_config
    ) == ("JOINT_DOMAIN_TOKEN", "JOINT_SECONDARY_TOKEN")


def test_articulation_domain_credential_config_errors_are_contextual(
    tmp_path: Path,
) -> None:
    joint_config = tmp_path / "joint.yaml"
    joint_config.write_text("provider: [unterminated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to load Joint Agent config"):
        articulation_runner._articulation_forbidden_environment_names(joint_config)


@pytest.mark.parametrize(
    ("child_runner", "claude_execution_mode"),
    [
        (shared_runner.RUNNER_CODEX, shared_runner.CLAUDE_EXECUTION_SDK),
        (shared_runner.RUNNER_CLAUDE, shared_runner.CLAUDE_EXECUTION_SDK),
        (shared_runner.RUNNER_CLAUDE, shared_runner.CLAUDE_EXECUTION_CLI),
    ],
)
def test_real_input_public_articulation_launch_builds_confined_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child_runner: str,
    claude_execution_mode: str,
) -> None:
    for environment_name in tuple(os.environ):
        if environment_name.startswith("CLAUDE_CODE_USE_"):
            monkeypatch.delenv(environment_name)
    source = REAL_ARTICULATION_SOURCE
    if not source.is_file():
        pytest.skip("internal articulation fixture is absent from this checkout")
    assert source.is_file()
    joint_config = tmp_path / "joint.yaml"
    joint_config.write_text(
        "project:\n"
        "  name: real-input-request-smoke\n"
        "provider:\n"
        "  base_url: https://joint-domain.example/v1\n"
        "  api_key_env: JOINT_DOMAIN_TOKEN\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    observed_environments: list[dict[str, str]] = []

    def stop_at_reasoning_transport(**kwargs: Any) -> int:
        observed_environments.append(dict(kwargs["env"]))
        command = kwargs["command"]
        if command[0] == "node":
            request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            for key, payload in (
                ("child_final_path", '{"status":"ok"}\n'),
                ("items_path", "[]\n"),
                ("result_path", "{}\n"),
            ):
                output = Path(request[key])
                output.write_text(payload, encoding="utf-8")
                output.chmod(0o600)
        else:
            kwargs["log_stream"].write('{"result":"ok","usage":{}}\n')
            kwargs["log_stream"].flush()
        return REAL_INPUT_TRANSPORT_STOP

    def forbid_domain_execution(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("real-input request smoke crossed into domain execution")

    monkeypatch.setenv("JOINT_DOMAIN_TOKEN", "must-not-cross")
    monkeypatch.setattr(
        shared_runner,
        "_run_subprocess_with_timeout",
        stop_at_reasoning_transport,
    )
    monkeypatch.setattr(
        shared_runner,
        "_find_claude_cli_binary",
        lambda: "/usr/bin/true",
    )
    monkeypatch.setattr(
        articulation_runner,
        "_execute_articulation_controller",
        forbid_domain_execution,
    )

    with pytest.raises(
        RuntimeError,
        match=f"Articulation child agent exited with code {REAL_INPUT_TRANSPORT_STOP}",
    ):
        articulation_runner.run_articulation_workflow(
            articulation_runner.ArticulationRunConfig(
                repo_root=REPO_ROOT,
                source_asset=source,
                output_dir=run_dir,
                joint_config=joint_config,
                intent="Identify reviewed drawer joints in the real task asset.",
                runner=child_runner,
                model=REAL_INPUT_CHILD_MODEL,
                model_reasoning_effort=REAL_INPUT_REASONING_EFFORT,
                claude_execution_mode=claude_execution_mode,
            )
        )

    request = json.loads(
        (run_dir / "raw" / "articulation_skill_routed_request.json").read_text(
            encoding="utf-8"
        )
    )
    descriptor = request["child_launch"]
    assert request["workflow"] == "articulation.author"
    assert request["workflow_skill"] == "content-workflow-articulation"
    assert request["run_dir"] == str(run_dir.resolve())
    assert descriptor["artifacts"]["run_root"] == str(run_dir.resolve())
    assert descriptor["runner_identity"] == {
        "runner": child_runner,
        "model": REAL_INPUT_CHILD_MODEL,
        "model_reasoning_effort": REAL_INPUT_REASONING_EFFORT,
        "claude_execution_mode": (
            claude_execution_mode
            if child_runner == shared_runner.RUNNER_CLAUDE
            else None
        ),
    }
    assert descriptor["network_policy"] == {
        "schema_version": "content-agents.child-launch-network-policy.v1",
        "mode": "reasoning_transport_only",
        "tool_network_access": False,
        "allowed_hosts": ["127.0.0.1"],
    }
    from world_understanding.utils.credentials import (
        API_KEY_ENV_VAR_MAP,
        NIM_API_KEY_ENV_VARS,
    )

    expected_forbidden_environment_names = {
        "JOINT_DOMAIN_TOKEN",
        "NGC_API_KEY",
        "NVCF_API_KEY",
        "NVCF_RENDER_FUNCTION_ID",
    }
    for environment_names in API_KEY_ENV_VAR_MAP.values():
        expected_forbidden_environment_names.update(environment_names)
    expected_forbidden_environment_names.update(NIM_API_KEY_ENV_VARS)
    if (
        child_runner == shared_runner.RUNNER_CLAUDE
        and claude_execution_mode == shared_runner.CLAUDE_EXECUTION_SDK
    ):
        expected_forbidden_environment_names.remove("ANTHROPIC_API_KEY")
    assert descriptor["credential_policy"] == {
        "mode": "reasoning_transport_only",
        "forbidden_environment_names": sorted(expected_forbidden_environment_names),
    }
    capability_inventory = _assert_bound_json_artifact(
        descriptor["capability_inventory"],
        run_dir / "raw" / "articulation_child_capability_inventory.json",
    )
    domain_policy = _assert_bound_json_artifact(
        descriptor["domain_policy_bounds"],
        run_dir / "raw" / "articulation_child_domain_policy.json",
    )
    public_request = json.loads(
        (run_dir / "articulation_agent_request.json").read_text(encoding="utf-8")
    )
    assert public_request["source_asset"] == str(source.resolve())
    assert capability_inventory["selection_owner"] == "articulation-domain-wrapper"
    assert domain_policy["child_authority"] == {
        "domain_client": False,
        "domain_credentials": False,
        "domain_network": False,
        "renderer_network": False,
        "semantic_operation_selection": False,
    }
    serialized_request = json.dumps(request, sort_keys=True)
    assert "joint-domain.example" not in serialized_request
    assert "must-not-cross" not in serialized_request
    assert observed_environments
    assert all("JOINT_DOMAIN_TOKEN" not in env for env in observed_environments)


def test_embedded_articulation_request_propagates_binding_failure(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    output_dir = tmp_path / "run"

    def reject_context(*_args: Any, **_kwargs: Any) -> DomainExecutionContext:
        raise AssetCompositionStateError(
            "Embedded articulation input differs from the active stage handoff"
        )

    monkeypatch.setattr(
        articulation_runner,
        "build_embedded_domain_execution_context",
        reject_context,
    )
    code = main(
        [
            "articulation",
            "run",
            "--usd",
            str(source),
            "--joint-config",
            str(joint_config),
            "--output-dir",
            str(output_dir),
            "--intent",
            "Identify reviewed drawer joints.",
            "--embedded-run-state",
            str(tmp_path / "asset-run" / "asset_run.json"),
            "--repo-root",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "differs from the active stage handoff" in captured.err
    assert captured.out == ""
    assert not output_dir.exists()


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_articulation_run_cli_returns_nonzero_for_terminal_failure(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
    status: str,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    monkeypatch.setattr(
        "content_workflow_cli.cli.run_articulation_workflow",
        lambda _config: _result(status),
    )

    code = main(
        [
            "articulation",
            "run",
            "--usd",
            str(source),
            "--joint-config",
            str(joint_config),
            "--output-dir",
            str(tmp_path / "run"),
            "--intent",
            "Infer drawer joints.",
        ]
    )

    assert code == 1
    assert f"message: simulated {status} outcome" in capsys.readouterr().out


def test_articulation_run_cli_returns_persisted_backend_failure(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    source = tmp_path / "cabinet.usda"
    source.write_text('#usda 1.0\n\ndef Xform "Cabinet" {\n}\n', encoding="utf-8")
    joint_config = tmp_path / "joint.yaml"
    joint_config.write_text("project:\n  name: fixture\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    inference_calls: list[bool] = []

    class FailingClient:
        def configuration_sha256(
            self,
            _request: ArticulationWorkflowRequest,
        ) -> str:
            return "1" * 64

        def infer(
            self,
            _request: ArticulationWorkflowRequest,
            *,
            resume: bool,
            cancel_checker: Any = None,
        ) -> Any:
            del cancel_checker
            inference_calls.append(resume)
            raise RuntimeError("simulated backend failure")

    class EvidenceCollector:
        def configuration_sha256(
            self,
            _request: ArticulationWorkflowRequest,
        ) -> str:
            return "2" * 64

    monkeypatch.setattr(
        "content_agent_workflows.articulation.workflow._source_identity",
        lambda _source_asset: ("3" * 64, "4" * 64),
    )
    monkeypatch.setattr(
        articulation_runner,
        "JointAgentLocalClient",
        lambda *_args, **_kwargs: FailingClient(),
    )
    monkeypatch.setattr(
        articulation_runner,
        "_build_live_evidence_collector",
        lambda _timeout: EvidenceCollector(),
    )

    code = main(
        [
            "articulation",
            "run",
            "--execution-mode",
            "fixed",
            "--usd",
            str(source),
            "--joint-config",
            str(joint_config),
            "--output-dir",
            str(run_dir),
            "--intent",
            "Infer drawer joints.",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["message"] == (
        "Joint Agent inference failed: simulated backend failure"
    )
    assert (
        ArticulationRunState.model_validate_json(
            (run_dir / "checkpoint.json").read_bytes()
        ).phase
        == "failed"
    )

    resume_code = main(
        [
            "articulation",
            "resume",
            "--run-dir",
            str(run_dir),
        ]
    )
    resume_output = capsys.readouterr().out
    assert resume_code == 1
    assert "articulation failed:" in resume_output
    assert (
        "message: Joint Agent inference failed: simulated backend failure"
        in resume_output
    )
    assert inference_calls == [False]


def test_articulation_runner_does_not_replay_unpersisted_workflow_error(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    request = ArticulationWorkflowRequest(
        source_asset=str(source),
        output_dir=run_dir,
        intent="Infer drawer joints.",
    )
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=run_dir,
        joint_config=joint_config,
        intent=request.intent,
        execution_mode=articulation_runner.ARTICULATION_EXECUTION_FIXED,
    )
    calls = 0

    monkeypatch.setattr(
        articulation_runner,
        "JointAgentLocalClient",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        articulation_runner,
        "_build_live_evidence_collector",
        lambda _timeout: object(),
    )

    def fail_without_checkpoint(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise ArticulationWorkflowError("preflight failed")

    monkeypatch.setattr(
        articulation_runner,
        "run_batch_articulation_workflow",
        fail_without_checkpoint,
    )

    with pytest.raises(ArticulationWorkflowError, match="preflight failed"):
        articulation_runner._run_request(request, config)

    assert calls == 1


def test_articulation_run_cli_reports_unpersisted_failure_distinctly(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """An error with no durable result exits 2, not the durable-failure exit 1."""

    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"

    monkeypatch.setattr(
        articulation_runner,
        "JointAgentLocalClient",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        articulation_runner,
        "_build_live_evidence_collector",
        lambda _timeout: object(),
    )

    def fail_without_checkpoint(*_args: Any, **_kwargs: Any) -> Any:
        raise ArticulationWorkflowError("preflight failed")

    monkeypatch.setattr(
        articulation_runner,
        "run_batch_articulation_workflow",
        fail_without_checkpoint,
    )

    code = main(
        [
            "articulation",
            "run",
            "--execution-mode",
            "fixed",
            "--usd",
            str(source),
            "--joint-config",
            str(joint_config),
            "--output-dir",
            str(run_dir),
            "--intent",
            "Infer drawer joints.",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "preflight failed" in captured.err
    assert not (run_dir / "checkpoint.json").exists()


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_articulation_cli_rejects_non_positive_or_non_finite_timeout(
    timeout: str,
) -> None:
    assert (
        main(
            [
                "articulation",
                "resume",
                "--run-dir",
                "run",
                "--scene-tool-timeout",
                timeout,
            ]
        )
        == 2
    )


def test_articulation_review_cli_binds_decisions_and_resumes(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps({"candidate_0001": "accept"}),
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    def fake_review(
        run_dir: Path,
        decisions_path: Path,
        **kwargs: Any,
    ) -> SimpleNamespace:
        captured.update(
            run_dir=run_dir,
            decisions_path=decisions_path,
            kwargs=kwargs,
        )
        return _result("completed")

    monkeypatch.setattr(
        "content_workflow_cli.cli.review_articulation_workflow",
        fake_review,
    )

    code = main(
        [
            "articulation",
            "review",
            "--run-dir",
            str(tmp_path / "run"),
            "--decisions-json",
            str(decisions),
            "--reviewer",
            "asset-owner",
        ]
    )

    assert code == 0
    assert captured["run_dir"] == tmp_path / "run"
    assert captured["decisions_path"] == decisions
    assert captured["kwargs"]["reviewer"] == "asset-owner"
    assert "articulation completed" in capsys.readouterr().out


def test_articulation_graph_revision_cli_binds_public_operation(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    patch = tmp_path / "graph_revision_patch.json"
    patch.write_text('{"schema_version":"fixture"}\n', encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_revision(
        run_dir: Path,
        revision_patch_path: Path,
        **kwargs: Any,
    ) -> SimpleNamespace:
        captured.update(
            run_dir=run_dir,
            revision_patch_path=revision_patch_path,
            kwargs=kwargs,
        )
        return _result("needs_review")

    monkeypatch.setattr(
        "content_workflow_cli.cli.revise_articulation_graph_workflow",
        fake_revision,
    )

    code = main(
        [
            "articulation",
            "revise-graph",
            "--run-dir",
            str(tmp_path / "run"),
            "--revision-patch",
            str(patch),
        ]
    )

    assert code == 0
    assert captured["run_dir"] == tmp_path / "run"
    assert captured["revision_patch_path"] == patch
    assert captured["kwargs"]["scene_timeout_seconds"] == 60.0
    assert "articulation needs_review" in capsys.readouterr().out


def test_review_runner_builds_exact_receipt_before_resume(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "run",
        joint_config=joint_config,
        intent="Infer drawer joints.",
    )
    request, _ = articulation_runner._build_request(config)
    _write_checkpointed_request(request, scene_evidence=True)
    run_dir = request.output_dir
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps(
            {
                "decisions": {
                    "candidate_0001": "accept",
                    "candidate_0002": "reject",
                }
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, Any]] = []
    expected = _result("completed")

    def fake_receipt(
        output_dir: Path,
        decisions: dict[str, str],
        *,
        reviewer: str,
    ) -> None:
        calls.append(("receipt", (output_dir, decisions, reviewer)))

    def fake_resume(output_dir: Path, **kwargs: Any) -> SimpleNamespace:
        calls.append(("resume", (output_dir, kwargs)))
        return expected

    monkeypatch.setattr(
        articulation_runner,
        "build_articulation_review_receipt",
        fake_receipt,
    )
    monkeypatch.setattr(
        articulation_runner,
        "_resume_articulation_workflow_locked",
        fake_resume,
    )

    actual = articulation_runner.review_articulation_workflow(
        run_dir,
        decisions_path,
        reviewer="asset-owner",
        repo_root=tmp_path,
    )

    assert actual is expected
    assert calls[0] == (
        "receipt",
        (
            run_dir,
            {
                "candidate_0001": "accept",
                "candidate_0002": "reject",
            },
            "asset-owner",
        ),
    )
    assert calls[1][0] == "resume"


def test_review_waits_for_parent_lease_before_publishing_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "run",
        joint_config=joint_config,
        intent="Infer drawer joints.",
    )
    request, _ = articulation_runner._build_request(config)
    _write_checkpointed_request(request, scene_evidence=True)
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps({"candidate_0001": "accept"}),
        encoding="utf-8",
    )
    ready_to_lock = Event()
    receipt_published = Event()
    expected = _result("completed")
    outcomes: list[object] = []

    def load_decisions(_path: str | Path) -> dict[str, str]:
        ready_to_lock.set()
        return {"candidate_0001": "accept"}

    def publish_receipt(*_args: Any, **_kwargs: Any) -> None:
        receipt_published.set()

    monkeypatch.setattr(articulation_runner, "_load_decisions", load_decisions)
    monkeypatch.setattr(
        articulation_runner,
        "build_articulation_review_receipt",
        publish_receipt,
    )
    monkeypatch.setattr(
        articulation_runner,
        "_resume_articulation_workflow_locked",
        lambda *_args, **_kwargs: expected,
    )

    def review_waiter() -> None:
        try:
            outcomes.append(
                articulation_runner.review_articulation_workflow(
                    request.output_dir,
                    decisions_path,
                    reviewer="asset-owner",
                    repo_root=tmp_path,
                )
            )
        except Exception as exc:  # noqa: BLE001 - assert the cross-thread failure
            outcomes.append(exc)

    with articulation_runner._skill_routed_run_lock(
        request.output_dir,
        domain="articulation",
    ):
        waiter = Thread(target=review_waiter)
        waiter.start()
        assert ready_to_lock.wait(timeout=5)
        assert not receipt_published.wait(timeout=0.1)
    waiter.join(timeout=5)

    assert not waiter.is_alive()
    assert receipt_published.is_set()
    assert outcomes == [expected]


@pytest.mark.parametrize("tampered_artifact", ["request", "checkpoint"])
def test_review_runner_verifies_integrity_before_receipt(
    tmp_path: Path,
    monkeypatch: Any,
    tampered_artifact: str,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "run",
        joint_config=joint_config,
        intent="Infer drawer joints.",
    )
    request, _ = articulation_runner._build_request(config)
    _write_checkpointed_request(request, scene_evidence=True)

    if tampered_artifact == "request":
        artifact_path = request.output_dir / "request.json"
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        payload["intent"] = "Tampered intent."
    else:
        artifact_path = request.output_dir / "checkpoint.json"
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        payload["request"]["sha256"] = "0" * 64
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps({"candidate_0001": "accept"}),
        encoding="utf-8",
    )
    side_effects: list[str] = []
    monkeypatch.setattr(
        articulation_runner,
        "_load_decisions",
        lambda *_args, **_kwargs: side_effects.append("decisions"),
    )
    monkeypatch.setattr(
        articulation_runner,
        "build_articulation_review_receipt",
        lambda *_args, **_kwargs: side_effects.append("receipt"),
    )
    monkeypatch.setattr(
        articulation_runner,
        "resume_articulation_workflow",
        lambda *_args, **_kwargs: side_effects.append("resume"),
    )

    with pytest.raises(ValueError, match="request digest differs"):
        articulation_runner.review_articulation_workflow(
            request.output_dir,
            decisions_path,
            reviewer="asset-owner",
            repo_root=tmp_path,
        )

    assert side_effects == []


@pytest.mark.parametrize(
    ("mode", "cli_owned", "message"),
    [
        (
            "interactive",
            True,
            "requires a content-workflow-cli batch run",
        ),
        (
            "batch",
            False,
            "was not created by content-workflow-cli",
        ),
    ],
)
def test_review_runner_rejects_non_cli_ownership_before_receipt(
    tmp_path: Path,
    monkeypatch: Any,
    mode: Literal["batch", "interactive"],
    cli_owned: bool,
    message: str,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    if cli_owned:
        config = articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=tmp_path / "run",
            joint_config=joint_config,
            intent="Infer drawer joints.",
        )
        request, _ = articulation_runner._build_request(config)
    else:
        request = ArticulationWorkflowRequest(
            source_asset=str(source),
            output_dir=tmp_path / "run",
            intent="Infer drawer joints.",
        )
    _write_checkpointed_request(
        request,
        scene_evidence=True,
        phase="needs_review",
        mode=mode,
    )
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps({"candidate_0001": "accept"}),
        encoding="utf-8",
    )
    side_effects: list[str] = []
    monkeypatch.setattr(
        articulation_runner,
        "_load_decisions",
        lambda *_args, **_kwargs: side_effects.append("decisions"),
    )
    monkeypatch.setattr(
        articulation_runner,
        "build_articulation_review_receipt",
        lambda *_args, **_kwargs: side_effects.append("receipt"),
    )
    monkeypatch.setattr(
        articulation_runner,
        "resume_articulation_workflow",
        lambda *_args, **_kwargs: side_effects.append("resume"),
    )

    with pytest.raises(ValueError, match=message):
        articulation_runner.review_articulation_workflow(
            request.output_dir,
            decisions_path,
            reviewer="asset-owner",
            repo_root=tmp_path,
        )

    assert side_effects == []
    assert not (request.output_dir / "review_receipt.json").exists()


def test_articulation_runner_always_uses_live_usd_cli_collector(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    request = ArticulationWorkflowRequest(
        source_asset=str(source),
        output_dir=run_dir,
        intent="Infer drawer joints.",
    )
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=run_dir,
        joint_config=joint_config,
        intent=request.intent,
        execution_mode=articulation_runner.ARTICULATION_EXECUTION_FIXED,
    )
    collector = object()
    client = object()
    calls: dict[str, Any] = {}

    monkeypatch.setattr(
        articulation_runner,
        "_build_live_evidence_collector",
        lambda timeout: (
            calls.setdefault("collector_timeout", timeout),
            collector,
        )[1],
    )
    monkeypatch.setattr(
        articulation_runner,
        "JointAgentLocalClient",
        lambda *args, **kwargs: (
            calls.setdefault("client_args", (args, kwargs)),
            client,
        )[1],
    )

    def fake_batch(
        actual_request: ArticulationWorkflowRequest,
        **kwargs: Any,
    ) -> SimpleNamespace:
        calls["batch"] = (actual_request, kwargs)
        return _result()

    monkeypatch.setattr(
        articulation_runner,
        "run_batch_articulation_workflow",
        fake_batch,
    )

    result = articulation_runner._run_request(request, config)

    assert result.status == "needs_review"
    assert calls["collector_timeout"] == 60.0
    assert calls["batch"][1] == {
        "client": client,
        "scene_evidence_collector": collector,
    }


def test_live_usd_cli_collector_uses_scene_timeout() -> None:
    collector = articulation_runner._build_live_evidence_collector(300.0)

    assert collector.timeout == 300.0


def test_articulation_resume_verifies_request_before_launcher_side_effects(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "run",
        joint_config=joint_config,
        intent="Infer drawer joints.",
        execution_mode=articulation_runner.ARTICULATION_EXECUTION_FIXED,
    )
    request, _ = articulation_runner._build_request(config)
    _write_checkpointed_request(request, scene_evidence=True)
    request_path = request.output_dir / "request.json"
    tampered = json.loads(request_path.read_text(encoding="utf-8"))
    tampered["metadata"]["content_workflow_cli"]["child_timeout_seconds"] = 0.5
    request_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    side_effects: list[str] = []

    with pytest.raises(ValueError, match="request digest differs"):
        articulation_runner.resume_articulation_workflow(
            request.output_dir,
            repo_root=tmp_path,
        )

    assert side_effects == []


def test_articulation_resume_retains_stored_embedded_context(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    binding = ExecutionArtifactBinding(
        path=str(source.resolve()),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        size_bytes=source.stat().st_size,
    )
    context = DomainExecutionContext(
        domain="articulation",
        mode="embedded",
        reasoning_loop_owner="asset_coordinator",
        embedded_stage=EmbeddedStageBinding(
            outer_run_id="asset-run",
            outer_request=binding,
            stage="articulation",
            stage_attempt=1,
            coordinator_plan=binding,
            input_asset=binding,
            domain_run_root=str((tmp_path / "run").resolve()),
        ),
    )
    monkeypatch.setattr(
        articulation_runner,
        "build_embedded_domain_execution_context",
        lambda *_args, **_kwargs: context,
    )
    request, _ = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=tmp_path / "run",
            joint_config=joint_config,
            intent="Identify reviewed drawer joints.",
            embedded_run_state=tmp_path / "asset-run" / "asset_run.json",
        )
    )
    _write_checkpointed_request(request, scene_evidence=True)
    captured: dict[str, Any] = {}
    expected = _result("completed")

    def fake_run(
        stored_request: ArticulationWorkflowRequest,
        config: articulation_runner.ArticulationRunConfig,
    ) -> SimpleNamespace:
        captured.update(request=stored_request, config=config)
        return expected

    def reject_context_rebuild(*_args: Any, **_kwargs: Any) -> DomainExecutionContext:
        raise AssertionError("resume must not rebuild embedded context")

    monkeypatch.setattr(
        articulation_runner,
        "_run_skill_routed_articulation_request_locked",
        fake_run,
    )
    monkeypatch.setattr(
        articulation_runner,
        "build_embedded_domain_execution_context",
        reject_context_rebuild,
    )

    actual = articulation_runner.resume_articulation_workflow(
        request.output_dir,
        repo_root=tmp_path,
    )

    assert actual is expected
    assert captured["request"].execution_context == context
    assert (
        captured["config"].embedded_run_state
        == (tmp_path / "asset-run" / "asset_run.json").resolve()
    )


def test_articulation_resume_skips_scene_collection_after_evidence_checkpoint(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "run",
        joint_config=joint_config,
        intent="Infer drawer joints.",
        execution_mode=articulation_runner.ARTICULATION_EXECUTION_FIXED,
    )
    request, _ = articulation_runner._build_request(config)
    _write_checkpointed_request(request, scene_evidence=True)
    calls: dict[str, Any] = {}
    client = object()
    collector = object()
    monkeypatch.setattr(
        articulation_runner,
        "JointAgentLocalClient",
        lambda *_args, **_kwargs: client,
    )
    monkeypatch.setattr(
        articulation_runner,
        "_build_live_evidence_collector",
        lambda _timeout: collector,
    )

    def fake_batch(
        actual_request: ArticulationWorkflowRequest,
        **kwargs: Any,
    ) -> SimpleNamespace:
        calls["batch"] = (actual_request, kwargs)
        return _result("completed")

    monkeypatch.setattr(
        articulation_runner,
        "run_batch_articulation_workflow",
        fake_batch,
    )

    result = articulation_runner.resume_articulation_workflow(
        request.output_dir,
        repo_root=tmp_path,
    )

    assert result.status == "completed"
    assert calls["batch"] == (
        request,
        {
            "client": client,
            "scene_evidence_collector": collector,
        },
    )


def test_articulation_resume_recovers_uncheckpointed_local_evidence(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "run",
        joint_config=joint_config,
        intent="Infer drawer joints.",
        execution_mode=articulation_runner.ARTICULATION_EXECUTION_FIXED,
    )
    request, _ = articulation_runner._build_request(config)
    _write_checkpointed_request(
        request,
        scene_evidence=False,
        phase="collecting_evidence",
    )
    evidence_path = request.output_dir / "scene_evidence" / "manifest.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text('{"evidence": "uncheckpointed"}\n', encoding="utf-8")

    calls: dict[str, Any] = {}
    client = object()
    collector = object()
    monkeypatch.setattr(
        articulation_runner,
        "JointAgentLocalClient",
        lambda *_args, **_kwargs: client,
    )
    monkeypatch.setattr(
        articulation_runner,
        "_build_live_evidence_collector",
        lambda _timeout: collector,
    )

    def fake_batch(
        actual_request: ArticulationWorkflowRequest,
        **kwargs: Any,
    ) -> SimpleNamespace:
        calls["batch"] = (actual_request, kwargs)
        return _result("completed")

    monkeypatch.setattr(
        articulation_runner,
        "run_batch_articulation_workflow",
        fake_batch,
    )

    result = articulation_runner._run_request(request, config)

    assert result.status == "completed"
    assert calls["batch"] == (
        request,
        {
            "client": client,
            "scene_evidence_collector": collector,
        },
    )


def test_articulation_run_rejects_symlinked_output_dir(tmp_path: Path) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_run = tmp_path / "linked-run"
    linked_run.symlink_to(outside, target_is_directory=True)
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=linked_run,
        joint_config=joint_config,
        intent="Infer drawer joints.",
    )

    with pytest.raises(RuntimeError, match="must resolve without traversing symlinks"):
        articulation_runner.run_articulation_workflow(config)

    assert list(outside.iterdir()) == []


def test_articulation_run_revalidates_output_before_launcher_write(
    tmp_path: Path,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=run_dir,
        joint_config=joint_config,
        intent="Infer drawer joints.",
    )
    request, normalized = articulation_runner._build_request(config)
    run_dir.mkdir()
    outside_launcher = tmp_path / "outside-launcher.json"
    outside_launcher.write_text("outside\n", encoding="utf-8")
    (run_dir / "articulation_agent_launcher.json").symlink_to(outside_launcher)

    with pytest.raises(RuntimeError, match="symlinks are not allowed"):
        articulation_runner._run_request(request, normalized)

    assert outside_launcher.read_text(encoding="utf-8") == "outside\n"


@pytest.mark.parametrize("operation", ["resume", "review"])
def test_articulation_existing_run_rejects_symlinked_directory(
    tmp_path: Path,
    operation: str,
) -> None:
    actual_run = tmp_path / "actual-run"
    actual_run.mkdir()
    linked_run = tmp_path / "linked-run"
    linked_run.symlink_to(actual_run, target_is_directory=True)

    with pytest.raises(RuntimeError, match="must resolve without traversing symlinks"):
        if operation == "resume":
            articulation_runner.resume_articulation_workflow(
                linked_run,
                repo_root=tmp_path,
            )
        else:
            decisions = tmp_path / "decisions.json"
            decisions.write_text(
                json.dumps({"candidate_0001": "accept"}),
                encoding="utf-8",
            )
            articulation_runner.review_articulation_workflow(
                linked_run,
                decisions,
                reviewer="asset-owner",
                repo_root=tmp_path,
            )

    assert list(actual_run.iterdir()) == []


def test_repeated_articulation_run_reuses_identical_request(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    config = articulation_runner.ArticulationRunConfig(
        repo_root=tmp_path,
        source_asset=source,
        output_dir=tmp_path / "run",
        joint_config=joint_config,
        intent="Infer six prismatic drawer joints.",
        review_policy="all",
        allowed_motion_types=("prismatic",),
        expected_candidate_count=6,
    )
    requests: list[dict[str, Any]] = []

    def fake_run(
        request: ArticulationWorkflowRequest,
        _config: Any,
    ) -> SimpleNamespace:
        requests.append(request.model_dump(mode="json"))
        return _result()

    monkeypatch.setattr(articulation_runner, "_run_request", fake_run)

    articulation_runner.run_articulation_workflow(config)
    articulation_runner.run_articulation_workflow(config)

    assert requests[0] == requests[1]
    assert requests[0]["metadata"]["content_workflow_cli"]["joint_config"] == str(
        joint_config
    )


def test_skill_routed_articulation_run_resumes_workflow_normalized_request(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    request, config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=tmp_path,
            source_asset=source,
            output_dir=run_dir,
            joint_config=joint_config,
            intent="Infer reviewed drawer joints.",
        )
    )
    persisted = request.model_copy(
        update={
            "metadata": {
                **request.metadata,
                "content_agent_workflows.scene_evidence_required": True,
            }
        }
    )
    _write_checkpointed_request(
        persisted,
        scene_evidence=False,
        phase="collecting_evidence",
    )
    expected = _result()
    calls: list[ArticulationWorkflowRequest] = []

    def launch(
        actual_request: ArticulationWorkflowRequest,
        _config: Any,
    ) -> SimpleNamespace:
        calls.append(actual_request)
        return expected

    monkeypatch.setattr(articulation_runner, "_launch_articulation_child", launch)

    assert articulation_runner._run_request(request, config) is expected
    assert calls == [request]


@pytest.mark.parametrize("checkpointed", [False, True])
def test_repeated_articulation_run_resumes_workflow_persisted_request(
    tmp_path: Path,
    monkeypatch: Any,
    checkpointed: bool,
) -> None:
    source, joint_config = _write_source_and_config(tmp_path)
    run_dir = tmp_path / "run"
    command = [
        "articulation",
        "run",
        "--execution-mode",
        "fixed",
        "--usd",
        str(source),
        "--joint-config",
        str(joint_config),
        "--output-dir",
        str(run_dir),
        "--intent",
        "Infer six prismatic drawer joints.",
        "--review-policy",
        "all",
        "--allowed-motion-type",
        "prismatic",
        "--expected-candidate-count",
        "6",
    ]
    calls: list[ArticulationWorkflowRequest] = []
    monkeypatch.setattr(
        articulation_runner,
        "JointAgentLocalClient",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        articulation_runner,
        "_build_live_evidence_collector",
        lambda _timeout: object(),
    )

    def fake_batch(
        request: ArticulationWorkflowRequest,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        calls.append(request)
        if len(calls) == 1:
            persisted_metadata = {
                **request.metadata,
                "content_agent_workflows.scene_evidence_required": True,
            }
            persisted_request = request.model_copy(
                update={"metadata": persisted_metadata}
            )
            if checkpointed:
                _write_checkpointed_request(
                    persisted_request,
                    scene_evidence=False,
                    phase="collecting_evidence",
                )
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "request.json").write_text(
                    persisted_request.model_dump_json(indent=2) + "\n",
                    encoding="utf-8",
                )
        return _result()

    monkeypatch.setattr(
        articulation_runner,
        "run_batch_articulation_workflow",
        fake_batch,
    )

    assert main(command) == 0
    assert main(command) == 0
    assert len(calls) == 2
    assert calls[0] == calls[1]


def test_articulation_child_status_reads_bounded_regular_file(tmp_path: Path) -> None:
    summary = tmp_path / "final.json"
    summary.write_text('{"status": "needs_review"}\n', encoding="utf-8")

    assert articulation_runner._articulation_child_status(summary) == "needs_review"


def test_articulation_child_status_reads_fenced_json_after_prose(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "final.json"
    summary.write_text(
        "The child completed its work.\n\n"
        "```json\n"
        '{"status": "awaiting_post_review", "final_summary_path": "final.json"}\n'
        "```\n",
        encoding="utf-8",
    )

    assert (
        articulation_runner._articulation_child_status(summary)
        == "awaiting_post_review"
    )


def test_articulation_child_status_uses_last_json_envelope(tmp_path: Path) -> None:
    summary = tmp_path / "final.json"
    summary.write_text(
        '{"status": "blocked", "detail": {"reason": "retry"}}\n'
        "Final result follows:\n"
        '{"status": "completed"}\n',
        encoding="utf-8",
    )

    assert articulation_runner._articulation_child_status(summary) == "completed"


def test_articulation_child_status_rejects_recursion_limited_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "final.json"
    summary.write_text('{"status": "completed"}\n', encoding="utf-8")

    def raise_recursion_error(*_args: Any, **_kwargs: Any) -> None:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(
        json.JSONDecoder,
        "raw_decode",
        raise_recursion_error,
    )

    assert articulation_runner._articulation_child_status(summary) is None


def test_articulation_child_status_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"status": "needs_review"}\n', encoding="utf-8")
    summary = tmp_path / "final.json"
    summary.symlink_to(target)

    assert articulation_runner._articulation_child_status(summary) is None


def test_articulation_child_status_rejects_oversized_file(tmp_path: Path) -> None:
    summary = tmp_path / "final.json"
    summary.write_bytes(
        b"x" * (articulation_runner.MAX_ARTICULATION_CHILD_FINAL_BYTES + 1)
    )

    assert articulation_runner._articulation_child_status(summary) is None
