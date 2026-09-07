# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public composed-asset launcher."""

from __future__ import annotations

import json
import os
import shutil
import signal
import site
import stat
import subprocess
import sys
import time
import venv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import content_agent_workflows.asset_composition.state as asset_state
import pytest
from content_agent_workflows.asset_composition import (
    ArtifactBinding,
    AssetCadParameterVariant,
    AssetCompositionRun,
    AssetCompositionStateError,
    AssetCoordinatorLeaseError,
    AssetCoordinatorSession,
    AssetExecutionGraph,
    AssetExecutionNode,
    AssetLeafCatalog,
    AssetLeafDescriptor,
    AssetTerminalValidation,
    StageName,
    begin_leaf,
    begin_stage,
    complete_leaf,
    discover_repository_asset_leaf_catalog,
    fail_leaf,
    fail_stage,
    freeze_execution_graph,
    leaf_directory,
    load_verified_asset_request,
    load_verified_run,
    record_coordinator_evidence_review,
    record_coordinator_plan,
    require_review,
    stage_directory,
    verify_frozen_asset_inputs,
)
from content_agent_workflows.asset_composition.catalog_adapters import (
    CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
    FOCUSED_VALIDATION_OPERATION_LEAF_ID,
    PROVIDED_VALIDATION_INGRESS_LEAF_ID,
    FocusedValidationOperationLeafInvocation,
)
from content_agent_workflows.common.artifacts import atomic_write_json, file_sha256
from content_agent_workflows.common.usd_cli import (
    USD_CLI_SOURCE_PATH,
    UsdCliPackageRoute,
)
from content_agent_workflows.common.usd_cli_session import (
    ParentUsdCliArtifactIdentity,
    ParentUsdCliSessionIdentity,
    ParentUsdCliStagedSourceIdentity,
)
from content_agent_workflows.validation import (
    prepare_validation_operations,
    run_validation_operation,
)
from pydantic import SecretStr
from world_understanding.validation import ValidationTemplateContext
from world_understanding.validation.models import (
    ValidationPlan,
    ValidationPlanStep,
    ValidationRequest,
    ValidationTemplateResult,
)

from content_workflow_cli import asset_runner, runner
from content_workflow_cli.asset_runner import (
    AssetRunConfig,
    resume_asset_workflow,
    resume_interactive_asset_workflow,
    review_asset_workflow,
    run_asset_workflow,
    run_interactive_asset_workflow,
)
from content_workflow_cli.cli import main


@pytest.fixture(autouse=True)
def _isolate_parent_usd_cli_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asset tests exercise launcher ownership without consuming a renderer."""

    repo = tmp_path / "repo"
    repo.mkdir()
    wrapper = repo / "usd-cli-tel"
    target = repo / "usd-cli"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    launcher = repo / "asset_runner.py"
    launcher.write_text("# test launcher\n", encoding="utf-8")
    monkeypatch.setattr(asset_runner, "__file__", str(launcher))
    route = runner.UsdCliTelemetryRoute(
        status="active",
        wrapper_path=wrapper,
        target_path=target,
    )
    identity = runner.UsdCliDaemonIdentity(
        pid=4242,
        process_start_token="test-start-token",
        project_id="a" * 24,
        instance_id="asset-test-instance",
        process_group_id=4242,
        os_session_id=4242,
    )
    lease = runner.UsdCliDaemonLease(
        target_path=target,
        identity=identity,
        server_state_bytes=json.dumps(
            {
                "pid": identity.pid,
                "process_start_token": identity.process_start_token,
                "project_id": identity.project_id,
                "instance_id": identity.instance_id,
                "lifecycle_owner": "external",
                "host": "127.0.0.1",
                "port": 4567,
            }
        ).encode("utf-8"),
        daemon_ledger_bytes=b"4242:test-start-token\n",
    )

    class FakeWorkflowSession:
        """Minimal append-only parent journal used by launcher unit tests."""

        session_id = "workflow-asset-test"

        def __init__(self, run_dir: Path) -> None:
            self.receipt_file = run_dir / "raw" / "usd_cli_command_receipts.jsonl"
            self.receipt_checkpoint_file = (
                run_dir / "raw" / "usd_cli_command_receipts.checkpoint.json"
            )
            self._journal_sha256: str | None = None
            self._journal_size: int | None = None
            self._checkpoint_sha256: str | None = None

        def open(self, _source: Path) -> None:
            self.receipt_file.parent.mkdir(parents=True, exist_ok=True)
            sequence = 1
            if self.receipt_file.exists():
                sequence += len(
                    self.receipt_file.read_text(encoding="utf-8").splitlines()
                )
            with self.receipt_file.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "operation_id": f"test-open-{sequence}",
                            "schema_version": (
                                "content-agent-workflows.usd-cli-command-receipt.v1"
                            ),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            self._journal_sha256 = file_sha256(self.receipt_file)
            self._journal_size = self.receipt_file.stat().st_size
            self.receipt_checkpoint_file.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "content-agent-workflows.usd-cli-receipt-checkpoint.v1"
                        ),
                        "workflow": "asset.run",
                        "session_id": self.session_id,
                        "receipt_device": self.receipt_file.stat().st_dev,
                        "receipt_inode": self.receipt_file.stat().st_ino,
                        "receipt_sha256": self._journal_sha256,
                        "receipt_size_bytes": self._journal_size,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            self._checkpoint_sha256 = file_sha256(self.receipt_checkpoint_file)

        def run_json(self, _arguments: list[str]) -> dict[str, Any]:
            self.open(Path("unused-generated-source.usda"))
            return {"ok": True}

        def verify_receipt_journal_integrity(self) -> None:
            if self._journal_sha256 is None:
                return
            if (
                not self.receipt_file.is_file()
                or file_sha256(self.receipt_file) != self._journal_sha256
                or self.receipt_file.stat().st_size != self._journal_size
                or not self.receipt_checkpoint_file.is_file()
                or file_sha256(self.receipt_checkpoint_file) != self._checkpoint_sha256
            ):
                raise RuntimeError(
                    "usd-cli receipt journal/checkpoint was replaced or modified"
                )

    def readiness(_repo_root: Path, **kwargs: Any) -> asset_runner.UsdCliReadiness:
        run_dir = Path(kwargs["run_dir"])
        artifact_stem = str(kwargs.get("artifact_stem", "ovrtx_probe"))
        artifact_path = run_dir / "raw" / f"{artifact_stem}.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "schema_version": "usd-cli.render-probe.v1",
                    "engine": "ovrtx",
                    "resolved_renderer": "ovrtx",
                    "transport": "local",
                    "ready": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return asset_runner.UsdCliReadiness(
            version="usd-cli test (renderer: ovrtx)",
            source_revision="b" * 40,
            probe={"ready": True},
            artifact_path=artifact_path,
        )

    monkeypatch.setattr(
        asset_runner,
        "_prepare_usd_cli_telemetry_route",
        lambda *_args, **_kwargs: route,
    )
    monkeypatch.setattr(
        asset_runner,
        "_start_usd_cli_run_daemon_strict",
        lambda **_kwargs: lease,
    )
    monkeypatch.setattr(
        asset_runner,
        "_activate_usd_cli_child_config",
        lambda _lease: None,
    )
    monkeypatch.setattr(
        asset_runner,
        "_attach_workflow_usd_cli_session",
        lambda **kwargs: FakeWorkflowSession(Path(str(kwargs["run_dir"]))),
    )
    monkeypatch.setattr(asset_runner, "ensure_usd_cli_ovrtx_ready", readiness)
    monkeypatch.setattr(
        runner,
        "verify_live_parent_usd_cli_daemon",
        lambda _identity: runner.ParentUsdCliConnection(
            server_url="http://127.0.0.1:4567",
            session_id=FakeWorkflowSession.session_id,
            authentication_token=SecretStr("parent-token"),
        ),
    )
    monkeypatch.setattr(
        asset_runner,
        "_stop_usd_cli_run_daemon_strict",
        lambda **_kwargs: runner.UsdCliDaemonTeardownEvidence(
            identity=identity,
            host="127.0.0.1",
            port=4567,
            daemon_was_started=True,
            process_released=True,
            descendants_released=True,
            sessions_released=True,
            listener_released=True,
            daemon_leases_released=True,
            state_directory_released=True,
        ),
    )


@pytest.mark.parametrize("raw_value", ["NaN", "Infinity", "-Infinity"])
def test_cad_parameter_values_reject_non_finite_numbers(raw_value: str) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        asset_runner._parameter_values([f"parameter.width={raw_value}"])


def test_cad_parameter_variant_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        AssetCadParameterVariant(
            id="invalid",
            parameter_values={"parameter.width": float("nan")},
        )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    (repo / "agentic" / ".agents" / "skills").mkdir(parents=True)
    source = repo / "cabinet.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    joint = repo / "joint.yaml"
    joint.write_text("model: test\n", encoding="utf-8")
    materials_usd = repo / "materials.usda"
    materials_usd.write_text("#usda 1.0\n", encoding="utf-8")
    materials = repo / "materials.yaml"
    materials.write_text(
        "library_path: materials.usda\nentries: []\n", encoding="utf-8"
    )
    reference = repo / "reference.png"
    reference.write_bytes(b"image")
    catalog = repo / "asset_leaf_catalog.json"
    catalog.write_text(
        discover_repository_asset_leaf_catalog().model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "repo": repo,
        "source": source,
        "joint": joint,
        "materials": materials,
        "materials_usd": materials_usd,
        "reference": reference,
        "catalog": catalog,
    }


def _config(paths: dict[str, Path], run_dir: Path) -> AssetRunConfig:
    return AssetRunConfig(
        repo_root=paths["repo"],
        usd_path=paths["source"],
        prompt=(
            "Find drawer joints, apply painted metal with subtle wear, author "
            "physics, validate, and package the result."
        ),
        selected_mode="compatibility_fixed",
        joint_config=paths["joint"],
        materials_yaml=paths["materials"],
        reference_images=[paths["reference"]],
        output_dir=run_dir,
        run_id="cabinet-composed",
        include_geometry_stage=False,
        dry_run=True,
    )


def _agentic_config(paths: dict[str, Path], run_dir: Path) -> AssetRunConfig:
    return AssetRunConfig(
        repo_root=paths["repo"],
        usd_path=paths["source"],
        prompt="Run only the selected opaque asset operations.",
        selected_mode="agentic",
        leaf_catalog=asset_runner._load_leaf_catalog(paths["catalog"]),
        reference_images=[paths["reference"]],
        output_dir=run_dir,
        run_id="agentic-asset",
        dry_run=True,
    )


def _allow_provider_free_agentic_readiness(
    paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = paths["repo"] / "usd-cli"
    wrapper = paths["repo"] / "usd-cli-tel"
    monkeypatch.setattr(
        asset_runner,
        "resolve_package_owned_usd_cli_route",
        lambda _root: UsdCliPackageRoute(
            wrapper=wrapper.resolve(),
            target=target.resolve(),
            source_root=paths["repo"].resolve(),
            source_revision="d" * 40,
        ),
    )
    monkeypatch.setattr(
        asset_runner,
        "run_bounded_usd_cli_subprocess",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="usd-cli test\n"),
    )


class _AssetFocusedValidationExecutor:
    template_versions = {"physics_sane": "test.physics-sane.v1"}

    def __init__(self, *, status: Literal["passed", "failed"]) -> None:
        self._status = status

    def plan(
        self,
        request: ValidationRequest,
        *,
        working_dir: Path,
    ) -> ValidationPlan:
        del working_dir
        return ValidationPlan(
            steps=tuple(
                ValidationPlanStep(
                    template_name=name,
                    reason="Selected by the public asset bridge fixture.",
                )
                for name in request.requested_templates
            ),
            reasoning_summary="Preserved the selected asset operation order.",
        )

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        del context
        return ValidationTemplateResult(
            template_name=template_name,
            status=self._status,
        )


def _write_focused_validation_leaf_result(
    *,
    attempt: Path,
    source_asset: str,
    status: Literal["passed", "failed"],
) -> tuple[Path, Path]:
    output_dir = attempt / "validation"
    executor = _AssetFocusedValidationExecutor(status=status)
    prepare_validation_operations(
        ValidationRequest(
            task_description="Validate the public asset bridge fixture.",
            inputs=(source_asset,),
            requested_templates=("physics_sane",),
        ),
        output_dir=output_dir,
        config_base_dir=Path(source_asset).parent,
        executor=executor,
    )
    operation = run_validation_operation(
        output_dir,
        template_name="physics_sane",
        executor=executor,
    )
    invocation = attempt / "invocation.json"
    result = attempt / "result.json"
    atomic_write_json(
        invocation,
        FocusedValidationOperationLeafInvocation(
            output_dir=str(output_dir),
            template_name="physics_sane",
        ),
    )
    atomic_write_json(result, operation)
    return invocation, result


def _complete_agentic_fixture_graph(run_dir: Path) -> None:
    state_path = run_dir / "asset_run.json"
    request = asset_runner.load_verified_asset_request(state_path)
    assert request.leaf_catalog is not None
    assert request.sole_coordinator_identity is not None
    descriptor = next(
        item
        for item in request.leaf_catalog.descriptors
        if item.leaf_id == FOCUSED_VALIDATION_OPERATION_LEAF_ID
    )
    graph = AssetExecutionGraph.create(
        sole_coordinator_identity_digest=(
            request.sole_coordinator_identity.identity_digest
        ),
        prompt_digest=str(request.prompt_digest),
        source_digest=str(request.source_digest),
        configuration_digest=str(request.configuration_digest),
        reference_digest=str(request.reference_digest),
        leaf_catalog_digest=request.leaf_catalog.catalog_digest,
        nodes=[
            AssetExecutionNode(
                leaf_id=descriptor.leaf_id,
                requirement="required",
                descriptor_digest=descriptor.descriptor_digest,
                terminal_output=True,
            )
        ],
        omitted_leaf_ids=[
            item.leaf_id
            for item in request.leaf_catalog.descriptors
            if item.leaf_id != descriptor.leaf_id
        ],
    )
    graph_path = run_dir / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path, actor="test-codex")
    begin_leaf(state_path, descriptor.leaf_id, actor="test-codex")
    attempt = leaf_directory(state_path, descriptor.leaf_id)
    invocation, result = _write_focused_validation_leaf_result(
        attempt=attempt,
        source_asset=request.source_asset,
        status="passed",
    )
    completed = complete_leaf(
        state_path,
        descriptor.leaf_id,
        invocation_path=invocation,
        result_path=result,
        actor="test-codex",
    )
    receipt_binding = completed.leaf_states[descriptor.leaf_id].receipt
    assert receipt_binding is not None
    receipt = json.loads(Path(receipt_binding.path).read_text(encoding="utf-8"))
    assert request.source_asset not in {item["path"] for item in receipt["evidence"]}


def _fail_agentic_fixture_graph(run_dir: Path) -> None:
    state_path = run_dir / "asset_run.json"
    request = asset_runner.load_verified_asset_request(state_path)
    assert request.leaf_catalog is not None
    assert request.sole_coordinator_identity is not None
    descriptor = next(
        item
        for item in request.leaf_catalog.descriptors
        if item.leaf_id == FOCUSED_VALIDATION_OPERATION_LEAF_ID
    )
    graph = AssetExecutionGraph.create(
        sole_coordinator_identity_digest=(
            request.sole_coordinator_identity.identity_digest
        ),
        prompt_digest=str(request.prompt_digest),
        source_digest=str(request.source_digest),
        configuration_digest=str(request.configuration_digest),
        reference_digest=str(request.reference_digest),
        leaf_catalog_digest=request.leaf_catalog.catalog_digest,
        nodes=[
            AssetExecutionNode(
                leaf_id=descriptor.leaf_id,
                requirement="required",
                descriptor_digest=descriptor.descriptor_digest,
                terminal_output=True,
            )
        ],
        omitted_leaf_ids=[
            item.leaf_id
            for item in request.leaf_catalog.descriptors
            if item.leaf_id != descriptor.leaf_id
        ],
    )
    graph_path = run_dir / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path, actor="test-codex")
    begin_leaf(state_path, descriptor.leaf_id, actor="test-codex")
    attempt = leaf_directory(state_path, descriptor.leaf_id)
    invocation, result = _write_focused_validation_leaf_result(
        attempt=attempt,
        source_asset=request.source_asset,
        status="failed",
    )
    fail_leaf(
        state_path,
        descriptor.leaf_id,
        reason=("focused Validation physics_sane ended with native status failed"),
        invocation_path=invocation,
        result_path=result,
        actor="test-codex",
    )


def test_asset_default_run_directory_is_at_repo_root(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    config = replace(
        _config(paths, tmp_path / "unused"),
        output_dir=None,
        run_id="root-asset",
    )

    run_id, run_dir = asset_runner._create_run_dir(config)

    assert run_id == "root-asset"
    assert run_dir == paths["repo"] / "runs" / "root-asset"


def test_asset_run_directories_ignore_permissive_umask(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    previous_umask = os.umask(0o022)
    try:
        _run_id, observed_run_dir = asset_runner._create_run_dir(
            _config(paths, run_dir)
        )
    finally:
        os.umask(previous_umask)

    assert observed_run_dir == run_dir
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((run_dir / "raw").stat().st_mode) == 0o700


def _plan_and_begin(state_path: Path, stage: StageName) -> None:
    draft = state_path.parent / "raw" / f"{stage}-plan-draft.json"
    draft.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-plan-draft.v1"
                ),
                "stage": stage,
                "objective": f"Execute and verify {stage}.",
                "steps": [
                    {
                        "stage": stage,
                        "objective": f"Run typed {stage} executor.",
                        "acceptance_evidence": ["typed result"],
                        "may_revisit": True,
                    }
                ],
                "evidence_paths": [str(state_path.parent / "request.json")],
                "revision_reason": "Initial evidence-backed stage plan.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record_coordinator_plan(state_path, plan_path=draft, actor="test")
    begin_stage(state_path, stage)


def _record_await_review(state_path: Path, candidates: Path) -> None:
    draft = state_path.parent / "raw" / "articulation-review-draft.json"
    draft.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-review-draft.v1"
                ),
                "stage": "articulation",
                "output_asset_path": None,
                "evidence_paths": [str(candidates)],
                "findings": ["Joint candidates require explicit user review."],
                "decision": "await_review",
                "target_stage": None,
                "decision_summary": "Pause at the digest-bound Joint gate.",
                "repair_scope": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record_coordinator_evidence_review(
        state_path,
        review_path=draft,
        actor="test",
    )


def _record_accept(
    state_path: Path,
    *,
    stage: StageName,
    output: Path,
    evidence: list[Path],
) -> None:
    draft = state_path.parent / "raw" / f"{stage}-accept-draft.json"
    draft.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-review-draft.v1"
                ),
                "stage": stage,
                "output_asset_path": str(output),
                "evidence_paths": [str(path) for path in evidence],
                "findings": [f"Typed {stage} evidence passed."],
                "decision": "accept",
                "target_stage": None,
                "decision_summary": f"Accept the exact {stage} output.",
                "repair_scope": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record_coordinator_evidence_review(state_path, review_path=draft, actor="test")


def _record_refine(
    state_path: Path,
    *,
    stage: StageName,
    evidence: list[Path],
) -> None:
    draft = state_path.parent / "raw" / f"{stage}-refine-draft.json"
    draft.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-review-draft.v1"
                ),
                "stage": stage,
                "output_asset_path": None,
                "evidence_paths": [str(path) for path in evidence],
                "findings": ["The exact typed candidate needs bounded repair."],
                "decision": "refine",
                "target_stage": None,
                "decision_summary": "Refine the reviewed semantic candidate.",
                "repair_scope": ["Repair only failed acceptance checks."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record_coordinator_evidence_review(state_path, review_path=draft, actor="test")


def test_public_asset_cli_discovers_repository_owned_catalog(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["asset", "catalog"]) == 0
    observed = json.loads(capsys.readouterr().out)
    expected = discover_repository_asset_leaf_catalog().model_dump(mode="json")
    assert observed == expected
    leaf_ids = [item["leaf_id"] for item in observed["descriptors"]]
    assert leaf_ids == sorted(leaf_ids)
    assert {
        CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
        FOCUSED_VALIDATION_OPERATION_LEAF_ID,
        PROVIDED_VALIDATION_INGRESS_LEAF_ID,
    } <= set(leaf_ids)


def test_asset_run_dry_run_freezes_prompt_and_creates_top_level_state(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    result = run_asset_workflow(_config(paths, run_dir))

    assert result.returncode == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["workflow"] == "asset.run"
    assert request["physics_validation_mode"] == "runtime_required"
    staged_source = run_dir / "inputs" / "asset_source" / "cabinet.usda"
    assert request["schema_version"] == "content-agents.asset-composition-request.v2"
    assert request["source_asset"] == str(staged_source)
    assert request["source_staging"]["original_source"]["path"] == str(paths["source"])
    assert request["source_staging"]["staged_source"]["path"] == str(staged_source)
    assert request["source_staging"]["dependency_digest_set_sha256"]
    assert request["joint_config"] == str(paths["joint"])
    assert request["joint_config_binding"]["path"] == str(paths["joint"])
    assert request["materials_yaml"] == str(paths["materials"])
    assert request["materials_yaml_binding"]["path"] == str(paths["materials"])
    assert request["materials_usd"] == str(paths["materials_usd"])
    assert request["materials_usd_binding"]["path"] == str(paths["materials_usd"])
    assert request["reference_images"] == [str(paths["reference"])]
    assert [item["path"] for item in request["reference_bindings"]] == [
        str(paths["reference"])
    ]
    run = load_verified_run(run_dir / "asset_run.json")
    assert run.current_stage == "articulation"
    assert run.stages["articulation"].status == "ready"
    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert "`content-workflow-asset` skill" in prompt
    assert "Material, Texture, Physics, Validation" in prompt
    assert "do not launch recursive coding" in prompt
    assert "agents from any domain workflow" in prompt
    assert "one long-running cross-domain reasoning loop" in prompt
    assert "typed coordinator plan" in " ".join(prompt.split())
    assert "Validation task/templates" in prompt


@pytest.mark.parametrize("suffix", [".step", ".stl", ".obj", ".urdf", ".mjcf"])
def test_asset_run_stages_non_usd_sources_without_usd_composition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    paths = _inputs(tmp_path)
    source = paths["repo"] / f"source{suffix}"
    source_bytes = {
        ".urdf": b'<robot name="immutable-test"/>\n',
        ".mjcf": b'<mujoco model="immutable-test"/>\n',
    }.get(suffix, f"immutable test source {suffix}\n".encode())
    source.write_bytes(source_bytes)
    run_dir = tmp_path / "run"

    def reject_usd_staging(**_kwargs: object) -> None:
        raise AssertionError("non-USD source entered USD dependency staging")

    monkeypatch.setattr(asset_runner, "_stage_usd_cli_input_tree", reject_usd_staging)
    result = run_asset_workflow(
        replace(
            _config(paths, run_dir),
            usd_path=source,
            include_geometry_stage=True,
        )
    )

    assert result.returncode == 0
    staged = run_dir / "inputs" / "asset_source" / source.name
    assert staged.read_bytes() == source_bytes
    request = load_verified_asset_request(run_dir / "asset_run.json")
    assert request.source_asset == str(staged)
    assert request.source_staging is not None
    assert request.source_staging.original_source.path == str(source)
    assert request.source_staging.staged_source.path == str(staged)
    manifest = json.loads(
        Path(request.source_staging.manifest.path).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == (
        "content-workflow-cli.immutable-source-closure.v1"
    )
    assert manifest["source_path"] == str(source)
    assert manifest["staged_path"] == str(staged)
    assert load_verified_run(run_dir / "asset_run.json").current_stage == "geometry"
    staged.chmod(0o600)
    staged.write_bytes(b"mutated after request freeze\n")
    with pytest.raises(AssetCompositionStateError, match="identity changed"):
        resume_asset_workflow(run_dir, dry_run=True)


@pytest.mark.parametrize(
    ("source_name", "source_bytes", "dependencies", "expected_paths", "strategy"),
    [
        (
            "robot.urdf",
            b'<robot name="fixture"><link name="base"><visual><geometry>'
            b'<mesh filename="meshes/link.stl"/></geometry></visual></link></robot>\n',
            {"meshes/link.stl": b"solid link\nendsolid link\n"},
            {"robot.urdf", "meshes/link.stl"},
            "urdf_xml_reference_graph",
        ),
        (
            "urdf/package_robot.urdf",
            b'<robot name="fixture"><link name="base"><visual><geometry>'
            b'<mesh filename="package://fixture_pkg/meshes/link.stl"/>'
            b"</geometry></visual></link></robot>\n",
            {
                "package.xml": (
                    b'<package format="3"><name>fixture_pkg</name>'
                    b"<version>1.0.0</version><description>fixture</description>"
                    b'<maintainer email="test@example.com">Test</maintainer>'
                    b"<license>Apache-2.0</license></package>\n"
                ),
                "meshes/link.stl": b"solid link\nendsolid link\n",
            },
            {"urdf/package_robot.urdf", "package.xml", "meshes/link.stl"},
            "urdf_xml_reference_graph",
        ),
        (
            "urdf/package_robot.xml",
            b'<robot name="fixture"><link name="base"><visual><geometry>'
            b'<mesh filename="package://fixture_pkg/meshes/link.stl"/>'
            b"</geometry></visual></link></robot>\n",
            {
                "package.xml": (
                    b'<package format="3"><name>fixture_pkg</name>'
                    b"<version>1.0.0</version><description>fixture</description>"
                    b'<maintainer email="test@example.com">Test</maintainer>'
                    b"<license>Apache-2.0</license></package>\n"
                ),
                "meshes/link.stl": b"solid link\nendsolid link\n",
            },
            {"urdf/package_robot.xml", "package.xml", "meshes/link.stl"},
            "urdf_xml_reference_graph",
        ),
        (
            "urdf/current_package_robot.urdf",
            b'<robot name="fixture"><link name="base"><collision><geometry>'
            b'<mesh filename="package:///meshes/link.stl"/>'
            b"</geometry></collision></link></robot>\n",
            {
                "package.xml": (
                    b'<package format="3"><name>fixture_pkg</name>'
                    b"<version>1.0.0</version><description>fixture</description>"
                    b'<maintainer email="test@example.com">Test</maintainer>'
                    b"<license>Apache-2.0</license></package>\n"
                ),
                "meshes/link.stl": b"solid link\nendsolid link\n",
            },
            {"urdf/current_package_robot.urdf", "package.xml", "meshes/link.stl"},
            "urdf_xml_reference_graph",
        ),
        (
            "urdf/single_slash_package_robot.urdf",
            b'<robot name="fixture"><link name="base"><visual><geometry>'
            b'<mesh filename="package:/meshes/link.stl"/>'
            b"</geometry></visual></link></robot>\n",
            {
                "package.xml": (
                    b'<package format="3"><name>fixture_pkg</name>'
                    b"<version>1.0.0</version><description>fixture</description>"
                    b'<maintainer email="test@example.com">Test</maintainer>'
                    b"<license>Apache-2.0</license></package>\n"
                ),
                "meshes/link.stl": b"solid link\nendsolid link\n",
            },
            {
                "urdf/single_slash_package_robot.urdf",
                "package.xml",
                "meshes/link.stl",
            },
            "urdf_xml_reference_graph",
        ),
        (
            "model.mjcf",
            b'<mujoco model="fixture"><compiler meshdir="meshes"/>'
            b'<include file="parts/body.xml"/></mujoco>\n',
            {
                "parts/body.xml": (
                    b'<mujoco><asset><mesh name="link" file="link.stl"/>'
                    b"</asset></mujoco>\n"
                ),
                "meshes/link.stl": b"solid link\nendsolid link\n",
            },
            {"model.mjcf", "parts/body.xml", "meshes/link.stl"},
            "mjcf_xml_reference_graph",
        ),
        (
            "model_without_asset_dirs.mjcf",
            b'<mujoco model="fixture"><include file="parts/body.xml"/></mujoco>\n',
            {
                "parts/body.xml": (
                    b'<mujoco><asset><mesh name="link" file="link.stl"/>'
                    b"</asset></mujoco>\n"
                ),
                "link.stl": b"solid link\nendsolid link\n",
            },
            {"model_without_asset_dirs.mjcf", "parts/body.xml", "link.stl"},
            "mjcf_xml_reference_graph",
        ),
        (
            "model_with_mesh_backed_assets.mjcf",
            b'<mujoco model="fixture"><compiler meshdir="geometry"/>'
            b'<asset><hfield name="terrain" file="terrain.png"/>'
            b'<skin name="cover" file="cover.skn"/></asset></mujoco>\n',
            {
                "geometry/terrain.png": b"height fixture",
                "geometry/cover.skn": b"skin fixture",
            },
            {
                "model_with_mesh_backed_assets.mjcf",
                "geometry/terrain.png",
                "geometry/cover.skn",
            },
            "mjcf_xml_reference_graph",
        ),
        (
            "model_with_nested_include.mjcf",
            b'<mujoco model="fixture"><compiler meshdir="meshes"/>'
            b'<include file="parts/body.xml"/></mujoco>\n',
            {
                "parts/body.xml": b'<mujoco><include file="shared/assets.xml"/></mujoco>\n',
                "shared/assets.xml": (
                    b'<mujoco><asset><mesh name="link" file="link.stl"/>'
                    b"</asset></mujoco>\n"
                ),
                "meshes/link.stl": b"solid link\nendsolid link\n",
            },
            {
                "model_with_nested_include.mjcf",
                "parts/body.xml",
                "shared/assets.xml",
                "meshes/link.stl",
            },
            "mjcf_xml_reference_graph",
        ),
        (
            "model.obj",
            b"mtllib materials/model.mtl\nv 0 0 0\n",
            {
                "materials/model.mtl": b"newmtl body\nmap_Kd ../textures/body.png\n",
                "textures/body.png": b"png fixture",
            },
            {"model.obj", "materials/model.mtl", "textures/body.png"},
            "obj_material_texture_graph",
        ),
        (
            "windows_paths.obj",
            b'mtllib "materials\\model finish.mtl"\nv 0 0 0\n',
            {
                "materials/model finish.mtl": (
                    b'newmtl body\nmap_Kd "..\\textures\\body finish.png"\n'
                ),
                "textures/body finish.png": b"png fixture",
            },
            {
                "windows_paths.obj",
                "materials/model finish.mtl",
                "textures/body finish.png",
            },
            "obj_material_texture_graph",
        ),
        (
            "hash_paths.obj",
            b'mtllib "materials/model #1.mtl" # material library\nv 0 0 0\n',
            {
                "materials/model #1.mtl": (
                    b"newmtl body\nmap_Kd ../textures/body#1.png # texture\n"
                ),
                "textures/body#1.png": b"png fixture",
            },
            {
                "hash_paths.obj",
                "materials/model #1.mtl",
                "textures/body#1.png",
            },
            "obj_material_texture_graph",
        ),
        (
            "model.gltf",
            json.dumps(
                {
                    "asset": {"version": "2.0"},
                    "buffers": [{"uri": "buffers/model.bin", "byteLength": 4}],
                    "images": [{"uri": "textures/body.png"}],
                }
            ).encode(),
            {
                "buffers/model.bin": b"mesh",
                "textures/body.png": b"png fixture",
            },
            {"model.gltf", "buffers/model.bin", "textures/body.png"},
            "gltf_uri_graph",
        ),
        (
            "model.dae",
            b'<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">'
            b'<library_images><image id="body"><init_from>'
            b"textures/body.png</init_from></image></library_images></COLLADA>\n",
            {"textures/body.png": b"png fixture"},
            {"model.dae", "textures/body.png"},
            "collada_reference_graph",
        ),
        (
            "model_1_5.dae",
            b'<COLLADA xmlns="https://www.khronos.org/collada/">'
            b'<library_images><image id="body"><init_from><ref>'
            b"textures/body.png</ref></init_from></image></library_images>"
            b"</COLLADA>\n",
            {"textures/body.png": b"png fixture"},
            {"model_1_5.dae", "textures/body.png"},
            "collada_reference_graph",
        ),
        (
            "assembly.dae",
            b'<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">'
            b'<library_nodes><node id="root"><instance_node '
            b'url="parts/body.dae#body"/></node></library_nodes></COLLADA>\n',
            {
                "parts/body.dae": (
                    b'<COLLADA xmlns="http://www.collada.org/2005/11/'
                    b'COLLADASchema"><library_images><image id="body">'
                    b"<init_from>../textures/body.png</init_from></image>"
                    b"</library_images></COLLADA>\n"
                ),
                "textures/body.png": b"png fixture",
            },
            {"assembly.dae", "parts/body.dae", "textures/body.png"},
            "collada_reference_graph",
        ),
        (
            "effects.dae",
            b'<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">'
            b'<library_materials><material id="body"><instance_effect '
            b'url="materials/effects.dae#paint"/></material></library_materials>'
            b"</COLLADA>\n",
            {
                "materials/effects.dae": (
                    b'<COLLADA xmlns="http://www.collada.org/2005/11/'
                    b'COLLADASchema"><library_effects><effect id="paint"/>'
                    b"</library_effects></COLLADA>\n"
                ),
            },
            {"effects.dae", "materials/effects.dae"},
            "collada_reference_graph",
        ),
    ],
)
def test_asset_run_stages_complete_non_usd_dependency_closure(
    tmp_path: Path,
    source_name: str,
    source_bytes: bytes,
    dependencies: dict[str, bytes],
    expected_paths: set[str],
    strategy: str,
) -> None:
    paths = _inputs(tmp_path)
    source_dir = paths["repo"] / "source-package"
    source_dir.mkdir()
    source = source_dir / source_name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(source_bytes)
    for relative, payload in dependencies.items():
        dependency = source_dir / relative
        dependency.parent.mkdir(parents=True, exist_ok=True)
        dependency.write_bytes(payload)
    run_dir = tmp_path / "run"

    result = run_asset_workflow(
        replace(
            _config(paths, run_dir),
            usd_path=source,
            include_geometry_stage=True,
        )
    )

    assert result.returncode == 0
    request = load_verified_asset_request(run_dir / "asset_run.json")
    assert request.source_staging is not None
    staged_root = run_dir / "inputs" / "asset_source"
    staged_paths = {
        Path(binding.path).relative_to(staged_root).as_posix()
        for binding in request.source_staging.staged_dependencies
    }
    assert staged_paths == expected_paths
    manifest = json.loads(
        Path(request.source_staging.manifest.path).read_text(encoding="utf-8")
    )
    assert manifest["dependency_discovery_strategy"] == strategy
    assert manifest["unresolved_dependencies"] == []
    assert manifest["self_containment"] == {
        "status": "verified",
        "escaped_paths": [],
    }
    run = load_verified_run(run_dir / "asset_run.json")
    assert len(run.source_dependencies) == len(expected_paths) - 1

    staged_dependency = next(
        Path(binding.path)
        for binding in request.source_staging.staged_dependencies
        if binding.path != request.source_staging.staged_source.path
    )
    staged_dependency.chmod(0o600)
    staged_dependency.write_bytes(b"mutated staged dependency\n")
    with pytest.raises(AssetCompositionStateError, match="dependency closure identity"):
        resume_asset_workflow(run_dir, dry_run=True)


@pytest.mark.parametrize(
    ("source_name", "source_bytes", "error"),
    [
        (
            "missing.urdf",
            b'<robot name="fixture"><link name="base"><visual><geometry>'
            b'<mesh filename="meshes/missing.stl"/></geometry></visual></link></robot>\n',
            "dependency is missing",
        ),
        (
            "escape.obj",
            b"mtllib ../outside.mtl\n",
            "escapes its approved root",
        ),
        (
            "encoded_windows_absolute.obj",
            b"mtllib C%3A%5Cmodels%5Coutside.mtl\n",
            "Absolute source dependency is not allowed",
        ),
        (
            "encoded_separator_windows_absolute.obj",
            b"mtllib C:%5Cmodels%5Coutside.mtl\n",
            "Absolute source dependency is not allowed",
        ),
        (
            "absolute_asset_with_meshdir.mjcf",
            b'<mujoco model="fixture"><compiler meshdir="meshes"/>'
            b'<asset><mesh name="body" file="/tmp/body.stl"/></asset></mujoco>\n',
            "Absolute source dependency is not allowed",
        ),
        (
            "encoded_absolute_asset_with_meshdir.mjcf",
            b'<mujoco model="fixture"><compiler meshdir="meshes"/>'
            b'<asset><mesh name="body" file="%2Ftmp%2Fbody.stl"/></asset></mujoco>\n',
            "Absolute source dependency is not allowed",
        ),
        (
            "remote.gltf",
            json.dumps(
                {
                    "asset": {"version": "2.0"},
                    "buffers": [
                        {"uri": "https://example.invalid/model.bin", "byteLength": 4}
                    ],
                }
            ).encode(),
            "Remote or absolute",
        ),
        (
            "entity.urdf",
            b'<!DOCTYPE robot [<!ENTITY name "fixture">]><robot name="&name;"/>\n',
            "must not declare a DTD or entity",
        ),
    ],
)
def test_asset_run_rejects_unclosed_non_usd_dependencies(
    tmp_path: Path,
    source_name: str,
    source_bytes: bytes,
    error: str,
) -> None:
    paths = _inputs(tmp_path)
    source_dir = paths["repo"] / "source-package"
    source_dir.mkdir()
    source = source_dir / source_name
    source.write_bytes(source_bytes)
    (paths["repo"] / "outside.mtl").write_text("newmtl outside\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        run_asset_workflow(
            replace(
                _config(paths, tmp_path / "run"),
                usd_path=source,
                include_geometry_stage=True,
            )
        )


@pytest.mark.parametrize(
    ("source_name", "source_template"),
    [
        ("absolute.obj", "mtllib <OUTSIDE>\nv 0 0 0\n"),
        (
            "absolute.gltf",
            '{"asset":{"version":"2.0"},"buffers":'
            '[{"uri":"<OUTSIDE>","byteLength":4}]}',
        ),
        (
            "absolute.dae",
            "<COLLADA><library_images><image><init_from><OUTSIDE>"
            "</init_from></image></library_images></COLLADA>\n",
        ),
        (
            "absolute.urdf",
            '<robot name="fixture"><link name="base"><visual><geometry>'
            '<mesh filename="<OUTSIDE>"/></geometry></visual></link></robot>\n',
        ),
        (
            "absolute.mjcf",
            '<mujoco model="fixture"><compiler meshdir="meshes"/>'
            '<asset><mesh name="body" file="<OUTSIDE>"/></asset></mujoco>\n',
        ),
    ],
)
@pytest.mark.parametrize("reference_kind", ["absolute", "escaping"])
def test_explicit_source_root_still_validates_parsed_references(
    tmp_path: Path,
    source_name: str,
    source_template: str,
    reference_kind: str,
) -> None:
    paths = _inputs(tmp_path)
    source_root = paths["repo"] / "parsed-source"
    source_root.mkdir()
    outside = paths["repo"] / "outside.bin"
    outside.write_bytes(b"outside dependency")
    source = source_root / source_name
    reference = (
        outside.as_posix() if reference_kind == "absolute" else "../../outside.bin"
    )
    source.write_text(
        source_template.replace("<OUTSIDE>", reference),
        encoding="utf-8",
    )

    expected_error = (
        "Absolute source dependency is not allowed"
        if reference_kind == "absolute"
        else "escapes its approved root"
    )
    with pytest.raises(ValueError, match=expected_error):
        run_asset_workflow(
            replace(
                _config(paths, tmp_path / f"rejected-{source.suffix[1:]}"),
                usd_path=source,
                source_root=source_root,
                include_geometry_stage=True,
            )
        )


def test_asset_run_rejects_symlinked_non_usd_dependency(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    source_dir = paths["repo"] / "source-package"
    source_dir.mkdir()
    source = source_dir / "model.urdf"
    source.write_text(
        '<robot name="fixture"><link name="base"><visual><geometry>'
        '<mesh filename="meshes/link.stl"/></geometry></visual></link></robot>\n',
        encoding="utf-8",
    )
    outside = paths["repo"] / "outside.stl"
    outside.write_text("solid outside\nendsolid outside\n", encoding="utf-8")
    mesh = source_dir / "meshes" / "link.stl"
    mesh.parent.mkdir()
    mesh.symlink_to(outside)

    with pytest.raises(ValueError, match="must not traverse a symlink"):
        run_asset_workflow(
            replace(
                _config(paths, tmp_path / "run"),
                usd_path=source,
                include_geometry_stage=True,
            )
        )


def test_asset_run_requires_explicit_root_for_distant_ros_package_manifest(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    source_root = paths["repo"] / "distant-package"
    source_root.mkdir()
    (source_root / "package.xml").write_text(
        '<package format="3"><name>distant_pkg</name><version>1.0.0</version>'
        "<description>fixture</description>"
        '<maintainer email="test@example.com">Test</maintainer>'
        "<license>Apache-2.0</license></package>\n",
        encoding="utf-8",
    )
    mesh = source_root / "meshes" / "link.stl"
    mesh.parent.mkdir()
    mesh.write_text("solid link\nendsolid link\n", encoding="utf-8")
    (source_root / "unreferenced.bin").write_bytes(b"must not be staged")
    source = source_root / "one" / "two" / "three" / "four" / "robot.xml"
    source.parent.mkdir(parents=True)
    source.write_text(
        '<robot name="fixture"><link name="base"><visual><geometry>'
        '<mesh filename="package://distant_pkg/meshes/link.stl"/>'
        "</geometry></visual></link></robot>\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provide --source-root"):
        run_asset_workflow(
            replace(
                _config(paths, tmp_path / "run"),
                usd_path=source,
                include_geometry_stage=True,
            )
        )

    approved_run = tmp_path / "approved-run"
    result = run_asset_workflow(
        replace(
            _config(paths, approved_run),
            usd_path=source,
            source_root=source_root,
            include_geometry_stage=True,
        )
    )

    assert result.returncode == 0
    request = load_verified_asset_request(approved_run / "asset_run.json")
    assert request.source_staging is not None
    staged_root = approved_run / "inputs" / "asset_source"
    assert {
        Path(binding.path).relative_to(staged_root).as_posix()
        for binding in request.source_staging.staged_dependencies
    } == {
        "meshes/link.stl",
        "one/two/three/four/robot.xml",
        "package.xml",
    }


def test_urdf_explicit_workspace_root_discovers_nested_ros_package(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    workspace_root = paths["repo"] / "workspace" / "src"
    package_root = workspace_root / "fixture_pkg"
    package_root.mkdir(parents=True)
    (package_root / "package.xml").write_text(
        '<package format="3"><name>fixture_pkg</name><version>1.0.0</version>'
        "<description>fixture</description>"
        '<maintainer email="test@example.com">Test</maintainer>'
        "<license>Apache-2.0</license></package>\n",
        encoding="utf-8",
    )
    mesh = package_root / "meshes" / "link.stl"
    mesh.parent.mkdir()
    mesh.write_text("solid link\nendsolid link\n", encoding="utf-8")
    source = package_root / "urdf" / "robot.urdf"
    source.parent.mkdir()
    source.write_text(
        '<robot name="fixture"><link name="base"><visual><geometry>'
        '<mesh filename="package://fixture_pkg/meshes/link.stl"/>'
        "</geometry></visual></link></robot>\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "workspace-root-run"
    result = run_asset_workflow(
        replace(
            _config(paths, run_dir),
            usd_path=source,
            source_root=workspace_root,
            include_geometry_stage=True,
        )
    )

    assert result.returncode == 0
    request = load_verified_asset_request(run_dir / "asset_run.json")
    assert request.source_staging is not None
    staged_root = run_dir / "inputs" / "asset_source"
    assert {
        Path(binding.path).relative_to(staged_root).as_posix()
        for binding in request.source_staging.staged_dependencies
    } == {
        "fixture_pkg/meshes/link.stl",
        "fixture_pkg/package.xml",
        "fixture_pkg/urdf/robot.urdf",
    }


def test_asset_run_rejects_encoded_absolute_ros_package_path(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    source_root = paths["repo"] / "source-package"
    source_root.mkdir()
    (source_root / "package.xml").write_text(
        '<package format="3"><name>fixture_pkg</name><version>1.0.0</version>'
        "<description>fixture</description>"
        '<maintainer email="test@example.com">Test</maintainer>'
        "<license>Apache-2.0</license></package>\n",
        encoding="utf-8",
    )
    source = source_root / "urdf" / "robot.urdf"
    source.parent.mkdir()
    source.write_text(
        '<robot name="fixture"><link name="base"><collision><geometry>'
        '<mesh filename="package://fixture_pkg/%2Foutside.stl"/>'
        "</geometry></collision></link></robot>\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Absolute source dependency is not allowed"):
        run_asset_workflow(
            replace(
                _config(paths, tmp_path / "run"),
                usd_path=source,
                include_geometry_stage=True,
            )
        )


def test_asset_run_requires_and_freezes_explicit_root_for_opaque_source(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    source_dir = paths["repo"] / "opaque-package"
    source_dir.mkdir()
    source = source_dir / "assembly.fbx"
    source.write_bytes(b"opaque assembly")
    sidecar = source_dir / "textures" / "body.png"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"texture")

    with pytest.raises(ValueError, match="Provide --source-root"):
        run_asset_workflow(
            replace(
                _config(paths, tmp_path / "rejected-run"),
                usd_path=source,
                include_geometry_stage=True,
            )
        )

    run_dir = tmp_path / "accepted-run"
    result = run_asset_workflow(
        replace(
            _config(paths, run_dir),
            usd_path=source,
            source_root=source_dir,
            include_geometry_stage=True,
        )
    )

    assert result.returncode == 0
    request = load_verified_asset_request(run_dir / "asset_run.json")
    assert request.source_staging is not None
    staged_root = run_dir / "inputs" / "asset_source"
    assert {
        Path(binding.path).relative_to(staged_root).as_posix()
        for binding in request.source_staging.staged_dependencies
    } == {"assembly.fbx", "textures/body.png"}


def test_asset_run_allows_bounded_source_closure_above_run_directory(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    source_dir = paths["repo"] / "source-package"
    source_dir.mkdir()
    source = source_dir / "part.step"
    source.write_bytes(b"ISO-10303-21;\nEND-ISO-10303-21;\n")
    run_dir = source_dir / "runs" / "part"

    result = run_asset_workflow(
        replace(
            _config(paths, run_dir),
            usd_path=source,
            include_geometry_stage=True,
        )
    )

    assert result.returncode == 0
    request = load_verified_asset_request(run_dir / "asset_run.json")
    assert request.source_staging is not None
    assert len(request.source_staging.staged_dependencies) == 1


def test_asset_run_excludes_run_artifacts_from_parsed_source_closure(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    source_dir = paths["repo"] / "source-package"
    source_dir.mkdir()
    mesh = source_dir / "meshes" / "link.stl"
    mesh.parent.mkdir()
    mesh.write_text("solid link\nendsolid link\n", encoding="utf-8")
    source = source_dir / "robot.xml"
    source.write_text(
        '<robot name="fixture"><link name="base"><visual><geometry>'
        '<mesh filename="meshes/link.stl"/>'
        "</geometry></visual></link></robot>\n",
        encoding="utf-8",
    )
    run_dir = source_dir / "runs" / "robot"

    result = run_asset_workflow(
        replace(
            _config(paths, run_dir),
            usd_path=source,
            include_geometry_stage=True,
        )
    )

    assert result.returncode == 0
    request = load_verified_asset_request(run_dir / "asset_run.json")
    assert request.source_staging is not None
    staged_root = run_dir / "inputs" / "asset_source"
    assert {
        Path(binding.path).relative_to(staged_root).as_posix()
        for binding in request.source_staging.staged_dependencies
    } == {"meshes/link.stl", "robot.xml"}


def test_asset_run_rejects_explicit_source_tree_containing_run_directory(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    source_dir = paths["repo"] / "source-package"
    source_dir.mkdir()
    source = source_dir / "assembly.fbx"
    source.write_bytes(b"opaque assembly")

    with pytest.raises(ValueError, match="Run directory must be outside"):
        run_asset_workflow(
            replace(
                _config(paths, source_dir / "run"),
                usd_path=source,
                source_root=source_dir,
                include_geometry_stage=True,
            )
        )


def test_parent_usd_cli_session_does_not_open_non_usd_source() -> None:
    calls: list[tuple[str, object]] = []

    class Session:
        def open(self, path: Path) -> None:
            calls.append(("open", path))

        def run_json(self, arguments: list[str]) -> None:
            calls.append(("run_json", arguments))

    asset_runner._initialize_parent_usd_cli_session(
        cast(Any, Session()),
        source_asset=None,
    )

    assert calls == [("run_json", ["info"])]


def test_source_free_asset_run_requires_geometry_agent_export(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "source-free-run"

    with pytest.raises(
        ValueError,
        match="Asset composition requires a provided source",
    ):
        run_asset_workflow(
            replace(
                _config(paths, run_dir),
                usd_path=None,
                source_images=[paths["reference"]],
            )
        )

    assert not run_dir.exists()


def test_cad_required_outputs_accepts_usda_as_the_geometry_representation() -> None:
    assert asset_runner._required_cad_outputs(["step"]) == ["step", "usd"]
    assert asset_runner._required_cad_outputs(["usda", "step"]) == ["usda", "step"]


def test_asset_run_agentic_default_freezes_catalog_without_fixed_stage_inputs(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "agentic-run"

    result = run_asset_workflow(_agentic_config(paths, run_dir))

    assert result.returncode == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["schema_version"] == "content-agents.asset-composition-request.v3"
    assert request["selected_mode"] == "agentic"
    assert request["leaf_catalog"]["catalog_digest"]
    assert request["sole_coordinator_identity"]["identity_digest"]
    assert "required_leaf_ids" not in request
    assert "required_terminal_leaf_ids" not in request
    assert "required_leaf_dependencies" not in request
    assert "joint_config" not in request
    assert "materials_yaml" not in request
    assert "physics_validation_mode" not in request
    run = load_verified_run(run_dir / "asset_run.json")
    assert run.selected_mode == "agentic"
    assert run.current_stage is None
    assert run.stages == {}
    assert run.execution_graph is None
    assert run.coordinator.next_action == "freeze_graph"
    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert "sole outer coordinator" in prompt
    assert "There is no built-in total domain or stage order" in prompt
    assert "fixed stage pipeline" in prompt
    assert "copy its absolute `path`, `sha256`, and `size_bytes` verbatim" in prompt
    assert "never reconstruct, abbreviate, or relocate the path" in " ".join(
        prompt.split()
    )
    assert "existing regular file below the declared run" in " ".join(prompt.split())
    assert (
        "Never create, pre-create, or write a descriptor-owned `native/`"
        in " ".join(prompt.split())
    )
    assert "only the opaque entrypoint initializes" in " ".join(prompt.split())
    assert "`articulation.proposal-provider.v1` is selected" in prompt
    assert "`proposal_status=not_evaluated`" in prompt
    assert "`proposal_status=not_requested` only when that leaf is omitted" in " ".join(
        prompt.split()
    )
    assert "`canonical_output_evidence_required=true`" in prompt
    assert "author-before-canonical-render dependency" in prompt
    assert "top-level `evidence_requirements` to a non-empty" in prompt
    assert "every `candidate_decisions` entry" in " ".join(prompt.split())
    assert "`canonical_graph.groups`" in prompt
    assert "`canonical_graph.memberships`" in prompt
    assert "`canonical_graph.joints`" in prompt
    assert "`canonical_graph.rigid_link_operations`" in prompt
    assert "before the first native call" in " ".join(prompt.split())
    assert "try to repair it after a native failure" in " ".join(prompt.split())
    assert "exact `evidence_id` values present" in prompt
    assert "these fields are not future post-author validation goals" in " ".join(
        prompt.split()
    )
    assert "`required_leaf_ids` entry" not in prompt
    assert "`required_terminal_leaf_ids` as terminal outputs" not in prompt
    assert "Controlled JSON artifact writes" in prompt
    assert "`execution_graph.json`" in prompt
    assert "Do not invoke the provider `Write`, `Edit`, or `MultiEdit`" in " ".join(
        prompt.split()
    )
    if os.name == "nt":
        assert "do not invoke `Copy-Item`, `Get-Item`, or" in prompt
        assert "use the run directory itself" in prompt
        assert "Do not probe for an alternate copy, stat, or hash" in prompt
        assert "content-workflow-asset-state invoke-leaf" in prompt
        assert "Do not\ninvoke the descriptor's domain executable directly" in prompt
        assert "Do not invoke `usd-cli-tel`" in prompt
        assert "Domain leaves invoked\nthrough `invoke-leaf`" in prompt
        assert "Read each staged skill once" in " ".join(prompt.split())
        assert "Select-Object" in prompt
        assert "Never assign a PowerShell variable" in prompt
    else:
        assert "`jq -nS`" in prompt


def test_parent_usd_cli_prompt_uses_attached_project_token_discovery(
    tmp_path: Path,
) -> None:
    artifact = ParentUsdCliArtifactIdentity.model_construct(
        path=str(tmp_path / "source.usda"),
        sha256="a" * 64,
        size_bytes=1,
    )
    source = ParentUsdCliStagedSourceIdentity.model_construct(
        original_source=artifact,
        staged_source=artifact,
        staging_manifest=artifact,
        dependency_digest_set_sha256="b" * 64,
    )
    identity = ParentUsdCliSessionIdentity.model_construct(
        source=source,
        run_dir=str(tmp_path),
        server_host="127.0.0.1",
        server_port=4567,
        parent_session_id="workflow-parent",
    )

    prompt = asset_runner._prompt_with_parent_usd_cli_session(
        "base prompt",
        identity=identity,
        identity_path=tmp_path / "identity.json",
        identity_sha256="c" * 64,
    )

    assert "omit `--server`" in prompt
    assert "pinned attached-project route" in prompt
    assert "`--session workflow-parent`" in prompt
    assert "--server http://127.0.0.1:4567" not in prompt


def test_agentic_required_leaf_contract_is_frozen_and_restored(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "required-agentic-run"
    config = replace(
        _agentic_config(paths, run_dir),
        required_leaf_ids=[
            FOCUSED_VALIDATION_OPERATION_LEAF_ID,
            PROVIDED_VALIDATION_INGRESS_LEAF_ID,
        ],
        required_terminal_leaf_ids=[FOCUSED_VALIDATION_OPERATION_LEAF_ID],
        required_leaf_dependencies={
            FOCUSED_VALIDATION_OPERATION_LEAF_ID: [PROVIDED_VALIDATION_INGRESS_LEAF_ID]
        },
        exact_leaf_scope=True,
    )

    result = run_asset_workflow(config)

    assert result.returncode == 0
    request = load_verified_asset_request(run_dir / "asset_run.json")
    assert request.required_leaf_ids == [
        FOCUSED_VALIDATION_OPERATION_LEAF_ID,
        PROVIDED_VALIDATION_INGRESS_LEAF_ID,
    ]
    assert request.required_terminal_leaf_ids == [FOCUSED_VALIDATION_OPERATION_LEAF_ID]
    assert request.required_leaf_dependencies == {
        FOCUSED_VALIDATION_OPERATION_LEAF_ID: [PROVIDED_VALIDATION_INGRESS_LEAF_ID]
    }
    assert request.exact_leaf_scope is True
    restored = asset_runner._config_from_request(request, dry_run=True)
    assert restored.required_leaf_ids == [
        FOCUSED_VALIDATION_OPERATION_LEAF_ID,
        PROVIDED_VALIDATION_INGRESS_LEAF_ID,
    ]
    assert restored.required_terminal_leaf_ids == [FOCUSED_VALIDATION_OPERATION_LEAF_ID]
    assert restored.required_leaf_dependencies == {
        FOCUSED_VALIDATION_OPERATION_LEAF_ID: [PROVIDED_VALIDATION_INGRESS_LEAF_ID]
    }
    assert restored.exact_leaf_scope is True
    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert "`required_leaf_ids` entry" in prompt
    assert "`required_terminal_leaf_ids` as terminal outputs" in prompt
    assert "every and only `required_leaf_ids`" in prompt
    assert "`selection_rationale`" in prompt


def test_agentic_required_leaf_contract_fails_before_run_creation(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "missing-required-leaf"

    with pytest.raises(
        ValueError,
        match="Required asset capabilities are absent",
    ):
        run_asset_workflow(
            replace(
                _agentic_config(paths, run_dir),
                required_leaf_ids=["missing.release-capability.v1"],
            )
        )

    assert not run_dir.exists()


def test_agentic_required_leaf_contract_rejects_duplicates_and_unbound_terminal(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    base = _agentic_config(paths, tmp_path / "invalid-required-leaf")

    with pytest.raises(ValueError, match="must not contain duplicates"):
        asset_runner._validated_config(
            replace(
                base,
                required_leaf_ids=[
                    FOCUSED_VALIDATION_OPERATION_LEAF_ID,
                    FOCUSED_VALIDATION_OPERATION_LEAF_ID,
                ],
            )
        )
    with pytest.raises(
        ValueError,
        match="must also be supplied with --required-leaf",
    ):
        asset_runner._validated_config(
            replace(
                base,
                required_terminal_leaf_ids=[FOCUSED_VALIDATION_OPERATION_LEAF_ID],
            )
        )
    with pytest.raises(
        ValueError,
        match="dependency IDs must also be supplied",
    ):
        asset_runner._validated_config(
            replace(
                base,
                required_leaf_ids=[FOCUSED_VALIDATION_OPERATION_LEAF_ID],
                required_leaf_dependencies={
                    FOCUSED_VALIDATION_OPERATION_LEAF_ID: [
                        PROVIDED_VALIDATION_INGRESS_LEAF_ID
                    ]
                },
            )
        )
    with pytest.raises(ValueError, match="requires at least one --required-leaf"):
        asset_runner._validated_config(replace(base, exact_leaf_scope=True))
    assert asset_runner._parse_required_leaf_dependencies(
        ["publish.v1=inspect.v1", "publish.v1=review.v1"]
    ) == {"publish.v1": ["inspect.v1", "review.v1"]}
    with pytest.raises(ValueError, match="requires LEAF=DEPENDENCY"):
        asset_runner._parse_required_leaf_dependencies(["publish.v1"])
    with pytest.raises(ValueError, match="must not repeat an edge"):
        asset_runner._parse_required_leaf_dependencies(
            ["publish.v1=inspect.v1", "publish.v1=inspect.v1"]
        )


def test_public_agentic_single_prompt_finishes_without_provider_preselection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "agentic-single-prompt"
    target = paths["repo"] / "usd-cli"
    wrapper = paths["repo"] / "usd-cli-tel"
    child_invocations = 0
    finalized_under_coordinator_lease = False

    monkeypatch.setattr(
        asset_runner,
        "resolve_package_owned_usd_cli_route",
        lambda _root: UsdCliPackageRoute(
            wrapper=wrapper.resolve(),
            target=target.resolve(),
            source_root=paths["repo"].resolve(),
            source_revision="d" * 40,
        ),
    )
    monkeypatch.setattr(
        asset_runner,
        "run_bounded_usd_cli_subprocess",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="usd-cli test\n"),
    )
    monkeypatch.setattr(
        asset_runner,
        "ensure_usd_cli_ovrtx_ready",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("agentic graph selection called a provider readiness path")
        ),
    )

    def one_prompt_child(**kwargs: object) -> int:
        nonlocal child_invocations
        child_invocations += 1
        _complete_agentic_fixture_graph(Path(str(kwargs["run_dir"])))
        return 0

    monkeypatch.setattr(asset_runner, "_run_child_agent", one_prompt_child)

    original_finalize = asset_runner.finalize_graph_run

    def finalize_under_lease(
        path: str | Path,
        *,
        parent_release_receipt_path: str | Path | None = None,
        parent_command_receipt_journal_path: str | Path | None = None,
        parent_command_receipt_checkpoint_path: str | Path | None = None,
        expected_parent_release_receipt: ArtifactBinding | None = None,
        expected_parent_command_receipt_journal: ArtifactBinding | None = None,
        expected_parent_command_receipt_checkpoint: ArtifactBinding | None = None,
        actor: str = "asset-coordinator",
    ) -> AssetCompositionRun:
        nonlocal finalized_under_coordinator_lease
        state_path = Path(path)
        with pytest.raises(AssetCoordinatorLeaseError):
            asset_runner.run_asset_coordinator_transition(
                state_path,
                transition=lambda: None,
            )
        finalized_under_coordinator_lease = True
        return original_finalize(
            state_path,
            parent_release_receipt_path=parent_release_receipt_path,
            parent_command_receipt_journal_path=(parent_command_receipt_journal_path),
            parent_command_receipt_checkpoint_path=(
                parent_command_receipt_checkpoint_path
            ),
            expected_parent_release_receipt=expected_parent_release_receipt,
            expected_parent_command_receipt_journal=(
                expected_parent_command_receipt_journal
            ),
            expected_parent_command_receipt_checkpoint=(
                expected_parent_command_receipt_checkpoint
            ),
            actor=actor,
        )

    monkeypatch.setattr(asset_runner, "finalize_graph_run", finalize_under_lease)

    result = run_asset_workflow(replace(_agentic_config(paths, run_dir), dry_run=False))

    assert child_invocations == 1
    assert finalized_under_coordinator_lease is True
    assert result.returncode == 0
    assert result.completed is True
    run = load_verified_run(run_dir / "asset_run.json")
    assert run.selected_mode == "agentic"
    assert run.terminal_status == "completed"
    readiness_paths = list((run_dir / "raw").glob("asset_usd_cli_probe_*.json"))
    assert len(readiness_paths) == 1
    readiness = json.loads(readiness_paths[0].read_text(encoding="utf-8"))
    assert readiness["probe"] == {
        "provider_readiness": "not_requested",
        "selected_mode": "agentic",
    }
    assert run.graph_terminal_receipt is not None
    terminal = json.loads(
        Path(run.graph_terminal_receipt.path).read_text(encoding="utf-8")
    )
    assert len(terminal["resource_release_receipts"]) == 1
    assert "asset_usd_cli_teardown_" in terminal["resource_release_receipts"][0]["path"]


def test_public_interactive_asset_lifecycle_owns_exact_parent_without_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "interactive-agentic"
    events: list[str] = []
    callback_count = 0
    original_start = asset_runner._start_usd_cli_run_daemon_strict
    original_attach = asset_runner._attach_workflow_usd_cli_session
    original_stop = asset_runner._stop_usd_cli_run_daemon_strict
    original_finalize = asset_runner.finalize_graph_run
    _allow_provider_free_agentic_readiness(paths, monkeypatch)

    def start(**kwargs: Any) -> Any:
        events.append("start")
        return original_start(**kwargs)

    def attach(**kwargs: Any) -> Any:
        events.append("attach")
        return original_attach(**kwargs)

    def stop(**kwargs: Any) -> Any:
        events.append("stop")
        return original_stop(**kwargs)

    def finalize(path: str | Path, **kwargs: Any) -> AssetCompositionRun:
        assert events[-1] == "stop"
        events.append("finalize")
        return original_finalize(path, **kwargs)

    monkeypatch.setattr(asset_runner, "_start_usd_cli_run_daemon_strict", start)
    monkeypatch.setattr(asset_runner, "_attach_workflow_usd_cli_session", attach)
    monkeypatch.setattr(asset_runner, "_stop_usd_cli_run_daemon_strict", stop)
    monkeypatch.setattr(asset_runner, "finalize_graph_run", finalize)
    monkeypatch.setattr(
        asset_runner,
        "_run_child_agent",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("interactive route reached the child reasoner")
        ),
    )
    monkeypatch.setattr(
        asset_runner,
        "run_batch_asset_coordinator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("interactive route reached the batch coordinator")
        ),
    )

    def reason(session: AssetCoordinatorSession) -> int:
        nonlocal callback_count
        callback_count += 1
        events.append("reason")
        assert session.mode == "interactive"
        identity = session.parent_usd_cli_session_identity
        identity_artifact = session.parent_usd_cli_session_identity_artifact
        assert identity is not None
        assert identity_artifact is not None
        assert identity.run_id == "agentic-asset"
        assert identity.run_dir == str(run_dir)
        assert identity.allowed_roots == [str(run_dir)]
        assert identity.renderer_credentials == "parent_confined"
        assert identity.lifecycle_authority == "parent_only"
        request = asset_runner.load_verified_asset_request(session.run_state_path)
        assert request.requires_parent_resource_release is True
        assert request.source_staging is not None
        assert identity.source.staged_source.path == request.source_asset
        assert identity.source.staged_source.sha256 == request.source_digest
        assert identity_artifact == asset_runner._frozen_file_binding(
            Path(identity_artifact.path)
        )
        _complete_agentic_fixture_graph(run_dir)
        with pytest.raises(
            AssetCompositionStateError,
            match="launcher releases parent resources before graph finalization",
        ):
            session.finalize_graph()
        return 0

    result = run_interactive_asset_workflow(
        replace(_agentic_config(paths, run_dir), dry_run=False),
        reasoning_loop=reason,
    )

    assert callback_count == 1
    assert events == ["start", "attach", "reason", "stop", "finalize"]
    assert result.returncode == 0
    assert result.completed is True
    run = load_verified_run(run_dir / "asset_run.json")
    assert run.terminal_status == "completed"
    assert run.graph_terminal_receipt is not None
    terminal = json.loads(
        Path(run.graph_terminal_receipt.path).read_text(encoding="utf-8")
    )
    teardown = _single_teardown_receipt(run_dir)
    assert terminal["parent_release_receipt"]["path"].endswith(
        f"asset_usd_cli_teardown_{teardown['launch_id']}.json"
    )
    assert (
        terminal["parent_command_receipt_journal"]
        == teardown["command_receipt_journal"]
    )
    assert (
        terminal["parent_command_receipt_checkpoint"]
        == teardown["command_receipt_checkpoint"]
    )


def test_public_interactive_asset_lifecycle_releases_without_finalizing_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "interactive-failure"
    events: list[str] = []
    original_stop = asset_runner._stop_usd_cli_run_daemon_strict
    _allow_provider_free_agentic_readiness(paths, monkeypatch)

    def stop(**kwargs: Any) -> Any:
        events.append("stop")
        return original_stop(**kwargs)

    monkeypatch.setattr(asset_runner, "_stop_usd_cli_run_daemon_strict", stop)
    monkeypatch.setattr(
        asset_runner,
        "finalize_graph_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a failed graph reached graph finalization")
        ),
    )
    monkeypatch.setattr(
        asset_runner,
        "_run_child_agent",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("interactive route reached the child reasoner")
        ),
    )

    def reason(_session: AssetCoordinatorSession) -> int:
        events.append("reason")
        _fail_agentic_fixture_graph(run_dir)
        return 7

    result = run_interactive_asset_workflow(
        replace(_agentic_config(paths, run_dir), dry_run=False),
        reasoning_loop=reason,
    )

    assert events == ["reason", "stop"]
    assert result.returncode == 7
    assert result.completed is False
    run = load_verified_run(run_dir / "asset_run.json")
    assert run.terminal_status == "failed"
    assert run.graph_terminal_receipt is None
    teardown = _single_teardown_receipt(run_dir)
    assert teardown["status"] == "released"
    assert teardown["boundary"] == "failed"


@pytest.mark.parametrize(
    ("error", "expected_returncode", "expected_boundary"),
    [
        (
            runner.ChildProcessInterrupted(15, "interactive asset reasoning"),
            130,
            "cancelled",
        ),
        (KeyboardInterrupt(), 130, "cancelled"),
        (RuntimeError("interactive reasoning failed"), 2, "incomplete"),
    ],
    ids=["cancelled", "interrupted", "exception"],
)
def test_public_interactive_asset_lifecycle_releases_on_reasoning_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    expected_returncode: int,
    expected_boundary: str,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "interactive-exit"
    events: list[str] = []
    original_stop = asset_runner._stop_usd_cli_run_daemon_strict
    _allow_provider_free_agentic_readiness(paths, monkeypatch)

    def stop(**kwargs: Any) -> Any:
        events.append("stop")
        return original_stop(**kwargs)

    def reason(_session: AssetCoordinatorSession) -> int:
        events.append("reason")
        raise error

    monkeypatch.setattr(asset_runner, "_stop_usd_cli_run_daemon_strict", stop)
    monkeypatch.setattr(
        asset_runner,
        "finalize_graph_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an unfinished graph reached graph finalization")
        ),
    )
    monkeypatch.setattr(
        asset_runner,
        "_run_child_agent",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("interactive route reached the child reasoner")
        ),
    )

    result = run_interactive_asset_workflow(
        replace(_agentic_config(paths, run_dir), dry_run=False),
        reasoning_loop=reason,
    )

    assert events == ["reason", "stop"]
    assert result.returncode == expected_returncode
    assert result.completed is False
    run = load_verified_run(run_dir / "asset_run.json")
    assert run.terminal_status == "active"
    assert run.graph_terminal_receipt is None
    teardown = _single_teardown_receipt(run_dir)
    assert teardown["status"] == "released"
    assert teardown["boundary"] == expected_boundary


def test_public_interactive_asset_lifecycle_setup_exception_skips_reasoner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "interactive-setup-exception"
    callback_count = 0
    stop_count = 0
    original_stop = asset_runner._stop_usd_cli_run_daemon_strict

    def stop(**kwargs: Any) -> Any:
        nonlocal stop_count
        stop_count += 1
        return original_stop(**kwargs)

    def reason(_session: AssetCoordinatorSession) -> int:
        nonlocal callback_count
        callback_count += 1
        return 0

    monkeypatch.setattr(
        asset_runner,
        "_asset_usd_cli_readiness",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("provider-free readiness failed")
        ),
    )
    monkeypatch.setattr(asset_runner, "_stop_usd_cli_run_daemon_strict", stop)
    monkeypatch.setattr(
        asset_runner,
        "_run_child_agent",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("interactive route reached the child reasoner")
        ),
    )

    result = run_interactive_asset_workflow(
        replace(_agentic_config(paths, run_dir), dry_run=False),
        reasoning_loop=reason,
    )

    assert callback_count == 0
    assert stop_count == 1
    assert result.returncode == 2
    assert result.completed is False
    teardown = _single_teardown_receipt(run_dir)
    assert teardown["status"] == "released"
    assert teardown["boundary"] == "setup_failed"


@pytest.mark.parametrize(
    ("drift", "path_argument"),
    [
        ("missing", "parent_release_receipt_path"),
        ("substituted", "parent_command_receipt_journal_path"),
        ("stale", "parent_command_receipt_checkpoint_path"),
    ],
)
def test_public_interactive_asset_lifecycle_rejects_release_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    path_argument: str,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / f"interactive-release-{drift}"
    original_finalize = asset_runner.finalize_graph_run
    _allow_provider_free_agentic_readiness(paths, monkeypatch)

    def finalize(path: str | Path, **kwargs: Any) -> AssetCompositionRun:
        artifact_path = Path(kwargs[path_argument])
        if drift == "missing":
            artifact_path.unlink()
        elif drift == "substituted":
            substitute = artifact_path.with_name(f"substituted-{artifact_path.name}")
            shutil.copyfile(artifact_path, substitute)
            kwargs[path_argument] = substitute
        else:
            artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
        return original_finalize(path, **kwargs)

    monkeypatch.setattr(asset_runner, "finalize_graph_run", finalize)
    monkeypatch.setattr(
        asset_runner,
        "_run_child_agent",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("interactive route reached the child reasoner")
        ),
    )

    result = run_interactive_asset_workflow(
        replace(_agentic_config(paths, run_dir), dry_run=False),
        reasoning_loop=lambda _session: _complete_agentic_fixture_graph(run_dir),
    )

    assert result.returncode == 2
    assert result.completed is False
    run = load_verified_run(run_dir / "asset_run.json")
    assert run.coordinator.next_action == "finalize_receipts"
    assert run.graph_terminal_receipt is None


def test_public_interactive_asset_resume_reuses_identity_with_one_new_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "interactive-resume"
    callbacks: list[str] = []
    _allow_provider_free_agentic_readiness(paths, monkeypatch)
    monkeypatch.setattr(
        asset_runner,
        "_run_child_agent",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("interactive route reached the child reasoner")
        ),
    )

    def interrupted(_session: AssetCoordinatorSession) -> int:
        callbacks.append("initial")
        raise RuntimeError("resume after this exact interactive boundary")

    first = run_interactive_asset_workflow(
        replace(_agentic_config(paths, run_dir), dry_run=False),
        reasoning_loop=interrupted,
    )
    request_before = (run_dir / "request.json").read_bytes()
    source_digest_before = asset_runner.load_verified_asset_request(
        run_dir / "asset_run.json"
    ).source_digest

    def complete(_session: AssetCoordinatorSession) -> int:
        callbacks.append("resume")
        _complete_agentic_fixture_graph(run_dir)
        return 0

    resumed = resume_interactive_asset_workflow(
        run_dir,
        reasoning_loop=complete,
    )

    assert first.returncode == 2
    assert first.completed is False
    assert resumed.returncode == 0
    assert resumed.completed is True
    assert callbacks == ["initial", "resume"]
    assert (run_dir / "request.json").read_bytes() == request_before
    assert (
        asset_runner.load_verified_asset_request(
            run_dir / "asset_run.json"
        ).source_digest
        == source_digest_before
    )
    launch_paths = sorted((run_dir / "raw").glob("asset_usd_cli_launch_*.json"))
    teardown_paths = sorted((run_dir / "raw").glob("asset_usd_cli_teardown_*.json"))
    identity_paths = sorted((run_dir / "raw").glob("asset_usd_cli_session_*.json"))
    assert len(launch_paths) == 2
    assert len(teardown_paths) == 2
    assert len(identity_paths) == 2
    assert (
        len(
            {
                json.loads(path.read_text(encoding="utf-8"))["launch_id"]
                for path in teardown_paths
            }
        )
        == 2
    )


def test_public_interactive_resume_finishes_frozen_graph_v1_without_upgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "interactive-legacy-graph"
    state_path = run_dir / "asset_run.json"
    request_path = run_dir / "request.json"
    _allow_provider_free_agentic_readiness(paths, monkeypatch)
    monkeypatch.setattr(
        asset_runner,
        "_run_child_agent",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("interactive route reached the child reasoner")
        ),
    )

    prepared = run_interactive_asset_workflow(
        _agentic_config(paths, run_dir),
        reasoning_loop=lambda _session: 0,
    )
    assert prepared.returncode == 0
    descriptor = AssetLeafDescriptor.create(
        leaf_id="legacy.inspect.v1",
        entrypoint="legacy inspect",
        invocation_schema_digest="1" * 64,
        result_schema_digest="2" * 64,
    )
    legacy_catalog = AssetLeafCatalog.create([descriptor])
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    request_payload["leaf_catalog"] = legacy_catalog.model_dump(mode="json")
    atomic_write_json(request_path, request_payload)
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    state_payload["request"] = asset_runner._frozen_file_binding(
        request_path
    ).model_dump(mode="json")
    atomic_write_json(state_path, state_payload)
    request = asset_runner.load_verified_asset_request(state_path)
    assert request.sole_coordinator_identity is not None
    assert request.leaf_catalog == legacy_catalog
    graph_payload = {
        "schema_version": "content-agents.asset-execution-graph.v1",
        "selected_mode": "agentic",
        "sole_coordinator_identity_digest": (
            request.sole_coordinator_identity.identity_digest
        ),
        "prompt_digest": request.prompt_digest,
        "source_digest": request.source_digest,
        "configuration_digest": request.configuration_digest,
        "reference_digest": request.reference_digest,
        "leaf_catalog_digest": legacy_catalog.catalog_digest,
        "selected_leaf_ids": [descriptor.leaf_id],
        "nodes": [
            {
                "leaf_id": descriptor.leaf_id,
                "depends_on": [],
                "requirement": "required",
                "descriptor_digest": descriptor.descriptor_digest,
                "terminal_output": True,
            }
        ],
        "omitted_leaf_ids": [],
    }
    graph = AssetExecutionGraph.model_validate(
        {
            **graph_payload,
            "graph_digest": asset_runner.canonical_asset_digest(graph_payload),
        }
    )
    graph_path = run_dir / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    graph_bytes_before = graph_path.read_bytes()
    freeze_execution_graph(state_path, graph_path=graph_path)
    begin_leaf(state_path, descriptor.leaf_id)
    attempt = leaf_directory(state_path, descriptor.leaf_id)
    invocation = attempt / "invocation.json"
    result = attempt / "result.json"
    evidence = attempt / "evidence.json"
    readback = attempt / "readback.json"
    for path, payload in (
        (invocation, {"kind": "legacy invocation"}),
        (result, {"kind": "legacy result"}),
        (evidence, {"kind": "legacy evidence"}),
        (readback, {"kind": "legacy readback"}),
    ):
        atomic_write_json(path, payload)
    complete_leaf(
        state_path,
        descriptor.leaf_id,
        invocation_path=invocation,
        result_path=result,
        evidence_paths=[evidence],
        saved_stage_readback_paths=[readback],
        native_disposition="passed",
        summary="historical leaf completed",
    )

    callbacks = 0

    def finish(session: AssetCoordinatorSession) -> int:
        nonlocal callbacks
        callbacks += 1
        assert session.load().coordinator.next_action == "finalize_receipts"
        prompt = (run_dir / "agent_resume_prompt.md").read_text(encoding="utf-8")
        assert "asset-execution-graph.v1" in prompt
        assert "Never regenerate, reorder, or upgrade" in prompt
        return 0

    resumed = resume_interactive_asset_workflow(
        run_dir,
        reasoning_loop=finish,
    )

    assert callbacks == 1
    assert resumed.returncode == 0
    assert resumed.completed is True
    completed = load_verified_run(state_path)
    assert completed.execution_graph is not None
    assert Path(completed.execution_graph.path).read_bytes() == graph_bytes_before
    assert completed.graph_terminal_receipt is not None
    terminal = json.loads(
        Path(completed.graph_terminal_receipt.path).read_text(encoding="utf-8")
    )
    assert terminal["schema_version"].endswith("graph-terminal-receipt.v1")
    assert "parent_release_receipt" not in terminal
    assert len(terminal["resource_release_receipts"]) == 1
    assert "asset_usd_cli_teardown_" in terminal["resource_release_receipts"][0]["path"]


def test_public_interactive_concurrent_resume_cannot_start_second_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "interactive-concurrent-lease"
    nested_callbacks = 0
    starts = 0
    original_start = asset_runner._start_usd_cli_run_daemon_strict
    _allow_provider_free_agentic_readiness(paths, monkeypatch)
    monkeypatch.setattr(
        asset_runner,
        "_run_child_agent",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("interactive route reached the child reasoner")
        ),
    )

    def start(**kwargs: Any) -> Any:
        nonlocal starts
        starts += 1
        return original_start(**kwargs)

    def nested_reasoner(_session: AssetCoordinatorSession) -> int:
        nonlocal nested_callbacks
        nested_callbacks += 1
        return 0

    def outer_reasoner(_session: AssetCoordinatorSession) -> int:
        state_path = run_dir / "asset_run.json"
        state_before = state_path.read_bytes()
        nested = resume_interactive_asset_workflow(
            run_dir,
            reasoning_loop=nested_reasoner,
        )
        assert nested.returncode == 2
        assert nested.completed is False
        assert nested_callbacks == 0
        assert state_path.read_bytes() == state_before
        assert not (run_dir / "agent_resume_prompt.md").exists()
        assert len(list((run_dir / "raw").glob("asset_usd_cli_launch_*.json"))) == 1
        assert len(list((run_dir / "raw").glob("asset_usd_cli_session_*.json"))) == 1
        _complete_agentic_fixture_graph(run_dir)
        return 0

    monkeypatch.setattr(asset_runner, "_start_usd_cli_run_daemon_strict", start)
    result = run_interactive_asset_workflow(
        replace(_agentic_config(paths, run_dir), dry_run=False),
        reasoning_loop=outer_reasoner,
    )

    assert result.returncode == 0
    assert result.completed is True
    assert starts == 1
    assert nested_callbacks == 0
    assert len(list((run_dir / "raw").glob("asset_usd_cli_launch_*.json"))) == 1
    assert len(list((run_dir / "raw").glob("asset_usd_cli_session_*.json"))) == 1
    assert len(list((run_dir / "raw").glob("asset_usd_cli_teardown_*.json"))) == 1


def test_batch_and_interactive_share_one_parent_lifecycle_each(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    batch_dir = tmp_path / "batch-agentic"
    interactive_dir = tmp_path / "interactive-agentic"
    events: list[tuple[str, str]] = []
    child_count = 0
    interactive_count = 0
    lease_runs: dict[int, str] = {}
    original_start = asset_runner._start_usd_cli_run_daemon_strict
    original_attach = asset_runner._attach_workflow_usd_cli_session
    original_stop = asset_runner._stop_usd_cli_run_daemon_strict
    _allow_provider_free_agentic_readiness(paths, monkeypatch)

    def start(**kwargs: Any) -> Any:
        run_name = Path(kwargs["run_dir"]).name
        events.append((run_name, "start"))
        lease = original_start(**kwargs)
        lease_runs[id(lease)] = run_name
        return lease

    def attach(**kwargs: Any) -> Any:
        events.append((Path(kwargs["run_dir"]).name, "attach"))
        return original_attach(**kwargs)

    def stop(**kwargs: Any) -> Any:
        events.append((lease_runs[id(kwargs["lease"])], "stop"))
        return original_stop(**kwargs)

    def child(**kwargs: Any) -> int:
        nonlocal child_count
        child_count += 1
        _complete_agentic_fixture_graph(Path(kwargs["run_dir"]))
        return 0

    def interactive(_session: AssetCoordinatorSession) -> int:
        nonlocal interactive_count
        interactive_count += 1
        _complete_agentic_fixture_graph(interactive_dir)
        return 0

    monkeypatch.setattr(asset_runner, "_start_usd_cli_run_daemon_strict", start)
    monkeypatch.setattr(asset_runner, "_attach_workflow_usd_cli_session", attach)
    monkeypatch.setattr(asset_runner, "_stop_usd_cli_run_daemon_strict", stop)
    monkeypatch.setattr(asset_runner, "_run_child_agent", child)

    batch = run_asset_workflow(
        replace(_agentic_config(paths, batch_dir), dry_run=False)
    )
    public = run_interactive_asset_workflow(
        replace(_agentic_config(paths, interactive_dir), dry_run=False),
        reasoning_loop=interactive,
    )

    assert batch.completed is True
    assert public.completed is True
    assert child_count == 1
    assert interactive_count == 1
    assert events == [
        ("batch-agentic", "start"),
        ("batch-agentic", "attach"),
        ("batch-agentic", "stop"),
        ("interactive-agentic", "start"),
        ("interactive-agentic", "attach"),
        ("interactive-agentic", "stop"),
    ]
    assert len(list((batch_dir / "raw").glob("asset_usd_cli_session_*.json"))) == 1
    assert (
        len(list((interactive_dir / "raw").glob("asset_usd_cli_session_*.json"))) == 1
    )


def test_public_agentic_real_codex_bridge_finishes_one_frozen_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the real Codex bridge regression")
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "agentic-real-codex"
    repo_root = Path(__file__).parents[4]
    target = paths["repo"] / "usd-cli"
    wrapper = paths["repo"] / "usd-cli-tel"

    monkeypatch.setattr(
        asset_runner,
        "resolve_package_owned_usd_cli_route",
        lambda _root: UsdCliPackageRoute(
            wrapper=wrapper.resolve(),
            target=target.resolve(),
            source_root=repo_root.resolve(),
            source_revision="d" * 40,
        ),
    )
    monkeypatch.setattr(
        asset_runner,
        "run_bounded_usd_cli_subprocess",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="usd-cli test\n"),
    )
    monkeypatch.setattr(
        asset_runner,
        "ensure_usd_cli_ovrtx_ready",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("agentic graph selection called a provider readiness path")
        ),
    )

    helper = tmp_path / "complete_agentic_graph.py"
    helper.write_text(
        """from importlib import metadata
from pathlib import Path
import sys

_installed_entry_points = metadata.entry_points


def _public_package_entry_points(**kwargs):
    if kwargs.get("group") == "world_understanding.model_backends":
        return ()
    return _installed_entry_points(**kwargs)


metadata.entry_points = _public_package_entry_points

from content_agent_workflows.asset_composition import (
    AssetExecutionGraph,
    AssetExecutionNode,
    begin_leaf,
    complete_leaf,
    freeze_execution_graph,
    leaf_directory,
)
from content_agent_workflows.asset_composition.catalog_adapters import (
    FOCUSED_VALIDATION_OPERATION_LEAF_ID,
    FocusedValidationOperationLeafInvocation,
)
from content_agent_workflows.common.artifacts import atomic_write_json
from content_agent_workflows.validation import (
    prepare_validation_operations,
    run_validation_operation,
)
from content_workflow_cli import asset_runner
from world_understanding.validation import (
    ValidationPlan,
    ValidationPlanStep,
    ValidationRequest,
    ValidationTemplateResult,
)


class FixtureExecutor:
    template_versions = {"physics_sane": "test.physics-sane.v1"}

    def plan(self, request, *, working_dir):
        del working_dir
        return ValidationPlan(
            steps=tuple(
                ValidationPlanStep(
                    template_name=name,
                    reason="Selected by the real bridge fixture.",
                )
                for name in request.requested_templates
            ),
            reasoning_summary="Preserved the selected bridge operation order.",
        )

    def run(self, template_name, context):
        del context
        return ValidationTemplateResult(
            template_name=template_name,
            status="passed",
        )

run_dir = Path(sys.argv[1])
state_path = run_dir / "asset_run.json"
request = asset_runner.load_verified_asset_request(state_path)
descriptor = next(
    item
    for item in request.leaf_catalog.descriptors
    if item.leaf_id == FOCUSED_VALIDATION_OPERATION_LEAF_ID
)
graph = AssetExecutionGraph.create(
    sole_coordinator_identity_digest=(
        request.sole_coordinator_identity.identity_digest
    ),
    prompt_digest=str(request.prompt_digest),
    source_digest=str(request.source_digest),
    configuration_digest=str(request.configuration_digest),
    reference_digest=str(request.reference_digest),
    leaf_catalog_digest=request.leaf_catalog.catalog_digest,
    nodes=[
        AssetExecutionNode(
            leaf_id=descriptor.leaf_id,
            requirement="required",
            descriptor_digest=descriptor.descriptor_digest,
            terminal_output=True,
        )
    ],
    omitted_leaf_ids=[
        item.leaf_id
        for item in request.leaf_catalog.descriptors
        if item.leaf_id != descriptor.leaf_id
    ],
)
graph_path = run_dir / "execution_graph.json"
atomic_write_json(graph_path, graph)
freeze_execution_graph(state_path, graph_path=graph_path, actor="real-codex")
begin_leaf(state_path, descriptor.leaf_id, actor="real-codex")
attempt = leaf_directory(state_path, descriptor.leaf_id)
invocation = attempt / "invocation.json"
result = attempt / "result.json"
validation_output = attempt / "validation"
executor = FixtureExecutor()
prepare_validation_operations(
    ValidationRequest(
        task_description="Validate the real public bridge fixture.",
        inputs=(request.source_asset,),
        requested_templates=("physics_sane",),
    ),
    output_dir=validation_output,
    config_base_dir=Path(request.source_asset).parent,
    executor=executor,
)
operation = run_validation_operation(
    validation_output,
    template_name="physics_sane",
    executor=executor,
)
atomic_write_json(
    invocation,
    FocusedValidationOperationLeafInvocation(
        output_dir=str(validation_output),
        template_name="physics_sane",
    ),
)
atomic_write_json(result, operation)
complete_leaf(
    state_path,
    descriptor.leaf_id,
    invocation_path=invocation,
    result_path=result,
    actor="real-codex",
)
""",
        encoding="utf-8",
    )

    app_dir = tmp_path / "bridge-app"
    app_dir.mkdir()
    shutil.copy2(
        Path(runner.__file__).with_name("codex_sdk_bridge.mjs"),
        app_dir / "codex_sdk_bridge.mjs",
    )
    shutil.copy2(
        Path(runner.__file__).with_name("descendant_reaper.py"),
        app_dir / "descendant_reaper.py",
    )
    monkeypatch.setattr(runner, "__file__", str(app_dir / "runner.py"))
    openai_modules = app_dir / "node_modules" / "@openai"
    codex_package = openai_modules / "codex"
    codex_package.mkdir(parents=True)
    (codex_package / "package.json").write_text(
        json.dumps(
            {
                "name": "@openai/codex",
                "version": "0.139.0",
                "bin": {"codex": "bin/codex.mjs"},
            }
        ),
        encoding="utf-8",
    )
    fake_codex = codex_package / "bin" / "codex.mjs"
    fake_codex.parent.mkdir()
    fake_codex.write_text(f"#!{node}\nprocess.exit(0);\n", encoding="utf-8")
    fake_codex.chmod(0o700)

    capture_path = tmp_path / "real-codex-invocation.json"
    sdk_package = openai_modules / "codex-sdk"
    sdk_package.mkdir()
    (sdk_package / "package.json").write_text(
        json.dumps(
            {
                "name": "@openai/codex-sdk",
                "version": "0.139.0",
                "type": "module",
                "exports": "./index.mjs",
            }
        ),
        encoding="utf-8",
    )
    (sdk_package / "index.mjs").write_text(
        "import fs from 'node:fs';\n"
        "import {spawnSync} from 'node:child_process';\n"
        "export class Codex {\n"
        "  constructor(options) { this.options = options; }\n"
        "  startThread() { return {run: async () => {\n"
        "    const identity = JSON.parse(fs.readFileSync(\n"
        "      this.options.env.CONTENT_WORKFLOW_PARENT_USD_CLI_SESSION_IDENTITY));\n"
        f"    const child = spawnSync({json.dumps(sys.executable)}, "
        f"[{json.dumps(str(helper))}, identity.run_dir], "
        "{cwd: identity.run_dir, env: {...this.options.env, "
        f"PYTHONPATH: {json.dumps(':'.join((str(repo_root), str(repo_root / 'agentic/packages/content_agent_workflows'), str(repo_root / 'agentic/packages/content_workflow_cli'))))}}}, "
        "encoding: 'utf8'});\n"
        "    if (child.status !== 0) throw new Error(child.stderr || child.stdout);\n"
        f"    fs.writeFileSync({json.dumps(str(capture_path))}, "
        "JSON.stringify({runDir: identity.run_dir, invocations: 1}));\n"
        "    return {finalResponse: 'completed one frozen graph', items: []};\n"
        "  }}; }\n"
        "}\n",
        encoding="utf-8",
    )

    config = replace(
        _agentic_config(paths, run_dir),
        repo_root=repo_root,
        dry_run=False,
        runner="codex",
        child_timeout_seconds=60,
    )
    result = run_asset_workflow(config)

    assert result.returncode == 0
    assert result.completed is True
    assert json.loads(capture_path.read_text(encoding="utf-8")) == {
        "runDir": str(run_dir),
        "invocations": 1,
    }
    bridge_requests = list((run_dir / "raw").glob("asset_composition_request.json"))
    assert len(bridge_requests) == 1
    bridge_request = json.loads(bridge_requests[0].read_text(encoding="utf-8"))
    assert bridge_request["workflow"] == "asset.run"
    assert (
        "There is no built-in total domain or stage order" in bridge_request["prompt"]
    )
    run = load_verified_run(run_dir / "asset_run.json")
    assert run.selected_mode == "agentic"
    assert run.terminal_status == "completed"
    assert run.graph_terminal_receipt is not None
    terminal = json.loads(
        Path(run.graph_terminal_receipt.path).read_text(encoding="utf-8")
    )
    assert terminal["selected_leaf_ids"] == [FOCUSED_VALIDATION_OPERATION_LEAF_ID]
    assert terminal["omitted_leaf_dispositions"] == {
        item.leaf_id: "not_requested"
        for item in discover_repository_asset_leaf_catalog().descriptors
        if item.leaf_id != FOCUSED_VALIDATION_OPERATION_LEAF_ID
    }
    assert len(terminal["resource_release_receipts"]) == 1


def test_public_asset_cli_requires_explicit_disjoint_mode_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _inputs(tmp_path)
    agentic_run = tmp_path / "agentic-cli"

    assert (
        main(
            [
                "asset",
                "run",
                "--usd",
                str(paths["source"]),
                "--prompt",
                "Select the necessary opaque leaf.",
                "--leaf-catalog",
                str(paths["catalog"]),
                "--repo-root",
                str(paths["repo"]),
                "--output-dir",
                str(agentic_run),
                "--dry-run",
            ]
        )
        == 0
    )
    request = json.loads((agentic_run / "request.json").read_text(encoding="utf-8"))
    assert request["selected_mode"] == "agentic"

    assert (
        main(
            [
                "asset",
                "run",
                "--usd",
                str(paths["source"]),
                "--prompt",
                "Must not infer compatibility from fixed inputs.",
                "--leaf-catalog",
                str(paths["catalog"]),
                "--joint-config",
                str(paths["joint"]),
                "--materials-yaml",
                str(paths["materials"]),
                "--repo-root",
                str(paths["repo"]),
                "--output-dir",
                str(tmp_path / "mixed-mode"),
                "--dry-run",
            ]
        )
        == 2
    )
    assert "cannot accept fixed-stage domain configuration" in capsys.readouterr().err


def test_asset_run_stages_and_binds_exact_source_dependency_closure(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    dependency = paths["repo"] / "geometry.usda"
    dependency.write_text('#usda 1.0\ndef Xform "Body" {}\n', encoding="utf-8")
    paths["source"].write_text(
        "#usda 1.0\n(\n    subLayers = [@geometry.usda@]\n)\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"

    run_asset_workflow(_config(paths, run_dir))

    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    staging = request["source_staging"]
    staged_paths = [Path(item["path"]) for item in staging["staged_dependencies"]]
    assert staged_paths == [
        run_dir / "inputs" / "asset_source" / "cabinet.usda",
        run_dir / "inputs" / "asset_source" / "geometry.usda",
    ]
    assert all(run_dir in path.parents for path in staged_paths)
    assert [item["sha256"] for item in staging["original_dependencies"]] == [
        item["sha256"] for item in staging["staged_dependencies"]
    ]
    staged_paths[1].chmod(0o600)
    staged_paths[1].write_text(
        '#usda 1.0\ndef Xform "Changed" {}\n',
        encoding="utf-8",
    )

    with pytest.raises(AssetCompositionStateError, match="identity changed"):
        resume_asset_workflow(run_dir, dry_run=True)


def test_asset_run_accepts_path_sorted_staging_for_root_first_durable_closure(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    dependency = paths["repo"] / "a-geometry.usda"
    dependency.write_text('#usda 1.0\ndef Xform "Body" {}\n', encoding="utf-8")
    source = paths["repo"] / "z-cabinet.usda"
    source.write_text(
        "#usda 1.0\n(\n    subLayers = [@a-geometry.usda@]\n)\n",
        encoding="utf-8",
    )
    paths["source"] = source
    run_dir = tmp_path / "run"

    run_asset_workflow(_config(paths, run_dir))

    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert [
        Path(item["path"]).name
        for item in request["source_staging"]["staged_dependencies"]
    ] == [
        "a-geometry.usda",
        "z-cabinet.usda",
    ]
    run = load_verified_run(run_dir / "asset_run.json")
    assert [
        Path(item.path).name for item in [run.source_asset, *run.source_dependencies]
    ] == [
        "z-cabinet.usda",
        "a-geometry.usda",
    ]

    result = resume_asset_workflow(run_dir, dry_run=True)

    assert result.returncode == 0


def test_asset_run_rejects_unresolved_source_closure_before_child(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    paths["source"].write_text(
        "#usda 1.0\n(\n    subLayers = [@missing.usda@]\n)\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="dependency closure is unresolved"):
        run_asset_workflow(_config(paths, run_dir))

    assert not run_dir.exists()


def test_asset_run_removes_fresh_root_when_source_contract_build_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        asset_runner,
        "_source_staging_contract",
        lambda _staged: (_ for _ in ()).throw(
            RuntimeError("staged source model rejected")
        ),
    )

    with pytest.raises(RuntimeError, match="staged source model rejected"):
        run_asset_workflow(_config(paths, run_dir))

    assert not run_dir.exists()


def test_asset_run_removes_fresh_root_when_source_staging_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        asset_runner,
        "_source_staging_contract",
        lambda _staged: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt):
        run_asset_workflow(_config(paths, run_dir))

    assert not run_dir.exists()


def test_uncommitted_fresh_run_cleanup_refuses_durable_request(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request_path = run_dir / "request.json"
    request_path.write_text('{"durable":true}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="durable asset request exists"):
        asset_runner._remove_uncommitted_fresh_run(
            run_dir,
            request_path=request_path,
        )

    assert run_dir.is_dir()
    assert request_path.is_file()


def test_cad_modeling_lease_maps_acquisition_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnavailableLock:
        def __init__(self, _path: str) -> None:
            pass

        def acquire(self, *, timeout: int) -> None:
            assert timeout == 0
            raise PermissionError("read-only lease directory")

    monkeypatch.setattr(asset_state, "_PersistentAttemptFileLock", UnavailableLock)
    state_path = tmp_path / "run" / "asset_run.json"

    with pytest.raises(
        AssetCompositionStateError,
        match="CAD modeling execution lease is unavailable",
    ):
        with asset_state._cad_attempt_execution_lease(
            state_path,
            run_id="lease-error",
            stage_attempt=1,
        ):
            pytest.fail("an unavailable lease must not enter its body")


def test_cad_modeling_lease_release_suppresses_record_unlink_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "attempt.lock"
    lock = asset_state._PersistentAttemptFileLock(lock_path)
    lock.acquire(timeout=0)
    record_path = tmp_path / "attempt.json"
    record_path.write_text('{"token":"owned"}\n', encoding="utf-8")
    lease = asset_state._CadAttemptExecutionLease(
        lock=lock,
        record_path=record_path,
        token="owned",
        run_id="lease-release",
        stage_attempt=1,
        stage="cad_modeling",
    )
    original_unlink = Path.unlink

    def unavailable_unlink(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        if path == record_path:
            raise PermissionError("record is busy")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unavailable_unlink)

    lease.release()

    assert lock.is_locked is False


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX crash semantics")
def test_cad_modeling_lease_reacquires_after_unclean_process_exit(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "run" / "asset_run.json"
    state_path.parent.mkdir()
    marker = tmp_path / "lease-acquired.txt"
    crash_script = tmp_path / "crash_with_lease.py"
    crash_script.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "import content_agent_workflows.asset_composition.state as state\n"
        "state_path = Path(sys.argv[1])\n"
        "with state._cad_attempt_execution_lease(\n"
        "    state_path, run_id='crash-run', stage_attempt=1\n"
        "):\n"
        "    Path(sys.argv[2]).write_text('owned', encoding='ascii')\n"
        "    os._exit(23)\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(str(item) for item in sys.path)

    crashed = subprocess.run(
        [sys.executable, str(crash_script), str(state_path), str(marker)],
        env=environment,
        check=False,
    )

    assert crashed.returncode == 23
    assert marker.read_text(encoding="ascii") == "owned"
    lease_path = asset_state._cad_attempt_lease_path(
        state_path,
        run_id="crash-run",
        stage_attempt=1,
    )
    assert lease_path.is_file()
    assert lease_path.with_suffix(".json").is_file()
    crashed_inode = (lease_path.stat().st_dev, lease_path.stat().st_ino)

    with asset_state._cad_attempt_execution_lease(
        state_path,
        run_id="crash-run",
        stage_attempt=1,
    ) as recovered:
        recovered.require_owner(run_id="crash-run", stage_attempt=1)
        assert (lease_path.stat().st_dev, lease_path.stat().st_ino) == crashed_inode

    assert lease_path.is_file()
    assert (lease_path.stat().st_dev, lease_path.stat().st_ino) == crashed_inode
    assert not lease_path.with_suffix(".json").exists()


def test_authoring_provider_starts_in_isolated_no_shell_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class CompletedProcess:
        pid = 12345
        returncode: int | None = 0
        stdout = None
        stderr = None

        def communicate(self) -> tuple[str, str]:
            return "cad output", ""

        def poll(self) -> int:
            return 0

    def fake_popen(argv: list[str], **kwargs: object) -> CompletedProcess:
        captured["argv"] = argv
        captured.update(kwargs)
        return CompletedProcess()

    monkeypatch.setattr(asset_state.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        asset_state,
        "_signal_cad_process_group",
        lambda *_args: False,
    )

    completed = asset_state._run_geometry_authoring_provider(
        request_path=tmp_path / "request.json",
        workspace=tmp_path / "workspace",
        artifact_dir=tmp_path / "artifacts",
        source_manifest_path=tmp_path / "result.json",
        command=("geometry-authoring-provider", "--profile", "release"),
    )

    assert completed.stdout == "cad output"
    assert captured["argv"] == [
        "geometry-authoring-provider",
        "--profile",
        "release",
        "author-source",
        str(tmp_path / "request.json"),
        "--workspace",
        str(tmp_path / "workspace"),
        "--artifact-dir",
        str(tmp_path / "artifacts"),
        "--out",
        str(tmp_path / "result.json"),
    ]
    assert captured["shell"] is False
    assert captured["start_new_session"] is True


def test_authoring_provider_process_publishes_only_a_canonical_source_bundle(
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.geometry-authoring-request.v1"
                ),
                "request_id": "asset-authoring-proof",
                "prompt": "Create a parameterized housing.",
                "references": [],
                "target_profile": "rigid-pick-place",
                "requested_formats": ["usda"],
                "parameters": [{"name": "width", "value": 24.0}],
            }
        ),
        encoding="utf-8",
    )
    worker = tmp_path / "provider_worker.py"
    worker.write_text(
        "import hashlib, json, sys\n"
        "from pathlib import Path\n"
        "from geometry_authoring_contracts import (\n"
        " GeometryArtifactBinding, GeometryAuthoringProviderIdentity, GeometryAuthoringRequest,\n"
        " GeometryCoordinateSystem, GeometryPartBinding,\n"
        " GeometryRepresentationBinding, GeometryRightsAssertion,\n"
        " GeometrySemanticParameter, GeometrySourceBundle,\n"
        " GeometrySourceProvenance, GeometryVerificationAssertion,\n"
        " geometry_authoring_request_digest, geometry_source_bundle_id,\n"
        ")\n"
        "request_path = Path(sys.argv[2])\n"
        "out = Path(sys.argv[sys.argv.index('--out') + 1])\n"
        "request = json.loads(request_path.read_text(encoding='utf-8'))\n"
        "request_model = GeometryAuthoringRequest.model_validate(request)\n"
        "usd = out.parent / 'provider-output.usda'\n"
        "usd.write_text('#usda 1.0\\ndef Xform \\\"Asset\\\" {}\\n', encoding='utf-8')\n"
        "content = usd.read_bytes()\n"
        "producer = GeometryAuthoringProviderIdentity(provider_id='fixture-provider', provider_version='1.0')\n"
        "coordinate = GeometryCoordinateSystem(meters_per_unit=0.001)\n"
        "representations = (GeometryRepresentationBinding(representation_id='render', role='render_geometry', format='usda', media_type='model/vnd.usda', artifact=GeometryArtifactBinding(path=usd.name, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content)), root_prim_path='/Asset'),)\n"
        "parts = (GeometryPartBinding(part_id='housing', name='Housing', representation_ids=('render',)),)\n"
        "parameters = tuple(GeometrySemanticParameter(name=item['name'], value=item['value'], unit=item.get('unit')) for item in request.get('parameters', ()))\n"
        "assertions = (GeometryVerificationAssertion(assertion_id='provider-topology', status='passed', summary='Provider reports valid topology.'),)\n"
        "provenance = GeometrySourceProvenance(request_digest=geometry_authoring_request_digest(request_model))\n"
        "rights = GeometryRightsAssertion(assertion='Fixture output is authorized.')\n"
        "revision = hashlib.sha256(request_path.read_bytes()).hexdigest()\n"
        "identity = geometry_source_bundle_id(provider=producer, source_revision=revision, coordinate_system=coordinate, representations=representations, parts=parts, parameters=parameters, verification_assertions=assertions, provenance=provenance, rights=rights)\n"
        "bundle = GeometrySourceBundle(bundle_id=identity, producer=producer, source_revision=revision, coordinate_system=coordinate, representations=representations, parts=parts, parameters=parameters, verification_assertions=assertions, provenance=provenance, rights=rights)\n"
        "out.write_text(bundle.model_dump_json(indent=2) + '\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    artifacts = tmp_path / "artifacts"
    workspace.mkdir()
    artifacts.mkdir()
    source_manifest_path = tmp_path / "geometry.source.json"

    completed = asset_state._run_geometry_authoring_provider(
        request_path=request_path,
        workspace=workspace,
        artifact_dir=artifacts,
        source_manifest_path=source_manifest_path,
        command=(sys.executable, str(worker)),
    )

    assert completed.returncode == 0, completed.stderr
    bundle = asset_state.load_external_source_bundle(source_manifest_path)
    assert bundle.producer.provider_id == "fixture-provider"
    assert bundle.parts[0].part_id == "housing"
    assert bundle.parameters[0].value == 24.0
    assert bundle.verification_assertions[0].status == "passed"
    assert bundle.representations[0].artifact.path == "provider-output.usda"


def test_authoring_provider_refuses_without_posix_group_supervision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asset_state.os, "name", "nt")
    monkeypatch.setattr(
        asset_state.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("unsupported platform launched CAD"),
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="require POSIX process-group supervision",
    ):
        asset_state._run_geometry_authoring_provider(
            request_path=tmp_path / "request.json",
            workspace=tmp_path / "workspace",
            artifact_dir=tmp_path / "artifacts",
            source_manifest_path=tmp_path / "result.json",
            command=("geometry-authoring-provider",),
        )


def test_authoring_provider_cancellation_escalates_to_process_group_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TestCancellation(BaseException):
        pass

    class RunningProcess:
        pid = 23456
        returncode: int | None = None
        stdout = None
        stderr = None

        def communicate(self) -> tuple[str, str]:
            raise TestCancellation

        def poll(self) -> int | None:
            return self.returncode

    process = RunningProcess()
    signals: list[signal.Signals] = []

    def signal_group(
        _process_group_id: int,
        signal_number: signal.Signals,
    ) -> bool:
        signals.append(signal_number)
        if signal_number == signal.SIGKILL:
            process.returncode = -signal.SIGKILL
        return True

    def wait_for_group(
        _process: RunningProcess,
        *,
        timeout: float,
    ) -> bool:
        del timeout
        return signals[-1] == signal.SIGKILL

    monkeypatch.setattr(asset_state.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(asset_state, "_signal_cad_process_group", signal_group)
    monkeypatch.setattr(asset_state, "_wait_for_cad_process_group_exit", wait_for_group)

    with pytest.raises(TestCancellation):
        asset_state._run_geometry_authoring_provider(
            request_path=tmp_path / "request.json",
            workspace=tmp_path / "workspace",
            artifact_dir=tmp_path / "artifacts",
            source_manifest_path=tmp_path / "result.json",
            command=("geometry-authoring-provider",),
        )

    assert signals == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_cad_sigterm_reaps_process_group_before_attempt_lease_releases(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "run" / "asset_run.json"
    state_path.parent.mkdir()
    request_path = tmp_path / "request.json"
    request_path.write_text("{}\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    artifact_dir = tmp_path / "artifacts"
    snapshot_path = tmp_path / "snapshot.json"
    leader_path = tmp_path / "cad-leader.pid"
    descendant_path = tmp_path / "cad-descendant.pid"
    late_write_path = tmp_path / "orphan-write.txt"
    cad_script = tmp_path / "cad_job.py"
    cad_script.write_text(
        "import os, pathlib, signal, sys, time\n"
        "leader = pathlib.Path(sys.argv[1])\n"
        "descendant = pathlib.Path(sys.argv[2])\n"
        "late_write = pathlib.Path(sys.argv[3])\n"
        "leader.write_text(str(os.getpid()), encoding='ascii')\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    descendant.touch()\n"
        "    time.sleep(0.05)\n"
        "    descendant.write_text(str(os.getpid()), encoding='ascii')\n"
        "    time.sleep(1.0)\n"
        "    late_write.write_text('orphan wrote', encoding='ascii')\n"
        "    os._exit(0)\n"
        "time.sleep(30.0)\n",
        encoding="utf-8",
    )
    wrapper_script = tmp_path / "cad_wrapper.py"
    wrapper_script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "import content_agent_workflows.asset_composition.state as state\n"
        "state._CAD_PROCESS_TERMINATION_GRACE_SECONDS = 0.5\n"
        "state._CAD_PROCESS_KILL_GRACE_SECONDS = 0.5\n"
        "state_path = Path(sys.argv[1])\n"
        "with state._cad_attempt_execution_lease(\n"
        "    state_path, run_id='sigterm-run', stage_attempt=1\n"
        "):\n"
        "    state._run_geometry_authoring_provider(\n"
        "        request_path=Path(sys.argv[2]),\n"
        "        workspace=Path(sys.argv[3]),\n"
        "        artifact_dir=Path(sys.argv[4]),\n"
        "        source_manifest_path=Path(sys.argv[5]),\n"
        "        command=(sys.executable, *sys.argv[6:10]),\n"
        "    )\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(str(item) for item in sys.path)
    wrapper = subprocess.Popen(
        [
            sys.executable,
            str(wrapper_script),
            str(state_path),
            str(request_path),
            str(workspace),
            str(artifact_dir),
            str(snapshot_path),
            str(cad_script),
            str(leader_path),
            str(descendant_path),
            str(late_write_path),
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    leader_pid: int | None = None
    descendant_pid: int | None = None

    def read_ready_pid(path: Path) -> int | None:
        try:
            candidate = int(path.read_text(encoding="ascii"))
        except (FileNotFoundError, ValueError):
            return None
        return candidate if candidate > 0 else None

    try:
        deadline = time.monotonic() + 5.0
        while leader_pid is None or descendant_pid is None:
            leader_pid = leader_pid or read_ready_pid(leader_path)
            descendant_pid = descendant_pid or read_ready_pid(descendant_path)
            if leader_pid is not None and descendant_pid is not None:
                break
            if wrapper.poll() is not None or time.monotonic() >= deadline:
                stdout, stderr = wrapper.communicate(timeout=1.0)
                pytest.fail(
                    "CAD wrapper did not start its descendant: "
                    f"stdout={stdout!r}, stderr={stderr!r}"
                )
            time.sleep(0.01)

        os.kill(wrapper.pid, signal.SIGTERM)
        deadline = time.monotonic() + 2.0
        while True:
            try:
                os.kill(leader_pid, 0)
            except ProcessLookupError:
                break
            if time.monotonic() >= deadline:
                pytest.fail("CAD command leader did not stop after parent SIGTERM")
            time.sleep(0.01)

        with pytest.raises(
            AssetCompositionStateError,
            match="Another executor already owns",
        ):
            with asset_state._cad_attempt_execution_lease(
                state_path,
                run_id="sigterm-run",
                stage_attempt=1,
            ):
                pytest.fail("attempt lease released before CAD group teardown")

        stdout, stderr = wrapper.communicate(timeout=5.0)
        assert wrapper.returncode == 128 + signal.SIGTERM, (stdout, stderr)
        with asset_state._cad_attempt_execution_lease(
            state_path,
            run_id="sigterm-run",
            stage_attempt=1,
        ):
            pass
        time.sleep(1.0)
        assert not late_write_path.exists()
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.communicate(timeout=2.0)
        if leader_pid is not None:
            try:
                os.killpg(leader_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if descendant_pid is not None:
            try:
                os.kill(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_geometry_attempt_lease_rejects_parallel_owner(tmp_path: Path) -> None:
    state_path = tmp_path / "run" / "asset_run.json"
    state_path.parent.mkdir()

    with asset_state._geometry_attempt_execution_lease(
        state_path,
        run_id="shared-run",
        stage_attempt=2,
    ):
        with pytest.raises(
            AssetCompositionStateError,
            match="Another executor already owns this Geometry stage attempt",
        ):
            with asset_state._geometry_attempt_execution_lease(
                state_path,
                run_id="shared-run",
                stage_attempt=2,
            ):
                pytest.fail("parallel Geometry executor acquired the same attempt")


def test_geometry_attempt_lease_reuses_its_persistent_inode(tmp_path: Path) -> None:
    state_path = tmp_path / "run" / "asset_run.json"
    state_path.parent.mkdir()
    lease_path = asset_state._attempt_lease_path(
        state_path,
        run_id="geometry-inode-run",
        stage="geometry",
        stage_attempt=2,
    )

    with asset_state._geometry_attempt_execution_lease(
        state_path,
        run_id="geometry-inode-run",
        stage_attempt=2,
    ):
        first_inode = (lease_path.stat().st_dev, lease_path.stat().st_ino)

    assert lease_path.is_file()
    assert (lease_path.stat().st_dev, lease_path.stat().st_ino) == first_inode
    with asset_state._geometry_attempt_execution_lease(
        state_path,
        run_id="geometry-inode-run",
        stage_attempt=2,
    ):
        assert (lease_path.stat().st_dev, lease_path.stat().st_ino) == first_inode

    assert lease_path.is_file()


def test_attempt_lease_identity_is_run_based_across_mount_paths(tmp_path: Path) -> None:
    first = tmp_path / "mount-a" / "same-run" / "asset_run.json"
    second = tmp_path / "mount-b" / "same-run" / "asset_run.json"

    first_path = asset_state._attempt_lease_path(
        first,
        run_id="logical-run-id",
        stage="cad_modeling",
        stage_attempt=1,
    )
    second_path = asset_state._attempt_lease_path(
        second,
        run_id="logical-run-id",
        stage="cad_modeling",
        stage_attempt=1,
    )

    assert first_path.name == second_path.name


def test_geometry_authoring_command_uses_admin_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CONTENT_AGENT_GEOMETRY_AUTHORING_COMMAND",
        '"/opt/geometry provider/bin/authoring-provider" --profile release',
    )

    assert asset_state._resolved_geometry_authoring_command(None) == (
        "/opt/geometry provider/bin/authoring-provider",
        "--profile",
        "release",
    )


def test_explicit_geometry_authoring_command_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CONTENT_AGENT_GEOMETRY_AUTHORING_COMMAND",
        "ignored-command",
    )

    assert asset_state._resolved_geometry_authoring_command(
        ("custom-cad", "--strict")
    ) == (
        "custom-cad",
        "--strict",
    )


def test_asset_run_freezes_and_reverifies_material_library_dependencies(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    dependency = paths["repo"] / "library-layer.usda"
    dependency.write_text('#usda 1.0\ndef Material "Original" {}\n', encoding="utf-8")
    paths["materials_usd"].write_text(
        "#usda 1.0\n(\n    subLayers = [@library-layer.usda@]\n)\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"

    run_asset_workflow(_config(paths, run_dir))

    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert [item["path"] for item in request["materials_usd_dependencies"]] == [
        str(dependency.resolve())
    ]
    dependency.write_text('#usda 1.0\ndef Material "Changed" {}\n', encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="dependency .*identity changed",
    ):
        resume_asset_workflow(run_dir, dry_run=True)


def test_asset_run_rejects_missing_material_manifest_before_fallback_parse(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    missing = paths["repo"] / "missing-materials.yaml"

    with pytest.raises(
        FileNotFoundError,
        match=r"materials manifest is not a file: .*missing-materials\.yaml",
    ):
        run_asset_workflow(
            replace(
                _config(paths, tmp_path / "run"),
                materials_yaml=missing,
            )
        )


def test_asset_run_rejects_duplicate_material_references(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)

    with pytest.raises(ValueError, match="Material references must not repeat"):
        run_asset_workflow(
            replace(
                _config(paths, tmp_path / "run"),
                reference_images=[paths["reference"], paths["reference"]],
            )
        )


def test_asset_review_binds_decisions_then_prepares_same_run_for_resume(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_asset_workflow(_config(paths, run_dir))
    state_path = run_dir / "asset_run.json"
    _plan_and_begin(state_path, "articulation")
    candidates_dir = stage_directory(state_path, "articulation")
    candidates_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidates_dir / "candidates.json"
    candidates.write_text('{"candidate_ids":["drawer"]}\n', encoding="utf-8")
    _record_await_review(state_path, candidates)
    require_review(state_path, candidates_path=candidates)
    decisions = tmp_path / "decisions.json"
    decisions.write_text('{"drawer":"accept"}\n', encoding="utf-8")

    result = review_asset_workflow(
        run_dir,
        decisions_path=decisions,
        reviewer="asset-owner",
        dry_run=True,
    )

    assert result.returncode == 0
    run = load_verified_run(state_path)
    assert run.stages["articulation"].status == "ready"
    assert run.stages["articulation"].review_decisions is not None
    assert (run_dir / "agent_review_prompt.md").is_file()


def test_asset_resume_fails_closed_when_frozen_request_changes(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_asset_workflow(_config(paths, run_dir))
    request_path = run_dir / "request.json"
    request_path.write_bytes(request_path.read_bytes() + b" ")

    with pytest.raises(AssetCompositionStateError, match="identity changed"):
        resume_asset_workflow(run_dir, dry_run=True)


def test_pre_library_request_is_rejected_before_resume_execution(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_asset_workflow(_config(paths, run_dir))
    request = asset_runner.AssetRunRequest.model_validate(
        json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    ).model_copy(update={"materials_usd": None, "materials_usd_binding": None})
    run = load_verified_run(run_dir / "asset_run.json")

    with pytest.raises(
        AssetCompositionStateError,
        match="predates the required materials library identity",
    ):
        verify_frozen_asset_inputs(
            request,
            run_source=run.source_asset,
        )


def test_asset_resume_recovers_only_a_stopped_current_stage(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_asset_workflow(_config(paths, run_dir))
    state_path = run_dir / "asset_run.json"

    with pytest.raises(AssetCompositionStateError, match="accepted only"):
        resume_asset_workflow(
            run_dir,
            recovery_reason="not stopped",
            dry_run=True,
        )

    _plan_and_begin(state_path, "articulation")
    interrupted_attempt = stage_directory(state_path, "articulation")
    fail_stage(
        state_path,
        "articulation",
        reason="simulated agent interruption",
        actor="test",
    )
    result = resume_asset_workflow(
        run_dir,
        recovery_reason="confirmed prior agent exited",
        dry_run=True,
    )

    assert result.returncode == 0
    run = load_verified_run(state_path)
    assert run.terminal_status == "active"
    assert run.stages["articulation"].status == "ready"
    assert stage_directory(state_path, "articulation") == interrupted_attempt


def test_agent_config_json_reports_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        asset_runner._merge_json_objects(['{"model":'])


def test_frozen_file_binding_rejects_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.yaml"
    target.write_text("value: 1\n", encoding="utf-8")
    link = tmp_path / "link.yaml"
    link.symlink_to(target)

    with pytest.raises(FileNotFoundError, match="not a regular file"):
        asset_runner._frozen_file_binding(link)


@pytest.mark.parametrize("input_name", ["joint", "materials", "reference"])
def test_asset_resume_fails_closed_when_frozen_input_changes(
    tmp_path: Path,
    input_name: str,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_asset_workflow(_config(paths, run_dir))
    paths[input_name].write_bytes(paths[input_name].read_bytes() + b"changed")

    with pytest.raises(AssetCompositionStateError, match="identity changed"):
        resume_asset_workflow(run_dir, dry_run=True)


def test_public_cli_exposes_asset_run(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    exit_code = main(
        [
            "asset",
            "run",
            "--usd",
            str(paths["source"]),
            "--prompt",
            "Find joints, materialize, texture, add physics, validate, and package.",
            "--compatibility-fixed-order",
            "--joint-config",
            str(paths["joint"]),
            "--materials-yaml",
            str(paths["materials"]),
            "--repo-root",
            str(paths["repo"]),
            "--output-dir",
            str(run_dir),
            "--physics-validation-mode",
            "schema-readback",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert (run_dir / "asset_run.json").is_file()
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    prompt_reference = run_dir / "inputs" / "prompt-reference.md"
    assert request["reference_images"] == []
    assert request["reference_files"] == [str(prompt_reference)]
    assert request["physics_validation_mode"] == "schema_readback"
    assert "Find joints, materialize" in prompt_reference.read_text(encoding="utf-8")


def test_public_asset_cli_launches_codex_child_with_asset_workflow_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    bridge_requests: list[dict[str, object]] = []

    def finish_bridge_process(**kwargs: object) -> int:
        command = cast(list[str], kwargs["command"])
        assert command[:2] == [
            "node",
            str(Path(runner.__file__).with_name("codex_sdk_bridge.mjs")),
        ]
        request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        bridge_requests.append(request)
        Path(request["child_final_path"]).write_text("stopped\n", encoding="utf-8")
        Path(request["items_path"]).write_text("[]\n", encoding="utf-8")
        Path(request["result_path"]).write_text("{}\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        runner,
        "_run_subprocess_with_timeout",
        finish_bridge_process,
    )

    exit_code = main(
        [
            "asset",
            "run",
            "--usd",
            str(paths["source"]),
            "--prompt",
            "Compose and validate the asset.",
            "--compatibility-fixed-order",
            "--joint-config",
            str(paths["joint"]),
            "--materials-yaml",
            str(paths["materials"]),
            "--repo-root",
            str(paths["repo"]),
            "--output-dir",
            str(run_dir),
            "--runner",
            "codex",
            "--child-timeout",
            "20",
        ]
    )

    assert exit_code == 1
    assert len(bridge_requests) == 1
    assert bridge_requests[0]["workflow"] == "asset.run"
    evidence_request = json.loads(
        (run_dir / "raw" / "asset_composition_request.json").read_text(encoding="utf-8")
    )
    assert evidence_request["workflow"] == "asset.run"


def test_public_asset_cli_real_codex_bridge_receives_parent_session_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.common import usd_cli_session

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the real Codex bridge regression")
    paths = _inputs(tmp_path)
    paths["source"].write_text(
        '#usda 1.0\n\ndef Cube "Cabinet" {}\n',
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    parent_state = tmp_path / ".usd-cli"
    parent_state.mkdir()
    (parent_state / "config.toml").write_text(
        "[render]\n"
        'renderer = "remote"\n'
        'remote_url = "https://ovrtx.example.test"\n'
        'remote_api_key = "parent-project-secret"\n',
        encoding="utf-8",
    )
    target = paths["repo"] / "usd-cli"
    wrapper = paths["repo"] / "usd-cli-tel"
    python_roots = [
        Path(__file__).parents[4] / USD_CLI_SOURCE_PATH / "src",
        Path(__file__).parents[2] / "content_agent_workflows",
        Path(__file__).parents[2] / "content_workflow_cli",
    ]
    monkeypatch.syspath_prepend(str(python_roots[0]))
    daemon_environment = tmp_path / "usd-cli-python"
    venv.EnvBuilder(with_pip=False).create(daemon_environment)
    daemon_site_packages = (
        daemon_environment
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    daemon_site_packages.joinpath("asset-launcher-test.pth").write_text(
        "\n".join([*(str(path) for path in python_roots), *site.getsitepackages()])
        + "\n",
        encoding="utf-8",
    )
    daemon_python = daemon_environment / "bin" / "python"
    target_launcher_text = (
        f"#!{daemon_python}\n"
        "import sys\n"
        + "\n".join(
            f"sys.path.insert(0, {json.dumps(str(path))})" for path in python_roots
        )
        + "\nfrom usd_cli.main import run\nrun()\n"
    )
    wrapper_launcher_text = (
        f"#!{daemon_python}\n"
        + "\n".join(
            f"import sys; sys.path.insert(0, {json.dumps(str(path))})"
            for path in python_roots
        )
        + f"\nimport os; os.environ.setdefault('USD_CLI_TEL_TARGET', "
        f"{json.dumps(str(target))})\n" + "from usd_telemetry.main import run\nrun()\n"
    )
    target.write_text(target_launcher_text, encoding="utf-8")
    wrapper.write_text(wrapper_launcher_text, encoding="utf-8")
    target.chmod(0o700)
    wrapper.chmod(0o700)
    route = runner.UsdCliTelemetryRoute(
        status="active",
        wrapper_path=wrapper,
        target_path=target,
    )
    package_route = UsdCliPackageRoute(
        wrapper=wrapper.resolve(),
        target=target.resolve(),
        source_root=paths["repo"].resolve(),
        source_revision="d" * 40,
    )
    monkeypatch.setattr(
        runner,
        "resolve_package_owned_usd_cli_route",
        lambda _repo: package_route,
    )
    monkeypatch.setattr(
        usd_cli_session,
        "resolve_package_owned_usd_cli_route",
        lambda _repo: package_route,
    )
    monkeypatch.setattr(
        asset_runner,
        "_prepare_usd_cli_telemetry_route",
        lambda *_args, **_kwargs: route,
    )
    monkeypatch.setattr(
        asset_runner,
        "_start_usd_cli_run_daemon_strict",
        runner._start_usd_cli_run_daemon_strict,
    )
    monkeypatch.setattr(
        asset_runner,
        "_attach_workflow_usd_cli_session",
        runner._attach_workflow_usd_cli_session,
    )
    monkeypatch.setattr(
        asset_runner,
        "_stop_usd_cli_run_daemon_strict",
        runner._stop_usd_cli_run_daemon_strict,
    )
    monkeypatch.setattr(
        asset_runner,
        "_activate_usd_cli_child_config",
        runner._activate_usd_cli_child_config,
    )
    app_dir = tmp_path / "bridge-app"
    app_dir.mkdir()
    bridge_path = app_dir / "codex_sdk_bridge.mjs"
    shutil.copy2(
        Path(runner.__file__).with_name("codex_sdk_bridge.mjs"),
        bridge_path,
    )
    shutil.copy2(
        Path(runner.__file__).with_name("descendant_reaper.py"),
        app_dir / "descendant_reaper.py",
    )
    monkeypatch.setattr(runner, "__file__", str(app_dir / "runner.py"))

    openai_modules = app_dir / "node_modules" / "@openai"
    codex_package = openai_modules / "codex"
    codex_package.mkdir(parents=True)
    (codex_package / "package.json").write_text(
        json.dumps(
            {
                "name": "@openai/codex",
                "version": "0.139.0",
                "bin": {"codex": "bin/codex.mjs"},
            }
        ),
        encoding="utf-8",
    )
    fake_codex = codex_package / "bin" / "codex.mjs"
    fake_codex.parent.mkdir()
    fake_codex.write_text(f"#!{node}\nprocess.exit(0);\n", encoding="utf-8")
    fake_codex.chmod(0o700)

    capture_path = tmp_path / "codex-child-environment.json"
    sdk_package = openai_modules / "codex-sdk"
    sdk_package.mkdir()
    (sdk_package / "package.json").write_text(
        json.dumps(
            {
                "name": "@openai/codex-sdk",
                "version": "0.139.0",
                "type": "module",
                "exports": "./index.mjs",
            }
        ),
        encoding="utf-8",
    )
    (sdk_package / "index.mjs").write_text(
        "import fs from 'node:fs';\n"
        "import crypto from 'node:crypto';\n"
        "import {spawnSync} from 'node:child_process';\n"
        "export class Codex {\n"
        "  constructor(options) {\n"
        "    this.options = options;\n"
        "  }\n"
        "  startThread() { return {run: async () => {\n"
        "    const identityBytes = fs.readFileSync("
        "this.options.env.CONTENT_WORKFLOW_PARENT_USD_CLI_SESSION_IDENTITY);\n"
        "    const identityDigest = crypto.createHash('sha256')"
        ".update(identityBytes).digest('hex');\n"
        "    if (identityDigest !== this.options.env."
        "CONTENT_WORKFLOW_PARENT_USD_CLI_SESSION_IDENTITY_SHA256) "
        "throw new Error('parent session identity digest changed');\n"
        "    const identity = JSON.parse(identityBytes);\n"
        "    const childConfig = fs.readFileSync("
        "`${identity.run_dir}/.usd-cli/config.toml`, 'utf8');\n"
        "    const child = spawnSync(identity.usd_cli_wrapper, ["
        "'--json', '--server', `http://${identity.server_host}:${identity.server_port}`, "
        "'--session', identity.parent_session_id, 'info'], "
        "{cwd: identity.run_dir, env: this.options.env, encoding: 'utf8'});\n"
        "    if (child.status !== 0) throw new Error(child.stderr || child.stdout);\n"
        "    const camera = spawnSync(identity.usd_cli_wrapper, ["
        "'--json', '--server', `http://${identity.server_host}:${identity.server_port}`, "
        "'--session', identity.parent_session_id, 'camera', 'fit', '/'], "
        "{cwd: identity.run_dir, env: this.options.env, encoding: 'utf8'});\n"
        "    if (camera.status !== 0) throw new Error(camera.stderr || camera.stdout);\n"
        "    const rejectedReload = spawnSync(identity.usd_cli_wrapper, ["
        "'--json', '--server', `http://${identity.server_host}:${identity.server_port}`, "
        "'--session', identity.parent_session_id, 'open', "
        "identity.source.staged_source.path], "
        "{cwd: identity.run_dir, env: this.options.env, encoding: 'utf8'});\n"
        "    if (rejectedReload.status === 0) "
        "throw new Error('plain reload discarded shared-session edits');\n"
        "    const forcedReload = spawnSync(identity.usd_cli_wrapper, ["
        "'--json', '--server', `http://${identity.server_host}:${identity.server_port}`, "
        "'--session', identity.parent_session_id, 'open', "
        "identity.source.staged_source.path, '--force-reload'], "
        "{cwd: identity.run_dir, env: this.options.env, encoding: 'utf8'});\n"
        "    if (forcedReload.status !== 0) "
        "throw new Error(forcedReload.stderr || forcedReload.stdout);\n"
        "    const lifecycle = spawnSync(identity.usd_cli_wrapper, ["
        "'--server', `http://${identity.server_host}:${identity.server_port}`, "
        "'--session', identity.parent_session_id, 'server', 'stop'], "
        "{cwd: identity.run_dir, env: this.options.env, encoding: 'utf8'});\n"
        "    if (lifecycle.status === 0) throw new Error('child stopped parent daemon');\n"
        "    const directEnv = {...this.options.env};\n"
        "    delete directEnv.CONTENT_WORKFLOW_PARENT_USD_CLI_MANAGED;\n"
        "    delete directEnv.USD_CLI_LIFECYCLE_EXTERNALLY_OWNED;\n"
        "    const directLifecycle = spawnSync(identity.usd_cli_executable, ["
        "'--server', `http://${identity.server_host}:${identity.server_port}`, "
        "'--session', identity.parent_session_id, 'server', 'stop'], "
        "{cwd: identity.run_dir, env: directEnv, encoding: 'utf8'});\n"
        "    if (directLifecycle.status === 0) "
        "throw new Error('child directly stopped parent daemon');\n"
        "    const observedEnvNames = ["
        "'CONTENT_WORKFLOW_PARENT_USD_CLI_SESSION_IDENTITY', "
        "'CONTENT_WORKFLOW_PARENT_USD_CLI_SESSION_IDENTITY_SHA256', "
        "'CONTENT_WORKFLOW_PARENT_USD_CLI_MANAGED', "
        "'USD_CLI_LIFECYCLE_EXTERNALLY_OWNED', 'USD_CLI_NO_DAEMON', "
        "'USD_CLI_SERVER_ALLOWED_WRITE_ROOTS'];\n"
        "    const observedEnv = Object.fromEntries(observedEnvNames.map("
        "name => [name, this.options.env[name]]));\n"
        "    const forbiddenEnvNames = ['OVRTX_API_KEY', "
        "'USD_CLI_RENDER_REMOTE_API_KEY', "
        "'USD_CLI_RENDER_BACKEND_API_KEYS_JSON', 'USD_CLI_TOKEN', "
        "'OV_SERVER_TOKEN'].filter(name => Object.hasOwn(this.options.env, name));\n"
        f"    fs.writeFileSync({json.dumps(str(capture_path))}, "
        "JSON.stringify({env: observedEnv, forbiddenEnvNames, "
        "childConfig, "
        "response: JSON.parse(child.stdout), lifecycleStatus: lifecycle.status, "
        "rejectedReloadStatus: rejectedReload.status, "
        "rejectedReloadError: rejectedReload.stderr || rejectedReload.stdout, "
        "forcedReloadStatus: forcedReload.status, "
        "lifecycleStderr: lifecycle.stderr, "
        "directLifecycleStatus: directLifecycle.status, "
        "directLifecycleStderr: directLifecycle.stderr}));\n"
        "    return {finalResponse: 'stopped', items: []};\n"
        "  }};\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    lifecycle: list[str] = []
    original_start = asset_runner._start_usd_cli_run_daemon_strict
    original_stop = asset_runner._stop_usd_cli_run_daemon_strict

    def start(**kwargs: Any) -> Any:
        lifecycle.append("daemon_start")
        lease = original_start(**kwargs)
        server_state = json.loads(lease.server_state_bytes)
        assert server_state["lifecycle_owner"] == "external"
        config_text = (run_dir / ".usd-cli" / "config.toml").read_text(encoding="utf-8")
        assert f'allowed_write_roots = ["{run_dir}"]' in config_text
        assert 'lifecycle_owner = "external"' in config_text
        assert "parent-project-secret" in config_text
        return lease

    def readiness(*_args: Any, **kwargs: Any) -> Any:
        lifecycle.append("authorized_render_negotiation")
        config_text = (run_dir / ".usd-cli" / "config.toml").read_text(encoding="utf-8")
        assert "parent-project-secret" in config_text
        artifact_path = run_dir / "raw" / f"{kwargs['artifact_stem']}.json"
        artifact_path.write_text('{"ready":true}\n', encoding="utf-8")
        return asset_runner.UsdCliReadiness(
            version="usd-cli test (authenticated parent session)",
            source_revision="d" * 40,
            probe={"ready": True},
            artifact_path=artifact_path,
        )

    def stop(**kwargs: Any) -> Any:
        lifecycle.append("daemon_stop")
        return original_stop(**kwargs)

    monkeypatch.setattr(asset_runner, "_start_usd_cli_run_daemon_strict", start)
    monkeypatch.setattr(asset_runner, "ensure_usd_cli_ovrtx_ready", readiness)
    monkeypatch.setattr(asset_runner, "_stop_usd_cli_run_daemon_strict", stop)
    renderer_secrets = {
        "OVRTX_API_KEY": "ovrtx-must-not-cross",
        "USD_CLI_RENDER_REMOTE_API_KEY": "remote-must-not-cross",
        "USD_CLI_RENDER_BACKEND_API_KEYS_JSON": '{"endpoint":"pool-secret"}',
        "USD_CLI_TOKEN": "unrelated-daemon-token",
        "OV_SERVER_TOKEN": "legacy-unrelated-daemon-token",
    }
    for name, value in renderer_secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(
        "USD_CLI_SERVER_ALLOWED_WRITE_ROOTS",
        str(tmp_path / "ambient-write-root-must-not-cross"),
    )

    exit_code = main(
        [
            "asset",
            "run",
            "--usd",
            str(paths["source"]),
            "--prompt",
            "Compose and validate the asset.",
            "--compatibility-fixed-order",
            "--joint-config",
            str(paths["joint"]),
            "--materials-yaml",
            str(paths["materials"]),
            "--repo-root",
            str(paths["repo"]),
            "--output-dir",
            str(run_dir),
            "--runner",
            "codex",
            "--child-timeout",
            "120",
        ]
    )

    assert exit_code == 1
    assert lifecycle == [
        "daemon_start",
        "authorized_render_negotiation",
        "daemon_stop",
    ]
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    child_env = capture["env"]
    assert child_env["CONTENT_WORKFLOW_PARENT_USD_CLI_SESSION_IDENTITY"]
    assert child_env["CONTENT_WORKFLOW_PARENT_USD_CLI_SESSION_IDENTITY_SHA256"]
    assert child_env["CONTENT_WORKFLOW_PARENT_USD_CLI_MANAGED"] == "1"
    assert child_env["USD_CLI_LIFECYCLE_EXTERNALLY_OWNED"] == "1"
    assert child_env["USD_CLI_NO_DAEMON"] == "1"
    assert child_env["USD_CLI_SERVER_ALLOWED_WRITE_ROOTS"] == str(run_dir)
    assert capture["forbiddenEnvNames"] == []
    assert "parent-project-secret" not in capture["childConfig"]
    assert 'lifecycle_owner = "external"' in capture["childConfig"]
    assert capture["response"]["ok"] is True
    assert capture["rejectedReloadStatus"] != 0
    assert "force-reload" in capture["rejectedReloadError"]
    assert capture["forcedReloadStatus"] == 0
    assert capture["lifecycleStatus"] == 2
    assert "externally owned usd-cli sessions forbid" in capture["lifecycleStderr"]
    assert capture["directLifecycleStatus"] != 0
    assert (
        "externally owned usd-cli sessions forbid" in capture["directLifecycleStderr"]
    )
    bridge_request = json.loads(
        (run_dir / "raw" / "asset_composition_request.json").read_text(encoding="utf-8")
    )
    assert bridge_request["workflow"] == "asset.run"
    assert bridge_request["sandbox_writable_roots"] == []
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert Path(request["source_asset"]).is_relative_to(run_dir)
    assert request["source_staging"]["manifest"]["path"].startswith(str(run_dir))
    identity_path = Path(child_env["CONTENT_WORKFLOW_PARENT_USD_CLI_SESSION_IDENTITY"])
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    assert identity["allowed_roots"] == [str(run_dir)]
    assert identity["renderer_credentials"] == "parent_confined"
    assert identity["lifecycle_authority"] == "parent_only"
    receipt = _single_teardown_receipt(run_dir)
    assert receipt["status"] == "released"
    assert receipt["descendants_released"] is True
    assert receipt["state_directory_released"] is True
    assert not (run_dir / ".usd-cli").exists()
    intents = list((run_dir / "raw").glob("asset_usd_cli_launch_*.json"))
    assert len(intents) == 1
    assert (
        json.loads(intents[0].read_text(encoding="utf-8"))["launch_id"]
        == receipt["launch_id"]
    )
    run_artifact_bytes = b"\n".join(
        path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file() and not path.is_relative_to(run_dir / "inputs")
    )
    for name, value in renderer_secrets.items():
        assert name.encode() not in run_artifact_bytes
        assert value.encode() not in run_artifact_bytes


def test_asset_child_workflow_identity_rejects_unknown_config(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)

    assert runner._child_workflow_name(_config(paths, tmp_path / "run")) == "asset.run"
    with pytest.raises(TypeError, match="Unsupported child workflow config: object"):
        runner._child_workflow_name(cast(Any, object()))


def test_asset_child_launch_rejects_missing_parent_session_identity(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="asset child launch requires a parent usd-cli session identity",
    ):
        runner._inject_parent_usd_cli_session_environment(
            cast(Any, runner.AssetCompositionChildConfig()),
            run_dir=tmp_path,
            env={},
        )


def test_prompt_reference_drift_fails_resume_closed(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    config = replace(_config(paths, run_dir), reference_images=[])
    run_asset_workflow(config)
    reference = run_dir / "inputs" / "prompt-reference.md"
    reference.write_text("changed\n", encoding="utf-8")

    with pytest.raises(AssetCompositionStateError, match="identity changed"):
        resume_asset_workflow(run_dir, dry_run=True)


def test_zero_exit_child_is_failed_when_top_level_run_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    def child_without_completion(**_kwargs: object) -> int:
        return 0

    monkeypatch.setattr(asset_runner, "_run_child_agent", child_without_completion)

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.returncode == 1
    assert not result.completed
    run = load_verified_run(run_dir / "asset_run.json")
    assert run.terminal_status == "failed"
    assert run.stages["articulation"].status == "failed"


def test_unbounded_asset_child_keeps_bounded_daemon_idle_backstop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    original_start = asset_runner._start_usd_cli_run_daemon_strict
    observed_idle_timeouts: list[int] = []

    def start(**kwargs: Any) -> Any:
        observed_idle_timeouts.append(kwargs["idle_timeout_seconds"])
        return original_start(**kwargs)

    monkeypatch.setattr(asset_runner, "_start_usd_cli_run_daemon_strict", start)
    monkeypatch.setattr(asset_runner, "_run_child_agent", lambda **_kwargs: 0)

    result = run_asset_workflow(
        replace(
            _config(paths, run_dir),
            child_timeout_seconds=0,
            dry_run=False,
        )
    )

    assert result.returncode == 1
    assert observed_idle_timeouts == [runner.USD_CLI_ASSET_MAX_IDLE_TIMEOUT_SECONDS]


def _single_teardown_receipt(run_dir: Path) -> dict[str, Any]:
    paths = list((run_dir / "raw").glob("asset_usd_cli_teardown_*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_asset_parent_session_releases_at_human_review_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    def stop_for_review(**kwargs: object) -> int:
        state_path = Path(kwargs["run_dir"]) / "asset_run.json"
        _plan_and_begin(state_path, "articulation")
        candidates_dir = stage_directory(state_path, "articulation")
        candidates_dir.mkdir(parents=True, exist_ok=True)
        candidates = candidates_dir / "candidates.json"
        candidates.write_text('{"candidate_ids":["door"]}\n', encoding="utf-8")
        _record_await_review(state_path, candidates)
        require_review(state_path, candidates_path=candidates)
        return 0

    monkeypatch.setattr(asset_runner, "_run_child_agent", stop_for_review)

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.returncode == 3
    assert result.needs_review
    receipt = _single_teardown_receipt(run_dir)
    assert receipt["status"] == "released"
    assert receipt["boundary"] == "human_review"
    assert receipt["process_released"] is True
    assert receipt["sessions_released"] is True
    assert receipt["listener_released"] is True
    assert receipt["daemon_leases_released"] is True
    assert receipt["source_integrity_verified"] is True
    journal = receipt["command_receipt_journal"]
    checkpoint = receipt["command_receipt_checkpoint"]
    assert Path(journal["path"]) == (run_dir / "raw" / "usd_cli_command_receipts.jsonl")
    assert Path(checkpoint["path"]) == (
        run_dir / "raw" / "usd_cli_command_receipts.checkpoint.json"
    )
    assert journal["sha256"] == file_sha256(Path(journal["path"]))
    assert checkpoint["sha256"] == file_sha256(Path(checkpoint["path"]))


def test_asset_released_history_accepts_repeated_unchanged_command_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    def stop_for_review(**kwargs: object) -> int:
        state_path = Path(kwargs["run_dir"]) / "asset_run.json"
        _plan_and_begin(state_path, "articulation")
        candidates_dir = stage_directory(state_path, "articulation")
        candidates_dir.mkdir(parents=True, exist_ok=True)
        candidates = candidates_dir / "candidates.json"
        candidates.write_text('{"candidate_ids":["door"]}\n', encoding="utf-8")
        _record_await_review(state_path, candidates)
        require_review(state_path, candidates_path=candidates)
        return 0

    monkeypatch.setattr(asset_runner, "_run_child_agent", stop_for_review)
    first = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))
    assert first.needs_review

    def interrupt_before_first_session_command(
        _repo_root: Path,
        **_kwargs: object,
    ) -> asset_runner.UsdCliReadiness:
        raise KeyboardInterrupt

    decisions = tmp_path / "decisions.json"
    decisions.write_text('{"door":"accept"}\n', encoding="utf-8")
    monkeypatch.setattr(
        asset_runner,
        "ensure_usd_cli_ovrtx_ready",
        interrupt_before_first_session_command,
    )
    second = review_asset_workflow(
        run_dir,
        decisions_path=decisions,
        reviewer="asset-owner",
        dry_run=False,
    )

    assert second.returncode == 130
    receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "raw").glob("asset_usd_cli_teardown_*.json"))
    ]
    assert len(receipts) == 2
    assert all(receipt["status"] == "released" for receipt in receipts)
    assert (
        receipts[0]["command_receipt_journal"] == receipts[1]["command_receipt_journal"]
    )
    assert (
        receipts[0]["command_receipt_checkpoint"]
        == receipts[1]["command_receipt_checkpoint"]
    )
    asset_runner._require_safe_asset_usd_cli_teardown_history(
        run_dir,
        run_id="cabinet-composed",
        terminal_valid=False,
    )


def test_asset_parent_session_releases_after_child_failure_and_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for label, failure, expected_boundary in (
        ("failure", RuntimeError("child failed"), "failed"),
        (
            "cancellation",
            runner.ChildProcessInterrupted(15, "asset child"),
            "cancelled",
        ),
    ):
        paths = _inputs(tmp_path / label)
        run_dir = tmp_path / f"run-{label}"

        def fail_child(**_kwargs: object) -> int:
            raise failure

        monkeypatch.setattr(asset_runner, "_run_child_agent", fail_child)
        result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

        assert result.returncode != 0
        receipt = _single_teardown_receipt(run_dir)
        assert receipt["status"] == "released"
        assert receipt["boundary"] == expected_boundary


@pytest.mark.parametrize(
    ("tamper_target", "error_fragment"),
    [
        ("identity", "parent session attestation"),
        ("readiness", "parent session attestation"),
        ("launch_intent", "asset launch intent integrity"),
    ],
)
def test_asset_parent_session_attestation_fails_closed_after_child_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_target: str,
    error_fragment: str,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    def tamper_with_parent_identity(**kwargs: object) -> int:
        config = cast(AssetRunConfig, kwargs["config"])
        assert config.parent_usd_cli_session_identity is not None
        identity_path = config.parent_usd_cli_session_identity
        if tamper_target == "identity":
            identity_path.write_text("{}\n", encoding="utf-8")
        elif tamper_target == "readiness":
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            Path(identity["readiness_artifact"]["path"]).write_text(
                '{"ready":false}\n',
                encoding="utf-8",
            )
        else:
            launch_intent = next((run_dir / "raw").glob("asset_usd_cli_launch_*.json"))
            launch_intent.write_text('{"tampered":true}\n', encoding="utf-8")
        return 0

    monkeypatch.setattr(
        asset_runner,
        "_run_child_agent",
        tamper_with_parent_identity,
    )

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.returncode == 2
    receipt = _single_teardown_receipt(run_dir)
    assert receipt["status"] == "failed"
    assert error_fragment in " ".join(receipt["errors"])


@pytest.mark.parametrize("tamper_mode", ["remove", "add"])
def test_asset_review_rejects_prior_lifecycle_history_tampering_after_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_mode: str,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    def stop_for_review(**kwargs: object) -> int:
        state_path = Path(kwargs["run_dir"]) / "asset_run.json"
        _plan_and_begin(state_path, "articulation")
        candidates_dir = stage_directory(state_path, "articulation")
        candidates_dir.mkdir(parents=True, exist_ok=True)
        candidates = candidates_dir / "candidates.json"
        candidates.write_text('{"candidate_ids":["door"]}\n', encoding="utf-8")
        _record_await_review(state_path, candidates)
        require_review(state_path, candidates_path=candidates)
        return 0

    monkeypatch.setattr(asset_runner, "_run_child_agent", stop_for_review)
    first = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))
    assert first.needs_review
    prior_lifecycle_paths = tuple(
        sorted((run_dir / "raw").glob("asset_usd_cli_launch_*.json"))
        + sorted((run_dir / "raw").glob("asset_usd_cli_teardown_*.json"))
    )
    assert len(prior_lifecycle_paths) == 2

    decisions = tmp_path / "decisions.json"
    decisions.write_text('{"door":"accept"}\n', encoding="utf-8")

    def tamper_with_prior_history(**_kwargs: object) -> int:
        if tamper_mode == "remove":
            for path in prior_lifecycle_paths:
                path.unlink()
        else:
            (run_dir / "raw" / "asset_usd_cli_launch_forged-by-child.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
        return 0

    monkeypatch.setattr(asset_runner, "_run_child_agent", tamper_with_prior_history)
    result = review_asset_workflow(
        run_dir,
        decisions_path=decisions,
        reviewer="asset-owner",
        dry_run=False,
    )

    assert result.returncode == 2
    receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_dir / "raw").glob("asset_usd_cli_teardown_*.json")
    ]
    failed_receipts = [receipt for receipt in receipts if receipt["status"] == "failed"]
    assert len(failed_receipts) == 1
    receipt = failed_receipts[0]
    assert receipt["status"] == "failed"
    assert "asset lifecycle history integrity" in " ".join(receipt["errors"])


@pytest.mark.parametrize(
    "receipt_field",
    ["session_identity_path", "daemon_log_path"],
)
@pytest.mark.parametrize("tamper_mode", ["remove", "replace"])
def test_asset_review_rejects_prior_referenced_lifecycle_artifact_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_field: str,
    tamper_mode: str,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    def stop_for_review(**kwargs: object) -> int:
        state_path = Path(kwargs["run_dir"]) / "asset_run.json"
        _plan_and_begin(state_path, "articulation")
        candidates_dir = stage_directory(state_path, "articulation")
        candidates_dir.mkdir(parents=True, exist_ok=True)
        candidates = candidates_dir / "candidates.json"
        candidates.write_text('{"candidate_ids":["door"]}\n', encoding="utf-8")
        _record_await_review(state_path, candidates)
        require_review(state_path, candidates_path=candidates)
        return 0

    monkeypatch.setattr(asset_runner, "_run_child_agent", stop_for_review)
    first = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))
    assert first.needs_review
    prior_receipt = _single_teardown_receipt(run_dir)
    if receipt_field == "daemon_log_path" and prior_receipt[receipt_field] is None:
        daemon_log = run_dir / "raw" / "usd_cli_daemon_0000000000000000.log"
        daemon_log.write_text("preserved daemon output\n", encoding="utf-8")
        prior_receipt["daemon_log_path"] = str(daemon_log)
        prior_receipt["daemon_log_sha256"] = file_sha256(daemon_log)
        receipt_path = next((run_dir / "raw").glob("asset_usd_cli_teardown_*.json"))
        receipt_path.write_text(
            json.dumps(prior_receipt, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    artifact_path = Path(prior_receipt[receipt_field])
    assert artifact_path.is_file()

    decisions = tmp_path / "decisions.json"
    decisions.write_text('{"door":"accept"}\n', encoding="utf-8")
    child_called = False

    def tamper_with_prior_artifact(**_kwargs: object) -> int:
        nonlocal child_called
        child_called = True
        if tamper_mode == "remove":
            artifact_path.unlink()
        else:
            artifact_path.write_text("tampered\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(asset_runner, "_run_child_agent", tamper_with_prior_artifact)
    result = review_asset_workflow(
        run_dir,
        decisions_path=decisions,
        reviewer="asset-owner",
        dry_run=False,
    )

    assert child_called
    assert result.returncode == 2
    receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_dir / "raw").glob("asset_usd_cli_teardown_*.json")
    ]
    failed_receipts = [receipt for receipt in receipts if receipt["status"] == "failed"]
    assert len(failed_receipts) == 1
    assert "asset lifecycle history integrity" in " ".join(failed_receipts[0]["errors"])


@pytest.mark.parametrize(
    "receipt_field",
    ["command_receipt_journal", "command_receipt_checkpoint"],
)
@pytest.mark.parametrize("tamper_mode", ["remove", "replace"])
def test_asset_review_rejects_prior_command_receipt_seal_tampering_before_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_field: str,
    tamper_mode: str,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    def stop_for_review(**kwargs: object) -> int:
        state_path = Path(kwargs["run_dir"]) / "asset_run.json"
        _plan_and_begin(state_path, "articulation")
        candidates_dir = stage_directory(state_path, "articulation")
        candidates_dir.mkdir(parents=True, exist_ok=True)
        candidates = candidates_dir / "candidates.json"
        candidates.write_text('{"candidate_ids":["door"]}\n', encoding="utf-8")
        _record_await_review(state_path, candidates)
        require_review(state_path, candidates_path=candidates)
        return 0

    monkeypatch.setattr(asset_runner, "_run_child_agent", stop_for_review)
    first = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))
    assert first.needs_review
    prior_receipt = _single_teardown_receipt(run_dir)
    artifact_path = Path(prior_receipt[receipt_field]["path"])
    if tamper_mode == "remove":
        artifact_path.unlink()
    else:
        artifact_path.write_text("tampered\n", encoding="utf-8")

    decisions = tmp_path / "decisions.json"
    decisions.write_text('{"door":"accept"}\n', encoding="utf-8")
    child_called = False

    def child(**_kwargs: object) -> int:
        nonlocal child_called
        child_called = True
        return 0

    monkeypatch.setattr(asset_runner, "_run_child_agent", child)
    with pytest.raises(AssetCompositionStateError, match="command receipt"):
        review_asset_workflow(
            run_dir,
            decisions_path=decisions,
            reviewer="asset-owner",
            dry_run=False,
        )

    assert child_called is False
    run = load_verified_run(run_dir / "asset_run.json")
    assert run.stages["articulation"].status == "needs_review"
    assert run.stages["articulation"].review_decisions is None


def test_asset_parent_session_receipt_journal_integrity_is_a_release_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    session = SimpleNamespace(
        session_id="workflow-asset-test",
        open=lambda _source: None,
        verify_receipt_journal_integrity=lambda: (_ for _ in ()).throw(
            RuntimeError("usd-cli receipt journal was replaced or modified")
        ),
    )
    monkeypatch.setattr(
        asset_runner,
        "_attach_workflow_usd_cli_session",
        lambda **_kwargs: session,
    )
    monkeypatch.setattr(asset_runner, "_run_child_agent", lambda **_kwargs: 0)

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.returncode == 2
    receipt = _single_teardown_receipt(run_dir)
    assert receipt["status"] == "failed"
    assert "parent usd-cli receipt journal integrity" in " ".join(receipt["errors"])


def test_asset_parent_session_finishes_teardown_after_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    original_stop = asset_runner._stop_usd_cli_run_daemon_strict
    calls = 0

    def interrupted_once(**kwargs: object) -> runner.UsdCliDaemonTeardownEvidence:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return original_stop(**kwargs)

    monkeypatch.setattr(
        asset_runner,
        "_stop_usd_cli_run_daemon_strict",
        interrupted_once,
    )
    monkeypatch.setattr(asset_runner, "_run_child_agent", lambda **_kwargs: 0)

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.returncode == 130
    assert calls == 2
    receipt = _single_teardown_receipt(run_dir)
    assert receipt["status"] == "released"
    assert receipt["boundary"] == "cancelled"
    assert receipt["interrupted"] is True
    assert receipt["state_directory_released"] is True


def test_asset_parent_session_releases_before_terminal_state_record_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    original_stop = asset_runner._stop_usd_cli_run_daemon_strict
    daemon_released = False

    def stop(**kwargs: object) -> runner.UsdCliDaemonTeardownEvidence:
        nonlocal daemon_released
        evidence = original_stop(**kwargs)
        daemon_released = True
        return evidence

    def interrupt_state_record(*_args: object, **_kwargs: object) -> None:
        assert daemon_released is True
        raise KeyboardInterrupt

    monkeypatch.setattr(asset_runner, "_stop_usd_cli_run_daemon_strict", stop)
    monkeypatch.setattr(
        asset_runner,
        "_record_unfinished_child_exit",
        interrupt_state_record,
    )
    monkeypatch.setattr(asset_runner, "_run_child_agent", lambda **_kwargs: 0)

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.returncode == 2
    assert daemon_released is True
    receipt = _single_teardown_receipt(run_dir)
    assert receipt["status"] == "failed"
    assert receipt["boundary"] == "cancelled"
    assert receipt["interrupted"] is True
    assert receipt["process_released"] is True
    assert "asset child terminal-state recording" in " ".join(receipt["errors"])


def test_asset_interruption_dominates_setup_failure_boundary(tmp_path: Path) -> None:
    assert (
        asset_runner._asset_execution_boundary(
            tmp_path / "unpublished-run-state.json",
            interrupted=True,
            setup_failed=True,
        )
        == "cancelled"
    )


def test_asset_review_rejects_launch_without_teardown_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    def stop_for_review(**kwargs: object) -> int:
        state_path = Path(kwargs["run_dir"]) / "asset_run.json"
        _plan_and_begin(state_path, "articulation")
        candidates_dir = stage_directory(state_path, "articulation")
        candidates_dir.mkdir(parents=True, exist_ok=True)
        candidates = candidates_dir / "candidates.json"
        candidates.write_text('{"candidate_ids":["door"]}\n', encoding="utf-8")
        _record_await_review(state_path, candidates)
        require_review(state_path, candidates_path=candidates)
        return 0

    monkeypatch.setattr(asset_runner, "_run_child_agent", stop_for_review)
    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))
    assert result.needs_review
    teardown_path = next((run_dir / "raw").glob("asset_usd_cli_teardown_*.json"))
    teardown_path.unlink()
    decisions = tmp_path / "decisions.json"
    decisions.write_text('{"door":"accept"}\n', encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="launch has no teardown receipt",
    ):
        review_asset_workflow(
            run_dir,
            decisions_path=decisions,
            reviewer="asset-owner",
            dry_run=True,
        )

    run = load_verified_run(run_dir / "asset_run.json")
    assert run.stages["articulation"].status == "needs_review"
    assert run.stages["articulation"].review_decisions is None


def test_asset_review_reserves_lifecycle_slot_before_recording_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    def stop_for_review(**kwargs: object) -> int:
        state_path = Path(kwargs["run_dir"]) / "asset_run.json"
        _plan_and_begin(state_path, "articulation")
        candidates_dir = stage_directory(state_path, "articulation")
        candidates_dir.mkdir(parents=True, exist_ok=True)
        candidates = candidates_dir / "candidates.json"
        candidates.write_text('{"candidate_ids":["door"]}\n', encoding="utf-8")
        _record_await_review(state_path, candidates)
        require_review(state_path, candidates_path=candidates)
        return 0

    monkeypatch.setattr(asset_runner, "_run_child_agent", stop_for_review)
    first = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))
    assert first.needs_review
    monkeypatch.setattr(asset_runner, "MAX_ASSET_USD_CLI_LIFECYCLE_RECORDS", 1)
    decisions = tmp_path / "decisions.json"
    decisions.write_text('{"door":"accept"}\n', encoding="utf-8")

    with pytest.raises(AssetCompositionStateError, match="no bounded slot"):
        review_asset_workflow(
            run_dir,
            decisions_path=decisions,
            reviewer="asset-owner",
            dry_run=False,
        )

    run = load_verified_run(run_dir / "asset_run.json")
    assert run.stages["articulation"].status == "needs_review"
    assert run.stages["articulation"].review_decisions is None
    assert len(list((run_dir / "raw").glob("asset_usd_cli_launch_*.json"))) == 1
    assert len(list((run_dir / "raw").glob("asset_usd_cli_teardown_*.json"))) == 1


def test_asset_resume_rejects_foreign_run_lifecycle_receipts(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_asset_workflow(_config(paths, run_dir))
    launch_id = "foreign-launch"
    (run_dir / "raw" / f"asset_usd_cli_launch_{launch_id}.json").write_text(
        json.dumps(
            {
                "schema_version": "content-workflow-cli.asset-usd-cli-launch.v1",
                "created_at": "2026-08-16T00:00:00Z",
                "launch_id": launch_id,
                "run_id": "foreign-run",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "raw" / f"asset_usd_cli_teardown_{launch_id}.json").write_text(
        json.dumps(
            {
                "schema_version": "content-workflow-cli.asset-usd-cli-teardown.v1",
                "created_at": "2026-08-16T00:00:01Z",
                "launch_id": launch_id,
                "run_id": "foreign-run",
                "daemon_was_started": False,
                "status": "not_started",
                "boundary": "setup_failed",
                "child_returncode": 2,
                "interrupted": False,
                "source_integrity_verified": True,
                "setup_error": "foreign setup failure",
                "errors": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="launch intent belongs to another run",
    ):
        resume_asset_workflow(run_dir, dry_run=True)


def test_asset_setup_failure_is_receipted_inside_the_coordinator_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    child_called = False

    def fail_route(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("package-owned usd-cli route unavailable")

    def child(**_kwargs: object) -> int:
        nonlocal child_called
        child_called = True
        return 0

    monkeypatch.setattr(asset_runner, "_prepare_usd_cli_telemetry_route", fail_route)
    monkeypatch.setattr(asset_runner, "_run_child_agent", child)

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.returncode == 2
    assert child_called is False
    receipt = _single_teardown_receipt(run_dir)
    assert receipt["status"] == "not_started"
    assert receipt["boundary"] == "setup_failed"
    assert receipt["daemon_was_started"] is False
    assert "package-owned usd-cli route unavailable" in receipt["setup_error"]
    assert receipt["errors"] == []


def test_asset_readiness_failure_before_first_session_command_releases_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    child_called = False

    def attach(**kwargs: object) -> asset_runner.WorkflowUsdCliSession:
        root = Path(str(kwargs["run_dir"]))
        return asset_runner.WorkflowUsdCliSession(
            project_dir=root,
            session_id="workflow-asset-early-failure",
            route=SimpleNamespace(
                wrapper=paths["repo"] / "usd-cli-tel",
                target=paths["repo"] / "usd-cli",
            ),
            workflow="asset.run",
            daemon_project_dir=root,
            daemon_server_url="http://127.0.0.1:4567",
        )

    def fail_readiness(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("usd-cli version readiness failed")

    def child(**_kwargs: object) -> int:
        nonlocal child_called
        child_called = True
        return 0

    monkeypatch.setattr(asset_runner, "_attach_workflow_usd_cli_session", attach)
    monkeypatch.setattr(asset_runner, "ensure_usd_cli_ovrtx_ready", fail_readiness)
    monkeypatch.setattr(asset_runner, "_run_child_agent", child)

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.returncode == 2
    assert child_called is False
    receipt = _single_teardown_receipt(run_dir)
    assert receipt["status"] == "released"
    assert receipt["boundary"] == "setup_failed"
    assert "usd-cli version readiness failed" in receipt["setup_error"]
    assert receipt["errors"] == []


def test_asset_receipt_construction_failure_writes_failed_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    def exit_after_launch(*_args: object, **_kwargs: object) -> object:
        raise SystemExit(9)

    monkeypatch.setattr(
        asset_runner,
        "_prepare_usd_cli_telemetry_route",
        exit_after_launch,
    )

    with pytest.raises(SystemExit, match="9"):
        run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    receipt = _single_teardown_receipt(run_dir)
    assert receipt["status"] == "failed"
    assert receipt["boundary"] == "setup_failed"
    assert "teardown receipt ValidationError" in " ".join(receipt["errors"])


def test_asset_interrupted_strict_start_preserves_release_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    identity = runner.UsdCliDaemonIdentity(
        pid=4242,
        process_start_token="test-start-token",
        project_id="a" * 24,
        instance_id="asset-test-instance",
        process_group_id=4242,
        os_session_id=4242,
    )
    release = runner.UsdCliDaemonTeardownEvidence(
        identity=identity,
        host="127.0.0.1",
        port=4567,
        daemon_was_started=True,
        process_released=True,
        descendants_released=True,
        sessions_released=True,
        listener_released=True,
        daemon_leases_released=True,
        state_directory_released=True,
    )

    def interrupt_start(**_kwargs: object) -> runner.UsdCliDaemonLease:
        raise runner.UsdCliDaemonStrictStartError(
            "interrupted after daemon registration",
            teardown_evidence=release,
            interrupted=True,
        )

    monkeypatch.setattr(
        asset_runner,
        "_start_usd_cli_run_daemon_strict",
        interrupt_start,
    )

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.returncode == 130
    receipt = _single_teardown_receipt(run_dir)
    assert receipt["status"] == "released"
    assert receipt["daemon_was_started"] is True
    assert receipt["process_released"] is True
    assert receipt["interrupted"] is True


def test_asset_retries_interrupt_during_strict_start_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    identity = runner.UsdCliDaemonIdentity(
        pid=4242,
        process_start_token="test-start-token",
        project_id="a" * 24,
        instance_id="asset-test-instance",
        process_group_id=4242,
        os_session_id=4242,
    )
    cleanup = runner.UsdCliDaemonCleanupEvidence(
        target_path=paths["repo"] / "usd-cli",
        identities=(identity,),
        daemon_ledger_bytes=b"4242:test-start-token\n",
    )

    def interrupt_start(**_kwargs: object) -> runner.UsdCliDaemonLease:
        raise runner.UsdCliDaemonStrictStartError(
            "startup and first cleanup were interrupted",
            teardown_evidence=None,
            cleanup_lease=cleanup,
            interrupted=True,
        )

    monkeypatch.setattr(
        asset_runner,
        "_start_usd_cli_run_daemon_strict",
        interrupt_start,
    )

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.returncode == 130
    receipt = _single_teardown_receipt(run_dir)
    assert receipt["status"] == "released"
    assert receipt["daemon_was_started"] is True
    assert receipt["process_released"] is True
    assert receipt["interrupted"] is True


def test_asset_failed_strict_start_retry_preserves_daemon_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    identity = runner.UsdCliDaemonIdentity(
        pid=4242,
        process_start_token="test-start-token",
        project_id="a" * 24,
        instance_id="asset-test-instance",
        process_group_id=4242,
        os_session_id=4242,
    )
    cleanup = runner.UsdCliDaemonCleanupEvidence(
        target_path=paths["repo"] / "usd-cli",
        identities=(identity,),
        daemon_ledger_bytes=b"4242:test-start-token\n",
    )

    def interrupt_start(**_kwargs: object) -> runner.UsdCliDaemonLease:
        raise runner.UsdCliDaemonStrictStartError(
            "startup and first cleanup were interrupted",
            teardown_evidence=None,
            cleanup_lease=cleanup,
            interrupted=True,
        )

    monkeypatch.setattr(
        asset_runner,
        "_start_usd_cli_run_daemon_strict",
        interrupt_start,
    )
    monkeypatch.setattr(
        asset_runner,
        "_stop_usd_cli_run_daemon_strict",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("captured daemon release remains unproven")
        ),
    )

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.returncode == 2
    receipt = _single_teardown_receipt(run_dir)
    assert receipt["status"] == "failed"
    assert receipt["daemon_was_started"] is True
    assert receipt["daemon_identity_sha256"] == asset_runner._daemon_identity_sha256(
        identity
    )
    assert "captured daemon release remains unproven" in " ".join(receipt["errors"])
    assert "teardown receipt ValidationError" not in " ".join(receipt["errors"])


def test_asset_teardown_failure_overrides_child_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(asset_runner, "_run_child_agent", lambda **_kwargs: 0)
    monkeypatch.setattr(
        asset_runner,
        "_stop_usd_cli_run_daemon_strict",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("captured listener remains occupied")
        ),
    )

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.returncode == 2
    assert not result.needs_review
    receipt = _single_teardown_receipt(run_dir)
    assert receipt["status"] == "failed"
    assert "listener remains occupied" in " ".join(receipt["errors"])

    shutil.rmtree(run_dir / ".usd-cli", ignore_errors=True)
    with pytest.raises(
        AssetCompositionStateError,
        match="prior asset usd-cli teardown failed",
    ):
        resume_asset_workflow(
            run_dir,
            recovery_reason="retry after failed teardown",
            dry_run=True,
        )


def test_asset_runner_reports_terminal_state_integrity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(asset_runner, "_run_child_agent", lambda **_kwargs: 0)

    def fail_load(_path: Path) -> None:
        raise AssetCompositionStateError("terminal state identity changed")

    monkeypatch.setattr(asset_runner, "load_verified_run", fail_load)

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.returncode == 2
    assert not result.completed
    assert not result.needs_review
    assert "terminal state identity changed" in result.child_output_path.read_text(
        encoding="utf-8"
    )


def test_validated_asset_helpers_reject_missing_material_library(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    config = replace(_config(paths, run_dir), materials_usd=None)

    with pytest.raises(ValueError, match="missing the materials library"):
        asset_runner._build_request(
            config,
            run_id="missing-library",
            run_dir=run_dir,
            run_state_path=run_dir / "asset_run.json",
        )


def test_asset_resume_migrates_pre_coordinator_run_before_launching_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_asset_workflow(_config(paths, run_dir))
    state_path = run_dir / "asset_run.json"
    legacy_state = json.loads(state_path.read_text(encoding="utf-8"))
    legacy_state.pop("coordinator")
    request_path = run_dir / "request.json"
    legacy_request = json.loads(request_path.read_text(encoding="utf-8"))
    legacy_request.pop("coordinator_mode")
    legacy_request.pop("physics_validation_mode")
    request_path.write_text(json.dumps(legacy_request) + "\n", encoding="utf-8")
    legacy_state["request"] = {
        "path": str(request_path),
        "sha256": file_sha256(request_path),
        "size_bytes": request_path.stat().st_size,
    }
    state_path.write_text(json.dumps(legacy_state) + "\n", encoding="utf-8")
    observed_modes: list[str] = []

    def observe_migration(**_kwargs: object) -> int:
        observed_modes.append(load_verified_run(state_path).coordinator.mode)
        return 7

    monkeypatch.setattr(asset_runner, "_run_child_agent", observe_migration)

    result = resume_asset_workflow(run_dir, dry_run=False)

    assert result.returncode == 7
    assert observed_modes == ["single_reasoning_loop"]


def test_asset_resume_rejects_legacy_unstaged_request_before_launcher_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_asset_workflow(_config(paths, run_dir))
    state_path = run_dir / "asset_run.json"
    request_path = run_dir / "request.json"
    legacy_request = json.loads(request_path.read_text(encoding="utf-8"))
    legacy_request["schema_version"] = "content-agents.asset-composition-request.v1"
    legacy_request.pop("source_staging")
    request_path.write_text(
        json.dumps(legacy_request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["request"] = {
        "path": str(request_path),
        "sha256": file_sha256(request_path),
        "size_bytes": request_path.stat().st_size,
    }
    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    setup_called = False

    def launcher_setup(*_args: object, **_kwargs: object) -> object:
        nonlocal setup_called
        setup_called = True
        raise AssertionError("legacy request must fail before launcher setup")

    monkeypatch.setattr(
        asset_runner,
        "_prepare_usd_cli_telemetry_route",
        launcher_setup,
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="predates run-confined source staging",
    ):
        resume_asset_workflow(run_dir, dry_run=False)

    assert setup_called is False
    assert list((run_dir / "raw").glob("asset_usd_cli_teardown_*.json")) == []


def test_competing_asset_resume_does_not_fail_stage_owned_by_other_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_asset_workflow(_config(paths, run_dir))
    state_path = run_dir / "asset_run.json"
    _plan_and_begin(state_path, "articulation")
    prompt_path = run_dir / "agent_resume_prompt.md"
    prompt_path.write_text("owner prompt\n", encoding="utf-8")

    def reject_competing_loop(*_args: object, **_kwargs: object) -> None:
        raise AssetCoordinatorLeaseError(
            "Another asset coordinator reasoning loop already owns this run"
        )

    monkeypatch.setattr(
        asset_runner,
        "run_batch_asset_coordinator",
        reject_competing_loop,
    )

    result = resume_asset_workflow(run_dir, dry_run=False)

    run = load_verified_run(state_path)
    assert result.returncode == 2
    assert run.terminal_status == "active"
    assert run.stages["articulation"].status == "running"
    assert prompt_path.read_text(encoding="utf-8") == "owner prompt\n"
    assert (
        "Another asset coordinator reasoning loop already owns"
        in capsys.readouterr().err
    )


def test_asset_recovery_transition_respects_active_coordinator_lease(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_asset_workflow(_config(paths, run_dir))
    state_path = run_dir / "asset_run.json"
    _plan_and_begin(state_path, "articulation")
    fail_stage(
        state_path,
        "articulation",
        reason="simulated executor failure",
    )

    def attempt_recovery(_session: object) -> None:
        with pytest.raises(AssetCoordinatorLeaseError):
            resume_asset_workflow(
                run_dir,
                recovery_reason="retry while old owner remains active",
                dry_run=True,
            )

    asset_runner.run_batch_asset_coordinator(
        state_path,
        reasoning_loop=attempt_recovery,
    )

    run = load_verified_run(state_path)
    assert run.terminal_status == "failed"
    assert run.stages["articulation"].status == "failed"


def test_asset_review_transition_respects_active_coordinator_lease(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_asset_workflow(_config(paths, run_dir))
    state_path = run_dir / "asset_run.json"
    _plan_and_begin(state_path, "articulation")
    stage_dir = stage_directory(state_path, "articulation")
    candidates = stage_dir / "articulation_candidates.json"
    candidates.write_text('{"candidate_ids":["drawer"]}\n', encoding="utf-8")
    _record_await_review(state_path, candidates)
    require_review(state_path, candidates_path=candidates)
    decisions = run_dir / "joint-decisions.json"
    decisions.write_text('{"drawer":"accept"}\n', encoding="utf-8")

    def attempt_review(_session: object) -> None:
        with pytest.raises(AssetCoordinatorLeaseError):
            review_asset_workflow(
                run_dir,
                decisions_path=decisions,
                reviewer="asset-owner",
                dry_run=True,
            )

    asset_runner.run_batch_asset_coordinator(
        state_path,
        reasoning_loop=attempt_review,
    )

    run = load_verified_run(state_path)
    assert run.stages["articulation"].status == "needs_review"
    assert run.stages["articulation"].review_decisions is None


def test_verified_terminal_completion_overrides_late_child_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    class LateChildFailure:
        returncode = 124

    monkeypatch.setattr(
        asset_runner,
        "run_batch_asset_coordinator",
        lambda *_args, **_kwargs: LateChildFailure(),
    )
    monkeypatch.setattr(
        asset_runner,
        "validate_terminal",
        lambda _path: AssetTerminalValidation(
            valid=True,
            terminal_status="completed",
            current_stage=None,
        ),
    )

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.completed
    assert result.returncode == 0


def test_coordinator_integrity_failure_is_not_overridden_by_terminal_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    def reject_integrity(*_args: object, **_kwargs: object) -> None:
        raise AssetCompositionStateError("Frozen materials manifest identity changed")

    monkeypatch.setattr(
        asset_runner,
        "run_batch_asset_coordinator",
        reject_integrity,
    )
    monkeypatch.setattr(
        asset_runner,
        "validate_terminal",
        lambda _path: AssetTerminalValidation(
            valid=True,
            terminal_status="completed",
            current_stage=None,
        ),
    )

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert not result.completed
    assert result.returncode == 2
    assert "Frozen materials manifest identity changed" in (
        result.child_output_path.read_text(encoding="utf-8")
    )


def test_review_pause_is_visible_and_does_not_fail_the_top_level_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    run_dir = tmp_path / "run"

    def pause_for_review(**kwargs: object) -> int:
        child_run_dir = Path(str(kwargs["run_dir"]))
        state_path = child_run_dir / "asset_run.json"
        _plan_and_begin(state_path, "articulation")
        directory = stage_directory(state_path, "articulation")
        directory.mkdir(parents=True, exist_ok=True)
        candidates = directory / "articulation_candidates.json"
        candidates.write_text('{"joints":[{"id":"drawer"}]}\n', encoding="utf-8")
        _record_await_review(state_path, candidates)
        require_review(state_path, candidates_path=candidates)
        return 0

    monkeypatch.setattr(asset_runner, "_run_child_agent", pause_for_review)

    result = run_asset_workflow(replace(_config(paths, run_dir), dry_run=False))

    assert result.returncode == 3
    assert result.needs_review
    run = load_verified_run(run_dir / "asset_run.json")
    assert run.terminal_status == "active"
    assert run.stages["articulation"].status == "needs_review"
