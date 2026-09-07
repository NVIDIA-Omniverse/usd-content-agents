# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from content_agent_workflows.articulation import (
    ARTICULATION_PREPARATION_LEAF_ID,
    ArticulationPreparationFailureReceipt,
    ArticulationPreparationInspectionReadback,
    ArticulationPreparationLeafInvocation,
    ArticulationPreparationLeafResult,
    ArticulationPreparationLeafTerminalResult,
    ArticulationProposalAttemptFailed,
    ArticulationProposalLeafInvocation,
    ArtifactJsonArticulationProposalProvider,
    EmbeddedArticulationError,
    EmbeddedArticulationPreparation,
    articulation_asset_leaf_runtime_bindings,
    request_embedded_articulation_provider_proposal,
)
from content_agent_workflows.asset_composition import (
    ArtifactBinding as AssetArtifactBinding,
)
from content_agent_workflows.asset_composition import (
    AssetExecutionGraph,
    AssetExecutionNode,
    AssetLeafReceipt,
    AssetRunRequest,
    AssetRuntimeRequest,
    AssetSoleCoordinatorIdentity,
    AssetSourceStaging,
    begin_leaf,
    canonical_asset_digest,
    create_run,
    fail_leaf,
    freeze_execution_graph,
    leaf_directory,
    load_verified_run,
    repository_asset_leaf_runtime_catalog,
    validate_terminal,
)
from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_decision import (
    DomainProposalPayload,
    ProducerIdentity,
    ProviderNeutralEvidenceRecord,
)
from world_understanding.functions.physics.joint_rigger import identify_usd_artifact
from world_understanding.utils.artifacts import ArtifactPathError

from content_workflow_cli.cli import main


@pytest.mark.parametrize(
    ("artifact_error", "expected_error", "message"),
    (
        (
            "Artifact source must be a regular file",
            ValueError,
            "bounded unlinked file",
        ),
        (
            "Refusing to traverse a symlinked artifact path",
            OSError,
            "Could not open selected-leaf invocation",
        ),
    ),
)
def test_selected_leaf_invocation_maps_confined_open_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_error: str,
    expected_error: type[Exception],
    message: str,
) -> None:
    import world_understanding.utils.artifacts as artifact_utils

    from content_workflow_cli import articulation_capability_runner as runner

    @contextmanager
    def fail_open(_path: Path):
        raise ArtifactPathError(artifact_error)
        yield  # pragma: no cover - contextmanager shape

    monkeypatch.setattr(artifact_utils, "open_regular_file_no_follow", fail_open)

    with pytest.raises(expected_error, match=message):
        runner._open_selected_leaf_invocation(tmp_path / "invocation.json")


def test_selected_leaf_identity_normalizes_windows_ctime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_workflow_cli import articulation_capability_runner as runner

    common = {
        "st_dev": 1,
        "st_ino": 2,
        "st_mode": 3,
        "st_nlink": 1,
        "st_size": 4,
        "st_mtime_ns": 5,
    }
    before = SimpleNamespace(**common, st_ctime_ns=6)
    after = SimpleNamespace(**common, st_ctime_ns=7)
    monkeypatch.setattr(runner, "_stable_ctime_ns", lambda _metadata: 0)

    assert before.st_ctime_ns != after.st_ctime_ns
    assert runner._selected_leaf_identity(before) == runner._selected_leaf_identity(
        after
    )


def _binding(path: Path) -> ExecutionArtifactBinding:
    resolved = path.resolve()
    return ExecutionArtifactBinding(
        path=str(resolved),
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _asset_binding(path: Path) -> AssetArtifactBinding:
    return AssetArtifactBinding.model_validate(_binding(path).model_dump(mode="json"))


def _create_preparation_graph_attempt(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = run_dir / "source.usda"
    source.write_text('#usda 1.0\ndef Xform "World" {}\n', encoding="utf-8")
    source_binding = _asset_binding(source)
    dependency_digest_set = canonical_asset_digest([source_binding.sha256])
    manifest = run_dir / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_usd_path": source_binding.path,
                "source_sha256": source_binding.sha256,
                "staged_usd_path": source_binding.path,
                "dependency_digest_set_sha256": dependency_digest_set,
                "unresolved_dependencies": [],
                "files": [
                    {
                        "source_path": source_binding.path,
                        "staged_path": source_binding.path,
                        "sha256": source_binding.sha256,
                        "size_bytes": source_binding.size_bytes,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    staging = AssetSourceStaging(
        original_source=source_binding,
        original_dependencies=[source_binding],
        staged_source=source_binding,
        staged_dependencies=[source_binding],
        manifest=_asset_binding(manifest),
        dependency_digest_set_sha256=dependency_digest_set,
    )
    catalog = repository_asset_leaf_runtime_catalog().catalog
    coordinator = AssetSoleCoordinatorIdentity.create(
        coordinator_id="asset-coordinator:preparation-failure-test",
        invocation_id="single-prompt:preparation-failure-test",
        actor="test-outer-reasoner",
        implementation="content-workflow-cli-test",
    )
    runtime = AssetRuntimeRequest(
        runner="codex",
        scene_tool_timeout_seconds=60.0,
        child_timeout_seconds=0.0,
        codex_sandbox_mode="workspace-write",
        claude_permission_mode="default",
        claude_execution_mode="sdk",
    )
    prompt = "Run the selected Articulation preparation leaf."
    configuration_digest = canonical_asset_digest(
        {
            "repository_root": str(tmp_path.resolve()),
            "runtime": runtime.model_dump(mode="json"),
            "requires_parent_resource_release": False,
        }
    )
    state_path = run_dir / "asset_run.json"
    request = AssetRunRequest(
        schema_version="content-agents.asset-composition-request.v3",
        created_at="2026-08-25T00:00:00+00:00",
        selected_mode="agentic",
        run_id="preparation-failure-test",
        run_dir=str(run_dir.resolve()),
        run_state=str(state_path.resolve()),
        repository_root=str(tmp_path.resolve()),
        source_asset=source_binding.path,
        source_staging=staging,
        prompt=prompt,
        prompt_digest=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        source_digest=source_binding.sha256,
        configuration_digest=configuration_digest,
        reference_digest=canonical_asset_digest([]),
        sole_coordinator_identity=coordinator,
        leaf_catalog=catalog,
        requires_parent_resource_release=False,
        runtime=runtime,
    )
    request_path = run_dir / "request.json"
    request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
    create_run(
        state_path,
        run_id=request.run_id,
        request_path=request_path,
        source_asset=source,
    )
    descriptor = next(
        item
        for item in catalog.descriptors
        if item.leaf_id == ARTICULATION_PREPARATION_LEAF_ID
    )
    graph = AssetExecutionGraph.create(
        sole_coordinator_identity_digest=coordinator.identity_digest,
        prompt_digest=request.prompt_digest,
        source_digest=request.source_digest,
        configuration_digest=request.configuration_digest,
        reference_digest=request.reference_digest,
        leaf_catalog_digest=catalog.catalog_digest,
        nodes=[
            AssetExecutionNode(
                leaf_id=ARTICULATION_PREPARATION_LEAF_ID,
                depends_on=[],
                requirement="required",
                descriptor_digest=descriptor.descriptor_digest,
                terminal_output=True,
            )
        ],
        omitted_leaf_ids=sorted(
            item.leaf_id
            for item in catalog.descriptors
            if item.leaf_id != ARTICULATION_PREPARATION_LEAF_ID
        ),
    )
    graph_path = run_dir / "execution_graph.json"
    graph_path.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    freeze_execution_graph(state_path, graph_path=graph_path)
    begin_leaf(state_path, ARTICULATION_PREPARATION_LEAF_ID)
    return state_path, leaf_directory(state_path, ARTICULATION_PREPARATION_LEAF_ID)


def _retained_claim(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    return {
        "relative_path": relative_path,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _wrong_v1_packaged_preparation_readback(
    tmp_path: Path,
) -> tuple[Path, Path, str, str]:
    from pxr import UsdUtils

    retained_root = tmp_path / "retained"
    retained_root.mkdir()
    (retained_root / "dep.usda").write_text(
        '#usda 1.0\ndef Xform "World" {}\n',
        encoding="utf-8",
    )
    (retained_root / "source.usda").write_text(
        "#usda 1.0\n(\n    subLayers = [@dep.usda@]\n)\n",
        encoding="utf-8",
    )
    package = retained_root / "source.usdz"
    assert UsdUtils.CreateNewUsdzPackage(
        str(retained_root / "source.usda"),
        str(package),
    )
    (retained_root / "source.usda").unlink()
    (retained_root / "dep.usda").unlink()
    memberships = [
        {
            "member_prim": "/World",
            "authoritative_owner_prim": "/World",
            "disposition": "independent_motion",
        }
    ]
    (retained_root / "inspector-config.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.articulation-preparation-configuration.v1"
                ),
                "membership_policy": "retained-explicit-membership-v1",
                "memberships": memberships,
                "capabilities": {
                    "canonical_output_evidence_required": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (retained_root / "inspector.py").write_text(
        "INSPECTOR_CONTRACT = 'articulation-preparation-v1'\n",
        encoding="utf-8",
    )
    (retained_root / "render.json").write_text("{}\n", encoding="utf-8")
    (retained_root / "scene.json").write_text("{}\n", encoding="utf-8")
    identity = identify_usd_artifact(package, uri=package.as_uri())
    assert identity.dependency_bundle_sha256 is not None
    observed_dependency = identity.dependency_bundle_sha256
    expected_dependency = "0" * 64
    assert observed_dependency != expected_dependency
    payload = {
        "schema_version": (
            "content-agent-workflows.articulation-preparation-readback.v1"
        ),
        "inspector_id": "deterministic-usd-readback",
        "inspector_implementation": "fixture.saved-stage-inspector.v1",
        "source": _retained_claim(retained_root, "source.usdz"),
        "dependencies": [],
        "dependency_entry_count": 0,
        "dependency_closure_complete": True,
        "source_dependency_bundle_sha256": expected_dependency,
        "configuration": _retained_claim(retained_root, "inspector-config.json"),
        "inspector_implementation_artifact": _retained_claim(
            retained_root,
            "inspector.py",
        ),
        "saved_stage": _retained_claim(retained_root, "source.usdz"),
        "saved_stage_dependency_bundle_sha256": expected_dependency,
        "hierarchy": [
            {
                "prim_path": "/World",
                "parent_prim_path": None,
                "type_name": "Xform",
            }
        ],
        "memberships": memberships,
        "render_artifacts": [_retained_claim(retained_root, "render.json")],
        "scene_artifacts": [_retained_claim(retained_root, "scene.json")],
        "proposal_status": "not_requested",
        "readback_complete": True,
    }
    readback = retained_root / "readback.json"
    readback.write_text(
        ArticulationPreparationInspectionReadback.model_validate(
            payload
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    return retained_root, readback, expected_dependency, observed_dependency


@pytest.mark.parametrize(
    ("operation", "runner_name"),
    (
        ("author", "run_articulation_author_asset_leaf"),
        ("evidence", "run_articulation_evidence_asset_leaf"),
        ("review", "run_articulation_review_asset_leaf"),
        ("publish", "run_articulation_publish_asset_leaf"),
    ),
)
def test_articulation_agentic_leaf_cli_dispatches_one_focused_executor(
    operation: str,
    runner_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invocation = tmp_path / f"{operation}-invocation.json"
    observed: list[Path] = []

    def run(path: Path) -> Any:
        observed.append(path)
        return SimpleNamespace(
            model_dump=lambda *, mode: {
                "leaf": operation,
                "mode": mode,
                "nested_coordinator_invoked": False,
            }
        )

    monkeypatch.setattr(
        f"content_agent_workflows.articulation.{runner_name}",
        run,
    )

    assert (
        main(
            [
                "articulation",
                "agentic-leaf",
                operation,
                "--invocation",
                str(invocation),
            ]
        )
        == 0
    )
    assert observed == [invocation]
    assert json.loads(capsys.readouterr().out) == {
        "leaf": operation,
        "mode": "json",
        "nested_coordinator_invoked": False,
    }


def test_articulation_publish_preparation_cli_forwards_exact_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readback = tmp_path / "readback.json"
    retained_root = tmp_path / "retained"
    output_dir = tmp_path / "publication"
    observed: dict[str, Any] = {}
    binding = ExecutionArtifactBinding(
        path=str((output_dir / "embedded_articulation_preparation.json").resolve()),
        sha256="a" * 64,
        size_bytes=1,
    )

    def publish(
        readback_path: str,
        *,
        retained_root: str,
        output_dir: str,
        expected_readback: ExecutionArtifactBinding | None = None,
    ) -> Any:
        observed.update(
            readback_path=readback_path,
            retained_root=retained_root,
            output_dir=output_dir,
            expected_readback=expected_readback,
        )
        return SimpleNamespace(
            model_dump=lambda *, mode: {"preparation": binding.model_dump(mode=mode)}
        )

    monkeypatch.setattr(
        "content_agent_workflows.articulation.publish_embedded_articulation_preparation",
        publish,
    )
    result = main(
        [
            "articulation",
            "publish-preparation",
            "--readback",
            str(readback),
            "--retained-root",
            str(retained_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert observed == {
        "readback_path": str(readback),
        "retained_root": str(retained_root),
        "output_dir": str(output_dir),
        "expected_readback": None,
    }
    assert json.loads(capsys.readouterr().out)["preparation"] == binding.model_dump(
        mode="json"
    )


def test_articulation_publish_preparation_cli_accepts_bound_leaf_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    retained_root = tmp_path / "retained"
    retained_root.mkdir()
    readback = retained_root / "readback.json"
    readback.write_text('{"fixture":true}\n', encoding="utf-8")
    invocation = ArticulationPreparationLeafInvocation(
        readback=_binding(readback),
        retained_root=str(retained_root),
    )
    invocation_path = tmp_path / "preparation-invocation.json"
    invocation_path.write_text(invocation.model_dump_json(indent=2), encoding="utf-8")
    observed: dict[str, Any] = {}

    def publish(
        readback_path: str,
        *,
        retained_root: str,
        output_dir: str,
        expected_readback: ExecutionArtifactBinding | None = None,
    ) -> Any:
        observed.update(
            readback_path=readback_path,
            retained_root=retained_root,
            output_dir=output_dir,
            expected_readback=expected_readback,
        )
        return SimpleNamespace(model_dump=lambda *, mode: {"mode": mode})

    monkeypatch.setattr(
        "content_agent_workflows.articulation.publish_embedded_articulation_preparation",
        publish,
    )
    assert (
        main(
            [
                "articulation",
                "publish-preparation",
                "--invocation",
                str(invocation_path),
            ]
        )
        == 0
    )
    assert observed == {
        "readback_path": str(readback),
        "retained_root": str(retained_root),
        "output_dir": str(tmp_path / "articulation-preparation-publication"),
        "expected_readback": _binding(readback),
    }
    assert json.loads(capsys.readouterr().out) == {"mode": "json"}


def test_articulation_publish_preparation_cli_seals_generic_typed_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    retained_root, readback, expected_dependency, observed_dependency = (
        _wrong_v1_packaged_preparation_readback(tmp_path)
    )
    invocation = ArticulationPreparationLeafInvocation(
        readback=_binding(readback),
        retained_root=str(retained_root),
    )
    invocation_path = tmp_path / "preparation-invocation.json"
    invocation_path.write_text(invocation.model_dump_json(indent=2), encoding="utf-8")

    assert (
        main(
            [
                "articulation",
                "publish-preparation",
                "--invocation",
                str(invocation_path),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert expected_dependency not in captured.out + captured.err
    assert observed_dependency not in captured.out + captured.err
    payload = json.loads(captured.out)
    terminal = ArticulationPreparationLeafTerminalResult.model_validate(payload)
    assert terminal.error == (
        "Articulation preparation input failed deterministic validation."
    )
    receipt = ArticulationPreparationFailureReceipt.model_validate_json(
        Path(terminal.native_terminal_receipt.path).read_bytes()
    )
    assert receipt.invocation == _binding(invocation_path)

    result_path = tmp_path / "preparation-result.json"
    result_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    runtime = next(
        item
        for item in articulation_asset_leaf_runtime_bindings()
        if item.descriptor.leaf_id == ARTICULATION_PREPARATION_LEAF_ID
    )
    parsed_result = ArticulationPreparationLeafResult.model_validate(payload)
    projection = runtime.project(
        invocation,
        parsed_result,
        invocation_artifact=AssetArtifactBinding.model_validate(
            _binding(invocation_path).model_dump(mode="json")
        ),
        result_artifact=AssetArtifactBinding.model_validate(
            _binding(result_path).model_dump(mode="json")
        ),
    )
    assert projection.payload.native_disposition == "failed"
    assert projection.payload.native_status == "invalid_preparation_input"
    assert projection.payload.error == terminal.error


def test_articulation_preparation_failure_replays_before_restored_readback_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    retained_root = tmp_path / "retained"
    retained_root.mkdir()
    readback = retained_root / "readback.json"
    readback_bytes = b'{"fixture":true}\n'
    readback.write_bytes(readback_bytes)
    invocation = ArticulationPreparationLeafInvocation(
        readback=_binding(readback),
        retained_root=str(retained_root),
    )
    invocation_path = tmp_path / "preparation-invocation.json"
    invocation_path.write_text(invocation.model_dump_json(indent=2), encoding="utf-8")
    readback.unlink()

    assert (
        main(
            [
                "articulation",
                "publish-preparation",
                "--invocation",
                str(invocation_path),
            ]
        )
        == 1
    )
    first_terminal = ArticulationPreparationLeafTerminalResult.model_validate_json(
        capsys.readouterr().out
    )
    failure_receipt_path = Path(first_terminal.native_terminal_receipt.path)
    failure_receipt_bytes = failure_receipt_path.read_bytes()
    readback.write_bytes(readback_bytes)
    publish_calls = 0

    def publish(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal publish_calls
        publish_calls += 1
        return SimpleNamespace(model_dump=lambda *, mode: {"mode": mode})

    monkeypatch.setattr(
        "content_agent_workflows.articulation.publish_embedded_articulation_preparation",
        publish,
    )
    assert (
        main(
            [
                "articulation",
                "publish-preparation",
                "--invocation",
                str(invocation_path),
            ]
        )
        == 1
    )
    replayed_terminal = ArticulationPreparationLeafTerminalResult.model_validate_json(
        capsys.readouterr().out
    )
    assert replayed_terminal == first_terminal
    assert failure_receipt_path.read_bytes() == failure_receipt_bytes
    assert publish_calls == 0
    assert not (tmp_path / "articulation-preparation-publication").exists()

    tampered_receipt = json.loads(failure_receipt_bytes)
    tampered_receipt["invocation"]["sha256"] = "0" * 64
    failure_receipt_path.write_text(
        json.dumps(tampered_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "articulation",
                "publish-preparation",
                "--invocation",
                str(invocation_path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "differs from the exact expected receipt" in captured.err
    assert publish_calls == 0
    assert not (tmp_path / "articulation-preparation-publication").exists()


def test_public_preparation_failure_crosses_exact_graph_fail_leaf_boundary(
    tmp_path: Path,
) -> None:
    state_path, attempt_root = _create_preparation_graph_attempt(tmp_path)
    retained_root, readback, expected_dependency, observed_dependency = (
        _wrong_v1_packaged_preparation_readback(attempt_root)
    )
    invocation = ArticulationPreparationLeafInvocation(
        readback=_binding(readback),
        retained_root=str(retained_root),
    )
    invocation_path = attempt_root / "preparation-invocation.json"
    invocation_path.write_text(invocation.model_dump_json(indent=2), encoding="utf-8")

    public_cli = Path(sys.executable).with_name(
        "content-workflow-cli.exe" if os.name == "nt" else "content-workflow-cli"
    )
    assert public_cli.is_file()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(str(item) for item in sys.path)
    completed = subprocess.run(  # noqa: S603 - exact installed public CLI
        [
            str(public_cli),
            "articulation",
            "publish-preparation",
            "--invocation",
            str(invocation_path),
        ],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        env=environment,
        text=False,
        timeout=60,
    )

    assert completed.returncode == 1
    assert completed.stderr == b""
    assert expected_dependency.encode("ascii") not in completed.stdout
    assert observed_dependency.encode("ascii") not in completed.stdout
    terminal = ArticulationPreparationLeafTerminalResult.model_validate_json(
        completed.stdout
    )
    assert terminal.error == (
        "Articulation preparation input failed deterministic validation."
    )
    result_path = attempt_root / "preparation-result.json"
    result_path.write_bytes(completed.stdout)
    assert result_path.read_bytes() == completed.stdout

    failed = fail_leaf(
        state_path,
        ARTICULATION_PREPARATION_LEAF_ID,
        reason=terminal.error,
        invocation_path=invocation_path,
        result_path=result_path,
    )

    assert failed.terminal_status == "failed"
    leaf_state = failed.leaf_states[ARTICULATION_PREPARATION_LEAF_ID]
    assert leaf_state.status == "failed"
    assert leaf_state.receipt is not None
    receipt_path = Path(leaf_state.receipt.path)
    assert receipt_path.parent == attempt_root
    receipt = AssetLeafReceipt.model_validate_json(receipt_path.read_bytes())
    assert receipt.invocation == _asset_binding(invocation_path)
    assert receipt.result == _asset_binding(result_path)
    assert receipt.native_terminal_receipt == AssetArtifactBinding.model_validate(
        terminal.native_terminal_receipt.model_dump(mode="json")
    )
    assert receipt.native_disposition == "failed"
    assert receipt.native_status == "invalid_preparation_input"
    assert receipt.error == terminal.error
    assert Path(receipt.native_terminal_receipt.path).parent == attempt_root
    assert not (attempt_root / "articulation-preparation-publication").exists()

    reread = load_verified_run(state_path)
    assert reread == failed
    validation = validate_terminal(state_path)
    assert validation.valid is False
    assert validation.terminal_status == "failed"
    assert any("Run terminal status is failed" in error for error in validation.errors)
    assert any(
        f"{ARTICULATION_PREPARATION_LEAF_ID}=failed" in error
        for error in validation.errors
    )


def test_articulation_preparation_failure_rejects_reformatted_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    retained_root = tmp_path / "retained"
    retained_root.mkdir()
    readback = retained_root / "readback.json"
    readback.write_text('{"fixture":true}\n', encoding="utf-8")
    invocation = ArticulationPreparationLeafInvocation(
        readback=_binding(readback),
        retained_root=str(retained_root),
    )
    invocation_path = tmp_path / "preparation-invocation.json"
    invocation_path.write_text(invocation.model_dump_json(indent=2), encoding="utf-8")
    original_bytes = invocation_path.read_bytes()

    def mutate_invocation_then_fail(*_args: Any, **_kwargs: Any) -> Any:
        invocation_path.write_text(
            json.dumps(
                invocation.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        assert invocation_path.read_bytes() != original_bytes
        assert (
            ArticulationPreparationLeafInvocation.model_validate_json(
                invocation_path.read_bytes()
            )
            == invocation
        )
        raise EmbeddedArticulationError("publisher rejected the retained input")

    monkeypatch.setattr(
        "content_agent_workflows.articulation.publish_embedded_articulation_preparation",
        mutate_invocation_then_fail,
    )
    assert (
        main(
            [
                "articulation",
                "publish-preparation",
                "--invocation",
                str(invocation_path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invocation changed before failure sealing" in captured.err
    assert not (tmp_path / "articulation_preparation_failure_receipt.json").exists()
    assert not (tmp_path / "articulation-preparation-publication").exists()


def test_articulation_validation_cli_leaves_forward_exact_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, str] = {}

    def validate_preparation(path: str) -> Any:
        observed["publication"] = path
        return SimpleNamespace(
            model_dump=lambda *, mode: {"kind": "preparation", "mode": mode}
        )

    def validate_attempt(path: str) -> Any:
        observed["terminal_receipt"] = path
        return SimpleNamespace(
            model_dump=lambda *, mode: {"kind": "attempt", "mode": mode}
        )

    monkeypatch.setattr(
        "content_agent_workflows.articulation.validate_embedded_articulation_preparation_publication",
        validate_preparation,
    )
    monkeypatch.setattr(
        "content_agent_workflows.articulation.validate_articulation_proposal_attempt_receipt",
        validate_attempt,
    )

    assert (
        main(
            [
                "articulation",
                "validate-preparation",
                "--publication",
                "publication.json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["kind"] == "preparation"
    assert (
        main(
            [
                "articulation",
                "validate-attempt",
                "--terminal-receipt",
                "terminal.json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["kind"] == "attempt"
    assert observed == {
        "publication": "publication.json",
        "terminal_receipt": "terminal.json",
    }


def _record(
    evidence_id: str,
    evidence_type: Literal["inspection", "render", "capability"],
) -> ProviderNeutralEvidenceRecord:
    return ProviderNeutralEvidenceRecord(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        status="available",
        summary=f"Exact {evidence_id} evidence.",
        facts={"complete": True},
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    preparation = EmbeddedArticulationPreparation(
        source_sha256="a" * 64,
        source_dependency_bundle_sha256="b" * 64,
        configuration_sha256="c" * 64,
        evidence_provider=ProducerIdentity(
            producer_id="deterministic-inspection",
            role="evidence_provider",
            implementation="test-inspector",
            implementation_digest="d" * 64,
        ),
        source_hierarchy=_record("source-hierarchy-inspection", "inspection"),
        source_members=_record("joint-source-member-inspection", "inspection"),
        authoritative_owners=_record(
            "joint-authoritative-owner-inspection", "inspection"
        ),
        capabilities=_record("joint-authoring-capabilities", "capability"),
        renders=_record("joint-render-inspection", "render"),
        scene=_record("joint-scene-inspection", "inspection"),
        proposal_status="not_evaluated",
    )
    preparation_path = tmp_path / "preparation.json"
    preparation_path.write_text(preparation.model_dump_json(indent=2), encoding="utf-8")
    payload_path = tmp_path / "provider-payload.json"
    payload_path.write_text(
        DomainProposalPayload(
            schema_version="replacement.example.v1",
            values={"candidate_hints": [{"id": "candidate-1"}]},
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    return preparation_path, payload_path


def test_articulation_propose_cli_uses_public_artifact_adapter_without_joint_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation, payload = _inputs(tmp_path)

    def forbidden_joint_client(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("JointAgentLocalClient must not be constructed")

    monkeypatch.setattr(
        "content_agent_workflows.articulation.client.JointAgentLocalClient.__init__",
        forbidden_joint_client,
    )
    output_dir = tmp_path / "proposal"
    result = main(
        [
            "articulation",
            "propose",
            "--preparation",
            str(preparation),
            "--output-dir",
            str(output_dir),
            "--intent",
            "Suggest advisory articulation candidates.",
            "--provider-adapter",
            "artifact-json",
            "--provider-id",
            "replacement-provider",
            "--capability-id",
            "articulation-proposal-v1",
            "--provider-payload",
            str(payload),
        ]
    )

    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["result"]["producer"]["producer_id"] == "replacement-provider"
    assert printed["joint_agent_local_client_invoked"] is False
    assert Path(printed["bound_preparation"]["path"]).is_file()
    assert (
        main(
            [
                "articulation",
                "validate-attempt",
                "--terminal-receipt",
                printed["terminal_receipt"]["path"],
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["disposition"] == "succeeded"


def test_articulation_propose_cli_has_no_default_provider_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation, _payload = _inputs(tmp_path)

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("an unselected proposal provider was constructed")

    monkeypatch.setattr(
        ArtifactJsonArticulationProposalProvider,
        "__init__",
        forbidden_provider,
    )
    monkeypatch.setattr(
        "content_agent_workflows.articulation.HttpJsonArticulationProposalProvider.__init__",
        forbidden_provider,
    )

    result = main(
        [
            "articulation",
            "propose",
            "--preparation",
            str(preparation),
            "--output-dir",
            str(tmp_path / "proposal"),
            "--intent",
            "Do not infer a provider backend.",
        ]
    )

    assert result == 2
    error = capsys.readouterr().err
    assert "--provider-adapter" in error
    assert "--provider-id" in error
    assert "--capability-id" in error
    assert not (tmp_path / "proposal").exists()


def test_articulation_propose_cli_executes_exact_selected_leaf_invocation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation, payload = _inputs(tmp_path)
    invocation = ArticulationProposalLeafInvocation(
        preparation=_binding(preparation),
        intent="Suggest advisory articulation candidates.",
        provider={
            "adapter": "artifact-json",
            "provider_id": "replacement-provider",
            "capability_id": "articulation-proposal-v1",
            "provider_payload": _binding(payload),
        },
    )
    invocation_path = tmp_path / "proposal-invocation.json"
    invocation_path.write_text(invocation.model_dump_json(indent=2), encoding="utf-8")

    assert (
        main(
            [
                "articulation",
                "propose",
                "--invocation",
                str(invocation_path),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["disposition"] == "succeeded"
    assert printed["provider"]["producer_id"] == "replacement-provider"
    assert Path(printed["proposal"]["path"]).is_file()
    assert (
        main(
            [
                "articulation",
                "propose",
                "--invocation",
                str(invocation_path),
            ]
        )
        == 2
    )
    assert "each provider attempt requires a fresh output directory" in (
        capsys.readouterr().err
    )


def test_articulation_propose_selected_leaf_returns_failed_terminal_on_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation, payload = _inputs(tmp_path)
    payload.write_text('{"invalid":true}\n', encoding="utf-8")
    invocation = ArticulationProposalLeafInvocation(
        preparation=_binding(preparation),
        intent="Retain the exact failed provider attempt.",
        provider={
            "adapter": "artifact-json",
            "provider_id": "invalid-provider",
            "capability_id": "articulation-proposal-v1",
            "provider_payload": _binding(payload),
        },
    )
    invocation_path = tmp_path / "failed-proposal-invocation.json"
    invocation_path.write_text(invocation.model_dump_json(indent=2), encoding="utf-8")

    assert (
        main(
            [
                "articulation",
                "propose",
                "--invocation",
                str(invocation_path),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert captured.err == ""
    assert printed["disposition"] == "invalid_response"
    retained = (
        tmp_path
        / "articulation-proposal-attempt"
        / "articulation_proposal_attempt_terminal.json"
    )
    assert json.loads(retained.read_text(encoding="utf-8")) == printed


def test_articulation_selected_leaf_invocation_rejects_bound_input_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation, payload = _inputs(tmp_path)
    invocation = ArticulationProposalLeafInvocation(
        preparation=_binding(preparation),
        intent="Do not invoke after input drift.",
        provider={
            "adapter": "artifact-json",
            "provider_id": "replacement-provider",
            "capability_id": "articulation-proposal-v1",
            "provider_payload": _binding(payload),
        },
    )
    invocation_path = tmp_path / "drifted-invocation.json"
    invocation_path.write_text(invocation.model_dump_json(indent=2), encoding="utf-8")
    payload.write_text('{"changed":true}\n', encoding="utf-8")

    assert (
        main(
            [
                "articulation",
                "propose",
                "--invocation",
                str(invocation_path),
            ]
        )
        == 2
    )
    assert "differs from the selected-leaf invocation" in capsys.readouterr().err
    assert not (tmp_path / "articulation-proposal-attempt").exists()


def test_articulation_selected_leaf_invocation_rejects_replacement_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation, payload = _inputs(tmp_path)
    terminal = tmp_path / "prior-terminal.json"
    terminal.write_text('{"attempt":"first"}\n', encoding="utf-8")
    invocation = ArticulationProposalLeafInvocation(
        preparation=_binding(preparation),
        intent="Replace the exact prior attempt.",
        provider={
            "adapter": "artifact-json",
            "provider_id": "replacement-provider",
            "capability_id": "articulation-proposal-v1",
            "provider_payload": _binding(payload),
        },
        replaces_terminal_receipt=_binding(terminal),
        replacement_reason="The prior provider attempt failed.",
    )
    invocation_path = tmp_path / "replacement-invocation.json"
    invocation_path.write_text(invocation.model_dump_json(indent=2), encoding="utf-8")
    terminal.write_text('{"attempt":"substituted"}\n', encoding="utf-8")

    assert (
        main(
            [
                "articulation",
                "propose",
                "--invocation",
                str(invocation_path),
            ]
        )
        == 2
    )
    assert "differs from the selected-leaf invocation" in capsys.readouterr().err
    assert not (tmp_path / "articulation-proposal-attempt").exists()


def test_articulation_selected_leaf_invocation_rejects_symlinked_envelope(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation, payload = _inputs(tmp_path)
    invocation = ArticulationProposalLeafInvocation(
        preparation=_binding(preparation),
        intent="Do not follow a substituted invocation.",
        provider={
            "adapter": "artifact-json",
            "provider_id": "replacement-provider",
            "capability_id": "articulation-proposal-v1",
            "provider_payload": _binding(payload),
        },
    )
    target = tmp_path / "target-invocation.json"
    target.write_text(invocation.model_dump_json(indent=2), encoding="utf-8")
    symlink = tmp_path / "invocation-link.json"
    symlink.symlink_to(target)

    assert (
        main(
            [
                "articulation",
                "propose",
                "--invocation",
                str(symlink),
            ]
        )
        == 2
    )
    assert "invocation is unavailable" in capsys.readouterr().err
    assert not (tmp_path / "articulation-proposal-attempt").exists()


def test_articulation_selected_leaf_invocation_rejects_symlinked_parent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation, payload = _inputs(tmp_path)
    invocation = ArticulationProposalLeafInvocation(
        preparation=_binding(preparation),
        intent="Do not traverse a symlinked invocation parent.",
        provider={
            "adapter": "artifact-json",
            "provider_id": "replacement-provider",
            "capability_id": "articulation-proposal-v1",
            "provider_payload": _binding(payload),
        },
    )
    real_parent = tmp_path / "real-invocation-parent"
    real_parent.mkdir()
    invocation_path = real_parent / "invocation.json"
    invocation_path.write_text(invocation.model_dump_json(indent=2), encoding="utf-8")
    linked_parent = tmp_path / "linked-invocation-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    assert (
        main(
            [
                "articulation",
                "propose",
                "--invocation",
                str(linked_parent / invocation_path.name),
            ]
        )
        == 2
    )
    assert "invocation is unavailable" in capsys.readouterr().err
    assert not (real_parent / "articulation-proposal-attempt").exists()


def test_articulation_selected_leaf_invocation_rejects_fifo_without_blocking(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invocation_fifo = tmp_path / "invocation.fifo"
    os.mkfifo(invocation_fifo)

    assert (
        main(
            [
                "articulation",
                "propose",
                "--invocation",
                str(invocation_fifo),
            ]
        )
        == 2
    )
    assert "bounded unlinked file" in capsys.readouterr().err


def test_articulation_help_lists_provider_neutral_leaf_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["articulation", "--help"]) == 0

    help_text = capsys.readouterr().out
    assert "publish-preparation" in help_text
    assert "validate-preparation" in help_text
    assert "validate-attempt" in help_text


@pytest.mark.parametrize(
    ("command", "direct_option", "expected_message"),
    (
        (
            "propose",
            ("--intent", "mixed direct intent"),
            "cannot be mixed with direct proposal options",
        ),
        (
            "publish-preparation",
            ("--readback", "mixed-readback.json"),
            "cannot be mixed with direct preparation options",
        ),
    ),
)
def test_articulation_selected_leaf_invocation_rejects_direct_option_mix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    direct_option: tuple[str, str],
    expected_message: str,
) -> None:
    assert (
        main(
            [
                "articulation",
                command,
                "--invocation",
                str(tmp_path / "unused-invocation.json"),
                *direct_option,
            ]
        )
        == 2
    )
    assert expected_message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "expected_message"),
    (
        ("propose", "direct proposal invocation requires"),
        ("publish-preparation", "direct preparation invocation requires"),
    ),
)
def test_articulation_direct_leaf_invocation_requires_complete_flags(
    capsys: pytest.CaptureFixture[str],
    command: str,
    expected_message: str,
) -> None:
    assert main(["articulation", command]) == 2
    assert expected_message in capsys.readouterr().err


def test_articulation_selected_leaf_rejects_model_authored_http_and_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation, _ = _inputs(tmp_path)
    invocation_path = tmp_path / "http-invocation.json"
    invocation_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.articulation-proposal-leaf-invocation.v1"
                ),
                "preparation": _binding(preparation).model_dump(mode="json"),
                "intent": "Do not resolve model-authored credentials.",
                "provider": {
                    "adapter": "http-json",
                    "provider_id": "http-provider",
                    "capability_id": "articulation-proposal-v1",
                    "endpoint_alias": "primary",
                    "endpoint_url": "https://proposal.example.test/v1",
                    "bearer_token_env": "AWS_SECRET_ACCESS_KEY",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "runtime-only-secret")
    assert (
        main(
            [
                "articulation",
                "propose",
                "--invocation",
                str(invocation_path),
            ]
        )
        == 2
    )
    assert "selected-leaf invocation is invalid" in capsys.readouterr().err
    assert not (tmp_path / "articulation-proposal-attempt").exists()


@pytest.mark.parametrize(
    "mixed_option",
    (
        ("--provider-url", "https://proposal.example.test/v1/articulation"),
        ("--provider-timeout", "1"),
    ),
)
def test_articulation_propose_cli_rejects_mixed_adapter_modes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mixed_option: tuple[str, str],
) -> None:
    preparation, payload = _inputs(tmp_path)
    result = main(
        [
            "articulation",
            "propose",
            "--preparation",
            str(preparation),
            "--output-dir",
            str(tmp_path / "mixed"),
            "--intent",
            "Suggest candidates.",
            "--provider-adapter",
            "artifact-json",
            "--provider-id",
            "replacement-provider",
            "--capability-id",
            "articulation-proposal-v1",
            "--provider-payload",
            str(payload),
            *mixed_option,
        ]
    )

    assert result == 2
    assert "cannot be mixed with HTTP provider options" in capsys.readouterr().err
    assert not (tmp_path / "mixed").exists()


def test_articulation_propose_cli_requires_complete_http_configuration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation, _payload = _inputs(tmp_path)

    assert (
        main(
            [
                "articulation",
                "propose",
                "--preparation",
                str(preparation),
                "--output-dir",
                str(tmp_path / "incomplete-http"),
                "--intent",
                "Reject incomplete trusted HTTP configuration.",
                "--provider-adapter",
                "http-json",
                "--provider-id",
                "http-provider",
                "--capability-id",
                "articulation-proposal-v1",
            ]
        )
        == 2
    )
    assert (
        "http-json requires --provider-url and --provider-endpoint-alias"
        in capsys.readouterr().err
    )
    assert not (tmp_path / "incomplete-http").exists()


def test_articulation_propose_cli_rejects_unset_http_token_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation, _payload = _inputs(tmp_path)
    monkeypatch.delenv("ARTICULATION_TEST_TOKEN", raising=False)

    assert (
        main(
            [
                "articulation",
                "propose",
                "--preparation",
                str(preparation),
                "--output-dir",
                str(tmp_path / "unset-token"),
                "--intent",
                "Reject an unavailable operator credential.",
                "--provider-adapter",
                "http-json",
                "--provider-id",
                "http-provider",
                "--capability-id",
                "articulation-proposal-v1",
                "--provider-url",
                "https://proposal.example.test/v1/articulation",
                "--provider-endpoint-alias",
                "primary",
                "--provider-token-env",
                "ARTICULATION_TEST_TOKEN",
            ]
        )
        == 2
    )
    assert "ARTICULATION_TEST_TOKEN is not set or is empty" in capsys.readouterr().err
    assert not (tmp_path / "unset-token").exists()


def test_articulation_propose_cli_forwards_trusted_http_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation, _payload = _inputs(tmp_path)
    output_dir = tmp_path / "trusted-http"
    observed: dict[str, Any] = {}
    monkeypatch.setenv("ARTICULATION_TEST_TOKEN", "runtime-only-token")

    class CapturingHttpProvider:
        def __init__(self, **kwargs: Any) -> None:
            observed["provider"] = kwargs

    def request(
        preparation_path: str,
        *,
        output_dir: str,
        intent: str,
        provider: Any,
        replaces_terminal_receipt: str | None,
        replacement_reason: str | None,
        expected_preparation: ExecutionArtifactBinding | None,
        expected_replacement_terminal: ExecutionArtifactBinding | None,
    ) -> Any:
        observed.update(
            preparation_path=preparation_path,
            output_dir=output_dir,
            intent=intent,
            provider_instance=provider,
            replaces_terminal_receipt=replaces_terminal_receipt,
            replacement_reason=replacement_reason,
            expected_preparation=expected_preparation,
            expected_replacement_terminal=expected_replacement_terminal,
        )
        return SimpleNamespace(model_dump=lambda *, mode: {"mode": mode})

    monkeypatch.setattr(
        "content_agent_workflows.articulation.HttpJsonArticulationProposalProvider",
        CapturingHttpProvider,
    )
    monkeypatch.setattr(
        "content_agent_workflows.articulation.request_embedded_articulation_provider_proposal",
        request,
    )

    assert (
        main(
            [
                "articulation",
                "propose",
                "--preparation",
                str(preparation),
                "--output-dir",
                str(output_dir),
                "--intent",
                "Forward exact trusted HTTP configuration.",
                "--provider-adapter",
                "http-json",
                "--provider-id",
                "http-provider",
                "--capability-id",
                "articulation-proposal-v1",
                "--provider-url",
                "https://proposal.example.test/v1/articulation",
                "--provider-endpoint-alias",
                "primary",
                "--provider-token-env",
                "ARTICULATION_TEST_TOKEN",
            ]
        )
        == 0
    )
    assert observed["provider"] == {
        "provider_id": "http-provider",
        "capability_id": "articulation-proposal-v1",
        "endpoint_alias": "primary",
        "endpoint_url": "https://proposal.example.test/v1/articulation",
        "bearer_token": "runtime-only-token",
        "timeout_seconds": 120.0,
    }
    assert observed["preparation_path"] == str(preparation)
    assert observed["output_dir"] == str(output_dir)
    assert isinstance(observed["provider_instance"], CapturingHttpProvider)
    assert json.loads(capsys.readouterr().out) == {"mode": "json"}


def test_articulation_propose_cli_binds_replacement_and_reports_failed_terminal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation, invalid_payload = _inputs(tmp_path)
    invalid_payload.write_text('{"invalid":true}\n', encoding="utf-8")
    with pytest.raises(ArticulationProposalAttemptFailed) as first_error:
        request_embedded_articulation_provider_proposal(
            preparation,
            output_dir=tmp_path / "failed",
            intent="Preserve this failed attempt.",
            provider=ArtifactJsonArticulationProposalProvider(
                provider_id="failed-provider",
                capability_id="articulation-proposal-v1",
                payload_path=invalid_payload,
            ),
        )
    first_receipt = first_error.value.receipt_binding.path

    failure_result = main(
        [
            "articulation",
            "propose",
            "--preparation",
            str(preparation),
            "--output-dir",
            str(tmp_path / "cli-failed"),
            "--intent",
            "Report the failed terminal.",
            "--provider-adapter",
            "artifact-json",
            "--provider-id",
            "failed-provider",
            "--capability-id",
            "articulation-proposal-v1",
            "--provider-payload",
            str(invalid_payload),
        ]
    )
    assert failure_result == 1
    failed_output = json.loads(capsys.readouterr().err)
    assert failed_output["status"] == "failed"
    assert Path(failed_output["terminal_receipt"]["path"]).is_file()

    valid_payload = tmp_path / "valid-provider-payload.json"
    valid_payload.write_text(
        DomainProposalPayload(
            schema_version="replacement.example.v1",
            values={"candidate_hints": [{"id": "candidate-2"}]},
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    replacement_result = main(
        [
            "articulation",
            "propose",
            "--preparation",
            str(preparation),
            "--output-dir",
            str(tmp_path / "replacement"),
            "--intent",
            "Replace the exact failed attempt.",
            "--provider-adapter",
            "artifact-json",
            "--provider-id",
            "replacement-provider",
            "--capability-id",
            "articulation-proposal-v1",
            "--provider-payload",
            str(valid_payload),
            "--replaces-terminal-receipt",
            first_receipt,
            "--replacement-reason",
            "The explicit provider payload was corrected.",
        ]
    )
    assert replacement_result == 0
    printed = json.loads(capsys.readouterr().out)
    terminal = json.loads(Path(printed["terminal_receipt"]["path"]).read_text())
    assert terminal["replacement"]["prior_attempt_id"] == (
        first_error.value.receipt.attempt_id
    )


@pytest.mark.parametrize(
    ("command", "function_name", "arguments"),
    (
        (
            "project-graph-apply",
            "project_joint_graph_apply_result",
            ("--run-dir", "run", "--output-dir", "projection"),
        ),
        (
            "project-gate3a",
            "project_joint_gate3a_result",
            (
                "--source",
                "source.usda",
                "--output",
                "output.usda",
                "--report",
                "gate3a.json",
                "--closeout",
                "closeout.json",
                "--run-plan",
                "run-plan.json",
                "--intake",
                "intake.json",
                "--authoring-receipt",
                "receipts/authoring.json",
                "--output-dir",
                "projection",
            ),
        ),
        (
            "project-gate3b",
            "project_joint_gate3b_result",
            (
                "--source",
                "source.usda",
                "--output",
                "output.usda",
                "--report",
                "gate3b.json",
                "--closeout",
                "closeout.json",
                "--run-plan",
                "run-plan.json",
                "--intake",
                "intake.json",
                "--authoring-receipt",
                "receipts/authoring.json",
                "--output-dir",
                "projection",
            ),
        ),
        (
            "project-dynamic",
            "project_joint_dynamic_result",
            (
                "--source",
                "output.usdz",
                "--output",
                "output.usdz",
                "--receipt",
                "dynamic-receipt.json",
                "--profile-id",
                "joint.dynamic.profile.v1",
                "--artifact-map",
                "artifact-map.json",
                "--output-dir",
                "projection",
            ),
        ),
    ),
)
def test_articulation_projectors_are_public_focused_cli_leaves(
    command: str,
    function_name: str,
    arguments: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def projector(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append((args, kwargs))
        return SimpleNamespace(
            model_dump=lambda **_kwargs: {
                "schema_version": "test-joint-projector-publication.v1",
                "joint_inference_invoked": False,
            }
        )

    monkeypatch.setattr(
        f"content_agent_workflows.articulation.{function_name}",
        projector,
    )
    result = main(["articulation", command, *arguments])

    assert result == 0
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out)["joint_inference_invoked"] is False
