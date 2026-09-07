# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import stat
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from world_understanding.functions.physics.joint_rigger import identify_usd_artifact

from content_agent_workflows.articulation import (
    ARTICULATION_AUTHOR_LEAF_ID,
    ARTICULATION_PREPARATION_READBACK_DRAFT_SCHEMA_VERSION,
    ArticulationAuthorLeafProgress,
    ArticulationFocusedLeafInvocation,
    ArticulationPreparationInspectionReadback,
    ArticulationPreparationInspectionReadbackDraft,
    ArticulationPreparationLeafResult,
    ArtifactJsonArticulationProposalProvider,
    EmbeddedArticulationError,
    EmbeddedArticulationPreparation,
    publish_embedded_articulation_preparation,
    request_embedded_articulation_provider_proposal,
    run_articulation_author_asset_leaf,
    validate_articulation_proposal_attempt_receipt,
    validate_embedded_articulation_preparation_publication,
)
from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_decision import (
    DomainProposalPayload,
)


def _claim(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    return {
        "relative_path": relative_path,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _binding(path: Path) -> ExecutionArtifactBinding:
    return ExecutionArtifactBinding(
        path=str(path.resolve()),
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _retained_readback(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    root = tmp_path / "retained"
    root.mkdir(parents=True)
    (root / "dep.usda").write_text(
        """#usda 1.0
def Xform "World"
{
    def Xform "Cabinet"
    {
        def Mesh "Drawer"
        {
        }
    }
}
""",
        encoding="utf-8",
    )
    (root / "source.usda").write_text(
        """#usda 1.0
(
    subLayers = [@dep.usda@]
)
""",
        encoding="utf-8",
    )
    capabilities = {
        "supported_joint_types": ["revolute", "prismatic"],
        "supported_frame_policies": ["body1_world_origin"],
        "fixed_joint_authoring_supported": False,
        "co_rigid_preservation_supported": True,
        "raw_predictions_authority": False,
        "provider_proposals_authority": False,
        "canonical_output_evidence_required": True,
    }
    memberships = [
        {
            "member_prim": "/World",
            "authoritative_owner_prim": "/World",
            "disposition": "independent_motion",
        },
        {
            "member_prim": "/World/Cabinet",
            "authoritative_owner_prim": "/World/Cabinet",
            "disposition": "independent_motion",
        },
        {
            "member_prim": "/World/Cabinet/Drawer",
            "authoritative_owner_prim": "/World/Cabinet/Drawer",
            "disposition": "independent_motion",
        },
    ]
    (root / "inspector-config.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.articulation-preparation-configuration.v1"
                ),
                "membership_policy": "retained-explicit-membership-v1",
                "memberships": memberships,
                "capabilities": capabilities,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "inspector.py").write_text(
        "INSPECTOR_CONTRACT = 'articulation-preparation-v1'\n",
        encoding="utf-8",
    )
    (root / "render.json").write_text(
        '{"camera":"front","source":"source.usda"}\n',
        encoding="utf-8",
    )
    (root / "scene.json").write_text(
        '{"default_prim":"/World","complete":true}\n',
        encoding="utf-8",
    )
    source = root / "source.usda"
    identity = identify_usd_artifact(source, uri=source.as_uri())
    assert identity.dependency_bundle_sha256 is not None
    payload: dict[str, Any] = {
        "inspector_id": "deterministic-usd-readback",
        "inspector_implementation": "example.saved_stage_inspector.v1",
        "source": _claim(root, "source.usda"),
        "dependencies": [_claim(root, "dep.usda")],
        "dependency_entry_count": 1,
        "dependency_closure_complete": True,
        "source_dependency_bundle_sha256": identity.dependency_bundle_sha256,
        "configuration": _claim(root, "inspector-config.json"),
        "inspector_implementation_artifact": _claim(root, "inspector.py"),
        "saved_stage": _claim(root, "source.usda"),
        "saved_stage_dependency_bundle_sha256": identity.dependency_bundle_sha256,
        "hierarchy": [
            {
                "prim_path": "/World",
                "parent_prim_path": None,
                "type_name": "Xform",
            },
            {
                "prim_path": "/World/Cabinet",
                "parent_prim_path": "/World",
                "type_name": "Xform",
            },
            {
                "prim_path": "/World/Cabinet/Drawer",
                "parent_prim_path": "/World/Cabinet",
                "type_name": "Mesh",
            },
        ],
        "memberships": memberships,
        "render_artifacts": [_claim(root, "render.json")],
        "scene_artifacts": [_claim(root, "scene.json")],
        "proposal_status": "not_requested",
        "readback_complete": True,
    }
    readback_path = root / "readback.json"
    readback_path.write_text(
        ArticulationPreparationInspectionReadback.model_validate(
            payload
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    return root, readback_path, payload


def _refresh_dependency_identity(root: Path, payload: dict[str, Any]) -> None:
    payload["dependencies"] = [_claim(root, "dep.usda")]
    source = root / "source.usda"
    identity = identify_usd_artifact(source, uri=source.as_uri())
    assert identity.dependency_bundle_sha256 is not None
    payload["source_dependency_bundle_sha256"] = identity.dependency_bundle_sha256
    payload["saved_stage_dependency_bundle_sha256"] = identity.dependency_bundle_sha256


def _replace_readback_source_with_self_contained_package(
    root: Path,
    payload: dict[str, Any],
) -> str:
    from pxr import UsdUtils

    package = root / "source.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(root / "source.usda"), str(package))
    with zipfile.ZipFile(package) as archive:
        members = set(archive.namelist())
        assert "source.usda" in members
        assert any(item.endswith("/dep.usda") for item in members)
    (root / "source.usda").unlink()
    (root / "dep.usda").unlink()
    payload["source"] = _claim(root, "source.usdz")
    payload["saved_stage"] = payload["source"]
    payload["dependencies"] = []
    payload["dependency_entry_count"] = 0
    identity = identify_usd_artifact(package, uri=package.as_uri())
    assert identity.dependency_bundle_sha256 is not None
    payload["source_dependency_bundle_sha256"] = identity.dependency_bundle_sha256
    payload["saved_stage_dependency_bundle_sha256"] = identity.dependency_bundle_sha256
    return identity.dependency_bundle_sha256


def test_preparation_readback_unknown_ancestor_is_typed_invalid_input(
    tmp_path: Path,
) -> None:
    retained, readback_path, payload = _retained_readback(tmp_path)
    payload["hierarchy"] = [payload["hierarchy"][2], payload["hierarchy"][1]]
    payload["memberships"] = [payload["memberships"][2], payload["memberships"][1]]
    readback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(EmbeddedArticulationError, match="unknown ancestor"):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "unknown-ancestor-publication",
        )


def test_preparation_rejects_saved_stage_larger_than_bounded_readback(
    tmp_path: Path,
) -> None:
    retained, readback_path, payload = _retained_readback(tmp_path)
    payload["hierarchy"].pop()
    payload["memberships"].pop()
    readback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(EmbeddedArticulationError, match="more prims"):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "bounded-hierarchy-publication",
        )


def test_preparation_hierarchy_rejects_cache_injected_after_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, readback_path, payload = _retained_readback(tmp_path)
    from pxr import Usd

    payload["hierarchy"].append(
        {
            "prim_path": "/CachedOnly",
            "parent_prim_path": None,
            "type_name": "Xform",
        }
    )
    cached_membership = {
        "member_prim": "/CachedOnly",
        "authoritative_owner_prim": "/CachedOnly",
        "disposition": "independent_motion",
    }
    payload["memberships"].append(cached_membership)
    configuration_path = retained / "inspector-config.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration["memberships"].append(cached_membership)
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    payload["configuration"] = _claim(retained, "inspector-config.json")
    readback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    from content_agent_workflows.articulation import preparation as preparation_module

    original_verify = preparation_module._verify_usd_identity
    cached_stages: list[Any] = []

    def inject_dirty_cache_after_identity(*args: Any, **kwargs: Any) -> str:
        dependency_identity = original_verify(*args, **kwargs)
        cached_stage = Usd.Stage.Open(str(retained / "source.usda"))
        assert cached_stage is not None
        cached_stage.DefinePrim("/CachedOnly", "Xform")
        cached_stages.append(cached_stage)
        return dependency_identity

    monkeypatch.setattr(
        preparation_module,
        "_verify_usd_identity",
        inject_dirty_cache_after_identity,
    )

    with pytest.raises(
        EmbeddedArticulationError,
        match=r"saved-stage hierarchy differs.*CachedOnly",
    ):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "cached-layer-publication",
        )


def test_preparation_publisher_rejects_instanceable_saved_stage(
    tmp_path: Path,
) -> None:
    retained, readback_path, payload = _retained_readback(tmp_path)
    (retained / "dep.usda").write_text(
        """#usda 1.0
def Xform "World"
{
    def Xform "Cabinet" (
        instanceable = true
    )
    {
        def Mesh "Drawer"
        {
        }
    }
}
""",
        encoding="utf-8",
    )
    _refresh_dependency_identity(retained, payload)
    readback_path.write_text(
        ArticulationPreparationInspectionReadback.model_validate(
            payload
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    with pytest.raises(EmbeddedArticulationError, match="instanceable prim"):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "instanceable-publication",
        )


def test_preparation_publisher_derives_six_records_and_is_deterministic(
    tmp_path: Path,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    first = publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=tmp_path / "publication-1",
    )
    second = publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=tmp_path / "publication-2",
    )

    first_bytes = Path(first.preparation.path).read_bytes()
    assert first_bytes == Path(second.preparation.path).read_bytes()
    preparation = EmbeddedArticulationPreparation.model_validate_json(first_bytes)
    assert tuple(item.evidence_id for item in preparation.evidence_records) == (
        "source-hierarchy-inspection",
        "joint-source-member-inspection",
        "joint-authoritative-owner-inspection",
        "joint-authoring-capabilities",
        "joint-render-inspection",
        "joint-scene-inspection",
    )
    assert preparation.source_members.facts["source_member_prims"] == (
        "/World",
        "/World/Cabinet",
        "/World/Cabinet/Drawer",
    )
    assert preparation.authoritative_owners.facts["authoritative_owner_prims"] == (
        "/World",
        "/World/Cabinet",
        "/World/Cabinet/Drawer",
    )
    assert preparation.capabilities.facts["canonical_output_evidence_required"] is True
    assert (tmp_path / "publication-1").stat().st_mode & 0o777 == 0o500
    retained_paths = {
        first.readback.path,
        first.source.path,
        *(item.path for item in first.dependencies),
        first.configuration.path,
        first.inspector_implementation.path,
        first.saved_stage.path,
        *(item.path for item in first.renders),
        *(item.path for item in first.scene_artifacts),
    }
    evidence_paths = {
        artifact.path
        for record in preparation.evidence_records
        for artifact in record.artifacts
    }
    assert retained_paths <= evidence_paths
    assert (
        validate_embedded_articulation_preparation_publication(
            tmp_path / "publication-1" / "articulation_preparation_publication.json"
        )
        == first
    )


def test_preparation_draft_lets_trusted_publisher_finalize_dependency_identity(
    tmp_path: Path,
) -> None:
    retained, readback_path, payload = _retained_readback(tmp_path)
    trusted_dependency_identity = _replace_readback_source_with_self_contained_package(
        retained,
        payload,
    )
    payload["schema_version"] = ARTICULATION_PREPARATION_READBACK_DRAFT_SCHEMA_VERSION
    payload.pop("source_dependency_bundle_sha256")
    payload.pop("saved_stage_dependency_bundle_sha256")
    draft = ArticulationPreparationInspectionReadbackDraft.model_validate(payload)
    readback_path.write_text(
        draft.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )
    readback_payload = json.loads(readback_path.read_text(encoding="utf-8"))
    assert "source_dependency_bundle_sha256" not in readback_payload
    assert "saved_stage_dependency_bundle_sha256" not in readback_payload

    publication = publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=tmp_path / "draft-publication",
    )
    asset_result = ArticulationPreparationLeafResult.model_validate(
        publication.model_dump(mode="json")
    )
    assert asset_result.root == publication
    assert asset_result.model_dump(mode="json") == publication.model_dump(mode="json")
    preparation = EmbeddedArticulationPreparation.model_validate_json(
        Path(publication.preparation.path).read_bytes()
    )
    source = retained / "source.usdz"
    trusted_identity = identify_usd_artifact(source, uri=source.as_uri())
    assert trusted_identity.dependency_bundle_sha256 is not None
    assert trusted_identity.dependency_bundle_sha256 == trusted_dependency_identity
    assert (
        preparation.source_dependency_bundle_sha256
        == trusted_identity.dependency_bundle_sha256
    )
    assert (
        preparation.scene.facts["source_dependency_bundle_sha256"]
        == trusted_identity.dependency_bundle_sha256
    )
    assert (
        preparation.scene.facts["saved_stage_dependency_bundle_sha256"]
        == trusted_identity.dependency_bundle_sha256
    )
    assert (
        validate_embedded_articulation_preparation_publication(
            tmp_path / "draft-publication" / "articulation_preparation_publication.json"
        )
        == publication
    )


def test_asset_author_leaf_accepts_published_canonical_output_requirement(
    tmp_path: Path,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    publication_dir = tmp_path / "publication"
    publication = publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=publication_dir,
    )
    attempt = tmp_path / "runs" / "asset-run" / "leaves" / "author" / "attempts" / "01"
    attempt.mkdir(parents=True)
    invocation = ArticulationFocusedLeafInvocation(
        leaf_id=ARTICULATION_AUTHOR_LEAF_ID,
        attempt_root=str(attempt),
        source=publication.source,
        preparation_publication=_binding(
            publication_dir / "articulation_preparation_publication.json"
        ),
        decision_patch_path=str(attempt / "articulation_decision_patch.json"),
        intent="Author only the exact accepted drawer articulation.",
    )
    invocation_path = attempt / "articulation_leaf_invocation.json"
    invocation_path.write_text(invocation.model_dump_json(indent=2), encoding="utf-8")

    result = run_articulation_author_asset_leaf(invocation_path)

    assert isinstance(result, ArticulationAuthorLeafProgress)
    assert result.native_status == "awaiting_decision"
    assert invocation.canonical_visual_envelope is None


def test_preparation_rejects_missing_canonical_output_requirement(
    tmp_path: Path,
) -> None:
    retained, readback_path, payload = _retained_readback(tmp_path)
    configuration_path = retained / "inspector-config.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration["capabilities"].pop("canonical_output_evidence_required")
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    payload["configuration"] = _claim(retained, "inspector-config.json")
    readback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(
        EmbeddedArticulationError,
        match="requires canonical output evidence",
    ):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "missing-output-evidence-publication",
        )


def test_preparation_publisher_rejects_selected_readback_substitution(
    tmp_path: Path,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    selected = _binding(readback_path)
    readback_path.write_text('{"substituted":true}\n', encoding="utf-8")

    with pytest.raises(
        EmbeddedArticulationError,
        match="differs from the selected-leaf invocation",
    ):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "publication",
            expected_readback=selected,
        )
    assert not (tmp_path / "publication").exists()


def test_preparation_publisher_self_validates_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    from content_agent_workflows.articulation import preparation as preparation_module

    original_validate = (
        preparation_module.validate_embedded_articulation_preparation_publication
    )
    validation_calls = 0

    def observe_validation(path: str | Path) -> Any:
        nonlocal validation_calls
        validation_calls += 1
        return original_validate(path)

    monkeypatch.setattr(
        preparation_module,
        "validate_embedded_articulation_preparation_publication",
        observe_validation,
    )
    publication = publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=tmp_path / "self-validated-publication",
    )

    assert validation_calls == 1
    assert publication.complete is True


def test_preparation_create_only_writer_removes_interrupted_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, readback_path, _ = _retained_readback(tmp_path)
    from content_agent_workflows.articulation import preparation as preparation_module

    payload = ArticulationPreparationInspectionReadback.model_validate_json(
        readback_path.read_bytes()
    )
    publication_root = tmp_path / "interrupted-write"
    publication_root.mkdir()
    original_write = preparation_module.os.write
    write_calls = 0

    def interrupt_after_one_byte(file_fd: int, data: Any) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return original_write(file_fd, data[:1])
        raise OSError("simulated interrupted preparation write")

    monkeypatch.setattr(preparation_module.os, "write", interrupt_after_one_byte)
    with pytest.raises(OSError, match="simulated interrupted"):
        preparation_module._write_json_create_only(
            publication_root,
            "partial.json",
            payload,
        )

    assert not (publication_root / "partial.json").exists()


def test_preparation_root_seals_through_held_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.articulation import preparation as preparation_module

    publication_root = tmp_path / "publication"
    publication_root.mkdir(mode=0o700)

    def reject_path_chmod(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("publication sealing must not chmod a path")

    monkeypatch.setattr(preparation_module.os, "chmod", reject_path_chmod)
    preparation_module._seal_publication_root(publication_root, set())
    assert stat.S_IMODE(publication_root.stat().st_mode) == 0o500


def test_sealed_preparation_root_validator_rejects_added_entry(
    tmp_path: Path,
) -> None:
    from content_agent_workflows.articulation import preparation as preparation_module

    publication_root = tmp_path / "publication"
    publication_root.mkdir(mode=0o700)
    preparation_module._seal_publication_root(publication_root, set())
    if os.name == "posix":
        publication_root.chmod(0o700)
    (publication_root / "unexpected.json").write_text("{}\n", encoding="utf-8")
    if os.name == "posix":
        publication_root.chmod(0o500)

    with pytest.raises(EmbeddedArticulationError, match="extra entries"):
        preparation_module._validate_sealed_publication_root(
            publication_root,
            set(),
        )


def test_preparation_publisher_accepts_exact_typeless_prim(tmp_path: Path) -> None:
    retained, readback_path, payload = _retained_readback(tmp_path)
    dependency = retained / "dep.usda"
    dependency.write_text(
        dependency.read_text(encoding="utf-8").replace(
            'def Xform "Cabinet"',
            'def "Cabinet"',
        ),
        encoding="utf-8",
    )
    _refresh_dependency_identity(retained, payload)
    payload["hierarchy"][1]["type_name"] = ""
    readback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    publication = publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=tmp_path / "typeless-publication",
    )

    assert (
        validate_embedded_articulation_preparation_publication(
            tmp_path
            / "typeless-publication"
            / "articulation_preparation_publication.json"
        )
        == publication
    )


def test_preparation_publisher_rejects_inactive_prim_claimed_active(
    tmp_path: Path,
) -> None:
    retained, readback_path, payload = _retained_readback(tmp_path)
    dependency = retained / "dep.usda"
    dependency.write_text(
        dependency.read_text(encoding="utf-8").replace(
            'def Xform "Cabinet"',
            'def Xform "Cabinet" (\n        active = false\n    )',
        ),
        encoding="utf-8",
    )
    _refresh_dependency_identity(retained, payload)
    readback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(
        EmbeddedArticulationError, match="saved-stage hierarchy differs"
    ):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "inactive-publication",
        )


def test_published_preparation_composes_with_proposal_leaf_from_sealed_root(
    tmp_path: Path,
) -> None:
    retained, readback_path, payload = _retained_readback(tmp_path)
    payload["proposal_status"] = "not_evaluated"
    readback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    preparation_publication = publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=tmp_path / "preparation-publication",
    )
    provider_payload = tmp_path / "provider-payload.json"
    provider_payload.write_text(
        DomainProposalPayload(
            schema_version="composition-test.example.v1",
            values={"candidate_hints": [{"id": "candidate-1"}]},
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    attempt = request_embedded_articulation_provider_proposal(
        preparation_publication.preparation.path,
        output_dir=tmp_path / "proposal-attempt",
        intent="Exercise the exact public preparation-to-proposal composition.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="composition-test-provider",
            capability_id="articulation-proposal-v1",
            payload_path=provider_payload,
        ),
    )

    terminal = validate_articulation_proposal_attempt_receipt(
        attempt.terminal_receipt.path
    )
    assert terminal.disposition == "succeeded"
    assert terminal.preparation == preparation_publication.preparation


def test_publisher_reserves_generated_document_headroom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    from content_agent_workflows.articulation import preparation as preparation_module

    input_limit = readback_path.stat().st_size + 1
    monkeypatch.setattr(
        preparation_module,
        "_MAX_ARTICULATION_INPUT_JSON_BYTES",
        input_limit,
    )
    publication = publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=tmp_path / "headroom-publication",
    )

    assert publication.preparation.size_bytes > input_limit
    assert (
        validate_embedded_articulation_preparation_publication(
            tmp_path
            / "headroom-publication"
            / "articulation_preparation_publication.json"
        )
        == publication
    )


def test_publisher_rejects_oversized_derived_document_before_root_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    from content_agent_workflows.articulation import preparation as preparation_module

    monkeypatch.setattr(
        preparation_module,
        "_MAX_ARTICULATION_GENERATED_JSON_BYTES",
        1,
    )
    output_dir = tmp_path / "oversized-publication"
    with pytest.raises(EmbeddedArticulationError, match="bounded limit"):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=output_dir,
        )
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda _root, payload: payload.update(
                {"dependencies": [], "dependency_entry_count": 0}
            ),
            "dependency closure is incomplete",
        ),
        (
            lambda _root, payload: payload.update(
                {"source_members": ["/World/Fabricated"]}
            ),
            "Extra inputs are not permitted",
        ),
        (
            lambda _root, payload: payload["memberships"].pop(),
            "must exactly cover the hierarchy prims",
        ),
        (
            lambda _root, payload: payload["memberships"][2].update(
                {"authoritative_owner_prim": "/World/Fabricated"}
            ),
            "must be rooted at the member|must contain their members|unknown owner",
        ),
        (
            lambda _root, payload: payload["memberships"][2].update(
                {
                    "authoritative_owner_prim": "/World/Cabinet",
                    "disposition": "co_rigid",
                }
            ),
            "differs from retained configuration authority",
        ),
        (
            lambda _root, payload: (
                payload["hierarchy"][2].update(
                    {"prim_path": "/World/Cabinet/Fabricated"}
                ),
                payload["memberships"][2].update(
                    {
                        "member_prim": "/World/Cabinet/Fabricated",
                        "authoritative_owner_prim": "/World/Cabinet/Fabricated",
                    }
                ),
            ),
            "saved-stage hierarchy differs",
        ),
        (
            lambda _root, payload: payload["hierarchy"][1].update(
                {"parent_prim_path": "/World/Cabinet/Drawer"}
            ),
            "lexical USD parent",
        ),
        (
            lambda _root, payload: (
                payload["hierarchy"][0].update({"prim_path": "/Wörld"}),
                payload["memberships"][0].update(
                    {
                        "member_prim": "/Wörld",
                        "authoritative_owner_prim": "/Wörld",
                    }
                ),
            ),
            "canonical absolute prim paths",
        ),
        (
            lambda _root, payload: payload["render_artifacts"][0].update(
                {"relative_path": "../outside.json"}
            ),
            "canonical relative paths",
        ),
    ],
)
def test_preparation_readback_rejects_fabricated_or_incomplete_authority(
    tmp_path: Path,
    mutate: Any,
    match: str,
) -> None:
    retained, readback_path, payload = _retained_readback(tmp_path)
    mutate(retained, payload)
    readback_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(EmbeddedArticulationError, match=match):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "rejected",
        )


def test_preparation_rejects_transitively_owned_authoritative_owner(
    tmp_path: Path,
) -> None:
    retained, readback_path, payload = _retained_readback(tmp_path)
    memberships = [
        {
            "member_prim": "/World",
            "authoritative_owner_prim": "/World",
            "disposition": "independent_motion",
        },
        {
            "member_prim": "/World/Cabinet",
            "authoritative_owner_prim": "/World",
            "disposition": "co_rigid",
        },
        {
            "member_prim": "/World/Cabinet/Drawer",
            "authoritative_owner_prim": "/World/Cabinet",
            "disposition": "co_rigid",
        },
    ]
    configuration_path = retained / "inspector-config.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration["memberships"] = memberships
    configuration_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    payload["configuration"] = _claim(retained, "inspector-config.json")
    payload["memberships"] = memberships
    readback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(EmbeddedArticulationError, match="transitively owned"):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "transitive-owner-publication",
        )


@pytest.mark.parametrize("entry_kind", ["symlink", "hardlink", "fifo"])
def test_preparation_publisher_rejects_unsafe_retained_entries(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    render = retained / "render.json"
    if entry_kind == "symlink":
        target = retained / "render-target.json"
        render.rename(target)
        render.symlink_to(target.name)
    elif entry_kind == "hardlink":
        os.link(render, retained / "render-alias.json")
    else:
        render.unlink()
        os.mkfifo(render)

    with pytest.raises(EmbeddedArticulationError, match="unsafe"):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "unsafe",
        )


def test_preparation_publisher_rejects_substitution_drift_and_reused_output(
    tmp_path: Path,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    (retained / "inspector-config.json").write_text(
        '{"membership_policy":"substituted"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(EmbeddedArticulationError, match="differs from saved readback"):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "substitution",
        )

    retained, readback_path, _ = _retained_readback(tmp_path / "fresh")
    with pytest.raises(EmbeddedArticulationError, match="create-only"):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=output,
        )


def test_usd_identity_mismatch_does_not_expose_dependency_or_source_digests(
    tmp_path: Path,
) -> None:
    retained, _readback_path, payload = _retained_readback(tmp_path)
    source = retained / "source.usda"
    dependency = retained / "dep.usda"
    expected_source = payload["source"]["sha256"]
    expected_dependency = payload["source_dependency_bundle_sha256"]
    source.write_text(
        source.read_text(encoding="utf-8") + "\n# identity drift\n",
        encoding="utf-8",
    )
    observed = identify_usd_artifact(source, uri=source.as_uri())
    assert observed.dependency_bundle_sha256 is not None

    from content_agent_workflows.articulation import preparation as preparation_module

    with pytest.raises(EmbeddedArticulationError) as failure:
        preparation_module._verify_usd_identity(
            source,
            expected_source_sha256=expected_source,
            expected_dependency_sha256=expected_dependency,
            expected_retained_paths=(source, dependency),
            label="fixture source",
        )

    message = str(failure.value)
    assert message == (
        "fixture source source or dependency closure differs from saved readback"
    )
    assert expected_source not in message
    assert observed.root_sha256 not in message
    assert expected_dependency not in message
    assert observed.dependency_bundle_sha256 not in message


def test_preparation_publication_rejects_post_publish_mutation(
    tmp_path: Path,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    publication = publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=tmp_path / "publication",
    )
    (tmp_path / "publication").chmod(0o700)
    (tmp_path / "publication" / "unexpected.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(EmbeddedArticulationError, match="mutable or extra"):
        validate_embedded_articulation_preparation_publication(
            Path(publication.preparation.path).parent
            / "articulation_preparation_publication.json"
        )


def test_preparation_validator_rejects_tampered_closure_digest(
    tmp_path: Path,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    publication_root = tmp_path / "publication"
    publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=publication_root,
    )
    publication_path = publication_root / "articulation_preparation_publication.json"
    publication_root.chmod(0o700)
    publication_path.chmod(0o644)
    payload = json.loads(publication_path.read_text(encoding="utf-8"))
    payload["retained_closure_digest"] = "0" * 64
    publication_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    publication_path.chmod(0o444)
    publication_root.chmod(0o500)

    with pytest.raises(EmbeddedArticulationError, match="retained closure changed"):
        validate_embedded_articulation_preparation_publication(publication_path)


def test_preparation_validator_rechecks_retained_inputs_at_final_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    publication_root = tmp_path / "publication"
    publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=publication_root,
    )
    from content_agent_workflows.articulation import preparation as preparation_module

    original_verify = preparation_module._verify_saved_stage_hierarchy
    mutated = False

    def mutate_render_after_hierarchy(*args: Any, **kwargs: Any) -> None:
        nonlocal mutated
        original_verify(*args, **kwargs)
        if not mutated:
            mutated = True
            (retained / "render.json").write_text(
                '{"camera":"substituted"}\n',
                encoding="utf-8",
            )

    monkeypatch.setattr(
        preparation_module,
        "_verify_saved_stage_hierarchy",
        mutate_render_after_hierarchy,
    )
    with pytest.raises(EmbeddedArticulationError, match="changed during validation"):
        validate_embedded_articulation_preparation_publication(
            publication_root / "articulation_preparation_publication.json"
        )


def test_preparation_publication_rejects_mutable_mode_and_symlinked_root(
    tmp_path: Path,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path / "mutable")
    publication = publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=tmp_path / "publication",
    )
    publication_root = Path(publication.preparation.path).parent
    publication_root.chmod(0o700)
    with pytest.raises(EmbeddedArticulationError, match="not sealed"):
        validate_embedded_articulation_preparation_publication(
            publication_root / "articulation_preparation_publication.json"
        )

    real_parent = tmp_path / "real-parent"
    retained, readback_path, _ = _retained_readback(real_parent)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(EmbeddedArticulationError, match="symlink-free"):
        publish_embedded_articulation_preparation(
            linked_parent / retained.name / readback_path.name,
            retained_root=linked_parent / retained.name,
            output_dir=tmp_path / "symlinked-root-output",
        )


def test_preparation_root_rechecks_path_after_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.articulation import preparation as preparation_module

    output = tmp_path / "publication"
    substituted = tmp_path / "substituted" / "publication"
    original_resolve = Path.resolve

    def race_resolve(path: Path, strict: bool = False) -> Path:
        if path == output and output.exists():
            return substituted
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", race_resolve)

    with pytest.raises(EmbeddedArticulationError, match="unsafe after creation"):
        preparation_module._fresh_publication_root(output)


def test_preparation_validator_rejects_symlinked_publication_root(
    tmp_path: Path,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    real_publication = tmp_path / "real-publication"
    publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=real_publication,
    )
    linked_publication = tmp_path / "linked-publication"
    linked_publication.symlink_to(real_publication, target_is_directory=True)

    with pytest.raises(EmbeddedArticulationError, match="traverses a symlink"):
        validate_embedded_articulation_preparation_publication(
            linked_publication / "articulation_preparation_publication.json"
        )


def test_preparation_validator_rejects_symlinked_retained_root(
    tmp_path: Path,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    publication = publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=tmp_path / "publication",
    )
    moved_retained = tmp_path / "moved-retained"
    retained.rename(moved_retained)
    retained.symlink_to(moved_retained, target_is_directory=True)

    with pytest.raises(EmbeddedArticulationError, match="symlink-free directory"):
        validate_embedded_articulation_preparation_publication(
            Path(publication.preparation.path).parent
            / "articulation_preparation_publication.json"
        )


def test_preparation_validator_rejects_readback_role_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    publication = publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=tmp_path / "publication",
    )
    from content_agent_workflows.articulation import preparation as preparation_module

    original_capture = preparation_module._capture_readback

    def alias_readback_role(*args: Any, **kwargs: Any) -> Any:
        _, readback = original_capture(*args, **kwargs)
        return publication.configuration, readback

    monkeypatch.setattr(
        preparation_module,
        "_capture_readback",
        alias_readback_role,
    )
    with pytest.raises(EmbeddedArticulationError, match="must not alias"):
        validate_embedded_articulation_preparation_publication(
            Path(publication.preparation.path).parent
            / "articulation_preparation_publication.json"
        )


def test_preparation_publisher_detects_readback_drift_during_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, readback_path, _ = _retained_readback(tmp_path)
    from content_agent_workflows.articulation import preparation as preparation_module

    original = preparation_module._verify_usd_identity
    calls = 0

    def mutate_once(*args: Any, **kwargs: Any) -> str:
        nonlocal calls
        dependency_identity = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            readback_path.write_text(
                readback_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
        return dependency_identity

    monkeypatch.setattr(preparation_module, "_verify_usd_identity", mutate_once)
    with pytest.raises(EmbeddedArticulationError, match="changed during preparation"):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "drift",
        )


def test_preparation_derives_co_rigid_owner_dedup_and_distinct_saved_stage(
    tmp_path: Path,
) -> None:
    retained, readback_path, payload = _retained_readback(tmp_path)
    (retained / "saved-stage.usda").write_text(
        """#usda 1.0
(
    subLayers = [@dep.usda@]
)
# saved-stage readback
""",
        encoding="utf-8",
    )
    saved_identity = identify_usd_artifact(
        retained / "saved-stage.usda",
        uri=(retained / "saved-stage.usda").as_uri(),
    )
    assert saved_identity.dependency_bundle_sha256 is not None
    payload["saved_stage"] = _claim(retained, "saved-stage.usda")
    payload["saved_stage_dependency_bundle_sha256"] = (
        saved_identity.dependency_bundle_sha256
    )
    payload["memberships"][2] = {
        "member_prim": "/World/Cabinet/Drawer",
        "authoritative_owner_prim": "/World/Cabinet",
        "disposition": "co_rigid",
    }
    configuration_path = retained / "inspector-config.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration["memberships"][2] = payload["memberships"][2]
    configuration_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    payload["configuration"] = _claim(retained, "inspector-config.json")
    readback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    publication = publish_embedded_articulation_preparation(
        readback_path,
        retained_root=retained,
        output_dir=tmp_path / "distinct-stage",
    )
    preparation = EmbeddedArticulationPreparation.model_validate_json(
        Path(publication.preparation.path).read_bytes()
    )
    assert preparation.authoritative_owners.facts["authoritative_owner_prims"] == (
        "/World",
        "/World/Cabinet",
    )
    rows = preparation.authoritative_owners.facts["membership_rows"]
    assert isinstance(rows, Sequence) and not isinstance(rows, str | bytes)
    assert isinstance(rows[-1], Mapping)
    assert rows[-1]["disposition"] == "co_rigid"


def test_preparation_rejects_distinct_saved_stage_hierarchy_drift(
    tmp_path: Path,
) -> None:
    retained, readback_path, payload = _retained_readback(tmp_path)
    saved_stage_path = retained / "saved-stage.usda"
    saved_stage_path.write_text(
        """#usda 1.0
(
    subLayers = [@dep.usda@]
)
def Xform "Unexpected"
{
}
""",
        encoding="utf-8",
    )
    saved_identity = identify_usd_artifact(
        saved_stage_path,
        uri=saved_stage_path.as_uri(),
    )
    assert saved_identity.dependency_bundle_sha256 is not None
    payload["saved_stage"] = _claim(retained, saved_stage_path.name)
    payload["saved_stage_dependency_bundle_sha256"] = (
        saved_identity.dependency_bundle_sha256
    )
    readback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(
        EmbeddedArticulationError,
        match=r"saved stage contains more prims than the bounded readback",
    ):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "distinct-stage-drift-publication",
        )


def test_preparation_rejects_alias_mismatch_readback_alias_and_bundle_mismatch(
    tmp_path: Path,
) -> None:
    retained, readback_path, payload = _retained_readback(tmp_path / "alias-stage")
    payload["saved_stage"]["size_bytes"] += 1
    readback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(EmbeddedArticulationError, match="aliased saved stage"):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "alias-stage-output",
        )

    retained, readback_path, payload = _retained_readback(tmp_path / "readback-alias")
    payload["render_artifacts"] = [
        {
            "relative_path": "readback.json",
            "sha256": "0" * 64,
            "size_bytes": 0,
        }
    ]
    readback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(EmbeddedArticulationError, match="must not alias"):
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "readback-alias-output",
        )

    retained, readback_path, payload = _retained_readback(tmp_path / "bundle")
    actual_dependency = _replace_readback_source_with_self_contained_package(
        retained,
        payload,
    )
    payload["source_dependency_bundle_sha256"] = "0" * 64
    payload["saved_stage_dependency_bundle_sha256"] = "0" * 64
    readback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(EmbeddedArticulationError) as failure:
        publish_embedded_articulation_preparation(
            readback_path,
            retained_root=retained,
            output_dir=tmp_path / "bundle-output",
        )
    message = str(failure.value)
    assert message == "source source or dependency closure differs from saved readback"
    assert "0" * 64 not in message
    assert actual_dependency not in message
