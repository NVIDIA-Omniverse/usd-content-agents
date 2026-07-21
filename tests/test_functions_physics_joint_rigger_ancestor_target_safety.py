# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for publication targets above bound USD inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from world_understanding.functions.physics.joint_rigger import (
    INPUT_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    FieldProvenanceV1,
    JointPlanV1,
    JointRiggerArtifactError,
    JointRiggerArtifactTargets,
    JointRiggerInputV1,
    JointRiggerPlanV1,
    JointTopologyV1,
    OwnedTopologyBackend,
    author_joint_rig,
    author_joint_topology,
    identify_usd_artifact,
)
from world_understanding.functions.physics.joint_rigger import author as author_module
from world_understanding.functions.physics.joint_rigger import facade as facade_module
from world_understanding.functions.physics.joint_rigger.artifacts import (
    validate_artifact_targets,
)

_SOURCE_USDA = """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Xform "Base" {}
    def Xform "Link" {}
}
"""


def _source_through_directory_symlink(tmp_path: Path) -> tuple[Path, Path, Path]:
    real_root = tmp_path / "real-source-root"
    real_root.mkdir()
    real_source = real_root / "source.usda"
    real_source.write_text(_SOURCE_USDA, encoding="utf-8")
    source_root_alias = tmp_path / "source-root.usda"
    source_root_alias.symlink_to(real_root, target_is_directory=True)
    return source_root_alias / real_source.name, source_root_alias, real_source


def _targets_with_ancestor(
    tmp_path: Path,
    *,
    target_field: str,
    ancestor: Path,
) -> tuple[JointRiggerArtifactTargets, dict[str, Path]]:
    target_values = {
        "output_path": tmp_path / "artifacts" / "rigged.usda",
        "diagnostics_path": tmp_path / "artifacts" / "diagnostics.json",
        "result_path": tmp_path / "artifacts" / "result.json",
    }
    target_values[target_field] = ancestor
    return JointRiggerArtifactTargets(**target_values), target_values


@pytest.mark.parametrize(
    "target_field",
    ["output_path", "diagnostics_path", "result_path"],
)
@pytest.mark.parametrize("read_coordinate", ["lexical", "resolved"])
def test_target_validator_rejects_file_target_above_read_path(
    tmp_path: Path,
    target_field: str,
    read_coordinate: str,
) -> None:
    source, source_root_alias, real_source = _source_through_directory_symlink(tmp_path)
    targets, _ = _targets_with_ancestor(
        tmp_path,
        target_field=target_field,
        ancestor=source_root_alias,
    )
    read_path = source if read_coordinate == "lexical" else real_source

    with pytest.raises(
        ValueError,
        match=rf"{target_field} must not be an ancestor of bound source USD",
    ):
        validate_artifact_targets(
            targets,
            read_paths=[("bound source USD", read_path)],
        )

    sibling_targets = JointRiggerArtifactTargets(
        output_path=tmp_path / "rigged.usda",
        diagnostics_path=tmp_path / "diagnostics.json",
        result_path=tmp_path / "result.json",
    )
    validate_artifact_targets(
        sibling_targets,
        read_paths=[("bound source USD", read_path)],
    )


@pytest.mark.parametrize(
    "target_field",
    ["output_path", "diagnostics_path", "result_path"],
)
@pytest.mark.parametrize("entrypoint", ["facade", "direct_backend"])
def test_owned_author_rejects_ancestor_symlink_target_for_remote_request(
    tmp_path: Path,
    target_field: str,
    entrypoint: str,
) -> None:
    pytest.importorskip("pxr")
    source, source_root_alias, real_source = _source_through_directory_symlink(tmp_path)
    source_bytes = real_source.read_bytes()
    identity = identify_usd_artifact(
        source,
        uri="https://assets.example.invalid/logical/source.usda",
    )
    topology = JointTopologyV1(
        joint_id="joint",
        joint_type="revolute",
        body0="/World/Base",
        body1="/World/Link",
        axis_stage=(0.0, 0.0, 1.0),
        field_provenance={
            field: FieldProvenanceV1(
                source="source_metadata",
                artifact=identity,
                prim_path="/World/Link",
                properties=(field,),
                evidence=f"Synthetic source evidence for {field}.",
            )
            for field in ("joint_type", "body0", "body1", "axis_stage")
        },
    )
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=identity,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(JointPlanV1(topology=topology),),
        ),
    )
    targets, target_values = _targets_with_ancestor(
        tmp_path,
        target_field=target_field,
        ancestor=source_root_alias,
    )
    backend = OwnedTopologyBackend(source)

    with pytest.raises(
        JointRiggerArtifactError,
        match=rf"{target_field} must not be an ancestor of bound source USD",
    ):
        if entrypoint == "facade":
            author_joint_rig(request, backend, targets)
        else:
            backend.author(request, targets)

    assert source_root_alias.is_symlink()
    assert source_root_alias.resolve() == real_source.parent
    assert real_source.read_bytes() == source_bytes
    for field, path in target_values.items():
        if field != target_field:
            assert not path.exists()
    artifact_root = tmp_path / "artifacts"
    assert not artifact_root.exists() or not any(artifact_root.rglob("*"))
    assert not any(tmp_path.rglob(".*.stage-*"))


@pytest.mark.parametrize(
    "move_timing",
    ["after_request_preflight", "after_author_binding", "after_source_identity"],
)
@pytest.mark.parametrize("entrypoint", ["facade", "owned-wrapper"])
def test_facade_preserves_captured_source_moved_onto_initially_absent_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    move_timing: str,
    entrypoint: str,
) -> None:
    """A byte-identical replacement must not hide destructive target drift."""

    pytest.importorskip("pxr")
    source_parent = tmp_path / "source"
    source_parent.mkdir()
    source = source_parent / "source.usda"
    source.write_text(_SOURCE_USDA, encoding="utf-8")
    source_bytes = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    identity = identify_usd_artifact(source, uri=str(source))
    topology = JointTopologyV1(
        joint_id="joint",
        joint_type="revolute",
        body0="/World/Base",
        body1="/World/Link",
        axis_stage=(0.0, 0.0, 1.0),
        field_provenance={
            field: FieldProvenanceV1(
                source="source_metadata",
                artifact=identity,
                prim_path="/World/Link",
                properties=(field,),
                evidence=f"Synthetic source evidence for {field}.",
            )
            for field in ("joint_type", "body0", "body1", "axis_stage")
        },
    )
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=identity,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(JointPlanV1(topology=topology),),
        ),
    )
    targets = JointRiggerArtifactTargets(
        output_path=tmp_path / "artifacts" / "rigged.usda",
        diagnostics_path=tmp_path / "artifacts" / "diagnostics.json",
        result_path=tmp_path / "artifacts" / "result.json",
    )
    moved = False

    def move_captured_source() -> None:
        nonlocal moved
        assert not moved
        assert targets.output_path.parent.is_dir()
        source.rename(targets.output_path)
        source.write_bytes(source_bytes)
        moved = True
        assert (
            targets.output_path.stat().st_dev,
            targets.output_path.stat().st_ino,
        ) == source_identity

    if move_timing == "after_request_preflight":
        real_preflight_request_inputs = facade_module._preflight_request_inputs

        def preflight_request_inputs_then_move(request_value: Any) -> Any:
            result = real_preflight_request_inputs(request_value)
            move_captured_source()
            return result

        monkeypatch.setattr(
            facade_module,
            "_preflight_request_inputs",
            preflight_request_inputs_then_move,
        )
    elif move_timing == "after_author_binding":
        real_create_binding = author_module.create_sealed_source_binding
        binding_calls = 0

        def create_binding_then_move(path: Path, *, expected: Any) -> Any:
            nonlocal binding_calls
            binding_calls += 1
            binding = real_create_binding(path, expected=expected)
            if binding_calls == 1:
                move_captured_source()
            return binding

        monkeypatch.setattr(
            author_module,
            "create_sealed_source_binding",
            create_binding_then_move,
        )
    else:
        real_require_source_identity = author_module._require_source_identity

        def require_source_identity_then_move(path: Path, request_value: Any) -> None:
            real_require_source_identity(path, request_value)
            move_captured_source()

        monkeypatch.setattr(
            author_module,
            "_require_source_identity",
            require_source_identity_then_move,
        )

    with pytest.raises(
        JointRiggerArtifactError,
        match="Artifact target changed after staged targets were created",
    ):
        if entrypoint == "facade":
            author_joint_rig(request, OwnedTopologyBackend(source), targets)
        else:
            author_joint_topology(
                request,
                source_usd_path=source,
                artifact_targets=targets,
            )

    assert moved
    assert source.read_bytes() == source_bytes
    assert (source.stat().st_dev, source.stat().st_ino) != source_identity
    assert targets.output_path.read_bytes() == source_bytes
    assert (
        targets.output_path.stat().st_dev,
        targets.output_path.stat().st_ino,
    ) == source_identity
    if move_timing == "after_author_binding":
        assert binding_calls == 1
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()
    assert not any(tmp_path.rglob(".*.stage-*"))
