# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the app-agnostic Joint Rigger execution facade."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

from world_understanding.functions.physics.joint_rigger import (
    DIAGNOSTICS_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION_V2,
    PLAN_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION_V2,
    RESULT_SCHEMA_VERSION,
    ArticulationRootPlanV1,
    ArtifactIdentityV1,
    ColliderPlanV1,
    FieldDecisionV1,
    FieldProvenanceV1,
    JointAnchorV1,
    JointDiagnosticV1,
    JointDriveV1,
    JointFrictionV1,
    JointLimitV1,
    JointMimicV1,
    JointPlanV1,
    JointRiggerArtifactError,
    JointRiggerArtifactTargets,
    JointRiggerBackendIncompatibleError,
    JointRiggerBackendUnavailableError,
    JointRiggerContractError,
    JointRiggerDiagnosticsV1,
    JointRiggerInputV1,
    JointRiggerInputV2,
    JointRiggerPlanV1,
    JointRiggerPlanV2,
    JointRiggerPostCommitCleanupError,
    JointRiggerResultV1,
    JointStateV1,
    JointTopologyV1,
    LegacyComponentAssignmentV1,
    LegacyComponentNameCompatibilityV1,
    MassPropertiesV1,
    RigidBodyPlanV1,
    RigidLinkMemberPlanV1,
    RigidLinkPlanV1,
    author_joint_rig,
    facade,
    identify_usd_artifact,
    sidecar_dependency_bundle_sha256,
)
from world_understanding.functions.physics.joint_rigger import (
    artifacts as artifacts_module,
)
from world_understanding.functions.physics.joint_rigger import (
    reference as reference_module,
)
from world_understanding.functions.physics.joint_rigger import (
    source_binding as source_binding_module,
)
from world_understanding.functions.physics.joint_rigger.artifacts import (
    CommittedArtifactPublicationCleanupError,
    StagedArtifact,
    StagedJointRiggerArtifacts,
)
from world_understanding.functions.physics.joint_rigger.models import (
    canonical_json,
    canonical_sha256,
)

_EMPTY_USDA = '#usda 1.0\n\ndef Xform "Generated" {}\n'
_DEPENDENCY_USDA = '#usda 1.0\n\ndef Xform "Dependency" {}\n'
_AUTO_BUNDLE = object()
_MOUNT_CAPABILITY_DENIAL_MARKERS = (
    "operation not permitted",
    "permission denied",
    "unprivileged user namespaces are disabled",
    "user namespaces are not enabled",
)


def _mount_capability_is_unavailable(
    completed: subprocess.CompletedProcess[str],
) -> bool:
    detail = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    ).lower()
    return (
        completed.returncode != 0
        and ("unshare:" in detail or "mount:" in detail)
        and any(marker in detail for marker in _MOUNT_CAPABILITY_DENIAL_MARKERS)
    )


@pytest.mark.parametrize(
    ("returncode", "stderr", "expected"),
    [
        (1, "unshare: cannot change propagation: Permission denied", True),
        (32, "mount: /: Operation not permitted", True),
        (1, "python: cannot read fixture: Permission denied", False),
        (1, "mount: invalid option", False),
        (0, "unshare: Operation not permitted", False),
    ],
)
def test_mount_capability_classifier_accepts_only_tool_denials(
    returncode: int,
    stderr: str,
    expected: bool,
) -> None:
    completed = subprocess.CompletedProcess[str]([], returncode, "", stderr)

    assert _mount_capability_is_unavailable(completed) is expected


def test_integrated_probe_marker_must_be_declared_on_exact_backend_class() -> None:
    class IntegratedBase:
        author_runs_probe_checks = True

        def __init__(self) -> None:
            self.probe_calls = 0

        def probe(self, request: Any) -> None:
            self.probe_calls += 1

        def author(self, request: Any, artifact_targets: Any) -> None:
            raise AssertionError("author is not called by the probe helper")

    class OverridingBackend(IntegratedBase):
        def author(self, request: Any, artifact_targets: Any) -> None:
            raise AssertionError("author is not called by the probe helper")

    request: Any = object()
    exact_backend = IntegratedBase()
    facade._probe_backend(exact_backend, request)
    assert exact_backend.probe_calls == 0

    overriding_backend = OverridingBackend()
    facade._probe_backend(overriding_backend, request)
    assert overriding_backend.probe_calls == 1


def test_facade_requires_explicit_v2_and_aggregate_backend_capabilities() -> None:
    class BackendWithoutV2Support:
        def __init__(self) -> None:
            self.probe_calls = 0

        def probe(self, request: Any) -> None:
            self.probe_calls += 1

        def author(self, request: Any, artifact_targets: Any) -> None:
            raise AssertionError("author is not called by the probe helper")

    provenance = FieldProvenanceV1(
        source="owner_approved_plan",
        evidence="The owner approved this aggregate topology.",
    )
    request = JointRiggerInputV2(
        schema_version=INPUT_SCHEMA_VERSION_V2,
        source_asset=ArtifactIdentityV1(
            uri="s3://example/source.usda",
            root_sha256="a" * 64,
        ),
        plan=JointRiggerPlanV2(
            schema_version=PLAN_SCHEMA_VERSION_V2,
            joints=(
                JointPlanV1(
                    topology=JointTopologyV1(
                        joint_id="drawer_slide",
                        joint_type="prismatic",
                        body0="/World/base",
                        body1="/World/drawer",
                        axis_stage=(0.0, 0.0, 1.0),
                        field_provenance=dict.fromkeys(
                            ("joint_type", "body0", "body1", "axis_stage"),
                            provenance,
                        ),
                    )
                ),
            ),
            articulation_roots=(
                ArticulationRootPlanV1(
                    prim_path="/World/base",
                    provenance=provenance,
                ),
            ),
        ),
        rigid_links=(
            RigidLinkPlanV1(
                link_id="base",
                body_authoring="existing",
                body_prim_path="/World/base",
                members=(
                    RigidLinkMemberPlanV1(
                        source_prim_path="/World/base",
                        authored_prim_path="/World/base",
                    ),
                ),
            ),
            RigidLinkPlanV1(
                link_id="drawer",
                body_authoring="aggregate",
                body_prim_path="/World/drawer",
                members=tuple(
                    RigidLinkMemberPlanV1(
                        source_prim_path=f"/World/panel_{suffix}",
                        authored_prim_path=f"/World/drawer/panel_{suffix}",
                    )
                    for suffix in ("a", "b")
                ),
            ),
        ),
    )
    backend = BackendWithoutV2Support()

    with pytest.raises(
        JointRiggerBackendIncompatibleError,
        match="does not support JointRiggerInputV2",
    ):
        facade._probe_backend(backend, request)

    assert backend.probe_calls == 0

    class BackendWithoutAggregateSupport(BackendWithoutV2Support):
        supports_joint_rigger_input_v2 = True

    aggregate_backend = BackendWithoutAggregateSupport()
    with pytest.raises(
        JointRiggerBackendIncompatibleError,
        match="does not support V2 aggregate rigid-link authoring",
    ):
        facade._probe_backend(aggregate_backend, request)
    assert aggregate_backend.probe_calls == 0

    class V2CapableBase(BackendWithoutV2Support):
        supports_joint_rigger_input_v2 = True
        supports_aggregate_rigid_links = True

    class InheritedV2Backend(V2CapableBase):
        pass

    inherited_backend = InheritedV2Backend()
    facade._probe_backend(inherited_backend, request)
    assert inherited_backend.probe_calls == 1


def test_mdl_comment_stripping_preserves_strings_and_generic_resources(
    tmp_path: Path,
) -> None:
    document = tmp_path / "Main.mdl"
    text = (
        'mdl 1.7;\nstring quoted = "literal \\" // marker"; // comment\n'
        "/* block\ncomment */\n"
        'string resource = "textures/albedo.png";\n'
        'string repeated_resource = "textures/albedo.png";\n'
    )

    stripped = facade._strip_mdl_comments(text, document=document)

    assert '"literal \\" // marker"' in stripped
    assert "// comment" not in stripped
    assert "block" not in stripped
    assert facade._mdl_local_references(
        stripped,
        document=document,
    ) == ("textures/albedo.png",)


def test_opaque_mdl_dependency_closure_is_recursive_and_self_contained(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "generated_assets"
    sidecar.mkdir()
    main = sidecar / "Main.mdl"
    peer = sidecar / "Peer.mdl"
    texture = sidecar / "albedo.png"
    main.write_text(
        "mdl 1.7;\nimport ::df::*;\nimport Peer::*;\n",
        encoding="utf-8",
    )

    with pytest.raises(JointRiggerArtifactError, match="missing"):
        facade._validate_opaque_dependency_closure([main], allowed_root=sidecar)

    peer.write_text(
        'mdl 1.7;\nimport ::tex::*;\ntexture_2d("albedo.png");\n',
        encoding="utf-8",
    )
    with pytest.raises(JointRiggerArtifactError, match="missing"):
        facade._validate_opaque_dependency_closure([main], allowed_root=sidecar)

    texture.write_bytes(b"real texture bytes")
    facade._validate_opaque_dependency_closure([main], allowed_root=sidecar)

    peer.write_text(
        "mdl 1.7;\nimport ::unpackaged_vendor::Material::*;\n",
        encoding="utf-8",
    )
    with pytest.raises(JointRiggerArtifactError, match="unapproved runtime module"):
        facade._validate_opaque_dependency_closure([main], allowed_root=sidecar)


@pytest.mark.parametrize("boundary", ["root", "files"])
def test_opaque_mdl_closure_accepts_bounded_sibling_resource(
    tmp_path: Path,
    boundary: str,
) -> None:
    sidecar = tmp_path / "generated_assets"
    materials = sidecar / "materials"
    textures = sidecar / "textures"
    materials.mkdir(parents=True)
    textures.mkdir()
    main = materials / "Main.mdl"
    texture = textures / "albedo.png"
    main.write_text(
        "mdl 1.7;\n"
        "using ::OmniPBR import OmniPBR;\n"
        'texture_2d("../textures/albedo.png");\n',
        encoding="utf-8",
    )
    texture.write_bytes(b"real sibling texture bytes")

    if boundary == "root":
        facade._validate_opaque_dependency_closure(
            [main],
            allowed_root=sidecar,
        )
    else:
        facade._validate_opaque_dependency_closure(
            [main, texture],
            allowed_files={main, texture},
        )


def test_opaque_materialx_dependency_closure_rejects_missing_and_external_paths(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "generated_assets"
    textures = sidecar / "textures"
    textures.mkdir(parents=True)
    material = sidecar / "surface.mtlx"
    material.write_text(
        '<materialx><input name="file" type="filename" '
        'value="textures/albedo.png" /></materialx>',
        encoding="utf-8",
    )

    with pytest.raises(JointRiggerArtifactError, match="missing"):
        facade._validate_opaque_dependency_closure([material], allowed_root=sidecar)

    (textures / "albedo.png").write_bytes(b"real texture bytes")
    facade._validate_opaque_dependency_closure([material], allowed_root=sidecar)

    material.write_text(
        '<materialx><xi:include xmlns:xi="http://www.w3.org/2001/XInclude" '
        'href="https://example.invalid/library.mtlx" /></materialx>',
        encoding="utf-8",
    )
    with pytest.raises(JointRiggerArtifactError, match="external or ambiguous"):
        facade._validate_opaque_dependency_closure([material], allowed_root=sidecar)


def test_no_sidecar_opaque_closure_requires_every_file_in_artifact_identity(
    tmp_path: Path,
) -> None:
    main = tmp_path / "Main.mdl"
    peer = tmp_path / "Peer.mdl"
    main.write_text("mdl 1.7;\nimport Peer::*;\n", encoding="utf-8")
    peer.write_text("mdl 1.7;\nimport ::df::*;\n", encoding="utf-8")

    with pytest.raises(JointRiggerArtifactError, match="artifact identity"):
        facade._validate_opaque_dependency_closure(
            [main],
            allowed_files={main},
        )

    facade._validate_opaque_dependency_closure(
        [main, peer],
        allowed_files={main, peer},
    )


def _write_empty_usda(path: Path) -> None:
    path.write_text("#usda 1.0\n", encoding="utf-8")


def _output_dependency_sha256(output_text: str) -> str:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "output.usda"
        path.write_text(output_text, encoding="utf-8")
        digest = identify_usd_artifact(
            path,
            uri="memory://generated.usda",
        ).dependency_bundle_sha256
    assert digest is not None
    return digest


def _publication_sidecar_root_text(targets: JointRiggerArtifactTargets) -> str:
    publication_output = targets.publication_output_path
    publication_sidecar = targets.publication_sidecar_path
    assert publication_output is not None
    assert publication_sidecar is not None
    relative_sidecar = Path(
        os.path.relpath(publication_sidecar, start=publication_output.parent)
    ).as_posix()
    return _root_with_reference(f"{relative_sidecar}/dependency.usda")


def _root_with_reference(reference: str) -> str:
    return f"#usda 1.0\n(\n    subLayers = [\n        @{reference}@\n    ]\n)\n"


def _root_with_asset_reference(reference: str) -> str:
    return (
        '#usda 1.0\n\ndef Xform "Generated"\n{\n'
        f"    custom asset dependency = @{reference}@\n"
        "}\n"
    )


def _request(
    source_path: Path,
    *,
    legacy_component_names: LegacyComponentNameCompatibilityV1 | None = None,
    bind_usd_dependencies: bool = True,
) -> JointRiggerInputV1:
    source = (
        identify_usd_artifact(source_path, uri=str(source_path))
        if bind_usd_dependencies
        and source_path.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}
        else ArtifactIdentityV1(
            uri=str(source_path),
            root_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        )
    )
    return JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=source,
        plan=JointRiggerPlanV1(schema_version=PLAN_SCHEMA_VERSION, joints=()),
        legacy_component_names=legacy_component_names,
    )


def _request_with_reference(
    source_path: Path,
    reference_path: Path,
    *,
    bind_reference_dependencies: bool = True,
) -> JointRiggerInputV1:
    reference = (
        identify_usd_artifact(reference_path, uri=reference_path.as_uri())
        if bind_reference_dependencies
        else ArtifactIdentityV1(
            uri=reference_path.as_uri(),
            root_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        )
    )
    provenance = {
        field: FieldProvenanceV1(
            source="authored_reference",
            artifact=reference,
            prim_path="/World/Joint",
            properties=(f"physics:{field}",),
            evidence=f"Reference evidence for {field}.",
        )
        for field in ("joint_type", "body0", "body1", "axis_stage")
    }
    source = identify_usd_artifact(source_path, uri=str(source_path))
    return JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=source,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(
                JointPlanV1(
                    topology=JointTopologyV1(
                        joint_id="joint",
                        joint_type="revolute",
                        body0="/World/Base",
                        body1="/World/Link",
                        axis_stage=(0.0, 0.0, 1.0),
                        field_provenance=provenance,
                    )
                ),
            ),
        ),
    )


def _planned_request_with_every_fact(source_path: Path) -> JointRiggerInputV1:
    def provenance(label: str) -> FieldProvenanceV1:
        return FieldProvenanceV1(
            source="owner_approved_plan",
            evidence=f"Owner-approved {label} fixture fact.",
        )

    hinge_topology = JointTopologyV1(
        joint_id="hinge",
        joint_type="revolute",
        body0="/World/Base",
        body1="/World/Link",
        axis_stage=(0.0, 0.0, 1.0),
        field_provenance={
            field: provenance(f"hinge topology {field}")
            for field in ("joint_type", "body0", "body1", "axis_stage")
        },
    )
    follower_topology = JointTopologyV1(
        joint_id="follower",
        joint_type="revolute",
        body0="/World/Base",
        body1="/World/Follower",
        axis_stage=(0.0, 1.0, 0.0),
        field_provenance={
            field: provenance(f"follower topology {field}")
            for field in ("joint_type", "body0", "body1", "axis_stage")
        },
    )
    body_provenance = provenance("rigid body")
    mass_provenance = provenance("mass")
    collider_provenance = provenance("collider")
    return JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=identify_usd_artifact(source_path, uri=str(source_path)),
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=(
                JointPlanV1(
                    topology=hinge_topology,
                    limit=JointLimitV1(
                        lower=-45.0,
                        upper=None,
                        unit="degrees",
                        provenance=provenance("hinge limit"),
                    ),
                    anchor=JointAnchorV1(
                        position_stage=(0.0, 1.0, 0.0),
                        provenance=provenance("hinge anchor"),
                    ),
                    drive=JointDriveV1(
                        drive_type="force",
                        stiffness=10.0,
                        damping=1.0,
                        max_force=100.0,
                        target_position=0.0,
                        target_velocity=0.0,
                        max_joint_velocity=2.0,
                        provenance=provenance("hinge drive"),
                    ),
                    state=JointStateV1(
                        position=0.0,
                        velocity=0.0,
                        provenance=provenance("hinge state"),
                    ),
                ),
                JointPlanV1(
                    topology=follower_topology,
                    mimic=JointMimicV1(
                        reference_joint_id="hinge",
                        gearing=1.0,
                        offset=0.0,
                        natural_frequency=1.0,
                        damping_ratio=1.0,
                        provenance=provenance("follower mimic"),
                    ),
                ),
            ),
            rigid_bodies=(
                RigidBodyPlanV1(
                    prim_path="/World/Base",
                    mass=MassPropertiesV1(
                        mass_kg=2.0,
                        center_of_mass_m=(0.0, 0.0, 0.0),
                        diagonal_inertia_kg_m2=(1.0, 1.0, 1.0),
                        principal_axes=(1.0, 0.0, 0.0, 0.0),
                        provenance=mass_provenance,
                    ),
                    colliders=(
                        ColliderPlanV1(
                            prim_path="/World/Base/Collision",
                            mesh_collision_api=True,
                            mesh_approximation="convexHull",
                            provenance=collider_provenance,
                        ),
                    ),
                    provenance=body_provenance,
                ),
            ),
            articulation_root=ArticulationRootPlanV1(
                prim_path="/World",
                provenance=provenance("articulation root"),
            ),
        ),
    )


def _diagnostics_for_every_planned_fact(
    request: JointRiggerInputV1,
) -> JointRiggerDiagnosticsV1:
    body = request.plan.rigid_bodies[0]
    assert body.mass is not None
    collider = body.colliders[0]
    top_level = (
        FieldDecisionV1(
            field="legacy_component_names",
            disposition="ignored",
            reason_code="legacy_component_name_compatibility_not_requested",
        ),
        FieldDecisionV1(
            field="articulation_root",
            disposition="accepted",
            provenance=request.plan.articulation_root.provenance,
        ),
        FieldDecisionV1(
            field="rigid_bodies[/World/Base].rigid_body",
            disposition="accepted",
            provenance=body.provenance,
        ),
        *(
            FieldDecisionV1(
                field=f"rigid_bodies[/World/Base].mass.{field}",
                disposition="accepted",
                provenance=body.mass.provenance,
            )
            for field in (
                "mass_kg",
                "center_of_mass_m",
                "diagonal_inertia_kg_m2",
                "principal_axes",
            )
        ),
        *(
            FieldDecisionV1(
                field=f"rigid_bodies[/World/Base].colliders[/World/Base/Collision].{field}",
                disposition="accepted",
                provenance=collider.provenance,
            )
            for field in (
                "collision",
                "mesh_collision_api",
                "mesh_approximation",
            )
        ),
    )
    joint_diagnostics = []
    for joint in request.plan.joints:
        decisions = [
            FieldDecisionV1(
                field=f"topology.{field}",
                disposition="accepted",
                provenance=provenance,
            )
            for field, provenance in joint.topology.field_provenance.items()
        ]
        for prefix, value, fields in (
            ("limit", joint.limit, ("lower", "unit")),
            ("anchor", joint.anchor, ("position_stage",)),
            (
                "drive",
                joint.drive,
                (
                    "drive_type",
                    "stiffness",
                    "damping",
                    "max_force",
                    "target_position",
                    "target_velocity",
                    "max_joint_velocity",
                ),
            ),
            ("state", joint.state, ("position", "velocity")),
            (
                "mimic",
                joint.mimic,
                (
                    "reference_joint_id",
                    "gearing",
                    "offset",
                    "natural_frequency",
                    "damping_ratio",
                ),
            ),
        ):
            if value is None:
                continue
            decisions.extend(
                FieldDecisionV1(
                    field=f"{prefix}.{field}",
                    disposition="accepted",
                    provenance=value.provenance,
                )
                for field in fields
            )
        joint_diagnostics.append(
            JointDiagnosticV1(
                joint_id=joint.topology.joint_id,
                field_decisions=tuple(decisions),
            )
        )
    return JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="complete-fake",
        field_decisions=top_level,
        joint_diagnostics=tuple(joint_diagnostics),
    )


def _result(
    request: JointRiggerInputV1,
    *,
    status: Literal["succeeded", "failed"] = "succeeded",
    output_text: str = _EMPTY_USDA,
    diagnostics: JointRiggerDiagnosticsV1 | None = None,
    dependency_bundle_sha256: str | None | object = _AUTO_BUNDLE,
) -> JointRiggerResultV1:
    if diagnostics is None:
        diagnostics = JointRiggerDiagnosticsV1(
            schema_version=DIAGNOSTICS_SCHEMA_VERSION,
            backend_name="fake",
            field_decisions=(
                FieldDecisionV1(
                    field="legacy_component_names",
                    disposition="ignored",
                    reason_code="legacy_component_name_compatibility_not_requested",
                ),
            ),
        )
    bundle_sha256 = (
        _output_dependency_sha256(output_text)
        if dependency_bundle_sha256 is _AUTO_BUNDLE
        else dependency_bundle_sha256
    )
    assert bundle_sha256 is None or isinstance(bundle_sha256, str)
    output_artifact = (
        ArtifactIdentityV1(
            uri="memory://generated.usda",
            root_sha256=hashlib.sha256(output_text.encode()).hexdigest(),
            dependency_bundle_sha256=bundle_sha256,
        )
        if status == "succeeded"
        else None
    )
    return JointRiggerResultV1(
        schema_version=RESULT_SCHEMA_VERSION,
        status=status,
        input_sha256=canonical_sha256(request),
        plan_sha256=canonical_sha256(request.plan),
        output_artifact=output_artifact,
        diagnostics=diagnostics,
    )


def _targets(tmp_path: Path, *, sidecar: bool = False) -> JointRiggerArtifactTargets:
    return JointRiggerArtifactTargets(
        output_path=tmp_path / "rigged.usda",
        diagnostics_path=tmp_path / "diagnostics.json",
        result_path=tmp_path / "result.json",
        sidecar_path=tmp_path / "rigged_assets" if sidecar else None,
    )


def _expected_sidecar_sha256(
    tmp_path: Path,
    *,
    content: str | None = "dependency",
) -> str:
    sidecar = tmp_path / ".expected-sidecar"
    sidecar.mkdir()
    if content is not None:
        (sidecar / "dependency.usda").write_text(content, encoding="utf-8")
    try:
        return sidecar_dependency_bundle_sha256(sidecar)
    finally:
        shutil.rmtree(sidecar)


def _write_complete_bundle(
    targets: JointRiggerArtifactTargets,
    *,
    marker: str = "old",
) -> None:
    targets.output_path.write_text(f"{marker} output", encoding="utf-8")
    targets.diagnostics_path.write_text(f"{marker} diagnostics", encoding="utf-8")
    targets.result_path.write_text(f"{marker} result", encoding="utf-8")
    if targets.sidecar_path is not None:
        targets.sidecar_path.mkdir()
        (targets.sidecar_path / "dependency.usda").write_text(
            f"{marker} sidecar",
            encoding="utf-8",
        )


def _assert_complete_bundle(
    targets: JointRiggerArtifactTargets,
    *,
    marker: str = "old",
) -> None:
    assert targets.output_path.read_text(encoding="utf-8") == f"{marker} output"
    assert (
        targets.diagnostics_path.read_text(encoding="utf-8") == f"{marker} diagnostics"
    )
    assert targets.result_path.read_text(encoding="utf-8") == f"{marker} result"
    if targets.sidecar_path is not None:
        assert (targets.sidecar_path / "dependency.usda").read_text(
            encoding="utf-8"
        ) == f"{marker} sidecar"


def _assert_unbound_staging_root_preserved(
    error: BaseException,
    parent: Path,
) -> Path:
    """Require evidence that an unbound sibling root was not adopted."""

    preserved = tuple(parent.glob(".*.stage-*"))
    assert len(preserved) == 1
    assert preserved[0].is_file()
    notes = "\n".join(error.__notes__)
    assert "no descriptor-bound cleanup identity" in notes
    assert "preserved" in notes
    return preserved[0]


@dataclass
class _WritingBackend:
    result: JointRiggerResultV1
    name: str = "fake"
    write_output: bool = True
    write_diagnostics: bool = True
    write_result: bool = True
    write_sidecar: bool = True
    output_text: str = _EMPTY_USDA
    sidecar_content: str | None = "dependency"
    author_publication_sidecar_reference: bool = False
    author_physical_sidecar_reference: bool = False
    output_uri_mode: (
        Literal[
            "publication_path",
            "publication_file_uri",
            "physical_path",
            "physical_file_uri",
            "unrelated_path",
            "unrelated_file_uri",
            "logical_uri",
        ]
        | None
    ) = None
    author_error: BaseException | None = None
    persisted_result: JointRiggerResultV1 | None = None
    probed: bool = False
    received_targets: JointRiggerArtifactTargets | None = None

    def probe(self, request: JointRiggerInputV1) -> None:
        self.probed = True

    def author(
        self,
        request: JointRiggerInputV1,
        artifact_targets: JointRiggerArtifactTargets,
    ) -> JointRiggerResultV1:
        self.received_targets = artifact_targets
        if self.author_publication_sidecar_reference:
            output_text = _publication_sidecar_root_text(artifact_targets)
        elif self.author_physical_sidecar_reference:
            assert artifact_targets.sidecar_path is not None
            relative = Path(
                os.path.relpath(
                    artifact_targets.sidecar_path,
                    start=artifact_targets.output_path.parent,
                )
            ).as_posix()
            output_text = _root_with_reference(f"{relative}/dependency.usda")
        else:
            output_text = self.output_text
        effective_result = self.result
        if self.author_physical_sidecar_reference:
            assert effective_result.output_artifact is not None
            effective_result = effective_result.model_copy(
                update={
                    "output_artifact": effective_result.output_artifact.model_copy(
                        update={
                            "root_sha256": hashlib.sha256(
                                output_text.encode()
                            ).hexdigest()
                        }
                    )
                }
            )
        if self.output_uri_mode is not None:
            assert effective_result.output_artifact is not None
            publication_output = artifact_targets.publication_output_path
            assert publication_output is not None
            unrelated_output = publication_output.with_name("unrelated.usda")
            output_uris = {
                "publication_path": str(publication_output),
                "publication_file_uri": publication_output.as_uri(),
                "physical_path": str(artifact_targets.output_path),
                "physical_file_uri": artifact_targets.output_path.as_uri(),
                "unrelated_path": str(unrelated_output),
                "unrelated_file_uri": unrelated_output.as_uri(),
                "logical_uri": "s3://bucket/generated.usda",
            }
            effective_result = effective_result.model_copy(
                update={
                    "output_artifact": effective_result.output_artifact.model_copy(
                        update={"uri": output_uris[self.output_uri_mode]}
                    )
                }
            )
        if self.write_output:
            artifact_targets.output_path.write_text(output_text, encoding="utf-8")
        if self.write_diagnostics:
            artifact_targets.diagnostics_path.write_text(
                self.result.diagnostics.model_dump_json(),
                encoding="utf-8",
            )
        if self.write_result:
            artifact_targets.result_path.write_text(
                (self.persisted_result or effective_result).model_dump_json(),
                encoding="utf-8",
            )
        if artifact_targets.sidecar_path is not None and self.write_sidecar:
            artifact_targets.sidecar_path.mkdir()
            if self.sidecar_content is not None:
                (artifact_targets.sidecar_path / "dependency.usda").write_text(
                    self.sidecar_content,
                    encoding="utf-8",
                )
        if self.author_error is not None:
            raise self.author_error
        return effective_result


@dataclass
class _ProjectedNoSidecarBackend:
    reference_mode: Literal[
        "staging_name",
        "publication_name",
        "sibling",
        "absolute",
    ]
    dependency_path: Path | None = None
    authored_reference: str | None = None
    dependency_kind: Literal["layer", "asset"] = "layer"
    probed: bool = False

    def probe(self, request: JointRiggerInputV1) -> None:
        self.probed = True

    def author(
        self,
        request: JointRiggerInputV1,
        artifact_targets: JointRiggerArtifactTargets,
    ) -> JointRiggerResultV1:
        publication_output = artifact_targets.publication_output_path
        assert publication_output is not None
        if self.reference_mode == "staging_name":
            reference = artifact_targets.output_path.name
        elif self.reference_mode == "publication_name":
            reference = publication_output.name
        else:
            assert self.dependency_path is not None
            reference = self.authored_reference or (
                self.dependency_path.name
                if self.reference_mode == "sibling"
                else str(self.dependency_path)
            )
        output_text = (
            _root_with_reference(reference)
            if self.dependency_kind == "layer"
            else _root_with_asset_reference(reference)
        )
        artifact_targets.output_path.write_text(output_text, encoding="utf-8")
        staged_identity = identify_usd_artifact(
            artifact_targets.output_path,
            uri="memory://generated.usda",
        )
        result = _result(
            request,
            output_text=output_text,
            dependency_bundle_sha256=staged_identity.dependency_bundle_sha256,
        )
        artifact_targets.diagnostics_path.write_text(
            result.diagnostics.model_dump_json(),
            encoding="utf-8",
        )
        artifact_targets.result_path.write_text(
            result.model_dump_json(),
            encoding="utf-8",
        )
        return result


@dataclass
class _TransactionDependencyBackend:
    dependency_target: Literal[
        "staged_diagnostics",
        "staged_result",
        "final_diagnostics",
        "final_result",
    ]
    final_targets: JointRiggerArtifactTargets
    probed: bool = False

    def probe(self, request: JointRiggerInputV1) -> None:
        self.probed = True

    def author(
        self,
        request: JointRiggerInputV1,
        artifact_targets: JointRiggerArtifactTargets,
    ) -> JointRiggerResultV1:
        dependency_paths = {
            "staged_diagnostics": artifact_targets.diagnostics_path,
            "staged_result": artifact_targets.result_path,
            "final_diagnostics": self.final_targets.diagnostics_path,
            "final_result": self.final_targets.result_path,
        }
        output_text = _root_with_asset_reference(
            str(dependency_paths[self.dependency_target])
        )
        artifact_targets.output_path.write_text(output_text, encoding="utf-8")
        result = _result(
            request,
            output_text=output_text,
            dependency_bundle_sha256="0" * 64,
        )
        artifact_targets.diagnostics_path.write_text(
            result.diagnostics.model_dump_json(),
            encoding="utf-8",
        )
        artifact_targets.result_path.write_text(
            result.model_dump_json(),
            encoding="utf-8",
        )
        return result


def test_facade_gives_backend_descriptor_owned_staging_targets_and_publishes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path, sidecar=True)
    _write_complete_bundle(targets)
    request = _request(source)
    root_text = _publication_sidecar_root_text(targets)
    result = _result(
        request,
        output_text=root_text,
        dependency_bundle_sha256=_expected_sidecar_sha256(
            tmp_path,
            content=_DEPENDENCY_USDA,
        ),
    )
    backend = _WritingBackend(
        result,
        sidecar_content=_DEPENDENCY_USDA,
        author_publication_sidecar_reference=True,
    )

    observed = author_joint_rig(request, backend, targets)

    assert observed == result
    assert result.input_sha256 == canonical_sha256(request)
    assert result.input_sha256 != request.source_asset.root_sha256
    assert backend.probed is True
    assert backend.received_targets is not None
    assert backend.received_targets.publication_output_path == targets.output_path
    assert backend.received_targets.publication_sidecar_path == targets.sidecar_path
    assert (
        backend.received_targets.publication_diagnostics_path
        == targets.diagnostics_path
    )
    assert backend.received_targets.publication_result_path == targets.result_path
    for staged_path, final_path, descriptor_owned in (
        (backend.received_targets.output_path, targets.output_path, False),
        (
            backend.received_targets.diagnostics_path,
            targets.diagnostics_path,
            True,
        ),
        (backend.received_targets.result_path, targets.result_path, True),
    ):
        assert staged_path != final_path
        if descriptor_owned:
            assert staged_path.parent.parent == final_path.parent
            assert staged_path.name == final_path.name
            assert staged_path.parent.name.startswith(f".{final_path.name}.stage-")
        else:
            assert staged_path.parent == final_path.parent
            assert staged_path.name.startswith(f".{final_path.stem}.stage-")
        assert staged_path.suffix == final_path.suffix
    staged_sidecar = backend.received_targets.sidecar_path
    assert staged_sidecar is not None
    assert targets.sidecar_path is not None
    assert staged_sidecar.name == targets.sidecar_path.name
    assert staged_sidecar.parent.parent == targets.sidecar_path.parent
    assert staged_sidecar.parent.name.startswith(".rigged_assets.stage-")
    assert targets.output_path.read_text(encoding="utf-8") == root_text
    assert targets.diagnostics_path.is_file()
    assert targets.result_path.is_file()
    assert targets.sidecar_path is not None
    assert (targets.sidecar_path / "dependency.usda").is_file()
    assert not any(tmp_path.glob(".*.stage-*"))


def test_facade_can_replace_a_sealed_sidecar_on_repeated_authoring(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path, sidecar=True)
    request = _request(source)
    root_text = _publication_sidecar_root_text(targets)
    result = _result(
        request,
        output_text=root_text,
        dependency_bundle_sha256=_expected_sidecar_sha256(
            tmp_path,
            content=_DEPENDENCY_USDA,
        ),
    )

    for _ in range(2):
        observed = author_joint_rig(
            request,
            _WritingBackend(
                result,
                sidecar_content=_DEPENDENCY_USDA,
                author_publication_sidecar_reference=True,
            ),
            targets,
        )
        assert observed == result
        assert targets.sidecar_path is not None
        assert stat.S_IMODE(targets.sidecar_path.stat().st_mode) & 0o222 == 0
        assert (targets.sidecar_path / "dependency.usda").read_text(
            encoding="utf-8"
        ) == _DEPENDENCY_USDA
        assert not any(tmp_path.glob(".joint-rigger.rollback-*"))
        assert not any(tmp_path.glob(".*.stage-*"))


def test_relative_sidecar_reference_resolves_after_publication(tmp_path: Path) -> None:
    from pxr import Usd

    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    targets = _targets(tmp_path, sidecar=True)
    assert targets.sidecar_path is not None
    root_text = _publication_sidecar_root_text(targets)
    dependency_text = _DEPENDENCY_USDA
    request = _request(source)
    result = _result(
        request,
        output_text=root_text,
        dependency_bundle_sha256=_expected_sidecar_sha256(
            tmp_path,
            content=dependency_text,
        ),
    )

    author_joint_rig(
        request,
        _WritingBackend(
            result,
            sidecar_content=dependency_text,
            author_publication_sidecar_reference=True,
        ),
        targets,
    )

    stage = Usd.Stage.Open(str(targets.output_path))
    assert stage is not None
    assert stage.GetRootLayer().subLayerPaths == [
        f"{targets.sidecar_path.name}/dependency.usda"
    ]
    dependency_path = (targets.sidecar_path / "dependency.usda").resolve()
    assert dependency_path.is_file()
    assert dependency_path in {
        Path(layer.realPath).resolve()
        for layer in stage.GetUsedLayers()
        if layer.realPath
    }
    assert not any(tmp_path.glob(".*.stage-*"))


def test_sidecar_root_must_reference_declared_publication_sidecar(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path, sidecar=True)
    request = _request(source)
    result = _result(
        request,
        dependency_bundle_sha256=_expected_sidecar_sha256(
            tmp_path,
            content=_DEPENDENCY_USDA,
        ),
    )

    with pytest.raises(JointRiggerArtifactError, match="does not reference"):
        author_joint_rig(
            request,
            _WritingBackend(result, sidecar_content=_DEPENDENCY_USDA),
            targets,
        )

    assert not targets.output_path.exists()


def test_standalone_root_can_publish_with_exactly_empty_sidecar(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path, sidecar=True)
    request = _request(source)
    result = _result(
        request,
        dependency_bundle_sha256=_expected_sidecar_sha256(
            tmp_path,
            content=None,
        ),
    )

    observed = author_joint_rig(
        request,
        _WritingBackend(result, sidecar_content=None),
        targets,
    )

    assert observed == result
    assert targets.output_path.is_file()
    assert targets.sidecar_path is not None
    assert targets.sidecar_path.is_dir()
    assert list(targets.sidecar_path.iterdir()) == []
    assert not any(tmp_path.glob(".*.stage-*"))


def test_sidecar_root_rejects_dependency_outside_declared_sidecar(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    outside = tmp_path / "outside.usda"
    _write_empty_usda(outside)
    targets = _targets(tmp_path, sidecar=True)
    request = _request(source)
    root_text = _root_with_reference("../outside.usda")
    result = _result(
        request,
        output_text=root_text,
        dependency_bundle_sha256=_expected_sidecar_sha256(
            tmp_path,
            content=_DEPENDENCY_USDA,
        ),
    )

    with pytest.raises(JointRiggerArtifactError, match="outside"):
        author_joint_rig(
            request,
            _WritingBackend(
                result,
                output_text=root_text,
                sidecar_content=_DEPENDENCY_USDA,
            ),
            targets,
        )


def test_sidecar_root_rejects_unresolved_dependency(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path, sidecar=True)
    request = _request(source)
    root_text = _root_with_reference("missing.usda")
    result = _result(
        request,
        output_text=root_text,
        dependency_bundle_sha256=_expected_sidecar_sha256(
            tmp_path,
            content=_DEPENDENCY_USDA,
        ),
    )

    with pytest.raises(JointRiggerArtifactError, match="unresolved"):
        author_joint_rig(
            request,
            _WritingBackend(
                result,
                output_text=root_text,
                sidecar_content=_DEPENDENCY_USDA,
            ),
            targets,
        )


def test_sidecar_root_rejects_physical_staging_reference(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path, sidecar=True)
    request = _request(source)
    result = _result(
        request,
        dependency_bundle_sha256=_expected_sidecar_sha256(
            tmp_path,
            content=_DEPENDENCY_USDA,
        ),
    )

    with pytest.raises(JointRiggerArtifactError, match="publication"):
        author_joint_rig(
            request,
            _WritingBackend(
                result,
                sidecar_content=_DEPENDENCY_USDA,
                author_physical_sidecar_reference=True,
            ),
            targets,
        )

    assert not targets.output_path.exists()


def test_sidecar_parent_preflight_fails_before_backend_probe(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    backend = _WritingBackend(_result(request))
    targets = JointRiggerArtifactTargets(
        output_path=tmp_path / "output" / "rigged.usda",
        diagnostics_path=tmp_path / "diagnostics.json",
        result_path=tmp_path / "result.json",
        sidecar_path=tmp_path / "dependencies" / "rigged_assets",
    )

    with pytest.raises(ValueError, match="share output_path's parent"):
        author_joint_rig(request, backend, targets)

    assert backend.probed is False
    assert not any(tmp_path.rglob(".*.stage-*"))


@pytest.mark.parametrize(
    ("author_error", "expected_error"),
    [
        (RuntimeError("author failed"), JointRiggerArtifactError),
        (OSError("author I/O failed"), JointRiggerArtifactError),
        (
            AttributeError("missing backend attribute"),
            JointRiggerBackendIncompatibleError,
        ),
        (TypeError("invalid backend call"), JointRiggerBackendIncompatibleError),
    ],
)
def test_backend_failure_is_typed_and_preserves_existing_complete_bundle(
    tmp_path: Path,
    author_error: Exception,
    expected_error: type[JointRiggerArtifactError]
    | type[JointRiggerBackendIncompatibleError],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    request = _request(source)
    backend = _WritingBackend(
        _result(request),
        author_error=author_error,
    )

    with pytest.raises(expected_error, match="during authoring") as raised:
        author_joint_rig(request, backend, targets)

    assert raised.value.__cause__ is author_error
    _assert_complete_bundle(targets)
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()


def test_backend_typed_facade_error_is_preserved_after_cleanup(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    request = _request(source)
    expected = JointRiggerArtifactError("typed backend artifact failure")
    backend = _WritingBackend(_result(request), author_error=expected)

    with pytest.raises(JointRiggerArtifactError) as raised:
        author_joint_rig(request, backend, targets)

    assert raised.value is expected
    _assert_complete_bundle(targets)
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()


@pytest.mark.parametrize("author_error", [KeyboardInterrupt(), SystemExit(17)])
def test_backend_termination_exception_propagates_after_cleanup(
    tmp_path: Path,
    author_error: BaseException,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    request = _request(source)
    backend = _WritingBackend(_result(request), author_error=author_error)

    with pytest.raises(type(author_error)) as raised:
        author_joint_rig(request, backend, targets)

    assert raised.value is author_error
    _assert_complete_bundle(targets)
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()


def test_backend_cannot_publish_source_hardlink_as_generated_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text(_EMPTY_USDA, encoding="utf-8")
    request = _request(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    class HardLinkingBackend(_WritingBackend):
        def author(
            self,
            request: JointRiggerInputV1,
            artifact_targets: JointRiggerArtifactTargets,
        ) -> JointRiggerResultV1:
            result = super().author(request, artifact_targets)
            os.link(source, artifact_targets.output_path)
            return result

    backend = HardLinkingBackend(_result(request), write_output=False)
    with pytest.raises(
        JointRiggerArtifactError,
        match="generated root.*exactly one hard link",
    ) as raised:
        author_joint_rig(request, backend, targets)

    assert source.read_text(encoding="utf-8") == _EMPTY_USDA
    assert source.stat().st_nlink == 2
    _assert_complete_bundle(targets)
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()
    assert source.stat().st_nlink == 1


def test_backend_cannot_publish_input_dependency_hardlink_in_sidecar(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "input-dependency.usda"
    dependency.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    source = tmp_path / "source.usda"
    source.write_text(_root_with_reference(dependency.name), encoding="utf-8")
    request = _request(source)
    targets = _targets(tmp_path, sidecar=True)
    _write_complete_bundle(targets)
    root_text = _publication_sidecar_root_text(targets)
    result = _result(
        request,
        output_text=root_text,
        dependency_bundle_sha256=_expected_sidecar_sha256(
            tmp_path,
            content=_DEPENDENCY_USDA,
        ),
    )

    class HardLinkingSidecarBackend(_WritingBackend):
        def author(
            self,
            request: JointRiggerInputV1,
            artifact_targets: JointRiggerArtifactTargets,
        ) -> JointRiggerResultV1:
            result = super().author(request, artifact_targets)
            assert artifact_targets.sidecar_path is not None
            artifact_targets.sidecar_path.mkdir()
            os.link(
                dependency,
                artifact_targets.sidecar_path / "dependency.usda",
            )
            return result

    backend = HardLinkingSidecarBackend(
        result,
        write_sidecar=False,
        author_publication_sidecar_reference=True,
    )
    with pytest.raises(
        JointRiggerArtifactError,
        match="invalid composition sidecar.*exactly one hard link",
    ) as raised:
        author_joint_rig(request, backend, targets)

    assert dependency.read_text(encoding="utf-8") == _DEPENDENCY_USDA
    assert dependency.stat().st_nlink == 1
    assert source.read_text(encoding="utf-8") == _root_with_reference(dependency.name)
    _assert_complete_bundle(targets)
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()


@pytest.mark.parametrize(
    "probe_error",
    [
        JointRiggerBackendUnavailableError("missing dependency"),
        ModuleNotFoundError("missing native dependency"),
    ],
)
def test_missing_backend_is_typed_and_preserves_existing_complete_bundle(
    tmp_path: Path,
    probe_error: BaseException,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    request = _request(source)

    class MissingBackend:
        name = "missing"

        def probe(self, request: JointRiggerInputV1) -> None:
            raise probe_error

        def author(
            self,
            request: JointRiggerInputV1,
            artifact_targets: JointRiggerArtifactTargets,
        ) -> JointRiggerResultV1:
            raise AssertionError("author must not run")

    with pytest.raises(JointRiggerBackendUnavailableError):
        author_joint_rig(request, MissingBackend(), targets)

    _assert_complete_bundle(targets)


def test_missing_backend_api_is_typed_and_leaves_no_staging(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path)
    request = _request(source)

    class MissingAuthor:
        def probe(self, request: JointRiggerInputV1) -> None:
            return None

    with pytest.raises(JointRiggerBackendIncompatibleError, match="author"):
        author_joint_rig(request, MissingAuthor(), targets)  # type: ignore[arg-type]

    assert not targets.output_path.exists()
    assert not any(tmp_path.glob(".*.stage-*"))


def test_lazy_dependency_failure_during_authoring_is_typed_and_cleaned(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path)
    request = _request(source)
    backend = _WritingBackend(
        _result(request),
        author_error=ImportError("optional runtime is incompatible"),
    )

    with pytest.raises(
        JointRiggerBackendUnavailableError,
        match="authoring",
    ) as raised:
        author_joint_rig(request, backend, targets)

    assert not targets.output_path.exists()
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()


def test_incomplete_reports_cannot_publish_generated_root(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    request = _request(source)
    backend = _WritingBackend(_result(request), write_result=False)

    with pytest.raises(JointRiggerArtifactError, match="result report") as raised:
        author_joint_rig(request, backend, targets)

    _assert_complete_bundle(targets)
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()


def test_model_reports_are_read_through_a_bounded_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "oversized-result.json"
    report.write_bytes(b"{}" + (b" " * 32))
    monkeypatch.setattr(facade, "_MAX_REPORT_BYTES", 16)

    with pytest.raises(JointRiggerArtifactError, match="exceeds the 16-byte limit"):
        facade._load_model_report(report, JointRiggerResultV1, "result")


def test_model_reports_reject_duplicate_json_object_keys(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    valid_payload = _result(_request(source)).model_dump_json()
    report = tmp_path / "ambiguous-result.json"
    report.write_text(
        valid_payload.replace(
            '"status":"succeeded"',
            '"status":"failed","status":"succeeded"',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="duplicate JSON object key: 'status'",
    ):
        facade._load_model_report(report, JointRiggerResultV1, "result")


def test_model_reports_wrap_deep_json_recursion_as_artifact_error(
    tmp_path: Path,
) -> None:
    # CPython's C JSON scanner can exceed the Python recursion limit by a
    # modest amount; this still-small payload deterministically crosses it.
    depth = 10_000
    payload = (b"[" * depth) + b"0" + (b"]" * depth)
    report = tmp_path / "deeply-nested-result.json"
    report.write_bytes(payload)

    with pytest.raises(
        JointRiggerArtifactError,
        match="Backend wrote an invalid result report",
    ) as direct:
        facade._load_model_report(report, JointRiggerResultV1, "result")
    assert isinstance(direct.value.__cause__, RecursionError)

    snapshot = facade._create_private_report_snapshot(
        tmp_path / "sealed-result.json",
        payload,
        label="result",
    )
    try:
        with pytest.raises(
            JointRiggerArtifactError,
            match="Private result contains invalid report JSON",
        ) as sealed:
            facade._load_sealed_model_report(
                snapshot,
                JointRiggerResultV1,
                "result",
            )
        assert isinstance(sealed.value.__cause__, RecursionError)
    finally:
        snapshot.cleanup()


def test_report_writer_descriptor_close_error_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = facade.os.open
    real_close = facade.os.close
    writer_descriptor: int | None = None
    close_calls: list[int] = []

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal writer_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if flags & os.O_CREAT and flags & os.O_WRONLY:
            writer_descriptor = descriptor
        return descriptor

    def close_writer_once(descriptor: int) -> None:
        close_calls.append(descriptor)
        real_close(descriptor)
        if descriptor == writer_descriptor:
            raise OSError(errno.EIO, "forced writer close failure")

    monkeypatch.setattr(facade.os, "open", track_open)
    monkeypatch.setattr(facade.os, "close", close_writer_once)

    with pytest.raises(JointRiggerArtifactError, match="forced writer close failure"):
        facade._create_private_report_snapshot(
            tmp_path / "result.json",
            b"{}",
            label="result",
        )

    assert writer_descriptor is not None
    assert close_calls.count(writer_descriptor) == 1
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_generated_root_descriptor_cleanup_is_single_shot_after_close_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generated.usda"
    root.write_text(_EMPTY_USDA, encoding="utf-8")
    snapshot = facade._seal_generated_root(
        root,
        expected_sha256=hashlib.sha256(_EMPTY_USDA.encode()).hexdigest(),
    )
    descriptor = snapshot.source_descriptor
    real_close = facade.os.close
    close_calls: list[int] = []

    def close_once_with_error(candidate: int) -> None:
        close_calls.append(candidate)
        real_close(candidate)
        if candidate == descriptor:
            raise OSError(errno.EIO, "forced generated-root close failure")

    monkeypatch.setattr(facade.os, "close", close_once_with_error)

    with pytest.raises(JointRiggerArtifactError, match="generated-root close"):
        snapshot.cleanup()
    snapshot.cleanup()

    assert close_calls.count(descriptor) == 1


@pytest.mark.parametrize("report_kind", ["result", "diagnostics"])
def test_validated_report_is_sealed_from_append_through_backend_held_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_kind: Literal["result", "diagnostics"],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    result = _result(request)
    targets = _targets(tmp_path)
    report_payloads = {
        "result": result.model_dump_json().encode(),
        "diagnostics": result.diagnostics.model_dump_json().encode(),
    }
    report_limit = max(len(payload) for payload in report_payloads.values()) + 64
    monkeypatch.setattr(facade, "_MAX_REPORT_BYTES", report_limit)

    class HeldDescriptorBackend:
        def __init__(self) -> None:
            self.delegate = _WritingBackend(result)
            self.append_now = threading.Event()
            self.appended = threading.Event()
            self.worker: threading.Thread | None = None

        def probe(self, candidate: JointRiggerInputV1) -> None:
            self.delegate.probe(candidate)

        def author(
            self,
            candidate: JointRiggerInputV1,
            artifact_targets: JointRiggerArtifactTargets,
        ) -> JointRiggerResultV1:
            observed = self.delegate.author(candidate, artifact_targets)
            report_path = getattr(artifact_targets, f"{report_kind}_path")
            descriptor = os.open(
                report_path,
                os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0),
            )

            def append_after_validation() -> None:
                try:
                    assert self.append_now.wait(timeout=5.0)
                    payload = memoryview(b"x" * (report_limit + 1))
                    while payload:
                        written = os.write(descriptor, payload)
                        assert written > 0
                        payload = payload[written:]
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                    self.appended.set()

            self.worker = threading.Thread(target=append_after_validation)
            self.worker.start()
            return observed

    backend = HeldDescriptorBackend()
    real_staged_promotion_artifacts = facade.staged_promotion_artifacts

    def validate_staging_after_backend_append(
        staged: StagedJointRiggerArtifacts,
    ) -> list[StagedArtifact]:
        # The facade has already copied the accepted bytes to a private path.
        # This append therefore mutates only the backend-known report inode.
        backend.append_now.set()
        assert backend.appended.wait(timeout=5.0)
        return real_staged_promotion_artifacts(staged)

    monkeypatch.setattr(
        facade,
        "staged_promotion_artifacts",
        validate_staging_after_backend_append,
    )

    observed = author_joint_rig(request, backend, targets)

    assert backend.worker is not None
    backend.worker.join(timeout=5.0)
    assert not backend.worker.is_alive()
    assert observed == result
    assert targets.result_path.read_bytes() == report_payloads["result"]
    assert targets.diagnostics_path.read_bytes() == report_payloads["diagnostics"]
    assert stat.S_IMODE(targets.result_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(targets.diagnostics_path.stat().st_mode) == 0o444
    assert (
        JointRiggerResultV1.model_validate_json(targets.result_path.read_bytes())
        == result
    )
    assert (
        JointRiggerDiagnosticsV1.model_validate_json(
            targets.diagnostics_path.read_bytes()
        )
        == result.diagnostics
    )


@pytest.mark.parametrize("report_kind", ["result", "diagnostics"])
def test_backend_known_report_replacement_is_reported_after_sealed_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_kind: Literal["result", "diagnostics"],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    result = _result(request)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    report_payloads = {
        "result": result.model_dump_json().encode(),
        "diagnostics": result.diagnostics.model_dump_json().encode(),
    }
    backend = _WritingBackend(result)
    real_staged_promotion_artifacts = facade.staged_promotion_artifacts

    def replace_after_staged_validation(
        staged: StagedJointRiggerArtifacts,
    ) -> list[StagedArtifact]:
        promotion = real_staged_promotion_artifacts(staged)
        report_path = getattr(staged.staged_targets, f"{report_kind}_path")
        replacement = report_path.with_name(f".{report_path.name}.attacker")
        if report_kind == "result":
            replacement.write_text(
                _result(request, status="failed").model_dump_json(),
                encoding="utf-8",
            )
        else:
            replacement.write_text(
                JointRiggerDiagnosticsV1(
                    schema_version=DIAGNOSTICS_SCHEMA_VERSION,
                    backend_name="attacker",
                    field_decisions=(
                        FieldDecisionV1(
                            field="legacy_component_names",
                            disposition="ignored",
                            reason_code="attacker_replacement",
                        ),
                    ),
                ).model_dump_json(),
                encoding="utf-8",
            )
        replacement.replace(report_path)
        return promotion

    monkeypatch.setattr(
        facade,
        "staged_promotion_artifacts",
        replace_after_staged_validation,
    )

    with pytest.raises(JointRiggerPostCommitCleanupError) as exc_info:
        author_joint_rig(request, backend, targets)

    assert exc_info.value.committed_result == result
    assert "Staging payload changed inode" in str(exc_info.value)
    assert targets.result_path.read_bytes() == report_payloads["result"]
    assert targets.diagnostics_path.read_bytes() == report_payloads["diagnostics"]
    assert (
        JointRiggerResultV1.model_validate_json(targets.result_path.read_bytes())
        == result
    )
    assert (
        JointRiggerDiagnosticsV1.model_validate_json(
            targets.diagnostics_path.read_bytes()
        )
        == result.diagnostics
    )
    assert backend.received_targets is not None
    staged_report = getattr(backend.received_targets, f"{report_kind}_path")
    assert staged_report.is_file()
    shutil.rmtree(staged_report.parent)
    assert not any(tmp_path.glob(".*.stage-*"))


@pytest.mark.parametrize("payload_kind", ["diagnostics", "sidecar"])
def test_backend_payload_cannot_replace_sealed_final_during_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload_kind: Literal["diagnostics", "sidecar"],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path, sidecar=payload_kind == "sidecar")
    if payload_kind == "sidecar":
        root_text = _publication_sidecar_root_text(targets)
        result = _result(
            request,
            output_text=root_text,
            dependency_bundle_sha256=_expected_sidecar_sha256(
                tmp_path,
                content=_DEPENDENCY_USDA,
            ),
        )
        backend = _WritingBackend(
            result,
            sidecar_content=_DEPENDENCY_USDA,
            author_publication_sidecar_reference=True,
        )
    else:
        result = _result(request)
        backend = _WritingBackend(result)

    original_cleanup = facade._cleanup_authoring_state
    displaced = tmp_path / f"displaced-sealed-{payload_kind}"

    def replace_final_then_cleanup(
        sealed_reports: Any,
        sealed_generated: Any,
        staged: StagedJointRiggerArtifacts,
    ) -> None:
        if payload_kind == "diagnostics":
            backend_payload = staged.staged_targets.diagnostics_path
            final_payload = staged.final_targets.diagnostics_path
        else:
            backend_payload = staged.staged_targets.sidecar_path
            final_payload = staged.final_targets.sidecar_path
            assert backend_payload is not None
            assert final_payload is not None
        final_payload.rename(displaced)
        if payload_kind == "sidecar":
            os.chmod(backend_payload, 0o700)
        backend_payload.rename(final_payload)
        if payload_kind == "diagnostics":
            final_payload.write_text('{"backend":"changed"}', encoding="utf-8")
        else:
            backend_dependency = final_payload / "dependency.usda"
            os.chmod(backend_dependency, 0o600)
            backend_dependency.write_text(
                "backend changed",
                encoding="utf-8",
            )
        original_cleanup(sealed_reports, sealed_generated, staged)

    monkeypatch.setattr(facade, "_cleanup_authoring_state", replace_final_then_cleanup)

    with pytest.raises(JointRiggerPostCommitCleanupError) as exc_info:
        author_joint_rig(request, backend, targets)

    assert exc_info.value.committed_result == result
    assert "without an exact recorded promotion" in str(exc_info.value)
    assert displaced.exists()
    if payload_kind == "diagnostics":
        assert targets.diagnostics_path.read_text(encoding="utf-8") == (
            '{"backend":"changed"}'
        )
    else:
        assert targets.sidecar_path is not None
        assert (targets.sidecar_path / "dependency.usda").read_text(
            encoding="utf-8"
        ) == "backend changed"
    assert backend.received_targets is not None
    if payload_kind == "diagnostics":
        owner = backend.received_targets.diagnostics_path.parent
    else:
        assert backend.received_targets.sidecar_path is not None
        owner = backend.received_targets.sidecar_path.parent
    shutil.rmtree(owner)


@pytest.mark.parametrize("report_kind", ["result", "diagnostics"])
def test_private_report_path_replacement_after_revalidation_cannot_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_kind: Literal["result", "diagnostics"],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    result = _result(request)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    real_promote_staged_artifacts = facade.promote_staged_artifacts
    replaced_snapshot: Path | None = None

    def replace_private_snapshot(
        promotion: list[StagedArtifact],
        *,
        precommit_validator: Callable[[], None],
    ) -> None:
        nonlocal replaced_snapshot
        target_path = getattr(targets, f"{report_kind}_path")
        report_artifact = next(
            artifact for artifact in promotion if artifact.target_path == target_path
        )
        assert report_artifact.source_descriptor is not None
        assert report_artifact.source_sha256 is not None
        replaced_snapshot = report_artifact.staged_path
        replacement = replaced_snapshot.with_name(f".{replaced_snapshot.name}.attacker")
        replacement.write_text('{"attacker":true}', encoding="utf-8")
        replacement.replace(replaced_snapshot)
        real_promote_staged_artifacts(
            promotion,
            precommit_validator=precommit_validator,
        )

    monkeypatch.setattr(
        facade,
        "promote_staged_artifacts",
        replace_private_snapshot,
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="Could not publish",
    ) as raised:
        author_joint_rig(request, _WritingBackend(result), targets)

    assert replaced_snapshot is not None
    assert any(
        "private snapshot entry changed inode; refusing deletion" in note
        for note in raised.value.__notes__
    )
    assert not replaced_snapshot.exists()
    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.stage-*"))


def test_private_report_cleanup_survives_parent_rename_and_recreation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    result = _result(request)
    live_parent = tmp_path / "live"
    displaced_parent = tmp_path / "displaced-live"
    live_parent.mkdir()
    targets = _targets(live_parent)
    _write_complete_bundle(targets)
    real_promote_staged_artifacts = facade.promote_staged_artifacts
    sealed_names: list[str] = []

    def swap_report_parent(
        promotion: list[StagedArtifact],
        *,
        precommit_validator: Callable[[], None],
    ) -> None:
        sealed_names.extend(
            artifact.staged_path.name
            for artifact in promotion
            if artifact.label in {"diagnostics report", "result report"}
        )
        live_parent.rename(displaced_parent)
        live_parent.mkdir()
        _write_complete_bundle(_targets(live_parent), marker="unrelated")
        real_promote_staged_artifacts(
            promotion,
            precommit_validator=precommit_validator,
        )

    monkeypatch.setattr(
        facade,
        "promote_staged_artifacts",
        swap_report_parent,
    )

    with pytest.raises(JointRiggerArtifactError, match="Could not publish"):
        author_joint_rig(request, _WritingBackend(result), targets)

    assert len(sealed_names) == 2
    _assert_complete_bundle(_targets(displaced_parent))
    _assert_complete_bundle(_targets(live_parent), marker="unrelated")
    for sealed_name in sealed_names:
        assert not (displaced_parent / sealed_name).exists()
        assert not (live_parent / sealed_name).exists()
    assert not any(displaced_parent.glob(".*.sealed-*"))
    assert not any(live_parent.glob(".*.sealed-*"))


def test_generated_root_held_writer_cannot_mutate_committed_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    result = _result(request)
    targets = _targets(tmp_path)

    class HeldRootWriterBackend(_WritingBackend):
        descriptor: int | None = None
        source_identity: tuple[int, int] | None = None

        def author(
            self,
            candidate: JointRiggerInputV1,
            artifact_targets: JointRiggerArtifactTargets,
        ) -> JointRiggerResultV1:
            observed = super().author(candidate, artifact_targets)
            self.descriptor = os.open(
                artifact_targets.output_path,
                os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0),
            )
            metadata = os.fstat(self.descriptor)
            self.source_identity = (metadata.st_dev, metadata.st_ino)
            return observed

    backend = HeldRootWriterBackend(result)
    real_promote_staged_artifacts = facade.promote_staged_artifacts

    def mutate_backend_inode_after_commit(
        promotion: list[StagedArtifact],
        *,
        precommit_validator: Callable[[], None],
    ) -> None:
        real_promote_staged_artifacts(
            promotion,
            precommit_validator=precommit_validator,
        )
        assert backend.descriptor is not None
        descriptor = backend.descriptor
        backend.descriptor = None
        try:
            os.write(descriptor, b"\n# backend-held writer mutation\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(
        facade,
        "promote_staged_artifacts",
        mutate_backend_inode_after_commit,
    )

    observed = author_joint_rig(request, backend, targets)

    assert observed == result
    assert targets.output_path.read_text(encoding="utf-8") == _EMPTY_USDA
    assert backend.source_identity is not None
    final_metadata = targets.output_path.stat()
    assert (final_metadata.st_dev, final_metadata.st_ino) != backend.source_identity
    assert final_metadata.st_mode & 0o222 == 0


def test_sealed_root_commit_state_does_not_authorize_backend_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    result = _result(request)
    targets = _targets(tmp_path)
    displaced_sealed = tmp_path / "sealed-accepted.usda"

    class HeldRootWriterBackend(_WritingBackend):
        descriptor: int | None = None

        def author(
            self,
            candidate: JointRiggerInputV1,
            artifact_targets: JointRiggerArtifactTargets,
        ) -> JointRiggerResultV1:
            observed = super().author(candidate, artifact_targets)
            self.descriptor = os.open(
                artifact_targets.output_path,
                os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
            )
            return observed

    backend = HeldRootWriterBackend(result)
    real_unlink_descriptor_source_name = artifacts_module._unlink_descriptor_source_name
    race_injected = False

    def replace_sealed_root_before_source_unlink(
        bound_artifact: Any,
    ) -> None:
        nonlocal race_injected
        if race_injected or bound_artifact.artifact.target_path != targets.output_path:
            real_unlink_descriptor_source_name(bound_artifact)
            return
        assert backend.received_targets is not None
        backend_root = backend.received_targets.output_path
        assert backend.descriptor is not None
        targets.output_path.rename(displaced_sealed)
        os.link(backend_root, targets.output_path, follow_symlinks=False)
        real_unlink_descriptor_source_name(bound_artifact)
        descriptor = backend.descriptor
        backend.descriptor = None
        try:
            payload = b"backend changed after seal"
            os.ftruncate(descriptor, 0)
            assert os.write(descriptor, payload) == len(payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        race_injected = True

    monkeypatch.setattr(
        artifacts_module,
        "_unlink_descriptor_source_name",
        replace_sealed_root_before_source_unlink,
    )

    try:
        with pytest.raises(JointRiggerPostCommitCleanupError) as raised:
            author_joint_rig(request, backend, targets)
    finally:
        if backend.descriptor is not None:
            os.close(backend.descriptor)
            backend.descriptor = None

    assert race_injected
    assert raised.value.committed_result == result
    assert "disappeared without reaching its exact publication target" in str(
        raised.value.cleanup_error
    )
    assert targets.output_path.read_bytes() == b"backend changed after seal"
    assert displaced_sealed.read_text(encoding="utf-8") == _EMPTY_USDA


def test_generated_root_held_writer_mutation_before_promotion_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    result = _result(request)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    class HeldRootWriterBackend(_WritingBackend):
        descriptor: int | None = None

        def author(
            self,
            candidate: JointRiggerInputV1,
            artifact_targets: JointRiggerArtifactTargets,
        ) -> JointRiggerResultV1:
            observed = super().author(candidate, artifact_targets)
            self.descriptor = os.open(
                artifact_targets.output_path,
                os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
            )
            return observed

    backend = HeldRootWriterBackend(result)
    real_promote_staged_artifacts = facade.promote_staged_artifacts

    def mutate_before_promotion(
        promotion: list[StagedArtifact],
        *,
        precommit_validator: Callable[[], None],
    ) -> None:
        assert backend.descriptor is not None
        descriptor = backend.descriptor
        backend.descriptor = None
        try:
            os.pwrite(descriptor, b"X", 0)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        real_promote_staged_artifacts(
            promotion,
            precommit_validator=precommit_validator,
        )

    monkeypatch.setattr(
        facade,
        "promote_staged_artifacts",
        mutate_before_promotion,
    )

    with pytest.raises(JointRiggerArtifactError, match="Could not publish"):
        author_joint_rig(request, backend, targets)

    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_generated_root_path_replacement_after_revalidation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    result = _result(request)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    real_promote_staged_artifacts = facade.promote_staged_artifacts
    replaced_path: Path | None = None

    def replace_generated_root(
        promotion: list[StagedArtifact],
        *,
        precommit_validator: Callable[[], None],
    ) -> None:
        nonlocal replaced_path
        root_artifact = next(
            artifact
            for artifact in promotion
            if artifact.target_path == targets.output_path
        )
        assert root_artifact.source_descriptor is not None
        assert root_artifact.source_sha256 is not None
        replaced_path = root_artifact.staged_path
        replacement = replaced_path.with_name(f".{replaced_path.name}.attacker")
        replacement.write_text(_EMPTY_USDA, encoding="utf-8")
        replacement.chmod(0o444)
        replacement.replace(replaced_path)
        real_promote_staged_artifacts(
            promotion,
            precommit_validator=precommit_validator,
        )

    monkeypatch.setattr(
        facade,
        "promote_staged_artifacts",
        replace_generated_root,
    )

    with pytest.raises(JointRiggerArtifactError, match="Could not publish"):
        author_joint_rig(request, _WritingBackend(result), targets)

    assert replaced_path is not None
    assert replaced_path.read_text(encoding="utf-8") == _EMPTY_USDA
    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_sidecar_held_writer_cannot_mutate_private_promotion_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path, sidecar=True)
    request = _request(source)
    root_text = _publication_sidecar_root_text(targets)
    expected_bundle_sha256 = _expected_sidecar_sha256(
        tmp_path,
        content=_DEPENDENCY_USDA,
    )
    result = _result(
        request,
        output_text=root_text,
        dependency_bundle_sha256=expected_bundle_sha256,
    )

    class HeldSidecarWriterBackend(_WritingBackend):
        descriptor: int | None = None
        source_identity: tuple[int, int] | None = None

        def author(
            self,
            candidate: JointRiggerInputV1,
            artifact_targets: JointRiggerArtifactTargets,
        ) -> JointRiggerResultV1:
            observed = super().author(candidate, artifact_targets)
            assert artifact_targets.sidecar_path is not None
            member = artifact_targets.sidecar_path / "dependency.usda"
            self.descriptor = os.open(
                member,
                os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0),
            )
            metadata = os.fstat(self.descriptor)
            self.source_identity = (metadata.st_dev, metadata.st_ino)
            return observed

    backend = HeldSidecarWriterBackend(
        result,
        sidecar_content=_DEPENDENCY_USDA,
        author_publication_sidecar_reference=True,
    )
    real_promote_staged_artifacts = facade.promote_staged_artifacts

    def mutate_original_sidecar(
        promotion: list[StagedArtifact],
        *,
        precommit_validator: Callable[[], None],
    ) -> None:
        assert backend.descriptor is not None
        descriptor = backend.descriptor
        backend.descriptor = None
        try:
            os.write(descriptor, b"\n# backend-held sidecar mutation\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        real_promote_staged_artifacts(
            promotion,
            precommit_validator=precommit_validator,
        )

    monkeypatch.setattr(
        facade,
        "promote_staged_artifacts",
        mutate_original_sidecar,
    )

    observed = author_joint_rig(request, backend, targets)

    assert observed == result
    assert targets.sidecar_path is not None
    final_member = targets.sidecar_path / "dependency.usda"
    assert final_member.read_text(encoding="utf-8") == _DEPENDENCY_USDA
    assert backend.source_identity is not None
    final_metadata = final_member.stat()
    assert (final_metadata.st_dev, final_metadata.st_ino) != backend.source_identity
    assert sidecar_dependency_bundle_sha256(targets.sidecar_path) == (
        expected_bundle_sha256
    )
    assert targets.output_path.read_text(encoding="utf-8") == root_text


def test_sidecar_mutation_during_private_copy_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path, sidecar=True)
    _write_complete_bundle(targets)
    request = _request(source)
    root_text = _publication_sidecar_root_text(targets)
    result = _result(
        request,
        output_text=root_text,
        dependency_bundle_sha256=_expected_sidecar_sha256(
            tmp_path,
            content=_DEPENDENCY_USDA,
        ),
    )
    real_copy_sidecar = facade.copy_sidecar_directory

    def copy_then_mutate_source(
        source_path: str | os.PathLike[str],
        target_descriptor: int,
        *,
        label: str,
        preserve_modes: bool = True,
    ) -> None:
        real_copy_sidecar(
            source_path,
            target_descriptor,
            label=label,
            preserve_modes=preserve_modes,
        )
        if label == "composition sidecar snapshot":
            (Path(source_path) / "dependency.usda").write_text(
                "mutated during copy",
                encoding="utf-8",
            )

    monkeypatch.setattr(facade, "copy_sidecar_directory", copy_then_mutate_source)

    with pytest.raises(JointRiggerArtifactError, match="changed while"):
        author_joint_rig(
            request,
            _WritingBackend(
                result,
                sidecar_content=_DEPENDENCY_USDA,
                author_publication_sidecar_reference=True,
            ),
            targets,
        )

    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_sealed_sidecar_composition_rejects_projection_only_safe_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    outside = tmp_path / "outside.usda"
    outside.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    unsafe_dependency = _root_with_reference(str(outside))
    safe_dependency = _DEPENDENCY_USDA
    targets = _targets(tmp_path, sidecar=True)
    _write_complete_bundle(targets)
    request = _request(source)
    root_text = _publication_sidecar_root_text(targets)
    result = _result(
        request,
        output_text=root_text,
        dependency_bundle_sha256=_expected_sidecar_sha256(
            tmp_path,
            content=unsafe_dependency,
        ),
    )

    class HeldSidecarWriterBackend(_WritingBackend):
        descriptor: int | None = None
        sidecar_path: Path | None = None

        def author(
            self,
            candidate: JointRiggerInputV1,
            artifact_targets: JointRiggerArtifactTargets,
        ) -> JointRiggerResultV1:
            observed = super().author(candidate, artifact_targets)
            assert artifact_targets.sidecar_path is not None
            self.sidecar_path = artifact_targets.sidecar_path
            self.descriptor = os.open(
                artifact_targets.sidecar_path / "dependency.usda",
                os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
            )
            return observed

    backend = HeldSidecarWriterBackend(
        result,
        sidecar_content=unsafe_dependency,
        author_publication_sidecar_reference=True,
    )
    real_copy_sidecar = facade.copy_sidecar_directory
    projection_swapped = False

    def copy_safe_projection_then_restore(
        source_path: str | os.PathLike[str],
        target_descriptor: int,
        *,
        label: str,
        preserve_modes: bool = True,
    ) -> None:
        nonlocal projection_swapped
        if not projection_swapped and Path(source_path) == backend.sidecar_path:
            projection_swapped = True
            assert backend.descriptor is not None
            os.ftruncate(backend.descriptor, 0)
            os.pwrite(backend.descriptor, safe_dependency.encode(), 0)
            os.fsync(backend.descriptor)
            try:
                real_copy_sidecar(
                    source_path,
                    target_descriptor,
                    label=label,
                    preserve_modes=preserve_modes,
                )
            finally:
                os.ftruncate(backend.descriptor, 0)
                os.pwrite(backend.descriptor, unsafe_dependency.encode(), 0)
                os.fsync(backend.descriptor)
            return
        real_copy_sidecar(
            source_path,
            target_descriptor,
            label=label,
            preserve_modes=preserve_modes,
        )

    monkeypatch.setattr(
        facade,
        "copy_sidecar_directory",
        copy_safe_projection_then_restore,
    )
    try:
        with pytest.raises(
            JointRiggerArtifactError,
            match="outside its composition sidecar",
        ):
            author_joint_rig(request, backend, targets)
    finally:
        if backend.descriptor is not None:
            os.close(backend.descriptor)

    assert projection_swapped
    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_private_sidecar_copy_fifo_swap_fails_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source-sidecar"
    source.mkdir()
    member = source / "dependency.usda"
    member.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    expected_sha256 = sidecar_dependency_bundle_sha256(source)
    real_open = facade.os.open
    member_open_count = 0
    swapped = False

    def swap_copy_source_to_fifo(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal member_open_count, swapped
        if os.fspath(path) == member.name and dir_fd is not None:
            member_open_count += 1
            if member_open_count == 2:
                swapped = True
                assert flags & getattr(os, "O_NONBLOCK", 0)
                os.unlink(member.name, dir_fd=dir_fd)
                os.mkfifo(member.name, dir_fd=dir_fd)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(facade.os, "open", swap_copy_source_to_fifo)

    with pytest.raises(JointRiggerArtifactError, match="Could not seal"):
        facade._create_private_sidecar_snapshot(
            source,
            private_parent=tmp_path,
            expected_sha256=expected_sha256,
        )

    assert swapped
    assert stat.S_ISFIFO(member.lstat().st_mode)
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_private_sidecar_seal_fifo_swap_fails_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = tmp_path / "private-sidecar"
    tree.mkdir()
    member = tree / "dependency.usda"
    member.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    descriptor = os.open(
        tree,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    real_open = facade.os.open
    swapped = False
    held_original_descriptor = -1
    held_original_identity: tuple[int, int] | None = None

    def swap_member_to_fifo(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal held_original_descriptor, held_original_identity, swapped
        if not swapped and path == member.name and dir_fd == descriptor:
            swapped = True
            assert flags & getattr(os, "O_NONBLOCK", 0)
            held_original_descriptor = real_open(
                member.name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            held_metadata = os.fstat(held_original_descriptor)
            held_original_identity = (held_metadata.st_dev, held_metadata.st_ino)
            os.unlink(member.name, dir_fd=descriptor)
            os.mkfifo(member.name, dir_fd=descriptor)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(facade.os, "open", swap_member_to_fifo)
    try:
        with pytest.raises(
            JointRiggerArtifactError,
            match=r"changed inode|invalid entry",
        ):
            facade._seal_directory_descriptor_tree(descriptor)
    finally:
        if held_original_descriptor >= 0:
            os.close(held_original_descriptor)
        os.close(descriptor)
        tree.chmod(0o700)

    assert swapped
    replacement_metadata = member.lstat()
    assert stat.S_ISFIFO(replacement_metadata.st_mode)
    assert held_original_identity is not None
    assert (replacement_metadata.st_dev, replacement_metadata.st_ino) != (
        held_original_identity
    )


def test_private_sidecar_path_replacement_after_revalidation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path, sidecar=True)
    _write_complete_bundle(targets)
    request = _request(source)
    root_text = _publication_sidecar_root_text(targets)
    result = _result(
        request,
        output_text=root_text,
        dependency_bundle_sha256=_expected_sidecar_sha256(
            tmp_path,
            content=_DEPENDENCY_USDA,
        ),
    )
    real_promote_staged_artifacts = facade.promote_staged_artifacts
    displaced_snapshot: Path | None = None

    def replace_private_sidecar(
        promotion: list[StagedArtifact],
        *,
        precommit_validator: Callable[[], None],
    ) -> None:
        nonlocal displaced_snapshot
        sidecar_artifact = next(
            artifact
            for artifact in promotion
            if artifact.label == "composition sidecar"
        )
        assert sidecar_artifact.source_descriptor is not None
        private_sidecar = sidecar_artifact.staged_path
        displaced_snapshot = private_sidecar.with_name(
            f".{private_sidecar.name}.displaced"
        )
        private_sidecar.rename(displaced_snapshot)
        private_sidecar.mkdir()
        (private_sidecar / "dependency.usda").write_text(
            "attacker sidecar",
            encoding="utf-8",
        )
        real_promote_staged_artifacts(
            promotion,
            precommit_validator=precommit_validator,
        )

    monkeypatch.setattr(
        facade,
        "promote_staged_artifacts",
        replace_private_sidecar,
    )

    with pytest.raises(JointRiggerArtifactError, match="Could not publish"):
        author_joint_rig(
            request,
            _WritingBackend(
                result,
                sidecar_content=_DEPENDENCY_USDA,
                author_publication_sidecar_reference=True,
            ),
            targets,
        )

    _assert_complete_bundle(targets)
    assert displaced_snapshot is not None
    assert displaced_snapshot.exists()
    assert targets.sidecar_path is not None
    assert (targets.sidecar_path / "dependency.usda").read_text(
        encoding="utf-8"
    ) == "old sidecar"


def test_private_sidecar_content_mutation_after_revalidation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path, sidecar=True)
    _write_complete_bundle(targets)
    request = _request(source)
    root_text = _publication_sidecar_root_text(targets)
    result = _result(
        request,
        output_text=root_text,
        dependency_bundle_sha256=_expected_sidecar_sha256(
            tmp_path,
            content=_DEPENDENCY_USDA,
        ),
    )
    real_promote_staged_artifacts = facade.promote_staged_artifacts

    def mutate_private_sidecar(
        promotion: list[StagedArtifact],
        *,
        precommit_validator: Callable[[], None],
    ) -> None:
        sidecar_artifact = next(
            artifact
            for artifact in promotion
            if artifact.label == "composition sidecar"
        )
        assert sidecar_artifact.source_descriptor is not None
        member = sidecar_artifact.staged_path / "dependency.usda"
        member.chmod(0o644)
        member.write_text("attacker sidecar", encoding="utf-8")
        member.chmod(0o444)
        real_promote_staged_artifacts(
            promotion,
            precommit_validator=precommit_validator,
        )

    monkeypatch.setattr(
        facade,
        "promote_staged_artifacts",
        mutate_private_sidecar,
    )

    with pytest.raises(JointRiggerArtifactError, match="Could not publish"):
        author_joint_rig(
            request,
            _WritingBackend(
                result,
                sidecar_content=_DEPENDENCY_USDA,
                author_publication_sidecar_reference=True,
            ),
            targets,
        )

    _assert_complete_bundle(targets)
    assert targets.sidecar_path is not None
    assert (targets.sidecar_path / "dependency.usda").read_text(
        encoding="utf-8"
    ) == "old sidecar"
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_private_sidecar_cleanup_survives_parent_rename_and_recreation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    live_parent = tmp_path / "live"
    displaced_parent = tmp_path / "displaced-live"
    live_parent.mkdir()
    targets = _targets(live_parent, sidecar=True)
    _write_complete_bundle(targets)
    request = _request(source)
    root_text = _publication_sidecar_root_text(targets)
    result = _result(
        request,
        output_text=root_text,
        dependency_bundle_sha256=_expected_sidecar_sha256(
            tmp_path,
            content=_DEPENDENCY_USDA,
        ),
    )
    real_promote_staged_artifacts = facade.promote_staged_artifacts
    private_sidecar: Path | None = None

    def swap_sidecar_parent(
        promotion: list[StagedArtifact],
        *,
        precommit_validator: Callable[[], None],
    ) -> None:
        nonlocal private_sidecar
        sidecar_artifact = next(
            artifact
            for artifact in promotion
            if artifact.label == "composition sidecar"
        )
        private_sidecar = sidecar_artifact.staged_path
        assert ".sealed-" in private_sidecar.name
        live_parent.rename(displaced_parent)
        live_parent.mkdir()
        _write_complete_bundle(_targets(live_parent, sidecar=True), marker="unrelated")
        real_promote_staged_artifacts(
            promotion,
            precommit_validator=precommit_validator,
        )

    monkeypatch.setattr(
        facade,
        "promote_staged_artifacts",
        swap_sidecar_parent,
    )

    with pytest.raises(JointRiggerArtifactError, match="Could not publish"):
        author_joint_rig(
            request,
            _WritingBackend(
                result,
                sidecar_content=_DEPENDENCY_USDA,
                author_publication_sidecar_reference=True,
            ),
            targets,
        )

    assert private_sidecar is not None
    assert not (displaced_parent / private_sidecar.name).exists()
    assert not (live_parent / private_sidecar.name).exists()
    _assert_complete_bundle(_targets(displaced_parent, sidecar=True))
    _assert_complete_bundle(_targets(live_parent, sidecar=True), marker="unrelated")
    assert not any(displaced_parent.glob(".*.sealed-*"))
    assert not any(live_parent.glob(".*.sealed-*"))


def test_second_report_snapshot_failure_preserves_primary_over_fatal_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_error = KeyboardInterrupt("forced result snapshot creation failure")
    cleanup_error = SystemExit("forced diagnostics snapshot cleanup failure")
    creation_order: list[str] = []
    cleanup_calls = 0

    class DiagnosticsSnapshot:
        def cleanup(self) -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1
            raise cleanup_error

    diagnostics_snapshot = DiagnosticsSnapshot()

    def create_snapshot(
        path: Path,
        payload: bytes,
        *,
        label: str,
    ) -> object:
        del path, payload
        creation_order.append(label)
        if label == "diagnostics":
            return diagnostics_snapshot
        raise primary_error

    monkeypatch.setattr(facade, "_create_private_report_snapshot", create_snapshot)
    parsed = facade._ParsedModelReport(model=None, payload=b"{}")

    with pytest.raises(KeyboardInterrupt) as raised:
        facade._seal_validated_reports(
            facade._ValidatedReports(result=parsed, diagnostics=parsed),
            _targets(tmp_path),
        )

    assert raised.value is primary_error
    assert creation_order == ["diagnostics", "result"]
    assert cleanup_calls == 1
    assert "SystemExit: forced diagnostics snapshot cleanup failure" in "\n".join(
        raised.value.__notes__
    )


def test_sealed_report_bundle_attempts_later_snapshot_after_fatal_cleanup() -> None:
    cleanup_order: list[str] = []
    primary_error = SystemExit("forced result snapshot cleanup failure")
    later_error = OSError("forced diagnostics snapshot cleanup failure")
    later_error.add_note("nested diagnostics cleanup detail")

    class SnapshotCleanup:
        def __init__(self, label: str, error: BaseException) -> None:
            self.label = label
            self.error = error

        def cleanup(self) -> None:
            cleanup_order.append(self.label)
            raise self.error

    bundle = facade._SealedReportSnapshots(
        result=SnapshotCleanup("result", primary_error),
        diagnostics=SnapshotCleanup("diagnostics", later_error),
    )

    with pytest.raises(SystemExit) as raised:
        bundle.cleanup()

    assert raised.value is primary_error
    assert cleanup_order == ["result", "diagnostics"]
    notes = "\n".join(raised.value.__notes__)
    assert "OSError: forced diagnostics snapshot cleanup failure" in notes
    assert "nested diagnostics cleanup detail" in notes


def test_sealed_generated_bundle_attempts_every_cleanup_after_fatal() -> None:
    cleanup_order: list[str] = []
    primary_error = KeyboardInterrupt("forced sidecar cleanup failure")

    class SnapshotCleanup:
        def __init__(
            self,
            label: str,
            error: BaseException | None = None,
        ) -> None:
            self.label = label
            self.path = Path(label)
            self.error = error

        def cleanup(self) -> None:
            cleanup_order.append(self.label)
            if self.error is not None:
                raise self.error

    bundle = facade._SealedGeneratedArtifacts(
        root=SnapshotCleanup("root"),
        sidecar=SnapshotCleanup("sidecar", primary_error),
        dependencies=(SnapshotCleanup("dependency"),),
    )

    with pytest.raises(KeyboardInterrupt) as raised:
        bundle.cleanup()

    assert raised.value is primary_error
    assert cleanup_order == ["sidecar", "dependency", "root"]


def test_sidecar_snapshot_failure_preserves_primary_over_fatal_root_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    primary_error = KeyboardInterrupt("forced sidecar snapshot creation failure")
    cleanup_error = SystemExit("forced generated-root cleanup failure")
    root_cleanup_calls = 0

    class RootSnapshot:
        def cleanup(self) -> None:
            nonlocal root_cleanup_calls
            root_cleanup_calls += 1
            raise cleanup_error

    def create_root(path: Path, *, expected_sha256: str) -> object:
        del path, expected_sha256
        return RootSnapshot()

    def fail_sidecar(
        source: Path,
        *,
        private_parent: Path,
        expected_sha256: str,
    ) -> object:
        del source, private_parent, expected_sha256
        raise primary_error

    monkeypatch.setattr(facade, "_seal_generated_root", create_root)
    monkeypatch.setattr(facade, "_create_private_sidecar_snapshot", fail_sidecar)

    with pytest.raises(KeyboardInterrupt) as raised:
        facade._seal_generated_artifacts(
            _result(request),
            _targets(tmp_path, sidecar=True),
        )

    assert raised.value is primary_error
    assert root_cleanup_calls == 1
    assert "SystemExit: forced generated-root cleanup failure" in "\n".join(
        raised.value.__notes__
    )


def test_cleanup_authoring_state_attempts_all_owners_after_fatal() -> None:
    cleanup_order: list[str] = []
    primary_error = SystemExit("forced sealed report cleanup failure")

    class CleanupOwner:
        def __init__(
            self,
            label: str,
            error: BaseException | None = None,
        ) -> None:
            self.label = label
            self.error = error

        def cleanup(self) -> None:
            cleanup_order.append(self.label)
            if self.error is not None:
                raise self.error

    sealed_reports = CleanupOwner("reports", primary_error)
    sealed_generated = CleanupOwner(
        "generated",
        KeyboardInterrupt("forced generated cleanup failure"),
    )
    staged = CleanupOwner("staged", OSError("forced staged cleanup failure"))

    with pytest.raises(SystemExit) as raised:
        facade._cleanup_authoring_state(
            sealed_reports,
            sealed_generated,
            staged,
        )

    assert raised.value is primary_error
    assert cleanup_order == ["reports", "generated", "staged"]
    notes = "\n".join(raised.value.__notes__)
    assert "KeyboardInterrupt: forced generated cleanup failure" in notes
    assert "OSError: forced staged cleanup failure" in notes


def test_cleanup_authoring_state_prioritizes_later_fatal_over_normal_error() -> None:
    cleanup_order: list[str] = []
    earlier_error = OSError("forced earlier normal cleanup failure")
    fatal_error = SystemExit("forced later fatal cleanup failure")

    class CleanupOwner:
        def __init__(
            self,
            label: str,
            error: BaseException | None = None,
        ) -> None:
            self.label = label
            self.error = error

        def cleanup(self) -> None:
            cleanup_order.append(self.label)
            if self.error is not None:
                raise self.error

    with pytest.raises(SystemExit) as raised:
        facade._cleanup_authoring_state(
            CleanupOwner("reports", earlier_error),
            CleanupOwner("generated", fatal_error),
            CleanupOwner("staged"),
        )

    assert raised.value is fatal_error
    assert cleanup_order == ["reports", "generated", "staged"]
    assert "OSError: forced earlier normal cleanup failure" in "\n".join(
        raised.value.__notes__
    )


def test_author_finalizer_preserves_active_primary_over_fatal_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path)
    primary_error = KeyboardInterrupt("forced backend termination")
    cleanup_error = SystemExit("forced author finalizer cleanup failure")
    cleanup_calls = 0

    def fail_cleanup(
        sealed_reports: facade._SealedReportSnapshots | None,
        sealed_generated: facade._SealedGeneratedArtifacts | None,
        staged: StagedJointRiggerArtifacts,
    ) -> None:
        nonlocal cleanup_calls
        del sealed_reports, sealed_generated, staged
        cleanup_calls += 1
        raise cleanup_error

    monkeypatch.setattr(facade, "_cleanup_authoring_state", fail_cleanup)

    with pytest.raises(KeyboardInterrupt) as raised:
        author_joint_rig(
            request,
            _WritingBackend(_result(request), author_error=primary_error),
            targets,
        )

    assert raised.value is primary_error
    assert cleanup_calls == 1
    assert "SystemExit: forced author finalizer cleanup failure" in "\n".join(
        raised.value.__notes__
    )


def test_post_commit_cleanup_failure_carries_committed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    result = _result(request)
    targets = _targets(tmp_path)
    real_cleanup_authoring_state = facade._cleanup_authoring_state

    def fail_after_cleanup(
        sealed_reports: facade._SealedReportSnapshots | None,
        sealed_generated: facade._SealedGeneratedArtifacts | None,
        staged: StagedJointRiggerArtifacts,
    ) -> None:
        real_cleanup_authoring_state(sealed_reports, sealed_generated, staged)
        raise JointRiggerArtifactError("forced post-commit cleanup failure")

    monkeypatch.setattr(
        facade,
        "_cleanup_authoring_state",
        fail_after_cleanup,
    )

    with pytest.raises(JointRiggerPostCommitCleanupError) as exc_info:
        author_joint_rig(request, _WritingBackend(result), targets)

    error = exc_info.value
    assert error.committed is True
    assert error.committed_result == result
    assert error.result == result
    assert isinstance(error.cleanup_error, JointRiggerArtifactError)
    assert not isinstance(error, JointRiggerArtifactError)
    assert "committed=True" in str(error)
    assert (
        JointRiggerResultV1.model_validate_json(targets.result_path.read_bytes())
        == result
    )
    assert (
        JointRiggerDiagnosticsV1.model_validate_json(
            targets.diagnostics_path.read_bytes()
        )
        == result.diagnostics
    )
    assert targets.output_path.read_text(encoding="utf-8") == _EMPTY_USDA


def test_committed_promoter_cleanup_failure_maps_to_post_commit_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    result = _result(request)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    real_promote_staged_artifacts = facade.promote_staged_artifacts
    promoter_cleanup_error = OSError("forced committed promoter cleanup failure")

    def commit_then_fail_cleanup(
        promotion: list[StagedArtifact],
        *,
        precommit_validator: Callable[[], None],
    ) -> None:
        real_promote_staged_artifacts(
            promotion,
            precommit_validator=precommit_validator,
        )
        raise CommittedArtifactPublicationCleanupError((promoter_cleanup_error,))

    monkeypatch.setattr(
        facade,
        "promote_staged_artifacts",
        commit_then_fail_cleanup,
    )

    with pytest.raises(JointRiggerPostCommitCleanupError) as exc_info:
        author_joint_rig(request, _WritingBackend(result), targets)

    error = exc_info.value
    assert error.committed is True
    assert error.committed_result == result
    assert error.result == result
    assert isinstance(
        error.cleanup_error,
        CommittedArtifactPublicationCleanupError,
    )
    assert error.cleanup_error.cleanup_errors == (promoter_cleanup_error,)
    assert not isinstance(error, JointRiggerArtifactError)
    assert "committed=True" in str(error)
    assert (
        JointRiggerResultV1.model_validate_json(targets.result_path.read_bytes())
        == result
    )
    assert (
        JointRiggerDiagnosticsV1.model_validate_json(
            targets.diagnostics_path.read_bytes()
        )
        == result.diagnostics
    )
    assert targets.output_path.read_text(encoding="utf-8") == _EMPTY_USDA


def test_precommit_publication_failure_remains_artifact_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    result = _result(request)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    def fail_before_commit(
        promotion: list[StagedArtifact],
        *,
        precommit_validator: Callable[[], None],
    ) -> None:
        del promotion, precommit_validator
        raise RuntimeError("forced precommit publication failure")

    monkeypatch.setattr(
        facade,
        "promote_staged_artifacts",
        fail_before_commit,
    )

    with pytest.raises(JointRiggerArtifactError) as exc_info:
        author_joint_rig(request, _WritingBackend(result), targets)

    assert not isinstance(exc_info.value, JointRiggerPostCommitCleanupError)
    assert not getattr(exc_info.value, "committed", False)
    assert "forced precommit publication failure" in str(exc_info.value)
    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_missing_generated_root_cannot_publish_complete_reports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path)
    request = _request(source)
    result = _result(request)
    backend = _WritingBackend(result, write_output=False)

    with pytest.raises(JointRiggerArtifactError, match="generated root"):
        author_joint_rig(request, backend, targets)

    assert not targets.output_path.exists()
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()


def test_configured_sidecar_must_be_complete_before_root_publication(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path, sidecar=True)
    _write_complete_bundle(targets)
    request = _request(source)
    result = _result(
        request,
        dependency_bundle_sha256=_expected_sidecar_sha256(tmp_path),
    )
    backend = _WritingBackend(result, write_sidecar=False)

    with pytest.raises(JointRiggerArtifactError, match="composition sidecar"):
        author_joint_rig(request, backend, targets)

    _assert_complete_bundle(targets)


def test_configured_sidecar_requires_declared_dependency_bundle_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path, sidecar=True)
    _write_complete_bundle(targets)
    request = _request(source)

    with pytest.raises(JointRiggerArtifactError, match="must claim"):
        author_joint_rig(
            request,
            _WritingBackend(_result(request, dependency_bundle_sha256=None)),
            targets,
        )

    _assert_complete_bundle(targets)


def test_sidecar_dependency_bundle_identity_must_match_staged_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path, sidecar=True)
    _write_complete_bundle(targets)
    request = _request(source)
    result = _result(request, dependency_bundle_sha256="0" * 64)

    with pytest.raises(JointRiggerArtifactError, match="does not match"):
        author_joint_rig(request, _WritingBackend(result), targets)

    _assert_complete_bundle(targets)


def test_sidecar_mutation_after_declared_digest_is_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path, sidecar=True)
    _write_complete_bundle(targets)
    request = _request(source)
    result = _result(
        request,
        dependency_bundle_sha256=_expected_sidecar_sha256(tmp_path),
    )

    with pytest.raises(JointRiggerArtifactError, match="does not match"):
        author_joint_rig(
            request,
            _WritingBackend(result, sidecar_content="mutated dependency"),
            targets,
        )

    _assert_complete_bundle(targets)


def test_no_sidecar_dependency_identity_is_verified_from_staged_usd(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    expected = tmp_path / "expected.usda"
    output_text = '#usda 1.0\n\ndef Xform "Rigged" {}\n'
    expected.write_text(output_text, encoding="utf-8")
    expected_identity = identify_usd_artifact(
        expected,
        uri="memory://generated.usda",
    )
    request = _request(source)
    result = _result(
        request,
        output_text=output_text,
        dependency_bundle_sha256=expected_identity.dependency_bundle_sha256,
    )

    observed = author_joint_rig(
        request,
        _WritingBackend(result, output_text=output_text),
        _targets(tmp_path),
    )

    assert observed == result


def test_no_sidecar_identity_publishes_transitive_opaque_dependencies(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    main = tmp_path / "Main.mdl"
    peer = tmp_path / "Peer.mdl"
    texture = tmp_path / "albedo.png"
    main.write_text("mdl 1.7;\nimport Peer::*;\n", encoding="utf-8")
    peer.write_text(
        'mdl 1.7;\nimport ::df::*;\ntexture_2d("albedo.png");\n',
        encoding="utf-8",
    )
    texture.write_bytes(b"real texture bytes")
    output_text = _root_with_asset_reference("Main.mdl")
    expected = tmp_path / "expected.usda"
    expected.write_text(output_text, encoding="utf-8")
    expected_identity = identify_usd_artifact(
        expected,
        uri="memory://generated.usda",
    )
    request = _request(source)
    result = _result(
        request,
        output_text=output_text,
        dependency_bundle_sha256=expected_identity.dependency_bundle_sha256,
    )

    observed = author_joint_rig(
        request,
        _WritingBackend(result, output_text=output_text),
        _targets(tmp_path),
    )

    assert observed == result
    assert set(
        reference_module.local_usd_dependency_paths(tmp_path / "rigged.usda")
    ) == {
        (tmp_path / "rigged.usda").resolve(),
        main.resolve(),
        peer.resolve(),
        texture.resolve(),
    }


def test_no_sidecar_opaque_dependency_mutation_rolls_back_publication(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    main = tmp_path / "Main.mdl"
    peer = tmp_path / "Peer.mdl"
    main.write_text("mdl 1.7;\nimport Peer::*;\n", encoding="utf-8")
    peer.write_text("mdl 1.7;\nimport ::df::*;\n", encoding="utf-8")
    output_text = _root_with_asset_reference("Main.mdl")
    expected = tmp_path / "expected.usda"
    expected.write_text(output_text, encoding="utf-8")
    expected_identity = identify_usd_artifact(
        expected,
        uri="memory://generated.usda",
    )
    request = _request(source)
    result = _result(
        request,
        output_text=output_text,
        dependency_bundle_sha256=expected_identity.dependency_bundle_sha256,
    )
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    class MutatingBackend(_WritingBackend):
        def author(
            self,
            request: JointRiggerInputV1,
            artifact_targets: JointRiggerArtifactTargets,
        ) -> JointRiggerResultV1:
            observed = super().author(request, artifact_targets)
            peer.write_text(
                "mdl 1.7;\nimport ::df::*;\n// changed bytes\n",
                encoding="utf-8",
            )
            return observed

    with pytest.raises(
        JointRiggerArtifactError,
        match="dependency_bundle_sha256 does not match",
    ):
        author_joint_rig(
            request,
            MutatingBackend(result, output_text=output_text),
            targets,
        )

    _assert_complete_bundle(targets)


def test_sealed_root_composition_rejects_projection_only_safe_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    outside = tmp_path / "outside.usda"
    outside.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    unsafe_output = _root_with_reference(str(outside))
    safe_output = _EMPTY_USDA
    safe_path = tmp_path / "safe.usda"
    safe_path.write_text(safe_output, encoding="utf-8")
    safe_identity = identify_usd_artifact(
        safe_path,
        uri="memory://generated.usda",
    )
    request = _request(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    result = _result(
        request,
        output_text=unsafe_output,
        dependency_bundle_sha256=safe_identity.dependency_bundle_sha256,
    )

    class HeldRootWriterBackend(_WritingBackend):
        descriptor: int | None = None

        def author(
            self,
            candidate: JointRiggerInputV1,
            artifact_targets: JointRiggerArtifactTargets,
        ) -> JointRiggerResultV1:
            observed = super().author(candidate, artifact_targets)
            self.descriptor = os.open(
                artifact_targets.output_path,
                os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
            )
            return observed

    backend = HeldRootWriterBackend(result, output_text=unsafe_output)
    real_project = facade._project_no_sidecar_output_identity
    projection_swapped = False

    def project_safe_root_then_restore(
        staged_artifacts: StagedJointRiggerArtifacts,
        *,
        uri: str,
    ) -> ArtifactIdentityV1:
        nonlocal projection_swapped
        projection_swapped = True
        assert backend.descriptor is not None
        os.ftruncate(backend.descriptor, 0)
        os.pwrite(backend.descriptor, safe_output.encode(), 0)
        os.fsync(backend.descriptor)
        try:
            return real_project(staged_artifacts, uri=uri)
        finally:
            os.ftruncate(backend.descriptor, 0)
            os.pwrite(backend.descriptor, unsafe_output.encode(), 0)
            os.fsync(backend.descriptor)

    monkeypatch.setattr(
        facade,
        "_project_no_sidecar_output_identity",
        project_safe_root_then_restore,
    )
    try:
        with pytest.raises(
            JointRiggerArtifactError,
            match="changed dependency identity",
        ):
            author_joint_rig(request, backend, targets)
    finally:
        if backend.descriptor is not None:
            os.close(backend.descriptor)

    assert projection_swapped
    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.sealed-*"))


@pytest.mark.parametrize("reference_mode", ["staging_name", "publication_name"])
def test_no_sidecar_name_dependent_closure_is_rejected_before_publication(
    tmp_path: Path,
    reference_mode: Literal["staging_name", "publication_name"],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path)
    old_output = '#usda 1.0\n\ndef Xform "Existing" {}\n'
    _write_complete_bundle(targets)
    targets.output_path.write_text(old_output, encoding="utf-8")
    backend = _ProjectedNoSidecarBackend(reference_mode)

    with pytest.raises(
        JointRiggerArtifactError,
        match="publication projection|existing publication root",
    ) as raised:
        author_joint_rig(request, backend, targets)

    assert backend.probed is True
    assert targets.output_path.read_text(encoding="utf-8") == old_output
    assert targets.diagnostics_path.read_text(encoding="utf-8") == "old diagnostics"
    assert targets.result_path.read_text(encoding="utf-8") == "old result"
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()
    assert not any(tmp_path.glob(".*.validate-*"))


def test_no_sidecar_projection_collapses_dot_segments_inside_isolated_tree(
    tmp_path: Path,
) -> None:
    publication_directory = tmp_path / "publication"
    publication_directory.mkdir()
    published_output = publication_directory / "rigged.usda"
    published_output.write_text("old output", encoding="utf-8")
    dot_segment_output = Path(
        "/" + "../" * 64 + published_output.as_posix().lstrip("/")
    )
    assert dot_segment_output.resolve(strict=False) == published_output

    staged_output = tmp_path / ".rigged.stage.usda"
    staged_output.write_text(_EMPTY_USDA, encoding="utf-8")
    final_targets = JointRiggerArtifactTargets(
        output_path=dot_segment_output,
        diagnostics_path=publication_directory / "diagnostics.json",
        result_path=publication_directory / "result.json",
    )
    staged_targets = JointRiggerArtifactTargets(
        output_path=staged_output,
        diagnostics_path=tmp_path / ".diagnostics.stage.json",
        result_path=tmp_path / ".result.stage.json",
        publication_output_path=dot_segment_output,
    )

    identity = facade._project_no_sidecar_output_identity(
        StagedJointRiggerArtifacts(final_targets, staged_targets),
        uri="memory://generated.usda",
    )

    assert identity.root_sha256 == hashlib.sha256(_EMPTY_USDA.encode()).hexdigest()
    assert published_output.read_text(encoding="utf-8") == "old output"
    assert not any(publication_directory.glob(".*.validate-*"))


@pytest.mark.parametrize(
    "callsite",
    [
        "sealed_sidecar",
        "sealed_no_sidecar",
        "staged_sidecar",
        "project_no_sidecar",
    ],
)
def test_projection_owner_substitution_preserves_foreign_and_displaced_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    callsite: Literal[
        "sealed_sidecar",
        "sealed_no_sidecar",
        "staged_sidecar",
        "project_no_sidecar",
    ],
) -> None:
    staged_output = tmp_path / "staged.usda"
    staged_output.write_text(_EMPTY_USDA, encoding="utf-8")
    final_targets = _targets(
        tmp_path,
        sidecar=callsite in {"sealed_sidecar", "staged_sidecar"},
    )
    staged_sidecar = (
        tmp_path / "staged-assets"
        if callsite in {"sealed_sidecar", "staged_sidecar"}
        else None
    )
    if staged_sidecar is not None:
        staged_sidecar.mkdir()
    staged_targets = JointRiggerArtifactTargets(
        output_path=staged_output,
        diagnostics_path=tmp_path / "staged-diagnostics.json",
        result_path=tmp_path / "staged-result.json",
        sidecar_path=staged_sidecar,
        publication_output_path=final_targets.output_path,
        publication_sidecar_path=final_targets.sidecar_path,
    )
    staged_artifacts = StagedJointRiggerArtifacts(final_targets, staged_targets)
    cleanup_steps: list[Callable[[], None]] = []

    if callsite == "staged_sidecar":

        def invoke() -> object:
            return facade._validate_staged_sidecar_composition(staged_targets)

    elif callsite == "project_no_sidecar":

        def invoke() -> object:
            return facade._project_no_sidecar_output_identity(
                staged_artifacts,
                uri="memory://generated.usda",
            )

    else:
        root = facade._seal_generated_root(
            staged_output,
            expected_sha256=hashlib.sha256(staged_output.read_bytes()).hexdigest(),
        )
        if callsite == "sealed_sidecar":
            assert staged_sidecar is not None
            sidecar = facade._create_private_sidecar_snapshot(
                staged_sidecar,
                private_parent=tmp_path,
                expected_sha256=sidecar_dependency_bundle_sha256(staged_sidecar),
            )
            sealed = facade._SealedGeneratedArtifacts(root=root, sidecar=sidecar)

            def invoke() -> object:
                return facade._validate_sealed_sidecar_composition(
                    sealed,
                    staged_targets,
                )

            cleanup_steps.append(sealed.cleanup)
        else:
            dependencies, records = facade._capture_sealed_no_sidecar_dependencies(
                root,
                staged_artifacts,
            )
            identity = identify_usd_artifact(
                staged_output,
                uri="memory://generated.usda",
            )
            assert identity.dependency_bundle_sha256 is not None

            def invoke() -> object:
                return facade._validate_sealed_no_sidecar_composition(
                    root,
                    dependencies,
                    records,
                    staged_artifacts,
                    uri="memory://generated.usda",
                    expected_bundle_sha256=identity.dependency_bundle_sha256,
                )

            cleanup_steps.extend([dependency.cleanup for dependency in dependencies])
            cleanup_steps.append(root.cleanup)

    primary = KeyboardInterrupt(f"forced {callsite} owner substitution")
    real_create_owner = facade._create_private_directory_owner
    real_owner_context = facade._private_directory_owner
    owner_descriptors: list[tuple[int, int]] = []
    foreign_paths: list[Path] = []
    displaced_paths: list[Path] = []

    def track_create_owner(
        parent_path: Path,
        *,
        prefix: str,
    ) -> facade._PrivateDirectoryOwner:
        owner = real_create_owner(parent_path, prefix=prefix)
        owner_descriptors.append((owner.parent_descriptor, owner.source_descriptor))
        return owner

    @contextmanager
    def substitute_owner(parent_path: Path, *, prefix: str) -> Iterator[Path]:
        with real_owner_context(parent_path, prefix=prefix) as owner_path:
            yield owner_path
            displaced = owner_path.with_name(f"{owner_path.name}-displaced")
            owner_path.rename(displaced)
            owner_path.mkdir()
            (owner_path / "foreign.txt").write_text("foreign", encoding="utf-8")
            foreign_paths.append(owner_path)
            displaced_paths.append(displaced)
            raise primary

    monkeypatch.setattr(facade, "_create_private_directory_owner", track_create_owner)
    monkeypatch.setattr(facade, "_private_directory_owner", substitute_owner)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            invoke()
    finally:
        for cleanup in cleanup_steps:
            cleanup()

    assert raised.value is primary
    assert "replacement preserved" in "\n".join(raised.value.__notes__)
    assert len(owner_descriptors) == 1
    assert len(foreign_paths) == 1
    assert len(displaced_paths) == 1
    assert (foreign_paths[0] / "foreign.txt").read_text(encoding="utf-8") == ("foreign")
    assert displaced_paths[0].is_dir()
    for parent_descriptor, source_descriptor in owner_descriptors:
        with pytest.raises(OSError):
            os.fstat(source_descriptor)
        with pytest.raises(OSError):
            os.fstat(parent_descriptor)


@pytest.mark.parametrize(
    "dependency_target",
    [
        "staged_diagnostics",
        "staged_result",
        "final_diagnostics",
        "final_result",
    ],
)
def test_no_sidecar_dependencies_must_not_overlap_transaction_targets(
    tmp_path: Path,
    dependency_target: Literal[
        "staged_diagnostics",
        "staged_result",
        "final_diagnostics",
        "final_result",
    ],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    backend = _TransactionDependencyBackend(dependency_target, targets)

    with pytest.raises(
        JointRiggerArtifactError,
        match="transaction target",
    ) as raised:
        author_joint_rig(request, backend, targets)

    assert backend.probed is True
    _assert_complete_bundle(targets)
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()
    assert not any(tmp_path.glob(".*.validate-*"))


@pytest.mark.parametrize("reference_mode", ["sibling", "absolute"])
def test_no_sidecar_publication_projection_preserves_stable_dependencies(
    tmp_path: Path,
    reference_mode: Literal["sibling", "absolute"],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    dependency = tmp_path / "dependency.usda"
    dependency.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    request = _request(source)
    targets = _targets(tmp_path)
    backend = _ProjectedNoSidecarBackend(reference_mode, dependency)

    result = author_joint_rig(request, backend, targets)

    assert result.output_artifact is not None
    published_identity = identify_usd_artifact(
        targets.output_path,
        uri=result.output_artifact.uri,
    )
    assert result.output_artifact == published_identity
    assert dependency in facade._local_usd_dependency_paths(
        targets.output_path,
        label="published output",
    )
    assert not any(tmp_path.glob(".*.stage-*"))
    assert not any(tmp_path.glob(".*.validate-*"))


def test_no_sidecar_owned_publication_keeps_tilde_dependency_literal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    literal_directory = tmp_path / "~"
    literal_directory.mkdir()
    literal_dependency = literal_directory / "dep.bin"
    literal_dependency.write_bytes(b"literal authored dependency")

    home = tmp_path / "home"
    home.mkdir()
    unrelated_dependency = tmp_path / "unrelated.bin"
    unrelated_dependency.write_bytes(b"unrelated home dependency")
    home_dependency = home / "dep.bin"
    home_dependency.symlink_to(unrelated_dependency)
    monkeypatch.setenv("HOME", str(home))

    inspected_locators: list[Path] = []
    normalize_without_symlinks = facade._normalize_local_path_without_symlinks

    def record_final_dependency_guard(
        path: Path,
        *,
        symlink_error: str,
    ) -> Path:
        if symlink_error.startswith("Generated root dependency path"):
            inspected_locators.append(path)
        return normalize_without_symlinks(path, symlink_error=symlink_error)

    monkeypatch.setattr(
        facade,
        "_normalize_local_path_without_symlinks",
        record_final_dependency_guard,
    )
    request = _request(source)
    targets = _targets(tmp_path)
    backend = _ProjectedNoSidecarBackend(
        "sibling",
        literal_dependency,
        authored_reference="~/dep.bin",
        dependency_kind="asset",
    )

    result = author_joint_rig(request, backend, targets)

    assert result.output_artifact is not None
    assert literal_dependency in facade._local_usd_dependency_paths(
        targets.output_path,
        label="published output",
    )
    assert literal_dependency in inspected_locators
    assert home_dependency not in inspected_locators
    assert home_dependency.is_symlink()
    assert not any(tmp_path.glob(".*.stage-*"))
    assert not any(tmp_path.glob(".*.validate-*"))


def test_no_sidecar_package_relative_dependencies_are_rejected(
    tmp_path: Path,
) -> None:
    from pxr import UsdUtils

    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    package_source = tmp_path / "package-source"
    package_source.mkdir()
    package_root = package_source / "root.usda"
    package_root.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    package = tmp_path / "dependency.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(package_root), str(package))
    output_text = _root_with_reference(f"{package}[root.usda]")
    identity_root = tmp_path / "identity-root.usda"
    identity_root.write_text(output_text, encoding="utf-8")
    claimed_identity = identify_usd_artifact(
        identity_root,
        uri="memory://generated.usda",
    )
    request = _request(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    result = _result(
        request,
        output_text=output_text,
        dependency_bundle_sha256=claimed_identity.dependency_bundle_sha256,
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="does not support package-relative USD dependencies",
    ):
        author_joint_rig(
            request,
            _WritingBackend(result, output_text=output_text),
            targets,
        )

    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_self_contained_usdz_package_dependencies_are_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf, Usd, UsdUtils

    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)

    package_source = tmp_path / "package-source"
    package_source.mkdir()
    package_root = package_source / "root.usda"
    texture = package_source / "texture.png"
    texture.write_bytes(b"package-internal texture")
    stage = Usd.Stage.CreateNew(str(package_root))
    root_prim = stage.DefinePrim("/Generated")
    stage.SetDefaultPrim(root_prim)
    asset = root_prim.CreateAttribute(
        "test:asset",
        Sdf.ValueTypeNames.Asset,
        custom=True,
    )
    assert asset.Set(Sdf.AssetPath("texture.png"))
    assert stage.GetRootLayer().Save()
    del stage
    package = tmp_path / "generated-source.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(package_root), str(package))

    targets = JointRiggerArtifactTargets(
        output_path=tmp_path / "rigged.usdz",
        diagnostics_path=tmp_path / "diagnostics.json",
        result_path=tmp_path / "result.json",
    )
    expected_identity = identify_usd_artifact(
        package,
        uri="memory://generated.usdz",
    )
    base_result = _result(request)
    result = base_result.model_copy(update={"output_artifact": expected_identity})

    class PackageBackend:
        name = "package-backend"

        def probe(self, candidate: JointRiggerInputV1) -> None:
            assert candidate is request

        def author(
            self,
            candidate: JointRiggerInputV1,
            artifact_targets: JointRiggerArtifactTargets,
        ) -> JointRiggerResultV1:
            assert candidate is request
            shutil.copyfile(package, artifact_targets.output_path)
            artifact_targets.diagnostics_path.write_text(
                result.diagnostics.model_dump_json(),
                encoding="utf-8",
            )
            artifact_targets.result_path.write_text(
                result.model_dump_json(),
                encoding="utf-8",
            )
            return result

    validation_calls = 0
    validate_package = facade._validate_sealed_usdz_composition

    def count_package_validation(*args: Any, **kwargs: Any) -> ArtifactIdentityV1:
        nonlocal validation_calls
        validation_calls += 1
        return validate_package(*args, **kwargs)

    monkeypatch.setattr(
        facade,
        "_validate_sealed_usdz_composition",
        count_package_validation,
    )

    observed = author_joint_rig(request, PackageBackend(), targets)

    assert observed == result
    assert validation_calls == 1
    assert targets.output_path.read_bytes() == package.read_bytes()
    assert (
        identify_usd_artifact(
            targets.output_path,
            uri=result.output_artifact.uri,
        )
        == result.output_artifact
    )
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_sealed_usdz_rejects_defensive_external_dependency_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import UsdUtils

    package_root = tmp_path / "root.usda"
    package_root.write_text(_EMPTY_USDA, encoding="utf-8")
    package = tmp_path / "generated.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(package_root), str(package))
    package.chmod(0o444)
    descriptor = os.open(package, os.O_RDONLY)
    metadata = os.fstat(descriptor)
    sealed_root = facade._SealedGeneratedRoot(
        path=package,
        source_descriptor=descriptor,
        source_identity=(metadata.st_dev, metadata.st_ino),
        source_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
        source_mode=stat.S_IMODE(metadata.st_mode),
    )
    targets = JointRiggerArtifactTargets(
        output_path=tmp_path / "published.usdz",
        diagnostics_path=tmp_path / "diagnostics.json",
        result_path=tmp_path / "result.json",
    )
    external = tmp_path / "external.usda"
    external.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    monkeypatch.setattr(
        source_binding_module,
        "_validate_bound_projection_dependencies",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        reference_module,
        "local_usd_dependency_paths",
        lambda path: (path, external),
    )

    try:
        with pytest.raises(
            JointRiggerArtifactError,
            match="dependencies outside its archive",
        ):
            facade._validate_sealed_usdz_composition(
                sealed_root,
                targets,
                uri="memory://generated.usdz",
            )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("cleanup_error", "expected_error"),
    [
        (
            OSError("forced discovery cleanup failure"),
            JointRiggerArtifactError,
        ),
        (
            SystemExit("forced discovery cleanup failure"),
            SystemExit,
        ),
    ],
)
def test_no_sidecar_discovery_cleanup_failure_closes_retained_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_error: BaseException,
    expected_error: type[BaseException],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    dependency = tmp_path / "dependency.usda"
    dependency.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    request = _request(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    backend = _ProjectedNoSidecarBackend("sibling", dependency)
    real_open_snapshot = facade._open_sealed_dependency_snapshot
    real_snapshot_cleanup = facade._SealedDependencySnapshot.cleanup
    real_remove = facade._remove_descriptor_entry
    real_unlink = Path.unlink
    snapshots: list[facade._SealedDependencySnapshot] = []
    cleanup_counts: dict[int, int] = {}
    discovery_unlinks = 0
    leaked_discovery: Path | None = None

    def track_snapshot(path: Path) -> facade._SealedDependencySnapshot:
        snapshot = real_open_snapshot(path)
        snapshots.append(snapshot)
        return snapshot

    def track_snapshot_cleanup(
        snapshot: facade._SealedDependencySnapshot,
    ) -> None:
        identity = id(snapshot)
        cleanup_counts[identity] = cleanup_counts.get(identity, 0) + 1
        real_snapshot_cleanup(snapshot)

    def fail_final_discovery_removal(*args: object, **kwargs: object) -> None:
        nonlocal discovery_unlinks, leaked_discovery
        label = str(kwargs.get("label"))
        if "sealed dependency discovery" in label:
            discovery_unlinks += 1
            if discovery_unlinks == 2:
                leaked_discovery = tmp_path / str(args[1])
                raise cleanup_error
        real_remove(*args, **kwargs)

    monkeypatch.setattr(
        facade,
        "_open_sealed_dependency_snapshot",
        track_snapshot,
    )
    monkeypatch.setattr(
        facade._SealedDependencySnapshot,
        "cleanup",
        track_snapshot_cleanup,
    )
    monkeypatch.setattr(
        facade,
        "_remove_descriptor_entry",
        fail_final_discovery_removal,
    )
    try:
        expected_message = (
            "sealed dependency discovery artifact.*forced discovery cleanup"
            if isinstance(cleanup_error, Exception)
            else "forced discovery cleanup failure"
        )
        with pytest.raises(expected_error, match=expected_message) as raised:
            author_joint_rig(request, backend, targets)
    finally:
        if leaked_discovery is not None:
            real_unlink(leaked_discovery, missing_ok=True)

    if not isinstance(cleanup_error, Exception):
        assert raised.value is cleanup_error
    assert discovery_unlinks == 2
    assert snapshots
    assert all(snapshot.source_descriptor == -1 for snapshot in snapshots)
    assert cleanup_counts == {id(snapshot): 1 for snapshot in snapshots}
    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_no_sidecar_identity_rejects_live_parent_swap_around_identify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    live_parent = tmp_path / "dependencies"
    alternate_parent = tmp_path / "alternate-dependencies"
    displaced_parent = tmp_path / "displaced-dependencies"
    live_parent.mkdir()
    alternate_parent.mkdir()
    dependency = live_parent / "dependency.usda"
    dependency.write_text(
        '#usda 1.0\n\ndef Xform "RetainedA" {}\n',
        encoding="utf-8",
    )
    (alternate_parent / dependency.name).write_text(
        '#usda 1.0\n\ndef Xform "ClaimedB" {}\n',
        encoding="utf-8",
    )
    output_text = _root_with_reference(str(dependency))
    identity_root = tmp_path / "identity-root.usda"
    identity_root.write_text(output_text, encoding="utf-8")
    live_parent.rename(displaced_parent)
    alternate_parent.rename(live_parent)
    try:
        claimed_identity = identify_usd_artifact(
            identity_root,
            uri="memory://generated.usda",
        )
    finally:
        live_parent.rename(alternate_parent)
        displaced_parent.rename(live_parent)
    retained_identity = identify_usd_artifact(
        identity_root,
        uri="memory://generated.usda",
    )
    assert claimed_identity.dependency_bundle_sha256 != (
        retained_identity.dependency_bundle_sha256
    )

    request = _request(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    result = _result(
        request,
        output_text=output_text,
        dependency_bundle_sha256=claimed_identity.dependency_bundle_sha256,
    )
    backend = _WritingBackend(result, output_text=output_text)
    real_identify = reference_module.identify_usd_artifact
    swap_count = 0

    def identify_with_alternate_parent(
        path: str | Path,
        *,
        uri: str,
    ) -> ArtifactIdentityV1:
        nonlocal swap_count
        if (
            backend.received_targets is not None
            and Path(path) == backend.received_targets.output_path
        ):
            swap_count += 1
            live_parent.rename(displaced_parent)
            alternate_parent.rename(live_parent)
            try:
                return real_identify(path, uri=uri)
            finally:
                live_parent.rename(alternate_parent)
                displaced_parent.rename(live_parent)
        return real_identify(path, uri=uri)

    monkeypatch.setattr(
        reference_module,
        "identify_usd_artifact",
        identify_with_alternate_parent,
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="records changed dependency identity",
    ):
        author_joint_rig(request, backend, targets)

    assert swap_count == 1
    assert dependency.read_text(encoding="utf-8") == (
        '#usda 1.0\n\ndef Xform "RetainedA" {}\n'
    )
    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_no_sidecar_dependency_symlink_swap_at_promoter_entry_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    dependency = tmp_path / "dependency.usda"
    dependency.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    identical = tmp_path / "identical.usda"
    identical.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    displaced = tmp_path / "dependency.original.usda"
    request = _request(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    real_promote_staged_artifacts = facade.promote_staged_artifacts
    swapped = False

    def swap_dependency_then_promote(
        promotion: list[StagedArtifact],
        *,
        precommit_validator: Callable[[], None],
    ) -> None:
        nonlocal swapped
        swapped = True
        dependency.rename(displaced)
        dependency.symlink_to(identical)
        real_promote_staged_artifacts(
            promotion,
            precommit_validator=precommit_validator,
        )

    monkeypatch.setattr(
        facade,
        "promote_staged_artifacts",
        swap_dependency_then_promote,
    )

    with pytest.raises(JointRiggerArtifactError, match="symlinks|dependency"):
        author_joint_rig(
            request,
            _ProjectedNoSidecarBackend("sibling", dependency),
            targets,
        )

    assert swapped
    assert dependency.is_symlink()
    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.sealed-*"))


def test_request_input_mutation_at_promoter_entry_rolls_back_before_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    result = _result(request)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    real_promote_staged_artifacts = facade.promote_staged_artifacts
    mutated = False

    def mutate_request_then_promote(
        promotion: list[StagedArtifact],
        *,
        precommit_validator: Callable[[], None],
    ) -> None:
        nonlocal mutated
        mutated = True
        source.write_text(_DEPENDENCY_USDA, encoding="utf-8")
        real_promote_staged_artifacts(
            promotion,
            precommit_validator=precommit_validator,
        )

    monkeypatch.setattr(
        facade,
        "promote_staged_artifacts",
        mutate_request_then_promote,
    )

    with pytest.raises(JointRiggerArtifactError, match="root_sha256"):
        author_joint_rig(request, _WritingBackend(result), targets)

    assert mutated
    assert source.read_text(encoding="utf-8") == _DEPENDENCY_USDA
    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.sealed-*"))


@pytest.mark.parametrize("symlink_kind", ["leaf", "ancestor"])
@pytest.mark.parametrize("dependency_kind", ["layer", "asset"])
def test_no_sidecar_symlink_dependencies_are_rejected_before_projection(
    tmp_path: Path,
    symlink_kind: Literal["leaf", "ancestor"],
    dependency_kind: Literal["layer", "asset"],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    real_directory = tmp_path / "real-dependencies"
    real_directory.mkdir()
    suffix = ".usda" if dependency_kind == "layer" else ".txt"
    real_dependency = real_directory / f"dependency{suffix}"
    real_dependency.write_text(
        _DEPENDENCY_USDA if dependency_kind == "layer" else "dependency bytes",
        encoding="utf-8",
    )
    if symlink_kind == "leaf":
        dependency_path = tmp_path / f"dependency-alias{suffix}"
        dependency_path.symlink_to(real_dependency)
        authored_reference = dependency_path.name
    else:
        alias_directory = tmp_path / "dependency-alias"
        alias_directory.symlink_to(real_directory, target_is_directory=True)
        dependency_path = alias_directory / real_dependency.name
        authored_reference = f"{alias_directory.name}/{real_dependency.name}"

    request = _request(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    backend = _ProjectedNoSidecarBackend(
        "sibling",
        dependency_path,
        authored_reference,
        dependency_kind,
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="dependency path must not contain symlinks",
    ) as raised:
        author_joint_rig(request, backend, targets)

    _assert_complete_bundle(targets)
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()
    assert not any(tmp_path.glob(".*.validate-*"))


def test_no_sidecar_nested_layer_symlink_dependency_is_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    real_dependency = tmp_path / "real-dependency.usda"
    real_dependency.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    alias_dependency = tmp_path / "dependency-alias.usda"
    alias_dependency.symlink_to(real_dependency)
    nested_layer = tmp_path / "nested.usda"
    nested_layer.write_text(
        _root_with_reference(alias_dependency.name),
        encoding="utf-8",
    )
    request = _request(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    with pytest.raises(
        JointRiggerArtifactError,
        match="dependency path must not contain symlinks",
    ) as raised:
        author_joint_rig(
            request,
            _ProjectedNoSidecarBackend("sibling", nested_layer),
            targets,
        )

    _assert_complete_bundle(targets)
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()
    assert not any(tmp_path.glob(".*.validate-*"))


def test_no_sidecar_publication_projection_rejects_changed_dependency_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_output = tmp_path / ".rigged.stage-test.usda"
    _write_empty_usda(staged_output)
    staged_dependency = tmp_path / "staged-dependency.usda"
    projected_dependency = tmp_path / "projected-dependency.usda"
    _write_empty_usda(staged_dependency)
    _write_empty_usda(projected_dependency)
    staged_targets = JointRiggerArtifactTargets(
        output_path=staged_output,
        diagnostics_path=tmp_path / ".diagnostics.stage-test.json",
        result_path=tmp_path / ".result.stage-test.json",
        publication_output_path=tmp_path / "rigged.usda",
    )
    final_targets = _targets(tmp_path)
    staged_artifacts = StagedJointRiggerArtifacts(
        final_targets=final_targets,
        staged_targets=staged_targets,
    )
    inventories = iter(((staged_dependency,), (projected_dependency,)))
    monkeypatch.setattr(
        facade,
        "_local_usd_dependency_paths",
        lambda path, *, label: next(inventories),
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="dependency closure changes.*missing=.*added=",
    ):
        facade._project_no_sidecar_output_identity(
            staged_artifacts,
            uri="memory://generated.usda",
        )

    assert not any(tmp_path.glob(".*.validate-*"))


def test_no_sidecar_succeeded_result_requires_dependency_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path)

    with pytest.raises(JointRiggerArtifactError, match="must claim"):
        author_joint_rig(
            request,
            _WritingBackend(_result(request, dependency_bundle_sha256=None)),
            targets,
        )

    assert not targets.output_path.exists()


def test_no_sidecar_dependency_identity_must_match_staged_usd(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    output_text = '#usda 1.0\n\ndef Xform "Rigged" {}\n'
    request = _request(source)
    result = _result(
        request,
        output_text=output_text,
        dependency_bundle_sha256="0" * 64,
    )
    targets = _targets(tmp_path)

    with pytest.raises(
        JointRiggerArtifactError,
        match="projected publication USD dependency closure",
    ):
        author_joint_rig(
            request,
            _WritingBackend(result, output_text=output_text),
            targets,
        )

    assert not targets.output_path.exists()


def test_no_sidecar_dependency_identity_wraps_usd_contract_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    request = _request(source)
    invalid_text = "not a USD layer"
    result = _result(
        request,
        output_text=invalid_text,
        dependency_bundle_sha256="0" * 64,
    )
    targets = _targets(tmp_path)

    with pytest.raises(JointRiggerArtifactError, match="Could not inspect") as caught:
        author_joint_rig(
            request,
            _WritingBackend(result, output_text=invalid_text),
            targets,
        )

    assert isinstance(caught.value.__cause__, JointRiggerContractError)
    assert not targets.output_path.exists()


def test_no_sidecar_dependency_verifier_unavailable_preserves_final_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    request = _request(source)
    result = _result(request, dependency_bundle_sha256="0" * 64)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    def fail_identity(*args: object, **kwargs: object) -> ArtifactIdentityV1:
        raise ImportError("pxr unavailable")

    monkeypatch.setattr(
        "world_understanding.functions.physics.joint_rigger.reference."
        "identify_usd_artifact",
        fail_identity,
    )

    with pytest.raises(JointRiggerBackendUnavailableError, match="unavailable"):
        author_joint_rig(request, _WritingBackend(result), targets)

    _assert_complete_bundle(targets)


def test_facade_requires_exact_diagnostics_for_every_planned_fact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _planned_request_with_every_fact(source)
    diagnostics = _diagnostics_for_every_planned_fact(request)

    result = author_joint_rig(
        request,
        _WritingBackend(_result(request, diagnostics=diagnostics)),
        _targets(tmp_path),
    )

    assert result.diagnostics == diagnostics


@pytest.mark.parametrize(
    ("joint_id", "field"),
    [
        (None, "articulation_root"),
        (None, "rigid_bodies[/World/Base].rigid_body"),
        (None, "rigid_bodies[/World/Base].mass.mass_kg"),
        (
            None,
            "rigid_bodies[/World/Base].colliders[/World/Base/Collision].collision",
        ),
        (
            None,
            "rigid_bodies[/World/Base].colliders[/World/Base/Collision].mesh_collision_api",
        ),
        (
            None,
            "rigid_bodies[/World/Base].colliders[/World/Base/Collision].mesh_approximation",
        ),
        ("hinge", "topology.body0"),
        ("hinge", "limit.lower"),
        ("hinge", "anchor.position_stage"),
        ("hinge", "drive.stiffness"),
        ("hinge", "state.velocity"),
        ("follower", "mimic.gearing"),
    ],
)
def test_facade_rejects_missing_planned_field_decisions(
    tmp_path: Path,
    joint_id: str | None,
    field: str,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _planned_request_with_every_fact(source)
    diagnostics = _diagnostics_for_every_planned_fact(request)
    if joint_id is None:
        diagnostics = diagnostics.model_copy(
            update={
                "field_decisions": tuple(
                    decision
                    for decision in diagnostics.field_decisions
                    if decision.field != field
                )
            }
        )
    else:
        diagnostics = diagnostics.model_copy(
            update={
                "joint_diagnostics": tuple(
                    item.model_copy(
                        update={
                            "field_decisions": tuple(
                                decision
                                for decision in item.field_decisions
                                if decision.field != field
                            )
                        }
                    )
                    if item.joint_id == joint_id
                    else item
                    for item in diagnostics.joint_diagnostics
                )
            }
        )
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    with pytest.raises(JointRiggerArtifactError, match="missing planned field"):
        author_joint_rig(
            request,
            _WritingBackend(_result(request, diagnostics=diagnostics)),
            targets,
        )

    _assert_complete_bundle(targets)


@pytest.mark.parametrize(
    ("field", "wrong_decision", "error_match"),
    [
        (
            "anchor.position_stage",
            FieldDecisionV1(
                field="anchor.position_stage",
                disposition="defaulted",
                reason_code="backend_default",
            ),
            "must be accepted",
        ),
        (
            "topology.body0",
            FieldDecisionV1(
                field="topology.body0",
                disposition="accepted",
                provenance=FieldProvenanceV1(
                    source="owner_approved_plan",
                    evidence="Different evidence must not substitute for the plan.",
                ),
            ),
            "provenance does not match",
        ),
    ],
)
def test_facade_rejects_wrong_disposition_or_provenance_for_planned_fact(
    tmp_path: Path,
    field: str,
    wrong_decision: FieldDecisionV1,
    error_match: str,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _planned_request_with_every_fact(source)
    diagnostics = _diagnostics_for_every_planned_fact(request)
    diagnostics = diagnostics.model_copy(
        update={
            "joint_diagnostics": tuple(
                item.model_copy(
                    update={
                        "field_decisions": tuple(
                            wrong_decision if decision.field == field else decision
                            for decision in item.field_decisions
                        )
                    }
                )
                if item.joint_id == "hinge"
                else item
                for item in diagnostics.joint_diagnostics
            )
        }
    )

    with pytest.raises(JointRiggerArtifactError, match=error_match):
        author_joint_rig(
            request,
            _WritingBackend(_result(request, diagnostics=diagnostics)),
            _targets(tmp_path),
        )


@pytest.mark.parametrize("scope", ["top-level", "joint", "joint-id"])
def test_facade_rejects_unexpected_or_duplicate_diagnostic_identity(
    tmp_path: Path,
    scope: Literal["top-level", "joint", "joint-id"],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _planned_request_with_every_fact(source)
    diagnostics = _diagnostics_for_every_planned_fact(request)
    if scope == "top-level":
        diagnostics = diagnostics.model_copy(
            update={
                "field_decisions": diagnostics.field_decisions
                + (
                    FieldDecisionV1(
                        field="unmodeled",
                        disposition="ignored",
                        reason_code="not_planned",
                    ),
                )
            }
        )
        error_match = "unexpected field"
    elif scope == "joint":
        first = diagnostics.joint_diagnostics[0]
        duplicate = first.field_decisions[0]
        diagnostics = diagnostics.model_copy(
            update={
                "joint_diagnostics": (
                    first.model_copy(
                        update={"field_decisions": first.field_decisions + (duplicate,)}
                    ),
                    *diagnostics.joint_diagnostics[1:],
                )
            }
        )
        error_match = "exactly one decision"
    else:
        duplicate = diagnostics.joint_diagnostics[0]
        diagnostics = diagnostics.model_copy(
            update={"joint_diagnostics": diagnostics.joint_diagnostics + (duplicate,)}
        )
        error_match = "joint identifiers must be unique"

    with pytest.raises(JointRiggerArtifactError, match=error_match):
        author_joint_rig(
            request,
            _WritingBackend(_result(request, diagnostics=diagnostics)),
            _targets(tmp_path),
        )


def test_facade_allows_diagnosed_absent_fields_and_backend_defaults(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    reference = tmp_path / "reference.usda"
    _write_empty_usda(source)
    _write_empty_usda(reference)
    request = _request_with_reference(source, reference)
    joint = request.plan.joints[0]
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="defaulting-fake",
        field_decisions=(
            FieldDecisionV1(
                field="legacy_component_names",
                disposition="ignored",
                reason_code="legacy_component_name_compatibility_not_requested",
            ),
            FieldDecisionV1(
                field="articulation_root",
                disposition="ignored",
                reason_code="not_provided",
            ),
            FieldDecisionV1(
                field="rigid_bodies",
                disposition="ignored",
                reason_code="not_provided",
            ),
        ),
        joint_diagnostics=(
            JointDiagnosticV1(
                joint_id=joint.topology.joint_id,
                field_decisions=(
                    *(
                        FieldDecisionV1(
                            field=f"topology.{field}",
                            disposition="accepted",
                            provenance=provenance,
                        )
                        for field, provenance in joint.topology.field_provenance.items()
                    ),
                    FieldDecisionV1(
                        field="limit",
                        disposition="ignored",
                        reason_code="not_provided",
                    ),
                    FieldDecisionV1(
                        field="anchor",
                        disposition="defaulted",
                        reason_code="inferred_body1_world_origin",
                    ),
                    FieldDecisionV1(
                        field="drive",
                        disposition="ignored",
                        reason_code="not_provided",
                    ),
                    FieldDecisionV1(
                        field="state",
                        disposition="ignored",
                        reason_code="not_provided",
                    ),
                    FieldDecisionV1(
                        field="mimic",
                        disposition="ignored",
                        reason_code="not_provided",
                    ),
                    FieldDecisionV1(
                        field="usd.joint_prim_path",
                        disposition="defaulted",
                        reason_code="deterministic_joint_path",
                    ),
                    FieldDecisionV1(
                        field="usd.local_frames",
                        disposition="defaulted",
                        reason_code="derived_from_stage_axis_and_anchor",
                    ),
                ),
            ),
        ),
    )

    result = author_joint_rig(
        request,
        _WritingBackend(_result(request, diagnostics=diagnostics)),
        _targets(tmp_path),
    )

    assert result.status == "succeeded"


@pytest.mark.parametrize(
    ("field", "expected_reason"),
    [
        ("anchor", "inferred_body1_world_origin"),
        ("anchor.position_stage", "inferred_body1_world_origin"),
        ("usd.joint_prim_path", "deterministic_joint_path"),
        ("usd.local_frames", "derived_from_stage_axis_and_anchor"),
    ],
)
def test_facade_rejects_undocumented_backend_default_reason(
    tmp_path: Path,
    field: str,
    expected_reason: str,
) -> None:
    source = tmp_path / "source.usda"
    reference = tmp_path / "reference.usda"
    _write_empty_usda(source)
    _write_empty_usda(reference)
    request = _request_with_reference(source, reference)
    joint = request.plan.joints[0]
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="arbitrary-default-fake",
        field_decisions=(
            FieldDecisionV1(
                field="legacy_component_names",
                disposition="ignored",
                reason_code="legacy_component_name_compatibility_not_requested",
            ),
        ),
        joint_diagnostics=(
            JointDiagnosticV1(
                joint_id=joint.topology.joint_id,
                field_decisions=(
                    *(
                        FieldDecisionV1(
                            field=f"topology.{topology_field}",
                            disposition="accepted",
                            provenance=provenance,
                        )
                        for topology_field, provenance in (
                            joint.topology.field_provenance.items()
                        )
                    ),
                    FieldDecisionV1(
                        field=field,
                        disposition="defaulted",
                        reason_code="arbitrary_backend_default",
                    ),
                ),
            ),
        ),
    )
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    with pytest.raises(
        JointRiggerArtifactError,
        match=f"reason_code={expected_reason}",
    ):
        author_joint_rig(
            request,
            _WritingBackend(_result(request, diagnostics=diagnostics)),
            targets,
        )

    _assert_complete_bundle(targets)


def test_legacy_component_assignments_require_stable_diagnostics_decisions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    compatibility = LegacyComponentNameCompatibilityV1(
        assignments=(
            LegacyComponentAssignmentV1(
                prim_path="/World/Door",
                component_name="door",
                source_field="role",
            ),
            LegacyComponentAssignmentV1(
                prim_path="/World/Handle",
                component_name="handle",
                source_field="component_name",
            ),
        )
    )
    request = _request(source, legacy_component_names=compatibility)
    provenance = FieldProvenanceV1(
        source="owner_approved_plan",
        evidence="Explicit compatibility component_name assignment.",
    )
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="fake",
        field_decisions=(
            FieldDecisionV1(
                field="legacy_component_names[/World/Door]",
                disposition="defaulted",
                reason_code="legacy_component_name_compatibility",
            ),
            FieldDecisionV1(
                field="legacy_component_names[/World/Handle]",
                disposition="accepted",
                provenance=provenance,
            ),
        ),
    )

    result = author_joint_rig(
        request,
        _WritingBackend(_result(request, diagnostics=diagnostics)),
        _targets(tmp_path),
    )

    assert result.diagnostics == diagnostics


@pytest.mark.parametrize("disposition", ["ignored", "rejected"])
def test_legacy_aggregate_no_fallback_decision_is_required(
    tmp_path: Path,
    disposition: Literal["ignored", "rejected"],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="fake",
        field_decisions=(
            FieldDecisionV1(
                field="legacy_component_names",
                disposition=disposition,
                reason_code="legacy_component_name_compatibility_not_requested",
            ),
        ),
    )

    result = author_joint_rig(
        request,
        _WritingBackend(_result(request, diagnostics=diagnostics)),
        _targets(tmp_path),
    )

    assert result.diagnostics == diagnostics


@pytest.mark.parametrize("decision_kind", ["missing", "accepted", "defaulted"])
def test_structured_request_rejects_missing_or_wrong_no_fallback_decision(
    tmp_path: Path,
    decision_kind: Literal["missing", "accepted", "defaulted"],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    if decision_kind == "missing":
        decisions: tuple[FieldDecisionV1, ...] = ()
    elif decision_kind == "accepted":
        decisions = (
            FieldDecisionV1(
                field="legacy_component_names",
                disposition="accepted",
                provenance=FieldProvenanceV1(
                    source="owner_approved_plan",
                    evidence="Incorrectly claims a consumed compatibility input.",
                ),
            ),
        )
    else:
        decisions = (
            FieldDecisionV1(
                field="legacy_component_names",
                disposition="defaulted",
                reason_code="legacy_component_name_compatibility",
            ),
        )
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="fake",
        field_decisions=decisions,
    )
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    with pytest.raises(
        JointRiggerArtifactError,
        match="exactly one|must be ignored or rejected",
    ) as raised:
        author_joint_rig(
            request,
            _WritingBackend(_result(request, diagnostics=diagnostics)),
            targets,
        )

    _assert_complete_bundle(targets)
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()


def test_structured_request_rejects_duplicate_no_fallback_decisions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    decision = FieldDecisionV1(
        field="legacy_component_names",
        disposition="ignored",
        reason_code="legacy_component_name_compatibility_not_requested",
    )
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="fake",
        field_decisions=(decision,),
    ).model_copy(update={"field_decisions": (decision, decision)})
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    with pytest.raises(JointRiggerArtifactError, match="exactly one") as raised:
        author_joint_rig(
            request,
            _WritingBackend(_result(request, diagnostics=diagnostics)),
            targets,
        )

    _assert_complete_bundle(targets)
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()


@pytest.mark.parametrize("with_expected_assignment", [False, True])
def test_legacy_component_assignments_reject_unrequested_diagnostics(
    tmp_path: Path,
    with_expected_assignment: bool,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    compatibility = (
        LegacyComponentNameCompatibilityV1(
            assignments=(
                LegacyComponentAssignmentV1(
                    prim_path="/World/Expected",
                    component_name="expected",
                    source_field="role",
                ),
            )
        )
        if with_expected_assignment
        else None
    )
    request = _request(source, legacy_component_names=compatibility)
    expected_decisions = (
        (
            FieldDecisionV1(
                field="legacy_component_names[/World/Expected]",
                disposition="defaulted",
                reason_code="legacy_component_name_compatibility",
            ),
        )
        if with_expected_assignment
        else (
            FieldDecisionV1(
                field="legacy_component_names",
                disposition="ignored",
                reason_code="legacy_component_name_compatibility_not_requested",
            ),
        )
    )
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="fake",
        field_decisions=expected_decisions
        + (
            FieldDecisionV1(
                field="legacy_component_names[/World/Unexpected]",
                disposition="defaulted",
                reason_code="legacy_component_name_compatibility",
            ),
        ),
    )
    targets = _targets(tmp_path)

    with pytest.raises(JointRiggerArtifactError, match="unexpected legacy component"):
        author_joint_rig(
            request,
            _WritingBackend(_result(request, diagnostics=diagnostics)),
            targets,
        )

    assert not targets.output_path.exists()


@pytest.mark.parametrize(
    ("source_field", "decision_kind", "error_match"),
    [
        ("role", "missing", "missing field decision"),
        ("role", "wrong_role", "must be defaulted"),
        ("component_name", "wrong_component", "must be accepted"),
    ],
)
def test_legacy_component_assignments_reject_missing_or_wrong_diagnostics(
    tmp_path: Path,
    source_field: Literal["role", "component_name"],
    decision_kind: str,
    error_match: str,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    prim_path = "/World/Part"
    compatibility = LegacyComponentNameCompatibilityV1(
        assignments=(
            LegacyComponentAssignmentV1(
                prim_path=prim_path,
                component_name="part",
                source_field=source_field,
            ),
        )
    )
    request = _request(source, legacy_component_names=compatibility)
    field = f"legacy_component_names[{prim_path}]"
    if decision_kind == "missing":
        decisions: tuple[FieldDecisionV1, ...] = ()
    elif decision_kind == "wrong_role":
        decisions = (
            FieldDecisionV1(
                field=field,
                disposition="accepted",
                provenance=FieldProvenanceV1(
                    source="owner_approved_plan",
                    evidence="Wrong disposition for a role-derived assignment.",
                ),
            ),
        )
    else:
        decisions = (
            FieldDecisionV1(
                field=field,
                disposition="defaulted",
                reason_code="legacy_component_name_compatibility",
            ),
        )
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="fake",
        field_decisions=decisions,
    )

    with pytest.raises(JointRiggerArtifactError, match=error_match):
        author_joint_rig(
            request,
            _WritingBackend(_result(request, diagnostics=diagnostics)),
            _targets(tmp_path),
        )


def test_persisted_result_must_match_returned_result(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    targets = _targets(tmp_path)
    request = _request(source)
    result = _result(request, status="succeeded")
    different = _result(request, status="failed")
    backend = _WritingBackend(result, persisted_result=different)

    with pytest.raises(JointRiggerArtifactError, match="does not match"):
        author_joint_rig(request, backend, targets)

    assert not targets.output_path.exists()


def test_non_success_result_cannot_publish_generated_root(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path)

    with pytest.raises(JointRiggerArtifactError, match="status=succeeded"):
        author_joint_rig(
            request,
            _WritingBackend(_result(request, status="failed")),
            targets,
        )

    assert not targets.output_path.exists()


def test_result_plan_identity_must_match_request(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path)
    wrong_plan = _result(request).model_copy(update={"plan_sha256": "3" * 64})

    with pytest.raises(JointRiggerArtifactError, match="plan_sha256"):
        author_joint_rig(request, _WritingBackend(wrong_plan), targets)

    assert not targets.output_path.exists()


def test_result_input_identity_must_match_canonical_request(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path)
    wrong_input = _result(request).model_copy(
        update={"input_sha256": request.source_asset.root_sha256}
    )
    assert wrong_input.input_sha256 != canonical_sha256(request)

    with pytest.raises(JointRiggerArtifactError, match="canonical request"):
        author_joint_rig(request, _WritingBackend(wrong_input), targets)

    assert not targets.output_path.exists()


def test_result_output_identity_must_match_generated_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path)
    wrong_output = _result(
        request,
        output_text='#usda 1.0\n\ndef Xform "Different" {}\n',
    )

    with pytest.raises(JointRiggerArtifactError, match="root_sha256"):
        author_joint_rig(request, _WritingBackend(wrong_output), targets)

    assert not targets.output_path.exists()


@pytest.mark.parametrize(
    "output_uri_mode",
    ["publication_path", "publication_file_uri", "logical_uri"],
)
def test_succeeded_output_uri_accepts_publication_or_logical_location(
    tmp_path: Path,
    output_uri_mode: Literal[
        "publication_path",
        "publication_file_uri",
        "logical_uri",
    ],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path)

    result = author_joint_rig(
        request,
        _WritingBackend(_result(request), output_uri_mode=output_uri_mode),
        targets,
    )

    assert result.output_artifact is not None
    expected_uri = {
        "publication_path": str(targets.output_path),
        "publication_file_uri": targets.output_path.as_uri(),
        "logical_uri": "s3://bucket/generated.usda",
    }[output_uri_mode]
    assert result.output_artifact.uri == expected_uri
    assert targets.output_path.is_file()


def test_succeeded_output_uri_cannot_bind_to_replaced_symlink_referent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path)
    old_referent = tmp_path / "old-output.usda"
    old_referent.write_text('#usda 1.0\n\ndef Scope "Old" {}\n', encoding="utf-8")
    targets.output_path.symlink_to(old_referent)
    result = _result(request)
    assert result.output_artifact is not None
    result = result.model_copy(
        update={
            "output_artifact": result.output_artifact.model_copy(
                update={"uri": str(old_referent)}
            )
        }
    )

    with pytest.raises(
        JointRiggerArtifactError, match="publication_output_path"
    ) as raised:
        author_joint_rig(request, _WritingBackend(result), targets)

    assert targets.output_path.is_symlink()
    assert targets.output_path.resolve() == old_referent
    assert '"Old"' in old_referent.read_text(encoding="utf-8")
    assert not targets.diagnostics_path.exists()
    assert not targets.result_path.exists()
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()


@pytest.mark.parametrize(
    ("output_uri_mode", "preserve_existing"),
    [
        ("physical_path", True),
        ("physical_file_uri", True),
        ("unrelated_path", False),
        ("unrelated_file_uri", False),
    ],
)
def test_succeeded_local_output_uri_must_match_publication_location(
    tmp_path: Path,
    output_uri_mode: Literal[
        "physical_path",
        "physical_file_uri",
        "unrelated_path",
        "unrelated_file_uri",
    ],
    preserve_existing: bool,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path)
    if preserve_existing:
        _write_complete_bundle(targets)

    with pytest.raises(
        JointRiggerArtifactError, match="publication_output_path"
    ) as raised:
        author_joint_rig(
            request,
            _WritingBackend(_result(request), output_uri_mode=output_uri_mode),
            targets,
        )

    if preserve_existing:
        _assert_complete_bundle(targets)
    else:
        assert not targets.output_path.exists()
        assert not targets.diagnostics_path.exists()
        assert not targets.result_path.exists()
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()


@pytest.mark.parametrize(
    "uri_case",
    [
        "remote_host",
        "query",
        "fragment",
        "params",
        "relative",
        "dot_segment",
        "non_round_trip",
    ],
)
def test_succeeded_file_output_uri_must_be_canonical_and_usable(
    tmp_path: Path,
    uri_case: Literal[
        "remote_host",
        "query",
        "fragment",
        "params",
        "relative",
        "dot_segment",
        "non_round_trip",
    ],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    canonical_uri = targets.output_path.as_uri()
    invalid_uris = {
        "remote_host": canonical_uri.replace("file:///", "file://remote-host/", 1),
        "query": f"{canonical_uri}?download=1",
        "fragment": f"{canonical_uri}#generated-root",
        "params": f"{canonical_uri};version=1",
        "relative": f"file:{targets.output_path.name}",
        "dot_segment": (
            f"{targets.output_path.parent.as_uri()}/../"
            f"{targets.output_path.parent.name}/{targets.output_path.name}"
        ),
        "non_round_trip": canonical_uri.replace("file:///", "file:/", 1),
    }
    result = _result(request)
    assert result.output_artifact is not None
    result = result.model_copy(
        update={
            "output_artifact": result.output_artifact.model_copy(
                update={"uri": invalid_uris[uri_case]}
            )
        }
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="exact canonical absolute file URI",
    ) as raised:
        author_joint_rig(request, _WritingBackend(result), targets)

    _assert_complete_bundle(targets)
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()


@pytest.mark.parametrize(
    "uri_case",
    [
        "remote_host",
        "query",
        "fragment",
        "params",
        "relative",
        "dot_segment",
        "non_round_trip",
    ],
)
def test_input_file_uri_must_be_canonical_and_local_before_probe(
    tmp_path: Path,
    uri_case: Literal[
        "remote_host",
        "query",
        "fragment",
        "params",
        "relative",
        "dot_segment",
        "non_round_trip",
    ],
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    canonical_uri = source.as_uri()
    invalid_uris = {
        "remote_host": canonical_uri.replace("file:///", "file://remote-host/", 1),
        "query": f"{canonical_uri}?download=1",
        "fragment": f"{canonical_uri}#source",
        "params": f"{canonical_uri};version=1",
        "relative": f"file:{source.name}",
        "dot_segment": (
            f"{source.parent.as_uri()}/../{source.parent.name}/{source.name}"
        ),
        "non_round_trip": canonical_uri.replace("file:///", "file:/", 1),
    }
    source_identity = identify_usd_artifact(source, uri=canonical_uri).model_copy(
        update={"uri": invalid_uris[uri_case]}
    )
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=source_identity,
        plan=JointRiggerPlanV1(schema_version=PLAN_SCHEMA_VERSION, joints=()),
    )
    backend = _WritingBackend(_result(request))

    with pytest.raises(
        JointRiggerArtifactError,
        match="exact canonical absolute file URI",
    ):
        author_joint_rig(request, backend, _targets(tmp_path))

    assert backend.probed is False


def test_canonical_file_input_uri_is_bound_before_authoring(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=identify_usd_artifact(source, uri=source.as_uri()),
        plan=JointRiggerPlanV1(schema_version=PLAN_SCHEMA_VERSION, joints=()),
    )
    backend = _WritingBackend(_result(request))

    observed = author_joint_rig(request, backend, _targets(tmp_path))

    assert observed.status == "succeeded"
    assert backend.probed is True


def test_source_artifact_must_not_alias_output(tmp_path: Path) -> None:
    output = tmp_path / "same.usda"
    _write_empty_usda(output)
    targets = JointRiggerArtifactTargets(
        output_path=output,
        diagnostics_path=tmp_path / "diagnostics.json",
        result_path=tmp_path / "result.json",
    )

    with pytest.raises(ValueError, match="must not alias source_asset"):
        request = _request(output)
        author_joint_rig(request, _WritingBackend(_result(request)), targets)

    assert output.read_text(encoding="utf-8") == "#usda 1.0\n"


def test_local_read_paths_include_and_dedupe_nested_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    reference = tmp_path / "reference asset.usda"
    _write_empty_usda(reference)
    request = _request_with_reference(source, reference)

    assert facade._request_local_read_paths(request) == [
        ("source_asset", source),
        ("plan provenance artifact", reference),
    ]


def test_local_read_paths_dedupe_dependency_reused_as_provenance(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.usda"
    _write_empty_usda(reference)
    source = tmp_path / "source.usda"
    source.write_text(_root_with_reference(reference.name), encoding="utf-8")
    request = _request_with_reference(source, reference)

    assert facade._request_local_read_paths(request) == [
        ("source_asset", source),
        ("source_asset dependency", reference),
    ]


@pytest.mark.parametrize("dependency_kind", ["layer", "asset"])
def test_source_dependency_symlink_inside_sidecar_fails_before_probe(
    tmp_path: Path,
    dependency_kind: Literal["layer", "asset"],
) -> None:
    suffix = ".usda" if dependency_kind == "layer" else ".bin"
    real_dependency = tmp_path / f"real-dependency{suffix}"
    if dependency_kind == "layer":
        _write_empty_usda(real_dependency)
    else:
        real_dependency.write_bytes(b"source dependency")
    sidecar = tmp_path / "rigged_assets"
    sidecar.mkdir()
    dependency_alias = sidecar / f"dependency{suffix}"
    dependency_alias.symlink_to(real_dependency)
    source = tmp_path / "source.usda"
    authored_reference = f"{sidecar.name}/{dependency_alias.name}"
    source.write_text(
        (
            _root_with_reference(authored_reference)
            if dependency_kind == "layer"
            else _root_with_asset_reference(authored_reference)
        ),
        encoding="utf-8",
    )
    request = _request(source)
    backend = _WritingBackend(_result(request))
    targets = _targets(tmp_path, sidecar=True)

    with pytest.raises(
        ValueError,
        match="source_asset dependency must not be inside sidecar_path",
    ):
        author_joint_rig(request, backend, targets)

    assert backend.probed is False
    assert dependency_alias.is_symlink()
    assert dependency_alias.resolve() == real_dependency
    assert real_dependency.is_file()
    assert facade._local_usd_dependency_paths(source, label="source_asset") == (
        real_dependency,
        dependency_alias,
    )


def test_source_dependency_chained_through_sidecar_fails_before_probe(
    tmp_path: Path,
) -> None:
    real_dependency = tmp_path / "real-dependency.usda"
    _write_empty_usda(real_dependency)
    sidecar = tmp_path / "rigged_assets"
    sidecar.mkdir()
    intermediate_alias = sidecar / "intermediate.usda"
    intermediate_alias.symlink_to(real_dependency)
    authored_alias = tmp_path / "authored-dependency.usda"
    authored_alias.symlink_to(intermediate_alias)
    source = tmp_path / "source.usda"
    source.write_text(
        _root_with_reference(authored_alias.name),
        encoding="utf-8",
    )
    request = _request(source)
    backend = _WritingBackend(_result(request))
    targets = _targets(tmp_path, sidecar=True)

    with pytest.raises(
        ValueError,
        match="source_asset dependency must not be inside sidecar_path",
    ):
        author_joint_rig(request, backend, targets)

    assert backend.probed is False
    assert authored_alias.is_symlink()
    assert intermediate_alias.is_symlink()
    assert real_dependency.is_file()
    assert set(facade._local_usd_dependency_paths(source, label="source_asset")) == {
        authored_alias,
        intermediate_alias,
        real_dependency,
    }


def test_normalize_local_input_path_resolves_lexical_parent_traversal(
    tmp_path: Path,
) -> None:
    lexical_path = tmp_path / "discarded" / ".." / "source.usda"

    normalized = facade._normalize_local_input_path(
        lexical_path,
        label="source_asset",
    )

    assert normalized == tmp_path / "source.usda"


@pytest.mark.parametrize("symlink_kind", ["leaf", "ancestor"])
def test_local_input_symlink_chain_fails_before_probe(
    tmp_path: Path,
    symlink_kind: Literal["leaf", "ancestor"],
) -> None:
    real_directory = tmp_path / "real-input"
    real_directory.mkdir()
    real_source = real_directory / "source.usda"
    _write_empty_usda(real_source)
    if symlink_kind == "leaf":
        request_path = tmp_path / "source.usda"
        request_path.symlink_to(real_source)
    else:
        alias_directory = tmp_path / "input-alias"
        alias_directory.symlink_to(real_directory, target_is_directory=True)
        request_path = alias_directory / real_source.name
    request = _request(real_source)
    request = request.model_copy(
        update={
            "source_asset": request.source_asset.model_copy(
                update={"uri": str(request_path)}
            )
        }
    )
    backend = _WritingBackend(_result(request))
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    with pytest.raises(
        JointRiggerArtifactError,
        match="source_asset local input path must not contain symlinks",
    ):
        author_joint_rig(request, backend, targets)

    assert backend.probed is False
    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.stage-*"))


@pytest.mark.parametrize("symlink_kind", ["leaf", "ancestor"])
def test_local_input_symlink_substitution_during_authoring_cannot_publish(
    tmp_path: Path,
    symlink_kind: Literal["leaf", "ancestor"],
) -> None:
    input_directory = tmp_path / "live-input"
    input_directory.mkdir()
    source = input_directory / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    class SubstitutingBackend(_WritingBackend):
        def author(
            self,
            request: JointRiggerInputV1,
            artifact_targets: JointRiggerArtifactTargets,
        ) -> JointRiggerResultV1:
            result = super().author(request, artifact_targets)
            if symlink_kind == "leaf":
                replacement = tmp_path / "replacement.usda"
                replacement.write_bytes(source.read_bytes())
                source.unlink()
                source.symlink_to(replacement)
            else:
                relocated = tmp_path / "relocated-input"
                input_directory.rename(relocated)
                input_directory.symlink_to(relocated, target_is_directory=True)
            return result

    backend = SubstitutingBackend(_result(request))
    with pytest.raises(
        JointRiggerArtifactError,
        match="source_asset local input path must not contain symlinks",
    ) as raised:
        author_joint_rig(request, backend, targets)

    assert backend.probed is True
    _assert_complete_bundle(targets)
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()


def test_local_non_usd_input_is_verified_as_root_only(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"opaque input")
    request = _request(source)

    result = author_joint_rig(
        request,
        _WritingBackend(_result(request)),
        _targets(tmp_path),
    )

    assert result.status == "succeeded"


@pytest.mark.parametrize("special_kind", ["fifo", "device"])
def test_local_special_file_input_is_rejected_before_probe(
    tmp_path: Path,
    special_kind: Literal["fifo", "device"],
) -> None:
    if special_kind == "fifo":
        source = tmp_path / "source.pipe"
        os.mkfifo(source)
    else:
        source = Path("/dev/zero")
        if not source.exists():  # pragma: no cover - Linux runtime invariant
            pytest.skip("/dev/zero is unavailable on this platform")
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=ArtifactIdentityV1(
            uri=str(source),
            root_sha256="0" * 64,
        ),
        plan=JointRiggerPlanV1(schema_version=PLAN_SCHEMA_VERSION, joints=()),
    )
    backend = _WritingBackend(_result(request))
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    with pytest.raises(JointRiggerArtifactError, match="regular file"):
        author_joint_rig(request, backend, targets)

    assert backend.probed is False
    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.stage-*"))


@pytest.mark.parametrize("artifact_kind", ["source", "provenance"])
def test_local_usd_input_requires_dependency_bundle_before_probe(
    tmp_path: Path,
    artifact_kind: Literal["source", "provenance"],
) -> None:
    source = tmp_path / "source.usda"
    reference = tmp_path / "reference.usda"
    _write_empty_usda(source)
    _write_empty_usda(reference)
    if artifact_kind == "source":
        request = _request(source, bind_usd_dependencies=False)
    else:
        request = _request_with_reference(
            source,
            reference,
            bind_reference_dependencies=False,
        )
    backend = _WritingBackend(_result(request))
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    with pytest.raises(
        JointRiggerArtifactError,
        match="local USD identity must provide dependency_bundle_sha256",
    ):
        author_joint_rig(request, backend, targets)

    assert backend.probed is False
    _assert_complete_bundle(targets)
    assert not any(tmp_path.glob(".*.stage-*"))


def test_remote_usd_identity_remains_logical_without_dependency_bundle(
    tmp_path: Path,
) -> None:
    request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=ArtifactIdentityV1(
            uri="s3://bucket/source.usda",
            root_sha256="0" * 64,
        ),
        plan=JointRiggerPlanV1(schema_version=PLAN_SCHEMA_VERSION, joints=()),
    )

    result = author_joint_rig(
        request,
        _WritingBackend(_result(request)),
        _targets(tmp_path),
    )

    assert result.status == "succeeded"


def test_remote_dependency_identifier_classification() -> None:
    assert facade._is_remote_dependency_identifier("https://example.test/a.usda")
    assert facade._is_remote_dependency_identifier("s://example.test/a.usda")
    assert facade._is_remote_dependency_identifier("s:opaque-resolver-asset")
    assert not facade._is_remote_dependency_identifier("file:///tmp/a.usda")
    assert not facade._is_remote_dependency_identifier("C:/assets/a.usda")
    assert not facade._is_remote_dependency_identifier(r"C:\assets\a.usda")


def test_remote_authored_layer_is_rejected_even_with_local_resolved_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import UsdUtils

    remote = "https://example.test/authored-layer.usda"
    local_cache = tmp_path / "cached-layer.usda"
    _write_empty_usda(local_cache)

    @dataclass
    class CachedLayer:
        identifier: str
        resolvedPath: str
        realPath: str

    monkeypatch.setattr(
        UsdUtils,
        "ComputeAllDependencies",
        lambda _path: (
            [CachedLayer(remote, str(local_cache), str(local_cache))],
            [],
            [],
        ),
    )

    with pytest.raises(JointRiggerArtifactError, match="external URIs.*https"):
        facade._reject_uri_usd_dependencies(tmp_path / "root.usda")


def test_remote_authored_asset_is_rejected_even_with_local_resolved_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import UsdUtils

    remote = "https://example.test/authored-texture.png"
    local_cache = tmp_path / "cached-texture.png"
    local_cache.write_bytes(b"cached texture")

    @dataclass
    class CachedAsset:
        path: str
        resolvedPath: str
        identifier: str

    monkeypatch.setattr(
        UsdUtils,
        "ComputeAllDependencies",
        lambda _path: (
            [],
            [CachedAsset(remote, str(local_cache), remote)],
            [],
        ),
    )

    with pytest.raises(JointRiggerArtifactError, match="external URIs.*https"):
        facade._reject_uri_usd_dependencies(tmp_path / "root.usda")


@pytest.mark.parametrize(
    ("dependency_kind", "symlink_field"),
    [
        ("layer", "resolvedPath"),
        ("layer", "realPath"),
        ("asset", "resolvedPath"),
        ("asset", "identifier"),
    ],
)
def test_symlink_guard_checks_every_usd_dependency_locator_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency_kind: Literal["layer", "asset"],
    symlink_field: str,
) -> None:
    from pxr import UsdUtils

    root = tmp_path / "root.usda"
    safe_locator = tmp_path / "authored.usda"
    real_dependency = tmp_path / "real-dependency.usda"
    symlink_locator = tmp_path / "resolved-alias.usda"
    _write_empty_usda(root)
    _write_empty_usda(safe_locator)
    _write_empty_usda(real_dependency)
    symlink_locator.symlink_to(real_dependency.name)

    @dataclass
    class LayerLocator:
        identifier: str
        resolvedPath: str
        realPath: str

    @dataclass
    class AssetLocator:
        path: str
        resolvedPath: str
        identifier: str

    layers: list[Any] = []
    assets: list[Any] = []
    if dependency_kind == "layer":
        dependency = LayerLocator(
            identifier=str(safe_locator),
            resolvedPath=str(real_dependency),
            realPath=str(real_dependency),
        )
        layers.append(dependency)
    else:
        dependency = AssetLocator(
            path=str(safe_locator),
            resolvedPath=str(real_dependency),
            identifier=str(safe_locator),
        )
        assets.append(dependency)
    setattr(dependency, symlink_field, str(symlink_locator))
    monkeypatch.setattr(
        UsdUtils,
        "ComputeAllDependencies",
        lambda _path: (layers, assets, []),
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="dependency path must not contain symlinks",
    ):
        facade._reject_symlink_usd_dependencies(root)


def test_symlink_guard_keeps_leading_tilde_locator_literal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import UsdUtils

    root = tmp_path / "root.usda"
    _write_empty_usda(root)
    literal_directory = tmp_path / "~"
    literal_directory.mkdir()
    literal_dependency = literal_directory / "dep.bin"
    literal_dependency.write_bytes(b"literal authored dependency")

    home = tmp_path / "home"
    home.mkdir()
    unrelated_dependency = tmp_path / "unrelated.bin"
    unrelated_dependency.write_bytes(b"unrelated home dependency")
    home_dependency = home / "dep.bin"
    home_dependency.symlink_to(unrelated_dependency)
    monkeypatch.setenv("HOME", str(home))

    @dataclass
    class AssetLocator:
        path: str
        resolvedPath: str
        identifier: str

    monkeypatch.setattr(
        UsdUtils,
        "ComputeAllDependencies",
        lambda _path: (
            [],
            [
                AssetLocator(
                    path="~/dep.bin",
                    resolvedPath=str(literal_dependency),
                    identifier=str(literal_dependency),
                )
            ],
            [],
        ),
    )

    facade._reject_symlink_usd_dependencies(root)

    assert home_dependency.is_symlink()


def test_provenance_artifact_must_not_alias_output_before_probe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    reference = tmp_path / "reference.usda"
    _write_empty_usda(reference)
    request = _request_with_reference(source, reference)
    backend = _WritingBackend(_result(request))
    targets = JointRiggerArtifactTargets(
        output_path=reference,
        diagnostics_path=tmp_path / "diagnostics.json",
        result_path=tmp_path / "result.json",
    )

    with pytest.raises(ValueError, match="must not alias plan provenance artifact"):
        author_joint_rig(request, backend, targets)

    assert backend.probed is False
    assert reference.read_text(encoding="utf-8") == "#usda 1.0\n"


def test_provenance_artifact_must_not_be_inside_sidecar_before_probe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    sidecar = tmp_path / "rigged_assets"
    sidecar.mkdir()
    reference = sidecar / "reference.usda"
    _write_empty_usda(reference)
    request = _request_with_reference(source, reference)
    backend = _WritingBackend(_result(request))
    targets = _targets(tmp_path, sidecar=True)

    with pytest.raises(
        ValueError,
        match="plan provenance artifact must not be inside sidecar_path",
    ):
        author_joint_rig(request, backend, targets)

    assert backend.probed is False
    assert reference.read_text(encoding="utf-8") == "#usda 1.0\n"


def test_source_dependency_must_not_alias_output_before_probe(tmp_path: Path) -> None:
    dependency = tmp_path / "dependency.usda"
    _write_empty_usda(dependency)
    source = tmp_path / "source.usda"
    source.write_text(_root_with_reference(dependency.name), encoding="utf-8")
    request = _request(source)
    backend = _WritingBackend(_result(request))
    targets = JointRiggerArtifactTargets(
        output_path=dependency,
        diagnostics_path=tmp_path / "diagnostics.json",
        result_path=tmp_path / "result.json",
    )

    with pytest.raises(ValueError, match="source_asset dependency"):
        author_joint_rig(request, backend, targets)

    assert backend.probed is False
    assert dependency.read_text(encoding="utf-8") == "#usda 1.0\n"


def test_source_dependency_must_not_be_inside_sidecar_before_probe(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "rigged_assets"
    sidecar.mkdir()
    dependency = sidecar / "dependency.usda"
    _write_empty_usda(dependency)
    source = tmp_path / "source.usda"
    source.write_text(
        _root_with_reference("rigged_assets/dependency.usda"),
        encoding="utf-8",
    )
    request = _request(source)
    backend = _WritingBackend(_result(request))
    targets = _targets(tmp_path, sidecar=True)

    with pytest.raises(ValueError, match="source_asset dependency"):
        author_joint_rig(request, backend, targets)

    assert backend.probed is False
    assert dependency.is_file()


def test_unresolved_source_dependency_fails_typed_before_probe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text(_root_with_reference("missing.usda"), encoding="utf-8")
    request = _request(source, bind_usd_dependencies=False)
    request = request.model_copy(
        update={
            "source_asset": request.source_asset.model_copy(
                update={"dependency_bundle_sha256": "0" * 64}
            )
        }
    )
    backend = _WritingBackend(_result(request))

    with pytest.raises(JointRiggerArtifactError, match="unresolved"):
        author_joint_rig(request, backend, _targets(tmp_path))

    assert backend.probed is False


def test_source_root_identity_mismatch_fails_typed_before_probe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source).model_copy(
        update={
            "source_asset": ArtifactIdentityV1(
                uri=str(source),
                root_sha256="0" * 64,
            )
        }
    )
    backend = _WritingBackend(_result(request))

    with pytest.raises(JointRiggerArtifactError, match="root_sha256"):
        author_joint_rig(request, backend, _targets(tmp_path))

    assert backend.probed is False


def test_source_dependency_identity_mismatch_fails_typed_before_probe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source).model_copy(
        update={
            "source_asset": ArtifactIdentityV1(
                uri=str(source),
                root_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                dependency_bundle_sha256="0" * 64,
            )
        }
    )
    backend = _WritingBackend(_result(request))

    with pytest.raises(JointRiggerArtifactError, match="dependency_bundle_sha256"):
        author_joint_rig(request, backend, _targets(tmp_path))

    assert backend.probed is False


def test_source_dependency_mutation_during_authoring_cannot_publish(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "dependency.usda"
    _write_empty_usda(dependency)
    source = tmp_path / "source.usda"
    source.write_text(_root_with_reference(dependency.name), encoding="utf-8")
    request = _request(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)

    class MutatingBackend(_WritingBackend):
        def author(
            self,
            request: JointRiggerInputV1,
            artifact_targets: JointRiggerArtifactTargets,
        ) -> JointRiggerResultV1:
            result = super().author(request, artifact_targets)
            dependency.write_text(
                '#usda 1.0\n\ndef Xform "Mutated" {}\n',
                encoding="utf-8",
            )
            return result

    with pytest.raises(
        JointRiggerArtifactError,
        match="dependency_bundle_sha256 does not match|closure changed",
    ):
        author_joint_rig(request, MutatingBackend(_result(request)), targets)

    _assert_complete_bundle(targets)


def test_source_dependency_runtime_unavailable_fails_before_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    backend = _WritingBackend(_result(request))

    def unavailable(
        path: Path,
        *,
        include_lexical_aliases: bool = False,
    ) -> tuple[Path, ...]:
        del include_lexical_aliases
        raise ImportError("pxr unavailable")

    monkeypatch.setattr(
        "world_understanding.functions.physics.joint_rigger.reference."
        "local_usd_dependency_paths",
        unavailable,
    )

    with pytest.raises(JointRiggerBackendUnavailableError, match="unavailable"):
        author_joint_rig(request, backend, _targets(tmp_path))

    assert backend.probed is False


def test_facade_preserves_typed_artifact_errors_during_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    request = _request(source)
    targets = _targets(tmp_path)
    _write_complete_bundle(targets)
    expected = JointRiggerArtifactError("typed publication failure")

    def fail_staged_validation(*args: object, **kwargs: object) -> list[object]:
        raise expected

    monkeypatch.setattr(
        facade,
        "staged_promotion_artifacts",
        fail_staged_validation,
    )

    with pytest.raises(JointRiggerArtifactError) as raised:
        author_joint_rig(request, _WritingBackend(_result(request)), targets)

    assert raised.value is expected
    _assert_complete_bundle(targets)
    _assert_unbound_staging_root_preserved(raised.value, tmp_path).unlink()


def test_local_read_path_discovery_ignores_missing_and_remote_sources() -> None:
    class MissingSource:
        pass

    remote_request = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=ArtifactIdentityV1(
            uri="s3://bucket/source.usda",
            root_sha256="0" * 64,
        ),
        plan=JointRiggerPlanV1(schema_version=PLAN_SCHEMA_VERSION, joints=()),
    )

    assert facade._request_local_read_paths(MissingSource()) == []  # type: ignore[arg-type]
    assert facade._request_local_read_paths(remote_request) == []


def test_sealed_dependency_snapshot_cleanup_is_idempotent(tmp_path: Path) -> None:
    dependency = tmp_path / "dependency.usda"
    dependency.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    snapshot = facade._open_sealed_dependency_snapshot(dependency)

    snapshot.cleanup()
    snapshot.cleanup()

    assert snapshot.source_descriptor == -1


def test_generated_root_seal_preserves_primary_error_when_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generated.usda"
    root.write_text(_EMPTY_USDA, encoding="utf-8")
    expected = RuntimeError("forced generated-root hash failure")
    real_close = facade.os.close
    failed_descriptor: int | None = None
    close_calls: list[int] = []

    def fail_hash(descriptor: int, *, label: str) -> str:
        nonlocal failed_descriptor
        del label
        failed_descriptor = descriptor
        raise expected

    def close_then_fail(descriptor: int) -> None:
        close_calls.append(descriptor)
        real_close(descriptor)
        if descriptor == failed_descriptor:
            raise OSError(errno.EIO, "forced generated-root seal close failure")

    monkeypatch.setattr(facade, "_stable_descriptor_sha256", fail_hash)
    monkeypatch.setattr(facade.os, "close", close_then_fail)

    with pytest.raises(
        JointRiggerArtifactError,
        match="forced generated-root hash failure",
    ) as raised:
        facade._seal_generated_root(root, expected_sha256="0" * 64)

    assert raised.value.__cause__ is expected
    assert failed_descriptor is not None
    assert close_calls.count(failed_descriptor) == 1
    with pytest.raises(OSError):
        os.fstat(failed_descriptor)
    assert "forced generated-root seal close failure" in "\n".join(
        raised.value.__notes__
    )


@pytest.mark.parametrize("persistent_rebind_failure", [False, True])
def test_private_report_identity_fatal_rebinds_or_preserves_with_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persistent_rebind_failure: bool,
) -> None:
    primary = KeyboardInterrupt("forced initial report identity fatal")
    cleanup_error = SystemExit("forced persistent report identity fatal")
    real_open = facade.os.open
    real_fstat = facade.os.fstat
    real_close = facade.os.close
    writer_descriptor: int | None = None
    parent_descriptor: int | None = None
    writer_fstat_calls = 0
    close_calls: list[int] = []

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal parent_descriptor, writer_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if flags & os.O_DIRECTORY and Path(path) == tmp_path:
            parent_descriptor = descriptor
        if (
            isinstance(path, str)
            and path.startswith(".result.json.sealed-")
            and flags & os.O_CREAT
        ):
            writer_descriptor = descriptor
        return descriptor

    def fail_identity(descriptor: int) -> os.stat_result:
        nonlocal writer_fstat_calls
        if descriptor == writer_descriptor:
            writer_fstat_calls += 1
            if writer_fstat_calls == 1:
                raise primary
            if persistent_rebind_failure and writer_fstat_calls == 2:
                raise cleanup_error
        return real_fstat(descriptor)

    def track_close(descriptor: int) -> None:
        close_calls.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(facade.os, "open", track_open)
    monkeypatch.setattr(facade.os, "fstat", fail_identity)
    monkeypatch.setattr(facade.os, "close", track_close)

    with pytest.raises(KeyboardInterrupt) as raised:
        facade._create_private_report_snapshot(
            tmp_path / "result.json",
            b"{}",
            label="result",
        )

    assert raised.value is primary
    assert writer_fstat_calls >= 2
    assert writer_descriptor is not None
    assert parent_descriptor is not None
    assert close_calls.count(writer_descriptor) == 1
    assert close_calls.count(parent_descriptor) == 1
    with pytest.raises(OSError):
        os.fstat(writer_descriptor)
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    residual = list(tmp_path.glob(".*.sealed-*"))
    if persistent_rebind_failure:
        assert len(residual) == 1
        assert "forced persistent report identity fatal" in "\n".join(primary.__notes__)
        residual[0].unlink()
    else:
        assert not residual


@pytest.mark.parametrize("persistent_rebind_failure", [False, True])
def test_private_sidecar_identity_fatal_rebinds_or_preserves_with_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persistent_rebind_failure: bool,
) -> None:
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    (sidecar / "asset.usda").write_text(_DEPENDENCY_USDA, encoding="utf-8")
    expected_sha256 = sidecar_dependency_bundle_sha256(sidecar)
    primary = KeyboardInterrupt("forced initial private sidecar identity fatal")
    cleanup_error = SystemExit("forced persistent private sidecar identity fatal")
    real_stat = facade.os.stat
    stat_calls = 0

    def fail_identity(
        path: str | os.PathLike[str],
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal stat_calls
        if os.fspath(path).startswith(".sidecar.sealed-"):
            stat_calls += 1
            if stat_calls == 1:
                raise primary
            if persistent_rebind_failure and stat_calls == 2:
                raise cleanup_error
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(facade.os, "stat", fail_identity)

    with pytest.raises(KeyboardInterrupt) as raised:
        facade._create_private_sidecar_snapshot(
            sidecar,
            private_parent=tmp_path,
            expected_sha256=expected_sha256,
        )

    assert raised.value is primary
    if persistent_rebind_failure:
        assert stat_calls == 2
    else:
        assert stat_calls >= 2
    residual = list(tmp_path.glob(".*.sealed-*"))
    if persistent_rebind_failure:
        assert len(residual) == 1
        assert "forced persistent private sidecar identity fatal" in "\n".join(
            primary.__notes__
        )
        residual[0].rmdir()
    else:
        assert not residual


def test_private_report_identity_failure_preserves_substituted_foreign_and_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = KeyboardInterrupt("forced report identity failure after substitution")
    real_open = facade.os.open
    real_fstat = facade.os.fstat
    real_close = facade.os.close
    writer_descriptor: int | None = None
    parent_descriptor: int | None = None
    private_path: Path | None = None
    displaced = tmp_path / "displaced-owned-report"
    close_calls: list[int] = []

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal parent_descriptor, private_path, writer_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == tmp_path and flags & os.O_DIRECTORY:
            parent_descriptor = descriptor
        if (
            isinstance(path, str)
            and path.startswith(".result.json.sealed-")
            and flags & os.O_CREAT
        ):
            writer_descriptor = descriptor
            private_path = tmp_path / path
        return descriptor

    def substitute_then_fail(descriptor: int) -> os.stat_result:
        if descriptor == writer_descriptor:
            assert private_path is not None
            private_path.rename(displaced)
            private_path.write_bytes(b"foreign replacement")
            monkeypatch.setattr(facade.os, "fstat", real_fstat)
            raise primary
        return real_fstat(descriptor)

    def track_close(descriptor: int) -> None:
        close_calls.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(facade.os, "open", track_open)
    monkeypatch.setattr(facade.os, "fstat", substitute_then_fail)
    monkeypatch.setattr(facade.os, "close", track_close)

    with pytest.raises(KeyboardInterrupt) as raised:
        facade._create_private_report_snapshot(
            tmp_path / "result.json",
            b"{}",
            label="result",
        )

    assert raised.value is primary
    assert private_path is not None
    assert private_path.read_bytes() == b"foreign replacement"
    assert displaced.read_bytes() == b""
    assert "replacement preserved" in "\n".join(raised.value.__notes__)
    assert writer_descriptor is not None
    assert parent_descriptor is not None
    assert close_calls.count(writer_descriptor) == 1
    assert close_calls.count(parent_descriptor) == 1
    with pytest.raises(OSError):
        os.fstat(writer_descriptor)
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)


def test_private_sidecar_identity_failure_preserves_substituted_foreign_and_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    expected_sha256 = sidecar_dependency_bundle_sha256(sidecar)
    primary = KeyboardInterrupt("forced sidecar identity failure after substitution")
    real_open = facade.os.open
    real_fstat = facade.os.fstat
    real_close = facade.os.close
    source_descriptor: int | None = None
    parent_descriptor: int | None = None
    private_path: Path | None = None
    displaced = tmp_path / "displaced-owned-sidecar"
    close_calls: list[int] = []

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal parent_descriptor, private_path, source_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == tmp_path and flags & os.O_DIRECTORY:
            parent_descriptor = descriptor
        if (
            isinstance(path, str)
            and path.startswith(".sidecar.sealed-")
            and flags & os.O_DIRECTORY
        ):
            source_descriptor = descriptor
            private_path = tmp_path / path
        return descriptor

    def substitute_then_fail(descriptor: int) -> os.stat_result:
        if descriptor == source_descriptor:
            assert private_path is not None
            private_path.rename(displaced)
            private_path.mkdir()
            (private_path / "foreign.txt").write_text("foreign", encoding="utf-8")
            monkeypatch.setattr(facade.os, "fstat", real_fstat)
            raise primary
        return real_fstat(descriptor)

    def track_close(descriptor: int) -> None:
        close_calls.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(facade.os, "open", track_open)
    monkeypatch.setattr(facade.os, "fstat", substitute_then_fail)
    monkeypatch.setattr(facade.os, "close", track_close)

    with pytest.raises(KeyboardInterrupt) as raised:
        facade._create_private_sidecar_snapshot(
            sidecar,
            private_parent=tmp_path,
            expected_sha256=expected_sha256,
        )

    assert raised.value is primary
    assert private_path is not None
    assert (private_path / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert list(displaced.iterdir()) == []
    assert "replacement preserved" in "\n".join(raised.value.__notes__)
    assert source_descriptor is not None
    assert parent_descriptor is not None
    assert close_calls.count(source_descriptor) >= 1
    assert close_calls.count(parent_descriptor) >= 1
    with pytest.raises(OSError):
        os.fstat(source_descriptor)
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)


def test_private_report_preserves_ambiguous_post_create_open_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = KeyboardInterrupt("forced report post-create open fatal")
    real_open = facade.os.open
    real_close = facade.os.close
    parent_descriptor: int | None = None
    created_descriptor: int | None = None
    created_path: Path | None = None
    close_calls: list[int] = []

    def create_close_then_interrupt(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal created_descriptor, created_path, parent_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == tmp_path and flags & os.O_DIRECTORY:
            parent_descriptor = descriptor
        if (
            isinstance(path, str)
            and path.startswith(".result.json.sealed-")
            and flags & os.O_CREAT
        ):
            created_descriptor = descriptor
            created_path = tmp_path / path
            real_close(descriptor)
            raise primary
        return descriptor

    def track_close(descriptor: int) -> None:
        close_calls.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(facade.os, "open", create_close_then_interrupt)
    monkeypatch.setattr(facade.os, "close", track_close)

    with pytest.raises(KeyboardInterrupt) as raised:
        facade._create_private_report_snapshot(
            tmp_path / "result.json",
            b"{}",
            label="result",
        )

    assert raised.value is primary
    assert "could not be bound through an owned descriptor" in "\n".join(
        raised.value.__notes__
    )
    assert created_path is not None
    assert created_path.read_bytes() == b""
    assert created_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(created_descriptor)
    assert parent_descriptor is not None
    assert close_calls.count(parent_descriptor) == 1
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    created_path.unlink()


def test_stable_copy_preserves_ambiguous_post_create_open_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    payload = b"payload"
    source.write_bytes(payload)
    source_descriptor = os.open(source, os.O_RDONLY)
    metadata = os.fstat(source_descriptor)
    primary = KeyboardInterrupt("forced stable-copy post-create open fatal")
    real_open = facade.os.open
    real_close = facade.os.close
    parent_descriptor: int | None = None
    created_descriptor: int | None = None
    close_calls: list[int] = []

    def create_close_then_interrupt(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal created_descriptor, parent_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == tmp_path and flags & os.O_DIRECTORY:
            parent_descriptor = descriptor
        if path == destination.name and dir_fd is not None and flags & os.O_CREAT:
            created_descriptor = descriptor
            real_close(descriptor)
            raise primary
        return descriptor

    def track_close(descriptor: int) -> None:
        close_calls.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(facade.os, "open", create_close_then_interrupt)
    monkeypatch.setattr(facade.os, "close", track_close)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            facade._copy_stable_regular_descriptor(
                source_descriptor,
                destination,
                expected_identity=(metadata.st_dev, metadata.st_ino),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_mode=stat.S_IMODE(metadata.st_mode),
                expected_nlink=metadata.st_nlink,
                label="coverage descriptor",
            )
    finally:
        real_close(source_descriptor)

    assert raised.value is primary
    assert "candidate name was preserved without deletion" in "\n".join(
        raised.value.__notes__
    )
    assert destination.read_bytes() == b""
    assert created_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(created_descriptor)
    assert parent_descriptor is not None
    assert close_calls.count(parent_descriptor) == 1
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    destination.unlink()


def test_stable_descriptor_copy_preserves_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    payload = b"trusted source"
    source.write_bytes(payload)
    destination.write_bytes(b"existing destination")
    source_descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    metadata = os.fstat(source_descriptor)
    descriptor_count = len(os.listdir("/proc/self/fd"))
    try:
        with pytest.raises(FileExistsError):
            facade._copy_stable_regular_descriptor(
                source_descriptor,
                destination,
                expected_identity=(metadata.st_dev, metadata.st_ino),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_mode=stat.S_IMODE(metadata.st_mode),
                expected_nlink=metadata.st_nlink,
                label="coverage descriptor",
            )
        assert len(os.listdir("/proc/self/fd")) == descriptor_count
    finally:
        os.close(source_descriptor)

    assert destination.read_bytes() == b"existing destination"


def test_discovery_preserves_ambiguous_post_create_open_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_output = tmp_path / "staged.usda"
    staged_output.write_text(_EMPTY_USDA, encoding="utf-8")
    final_targets = _targets(tmp_path)
    staged_targets = JointRiggerArtifactTargets(
        output_path=staged_output,
        diagnostics_path=tmp_path / "staged-diagnostics.json",
        result_path=tmp_path / "staged-result.json",
        publication_output_path=final_targets.output_path,
    )
    staged_artifacts = StagedJointRiggerArtifacts(final_targets, staged_targets)
    root = facade._seal_generated_root(
        staged_output,
        expected_sha256=hashlib.sha256(staged_output.read_bytes()).hexdigest(),
    )
    primary = KeyboardInterrupt("forced discovery post-create open fatal")
    real_open = facade.os.open
    real_close = facade.os.close
    parent_descriptors: list[int] = []
    active_parent_descriptors: set[int] = set()
    parent_close_calls: list[int] = []
    created_descriptor: int | None = None
    created_path: Path | None = None

    def create_close_then_interrupt(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal created_descriptor, created_path
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == tmp_path and flags & os.O_DIRECTORY:
            parent_descriptors.append(descriptor)
            active_parent_descriptors.add(descriptor)
        if (
            isinstance(path, str)
            and ".sealed-discovery-" in path
            and dir_fd is not None
            and flags & os.O_CREAT
        ):
            created_descriptor = descriptor
            created_path = tmp_path / path
            real_close(descriptor)
            raise primary
        return descriptor

    def track_close(descriptor: int) -> None:
        if descriptor in active_parent_descriptors:
            parent_close_calls.append(descriptor)
            active_parent_descriptors.remove(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(facade.os, "open", create_close_then_interrupt)
    monkeypatch.setattr(facade.os, "close", track_close)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            facade._capture_sealed_no_sidecar_dependencies(root, staged_artifacts)
    finally:
        root.cleanup()

    assert raised.value is primary
    assert "candidate name was preserved without deletion" in "\n".join(
        raised.value.__notes__
    )
    assert created_path is not None
    assert created_path.read_bytes() == b""
    assert created_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(created_descriptor)
    assert len(parent_descriptors) == 2
    assert parent_close_calls == parent_descriptors
    assert active_parent_descriptors == set()
    for parent_descriptor in set(parent_descriptors):
        with pytest.raises(OSError):
            os.fstat(parent_descriptor)
    created_path.unlink()


def test_private_directory_owner_preserves_ambiguous_post_create_mkdir_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = KeyboardInterrupt("forced private owner post-create mkdir fatal")
    real_open = facade.os.open
    real_close = facade.os.close
    real_mkdir = facade.os.mkdir
    parent_descriptor: int | None = None
    created_path: Path | None = None
    close_calls: list[int] = []

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal parent_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == tmp_path and flags & os.O_DIRECTORY:
            parent_descriptor = descriptor
        return descriptor

    def create_then_interrupt(
        path: str | os.PathLike[str],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal created_path
        real_mkdir(path, mode=mode, dir_fd=dir_fd)
        if os.fspath(path).startswith(".owner-post-create-"):
            created_path = tmp_path / os.fspath(path)
            raise primary

    def track_close(descriptor: int) -> None:
        close_calls.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(facade.os, "open", track_open)
    monkeypatch.setattr(facade.os, "mkdir", create_then_interrupt)
    monkeypatch.setattr(facade.os, "close", track_close)

    with pytest.raises(KeyboardInterrupt) as raised:
        facade._create_private_directory_owner(
            tmp_path,
            prefix=".owner-post-create-",
        )

    assert raised.value is primary
    assert "could not be bound through an owned descriptor" in "\n".join(
        raised.value.__notes__
    )
    assert created_path is not None
    assert created_path.is_dir()
    assert parent_descriptor is not None
    assert close_calls.count(parent_descriptor) == 1
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    created_path.rmdir()


def test_private_sidecar_preserves_ambiguous_post_create_mkdir_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    expected_sha256 = sidecar_dependency_bundle_sha256(sidecar)
    primary = KeyboardInterrupt("forced sidecar post-create mkdir fatal")
    real_open = facade.os.open
    real_close = facade.os.close
    real_mkdir = facade.os.mkdir
    parent_descriptor: int | None = None
    created_path: Path | None = None
    close_calls: list[int] = []

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal parent_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == tmp_path and flags & os.O_DIRECTORY:
            parent_descriptor = descriptor
        return descriptor

    def create_then_interrupt(
        path: str | os.PathLike[str],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal created_path
        real_mkdir(path, mode=mode, dir_fd=dir_fd)
        if os.fspath(path).startswith(".sidecar.sealed-"):
            created_path = tmp_path / os.fspath(path)
            raise primary

    def track_close(descriptor: int) -> None:
        close_calls.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(facade.os, "open", track_open)
    monkeypatch.setattr(facade.os, "mkdir", create_then_interrupt)
    monkeypatch.setattr(facade.os, "close", track_close)

    with pytest.raises(KeyboardInterrupt) as raised:
        facade._create_private_sidecar_snapshot(
            sidecar,
            private_parent=tmp_path,
            expected_sha256=expected_sha256,
        )

    assert raised.value is primary
    assert "could not be bound through an owned descriptor" in "\n".join(
        raised.value.__notes__
    )
    assert created_path is not None
    assert created_path.is_dir()
    assert parent_descriptor is not None
    assert close_calls.count(parent_descriptor) >= 1
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    created_path.rmdir()


@pytest.mark.parametrize("persistent_unlink_failure", [False, True])
def test_discovery_acquisition_close_fatal_runs_name_cleanup_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persistent_unlink_failure: bool,
) -> None:
    staged_output = tmp_path / "staged.usda"
    staged_output.write_text(_EMPTY_USDA, encoding="utf-8")
    final_targets = _targets(tmp_path)
    staged_targets = JointRiggerArtifactTargets(
        output_path=staged_output,
        diagnostics_path=tmp_path / "staged-diagnostics.json",
        result_path=tmp_path / "staged-result.json",
        publication_output_path=final_targets.output_path,
    )
    staged_artifacts = StagedJointRiggerArtifacts(
        final_targets=final_targets,
        staged_targets=staged_targets,
    )
    root = facade._seal_generated_root(
        staged_output,
        expected_sha256=hashlib.sha256(staged_output.read_bytes()).hexdigest(),
    )
    primary = KeyboardInterrupt("forced discovery descriptor close fatal")
    unlink_error = SystemExit("forced discovery placeholder cleanup fatal")
    real_mkstemp = facade.tempfile.mkstemp
    real_close = facade.os.close
    real_remove = facade._remove_descriptor_entry
    discovery_descriptor: int | None = None
    discovery_path: Path | None = None
    close_calls = 0

    def track_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        nonlocal discovery_descriptor, discovery_path
        descriptor, value = real_mkstemp(*args, **kwargs)
        discovery_descriptor = descriptor
        discovery_path = Path(value)
        return descriptor, value

    def close_then_fail(descriptor: int) -> None:
        nonlocal close_calls
        if descriptor == discovery_descriptor:
            close_calls += 1
            real_close(descriptor)
            raise primary
        real_close(descriptor)

    def maybe_fail_removal(*args: object, **kwargs: object) -> None:
        if persistent_unlink_failure and "placeholder" in str(kwargs.get("label")):
            raise unlink_error
        real_remove(*args, **kwargs)

    monkeypatch.setattr(facade.tempfile, "mkstemp", track_mkstemp)
    monkeypatch.setattr(facade.os, "close", close_then_fail)
    monkeypatch.setattr(facade, "_remove_descriptor_entry", maybe_fail_removal)
    try:
        expected_error = SystemExit if persistent_unlink_failure else KeyboardInterrupt
        with pytest.raises(expected_error) as raised:
            facade._capture_sealed_no_sidecar_dependencies(root, staged_artifacts)
    finally:
        monkeypatch.setattr(facade.os, "close", real_close)
        root.cleanup()

    assert raised.value is (unlink_error if persistent_unlink_failure else primary)
    assert close_calls == 1
    assert discovery_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(discovery_descriptor)
    assert discovery_path is not None
    if persistent_unlink_failure:
        assert discovery_path.exists()
        assert "forced discovery descriptor close fatal" in "\n".join(
            raised.value.__notes__
        )
        discovery_path.unlink()
    else:
        assert not discovery_path.exists()


@pytest.mark.parametrize("persistent_rebind_failure", [False, True])
def test_discovery_placeholder_identity_fatal_rebinds_only_through_mkstemp_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persistent_rebind_failure: bool,
) -> None:
    staged_output = tmp_path / "staged.usda"
    staged_output.write_text(_EMPTY_USDA, encoding="utf-8")
    final_targets = _targets(tmp_path)
    staged_targets = JointRiggerArtifactTargets(
        output_path=staged_output,
        diagnostics_path=tmp_path / "staged-diagnostics.json",
        result_path=tmp_path / "staged-result.json",
        publication_output_path=final_targets.output_path,
    )
    staged_artifacts = StagedJointRiggerArtifacts(final_targets, staged_targets)
    root = facade._seal_generated_root(
        staged_output,
        expected_sha256=hashlib.sha256(staged_output.read_bytes()).hexdigest(),
    )
    primary = KeyboardInterrupt("forced initial discovery identity fatal")
    cleanup_error = SystemExit("forced persistent discovery identity fatal")
    real_mkstemp = facade.tempfile.mkstemp
    real_fstat = facade.os.fstat
    real_close = facade.os.close
    discovery_descriptor: int | None = None
    discovery_path: Path | None = None
    identity_calls = 0
    close_calls: list[int] = []

    def track_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        nonlocal discovery_descriptor, discovery_path
        descriptor, value = real_mkstemp(*args, **kwargs)
        discovery_descriptor = descriptor
        discovery_path = Path(value)
        return descriptor, value

    def fail_identity(descriptor: int) -> os.stat_result:
        nonlocal identity_calls
        if descriptor == discovery_descriptor:
            identity_calls += 1
            if identity_calls == 1:
                raise primary
            if persistent_rebind_failure and identity_calls == 2:
                raise cleanup_error
        return real_fstat(descriptor)

    def track_close(descriptor: int) -> None:
        close_calls.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(facade.tempfile, "mkstemp", track_mkstemp)
    monkeypatch.setattr(facade.os, "fstat", fail_identity)
    monkeypatch.setattr(facade.os, "close", track_close)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            facade._capture_sealed_no_sidecar_dependencies(root, staged_artifacts)
    finally:
        monkeypatch.setattr(facade.os, "close", real_close)
        root.cleanup()

    assert raised.value is primary
    assert identity_calls == (2 if persistent_rebind_failure else 3)
    assert discovery_descriptor is not None
    assert close_calls.count(discovery_descriptor) == 1
    with pytest.raises(OSError):
        os.fstat(discovery_descriptor)
    assert discovery_path is not None
    if persistent_rebind_failure:
        assert discovery_path.exists()
        assert "forced persistent discovery identity fatal" in "\n".join(
            raised.value.__notes__
        )
        discovery_path.unlink()
    else:
        assert not discovery_path.exists()


def test_discovery_placeholder_preserved_when_parent_cannot_be_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_output = tmp_path / "staged.usda"
    staged_output.write_text(_EMPTY_USDA, encoding="utf-8")
    final_targets = _targets(tmp_path)
    staged_targets = JointRiggerArtifactTargets(
        output_path=staged_output,
        diagnostics_path=tmp_path / "staged-diagnostics.json",
        result_path=tmp_path / "staged-result.json",
        publication_output_path=final_targets.output_path,
    )
    staged_artifacts = StagedJointRiggerArtifacts(final_targets, staged_targets)
    root = facade._seal_generated_root(
        staged_output,
        expected_sha256=hashlib.sha256(staged_output.read_bytes()).hexdigest(),
    )
    primary = KeyboardInterrupt("forced discovery parent bind failure")
    real_mkstemp = facade.tempfile.mkstemp
    discovery_descriptor: int | None = None
    discovery_path: Path | None = None

    def track_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        nonlocal discovery_descriptor, discovery_path
        descriptor, value = real_mkstemp(*args, **kwargs)
        discovery_descriptor = descriptor
        discovery_path = Path(value)
        return descriptor, value

    def fail_parent_bind(destination: Path) -> int:
        del destination
        raise primary

    monkeypatch.setattr(facade.tempfile, "mkstemp", track_mkstemp)
    monkeypatch.setattr(facade, "_open_stable_copy_parent", fail_parent_bind)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            facade._capture_sealed_no_sidecar_dependencies(root, staged_artifacts)
    finally:
        root.cleanup()

    assert raised.value is primary
    assert "private name was preserved" in "\n".join(raised.value.__notes__)
    assert discovery_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(discovery_descriptor)
    assert discovery_path is not None
    assert discovery_path.exists()
    discovery_path.unlink()


def test_discovery_placeholder_substitution_preserves_foreign_and_displaced_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_output = tmp_path / "staged.usda"
    staged_output.write_text(_EMPTY_USDA, encoding="utf-8")
    final_targets = _targets(tmp_path)
    staged_targets = JointRiggerArtifactTargets(
        output_path=staged_output,
        diagnostics_path=tmp_path / "staged-diagnostics.json",
        result_path=tmp_path / "staged-result.json",
        publication_output_path=final_targets.output_path,
    )
    staged_artifacts = StagedJointRiggerArtifacts(final_targets, staged_targets)
    root = facade._seal_generated_root(
        staged_output,
        expected_sha256=hashlib.sha256(staged_output.read_bytes()).hexdigest(),
    )
    real_remove = facade._remove_descriptor_entry
    real_close = facade.os.close
    placeholder: Path | None = None
    displaced = tmp_path / "displaced-owned-placeholder.usda"
    substituted = False

    def substitute_before_removal(*args: object, **kwargs: object) -> None:
        nonlocal placeholder, substituted
        label = str(kwargs.get("label"))
        if "placeholder" in label and not substituted:
            substituted = True
            placeholder = tmp_path / str(args[1])
            placeholder.rename(displaced)
            placeholder.write_bytes(b"foreign replacement")
        real_remove(*args, **kwargs)

    monkeypatch.setattr(facade, "_remove_descriptor_entry", substitute_before_removal)
    try:
        with pytest.raises(RuntimeError, match="replacement preserved"):
            facade._capture_sealed_no_sidecar_dependencies(root, staged_artifacts)
    finally:
        monkeypatch.setattr(facade.os, "close", real_close)
        root.cleanup()

    assert substituted
    assert placeholder is not None
    assert placeholder.read_bytes() == b"foreign replacement"
    assert displaced.read_bytes() == b""


def test_copied_discovery_substitution_preserves_foreign_and_displaced_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_output = tmp_path / "staged.usda"
    staged_output.write_text(_EMPTY_USDA, encoding="utf-8")
    final_targets = _targets(tmp_path)
    staged_targets = JointRiggerArtifactTargets(
        output_path=staged_output,
        diagnostics_path=tmp_path / "staged-diagnostics.json",
        result_path=tmp_path / "staged-result.json",
        publication_output_path=final_targets.output_path,
    )
    staged_artifacts = StagedJointRiggerArtifacts(final_targets, staged_targets)
    root = facade._seal_generated_root(
        staged_output,
        expected_sha256=hashlib.sha256(staged_output.read_bytes()).hexdigest(),
    )
    primary = KeyboardInterrupt("forced copied discovery substitution failure")
    discovery: Path | None = None
    displaced = tmp_path / "displaced-owned-discovery.usda"

    def substitute_then_fail(path: Path, *, label: str) -> tuple[Path, ...]:
        nonlocal discovery
        del label
        discovery = path
        path.rename(displaced)
        path.write_bytes(b"foreign replacement")
        raise primary

    monkeypatch.setattr(facade, "_local_usd_dependency_paths", substitute_then_fail)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            facade._capture_sealed_no_sidecar_dependencies(root, staged_artifacts)
    finally:
        root.cleanup()

    assert raised.value is primary
    assert "replacement preserved" in "\n".join(raised.value.__notes__)
    assert discovery is not None
    assert discovery.read_bytes() == b"foreign replacement"
    assert displaced.read_bytes() == _EMPTY_USDA.encode()


def test_dependency_snapshot_keeps_primary_over_descriptor_close_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = tmp_path / "dependency.usda"
    dependency.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    primary = KeyboardInterrupt("forced dependency hash primary")
    cleanup_error = SystemExit("forced dependency descriptor close fatal")
    real_open = facade.os.open
    real_close = facade.os.close
    descriptor: int | None = None
    close_calls = 0

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        return descriptor

    def fail_hash(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise primary

    def close_then_fail(candidate: int) -> None:
        nonlocal close_calls
        real_close(candidate)
        if candidate == descriptor:
            close_calls += 1
            raise cleanup_error

    monkeypatch.setattr(facade.os, "open", track_open)
    monkeypatch.setattr(facade, "_stable_descriptor_sha256", fail_hash)
    monkeypatch.setattr(facade.os, "close", close_then_fail)

    with pytest.raises(KeyboardInterrupt) as raised:
        facade._open_sealed_dependency_snapshot(dependency)

    assert raised.value is primary
    assert close_calls == 1
    assert descriptor is not None
    with pytest.raises(OSError):
        os.fstat(descriptor)
    assert "forced dependency descriptor close fatal" in "\n".join(primary.__notes__)


def test_private_sidecar_seal_keeps_primary_over_file_close_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    member = sidecar / "asset.usda"
    member.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    root_descriptor = os.open(
        sidecar,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    primary = KeyboardInterrupt("forced sidecar file seal primary")
    cleanup_error = SystemExit("forced sidecar file close fatal")
    real_open = facade.os.open
    real_close = facade.os.close
    real_fchmod = facade.os.fchmod
    file_descriptor: int | None = None
    close_calls = 0
    seal_started = False

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal file_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == member.name and dir_fd == root_descriptor:
            file_descriptor = descriptor
        return descriptor

    def fail_file_fchmod(descriptor: int, mode: int) -> None:
        nonlocal seal_started
        if descriptor == file_descriptor:
            seal_started = True
            raise primary
        real_fchmod(descriptor, mode)

    def close_then_fail(descriptor: int) -> None:
        nonlocal close_calls
        real_close(descriptor)
        if descriptor == file_descriptor and seal_started:
            close_calls += 1
            raise cleanup_error

    monkeypatch.setattr(facade.os, "open", track_open)
    monkeypatch.setattr(facade.os, "fchmod", fail_file_fchmod)
    monkeypatch.setattr(facade.os, "close", close_then_fail)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            facade._seal_directory_descriptor_tree(root_descriptor)
    finally:
        real_close(root_descriptor)

    assert raised.value is primary
    assert close_calls == 1
    assert file_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(file_descriptor)
    assert "forced sidecar file close fatal" in "\n".join(primary.__notes__)


@pytest.mark.parametrize("operation", ["opaque", "report", "hash"])
def test_reader_helpers_keep_primary_over_descriptor_close_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("payload", encoding="utf-8")
    primary = KeyboardInterrupt(f"forced {operation} reader primary")
    cleanup_error = SystemExit(f"forced {operation} reader close fatal")
    real_open = facade.os.open
    real_fstat = facade.os.fstat
    real_close = facade.os.close
    descriptor: int | None = None
    close_calls = 0

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal descriptor
        candidate = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == source:
            descriptor = candidate
        return candidate

    def fail_fstat(candidate: int) -> os.stat_result:
        if candidate == descriptor:
            raise primary
        return real_fstat(candidate)

    def close_then_fail(candidate: int) -> None:
        nonlocal close_calls
        real_close(candidate)
        if candidate == descriptor:
            close_calls += 1
            raise cleanup_error

    monkeypatch.setattr(facade.os, "open", track_open)
    monkeypatch.setattr(facade.os, "fstat", fail_fstat)
    monkeypatch.setattr(facade.os, "close", close_then_fail)

    with pytest.raises(KeyboardInterrupt) as raised:
        if operation == "opaque":
            facade._read_bounded_opaque_document(source)
        elif operation == "report":
            facade._load_model_report(source, object, "result")
        else:
            facade._file_sha256(source)

    assert raised.value is primary
    assert close_calls == 1
    assert descriptor is not None
    with pytest.raises(OSError):
        real_fstat(descriptor)
    assert f"forced {operation} reader close fatal" in "\n".join(primary.__notes__)


def test_file_sha256_rejects_growth_after_initial_extent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "growing.usda"
    payload = b"initial payload"
    source.write_bytes(payload)
    real_pread = os.pread
    grew = False

    def grow_before_tail_check(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal grew
        if not grew and size == 1 and offset == len(payload):
            grew = True
            with source.open("ab") as stream:
                stream.write(b"!")
        return real_pread(descriptor, size, offset)

    monkeypatch.setattr(facade.os, "pread", grow_before_tail_check)

    with pytest.raises(RuntimeError, match="grew while it was hashed"):
        facade._file_sha256(source)


def test_model_comparison_uses_contract_canonical_json(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)
    diagnostics = _result(_request(source)).diagnostics

    assert facade._canonical_model_payload(diagnostics) == canonical_json(diagnostics)


def test_nested_private_sidecar_tree_is_sealed_and_removed_through_descriptors(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "private-sidecar"
    nested = tree / "nested"
    nested.mkdir(parents=True)
    member = nested / "dependency.usda"
    member.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    descriptor = os.open(tree, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        facade._seal_directory_descriptor_tree(descriptor)
    finally:
        os.close(descriptor)

    assert tree.stat().st_mode & 0o222 == 0
    assert nested.stat().st_mode & 0o222 == 0
    assert member.stat().st_mode & 0o222 == 0

    source_descriptor = os.open(
        tree,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    source_metadata = os.fstat(source_descriptor)
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        facade._remove_private_snapshot_entry(
            parent_descriptor,
            tree.name,
            expected_identity=(source_metadata.st_dev, source_metadata.st_ino),
            source_descriptor=source_descriptor,
        )
    finally:
        os.close(source_descriptor)
        os.close(parent_descriptor)

    assert not tree.exists()


def test_private_snapshot_cleanup_binds_regular_entry_without_retained_fd(
    tmp_path: Path,
) -> None:
    reserved = tmp_path / ".report.sealed-owned"
    reserved.write_text("owned payload", encoding="utf-8")
    metadata = reserved.stat()
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        facade._remove_private_snapshot_entry(
            parent_descriptor,
            reserved.name,
            expected_identity=(metadata.st_dev, metadata.st_ino),
        )
    finally:
        os.close(parent_descriptor)

    assert not reserved.exists()


@pytest.mark.parametrize(
    "fatal_type",
    [KeyboardInterrupt, SystemExit, BaseException],
)
def test_private_snapshot_cleanup_closes_descriptors_before_fatal_propagation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fatal_type: type[BaseException],
) -> None:
    reserved = tmp_path / ".report.sealed-owned"
    reserved.write_text("owned payload", encoding="utf-8")
    writer_descriptor = os.open(reserved, os.O_WRONLY | os.O_NOFOLLOW)
    source_descriptor = os.open(reserved, os.O_RDONLY | os.O_NOFOLLOW)
    source_metadata = os.fstat(source_descriptor)
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    parent_metadata = os.fstat(parent_descriptor)
    fatal_error = fatal_type("forced fatal snapshot removal")

    def fail_removal(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise fatal_error

    monkeypatch.setattr(facade, "_remove_private_snapshot_entry", fail_removal)

    with pytest.raises(fatal_type) as raised:
        facade._cleanup_private_snapshot_resources(
            entry_name=reserved.name,
            parent_descriptor=parent_descriptor,
            parent_identity=(parent_metadata.st_dev, parent_metadata.st_ino),
            source_descriptor=source_descriptor,
            source_identity=(source_metadata.st_dev, source_metadata.st_ino),
            writer_descriptor=writer_descriptor,
        )

    assert raised.value is fatal_error
    for descriptor in (writer_descriptor, source_descriptor, parent_descriptor):
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert reserved.read_text(encoding="utf-8") == "owned payload"


def test_private_snapshot_cleanup_attaches_all_single_shot_close_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserved = tmp_path / ".report.sealed-owned"
    reserved.write_text("owned payload", encoding="utf-8")
    writer_descriptor = os.open(reserved, os.O_WRONLY | os.O_NOFOLLOW)
    source_descriptor = os.open(reserved, os.O_RDONLY | os.O_NOFOLLOW)
    source_metadata = os.fstat(source_descriptor)
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    parent_metadata = os.fstat(parent_descriptor)
    fatal_error = KeyboardInterrupt("forced fatal snapshot removal")
    real_close = facade.os.close
    close_calls: list[int] = []

    def fail_removal(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise fatal_error

    def close_once_then_fail(descriptor: int) -> None:
        close_calls.append(descriptor)
        real_close(descriptor)
        message = f"forced descriptor {descriptor} close failure"
        if descriptor == writer_descriptor:
            raise SystemExit(message)
        raise OSError(errno.EIO, message)

    monkeypatch.setattr(facade, "_remove_private_snapshot_entry", fail_removal)
    monkeypatch.setattr(facade.os, "close", close_once_then_fail)

    with pytest.raises(KeyboardInterrupt) as raised:
        facade._cleanup_private_snapshot_resources(
            entry_name=reserved.name,
            parent_descriptor=parent_descriptor,
            parent_identity=(parent_metadata.st_dev, parent_metadata.st_ino),
            source_descriptor=source_descriptor,
            source_identity=(source_metadata.st_dev, source_metadata.st_ino),
            writer_descriptor=writer_descriptor,
        )

    assert raised.value is fatal_error
    assert close_calls == [writer_descriptor, source_descriptor, parent_descriptor]
    notes = "\n".join(raised.value.__notes__)
    for label, descriptor in (
        ("writer", writer_descriptor),
        ("source", source_descriptor),
        ("parent", parent_descriptor),
    ):
        assert f"close {label} descriptor" in notes
        assert f"forced descriptor {descriptor} close failure" in notes
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize(
    "primary_error",
    [
        KeyboardInterrupt("forced report creation termination"),
        JointRiggerArtifactError("forced typed report creation failure"),
    ],
)
def test_private_report_creation_preserves_primary_over_fatal_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_error: BaseException,
) -> None:
    real_open = facade.os.open
    opened_descriptors: list[int] = []
    cleanup_error = SystemExit("forced fatal snapshot cleanup")

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened_descriptors.append(descriptor)
        return descriptor

    def fail_write(descriptor: int, payload: bytes) -> int:
        del descriptor, payload
        raise primary_error

    def fail_removal(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise cleanup_error

    monkeypatch.setattr(facade.os, "open", track_open)
    monkeypatch.setattr(facade.os, "write", fail_write)
    monkeypatch.setattr(facade, "_remove_private_snapshot_entry", fail_removal)

    with pytest.raises(type(primary_error)) as raised:
        facade._create_private_report_snapshot(
            tmp_path / "result.json",
            b"{}",
            label="result",
        )

    assert raised.value is primary_error
    assert "SystemExit: forced fatal snapshot cleanup" in "\n".join(
        raised.value.__notes__
    )
    assert len(opened_descriptors) == 2
    for descriptor in opened_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("entry_kind", ["file", "directory"])
def test_private_snapshot_cleanup_preserves_substituted_reserved_entry(
    tmp_path: Path,
    entry_kind: Literal["file", "directory"],
) -> None:
    reserved = tmp_path / ".artifact.sealed-owned"
    displaced = tmp_path / "displaced-owned-snapshot"
    if entry_kind == "directory":
        reserved.mkdir()
        (reserved / "owned.txt").write_text("owned payload", encoding="utf-8")
        source_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    else:
        reserved.write_text("owned payload", encoding="utf-8")
        source_flags = os.O_RDONLY | os.O_NOFOLLOW
    source_descriptor = os.open(reserved, source_flags)
    source_metadata = os.fstat(source_descriptor)
    source_identity = (source_metadata.st_dev, source_metadata.st_ino)
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    parent_metadata = os.fstat(parent_descriptor)

    reserved.rename(displaced)
    if entry_kind == "directory":
        reserved.mkdir()
        replacement_member = reserved / "replacement.txt"
        replacement_member.write_text("replacement payload", encoding="utf-8")
    else:
        reserved.write_text("replacement payload", encoding="utf-8")

    errors = facade._cleanup_private_snapshot_resources(
        entry_name=reserved.name,
        parent_descriptor=parent_descriptor,
        parent_identity=(parent_metadata.st_dev, parent_metadata.st_ino),
        source_descriptor=source_descriptor,
        source_identity=source_identity,
    )

    assert errors == [
        "remove reserved entry: private snapshot entry changed inode; "
        "refusing deletion; replacement preserved"
    ]
    assert (displaced.stat().st_dev, displaced.stat().st_ino) == source_identity
    if entry_kind == "directory":
        assert replacement_member.read_text(encoding="utf-8") == ("replacement payload")
        assert (displaced / "owned.txt").read_text(encoding="utf-8") == (
            "owned payload"
        )
    else:
        assert reserved.read_text(encoding="utf-8") == "replacement payload"
        assert displaced.read_text(encoding="utf-8") == "owned payload"
    with pytest.raises(OSError):
        os.fstat(source_descriptor)
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)


@pytest.mark.parametrize("entry_kind", ["file", "directory"])
def test_private_snapshot_quarantine_preserves_root_substitution_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: Literal["file", "directory"],
) -> None:
    reserved = tmp_path / ".artifact.sealed-owned"
    displaced = tmp_path / "displaced-owned-snapshot"
    if entry_kind == "directory":
        reserved.mkdir()
        (reserved / "owned.txt").write_text("owned payload", encoding="utf-8")
        source_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    else:
        reserved.write_text("owned payload", encoding="utf-8")
        source_flags = os.O_RDONLY | os.O_NOFOLLOW
    source_descriptor = os.open(reserved, source_flags)
    source_metadata = os.fstat(source_descriptor)
    source_identity = (source_metadata.st_dev, source_metadata.st_ino)
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    real_rename = artifacts_module._rename_descriptor_entry_noreplace
    substituted = False

    def substitute_before_quarantine(
        source_parent_descriptor: int,
        source_name: str,
        target_parent_descriptor: int,
        target_name: str,
        *,
        label: str,
    ) -> None:
        nonlocal substituted
        if not substituted and source_name == reserved.name:
            substituted = True
            reserved.rename(displaced)
            if entry_kind == "directory":
                reserved.mkdir()
                (reserved / "replacement.txt").write_text(
                    "replacement payload",
                    encoding="utf-8",
                )
            else:
                reserved.write_text("replacement payload", encoding="utf-8")
        real_rename(
            source_parent_descriptor,
            source_name,
            target_parent_descriptor,
            target_name,
            label=label,
        )

    monkeypatch.setattr(
        artifacts_module,
        "_rename_descriptor_entry_noreplace",
        substitute_before_quarantine,
    )
    try:
        with pytest.raises(
            RuntimeError, match="changed inode during atomic quarantine"
        ):
            facade._remove_private_snapshot_entry(
                parent_descriptor,
                reserved.name,
                expected_identity=source_identity,
                source_descriptor=source_descriptor,
            )
    finally:
        os.close(source_descriptor)
        os.close(parent_descriptor)

    assert substituted
    if entry_kind == "directory":
        assert (reserved / "replacement.txt").read_text(encoding="utf-8") == (
            "replacement payload"
        )
        assert (displaced / "owned.txt").read_text(encoding="utf-8") == (
            "owned payload"
        )
    else:
        assert reserved.read_text(encoding="utf-8") == "replacement payload"
        assert displaced.read_text(encoding="utf-8") == "owned payload"
    assert not any(tmp_path.glob(".joint-rigger.cleanup-*"))


@pytest.mark.parametrize("mount_entry", ["nested_directory", "nested_file"])
def test_private_snapshot_cleanup_rejects_nested_mount_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mount_entry: Literal["nested_directory", "nested_file"],
) -> None:
    reserved = tmp_path / ".sidecar.sealed-owned"
    nested = reserved / "nested"
    nested.mkdir(parents=True)
    member = nested / "member.bin"
    member.write_bytes(b"sentinel bytes")
    source_descriptor = os.open(
        reserved,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    source_metadata = os.fstat(source_descriptor)
    source_identity = (source_metadata.st_dev, source_metadata.st_ino)
    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    target = nested if mount_entry == "nested_directory" else member
    target_identity = (target.stat().st_dev, target.stat().st_ino)
    real_mount_id = artifacts_module._descriptor_mount_id

    def inject_distinct_mount_id(descriptor: int) -> int:
        mount_id = real_mount_id(descriptor)
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == target_identity:
            return mount_id + 10_000
        return mount_id

    monkeypatch.setattr(
        artifacts_module,
        "_descriptor_mount_id",
        inject_distinct_mount_id,
    )
    try:
        with pytest.raises(ValueError, match="cleanup crossed a mount"):
            facade._remove_private_snapshot_entry(
                parent_descriptor,
                reserved.name,
                expected_identity=source_identity,
                source_descriptor=source_descriptor,
            )
    finally:
        os.close(source_descriptor)
        os.close(parent_descriptor)

    assert member.read_bytes() == b"sentinel bytes"
    assert reserved.is_dir()


def test_private_snapshot_cleanup_rejects_real_nested_bind_mount(
    tmp_path: Path,
) -> None:
    unshare = shutil.which("unshare")
    mount = shutil.which("mount")
    if unshare is None or mount is None:
        pytest.skip("Linux mount-boundary test requires unshare and mount")

    probe_source = tmp_path / "probe-source"
    probe_target = tmp_path / "probe-target"
    probe_source.mkdir()
    probe_target.mkdir()
    probe_script = textwrap.dedent(
        """
        import subprocess
        import sys

        mount, source, target = sys.argv[1:]
        try:
            subprocess.run(
                [mount, "--make-rprivate", "/"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            subprocess.run(
                [mount, "--bind", source, target],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.CalledProcessError as exc:
            detail = "\\n".join(
                part.strip() for part in (exc.stdout, exc.stderr) if part
            )
            print(detail, file=sys.stderr)
            raise SystemExit(exc.returncode or 1) from exc
        """
    )
    probe = subprocess.run(
        [
            unshare,
            "--user",
            "--map-root-user",
            "--mount",
            "--fork",
            sys.executable,
            "-c",
            probe_script,
            mount,
            str(probe_source),
            str(probe_target),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if probe.returncode != 0:
        if _mount_capability_is_unavailable(probe):
            pytest.skip("unprivileged user/mount namespaces are unavailable")
        pytest.fail(
            "mount capability probe failed unexpectedly: "
            + (probe.stderr.strip() or probe.stdout.strip())
        )

    reserved = tmp_path / ".sidecar.sealed-owned"
    nested = reserved / "nested"
    nested.mkdir(parents=True)
    (nested / "underlying.txt").write_text("underlying", encoding="utf-8")
    mounted_source = tmp_path / "mounted-source"
    mounted_source.mkdir()
    (mounted_source / "sentinel.txt").write_text("mounted", encoding="utf-8")
    script = textwrap.dedent(
        """
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        from world_understanding.functions.physics.joint_rigger import facade

        root = Path(sys.argv[1])
        mount = sys.argv[2]
        denial_markers = json.loads(sys.argv[3])
        reserved = root / ".sidecar.sealed-owned"
        nested = reserved / "nested"
        mounted_source = root / "mounted-source"
        try:
            subprocess.run(
                [mount, "--make-rprivate", "/"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            subprocess.run(
                [mount, "--bind", str(mounted_source), str(nested)],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.CalledProcessError as exc:
            detail = "\\n".join(
                part.strip() for part in (exc.stdout, exc.stderr) if part
            )
            print(detail, file=sys.stderr)
            if any(marker in detail.lower() for marker in denial_markers):
                raise SystemExit(77) from exc
            raise

        parent_descriptor = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        source_descriptor = os.open(
            reserved,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        source_metadata = os.fstat(source_descriptor)
        try:
            try:
                facade._remove_private_snapshot_entry(
                    parent_descriptor,
                    reserved.name,
                    expected_identity=(
                        source_metadata.st_dev,
                        source_metadata.st_ino,
                    ),
                    source_descriptor=source_descriptor,
                )
            except ValueError as exc:
                assert "cleanup crossed a mount" in str(exc)
            else:
                raise AssertionError("nested bind mount was not rejected")
            assert (nested / "sentinel.txt").read_text(encoding="utf-8") == "mounted"
            assert reserved.is_dir()
        finally:
            os.close(source_descriptor)
            os.close(parent_descriptor)
        """
    )
    completed = subprocess.run(
        [
            unshare,
            "--user",
            "--map-root-user",
            "--mount",
            "--fork",
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            mount,
            json.dumps(_MOUNT_CAPABILITY_DENIAL_MARKERS),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode == 77 or _mount_capability_is_unavailable(completed):
        pytest.skip("unprivileged user/mount namespaces are unavailable")
    assert completed.returncode == 0, completed.stderr
    assert (nested / "underlying.txt").read_text(encoding="utf-8") == "underlying"
    assert (mounted_source / "sentinel.txt").read_text(encoding="utf-8") == "mounted"


def test_dependency_swap_after_stat_uses_typed_error_and_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = tmp_path / "dependency.usda"
    replacement = tmp_path / "replacement.usda"
    displaced = tmp_path / "displaced.usda"
    dependency.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    replacement.write_text(_EMPTY_USDA, encoding="utf-8")
    real_open = facade.os.open
    opened_descriptor: int | None = None
    swapped = False

    def swap_before_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal opened_descriptor, swapped
        if Path(path) == dependency and dir_fd is None and not swapped:
            swapped = True
            dependency.replace(displaced)
            replacement.replace(dependency)
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == dependency and dir_fd is None:
            opened_descriptor = descriptor
        return descriptor

    monkeypatch.setattr(facade.os, "open", swap_before_open)

    with pytest.raises(
        JointRiggerArtifactError,
        match="changed before it was sealed",
    ):
        facade._open_sealed_dependency_snapshot(dependency)

    assert swapped
    assert opened_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(opened_descriptor)


def test_descriptor_copy_keeps_primary_error_and_records_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    payload = b"descriptor-copy payload"
    source.write_bytes(payload)
    source_descriptor = os.open(source, os.O_RDONLY)
    metadata = os.fstat(source_descriptor)
    expected = RuntimeError("forced descriptor-copy fsync failure")
    real_close = facade.os.close
    real_open = facade.os.open
    target_descriptor: int | None = None
    parent_descriptor: int | None = None
    close_failure_injected = False
    displaced = tmp_path / "displaced-owned-copy.bin"

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal parent_descriptor, target_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == tmp_path and flags & os.O_DIRECTORY:
            parent_descriptor = descriptor
        if path == destination.name and dir_fd is not None and flags & os.O_CREAT:
            target_descriptor = descriptor
        return descriptor

    def fail_fsync(descriptor: int) -> None:
        assert descriptor == target_descriptor
        destination.rename(displaced)
        destination.write_bytes(b"foreign replacement")
        raise expected

    def close_target_then_fail(descriptor: int) -> None:
        nonlocal close_failure_injected
        real_close(descriptor)
        if descriptor == target_descriptor and not close_failure_injected:
            close_failure_injected = True
            raise OSError(errno.EIO, "forced descriptor-copy close failure")

    monkeypatch.setattr(facade.os, "open", track_open)
    monkeypatch.setattr(facade.os, "fsync", fail_fsync)
    monkeypatch.setattr(facade.os, "close", close_target_then_fail)
    try:
        with pytest.raises(RuntimeError) as raised:
            facade._copy_stable_regular_descriptor(
                source_descriptor,
                destination,
                expected_identity=(metadata.st_dev, metadata.st_ino),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_mode=stat.S_IMODE(metadata.st_mode),
                expected_nlink=metadata.st_nlink,
                label="coverage descriptor",
            )
    finally:
        real_close(source_descriptor)

    assert raised.value is expected
    assert close_failure_injected
    assert any(
        "forced descriptor-copy close failure" in note
        for note in raised.value.__notes__
    )
    assert any("replacement preserved" in note for note in raised.value.__notes__)
    assert destination.read_bytes() == b"foreign replacement"
    assert displaced.read_bytes() == payload
    assert target_descriptor is not None
    assert parent_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(target_descriptor)
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    assert not any(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_descriptor_copy_reports_success_path_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    payload = b"descriptor-copy payload"
    source.write_bytes(payload)
    source_descriptor = os.open(source, os.O_RDONLY)
    metadata = os.fstat(source_descriptor)
    real_open = facade.os.open
    real_close = facade.os.close
    target_descriptor: int | None = None
    close_failure_injected = False

    def track_target_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal target_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == destination.name and dir_fd is not None and flags & os.O_CREAT:
            target_descriptor = descriptor
        return descriptor

    def close_target_then_fail(descriptor: int) -> None:
        nonlocal close_failure_injected
        real_close(descriptor)
        if descriptor == target_descriptor and not close_failure_injected:
            close_failure_injected = True
            raise OSError(errno.EIO, "forced successful-copy close failure")

    monkeypatch.setattr(facade.os, "open", track_target_open)
    monkeypatch.setattr(facade.os, "close", close_target_then_fail)
    try:
        with pytest.raises(OSError, match="forced successful-copy close failure"):
            facade._copy_stable_regular_descriptor(
                source_descriptor,
                destination,
                expected_identity=(metadata.st_dev, metadata.st_ino),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_mode=stat.S_IMODE(metadata.st_mode),
                expected_nlink=metadata.st_nlink,
                label="coverage descriptor",
            )
    finally:
        real_close(source_descriptor)

    assert close_failure_injected
    assert not destination.exists()


@pytest.mark.parametrize("persistent_rebind_failure", [False, True])
def test_descriptor_copy_identity_fatal_rebinds_only_through_target_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persistent_rebind_failure: bool,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    payload = b"descriptor-copy payload"
    source.write_bytes(payload)
    source_descriptor = os.open(source, os.O_RDONLY)
    metadata = os.fstat(source_descriptor)
    primary = KeyboardInterrupt("forced initial target identity fatal")
    cleanup_error = SystemExit("forced persistent target identity fatal")
    real_open = facade.os.open
    real_fstat = facade.os.fstat
    real_close = facade.os.close
    target_descriptor: int | None = None
    parent_descriptor: int | None = None
    target_fstat_calls = 0
    close_calls: list[int] = []

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal parent_descriptor, target_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == tmp_path and flags & os.O_DIRECTORY:
            parent_descriptor = descriptor
        if path == destination.name and dir_fd is not None and flags & os.O_CREAT:
            target_descriptor = descriptor
        return descriptor

    def fail_target_identity(descriptor: int) -> os.stat_result:
        nonlocal target_fstat_calls
        if descriptor == target_descriptor:
            target_fstat_calls += 1
            if target_fstat_calls == 1:
                raise primary
            if persistent_rebind_failure and target_fstat_calls == 2:
                raise cleanup_error
        return real_fstat(descriptor)

    def track_close(descriptor: int) -> None:
        close_calls.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(facade.os, "open", track_open)
    monkeypatch.setattr(facade.os, "fstat", fail_target_identity)
    monkeypatch.setattr(facade.os, "close", track_close)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            facade._copy_stable_regular_descriptor(
                source_descriptor,
                destination,
                expected_identity=(metadata.st_dev, metadata.st_ino),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_mode=stat.S_IMODE(metadata.st_mode),
                expected_nlink=metadata.st_nlink,
                label="coverage descriptor",
            )
    finally:
        real_close(source_descriptor)

    assert raised.value is primary
    assert target_fstat_calls == (2 if persistent_rebind_failure else 3)
    assert target_descriptor is not None
    assert parent_descriptor is not None
    assert close_calls.count(target_descriptor) == 1
    assert close_calls.count(parent_descriptor) == 1
    with pytest.raises(OSError):
        os.fstat(target_descriptor)
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    if persistent_rebind_failure:
        assert destination.exists()
        assert "forced persistent target identity fatal" in "\n".join(
            raised.value.__notes__
        )
        destination.unlink()
    else:
        assert not destination.exists()


def test_descriptor_copy_rejects_substituted_supplied_parent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    source_descriptor = os.open(source, os.O_RDONLY)
    metadata = os.fstat(source_descriptor)
    destination_parent = tmp_path / "destination-parent"
    destination_parent.mkdir()
    destination = destination_parent / "destination.bin"
    parent_descriptor = facade._open_stable_copy_parent(destination)
    displaced_parent = tmp_path / "displaced-parent"
    destination_parent.rename(displaced_parent)
    destination_parent.mkdir()
    (destination_parent / "foreign.txt").write_text("foreign", encoding="utf-8")

    try:
        with pytest.raises(RuntimeError, match="parent changed"):
            facade._copy_stable_regular_descriptor(
                source_descriptor,
                destination,
                expected_identity=(metadata.st_dev, metadata.st_ino),
                expected_sha256=hashlib.sha256(b"payload").hexdigest(),
                expected_mode=stat.S_IMODE(metadata.st_mode),
                expected_nlink=metadata.st_nlink,
                label="coverage descriptor",
                destination_parent_descriptor=parent_descriptor,
            )
    finally:
        os.close(parent_descriptor)
        os.close(source_descriptor)

    assert not destination.exists()
    assert not (displaced_parent / destination.name).exists()
    assert (destination_parent / "foreign.txt").read_text(encoding="utf-8") == (
        "foreign"
    )


def test_descriptor_copy_parent_close_failure_rolls_back_created_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    payload = b"payload"
    source.write_bytes(payload)
    source_descriptor = os.open(source, os.O_RDONLY)
    metadata = os.fstat(source_descriptor)
    expected = OSError(errno.EIO, "forced stable-copy parent close failure")
    real_open = facade.os.open
    real_close = facade.os.close
    parent_descriptor: int | None = None
    failure_injected = False

    def track_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal parent_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if (
            parent_descriptor is None
            and Path(path) == tmp_path
            and flags & os.O_DIRECTORY
        ):
            parent_descriptor = descriptor
        return descriptor

    def close_parent_then_fail(descriptor: int) -> None:
        nonlocal failure_injected
        real_close(descriptor)
        if descriptor == parent_descriptor and not failure_injected:
            failure_injected = True
            raise expected

    monkeypatch.setattr(facade.os, "open", track_open)
    monkeypatch.setattr(facade.os, "close", close_parent_then_fail)
    try:
        with pytest.raises(OSError) as raised:
            facade._copy_stable_regular_descriptor(
                source_descriptor,
                destination,
                expected_identity=(metadata.st_dev, metadata.st_ino),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_mode=stat.S_IMODE(metadata.st_mode),
                expected_nlink=metadata.st_nlink,
                label="coverage descriptor",
            )
    finally:
        real_close(source_descriptor)

    assert raised.value is expected
    assert failure_injected
    assert not destination.exists()


@pytest.mark.parametrize("artifact_kind", ["root", "sidecar"])
def test_sealed_generated_typed_revalidation_errors_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: Literal["root", "sidecar"],
) -> None:
    root_path = tmp_path / "generated.usda"
    root_path.write_text(_EMPTY_USDA, encoding="utf-8")
    root = facade._seal_generated_root(
        root_path,
        expected_sha256=hashlib.sha256(_EMPTY_USDA.encode()).hexdigest(),
    )
    sidecar = None
    if artifact_kind == "sidecar":
        backend_sidecar = tmp_path / "backend-sidecar"
        backend_sidecar.mkdir()
        (backend_sidecar / "dependency.usda").write_text(
            _DEPENDENCY_USDA,
            encoding="utf-8",
        )
        sidecar = facade._create_private_sidecar_snapshot(
            backend_sidecar,
            private_parent=tmp_path,
            expected_sha256=sidecar_dependency_bundle_sha256(backend_sidecar),
        )
    sealed = facade._SealedGeneratedArtifacts(root=root, sidecar=sidecar)
    expected = JointRiggerArtifactError(f"forced typed {artifact_kind} failure")

    def fail_revalidation(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise expected

    if artifact_kind == "root":
        monkeypatch.setattr(facade, "_stable_descriptor_sha256", fail_revalidation)
    else:
        monkeypatch.setattr(
            facade,
            "directory_descriptor_tree_sha256",
            fail_revalidation,
        )
    try:
        with pytest.raises(JointRiggerArtifactError) as raised:
            facade._revalidate_sealed_generated_artifacts(sealed)
    finally:
        sealed.cleanup()

    assert raised.value is expected


def test_sealed_publication_projection_reports_missing_and_added_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = tmp_path / "dependency.usda"
    dependency.write_text(_DEPENDENCY_USDA, encoding="utf-8")
    staged_output = tmp_path / ".rigged.stage-test.usda"
    staged_output.write_text(
        _root_with_reference(dependency.as_posix()),
        encoding="utf-8",
    )
    final_targets = _targets(tmp_path)
    staged_targets = JointRiggerArtifactTargets(
        output_path=staged_output,
        diagnostics_path=tmp_path / ".diagnostics.stage-test.json",
        result_path=tmp_path / ".result.stage-test.json",
        publication_output_path=final_targets.output_path,
    )
    staged_artifacts = StagedJointRiggerArtifacts(
        final_targets=final_targets,
        staged_targets=staged_targets,
    )
    root = facade._seal_generated_root(
        staged_output,
        expected_sha256=hashlib.sha256(staged_output.read_bytes()).hexdigest(),
    )
    dependencies: tuple[facade._SealedDependencySnapshot, ...] = ()
    try:
        dependencies, records = facade._capture_sealed_no_sidecar_dependencies(
            root,
            staged_artifacts,
        )
        expected_identity = identify_usd_artifact(
            staged_output,
            uri=str(final_targets.output_path),
        )
        assert expected_identity.dependency_bundle_sha256 is not None
        unexpected = tmp_path / "unexpected.usda"
        unexpected.write_text(_EMPTY_USDA, encoding="utf-8")
        monkeypatch.setattr(
            facade,
            "_local_usd_dependency_paths",
            lambda path, *, label: (unexpected,),
        )

        with pytest.raises(
            JointRiggerArtifactError,
            match="dependency closure changes.*missing=.*added=",
        ):
            facade._validate_sealed_no_sidecar_composition(
                root,
                dependencies,
                records,
                staged_artifacts,
                uri=str(final_targets.output_path),
                expected_bundle_sha256=(expected_identity.dependency_bundle_sha256),
            )
    finally:
        for snapshot in dependencies:
            snapshot.cleanup()
        root.cleanup()


def test_planned_decisions_cover_absent_nested_physics_facts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    _write_empty_usda(source)

    def provenance(label: str) -> FieldProvenanceV1:
        return FieldProvenanceV1(
            source="owner_approved_plan",
            evidence=f"Owner-approved {label}.",
        )

    massless_body = RigidBodyPlanV1(
        prim_path="/World/Massless",
        colliders=(
            ColliderPlanV1(
                prim_path="/World/Massless/Collision",
                provenance=provenance("collider"),
            ),
        ),
        provenance=provenance("massless body"),
    )
    mass_without_axes = RigidBodyPlanV1(
        prim_path="/World/Inertial",
        mass=MassPropertiesV1(
            mass_kg=1.0,
            diagonal_inertia_kg_m2=(1.0, 1.0, 1.0),
            provenance=provenance("mass"),
        ),
        provenance=provenance("inertial body"),
    )
    spherical = JointPlanV1(
        topology=JointTopologyV1(
            joint_id="ball",
            joint_type="spherical",
            body0="/World/Massless",
            body1="/World/Inertial",
            field_provenance={
                field: provenance(f"spherical {field}")
                for field in ("joint_type", "body0", "body1")
            },
        )
    )
    driven = JointPlanV1(
        topology=JointTopologyV1(
            joint_id="hinge",
            joint_type="revolute",
            body0="/World/Massless",
            body1="/World/Inertial",
            axis_stage=(0.0, 0.0, 1.0),
            field_provenance={
                field: provenance(f"hinge {field}")
                for field in ("joint_type", "body0", "body1", "axis_stage")
            },
        ),
        joint_friction=JointFrictionV1(
            coefficient=0.25,
            provenance=provenance("joint friction"),
        ),
        drive=JointDriveV1(
            drive_type="force",
            stiffness=1.0,
            damping=0.5,
            max_force=10.0,
            target_position=0.0,
            target_velocity=0.0,
            provenance=provenance("drive"),
        ),
    )
    request = _request(source).model_copy(
        update={
            "plan": JointRiggerPlanV1(
                schema_version=PLAN_SCHEMA_VERSION,
                joints=(spherical, driven),
                rigid_bodies=(massless_body, mass_without_axes),
            )
        }
    )

    _, top_level_allowed = facade._planned_top_level_decisions(request)
    assert top_level_allowed == {
        "rigid_bodies[/World/Inertial].mass.center_of_mass_m": frozenset({"ignored"}),
        "rigid_bodies[/World/Inertial].mass.principal_axes": frozenset({"ignored"}),
        "rigid_bodies[/World/Massless].mass": frozenset({"ignored"}),
        "rigid_bodies[/World/Massless].colliders"
        "[/World/Massless/Collision].mesh_collision_api": frozenset({"ignored"}),
        "rigid_bodies[/World/Massless].colliders"
        "[/World/Massless/Collision].mesh_approximation": frozenset({"ignored"}),
    }

    _, spherical_allowed, _, _ = facade._planned_joint_decisions(spherical)
    assert spherical_allowed["topology.axis_stage"] == frozenset({"ignored"})
    assert spherical_allowed["joint_friction.coefficient"] == frozenset({"ignored"})
    driven_expected, driven_allowed, _, _ = facade._planned_joint_decisions(driven)
    assert driven.joint_friction is not None
    assert (
        driven_expected["joint_friction.coefficient"]
        == driven.joint_friction.provenance
    )
    assert driven_allowed["drive.max_joint_velocity"] == frozenset({"ignored"})


def test_sealed_report_typed_size_error_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"{}"
    snapshot = facade._create_private_report_snapshot(
        tmp_path / "result.json",
        payload,
        label="result",
    )
    monkeypatch.setattr(facade, "_MAX_REPORT_BYTES", 1)
    try:
        with pytest.raises(
            JointRiggerArtifactError,
            match="Private result exceeds the 1-byte limit",
        ):
            facade._load_sealed_model_report(
                snapshot,
                JointRiggerResultV1,
                "result",
            )
    finally:
        snapshot.cleanup()


def test_sealed_report_invalid_json_is_rejected(tmp_path: Path) -> None:
    payload = b"{invalid-json"
    snapshot = facade._create_private_report_snapshot(
        tmp_path / "result.json",
        payload,
        label="result",
    )
    try:
        with pytest.raises(
            JointRiggerArtifactError,
            match="Private result contains invalid report JSON",
        ):
            facade._load_sealed_model_report(
                snapshot,
                JointRiggerResultV1,
                "result",
            )
    finally:
        snapshot.cleanup()


def test_stable_file_copy_keeps_primary_error_and_all_cleanup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"stable-copy payload")
    real_open = facade.os.open
    real_close = facade.os.close
    real_stat = facade.os.stat
    source_descriptor: int | None = None
    source_stat_calls = 0
    close_failure_injected = False
    displaced = tmp_path / "displaced-owned-copy.bin"

    def track_source_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal source_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == source and dir_fd is None:
            source_descriptor = descriptor
        return descriptor

    def fail_postcopy_source_stat(
        path: str | os.PathLike[str],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal source_stat_calls
        if Path(path) == source and dir_fd is None:
            source_stat_calls += 1
            if source_stat_calls == 2:
                destination.rename(displaced)
                destination.write_bytes(b"foreign replacement")
                raise RuntimeError("forced post-copy source failure")
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    def close_source_then_fail(descriptor: int) -> None:
        nonlocal close_failure_injected
        real_close(descriptor)
        if descriptor == source_descriptor and not close_failure_injected:
            close_failure_injected = True
            raise OSError(errno.EIO, "forced stable-copy close failure")

    monkeypatch.setattr(facade.os, "open", track_source_open)
    monkeypatch.setattr(facade.os, "stat", fail_postcopy_source_stat)
    monkeypatch.setattr(facade.os, "close", close_source_then_fail)
    with pytest.raises(RuntimeError, match="forced post-copy source failure") as raised:
        facade._copy_stable_regular_file(source, destination)

    assert close_failure_injected
    notes = "\n".join(raised.value.__notes__)
    assert "replacement preserved" in notes
    assert "forced stable-copy close failure" in notes
    assert destination.read_bytes() == b"foreign replacement"
    assert displaced.read_bytes() == b"stable-copy payload"


def test_stable_file_copy_reports_success_path_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    payload = b"stable-copy payload"
    source.write_bytes(payload)
    real_open = facade.os.open
    real_close = facade.os.close
    source_descriptor: int | None = None
    close_failure_injected = False

    def track_source_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal source_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == source and dir_fd is None:
            source_descriptor = descriptor
        return descriptor

    def close_source_then_fail(descriptor: int) -> None:
        nonlocal close_failure_injected
        real_close(descriptor)
        if descriptor == source_descriptor and not close_failure_injected:
            close_failure_injected = True
            raise OSError(errno.EIO, "forced successful stable-copy close failure")

    monkeypatch.setattr(facade.os, "open", track_source_open)
    monkeypatch.setattr(facade.os, "close", close_source_then_fail)

    with pytest.raises(OSError, match="forced successful stable-copy close failure"):
        facade._copy_stable_regular_file(source, destination)

    assert close_failure_injected
    assert destination.read_bytes() == payload
