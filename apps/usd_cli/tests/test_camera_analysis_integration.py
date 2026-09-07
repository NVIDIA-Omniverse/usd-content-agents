# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport, policy, and atomicity coverage for camera-analysis commands."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pxr import Gf, Usd, UsdGeom
from typer.testing import CliRunner

from usd_cli import client, main
from usd_core.config import Config
from usd_core.history import Op
from usd_core.models import Response
from usd_core.session import Session
from usd_server import app as server_app


def _memory_session(tmp_path: Path) -> Session:
    session = Session(Config(project_dir=tmp_path))
    session._stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(session._stage, "/World")
    session._stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Cube.Define(session._stage, "/World/Floor")
    session._index_prims()
    return session


def _add_export_rig(session: Session, *, cameras: int = 2) -> None:
    UsdGeom.Xform.Define(session._stage, "/World/Rig")
    for index in range(cameras):
        UsdGeom.Camera.Define(session._stage, f"/World/Rig/Camera_{index + 1:03d}")
    session._index_prims()


def _fake_staged_verification(output_dir: str | Path, artifact_base: str | Path):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "camera_001.png"
    payload = b"immutable-camera-evidence"
    artifact.write_bytes(payload)
    relative = artifact.resolve().relative_to(Path(artifact_base).resolve()).as_posix()
    record = {
        "artifact_path_base": "rig_document_directory",
        "request_digest": "sha256:" + "1" * 64,
        "artifacts": [
            {
                "camera": "/World/Rig/Camera_001",
                "relative_path": relative,
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        ],
        "evidence_digest": "sha256:" + "2" * 64,
    }
    return record, [str(artifact)]


def _patch_verified_export_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    from usd_core import render
    from usd_core.camera_analysis import newton_backend
    from usd_core.camera_analysis.contracts import RayHits

    class FakeNewton:
        max_camera_rays = 4_194_304
        versions = {"newton": "test", "warp": "test"}

        def __init__(self, scene, *_args, **_kwargs) -> None:
            self._floor_shape_id = next(
                shape.shape_id
                for shape in scene.shapes
                if shape.prim_path == "/World/Floor"
            )

        def evaluate_rays(self, origins, _directions, **_kwargs) -> RayHits:
            count = len(origins)
            return RayHits(
                distances_m=np.ones(count, dtype=np.float64),
                shape_ids=np.full(count, self._floor_shape_id, dtype=np.int32),
            )

    monkeypatch.setattr(newton_backend, "NewtonVisibilityBackend", FakeNewton)
    monkeypatch.setattr(
        newton_backend,
        "backend_versions",
        lambda: {"newton": "1.5.test", "warp": "1.16.test", "qualified": True},
    )
    monkeypatch.setattr(render, "make_backend", lambda *_args, **_kwargs: object())


def _patch_camera_place_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from usd_core.camera_analysis import authoring, newton_backend, placement, scene

    report = {
        "schema": "usd-cli.camera-placement.v1",
        "backend": {"qualified": True},
        "cameras": [{}],
        "passed": True,
        "selected_count": 1,
        "stop_reason": "target_reached",
        "achieved_coverage": 1.0,
    }
    evaluation = SimpleNamespace(poses=(object(),), report=report)
    monkeypatch.setattr(
        scene, "build_scene_analysis_ir", lambda _stage, **_kwargs: object()
    )
    monkeypatch.setattr(
        scene,
        "analysis_bounds",
        lambda _scene, **_kwargs: (np.zeros(3), np.ones(3)),
    )
    monkeypatch.setattr(
        newton_backend, "NewtonVisibilityBackend", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        newton_backend,
        "backend_versions",
        lambda: {
            "newton": "1.5.test",
            "warp": "1.16.test",
            "qualified": True,
        },
    )
    monkeypatch.setattr(
        placement, "place_max_coverage", lambda *_args, **_kwargs: evaluation
    )

    def author_rig(stage, _scene, poses, _report, *, author_under, on_existing):
        del on_existing
        UsdGeom.Xform.Define(stage, author_under)
        camera_paths = []
        for index, _pose in enumerate(poses, 1):
            path = f"{author_under}/Camera_{index:03d}"
            UsdGeom.Camera.Define(stage, path)
            camera_paths.append(path)
        return {"rig_path": author_under, "camera_paths": camera_paths}

    monkeypatch.setattr(authoring, "author_camera_rig", author_rig)


def _patch_minimal_verified_rig_document(monkeypatch: pytest.MonkeyPatch) -> None:
    from usd_core.camera_analysis import rig_export

    monkeypatch.setattr(
        rig_export,
        "build_rig_document",
        lambda *_args, verification, **_kwargs: {
            "schema": "usd-cli.camera-rig.v1",
            "cameras": [{}],
            "verification": verification,
        },
    )


def test_rig_verification_labels_only_measured_floor_shapes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence

    session = _memory_session(tmp_path)
    shelf = UsdGeom.Cube.Define(session._stage, "/World/Shelf")
    shelf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 2.0))
    shelf.AddScaleOp().Set(Gf.Vec3f(0.4, 0.4, 0.4))
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    semantic_calls: list[dict[str, str]] = []

    def capture_semantics(*_args, semantic_roles, **_kwargs):
        semantic_calls.append(dict(semantic_roles))
        raise RuntimeError("captured semantic roles")

    monkeypatch.setattr(evidence, "verify_rig_with_ovrtx", capture_semantics)

    measured = session.camera_rig_export(
        "/World/Rig",
        res=[8, 8],
        include_visibility=True,
        verify=True,
        scope="/World",
        grid=4,
        output=str(tmp_path / "measured.json"),
    )
    unmeasured = session.camera_rig_export(
        "/World/Rig",
        res=[8, 8],
        include_visibility=False,
        verify=True,
        scope="/World",
        output=str(tmp_path / "unmeasured.json"),
    )

    assert not measured.ok and not unmeasured.ok
    assert "captured semantic roles" in measured.issues[0].message
    assert "captured semantic roles" in unmeasured.issues[0].message
    assert semantic_calls == [{"/World/Floor": "floor"}, {}]


def test_camera_publication_lock_is_shared_by_canonical_destination(tmp_path):
    import threading

    canonical = tmp_path / "rig.json"
    alias = tmp_path / "reports" / ".." / "rig.json"
    first = Session._camera_publish_lock(str(canonical))
    second = Session._camera_publish_lock(str(alias))
    assert first is second

    attempted = threading.Event()
    entered = threading.Event()

    def contend() -> None:
        attempted.set()
        with second:
            entered.set()

    with first:
        thread = threading.Thread(target=contend)
        thread.start()
        assert attempted.wait(timeout=1.0)
        assert not entered.wait(timeout=0.05)
    assert entered.wait(timeout=1.0)
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_verified_rig_failure_discards_unique_staging_and_preserves_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence

    session = _memory_session(tmp_path)
    _add_export_rig(session)
    _patch_verified_export_runtime(monkeypatch)
    output = tmp_path / "rig.json"
    old_json = b'{"old":true}\n'
    output.write_bytes(old_json)
    evidence_root = tmp_path / "rig_evidence"
    evidence_root.mkdir()
    sentinel = evidence_root / "old-generation.txt"
    sentinel.write_text("still referenced")

    def fail_on_second_camera(*_args, output_dir, **_kwargs):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "camera_001.png").write_bytes(b"first camera only")
        raise RuntimeError("second camera render failed")

    monkeypatch.setattr(evidence, "verify_rig_with_ovrtx", fail_on_second_camera)
    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not response.ok
    assert "second camera" in response.issues[0].message
    assert output.read_bytes() == old_json
    assert sentinel.read_text() == "still referenced"
    assert sorted(path.name for path in evidence_root.iterdir()) == [sentinel.name]
    assert not list(tmp_path.glob(".rig_evidence_staging.*"))


def test_verified_rig_partial_generation_copy_is_descriptor_cleaned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    _patch_minimal_verified_rig_document(monkeypatch)
    output = tmp_path / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    monkeypatch.setattr(
        evidence,
        "verify_rig_with_ovrtx",
        lambda *_args, output_dir, artifact_base_dir, **_kwargs: (
            _fake_staged_verification(output_dir, artifact_base_dir)
        ),
    )

    def fail_after_partial_copy(*, destination_dir_fd, **_kwargs):
        descriptor = os.open(
            "partial.bin",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=destination_dir_fd,
        )
        try:
            os.write(descriptor, b"partial unpublished evidence")
        finally:
            os.close(descriptor)
        raise OSError("injected partial generation copy failure")

    monkeypatch.setattr(
        rig_export, "copy_verification_staging", fail_after_partial_copy
    )

    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not response.ok
    assert "partial generation copy" in response.issues[0].message
    assert output.read_bytes() == old_json
    evidence_root = tmp_path / "rig_evidence"
    assert evidence_root.is_dir()
    assert list(evidence_root.iterdir()) == []


def test_verification_copy_closes_first_duplicate_when_second_dup_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "artifact.bin").write_bytes(b"artifact")
    source_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
    destination_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    original_dup = rig_export.os.dup
    duplicated: list[int] = []

    def fail_second_dup(descriptor: int) -> int:
        if duplicated:
            raise OSError(24, "injected descriptor exhaustion")
        result = original_dup(descriptor)
        duplicated.append(result)
        return result

    monkeypatch.setattr(rig_export.os, "dup", fail_second_dup)
    try:
        with pytest.raises(ValueError, match="could not be copied"):
            rig_export.copy_verification_staging(
                source_dir_fd=source_fd,
                destination_dir_fd=destination_fd,
                artifact_relative_paths=["artifact.bin"],
            )
    finally:
        os.close(destination_fd)
        os.close(source_fd)

    assert len(duplicated) == 1
    with pytest.raises(OSError):
        os.fstat(duplicated[0])


def test_verification_copy_nested_directory_is_private_under_restrictive_umask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "nested").mkdir(parents=True)
    destination.mkdir()
    (source / "nested" / "artifact.bin").write_bytes(b"artifact")
    source_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
    destination_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    original_open = rig_export.os.open

    def fail_artifact_copy(path, flags, *args, **kwargs):
        if path == "artifact.bin" and flags & os.O_WRONLY:
            raise OSError("injected artifact copy failure")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(rig_export.os, "open", fail_artifact_copy)
    previous_umask = os.umask(0o777)
    try:
        with pytest.raises(ValueError, match="could not be copied"):
            rig_export.copy_verification_staging(
                source_dir_fd=source_fd,
                destination_dir_fd=destination_fd,
                artifact_relative_paths=["nested/artifact.bin"],
            )
    finally:
        os.umask(previous_umask)

    try:
        assert stat.S_IMODE((destination / "nested").stat().st_mode) == 0o700
        rig_export.PreparedVerificationStaging._clear_directory(destination_fd)
        assert list(destination.iterdir()) == []
    finally:
        os.close(destination_fd)
        os.close(source_fd)


def test_verification_copy_cleans_nested_dir_on_first_caller_stat_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "nested").mkdir(parents=True)
    destination.mkdir()
    (source / "nested" / "artifact.bin").write_bytes(b"artifact")
    source_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
    destination_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    original_create = rig_export.create_owned_private_directory
    original_stat = rig_export.os.stat
    helper_returned = False
    injected = False

    def record_nested_directory(parent_dir_fd: int, name: str):
        nonlocal helper_returned
        result = original_create(parent_dir_fd, name)
        helper_returned = True
        return result

    def fail_first_caller_stat(path, *args, **kwargs):
        nonlocal injected
        if helper_returned and path == "nested" and not injected:
            injected = True
            raise OSError("injected nested caller stat failure")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(
        rig_export,
        "create_owned_private_directory",
        record_nested_directory,
    )
    monkeypatch.setattr(rig_export.os, "stat", fail_first_caller_stat)
    try:
        with pytest.raises(ValueError, match="could not be copied"):
            rig_export.copy_verification_staging(
                source_dir_fd=source_fd,
                destination_dir_fd=destination_fd,
                artifact_relative_paths=["nested/artifact.bin"],
            )
    finally:
        os.close(destination_fd)
        os.close(source_fd)

    assert injected
    assert list(destination.iterdir()) == []


def test_unpublished_generation_cleanup_preserves_substituted_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    _patch_minimal_verified_rig_document(monkeypatch)
    output = tmp_path / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    monkeypatch.setattr(
        evidence,
        "verify_rig_with_ovrtx",
        lambda *_args, output_dir, artifact_base_dir, **_kwargs: (
            _fake_staged_verification(output_dir, artifact_base_dir)
        ),
    )
    moved_generation = tmp_path / "moved-unpublished-generation"
    replacement_victims: list[Path] = []

    def swap_unpublished_name(*, destination_dir_fd, **_kwargs):
        evidence_root = tmp_path / "rig_evidence"
        unpublished = next(
            path for path in evidence_root.iterdir() if path.name.startswith(".")
        )
        unpublished.rename(moved_generation)
        unpublished.mkdir()
        victim = unpublished / "VICTIM"
        victim.write_text("replacement survives\n")
        replacement_victims.append(victim)
        descriptor = os.open(
            "owned-partial.bin",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=destination_dir_fd,
        )
        os.close(descriptor)
        raise OSError("injected unpublished generation substitution")

    monkeypatch.setattr(
        rig_export, "copy_verification_staging", swap_unpublished_name
    )

    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not response.ok
    assert output.read_bytes() == old_json
    assert replacement_victims
    assert replacement_victims[0].read_text() == "replacement survives\n"
    assert list(moved_generation.iterdir()) == []


def test_verified_rig_generation_promotion_never_replaces_concurrent_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    _patch_minimal_verified_rig_document(monkeypatch)
    output = tmp_path / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    monkeypatch.setattr(
        evidence,
        "verify_rig_with_ovrtx",
        lambda *_args, output_dir, artifact_base_dir, **_kwargs: (
            _fake_staged_verification(output_dir, artifact_base_dir)
        ),
    )

    original_rename = rig_export.rename_noreplace
    concurrent_generation: list[Path] = []

    def insert_final_generation(
        source, destination, *, source_dir_fd, destination_dir_fd
    ):
        if source.startswith(".") and destination.startswith("request-"):
            os.mkdir(destination, mode=0o700, dir_fd=destination_dir_fd)
            concurrent_generation.append(
                tmp_path / "rig_evidence" / destination
            )
        return original_rename(
            source,
            destination,
            source_dir_fd=source_dir_fd,
            destination_dir_fd=destination_dir_fd,
        )

    monkeypatch.setattr(rig_export, "rename_noreplace", insert_final_generation)

    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not response.ok
    assert output.read_bytes() == old_json
    assert len(concurrent_generation) == 1
    assert concurrent_generation[0].is_dir()
    assert list(concurrent_generation[0].iterdir()) == []
    assert not [
        path
        for path in concurrent_generation[0].parent.iterdir()
        if path.name.startswith(".")
    ]


def test_verified_rig_preserves_generation_when_promotion_commits_then_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    _patch_minimal_verified_rig_document(monkeypatch)
    output = tmp_path / "rig.json"
    monkeypatch.setattr(
        evidence,
        "verify_rig_with_ovrtx",
        lambda *_args, output_dir, artifact_base_dir, **_kwargs: (
            _fake_staged_verification(output_dir, artifact_base_dir)
        ),
    )

    def publish(document, destination, **_kwargs):
        destination = Path(destination)
        destination.write_text(json.dumps(document))
        return str(destination), "sha256:" + "3" * 64

    monkeypatch.setattr(rig_export, "publish_rig_document", publish)
    original_rename = rig_export.rename_noreplace
    promotions = 0

    def promote_then_raise(
        source, destination, *, source_dir_fd, destination_dir_fd
    ):
        nonlocal promotions
        promotions += 1
        original_rename(
            source,
            destination,
            source_dir_fd=source_dir_fd,
            destination_dir_fd=destination_dir_fd,
        )
        raise OSError("injected indeterminate promotion result")

    monkeypatch.setattr(rig_export, "rename_noreplace", promote_then_raise)

    first = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert first.ok, first.issues
    first_document = json.loads(output.read_text())
    artifact = tmp_path / first_document["verification"]["artifacts"][0][
        "relative_path"
    ]
    generation = artifact.parent
    assert artifact.read_bytes() == b"immutable-camera-evidence"
    assert sorted(path.name for path in generation.iterdir()) == ["camera_001.png"]

    second = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert second.ok, second.issues
    assert promotions == 1
    assert artifact.read_bytes() == b"immutable-camera-evidence"
    assert [path for path in generation.parent.iterdir() if path.name.startswith(".")] == []


@pytest.mark.parametrize("mutation", ["extra", "generation_symlink"])
def test_verified_rig_revalidates_bound_generation_inside_json_publisher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    _patch_minimal_verified_rig_document(monkeypatch)
    output = tmp_path / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    monkeypatch.setattr(
        evidence,
        "verify_rig_with_ovrtx",
        lambda *_args, output_dir, artifact_base_dir, **_kwargs: (
            _fake_staged_verification(output_dir, artifact_base_dir)
        ),
    )
    moved_generations: list[Path] = []

    def mutate_after_session_validation(document, destination, **kwargs):
        relative = document["verification"]["artifacts"][0]["relative_path"]
        root_name, generation_name, *_ = relative.split("/")
        generation = tmp_path / root_name / generation_name
        if mutation == "extra":
            (generation / "UNBOUND-EXTRA").write_text("unbound\n")
        else:
            moved = tmp_path / f"moved-{generation_name}"
            generation.rename(moved)
            generation.symlink_to(moved, target_is_directory=True)
            moved_generations.append(moved)
        kwargs["verification_generation"].validate(document["verification"])
        pytest.fail("mutated bound generation must fail before JSON publication")

    monkeypatch.setattr(
        rig_export, "publish_rig_document", mutate_after_session_validation
    )

    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not response.ok
    assert output.read_bytes() == old_json
    if mutation == "extra":
        assert "inventory differs" in response.issues[0].message
    else:
        assert "generation changed" in response.issues[0].message
        assert moved_generations and (
            moved_generations[0] / "camera_001.png"
        ).is_file()


def test_verified_rig_runs_bound_generation_guard_at_json_precommit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export
    from usd_core.camera_analysis.scene import build_scene_analysis_ir

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    scene = build_scene_analysis_ir(session._stage)
    document = rig_export.build_rig_document(
        session._stage,
        scene,
        "/World/Rig",
        resolution=(16, 12),
        verification=None,
    )
    verification = {"status": "verified-for-precommit-test"}
    document["verification"] = verification
    monkeypatch.setattr(
        rig_export,
        "_verification_record",
        lambda supplied, **_kwargs: supplied,
    )
    output = tmp_path / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    mutated = False

    class RejectingBinding:
        def validate(self, supplied) -> None:
            nonlocal mutated
            assert supplied is verification
            private_directories = list(tmp_path.glob(".rig.json.publish.*.tmp"))
            assert len(private_directories) == 1
            documents = list(private_directories[0].glob("document-*.json"))
            assert len(documents) == 1
            mutated = True
            raise ValueError("bound generation changed at JSON precommit")

    with pytest.raises(ValueError, match="changed at JSON precommit"):
        rig_export.publish_rig_document(
            document,
            output,
            verification_generation=RejectingBinding(),
        )

    assert mutated
    assert output.read_bytes() == old_json
    assert list(tmp_path.glob(".rig.json.publish.*.tmp")) == []


def test_verified_rig_cleanup_never_deletes_a_substituted_staging_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    staging_paths: list[tuple[Path, Path, Path]] = []

    def substitute_staging_then_fail(*_args, output_dir, **_kwargs):
        staging = Path(output_dir)
        (staging / "camera_001.png").write_bytes(b"owned staging artifact")
        moved = staging.with_name(f"{staging.name}.moved")
        staging.rename(moved)
        staging.mkdir()
        victim = staging / "VICTIM"
        victim.write_text("must survive cleanup\n")
        staging_paths.append((staging, moved, victim))
        raise RuntimeError("verification failed after staging substitution")

    monkeypatch.setattr(
        evidence, "verify_rig_with_ovrtx", substitute_staging_then_fail
    )
    response = session.camera_rig_export(
        "/World/Rig",
        res=[16, 12],
        verify=True,
        output=str(tmp_path / "rig.json"),
    )

    assert not response.ok
    assert "staging substitution" in response.issues[0].message
    assert len(staging_paths) == 1
    staging, moved, victim = staging_paths[0]
    assert victim.read_text() == "must survive cleanup\n"
    assert list(moved.iterdir()) == []
    assert not (tmp_path / "rig.json").exists()

    victim.unlink()
    staging.rmdir()
    moved.rmdir()


def test_verified_rig_cancellation_cleans_private_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence
    from usd_core.camera_analysis.cancellation import CameraAnalysisCancelled

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    staging_paths: list[Path] = []

    def cancel_verification(*_args, output_dir, **_kwargs):
        staging = Path(output_dir)
        staging_paths.append(staging)
        (staging / "camera_001.png").write_bytes(b"partial evidence")
        raise CameraAnalysisCancelled("camera analysis cancelled")

    monkeypatch.setattr(evidence, "verify_rig_with_ovrtx", cancel_verification)

    with pytest.raises(CameraAnalysisCancelled, match="camera analysis cancelled"):
        session.camera_rig_export(
            "/World/Rig",
            res=[16, 12],
            verify=True,
            output=str(tmp_path / "rig.json"),
        )

    assert staging_paths and all(not path.exists() for path in staging_paths)
    assert not (tmp_path / "rig.json").exists()


def test_verified_rig_publish_failure_preserves_json_and_keeps_safe_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    output = tmp_path / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    evidence_root = tmp_path / "rig_evidence"
    evidence_root.mkdir()
    sentinel = evidence_root / "old-generation.txt"
    sentinel.write_text("still referenced")

    monkeypatch.setattr(
        evidence,
        "verify_rig_with_ovrtx",
        lambda *_args, output_dir, artifact_base_dir, **_kwargs: (
            _fake_staged_verification(output_dir, artifact_base_dir)
        ),
    )
    monkeypatch.setattr(
        rig_export,
        "build_rig_document",
        lambda *_args, verification, **_kwargs: {
            "schema": "usd-cli.camera-rig.v1",
            "cameras": [{}],
            "verification": verification,
        },
    )

    def fail_before_atomic_swap(_document, _destination, **_kwargs):
        raise RuntimeError("publish failed before atomic JSON swap")

    monkeypatch.setattr(
        rig_export, "publish_rig_document", fail_before_atomic_swap
    )
    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not response.ok
    assert output.read_bytes() == old_json
    assert sentinel.read_text() == "still referenced"
    generations = [path for path in evidence_root.iterdir() if path.is_dir()]
    assert len(generations) == 1
    assert (generations[0] / "camera_001.png").read_bytes() == (
        b"immutable-camera-evidence"
    )
    assert not list(tmp_path.glob(".rig_evidence_staging.*"))


def test_verified_rig_never_renames_private_staging_across_filesystems(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    output = tmp_path / "rig.json"
    private_staging: list[Path] = []

    def fake_verify(*_args, output_dir, artifact_base_dir, **_kwargs):
        private_staging.append(Path(output_dir))
        return _fake_staged_verification(output_dir, artifact_base_dir)

    monkeypatch.setattr(evidence, "verify_rig_with_ovrtx", fake_verify)
    monkeypatch.setattr(
        rig_export,
        "build_rig_document",
        lambda *_args, verification, **_kwargs: {
            "schema": "usd-cli.camera-rig.v1",
            "cameras": [{}],
            "verification": verification,
        },
    )

    def fake_publish(document, destination, **_kwargs):
        destination = Path(destination)
        destination.write_text(json.dumps(document))
        return str(destination), "sha256:" + "3" * 64

    monkeypatch.setattr(rig_export, "publish_rig_document", fake_publish)
    original_rename = rig_export.rename_noreplace
    promotions: list[tuple[object, object, int, int]] = []

    def reject_cross_filesystem_rename(
        source, destination, *, source_dir_fd, destination_dir_fd
    ):
        promotions.append((source, destination, source_dir_fd, destination_dir_fd))
        return original_rename(
            source,
            destination,
            source_dir_fd=source_dir_fd,
            destination_dir_fd=destination_dir_fd,
        )

    monkeypatch.setattr(rig_export, "rename_noreplace", reject_cross_filesystem_rename)

    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert response.ok, response.issues
    assert private_staging and all(not path.exists() for path in private_staging)
    assert len(promotions) == 1
    assert promotions[0][2] == promotions[0][3]
    document = json.loads(output.read_text())
    artifact = tmp_path / document["verification"]["artifacts"][0]["relative_path"]
    assert artifact.read_bytes() == b"immutable-camera-evidence"


def test_verified_rig_copies_only_declared_artifacts_and_rejects_extra_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    output = tmp_path / "rig.json"

    def verification_with_extra(*_args, output_dir, artifact_base_dir, **_kwargs):
        result = _fake_staged_verification(output_dir, artifact_base_dir)
        (Path(output_dir) / "UNBOUND-EXTRA").write_text("must not publish\n")
        return result

    monkeypatch.setattr(
        evidence, "verify_rig_with_ovrtx", verification_with_extra
    )
    monkeypatch.setattr(
        rig_export,
        "build_rig_document",
        lambda *_args, verification, **_kwargs: {
            "schema": "usd-cli.camera-rig.v1",
            "cameras": [{}],
            "verification": verification,
        },
    )

    def fake_publish(document, destination, **_kwargs):
        destination = Path(destination)
        destination.write_text(json.dumps(document))
        return str(destination), "sha256:" + "3" * 64

    monkeypatch.setattr(rig_export, "publish_rig_document", fake_publish)

    first = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert first.ok, first.issues
    document = json.loads(output.read_text())
    artifact = tmp_path / document["verification"]["artifacts"][0]["relative_path"]
    generation = artifact.parent
    assert artifact.is_file()
    assert not (generation / "UNBOUND-EXTRA").exists()

    (generation / "UNBOUND-EXTRA").write_text("unbound generation mutation\n")
    before = output.read_bytes()
    second = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not second.ok
    assert "inventory differs from declared artifacts" in second.issues[0].message
    assert output.read_bytes() == before


def test_verified_rig_rejects_preexisting_shared_writable_evidence_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    _patch_minimal_verified_rig_document(monkeypatch)
    output = tmp_path / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    evidence_root = tmp_path / "rig_evidence"
    evidence_root.mkdir()
    evidence_root.chmod(0o777)
    monkeypatch.setattr(
        evidence,
        "verify_rig_with_ovrtx",
        lambda *_args, output_dir, artifact_base_dir, **_kwargs: (
            _fake_staged_verification(output_dir, artifact_base_dir)
        ),
    )

    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not response.ok
    assert "writable by group/other" in response.issues[0].message
    assert output.read_bytes() == old_json
    assert stat.S_IMODE(evidence_root.stat().st_mode) == 0o777


@pytest.mark.parametrize("writable_entry", ["generation", "artifact"])
def test_verified_rig_rejects_shared_writable_reused_generation_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    writable_entry: str,
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    _patch_minimal_verified_rig_document(monkeypatch)
    monkeypatch.setattr(
        evidence,
        "verify_rig_with_ovrtx",
        lambda *_args, output_dir, artifact_base_dir, **_kwargs: (
            _fake_staged_verification(output_dir, artifact_base_dir)
        ),
    )
    publications: list[dict] = []

    def publish(document, destination, **_kwargs):
        publications.append(document)
        return str(destination), "sha256:" + "a" * 64

    monkeypatch.setattr(rig_export, "publish_rig_document", publish)
    output = tmp_path / "rig.json"
    first = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )
    assert first.ok, first.issues
    evidence_root = tmp_path / "rig_evidence"
    generations = list(evidence_root.iterdir())
    assert len(generations) == 1
    generation = generations[0]
    artifact = generation / "camera_001.png"
    if writable_entry == "generation":
        generation.chmod(0o777)
    else:
        artifact.chmod(0o666)

    second = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not second.ok
    assert len(publications) == 1
    mutated = generation if writable_entry == "generation" else artifact
    assert stat.S_IMODE(mutated.stat().st_mode) & 0o022


def test_verified_rig_rollback_does_not_delete_a_swapped_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    output = tmp_path / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    monkeypatch.setattr(
        evidence,
        "verify_rig_with_ovrtx",
        lambda *_args, output_dir, artifact_base_dir, **_kwargs: (
            _fake_staged_verification(output_dir, artifact_base_dir)
        ),
    )
    monkeypatch.setattr(
        rig_export,
        "build_rig_document",
        lambda *_args, verification, **_kwargs: {
            "schema": "usd-cli.camera-rig.v1",
            "cameras": [{}],
            "verification": verification,
        },
    )
    moved_generation = tmp_path / "moved-generation"
    replacement_generation: list[Path] = []

    def swap_generation_then_fail(document, _destination, **_kwargs):
        relative = document["verification"]["artifacts"][0]["relative_path"]
        root_name, generation_name, *_ = relative.split("/")
        generation = tmp_path / root_name / generation_name
        generation.rename(moved_generation)
        generation.mkdir()
        (generation / "VICTIM").write_text("must survive rollback")
        replacement_generation.append(generation)
        raise RuntimeError("publish failed after generation path swap")

    monkeypatch.setattr(
        rig_export, "publish_rig_document", swap_generation_then_fail
    )

    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not response.ok
    assert "generation path swap" in response.issues[0].message
    assert output.read_bytes() == old_json
    assert (replacement_generation[0] / "VICTIM").read_text() == (
        "must survive rollback"
    )
    assert (moved_generation / "camera_001.png").is_file()


def test_verified_rig_rollback_does_not_delete_a_swapped_evidence_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    output = tmp_path / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    monkeypatch.setattr(
        evidence,
        "verify_rig_with_ovrtx",
        lambda *_args, output_dir, artifact_base_dir, **_kwargs: (
            _fake_staged_verification(output_dir, artifact_base_dir)
        ),
    )
    monkeypatch.setattr(
        rig_export,
        "build_rig_document",
        lambda *_args, verification, **_kwargs: {
            "schema": "usd-cli.camera-rig.v1",
            "cameras": [{}],
            "verification": verification,
        },
    )
    moved_root = tmp_path / "moved-evidence-root"
    replacement_root = tmp_path / "rig_evidence"

    def swap_root_then_fail(_document, _destination, **_kwargs):
        replacement_root.rename(moved_root)
        replacement_root.mkdir()
        (replacement_root / "VICTIM").write_text("must survive rollback")
        raise RuntimeError("publish failed after evidence-root path swap")

    monkeypatch.setattr(rig_export, "publish_rig_document", swap_root_then_fail)

    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not response.ok
    assert "evidence-root path swap" in response.issues[0].message
    assert output.read_bytes() == old_json
    assert (replacement_root / "VICTIM").read_text() == "must survive rollback"
    generations = [path for path in moved_root.iterdir() if path.is_dir()]
    assert len(generations) == 1
    assert (generations[0] / "camera_001.png").is_file()


def test_verified_rig_deduplicates_symlink_aliased_publication_locks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    _patch_minimal_verified_rig_document(monkeypatch)
    output = tmp_path / "rig.json"
    evidence_root = tmp_path / "rig_evidence"
    evidence_root.symlink_to(output.name)
    monkeypatch.setattr(
        evidence,
        "verify_rig_with_ovrtx",
        lambda *_args, output_dir, artifact_base_dir, **_kwargs: (
            _fake_staged_verification(output_dir, artifact_base_dir)
        ),
    )
    assert session._camera_publish_lock(output) is session._camera_publish_lock(
        evidence_root
    )
    result: list[Response] = []

    def export() -> None:
        result.append(
            session.camera_rig_export(
                "/World/Rig", res=[16, 12], verify=True, output=str(output)
            )
        )

    worker = threading.Thread(target=export, daemon=True)
    worker.start()
    worker.join(timeout=2.0)

    assert not worker.is_alive(), "aliased JSON/evidence locks must not self-deadlock"
    assert len(result) == 1
    assert not result[0].ok
    assert "evidence root must be a real directory" in result[0].issues[0].message
    assert evidence_root.is_symlink()
    assert not output.exists()


def test_verified_rig_rejects_evidence_root_swap_during_generation_promotion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    output = tmp_path / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    monkeypatch.setattr(
        evidence,
        "verify_rig_with_ovrtx",
        lambda *_args, output_dir, artifact_base_dir, **_kwargs: (
            _fake_staged_verification(output_dir, artifact_base_dir)
        ),
    )
    monkeypatch.setattr(
        rig_export,
        "build_rig_document",
        lambda *_args, verification, **_kwargs: {
            "schema": "usd-cli.camera-rig.v1",
            "cameras": [{}],
            "verification": verification,
        },
    )
    evidence_root = tmp_path / "rig_evidence"
    moved_root = tmp_path / "moved-evidence-root"
    original_rename = rig_export.rename_noreplace
    swapped = False

    def swap_root_during_promotion(
        src, dst, *, source_dir_fd, destination_dir_fd
    ):
        nonlocal swapped
        if not swapped:
            swapped = True
            evidence_root.rename(moved_root)
            evidence_root.mkdir()
            (evidence_root / "VICTIM").write_text("must survive rejection")
        return original_rename(
            src,
            dst,
            source_dir_fd=source_dir_fd,
            destination_dir_fd=destination_dir_fd,
        )

    publication_calls: list[object] = []
    monkeypatch.setattr(rig_export, "rename_noreplace", swap_root_during_promotion)
    monkeypatch.setattr(
        rig_export,
        "publish_rig_document",
        lambda *_args, **_kwargs: publication_calls.append(object()),
    )

    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not response.ok
    assert "evidence root changed during validation" in response.issues[0].message
    assert output.read_bytes() == old_json
    assert publication_calls == []
    assert (evidence_root / "VICTIM").read_text() == "must survive rejection"
    generations = [path for path in moved_root.iterdir() if path.is_dir()]
    assert len(generations) == 1
    assert (generations[0] / "camera_001.png").is_file()


def test_verified_rig_rejects_symlink_evidence_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    output = tmp_path / "rig.json"
    output.write_text("old")
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "rig_evidence").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        evidence,
        "verify_rig_with_ovrtx",
        lambda *_args, output_dir, artifact_base_dir, **_kwargs: (
            _fake_staged_verification(output_dir, artifact_base_dir)
        ),
    )
    monkeypatch.setattr(
        rig_export,
        "build_rig_document",
        lambda *_args, verification, **_kwargs: {
            "schema": "usd-cli.camera-rig.v1",
            "cameras": [{}],
            "verification": verification,
        },
    )
    monkeypatch.setattr(
        rig_export,
        "publish_rig_document",
        lambda *_args, **_kwargs: pytest.fail("unsafe bundle must not publish"),
    )

    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not response.ok
    assert "real directory" in response.issues[0].message
    assert output.read_text() == "old"
    assert list(outside.iterdir()) == []
    assert not list(tmp_path.glob(".rig_evidence_staging.*"))


def test_verified_rig_parent_swap_never_writes_json_or_evidence_outside(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    allowed_parent = tmp_path / "allowed"
    moved_parent = tmp_path / "validated-parent"
    outside = tmp_path / "outside"
    allowed_parent.mkdir()
    outside.mkdir()
    output = allowed_parent / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    (outside / "SENTINEL").write_text("outside must remain untouched\n")
    private_staging: list[Path] = []

    def verify_then_swap_parent(
        *_args, output_dir, artifact_base_dir, **_kwargs
    ):
        result = _fake_staged_verification(output_dir, artifact_base_dir)
        private_staging.append(Path(output_dir))
        allowed_parent.rename(moved_parent)
        allowed_parent.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(evidence, "verify_rig_with_ovrtx", verify_then_swap_parent)
    monkeypatch.setattr(
        rig_export,
        "build_rig_document",
        lambda *_args, verification, **_kwargs: {
            "schema": "usd-cli.camera-rig.v1",
            "cameras": [{}],
            "verification": verification,
        },
    )
    monkeypatch.setattr(
        rig_export,
        "publish_rig_document",
        lambda *_args, **_kwargs: pytest.fail("swapped parent must not publish"),
    )

    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not response.ok
    assert "directory changed after validation" in response.issues[0].message
    assert private_staging and all(not path.exists() for path in private_staging)
    assert (moved_parent / "rig.json").read_bytes() == old_json
    assert sorted(path.name for path in outside.iterdir()) == ["SENTINEL"]
    assert not (outside / "rig.json").exists()
    assert not (outside / "rig_evidence").exists()


def test_verified_rig_rejects_symlinked_content_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    output = tmp_path / "rig.json"
    output.write_text("old")
    monkeypatch.setattr(
        evidence,
        "verify_rig_with_ovrtx",
        lambda *_args, output_dir, artifact_base_dir, **_kwargs: (
            _fake_staged_verification(output_dir, artifact_base_dir)
        ),
    )
    original_prepare = rig_export.prepare_verification_generation
    generation_links: list[Path] = []

    def prepare_with_alias(*args, staging_dir, **kwargs):
        record, generation_dir, paths = original_prepare(
            *args, staging_dir=staging_dir, **kwargs
        )
        alias_target = tmp_path / "aliased-generation"
        alias_target.mkdir()
        staged_artifact = Path(staging_dir) / "camera_001.png"
        (alias_target / staged_artifact.name).write_bytes(staged_artifact.read_bytes())
        generation_dir.parent.mkdir()
        generation_dir.symlink_to(alias_target, target_is_directory=True)
        generation_links.append(generation_dir)
        return record, generation_dir, paths

    monkeypatch.setattr(
        rig_export, "prepare_verification_generation", prepare_with_alias
    )
    monkeypatch.setattr(
        rig_export,
        "build_rig_document",
        lambda *_args, verification, **_kwargs: {
            "schema": "usd-cli.camera-rig.v1",
            "cameras": [{}],
            "verification": verification,
        },
    )
    monkeypatch.setattr(
        rig_export,
        "publish_rig_document",
        lambda *_args, **_kwargs: pytest.fail("symlinked generation must not publish"),
    )

    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], verify=True, output=str(output)
    )

    assert not response.ok
    assert "real directory" in response.issues[0].message
    assert output.read_text() == "old"
    assert generation_links and generation_links[0].is_symlink()
    assert not list(tmp_path.glob(".rig_evidence_staging.*"))


def test_same_stem_verified_exports_serialize_shared_evidence_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export

    first_session = _memory_session(tmp_path)
    second_session = _memory_session(tmp_path)
    _add_export_rig(first_session, cameras=1)
    _add_export_rig(second_session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    first_output = tmp_path / "rig.json"
    second_output = tmp_path / "rig.txt"
    first_at_publish = threading.Event()
    release_first = threading.Event()
    second_entered_verify = threading.Event()

    def fake_verify(*_args, output_dir, artifact_base_dir, **_kwargs):
        if threading.current_thread().name == "second-export":
            second_entered_verify.set()
        return _fake_staged_verification(output_dir, artifact_base_dir)

    monkeypatch.setattr(evidence, "verify_rig_with_ovrtx", fake_verify)
    monkeypatch.setattr(
        rig_export,
        "build_rig_document",
        lambda *_args, verification, **_kwargs: {
            "schema": "usd-cli.camera-rig.v1",
            "cameras": [{}],
            "verification": verification,
        },
    )

    def publish(document, destination, **_kwargs):
        destination = Path(destination)
        if threading.current_thread().name == "first-export":
            first_at_publish.set()
            assert release_first.wait(timeout=5.0)
            raise RuntimeError("first publish failed before atomic swap")
        destination.write_text(json.dumps(document))
        return str(destination), "sha256:" + "3" * 64

    monkeypatch.setattr(rig_export, "publish_rig_document", publish)
    responses: dict[str, Response] = {}

    def run(name: str, session: Session, destination: Path) -> None:
        responses[name] = session.camera_rig_export(
            "/World/Rig", res=[16, 12], verify=True, output=str(destination)
        )

    first = threading.Thread(
        target=run,
        args=("first", first_session, first_output),
        name="first-export",
    )
    second = threading.Thread(
        target=run,
        args=("second", second_session, second_output),
        name="second-export",
    )
    first.start()
    assert first_at_publish.wait(timeout=5.0)
    second.start()
    assert not second_entered_verify.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert not first.is_alive() and not second.is_alive()
    assert not responses["first"].ok
    assert responses["second"].ok
    assert not first_output.exists()
    published = json.loads(second_output.read_text())
    for artifact in published["verification"]["artifacts"]:
        assert (tmp_path / artifact["relative_path"]).is_file()
    assert second_entered_verify.is_set()
    assert not list(tmp_path.glob(".rig_evidence_staging.*"))


def test_dynamic_mutation_classification_and_read_only_fast_fail() -> None:
    reader = SimpleNamespace(read_only=True, name="reader")

    assert not server_app._request_is_mutating("camera.coverage", {})
    assert not server_app._request_is_mutating(
        "camera.place", {"method": "max_coverage", "preview": True}
    )
    assert server_app._request_is_mutating(
        "camera.place", {"method": "max_coverage", "author_under": "/World/Rig"}
    )
    assert server_app._request_is_mutating("camera.fit", {})

    assert server_app._read_only_reject(reader, "camera.coverage", {}) is None
    assert server_app._read_only_reject(reader, "camera.rig-export", {}) is None
    assert (
        server_app._read_only_reject(
            reader, "camera.place", {"method": "max_coverage", "preview": True}
        )
        is None
    )
    rejected = server_app._read_only_reject(
        reader,
        "camera.place",
        {"method": "max_coverage", "author_under": "/World/Rig"},
    )
    assert rejected is not None
    assert not rejected.ok
    assert "read-only" in rejected.issues[0].message


def test_http_read_only_preview_runs_but_author_is_rejected_before_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSession:
        calls: list[dict] = []

        def __init__(self, _config, name: str = "default") -> None:
            self.name = name
            self.read_only = True
            self._stage_path = None

        def camera_place(
            self,
            method: str,
            preview: bool = False,
            author_under: str | None = None,
        ) -> Response:
            type(self).calls.append(
                {
                    "method": method,
                    "preview": preview,
                    "author_under": author_under,
                }
            )
            return Response(command="camera.place", summary={"preview": True})

    monkeypatch.setattr(server_app, "Session", FakeSession)
    app = server_app.build_app(
        Config(project_dir=tmp_path), token="token", instance_id="camera-test"
    )
    headers = {"x-usd-cli-token": "token"}
    with TestClient(app) as http:
        preview = http.post(
            "/cmd",
            json={
                "command": "camera.place",
                "payload": {
                    "method": "max_coverage",
                    "preview": True,
                    "detach": True,
                },
            },
            headers=headers,
        )
        completed = http.post(
            "/cmd",
            json={
                "command": "wait",
                "payload": {
                    "job": preview.json()["summary"]["job"],
                    "timeout": 30,
                },
            },
            headers=headers,
        )
        authored = http.post(
            "/cmd",
            json={
                "command": "camera.place",
                "payload": {
                    "method": "max_coverage",
                    "author_under": "/World/Rig",
                },
            },
            headers=headers,
        )

    assert preview.status_code == 200
    assert preview.json()["ok"] is True
    assert completed.status_code == 200
    assert completed.json()["ok"] is True
    assert FakeSession.calls == [
        {"method": "max_coverage", "preview": True, "author_under": None}
    ]
    assert authored.status_code == 200
    assert authored.json()["ok"] is False
    assert "read-only" in authored.json()["issues"][0]["message"]


@pytest.mark.parametrize(
    "command",
    ["camera.coverage", "camera.place", "camera.rig-export"],
)
def test_camera_report_outputs_obey_write_root_policy(
    tmp_path: Path, command: str
) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    cfg = Config(
        project_dir=project,
        server={
            "allowed_roots": [str(project), str(outside)],
            "allowed_write_roots": [str(project)],
        },
    )

    server_app._validate_request_policy(
        cfg,
        command,
        {"output": str(project / "report.json")},
        shared=False,
    )
    with pytest.raises(ValueError, match="allowed_write_roots"):
        server_app._validate_request_policy(
            cfg,
            command,
            {"output": str(outside / "report.json")},
            shared=False,
        )


def test_implicit_rig_export_destination_is_checked_against_write_roots(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    write_root = tmp_path / "explicit-writes-only"
    project.mkdir()
    write_root.mkdir()
    cfg = Config(
        project_dir=project,
        server={
            "allowed_roots": [str(project)],
            "allowed_write_roots": [str(write_root)],
        },
    )

    with pytest.raises(ValueError, match="allowed_write_roots"):
        server_app._validate_request_policy(
            cfg,
            "camera.rig-export",
            {"rig": "/World/Rig"},
            shared=False,
        )


@pytest.mark.parametrize(
    "command,payload,message",
    [
        ("camera.coverage", {"grid": 1}, "grid"),
        ("camera.rig-export", {"grid": 513}, "grid"),
        ("camera.coverage", {"cameras": [f"/C{i}" for i in range(65)]}, "64"),
        ("camera.place", {"candidates": 3}, "candidates"),
        ("camera.place", {"candidates": 513}, "candidates"),
        ("camera.place", {"max_cameras": 0}, "max-cameras"),
        ("camera.place", {"cameras": 65}, "cameras"),
        ("camera.rig-export", {"res": [4097, 1]}, "resolution"),
        (
            "camera.rig-export",
            {"verify": True, "res": [4096, 4096]},
            "4,194,304",
        ),
    ],
)
def test_camera_requests_are_resource_bounded(
    tmp_path: Path, command: str, payload: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        server_app._validate_request_policy(
            Config(project_dir=tmp_path),
            command,
            payload,
            shared=False,
        )


def test_daemon_defers_aspect_sensitive_ray_budget_to_analysis_engine(
    tmp_path: Path,
) -> None:
    from usd_core.camera_analysis.coverage import validate_grid_workload

    session = _memory_session(tmp_path)
    UsdGeom.Camera.Define(session._stage, "/World/Camera")

    # Admission has no live stage and therefore cannot know how many cameras
    # implicit selection will find.  A 200x200 grid is valid for this one-camera
    # stage and must reach the session-aware validation pass.
    server_app._validate_request_policy(
        session.config,
        "camera.coverage",
        {"grid": 200},
        shared=False,
    )
    server_app._validate_request_policy(
        session.config,
        "camera.coverage",
        {"grid": 200},
        shared=False,
        session=session,
    )

    # A square estimate would reject each request even though a narrow scope can
    # have far fewer cells. The command engine owns the exact bounds-aware budget.
    for command, payload in (
        (
            "camera.coverage",
            {"grid": 256, "cameras": [f"/Camera_{index}" for index in range(32)]},
        ),
        ("camera.place", {"grid": 100, "candidates": 201}),
        (
            "camera.rig-export",
            {"rig": "/World/Rig", "include_visibility": True, "grid": 177},
        ),
    ):
        server_app._validate_request_policy(
            session.config,
            command,
            payload,
            shared=False,
        )

    narrow_bounds = (
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([10.0, 1.0, 1.0]),
    )
    assert validate_grid_workload(
        narrow_bounds,
        grid=256,
        cell_size_m=None,
        view_count=32,
        operation="coverage evaluation",
    ) == (26, 256)
    with pytest.raises(ValueError, match="2,000,000"):
        validate_grid_workload(
            (
                np.asarray([0.0, 0.0, 0.0]),
                np.asarray([1.0, 1.0, 1.0]),
            ),
            grid=300,
            cell_size_m=None,
            view_count=32,
            operation="coverage evaluation",
        )


def test_rig_camera_budget_counts_derived_camera_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf, Usd

    from usd_core.camera_analysis import scene as scene_module

    class DerivedCameraPrim:
        def IsA(self, schema) -> bool:
            return schema is UsdGeom.Camera

        def GetTypeName(self) -> str:
            return "DerivedCamera"

        def GetPath(self):
            return Sdf.Path("/World/Rig/DerivedCamera")

    root = SimpleNamespace(IsValid=lambda: True)
    session = SimpleNamespace(
        _path_of=lambda value: value,
        _stage=SimpleNamespace(GetPrimAtPath=lambda _path: root),
    )
    monkeypatch.setattr(Usd, "PrimRange", lambda *_args: [DerivedCameraPrim()])
    monkeypatch.setattr(
        scene_module, "imageable_analysis_policy", lambda _prim: (True, "default")
    )

    assert server_app._effective_camera_count(session, "/World/Rig") == 1


@pytest.mark.parametrize("command", ["camera.coverage", "camera.rig-export"])
def test_implicit_stage_camera_selection_cannot_bypass_camera_cap(
    tmp_path: Path, command: str
) -> None:
    session = _memory_session(tmp_path)
    UsdGeom.Xform.Define(session._stage, "/World/Rig")
    for index in range(65):
        UsdGeom.Camera.Define(session._stage, f"/World/Rig/Camera_{index:03d}")
    payload = {} if command == "camera.coverage" else {"rig": "/World/Rig"}

    with pytest.raises(ValueError, match="64"):
        server_app._validate_request_policy(
            session.config,
            command,
            payload,
            shared=False,
            session=session,
        )


def test_implicit_coverage_cap_ignores_unselected_orthographic_cameras(
    tmp_path: Path,
) -> None:
    session = _memory_session(tmp_path)
    UsdGeom.Camera.Define(session._stage, "/World/Perspective")
    for index in range(64):
        camera = UsdGeom.Camera.Define(session._stage, f"/World/Ortho_{index:03d}")
        camera.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
    hidden = UsdGeom.Xform.Define(session._stage, "/World/Hidden")
    hidden.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    for index in range(32):
        UsdGeom.Camera.Define(session._stage, f"/World/Hidden/Camera_{index:03d}")
    for index in range(32):
        guide = UsdGeom.Camera.Define(session._stage, f"/World/Guide_{index:03d}")
        guide.CreatePurposeAttr(UsdGeom.Tokens.guide)

    server_app._validate_request_policy(
        session.config,
        "camera.coverage",
        {},
        shared=False,
        session=session,
    )

    for index in range(64):
        UsdGeom.Camera.Define(session._stage, f"/World/Perspective_{index:03d}")
    with pytest.raises(ValueError, match="64"):
        server_app._validate_request_policy(
            session.config,
            "camera.coverage",
            {},
            shared=False,
            session=session,
        )


@pytest.mark.parametrize("command", ["coverage", "place", "rig-export"])
def test_unqualified_newton_warp_fails_before_analysis_or_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str
) -> None:
    from usd_core.camera_analysis import newton_backend, scene as scene_module

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    output_parent = tmp_path / "must-not-be-created"
    output = output_parent / f"{command}.json"
    stage_before = session._stage.GetRootLayer().ExportToString()
    history_before = session.history.snapshot_state()
    epoch_before = session._mutation_epoch

    monkeypatch.setattr(
        newton_backend,
        "backend_versions",
        lambda: {"newton": "1.4.0", "warp": "1.16.0", "qualified": False},
    )
    monkeypatch.setattr(
        scene_module,
        "build_scene_analysis_ir",
        lambda *_args, **_kwargs: pytest.fail(
            "scene analysis must not start with an unqualified backend"
        ),
    )

    if command == "coverage":
        response = session.camera_coverage(output=str(output))
    elif command == "place":
        response = session.camera_place(
            "max_coverage",
            author_under="/World/NewRig",
            output=str(output),
        )
    else:
        response = session.camera_rig_export(
            "/World/Rig",
            include_visibility=True,
            scope="/World",
            output=str(output),
        )

    assert response.ok is False
    assert "requires qualified Newton 1.5.x and Warp 1.16.x" in response.issues[0].message
    assert session._stage.GetRootLayer().ExportToString() == stage_before
    assert session.history.snapshot_state() == history_before
    assert session._mutation_epoch == epoch_before
    assert not output.exists()
    assert not output_parent.exists()
    assert not session._stage.GetPrimAtPath("/World/NewRig").IsValid()


def test_rig_verification_budget_fails_before_output_parent_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    output_parent = tmp_path / "must-not-be-created"

    response = session.camera_rig_export(
        "/World/Rig",
        res=[2049, 2048],
        verify=True,
        output=str(output_parent / "rig.json"),
    )

    assert not response.ok
    assert "rays per camera" in response.issues[0].message
    assert not output_parent.exists()


def test_missing_visibility_scope_fails_before_output_parent_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    _patch_verified_export_runtime(monkeypatch)
    output_parent = tmp_path / "must-not-be-created"

    response = session.camera_rig_export(
        "/World/Rig",
        res=[64, 48],
        include_visibility=True,
        output=str(output_parent / "rig.json"),
    )

    assert not response.ok
    assert "requires --scope" in response.issues[0].message
    assert not output_parent.exists()


def test_calibration_only_rig_export_ignores_unrelated_unsupported_geometry(
    tmp_path: Path,
) -> None:
    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    UsdGeom.PointInstancer.Define(session._stage, "/World/UnrelatedCrowd")
    output = tmp_path / "calibration-only.json"

    response = session.camera_rig_export(
        "/World/Rig",
        res=[64, 48],
        include_visibility=False,
        verify=False,
        output=str(output),
    )

    assert response.ok, response.issues
    document = json.loads(output.read_text())
    assert len(document["cameras"]) == 1
    assert document.get("coverage") is None
    assert document["verification"]["status"] == "not_requested"


def test_direct_session_relative_rig_output_keeps_process_cwd_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    process_cwd = tmp_path / "daemon-cwd"
    project.mkdir()
    process_cwd.mkdir()
    session = _memory_session(project)
    _add_export_rig(session, cameras=1)
    monkeypatch.chdir(process_cwd)

    response = session.camera_rig_export(
        "/World/Rig",
        res=[64, 48],
        output="reports/rig.json",
    )

    expected = process_cwd / "reports" / "rig.json"
    assert response.ok, response.issues
    assert response.summary["output"] == str(expected)
    assert expected.is_file()
    assert not (project / "reports" / "rig.json").exists()


def test_direct_cli_relative_rig_output_is_resolved_from_process_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process_cwd = tmp_path / "cli-cwd"
    process_cwd.mkdir()
    dispatched: list[tuple[str, dict]] = []

    def dispatch(command: str, payload: dict) -> Response:
        dispatched.append((command, payload))
        return Response(command=command)

    monkeypatch.chdir(process_cwd)
    monkeypatch.setattr(main, "dispatch", dispatch)
    result = CliRunner().invoke(
        main.app,
        [
            "camera",
            "rig-export",
            "/World/Rig",
            "--output",
            "reports/rig.json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(dispatched) == 1
    assert dispatched[0][0] == "camera.rig-export"
    assert dispatched[0][1]["output"] == str(
        process_cwd / "reports" / "rig.json"
    )


def test_http_relative_and_implicit_rig_outputs_use_configured_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    daemon_cwd = tmp_path / "daemon-cwd"
    project.mkdir()
    daemon_cwd.mkdir()

    created_sessions: list[Session] = []

    def session_factory(config: Config, name: str = "default") -> Session:
        session = Session(config, name=name)
        session._stage = Usd.Stage.CreateInMemory()
        world = UsdGeom.Xform.Define(session._stage, "/World")
        session._stage.SetDefaultPrim(world.GetPrim())
        _add_export_rig(session, cameras=1)
        created_sessions.append(session)
        return session

    monkeypatch.chdir(daemon_cwd)
    monkeypatch.setattr(server_app, "Session", session_factory)
    app = server_app.build_app(
        Config(project_dir=project),
        token="token",
        instance_id="camera-output-test",
    )
    headers = {"x-usd-cli-token": "token"}
    with TestClient(app) as http:
        relative = http.post(
            "/cmd",
            json={
                "command": "camera.rig-export",
                "payload": {
                    "rig": "/World/Rig",
                    "output": "reports/rig.json",
                },
            },
            headers=headers,
        )
        implicit = http.post(
            "/cmd",
            json={
                "command": "camera.rig-export",
                "payload": {"rig": "/World/Rig"},
            },
            headers=headers,
        )

    assert relative.status_code == 200 and relative.json()["ok"] is True
    assert implicit.status_code == 200 and implicit.json()["ok"] is True
    assert relative.json()["summary"]["output"] == str(
        project / "reports" / "rig.json"
    )
    assert implicit.json()["summary"]["output"] == str(project / "rig.json")
    assert (project / "reports" / "rig.json").is_file()
    assert (project / "rig.json").is_file()
    assert not (daemon_cwd / "reports" / "rig.json").exists()
    assert not (daemon_cwd / "rig.json").exists()
    assert created_sessions[0].config.server["allowed_write_roots"] == [str(project)]


def test_http_admitted_project_root_swap_cannot_publish_outside(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    moved_project = tmp_path / "validated-project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (outside / "SENTINEL").write_text("outside must remain untouched\n")

    class SwapBeforePublicationSession(Session):
        swapped = False

        def __init__(self, config: Config, name: str = "default") -> None:
            super().__init__(config, name=name)
            self._stage = Usd.Stage.CreateInMemory()
            world = UsdGeom.Xform.Define(self._stage, "/World")
            self._stage.SetDefaultPrim(world.GetPrim())
            _add_export_rig(self, cameras=1)

        def camera_rig_export(
            self, rig: str, output: str = "rig.json"
        ) -> Response:
            if not type(self).swapped:
                type(self).swapped = True
                project.rename(moved_project)
                project.symlink_to(outside, target_is_directory=True)
            return super().camera_rig_export(rig, output=output)

    monkeypatch.setattr(server_app, "Session", SwapBeforePublicationSession)
    app = server_app.build_app(
        Config(project_dir=project),
        token="token",
        instance_id="camera-root-swap-test",
    )
    with TestClient(app) as http:
        response = http.post(
            "/cmd",
            json={
                "command": "camera.rig-export",
                "payload": {
                    "rig": "/World/Rig",
                    "output": "reports/nested/rig.json",
                },
            },
            headers={"x-usd-cli-token": "token"},
        )

    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert SwapBeforePublicationSession.swapped, response.json().get("issues")
    assert sorted(path.name for path in outside.iterdir()) == ["SENTINEL"]
    assert not (outside / "rig.json").exists()
    assert not (outside / "reports").exists()
    assert not (moved_project / "rig.json").exists()
    assert not (moved_project / "reports").exists()


def test_json_publication_replace_is_bound_to_validated_parent_descriptor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    allowed_parent = tmp_path / "allowed"
    moved_parent = tmp_path / "validated-parent"
    outside = tmp_path / "outside"
    allowed_parent.mkdir()
    outside.mkdir()
    publication = rig_export.prepare_json_publication(allowed_parent / "rig.json")
    original_replace = rig_export.os.replace
    replaced = False

    def swap_parent_then_replace(source, destination, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            allowed_parent.rename(moved_parent)
            allowed_parent.symlink_to(outside, target_is_directory=True)
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(rig_export.os, "replace", swap_parent_then_replace)
    try:
        published_path, _digest = rig_export.publish_json_document(
            {"schema": "usd-cli.test-publication.v1", "value": 1},
            publication,
        )
    finally:
        publication.close()

    assert replaced
    assert published_path == str(allowed_parent / "rig.json")
    assert not (outside / "rig.json").exists()
    assert json.loads((moved_parent / "rig.json").read_text()) == {
        "schema": "usd-cli.test-publication.v1",
        "value": 1,
    }


def test_json_publication_private_temp_namespace_survives_top_level_substitution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    output_parent = tmp_path / "group-writable"
    output_parent.mkdir(mode=0o770)
    output = output_parent / "rig.json"
    publication = rig_export.prepare_json_publication(output)
    original_replace = rig_export.os.replace
    moved_private = output_parent / "held-private"
    substituted_private: Path | None = None
    substituted_document_name: str | None = None
    intercepted = False

    def substitute_private_name_then_replace(source, destination, *args, **kwargs):
        nonlocal intercepted, substituted_document_name, substituted_private
        if destination == output.name and not intercepted:
            intercepted = True
            source_fd = kwargs["src_dir_fd"]
            destination_fd = kwargs["dst_dir_fd"]
            assert source.startswith("document-") and source.endswith(".json")
            assert source_fd != destination_fd
            assert stat.S_IMODE(os.fstat(source_fd).st_mode) == 0o700
            private_path = Path(os.readlink(f"/proc/self/fd/{source_fd}"))
            private_path.rename(moved_private)
            private_path.mkdir(mode=0o700)
            substituted_private = private_path
            substituted_document_name = source
            (private_path / source).write_text("attacker bytes\n")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(rig_export.os, "replace", substitute_private_name_then_replace)
    try:
        rig_export.publish_json_document(
            {"schema": "usd-cli.test-publication.v1", "value": 7},
            publication,
        )
    finally:
        publication.close()

    assert intercepted
    assert json.loads(output.read_text()) == {
        "schema": "usd-cli.test-publication.v1",
        "value": 7,
    }
    assert substituted_private is not None
    assert substituted_document_name is not None
    assert (substituted_private / substituted_document_name).read_text() == (
        "attacker bytes\n"
    )
    assert list(moved_private.iterdir()) == []


def test_json_publication_rejects_private_directory_replacement_during_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    output_parent = tmp_path / "group-writable"
    output_parent.mkdir(mode=0o770)
    output = output_parent / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    original_open = rig_export.os.open
    moved_private = output_parent / "moved-private"
    replacement_private: list[Path] = []

    def substitute_before_path_open(path, flags, *args, **kwargs):
        if (
            isinstance(path, str)
            and path.startswith(".rig.json.publish.")
            and flags & os.O_PATH
            and not replacement_private
        ):
            parent = Path(os.readlink(f"/proc/self/fd/{kwargs['dir_fd']}"))
            created = parent / path
            created.rename(moved_private)
            created.mkdir(mode=0o700)
            created.chmod(0o777)
            (created / "VICTIM").write_text("replacement survives\n")
            replacement_private.append(created)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(rig_export.os, "open", substitute_before_path_open)

    with pytest.raises(ValueError, match="changed during path open"):
        rig_export.publish_json_document(
            {"schema": "usd-cli.test-publication.v1", "value": 9},
            output,
        )

    assert output.read_bytes() == old_json
    assert replacement_private
    assert (replacement_private[0] / "VICTIM").read_text() == (
        "replacement survives\n"
    )
    assert moved_private.is_dir()


def test_json_failure_cleanup_preserves_substituted_private_victim_contents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    output_parent = tmp_path / "group-writable"
    output_parent.mkdir(mode=0o770)
    output = output_parent / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    victim = output_parent / "exporter-owned-victim"
    victim.mkdir(mode=0o700)
    victim.chmod(0o2700)
    (victim / "SENTINEL").write_text("victim survives\n")
    moved_created = output_parent / "moved-created-private"
    replacement_path: Path | None = None
    original_open = rig_export.os.open

    def substitute_same_owner_victim(path, flags, *args, **kwargs):
        nonlocal replacement_path
        if (
            isinstance(path, str)
            and path.startswith(".rig.json.publish.")
            and flags & os.O_PATH
            and replacement_path is None
        ):
            parent = Path(os.readlink(f"/proc/self/fd/{kwargs['dir_fd']}"))
            created = parent / path
            created.rename(moved_created)
            victim.rename(created)
            replacement_path = created
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(rig_export.os, "open", substitute_same_owner_victim)

    def fail_precommit() -> None:
        raise ValueError("injected JSON precommit failure")

    with pytest.raises(ValueError, match="injected JSON precommit failure"):
        rig_export.publish_json_document(
            {"schema": "usd-cli.test-publication.v1", "value": 10},
            output,
            precommit=fail_precommit,
        )

    assert output.read_bytes() == old_json
    assert replacement_path is not None
    assert stat.S_IMODE(replacement_path.stat().st_mode) == 0o2700
    assert sorted(path.name for path in replacement_path.iterdir()) == ["SENTINEL"]
    assert (replacement_path / "SENTINEL").read_text() == "victim survives\n"
    assert moved_created.is_dir()
    assert list(moved_created.iterdir()) == []


def test_private_directory_preserves_special_bits_on_shared_parent_victim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    shared_parent = tmp_path / "shared-parent"
    shared_parent.mkdir()
    shared_parent.chmod(0o770)
    victim = shared_parent / "victim"
    victim.mkdir()
    victim.chmod(0o2700)
    (victim / "KEEP").write_text("metadata and bytes survive\n")
    moved_created = shared_parent / "moved-created"
    parent_fd = os.open(shared_parent, os.O_RDONLY | os.O_DIRECTORY)
    descriptor = -1
    original_open = rig_export.os.open
    substituted = False

    def substitute_special_mode_victim(path, flags, *args, **kwargs):
        nonlocal substituted
        if path == "private" and flags & os.O_PATH and not substituted:
            substituted = True
            (shared_parent / "private").rename(moved_created)
            victim.rename(shared_parent / "private")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(rig_export.os, "open", substitute_special_mode_victim)
    try:
        descriptor, identity = rig_export.create_owned_private_directory(
            parent_fd, "private"
        )
        opened = os.fstat(descriptor)
        assert (opened.st_dev, opened.st_ino) == identity
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)

    substituted_victim = shared_parent / "private"
    assert substituted
    assert stat.S_IMODE(substituted_victim.stat().st_mode) == 0o2700
    assert (substituted_victim / "KEEP").read_text() == (
        "metadata and bytes survive\n"
    )
    assert moved_created.is_dir()


def test_json_publication_preserves_setgid_private_children_without_residue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    setgid_parent = tmp_path / "setgid-parent"
    setgid_parent.mkdir()
    setgid_parent.chmod(0o2770)
    output = setgid_parent / "rig.json"
    original_create = rig_export.create_owned_private_directory
    created_modes: list[int] = []

    def record_private_mode(parent_dir_fd: int, name: str):
        result = original_create(parent_dir_fd, name)
        created_modes.append(stat.S_IMODE(os.fstat(result[0]).st_mode))
        return result

    monkeypatch.setattr(
        rig_export,
        "create_owned_private_directory",
        record_private_mode,
    )
    for value in range(8):
        rig_export.publish_json_document(
            {"schema": "usd-cli.test-publication.v1", "value": value},
            output,
        )

    assert json.loads(output.read_text())["value"] == 7
    assert len(created_modes) == 8
    assert all(mode & 0o777 == 0o700 for mode in created_modes)
    assert all(mode & stat.S_ISGID for mode in created_modes)
    assert sorted(path.name for path in setgid_parent.iterdir()) == ["rig.json"]


def test_private_directory_rejects_restrictive_umask_in_shared_parent(
    tmp_path: Path,
) -> None:
    from usd_core.camera_analysis import rig_export

    shared_parent = tmp_path / "shared-parent"
    shared_parent.mkdir()
    shared_parent.chmod(0o770)
    parent_fd = os.open(shared_parent, os.O_RDONLY | os.O_DIRECTORY)
    previous_umask = os.umask(0o777)
    try:
        with pytest.raises(ValueError, match="ambiguous in a shared parent"):
            rig_export.create_owned_private_directory(parent_fd, "private")
    finally:
        os.umask(previous_umask)
        os.close(parent_fd)

    ambiguous = shared_parent / "private"
    assert ambiguous.is_dir()
    assert stat.S_IMODE(ambiguous.stat().st_mode) == 0o000
    ambiguous.chmod(0o700)
    ambiguous.rmdir()


def test_private_directory_path_open_failure_cleans_created_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    original_open = rig_export.os.open
    injected = False

    def fail_path_open(path, flags, *args, **kwargs):
        nonlocal injected
        if path == "private" and flags & os.O_PATH and not injected:
            injected = True
            raise PermissionError("injected private-directory path open failure")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(rig_export.os, "open", fail_path_open)
    try:
        with pytest.raises(PermissionError, match="injected private-directory"):
            rig_export.create_owned_private_directory(parent_fd, "private")
    finally:
        os.close(parent_fd)

    assert not (tmp_path / "private").exists()


def test_private_directory_first_fstat_failure_cleans_created_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    names = [f"private-{index}" for index in range(8)]
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    original_fstat = rig_export.os.fstat
    fail_next = False
    failures = 0

    def fail_first_fstat(descriptor: int):
        nonlocal fail_next, failures
        if fail_next:
            fail_next = False
            failures += 1
            raise OSError("injected first fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(rig_export.os, "fstat", fail_first_fstat)
    try:
        for name in names:
            fail_next = True
            with pytest.raises(OSError, match="first fstat"):
                rig_export.create_owned_private_directory(parent_fd, name)
    finally:
        os.close(parent_fd)

    assert failures == len(names)
    assert list(tmp_path.iterdir()) == []


def test_private_directory_first_named_stat_failure_cleans_created_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    names = {f"private-{index}" for index in range(8)}
    failed: set[str] = set()
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    original_stat = rig_export.os.stat

    def fail_first_named_stat(path, *args, **kwargs):
        if (
            path in names
            and kwargs.get("dir_fd") == parent_fd
            and path not in failed
        ):
            failed.add(path)
            raise OSError("injected first named-stat failure")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(rig_export.os, "stat", fail_first_named_stat)
    try:
        for name in sorted(names):
            with pytest.raises(OSError, match="first named-stat"):
                rig_export.create_owned_private_directory(parent_fd, name)
    finally:
        os.close(parent_fd)

    assert failed == names
    assert list(tmp_path.iterdir()) == []


def test_json_publication_cleans_private_dir_on_first_caller_fstat_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    output = tmp_path / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    original_create = rig_export.create_owned_private_directory
    original_fstat = rig_export.os.fstat
    returned_descriptor = -1
    injected = False

    def record_private_directory(parent_dir_fd: int, name: str):
        nonlocal returned_descriptor
        result = original_create(parent_dir_fd, name)
        returned_descriptor = result[0]
        return result

    def fail_first_caller_fstat(descriptor: int):
        nonlocal injected
        if descriptor == returned_descriptor and not injected:
            injected = True
            raise OSError("injected caller fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(
        rig_export,
        "create_owned_private_directory",
        record_private_directory,
    )
    monkeypatch.setattr(rig_export.os, "fstat", fail_first_caller_fstat)

    with pytest.raises(OSError, match="caller fstat"):
        rig_export.publish_json_document(
            {"schema": "usd-cli.test-publication.v1", "value": 13},
            output,
        )

    assert injected
    assert output.read_bytes() == old_json
    assert list(tmp_path.glob(".rig.json.publish.*.tmp")) == []


def test_json_publication_entropy_failure_leaks_no_parent_descriptors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    output = tmp_path / "rig.json"
    descriptors_before = len(os.listdir("/proc/self/fd"))

    def fail_entropy(_size: int) -> bytes:
        raise OSError("injected entropy failure")

    monkeypatch.setattr(rig_export.os, "urandom", fail_entropy)
    for _attempt in range(8):
        with pytest.raises(OSError, match="injected entropy"):
            rig_export.publish_json_document(
                {"schema": "usd-cli.test-publication.v1", "value": 13},
                output,
            )

    assert len(os.listdir("/proc/self/fd")) == descriptors_before
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_verification_staging_parent_open_failure_creates_no_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    monkeypatch.setattr(rig_export.tempfile, "tempdir", str(tmp_path))
    original_open = rig_export.os.open
    injected = False

    def fail_temp_root_open(path, flags, *args, **kwargs):
        nonlocal injected
        if Path(path) == tmp_path and flags & os.O_DIRECTORY and not injected:
            injected = True
            raise OSError(24, "injected temp-root descriptor exhaustion")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(rig_export.os, "open", fail_temp_root_open)

    with pytest.raises(OSError, match="temp-root descriptor exhaustion"):
        rig_export.PreparedVerificationStaging("verification-")

    assert injected
    assert list(tmp_path.iterdir()) == []


def test_verification_staging_rejects_shared_nonsticky_temp_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    unsafe_root = tmp_path / "shared-nonsticky"
    unsafe_root.mkdir()
    unsafe_root.chmod(0o777)
    monkeypatch.setattr(rig_export.tempfile, "tempdir", str(unsafe_root))

    with pytest.raises(ValueError, match="trusted private or sticky"):
        rig_export.PreparedVerificationStaging("verification-")

    assert list(unsafe_root.iterdir()) == []


def test_private_publication_and_verification_staging_ignore_restrictive_umask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    monkeypatch.setattr(rig_export.tempfile, "tempdir", str(tmp_path))
    output = tmp_path / "rig.json"
    previous_umask = os.umask(0o777)
    try:
        rig_export.publish_json_document(
            {"schema": "usd-cli.test-publication.v1", "value": 11},
            output,
        )
        staging = rig_export.PreparedVerificationStaging("verification-")
    finally:
        os.umask(previous_umask)

    assert json.loads(output.read_text())["value"] == 11
    assert stat.S_IMODE(os.fstat(staging.directory_descriptor).st_mode) == 0o700
    staging_path = staging.path
    staging.cleanup()
    assert not staging_path.exists()
    assert list(tmp_path.glob(".rig.json.publish.*.tmp")) == []


def test_verified_generation_private_directories_ignore_restrictive_umask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import evidence, rig_export, scene

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    analysis_scene = scene.build_scene_analysis_ir(session._stage)
    monkeypatch.setattr(
        scene,
        "build_scene_analysis_ir",
        lambda *_args, **_kwargs: analysis_scene,
    )
    _patch_verified_export_runtime(monkeypatch)
    _patch_minimal_verified_rig_document(monkeypatch)

    def staged_evidence(*_args, output_dir, artifact_base_dir, **_kwargs):
        record, paths = _fake_staged_verification(output_dir, artifact_base_dir)
        for path in paths:
            Path(path).chmod(0o600)
        return record, paths

    monkeypatch.setattr(evidence, "verify_rig_with_ovrtx", staged_evidence)
    monkeypatch.setattr(
        rig_export,
        "publish_rig_document",
        lambda _document, destination, **_kwargs: (
            str(destination),
            "sha256:" + "a" * 64,
        ),
    )
    output = tmp_path / "rig.json"
    previous_umask = os.umask(0o777)
    try:
        response = session.camera_rig_export(
            "/World/Rig", res=[16, 12], verify=True, output=str(output)
        )
    finally:
        os.umask(previous_umask)

    assert response.ok, response.issues
    evidence_root = tmp_path / "rig_evidence"
    generations = list(evidence_root.iterdir())
    assert len(generations) == 1
    assert stat.S_IMODE(evidence_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(generations[0].stat().st_mode) == 0o700
    assert stat.S_IMODE((generations[0] / "camera_001.png").stat().st_mode) == 0o600


def test_json_publication_treats_delegate_then_raise_replace_as_committed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    output = tmp_path / "rig.json"
    output.write_text('{"generation":"old"}\n')
    original_replace = rig_export.os.replace
    injected = False

    def replace_then_raise(source, destination, *args, **kwargs):
        nonlocal injected
        result = original_replace(source, destination, *args, **kwargs)
        if destination == output.name and not injected:
            injected = True
            raise OSError("injected indeterminate replace result")
        return result

    monkeypatch.setattr(rig_export.os, "replace", replace_then_raise)

    published_path, _digest = rig_export.publish_json_document(
        {"schema": "usd-cli.test-publication.v1", "value": 13},
        output,
    )

    assert injected
    assert published_path == str(output)
    assert json.loads(output.read_text())["value"] == 13
    assert output.stat().st_size > 0
    assert list(tmp_path.glob(".rig.json.publish.*.tmp")) == []


def test_indeterminate_commit_with_external_overwrite_uses_last_writer_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    output = tmp_path / "rig.json"
    output.write_text('{"generation":"old"}\n')
    concurrent_json = b'{"generation":"external-last-writer"}\n'
    original_replace = rig_export.os.replace
    injected = False

    def commit_overwrite_then_raise(source, destination, *args, **kwargs):
        nonlocal injected
        result = original_replace(source, destination, *args, **kwargs)
        if destination == output.name and not injected:
            injected = True
            external = tmp_path / "external.json"
            external.write_bytes(concurrent_json)
            original_replace(external, output)
            raise OSError("injected indeterminate result after external overwrite")
        return result

    monkeypatch.setattr(rig_export.os, "replace", commit_overwrite_then_raise)

    published_path, _digest = rig_export.publish_json_document(
        {"schema": "usd-cli.test-publication.v1", "value": 17},
        output,
    )

    assert injected
    assert published_path == str(output)
    assert output.read_bytes() == concurrent_json
    assert list(tmp_path.glob(".rig.json.publish.*.tmp")) == []


def test_json_precommit_failures_clean_private_payload_staging(tmp_path: Path) -> None:
    from usd_core.camera_analysis import rig_export

    output = tmp_path / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    document = {
        "schema": "usd-cli.test-publication.v1",
        "payload": "x" * (128 * 1024),
    }

    def fail_precommit() -> None:
        raise RuntimeError("injected precommit failure")

    for _ in range(12):
        with pytest.raises(RuntimeError, match="injected precommit failure"):
            rig_export.publish_json_document(
                document,
                output,
                precommit=fail_precommit,
            )

    assert output.read_bytes() == old_json
    assert list(tmp_path.glob(".rig.json.publish.*.tmp")) == []


@pytest.mark.parametrize("destination_type", ["directory", "symlink", "hardlink", "fifo"])
def test_json_publication_rejects_unsafe_destination_before_temp_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    destination_type: str,
) -> None:
    from usd_core.camera_analysis import rig_export

    output = tmp_path / "rig.json"
    if destination_type == "directory":
        output.mkdir()
    elif destination_type == "symlink":
        target = tmp_path / "target.json"
        target.write_text("target survives\n")
        output.symlink_to(target)
    elif destination_type == "hardlink":
        target = tmp_path / "target.json"
        target.write_text("target survives\n")
        os.link(target, output)
    else:
        os.mkfifo(output)
    monkeypatch.setattr(
        rig_export.PreparedJsonPublication,
        "_temporary_name",
        staticmethod(
            lambda *_args: pytest.fail(
                "unsafe destinations must fail before private temp creation"
            )
        ),
    )

    with pytest.raises(ValueError, match="single-link regular file"):
        rig_export.publish_json_document(
            {"schema": "usd-cli.test-publication.v1"},
            output,
        )

    assert output.exists() or output.is_symlink()


def test_rig_export_treats_post_replace_directory_sync_failure_as_committed_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    output = tmp_path / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    original_fsync = rig_export.os.fsync
    injected = False

    def fail_first_directory_sync(descriptor: int) -> None:
        nonlocal injected
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and not injected:
            injected = True
            raise OSError("injected post-replace directory sync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(rig_export.os, "fsync", fail_first_directory_sync)

    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], output=str(output)
    )

    assert response.ok, response.issues
    assert injected
    assert output.read_bytes() != old_json
    assert json.loads(output.read_text())["schema"] == "usd-cli.camera-rig.v1"
    assert list(tmp_path.glob(".rig.json.publish.*.tmp")) == []


def test_rig_export_commit_stays_bound_to_held_parent_after_post_replace_swap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    allowed_parent = tmp_path / "allowed"
    moved_parent = tmp_path / "held-parent"
    outside = tmp_path / "outside"
    allowed_parent.mkdir()
    outside.mkdir()
    output = allowed_parent / "rig.json"
    old_json = b'{"generation":"old"}\n'
    output.write_bytes(old_json)
    (outside / "SENTINEL").write_text("outside survives\n")
    original_fsync = rig_export.os.fsync
    swapped = False

    def swap_parent_after_directory_sync(descriptor: int) -> None:
        nonlocal swapped
        original_fsync(descriptor)
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and not swapped:
            swapped = True
            allowed_parent.rename(moved_parent)
            allowed_parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(rig_export.os, "fsync", swap_parent_after_directory_sync)

    response = session.camera_rig_export(
        "/World/Rig", res=[16, 12], output=str(output)
    )

    assert response.ok, response.issues
    assert swapped
    assert (moved_parent / "rig.json").read_bytes() != old_json
    assert (
        json.loads((moved_parent / "rig.json").read_text())["schema"]
        == "usd-cli.camera-rig.v1"
    )
    assert sorted(path.name for path in outside.iterdir()) == ["SENTINEL"]
    assert list(moved_parent.glob(".rig.json.publish.*.tmp")) == []


def test_json_publication_uses_last_writer_wins_without_hidden_backups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    output = tmp_path / "rig.json"
    old_json = b'{"generation":"old"}\n'
    concurrent_json = b'{"generation":"concurrent"}\n'
    output.write_bytes(old_json)
    publication = rig_export.prepare_json_publication(output)
    original_fsync = rig_export.os.fsync
    substituted = False

    def substitute_after_directory_sync(descriptor: int) -> None:
        nonlocal substituted
        original_fsync(descriptor)
        if descriptor == publication.directory_descriptor and not substituted:
            substituted = True
            replacement = tmp_path / "concurrent.json"
            replacement.write_bytes(concurrent_json)
            os.replace(replacement, output)

    monkeypatch.setattr(rig_export.os, "fsync", substitute_after_directory_sync)
    try:
        published_path, _digest = rig_export.publish_json_document(
            {"schema": "usd-cli.test-publication.v1", "value": 3},
            publication,
        )
    finally:
        publication.close()

    assert substituted
    assert published_path == str(output)
    assert output.read_bytes() == concurrent_json
    assert list(tmp_path.glob(".rig.json.publish.*.tmp")) == []


def test_artifact_hash_rejects_final_name_swap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    artifact = tmp_path / "camera.png"
    artifact.write_bytes(b"bound artifact bytes")
    moved = tmp_path / "moved-camera.png"
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    original_read = rig_export.os.read
    swapped = False

    def swap_name_after_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, size)
        if chunk and not swapped:
            swapped = True
            artifact.rename(moved)
            artifact.symlink_to(moved)
        return chunk

    monkeypatch.setattr(rig_export.os, "read", swap_name_after_read)
    try:
        with pytest.raises(ValueError, match="changed while it was hashed"):
            rig_export._sha256_file_at(
                directory_fd, rig_export.PurePosixPath("camera.png")
            )
    finally:
        os.close(directory_fd)

    assert swapped
    assert artifact.is_symlink()
    assert moved.read_bytes() == b"bound artifact bytes"


def test_generation_restats_earlier_artifacts_after_hashing_the_full_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    generation = tmp_path / "generation"
    generation.mkdir()
    first = generation / "first.bin"
    second = generation / "second.bin"
    first_payload = b"first artifact bytes"
    second_payload = b"second artifact bytes"
    first.write_bytes(first_payload)
    second.write_bytes(second_payload)
    verification = {
        "artifacts": [
            {
                "relative_path": "root/generation/first.bin",
                "sha256": "sha256:" + hashlib.sha256(first_payload).hexdigest(),
                "size_bytes": len(first_payload),
            },
            {
                "relative_path": "root/generation/second.bin",
                "sha256": "sha256:" + hashlib.sha256(second_payload).hexdigest(),
                "size_bytes": len(second_payload),
            },
        ]
    }
    generation_fd = os.open(generation, os.O_RDONLY | os.O_DIRECTORY)
    original_hash = rig_export._sha256_file_at
    mutated = False

    def rewrite_first_while_hashing_second(directory_fd, relative_path):
        nonlocal mutated
        if relative_path == rig_export.PurePosixPath("second.bin") and not mutated:
            first.write_bytes(b"first artifact bytes changed after its hash")
            mutated = True
        return original_hash(directory_fd, relative_path)

    monkeypatch.setattr(
        rig_export, "_sha256_file_at", rewrite_first_while_hashing_second
    )
    try:
        with pytest.raises(ValueError, match="changed after it was hashed"):
            rig_export.validate_verification_generation(
                verification,
                artifact_base_dir=tmp_path,
                generation_dir_fd=generation_fd,
                generation_relative_path="root/generation",
            )
    finally:
        os.close(generation_fd)

    assert mutated


def test_json_publication_preserves_an_unowned_temporary_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    output = tmp_path / "rig.json"
    candidate_name = ".rig.json.publish.preexisting.tmp"
    candidate = tmp_path / candidate_name
    victim = tmp_path / "victim"
    victim.write_text("must survive\n")
    candidate.symlink_to(victim)
    publication = rig_export.prepare_json_publication(output)
    monkeypatch.setattr(
        rig_export.PreparedJsonPublication,
        "_temporary_name",
        staticmethod(lambda _destination, _purpose: candidate_name),
    )
    try:
        with pytest.raises(FileExistsError):
            rig_export.publish_json_document(
                {"schema": "usd-cli.test-publication.v1"},
                publication,
            )
    finally:
        publication.close()

    assert candidate.is_symlink()
    assert victim.read_text() == "must survive\n"


def test_rig_export_and_server_cap_share_composed_camera_policy(tmp_path: Path) -> None:
    session = _memory_session(tmp_path)
    _add_export_rig(session, cameras=1)
    hidden = UsdGeom.Camera.Define(session._stage, "/World/Rig/Hidden")
    hidden.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    guide = UsdGeom.Camera.Define(session._stage, "/World/Rig/Guide")
    guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    session._index_prims()
    output = tmp_path / "eligible-rig.json"

    assert server_app._effective_camera_count(session, "/World/Rig") == 1
    response = session.camera_rig_export(
        "/World/Rig",
        res=[64, 48],
        output=str(output),
    )

    assert response.ok, response.issues
    document = json.loads(output.read_text())
    assert [camera["path"] for camera in document["cameras"]] == [
        "/World/Rig/Camera_001"
    ]

    visible = UsdGeom.Camera(session._stage.GetPrimAtPath("/World/Rig/Camera_001"))
    visible.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    rejected_output = tmp_path / "no-eligible-cameras.json"
    rejected = session.camera_rig_export(
        "/World/Rig",
        res=[64, 48],
        output=str(rejected_output),
    )

    assert not rejected.ok
    assert "no cameras eligible" in rejected.issues[0].message
    assert not rejected_output.exists()


def test_camera_place_resolves_author_ref_and_rejects_instance_proxy(
    tmp_path: Path,
) -> None:
    session = _memory_session(tmp_path)
    session._stage.CreateClassPrim("/_RigPrototype")
    UsdGeom.Xform.Define(session._stage, "/_RigPrototype/Rig")
    instance = session._stage.DefinePrim("/World/Instance", "Xform")
    instance.GetReferences().AddInternalReference("/_RigPrototype")
    instance.SetInstanceable(True)
    session._index_prims()
    proxy_path = "/World/Instance/Rig"
    proxy_ref = session.refs.ref_for_path(proxy_path)
    assert proxy_ref is not None
    assert session._stage.GetPrimAtPath(proxy_path).IsInstanceProxy()

    response = session.camera_place(
        "max_coverage",
        author_under=proxy_ref,
        on_existing="replace",
    )

    assert response.ok is False
    assert "inside a native instance" in response.issues[0].message


def test_rig_export_request_preflights_total_visibility_and_verification_rays(
    tmp_path: Path,
) -> None:
    session = _memory_session(tmp_path)
    UsdGeom.Xform.Define(session._stage, "/World/Rig")
    for index in range(64):
        UsdGeom.Camera.Define(session._stage, f"/World/Rig/Camera_{index:03d}")

    # Grid cell count depends on scope aspect ratio, so transport policy must not
    # reject with a square estimate. Session validates the exact visibility budget.
    server_app._validate_request_policy(
        session.config,
        "camera.rig-export",
        {
            "rig": "/World/Rig",
            "include_visibility": True,
            "grid": 177,
        },
        shared=False,
        session=session,
    )

    with pytest.raises(ValueError, match="67,108,864"):
        server_app._validate_request_policy(
            session.config,
            "camera.rig-export",
            {"rig": "/World/Rig", "verify": True, "res": [2048, 2048]},
            shared=False,
            session=session,
        )


def test_camera_cli_and_server_wiring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, dict]] = []

    def dispatch(command: str, payload: dict) -> Response:
        calls.append((command, payload))
        return Response(command=command)

    monkeypatch.setattr(main, "dispatch", dispatch)
    runner = CliRunner()
    coverage_output = tmp_path / "coverage.json"
    rig_output = tmp_path / "rig.json"

    coverage = runner.invoke(
        main.app,
        [
            "camera",
            "coverage",
            "--scope",
            "/World/Floor",
            "--camera",
            "/World/C1",
            "--camera",
            "@c2",
            "--cameras",
            "/World/C3, @c4",
            "--grid",
            "8",
            "-o",
            str(coverage_output),
            "--detach",
        ],
    )
    placement = runner.invoke(
        main.app,
        [
            "camera",
            "place",
            "--method",
            "LOOK_AT",
            "--target",
            "/World/Target",
            "--cameras",
            "3",
            "--occlusion-threshold",
            "0.25",
            "--min-height",
            "1.0",
            "--max-height",
            "2.0",
            "--min-look-down",
            "15",
            "--max-look-down",
            "50",
            "--min-x",
            "-3",
            "--max-x",
            "3",
            "--min-y",
            "-4",
            "--max-y",
            "4",
            "--preview",
            "--detach",
        ],
    )
    export = runner.invoke(
        main.app,
        [
            "camera",
            "rig-export",
            "/World/Rig",
            "--res",
            "640x480",
            "--include-visibility",
            "-o",
            str(rig_output),
            "--detach",
        ],
    )
    max_placement = runner.invoke(
        main.app,
        [
            "camera",
            "place",
            "--method",
            "max_coverage",
            "--scope",
            "/World/Floor",
            "--patch-size",
            "0.5",
            "--min-look-down",
            "10",
            "--max-look-down",
            "45",
            "--preview",
        ],
    )

    assert (
        coverage.exit_code
        == placement.exit_code
        == export.exit_code
        == max_placement.exit_code
        == 0
    )
    assert calls[0] == (
        "camera.coverage",
        {
            "scope": "/World/Floor",
            "cameras": ["/World/C1", "@c2", "/World/C3", "@c4"],
            "target": 0.95,
            "per_cell": 1,
            "grid": 8,
            "device": "cpu",
            "output": str(coverage_output),
            "detach": True,
        },
    )
    assert calls[1][0] == "camera.place"
    assert calls[1][1] == {
        "method": "look_at",
        "target": "/World/Target",
        "cameras": 3,
        "candidates": 64,
        "min_height": 1.0,
        "max_height": 2.0,
        "min_look_down": 15.0,
        "max_look_down": 50.0,
        "occlusion_threshold": 0.25,
        "min_x": -3.0,
        "max_x": 3.0,
        "min_y": -4.0,
        "max_y": 4.0,
        "seed": 0,
        "aperture": 36.0,
        "device": "cpu",
        "preview": True,
        "on_existing": "error",
        "detach": True,
    }
    assert calls[2] == (
        "camera.rig-export",
        {
            "rig": "/World/Rig",
            "res": [640, 480],
            "include_visibility": True,
            "grid": 32,
            "device": "cpu",
            "output": str(rig_output),
            "detach": True,
        },
    )
    assert calls[3] == (
        "camera.place",
        {
            "method": "max_coverage",
            "scope": "/World/Floor",
            "patch_size": 0.5,
            "candidates": 64,
            "min_look_down": 10.0,
            "max_look_down": 45.0,
            "seed": 0,
            "aperture": 36.0,
            "device": "cpu",
            "preview": True,
            "on_existing": "error",
        },
    )
    assert server_app.COMMANDS["camera.coverage"] == "camera_coverage"
    assert server_app.COMMANDS["camera.place"] == "camera_place"
    assert server_app.COMMANDS["camera.rig-export"] == "camera_rig_export"
    assert {
        "camera.coverage",
        "camera.place",
        "camera.rig-export",
    } <= client._SLOW_COMMANDS


@pytest.mark.parametrize(
    "method,kwargs,expected",
    [
        ("max_coverage", {"cameras": 3}, "look_at"),
        ("look_at", {"target": "/World/Floor", "max_cameras": 3}, "max_coverage"),
        ("max_coverage", {"min_height": 1.0}, "look_at"),
        ("look_at", {"target": "/World/Floor", "patch_size": 0.5}, "max_coverage"),
        ("max_coverage", {"on_existing": "replace"}, "author-under"),
    ],
)
def test_place_rejects_flags_for_the_other_method(
    tmp_path: Path, method: str, kwargs: dict, expected: str
) -> None:
    response = _memory_session(tmp_path).camera_place(method=method, **kwargs)

    assert not response.ok
    assert expected in response.issues[0].message


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"min_look_down": 60.0, "max_look_down": 20.0}, "look-down envelope"),
        ({"min_x": -1.0}, "XY bounds require all"),
        ({"grid": 16, "patch_size": 0.5}, "mutually exclusive"),
    ],
)
def test_place_envelopes_fail_before_scene_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kwargs: dict,
    expected: str,
) -> None:
    from usd_core.camera_analysis import scene

    monkeypatch.setattr(
        scene,
        "build_scene_analysis_ir",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid envelopes must fail before ingest"
        ),
    )
    method = "look_at" if "min_x" in kwargs else "max_coverage"
    output_parent = tmp_path / "invalid-place-output"
    response = _memory_session(tmp_path).camera_place(
        method=method,
        target="/World/Floor" if method == "look_at" else None,
        output=str(output_parent / "report.json"),
        **kwargs,
    )

    assert not response.ok
    assert expected in response.issues[0].message
    assert not output_parent.exists()


def test_place_invalid_author_destination_fails_before_output_parent_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _memory_session(tmp_path)
    UsdGeom.Xform.Define(session._stage, "/World/Rig")
    session._index_prims()
    _patch_camera_place_success(monkeypatch)
    output_parent = tmp_path / "must-not-be-created"

    response = session.camera_place(
        method="max_coverage",
        scope="/World/Floor",
        author_under="/World/Rig",
        output=str(output_parent / "placement.json"),
    )

    assert not response.ok
    assert "already exists" in response.issues[0].message
    assert not output_parent.exists()


def test_camera_place_uses_last_writer_wins_without_hidden_backups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import rig_export

    session = _memory_session(tmp_path)
    _patch_camera_place_success(monkeypatch)
    output = tmp_path / "placement.json"
    old_json = b'{"generation":"old"}\n'
    concurrent_json = b'{"generation":"concurrent"}\n'
    output.write_bytes(old_json)
    epoch_before = session._mutation_epoch
    viewer_before = session._viewer_revision
    original_replace = rig_export.os.replace
    injected = False

    def insert_concurrent_write_before_commit(source, destination, *args, **kwargs):
        nonlocal injected
        if destination == output.name and not injected:
            injected = True
            replacement = tmp_path / "concurrent-placement.json"
            replacement.write_bytes(concurrent_json)
            original_replace(replacement, output)
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(rig_export.os, "replace", insert_concurrent_write_before_commit)
    response = session.camera_place(
        method="max_coverage",
        scope="/World/Floor",
        author_under="/World/Rig",
        output=str(output),
    )

    assert response.ok, response.issues
    assert injected
    assert session._stage.GetPrimAtPath("/World/Rig").IsValid()
    assert len(session.history.entries()) == 1
    assert session._mutation_epoch == epoch_before + 1
    assert session._viewer_revision == viewer_before + 1
    assert json.loads(output.read_text())["schema"] == "usd-cli.camera-placement.v1"
    assert list(tmp_path.glob(".placement.json.publish.*.tmp")) == []


@pytest.mark.parametrize(
    "failure,output_state",
    [
        ("publish", "missing"),
        ("history", None),
        ("history", "missing"),
        ("history", "existing"),
    ],
)
def test_failed_camera_authoring_rolls_back_the_whole_rig(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
    output_state: str | None,
) -> None:
    from usd_core.camera_analysis import (
        authoring,
        newton_backend,
        placement,
        rig_export,
        scene,
    )

    session = _memory_session(tmp_path)
    root_before = session._stage.GetRootLayer().ExportToString()
    session_before = session._stage.GetSessionLayer().ExportToString()
    epoch_before = session._mutation_epoch
    viewer_revision_before = session._viewer_revision
    session._viewer_line_geometry_publications["preserved"] = {"digest": "old"}
    history_before = session.history.snapshot_state()
    output_path = tmp_path / "placement.json"
    old_output = b"previous placement report\n"
    if output_state == "existing":
        output_path.write_bytes(old_output)

    report = {
        "schema": "usd-cli.camera-placement.v1",
        "backend": {"qualified": True},
        "cameras": [{}],
        "passed": True,
        "selected_count": 1,
        "stop_reason": "target_reached",
        "achieved_coverage": 1.0,
    }
    evaluation = SimpleNamespace(poses=(object(),), report=report)
    monkeypatch.setattr(
        scene, "build_scene_analysis_ir", lambda _stage, **_kwargs: object()
    )
    monkeypatch.setattr(
        scene,
        "analysis_bounds",
        lambda _scene, **_kwargs: (np.zeros(3), np.ones(3)),
    )
    monkeypatch.setattr(
        newton_backend, "NewtonVisibilityBackend", lambda *_args, **_kw: object()
    )
    monkeypatch.setattr(
        newton_backend,
        "backend_versions",
        lambda: {
            "newton": "1.5.test",
            "warp": "1.16.test",
            "qualified": True,
        },
    )
    monkeypatch.setattr(
        placement, "place_max_coverage", lambda *_args, **_kw: evaluation
    )

    def author_rig(stage, _scene, poses, _report, *, author_under, on_existing):
        del on_existing
        UsdGeom.Xform.Define(stage, author_under)
        paths = []
        for index, _pose in enumerate(poses, 1):
            path = f"{author_under}/Camera_{index:03d}"
            UsdGeom.Camera.Define(stage, path)
            paths.append(path)
        return {"rig_path": author_under, "camera_paths": paths}

    monkeypatch.setattr(authoring, "author_camera_rig", author_rig)

    if failure == "publish":

        def fail_publish(*_args, **_kwargs):
            raise RuntimeError("injected publish failure")

        monkeypatch.setattr(
            rig_export,
            "publish_json_document",
            fail_publish,
        )
    else:
        original_record = session._record

        def fail_record(*args, **kwargs) -> None:
            original_record(*args, **kwargs)
            raise RuntimeError("injected history failure")

        monkeypatch.setattr(session, "_record", fail_record)
    response = session.camera_place(
        method="max_coverage",
        scope="/World/Floor",
        author_under="/World/Rig",
        output=(
            str(output_path)
            if failure == "publish" or output_state is not None
            else None
        ),
    )

    assert not response.ok
    assert f"injected {failure} failure" in response.issues[0].message
    assert session._stage.GetRootLayer().ExportToString() == root_before
    assert session._stage.GetSessionLayer().ExportToString() == session_before
    assert not session._stage.GetPrimAtPath("/World/Rig").IsValid()
    assert session.history.entries() == []
    assert session.history.snapshot_state() == history_before
    assert session._mutation_epoch == epoch_before
    assert session._viewer_revision == viewer_revision_before
    assert session._viewer_line_geometry_publications == {
        "preserved": {"digest": "old"}
    }
    if output_state == "existing":
        assert output_path.read_bytes() == old_output
    elif output_state == "missing":
        assert not output_path.exists()
    assert not list(tmp_path.glob(".placement.json.rollback.*"))


def test_failed_snapshot_undo_restores_root_session_target_and_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _memory_session(tmp_path)
    stage = session._stage
    root = stage.GetRootLayer()
    session_layer = stage.GetSessionLayer()

    stage.SetEditTarget(root)
    UsdGeom.Xform.Define(stage, "/World/RootA")
    stage.SetEditTarget(session_layer)
    UsdGeom.Xform.Define(stage, "/World/SessionA")
    stage.SetEditTarget(root)
    session._active_cam = "/World/CameraA"
    state_a = session._stage_layer_snapshot()

    stage.SetEditTarget(root)
    UsdGeom.Xform.Define(stage, "/World/RootB")
    stage.SetEditTarget(session_layer)
    UsdGeom.Xform.Define(stage, "/World/SessionB")
    session._active_cam = "/World/CameraB"
    state_b = session._stage_layer_snapshot()
    session.history.record(
        Op(
            command="camera.place",
            inverse={"undo": state_a, "redo": state_b},
        )
    )

    original_apply = session._apply_change

    def apply_then_fail(change: dict) -> None:
        original_apply(change)
        raise RuntimeError("injected undo failure")

    monkeypatch.setattr(session, "_apply_change", apply_then_fail)
    response = session.undo()

    assert not response.ok
    assert "rolled back" in response.issues[0].message
    assert root.ExportToString() == state_b["root"]
    assert session_layer.ExportToString() == state_b["session"]
    assert stage.GetEditTarget().GetLayer().identifier == session_layer.identifier
    assert session._active_cam == "/World/CameraB"
    assert len(session.history.entries()) == 1
    assert not session.history.can_redo

    monkeypatch.setattr(session, "_apply_change", original_apply)
    undone = session.undo()
    assert undone.ok
    assert root.ExportToString() == state_a["root"]
    assert session_layer.ExportToString() == state_a["session"]
    assert stage.GetEditTarget().GetLayer().identifier == root.identifier
    assert session._active_cam == "/World/CameraA"
    assert session.history.can_redo

    redone = session.redo()
    assert redone.ok
    assert root.ExportToString() == state_b["root"]
    assert session_layer.ExportToString() == state_b["session"]
    assert stage.GetEditTarget().GetLayer().identifier == session_layer.identifier
    assert session._active_cam == "/World/CameraB"
    assert not session.history.can_redo
