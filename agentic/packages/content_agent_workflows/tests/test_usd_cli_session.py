# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tomllib
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from content_agent_workflows.common import usd_cli_session
from content_agent_workflows.common.usd_cli import UsdCliPackageRoute
from content_agent_workflows.common.usd_cli_session import (
    MAX_PARENT_USD_CLI_SERVER_STATE_BYTES,
    PARENT_USD_CLI_SESSION_IDENTITY_ENV,
    PARENT_USD_CLI_SESSION_IDENTITY_SHA256_ENV,
    ParentUsdCliArtifactIdentity,
    ParentUsdCliGeneratedSourceIdentity,
    ParentUsdCliSessionIdentity,
    ParentUsdCliStagedSourceIdentity,
    WorkflowUsdCliCommandResult,
    WorkflowUsdCliSession,
    _linux_process_identity,
    _parent_usd_cli_session_identity_from_environment,
    _validated_parent_usd_cli_session_identity,
    _verify_live_parent_usd_cli_daemon,
    parent_usd_cli_daemon_identity_sha256,
    validated_ovrtx_render_metadata,
)
from content_agent_workflows.physics import usd_cli_ops


def _symlink_or_skip(
    link: Path,
    target: Path,
    *,
    target_is_directory: bool = False,
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise


def _descriptor_matches_path(descriptor: int, path: Path) -> bool:
    if os.name == "nt":
        try:
            opened = os.fstat(descriptor)
            current = path.stat()
        except OSError:
            return False
        return (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)
    try:
        target = os.readlink(f"/proc/self/fd/{descriptor}")
    except OSError:
        return False
    return Path(target).name == path.name


def _write_package_route(
    root: Path,
    *,
    source_revision: str = "d" * 40,
) -> UsdCliPackageRoute:
    root.mkdir(parents=True, exist_ok=True)
    wrapper = root / "usd-cli-tel"
    target = root / "usd-cli"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    wrapper.chmod(0o700)
    target.chmod(0o700)
    return UsdCliPackageRoute(
        wrapper=wrapper,
        target=target,
        source_root=root,
        source_revision=source_revision,
    )


def _write_daemon_server_state(run_dir: Path) -> None:
    state_dir = run_dir / ".usd-cli"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "server.json").write_text(
        json.dumps(
            {
                "host": "127.0.0.1",
                "port": 4567,
                "token": "test-daemon-token",
            }
        ),
        encoding="utf-8",
    )


def _write_parent_session_identity(
    *,
    repo_root: Path,
    run_dir: Path,
    route: UsdCliPackageRoute,
    generated: bool = False,
) -> tuple[Path, str]:
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)

    def artifact(path: Path) -> ParentUsdCliArtifactIdentity:
        payload = path.read_bytes()
        return ParentUsdCliArtifactIdentity(
            path=str(path.resolve()),
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    if generated:
        source_intent = run_dir / "inputs" / "source-intent.json"
        source_intent.parent.mkdir(parents=True)
        source_intent.write_text('{"mode":"cad_modeling"}\n', encoding="utf-8")
        source_image = repo_root / "source.png"
        source_image.write_bytes(b"source-image")
        source: ParentUsdCliStagedSourceIdentity | ParentUsdCliGeneratedSourceIdentity
        source = ParentUsdCliGeneratedSourceIdentity(
            source_intent=artifact(source_intent),
            source_images=[artifact(source_image)],
        )
    else:
        original = repo_root / "original.usda"
        original.write_text("#usda 1.0\n", encoding="utf-8")
        staged = run_dir / "inputs" / "asset_source" / "asset.usda"
        staged.parent.mkdir(parents=True)
        staged.write_bytes(original.read_bytes())
        manifest = raw_dir / "staged_input_asset_source.json"
        manifest.write_text("{}\n", encoding="utf-8")
        source = ParentUsdCliStagedSourceIdentity(
            original_source=artifact(original),
            staged_source=artifact(staged),
            staging_manifest=artifact(manifest),
            dependency_digest_set_sha256="c" * 64,
        )
    probe_dir = raw_dir / "ovrtx_probe"
    probe_dir.mkdir()
    probe_image = probe_dir / "ovrtx_readiness_probe.png"
    Image.new("RGB", (64, 64), "green").save(probe_image)
    readiness = raw_dir / "ovrtx_probe.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.ovrtx-probe.v1",
                "probe": {
                    "schema_version": "usd-cli.render-probe.v1",
                    "capabilities": ["appearance.clear.v1"],
                    "required_engine": "ovrtx",
                    "resolved_renderer": "ovrtx",
                    "engine": "ovrtx",
                    "transport": "local",
                    "ready": True,
                    "render": {
                        "path": str(probe_image),
                        "width": 64,
                        "height": 64,
                        "size_bytes": probe_image.stat().st_size,
                        "sha256": hashlib.sha256(probe_image.read_bytes()).hexdigest(),
                        "backend": "ovrtx",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    launcher = repo_root / "launcher.py"
    launcher.write_text("# launcher\n", encoding="utf-8")

    launch_id = "test-parent-launch"
    identity = ParentUsdCliSessionIdentity(
        created_at="2026-08-16T00:00:00Z",
        launch_id=launch_id,
        run_id="asset-test",
        run_dir=str(run_dir.resolve()),
        repository_root=str(repo_root.resolve()),
        parent_session_id="workflow-parent",
        project_id="a" * 24,
        instance_id="parent-instance",
        daemon_identity_sha256="b" * 64,
        server_port=4567,
        allowed_roots=[str(run_dir.resolve())],
        source=source,
        readiness_artifact=artifact(readiness),
        launcher_implementation=artifact(launcher),
        usd_cli_version="usd-cli test",
        usd_cli_source_revision=route.source_revision,
        usd_cli_wrapper=str(route.wrapper.resolve()),
        usd_cli_executable=str(route.target.resolve()),
    )
    identity_path = raw_dir / f"asset_usd_cli_session_{launch_id}.json"
    payload = (
        json.dumps(identity.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    identity_path.write_bytes(payload)
    return identity_path, hashlib.sha256(payload).hexdigest()


def _write_physics_source(path: Path, *, instanceable: bool = False) -> Path:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    if instanceable:
        UsdGeom.Xform.Define(stage, "/Template")
        UsdGeom.Mesh.Define(stage, "/Template/Mesh")
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        body.GetReferences().AddInternalReference("/Template")
        body.SetInstanceable(True)
    else:
        UsdGeom.Xform.Define(stage, "/World/Body")
        UsdGeom.Mesh.Define(stage, "/World/Body/Mesh")
    assert stage.GetRootLayer().Save()
    return path


def test_direction_angles_rejects_invalid_direction() -> None:
    with pytest.raises(ValueError, match="Invalid render direction"):
        WorkflowUsdCliSession.direction_angles("front")


def test_stage_up_axis_drives_direction_angles(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    source = tmp_path / "y-up.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/World")
    assert stage.GetRootLayer().Save()

    assert WorkflowUsdCliSession.stage_up_axis_is_y(source) is True
    assert WorkflowUsdCliSession.direction_angles("+z", up_axis_y=True) == (
        0.0,
        0.0,
    )
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    assert stage.GetRootLayer().Save()
    assert WorkflowUsdCliSession.stage_up_axis_is_y(source) is False
    assert WorkflowUsdCliSession.direction_angles("+z", up_axis_y=False) == (
        0.0,
        90.0,
    )


def test_physics_visual_review_selects_bounded_representative_frames(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    recording = tmp_path / "recording.usda"
    stage = Usd.Stage.CreateNew(str(recording))
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(30)
    stage.GetRootLayer().Save()

    assert usd_cli_ops._representative_frame_spec(recording) == ("0,4,9,13,17,21,26,30")


def test_physics_visual_review_preserves_fractional_authored_range(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    recording = tmp_path / "fractional-authored.usda"
    stage = Usd.Stage.CreateNew(str(recording))
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetStartTimeCode(0.25)
    stage.SetEndTimeCode(1.25)
    stage.GetRootLayer().Save()

    assert usd_cli_ops._representative_frame_spec(recording) == "0.25,1.25"


def test_physics_visual_review_preserves_fractional_sample_extrema(
    tmp_path: Path,
) -> None:
    from pxr import Gf, Usd, UsdGeom

    recording = tmp_path / "fractional-samples.usda"
    stage = Usd.Stage.CreateNew(str(recording))
    xform = UsdGeom.Xform.Define(stage, "/World")
    translate = xform.AddTranslateOp()
    translate.Set(Gf.Vec3d(0), 0.5)
    translate.Set(Gf.Vec3d(1), 1.5)
    stage.GetRootLayer().Save()

    assert usd_cli_ops._representative_frame_spec(recording) == "0.5,1.5"


def test_session_grants_external_inputs_read_only_daemon_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_root = tmp_path / "workflow"
    project = owner_root / "usd-project"
    source_root = tmp_path / "immutable-sources"
    owner_root.mkdir()
    source_root.mkdir()
    package_route = _write_package_route(tmp_path / "package-route")
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.find_usd_cli_repository_root",
        lambda _path: tmp_path,
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.resolve_package_owned_usd_cli_route",
        lambda _path: package_route,
    )
    WorkflowUsdCliSession.create(
        owner_root=owner_root,
        project_dir=project,
        identity="test",
        workflow="test-workflow",
        input_roots=(source_root,),
    )

    config = tomllib.loads((project / ".usd-cli" / "config.toml").read_text())
    assert set(config["server"]["allowed_roots"]) == {
        str(owner_root.resolve()),
        str(source_root.resolve()),
    }
    assert config["server"]["allowed_write_roots"] == [str(project.absolute())]


def test_session_rejects_package_route_before_creating_local_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_root = tmp_path / "workflow"
    project = owner_root / "usd-project"
    owner_root.mkdir()
    monkeypatch.setattr(
        usd_cli_session,
        "find_usd_cli_repository_root",
        lambda _path: tmp_path,
    )

    def reject_route(_repo_root: Path) -> Any:
        assert not project.exists()
        raise RuntimeError("editable from a foreign source")

    monkeypatch.setattr(
        usd_cli_session,
        "resolve_package_owned_usd_cli_route",
        reject_route,
    )

    with pytest.raises(RuntimeError, match="editable from a foreign source"):
        WorkflowUsdCliSession.create(
            owner_root=owner_root,
            project_dir=project,
            identity="test",
            workflow="test-workflow",
        )

    assert not project.exists()


def test_session_rejects_drift_from_provided_route_pin_before_local_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_root = tmp_path / "workflow"
    project = owner_root / "usd-project"
    owner_root.mkdir()
    repository_root = tmp_path / "repository"
    original = _write_package_route(repository_root)
    current = {"route": original}
    monkeypatch.setattr(
        usd_cli_session,
        "find_usd_cli_repository_root",
        lambda _path: repository_root,
    )
    monkeypatch.setattr(
        usd_cli_session,
        "resolve_package_owned_usd_cli_route",
        lambda _path: current["route"],
    )
    route_pin = WorkflowUsdCliSession.preflight_package_route(
        repository_root=repository_root
    )
    current["route"] = UsdCliPackageRoute(
        wrapper=original.wrapper,
        target=original.target,
        source_root=original.source_root,
        source_revision="e" * 40,
    )

    with pytest.raises(RuntimeError, match="implementation identity changed"):
        WorkflowUsdCliSession.create(
            owner_root=owner_root,
            project_dir=project,
            identity="test",
            workflow="test-workflow",
            package_route=route_pin,
        )

    assert not project.exists()


@pytest.mark.parametrize("drift", ["path", "source_revision"])
def test_session_rejects_pinned_package_route_drift_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    owner_root = tmp_path / "workflow"
    project = owner_root / "usd-project"
    owner_root.mkdir()
    repository_root = tmp_path / "repository"
    original = _write_package_route(repository_root)
    current = {"route": original}
    monkeypatch.setattr(
        usd_cli_session,
        "find_usd_cli_repository_root",
        lambda _path: repository_root,
    )
    monkeypatch.setattr(
        usd_cli_session,
        "resolve_package_owned_usd_cli_route",
        lambda _path: current["route"],
    )
    session = WorkflowUsdCliSession.create(
        owner_root=owner_root,
        project_dir=project,
        identity="test",
        workflow="test-workflow",
    )
    if drift == "path":
        replacement_wrapper = repository_root / "alternate-usd-cli-tel"
        replacement_wrapper.write_bytes(original.wrapper.read_bytes())
        replacement_wrapper.chmod(0o700)
        current["route"] = UsdCliPackageRoute(
            wrapper=replacement_wrapper,
            target=original.target,
            source_root=original.source_root,
            source_revision=original.source_revision,
        )
    else:
        current["route"] = UsdCliPackageRoute(
            wrapper=original.wrapper,
            target=original.target,
            source_root=original.source_root,
            source_revision="e" * 40,
        )
    monkeypatch.setattr(
        usd_cli_session,
        "controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        usd_cli_session,
        "run_bounded_usd_cli_subprocess",
        lambda *_args, **_kwargs: pytest.fail("drifted launcher was executed"),
    )

    with pytest.raises(RuntimeError, match="implementation identity changed"):
        session.run_json(["info"])
    with pytest.raises(RuntimeError, match="implementation identity changed"):
        session.close()

    with pytest.raises(ValueError, match="primary operation failed") as raised:
        try:
            raise ValueError("primary operation failed")
        finally:
            session.close()
    cleanup_error = getattr(raised.value, "usd_cli_cleanup_error", None)
    assert isinstance(cleanup_error, RuntimeError)
    assert "implementation identity changed" in str(cleanup_error)


def test_session_rejects_same_uid_wrapper_replacement_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_root = tmp_path / "workflow"
    project = owner_root / "usd-project"
    owner_root.mkdir()
    repository_root = tmp_path / "repository"
    route = _write_package_route(repository_root)
    monkeypatch.setattr(
        usd_cli_session,
        "find_usd_cli_repository_root",
        lambda _path: repository_root,
    )
    monkeypatch.setattr(
        usd_cli_session,
        "resolve_package_owned_usd_cli_route",
        lambda _path: route,
    )
    session = WorkflowUsdCliSession.create(
        owner_root=owner_root,
        project_dir=project,
        identity="test",
        workflow="test-workflow",
    )
    original_uid = route.wrapper.stat().st_uid
    replacement = repository_root / "replacement"
    replacement.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    replacement.chmod(0o700)
    os.replace(replacement, route.wrapper)
    assert route.wrapper.stat().st_uid == original_uid
    monkeypatch.setattr(
        usd_cli_session,
        "controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        usd_cli_session,
        "run_bounded_usd_cli_subprocess",
        lambda *_args, **_kwargs: pytest.fail("replaced launcher was executed"),
    )

    with pytest.raises(RuntimeError, match="implementation identity changed"):
        session.run_json(["info"])
    with pytest.raises(RuntimeError, match="implementation identity changed"):
        session.close()


def test_session_preserves_nearest_project_render_config_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_root = tmp_path / "caller-project"
    project = owner_root / "workflow-runs" / "run-1"
    caller_state = owner_root / ".usd-cli"
    caller_state.mkdir(parents=True)
    package_route = _write_package_route(tmp_path / "package-route")
    (caller_state / "config.toml").write_text(
        """
[server]
host = "192.0.2.10"
allowed_roots = ["/caller-only"]

[render]
renderer = "remote"
remote_url = "https://ovrtx.example.test"
remote_api_key = "project-key"
remote_verify_version = true
[[render.backends]]
url = "https://ovrtx.example.test"
api_key = "primary-key"
[[render.backends]]
url = "https://ovrtx-backup.example.test/"
api_key = "backup-key"

[session]
private_value = "do-not-inherit"
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.find_usd_cli_repository_root",
        lambda _path: tmp_path,
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.resolve_package_owned_usd_cli_route",
        lambda _path: package_route,
    )
    monkeypatch.setenv("NVCF_RENDER_FUNCTION_ID", "ambient-render-function")
    monkeypatch.setenv("NGC_API_KEY", "ambient-key-must-not-win")

    session = WorkflowUsdCliSession.create(
        owner_root=owner_root,
        project_dir=project,
        identity="remote-render",
        workflow="test-workflow",
    )

    config = tomllib.loads((project / ".usd-cli" / "config.toml").read_text())
    assert config["render"] == {
        "renderer": "remote",
        "remote_url": "https://ovrtx.example.test",
        "remote_api_key": "",
        "remote_verify_version": True,
        "backends": [
            {"url": "https://ovrtx.example.test"},
            {"url": "https://ovrtx-backup.example.test/"},
        ],
    }
    assert "project-key" not in repr(session)
    assert "primary-key" not in repr(session)
    assert "backup-key" not in repr(session)
    assert config["server"]["host"] == "127.0.0.1"
    assert config["server"]["allowed_roots"] == [str(owner_root.resolve())]
    assert "session" not in config
    assert session._render_credentials["remote_api_key"] == "project-key"


def test_session_maps_documented_nvcf_renderer_without_persisting_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    global_state = home / ".config" / "usd-cli"
    global_state.mkdir(parents=True)
    (global_state / "config.toml").write_text(
        """
[render]
renderer = "remote"
remote_url = "https://stale.example.test"
remote_api_key = "stale-global-key"
[[render.backends]]
url = "https://stale.example.test"
api_key = "stale-pool-key"
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    owner_root = tmp_path / "caller-project"
    owner_root.mkdir()
    project = owner_root / "workflow-runs" / "run-1"
    package_route = _write_package_route(tmp_path / "package-route")
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.find_usd_cli_repository_root",
        lambda _path: tmp_path,
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.resolve_package_owned_usd_cli_route",
        lambda _path: package_route,
    )
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.setenv("NVCF_RENDER_FUNCTION_ID", "render-function-id")
    monkeypatch.setenv("NGC_API_KEY", "nvcf-parent-only-key")

    session = WorkflowUsdCliSession.create(
        owner_root=owner_root,
        project_dir=project,
        identity="nvcf-render",
        workflow="test-workflow",
        render_config=usd_cli_session._environment_remote_render_config(),
    )

    config_text = (project / ".usd-cli" / "config.toml").read_text(encoding="utf-8")
    config = tomllib.loads(config_text)
    assert config["render"] == {
        "renderer": "remote",
        "remote_url": "https://render-function-id.invocation.api.nvcf.nvidia.com",
        "remote_api_key": "",
        "backends": [],
    }
    assert "nvcf-parent-only-key" not in config_text
    assert "nvcf-parent-only-key" not in repr(session)
    assert session._render_credentials == {"remote_api_key": "nvcf-parent-only-key"}
    from usd_core.config import load_config

    resolved = load_config(project)
    assert resolved.render["remote_url"] == (
        "https://render-function-id.invocation.api.nvcf.nvidia.com"
    )
    assert resolved.render["remote_api_key"] == ""
    assert resolved.render["backends"] == []
    assert "stale" not in repr(resolved.render)


def test_session_explicit_local_renderer_shadows_global_remote_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    global_state = home / ".config" / "usd-cli"
    global_state.mkdir(parents=True)
    (global_state / "config.toml").write_text(
        """
[render]
renderer = "remote"
remote_url = "https://stale.example.test"
remote_api_key = "stale-global-key"
[[render.backends]]
url = "https://stale.example.test"
api_key = "stale-pool-key"
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    owner_root = tmp_path / "caller-project"
    owner_root.mkdir()
    project = owner_root / "workflow-runs" / "run-1"
    package_route = _write_package_route(tmp_path / "package-route")
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.find_usd_cli_repository_root",
        lambda _path: tmp_path,
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.resolve_package_owned_usd_cli_route",
        lambda _path: package_route,
    )

    session = WorkflowUsdCliSession.create(
        owner_root=owner_root,
        project_dir=project,
        identity="local-render",
        workflow="test-workflow",
        render_config={
            "renderer": "ovrtx",
            "remote_url": "",
            "remote_api_key": "",
            "backends": [],
        },
    )

    config_text = (project / ".usd-cli" / "config.toml").read_text(encoding="utf-8")
    config = tomllib.loads(config_text)
    assert config["render"] == {
        "renderer": "ovrtx",
        "remote_url": "",
        "remote_api_key": "",
        "backends": [],
    }
    assert session._render_credentials == {}
    from usd_core.config import load_config

    resolved = load_config(project)
    assert resolved.render["renderer"] == "ovrtx"
    assert resolved.render["remote_url"] == ""
    assert resolved.render["remote_api_key"] == ""
    assert resolved.render["backends"] == []
    assert "stale" not in repr(resolved.render)


def test_session_does_not_send_ngc_key_to_custom_render_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_root = tmp_path / "caller-project"
    owner_root.mkdir()
    project = owner_root / "workflow-run"
    package_route = _write_package_route(tmp_path / "package-route")
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.find_usd_cli_repository_root",
        lambda _path: tmp_path,
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.resolve_package_owned_usd_cli_route",
        lambda _path: package_route,
    )
    monkeypatch.setenv("RENDER_ENDPOINT", "https://renderer.example.test")
    monkeypatch.setenv("NVCF_RENDER_FUNCTION_ID", "ignored-function-id")
    monkeypatch.setenv("NGC_API_KEY", "must-not-cross-to-custom-endpoint")

    session = WorkflowUsdCliSession.create(
        owner_root=owner_root,
        project_dir=project,
        identity="custom-render",
        workflow="test-workflow",
        render_config=usd_cli_session._environment_remote_render_config(),
    )

    config = tomllib.loads((project / ".usd-cli" / "config.toml").read_text())
    assert config["render"] == {
        "renderer": "remote",
        "remote_url": "https://renderer.example.test",
        "remote_api_key": "",
        "backends": [],
    }
    assert session._render_credentials == {}


def test_session_rejects_unsafe_environment_render_endpoint_before_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_root = tmp_path / "caller-project"
    owner_root.mkdir()
    project = owner_root / "workflow-run"
    package_route = _write_package_route(tmp_path / "package-route")
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.find_usd_cli_repository_root",
        lambda _path: tmp_path,
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.resolve_package_owned_usd_cli_route",
        lambda _path: package_route,
    )
    monkeypatch.setenv(
        "RENDER_ENDPOINT",
        "https://user:secret@renderer.example.test?redirect=1",
    )

    with pytest.raises(RuntimeError, match="without userinfo"):
        WorkflowUsdCliSession.create(
            owner_root=owner_root,
            project_dir=project,
            identity="unsafe-render",
            workflow="test-workflow",
            render_config=usd_cli_session._environment_remote_render_config(),
        )

    assert not project.exists()


def test_session_does_not_select_remote_renderer_from_ambient_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_root = tmp_path / "caller-project"
    owner_root.mkdir()
    project = owner_root / "workflow-run"
    package_route = _write_package_route(tmp_path / "package-route")
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.find_usd_cli_repository_root",
        lambda _path: tmp_path,
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.resolve_package_owned_usd_cli_route",
        lambda _path: package_route,
    )
    monkeypatch.setenv("RENDER_ENDPOINT", "https://renderer.example.test")

    session = WorkflowUsdCliSession.create(
        owner_root=owner_root,
        project_dir=project,
        identity="local-render",
        workflow="test-workflow",
    )

    config = tomllib.loads((project / ".usd-cli" / "config.toml").read_text())
    assert "render" not in config
    assert session._render_credentials == {}


def test_child_session_reuses_parent_daemon_without_lifecycle_or_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    route = _write_package_route(repo_root)
    wrapper = route.wrapper
    parent_run = tmp_path / "asset-run"
    identity_path, identity_sha256 = _write_parent_session_identity(
        repo_root=repo_root,
        run_dir=parent_run,
        route=route,
    )
    _write_daemon_server_state(parent_run)
    child_run = parent_run / "articulation" / "domain-run"
    child_run.mkdir(parents=True)
    staged_source = parent_run / "inputs" / "asset_source" / "asset.usda"
    monkeypatch.setenv(PARENT_USD_CLI_SESSION_IDENTITY_ENV, str(identity_path))
    monkeypatch.setenv(
        PARENT_USD_CLI_SESSION_IDENTITY_SHA256_ENV,
        identity_sha256,
    )
    monkeypatch.setenv("OVRTX_API_KEY", "must-not-cross")
    monkeypatch.setenv("USD_CLI_RENDER_REMOTE_API_KEY", "also-must-not-cross")
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.find_usd_cli_repository_root",
        lambda _path: repo_root,
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.resolve_package_owned_usd_cli_route",
        lambda _path: route,
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session._verify_live_parent_usd_cli_daemon",
        lambda _identity, require_os_identity=False: None,
    )
    calls: list[dict[str, Any]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"ok": True}),
            stderr="",
        )

    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.run_bounded_usd_cli_subprocess",
        run,
    )

    session = WorkflowUsdCliSession.create(
        owner_root=child_run,
        project_dir=child_run,
        identity=str(child_run),
        workflow="asset-articulation",
        input_roots=(staged_source,),
    )
    session.open(staged_source)
    session.open(staged_source)
    reused_probe = session.require_ovrtx(child_run / "probe")
    with pytest.raises(RuntimeError, match="cannot mutate parent daemon lifecycle"):
        session.run_json(["server", "stop"])
    session.close()
    session.close()

    assert session.daemon_project_dir == parent_run.resolve()
    expected_session_id = (
        "workflow-" + hashlib.sha256(str(child_run).encode("utf-8")).hexdigest()[:20]
    )
    assert session.session_id == expected_session_id
    assert (child_run / ".parent-usd-cli-session").read_text(
        encoding="utf-8"
    ) == f"{expected_session_id}\n"
    sibling_run = parent_run / "physics" / "domain-run"
    sibling_run.mkdir(parents=True)
    sibling_session = WorkflowUsdCliSession.create(
        owner_root=sibling_run,
        project_dir=sibling_run,
        identity=str(sibling_run),
        workflow="asset-physics",
        input_roots=(staged_source,),
    )
    assert sibling_session.daemon_project_dir == parent_run.resolve()
    assert sibling_session.session_id != session.session_id
    assert reused_probe["ready"] is True
    assert reused_probe["execution_source"] == "parent-readiness-reuse"
    reused_render = reused_probe["render"]
    assert isinstance(reused_render, dict)
    assert Path(str(reused_render["path"])).parent == child_run / "probe"
    assert Path(str(reused_render["path"])).is_file()

    parent_render = parent_run / "raw" / "ovrtx_probe" / "ovrtx_readiness_probe.png"
    original_size = parent_render.stat().st_size
    Image.new("RGB", (64, 64), "red").save(parent_render)
    assert parent_render.stat().st_size == original_size
    with pytest.raises(RuntimeError, match="render evidence changed"):
        session.require_ovrtx(child_run / "tampered-probe")
    assert not (child_run / ".usd-cli").exists()
    assert len(calls) == 3
    for call in calls[:2]:
        assert call["cwd"] == child_run.resolve()
        assert call["env"]["USD_CLI_NO_DAEMON"] == "1"
        assert call["env"]["CONTENT_WORKFLOW_PARENT_USD_CLI_MANAGED"] == "1"
        assert call["env"]["CONTENT_WORKFLOW_PARENT_USD_CLI_PROXY_ALLOWED"] == "1"
        assert call["env"]["USD_CLI_LIFECYCLE_EXTERNALLY_OWNED"] == "1"
        assert "OVRTX_API_KEY" not in call["env"]
        assert "USD_CLI_RENDER_REMOTE_API_KEY" not in call["env"]
        assert call["command"][:6] == [
            str(wrapper),
            "--json",
            "--server",
            "http://127.0.0.1:4567",
            "--session",
            session.session_id,
        ]
        assert call["command"][-3:] == [
            "open",
            str(staged_source.resolve()),
            "--force-reload",
        ]
    release_call = calls[2]
    assert release_call["cwd"] == child_run.resolve()
    assert release_call["env"]["USD_CLI_NO_DAEMON"] == "1"
    assert release_call["env"]["CONTENT_WORKFLOW_PARENT_USD_CLI_PROXY_ALLOWED"] == "1"
    assert release_call["command"][:6] == [
        str(wrapper),
        "--json",
        "--server",
        "http://127.0.0.1:4567",
        "--session",
        session.session_id,
    ]
    assert release_call["command"][-4:] == [
        "server",
        "release-session",
        "--name",
        session.session_id,
    ]
    (child_run / ".usd-cli").mkdir()
    with pytest.raises(RuntimeError, match="competing daemon state"):
        WorkflowUsdCliSession.create(
            owner_root=child_run,
            project_dir=child_run,
            identity=str(child_run),
            workflow="asset-articulation",
            input_roots=(staged_source,),
        )


def test_standalone_session_reports_render_probe_execution_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel"),
        workflow="material-task",
    )
    calls: list[list[str]] = []

    def run_json(
        _session: WorkflowUsdCliSession,
        arguments: list[str],
        *,
        timeout_seconds: float = 1800.0,
    ) -> dict[str, Any]:
        del timeout_seconds
        calls.append(arguments)
        return {"schema_version": "usd-cli.render-probe.v1", "ready": True}

    monkeypatch.setattr(WorkflowUsdCliSession, "run_json", run_json)

    probe = session.require_ovrtx(tmp_path / "probe")

    assert probe["execution_source"] == "render-probe"
    assert calls == [
        [
            "render-probe",
            "--require-engine",
            "ovrtx",
            "--output-dir",
            str(tmp_path / "probe"),
        ]
    ]


def test_attached_provider_free_readiness_runs_exact_parent_render_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "asset-run"
    raw = parent / "raw"
    child = parent / "articulation" / "domain-run"
    raw.mkdir(parents=True)
    child.mkdir(parents=True)
    readiness = raw / "provider-free-readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-workflow-cli.asset-usd-cli-provider-free-readiness.v1"
                ),
                "usd_cli_version": "usd-cli 1.2.3",
                "usd_cli_source_revision": "d" * 40,
                "probe": {
                    "selected_mode": "agentic",
                    "provider_readiness": "not_requested",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = readiness.read_bytes()
    artifact = ParentUsdCliArtifactIdentity(
        path=str(readiness.resolve()),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    session = WorkflowUsdCliSession(
        project_dir=child,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel"),
        workflow="asset-articulation",
        daemon_project_dir=parent,
        _parent_readiness_artifact=artifact,
    )
    calls: list[list[str]] = []

    def run_json(
        _session: WorkflowUsdCliSession,
        arguments: list[str],
        *,
        timeout_seconds: float = 1800.0,
    ) -> dict[str, Any]:
        del timeout_seconds
        calls.append(arguments)
        return {"schema_version": "usd-cli.render-probe.v1", "ready": True}

    monkeypatch.setattr(WorkflowUsdCliSession, "run_json", run_json)
    output = child / "probe"

    probe = session.require_ovrtx(output)

    assert probe == {
        "schema_version": "usd-cli.render-probe.v1",
        "ready": True,
        "execution_source": "attached-render-probe",
    }
    assert calls == [
        [
            "render-probe",
            "--require-engine",
            "ovrtx",
            "--output-dir",
            str(output),
        ]
    ]


@pytest.mark.parametrize(
    "readiness_mutation",
    [
        {"extra": "not-exact"},
        {"probe": {"selected_mode": "agentic"}},
        {
            "probe": {
                "selected_mode": "agentic",
                "provider_readiness": "ready",
            }
        },
    ],
)
def test_attached_missing_render_fails_closed_outside_exact_provider_free_schema(
    tmp_path: Path,
    readiness_mutation: dict[str, Any],
) -> None:
    parent = tmp_path / "asset-run"
    raw = parent / "raw"
    child = parent / "validation" / "domain-run"
    raw.mkdir(parents=True)
    child.mkdir(parents=True)
    readiness_payload: dict[str, Any] = {
        "schema_version": (
            "content-workflow-cli.asset-usd-cli-provider-free-readiness.v1"
        ),
        "usd_cli_version": "usd-cli 1.2.3",
        "usd_cli_source_revision": "d" * 40,
        "probe": {
            "selected_mode": "agentic",
            "provider_readiness": "not_requested",
        },
    }
    readiness_payload.update(readiness_mutation)
    readiness = raw / "readiness.json"
    readiness.write_text(json.dumps(readiness_payload) + "\n", encoding="utf-8")
    payload = readiness.read_bytes()
    session = WorkflowUsdCliSession(
        project_dir=child,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel"),
        workflow="asset-validation",
        daemon_project_dir=parent,
        _parent_readiness_artifact=ParentUsdCliArtifactIdentity(
            path=str(readiness.resolve()),
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        ),
    )

    with pytest.raises(RuntimeError, match="has no render evidence"):
        session.require_ovrtx(child / "probe")


def test_child_session_rejects_parent_render_configuration_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    wrapper = repo_root / "usd-cli-tel"
    target = repo_root / "usd-cli"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    route = UsdCliPackageRoute(
        wrapper=wrapper,
        target=target,
        source_root=repo_root,
        source_revision="d" * 40,
    )
    parent_run = tmp_path / "asset-run"
    identity_path, identity_sha256 = _write_parent_session_identity(
        repo_root=repo_root,
        run_dir=parent_run,
        route=route,
    )
    child_run = parent_run / "validation" / "domain-run"
    child_run.mkdir(parents=True)
    monkeypatch.setenv(PARENT_USD_CLI_SESSION_IDENTITY_ENV, str(identity_path))
    monkeypatch.setenv(
        PARENT_USD_CLI_SESSION_IDENTITY_SHA256_ENV,
        identity_sha256,
    )

    with pytest.raises(
        RuntimeError,
        match="cannot override the parent render configuration",
    ):
        WorkflowUsdCliSession.create(
            owner_root=child_run,
            project_dir=child_run,
            identity=str(child_run),
            workflow="asset-validation",
            render_config={},
        )


def test_parent_handoff_rejects_incomplete_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PARENT_USD_CLI_SESSION_IDENTITY_ENV, "/tmp/identity.json")
    monkeypatch.delenv(PARENT_USD_CLI_SESSION_IDENTITY_SHA256_ENV, raising=False)

    with pytest.raises(RuntimeError, match="environment is incomplete"):
        _parent_usd_cli_session_identity_from_environment()


def test_process_identity_reads_current_process() -> None:
    identity = _linux_process_identity(os.getpid())

    assert identity is not None
    if os.name == "nt":
        assert identity[0].startswith("w")
        assert identity[1:] == (os.getpid(), os.getpid())
    else:
        assert identity[0].startswith("t")
        assert identity[1] == os.getpgrp()
        assert identity[2] == os.getsid(0)


def test_parent_handoff_authenticates_exact_live_daemon_and_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    wrapper = repo_root / "usd-cli-tel"
    target = repo_root / "usd-cli"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    route = UsdCliPackageRoute(
        wrapper=wrapper,
        target=target,
        source_root=repo_root,
        source_revision="d" * 40,
    )
    parent_run = tmp_path / "asset-run"
    identity_path, _identity_sha256 = _write_parent_session_identity(
        repo_root=repo_root,
        run_dir=parent_run,
        route=route,
    )
    identity = ParentUsdCliSessionIdentity.model_validate_json(
        identity_path.read_bytes()
    )
    state = {
        "pid": 123,
        "process_start_token": "t1",
        "project_id": identity.project_id,
        "instance_id": identity.instance_id,
        "lifecycle_owner": "external",
        "host": identity.server_host,
        "port": identity.server_port,
        "token": "daemon-auth-token",
    }
    state_dir = parent_run / ".usd-cli"
    state_dir.mkdir()
    (state_dir / "server.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )
    identity = identity.model_copy(
        update={
            "daemon_identity_sha256": parent_usd_cli_daemon_identity_sha256(
                pid=123,
                process_start_token="t1",
                project_id=identity.project_id,
                instance_id=identity.instance_id,
                process_group_id=123,
                os_session_id=123,
            )
        }
    )
    requests: list[tuple[str, str, dict[str, str]]] = []

    class Response:
        status = 200

        @staticmethod
        def read(_limit: int) -> bytes:
            return json.dumps(
                {
                    "ok": True,
                    **{
                        field: state[field]
                        for field in (
                            "pid",
                            "process_start_token",
                            "project_id",
                            "instance_id",
                            "lifecycle_owner",
                        )
                    },
                    "sessions": {identity.parent_session_id: {}},
                }
            ).encode("utf-8")

    class Connection:
        def __init__(self, host: str, port: int, *, timeout: float) -> None:
            assert (host, port, timeout) == ("127.0.0.1", 4567, 2.0)

        def request(
            self,
            method: str,
            path: str,
            *,
            headers: dict[str, str],
        ) -> None:
            requests.append((method, path, headers))

        @staticmethod
        def getresponse() -> Response:
            return Response()

        @staticmethod
        def close() -> None:
            return None

    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session._linux_process_identity",
        lambda _pid: ("t1", 123, 123),
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.http.client.HTTPConnection",
        Connection,
    )

    _verify_live_parent_usd_cli_daemon(identity)

    assert requests == [
        (
            "GET",
            "/health",
            {
                "x-usd-cli-token": "daemon-auth-token",
                "x-ov-token": "daemon-auth-token",
                "x-3dsc-token": "daemon-auth-token",
            },
        )
    ]
    requests.clear()
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session._linux_process_identity",
        lambda _pid: pytest.fail("sandboxed child must not inspect the parent PID"),
    )
    connection = _verify_live_parent_usd_cli_daemon(identity, False)
    assert connection.server_url == "http://127.0.0.1:4567"
    assert connection.session_id == identity.parent_session_id
    assert connection.authentication_token.get_secret_value() == "daemon-auth-token"
    assert "daemon-auth-token" not in repr(connection)
    assert requests
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session._linux_process_identity",
        lambda _pid: ("t1", 999, 999),
    )
    with pytest.raises(RuntimeError, match="OS identity changed"):
        _verify_live_parent_usd_cli_daemon(identity)

    class OversizedResponse(Response):
        @staticmethod
        def read(_limit: int) -> bytes:
            return b"{" + b"x" * MAX_PARENT_USD_CLI_SERVER_STATE_BYTES

    class OversizedConnection(Connection):
        @staticmethod
        def getresponse() -> OversizedResponse:
            return OversizedResponse()

    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session._linux_process_identity",
        lambda _pid: ("t1", 123, 123),
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.http.client.HTTPConnection",
        OversizedConnection,
    )
    with pytest.raises(RuntimeError, match="health exceeds its size limit"):
        _verify_live_parent_usd_cli_daemon(identity)


@pytest.mark.parametrize(
    "proxy_url",
    [
        "http://srt:opaque-sdk-token@localhost:32123",
        "http://srt.run-context:opaque-cli-token@localhost:32123",
    ],
)
def test_parent_handoff_uses_authenticated_codex_proxy_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proxy_url: str,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    wrapper = repo_root / "usd-cli-tel"
    target = repo_root / "usd-cli"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    route = UsdCliPackageRoute(
        wrapper=wrapper,
        target=target,
        source_root=repo_root,
        source_revision="d" * 40,
    )
    parent_run = tmp_path / "asset-run"
    identity_path, _identity_sha256 = _write_parent_session_identity(
        repo_root=repo_root,
        run_dir=parent_run,
        route=route,
    )
    identity = ParentUsdCliSessionIdentity.model_validate_json(
        identity_path.read_bytes()
    )
    state = {
        "pid": 123,
        "process_start_token": "t1",
        "project_id": identity.project_id,
        "instance_id": identity.instance_id,
        "lifecycle_owner": "external",
        "host": identity.server_host,
        "port": identity.server_port,
        "token": "daemon-auth-token",
    }
    state_dir = parent_run / ".usd-cli"
    state_dir.mkdir()
    (state_dir / "server.json").write_text(json.dumps(state), encoding="utf-8")

    class ProxyResponse:
        status_code = 200
        content = json.dumps(
            {
                "ok": True,
                **{
                    field: state[field]
                    for field in (
                        "pid",
                        "process_start_token",
                        "project_id",
                        "instance_id",
                        "lifecycle_owner",
                    )
                },
                "sessions": {identity.parent_session_id: {}},
            }
        ).encode("utf-8")

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_args) -> None:  # noqa: ANN002
            return None

        def iter_bytes(self):  # noqa: ANN201
            yield self.content

    observed: list[tuple[str, dict[str, str]]] = []

    class ProxyClient:
        def __init__(self, *, proxy: str, trust_env: bool) -> None:
            assert proxy == proxy_url
            assert trust_env is False

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_args) -> None:  # noqa: ANN002
            return None

        def stream(
            self,
            method: str,
            url: str,
            *,
            headers: dict[str, str],
            timeout: float,
        ):
            assert method == "GET"
            assert timeout == 2.0
            observed.append((url, headers))
            return ProxyResponse()

    monkeypatch.setenv(
        usd_cli_session.PARENT_USD_CLI_PROXY_ALLOWED_ENV,
        "1",
    )
    monkeypatch.setenv("USD_CLI_ATTACHED_PROJECT_DIR", str(parent_run.resolve()))
    monkeypatch.setenv(
        "HTTP_PROXY",
        proxy_url,
    )
    monkeypatch.setattr("httpx.Client", ProxyClient)

    connection = _verify_live_parent_usd_cli_daemon(identity, False)

    assert connection.server_url == "http://127.0.0.1:4567"
    assert observed == [
        (
            "http://127.0.0.1:4567/health",
            {
                "x-usd-cli-token": "daemon-auth-token",
                "x-ov-token": "daemon-auth-token",
                "x-3dsc-token": "daemon-auth-token",
            },
        )
    ]


@pytest.mark.parametrize(
    "proxy_url",
    [
        "http://srt.:opaque-token@localhost:32123",
        "http://srt.run-context@localhost:32123",
        "http://other:opaque-token@localhost:32123",
    ],
)
def test_parent_handoff_rejects_untrusted_proxy_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proxy_url: str,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    wrapper = repo_root / "usd-cli-tel"
    target = repo_root / "usd-cli"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    route = UsdCliPackageRoute(
        wrapper=wrapper,
        target=target,
        source_root=repo_root,
        source_revision="d" * 40,
    )
    parent_run = tmp_path / "asset-run"
    identity_path, _identity_sha256 = _write_parent_session_identity(
        repo_root=repo_root,
        run_dir=parent_run,
        route=route,
    )
    identity = ParentUsdCliSessionIdentity.model_validate_json(
        identity_path.read_bytes()
    )
    monkeypatch.setenv(usd_cli_session.PARENT_USD_CLI_PROXY_ALLOWED_ENV, "1")
    monkeypatch.setenv("USD_CLI_ATTACHED_PROJECT_DIR", str(parent_run.resolve()))
    monkeypatch.setenv("HTTP_PROXY", proxy_url)

    assert usd_cli_session._attached_parent_http_proxy(identity) is None


def test_child_session_rejects_changed_parent_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    wrapper = repo_root / "usd-cli-tel"
    target = repo_root / "usd-cli"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    route = UsdCliPackageRoute(
        wrapper=wrapper,
        target=target,
        source_root=repo_root,
        source_revision="d" * 40,
    )
    parent_run = tmp_path / "asset-run"
    identity_path, identity_sha256 = _write_parent_session_identity(
        repo_root=repo_root,
        run_dir=parent_run,
        route=route,
    )
    child_run = parent_run / "domain-run"
    child_run.mkdir()
    identity_path.write_bytes(identity_path.read_bytes() + b" ")
    monkeypatch.setenv(PARENT_USD_CLI_SESSION_IDENTITY_ENV, str(identity_path))
    monkeypatch.setenv(
        PARENT_USD_CLI_SESSION_IDENTITY_SHA256_ENV,
        identity_sha256,
    )

    with pytest.raises(RuntimeError, match="identity digest changed"):
        WorkflowUsdCliSession.create(
            owner_root=child_run,
            project_dir=child_run,
            identity=str(child_run),
            workflow="asset-test",
        )


def test_child_session_rejects_changed_staged_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    wrapper = repo_root / "usd-cli-tel"
    target = repo_root / "usd-cli"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    route = UsdCliPackageRoute(
        wrapper=wrapper,
        target=target,
        source_root=repo_root,
        source_revision="d" * 40,
    )
    parent_run = tmp_path / "asset-run"
    identity_path, identity_sha256 = _write_parent_session_identity(
        repo_root=repo_root,
        run_dir=parent_run,
        route=route,
    )
    child_run = parent_run / "domain-run"
    child_run.mkdir()
    staged_source = parent_run / "inputs" / "asset_source" / "asset.usda"
    staged_source.write_text('#usda 1.0\ndef Xform "Changed" {}\n', encoding="utf-8")
    monkeypatch.setenv(PARENT_USD_CLI_SESSION_IDENTITY_ENV, str(identity_path))
    monkeypatch.setenv(
        PARENT_USD_CLI_SESSION_IDENTITY_SHA256_ENV,
        identity_sha256,
    )

    with pytest.raises(RuntimeError, match="staged source bytes changed"):
        WorkflowUsdCliSession.create(
            owner_root=child_run,
            project_dir=child_run,
            identity=str(child_run),
            workflow="asset-test",
        )


def test_child_session_rejects_changed_generated_source_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    wrapper = repo_root / "usd-cli-tel"
    target = repo_root / "usd-cli"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    route = UsdCliPackageRoute(
        wrapper=wrapper,
        target=target,
        source_root=repo_root,
        source_revision="d" * 40,
    )
    parent_run = tmp_path / "asset-run"
    identity_path, identity_sha256 = _write_parent_session_identity(
        repo_root=repo_root,
        run_dir=parent_run,
        route=route,
        generated=True,
    )
    (repo_root / "source.png").write_bytes(b"changed-image")
    monkeypatch.setenv(PARENT_USD_CLI_SESSION_IDENTITY_ENV, str(identity_path))
    monkeypatch.setenv(
        PARENT_USD_CLI_SESSION_IDENTITY_SHA256_ENV,
        identity_sha256,
    )

    with pytest.raises(RuntimeError, match="source image 1 bytes changed"):
        _parent_usd_cli_session_identity_from_environment()


def test_parent_identity_reuses_unchanged_large_artifact_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    wrapper = repo_root / "usd-cli-tel"
    target = repo_root / "usd-cli"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    route = UsdCliPackageRoute(
        wrapper=wrapper,
        target=target,
        source_root=repo_root,
        source_revision="d" * 40,
    )
    parent_run = tmp_path / "asset-run"
    identity_path, identity_sha256 = _write_parent_session_identity(
        repo_root=repo_root,
        run_dir=parent_run,
        route=route,
    )
    staged_source = parent_run / "inputs" / "asset_source" / "asset.usda"
    monkeypatch.setenv(PARENT_USD_CLI_SESSION_IDENTITY_ENV, str(identity_path))
    monkeypatch.setenv(
        PARENT_USD_CLI_SESSION_IDENTITY_SHA256_ENV,
        identity_sha256,
    )
    original_read = usd_cli_session.read_contained_artifact
    staged_reads = 0

    def counted_read(*args: Any, **kwargs: Any) -> Any:
        nonlocal staged_reads
        if Path(args[1]).resolve() == staged_source.resolve():
            staged_reads += 1
        return original_read(*args, **kwargs)

    _validated_parent_usd_cli_session_identity.cache_clear()
    monkeypatch.setattr(usd_cli_session, "read_contained_artifact", counted_read)

    first = _parent_usd_cli_session_identity_from_environment()
    second = _parent_usd_cli_session_identity_from_environment()

    assert first == second
    assert staged_reads == 1


def test_render_view_uses_camera_and_requires_ovrtx_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "run"
    project.mkdir()
    session = WorkflowUsdCliSession(
        project_dir=project,
        session_id="workflow-test",
        route=object(),  # type: ignore[arg-type]
        workflow="physics-apply",
    )
    calls: list[list[str]] = []

    def fake_run_json(
        _self: WorkflowUsdCliSession,
        arguments: list[str],
        *,
        timeout_seconds: float = 1800.0,
    ) -> dict[str, Any]:
        del timeout_seconds
        calls.append(arguments)
        if arguments[0] == "render":
            Path(arguments[-1]).write_bytes(b"png")
            return {
                "ok": True,
                "summary": {
                    "backend": "ovrtx",
                    "camera_pos": [1, 2, 3],
                    "camera_dir": [-1, -2, -3],
                    "ovrtx_render_mode": "rt2",
                    "ovrtx_num_sensor_updates": 32,
                    "active_aov": "LdrColor",
                },
            }
        return {"ok": True}

    monkeypatch.setattr(WorkflowUsdCliSession, "run_json", fake_run_json)

    record = session.render_view(
        output_dir=project / "views",
        name="classification_top",
        direction="+z",
        backend="ovrtx",
    )

    assert [call[:2] for call in calls] == [
        ["camera", "fit"],
        ["camera", "orbit"],
        ["render", "--photoreal"],
    ]
    assert calls[-1][2:4] == ["--mode", "quality"]
    assert "--renderer" not in calls[-1]
    assert record["renderer"] == "ovrtx"
    assert Path(record["image_path"]).is_file()
    assert Path(record["camera_json_path"]).is_file()
    assert Path(record["response_path"]).is_file()
    camera = json.loads(Path(record["camera_json_path"]).read_text())
    assert camera["camera_position"] == [1, 2, 3]
    assert camera["camera_view_direction"] == [-1, -2, -3]
    assert camera["renderer"] == "ovrtx"
    assert camera["transport"] == "local"


def test_open_can_explicitly_reload_a_shared_session_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "run"
    project.mkdir()
    source = project / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    session = WorkflowUsdCliSession(
        project_dir=project,
        session_id="workflow-parent",
        route=object(),  # type: ignore[arg-type]
        workflow="asset-material",
    )
    calls: list[list[str]] = []

    def fake_run_json(
        _self: WorkflowUsdCliSession,
        arguments: list[str],
        *,
        timeout_seconds: float = 1800.0,
        allow_not_ok: bool = False,
    ) -> dict[str, Any]:
        del timeout_seconds, allow_not_ok
        calls.append(arguments)
        return {"ok": True}

    monkeypatch.setattr(WorkflowUsdCliSession, "run_json", fake_run_json)

    session.open(source, force_reload=True)

    assert calls == [["open", str(source.resolve()), "--force-reload"]]


def test_render_view_rejects_backend_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "run"
    project.mkdir()
    session = WorkflowUsdCliSession(
        project_dir=project,
        session_id="workflow-test",
        route=object(),  # type: ignore[arg-type]
        workflow="validation-canonical-visual",
    )

    def fake_run_json(
        _self: WorkflowUsdCliSession,
        arguments: list[str],
        *,
        timeout_seconds: float = 1800.0,
    ) -> dict[str, Any]:
        del timeout_seconds
        if arguments[0] == "render":
            Path(arguments[-1]).write_bytes(b"png")
            return {
                "ok": True,
                "summary": {
                    "backend": "ovrtx",
                    "ovrtx_render_mode": "rt2",
                    "ovrtx_num_sensor_updates": 32,
                    "active_aov": "LdrColor",
                },
            }
        return {"ok": True}

    monkeypatch.setattr(WorkflowUsdCliSession, "run_json", fake_run_json)

    with pytest.raises(RuntimeError, match="different OVRTX backend"):
        session.render_view(
            output_dir=project / "views",
            name="classification_top",
            direction="+z",
            backend="remote",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("ovrtx_render_mode", None),
        ("ovrtx_num_sensor_updates", 0),
        ("active_aov", ""),
    ),
)
def test_render_view_rejects_incomplete_executed_ovrtx_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    project = tmp_path / "run"
    project.mkdir()
    session = WorkflowUsdCliSession(
        project_dir=project,
        session_id="workflow-test",
        route=object(),  # type: ignore[arg-type]
        workflow="physics-apply",
    )

    def fake_run_json(
        _self: WorkflowUsdCliSession,
        arguments: list[str],
        *,
        timeout_seconds: float = 1800.0,
    ) -> dict[str, Any]:
        del timeout_seconds
        if arguments[0] != "render":
            return {"ok": True}
        Path(arguments[-1]).write_bytes(b"png")
        summary: dict[str, object] = {
            "backend": "ovrtx",
            "ovrtx_render_mode": "rt2",
            "ovrtx_num_sensor_updates": 32,
            "active_aov": "LdrColor",
        }
        summary[field] = value
        return {"ok": True, "summary": summary}

    monkeypatch.setattr(WorkflowUsdCliSession, "run_json", fake_run_json)

    with pytest.raises(RuntimeError, match="verified OVRTX"):
        session.render_view(
            output_dir=project / "views",
            name="classification_top",
            direction="+z",
        )


def test_render_view_preserves_exact_remote_ovrtx_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "run"
    project.mkdir()
    session = WorkflowUsdCliSession(
        project_dir=project,
        session_id="workflow-test",
        route=object(),  # type: ignore[arg-type]
        workflow="physics-apply",
    )
    identity = {
        "endpoint": "https://renderer.example.test",
        "engine": "ovrtx",
        "protocol_version": 3,
        "status": "ready",
    }

    def fake_run_json(
        _self: WorkflowUsdCliSession,
        arguments: list[str],
        *,
        timeout_seconds: float = 1800.0,
    ) -> dict[str, Any]:
        del timeout_seconds
        if arguments[0] != "render":
            return {"ok": True}
        Path(arguments[-1]).write_bytes(b"png")
        return {
            "ok": True,
            "summary": {
                "backend": "remote",
                "ovrtx_render_mode": "rt2",
                "ovrtx_num_sensor_updates": 32,
                "active_aov": "LdrColor",
            },
            "data": {"results": [{"renderer_identity": identity}]},
        }

    monkeypatch.setattr(WorkflowUsdCliSession, "run_json", fake_run_json)

    record = session.render_view(
        output_dir=project / "views",
        name="classification_top",
        direction="+z",
    )

    assert record["renderer"] == "remote"
    assert record["renderer_identity"] == identity


@pytest.mark.parametrize(
    "identity",
    (
        None,
        {"engine": "ovrtx", "protocol_version": 3, "status": "ready"},
        {
            "endpoint": "https://renderer.example.test",
            "engine": "other",
            "protocol_version": 3,
            "status": "ready",
        },
    ),
)
def test_remote_render_metadata_requires_exact_ovrtx_identity(
    identity: dict[str, object] | None,
) -> None:
    result = {} if identity is None else {"renderer_identity": identity}

    with pytest.raises(RuntimeError, match="exact OVRTX service identity"):
        validated_ovrtx_render_metadata(
            {
                "summary": {
                    "backend": "remote",
                    "ovrtx_render_mode": "rt2",
                    "ovrtx_num_sensor_updates": 32,
                    "active_aov": "LdrColor",
                },
                "data": {"results": [result]},
            }
        )


def test_render_view_rejects_non_ovrtx_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "run"
    project.mkdir()
    session = WorkflowUsdCliSession(
        project_dir=project,
        session_id="workflow-test",
        route=object(),  # type: ignore[arg-type]
        workflow="physics-apply",
    )

    def fake_run_json(
        _self: WorkflowUsdCliSession,
        arguments: list[str],
        *,
        timeout_seconds: float = 1800.0,
    ) -> dict[str, Any]:
        del timeout_seconds
        if arguments[0] == "render":
            Path(arguments[-1]).write_bytes(b"png")
            return {"ok": True, "summary": {"backend": "usdrecord"}}
        return {"ok": True}

    monkeypatch.setattr(WorkflowUsdCliSession, "run_json", fake_run_json)

    with pytest.raises(RuntimeError, match="verified OVRTX"):
        session.render_view(
            output_dir=project / "views",
            name="classification_top",
            direction="+z",
        )


def test_physics_apply_reuses_caller_owned_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    source = _write_physics_source(
        tmp_path / "source.usda",
        instanceable=True,
    )
    output = run_dir / "physics.usda"
    session = WorkflowUsdCliSession(
        project_dir=run_dir,
        session_id="workflow-shared",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel"),
        workflow="content-workflow-cli",
    )
    calls: list[tuple[list[str], bool]] = []

    def fake_run_json(
        _self: WorkflowUsdCliSession,
        arguments: list[str],
        *,
        timeout_seconds: float = 1800.0,
        allow_not_ok: bool = False,
    ) -> dict[str, Any]:
        del timeout_seconds
        calls.append((arguments, allow_not_ok))
        if arguments[0] == "save":
            Path(arguments[1]).write_text("#usda 1.0\n", encoding="utf-8")
        return {"ok": arguments[:2] != ["physics", "validate"]}

    def fake_run_json_result(
        _self: WorkflowUsdCliSession,
        arguments: list[str],
        *,
        timeout_seconds: float = 1800.0,
        allow_not_ok: bool = False,
    ) -> WorkflowUsdCliCommandResult:
        return WorkflowUsdCliCommandResult(
            payload=fake_run_json(
                _self,
                arguments,
                timeout_seconds=timeout_seconds,
                allow_not_ok=allow_not_ok,
            ),
            returncode=2,
        )

    monkeypatch.setattr(WorkflowUsdCliSession, "run_json", fake_run_json)
    monkeypatch.setattr(
        WorkflowUsdCliSession,
        "run_json_result",
        fake_run_json_result,
    )
    monkeypatch.setattr(usd_cli_ops, "_require_ovrtx_probe", lambda *_args: None)
    monkeypatch.setattr(
        usd_cli_ops,
        "_create_owned_session",
        lambda **_kwargs: pytest.fail(
            "a supplied session must not create an owned sidecar"
        ),
    )

    result = usd_cli_ops.apply_physics_patch(
        source_usd=source,
        output_usd=output,
        raw_dir=raw_dir,
        workflow_decisions=[
            {
                "component_id": "body",
                "mass_authoring_path": "/World/Body",
                "collider_paths": ["/World/Body/Mesh"],
                "collision_approximation": "convexHull",
                "physical_properties": {"density": 1000.0},
            }
        ],
        author_rigid_body=True,
        physics_scene_path="/World/PhysicsScene",
        usd_cli_session=session,
    )

    assert result["session_id"] == "workflow-shared"
    assert result["reused_workflow_session"] is True
    assert result["deinstanced_roots"] == ["/World/Body"]
    assert (raw_dir / "workflow_physics_patch.json").is_file()
    assert [arguments[:2] for arguments, _allow_not_ok in calls] == [
        ["render-probe", "--require-engine"],
        ["open", str(source.resolve())],
        ["checkpoint", "save"],
        ["set", "/World/Body"],
        ["physics", "apply"],
        ["physics", "validate"],
        ["checkpoint", "save"],
        ["save", str(output.resolve())],
    ]
    assert calls[1][0] == ["open", str(source.resolve()), "--force-reload"]
    assert calls[5][1] is True


def test_physics_render_reuses_caller_owned_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    recording = run_dir / "recording.usda"
    recording.write_text("#usda 1.0\n", encoding="utf-8")
    session = WorkflowUsdCliSession(
        project_dir=run_dir,
        session_id="workflow-shared",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel"),
        workflow="content-workflow-cli",
    )
    calls: list[list[str]] = []

    def fake_run_json(
        _self: WorkflowUsdCliSession,
        arguments: list[str],
        *,
        timeout_seconds: float = 1800.0,
        allow_not_ok: bool = False,
    ) -> dict[str, Any]:
        del timeout_seconds, allow_not_ok
        calls.append(arguments)
        if arguments[0] == "render-frames":
            return {
                "ok": True,
                "summary": {"backend": "ovrtx"},
                "data": {
                    "frame_paths": ["frame.png"],
                    "renderer_identities": [{"unused": "local"}],
                },
            }
        if arguments[0] == "render-probe":
            return {"ok": True, "resolved_renderer": "ovrtx"}
        return {"ok": True}

    monkeypatch.setattr(WorkflowUsdCliSession, "run_json", fake_run_json)
    monkeypatch.setattr(usd_cli_ops, "_require_ovrtx_probe", lambda *_args: None)
    monkeypatch.setattr(
        usd_cli_ops,
        "read_contained_artifact",
        lambda root, path, **_kwargs: SimpleNamespace(
            path=Path(root) / path,
            size_bytes=1,
        ),
    )
    monkeypatch.setattr(
        usd_cli_ops,
        "_create_owned_session",
        lambda **_kwargs: pytest.fail(
            "a supplied session must not create an owned sidecar"
        ),
    )

    frames, record = usd_cli_ops.render_physics_frames(
        recording_usd=recording,
        output_dir=run_dir / "frames",
        raw_dir=raw_dir,
        focus_prim_path="/World/Body",
        usd_cli_session=session,
    )

    assert frames == [str(run_dir / "frames" / "frame.png")]
    assert record["reused_workflow_session"] is True
    assert record["focus_prim_path"] == "/World/Body"
    assert calls[-1][calls[-1].index("--focus") + 1] == "/World/Body"
    assert [arguments[0] for arguments in calls] == [
        "open",
        "render-probe",
        "render-frames",
    ]


def test_physics_apply_owns_one_shared_session_when_none_is_supplied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    source = _write_physics_source(tmp_path / "source.usda")
    output = run_dir / "physics.usda"
    created: list[dict[str, object]] = []
    closed: list[str] = []
    calls: list[list[str]] = []

    class FakeSession:
        project_dir = run_dir
        session_id = "physics-owned"
        route = SimpleNamespace(wrapper=tmp_path / "usd-cli-tel")

        @classmethod
        def create(cls, **kwargs: object) -> FakeSession:
            created.append(kwargs)
            return cls()

        def run_json(
            self,
            arguments: list[str],
            *,
            timeout_seconds: float = 1800.0,
            allow_not_ok: bool = False,
        ) -> dict[str, object]:
            del timeout_seconds
            calls.append(arguments)
            if arguments[0] == "save":
                Path(arguments[1]).write_text("#usda 1.0\n", encoding="utf-8")
            return {
                "ok": not (arguments[:2] == ["physics", "validate"] and allow_not_ok)
            }

        def run_json_result(
            self,
            arguments: list[str],
            *,
            timeout_seconds: float = 1800.0,
            allow_not_ok: bool = False,
        ) -> WorkflowUsdCliCommandResult:
            return WorkflowUsdCliCommandResult(
                payload=self.run_json(
                    arguments,
                    timeout_seconds=timeout_seconds,
                    allow_not_ok=allow_not_ok,
                ),
                returncode=2,
            )

        def close(self) -> None:
            closed.append(self.session_id)

    monkeypatch.setattr(usd_cli_ops, "WorkflowUsdCliSession", FakeSession)
    monkeypatch.setattr(usd_cli_ops, "_require_ovrtx_probe", lambda *_args: None)

    result = usd_cli_ops.apply_physics_patch(
        source_usd=source,
        output_usd=output,
        raw_dir=raw_dir,
        workflow_decisions=[
            {
                "component_id": "body",
                "mass_authoring_path": "/World/Body",
                "collider_paths": ["/World/Body/Mesh"],
                "collision_approximation": "convexHull",
                "physical_properties": {"density": 1000.0},
            }
        ],
        author_rigid_body=True,
        physics_scene_path="/World/PhysicsScene",
    )

    assert len(created) == 1
    assert created[0]["owner_root"] == run_dir
    assert created[0]["project_dir"] == run_dir
    assert created[0]["input_roots"] == (source.resolve(),)
    assert closed == ["physics-owned"]
    assert result["session_id"] == "physics-owned"
    assert result["reused_workflow_session"] is False
    assert [arguments[0] for arguments in calls] == [
        "render-probe",
        "open",
        "checkpoint",
        "physics",
        "physics",
        "checkpoint",
        "save",
    ]


def test_run_json_records_structured_receipt_and_contained_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "renders" / "view.png"
    artifact.parent.mkdir()
    artifact_link = tmp_path / "renders" / "view-link.png"
    source = tmp_path / "source.usda"
    source.write_bytes(b"#usda 1.0\n")
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(
            wrapper=tmp_path / "usd-cli-tel", target=tmp_path / "usd-cli"
        ),
        workflow="test-workflow",
        _render_credentials={
            "remote_api_key": "project-key",
            "backend_api_keys": {
                "https://ovrtx.example.test": "primary-key",
                "https://ovrtx-backup.example.test": "backup-key",
            },
        },
    )
    captured_environment: dict[str, str] = {}
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {},
    )

    def fake_subprocess(*_args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured_environment.update(kwargs["env"])
        artifact.write_bytes(b"ovrtx-image")
        _symlink_or_skip(artifact_link, artifact)
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "summary": {"backend": "ovrtx"},
                    "data": {
                        "path": str(artifact),
                        "untrusted_symlink": str(artifact_link),
                        "echoed_token": "top-secret",
                    },
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.run_bounded_usd_cli_subprocess",
        fake_subprocess,
    )

    response = session.run_json(
        [
            "render",
            str(source),
            "--output",
            str(artifact),
            "--token",
            "top-secret",
        ]
    )

    assert response["ok"] is True
    assert captured_environment["USD_CLI_LOCK_RENDER_CONFIG"] == "1"
    assert captured_environment["USD_CLI_RENDER_REMOTE_API_KEY"] == "project-key"
    assert json.loads(captured_environment["USD_CLI_RENDER_BACKEND_API_KEYS_JSON"]) == {
        "https://ovrtx.example.test": "primary-key",
        "https://ovrtx-backup.example.test": "backup-key",
    }
    receipt = json.loads(session.receipt_file.read_text(encoding="utf-8"))
    assert receipt["schema_version"].endswith("usd-cli-command-receipt.v1")
    assert receipt["arguments"][-1] == "<redacted>"
    assert receipt["status"] == "completed"
    assert receipt["response"]["summary"]["backend"] == "ovrtx"
    assert receipt["response"]["data"]["echoed_token"] == "<redacted>"
    assert "top-secret" not in receipt["stdout"]
    assert receipt["inputs"] == [
        {
            "path": str(source.resolve()),
            "sha256": hashlib.sha256(b"#usda 1.0\n").hexdigest(),
            "size_bytes": len(b"#usda 1.0\n"),
        }
    ]
    assert receipt["artifacts"] == [
        {
            "path": "renders/view.png",
            "sha256": hashlib.sha256(b"ovrtx-image").hexdigest(),
            "size_bytes": len(b"ovrtx-image"),
        }
    ]


def test_session_run_json_pins_uv_outside_sanitized_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="texture-generate",
    )
    uv_executable = tmp_path / "user-local" / "uv.exe"
    uv_executable.parent.mkdir()
    uv_executable.write_text("uv", encoding="utf-8")
    monkeypatch.setenv("PATH", str(uv_executable.parent))
    monkeypatch.delenv("USD_CLI_UV_EXECUTABLE", raising=False)
    monkeypatch.setattr(
        usd_cli_session,
        "controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {"PATH": os.defpath},
    )
    monkeypatch.setattr(
        usd_cli_session.shutil,
        "which",
        lambda name, *, path: str(uv_executable) if name == "uv" else None,
    )
    captured_environment: dict[str, str] = {}

    def fake_subprocess(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        captured_environment.update(kwargs["env"])
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps({"ok": True}), stderr=""
        )

    monkeypatch.setattr(
        usd_cli_session, "run_bounded_usd_cli_subprocess", fake_subprocess
    )

    session.run_json(["info"])

    assert captured_environment["USD_CLI_UV_EXECUTABLE"] == str(uv_executable.resolve())
    assert str(uv_executable.parent) not in captured_environment["PATH"].split(
        os.pathsep
    )


def test_session_require_ovrtx_retries_background_provisioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="texture-generate",
    )
    attempts: list[list[str]] = []
    sleeps: list[float] = []

    def fake_run_json(
        _self: WorkflowUsdCliSession,
        arguments: list[str],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        attempts.append(arguments)
        if len(attempts) == 1:
            raise RuntimeError("ovrtx auto-install STARTED in the background")
        if len(attempts) == 2:
            raise RuntimeError("ovrtx auto-install in progress (5s elapsed)")
        return {"ready": True, "render": {"backend": "ovrtx"}}

    monkeypatch.setattr(WorkflowUsdCliSession, "run_json", fake_run_json)
    monkeypatch.setattr(usd_cli_session.time, "sleep", sleeps.append)

    probe = session.require_ovrtx(tmp_path / "probe")

    assert probe["ready"] is True
    assert len(attempts) == 3
    assert sleeps == [
        usd_cli_session._OVRTX_PROVISIONING_POLL_SECONDS,
        usd_cli_session._OVRTX_PROVISIONING_POLL_SECONDS,
    ]


def test_run_json_result_preserves_allowed_failure_exit_code_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="physics-operations",
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.run_bounded_usd_cli_subprocess",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=7,
            stdout=json.dumps({"ok": False, "issues": ["invalid physics"]}),
            stderr="validation failed",
        ),
    )

    result = session.run_json_result(
        ["physics", "validate"],
        allow_not_ok=True,
    )

    assert result == WorkflowUsdCliCommandResult(
        payload={"ok": False, "issues": ["invalid physics"]},
        returncode=7,
    )
    receipt = json.loads(session.receipt_file.read_text(encoding="utf-8"))
    assert receipt["returncode"] == 7
    assert receipt["status"] == "failed"
    assert receipt["response"] == result.payload


def test_run_json_result_enforces_force_reload_on_direct_open_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parent-attached session must force-reload every ``open``, including
    argv built directly by callers instead of through ``session.open()`` —
    a plain re-open of a path holding unsaved sibling edits is refused by
    the same-session reload guard."""
    scene = tmp_path / "scene.usda"
    scene.write_text("#usda 1.0\n", encoding="utf-8")
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="physics-operations",
        _force_reload_on_every_open=True,
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {},
    )
    commands: list[list[str]] = []

    def fake_subprocess(
        command: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps({"ok": True}),
            stderr="",
        )

    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.run_bounded_usd_cli_subprocess",
        fake_subprocess,
    )

    session.run_json(["open", str(scene)])
    assert commands[-1][-1] == "--force-reload"

    # Idempotent when the caller already passed the flag, and other commands
    # stay untouched.
    session.run_json(["open", str(scene), "--force-reload"])
    assert commands[-1].count("--force-reload") == 1
    session.run_json(["physics", "validate"])
    assert "--force-reload" not in commands[-1]


def test_physics_allow_not_ok_evidence_uses_command_result_exit_code(
    tmp_path: Path,
) -> None:
    class Session:
        session_id = "workflow-physics"
        project_dir = tmp_path
        route = SimpleNamespace(wrapper=tmp_path / "custom-usd-cli-wrapper")

        def run_json_result(
            self,
            arguments: list[str],
            *,
            timeout_seconds: float,
            allow_not_ok: bool,
        ) -> WorkflowUsdCliCommandResult:
            assert arguments == ["physics", "validate"]
            assert timeout_seconds == 30.0
            assert allow_not_ok is True
            return WorkflowUsdCliCommandResult(
                payload={"ok": False, "issues": ["invalid physics"]},
                returncode=7,
            )

    payload = usd_cli_ops._run_json(
        project_dir=tmp_path,
        session_id="workflow-physics",
        arguments=["physics", "validate"],
        timeout_seconds=30.0,
        allow_not_ok=True,
        workflow_session=Session(),  # type: ignore[arg-type]
    )

    assert payload["_workflow_command"] == {
        "argv": [
            str(tmp_path / "custom-usd-cli-wrapper"),
            "--json",
            "--session",
            "workflow-physics",
            "physics",
            "validate",
        ],
        "returncode": 7,
    }


def test_run_json_records_structured_timeout_and_redacts_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(
            wrapper=tmp_path / "usd-cli-tel",
            target=tmp_path / "usd-cli",
            source_revision="tree-digest",
        ),
        workflow="test-workflow",
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {},
    )

    def time_out(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(
            cmd="usd-cli",
            timeout=1.0,
            output="partial top-secret",
            stderr="timed out with top-secret",
        )

    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.run_bounded_usd_cli_subprocess",
        time_out,
    )

    with pytest.raises(RuntimeError, match="<redacted>") as error:
        session.run_json(["render", "--token", "top-secret"], timeout_seconds=1.0)

    assert "top-secret" not in str(error.value)
    receipt = json.loads(session.receipt_file.read_text(encoding="utf-8"))
    assert receipt["status"] == "timed_out"
    assert receipt["returncode"] is None
    assert receipt["tool"]["source_revision"] == "tree-digest"
    assert receipt["stdout"] == "partial <redacted>"
    assert receipt["stderr"] == "timed out with <redacted>"


@pytest.mark.parametrize(
    "arguments,match",
    [
        (["save"], "requires an explicit output path"),
        (["render", "--output"], "requires a value"),
        (["render", "--output-dir="], "require non-empty paths"),
        (["save", "../source.usda"], "escapes workflow project"),
        (
            ["render", "--output", "renders/view.png", "--output", "../view.png"],
            "escapes workflow project",
        ),
    ],
)
def test_run_json_rejects_unconfined_outputs_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    match: str,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.run_bounded_usd_cli_subprocess",
        lambda *_args, **_kwargs: pytest.fail("unconfined command must not launch"),
    )

    with pytest.raises(RuntimeError, match=match):
        session.run_json(arguments)


@pytest.mark.parametrize(
    ("arguments", "expected_tail"),
    [
        (
            ["render", "--output", "renders/view.png"],
            ["render", "--output", "renders/view.png"],
        ),
        (
            ["render", "--output=renders/view.png"],
            ["render", "--output=renders/view.png"],
        ),
        (
            ["render", "-o", "renders/view.png"],
            ["render", "-o", "renders/view.png"],
        ),
        (
            ["render-probe", "--output-dir", "probe"],
            ["render-probe", "--output-dir", "probe"],
        ),
        (["save", "saved.usda"], ["save", "saved.usda"]),
        (["export", "glb", "saved.glb"], ["export", "glb", "saved.glb"]),
        (
            ["convert", "scene.usda", "converted.usda"],
            ["convert", "scene.usda", "converted.usda"],
        ),
    ],
)
def test_attached_session_absolutizes_relative_output_before_daemon_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    expected_tail: list[str],
) -> None:
    project = tmp_path / "domain"
    project.mkdir()
    _write_daemon_server_state(tmp_path)
    (project / "scene.usda").write_text("#usda 1.0\n", encoding="utf-8")
    route = SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path)
    session = WorkflowUsdCliSession(
        project_dir=project,
        session_id="workflow-parent",
        route=route,
        workflow="asset-articulation",
        daemon_project_dir=tmp_path,
        daemon_server_url="http://127.0.0.1:4567",
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {},
    )

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"ok": True}),
            stderr="",
        )

    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.run_bounded_usd_cli_subprocess",
        run,
    )

    session.run_json(arguments)

    expected = list(expected_tail)
    for index, value in enumerate(expected):
        if value.startswith("--output="):
            expected[index] = f"--output={project / value.partition('=')[2]}"
            continue
        if value in {
            "renders/view.png",
            "saved.usda",
            "saved.glb",
            "converted.usda",
            "scene.usda",
            "probe",
        }:
            expected[index] = str(project / value)
    assert calls[0][-len(expected) :] == expected


@pytest.mark.parametrize(
    ("arguments", "expected_tail", "expected_input"),
    [
        (
            ["open", "scene.usda"],
            ["open", "{project}/scene.usda"],
            "scene.usda",
        ),
        (
            ["material", "/World/Panel", "--library=library.usda"],
            ["material", "/World/Panel", "--library={project}/library.usda"],
            "library.usda",
        ),
        (
            ["material", "/World/Panel", "--mdl=OmniPBR.mdl"],
            ["material", "/World/Panel", "--mdl=OmniPBR.mdl"],
            None,
        ),
        (
            ["material", "/World/Panel", "--mdl", "mdl/Custom.mdl"],
            ["material", "/World/Panel", "--mdl", "mdl/Custom.mdl"],
            "mdl/Custom.mdl",
        ),
        (
            ["material", "/World/Panel", "--diffuse-tex", "textures/diffuse.png"],
            ["material", "/World/Panel", "--diffuse-tex", "textures/diffuse.png"],
            "textures/diffuse.png",
        ),
        (
            ["material", "/World/Panel", "--normal-tex=textures/normal.png"],
            ["material", "/World/Panel", "--normal-tex=textures/normal.png"],
            "textures/normal.png",
        ),
        (
            ["material", "/World/Panel", "--orm-tex=textures/orm.png"],
            ["material", "/World/Panel", "--orm-tex=textures/orm.png"],
            "textures/orm.png",
        ),
        (
            ["material", "/World/Panel", "--roughness-tex", "textures/rough.png"],
            ["material", "/World/Panel", "--roughness-tex", "textures/rough.png"],
            "textures/rough.png",
        ),
        (
            ["material", "/World/Panel", "--metallic-tex=textures/metal.png"],
            ["material", "/World/Panel", "--metallic-tex=textures/metal.png"],
            "textures/metal.png",
        ),
        (
            ["import", "mesh.usda"],
            ["import", "mesh.usda"],
            "mesh.usda",
        ),
        (
            ["physics", "apply", "--file", "physics.json"],
            ["physics", "apply", "--file", "{project}/physics.json"],
            "physics.json",
        ),
        (
            ["physics", "apply", "-f=physics.json"],
            ["physics", "apply", "-f={project}/physics.json"],
            "physics.json",
        ),
        (
            ["physics", "simulate", "--scene", "scene.usda"],
            ["physics", "simulate", "--scene", "{project}/scene.usda"],
            "scene.usda",
        ),
        (
            ["render", "--against=baseline.png"],
            ["render", "--against={project}/baseline.png"],
            "baseline.png",
        ),
        (
            ["render-frames", "--scene", "scene.usda"],
            ["render-frames", "--scene", "{project}/scene.usda"],
            "scene.usda",
        ),
    ],
)
def test_attached_session_rewrites_only_daemon_read_input_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    expected_tail: list[str],
    expected_input: str | None,
) -> None:
    project = tmp_path / "domain"
    project.mkdir()
    _write_daemon_server_state(tmp_path)
    for filename in {
        value
        for value in (
            expected_input,
            "scene.usda",
        )
        if value is not None
    }:
        input_path = project / filename
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text("input\n", encoding="utf-8")
    session = WorkflowUsdCliSession(
        project_dir=project,
        session_id="workflow-parent",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="asset-articulation",
        daemon_project_dir=tmp_path,
        daemon_server_url="http://127.0.0.1:4567",
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {},
    )

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"ok": True}),
            stderr="",
        )

    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.run_bounded_usd_cli_subprocess",
        run,
    )

    session.run_json(arguments)

    expected = [
        value.replace("{project}/", f"{project}{os.sep}") for value in expected_tail
    ]
    assert calls[0][-len(expected) :] == expected
    receipt = json.loads(session.receipt_file.read_text(encoding="utf-8"))
    assert receipt["arguments"] == expected
    assert [item["path"] for item in receipt["inputs"]] == (
        [str((project / expected_input).resolve())]
        if expected_input is not None
        else []
    )


def test_attached_session_does_not_rewrite_non_path_operand_file_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "domain"
    project.mkdir()
    _write_daemon_server_state(tmp_path)
    (project / "plastic").write_text("not an operand path\n", encoding="utf-8")
    session = WorkflowUsdCliSession(
        project_dir=project,
        session_id="workflow-parent",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="asset-material",
        daemon_project_dir=tmp_path,
        daemon_server_url="http://127.0.0.1:4567",
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {},
    )

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"ok": True}),
            stderr="",
        )

    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.run_bounded_usd_cli_subprocess",
        run,
    )

    session.run_json(["material", "plastic", "--name", "plastic"])

    assert calls[0][-4:] == ["material", "plastic", "--name", "plastic"]


@pytest.mark.parametrize(
    ("consumer", "relative_probe", "relative_render"),
    [
        (
            "material-appearance",
            "ovrtx_probe",
            "swatches/material.png",
        ),
        (
            "texture-validation",
            "ovrtx_probe",
            "unit-000/member-000-view-00.png",
        ),
        (
            "articulation-evidence",
            "ovrtx_probe",
            "collection-000/candidate-000/render-000/image.png",
        ),
    ],
)
def test_consumer_project_roots_confine_probe_and_render_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
    relative_probe: str,
    relative_render: str,
) -> None:
    """Exercise real adapter confinement without launching a usd-cli daemon."""

    project = tmp_path / consumer
    project.mkdir()
    session = WorkflowUsdCliSession(
        project_dir=project,
        session_id=f"workflow-{consumer}",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow=consumer,
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {},
    )

    def complete(
        command: list[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        payload = (
            {"schema_version": "usd-cli.render-probe.v1", "ready": True}
            if "render-probe" in command
            else {"ok": True}
        )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.run_bounded_usd_cli_subprocess",
        complete,
    )

    session.run_json(
        [
            "render-probe",
            "--output-dir",
            str(project / relative_probe),
        ]
    )
    session.run_json(
        [
            "render",
            "--output",
            str(project / relative_render),
        ]
    )

    assert len(calls) == 2


def test_run_json_rejects_output_symlink_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    _symlink_or_skip(
        tmp_path / "renders",
        outside,
        target_is_directory=True,
    )
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )
    monkeypatch.setattr(
        "content_agent_workflows.common.usd_cli_session.controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {},
    )

    with pytest.raises(RuntimeError, match="escapes workflow project|symlink"):
        session.run_json(["render", "--output", "renders/view.png"])


def test_session_rejects_symlinked_raw_artifact_directory(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-raw"
    outside.mkdir()
    _symlink_or_skip(
        tmp_path / "raw",
        outside,
        target_is_directory=True,
    )
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )

    with pytest.raises(OSError):
        _ = session.telemetry_file

    assert not (outside / "usd_cli_telemetry.jsonl").exists()
    assert not (outside / "usd_cli_command_receipts.jsonl").exists()


def test_session_detects_renamed_raw_directory_replacement(tmp_path: Path) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )
    raw = session.prepare_raw_directory()
    original = tmp_path / "original-raw"
    raw.rename(original)
    raw.mkdir(mode=0o700)
    os.chmod(raw, 0o700)

    with pytest.raises(RuntimeError, match="changed during the session"):
        _ = session.telemetry_file


def _append_test_receipt(session: WorkflowUsdCliSession, operation_id: str) -> None:
    session._append_receipt(
        operation_id=operation_id,
        started_ns=1,
        arguments=["query", "tree"],
        returncode=0,
        stdout='{"ok":true}',
        stderr="",
        response={"ok": True},
        status="completed",
        input_files=[],
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows receipt journal creation")
def test_windows_receipt_journal_first_creation_is_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )
    original_open = usd_cli_session.open_confined_lock_file
    exclusive_attempts: list[bool] = []

    def record_open(root: object, relative_key: str, **kwargs: Any):
        exclusive_attempts.append(bool(kwargs.get("exclusive_create")))
        return original_open(root, relative_key, **kwargs)

    monkeypatch.setattr(usd_cli_session, "open_confined_lock_file", record_open)

    _append_test_receipt(session, "operation-one")

    assert exclusive_attempts == [True]
    assert session.receipt_file.is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows receipt journal creation")
def test_windows_receipt_journal_exclusive_create_race_uses_existing_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )
    original_open = usd_cli_session.open_confined_lock_file
    exclusive_attempts: list[bool] = []
    competing_payload = b'{"competing":true}\n'

    def race_open(root: object, relative_key: str, **kwargs: Any):
        exclusive = bool(kwargs.get("exclusive_create"))
        exclusive_attempts.append(exclusive)
        if exclusive:
            with original_open(root, relative_key, **kwargs) as descriptor:
                assert os.write(descriptor, competing_payload) == len(competing_payload)
                os.fsync(descriptor)
        return original_open(root, relative_key, **kwargs)

    monkeypatch.setattr(usd_cli_session, "open_confined_lock_file", race_open)

    with pytest.raises(
        RuntimeError,
        match="without its sealed checkpoint digest|single-link regular file",
    ):
        _append_test_receipt(session, "operation-one")

    assert exclusive_attempts == [True, False]
    assert (
        tmp_path / "raw" / "usd_cli_command_receipts.jsonl"
    ).read_bytes() == competing_payload


def test_session_freezes_public_identity_while_receipt_state_evolves(
    tmp_path: Path,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=UsdCliPackageRoute(
            wrapper=tmp_path / "usd-cli-tel",
            target=tmp_path / "usd-cli",
            source_root=tmp_path,
            source_revision="test-revision",
        ),
        workflow="test-workflow",
    )
    identity_hash = hash(session)

    _append_test_receipt(session, "operation-one")

    assert hash(session) == identity_hash
    assert session.receipt_file.is_file()
    assert session._receipt_state["receipt_size"] > 0
    with pytest.raises(FrozenInstanceError):
        session.session_id = "different-session"  # type: ignore[misc]


def test_session_fsyncs_receipt_before_pinning_and_checkpointing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )
    events: list[str] = []
    original_fsync = os.fsync
    original_checkpoint = WorkflowUsdCliSession._write_receipt_checkpoint
    receipt_path = tmp_path / "raw" / "usd_cli_command_receipts.jsonl"

    def record_fsync(descriptor: int) -> None:
        if _descriptor_matches_path(descriptor, receipt_path):
            events.append("receipt_fsync")
        original_fsync(descriptor)

    def record_checkpoint(
        self: WorkflowUsdCliSession,
        receipt_stat: os.stat_result,
    ) -> None:
        events.append("checkpoint")
        assert events == ["receipt_fsync", "checkpoint"]
        original_checkpoint(self, receipt_stat)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(
        WorkflowUsdCliSession,
        "_write_receipt_checkpoint",
        record_checkpoint,
    )

    _append_test_receipt(session, "operation-one")

    assert events == ["receipt_fsync", "checkpoint"]


def test_session_propagates_receipt_fsync_failure_without_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )
    original_fsync = os.fsync
    receipt_path = tmp_path / "raw" / "usd_cli_command_receipts.jsonl"

    def fail_receipt_fsync(descriptor: int) -> None:
        if _descriptor_matches_path(descriptor, receipt_path):
            raise OSError("simulated receipt sync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_receipt_fsync)

    with pytest.raises(OSError, match="simulated receipt sync failure"):
        _append_test_receipt(session, "operation-one")

    assert not (tmp_path / "raw" / "usd_cli_command_receipts.checkpoint.json").exists()
    assert session._receipt_state["receipt_size"] == 0


def test_session_rejects_precreated_receipt_hardlink(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(mode=0o700)
    os.chmod(raw, 0o700)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("attacker-controlled\n", encoding="utf-8")
    os.chmod(outside, 0o600)
    os.link(outside, raw / "usd_cli_command_receipts.jsonl")
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )

    with pytest.raises(
        RuntimeError,
        match="without its sealed checkpoint digest|single-link regular file",
    ):
        _append_test_receipt(session, "operation-one")

    assert outside.read_text(encoding="utf-8") == "attacker-controlled\n"


def test_session_rejects_unsealed_preexisting_receipt_with_safe_recovery(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(mode=0o700)
    os.chmod(raw, 0o700)
    receipt = raw / "usd_cli_command_receipts.jsonl"
    receipt.write_text('{"interrupted":true}\n', encoding="utf-8")
    os.chmod(receipt, 0o600)
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "start a new workflow project directory|"
            "without its sealed checkpoint digest"
        ),
    ):
        _append_test_receipt(session, "operation-one")

    assert receipt.read_text(encoding="utf-8") == '{"interrupted":true}\n'


def test_session_detects_receipt_rewrite_between_appends(tmp_path: Path) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )
    _append_test_receipt(session, "operation-one")
    receipt = tmp_path / "raw" / "usd_cli_command_receipts.jsonl"
    receipt.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="replaced or modified"):
        _append_test_receipt(session, "operation-two")


def test_session_explicit_integrity_check_detects_receipt_rewrite(
    tmp_path: Path,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )
    _append_test_receipt(session, "operation-one")
    receipt = tmp_path / "raw" / "usd_cli_command_receipts.jsonl"
    receipt.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="replaced or modified"):
        session.verify_receipt_journal_integrity()


@pytest.mark.parametrize("tamper_mode", ["remove", "replace"])
def test_session_explicit_integrity_check_detects_checkpoint_tampering(
    tmp_path: Path,
    tamper_mode: str,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )
    _append_test_receipt(session, "operation-one")
    checkpoint = tmp_path / "raw" / "usd_cli_command_receipts.checkpoint.json"
    if tamper_mode == "remove":
        checkpoint.unlink()
    else:
        checkpoint.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="usd-cli receipt checkpoint"):
        session.verify_receipt_journal_integrity()


def test_session_integrity_check_accepts_setup_before_first_command(
    tmp_path: Path,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )

    session.verify_receipt_journal_integrity()

    assert not (tmp_path / "raw" / "usd_cli_command_receipts.jsonl").exists()


def test_session_resumes_receipts_from_digest_bound_checkpoint(tmp_path: Path) -> None:
    route = SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path)
    first = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=route,
        workflow="test-workflow",
    )
    _append_test_receipt(first, "operation-one")
    checkpoint = first.receipt_checkpoint_file
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    resumed = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=route,
        workflow="test-workflow",
        receipt_checkpoint_sha256=checkpoint_sha256,
    )
    _append_test_receipt(resumed, "operation-two")

    receipts = [
        json.loads(line)
        for line in resumed.receipt_file.read_text(encoding="utf-8").splitlines()
    ]
    assert [receipt["operation_id"] for receipt in receipts] == [
        "operation-one",
        "operation-two",
    ]


def test_session_refuses_to_recreate_journal_sealed_by_checkpoint(
    tmp_path: Path,
) -> None:
    route = SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path)
    first = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=route,
        workflow="test-workflow",
    )
    _append_test_receipt(first, "operation-one")
    checkpoint = first.receipt_checkpoint_file
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    first.receipt_file.unlink()

    resumed = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=route,
        workflow="test-workflow",
        receipt_checkpoint_sha256=checkpoint_sha256,
    )
    with pytest.raises(
        RuntimeError,
        match="receipt journal sealed by the checkpoint is missing",
    ):
        _append_test_receipt(resumed, "operation-two")

    assert checkpoint.read_bytes() == checkpoint_bytes
    assert not (tmp_path / "raw" / "usd_cli_command_receipts.jsonl").exists()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows denies a second writer while the receipt descriptor is held",
)
def test_session_detects_rewrite_during_receipt_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )
    _append_test_receipt(session, "operation-one")
    receipt = tmp_path / "raw" / "usd_cli_command_receipts.jsonl"
    original_write = os.write
    tampered = False

    def rewrite_before_append(descriptor: int, data: bytes) -> int:
        nonlocal tampered
        if not tampered and _descriptor_matches_path(descriptor, receipt):
            existing = receipt.read_bytes()
            receipt.write_bytes(b"X" + existing[1:])
            tampered = True
        return original_write(descriptor, data)

    monkeypatch.setattr(os, "write", rewrite_before_append)

    with pytest.raises(RuntimeError, match="changed concurrently during append"):
        _append_test_receipt(session, "operation-two")


def test_session_detects_receipt_hardlink_added_after_creation(tmp_path: Path) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )
    _append_test_receipt(session, "operation-one")
    receipt = tmp_path / "raw" / "usd_cli_command_receipts.jsonl"
    os.link(receipt, tmp_path / "stolen-receipts.jsonl")

    with pytest.raises(RuntimeError, match="singly linked"):
        _ = session.receipt_file


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced")
def test_session_rejects_child_created_permissive_raw_directory(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(mode=0o755)
    os.chmod(raw, 0o755)
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )

    with pytest.raises(RuntimeError, match="owner-controlled 0700"):
        _ = session.telemetry_file


def test_session_normalizes_setgid_bit_inherited_by_new_raw_directory(
    tmp_path: Path,
) -> None:
    os.chmod(tmp_path, 0o2770)
    probe = tmp_path / "setgid_probe"
    probe.mkdir(mode=0o700)
    if not stat.S_IMODE(probe.stat().st_mode) & stat.S_ISGID:
        pytest.skip("filesystem does not propagate setgid to child directories")
    probe.rmdir()
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )

    assert session.telemetry_file == tmp_path / "raw" / "usd_cli_telemetry.jsonl"
    assert stat.S_IMODE((tmp_path / "raw").stat().st_mode) == 0o700


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced")
def test_session_rejects_new_raw_directory_made_group_writable_during_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_mkdir = os.mkdir

    def compromised_mkdir(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        real_mkdir(path, mode=mode, dir_fd=dir_fd)
        if path == "raw":
            os.chmod(tmp_path / "raw", 0o2770)

    monkeypatch.setattr(os, "mkdir", compromised_mkdir)
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )

    with pytest.raises(RuntimeError, match="owner-controlled 0700"):
        session.prepare_raw_directory()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced")
def test_session_rehardens_identity_pinned_collaborative_raw_directory(
    tmp_path: Path,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )
    raw = session.prepare_raw_directory()
    pinned_identity = (raw.stat().st_dev, raw.stat().st_ino)
    os.chmod(raw, 0o2770)

    assert session.telemetry_file == raw / "usd_cli_telemetry.jsonl"
    assert (raw.stat().st_dev, raw.stat().st_ino) == pinned_identity
    assert stat.S_IMODE(raw.stat().st_mode) == 0o700


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced")
def test_session_rehardens_runner_pinned_collaborative_raw_directory(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(mode=0o700)
    pinned_identity = (raw.stat().st_dev, raw.stat().st_ino)
    os.chmod(raw, 0o2770)
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
        raw_directory_identity=pinned_identity,
    )

    assert session.telemetry_file == raw / "usd_cli_telemetry.jsonl"
    assert (raw.stat().st_dev, raw.stat().st_ino) == pinned_identity
    assert stat.S_IMODE(raw.stat().st_mode) == 0o700


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced")
def test_session_rejects_preexisting_collaborative_raw_directory(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(mode=0o700)
    os.chmod(raw, 0o2770)
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )

    with pytest.raises(RuntimeError, match="owner-controlled 0700"):
        _ = session.telemetry_file


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced")
def test_session_rehardens_identity_pinned_collaborative_receipt(
    tmp_path: Path,
) -> None:
    session = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path),
        workflow="test-workflow",
    )
    _append_test_receipt(session, "operation-one")
    receipt = tmp_path / "raw" / "usd_cli_command_receipts.jsonl"
    pinned_identity = (receipt.stat().st_dev, receipt.stat().st_ino)
    os.chmod(receipt, 0o660)

    _append_test_receipt(session, "operation-two")

    receipts = [
        json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["operation_id"] for entry in receipts] == [
        "operation-one",
        "operation-two",
    ]
    assert (receipt.stat().st_dev, receipt.stat().st_ino) == pinned_identity
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced")
def test_session_rehardens_digest_bound_collaborative_checkpoint(
    tmp_path: Path,
) -> None:
    route = SimpleNamespace(wrapper=tmp_path / "usd-cli-tel", target=tmp_path)
    first = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=route,
        workflow="test-workflow",
    )
    _append_test_receipt(first, "operation-one")
    receipt = first.receipt_file
    checkpoint = first.receipt_checkpoint_file
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    os.chmod(receipt, 0o660)
    os.chmod(checkpoint, 0o660)

    resumed = WorkflowUsdCliSession(
        project_dir=tmp_path,
        session_id="workflow-test",
        route=route,
        workflow="test-workflow",
        receipt_checkpoint_sha256=checkpoint_sha256,
    )
    _append_test_receipt(resumed, "operation-two")

    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o600
