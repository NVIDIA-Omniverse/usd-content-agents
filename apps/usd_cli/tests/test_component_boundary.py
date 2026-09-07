# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for component-boundary security and reliability findings."""

from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from fastapi.testclient import TestClient
from pxr import Sdf, Usd, UsdGeom

from conftest import CUBE
from usd_cli import client as cli_client
from usd_cli import daemon as cli_daemon
from usd_cli import main
from usd_cli import output
from usd_cli.state import G
from usd_core.config import Config
from usd_core import config as config_module
from usd_core.history import Op
from usd_core.models import Response
from usd_core.session import Session
from usd_server import app as server_app

SERVICE_ROOT = Path(__file__).resolve().parents[1] / "apps" / "ovrtx_rendering_api"
USD_CLI_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_USD_CLI_SKILL = (
    USD_CLI_ROOT.parents[1]
    / "agentic"
    / ".agents"
    / "skills"
    / "usd-cli"
    / "SKILL.md"
)


def test_canonical_skill_installs_exactly_one_openusd_provider() -> None:
    skill = CANONICAL_USD_CLI_SKILL.read_text(encoding="utf-8")
    install_lines = [line for line in skill.splitlines() if "pip install" in line]

    assert install_lines
    for line in install_lines:
        match = re.search(r'apps/usd_cli\[([^]]+)\]', line)
        assert match is not None
        assert {"cli", "server"}.issubset(match.group(1).split(","))
    assert all(
        "--overrides apps/usd_cli/requirements/usd-exchange-override.txt" in line
        for line in install_lines
    )
    assert "usd-cli[cli,server,usd]" not in skill
    assert '".[cli,server,usd]"' not in skill
    assert not re.search(r"(?:pip|uv pip) install[^\n]*\busd-core\b", skill)


def test_component_docs_use_the_canonical_usd_cli_skill() -> None:
    readme = (USD_CLI_ROOT / "README.md").read_text(encoding="utf-8")

    assert "../../agentic/.agents/skills/usd-cli/SKILL.md" in readme
    assert ".claude/skills/usd-cli" not in readme
    assert "not a standalone repository, product, workflow" in readme
    assert "ROADMAP.md" not in readme


def test_daemon_requires_correct_token_and_rejects_private_dispatch(tmp_path):
    app = server_app.build_app(Config(project_dir=tmp_path), token="right", instance_id="one")
    with TestClient(app) as http:
        body = {"command": "info", "payload": {}}
        assert http.post("/cmd", json=body).status_code == 401
        assert http.post("/cmd", json=body, headers={"x-usd-cli-token": "wrong"}).status_code == 401
        good = http.post("/cmd", json={"command": "_open", "payload": {}},
                         headers={"x-usd-cli-token": "right"})
        assert good.status_code == 200
        assert good.json()["ok"] is False
        assert "unknown command" in good.json()["issues"][0]["message"]
        assert http.post("/cmd", json={**body, "unexpected": True},
                         headers={"x-usd-cli-token": "right"}).status_code == 422


def test_daemon_serializes_session_access(monkeypatch, tmp_path):
    class FakeSession:
        active = 0
        maximum = 0

        def __init__(self, _config):
            self._stage_path = None

        def info(self):
            type(self).active += 1
            type(self).maximum = max(type(self).maximum, type(self).active)
            time.sleep(0.08)
            type(self).active -= 1
            return Response(command="info")

    monkeypatch.setattr(server_app, "Session", FakeSession)
    app = server_app.build_app(Config(project_dir=tmp_path), token="token", instance_id="one")
    headers = {"x-usd-cli-token": "token"}
    with TestClient(app) as http:
        threads = [threading.Thread(target=lambda: http.post(
            "/cmd", json={"command": "info", "payload": {}}, headers=headers)) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert FakeSession.maximum == 1


@pytest.mark.parametrize("name", ["../escape", "../../target", "/absolute", "a/b", "a\\b", ".."])
def test_checkpoint_names_cannot_escape(tmp_path, name):
    session = Session(Config(project_dir=tmp_path))
    with pytest.raises(ValueError):
        session.checkpoint_path(name)


def test_checkpoint_publish_is_atomic(tmp_path):
    session = Session.open(CUBE.path, Config(project_dir=tmp_path))
    result = session.checkpoint_save("safe-name")
    assert result.ok
    target = tmp_path / ".usd-cli" / "checkpoints" / "safe-name.usd"
    assert target.exists()
    assert not list(target.parent.glob("*.tmp.usd"))


@pytest.mark.parametrize("command", ["checkpoint.save", "checkpoint.load", "checkpoint.delete"])
def test_checkpoint_commands_reject_traversal(tmp_path, command):
    session = Session.open(CUBE.path, Config(project_dir=tmp_path))
    result = server_app.dispatch(session, command, {"name": "../../../escape"})
    assert not result.ok
    assert "checkpoint name" in result.issues[0].message


def test_unexpected_daemon_start_failure_is_not_a_successful_stub(monkeypatch, tmp_path):
    from usd_cli import daemon

    monkeypatch.setattr(cli_client, "load_config", lambda: Config(project_dir=tmp_path))
    monkeypatch.setattr(daemon, "ensure_running", lambda _cfg: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.delenv("USD_CLI_NO_DAEMON", raising=False)
    old_server = G.server
    G.server = None
    try:
        result = cli_client.dispatch("info", {})
    finally:
        G.server = old_server
    assert not result.ok
    assert result.summary["error_type"] == "startup"
    assert not result.summary.get("stub")


def test_stop_does_not_signal_unverified_stale_pid(monkeypatch, tmp_path):
    cfg = Config(project_dir=tmp_path)
    state = {"pid": 42, "host": "127.0.0.1", "port": 9999, "token": "old",
             "instance_id": "expected", "project_id": "project"}
    cfg.state_dir.mkdir()
    cfg.server_state_path.write_text(__import__("json").dumps(state))
    monkeypatch.setattr(cli_daemon, "_alive", lambda _pid: True)
    monkeypatch.setattr(cli_daemon, "_health", lambda *_args, **_kwargs: {
        "instance_id": "different", "project_id": "project"})
    signaled = []
    monkeypatch.setattr(cli_daemon.os, "kill", lambda *args: signaled.append(args))
    with pytest.raises(cli_daemon.DaemonStopRefused, match="malformed|foreign"):
        cli_daemon.stop(cfg)
    assert signaled == []


@pytest.mark.parametrize(
    ("command", "field"),
    [
        ("open", "file"),
        ("save", "path"),
        ("export", "path"),
        ("import", "source"),
        ("convert", "source"),
        ("convert", "output"),
        ("verify", "file"),
        ("render", "output"),
        ("render", "against"),
        ("render-frames", "scene"),
        ("render-frames", "output"),
        ("physics.simulate", "scene"),
        ("physics.simulate", "output"),
        ("material", "library"),
        ("material", "mdl"),
        ("material", "diffuse_texture"),
        ("material", "normal_texture"),
        ("material", "orm_texture"),
        ("material", "roughness_texture"),
        ("material", "metallic_texture"),
    ],
)
def test_loopback_daemon_sandboxes_every_file_payload(
    tmp_path: Path,
    command: str,
    field: str,
) -> None:
    cfg = Config(
        project_dir=tmp_path,
        server={"host": "127.0.0.1", "allowed_roots": [str(tmp_path)]},
    )
    server_app._validate_request_policy(
        cfg,
        command,
        {field: str(tmp_path / "inside.asset")},
        shared=False,
    )
    with pytest.raises(ValueError, match=r"allowed_(?:write_)?roots"):
        server_app._validate_request_policy(
            cfg,
            command,
            {field: str(tmp_path.parent / "outside.asset")},
            shared=False,
        )


def test_read_capability_never_grants_write_capability(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source_library = tmp_path / "source-library"
    project.mkdir()
    source_library.mkdir()
    source = source_library / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    cfg = Config(
        project_dir=project,
        server={
            "allowed_roots": [str(project), str(source_library)],
            "allowed_write_roots": [str(project)],
        },
    )

    server_app._validate_request_policy(
        cfg, "open", {"file": str(source)}, shared=False
    )
    server_app._validate_request_policy(
        cfg,
        "render",
        {"output": str(project / "renders")},
        shared=False,
    )
    with pytest.raises(ValueError, match="allowed_write_roots"):
        server_app._validate_request_policy(
            cfg,
            "render",
            {"output": str(source_library / "renders")},
            shared=False,
        )


def test_physics_simulate_reads_external_scene_but_writes_only_to_project(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source_library = tmp_path / "source-library"
    project.mkdir()
    source_library.mkdir()
    scene = source_library / "drop-scenario.usda"
    scene.write_text("#usda 1.0\n", encoding="utf-8")
    cfg = Config(
        project_dir=project,
        server={
            "allowed_roots": [str(project), str(source_library)],
            "allowed_write_roots": [str(project)],
        },
    )

    server_app._validate_request_policy(
        cfg,
        "physics.simulate",
        {
            "scene": str(scene),
            "output": str(project / "simulation"),
        },
        shared=False,
    )

    with pytest.raises(ValueError, match="allowed_write_roots"):
        server_app._validate_request_policy(
            cfg,
            "physics.simulate",
            {
                "scene": str(scene),
                "output": str(source_library / "simulation"),
            },
            shared=False,
        )


@pytest.mark.parametrize("command,payload", [("save", {}), ("export", {"format": "usda"})])
def test_implicit_stage_writes_must_stay_in_write_roots(
    tmp_path: Path,
    command: str,
    payload: dict[str, str],
) -> None:
    project = tmp_path / "project"
    source_library = tmp_path / "source-library"
    project.mkdir()
    source_library.mkdir()
    cfg = Config(
        project_dir=project,
        server={
            "allowed_roots": [str(project), str(source_library)],
            "allowed_write_roots": [str(project)],
        },
    )
    session = SimpleNamespace(_stage_path=str(source_library / "source.usda"))

    with pytest.raises(ValueError, match="allowed_write_roots"):
        server_app._validate_request_policy(
            cfg,
            command,
            payload,
            shared=False,
            session=session,
        )


def test_implicit_conversion_cannot_publish_beside_external_source(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source_library = tmp_path / "source-library"
    project.mkdir()
    source_library.mkdir()
    cfg = Config(
        project_dir=project,
        server={
            "allowed_roots": [str(project), str(source_library)],
            "allowed_write_roots": [str(project)],
        },
    )

    with pytest.raises(ValueError, match="allowed_write_roots"):
        server_app._validate_request_policy(
            cfg,
            "convert",
            {"source": str(source_library / "source.obj")},
            shared=False,
        )


@pytest.mark.parametrize(
    "inputs",
    [
        {"diffuse_texture": "../outside.png"},
        ["diffuse_texture=../outside.png"],
    ],
)
def test_loopback_daemon_sandboxes_nested_material_asset_inputs(
    tmp_path: Path,
    inputs: object,
) -> None:
    cfg = Config(
        project_dir=tmp_path,
        server={"host": "127.0.0.1", "allowed_roots": [str(tmp_path)]},
    )
    with pytest.raises(ValueError, match="allowed_roots"):
        server_app._validate_request_policy(
            cfg,
            "material",
            {"inputs": inputs},
            shared=False,
        )


@pytest.mark.parametrize(
    "uri",
    [
        "omniverse://server/private.usd",
        "omniverse:private.usd",
        "file:/private.usd",
        "s3:private.usd",
    ],
)
def test_daemon_path_policy_rejects_uri_disguised_as_relative_path(
    tmp_path: Path,
    uri: str,
) -> None:
    cfg = Config(project_dir=tmp_path)
    with pytest.raises(ValueError, match="URI is not permitted"):
        server_app._validate_request_policy(
            cfg,
            "open",
            {"file": uri},
            shared=False,
        )


def test_loopback_http_route_enforces_allowed_roots(tmp_path: Path) -> None:
    app = server_app.build_app(
        Config(project_dir=tmp_path), token="token", instance_id="one"
    )
    with TestClient(app) as http:
        response = http.post(
            "/cmd",
            json={
                "command": "open",
                "payload": {"file": str(tmp_path.parent / "outside.usda")},
            },
            headers={"x-usd-cli-token": "token"},
        )
    assert response.status_code == 400
    assert "allowed_roots" in response.json()["detail"]


def test_policy_refusal_is_not_reported_as_a_transport_failure() -> None:
    """A 400 from the daemon must keep its reason and NOT read as 'unreachable'.

    Path validation now runs for every request, so an out-of-root path is the
    common 400. Classifying it as "transport" (exit 3) both discarded the only
    line explaining the refusal and told agents to restart a healthy daemon."""
    import httpx

    request = httpx.Request("POST", "http://127.0.0.1:9999/cmd")
    response = httpx.Response(
        400,
        json={"detail": "file path is outside server.allowed_roots"},
        request=request,
    )
    exc = httpx.HTTPStatusError("400", request=request, response=response)

    result = cli_client._transport_error(
        "open", "http://127.0.0.1:9999", "token", exc, 30.0
    )
    assert not result.ok
    assert result.summary["error_type"] == "policy"
    assert result.summary["status"] == 400
    assert "outside server.allowed_roots" in result.issues[0].message
    # Exit 1 (the command failed), never 3 ("the daemon is down, restart it").
    assert output.EXIT_RUNTIME == output._error_exit_code(result)


def test_foreign_listener_404_stays_a_transport_failure() -> None:
    """Only the statuses the daemon emits for refusals are reclassified: a 404
    is likelier an unrelated service on the port, which IS a transport problem."""
    import httpx

    request = httpx.Request("POST", "http://127.0.0.1:9999/cmd")
    response = httpx.Response(404, text="Not Found", request=request)
    exc = httpx.HTTPStatusError("404", request=request, response=response)

    result = cli_client._transport_error(
        "open", "http://127.0.0.1:9999", None, exc, 30.0
    )
    assert result.summary["error_type"] == "transport"


def test_server_stop_reports_a_refusal_without_a_traceback(monkeypatch, capsys):
    """`DaemonStopRefused` is a decision the CLI must render, not let escape.

    It reached the user as a raw Python traceback because nothing between
    daemon.stop() and typer handled it."""
    monkeypatch.setattr(
        cli_daemon,
        "stop",
        lambda _cfg: (_ for _ in ()).throw(
            cli_daemon.DaemonStopRefused("refused to signal unverifiable live daemon")
        ),
    )
    with pytest.raises(typer.Exit) as excinfo:
        main._daemon_op("stop")
    assert excinfo.value.exit_code == 1
    assert "refused to signal unverifiable live daemon" in capsys.readouterr().err


def test_server_restart_does_not_start_a_second_daemon_after_a_refused_stop(
    monkeypatch, capsys
):
    """A refused stop leaves the old daemon possibly live; starting anyway would
    orphan it and hijack the state file — the exact race the ownership work closes."""
    monkeypatch.setattr(
        cli_daemon,
        "stop",
        lambda _cfg: (_ for _ in ()).throw(cli_daemon.DaemonStopRefused("refused")),
    )
    started = []
    monkeypatch.setattr(cli_daemon, "start", lambda cfg: started.append(cfg))

    with pytest.raises(typer.Exit) as excinfo:
        main.server_restart()
    assert excinfo.value.exit_code == 1
    assert started == []
    assert "not starting a replacement" in capsys.readouterr().err


def test_set_cannot_author_an_out_of_root_asset_dependency(tmp_path: Path) -> None:
    scene_path = tmp_path / "asset-attribute.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    prim = stage.DefinePrim("/World/Shader", "Shader")
    prim.CreateAttribute("asset:test", Sdf.ValueTypeNames.Asset)
    stage.GetRootLayer().Save()
    session = Session.open(scene_path, Config(project_dir=tmp_path))

    with pytest.raises(ValueError, match="allowed_roots"):
        server_app._validate_request_policy(
            session.config,
            "set",
            {
                "ref": "/World/Shader",
                "attr": "asset:test",
                "value": str(tmp_path.parent / "outside.usda"),
            },
            shared=False,
            session=session,
        )
    session.new()


def test_chunked_request_body_is_bounded_before_json_parsing(tmp_path: Path) -> None:
    app = server_app.build_app(
        Config(project_dir=tmp_path), token="token", instance_id="one"
    )
    prefix = b'{"command":"info","payload":{"padding":"'
    suffix = b'"}}'
    chunks = iter((prefix, b"x" * (2 * 1024 * 1024), suffix))
    with TestClient(app) as http:
        response = http.post(
            "/cmd",
            content=chunks,
            headers={
                "content-type": "application/json",
                "x-usd-cli-token": "token",
            },
        )
    assert response.status_code == 413
    assert response.json()["detail"] == "request body too large"


@pytest.mark.parametrize("section,key,value", [
    ("backend", "engine", "blender"),
    ("render", "renderer", "imaginary"),
])
def test_unsupported_backend_configuration_is_rejected(monkeypatch, tmp_path, section, key, value):
    monkeypatch.setattr(config_module, "find_project_dir", lambda _start=None: tmp_path)
    monkeypatch.setattr(config_module, "_read_toml", lambda path: {
        section: {key: value}} if path == tmp_path / ".usd-cli" / "config.toml" else {})
    (tmp_path / ".usd-cli").mkdir()
    with pytest.raises(ValueError, match="unsupported"):
        config_module.load_config(tmp_path)


@pytest.mark.parametrize("failure_position", [1, 2, 3])
def test_failed_compound_undo_restores_stage_and_history(monkeypatch, tmp_path, failure_position):
    session = Session(Config(project_dir=tmp_path))
    session._stage = Usd.Stage.CreateInMemory()
    paths = [f"/{name}" for name in ("A", "B", "C")]
    for path in paths:
        UsdGeom.Xform.Define(session._stage, path)
        session._stage.GetPrimAtPath(path).SetActive(False)
    session.history.record(Op(command="chain", children=[Op(command="test", inverse={
            "undo": {"kind": "set_active", "path": path, "active": True},
            "redo": {"kind": "set_active", "path": path, "active": False},
        }) for path in paths]))
    before = session._stage.GetRootLayer().ExportToString()
    original = session._apply_change
    calls = 0

    def fail_at_position(change):
        nonlocal calls
        calls += 1
        if calls == failure_position:
            raise RuntimeError("injected failure")
        original(change)

    monkeypatch.setattr(session, "_apply_change", fail_at_position)
    result = session.undo(1)
    assert not result.ok
    assert "rolled back" in result.issues[0].message
    assert session._stage.GetRootLayer().ExportToString() == before
    assert len(session.history.entries()) == 1
    assert not session.history.can_redo


def test_remote_request_enforces_exact_source_and_resource_limits():
    sys.path.insert(0, str(SERVICE_ROOT))
    try:
        from service.models import RenderRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="exactly one"):
            RenderRequest(usd="#usda 1.0", usdz_base64="eA==", cameras=["/Camera"])
        with pytest.raises(ValidationError):
            RenderRequest(usd="#usda 1.0", cameras=[f"/Camera{i}" for i in range(9)])
        with pytest.raises(ValidationError):
            RenderRequest(usd="#usda 1.0", cameras=["/Camera"], image_width=4097)
    finally:
        sys.path.remove(str(SERVICE_ROOT))


def test_remote_api_key_is_mandatory(monkeypatch):
    sys.path.insert(0, str(SERVICE_ROOT))
    try:
        from fastapi import HTTPException
        from service.main import authenticate

        monkeypatch.delenv("OVRTX_API_KEY", raising=False)
        with pytest.raises(HTTPException) as missing:
            authenticate(None)
        assert missing.value.status_code == 503
        monkeypatch.setenv("OVRTX_API_KEY", "correct")
        with pytest.raises(HTTPException) as wrong:
            authenticate("Bearer wrong")
        assert wrong.value.status_code == 401
        assert authenticate("Bearer correct") is None
    finally:
        sys.path.remove(str(SERVICE_ROOT))


def test_remote_liveness_and_readiness_are_distinct(monkeypatch):
    sys.path.insert(0, str(SERVICE_ROOT))
    try:
        from service import main as service_main

        monkeypatch.setenv("OVRTX_API_KEY", "correct")
        service_main._init_state = "initializing"
        service_main._renderer = None
        http = TestClient(service_main.app)
        assert http.get("/live").status_code == 200
        assert http.get("/health").status_code == 401
        health = http.get("/health", headers={"Authorization": "Bearer correct"})
        assert health.status_code == 200 and health.json()["status"] == "initializing"
        assert http.get("/ready", headers={"Authorization": "Bearer correct"}).status_code == 503
    finally:
        sys.path.remove(str(SERVICE_ROOT))
