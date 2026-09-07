# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from typer.testing import CliRunner
from usd_cli import main as main_module
from usd_cli.main import app
from usd_core.models import Response
from usd_core.render import probe as probe_module
from usd_core.render.base import RenderResult
from usd_server.app import SIDE_EFFECT_COMMANDS


def test_render_probe_is_side_effecting_daemon_work() -> None:
    assert "render-probe" in SIDE_EFFECT_COMMANDS


class _FakeBackend:
    name = "ovrtx"

    def render(
        self,
        _stage,
        cameras: list[str],
        width: int,
        height: int,
        out_dir: Path,
        **_kwargs,
    ) -> list[RenderResult]:
        path = Path(out_dir) / "probe.png"
        Image.new("RGB", (width, height), "red").save(path)
        return [
            RenderResult(
                path=str(path),
                camera=cameras[0],
                width=width,
                height=height,
                backend=self.name,
            )
        ]


def _config() -> SimpleNamespace:
    return SimpleNamespace(render={})


def test_probe_rejects_non_ovrtx_renderer_before_render(monkeypatch) -> None:
    monkeypatch.setattr(probe_module, "resolved_renderer", lambda _config: "none")
    called = False

    def unexpected_backend(_config):
        nonlocal called
        called = True
        return _FakeBackend()

    monkeypatch.setattr(probe_module, "make_backend", unexpected_backend)
    result = probe_module.probe_render_engine(_config())

    assert result["ready"] is False
    assert result["engine"] == "none"
    assert "required render engine" in result["error"]
    assert called is False


def test_probe_exercises_local_ovrtx_and_records_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(probe_module, "resolved_renderer", lambda _config: "ovrtx")
    monkeypatch.setattr(probe_module, "make_backend", lambda _config: _FakeBackend())

    result = probe_module.probe_render_engine(_config(), output_dir=tmp_path)

    assert result["ready"] is True
    assert result["engine"] == "ovrtx"
    assert result["transport"] == "local"
    assert result["render"]["width"] == 64
    assert result["render"]["height"] == 64
    assert Path(result["render"]["path"]).is_file()


def test_probe_rejects_backend_factory_identity_mismatch(monkeypatch) -> None:
    backend = _FakeBackend()
    backend.name = "invalid"
    monkeypatch.setattr(probe_module, "resolved_renderer", lambda _config: "ovrtx")
    monkeypatch.setattr(probe_module, "make_backend", lambda _config: backend)

    result = probe_module.probe_render_engine(_config())

    assert result["ready"] is False
    assert "did not produce an OVRTX-backed renderer" in result["error"]


def test_probe_rejects_remote_without_declared_ovrtx_identity(monkeypatch) -> None:
    from usd_core import remote_protocol

    monkeypatch.setattr(probe_module, "resolved_renderer", lambda _config: "remote")
    monkeypatch.setattr(
        probe_module,
        "resolve_render_backends",
        lambda _render: [{"url": "https://renderer.example.test"}],
    )

    def reject_identity(_url, **_kwargs):
        raise remote_protocol.ProtocolMismatchError(
            "remote backend reports engine 'unknown'"
        )

    monkeypatch.setattr(remote_protocol, "check_remote_protocol", reject_identity)
    result = probe_module.probe_render_engine(_config())

    assert result["ready"] is False
    assert result["transport"] == "remote"
    assert "identity/protocol probe failed" in result["error"]
    assert "unknown" in result["error"]


def test_probe_accepts_remote_only_after_identity_and_real_render(
    tmp_path: Path, monkeypatch
) -> None:
    from usd_core import remote_protocol

    backend = _FakeBackend()
    backend.name = "remote"
    monkeypatch.setattr(probe_module, "resolved_renderer", lambda _config: "remote")
    monkeypatch.setattr(
        probe_module,
        "resolve_render_backends",
        lambda _render: [{"url": "https://renderer.example.test"}],
    )
    monkeypatch.setattr(
        remote_protocol,
        "check_remote_protocol",
        lambda _url, **_kwargs: {
            "status": "alive",
            "protocol_version": remote_protocol.PROTOCOL_VERSION,
            "engine": "ovrtx",
        },
    )
    monkeypatch.setattr(probe_module, "make_backend", lambda _config: backend)

    result = probe_module.probe_render_engine(_config(), output_dir=tmp_path)

    assert result["ready"] is True
    assert result["engine"] == "ovrtx"
    assert result["transport"] == "remote"
    assert result["backends"] == [
        {
            "url": "https://renderer.example.test",
            "engine": "ovrtx",
            "protocol_version": remote_protocol.PROTOCOL_VERSION,
            "status": "alive",
        }
    ]


def test_render_probe_dispatches_to_project_daemon(
    tmp_path: Path, monkeypatch
) -> None:
    expected = {
        "schema_version": "usd-cli.render-probe.v1",
        "capabilities": ["appearance.clear.v1"],
        "engine": "ovrtx",
        "resolved_renderer": "ovrtx",
        "transport": "local",
        "ready": True,
        "render": {
            "path": str(tmp_path / "probe.png"),
            "width": 64,
            "height": 64,
            "size_bytes": 10,
        },
    }
    observed: dict[str, object] = {}

    def dispatch(command: str, payload: dict[str, object]) -> Response:
        observed.update(command=command, payload=payload)
        return Response(command=command, data={"probe": expected})

    monkeypatch.setattr(main_module, "dispatch", dispatch)
    monkeypatch.setattr(
        probe_module,
        "probe_render_engine",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("client process must not initialize the renderer")
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "render-probe",
            "--require-engine",
            "ovrtx",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == expected
    assert observed == {
        "command": "render-probe",
        "payload": {
            "required_engine": "ovrtx",
            "output_dir": str(tmp_path.resolve()),
        },
    }


def test_render_probe_dispatches_to_explicit_parent_server(
    tmp_path: Path, monkeypatch
) -> None:
    expected = {
        "schema_version": "usd-cli.render-probe.v1",
        "engine": "ovrtx",
        "transport": "local",
        "ready": True,
    }
    observed: dict[str, object] = {}

    def dispatch(command: str, payload: dict[str, object]) -> Response:
        observed.update(command=command, payload=payload)
        return Response(
            command=command,
            ok=True,
            summary={},
            data={"probe": expected},
        )

    monkeypatch.setattr(main_module, "dispatch", dispatch)
    monkeypatch.setattr(
        probe_module,
        "probe_render_engine",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("client process must not initialize the renderer")
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "--server",
            "http://127.0.0.1:43210",
            "--session",
            "parent-session",
            "render-probe",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == expected
    assert observed == {
        "command": "render-probe",
        "payload": {
            "required_engine": "ovrtx",
            "output_dir": str(tmp_path.resolve()),
        },
    }


def test_render_probe_fails_fast_when_child_local_gpu_is_forbidden(
    monkeypatch,
) -> None:
    monkeypatch.setenv("USD_CLI_LOCAL_GPU_FORBIDDEN", "1")
    monkeypatch.delenv("CONTENT_WORKFLOW_USD_CLI_SERVER_URL", raising=False)
    monkeypatch.setattr(
        probe_module,
        "probe_render_engine",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sandboxed client must not initialize the renderer")
        ),
    )

    result = CliRunner().invoke(app, ["render-probe"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ready"] is False
    assert payload["error_type"] == "parent_renderer_capability_required"


def test_render_fails_before_dispatch_when_child_has_no_parent_server(
    monkeypatch,
) -> None:
    monkeypatch.setenv("USD_CLI_LOCAL_GPU_FORBIDDEN", "1")
    monkeypatch.delenv("CONTENT_WORKFLOW_USD_CLI_SERVER_URL", raising=False)
    monkeypatch.setattr(
        main_module,
        "dispatch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sandboxed client must not start or discover a renderer")
        ),
    )

    result = CliRunner().invoke(app, ["--json", "render"])

    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["data"]["error_type"] == "parent_renderer_capability_required"
