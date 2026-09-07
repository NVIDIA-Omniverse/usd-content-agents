# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Client ↔ backend version handshake (usd_core.remote_protocol).

A backend built from a different checkout silently drifts the wire contract (wrong
renders, not errors), so clients must refuse to run against a mismatched — or
unversioned — service and ask for a backend re-deploy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
from usd_core.remote_protocol import (
    PROTOCOL_VERSION,
    ProtocolMismatchError,
    check_remote_protocol,
)

SERVICE_DIR = Path(__file__).resolve().parents[1] / "apps" / "ovrtx_rendering_api"


def _live_response(url: str, payload: dict | None, status: int = 200) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", url),
                          json=payload if payload is not None else {})


# ── the checker itself ───────────────────────────────────────────────────────────


def test_check_passes_on_matching_version(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _live_response(
        url, {"status": "alive", "protocol_version": PROTOCOL_VERSION}))
    # v2+: the checker returns the whole /live payload so clients can adopt
    # advertised capabilities (max_body_bytes, features) without a second probe
    info = check_remote_protocol("http://gpu:8000/")
    assert info["protocol_version"] == PROTOCOL_VERSION


def test_check_authenticates_live_probe_when_api_key_is_configured(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return _live_response(
            url,
            {"status": "alive", "protocol_version": PROTOCOL_VERSION},
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    check_remote_protocol("https://function.invocation.api.nvcf.nvidia.com", api_key="k")

    assert captured["headers"] == {"Authorization": "Bearer k"}


def test_check_can_require_declared_engine_identity(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _live_response(
            url,
            {
                "status": "alive",
                "protocol_version": PROTOCOL_VERSION,
                "engine": "ovrtx",
            },
        ),
    )
    info = check_remote_protocol(
        "http://gpu:8000/",
        required_engine="ovrtx",
    )
    assert info["engine"] == "ovrtx"

    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _live_response(
            url,
            {"status": "alive", "protocol_version": PROTOCOL_VERSION},
        ),
    )
    with pytest.raises(ProtocolMismatchError, match="reports engine 'unknown'"):
        check_remote_protocol(
            "http://gpu:8000/",
            required_engine="ovrtx",
        )


def test_check_can_require_advertised_features(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _live_response(
            url,
            {
                "status": "alive",
                "protocol_version": PROTOCOL_VERSION,
                "features": ["cas", "physics-simulate"],
            },
        ),
    )

    info = check_remote_protocol(
        "http://gpu:8000/",
        required_features=("physics-simulate",),
    )
    assert "physics-simulate" in info["features"]

    with pytest.raises(ProtocolMismatchError, match="missing-feature"):
        check_remote_protocol(
            "http://gpu:8000/",
            required_features=("missing-feature",),
        )


def test_check_refuses_on_version_mismatch(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _live_response(
        url, {"status": "alive", "protocol_version": PROTOCOL_VERSION + 1}))
    with pytest.raises(ProtocolMismatchError, match="re-deploy"):
        check_remote_protocol("http://gpu:8000")


def test_check_refuses_an_unversioned_backend(monkeypatch):
    # a service deployed before version reporting answers /live without the field
    monkeypatch.setattr(httpx, "get",
                        lambda url, **kw: _live_response(url, {"status": "alive"}))
    with pytest.raises(ProtocolMismatchError, match="does not report"):
        check_remote_protocol("http://gpu:8000")

    # ...and a service so old it has no /live at all (404) is the same story
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _live_response(url, None, 404))
    with pytest.raises(ProtocolMismatchError, match="re-deploy") as exc_info:
        check_remote_protocol("http://gpu:8000")
    message = str(exc_info.value)
    assert "`remote_verify_version = false` directly under `[render]`" in message
    assert "`USD_CLI_RENDER_REMOTE_VERIFY_VERSION=false`" in message


def test_check_reports_unreachable_service_clearly(monkeypatch):
    def _boom(url, **kw):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(httpx, "get", _boom)
    with pytest.raises(RuntimeError, match="cannot reach"):
        check_remote_protocol("http://gpu:8000")


# ── render client refuses before doing any work ──────────────────────────────────


def _patch_live(monkeypatch, payload: dict):
    def fake_get(self, url, **kw):
        assert url.endswith("/live")
        return _live_response(url, payload)
    monkeypatch.setattr(httpx.Client, "get", fake_get)


def test_remote_render_refuses_mismatched_backend_before_packaging(monkeypatch, tmp_path):
    from usd_core.render.remote import RemoteRenderBackend

    _patch_live(monkeypatch, {"status": "alive", "protocol_version": PROTOCOL_VERSION + 1})
    backend = RemoteRenderBackend("http://gpu:8000")
    # stage=None: packaging would crash on it, so the version check must fire first
    with pytest.raises(ProtocolMismatchError, match="re-deploy"):
        backend.render(None, ["/World/cam"], 64, 64, tmp_path / "out")


def test_remote_render_uses_backend_key_for_version_probe(monkeypatch):
    from usd_core.render.remote import RemoteRenderBackend

    captured = {}

    def fake_get(self, url, **kwargs):
        captured.update(kwargs)
        return _live_response(
            url,
            {"status": "alive", "protocol_version": PROTOCOL_VERSION},
        )

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    backend = RemoteRenderBackend("http://gpu:8000", api_key="k")

    with httpx.Client() as client:
        backend._ensure_protocol(client)

    assert captured["headers"] == {"Authorization": "Bearer k"}


def test_remote_render_verifies_once_then_proceeds(monkeypatch, tmp_path):
    import base64

    from usd_core.render.remote import RemoteRenderBackend

    calls = {"live": 0}

    def fake_get(self, url, **kw):
        calls["live"] += 1
        return _live_response(url, {"status": "alive",
                                    "protocol_version": PROTOCOL_VERSION})
    monkeypatch.setattr(httpx.Client, "get", fake_get)

    backend = RemoteRenderBackend("http://gpu:8000")
    img = base64.b64encode(b"png").decode()

    def fake_execute_upload(client, send_path, compression, params):
        return _live_response("http://gpu:8000/render/upload",
                              {"results": [{"camera": c, "image_base64": img}
                                           for c in params["cameras"]]})
    monkeypatch.setattr(backend, "_post_multipart", fake_execute_upload)
    monkeypatch.setattr(RemoteRenderBackend, "_package_usdz",
                        classmethod(lambda cls, stage, wd: (wd / "scene_bundle.usdz")))
    monkeypatch.setattr(backend, "_prepare_upload", lambda p: (_touch(p), "none"))

    backend.render(None, ["/World/cam"], 64, 64, tmp_path / "out")
    backend.render(None, ["/World/cam"], 64, 64, tmp_path / "out")
    assert calls["live"] == 1  # verified once per backend instance, not per render


def _touch(p: Path) -> Path:
    p.write_bytes(b"usdz")
    return p


def test_remote_render_version_check_can_be_disabled(monkeypatch, tmp_path):
    from usd_core.render.remote import RemoteRenderBackend

    def fail_get(self, url, **kw):
        raise AssertionError("/live must not be probed when verification is off")
    monkeypatch.setattr(httpx.Client, "get", fail_get)

    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)
    backend._ensure_protocol(httpx.Client())  # no probe, no raise


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        (
            "render.remote_verify_version = false",
            "directly (without the `render.` prefix)",
        ),
        (
            "remote_verify_vesion = false",
            "did you mean 'remote_verify_version'?",
        ),
        (
            "render.remote_verify_vesion = false",
            "directly (without the `render.` prefix)",
        ),
    ],
)
def test_malformed_remote_version_override_is_rejected(
    monkeypatch, tmp_path, setting, message
):
    from usd_core.config import load_config

    monkeypatch.setenv("HOME", str(tmp_path))
    state_dir = tmp_path / "project" / ".usd-cli"
    state_dir.mkdir(parents=True)
    (state_dir / "config.toml").write_text(
        '[render]\nrenderer = "remote"\nremote_url = "http://gpu:8000"\n'
        f"{setting}\n"
    )

    with pytest.raises(ValueError, match="remote_verify") as exc_info:
        load_config(tmp_path / "project")
    assert message in str(exc_info.value)


def test_canonical_config_override_skips_live_and_reaches_render(
    monkeypatch, tmp_path
):
    import base64

    from usd_core.config import load_config
    from usd_core.render.factory import make_backend
    from usd_core.render.remote import RemoteRenderBackend

    monkeypatch.setenv("HOME", str(tmp_path))
    state_dir = tmp_path / "project" / ".usd-cli"
    state_dir.mkdir(parents=True)
    (state_dir / "config.toml").write_text(
        '[render]\nrenderer = "remote"\nremote_url = "http://gpu:8000"\n'
        "remote_verify_version = false\n"
    )

    def fail_get(self, url, **kw):
        raise AssertionError("/live must not be probed when verification is off")

    monkeypatch.setattr(httpx.Client, "get", fail_get)
    backend = make_backend(load_config(tmp_path / "project"))
    assert backend._verify_version is False
    image = base64.b64encode(b"png").decode()
    monkeypatch.setattr(
        backend,
        "_post_multipart",
        lambda client, send_path, compression, params: _live_response(
            "http://gpu:8000/render/upload",
            {"results": [{"camera": camera, "image_base64": image}
                         for camera in params["cameras"]]},
        ),
    )
    monkeypatch.setattr(
        RemoteRenderBackend,
        "_package_usdz",
        classmethod(lambda cls, stage, work_dir: work_dir / "scene_bundle.usdz"),
    )
    monkeypatch.setattr(backend, "_prepare_upload", lambda path: (_touch(path), "none"))

    results = backend.render(
        None, ["/World/cam"], 64, 64, tmp_path / "renders"
    )
    assert len(results) == 1


def test_environment_override_disables_remote_version_check(monkeypatch, tmp_path):
    from usd_core.config import load_config
    from usd_core.render.factory import make_backend

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USD_CLI_RENDER_RENDERER", "remote")
    monkeypatch.setenv("USD_CLI_RENDER_REMOTE_URL", "http://gpu:8000")
    monkeypatch.setenv("USD_CLI_RENDER_REMOTE_VERIFY_VERSION", "false")

    backend = make_backend(load_config(tmp_path))
    assert backend._verify_version is False


def test_factory_wires_remote_verify_version():
    from usd_core.config import Config
    from usd_core.render.factory import make_backend

    cfg = Config()
    cfg.render.update(renderer="remote", remote_url="http://gpu:8000")
    assert make_backend(cfg)._verify_version is True  # default: strict

    cfg.render["remote_verify_version"] = "false"
    assert make_backend(cfg)._verify_version is False


def test_remote_serve_command_selects_monorepo_physics_service():
    from usd_cli.remote import serve_cmd

    docker = serve_cmd()
    bare = serve_cmd(bare=True)

    assert "USD_CLI_ROOT=apps/usd_cli" in docker
    assert 'SERVICE_DIR="$USD_CLI_ROOT/apps/ovrtx_rendering_api"' in docker
    assert '"$SERVICE_DIR/docker-compose.yml"' in docker
    assert "USD_CLI_ROOT=apps/usd_cli" in bare
    assert 'SERVICE_DIR="$USD_CLI_ROOT/apps/ovrtx_rendering_api"' in bare
    assert 'pip install -e "$SERVICE_DIR"' in bare


# ── physics client refuses too ────────────────────────────────────────────────────


def test_remote_physics_refuses_mismatched_backend(monkeypatch, tmp_path):
    from usd_core.physics_runtime import evaluate_remote

    monkeypatch.setattr(httpx, "get",
                        lambda url, **kw: _live_response(url, {"status": "alive"}))
    scene = tmp_path / "scene.usda"
    scene.write_text("#usda 1.0\n")
    with pytest.raises(ProtocolMismatchError, match="re-deploy"):
        evaluate_remote(scene, body_pattern="/World/Body", duration_s=1.0, dt=1 / 240,
                        sample_fps=30, base_url="http://gpu:8000")


# ── service reports its baked-in version ─────────────────────────────────────────


def test_service_reports_protocol_version(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(SERVICE_DIR))
    monkeypatch.setenv("OVRTX_API_KEY", "test-key")
    import service.main as sm

    async def _no_init():
        return None
    monkeypatch.setattr(sm, "_background_init", _no_init)

    with TestClient(sm.app) as client:
        live = client.get("/live").json()
        assert live["protocol_version"] == PROTOCOL_VERSION  # public, pre-auth
        assert live["engine"] == "ovrtx"
        assert live["renderer"] == "ovrtx"
        assert "physics-simulate" in live["features"]

        health = client.get("/health",
                            headers={"Authorization": "Bearer test-key"}).json()
        assert health["protocol_version"] == PROTOCOL_VERSION
