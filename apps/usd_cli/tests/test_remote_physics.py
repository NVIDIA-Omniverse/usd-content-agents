# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote physics runtime: `POST /physics/simulate` on the OVRTX service + the
low-level simulation fallback used when the local host cannot run ovphysx.

Unit tests (no GPU / no remote service / no ovphysx): the service side runs against a fake
simulator, the client side against a monkeypatched httpx. The real end-to-end solve is
covered by tests/test_physics_runtime.py on a supported host.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pxr")

SERVICE_DIR = Path(__file__).resolve().parents[1] / "apps" / "ovrtx_rendering_api"


def _rigid_body_scene(tmp_path: Path) -> Path:
    """Write a minimal, explicitly pre-authored physics scenario."""
    from pxr import Usd, UsdGeom, UsdPhysics

    scene = tmp_path / "physics_scenario.usda"
    stage = Usd.Stage.CreateNew(str(scene))
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    cube = UsdGeom.Cube.Define(stage, "/World/Body")
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.Scene.Define(stage, "/PhysicsScenario")
    stage.GetRootLayer().Save()
    return scene


def _stub_live(monkeypatch):
    """Answer the client's version handshake (GET /live) as a compatible backend."""
    import httpx
    from usd_core.remote_protocol import PROTOCOL_VERSION

    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(
        200, request=httpx.Request("GET", url),
        json={
            "status": "alive",
            "protocol_version": PROTOCOL_VERSION,
            "engine": "ovrtx",
            "features": ["physics-simulate"],
        }))


def _settling_trajectory():
    """[t, pose7, vel6] samples: a body that falls, lands, and stays still long enough
    for the settle detector (lin speed < 0.05 sustained >= 0.2 s)."""
    still = [0.0] * 6
    return [
        [0.0, [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -0.5, 0.0, 0.0, 0.0]],
        [0.2, [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -2.0, 0.0, 0.0, 0.0]],
        [0.4, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], still],
        [0.6, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], still],
        [0.8, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], still],
        [1.0, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], still],
    ]


def test_physics_simulate_endpoint_accepts_gzipped_scene(monkeypatch):
    """POST /physics/simulate: multipart scene + params form field, gzip honored,
    independent of render warm-up state."""
    pytest.importorskip("fastapi")
    pytest.importorskip("python_multipart")
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(SERVICE_DIR))
    monkeypatch.setenv("OVRTX_API_KEY", "test-key")
    import service.main as sm

    captured = {}

    class FakeSimulator:
        daemon_running = True

        def simulate_scene_bytes(self, data, *, filename, body_pattern, duration_s, dt,
                                 sample_fps):
            captured.update(data=data, filename=filename, body_pattern=body_pattern,
                            duration_s=duration_s, dt=dt, sample_fps=sample_fps)
            return {"trajectory": _settling_trajectory(), "n_bodies": 1, "n_steps": 240}

        def close(self):
            pass

    async def _no_init():
        return None

    monkeypatch.setattr(sm, "_background_init", _no_init)
    monkeypatch.setattr(sm, "_simulator", FakeSimulator())
    # renderer deliberately NOT ready — physics must work regardless
    monkeypatch.setattr(sm, "_init_state", "initializing")

    with TestClient(sm.app) as client:
        body = gzip.compress(b"#usda 1.0 fake scene")
        resp = client.post(
            "/physics/simulate",
            files={"file": ("drop_settle_scene.usda.gz", body, "application/octet-stream")},
            data={"params": json.dumps({"body_pattern": "/World/Body", "duration_s": 1.0,
                                        "dt": 1.0 / 240.0, "sample_fps": 30,
                                        "compression": "gzip"})},
            headers={"Authorization": "Bearer test-key"})
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["n_bodies"] == 1 and payload["n_steps"] == 240
        assert len(payload["trajectory"]) == 6
        assert captured["data"] == b"#usda 1.0 fake scene"  # decompressed server-side
        assert captured["body_pattern"] == "/World/Body"

        # bad gzip → 400, not 500
        resp = client.post(
            "/physics/simulate",
            files={"file": ("x.usda.gz", b"not gzip", "application/octet-stream")},
            data={"params": json.dumps({"body_pattern": "/b", "compression": "gzip"})},
            headers={"Authorization": "Bearer test-key"})
        assert resp.status_code == 400

        # bad params → 400
        resp = client.post(
            "/physics/simulate",
            files={"file": ("x.usda", b"bytes", "application/octet-stream")},
            data={"params": json.dumps({"duration_s": 1.0})},  # body_pattern missing
            headers={"Authorization": "Bearer test-key"})
        assert resp.status_code == 400

        # auth still enforced
        resp = client.post(
            "/physics/simulate",
            files={"file": ("x.usda", b"bytes", "application/octet-stream")},
            data={"params": json.dumps({"body_pattern": "/b"})})
        assert resp.status_code == 401


def test_simulate_scene_falls_back_to_remote(tmp_path, monkeypatch):
    """The explicit scenario is shipped remotely and recorded locally."""
    import httpx
    from usd_core import physics_runtime

    monkeypatch.setattr(physics_runtime, "ovphysx_platform_supported", lambda: False)
    _stub_live(monkeypatch)
    calls = {}

    def fake_post(self, url, **kw):
        calls["url"] = url
        calls["headers"] = kw.get("headers")
        calls["params"] = json.loads(kw["data"]["params"])
        name, data, _ctype = kw["files"]["file"]
        if calls["params"]["compression"] == "gzip":
            data = gzip.decompress(data)
        calls["scene_bytes"] = data
        return httpx.Response(
            200, request=httpx.Request("POST", url),
            json={"trajectory": _settling_trajectory(), "n_bodies": 1, "n_steps": 240})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    report = physics_runtime.simulate_scene(
        _rigid_body_scene(tmp_path),
        str(tmp_path / "out"),
        body_path="/World/Body",
        rest_position=[0.0, 0.0, 1.0],
        world_up=[0.0, 0.0, 1.0],
        remote={"base_url": "http://gpu:8000/", "api_key": "k", "timeout": 60.0},
    )

    assert calls["url"] == "http://gpu:8000/physics/simulate"
    assert calls["headers"] == {"Authorization": "Bearer k"}
    assert calls["scene_bytes"].startswith(b"PXR-USDC")
    assert report["executor"] == "remote"
    assert "ok" not in report
    assert report["n_bodies"] == 1
    assert Path(report["recording_usda"]).exists()
    assert report["metrics"]["n_samples"] == 6


def test_remote_physics_resolver_uses_authoritative_backend_pool(monkeypatch):
    from usd_core import physics_runtime, remote_protocol

    observed: list[tuple[str, str, tuple[str, ...], str | None]] = []

    def compatible(url, **kwargs):  # noqa: ANN001
        observed.append(
            (
                url,
                kwargs["required_engine"],
                tuple(kwargs["required_features"]),
                kwargs["api_key"],
            )
        )
        return {
            "engine": "ovrtx",
            "features": [physics_runtime.REMOTE_PHYSICS_FEATURE],
        }

    monkeypatch.setattr(remote_protocol, "check_remote_protocol", compatible)
    remote = physics_runtime.resolve_remote_physics_backend(
        {
            "remote_url": "https://legacy.example.test",
            "remote_api_key": "legacy-secret",
            "remote_timeout": 45,
            "backends": [
                {"url": "https://pool.example.test/", "api_key": "pool-secret"}
            ],
        }
    )

    assert remote == {
        "base_url": "https://pool.example.test",
        "api_key": "pool-secret",
        "timeout": 45.0,
        "verify_version": True,
    }
    assert observed == [
        (
            "https://pool.example.test",
            "ovrtx",
            ("physics-simulate",),
            "pool-secret",
        )
    ]


def test_remote_physics_resolver_honors_disabled_protocol_verification(
    monkeypatch,
):
    from usd_core import physics_runtime, remote_protocol

    monkeypatch.setattr(
        remote_protocol,
        "check_remote_protocol",
        lambda *_args, **_kwargs: pytest.fail(
            "disabled remote verification must not probe the backend"
        ),
    )

    remote = physics_runtime.resolve_remote_physics_backend(
        {
            "backends": [{"url": "https://legacy.example.test"}],
            "remote_verify_version": False,
        },
        verify=True,
    )

    assert remote == {
        "base_url": "https://legacy.example.test",
        "api_key": None,
        "timeout": 300.0,
        "verify_version": False,
    }


def test_simulate_scene_requires_local_or_remote(tmp_path, monkeypatch):
    """No local ovphysx platform and no remote configured → a clear, actionable error."""
    from usd_core import physics_runtime

    monkeypatch.setattr(physics_runtime, "ovphysx_platform_supported", lambda: False)
    with pytest.raises(RuntimeError, match="remote configure"):
        physics_runtime.simulate_scene(
            _rigid_body_scene(tmp_path),
            str(tmp_path / "out"),
            body_path="/World/Body",
            rest_position=[0.0, 0.0, 1.0],
            world_up=[0.0, 0.0, 1.0],
        )


def test_remote_missing_endpoint_is_actionable(tmp_path, monkeypatch):
    """A stale service without /physics/simulate (404) → tells the user to update it."""
    import httpx
    from usd_core import physics_runtime

    monkeypatch.setattr(physics_runtime, "ovphysx_platform_supported", lambda: False)
    _stub_live(monkeypatch)

    def fake_post(self, url, **kw):
        return httpx.Response(404, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    with pytest.raises(RuntimeError, match="/physics/simulate"):
        physics_runtime.simulate_scene(
            _rigid_body_scene(tmp_path),
            str(tmp_path / "out"),
            body_path="/World/Body",
            rest_position=[0.0, 0.0, 1.0],
            world_up=[0.0, 0.0, 1.0],
            remote={"base_url": "http://gpu:8000"},
        )
