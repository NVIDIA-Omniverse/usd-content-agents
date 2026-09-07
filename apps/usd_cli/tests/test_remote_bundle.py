# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The remote backend ships a self-contained USDZ bundle (geometry + textures).

Unit test (imports usd_core; run with `PYTHONPATH=src`) — no GPU / no remote service needed.
The full remote render path is exercised separately when USD_CLI_TEST_REMOTE_URL is set.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

from conftest import SPRAY

SERVICE_DIR = Path(__file__).resolve().parents[1] / "apps" / "ovrtx_rendering_api"


def _open_spray():
    from pxr import Usd
    stage = Usd.Stage.Open(str(SPRAY.path))
    if not stage:
        pytest.skip("SPRAY sample asset not available")
    return stage


def test_package_usdz_bundles_textures_and_edits(tmp_path):
    from pxr import UsdGeom
    from usd_core.edit import set_translate
    from usd_core.render.remote import RemoteRenderBackend

    stage = _open_spray()
    set_translate(stage, "/spray_bottle/Geometry/bottle_body", [9.0, 0.0, 0.0])  # in-memory edit

    usdz_path = RemoteRenderBackend._package_usdz(stage, tmp_path)
    raw = usdz_path.read_bytes()

    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
    textures = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg", ".exr"))]
    assert textures, f"USDZ carried no textures: {names}"

    # reopen the bundle: geometry resolves and the in-memory edit survived the round-trip
    from pxr import Usd
    pkg = tmp_path / "roundtrip.usdz"
    pkg.write_bytes(raw)
    reopened = Usd.Stage.Open(str(pkg))
    assert sum(1 for _ in reopened.Traverse()) == 8
    body = reopened.GetPrimAtPath("/spray_bottle/Geometry/bottle_body")
    tx = list(UsdGeom.Xformable(body).GetOrderedXformOps()[0].Get())
    assert tx == [9.0, 0.0, 0.0]

    # the intermediate exported layer is cleaned up; only the bundle (and what the test
    # itself wrote) remains
    assert sorted(f.name for f in tmp_path.iterdir()) == ["roundtrip.usdz", "scene_bundle.usdz"]


def test_prepare_render_input_keeps_usdz_geometry_resolvable(tmp_path):
    # Service side: the stage arrives as a USDZ bundle. Exporting its root layer to a
    # loose .usda leaves the package-internal payload/texture references dangling and
    # the renderer draws a blank frame — the package itself must be handed over instead.
    from pxr import Usd, UsdGeom
    from usd_core.render.base import prepare_render_input
    from usd_core.render.remote import RemoteRenderBackend

    stage = _open_spray()
    pkg = RemoteRenderBackend._package_usdz(stage, tmp_path)

    bundled = Usd.Stage.Open(str(pkg))
    prepared, is_temp = prepare_render_input(bundled, tmp_path / "out")
    assert prepared == pkg
    assert not is_temp  # the package is the original — the backend must not delete it

    reopened = Usd.Stage.Open(str(prepared))
    meshes = [p for p in reopened.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert meshes, "prepared render input lost the bundled geometry"


def test_package_usdz_forwards_camera_only_usdz_without_nesting(tmp_path):
    """Protocol-v2 cameras must not turn a self-contained USDZ into a nested USDZ."""
    from pxr import Usd, UsdGeom
    from usd_core.render.remote import _CAMERA_STRIP_PATHS, RemoteRenderBackend

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = RemoteRenderBackend._package_usdz(_open_spray(), source_dir)
    source_bytes = source.read_bytes()

    stage = Usd.Stage.Open(str(source))
    stage.SetEditTarget(stage.GetSessionLayer())
    camera = UsdGeom.Camera.Define(stage, "/spray_bottle/usd_cam")
    camera.GetPrim().SetCustomDataByKey("usdAuthoredCamera", True)

    render_dir = tmp_path / "render"
    render_dir.mkdir()
    token = _CAMERA_STRIP_PATHS.set(frozenset({"/spray_bottle/usd_cam"}))
    try:
        bundled = RemoteRenderBackend._package_usdz(stage, render_dir)
    finally:
        _CAMERA_STRIP_PATHS.reset(token)

    assert bundled.read_bytes() == source_bytes
    with zipfile.ZipFile(bundled) as archive:
        names = archive.namelist()
    assert not any(name.lower().endswith(".usdz") for name in names)
    reopened = Usd.Stage.Open(str(bundled))
    assert any(prim.IsA(UsdGeom.Mesh) for prim in reopened.Traverse())
    assert not reopened.GetPrimAtPath("/spray_bottle/usd_cam")


def test_package_usdz_preserves_non_camera_session_opinions(tmp_path):
    """The no-nesting path must fail closed when the session carries a real edit."""
    from pxr import Sdf, Usd
    from usd_core.render.remote import RemoteRenderBackend

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = RemoteRenderBackend._package_usdz(_open_spray(), source_dir)

    stage = Usd.Stage.Open(str(source))
    stage.SetEditTarget(stage.GetSessionLayer())
    marker = stage.OverridePrim("/spray_bottle")
    marker.CreateAttribute("textureDemoSessionMarker", Sdf.ValueTypeNames.String).Set(
        "preserved"
    )

    render_dir = tmp_path / "render"
    render_dir.mkdir()
    bundled = RemoteRenderBackend._package_usdz(stage, render_dir)

    assert bundled.read_bytes() != source.read_bytes()
    reopened = Usd.Stage.Open(str(bundled))
    assert (
        reopened.GetPrimAtPath("/spray_bottle")
        .GetAttribute("textureDemoSessionMarker")
        .Get()
        == "preserved"
    )


def test_request_model_requires_a_scene_source():
    sys.path.insert(0, str(SERVICE_DIR))
    from service.models import RenderRequest

    assert RenderRequest(usdz_base64="AA==", cameras=["/c"]).usdz_base64
    assert RenderRequest(usd="#usda 1.0\n", cameras=["/c"]).usd
    with pytest.raises(ValueError):
        RenderRequest(cameras=["/c"])


# ── multipart / streamed binary transport ───────────────────────────────────────


def test_upload_params_model_validates():
    sys.path.insert(0, str(SERVICE_DIR))
    from service.models import RenderUploadParams

    p = RenderUploadParams.model_validate_json(
        json.dumps({"cameras": ["/c"], "compression": "gzip"}))
    assert p.compression == "gzip" and p.image_width == 1024


def test_render_models_accept_frames_batch():
    sys.path.insert(0, str(SERVICE_DIR))
    from service.models import RenderRequest, RenderResultItem, RenderUploadParams

    # frames omitted → None (single default-time render, backward compatible)
    assert RenderRequest(usd="#usda 1.0\n", cameras=["/c"]).frames is None
    # frames provided → carried through
    assert RenderRequest(usd="#usda 1.0\n", cameras=["/c"], frames=[0.0, 1.0, 2.0]).frames == [0, 1, 2]
    assert RenderUploadParams(cameras=["/c"], frames=[0.0, 1.0]).frames == [0, 1]
    # cameras × frames is bounded
    with pytest.raises(ValueError):
        RenderUploadParams(cameras=["/a", "/b"], frames=list(range(200)))
    with pytest.raises(ValueError):
        RenderUploadParams(cameras=["/c"], image_width=9999, image_height=64, mode="fast")
    with pytest.raises(ValueError):
        RenderUploadParams(cameras=[], mode="fast")
    # zstd is a first-class wire codec since the upload-dedup plan's Phase 0
    assert RenderUploadParams(cameras=["/c"], compression="zstd",
                              mode="fast").compression == "zstd"
    with pytest.raises(ValueError):
        RenderUploadParams(cameras=["/c"], compression="br", mode="fast")
    result = RenderResultItem(
        camera="/c",
        image_base64="AA==",
        ovrtx_render_mode="rt2",
        ovrtx_num_sensor_updates=17,
        active_aov="LdrColor",
    )
    assert result.ovrtx_render_mode == "rt2"
    assert result.ovrtx_num_sensor_updates == 17
    assert result.active_aov == "LdrColor"


def test_render_service_models_default_to_quality():
    sys.path.insert(0, str(SERVICE_DIR))
    from service.models import ManifestRenderRequest, RenderRequest, RenderUploadParams

    assert RenderRequest(usd="#usda 1.0\n", cameras=["/c"]).mode == "quality"
    assert RenderUploadParams(cameras=["/c"]).mode == "quality"
    assert ManifestRenderRequest(
        files=[{"path": "scene.usda", "sha256": "0" * 64, "size": 1}],
        root="scene.usda",
        cameras=["/c"],
    ).mode == "quality"


def test_prepare_upload_gzips_only_when_it_helps(tmp_path):
    from usd_core.render.remote import RemoteRenderBackend

    backend = RemoteRenderBackend("http://gpu:8000")
    compressible = tmp_path / "scene_bundle.usdz"
    compressible.write_bytes(b"0" * 1_000_000)
    send, comp = backend._prepare_upload(compressible)
    assert comp == "gzip" and send.name.endswith(".usdz.gz")
    assert send.stat().st_size < compressible.stat().st_size

    incompressible = tmp_path / "noise" / "scene_bundle.usdz"
    incompressible.parent.mkdir()
    incompressible.write_bytes(os.urandom(1_000_000))
    send, comp = backend._prepare_upload(incompressible)
    assert comp == "none" and send == incompressible

    off = RemoteRenderBackend("http://gpu:8000", compress=False)
    send, comp = off._prepare_upload(compressible)
    assert comp == "none" and send == compressible


def test_remote_render_frames_batches_one_upload(tmp_path, monkeypatch):
    """render_frames sends one request with all frames and writes a PNG per frame in order."""
    import base64 as _b64

    from usd_core.render.remote import RemoteRenderBackend

    backend = RemoteRenderBackend("http://gpu:8000")
    calls = {"n": 0}

    def fake_execute(stage, params):
        calls["n"] += 1
        calls["params"] = params
        img = _b64.b64encode(b"png").decode()
        return ([{"camera": params["cameras"][0], "frame": f, "image_base64": img}
                 for f in params["frames"]], 0.5)

    monkeypatch.setattr(backend, "_execute", fake_execute)
    results = backend.render_frames(None, "/World/cam", 320, 240, tmp_path, [0.0, 1.0, 2.0],
                                    mode="fast")
    assert calls["n"] == 1  # ONE upload for all three frames
    assert calls["params"]["frames"] == [0.0, 1.0, 2.0]
    assert len(results) == 3
    assert all(Path(r.path).exists() for r in results)

    # a short result set (service without frames support) is a clear error, not silent
    monkeypatch.setattr(backend, "_execute", lambda s, p: ([], 0.1))
    with pytest.raises(RuntimeError, match="batched"):
        backend.render_frames(None, "/World/cam", 320, 240, tmp_path, [0.0, 1.0], mode="fast")


def test_client_side_upload_limit(tmp_path):
    from usd_core.render.remote import RemoteRenderBackend

    f = tmp_path / "scene_bundle.usdz"
    f.write_bytes(b"x" * 700_000)
    limited = RemoteRenderBackend("http://gpu:8000", max_upload_mb=0.5)
    with pytest.raises(RuntimeError, match="remote_max_upload_mb"):
        limited._check_limit(f)
    RemoteRenderBackend("http://gpu:8000")._check_limit(f)  # default: unlimited, no raise


def test_render_upload_endpoint_accepts_gzipped_binary(monkeypatch):
    """POST /render/upload: multipart binary USDZ + params form field, gzip honored."""
    pytest.importorskip("fastapi")
    pytest.importorskip("python_multipart")
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(SERVICE_DIR))
    monkeypatch.setenv("OVRTX_API_KEY", "test-key")
    import service.main as sm

    captured = {}

    class FakeRenderer:
        is_ready = is_initialized = daemon_running = True

        def render_usdz_bytes(self, cameras, width, height, mode, data, frames=None,
                              camera_defs=None):
            captured.update(cameras=cameras, width=width, height=height,
                            mode=mode, data=data, frames=frames)
            return [{"camera": c, "image_base64": "AA=="} for c in cameras]

        def close(self):
            pass

    async def _no_init():
        return None

    monkeypatch.setattr(sm, "_background_init", _no_init)
    monkeypatch.setattr(sm, "_renderer", FakeRenderer())
    monkeypatch.setattr(sm, "_init_state", "ready")

    with TestClient(sm.app) as client:
        body = gzip.compress(b"PK\x03\x04 fake usdz bytes")
        resp = client.post(
            "/render/upload",
            files={"file": ("scene_bundle.usdz.gz", body, "application/octet-stream")},
            data={"params": json.dumps({"cameras": ["/World/cam"], "image_width": 640,
                                        "image_height": 480, "mode": "fast",
                                        "compression": "gzip"})},
            headers={"Authorization": "Bearer test-key"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["results"][0]["camera"] == "/World/cam"
        assert captured["data"] == b"PK\x03\x04 fake usdz bytes"  # decompressed server-side
        assert (captured["width"], captured["height"]) == (640, 480)
        assert captured["frames"] is None  # no frames → single default-time render

        # frames batch: the animation time codes reach the renderer
        resp = client.post(
            "/render/upload",
            files={"file": ("scene.usdz", b"PK\x03\x04 fake", "application/octet-stream")},
            data={"params": json.dumps({"cameras": ["/World/cam"], "frames": [0.0, 1.0, 2.0]})},
            headers={"Authorization": "Bearer test-key"})
        assert resp.status_code == 200, resp.text
        assert captured["frames"] == [0.0, 1.0, 2.0]

        # bad gzip → 400, not 500
        resp = client.post(
            "/render/upload",
            files={"file": ("x.usdz.gz", b"not gzip", "application/octet-stream")},
            data={"params": json.dumps({"cameras": ["/c"], "compression": "gzip"})},
            headers={"Authorization": "Bearer test-key"})
        assert resp.status_code == 400

        # bad params JSON → 400
        resp = client.post(
            "/render/upload",
            files={"file": ("x.usdz", b"bytes", "application/octet-stream")},
            data={"params": "{"},
            headers={"Authorization": "Bearer test-key"})
        assert resp.status_code == 400

        # auth still enforced
        resp = client.post(
            "/render/upload",
            files={"file": ("x.usdz", b"bytes", "application/octet-stream")},
            data={"params": json.dumps({"cameras": ["/c"]})})
        assert resp.status_code == 401


def test_remote_backend_falls_back_to_legacy_json(tmp_path, monkeypatch):
    """A service without /render/upload (404) still renders via base64 JSON."""
    import base64 as b64
    import httpx
    from usd_core.remote_protocol import PROTOCOL_VERSION
    from usd_core.render.remote import RemoteRenderBackend

    stage = _open_spray()
    calls = []

    def fake_get(self, url, **kw):  # the version handshake: a compatible backend
        return httpx.Response(200, request=httpx.Request("GET", url),
                              json={"status": "alive",
                                    "protocol_version": PROTOCOL_VERSION})

    def fake_post(self, url, **kw):
        calls.append(url)
        if url.endswith("/render/upload"):
            return httpx.Response(404, request=httpx.Request("POST", url))
        payload = kw["json"]
        assert payload["usdz_base64"]  # legacy body carries the bundle inline
        return httpx.Response(
            200, request=httpx.Request("POST", url),
            json={"results": [{"camera": c, "image_base64": b64.b64encode(b"png").decode()}
                              for c in payload["cameras"]]})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    monkeypatch.setattr(httpx.Client, "post", fake_post)
    backend = RemoteRenderBackend("http://gpu:8000")
    results = backend.render(stage, ["/spray_bottle"], 64, 64, tmp_path / "out")
    assert [Path(r.path).name for r in results] == ["spray_bottle.png"]
    assert calls == ["http://gpu:8000/render/upload", "http://gpu:8000/render"]


def test_package_usdz_survives_broken_asset_refs(tmp_path):
    """Regression (task-08 benchmark): one broken absolute texture path anywhere in the
    composed scene — e.g. inside a shared material library the scene merely sublayers —
    made CreateNewUsdzPackage fail and killed every remote render. Packaging now falls
    back to a flattened copy with the unresolvable references stripped."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    from usd_core.render.remote import RemoteRenderBackend

    scene = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene))
    UsdGeom.Mesh.Define(stage, "/World/Mesh")
    # an unrelated, unbound material whose texture file does not exist
    UsdShade.Material.Define(stage, "/World/Looks/Broken")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/Broken/Tex")
    sh.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("/nonexistent/textures/Copper_Brushed_normal.jpg"))
    stage.Save()

    work = tmp_path / "work"
    work.mkdir()
    usdz = RemoteRenderBackend._package_usdz(stage, work)
    assert usdz.is_file() and usdz.stat().st_size > 0
    reopened = Usd.Stage.Open(str(usdz))
    assert reopened.GetPrimAtPath("/World/Mesh").IsValid()


def test_resolve_auto_falls_back_to_configured_remote(monkeypatch):
    """Regression (task-05 benchmark): `--renderer auto` on a CPU-only box claimed no
    renderer was available even though render.remote_url was configured, so the agent
    wrongly waived rendering."""
    import usd_core.render.factory as factory
    from usd_core.config import Config

    monkeypatch.setattr(factory, "_ovrtx_available", lambda: False)
    cfg = Config()
    cfg.render["remote_url"] = "http://gpu-box:8000"
    assert factory._resolve_auto(cfg) == "remote"
    assert factory.resolved_renderer(cfg) == "remote"

    cfg2 = Config()
    cfg2.render["remote_url"] = ""
    with pytest.raises(RuntimeError, match="no OVRTX renderer"):
        factory._resolve_auto(cfg2)
