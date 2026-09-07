# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the content-addressed manifest transport.

  1  client manifest = the packaged USDZ's zip entries hashed per file, with a
     stat-validated sidecar so cached bundles are not re-hashed every render
  2  service blob store: sha verified on write, LRU/TTL eviction, hardlink
     materialization, path-traversal rejection
  3  wire flow: negotiate → PUT only missing blobs → render by manifest;
     409 (evicted mid-flight) re-uploads exactly the missing set and retries once;
     a backend without the endpoints falls back to the bundle upload
  4  a warm re-render after a one-file edit uploads exactly that file
"""

from __future__ import annotations

import base64
import os
import stat
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi import FastAPI, Form, UploadFile
from fastapi.testclient import TestClient

SERVICE_ROOT = Path(__file__).resolve().parents[1] / "apps" / "ovrtx_rendering_api"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from service import cas as service_cas  # noqa: E402
from service import main as service_main  # noqa: E402

from usd_core.render import remote_cas  # noqa: E402
from usd_core.render.remote import RemoteRenderBackend  # noqa: E402

_MB = 1024 * 1024
AUTH = {"Authorization": "Bearer k"}
ENTRIES = {
    "scene.usdc": b"#usda-ish root layer " + b"def Xform 'a' {}\n" * 400,
    "0/tex.png": bytes(range(256)) * 40,  # binary-ish, .png => never compressed
    "0/payload.usdc": b"payload geometry bytes " * 500,
}


def make_usdz(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def blob_count(store: service_cas.BlobStore) -> int:
    return sum(1 for d in store.objects.iterdir() if d.is_dir()
               for _ in d.iterdir())


class FakeRenderer:
    """Stands in for the OVRTX renderer: records calls, returns one PNG per camera."""

    is_ready = True
    is_initialized = True
    daemon_running = True

    def __init__(self):
        self.scene_paths: list[str] = []

    def _items(self, cameras):
        img = base64.b64encode(b"not-a-real-png").decode("ascii")
        return [{"camera": c, "image_base64": img} for c in cameras]

    def render_scene_path(self, cameras, width, height, mode, scene_path,
                          frames=None, camera_defs=None):
        assert Path(scene_path).is_file(), "root layer must be materialized"
        self.scene_paths.append(scene_path)
        return self._items(cameras)

    def render_usdz_bytes(self, cameras, width, height, mode, usdz_bytes,
                          frames=None, camera_defs=None):
        return self._items(cameras)


@pytest.fixture
def svc(tmp_path, monkeypatch):
    """The real service app with a fresh CAS store and a stubbed renderer.
    TestClient without a context manager, so the GPU lifespan never runs."""
    monkeypatch.setenv("OVRTX_API_KEY", "k")
    monkeypatch.setattr(service_main, "_cas_store",
                        service_cas.BlobStore(tmp_path / "cas", max_bytes=1 << 30))
    monkeypatch.setattr(service_main, "_renderer", FakeRenderer())
    monkeypatch.setattr(service_main, "_init_state", "ready")
    return TestClient(service_main.app)


def _cas_backend(**kw) -> RemoteRenderBackend:
    kw.setdefault("verify_version", False)
    kw.setdefault("bundle_cache", False)
    b = RemoteRenderBackend("http://testserver", api_key="k", **kw)
    for s in b._pool:
        s.verified = False
        s.info = {}
    b._backend_info = {"features": ["cas", "zstd"],
                       "max_body_bytes": 70 * _MB, "max_scene_bytes": 560 * _MB}
    return b


# ── 1: client manifest ───────────────────────────────────────────────────────────


def test_manifest_hashes_entries_and_first_entry_is_root(tmp_path):
    import hashlib

    usdz = make_usdz(tmp_path / "b.usdz", ENTRIES)
    m = remote_cas.manifest_for_usdz(usdz)
    assert m["root"] == "scene.usdc"  # USDZ default layer = first archive entry
    assert [f["path"] for f in m["files"]] == list(ENTRIES)
    for f in m["files"]:
        assert f["sha256"] == hashlib.sha256(ENTRIES[f["path"]]).hexdigest()
        assert f["size"] == len(ENTRIES[f["path"]])


def test_manifest_sidecar_reused_and_invalidated(tmp_path, monkeypatch):
    usdz = make_usdz(tmp_path / "b.usdz", ENTRIES)
    first = remote_cas.manifest_for_usdz(usdz)
    assert (tmp_path / "b.usdz.manifest.json").is_file()
    # second call must come from the sidecar: a rehash would blow up here
    monkeypatch.setattr(remote_cas, "_hash_entries",
                        lambda p: (_ for _ in ()).throw(AssertionError("rehashed")))
    assert remote_cas.manifest_for_usdz(usdz) == {"root": first["root"],
                                                  "files": first["files"]}
    monkeypatch.undo()
    # rewriting the bundle (size changes) must invalidate the sidecar
    changed = dict(ENTRIES, **{"scene.usdc": b"different root content"})
    make_usdz(usdz, changed)
    m2 = remote_cas.manifest_for_usdz(usdz)
    assert m2 != first
    assert m2["files"][0]["size"] == len(changed["scene.usdc"])


def test_stage_blob_compresses_text_but_not_png(tmp_path):
    usdz = make_usdz(tmp_path / "b.usdz", ENTRIES)
    with zipfile.ZipFile(usdz) as zf:
        sent, nbytes, comp = remote_cas.stage_blob(zf, "scene.usdc", tmp_path, "gzip")
        assert comp == "gzip" and nbytes < len(ENTRIES["scene.usdc"])
        sent, nbytes, comp = remote_cas.stage_blob(zf, "0/tex.png", tmp_path, "gzip")
        assert comp == "none" and nbytes == len(ENTRIES["0/tex.png"])


# ── 2: service blob store ────────────────────────────────────────────────────────


def test_blobstore_verifies_digest_on_write(tmp_path):
    import hashlib

    store = service_cas.BlobStore(tmp_path, max_bytes=1 << 20)
    data = b"blob bytes"
    sha = hashlib.sha256(data).hexdigest()
    assert store.put(sha, data) is True
    assert store.put(sha, data) is False  # idempotent
    assert store.missing([sha, "0" * 64]) == ["0" * 64]
    with pytest.raises(ValueError, match="digest mismatch"):
        store.put("1" * 64, data)
    assert not store.blob_path("1" * 64).exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission contract")
def test_blobstore_normalizes_private_directory_and_blob_permissions(tmp_path):
    import hashlib

    root = tmp_path / "cas"
    root.mkdir(mode=0o777)
    root.chmod(0o777)
    store = service_cas.BlobStore(root, max_bytes=1 << 20)

    for directory in (store.root, store.objects, store.staging):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700

    data = b"private render input"
    sha = hashlib.sha256(data).hexdigest()
    assert store.put(sha, data) is True
    assert stat.S_IMODE(store.blob_path(sha).parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.blob_path(sha).stat().st_mode) == 0o400

    # Existing stores created by the earlier 0444 contract are tightened on use.
    store.blob_path(sha).chmod(0o444)
    assert store.put(sha, data) is False
    assert stat.S_IMODE(store.blob_path(sha).stat().st_mode) == 0o400


@pytest.mark.skipif(os.name == "nt", reason="symlink setup is POSIX-specific")
def test_blobstore_rejects_a_symlinked_root_without_touching_its_target(tmp_path):
    target = tmp_path / "operator-data"
    target.mkdir()
    survivor = target / "keep.txt"
    survivor.write_text("keep", encoding="utf-8")
    linked_root = tmp_path / "cas"
    linked_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="may not be a symlink"):
        service_cas.BlobStore(linked_root, max_bytes=1 << 20)

    assert survivor.read_text(encoding="utf-8") == "keep"
    assert not (target / "objects").exists()


def test_blobstore_materialize_hardlinks_tree_and_reports_missing(tmp_path):
    import hashlib

    store = service_cas.BlobStore(tmp_path / "cas", max_bytes=1 << 20)
    files = []
    for path, data in ENTRIES.items():
        sha = hashlib.sha256(data).hexdigest()
        store.put(sha, data)
        files.append({"path": path, "sha256": sha, "size": len(data)})
    dst = tmp_path / "stage"
    root = store.materialize(files, "scene.usdc", dst)
    assert root == dst / "scene.usdc"
    for path, data in ENTRIES.items():
        assert (dst / path).read_bytes() == data
    # missing blob → MissingBlobsError with the exact sha, nothing rendered
    gone = files[1]["sha256"]
    store.blob_path(gone).unlink()
    with pytest.raises(service_cas.MissingBlobsError) as err:
        store.materialize(files, "scene.usdc", tmp_path / "stage2")
    assert err.value.missing == [gone]


@pytest.mark.parametrize("bad", ["/abs", "../up", "a/../b", "a//b", "a\\b", ".", ""])
def test_blobstore_rejects_traversal_paths(bad):
    with pytest.raises(ValueError):
        service_cas.validate_rel_path(bad)


def test_blobstore_evicts_oldest_over_quota(tmp_path):
    import hashlib
    import os
    import time

    store = service_cas.BlobStore(tmp_path, max_bytes=250)
    shas = []
    now = time.time()
    for i in range(3):
        data = bytes([i]) * 100
        sha = hashlib.sha256(data).hexdigest()
        store.put(sha, data)
        shas.append(sha)
        # recent but strictly ordered mtimes — old enough for LRU order, fresh
        # enough that the 24h TTL plays no part in this test
        t = now - 300 + i * 100
        os.utime(store.blob_path(sha), (t, t))
    store._evict()
    survivors = [s for s in shas if store.blob_path(s).exists()]
    assert survivors == shas[1:]  # oldest evicted, newest two fit the quota


# ── 3: wire flow against the real service app ────────────────────────────────────


def test_negotiate_upload_render_roundtrip(svc, tmp_path):
    usdz = make_usdz(tmp_path / "b.usdz", ENTRIES)
    m = remote_cas.manifest_for_usdz(usdz)
    resp = svc.post("/render/negotiate", json={"files": m["files"]}, headers=AUTH)
    assert resp.status_code == 200
    assert sorted(resp.json()["missing"]) == sorted(f["sha256"] for f in m["files"])

    for f in m["files"]:
        with zipfile.ZipFile(usdz) as zf:
            body = zf.read(f["path"])
        resp = svc.put(f"/blobs/{f['sha256']}", content=body, headers=AUTH)
        assert resp.status_code == 200 and resp.json()["stored"] is True

    resp = svc.post("/render/negotiate", json={"files": m["files"]}, headers=AUTH)
    assert resp.json()["missing"] == []

    body = {"files": m["files"], "root": m["root"], "cameras": ["/cam_a", "/cam_b"],
            "image_width": 64, "image_height": 64, "mode": "fast"}
    resp = svc.post("/render/manifest", json=body, headers=AUTH)
    assert resp.status_code == 200
    assert [r["camera"] for r in resp.json()["results"]] == ["/cam_a", "/cam_b"]
    # the staging tree is per-render and cleaned up afterwards
    staged = service_main._cas_store.staging
    assert list(staged.iterdir()) == []


def test_render_manifest_answers_409_with_missing_list(svc, tmp_path):
    usdz = make_usdz(tmp_path / "b.usdz", ENTRIES)
    m = remote_cas.manifest_for_usdz(usdz)
    body = {"files": m["files"], "root": m["root"], "cameras": ["/cam"],
            "image_width": 64, "image_height": 64, "mode": "fast"}
    resp = svc.post("/render/manifest", json=body, headers=AUTH)
    assert resp.status_code == 409
    assert sorted(resp.json()["detail"]["missing"]) == sorted(
        f["sha256"] for f in m["files"])


def test_put_blob_rejects_wrong_digest(svc):
    resp = svc.put(f"/blobs/{'a' * 64}", content=b"whatever", headers=AUTH)
    assert resp.status_code == 400
    assert "digest mismatch" in resp.json()["detail"]


def test_manifest_models_reject_traversal_and_alien_root(svc):
    good = {"path": "a.usdc", "sha256": "a" * 64, "size": 1}
    evil = {"path": "../escape.usdc", "sha256": "b" * 64, "size": 1}
    resp = svc.post("/render/negotiate", json={"files": [evil]}, headers=AUTH)
    assert resp.status_code == 422
    body = {"files": [good], "root": "other.usdc", "cameras": ["/cam"]}
    resp = svc.post("/render/manifest", json=body, headers=AUTH)
    assert resp.status_code == 422


def test_live_advertises_cas_and_scene_limit(svc):
    info = svc.get("/live").json()
    assert "cas" in info["features"]
    assert info["max_scene_bytes"] == service_main.MAX_SCENE_BYTES


def test_cas_endpoints_require_auth(svc):
    assert svc.post("/render/negotiate", json={"files": []}).status_code == 401
    assert svc.put(f"/blobs/{'a' * 64}", content=b"x").status_code == 401
    assert svc.post("/render/manifest", json={}).status_code == 401


# ── 4: end-to-end client dispatch ────────────────────────────────────────────────


def _params():
    return {"cameras": ["/cam"], "image_width": 64, "image_height": 64,
            "mode": "fast"}


def test_dispatch_warm_path_uploads_only_the_changed_entry(svc, tmp_path):
    b = _cas_backend()
    store = service_main._cas_store
    usdz = make_usdz(tmp_path / "scene_bundle.usdz", ENTRIES)
    items = b._dispatch(svc, usdz, usdz, "none", _params(), "scene")
    assert len(items) == 1
    cold_blobs = blob_count(store)
    assert cold_blobs == len(ENTRIES)

    # identical re-render: nothing new uploaded
    b._dispatch(svc, usdz, usdz, "none", _params(), "scene")
    assert blob_count(store) == cold_blobs

    # one-entry edit: exactly one new blob lands (the plan's warm-path win)
    changed = dict(ENTRIES, **{"scene.usdc": b"#edited root " + b"x" * 4000})
    usdz2 = make_usdz(tmp_path / "scene_bundle2.usdz", changed)
    items = b._dispatch(svc, usdz2, usdz2, "none", _params(), "scene")
    assert len(items) == 1
    assert blob_count(store) == cold_blobs + 1


def test_cas_blob_staging_survives_launcher_tmpdir_cleanup(
        svc, tmp_path, monkeypatch):
    """CAS extraction must use the pinned work root, not launcher TMPDIR."""
    import shutil

    import usd_core.render.remote as remote_mod

    launcher_dir = tmp_path / ".content-workflow-codex-launcher-repro"
    launcher_dir.mkdir(mode=0o700)
    work_root = tmp_path / "remote-render-work"
    monkeypatch.setenv(
        remote_mod.REMOTE_RENDER_STAGING_ROOT_ENV,
        str(work_root),
    )
    monkeypatch.setattr(
        remote_mod.tempfile,
        "gettempdir",
        lambda: str(launcher_dir),
    )
    staged_roots = []
    original_stage_blob = remote_cas.stage_blob

    def stage_after_launcher_cleanup(zf, member, directory, codec):
        staged_roots.append(Path(directory))
        shutil.rmtree(launcher_dir, ignore_errors=True)
        return original_stage_blob(zf, member, directory, codec)

    monkeypatch.setattr(remote_cas, "stage_blob", stage_after_launcher_cleanup)
    backend = _cas_backend()
    usdz = make_usdz(tmp_path / "scene_bundle.usdz", ENTRIES)

    items = backend._dispatch(
        svc,
        usdz,
        usdz,
        "none",
        _params(),
        "scene",
    )

    assert len(items) == 1
    assert staged_roots
    assert {path.parent for path in staged_roots} == {work_root}
    assert not launcher_dir.exists()
    assert list(work_root.iterdir()) == [], "CAS staging artifacts leaked"


def test_dispatch_recovers_from_mid_flight_eviction(svc, tmp_path, monkeypatch):
    b = _cas_backend()
    store = service_main._cas_store
    usdz = make_usdz(tmp_path / "scene_bundle.usdz", ENTRIES)
    b._dispatch(svc, usdz, usdz, "none", _params(), "scene")
    # evict one blob behind the client's back AND make negotiate lie ("all
    # present"), so only the 409 path can save the render
    victim = remote_cas.manifest_for_usdz(usdz)["files"][2]["sha256"]
    store.blob_path(victim).unlink()
    monkeypatch.setattr(store, "missing", lambda shas: [])
    items = b._dispatch(svc, usdz, usdz, "none", _params(), "scene")
    assert len(items) == 1
    assert store.blob_path(victim).exists()  # re-uploaded by the 409 retry


def test_dispatch_falls_back_when_service_lacks_cas_endpoints(tmp_path):
    """A backend that (wrongly) advertises 'cas' but 404s the endpoints must get
    the bundle upload, not a failed render."""
    import json as json_mod

    stub = FastAPI()
    calls = {"upload": 0}

    @stub.post("/render/upload")
    async def upload(file: UploadFile, params: str = Form(...)):  # noqa: ARG001
        calls["upload"] += 1
        cams = json_mod.loads(params)["cameras"]
        img = base64.b64encode(b"png").decode("ascii")
        return {"results": [{"camera": c, "image_base64": img} for c in cams]}

    client = TestClient(stub)
    b = _cas_backend()
    usdz = make_usdz(tmp_path / "scene_bundle.usdz", ENTRIES)
    items = b._dispatch(client, usdz, usdz, "none", _params(), "scene")
    assert len(items) == 1 and calls["upload"] == 1


def test_dispatch_skips_cas_when_disabled_or_not_advertised(svc):
    for b in (_cas_backend(cas=False), _cas_backend()):
        if b._cas:
            b._backend_info = {"features": ["zstd"]}  # no cas advertised
        assert b._slot_supports_cas() is False


# ── packaging interaction ────────────────────────────────────────────────────────


def test_package_payload_defers_compression_on_cas_ready_fleet(tmp_path, monkeypatch):
    monkeypatch.setattr(
        RemoteRenderBackend, "_package_usdz",
        classmethod(lambda cls, stage, wd: make_usdz(wd / "scene_bundle.usdz",
                                                     ENTRIES)))
    cas_b = _cas_backend()
    (tmp_path / "a").mkdir()
    usdz, send, comp = cas_b._package_payload(object(), tmp_path / "a", "scene")
    assert comp == "none" and send == usdz  # per-blob compression happens at upload
    assert not list(usdz.parent.glob("*.gz")) and not list(usdz.parent.glob("*.zst"))

    plain = _cas_backend()
    plain._backend_info = {}  # unknown fleet → bundle transport w/ compression
    (tmp_path / "b").mkdir()
    usdz, send, comp = plain._package_payload(object(), tmp_path / "b", "scene")
    assert comp == "gzip" and send.name.endswith(".usdz.gz")


def test_pool_feature_requires_every_known_backend(tmp_path):
    b = _cas_backend()
    b._pool[0].verified = True
    b._pool[0].info = {"features": []}  # one node without cas
    assert b._pool_supports_feature("cas") is False
    b._pool[0].info = {"features": ["cas"]}
    assert b._pool_supports_feature("cas") is True


def test_estimate_cas_fails_fast_only_over_double_scene_limit():
    b = _cas_backend()
    b._backend_info["max_scene_bytes"] = 100 * _MB
    b._check_estimate_cas(150 * _MB, 10, "scene")  # 1–2×: warn, proceed
    with pytest.raises(RuntimeError, match="staged-scene limit"):
        b._check_estimate_cas(250 * _MB, 10, "scene")
