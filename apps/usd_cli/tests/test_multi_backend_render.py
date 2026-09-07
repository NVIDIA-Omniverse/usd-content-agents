# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-backend parallel remote rendering (the [[render.backends]] pool).

GPU-free unit tests: config pool parsing (TOML array-of-tables, legacy fallback,
dedup), the least-busy scheduler with unhealthy-skip, frame sharding math, the
package-once/upload-in-parallel render_frames path with per-chunk retry,
`remote configure --add` / `remote backends`, and blank_suspect serialization.
Remote services are stubbed via monkeypatched httpx, like the other remote tests.

v4 round (sections at the bottom): throughput/failure-aware demotion — per-slot
rolling upload-throughput records, degraded slots (recent failure/stall or
throughput under 10% of the pool's best) deprioritized by the scheduler and left
out of the frame-shard fan-out, but never starved; upload outcomes feed the shared
pool state through the `_dispatch` hooks.

v5 round (final sections): demotion dynamics — failure demotion expires after a
cooldown (the slot re-enters selection as a canary; a success clears the flag,
another failure re-demotes), the pool-best throughput baseline excludes failing
slots (a fast-but-failing node must not demote every healthy slower peer), and a
canary rule routes one pick per non-demoted streak to the best demoted slot.
Plus: the legacy base64-JSON fallback records the succeeding transport's actual
bytes/duration (not the rejected multipart attempt's), and the single-slot shard
fallback still reserves the slot's in_flight count.
"""

from __future__ import annotations

import base64
import json
import threading
import tomllib
from pathlib import Path

import httpx
import pytest

from usd_core.config import Config, dump_toml, load_config, resolve_render_backends
from usd_core.remote_protocol import PROTOCOL_VERSION
from usd_core.render.base import RenderResult
from usd_core.render.remote import RemoteRenderBackend, _shard_frames

IMG = base64.b64encode(b"png").decode()


@pytest.fixture(autouse=True)
def _fresh_pool_registry():
    """Multi-backend scheduler state persists process-wide by design (the daemon
    builds a new client per command and must share slots); isolate it per test."""
    import usd_core.render.remote as remote_mod

    with remote_mod._POOL_REGISTRY_LOCK:
        remote_mod._POOL_REGISTRY.clear()
    yield
    with remote_mod._POOL_REGISTRY_LOCK:
        remote_mod._POOL_REGISTRY.clear()


# ── config: pool parsing ──────────────────────────────────────────────────────────


def test_toml_array_of_tables_becomes_the_pool(tmp_path, monkeypatch):
    # the exact shape the benchmark launcher writes
    (tmp_path / ".usd-cli").mkdir()
    (tmp_path / ".usd-cli" / "config.toml").write_text(
        '[render]\n'
        'renderer = "remote"\n'
        'remote_url = "http://a:8000"\n'
        'remote_api_key = "k1"\n'
        '[[render.backends]]\n'
        'url = "http://a:8000"\n'
        'api_key = "k1"\n'
        '[[render.backends]]\n'
        'url = "http://b:8000"\n'
        'api_key = "k2"\n')
    monkeypatch.setenv("HOME", str(tmp_path))  # no bleed from ~/.config/usd-cli
    cfg = load_config(start=tmp_path)
    pool = resolve_render_backends(cfg.render)
    assert pool == [{"url": "http://a:8000", "api_key": "k1"},
                    {"url": "http://b:8000", "api_key": "k2"}]


def test_backend_pool_keys_can_be_injected_without_persisting_them(
    tmp_path, monkeypatch
):
    (tmp_path / ".usd-cli").mkdir()
    (tmp_path / ".usd-cli" / "config.toml").write_text(
        '[render]\n'
        'renderer = "remote"\n'
        '[[render.backends]]\n'
        'url = "http://a:8000/"\n'
        '[[render.backends]]\n'
        'url = "http://b:8000"\n'
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(
        "USD_CLI_RENDER_BACKEND_API_KEYS_JSON",
        json.dumps({"http://a:8000": "k1", "http://b:8000": "k2"}),
    )

    cfg = load_config(start=tmp_path)

    assert resolve_render_backends(cfg.render) == [
        {"url": "http://a:8000", "api_key": "k1"},
        {"url": "http://b:8000", "api_key": "k2"},
    ]
    assert "backend_api_keys_json" not in cfg.render


def test_legacy_single_fields_form_a_one_entry_pool():
    render = {"remote_url": "http://a:8000/", "remote_api_key": "k1", "backends": []}
    assert resolve_render_backends(render) == [{"url": "http://a:8000", "api_key": "k1"}]
    assert resolve_render_backends({"remote_url": ""}) == []


def test_pool_dedups_by_url_and_skips_blank_entries():
    render = {"backends": [
        {"url": "http://a:8000", "api_key": "k1"},
        {"url": "http://a:8000/", "api_key": "ignored-duplicate"},  # same after strip
        {"url": "", "api_key": "no-url"},
        {"url": "http://b:8000"},
    ]}
    pool = resolve_render_backends(render)
    assert pool == [{"url": "http://a:8000", "api_key": "k1"},
                    {"url": "http://b:8000", "api_key": ""}]


def test_dump_toml_round_trips_the_backends_array():
    data = {"render": {"renderer": "remote", "remote_url": "http://a:8000",
                       "backends": [{"url": "http://a:8000", "api_key": "k1"},
                                    {"url": "http://b:8000", "api_key": "k2"}]}}
    parsed = tomllib.loads(dump_toml(data))
    assert parsed == data


# ── pool construction + scheduler ─────────────────────────────────────────────────


def _pool_backend(*urls: str, verify: bool = False) -> RemoteRenderBackend:
    return RemoteRenderBackend(
        "", backends=[{"url": u, "api_key": f"key-{i}"} for i, u in enumerate(urls)],
        verify_version=verify)


def test_one_entry_pool_matches_the_legacy_single_backend():
    legacy = RemoteRenderBackend("http://gpu:8000/", api_key="k")
    assert len(legacy._pool) == 1
    assert legacy._base_url == "http://gpu:8000"
    assert legacy._api_key == "k"
    with pytest.raises(ValueError, match="remote_url"):
        RemoteRenderBackend("", backends=[])


def test_scheduler_picks_least_busy_backend():
    backend = _pool_backend("http://a:8000", "http://b:8000", "http://c:8000")
    backend._pool[0].in_flight = 5
    backend._pool[2].in_flight = 5
    slot = backend._acquire_slot()
    assert slot.url == "http://b:8000"
    assert slot.in_flight == 1  # reserved on pick


def test_scheduler_round_robins_between_ties():
    backend = _pool_backend("http://a:8000", "http://b:8000", "http://c:8000")
    # three picks without release: each reservation bumps in_flight, so the
    # least-busy rule + round-robin tiebreak must cover all three backends
    urls = {backend._acquire_slot().url for _ in range(3)}
    assert urls == {"http://a:8000", "http://b:8000", "http://c:8000"}


def test_scheduler_skips_unhealthy_and_excluded():
    backend = _pool_backend("http://a:8000", "http://b:8000")
    backend._pool[0].healthy = False
    slot = backend._acquire_slot()
    assert slot.url == "http://b:8000"
    assert backend._acquire_slot(exclude=slot) is None  # nothing else left


def test_unhealthy_backend_warned_once_then_skipped(monkeypatch, caplog):
    def fake_get(url, **kw):
        if "bad" in url:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, request=httpx.Request("GET", url),
                              json={"protocol_version": PROTOCOL_VERSION})
    monkeypatch.setattr(httpx, "get", fake_get)

    backend = _pool_backend("http://bad:8000", "http://good:8000", verify=True)
    with caplog.at_level("WARNING", logger="usd_core.render.remote"):
        assert [s.url for s in backend._healthy_slots()] == ["http://good:8000"]
        assert [s.url for s in backend._healthy_slots()] == ["http://good:8000"]
    warnings = [r for r in caplog.records if "http://bad:8000" in r.getMessage()]
    assert len(warnings) == 1  # ONE clear warning, not one per render


def test_whole_pool_unhealthy_is_an_error(monkeypatch):
    def fake_get(url, **kw):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(httpx, "get", fake_get)

    backend = _pool_backend("http://a:8000", "http://b:8000", verify=True)
    with pytest.raises(RuntimeError, match="all 2 configured"):
        backend._healthy_slots()


def test_single_renders_rotate_across_the_pool(monkeypatch, tmp_path):
    used = []

    def fake_dispatch_to(self, slot, usdz, send, comp, params, scene):
        used.append(slot.url)
        return [{"camera": c, "image_base64": IMG} for c in params["cameras"]]
    monkeypatch.setattr(RemoteRenderBackend, "_dispatch_to", fake_dispatch_to)
    monkeypatch.setattr(
        RemoteRenderBackend, "_package_usdz",
        classmethod(lambda cls, stage, wd: _touch(wd / "scene_bundle.usdz")))

    backend = _pool_backend("http://a:8000", "http://b:8000")
    backend.render(None, ["/World/cam"], 64, 64, tmp_path / "o1")
    backend.render(None, ["/World/cam"], 64, 64, tmp_path / "o2")
    assert sorted(used) == ["http://a:8000", "http://b:8000"]
    assert all(s.in_flight == 0 for s in backend._pool)  # released after each render


# ── pool persistence: scheduler state survives per-command reconstruction ──────────


def test_pool_state_is_shared_across_backend_constructions():
    """The daemon builds a new RemoteRenderBackend per command; slots (health,
    in_flight, round-robin) must come from a process-wide registry keyed by the
    normalized pool config, not reset to zero on every command."""
    b1 = _pool_backend("http://a:8000", "http://b:8000")
    b2 = _pool_backend("http://a:8000", "http://b:8000")
    assert b1._pool[0] is b2._pool[0] and b1._pool[1] is b2._pool[1]
    b1._pool[1].healthy = False
    assert not b2._pool[1].healthy  # one scheduler, seen by both clients

    # normalization: a trailing slash is the same backend
    b3 = RemoteRenderBackend(
        "", backends=[{"url": "http://a:8000/", "api_key": "key-0"},
                      {"url": "http://b:8000/", "api_key": "key-1"}],
        verify_version=False)
    assert b3._pool[0] is b1._pool[0]

    # a different pool config (here: the verify flag) gets its own state
    b4 = _pool_backend("http://a:8000", "http://b:8000", verify=True)
    assert b4._pool[0] is not b1._pool[0]

    # single-entry pools keep the legacy per-instance behavior (no registry)
    s1 = RemoteRenderBackend("http://solo:8000")
    s2 = RemoteRenderBackend("http://solo:8000")
    assert s1._pool[0] is not s2._pool[0]


def test_renders_rotate_across_fresh_constructions(monkeypatch, tmp_path):
    """Ordinary renders used to ALL start at backend zero because every command's
    fresh construction reset the round-robin cursor."""
    used = []

    def fake_dispatch_to(self, slot, usdz, send, comp, params, scene):
        used.append(slot.url)
        return [{"camera": c, "image_base64": IMG} for c in params["cameras"]]
    monkeypatch.setattr(RemoteRenderBackend, "_dispatch_to", fake_dispatch_to)
    monkeypatch.setattr(
        RemoteRenderBackend, "_package_usdz",
        classmethod(lambda cls, stage, wd: _touch(wd / "scene_bundle.usdz")))

    for i in range(4):  # a NEW backend per render — the daemon's per-command shape
        backend = _pool_backend("http://a:8000", "http://b:8000")
        backend.render(None, ["/World/cam"], 64, 64, tmp_path / f"o{i}")
    assert used.count("http://a:8000") == 2 and used.count("http://b:8000") == 2
    assert all(s.in_flight == 0 for s in backend._pool)


def test_slot_probe_is_single_flight(monkeypatch):
    """Concurrent commands sharing a slot must not race health transitions or probe
    the same backend more than once."""
    import time as time_mod

    import usd_core.remote_protocol as protocol_mod

    calls: list[str] = []
    gauge = {"now": 0, "max": 0}
    glock = threading.Lock()

    def slow_check(url, **kw):
        with glock:
            gauge["now"] += 1
            gauge["max"] = max(gauge["max"], gauge["now"])
        time_mod.sleep(0.05)
        with glock:
            gauge["now"] -= 1
            calls.append(url)
        return 1

    monkeypatch.setattr(protocol_mod, "check_remote_protocol", slow_check)
    backend = _pool_backend("http://a:8000", "http://b:8000", verify=True)
    slot = backend._pool[0]

    threads = [threading.Thread(target=backend._probe_slot, args=(slot,))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert calls == ["http://a:8000"]  # exactly one probe despite 8 racers
    assert gauge["max"] == 1  # never two probes of the same slot in flight
    assert slot.verified and slot.healthy


def test_unhealthy_transition_is_locked_and_reason_sanitized(monkeypatch):
    import usd_core.remote_protocol as protocol_mod

    def failing_check(url, **kw):
        raise RuntimeError("backend said \x1b[31mno\x07")

    monkeypatch.setattr(protocol_mod, "check_remote_protocol", failing_check)
    backend = _pool_backend("http://a:8000", "http://b:8000", verify=True)
    assert backend._probe_slot(backend._pool[0]) is False
    assert not backend._pool[0].healthy
    # untrusted text is escaped before it can reach logs/errors
    assert "\x1b" not in backend._pool[0].reason
    assert "\\x1b" in backend._pool[0].reason

    with pytest.raises(RuntimeError, match="all 2 configured") as ei:
        monkeypatch.setattr(protocol_mod, "check_remote_protocol", failing_check)
        backend._probe_slot(backend._pool[1])
        backend._healthy_slots()
    assert "\x1b" not in str(ei.value)


# ── frame sharding math ───────────────────────────────────────────────────────────


def test_shard_frames_covers_the_range_contiguously():
    frames = [float(f) for f in range(11)]
    chunks = _shard_frames(frames, 3)
    assert len(chunks) == 3
    assert [item for c in chunks for item in c] == frames  # exactly 0:N, in order
    assert all(c == sorted(c) for c in chunks)
    assert max(len(c) for c in chunks) - min(len(c) for c in chunks) <= 1


def test_shard_frames_edge_cases():
    assert _shard_frames([0.0, 1.0], 5) == [[0.0], [1.0]]  # never an empty chunk
    assert _shard_frames([0.0, 1.0, 2.0], 1) == [[0.0, 1.0, 2.0]]  # single backend


# ── sharded render_frames: package once, upload in parallel, assemble in order ────


class _FakeService:
    """Stub fleet: answers /live per URL and renders `frames` chunks per backend."""

    def __init__(self, fail_once_on: str | None = None, fail_always: bool = False):
        self.posts: list[tuple[str, list[float]]] = []
        self.failed: list[str] = []
        self.fail_once_on = fail_once_on
        self.fail_always = fail_always
        self._lock = threading.Lock()

    def get(self, url, **kw):
        return httpx.Response(200, request=httpx.Request("GET", url),
                              json={"protocol_version": PROTOCOL_VERSION,
                                    "engine": "ovrtx", "status": "ready"})

    def post(self, url, **kw):
        params = json.loads(kw["data"]["params"])
        frames = params.get("frames") or []
        with self._lock:
            self.posts.append((url, frames))
            fail = self.fail_always or (self.fail_once_on is not None
                                        and self.fail_once_on in url
                                        and url not in self.failed)
            if fail:
                self.failed.append(url)
                return httpx.Response(500, request=httpx.Request("POST", url),
                                      text="GPU exploded")
        if frames:
            results = [{"camera": params["cameras"][0], "frame": f,
                        "image_base64": IMG} for f in frames]
        else:  # frames-less request: one result per camera (like the real service)
            results = [{"camera": c, "image_base64": IMG}
                       for c in params["cameras"]]
        return httpx.Response(
            200, request=httpx.Request("POST", url), json={"results": results})


def _wire(monkeypatch, service: _FakeService) -> None:
    monkeypatch.setattr(httpx, "get", service.get)
    monkeypatch.setattr(httpx.Client, "get",
                        lambda self, url, **kw: service.get(url, **kw))
    monkeypatch.setattr(httpx.Client, "post",
                        lambda self, url, **kw: service.post(url, **kw))
    # no real USD stage in these tests: packaging is stubbed to a tiny bundle
    monkeypatch.setattr(
        RemoteRenderBackend, "_package_usdz",
        classmethod(lambda cls, stage, wd: _touch(wd / "scene_bundle.usdz")))


def _touch(p: Path) -> Path:
    p.write_bytes(b"usdz")
    return p


def test_render_frames_shards_across_backends(monkeypatch, tmp_path):
    service = _FakeService()
    _wire(monkeypatch, service)
    packagings = []
    orig = RemoteRenderBackend._package_usdz.__func__
    monkeypatch.setattr(RemoteRenderBackend, "_package_usdz", classmethod(
        lambda cls, stage, wd: (packagings.append(1), orig(cls, stage, wd))[1]))

    backend = _pool_backend("http://a:8000", "http://b:8000", verify=True)
    frames = [float(f) for f in range(6)]
    results = backend.render_frames(None, "/World/cam", 64, 64, tmp_path, frames)

    assert len(packagings) == 1  # packaged ONCE, uploaded to both
    uploads = [u for u, _ in service.posts]
    assert sorted(uploads) == ["http://a:8000/render/upload",
                               "http://b:8000/render/upload"]
    # each backend got one contiguous chunk; together they cover 0:6 in order
    sharded = sorted((f for _, f in service.posts), key=lambda c: c[0])
    assert [x for c in sharded for x in c] == frames
    # results reassemble in frame order with one PNG per frame
    assert len(results) == 6
    assert [Path(r.path).name for r in results] == [f"f{i:04d}.png" for i in range(6)]
    assert all(Path(r.path).is_file() for r in results)
    assert [r.renderer_identity["endpoint"] for r in results] == [
        "http://a:8000", "http://a:8000", "http://a:8000",
        "http://b:8000", "http://b:8000", "http://b:8000",
    ]
    assert all(r.renderer_identity["engine"] == "ovrtx" for r in results)
    assert all(r.renderer_identity["status"] == "ready" for r in results)
    assert all(
        r.renderer_identity["protocol_version"] == PROTOCOL_VERSION
        for r in results
    )
    assert all(s.in_flight == 0 for s in backend._pool)


def test_render_frames_single_entry_pool_is_unchanged(monkeypatch, tmp_path):
    """One backend → the classic batched path: one call through _execute, no sharding."""
    calls = {"n": 0}

    def fake_execute(self, stage, params):
        calls["n"] += 1
        return ([{"camera": params["cameras"][0], "frame": f, "image_base64": IMG}
                 for f in params["frames"]], 0.1)
    monkeypatch.setattr(RemoteRenderBackend, "_execute", fake_execute)
    monkeypatch.setattr(RemoteRenderBackend, "_execute_sharded",
                        lambda *a, **kw: pytest.fail("single pool must not shard"))

    backend = RemoteRenderBackend("http://gpu:8000")
    results = backend.render_frames(None, "/World/cam", 64, 64, tmp_path,
                                    [0.0, 1.0, 2.0])
    assert calls["n"] == 1 and len(results) == 3


def test_failed_chunk_retries_once_on_another_backend(monkeypatch, tmp_path):
    service = _FakeService(fail_once_on="http://b:8000")
    _wire(monkeypatch, service)

    backend = _pool_backend("http://a:8000", "http://b:8000", verify=True)
    frames = [0.0, 1.0, 2.0, 3.0]
    results = backend.render_frames(None, "/World/cam", 64, 64, tmp_path, frames)

    assert len(results) == 4  # b's chunk was re-rendered elsewhere
    retried = [(u, f) for u, f in service.posts if "a:8000" in u]
    assert len(retried) == 2  # a's own chunk + b's retried chunk
    assert all(
        result.renderer_identity["endpoint"] == "http://a:8000"
        for result in results
    )
    assert all(s.in_flight == 0 for s in backend._pool)


def test_all_chunk_attempts_failing_surfaces_a_clear_error(monkeypatch, tmp_path):
    service = _FakeService(fail_always=True)
    _wire(monkeypatch, service)

    backend = _pool_backend("http://a:8000", "http://b:8000", verify=True)
    with pytest.raises(RuntimeError, match=r"frame chunk\(s\) failed"):
        backend.render_frames(None, "/World/cam", 64, 64, tmp_path,
                              [0.0, 1.0, 2.0, 3.0])


def test_sharded_render_falls_back_when_only_one_backend_is_healthy(monkeypatch, tmp_path):
    def fake_get(url, **kw):
        if "bad" in url:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, request=httpx.Request("GET", url),
                              json={"protocol_version": PROTOCOL_VERSION})
    service = _FakeService()
    _wire(monkeypatch, service)
    monkeypatch.setattr(httpx, "get", fake_get)

    backend = _pool_backend("http://bad:8000", "http://good:8000", verify=True)
    results = backend.render_frames(None, "/World/cam", 64, 64, tmp_path, [0.0, 1.0])
    assert len(results) == 2
    assert all("good:8000" in u for u, _ in service.posts)


# ── CLI: remote configure --add / remote backends ─────────────────────────────────


@pytest.fixture
def cli_project(tmp_path, monkeypatch):
    pytest.importorskip("typer")
    (tmp_path / ".usd-cli").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))  # isolate ~/.config/usd-cli
    monkeypatch.delenv("OVRTX_API_KEY", raising=False)
    monkeypatch.delenv("USD_CLI_RENDER_REMOTE_API_KEY", raising=False)
    return tmp_path


def _invoke(*args):
    from typer.testing import CliRunner

    from usd_cli.main import app
    return CliRunner().invoke(app, list(args))


def _configure_with_key(monkeypatch, url, key, *options):
    if key is None:
        monkeypatch.delenv("OVRTX_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OVRTX_API_KEY", key)
    return _invoke("remote", "configure", url, *options, "--no-test")


def test_remote_configure_uses_environment_key_and_rejects_api_key_argument(
    cli_project, monkeypatch
):
    cfg_file = cli_project / ".usd-cli" / "config.toml"

    help_result = _invoke("remote", "configure", "--help")
    assert help_result.exit_code == 0, help_result.output
    assert "--api-key" not in help_result.output
    assert "OVRTX_API_KEY" in help_result.output

    rejected = _invoke(
        "remote", "configure", "http://a:8000", "--api-key", "leaked", "--no-test"
    )
    assert rejected.exit_code != 0
    assert "leaked" not in rejected.output

    monkeypatch.setenv("USD_CLI_RENDER_REMOTE_API_KEY", "fallback-key")
    r = _configure_with_key(monkeypatch, "http://a:8000", "k1")
    assert r.exit_code == 0, r.output
    r = _configure_with_key(monkeypatch, "http://b:8000", "k2", "--add")
    assert r.exit_code == 0, r.output

    data = tomllib.loads(cfg_file.read_text())
    # the array was created and seeded with the existing primary first
    assert data["render"]["backends"] == [{"url": "http://a:8000", "api_key": "k1"},
                                          {"url": "http://b:8000", "api_key": "k2"}]
    assert data["render"]["remote_url"] == "http://a:8000"  # primary untouched
    assert data["render"]["renderer"] == "remote"


def test_remote_configure_falls_back_to_existing_render_key_environment(
    cli_project, monkeypatch
):
    monkeypatch.delenv("OVRTX_API_KEY", raising=False)
    monkeypatch.setenv("USD_CLI_RENDER_REMOTE_API_KEY", "fallback-key")

    result = _invoke("remote", "configure", "http://gpu:8000", "--no-test")

    assert result.exit_code == 0, result.output
    data = tomllib.loads((cli_project / ".usd-cli" / "config.toml").read_text())
    assert data["render"]["remote_api_key"] == "fallback-key"


def test_remote_configure_add_without_prior_primary_seeds_it(cli_project, monkeypatch):
    r = _configure_with_key(monkeypatch, "http://solo:8000", "k", "--add")
    assert r.exit_code == 0, r.output
    data = tomllib.loads((cli_project / ".usd-cli" / "config.toml").read_text())
    assert data["render"]["backends"] == [{"url": "http://solo:8000", "api_key": "k"}]
    assert data["render"]["remote_url"] == "http://solo:8000"


def test_plain_configure_repoints_pool_entry_zero_and_legacy_fields_together(
    cli_project, monkeypatch
):
    """With [[render.backends]] present, pool resolution ignores remote_url — a plain
    (non---add) configure must therefore update BOTH: the new URL becomes pool entry
    zero AND the legacy primary, or the CLI probes the new server while renders keep
    going to the old pool. Existing entries are preserved after it."""
    _configure_with_key(monkeypatch, "http://a:8000", "k1")
    _configure_with_key(monkeypatch, "http://b:8000", "k2", "--add")
    r = _configure_with_key(monkeypatch, "http://c:8000", None)  # re-point primary
    assert r.exit_code == 0, r.output
    data = tomllib.loads((cli_project / ".usd-cli" / "config.toml").read_text())
    assert data["render"]["remote_url"] == "http://c:8000"
    assert [b["url"] for b in data["render"]["backends"]] == [
        "http://c:8000", "http://a:8000", "http://b:8000"]
    # the RESOLVED pool now actually contains the newly configured primary
    assert resolve_render_backends(data["render"])[0]["url"] == "http://c:8000"
    # no key was given for c and none was stored for it: the old backend's key must
    # not be carried over to the new server
    assert "api_key" not in data["render"]["backends"][0]
    assert "remote_api_key" not in data["render"]


def test_plain_configure_moves_a_pooled_url_to_entry_zero_preserving_its_key(
    cli_project, monkeypatch
):
    _configure_with_key(monkeypatch, "http://a:8000", "k1")
    _configure_with_key(monkeypatch, "http://b:8000", "k2", "--add")
    r = _configure_with_key(monkeypatch, "http://b:8000", None)  # promote b
    assert r.exit_code == 0, r.output
    data = tomllib.loads((cli_project / ".usd-cli" / "config.toml").read_text())
    assert data["render"]["remote_url"] == "http://b:8000"
    assert data["render"]["remote_api_key"] == "k2"  # b's own stored key, not a's
    assert data["render"]["backends"] == [{"url": "http://b:8000", "api_key": "k2"},
                                          {"url": "http://a:8000", "api_key": "k1"}]


def test_re_add_without_environment_key_preserves_the_stored_key(
    cli_project, monkeypatch
):
    """An idempotent re-add without an environment key preserves its credential."""
    _configure_with_key(monkeypatch, "http://a:8000", "k1")
    _configure_with_key(monkeypatch, "http://b:8000", "k2", "--add")
    r = _configure_with_key(monkeypatch, "http://b:8000", None, "--add")
    assert r.exit_code == 0, r.output
    data = tomllib.loads((cli_project / ".usd-cli" / "config.toml").read_text())
    assert data["render"]["backends"] == [{"url": "http://a:8000", "api_key": "k1"},
                                          {"url": "http://b:8000", "api_key": "k2"}]

    r = _configure_with_key(monkeypatch, "http://b:8000", "k3", "--add")
    assert r.exit_code == 0, r.output
    data = tomllib.loads((cli_project / ".usd-cli" / "config.toml").read_text())
    assert data["render"]["backends"][1] == {"url": "http://b:8000", "api_key": "k3"}


# ── config transaction: locking, atomicity, permissions ───────────────────────────


def test_configure_writes_the_config_mode_0600(cli_project, monkeypatch):
    import stat

    r = _configure_with_key(monkeypatch, "http://a:8000", "s3cret")
    assert r.exit_code == 0, r.output
    cfg_file = cli_project / ".usd-cli" / "config.toml"
    assert stat.S_IMODE(cfg_file.stat().st_mode) == 0o600
    # keys must never touch disk world-readable, even transiently: the publish goes
    # through a 0600 temp sibling + rename, and --add keeps the mode
    r = _configure_with_key(monkeypatch, "http://b:8000", None, "--add")
    assert r.exit_code == 0, r.output
    assert stat.S_IMODE(cfg_file.stat().st_mode) == 0o600


def test_write_config_atomic_survives_a_crash_at_publish(cli_project, monkeypatch):
    """A kill between temp-write and rename must leave the OLD config intact and no
    temp litter — the old write_text could leave a truncated TOML."""
    import os

    from usd_core.config import write_config_atomic

    cfg_file = cli_project / ".usd-cli" / "config.toml"
    write_config_atomic(cfg_file, "a = 1\n")
    assert cfg_file.read_text() == "a = 1\n"

    def crashed(_src, _dst):
        raise OSError("simulated kill at publish")

    monkeypatch.setattr(os, "replace", crashed)
    with pytest.raises(OSError, match="simulated kill"):
        write_config_atomic(cfg_file, "b = 2\n")
    assert cfg_file.read_text() == "a = 1\n"                    # old content intact
    assert not list(cfg_file.parent.glob("*.tmp"))              # no temp litter
    assert not list(cfg_file.parent.glob(".config.toml.*"))


def test_concurrent_add_from_two_processes_keeps_every_entry(cli_project):
    """Two `remote configure --add` transactions racing from SEPARATE processes must
    both land: the unlocked read-modify-write used to last-write-wins away entries.
    Each process appends 4 distinct backends; all 8 (plus their keys) must survive."""
    import subprocess
    import sys

    script = (
        "import sys\n"
        "from usd_cli.main import _append_render_backend\n"
        "tag = sys.argv[1]\n"
        "for i in range(4):\n"
        "    _append_render_backend(f'http://{tag}-{i}:8000', f'key-{tag}-{i}',\n"
        "                           use_global=False)\n"
    )
    procs = [subprocess.Popen([sys.executable, "-c", script, tag],
                              cwd=cli_project,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
             for tag in ("a", "b")]
    for p in procs:
        _out, err = p.communicate(timeout=60)
        assert p.returncode == 0, err.decode()

    data = tomllib.loads((cli_project / ".usd-cli" / "config.toml").read_text())
    got = {b["url"]: b.get("api_key") for b in data["render"]["backends"]}
    want = {f"http://{tag}-{i}:8000": f"key-{tag}-{i}"
            for tag in ("a", "b") for i in range(4)}
    assert got == want  # no entry lost, no key lost


def test_remote_backends_lists_pool_with_probe_results(cli_project, monkeypatch):
    _invoke("remote", "configure", "http://a:8000", "--no-test")
    _invoke("remote", "configure", "http://b:8000", "--add", "--no-test")

    def fake_get(url, **kw):
        if "a:8000" in url:
            return httpx.Response(200, request=httpx.Request("GET", url),
                                  json={"protocol_version": PROTOCOL_VERSION})
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(httpx, "get", fake_get)

    r = _invoke("remote", "backends")
    assert r.exit_code == 0, r.output  # at least one backend usable
    assert f"http://a:8000 — reachable, protocol v{PROTOCOL_VERSION}" in r.output
    assert "http://b:8000 — unreachable" in r.output
    assert "1 of 2 backend(s) usable" in r.output


def test_remote_backends_with_nothing_configured_fails_clearly(cli_project):
    r = _invoke("remote", "backends")
    assert r.exit_code == 1
    assert "no remote render backends configured" in r.output


# ── blank_suspect serialization (session render responses) ────────────────────────


class _StubBackend:
    """Minimal RenderBackend that flags its output as blank_suspect."""

    name = "stub"

    def render(self, stage, cameras, width, height, out_dir, mode="fast",
               names=None, frame=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for i, cam in enumerate(cameras):
            stem = names[i] if names else f"cam{i}"
            p = out_dir / f"{stem}.png"
            p.write_bytes(b"png")
            results.append(RenderResult(path=str(p), camera=cam, width=width,
                                        height=height, backend=self.name,
                                        blank_suspect=True))
        return results


def _cube_session(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from usd_core.session import Session
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Cube.Define(stage, "/World/cube")
    path = tmp_path / "scene.usda"
    stage.GetRootLayer().Export(str(path))
    s = Session(Config(project_dir=tmp_path))
    s.open_stage(str(path))
    return s


def test_session_render_reports_blank_suspects(tmp_path, monkeypatch):
    import usd_core.render as render_pkg
    monkeypatch.setattr(render_pkg, "make_backend", lambda cfg: _StubBackend())

    s = _cube_session(tmp_path)
    resp = s.render(res=[64, 64], output=str(tmp_path / "out"))
    assert resp.ok, resp.data
    suspects = resp.data.get("blank_suspect")
    assert suspects == [r["path"] for r in resp.data["results"]]


def test_session_render_omits_blank_suspect_when_clean(tmp_path, monkeypatch):
    import usd_core.render as render_pkg

    class CleanBackend(_StubBackend):
        def render(self, *a, **kw):
            results = super().render(*a, **kw)
            for r in results:
                r.blank_suspect = False
            return results

    monkeypatch.setattr(render_pkg, "make_backend", lambda cfg: CleanBackend())
    s = _cube_session(tmp_path)
    resp = s.render(res=[64, 64], output=str(tmp_path / "out"))
    assert resp.ok
    assert "blank_suspect" not in resp.data


def test_session_render_reports_executed_ovrtx_settings(tmp_path, monkeypatch):
    import usd_core.render as render_pkg

    class ProvenanceBackend(_StubBackend):
        name = "ovrtx"

        def render(self, *args, **kwargs):
            results = super().render(*args, **kwargs)
            for result in results:
                result.ovrtx_render_mode = "rt2"
                result.ovrtx_num_sensor_updates = 17
                result.active_aov = "LdrColor"
                result.renderer_identity = {
                    "endpoint": "http://renderer.test",
                    "engine": "ovrtx",
                }
            return results

    monkeypatch.setattr(render_pkg, "make_backend", lambda cfg: ProvenanceBackend())
    session = _cube_session(tmp_path)

    response = session.render(res=[64, 64], output=str(tmp_path / "out"))

    assert response.ok
    assert response.summary["ovrtx_render_mode"] == "rt2"
    assert response.summary["ovrtx_num_sensor_updates"] == 17
    assert response.summary["active_aov"] == "LdrColor"
    assert response.data["results"][0]["ovrtx_num_sensor_updates"] == 17
    assert response.data["results"][0]["renderer_identity"] == {
        "endpoint": "http://renderer.test",
        "engine": "ovrtx",
    }


def test_remote_render_preserves_service_ovrtx_settings(tmp_path, monkeypatch):
    backend = RemoteRenderBackend("http://gpu:8000", verify_version=False)

    def fake_execute_scheduled(stage, params):
        del stage
        return (
            [
                {
                    "camera": params["cameras"][0],
                    "image_base64": IMG,
                    "ovrtx_render_mode": "rt1",
                    "ovrtx_num_sensor_updates": 9,
                    "active_aov": "LdrColor",
                }
            ],
            0.1,
        )

    monkeypatch.setattr(backend, "_execute_scheduled", fake_execute_scheduled)

    result = backend.render(None, ["/World/camera"], 64, 64, tmp_path)[0]

    assert result.ovrtx_render_mode == "rt1"
    assert result.ovrtx_num_sensor_updates == 9
    assert result.active_aov == "LdrColor"


# ── v4: throughput/failure-aware demotion (scheduler policy) ───────────────────────


def test_scheduler_prefers_higher_throughput_among_equally_busy_slots():
    backend = _pool_backend("http://fast:8000", "http://slow:8000")
    backend._pool[0].recent_bps.append(100e6)
    backend._pool[1].recent_bps.append(50e6)  # 50% of best — slower, NOT demoted
    assert backend._acquire_slot().url == "http://fast:8000"
    # least-busy still comes first: with fast now holding a request, slow is picked
    assert backend._acquire_slot().url == "http://slow:8000"


def test_scheduler_demotes_a_slot_far_below_the_pools_best_throughput():
    """The benchmark shape: one node moved 18.1 MB in 4.5s, its sibling took 9m12s
    for the same payload — yet least-busy kept feeding the slow one. A slot under
    10% of the pool's best recent throughput must lose to a BUSIER healthy slot."""
    backend = _pool_backend("http://fast:8000", "http://slow:8000")
    payload = 18.1 * 1024 * 1024
    backend._pool[0].recent_bps.append(payload / 4.5)
    backend._pool[1].recent_bps.append(payload / 552.0)  # 9m12s — under 10% of best
    backend._pool[0].in_flight = 2  # busier, still preferred over the degraded slot
    for _ in range(3):
        assert backend._acquire_slot().url == "http://fast:8000"


def test_scheduler_demotes_a_recently_failed_slot_until_its_next_success():
    backend = _pool_backend("http://a:8000", "http://b:8000")
    state = backend._pool_state
    state.record_failure(backend._pool[0])
    backend._pool[1].in_flight = 3  # much busier, still preferred over the failure
    assert backend._acquire_slot().url == "http://b:8000"

    state.record_upload(backend._pool[0], 10 * 1024 * 1024, 2.0)  # success clears it
    assert not backend._pool[0].recent_failure
    assert backend._pool[0].recent_bps  # and contributed a throughput sample
    assert backend._acquire_slot().url == "http://a:8000"  # least busy again


def test_scheduler_never_starves_when_every_slot_is_degraded():
    backend = _pool_backend("http://a:8000", "http://b:8000")
    for s in backend._pool:
        s.recent_failure = True
    slot = backend._acquire_slot()
    assert slot is not None and slot.in_flight == 1  # still schedulable


def test_unknown_throughput_is_not_demoted_and_ranks_first():
    """A slot with no samples yet must be explored (ranked best among ties), never
    read as slow — otherwise the first measured slot would absorb every render."""
    backend = _pool_backend("http://sampled:8000", "http://new:8000")
    backend._pool[0].recent_bps.append(100e6)
    assert backend._acquire_slot().url == "http://new:8000"


def test_throughput_window_is_bounded_and_ignores_bad_samples():
    from usd_core.render.remote import _THROUGHPUT_WINDOW

    backend = _pool_backend("http://a:8000", "http://b:8000")
    state, slot = backend._pool_state, backend._pool[0]
    for i in range(_THROUGHPUT_WINDOW + 5):
        state.record_upload(slot, 1024 * 1024, 1.0 + i)
    assert len(slot.recent_bps) == _THROUGHPUT_WINDOW  # rolling, not unbounded
    before = list(slot.recent_bps)
    state.record_upload(slot, 0, 1.0)     # zero bytes: no sample
    state.record_upload(slot, 1024, 0.0)  # zero duration: no sample
    assert list(slot.recent_bps) == before


# ── v4: upload outcomes feed the shared pool state (via the _dispatch hooks) ──────


def test_successful_uploads_record_per_slot_throughput(monkeypatch, tmp_path):
    service = _FakeService()
    _wire(monkeypatch, service)

    backend = _pool_backend("http://a:8000", "http://b:8000", verify=True)
    results = backend.render_frames(None, "/World/cam", 64, 64, tmp_path,
                                    [0.0, 1.0, 2.0, 3.0])
    assert len(results) == 4
    for slot in backend._pool:
        assert slot.recent_bps, f"{slot.url} recorded no throughput sample"
        assert not slot.recent_failure


def test_failed_upload_marks_the_slot_degraded(monkeypatch, tmp_path):
    service = _FakeService(fail_once_on="http://b:8000")
    _wire(monkeypatch, service)

    backend = _pool_backend("http://a:8000", "http://b:8000", verify=True)
    results = backend.render_frames(None, "/World/cam", 64, 64, tmp_path,
                                    [0.0, 1.0, 2.0, 3.0])
    assert len(results) == 4  # the chunk was retried elsewhere
    b_slot = next(s for s in backend._pool if "b:8000" in s.url)
    a_slot = next(s for s in backend._pool if "a:8000" in s.url)
    assert b_slot.recent_failure  # demoted until its next success
    assert not a_slot.recent_failure and a_slot.recent_bps


def test_single_render_failure_demotes_the_slot_and_retries_on_the_sibling(
        monkeypatch, tmp_path):
    """A failed scheduled render feeds the demotion state AND retries once on a
    sibling backend (round 6): the caller gets a result, the failed slot is
    demoted, and the next render prefers the healthy sibling."""
    service = _FakeService(fail_once_on="http://a:8000")
    _wire(monkeypatch, service)

    backend = _pool_backend("http://a:8000", "http://b:8000", verify=True)
    backend._pool[1].in_flight = 1  # steer the first pick onto a
    items, _dt = backend._execute_scheduled(
        None, {"cameras": ["/World/cam"], "frames": [1.0]})
    assert [u.split("/render")[0] for u, _ in service.posts] == [
        "http://a:8000", "http://b:8000"]
    assert len(items) == 1  # the sibling served the retry (count-validated)
    backend._pool[1].in_flight = 0
    assert next(s for s in backend._pool if "a:8000" in s.url).recent_failure
    # the next scheduled render lands on the healthy sibling, not round-robin's a
    used = []

    def fake_dispatch_to(self, slot, usdz, send, comp, params, scene):
        used.append(slot.url)
        return [{"camera": c, "image_base64": IMG} for c in params["cameras"]]
    monkeypatch.setattr(RemoteRenderBackend, "_dispatch_to", fake_dispatch_to)
    backend.render(None, ["/World/cam"], 64, 64, tmp_path / "o")
    assert used == ["http://b:8000"]


def test_single_render_raises_when_the_retry_sibling_also_fails(monkeypatch):
    """When the retry sibling fails too, the error names both attempts."""
    service = _FakeService(fail_always=True)
    _wire(monkeypatch, service)
    backend = _pool_backend("http://a:8000", "http://b:8000", verify=True)
    with pytest.raises(RuntimeError, match="retry"):
        backend._execute_scheduled(
            None, {"cameras": ["/World/cam"], "frames": [1.0]})


# ── v4: frame sharding avoids degraded slots (never starves) ──────────────────────


def test_sharding_leaves_degraded_backends_out_of_the_fan_out(monkeypatch, tmp_path):
    service = _FakeService()
    _wire(monkeypatch, service)

    backend = _pool_backend("http://a:8000", "http://b:8000", "http://c:8000",
                            verify=True)
    backend._pool[0].recent_bps.append(1e6)
    backend._pool[1].recent_bps.append(9e5)
    backend._pool[2].recent_bps.append(1e3)  # far under 10% of the pool's best
    results = backend.render_frames(None, "/World/cam", 64, 64, tmp_path,
                                    [float(f) for f in range(6)])
    assert len(results) == 6
    assert not any("c:8000" in u for u, _ in service.posts), (
        "a degraded backend must not receive a shard while healthy ones exist")
    assert {u.split("//")[1].split(":")[0] for u, _ in service.posts} == {"a", "b"}


def test_sharding_uses_all_slots_when_every_one_is_degraded(monkeypatch, tmp_path):
    service = _FakeService()
    _wire(monkeypatch, service)

    backend = _pool_backend("http://a:8000", "http://b:8000", verify=True)
    for s in backend._pool:
        s.recent_failure = True  # everything degraded — never starve
    results = backend.render_frames(None, "/World/cam", 64, 64, tmp_path,
                                    [0.0, 1.0, 2.0, 3.0])
    assert len(results) == 4
    assert {u for u, _ in service.posts} == {"http://a:8000/render/upload",
                                             "http://b:8000/render/upload"}
    for s in backend._pool:  # the successful uploads cleared the failure flags
        assert not s.recent_failure


# ── v5: demotion dynamics — cooldown, canary, failure-free baseline ────────────────


def test_failure_demotion_cooldown_and_canary_recovery(monkeypatch):
    """Review item 11: a transient failure must not starve a slot forever. The
    demotion expires after _FAILURE_COOLDOWN_S; the slot then re-enters selection
    as a canary — one success clears the flag, a fresh failure re-demotes it."""
    import usd_core.render.remote as remote_mod

    clock = {"t": 1000.0}
    monkeypatch.setattr(remote_mod, "_monotonic", lambda: clock["t"])
    backend = _pool_backend("http://a:8000", "http://b:8000")
    state = backend._pool_state
    a, b = backend._pool
    state.record_failure(a)

    # inside the cooldown: demoted — the (much busier) sibling still wins
    b.in_flight = 3
    slot = backend._acquire_slot()
    assert slot is b
    backend._release_slot(slot)

    # cooldown elapsed: the slot re-enters selection even though the flag is set
    clock["t"] += remote_mod._FAILURE_COOLDOWN_S + 1
    assert a.recent_failure  # only selection re-admits it; the flag needs a success
    slot = backend._acquire_slot()
    assert slot is a, "a cooled-down slot must get a canary render"
    backend._release_slot(slot)
    b.in_flight = 0

    # the canary render succeeds: the failure flag clears for good
    state.record_upload(a, 10 * 1024 * 1024, 1.0)
    assert not a.recent_failure

    # a fresh failure re-demotes with a fresh cooldown stamp
    state.record_failure(a)
    assert a.failure_at == clock["t"]
    b.in_flight = 3
    assert backend._acquire_slot() is b


def test_pool_best_throughput_baseline_excludes_failed_slots():
    """Review item 11: a historically fast slot that is currently FAILING must not
    set the throughput baseline — that demoted every healthy slower peer and
    immediately nullified the failed slot's own demotion."""
    backend = _pool_backend("http://fastfail:8000", "http://steady:8000")
    fastfail, steady = backend._pool
    fastfail.recent_bps.append(100e6)
    backend._pool_state.record_failure(fastfail)  # fast but failing
    steady.recent_bps.append(1e6)  # 1% of the failed slot's speed — still healthy

    with backend._pool_lock:
        assert backend._demoted(steady, backend._pool) is False
        assert backend._demoted(fastfail, backend._pool) is True
    assert backend._acquire_slot() is steady


def test_canary_routes_one_pick_to_the_best_demoted_slot(monkeypatch):
    """Review item 11: when all traffic keeps landing on non-demoted slots, one
    pick per _CANARY_EVERY streak goes to the best demoted slot so it can earn the
    success that clears its demotion (even inside the failure cooldown)."""
    import usd_core.render.remote as remote_mod

    clock = {"t": 5000.0}
    monkeypatch.setattr(remote_mod, "_monotonic", lambda: clock["t"])
    backend = _pool_backend("http://a:8000", "http://b:8000")
    a, b = backend._pool
    backend._pool_state.record_failure(a)  # demoted for the whole (frozen) test

    picks = []
    for _ in range(remote_mod._CANARY_EVERY + 1):
        slot = backend._acquire_slot()
        picks.append(slot)
        backend._release_slot(slot)
    assert picks[:remote_mod._CANARY_EVERY] == [b] * remote_mod._CANARY_EVERY
    assert picks[-1] is a, "the streak must end in a canary pick of the demoted slot"
    # the streak restarts after the canary: traffic returns to the healthy slot
    assert backend._acquire_slot() is b


# ── v5: the legacy fallback records the transport that actually succeeded ──────────


def test_legacy_fallback_records_the_succeeding_transports_bytes_and_time(
        monkeypatch, tmp_path):
    """Review item 19: when /render/upload is rejected (404) and the base64-JSON
    legacy transport succeeds, the throughput sample must reflect the legacy body
    and ITS duration — not the compressed multipart size over the rejected
    attempt's streaming time (which ranked legacy nodes arbitrarily)."""
    import time as time_mod

    clock = {"t": 0.0}
    monkeypatch.setattr(time_mod, "perf_counter", lambda: clock["t"])

    usdz = tmp_path / "scene_bundle.usdz"
    usdz.write_bytes(b"usdz-bytes" * 64)
    send = tmp_path / "scene_bundle.usdz.gz"
    send.write_bytes(b"gz")  # tiny compressed multipart payload

    class FakeClient:
        def post(self, url, files=None, data=None, json=None, headers=None,
                 timeout=None):
            if url.endswith("/render/upload"):
                reader = files["file"][1]
                reader.read(1)  # streaming starts at t
                clock["t"] += 300.0  # a SLOW, ultimately REJECTED attempt
                while reader.read(1 << 16):
                    pass  # EOF closes the streaming window at t+300
                return httpx.Response(404, request=httpx.Request("POST", url))
            clock["t"] += 2.0  # the legacy transport itself takes 2 seconds
            return httpx.Response(
                200, request=httpx.Request("POST", url),
                json={"results": [{"camera": "/c", "image_base64": IMG}]})

    backend = _pool_backend("http://a:8000", "http://b:8000")
    slot = backend._pool[0]
    child = backend._child(slot)
    items = child._dispatch(FakeClient(), usdz, send, "gzip",
                            {"cameras": ["/c"]}, "scene")
    assert len(items) == 1
    legacy_bytes = len(base64.b64encode(usdz.read_bytes()))
    assert list(slot.recent_bps) == [pytest.approx(legacy_bytes / 2.0)], (
        "the sample must be the legacy body over the legacy duration")
    assert not slot.recent_failure


# ── v5: the single-slot shard fallback still reserves the slot ─────────────────────


def test_single_slot_shard_fanout_reserves_the_slot(monkeypatch, tmp_path):
    """Review item 20: when demotion shrinks the fan-out to ONE slot, the
    single-backend fallback used to run without touching in_flight — concurrent
    scheduling then piled work onto an apparently idle backend."""
    service = _FakeService()
    _wire(monkeypatch, service)

    backend = _pool_backend("http://a:8000", "http://b:8000", verify=True)
    backend._pool_state.record_failure(backend._pool[1])  # only a stays preferred
    seen: dict = {}

    def fake_execute(self, stage, params):
        seen["in_flight_during"] = backend._pool[0].in_flight
        return ([{"camera": params["cameras"][0], "frame": f, "image_base64": IMG}
                 for f in params["frames"]], 0.1)

    monkeypatch.setattr(RemoteRenderBackend, "_execute", fake_execute)
    results = backend.render_frames(None, "/World/cam", 64, 64, tmp_path,
                                    [0.0, 1.0, 2.0])
    assert len(results) == 3
    assert seen["in_flight_during"] == 1, "the fallback must reserve the slot"
    assert all(s.in_flight == 0 for s in backend._pool)
