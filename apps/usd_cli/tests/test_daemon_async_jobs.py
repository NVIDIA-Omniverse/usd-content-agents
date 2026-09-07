# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async-job surface: render --detach / usd-cli jobs / usd-cli wait (round-8 additions).

Round-8 field note: a `cmd && render --detach` chain lost its printed job id to
an exec yield and the agent re-issued an identical render. Bare `usd-cli wait`
(no id = newest job) is the fix; these are the first tests on the surface.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from conftest import CUBE
from usd_core.config import Config
from usd_server import app as server_app

HEADERS = {"x-usd-cli-token": "t"}


def _app(tmp_path):
    return server_app.build_app(
        Config(
            project_dir=tmp_path,
            server={"allowed_roots": [str(tmp_path), str(CUBE.path.parent)]},
        ),
        token="t",
        instance_id="i",
    )


def _cmd(http, command: str, payload: dict | None = None, session: str = "main") -> dict:
    res = http.post("/cmd", json={"command": command, "payload": payload or {},
                                  "session": session}, headers=HEADERS)
    assert res.status_code == 200, res.text
    return res.json()


def test_wait_without_jobs_says_so(tmp_path):
    with TestClient(_app(tmp_path)) as http:
        out = _cmd(http, "wait", {"job": None, "timeout": 5})
        assert out["ok"] is False
        assert "no detached jobs" in out["issues"][0]["message"]


def test_wait_unknown_id_names_it(tmp_path):
    with TestClient(_app(tmp_path)) as http:
        out = _cmd(http, "wait", {"job": "j99", "timeout": 5})
        assert out["ok"] is False
        assert "unknown job 'j99'" in out["issues"][0]["message"]


def test_detach_then_bare_wait_returns_newest_result(tmp_path):
    with TestClient(_app(tmp_path)) as http:
        assert _cmd(http, "open", {"file": str(CUBE.path)})["ok"] is True
        detached = _cmd(http, "stats", {"detach": True})
        job_id = detached["summary"]["job"]
        assert detached["summary"]["state"] == "running" or detached["summary"]["state"] == "done"
        # bare wait (no id) adopts the newest job
        out = _cmd(http, "wait", {"job": "", "timeout": 30})
        assert out["ok"] is True, out
        assert out["command"] == "stats"
        listed = _cmd(http, "jobs")
        assert any(r["job"] == job_id and r["state"] == "done"
                   for r in listed["data"]["jobs"])


def test_jobs_reports_failed_state_not_done(tmp_path):
    # round 9, task-03: a stalled render surfaced as `j2 done 188.8s` and the
    # agent trusted it — "done" must mean the job SUCCEEDED. A finished job
    # whose envelope is ok=False lists as "failed" with the error, and its
    # elapsed_s freezes at completion instead of aging forever.
    import time
    with TestClient(_app(tmp_path)) as http:
        detached = _cmd(http, "verify",
                        {"file": str(tmp_path / "missing.usda"), "detach": True})
        job_id = detached["summary"]["job"]
        out = _cmd(http, "wait", {"job": job_id, "timeout": 30})
        assert out["ok"] is False

        def _row():
            listed = _cmd(http, "jobs")
            return next(r for r in listed["data"]["jobs"] if r["job"] == job_id)

        row = _row()
        assert row["state"] == "failed"
        assert row.get("error")
        time.sleep(0.3)
        assert _row()["elapsed_s"] == row["elapsed_s"]


def test_save_detach_returns_job_and_completes(tmp_path):
    # round 8: a 7.8-minute synchronous save was polled blind 400+ times;
    # --detach is generic in the dispatch path — prove it for save
    import shutil
    with TestClient(_app(tmp_path)) as http:
        work = tmp_path / "scene.usda"
        shutil.copyfile(CUBE.path, work)
        assert _cmd(http, "open", {"file": str(work)})["ok"] is True
        detached = _cmd(http, "save", {"detach": True})
        assert detached["summary"]["job"]
        out = _cmd(http, "wait", {"job": "", "timeout": 30})
        assert out["ok"] is True, out
        assert out["command"] == "save"


def test_detached_job_revalidates_policy_at_execution(monkeypatch, tmp_path):
    calls = 0
    original = server_app._validate_request_policy

    def changing_policy(config, command, payload, *, shared, session=None):
        nonlocal calls
        if command == "stats":
            calls += 1
            if calls > 1:
                raise ValueError("path changed outside server.allowed_roots")
        return original(
            config,
            command,
            payload,
            shared=shared,
            session=session,
        )

    monkeypatch.setattr(server_app, "_validate_request_policy", changing_policy)
    with TestClient(_app(tmp_path)) as http:
        detached = _cmd(http, "stats", {"detach": True})
        result = _cmd(
            http,
            "wait",
            {"job": detached["summary"]["job"], "timeout": 30},
        )
    assert result["ok"] is False
    assert "allowed_roots" in result["issues"][0]["message"]
    assert calls >= 2
