# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[2]
app_root = str(APP_ROOT)
while app_root in sys.path:
    sys.path.remove(app_root)
sys.path.insert(0, app_root)

for module_name in list(sys.modules):
    module = sys.modules[module_name]
    module_file = getattr(module, "__file__", "")
    if module_name == "service" or module_name.startswith("service."):
        if not module_file or not Path(module_file).is_relative_to(APP_ROOT):
            sys.modules.pop(module_name, None)

from service.dispatcher import (  # noqa: E402
    OVRTXDispatcher,
    _is_renderer_not_initialized_response,
    parse_gpu_workers,
)


class _FakeResponse:
    def __init__(
        self,
        payload: dict,
        *,
        status_code: int = 200,
        text: str | None = None,
    ):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json_dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            from requests import HTTPError

            raise HTTPError(f"{self.status_code} error", response=self)
        return None

    def json(self) -> dict:
        return self._payload


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload)


def _seed_daemon_telemetry(worker) -> None:
    worker.daemon_pid = 123
    worker.daemon_completed_renders = 11
    worker.daemon_rss_bytes = 4096
    worker.daemon_recycle_count = 3
    worker.daemon_last_recycle_reason = "completed_render_limit"
    worker.daemon_pending_recycle_reason = "rss_limit"


def _assert_daemon_telemetry_cleared(worker) -> None:
    assert worker.daemon_pid is None
    assert worker.daemon_completed_renders is None
    assert worker.daemon_rss_bytes is None
    assert worker.daemon_recycle_count is None
    assert worker.daemon_last_recycle_reason is None
    assert worker.daemon_pending_recycle_reason is None


def test_parse_gpu_workers_accepts_count_and_explicit_ids() -> None:
    assert parse_gpu_workers(None) == []
    assert parse_gpu_workers("") == []
    assert parse_gpu_workers("2") == ["0", "1"]
    assert parse_gpu_workers("-1") == []
    assert parse_gpu_workers("0,1,3") == ["0", "1", "3"]
    assert parse_gpu_workers("GPU-abcd") == ["GPU-abcd"]


def test_ready_workers_property_counts_ready_workers() -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0", "1"])
    dispatcher._workers[0].ready = True

    assert dispatcher.total_workers == 2
    assert dispatcher.ready_workers == 1


def test_health_aggregates_ready_workers() -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0", "1"])
    dispatcher._workers[0].ready = True
    dispatcher._workers[0].renderer_initialized = True
    dispatcher._workers[0].daemon_running = True
    dispatcher._workers[0].status = "healthy"
    dispatcher._workers[1].status = "initializing"

    payload = dispatcher.health()

    assert payload["status"] == "healthy"
    assert payload["gpu_initialized"] is True
    assert payload["ready_workers"] == 1
    assert payload["total_workers"] == 2
    assert payload["workers"][0]["gpu"] == "0"
    assert payload["workers"][1]["ready"] is False


def test_health_reports_initializing_when_process_alive_but_not_ready() -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0"])

    class _LiveProcess:
        def poll(self):
            return None

    dispatcher._workers[0].process = _LiveProcess()

    payload = dispatcher.health()

    assert payload["status"] == "initializing"
    assert payload["gpu_initialized"] is False


def test_health_reports_unhealthy_when_no_worker_is_alive() -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0"])

    payload = dispatcher.health()

    assert payload["status"] == "unhealthy"
    assert payload["gpu_initialized"] is False


def test_render_routes_to_idle_workers(monkeypatch) -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0", "1"])
    for worker in dispatcher._workers:
        worker.ready = True
        worker.status = "healthy"

    entered_first_request = threading.Event()
    release_first_request = threading.Event()
    called_urls: list[str] = []
    lock = threading.Lock()

    def fake_post(url: str, **_kwargs):
        with lock:
            called_urls.append(url)
            call_number = len(called_urls)
        if call_number == 1:
            entered_first_request.set()
            assert release_first_request.wait(timeout=2.0)
        return _FakeResponse({"status": "success", "error": None, "images": {}})

    monkeypatch.setattr("service.dispatcher.requests.post", fake_post)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(dispatcher.render, {"url": "data:,x"})
        assert entered_first_request.wait(timeout=2.0)
        second = executor.submit(dispatcher.render, {"url": "data:,x"})
        release_first_request.set()
        responses = [first.result(timeout=2.0), second.result(timeout=2.0)]

    assert [response["status"] for response in responses] == ["success", "success"]
    assert called_urls == [
        "http://127.0.0.1:8100/render",
        "http://127.0.0.1:8101/render",
    ]


def test_render_times_out_when_no_worker_is_ready() -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0"], queue_timeout_seconds=0.01)

    response = dispatcher.render({"url": "data:,x"})

    assert response["status"] == "exception"
    assert "Timed out waiting for a ready OVRTX worker" in response["error"]


def test_render_does_not_mark_worker_unhealthy_for_client_http_error(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"])
    worker = dispatcher._workers[0]
    worker.ready = True
    worker.status = "healthy"

    def fake_post(url: str, **_kwargs):
        return _FakeResponse(
            {"detail": "bad request"},
            status_code=422,
            text='{"detail":"bad request"}',
        )

    monkeypatch.setattr("service.dispatcher.requests.post", fake_post)

    response = dispatcher.render({"url": "data:,x"})

    assert response["status"] == "exception"
    assert "HTTP 422" in response["error"]
    assert worker.ready is True
    assert worker.status == "healthy"


def test_render_preserves_blank_render_worker_detail(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"])
    worker = dispatcher._workers[0]
    worker.ready = True
    worker.status = "healthy"

    def fake_post(url: str, **_kwargs):
        return _FakeResponse(
            {
                "detail": {
                    "status": "blank_render",
                    "error": "1/1 OVRTX render frames are blank or near-blank.",
                    "warnings": ["blank frame"],
                    "blank_render_frames": [{"frame": 0}],
                }
            },
            status_code=422,
        )

    monkeypatch.setattr("service.dispatcher.requests.post", fake_post)

    response = dispatcher.render({"url": "data:,x"})

    assert response["status"] == "blank_render"
    assert response["images"] == {}
    assert response["blank_render_frames"] == [{"frame": 0}]
    assert worker.ready is True
    assert worker.status == "healthy"


def test_render_treats_fastapi_validation_422_as_client_error(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"])
    worker = dispatcher._workers[0]
    worker.ready = True
    worker.status = "healthy"

    def fake_post(url: str, **_kwargs):
        return _FakeResponse(
            {
                "detail": [
                    {
                        "type": "missing",
                        "loc": ["body", "url"],
                        "msg": "Field required",
                    }
                ]
            },
            status_code=422,
        )

    monkeypatch.setattr("service.dispatcher.requests.post", fake_post)

    response = dispatcher.render({"url": "data:,x"})

    assert response["status"] == "exception"
    assert "HTTP 422" in response["error"]
    assert worker.ready is True
    assert worker.status == "healthy"


def test_render_marks_worker_unhealthy_for_server_http_error(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"])
    worker = dispatcher._workers[0]
    worker.ready = True
    worker.status = "healthy"

    def fake_post(url: str, **_kwargs):
        return _FakeResponse(
            {"detail": "server error"},
            status_code=503,
            text='{"detail":"server error"}',
        )

    monkeypatch.setattr("service.dispatcher.requests.post", fake_post)

    response = dispatcher.render({"url": "data:,x"})

    assert response["status"] == "exception"
    assert worker.ready is False
    assert worker.renderer_initialized is False
    assert worker.daemon_running is False
    assert worker.status == "unhealthy"
    assert worker.unhealthy_since is not None


def test_render_restarts_worker_for_renderer_not_initialized_payload(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"], restart_cooldown_seconds=3600.0)
    worker = dispatcher._workers[0]
    worker.ready = True
    worker.renderer_initialized = True
    worker.daemon_running = True
    _seed_daemon_telemetry(worker)
    worker.status = "healthy"
    starts = 0

    def fake_post(url: str, **_kwargs):
        return _FakeResponse(
            {
                "status": "exception",
                "error": "Renderer not initialized",
                "images": {},
            }
        )

    def fake_start_worker(_worker):
        nonlocal starts
        starts += 1
        _worker.status = "starting"

    monkeypatch.setattr("service.dispatcher.requests.post", fake_post)
    monkeypatch.setattr(dispatcher, "_start_worker", fake_start_worker)

    response = dispatcher.render({"url": "data:,x"})

    assert response == {
        "status": "exception",
        "error": "Renderer not initialized",
        "images": {},
    }
    assert starts == 1
    assert worker.ready is False
    assert worker.renderer_initialized is False
    assert worker.daemon_running is False
    assert worker.in_flight == 0
    assert worker.restart_count == 1
    assert worker.status == "starting"
    assert worker.last_error == "Renderer not initialized"
    assert worker.unhealthy_since is None
    assert worker.restart_requested is False
    _assert_daemon_telemetry_cleared(worker)


def test_mark_worker_exited_clears_daemon_telemetry() -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0"], restart_cooldown_seconds=10.0)
    worker = dispatcher._workers[0]
    worker.ready = True
    worker.renderer_initialized = True
    worker.daemon_running = True
    _seed_daemon_telemetry(worker)
    worker.status = "healthy"

    dispatcher._mark_worker_exited(worker, 17)

    assert worker.ready is False
    assert worker.renderer_initialized is False
    assert worker.daemon_running is False
    assert worker.status == "exited"
    assert worker.last_error == "worker exited with code 17"
    _assert_daemon_telemetry_cleared(worker)


def test_immediate_restart_request_survives_health_timer_update(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"], restart_cooldown_seconds=3600.0)
    worker = dispatcher._workers[0]
    worker.ready = True
    worker.status = "healthy"

    class _LiveProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    process = _LiveProcess()
    worker.process = process
    starts = 0

    def fake_start_worker(_worker):
        nonlocal starts
        starts += 1
        _worker.status = "starting"

    monkeypatch.setattr(dispatcher, "_start_worker", fake_start_worker)

    dispatcher._mark_worker_unhealthy(
        worker,
        "Renderer not initialized",
        restart_immediately=True,
    )

    # Simulate a racing health poll that refreshes the unhealthy timestamp.
    with dispatcher._condition:
        worker.unhealthy_since = time.monotonic()

    dispatcher._restart_unhealthy_worker_if_due(worker)

    assert starts == 1
    assert process.terminated is True
    assert worker.restart_requested is False


def test_renderer_not_initialized_detection_requires_exact_error(caplog) -> None:
    assert _is_renderer_not_initialized_response(
        {
            "status": "exception",
            "error": "Renderer not initialized",
            "images": {},
        }
    )

    caplog.set_level(logging.WARNING, logger="service.dispatcher")
    assert not _is_renderer_not_initialized_response(
        {
            "status": "exception",
            "error": "Renderer not initialized during warmup",
            "images": {},
        }
    )
    assert "did not match the exact restart sentinel" in caplog.text


def test_check_worker_updates_readiness_from_health(monkeypatch) -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0"])

    class _FakeProcess:
        def poll(self):
            return None

    dispatcher._workers[0].process = _FakeProcess()

    def fake_get(url: str, **_kwargs):
        assert url == "http://127.0.0.1:8100/health"
        return _FakeResponse(
            {
                "status": "healthy",
                "gpu_initialized": True,
                "renderer_initialized": True,
                "daemon_running": True,
                "daemon_pid": 123,
                "daemon_completed_renders": 11,
                "daemon_rss_bytes": 4096,
                "daemon_recycle_count": 3,
                "daemon_last_recycle_reason": "completed_render_limit",
                "daemon_pending_recycle_reason": "rss_limit",
            }
        )

    monkeypatch.setattr("service.dispatcher.requests.get", fake_get)

    dispatcher._check_worker(dispatcher._workers[0])

    assert dispatcher._workers[0].ready is True
    assert dispatcher._workers[0].renderer_initialized is True
    assert dispatcher._workers[0].daemon_running is True
    assert dispatcher._workers[0].status == "healthy"
    assert dispatcher._workers[0].health_payload() == {
        "gpu": "0",
        "port": 8100,
        "ready": True,
        "busy": False,
        "in_flight": 0,
        "status": "healthy",
        "renderer_initialized": True,
        "daemon_running": True,
        "daemon_pid": 123,
        "daemon_completed_renders": 11,
        "daemon_rss_bytes": 4096,
        "daemon_recycle_count": 3,
        "daemon_last_recycle_reason": "completed_render_limit",
        "daemon_pending_recycle_reason": "rss_limit",
        "restart_count": 0,
        "last_error": None,
    }


def test_start_worker_sets_worker_environment_and_process(monkeypatch) -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["2"], port_base=8200)
    worker = dispatcher._workers[0]
    captured: dict[str, object] = {}

    class _FakeProcess:
        def poll(self):
            return None

    def fake_popen(cmd, *, env):
        captured["cmd"] = cmd
        captured["env"] = env
        return _FakeProcess()

    monkeypatch.setattr("service.dispatcher.subprocess.Popen", fake_popen)

    dispatcher._start_worker(worker)

    assert worker.process is not None
    assert captured["cmd"][:3] == [sys.executable, "-m", "uvicorn"]
    env = captured["env"]
    assert env["OVRTX_WORKER_MODE"] == "1"
    assert env["OVRTX_WORKER_GPU_INDEX"] == "2"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["NVIDIA_VISIBLE_DEVICES"] == "2"


def test_check_worker_restarts_after_cooldown_once(monkeypatch) -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0"], restart_cooldown_seconds=0.01)
    worker = dispatcher._workers[0]

    class _ExitedProcess:
        returncode = 9

        def poll(self):
            return 9

    worker.process = _ExitedProcess()
    starts = 0

    def fake_start_worker(_worker):
        nonlocal starts
        starts += 1
        _worker.status = "starting"

    monkeypatch.setattr(dispatcher, "_start_worker", fake_start_worker)

    dispatcher._check_worker(worker)
    assert starts == 0
    first_deadline = worker.next_restart_at

    dispatcher._check_worker(worker)
    assert starts == 0
    assert worker.next_restart_at == first_deadline

    import time

    time.sleep(0.02)
    dispatcher._check_worker(worker)

    assert starts == 1


def test_check_worker_restarts_live_unhealthy_worker_after_cooldown(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"], restart_cooldown_seconds=0.01)
    worker = dispatcher._workers[0]

    class _LiveProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    process = _LiveProcess()
    worker.process = process
    starts = 0

    def fake_get(url: str, **_kwargs):
        return _FakeResponse(
            {
                "status": "unhealthy",
                "gpu_initialized": False,
                "renderer_initialized": False,
                "daemon_running": False,
                "error": "warm-up failed",
            }
        )

    def fake_start_worker(_worker):
        nonlocal starts
        starts += 1
        _worker.status = "starting"

    monkeypatch.setattr("service.dispatcher.requests.get", fake_get)
    monkeypatch.setattr(dispatcher, "_start_worker", fake_start_worker)

    dispatcher._check_worker(worker)
    assert starts == 0
    assert worker.unhealthy_since is not None

    import time

    time.sleep(0.02)
    dispatcher._check_worker(worker)

    assert starts == 1
    assert process.terminated is True


def test_check_worker_restarts_live_worker_after_health_exception(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"], restart_cooldown_seconds=0.01)
    worker = dispatcher._workers[0]
    worker.ready = True
    worker.status = "healthy"

    class _LiveProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    process = _LiveProcess()
    worker.process = process
    starts = 0

    def fake_get(url: str, **_kwargs):
        raise TimeoutError("health timed out")

    def fake_start_worker(_worker):
        nonlocal starts
        starts += 1
        _worker.status = "starting"

    monkeypatch.setattr("service.dispatcher.requests.get", fake_get)
    monkeypatch.setattr(dispatcher, "_start_worker", fake_start_worker)

    dispatcher._check_worker(worker)
    assert starts == 0
    assert worker.ready is False
    assert worker.status == "unhealthy"
    assert worker.unhealthy_since is not None

    time.sleep(0.02)
    dispatcher._check_worker(worker)

    assert starts == 1
    assert process.terminated is True


def test_check_worker_defers_unhealthy_restart_while_render_in_flight(monkeypatch):
    dispatcher = OVRTXDispatcher(gpu_ids=["0"], restart_cooldown_seconds=0.01)
    worker = dispatcher._workers[0]
    worker.process = object()
    worker.ready = False
    worker.status = "unhealthy"
    worker.unhealthy_since = time.monotonic() - 1.0
    worker.in_flight = 1
    starts = 0

    def fake_start_worker(_worker):
        nonlocal starts
        starts += 1

    monkeypatch.setattr(dispatcher, "_start_worker", fake_start_worker)

    dispatcher._restart_unhealthy_worker_if_due(worker)

    assert starts == 0
    assert worker.process is not None
    assert worker.status == "unhealthy"


def test_stop_kills_and_reaps_stuck_worker() -> None:
    import asyncio
    import subprocess

    dispatcher = OVRTXDispatcher(gpu_ids=["0"])
    worker = dispatcher._workers[0]

    class _StuckProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.waits = 0

        def poll(self):
            return None if not self.killed else -9

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.waits += 1
            if not self.killed:
                raise subprocess.TimeoutExpired("worker", timeout)
            return -9

    process = _StuckProcess()
    worker.process = process

    asyncio.run(dispatcher.stop())

    assert process.terminated is True
    assert process.killed is True
    assert process.waits >= 2


def test_stop_worker_process_ignores_missing_process() -> None:
    dispatcher = OVRTXDispatcher(gpu_ids=["0"])
    worker = dispatcher._workers[0]

    dispatcher._stop_worker_process(worker)

    assert worker.process is None


def test_stop_process_logs_when_killed_process_never_reaps(caplog) -> None:
    import subprocess

    class _NeverReapsProcess:
        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("worker", timeout)

    caplog.set_level(logging.WARNING, logger="service.dispatcher")

    OVRTXDispatcher._stop_process(
        _NeverReapsProcess(),
        "0",
        timeout_seconds=0.0,
    )

    assert "did not exit after kill" in caplog.text


def test_start_cleans_up_started_workers_on_start_failure(monkeypatch) -> None:
    import asyncio

    dispatcher = OVRTXDispatcher(gpu_ids=["0", "1"])

    class _LiveProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.waited = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True
            return 0

        def kill(self):
            raise AssertionError("terminate should drain without kill")

    process = _LiveProcess()

    def fake_start_worker(worker):
        if worker.spec.gpu_id == "0":
            worker.process = process
            return
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(dispatcher, "_start_worker", fake_start_worker)

    with pytest.raises(RuntimeError, match="spawn failed"):
        asyncio.run(dispatcher.start())

    assert process.terminated is True
    assert process.waited is True
    assert dispatcher._monitor_task is None


def test_monitor_survives_worker_check_exception(monkeypatch):
    import asyncio

    dispatcher = OVRTXDispatcher(gpu_ids=["0"], health_interval_seconds=0.01)
    checks = 0

    def fake_check_worker(_worker):
        nonlocal checks
        checks += 1
        if checks == 1:
            raise OSError("spawn failed")
        dispatcher._stop_event.set()

    monkeypatch.setattr(dispatcher, "_check_worker", fake_check_worker)

    asyncio.run(dispatcher._monitor_workers())

    assert checks >= 2


def test_start_staggers_workers_and_creates_monitor_task(monkeypatch):
    import asyncio

    dispatcher = OVRTXDispatcher(
        gpu_ids=["0", "1"],
        worker_start_stagger_seconds=0.01,
    )
    starts: list[str] = []
    sleeps: list[float] = []

    def fake_start_worker(worker):
        starts.append(worker.spec.gpu_id)

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    async def fake_monitor():
        return None

    monkeypatch.setattr(dispatcher, "_start_worker", fake_start_worker)
    monkeypatch.setattr(dispatcher, "_monitor_workers", fake_monitor)
    monkeypatch.setattr("service.dispatcher.asyncio.sleep", fake_sleep)

    asyncio.run(dispatcher.start())

    assert starts == ["0", "1"]
    assert sleeps == [0.01]
    assert dispatcher._monitor_task is not None


def test_stop_cancels_monitor_task() -> None:
    import asyncio

    async def scenario() -> None:
        dispatcher = OVRTXDispatcher(gpu_ids=["0"])
        started = asyncio.Event()

        async def monitor() -> None:
            started.set()
            await asyncio.sleep(60)

        dispatcher._monitor_task = asyncio.create_task(monitor())
        await started.wait()

        await dispatcher.stop()

        assert dispatcher._monitor_task.cancelled()

    asyncio.run(scenario())


def test_worker_http_error_response_handles_non_json_body() -> None:
    from service.dispatcher import _worker_http_error_response

    class _TextResponse:
        status_code = 404
        text = "not json"

        def json(self):
            raise ValueError("bad json")

    response = _worker_http_error_response(_TextResponse())

    assert response["status"] == "exception"
    assert "HTTP 404: not json" in response["error"]


def test_renderer_not_initialized_detection_rejects_non_matching_payloads() -> None:
    assert _is_renderer_not_initialized_response("not a dict") is False
    assert _is_renderer_not_initialized_response({"status": "success"}) is False
    assert (
        _is_renderer_not_initialized_response(
            {"status": "exception", "error": {"message": "Renderer not initialized"}}
        )
        is False
    )


def test_dispatcher_rejects_worker_port_collision() -> None:
    with pytest.raises(ValueError, match="collides"):
        OVRTXDispatcher(gpu_ids=["0"], port_base=8000, parent_port=8000)


def test_dispatcher_rejects_duplicate_gpu_ids() -> None:
    with pytest.raises(ValueError, match="duplicate GPU id"):
        OVRTXDispatcher(gpu_ids=["0", "0"])
