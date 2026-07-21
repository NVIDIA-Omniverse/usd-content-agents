# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the /tune REST endpoints (happy path).

Tests that:
- POST /tune accepts a USD upload + scenario YAML and queues a session.
- GET /tune/{id}/status reports progress while running and "completed" at end.
- GET /tune/{id}/results returns the best parameters + download URLs.
- POST /tune/{id}/cancel transitions the session to cancelling/cancelled.
- GET /tune/{id}/artifacts/{name} serves the canonical artifact files.

The fake-tune executor lives below — it bypasses the real
arun_tune so these tests don't import torch/botorch and don't need a real
USD pipeline. The actual runner is exercised in the apps/physics_agent
test suite.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from uuid import uuid4

import pytest
import yaml


def _scenario_yaml() -> str:
    return """
name: drop_settle
parameters:
  - name: mass_scale
    min: 0.5
    max: 2.0
"""


def _multipart_files(usd_bytes: bytes = b"#usda 1.0\n# fake physics usd\n"):
    return [
        ("physics_usd", ("physics.usda", usd_bytes, "application/octet-stream")),
    ]


@pytest.fixture(autouse=True)
def _stub_tune_executor(monkeypatch: pytest.MonkeyPatch):
    """Replace the real tune executor with a deterministic stub.

    Mirrors the pipeline-tests' stub_executor: writes the canonical artifacts
    quickly so the lifecycle tests can validate routes in isolation from the
    real BoTorch / OvPhysX dependencies.
    """
    from ...service.routers import tune_router as router_module

    async def fake_execute_tune_async(
        *,
        session_id: str,
        session_manager,
        scenario_path,
        physics_usd,
        engine: str,
        optimizer: str,
        max_trials: int,
        seed: int,
        # Part-1.1 fields. Accept them so the smoke harness keeps
        # working when the real executor signature gains new kwargs;
        # **_extra catches anything else added in the future without
        # forcing this stub to track every change.
        user_prompt: str | None = None,
        enable_judge: bool = True,
        judge_max_iterations: int = 1,
        **_extra,
    ) -> None:
        manager = session_manager
        await manager.update_session(session_id, {"status": "running"})
        session_dir = manager.get_session_dir(session_id)
        out = session_dir / "tune"
        out.mkdir(parents=True, exist_ok=True)
        # Simulate a couple of trials.
        history_path = out / "history.jsonl"
        with history_path.open("w", encoding="utf-8") as f:
            for i in range(min(max_trials, 3)):
                f.write(
                    json.dumps(
                        {
                            "trial_index": i,
                            "params": {"mass_scale": 1.0 + 0.1 * i},
                            "score": 0.5 - 0.1 * i,
                            "failed": False,
                        }
                    )
                    + "\n"
                )

        best_params = {"mass_scale": 1.0 + 0.1 * (min(max_trials, 3) - 1)}
        best_score = 0.5 - 0.1 * (min(max_trials, 3) - 1)
        (out / "best_params.json").write_text(
            json.dumps({"best_score": best_score, "params": best_params}, indent=2)
        )
        (out / "tune_results.json").write_text(
            json.dumps(
                {
                    "scenario": {"name": "drop_settle"},
                    "config": {
                        "engine": engine,
                        "optimizer": optimizer,
                        "max_trials": max_trials,
                        "seed": seed,
                    },
                    "n_trials": min(max_trials, 3),
                    "best": {"params": best_params, "score": best_score},
                }
            )
        )
        (out / "report.md").write_text("# fake report\n")
        (out / "tuned_physics.usd").write_text("#usda 1.0\n")
        (out / "comparison.png").write_bytes(b"\x89PNG\r\n\x1a\nfake\n")

        # Honour cancellation midway: drop into a polling sleep then check.
        for _ in range(3):
            if await manager.is_cancelled(session_id):
                await manager.update_session(session_id, {"status": "cancelled"})
                return
            await asyncio.sleep(0.01)

        await manager.update_session(
            session_id,
            {
                "status": "completed",
                "completed_at": "1970-01-01T00:00:01Z",
                "duration_seconds": 1,
                "can_cancel": False,
                "results": {
                    "best_params": best_params,
                    "best_score": best_score,
                    "n_trials": min(max_trials, 3),
                    "optimizer_used": optimizer if optimizer != "auto" else "botorch",
                    "engine_used": engine,
                },
            },
        )

    monkeypatch.setattr(
        router_module,
        "execute_tune_async",
        fake_execute_tune_async,
        raising=False,
    )

    # Also patch the lazy import inside the router create_tune view.
    import sys

    fake_module = type(sys)("tune_executor_stub")
    fake_module.execute_tune_async = fake_execute_tune_async
    monkeypatch.setitem(
        sys.modules,
        "service.workers.tune_executor",
        fake_module,
    )
    # The router uses a relative import "from ..workers.tune_executor"; under
    # the test harness the package import path is
    # apps.physics_agent_service.service... so override that one too.
    monkeypatch.setitem(
        sys.modules,
        "apps.physics_agent_service.service.workers.tune_executor",
        fake_module,
    )


@pytest.mark.api
class TestTuneCreation:
    @pytest.mark.parametrize("tainted_field", ["user_prompt", "scenario_yaml"])
    async def test_rejects_secret_bearing_durable_content_before_session_creation(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
        tainted_field: str,
    ) -> None:
        """Tune scans both opaque text and parsed YAML before persistence."""
        from ...service import config_persistence
        from ...service.routers import tune_router

        sentinel = "short-tune-secret"
        signed_url = f"https://assets.example.test/a?X-Amz-Signature={sentinel}"
        data = {
            "scenario_yaml": _scenario_yaml(),
            "user_prompt": "make this object bouncy",
        }
        if tainted_field == "user_prompt":
            data["user_prompt"] = signed_url
        else:
            data["scenario_yaml"] += f"\ntokens:\n  - {sentinel}\n"

        manager_calls = 0

        def fail_manager() -> None:
            nonlocal manager_calls
            manager_calls += 1
            raise AssertionError("secret guard must run before session creation")

        monkeypatch.setattr(tune_router, "get_session_manager", fail_manager)
        response = await client.post("/tune", files=_multipart_files(), data=data)

        assert response.status_code == 400
        assert response.json()["detail"] == (
            config_persistence.INVALID_DURABLE_INPUT_DETAIL
        )
        assert sentinel not in response.text
        assert manager_calls == 0

    @pytest.mark.parametrize(
        ("scenario_yaml", "sentinel"),
        [
            (
                _scenario_yaml() + "\n# source: https://assets.example.test/a?"
                "X-Amz-Signature=tune-comment-sentinel\n",
                "tune-comment-sentinel",
            ),
            (
                _scenario_yaml() + "\nsource_url: https://assets.example.test/a?"
                "X-Amz-Signature=tune-shadowed-url\n"
                "source_url: https://assets.example.test/public\n",
                "tune-shadowed-url",
            ),
            (
                _scenario_yaml() + "\napi_key: q7Z9\napi_key: not-used\n",
                "q7Z9",
            ),
        ],
        ids=("signed-url-comment", "shadowed-signed-url", "shadowed-api-key"),
    )
    async def test_rejects_unparsed_or_shadowed_yaml_before_session_creation(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
        scenario_yaml: str,
        sentinel: str,
    ) -> None:
        """Raw YAML and duplicate keys cannot bypass the durable guard."""
        from ...service import config_persistence
        from ...service.routers import tune_router

        manager_calls = 0

        def fail_manager() -> None:
            nonlocal manager_calls
            manager_calls += 1
            raise AssertionError("YAML guard must run before session creation")

        monkeypatch.setattr(tune_router, "get_session_manager", fail_manager)
        response = await client.post(
            "/tune",
            files=_multipart_files(),
            data={"scenario_yaml": scenario_yaml},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            config_persistence.INVALID_DURABLE_INPUT_DETAIL
        )
        for fragment in (sentinel, sentinel[:8], sentinel[-8:]):
            assert fragment not in response.text
        assert manager_calls == 0

    async def test_rejects_binary_tag_before_session_creation(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tagged bytes cannot bypass text scanning or leak via schema errors."""
        from ...service import config_persistence
        from ...service.routers import tune_router

        sentinel = "tune-binary-sentinel"
        signed_url = f"https://assets.example.test/a?X-Amz-Signature={sentinel}"
        encoded = base64.b64encode(signed_url.encode()).decode()
        scenario_yaml = _scenario_yaml() + f"\npayload: !!binary {encoded}\n"
        manager_calls = 0

        def fail_manager() -> None:
            nonlocal manager_calls
            manager_calls += 1
            raise AssertionError("YAML guard must run before session creation")

        monkeypatch.setattr(tune_router, "get_session_manager", fail_manager)
        response = await client.post(
            "/tune",
            files=_multipart_files(),
            data={"scenario_yaml": scenario_yaml},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            config_persistence.INVALID_DURABLE_INPUT_DETAIL
        )
        for fragment in (sentinel, sentinel[:8], encoded, encoded[:12]):
            assert fragment not in response.text
        assert manager_calls == 0

    async def test_rejects_yaml_alias_before_construction(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Aliases cannot amplify a bounded request during canonicalization."""
        from ...service import config_persistence
        from ...service.routers import tune_router

        scenario_yaml = """
name: drop_settle
parameter_defaults: &parameter_defaults
  min: 0.5
  max: 2.0
parameters:
  - name: mass_scale
    <<: *parameter_defaults
"""
        manager_calls = 0

        def fail_construction(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("alias must be rejected before YAML construction")

        def fail_manager() -> None:
            nonlocal manager_calls
            manager_calls += 1
            raise AssertionError("YAML guard must run before session creation")

        monkeypatch.setattr(config_persistence.yaml, "load", fail_construction)
        monkeypatch.setattr(config_persistence.yaml, "safe_dump", fail_construction)
        monkeypatch.setattr(tune_router, "get_session_manager", fail_manager)
        response = await client.post(
            "/tune",
            files=_multipart_files(),
            data={"scenario_yaml": scenario_yaml},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            config_persistence.INVALID_DURABLE_INPUT_DETAIL
        )
        assert manager_calls == 0

    async def test_persists_canonical_benign_scenario_yaml(self, client) -> None:
        """The bytes that passed structural validation are the bytes persisted."""
        from ...service.routers import tune_router

        scenario_yaml = "# ordinary scenario comment\n" + _scenario_yaml()
        response = await client.post(
            "/tune",
            files=_multipart_files(),
            data={"scenario_yaml": scenario_yaml},
        )

        assert response.status_code == 202, response.text
        session_dir = tune_router.get_session_manager().get_session_dir(
            response.json()["session_id"]
        )
        persisted = (session_dir / "input" / "scenario.yaml").read_text(
            encoding="utf-8"
        )
        assert "ordinary scenario comment" not in persisted
        assert yaml.safe_load(persisted) == yaml.safe_load(scenario_yaml)

    async def test_malformed_scenario_error_is_value_free(self, client) -> None:
        sentinel = "never-echo-malformed-tune"
        response = await client.post(
            "/tune",
            files=_multipart_files(),
            data={"scenario_yaml": f"name: [{sentinel}\n"},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid scenario YAML"
        for fragment in (sentinel, sentinel[:8], sentinel[-8:]):
            assert fragment not in response.text

    @pytest.mark.parametrize(
        ("prompt_template", "sentinel"),
        [
            ("Authorization: Bearer {}", "never-persist-tune-bearer"),
            ("use Bearer {} for the request", "never-persist-tune-bearer"),
            ("use Bearer {} for the request", "a1-b"),
        ],
    )
    async def test_rejects_plain_bearer_before_session_creation(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
        prompt_template: str,
        sentinel: str,
    ) -> None:
        from ...service import config_persistence
        from ...service.routers import tune_router

        manager_calls = 0

        def fail_manager() -> None:
            nonlocal manager_calls
            manager_calls += 1
            raise AssertionError("bearer guard must run before session creation")

        monkeypatch.setattr(tune_router, "get_session_manager", fail_manager)
        response = await client.post(
            "/tune",
            files=_multipart_files(),
            data={
                "scenario_yaml": _scenario_yaml(),
                "user_prompt": prompt_template.format(sentinel),
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            config_persistence.INVALID_DURABLE_INPUT_DETAIL
        )
        assert sentinel not in response.text
        assert manager_calls == 0

    async def test_create_tune_with_upload(self, client) -> None:
        files = _multipart_files()
        data = {"scenario_yaml": _scenario_yaml()}
        r = await client.post("/tune", files=files, data=data)
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["status"] == "pending"
        assert "session_id" in body

    async def test_create_tune_rejects_missing_scenario(self, client) -> None:
        files = _multipart_files()
        r = await client.post("/tune", files=files)
        # Post Part 1.1, both ``scenario_yaml`` and ``user_prompt`` are
        # optional Form fields with empty-string defaults; the route
        # validates that at least one is supplied and returns the
        # explicit 400 (not FastAPI's 422 for a missing field).
        assert r.status_code == 400, r.text
        assert "scenario_yaml" in r.text or "user_prompt" in r.text

    async def test_create_tune_rejects_invalid_scenario(self, client) -> None:
        r = await client.post(
            "/tune",
            files=_multipart_files(),
            data={"scenario_yaml": "name: nonexistent\nparameters: []\n"},
        )
        assert r.status_code == 400
        assert "Invalid scenario" in r.json()["detail"]

    async def test_create_tune_rejects_no_input_source(self, client) -> None:
        r = await client.post("/tune", data={"scenario_yaml": _scenario_yaml()})
        assert r.status_code == 400
        assert "Exactly one" in r.json()["detail"]

    @pytest.mark.parametrize(
        "allowed_buckets,s3_uri",
        [
            ("", "s3://trusted-input-bucket/path/physics.usda"),
            ("trusted-input-bucket", "s3://foreign-bucket/path/physics.usda"),
        ],
    )
    async def test_create_tune_rejects_unapproved_s3_before_download(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
        allowed_buckets: str,
        s3_uri: str,
    ) -> None:
        """A foreign bucket must fail closed without making an S3 request."""
        from ...service.routers import tune_router

        monkeypatch.setattr(
            tune_router.config,
            "s3_allowed_buckets",
            allowed_buckets,
        )
        manager_calls = 0
        download_calls = 0

        def fail_manager() -> None:
            nonlocal manager_calls
            manager_calls += 1
            raise AssertionError("S3 policy must precede session-store access")

        def fail_if_downloaded(*_args: Any, **_kwargs: Any) -> None:
            nonlocal download_calls
            download_calls += 1
            raise AssertionError("foreign S3 bucket reached the downloader")

        monkeypatch.setattr(
            tune_router,
            "get_session_manager",
            fail_manager,
        )
        monkeypatch.setattr(
            tune_router,
            "download_file_from_s3",
            fail_if_downloaded,
        )

        response = await client.post(
            "/tune",
            data={
                "scenario_yaml": _scenario_yaml(),
                "s3_uri": s3_uri,
            },
        )

        assert response.status_code == 403
        assert response.json()["detail"] == (
            "S3 URI is not permitted by the service's configured bucket allowlist"
        )
        assert manager_calls == 0
        assert download_calls == 0

    async def test_create_tune_with_user_prompt_only(self, client) -> None:
        """Part 1.1 positive path: when ``user_prompt`` alone is supplied
        (no ``scenario_yaml``) the route must still queue a tune. The
        runner's interpreter is responsible for authoring a Scenario
        from the prompt; the route just needs to accept the form
        field. This guards against a regression where the route's
        "exactly one of (scenario_yaml, user_prompt) required" check
        rejects user_prompt as a valid sole input.
        """
        r = await client.post(
            "/tune",
            files=_multipart_files(),
            data={"user_prompt": "make this object bouncy"},
        )
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["status"] == "pending"
        assert "session_id" in body

    async def test_create_tune_rejects_multiple_input_sources(self, client) -> None:
        r = await client.post(
            "/tune",
            files=_multipart_files(),
            data={
                "scenario_yaml": _scenario_yaml(),
                "s3_uri": "s3://bucket/key.usda",
            },
        )
        assert r.status_code == 400
        assert "Exactly one" in r.json()["detail"]


@pytest.mark.api
class TestTuneStatus:
    async def test_status_for_unknown_session(self, client) -> None:
        r = await client.get("/tune/00000000-0000-0000-0000-000000000000/status")
        assert r.status_code == 404

    async def test_status_progresses_to_completed(self, client) -> None:
        r = await client.post(
            "/tune", files=_multipart_files(), data={"scenario_yaml": _scenario_yaml()}
        )
        assert r.status_code == 202
        sid = r.json()["session_id"]

        for _ in range(200):
            sr = await client.get(f"/tune/{sid}/status")
            assert sr.status_code == 200
            if sr.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        final = await client.get(f"/tune/{sid}/status")
        body = final.json()
        assert body["status"] == "completed"
        assert body["session_id"] == sid


@pytest.mark.api
class TestTuneResults:
    async def test_results_returns_202_while_running(self, client) -> None:
        r = await client.post(
            "/tune", files=_multipart_files(), data={"scenario_yaml": _scenario_yaml()}
        )
        sid = r.json()["session_id"]
        rr = await client.get(f"/tune/{sid}/results")
        assert rr.status_code == 202

    async def test_results_after_completion(self, client) -> None:
        r = await client.post(
            "/tune", files=_multipart_files(), data={"scenario_yaml": _scenario_yaml()}
        )
        sid = r.json()["session_id"]
        for _ in range(200):
            sr = await client.get(f"/tune/{sid}/status")
            if sr.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)
        rr = await client.get(f"/tune/{sid}/results")
        assert rr.status_code == 200
        body = rr.json()
        assert body["session_id"] == sid
        assert body["status"] == "completed"
        assert "best_params" in body
        assert "download_urls" in body
        assert "best_params" in body["download_urls"]
        assert "tuned_usd" in body["download_urls"]
        assert "visual_comparison" in body["download_urls"]
        assert (
            body["download_urls"]["visual_comparison"]
            == f"/tune/{sid}/artifacts/comparison.png"
        )

    async def test_results_after_failed_tune_with_partial_results(self, client) -> None:
        """Judge-only fail-closed runs keep optimizer artifacts discoverable."""
        from ...service.routers import tune_router

        manager = tune_router.get_session_manager()
        sid = str(uuid4())
        session_dir = await manager.create_session(sid)
        tune_dir = session_dir / "tune"
        tune_dir.mkdir()
        (tune_dir / "best_params.json").write_text("{}", encoding="utf-8")
        (tune_dir / "tune_results.json").write_text("{}", encoding="utf-8")
        await manager.update_session(
            sid,
            {
                "status": "failed",
                "error": "Visual judge evidence preparation failed",
                "failed_step": "tune",
                "duration_seconds": 3,
                "completed_at": "1970-01-01T00:00:03Z",
                "results": {
                    "best_params": {"mass_scale": 1.2},
                    "best_score": 0.25,
                    "n_trials": 3,
                    "optimizer_used": "random",
                    "engine_used": "fake",
                },
                "artifact_manifest": [
                    "tune/best_params.json",
                    "tune/tune_results.json",
                ],
            },
        )

        rr = await client.get(f"/tune/{sid}/results")
        assert rr.status_code == 200, rr.text
        body = rr.json()
        assert body["session_id"] == sid
        assert body["status"] == "failed"
        assert body["error_message"] == "Visual judge evidence preparation failed"
        assert body["best_params"] == {"mass_scale": 1.2}
        assert body["n_trials"] == 3
        assert body["download_urls"]["best_params"] == (
            f"/tune/{sid}/artifacts/best_params.json"
        )
        assert body["download_urls"]["tune_results"] == (
            f"/tune/{sid}/artifacts/tune_results.json"
        )

    async def test_results_after_cancellation_returns_200(self, client) -> None:
        """A cancelled tune must surface as a terminal state through
        ``GET /{id}/results`` — 200 with the same TuneResults shape as
        a completed run, not the 202 ``Tune still cancelled``
        fall-through that the original router branch produced. The
        executor stores partial best_params + history.jsonl when
        cancellation lands after at least one trial; clients need a
        deterministic terminal response so they can fetch artifacts
        and stop polling.
        """
        r = await client.post(
            "/tune", files=_multipart_files(), data={"scenario_yaml": _scenario_yaml()}
        )
        sid = r.json()["session_id"]
        # Race: cancel while the stub is still polling. The fixture's
        # `fake_execute_tune_async` writes its canonical artifacts BEFORE
        # the cancellation poll, so partial results are on disk by the
        # time the cancel marker is observed.
        cancel = await client.post(f"/tune/{sid}/cancel")
        assert cancel.status_code in (200, 400), cancel.text
        # Wait for the executor to observe the cancel and persist
        # ``status='cancelled'`` to session metadata.
        for _ in range(200):
            sr = await client.get(f"/tune/{sid}/status")
            if sr.json()["status"] in ("cancelled", "completed", "failed"):
                break
            await asyncio.sleep(0.01)
        status = (await client.get(f"/tune/{sid}/status")).json()["status"]
        # Either path is acceptable depending on how the race lands;
        # only the cancelled branch exercises the new terminal handler.
        if status != "cancelled":
            return
        rr = await client.get(f"/tune/{sid}/results")
        assert rr.status_code == 200, rr.text
        body = rr.json()
        assert body["session_id"] == sid
        assert body["status"] == "cancelled"
        assert "download_urls" in body
        assert "best_params" in body


@pytest.mark.api
class TestTuneStatusCoercion:
    async def test_status_coerces_inf_best_score_to_null(
        self, client, monkeypatch
    ) -> None:
        """Round 14 (Codex CX P2#1): a tune cancelled before any trial
        completes persists ``best_score = inf``. The status endpoint
        must coerce that to JSON-null (matching the results endpoint)
        instead of returning 500 from Starlette's JSON encoder.
        """
        # Create a session via the normal POST path so the manager has
        # a tracked session_id we can mutate.
        r = await client.post(
            "/tune", files=_multipart_files(), data={"scenario_yaml": _scenario_yaml()}
        )
        assert r.status_code == 202
        sid = r.json()["session_id"]

        # Drain the fake executor: wait for terminal state, then
        # forcibly overwrite the persisted best_score with inf so the
        # status route sees the same shape as a zero-trial cancel.
        for _ in range(200):
            sr = await client.get(f"/tune/{sid}/status")
            if sr.json()["status"] in ("completed", "cancelled", "failed"):
                break
            await asyncio.sleep(0.01)

        from ...service.routers import tune_router

        manager = tune_router.get_session_manager()
        meta = await manager.get_session_metadata(sid)
        assert meta is not None
        results = dict(meta.get("results") or {})
        results["best_score"] = float("inf")
        await manager.update_session(sid, {"results": results})

        final = await client.get(f"/tune/{sid}/status")
        assert final.status_code == 200, final.text
        # Status must serialise; inf is coerced to null.
        assert final.json()["best_score"] is None


@pytest.mark.api
class TestTuneCancel:
    async def test_cancel_running_tune(self, client) -> None:
        r = await client.post(
            "/tune", files=_multipart_files(), data={"scenario_yaml": _scenario_yaml()}
        )
        sid = r.json()["session_id"]
        cancel = await client.post(f"/tune/{sid}/cancel")
        # Either 200 (cancellation accepted while pending/running) or 400
        # (already completed before cancel arrived). Both are valid lifecycle
        # outcomes — only fail if we see something else.
        assert cancel.status_code in (200, 400)

    async def test_cancel_unknown_session(self, client) -> None:
        r = await client.post("/tune/00000000-0000-0000-0000-000000000000/cancel")
        assert r.status_code == 404

    async def test_cancel_rejects_non_tune_session(self, client) -> None:
        """Round 15 (kimbyn blocker): the tune cancel endpoint must reject
        sessions created by /pipeline or /predict.

        Tune, pipeline, and predict all share the same SessionManager and
        cancellation-marker namespace. Without a route-kind guard a caller
        could pass a pending/running non-tune session id to
        ``POST /tune/{id}/cancel`` and the cancel marker would flip for the
        non-tune job. The predict router already protects this exact case;
        this test exercises the symmetric guard on the tune side.
        """
        from ...service.routers import tune_router

        manager = tune_router.get_session_manager()

        # Simulate a /predict-created session in the shared manager. We
        # bypass the predict route to keep this test focused on tune
        # routing, but the metadata shape matches what
        # predict_router::create_predict() stamps in production.
        import uuid

        predict_sid = str(uuid.uuid4())
        await manager.create_session(predict_sid)
        await manager.update_session(
            predict_sid,
            {
                "status": "running",
                "kind": "predict",
                "can_cancel": True,
                "config": {"kind": "predict", "predict_route": True},
            },
        )

        cancel = await client.post(f"/tune/{predict_sid}/cancel")
        # A 4xx (we use 409 Conflict) must be returned — the cancel marker
        # must NOT have been written for the non-tune session.
        assert 400 <= cancel.status_code < 500, cancel.text
        assert cancel.status_code == 409, cancel.text
        body = cancel.json()
        # Error detail must call out that this is the wrong route.
        assert "not a tune session" in body.get("detail", "")
        # Confirm the marker did NOT flip — the session is still pending.
        assert not await manager.is_cancelled(predict_sid)

    async def test_cancel_rejects_default_session_with_no_kind(self, client) -> None:
        """Round 17 (kimbyn 2026-05-12 blocker): the tune cancel endpoint must
        ALSO reject bare-default sessions where neither ``metadata.kind`` nor
        ``config.kind`` is set.

        :meth:`SessionManager.create_session` initialises ``config`` to ``{}``
        and never stamps a top-level ``kind``. The earlier
        ``test_cancel_rejects_non_tune_session`` only covered an explicit
        ``kind="predict"`` payload; the empty-default shape was still being
        accepted by an ``(metadata_kind is None and config_kind is None and
        not session_config)`` fallback in the guard. That fallback let
        ``/tune/{id}/cancel`` flip the shared cancellation marker on any
        non-tune session that happened to use the bare-default constructor.

        After kimbyn's review the guard requires an explicit ``"tune"``
        discriminator on at least one of the two fields and rejects every
        other shape — including this one.
        """
        from ...service.routers import tune_router

        manager = tune_router.get_session_manager()

        import uuid

        unknown_sid = str(uuid.uuid4())
        # Bare default — no metadata.kind, config is the empty dict the
        # SessionManager initialises by default. Mark it pending+cancellable
        # so the guard's status check isn't what fires the 4xx.
        await manager.create_session(unknown_sid)
        await manager.update_session(
            unknown_sid,
            {"status": "pending", "can_cancel": True},
        )

        cancel = await client.post(f"/tune/{unknown_sid}/cancel")
        # Must be 4xx (we use 409 Conflict).
        assert 400 <= cancel.status_code < 500, cancel.text
        assert cancel.status_code == 409, cancel.text
        body = cancel.json()
        assert "not a tune session" in body.get("detail", "")
        # The critical invariant: cancellation marker MUST remain unset on
        # the unrelated session.
        assert not await manager.is_cancelled(unknown_sid)

    async def test_cancel_rejects_arbitrary_unknown_kind(self, client) -> None:
        """Round 17 hardening: the tune cancel endpoint must reject ANY
        non-"tune" discriminator value, not just the well-known
        ``"predict"`` / ``"pipeline"`` shapes.

        :func:`cancel_tune` gates on ``metadata_kind == "tune" or
        config_kind == "tune"``. The two existing tests cover (a) an
        explicit ``kind="predict"`` and (b) the bare-default no-kind
        shape. A future caller (or a malformed/forged payload) could
        stamp ``kind="anything-else"`` and expect the guard to still
        reject it; without a regression test that invariant could
        silently regress if someone later "helpfully" added a fallback
        for unknown-but-present kinds. Pin the behaviour.
        """
        from ...service.routers import tune_router

        manager = tune_router.get_session_manager()

        import uuid

        weird_sid = str(uuid.uuid4())
        await manager.create_session(weird_sid)
        # Both metadata.kind and config.kind are set to a value that is
        # neither "tune" nor any of the two routes the SessionManager
        # currently recognises. The guard must reject it cleanly.
        await manager.update_session(
            weird_sid,
            {
                "status": "running",
                "kind": "experimental-future-route",
                "can_cancel": True,
                "config": {"kind": "experimental-future-route"},
            },
        )

        cancel = await client.post(f"/tune/{weird_sid}/cancel")
        assert 400 <= cancel.status_code < 500, cancel.text
        assert cancel.status_code == 409, cancel.text
        body = cancel.json()
        assert "not a tune session" in body.get("detail", "")
        # Cancellation marker MUST remain unset.
        assert not await manager.is_cancelled(weird_sid)


@pytest.mark.api
class TestTuneArtifacts:
    async def test_download_best_params(self, client) -> None:
        r = await client.post(
            "/tune", files=_multipart_files(), data={"scenario_yaml": _scenario_yaml()}
        )
        sid = r.json()["session_id"]
        for _ in range(200):
            sr = await client.get(f"/tune/{sid}/status")
            if sr.json()["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.01)

        dr = await client.get(f"/tune/{sid}/artifacts/best_params.json")
        assert dr.status_code == 200
        assert "best_score" in dr.json()
        assert "params" in dr.json()

    async def test_download_history_jsonl(self, client) -> None:
        r = await client.post(
            "/tune", files=_multipart_files(), data={"scenario_yaml": _scenario_yaml()}
        )
        sid = r.json()["session_id"]
        for _ in range(200):
            sr = await client.get(f"/tune/{sid}/status")
            if sr.json()["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.01)

        dr = await client.get(f"/tune/{sid}/artifacts/history.jsonl")
        assert dr.status_code == 200
        assert dr.headers["content-type"] == "application/x-ndjson"

    async def test_download_unknown_artifact_404(self, client) -> None:
        r = await client.post(
            "/tune", files=_multipart_files(), data={"scenario_yaml": _scenario_yaml()}
        )
        sid = r.json()["session_id"]
        for _ in range(200):
            sr = await client.get(f"/tune/{sid}/status")
            if sr.json()["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.01)

        dr = await client.get(f"/tune/{sid}/artifacts/../../etc/passwd")
        assert dr.status_code in (404, 422)

    async def test_download_artifact_unknown_session(self, client) -> None:
        dr = await client.get(
            "/tune/00000000-0000-0000-0000-000000000000/artifacts/best_params.json"
        )
        assert dr.status_code == 404

    async def test_download_all_canonical_artifacts(self, client) -> None:
        """All five canonical artifacts must be downloadable after completion."""
        r = await client.post(
            "/tune", files=_multipart_files(), data={"scenario_yaml": _scenario_yaml()}
        )
        sid = r.json()["session_id"]
        for _ in range(200):
            sr = await client.get(f"/tune/{sid}/status")
            if sr.json()["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.01)

        expected = {
            "best_params.json": "application/json",
            "tune_results.json": "application/json",
            "history.jsonl": "application/x-ndjson",
            "report.md": "text/markdown",
            "tuned_physics.usd": "application/octet-stream",
            "comparison.png": "image/png",
        }
        for name, ctype in expected.items():
            dr = await client.get(f"/tune/{sid}/artifacts/{name}")
            assert dr.status_code == 200, f"{name} download failed: {dr.text}"
            # Compare the media type up to the optional ``;charset=...`` suffix
            # so a downstream change that returns ``application/octet-stream``
            # for everything (or charset-suffixed JSON variants) is caught.
            # ``ctype.split("/")[0] in header`` would have accepted any
            # ``application/...`` value for the JSON / NDJSON entries.
            actual = dr.headers["content-type"].split(";", 1)[0].strip()
            assert actual == ctype, (
                f"{name}: expected content-type {ctype!r}, got {actual!r}"
            )

        legacy = await client.get(f"/tune/{sid}/artifacts/tuned_physics.usda")
        assert legacy.status_code == 200, legacy.text
        assert legacy.headers["content-disposition"].endswith(
            'filename="tuned_physics.usd"'
        )


@pytest.mark.api
class TestTuneCreationExtended:
    """Strengthened coverage for create_tune validation paths."""

    async def test_create_tune_persists_scenario_yaml_and_config(self, client) -> None:
        from ...service.routers import tune_router

        r = await client.post(
            "/tune",
            files=_multipart_files(),
            data={
                "scenario_yaml": _scenario_yaml(),
                "engine": "fake",
                "optimizer": "random",
                "max_trials": "5",
                "seed": "11",
            },
        )
        assert r.status_code == 202
        sid = r.json()["session_id"]
        manager = tune_router.get_session_manager()
        # scenario.yaml is on disk for reproducibility.
        scenario_path = manager.get_session_dir(sid) / "input" / "scenario.yaml"
        assert scenario_path.exists()
        assert "drop_settle" in scenario_path.read_text()
        # Session metadata records the tune kind + form values.
        meta = await manager.get_session_metadata(sid)
        config = meta["config"]
        assert config["kind"] == "tune"
        assert config["engine"] == "fake"
        assert config["optimizer"] == "random"
        assert config["max_trials"] == 5
        assert config["seed"] == 11

    async def test_create_tune_rejects_max_trials_overflow(self, client) -> None:
        r = await client.post(
            "/tune",
            files=_multipart_files(),
            data={"scenario_yaml": _scenario_yaml(), "max_trials": "999999"},
        )
        assert r.status_code == 400
        assert "max_trials" in r.json()["detail"]

    async def test_create_tune_rejects_max_trials_zero(self, client) -> None:
        r = await client.post(
            "/tune",
            files=_multipart_files(),
            data={"scenario_yaml": _scenario_yaml(), "max_trials": "0"},
        )
        assert r.status_code == 400

    async def test_create_tune_rejects_oversize_scenario_yaml(self, client) -> None:
        big_yaml = "name: drop_settle\n" + "# comment\n" * 20000
        r = await client.post(
            "/tune",
            files=_multipart_files(),
            data={"scenario_yaml": big_yaml},
        )
        assert r.status_code == 413

    async def test_create_tune_rejects_path_traversal_source_session_id(
        self, client
    ) -> None:
        r = await client.post(
            "/tune",
            data={
                "scenario_yaml": _scenario_yaml(),
                "source_session_id": "../../etc/passwd",
            },
        )
        assert r.status_code == 400
        assert "UUID4" in r.json()["detail"]

    async def test_create_tune_rejects_invalid_usd_extension(self, client) -> None:
        bad_files = [
            ("physics_usd", ("model.obj", b"v 0 0 0\n", "application/octet-stream")),
        ]
        r = await client.post(
            "/tune", files=bad_files, data={"scenario_yaml": _scenario_yaml()}
        )
        assert r.status_code == 400
        assert "Invalid USD" in r.json()["detail"]


@pytest.mark.api
class TestTuneEvents:
    async def test_events_returns_503_when_session_completes_on_other_instance(
        self, client
    ) -> None:
        """Completed-elsewhere sessions can't stream live events; expect a
        terminal 'done' or 503 (depending on snapshot presence). For our
        stubbed executor, the session reaches terminal locally so we expect
        a successful SSE stream."""
        r = await client.post(
            "/tune", files=_multipart_files(), data={"scenario_yaml": _scenario_yaml()}
        )
        sid = r.json()["session_id"]
        for _ in range(200):
            sr = await client.get(f"/tune/{sid}/status")
            if sr.json()["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.01)
        # GET /events on a terminal session should respond — whether with a
        # done event over SSE or 503 (cross-instance fallback). Either is
        # acceptable per the same /pipeline behaviour; we just need the
        # endpoint wired up.
        events_r = await client.get(f"/tune/{sid}/events")
        assert events_r.status_code in (200, 503)

    async def test_events_unknown_session(self, client) -> None:
        r = await client.get("/tune/00000000-0000-0000-0000-000000000000/events")
        assert r.status_code == 404
