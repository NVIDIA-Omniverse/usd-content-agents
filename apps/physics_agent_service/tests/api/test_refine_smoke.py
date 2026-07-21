# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the /refine REST endpoints."""

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
metric: settle_distance
target:
  drop_height_m: 0.5
  duration_s: 2.0
  gravity: -9.81
parameters:
  - name: mass_scale
    min: 0.5
    max: 2.0
  - name: restitution
    min: 0.4
    max: 0.95
"""


def _multipart_files(usd_bytes: bytes = b"#usda 1.0\n# fake physics usd\n"):
    return [
        ("physics_usd", ("physics.usda", usd_bytes, "application/octet-stream")),
    ]


async def _create_non_refine_session(kind: str = "tune") -> str:
    from ...service.routers import refine_router

    manager = refine_router.get_session_manager()
    sid = str(uuid4())
    await manager.create_session(sid)
    await manager.update_session(
        sid,
        {
            "status": "running",
            "kind": kind,
            "can_cancel": True,
            "config": {"kind": kind},
        },
    )
    return sid


@pytest.fixture(autouse=True)
def _stub_refine_executor(monkeypatch: pytest.MonkeyPatch):
    async def fake_execute_refine_async(
        *,
        session_id: str,
        session_manager,
        scenario_path,
        physics_usd,
        user_prompt: str,
        engine: str,
        optimizer: str,
        max_trials: int,
        seed: int,
        max_iterations: int,
        score_threshold: float,
        **_extra,
    ) -> None:
        manager = session_manager
        await manager.update_session(session_id, {"status": "running"})
        session_dir = manager.get_session_dir(session_id)
        out = session_dir / "refine"
        final = out / "final"
        final.mkdir(parents=True, exist_ok=True)

        best_params = {"mass_scale": 1.2, "restitution": 0.9}
        iteration = {
            "iteration": 1,
            "iteration_dir": str(out / "iter_1"),
            "judge_decision": "approve",
            "judge_score": 0.96,
            "judge_reasoning": "stubbed",
            "best_score": 0.12,
            "n_trials": min(max_trials, 3),
            "metric_name": "settle_distance",
            "metric_value": 0.12,
            "cancelled": False,
            "error": None,
        }
        results = {
            "termination_reason": "approved",
            "iteration_count": 1,
            "final_iteration": 1,
            "final_judge_score": 0.96,
            "final_best_params": best_params,
            "iterations": [iteration],
            "final_dir": str(final),
            "output_dir": str(out),
        }

        (out / "refine_summary.json").write_text(json.dumps(results), encoding="utf-8")
        (final / "scenario.yaml").write_text(
            scenario_path.read_text(), encoding="utf-8"
        )
        (final / "best_params.json").write_text(
            json.dumps({"params": best_params, "best_score": 0.12}),
            encoding="utf-8",
        )
        (final / "tune_results.json").write_text(
            json.dumps(
                {
                    "user_prompt": user_prompt,
                    "config": {
                        "engine": engine,
                        "optimizer": optimizer,
                        "seed": seed,
                        "score_threshold": score_threshold,
                        "max_iterations": max_iterations,
                    },
                }
            ),
            encoding="utf-8",
        )
        (final / "history.jsonl").write_text(
            json.dumps({"trial_index": 0, "params": best_params, "score": 0.12}) + "\n",
            encoding="utf-8",
        )
        (final / "judge_result.json").write_text(
            json.dumps({"decision": "approve", "score": 0.96}),
            encoding="utf-8",
        )
        (final / "tuned_physics.usd").write_text("#usda 1.0\n", encoding="utf-8")
        (final / "report.md").write_text("# fake refine report\n", encoding="utf-8")

        for _ in range(3):
            if await manager.is_cancelled(session_id):
                await manager.update_session(
                    session_id,
                    {
                        "status": "cancelled",
                        "completed_at": "1970-01-01T00:00:01Z",
                        "duration_seconds": 1,
                        "can_cancel": False,
                        "results": {
                            **results,
                            "termination_reason": "cancelled",
                        },
                    },
                )
                return
            await asyncio.sleep(0.01)

        await manager.update_session(
            session_id,
            {
                "status": "completed",
                "completed_at": "1970-01-01T00:00:01Z",
                "duration_seconds": 1,
                "can_cancel": False,
                "results": results,
            },
        )

    import sys

    fake_module = type(sys)("refine_executor_stub")
    fake_module.execute_refine_async = fake_execute_refine_async
    monkeypatch.setitem(sys.modules, "service.workers.refine_executor", fake_module)
    monkeypatch.setitem(
        sys.modules,
        "apps.physics_agent_service.service.workers.refine_executor",
        fake_module,
    )


@pytest.mark.api
class TestRefineCreation:
    @pytest.mark.parametrize("tainted_field", ["user_prompt", "scenario_yaml"])
    async def test_rejects_secret_bearing_durable_content_before_session_creation(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
        tainted_field: str,
    ) -> None:
        """Every persisted free-text boundary is gated before publication."""
        from ...service import config_persistence
        from ...service.routers import refine_router

        sentinel = "short-refine-secret"
        signed_url = f"https://assets.example.test/a?X-Amz-Signature={sentinel}"
        data = {
            "scenario_yaml": _scenario_yaml(),
            "user_prompt": "make this object bouncy",
        }
        if tainted_field == "user_prompt":
            data["user_prompt"] = signed_url
        else:
            data["scenario_yaml"] += f"\napi_key: {sentinel}\n"

        manager_calls = 0

        def fail_manager() -> None:
            nonlocal manager_calls
            manager_calls += 1
            raise AssertionError("secret guard must run before session creation")

        monkeypatch.setattr(refine_router, "get_session_manager", fail_manager)
        response = await client.post("/refine", files=_multipart_files(), data=data)

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
                "X-Amz-Signature=refine-comment-sentinel\n",
                "refine-comment-sentinel",
            ),
            (
                _scenario_yaml() + "\nsource_url: https://assets.example.test/a?"
                "X-Amz-Signature=refine-shadowed-url\n"
                "source_url: https://assets.example.test/public\n",
                "refine-shadowed-url",
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
        from ...service.routers import refine_router

        manager_calls = 0

        def fail_manager() -> None:
            nonlocal manager_calls
            manager_calls += 1
            raise AssertionError("YAML guard must run before session creation")

        monkeypatch.setattr(refine_router, "get_session_manager", fail_manager)
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={
                "scenario_yaml": scenario_yaml,
                "user_prompt": "make this object bouncy",
            },
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
        from ...service.routers import refine_router

        sentinel = "refine-binary-sentinel"
        signed_url = f"https://assets.example.test/a?X-Amz-Signature={sentinel}"
        encoded = base64.b64encode(signed_url.encode()).decode()
        scenario_yaml = _scenario_yaml().replace(
            "drop_height_m: 0.5",
            f"drop_height_m: !!binary {encoded}",
        )
        manager_calls = 0

        def fail_manager() -> None:
            nonlocal manager_calls
            manager_calls += 1
            raise AssertionError("YAML guard must run before session creation")

        monkeypatch.setattr(refine_router, "get_session_manager", fail_manager)
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={
                "scenario_yaml": scenario_yaml,
                "user_prompt": "make this object bouncy",
            },
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
        from ...service.routers import refine_router

        scenario_yaml = """
name: drop_settle
metric: settle_distance
target_defaults: &target_defaults
  drop_height_m: 0.5
  duration_s: 2.0
  gravity: -9.81
target:
  <<: *target_defaults
parameters:
  - name: mass_scale
    min: 0.5
    max: 2.0
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
        monkeypatch.setattr(refine_router, "get_session_manager", fail_manager)
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={
                "scenario_yaml": scenario_yaml,
                "user_prompt": "make this object bouncy",
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            config_persistence.INVALID_DURABLE_INPUT_DETAIL
        )
        assert manager_calls == 0

    async def test_persists_canonical_benign_scenario_yaml(self, client) -> None:
        """The bytes that passed structural validation are the bytes persisted."""
        from ...service.routers import refine_router

        scenario_yaml = "# ordinary scenario comment\n" + _scenario_yaml()
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={
                "scenario_yaml": scenario_yaml,
                "user_prompt": "make this object bouncy",
            },
        )

        assert response.status_code == 202, response.text
        session_dir = refine_router.get_session_manager().get_session_dir(
            response.json()["session_id"]
        )
        persisted = (session_dir / "input" / "scenario.yaml").read_text(
            encoding="utf-8"
        )
        assert "ordinary scenario comment" not in persisted
        assert yaml.safe_load(persisted) == yaml.safe_load(scenario_yaml)

    async def test_malformed_scenario_error_is_value_free(self, client) -> None:
        sentinel = "never-echo-malformed-refine"
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={
                "scenario_yaml": f"name: [{sentinel}\n",
                "user_prompt": "make this object bouncy",
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid scenario YAML"
        for fragment in (sentinel, sentinel[:8], sentinel[-8:]):
            assert fragment not in response.text

    @pytest.mark.parametrize(
        ("prompt_template", "sentinel"),
        [
            ("Authorization: Bearer {}", "never-persist-refine-bearer"),
            ("use Bearer {} for the request", "never-persist-refine-bearer"),
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
        from ...service.routers import refine_router

        manager_calls = 0

        def fail_manager() -> None:
            nonlocal manager_calls
            manager_calls += 1
            raise AssertionError("bearer guard must run before session creation")

        monkeypatch.setattr(refine_router, "get_session_manager", fail_manager)
        response = await client.post(
            "/refine",
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

    async def test_create_refine_with_upload(self, client) -> None:
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={
                "scenario_yaml": _scenario_yaml(),
                "user_prompt": "make this object bouncy",
            },
        )
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["status"] == "pending"
        assert "session_id" in body

    async def test_create_refine_defaults_to_botorch_threshold_seed(
        self, client
    ) -> None:
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={
                "scenario_yaml": _scenario_yaml(),
                "user_prompt": "make this object bouncy",
            },
        )
        assert response.status_code == 202, response.text
        sid = response.json()["session_id"]

        from ...service.routers import refine_router

        metadata = await refine_router.get_session_manager().get_session_metadata(sid)
        assert metadata is not None
        config = metadata["config"]
        assert config["optimizer"] == "botorch"
        assert config["score_threshold"] == 0.9
        assert config["seed"] == 42

    async def test_create_refine_rejects_missing_prompt(self, client) -> None:
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={"scenario_yaml": _scenario_yaml()},
        )
        assert response.status_code == 400
        assert "user_prompt" in response.json()["detail"]

    async def test_create_refine_rejects_missing_scenario(self, client) -> None:
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={"user_prompt": "make it bouncy"},
        )
        assert response.status_code == 400
        assert "scenario_yaml" in response.json()["detail"]

    async def test_create_refine_rejects_multiple_sources(self, client) -> None:
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={
                "scenario_yaml": _scenario_yaml(),
                "user_prompt": "make it bouncy",
                "s3_uri": "s3://bucket/key.usda",
            },
        )
        assert response.status_code == 400
        assert "Exactly one" in response.json()["detail"]

    @pytest.mark.parametrize(
        "allowed_buckets,s3_uri",
        [
            ("", "s3://trusted-input-bucket/path/physics.usda"),
            ("trusted-input-bucket", "s3://foreign-bucket/path/physics.usda"),
        ],
    )
    async def test_create_refine_rejects_unapproved_s3_before_download(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
        allowed_buckets: str,
        s3_uri: str,
    ) -> None:
        """Refine inherits the tune helper's fail-closed S3 policy."""
        from ...service.routers import refine_router, tune_router

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
            refine_router,
            "get_session_manager",
            fail_manager,
        )
        monkeypatch.setattr(
            tune_router,
            "download_file_from_s3",
            fail_if_downloaded,
        )

        response = await client.post(
            "/refine",
            data={
                "s3_uri": s3_uri,
                "scenario_yaml": _scenario_yaml(),
                "user_prompt": "make it bouncy",
            },
        )

        assert response.status_code == 403
        assert response.json()["detail"] == (
            "S3 URI is not permitted by the service's configured bucket allowlist"
        )
        assert manager_calls == 0
        assert download_calls == 0

    async def test_create_refine_sanitizes_provisioning_errors(
        self, client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ...service.routers import refine_router, tune_router

        monkeypatch.setattr(tune_router.config, "s3_allowed_buckets", "bucket")

        def fail_download(_s3_uri: str, _session_dir) -> None:
            raise RuntimeError("/tmp/internal/path should not leak")

        monkeypatch.setattr(refine_router, "_download_s3_to_session", fail_download)

        response = await client.post(
            "/refine",
            data={
                "s3_uri": "s3://bucket/input.usda",
                "scenario_yaml": _scenario_yaml(),
                "user_prompt": "make it bouncy",
            },
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to provision input physics USD"


@pytest.mark.api
class TestRefineLifecycle:
    async def test_status_and_results_after_completion(self, client) -> None:
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={
                "scenario_yaml": _scenario_yaml(),
                "user_prompt": "make this object bouncy",
            },
        )
        sid = response.json()["session_id"]
        for _ in range(200):
            status_response = await client.get(f"/refine/{sid}/status")
            assert status_response.status_code == 200
            if status_response.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        status = (await client.get(f"/refine/{sid}/status")).json()
        assert status["status"] == "completed"
        assert status["iteration"] == 1
        assert status["max_iterations"] == 5
        assert status["best_params"] == {"mass_scale": 1.2, "restitution": 0.9}
        assert status["judge_score"] == 0.96

        results_response = await client.get(f"/refine/{sid}/results")
        assert results_response.status_code == 200, results_response.text
        results = results_response.json()
        assert results["status"] == "completed"
        assert results["termination_reason"] == "approved"
        assert results["iteration_count"] == 1
        assert results["final_judge_score"] == 0.96
        assert "refine_summary" in results["download_urls"]

    async def test_results_returns_202_while_running(self, client) -> None:
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={
                "scenario_yaml": _scenario_yaml(),
                "user_prompt": "make this object bouncy",
            },
        )
        sid = response.json()["session_id"]
        results_response = await client.get(f"/refine/{sid}/results")
        assert results_response.status_code in (200, 202)

    async def test_cancel_rejects_non_refine_session(self, client) -> None:
        from ...service.routers import refine_router

        manager = refine_router.get_session_manager()
        sid = await _create_non_refine_session()
        response = await client.post(f"/refine/{sid}/cancel")
        assert response.status_code == 409
        assert "not a refine session" in response.json()["detail"]
        assert not await manager.is_cancelled(sid)

    async def test_read_routes_reject_non_refine_session(self, client) -> None:
        sid = await _create_non_refine_session()

        for path in (
            f"/refine/{sid}/status",
            f"/refine/{sid}/results",
            f"/refine/{sid}/events",
            f"/refine/{sid}/artifacts/refine_summary.json",
        ):
            response = await client.get(path)
            assert response.status_code == 409, (path, response.text)
            assert "not a refine session" in response.json()["detail"]

    async def test_routes_reject_malformed_session_id(self, client) -> None:
        for method, path in (
            ("GET", "/refine/not-a-uuid/status"),
            ("GET", "/refine/not-a-uuid/results"),
            ("GET", "/refine/not-a-uuid/events"),
            ("GET", "/refine/not-a-uuid/artifacts/refine_summary.json"),
            ("POST", "/refine/not-a-uuid/cancel"),
        ):
            response = await client.request(method, path)
            assert response.status_code == 400, (method, path, response.text)
            assert "session_id" in response.json()["detail"]

    async def test_late_events_subscriber_returns_terminal_done(self, client) -> None:
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={
                "scenario_yaml": _scenario_yaml(),
                "user_prompt": "make this object bouncy",
            },
        )
        sid = response.json()["session_id"]
        for _ in range(200):
            status_response = await client.get(f"/refine/{sid}/status")
            if status_response.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        events_response = await client.get(f"/refine/{sid}/events")
        assert events_response.status_code == 200, events_response.text
        assert "event: done" in events_response.text
        assert '"final_state": "completed"' in events_response.text


@pytest.mark.api
class TestRefineArtifacts:
    async def test_download_refine_summary_and_final_usd(self, client) -> None:
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={
                "scenario_yaml": _scenario_yaml(),
                "user_prompt": "make this object bouncy",
            },
        )
        sid = response.json()["session_id"]
        for _ in range(200):
            status = (await client.get(f"/refine/{sid}/status")).json()["status"]
            if status == "completed":
                break
            await asyncio.sleep(0.01)

        summary = await client.get(f"/refine/{sid}/artifacts/refine_summary.json")
        assert summary.status_code == 200, summary.text
        assert summary.json()["termination_reason"] == "approved"

        tuned_usd = await client.get(f"/refine/{sid}/artifacts/final/tuned_physics.usd")
        assert tuned_usd.status_code == 200, tuned_usd.text
        assert tuned_usd.text.startswith("#usda")

        legacy_tuned_usd = await client.get(
            f"/refine/{sid}/artifacts/final/tuned_physics.usda"
        )
        assert legacy_tuned_usd.status_code == 200, legacy_tuned_usd.text
        assert legacy_tuned_usd.headers["content-disposition"].endswith(
            'filename="tuned_physics.usd"'
        )

    async def test_download_unknown_artifact_404(self, client) -> None:
        response = await client.post(
            "/refine",
            files=_multipart_files(),
            data={
                "scenario_yaml": _scenario_yaml(),
                "user_prompt": "make this object bouncy",
            },
        )
        sid = response.json()["session_id"]
        response = await client.get(f"/refine/{sid}/artifacts/../../etc/passwd")
        assert response.status_code in (404, 422)
