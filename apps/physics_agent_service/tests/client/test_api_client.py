# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from typing import cast

import pytest

from ...client import client as client_module
from ...client import client_v2
from ...client.client import PhysicsAgentClient, build_arg_parser

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


class _FakeResponse:
    ok = True
    status_code = 202

    def __init__(
        self,
        payload: dict | None = None,
        *,
        content: bytes = b"artifact",
        text: str = "artifact",
    ) -> None:
        self._payload = payload or {"session_id": "session-123"}
        self.content = content
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.posts: list[dict] = []
        self.gets: list[dict] = []
        self.deletes: list[dict] = []

    def post(self, url, data=None, files=None, json=None, timeout=None):
        self.posts.append(
            {
                "url": url,
                "data": data,
                "files": files,
                "json": json,
                "timeout": timeout,
            }
        )
        return _FakeResponse()

    def get(self, url, **kwargs):
        self.gets.append({"url": url, **kwargs})
        return _FakeResponse({"status": "completed", "total": 1})

    def delete(self, url, **kwargs):
        self.deletes.append({"url": url, **kwargs})
        return _FakeResponse()


class _FakeStreamResponse:
    def __init__(self, lines: list[str | None]) -> None:
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = True):
        yield from self._lines


class _FakeStreamSession:
    def __init__(self, lines: list[str | None]) -> None:
        self.headers: dict[str, str] = {}
        self.lines = lines
        self.gets: list[dict] = []

    def get(self, url, **kwargs):
        self.gets.append({"url": url, **kwargs})
        return _FakeStreamResponse(self.lines)


@pytest.fixture
def api_client() -> PhysicsAgentClient:
    client = PhysicsAgentClient(base_url=BASE_URL)
    return client


def test_client_auth_header_and_helpers(monkeypatch):
    monkeypatch.setenv("PHYSICS_AGENT_TOKEN", "secret")
    client = PhysicsAgentClient(base_url="http://test/")
    assert client.base_url == "http://test"
    assert client._http.headers["Authorization"] == "Bearer secret"
    assert client._http.headers["User-Agent"] == "physics-agent-client/2.0"

    assert client_module._bool_form(True) == "true"
    assert client_module._bool_form(False) == "false"
    assert client_module._json_array_arg(["front", "side"]) == '["front", "side"]'
    with pytest.raises(ValueError, match="scenario_yaml"):
        client_module._read_text_arg(
            value="name: x", path="x.yaml", name="scenario_yaml"
        )
    with pytest.raises(ValueError, match="family must be one of"):
        client.get_status("session-123", family="bad")  # type: ignore[arg-type]

    assert client_v2.PhysicsAgentClient is PhysicsAgentClient


def test_client_defers_render_backend_validation_to_server() -> None:
    args = build_arg_parser().parse_args(
        ["--render-backend", "future-server-backend", "asset.usdz"]
    )

    assert args.render_backend == "future-server-backend"


def test_start_pipeline_forwards_optimizer_flags(tmp_path):
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    fake_session = _FakeSession()

    client = PhysicsAgentClient(base_url="http://test", timeout_seconds=7)
    client._http = fake_session

    session_id = client.start_pipeline(
        usd_path=str(usd_path),
        optimize_usd=True,
        enable_deinstance=False,
        enable_split=True,
        enable_deduplicate=True,
    )

    assert session_id == "session-123"
    post = fake_session.posts[0]
    assert post["url"] == "http://test/pipeline"
    assert post["timeout"] == 7
    assert post["data"]["optimize_usd"] == "true"
    assert post["data"]["enable_deinstance"] == "false"
    assert post["data"]["enable_split"] == "true"
    assert post["data"]["enable_deduplicate"] == "true"
    assert post["files"][0][0] == "usd_file"


def test_start_predict_posts_prediction_route_payload(tmp_path):
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    fake_session = _FakeSession()
    client = PhysicsAgentClient(base_url="http://test")
    client._http = fake_session

    session_id = client.start_predict(
        usd_path=str(usd_path),
        user_prompt="predict only",
        render_backend="remote",
        optimize_usd=True,
        enable_deinstance=True,
        enable_split=True,
    )

    assert session_id == "session-123"
    post = fake_session.posts[0]
    assert post["url"] == "http://test/predict"
    assert post["data"]["user_prompt"] == "predict only"
    assert post["data"]["render_backend"] == "remote"
    assert post["data"]["optimize_usd"] == "true"
    assert post["data"]["enable_deinstance"] == "true"
    assert post["data"]["enable_split"] == "true"
    assert post["data"]["enable_deduplicate"] == "false"
    assert post["files"][0][0] == "usd_file"


def test_start_predict_allows_dataset_path_mode():
    fake_session = _FakeSession()
    client = PhysicsAgentClient(base_url="http://test")
    client._http = fake_session

    session_id = client.start_predict(dataset_path="/srv/sessions/s/dataset.jsonl")

    assert session_id == "session-123"
    post = fake_session.posts[0]
    assert post["url"] == "http://test/predict"
    assert post["data"]["dataset_path"] == "/srv/sessions/s/dataset.jsonl"
    assert post["files"] is None


def test_start_predict_allows_session_dataset_override():
    fake_session = _FakeSession()
    client = PhysicsAgentClient(base_url="http://test")
    client._http = fake_session

    session_id = client.start_predict(
        session_id="existing-session",
        dataset_path="/srv/sessions/s/dataset.jsonl",
    )

    assert session_id == "session-123"
    post = fake_session.posts[0]
    assert post["data"]["session_id"] == "existing-session"
    assert post["data"]["dataset_path"] == "/srv/sessions/s/dataset.jsonl"
    assert post["files"] is None


@pytest.mark.parametrize(
    "sources",
    (
        {"session_id": "session", "usd_path": "scene.usd"},
        {"session_id": "session", "s3_uri": "s3://bucket/scene.usd"},
        {"usd_path": "scene.usd", "s3_uri": "s3://bucket/scene.usd"},
    ),
)
def test_start_predict_rejects_multiple_primary_sources_without_http(sources):
    fake_session = _FakeSession()
    client = PhysicsAgentClient(base_url="http://test")
    client._http = fake_session

    with pytest.raises(ValueError, match="Provide at most one"):
        client.start_predict(**sources)

    assert fake_session.posts == []


@pytest.mark.parametrize("source", ("usd_path", "s3_uri"))
def test_start_predict_rejects_dataset_with_mode_b_source_without_http(source):
    fake_session = _FakeSession()
    client = PhysicsAgentClient(base_url="http://test")
    client._http = fake_session
    value = "scene.usd" if source == "usd_path" else "s3://bucket/scene.usd"

    with pytest.raises(ValueError, match="dataset_path may be used alone"):
        client.start_predict(
            dataset_path="/srv/sessions/s/dataset.jsonl",
            **{source: value},
        )

    assert fake_session.posts == []


def test_start_predict_rejects_missing_source_without_http():
    fake_session = _FakeSession()
    client = PhysicsAgentClient(base_url="http://test")
    client._http = fake_session

    with pytest.raises(ValueError, match="One of session_id"):
        client.start_predict()

    assert fake_session.posts == []


def test_start_tune_posts_source_session_defaults_and_reference_media(tmp_path):
    image = tmp_path / "ref.png"
    image.write_bytes(b"png")
    fake_session = _FakeSession()
    client = PhysicsAgentClient(base_url="http://test")
    client._http = fake_session

    session_id = client.start_tune(
        source_session_id="pipeline-session",
        scenario_yaml="name: drop_settle\nparameters: []\n",
        user_prompt="make it bouncy",
        reference_images=[str(image)],
        reference_descriptions=["front view"],
    )

    assert session_id == "session-123"
    post = fake_session.posts[0]
    assert post["url"] == "http://test/tune"
    assert post["data"]["source_session_id"] == "pipeline-session"
    assert post["data"]["scenario_yaml"].startswith("name: drop_settle")
    assert post["data"]["user_prompt"] == "make it bouncy"
    assert post["data"]["optimizer"] == "auto"
    assert post["data"]["engine"] == "ovphysx"
    assert post["data"]["max_trials"] == "30"
    assert post["data"]["seed"] == "42"
    assert post["data"]["enable_judge"] == "true"
    assert post["data"]["judge_max_iterations"] == "3"
    assert post["data"]["reference_video_frames"] == "8"
    assert post["data"]["judge_reference_frames"] == "8"
    assert post["data"]["judge_generated_frames"] == "16"
    assert post["data"]["reference_descriptions"] == '["front view"]'
    assert post["files"][0][0] == "reference_images"


def test_start_tune_uploads_physics_usd_and_reads_scenario_file(tmp_path):
    physics_usd = tmp_path / "physics.usda"
    scenario = tmp_path / "scenario.yaml"
    physics_usd.write_text("#usda 1.0\n", encoding="utf-8")
    scenario.write_text("name: drop_settle\nparameters: []\n", encoding="utf-8")
    fake_session = _FakeSession()
    client = PhysicsAgentClient(base_url="http://test")
    client._http = fake_session

    session_id = client.start_tune(
        physics_usd_path=str(physics_usd),
        scenario_yaml_path=str(scenario),
        optimizer="botorch",
        max_trials=5,
        enable_judge=False,
    )

    assert session_id == "session-123"
    post = fake_session.posts[0]
    assert post["url"] == "http://test/tune"
    assert post["data"]["scenario_yaml"].startswith("name: drop_settle")
    assert post["data"]["optimizer"] == "botorch"
    assert post["data"]["max_trials"] == "5"
    assert post["data"]["enable_judge"] == "false"
    assert post["files"][0][0] == "physics_usd"
    assert all(field != "usd_file" for field, _ in post["files"])


def test_start_refine_posts_refine_defaults():
    fake_session = _FakeSession()
    client = PhysicsAgentClient(base_url="http://test")
    client._http = fake_session

    session_id = client.start_refine(
        source_session_id="pipeline-session",
        scenario_yaml="name: drop_settle\nparameters: []\n",
        user_prompt="settle on the target",
    )

    assert session_id == "session-123"
    post = fake_session.posts[0]
    assert post["url"] == "http://test/refine"
    assert post["data"]["source_session_id"] == "pipeline-session"
    assert post["data"]["scenario_yaml"].startswith("name: drop_settle")
    assert post["data"]["user_prompt"] == "settle on the target"
    assert post["data"]["optimizer"] == "botorch"
    assert post["data"]["engine"] == "ovphysx"
    assert post["data"]["max_trials"] == "30"
    assert post["data"]["max_iterations"] == "5"
    assert post["data"]["score_threshold"] == "0.9"
    assert post["data"]["seed"] == "42"
    assert post["data"]["visual_evidence_enabled"] == "true"
    assert post["data"]["llm_timeout_seconds"] == "180.0"
    assert post["data"]["reference_video_frames"] == "8"
    assert post["data"]["judge_reference_frames"] == "8"
    assert post["data"]["judge_generated_frames"] == "16"


def test_tune_and_refine_validate_sources_and_required_fields():
    client = PhysicsAgentClient(base_url="http://test")

    with pytest.raises(ValueError, match="Exactly one"):
        client.start_tune(
            scenario_yaml="name: drop_settle\n",
            physics_usd_path="a.usda",
            source_session_id="sid",
        )
    with pytest.raises(ValueError, match="Either scenario_yaml or user_prompt"):
        client.start_tune(source_session_id="sid")
    with pytest.raises(ValueError, match="Either scenario_yaml or user_prompt"):
        client.start_tune(source_session_id="sid", scenario_yaml="   ")
    with pytest.raises(ValueError, match="Exactly one"):
        client.start_refine(
            scenario_yaml="name: drop_settle\n",
            user_prompt="target",
            physics_usd_path="a.usda",
            s3_uri="s3://bucket/a.usda",
        )
    with pytest.raises(ValueError, match="scenario_yaml is required"):
        client.start_refine(source_session_id="sid", scenario_yaml="", user_prompt="x")
    with pytest.raises(ValueError, match="user_prompt is required"):
        client.start_refine(
            source_session_id="sid",
            scenario_yaml="name: drop_settle\n",
            user_prompt=" ",
        )


@pytest.mark.parametrize(
    ("method_name", "base_kwargs"),
    [
        (
            "start_tune",
            {
                "source_session_id": "pipeline-session",
                "scenario_yaml": "name: drop_settle\nparameters: []\n",
            },
        ),
        (
            "start_refine",
            {
                "source_session_id": "pipeline-session",
                "scenario_yaml": "name: drop_settle\nparameters: []\n",
                "user_prompt": "settle on the target",
            },
        ),
    ],
)
@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("reference_video_frames", 0),
        ("judge_reference_frames", 65),
        ("judge_generated_frames", cast(int, 3.0)),
    ],
)
def test_tune_and_refine_validate_visual_frame_counts_before_request(
    method_name: str,
    base_kwargs: dict[str, str],
    field_name: str,
    bad_value: int,
) -> None:
    fake_session = _FakeSession()
    client = PhysicsAgentClient(base_url="http://test")
    client._http = fake_session
    kwargs = {**base_kwargs, field_name: bad_value}

    with pytest.raises(ValueError, match=field_name):
        getattr(client, method_name)(**kwargs)

    assert fake_session.posts == []


def test_route_family_methods_call_expected_endpoints():
    fake_session = _FakeSession()
    client = PhysicsAgentClient(base_url="http://test")
    client._http = fake_session

    assert client.get_status("p")["status"] == "completed"
    assert client.get_predict_status("pred")["status"] == "completed"
    assert client.get_tune_results("tune")["status"] == "completed"
    assert client.get_refine_results("ref")["status"] == "completed"
    client.cancel_predict("pred")
    client.cancel_tune("tune")
    client.cancel_refine("ref")

    assert [call["url"] for call in fake_session.gets] == [
        "http://test/pipeline/p/status",
        "http://test/predict/pred/status",
        "http://test/tune/tune/results",
        "http://test/refine/ref/results",
    ]
    assert [call["url"] for call in fake_session.posts] == [
        "http://test/predict/pred/cancel",
        "http://test/tune/tune/cancel",
        "http://test/refine/ref/cancel",
    ]


def test_stream_events_uses_requested_route_family():
    client = PhysicsAgentClient(base_url="http://test")
    fake_session = _FakeStreamSession(
        [
            "event: progress",
            'data: {"step": "tune"}',
            "",
        ]
    )
    client._http = fake_session

    messages = list(client.stream_tune_events("tune-session", request_timeout=9))

    assert messages[0].event == "progress"
    assert messages[0].json() == {"step": "tune"}
    assert fake_session.gets[0]["url"] == "http://test/tune/tune-session/events"
    assert fake_session.gets[0]["headers"]["Accept"] == "text/event-stream"
    assert fake_session.gets[0]["timeout"] == 9


def test_artifact_helpers_call_expected_paths():
    fake_session = _FakeSession()
    client = PhysicsAgentClient(base_url="http://test")
    client._http = fake_session

    assert client.download_predictions("sid") == b"artifact"
    assert client.download_report("sid") == "artifact"
    assert client.download_dataset("sid") == b"artifact"
    assert client.download_output_usd("sid") == b"artifact"
    assert client.download_tune_artifact("tid", "best_params.json") == b"artifact"
    assert (
        client.download_refine_artifact("rid", "final/tuned_physics.usd") == b"artifact"
    )
    assert client.download_tune_artifact("tid", "best params?.json") == b"artifact"
    assert (
        client.download_refine_artifact("rid", "final/tuned physics#1.usda")
        == b"artifact"
    )

    assert [call["url"] for call in fake_session.gets] == [
        "http://test/artifacts/sid/predictions",
        "http://test/artifacts/sid/report",
        "http://test/artifacts/sid/dataset",
        "http://test/artifacts/sid/output-usd",
        "http://test/tune/tid/artifacts/best_params.json",
        "http://test/refine/rid/artifacts/final/tuned_physics.usd",
        "http://test/tune/tid/artifacts/best%20params%3F.json",
        "http://test/refine/rid/artifacts/final/tuned%20physics%231.usda",
    ]


def test_run_and_monitor_polling_fallback_prints_step_progress(monkeypatch, capsys):
    client = PhysicsAgentClient(base_url="http://test")
    monkeypatch.setattr(client_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(client, "start_pipeline", lambda **_: "session-123")

    def fail_sse(_session_id: str):
        raise RuntimeError("503 Service Unavailable")

    statuses = [
        {
            "status": "running",
            "overall_progress": {"percent": 40},
            "current_step": {
                "name": "build_dataset_usd",
                "progress": {
                    "current": 0,
                    "total": 10,
                    "message": "Rendering: 0/10",
                },
            },
        },
        {"status": "completed", "overall_progress": {"percent": 100}},
        {"status": "completed", "overall_progress": {"percent": 100}},
    ]

    monkeypatch.setattr(client, "stream_events", fail_sse)
    monkeypatch.setattr(client, "get_status", lambda _session_id: statuses.pop(0))

    session_id, status = client.run_and_monitor(
        usd_path="scene.usda",
        print_stream=True,
    )

    assert session_id == "session-123"
    assert status == {"status": "completed", "overall_progress": {"percent": 100}}
    output = capsys.readouterr().out
    assert "SSE unavailable (cross-instance (503)), falling back to polling" in output
    assert "[build_dataset_usd] running  overall=40%  0/10 Rendering: 0/10" in output


@pytest.mark.parametrize(
    "kwargs",
    [
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../physics_agent/data/examples/Lightbulb01/light_bulb_01.usdz",
                ),
            }
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../physics_agent/data/examples/Lightbulb01/light_bulb_01.usdz",
                ),
                "user_prompt": "Focus on identifying the glass and metal parts of the lightbulb.",
            }
        ),
    ],
)
@pytest.mark.skipif(
    os.getenv("RUN_CLIENT_TESTS", "false").lower() not in ["true", "1", "yes", "y"],
    reason="Skipping test in CI",
)
async def test_basic_pipeline(api_client: PhysicsAgentClient, kwargs: dict):
    session_id, status = api_client.run_and_monitor(**kwargs)
    assert status is not None
    assert status["status"] == "completed"

    predictions = api_client.download_predictions(session_id)
    assert len(predictions) > 0
    print(f"Predictions downloaded: {len(predictions)} bytes")

    report = api_client._http.get(
        f"{api_client.base_url}/artifacts/{session_id}/report"
    )
    assert report.status_code == 200
    report_text = report.text
    assert len(report_text) > 0
    print(f"Report downloaded: {len(report_text)} bytes")

    if "user_prompt" in kwargs:
        assert kwargs["user_prompt"] in report_text
        print(f"user_prompt: {kwargs['user_prompt']} found in report")
