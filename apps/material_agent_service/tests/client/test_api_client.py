# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os

import pytest
import requests

from ...client import client as client_module
from ...client.client import MaterialAgentClient

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


class _FakeResponse:
    def __init__(self, payload: dict | None = None, status_code: int = 200):
        self._payload = payload or {}
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"unexpected HTTP status {self.status_code}")


class _FakeSession:
    def __init__(self):
        self.posts: list[dict] = []
        self.gets: list[dict] = []
        self.heads: list[dict] = []
        self.head_responses: list[_FakeResponse] = []

    def post(self, url: str, **kwargs):
        self.posts.append({"url": url, **kwargs})
        return _FakeResponse(
            {
                "status": "ok",
                "session_id": "session-1",
                "reference_id": "ref-1",
                "image_url": "/assets/s/generated-ref/ref-1",
            }
        )

    def get(self, url: str, **kwargs):
        self.gets.append({"url": url, **kwargs})
        return _FakeResponse({"events": [{"step": "predict"}], "total": 1})

    def head(self, url: str, **kwargs):
        self.heads.append({"url": url, **kwargs})
        if self.head_responses:
            return self.head_responses.pop(0)
        return _FakeResponse(status_code=200)


class _FakeStreamResponse:
    def __init__(self, lines: list[str | None]):
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
    def __init__(self, lines: list[str | None]):
        self.lines = lines
        self.gets: list[dict] = []

    def get(self, url: str, **kwargs):
        self.gets.append({"url": url, **kwargs})
        return _FakeStreamResponse(self.lines)


@pytest.fixture
def api_client() -> MaterialAgentClient:
    client = MaterialAgentClient(base_url=BASE_URL)
    return client


def test_client_validation_helpers_and_auth_header(monkeypatch):
    monkeypatch.setenv("MATERIAL_AGENT_TOKEN", "secret")
    client = MaterialAgentClient(base_url="http://service/")
    assert client.base_url == "http://service"
    assert client._http.headers["Authorization"] == "Bearer secret"

    assert client_module._max_from_env("MISSING_MAX", 7) == 7
    monkeypatch.setenv("BAD_MAX", "abc")
    with pytest.raises(ValueError, match="BAD_MAX must be an integer"):
        client_module._max_from_env("BAD_MAX", 7)
    monkeypatch.setenv("ZERO_MAX", "0")
    with pytest.raises(ValueError, match="ZERO_MAX must be at least 1"):
        client_module._max_from_env("ZERO_MAX", 7)

    with pytest.raises(ValueError, match="workers must be an integer"):
        client_module._validate_worker_override("workers", True, "MISSING_MAX", 7)
    with pytest.raises(ValueError, match="count must be at least 1"):
        client_module._validate_positive_override("count", True)
    with pytest.raises(ValueError, match="ratio must be between"):
        client_module._validate_unit_interval_override("ratio", object())
    with pytest.raises(ValueError, match="ratio must be between"):
        client_module._validate_unit_interval_override("ratio", 2)
    with pytest.raises(ValueError, match="texture must be an integer"):
        client_module._validate_texture_size("texture", True)
    with pytest.raises(ValueError, match="texture must be between"):
        client_module._validate_texture_size("texture", 8192)

    assert client_module._parse_json_object_arg(None, "--filters") is None
    with pytest.raises(ValueError, match="valid JSON"):
        client_module._parse_json_object_arg("{", "--filters")
    with pytest.raises(ValueError, match="JSON object"):
        client_module._parse_json_object_arg("[]", "--filters")

    message = client_module.SSEMessage(event="progress", data='{"ok": true}')
    assert message.json() == {"ok": True}


def test_upload_usd_and_start_pipeline_full_payload(tmp_path):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    ref_path = tmp_path / "ref.png"
    pdf_path = tmp_path / "ref.pdf"
    materials_zip = tmp_path / "materials.zip"
    usd_path.write_text("#usda 1.0\n")
    ref_path.write_bytes(b"png")
    pdf_path.write_bytes(b"%PDF")
    materials_zip.write_bytes(b"zip")

    assert client.upload_usd(str(usd_path)) == "session-1"
    assert fake_session.posts[0]["url"] == "http://service/pipeline/upload-usd"

    session_id = client.start_pipeline(
        session_id="existing-session",
        reference_images=[str(ref_path)],
        reference_pdfs=[str(pdf_path)],
        reference_descriptions=["front reference"],
        pdf_descriptions=["spec sheet"],
        user_prompt="prefer brushed metal",
        camera_views="+x",
        pdf_first_page=1,
        pdf_last_page=2,
        optimize_usd=True,
        enable_deinstance=True,
        enable_split=False,
        enable_deduplicate=True,
        materials_zip_path=str(materials_zip),
        vlm_model="nim/model",
        generated_reference_id="ref-1",
        coverage_policy="allow_partial",
        layer_only=True,
        large_scene=True,
        scene_workers=1,
        scene_assets="AssetA",
        scene_resume=True,
        scene_from_step="predict",
        scene_skip_existing=True,
        scene_no_render=True,
        scene_simulate=True,
        scene_simulate_mock_analyze=True,
        scene_fail_on_validation_error=True,
        scene_filters={"include": ["AssetA"]},
    )

    assert session_id == "session-1"
    post = fake_session.posts[1]
    assert post["data"]["session_id"] == "existing-session"
    assert post["data"]["reference_descriptions"] == '["front reference"]'
    assert post["data"]["pdf_descriptions"] == '["spec sheet"]'
    assert post["data"]["user_prompt"] == "prefer brushed metal"
    assert post["data"]["camera_views"] == "+x"
    assert post["data"]["pdf_first_page"] == "1"
    assert post["data"]["pdf_last_page"] == "2"
    assert post["data"]["vlm_model"] == "nim/model"
    assert post["data"]["generated_reference_id"] == "ref-1"
    assert post["data"]["layer_only"] == "true"
    assert post["data"]["scene_assets"] == "AssetA"
    assert post["data"]["optimize_usd"] == "true"
    assert post["data"]["enable_deinstance"] == "true"
    assert post["data"]["enable_split"] == "false"
    assert post["data"]["enable_deduplicate"] == "true"
    assert json.loads(post["data"]["scene_filters"]) == {"include": ["AssetA"]}
    assert [item[0] for item in post["files"]] == [
        "reference_images",
        "reference_pdfs",
        "materials_zip",
    ]


def test_wait_for_input_render_non_terminal_paths(monkeypatch):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    fake_session.head_responses.append(_FakeResponse(status_code=500))
    client._http = fake_session  # type: ignore[assignment]
    with pytest.raises(AssertionError, match="unexpected HTTP status 500"):
        client.wait_for_input_render("session-1", poll_interval_seconds=0)

    fake_session = _FakeSession()
    fake_session.head_responses.append(_FakeResponse(status_code=404))
    client._http = fake_session  # type: ignore[assignment]
    with pytest.raises(TimeoutError, match="Input preview was not available"):
        client.wait_for_input_render("session-1", timeout_seconds=0)

    sleeps: list[float] = []
    fake_session = _FakeSession()
    fake_session.head_responses.extend(
        [_FakeResponse(status_code=503), _FakeResponse(status_code=200)]
    )
    client._http = fake_session  # type: ignore[assignment]
    monkeypatch.setattr(client_module.time, "sleep", lambda delay: sleeps.append(delay))
    client.wait_for_input_render(
        "session-1", timeout_seconds=60, poll_interval_seconds=3
    )
    assert sleeps == [3]


def test_stream_events_parses_sse_messages():
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeStreamSession(
        [
            "",
            None,
            ": heartbeat",
            "event: progress",
            "id: 1",
            "retry: bad",
            'data: {"step": "predict"}',
            "",
            "retry: 5",
            'data: {"done": true}',
            "",
            "unknown",
            "data: tail",
        ]
    )
    client._http = fake_session  # type: ignore[assignment]

    messages = list(client.stream_events("session-1", request_timeout=7))

    assert messages[0].event == "progress"
    assert messages[0].id == "1"
    assert messages[0].retry is None
    assert messages[0].json() == {"step": "predict"}
    assert messages[1].event == "message"
    assert messages[1].retry == 5
    assert messages[2].data == "tail"
    assert fake_session.gets[0]["headers"]["Accept"] == "text/event-stream"
    assert fake_session.gets[0]["timeout"] == 7


def test_simple_client_methods_call_expected_endpoints():
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]

    assert client.get_status("session-1")["total"] == 1
    assert client.get_results("session-1")["total"] == 1
    client.cancel("session-1")
    assert client.sessions()["total"] == 1
    assert client.health()["total"] == 1

    assert [call["url"] for call in fake_session.gets] == [
        "http://service/pipeline/session-1/status",
        "http://service/pipeline/session-1/results",
        "http://service/sessions",
        "http://service/health",
    ]
    assert fake_session.posts[-1]["url"] == "http://service/pipeline/session-1/cancel"


def test_get_results_preserves_failed_artifact_download_urls():
    client = MaterialAgentClient(base_url="http://service")
    expected = {
        "status": "failed",
        "failed_step": "coverage_validation",
        "download_urls": {
            "output_usd": "/artifacts/session-1/output",
            "predictions": "/artifacts/session-1/predictions",
        },
    }

    class _FailedResultsSession(_FakeSession):
        def get(self, url: str, **kwargs):
            self.gets.append({"url": url, **kwargs})
            return _FakeResponse(expected)

    fake_session = _FailedResultsSession()
    client._http = fake_session  # type: ignore[assignment]

    assert client.get_results("session-1") == expected


def test_get_results_raises_http_error_while_results_are_pending():
    client = MaterialAgentClient(base_url="http://service")

    class _PendingResultsSession(_FakeSession):
        def get(self, url: str, **kwargs):
            self.gets.append({"url": url, **kwargs})
            return _FakeResponse({"status": "pending"}, status_code=202)

    fake_session = _PendingResultsSession()
    client._http = fake_session  # type: ignore[assignment]

    with pytest.raises(
        requests.HTTPError, match="pipeline results are not ready"
    ) as exc:
        client.get_results("session-1")

    assert exc.value.response is not None
    assert exc.value.response.status_code == 202
    assert fake_session.gets == [
        {
            "url": "http://service/pipeline/session-1/results",
            "timeout": client.timeout_seconds,
        }
    ]


def test_generate_reference_image_posts_prompt():
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]

    result = client.generate_reference_image("session-1", "matte blue plastic")

    assert result["reference_id"] == "ref-1"
    assert result["image_url"] == "/assets/s/generated-ref/ref-1"
    assert fake_session.posts == [
        {
            "url": "http://service/pipeline/session-1/generate-reference-image",
            "data": {"prompt": "matte blue plastic"},
            "timeout": client.timeout_seconds,
        }
    ]


def test_start_pipeline_posts_worker_overrides(tmp_path):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    session_id = client.start_pipeline(
        usd_path=str(usd_path),
        user_email="test@example.com",
        vlm_max_workers=2,
        render_num_workers=1,
    )

    assert session_id == "session-1"
    assert fake_session.posts[0]["url"] == "http://service/pipeline"
    assert fake_session.posts[0]["data"]["vlm_max_workers"] == "2"
    assert fake_session.posts[0]["data"]["render_num_workers"] == "1"


def test_start_pipeline_rejects_large_scene_with_material_generation():
    client = MaterialAgentClient(base_url="http://service")

    with pytest.raises(
        ValueError,
        match="large_scene is not compatible with enable_material_generation",
    ):
        client.start_pipeline(
            usd_path="unused.usda",
            large_scene=True,
            enable_material_generation=True,
        )


def test_run_and_monitor_rejects_large_scene_with_material_generation():
    client = MaterialAgentClient(base_url="http://service")

    with pytest.raises(
        ValueError,
        match="large_scene is not compatible with enable_material_generation",
    ):
        client.run_and_monitor(
            usd_path="unused.usda",
            large_scene=True,
            enable_material_generation=True,
            print_stream=False,
        )


def test_regenerate_posts_json_body():
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]

    result = client.regenerate(
        "session-1",
        steps=["predict", "apply"],
        user_prompt="Prefer brushed aluminum",
        layer_only=True,
        coverage_policy="strict",
    )

    assert result["session_id"] == "session-1"
    assert fake_session.posts == [
        {
            "url": "http://service/pipeline/session-1/regenerate",
            "json": {
                "steps": ["predict", "apply"],
                "user_prompt": "Prefer brushed aluminum",
                "layer_only": True,
                "coverage_policy": "strict",
            },
            "timeout": client.timeout_seconds,
        }
    ]


def test_get_event_log_uses_persisted_history_endpoint():
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]

    result = client.get_event_log("session-1")

    assert result == {"events": [{"step": "predict"}], "total": 1}
    assert fake_session.gets == [
        {
            "url": "http://service/pipeline/session-1/event-log",
            "timeout": client.timeout_seconds,
        }
    ]


def test_start_pipeline_allows_service_default_vlm_worker_cap(tmp_path):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    session_id = client.start_pipeline(
        usd_path=str(usd_path),
        user_email="test@example.com",
        vlm_max_workers=64,
    )

    assert session_id == "session-1"
    assert fake_session.posts[0]["data"]["vlm_max_workers"] == "64"
    assert fake_session.posts[0]["data"]["coverage_policy"] == "strict"


def test_start_pipeline_posts_allow_partial_coverage_override(tmp_path):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    client.start_pipeline(
        usd_path=str(usd_path),
        coverage_policy="allow_partial",
    )

    assert fake_session.posts[0]["data"]["coverage_policy"] == "allow_partial"


def test_start_pipeline_rejects_unknown_coverage_policy(tmp_path):
    client = MaterialAgentClient(base_url="http://service")
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    with pytest.raises(ValueError, match="coverage_policy"):
        client.start_pipeline(
            usd_path=str(usd_path),
            coverage_policy="best_effort",
        )


def test_start_pipeline_posts_prim_clustering_overrides(tmp_path):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    session_id = client.start_pipeline(
        usd_path=str(usd_path),
        user_email="test@example.com",
        enable_prim_clustering=True,
        cluster_min_prims=25,
        cluster_embedding_backend="nim",
        cluster_embedding_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        cluster_embedding_base_url="http://embedding-nim:8000/v1",
        cluster_embedding_max_workers=2,
        cluster_embedding_batch_size=8,
        cluster_max_size=11,
        cluster_similarity_threshold_low=0.97,
        cluster_similarity_threshold_medium=0.94,
        cluster_similarity_threshold_high=0.88,
        cluster_report=False,
    )

    assert session_id == "session-1"
    data = fake_session.posts[0]["data"]
    assert data["enable_prim_clustering"] == "true"
    assert data["cluster_min_prims"] == "25"
    assert data["cluster_embedding_backend"] == "nim"
    assert data["cluster_embedding_model"] == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert data["cluster_embedding_base_url"] == "http://embedding-nim:8000/v1"
    assert data["cluster_embedding_max_workers"] == "2"
    assert data["cluster_embedding_batch_size"] == "8"
    assert data["cluster_max_size"] == "11"
    assert data["cluster_similarity_threshold_low"] == "0.97"
    assert data["cluster_similarity_threshold_medium"] == "0.94"
    assert data["cluster_similarity_threshold_high"] == "0.88"
    assert data["cluster_report"] == "false"


def test_start_pipeline_posts_material_generation_options(tmp_path):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    session_id = client.start_pipeline(
        usd_path=str(usd_path),
        user_email="test@example.com",
        enable_material_generation=True,
        material_generation_guidance="orange glossy enclosure",
        material_generation_texture_size=512,
    )

    assert session_id == "session-1"
    data = fake_session.posts[0]["data"]
    assert data["enable_material_generation"] == "true"
    assert data["material_generation_guidance"] == "orange glossy enclosure"
    assert data["material_generation_texture_size"] == "512"


def test_start_pipeline_rejects_bad_material_generation_texture_size(tmp_path):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    with pytest.raises(ValueError, match="material_generation_texture_size"):
        client.start_pipeline(
            usd_path=str(usd_path),
            enable_material_generation=True,
            material_generation_texture_size=32,
        )


def test_start_pipeline_posts_large_scene_options(tmp_path):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    with pytest.warns(UserWarning, match="defaults coverage_policy"):
        session_id = client.start_pipeline(
            usd_path=str(usd_path),
            user_email="test@example.com",
            large_scene=True,
            scene_workers=2,
            scene_assets=["AssetA", "/World/AssetB"],
            scene_resume=True,
            scene_from_step="predict",
            scene_skip_existing=True,
            scene_no_render=True,
            scene_simulate=True,
            scene_simulate_mock_analyze=True,
            scene_fail_on_validation_error=True,
            scene_filters={"include_prim_paths": ["/World"]},
        )

    assert session_id == "session-1"
    data = fake_session.posts[0]["data"]
    assert data["large_scene"] == "true"
    assert data["coverage_policy"] == "allow_partial"
    assert data["scene_workers"] == "2"
    assert data["scene_assets"] == "AssetA,/World/AssetB"
    assert data["scene_resume"] == "true"
    assert data["scene_from_step"] == "predict"
    assert data["scene_skip_existing"] == "true"
    assert data["scene_no_render"] == "true"
    assert data["scene_simulate"] == "true"
    assert data["scene_simulate_mock_analyze"] == "true"
    assert data["scene_fail_on_validation_error"] == "true"
    assert json.loads(data["scene_filters"]) == {"include_prim_paths": ["/World"]}
    assert "scene_analyze_llm" not in data


def test_start_pipeline_preserves_explicit_strict_large_scene_policy(tmp_path):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    client.start_pipeline(
        usd_path=str(usd_path),
        large_scene=True,
        coverage_policy="strict",
    )

    assert fake_session.posts[0]["data"]["coverage_policy"] == "strict"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"cluster_min_prims": 0}, "cluster_min_prims must be at least 1"),
        (
            {"cluster_embedding_max_workers": 0},
            "cluster_embedding_max_workers must be at least 1",
        ),
        (
            {"cluster_embedding_batch_size": 0},
            "cluster_embedding_batch_size must be at least 1",
        ),
        ({"cluster_max_size": 0}, "cluster_max_size must be at least 1"),
        (
            {"cluster_similarity_threshold_low": -0.1},
            "cluster_similarity_threshold_low must be between 0.0 and 1.0",
        ),
        (
            {"cluster_similarity_threshold_medium": 1.1},
            "cluster_similarity_threshold_medium must be between 0.0 and 1.0",
        ),
    ],
)
def test_start_pipeline_rejects_bad_prim_clustering_overrides(tmp_path, kwargs, match):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    with pytest.raises(ValueError, match=match):
        client.start_pipeline(
            usd_path=str(usd_path),
            user_email="test@example.com",
            enable_prim_clustering=True,
            **kwargs,
        )

    assert fake_session.posts == []


def test_start_pipeline_rejects_worker_overrides_above_client_cap(
    monkeypatch, tmp_path
):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")
    monkeypatch.setenv("RENDER_NUM_WORKERS_MAX", "1")

    with pytest.raises(ValueError, match="render_num_workers must be between 1 and 1"):
        client.start_pipeline(
            usd_path=str(usd_path),
            user_email="test@example.com",
            render_num_workers=2,
        )

    assert fake_session.posts == []


def test_run_and_monitor_rejects_worker_overrides_before_upload(monkeypatch):
    client = MaterialAgentClient(base_url="http://service")
    monkeypatch.setenv("VLM_MAX_WORKERS_MAX", "1")

    def fail_upload(_usd_path: str) -> str:
        raise AssertionError("upload_usd should not be called")

    monkeypatch.setattr(client, "upload_usd", fail_upload)

    with pytest.raises(ValueError, match="vlm_max_workers must be between 1 and 1"):
        client.run_and_monitor(
            usd_path="/tmp/scene.usd",
            upload_first=True,
            vlm_max_workers=2,
            print_stream=False,
        )


def test_run_and_monitor_rejects_cluster_overrides_before_upload(monkeypatch):
    client = MaterialAgentClient(base_url="http://service")

    def fail_upload(_usd_path: str) -> str:
        raise AssertionError("upload_usd should not be called")

    monkeypatch.setattr(client, "upload_usd", fail_upload)

    with pytest.raises(ValueError, match="cluster_min_prims must be at least 1"):
        client.run_and_monitor(
            usd_path="/tmp/scene.usd",
            upload_first=True,
            enable_prim_clustering=True,
            cluster_min_prims=0,
            print_stream=False,
        )


def test_wait_for_input_render_stops_on_terminal_failure():
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    fake_session.head_responses.append(_FakeResponse(status_code=424))
    client._http = fake_session  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="Input preview render failed"):
        client.wait_for_input_render("session-1", poll_interval_seconds=0)


def test_wait_for_input_render_follows_presigned_redirects():
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    fake_session.head_responses.append(_FakeResponse(status_code=302))
    client._http = fake_session  # type: ignore[assignment]

    client.wait_for_input_render("session-1", poll_interval_seconds=0)

    assert fake_session.heads == [
        {
            "url": "http://service/assets/session-1/input-render",
            "timeout": client.timeout_seconds,
            "allow_redirects": True,
        }
    ]


def test_run_and_monitor_generates_reference_before_pipeline(monkeypatch):
    client = MaterialAgentClient(base_url="http://service")
    calls: list[str] = []
    captured_start_kwargs: dict = {}

    def upload_usd(usd_path: str) -> str:
        calls.append(f"upload:{usd_path}")
        return "session-1"

    def wait_for_input_render(session_id: str, timeout_seconds: int = 180, **_) -> None:
        calls.append(f"wait:{session_id}:{timeout_seconds}")

    def generate_reference_image(session_id: str, prompt: str) -> dict:
        calls.append(f"generate:{session_id}:{prompt}")
        return {"status": "ok", "reference_id": "ref-1"}

    def start_pipeline(**kwargs) -> str:
        captured_start_kwargs.update(kwargs)
        calls.append(f"start:{kwargs['session_id']}:{kwargs['generated_reference_id']}")
        return kwargs["session_id"]

    monkeypatch.setattr(client, "upload_usd", upload_usd)
    monkeypatch.setattr(client, "wait_for_input_render", wait_for_input_render)
    monkeypatch.setattr(client, "generate_reference_image", generate_reference_image)
    monkeypatch.setattr(client, "start_pipeline", start_pipeline)
    monkeypatch.setattr(client, "stream_events", lambda _session_id: iter(()))
    monkeypatch.setattr(
        client, "get_status", lambda _session_id: {"status": "completed"}
    )

    session_id, status = client.run_and_monitor(
        usd_path="/tmp/scene.usd",
        generated_reference_prompt="matte blue plastic",
        preview_timeout_seconds=12,
        vlm_max_workers=2,
        render_num_workers=1,
        print_stream=False,
    )

    assert session_id == "session-1"
    assert status == {"status": "completed"}
    assert calls == [
        "upload:/tmp/scene.usd",
        "wait:session-1:12",
        "generate:session-1:matte blue plastic",
        "start:session-1:ref-1",
    ]
    assert captured_start_kwargs["vlm_max_workers"] == 2
    assert captured_start_kwargs["render_num_workers"] == 1


def test_run_and_monitor_passes_prim_clustering_overrides(monkeypatch):
    client = MaterialAgentClient(base_url="http://service")
    captured_start_kwargs: dict = {}

    def start_pipeline(**kwargs) -> str:
        captured_start_kwargs.update(kwargs)
        return "session-1"

    monkeypatch.setattr(client, "start_pipeline", start_pipeline)
    monkeypatch.setattr(client, "stream_events", lambda _session_id: iter(()))
    monkeypatch.setattr(
        client, "get_status", lambda _session_id: {"status": "completed"}
    )

    session_id, status = client.run_and_monitor(
        usd_path="/tmp/scene.usd",
        enable_prim_clustering=True,
        cluster_min_prims=25,
        cluster_embedding_backend="nim",
        cluster_embedding_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        cluster_embedding_base_url="http://embedding-nim:8000/v1",
        cluster_embedding_max_workers=2,
        cluster_embedding_batch_size=8,
        cluster_max_size=11,
        cluster_similarity_threshold_low=0.97,
        cluster_similarity_threshold_medium=0.94,
        cluster_similarity_threshold_high=0.88,
        cluster_report=False,
        print_stream=False,
    )

    assert session_id == "session-1"
    assert status == {"status": "completed"}
    assert captured_start_kwargs["enable_prim_clustering"] is True
    assert captured_start_kwargs["cluster_min_prims"] == 25
    assert captured_start_kwargs["cluster_embedding_backend"] == "nim"
    assert (
        captured_start_kwargs["cluster_embedding_model"]
        == "nvidia/llama-nemotron-embed-vl-1b-v2"
    )
    assert (
        captured_start_kwargs["cluster_embedding_base_url"]
        == "http://embedding-nim:8000/v1"
    )
    assert captured_start_kwargs["cluster_embedding_max_workers"] == 2
    assert captured_start_kwargs["cluster_embedding_batch_size"] == 8
    assert captured_start_kwargs["cluster_max_size"] == 11
    assert captured_start_kwargs["cluster_similarity_threshold_low"] == 0.97
    assert captured_start_kwargs["cluster_similarity_threshold_medium"] == 0.94
    assert captured_start_kwargs["cluster_similarity_threshold_high"] == 0.88
    assert captured_start_kwargs["cluster_report"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"upload_first": True},
        {"generated_reference_prompt": "matte blue plastic"},
    ],
)
def test_run_and_monitor_rejects_preview_upload_modes_for_large_scene(
    monkeypatch,
    kwargs,
):
    client = MaterialAgentClient(base_url="http://service")

    def upload_usd(_usd_path: str) -> str:
        raise AssertionError("large_scene must not call upload_usd")

    monkeypatch.setattr(client, "upload_usd", upload_usd)

    with pytest.raises(ValueError, match="large_scene is not compatible"):
        client.run_and_monitor(
            usd_path="/tmp/scene.usda",
            large_scene=True,
            print_stream=False,
            **kwargs,
        )


def test_run_and_monitor_generated_reference_requires_reference_id(monkeypatch, capsys):
    client = MaterialAgentClient(base_url="http://service")

    monkeypatch.setattr(client, "upload_usd", lambda _path: "session-1")
    monkeypatch.setattr(client, "wait_for_input_render", lambda *_, **__: None)
    monkeypatch.setattr(client, "generate_reference_image", lambda *_: {"status": "ok"})

    with pytest.raises(RuntimeError, match="did not return reference_id"):
        client.run_and_monitor(
            usd_path="/tmp/scene.usda",
            generated_reference_prompt="matte blue plastic",
            print_stream=True,
            preview_timeout_seconds=5,
        )

    captured = capsys.readouterr()
    assert "Waiting for input preview" in captured.out
    assert "Generating reference image" in captured.out


def test_run_and_monitor_streams_progress_and_done(monkeypatch, capsys):
    client = MaterialAgentClient(base_url="http://service")

    monkeypatch.setattr(client, "start_pipeline", lambda **_: "session-1")
    monkeypatch.setattr(
        client,
        "stream_events",
        lambda _session_id: iter(
            [
                client_module.SSEMessage(event="ping", data=""),
                client_module.SSEMessage(
                    event="progress",
                    data=(
                        '{"step": "predict", "state": "running", '
                        '"overall_percent": 70, "message": "working"}'
                    ),
                ),
                client_module.SSEMessage(event="progress", data="not-json"),
                client_module.SSEMessage(event="done", data="{}"),
            ]
        ),
    )
    monkeypatch.setattr(
        client, "get_status", lambda _session_id: {"status": "completed"}
    )

    session_id, status = client.run_and_monitor(
        usd_path="/tmp/scene.usda",
        print_stream=True,
    )

    assert session_id == "session-1"
    assert status == {"status": "completed"}
    captured = capsys.readouterr()
    assert "Started session: session-1" in captured.out
    assert "[predict] running overall=70% working" in captured.out
    assert "[None] None overall=None%" in captured.out


def test_run_and_monitor_confirms_sse_done_before_returning(monkeypatch):
    client = MaterialAgentClient(base_url="http://service")
    statuses = iter(
        [
            {"status": "running"},
            {"status": "running"},
            {"status": "completed"},
        ]
    )
    sleeps: list[float] = []

    monkeypatch.setattr(client, "start_pipeline", lambda **_: "session-1")
    monkeypatch.setattr(
        client,
        "stream_events",
        lambda _session_id: iter([client_module.SSEMessage(event="done", data="{}")]),
    )
    monkeypatch.setattr(client, "get_status", lambda _session_id: next(statuses))
    monkeypatch.setattr(client_module.time, "sleep", lambda delay: sleeps.append(delay))

    session_id, status = client.run_and_monitor(
        usd_path="/tmp/scene.usda",
        print_stream=False,
    )

    assert session_id == "session-1"
    assert status == {"status": "completed"}
    assert sleeps == [2]


def test_run_and_monitor_polls_when_post_done_status_fetch_fails(monkeypatch):
    client = MaterialAgentClient(base_url="http://service")
    calls = {"count": 0}

    def get_status(_session_id: str):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("status not replicated yet")
        return {"status": "completed"}

    monkeypatch.setattr(client, "start_pipeline", lambda **_: "session-1")
    monkeypatch.setattr(
        client,
        "stream_events",
        lambda _session_id: iter([client_module.SSEMessage(event="done", data="{}")]),
    )
    monkeypatch.setattr(client, "get_status", get_status)

    session_id, status = client.run_and_monitor(
        usd_path="/tmp/scene.usda",
        print_stream=False,
    )

    assert session_id == "session-1"
    assert status == {"status": "completed"}
    assert calls["count"] == 2


def test_run_and_monitor_retries_sse_then_polls(monkeypatch, capsys):
    client = MaterialAgentClient(base_url="http://service")
    stream_calls = {"count": 0}
    statuses = [{"status": "running", "overall_percent": 10}, {"status": "completed"}]
    sleeps: list[float] = []

    def stream_events(_session_id: str):
        stream_calls["count"] += 1
        if stream_calls["count"] == 1:
            raise RuntimeError("sse down")
        return iter(())

    def get_status(_session_id: str):
        if statuses:
            return statuses.pop(0)
        raise RuntimeError("status unavailable")

    monkeypatch.setattr(client, "start_pipeline", lambda **_: "session-1")
    monkeypatch.setattr(client, "stream_events", stream_events)
    monkeypatch.setattr(client, "get_status", get_status)
    monkeypatch.setattr(client_module.time, "sleep", lambda delay: sleeps.append(delay))

    session_id, status = client.run_and_monitor(
        usd_path="/tmp/scene.usda",
        print_stream=True,
        reconnect_attempts=1,
        reconnect_backoff_seconds=4,
    )

    assert session_id == "session-1"
    assert status == {"status": "completed"}
    assert sleeps == [4, 2]
    captured = capsys.readouterr()
    assert "SSE error (sse down), retrying in 4s" in captured.out
    assert "Polling status" in captured.out
    assert "status=running overall=10" in captured.out
    assert "status=completed overall=-" in captured.out


def test_main_passes_worker_overrides(monkeypatch, tmp_path, capsys):
    captured_kwargs: dict = {}

    class FakeClient:
        def __init__(self, base_url: str, token: str | None = None):
            self.base_url = base_url

        def run_and_monitor(self, **kwargs):
            captured_kwargs.update(kwargs)
            return "session-1", {
                "status": "completed",
                "coverage": {
                    "readiness_grade": "complete",
                    "prediction_coverage_ratio": 1.0,
                    "binding_coverage_ratio": 1.0,
                },
            }

    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    monkeypatch.setattr(client_module, "MaterialAgentClient", FakeClient)

    exit_code = client_module.main(
        [
            "--base-url",
            "http://service",
            "--email",
            "test@example.com",
            "--vlm-max-workers",
            "2",
            "--render-num-workers",
            "1",
            "--quiet",
            str(usd_path),
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["vlm_max_workers"] == 2
    assert captured_kwargs["render_num_workers"] == 1
    assert captured_kwargs["user_email"] == "test@example.com"
    assert captured_kwargs["enable_prim_clustering"] is None
    assert captured_kwargs["coverage_policy"] == "strict"
    captured = capsys.readouterr()
    assert "Session: session-1" in captured.out
    assert "Material readiness: complete" in captured.out


def test_main_email_is_optional(monkeypatch, tmp_path, capsys):
    captured_kwargs: dict = {}

    class FakeClient:
        def __init__(self, base_url: str, token: str | None = None):
            self.base_url = base_url

        def run_and_monitor(self, **kwargs):
            captured_kwargs.update(kwargs)
            return "session-1", {
                "status": "completed",
                "coverage": {"readiness_grade": "complete"},
            }

    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    monkeypatch.setattr(client_module, "MaterialAgentClient", FakeClient)

    exit_code = client_module.main(
        [
            "--base-url",
            "http://service",
            "--quiet",
            str(usd_path),
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["user_email"] == ""
    captured = capsys.readouterr()
    assert "Session: session-1" in captured.out


def test_main_can_explicitly_disable_prim_clustering(monkeypatch, tmp_path, capsys):
    captured_kwargs: dict = {}

    class FakeClient:
        def __init__(self, base_url: str, token: str | None = None):
            self.base_url = base_url

        def run_and_monitor(self, **kwargs):
            captured_kwargs.update(kwargs)
            return "session-1", {
                "status": "completed",
                "coverage": {"readiness_grade": "complete"},
            }

    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    monkeypatch.setattr(client_module, "MaterialAgentClient", FakeClient)

    exit_code = client_module.main(
        [
            "--base-url",
            "http://service",
            "--email",
            "test@example.com",
            "--disable-prim-clustering",
            "--quiet",
            str(usd_path),
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["enable_prim_clustering"] is False
    captured = capsys.readouterr()
    assert "Session: session-1" in captured.out


def test_main_passes_prim_clustering_overrides(monkeypatch, tmp_path, capsys):
    captured_kwargs: dict = {}

    class FakeClient:
        def __init__(self, base_url: str, token: str | None = None):
            self.base_url = base_url

        def run_and_monitor(self, **kwargs):
            captured_kwargs.update(kwargs)
            return "session-1", {
                "status": "completed",
                "coverage": {"readiness_grade": "complete"},
            }

    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    monkeypatch.setattr(client_module, "MaterialAgentClient", FakeClient)

    exit_code = client_module.main(
        [
            "--base-url",
            "http://service",
            "--email",
            "test@example.com",
            "--enable-prim-clustering",
            "--cluster-min-prims",
            "25",
            "--cluster-embedding-backend",
            "nim",
            "--cluster-embedding-model",
            "nvidia/llama-nemotron-embed-vl-1b-v2",
            "--cluster-embedding-base-url",
            "http://embedding-nim:8000/v1",
            "--cluster-embedding-max-workers",
            "2",
            "--cluster-embedding-batch-size",
            "8",
            "--cluster-max-size",
            "11",
            "--cluster-similarity-threshold-low",
            "0.97",
            "--cluster-similarity-threshold-medium",
            "0.94",
            "--cluster-similarity-threshold-high",
            "0.88",
            "--no-cluster-report",
            "--quiet",
            str(usd_path),
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["enable_prim_clustering"] is True
    assert captured_kwargs["cluster_min_prims"] == 25
    assert captured_kwargs["cluster_embedding_backend"] == "nim"
    assert (
        captured_kwargs["cluster_embedding_model"]
        == "nvidia/llama-nemotron-embed-vl-1b-v2"
    )
    assert (
        captured_kwargs["cluster_embedding_base_url"] == "http://embedding-nim:8000/v1"
    )
    assert captured_kwargs["cluster_embedding_max_workers"] == 2
    assert captured_kwargs["cluster_embedding_batch_size"] == 8
    assert captured_kwargs["cluster_max_size"] == 11
    assert captured_kwargs["cluster_similarity_threshold_low"] == 0.97
    assert captured_kwargs["cluster_similarity_threshold_medium"] == 0.94
    assert captured_kwargs["cluster_similarity_threshold_high"] == 0.88
    assert captured_kwargs["cluster_report"] is False
    captured = capsys.readouterr()
    assert "Session: session-1" in captured.out


def test_main_passes_large_scene_options(monkeypatch, tmp_path, capsys):
    captured_kwargs: dict = {}

    class FakeClient:
        def __init__(self, base_url: str, token: str | None = None):
            self.base_url = base_url

        def run_and_monitor(self, **kwargs):
            captured_kwargs.update(kwargs)
            return "session-1", {"status": "completed"}

    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    monkeypatch.setattr(client_module, "MaterialAgentClient", FakeClient)

    exit_code = client_module.main(
        [
            "--base-url",
            "http://service",
            "--email",
            "test@example.com",
            "--large-scene",
            "--scene-workers",
            "2",
            "--scene-assets",
            "AssetA,/World/AssetB",
            "--scene-resume",
            "--scene-from-step",
            "predict",
            "--scene-skip-existing",
            "--scene-no-render",
            "--scene-simulate",
            "--scene-simulate-mock-analyze",
            "--scene-fail-on-validation-error",
            "--scene-filters-json",
            '{"include_prim_paths": ["/World"]}',
            "--quiet",
            str(usd_path),
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["large_scene"] is True
    assert captured_kwargs["coverage_policy"] == "allow_partial"
    assert captured_kwargs["scene_workers"] == 2
    assert captured_kwargs["scene_assets"] == "AssetA,/World/AssetB"
    assert captured_kwargs["scene_resume"] is True
    assert captured_kwargs["scene_from_step"] == "predict"
    assert captured_kwargs["scene_skip_existing"] is True
    assert captured_kwargs["scene_no_render"] is True
    assert captured_kwargs["scene_simulate"] is True
    assert captured_kwargs["scene_simulate_mock_analyze"] is True
    assert captured_kwargs["scene_fail_on_validation_error"] is True
    assert captured_kwargs["scene_filters"] == {"include_prim_paths": ["/World"]}
    captured = capsys.readouterr()
    assert "defaults coverage_policy to allow_partial" in captured.err
    assert "Scene manifest:" in captured.out
    assert "Predictions JSONL" not in captured.out
    assert "Report HTML" not in captured.out


def test_main_prints_large_scene_final_render_when_render_enabled(
    monkeypatch, tmp_path, capsys
):
    class FakeClient:
        def __init__(self, base_url: str, token: str | None = None):
            self.base_url = base_url

        def run_and_monitor(self, **kwargs):
            return "session-1", {"status": "completed"}

    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")
    monkeypatch.setattr(client_module, "MaterialAgentClient", FakeClient)

    exit_code = client_module.main(
        ["--base-url", "http://service", "--large-scene", "--quiet", str(usd_path)]
    )

    assert exit_code == 0
    assert "Final render:" in capsys.readouterr().out


def test_main_prints_no_results_when_status_missing(monkeypatch, tmp_path, capsys):
    class FakeClient:
        def __init__(self, base_url: str, token: str | None = None):
            self.base_url = base_url

        def run_and_monitor(self, **kwargs):
            return "session-1", None

    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")
    monkeypatch.setattr(client_module, "MaterialAgentClient", FakeClient)

    assert client_module.main(["--quiet", str(usd_path)]) == 1
    assert "No results available yet." in capsys.readouterr().out


@pytest.mark.parametrize(
    "status",
    [
        {"status": "failed"},
        {"status": "cancelled"},
        {"status": "running"},
        {"status": "completed"},
        {
            "status": "completed",
            "coverage": {"readiness_grade": "partial"},
        },
    ],
)
def test_main_returns_nonzero_until_strict_completion_contract_is_met(
    monkeypatch,
    tmp_path,
    status,
):
    class FakeClient:
        def __init__(self, base_url: str, token: str | None = None):
            self.base_url = base_url

        def run_and_monitor(self, **kwargs):
            return "session-1", status

    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")
    monkeypatch.setattr(client_module, "MaterialAgentClient", FakeClient)

    assert client_module.main(["--quiet", str(usd_path)]) == 1


def test_main_rejects_description_count_mismatches(tmp_path, capsys):
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    assert (
        client_module.main(
            [
                "--ref",
                "a.png",
                "--ref-desc",
                "one",
                "--ref-desc",
                "two",
                str(usd_path),
            ]
        )
        == 2
    )
    assert "--ref and --ref-desc counts must match" in capsys.readouterr().err

    assert (
        client_module.main(
            [
                "--ref-pdf",
                "a.pdf",
                "--pdf-desc",
                "one",
                "--pdf-desc",
                "two",
                str(usd_path),
            ]
        )
        == 2
    )
    assert "--ref-pdf and --pdf-desc counts must match" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args,error",
    [
        (["--scene-filters-json", "[]"], "--scene-filters-json must be a JSON object"),
        (
            ["--large-scene", "--upload-first"],
            "--large-scene is not compatible with --upload-first",
        ),
        (
            ["--large-scene", "--enable-material-generation"],
            "--large-scene is not compatible with --enable-material-generation",
        ),
    ],
)
def test_main_parser_error_paths(args, error, tmp_path, capsys):
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    with pytest.raises(SystemExit) as exc_info:
        client_module.main([*args, str(usd_path)])

    assert exc_info.value.code == 2
    assert error in capsys.readouterr().err


@pytest.mark.parametrize(
    "kwargs,not_found_in_report",
    [
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../material_agent/data/examples/ladder/sources/usd/ladder.usd",
                ),
                "user_email": "test@nvidia.com",
            },
            None,
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../material_agent/data/examples/ladder/sources/usd/ladder.usd",
                ),
                "user_email": "test@nvidia.com",
                "reference_images": [
                    os.path.join(
                        os.path.dirname(__file__),
                        "../../../material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg",
                    ),
                    os.path.join(
                        os.path.dirname(__file__),
                        "../../../material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg",
                    ),
                ],
            },
            None,
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../material_agent/data/examples/ladder/sources/usd/ladder.usd",
                ),
                "user_email": "test@nvidia.com",
                "reference_images": [
                    os.path.join(
                        os.path.dirname(__file__),
                        "../../../material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg",
                    ),
                    os.path.join(
                        os.path.dirname(__file__),
                        "../../../material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg",
                    ),
                ],
                "reference_descriptions": [
                    "This is a reference image of the ladder (front view) that you can use to identify the material of the parts.",
                    "This is a reference image of the ladder (rear view) that you can use to identify the material of the parts.",
                ],
            },
            None,
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../material_agent/data/examples/ladder/sources/usd/ladder.usd",
                ),
                "user_email": "test@nvidia.com",
                "user_prompt": "Identify what the object part is, and then select a material from the predefined list of materials for this highlighted object part. Provide the identified object part in the reasoning.",
            },
            None,
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../material_agent/data/examples/ladder/sources/usd/ladder.usd",
                ),
                "user_email": "test@nvidia.com",
                "camera_views": "+x+y+z",
            },
            ["-x+y+z"],
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../material_agent/data/examples/ladder/sources/usd/ladder.usd",
                ),
                "user_email": "test@nvidia.com",
                "optimize_usd": True,
            },
            None,
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../material_agent/data/examples/ladder/sources/usd/ladder.usd",
                ),
                "user_email": "test@nvidia.com",
                "optimize_usd": False,
            },
            None,
        ),
    ],
)
@pytest.mark.skipif(
    os.getenv("RUN_CLIENT_TESTS", "false").lower() not in ["true", "1", "yes", "y"],
    reason="Skipping test in CI",
)
async def test_basic_pipeline(
    api_client: MaterialAgentClient, kwargs: dict, not_found_in_report: list[str] | None
):
    session_id, status = api_client.run_and_monitor(**kwargs)
    assert status is not None
    assert status["status"] == "completed"
    await asyncio.sleep(5)
    report = api_client._http.get(
        f"{api_client.base_url}/artifacts/{session_id}/report"
    )
    assert report.status_code == 200
    report_text = report.text
    if "reference_images" in kwargs:
        response = api_client._http.get(
            f"{api_client.base_url}/assets/{session_id}/references"
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response is not None
        assert json_response["references"] is not None
        assert len(json_response["references"]) == len(kwargs["reference_images"])

        if "reference_descriptions" not in kwargs:
            # make sure reference images were used and are mentioned in the report
            for ind in range(len(json_response["references"])):
                assert (
                    f"This is reference image {ind + 1} of the asset you will match this look exactly"
                    in report_text
                )
                print("default reference descriptions found in report")
        else:
            assert len(kwargs["reference_descriptions"]) == len(
                json_response["references"]
            )
            print(
                f"reference_descriptions: {kwargs['reference_descriptions']} found in report"
            )
            for description in kwargs["reference_descriptions"]:
                assert description in report_text
                print(f"reference_description: {description} found in report")

    if "user_prompt" in kwargs:
        assert kwargs["user_prompt"] in report_text
        print(f"user_prompt: {kwargs['user_prompt']} found in report")

    if "camera_views" in kwargs:
        for camera_view in kwargs["camera_views"].split(","):
            assert camera_view.strip().lower() in report_text.lower()
            print(f"camera_view: {camera_view.strip().lower()} found in report")

    if "optimize_usd" in kwargs:
        optimization_report = api_client._http.get(
            f"{api_client.base_url}/artifacts/{session_id}/optimization-report"
        )
        # Check whether optimize_usd actually ran (it may be silently skipped
        # when the Scene Optimizer backend is unavailable in the test environment).
        completed_step_names = [s["name"] for s in status.get("completed_steps", [])]
        optimize_actually_ran = "optimize_usd" in completed_step_names

        if kwargs["optimize_usd"] and optimize_actually_ran:
            assert optimization_report.status_code == 200
            optimization_report_json = optimization_report.json()
            assert "report" in optimization_report_json
            assert "operations_executed" in optimization_report_json
            assert len(optimization_report_json["operations_executed"]) > 0
        else:
            assert optimization_report.status_code == 404

    if not_found_in_report:
        for not_found in not_found_in_report:
            assert not_found.strip().lower() not in report_text.lower()
            print(
                f"not_found_in_report: {not_found.strip().lower()} not found in report"
            )
