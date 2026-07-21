# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import json
from pathlib import Path
from unittest.mock import patch

import httpx

from texture_agent.functions.rest_client import RestTextureVariationClient
from texture_agent.functions.texture_generation import (
    BackendCapabilities,
    Conditioning,
    TextureTarget,
    TextureVariationConfig,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "issue116_projection_backend"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_build_request_body_includes_projection_contract_fields() -> None:
    body = RestTextureVariationClient._build_request_body(
        source_asset_uri="file:///work/prepared.usd",
        conditioning=Conditioning(
            text_prompt="scuffed aluminum",
            reference_image_uris=["file:///ref.png"],
            multiview_image_uris=["file:///view0.png"],
        ),
        target=TextureTarget(
            material_name="Aluminum_Matte",
            material_path="/RootNode/Looks/Aluminum_Matte",
            prim_paths=["/RootNode/Ladder"],
        ),
        config=TextureVariationConfig(
            strength=0.8,
            seed=11631,
            variant_name="Aluminum_Matte",
            engine="fake_projection",
            texture_size=1024,
            custom_parameters={"variant": "success_full_pbr"},
        ),
        capabilities=BackendCapabilities(
            image_conditioning=True,
            multiview=False,
            normal_map=True,
            orm=True,
            masks=True,
            coverage=True,
            geometry_output="none",
        ),
    )

    assert body["source_asset_uri"] == "file:///work/prepared.usd"
    assert body["target"]["material_path"] == "/RootNode/Looks/Aluminum_Matte"
    assert body["target"]["strict_scope"] is True
    assert body["conditioning"]["turntable_video_uri"] is None
    assert body["conditioning"]["multiview_image_uris"] == ["file:///view0.png"]
    assert body["configuration"]["texture_size"] == 1024
    assert body["capabilities"]["geometry_output"] == "none"


def test_build_request_body_matches_issue116_request_fixture() -> None:
    body = RestTextureVariationClient._build_request_body(
        source_asset_uri="file:///work/ladder/.issue116/prepared/ladder_uv_ready.usd",
        conditioning=Conditioning(
            text_prompt="brushed aluminum with visible scuffs",
            reference_image_uris=[
                "file:///work/ladder/references/scuffed_aluminum.png"
            ],
        ),
        target=TextureTarget(
            material_name="Aluminum_Matte",
            material_path="/RootNode/Looks/Aluminum_Matte",
            prim_paths=["/RootNode/SM_Ladder_A/SM_Ladder_A_Aluminum_0"],
            mode="per_material",
            strict_scope=True,
        ),
        config=TextureVariationConfig(
            strength=0.8,
            seed=11631,
            variant_name="ladder_aluminum_scuffed",
            engine="fake_projection",
            texture_size=1024,
            custom_parameters={"variant": "success_full_pbr"},
        ),
        capabilities=BackendCapabilities(
            image_conditioning=True,
            multiview=False,
            normal_map=True,
            orm=True,
            masks=True,
            coverage=True,
            geometry_output="none",
        ),
    )

    assert body == _fixture("request_ladder_aluminum_matte.json")


def test_parse_status_preserves_normalized_projection_response() -> None:
    status = RestTextureVariationClient._parse_status(
        _fixture("response_success_full_pbr.json")
    )

    assert status.status == "completed"
    assert status.result is not None
    assert set(status.result.maps) == {"albedo", "normal", "orm"}
    assert status.result.maps["orm"].packing == "r=occlusion,g=roughness,b=metalness"
    assert status.result.metadata["backend_name"] == "fake_projection_backend"
    assert status.result.metadata["capabilities"]["coverage"] is True


def test_parse_status_preserves_degraded_channels_and_diagnostics() -> None:
    status = RestTextureVariationClient._parse_status(
        _fixture("response_degraded_low_coverage.json")
    )

    assert status.status == "completed"
    assert status.result is not None
    assert status.result.generated_textures.normal is None
    assert status.result.generated_textures.orm is None
    assert status.result.metadata["degraded_channels"] == ["normal", "orm"]
    assert {item["code"] for item in status.result.diagnostics} >= {
        "BACKEND_MAP_MISSING",
        "BACKEND_LOW_COVERAGE",
        "BACKEND_GEOMETRY_IGNORED",
    }


def test_parse_status_keeps_failed_missing_albedo_result_for_diagnostics() -> None:
    status = RestTextureVariationClient._parse_status(
        _fixture("response_failure_missing_albedo.json")
    )

    assert status.status == "failed"
    assert status.result is not None
    assert status.result.generated_textures.albedo is None
    assert set(status.result.maps) == {"normal"}
    assert status.result.diagnostics[0]["code"] == "BACKEND_MAP_MISSING"


def test_get_status_retries_transient_transport_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        "texture_agent.functions.rest_client.time.sleep",
        lambda _delay: None,
    )
    request = httpx.Request("GET", "http://texture-service/v1/texture-variations/job")

    class FlakyClient:
        calls = 0

        def get(self, url: str) -> httpx.Response:
            self.calls += 1
            if self.calls == 1:
                raise httpx.ConnectError(
                    "[Errno -3] Temporary failure in name resolution",
                    request=request,
                )
            return httpx.Response(
                200,
                json={"job_id": "job", "status": "completed", "progress": 100},
                request=httpx.Request("GET", url),
            )

    flaky_client = FlakyClient()

    status = RestTextureVariationClient("http://texture-service").get_status(
        "job",
        client=flaky_client,  # type: ignore[arg-type]
    )

    assert flaky_client.calls == 2
    assert status.job_id == "job"
    assert status.status == "completed"


def test_submit_retries_busy_backpressure(monkeypatch) -> None:
    monkeypatch.setattr(
        "texture_agent.functions.rest_client.time.sleep",
        lambda _delay: None,
    )

    class BusyThenAcceptedClient:
        calls = 0

        def post(self, url: str, json: dict) -> httpx.Response:
            self.calls += 1
            request = httpx.Request("POST", url)
            if self.calls == 1:
                return httpx.Response(
                    429,
                    json={"detail": "Texture generation queue is full."},
                    headers={"Retry-After": "1"},
                    request=request,
                )
            return httpx.Response(
                202,
                json={"job_id": "job", "status": "queued", "progress": 0},
                request=request,
            )

    fake_client = BusyThenAcceptedClient()
    response = RestTextureVariationClient(
        "http://texture-service",
        submit_retry_timeout_sec=10,
    )._post_with_backpressure_retry(
        fake_client,  # type: ignore[arg-type]
        "http://texture-service/v1/texture-variations",
        {},
    )

    assert fake_client.calls == 2
    assert response.status_code == 202


def test_submit_retries_transient_transport_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        "texture_agent.functions.rest_client.time.sleep",
        lambda _delay: None,
    )
    request = httpx.Request("POST", "http://texture-service/v1/texture-variations")

    class FlakySubmitClient:
        calls = 0

        def post(self, url: str, json: dict) -> httpx.Response:
            self.calls += 1
            if self.calls == 1:
                raise httpx.ConnectError("temporary restart", request=request)
            return httpx.Response(
                202,
                json={"job_id": "job", "status": "queued", "progress": 0},
                request=httpx.Request("POST", url),
            )

    fake_client = FlakySubmitClient()
    response = RestTextureVariationClient(
        "http://texture-service",
        submit_retry_timeout_sec=10,
        submit_retry_base_delay_sec=0.1,
    )._post_with_backpressure_retry(
        fake_client,  # type: ignore[arg-type]
        "http://texture-service/v1/texture-variations",
        {},
    )

    assert fake_client.calls == 2
    assert response.status_code == 202


def test_submit_does_not_retry_backend_not_ready(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "texture_agent.functions.rest_client.time.sleep",
        lambda delay: sleeps.append(delay),
    )

    class BackendNotReadyClient:
        calls = 0

        def post(self, url: str, json: dict) -> httpx.Response:
            self.calls += 1
            return httpx.Response(
                503,
                json={"detail": "Step1X runtime is not configured."},
                request=httpx.Request("POST", url),
            )

    fake_client = BackendNotReadyClient()
    response = RestTextureVariationClient(
        "http://texture-service",
        submit_retry_timeout_sec=3600,
    )._post_with_backpressure_retry(
        fake_client,  # type: ignore[arg-type]
        "http://texture-service/v1/texture-variations",
        {},
    )

    assert fake_client.calls == 1
    assert response.status_code == 503
    assert sleeps == []


def test_generate_returns_failed_status_on_submit_transport_error() -> None:
    class SubmitTransportErrorClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

        def post(self, url: str, json: dict) -> httpx.Response:
            request = httpx.Request("POST", url)
            raise httpx.ConnectError("connection refused", request=request)

    client = RestTextureVariationClient(
        "http://texture-service",
        submit_retry_timeout_sec=0,
    )

    with patch("httpx.Client", SubmitTransportErrorClient):
        status = client.generate(
            source_asset_uri="file:///tmp/model.usd",
            conditioning=Conditioning(text_prompt="aged leather"),
            config=TextureVariationConfig(strength=0.8),
            wait=False,
        )

    assert status.status == "failed"
    assert status.error_message is not None
    assert "Texture service submit failed" in status.error_message
    assert "connection refused" in status.error_message


def test_generate_returns_failed_status_on_poll_transport_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "texture_agent.functions.rest_client.time.sleep",
        lambda _delay: None,
    )

    class PollTransportErrorClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

        def post(self, url: str, json: dict) -> httpx.Response:
            return httpx.Response(
                202,
                json={"job_id": "job", "status": "queued", "progress": 0},
                request=httpx.Request("POST", url),
            )

        def get(self, url: str) -> httpx.Response:
            request = httpx.Request("GET", url)
            raise httpx.ConnectError("temporary poll failure", request=request)

    client = RestTextureVariationClient("http://texture-service")

    with patch("httpx.Client", PollTransportErrorClient):
        status = client.generate(
            source_asset_uri="file:///tmp/model.usd",
            conditioning=Conditioning(text_prompt="aged leather"),
            config=TextureVariationConfig(strength=0.8),
            timeout_sec=10,
        )

    assert status.job_id == "job"
    assert status.status == "failed"
    assert status.error_message is not None
    assert "Texture service poll failed for job job" in status.error_message


def test_generate_timeout_returns_explicit_failed_status(monkeypatch) -> None:
    monkeypatch.setattr(
        "texture_agent.functions.rest_client.time.sleep",
        lambda _delay: None,
    )
    times = itertools.count(0.0, 2.0)
    monkeypatch.setattr(
        "texture_agent.functions.rest_client.time.time",
        lambda: next(times),
    )

    class AcceptedClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self) -> AcceptedClient:
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict) -> httpx.Response:
            return httpx.Response(
                202,
                json={"job_id": "job-timeout", "status": "queued", "progress": 0},
                request=httpx.Request("POST", url),
            )

        def get(self, url: str) -> httpx.Response:
            return httpx.Response(
                200,
                json={"job_id": "job-timeout", "status": "queued", "progress": 0},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(
        "texture_agent.functions.rest_client.httpx.Client",
        AcceptedClient,
    )

    status = RestTextureVariationClient("http://texture-service").generate(
        source_asset_uri="file:///asset.usd",
        conditioning=Conditioning(text_prompt="rusted paint"),
        config=TextureVariationConfig(strength=0.8),
        wait=True,
        timeout_sec=1,
    )

    assert status.status == "failed"
    assert status.error_message is not None
    assert "Timed out waiting for job job-timeout after 1s" in status.error_message
