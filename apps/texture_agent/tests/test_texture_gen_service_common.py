# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import threading
import time
import zipfile
from pathlib import Path

import pytest
from apps.texture_gen_service_common import (
    BackendCapabilities,
    BackendHealth,
    Conditioning,
    CreateJobRequest,
    GeneratedTextures,
    GenerationResult,
    JobStatus,
    MapArtifact,
    TextureGenerationBackend,
    TextureGenerationBackendError,
    TextureVariationService,
    create_app,
    local_file_uri,
    local_path_from_file_uri,
    require_visible_file,
)
from apps.texture_gen_service_common import service as service_module
from apps.texture_gen_service_common import usd_package as service_usd_package
from apps.texture_gen_service_common.service import _JobRecord
from apps.texture_gen_simple_service.client.client import TextureVariationClient
from fastapi.testclient import TestClient
from PIL import Image

from texture_agent.functions.rest_client import RestTextureVariationClient
from texture_agent.functions.texture_generation import (
    BackendCapabilities as ClientBackendCapabilities,
)
from texture_agent.functions.texture_generation import (
    Conditioning as ClientConditioning,
)
from texture_agent.functions.texture_generation import (
    TextureTarget as ClientTextureTarget,
)
from texture_agent.functions.texture_generation import (
    TextureVariationConfig as ClientTextureVariationConfig,
)


class _ImmediateBackend(TextureGenerationBackend):
    def __init__(self) -> None:
        self.requests: list[CreateJobRequest] = []

    @property
    def name(self) -> str:
        return "immediate-test-backend"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            image_conditioning=True,
            multiview=True,
            normal_map=False,
            orm=False,
            masks=True,
            coverage=True,
            geometry_output="none",
        )

    def health(self) -> BackendHealth:
        return BackendHealth(
            ready=True,
            warmup_complete=True,
            gpu_available=False,
            capabilities=self.capabilities(),
        )

    def generate(
        self,
        request: CreateJobRequest,
        *,
        job_id: str,
        output_dir: Path,
        cancel_event: threading.Event,
    ) -> GenerationResult:
        self.requests.append(request)
        albedo = output_dir / "albedo.png"
        albedo.write_bytes(b"fake-png")
        return GenerationResult(
            variant_asset_uri=request.source_asset_uri,
            variant_name=request.configuration.variant_name or job_id,
            generated_textures=GeneratedTextures(albedo=albedo.as_uri()),
            maps={
                "albedo": MapArtifact(
                    uri=albedo.as_uri(),
                    width=request.configuration.texture_size,
                    height=request.configuration.texture_size,
                    colorspace="srgb",
                )
            },
            metadata={
                "backend_name": self.name,
                "degraded_channels": ["normal", "orm"],
            },
            diagnostics=[
                {
                    "code": "BACKEND_MAP_MISSING",
                    "severity": "warning",
                    "message": "normal/orm unavailable in test backend",
                }
            ],
        )


class _BlockingBackend(_ImmediateBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    @property
    def name(self) -> str:
        return "blocking-test-backend"

    def generate(
        self,
        request: CreateJobRequest,
        *,
        job_id: str,
        output_dir: Path,
        cancel_event: threading.Event,
    ) -> GenerationResult:
        self.started.set()
        self.release.wait(timeout=5)
        return super().generate(
            request,
            job_id=job_id,
            output_dir=output_dir,
            cancel_event=cancel_event,
        )


class _FailingBackend(_ImmediateBackend):
    @property
    def name(self) -> str:
        return "failing-test-backend"

    def generate(
        self,
        request: CreateJobRequest,
        *,
        job_id: str,
        output_dir: Path,
        cancel_event: threading.Event,
    ) -> GenerationResult:
        raise RuntimeError("intentional backend failure")


class _PartialResultFailingBackend(_ImmediateBackend):
    @property
    def name(self) -> str:
        return "partial-result-failing-test-backend"

    def generate(
        self,
        request: CreateJobRequest,
        *,
        job_id: str,
        output_dir: Path,
        cancel_event: threading.Event,
    ) -> GenerationResult:
        albedo = output_dir / "partial-albedo.png"
        albedo.write_bytes(b"partial-png")
        partial = GenerationResult(
            variant_asset_uri=request.source_asset_uri,
            variant_name="partial-result",
            generated_textures=GeneratedTextures(albedo=albedo.as_uri()),
        )
        raise TextureGenerationBackendError(
            "intentional partial backend failure",
            result=partial,
        )


class _NotReadyBackend(_ImmediateBackend):
    @property
    def name(self) -> str:
        return "not-ready-test-backend"

    def health(self) -> BackendHealth:
        return BackendHealth(
            status="not_ready",
            ready=False,
            capabilities=self.capabilities(),
            error="runtime missing",
        )


class _DefaultHealthBackend(TextureGenerationBackend):
    @property
    def name(self) -> str:
        return "default-health-test-backend"

    def generate(
        self,
        request: CreateJobRequest,
        *,
        job_id: str,
        output_dir: Path,
        cancel_event: threading.Event,
    ) -> GenerationResult:
        return GenerationResult(
            variant_asset_uri=request.source_asset_uri,
            variant_name=job_id,
            generated_textures=GeneratedTextures(),
        )


class _CancelFailingBackend(_ImmediateBackend):
    @property
    def name(self) -> str:
        return "cancel-failing-test-backend"

    def generate(
        self,
        request: CreateJobRequest,
        *,
        job_id: str,
        output_dir: Path,
        cancel_event: threading.Event,
    ) -> GenerationResult:
        cancel_event.set()
        raise RuntimeError("backend stopped after cancellation")


class _CancelAfterResultBackend(_ImmediateBackend):
    @property
    def name(self) -> str:
        return "cancel-after-result-test-backend"

    def generate(
        self,
        request: CreateJobRequest,
        *,
        job_id: str,
        output_dir: Path,
        cancel_event: threading.Event,
    ) -> GenerationResult:
        result = super().generate(
            request,
            job_id=job_id,
            output_dir=output_dir,
            cancel_event=cancel_event,
        )
        cancel_event.set()
        return result


def _request(variant: str = "variant") -> CreateJobRequest:
    return CreateJobRequest(
        source_asset_uri="file:///work/asset.usd",
        conditioning=Conditioning(multiview_image_uris=["file:///work/view0.png"]),
        configuration={
            "variant_name": variant,
            "engine": "test",
            "texture_size": 16,
        },
        target={
            "material_name": "Steel",
            "material_path": "/World/Looks/Steel",
            "prim_paths": ["/World/Mesh"],
            "mode": "per_material",
            "strict_scope": True,
        },
        capabilities={"normal_map": False, "orm": False, "masks": True},
    )


def test_conditioning_accepts_multiview_without_text_prompt() -> None:
    conditioning = Conditioning(multiview_image_uris=["file:///view.png"])

    assert conditioning.text_prompt is None
    assert conditioning.multiview_image_uris == ["file:///view.png"]


def test_conditioning_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="At least one non-empty conditioning"):
        Conditioning()


def test_backend_defaults_and_error_result_payload() -> None:
    backend = _DefaultHealthBackend()
    result = GenerationResult(
        variant_asset_uri="file:///work/asset.usd",
        variant_name="partial",
        generated_textures=GeneratedTextures(albedo="file:///work/albedo.png"),
    )
    error = TextureGenerationBackendError("partial failure", result=result)

    assert backend.capabilities() == {}
    assert backend.health().capabilities == {}
    assert str(error) == "partial failure"
    assert error.result is result


def test_create_app_submits_polls_and_reports_health(tmp_path: Path) -> None:
    backend = _ImmediateBackend()
    app = create_app(
        backend=backend,
        output_dir=tmp_path,
        title="Test Texture API",
        service_name="test-texture-api",
        max_workers=1,
    )
    client = TestClient(app)

    health = client.get("/health").json()
    assert health["status"] == "healthy"
    assert health["backend"] == "immediate-test-backend"
    assert health["accepting_jobs"] is True
    assert health["capabilities"]["multiview"] is True

    response = client.post(
        "/v1/texture-variations",
        json=_request("steel_test").model_dump(),
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    status = response.json()
    for _ in range(20):
        status = client.get(f"/v1/texture-variations/{job_id}").json()
        if status["status"] == "completed":
            break
        time.sleep(0.05)

    assert status["status"] == "completed"
    assert status["result"]["variant_name"] == "steel_test"
    assert status["result"]["generated_textures"]["normal"] is None
    assert status["result"]["metadata"]["degraded_channels"] == ["normal", "orm"]
    assert backend.requests[0].conditioning.multiview_image_uris == [
        "file:///work/view0.png"
    ]


def test_common_models_round_trip_with_texture_agent_rest_client() -> None:
    body = RestTextureVariationClient._build_request_body(
        source_asset_uri="file:///work/prepared.usd",
        conditioning=ClientConditioning(
            text_prompt="scratched blue paint",
            reference_image_uris=["file:///work/ref.png"],
            multiview_image_uris=["file:///work/view0.png"],
        ),
        config=ClientTextureVariationConfig(
            strength=0.7,
            seed=317,
            variant_name="blue_paint",
            engine="step1x",
            texture_size=1024,
            custom_parameters={"guidance": 6.0},
        ),
        target=ClientTextureTarget(
            material_name="Paint",
            material_path="/World/Looks/Paint",
            prim_paths=["/World/Mesh"],
        ),
        capabilities=ClientBackendCapabilities(
            image_conditioning=True,
            multiview=True,
            normal_map=False,
            orm=False,
            masks=True,
            coverage=True,
            geometry_output="none",
        ),
    )

    request = CreateJobRequest.model_validate(body)

    assert request.conditioning.multiview_image_uris == ["file:///work/view0.png"]
    assert request.target is not None
    assert request.target.strict_scope is True
    assert request.configuration.custom_parameters == {"guidance": 6.0}

    status = RestTextureVariationClient._parse_status(
        JobStatus(
            job_id="vj-test",
            status="completed",
            progress=100,
            result=GenerationResult(
                variant_asset_uri=request.source_asset_uri,
                variant_name="blue_paint",
                generated_textures=GeneratedTextures(
                    albedo="file:///work/albedo.png",
                    normal=None,
                    orm=None,
                ),
                maps={
                    "albedo": MapArtifact(
                        uri="file:///work/albedo.png",
                        width=1024,
                        height=1024,
                        colorspace="srgb",
                    )
                },
                metadata={"degraded_channels": ["normal", "orm"]},
                diagnostics=[
                    {
                        "code": "BACKEND_MAP_MISSING",
                        "severity": "warning",
                        "message": "normal/orm unavailable",
                    }
                ],
            ),
        ).model_dump(mode="json")
    )

    assert status.result is not None
    assert status.result.generated_textures.normal is None
    assert status.result.metadata["degraded_channels"] == ["normal", "orm"]


def test_simple_service_rest_preserves_facmogu_prompt_with_fake_nim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import apps.texture_gen_simple_service.app as simple_app

    monkeypatch.setattr(simple_app, "_BACKEND", "nim")
    model_prompts: list[str] = []

    class FakeNimModel:
        model_name = "fake-nim-image-model"
        supports_image_conditioning = False

        def generate(
            self,
            prompt: str,
            *,
            images: list[Image.Image] | None = None,
        ) -> Image.Image:
            assert images is None
            model_prompts.append(prompt)
            return Image.new("RGB", (4, 4), (32, 64, 96))

    monkeypatch.setattr(simple_app, "_model_instance", FakeNimModel())
    prompt = (
        "Extract and reproduce only the lower rectangular black-on-white Facmogu "
        "electrical rating label from the supplied reference image as a flat, "
        "front-facing albedo texture. Preserve the exact visible wording, line "
        "breaks, typography, and certification marks: Facmogu; AC/DC ADAPTER; "
        "MODEL: AL-1230; INPUT: AC100-240V 50/60Hz; OUTPUT: DC12V 3A; CE; FCC; "
        "UL LISTED; MADE IN CHINA; and the center-positive polarity symbol. Use a "
        "clean off-white label background with crisp black printing. Do not include "
        "the adapter enclosure, upper molded caution panel, cable, plug blades, "
        "lighting, shadows, perspective, checkerboard, or transparency."
    )
    request = CreateJobRequest(
        source_asset_uri="file:///tmp/facmogu.usda",
        target={
            "material_name": "Mesh_Mat",
            "material_path": (
                "/Asset/SimBodies/product/printed_electrical_rating_label/Mesh_Mat"
            ),
            "prim_paths": [
                "/Asset/SimBodies/product/printed_electrical_rating_label/Mesh"
            ],
            "mode": "per_material",
        },
        conditioning={"text_prompt": prompt},
        configuration={
            "engine": "simple_image_gen",
            "texture_size": 4,
            "variant_name": "facmogu_label",
        },
    )
    app = create_app(
        backend=simple_app.SimpleImageGenerationBackend(),
        output_dir=tmp_path / "service-output",
        title="Simple Image Gen Prompt Test",
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/texture-variations",
            json=request.model_dump(mode="json"),
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        status = response.json()
        for _ in range(100):
            status = client.get(f"/v1/texture-variations/{job_id}").json()
            if status["status"] not in {"queued", "processing"}:
                break
            time.sleep(0.01)

    expected_channel_instructions = [
        simple_app._MINIMUM_CHANNEL_INSTRUCTIONS[simple_app._ALBEDO_SUFFIX],
        simple_app._MINIMUM_CHANNEL_INSTRUCTIONS[simple_app._NORMAL_SUFFIX],
        simple_app._MINIMUM_CHANNEL_INSTRUCTIONS[simple_app._ROUGHNESS_SUFFIX],
    ]

    assert len(prompt) == 631
    assert len(f"{prompt}. {simple_app._ALBEDO_SUFFIX}") == 840
    assert status["status"] == "completed"
    assert len(model_prompts) == 3
    assert len(set(model_prompts)) == 3
    expected_channel_prompts = [
        prompt,
        f"{simple_app._NORMAL_PROMPT_PREFIX}{prompt}",
        f"{simple_app._ROUGHNESS_PROMPT_PREFIX}{prompt}",
    ]
    assert all(
        item.startswith(f"{channel_prompt}. ")
        for item, channel_prompt in zip(
            model_prompts,
            expected_channel_prompts,
            strict=True,
        )
    )
    assert all(len(item) <= simple_app._NIM_MAX_PROMPT_CHARS for item in model_prompts)
    for generated_prompt, channel_instruction in zip(
        model_prompts,
        expected_channel_instructions,
        strict=True,
    ):
        assert channel_instruction in generated_prompt


@pytest.mark.parametrize(
    ("prompt_length", "rejected_channel", "maximum_text_prompt_length"),
    [
        (744, "roughness", 743),
        (750, "roughness", 743),
        (760, "roughness", 743),
        (801, "roughness", 743),
    ],
    ids=["roughness-prefix", "normal-prefix", "near-limit", "over-limit"],
)
def test_simple_service_rest_normalizes_nim_prompt_budget_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prompt_length: int,
    rejected_channel: str,
    maximum_text_prompt_length: int,
) -> None:
    import apps.texture_gen_simple_service.app as simple_app

    monkeypatch.setattr(simple_app, "_BACKEND", "nim")
    model_launches = 0

    def fail_if_model_requested() -> object:
        nonlocal model_launches
        model_launches += 1
        raise AssertionError("model must not be launched")

    monkeypatch.setattr(simple_app, "_get_model", fail_if_model_requested)
    request = CreateJobRequest(
        source_asset_uri="file:///tmp/facmogu.usda",
        conditioning={"text_prompt": "x" * prompt_length},
        configuration={
            "engine": "simple_image_gen",
            "texture_size": 4,
            "variant_name": "prompt_too_long",
        },
    )
    app = create_app(
        backend=simple_app.SimpleImageGenerationBackend(),
        output_dir=tmp_path / "service-output",
        title="Simple Image Gen Prompt Rejection Test",
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/texture-variations",
            json=request.model_dump(mode="json"),
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        status = response.json()
        for _ in range(100):
            status = client.get(f"/v1/texture-variations/{job_id}").json()
            if status["status"] not in {"queued", "processing"}:
                break
            time.sleep(0.01)

    assert model_launches == 0
    assert status["status"] == "failed"
    assert "BACKEND_PROMPT_TOO_LONG" in status["error_message"]
    result = GenerationResult.model_validate(status["result"])
    assert result.generated_textures == GeneratedTextures()
    assert result.metadata["skipped_before_backend_launch"] is True
    diagnostic = result.diagnostics[0]
    assert diagnostic["code"] == "BACKEND_PROMPT_TOO_LONG"
    assert diagnostic["severity"] == "error"
    assert diagnostic["details"]["backend"] == "nim"
    assert diagnostic["details"]["channel"] == rejected_channel
    assert diagnostic["details"]["prompt_length"] == prompt_length
    assert diagnostic["details"]["maximum_text_prompt_length"] == (
        maximum_text_prompt_length
    )


def test_simple_service_advertised_nim_prompt_limit_composes_every_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apps.texture_gen_simple_service.app as simple_app

    monkeypatch.setattr(simple_app, "_BACKEND", "nim")
    prompt = "x" * 743
    channel_specs = [
        (prompt, simple_app._ALBEDO_SUFFIX, 0),
        (
            f"{simple_app._NORMAL_PROMPT_PREFIX}{prompt}",
            simple_app._NORMAL_SUFFIX,
            len(simple_app._NORMAL_PROMPT_PREFIX),
        ),
        (
            f"{simple_app._ROUGHNESS_PROMPT_PREFIX}{prompt}",
            simple_app._ROUGHNESS_SUFFIX,
            len(simple_app._ROUGHNESS_PROMPT_PREFIX),
        ),
    ]

    composed = [
        simple_app._image_prompt(
            channel_prompt,
            instruction,
            service_prefix_chars=service_prefix_chars,
        )
        for channel_prompt, instruction, service_prefix_chars in channel_specs
    ]

    assert all(len(item) <= simple_app._NIM_MAX_PROMPT_CHARS for item in composed)
    assert len(composed[-1]) == simple_app._NIM_MAX_PROMPT_CHARS


def test_simple_service_non_nim_prompt_keeps_full_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apps.texture_gen_simple_service.app as simple_app

    monkeypatch.setattr(simple_app, "_BACKEND", "gemini")
    prompt = "intricate glazed ceramic motif " * 30
    channel_specs = [
        (prompt, simple_app._ALBEDO_SUFFIX, 0),
        (
            f"{simple_app._NORMAL_PROMPT_PREFIX}{prompt}",
            simple_app._NORMAL_SUFFIX,
            len(simple_app._NORMAL_PROMPT_PREFIX),
        ),
        (
            f"{simple_app._ROUGHNESS_PROMPT_PREFIX}{prompt}",
            simple_app._ROUGHNESS_SUFFIX,
            len(simple_app._ROUGHNESS_PROMPT_PREFIX),
        ),
    ]

    for channel_prompt, instruction, service_prefix_chars in channel_specs:
        expected = f"{channel_prompt.strip()}. {instruction}"

        assert len(expected) > simple_app._NIM_MAX_PROMPT_CHARS
        assert (
            simple_app._image_prompt(
                channel_prompt,
                instruction,
                service_prefix_chars=service_prefix_chars,
            )
            == expected
        )


def test_simple_service_rejects_reference_with_capability_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import apps.texture_gen_simple_service.app as simple_app

    model_requested = False

    def fail_if_model_requested() -> object:
        nonlocal model_requested
        model_requested = True
        raise AssertionError("model must not be launched")

    monkeypatch.setattr(simple_app, "_get_model", fail_if_model_requested)
    request = CreateJobRequest(
        source_asset_uri="file:///tmp/facmogu.usda",
        target={
            "material_name": "Mesh_Mat",
            "material_path": "/World/Looks/Mesh_Mat",
            "prim_paths": ["/World/Label"],
            "mode": "per_material",
        },
        conditioning={
            "text_prompt": "reproduce the rating label",
            "reference_image_uris": ["file:///tmp/facmogu-reference.png"],
        },
    )

    with pytest.raises(
        TextureGenerationBackendError,
        match="BACKEND_CONDITIONING_UNSUPPORTED",
    ) as exc_info:
        simple_app.SimpleImageGenerationBackend().generate(
            request,
            job_id="vj-reference-rejected",
            output_dir=tmp_path,
            cancel_event=threading.Event(),
        )

    assert model_requested is False
    result = exc_info.value.result
    assert result is not None
    assert result.metadata["skipped_before_backend_launch"] is True
    diagnostic = result.diagnostics[0]
    assert diagnostic["code"] == "BACKEND_CONDITIONING_UNSUPPORTED"
    assert diagnostic["severity"] == "error"
    assert diagnostic["details"]["unsupported_fields"] == ["reference_image_uris"]


def test_service_reports_failed_job_and_unknown_job(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            backend=_FailingBackend(),
            output_dir=tmp_path,
            title="Failing Texture API",
        )
    )

    response = client.post(
        "/v1/texture-variations",
        json=_request("failure").model_dump(),
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    status = response.json()
    for _ in range(20):
        status = client.get(f"/v1/texture-variations/{job_id}").json()
        if status["status"] == "failed":
            break
        time.sleep(0.05)

    assert status["status"] == "failed"
    assert "intentional backend failure" in status["error_message"]
    assert client.get("/v1/texture-variations/missing").status_code == 404


def test_service_marks_job_failed_when_output_dir_cannot_be_created(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "not-a-directory"
    output_root.write_text("blocking file", encoding="utf-8")
    backend = _ImmediateBackend()
    service = TextureVariationService(
        backend=backend,
        output_dir=output_root,
    )

    try:
        submitted = service.submit(_request("bad_output_dir"))
        status = submitted
        for _ in range(20):
            status = service.get_status(submitted.job_id)
            if status.status == "failed":
                break
            time.sleep(0.05)

        assert status.status == "failed"
        assert status.error_message
        assert backend.requests == []
        assert service.health().active_jobs == 0
    finally:
        service.shutdown()


def test_create_app_rejects_job_when_backend_not_ready(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            backend=_NotReadyBackend(),
            output_dir=tmp_path,
            title="Not Ready Texture API",
        )
    )

    health = client.get("/health").json()
    assert health["ready"] is False
    assert health["accepting_jobs"] is False

    response = client.post(
        "/v1/texture-variations",
        json=_request("not_ready").model_dump(),
    )

    assert response.status_code == 503
    assert "runtime missing" in response.json()["detail"]


def test_health_reports_busy_when_worker_capacity_is_full(tmp_path: Path) -> None:
    backend = _BlockingBackend()
    service = TextureVariationService(
        backend=backend,
        output_dir=tmp_path,
        max_workers=1,
    )

    try:
        service.submit(_request("busy"))
        assert backend.started.wait(timeout=2)

        health = service.health()

        assert health.status == "busy"
        assert health.active_jobs == 1
        assert health.accepting_jobs is False
    finally:
        backend.release.set()
        service.shutdown()


def test_artifact_helpers_require_visible_local_files(tmp_path: Path) -> None:
    artifact = tmp_path / "albedo.png"
    artifact.write_bytes(b"fake-png")

    uri = local_file_uri(artifact)

    assert require_visible_file(uri, label="albedo") == artifact.resolve()
    with pytest.raises(ValueError, match="not visible"):
        require_visible_file((tmp_path / "missing.png").as_uri(), label="albedo")
    with pytest.raises(ValueError, match="not a local file URI"):
        require_visible_file("s3://bucket/albedo.png", label="albedo")


def test_file_uri_helper_treats_localhost_authority_as_local_path() -> None:
    assert local_path_from_file_uri("file://localhost/tmp/albedo.png") == Path(
        "/tmp/albedo.png"
    )


def test_file_uri_helper_preserves_bare_paths_and_decodes_file_uris() -> None:
    assert local_path_from_file_uri("/tmp/ref%20map.png") == Path("/tmp/ref%20map.png")
    assert local_path_from_file_uri("file:///tmp/ref%20map.png") == Path(
        "/tmp/ref map.png"
    )
    assert local_path_from_file_uri("C:/textures/albedo.png") == Path(
        "C:/textures/albedo.png"
    )
    assert local_path_from_file_uri("file:///C:/textures/albedo.png") == Path(
        "C:/textures/albedo.png"
    )
    assert local_path_from_file_uri("file://C:/textures/albedo.png") == Path(
        "C:/textures/albedo.png"
    )
    assert local_path_from_file_uri("file://server/share/albedo.png") == Path(
        "/server/share/albedo.png"
    )
    assert local_path_from_file_uri("file://C:\\textures\\albedo.png") == Path(
        "C:\\textures\\albedo.png"
    )


def test_file_uri_helper_prefers_existing_literal_bare_path(
    tmp_path: Path,
) -> None:
    literal = tmp_path / "ref%20map.png"
    decoded = tmp_path / "ref map.png"
    literal.write_text("literal", encoding="utf-8")
    decoded.write_text("decoded", encoding="utf-8")

    assert local_path_from_file_uri(str(literal)) == literal


def test_file_uri_helper_falls_back_to_existing_decoded_bare_path(
    tmp_path: Path,
) -> None:
    encoded = tmp_path / "ref%20map.png"
    decoded = tmp_path / "ref map.png"
    decoded.write_text("decoded", encoding="utf-8")

    assert local_path_from_file_uri(str(encoded)) == decoded


def test_service_common_usd_package_copies_with_limits() -> None:
    src = io.BytesIO(b"abcdef")
    dst = io.BytesIO()

    assert (
        service_usd_package.copy_stream_limited(
            src,
            dst,
            max_bytes=6,
            chunk_size=2,
        )
        == 6
    )
    assert dst.getvalue() == b"abcdef"

    with pytest.raises(ValueError, match="max_bytes"):
        service_usd_package.copy_stream_limited(
            io.BytesIO(b"a"),
            io.BytesIO(),
            max_bytes=-1,
        )
    with pytest.raises(ValueError, match="chunk_size"):
        service_usd_package.copy_stream_limited(
            io.BytesIO(b"a"),
            io.BytesIO(),
            max_bytes=1,
            chunk_size=0,
        )
    with pytest.raises(service_usd_package.ArchiveSizeLimitExceeded) as exc_info:
        service_usd_package.copy_stream_limited(
            io.BytesIO(b"abc"),
            io.BytesIO(),
            max_bytes=2,
            chunk_size=3,
        )
    assert exc_info.value.max_bytes == 2
    assert exc_info.value.attempted_bytes == 3


def test_service_common_usd_package_resolves_and_parses_members(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset[variant].usdz"
    package.write_bytes(b"placeholder")

    assert (
        service_usd_package.resolve_local_package_path(package.as_uri())
        == package.resolve()
    )
    assert (
        service_usd_package.resolve_local_package_path("relative.usdz", tmp_path)
        == (tmp_path / "relative.usdz").resolve()
    )
    assert (
        service_usd_package.resolve_local_package_path(
            "file://server/share/asset.usdz"
        ).as_posix()
        == "//server/share/asset.usdz"
    )
    assert service_usd_package.split_package_member_asset_path(
        "asset.usdz[textures/base.png]"
    ) == ("asset.usdz", "textures/base.png")
    assert service_usd_package.split_package_member_asset_path("asset.usdz") is None
    assert (
        service_usd_package.split_package_member_asset_path("asset.usd[base.png]")
        is None
    )
    assert service_usd_package.split_package_member_asset_path("asset.usdz[]") is None
    assert service_usd_package.parse_package_member_asset_path(
        f"{package}[textures/[base].png]",
        base_dir=tmp_path,
    ) == (package.resolve(), "textures/[base].png")
    assert (
        service_usd_package.parse_package_member_asset_path(
            str(package),
            base_dir=tmp_path,
        )
        is None
    )
    assert (
        service_usd_package.parse_package_member_asset_path(
            f"{package}[../escape.png]",
            base_dir=tmp_path,
        )
        is None
    )


def test_service_common_usd_package_normalizes_safe_member_names(
    tmp_path: Path,
) -> None:
    package = tmp_path / "my asset!.usdz"
    package.write_bytes(b"placeholder")
    unsafe_stem = tmp_path / "!!!.usdz"

    assert service_usd_package.safe_usdz_member_parts(
        "/textures/base%20color.png",
        allow_leading_slash=True,
    ) == ("textures", "base color.png")
    assert service_usd_package.safe_usdz_member_parts("textures\\normal.png") == (
        "textures",
        "normal.png",
    )
    assert service_usd_package.safe_usdz_member_parts("/textures/albedo.png") is None
    assert service_usd_package.safe_usdz_member_parts("../escape.png") is None
    assert service_usd_package.safe_usdz_member_parts("./") is None
    assert (
        service_usd_package.safe_usdz_member_name(
            "/textures/base.png",
            allow_leading_slash=True,
        )
        == "textures/base.png"
    )
    assert service_usd_package.safe_usdz_member_name("../base.png") is None
    assert service_usd_package.package_member_cache_name(package) == "my_asset"
    assert service_usd_package.package_member_cache_name(unsafe_stem) == "package"
    assert service_usd_package.package_member_cache_name(
        package,
        digest_len=8,
    ).startswith("my_asset-")


def test_service_common_usd_package_extracts_and_filters_members(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("textures/", b"")
        archive.writestr("textures/albedo.png", b"png")
        symlink = zipfile.ZipInfo("textures/link.png")
        symlink.external_attr = 0xA000 << 16
        archive.writestr(symlink, b"target")

    out = tmp_path / "out" / "albedo.png"
    assert (
        service_usd_package.extract_usdz_member_to_path(
            package,
            "textures/albedo.png",
            out,
            allowed_suffixes={".png"},
        )
        == 3
    )
    assert out.read_bytes() == b"png"
    assert (
        service_usd_package.extract_usdz_member_to_path(
            tmp_path / "missing.usdz",
            "textures/albedo.png",
            tmp_path / "missing.png",
        )
        is None
    )
    assert (
        service_usd_package.extract_usdz_member_to_path(
            package.with_suffix(".zip"),
            "textures/albedo.png",
            tmp_path / "wrong-ext.png",
        )
        is None
    )
    assert (
        service_usd_package.extract_usdz_member_to_path(
            package,
            "../escape.png",
            tmp_path / "escape.png",
        )
        is None
    )
    assert (
        service_usd_package.extract_usdz_member_to_path(
            package,
            "textures/albedo.png",
            tmp_path / "disallowed.png",
            allowed_suffixes={".jpg"},
        )
        is None
    )
    assert (
        service_usd_package.extract_usdz_member_to_path(
            package,
            "textures/missing.png",
            tmp_path / "absent.png",
        )
        is None
    )
    assert (
        service_usd_package.extract_usdz_member_to_path(
            package,
            "textures/",
            tmp_path / "dir.png",
        )
        is None
    )
    assert (
        service_usd_package.extract_usdz_member_to_path(
            package,
            "textures/link.png",
            tmp_path / "link.png",
        )
        is None
    )

    bad_zip = tmp_path / "bad.usdz"
    bad_zip.write_bytes(b"not a zip")
    assert (
        service_usd_package.extract_usdz_member_to_path(
            bad_zip,
            "textures/albedo.png",
            tmp_path / "bad.png",
        )
        is None
    )

    limited_dest = tmp_path / "limited.png"
    with pytest.raises(service_usd_package.ArchiveSizeLimitExceeded):
        service_usd_package.extract_usdz_member_to_path(
            package,
            "textures/albedo.png",
            limited_dest,
            max_bytes=2,
        )
    assert not limited_dest.exists()


def test_service_common_usd_package_extracts_members_to_cache_dir(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("textures/albedo.png", b"png")

    extract_root = tmp_path / "extract"
    cached = extract_root / "textures" / "albedo.png"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached")

    assert (
        service_usd_package.extract_usdz_member_to_dir(
            package,
            "textures/albedo.png",
            extract_root,
            allowed_suffixes={".jpg"},
        )
        is None
    )
    assert (
        service_usd_package.extract_usdz_member_to_dir(
            package,
            "../escape.png",
            extract_root,
        )
        is None
    )
    assert (
        service_usd_package.extract_usdz_member_to_dir(
            package,
            "textures/albedo.png",
            extract_root,
            allowed_suffixes={".png"},
        )
        == cached
    )

    cached.unlink()
    extracted = service_usd_package.extract_usdz_member_to_dir(
        package,
        "textures/albedo.png",
        extract_root,
    )

    assert extracted == cached
    assert cached.read_bytes() == b"png"
    assert (
        service_usd_package.extract_usdz_member_to_dir(
            package,
            "textures/missing.png",
            extract_root,
        )
        is None
    )


def test_service_can_cancel_queued_job(tmp_path: Path) -> None:
    backend = _BlockingBackend()
    service = TextureVariationService(
        backend=backend,
        output_dir=tmp_path,
        max_workers=1,
        max_queue_size=1,
    )

    try:
        first = service.submit(_request("first"))
        assert backend.started.wait(timeout=2)
        second = service.submit(_request("second"))

        service.cancel(second.job_id)

        assert service.get_status(second.job_id).status == "cancelled"
        backend.release.set()
        for _ in range(20):
            if service.get_status(first.job_id).status == "completed":
                break
            time.sleep(0.05)
        assert service.get_status(first.job_id).status == "completed"
    finally:
        backend.release.set()
        service.shutdown()


def test_service_marks_pre_cancelled_job_without_backend_call(tmp_path: Path) -> None:
    backend = _ImmediateBackend()
    service = TextureVariationService(
        backend=backend,
        output_dir=tmp_path,
    )
    status = JobStatus(
        job_id="pre-cancelled",
        status="queued",
        progress=0,
    )
    record = _JobRecord(status)
    record.cancel_event.set()
    service._jobs[status.job_id] = record

    try:
        service._run_job(status.job_id, _request("pre-cancelled"), record)

        updated = service.get_status(status.job_id)
        assert updated.status == "cancelled"
        assert updated.progress == 100
        assert updated.message == "Cancelled before execution."
        assert backend.requests == []
    finally:
        service.shutdown()


def test_service_preserves_partial_result_from_backend_error(tmp_path: Path) -> None:
    service = TextureVariationService(
        backend=_PartialResultFailingBackend(),
        output_dir=tmp_path,
    )

    try:
        submitted = service.submit(_request("partial-failure"))
        status = submitted
        for _ in range(20):
            status = service.get_status(submitted.job_id)
            if status.status == "failed":
                break
            time.sleep(0.05)

        assert status.status == "failed"
        assert status.error_message == "intentional partial backend failure"
        assert status.result is not None
        assert status.result.variant_name == "partial-result"
        assert status.result.generated_textures.albedo is not None
    finally:
        service.shutdown()


def test_service_shutdown_marks_queued_jobs_cancelled(tmp_path: Path) -> None:
    backend = _BlockingBackend()
    service = TextureVariationService(
        backend=backend,
        output_dir=tmp_path,
        max_workers=1,
        max_queue_size=1,
    )

    try:
        service.submit(_request("first"))
        assert backend.started.wait(timeout=2)
        queued = service.submit(_request("queued"))

        service.shutdown()

        status = service.get_status(queued.job_id)
        assert status.status == "cancelled"
        assert status.message == "Service is shutting down."
    finally:
        backend.release.set()
        service.shutdown()


def test_create_app_reports_busy_and_cancel_route_errors(tmp_path: Path) -> None:
    backend = _BlockingBackend()
    with TestClient(
        create_app(
            backend=backend,
            output_dir=tmp_path,
            title="Busy Texture API",
            max_workers=1,
            max_queue_size=0,
        )
    ) as client:
        try:
            first = client.post(
                "/v1/texture-variations",
                json=_request("busy-route").model_dump(),
            )
            assert first.status_code == 202
            assert backend.started.wait(timeout=2)

            busy = client.post(
                "/v1/texture-variations",
                json=_request("too-busy").model_dump(),
            )
            assert busy.status_code == 429
            assert "queue is full" in busy.json()["detail"]
            assert client.delete("/v1/texture-variations/missing").status_code == 404

            job_id = first.json()["job_id"]
            backend.release.set()
            for _ in range(20):
                status = client.get(f"/v1/texture-variations/{job_id}").json()
                if status["status"] == "completed":
                    break
                time.sleep(0.05)

            response = client.delete(f"/v1/texture-variations/{job_id}")

            assert response.status_code == 409
            assert "terminal state" in response.json()["detail"]
        finally:
            backend.release.set()


def test_service_marks_running_exception_after_cancel_as_cancelled(
    tmp_path: Path,
) -> None:
    service = TextureVariationService(
        backend=_CancelFailingBackend(),
        output_dir=tmp_path,
    )

    try:
        submitted = service.submit(_request("cancel-failure"))
        status = submitted
        for _ in range(20):
            status = service.get_status(submitted.job_id)
            if status.status == "cancelled":
                break
            time.sleep(0.05)

        assert status.status == "cancelled"
        assert status.message == "Cancellation requested while backend was running."
        assert status.error_message is None
    finally:
        service.shutdown()


def test_service_preserves_cancelled_result_when_cancelled_after_backend_result(
    tmp_path: Path,
) -> None:
    service = TextureVariationService(
        backend=_CancelAfterResultBackend(),
        output_dir=tmp_path,
    )

    try:
        submitted = service.submit(_request("cancel-result"))
        status = submitted
        for _ in range(20):
            status = service.get_status(submitted.job_id)
            if status.status == "cancelled":
                break
            time.sleep(0.05)

        assert status.status == "cancelled"
        assert status.result is not None
        assert status.result.variant_name == "cancel-result"
    finally:
        service.shutdown()


def test_service_evicts_terminal_jobs_after_ttl(tmp_path: Path) -> None:
    service = TextureVariationService(
        backend=_ImmediateBackend(),
        output_dir=tmp_path,
        terminal_job_ttl_sec=60,
    )

    try:
        submitted = service.submit(_request("ttl"))
        for _ in range(20):
            status = service.get_status(submitted.job_id)
            if status.status == "completed":
                break
            time.sleep(0.01)

        assert status.status == "completed"
        job_dir = tmp_path / submitted.job_id
        assert job_dir.is_dir()
        with service._lock:
            record = service._jobs[submitted.job_id]
            assert record.completed_at is not None
            record.completed_at = time.monotonic() - 61
        with pytest.raises(KeyError):
            service.get_status(submitted.job_id)
        assert not job_dir.exists()
        assert submitted.job_id not in service._pending_output_cleanup
    finally:
        service.shutdown()


def test_service_evicts_terminal_jobs_without_locking_output_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = TextureVariationService(
        backend=_ImmediateBackend(),
        output_dir=tmp_path,
        terminal_job_ttl_sec=60,
    )
    original_rmtree = service_module.shutil.rmtree
    cleanup_saw_unlocked = False

    def assert_unlocked_rmtree(path: Path) -> None:
        nonlocal cleanup_saw_unlocked
        cleanup_saw_unlocked = service._lock.acquire(blocking=False)
        if cleanup_saw_unlocked:
            service._lock.release()
        original_rmtree(path)

    monkeypatch.setattr(service_module.shutil, "rmtree", assert_unlocked_rmtree)

    try:
        submitted = service.submit(_request("ttl-unlocked"))
        for _ in range(20):
            status = service.get_status(submitted.job_id)
            if status.status == "completed":
                break
            time.sleep(0.01)

        assert status.status == "completed"
        with service._lock:
            service._jobs[submitted.job_id].completed_at = time.monotonic() - 61

        with pytest.raises(KeyError):
            service.get_status(submitted.job_id)

        assert cleanup_saw_unlocked is True
        assert not (tmp_path / submitted.job_id).exists()
    finally:
        service.shutdown()


def test_service_retries_failed_output_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    service = TextureVariationService(
        backend=_ImmediateBackend(),
        output_dir=tmp_path,
        terminal_job_ttl_sec=60,
    )
    original_rmtree = service_module.shutil.rmtree
    attempts = 0

    def flaky_rmtree(path: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(f"cannot remove {path}")
        original_rmtree(path)

    monkeypatch.setattr(service_module.shutil, "rmtree", flaky_rmtree)

    try:
        submitted = service.submit(_request("ttl-retry"))
        for _ in range(20):
            status = service.get_status(submitted.job_id)
            if status.status == "completed":
                break
            time.sleep(0.01)

        assert status.status == "completed"
        job_dir = tmp_path / submitted.job_id
        assert job_dir.is_dir()
        with service._lock:
            service._jobs[submitted.job_id].completed_at = time.monotonic() - 61

        with caplog.at_level("WARNING", logger=service_module.logger.name):
            with pytest.raises(KeyError):
                service.get_status(submitted.job_id)
        assert "Failed to clean texture generation job output" in caplog.text
        assert job_dir.is_dir()
        assert submitted.job_id in service._pending_output_cleanup

        service.health()

        assert attempts == 2
        assert not job_dir.exists()
        assert submitted.job_id not in service._pending_output_cleanup
    finally:
        service.shutdown()


def test_service_keeps_terminal_jobs_when_ttl_disabled(tmp_path: Path) -> None:
    service = TextureVariationService(
        backend=_ImmediateBackend(),
        output_dir=tmp_path,
        terminal_job_ttl_sec=0,
    )

    try:
        submitted = service.submit(_request("ttl-disabled"))
        for _ in range(20):
            status = service.get_status(submitted.job_id)
            if status.status == "completed":
                break
            time.sleep(0.01)
        assert status.status == "completed"

        with service._lock:
            service._jobs[submitted.job_id].completed_at = time.monotonic() - 999

        assert service.get_status(submitted.job_id).status == "completed"
        assert (tmp_path / submitted.job_id).is_dir()
    finally:
        service.shutdown()


def test_service_cleanup_job_output_ignores_missing_directory(tmp_path: Path) -> None:
    service = TextureVariationService(
        backend=_ImmediateBackend(),
        output_dir=tmp_path,
    )

    try:
        assert service._cleanup_job_output("missing-job") is True
    finally:
        service.shutdown()


def test_service_cleanup_job_output_warns_when_delete_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    service = TextureVariationService(
        backend=_ImmediateBackend(),
        output_dir=tmp_path,
    )
    job_dir = tmp_path / "stuck-job"
    job_dir.mkdir()

    def fail_rmtree(path: Path) -> None:
        raise OSError(f"cannot remove {path}")

    monkeypatch.setattr(service_module.shutil, "rmtree", fail_rmtree)

    try:
        with caplog.at_level("WARNING", logger=service_module.logger.name):
            assert service._cleanup_job_output("stuck-job") is False
        assert "Failed to clean texture generation job output" in caplog.text
    finally:
        service.shutdown()


def test_simple_service_client_accepts_empty_cancel_response(monkeypatch) -> None:
    class _CancelResponse:
        status = 204

        def __enter__(self) -> _CancelResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b""

    captured: dict[str, object] = {}

    def fake_urlopen(request, *, timeout: float) -> _CancelResponse:
        captured["method"] = request.get_method()
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _CancelResponse()

    monkeypatch.setattr(
        "apps.texture_gen_simple_service.client.client.urlopen",
        fake_urlopen,
    )

    client = TextureVariationClient("http://texture-service", timeout=3.5)

    assert client.cancel_texture_variation("vj-cancel") is None
    assert captured == {
        "method": "DELETE",
        "url": "http://texture-service/v1/texture-variations/vj-cancel",
        "timeout": 3.5,
    }
