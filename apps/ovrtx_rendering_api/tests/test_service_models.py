# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic validation tests for the OVRtx rendering-api request models.

Mirrors ``kit-gen-ai-service``'s ``tests/test_rendering_models.py`` coverage:
per-field defaults, per-field validation (reject + accept), render-mode
literal handling, and wire-format JSON round-trips so the OVRtx service
stays request-schema-compatible with Kit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

# service/ is on sys.path via [tool.pytest.ini_options].pythonpath.
from service.models import (
    CameraParameters,
    FrameRange,
    ProtocolV3CameraDef,
    ProtocolV3RenderUploadParams,
    RenderRequest,
    RenderResponse,
    RenderSettings,
)
from service.protocol import PROTOCOL_VERSION
from usd_core.remote_protocol import PROTOCOL_VERSION as USD_CLI_PROTOCOL_VERSION


def test_protocol_version_is_owned_by_package_usd_cli() -> None:
    assert PROTOCOL_VERSION == USD_CLI_PROTOCOL_VERSION == 3


def test_protocol_v3_render_probe_payload_is_accepted() -> None:
    params = ProtocolV3RenderUploadParams(
        cameras=["/World/Camera"],
        image_width=64,
        image_height=64,
        mode="quality",
        compression="gzip",
        frames=[0.0],
        camera_defs=[
            ProtocolV3CameraDef(
                path="/World/Camera",
                matrix=[1.0, 0.0, 0.0, 0.0] * 4,
                clipping_range=[0.1, 1000.0],
            )
        ],
    )

    assert params.cameras == ["/World/Camera"]
    assert params.compression == "gzip"


@pytest.mark.parametrize(
    "payload",
    [
        {"compression": "zstd"},
        {"frames": []},
        {"frames": [float("nan")]},
        {"cameras": [f"/World/Camera_{index}" for index in range(9)]},
        {"image_width": 4096, "image_height": 4096, "frames": list(range(17))},
        {"unexpected": True},
    ],
)
def test_protocol_v3_render_probe_payload_fails_closed(payload: dict) -> None:
    base = {"cameras": ["/World/Camera"], "image_width": 64, "image_height": 64}

    with pytest.raises(ValidationError):
        ProtocolV3RenderUploadParams(**{**base, **payload})


def test_protocol_v3_render_probe_preserves_fractional_and_duplicate_frames() -> None:
    params = ProtocolV3RenderUploadParams(
        cameras=["/World/Camera"], frames=[5.0, 1.5, 1.5]
    )

    assert params.frames == [5.0, 1.5, 1.5]


@pytest.mark.parametrize(
    "field,value",
    [
        ("matrix", [1.0] * 15 + [float("inf")]),
        ("clipping_range", [10.0, 1.0]),
    ],
)
def test_protocol_v3_camera_definition_fails_closed(field: str, value: list) -> None:
    payload = {
        "path": "/World/Camera",
        "matrix": [1.0, 0.0, 0.0, 0.0] * 4,
        field: value,
    }

    with pytest.raises(ValidationError):
        ProtocolV3CameraDef(**payload)


def test_openapi_num_sensor_updates_matches_model_bounds() -> None:
    schema = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "openapi.yaml").read_text(
            encoding="utf-8"
        )
    )
    property_schema = schema["components"]["schemas"]["RenderSettings"]["properties"][
        "num_sensor_updates"
    ]

    assert property_schema == {
        "type": "integer",
        "minimum": 1,
        "maximum": 5000,
        "nullable": True,
    }


def test_standalone_image_installs_exact_checkout_usd_cli_protocol() -> None:
    app_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((app_root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    dependencies = project["dependencies"]
    dockerfile = (app_root / "Dockerfile").read_text(encoding="utf-8")

    assert project["requires-python"] == ">=3.11,<3.13"
    assert "usd-cli==0.0.1" in dependencies
    assert "python-multipart>=0.0.9" in dependencies
    assert "COPY apps/usd_cli /app/apps/usd_cli" in dockerfile
    assert "cd /app/apps/usd_cli && uv pip install -e . --no-config --no-sources" in (
        dockerfile
    )


def test_openapi_retryable_incomplete_output_contract_is_required() -> None:
    schema = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "openapi.yaml").read_text(
            encoding="utf-8"
        )
    )
    response_schema = schema["components"]["schemas"]["IncompleteRenderResponse"]
    retry_after = schema["paths"]["/render"]["post"]["responses"]["503"]["headers"][
        "Retry-After"
    ]

    assert schema["paths"]["/render"]["post"]["responses"]["503"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/IncompleteRenderResponse"}
    assert set(response_schema["required"]) == {
        "status",
        "error",
        "images",
        "error_code",
        "retryable",
        "requested_output_count",
        "output_count",
        "missing_output_count",
        "missing_camera_count",
    }
    assert response_schema["properties"]["status"]["enum"] == ["exception"]
    assert response_schema["properties"]["error_code"]["enum"] == [
        "incomplete_render_output"
    ]
    assert response_schema["properties"]["retryable"]["enum"] == [True]
    assert retry_after["required"] is True


def test_openapi_protocol_v3_contract_matches_package_owned_floor() -> None:
    schema = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "openapi.yaml").read_text(
            encoding="utf-8"
        )
    )
    live = schema["components"]["schemas"]["LiveResponse"]
    upload = schema["components"]["schemas"]["ProtocolV3RenderUploadParams"]
    item = schema["components"]["schemas"]["ProtocolV3RenderResultItem"]
    health = schema["components"]["schemas"]["HealthResponse"]

    assert "503" not in schema["paths"]["/live"]["get"]["responses"]
    assert schema["paths"]["/render/upload"]["post"]["responses"]["503"]
    assert live["properties"]["protocol_version"]["enum"] == [3]
    assert live["properties"]["features"]["maxItems"] == 0
    assert live["properties"]["features"]["items"] == {"type": "string"}
    assert "protocol_version" in health["required"]
    assert health["properties"]["protocol_version"]["enum"] == [3]
    assert "nullable" not in health["properties"]["protocol_version"]
    assert upload["properties"]["compression"]["enum"] == ["none", "gzip"]
    assert upload["properties"]["frames"]["items"] == {"type": "number"}
    assert "uniqueItems" not in upload["properties"]["frames"]
    assert set(item["required"]) == {
        "camera",
        "image_base64",
        "ovrtx_render_mode",
        "ovrtx_num_sensor_updates",
        "active_aov",
    }


class TestCameraParameters:
    def test_defaults(self):
        cp = CameraParameters()
        assert cp.width == 1024
        assert cp.height == 1024

    def test_custom_values(self):
        cp = CameraParameters(width=512, height=256)
        assert cp.width == 512
        assert cp.height == 256

    @pytest.mark.parametrize("w,h", [(1, 1), (512, 512), (1024, 1024), (8192, 8192)])
    def test_accepts_reasonable_resolutions(self, w, h):
        cp = CameraParameters(width=w, height=h)
        assert cp.width == w
        assert cp.height == h

    @pytest.mark.parametrize("w,h", [(0, 1), (1, 0), (8193, 1), (1, 8193)])
    def test_rejects_unbounded_resolutions(self, w, h):
        with pytest.raises(ValidationError):
            CameraParameters(width=w, height=h)


class TestFrameRange:
    def test_defaults(self):
        fr = FrameRange()
        assert fr.start == 0
        assert fr.end == 0

    def test_custom_values(self):
        fr = FrameRange(start=5, end=10)
        assert fr.start == 5
        assert fr.end == 10

    def test_single_frame(self):
        fr = FrameRange(start=0, end=0)
        assert fr.start == fr.end


class TestRenderMode:
    """``RenderSettings.render_mode`` is ``Literal["rt1","rt2","pt"] | None``."""

    @pytest.mark.parametrize("mode", ["rt1", "rt2", "pt"])
    def test_accepts_each_supported_mode(self, mode):
        rs = RenderSettings(render_mode=mode)
        assert rs.render_mode == mode

    def test_default_is_none_so_env_default_wins(self):
        """Unset ``render_mode`` lets ``OVRTX_RENDER_MODE`` at the service
        boot time govern — same escape-hatch pattern as ``num_sensor_updates``."""
        assert RenderSettings().render_mode is None

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValidationError):
            RenderSettings(render_mode="raytracing")  # not in {rt1,rt2,pt}

    def test_rejects_empty_string(self):
        with pytest.raises(ValidationError):
            RenderSettings(render_mode="")

    def test_rejects_kit_long_tokens(self):
        # Kit's schema uses the long ovrtx token names; our wire contract
        # uses the short kit-gen-ai-service enum values. Accept short only.
        with pytest.raises(ValidationError):
            RenderSettings(render_mode="PathTracing")


class TestMaterialTarget:
    @pytest.mark.parametrize(
        "target",
        [
            "auto",
            "display_color",
            "preview_surface",
            "openpbr_materialx",
            "omnipbr_mdl",
        ],
    )
    def test_accepts_each_supported_target(self, target):
        rs = RenderSettings(material_target=target)
        assert rs.material_target == target

    def test_default_is_none_so_backend_default_wins(self):
        assert RenderSettings().material_target is None

    def test_rejects_unknown_target(self):
        with pytest.raises(ValidationError):
            RenderSettings(material_target="silent_preview_fallback")


class TestRenderSettingsDefaults:
    def test_defaults(self):
        rs = RenderSettings()
        assert rs.camera_paths == ["/Camera"]
        assert rs.frame_range.start == 0
        assert rs.frame_range.end == 0
        assert rs.camera_parameters.width == 1024
        assert rs.camera_parameters.height == 1024
        assert rs.sensors is None
        assert rs.apply_background_mask is False
        assert rs.num_sensor_updates is None
        assert rs.render_mode is None
        assert rs.material_target is None

    def test_num_sensor_updates_rejects_zero(self):
        with pytest.raises(ValidationError):
            RenderSettings(num_sensor_updates=0)

    def test_num_sensor_updates_rejects_negative(self):
        with pytest.raises(ValidationError):
            RenderSettings(num_sensor_updates=-1)

    def test_num_sensor_updates_accepts_positive(self):
        assert RenderSettings(num_sensor_updates=1).num_sensor_updates == 1
        assert RenderSettings(num_sensor_updates=500).num_sensor_updates == 500
        assert RenderSettings(num_sensor_updates=5000).num_sensor_updates == 5000

    def test_num_sensor_updates_rejects_unbounded_work(self):
        with pytest.raises(ValidationError):
            RenderSettings(num_sensor_updates=5001)


class TestRenderRequest:
    def test_defaults(self):
        rr = RenderRequest(url="data:model/vnd.usda;base64,AA==")
        assert rr.url == "data:model/vnd.usda;base64,AA=="
        assert rr.force_render is True
        assert isinstance(rr.render_settings, RenderSettings)

    def test_url_is_required(self):
        with pytest.raises(ValidationError):
            RenderRequest()  # type: ignore[call-arg]

    def test_full_custom_request(self):
        rr = RenderRequest(
            url="https://example.com/scene.usd",
            force_render=False,
            render_settings=RenderSettings(
                camera_paths=["/World/Cam1", "/World/Cam2"],
                frame_range=FrameRange(start=0, end=10),
                camera_parameters=CameraParameters(width=512, height=512),
                sensors=["depth"],
                num_sensor_updates=25,
                render_mode="pt",
            ),
        )
        assert rr.url == "https://example.com/scene.usd"
        assert rr.force_render is False
        assert rr.render_settings.camera_paths == ["/World/Cam1", "/World/Cam2"]
        assert rr.render_settings.frame_range.end == 10
        assert rr.render_settings.num_sensor_updates == 25
        assert rr.render_settings.render_mode == "pt"


class TestJsonRoundTrip:
    """Wire-format compatibility: the request body we accept must survive
    a json.dumps/loads round-trip without data loss. Kit-gen-ai-service
    clients serialise RenderRequest this way before POSTing, so we need
    to round-trip cleanly across the same fields.
    """

    def test_request_roundtrip_minimal(self):
        original = RenderRequest(url="data:model/vnd.usda;base64,AA==")
        data = json.loads(original.model_dump_json())
        restored = RenderRequest(**data)
        assert restored == original

    def test_request_roundtrip_full(self):
        original = RenderRequest(
            url="https://example.com/scene.usd",
            render_settings=RenderSettings(
                camera_paths=["/World/Cam"],
                frame_range=FrameRange(start=0, end=5),
                camera_parameters=CameraParameters(width=256, height=256),
                sensors=["depth"],
                num_sensor_updates=100,
                render_mode="rt2",
            ),
        )
        data = json.loads(original.model_dump_json())
        restored = RenderRequest(**data)
        assert restored == original

    def test_response_roundtrip(self):
        resp = RenderResponse(
            status="success",
            error=None,
            images={"0": {"/World/Cam": {"images": "base64..."}}},
        )
        data = json.loads(resp.model_dump_json())
        restored = RenderResponse(**data)
        assert restored == resp
