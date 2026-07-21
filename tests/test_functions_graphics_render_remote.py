# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for remote render response parsing, including V2-to-V1 conversion."""

import base64
import io
import json
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

import world_understanding.functions.graphics.render_remote as render_remote
from world_understanding.functions.graphics import render_remote_async
from world_understanding.functions.graphics.render_remote import (
    RenderingStatus,
    _add_texture_file_fallbacks_for_remote_export,
    _bundle_stage_with_local_assets,
    _convert_v2_sensor,
    _convert_v2_to_v1,
    _export_stage_and_get_url,
    _http_error_detail,
    _http_error_payload,
    _is_local_composition_asset_path,
    _is_v2_response,
    _prefer_preview_surface_for_remote_export,
    _resolve_export_asset_path,
    _stage_has_local_composition_arcs,
    export_stage_to_s3,
    render_all_cameras,
    render_single_camera_from_url,
    save_render_results,
)
from world_understanding.rendering_backend_contract import (
    RemoteRenderingSlotTimeoutError,
)


def _png_b64(color: tuple[int, int, int] = (1, 2, 3)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_legacy_render_nvcf_module_aliases_remote_helpers() -> None:
    """Old NVCF module imports should keep working during the rename window."""
    from world_understanding.functions.graphics.render_nvcf import (
        RenderingStatus as LegacyRenderingStatus,
    )
    from world_understanding.functions.graphics.render_nvcf import (
        _is_v2_response as legacy_is_v2_response,
    )

    assert LegacyRenderingStatus is RenderingStatus
    assert legacy_is_v2_response({"rendered_data": {}, "total_cameras": 0}) is True


def test_export_stage_to_s3_encodes_asset_bundle_as_data_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Data URI mode must preserve local asset bundles instead of S3 fallback."""
    from pxr import Usd

    def fake_bundle(
        stage: object,
        temp_dir: Path,
        base_dir: object | None = None,
        has_local_composition_arcs: bool | None = None,
        add_preview_fallbacks: bool | None = None,
    ) -> tuple[Path, bool]:
        assert add_preview_fallbacks is False
        zip_path = temp_dir / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stage.usda", '#usda 1.0\ndef Xform "Root" {}\n')
            zf.writestr("textures/albedo.png", b"not-a-real-png")
        return zip_path, True

    def fail_upload(*args: object, **kwargs: object) -> None:
        raise AssertionError("data URI bundle path must not upload to S3")

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote._bundle_stage_with_local_assets",
        fake_bundle,
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote.upload_file_to_s3",
        fail_upload,
    )

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Root", "Xform")

    asset_url, s3_uri = export_stage_to_s3(stage, use_data_uri=True)

    assert s3_uri is None
    assert asset_url.startswith("data:application/zip;name=bundle.zip;base64,")
    payload = base64.b64decode(asset_url.split(",", 1)[1])
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        assert "stage.usda" in zf.namelist()
        assert "textures/albedo.png" in zf.namelist()


def test_export_stage_to_s3_uploads_bundle_and_ignores_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd

    def fake_bundle(
        stage: object,
        temp_dir: Path,
        base_dir: object | None = None,
        has_local_composition_arcs: bool | None = None,
        add_preview_fallbacks: bool | None = None,
    ) -> tuple[Path, bool]:
        zip_path = temp_dir / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stage.usda", "#usda 1.0\n")
        return zip_path, True

    monkeypatch.setattr(render_remote, "_bundle_stage_with_local_assets", fake_bundle)
    monkeypatch.setattr(
        render_remote.uuid,
        "uuid4",
        lambda: type("FakeUuid", (), {"hex": "bundleid"})(),
    )
    monkeypatch.setattr(
        render_remote,
        "upload_file_to_s3",
        lambda file_path, s3_path, profile_name=None: s3_path,
    )
    monkeypatch.setattr(
        render_remote,
        "s3_uri_to_https_url",
        lambda s3_uri, region: f"https://{region}/{s3_uri}",
    )
    monkeypatch.setattr(
        "shutil.rmtree",
        lambda path: (_ for _ in ()).throw(OSError("cleanup failed")),
    )

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Root", "Xform")

    asset_url, s3_uri = export_stage_to_s3(
        stage,
        use_data_uri=False,
        s3_bucket="bucket",
        s3_region="us-west-2",
        s3_profile="profile",
    )

    assert asset_url == "https://us-west-2/s3://bucket/nvcf-renders/bundleid/bundle.zip"
    assert s3_uri == "s3://bucket/nvcf-renders/bundleid/bundle.zip"


def test_export_stage_to_s3_cleans_failed_bundle_attempt_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd

    def fake_bundle(
        stage: object,
        temp_dir: Path,
        base_dir: object | None = None,
        has_local_composition_arcs: bool | None = None,
        add_preview_fallbacks: bool | None = None,
    ) -> tuple[None, bool]:
        (temp_dir / "partial").write_text("incomplete", encoding="utf-8")
        return None, False

    monkeypatch.setattr(render_remote, "_bundle_stage_with_local_assets", fake_bundle)
    monkeypatch.setattr(
        "shutil.rmtree",
        lambda path: (_ for _ in ()).throw(OSError("cleanup failed")),
    )
    monkeypatch.setattr(
        render_remote,
        "_export_stage_and_get_url",
        lambda **kwargs: ("data:model/vnd.usd;base64,AA==", None),
    )

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Root", "Xform")

    assert export_stage_to_s3(stage, use_data_uri=True) == (
        "data:model/vnd.usd;base64,AA==",
        None,
    )


def test_export_stage_to_s3_fallback_logs_preview_updates_and_unlink_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd

    monkeypatch.setattr(
        render_remote,
        "add_ovrtx_preview_fallbacks_to_stage_file",
        lambda path: 1,
    )
    monkeypatch.setattr(
        render_remote,
        "_add_texture_file_fallbacks_for_remote_export",
        lambda path: (2, 3),
    )
    monkeypatch.setattr(
        render_remote,
        "_prefer_preview_surface_for_remote_export",
        lambda path: 4,
    )
    monkeypatch.setattr(
        render_remote,
        "_export_stage_and_get_url",
        lambda **kwargs: ("data:model/vnd.usd;base64,AA==", None),
    )
    monkeypatch.setattr(
        render_remote.os,
        "unlink",
        lambda path: (_ for _ in ()).throw(OSError("unlink failed")),
    )

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Root", "Xform")

    assert export_stage_to_s3(
        stage,
        use_data_uri=True,
        bundle_mdl_assets=False,
        material_target="preview_surface",
    ) == ("data:model/vnd.usd;base64,AA==", None)


def test_render_all_cameras_passes_base_dir_to_stage_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_export_stage_to_s3(**kwargs: object) -> tuple[str, str | None]:
        captured["base_dir"] = kwargs.get("base_dir")
        captured["material_target"] = kwargs.get("material_target")
        return "data:model/vnd.usd;base64,ZmFrZQ==", None

    def fake_render_single_camera_from_url(**kwargs: object) -> dict[str, object]:
        captured["render_material_target"] = kwargs.get("material_target")
        return {
            "camera": kwargs["camera"],
            "images": [],
            "frame_count": 1,
            "status": RenderingStatus.success,
        }

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote.export_stage_to_s3",
        fake_export_stage_to_s3,
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote.render_single_camera_from_url",
        fake_render_single_camera_from_url,
    )

    result = render_all_cameras(
        stage=object(),
        cameras=["/Camera"],
        base_dir=tmp_path,
        material_target="openpbr_materialx",
        max_workers=1,
    )

    assert captured["base_dir"] == tmp_path
    assert captured["material_target"] == "openpbr_materialx"
    assert captured["render_material_target"] == "openpbr_materialx"
    assert result["successful_cameras"] == 1


def test_render_all_cameras_returns_structured_failure_on_export_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_export_stage_to_s3(**kwargs: object) -> tuple[str, str | None]:
        raise RuntimeError("Remote REST rendering requires a flattened stage")

    def fail_render_single_camera_from_url(**kwargs: object) -> dict[str, object]:
        raise AssertionError("rendering should not start when stage export fails")

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote.export_stage_to_s3",
        fail_export_stage_to_s3,
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote.render_single_camera_from_url",
        fail_render_single_camera_from_url,
    )

    result = render_all_cameras(
        stage=object(),
        cameras=["/CameraA", "/CameraB"],
        max_workers=1,
    )

    assert result["total_cameras"] == 2
    assert result["successful_cameras"] == 0
    assert result["failed_cameras"] == 2
    assert [item["camera"] for item in result["results"]] == [
        "/CameraA",
        "/CameraB",
    ]
    assert {item["status"] for item in result["results"]} == {RenderingStatus.exception}
    assert all(
        "requires a flattened stage" in item["error"] for item in result["results"]
    )
    assert {item["error_type"] for item in result["results"]} == {"RuntimeError"}


def test_render_all_cameras_parallel_and_cleanup_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(
        render_remote,
        "export_stage_to_s3",
        lambda **kwargs: ("https://example.invalid/stage.usd", "s3://bucket/stage"),
    )
    monkeypatch.setattr(
        render_remote,
        "delete_s3_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("delete failed")),
    )

    def fake_render_single_camera_from_url(**kwargs: object) -> dict[str, object]:
        camera = kwargs["camera"]
        if camera == "/Boom":
            raise RuntimeError("camera exploded")
        return {
            "camera": camera,
            "images": [],
            "sensors": {},
            "render_time": 0.1,
            "frame_count": 0,
            "status": RenderingStatus.success
            if camera == "/Ok"
            else RenderingStatus.load_error,
        }

    monkeypatch.setattr(
        render_remote,
        "render_single_camera_from_url",
        fake_render_single_camera_from_url,
    )

    result = render_all_cameras(
        stage=stage,
        cameras=["/Ok", "/Bad", "/Boom"],
        max_workers=2,
    )

    assert result["successful_cameras"] == 1
    assert result["failed_cameras"] == 2
    assert {item["camera"] for item in result["results"]} == {"/Ok", "/Bad", "/Boom"}


@pytest.mark.parametrize("max_workers", (0, 33, True, "2"))
def test_render_all_cameras_rejects_unbounded_or_invalid_workers_before_export(
    monkeypatch: pytest.MonkeyPatch,
    max_workers: object,
) -> None:
    monkeypatch.setattr(
        render_remote,
        "export_stage_to_s3",
        lambda **kwargs: pytest.fail("invalid workers must fail before export"),
    )

    with pytest.raises(
        ValueError,
        match="Remote render max_workers must be an integer between 1 and 32",
    ):
        render_all_cameras(stage=object(), max_workers=max_workers)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_workers", (1, 32))
def test_render_all_cameras_accepts_worker_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    max_workers: int,
) -> None:
    monkeypatch.setattr(
        render_remote,
        "export_stage_to_s3",
        lambda **kwargs: ("https://example.invalid/stage.usd", None),
    )
    monkeypatch.setattr(
        render_remote,
        "render_single_camera_from_url",
        lambda **kwargs: {
            "camera": kwargs["camera"],
            "status": RenderingStatus.success,
        },
    )

    result = render_all_cameras(
        stage=object(),
        cameras=["/Camera"],
        max_workers=max_workers,
    )

    assert result["successful_cameras"] == 1


def test_render_all_cameras_parallel_honors_global_request_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_requests = 0
    max_active_requests = 0
    counters_lock = threading.Lock()

    monkeypatch.setattr(
        render_remote,
        "export_stage_to_s3",
        lambda **kwargs: ("https://example.invalid/stage.usd", None),
    )

    def fake_render_single_camera_from_url(**kwargs: object) -> dict[str, object]:
        nonlocal active_requests, max_active_requests
        with counters_lock:
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
        time.sleep(0.02)
        with counters_lock:
            active_requests -= 1
        return {
            "camera": kwargs["camera"],
            "images": [],
            "sensors": {},
            "render_time": 0.02,
            "frame_count": 0,
            "status": RenderingStatus.success,
        }

    monkeypatch.setattr(
        render_remote,
        "render_single_camera_from_url",
        fake_render_single_camera_from_url,
    )
    monkeypatch.setenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", "2")
    render_remote_async._reset_global_remote_render_semaphore_for_tests()

    try:
        result = render_all_cameras(
            stage=object(),
            cameras=[f"/Camera{i}" for i in range(8)],
            max_workers=8,
            use_global_render_slots=True,
        )
    finally:
        render_remote_async._reset_global_remote_render_semaphore_for_tests()

    assert result["successful_cameras"] == 8
    assert max_active_requests == 2


def test_render_all_cameras_parallel_reports_slot_timeout_per_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimeoutSlot:
        def __enter__(self) -> float:
            raise RemoteRenderingSlotTimeoutError("slot capacity unavailable")

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        render_remote,
        "export_stage_to_s3",
        lambda **kwargs: ("https://example.invalid/stage.usd", None),
    )
    monkeypatch.setattr(
        render_remote_async,
        "global_remote_render_slot",
        lambda **kwargs: TimeoutSlot(),
    )
    monkeypatch.setattr(
        render_remote,
        "render_single_camera_from_url",
        lambda **kwargs: pytest.fail("timed-out slots must not issue requests"),
    )

    result = render_all_cameras(
        stage=object(),
        cameras=["/CameraA", "/CameraB"],
        max_workers=2,
        use_global_render_slots=True,
        render_slot_timeout_sec=0.01,
    )

    assert result["successful_cameras"] == 0
    assert result["failed_cameras"] == 2
    assert {item["camera"] for item in result["results"]} == {
        "/CameraA",
        "/CameraB",
    }
    assert all(
        item["status"] == RenderingStatus.exception
        and item["error"] == "slot capacity unavailable"
        for item in result["results"]
    )


def test_render_all_cameras_defaults_camera_and_cleans_s3_successfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    deleted: list[str] = []

    monkeypatch.setattr(
        render_remote,
        "export_stage_to_s3",
        lambda **kwargs: ("https://example.invalid/stage.usd", "s3://bucket/stage"),
    )
    monkeypatch.setattr(
        render_remote,
        "delete_s3_path",
        lambda s3_uri, profile_name=None: deleted.append(s3_uri),
    )
    monkeypatch.setattr(
        render_remote,
        "render_single_camera_from_url",
        lambda **kwargs: {
            "camera": kwargs["camera"],
            "images": [],
            "sensors": {},
            "render_time": 0.0,
            "frame_count": 0,
            "status": RenderingStatus.success,
        },
    )

    result = render_all_cameras(stage=stage, cameras=None, max_workers=1)

    assert result["total_cameras"] == 1
    assert result["successful_cameras"] == 1
    assert result["results"][0]["camera"] == "/Camera"
    assert deleted == ["s3://bucket/stage"]


def test_render_all_cameras_sequential_counts_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(
        render_remote,
        "export_stage_to_s3",
        lambda **kwargs: ("https://example.invalid/stage.usd", None),
    )
    monkeypatch.setattr(
        render_remote,
        "render_single_camera_from_url",
        lambda **kwargs: {
            "camera": kwargs["camera"],
            "images": [],
            "sensors": {},
            "render_time": 0.0,
            "frame_count": 0,
            "status": RenderingStatus.load_error,
        },
    )

    result = render_all_cameras(stage=stage, cameras=["/Bad"], max_workers=1)

    assert result["successful_cameras"] == 0
    assert result["failed_cameras"] == 1


def test_render_all_cameras_from_url_default_sequential_and_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_render_single_camera_from_url(**kwargs: object) -> dict[str, object]:
        camera = kwargs["camera"]
        if camera == "/Raise":
            raise RuntimeError("render failed")
        return {
            "camera": camera,
            "images": [],
            "sensors": {},
            "render_time": 0.0,
            "frame_count": 0,
            "status": RenderingStatus.success
            if camera in {"/Camera", "/Ok"}
            else RenderingStatus.load_error,
        }

    monkeypatch.setattr(
        render_remote,
        "render_single_camera_from_url",
        fake_render_single_camera_from_url,
    )

    default_result = render_remote.render_all_cameras_from_url(
        "https://example.invalid/stage.usd",
        cameras=[],
        max_workers=1,
    )
    assert default_result["total_cameras"] == 1
    assert default_result["successful_cameras"] == 1

    parallel_result = render_remote.render_all_cameras_from_url(
        "https://example.invalid/stage.usd",
        cameras=["/Ok", "/Bad", "/Raise"],
        max_workers=2,
    )
    assert parallel_result["successful_cameras"] == 1
    assert parallel_result["failed_cameras"] == 2


def test_render_all_cameras_from_url_sequential_counts_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        render_remote,
        "render_single_camera_from_url",
        lambda **kwargs: {
            "camera": kwargs["camera"],
            "images": [],
            "sensors": {},
            "render_time": 0.0,
            "frame_count": 0,
            "status": RenderingStatus.load_error,
        },
    )

    result = render_remote.render_all_cameras_from_url(
        "https://example.invalid/stage.usd",
        cameras=["/Bad"],
        max_workers=1,
    )

    assert result["successful_cameras"] == 0
    assert result["failed_cameras"] == 1


def test_batch_render_assets_counts_success_failure_and_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="No asset URLs"):
        render_remote.batch_render_assets([])

    def fake_render_single_camera_from_url(**kwargs: object) -> dict[str, object]:
        asset = kwargs["usd_url"]
        camera = kwargs["camera"]
        if "explode" in asset:
            raise RuntimeError("asset exploded")
        return {
            "camera": camera,
            "images": [],
            "sensors": {},
            "render_time": 0.0,
            "frame_count": 0,
            "status": RenderingStatus.success
            if "ok" in asset
            else RenderingStatus.load_error,
        }

    monkeypatch.setattr(
        render_remote,
        "render_single_camera_from_url",
        fake_render_single_camera_from_url,
    )

    result = render_remote.batch_render_assets(
        [
            "https://example.invalid/ok.usd",
            "https://example.invalid/bad.usd",
            "https://example.invalid/explode.usd",
        ],
        cameras=[],
        max_workers=2,
    )

    assert result["total_assets"] == 3
    assert result["successful_assets"] == 1
    assert result["failed_assets"] == 2
    assert (
        result["asset_results"]["https://example.invalid/explode.usd"]["failed_cameras"]
        == 1
    )


def test_save_render_results_writes_images_depth_and_segmentation(
    tmp_path: Path,
) -> None:
    result = {
        "images": [Image.new("RGB", (2, 2), (1, 2, 3))],
        "sensors": {
            "linear_depth": {0: np.ones((2, 2, 1), dtype=np.float32)},
            "instance_id_segmentation": {
                1: np.array([[[0], [1]], [[2], [3]]], dtype=np.uint32)
            },
        },
    }

    stats = save_render_results(
        result,
        tmp_path,
        file_name="scene",
        image_width=2,
        image_height=2,
        save_npy=True,
    )

    assert stats == {"total_count": 3, "success_count": 3, "error_count": 0}
    assert (tmp_path / "scene_f0000_images.png").exists()
    assert (tmp_path / "scene_f0000_linear_depth.npy").exists()
    assert (tmp_path / "scene_f0000_linear_depth.png").exists()
    assert (tmp_path / "scene_f0001_instance_id_segmentation.npy").exists()
    assert (tmp_path / "scene_f0001_instance_id_segmentation.png").exists()


def test_http_error_detail_extracts_blank_render_error() -> None:
    response = requests.Response()
    response.status_code = 422
    response._content = (
        b'{"detail":{"status":"blank_render",'
        b'"error":"1/1 OVRTX render frames are blank or near-blank."}}'
    )
    response.headers["Content-Type"] = "application/json"

    assert _http_error_detail(response) == (
        "1/1 OVRTX render frames are blank or near-blank."
    )


def test_http_error_payload_handles_text_detail_and_non_json() -> None:
    text_response = requests.Response()
    text_response.status_code = 500
    text_response._content = b"plain renderer failure"
    text_response.reason = "Internal Server Error"

    assert _http_error_payload(text_response) == {"error": "plain renderer failure"}

    detail_response = requests.Response()
    detail_response.status_code = 400
    detail_response._content = b'{"detail":"bad request"}'
    detail_response.headers["Content-Type"] = "application/json"

    assert _http_error_payload(detail_response) == {"error": "bad request"}

    payload_response = requests.Response()
    payload_response.status_code = 400
    payload_response._content = b'{"code":"bad"}'
    payload_response.headers["Content-Type"] = "application/json"

    assert _http_error_payload(payload_response) == {"error": '{"code": "bad"}'}


def test_http_error_detail_serializes_structured_payload_without_error() -> None:
    response = requests.Response()
    response.status_code = 400
    response._content = b'{"detail":{"code":"bad","status":"failed"}}'
    response.headers["Content-Type"] = "application/json"

    assert _http_error_detail(response) == '{"code": "bad", "status": "failed"}'


def test_convert_v2_to_v1_preserves_passthrough_sensor_values() -> None:
    result = {
        "total_cameras": 1,
        "total_frames": 1,
        "rendered_data": {
            "/Camera": {
                "0": {
                    "metadata": "raw-value",
                    "normal": {"data": "base64-without-shape"},
                }
            }
        },
    }

    converted = _convert_v2_to_v1(result)

    assert converted["status"] == RenderingStatus.success
    assert converted["images"]["0"]["/Camera"]["metadata"] == "raw-value"
    assert converted["images"]["0"]["/Camera"]["normal"] == "base64-without-shape"


def test_export_stage_and_get_url_uses_data_uri_and_s3_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "stage.usda"
    stage_path.write_text("#usda 1.0\n", encoding="utf-8")

    monkeypatch.setattr(
        render_remote,
        "create_data_uri_from_file",
        lambda path: f"data://{Path(path).name}",
    )
    assert _export_stage_and_get_url(
        str(stage_path),
        use_data_uri=True,
        s3_bucket="bucket",
        s3_profile=None,
        s3_region="us-east-1",
    ) == ("data://stage.usda", None)

    monkeypatch.setattr(
        render_remote.uuid, "uuid4", lambda: type("U", (), {"hex": "abc"})()
    )
    monkeypatch.setattr(
        render_remote,
        "upload_file_to_s3",
        lambda file_path, s3_path, profile_name=None: s3_path,
    )
    monkeypatch.setattr(
        render_remote,
        "s3_uri_to_https_url",
        lambda s3_uri, region: f"https://{region}/{s3_uri}",
    )

    assert _export_stage_and_get_url(
        str(stage_path),
        use_data_uri=False,
        s3_bucket="bucket",
        s3_profile="profile",
        s3_region="us-west-2",
    ) == (
        "https://us-west-2/s3://bucket/nvcf-renders/abc/stage.usda",
        "s3://bucket/nvcf-renders/abc/stage.usda",
    )


def test_render_single_camera_preserves_blank_render_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = requests.Response()
    response.status_code = 422
    response._content = (
        b'{"detail":{"status":"blank_render",'
        b'"error":"1/1 OVRTX render frames are blank or near-blank.",'
        b'"warnings":["1/1 frames blank"],'
        b'"blank_render_frames":[{"frame":0,"camera":"/Camera"}]}}'
    )
    response.headers["Content-Type"] = "application/json"
    response.url = "http://renderer/render"
    response.reason = "Unprocessable Entity"

    def fake_post(*args: object, **kwargs: object) -> requests.Response:
        return response

    monkeypatch.setattr(requests, "post", fake_post)

    result = render_single_camera_from_url(
        "data:model/vnd.usd;base64,ZmFrZQ==",
        "/Camera",
        api_key="test-key",
        base_url="http://renderer",
        max_retries=0,
    )

    assert result["status"] == RenderingStatus.blank_render
    assert result["warnings"] == ["1/1 frames blank"]
    assert result["blank_render_frames"] == [{"frame": 0, "camera": "/Camera"}]


def test_render_single_camera_from_url_forwards_material_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    response = requests.Response()
    response.status_code = 200
    response._content = b'{"status":"success","images":{"0":{"/Camera":{}}}}'
    response.headers["Content-Type"] = "application/json"

    def fake_post(*args: object, **kwargs: object) -> requests.Response:
        captured["json"] = kwargs["json"]
        return response

    monkeypatch.setattr(requests, "post", fake_post)

    result = render_single_camera_from_url(
        "data:model/vnd.usd;base64,ZmFrZQ==",
        "/Camera",
        api_key="test-key",
        base_url="http://renderer",
        max_retries=0,
        material_target="openpbr_materialx",
    )

    assert result["status"] == RenderingStatus.success
    assert captured["json"]["render_settings"]["material_target"] == (
        "openpbr_materialx"
    )


def test_render_single_camera_rejects_non_usd_stage() -> None:
    with pytest.raises(ValueError, match="stage must be a Usd.Stage"):
        render_remote.render_single_camera(
            object(),
            "/Camera",
            api_key="test-key",
            base_url="http://renderer",
        )


def test_render_single_camera_exports_renders_and_cleans_s3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Root", "Xform")
    deleted: list[tuple[str, str | None]] = []
    captured: dict[str, object] = {}

    def fake_export_stage_and_get_url(**kwargs: object) -> tuple[str, str | None]:
        captured["export_exists"] = Path(str(kwargs["stage_path"])).exists()
        captured["use_data_uri"] = kwargs["use_data_uri"]
        return "https://example.invalid/stage.usd", "s3://bucket/stage"

    monkeypatch.setattr(
        render_remote,
        "_export_stage_and_get_url",
        fake_export_stage_and_get_url,
    )
    monkeypatch.setattr(
        render_remote,
        "render_single_camera_from_url",
        lambda **kwargs: {
            "camera": kwargs["camera"],
            "images": [],
            "sensors": {},
            "render_time": 0.0,
            "frame_count": 0,
            "status": RenderingStatus.success,
        },
    )
    monkeypatch.setattr(
        render_remote,
        "delete_s3_path",
        lambda s3_uri, profile_name=None: deleted.append((s3_uri, profile_name)),
    )
    monkeypatch.setattr(
        render_remote.os,
        "unlink",
        lambda path: (_ for _ in ()).throw(OSError("unlink failed")),
    )

    result = render_remote.render_single_camera(
        stage,
        "/Camera",
        use_data_uri=False,
        s3_profile="profile",
        api_key="test-key",
        base_url="http://renderer",
    )

    assert result["status"] == RenderingStatus.success
    assert captured == {"export_exists": True, "use_data_uri": False}
    assert deleted == [("s3://bucket/stage", "profile")]


def test_render_single_camera_from_url_retries_then_converts_v2_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgb = np.full((2, 2, 4), 128, dtype=np.uint8)
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(
        {
            "total_cameras": 1,
            "total_frames": 1,
            "rendered_data": {
                "/Camera": {
                    "2": {
                        "rgb": {
                            "type": "array",
                            "data": base64.b64encode(rgb.tobytes()).decode(),
                            "shape": [2, 2, 4],
                            "dtype": "uint8",
                        }
                    }
                }
            },
        },
    ).encode()
    response.headers["Content-Type"] = "application/json"
    calls = {"count": 0}

    def fake_post(*args: object, **kwargs: object) -> requests.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.ConnectionError("temporary outage")
        return response

    sleeps: list[float] = []
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(render_remote.time, "sleep", lambda delay: sleeps.append(delay))

    result = render_single_camera_from_url(
        "data:model/vnd.usd;base64,ZmFrZQ==",
        "/Camera",
        api_key="test-key",
        base_url="http://renderer",
        frames="2:4",
        max_retries=1,
        retry_delay=0.25,
        retry_jitter=0.0,
    )

    assert calls["count"] == 2
    assert sleeps == [0.25]
    assert result["status"] == RenderingStatus.success
    assert result["frame_count"] == 1
    assert result["images"][0].size == (2, 2)


def test_render_single_camera_from_url_rejects_bad_content_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = requests.Response()
    response.status_code = 200
    response._content = b"not-json"
    response.headers["Content-Type"] = "text/plain"

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(ValueError, match="Unexpected content type"):
        render_single_camera_from_url(
            "data:model/vnd.usd;base64,ZmFrZQ==",
            "/Camera",
            api_key="test-key",
            base_url="http://renderer",
            max_retries=0,
        )


def test_render_single_camera_from_url_rejects_unparseable_zip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = requests.Response()
    response.status_code = 200
    response._content = b"zip-bytes"
    response.headers["Content-Type"] = "application/zip"

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)
    monkeypatch.setattr(render_remote, "_parse_zip_response", lambda content: None)

    with pytest.raises(ValueError, match="Failed to parse ZIP response"):
        render_single_camera_from_url(
            "data:model/vnd.usd;base64,ZmFrZQ==",
            "/Camera",
            api_key="test-key",
            base_url="http://renderer",
            max_retries=0,
        )


def test_parse_zip_response_and_render_single_camera_zip_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "status": "success",
        "images": {"0": {"/Camera": {"images": _png_b64()}}},
        "error": None,
    }
    direct_buffer = io.BytesIO()
    with zipfile.ZipFile(direct_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("render.response", json.dumps(payload))
    assert render_remote._parse_zip_response(direct_buffer.getvalue())["status"] == (
        "success"
    )

    missing_response = io.BytesIO()
    with zipfile.ZipFile(missing_response, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("readme.txt", "no response")
    with pytest.raises(ValueError, match="No .response"):
        render_remote._parse_zip_response(missing_response.getvalue())

    response = requests.Response()
    response.status_code = 200
    response._content = direct_buffer.getvalue()
    response.headers["Content-Type"] = "application/zip"

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)

    result = render_single_camera_from_url(
        "data:model/vnd.usd;base64,ZmFrZQ==",
        "/Camera",
        api_key="test-key",
        base_url="http://renderer",
        max_retries=0,
    )

    assert result["status"] == RenderingStatus.success
    assert result["frame_count"] == 1
    assert result["images"][0].size == (2, 2)


def test_render_single_camera_success_decodes_images_and_sensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    segmentation = np.array([1, 2, 3, 4], dtype=np.uint32)
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(
        {
            "status": "success",
            "warnings": ["careful"],
            "blank_render_frames": [{"frame": 1}],
            "images": {
                "1": {
                    "/Camera": {
                        "images": _png_b64((4, 5, 6)),
                        "linear_depth": base64.b64encode(depth.tobytes()).decode(),
                        "instance_id_segmentation": base64.b64encode(
                            segmentation.tobytes()
                        ).decode(),
                    }
                }
            },
        }
    ).encode()
    response.headers["Content-Type"] = "application/json"

    captured: dict[str, object] = {}

    def fake_post(*args: object, **kwargs: object) -> requests.Response:
        captured["json"] = kwargs["json"]
        return response

    monkeypatch.setattr(requests, "post", fake_post)

    result = render_single_camera_from_url(
        "data:model/vnd.usd;base64,ZmFrZQ==",
        "/Camera",
        api_key="test-key",
        base_url="http://renderer",
        frames="1",
        sensors=["linear_depth", "instance_id_segmentation"],
        max_retries=0,
    )

    assert captured["json"]["render_settings"]["frame_range"] == {"start": 1, "end": 1}
    assert result["status"] == RenderingStatus.success
    assert result["warnings"] == ["careful"]
    assert result["blank_render_frames"] == [{"frame": 1}]
    assert result["images"][0].size == (2, 2)
    assert result["sensors"]["linear_depth"][1].dtype == np.float32
    assert result["sensors"]["instance_id_segmentation"][1].dtype == np.uint32


def test_render_single_camera_handles_failure_status_and_bad_decodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure_response = requests.Response()
    failure_response.status_code = 200
    failure_response._content = b'{"status":"load_error","error":"bad stage"}'
    failure_response.headers["Content-Type"] = "application/json"

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: failure_response)
    failed = render_single_camera_from_url(
        "data:model/vnd.usd;base64,ZmFrZQ==",
        "/Camera",
        api_key="test-key",
        base_url="http://renderer",
        max_retries=0,
    )
    assert failed["status"] == "load_error"
    assert "bad stage" in failed["error"]

    bad_decode_response = requests.Response()
    bad_decode_response.status_code = 200
    bad_decode_response._content = b"""
    {
      "status": "success",
      "images": {
        "0": {
          "/Camera": {
            "images": "not-image",
            "linear_depth": "not-array"
          }
        }
      }
    }
    """
    bad_decode_response.headers["Content-Type"] = "application/json"

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: bad_decode_response)
    decoded = render_single_camera_from_url(
        "data:model/vnd.usd;base64,ZmFrZQ==",
        "/Camera",
        api_key="test-key",
        base_url="http://renderer",
        sensors=["linear_depth"],
        max_retries=0,
    )
    assert decoded["status"] == RenderingStatus.success
    assert decoded["frame_count"] == 0
    assert decoded["sensors"]["linear_depth"] == {}


def test_local_composition_asset_path_detection_uses_shared_uri_semantics() -> None:
    assert _is_local_composition_asset_path("./geometry.usda")
    assert _is_local_composition_asset_path("/tmp/geometry.usda")
    assert _is_local_composition_asset_path("C:/assets/geometry.usda")
    assert _is_local_composition_asset_path(r"C:\assets\geometry.usda")
    assert _is_local_composition_asset_path("file:/tmp/geometry.usda")

    assert not _is_local_composition_asset_path("")
    assert _is_local_composition_asset_path("anon:000002")
    assert not _is_local_composition_asset_path("https://example.com/geometry.usda")
    assert not _is_local_composition_asset_path("s3://bucket/geometry.usda")
    assert not _is_local_composition_asset_path("omniverse://server/geometry.usda")
    assert not _is_local_composition_asset_path(
        "data:application/octet-stream;base64,AA"
    )


def test_local_composition_arc_guard_ignores_deleted_list_ops() -> None:
    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/DeleteReference", "Xform")
    prim_spec = stage.GetRootLayer().GetPrimAtPath("/World/DeleteReference")
    prim_spec.referenceList.deletedItems = [Sdf.Reference("./deleted.usda")]
    prim_spec.payloadList.deletedItems = [Sdf.Payload("./deleted_payload.usda")]

    assert not _stage_has_local_composition_arcs(stage)


def test_local_composition_arc_guard_rejects_anonymous_arcs() -> None:
    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Reference", "Xform")
    prim_spec = stage.GetRootLayer().GetPrimAtPath("/World/Reference")
    prim_spec.referenceList.addedItems = [Sdf.Reference("anon:000002:layer.usda")]

    assert _stage_has_local_composition_arcs(stage)


def test_local_composition_arc_guard_rejects_sublayers_and_payloads(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd

    sublayer_stage = Usd.Stage.CreateInMemory()
    sublayer_stage.GetRootLayer().subLayerPaths.append("./local_sublayer.usda")
    assert _stage_has_local_composition_arcs(sublayer_stage)

    payload_stage = Usd.Stage.CreateInMemory()
    payload_stage.DefinePrim("/World/Payload", "Xform")
    prim_spec = payload_stage.GetRootLayer().GetPrimAtPath("/World/Payload")
    prim_spec.payloadList.addedItems = [Sdf.Payload(str(tmp_path / "payload.usda"))]

    assert _stage_has_local_composition_arcs(payload_stage)


def test_resolve_export_asset_path_falls_back_when_resolve_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_path = tmp_path / "textures" / "bad.png"

    def raise_os_error(self: Path, strict: bool = False) -> Path:
        raise OSError("bad path")

    monkeypatch.setattr(Path, "resolve", raise_os_error)

    assert _resolve_export_asset_path("textures/bad.png", tmp_path) == str(
        expected_path
    )


def test_resolve_export_asset_path_preserves_windows_file_uri(tmp_path: Path) -> None:
    resolved = _resolve_export_asset_path("file:///C:/textures/albedo.png", tmp_path)

    assert str(tmp_path) not in resolved
    assert resolved.replace("\\", "/").endswith("C:/textures/albedo.png")


def test_resolve_export_asset_path_preserves_file_uri_netloc(tmp_path: Path) -> None:
    resolved = _resolve_export_asset_path("file://server/share/albedo.png", tmp_path)

    assert resolved.endswith("/server/share/albedo.png")


def test_bundle_stage_rewrites_relative_mdl_source_asset(tmp_path: Path) -> None:
    """Bundled stages must point MDL source assets at copied bundle paths."""
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    (asset_root / "materials" / "OmniPBR").mkdir(parents=True)
    (asset_root / "textures").mkdir()
    (asset_root / "materials" / "OmniPBR" / "OmniPBR.mdl").write_text(
        "mdl 1.7;\n",
        encoding="utf-8",
    )
    (asset_root / "textures" / "albedo.png").write_bytes(b"not-a-real-png")

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Shader")
    shader.CreateIdAttr("mdl:OmniPBR")
    shader.GetPrim().CreateAttribute(
        "info:implementationSource",
        Sdf.ValueTypeNames.Token,
    ).Set("sourceAsset")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./materials/OmniPBR/OmniPBR.mdl"))
    shader.GetPrim().CreateAttribute(
        "inputs:diffuse_texture",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./textures/albedo.png"))
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(stage, tmp_path / "bundle")

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        assert "mdl_materials/OmniPBR/OmniPBR.mdl" in zf.namelist()
        assert "textures/albedo.png" in zf.namelist()
        stage_text = zf.read("stage.usda").decode("utf-8")

    assert "@mdl_materials/OmniPBR/OmniPBR.mdl@" in stage_text
    assert "@./materials/OmniPBR/OmniPBR.mdl@" not in stage_text
    assert "@textures/albedo.png@" in stage_text


def test_bundle_stage_handles_duplicate_mdl_dir_names_and_fallback_rewrites(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    first_dir = asset_root / "first" / "Shared"
    second_dir = asset_root / "second" / "Shared"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    first_mdl = first_dir / "Main.mdl"
    fallback_mdl = first_dir / "Other.mdl"
    second_mdl = second_dir / "Main.mdl"
    first_mdl.write_text("mdl 1.7;\n", encoding="utf-8")
    fallback_mdl.write_text("mdl 1.7;\n", encoding="utf-8")
    second_mdl.write_text("mdl 1.7;\n", encoding="utf-8")

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    for name, source_asset in {
        "Direct": first_mdl,
        "Fallback": fallback_mdl,
        "Collision": second_mdl,
    }.items():
        shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
        shader.CreateIdAttr("mdl:OmniPBR")
        shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath(str(source_asset)))
    stage.GetPrimAtPath("/World/Looks/Direct/Shader").CreateAttribute(
        "inputs:unusedAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./not-a-texture.png"))
    stage.GetPrimAtPath("/World/Looks/Direct/Shader").CreateAttribute(
        "inputs:emptyAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(""))
    stage.GetRootLayer().Save()

    monkeypatch.setattr(
        render_remote,
        "get_local_mdl_assets",
        lambda stage_arg, base_dir=None: [
            {"is_local": True, "resolved_path": str(first_mdl)},
            {"is_local": True, "resolved_path": str(second_mdl)},
        ],
    )

    zip_path, bundled = _bundle_stage_with_local_assets(
        stage,
        tmp_path / "bundle",
        base_dir=asset_root,
    )

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        stage_text = zf.read("stage.usda").decode("utf-8")

    assert "mdl_materials/Shared/Main.mdl" in names
    assert "mdl_materials/Shared_1/Main.mdl" in names
    assert "@mdl_materials/Shared/Main.mdl@" in stage_text
    assert "@mdl_materials/Shared/Other.mdl@" in stage_text
    assert "@mdl_materials/Shared_1/Main.mdl@" in stage_text


def test_bundle_stage_returns_false_when_asset_copying_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    mdl_dir = asset_root / "materials"
    texture_dir = asset_root / "textures"
    mdl_dir.mkdir(parents=True)
    texture_dir.mkdir()
    mdl_path = mdl_dir / "Broken.mdl"
    texture_path = texture_dir / "broken.png"
    mdl_path.write_text("mdl 1.7;\n", encoding="utf-8")
    texture_path.write_bytes(b"texture")

    stage = Usd.Stage.CreateNew(str(asset_root / "scene.usda"))
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Shader")
    shader.CreateIdAttr("mdl:Broken")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(mdl_path)))
    shader.GetPrim().CreateAttribute("inputs:file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(texture_path)),
    )
    stage.GetRootLayer().Save()

    monkeypatch.setattr(
        "shutil.copytree",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("copytree failed")),
    )
    monkeypatch.setattr(
        "shutil.copy2",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("copy2 failed")),
    )

    assert _bundle_stage_with_local_assets(stage, tmp_path / "bundle") == (None, False)


def test_bundle_stage_returns_false_when_exported_layer_cannot_reopen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    mdl_dir = asset_root / "materials"
    mdl_dir.mkdir(parents=True)
    mdl_path = mdl_dir / "Mat.mdl"
    mdl_path.write_text("mdl 1.7;\n", encoding="utf-8")

    stage = Usd.Stage.CreateNew(str(asset_root / "scene.usda"))
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Shader")
    shader.CreateIdAttr("mdl:Mat")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(mdl_path)))
    stage.GetRootLayer().Save()

    monkeypatch.setattr(Sdf.Layer, "FindOrOpen", staticmethod(lambda path: None))

    assert _bundle_stage_with_local_assets(stage, tmp_path / "bundle") == (None, False)


def test_prefer_preview_surface_handles_missing_and_removes_mdl_outputs(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    stage_path = tmp_path / "preview.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    material = UsdShade.Material.Define(stage, "/World/Looks/Mat")
    preview = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token),
    )
    material.GetPrim().CreateAttribute(
        "outputs:mdl:surface",
        Sdf.ValueTypeNames.Token,
    ).Set("mdl")
    stage.GetRootLayer().Save()

    assert _prefer_preview_surface_for_remote_export(stage_path) == 1
    reopened = Usd.Stage.Open(str(stage_path))
    assert not reopened.GetPrimAtPath("/World/Looks/Mat").HasProperty(
        "outputs:mdl:surface"
    )


def test_texture_file_fallback_export_helper_saves_when_updates_are_authored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pxr import Usd

    stage_path = tmp_path / "stage.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    stage.DefinePrim("/World", "Xform")
    stage.GetRootLayer().Save()

    monkeypatch.setattr(
        render_remote,
        "bake_texture_file_materials_to_display_color_for_render",
        lambda stage_arg: 2,
    )
    captured: dict[str, object] = {}

    def fake_add_fallbacks(stage_arg, **kwargs):
        captured.update(kwargs)
        return 3

    monkeypatch.setattr(
        render_remote,
        "add_ovrtx_preview_fallbacks_for_texture_file_materials",
        fake_add_fallbacks,
    )

    assert _add_texture_file_fallbacks_for_remote_export(stage_path) == (2, 3)
    assert captured["diffuse_color_primvar"] == "displayColor"

    captured.clear()
    assert _add_texture_file_fallbacks_for_remote_export(
        stage_path,
        connect_diffuse_texture=True,
        preserve_mdl_surface=True,
    ) == (0, 3)
    assert captured["connect_diffuse_texture"] is True
    assert captured["skip_connected_mdl_surface"] is True
    assert captured["diffuse_color_primvar"] is None


def test_preview_and_texture_fallback_helpers_handle_missing_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pxr import Usd

    monkeypatch.setattr(Usd.Stage, "Open", staticmethod(lambda path: None))

    assert _prefer_preview_surface_for_remote_export(tmp_path / "missing.usda") == 0
    assert _add_texture_file_fallbacks_for_remote_export(
        tmp_path / "missing.usda",
    ) == (0, 0)


def test_prefer_preview_surface_skips_non_preview_and_defensive_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    stage_path = tmp_path / "materials.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdShade.Material.Define(stage, "/World/Looks/NoSurface")
    non_preview = UsdShade.Material.Define(stage, "/World/Looks/NonPreview")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/NonPreview/Shader")
    shader.CreateIdAttr("ND_standard_surface_surfaceshader")
    non_preview.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token),
    )
    stage.GetRootLayer().Save()

    assert _prefer_preview_surface_for_remote_export(stage_path) == 0

    monkeypatch.setattr(
        UsdShade.Material,
        "ComputeSurfaceSource",
        lambda self: (_ for _ in ()).throw(RuntimeError("bad source")),
    )
    assert _prefer_preview_surface_for_remote_export(stage_path) == 0

    monkeypatch.setattr(
        UsdShade.Material,
        "ComputeSurfaceSource",
        lambda self: (None,),
    )
    assert _prefer_preview_surface_for_remote_export(stage_path) == 0

    monkeypatch.setattr(
        UsdShade.Material,
        "ComputeSurfaceSource",
        lambda self: (),
    )
    assert _prefer_preview_surface_for_remote_export(stage_path) == 0


def test_bundle_stage_rewrites_file_uri_mdl_source_asset(tmp_path: Path) -> None:
    """Local file-URI MDL refs must be rewritten to copied bundle paths."""
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    mdl_dir = asset_root / "materials" / "OmniPBR"
    mdl_dir.mkdir(parents=True)
    mdl_path = mdl_dir / "OmniPBR.mdl"
    mdl_path.write_text("mdl 1.7;\n", encoding="utf-8")

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Shader")
    shader.CreateIdAttr("mdl:OmniPBR")
    shader.GetPrim().CreateAttribute(
        "info:implementationSource",
        Sdf.ValueTypeNames.Token,
    ).Set("sourceAsset")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(mdl_path.as_uri()))
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(stage, tmp_path / "bundle")

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        assert "mdl_materials/OmniPBR/OmniPBR.mdl" in zf.namelist()
        stage_text = zf.read("stage.usda").decode("utf-8")

    assert "@mdl_materials/OmniPBR/OmniPBR.mdl@" in stage_text
    assert mdl_path.as_uri() not in stage_text


def test_bundle_stage_rewrites_texture_paths_by_resolved_asset_path(
    tmp_path: Path,
) -> None:
    """Textures with the same basename must not collapse to one bundle path."""
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    mat_a = asset_root / "mat_a"
    mat_b = asset_root / "mat_b"
    mat_a.mkdir(parents=True)
    mat_b.mkdir()
    (mat_a / "diffuse.png").write_bytes(b"texture-a")
    (mat_b / "diffuse.png").write_bytes(b"texture-b")

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    shader_a = UsdShade.Shader.Define(stage, "/World/Looks/MatA/Shader")
    shader_a.GetPrim().CreateAttribute(
        "inputs:file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./mat_a/diffuse.png"))
    shader_b = UsdShade.Shader.Define(stage, "/World/Looks/MatB/Shader")
    shader_b.GetPrim().CreateAttribute(
        "inputs:file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./mat_b/diffuse.png"))
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(stage, tmp_path / "bundle")

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        assert "textures/diffuse.png" in zf.namelist()
        assert "textures/diffuse_1.png" in zf.namelist()
        stage_text = zf.read("stage.usda").decode("utf-8")

    layer = Sdf.Layer.CreateAnonymous("bundled-stage.usda")
    assert layer.ImportFromString(stage_text)
    shader_a_spec = layer.GetPrimAtPath("/World/Looks/MatA/Shader")
    shader_b_spec = layer.GetPrimAtPath("/World/Looks/MatB/Shader")

    assert shader_a_spec.attributes["inputs:file"].default == Sdf.AssetPath(
        "textures/diffuse.png"
    )
    assert shader_b_spec.attributes["inputs:file"].default == Sdf.AssetPath(
        "textures/diffuse_1.png"
    )


def test_bundle_stage_skips_duplicate_texture_copy_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    texture_dir = asset_root / "textures"
    texture_dir.mkdir(parents=True)
    texture_path = texture_dir / "diffuse.png"
    texture_path.write_bytes(b"texture")

    stage = Usd.Stage.CreateNew(str(asset_root / "scene.usda"))
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Shader")
    shader.GetPrim().CreateAttribute("inputs:file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(texture_path)),
    )
    stage.GetRootLayer().Save()

    monkeypatch.setattr(render_remote, "get_local_mdl_assets", lambda *args, **kw: [])
    monkeypatch.setattr(
        render_remote,
        "get_local_texture_file_assets",
        lambda *args, **kw: [
            {
                "is_local": True,
                "resolved_path": str(texture_path),
                "prim_path": "/World/Looks/Mat/Shader",
                "attr_name": "inputs:file",
            },
            {
                "is_local": True,
                "resolved_path": str(texture_path),
                "prim_path": "/World/Looks/Mat/Shader",
                "attr_name": "inputs:file",
            },
        ],
    )

    zip_path, bundled = _bundle_stage_with_local_assets(stage, tmp_path / "bundle")

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist().count("textures/diffuse.png") == 1


def test_bundle_stage_preserves_texture_material_as_uv_texture_fallback(
    tmp_path: Path,
) -> None:
    """Preview fallback bundles should keep edited albedo detail through UsdUVTexture."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    asset_root = tmp_path / "asset"
    texture_dir = asset_root / "textures"
    texture_dir.mkdir(parents=True)
    texture_path = texture_dir / "painted_albedo.png"
    image = Image.new("RGB", (2, 2))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (0, 255, 0))
    image.putpixel((0, 1), (0, 0, 255))
    image.putpixel((1, 1), (255, 255, 0))
    image.save(texture_path)

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(1, 0, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(0, 1, 0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        "faceVarying",
    )
    st.Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0, 0),
                Gf.Vec2f(1, 0),
                Gf.Vec2f(1, 1),
                Gf.Vec2f(0, 1),
            ],
        ),
    )
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./textures/painted_albedo.png"))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(
        stage,
        tmp_path / "bundle",
        add_preview_fallbacks=True,
    )

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        assert "textures/painted_albedo.png" in zf.namelist()
        stage_text = zf.read("stage.usda").decode("utf-8")

    assert "@textures/painted_albedo.png@" in stage_text
    layer = Sdf.Layer.CreateAnonymous("bundled-stage.usda")
    assert layer.ImportFromString(stage_text)
    bundled_stage = Usd.Stage.Open(layer)
    assert bundled_stage is not None

    bundled_material = UsdShade.Material(
        bundled_stage.GetPrimAtPath("/World/Looks/Painted"),
    )
    surface = bundled_material.GetSurfaceOutput()
    sources, _ = surface.GetConnectedSources()
    preview_shader = UsdShade.Shader(sources[0].source.GetPrim())
    diffuse_sources, _ = preview_shader.GetInput("diffuseColor").GetConnectedSources()
    reader = UsdShade.Shader(diffuse_sources[0].source.GetPrim())

    assert preview_shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert reader.GetIdAttr().Get() == "UsdUVTexture"
    assert reader.GetInput("file").Get() == Sdf.AssetPath(
        "textures/painted_albedo.png",
    )


def test_bundle_stage_logs_preview_and_texture_fallback_updates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    texture_dir = asset_root / "textures"
    texture_dir.mkdir(parents=True)
    texture_path = texture_dir / "painted_albedo.png"
    Image.new("RGB", (1, 1), (8, 9, 10)).save(texture_path)

    stage = Usd.Stage.CreateNew(str(asset_root / "scene.usda"))
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Shader")
    shader.GetPrim().CreateAttribute("inputs:file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/painted_albedo.png"),
    )
    stage.GetRootLayer().Save()

    monkeypatch.setattr(
        render_remote,
        "add_ovrtx_preview_fallbacks_to_stage_file",
        lambda path: 1,
    )
    monkeypatch.setattr(
        render_remote,
        "_add_texture_file_fallbacks_for_remote_export",
        lambda *args, **kwargs: (2, 3),
    )
    monkeypatch.setattr(
        render_remote,
        "_prefer_preview_surface_for_remote_export",
        lambda path: 4,
    )

    zip_path, bundled = _bundle_stage_with_local_assets(
        stage,
        tmp_path / "bundle",
        base_dir=asset_root,
        add_preview_fallbacks=True,
    )

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        assert "textures/painted_albedo.png" in zf.namelist()


def test_bundle_stage_preserves_connected_mdl_texture_material(
    tmp_path: Path,
) -> None:
    """Self-contained OVRTX bundles should keep richer MDL texture graphs."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    asset_root = tmp_path / "asset"
    texture_dir = asset_root / "textures"
    texture_dir.mkdir(parents=True)
    Image.new("RGB", (2, 2), (64, 32, 16)).save(texture_dir / "painted_albedo.png")
    mdl_dir = asset_root / "materials" / "OmniPBR"
    mdl_dir.mkdir(parents=True)
    (mdl_dir / "OmniPBR.mdl").write_text("mdl 1.7;\n", encoding="utf-8")

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Painted/Shader")
    shader.CreateIdAttr("mdl:OmniPBR")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./materials/OmniPBR/OmniPBR.mdl"))
    shader.GetPrim().CreateAttribute(
        "inputs:diffuse_texture",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./textures/painted_albedo.png"))
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token),
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(stage, tmp_path / "bundle")

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        assert "textures/painted_albedo.png" in zf.namelist()
        assert "mdl_materials/OmniPBR/OmniPBR.mdl" in zf.namelist()
        stage_text = zf.read("stage.usda").decode("utf-8")

    assert "@textures/painted_albedo.png@" in stage_text
    assert "@mdl_materials/OmniPBR/OmniPBR.mdl@" in stage_text
    assert "outputs:mdl:surface.connect" in stage_text
    assert "OVRTXPreviewSurface" not in stage_text
    assert "outputs:surface.connect" not in stage_text


def test_bundle_stage_preserves_nested_connected_mdl_texture_material(
    tmp_path: Path,
) -> None:
    """Nested MDL texture graphs should still be treated as texture-capable."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    asset_root = tmp_path / "asset"
    texture_dir = asset_root / "textures"
    texture_dir.mkdir(parents=True)
    Image.new("RGB", (2, 2), (64, 32, 16)).save(texture_dir / "painted_albedo.png")
    mdl_dir = asset_root / "materials" / "OmniPBR"
    mdl_dir.mkdir(parents=True)
    (mdl_dir / "OmniPBR.mdl").write_text("mdl 1.7;\n", encoding="utf-8")

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    surface = UsdShade.Shader.Define(stage, "/World/Looks/Painted/Surface")
    surface.CreateIdAttr("mdl:OmniPBR")
    surface.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./materials/OmniPBR/OmniPBR.mdl"))
    surface_input = surface.CreateInput("diffuse_color", Sdf.ValueTypeNames.Color3f)

    texture_node = UsdShade.Shader.Define(stage, "/World/Looks/Painted/TextureNode")
    texture_node.CreateIdAttr("mdl:TextureNode")
    texture_node.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/painted_albedo.png")
    )
    surface_input.ConnectToSource(
        texture_node.CreateOutput("out", Sdf.ValueTypeNames.Color3f),
    )
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        surface.CreateOutput("out", Sdf.ValueTypeNames.Token),
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(stage, tmp_path / "bundle")

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        assert "textures/painted_albedo.png" in zf.namelist()
        assert "mdl_materials/OmniPBR/OmniPBR.mdl" in zf.namelist()
        stage_text = zf.read("stage.usda").decode("utf-8")

    assert "@textures/painted_albedo.png@" in stage_text
    assert "@mdl_materials/OmniPBR/OmniPBR.mdl@" in stage_text
    assert "outputs:mdl:surface.connect" in stage_text
    assert "OVRTXPreviewSurface" not in stage_text
    assert "outputs:surface.connect" not in stage_text


def test_remote_stage_export_rejects_unflattened_local_composition(
    tmp_path: Path,
) -> None:
    from pxr import Usd

    reference_path = tmp_path / "geometry.usda"
    reference_stage = Usd.Stage.CreateNew(str(reference_path))
    reference_stage.DefinePrim("/ReferencedRoot", "Xform")
    reference_stage.Save()

    root_path = tmp_path / "root.usda"
    root_stage = Usd.Stage.CreateNew(str(root_path))
    root_stage.DefinePrim("/World/Reference", "Xform").GetReferences().AddReference(
        "./geometry.usda",
        "/ReferencedRoot",
    )
    root_stage.Save()

    with pytest.raises(RuntimeError, match="requires a flattened stage"):
        export_stage_to_s3(
            root_stage,
            use_data_uri=True,
            bundle_mdl_assets=False,
        )


def test_export_stage_to_s3_preview_surface_target_adds_ovrtx_preview_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Gold")
    material.GetPrim().CreateAttribute(
        "inputs:base_color",
        Sdf.ValueTypeNames.Color3f,
    ).Set((1.0, 0.766, 0.336))

    shader = UsdShade.Shader.Define(
        stage,
        "/World/Looks/Gold/open_pbr_surface_surfaceshader",
    )
    shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
    material.CreateSurfaceOutput("mtlx").ConnectToSource(
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token),
    )

    captured: dict[str, object] = {}

    def fake_export_stage_and_get_url(**kwargs: object) -> tuple[str, str | None]:
        exported_stage = Usd.Stage.Open(str(kwargs["stage_path"]))
        assert exported_stage is not None
        exported_material = UsdShade.Material(
            exported_stage.GetPrimAtPath("/World/Looks/Gold"),
        )
        surface = exported_material.GetSurfaceOutput()
        sources, _ = surface.GetConnectedSources()
        preview_shader = UsdShade.Shader(sources[0].source.GetPrim())
        captured["shader_id"] = preview_shader.GetIdAttr().Get()
        captured["mtlx_connected"] = exported_material.GetSurfaceOutput(
            "mtlx",
        ).HasConnectedSource()
        return "data:model/vnd.usd;base64,AA==", None

    monkeypatch.setattr(
        render_remote,
        "_export_stage_and_get_url",
        fake_export_stage_and_get_url,
    )

    asset_url, s3_uri = export_stage_to_s3(
        stage,
        use_data_uri=True,
        material_target="preview_surface",
    )

    assert asset_url == "data:model/vnd.usd;base64,AA=="
    assert s3_uri is None
    assert captured == {"shader_id": "UsdPreviewSurface", "mtlx_connected": False}


def test_export_stage_to_s3_auto_preserves_openpbr_without_preview_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Gold")
    material.GetPrim().CreateAttribute(
        "inputs:base_color",
        Sdf.ValueTypeNames.Color3f,
    ).Set((1.0, 0.766, 0.336))

    shader = UsdShade.Shader.Define(
        stage,
        "/World/Looks/Gold/open_pbr_surface_surfaceshader",
    )
    shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
    material.CreateSurfaceOutput("mtlx").ConnectToSource(
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token),
    )

    captured: dict[str, object] = {}

    def fake_export_stage_and_get_url(**kwargs: object) -> tuple[str, str | None]:
        exported_stage = Usd.Stage.Open(str(kwargs["stage_path"]))
        assert exported_stage is not None
        exported_material = UsdShade.Material(
            exported_stage.GetPrimAtPath("/World/Looks/Gold"),
        )
        surface = exported_material.GetSurfaceOutput()
        captured["universal_connected"] = bool(
            surface and surface.HasConnectedSource(),
        )
        captured["mtlx_connected"] = exported_material.GetSurfaceOutput(
            "mtlx",
        ).HasConnectedSource()
        captured["has_preview_child"] = exported_stage.GetPrimAtPath(
            "/World/Looks/Gold/OVRTXPreviewSurface",
        ).IsValid()
        return "data:model/vnd.usd;base64,AA==", None

    monkeypatch.setattr(
        render_remote,
        "_export_stage_and_get_url",
        fake_export_stage_and_get_url,
    )

    asset_url, s3_uri = export_stage_to_s3(stage, use_data_uri=True)

    assert asset_url == "data:model/vnd.usd;base64,AA=="
    assert s3_uri is None
    assert captured == {
        "universal_connected": False,
        "mtlx_connected": True,
        "has_preview_child": False,
    }


def test_export_stage_to_s3_openpbr_material_target_preserves_native_openpbr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Gold")
    shader = UsdShade.Shader.Define(
        stage,
        "/World/Looks/Gold/open_pbr_surface_surfaceshader",
    )
    shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
    material.CreateSurfaceOutput("mtlx").ConnectToSource(
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token),
    )

    captured: dict[str, object] = {}

    def fake_export_stage_and_get_url(**kwargs: object) -> tuple[str, str | None]:
        exported_stage = Usd.Stage.Open(str(kwargs["stage_path"]))
        assert exported_stage is not None
        exported_material = UsdShade.Material(
            exported_stage.GetPrimAtPath("/World/Looks/Gold"),
        )
        surface = exported_material.GetSurfaceOutput()
        captured["universal_connected"] = bool(
            surface and surface.HasConnectedSource(),
        )
        captured["mtlx_connected"] = exported_material.GetSurfaceOutput(
            "mtlx",
        ).HasConnectedSource()
        captured["has_preview_child"] = exported_stage.GetPrimAtPath(
            "/World/Looks/Gold/OVRTXPreviewSurface",
        ).IsValid()
        return "data:model/vnd.usd;base64,AA==", None

    monkeypatch.setattr(
        render_remote,
        "_export_stage_and_get_url",
        fake_export_stage_and_get_url,
    )

    asset_url, s3_uri = export_stage_to_s3(
        stage,
        use_data_uri=True,
        material_target="openpbr_materialx",
    )

    assert asset_url == "data:model/vnd.usd;base64,AA=="
    assert s3_uri is None
    assert captured == {
        "universal_connected": False,
        "mtlx_connected": True,
        "has_preview_child": False,
    }


def test_export_stage_to_s3_strips_mdl_when_preview_surface_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Gold")
    preview_shader = UsdShade.Shader.Define(stage, "/World/Looks/Gold/Preview")
    preview_shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token),
    )
    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/Gold/Mdl")
    mdl_shader.CreateIdAttr("mdl:OmniSurface")
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        mdl_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token),
    )

    captured: dict[str, object] = {}

    def fake_export_stage_and_get_url(**kwargs: object) -> tuple[str, str | None]:
        exported_stage = Usd.Stage.Open(str(kwargs["stage_path"]))
        assert exported_stage is not None
        exported_material_prim = exported_stage.GetPrimAtPath("/World/Looks/Gold")
        captured["has_mdl_output"] = any(
            prop.GetName().startswith("outputs:mdl:")
            for prop in exported_material_prim.GetProperties()
        )
        return "data:model/vnd.usd;base64,AA==", None

    monkeypatch.setattr(
        render_remote,
        "_export_stage_and_get_url",
        fake_export_stage_and_get_url,
    )

    asset_url, s3_uri = export_stage_to_s3(
        stage,
        use_data_uri=True,
        material_target="preview_surface",
    )

    assert asset_url == "data:model/vnd.usd;base64,AA=="
    assert s3_uri is None
    assert captured == {"has_mdl_output": False}


def test_export_stage_to_s3_preserves_mdl_when_preview_fallbacks_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Gold")
    preview_shader = UsdShade.Shader.Define(stage, "/World/Looks/Gold/Preview")
    preview_shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token),
    )
    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/Gold/Mdl")
    mdl_shader.CreateIdAttr("mdl:OmniSurface")
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        mdl_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token),
    )

    captured: dict[str, object] = {}

    def fake_export_stage_and_get_url(**kwargs: object) -> tuple[str, str | None]:
        exported_stage = Usd.Stage.Open(str(kwargs["stage_path"]))
        assert exported_stage is not None
        exported_material_prim = exported_stage.GetPrimAtPath("/World/Looks/Gold")
        captured["has_mdl_output"] = any(
            prop.GetName().startswith("outputs:mdl:")
            for prop in exported_material_prim.GetProperties()
        )
        return "data:model/vnd.usd;base64,AA==", None

    monkeypatch.setattr(
        render_remote,
        "_export_stage_and_get_url",
        fake_export_stage_and_get_url,
    )

    asset_url, s3_uri = export_stage_to_s3(
        stage,
        use_data_uri=True,
        add_preview_fallbacks=False,
    )

    assert asset_url == "data:model/vnd.usd;base64,AA=="
    assert s3_uri is None
    assert captured == {"has_mdl_output": True}


def test_bundle_stage_prefers_preview_surface_over_mdl_output_when_fallbacks_enabled(
    tmp_path: Path,
) -> None:
    """Remote bundles should use preview output when both preview and MDL exist."""
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    (asset_root / "materials" / "OmniPBR").mkdir(parents=True)
    (asset_root / "materials" / "OmniPBR" / "OmniPBR.mdl").write_text(
        "mdl 1.7;\n",
        encoding="utf-8",
    )

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    material = UsdShade.Material.Define(stage, "/World/Looks/Mat")

    preview_shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Preview")
    preview_shader.CreateIdAttr("UsdPreviewSurface")
    preview_output = preview_shader.CreateOutput(
        "surface",
        Sdf.ValueTypeNames.Token,
    )
    material.CreateSurfaceOutput().ConnectToSource(preview_output)

    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Mdl")
    mdl_shader.CreateIdAttr("mdl:OmniPBR")
    mdl_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./materials/OmniPBR/OmniPBR.mdl"))
    mdl_output = mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_output)
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(
        stage,
        tmp_path / "bundle",
        add_preview_fallbacks=True,
    )

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        stage_text = zf.read("stage.usda").decode("utf-8")

    assert "outputs:surface.connect" in stage_text
    assert "outputs:mdl:surface" not in stage_text
    assert "@mdl_materials/OmniPBR/OmniPBR.mdl@" in stage_text


def test_bundle_stage_preserves_mdl_when_preview_fallbacks_disabled(
    tmp_path: Path,
) -> None:
    """OpenPBR/MDL-capable endpoints should receive native material outputs."""
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    (asset_root / "materials" / "OmniPBR").mkdir(parents=True)
    (asset_root / "materials" / "OmniPBR" / "OmniPBR.mdl").write_text(
        "mdl 1.7;\n",
        encoding="utf-8",
    )

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    material = UsdShade.Material.Define(stage, "/World/Looks/Mat")

    preview_shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Preview")
    preview_shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token),
    )

    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Mdl")
    mdl_shader.CreateIdAttr("mdl:OmniPBR")
    mdl_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./materials/OmniPBR/OmniPBR.mdl"))
    mdl_output = mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_output)
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(
        stage,
        tmp_path / "bundle",
        add_preview_fallbacks=False,
    )

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        stage_text = zf.read("stage.usda").decode("utf-8")

    assert "outputs:surface.connect" in stage_text
    assert "outputs:mdl:surface" in stage_text
    assert "@mdl_materials/OmniPBR/OmniPBR.mdl@" in stage_text


def test_bundle_stage_keeps_mdl_only_surface_output(tmp_path: Path) -> None:
    """MDL-only materials should not lose their only material output."""
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    (asset_root / "materials" / "OmniPBR").mkdir(parents=True)
    (asset_root / "materials" / "OmniPBR" / "OmniPBR.mdl").write_text(
        "mdl 1.7;\n",
        encoding="utf-8",
    )

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    material = UsdShade.Material.Define(stage, "/World/Looks/Mat")
    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Mdl")
    mdl_shader.CreateIdAttr("mdl:OmniPBR")
    mdl_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./materials/OmniPBR/OmniPBR.mdl"))
    mdl_output = mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_output)
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(stage, tmp_path / "bundle")

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        stage_text = zf.read("stage.usda").decode("utf-8")

    assert "outputs:mdl:surface" in stage_text
    assert "@mdl_materials/OmniPBR/OmniPBR.mdl@" in stage_text


def test_bundle_stage_keeps_mdl_output_when_universal_surface_is_not_preview(
    tmp_path: Path,
) -> None:
    """Only UsdPreviewSurface fallback materials should have MDL outputs stripped."""
    from pxr import Sdf, Usd, UsdShade

    asset_root = tmp_path / "asset"
    (asset_root / "materials" / "OmniPBR").mkdir(parents=True)
    (asset_root / "materials" / "OmniPBR" / "OmniPBR.mdl").write_text(
        "mdl 1.7;\n",
        encoding="utf-8",
    )

    stage_path = asset_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    material = UsdShade.Material.Define(stage, "/World/Looks/Mat")

    materialx_shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/MaterialX")
    materialx_shader.CreateIdAttr("ND_standard_surface_surfaceshader")
    materialx_output = materialx_shader.CreateOutput(
        "surface",
        Sdf.ValueTypeNames.Token,
    )
    material.CreateSurfaceOutput().ConnectToSource(materialx_output)

    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Mdl")
    mdl_shader.CreateIdAttr("mdl:OmniPBR")
    mdl_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./materials/OmniPBR/OmniPBR.mdl"))
    mdl_output = mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_output)
    stage.GetRootLayer().Save()

    zip_path, bundled = _bundle_stage_with_local_assets(stage, tmp_path / "bundle")

    assert bundled is True
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        stage_text = zf.read("stage.usda").decode("utf-8")

    assert "outputs:surface.connect" in stage_text
    assert "outputs:mdl:surface" in stage_text
    assert "@mdl_materials/OmniPBR/OmniPBR.mdl@" in stage_text


def test_save_render_results_preserves_instance_segmentation_ids(
    tmp_path: Path,
) -> None:
    result = {
        "sensors": {
            "instance_id_segmentation": {
                0: np.array([[1, 256]], dtype=np.uint32),
            }
        }
    }

    stats = save_render_results(
        result,
        tmp_path,
        file_name="seg",
        image_width=2,
        image_height=1,
        save_npy=True,
    )

    assert stats["success_count"] == 1
    saved = np.load(tmp_path / "seg_f0000_instance_id_segmentation.npy")
    assert saved.dtype == np.uint32
    assert int(saved.max()) == 256
    assert (tmp_path / "seg_f0000_instance_id_segmentation.png").exists()


class TestIsV2Response:
    """Tests for V2 response detection."""

    def test_detects_v2_response(self):
        result = {
            "total_cameras": 1,
            "total_frames": 1,
            "rendered_data": {"Camera": {}},
        }
        assert _is_v2_response(result) is True

    def test_rejects_v1_response(self):
        result = {
            "images": {"0": {}},
            "status": "success",
        }
        assert _is_v2_response(result) is False

    def test_rejects_empty_dict(self):
        assert _is_v2_response({}) is False

    def test_rejects_partial_v2(self):
        # Has rendered_data but not total_cameras
        assert _is_v2_response({"rendered_data": {}}) is False


class TestConvertV2Sensor:
    """Tests for V2 sensor data conversion."""

    def test_converts_uint8_rgb(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[0, 0] = [255, 0, 0, 255]
        sensor_obj = {
            "type": "array",
            "data": base64.b64encode(arr.tobytes()).decode(),
            "shape": [4, 4, 4],
            "dtype": "uint8",
        }
        result = _convert_v2_sensor(sensor_obj)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 4, 4)
        assert result[0, 0, 0] == 255

    def test_returns_string_when_no_shape(self):
        sensor_obj = {"data": "abc123"}
        result = _convert_v2_sensor(sensor_obj)
        assert result == "abc123"

    def test_returns_empty_string_when_no_data(self):
        sensor_obj = {"shape": [4, 4]}
        result = _convert_v2_sensor(sensor_obj)
        assert result == ""


class TestConvertV2ToV1:
    """Tests for V2→V1 full response conversion."""

    def _make_v2_response(
        self, width: int = 4, height: int = 4, n_cameras: int = 1
    ) -> dict:
        """Create a minimal V2 response with an RGB image."""
        rendered_data = {}
        for i in range(n_cameras):
            arr = np.full((height, width, 4), 128, dtype=np.uint8)
            cam_name = f"Camera{i}"
            rendered_data[cam_name] = {
                "0": {
                    "rgb": {
                        "type": "array",
                        "data": base64.b64encode(arr.tobytes()).decode(),
                        "shape": [height, width, 4],
                        "dtype": "uint8",
                    }
                }
            }
        return {
            "total_cameras": n_cameras,
            "total_frames": 1,
            "rendered_data": rendered_data,
        }

    def test_v1_has_status_success(self):
        v2 = self._make_v2_response()
        v1 = _convert_v2_to_v1(v2)
        assert v1["status"] == RenderingStatus.success

    def test_v1_has_images_key(self):
        v2 = self._make_v2_response()
        v1 = _convert_v2_to_v1(v2)
        assert "images" in v1
        assert "0" in v1["images"]

    def test_v1_frame_camera_nesting(self):
        """V1 format nests frame→camera (opposite of V2 camera→frame)."""
        v2 = self._make_v2_response()
        v1 = _convert_v2_to_v1(v2)
        frame_data = v1["images"]["0"]
        assert "Camera0" in frame_data

    def test_v1_rgb_converted_to_base64_png(self):
        """V2 raw array data should become a base64 PNG in V1 'images' key."""
        v2 = self._make_v2_response()
        v1 = _convert_v2_to_v1(v2)
        camera_data = v1["images"]["0"]["Camera0"]
        assert "images" in camera_data  # rgb → images
        # Should be valid base64 PNG
        png_bytes = base64.b64decode(camera_data["images"])
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (4, 4)

    def test_multi_camera_response(self):
        v2 = self._make_v2_response(n_cameras=3)
        v1 = _convert_v2_to_v1(v2)
        frame_data = v1["images"]["0"]
        assert len(frame_data) == 3
        for i in range(3):
            assert f"Camera{i}" in frame_data

    def test_sensor_name_mapping(self):
        """V2 sensor names should be mapped to V1 equivalents."""
        arr = np.zeros((4, 4), dtype=np.float32)
        v2 = {
            "total_cameras": 1,
            "total_frames": 1,
            "rendered_data": {
                "Camera": {
                    "0": {
                        "rgb": {
                            "type": "array",
                            "data": base64.b64encode(
                                np.zeros((4, 4, 4), dtype=np.uint8).tobytes()
                            ).decode(),
                            "shape": [4, 4, 4],
                            "dtype": "uint8",
                        },
                        "distance_to_image_plane": {
                            "type": "array",
                            "data": base64.b64encode(arr.tobytes()).decode(),
                            "shape": [4, 4],
                            "dtype": "float32",
                        },
                        "instance_segmentation": {
                            "type": "array",
                            "data": base64.b64encode(
                                np.zeros((4, 4), dtype=np.uint32).tobytes()
                            ).decode(),
                            "shape": [4, 4],
                            "dtype": "uint32",
                        },
                    }
                }
            },
        }
        v1 = _convert_v2_to_v1(v2)
        camera_data = v1["images"]["0"]["Camera"]
        assert "images" in camera_data  # rgb → images
        assert "linear_depth" in camera_data  # distance_to_image_plane → linear_depth
        assert (
            "instance_id_segmentation" in camera_data
        )  # instance_segmentation → instance_id_segmentation

    def test_empty_rendered_data(self):
        v2 = {"total_cameras": 0, "total_frames": 0, "rendered_data": {}}
        v1 = _convert_v2_to_v1(v2)
        assert v1["status"] == RenderingStatus.success
        assert v1["images"] == {}
