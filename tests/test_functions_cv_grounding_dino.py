# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from world_understanding.functions.cv import grounding_dino as gd
from world_understanding.tools.cv import grounding_dino as gd_tool


class _Response:
    def __init__(
        self,
        *,
        status_code: int = 200,
        content: bytes = b"",
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self.content = content
        self._json_data = json_data or {}
        self.headers = headers or {}
        self.text = text
        self.raised = False

    def json(self) -> dict[str, Any]:
        return self._json_data

    def raise_for_status(self) -> None:
        self.raised = True


def _response_zip(
    payload: dict[str, Any], *, filename: str = "result.response"
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(filename, json.dumps(payload))
    return buffer.getvalue()


def _sample_payload() -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": {
                        "boundingBoxes": [
                            {
                                "phrase": "cup",
                                "bboxes": [[1, 2, 3, 4]],
                                "confidence": [0.91],
                            }
                        ]
                    }
                }
            }
        ]
    }


def _write_image(path: Path) -> None:
    Image.new("RGB", (3, 2), color=(10, 20, 30)).save(path)


def test_upload_asset_posts_metadata_and_puts_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_id = uuid.uuid4()
    calls: list[tuple[str, Any]] = []

    def fake_post(
        url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: int
    ):
        calls.append(("post", url, headers, json, timeout))
        return _Response(
            json_data={"uploadUrl": "https://upload.example", "assetId": str(asset_id)}
        )

    def fake_put(url: str, *, data: bytes, headers: dict[str, str], timeout: int):
        calls.append(("put", url, data, headers, timeout))
        return _Response()

    monkeypatch.setattr(gd.requests, "post", fake_post)
    monkeypatch.setattr(gd.requests, "put", fake_put)

    assert gd._upload_asset(b"image", "Input Image", "image/png", "secret") == asset_id
    assert calls[0][0] == "post"
    assert calls[0][2]["Authorization"] == "Bearer secret"
    assert calls[0][3] == {"contentType": "image/png", "description": "Input Image"}
    assert calls[1] == (
        "put",
        "https://upload.example",
        b"image",
        {
            "x-amz-meta-nvcf-asset-description": "Input Image",
            "content-type": "image/png",
        },
        gd.UPLOAD_ASSET_TIMEOUT,
    )


def test_upload_asset_requires_api_key() -> None:
    with pytest.raises(RuntimeError, match="NVIDIA API key is required"):
        gd._upload_asset(b"image", "Input Image", "image/png", "")


def test_run_grounding_dino_synchronous_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "image.unknown"
    image_path.write_bytes(b"not really an image")
    asset_id = uuid.uuid4()
    seen_payloads: list[dict[str, Any]] = []

    monkeypatch.setattr(gd, "_upload_asset", lambda *args, **kwargs: asset_id)

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any]):
        seen_payloads.append(json)
        assert url == gd.NVAI_URL
        assert headers["NVCF-INPUT-ASSET-REFERENCES"] == str(asset_id)
        return _Response(content=_response_zip(_sample_payload()))

    monkeypatch.setattr(gd.requests, "post", fake_post)

    assert gd._run_grounding_dino(str(image_path), "cup", api_key="secret") == [
        {"phrase": "cup", "bboxes": [[1, 2, 3, 4]], "confidence": [0.91]}
    ]
    media_url = seen_payloads[0]["messages"][0]["content"][1]["media_url"]["url"]
    assert media_url == f"data:image/jpeg;asset_id,{asset_id}"


def test_run_grounding_dino_polling_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image-bytes")
    monkeypatch.setattr(gd, "_upload_asset", lambda *args, **kwargs: uuid.uuid4())
    monkeypatch.setattr(gd.time, "sleep", lambda _delay: None)

    monkeypatch.setattr(
        gd.requests,
        "post",
        lambda *args, **kwargs: _Response(
            status_code=202, headers={"NVCF-REQID": "abc"}
        ),
    )
    poll_responses = iter(
        [
            _Response(status_code=202),
            _Response(status_code=200, content=_response_zip(_sample_payload())),
        ]
    )
    monkeypatch.setattr(
        gd.requests, "get", lambda *args, **kwargs: next(poll_responses)
    )

    detections = gd._run_grounding_dino(str(image_path), "cup", api_key="secret")

    assert detections[0]["phrase"] == "cup"


@pytest.mark.parametrize(
    ("post_response", "poll_responses", "expected_error"),
    [
        (
            _Response(status_code=500, text="bad"),
            [],
            RuntimeError,
        ),
        (
            _Response(status_code=202, headers={"NVCF-REQID": "abc"}),
            [_Response(status_code=500)],
            RuntimeError,
        ),
        (
            _Response(status_code=202, headers={"NVCF-REQID": "abc"}),
            [_Response(status_code=202)] * gd.MAX_RETRIES,
            TimeoutError,
        ),
    ],
)
def test_run_grounding_dino_response_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    post_response: _Response,
    poll_responses: list[_Response],
    expected_error: type[Exception],
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image-bytes")
    monkeypatch.setattr(gd, "_upload_asset", lambda *args, **kwargs: uuid.uuid4())
    monkeypatch.setattr(gd.requests, "post", lambda *args, **kwargs: post_response)
    monkeypatch.setattr(gd.time, "sleep", lambda _delay: None)
    responses = iter(poll_responses)
    monkeypatch.setattr(gd.requests, "get", lambda *args, **kwargs: next(responses))

    with pytest.raises(expected_error):
        gd._run_grounding_dino(str(image_path), "cup", api_key="secret")


def test_run_grounding_dino_validates_inputs(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="NVIDIA API key is required"):
        gd._run_grounding_dino(str(tmp_path / "missing.png"), "cup", api_key="")

    with pytest.raises(FileNotFoundError):
        gd._run_grounding_dino(str(tmp_path / "missing.png"), "cup", api_key="secret")


@pytest.mark.parametrize(
    ("zip_bytes", "expected_error"),
    [
        (
            _response_zip(_sample_payload(), filename="not_response.txt"),
            FileNotFoundError,
        ),
        (_response_zip({"choices": []}), ValueError),
    ],
    ids=["missing-result-response", "empty-choices"],
)
def test_run_grounding_dino_result_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    zip_bytes: bytes,
    expected_error: type[Exception],
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image-bytes")
    monkeypatch.setattr(gd, "_upload_asset", lambda *args, **kwargs: uuid.uuid4())
    monkeypatch.setattr(
        gd.requests, "post", lambda *args, **kwargs: _Response(content=zip_bytes)
    )

    with pytest.raises(expected_error):
        gd._run_grounding_dino(str(image_path), "cup", api_key="secret")


def test_detect_objects_accepts_path_pil_and_numpy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detections = [
        {
            "phrase": "cup",
            "bboxes": [[1, 2, 3, 4], [4, 3, 2, 1]],
            "confidence": [0.9, 0.8],
        }
    ]
    seen_paths: list[str] = []

    def fake_run(image_path: str, prompt: str, threshold: float, *, api_key: str):
        seen_paths.append(image_path)
        assert prompt == "cup"
        assert threshold == 0.5
        assert api_key == "env-key"
        return detections

    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
    monkeypatch.setattr(gd, "_run_grounding_dino", fake_run)

    image_path = tmp_path / "image.png"
    _write_image(image_path)
    assert gd.detect_objects_with_grounding_dino(image_path, "cup", threshold=0.5) == {
        "detections": detections,
        "total_detections": 2,
        "image_size": (3, 2),
    }

    pil_result = gd.detect_objects_with_grounding_dino(
        Image.new("RGB", (4, 5)), "cup", threshold=0.5
    )
    np_result = gd.detect_objects_with_grounding_dino(
        np.zeros((6, 7, 3), dtype=np.uint8), "cup", threshold=0.5
    )

    assert pil_result["image_size"] == (4, 5)
    assert np_result["image_size"] == (7, 6)
    assert not Path(seen_paths[-2]).exists()
    assert not Path(seen_paths[-1]).exists()


def test_detect_objects_rejects_unsupported_image_type() -> None:
    with pytest.raises(ValueError, match="Unsupported image type"):
        gd.detect_objects_with_grounding_dino(object(), "cup")


class _Console:
    def __init__(self) -> None:
        self.printed: list[Any] = []

    def print(self, value: Any) -> None:
        self.printed.append(value)


def test_grounding_dino_tool_and_display(monkeypatch: pytest.MonkeyPatch) -> None:
    result = {
        "detections": [
            {"phrase": "cup", "bboxes": [[1, 2, 3, 4]], "confidence": [0.91]},
        ],
        "total_detections": 1,
        "image_size": (10, 20),
    }

    monkeypatch.setattr(
        gd_tool, "detect_objects_with_grounding_dino", lambda **kwargs: result
    )
    output = gd_tool.grounding_dino_tool(
        gd_tool.GroundingDinoInput(
            image_path="image.png",
            prompt="cup",
            threshold=0.4,
            api_key="secret",
        )
    )

    assert output.total_detections == 1
    assert output.detections[0].phrase == "cup"

    console = _Console()
    gd_tool._display_detection_results(output.model_dump(), console, indent="  ")
    assert any("Total detections: 1" in str(item) for item in console.printed)
    assert any(item.__class__.__name__ == "Table" for item in console.printed)


def test_grounding_dino_tool_uses_env_key_and_reports_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NVIDIA API key required"):
        gd_tool.grounding_dino_tool(
            gd_tool.GroundingDinoInput(image_path="image.png", prompt="cup")
        )

    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")

    def fail_detection(**kwargs):
        assert kwargs["api_key"] == "env-key"
        raise RuntimeError("boom")

    monkeypatch.setattr(gd_tool, "detect_objects_with_grounding_dino", fail_detection)
    with pytest.raises(RuntimeError, match="boom"):
        gd_tool.grounding_dino_tool(
            gd_tool.GroundingDinoInput(image_path="image.png", prompt="cup")
        )


def test_display_detection_results_handles_empty_results() -> None:
    console = _Console()

    gd_tool._display_detection_results(
        {"detections": [], "total_detections": 0, "image_width": 1, "image_height": 1},
        console,
    )

    assert any("No objects detected" in str(item) for item in console.printed)
