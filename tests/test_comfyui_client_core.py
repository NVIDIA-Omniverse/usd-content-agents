# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for the ComfyUI HTTP/WebSocket client."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from world_understanding.utils import comfyui_client


def _png_bytes(color: str = "red") -> bytes:
    image = Image.new("RGB", (1, 1), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: dict[str, Any] | None = None,
        text: str = "error",
        content: bytes | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.content = content or b""

    def json(self) -> dict[str, Any]:
        return self._payload


def test_comfyui_init_http_methods_and_websocket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COMFYUI_URL", raising=False)
    with pytest.raises(ValueError, match="server URL"):
        comfyui_client.ComfyUIClient()

    monkeypatch.setenv("COMFYUI_URL", "http://env-server/")
    env_client = comfyui_client.ComfyUIClient()
    assert env_client.server_url == "http://env-server"

    client = comfyui_client.ComfyUIClient("https://server/")
    image_path = tmp_path / "input.unknown"
    image_path.write_bytes(_png_bytes())

    posts: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        posts.append({"url": url, **kwargs})
        if url.endswith("/upload/image"):
            return FakeResponse(payload={"name": "uploaded.png"})
        if url.endswith("/prompt"):
            return FakeResponse(payload={"prompt_id": "prompt-1"})
        raise AssertionError(url)

    monkeypatch.setattr(comfyui_client.requests, "post", fake_post)
    assert client.upload_image(image_path) == ("uploaded.png", "", "input")
    assert posts[0]["files"]["image"][2] == "image/png"
    assert client.queue_prompt({"1": {"class_type": "SaveImage"}}) == "prompt-1"
    assert posts[1]["json"]["client_id"] == client.client_id

    gets: list[dict[str, Any]] = []

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        gets.append({"url": url, **kwargs})
        if "/history/" in url:
            return FakeResponse(payload={"prompt-1": {"outputs": {}}})
        if url.endswith("/view"):
            return FakeResponse(content=_png_bytes("blue"))
        raise AssertionError(url)

    monkeypatch.setattr(comfyui_client.requests, "get", fake_get)
    assert client.get_history("prompt-1") == {"outputs": {}}
    assert client.get_history("unknown") is None
    assert client.get_image("out.png", "sub", "output").startswith(b"\x89PNG")
    assert gets[-1]["params"] == {
        "filename": "out.png",
        "subfolder": "sub",
        "type": "output",
    }

    class FakeWebSocket:
        def __init__(self) -> None:
            self.connected_to: str | None = None

        def connect(self, url: str) -> None:
            self.connected_to = url

    fake_ws = FakeWebSocket()
    monkeypatch.setattr(comfyui_client.websocket, "WebSocket", lambda: fake_ws)
    assert client.connect_websocket() is fake_ws
    assert fake_ws.connected_to == f"wss://server/ws?clientId={client.client_id}"


def test_comfyui_http_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = comfyui_client.ComfyUIClient("http://server")
    image_path = tmp_path / "input.png"
    image_path.write_bytes(_png_bytes())

    monkeypatch.setattr(
        comfyui_client.requests,
        "post",
        lambda *_args, **_kwargs: FakeResponse(status_code=500, text="bad post"),
    )
    with pytest.raises(Exception, match="Failed to upload image"):
        client.upload_image(image_path)
    with pytest.raises(Exception, match="Failed to queue prompt"):
        client.queue_prompt({})

    monkeypatch.setattr(
        comfyui_client.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(status_code=500, text="bad get"),
    )
    with pytest.raises(Exception, match="Failed to get history"):
        client.get_history("prompt")
    with pytest.raises(Exception, match="Failed to get image"):
        client.get_image("image.png")


class ScriptedWebSocket:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = list(messages)
        self.timeouts: list[float] = []
        self.closed = False

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)

    def recv(self) -> Any:
        if not self.messages:
            return '{"type": "executing", "data": {"node": null}}'
        item = self.messages.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


class WorkflowClient(comfyui_client.ComfyUIClient):
    def __init__(
        self,
        ws: ScriptedWebSocket,
        histories: list[dict[str, Any] | None],
        image_bytes: bytes | None = None,
    ) -> None:
        super().__init__("http://server")
        self.ws = ws
        self.histories = list(histories)
        self.image_bytes = image_bytes or _png_bytes("green")
        self.uploads: list[str | Path] = []
        self.queued: list[dict[str, Any]] = []
        self.downloads: list[tuple[str, str, str]] = []

    def upload_image(self, image_path: str | Path) -> tuple[str, str, str]:
        self.uploads.append(image_path)
        return ("uploaded.png", "inputs", "input")

    def connect_websocket(self) -> ScriptedWebSocket:
        return self.ws

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        self.queued.append(workflow)
        return "prompt-1"

    def get_history(self, prompt_id: str) -> dict[str, Any] | None:
        return self.histories.pop(0) if self.histories else None

    def get_image(
        self, filename: str, subfolder: str = "", img_type: str = "output"
    ) -> bytes:
        self.downloads.append((filename, subfolder, img_type))
        return self.image_bytes


def test_execute_workflow_success_output_nodes_and_all_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(comfyui_client.time, "sleep", lambda _seconds: None)
    input_image = tmp_path / "input.png"
    input_image.write_bytes(_png_bytes())

    ws = ScriptedWebSocket(
        [
            b"\xff",
            "not json",
            '{"type": "executing", "data": {"node": "7"}}',
            comfyui_client.websocket.WebSocketTimeoutException(),
            comfyui_client.websocket.WebSocketException(),
            '{"type": "executing", "data": {"node": null}}',
        ]
    )
    history = {
        "outputs": {
            "save": {
                "images": [
                    {
                        "filename": "out.png",
                        "subfolder": "sub",
                        "type": "output",
                    }
                ]
            },
            "ignored": {},
        }
    }
    client = WorkflowClient(ws, [None, history])
    inputs: dict[str, Any] = {"source_image": str(input_image), "prompt": "hello"}
    result = client.execute_workflow({}, inputs, output_nodes=["save"])

    assert list(result) == ["save"]
    assert result["save"].size == (1, 1)
    assert inputs["source_image_uploaded"] == {
        "filename": "uploaded.png",
        "subfolder": "inputs",
        "type": "input",
    }
    assert client.downloads == [("out.png", "sub", "output")]
    assert ws.closed is True

    all_ws = ScriptedWebSocket(['{"type": "executing", "data": {"node": null}}'])
    all_client = WorkflowClient(all_ws, [history])
    all_result = all_client.execute_workflow({}, {}, output_nodes=None)
    assert list(all_result) == ["save"]


def test_execute_workflow_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comfyui_client.time, "sleep", lambda _seconds: None)
    error_ws = ScriptedWebSocket(
        ['{"type": "execution_error", "data": {"message": "boom"}}']
    )
    error_client = WorkflowClient(error_ws, [])
    with pytest.raises(Exception, match="Execution error"):
        error_client.execute_workflow({}, {}, timeout=1)
    assert error_ws.closed is True

    missing_outputs_ws = ScriptedWebSocket(
        ['{"type": "executing", "data": {"node": null}}']
    )
    missing_outputs_client = WorkflowClient(missing_outputs_ws, [None] * 30)
    with pytest.raises(Exception, match="Failed to get execution outputs"):
        missing_outputs_client.execute_workflow({}, {}, timeout=1)
    assert missing_outputs_ws.closed is True
