# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ComfyUI API client for executing workflows."""

import io
import json
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
import websocket
from PIL import Image

_WEBSOCKET_SCHEMES = {"http": "ws", "https": "wss"}


class ComfyUIClient:
    """Client for interacting with ComfyUI server."""

    def __init__(self, server_url: str | None = None):
        """Initialize the ComfyUI client.

        Args:
            server_url: The URL of the ComfyUI server (uses COMFYUI_URL env var if not provided)
        """
        self.server_url = (server_url or os.getenv("COMFYUI_URL", "")).rstrip("/")
        if not self.server_url:
            raise ValueError(
                "ComfyUI server URL must be provided or set in COMFYUI_URL environment variable"
            )
        self.client_id = str(uuid.uuid4())

    def upload_image(self, image_path: str | Path) -> tuple[str, str, str]:
        """Upload an image to the ComfyUI server.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (filename, subfolder, type) as returned by the server
        """
        image_path = Path(image_path)
        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"  # fallback

        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, mime_type)}
            data = {"type": "input", "overwrite": "true"}

            response = requests.post(
                f"{self.server_url}/upload/image", files=files, data=data
            )

            if response.status_code != 200:
                raise Exception(f"Failed to upload image: {response.text}")

            result = response.json()
            return (
                result["name"],
                result.get("subfolder", ""),
                result.get("type", "input"),
            )

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """Queue a workflow for execution.

        Args:
            workflow: The workflow dictionary

        Returns:
            The prompt ID
        """
        payload = {"prompt": workflow, "client_id": self.client_id}

        response = requests.post(f"{self.server_url}/prompt", json=payload)

        if response.status_code != 200:
            raise Exception(f"Failed to queue prompt: {response.text}")

        return response.json()["prompt_id"]

    def get_history(self, prompt_id: str) -> dict[str, Any] | None:
        """Get the execution history for a prompt.

        Args:
            prompt_id: The prompt ID

        Returns:
            The history data or None if not ready
        """
        response = requests.get(f"{self.server_url}/history/{prompt_id}")

        if response.status_code != 200:
            raise Exception(f"Failed to get history: {response.text}")

        history = response.json()
        return history.get(prompt_id)

    def get_image(
        self, filename: str, subfolder: str = "", img_type: str = "output"
    ) -> bytes:
        """Download an image from the server.

        Args:
            filename: The filename of the image
            subfolder: The subfolder (if any)
            img_type: The type of image ('output', 'input', etc.)

        Returns:
            The image data as bytes
        """
        params = {"filename": filename, "subfolder": subfolder, "type": img_type}

        response = requests.get(f"{self.server_url}/view", params=params)

        if response.status_code != 200:
            raise Exception(f"Failed to get image: {response.text}")

        return response.content

    def connect_websocket(self) -> websocket.WebSocket:
        """Connect to the ComfyUI WebSocket for real-time updates.

        Returns:
            Connected WebSocket instance
        """
        parsed = urlsplit(self.server_url)
        ws_url = urlunsplit(
            parsed._replace(scheme=_WEBSOCKET_SCHEMES.get(parsed.scheme, parsed.scheme))
        )
        ws = websocket.WebSocket()
        ws.connect(f"{ws_url}/ws?clientId={self.client_id}")
        return ws

    def execute_workflow(
        self,
        workflow: dict[str, Any],
        inputs: dict[str, Any],
        output_nodes: list[str] | None = None,
        timeout: int = 300,
    ) -> dict[str, Image.Image]:
        """Execute a workflow and return outputs from specified nodes.

        Args:
            workflow: The workflow dictionary
            inputs: Input parameters for the workflow
            output_nodes: List of node IDs to get outputs from
            timeout: Maximum time to wait for completion in seconds

        Returns:
            Dict mapping node_id to PIL Image objects
        """
        # Upload images if needed
        for key, value in list(inputs.items()):
            if key.endswith("_image") and isinstance(value, str | Path):
                # This is an image input, upload it
                filename, subfolder, img_type = self.upload_image(value)
                inputs[key + "_uploaded"] = {
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": img_type,
                }

        # Connect to WebSocket for real-time updates
        ws = self.connect_websocket()

        # Queue the prompt
        prompt_id = self.queue_prompt(workflow)

        # Monitor execution via WebSocket
        execution_complete = False
        start_time = time.time()

        try:
            while not execution_complete and (time.time() - start_time) < timeout:
                try:
                    ws.settimeout(1.0)
                    message = ws.recv()

                    if message:
                        # Handle binary messages
                        if isinstance(message, bytes):
                            try:
                                message = message.decode("utf-8")
                            except UnicodeDecodeError:
                                continue

                        # Parse JSON message
                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError:
                            continue

                        # Check for execution updates
                        if data.get("type") == "executing":
                            node = data["data"]["node"]
                            if node is None:
                                execution_complete = True

                        # Check for execution errors
                        elif data.get("type") == "execution_error":
                            error_data = data["data"]
                            raise Exception(f"Execution error: {error_data}")

                except websocket.WebSocketTimeoutException:
                    pass
                except websocket.WebSocketException:
                    continue

            # Check execution history to get outputs
            history = None
            for _ in range(30):  # Wait up to 30 seconds for history
                history = self.get_history(prompt_id)
                if history and "outputs" in history:
                    break
                time.sleep(1)

            if not history or "outputs" not in history:
                raise Exception("Failed to get execution outputs")

            # Process outputs
            outputs = history["outputs"]
            result_images = {}

            # Debug: print available output nodes
            # print(f"Available output nodes: {list(outputs.keys())}")

            # Get images from specified nodes
            if output_nodes:
                for node_id in output_nodes:
                    if node_id in outputs and "images" in outputs[node_id]:
                        img_info = outputs[node_id]["images"][0]
                        image_data = self.get_image(
                            img_info["filename"],
                            img_info.get("subfolder", ""),
                            img_info.get("type", "output"),
                        )
                        result_images[node_id] = Image.open(io.BytesIO(image_data))
            else:
                # Get all images from any SaveImage nodes
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        img_info = node_output["images"][0]
                        image_data = self.get_image(
                            img_info["filename"],
                            img_info.get("subfolder", ""),
                            img_info.get("type", "output"),
                        )
                        result_images[node_id] = Image.open(io.BytesIO(image_data))

            return result_images

        finally:
            ws.close()
