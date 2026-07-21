# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Generator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

TRANSIENT_STATUS_CODES = {502, 503, 504}
STATUS_POLL_TIMEOUT_SECONDS = 30
MAX_TRANSIENT_STATUS_ERRORS = 5
TRANSIENT_STATUS_RETRY_SECONDS = 5


@dataclass(frozen=True)
class SSEMessage:
    """Represents a parsed Server-Sent Event (SSE) message."""

    event: str
    data: str
    id: str | None = None
    retry: int | None = None

    def json(self) -> dict:
        """Returns the message data parsed as JSON."""
        return json.loads(self.data)


class TextureAgentClient:
    """Client for the Texture Agent Service.

    Endpoints:
      - POST /pipeline                         (start pipeline)
      - POST /pipeline/upload-usd              (upload USD, returns session_id)
      - GET  /pipeline/{session_id}/events     (SSE stream)
      - GET  /pipeline/{session_id}/status     (polling status)
      - GET  /pipeline/{session_id}/results    (final results)
      - POST /pipeline/{session_id}/cancel     (cancel run)
      - POST /pipeline/{session_id}/regenerate (re-run specific steps)
      - GET  /artifacts/{session_id}/materials (materials JSON)
      - GET  /artifacts/{session_id}/textures  (textures ZIP)
      - GET  /artifacts/{session_id}/output (textured USDZ)
      - GET  /artifacts/{session_id}/renders   (renders ZIP)
      - GET  /sessions                         (list sessions)
      - DELETE /sessions/{session_id}          (delete session)
      - GET  /health                           (service health)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout_seconds: int = 600,
        token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._token = token or os.getenv("TEXTURE_AGENT_TOKEN")
        self._http = requests.Session()
        self._http.headers.update({"User-Agent": "texture-agent-client/1.0"})
        if self._token:
            self._http.headers.update({"Authorization": f"Bearer {self._token}"})

    # -------- Core operations
    def upload_usd(
        self, usd_path: str | None = None, *, s3_uri: str | None = None
    ) -> str:
        """Upload a USD file (or reference one on S3) and create a session.

        Returns:
            The session_id of the created session.
        """
        if not usd_path and not s3_uri:
            raise ValueError("Either usd_path or s3_uri must be provided")

        url = f"{self.base_url}/pipeline/upload-usd"

        if s3_uri:
            response = self._http.post(
                url, data={"s3_uri": s3_uri}, timeout=self.timeout_seconds
            )
            response.raise_for_status()
            return response.json()["session_id"]

        with open(usd_path, "rb") as f:
            files = [
                (
                    "usd_file",
                    (Path(usd_path).name, f, "application/octet-stream"),
                )
            ]
            response = self._http.post(url, files=files, timeout=self.timeout_seconds)
            response.raise_for_status()
            return response.json()["session_id"]

    def start_pipeline(
        self,
        session_id: str | None = None,
        usd_path: str | None = None,
        s3_uri: str | None = None,
        material_textures: dict[str, Any] | None = None,
        user_prompt: str | None = None,
        auto_prompt_enabled: bool | None = None,
        texture_backend: str | None = None,
        texture_endpoint: str | None = None,
        backend_engine: str | None = None,
        backend_custom_parameters: dict[str, Any] | None = None,
        detail_policy: str | None = None,
        reference_image_uris: list[str] | None = None,
        reference_image_path: str | None = None,
        turntable_video_uri: str | None = None,
        multiview_image_uris: list[str] | None = None,
        seed: int | None = None,
        strength: float | None = None,
        strict_scope: bool | None = None,
        uv_policy: str | None = None,
        uv_scope: str | None = None,
        uv_backend: str | None = None,
        uv_projection: str | None = None,
        uv_overwrite_existing: bool | None = None,
        uv_rebake_source_albedo: bool | None = None,
        uv_rebake_size: int | None = None,
        uv_normalize_out_of_range: bool | None = None,
    ) -> str:
        """Start the pipeline.

        Args:
            session_id: Existing session ID (from upload_usd)
            usd_path: Path to USD file (if not using session_id or s3_uri)
            s3_uri: S3 URI to a USD file
            material_textures: Per-material prompt/opacity config
            user_prompt: Aesthetic direction for auto-prompt generation
            auto_prompt_enabled: Set False for strict material_textures scope.
                None preserves the service default.
            texture_backend: Optional texture backend override, e.g. "service".
            texture_endpoint: Optional texture variation backend endpoint.
            backend_engine: Optional backend engine/model route hint.
            backend_custom_parameters: Optional backend custom parameter object.
            detail_policy: Optional texture detail policy, e.g. "surface_only".
            reference_image_uris: Optional global reference image URI list.
            reference_image_path: Optional global reference image upload path.
            turntable_video_uri: Optional global turntable video URI.
            multiview_image_uris: Optional global multi-view image URI list.
            seed: Optional texture backend seed.
            strength: Optional texture edit strength.
            strict_scope: Optional selected-scope enforcement flag.
            uv_policy: Optional UV preparation policy override.
            uv_scope: Optional UV projection scope override.
            uv_backend: Optional UV preparation backend override.
            uv_projection: Optional UV projection mode override.
            uv_overwrite_existing: Optional existing UV overwrite override.
            uv_rebake_source_albedo: Optional scoped source texture rebake override.
            uv_rebake_size: Optional scoped source texture rebake resolution.
            uv_normalize_out_of_range: Optional out-of-range UV normalization override.

        Returns the session_id of the started run.
        """
        url = f"{self.base_url}/pipeline"
        files: list[tuple[str, tuple[str, object, str]]] = []

        data: dict[str, str] = {}
        if session_id:
            data["session_id"] = session_id
        if s3_uri:
            data["s3_uri"] = s3_uri
        if material_textures:
            data["material_textures_json"] = json.dumps(material_textures)
        if user_prompt:
            data["user_prompt"] = user_prompt
        if auto_prompt_enabled is not None:
            data["auto_prompt_enabled"] = "true" if auto_prompt_enabled else "false"
        if texture_backend:
            data["texture_backend"] = texture_backend
        if texture_endpoint:
            data["texture_endpoint"] = texture_endpoint
        if backend_engine:
            data["backend_engine"] = backend_engine
        if backend_custom_parameters:
            data["backend_custom_parameters_json"] = json.dumps(
                backend_custom_parameters
            )
        if detail_policy:
            data["detail_policy"] = detail_policy
        if reference_image_uris:
            data["reference_image_uris_json"] = json.dumps(reference_image_uris)
        if turntable_video_uri:
            data["turntable_video_uri"] = turntable_video_uri
        if multiview_image_uris:
            data["multiview_image_uris_json"] = json.dumps(multiview_image_uris)
        if seed is not None:
            data["seed"] = str(seed)
        if strength is not None:
            data["strength"] = str(strength)
        if strict_scope is not None:
            data["strict_scope"] = "true" if strict_scope else "false"
        if uv_policy:
            data["uv_policy"] = uv_policy
        if uv_scope:
            data["uv_scope"] = uv_scope
        if uv_backend:
            data["uv_backend"] = uv_backend
        if uv_projection:
            data["uv_projection"] = uv_projection
        if uv_overwrite_existing is not None:
            data["uv_overwrite_existing"] = "true" if uv_overwrite_existing else "false"
        if uv_rebake_source_albedo is not None:
            data["uv_rebake_source_albedo"] = (
                "true" if uv_rebake_source_albedo else "false"
            )
        if uv_rebake_size is not None:
            data["uv_rebake_size"] = str(uv_rebake_size)
        if uv_normalize_out_of_range is not None:
            data["uv_normalize_out_of_range"] = (
                "true" if uv_normalize_out_of_range else "false"
            )

        with ExitStack() as stack:
            if usd_path:
                uf = stack.enter_context(open(usd_path, "rb"))
                files.append(
                    (
                        "usd_file",
                        (Path(usd_path).name, uf, "application/octet-stream"),
                    )
                )
            if reference_image_path:
                rf = stack.enter_context(open(reference_image_path, "rb"))
                files.append(
                    (
                        "reference_image_file",
                        (
                            Path(reference_image_path).name,
                            rf,
                            "application/octet-stream",
                        ),
                    )
                )
            response = self._http.post(
                url,
                data=data or None,
                files=files or None,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            result = response.json()
            return result["session_id"]

    def regenerate(
        self,
        session_id: str,
        steps: list[str],
        material_textures: dict[str, Any] | None = None,
        texture_unit_ids: list[str] | None = None,
    ) -> dict:
        """Re-run specific pipeline steps.

        Args:
            session_id: Session to regenerate
            steps: List of step names to re-run
            material_textures: Optional material config override
            texture_unit_ids: Exact approved plan-unit IDs to regenerate. Omit
                to regenerate all approved units.
        """
        url = f"{self.base_url}/pipeline/{session_id}/regenerate"
        body: dict[str, Any] = {"steps": steps}
        if material_textures:
            body["material_textures"] = material_textures
        if texture_unit_ids is not None:
            body["texture_unit_ids"] = texture_unit_ids
        resp = self._http.post(url, json=body, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    # -------- Monitoring and results
    def stream_events(
        self, session_id: str, request_timeout: int | None = None
    ) -> Generator[SSEMessage, None, None]:
        """Connect to the SSE endpoint and yield parsed SSEMessage objects."""
        url = f"{self.base_url}/pipeline/{session_id}/events"
        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        timeout = request_timeout or max(self.timeout_seconds, 60)
        with self._http.get(url, headers=headers, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            buffer_event: str | None = None
            buffer_data_lines: list[str] = []
            buffer_id: str | None = None
            buffer_retry: int | None = None

            def emit_if_any() -> SSEMessage | None:
                if (
                    buffer_event is None
                    and not buffer_data_lines
                    and buffer_id is None
                    and buffer_retry is None
                ):
                    return None
                data_str = "\n".join(buffer_data_lines) if buffer_data_lines else ""
                return SSEMessage(
                    event=buffer_event or "message",
                    data=data_str,
                    id=buffer_id,
                    retry=buffer_retry,
                )

            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.rstrip("\r")
                if line == "":
                    msg = emit_if_any()
                    if msg:
                        yield msg
                    buffer_event = None
                    buffer_data_lines = []
                    buffer_id = None
                    buffer_retry = None
                    continue

                if line.startswith(":"):
                    continue

                field, sep, value = line.partition(":")
                if sep:
                    value = value.lstrip(" ")
                else:
                    value = ""

                if field == "event":
                    buffer_event = value
                elif field == "data":
                    buffer_data_lines.append(value)
                elif field == "id":
                    buffer_id = value
                elif field == "retry":
                    try:
                        buffer_retry = int(value)
                    except ValueError:
                        buffer_retry = None

            final_msg = emit_if_any()
            if final_msg:
                yield final_msg

    def get_status(self, session_id: str) -> dict:
        url = f"{self.base_url}/pipeline/{session_id}/status"
        resp = self._http.get(
            url,
            timeout=min(self.timeout_seconds, STATUS_POLL_TIMEOUT_SECONDS),
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _is_transient_status_error(exc: Exception) -> bool:
        if isinstance(exc, requests.Timeout | requests.ConnectionError):
            return True
        if not isinstance(exc, requests.HTTPError):
            return False
        response = exc.response
        return response is not None and response.status_code in TRANSIENT_STATUS_CODES

    def _get_status_with_transient_retries(
        self,
        session_id: str,
        *,
        print_stream: bool,
    ) -> dict:
        transient_errors = 0
        while True:
            try:
                return self.get_status(session_id)
            except Exception as exc:
                if (
                    transient_errors < MAX_TRANSIENT_STATUS_ERRORS
                    and self._is_transient_status_error(exc)
                ):
                    transient_errors += 1
                    if print_stream:
                        print(
                            f"Transient status poll error ({exc}); retrying...",
                            flush=True,
                        )
                    time.sleep(TRANSIENT_STATUS_RETRY_SECONDS)
                    continue
                raise

    def get_results(self, session_id: str) -> dict:
        url = f"{self.base_url}/pipeline/{session_id}/results"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def get_event_log(self, session_id: str) -> dict:
        url = f"{self.base_url}/pipeline/{session_id}/event-log"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def cancel(self, session_id: str) -> None:
        url = f"{self.base_url}/pipeline/{session_id}/cancel"
        resp = self._http.post(url, timeout=self.timeout_seconds)
        resp.raise_for_status()

    # -------- Artifact downloads
    def download_materials(self, session_id: str) -> dict:
        """Download discovered materials JSON."""
        url = f"{self.base_url}/artifacts/{session_id}/materials"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def download_textures(self, session_id: str, output_dir: str) -> list[str]:
        """Download all textures as ZIP and extract to output_dir.

        Returns list of extracted file paths.
        """
        import zipfile
        from io import BytesIO

        url = f"{self.base_url}/artifacts/{session_id}/textures"
        resp = self._http.get(url, timeout=self.timeout_seconds * 2)
        resp.raise_for_status()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        extracted = []
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            for member in zf.namelist():
                member_path = Path(member)
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue
                target = (output_path / member).resolve()
                if not target.is_relative_to(output_path.resolve()):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
                extracted.append(str(target))

        return extracted

    def download_output(self, session_id: str, output_path: str) -> str:
        """Download the self-contained USDZ output (USD + textures bundled).

        Returns the local file path.
        """
        url = f"{self.base_url}/artifacts/{session_id}/output"
        resp = self._http.get(url, timeout=self.timeout_seconds * 2)
        resp.raise_for_status()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(resp.content)
        return str(out)

    def download_renders(self, session_id: str, output_dir: str) -> list[str]:
        """Download all renders as ZIP and extract to output_dir.

        Returns list of extracted file paths.
        """
        import zipfile
        from io import BytesIO

        url = f"{self.base_url}/artifacts/{session_id}/renders"
        resp = self._http.get(url, timeout=self.timeout_seconds * 2)
        resp.raise_for_status()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        extracted = []
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            for member in zf.namelist():
                member_path = Path(member)
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue
                target = (output_path / member).resolve()
                if not target.is_relative_to(output_path.resolve()):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
                extracted.append(str(target))

        return extracted

    # -------- Utilities
    def sessions(self) -> dict:
        url = f"{self.base_url}/sessions"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def delete_session(self, session_id: str) -> None:
        url = f"{self.base_url}/sessions/{session_id}"
        resp = self._http.delete(url, timeout=self.timeout_seconds)
        resp.raise_for_status()

    def health(self) -> dict:
        url = f"{self.base_url}/health"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _current_step_name(current_step: Any) -> str:
        if current_step is None or current_step == "":
            return "none"
        if isinstance(current_step, dict):
            return str(
                current_step.get("name")
                or current_step.get("step")
                or current_step.get("display_name")
                or "unknown"
            )
        return str(current_step)

    @staticmethod
    def _current_step_message(current_step: Any) -> str:
        if not isinstance(current_step, dict):
            return ""
        progress = current_step.get("progress")
        if not isinstance(progress, dict):
            return ""
        return str(progress.get("message") or "")

    # -------- Convenience workflow
    def run_and_monitor(
        self,
        usd_path: str | None = None,
        s3_uri: str | None = None,
        material_textures: dict[str, Any] | None = None,
        user_prompt: str | None = None,
        auto_prompt_enabled: bool | None = None,
        texture_backend: str | None = None,
        texture_endpoint: str | None = None,
        backend_engine: str | None = None,
        backend_custom_parameters: dict[str, Any] | None = None,
        detail_policy: str | None = None,
        reference_image_uris: list[str] | None = None,
        reference_image_path: str | None = None,
        turntable_video_uri: str | None = None,
        multiview_image_uris: list[str] | None = None,
        seed: int | None = None,
        strength: float | None = None,
        strict_scope: bool | None = None,
        uv_policy: str | None = None,
        uv_scope: str | None = None,
        uv_backend: str | None = None,
        uv_projection: str | None = None,
        uv_overwrite_existing: bool | None = None,
        uv_rebake_source_albedo: bool | None = None,
        uv_rebake_size: int | None = None,
        uv_normalize_out_of_range: bool | None = None,
        upload_first: bool = False,
        print_stream: bool = True,
        reconnect_attempts: int = 3,
        reconnect_backoff_seconds: float = 2.0,
        max_polls: int = 300,
        max_stale_pending_polls: int = 0,
    ) -> tuple[str, dict | None]:
        """High-level helper that starts the pipeline and monitors it.

        Args:
            usd_path: Path to USD file to process (local).
            s3_uri: S3 URI to a USD file.
            material_textures: Per-material texture config.
            user_prompt: Aesthetic direction for auto-prompt generation.
            auto_prompt_enabled: Set False for strict material_textures scope.
                None preserves the service default.
            texture_backend: Optional texture backend override, e.g. "service".
            texture_endpoint: Optional texture variation backend endpoint.
            backend_engine: Optional backend engine/model route hint.
            backend_custom_parameters: Optional backend custom parameter object.
            detail_policy: Optional texture detail policy, e.g. "surface_only".
            reference_image_uris: Optional global reference image URI list.
            reference_image_path: Optional global reference image upload path.
            turntable_video_uri: Optional global turntable video URI.
            multiview_image_uris: Optional global multi-view image URI list.
            seed: Optional texture backend seed.
            strength: Optional texture edit strength.
            strict_scope: Optional selected-scope enforcement flag.
            uv_policy: Optional UV preparation policy override.
            uv_scope: Optional UV projection scope override.
            uv_backend: Optional UV preparation backend override.
            uv_projection: Optional UV projection mode override.
            uv_overwrite_existing: Optional existing UV overwrite override.
            uv_rebake_source_albedo: Optional scoped source texture rebake override.
            uv_rebake_size: Optional scoped source texture rebake resolution.
            uv_normalize_out_of_range: Optional out-of-range UV normalization override.
            upload_first: If True, upload USD first via /upload-usd.
            print_stream: Print progress updates to stdout.
            reconnect_attempts: Number of SSE reconnect attempts.
            reconnect_backoff_seconds: Seconds between reconnect attempts.
            max_polls: Maximum number of 2-second status polls after SSE fallback.
            max_stale_pending_polls: Stop polling after this many unchanged pending
                or startup polls with no real pipeline step. Set 0 to disable the
                guard.

        Returns (session_id, status_dict_or_none).
        """
        if not usd_path and not s3_uri:
            raise ValueError("Either usd_path or s3_uri must be provided")
        if max_polls <= 0:
            raise ValueError("max_polls must be positive")
        if max_stale_pending_polls < 0:
            raise ValueError("max_stale_pending_polls must be non-negative")

        projection_kwargs = {
            "texture_backend": texture_backend,
            "texture_endpoint": texture_endpoint,
            "backend_engine": backend_engine,
            "backend_custom_parameters": backend_custom_parameters,
            "detail_policy": detail_policy,
            "reference_image_uris": reference_image_uris,
            "reference_image_path": reference_image_path,
            "turntable_video_uri": turntable_video_uri,
            "multiview_image_uris": multiview_image_uris,
            "seed": seed,
            "strength": strength,
            "strict_scope": strict_scope,
            "uv_policy": uv_policy,
            "uv_scope": uv_scope,
            "uv_backend": uv_backend,
            "uv_projection": uv_projection,
            "uv_overwrite_existing": uv_overwrite_existing,
            "uv_rebake_source_albedo": uv_rebake_source_albedo,
            "uv_rebake_size": uv_rebake_size,
            "uv_normalize_out_of_range": uv_normalize_out_of_range,
        }

        if s3_uri:
            session_id = self.upload_usd(s3_uri=s3_uri)
            if print_stream:
                print(
                    f"Downloaded USD from S3, session: {session_id}",
                    flush=True,
                )
            session_id = self.start_pipeline(
                session_id=session_id,
                material_textures=material_textures,
                user_prompt=user_prompt,
                auto_prompt_enabled=auto_prompt_enabled,
                **projection_kwargs,
            )
        elif upload_first:
            session_id = self.upload_usd(usd_path)
            if print_stream:
                print(f"Uploaded USD, session: {session_id}", flush=True)
            session_id = self.start_pipeline(
                session_id=session_id,
                material_textures=material_textures,
                user_prompt=user_prompt,
                auto_prompt_enabled=auto_prompt_enabled,
                **projection_kwargs,
            )
        else:
            session_id = self.start_pipeline(
                usd_path=usd_path,
                material_textures=material_textures,
                user_prompt=user_prompt,
                auto_prompt_enabled=auto_prompt_enabled,
                **projection_kwargs,
            )

        if print_stream:
            print(f"Started session: {session_id}", flush=True)

        last_status: dict | None = None

        saw_done = False
        if max_stale_pending_polls > 0:
            if print_stream:
                print(
                    "Using bounded status polling; skipping SSE stream.",
                    flush=True,
                )
        else:
            # Try SSE; if it fails, fall back to polling. Bounded CI runs skip
            # SSE because keepalive pings can otherwise keep this loop alive
            # forever while startup is stale.
            attempts_left = reconnect_attempts
            while attempts_left >= 0 and not saw_done:
                try:
                    for msg in self.stream_events(session_id):
                        if msg.event == "ping":
                            continue
                        if msg.event == "progress":
                            try:
                                payload = msg.json()
                            except Exception:
                                payload = {"raw": msg.data}
                            if print_stream:
                                step = payload.get("step")
                                state = payload.get("state")
                                overall = payload.get("overall_percent")
                                message = payload.get("message")
                                print(
                                    f"[{step}] {state} overall={overall}% {message or ''}".rstrip(),
                                    flush=True,
                                )
                        elif msg.event == "done":
                            try:
                                done_payload = msg.json()
                            except Exception:
                                done_payload = {}
                            final_state = done_payload.get("final_state")
                            if isinstance(final_state, str):
                                last_status = {**done_payload, "status": final_state}
                            saw_done = True
                            break
                    if not saw_done:
                        break
                except Exception as e:
                    if attempts_left == 0:
                        if print_stream:
                            print(
                                f"SSE failed, falling back to polling: {e}",
                                flush=True,
                            )
                        break
                    if print_stream:
                        print(
                            f"SSE error ({e}), retrying in {reconnect_backoff_seconds}s...",
                            flush=True,
                        )
                    time.sleep(reconnect_backoff_seconds)
                    attempts_left -= 1

        if not saw_done:
            if print_stream:
                print("Polling status...", flush=True)
            stale_pending_polls = 0
            last_pending_signature: tuple[str, str, str, str, str] | None = None
            for _ in range(max_polls):
                status = self._get_status_with_transient_retries(
                    session_id,
                    print_stream=print_stream,
                )
                last_status = status
                st = status.get("status")
                overall = status.get("overall_percent") or status.get("progress")
                if overall is None and isinstance(status.get("overall_progress"), dict):
                    overall = status["overall_progress"].get("percent")
                if overall is None:
                    overall = "-"
                current_step = status.get("current_step") or status.get("currentStep")
                current_step_name = self._current_step_name(current_step)
                current_step_message = self._current_step_message(current_step)
                completed_steps = (
                    status.get("completed_steps") or status.get("completedSteps") or []
                )
                if isinstance(completed_steps, list):
                    completed_summary = str(len(completed_steps))
                else:
                    completed_summary = str(completed_steps)
                if print_stream:
                    print(
                        "status="
                        f"{st} overall={overall} current_step={current_step_name} "
                        f"completed_steps={completed_summary}",
                        flush=True,
                    )
                if st in {"completed", "failed", "cancelled"}:
                    break
                stale_startup_state = completed_summary == "0" and (
                    (st == "pending" and current_step_name == "none")
                    or (st == "running" and current_step_name == "pipeline_startup")
                )
                if max_stale_pending_polls > 0 and stale_startup_state:
                    pending_signature = (
                        str(st),
                        str(overall),
                        current_step_name,
                        current_step_message,
                        str(completed_summary),
                    )
                    if pending_signature == last_pending_signature:
                        stale_pending_polls += 1
                    else:
                        last_pending_signature = pending_signature
                        stale_pending_polls = 1
                    if stale_pending_polls >= max_stale_pending_polls:
                        if print_stream:
                            print(
                                "No pipeline progress after "
                                f"{stale_pending_polls} unchanged polls; "
                                f"leaving session {st} for caller.",
                                flush=True,
                            )
                        break
                else:
                    last_pending_signature = None
                    stale_pending_polls = 0
                time.sleep(2)
            else:
                if print_stream:
                    print(
                        f"Timed out waiting for pipeline completion after {max_polls} polls",
                        flush=True,
                    )

        try:
            status = self._get_status_with_transient_retries(
                session_id,
                print_stream=print_stream,
            )
        except Exception:
            status = last_status
        return session_id, status


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Texture Agent Service client")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8001",
        help="Service base URL",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token (or set TEXTURE_AGENT_TOKEN)",
    )
    parser.add_argument(
        "--material-textures",
        default=None,
        help='Per-material config as JSON string, e.g. \'{"Steel": {"prompt": "rusted steel", "opacity": 0.85}}\'',
    )
    parser.add_argument(
        "--user-prompt",
        default=None,
        help="Aesthetic direction for auto-prompt generation (e.g. 'old and weathered')",
    )
    parser.add_argument(
        "--disable-auto-prompt",
        action="store_true",
        help=(
            "Do not auto-generate prompts for materials missing from "
            "--material-textures"
        ),
    )
    parser.add_argument(
        "--detail-policy",
        choices=("default", "surface_only"),
        default=None,
        help=(
            "Texture detail policy. Use surface_only for AOI/CAD assets where "
            "semantic details already exist as geometry."
        ),
    )
    parser.add_argument(
        "--texture-backend",
        default=None,
        help=(
            "Texture backend override, for example service or simple_image_gen. "
            "Canonical sidecar deployments route simple_image_gen to the simple "
            "Texture Variation API sidecar."
        ),
    )
    parser.add_argument(
        "--texture-endpoint",
        default=None,
        help="Texture Variation API endpoint override for service-backed requests.",
    )
    parser.add_argument(
        "--backend-engine",
        default=None,
        help="Texture Variation API engine/model route hint.",
    )
    parser.add_argument(
        "--uv-scope",
        default=None,
        help="UV projection scope override, for example stage or target_prims.",
    )
    parser.add_argument(
        "--upload-first",
        action="store_true",
        help="Upload USD via /upload-usd before starting pipeline",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print streaming updates",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--reconnect-attempts",
        type=int,
        default=3,
        help="Number of SSE reconnect attempts before polling fallback",
    )
    parser.add_argument(
        "--reconnect-backoff-seconds",
        type=float,
        default=2.0,
        help="Seconds to wait between SSE reconnect attempts",
    )
    parser.add_argument(
        "--max-polls",
        type=int,
        default=300,
        help="Maximum number of 2-second status polls after SSE fallback",
    )
    parser.add_argument(
        "--max-stale-pending-polls",
        type=int,
        default=0,
        help=(
            "Stop status polling after this many unchanged pending/startup polls "
            "with no real pipeline step; 0 disables the guard"
        ),
    )
    parser.add_argument(
        "--status-output",
        default=None,
        help="Optional path to write the final pipeline status JSON",
    )

    # USD source: local file or S3 URI
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "usd_path", nargs="?", default=None, help="Path to local USD file"
    )
    source.add_argument(
        "--s3-uri",
        default=None,
        help="S3 URI to a USD file (e.g. s3://bucket/path/scene.usdz)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    client = TextureAgentClient(
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        token=args.token,
    )

    material_textures = None
    if args.material_textures:
        material_textures = json.loads(args.material_textures)

    session_id, status = client.run_and_monitor(
        usd_path=args.usd_path,
        s3_uri=args.s3_uri,
        material_textures=material_textures,
        user_prompt=args.user_prompt,
        auto_prompt_enabled=False if args.disable_auto_prompt else None,
        detail_policy=args.detail_policy,
        texture_backend=args.texture_backend,
        texture_endpoint=args.texture_endpoint,
        backend_engine=args.backend_engine,
        uv_scope=args.uv_scope,
        upload_first=args.upload_first,
        print_stream=not args.quiet,
        reconnect_attempts=args.reconnect_attempts,
        reconnect_backoff_seconds=args.reconnect_backoff_seconds,
        max_polls=args.max_polls,
        max_stale_pending_polls=args.max_stale_pending_polls,
    )

    print(f"\nSession: {session_id}")
    status_text = "unknown"
    if status is not None:
        status_text = str(status.get("status", "unknown"))
        print(f"Pipeline status: {status_text}")
        if args.status_output:
            status_path = Path(args.status_output)
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(
                json.dumps(status, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"Status JSON: {status_path}")
        if status_text == "failed":
            failed_step = status.get("failed_step")
            error = status.get("error")
            failed_step_stats = status.get("failed_step_stats")
            if failed_step:
                print(f"Failed step: {failed_step}")
            if error:
                print(f"Error: {error}")
            if failed_step_stats:
                print("Failed step stats:")
                print(json.dumps(failed_step_stats, indent=2, sort_keys=True))
        print("\nArtifacts:")
        print(f"- Pipeline Status:  {client.base_url}/pipeline/{session_id}/status")
        print(f"- Materials JSON:   {client.base_url}/artifacts/{session_id}/materials")
        print(f"- Textures ZIP:     {client.base_url}/artifacts/{session_id}/textures")
        print(f"- Output USDZ:      {client.base_url}/artifacts/{session_id}/output")
        print(f"- Renders ZIP:      {client.base_url}/artifacts/{session_id}/renders")
    else:
        if args.status_output:
            status_path = Path(args.status_output)
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(
                json.dumps(
                    {"session_id": session_id, "status": None},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Status JSON: {status_path}")
        print("No results available yet.")
    return 0 if status_text == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
