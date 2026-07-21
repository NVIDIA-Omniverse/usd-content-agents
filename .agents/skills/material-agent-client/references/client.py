# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Simplified public reference client. See
# apps/material_agent_service/client/client.py for the full service client.
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Generator, Iterable
from contextlib import ExitStack
from dataclasses import dataclass

import requests

DEFAULT_MAX_VLM_WORKERS = 64
DEFAULT_MAX_RENDER_NUM_WORKERS = 32


def _max_from_env(env_name: str, default: int) -> int:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{env_name} must be at least 1")
    return value


def _validate_worker_override(
    name: str,
    value: int | None,
    env_name: str,
    default_max: int,
) -> None:
    if value is None:
        return
    max_value = _max_from_env(env_name, default_max)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer between 1 and {max_value}")
    if value < 1 or value > max_value:
        raise ValueError(
            f"{name} must be between 1 and {max_value} (set {env_name} to adjust)"
        )


def _validate_positive_override(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")


def _validate_unit_interval_override(name: str, value: int | float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be an int or float")
    if value < 0.0 or value > 1.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be between 0.0 and 1.0")


@dataclass(frozen=True)
class SSEMessage:
    """
    Represents a parsed Server-Sent Event (SSE) message.
    """

    event: str
    data: str
    id: str | None = None
    retry: int | None = None

    def json(self) -> dict:
        """
        Returns the message data parsed as JSON. Raises ValueError if parsing fails.
        """
        return json.loads(self.data)


class MaterialAgentClient:
    """
    Client for the Material Agent Service.

    Endpoints discovered from the frontend:
      - POST /pipeline                         (start pipeline; accepts usd_file or session_id + extras)
      - POST /pipeline/upload-usd              (upload file first; returns session_id)
      - POST /pipeline/{session_id}/generate-reference-image
      - GET  /pipeline/{session_id}/events     (SSE stream: progress/done/ping)
      - GET  /pipeline/{session_id}/status     (polling status)
      - GET  /pipeline/{session_id}/results    (final results)
      - POST /pipeline/{session_id}/cancel     (cancel run)
      - POST /pipeline/{session_id}/regenerate (re-run specific steps)
      - GET  /pipeline/{session_id}/event-log  (historic events)
      - GET  /assets/{session_id}/input-render (input preview)
      - GET  /assets/{session_id}/generated-ref/{reference_id}
      - GET  /sessions                         (list sessions)
      - GET  /health                           (service health)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout_seconds: int = 180,
        token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._token = token or os.getenv("MATERIAL_AGENT_TOKEN")
        self._http = requests.Session()
        self._http.headers.update({"User-Agent": "material-agent-client/1.0"})
        if self._token:
            self._http.headers.update({"Authorization": f"Bearer {self._token}"})

    # -------- Core operations
    def upload_usd(self, usd_path: str) -> str:
        """
        Upload a USD file to create a pending session. Returns session_id.
        """
        url = f"{self.base_url}/pipeline/upload-usd"
        with open(usd_path, "rb") as f:
            files = {
                "usd_file": (usd_path.split("/")[-1], f, "application/octet-stream")
            }
            response = self._http.post(url, files=files, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        session_id = data["session_id"]
        return session_id

    def wait_for_input_render(
        self,
        session_id: str,
        timeout_seconds: int = 180,
        poll_interval_seconds: float = 2.0,
    ) -> None:
        """
        Wait until the uploaded USD preview render is available.

        The generated-reference endpoint uses this preview as its conditioning
        image, so callers should wait for it after upload_usd().
        """
        url = f"{self.base_url}/assets/{session_id}/input-render"
        deadline = time.monotonic() + timeout_seconds

        while True:
            response = self._http.head(
                url, timeout=self.timeout_seconds, allow_redirects=True
            )
            if response.status_code in {200, 302, 303, 307, 308}:
                return
            if response.status_code == 424:
                raise RuntimeError("Input preview render failed on the service")
            if response.status_code not in {404, 503}:
                response.raise_for_status()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Input preview was not available within {timeout_seconds}s"
                )
            time.sleep(poll_interval_seconds)

    def generate_reference_image(self, session_id: str, prompt: str) -> dict:
        """
        Generate an AI reference image from the uploaded USD preview and prompt.

        Returns a reference_id. Pass that ID to start_pipeline() to use the
        generated image for material prediction.
        """
        url = f"{self.base_url}/pipeline/{session_id}/generate-reference-image"
        response = self._http.post(
            url,
            data={"prompt": prompt},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def start_pipeline(
        self,
        session_id: str | None = None,
        usd_path: str | None = None,
        reference_images: Iterable[str] | None = None,
        reference_pdfs: Iterable[str] | None = None,
        reference_descriptions: Iterable[str] | None = None,
        pdf_descriptions: Iterable[str] | None = None,
        user_prompt: str | None = None,
        camera_views: str | None = None,
        pdf_first_page: int | None = None,
        pdf_last_page: int | None = None,
        optimize_usd: bool | None = None,
        enable_deinstance: bool | None = None,
        enable_split: bool | None = None,
        enable_deduplicate: bool | None = None,
        materials_zip_path: str | None = None,
        vlm_model: str | None = None,
        vlm_max_workers: int | None = None,
        render_num_workers: int | None = None,
        generated_reference_id: str | None = None,
        user_email: str | None = None,
        layer_only: bool | None = None,
        enable_prim_clustering: bool | None = None,
        cluster_min_prims: int | None = None,
        cluster_embedding_backend: str | None = None,
        cluster_embedding_model: str | None = None,
        cluster_embedding_base_url: str | None = None,
        cluster_embedding_max_workers: int | None = None,
        cluster_embedding_batch_size: int | None = None,
        cluster_max_size: int | None = None,
        cluster_similarity_threshold_low: float | None = None,
        cluster_similarity_threshold_medium: float | None = None,
        cluster_similarity_threshold_high: float | None = None,
        cluster_report: bool | None = None,
    ) -> str:
        """
        Start the pipeline. You can either pass a pre-created session_id (from upload_usd)
        or provide a usd_path directly (server will accept the file inline).

        Args:
            materials_zip_path: Optional path to a ZIP file containing custom materials
                               (materials.yaml + USD library). Overrides server defaults.
            vlm_model: Optional VLM model override (e.g. "nim/nvidia/cosmos-reason2-8b").
            vlm_max_workers: Optional max parallel VLM workers.
            render_num_workers: Optional max parallel render workers.
            enable_prim_clustering: Enable or disable image-based clustering of
                visually similar prims before prediction.
            cluster_min_prims: Minimum prim count before clustering runs.
            cluster_embedding_backend: Embedding backend for clustering.
            cluster_embedding_model: Embedding model for clustering.
            cluster_embedding_base_url: Optional embedding endpoint base URL.
            cluster_embedding_max_workers: Optional parallel embedding workers.
            cluster_embedding_batch_size: Optional embedding batch size.
            cluster_max_size: Optional maximum prims per cluster.
            cluster_similarity_threshold_low: Optional low-complexity threshold.
            cluster_similarity_threshold_medium: Optional medium-complexity threshold.
            cluster_similarity_threshold_high: Optional high-complexity threshold.
            cluster_report: Enable or disable the clustering HTML report.
            user_email: Optional tracking email. Omit or pass blank to use the
                service fallback.
            layer_only: Enable or disable material-binding-layer-only output for
                this initial pipeline submission.

        Returns the session_id of the started run.
        """
        _validate_worker_override(
            "vlm_max_workers",
            vlm_max_workers,
            "VLM_MAX_WORKERS_MAX",
            DEFAULT_MAX_VLM_WORKERS,
        )
        _validate_worker_override(
            "render_num_workers",
            render_num_workers,
            "RENDER_NUM_WORKERS_MAX",
            DEFAULT_MAX_RENDER_NUM_WORKERS,
        )
        _validate_positive_override("cluster_min_prims", cluster_min_prims)
        _validate_positive_override(
            "cluster_embedding_max_workers", cluster_embedding_max_workers
        )
        _validate_positive_override(
            "cluster_embedding_batch_size", cluster_embedding_batch_size
        )
        _validate_positive_override("cluster_max_size", cluster_max_size)
        _validate_unit_interval_override(
            "cluster_similarity_threshold_low", cluster_similarity_threshold_low
        )
        _validate_unit_interval_override(
            "cluster_similarity_threshold_medium",
            cluster_similarity_threshold_medium,
        )
        _validate_unit_interval_override(
            "cluster_similarity_threshold_high", cluster_similarity_threshold_high
        )

        url = f"{self.base_url}/pipeline"
        files: list[tuple[str, tuple[str, object, str]]] = []
        data: dict[str, str] = {}

        with ExitStack() as stack:
            if session_id:
                data["session_id"] = session_id
            elif usd_path:
                f = stack.enter_context(open(usd_path, "rb"))
                files.append(
                    (
                        "usd_file",
                        (usd_path.split("/")[-1], f, "application/octet-stream"),
                    )
                )
            else:
                raise ValueError("Either session_id or usd_path must be provided.")

            normalized_user_email = (user_email or "").strip()
            if normalized_user_email:
                data["user_email"] = normalized_user_email

            if reference_images:
                for p in reference_images:
                    rf = stack.enter_context(open(p, "rb"))
                    files.append(
                        (
                            "reference_images",
                            (p.split("/")[-1], rf, "application/octet-stream"),
                        )
                    )

            if reference_pdfs:
                for p in reference_pdfs:
                    rf = stack.enter_context(open(p, "rb"))
                    files.append(
                        (
                            "reference_pdfs",
                            (p.split("/")[-1], rf, "application/pdf"),
                        )
                    )

            if materials_zip_path:
                mf = stack.enter_context(open(materials_zip_path, "rb"))
                files.append(
                    (
                        "materials_zip",
                        (materials_zip_path.split("/")[-1], mf, "application/zip"),
                    )
                )

            if reference_descriptions:
                # The service expects JSON string when descriptions are provided
                data["reference_descriptions"] = json.dumps(
                    list(reference_descriptions)
                )

            if pdf_descriptions:
                # The service expects JSON string when descriptions are provided
                data["pdf_descriptions"] = json.dumps(list(pdf_descriptions))

            if user_prompt:
                data["user_prompt"] = user_prompt
            if camera_views:
                data["camera_views"] = camera_views
            if pdf_first_page is not None:
                data["pdf_first_page"] = str(pdf_first_page)
            if pdf_last_page is not None:
                data["pdf_last_page"] = str(pdf_last_page)
            if vlm_model:
                data["vlm_model"] = vlm_model
            if vlm_max_workers is not None:
                data["vlm_max_workers"] = str(vlm_max_workers)
            if render_num_workers is not None:
                data["render_num_workers"] = str(render_num_workers)
            if enable_prim_clustering is not None:
                data["enable_prim_clustering"] = (
                    "true" if enable_prim_clustering else "false"
                )
            if cluster_min_prims is not None:
                data["cluster_min_prims"] = str(cluster_min_prims)
            if cluster_embedding_backend:
                data["cluster_embedding_backend"] = cluster_embedding_backend
            if cluster_embedding_model:
                data["cluster_embedding_model"] = cluster_embedding_model
            if cluster_embedding_base_url:
                data["cluster_embedding_base_url"] = cluster_embedding_base_url
            if cluster_embedding_max_workers is not None:
                data["cluster_embedding_max_workers"] = str(
                    cluster_embedding_max_workers
                )
            if cluster_embedding_batch_size is not None:
                data["cluster_embedding_batch_size"] = str(cluster_embedding_batch_size)
            if cluster_max_size is not None:
                data["cluster_max_size"] = str(cluster_max_size)
            if cluster_similarity_threshold_low is not None:
                data["cluster_similarity_threshold_low"] = str(
                    cluster_similarity_threshold_low
                )
            if cluster_similarity_threshold_medium is not None:
                data["cluster_similarity_threshold_medium"] = str(
                    cluster_similarity_threshold_medium
                )
            if cluster_similarity_threshold_high is not None:
                data["cluster_similarity_threshold_high"] = str(
                    cluster_similarity_threshold_high
                )
            if cluster_report is not None:
                data["cluster_report"] = "true" if cluster_report else "false"
            if generated_reference_id:
                data["generated_reference_id"] = generated_reference_id
            if layer_only is not None:
                data["layer_only"] = "true" if layer_only else "false"
            if optimize_usd is not None:
                data["optimize_usd"] = "true" if optimize_usd else "false"

                # Add operation flags if optimize is enabled
                if optimize_usd:
                    if enable_deinstance is not None:
                        data["enable_deinstance"] = (
                            "true" if enable_deinstance else "false"
                        )
                    if enable_split is not None:
                        data["enable_split"] = "true" if enable_split else "false"
                    if enable_deduplicate is not None:
                        data["enable_deduplicate"] = (
                            "true" if enable_deduplicate else "false"
                        )

            response = self._http.post(
                url, data=data, files=files or None, timeout=self.timeout_seconds
            )
            response.raise_for_status()
            result = response.json()
            return result["session_id"]

    def regenerate(
        self,
        session_id: str,
        steps: list[str],
        user_prompt: str | None = None,
        layer_only: bool | None = None,
    ) -> dict:
        """
        Re-run specific pipeline steps from cached session data.

        Args:
            session_id: Session to regenerate
            steps: List of step names to re-run
            user_prompt: Optional prompt override
            layer_only: Enable or disable material-binding-layer-only output.
                Omit to leave the service default unchanged.

        Returns the response JSON.
        """
        url = f"{self.base_url}/pipeline/{session_id}/regenerate"
        body: dict[str, object] = {"steps": steps}
        if user_prompt is not None:
            body["user_prompt"] = user_prompt
        if layer_only is not None:
            body["layer_only"] = layer_only
        resp = self._http.post(url, json=body, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    # -------- Monitoring and results
    def stream_events(
        self, session_id: str, request_timeout: int | None = None
    ) -> Generator[SSEMessage, None, None]:
        """
        Connect to the SSE endpoint and yield parsed SSEMessage objects as they arrive.
        This method handles basic SSE parsing without external dependencies.
        """
        url = f"{self.base_url}/pipeline/{session_id}/events"
        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        timeout = request_timeout or max(self.timeout_seconds, 60)
        with self._http.get(url, headers=headers, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            # Parse SSE: messages separated by blank line; fields: event:, data:, id:, retry:
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
                msg = SSEMessage(
                    event=buffer_event or "message",
                    data=data_str,
                    id=buffer_id,
                    retry=buffer_retry,
                )
                return msg

            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    # keep-alive heartbeat (usually)
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
                    # comment/heartbeat; ignore
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
                # else: ignore unknown fields

            # Flush on EOF
            final_msg = emit_if_any()
            if final_msg:
                yield final_msg

    def get_status(self, session_id: str) -> dict:
        url = f"{self.base_url}/pipeline/{session_id}/status"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

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

    # -------- Utilities
    def sessions(self) -> dict:
        url = f"{self.base_url}/sessions"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        url = f"{self.base_url}/health"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    # -------- Convenience workflow
    def run_and_monitor(
        self,
        usd_path: str,
        reference_images: Iterable[str] | None = None,
        reference_pdfs: Iterable[str] | None = None,
        reference_descriptions: Iterable[str] | None = None,
        pdf_descriptions: Iterable[str] | None = None,
        user_prompt: str | None = None,
        camera_views: str | None = None,
        pdf_first_page: int | None = None,
        pdf_last_page: int | None = None,
        upload_first: bool = False,
        generated_reference_prompt: str | None = None,
        preview_timeout_seconds: int = 180,
        print_stream: bool = True,
        reconnect_attempts: int = 3,
        reconnect_backoff_seconds: float = 2.0,
        optimize_usd: bool | None = None,
        materials_zip_path: str | None = None,
        vlm_model: str | None = None,
        vlm_max_workers: int | None = None,
        render_num_workers: int | None = None,
        user_email: str | None = None,
        layer_only: bool | None = None,
        enable_prim_clustering: bool | None = None,
        cluster_min_prims: int | None = None,
        cluster_embedding_backend: str | None = None,
        cluster_embedding_model: str | None = None,
        cluster_embedding_base_url: str | None = None,
        cluster_embedding_max_workers: int | None = None,
        cluster_embedding_batch_size: int | None = None,
        cluster_max_size: int | None = None,
        cluster_similarity_threshold_low: float | None = None,
        cluster_similarity_threshold_medium: float | None = None,
        cluster_similarity_threshold_high: float | None = None,
        cluster_report: bool | None = None,
    ) -> tuple[str, dict | None]:
        """
        High-level helper that starts the pipeline and monitors it until completion.

        Args:
            materials_zip_path: Optional path to a ZIP file containing custom materials
                               (materials.yaml + USD library). Overrides server defaults.
            vlm_model: Optional VLM model override (e.g. "nim/nvidia/cosmos-reason2-8b").
            vlm_max_workers: Optional max parallel VLM workers.
            render_num_workers: Optional max parallel render workers.
            enable_prim_clustering: Enable or disable image-based prim clustering.
            cluster_min_prims: Minimum prim count before clustering runs.
            cluster_embedding_backend: Embedding backend for clustering.
            cluster_embedding_model: Embedding model for clustering.
            cluster_embedding_base_url: Optional embedding endpoint base URL.
            cluster_embedding_max_workers: Optional parallel embedding workers.
            cluster_embedding_batch_size: Optional embedding batch size.
            cluster_max_size: Optional maximum prims per cluster.
            cluster_similarity_threshold_low: Optional low-complexity threshold.
            cluster_similarity_threshold_medium: Optional medium-complexity threshold.
            cluster_similarity_threshold_high: Optional high-complexity threshold.
            cluster_report: Enable or disable the clustering HTML report.
            user_email: Optional tracking email. Omit or pass blank to use the
                service fallback.
            layer_only: Enable or disable material-binding-layer-only output.
            generated_reference_prompt: If set, upload first, wait for the input
                preview, generate an AI reference image, then start the pipeline.

        Returns (session_id, results_dict_or_none).
        """
        # Validate before upload or reference-generation calls can create
        # server-side state. start_pipeline repeats these checks for direct callers.
        _validate_worker_override(
            "vlm_max_workers",
            vlm_max_workers,
            "VLM_MAX_WORKERS_MAX",
            DEFAULT_MAX_VLM_WORKERS,
        )
        _validate_worker_override(
            "render_num_workers",
            render_num_workers,
            "RENDER_NUM_WORKERS_MAX",
            DEFAULT_MAX_RENDER_NUM_WORKERS,
        )
        _validate_positive_override("cluster_min_prims", cluster_min_prims)
        _validate_positive_override(
            "cluster_embedding_max_workers", cluster_embedding_max_workers
        )
        _validate_positive_override(
            "cluster_embedding_batch_size", cluster_embedding_batch_size
        )
        _validate_positive_override("cluster_max_size", cluster_max_size)
        _validate_unit_interval_override(
            "cluster_similarity_threshold_low", cluster_similarity_threshold_low
        )
        _validate_unit_interval_override(
            "cluster_similarity_threshold_medium",
            cluster_similarity_threshold_medium,
        )
        _validate_unit_interval_override(
            "cluster_similarity_threshold_high", cluster_similarity_threshold_high
        )

        generated_reference_id = None
        if upload_first or generated_reference_prompt:
            session_id = self.upload_usd(usd_path)
            if generated_reference_prompt:
                if print_stream:
                    print("Waiting for input preview...", flush=True)
                self.wait_for_input_render(
                    session_id,
                    timeout_seconds=preview_timeout_seconds,
                )
                if print_stream:
                    print("Generating reference image...", flush=True)
                generated_ref = self.generate_reference_image(
                    session_id, generated_reference_prompt
                )
                generated_reference_id = generated_ref.get("reference_id")
                if not generated_reference_id:
                    raise RuntimeError(
                        "generate-reference-image did not return reference_id"
                    )
            session_id = self.start_pipeline(
                session_id=session_id,
                reference_images=reference_images,
                reference_pdfs=reference_pdfs,
                reference_descriptions=reference_descriptions,
                pdf_descriptions=pdf_descriptions,
                user_prompt=user_prompt,
                camera_views=camera_views,
                pdf_first_page=pdf_first_page,
                pdf_last_page=pdf_last_page,
                optimize_usd=optimize_usd,
                materials_zip_path=materials_zip_path,
                vlm_model=vlm_model,
                vlm_max_workers=vlm_max_workers,
                render_num_workers=render_num_workers,
                enable_prim_clustering=enable_prim_clustering,
                cluster_min_prims=cluster_min_prims,
                cluster_embedding_backend=cluster_embedding_backend,
                cluster_embedding_model=cluster_embedding_model,
                cluster_embedding_base_url=cluster_embedding_base_url,
                cluster_embedding_max_workers=cluster_embedding_max_workers,
                cluster_embedding_batch_size=cluster_embedding_batch_size,
                cluster_max_size=cluster_max_size,
                cluster_similarity_threshold_low=cluster_similarity_threshold_low,
                cluster_similarity_threshold_medium=(
                    cluster_similarity_threshold_medium
                ),
                cluster_similarity_threshold_high=cluster_similarity_threshold_high,
                cluster_report=cluster_report,
                generated_reference_id=generated_reference_id,
                user_email=user_email,
                layer_only=layer_only,
            )
        else:
            session_id = self.start_pipeline(
                usd_path=usd_path,
                reference_images=reference_images,
                reference_pdfs=reference_pdfs,
                reference_descriptions=reference_descriptions,
                pdf_descriptions=pdf_descriptions,
                user_prompt=user_prompt,
                camera_views=camera_views,
                pdf_first_page=pdf_first_page,
                pdf_last_page=pdf_last_page,
                optimize_usd=optimize_usd,
                materials_zip_path=materials_zip_path,
                vlm_model=vlm_model,
                vlm_max_workers=vlm_max_workers,
                render_num_workers=render_num_workers,
                enable_prim_clustering=enable_prim_clustering,
                cluster_min_prims=cluster_min_prims,
                cluster_embedding_backend=cluster_embedding_backend,
                cluster_embedding_model=cluster_embedding_model,
                cluster_embedding_base_url=cluster_embedding_base_url,
                cluster_embedding_max_workers=cluster_embedding_max_workers,
                cluster_embedding_batch_size=cluster_embedding_batch_size,
                cluster_max_size=cluster_max_size,
                cluster_similarity_threshold_low=cluster_similarity_threshold_low,
                cluster_similarity_threshold_medium=(
                    cluster_similarity_threshold_medium
                ),
                cluster_similarity_threshold_high=cluster_similarity_threshold_high,
                cluster_report=cluster_report,
                user_email=user_email,
                layer_only=layer_only,
            )

        if print_stream:
            print(f"Started session: {session_id}", flush=True)

        # Try SSE; if it fails, fall back to polling.
        attempts_left = reconnect_attempts
        saw_done = False
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
                        saw_done = True
                        break
                # If stream ended without 'done', break to poll
                if not saw_done:
                    break
            except Exception as e:
                if attempts_left == 0:
                    if print_stream:
                        print(f"SSE failed, falling back to polling: {e}", flush=True)
                    break
                if print_stream:
                    print(
                        f"SSE error ({e}), retrying in {reconnect_backoff_seconds}s...",
                        flush=True,
                    )
                time.sleep(reconnect_backoff_seconds)
                attempts_left -= 1

        if not saw_done:
            # Polling fallback until terminal state
            if print_stream:
                print("Polling status...", flush=True)
            while True:
                status = self.get_status(session_id)
                st = status.get("status")
                if print_stream:
                    overall = (
                        status.get("overall_percent") or status.get("progress") or "-"
                    )
                    print(f"status={st} overall={overall}", flush=True)
                if st in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(2)

        # Try to fetch results (may only exist on completion)
        try:
            status = self.get_status(session_id)
        except Exception:
            status = None
        return session_id, status


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Material Agent Service client")
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="Service base URL"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token for Authorization header (or set MATERIAL_AGENT_TOKEN)",
    )
    parser.add_argument(
        "--upload-first",
        action="store_true",
        help="Upload USD first to create session, then start",
    )
    parser.add_argument(
        "--camera-views", default=None, help="Camera views, e.g. '+x+y+z,-x-y-z'"
    )
    parser.add_argument(
        "--prompt", default=None, help="Additional guidance for the VLM"
    )
    parser.add_argument(
        "--generate-ref-prompt",
        default=None,
        help=(
            "Generate an AI reference image from the input preview before "
            "starting the pipeline"
        ),
    )
    parser.add_argument(
        "--preview-timeout",
        type=int,
        default=180,
        help="Seconds to wait for input preview when --generate-ref-prompt is used",
    )
    parser.add_argument(
        "--ref", action="append", default=None, help="Reference image path (repeatable)"
    )
    parser.add_argument(
        "--ref-pdf",
        action="append",
        default=None,
        help="Reference PDF path (repeatable)",
    )
    parser.add_argument(
        "--ref-desc",
        action="append",
        default=None,
        help="Reference image description (repeatable; order must match --ref)",
    )
    parser.add_argument(
        "--pdf-desc",
        action="append",
        default=None,
        help="Reference PDF description (repeatable; order must match --ref-pdf)",
    )
    parser.add_argument(
        "--pdf-first-page",
        type=int,
        default=None,
        help="First page to convert from PDFs (1-indexed)",
    )
    parser.add_argument(
        "--pdf-last-page",
        type=int,
        default=None,
        help="Last page to convert from PDFs (1-indexed)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Do not print streaming updates"
    )
    parser.add_argument(
        "--optimize-usd",
        action="store_true",
        help="Enable USD optimization before/within pipeline",
    )
    parser.add_argument(
        "--materials-zip",
        default=None,
        help="Path to ZIP file with custom materials (materials.yaml + USD library)",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help=(
            "VLM model override "
            "(e.g. 'gcp/google/gemini-3.1-pro-preview', 'nim/nvidia/cosmos-reason2-8b')"
        ),
    )
    parser.add_argument(
        "--vlm-max-workers",
        type=int,
        default=None,
        help="Maximum parallel VLM workers for prediction",
    )
    parser.add_argument(
        "--render-num-workers",
        type=int,
        default=None,
        help="Maximum parallel render workers for build_dataset_usd",
    )
    clustering_group = parser.add_mutually_exclusive_group()
    clustering_group.add_argument(
        "--enable-prim-clustering",
        dest="enable_prim_clustering",
        action="store_true",
        default=None,
        help="Enable image-based clustering of visually similar prims",
    )
    clustering_group.add_argument(
        "--disable-prim-clustering",
        dest="enable_prim_clustering",
        action="store_false",
        default=None,
        help="Explicitly disable image-based prim clustering",
    )
    parser.add_argument("--cluster-min-prims", type=int, default=None)
    parser.add_argument("--cluster-embedding-backend", default=None)
    parser.add_argument("--cluster-embedding-model", default=None)
    parser.add_argument("--cluster-embedding-base-url", default=None)
    parser.add_argument("--cluster-embedding-max-workers", type=int, default=None)
    parser.add_argument("--cluster-embedding-batch-size", type=int, default=None)
    parser.add_argument("--cluster-max-size", type=int, default=None)
    parser.add_argument("--cluster-similarity-threshold-low", type=float, default=None)
    parser.add_argument(
        "--cluster-similarity-threshold-medium", type=float, default=None
    )
    parser.add_argument("--cluster-similarity-threshold-high", type=float, default=None)
    cluster_report_group = parser.add_mutually_exclusive_group()
    cluster_report_group.add_argument(
        "--cluster-report",
        dest="cluster_report",
        action="store_true",
        default=None,
        help="Enable the prim-clustering HTML report",
    )
    cluster_report_group.add_argument(
        "--no-cluster-report",
        dest="cluster_report",
        action="store_false",
        default=None,
        help="Disable the prim-clustering HTML report",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Optional user email for usage tracking; service fallback if omitted",
    )
    layer_only_group = parser.add_mutually_exclusive_group()
    layer_only_group.add_argument(
        "--layer-only",
        dest="layer_only",
        action="store_true",
        default=None,
        help="Output only material bindings layer (preserves original scene structure)",
    )
    layer_only_group.add_argument(
        "--no-layer-only",
        dest="layer_only",
        action="store_false",
        default=None,
        help="Explicitly disable material-binding-layer-only output",
    )
    parser.add_argument("usd", help="Path to USD file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    client = MaterialAgentClient(base_url=args.base_url, token=args.token)

    if args.ref_desc and args.ref and len(args.ref_desc) != len(args.ref):
        print("Error: --ref and --ref-desc counts must match.", file=sys.stderr)
        return 2

    if args.pdf_desc and args.ref_pdf and len(args.pdf_desc) != len(args.ref_pdf):
        print("Error: --ref-pdf and --pdf-desc counts must match.", file=sys.stderr)
        return 2

    session_id, status = client.run_and_monitor(
        usd_path=args.usd,
        reference_images=args.ref,
        reference_pdfs=args.ref_pdf,
        reference_descriptions=args.ref_desc,
        pdf_descriptions=args.pdf_desc,
        user_prompt=args.prompt,
        camera_views=args.camera_views,
        pdf_first_page=args.pdf_first_page,
        pdf_last_page=args.pdf_last_page,
        upload_first=args.upload_first,
        generated_reference_prompt=args.generate_ref_prompt,
        preview_timeout_seconds=args.preview_timeout,
        print_stream=not args.quiet,
        optimize_usd=args.optimize_usd,
        materials_zip_path=args.materials_zip,
        vlm_model=args.vlm_model,
        vlm_max_workers=args.vlm_max_workers,
        render_num_workers=args.render_num_workers,
        enable_prim_clustering=args.enable_prim_clustering,
        cluster_min_prims=args.cluster_min_prims,
        cluster_embedding_backend=args.cluster_embedding_backend,
        cluster_embedding_model=args.cluster_embedding_model,
        cluster_embedding_base_url=args.cluster_embedding_base_url,
        cluster_embedding_max_workers=args.cluster_embedding_max_workers,
        cluster_embedding_batch_size=args.cluster_embedding_batch_size,
        cluster_max_size=args.cluster_max_size,
        cluster_similarity_threshold_low=args.cluster_similarity_threshold_low,
        cluster_similarity_threshold_medium=args.cluster_similarity_threshold_medium,
        cluster_similarity_threshold_high=args.cluster_similarity_threshold_high,
        cluster_report=args.cluster_report,
        user_email=args.email,
        layer_only=args.layer_only,
    )

    print(f"\nSession: {session_id}")
    if status is not None:
        print(f"Pipeline status: {status['status']}")
        # Useful artifact endpoints
        print("\nArtifacts:")
        print(f"- Pipeline Status:    {client.base_url}/pipeline/{session_id}/status")
        print(f"- USD with materials: {client.base_url}/artifacts/{session_id}/output")
        print(
            f"- Predictions JSONL:  {client.base_url}/artifacts/{session_id}/predictions"
        )
        print(f"- Report HTML:        {client.base_url}/artifacts/{session_id}/report")
    else:
        print("No results available yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
