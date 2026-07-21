# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Generator, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from numbers import Integral
from typing import Literal
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

RouteFamily = Literal["pipeline", "predict", "tune", "refine"]
_VALID_ROUTE_FAMILIES = {"pipeline", "predict", "tune", "refine"}
_MIN_VISUAL_FRAME_COUNT = 1
_MAX_VISUAL_FRAME_COUNT = 64


def _bool_form(value: bool) -> str:
    return "true" if value else "false"


def _read_text_arg(
    *,
    value: str | None,
    path: str | None,
    name: str,
) -> str | None:
    if value is not None and path is not None:
        raise ValueError(f"Provide either {name} or {name}_path, not both")
    if path is None:
        return value
    with open(path, encoding="utf-8") as f:
        return f.read()


def _json_array_arg(values: Iterable[str] | None) -> str | None:
    if values is None:
        return None
    return json.dumps(list(values))


def _validate_family(family: str) -> RouteFamily:
    if family not in _VALID_ROUTE_FAMILIES:
        allowed = ", ".join(sorted(_VALID_ROUTE_FAMILIES))
        raise ValueError(f"family must be one of: {allowed}")
    return family  # type: ignore[return-value]


def _validate_visual_frame_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"{name} must be an integer between {_MIN_VISUAL_FRAME_COUNT} and "
            f"{_MAX_VISUAL_FRAME_COUNT}, got {value!r}"
        )
    count = int(value)
    if not (_MIN_VISUAL_FRAME_COUNT <= count <= _MAX_VISUAL_FRAME_COUNT):
        raise ValueError(
            f"{name} must be between {_MIN_VISUAL_FRAME_COUNT} and "
            f"{_MAX_VISUAL_FRAME_COUNT}, got {value}"
        )
    return count


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


class PhysicsAgentClient:
    """
    Client for the Physics Agent Service.

    Endpoints:
      - POST /pipeline                         (start pipeline; accepts usd_file, s3_uri, or session_id)
      - POST /pipeline/upload-usd              (upload USD file or provide s3_uri, returns session_id)
      - POST /predict                          (prediction-only workflow)
      - POST /tune                             (single-shot physics tuning)
      - POST /refine                           (iterative tune-judge-refine workflow)
      - GET  /pipeline/{session_id}/events     (SSE stream: progress/done/ping)
      - GET  /pipeline/{session_id}/status     (polling status)
      - GET  /pipeline/{session_id}/results    (final results)
      - GET  /{predict,tune,refine}/{session_id}/events
      - GET  /{predict,tune,refine}/{session_id}/status
      - GET  /{predict,tune,refine}/{session_id}/results
      - POST /pipeline/{session_id}/cancel     (cancel run)
      - POST /pipeline/{session_id}/regenerate (re-run specific steps)
      - GET  /artifacts/{session_id}/predictions (predictions JSONL)
      - GET  /artifacts/{session_id}/report    (HTML report)
      - GET  /artifacts/{session_id}/dataset   (dataset JSONL)
      - GET  /artifacts/{session_id}/output-usd (simulation-ready USD)
      - GET  /tune/{session_id}/artifacts/{name}
      - GET  /refine/{session_id}/artifacts/{name}
      - GET  /sessions                         (list sessions)
      - DELETE /sessions/{session_id}          (delete session)
      - GET  /health                           (service health)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout_seconds: int = 600,
        token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._token = token or os.getenv("PHYSICS_AGENT_TOKEN")
        self._http = requests.Session()
        self._http.headers.update({"User-Agent": "physics-agent-client/2.0"})
        if self._token:
            self._http.headers.update({"Authorization": f"Bearer {self._token}"})

    def _session_url(self, family: str, session_id: str, suffix: str) -> str:
        family = _validate_family(family)
        return f"{self.base_url}/{family}/{session_id}/{suffix}"

    # -------- Core operations
    def upload_usd(
        self, usd_path: str | None = None, *, s3_uri: str | None = None
    ) -> str:
        """Upload a USD file (or reference one on S3) and create a session.

        Args:
            usd_path: Path to USD file on disk.
            s3_uri: S3 URI to a USD file (e.g. ``s3://bucket/path/scene.usdz``).
                The service downloads it server-side — useful for large files.

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
                    (os.path.basename(usd_path), f, "application/octet-stream"),
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
        user_prompt: str | None = None,
        render_backend: str | None = None,
        optimize_usd: bool = False,
        enable_deinstance: bool = True,
        enable_split: bool = False,
        enable_deduplicate: bool = False,
    ) -> str:
        """
        Start the pipeline by uploading a USD file, referencing S3, or
        referencing an existing session.

        Args:
            session_id: Existing session ID (from upload_usd)
            usd_path: Path to USD file (if not using session_id or s3_uri)
            s3_uri: S3 URI to a USD file (service downloads server-side)
            user_prompt: Optional user prompt override
            render_backend: Rendering backend name validated by the server. If
                None, uses the server default (the bundled compose defaults to
                "remote").
            optimize_usd: Enable Scene Optimizer before rendering/prediction.
            enable_deinstance: Enable deinstance when optimize_usd is true.
            enable_split: Enable split meshes when optimize_usd is true.
            enable_deduplicate: Enable deduplicate when optimize_usd is true.

        Returns the session_id of the started run.
        """
        url = f"{self.base_url}/pipeline"
        files: list[tuple[str, tuple[str, object, str]]] = []
        file_handles: list = []

        if usd_path:
            uf = open(usd_path, "rb")
            file_handles.append(uf)
            files.append(
                (
                    "usd_file",
                    (os.path.basename(usd_path), uf, "application/octet-stream"),
                )
            )

        data: dict[str, str] = {}
        if session_id:
            data["session_id"] = session_id
        if s3_uri:
            data["s3_uri"] = s3_uri
        if user_prompt:
            data["user_prompt"] = user_prompt
        if render_backend:
            data["render_backend"] = render_backend
        data["optimize_usd"] = str(optimize_usd).lower()
        data["enable_deinstance"] = str(enable_deinstance).lower()
        data["enable_split"] = str(enable_split).lower()
        data["enable_deduplicate"] = str(enable_deduplicate).lower()

        try:
            response = self._http.post(
                url,
                data=data or None,
                files=files or None,
                timeout=self.timeout_seconds,
            )
            if not response.ok:
                logger.error(f"Error {response.status_code}: {response.text}")
            response.raise_for_status()
            result = response.json()
            return result["session_id"]
        finally:
            for fh in file_handles:
                try:
                    fh.close()
                except Exception:
                    pass

    def start_predict(
        self,
        session_id: str | None = None,
        usd_path: str | None = None,
        s3_uri: str | None = None,
        dataset_path: str | None = None,
        user_prompt: str | None = None,
        render_backend: str | None = None,
        optimize_usd: bool = False,
        enable_deinstance: bool = True,
        enable_split: bool = False,
        enable_deduplicate: bool = False,
    ) -> str:
        """
        Start a prediction-only job via POST /predict.

        Accepts a USD file, S3 URI, existing session_id, or prepared
        dataset_path. Use POST /pipeline when you need apply_physics output.
        """
        primary_sources = [
            name
            for name, value in (
                ("session_id", session_id),
                ("usd_path", usd_path),
                ("s3_uri", s3_uri),
            )
            if value
        ]
        if len(primary_sources) > 1:
            raise ValueError(
                "Provide at most one of session_id, usd_path, or s3_uri "
                f"(got: {', '.join(primary_sources)})"
            )
        if dataset_path and (usd_path or s3_uri):
            raise ValueError(
                "dataset_path may be used alone or with session_id, but not "
                "with usd_path or s3_uri"
            )
        if not primary_sources and not dataset_path:
            raise ValueError(
                "One of session_id, usd_path, s3_uri, or dataset_path is required"
            )

        url = f"{self.base_url}/predict"
        data: dict[str, str] = {}
        if session_id:
            data["session_id"] = session_id
        if s3_uri:
            data["s3_uri"] = s3_uri
        if dataset_path:
            data["dataset_path"] = dataset_path
        if user_prompt:
            data["user_prompt"] = user_prompt
        if render_backend:
            data["render_backend"] = render_backend
        data["optimize_usd"] = _bool_form(optimize_usd)
        data["enable_deinstance"] = _bool_form(enable_deinstance)
        data["enable_split"] = _bool_form(enable_split)
        data["enable_deduplicate"] = _bool_form(enable_deduplicate)

        with ExitStack() as stack:
            files: list[tuple[str, tuple[str, object, str]]] = []
            if usd_path:
                f = stack.enter_context(open(usd_path, "rb"))
                files.append(
                    (
                        "usd_file",
                        (os.path.basename(usd_path), f, "application/octet-stream"),
                    )
                )
            response = self._http.post(
                url,
                data=data,
                files=files or None,
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        return response.json()["session_id"]

    def start_tune(
        self,
        physics_usd_path: str | None = None,
        *,
        s3_uri: str | None = None,
        source_session_id: str | None = None,
        scenario_yaml: str | None = None,
        scenario_yaml_path: str | None = None,
        user_prompt: str | None = None,
        reference_images: Iterable[str] | None = None,
        reference_videos: Iterable[str] | None = None,
        reference_descriptions: Iterable[str] | None = None,
        reference_video_descriptions: Iterable[str] | None = None,
        reference_video_frames: int = 8,
        judge_reference_frames: int = 8,
        judge_generated_frames: int = 16,
        optimizer: str = "auto",
        engine: str = "ovphysx",
        max_trials: int = 30,
        seed: int = 42,
        enable_judge: bool = True,
        judge_max_iterations: int = 3,
        judge_max_tokens: int | None = None,
        judge_temperature: float | None = None,
    ) -> str:
        """
        Start a single-shot physics tuning job via POST /tune.

        Provide exactly one physics USD source: ``physics_usd_path``,
        ``s3_uri``, or ``source_session_id`` from a completed /pipeline run.
        """
        source_count = sum(
            1 for source in (physics_usd_path, s3_uri, source_session_id) if source
        )
        if source_count != 1:
            raise ValueError(
                "Exactly one of physics_usd_path, s3_uri, or source_session_id "
                "must be provided"
            )

        scenario_text = _read_text_arg(
            value=scenario_yaml,
            path=scenario_yaml_path,
            name="scenario_yaml",
        )
        scenario_payload = (
            scenario_text if scenario_text and scenario_text.strip() else None
        )
        prompt_payload = user_prompt if user_prompt and user_prompt.strip() else None
        if not scenario_payload and not prompt_payload:
            raise ValueError("Either scenario_yaml or user_prompt must be provided")

        reference_video_frames = _validate_visual_frame_count(
            "reference_video_frames",
            reference_video_frames,
        )
        judge_reference_frames = _validate_visual_frame_count(
            "judge_reference_frames",
            judge_reference_frames,
        )
        judge_generated_frames = _validate_visual_frame_count(
            "judge_generated_frames",
            judge_generated_frames,
        )

        url = f"{self.base_url}/tune"
        data: dict[str, str] = {
            "optimizer": optimizer,
            "engine": engine,
            "max_trials": str(max_trials),
            "seed": str(seed),
            "enable_judge": _bool_form(enable_judge),
            "judge_max_iterations": str(judge_max_iterations),
            "reference_video_frames": str(reference_video_frames),
            "judge_reference_frames": str(judge_reference_frames),
            "judge_generated_frames": str(judge_generated_frames),
        }
        if s3_uri:
            data["s3_uri"] = s3_uri
        if source_session_id:
            data["source_session_id"] = source_session_id
        if scenario_payload:
            data["scenario_yaml"] = scenario_payload
        if prompt_payload:
            data["user_prompt"] = prompt_payload
        if judge_max_tokens is not None:
            data["judge_max_tokens"] = str(judge_max_tokens)
        if judge_temperature is not None:
            data["judge_temperature"] = str(judge_temperature)
        descriptions = _json_array_arg(reference_descriptions)
        if descriptions is not None:
            data["reference_descriptions"] = descriptions
        video_descriptions = _json_array_arg(reference_video_descriptions)
        if video_descriptions is not None:
            data["reference_video_descriptions"] = video_descriptions

        with ExitStack() as stack:
            files: list[tuple[str, tuple[str, object, str]]] = []
            if physics_usd_path:
                f = stack.enter_context(open(physics_usd_path, "rb"))
                files.append(
                    (
                        "physics_usd",
                        (
                            os.path.basename(physics_usd_path),
                            f,
                            "application/octet-stream",
                        ),
                    )
                )
            for path in reference_images or ():
                f = stack.enter_context(open(path, "rb"))
                files.append(
                    (
                        "reference_images",
                        (os.path.basename(path), f, "application/octet-stream"),
                    )
                )
            for path in reference_videos or ():
                f = stack.enter_context(open(path, "rb"))
                files.append(
                    (
                        "reference_videos",
                        (os.path.basename(path), f, "application/octet-stream"),
                    )
                )

            response = self._http.post(
                url,
                data=data,
                files=files or None,
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        return response.json()["session_id"]

    def start_refine(
        self,
        physics_usd_path: str | None = None,
        *,
        s3_uri: str | None = None,
        source_session_id: str | None = None,
        scenario_yaml: str | None = None,
        scenario_yaml_path: str | None = None,
        user_prompt: str,
        reference_images: Iterable[str] | None = None,
        reference_videos: Iterable[str] | None = None,
        reference_descriptions: Iterable[str] | None = None,
        reference_video_descriptions: Iterable[str] | None = None,
        reference_video_frames: int = 8,
        judge_reference_frames: int = 8,
        judge_generated_frames: int = 16,
        optimizer: str = "botorch",
        engine: str = "ovphysx",
        max_trials: int = 30,
        max_iterations: int = 5,
        score_threshold: float = 0.9,
        seed: int = 42,
        judge_max_tokens: int | None = None,
        judge_temperature: float | None = None,
        visual_evidence_enabled: bool = True,
        llm_timeout_seconds: float = 180.0,
    ) -> str:
        """
        Start an iterative tune-judge-refine job via POST /refine.

        Provide exactly one physics USD source: ``physics_usd_path``,
        ``s3_uri``, or ``source_session_id`` from a completed /pipeline run.
        """
        source_count = sum(
            1 for source in (physics_usd_path, s3_uri, source_session_id) if source
        )
        if source_count != 1:
            raise ValueError(
                "Exactly one of physics_usd_path, s3_uri, or source_session_id "
                "must be provided"
            )
        scenario_text = _read_text_arg(
            value=scenario_yaml,
            path=scenario_yaml_path,
            name="scenario_yaml",
        )
        if not scenario_text or not scenario_text.strip():
            raise ValueError("scenario_yaml is required for refine")
        if not user_prompt.strip():
            raise ValueError("user_prompt is required for refine")

        reference_video_frames = _validate_visual_frame_count(
            "reference_video_frames",
            reference_video_frames,
        )
        judge_reference_frames = _validate_visual_frame_count(
            "judge_reference_frames",
            judge_reference_frames,
        )
        judge_generated_frames = _validate_visual_frame_count(
            "judge_generated_frames",
            judge_generated_frames,
        )

        url = f"{self.base_url}/refine"
        data: dict[str, str] = {
            "scenario_yaml": scenario_text,
            "user_prompt": user_prompt,
            "optimizer": optimizer,
            "engine": engine,
            "max_trials": str(max_trials),
            "max_iterations": str(max_iterations),
            "score_threshold": str(score_threshold),
            "seed": str(seed),
            "visual_evidence_enabled": _bool_form(visual_evidence_enabled),
            "llm_timeout_seconds": str(llm_timeout_seconds),
            "reference_video_frames": str(reference_video_frames),
            "judge_reference_frames": str(judge_reference_frames),
            "judge_generated_frames": str(judge_generated_frames),
        }
        if s3_uri:
            data["s3_uri"] = s3_uri
        if source_session_id:
            data["source_session_id"] = source_session_id
        if judge_max_tokens is not None:
            data["judge_max_tokens"] = str(judge_max_tokens)
        if judge_temperature is not None:
            data["judge_temperature"] = str(judge_temperature)
        descriptions = _json_array_arg(reference_descriptions)
        if descriptions is not None:
            data["reference_descriptions"] = descriptions
        video_descriptions = _json_array_arg(reference_video_descriptions)
        if video_descriptions is not None:
            data["reference_video_descriptions"] = video_descriptions

        with ExitStack() as stack:
            files: list[tuple[str, tuple[str, object, str]]] = []
            if physics_usd_path:
                f = stack.enter_context(open(physics_usd_path, "rb"))
                files.append(
                    (
                        "physics_usd",
                        (
                            os.path.basename(physics_usd_path),
                            f,
                            "application/octet-stream",
                        ),
                    )
                )
            for path in reference_images or ():
                f = stack.enter_context(open(path, "rb"))
                files.append(
                    (
                        "reference_images",
                        (os.path.basename(path), f, "application/octet-stream"),
                    )
                )
            for path in reference_videos or ():
                f = stack.enter_context(open(path, "rb"))
                files.append(
                    (
                        "reference_videos",
                        (os.path.basename(path), f, "application/octet-stream"),
                    )
                )
            response = self._http.post(
                url,
                data=data,
                files=files or None,
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        return response.json()["session_id"]

    def regenerate(
        self,
        session_id: str,
        steps: list[str],
        user_prompt: str | None = None,
    ) -> dict:
        """
        Re-run specific pipeline steps.

        Args:
            session_id: Session to regenerate
            steps: List of step names to re-run
            user_prompt: Optional prompt override

        Returns the response JSON.
        """
        url = f"{self.base_url}/pipeline/{session_id}/regenerate"
        body: dict = {"steps": steps}
        if user_prompt:
            body["user_prompt"] = user_prompt
        resp = self._http.post(url, json=body, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    # -------- Monitoring and results
    def stream_events(
        self,
        session_id: str,
        request_timeout: int | None = None,
        *,
        family: RouteFamily = "pipeline",
    ) -> Generator[SSEMessage, None, None]:
        """
        Connect to the SSE endpoint and yield parsed SSEMessage objects as they arrive.
        This method handles basic SSE parsing without external dependencies.
        """
        url = self._session_url(family, session_id, "events")
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
                msg = SSEMessage(
                    event=buffer_event or "message",
                    data=data_str,
                    id=buffer_id,
                    retry=buffer_retry,
                )
                return msg

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

    def get_status(self, session_id: str, *, family: RouteFamily = "pipeline") -> dict:
        url = self._session_url(family, session_id, "status")
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def get_results(self, session_id: str, *, family: RouteFamily = "pipeline") -> dict:
        url = self._session_url(family, session_id, "results")
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def get_event_log(self, session_id: str) -> dict:
        url = f"{self.base_url}/pipeline/{session_id}/event-log"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def cancel(self, session_id: str, *, family: RouteFamily = "pipeline") -> None:
        url = self._session_url(family, session_id, "cancel")
        resp = self._http.post(url, timeout=self.timeout_seconds)
        resp.raise_for_status()

    def stream_predict_events(
        self, session_id: str, request_timeout: int | None = None
    ) -> Generator[SSEMessage, None, None]:
        yield from self.stream_events(
            session_id, request_timeout=request_timeout, family="predict"
        )

    def get_predict_status(self, session_id: str) -> dict:
        return self.get_status(session_id, family="predict")

    def get_predict_results(self, session_id: str) -> dict:
        return self.get_results(session_id, family="predict")

    def cancel_predict(self, session_id: str) -> None:
        self.cancel(session_id, family="predict")

    def stream_tune_events(
        self, session_id: str, request_timeout: int | None = None
    ) -> Generator[SSEMessage, None, None]:
        yield from self.stream_events(
            session_id, request_timeout=request_timeout, family="tune"
        )

    def get_tune_status(self, session_id: str) -> dict:
        return self.get_status(session_id, family="tune")

    def get_tune_results(self, session_id: str) -> dict:
        return self.get_results(session_id, family="tune")

    def cancel_tune(self, session_id: str) -> None:
        self.cancel(session_id, family="tune")

    def stream_refine_events(
        self, session_id: str, request_timeout: int | None = None
    ) -> Generator[SSEMessage, None, None]:
        yield from self.stream_events(
            session_id, request_timeout=request_timeout, family="refine"
        )

    def get_refine_status(self, session_id: str) -> dict:
        return self.get_status(session_id, family="refine")

    def get_refine_results(self, session_id: str) -> dict:
        return self.get_results(session_id, family="refine")

    def cancel_refine(self, session_id: str) -> None:
        self.cancel(session_id, family="refine")

    def download_predictions(self, session_id: str) -> bytes:
        url = f"{self.base_url}/artifacts/{session_id}/predictions"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.content

    def download_report(self, session_id: str) -> str:
        url = f"{self.base_url}/artifacts/{session_id}/report"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.text

    def download_dataset(self, session_id: str) -> bytes:
        url = f"{self.base_url}/artifacts/{session_id}/dataset"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.content

    def download_output_usd(self, session_id: str) -> bytes:
        url = f"{self.base_url}/artifacts/{session_id}/output-usd"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.content

    def download_tune_artifact(self, session_id: str, name: str) -> bytes:
        artifact_name = quote(name, safe="/")
        url = f"{self.base_url}/tune/{session_id}/artifacts/{artifact_name}"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.content

    def download_refine_artifact(self, session_id: str, name: str) -> bytes:
        artifact_name = quote(name, safe="/")
        url = f"{self.base_url}/refine/{session_id}/artifacts/{artifact_name}"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.content

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

    # -------- Convenience workflow
    def run_and_monitor(
        self,
        usd_path: str | None = None,
        s3_uri: str | None = None,
        user_prompt: str | None = None,
        render_backend: str | None = None,
        optimize_usd: bool = False,
        enable_deinstance: bool = True,
        enable_split: bool = False,
        enable_deduplicate: bool = False,
        upload_first: bool = False,
        print_stream: bool = True,
        reconnect_attempts: int = 3,
        reconnect_backoff_seconds: float = 2.0,
    ) -> tuple[str, dict | None]:
        """
        High-level helper that starts the pipeline and monitors it until completion.

        Args:
            usd_path: Path to USD file to process (local).
            s3_uri: S3 URI to a USD file (e.g. ``s3://bucket/path/scene.usdz``).
                The service downloads it server-side.
            user_prompt: Optional user prompt for VLM.
            render_backend: Rendering backend name validated by the server. If
                None, uses the server default (the bundled compose defaults to
                "remote").
            optimize_usd: Enable Scene Optimizer before rendering/prediction.
            enable_deinstance: Enable deinstance when optimize_usd is true.
            enable_split: Enable split meshes when optimize_usd is true.
            enable_deduplicate: Enable deduplicate when optimize_usd is true.
            upload_first: If True, upload USD first via /upload-usd, then start
                pipeline with session_id.  If False, upload USD inline.
            print_stream: Print progress updates to stdout.
            reconnect_attempts: Number of SSE reconnect attempts.
            reconnect_backoff_seconds: Seconds between reconnect attempts.

        Returns (session_id, status_dict_or_none).
        """
        if not usd_path and not s3_uri:
            raise ValueError("Either usd_path or s3_uri must be provided")

        if s3_uri:
            # S3 path — always goes through upload-first to separate download from run
            session_id = self.upload_usd(s3_uri=s3_uri)
            if print_stream:
                print(f"Downloaded USD from S3, session: {session_id}", flush=True)
            session_id = self.start_pipeline(
                session_id=session_id,
                user_prompt=user_prompt,
                render_backend=render_backend,
                optimize_usd=optimize_usd,
                enable_deinstance=enable_deinstance,
                enable_split=enable_split,
                enable_deduplicate=enable_deduplicate,
            )
        elif upload_first:
            session_id = self.upload_usd(usd_path)
            if print_stream:
                print(f"Uploaded USD, session: {session_id}", flush=True)
            session_id = self.start_pipeline(
                session_id=session_id,
                user_prompt=user_prompt,
                render_backend=render_backend,
                optimize_usd=optimize_usd,
                enable_deinstance=enable_deinstance,
                enable_split=enable_split,
                enable_deduplicate=enable_deduplicate,
            )
        else:
            session_id = self.start_pipeline(
                usd_path=usd_path,
                user_prompt=user_prompt,
                render_backend=render_backend,
                optimize_usd=optimize_usd,
                enable_deinstance=enable_deinstance,
                enable_split=enable_split,
                enable_deduplicate=enable_deduplicate,
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
                            msg = message or ""
                            status_line = f"[{step}] {state} overall={overall}% {msg}"
                            print(status_line.rstrip(), flush=True)
                    elif msg.event == "done":
                        saw_done = True
                        break
                if not saw_done:
                    break
            except Exception as e:
                # 503 = intentional cross-instance signal: this pod does not own
                # the session, so SSE will never work. Go straight to polling.
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                is_503 = status_code == 503 or (status_code is None and "503" in str(e))
                if is_503 or attempts_left == 0:
                    if print_stream:
                        reason = "cross-instance (503)" if is_503 else str(e)
                        print(
                            f"SSE unavailable ({reason}), falling back to polling",
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

        # Polling fallback shows step-level progress from the status endpoint.
        if not saw_done:
            if print_stream:
                print("Polling status...", flush=True)
            while True:
                status = self.get_status(session_id)
                st = status.get("status")
                if print_stream:
                    overall = (status.get("overall_progress") or {}).get("percent", "-")
                    current_step = status.get("current_step")
                    if current_step:
                        step_name = current_step.get("name", "")
                        progress = current_step.get("progress", {})
                        msg = progress.get("message", "")
                        cur = progress.get("current")
                        tot = progress.get("total")
                        detail = (
                            f"{cur}/{tot} "
                            if cur is not None and tot is not None
                            else ""
                        )
                        status_line = (
                            f"[{step_name}] running  overall={overall}%  {detail}{msg}"
                        )
                        print(status_line.rstrip(), flush=True)
                    else:
                        print(f"status={st}  overall={overall}%", flush=True)
                if st in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(2)

        try:
            status = self.get_status(session_id)
        except Exception:
            status = None
        return session_id, status


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Physics Agent Service client")
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="Service base URL"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token for Authorization header (or set PHYSICS_AGENT_TOKEN)",
    )
    parser.add_argument(
        "--prompt", default=None, help="Additional guidance for the VLM"
    )
    parser.add_argument(
        "--render-backend",
        default=None,
        help=(
            "Rendering backend name, validated by the server "
            "(default: server default, typically 'remote' in the bundled compose)"
        ),
    )
    parser.add_argument(
        "--upload-first",
        action="store_true",
        help="Upload USD via /upload-usd before starting pipeline",
    )
    parser.add_argument(
        "--optimize-usd",
        action="store_true",
        help="Run Scene Optimizer before rendering/prediction",
    )
    parser.add_argument(
        "--disable-deinstance",
        dest="enable_deinstance",
        action="store_false",
        help="Disable deinstance when --optimize-usd is set",
    )
    parser.set_defaults(enable_deinstance=True)
    parser.add_argument(
        "--enable-split",
        action="store_true",
        help="Enable split meshes when --optimize-usd is set",
    )
    parser.add_argument(
        "--enable-deduplicate",
        action="store_true",
        help="Enable deduplicate when --optimize-usd is set",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Do not print streaming updates"
    )

    # USD source: local file or S3 URI (mutually exclusive)
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
    client = PhysicsAgentClient(base_url=args.base_url, token=args.token)

    session_id, status = client.run_and_monitor(
        usd_path=args.usd_path,
        s3_uri=args.s3_uri,
        user_prompt=args.prompt,
        render_backend=args.render_backend,
        optimize_usd=args.optimize_usd,
        enable_deinstance=args.enable_deinstance,
        enable_split=args.enable_split,
        enable_deduplicate=args.enable_deduplicate,
        upload_first=args.upload_first,
        print_stream=not args.quiet,
    )

    print(f"\nSession: {session_id}")
    if status is not None:
        print(f"Pipeline status: {status['status']}")
        base = client.base_url
        print("\nArtifacts:")
        print(f"- Pipeline Status:    {base}/pipeline/{session_id}/status")
        print(f"- Predictions JSONL:  {base}/artifacts/{session_id}/predictions")
        print(f"- Report HTML:        {base}/artifacts/{session_id}/report")
        print(f"- Dataset JSONL:      {base}/artifacts/{session_id}/dataset")
        print(f"- Output USD:         {base}/artifacts/{session_id}/output-usd")
    else:
        print("No results available yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
