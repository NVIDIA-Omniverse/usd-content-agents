# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint Agent Service client v2.

Identical to client.py except the polling fallback shows step-level progress.
When SSE returns 503 (cross-instance with multiple pods), the client falls back
to polling and now prints:

    [build_dataset_usd] running  overall=30%  3/10 Rendering: 3/10

The original client.py is kept for backwards-compatibility regression testing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Generator
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


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


class JointAgentClient:
    """
    Client for the Joint Agent Service.

    Endpoints:
      - POST /pipeline                         (start pipeline; accepts usd_file, s3_uri, or session_id)
      - POST /pipeline/upload-usd              (upload USD file or provide s3_uri, returns session_id)
      - GET  /pipeline/{session_id}/events     (SSE stream: progress/done/ping)
      - GET  /pipeline/{session_id}/status     (polling status)
      - GET  /pipeline/{session_id}/results    (final results)
      - POST /pipeline/{session_id}/cancel     (cancel run)
      - POST /pipeline/{session_id}/regenerate (re-run specific steps)
      - GET  /artifacts/{session_id}/predictions (predictions JSONL)
      - GET  /artifacts/{session_id}/report    (HTML report)
      - GET  /artifacts/{session_id}/dataset   (dataset JSONL)
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
        self._token = token or os.getenv("JOINT_AGENT_TOKEN")
        self._http = requests.Session()
        self._run_ids: dict[str, str] = {}
        self._http.headers.update({"User-Agent": "joint-agent-client/2.0"})
        if self._token:
            self._http.headers.update({"Authorization": f"Bearer {self._token}"})

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
                None, uses the server default.

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
            started_session_id = result["session_id"]
            self._run_ids[started_session_id] = result["run_id"]
            return started_session_id
        finally:
            for fh in file_handles:
                try:
                    fh.close()
                except Exception:
                    pass

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
        result = resp.json()
        self._run_ids[session_id] = result["run_id"]
        return result

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

    def cancel(self, session_id: str, run_id: str | None = None) -> None:
        selected_run_id = (
            run_id if run_id is not None else self._run_ids.get(session_id)
        )
        if selected_run_id is None:
            raise ValueError(
                "run_id is required when this client did not start the pipeline run"
            )
        url = f"{self.base_url}/pipeline/{session_id}/cancel"
        resp = self._http.post(
            url,
            params={"run_id": selected_run_id},
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()

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
                None, uses the server default.
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
            )
        elif upload_first:
            session_id = self.upload_usd(usd_path)
            if print_stream:
                print(f"Uploaded USD, session: {session_id}", flush=True)
            session_id = self.start_pipeline(
                session_id=session_id,
                user_prompt=user_prompt,
                render_backend=render_backend,
            )
        else:
            session_id = self.start_pipeline(
                usd_path=usd_path,
                user_prompt=user_prompt,
                render_backend=render_backend,
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
                # 503 = intentional cross-instance signal: this pod doesn't own
                # the session, so SSE will never work. Go straight to polling.
                is_503 = "503" in str(e)
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

        # Polling fallback — shows step-level progress from the status endpoint.
        # The status response includes overall_progress.percent and current_step
        # with per-step progress details (current, total, percent, message).
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
                        detail = f"{cur}/{tot} " if cur and tot else ""
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
    parser = argparse.ArgumentParser(description="Joint Agent Service client v2")
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="Service base URL"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token for Authorization header (or set JOINT_AGENT_TOKEN)",
    )
    parser.add_argument(
        "--prompt", default=None, help="Additional guidance for the VLM"
    )
    parser.add_argument(
        "--render-backend",
        default=None,
        help=(
            "Rendering backend name, validated by the server "
            "(default: server default, typically 'remote')"
        ),
    )
    parser.add_argument(
        "--upload-first",
        action="store_true",
        help="Upload USD via /upload-usd before starting pipeline",
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
    client = JointAgentClient(base_url=args.base_url, token=args.token)

    session_id, status = client.run_and_monitor(
        usd_path=args.usd_path,
        s3_uri=args.s3_uri,
        user_prompt=args.prompt,
        render_backend=args.render_backend,
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
    else:
        print("No results available yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
