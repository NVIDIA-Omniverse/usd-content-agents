"""Run-scoped Responses relay. Private service; never mount its DB/auth in a lane.

No automatic retries. A disconnect without terminal usage remains incomplete.
TLS/network authorization and the upstream route are qualified before freezing.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import http.client
import json
import os
import ssl
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from common import canonical, read, require
from ledger import Ledger
from request_identity import strict_json


class Events:
    """Incremental SSE parser; accepts split UTF-8 and LF/CRLF boundaries."""
    def __init__(self):
        self.buffer = b""

    def feed(self, data):
        self.buffer += data
        require(len(self.buffer) <= 64 * 1024 * 1024, "Oversized SSE event")
        result = []
        while b"\n\n" in self.buffer.replace(b"\r\n", b"\n"):
            # Normalize only complete received CRLF pairs; a split CR stays put.
            self.buffer = self.buffer.replace(b"\r\n", b"\n")
            frame, self.buffer = self.buffer.split(b"\n\n", 1)
            lines = [x[5:].lstrip() for x in frame.split(b"\n") if x.startswith(b"data:")]
            if lines:
                payload = b"\n".join(lines)
                if payload.strip() not in (b"", b"[DONE]"):
                    result.append(json.loads(payload))
        return result


def upstream_headers(auth_file):
    auth = read(auth_file)
    require(auth.get("auth_mode") == "chatgpt", "Expected qualified ChatGPT authorization")
    tokens = auth.get("tokens", {})
    require(tokens.get("access_token") and tokens.get("account_id"), "Missing upstream credentials")
    return {"Authorization": "Bearer " + tokens["access_token"], "ChatGPT-Account-Id": tokens["account_id"]}


def resolve_token(routes, authorization):
    require(isinstance(authorization, str) and authorization.startswith("Bearer "), "Missing run credential")
    digest = hashlib.sha256(authorization[7:].encode()).hexdigest()
    matches = [r for r in routes if hmac.compare_digest(r["token_sha256"], digest)]
    require(len(matches) == 1, "Invalid run credential")
    return matches[0]["run_id"]


def relay_response(response, is_stream, finish, write, deadline, *, connected=True,
                   clock=time.time, set_timeout=None, on_disconnect=None):
    """Drain accounting after a client disconnect, bounded by the same deadline.

    A downstream disconnect does not cancel an already billable upstream call.
    No data is delivered after the deadline; absence of terminal usage remains
    unknown. This function never retries or starts another upstream request.
    """
    events, raw = Events(), bytearray()
    if not connected and on_disconnect:
        on_disconnect()
    while True:
        remaining = deadline - clock()
        require(remaining > 0, "Wall budget reached during model response")
        if set_timeout:
            set_timeout(min(300, remaining))
        data = response.read1(64 * 1024)
        if not data:
            break
        if is_stream:
            for event in events.feed(data):
                if event.get("type") in ("response.completed", "response.failed", "response.incomplete", "response.cancelled"):
                    finish(event["response"])
        else:
            raw.extend(data)
            require(len(raw) <= 64 * 1024 * 1024, "Oversized response")
        require(clock() < deadline, "Wall budget reached; no postdeadline output")
        if connected:
            try:
                write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                connected = False
                if on_disconnect:
                    on_disconnect()
    if not is_stream:
        finish(json.loads(raw))
    return {"downstream_disconnected": not connected}


def serve(config):
    upstream = urlsplit(config["upstream_base_url"])
    require(upstream.scheme == "https" and upstream.hostname and not upstream.username and not upstream.query,
            "A fixed HTTPS upstream is required")
    require(upstream.hostname == "chatgpt.com" and upstream.path.rstrip("/") == "/backend-api/codex",
            "Only the predeclared Codex Responses upstream is allowed")
    require(os.stat(config["auth_file"]).st_mode & 0o077 == 0, "Upstream auth must be controller-private")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *_args):
            pass  # URLs/auth/body never enter generic access logs.

        def error(self, code, message):
            data = canonical({"error": {"message": message}})
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.error(405, "Only run-scoped POST /v1/responses is supported")

        def do_POST(self):
            request_id = None
            db = Ledger(config["database"])
            connection = None
            headers_sent = False
            try:
                require(self.path in ("/v1/responses", "/responses"), "Unsupported route")
                require(self.headers.get("Transfer-Encoding") is None, "Chunked requests unsupported")
                size = int(self.headers.get("Content-Length", "-1"))
                require(0 < size <= 64 * 1024 * 1024, "Request length invalid")
                self.connection.settimeout(30)
                data = self.rfile.read(size)
                require(len(data) == size, "Truncated request")
                # Private routing file may add a newly reserved run without restart.
                run = resolve_token(read(config["routes_file"])["routes"], self.headers.get("Authorization"))
                body = strict_json(data)
                request_id = db.begin_request(run, body)
                headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if body.get("stream") else "application/json",
                           **upstream_headers(config["auth_file"])}
                # Fixed, qualified product headers only. Never forward arbitrary
                # caller headers, cookies, URLs, or alternate provider credentials.
                headers.update(config.get("qualified_product_headers", {}))
                require(set(config.get("qualified_product_headers", {})) <= {"OpenAI-Beta", "originator", "User-Agent"}, "Unexpected product header")
                connection = http.client.HTTPSConnection(upstream.hostname, upstream.port or 443, timeout=60, context=ssl.create_default_context())
                connection.request("POST", upstream.path.rstrip("/") + "/responses", body=data, headers=headers)
                response = connection.getresponse()
                require(response.status == 200, f"Upstream HTTP {response.status}; usage unknown")
                content_type = response.getheader("Content-Type", "")
                # The qualified Codex upstream returns SSE for stream=true but
                # may omit Content-Type. Request semantics plus strict SSE frame
                # parsing determine the stream; never json.loads the full SSE.
                is_stream = body.get("stream") is True or content_type.partition(";")[0].strip().lower() == "text/event-stream"
                headers_sent = True
                connected = True
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream" if is_stream else content_type or "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    connected = False
                deadline = db.db.execute("SELECT deadline FROM runs WHERE id=?", (run,)).fetchone()[0]
                def write(data):
                    self.wfile.write(data)
                    self.wfile.flush()
                sock = connection.sock or getattr(getattr(response.fp, "raw", None), "_sock", None)
                relay = relay_response(response, is_stream, lambda value: db.finish_request(request_id, value),
                               write, deadline, connected=connected,
                               set_timeout=sock.settimeout if sock else None,
                               on_disconnect=lambda: db.record_relay(request_id, True))
                db.record_relay(request_id, relay['downstream_disconnected'])
                current = db.db.execute("SELECT usage FROM requests WHERE id=?", (request_id,)).fetchone()
                require(current[0] is not None, "Stream ended without terminal usage")
            except Exception as exc:
                # Exception type only: parser exceptions can include response text.
                if request_id:
                    db.failed_transport(request_id, type(exc).__name__ + ": terminal accounting not established")
                if not headers_sent:
                    try:
                        self.error(502 if request_id else 403, "Request refused or incomplete; see controller accounting")
                    except OSError:
                        pass
            finally:
                if connection:
                    connection.close()
                db.db.close()
                self.close_connection = True

    server = ThreadingHTTPServer((config["listen_host"], config["listen_port"]), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    serve(read(args.config))
