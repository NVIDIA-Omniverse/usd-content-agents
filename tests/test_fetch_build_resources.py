# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import shutil
import subprocess
import threading
import zipfile
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_build_resources.sh"


def scene_optimizer_zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for subdir in ("python", "lib", "extraLibs", "usdpy"):
            archive.writestr(f"{subdir}/.keep", "")
    return buffer.getvalue()


@contextlib.contextmanager
def serve_responses(
    responses: list[tuple[int, bytes]],
) -> Iterator[tuple[str, dict[str, int]]]:
    state = {"request_count": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            response_index = min(state["request_count"], len(responses) - 1)
            status, body = responses[response_index]
            state["request_count"] += 1
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/scene_optimizer_core.zip", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextlib.contextmanager
def serve_stalled_response() -> Iterator[tuple[str, dict[str, int]]]:
    state = {"request_count": 0}
    release_response = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            state["request_count"] += 1
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Length", str(1024 * 1024))
            self.end_headers()
            self.wfile.write(b"partial body")
            self.wfile.flush()
            release_response.wait(timeout=10)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/scene_optimizer_core.zip", state
    finally:
        release_response.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def url_sha256(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def local_http_env() -> dict[str, str]:
    env = os.environ.copy()
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    return env


def fetch_url_for_arch(arch: str) -> str:
    env = os.environ.copy()
    env["SO_CORE_ARCH"] = arch
    env["SO_CORE_PRINT_URL_ONLY"] = "1"
    result = subprocess.run(
        [str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_fetch_build_resources_defaults_to_x86_package_for_x86_64() -> None:
    url = fetch_url_for_arch("x86_64")

    assert "github.com/NVIDIA-Omniverse/usd-optimize/releases/download/v1.0.3" in url
    assert "scene_optimizer_core_usd_25.11_py_3.12%401.0.3." in url
    assert "manylinux_2_35_x86_64.release.zip" in url


def test_fetch_build_resources_defaults_to_aarch64_package_for_aarch64() -> None:
    url = fetch_url_for_arch("aarch64")

    assert "github.com/NVIDIA-Omniverse/usd-optimize/releases/download/v1.0.3" in url
    assert "scene_optimizer_core_usd_25.11_py_3.12%401.0.3." in url
    assert "manylinux_2_35_aarch64.release.zip" in url


def test_fetch_build_resources_rejects_unknown_architecture() -> None:
    env = os.environ.copy()
    env["SO_CORE_ARCH"] = "riscv64"
    env["SO_CORE_PRINT_URL_ONLY"] = "1"

    result = subprocess.run(
        [str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "unsupported Scene Optimizer Core architecture: riscv64" in result.stderr


def test_fetch_build_resources_allows_unknown_architecture_with_explicit_url() -> None:
    env = os.environ.copy()
    env["SO_CORE_ARCH"] = "riscv64"
    env["SO_CORE_URL"] = "https://example.invalid/scene_optimizer_core_custom.zip"
    env["SO_CORE_PRINT_URL_ONLY"] = "1"

    result = subprocess.run(
        [str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == env["SO_CORE_URL"]


def test_fetch_build_resources_refetches_wrong_architecture_package(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "scene_optimizer_core"
    for subdir in ("python", "lib", "extraLibs", "usdpy"):
        (existing / subdir).mkdir(parents=True)
    tf_binary = existing / "usdpy" / "pxr" / "Tf" / "_tf.so"
    tf_binary.parent.mkdir(parents=True)
    tf_binary.write_text("not an aarch64 ELF", encoding="utf-8")

    package_zip = tmp_path / "scene_optimizer_core.zip"
    with zipfile.ZipFile(package_zip, "w") as archive:
        for subdir in ("python", "lib", "extraLibs", "usdpy"):
            archive.writestr(f"{subdir}/.keep", "")
        archive.writestr("usdpy/pxr/Tf/_tf.so", "replacement package")

    env = os.environ.copy()
    env["SO_CORE_ARCH"] = "aarch64"
    env["SO_CORE_BUILD_RESOURCES"] = str(tmp_path)
    env["SO_CORE_URL"] = package_zip.as_uri()

    result = subprocess.run(
        [str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "does not match manylinux_2_35_aarch64; refetching" in result.stdout
    marker = (existing / ".so_core_platform").read_text(encoding="utf-8")
    assert "platform=manylinux_2_35_aarch64" in marker
    assert f"url_sha256={url_sha256(package_zip.as_uri())}" in marker
    assert package_zip.as_uri() not in marker


def test_fetch_build_resources_refetches_when_url_changes(tmp_path: Path) -> None:
    existing = tmp_path / "scene_optimizer_core"
    for subdir in ("python", "lib", "extraLibs", "usdpy"):
        (existing / subdir).mkdir(parents=True)
    (existing / ".so_core_platform").write_text(
        "platform=manylinux_2_35_aarch64\n"
        f"url_sha256={url_sha256('https://example.invalid/old_scene_optimizer_core.zip')}\n",
        encoding="utf-8",
    )

    package_zip = tmp_path / "scene_optimizer_core.zip"
    with zipfile.ZipFile(package_zip, "w") as archive:
        for subdir in ("python", "lib", "extraLibs", "usdpy"):
            archive.writestr(f"{subdir}/.keep", "")

    env = os.environ.copy()
    env["SO_CORE_ARCH"] = "aarch64"
    env["SO_CORE_BUILD_RESOURCES"] = str(tmp_path)
    env["SO_CORE_URL"] = package_zip.as_uri()

    result = subprocess.run(
        [str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "does not match manylinux_2_35_aarch64; refetching" in result.stdout
    marker = (existing / ".so_core_platform").read_text(encoding="utf-8")
    assert "platform=manylinux_2_35_aarch64" in marker
    assert f"url_sha256={url_sha256(package_zip.as_uri())}" in marker
    assert package_zip.as_uri() not in marker


def test_fetch_build_resources_recovers_from_transient_http_failures(
    tmp_path: Path,
) -> None:
    responses = [
        (HTTPStatus.SERVICE_UNAVAILABLE, b"temporary failure"),
        (HTTPStatus.SERVICE_UNAVAILABLE, b"temporary failure"),
        (HTTPStatus.OK, scene_optimizer_zip_bytes()),
    ]
    with serve_responses(responses) as (url, state):
        env = local_http_env()
        env["SO_CORE_ARCH"] = "x86_64"
        env["SO_CORE_BUILD_RESOURCES"] = str(tmp_path)
        env["SO_CORE_URL"] = url

        result = subprocess.run(
            [str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    assert result.returncode == 0, result.stderr
    assert state["request_count"] == 3
    installed = tmp_path / "scene_optimizer_core"
    for subdir in ("python", "lib", "extraLibs", "usdpy"):
        assert (installed / subdir).is_dir()


@pytest.mark.parametrize(
    ("help_text", "expect_retry_connrefused"),
    [
        ("--retry-connrefused", True),
        ("curl compatibility help", False),
    ],
)
def test_fetch_build_resources_falls_back_for_older_curl(
    tmp_path: Path,
    help_text: str,
    expect_retry_connrefused: bool,
) -> None:
    real_curl = shutil.which("curl")
    assert real_curl is not None
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl_args_log = tmp_path / "curl-args.log"
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "--help" ]]; then\n'
        f"  printf '%s\\n' '{help_text}'\n"
        "  exit 0\n"
        "fi\n"
        'printf \'%s\\n\' "$@" > "$FAKE_CURL_ARGS_LOG"\n'
        f'exec "{real_curl}" "$@"\n',
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)

    with serve_responses([(HTTPStatus.OK, scene_optimizer_zip_bytes())]) as (
        url,
        _state,
    ):
        env = local_http_env()
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        env["FAKE_CURL_ARGS_LOG"] = str(curl_args_log)
        env["SO_CORE_ARCH"] = "x86_64"
        env["SO_CORE_BUILD_RESOURCES"] = str(tmp_path / "build-resources")
        env["SO_CORE_URL"] = url

        result = subprocess.run(
            [str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    assert result.returncode == 0, result.stderr
    curl_args = curl_args_log.read_text(encoding="utf-8").splitlines()
    assert ("--retry-connrefused" in curl_args) is expect_retry_connrefused
    assert "--retry-all-errors" not in curl_args
    assert curl_args[curl_args.index("--max-time") + 1] == "600"
    assert "using compatible retries" in result.stderr


def test_fetch_build_resources_bounds_permanent_http_failure(tmp_path: Path) -> None:
    with serve_responses([(HTTPStatus.SERVICE_UNAVAILABLE, b"still unavailable")]) as (
        url,
        state,
    ):
        env = local_http_env()
        env["SO_CORE_ARCH"] = "x86_64"
        env["SO_CORE_BUILD_RESOURCES"] = str(tmp_path)
        env["SO_CORE_URL"] = url
        env["SO_CORE_CURL_RETRIES"] = "2"
        env["SO_CORE_CURL_RETRY_MAX_TIME"] = "10"

        result = subprocess.run(
            [str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    assert result.returncode != 0
    assert state["request_count"] == 3
    assert url in result.stdout
    assert "503" in result.stderr
    assert not (tmp_path / "scene_optimizer_core").exists()


def test_fetch_build_resources_bounds_stalled_transfer(tmp_path: Path) -> None:
    with serve_stalled_response() as (url, state):
        env = local_http_env()
        env["SO_CORE_ARCH"] = "x86_64"
        env["SO_CORE_BUILD_RESOURCES"] = str(tmp_path)
        env["SO_CORE_URL"] = url
        env["SO_CORE_CURL_RETRIES"] = "0"
        env["SO_CORE_CURL_RETRY_MAX_TIME"] = "5"
        env["SO_CORE_CURL_SPEED_LIMIT"] = "1024"
        env["SO_CORE_CURL_SPEED_TIME"] = "1"

        result = subprocess.run(
            [str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )

    assert result.returncode == 28
    assert state["request_count"] == 1
    assert "Operation too slow" in result.stderr
    assert not (tmp_path / "scene_optimizer_core").exists()


@pytest.mark.parametrize(
    ("env_name", "value", "expected_error"),
    [
        (
            "SO_CORE_CURL_RETRIES",
            "not-a-number",
            "SO_CORE_CURL_RETRIES must be a non-negative integer",
        ),
        (
            "SO_CORE_CURL_MAX_TIME",
            "0",
            "SO_CORE_CURL_MAX_TIME must be greater than zero",
        ),
    ],
)
def test_fetch_build_resources_rejects_invalid_curl_configuration(
    env_name: str,
    value: str,
    expected_error: str,
) -> None:
    env = os.environ.copy()
    env["SO_CORE_ARCH"] = "x86_64"
    env["SO_CORE_PRINT_URL_ONLY"] = "1"
    env[env_name] = value

    result = subprocess.run(
        [str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_failed_http_refetch_preserves_existing_package(tmp_path: Path) -> None:
    existing = tmp_path / "scene_optimizer_core"
    for subdir in ("python", "lib", "extraLibs", "usdpy"):
        (existing / subdir).mkdir(parents=True)
    sentinel = existing / "usdpy" / "existing.txt"
    sentinel.write_text("existing package is still usable", encoding="utf-8")

    with serve_responses([(HTTPStatus.SERVICE_UNAVAILABLE, b"still unavailable")]) as (
        url,
        state,
    ):
        env = local_http_env()
        env["SO_CORE_ARCH"] = "aarch64"
        env["SO_CORE_BUILD_RESOURCES"] = str(tmp_path)
        env["SO_CORE_URL"] = url
        env["SO_CORE_CURL_RETRIES"] = "1"
        env["SO_CORE_CURL_RETRY_MAX_TIME"] = "5"

        result = subprocess.run(
            [str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    assert result.returncode != 0
    assert state["request_count"] == 2
    assert sentinel.read_text(encoding="utf-8") == "existing package is still usable"
    for subdir in ("python", "lib", "extraLibs", "usdpy"):
        assert (existing / subdir).is_dir()


def test_fetch_build_resources_preserves_existing_package_when_refetch_fails(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "scene_optimizer_core"
    for subdir in ("python", "lib", "extraLibs", "usdpy"):
        (existing / subdir).mkdir(parents=True)
    sentinel = existing / "usdpy" / "existing.txt"
    sentinel.write_text("existing package is still usable", encoding="utf-8")

    env = os.environ.copy()
    env["SO_CORE_ARCH"] = "aarch64"
    env["SO_CORE_BUILD_RESOURCES"] = str(tmp_path)
    env["SO_CORE_URL"] = (tmp_path / "missing_scene_optimizer_core.zip").as_uri()
    env["SO_CORE_CURL_RETRIES"] = "0"

    result = subprocess.run(
        [str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "does not match manylinux_2_35_aarch64; refetching" in result.stdout
    assert sentinel.read_text(encoding="utf-8") == "existing package is still usable"
    for subdir in ("python", "lib", "extraLibs", "usdpy"):
        assert (existing / subdir).is_dir()
