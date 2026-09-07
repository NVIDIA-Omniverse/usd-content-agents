# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from apps.texture_gen_step1x_service import app as step1x_app
from apps.texture_gen_step1x_service import healthcheck


def test_liveness_healthcheck_uses_cheap_livez_endpoint(monkeypatch) -> None:
    requests: list[str] = []

    class FakeResponse:
        status = 200

        def read(self) -> bytes:
            return b'{"status": "healthy"}'

    class FakeConnection:
        def __init__(self, host: str, port: int, timeout: int) -> None:
            assert host == "localhost"
            assert port == 8000
            assert timeout == 5

        def request(self, _method: str, path: str) -> None:
            requests.append(path)

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(healthcheck.http.client, "HTTPConnection", FakeConnection)

    assert healthcheck._healthy(require_ready=False) is True
    assert requests == ["/livez"]


def test_readiness_healthcheck_uses_full_health_endpoint(monkeypatch) -> None:
    requests: list[str] = []
    timeouts: list[int] = []

    class FakeResponse:
        status = 200

        def read(self) -> bytes:
            return json.dumps(
                {
                    "ready": True,
                    "capabilities": {
                        "external_runtime": {
                            "runtime_profiles": {"selected": ["texture-step1x-core"]}
                        }
                    },
                }
            ).encode()

    class FakeConnection:
        def __init__(self, _host: str, _port: int, timeout: int) -> None:
            timeouts.append(timeout)

        def request(self, _method: str, path: str) -> None:
            requests.append(path)

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(healthcheck.http.client, "HTTPConnection", FakeConnection)
    monkeypatch.delenv("TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS", raising=False)

    assert healthcheck._healthy(require_ready=True) is True
    assert requests == ["/health"]
    assert timeouts == [180]


def test_http_timeout_invalid_values_fall_back(monkeypatch) -> None:
    monkeypatch.setenv("TEXTURE_STEP1X_READINESS_HTTP_TIMEOUT_SEC", "bad")
    monkeypatch.setenv("TEXTURE_STEP1X_LIVENESS_HTTP_TIMEOUT_SEC", "also-bad")

    assert healthcheck._http_timeout(require_ready=True) == 180
    assert healthcheck._http_timeout(require_ready=False) == 5


def test_readiness_healthcheck_keeps_health_endpoint_after_preflight_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    requests: list[str] = []
    timeouts: list[int] = []

    class FakeResponse:
        status = 200

        def read(self) -> bytes:
            return json.dumps(
                {
                    "ready": True,
                    "capabilities": {
                        "external_runtime": {
                            "runtime_profiles": {"selected": ["texture-step1x-core"]}
                        }
                    },
                }
            ).encode()

    class FakeConnection:
        def __init__(self, _host: str, _port: int, timeout: int) -> None:
            timeouts.append(timeout)

        def request(self, _method: str, path: str) -> None:
            requests.append(path)

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(healthcheck.http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(healthcheck, "_PREFLIGHT_MARKER", tmp_path / "ok")
    monkeypatch.setenv("TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS", "true")
    monkeypatch.setenv("TEXTURE_STEP1X_PYTHON", "/runtime/bin/python")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/runtime/lib")
    (tmp_path / "ok").write_text(
        healthcheck._runtime_fingerprint() or "",
        encoding="utf-8",
    )

    assert healthcheck._healthy(require_ready=True) is True
    assert requests == ["/health"]
    assert timeouts == [180]


def test_runtime_import_preflight_exercises_cupy_nvrtc_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def fake_run(cmd: list[str], **kwargs):
        calls.append(cmd)
        environments.append(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(healthcheck, "_PREFLIGHT_MARKER", tmp_path / "ok")
    monkeypatch.setenv(
        "TEXTURE_STEP1X_PYTHON", "/opt/texture-editing/.venv_gen/bin/python"
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/texture-editing/.venv_gen/lib")
    monkeypatch.delenv("TEXTURE_STEP1X_SKIP_MA", raising=False)
    monkeypatch.setattr(healthcheck.subprocess, "run", fake_run)

    assert healthcheck._runtime_import_preflight() is True

    assert calls
    assert calls[0][:2] == ["/opt/texture-editing/.venv_gen/bin/python", "-c"]
    probe = calls[0][2]
    assert "import torch" in probe
    assert "import cupy as cp" in probe
    assert '"pymeshlab",' in probe
    assert "importlib.util.find_spec(module_name)" in probe
    assert "step1x3d_geometry" in probe
    assert "cp.cuda.nvrtc.getVersion()" in probe
    assert "cp.sum(values)" in probe
    core_modules = probe.split('if "texture-material-anything" in profiles', 1)[0]
    assert '"kaolin"' not in core_modules
    assert '"kaolin"' in probe
    assert environments[0]["TEXTURE_STEP1X_RUNTIME_PROFILES"] == ("texture-step1x-core")


def test_runtime_import_preflight_selects_profile_from_legacy_flags(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def fake_run(cmd: list[str], **kwargs):
        calls.append(cmd)
        environments.append(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(healthcheck, "_PREFLIGHT_MARKER", tmp_path / "ok")
    monkeypatch.setenv("TEXTURE_STEP1X_PYTHON", "/runtime/bin/python")
    monkeypatch.setenv("TEXTURE_STEP1X_SKIP_MA", "true")
    monkeypatch.setattr(healthcheck.subprocess, "run", fake_run)

    assert healthcheck._runtime_import_preflight() is True

    probe = calls[0][2]
    assert "import torch" in probe
    assert "import cupy as cp" in probe
    assert "texture-material-anything" in probe
    assert "texture-full-pbr-upscale" in probe
    assert environments[0]["TEXTURE_STEP1X_RUNTIME_PROFILES"] == ("texture-step1x-core")


@pytest.mark.parametrize(
    "profile",
    (
        "texture-step1x-core",
        "texture-step1x-core,texture-material-anything",
        "texture-step1x-core,texture-swin2sr",
        "texture-full-pbr-upscale",
    ),
)
def test_runtime_import_preflight_honors_explicit_profile(
    tmp_path: Path,
    monkeypatch,
    profile: str,
) -> None:
    environments: list[dict[str, str]] = []

    def fake_run(_cmd: list[str], **kwargs):
        environments.append(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(healthcheck, "_PREFLIGHT_MARKER", tmp_path / "ok")
    monkeypatch.setenv("TEXTURE_STEP1X_PYTHON", "/runtime/bin/python")
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_PROFILES", profile)
    monkeypatch.setattr(healthcheck.subprocess, "run", fake_run)

    assert healthcheck._runtime_import_preflight() is True
    assert environments[0]["TEXTURE_STEP1X_RUNTIME_PROFILES"] == profile
    pythonpath = environments[0]["PYTHONPATH"].split(os.pathsep)
    assert pythonpath[:2] == [
        str(tmp_path / "runtime" / "src"),
        str(tmp_path / "runtime" / "third_party" / "Step1X-3D"),
    ]
    if "material-anything" in profile or "full-pbr" in profile:
        assert pythonpath[2] == str(
            tmp_path / "runtime" / "third_party" / "MaterialAnything"
        )


@pytest.mark.parametrize(
    ("marker_text", "error_code"),
    (
        ("not-json\n", "STEP1X_RUNTIME_MARKER_INVALID"),
        (
            '{"runtime_source":"compose_managed"}\n',
            "STEP1X_RUNTIME_PROFILES_MISSING_FROM_MARKER",
        ),
        (
            '{"runtime_source":"compose_managed","runtime_profiles":"core"}\n',
            "STEP1X_RUNTIME_PROFILES_INVALID_IN_MARKER",
        ),
    ),
)
def test_runtime_import_preflight_rejects_invalid_present_runtime_marker(
    tmp_path: Path,
    monkeypatch,
    capsys,
    marker_text: str,
    error_code: str,
) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / ".texture-agent-runtime.json").write_text(
        marker_text,
        encoding="utf-8",
    )
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("TEXTURE_STEP1X_PYTHON", "/runtime/bin/python")
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_PROFILES", "texture-step1x-core")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        healthcheck.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command),
    )

    assert healthcheck._runtime_import_preflight() is False
    assert calls == []
    assert error_code in capsys.readouterr().err


def test_runtime_import_preflight_rejects_unknown_profile_without_subprocess(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TEXTURE_STEP1X_PYTHON", "/runtime/bin/python")
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_PROFILES", "texture-mystery")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        healthcheck.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command),
    )

    assert healthcheck._runtime_import_preflight() is False
    assert calls == []


def test_runtime_import_preflight_uses_success_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(healthcheck, "_PREFLIGHT_MARKER", tmp_path / "ok")
    monkeypatch.setenv("TEXTURE_STEP1X_PYTHON", "/runtime/bin/python")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/runtime/lib")
    monkeypatch.setattr(healthcheck.subprocess, "run", fake_run)

    assert healthcheck._runtime_import_preflight() is True
    assert healthcheck._runtime_import_preflight() is True

    assert len(calls) == 1


def test_runtime_import_preflight_fails_without_python(monkeypatch) -> None:
    monkeypatch.delenv("TEXTURE_STEP1X_PYTHON", raising=False)
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_DIR", "/definitely/missing")

    assert healthcheck._runtime_import_preflight() is False


def test_runtime_import_preflight_rejects_missing_fingerprint(monkeypatch) -> None:
    monkeypatch.setenv("TEXTURE_STEP1X_PYTHON", "/runtime/bin/python")
    monkeypatch.setattr(healthcheck, "_runtime_fingerprint", lambda: None)

    assert healthcheck._runtime_import_preflight() is False


def test_runtime_fingerprint_missing_without_runtime_python(monkeypatch) -> None:
    monkeypatch.delenv("TEXTURE_STEP1X_PYTHON", raising=False)
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_DIR", "/definitely/missing")

    assert healthcheck._runtime_fingerprint() is None
    assert healthcheck._runtime_preflight_marker_matches() is False


def test_configured_runtime_python_prefers_explicit_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    explicit = tmp_path / "python"
    monkeypatch.setenv("TEXTURE_STEP1X_PYTHON", str(explicit))

    assert healthcheck._configured_runtime_python() == str(explicit)


def test_configured_runtime_python_discovers_runtime_venvs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("TEXTURE_STEP1X_PYTHON", raising=False)
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_DIR", str(tmp_path))
    venv = tmp_path / ".venv" / "bin" / "python"
    venv.parent.mkdir(parents=True)
    venv.write_text("#!/bin/sh\n", encoding="utf-8")

    assert healthcheck._configured_runtime_python() == str(venv)

    venv_gen = tmp_path / ".venv_gen" / "bin" / "python"
    venv_gen.parent.mkdir(parents=True)
    venv_gen.write_text("#!/bin/sh\n", encoding="utf-8")
    assert healthcheck._configured_runtime_python() == str(venv_gen)


def test_runtime_import_preflight_reports_subprocess_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    def fake_run(cmd: list[str], **kwargs: object):
        assert kwargs["timeout"] == 180
        return SimpleNamespace(returncode=2, stdout="out", stderr="err")

    monkeypatch.setattr(healthcheck, "_PREFLIGHT_MARKER", tmp_path / "ok")
    monkeypatch.setenv("TEXTURE_STEP1X_PYTHON", "/runtime/bin/python")
    monkeypatch.setattr(healthcheck.subprocess, "run", fake_run)

    assert healthcheck._runtime_import_preflight() is False
    captured = capsys.readouterr()
    assert "Step1X runtime import preflight failed" in captured.err
    assert "out" in captured.err
    assert "err" in captured.err


def test_runtime_import_preflight_invalid_timeout_falls_back(
    tmp_path: Path,
    monkeypatch,
) -> None:
    timeouts: list[int] = []

    def fake_run(_cmd: list[str], **kwargs: object):
        timeouts.append(kwargs["timeout"])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(healthcheck, "_PREFLIGHT_MARKER", tmp_path / "ok")
    monkeypatch.setenv("TEXTURE_STEP1X_PYTHON", "/runtime/bin/python")
    monkeypatch.setenv("TEXTURE_STEP1X_PREFLIGHT_TIMEOUT_SEC", "bad")
    monkeypatch.setattr(healthcheck.subprocess, "run", fake_run)

    assert healthcheck._runtime_import_preflight() is True
    assert timeouts == [180]


def test_runtime_import_preflight_ignores_marker_write_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run(_cmd: list[str], **_kwargs: object):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fail_write(path: Path, *_args: object, **_kwargs: object) -> int:
        if path == tmp_path / "ok":
            raise OSError("read-only marker")
        return original_write_text(path, *_args, **_kwargs)

    original_write_text = Path.write_text
    monkeypatch.setattr(healthcheck, "_PREFLIGHT_MARKER", tmp_path / "ok")
    monkeypatch.setenv("TEXTURE_STEP1X_PYTHON", "/runtime/bin/python")
    monkeypatch.setattr(healthcheck.subprocess, "run", fake_run)
    monkeypatch.setattr(Path, "write_text", fail_write)

    assert healthcheck._runtime_import_preflight() is True


def test_healthy_rejects_non_2xx_and_not_ready(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self, status: int, body: bytes) -> None:
            self.status = status
            self._body = body

        def read(self) -> bytes:
            return self._body

    class FakeConnection:
        responses = [
            FakeResponse(503, b'{"ready": true}'),
            FakeResponse(200, b'{"ready": false}'),
            FakeResponse(
                200,
                json.dumps(
                    {
                        "ready": True,
                        "capabilities": {
                            "external_runtime": {
                                "runtime_profiles": {
                                    "selected": ["texture-step1x-core"]
                                }
                            }
                        },
                    }
                ).encode(),
            ),
        ]

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def request(self, _method: str, _path: str) -> None:
            pass

        def getresponse(self) -> FakeResponse:
            return self.responses.pop(0)

    monkeypatch.setattr(healthcheck.http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setenv("TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS", "yes")
    monkeypatch.setattr(healthcheck, "_runtime_import_preflight", lambda: True)

    assert healthcheck._healthy(require_ready=True) is False
    assert healthcheck._healthy(require_ready=True) is False
    assert healthcheck._healthy(require_ready=True) is True


def test_readiness_requires_service_to_report_explicit_selected_profile(
    monkeypatch,
) -> None:
    class FakeResponse:
        status = 200

        def __init__(self, selected: list[str]) -> None:
            self.selected = selected

        def read(self) -> bytes:
            return json.dumps(
                {
                    "ready": True,
                    "capabilities": {
                        "external_runtime": {
                            "runtime_profiles": {"selected": self.selected}
                        }
                    },
                }
            ).encode()

    class FakeConnection:
        responses = iter(
            [
                FakeResponse(["texture-step1x-core"]),
                FakeResponse(["texture-full-pbr-upscale"]),
            ]
        )

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def request(self, _method: str, _path: str) -> None:
            pass

        def getresponse(self) -> FakeResponse:
            return next(self.responses)

    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_PROFILES", "texture-full-pbr-upscale")
    monkeypatch.delenv("TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS", raising=False)
    monkeypatch.setattr(healthcheck.http.client, "HTTPConnection", FakeConnection)

    assert healthcheck._healthy(require_ready=True) is False
    assert healthcheck._healthy(require_ready=True) is True


def test_readiness_requires_service_to_report_legacy_selected_profile(
    monkeypatch,
) -> None:
    class FakeResponse:
        status = 200

        def __init__(self, selected: list[str]) -> None:
            self.selected = selected

        def read(self) -> bytes:
            return json.dumps(
                {
                    "ready": True,
                    "capabilities": {
                        "external_runtime": {
                            "runtime_profiles": {"selected": self.selected}
                        }
                    },
                }
            ).encode()

    class FakeConnection:
        responses = iter(
            [
                FakeResponse(["texture-full-pbr-upscale"]),
                FakeResponse(["texture-step1x-core"]),
            ]
        )

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def request(self, _method: str, _path: str) -> None:
            pass

        def getresponse(self) -> FakeResponse:
            return next(self.responses)

    monkeypatch.delenv("TEXTURE_STEP1X_RUNTIME_PROFILES", raising=False)
    monkeypatch.delenv("TEXTURE_STEP1X_RUNTIME_PROFILE", raising=False)
    monkeypatch.setenv("TEXTURE_STEP1X_SKIP_MA", "true")
    monkeypatch.setenv("TEXTURE_STEP1X_REQUIRE_UPSCALER", "false")
    monkeypatch.delenv("TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS", raising=False)
    monkeypatch.setattr(healthcheck.http.client, "HTTPConnection", FakeConnection)

    assert healthcheck._healthy(require_ready=True) is False
    assert healthcheck._healthy(require_ready=True) is True


def test_healthcheck_main_maps_success_and_exceptions(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["healthcheck.py", "--liveness"])
    monkeypatch.setattr(
        healthcheck, "_healthy", lambda require_ready: not require_ready
    )
    assert healthcheck.main() == 0

    monkeypatch.setattr(sys, "argv", ["healthcheck.py"])
    monkeypatch.setattr(healthcheck, "_healthy", lambda require_ready: False)
    assert healthcheck.main() == 1

    def fail(_require_ready: bool) -> bool:
        raise RuntimeError("boom")

    monkeypatch.setattr(healthcheck, "_healthy", fail)
    assert healthcheck.main() == 1


def test_app_env_helpers_use_configured_and_fallback_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TEXTURE_OUTPUT_DIR", str(tmp_path / "out"))
    assert step1x_app._output_dir() == tmp_path / "out"

    monkeypatch.setenv("TEXTURE_OUTPUT_DIR", " ")
    assert step1x_app._output_dir().name == "texture_gen_step1x_service"

    monkeypatch.setenv("TEXTURE_STEP1X_MAX_WORKERS", "3")
    assert step1x_app._max_workers() == 3
    monkeypatch.setenv("TEXTURE_STEP1X_MAX_WORKERS", "0")
    assert step1x_app._max_workers() == 1
    monkeypatch.setenv("TEXTURE_STEP1X_MAX_WORKERS", "bad")
    assert step1x_app._max_workers() == 1


def test_runtime_marker_profiles_select_valid_and_legacy_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    marker_path = runtime_dir / ".texture-agent-runtime.json"
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.delenv("TEXTURE_STEP1X_RUNTIME_PROFILES", raising=False)
    monkeypatch.delenv("TEXTURE_STEP1X_RUNTIME_PROFILE", raising=False)

    marker_path.write_text(
        json.dumps(
            {
                "runtime_source": "compose_managed",
                "runtime_profiles": ["texture-step1x-core"],
            }
        ),
        encoding="utf-8",
    )
    assert healthcheck._runtime_marker_profiles() == ("texture-step1x-core",)
    assert healthcheck._selected_runtime_profiles() == (
        ("texture-step1x-core",),
        "runtime_marker",
    )

    marker_path.write_text(
        json.dumps({"runtime_profile": "texture-step1x-core"}),
        encoding="utf-8",
    )
    assert healthcheck._runtime_marker_profiles() == ("texture-step1x-core",)


@pytest.mark.parametrize(
    "profiles",
    ([], [""], ["   "], [1], "texture-step1x-core"),
)
def test_runtime_marker_rejects_any_present_invalid_profile_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profiles: object,
) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / ".texture-agent-runtime.json").write_text(
        json.dumps(
            {
                "runtime_source": "operator_mounted",
                "runtime_profiles": profiles,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_PROFILES", "texture-step1x-core")

    with pytest.raises(
        healthcheck._RuntimeMarkerError,
        match="STEP1X_RUNTIME_PROFILES_INVALID_IN_MARKER",
    ):
        healthcheck._selected_runtime_profiles()


def test_runtime_probe_pythonpath_appends_configured_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    configured = (tmp_path / "one", tmp_path / "two")
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(str(path) for path in configured))

    pythonpath = healthcheck._runtime_probe_pythonpath(("texture-step1x-core",)).split(
        os.pathsep
    )

    assert pythonpath[-2:] == [str(path) for path in configured]


def test_selected_runtime_profiles_honors_legacy_explicit_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TEXTURE_STEP1X_RUNTIME_PROFILES", raising=False)
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_PROFILE", "texture-step1x-core")

    assert healthcheck._selected_runtime_profiles() == (
        ("texture-step1x-core",),
        "explicit",
    )


@pytest.mark.parametrize(
    ("skip_ma", "require_upscaler", "expected_profiles"),
    (
        ("false", "true", ("texture-full-pbr-upscale",)),
        (
            "false",
            "false",
            ("texture-step1x-core", "texture-material-anything"),
        ),
        ("true", "true", ("texture-step1x-core", "texture-swin2sr")),
    ),
)
def test_selected_runtime_profiles_maps_legacy_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    skip_ma: str,
    require_upscaler: str,
    expected_profiles: tuple[str, ...],
) -> None:
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TEXTURE_STEP1X_RUNTIME_PROFILES", raising=False)
    monkeypatch.delenv("TEXTURE_STEP1X_RUNTIME_PROFILE", raising=False)
    monkeypatch.setenv("TEXTURE_STEP1X_SKIP_MA", skip_ma)
    monkeypatch.setenv("TEXTURE_STEP1X_REQUIRE_UPSCALER", require_upscaler)

    assert healthcheck._selected_runtime_profiles() == (
        expected_profiles,
        "legacy_flags",
    )


def test_marker_errors_fail_closed_in_preflight_and_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_marker_error() -> None:
        raise healthcheck._RuntimeMarkerError("invalid marker")

    monkeypatch.setattr(healthcheck, "_runtime_fingerprint", raise_marker_error)
    assert healthcheck._runtime_preflight_marker_matches() is False

    monkeypatch.setattr(
        healthcheck,
        "_http_health",
        lambda _path, *, require_ready: (200, {"ready": require_ready}),
    )
    monkeypatch.setattr(healthcheck, "_selected_runtime_profiles", raise_marker_error)
    assert healthcheck._healthy(require_ready=True) is False


def test_healthcheck_script_entrypoint_exits_with_main_result(monkeypatch) -> None:
    class FakeResponse:
        status = 200

        def read(self) -> bytes:
            return b'{"status": "healthy"}'

    class FakeConnection:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def request(self, _method: str, _path: str) -> None:
            pass

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(sys, "argv", [str(Path(healthcheck.__file__)), "--liveness"])
    monkeypatch.setattr(healthcheck.http.client, "HTTPConnection", FakeConnection)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(healthcheck.__file__)), run_name="__main__")
    assert exc_info.value.code == 0
