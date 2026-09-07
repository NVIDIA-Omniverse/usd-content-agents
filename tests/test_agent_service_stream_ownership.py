# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent service imports must not own host standard streams."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from world_understanding.utils import logging as logging_utils

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SERVICES = (
    ("geometry_agent_service", "geometry_agent_service.main"),
    ("material_agent_service", "service.main"),
    ("physics_agent_service", "service.main"),
    ("joint_agent_service", "service.main"),
    ("texture_agent_service", "service.main"),
)

_EMBEDDER_SCRIPT = r"""
import gc
import importlib
import io
import os
import sys
import types

from starlette.responses import Response


class EventSourceResponse(Response):
    pass


sse_starlette = types.ModuleType("sse_starlette")
sse_starlette.EventSourceResponse = EventSourceResponse
sys.modules.setdefault("sse_starlette", sse_starlette)

aioboto3 = types.ModuleType("aioboto3")
aioboto3.Session = type("Session", (), {})
sys.modules.setdefault("aioboto3", aioboto3)

aiobotocore = types.ModuleType("aiobotocore")
aiobotocore_config = types.ModuleType("aiobotocore.config")
aiobotocore_config.AioConfig = type("AioConfig", (), {})
sys.modules.setdefault("aiobotocore", aiobotocore)
sys.modules.setdefault("aiobotocore.config", aiobotocore_config)

python_multipart = types.ModuleType("python_multipart")
python_multipart.__version__ = "1.0.0"
sys.modules.setdefault("python_multipart", python_multipart)

cachetools = types.ModuleType("cachetools")
cachetools.TTLCache = type("TTLCache", (dict,), {})
sys.modules.setdefault("cachetools", cachetools)

original_stdout = sys.stdout
original_stderr = sys.stderr
host_stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
host_stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
error = None
try:
    sys.stdout = host_stdout
    sys.stderr = host_stderr
    module = importlib.import_module(os.environ["SERVICE_MODULE"])
    module = importlib.reload(module)
    sys.modules["uvicorn"] = types.SimpleNamespace(run=lambda *args, **kwargs: None)
    if hasattr(module, "uvicorn"):
        module.uvicorn = sys.modules["uvicorn"]
    module.main()
    module.main()
    gc.collect()
    assert sys.stdout is host_stdout
    assert sys.stderr is host_stderr
    assert not host_stdout.closed
    assert not host_stderr.closed
    assert not host_stdout.buffer.closed
    assert not host_stderr.buffer.closed
except BaseException as exc:
    error = exc
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
if error is not None:
    raise error
"""


@pytest.mark.parametrize(("service_root", "service_module"), _SERVICES)
def test_service_import_and_entrypoint_preserve_host_streams(
    service_root: str,
    service_module: str,
) -> None:
    env = os.environ.copy()
    python_path = [
        str(_REPO_ROOT / "apps" / service_root),
        str(_REPO_ROOT),
    ]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    env["SERVICE_MODULE"] = service_module
    env["GEOMETRY_AGENT_SERVICE_API_KEY"] = "test-only"
    completed = subprocess.run(
        [sys.executable, "-c", _EMBEDDER_SCRIPT],
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_windows_service_stream_adaptation_preserves_host_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_stdout = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    host_stderr = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    monkeypatch.setattr(logging_utils.sys, "platform", "win32")
    monkeypatch.setattr(logging_utils.sys, "stdout", host_stdout)
    monkeypatch.setattr(logging_utils.sys, "stderr", host_stderr)

    logging_utils.configure_service_standard_streams()
    logging_utils.configure_service_standard_streams()

    assert logging_utils.sys.stdout is host_stdout
    assert logging_utils.sys.stderr is host_stderr
    assert host_stdout.encoding == "utf-8"
    assert host_stderr.encoding == "utf-8"
    assert not host_stdout.closed
    assert not host_stderr.closed
    assert not host_stdout.buffer.closed
    assert not host_stderr.buffer.closed
