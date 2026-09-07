# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

import geometry_authoring_connectors.command_worker as command_worker_module
from geometry_authoring_connectors import (
    AuthoringRequest,
    CommandAuthoringExecutionBackend,
    ExternalAuthoringWorkerRunner,
    ForgeCadWorkerRunner,
    InvalidProviderResponseError,
    WireSourceBundle,
    WorkerIsolationError,
)

_PROVIDER_SCRIPT = r"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--result", type=Path, required=True)
args = parser.parse_args()
request = json.loads(args.request.read_text(encoding="utf-8"))
geometry = args.output_dir / "model.step"
geometry.write_text(
    "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('worker fixture'),'2;1');\n"
    "ENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n",
    encoding="ascii",
)
parameters = args.output_dir / "parameters.json"
parameters.write_text(
    json.dumps({"width_mm": request.get("parameters", {}).get("width_mm", 40.0)}),
    encoding="utf-8",
)
width = request.get("parameters", {}).get("width_mm", 40.0)
result = {
    "provider_version": "provider-fixture-1",
    "source_revision": "provider-revision-1",
    "units": "millimeter",
    "up_axis": "Z",
    "forward_axis": "+X",
    "handedness": "right",
    "upstream_edit_uri": "https://cad.example.test/document/provider-revision-1",
    "artifacts": [
        {
            "path": str(geometry),
            "filename": "model.step",
            "role": "cad_geometry",
            "media_type": "model/step",
        },
        {
            "path": str(parameters),
            "filename": "parameters.json",
            "role": "supporting_asset",
            "media_type": "application/json",
        },
    ],
    "parts": [{"part_id": "body", "name": "Body", "artifact_filenames": ["model.step"]}],
    "parameters": [{
        "name": "width_mm",
        "value": width,
        "value_type": "number",
        "unit": "mm",
        "minimum": 20.0,
        "maximum": 100.0,
        "step": 1.0,
        "semantic_role": "overall_width",
        "effects": ["exact_geometry"],
        "affects": ["body"],
        "description": "Overall body width",
    }],
    "verification_assertions": [{
        "assertion_id": "provider-build",
        "status": "passed",
        "summary": "Provider completed deterministic construction.",
        "metrics": {"solid_count": 1},
    }],
    "metadata": {
        "request_id": request["request_id"],
        "parameter_units": request.get("parameter_units", {}),
    },
}
args.result.write_text(json.dumps(result), encoding="utf-8")
"""


def test_fixed_command_backend_drives_the_common_external_worker_contract(
    tmp_path: Path,
) -> None:
    script = tmp_path / "provider.py"
    script.write_text(_PROVIDER_SCRIPT, encoding="utf-8")
    backend = CommandAuthoringExecutionBackend(
        (str(Path(sys.executable).resolve()), str(script)),
        isolation_kind="sandboxed-process",
        provider_id="forgecad-http",
    )
    runner = ExternalAuthoringWorkerRunner(
        backend,
        provider_id="forgecad-http",
        provider_label="Authorized ForgeCAD worker",
        returns_native_source=False,
        workspace_parent=tmp_path,
    )

    payload = runner.handle(
        AuthoringRequest(
            request_id="forgecad-parameter-test",
            prompt="Create a 55 mm rigid body.",
            parameters={"width_mm": 55.0},
            parameter_units={"width_mm": "mm"},
            target_formats=("step",),
        ).model_dump(mode="json")
    )
    bundle = WireSourceBundle.model_validate(payload)

    assert bundle.provider_id == "forgecad-http"
    assert bundle.forward_axis == "+X"
    assert bundle.upstream_edit_uri == "https://cad.example.test/document/provider-revision-1"
    assert bundle.parts[0].part_id == "body"
    assert bundle.parameters[0].value == 55.0
    assert bundle.parameters[0].step == 1.0
    assert bundle.parameters[0].effects == ("exact_geometry",)
    assert bundle.verification_assertions[0].status == "passed"
    assert {item.filename for item in bundle.artifacts} == {
        "model.step",
        "parameters.json",
    }
    assert bundle.metadata["worker_isolation"] == "sandboxed-process"
    assert bundle.metadata["parameter_units"] == {"width_mm": "mm"}

    exported = WireSourceBundle.model_validate(
        runner.handle(
            AuthoringRequest(
                request_id="forgecad-export-test",
                operation="export",
                prior_source_revision=bundle.source_revision,
                target_formats=("step",),
            ).model_dump(mode="json")
        )
    )
    assert exported.source_revision == bundle.source_revision


def test_command_backend_rejects_request_selected_executables(tmp_path: Path) -> None:
    script = tmp_path / "provider.py"
    script.write_text(_PROVIDER_SCRIPT, encoding="utf-8")

    try:
        CommandAuthoringExecutionBackend(
            ("python", str(script)),
            isolation_kind="sandboxed-process",
        )
    except ValueError as exc:
        assert "absolute path" in str(exc)
    else:
        raise AssertionError("relative worker executables must be rejected")


def test_command_backend_rejects_a_preexisting_output_path_as_isolation_failure(
    tmp_path: Path,
) -> None:
    script = tmp_path / "provider.py"
    script.write_text(_PROVIDER_SCRIPT, encoding="utf-8")
    backend = CommandAuthoringExecutionBackend(
        (str(Path(sys.executable).resolve()), str(script)),
        isolation_kind="sandboxed-process",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "output").symlink_to(tmp_path)

    with pytest.raises(WorkerIsolationError, match="must not pre-exist"):
        backend.execute(
            AuthoringRequest(
                request_id="preexisting-output",
                prompt="Create a body.",
                target_formats=("step",),
            ),
            workspace=workspace,
        )


def test_provider_specific_runners_share_the_worker_boundary(tmp_path: Path) -> None:
    script = tmp_path / "provider.py"
    script.write_text(_PROVIDER_SCRIPT, encoding="utf-8")
    backend = CommandAuthoringExecutionBackend(
        (str(Path(sys.executable).resolve()), str(script)),
        isolation_kind="container",
    )

    try:
        ForgeCadWorkerRunner(backend, automated_use_authorized=False)
    except WorkerIsolationError as exc:
        assert "authorization" in str(exc)
    else:
        raise AssertionError("ForgeCAD worker must fail closed without authorization")


def test_command_backend_does_not_inherit_the_geometry_agent_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "environment.py"
    script.write_text(
        _PROVIDER_SCRIPT.replace(
            'request = json.loads(args.request.read_text(encoding="utf-8"))',
            'request = json.loads(args.request.read_text(encoding="utf-8"))\n'
            'assert "GEOMETRY_AGENT_PRIVATE_TOKEN" not in __import__("os").environ',
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GEOMETRY_AGENT_PRIVATE_TOKEN", "must-not-cross")
    backend = CommandAuthoringExecutionBackend(
        (str(Path(sys.executable).resolve()), str(script)),
        isolation_kind="container",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = backend.execute(
        AuthoringRequest(
            request_id="environment-boundary",
            prompt="Create a body.",
            target_formats=("step",),
        ),
        workspace=workspace,
    )

    assert result.provider_version == "provider-fixture-1"


def test_command_backend_reserves_isolation_owned_environment(tmp_path: Path) -> None:
    script = tmp_path / "provider.py"
    script.write_text(_PROVIDER_SCRIPT, encoding="utf-8")

    with pytest.raises(ValueError, match="isolation-owned"):
        CommandAuthoringExecutionBackend(
            (str(Path(sys.executable).resolve()), str(script)),
            isolation_kind="container",
            environment={"HOME": "/operator-selected-home"},
        )


def test_command_backend_rejects_result_manifest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "provider.py"
    script.write_text(_PROVIDER_SCRIPT, encoding="utf-8")
    backend = CommandAuthoringExecutionBackend(
        (str(Path(sys.executable).resolve()), str(script)),
        isolation_kind="container",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result_path = workspace / "authoring-result.json"
    real_fstat = os.fstat
    calls = 0

    def drifting_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 2:
            content = result_path.read_bytes()
            result_path.write_bytes(content)
            current = real_fstat(descriptor)
            os.utime(
                result_path,
                ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000),
            )
        return real_fstat(descriptor)

    monkeypatch.setattr(command_worker_module.os, "fstat", drifting_fstat)

    with pytest.raises(InvalidProviderResponseError, match="changed while being read"):
        backend.execute(
            AuthoringRequest(
                request_id="manifest-drift",
                prompt="Create a body.",
                target_formats=("step",),
            ),
            workspace=workspace,
        )


def test_command_backend_terminates_provider_descendants_after_success(
    tmp_path: Path,
) -> None:
    script = tmp_path / "descendant.py"
    descendant_code = (
        "import time\n"
        "from pathlib import Path\n"
        "time.sleep(0.25)\n"
        "Path('orphan-marker').write_text('survived', encoding='utf-8')\n"
    )
    script.write_text(
        _PROVIDER_SCRIPT.replace(
            "import json",
            "import json\nimport subprocess\nimport sys",
        ).replace(
            'request = json.loads(args.request.read_text(encoding="utf-8"))',
            'request = json.loads(args.request.read_text(encoding="utf-8"))\n'
            f"subprocess.Popen([sys.executable, '-c', {descendant_code!r}])",
        ),
        encoding="utf-8",
    )
    backend = CommandAuthoringExecutionBackend(
        (str(Path(sys.executable).resolve()), str(script)),
        isolation_kind="container",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    backend.execute(
        AuthoringRequest(
            request_id="descendant-cleanup",
            prompt="Create a body.",
            target_formats=("step",),
        ),
        workspace=workspace,
    )
    time.sleep(0.5)

    assert not (workspace / "orphan-marker").exists()
