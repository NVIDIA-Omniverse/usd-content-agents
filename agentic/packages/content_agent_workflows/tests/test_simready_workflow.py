# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SimReady Foundation workflow adapters."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

import content_agent_workflows.simready.conform_profile as conform_profile_module
import content_agent_workflows.simready.foundation_runtime as foundation_runtime_module
import content_agent_workflows.simready.validate_profile as validate_profile_module
from content_agent_workflows.simready import (
    DEFAULT_SIMREADY_FOUNDATION_COMMIT,
    DEFAULT_SIMREADY_FOUNDATION_REF,
    SimReadyConformanceInput,
    SimReadyValidationInput,
    preflight_simready_foundation,
    run_simready_profile_conformance,
    run_simready_profile_validation,
)
from content_agent_workflows.simready.foundation_runtime import (
    SIMREADY_CACHE_DIR_ENV,
    SIMREADY_FOUNDATION_REF_ENV,
    SIMREADY_USD_CORE_OVERRIDE,
    SIMREADY_USD_EXCHANGE_REQUIREMENT,
    SIMREADY_USD_PROVIDER_ENV,
    _acquire_venv_lock,
    _install_command,
    _prepare_validation_venv,
)
from content_agent_workflows.simready.models import (
    SIMREADY_GRASP_PLAN_SCHEMA_VERSION,
    default_simready_report_name,
)


def _write_fake_foundation(tmp_path: Path, *, skill_layout: str = "legacy") -> Path:
    root = tmp_path / "simready-foundation"
    spec_root = root / "nv_core" / "sr_specs" / "docs"
    (spec_root / "capabilities").mkdir(parents=True)
    (spec_root / "features").mkdir(parents=True)
    (spec_root / "profiles").mkdir(parents=True)
    (spec_root / "profiles" / "profiles.toml").write_text(
        '[Prop-Robotics-Neutral]\n"1.0.0" = { features = [] }\n',
        encoding="utf-8",
    )
    for skill_name in [
        "simready-foundation-conform-fet-000-core",
        "simready-foundation-conform-fet-003-rigid-body-physics",
        "simready-foundation-conform-fet-005-simulate-grasp-physics",
        "simready-foundation-conform-fet-006-materials",
        "simready-foundation-conform-fet-007-nonvisual-materials",
    ]:
        if skill_layout == "agents":
            skill = (
                root
                / ".agents"
                / "skills"
                / conform_profile_module._foundation_agent_skill_name(skill_name)
            )
        else:
            skill = root / "skills" / skill_name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"# {skill_name}\n", encoding="utf-8")
    return root


def _commit_fake_foundation(root: Path) -> str:
    git = ["git", "-c", "core.autocrlf=false"]
    subprocess.run([*git, "init", "-q", str(root)], check=True)
    subprocess.run([*git, "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            *git,
            "-C",
            str(root),
            "-c",
            "user.name=SimReady Test",
            "-c",
            "user.email=simready@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return subprocess.run(
        [*git, "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _venv_marker_identity() -> dict[str, object]:
    return foundation_runtime_module._expected_venv_marker_identity(
        foundation_commit="a" * 40,
        foundation_requirements_sha256="b" * 64,
    )


def test_simready_runtime_contract_includes_minimal_pydantic_closure() -> None:
    expected = {
        "annotated-types": "0.7.0",
        "pydantic": "2.12.5",
        "pydantic-core": "2.41.5",
        "typing-extensions": "4.15.0",
        "typing-inspection": "0.4.2",
    }

    assert expected.items() <= (
        foundation_runtime_module.SIMREADY_VALIDATOR_DISTRIBUTIONS.items()
    )
    assert {f"{name}=={version}" for name, version in expected.items()} <= set(
        foundation_runtime_module.SIMREADY_VALIDATOR_CONSTRAINTS
    )
    assert "pillow" not in foundation_runtime_module.SIMREADY_VALIDATOR_DISTRIBUTIONS
    assert "filelock" not in foundation_runtime_module.SIMREADY_VALIDATOR_DISTRIBUTIONS


def _write_fake_runtime_inventory(
    venv: Path,
    *,
    marker_identity: dict[str, object],
) -> None:
    site_packages = venv / "Lib" / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)
    provider = marker_identity["usd_provider"]
    assert isinstance(provider, str)
    inventory = foundation_runtime_module._expected_distribution_inventory(
        usd_provider=provider
    )
    for name, version in inventory.items():
        dist_info = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            encoding="utf-8",
        )


def _trusted_conformance_runtime(
    foundation_root: Path,
) -> foundation_runtime_module.SimReadyRuntimeInfo:
    validator = foundation_root / ".test-runtime" / "bin" / "simready-validate"
    validator.parent.mkdir(parents=True, exist_ok=True)
    validator.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    return foundation_runtime_module.SimReadyRuntimeInfo(
        foundation_root=str(foundation_root.resolve()),
        foundation_commit="a" * 40,
        foundation_checkout_verified=True,
        foundation_requirements_sha256="b" * 64,
        foundation_spec_tree_sha256="c" * 64,
        foundation_spec_root=str(
            (foundation_root / "nv_core" / "sr_specs" / "docs").resolve()
        ),
        validator_executable=str(validator.resolve()),
        validator_executable_sha256="d" * 64,
        validator_distributions_sha256="e" * 64,
        validator_runtime_verified=True,
        runtime_contract_sha256="f" * 64,
        specs_ready=True,
        runtime_ready=True,
    )


def _write_trusted_validation_report(
    path: Path,
    *,
    asset_path: Path,
    runtime: foundation_runtime_module.SimReadyRuntimeInfo,
    rerun_reasons: list[str],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": conform_profile_module.SIMREADY_VALIDATION_SCHEMA_VERSION,
        "asset_path": str(asset_path.resolve()),
        "asset_sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
        "asset_dependency_manifest": (
            validate_profile_module.build_asset_dependency_manifest(asset_path)
        ),
        "profile_name": "Prop-Robotics-Neutral",
        "profile_version": "1.0.0",
        "profile_target": "Prop-Robotics-Neutral@1.0.0",
        "foundation_root": runtime.foundation_root,
        "foundation_commit": runtime.foundation_commit,
        "foundation_checkout_verified": True,
        "foundation_requirements_sha256": runtime.foundation_requirements_sha256,
        "foundation_spec_tree_sha256": runtime.foundation_spec_tree_sha256,
        "foundation_spec_root": runtime.foundation_spec_root,
        "validator_executable": runtime.validator_executable,
        "validator_executable_sha256": runtime.validator_executable_sha256,
        "validator_distributions_sha256": (runtime.validator_distributions_sha256),
        "validator_runtime_verified": True,
        "runtime_contract_sha256": runtime.runtime_contract_sha256,
        "validator_tool": "simready-validate",
        "passed": False,
        "status": "FAIL",
        "needs_rerun": True,
        "rerun_reasons": rerun_reasons,
        "issues": [],
        "ignored_issues": [],
        "feature_results": None,
        "warnings": [],
        "errors": [],
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_fake_venv(tmp_path: Path, *, exit_code: int = 1) -> Path:
    venv = tmp_path / "simready-venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    executable = bin_dir / (
        "simready-validate.exe" if os.name == "nt" else "simready-validate"
    )
    executable.write_text(
        f"""#!/usr/bin/env python3
import json
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({{
    "passed": False,
    "status": "FAIL",
    "issues": [
        {{
            "requirement_id": "NP.006",
            "severity": "ERROR",
            "message": "Missing SimReady metadata."
        }}
    ],
    "feature_results": [
        {{
            "id": "FET_000_CORE",
            "passed": False,
            "failing_requirements": ["NP.006"]
        }}
    ],
    "warnings": []
}}), encoding="utf-8")
sys.exit({exit_code})
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return venv


def _write_flooding_pass_venv(tmp_path: Path, *, descriptor: int) -> Path:
    venv = tmp_path / f"simready-flood-venv-{descriptor}"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    executable = bin_dir / (
        "simready-validate.exe" if os.name == "nt" else "simready-validate"
    )
    executable.write_text(
        f"""#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({{
    "passed": True,
    "status": "PASS",
    "issues": [],
    "feature_results": [],
    "warnings": []
}}), encoding="utf-8")
os.write({descriptor}, b"x" * 65536)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return venv


def _write_fake_features_summary_venv(tmp_path: Path, *, exit_code: int = 0) -> Path:
    venv = tmp_path / "simready-summary-venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    executable = bin_dir / (
        "simready-validate.exe" if os.name == "nt" else "simready-validate"
    )
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

asset = Path(sys.argv[-1]).resolve()
output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    str(asset): {
        "profile_id": "Prop-Robotics-Physx",
        "profile_version": "1.0.0",
        "features_summary": {
            "FET003_BASE_PHYSX": {
                "dependencies": "[]",
                "passed": True,
                "version": "0.1.0"
            },
            "FET004_BASE_PHYSX": {
                "dependencies": "[]",
                "failing requirements": "['RB.MB.001']",
                "passed": False,
                "version": "0.1.0"
            }
        }
    }
}), encoding="utf-8")
sys.exit(0)
""".replace("sys.exit(0)", f"sys.exit({exit_code})"),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return venv


def _write_asset_capture_venv(tmp_path: Path) -> Path:
    venv = tmp_path / "simready-asset-capture-venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    executable = bin_dir / (
        "simready-validate.exe" if os.name == "nt" else "simready-validate"
    )
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

asset = Path(sys.argv[-1]).resolve()
accepted = asset.suffix.lower() in {".usd", ".usda"}
output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    "passed": accepted,
    "status": "PASS" if accepted else "FAIL",
    "warnings": [],
    str(asset): {
        "features_summary": {},
        "validator_target": str(asset),
        "relative_dependency_available": (
            asset.parent / "layers" / "sub.usda"
        ).is_file(),
    },
}), encoding="utf-8")
sys.exit(0 if accepted else 1)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return venv


def _write_dependency_mutating_venv(tmp_path: Path) -> Path:
    venv = tmp_path / "simready-dependency-mutating-venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    executable = bin_dir / (
        "simready-validate.exe" if os.name == "nt" else "simready-validate"
    )
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

asset = Path(sys.argv[-1]).resolve()
(asset.parent / "layers" / "sub.usda").write_text(
    '#usda 1.0\\n\\ndef Xform "ChangedDuringValidation" {}\\n',
    encoding="utf-8",
)
output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    "passed": True,
    "status": "PASS",
    "issues": [],
    "warnings": [],
}), encoding="utf-8")
sys.exit(0)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return venv


def _write_malformed_json_venv(tmp_path: Path) -> Path:
    venv = tmp_path / "simready-malformed-venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    executable = bin_dir / (
        "simready-validate.exe" if os.name == "nt" else "simready-validate"
    )
    executable.write_text(
        """#!/usr/bin/env python3
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("{not valid json", encoding="utf-8")
sys.exit(1)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return venv


def _write_minimal_wheel(
    wheelhouse: Path,
    *,
    name: str,
    version: str,
    requires: list[str] | None = None,
) -> Path:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    normalized = name.replace("-", "_")
    dist_info = f"{normalized}-{version}.dist-info"
    wheel_path = wheelhouse / f"{normalized}-{version}-py3-none-any.whl"
    metadata = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
        *(f"Requires-Dist: {requirement}" for requirement in requires or []),
    ]
    with ZipFile(wheel_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr(f"{normalized}/__init__.py", "")
        archive.writestr(f"{dist_info}/METADATA", "\n".join(metadata) + "\n")
        archive.writestr(
            f"{dist_info}/WHEEL",
            "\n".join(
                [
                    "Wheel-Version: 1.0",
                    "Generator: content-agent-workflows-test",
                    "Root-Is-Purelib: true",
                    "Tag: py3-none-any",
                ]
            )
            + "\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    return wheel_path


def _write_error_status_venv(tmp_path: Path) -> Path:
    venv = tmp_path / "simready-error-status-venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    executable = bin_dir / (
        "simready-validate.exe" if os.name == "nt" else "simready-validate"
    )
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    "passed": False,
    "status": "ERROR",
    "issues": [{"requirement_id": "NP.006", "severity": "ERROR"}],
}), encoding="utf-8")
sys.exit(0)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return venv


def _write_invalid_utf8_venv(tmp_path: Path) -> Path:
    venv = tmp_path / "simready-invalid-utf8-venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    executable = bin_dir / (
        "simready-validate.exe" if os.name == "nt" else "simready-validate"
    )
    executable.write_text(
        """#!/usr/bin/env python3
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(b"\\xff\\xfe")
sys.exit(1)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return venv


def _write_no_report_venv(tmp_path: Path, *, exit_code: int = 0) -> Path:
    venv = tmp_path / "simready-no-report-venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    executable = bin_dir / (
        "simready-validate.exe" if os.name == "nt" else "simready-validate"
    )
    executable.write_text(
        f"""#!/usr/bin/env python3
import sys

sys.exit({exit_code})
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return venv


def _write_single_mesh_asset(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "coffee_mug"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "coffee_mug"
{
    def Mesh "Mesh"
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
    }
}
""",
        encoding="utf-8",
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_isa001_compliant_asset(path: Path) -> None:
    from pxr import Kind, Sdf

    payload_dir = path.parent / "payloads"
    payload_dir.mkdir(parents=True)
    root_path = Sdf.Path("/Asset")

    meshes = Sdf.Layer.CreateNew(str(payload_dir / "asset_meshes.usd"))
    meshes.defaultPrim = "Asset"
    meshes_root = Sdf.CreatePrimInLayer(meshes, root_path)
    meshes_root.specifier = Sdf.SpecifierDef
    meshes_root.typeName = "Xform"
    child = Sdf.CreatePrimInLayer(meshes, root_path.AppendChild("Mesh"))
    child.specifier = Sdf.SpecifierDef
    child.typeName = "Mesh"
    assert meshes.Save()

    base = Sdf.Layer.CreateNew(str(payload_dir / "asset_base.usd"))
    base.defaultPrim = "Asset"
    base_root = Sdf.CreatePrimInLayer(base, root_path)
    base_root.specifier = Sdf.SpecifierDef
    base_root.referenceList.prependedItems = [
        Sdf.Reference("./asset_meshes.usd", root_path)
    ]
    assert base.Save()

    physics = Sdf.Layer.CreateNew(str(payload_dir / "asset_physics.usd"))
    physics.defaultPrim = "Asset"
    physics_root = Sdf.CreatePrimInLayer(physics, root_path)
    physics_root.specifier = Sdf.SpecifierDef
    assert physics.Save()

    main = Sdf.Layer.CreateNew(str(path))
    main.defaultPrim = "Asset"
    main_root = Sdf.CreatePrimInLayer(main, root_path)
    main_root.specifier = Sdf.SpecifierOver
    main_root.SetInfo("kind", Kind.Tokens.component)
    main_root.referenceList.prependedItems = [
        Sdf.Reference("./payloads/asset_base.usd", root_path)
    ]
    main_root.payloadList.prependedItems = [
        Sdf.Payload("./payloads/asset_physics.usd", root_path)
    ]
    assert main.Save()


def _write_mixed_subset_asset(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "asset"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "asset"
{
    def Mesh "MeshWithSubset"
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]

        def GeomSubset "partA"
        {
            uniform token elementType = "face"
            int[] indices = [0]
        }
    }

    def Mesh "PlainMesh"
    {
        point3f[] points = [(0, 0, 1), (1, 0, 1), (0, 1, 1)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
    }
}
""",
        encoding="utf-8",
    )


def _write_validation_usdz(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "root.usda",
            """#usda 1.0
(
    subLayers = [@layers/sub.usda@]
)

def Xform "World"
{
}
""",
        )
        archive.writestr(
            "layers/sub.usda",
            '#usda 1.0\n\ndef Xform "Dependency"\n{\n}\n',
        )


def test_simready_preflight_accepts_foundation_root_and_venv(tmp_path: Path) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_fake_venv(tmp_path)

    report = preflight_simready_foundation(
        foundation_root=foundation_root,
        venv_path=venv,
        install_missing=False,
    )

    assert report.passed
    assert report.status == "PASS"
    assert report.specs_ready
    assert report.runtime_ready
    assert report.available_profiles == ["Prop-Robotics-Neutral"]
    assert report.foundation_root == str(foundation_root.resolve())


def test_default_simready_foundation_is_release_pinned() -> None:
    assert DEFAULT_SIMREADY_FOUNDATION_REF == "v2026.04.1"
    assert DEFAULT_SIMREADY_FOUNDATION_COMMIT == (
        "a1e9dd68ee2d107f74dc6cd6da875b54ad3f8fd3"
    )


def test_simready_managed_commit_mismatch_never_installs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache = tmp_path / "cache"
    root = cache / "checkouts" / "simready-foundation-v2026.04.1"
    spec_root = root / "nv_core" / "sr_specs" / "docs"
    (spec_root / "capabilities").mkdir(parents=True)
    (spec_root / "features").mkdir()
    (spec_root / "profiles").mkdir()
    (spec_root / "profiles" / "profiles.toml").write_text(
        '[Prop-Robotics-Neutral]\n"1.0.0" = { features = [] }\n',
        encoding="utf-8",
    )
    monkeypatch.setenv(SIMREADY_CACHE_DIR_ENV, str(cache))
    monkeypatch.setattr(
        foundation_runtime_module,
        "_foundation_commit",
        lambda _root: "b" * 40,
    )

    def unexpected_install(_command, **_kwargs):
        raise AssertionError("identity mismatch must short-circuit installation")

    monkeypatch.setattr(
        foundation_runtime_module,
        "_prepare_validation_venv",
        unexpected_install,
    )

    runtime = foundation_runtime_module.resolve_simready_runtime(install_missing=True)

    assert not runtime.passed
    assert runtime.install_command == []
    assert any("identity differs" in item for item in runtime.errors)


def test_simready_managed_refs_must_be_immutable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(SIMREADY_CACHE_DIR_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(SIMREADY_FOUNDATION_REF_ENV, "main")

    report = preflight_simready_foundation(install_missing=True)

    assert not report.passed
    assert any("must be immutable" in item for item in report.errors)
    assert report.install_command == []


def test_simready_managed_checkout_refuses_in_place_update(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache = tmp_path / "cache"
    root = cache / "checkouts" / "simready-foundation-v2026.04.1"
    root.mkdir(parents=True)
    monkeypatch.setenv(SIMREADY_CACHE_DIR_ENV, str(cache))

    runtime = foundation_runtime_module.resolve_simready_runtime(
        install_missing=False,
        update_foundation=True,
    )

    assert not runtime.passed
    assert any("immutable" in item for item in runtime.errors)
    assert runtime.install_command == []


def test_simready_managed_runtime_blocks_alternate_spec_and_unmarked_venv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _write_fake_foundation(tmp_path)
    alternate_spec = tmp_path / "alternate-spec"
    (alternate_spec / "capabilities").mkdir(parents=True)
    (alternate_spec / "features").mkdir()
    (alternate_spec / "profiles").mkdir()
    (alternate_spec / "profiles" / "profiles.toml").write_text(
        '[Prop-Robotics-Neutral]\n"1.0.0" = { features = [] }\n',
        encoding="utf-8",
    )
    alternate_venv = _write_fake_venv(tmp_path)
    monkeypatch.setattr(
        foundation_runtime_module,
        "_resolve_foundation_root",
        lambda *_args, **_kwargs: (root.resolve(), True),
    )
    monkeypatch.setattr(
        foundation_runtime_module,
        "_foundation_commit",
        lambda _root: DEFAULT_SIMREADY_FOUNDATION_COMMIT,
    )
    monkeypatch.setattr(
        foundation_runtime_module,
        "_managed_checkout_verification",
        lambda *_args, **_kwargs: ([], "c" * 64),
    )

    runtime = foundation_runtime_module.resolve_simready_runtime(
        foundation_spec_root=alternate_spec,
        venv_path=alternate_venv,
        install_missing=False,
    )

    assert not runtime.passed
    assert runtime.validator_executable is None
    assert any("spec root must use" in item for item in runtime.errors)
    assert any("executable is unavailable" in item for item in runtime.errors)


def test_simready_managed_checkout_rejects_local_bytes_and_ignored_spec_files(
    tmp_path: Path,
) -> None:
    root = _write_fake_foundation(tmp_path)
    _commit_fake_foundation(root)
    profiles = root / "nv_core" / "sr_specs" / "docs" / "profiles"
    (profiles / "profiles.toml").write_text("modified\n", encoding="utf-8")
    (root / ".gitignore").write_text("*.rogue\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", ".gitignore"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=SimReady Test",
            "-c",
            "user.email=simready@example.test",
            "commit",
            "-qm",
            "ignore fixture",
        ],
        check=True,
    )
    (profiles / "injected.rogue").write_text("injected\n", encoding="utf-8")

    errors = foundation_runtime_module._managed_checkout_errors(root)

    assert any("local changes" in item for item in errors)
    assert any("source file set differs" in item for item in errors)
    assert any("source bytes differ" in item for item in errors)


def test_simready_managed_checkout_checks_full_worktree_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_fake_foundation(tmp_path)
    commit = _commit_fake_foundation(root)
    commands: list[list[str]] = []
    real_run = subprocess.run

    def recording_run(*args, **kwargs):
        command = args[0]
        if isinstance(command, list):
            commands.append(command)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(foundation_runtime_module.subprocess, "run", recording_run)

    errors = foundation_runtime_module._managed_checkout_errors(
        root,
        expected_commit=commit,
    )

    assert errors == []
    status_command = next(command for command in commands if "status" in command)
    assert status_command[status_command.index("status") + 1 :] == [
        "--porcelain",
        "--untracked-files=all",
    ]


def test_simready_managed_checkout_attests_validator_source_bytes(
    tmp_path: Path,
) -> None:
    root = _write_fake_foundation(tmp_path)
    relative = "nv_core/validator_sample/validate_asset.py"
    validator = root / relative
    validator.parent.mkdir(parents=True)
    validator.write_text("print('validator')\n", encoding="utf-8")
    commit = _commit_fake_foundation(root)
    subprocess.run(
        ["git", "-C", str(root), "update-index", "--assume-unchanged", relative],
        check=True,
    )
    validator.write_text("print('tampered')\n", encoding="utf-8")

    errors = foundation_runtime_module._managed_checkout_errors(
        root,
        expected_commit=commit,
    )

    assert not any("local changes" in item for item in errors)
    assert any("special index flags" in item for item in errors)
    assert any("source bytes differ" in item for item in errors)


def test_simready_managed_checkout_accepts_exact_git_lfs_payload() -> None:
    payload = b"pinned foundation image bytes\n"
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        + f"oid sha256:{hashlib.sha256(payload).hexdigest()}\n".encode()
        + f"size {len(payload)}\n".encode()
    )

    assert foundation_runtime_module._matches_git_lfs_pointer(pointer, payload)


def test_simready_managed_clone_materializes_lfs_with_trusted_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_executable = foundation_runtime_module._managed_git_executable()
    git_lfs_executable = foundation_runtime_module._managed_git_lfs_executable()
    if git_executable is None or git_lfs_executable is None:
        pytest.skip("trusted git and git-lfs executables are required")

    environment = foundation_runtime_module._managed_git_environment(
        git_executable,
        allow_network=True,
    )
    source = tmp_path / "source"
    origin = tmp_path / "origin.git"
    checkout = tmp_path / "checkout"
    subprocess.run(
        foundation_runtime_module._managed_git_command(
            git_executable,
            "init",
            "-q",
            str(source),
        ),
        check=True,
        env=environment,
    )
    spec_root = source / "nv_core" / "sr_specs" / "docs"
    spec_root.mkdir(parents=True)
    (source / ".gitattributes").write_text(
        "*.bin filter=lfs diff=lfs merge=lfs -text\n",
        encoding="ascii",
    )
    payload = b"materialized foundation payload\n" * 16
    (spec_root / "fixture.bin").write_bytes(payload)
    subprocess.run(
        foundation_runtime_module._managed_git_command(
            git_executable,
            "-C",
            str(source),
            "add",
            ".",
        ),
        check=True,
        env=environment,
    )
    subprocess.run(
        foundation_runtime_module._managed_git_command(
            git_executable,
            "-C",
            str(source),
            "-c",
            "user.name=SimReady Test",
            "-c",
            "user.email=simready@example.test",
            "commit",
            "-qm",
            "LFS fixture",
        ),
        check=True,
        env=environment,
    )
    commit = subprocess.run(
        foundation_runtime_module._managed_git_command(
            git_executable,
            "-C",
            str(source),
            "rev-parse",
            "HEAD",
        ),
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    ).stdout.strip()
    subprocess.run(
        foundation_runtime_module._managed_git_command(
            git_executable,
            "clone",
            "--bare",
            str(source),
            str(origin),
        ),
        check=True,
        env=environment,
    )
    subprocess.run(
        foundation_runtime_module._managed_git_command(
            git_executable,
            "-C",
            str(source),
            "remote",
            "add",
            "origin",
            str(origin),
        ),
        check=True,
        env=environment,
    )
    subprocess.run(
        foundation_runtime_module._managed_git_command(
            git_executable,
            "-C",
            str(source),
            "lfs",
            "push",
            "--all",
            "origin",
        ),
        check=True,
        env=environment,
    )
    monkeypatch.setattr(
        foundation_runtime_module,
        "DEFAULT_SIMREADY_FOUNDATION_REPO_URL",
        str(origin),
    )

    error = foundation_runtime_module._clone_foundation(checkout, ref=commit)

    assert error is None
    assert (checkout / "nv_core/sr_specs/docs/fixture.bin").read_bytes() == payload
    assert (
        foundation_runtime_module._managed_checkout_errors(
            checkout,
            expected_commit=commit,
        )
        == []
    )


def test_simready_managed_checkout_rejects_unsmudged_git_lfs_pointer(
    tmp_path: Path,
) -> None:
    root = _write_fake_foundation(tmp_path)
    relative = "nv_core/sr_specs/docs/profiles/profiles.toml"
    payload = b"pinned foundation profile bytes\n"
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        + f"oid sha256:{hashlib.sha256(payload).hexdigest()}\n".encode()
        + f"size {len(payload)}\n".encode()
    )
    (root / relative).write_bytes(pointer)
    commit = _commit_fake_foundation(root)

    errors = foundation_runtime_module._managed_checkout_errors(
        root,
        expected_commit=commit,
    )

    assert any("unresolved Git LFS pointer" in item for item in errors)


@pytest.mark.parametrize(
    ("pointer_transform", "payload_transform"),
    [
        (lambda value: value.replace(b"sha256:", b"sha512:"), lambda value: value),
        (lambda value: value.replace(b"size 30", b"size 29"), lambda value: value),
        (lambda value: value, lambda value: value + b"tampered"),
    ],
)
def test_simready_managed_checkout_rejects_invalid_git_lfs_payload(
    pointer_transform,
    payload_transform,
) -> None:
    payload = b"pinned foundation image bytes\n"
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        + f"oid sha256:{hashlib.sha256(payload).hexdigest()}\n".encode()
        + f"size {len(payload)}\n".encode()
    )

    assert not foundation_runtime_module._matches_git_lfs_pointer(
        pointer_transform(pointer),
        payload_transform(payload),
    )


def test_simready_managed_checkout_rejects_assume_unchanged_spec_bytes(
    tmp_path: Path,
) -> None:
    root = _write_fake_foundation(tmp_path)
    commit = _commit_fake_foundation(root)
    relative = "nv_core/sr_specs/docs/profiles/profiles.toml"
    subprocess.run(
        ["git", "-C", str(root), "update-index", "--assume-unchanged", relative],
        check=True,
    )
    (root / relative).write_text("tampered\n", encoding="utf-8")

    errors = foundation_runtime_module._managed_checkout_errors(
        root,
        expected_commit=commit,
    )

    assert not any("local changes" in item for item in errors)
    assert any("special index flags" in item for item in errors)
    assert any("source bytes differ" in item for item in errors)


def test_simready_managed_checkout_attests_fallback_requirements(
    tmp_path: Path,
) -> None:
    root = _write_fake_foundation(tmp_path)
    relative = "nv_core/validator_sample/requirements.txt"
    fallback = root / relative
    fallback.parent.mkdir(parents=True)
    fallback.write_text("simready-validate==2026.4.8\n", encoding="utf-8")
    commit = _commit_fake_foundation(root)
    subprocess.run(
        ["git", "-C", str(root), "update-index", "--assume-unchanged", relative],
        check=True,
    )
    fallback.write_text("untrusted-package==1.0\n", encoding="utf-8")

    errors = foundation_runtime_module._managed_checkout_errors(
        root,
        expected_commit=commit,
    )

    assert not any("local changes" in item for item in errors)
    assert any("special index flags" in item for item in errors)
    assert any("source bytes differ" in item for item in errors)


def test_simready_managed_checkout_rejects_replacement_refs(tmp_path: Path) -> None:
    root = _write_fake_foundation(tmp_path)
    original_commit = _commit_fake_foundation(root)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=SimReady Test",
            "-c",
            "user.email=simready@example.test",
            "commit",
            "--allow-empty",
            "-qm",
            "replacement target",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(root), "replace", head, original_commit],
        check=True,
    )

    errors = foundation_runtime_module._managed_checkout_errors(
        root,
        expected_commit=head,
    )

    assert any("contains replacement refs" in item for item in errors)


def test_simready_preflight_rejects_option_like_foundation_ref(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(SIMREADY_CACHE_DIR_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(SIMREADY_FOUNDATION_REF_ENV, "--bad-ref")

    report = preflight_simready_foundation(install_missing=True)

    assert not report.passed
    assert any("Invalid SimReady Foundation ref" in item for item in report.errors)


def test_simready_prepare_validation_venv_reports_malformed_command() -> None:
    assert (
        _prepare_validation_venv(
            ["uv", "venv"],
            expected_marker_identity=_venv_marker_identity(),
        )
        == "Malformed SimReady validation venv install command."
    )


def test_simready_managed_tool_environment_is_allowlisted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/injected.so")
    monkeypatch.setenv("PIP_INDEX_URL", "https://pip.example.test/simple")
    monkeypatch.setenv("PYTHONPATH", "/tmp/injected-python")
    monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://extra.example.test/simple")
    monkeypatch.setenv("UV_INDEX_URL", "https://index.example.test/simple")
    monkeypatch.setenv("UV_NATIVE_TLS", "true")
    monkeypatch.setenv("LANG", "C.UTF-8")

    environment = foundation_runtime_module.build_simready_subprocess_environment(
        executable_dir=tmp_path / "bin",
        allow_network=True,
    )

    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "LD_PRELOAD" not in environment
    assert "PIP_INDEX_URL" not in environment
    assert "PYTHONPATH" not in environment
    assert "UV_EXTRA_INDEX_URL" not in environment
    assert "UV_INDEX_URL" not in environment
    assert "UV_NATIVE_TLS" not in environment
    assert environment["LANG"] == "C.UTF-8"
    assert environment["PATH"].split(os.pathsep)[0] == str(tmp_path / "bin")
    assert environment["UV_NO_CONFIG"] == "1"


def test_simready_managed_git_environment_disables_external_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "/tmp/injected-monitor")
    monkeypatch.setenv("GIT_SSH_COMMAND", "/tmp/injected-ssh")

    environment = foundation_runtime_module._managed_git_environment(
        tmp_path / "bin" / "git",
        allow_network=True,
    )

    assert "GIT_CONFIG_COUNT" not in environment
    assert "GIT_CONFIG_KEY_0" not in environment
    assert "GIT_CONFIG_VALUE_0" not in environment
    assert "GIT_SSH_COMMAND" not in environment
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_SYSTEM"] == os.devnull
    assert environment["GIT_LFS_SKIP_SMUDGE"] == "0"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"


def test_simready_managed_git_ignores_caller_path_and_wires_trusted_lfs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = tmp_path / "git"
    hostile.write_text("#!/bin/sh\nexit 99\n", encoding="ascii")
    hostile.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    git_executable = foundation_runtime_module._managed_git_executable()
    if git_executable is None:
        pytest.skip("trusted git executable is unavailable")
    assert git_executable != hostile.resolve()

    git_lfs_executable = tmp_path / "trusted git-lfs"
    monkeypatch.setattr(
        foundation_runtime_module,
        "_managed_git_lfs_executable",
        lambda: git_lfs_executable,
    )
    command = foundation_runtime_module._managed_git_command(
        git_executable,
        "status",
    )
    config_values = [
        command[index + 1] for index, value in enumerate(command[:-1]) if value == "-c"
    ]

    assert command[0] == str(git_executable)
    if os.name == "nt":
        assert "core.longpaths=true" in config_values
    assert any(
        value.startswith("filter.lfs.clean=") and str(git_lfs_executable) in value
        for value in config_values
    )
    assert any(
        value.startswith("filter.lfs.smudge=") and str(git_lfs_executable) in value
        for value in config_values
    )
    assert any(
        value.startswith("filter.lfs.process=") and str(git_lfs_executable) in value
        for value in config_values
    )


def test_simready_install_command_uses_usd_exchange_provider_on_linux_arm64(
    tmp_path: Path,
    monkeypatch,
) -> None:
    foundation_root = tmp_path / "foundation"
    requirements = foundation_root / "requirements.txt"
    requirements.parent.mkdir()
    requirements.write_text(
        "\n".join(
            [
                "simready-validate>=2026.4.8",
                "usd-core==25.5",
                "# usd-core in comments is preserved",
                "usd-core-extra==1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    venv = tmp_path / "simready-venv"
    monkeypatch.setattr(foundation_runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(foundation_runtime_module.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(
        foundation_runtime_module.platform, "machine", lambda: "aarch64"
    )

    command = _install_command(foundation_root, venv)

    filtered = venv.with_name(f"{venv.name}-usd-exchange-requirements.txt")
    overrides = venv.with_name(f"{venv.name}-usd-exchange-overrides.txt")
    assert command[command.index("--overrides") + 1] == str(overrides)
    assert command[command.index("-r") + 1] == str(filtered)
    assert SIMREADY_USD_EXCHANGE_REQUIREMENT in command
    filtered_text = filtered.read_text(encoding="utf-8")
    assert "# Replaced by usd-exchange==2.3.0: usd-core==25.5" in filtered_text
    assert "# usd-core in comments is preserved" in filtered_text
    assert "usd-core-extra==1.0" in filtered_text
    assert overrides.read_text(encoding="utf-8").splitlines()[-1] == (
        SIMREADY_USD_CORE_OVERRIDE
    )


def test_simready_install_command_uses_supported_uv_pip_flags(
    tmp_path: Path,
    monkeypatch,
) -> None:
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to guard SimReady install command flags"

    foundation_root = tmp_path / "foundation"
    requirements = foundation_root / "requirements.txt"
    requirements.parent.mkdir()
    requirements.write_text("simready-validate>=2026.4.8\n", encoding="utf-8")
    monkeypatch.setenv(SIMREADY_USD_PROVIDER_ENV, "usd-exchange")

    command = _install_command(foundation_root, tmp_path / "simready-venv")
    completed = subprocess.run(
        [uv, "pip", "install", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    pip_install = command[command.index("install") + 1 :]
    generated_long_flags = [arg for arg in pip_install if arg.startswith("--")]

    for flag in generated_long_flags:
        assert flag in completed.stdout

    wheelhouse = tmp_path / "wheelhouse"
    _write_minimal_wheel(
        wheelhouse,
        name="usd-exchange",
        version="2.3.0",
    )
    _write_minimal_wheel(
        wheelhouse,
        name="simready-validate",
        version="2026.4.9",
        requires=["usd-core ==25.5"],
    )
    dry_run = subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--dry-run",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "--target",
            str(tmp_path / "target"),
            SIMREADY_USD_EXCHANGE_REQUIREMENT,
            "--overrides",
            command[command.index("--overrides") + 1],
            "-r",
            command[command.index("-r") + 1],
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
        text=True,
    )
    resolver_output = dry_run.stdout + dry_run.stderr
    assert "simready-validate==2026.4.9" in resolver_output
    assert "usd-exchange==2.3.0" in resolver_output
    assert "usd-core" not in resolver_output


def test_simready_managed_validation_venv_cache_includes_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    foundation_root = tmp_path / "foundation"
    monkeypatch.setenv(SIMREADY_CACHE_DIR_ENV, str(tmp_path / "cache"))

    monkeypatch.setenv(SIMREADY_USD_PROVIDER_ENV, "usd-core")
    usd_core_venv, usd_core_managed = foundation_runtime_module._resolve_venv_path(
        None,
        foundation_root,
        foundation_commit="a" * 40,
    )

    monkeypatch.setenv(SIMREADY_USD_PROVIDER_ENV, "usd-exchange")
    usd_exchange_venv, usd_exchange_managed = (
        foundation_runtime_module._resolve_venv_path(
            None,
            foundation_root,
            foundation_commit="a" * 40,
        )
    )

    assert usd_core_managed
    assert usd_exchange_managed
    assert usd_core_venv != usd_exchange_venv


def test_simready_managed_validation_venv_cache_includes_foundation_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    foundation_root = tmp_path / "foundation"
    monkeypatch.setenv(SIMREADY_CACHE_DIR_ENV, str(tmp_path / "cache"))

    first, first_managed = foundation_runtime_module._resolve_venv_path(
        None,
        foundation_root,
        foundation_commit="a" * 40,
    )
    second, second_managed = foundation_runtime_module._resolve_venv_path(
        None,
        foundation_root,
        foundation_commit="b" * 40,
    )

    assert first_managed
    assert second_managed
    assert first != second


def test_simready_managed_validation_venv_cache_is_checkout_path_independent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_root = tmp_path / "first" / "foundation"
    second_root = tmp_path / "relocated" / "foundation"
    for root in (first_root, second_root):
        root.mkdir(parents=True)
        (root / "requirements.txt").write_text(
            "simready-validate==2026.4.9\n",
            encoding="utf-8",
        )
    monkeypatch.setenv(SIMREADY_CACHE_DIR_ENV, str(tmp_path / "cache"))

    first, first_managed = foundation_runtime_module._resolve_venv_path(
        None,
        first_root,
        foundation_commit="a" * 40,
    )
    second, second_managed = foundation_runtime_module._resolve_venv_path(
        None,
        second_root,
        foundation_commit="a" * 40,
    )

    assert first_managed
    assert second_managed
    assert first == second


def test_simready_managed_validation_venv_cache_includes_platform_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    foundation_root = tmp_path / "foundation"
    requirements = foundation_root / "requirements.txt"
    requirements.parent.mkdir()
    requirements.write_text("simready-validate>=2026.4.8\n", encoding="utf-8")
    monkeypatch.setenv(SIMREADY_CACHE_DIR_ENV, str(tmp_path / "cache"))

    monkeypatch.setattr(
        foundation_runtime_module, "_runtime_platform_identity", lambda: "first"
    )
    first, _ = foundation_runtime_module._resolve_venv_path(
        None,
        foundation_root,
        foundation_commit="a" * 40,
    )
    monkeypatch.setattr(
        foundation_runtime_module, "_runtime_platform_identity", lambda: "second"
    )
    second, _ = foundation_runtime_module._resolve_venv_path(
        None,
        foundation_root,
        foundation_commit="a" * 40,
    )

    assert first != second


def test_simready_managed_validation_venv_cache_includes_requirements(
    tmp_path: Path,
    monkeypatch,
) -> None:
    foundation_root = tmp_path / "foundation"
    requirements = foundation_root / "requirements.txt"
    requirements.parent.mkdir()
    monkeypatch.setenv(SIMREADY_CACHE_DIR_ENV, str(tmp_path / "cache"))
    requirements.write_text("simready-validate==2026.4.8\n", encoding="utf-8")
    first, _ = foundation_runtime_module._resolve_venv_path(
        None,
        foundation_root,
        foundation_commit="a" * 40,
    )
    requirements.write_text("simready-validate==2026.4.9\n", encoding="utf-8")
    second, _ = foundation_runtime_module._resolve_venv_path(
        None,
        foundation_root,
        foundation_commit="a" * 40,
    )

    assert first != second


def test_simready_install_command_can_force_usd_core_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    foundation_root = tmp_path / "foundation"
    requirements = foundation_root / "requirements.txt"
    requirements.parent.mkdir()
    requirements.write_text("simready-validate>=2026.4.8\n", encoding="utf-8")
    monkeypatch.setenv(SIMREADY_USD_PROVIDER_ENV, "usd-core")
    monkeypatch.setattr(foundation_runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(foundation_runtime_module.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(
        foundation_runtime_module.platform, "machine", lambda: "aarch64"
    )

    command = _install_command(foundation_root, tmp_path / "simready-venv")

    assert "--overrides" not in command
    assert SIMREADY_USD_EXCHANGE_REQUIREMENT not in command


def test_simready_prepare_validation_venv_removes_partial_on_install_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    venv = tmp_path / "simready-venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    validator = bin_dir / (
        "simready-validate.exe" if os.name == "nt" else "simready-validate"
    )
    python_executable = bin_dir / ("python.exe" if os.name == "nt" else "python")
    monkeypatch.setattr(foundation_runtime_module.shutil, "which", lambda name: name)

    def fake_run(command, **_kwargs):
        if Path(command[0]).name == "uv" and command[1] == "venv":
            bin_dir.mkdir(parents=True)
            python_executable.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            validator.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stderr="")
        return SimpleNamespace(returncode=1, stderr="install failed")

    monkeypatch.setattr(foundation_runtime_module.subprocess, "run", fake_run)

    error = _prepare_validation_venv(
        [
            "uv",
            "venv",
            "--python",
            "python",
            str(venv),
            "&&",
            "uv",
            "pip",
            "install",
            "--python",
            str(python_executable),
            "-r",
            str(tmp_path / "requirements.txt"),
        ],
        expected_marker_identity=_venv_marker_identity(),
    )

    assert error is not None
    assert "Failed to install SimReady validation dependencies" in error
    assert not venv.exists()


def test_simready_prepare_validation_venv_clears_broken_existing_before_install(
    tmp_path: Path,
    monkeypatch,
) -> None:
    venv = tmp_path / "simready-venv"
    stale = venv / "stale.txt"
    stale.parent.mkdir()
    stale.write_text("broken", encoding="utf-8")
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    validator = bin_dir / (
        "simready-validate.exe" if os.name == "nt" else "simready-validate"
    )
    python_executable = bin_dir / ("python.exe" if os.name == "nt" else "python")
    marker_identity = _venv_marker_identity()
    isolated_names = {
        "PIP_INDEX_URL",
        "PYTHONPATH",
        "UV_EXTRA_INDEX_URL",
        "UV_INDEX_URL",
        "VIRTUAL_ENV",
    }
    for name in isolated_names:
        monkeypatch.setenv(name, f"injected-{name.lower()}")
    monkeypatch.setattr(foundation_runtime_module.shutil, "which", lambda name: name)
    observed_commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        observed_commands.append(command)
        assert kwargs["cwd"] == venv.parent
        assert not isolated_names & kwargs["env"].keys()
        assert kwargs["env"]["UV_NO_CONFIG"] == "1"
        if Path(command[0]).name == "uv" and command[1] == "venv":
            assert not stale.exists()
            bin_dir.mkdir(parents=True)
            python_executable.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            validator.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            _write_fake_runtime_inventory(
                venv,
                marker_identity=marker_identity,
            )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(foundation_runtime_module.subprocess, "run", fake_run)

    error = _prepare_validation_venv(
        [
            "uv",
            "venv",
            "--python",
            "python",
            str(venv),
            "&&",
            "uv",
            "pip",
            "install",
            "--python",
            str(python_executable),
            "-r",
            str(tmp_path / "requirements.txt"),
        ],
        expected_marker_identity=marker_identity,
    )

    assert error is None
    assert len(observed_commands) == 2
    assert foundation_runtime_module._venv_ready_marker(venv).exists()


@pytest.mark.parametrize("tamper", ["marker", "entrypoint", "inventory"])
def test_simready_validator_from_managed_venv_requires_ready_marker(
    tmp_path: Path,
    tamper: str,
) -> None:
    venv = _write_fake_venv(tmp_path)
    validator = foundation_runtime_module._validator_from_venv(venv)
    marker_identity = _venv_marker_identity()
    _write_fake_runtime_inventory(venv, marker_identity=marker_identity)

    assert validator is not None
    assert (
        foundation_runtime_module._validator_from_venv(
            venv,
            require_ready_marker=True,
            expected_marker_identity=marker_identity,
        )
        is None
    )

    foundation_runtime_module._write_venv_ready_marker(
        venv,
        expected_marker_identity=marker_identity,
    )

    assert (
        foundation_runtime_module._validator_from_venv(
            venv,
            require_ready_marker=True,
            expected_marker_identity=marker_identity,
        )
        == validator
    )

    if tamper == "marker":
        foundation_runtime_module._venv_ready_marker(venv).write_text(
            "{malformed",
            encoding="utf-8",
        )
    elif tamper == "entrypoint":
        assert validator is not None
        validator.write_text("#!/usr/bin/env python3\n# tampered\n", encoding="utf-8")
    else:
        metadata = next((venv / "Lib" / "site-packages").rglob("METADATA"))
        metadata.write_text(
            metadata.read_text(encoding="utf-8").replace("Version: ", "Version: 999."),
            encoding="utf-8",
        )
    assert (
        foundation_runtime_module._validator_from_venv(
            venv,
            require_ready_marker=True,
            expected_marker_identity=marker_identity,
        )
        is None
    )


def test_simready_validation_venv_lock_excludes_concurrent_process_and_reacquires(
    tmp_path: Path,
) -> None:
    venv = tmp_path / "simready-venv"
    lock_path = venv.with_name(f"{venv.name}.lock")
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sys",
                    "from pathlib import Path",
                    "sys.path.insert(0, sys.argv[2])",
                    (
                        "from content_agent_workflows.simready.foundation_runtime "
                        "import _acquire_venv_lock, _release_venv_lock"
                    ),
                    "path, fd, error = _acquire_venv_lock(Path(sys.argv[1]))",
                    "assert path is not None and fd is not None and error is None",
                    "print('locked', flush=True)",
                    "sys.stdin.readline()",
                    "_release_venv_lock(path, fd)",
                ]
            ),
            str(venv),
            str(Path(__file__).resolve().parents[1]),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "locked"

    acquired_path, lock_fd, error = _acquire_venv_lock(venv, timeout_s=0.0)

    assert acquired_path is None
    assert lock_fd is None
    assert error is not None
    assert "Timed out waiting" in error
    stdout, stderr = child.communicate("\n", timeout=10)
    assert child.returncode == 0, stdout + stderr
    assert lock_path.exists()
    metadata = lock_path.stat()
    assert stat.S_ISREG(metadata.st_mode)
    assert metadata.st_nlink == 1
    if os.name != "nt":
        assert stat.S_IMODE(metadata.st_mode) == 0o600
    if hasattr(os, "geteuid"):
        assert metadata.st_uid == os.geteuid()

    acquired_path, lock_fd, error = _acquire_venv_lock(venv, timeout_s=1.0)

    assert acquired_path == lock_path
    assert lock_fd is not None
    assert error is None
    foundation_runtime_module._release_venv_lock(acquired_path, lock_fd)
    assert lock_path.exists()


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "O_NOFOLLOW"),
    reason="requires POSIX symlinks and O_NOFOLLOW",
)
def test_simready_validation_venv_lock_refuses_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    venv = tmp_path / "simready-venv"
    lock_path = venv.with_name(f"{venv.name}.lock")
    target = tmp_path / "lock-target.txt"
    original = b"do not modify\n"
    target.write_bytes(original)
    try:
        lock_path.symlink_to(target)
    except OSError as exc:  # pragma: no cover - platform permission dependent
        pytest.skip(f"symlink creation is unavailable: {exc}")

    acquired_path, lock_fd, error = _acquire_venv_lock(venv, timeout_s=0.0)

    assert acquired_path is None
    assert lock_fd is None
    assert error is not None
    assert "Failed to open" in error or "Refusing unsafe" in error
    assert lock_path.is_symlink()
    assert target.read_bytes() == original


def test_simready_resolve_runtime_locks_managed_foundation_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(SIMREADY_CACHE_DIR_ENV, str(cache_dir))
    monkeypatch.setattr(foundation_runtime_module.shutil, "which", lambda _name: None)
    observed_lock_exists = False

    def fake_clone(root: Path, *, ref: str) -> None:
        nonlocal observed_lock_exists
        observed_lock_exists = root.with_name(f"{root.name}.lock").exists()
        spec_root = root / "nv_core" / "sr_specs" / "docs"
        (spec_root / "capabilities").mkdir(parents=True)
        (spec_root / "features").mkdir(parents=True)
        (spec_root / "profiles").mkdir(parents=True)
        (spec_root / "profiles" / "profiles.toml").write_text(
            '[Prop-Robotics-Neutral]\n"1.0.0" = { features = [] }\n',
            encoding="utf-8",
        )
        return None

    monkeypatch.setattr(foundation_runtime_module, "_clone_foundation", fake_clone)

    runtime = foundation_runtime_module.resolve_simready_runtime(
        install_missing=True,
    )

    assert observed_lock_exists
    assert runtime.foundation_root is not None
    lock_path = Path(runtime.foundation_root).with_name(
        f"{Path(runtime.foundation_root).name}.lock"
    )
    assert lock_path.exists()
    acquired_path, lock_fd, error = foundation_runtime_module._acquire_foundation_lock(
        Path(runtime.foundation_root), timeout_s=0.0
    )
    assert acquired_path == lock_path
    assert lock_fd is not None
    assert error is None
    foundation_runtime_module._release_pid_lock(acquired_path, lock_fd)


def test_simready_clone_foundation_removes_partial_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "simready-foundation"

    def fail_clone(*_args, **_kwargs):
        root.mkdir()
        (root / ".git").mkdir()
        return SimpleNamespace(returncode=1, stderr="clone failed")

    monkeypatch.setattr(foundation_runtime_module.subprocess, "run", fail_clone)

    error = foundation_runtime_module._clone_foundation(root, ref="main")

    assert error is not None
    assert "Failed to clone SimReady Foundation" in error
    assert not root.exists()


def test_simready_validation_normalizes_failed_profile(tmp_path: Path) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_fake_venv(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")
    report_path = tmp_path / "simready-profile.json"

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(report_path),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert not report.passed
    assert report.status == "FAIL"
    assert report.needs_rerun
    assert report.rerun_reasons == ["NP.006"]
    assert report.next_step == "simready-conform-profile"
    assert report.report_path == str(report_path.resolve())
    assert Path(report.raw_report_path or "").exists()
    assert (
        json.loads(report_path.read_text(encoding="utf-8"))["issues"][0][
            "requirement_id"
        ]
        == "NP.006"
    )
    run_manifest_path = Path(report.workflow_run_manifest_path or "")
    assert run_manifest_path == (
        tmp_path / ".simready-profile.json.workflow-run" / "workflow_run_manifest.json"
    )
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert run_manifest["workflow"] == "simready_profile_validation"
    assert run_manifest["status"] == "fail"
    assert run_manifest["source_path"] == str(asset.resolve())
    assert run_manifest["source_sha256"]
    assert run_manifest["backend"]["validator"] == "simready-validate"
    assert run_manifest["policy"] == {
        "install_missing": False,
        "profile": "Prop-Robotics-Neutral",
        "profile_version": "1.0.0",
        "update_foundation": False,
    }
    assert run_manifest["required_artifacts"] == ["validation_report"]
    assert run_manifest["checkpoints"][-1]["phase"] == "validated"

    resumed = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(report_path),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
            resume=True,
        )
    )
    resumed_manifest = json.loads(
        Path(resumed.workflow_run_manifest_path or "").read_text(encoding="utf-8")
    )
    assert [item["phase"] for item in resumed_manifest["checkpoints"]] == [
        "validated",
        "validated",
    ]


def test_simready_validation_isolates_validator_environment(monkeypatch) -> None:
    injected = {
        "AWS_SECRET_ACCESS_KEY": "secret",
        "DYLD_INSERT_LIBRARIES": "/runtime/injected.dylib",
        "LD_LIBRARY_PATH": "/runtime/native-libraries",
        "LD_PRELOAD": "/runtime/injected.so",
        "PYTHONHOME": "/runtime/python-home",
        "PYTHONINSPECT": "1",
        "PYTHONPATH": "/runtime/openusd:/runtime/packages",
        "PXR_PLUGINPATH_NAME": "/runtime/usd-plugins",
        "USD_PLUGIN_PATH": "/runtime/legacy-usd-plugins",
        "VIRTUAL_ENV": "/runtime/venv",
    }
    for name, value in injected.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("SIMREADY_TEST_NOT_PRESERVED", "discarded")

    validator = "/managed/validator/bin/simready-validate"
    environment = validate_profile_module._validator_subprocess_environment(validator)

    assert not injected.keys() & environment.keys()
    assert "SIMREADY_TEST_NOT_PRESERVED" not in environment
    assert environment["LANG"] == "C.UTF-8"
    assert environment["PATH"].split(os.pathsep)[0] == str(
        Path(validator).resolve().parent
    )
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONSAFEPATH"] == "1"


@pytest.mark.parametrize(("descriptor", "log_name"), [(1, "stdout"), (2, "stderr")])
def test_bounded_validator_subprocess_retains_exact_prefix_and_fails(
    tmp_path: Path,
    descriptor: int,
    log_name: str,
) -> None:
    stdout_path = tmp_path / "validator.stdout.log"
    stderr_path = tmp_path / "validator.stderr.log"
    limit = 257

    returncode, error = validate_profile_module._run_bounded_validator_subprocess(
        [
            sys.executable,
            "-c",
            f"import os; os.write({descriptor}, b'x' * 65536)",
        ],
        cwd=tmp_path,
        env=dict(os.environ),
        stdout_log_path=stdout_path,
        stderr_log_path=stderr_path,
        timeout_s=10.0,
        stdout_limit_bytes=limit,
        stderr_limit_bytes=limit,
    )

    selected = stdout_path if log_name == "stdout" else stderr_path
    other = stderr_path if log_name == "stdout" else stdout_path
    assert returncode == 1
    assert error is not None
    assert error.startswith("simready_validator_output_limit_exceeded:")
    assert selected.read_bytes() == b"x" * limit
    assert other.read_bytes() == b""


@pytest.mark.skipif(os.name != "nt", reason="Windows pipe-pump regression")
def test_bounded_validator_fails_early_on_small_output_overflow(
    tmp_path: Path,
) -> None:
    stdout_path = tmp_path / "validator.stdout.log"
    stderr_path = tmp_path / "validator.stderr.log"
    limit = 257
    started = time.monotonic()

    returncode, error = validate_profile_module._run_bounded_validator_subprocess(
        [
            sys.executable,
            "-c",
            f"import os, time; os.write(1, b'x' * {limit + 1}); time.sleep(10)",
        ],
        cwd=tmp_path,
        env=dict(os.environ),
        stdout_log_path=stdout_path,
        stderr_log_path=stderr_path,
        timeout_s=5.0,
        stdout_limit_bytes=limit,
        stderr_limit_bytes=limit,
    )

    assert time.monotonic() - started < 2.0
    assert returncode == 1
    assert error is not None
    assert error.startswith("simready_validator_output_limit_exceeded:")
    assert stdout_path.read_bytes() == b"x" * limit
    assert stderr_path.read_bytes() == b""


@pytest.mark.skipif(os.name == "nt", reason="Gate 3 process containment is Linux-only")
def test_bounded_validator_waits_for_leader_after_both_output_pipes_close(
    tmp_path: Path,
) -> None:
    stdout_path = tmp_path / "validator.stdout.log"
    stderr_path = tmp_path / "validator.stderr.log"
    report_path = tmp_path / "validator-report.json"

    returncode, error = validate_profile_module._run_bounded_validator_subprocess(
        [
            sys.executable,
            "-c",
            (
                "import os, pathlib, sys, time; "
                "null_fd = os.open(os.devnull, os.O_WRONLY); "
                "os.dup2(null_fd, 1); os.dup2(null_fd, 2); os.close(null_fd); "
                "time.sleep(0.25); "
                "pathlib.Path(sys.argv[1]).write_text('complete', encoding='utf-8')"
            ),
            str(report_path),
        ],
        cwd=tmp_path,
        env=dict(os.environ),
        stdout_log_path=stdout_path,
        stderr_log_path=stderr_path,
        timeout_s=2.0,
        stdout_limit_bytes=1024,
        stderr_limit_bytes=1024,
    )

    assert returncode == 0
    assert error is None
    assert report_path.read_text(encoding="utf-8") == "complete"
    assert stdout_path.read_bytes() == b""
    assert stderr_path.read_bytes() == b""


@pytest.mark.skipif(os.name == "nt", reason="Gate 3 process containment is Linux-only")
def test_bounded_validator_times_out_after_both_output_pipes_close(
    tmp_path: Path,
) -> None:
    stdout_path = tmp_path / "validator.stdout.log"
    stderr_path = tmp_path / "validator.stderr.log"
    late_report_path = tmp_path / "late-validator-report.json"

    returncode, error = validate_profile_module._run_bounded_validator_subprocess(
        [
            sys.executable,
            "-c",
            (
                "import os, pathlib, sys, time; "
                "null_fd = os.open(os.devnull, os.O_WRONLY); "
                "os.dup2(null_fd, 1); os.dup2(null_fd, 2); os.close(null_fd); "
                "time.sleep(10); "
                "pathlib.Path(sys.argv[1]).write_text('too late', encoding='utf-8')"
            ),
            str(late_report_path),
        ],
        cwd=tmp_path,
        env=dict(os.environ),
        stdout_log_path=stdout_path,
        stderr_log_path=stderr_path,
        timeout_s=0.2,
        stdout_limit_bytes=1024,
        stderr_limit_bytes=1024,
    )

    assert returncode == 1
    assert error == "simready_validator_timeout_exceeded: exceeded 0.2 seconds"
    assert not late_report_path.exists()
    assert stdout_path.read_bytes() == b""
    assert stderr_path.read_bytes() == b""


def test_bounded_validator_cleans_up_descendant_after_leader_exit(
    tmp_path: Path,
) -> None:
    stdout_path = tmp_path / "validator.stdout.log"
    stderr_path = tmp_path / "validator.stderr.log"
    descendant_marker = tmp_path / "descendant-survived.txt"
    descendant_code = (
        "import pathlib, sys, time; "
        "time.sleep(0.5); "
        "pathlib.Path(sys.argv[1]).write_text('survived', encoding='utf-8')"
    )
    leader_code = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {descendant_code!r}, sys.argv[1]])"
    )

    returncode, error = validate_profile_module._run_bounded_validator_subprocess(
        [sys.executable, "-c", leader_code, str(descendant_marker)],
        cwd=tmp_path,
        env=dict(os.environ),
        stdout_log_path=stdout_path,
        stderr_log_path=stderr_path,
        timeout_s=2.0,
        stdout_limit_bytes=1024,
        stderr_limit_bytes=1024,
    )

    assert returncode == 0
    assert error is None
    time.sleep(0.7)
    assert not descendant_marker.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
@pytest.mark.parametrize("failure_kind", ["timeout", "output_limit"])
def test_bounded_validator_windows_failure_cleans_up_descendants(
    tmp_path: Path,
    failure_kind: str,
) -> None:
    stdout_path = tmp_path / "validator.stdout.log"
    stderr_path = tmp_path / "validator.stderr.log"
    descendant_marker = tmp_path / f"{failure_kind}-descendant-survived.txt"
    descendant_ready = tmp_path / f"{failure_kind}-descendant-started.txt"
    descendant_code = (
        "import pathlib, sys, time; "
        "pathlib.Path(sys.argv[2]).write_text('started', encoding='utf-8'); "
        "time.sleep(0.5); "
        "pathlib.Path(sys.argv[1]).write_text('survived', encoding='utf-8')"
    )
    limit = 257
    failure_action = (
        "time.sleep(10)"
        if failure_kind == "timeout"
        else f"os.write(1, b'x' * {limit + 1}); time.sleep(10)"
    )
    leader_code = (
        "import os, pathlib, subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {descendant_code!r}, "
        "sys.argv[1], sys.argv[2]]); "
        "deadline = time.monotonic() + 2; "
        "ready = pathlib.Path(sys.argv[2]); "
        'exec("while not ready.exists() and time.monotonic() < deadline:\\n '
        '   time.sleep(0.01)"); '
        f"{failure_action}"
    )
    started = time.monotonic()

    returncode, error = validate_profile_module._run_bounded_validator_subprocess(
        [
            sys.executable,
            "-c",
            leader_code,
            str(descendant_marker),
            str(descendant_ready),
        ],
        cwd=tmp_path,
        env=dict(os.environ),
        stdout_log_path=stdout_path,
        stderr_log_path=stderr_path,
        timeout_s=0.2 if failure_kind == "timeout" else 5.0,
        stdout_limit_bytes=limit,
        stderr_limit_bytes=limit,
    )

    assert time.monotonic() - started < 2.0
    assert returncode == 1
    assert error is not None
    assert error.startswith(f"simready_validator_{failure_kind}_exceeded:")
    assert descendant_ready.read_text(encoding="utf-8") == "started"
    time.sleep(0.7)
    assert not descendant_marker.exists()


def test_simready_log_limits_must_be_configured_as_a_pair() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "max_stdout_log_bytes and max_stderr_log_bytes must be provided together"
        ),
    ):
        SimReadyValidationInput(
            asset_path="asset.usda",
            max_stdout_log_bytes=1024,
        )


@pytest.mark.skipif(os.name == "nt", reason="Gate 3 process containment is Linux-only")
def test_bounded_validator_drain_waits_through_an_empty_poll() -> None:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"late forensic bytes")
    os.close(write_fd)
    destination = io.BytesIO()
    key = SimpleNamespace(
        fd=read_fd,
        fileobj=read_fd,
        data=(destination, 1024),
    )

    class DelayedSelector:
        def __init__(self) -> None:
            self.active = True
            self.poll_count = 0

        def get_map(self) -> dict[int, object]:
            return {read_fd: key} if self.active else {}

        def select(self, _timeout: float) -> list[tuple[object, int]]:
            self.poll_count += 1
            if self.poll_count == 1:
                return []
            return [(key, 1)]

        def unregister(self, _fileobj: object) -> None:
            self.active = False

    selector = DelayedSelector()
    try:
        exceeded = validate_profile_module._drain_ready_validator_output(
            selector,  # type: ignore[arg-type]
            written={read_fd: 0},
        )
    finally:
        os.close(read_fd)

    assert exceeded is False
    assert destination.getvalue() == b"late forensic bytes"
    assert selector.poll_count >= 2


@pytest.mark.skipif(os.name == "nt", reason="Gate 3 process containment is Linux-only")
@pytest.mark.parametrize(("descriptor", "log_name"), [(1, "stdout"), (2, "stderr")])
def test_simready_output_limit_overrides_passing_validator_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    descriptor: int,
    log_name: str,
) -> None:
    monkeypatch.setenv(SIMREADY_CACHE_DIR_ENV, str(tmp_path / "simready-cache"))
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_flooding_pass_venv(tmp_path, descriptor=descriptor)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")
    report_path = tmp_path / "simready-profile.json"
    limit = 257

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(report_path),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
            max_stdout_log_bytes=limit,
            max_stderr_log_bytes=limit,
        )
    )

    selected_path = Path(
        report.stdout_log_path if log_name == "stdout" else report.stderr_log_path
    )
    assert report.status == "ERROR"
    assert report.passed is False
    assert report.next_step == "fix-simready-validator-runtime"
    assert any(
        error.startswith("simready_validator_output_limit_exceeded:")
        for error in report.errors
    )
    assert selected_path.read_bytes() == b"x" * limit


def test_simready_validation_stages_usdz_root_and_preserves_original(
    tmp_path: Path,
) -> None:
    from pxr import Usd

    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_asset_capture_venv(tmp_path)
    asset = tmp_path / "asset.usdz"
    _write_validation_usdz(asset)
    assert Usd.Stage.Open(str(asset)) is not None
    original_bytes = asset.read_bytes()
    original_mtime_ns = asset.stat().st_mtime_ns
    report_path = tmp_path / "simready-profile.json"

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(report_path),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    staging = report.validation_policy["usdz_validation_staging"]
    validator_target = Path(staging["validator_target"])
    assert report.passed
    assert report.status == "PASS"
    assert report.asset_path == str(asset.resolve())
    assert report.command[-1] == str(asset.resolve())
    assert staging == {
        "original_asset_path": str(asset.resolve()),
        "package_root": "root.usda",
        "validator_target": str(validator_target),
        "workspace_path": str(validator_target.parents[0]),
        "workspace_cleanup": "removed",
    }
    assert validator_target.suffix == ".usda"
    assert validator_target != asset.resolve()
    assert not validator_target.exists()
    assert report.profile_results["validator_target"] == str(validator_target)
    assert report.profile_results["relative_dependency_available"]
    assert any("normalized evidence remains bound" in item for item in report.warnings)
    assert str(validator_target) in json.loads(
        Path(report.raw_report_path or "").read_text(encoding="utf-8")
    )
    assert asset.read_bytes() == original_bytes
    assert asset.stat().st_mtime_ns == original_mtime_ns


def test_simready_support_budget_presizes_normalized_report(tmp_path: Path) -> None:
    report_path = tmp_path / "simready-profile.json"

    with pytest.raises(
        RuntimeError,
        match=r"simready_support_budget_exceeded: file_bytes limit=1",
    ):
        run_simready_profile_validation(
            SimReadyValidationInput(
                asset_path=str(tmp_path / "missing.usda"),
                report_path=str(report_path),
                max_support_file_bytes=1,
                max_support_total_bytes=1,
                max_support_file_count=1,
                max_support_directory_count=1,
            )
        )

    assert not report_path.exists()


def test_simready_support_budget_rejects_raw_report_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_report = tmp_path / "raw.json"
    with raw_report.open("wb") as stream:
        stream.seek(1024 * 1024)
        stream.write(b"x")

    monkeypatch.setattr(
        validate_profile_module.os,
        "read",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oversized raw report must not be read")
        ),
    )

    payload = validate_profile_module._load_json_if_present(
        raw_report,
        max_bytes=1024,
    )

    assert payload[validate_profile_module.TOOLCHAIN_ERROR_KEY] is True
    assert any(
        "simready_support_budget_exceeded: file_bytes limit=1024" in error
        for error in payload["errors"]
    )


def test_simready_raw_report_open_fails_closed_when_confined_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_report = tmp_path / "raw.json"
    raw_report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        validate_profile_module,
        "open_held_confined_artifact",
        lambda *_args: (_ for _ in ()).throw(
            validate_profile_module.ArtifactPathError("confined open unavailable")
        ),
    )

    payload = validate_profile_module._load_json_if_present(raw_report)

    assert payload[validate_profile_module.TOOLCHAIN_ERROR_KEY] is True
    assert any("confined open unavailable" in error for error in payload["errors"])


def test_unbounded_raw_report_read_stops_at_opened_size_when_file_keeps_growing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_report = tmp_path / "raw.json"
    raw_report.write_bytes(b"{}")
    requested_sizes: list[int] = []

    def endlessly_growing_read(_descriptor: int, size: int) -> bytes:
        requested_sizes.append(size)
        return b"x" * size

    monkeypatch.setattr(validate_profile_module.os, "read", endlessly_growing_read)

    payload = validate_profile_module._load_json_if_present(raw_report)

    assert payload[validate_profile_module.TOOLCHAIN_ERROR_KEY] is True
    assert any("changed while it was read" in error for error in payload["errors"])
    assert requested_sizes == [raw_report.stat().st_size + 1]


@pytest.mark.parametrize(
    ("member_name", "payload_size", "directory_limit", "dimension"),
    (
        ("root.usda", 64 * 1024 + 1, 8, "file_bytes"),
        ("a/b/c/root.usda", 16, 3, "directory_count"),
    ),
)
def test_simready_support_budget_rejects_usdz_before_workspace_creation(
    tmp_path: Path,
    member_name: str,
    payload_size: int,
    directory_limit: int,
    dimension: str,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_asset_capture_venv(tmp_path)
    asset = tmp_path / "budgeted.usdz"
    with ZipFile(asset, "w") as archive:
        archive.writestr(member_name, b"x" * payload_size)
    report_path = tmp_path / "simready-profile.json"

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(report_path),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
            max_support_file_bytes=64 * 1024,
            max_support_total_bytes=128 * 1024,
            max_support_file_count=8,
            max_support_directory_count=directory_limit,
        )
    )

    assert report.status == "BLOCKED"
    assert any(
        f"simready_support_budget_exceeded: {dimension}" in error
        for error in report.errors
    )
    assert not validate_profile_module._validation_workspace_path(
        report_path.resolve()
    ).exists()


def test_simready_support_budget_must_be_complete() -> None:
    with pytest.raises(
        ValueError,
        match="all SimReady support budget limits must be provided together",
    ):
        SimReadyValidationInput(
            asset_path="asset.usda",
            max_support_file_bytes=1024,
        )


def test_simready_validation_rejects_usdz_traversal_member(tmp_path: Path) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_asset_capture_venv(tmp_path)
    asset = tmp_path / "unsafe.usdz"
    with ZipFile(asset, "w") as archive:
        archive.writestr("root.usda", '#usda 1.0\n\ndef Xform "World" {}\n')
        archive.writestr("../escaped.usda", "#usda 1.0\n")
    original_bytes = asset.read_bytes()
    report_path = tmp_path / "simready-profile.json"

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(report_path),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    workspace = validate_profile_module._validation_workspace_path(
        report_path.resolve()
    )
    assert not report.passed
    assert report.status == "BLOCKED"
    assert report.next_step == "fix-simready-usdz-package"
    assert any("unsafe archive member" in item for item in report.errors)
    assert not (tmp_path / "escaped.usda").exists()
    assert not workspace.exists()
    assert not Path(report.raw_report_path or "").exists()
    assert Path(report.stdout_log_path or "").read_text(encoding="utf-8") == ""
    assert Path(report.stderr_log_path or "").read_text(encoding="utf-8") == ""
    assert asset.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("package_case", "expected_error"),
    [
        ("missing", "no package root"),
        ("unsupported", "unsupported SimReady validator suffix"),
        ("ambiguous", "package root is ambiguous"),
        ("malformed", "Malformed USDZ archive"),
    ],
)
def test_simready_validation_rejects_bad_usdz_package_root(
    tmp_path: Path,
    package_case: str,
    expected_error: str,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_asset_capture_venv(tmp_path)
    asset = tmp_path / f"{package_case}.usdz"
    if package_case == "malformed":
        asset.write_bytes(b"not a zip archive")
    else:
        with ZipFile(asset, "w") as archive:
            if package_case == "unsupported":
                archive.writestr("root.usdc", b"PXR-USDC")
                archive.writestr("fallback.usda", "#usda 1.0\n")
            elif package_case == "ambiguous":
                archive.writestr("root.usda", "#usda 1.0\n")
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr("root.usda", "#usda 1.0\n")
    report_path = tmp_path / "simready-profile.json"

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(report_path),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert not report.passed
    assert report.status == "BLOCKED"
    assert any(expected_error in item for item in report.errors)
    assert not validate_profile_module._validation_workspace_path(
        report_path.resolve()
    ).exists()
    assert not Path(report.raw_report_path or "").exists()


def test_simready_validation_rejects_usdz_workspace_collision(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_asset_capture_venv(tmp_path)
    asset = tmp_path / "asset.usdz"
    _write_validation_usdz(asset)
    report_path = tmp_path / "simready-profile.json"
    workspace = validate_profile_module._validation_workspace_path(
        report_path.resolve()
    )
    workspace.mkdir()
    marker = workspace / "owner.txt"
    marker.write_text("existing owner", encoding="utf-8")

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(report_path),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert not report.passed
    assert report.status == "BLOCKED"
    assert any("already exists" in item for item in report.errors)
    assert marker.read_text(encoding="utf-8") == "existing owner"
    assert not Path(report.raw_report_path or "").exists()


def test_simready_validation_keeps_usda_validator_path_unchanged(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_asset_capture_venv(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {}\n', encoding="utf-8")
    report_path = tmp_path / "simready-profile.json"

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(report_path),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert report.passed
    assert report.status == "PASS"
    assert report.asset_path == str(asset.resolve())
    assert report.command[-1] == str(asset.resolve())
    assert report.profile_results["validator_target"] == str(asset.resolve())
    assert "usdz_validation_staging" not in report.validation_policy
    assert not report.warnings
    assert not validate_profile_module._validation_workspace_path(
        report_path.resolve()
    ).exists()


def test_simready_validation_binds_and_rechecks_composed_dependencies(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_dependency_mutating_venv(tmp_path)
    layers = tmp_path / "layers"
    layers.mkdir()
    dependency = layers / "sub.usda"
    dependency.write_text(
        '#usda 1.0\n\ndef Xform "Dependency" {}\n',
        encoding="utf-8",
    )
    asset = tmp_path / "asset.usda"
    asset.write_text(
        "#usda 1.0\n(\n    subLayers = [@layers/sub.usda@]\n)\n",
        encoding="utf-8",
    )

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(tmp_path / "simready-profile.json"),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    manifest_paths = {
        item["path"] for item in report.asset_dependency_manifest.get("files", [])
    }
    assert manifest_paths == {str(asset.resolve()), str(dependency.resolve())}
    assert not report.passed
    assert report.status == "ERROR"
    assert any(
        "resolved USD dependency content changed" in item for item in report.errors
    )


def test_simready_validation_normalizes_malformed_raw_json(tmp_path: Path) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_malformed_json_venv(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(tmp_path / "simready-profile.json"),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert not report.passed
    assert report.status == "ERROR"
    assert any("could not be parsed" in item for item in report.errors)
    assert report.next_step == "fix-simready-validator-runtime"


def test_simready_validation_status_error_uses_profile_rerun_next_step(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_error_status_venv(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(tmp_path / "simready-profile.json"),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert not report.passed
    assert report.status == "ERROR"
    assert report.needs_rerun
    assert report.rerun_reasons == ["NP.006"]
    assert report.next_step == "simready-conform-profile"


def test_simready_validation_cli_non_strict_allows_profile_error_status(
    tmp_path: Path,
    capsys,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_error_status_venv(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")

    code = validate_profile_module.main(
        [
            str(asset),
            "--foundation-root",
            str(foundation_root),
            "--venv",
            str(venv),
            "--no-install-missing",
            "--report",
            str(tmp_path / "simready-profile.json"),
        ]
    )

    capsys.readouterr()
    assert code == 0


def test_simready_validation_normalizes_invalid_utf8_raw_json(tmp_path: Path) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_invalid_utf8_venv(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(tmp_path / "simready-profile.json"),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert not report.passed
    assert report.status == "ERROR"
    assert any("could not be parsed" in item for item in report.errors)
    assert report.next_step == "fix-simready-validator-runtime"


def test_simready_validation_rejects_stale_pass_when_fresh_raw_report_missing(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_no_report_venv(tmp_path, exit_code=17)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")
    report_path = tmp_path / "simready-profile.json"
    stale_raw_report = report_path.with_suffix(".raw.json")
    stale_raw_report.write_text(
        json.dumps({"passed": True, "status": "PASS"}),
        encoding="utf-8",
    )

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(report_path),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert not report.passed
    assert report.status == "ERROR"
    assert any("raw report was not written" in item for item in report.errors)
    assert report.raw_report_path is not None
    assert Path(report.raw_report_path) == stale_raw_report.resolve()
    assert not stale_raw_report.exists()
    assert report.next_step == "fix-simready-validator-runtime"


def test_simready_validation_ignores_rb_mb001_for_single_component_asset(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_fake_features_summary_venv(tmp_path)
    asset = tmp_path / "coffee_mug.usda"
    _write_single_mesh_asset(asset)
    report_path = tmp_path / "simready-profile.json"

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            profile="Prop-Robotics-Physx",
            profile_version="1.0.0",
            report_path=str(report_path),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert report.passed
    assert report.status == "PASS"
    assert not report.warnings
    assert not report.issues
    assert not report.needs_rerun
    assert report.rerun_reasons == []
    assert report.next_step == "complete"
    assert report.validation_policy["single_component_requirement_ignored"]
    assert report.ignored_issues == [
        {
            "feature_id": "FET004_BASE_PHYSX",
            "ignored_reason": (
                "Physical AI Skill Hub SimReady conformance policy treats "
                "RB.MB.001 as non-blocking/not applicable for single-body props "
                "and forbids inventing rigid bodies to satisfy it."
            ),
            "message": (
                "RB.MB.001 is not applicable to a single-component or explicitly "
                "single-rigid-body prop."
            ),
            "requirement_id": "RB.MB.001",
            "severity": "IGNORED",
        }
    ]


def test_simready_validation_accepts_nonzero_exit_for_ignored_rb_mb001(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_fake_features_summary_venv(tmp_path, exit_code=1)
    asset = tmp_path / "coffee_mug.usda"
    _write_single_mesh_asset(asset)

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            profile="Prop-Robotics-Physx",
            profile_version="1.0.0",
            report_path=str(tmp_path / "simready-profile.json"),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert report.passed
    assert report.status == "PASS"
    assert report.rerun_reasons == []
    assert report.validation_policy["single_component_requirement_ignored"]


def test_simready_validation_rejects_unexpected_exit_for_ignored_rb_mb001(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_fake_features_summary_venv(tmp_path, exit_code=2)
    asset = tmp_path / "coffee_mug.usda"
    _write_single_mesh_asset(asset)

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            profile="Prop-Robotics-Physx",
            profile_version="1.0.0",
            report_path=str(tmp_path / "simready-profile.json"),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert not report.passed
    assert report.status == "ERROR"
    assert report.next_step == "fix-simready-validator-runtime"
    assert any("unexpected status 2" in error for error in report.errors)


def test_simready_validation_counts_mixed_mesh_and_subset_components(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_fake_features_summary_venv(tmp_path)
    asset = tmp_path / "mixed.usda"
    _write_mixed_subset_asset(asset)

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            profile="Prop-Robotics-Physx",
            profile_version="1.0.0",
            report_path=str(tmp_path / "simready-profile.json"),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert report.asset_topology["mesh_count"] == 2
    assert report.asset_topology["geom_subset_count"] == 1
    assert report.asset_topology["mesh_with_geom_subset_count"] == 1
    assert report.asset_topology["component_count"] == 2
    assert not report.asset_topology["single_prim_or_geomsubset"]
    assert not report.passed
    assert report.rerun_reasons == ["RB.MB.001"]


def test_simready_validation_ignores_rb_mb001_for_authored_single_rigid_body(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_fake_features_summary_venv(tmp_path)
    asset = tmp_path / "single_body_multi_mesh.usda"
    _write_mixed_subset_asset(asset)
    stage = Usd.Stage.Open(str(asset))
    assert stage is not None
    UsdPhysics.RigidBodyAPI.Apply(stage.GetDefaultPrim())
    stage.GetRootLayer().Save()

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            profile="Prop-Robotics-Physx",
            profile_version="1.0.0",
            report_path=str(tmp_path / "simready-profile.json"),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert report.asset_topology["component_count"] == 2
    assert report.asset_topology["rigid_body_count"] == 1
    assert report.asset_topology["single_rigid_body"]
    assert report.passed
    assert report.rerun_reasons == []


def test_simready_validation_does_not_ignore_rb_mb001_without_mesh_components(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_fake_features_summary_venv(tmp_path)
    asset = tmp_path / "xform_only.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            profile="Prop-Robotics-Physx",
            profile_version="1.0.0",
            report_path=str(tmp_path / "simready-profile.json"),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert report.asset_topology["mesh_count"] == 0
    assert report.asset_topology["component_count"] == 0
    assert not report.asset_topology["single_prim_or_geomsubset"]
    assert not report.passed
    assert report.rerun_reasons == ["RB.MB.001"]


def test_simready_validation_blocks_when_dependency_inspection_cannot_open_asset(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_fake_venv(tmp_path)
    asset = tmp_path / "malformed.usda"
    asset.write_text("not a usd file", encoding="utf-8")

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset),
            report_path=str(tmp_path / "simready-profile.json"),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )

    assert report.status == "BLOCKED"
    assert not report.passed
    assert report.next_step == "resolve-usd-dependencies"
    assert report.asset_dependency_manifest == {}
    assert any(
        "Could not bind asset dependency identity" in error for error in report.errors
    )
    assert any(
        "OpenUSD could not enumerate dependencies" in error for error in report.errors
    )


def test_simready_conformance_repairs_identity_bound_failed_requirement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    foundation_root = _write_fake_foundation(tmp_path)
    runtime = _trusted_conformance_runtime(foundation_root)
    monkeypatch.setattr(
        conform_profile_module,
        "resolve_simready_runtime",
        lambda **_kwargs: runtime,
    )
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")
    validation_report = tmp_path / "simready-profile.json"
    _write_trusted_validation_report(
        validation_report,
        asset_path=asset,
        runtime=runtime,
        rerun_reasons=["NP.006"],
        extra={
            "issues": [
                {
                    "requirement_id": "NP.006",
                    "severity": "ERROR",
                    "message": "Missing SimReady metadata.",
                }
            ]
        },
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            validation_report_path=str(validation_report),
            foundation_root=str(foundation_root),
        )
    )

    assert report.passed
    assert report.status == "PASS"
    assert report.failed_requirements == ["NP.006"]
    assert report.requirements_repaired == ["NP.006"]
    assert report.requirements_blocked == []
    assert (
        report.steps[0]["upstream_skill"] == "simready-foundation-conform-fet-000-core"
    )
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    assert "SimReady_Metadata" in repaired_stage.GetRootLayer().customLayerData


def test_np005_repair_blocks_when_dependency_discovery_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "widget.usda"
    asset.write_text('#usda 1.0\ndef Xform "Widget" {}\n', encoding="utf-8")
    monkeypatch.setattr(
        conform_profile_module,
        "_usd_dependency_paths",
        lambda _path: ([asset], ["OpenUSD dependency inspection unavailable"]),
    )

    result = conform_profile_module._repair_asset_folder_structure(
        requirement="NP.005",
        asset_path=asset,
        output_dir=tmp_path / "conform",
    )

    assert result.status == "BLOCKED"
    assert result.passed is False
    assert result.report["dependency_warnings"] == [
        "OpenUSD dependency inspection unavailable"
    ]


def test_np005_repair_packages_and_rewrites_external_texture_dependency(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    source_dir = tmp_path / "source"
    texture = source_dir / "textures/albedo.png"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"png fixture")
    asset = source_dir / "widget.usda"
    asset.write_text(
        '#usda 1.0\n\ndef Xform "Widget" {\n'
        "    custom asset texture = @textures/albedo.png@\n}\n",
        encoding="utf-8",
    )

    result = conform_profile_module._repair_asset_folder_structure(
        requirement="NP.005",
        asset_path=asset,
        output_dir=tmp_path / "conform",
    )

    assert result.status == "REPAIRED"
    assert result.passed is True
    assert result.package_root is not None
    output = result.output_path
    repaired_stage = Usd.Stage.Open(str(output))
    assert repaired_stage is not None
    authored = repaired_stage.GetPrimAtPath("/Widget").GetAttribute("texture").Get()
    assert authored is not None
    assert authored.path.startswith("../dependencies/")
    relocated_texture = (output.parent / authored.path).resolve()
    assert relocated_texture.read_bytes() == texture.read_bytes()
    assert relocated_texture.is_relative_to(result.package_root.resolve())
    assert len(result.report["output_tree_sha256"]) == 64
    assert result.report["packaged_dependencies"] == [
        {
            "source_path": str(texture.resolve()),
            "output_path": str(relocated_texture),
            "sha256": hashlib.sha256(texture.read_bytes()).hexdigest(),
        }
    ]
    repeated = conform_profile_module._repair_asset_folder_structure(
        requirement="NP.005",
        asset_path=asset,
        output_dir=tmp_path / "conform",
    )
    assert repeated.status == "REPAIRED"
    assert repeated.report["reused_output"] is True
    assert repeated.output_path == result.output_path


def test_np005_repair_rewrites_nested_usd_layer_dependencies(tmp_path: Path) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    source_dir = tmp_path / "source"
    texture = source_dir / "textures/albedo.png"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"nested png fixture")
    child = source_dir / "layers/child.usda"
    child.parent.mkdir(parents=True)
    child.write_text(
        '#usda 1.0\n\ndef Xform "Child" {\n'
        "    custom asset texture = @../textures/albedo.png@\n}\n",
        encoding="utf-8",
    )
    asset = source_dir / "widget.usda"
    asset.write_text(
        "#usda 1.0\n(\n    subLayers = [@layers/child.usda@]\n)\n"
        '\ndef Xform "Widget" {}\n',
        encoding="utf-8",
    )

    result = conform_profile_module._repair_asset_folder_structure(
        requirement="NP.005",
        asset_path=asset,
        output_dir=tmp_path / "conform",
    )

    assert result.status == "REPAIRED"
    assert result.package_root is not None
    repaired_root = Sdf.Layer.FindOrOpen(str(result.output_path))
    assert repaired_root is not None
    assert len(repaired_root.subLayerPaths) == 1
    copied_child = (
        result.output_path.parent / repaired_root.subLayerPaths[0]
    ).resolve()
    child_stage = Usd.Stage.Open(str(copied_child))
    assert child_stage is not None
    authored = child_stage.GetPrimAtPath("/Child").GetAttribute("texture").Get()
    assert authored is not None
    relocated_texture = (copied_child.parent / authored.path).resolve()
    assert relocated_texture.read_bytes() == texture.read_bytes()
    assert relocated_texture.is_relative_to(result.package_root.resolve())


def test_aa001_accepts_np005_content_addressed_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "widget.usda"
    asset.write_text('#usda 1.0\ndef Xform "Widget" {}\n', encoding="utf-8")
    monkeypatch.setattr(
        conform_profile_module,
        "_usd_dependency_paths",
        lambda _path: ([asset], []),
    )
    output_dir = tmp_path / "conform"
    np005 = conform_profile_module._repair_asset_folder_structure(
        requirement="NP.005",
        asset_path=asset,
        output_dir=output_dir,
    )

    assert np005.status == "REPAIRED"
    assert np005.package_root is not None
    source_tree, root_path, _, cleanup_dir = (
        conform_profile_module._aa001_source_package(
            asset_path=np005.output_path,
            package_root=np005.package_root,
            output_dir=output_dir,
        )
    )
    assert source_tree == np005.package_root.resolve()
    assert root_path == np005.output_path.resolve()
    assert cleanup_dir is None


def test_np005_repair_publishes_changed_input_to_new_content_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "widget.usda"
    asset.write_text('#usda 1.0\ndef Xform "Widget" {}\n', encoding="utf-8")
    monkeypatch.setattr(
        conform_profile_module,
        "_usd_dependency_paths",
        lambda _path: ([asset], []),
    )
    output_dir = tmp_path / "conform"

    first = conform_profile_module._repair_asset_folder_structure(
        requirement="NP.005",
        asset_path=asset,
        output_dir=output_dir,
    )
    first_bytes = first.output_path.read_bytes()
    asset.write_text(
        '#usda 1.0\ndef Xform "Widget" { custom string revision = "two" }\n',
        encoding="utf-8",
    )
    second = conform_profile_module._repair_asset_folder_structure(
        requirement="NP.005",
        asset_path=asset,
        output_dir=output_dir,
    )

    assert first.status == second.status == "REPAIRED"
    assert first.output_path != second.output_path
    assert first.report["output_tree_sha256"] != second.report["output_tree_sha256"]
    assert first.output_path.read_bytes() == first_bytes
    assert second.output_path.read_bytes() == asset.read_bytes()


@pytest.mark.parametrize("symlink_component", ["asset_root", "intermediate", "output"])
def test_np005_repair_rejects_symlinked_output_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_component: str,
) -> None:
    asset = tmp_path / "widget.usda"
    asset.write_text('#usda 1.0\ndef Xform "Widget" {}\n', encoding="utf-8")
    monkeypatch.setattr(
        conform_profile_module,
        "_usd_dependency_paths",
        lambda _path: ([asset], []),
    )
    output_dir = tmp_path / "conform"
    initial = conform_profile_module._repair_asset_folder_structure(
        requirement="NP.005",
        asset_path=asset,
        output_dir=output_dir,
    )
    assert initial.status == "REPAIRED"
    assert initial.package_root is not None
    asset_root = initial.package_root
    intermediate = asset_root / "simready_usd"
    output = intermediate / "widget.usda"
    victim_dir = tmp_path / "victim-dir"
    victim_dir.mkdir()
    if symlink_component == "asset_root":
        shutil.rmtree(asset_root)
        asset_root.symlink_to(victim_dir, target_is_directory=True)
    elif symlink_component == "intermediate":
        shutil.rmtree(intermediate)
        intermediate.symlink_to(victim_dir, target_is_directory=True)
    else:
        output.unlink()
        victim = tmp_path / "victim.usda"
        victim.write_bytes(asset.read_bytes())
        output.symlink_to(victim)

    result = conform_profile_module._repair_asset_folder_structure(
        requirement="NP.005",
        asset_path=asset,
        output_dir=output_dir,
    )

    assert result.status == "BLOCKED"
    assert result.passed is False
    assert "symlink" in result.reason


def test_simready_conformance_repairs_np005_before_metadata(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "physics.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["NP.006", "NP.005"],
            foundation_root=str(foundation_root),
        )
    )

    assert report.passed
    assert report.requirements_repaired == ["NP.005", "NP.006"]
    assert [step["requirement"] for step in report.steps] == ["NP.005", "NP.006"]
    output = Path(report.output_usd_path)
    assert output.name == "physics.usda"
    assert output.parent.name == "simready_usd"
    assert output.parent.parent.name == "physics"
    assert asset.read_bytes() == source_bytes
    repaired_stage = Usd.Stage.Open(str(output))
    assert repaired_stage is not None
    assert "SimReady_Metadata" in repaired_stage.GetRootLayer().customLayerData


def test_simready_conformance_blocks_np006_on_read_only_usdz(tmp_path: Path) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdUtils = pytest.importorskip("pxr.UsdUtils")
    foundation_root = _write_fake_foundation(tmp_path)
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    assert stage.GetRootLayer().Save()
    asset = tmp_path / "asset.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(source), str(asset))
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["NP.006"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["NP.006"]
    assert "read-only USDZ package layer" in report.steps[0]["reason"]
    output = Path(report.output_usd_path)
    assert output.read_bytes() == source_bytes
    assert asset.read_bytes() == source_bytes
    assert Path(report.reports["NP.006"]).is_file()


def test_simready_conformance_normalizes_z_up_meter_native_stage(
    tmp_path: Path,
) -> None:
    Gf = pytest.importorskip("pxr.Gf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = UsdGeom.Xform.Define(stage, "/asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    cube = UsdGeom.Cube.Define(stage, "/asset/cube")
    cube.CreateSizeAttr(1000.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(1000.0, 2000.0, 3000.0))
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    rigid_body.CreateVelocityAttr().Set(Gf.Vec3f(1000.0, 2000.0, 3000.0))
    physics_scene = UsdPhysics.Scene.Define(stage, "/asset/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9810.0)
    prismatic = UsdPhysics.PrismaticJoint.Define(stage, "/asset/prismatic")
    prismatic.CreateLowerLimitAttr(10.0)
    prismatic.CreateUpperLimitAttr(20.0)
    distance = UsdPhysics.DistanceJoint.Define(stage, "/asset/distance")
    distance.CreateMinDistanceAttr(30.0)
    distance.CreateMaxDistanceAttr(40.0)
    stage.GetRootLayer().Save()

    original_range = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(stage.GetDefaultPrim())
        .ComputeAlignedRange()
    )
    original_size_m = [float(value) * 0.001 for value in original_range.GetSize()]

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["UN.006", "UN.007"],
            foundation_root=str(foundation_root),
        )
    )

    assert report.passed
    assert report.requirements_repaired == ["UN.006", "UN.007"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    assert UsdGeom.GetStageUpAxis(repaired_stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(repaired_stage) == 1.0
    repaired_root = UsdGeom.Xformable(repaired_stage.GetDefaultPrim())
    assert list(repaired_root.GetXformOpOrderAttr().Get() or []) == []
    repaired_cube = UsdGeom.Xformable(repaired_stage.GetPrimAtPath("/asset/cube"))
    assert [str(token) for token in repaired_cube.GetXformOpOrderAttr().Get()] == [
        "xformOp:scale:simreadyMetersPerUnit",
        "xformOp:transform:simreadyUpAxis",
        "xformOp:translate",
    ]
    assert repaired_cube.GetOrderedXformOps()[-1].Get() == Gf.Vec3d(
        1000.0, 2000.0, 3000.0
    )
    repaired_scene = UsdPhysics.Scene.Get(repaired_stage, "/asset/PhysicsScene")
    assert list(repaired_scene.GetGravityDirectionAttr().Get()) == pytest.approx(
        [0.0, 0.0, -1.0]
    )
    assert repaired_scene.GetGravityMagnitudeAttr().Get() == pytest.approx(9.81)
    repaired_prismatic = UsdPhysics.PrismaticJoint.Get(
        repaired_stage, "/asset/prismatic"
    )
    assert repaired_prismatic.GetLowerLimitAttr().Get() == pytest.approx(0.01)
    assert repaired_prismatic.GetUpperLimitAttr().Get() == pytest.approx(0.02)
    repaired_distance = UsdPhysics.DistanceJoint.Get(repaired_stage, "/asset/distance")
    assert repaired_distance.GetMinDistanceAttr().Get() == pytest.approx(0.03)
    assert repaired_distance.GetMaxDistanceAttr().Get() == pytest.approx(0.04)
    repaired_velocity = UsdPhysics.RigidBodyAPI(
        repaired_stage.GetPrimAtPath("/asset/cube")
    ).GetVelocityAttr()
    assert list(repaired_velocity.Get()) == pytest.approx([1.0, 2.0, 3.0])
    repaired_range = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(repaired_stage.GetDefaultPrim())
        .ComputeAlignedRange()
    )
    assert list(repaired_range.GetSize()) == pytest.approx(original_size_m)


def test_simready_conformance_normalizes_every_stage_metric_frontier(
    tmp_path: Path,
) -> None:
    Gf = pytest.importorskip("pxr.Gf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = UsdGeom.Xform.Define(stage, "/asset")
    stage.SetDefaultPrim(root.GetPrim())
    group = UsdGeom.Xform.Define(stage, "/asset/group")
    UsdGeom.Cube.Define(stage, "/asset/group/cube")
    reset = UsdGeom.Xform.Define(stage, "/asset/group/reset")
    reset.SetResetXformStack(True)
    UsdGeom.Cube.Define(stage, "/asset/group/reset/cube")
    UsdGeom.Xform.Define(stage, "/Accessory")
    UsdGeom.Cube.Define(stage, "/Accessory/cube")
    UsdGeom.Scope.Define(stage, "/Scoped")
    UsdGeom.Xform.Define(stage, "/Scoped/Group")
    UsdGeom.Cube.Define(stage, "/Scoped/Group/cube")
    UsdGeom.Scope.Define(stage, "/Looks")
    stage.DefinePrim("/Looks/Material", "Material")
    UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["UN.006", "UN.007"],
            foundation_root=str(foundation_root),
        )
    )

    assert report.passed
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    assert Gf.IsClose(
        UsdGeom.Xformable(repaired_stage.GetDefaultPrim()).GetLocalTransformation(),
        Gf.Matrix4d(1.0),
        1e-9,
    )
    expected_orders = {
        "/asset/group": [
            "xformOp:scale:simreadyMetersPerUnit",
            "xformOp:transform:simreadyUpAxis",
        ],
        "/asset/group/reset": [
            "!resetXformStack!",
            "xformOp:scale:simreadyMetersPerUnit",
            "xformOp:transform:simreadyUpAxis",
        ],
        "/Accessory": [
            "xformOp:scale:simreadyMetersPerUnit",
            "xformOp:transform:simreadyUpAxis",
        ],
        "/Scoped/Group": [
            "xformOp:scale:simreadyMetersPerUnit",
            "xformOp:transform:simreadyUpAxis",
        ],
    }
    for prim_path, expected_order in expected_orders.items():
        xformable = UsdGeom.Xformable(repaired_stage.GetPrimAtPath(prim_path))
        assert [str(token) for token in xformable.GetXformOpOrderAttr().Get()] == (
            expected_order
        )
    assert list(group.GetXformOpOrderAttr().Get() or []) == []


def test_simready_conformance_blocks_geometry_bearing_default_prim(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    cube = UsdGeom.Cube.Define(stage, "/Cube")
    stage.SetDefaultPrim(cube.GetPrim())
    UsdGeom.Xform.Define(stage, "/Accessory")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["UN.006"],
            foundation_root=str(foundation_root),
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["UN.006"]
    assert "geometry-bearing default prim" in report.steps[0]["reason"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    assert UsdGeom.GetStageUpAxis(repaired_stage) == UsdGeom.Tokens.y


def test_simready_conformance_blocks_unauthored_source_units(tmp_path: Path) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = UsdGeom.Xform.Define(stage, "/asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Cube.Define(stage, "/asset/cube")
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            source_asset=str(tmp_path / "asset.step"),
            repair_requirements=["UN.007"],
            foundation_root=str(foundation_root),
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["UN.007"]
    assert "OpenUSD fallback is not source evidence" in report.steps[0]["reason"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    assert not repaired_stage.HasAuthoredMetadata("metersPerUnit")


def test_simready_conformance_blocks_unsupported_authored_physics_units(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = UsdGeom.Xform.Define(stage, "/asset")
    stage.SetDefaultPrim(root.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/asset/cube")
    mass = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    mass.CreateDensityAttr().Set(0.001)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    assert stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["UN.007"],
            foundation_root=str(foundation_root),
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["UN.007"]
    assert "unsupported authored physics quantities" in report.steps[0]["reason"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    assert UsdGeom.GetStageMetersPerUnit(repaired_stage) == pytest.approx(0.01)
    repaired_mass = UsdPhysics.MassAPI(repaired_stage.GetPrimAtPath("/asset/cube"))
    assert repaired_mass.GetDensityAttr().Get() == pytest.approx(0.001)


def test_simready_conformance_aa001_anchors_dependencies_atomically(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    package = tmp_path / "package"
    layers = package / "layers"
    textures = package / "textures"
    layers.mkdir(parents=True)
    textures.mkdir()
    (layers / "sub.usda").write_text(
        '#usda 1.0\n\nover "Asset" {\n    custom string sub = "kept"\n}\n',
        encoding="utf-8",
    )
    (layers / "reference.usda").write_text(
        '#usda 1.0\n\ndef Xform "Reference" {}\n',
        encoding="utf-8",
    )
    (layers / "payload.usda").write_text(
        '#usda 1.0\n\ndef Xform "Payload" {}\n',
        encoding="utf-8",
    )
    (textures / "a.png").write_bytes(b"texture-bytes")
    (package / "inventory.bin").write_bytes(b"inventory-bytes")
    asset = package / "asset.usda"
    asset.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
    subLayers = [@layers/sub.usda@]
)

def Xform "Asset" (
    prepend references = @layers/reference.usda@</Reference>
    prepend payload = @layers/payload.usda@</Payload>
)
{
    custom asset texture = @textures/a.png@
}
""",
        encoding="utf-8",
    )
    source_bytes = _tree_bytes(package)
    params = SimReadyConformanceInput(
        asset_path=str(package),
        output_dir=str(tmp_path / "conform"),
        repair_requirements=["AA.001"],
        foundation_root=str(tmp_path / "missing-foundation"),
        force=True,
    )

    first_report = run_simready_profile_conformance(params)

    assert first_report.passed
    assert first_report.requirements_repaired == ["AA.001"]
    output = Path(first_report.output_usd_path)
    assert output != asset
    assert output.parent.parent == tmp_path / "conform" / "conformed"
    assert _tree_bytes(package) == source_bytes
    assert _tree_bytes(output.parent).keys() == source_bytes.keys()
    assert (output.parent / "textures" / "a.png").read_bytes() == b"texture-bytes"
    assert (output.parent / "inventory.bin").read_bytes() == b"inventory-bytes"

    output_layer = Sdf.Layer.FindOrOpen(str(output))
    assert output_layer is not None
    assert output_layer.subLayerPaths == ["./layers/sub.usda"]
    root_spec = output_layer.GetPrimAtPath("/Asset")
    assert [str(item.assetPath) for item in root_spec.referenceList.prependedItems] == [
        "./layers/reference.usda"
    ]
    assert [str(item.assetPath) for item in root_spec.payloadList.prependedItems] == [
        "./layers/payload.usda"
    ]
    texture = output_layer.GetAttributeAtPath("/Asset.texture")
    assert texture.default == Sdf.AssetPath("./textures/a.png")
    first_receipt = json.loads(
        Path(first_report.reports["AA.001"]).read_text(encoding="utf-8")
    )
    assert {change["source_path"] for change in first_receipt["changes"]} == {
        "layers/payload.usda",
        "layers/reference.usda",
        "layers/sub.usda",
        "textures/a.png",
    }

    second_report = run_simready_profile_conformance(params)

    assert second_report.passed
    assert second_report.output_usd_path == first_report.output_usd_path
    assert len(list((tmp_path / "conform" / "conformed").iterdir())) == 1
    second_receipt = json.loads(
        Path(second_report.reports["AA.001"]).read_text(encoding="utf-8")
    )
    assert second_receipt["reused_output"]
    assert _tree_bytes(package) == source_bytes


def test_simready_conformance_rejects_root_symlink_before_repairs(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    owner = stage.DefinePrim("/Asset/Collider", "Xform")
    UsdPhysics.CollisionAPI.Apply(owner)
    UsdGeom.Cube.Define(stage, "/Asset/Collider/Shape")
    material = UsdShade.Material.Define(stage, "/Asset/PhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    assert stage.GetRootLayer().Save()
    stage = None
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    asset = tmp_path / "asset.usda"
    try:
        asset.symlink_to(source)
    except OSError as exc:  # pragma: no cover - platform permission dependent
        pytest.skip(f"symlink creation is unavailable: {exc}")

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["RB.COL.001", "AA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.status == "FAIL"
    assert report.steps == []
    assert report.next_step == "fix-asset-staging"
    assert any("root must not be a symlink" in error for error in report.errors)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha256
    assert asset.is_symlink()
    assert not (tmp_path / "conform" / "staged" / "asset.usda").exists()


def test_simready_conformance_aa001_repairs_inactive_variant_layer_closure(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr.Usd")
    package = tmp_path / "package"
    layers = package / "layers"
    layers.mkdir(parents=True)
    (layers / "leaf.usda").write_text(
        '#usda 1.0\n\ndef Xform "Leaf" {}\n',
        encoding="utf-8",
    )
    (layers / "inactive.usda").write_text(
        """#usda 1.0

def Xform "Inactive" (
    references = @leaf.usda@</Leaf>
)
{
}
""",
        encoding="utf-8",
    )
    asset = package / "asset.usda"
    asset.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
)

def Xform "Asset" (
    variants = {
        string model = "active"
    }
    prepend variantSets = "model"
)
{
    variantSet "model" = {
        "active" {
        }
        "inactive" {
            def Xform "Referenced" (
                references = @layers/inactive.usda@</Inactive>
            )
            {
            }
        }
    }
}
""",
        encoding="utf-8",
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(package),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["AA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    output = Path(report.output_usd_path)
    output_inactive = output.parent / "layers" / "inactive.usda"
    assert "@./layers/inactive.usda@" in output.read_text(encoding="utf-8")
    assert "@./leaf.usda@" in output_inactive.read_text(encoding="utf-8")
    receipt = json.loads(Path(report.reports["AA.001"]).read_text(encoding="utf-8"))
    assert receipt["layers"] == [
        "asset.usda",
        "layers/inactive.usda",
        "layers/leaf.usda",
    ]
    assert receipt["remaining_findings"] == []


def test_simready_conformance_aa001_repairs_sdf_asset_path_arrays_and_samples(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    package = tmp_path / "package"
    textures = package / "textures"
    textures.mkdir(parents=True)
    for name in ("a.png", "b.png", "c.png"):
        (textures / name).write_bytes(name.encode("ascii"))
    asset = package / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    paths = root.CreateAttribute("textures", Sdf.ValueTypeNames.AssetArray)
    assert paths.Set(
        Sdf.AssetPathArray(
            [Sdf.AssetPath("textures/a.png"), Sdf.AssetPath("textures/b.png")]
        )
    )
    assert paths.Set(
        Sdf.AssetPathArray(
            [Sdf.AssetPath("textures/b.png"), Sdf.AssetPath("textures/c.png")]
        ),
        Usd.TimeCode(1),
    )
    assert stage.GetRootLayer().Save()
    stage = None
    source_layer = Sdf.Layer.FindOrOpen(str(asset))
    source_spec = source_layer.GetAttributeAtPath("/Asset.textures")
    assert isinstance(source_spec.default, Sdf.AssetPathArray)
    assert isinstance(
        source_layer.QueryTimeSample(source_spec.path, 1), Sdf.AssetPathArray
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(package),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["AA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    output_stage = Usd.Stage.Open(report.output_usd_path)
    assert output_stage is not None
    output_paths = output_stage.GetPrimAtPath("/Asset").GetAttribute("textures")
    assert [item.path for item in output_paths.Get()] == [
        "./textures/a.png",
        "./textures/b.png",
    ]
    assert [item.path for item in output_paths.Get(Usd.TimeCode(1))] == [
        "./textures/b.png",
        "./textures/c.png",
    ]


def test_simready_conformance_aa001_removes_variant_asset_identity(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    variants = root.GetVariantSets().AddVariantSet("model")
    variants.AddVariant("active")
    variants.SetVariantSelection("active")
    with variants.GetVariantEditContext():
        root.SetAssetInfoByKey(
            "identifier",
            Sdf.AssetPath("./missing-active.usda"),
        )
        root.SetAssetInfoByKey("name", "active-model")
    assert stage.GetRootLayer().Save()
    stage = None
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["AA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    assert asset.read_bytes() == source_bytes
    output_layer = Sdf.Layer.FindOrOpen(report.output_usd_path)
    variant_spec = output_layer.GetObjectAtPath(Sdf.Path("/Asset{model=active}"))
    assert variant_spec.GetInfo("assetInfo") == {"name": "active-model"}
    receipt = json.loads(Path(report.reports["AA.001"]).read_text(encoding="utf-8"))
    assert receipt["changes"] == [
        {
            "action": "remove_asset_info_identifier",
            "layer": "asset.usda",
            "prim_path": "/Asset{model=active}",
            "source_path": "./missing-active.usda",
        }
    ]


def test_simready_conformance_aa001_removes_self_layer_asset_identity(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    root.SetAssetInfoByKey("identifier", Sdf.AssetPath("asset.usda"))
    root.SetAssetInfoByKey("name", "saved-as-asset")
    assert stage.GetRootLayer().Save()
    stage = None
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["AA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    assert asset.read_bytes() == source_bytes
    output_stage = Usd.Stage.Open(report.output_usd_path)
    assert output_stage is not None
    assert output_stage.GetDefaultPrim().GetAssetInfo() == {"name": "saved-as-asset"}
    receipt = json.loads(Path(report.reports["AA.001"]).read_text(encoding="utf-8"))
    assert receipt["changes"] == [
        {
            "action": "remove_asset_info_identifier",
            "layer": "asset.usda",
            "prim_path": "/Asset",
            "source_path": "asset.usda",
        }
    ]


def test_simready_conformance_temp_directories_are_mode_0700(tmp_path: Path) -> None:
    temporary = conform_profile_module._private_mkdtemp(
        prefix=".mode-check-",
        directory=tmp_path,
    )
    try:
        if os.name == "nt":
            assert temporary.is_dir()
            assert not temporary.is_symlink()
        else:
            assert stat.S_IMODE(temporary.stat().st_mode) == 0o700
    finally:
        shutil.rmtree(temporary)


def test_simready_cleanup_rejects_windows_reparse_point_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_dir = tmp_path / "build"
    child = build_dir / "child"
    child.mkdir(parents=True)
    original_stat = Path.stat
    reparse_attribute = 0x400
    monkeypatch.setattr(
        stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        reparse_attribute,
        raising=False,
    )

    def reparse_stat(path: Path, *args: object, **kwargs: object) -> object:
        metadata = original_stat(path, *args, **kwargs)
        if path == child:
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_file_attributes=reparse_attribute,
            )
        return metadata

    monkeypatch.setattr(Path, "stat", reparse_stat)

    with pytest.raises(ValueError, match="symlink or reparse point"):
        conform_profile_module._remove_gsp001_build_tree(build_dir)

    assert build_dir.exists()


def test_simready_conformance_aa001_removes_only_stale_asset_identity(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    root.SetAssetInfoByKey(
        "identifier",
        Sdf.AssetPath("./SubUSDs/sm_cabinet_file_c04_01.usd"),
    )
    root.SetAssetInfoByKey("name", "cabinet_file_c04")
    assert stage.GetRootLayer().Save()
    del stage
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["AA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    assert asset.read_bytes() == source_bytes
    output = Path(report.output_usd_path)
    output_stage = Usd.Stage.Open(str(output))
    assert output_stage is not None
    assert output_stage.GetDefaultPrim().GetAssetInfo() == {"name": "cabinet_file_c04"}
    assert not (output.parent / "SubUSDs").exists()
    assert {path.name for path in output.parent.iterdir()} == {"asset.usda"}
    receipt = json.loads(Path(report.reports["AA.001"]).read_text(encoding="utf-8"))
    assert receipt["changes"] == [
        {
            "action": "remove_asset_info_identifier",
            "layer": "asset.usda",
            "prim_path": "/Asset",
            "source_path": "./SubUSDs/sm_cabinet_file_c04_01.usd",
        }
    ]


def test_simready_conformance_aa001_removes_resolvable_asset_identity(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    identity = tmp_path / "identity.usda"
    identity.write_text('#usda 1.0\n\ndef Xform "Identity" {}\n', encoding="utf-8")
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    root.SetAssetInfoByKey("identifier", Sdf.AssetPath("identity.usda"))
    root.SetAssetInfoByKey("name", "asset-name")
    assert stage.GetRootLayer().Save()
    del stage

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["AA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    output_stage = Usd.Stage.Open(report.output_usd_path)
    assert output_stage is not None
    assert output_stage.GetDefaultPrim().GetAssetInfo() == {"name": "asset-name"}
    receipt = json.loads(Path(report.reports["AA.001"]).read_text(encoding="utf-8"))
    assert receipt["changes"] == [
        {
            "action": "remove_asset_info_identifier",
            "layer": "asset.usda",
            "prim_path": "/Asset",
            "source_path": "identity.usda",
        }
    ]
    assert not any(
        change["action"] == "anchor_asset_path"
        and change["source_path"] == "identity.usda"
        for change in receipt["changes"]
    )


def test_simready_conformance_aa001_noop_usdz_preserves_package(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr.Usd")
    asset = tmp_path / "asset.usdz"
    with ZipFile(asset, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "asset.usda",
            '#usda 1.0\n(defaultPrim = "Asset")\n\ndef Xform "Asset" {}\n',
        )
    source_bytes = asset.read_bytes()
    output_dir = tmp_path / "conform"

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(output_dir),
            repair_requirements=["AA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    output = Path(report.output_usd_path)
    assert output == output_dir / "staged" / "asset.usdz"
    assert output.read_bytes() == source_bytes
    assert asset.read_bytes() == source_bytes
    assert not (output_dir / "conformed").exists()
    receipt = json.loads(Path(report.reports["AA.001"]).read_text(encoding="utf-8"))
    assert receipt["source_was_usdz"]
    assert receipt["changes"] == []
    assert receipt["reused_output"]


def test_simready_conformance_aa001_integrity_drift_blocks_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pxr.Usd")
    package = tmp_path / "package"
    textures = package / "textures"
    textures.mkdir(parents=True)
    (textures / "a.png").write_bytes(b"texture")
    (package / "inventory.bin").write_bytes(b"original inventory")
    asset = package / "package.usda"
    asset.write_text(
        """#usda 1.0

def Xform "Asset"
{
    custom asset texture = @textures/a.png@
}
""",
        encoding="utf-8",
    )
    source_bytes = _tree_bytes(package)
    original_apply = conform_profile_module._apply_aa001_plan

    def apply_with_drift(**kwargs):
        changes = original_apply(**kwargs)
        (kwargs["build_tree"] / "inventory.bin").write_bytes(b"injected drift")
        return changes

    monkeypatch.setattr(
        conform_profile_module,
        "_apply_aa001_plan",
        apply_with_drift,
    )
    output_dir = tmp_path / "conform"

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(package),
            output_dir=str(output_dir),
            repair_requirements=["AA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert "changed bytes outside" in report.steps[0]["reason"]
    assert _tree_bytes(package) == source_bytes
    publish_root = output_dir / "conformed"
    assert publish_root.is_dir()
    assert list(publish_root.iterdir()) == []


def test_simready_conformance_aa001_blocks_unresolved_actual_dependency(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr.Usd")
    asset = tmp_path / "asset.usda"
    asset.write_text(
        """#usda 1.0

def Xform "Asset"
{
    custom asset texture = @./textures/missing.png@
}
""",
        encoding="utf-8",
    )
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["AA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["AA.001"]
    assert "actual dependency is unresolved" in report.steps[0]["reason"]
    assert asset.read_bytes() == source_bytes
    assert not (tmp_path / "conform" / "conformed").exists()


def test_simready_conformance_aa001_blocks_outside_asset_root(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr.Usd")
    package = tmp_path / "package"
    package.mkdir()
    outside = tmp_path / "outside.usda"
    outside.write_text('#usda 1.0\n\ndef Xform "Outside" {}\n', encoding="utf-8")
    asset = package / "asset.usda"
    asset.write_text(
        """#usda 1.0

def Xform "Asset" (
    references = @../outside.usda@</Outside>
)
{
}
""",
        encoding="utf-8",
    )
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["AA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["AA.001"]
    assert "outside the copied asset root" in report.steps[0]["reason"]
    assert asset.read_bytes() == source_bytes
    assert outside.is_file()
    assert not (tmp_path / "conform" / "conformed").exists()


def test_simready_conformance_aa001_blocks_absolute_url_and_search_paths(
    tmp_path: Path,
) -> None:
    Ar = pytest.importorskip("pxr.Ar")
    pytest.importorskip("pxr.Usd")
    search_root = tmp_path / "search"
    search_root.mkdir()
    (search_root / "found.png").write_bytes(b"search-path")
    authored_paths = [
        str(search_root / "found.png"),
        "https://example.invalid/found.png",
        "C:/outside/found.png",
        r"C:\outside\found.png",
        "found.png",
    ]
    resolver = Ar.GetUnderlyingResolver()
    resolver.SetDefaultSearchPath([str(search_root)])
    try:
        for index, authored_path in enumerate(authored_paths):
            case = tmp_path / f"case-{index}"
            case.mkdir()
            asset = case / "asset.usda"
            asset.write_text(
                f"""#usda 1.0

def Xform "Asset"
{{
    custom asset texture = @{authored_path}@
}}
""",
                encoding="utf-8",
            )
            report = run_simready_profile_conformance(
                SimReadyConformanceInput(
                    asset_path=str(asset),
                    output_dir=str(case / "conform"),
                    repair_requirements=["AA.001"],
                    foundation_root=str(tmp_path / "missing-foundation"),
                    force=True,
                )
            )
            assert not report.passed
            assert report.requirements_blocked == ["AA.001"]
            assert not (case / "conform" / "conformed").exists()
    finally:
        resolver.SetDefaultSearchPath([])


def test_simready_conformance_aa001_blocks_symlink_dependency(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr.Usd")
    package = tmp_path / "package"
    texture_dir = package / "textures"
    texture_dir.mkdir(parents=True)
    external = tmp_path / "external.png"
    external.write_bytes(b"external")
    symlink = texture_dir / "a.png"
    try:
        symlink.symlink_to(external)
    except OSError as exc:  # pragma: no cover - platform permission dependent
        pytest.skip(f"symlink creation is unavailable: {exc}")
    asset = package / "package.usda"
    asset.write_text(
        """#usda 1.0

def Xform "Asset"
{
    custom asset texture = @./textures/a.png@
}
""",
        encoding="utf-8",
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(package),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["AA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.status == "FAIL"
    assert report.requirements_blocked == []
    assert report.steps == []
    assert report.next_step == "fix-asset-staging"
    assert any("contains a symlink" in error for error in report.errors)
    assert symlink.is_symlink()
    assert external.read_bytes() == b"external"
    assert not (tmp_path / "conform" / "conformed").exists()


def test_simready_conformance_aa001_blocks_identity_dependency_overlap(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    (texture_dir / "a.png").write_bytes(b"texture")
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    path = Sdf.AssetPath("./textures/a.png")
    root.CreateAttribute("texture", Sdf.ValueTypeNames.Asset).Set(path)
    root.SetAssetInfoByKey("identifier", path)
    assert stage.GetRootLayer().Save()
    del stage

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["AA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["AA.001"]
    assert "overlaps an actual dependency" in report.steps[0]["reason"]
    assert not (tmp_path / "conform" / "conformed").exists()


def test_simready_conformance_composes_gsp001_aa001_then_isa001(
    tmp_path: Path,
) -> None:
    Kind = pytest.importorskip("pxr.Kind")
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    (texture_dir / "a.png").write_bytes(b"texture")
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    root.CreateAttribute("texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/a.png")
    )
    root.SetAssetInfoByKey("identifier", Sdf.AssetPath("./missing.usd"))
    assert stage.GetRootLayer().Save()
    del stage
    grasp_plan = tmp_path / "grasp-plan.json"
    grasp_plan.write_text(
        json.dumps(
            {
                "schema_version": SIMREADY_GRASP_PLAN_SCHEMA_VERSION,
                "source_asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                "default_prim_path": "/Asset",
                "provenance": {
                    "source": "owner_approved_plan",
                    "approved_by": "simready-owner@example.com",
                    "evidence": ["review://fixture/asset-grasp-v1"],
                },
                "grasp_lines": [
                    {
                        "prim_path": "/Asset/grasp_identifier_01",
                        "coordinate_space": "local",
                        "points": [[0, 0, 0], [0, 0, 1]],
                        "widths": [0.01],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_bytes = _tree_bytes(tmp_path)
    params = SimReadyConformanceInput(
        asset_path=str(asset),
        output_dir=str(tmp_path / "conform"),
        repair_requirements=["AA.001", "GSP.001", "ISA.001"],
        grasp_plan_path=str(grasp_plan),
        source_asset=str(asset),
        foundation_root=str(tmp_path / "missing-foundation"),
        force=True,
    )

    first_report = run_simready_profile_conformance(params)

    assert first_report.passed
    assert first_report.requirements_repaired == ["AA.001", "GSP.001", "ISA.001"]
    assert [step["requirement"] for step in first_report.steps] == [
        "GSP.001",
        "AA.001",
        "ISA.001",
    ]
    staged_source = tmp_path / "conform" / "staged" / "asset.usda"
    assert first_report.steps[0]["input_usd_path"] == str(staged_source)
    assert first_report.steps[0]["output_usd_path"] != str(staged_source)
    assert Path(first_report.steps[0]["output_usd_path"]).is_relative_to(
        tmp_path / "conform" / "grasp-conformed"
    )
    assert (
        first_report.steps[1]["input_usd_path"]
        == first_report.steps[0]["output_usd_path"]
    )
    assert (
        first_report.steps[2]["input_usd_path"]
        == first_report.steps[1]["output_usd_path"]
    )
    output = Path(first_report.output_usd_path)
    output_stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    assert output_stage is not None
    output_root = output_stage.GetDefaultPrim()
    assert Usd.ModelAPI(output_root).GetKind() == Kind.Tokens.component
    assert output_root.GetAttribute("texture").Get().path == "./textures/a.png"
    assert "identifier" not in output_root.GetAssetInfo()
    assert output_stage.GetPrimAtPath("/Asset/grasp_identifier_01").IsA(
        UsdGeom.BasisCurves
    )
    root_spec = output_stage.GetRootLayer().GetPrimAtPath("/Asset")
    assert [str(item.assetPath) for item in root_spec.referenceList.prependedItems] == [
        "./payloads/asset_base.usd"
    ]
    assert [str(item.assetPath) for item in root_spec.payloadList.prependedItems] == [
        "./payloads/asset_physics.usd"
    ]
    assert _tree_bytes(tmp_path / "textures") == {
        "a.png": source_bytes["textures/a.png"]
    }
    assert asset.read_bytes() == source_bytes["asset.usda"]

    second_report = run_simready_profile_conformance(params)

    assert second_report.passed
    assert second_report.output_usd_path == first_report.output_usd_path
    aa_receipt = json.loads(
        Path(second_report.reports["AA.001"]).read_text(encoding="utf-8")
    )
    isa_receipt = json.loads(
        Path(second_report.reports["ISA.001"]).read_text(encoding="utf-8")
    )
    assert aa_receipt["reused_output"]
    assert isa_receipt["reused_output"]
    assert len(list((tmp_path / "conform" / "conformed").iterdir())) == 2


def test_simready_conformance_keeps_existing_isa001_package_byte_exact(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr.Usd")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    asset = source_dir / "asset.usda"
    _write_isa001_compliant_asset(asset)
    source_bytes = _tree_bytes(source_dir)

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    assert report.status == "PASS"
    assert report.requirements_repaired == ["ISA.001"]
    output = Path(report.output_usd_path)
    assert output == tmp_path / "conform" / "staged" / "asset.usda"
    assert _tree_bytes(source_dir) == source_bytes
    assert _tree_bytes(output.parent) == source_bytes
    repair_report = json.loads(
        Path(report.reports["ISA.001"]).read_text(encoding="utf-8")
    )
    assert repair_report["changes"] == []
    assert repair_report["reused_output"]
    assert not (tmp_path / "conform" / "conformed").exists()


def test_simready_conformance_repairs_raw_usd_to_isa001_atomically(
    tmp_path: Path,
) -> None:
    Gf = pytest.importorskip("pxr.Gf")
    Kind = pytest.importorskip("pxr.Kind")
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    stage.SetMetadata("upAxis", UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("customLayerData", {"fixture": "isa001"})
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    root.SetAssetInfoByKey(
        "identifier",
        Sdf.AssetPath("./provenance/original_asset.usd"),
    )
    UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(2.0, 3.0, 4.0))
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_body.CreateRigidBodyEnabledAttr().Set(False)

    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Visuals/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdGeom.Scope.Define(stage, "/SharedLooks")
    material = UsdShade.Material.Define(stage, "/SharedLooks/PaintedMetal")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    body = stage.DefinePrim("/Asset/Body", "Xform")
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/Asset/Joints/Slide")
    joint.CreateBody0Rel().SetTargets([root.GetPath()])
    joint.CreateBody1Rel().SetTargets([body.GetPath()])
    joint.CreateAxisAttr().Set(UsdPhysics.Tokens.x)
    assert stage.GetRootLayer().Save()
    source_bytes = asset.read_bytes()
    source_world = UsdGeom.Xformable(mesh.GetPrim()).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    params = SimReadyConformanceInput(
        asset_path=str(asset),
        output_dir=str(tmp_path / "conform"),
        repair_requirements=["ISA.001"],
        foundation_root=str(tmp_path / "missing-foundation"),
        force=True,
    )
    first_report = run_simready_profile_conformance(params)

    assert first_report.passed
    assert first_report.failed_requirements == []
    assert first_report.requirements_repaired == ["ISA.001"]
    assert asset.read_bytes() == source_bytes
    output = Path(first_report.output_usd_path)
    payload_dir = output.parent / "payloads"
    expected_payloads = {
        "asset_base.usd",
        "asset_meshes.usd",
        "asset_physics.usd",
    }
    assert {path.name for path in payload_dir.iterdir()} == expected_payloads

    repaired_stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    assert repaired_stage is not None
    repaired_root = repaired_stage.GetDefaultPrim()
    assert repaired_root.GetPath() == Sdf.Path("/Asset")
    assert repaired_root.GetAssetInfo()["identifier"] == Sdf.AssetPath(
        "./provenance/original_asset.usd"
    )
    assert Usd.ModelAPI(repaired_root).GetKind() == Kind.Tokens.component
    root_spec = repaired_stage.GetRootLayer().GetPrimAtPath("/Asset")
    assert [str(item.assetPath) for item in root_spec.referenceList.prependedItems] == [
        "./payloads/asset_base.usd"
    ]
    assert [str(item.assetPath) for item in root_spec.payloadList.prependedItems] == [
        "./payloads/asset_physics.usd"
    ]
    assert repaired_stage.GetRootLayer().GetPrimAtPath("/Asset/Visuals") is not None
    sibling_spec = repaired_stage.GetRootLayer().GetPrimAtPath("/SharedLooks")
    assert sibling_spec is not None
    assert not sibling_spec.referenceList.GetAppliedItems()

    base_layer = Sdf.Layer.FindOrOpen(str(payload_dir / "asset_base.usd"))
    base_spec = base_layer.GetPrimAtPath("/Asset")
    assert [str(item.assetPath) for item in base_spec.referenceList.prependedItems] == [
        "./asset_meshes.usd"
    ]
    for payload_name in expected_payloads:
        payload_stage = Usd.Stage.Open(str(payload_dir / payload_name))
        assert payload_stage is not None
        assert payload_stage.GetDefaultPrim().GetPath() == Sdf.Path("/Asset")

    repaired_mesh = repaired_stage.GetPrimAtPath("/Asset/Visuals/Mesh")
    assert (
        UsdGeom.Xformable(repaired_mesh).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        == source_world
    )
    assert repaired_root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert (
        UsdPhysics.RigidBodyAPI(repaired_root).GetRigidBodyEnabledAttr().Get() is False
    )
    assert repaired_mesh.HasAPI(UsdPhysics.CollisionAPI)
    bound_material, _relationship = UsdShade.MaterialBindingAPI(
        repaired_mesh
    ).ComputeBoundMaterial()
    assert bound_material.GetPath() == material.GetPath()
    repaired_joint = UsdPhysics.PrismaticJoint(
        repaired_stage.GetPrimAtPath("/Asset/Joints/Slide")
    )
    assert repaired_joint.GetBody0Rel().GetTargets() == [root.GetPath()]
    assert repaired_joint.GetBody1Rel().GetTargets() == [body.GetPath()]
    assert repaired_joint.GetAxisAttr().Get() == UsdPhysics.Tokens.x
    assert repaired_stage.GetMetadata("customLayerData") == {"fixture": "isa001"}

    first_tree = _tree_bytes(output.parents[1])
    repair_receipt = json.loads(
        Path(first_report.reports["ISA.001"]).read_text(encoding="utf-8")
    )
    assert repair_receipt["ignored_unresolved_asset_identity_paths"] == [
        "./provenance/original_asset.usd"
    ]
    second_report = run_simready_profile_conformance(params)
    assert second_report.passed
    assert second_report.output_usd_path == first_report.output_usd_path
    assert _tree_bytes(output.parents[1]) == first_tree
    assert len(list((tmp_path / "conform" / "conformed").iterdir())) == 1
    second_repair_report = json.loads(
        Path(second_report.reports["ISA.001"]).read_text(encoding="utf-8")
    )
    assert second_repair_report["reused_output"]
    assert asset.read_bytes() == source_bytes


@pytest.mark.parametrize("suffix", [".usd", ".usdc"])
def test_simready_conformance_isa001_repairs_binary_root_layers(
    tmp_path: Path,
    suffix: str,
) -> None:
    Kind = pytest.importorskip("pxr.Kind")
    Usd = pytest.importorskip("pxr.Usd")
    asset = tmp_path / f"asset{suffix}"
    stage = Usd.Stage.CreateNew(str(asset))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    stage.DefinePrim("/Asset/Child", "Xform")
    assert stage.GetRootLayer().Save()
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    assert asset.read_bytes() == source_bytes
    output = Path(report.output_usd_path)
    assert output.suffix == suffix
    repaired_stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    assert repaired_stage is not None
    assert repaired_stage.GetPrimAtPath("/Asset/Child")
    assert Usd.ModelAPI(repaired_stage.GetDefaultPrim()).GetKind() == (
        Kind.Tokens.component
    )
    for payload_name in (
        "asset_base.usd",
        "asset_meshes.usd",
        "asset_physics.usd",
    ):
        payload_stage = Usd.Stage.Open(str(output.parent / "payloads" / payload_name))
        assert payload_stage is not None


def test_simready_conformance_isa001_preserves_dependencies_in_main_layer(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    package = tmp_path / "package"
    scene_dir = package / "Scenes"
    payload_dir = package / "Payload"
    texture_dir = package / "textures"
    scene_dir.mkdir(parents=True)
    payload_dir.mkdir()
    texture_dir.mkdir()
    existing_payload_dir = scene_dir / "payloads"
    existing_payload_dir.mkdir()
    (existing_payload_dir / "keep.bin").write_bytes(b"existing-payload-data")
    (texture_dir / "albedo.png").write_bytes(b"dependency-texture")
    (payload_dir / "contents.usda").write_text(
        '#usda 1.0\n\ndef Xform "Asset"\n{\n    def Xform "Child" {}\n}\n',
        encoding="utf-8",
    )
    asset = scene_dir / "asset.usda"
    asset.write_text(
        f'''#usda 1.0
(
    defaultPrim = "Asset"
    subLayers = [@../Payload/contents.usda@]
)

over "Asset"
{{
    custom asset previewTexture = @../textures/albedo.png@
    custom asset supportData = @payloads/keep.bin@
    custom string dependencyMarker = "{package.name}"
}}
''',
        encoding="utf-8",
    )
    source_bytes = _tree_bytes(package)

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    assert _tree_bytes(package) == source_bytes
    output = Path(report.output_usd_path)
    output_package = output.parent.parent
    assert (output_package / "Payload" / "contents.usda").read_bytes() == (
        source_bytes["Payload/contents.usda"]
    )
    assert (output_package / "textures" / "albedo.png").read_bytes() == (
        b"dependency-texture"
    )
    assert (output.parent / "payloads" / "keep.bin").read_bytes() == (
        b"existing-payload-data"
    )
    main_text = output.read_text(encoding="utf-8")
    assert "@../Payload/contents.usda@" in main_text
    assert "@../textures/albedo.png@" in main_text
    assert "@payloads/keep.bin@" in main_text
    meshes_stage = Usd.Stage.Open(str(output.parent / "payloads" / "asset_meshes.usd"))
    assert meshes_stage is not None
    assert str(meshes_stage.GetDefaultPrim().GetPath()) == "/Asset"
    repaired_stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    assert repaired_stage is not None
    assert repaired_stage.GetPrimAtPath("/Asset/Child")
    texture = (
        repaired_stage.GetPrimAtPath("/Asset").GetAttribute("previewTexture").Get()
    )
    assert Path(texture.resolvedPath) == output_package / "textures" / "albedo.png"


def test_simready_conformance_isa001_accepts_prior_derivative_package(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pytest.importorskip("pxr.Usd")
    package = tmp_path / "package"
    scene_dir = package / "Scenes"
    layer_dir = package / "Layers"
    scene_dir.mkdir(parents=True)
    layer_dir.mkdir()
    (layer_dir / "contents.usda").write_text(
        '#usda 1.0\n\ndef Xform "Asset" {\n    def Xform "Child" {}\n}\n',
        encoding="utf-8",
    )
    asset = scene_dir / "asset.usda"
    asset.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
    subLayers = [@../Layers/contents.usda@]
)

over "Asset" {}
""",
        encoding="utf-8",
    )
    original_repair = conform_profile_module._repair_requirement

    def repair_with_derivative(
        *,
        requirement: str,
        asset_path: Path,
        package_root: Path,
        output_dir: Path,
        grasp_plan_path: Path | None = None,
        grasp_source_asset_path: Path | None = None,
        grasp_source_lineage: conform_profile_module._GSP001SourceLineage | None = None,
        expected_physics_inventory_sha256: str | None = None,
        source_asset: str | None = None,
        grasp_prim_path: str | None = None,
    ):
        if requirement != "GSP.001":
            return original_repair(
                requirement=requirement,
                asset_path=asset_path,
                package_root=package_root,
                output_dir=output_dir,
                grasp_plan_path=grasp_plan_path,
                grasp_source_asset_path=grasp_source_asset_path,
                grasp_source_lineage=grasp_source_lineage,
                expected_physics_inventory_sha256=(expected_physics_inventory_sha256),
                source_asset=source_asset,
                grasp_prim_path=grasp_prim_path,
            )
        derivative = output_dir / "prior-derivative" / "fixture"
        shutil.copytree(package_root, derivative)
        derivative_asset = derivative / asset_path.relative_to(package_root)
        return conform_profile_module._RepairResult(
            status="REPAIRED",
            passed=True,
            reason="Published a prior deterministic package derivative.",
            output_path=derivative_asset,
            package_root=derivative,
        )

    monkeypatch.setattr(
        conform_profile_module,
        "_repair_requirement",
        repair_with_derivative,
    )
    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["GSP.001", "ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    assert report.requirements_repaired == ["GSP.001", "ISA.001"]
    output = Path(report.output_usd_path)
    assert output.is_file()
    assert output.parent.parent.parent == tmp_path / "conform" / "conformed"
    assert (output.parent.parent / "Layers" / "contents.usda").is_file()


def test_simready_conformance_isa001_resolves_deep_sublayer_assets(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    package = tmp_path / "package"
    scene_dir = package / "Scenes"
    layer_dir = package / "Layers" / "deep"
    texture_dir = package / "Layers" / "textures"
    scene_dir.mkdir(parents=True)
    layer_dir.mkdir(parents=True)
    texture_dir.mkdir(parents=True)
    (texture_dir / "albedo.png").write_bytes(b"deep-layer-texture")
    (layer_dir / "contents.usda").write_text(
        """#usda 1.0

def Xform "Asset"
{
    custom asset previewTexture = @../textures/albedo.png@
}
""",
        encoding="utf-8",
    )
    asset = scene_dir / "asset.usda"
    asset.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
    subLayers = [@../Layers/deep/contents.usda@]
)

over "Asset" {}
""",
        encoding="utf-8",
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    output = Path(report.output_usd_path)
    repaired_stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    assert repaired_stage is not None
    texture = (
        repaired_stage.GetPrimAtPath("/Asset").GetAttribute("previewTexture").Get()
    )
    assert Path(texture.resolvedPath) == (
        output.parent.parent / "Layers" / "textures" / "albedo.png"
    )


def test_simready_conformance_isa001_extracts_usdz_with_dependencies(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    asset = tmp_path / "asset.usdz"
    with ZipFile(asset, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "root.usda",
            """#usda 1.0
(
    defaultPrim = "Asset"
    subLayers = [@layers/contents.usda@]
)

over "Asset"
{
    custom asset previewTexture = @textures/albedo.png@
}
""",
        )
        archive.writestr(
            "layers/contents.usda",
            '#usda 1.0\n\ndef Xform "Asset"\n{\n    def Xform "Child" {}\n}\n',
        )
        archive.writestr("textures/albedo.png", b"usdz-texture")
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    assert asset.read_bytes() == source_bytes
    output = Path(report.output_usd_path)
    assert output.name == "root.usda"
    assert output.suffix == ".usda"
    assert (output.parent / "layers" / "contents.usda").is_file()
    assert (output.parent / "textures" / "albedo.png").read_bytes() == (b"usdz-texture")
    assert {path.name for path in (output.parent / "payloads").iterdir()} == {
        "root_base.usd",
        "root_meshes.usd",
        "root_physics.usd",
    }
    repaired_stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    assert repaired_stage is not None
    assert repaired_stage.GetPrimAtPath("/Asset/Child")
    repair_report = json.loads(
        Path(report.reports["ISA.001"]).read_text(encoding="utf-8")
    )
    assert repair_report["source_was_usdz"]


def test_simready_conformance_isa001_rejects_non_usd_usdz_entrypoint(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr.Usd")
    asset = tmp_path / "invalid-entrypoint.usdz"
    with ZipFile(asset, "w", ZIP_DEFLATED) as archive:
        archive.writestr("textures/albedo.png", b"not-a-package-root")
        archive.writestr(
            "root.usda",
            '#usda 1.0\n\ndef Xform "Asset" {}\n',
        )
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["ISA.001"]
    assert "package root is not a supported USD layer" in report.steps[0]["reason"]
    assert asset.read_bytes() == source_bytes
    assert not (tmp_path / "conform" / "conformed").exists()


def test_simready_conformance_isa001_extracts_existing_compliant_usdz(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr.Usd")
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    package_root = package_dir / "asset.usda"
    _write_isa001_compliant_asset(package_root)
    package_bytes = _tree_bytes(package_dir)
    asset = tmp_path / "asset.usdz"
    with ZipFile(asset, "w", ZIP_DEFLATED) as archive:
        archive.write(package_root, "asset.usda")
        for path in sorted((package_dir / "payloads").iterdir()):
            archive.write(path, f"payloads/{path.name}")
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    assert asset.read_bytes() == source_bytes
    output = Path(report.output_usd_path)
    assert _tree_bytes(output.parent) == package_bytes
    repair_report = json.loads(
        Path(report.reports["ISA.001"]).read_text(encoding="utf-8")
    )
    assert repair_report["changes"] == []
    assert repair_report["source_was_usdz"]
    assert not repair_report["reused_output"]
    assert report.steps[0]["reason"] == (
        "Published an extracted ISA.001-compliant USDZ package."
    )


def test_simready_conformance_isa001_blocks_ambiguous_top_level_roots(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr.Usd")
    asset = tmp_path / "asset.usda"
    asset.write_text(
        '#usda 1.0\n\ndef Xform "First" {}\n\ndef Xform "Second" {}\n',
        encoding="utf-8",
    )
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.status == "BLOCKED"
    assert report.requirements_blocked == ["ISA.001"]
    assert "valid default prim" in report.steps[0]["reason"]
    assert asset.read_bytes() == source_bytes
    assert not (tmp_path / "conform" / "conformed").exists()


def test_simready_conformance_isa001_blocks_existing_target_collision(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr.Usd")
    package = tmp_path / "package"
    payload_dir = package / "payloads"
    payload_dir.mkdir(parents=True)
    asset = package / "asset.usda"
    asset.write_text(
        '#usda 1.0\n\ndef Xform "Asset" {}\n',
        encoding="utf-8",
    )
    (payload_dir / "asset_base.usd").write_text("#usda 1.0\n", encoding="utf-8")
    source_bytes = _tree_bytes(package)

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(package),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["ISA.001"]
    assert "target layer already exists" in report.steps[0]["reason"]
    assert _tree_bytes(package) == source_bytes
    assert not (tmp_path / "conform" / "conformed").exists()


def test_simready_conformance_isa001_blocks_external_absolute_dependency(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr.Usd")
    external = tmp_path / "external.usda"
    external.write_text(
        '#usda 1.0\n\ndef Xform "Asset" {}\n',
        encoding="utf-8",
    )
    asset = tmp_path / "asset.usda"
    asset.write_text(
        f"""#usda 1.0
(
    defaultPrim = "Asset"
    subLayers = [@{external}@]
)
""",
        encoding="utf-8",
    )
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["ISA.001"]
    assert "outside the staged package" in report.steps[0]["reason"]
    assert asset.read_bytes() == source_bytes
    assert external.is_file()
    assert not (tmp_path / "conform" / "conformed").exists()


def test_simready_conformance_isa001_blocks_unresolved_dependency(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr.Usd")
    asset = tmp_path / "asset.usda"
    asset.write_text(
        """#usda 1.0

def Xform "Asset"
{
    custom asset missingTexture = @missing.png@
}
""",
        encoding="utf-8",
    )
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["ISA.001"]
    assert "dependency closure is unresolved" in report.steps[0]["reason"]
    assert asset.read_bytes() == source_bytes
    assert not (tmp_path / "conform" / "conformed").exists()


def test_simready_conformance_isa001_only_ignores_asset_identifier_metadata(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    root.SetAssetInfoByKey("identifier", Sdf.AssetPath("./identity.usd"))
    root.SetAssetInfoByKey("thumbnail", Sdf.AssetPath("./thumbnail.png"))
    assert stage.GetRootLayer().Save()
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["ISA.001"]
    assert "thumbnail.png" in report.steps[0]["reason"]
    assert "identity.usd" not in report.steps[0]["reason"]
    assert asset.read_bytes() == source_bytes


def test_simready_conformance_isa001_rejects_unsafe_usdz_member(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr.Usd")
    asset = tmp_path / "unsafe.usdz"
    with ZipFile(asset, "w") as archive:
        archive.writestr(
            "root.usda",
            '#usda 1.0\n\ndef Xform "Asset" {}\n',
        )
        archive.writestr("../escape.bin", b"unsafe")
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["ISA.001"]
    assert "unsafe entry path" in report.steps[0]["reason"]
    assert asset.read_bytes() == source_bytes
    assert not (tmp_path / "conform" / "conformed").exists()


def test_simready_conformance_isa001_cleans_partial_build_on_publish_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pytest.importorskip("pxr.Usd")
    asset = tmp_path / "asset.usda"
    asset.write_text(
        '#usda 1.0\n\ndef Xform "Asset" {}\n',
        encoding="utf-8",
    )
    source_bytes = asset.read_bytes()

    def fail_publish(**_kwargs):
        raise OSError("injected publish failure")

    monkeypatch.setattr(
        conform_profile_module,
        "_publish_isa001_tree",
        fail_publish,
    )
    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["ISA.001"]
    assert "injected publish failure" in report.steps[0]["reason"]
    assert asset.read_bytes() == source_bytes
    publish_root = tmp_path / "conform" / "conformed"
    assert publish_root.is_dir()
    assert list(publish_root.iterdir()) == []


def test_simready_conformance_isa001_cleans_extraction_on_dependency_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    UsdUtils = pytest.importorskip("pxr.UsdUtils")
    asset = tmp_path / "asset.usdz"
    with ZipFile(asset, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "root.usda",
            '#usda 1.0\n\ndef Xform "Asset" {}\n',
        )

    def fail_dependency_inspection(_path: str):
        raise Exception("injected OpenUSD dependency failure")

    monkeypatch.setattr(
        UsdUtils,
        "ComputeAllDependencies",
        fail_dependency_inspection,
    )
    output_dir = tmp_path / "conform"
    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(output_dir),
            repair_requirements=["ISA.001"],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["ISA.001"]
    assert "injected OpenUSD dependency failure" in report.steps[0]["reason"]
    assert not list(output_dir.glob(".isa001-source-*"))
    assert not (output_dir / "conformed").exists()


def test_isa001_publish_rejects_build_tree_outside_publish_root(
    tmp_path: Path,
) -> None:
    publish_root = tmp_path / "published"
    publish_root.mkdir()
    build_dir = tmp_path / "external-build"
    build_dir.mkdir()
    (build_dir / "asset.usda").write_text(
        '#usda 1.0\n\ndef Xform "Asset" {}\n',
        encoding="utf-8",
    )
    tree_sha256 = conform_profile_module._isa001_tree_sha256(build_dir)

    with pytest.raises(ValueError, match="direct child of the publish root"):
        conform_profile_module._publish_isa001_tree(
            build_dir=build_dir,
            publish_root=publish_root,
            tree_sha256=tree_sha256,
        )

    assert build_dir.is_dir()
    assert list(publish_root.iterdir()) == []


def test_isa001_publish_reuses_verified_concurrent_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    publish_root = tmp_path / "published"
    publish_root.mkdir()
    build_dir = publish_root / ".isa001-build-fixture"
    build_dir.mkdir()
    (build_dir / "asset.usda").write_text(
        '#usda 1.0\n\ndef Xform "Asset" {}\n',
        encoding="utf-8",
    )
    tree_sha256 = conform_profile_module._isa001_tree_sha256(build_dir)

    def lose_publish_race(source: Path, destination: Path):
        shutil.copytree(source, destination)
        raise FileExistsError("injected concurrent publisher")

    monkeypatch.setattr(Path, "replace", lose_publish_race)
    final_tree, reused = conform_profile_module._publish_isa001_tree(
        build_dir=build_dir,
        publish_root=publish_root,
        tree_sha256=tree_sha256,
    )

    assert reused
    assert (
        final_tree
        == publish_root / conform_profile_module._content_address_component(tree_sha256)
    )
    assert conform_profile_module._isa001_tree_sha256(final_tree) == tree_sha256
    assert not build_dir.exists()


def test_simready_conformance_repairs_deterministic_profile_failures(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)

    visual_mesh = UsdGeom.Mesh.Define(stage, "/robot/link/visual_mesh")
    visual_mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    visual_mesh.CreateFaceVertexCountsAttr([3])
    visual_mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    collider_owner = stage.DefinePrim("/robot/link/collider", "Xform")
    collision_api = UsdPhysics.CollisionAPI.Apply(collider_owner)
    collision_api.CreateCollisionEnabledAttr().Set(False)
    UsdPhysics.MeshCollisionAPI.Apply(collider_owner)
    collider_mesh = UsdGeom.Mesh.Define(stage, "/robot/link/collider/mesh")
    collider_mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    collider_mesh.CreateFaceVertexCountsAttr([3])
    collider_mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    visual_material = UsdShade.Material.Define(stage, "/robot/Looks/VisualMaterial")
    shader = UsdShade.Shader.Define(stage, "/robot/Looks/VisualMaterial/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    visual_material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    physics_material = UsdShade.Material.Define(stage, "/robot/Looks/PhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    collider_owner.CreateRelationship("material:binding:physics").SetTargets(
        [physics_material.GetPath()]
    )

    grasp = UsdGeom.BasisCurves.Define(stage, "/robot/grasp_identifier_01")
    grasp.CreateTypeAttr(UsdGeom.Tokens.linear)
    grasp.CreateCurveVertexCountsAttr([2])
    grasp.CreatePointsAttr([(0, 0, 0), (0, 0, 1)])
    grasp.CreateWidthsAttr([0.01])
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=[
                "GSP.001",
                "RB.COL.001",
                "RB.COL.002",
                "VM.MAT.001",
            ],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    assert report.status == "PASS"
    assert report.requirements_blocked == []
    assert report.requirements_repaired == [
        "GSP.001",
        "PMT.001",
        "RB.COL.001",
        "RB.COL.002",
        "VM.MAT.001",
    ]

    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    repaired_owner = repaired_stage.GetPrimAtPath("/robot/link/collider")
    repaired_mesh = repaired_stage.GetPrimAtPath("/robot/link/collider/mesh")
    assert not repaired_owner.HasAPI(UsdPhysics.CollisionAPI)
    assert not repaired_owner.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert repaired_mesh.HasAPI(UsdPhysics.CollisionAPI)
    assert repaired_mesh.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert (
        UsdPhysics.CollisionAPI(repaired_mesh).GetCollisionEnabledAttr().Get() is False
    )
    physics_binding = repaired_mesh.GetRelationship("material:binding:physics")
    physics_targets = physics_binding.GetTargets()
    assert physics_targets == [physics_material.GetPath()]

    for path in ["/robot/link/visual_mesh", "/robot/link/collider/mesh"]:
        prim = repaired_stage.GetPrimAtPath(path)
        material, _rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        assert material
        assert material.GetPath() == visual_material.GetPath()

    grasp = repaired_stage.GetPrimAtPath("/robot/grasp_identifier_01")
    assert grasp.IsA(UsdGeom.BasisCurves)
    assert len(grasp.GetAttribute("points").Get()) == 2


def test_simready_conformance_blocks_grasp_without_evidence_and_no_colliders(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    visual_mesh = UsdGeom.Mesh.Define(stage, "/robot/visual_mesh")
    visual_mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    visual_mesh.CreateFaceVertexCountsAttr([3])
    visual_mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["GSP.001"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.status == "BLOCKED"
    assert report.requirements_repaired == []
    assert report.requirements_blocked == ["GSP.001"]
    assert "PMT.001" not in {step["requirement"] for step in report.steps}


def test_simready_conformance_authors_grasp_from_explicit_target_bounds(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = UsdGeom.Xform.Define(stage, "/asset")
    stage.SetDefaultPrim(root.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/asset/grasp_target")
    cube.CreateSizeAttr(2.0)
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["GSP.001"],
            grasp_prim_path="/asset/grasp_target",
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    assert report.requirements_repaired == ["GSP.001"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    grasp = UsdGeom.BasisCurves(
        repaired_stage.GetPrimAtPath("/asset/grasp_identifier_01")
    )
    assert grasp
    assert len(grasp.GetPointsAttr().Get()) == 2


def test_simready_conformance_blocks_explicit_grasp_on_read_only_usdz(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdUtils = pytest.importorskip("pxr.UsdUtils")
    foundation_root = _write_fake_foundation(tmp_path)
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Cube.Define(stage, "/Asset/grasp_target")
    assert stage.GetRootLayer().Save()
    asset = tmp_path / "asset.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(source), str(asset))
    source_bytes = asset.read_bytes()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["GSP.001"],
            grasp_prim_path="/Asset/grasp_target",
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["GSP.001"]
    assert "read-only USDZ package layer" in report.steps[0]["reason"]
    output = Path(report.output_usd_path)
    assert output.read_bytes() == source_bytes
    assert asset.read_bytes() == source_bytes
    assert Path(report.reports["GSP.001"]).is_file()


def test_simready_conformance_blocks_visual_material_without_sourced_material(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/visual_mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["VM.MAT.001"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["VM.MAT.001"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    assert not repaired_stage.GetPrimAtPath(
        "/robot/Looks/SimReadyFallbackMaterial"
    ).IsValid()


def test_simready_conformance_blocks_ambiguous_vm_tex_002_color_spaces(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = stage.DefinePrim("/asset", "Xform")
    stage.SetDefaultPrim(root)
    shader = UsdShade.Shader.Define(stage, "/asset/Looks/Shader")
    base_color = shader.CreateInput(
        "base_color_texture_file", Sdf.ValueTypeNames.Asset
    ).GetAttr()
    base_color.Set(Sdf.AssetPath("albedo.png"))
    base_color.SetColorSpace("sRGB")
    vertex_color = shader.CreateInput(
        "UV_VertexColor", Sdf.ValueTypeNames.Color3f
    ).GetAttr()
    vertex_color.SetColorSpace("raw")
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["VM.TEX.002"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["VM.TEX.002"]
    assert asset.read_text().count('colorSpace = "sRGB"') == 1
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    repaired_shader = repaired_stage.GetPrimAtPath("/asset/Looks/Shader")
    assert (
        repaired_shader.GetAttribute("inputs:base_color_texture_file").GetColorSpace()
        == "sRGB"
    )
    assert (
        repaired_shader.GetAttribute("inputs:UV_VertexColor").GetColorSpace() == "raw"
    )
    repair_payload = json.loads(Path(report.reports["VM.TEX.002"]).read_text())
    assert repair_payload["changed_attributes"] == []
    assert repair_payload["ambiguous_attributes"] == [
        {
            "attribute_path": "/asset/Looks/Shader.inputs:base_color_texture_file",
            "color_space": "sRGB",
            "validator_expected_color_space": "raw",
        }
    ]


def test_simready_conformance_repairs_empty_vm_tex_002_texture_slot(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = stage.DefinePrim("/asset", "Xform")
    stage.SetDefaultPrim(root)
    shader = UsdShade.Shader.Define(stage, "/asset/Looks/Shader")
    empty_texture = shader.CreateInput(
        "base_color_texture_file", Sdf.ValueTypeNames.Asset
    ).GetAttr()
    empty_texture.Set(Sdf.AssetPath(""))
    empty_texture.SetColorSpace("auto")
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["VM.TEX.002"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    repaired = repaired_stage.GetAttributeAtPath(
        "/asset/Looks/Shader.inputs:base_color_texture_file"
    )
    assert repaired.Get().path == ""
    assert repaired.GetColorSpace() == "raw"
    repair_payload = json.loads(Path(report.reports["VM.TEX.002"]).read_text())
    assert repair_payload.get("ambiguous_attributes", []) == []
    assert repair_payload["changed_attributes"] == [
        {
            "attribute_path": ("/asset/Looks/Shader.inputs:base_color_texture_file"),
            "color_space": "raw",
            "previous_color_space": "auto",
        }
    ]


def test_simready_conformance_blocks_time_sampled_vm_tex_002_texture_slot(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = stage.DefinePrim("/asset", "Xform")
    stage.SetDefaultPrim(root)
    texture = (
        UsdShade.Shader.Define(stage, "/asset/Looks/Shader")
        .CreateInput("coat_texture_file", Sdf.ValueTypeNames.Asset)
        .GetAttr()
    )
    texture.Set(Sdf.AssetPath("coat.png"), 1.0)
    texture.SetColorSpace("auto")
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["VM.TEX.002"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["VM.TEX.002"]
    repair_payload = json.loads(Path(report.reports["VM.TEX.002"]).read_text())
    assert repair_payload["ambiguous_attributes"] == [
        {
            "attribute_path": "/asset/Looks/Shader.inputs:coat_texture_file",
            "color_space": "auto",
            "validator_expected_color_space": "raw",
        }
    ]


def test_simready_conformance_blocks_ambiguous_visual_material_assignment(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/visual_mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    for name in ["RedMaterial", "BlueMaterial"]:
        material = UsdShade.Material.Define(stage, f"/robot/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"/robot/Looks/{name}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["VM.MAT.001"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.status == "BLOCKED"
    assert report.requirements_blocked == ["VM.MAT.001"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    bound, _rel = UsdShade.MaterialBindingAPI(
        repaired_stage.GetPrimAtPath("/robot/visual_mesh")
    ).ComputeBoundMaterial()
    assert not bound


def test_simready_conformance_binds_visual_material_subsets(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/visual_mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    subset = UsdShade.MaterialBindingAPI(mesh.GetPrim()).CreateMaterialBindSubset(
        "partA", [0]
    )
    material = UsdShade.Material.Define(stage, "/robot/Looks/VisualMaterial")
    shader = UsdShade.Shader.Define(stage, "/robot/Looks/VisualMaterial/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["VM.MAT.001"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    for path in ["/robot/visual_mesh", str(subset.GetPath())]:
        bound, _rel = UsdShade.MaterialBindingAPI(
            repaired_stage.GetPrimAtPath(path)
        ).ComputeBoundMaterial()
        assert bound
        assert bound.GetPath() == material.GetPath()


def test_simready_conformance_preserves_full_purpose_visual_bindings(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    full_bound_mesh = UsdGeom.Mesh.Define(stage, "/robot/full_bound_mesh")
    full_bound_mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    full_bound_mesh.CreateFaceVertexCountsAttr([3])
    full_bound_mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    unbound_mesh = UsdGeom.Mesh.Define(stage, "/robot/unbound_mesh")
    unbound_mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    unbound_mesh.CreateFaceVertexCountsAttr([3])
    unbound_mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    material = UsdShade.Material.Define(stage, "/robot/Looks/VisualMaterial")
    shader = UsdShade.Shader.Define(stage, "/robot/Looks/VisualMaterial/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(full_bound_mesh.GetPrim()).Bind(
        material, materialPurpose=UsdShade.Tokens.full
    )
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["VM.MAT.001"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    repaired_full_bound = repaired_stage.GetPrimAtPath("/robot/full_bound_mesh")
    assert not repaired_full_bound.GetRelationship("material:binding").IsValid()
    full_material, _rel = UsdShade.MaterialBindingAPI(
        repaired_full_bound
    ).ComputeBoundMaterial(materialPurpose=UsdShade.Tokens.full)
    assert full_material
    assert full_material.GetPath() == material.GetPath()
    repaired_unbound = repaired_stage.GetPrimAtPath("/robot/unbound_mesh")
    unbound_material, _rel = UsdShade.MaterialBindingAPI(
        repaired_unbound
    ).ComputeBoundMaterial(materialPurpose=UsdShade.Tokens.full)
    assert unbound_material
    assert unbound_material.GetPath() == material.GetPath()


def test_simready_conformance_resolves_agent_skill_layout_for_local_repairs(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path, skill_layout="agents")
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/visual_mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    material = UsdShade.Material.Define(stage, "/robot/Looks/VisualMaterial")
    shader = UsdShade.Shader.Define(stage, "/robot/Looks/VisualMaterial/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["VM.MAT.001"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    assert (
        report.steps[0]["upstream_skill_path"]
        .replace("\\", "/")
        .endswith(".agents/skills/simready-conform-fet_006-materials/SKILL.md")
    )


def test_simready_conformance_blocks_physics_material_without_sourced_material(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/collider_mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["PMT.001"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["PMT.001"]
    assert report.steps[0]["upstream_skill"].endswith("fet-007-nonvisual-materials")
    assert (
        report.steps[0]["upstream_skill_path"]
        .replace("\\", "/")
        .endswith(
            "skills/simready-foundation-conform-fet-007-nonvisual-materials/SKILL.md"
        )
    )
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    assert not repaired_stage.GetPrimAtPath(
        "/robot/Looks/SimReadyPhysicsMaterial"
    ).IsValid()


def test_simready_conformance_repairs_invalid_physics_material_targets(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/collider_mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh.GetPrim().CreateRelationship("material:binding:physics").SetTargets(
        ["/robot/Looks/MissingPhysicsMaterial"]
    )
    material = UsdShade.Material.Define(stage, "/robot/Looks/PhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["PMT.001"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    targets = (
        repaired_stage.GetPrimAtPath("/robot/collider_mesh")
        .GetRelationship("material:binding:physics")
        .GetTargets()
    )
    assert targets == [material.GetPath()]


def test_simready_conformance_blocks_ambiguous_physics_material_assignment(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/collider_mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    for path in [
        "/robot/Looks/RubberPhysicsMaterial",
        "/robot/Looks/MetalPhysicsMaterial",
    ]:
        material = UsdShade.Material.Define(stage, path)
        UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["PMT.001"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["PMT.001"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    relationship = repaired_stage.GetPrimAtPath("/robot/collider_mesh").GetRelationship(
        "material:binding:physics"
    )
    assert not relationship or not relationship.GetTargets()


def test_simready_conformance_materializes_inherited_physics_material_binding(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/collider_mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    rubber = UsdShade.Material.Define(stage, "/robot/Looks/RubberPhysicsMaterial")
    metal = UsdShade.Material.Define(stage, "/robot/Looks/MetalPhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(rubber.GetPrim())
    UsdPhysics.MaterialAPI.Apply(metal.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(robot).Bind(rubber, materialPurpose="physics")
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["PMT.001"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    repaired_mesh = repaired_stage.GetPrimAtPath("/robot/collider_mesh")
    assert repaired_mesh.GetRelationship("material:binding:physics").GetTargets() == [
        rubber.GetPath()
    ]
    material, _rel = UsdShade.MaterialBindingAPI(repaired_mesh).ComputeBoundMaterial(
        materialPurpose="physics"
    )
    assert material
    assert material.GetPath() == rubber.GetPath()


def test_simready_conformance_pairs_mesh_collision_api_with_collision_api(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/collisionMesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    material = UsdShade.Material.Define(stage, "/robot/Looks/PhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["RB.COL.002"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    repaired_mesh = repaired_stage.GetPrimAtPath("/robot/collisionMesh")
    assert repaired_mesh.HasAPI(UsdPhysics.CollisionAPI)
    assert repaired_mesh.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert report.requirements_repaired == ["PMT.001", "RB.COL.002"]
    targets = repaired_mesh.GetRelationship("material:binding:physics").GetTargets()
    assert targets == [material.GetPath()]


def test_simready_conformance_migrates_collision_only_to_analytic_gprim(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    owner = stage.DefinePrim("/robot/link_collider", "Xform")
    UsdPhysics.CollisionAPI.Apply(owner)
    cube = UsdGeom.Cube.Define(stage, "/robot/link_collider/shape")
    visual_mesh = UsdGeom.Mesh.Define(stage, "/robot/link_collider/visual_collider")
    visual_mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    visual_mesh.CreateFaceVertexCountsAttr([3])
    visual_mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    material = UsdShade.Material.Define(stage, "/robot/Looks/PhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["RB.COL.001"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    assert report.requirements_repaired == ["PMT.001", "RB.COL.001"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    repaired_owner = repaired_stage.GetPrimAtPath("/robot/link_collider")
    repaired_cube = repaired_stage.GetPrimAtPath(cube.GetPath())
    repaired_visual_mesh = repaired_stage.GetPrimAtPath(visual_mesh.GetPath())
    assert not repaired_owner.HasAPI(UsdPhysics.CollisionAPI)
    assert repaired_cube.HasAPI(UsdPhysics.CollisionAPI)
    assert not repaired_visual_mesh.HasAPI(UsdPhysics.CollisionAPI)
    targets = repaired_cube.GetRelationship("material:binding:physics").GetTargets()
    assert targets == [material.GetPath()]


def test_simready_collider_migration_candidates_use_mesh_schema_inheritance() -> None:
    class FakeMeshSchema:
        pass

    class FakeUsdGeom:
        Mesh = FakeMeshSchema

    class FakePrim:
        def __init__(self, name: str, *, is_mesh: bool) -> None:
            self.name = name
            self.is_mesh = is_mesh

        def IsA(self, schema: object) -> bool:
            assert schema is FakeMeshSchema
            return self.is_mesh

    derived_mesh = FakePrim("derivedMesh", is_mesh=True)
    analytic_gprim = FakePrim("capsule", is_mesh=False)

    assert conform_profile_module._collider_migration_candidates(
        descendant_gprims=[derived_mesh, analytic_gprim],
        requires_mesh=True,
        UsdGeom=FakeUsdGeom,
    ) == [derived_mesh]
    assert conform_profile_module._collider_migration_candidates(
        descendant_gprims=[derived_mesh, analytic_gprim],
        requires_mesh=False,
        UsdGeom=FakeUsdGeom,
    ) == [analytic_gprim]
    assert conform_profile_module._collider_migration_candidates(
        descendant_gprims=[derived_mesh],
        requires_mesh=False,
        UsdGeom=FakeUsdGeom,
    ) == [derived_mesh]


def test_simready_collider_migration_uses_collision_designated_owner_path() -> None:
    class FakePrim:
        def __init__(self, path: str, *, display_name: str | None = None) -> None:
            self.path = path
            self.display_name = display_name

        def GetPath(self) -> str:
            return self.path

        def GetName(self) -> str:
            return self.path.rsplit("/", 1)[-1]

        def GetMetadata(self, name: str) -> str | None:
            assert name == "displayName"
            return self.display_name

    owner = FakePrim(
        "/robot/links/body/collisions/wrapper",
        display_name="collider: wrapper",
    )
    candidates = [
        FakePrim(f"{owner.path}/geometry_a"),
        FakePrim(f"{owner.path}/geometry_b"),
    ]

    assert conform_profile_module._collider_migration_targets(owner, candidates) == (
        candidates
    )
    unmarked_collision_owner = FakePrim("/robot/links/body/collisions/wrapper")
    assert (
        conform_profile_module._collider_migration_targets(
            unmarked_collision_owner,
            [FakePrim(f"{unmarked_collision_owner.path}/geometry")],
        )
        == []
    )
    visual_owner = FakePrim("/robot/links/body/visuals/wrapper")
    assert (
        conform_profile_module._collider_migration_targets(
            visual_owner,
            [FakePrim(f"{visual_owner.path}/geometry")],
        )
        == []
    )
    distant_collision_owner = FakePrim("/collision_robot/links/body/visuals/wrapper")
    assert (
        conform_profile_module._collider_migration_targets(
            distant_collision_owner,
            [FakePrim(f"{distant_collision_owner.path}/geometry")],
        )
        == []
    )


def test_simready_conformance_expands_pmt_for_non_mesh_collision_migration(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    owner = stage.DefinePrim("/robot/collider", "Xform")
    UsdPhysics.MeshCollisionAPI.Apply(owner)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/collider/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    material = UsdShade.Material.Define(stage, "/robot/Looks/PhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["RB.COL.002"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    assert report.requirements_repaired == ["PMT.001", "RB.COL.002"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    repaired_owner = repaired_stage.GetPrimAtPath("/robot/collider")
    repaired_mesh = repaired_stage.GetPrimAtPath("/robot/collider/mesh")
    assert not repaired_owner.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert repaired_mesh.HasAPI(UsdPhysics.CollisionAPI)
    assert repaired_mesh.HasAPI(UsdPhysics.MeshCollisionAPI)
    targets = repaired_mesh.GetRelationship("material:binding:physics").GetTargets()
    assert targets == [material.GetPath()]


def test_simready_conformance_preserves_mesh_collision_only_physics_binding(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    owner = stage.DefinePrim("/robot/collider", "Xform")
    UsdPhysics.MeshCollisionAPI.Apply(owner)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/collider/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    rubber = UsdShade.Material.Define(stage, "/robot/Looks/RubberPhysicsMaterial")
    metal = UsdShade.Material.Define(stage, "/robot/Looks/MetalPhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(rubber.GetPrim())
    UsdPhysics.MaterialAPI.Apply(metal.GetPrim())
    owner.CreateRelationship("material:binding:physics").SetTargets([rubber.GetPath()])
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["RB.COL.002"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    repaired_owner = repaired_stage.GetPrimAtPath("/robot/collider")
    repaired_mesh = repaired_stage.GetPrimAtPath("/robot/collider/mesh")
    assert not repaired_owner.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert not repaired_owner.GetRelationship("material:binding:physics")
    assert repaired_mesh.HasAPI(UsdPhysics.CollisionAPI)
    targets = repaired_mesh.GetRelationship("material:binding:physics").GetTargets()
    assert targets == [rubber.GetPath()]


def test_simready_conformance_blocks_conflicting_collision_migration_bindings(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    owner = stage.DefinePrim("/robot/collider", "Xform")
    UsdPhysics.MeshCollisionAPI.Apply(owner)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/collider/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    rubber = UsdShade.Material.Define(stage, "/robot/Looks/RubberPhysicsMaterial")
    metal = UsdShade.Material.Define(stage, "/robot/Looks/MetalPhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(rubber.GetPrim())
    UsdPhysics.MaterialAPI.Apply(metal.GetPrim())
    owner.CreateRelationship("material:binding:physics").SetTargets([rubber.GetPath()])
    mesh.GetPrim().CreateRelationship("material:binding:physics").SetTargets(
        [metal.GetPath()]
    )
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["RB.COL.002"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["PMT.001", "RB.COL.002"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    repaired_owner = repaired_stage.GetPrimAtPath("/robot/collider")
    repaired_mesh = repaired_stage.GetPrimAtPath("/robot/collider/mesh")
    assert repaired_owner.HasAPI(UsdPhysics.MeshCollisionAPI)
    targets = repaired_mesh.GetRelationship("material:binding:physics").GetTargets()
    assert targets == [metal.GetPath()]


def test_simready_conformance_blocks_inherited_collision_binding_conflict(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    owner = stage.DefinePrim("/robot/collider", "Xform")
    UsdPhysics.CollisionAPI.Apply(owner)
    UsdPhysics.MeshCollisionAPI.Apply(owner)
    group = stage.DefinePrim("/robot/collider/group", "Xform")
    mesh = UsdGeom.Mesh.Define(stage, "/robot/collider/group/collisionMesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    rubber = UsdShade.Material.Define(stage, "/robot/Looks/RubberPhysicsMaterial")
    metal = UsdShade.Material.Define(stage, "/robot/Looks/MetalPhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(rubber.GetPrim())
    UsdPhysics.MaterialAPI.Apply(metal.GetPrim())
    owner.CreateRelationship("material:binding:physics").SetTargets([rubber.GetPath()])
    UsdShade.MaterialBindingAPI.Apply(group).Bind(metal, materialPurpose="physics")
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["RB.COL.001", "RB.COL.002"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["RB.COL.001", "RB.COL.002"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    repaired_owner = repaired_stage.GetPrimAtPath("/robot/collider")
    repaired_mesh = repaired_stage.GetPrimAtPath("/robot/collider/group/collisionMesh")
    assert repaired_owner.HasAPI(UsdPhysics.CollisionAPI)
    assert repaired_owner.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert not repaired_mesh.GetRelationship("material:binding:physics").IsValid()
    material, _rel = UsdShade.MaterialBindingAPI(repaired_mesh).ComputeBoundMaterial(
        materialPurpose="physics"
    )
    assert material
    assert material.GetPath() == metal.GetPath()


def test_simready_conformance_preserves_target_collider_settings(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    owner = stage.DefinePrim("/robot/collider", "Xform")
    UsdPhysics.CollisionAPI.Apply(owner).CreateCollisionEnabledAttr().Set(False)
    UsdPhysics.MeshCollisionAPI.Apply(owner)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/collider/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh_collision_api = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision_api.CreateCollisionEnabledAttr().Set(True)
    mesh_mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_mesh_collision_api.CreateApproximationAttr().Set("convexHull")
    material = UsdShade.Material.Define(stage, "/robot/Looks/PhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mesh.GetPrim().CreateRelationship("material:binding:physics").SetTargets(
        [material.GetPath()]
    )
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["RB.COL.001", "RB.COL.002"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    repaired_owner = repaired_stage.GetPrimAtPath("/robot/collider")
    repaired_mesh = repaired_stage.GetPrimAtPath("/robot/collider/mesh")
    assert not repaired_owner.HasAPI(UsdPhysics.CollisionAPI)
    assert not repaired_owner.HasAPI(UsdPhysics.MeshCollisionAPI)
    repaired_collision_api = UsdPhysics.CollisionAPI(repaired_mesh)
    assert repaired_collision_api.GetCollisionEnabledAttr().Get() is True
    repaired_mesh_collision_api = UsdPhysics.MeshCollisionAPI(repaired_mesh)
    assert repaired_mesh_collision_api.GetApproximationAttr().Get() == "convexHull"


def test_simready_conformance_repairs_directory_package_root_usd(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    asset = package_dir / "package.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/collisionMesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    material = UsdShade.Material.Define(stage, "/robot/Looks/PhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(package_dir),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["RB.COL.002"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    output_dir = Path(report.output_usd_path)
    assert output_dir.is_dir()
    repaired_stage = Usd.Stage.Open(str(output_dir / "package.usda"))
    assert repaired_stage is not None
    repaired_mesh = repaired_stage.GetPrimAtPath("/robot/collisionMesh")
    assert repaired_mesh.HasAPI(UsdPhysics.CollisionAPI)
    targets = repaired_mesh.GetRelationship("material:binding:physics").GetTargets()
    assert targets == [material.GetPath()]


def test_simready_conformance_migrates_collision_only_to_identified_meshes(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    owner = stage.DefinePrim("/robot/link", "Xform")
    UsdPhysics.CollisionAPI.Apply(owner)
    UsdPhysics.MeshCollisionAPI.Apply(owner)
    for path in ["/robot/link/visual_mesh", "/robot/link/collisionMesh"]:
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    material = UsdShade.Material.Define(stage, "/robot/Looks/PhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["RB.COL.001", "RB.COL.002"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert report.passed
    assert report.requirements_repaired == ["PMT.001", "RB.COL.001", "RB.COL.002"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    repaired_owner = repaired_stage.GetPrimAtPath("/robot/link")
    visual_mesh = repaired_stage.GetPrimAtPath("/robot/link/visual_mesh")
    collision_mesh = repaired_stage.GetPrimAtPath("/robot/link/collisionMesh")
    assert not repaired_owner.HasAPI(UsdPhysics.CollisionAPI)
    assert not repaired_owner.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert not visual_mesh.HasAPI(UsdPhysics.CollisionAPI)
    assert not visual_mesh.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert collision_mesh.HasAPI(UsdPhysics.CollisionAPI)
    assert collision_mesh.HasAPI(UsdPhysics.MeshCollisionAPI)
    targets = collision_mesh.GetRelationship("material:binding:physics").GetTargets()
    assert targets == [material.GetPath()]


def test_simready_conformance_blocks_ambiguous_collision_migration(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    owner = stage.DefinePrim("/robot/link", "Xform")
    UsdPhysics.CollisionAPI.Apply(owner)
    UsdPhysics.MeshCollisionAPI.Apply(owner)
    for path in ["/robot/link/mesh_a", "/robot/link/mesh_b"]:
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["RB.COL.001", "RB.COL.002"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["RB.COL.001", "RB.COL.002"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    assert repaired_stage.GetPrimAtPath("/robot/link").HasAPI(UsdPhysics.CollisionAPI)


def test_simready_conformance_does_not_save_partial_collision_migration_on_block(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)

    unambiguous_owner = stage.DefinePrim("/robot/good_collider", "Xform")
    UsdPhysics.CollisionAPI.Apply(unambiguous_owner)
    UsdPhysics.MeshCollisionAPI.Apply(unambiguous_owner)
    good_mesh = UsdGeom.Mesh.Define(stage, "/robot/good_collider/mesh")
    good_mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    good_mesh.CreateFaceVertexCountsAttr([3])
    good_mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    ambiguous_owner = stage.DefinePrim("/robot/ambiguous_link", "Xform")
    UsdPhysics.CollisionAPI.Apply(ambiguous_owner)
    UsdPhysics.MeshCollisionAPI.Apply(ambiguous_owner)
    for path in ["/robot/ambiguous_link/mesh_a", "/robot/ambiguous_link/mesh_b"]:
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["RB.COL.001", "RB.COL.002"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["PMT.001", "RB.COL.001", "RB.COL.002"]
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    assert repaired_stage.GetPrimAtPath("/robot/good_collider").HasAPI(
        UsdPhysics.CollisionAPI
    )
    assert repaired_stage.GetPrimAtPath("/robot/good_collider").HasAPI(
        UsdPhysics.MeshCollisionAPI
    )
    assert not repaired_stage.GetPrimAtPath("/robot/good_collider/mesh").HasAPI(
        UsdPhysics.CollisionAPI
    )


def test_simready_conformance_blocks_when_repaired_layer_save_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/collider_mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    material = UsdShade.Material.Define(stage, "/robot/Looks/PhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    stage.GetRootLayer().Save()

    save_error = "Could not save repaired USD layer /tmp/readonly.usda."
    monkeypatch.setattr(
        conform_profile_module,
        "_save_stage_root_layer",
        lambda _stage: save_error,
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["PMT.001"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["PMT.001"]
    assert report.steps[0]["reason"] == save_error
    repaired_stage = Usd.Stage.Open(report.output_usd_path)
    assert repaired_stage is not None
    relationship = repaired_stage.GetPrimAtPath("/robot/collider_mesh").GetRelationship(
        "material:binding:physics"
    )
    assert not relationship or not relationship.GetTargets()


def test_simready_conformance_blocks_multi_target_physics_material_binding(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/collider_mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    rubber = UsdShade.Material.Define(stage, "/robot/Looks/RubberPhysicsMaterial")
    metal = UsdShade.Material.Define(stage, "/robot/Looks/MetalPhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(rubber.GetPrim())
    UsdPhysics.MaterialAPI.Apply(metal.GetPrim())
    mesh.GetPrim().CreateRelationship("material:binding:physics").SetTargets(
        [rubber.GetPath(), metal.GetPath()]
    )
    stage.GetRootLayer().Save()

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["PMT.001"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == ["PMT.001"]


def test_simready_renderable_gprims_uses_computed_purpose() -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    stage = Usd.Stage.CreateInMemory()
    proxy_root = stage.DefinePrim("/proxy", "Xform")
    UsdGeom.Imageable(proxy_root).CreatePurposeAttr("proxy")
    mesh = UsdGeom.Mesh.Define(stage, "/proxy/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    assert conform_profile_module._renderable_gprims(stage, UsdGeom) == []


def test_simready_visual_materials_accept_render_context_surface_outputs() -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/Looks/MDLMaterial")
    shader = UsdShade.Shader.Define(stage, "/Looks/MDLMaterial/Shader")
    shader.CreateIdAttr("NVIDIA_MDL")
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader_output)

    materials = conform_profile_module._visual_materials(stage, UsdPhysics, UsdShade)

    assert [str(item.GetPath()) for item in materials] == ["/Looks/MDLMaterial"]


def test_simready_conformance_uses_rerun_reasons_over_ignored_features(
    tmp_path: Path,
    monkeypatch,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    runtime = _trusted_conformance_runtime(foundation_root)
    monkeypatch.setattr(
        conform_profile_module,
        "resolve_simready_runtime",
        lambda **_kwargs: runtime,
    )
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")
    validation_report = tmp_path / "simready-profile.json"
    _write_trusted_validation_report(
        validation_report,
        asset_path=asset,
        runtime=runtime,
        rerun_reasons=["NP.006"],
        extra={
            "ignored_issues": [{"requirement_id": "RB.MB.001"}],
            "feature_results": [
                {
                    "id": "FET004_BASE_PHYSX",
                    "passed": False,
                    "failing requirements": "['RB.MB.001']",
                }
            ],
        },
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            validation_report_path=str(validation_report),
            foundation_root=str(foundation_root),
        )
    )

    assert report.failed_requirements == ["NP.006"]
    assert report.requirements_repaired == ["NP.006"]
    assert report.requirements_blocked == []
    assert "RB.MB.001" not in {step["requirement"] for step in report.steps}


def test_simready_conformance_rejects_unsupported_identity_bound_rerun_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    runtime = _trusted_conformance_runtime(foundation_root)
    monkeypatch.setattr(
        conform_profile_module,
        "resolve_simready_runtime",
        lambda **_kwargs: runtime,
    )
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")
    validation_report = tmp_path / "simready-profile.json"
    _write_trusted_validation_report(
        validation_report,
        asset_path=asset,
        runtime=runtime,
        rerun_reasons=["ZZ.001"],
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            validation_report_path=str(validation_report),
            foundation_root=str(foundation_root),
        )
    )

    assert not report.passed
    assert report.status == "BLOCKED"
    assert report.requirements_skipped == []
    assert any(
        "rerun_reasons contains an unsupported requirement" in item
        for item in report.errors
    )
    assert report.output_usd_path == str(asset.resolve())


def test_simready_conformance_reports_missing_validation_report(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            validation_report_path=str(tmp_path / "missing.json"),
            foundation_root=str(foundation_root),
        )
    )

    assert not report.passed
    assert report.status == "BLOCKED"
    assert any("Validation report does not exist" in item for item in report.errors)
    assert report.next_step == "simready-validate"


def test_simready_conformance_reports_malformed_validation_report(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")
    validation_report = tmp_path / "simready-profile.json"
    validation_report.write_text("{not valid json", encoding="utf-8")

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            validation_report_path=str(validation_report),
            foundation_root=str(foundation_root),
        )
    )

    assert not report.passed
    assert report.status == "BLOCKED"
    assert any(
        "Validation report could not be parsed" in item for item in report.errors
    )
    assert report.next_step == "simready-validate"


def test_simready_conformance_reports_invalid_utf8_validation_report(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")
    validation_report = tmp_path / "simready-profile.json"
    validation_report.write_bytes(b"\xff\xfe")

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            validation_report_path=str(validation_report),
            foundation_root=str(foundation_root),
        )
    )

    assert not report.passed
    assert report.status == "BLOCKED"
    assert any(
        "Validation report could not be parsed" in item for item in report.errors
    )
    assert report.next_step == "simready-validate"


def test_simready_conformance_blocks_untrusted_validation_report(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {}\n', encoding="utf-8")
    validation_report = tmp_path / "simready-profile.json"
    validation_report.write_text(
        json.dumps({"rerun_reasons": ["NP.006"]}),
        encoding="utf-8",
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            validation_report_path=str(validation_report),
            foundation_root=str(foundation_root),
        )
    )

    assert not report.passed
    assert report.status == "BLOCKED"
    assert any("not identity-bound" in item for item in report.errors)
    assert report.output_usd_path == str(asset.resolve())
    assert report.next_step == "simready-validate"


@pytest.mark.parametrize(
    "identity_field",
    [
        "asset_path",
        "profile_name",
        "profile_version",
        "profile_target",
        "foundation_commit",
        "foundation_root",
        "foundation_spec_root",
        "validator_executable",
        "foundation_requirements_sha256",
        "foundation_spec_tree_sha256",
        "runtime_contract_sha256",
        "validator_executable_sha256",
        "validator_distributions_sha256",
        "validator_runtime_verified",
    ],
)
def test_simready_conformance_rejects_wrong_validation_identity(
    tmp_path: Path,
    monkeypatch,
    identity_field: str,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    runtime = _trusted_conformance_runtime(foundation_root)
    monkeypatch.setattr(
        conform_profile_module,
        "resolve_simready_runtime",
        lambda **_kwargs: runtime,
    )
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {}\n', encoding="utf-8")
    other_asset = tmp_path / "other.usda"
    other_asset.write_text('#usda 1.0\n\ndef Xform "Other" {}\n', encoding="utf-8")
    other_root = tmp_path / "other-runtime"
    other_root.mkdir()
    validation_report = tmp_path / "simready-profile.json"
    payload = _write_trusted_validation_report(
        validation_report,
        asset_path=asset,
        runtime=runtime,
        rerun_reasons=["NP.006"],
    )
    wrong_values: dict[str, object] = {
        "asset_path": str(other_asset.resolve()),
        "profile_name": "Prop-Robotics-Physx",
        "profile_version": "9.9.9",
        "profile_target": "Prop-Robotics-Physx@9.9.9",
        "foundation_commit": "9" * 40,
        "foundation_root": str(other_root.resolve()),
        "foundation_spec_root": str(other_root.resolve()),
        "validator_executable": str(other_asset.resolve()),
        "foundation_requirements_sha256": "9" * 64,
        "foundation_spec_tree_sha256": "9" * 64,
        "runtime_contract_sha256": "9" * 64,
        "validator_executable_sha256": "9" * 64,
        "validator_distributions_sha256": "9" * 64,
        "validator_runtime_verified": False,
    }
    payload[identity_field] = wrong_values[identity_field]
    validation_report.write_text(json.dumps(payload), encoding="utf-8")

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            validation_report_path=str(validation_report),
        )
    )

    assert not report.passed
    assert report.status == "BLOCKED"
    assert any(identity_field in item for item in report.errors)
    assert report.output_usd_path == str(asset.resolve())


def test_simready_conformance_rejects_asset_changed_after_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    runtime = _trusted_conformance_runtime(foundation_root)
    monkeypatch.setattr(
        conform_profile_module,
        "resolve_simready_runtime",
        lambda **_kwargs: runtime,
    )
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {}\n', encoding="utf-8")
    validation_report = tmp_path / "simready-profile.json"
    payload = _write_trusted_validation_report(
        validation_report,
        asset_path=asset,
        runtime=runtime,
        rerun_reasons=["NP.006"],
    )
    original_sha256 = payload["asset_sha256"]
    asset.write_text('#usda 1.0\n\ndef Xform "Changed" {}\n', encoding="utf-8")

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            validation_report_path=str(validation_report),
        )
    )

    assert hashlib.sha256(asset.read_bytes()).hexdigest() != original_sha256
    assert not report.passed
    assert report.status == "BLOCKED"
    assert any("asset_sha256 mismatch" in item for item in report.errors)


def test_simready_conformance_rejects_dependency_changed_after_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    runtime = _trusted_conformance_runtime(foundation_root)
    monkeypatch.setattr(
        conform_profile_module,
        "resolve_simready_runtime",
        lambda **_kwargs: runtime,
    )
    layers = tmp_path / "layers"
    layers.mkdir()
    dependency = layers / "sub.usda"
    dependency.write_text(
        '#usda 1.0\n\ndef Xform "Dependency" {}\n',
        encoding="utf-8",
    )
    asset = tmp_path / "asset.usda"
    asset.write_text(
        "#usda 1.0\n(\n    subLayers = [@layers/sub.usda@]\n)\n",
        encoding="utf-8",
    )
    validation_report = tmp_path / "simready-profile.json"
    _write_trusted_validation_report(
        validation_report,
        asset_path=asset,
        runtime=runtime,
        rerun_reasons=["NP.006"],
    )
    dependency.write_text(
        '#usda 1.0\n\ndef Xform "ChangedAfterValidation" {}\n',
        encoding="utf-8",
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            validation_report_path=str(validation_report),
        )
    )

    assert not report.passed
    assert report.status == "BLOCKED"
    assert any(
        "resolved USD dependency content changed" in item for item in report.errors
    )


def test_simready_conformance_rejects_staged_asset_identity_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    runtime = _trusted_conformance_runtime(foundation_root)
    monkeypatch.setattr(
        conform_profile_module,
        "resolve_simready_runtime",
        lambda **_kwargs: runtime,
    )
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {}\n', encoding="utf-8")
    validation_report = tmp_path / "simready-profile.json"
    _write_trusted_validation_report(
        validation_report,
        asset_path=asset,
        runtime=runtime,
        rerun_reasons=["NP.006"],
    )
    stage_input = conform_profile_module._stage_input

    def stage_then_mutate(*args, **kwargs):
        staged_asset, package_root, warnings = stage_input(*args, **kwargs)
        staged_asset.write_text(
            '#usda 1.0\n\ndef Xform "ChangedDuringStaging" {}\n',
            encoding="utf-8",
        )
        return staged_asset, package_root, warnings

    monkeypatch.setattr(conform_profile_module, "_stage_input", stage_then_mutate)

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            validation_report_path=str(validation_report),
        )
    )

    assert not report.passed
    assert report.status == "BLOCKED"
    assert any("Staged asset content" in item for item in report.errors)


def test_simready_conformance_reports_staging_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")

    def fail_stage(*_args, **_kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(conform_profile_module, "_stage_input", fail_stage)

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            foundation_root=str(foundation_root),
        )
    )

    assert not report.passed
    assert report.status == "FAIL"
    assert any("Could not stage asset" in item for item in report.errors)
    assert report.next_step == "fix-asset-staging"


def test_simready_conformance_rejects_unstaged_external_relative_dependency(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    package = tmp_path / "package"
    scene_dir = package / "Scenes"
    scene_dir.mkdir(parents=True)
    external = tmp_path / "shared.usda"
    external.write_text("#usda 1.0\n", encoding="utf-8")
    asset = scene_dir / "asset.usda"
    asset.write_text(
        '#usda 1.0\n\ndef Xform "World" (\n    payload = @../../shared.usda@\n) {\n}\n',
        encoding="utf-8",
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["NP.006"],
            foundation_root=str(foundation_root),
        )
    )

    assert not report.passed
    assert report.status == "FAIL"
    assert any("outside the asset package" in item for item in report.errors)
    assert report.next_step == "fix-asset-staging"


def test_simready_conformance_directory_rejects_external_usd_dependency(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    asset_dir = tmp_path / "asset_package"
    scene_dir = asset_dir / "Scenes"
    scene_dir.mkdir(parents=True)
    external_dir = tmp_path / "Payload"
    external_dir.mkdir()
    (external_dir / "Contents.usda").write_text("#usda 1.0\n", encoding="utf-8")
    asset = scene_dir / "asset.usda"
    asset.write_text(
        '#usda 1.0\n\ndef Xform "World" (\n    payload = @../../Payload/Contents.usda@\n) {\n}\n',
        encoding="utf-8",
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset_dir),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["NP.006"],
            foundation_root=str(foundation_root),
        )
    )

    assert not report.passed
    assert report.status == "FAIL"
    assert any("outside the directory package" in item for item in report.errors)
    assert report.next_step == "fix-asset-staging"


def test_simready_conformance_force_clears_stale_staged_output(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    package = tmp_path / "package"
    package.mkdir()
    asset = package / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")
    output_dir = tmp_path / "conform"
    stale = output_dir / "staged" / "stale.usda"
    stale.parent.mkdir(parents=True)
    stale.write_text("#usda 1.0\n", encoding="utf-8")

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(output_dir),
            repair_requirements=["NP.006"],
            foundation_root=str(foundation_root),
            force=True,
        )
    )

    assert Path(report.output_usd_path).exists()
    assert not stale.exists()


def test_simready_layer_authored_dependencies_skips_text_export_for_usdc() -> None:
    class FakeFormat:
        formatId = "usdc"

    class FakeLayer:
        identifier = "asset.usdc"

        def GetFileFormat(self):
            return FakeFormat()

        def GetCompositionAssetDependencies(self):
            return ["from-api.usda"]

        def ExportToString(self):
            raise AssertionError("binary layers should not be text-exported")

    assert conform_profile_module._layer_authored_dependencies(FakeLayer()) == [
        "from-api.usda"
    ]


def test_simready_layer_authored_dependencies_allows_text_export_for_usda() -> None:
    class FakeFormat:
        formatId = "usda"

    class FakeLayer:
        identifier = "asset.usda"

        def GetFileFormat(self):
            return FakeFormat()

        def ExportToString(self):
            return '#usda 1.0\n\ndef Xform "World" (payload = @Payload.usda@) {}\n'

    assert conform_profile_module._layer_authored_dependencies(FakeLayer()) == [
        "Payload.usda"
    ]


def test_simready_layer_dependencies_skip_text_export_after_native_api() -> None:
    class FakeFormat:
        formatId = "usda"

    class FakeLayer:
        identifier = "asset.usda"

        def GetFileFormat(self):
            return FakeFormat()

        def GetExternalReferences(self):
            return ["from-api.usda"]

        def ExportToString(self):
            raise AssertionError(
                "text fallback should not run after native API success"
            )

    assert conform_profile_module._layer_authored_dependencies(FakeLayer()) == [
        "from-api.usda"
    ]


def test_simready_conformance_blocks_unverified_managed_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {}\n', encoding="utf-8")
    foundation_root = tmp_path / "managed-foundation"
    foundation_root.mkdir()
    runtime = foundation_runtime_module.SimReadyRuntimeInfo(
        foundation_root=str(foundation_root),
        foundation_commit="b" * 40,
        foundation_spec_root=str(foundation_root / "specs"),
        managed_foundation_checkout=True,
        foundation_checkout_verified=False,
        errors=["Managed SimReady Foundation release identity differs."],
    )
    monkeypatch.setattr(
        conform_profile_module,
        "resolve_simready_runtime",
        lambda **_kwargs: runtime,
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
        )
    )

    assert not report.passed
    assert report.status == "BLOCKED"
    assert not report.foundation_checkout_verified
    assert report.output_usd_path == str(asset.resolve())
    assert report.next_step == "prepare-simready-foundation-runtime"


def test_simready_conformance_noop_passes_without_foundation(tmp_path: Path) -> None:
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            foundation_root=str(tmp_path / "missing-foundation"),
        )
    )

    assert report.passed
    assert report.status == "PASS"
    assert not report.requirements_blocked
    assert any("No validation report" in item for item in report.warnings)


def test_simready_conformance_package_root_falls_back_on_commonpath_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset = tmp_path / "Scenes" / "asset.usda"
    dependency = tmp_path / "Payload" / "Contents.usda"
    asset.parent.mkdir(parents=True)
    dependency.parent.mkdir(parents=True)

    def fail_commonpath(_paths):
        raise ValueError("paths are on different drives")

    monkeypatch.setattr(conform_profile_module.os.path, "commonpath", fail_commonpath)

    assert (
        conform_profile_module._staging_package_root(asset, [dependency])
        == asset.parent.resolve()
    )


def test_simready_conformance_package_root_ignores_external_dependencies(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    asset = package / "Scenes" / "asset.usda"
    local_dependency = package / "Payload" / "Contents.usda"
    external_dependency = Path("/opt/shared/lib.usda")
    asset.parent.mkdir(parents=True)
    local_dependency.parent.mkdir(parents=True)

    assert (
        conform_profile_module._staging_package_root(
            asset, [external_dependency, local_dependency]
        )
        == package.resolve()
    )


def test_simready_dependency_path_rejects_absolute_authored_paths(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "Scenes"
    external_dependency = tmp_path / "external" / "lib.usda"
    source_root.mkdir()
    external_dependency.parent.mkdir()
    external_dependency.write_text("#usda 1.0\n", encoding="utf-8")

    assert (
        conform_profile_module._dependency_path(str(external_dependency), source_root)
        is None
    )
    assert (
        conform_profile_module._dependency_path(
            "C:\\omniverse\\shared\\lib.usda",
            source_root,
        )
        is None
    )


def test_simready_dependency_staging_rejects_authored_symlink_path(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    source_root = package / "Scenes"
    payload_dir = package / "Payload"
    external_dir = tmp_path / "external"
    source_root.mkdir(parents=True)
    payload_dir.mkdir()
    external_dir.mkdir()
    asset = source_root / "asset.usda"
    asset.write_text(
        '#usda 1.0\n\ndef Xform "World" (\n    payload = @../Payload/link.usda@\n) {\n}\n',
        encoding="utf-8",
    )
    external_dependency = external_dir / "Contents.usda"
    external_dependency.write_text("#usda 1.0\n", encoding="utf-8")
    authored_link = payload_dir / "link.usda"
    try:
        authored_link.symlink_to(external_dependency)
    except OSError as exc:  # pragma: no cover - platform permission dependent
        pytest.skip(f"symlink creation is unavailable: {exc}")

    dependency = conform_profile_module._dependency_path(
        "../Payload/link.usda",
        source_root,
    )

    assert dependency == conform_profile_module._absolute_path(authored_link)
    assert dependency != external_dependency.resolve()

    staged_dir = tmp_path / "conform" / "staged"
    with pytest.raises(ValueError, match="through a symlink"):
        conform_profile_module._stage_usd_dependencies(
            asset_path=asset,
            dependencies=[dependency],
            package_root=conform_profile_module._absolute_path(package),
            staged_dir=staged_dir,
        )

    assert not (staged_dir / "Payload" / "link.usda").exists()
    assert not (staged_dir / "external" / "Contents.usda").exists()
    assert external_dependency.read_text(encoding="utf-8") == "#usda 1.0\n"


def test_simready_conformance_stages_texture_asset_dependencies(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    package = tmp_path / "package"
    scene_dir = package / "Scenes"
    texture_dir = package / "textures"
    scene_dir.mkdir(parents=True)
    texture_dir.mkdir()
    (texture_dir / "albedo.png").write_bytes(b"png")
    asset = scene_dir / "asset.usda"
    asset.write_text(
        '#usda 1.0\n\ndef Material "M"\n{\n    asset inputs:file = @../textures/albedo.png@\n}\n',
        encoding="utf-8",
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["NP.006"],
            foundation_root=str(foundation_root),
        )
    )

    output_path = Path(report.output_usd_path)
    assert output_path.parent.name == "Scenes"
    assert (output_path.parent.parent / "textures" / "albedo.png").read_bytes() == (
        b"png"
    )


def test_simready_conformance_stages_relative_payloads(tmp_path: Path) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    package = tmp_path / "package"
    payload_dir = package / "Payload"
    payload_dir.mkdir(parents=True)
    (payload_dir / "Contents.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (package / "unreferenced.bin").write_bytes(b"do not copy")
    asset = package / "asset.usda"
    asset.write_text(
        '#usda 1.0\n\ndef Xform "World" (\n    payload = @Payload/Contents.usda@\n) {\n}\n',
        encoding="utf-8",
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["NP.006"],
            foundation_root=str(foundation_root),
        )
    )

    output_path = Path(report.output_usd_path)
    assert output_path.exists()
    assert (output_path.parent / "Payload" / "Contents.usda").exists()
    assert not (output_path.parent / "unreferenced.bin").exists()


def test_simready_conformance_preserves_parent_traversal_payloads(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    package = tmp_path / "package"
    scene_dir = package / "Scenes"
    payload_dir = package / "Payload"
    scene_dir.mkdir(parents=True)
    payload_dir.mkdir(parents=True)
    (payload_dir / "Contents.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (package / "unreferenced.bin").write_bytes(b"do not copy")
    asset = scene_dir / "asset.usda"
    asset.write_text(
        '#usda 1.0\n\ndef Xform "World" (\n    payload = @../Payload/Contents.usda@\n) {\n}\n',
        encoding="utf-8",
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["NP.006"],
            foundation_root=str(foundation_root),
        )
    )

    output_path = Path(report.output_usd_path)
    assert output_path.exists()
    assert output_path.name == "asset.usda"
    assert output_path.parent.name == "Scenes"
    assert (output_path.parent.parent / "Payload" / "Contents.usda").exists()
    assert not (output_path.parent.parent / "unreferenced.bin").exists()


def test_default_simready_report_names_are_asset_namespaced(tmp_path: Path) -> None:
    first = tmp_path / "one" / "asset.usda"
    second = tmp_path / "two" / "asset.usda"

    first_name = default_simready_report_name(first, kind="profile")
    second_name = default_simready_report_name(second, kind="profile")

    assert first_name.startswith("asset-")
    assert first_name.endswith("-simready-profile.json")
    assert first_name != second_name


def test_simready_validation_namespaces_same_directory_run_records(
    tmp_path: Path,
) -> None:
    foundation_root = _write_fake_foundation(tmp_path)
    venv = _write_fake_venv(tmp_path)
    asset_a = tmp_path / "asset-a.usda"
    asset_b = tmp_path / "asset-b.usda"
    asset_a.write_text("#usda 1.0\n", encoding="utf-8")
    asset_b.write_text("#usda 1.0\n", encoding="utf-8")
    report_a = tmp_path / "asset-a.simready.json"
    report_b = tmp_path / "asset-b.simready.json"
    generic_request = tmp_path / "request.json"
    generic_manifest = tmp_path / "workflow_run_manifest.json"
    generic_request.write_text('{"owner":"existing-request"}\n', encoding="utf-8")
    generic_manifest.write_text('{"owner":"existing-manifest"}\n', encoding="utf-8")

    result_a = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset_a),
            report_path=str(report_a),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )
    manifest_a_path = Path(result_a.workflow_run_manifest_path or "")
    manifest_a_before = manifest_a_path.read_bytes()

    result_b = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(asset_b),
            report_path=str(report_b),
            foundation_root=str(foundation_root),
            venv_path=str(venv),
            install_missing=False,
        )
    )
    manifest_b_path = Path(result_b.workflow_run_manifest_path or "")

    assert manifest_a_path == (
        tmp_path / ".asset-a.simready.json.workflow-run" / "workflow_run_manifest.json"
    )
    assert manifest_b_path == (
        tmp_path / ".asset-b.simready.json.workflow-run" / "workflow_run_manifest.json"
    )
    assert manifest_a_path != manifest_b_path
    assert manifest_a_path.read_bytes() == manifest_a_before
    assert generic_request.read_text(encoding="utf-8") == (
        '{"owner":"existing-request"}\n'
    )
    assert generic_manifest.read_text(encoding="utf-8") == (
        '{"owner":"existing-manifest"}\n'
    )


def test_simready_validation_records_unexpected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    report_path = tmp_path / "asset.simready.json"

    def fail_runtime(**_kwargs: object) -> object:
        raise RuntimeError("unexpected validation failure")

    monkeypatch.setattr(
        validate_profile_module,
        "resolve_simready_runtime",
        fail_runtime,
    )

    with pytest.raises(RuntimeError, match="unexpected validation failure"):
        run_simready_profile_validation(
            SimReadyValidationInput(
                asset_path=str(asset),
                report_path=str(report_path),
            )
        )

    manifest_path = (
        tmp_path / ".asset.simready.json.workflow-run" / "workflow_run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "fail"
    assert manifest["failure"] == {
        "code": "unexpected_workflow_error",
        "error_type": "RuntimeError",
        "message": "unexpected validation failure",
    }


def test_simready_conformance_records_unexpected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    output_dir = tmp_path / "conform"

    def fail_runtime(**_kwargs: object) -> object:
        raise RuntimeError("unexpected conformance failure")

    monkeypatch.setattr(
        conform_profile_module,
        "resolve_simready_runtime",
        fail_runtime,
    )

    with pytest.raises(RuntimeError, match="unexpected conformance failure"):
        run_simready_profile_conformance(
            SimReadyConformanceInput(
                asset_path=str(asset),
                output_dir=str(output_dir),
            )
        )

    report_name = default_simready_report_name(asset, kind="conformance")
    manifest = json.loads(
        (
            output_dir / f".{report_name}.workflow-run" / "workflow_run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "fail"
    assert manifest["failure"] == {
        "code": "unexpected_workflow_error",
        "error_type": "RuntimeError",
        "message": "unexpected conformance failure",
    }
