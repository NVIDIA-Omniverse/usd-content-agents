# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import math
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import apps.texture_gen_step1x_service.backend as step1x_backend_module
import pytest
from apps.texture_gen_service_common import (
    Conditioning,
    CreateJobRequest,
    TextureGenerationBackendError,
)
from apps.texture_gen_step1x_service.backend import (
    ExternalStep1XRunner,
    Step1XBackend,
    Step1XBackendConfig,
    Step1XRunRequest,
    Step1XRunResult,
    Step1XScopeInfo,
    _bundled_runtime_dir,
    _redact_command,
)
from fastapi.testclient import TestClient


class _RecordingRunner:
    def __init__(self) -> None:
        self.requests: list[Step1XRunRequest] = []

    def run(
        self,
        request: Step1XRunRequest,
        *,
        cancel_event: threading.Event,
    ) -> Step1XRunResult:
        self.requests.append(request)
        assert not cancel_event.is_set()
        return Step1XRunResult(
            albedo_uri=(request.output_dir / "albedo.png").as_uri(),
            width=request.texture_size,
            height=request.texture_size,
            metadata={"runner": "fake-step1x"},
        )


def _request() -> CreateJobRequest:
    return CreateJobRequest(
        source_asset_uri="file:///assets/chair.usd",
        conditioning=Conditioning(
            text_prompt="aged red leather with worn edges",
            reference_image_uris=["file:///assets/ref.png"],
            multiview_image_uris=["file:///assets/view0.png"],
        ),
        configuration={
            "seed": 1234,
            "strength": 0.65,
            "texture_size": 512,
            "variant_name": "aged_leather",
            "custom_parameters": {"guidance_scale": 7.5},
        },
        target={
            "material_name": "Leather",
            "material_path": "/World/Looks/Leather",
            "prim_paths": ["/World/Chair/Seat"],
            "mode": "per_material",
            "strict_scope": True,
        },
    )


def _copy_request(
    request: CreateJobRequest,
    *,
    source_asset_uri: str | None = None,
    target_updates: dict[str, object] | None = None,
    custom_parameters: dict[str, object] | None = None,
) -> CreateJobRequest:
    updates: dict[str, object] = {}
    if source_asset_uri is not None:
        updates["source_asset_uri"] = source_asset_uri
    if target_updates:
        assert request.target is not None
        updates["target"] = request.target.model_copy(update=target_updates)
    if custom_parameters is not None:
        updates["configuration"] = request.configuration.model_copy(
            update={"custom_parameters": custom_parameters}
        )
    return request.model_copy(update=updates)


def test_generate_passes_texture_request_fields_to_runner(tmp_path: Path) -> None:
    runner = _RecordingRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path / "step1x",
            model_dir=tmp_path / "models",
            cache_dir=tmp_path / "cache",
            validate_assets=False,
        ),
        runner=runner,
    )

    result = backend.generate(
        _request(),
        job_id="vj-test",
        output_dir=tmp_path / "out",
        cancel_event=threading.Event(),
    )

    assert len(runner.requests) == 1
    run_request = runner.requests[0]
    assert run_request.prompt == "aged red leather with worn edges"
    assert run_request.seed == 1234
    assert run_request.strength == 0.65
    assert run_request.texture_size == 512
    assert run_request.source_asset_uri == "file:///assets/chair.usd"
    assert run_request.reference_image_uris == ("file:///assets/ref.png",)
    assert run_request.multiview_image_uris == ("file:///assets/view0.png",)
    assert run_request.target is not None
    assert run_request.target.material_name == "Leather"
    assert run_request.target.prim_paths == ["/World/Chair/Seat"]
    assert run_request.runtime_dir == tmp_path / "step1x"
    assert run_request.model_dir == tmp_path / "models"
    assert run_request.cache_dir == tmp_path / "cache"
    assert run_request.custom_parameters == {"guidance_scale": 7.5}

    assert result.variant_asset_uri == "file:///assets/chair.usd"
    assert result.variant_name == "aged_leather"


def test_albedo_only_result_reports_degraded_normal_and_orm(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=False),
        runner=runner,
    )

    result = backend.generate(
        _request(),
        job_id="vj-test",
        output_dir=tmp_path,
        cancel_event=threading.Event(),
    )

    assert result.generated_textures.albedo == (tmp_path / "albedo.png").as_uri()
    assert result.generated_textures.normal is None
    assert result.generated_textures.orm is None
    assert set(result.maps) == {"albedo"}
    assert result.metadata["degraded_channels"] == ["normal", "orm"]
    assert result.diagnostics == [
        {
            "code": "STEP1X_MAPS_DEGRADED",
            "severity": "warning",
            "message": "Step1X output omitted optional PBR maps.",
            "channels": ["normal", "orm"],
        }
    ]


def test_generate_rejects_material_anything_when_inputs_unavailable(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=False),
        runner=runner,
    )
    request = _copy_request(
        _request(),
        custom_parameters={"skip_material_anything": False},
    )

    with pytest.raises(TextureGenerationBackendError) as exc_info:
        backend.generate(
            request,
            job_id="vj-ma-unavailable",
            output_dir=tmp_path / "out",
            cancel_event=threading.Event(),
        )

    assert "STEP1X_MATERIAL_ANYTHING_UNAVAILABLE" in str(exc_info.value)
    assert exc_info.value.result is not None
    assert exc_info.value.result.diagnostics[0]["code"] == (
        "STEP1X_MATERIAL_ANYTHING_UNAVAILABLE"
    )
    assert runner.requests == []


def test_generate_rejects_upscale_when_upscaler_unavailable(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=False),
        runner=runner,
    )
    request = _copy_request(_request(), custom_parameters={"upscale": True})

    with pytest.raises(TextureGenerationBackendError) as exc_info:
        backend.generate(
            request,
            job_id="vj-upscaler-unavailable",
            output_dir=tmp_path / "out",
            cancel_event=threading.Event(),
        )

    assert "STEP1X_UPSCALER_UNAVAILABLE" in str(exc_info.value)
    assert exc_info.value.result is not None
    assert exc_info.value.result.diagnostics[0]["code"] == (
        "STEP1X_UPSCALER_UNAVAILABLE"
    )
    assert runner.requests == []


def test_backend_runtime_python_executable_resolution(tmp_path: Path) -> None:
    explicit_python = tmp_path / "explicit-python"
    assert (
        Step1XBackend(
            config=Step1XBackendConfig(python_executable=explicit_python)
        )._runtime_python_executable()
        == explicit_python
    )

    assert (
        Step1XBackend(
            config=Step1XBackendConfig(runtime_dir=None)
        )._runtime_python_executable()
        is None
    )

    runtime_dir = tmp_path / "runtime"
    venv_gen_python = runtime_dir / ".venv_gen" / "bin" / "python"
    venv_gen_python.parent.mkdir(parents=True)
    venv_gen_python.write_text("#!/bin/sh\n", encoding="utf-8")
    assert (
        Step1XBackend(
            config=Step1XBackendConfig(runtime_dir=runtime_dir)
        )._runtime_python_executable()
        == venv_gen_python
    )

    runtime_dir = tmp_path / "runtime-venv"
    venv_python = runtime_dir / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    assert (
        Step1XBackend(
            config=Step1XBackendConfig(runtime_dir=runtime_dir)
        )._runtime_python_executable()
        == venv_python
    )

    empty_runtime = tmp_path / "runtime-empty"
    empty_runtime.mkdir()
    assert (
        Step1XBackend(
            config=Step1XBackendConfig(runtime_dir=empty_runtime)
        )._runtime_python_executable()
        is None
    )


def test_material_anything_readiness_probes_runtime_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    for label, path in step1x_backend_module._material_anything_paths(
        runtime_dir
    ).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if label in {"material_estimator", "material_refiner"}:
            path.mkdir()
        else:
            path.write_text("placeholder\n", encoding="utf-8")
    runtime_python = runtime_dir / ".venv_gen" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("#!/bin/sh\n", encoding="utf-8")
    captured: list[Path | None] = []

    def fake_missing_modules(
        python_executable: Path | None,
        modules: tuple[str, ...],
    ) -> list[str]:
        captured.append(python_executable)
        assert modules == step1x_backend_module._MATERIAL_ANYTHING_REQUIRED_MODULES
        return ["python module kaolin (not importable)"]

    monkeypatch.setattr(
        step1x_backend_module,
        "_missing_python_modules_in_runtime",
        fake_missing_modules,
    )
    backend = Step1XBackend(
        config=Step1XBackendConfig(runtime_dir=runtime_dir, validate_assets=False),
    )

    assert backend._missing_material_anything_inputs() == [
        "Material Anything python module kaolin (not importable)"
    ]
    assert captured == [runtime_python]


def test_full_pbr_result_exposes_orm_without_degradation(tmp_path: Path) -> None:
    class PBRRunner(_RecordingRunner):
        def run(
            self,
            request: Step1XRunRequest,
            *,
            cancel_event: threading.Event,
        ) -> Step1XRunResult:
            self.requests.append(request)
            assert not cancel_event.is_set()
            return Step1XRunResult(
                albedo_uri=(request.output_dir / "final_albedo.png").as_uri(),
                orm_uri=(request.output_dir / "final_orm.png").as_uri(),
                width=request.texture_size,
                height=request.texture_size,
                metadata={"runner": "fake-step1x-ma"},
            )

    runner = PBRRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=False),
        runner=runner,
    )

    result = backend.generate(
        _request(),
        job_id="vj-test",
        output_dir=tmp_path,
        cancel_event=threading.Event(),
    )

    assert result.generated_textures.orm == (tmp_path / "final_orm.png").as_uri()
    assert result.maps["orm"].uri == result.generated_textures.orm
    assert result.maps["orm"].packing == "occlusion_roughness_metallic"
    assert result.metadata["degraded_channels"] == ["normal"]
    assert result.diagnostics[0]["channels"] == ["normal"]


def test_scoped_result_preserves_source_normal_when_runner_omits_normal(
    tmp_path: Path,
) -> None:
    stage_path = _write_preview_surface_texture_stage(tmp_path)
    from PIL import Image
    from pxr import Sdf, Usd

    Image.new("RGB", (2, 2), (127, 128, 255)).save(tmp_path / "paint_normal.bmp")
    stage = Usd.Stage.Open(str(stage_path))
    normal_file = stage.GetPrimAtPath("/World/Looks/Paint/NormalTex").GetAttribute(
        "inputs:file"
    )
    normal_file.Set(Sdf.AssetPath("paint_normal.bmp"))
    stage.GetRootLayer().Save()

    runner = _RecordingRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=True),
        runner=runner,
    )
    output_dir = tmp_path / "out"
    request = _copy_request(
        _request(),
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": "Paint",
            "material_path": "/World/Looks/Paint",
            "prim_paths": ["/World/Mesh"],
        },
    )

    result = backend.generate(
        request,
        job_id="vj-preserve-normal",
        output_dir=output_dir,
        cancel_event=threading.Event(),
    )

    preserved_normal = output_dir / "final_normal.png"
    assert result.generated_textures.normal == preserved_normal.as_uri()
    assert result.maps["normal"].uri == preserved_normal.as_uri()
    with Image.open(preserved_normal) as image:
        assert image.format == "PNG"
        assert image.size == (request.configuration.texture_size,) * 2
        assert image.convert("RGB").getpixel((0, 0)) == (127, 128, 255)
    assert result.metadata["normal_source"] == "source_preserved"
    assert (
        result.metadata["source_normal_uri"] == (tmp_path / "paint_normal.bmp").as_uri()
    )
    assert result.metadata["preserved_channels"] == ["normal"]
    assert result.metadata["degraded_channels"] == ["orm"]
    assert result.diagnostics[0]["channels"] == ["orm"]


def test_preserve_source_normal_uses_source_size_when_result_omits_dimensions(
    tmp_path: Path,
) -> None:
    from PIL import Image

    source_normal = tmp_path / "source_normal.bmp"
    Image.new("RGB", (3, 5), (127, 128, 255)).save(source_normal)

    output_dir = tmp_path / "out"
    raw_result = Step1XRunResult(
        albedo_uri=(tmp_path / "final_albedo.png").as_uri(),
        width=None,
        height=None,
    )
    scoped_result = step1x_backend_module._preserve_source_normal_if_missing(
        raw_result,
        scope=Step1XScopeInfo(
            source_asset_path=tmp_path / "asset.usd",
            source_normal_path=source_normal,
        ),
        output_dir=output_dir,
    )

    preserved_normal = output_dir / "final_normal.png"
    assert scoped_result.normal_uri == preserved_normal.as_uri()
    with Image.open(preserved_normal) as image:
        assert image.size == (3, 5)
        assert image.convert("RGB").getpixel((0, 0)) == (127, 128, 255)


def test_health_reports_missing_runtime_not_ready() -> None:
    backend = Step1XBackend(config=Step1XBackendConfig())

    health = backend.health()

    assert health.status == "not_ready"
    assert health.ready is False
    assert health.error is not None
    assert "TEXTURE_STEP1X_RUNTIME_DIR" in health.error
    assert "TEXTURE_STEP1X_EDIT_SCRIPT" in health.error


def test_health_reports_inaccessible_edit_script_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    edit_script = runtime_dir / "edit_texture.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")

    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if path == edit_script:
            raise PermissionError(13, "Permission denied", str(path))
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=runtime_dir,
            edit_script=edit_script,
            validate_assets=False,
        )
    )

    health = backend.health()

    assert health.status == "not_ready"
    assert health.ready is False
    assert health.error is not None
    assert "TEXTURE_STEP1X_EDIT_SCRIPT" in health.error
    assert "not accessible" in health.error
    assert "Permission denied" in health.error


def test_missing_upscaler_inputs_rejects_unsupported_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    module_path = runtime_dir / "src" / "texture_edit" / "upscaler.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# test upscaler module\n", encoding="utf-8")
    monkeypatch.setenv("TEXTURE_UPSCALER_BACKEND", "bogus")

    missing = step1x_backend_module._missing_upscaler_inputs(runtime_dir)

    assert any(
        "unsupported TEXTURE_UPSCALER_BACKEND='bogus'" in item for item in missing
    )
    assert step1x_backend_module._upscaler_auto_download_writable(runtime_dir) is False


def test_swin2sr_backend_available_allows_auto_and_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = (
        Path(__file__).parents[1] / "runtime" / "src" / "texture_edit" / "upscaler.py"
    )
    if not module_path.is_file():
        pytest.skip("Step1X runtime upscaler is not present in this checkout")

    spec = importlib.util.spec_from_file_location(
        "texture_edit_upscaler_test", module_path
    )
    assert spec is not None and spec.loader is not None
    upscaler = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upscaler
    try:
        spec.loader.exec_module(upscaler)
    finally:
        sys.modules.pop(spec.name, None)

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Torch:
        cuda = _Cuda()

    monkeypatch.setattr(upscaler, "_import_swin2sr_modules", lambda: {"torch": _Torch})

    monkeypatch.delenv("TEXTURE_SWIN2SR_DEVICE", raising=False)
    assert upscaler._swin2sr_backend_available() is True
    monkeypatch.setenv("TEXTURE_SWIN2SR_DEVICE", "cpu")
    assert upscaler._swin2sr_backend_available() is True
    monkeypatch.setenv("TEXTURE_SWIN2SR_DEVICE", "cuda")
    assert upscaler._swin2sr_backend_available() is False


def test_config_from_env_uses_internal_runtime_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TEXTURE_STEP1X_RUNTIME_DIR",
        "TEXTURE_STEP1X_EDIT_SCRIPT",
        "TEXTURE_STEP1X_MODEL_DIR",
        "TEXTURE_STEP1X_CACHE_DIR",
        "TEXTURE_STEP1X_PYTHON",
        "TEXTURE_STEP1X_COMMAND_TEMPLATE",
    ):
        monkeypatch.delenv(name, raising=False)

    config = Step1XBackendConfig.from_env()

    assert config.runtime_dir == _bundled_runtime_dir()


def test_health_reports_internal_runtime_management(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(step1x_backend_module, "_BUNDLED_RUNTIME_SOURCE_PATHS", ())
    bundled_runtime = _bundled_runtime_dir()
    if bundled_runtime is None:
        pytest.skip("repo-internal Step1X runtime is not present in this checkout")

    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=bundled_runtime,
            validate_assets=False,
        )
    )

    health = backend.health()
    runtime_info = health.capabilities.model_dump()["external_runtime"]

    assert health.ready is True
    assert runtime_info["step1x_runtime"] == "repo_internal"
    assert runtime_info["runtime_source"] == "repo_internal"
    assert runtime_info["weights_policy"] == "downloadable_not_committed"


def test_health_reports_missing_internal_runtime_sources_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled_runtime = _bundled_runtime_dir()
    if bundled_runtime is None:
        pytest.skip("repo-internal Step1X runtime is not present in this checkout")
    monkeypatch.setattr(
        step1x_backend_module,
        "_BUNDLED_RUNTIME_SOURCE_PATHS",
        (("Step1X-3D source", Path("third_party") / "Step1X-3D" / "missing"),),
    )
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=bundled_runtime,
            validate_assets=False,
        )
    )

    health = backend.health()

    assert health.ready is False
    assert health.status == "not_ready"
    assert health.error is not None
    assert "Step1X-3D source" in health.error
    assert "run setup_env.sh" in health.error


def test_capabilities_advertise_template_conditioning_only_when_configured() -> None:
    default_backend = Step1XBackend(config=Step1XBackendConfig())
    template_backend = Step1XBackend(
        config=Step1XBackendConfig(command_template="runner --input {source_asset}")
    )

    assert default_backend.capabilities().image_conditioning is False
    assert default_backend.capabilities().multiview is False
    assert template_backend.capabilities().image_conditioning is True
    assert template_backend.capabilities().multiview is True


def test_command_template_health_ignores_missing_runtime_python(tmp_path: Path) -> None:
    missing_python = tmp_path / "missing-python"
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            command_template="runner --input {source_asset}",
            python_executable=missing_python,
            validate_assets=False,
            required_executables=(),
        )
    )

    health = backend.health()

    assert health.ready is True
    assert health.error is None


def test_health_reports_external_runtime_management(tmp_path: Path) -> None:
    edit_script = tmp_path / "custom_edit.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            cache_dir=cache_dir,
            validate_assets=False,
        )
    )

    runtime_info = backend.health().capabilities.model_dump()["external_runtime"]

    assert runtime_info == {
        "api_service": "repo_owned",
        "step1x_runtime": "operator_mounted",
        "runtime_source": "operator_mounted",
        "runtime_dir": str(tmp_path),
        "edit_script": str(edit_script),
        "edit_script_configured": True,
        "command_template_configured": False,
        "model_dir": None,
        "cache_dir": str(cache_dir),
        "validate_assets": False,
        "skip_material_anything_default": True,
        "require_upscaler": False,
        "weights_policy": "downloadable_not_committed",
        "required_executables": [],
    }


def test_health_reports_compose_managed_runtime(tmp_path: Path) -> None:
    edit_script = tmp_path / "edit_texture.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    (tmp_path / ".texture-agent-runtime.json").write_text(
        '{"runtime_source": "compose_managed"}\n',
        encoding="utf-8",
    )
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            validate_assets=False,
            required_executables=(),
        )
    )

    runtime_info = backend.health().capabilities.model_dump()["external_runtime"]

    assert runtime_info["step1x_runtime"] == "compose_managed"
    assert runtime_info["runtime_source"] == "compose_managed"


def test_health_ignores_invalid_compose_runtime_marker(tmp_path: Path) -> None:
    edit_script = tmp_path / "edit_texture.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    (tmp_path / ".texture-agent-runtime.json").write_text(
        "not-json\n",
        encoding="utf-8",
    )
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            validate_assets=False,
            required_executables=(),
        )
    )

    runtime_info = backend.health().capabilities.model_dump()["external_runtime"]

    assert runtime_info["step1x_runtime"] == "operator_mounted"
    assert runtime_info["runtime_source"] == "operator_mounted"


def test_health_reports_material_anything_disabled_without_blocking(
    tmp_path: Path,
) -> None:
    edit_script = tmp_path / "custom_edit.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            validate_assets=False,
            skip_material_anything=True,
            required_executables=(),
        )
    )

    health = backend.health()
    ma_info = health.capabilities.model_dump()["material_anything"]

    assert health.ready is True
    assert ma_info["enabled_by_default"] is False
    assert ma_info["ready"] is False
    assert any("material_estimator" in item for item in ma_info["missing"])


def test_health_requires_material_anything_when_enabled_by_default(
    tmp_path: Path,
) -> None:
    edit_script = tmp_path / "custom_edit.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            validate_assets=False,
            skip_material_anything=False,
            required_executables=(),
        )
    )

    health = backend.health()
    ma_info = health.capabilities.model_dump()["material_anything"]

    assert health.ready is False
    assert health.error is not None
    assert "Material Anything" in health.error
    assert ma_info["enabled_by_default"] is True
    assert ma_info["ready"] is False


def test_health_reports_material_anything_ready_when_assets_exist(
    tmp_path: Path,
) -> None:
    edit_script = tmp_path / "custom_edit.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    ma_dir = tmp_path / "third_party" / "MaterialAnything"
    (ma_dir / "scripts").mkdir(parents=True)
    (ma_dir / "scripts" / "generate_texture_pbr_3d.py").write_text(
        "print('ma')\n",
        encoding="utf-8",
    )
    (ma_dir / "pretrained_models" / "material_estimator").mkdir(parents=True)
    (ma_dir / "pretrained_models" / "material_refiner").mkdir(parents=True)
    controlnet = ma_dir / "models" / "ControlNet" / "models"
    controlnet.mkdir(parents=True)
    (controlnet / "control_sd15_depth.pth").write_bytes(b"fake")
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            validate_assets=False,
            skip_material_anything=False,
            required_executables=(),
        )
    )

    health = backend.health()
    ma_info = health.capabilities.model_dump()["material_anything"]

    assert health.ready is True
    assert ma_info["enabled_by_default"] is True
    assert ma_info["ready"] is True
    assert ma_info["missing"] == []


def test_health_checks_material_anything_modules_with_runtime_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edit_script = tmp_path / "custom_edit.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    ma_dir = tmp_path / "third_party" / "MaterialAnything"
    (ma_dir / "scripts").mkdir(parents=True)
    (ma_dir / "scripts" / "generate_texture_pbr_3d.py").write_text(
        "print('fake')\n",
        encoding="utf-8",
    )
    (ma_dir / "pretrained_models" / "material_estimator").mkdir(parents=True)
    (ma_dir / "pretrained_models" / "material_refiner").mkdir(parents=True)
    controlnet = ma_dir / "models" / "ControlNet" / "models"
    controlnet.mkdir(parents=True)
    (controlnet / "control_sd15_depth.pth").write_bytes(b"fake")
    python_executable = tmp_path / "runtime_python"
    python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_executable.chmod(0o755)
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        **_: object,
    ) -> object:
        calls.append(command)

        class Result:
            returncode = 0
            stdout = "third-party import banner\nWU_MISSING_MODULES_JSON=[]\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(step1x_backend_module.subprocess, "run", fake_run)
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            python_executable=python_executable,
            validate_assets=False,
            skip_material_anything=False,
            required_executables=(),
        )
    )

    health = backend.health()

    assert health.ready is True
    assert calls
    assert calls[0][0] == str(python_executable)
    assert "kaolin" in calls[0][2]


def test_health_reports_material_anything_missing_runtime_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edit_script = tmp_path / "custom_edit.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    ma_dir = tmp_path / "third_party" / "MaterialAnything"
    (ma_dir / "scripts").mkdir(parents=True)
    (ma_dir / "scripts" / "generate_texture_pbr_3d.py").write_text(
        "print('fake')\n",
        encoding="utf-8",
    )
    (ma_dir / "pretrained_models" / "material_estimator").mkdir(parents=True)
    (ma_dir / "pretrained_models" / "material_refiner").mkdir(parents=True)
    controlnet = ma_dir / "models" / "ControlNet" / "models"
    controlnet.mkdir(parents=True)
    (controlnet / "control_sd15_depth.pth").write_bytes(b"fake")
    python_executable = tmp_path / "runtime_python"
    python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_executable.chmod(0o755)

    def fake_run(
        command: list[str],
        **_: object,
    ) -> object:
        class Result:
            returncode = 0
            stdout = 'WU_MISSING_MODULES_JSON=["kaolin"]\n'
            stderr = ""

        return Result()

    monkeypatch.setattr(step1x_backend_module.subprocess, "run", fake_run)
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            python_executable=python_executable,
            validate_assets=False,
            skip_material_anything=False,
            required_executables=(),
        )
    )

    health = backend.health()
    ma_info = health.capabilities.model_dump()["material_anything"]

    assert health.ready is False
    assert "Material Anything python module kaolin (not importable)" in health.error
    assert ma_info["missing"] == [
        "Material Anything python module kaolin (not importable)"
    ]


def test_health_reports_material_anything_runtime_probe_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edit_script = tmp_path / "custom_edit.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    ma_dir = tmp_path / "third_party" / "MaterialAnything"
    (ma_dir / "scripts").mkdir(parents=True)
    (ma_dir / "scripts" / "generate_texture_pbr_3d.py").write_text(
        "print('fake')\n",
        encoding="utf-8",
    )
    (ma_dir / "pretrained_models" / "material_estimator").mkdir(parents=True)
    (ma_dir / "pretrained_models" / "material_refiner").mkdir(parents=True)
    controlnet = ma_dir / "models" / "ControlNet" / "models"
    controlnet.mkdir(parents=True)
    (controlnet / "control_sd15_depth.pth").write_bytes(b"fake")
    python_executable = tmp_path / "runtime_python"
    python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_executable.chmod(0o755)

    def fake_run(command: list[str], **kwargs: object) -> object:
        raise step1x_backend_module.subprocess.TimeoutExpired(
            command,
            timeout=kwargs.get("timeout", 30),
        )

    monkeypatch.setattr(step1x_backend_module.subprocess, "run", fake_run)
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            python_executable=python_executable,
            validate_assets=False,
            skip_material_anything=False,
            required_executables=(),
        )
    )

    health = backend.health()
    ma_info = health.capabilities.model_dump()["material_anything"]

    assert health.ready is False
    assert "Material Anything runtime python module probe failed" in health.error
    assert "timed out after 30s" in health.error
    assert ma_info["missing"] == [
        "Material Anything runtime python module probe failed (timed out after 30s)"
    ]


def test_runtime_module_probe_timeout_uses_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_executable = tmp_path / "runtime_python"
    python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_executable.chmod(0o755)
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_MODULE_PROBE_TIMEOUT_SEC", "180")

    def fake_run(command: list[str], **kwargs: object) -> object:
        raise step1x_backend_module.subprocess.TimeoutExpired(
            command,
            timeout=kwargs.get("timeout", 30),
        )

    monkeypatch.setattr(step1x_backend_module.subprocess, "run", fake_run)

    assert step1x_backend_module._missing_python_modules_in_runtime(
        python_executable,
        ("torch",),
    ) == ["runtime python module probe failed (timed out after 180s)"]


def test_health_can_require_upscaler_for_ready_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEXTURE_UPSCALER_BACKEND", "ncnn-vulkan")
    edit_script = tmp_path / "custom_edit.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    runtime_src = tmp_path / "src" / "texture_edit"
    runtime_src.mkdir(parents=True)
    (runtime_src / "upscaler.py").write_text("print('upscale')\n", encoding="utf-8")
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            validate_assets=False,
            require_upscaler=True,
            required_executables=(),
        )
    )

    health = backend.health()
    upscaler = health.capabilities.model_dump()["upscaler"]

    assert health.ready is True
    assert upscaler["required_for_ready"] is True
    assert upscaler["auto_download_writable"] is True


def test_health_reports_swin2sr_as_default_upscaler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEXTURE_UPSCALER_BACKEND", raising=False)
    monkeypatch.delenv("TEXTURE_REALESRGAN_BACKEND", raising=False)
    monkeypatch.setattr(
        step1x_backend_module,
        "_missing_swin2sr_upscaler_inputs",
        lambda python_executable=None: [],
    )
    edit_script = tmp_path / "custom_edit.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    runtime_src = tmp_path / "src" / "texture_edit"
    runtime_src.mkdir(parents=True)
    (runtime_src / "upscaler.py").write_text("print('upscale')\n", encoding="utf-8")
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            validate_assets=False,
            require_upscaler=True,
            required_executables=(),
        )
    )

    health = backend.health()
    upscaler = health.capabilities.model_dump()["upscaler"]

    assert health.ready is True
    assert upscaler["backend"] == "swin2sr"
    assert upscaler["ready"] is True
    assert upscaler["auto_download_writable"] is False
    assert upscaler["swin2sr_models"] == {
        "x2": "caidas/swin2SR-classical-sr-x2-64",
        "x4": "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    }


def test_health_checks_swin2sr_modules_with_runtime_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEXTURE_UPSCALER_BACKEND", raising=False)
    monkeypatch.delenv("TEXTURE_REALESRGAN_BACKEND", raising=False)
    edit_script = tmp_path / "custom_edit.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    runtime_src = tmp_path / "src" / "texture_edit"
    runtime_src.mkdir(parents=True)
    (runtime_src / "upscaler.py").write_text("print('upscale')\n", encoding="utf-8")
    python_executable = tmp_path / "runtime_python"
    python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_executable.chmod(0o755)
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        **_: object,
    ) -> object:
        calls.append(command)

        class Result:
            returncode = 0
            stdout = "WU_MISSING_MODULES_JSON=[]\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(step1x_backend_module.subprocess, "run", fake_run)
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            python_executable=python_executable,
            validate_assets=False,
            require_upscaler=True,
            required_executables=(),
        )
    )

    health = backend.health()

    assert health.ready is True
    assert calls
    assert calls[0][0] == str(python_executable)
    assert "importlib.import_module" in calls[0][2]


def test_health_reports_swin2sr_missing_module_from_runtime_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEXTURE_UPSCALER_BACKEND", raising=False)
    monkeypatch.delenv("TEXTURE_REALESRGAN_BACKEND", raising=False)
    edit_script = tmp_path / "custom_edit.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    runtime_src = tmp_path / "src" / "texture_edit"
    runtime_src.mkdir(parents=True)
    (runtime_src / "upscaler.py").write_text("print('upscale')\n", encoding="utf-8")
    python_executable = tmp_path / "runtime_python"
    python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_executable.chmod(0o755)

    def fake_run(
        command: list[str],
        **_: object,
    ) -> object:
        class Result:
            returncode = 0
            stdout = 'WU_MISSING_MODULES_JSON=["transformers"]\n'
            stderr = ""

        return Result()

    monkeypatch.setattr(step1x_backend_module.subprocess, "run", fake_run)
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            python_executable=python_executable,
            validate_assets=False,
            require_upscaler=True,
            required_executables=(),
        )
    )

    health = backend.health()
    upscaler = health.capabilities.model_dump()["upscaler"]

    assert health.ready is False
    assert "upscaler python module transformers (not importable)" in health.error
    assert upscaler["missing"] == ["python module transformers (not importable)"]


def test_health_reports_missing_required_executable_not_ready(
    tmp_path: Path,
) -> None:
    edit_script = tmp_path / "custom_edit.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            validate_assets=False,
            required_executables=("definitely-missing-step1x-executable",),
        )
    )

    health = backend.health()
    runtime_info = health.capabilities.model_dump()["external_runtime"]

    assert health.ready is False
    assert health.status == "not_ready"
    assert health.error is not None
    assert "TEXTURE_STEP1X_REQUIRED_EXECUTABLES" in health.error
    assert runtime_info["required_executables"] == [
        {
            "name": "definitely-missing-step1x-executable",
            "available": False,
            "path": None,
        }
    ]


def test_create_app_uses_shared_harness(tmp_path: Path) -> None:
    from apps.texture_gen_service_common import create_app

    runner = _RecordingRunner()
    edit_script = tmp_path / "edit_texture.py"
    edit_script.write_text("print('fake')\n", encoding="utf-8")
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            model_dir=tmp_path,
            edit_script=edit_script,
            validate_assets=False,
        ),
        runner=runner,
    )
    client = TestClient(
        create_app(
            backend=backend,
            output_dir=tmp_path,
            title="Step1X Test",
            service_name="texture-gen-step1x-service",
        )
    )

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ready"] is True
    assert health.json()["backend"] == "step1x"

    created = client.post(
        "/v1/texture-variations",
        json=_request().model_dump(mode="json"),
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]

    status = created.json()
    for _ in range(20):
        status = client.get(f"/v1/texture-variations/{job_id}").json()
        if status["status"] == "completed":
            break
        time.sleep(0.05)

    assert status["status"] == "completed"
    assert status["result"]["generated_textures"]["normal"] is None
    assert runner.requests[0].prompt == "aged red leather with worn edges"


def test_livez_route_reports_process_liveness() -> None:
    from apps.texture_gen_step1x_service.app import app

    client = TestClient(app)

    response = client.get("/livez")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "texture-gen-step1x-service",
    }


def test_scope_resolution_uses_selected_target_material_not_first_mesh(
    tmp_path: Path,
) -> None:
    stage_path = _write_two_material_stage(tmp_path)
    runner = _RecordingRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=True),
        runner=runner,
    )
    request = _request()
    request = _copy_request(
        request,
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": "Blue",
            "material_path": "/World/Looks/Blue",
            "prim_paths": ["/World/MeshB"],
        },
    )

    backend.generate(
        request,
        job_id="vj-scope",
        output_dir=tmp_path / "out",
        cancel_event=threading.Event(),
    )

    assert len(runner.requests) == 1
    scope = runner.requests[0].scope
    assert scope is not None
    assert scope.material_path == "/World/Looks/Blue"
    assert scope.prim_paths == ("/World/MeshB",)
    assert scope.source_albedo_path == (tmp_path / "blue_albedo.png").resolve()


def test_scope_resolution_reads_texture_paths_from_local_mdl(
    tmp_path: Path,
) -> None:
    stage_path = _write_mdl_material_stage(tmp_path)
    runner = _RecordingRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=True),
        runner=runner,
    )
    request = _request()
    request = _copy_request(
        request,
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": "Steel_A",
            "material_path": "/World/Looks/Steel_A",
            "prim_paths": ["/World/Mesh"],
        },
    )

    backend.generate(
        request,
        job_id="vj-mdl-scope",
        output_dir=tmp_path / "out",
        cancel_event=threading.Event(),
    )

    scope = runner.requests[0].scope
    assert scope is not None
    assert (
        scope.source_albedo_path
        == (tmp_path / "materials" / "textures" / "Steel_A" / "albedo.png").resolve()
    )
    assert (
        scope.source_normal_path
        == (tmp_path / "materials" / "textures" / "Steel_A" / "normal.png").resolve()
    )
    assert (
        scope.source_orm_path
        == (tmp_path / "materials" / "textures" / "Steel_A" / "orm.png").resolve()
    )


def test_scope_resolution_infers_usduvtexture_channels_from_connections(
    tmp_path: Path,
) -> None:
    stage_path = _write_preview_surface_texture_stage(tmp_path)
    runner = _RecordingRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=True),
        runner=runner,
    )
    request = _copy_request(
        _request(),
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": "Paint",
            "material_path": "/World/Looks/Paint",
            "prim_paths": ["/World/Mesh"],
        },
    )

    backend.generate(
        request,
        job_id="vj-preview-scope",
        output_dir=tmp_path / "out",
        cancel_event=threading.Event(),
    )

    scope = runner.requests[0].scope
    assert scope is not None
    assert scope.source_albedo_path == (tmp_path / "paint_albedo.png").resolve()
    assert scope.source_normal_path == (tmp_path / "paint_normal.png").resolve()
    assert scope.source_orm_path == (tmp_path / "paint_orm.png").resolve()


def test_scope_resolution_extracts_texture_paths_from_usdz_package(
    tmp_path: Path,
) -> None:
    from pxr import UsdUtils

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    stage_path = _write_preview_surface_texture_stage(source_dir)
    package_path = tmp_path / "preview_surface_material.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(stage_path), str(package_path))

    runner = _RecordingRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=True),
        runner=runner,
    )
    request = _copy_request(
        _request(),
        source_asset_uri=package_path.as_uri(),
        target_updates={
            "material_name": "Paint",
            "material_path": "/World/Looks/Paint",
            "prim_paths": ["/World/Mesh"],
        },
    )

    backend.generate(
        request,
        job_id="vj-usdz-scope",
        output_dir=tmp_path / "out",
        cancel_event=threading.Event(),
    )

    scope = runner.requests[0].scope
    assert scope is not None
    assert scope.source_albedo_path.is_relative_to(tmp_path / "out")
    assert ".step1x_package_assets" in scope.source_albedo_path.parts
    assert scope.source_albedo_path.name == "paint_albedo.png"
    assert scope.source_albedo_path.exists()
    assert scope.source_normal_path is not None
    assert scope.source_normal_path.name == "paint_normal.png"
    assert scope.source_normal_path.exists()
    assert scope.source_orm_path is not None
    assert scope.source_orm_path.name == "paint_orm.png"
    assert scope.source_orm_path.exists()


def test_scope_resolution_does_not_treat_bare_file_input_as_albedo(
    tmp_path: Path,
) -> None:
    stage_path = _write_disconnected_file_texture_stage(tmp_path)
    runner = _RecordingRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=True),
        runner=runner,
    )
    request = _copy_request(
        _request(),
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": "Paint",
            "material_path": "/World/Looks/Paint",
            "prim_paths": ["/World/Mesh"],
        },
    )

    result = backend.generate(
        request,
        job_id="vj-bare-file-input",
        output_dir=tmp_path / "out",
        cancel_event=threading.Event(),
    )

    scope = runner.requests[0].scope
    assert scope is not None
    assert scope.source_albedo_path is not None
    assert scope.source_albedo_path.name == "source_albedo_Paint.png"
    assert scope.source_albedo_path != (tmp_path / "unknown.png").resolve()
    assert result.diagnostics[0]["code"] == "STEP1X_SOURCE_ALBEDO_SYNTHESIZED"


def test_scope_resolution_synthesizes_albedo_for_constant_color_material(
    tmp_path: Path,
) -> None:
    from PIL import Image

    stage_path = _write_constant_color_stage(tmp_path)
    runner = _RecordingRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=True),
        runner=runner,
    )
    request = _copy_request(
        _request(),
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": "Paint",
            "material_path": "/World/Looks/Paint",
            "prim_paths": ["/World/Mesh"],
        },
    )

    result = backend.generate(
        request,
        job_id="vj-constant-color",
        output_dir=tmp_path / "out",
        cancel_event=threading.Event(),
    )

    scope = runner.requests[0].scope
    assert scope is not None
    assert scope.source_albedo_path is not None
    assert scope.source_albedo_path.name == "source_albedo_Paint.png"
    assert scope.source_albedo_path.exists()
    image = Image.open(scope.source_albedo_path).convert("RGB")
    assert image.size == (512, 512)
    assert image.getpixel((0, 0))[2] > image.getpixel((0, 0))[0]
    assert result.metadata["scope"]["source_albedo_path"] == str(
        scope.source_albedo_path
    )
    assert result.diagnostics[0]["code"] == "STEP1X_SOURCE_ALBEDO_SYNTHESIZED"


def test_scope_resolution_synthesizes_default_albedo_for_nan_color_material(
    tmp_path: Path,
) -> None:
    from PIL import Image

    stage_path = _write_constant_color_stage(
        tmp_path,
        diffuse_color=(float("nan"), 0.1, 0.8),
    )
    runner = _RecordingRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=True),
        runner=runner,
    )
    request = _copy_request(
        _request(),
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": "Paint",
            "material_path": "/World/Looks/Paint",
            "prim_paths": ["/World/Mesh"],
        },
    )

    backend.generate(
        request,
        job_id="vj-nan-color",
        output_dir=tmp_path / "out",
        cancel_event=threading.Event(),
    )

    scope = runner.requests[0].scope
    assert scope is not None
    assert scope.source_albedo_path is not None
    image = Image.open(scope.source_albedo_path).convert("RGB")
    pixel = image.getpixel((0, 0))
    assert pixel == (188, 188, 188)


def test_scope_metadata_merges_runner_and_validation_fields(tmp_path: Path) -> None:
    class ScopeMetadataRunner(_RecordingRunner):
        def run(
            self,
            request: Step1XRunRequest,
            *,
            cancel_event: threading.Event,
        ) -> Step1XRunResult:
            self.requests.append(request)
            return Step1XRunResult(
                albedo_uri=(request.output_dir / "albedo.png").as_uri(),
                width=request.texture_size,
                height=request.texture_size,
                metadata={"scope": {"runner_scope": "kept"}},
            )

    stage_path = _write_constant_color_stage(tmp_path)
    runner = ScopeMetadataRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=True),
        runner=runner,
    )
    request = _copy_request(
        _request(),
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": "Paint",
            "material_path": "/World/Looks/Paint",
            "prim_paths": ["/World/Mesh"],
        },
    )

    result = backend.generate(
        request,
        job_id="vj-scope-merge",
        output_dir=tmp_path / "out",
        cancel_event=threading.Event(),
    )

    assert result.metadata["scope"]["runner_scope"] == "kept"
    assert result.metadata["scope"]["source_albedo_path"].endswith(
        "source_albedo_Paint.png",
    )


def test_package_member_extraction_rejects_oversized_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_path = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package_path, "w") as archive:
        archive.writestr("textures/huge.png", b"abcd")

    monkeypatch.setattr(step1x_backend_module, "_MAX_PACKAGE_ASSET_BYTES", 3)

    with pytest.raises(
        step1x_backend_module.Step1XExtractionError,
        match="STEP1X_PACKAGE_EXTRACTION_FAILED",
    ):
        step1x_backend_module._resolve_package_asset_path(
            f"{package_path}[textures/huge.png]",
            tmp_path,
            extraction_root=tmp_path / "out",
        )


def test_scope_resolution_rejects_unsupported_per_prim_mode(tmp_path: Path) -> None:
    stage_path = _write_two_material_stage(tmp_path)
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=True),
        runner=_RecordingRunner(),
    )
    request = _request()
    request = _copy_request(
        request,
        source_asset_uri=stage_path.as_uri(),
        target_updates={"mode": "per_prim"},
    )

    with pytest.raises(
        TextureGenerationBackendError,
        match="STEP1X_SCOPE_UNSUPPORTED",
    ) as exc_info:
        backend.generate(
            request,
            job_id="vj-scope",
            output_dir=tmp_path / "out",
            cancel_event=threading.Event(),
        )
    assert exc_info.value.result is not None
    assert exc_info.value.result.diagnostics[0]["code"] == "STEP1X_SCOPE_UNSUPPORTED"


def test_external_runner_invokes_configured_edit_script_and_parses_outputs(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "asset.usda"
    source_usd.write_text("#usda 1.0\n", encoding="utf-8")
    edit_script = _write_fake_step1x_script(tmp_path)
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            python_executable=Path(sys.executable),
            validate_assets=False,
        ),
        runner=ExternalStep1XRunner(
            Step1XBackendConfig(
                runtime_dir=tmp_path,
                edit_script=edit_script,
                python_executable=Path(sys.executable),
            )
        ),
    )
    request = _request()
    request = request.model_copy(update={"source_asset_uri": source_usd.as_uri()})

    result = backend.generate(
        request,
        job_id="vj-external",
        output_dir=tmp_path / "out",
        cancel_event=threading.Event(),
    )

    assert result.generated_textures.albedo.endswith("/final_albedo.png")
    assert result.maps["albedo"].width == 2
    assert result.maps["albedo"].height == 2
    assert result.variant_asset_uri.endswith("/edited_asset.usda")
    assert result.metadata["runner"] == "external_step1x_cli"
    assert "--skip-ma" in result.metadata["command"]


def test_external_runner_uses_scoped_usd_for_selected_material(tmp_path: Path) -> None:
    from pxr import Usd, UsdShade

    stage_path = _write_two_material_stage(tmp_path)
    edit_script = _write_fake_step1x_script(tmp_path)
    config = Step1XBackendConfig(
        runtime_dir=tmp_path,
        edit_script=edit_script,
        python_executable=Path(sys.executable),
        validate_assets=True,
    )
    backend = Step1XBackend(config=config, runner=ExternalStep1XRunner(config))
    request = _request()
    request = _copy_request(
        request,
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": "Blue",
            "material_path": "/World/Looks/Blue",
            "prim_paths": ["/World/MeshB"],
        },
    )

    result = backend.generate(
        request,
        job_id="vj-scoped-external",
        output_dir=tmp_path / "out",
        cancel_event=threading.Event(),
    )

    scoped_uri = result.metadata["scoped_source_asset_uri"]
    assert scoped_uri.endswith("/step1x_scoped_source.usda")
    assert result.metadata["scope"]["source_albedo_path"].endswith("blue_albedo.png")
    assert result.metadata["command_source_asset_uri"] == scoped_uri
    assert (
        (tmp_path / "out" / "input_usd.txt")
        .read_text(encoding="utf-8")
        .endswith("step1x_scoped_source.usda")
    )
    scoped_stage = Usd.Stage.Open(scoped_uri.removeprefix("file://"))
    scoped_shader = UsdShade.Shader.Get(
        scoped_stage,
        "/World/Looks/SelectedMaterial/Shader",
    )
    assert scoped_shader.GetIdAttr().Get() == "OmniPBR"


def test_command_template_source_asset_uri_uses_scoped_usd_after_validation(
    tmp_path: Path,
) -> None:
    stage_path = _write_two_material_stage(tmp_path)
    edit_script = _write_fake_step1x_script(tmp_path)
    config = Step1XBackendConfig(
        command_template=(
            f"{sys.executable} {edit_script} "
            '--usd {source_asset_uri} --prompt "{prompt}" --output {output_dir}'
        ),
        validate_assets=True,
    )
    backend = Step1XBackend(config=config, runner=ExternalStep1XRunner(config))
    request = _copy_request(
        _request(),
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": "Blue",
            "material_path": "/World/Looks/Blue",
            "prim_paths": ["/World/MeshB"],
        },
    )

    result = backend.generate(
        request,
        job_id="vj-template-scoped-external",
        output_dir=tmp_path / "out-template",
        cancel_event=threading.Event(),
    )

    scoped_uri = result.metadata["scoped_source_asset_uri"]
    assert result.metadata["command_source_asset_uri"] == scoped_uri
    assert (tmp_path / "out-template" / "input_usd.txt").read_text(
        encoding="utf-8",
    ) == scoped_uri
    assert scoped_uri != stage_path.as_uri()


def test_external_runner_authors_preview_surface_albedo_texture_network(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdShade

    stage_path = _write_preview_surface_texture_stage(tmp_path)
    edit_script = _write_fake_step1x_script(tmp_path)
    config = Step1XBackendConfig(
        runtime_dir=tmp_path,
        edit_script=edit_script,
        python_executable=Path(sys.executable),
        validate_assets=True,
    )
    backend = Step1XBackend(config=config, runner=ExternalStep1XRunner(config))
    request = _copy_request(
        _request(),
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": "Paint",
            "material_path": "/World/Looks/Paint",
            "prim_paths": ["/World/Mesh"],
        },
    )

    result = backend.generate(
        request,
        job_id="vj-preview-scoped-network",
        output_dir=tmp_path / "out-preview",
        cancel_event=threading.Event(),
    )

    scoped_path = result.metadata["scoped_source_asset_uri"].removeprefix("file://")
    scoped_stage = Usd.Stage.Open(scoped_path)
    scoped_shader = UsdShade.Shader.Get(
        scoped_stage,
        "/World/Looks/SelectedMaterial/Shader",
    )
    assert scoped_shader.GetIdAttr().Get() == "UsdPreviewSurface"
    diffuse_input = scoped_shader.GetInput("diffuseColor")
    source, source_name, _source_type = diffuse_input.GetConnectedSource()
    assert source_name == "rgb"
    albedo_texture = UsdShade.Shader(source.GetPrim())
    assert albedo_texture.GetIdAttr().Get() == "UsdUVTexture"
    assert albedo_texture.GetInput("file").Get().path == str(
        (tmp_path / "paint_albedo.png").resolve()
    )
    assert albedo_texture.GetInput("sourceColorSpace").Get() == "sRGB"
    st_input = albedo_texture.GetInput("st")
    uv_source, uv_source_name, _uv_source_type = st_input.GetConnectedSource()
    assert uv_source_name == "result"
    uv_reader = UsdShade.Shader(uv_source.GetPrim())
    assert uv_reader.GetIdAttr().Get() == "UsdPrimvarReader_float2"
    assert uv_reader.GetInput("varname").Get() == "st"


def test_external_runner_filters_parent_scope_to_selected_material(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    stage_path = _write_two_material_stage(tmp_path)
    edit_script = _write_fake_step1x_script(tmp_path)
    config = Step1XBackendConfig(
        runtime_dir=tmp_path,
        edit_script=edit_script,
        python_executable=Path(sys.executable),
        validate_assets=True,
    )
    backend = Step1XBackend(config=config, runner=ExternalStep1XRunner(config))
    request = _copy_request(
        _request(),
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": "Blue",
            "material_path": "/World/Looks/Blue",
            "prim_paths": ["/World"],
        },
    )

    result = backend.generate(
        request,
        job_id="vj-scoped-parent",
        output_dir=tmp_path / "out-parent",
        cancel_event=threading.Event(),
    )

    scoped_path = Path(
        result.metadata["scoped_source_asset_uri"].removeprefix("file://")
    )
    scoped_stage = Usd.Stage.Open(str(scoped_path))
    mesh_paths = [
        str(prim.GetPath())
        for prim in scoped_stage.Traverse()
        if prim.IsA(UsdGeom.Mesh)
    ]
    assert mesh_paths == ["/World/SelectedMesh"]


def test_external_runner_merges_multi_mesh_material_scope_for_step1x(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    stage_path = _write_multi_mesh_blue_stage(tmp_path)
    edit_script = _write_fake_step1x_script(tmp_path)
    config = Step1XBackendConfig(
        runtime_dir=tmp_path,
        edit_script=edit_script,
        python_executable=Path(sys.executable),
        validate_assets=True,
    )
    backend = Step1XBackend(config=config, runner=ExternalStep1XRunner(config))
    request = _copy_request(
        _request(),
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": "Blue",
            "material_path": "/World/Looks/Blue",
            "prim_paths": ["/World"],
        },
    )

    result = backend.generate(
        request,
        job_id="vj-scoped-merged",
        output_dir=tmp_path / "out-merged",
        cancel_event=threading.Event(),
    )

    scoped_path = Path(
        result.metadata["scoped_source_asset_uri"].removeprefix("file://")
    )
    scoped_stage = Usd.Stage.Open(str(scoped_path))
    meshes = [
        UsdGeom.Mesh(prim) for prim in scoped_stage.Traverse() if prim.IsA(UsdGeom.Mesh)
    ]
    assert len(meshes) == 1
    mesh = meshes[0]
    assert str(mesh.GetPath()) == "/World/SelectedMesh"
    assert len(mesh.GetPointsAttr().Get()) == 6
    assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3, 3]
    assert list(mesh.GetFaceVertexIndicesAttr().Get()) == [0, 1, 2, 3, 4, 5]
    points = list(mesh.GetPointsAttr().Get())
    for axis in range(3):
        bounds_midpoint = (
            min(float(point[axis]) for point in points)
            + max(float(point[axis]) for point in points)
        ) * 0.5
        assert bounds_midpoint == pytest.approx(
            0.0,
            abs=1e-6,
        )
    assert max(point[0] for point in points) - min(
        point[0] for point in points
    ) == pytest.approx(3.0)
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("st")
    assert st.GetInterpolation() == UsdGeom.Tokens.faceVarying
    assert len(st.Get()) == 6


def test_scope_resolution_rejects_non_finite_uv_before_runner_launch(
    tmp_path: Path,
) -> None:
    stage_path = _write_non_finite_uv_stage(tmp_path)
    runner = _RecordingRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=True),
        runner=runner,
    )
    request = _copy_request(
        _request(),
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": "Blue",
            "material_path": "/World/Looks/Blue",
            "prim_paths": ["/World/MeshB"],
        },
    )

    with pytest.raises(
        TextureGenerationBackendError,
        match="STEP1X_UV_INVALID",
    ) as exc_info:
        backend.generate(
            request,
            job_id="vj-invalid-uv",
            output_dir=tmp_path / "out-invalid-uv",
            cancel_event=threading.Event(),
        )

    assert runner.requests == []
    assert exc_info.value.result is not None
    assert exc_info.value.result.diagnostics[0]["code"] == "STEP1X_UV_INVALID"


def test_external_runner_rejects_blank_albedo(tmp_path: Path) -> None:
    source_usd = tmp_path / "asset.usda"
    source_usd.write_text("#usda 1.0\n", encoding="utf-8")
    edit_script = _write_fake_step1x_script(tmp_path, blank=True)
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            python_executable=Path(sys.executable),
            validate_assets=False,
        )
    )
    request = _request()
    request = request.model_copy(update={"source_asset_uri": source_usd.as_uri()})

    with pytest.raises(
        TextureGenerationBackendError,
        match="STEP1X_OUTPUT_BLANK",
    ) as exc_info:
        backend.generate(
            request,
            job_id="vj-external",
            output_dir=tmp_path / "out",
            cancel_event=threading.Event(),
        )
    assert exc_info.value.result is not None
    assert exc_info.value.result.diagnostics[0]["code"] == "STEP1X_OUTPUT_BLANK"


def test_external_runner_command_failure_returns_structured_diagnostic(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "asset.usda"
    source_usd.write_text("#usda 1.0\n", encoding="utf-8")
    edit_script = tmp_path / "edit_texture.py"
    edit_script.write_text(
        "from __future__ import annotations\n"
        "import sys\n"
        "print('boom', file=sys.stderr)\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=edit_script,
            python_executable=Path(sys.executable),
            validate_assets=False,
        )
    )
    request = _request().model_copy(update={"source_asset_uri": source_usd.as_uri()})

    with pytest.raises(
        TextureGenerationBackendError,
        match="STEP1X_COMMAND_FAILED",
    ) as exc_info:
        backend.generate(
            request,
            job_id="vj-external-fail",
            output_dir=tmp_path / "out",
            cancel_event=threading.Event(),
        )
    assert exc_info.value.result is not None
    assert exc_info.value.result.diagnostics[0]["code"] == "STEP1X_COMMAND_FAILED"


def test_command_template_runs_without_runtime_dir(tmp_path: Path) -> None:
    source_usd = tmp_path / "asset.usda"
    source_usd.write_text("#usda 1.0\n", encoding="utf-8")
    edit_script = _write_fake_step1x_script(tmp_path)
    config = Step1XBackendConfig(
        command_template=(
            f"{sys.executable} {edit_script} "
            '--usd {source_asset} --prompt "{prompt}" --output {output_dir}'
        ),
        validate_assets=False,
    )
    backend = Step1XBackend(config=config, runner=ExternalStep1XRunner(config))
    request = _request().model_copy(update={"source_asset_uri": source_usd.as_uri()})

    result = backend.generate(
        request,
        job_id="vj-template-runtime-free",
        output_dir=tmp_path / "out",
        cancel_event=threading.Event(),
    )

    assert result.generated_textures.albedo.endswith("/final_albedo.png")


def test_redact_command_handles_hyphenated_and_inline_secret_flags() -> None:
    assert _redact_command(
        [
            "python",
            "edit.py",
            "--api-key",
            "sk-secret",
            "--auth-token=tok-secret",
            "--password",
            "pw-secret",
            "--prompt",
            "visible prompt",
        ]
    ) == [
        "python",
        "edit.py",
        "<redacted>",
        "<redacted>",
        "<redacted>",
        "<redacted>",
        "<redacted>",
        "--prompt",
        "visible prompt",
    ]


def test_redact_command_does_not_hide_plain_prompt_words() -> None:
    assert _redact_command(
        [
            "python",
            "edit.py",
            "--prompt",
            "secret garden token of appreciation",
            "--note=credentialed material",
        ]
    ) == [
        "python",
        "edit.py",
        "--prompt",
        "secret garden token of appreciation",
        "--note=credentialed material",
    ]


def test_command_template_preserves_seed_zero_and_tolerates_unknown_keys(
    tmp_path: Path,
) -> None:
    runner = ExternalStep1XRunner(
        Step1XBackendConfig(
            command_template="runner --seed {seed} --missing {unknown_key}",
        )
    )
    request = Step1XRunRequest(
        prompt="test",
        seed=0,
        strength=0.5,
        texture_size=128,
        source_asset_uri=(tmp_path / "asset.usd").as_uri(),
        source_asset_path=tmp_path / "asset.usd",
        source_albedo_path=None,
        reference_image_uris=(),
        turntable_video_uri=None,
        multiview_image_uris=(),
        target=None,
        scope=None,
        job_id="vj-template",
        output_dir=tmp_path / "out",
        runtime_dir=None,
        model_dir=None,
        cache_dir=None,
        custom_parameters={},
    )

    assert runner._build_command(request, tmp_path / "asset.usd") == [
        "runner",
        "--seed",
        "0",
        "--missing",
    ]


def test_command_template_source_asset_uri_uses_scoped_command_asset(
    tmp_path: Path,
) -> None:
    runner = ExternalStep1XRunner(
        Step1XBackendConfig(
            command_template="runner --usd-uri {source_asset_uri} --usd {source_asset}",
        )
    )
    original_asset = tmp_path / "original.usd"
    scoped_asset = tmp_path / "step1x_scoped_source.usda"
    original_asset.write_text("#usda 1.0\n", encoding="utf-8")
    scoped_asset.write_text("#usda 1.0\n", encoding="utf-8")
    request = Step1XRunRequest(
        prompt="test",
        seed=0,
        strength=0.5,
        texture_size=128,
        source_asset_uri=original_asset.as_uri(),
        source_asset_path=original_asset,
        source_albedo_path=None,
        reference_image_uris=(),
        turntable_video_uri=None,
        multiview_image_uris=(),
        target=None,
        scope=Step1XScopeInfo(
            source_asset_path=original_asset,
            material_path="/World/Looks/Steel",
            material_name="Steel",
            prim_paths=("/World/Mesh",),
        ),
        job_id="vj-template-scoped-uri",
        output_dir=tmp_path / "out",
        runtime_dir=None,
        model_dir=None,
        cache_dir=None,
        custom_parameters={},
    )

    assert runner._build_command(request, scoped_asset) == [
        "runner",
        "--usd-uri",
        scoped_asset.as_uri(),
        "--usd",
        str(scoped_asset),
    ]


def test_command_template_substitution_keeps_prompt_in_single_argv_token(
    tmp_path: Path,
) -> None:
    runner = ExternalStep1XRunner(
        Step1XBackendConfig(
            command_template="runner --prompt {prompt} --usd {source_asset}",
        )
    )
    source_asset = tmp_path / "asset.usd"
    request = Step1XRunRequest(
        prompt='aged "red leather" --output /tmp/injected with spaces',
        seed=123,
        strength=0.5,
        texture_size=128,
        source_asset_uri=source_asset.as_uri(),
        source_asset_path=source_asset,
        source_albedo_path=None,
        reference_image_uris=(),
        turntable_video_uri=None,
        multiview_image_uris=(),
        target=None,
        scope=None,
        job_id="vj-template-injection",
        output_dir=tmp_path / "out",
        runtime_dir=None,
        model_dir=None,
        cache_dir=None,
        custom_parameters={},
    )

    command = runner._build_command(request, source_asset)

    assert command == [
        "runner",
        "--prompt",
        'aged "red leather" --output /tmp/injected with spaces',
        "--usd",
        str(source_asset),
    ]


def test_default_command_can_enable_material_anything_and_upscale(
    tmp_path: Path,
) -> None:
    runner = ExternalStep1XRunner(
        Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=tmp_path / "edit_texture.py",
            python_executable=Path(sys.executable),
            skip_material_anything=True,
        )
    )
    request = _step1x_run_request(
        tmp_path,
        custom_parameters={
            "skip_material_anything": False,
            "ma_steps": 10,
            "gpu": 1,
            "upscale": True,
        },
    )

    command = runner._build_command(request, tmp_path / "asset.usd")

    assert "--skip-ma" not in command
    assert command[command.index("--ma-steps") + 1] == "10"
    assert command[command.index("--gpu") + 1] == "1"
    assert "--upscale" in command


def test_default_command_parses_string_false_for_material_anything(
    tmp_path: Path,
) -> None:
    runner = ExternalStep1XRunner(
        Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=tmp_path / "edit_texture.py",
            python_executable=Path(sys.executable),
            skip_material_anything=True,
        )
    )
    request = _step1x_run_request(
        tmp_path,
        custom_parameters={
            "skip_material_anything": "false",
            "upscale": "false",
        },
    )

    command = runner._build_command(request, tmp_path / "asset.usd")

    assert "--skip-ma" not in command
    assert "--upscale" not in command


def test_default_command_can_force_skip_ma_when_global_default_enables_it(
    tmp_path: Path,
) -> None:
    runner = ExternalStep1XRunner(
        Step1XBackendConfig(
            runtime_dir=tmp_path,
            edit_script=tmp_path / "edit_texture.py",
            python_executable=Path(sys.executable),
            skip_material_anything=False,
        )
    )
    request = _step1x_run_request(
        tmp_path,
        custom_parameters={"skip_material_anything": True},
    )

    command = runner._build_command(request, tmp_path / "asset.usd")

    assert "--skip-ma" in command


def test_external_runner_prefers_venv_gen_python(tmp_path: Path) -> None:
    venv_gen_python = tmp_path / ".venv_gen" / "bin" / "python"
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_gen_python.parent.mkdir(parents=True)
    venv_python.parent.mkdir(parents=True)
    venv_gen_python.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    runner = ExternalStep1XRunner(Step1XBackendConfig(runtime_dir=tmp_path))

    assert runner._python_executable() == venv_gen_python


def test_step1x_runner_protocol_placeholder_is_executable(tmp_path: Path) -> None:
    assert (
        step1x_backend_module.Step1XRunner.run(
            object(),
            _step1x_run_request(tmp_path),
            cancel_event=threading.Event(),
        )
        is None
    )


def test_external_runner_terminates_on_cancel_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes: list[object] = []
    cancel_event = threading.Event()

    class CancelProcess:
        returncode = None

        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def wait(self, timeout: float) -> int:
            if self.terminated:
                self.returncode = -15
                return self.returncode
            cancel_event.set()
            raise step1x_backend_module.subprocess.TimeoutExpired("runner", timeout)

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    def fake_cancel_popen(*_args: object, **_kwargs: object) -> CancelProcess:
        process = CancelProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(
        step1x_backend_module.subprocess,
        "Popen",
        fake_cancel_popen,
    )
    runner = ExternalStep1XRunner(
        Step1XBackendConfig(
            command_template="runner --usd {source_asset}",
            timeout_sec=30,
        )
    )

    with pytest.raises(RuntimeError, match="cancelled while running"):
        runner.run(_step1x_run_request(tmp_path), cancel_event=cancel_event)
    assert isinstance(processes[0], CancelProcess)
    assert processes[0].terminated is True

    class TimeoutProcess:
        returncode = None

        def __init__(self) -> None:
            self.terminated = False

        def wait(self, timeout: float) -> int:
            assert timeout == 10
            self.returncode = -15
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            raise AssertionError("timeout process should terminate cleanly")

    timeout_process = TimeoutProcess()
    monkeypatch.setattr(
        step1x_backend_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: timeout_process,
    )
    timeout_runner = ExternalStep1XRunner(
        Step1XBackendConfig(
            command_template="runner --usd {source_asset}",
            timeout_sec=0,
        )
    )

    with pytest.raises(RuntimeError, match="STEP1X_TIMEOUT"):
        timeout_runner.run(
            _step1x_run_request(tmp_path),
            cancel_event=threading.Event(),
        )
    assert timeout_process.terminated is True


def test_external_runner_path_helpers_cover_fallbacks(tmp_path: Path) -> None:
    edit_script = tmp_path / "scripts" / "edit_texture.py"
    edit_script.parent.mkdir()
    edit_script.write_text("print('edit')\n", encoding="utf-8")
    edit_runner = ExternalStep1XRunner(Step1XBackendConfig(edit_script=edit_script))

    assert edit_runner._working_dir() == edit_script.parent
    assert edit_runner._edit_script() == edit_script

    with pytest.raises(RuntimeError, match="TEXTURE_STEP1X_RUNTIME_DIR"):
        ExternalStep1XRunner(Step1XBackendConfig())._edit_script()

    runtime_dir = tmp_path / "runtime"
    venv_python = runtime_dir / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
    runtime_runner = ExternalStep1XRunner(Step1XBackendConfig(runtime_dir=runtime_dir))

    assert runtime_runner._python_executable() == venv_python
    assert runtime_runner._edit_script() == runtime_dir / "edit_texture.py"
    assert ExternalStep1XRunner(
        Step1XBackendConfig(runtime_dir=tmp_path / "empty")
    )._python_executable() == Path(sys.executable)


def test_env_and_readiness_helpers_cover_edge_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WU_PATH", str(tmp_path))
    monkeypatch.setenv("WU_STR", " configured ")
    monkeypatch.setenv("WU_INT", "7")
    monkeypatch.setenv("WU_BOOL", "yes")
    monkeypatch.setenv("WU_EXECUTABLES", "uv, python")

    assert step1x_backend_module._optional_path("WU_PATH") == tmp_path
    assert step1x_backend_module._optional_str("WU_STR") == "configured"
    assert step1x_backend_module._optional_int("WU_INT", default=3) == 7
    assert step1x_backend_module._optional_bool("WU_BOOL", False) is True
    assert step1x_backend_module._optional_executables(
        "WU_EXECUTABLES",
        default=(),
    ) == ("uv", "python")

    monkeypatch.setenv("WU_INT", "bad")
    monkeypatch.setenv("WU_EXECUTABLES", " ")
    monkeypatch.setenv("TEXTURE_STEP1X_RUNTIME_MODULE_PROBE_TIMEOUT_SEC", "bad")
    monkeypatch.setenv("TEXTURE_STEP1X_PREFLIGHT_TIMEOUT_SEC", "12.5")
    assert step1x_backend_module._optional_int("WU_INT", default=3) == 3
    assert (
        step1x_backend_module._optional_executables(
            "WU_EXECUTABLES",
            default=("uv",),
        )
        == ()
    )
    assert step1x_backend_module._required_executable_status((" ",)) == []
    assert step1x_backend_module._runtime_module_probe_timeout_sec() == 12.5
    assert step1x_backend_module._coerce_bool(1, default=False) is True
    assert step1x_backend_module._coerce_bool(" on ", default=False) is True
    assert step1x_backend_module._coerce_bool("maybe", default=False) is True
    monkeypatch.setattr(
        step1x_backend_module,
        "_BUNDLED_RUNTIME_RELATIVE",
        Path("missing-runtime-for-coverage"),
    )
    assert step1x_backend_module._bundled_runtime_dir() is None

    monkeypatch.setattr(
        step1x_backend_module,
        "_GPU_CACHE_AT",
        -step1x_backend_module._GPU_CACHE_TTL_SEC - 1.0,
    )
    monkeypatch.setattr(step1x_backend_module, "_GPU_CACHE_VALUE", True)

    def missing_nvidia_smi(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(step1x_backend_module.subprocess, "run", missing_nvidia_smi)
    assert step1x_backend_module._detect_gpu_available() is None

    monkeypatch.setattr(
        step1x_backend_module,
        "_GPU_CACHE_AT",
        -step1x_backend_module._GPU_CACHE_TTL_SEC - 1.0,
    )
    monkeypatch.setattr(step1x_backend_module, "_GPU_CACHE_VALUE", None)

    def detected_nvidia_smi(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(returncode=0, stdout="GPU 0: test\n")

    monkeypatch.setattr(step1x_backend_module.subprocess, "run", detected_nvidia_smi)
    assert step1x_backend_module._detect_gpu_available() is True


def test_missing_runtime_inputs_cover_optional_path_appends(
    tmp_path: Path,
) -> None:
    command_template_backend = Step1XBackend(
        config=Step1XBackendConfig(
            command_template="runner --usd {source_asset}",
            runtime_dir=tmp_path / "missing-runtime",
            model_dir=tmp_path / "missing-models",
            required_executables=(),
        )
    )

    command_template_missing = command_template_backend._missing_runtime_inputs()

    assert any(
        "TEXTURE_STEP1X_RUNTIME_DIR" in item for item in command_template_missing
    )
    assert any("TEXTURE_STEP1X_MODEL_DIR" in item for item in command_template_missing)

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    edit_script = runtime_dir / "edit_texture.py"
    edit_script.write_text("print('edit')\n", encoding="utf-8")
    default_backend = Step1XBackend(
        config=Step1XBackendConfig(
            runtime_dir=runtime_dir,
            model_dir=tmp_path / "missing-models",
            python_executable=tmp_path / "missing-python",
            validate_assets=False,
            required_executables=(),
        )
    )

    default_missing = default_backend._missing_runtime_inputs()

    assert any("TEXTURE_STEP1X_MODEL_DIR" in item for item in default_missing)
    assert any("TEXTURE_STEP1X_PYTHON" in item for item in default_missing)


def test_runtime_source_ignores_resolve_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    bundled_dir = tmp_path / "bundled"
    monkeypatch.setattr(
        step1x_backend_module,
        "_bundled_runtime_dir",
        lambda: bundled_dir,
    )
    original_resolve = Path.resolve

    def raising_resolve(path: Path, *args: object, **kwargs: object) -> Path:
        if path == runtime_dir:
            raise OSError("resolve blocked")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", raising_resolve)
    backend = Step1XBackend(config=Step1XBackendConfig(runtime_dir=runtime_dir))

    assert backend._runtime_source() == "operator_mounted"


def test_path_readiness_reports_directory_and_access_issues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")
    assert "not a directory" in step1x_backend_module._path_readiness_issue(
        "TARGET",
        target,
        require_directory=True,
    )

    original_is_dir = Path.is_dir

    def raising_is_dir(path: Path) -> bool:
        if path == target:
            raise PermissionError(13, "Permission denied", str(path))
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", raising_is_dir)
    assert "not accessible" in step1x_backend_module._path_readiness_issue(
        "TARGET",
        target,
        require_directory=True,
    )

    monkeypatch.setattr(Path, "is_dir", original_is_dir)
    monkeypatch.setattr(step1x_backend_module.os, "access", lambda *_args: False)
    assert "not readable/executable" in step1x_backend_module._path_readiness_issue(
        "TARGET",
        target,
        require_read=True,
        require_execute=True,
    )


def test_runtime_probe_helpers_report_failure_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_path_issue = step1x_backend_module._path_readiness_issue
    monkeypatch.setattr(
        step1x_backend_module,
        "_path_readiness_issue",
        lambda *_args, **_kwargs: "TEXTURE_STEP1X_PYTHON",
    )
    assert step1x_backend_module._missing_python_modules_in_runtime(
        tmp_path / "python",
        ("torch",),
    ) == ["TEXTURE_STEP1X_PYTHON"]
    monkeypatch.setattr(
        step1x_backend_module,
        "_path_readiness_issue",
        original_path_issue,
    )

    python_executable = tmp_path / "python"
    python_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    python_executable.chmod(0o755)

    monkeypatch.setattr(
        step1x_backend_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="bad import",
        ),
    )
    assert step1x_backend_module._missing_python_modules_in_runtime(
        python_executable,
        ("torch",),
    ) == ["runtime python module probe failed (bad import)"]

    monkeypatch.setattr(
        step1x_backend_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=3,
            stdout="",
            stderr="",
        ),
    )
    assert step1x_backend_module._missing_python_modules_in_runtime(
        python_executable,
        ("torch",),
    ) == ["runtime python module probe failed (exit 3)"]

    monkeypatch.setattr(
        step1x_backend_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="banner only\n",
            stderr="",
        ),
    )
    assert step1x_backend_module._missing_python_modules_in_runtime(
        python_executable,
        ("torch",),
    ) == ["runtime python module probe failed (missing module probe output)"]

    monkeypatch.setattr(
        step1x_backend_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="WU_MISSING_MODULES_JSON=42\n",
            stderr="",
        ),
    )
    assert step1x_backend_module._missing_python_modules_in_runtime(
        python_executable,
        ("torch",),
    ) == ["runtime python module probe failed (unexpected output)"]


def test_upscaler_readiness_helpers_cover_auto_and_download_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "runtime"
    module_path = runtime_dir / "src" / "texture_edit" / "upscaler.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# upscaler\n", encoding="utf-8")

    monkeypatch.setenv("TEXTURE_UPSCALER_BACKEND", "auto")
    monkeypatch.setattr(
        step1x_backend_module,
        "_missing_swin2sr_upscaler_inputs",
        lambda python_executable=None: ["python module torch (not importable)"],
    )
    monkeypatch.setattr(
        step1x_backend_module,
        "_missing_ncnn_upscaler_inputs",
        lambda paths: ["ncnn_binary (not found)"],
    )
    assert step1x_backend_module._missing_upscaler_inputs(runtime_dir) == [
        "swin2sr python module torch (not importable)",
        "ncnn-vulkan ncnn_binary (not found)",
    ]

    monkeypatch.delenv("TEXTURE_UPSCALER_BACKEND", raising=False)
    monkeypatch.setenv("TEXTURE_REALESRGAN_BACKEND", "vulkan")
    assert step1x_backend_module._upscaler_backend() == "ncnn-vulkan"

    monkeypatch.setenv("TEXTURE_UPSCALER_BACKEND", "ncnn-vulkan")
    original_path_issue = step1x_backend_module._path_readiness_issue
    monkeypatch.setattr(
        step1x_backend_module,
        "_path_readiness_issue",
        lambda *_args, **_kwargs: "module (not readable)",
    )
    assert step1x_backend_module._upscaler_auto_download_writable(runtime_dir) is False
    monkeypatch.setattr(
        step1x_backend_module,
        "_path_readiness_issue",
        original_path_issue,
    )

    original_exists = Path.exists
    bin_dir = runtime_dir / "bin"

    def raising_exists(path: Path) -> bool:
        if path == bin_dir:
            raise OSError("blocked")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", raising_exists)
    assert step1x_backend_module._upscaler_auto_download_writable(runtime_dir) is False

    monkeypatch.setattr(Path, "exists", original_exists)
    bin_dir.mkdir()
    monkeypatch.setattr(
        step1x_backend_module.os,
        "access",
        lambda path, mode: path == module_path
        and mode == step1x_backend_module.os.R_OK,
    )
    assert step1x_backend_module._upscaler_auto_download_writable(runtime_dir) is False


def test_step1x_error_and_uri_helpers_cover_negative_cases() -> None:
    assert step1x_backend_module._step1x_error_code("plain failure") is None
    assert (
        step1x_backend_module._local_path_from_uri("https://example.test/a.usd") is None
    )


def test_scope_resolution_infers_single_bound_material_from_target_prims(
    tmp_path: Path,
) -> None:
    stage_path = _write_two_material_stage(tmp_path)
    runner = _RecordingRunner()
    backend = Step1XBackend(
        config=Step1XBackendConfig(validate_assets=True),
        runner=runner,
    )
    request = _copy_request(
        _request(),
        source_asset_uri=stage_path.as_uri(),
        target_updates={
            "material_name": None,
            "material_path": None,
            "prim_paths": ["/World/MeshB"],
        },
    )

    backend.generate(
        request,
        job_id="vj-bound-material",
        output_dir=tmp_path / "out",
        cancel_event=threading.Event(),
    )

    assert runner.requests[0].scope is not None
    assert runner.requests[0].scope.material_path == "/World/Looks/Blue"


def test_scope_resolution_without_output_dir_reports_missing_albedo(
    tmp_path: Path,
) -> None:
    stage_path = _write_constant_color_stage(tmp_path)
    target = _request().target
    assert target is not None
    target = target.model_copy(
        update={
            "material_name": "Paint",
            "material_path": "/World/Looks/Paint",
            "prim_paths": ["/World/Mesh"],
        }
    )

    with pytest.raises(RuntimeError, match="STEP1X_TEXTURE_MISSING"):
        step1x_backend_module._inspect_step1x_scope(
            stage_path.as_uri(),
            target,
            output_dir=None,
        )


def test_scope_helpers_cover_material_and_prim_edge_cases(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdShade

    one_material_stage = Usd.Stage.Open(str(_write_constant_color_stage(tmp_path)))
    assert (
        step1x_backend_module._resolve_target_material_path(
            one_material_stage,
            None,
        )
        == "/World/Looks/Paint"
    )

    two_material_stage = Usd.Stage.Open(str(_write_two_material_stage(tmp_path)))
    target = _request().target
    assert target is not None
    assert (
        step1x_backend_module._resolve_target_material_path(
            two_material_stage,
            target.model_copy(
                update={"material_name": "Blue", "material_path": None},
            ),
        )
        == "/World/Looks/Blue"
    )
    assert (
        step1x_backend_module._resolve_target_material_path(
            two_material_stage,
            target.model_copy(
                update={"material_name": "Missing", "material_path": None},
            ),
        )
        is None
    )

    ambiguous_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Scope.Define(ambiguous_stage, "/World")
    UsdGeom.Scope.Define(ambiguous_stage, "/World/A")
    UsdGeom.Scope.Define(ambiguous_stage, "/World/B")
    UsdShade.Material.Define(ambiguous_stage, "/World/A/Blue")
    UsdShade.Material.Define(ambiguous_stage, "/World/B/Blue")
    with pytest.raises(RuntimeError, match="material_name is ambiguous"):
        step1x_backend_module._resolve_target_material_path(
            ambiguous_stage,
            target.model_copy(
                update={"material_name": "Blue", "material_path": None},
            ),
        )

    assert (
        step1x_backend_module._resolve_target_prim_paths(
            two_material_stage,
            None,
        )
        == []
    )
    assert (
        step1x_backend_module._resolve_target_prim_paths(
            two_material_stage,
            target.model_copy(
                update={"prim_paths": ["/Missing"], "strict_scope": False}
            ),
        )
        == []
    )
    assert (
        step1x_backend_module._bound_material_paths(
            two_material_stage,
            ["/Missing"],
        )
        == set()
    )


def test_source_shader_and_mesh_helpers_cover_empty_cases(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(_write_two_material_stage(tmp_path)))
    assert step1x_backend_module._source_material_surface_shader_id(stage, None) is None
    assert (
        step1x_backend_module._source_material_surface_shader_id(stage, "/World/MeshB")
        is None
    )

    class FakeConnection:
        def GetPrimPath(self) -> str:  # noqa: N802
            return "/NotAShader"

    class FakeAttr:
        def GetName(self) -> str:  # noqa: N802
            return "outputs:surface"

        def GetConnections(self) -> list[FakeConnection]:  # noqa: N802
            return [FakeConnection()]

    class FakePrim:
        def __init__(self, *, material: bool, shader: bool = False) -> None:
            self.material = material
            self.shader = shader

        def IsA(self, cls: object) -> bool:  # noqa: N802
            name = getattr(cls, "__name__", "")
            return (name == "Material" and self.material) or (
                name == "Shader" and self.shader
            )

        def GetAttributes(self) -> list[FakeAttr]:  # noqa: N802
            return [FakeAttr()]

    class FakeStage:
        def GetPrimAtPath(self, path: object) -> FakePrim:  # noqa: N802
            return FakePrim(material=path == "/Material")

    assert (
        step1x_backend_module._source_material_surface_shader_id(
            FakeStage(),
            "/Material",
        )
        is None
    )

    shaderless_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Scope.Define(shaderless_stage, "/World")
    material = UsdShade.Material.Define(shaderless_stage, "/World/Looks/Paint")
    shader = UsdShade.Shader.Define(shaderless_stage, "/World/Looks/Paint/Shader")
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )
    assert (
        step1x_backend_module._source_material_surface_shader_id(
            shaderless_stage,
            "/World/Looks/Paint",
        )
        is None
    )

    scope = Step1XScopeInfo(
        source_asset_path=tmp_path / "asset.usd",
        material_path="/World/Looks/Blue",
    )
    assert [
        str(prim.GetPath())
        for prim in step1x_backend_module._target_mesh_prims(stage, scope)
    ] == ["/World/MeshB"]

    unbound_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(unbound_stage, "/World")
    UsdGeom.Mesh.Define(unbound_stage, "/World/Loose")
    assert (
        step1x_backend_module._target_mesh_prims(
            unbound_stage,
            Step1XScopeInfo(
                source_asset_path=tmp_path / "asset.usd",
                prim_paths=("/Missing", "/World/Loose"),
            ),
        )
        == []
    )
    assert (
        step1x_backend_module._target_mesh_prims(
            unbound_stage,
            Step1XScopeInfo(
                source_asset_path=tmp_path / "asset.usd",
                material_path="/World/Looks/Blue",
                prim_paths=("/World/Loose",),
            ),
        )
        == []
    )

    merge_stage = Usd.Stage.CreateInMemory()
    empty = UsdGeom.Mesh.Define(merge_stage, "/World/Empty")
    valid = UsdGeom.Mesh.Define(merge_stage, "/World/Valid")
    valid.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    valid.CreateFaceVertexCountsAttr([3])
    valid.CreateFaceVertexIndicesAttr([0, 1, 2])
    dest = UsdGeom.Mesh.Define(merge_stage, "/World/Dest")

    empty_dest = UsdGeom.Mesh.Define(merge_stage, "/World/EmptyDest")
    with pytest.raises(RuntimeError, match="STEP1X_SCOPE_INVALID"):
        step1x_backend_module._copy_merged_meshes_for_step1x(
            [empty],
            empty_dest,
            UsdGeom,
        )

    step1x_backend_module._copy_merged_meshes_for_step1x(
        [empty, valid],
        dest,
        UsdGeom,
    )

    assert len(dest.GetPointsAttr().Get()) == 3

    for label, coordinate in (
        ("NaN", float("nan")),
        ("PositiveInfinity", float("inf")),
        ("NegativeInfinity", float("-inf")),
    ):
        nonfinite = UsdGeom.Mesh.Define(merge_stage, f"/World/{label}")
        nonfinite.CreatePointsAttr([(coordinate, 0, 0), (1, 0, 0), (0, 1, 0)])
        nonfinite.CreateFaceVertexCountsAttr([3])
        nonfinite.CreateFaceVertexIndicesAttr([0, 1, 2])
        nonfinite_dest = UsdGeom.Mesh.Define(
            merge_stage,
            f"/World/{label}Dest",
        )

        with pytest.raises(RuntimeError, match="STEP1X_GEOMETRY_INVALID"):
            step1x_backend_module._copy_merged_meshes_for_step1x(
                [nonfinite],
                nonfinite_dest,
                UsdGeom,
            )


def test_finite_bounds_midpoint_handles_extremes_and_subnormals() -> None:
    midpoint = step1x_backend_module._finite_bounds_midpoint
    maximum = sys.float_info.max
    subnormal = math.ulp(0.0)

    assert midpoint(maximum, maximum) == maximum
    assert midpoint(-maximum, -maximum) == -maximum
    assert midpoint(-maximum, maximum) == 0.0
    assert midpoint(subnormal, maximum) == maximum * 0.5
    assert midpoint(-maximum, -subnormal) == -maximum * 0.5
    assert midpoint(subnormal, subnormal) == subnormal
    assert midpoint(-subnormal, -subnormal) == -subnormal
    assert midpoint(subnormal, subnormal * 2.0) == subnormal * 2.0
    assert midpoint(-subnormal * 2.0, -subnormal) == -subnormal * 2.0
    with pytest.raises(ValueError, match="must be finite"):
        midpoint(float("-inf"), float("inf"))


def test_uv_resolution_helper_edges() -> None:
    class Primvar:
        def __init__(
            self,
            values: list[object],
            *,
            indices: list[int] | None = None,
            interpolation: str = "vertex",
        ) -> None:
            self.values = values
            self.indices = indices or []
            self.interpolation = interpolation

        def HasValue(self) -> bool:  # noqa: N802
            return True

        def Get(self) -> list[object]:  # noqa: N802
            return self.values

        def GetIndices(self) -> list[int]:  # noqa: N802
            return self.indices

        def GetInterpolation(self) -> str:  # noqa: N802
            return self.interpolation

    assert (
        step1x_backend_module._resolve_face_varying_st_values(
            Primvar([]),
            point_count=3,
            face_vertex_indices=[0, 1, 2],
        )
        is None
    )
    with pytest.raises(RuntimeError, match="indexed st primvar length"):
        step1x_backend_module._resolve_face_varying_st_values(
            Primvar([(0, 0)], indices=[0, 1]),
            point_count=3,
            face_vertex_indices=[0, 1, 2],
        )
    with pytest.raises(RuntimeError, match="references missing values"):
        step1x_backend_module._resolve_face_varying_st_values(
            Primvar([(0, 0)], indices=[0, 2, -1]),
            point_count=3,
            face_vertex_indices=[0, 1, 2],
        )
    assert step1x_backend_module._resolve_face_varying_st_values(
        Primvar([(0, 0), (1, 0)], indices=[1, 0, 1]),
        point_count=3,
        face_vertex_indices=[0, 1, 2],
    ) == [(1, 0), (0, 0), (1, 0)]
    assert step1x_backend_module._resolve_face_varying_st_values(
        Primvar([(0, 0), (1, 0), (0, 1)]),
        point_count=3,
        face_vertex_indices=[2, 1, 0, 2],
    ) == [(0, 1), (1, 0), (0, 0), (0, 1)]
    with pytest.raises(RuntimeError, match="cannot be converted"):
        step1x_backend_module._resolve_face_varying_st_values(
            Primvar([(0, 0), (1, 0)], interpolation="constant"),
            point_count=3,
            face_vertex_indices=[0, 1, 2],
        )


def test_texture_path_and_color_helpers_cover_edge_cases(tmp_path: Path) -> None:
    assert step1x_backend_module._coerce_color3(None) is None
    assert step1x_backend_module._coerce_color3(["bad"]) is None
    assert step1x_backend_module._linear_channel_to_srgb_byte("bad") == 188
    assert step1x_backend_module._linear_channel_to_srgb_byte(float("nan")) == 188
    assert step1x_backend_module._linear_channel_to_srgb_byte(0.001) == 3

    missing_mdl = tmp_path / "missing.mdl"
    assert step1x_backend_module._find_mdl_texture_paths(missing_mdl) == {}
    mdl = tmp_path / "material.mdl"
    mdl.write_text(
        'unknown_input: texture_2d("ignored.png")\n'
        'diffuse_texture: texture_2d("https://example.test/albedo.png")\n',
        encoding="utf-8",
    )
    assert step1x_backend_module._find_mdl_texture_paths(mdl) == {}

    assert (
        step1x_backend_module._resolve_asset_path(
            SimpleNamespace(resolvedPath="", path=""),
            tmp_path,
            extraction_root=None,
        )
        is None
    )
    local = tmp_path / "albedo.png"
    local.write_bytes(b"png")
    assert (
        step1x_backend_module._resolve_asset_path(
            SimpleNamespace(resolvedPath=local.as_uri(), path=""),
            tmp_path,
            extraction_root=None,
        )
        == local.resolve()
    )
    assert (
        step1x_backend_module._resolve_asset_path(
            SimpleNamespace(resolvedPath="https://example.test/albedo.png", path=""),
            tmp_path,
            extraction_root=None,
        )
        is None
    )
    assert (
        step1x_backend_module._resolve_asset_path(
            SimpleNamespace(resolvedPath="", path="relative.png"),
            tmp_path,
            extraction_root=None,
        )
        == tmp_path / "relative.png"
    )


def test_package_asset_resolution_cache_and_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_path = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package_path, "w") as archive:
        archive.writestr("textures/cached.png", b"cached")
        archive.writestr("textures/missing.png", b"missing")
        archive.writestr("textures/bad.png", b"bad")

    output_dir = tmp_path / "out"
    cache_root = (
        output_dir
        / ".step1x_package_assets"
        / step1x_backend_module.package_member_cache_name(package_path)
    )
    cached = cache_root / "textures" / "cached.png"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached")
    package_mtime = package_path.stat().st_mtime
    stale_time = package_mtime + 1
    step1x_backend_module.os.utime(cached, (stale_time, stale_time))

    assert (
        step1x_backend_module._resolve_package_asset_path(
            f"{package_path}[textures/cached.png]",
            tmp_path,
            extraction_root=output_dir,
        )
        == cached.resolve()
    )

    original_stat = Path.stat
    missing_cached = cache_root / "textures" / "missing.png"
    missing_cached.parent.mkdir(parents=True, exist_ok=True)
    missing_cached.write_bytes(b"stale")

    def raising_stat(path: Path, *args: object, **kwargs: object) -> object:
        if path == missing_cached:
            raise OSError("stale cache blocked")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", raising_stat)
    monkeypatch.setattr(
        step1x_backend_module,
        "extract_usdz_member_to_dir",
        lambda *_args, **_kwargs: None,
    )
    assert (
        step1x_backend_module._resolve_package_asset_path(
            f"{package_path}[textures/missing.png]",
            tmp_path,
            extraction_root=output_dir,
        )
        is None
    )

    monkeypatch.setattr(Path, "stat", original_stat)

    def raise_value_error(*_args: object, **_kwargs: object) -> Path:
        raise ValueError("bad member")

    monkeypatch.setattr(
        step1x_backend_module,
        "extract_usdz_member_to_dir",
        raise_value_error,
    )
    assert (
        step1x_backend_module._resolve_package_asset_path(
            f"{package_path}[textures/bad.png]",
            tmp_path,
            extraction_root=output_dir,
        )
        is None
    )


def test_preserve_normal_and_process_helpers_cover_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_result = Step1XRunResult(albedo_uri=(tmp_path / "albedo.png").as_uri())
    missing_scope = Step1XScopeInfo(
        source_asset_path=tmp_path / "asset.usd",
        source_normal_path=tmp_path / "missing.png",
    )
    assert (
        step1x_backend_module._preserve_source_normal_if_missing(
            raw_result,
            scope=missing_scope,
            output_dir=tmp_path / "out",
        )
        is raw_result
    )

    source_normal = tmp_path / "normal.png"
    source_normal.write_bytes(b"not-an-image")
    monkeypatch.setattr(
        step1x_backend_module,
        "_write_preserved_normal_png",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad normal")),
    )
    assert (
        step1x_backend_module._preserve_source_normal_if_missing(
            raw_result,
            scope=Step1XScopeInfo(
                source_asset_path=tmp_path / "asset.usd",
                source_normal_path=source_normal,
            ),
            output_dir=tmp_path / "out",
        )
        is raw_result
    )

    class SlowProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: int) -> int:
            if not self.killed:
                raise step1x_backend_module.subprocess.TimeoutExpired("cmd", timeout)
            return -9

        def kill(self) -> None:
            self.killed = True

    process = SlowProcess()
    step1x_backend_module._terminate_process(process)
    assert process.terminated is True
    assert process.killed is True
    assert step1x_backend_module._tail_text(tmp_path / "missing.log") == ""


def _step1x_run_request(
    tmp_path: Path,
    *,
    custom_parameters: dict[str, object] | None = None,
) -> Step1XRunRequest:
    source_asset = tmp_path / "asset.usd"
    return Step1XRunRequest(
        prompt="test",
        seed=0,
        strength=0.5,
        texture_size=128,
        source_asset_uri=source_asset.as_uri(),
        source_asset_path=source_asset,
        source_albedo_path=None,
        reference_image_uris=(),
        turntable_video_uri=None,
        multiview_image_uris=(),
        target=None,
        scope=None,
        job_id="vj-command",
        output_dir=tmp_path / "out",
        runtime_dir=tmp_path,
        model_dir=None,
        cache_dir=None,
        custom_parameters=dict(custom_parameters or {}),
    )


def _write_fake_step1x_script(tmp_path: Path, *, blank: bool = False) -> Path:
    script = tmp_path / "edit_texture.py"
    script.write_text(
        """
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--strength")
parser.add_argument("--seed")
parser.add_argument("--resolution")
parser.add_argument("--skip-ma", action="store_true")
args, _ = parser.parse_known_args()

out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
image = Image.new("RGB", (2, 2), (32, 32, 32))
if not __BLANK__:
    image.putpixel((1, 1), (240, 180, 40))
image.save(out / "final_albedo.png")
(out / "input_usd.txt").write_text(args.usd, encoding="utf-8")
(out / f"edited_{Path(args.usd).name}").write_text("#usda 1.0\\n", encoding="utf-8")
""".replace("__BLANK__", "True" if blank else "False"),
        encoding="utf-8",
    )
    return script


def _write_two_material_stage(tmp_path: Path) -> Path:
    from PIL import Image
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    Image.new("RGB", (2, 2), (160, 20, 20)).save(tmp_path / "red_albedo.png")
    Image.new("RGB", (2, 2), (20, 20, 160)).save(tmp_path / "blue_albedo.png")
    stage_path = tmp_path / "two_materials.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")

    red = UsdShade.Material.Define(stage, "/World/Looks/Red")
    red_shader = UsdShade.Shader.Define(stage, "/World/Looks/Red/Shader")
    red_shader.CreateIdAttr("UsdPreviewSurface")
    red_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("red_albedo.png")
    )
    red.CreateSurfaceOutput().ConnectToSource(red_shader.ConnectableAPI(), "surface")

    blue = UsdShade.Material.Define(stage, "/World/Looks/Blue")
    blue_shader = UsdShade.Shader.Define(stage, "/World/Looks/Blue/Shader")
    blue_shader.CreateIdAttr("OmniPBR")
    blue_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("blue_albedo.png")
    )
    blue.CreateSurfaceOutput().ConnectToSource(blue_shader.ConnectableAPI(), "surface")

    mesh_a = UsdGeom.Mesh.Define(stage, "/World/MeshA")
    mesh_b = UsdGeom.Mesh.Define(stage, "/World/MeshB")
    for mesh in (mesh_a, mesh_b):
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdShade.MaterialBindingAPI.Apply(mesh_a.GetPrim()).Bind(red)
    UsdShade.MaterialBindingAPI.Apply(mesh_b.GetPrim()).Bind(blue)
    stage.Save()
    return stage_path


def _write_multi_mesh_blue_stage(tmp_path: Path) -> Path:
    from PIL import Image
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    Image.new("RGB", (2, 2), (20, 20, 160)).save(tmp_path / "blue_albedo.png")
    stage_path = tmp_path / "multi_mesh_blue.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    world.AddTranslateOp().Set((318.0, -2.0, -135.0))
    UsdGeom.Scope.Define(stage, "/World/Looks")

    blue = UsdShade.Material.Define(stage, "/World/Looks/Blue")
    blue_shader = UsdShade.Shader.Define(stage, "/World/Looks/Blue/Shader")
    blue_shader.CreateIdAttr("OmniPBR")
    blue_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("blue_albedo.png")
    )
    blue.CreateSurfaceOutput().ConnectToSource(blue_shader.ConnectableAPI(), "surface")

    for index, x_offset in enumerate((0.0, 2.0), start=1):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/BlueMesh{index}")
        mesh.CreatePointsAttr(
            [
                (x_offset, 0, 0),
                (x_offset + 1, 0, 0),
                (x_offset, 1, 0),
            ]
        )
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        )
        st.Set([(0, 0), (1, 0), (0, 1)])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(blue)

    stage.Save()
    return stage_path


def _write_non_finite_uv_stage(tmp_path: Path) -> Path:
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    Image.new("RGB", (2, 2), (20, 20, 160)).save(tmp_path / "blue_albedo.png")
    stage_path = tmp_path / "non_finite_uv.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")

    blue = UsdShade.Material.Define(stage, "/World/Looks/Blue")
    blue_shader = UsdShade.Shader.Define(stage, "/World/Looks/Blue/Shader")
    blue_shader.CreateIdAttr("OmniPBR")
    blue_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("blue_albedo.png")
    )
    blue.CreateSurfaceOutput().ConnectToSource(blue_shader.ConnectableAPI(), "surface")

    mesh = UsdGeom.Mesh.Define(stage, "/World/MeshB")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    st.Set(
        [
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(float("nan"), 0.0),
            Gf.Vec2f(0.0, 1.0),
        ]
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(blue)
    stage.Save()
    return stage_path


def _write_disconnected_file_texture_stage(tmp_path: Path) -> Path:
    from PIL import Image
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    Image.new("RGB", (2, 2), (64, 64, 64)).save(tmp_path / "unknown.png")

    stage_path = tmp_path / "disconnected_file_texture.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/Paint")
    preview = UsdShade.Shader.Define(stage, "/World/Looks/Paint/PreviewSurface")
    preview.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")

    texture = UsdShade.Shader.Define(stage, "/World/Looks/Paint/UnknownTex")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("unknown.png")
    )

    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.Save()
    return stage_path


def _write_constant_color_stage(
    tmp_path: Path,
    *,
    diffuse_color: tuple[float, float, float] = (0.05, 0.1, 0.8),
) -> Path:
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    stage_path = tmp_path / "constant_color.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")

    material = UsdShade.Material.Define(stage, "/World/Looks/Paint")
    preview = UsdShade.Shader.Define(stage, "/World/Looks/Paint/PreviewSurface")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse_color)
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")

    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.Save()
    return stage_path


def _write_mdl_material_stage(tmp_path: Path) -> Path:
    from PIL import Image
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    texture_dir = tmp_path / "materials" / "textures" / "Steel_A"
    texture_dir.mkdir(parents=True)
    Image.new("RGB", (2, 2), (160, 160, 160)).save(texture_dir / "albedo.png")
    Image.new("RGB", (2, 2), (128, 128, 255)).save(texture_dir / "normal.png")
    Image.new("RGB", (2, 2), (255, 128, 0)).save(texture_dir / "orm.png")
    (tmp_path / "materials" / "Steel_A.mdl").write_text(
        """
mdl 1.4;

export material Steel_A(*)
 = OmniPBR(
    diffuse_texture: texture_2d("./textures/Steel_A/albedo.png", ::tex::gamma_srgb),
    normalmap_texture: texture_2d("./textures/Steel_A/normal.png", ::tex::gamma_linear),
    ORM_texture: texture_2d("./textures/Steel_A/orm.png", ::tex::gamma_linear));
""",
        encoding="utf-8",
    )

    stage_path = tmp_path / "mdl_material.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/Steel_A")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Steel_A/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("./materials/Steel_A.mdl"))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.Save()
    return stage_path


def _write_preview_surface_texture_stage(tmp_path: Path) -> Path:
    from PIL import Image
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    Image.new("RGB", (2, 2), (10, 80, 160)).save(tmp_path / "paint_albedo.png")
    Image.new("RGB", (2, 2), (128, 128, 255)).save(tmp_path / "paint_normal.png")
    Image.new("RGB", (2, 2), (255, 128, 0)).save(tmp_path / "paint_orm.png")

    stage_path = tmp_path / "preview_surface_material.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/Paint")
    preview = UsdShade.Shader.Define(stage, "/World/Looks/Paint/PreviewSurface")
    preview.CreateIdAttr("UsdPreviewSurface")

    albedo_texture = UsdShade.Shader.Define(stage, "/World/Looks/Paint/AlbedoTex")
    albedo_texture.CreateIdAttr("UsdUVTexture")
    albedo_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("paint_albedo.png")
    )
    albedo_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        albedo_texture.ConnectableAPI(), "rgb"
    )

    normal_texture = UsdShade.Shader.Define(stage, "/World/Looks/Paint/NormalTex")
    normal_texture.CreateIdAttr("UsdUVTexture")
    normal_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("paint_normal.png")
    )
    normal_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    preview.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        normal_texture.ConnectableAPI(), "rgb"
    )

    orm_texture = UsdShade.Shader.Define(stage, "/World/Looks/Paint/OrmTex")
    orm_texture.CreateIdAttr("UsdUVTexture")
    orm_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("paint_orm.png")
    )
    orm_texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
    preview.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(
        orm_texture.ConnectableAPI(), "r"
    )

    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.Save()
    return stage_path
