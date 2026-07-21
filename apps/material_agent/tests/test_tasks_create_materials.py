# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the WP6 create_materials pipeline step."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from PIL import Image

pytest.importorskip("pxr")

from pxr import Sdf  # noqa: E402

import material_agent.tasks.create_materials as create_materials_task  # noqa: E402
import material_agent.tasks.unified_pipeline_executor as upe  # noqa: E402
from material_agent.material_library_generation.conditioning import (  # noqa: E402
    OVRTX_CONDITIONING_SCHEMA_VERSION,
    REAL_SEED_MATERIAL_SCHEMA_VERSION,
)
from material_agent.material_library_generation.creation_contract import (  # noqa: E402
    BackendMaterialResult,
    MaterialConditioningKind,
    MaterialCreationError,
    MaterialCreationErrorCode,
    MaterialCreationProvenance,
)
from material_agent.material_library_generation.fake_backend import (  # noqa: E402
    FakeMaterialCreationBackend as RealFakeMaterialCreationBackend,
)
from material_agent.materials import FALLBACK_MATERIAL_NAME  # noqa: E402
from material_agent.tasks.create_materials import CreateMaterialsTask  # noqa: E402
from material_agent.tasks.unified_pipeline_executor import (  # noqa: E402
    UnifiedPipelineExecutorTask,
)


def test_create_materials_fake_backend_registers_and_assigns(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    predictions_path = _write_predictions(tmp_path / "predictions.jsonl")

    result = CreateMaterialsTask().run(
        {
            "source_usd": str(source_usd),
            "predictions_path": str(predictions_path),
            "output_dir": str(tmp_path / "created"),
            "output_predictions_path": str(tmp_path / "created_predictions.jsonl"),
            "creation_requests": [_creation_request()],
            "backend": "fake",
            "material_profile": "auto",
        }
    )

    assert result["created_material_count"] == 1
    assert result["assignment_count"] == 1
    assert Path(result["created_material_library_path"]).is_file()
    assert Path(result["created_materials_manifest_path"]).is_file()
    assert Path(result["created_materials_yaml_path"]).is_file()

    assigned_predictions = _read_jsonl(Path(result["predictions_path"]))
    assert assigned_predictions[0]["id"] == "/World/Asset/Housing"
    assert assigned_predictions[0]["materials"]["material"] == "Satin Blue Plastic"
    assert assigned_predictions[0]["materials"]["creation_action"] == "create_new"
    assert assigned_predictions[0]["material_creation"]["material_id"] == (
        "satin_blue_plastic"
    )

    materials_yaml = yaml.safe_load(
        Path(result["created_materials_yaml_path"]).read_text(encoding="utf-8")
    )
    assert materials_yaml["library_path"].endswith("material_library.usda")
    assert materials_yaml["entries"][0]["name"] == "Satin Blue Plastic"
    assert materials_yaml["entries"][0]["source"] == "generated"

    status_manifest = json.loads(
        Path(result["created_materials_manifest_path"]).read_text(encoding="utf-8")
    )
    assert status_manifest["schema_version"] == (
        "material-agent-create-materials-step.v1"
    )
    assert status_manifest["statuses"][0]["status"] == "created"
    assert status_manifest["statuses"][0]["cache_hit"] is False


def test_create_materials_step1x_dispatches_to_non_fake_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_usd = _write_conditioned_source_usd(tmp_path / "asset.usda")
    predictions_path = _write_predictions(tmp_path / "predictions.jsonl")
    constructed: list[_Step1XDispatchStubBackend] = []

    def fake_backend_must_not_be_registered(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("fake backend should not be registered for Step1X")

    def step1x_backend(context: dict[str, Any]) -> _Step1XDispatchStubBackend:
        assert context["step1x"]["strength"] == 0.4
        backend = _Step1XDispatchStubBackend()
        constructed.append(backend)
        return backend

    monkeypatch.setattr(
        create_materials_task,
        "FakeMaterialCreationBackend",
        fake_backend_must_not_be_registered,
    )
    monkeypatch.setattr(
        create_materials_task,
        "_create_step1x_backend",
        step1x_backend,
    )
    conditioning = _temporary_real_conditioning_config(tmp_path, source_usd)
    request_spec = _creation_request()
    request_spec["conditioning"] = conditioning

    result = CreateMaterialsTask().run(
        {
            "source_usd": str(source_usd),
            "predictions_path": str(predictions_path),
            "output_dir": str(tmp_path / "created"),
            "creation_requests": [request_spec],
            "backend": "step1x_material_anything",
            "step1x": {"strength": 0.4},
            "conditioning": [],
        }
    )

    assert result["created_material_count"] == 1
    assert constructed
    assert constructed[0].calls
    assert constructed[0].conditioning is not None
    assert MaterialConditioningKind.SCOPED_USD in {
        artifact.kind for artifact in constructed[0].conditioning.artifacts
    }
    creation_manifest = json.loads(
        Path(result["statuses"][0]["creation_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert creation_manifest["request"]["backend"] == "step1x_material_anything"
    assert creation_manifest["backend_result"]["provenance"]["backend"] == (
        "step1x_material_anything"
    )
    assert creation_manifest["created_material"]["provenance"]["backend"] == (
        "step1x_material_anything"
    )
    assert (
        creation_manifest["backend_result"]["provenance"]["conditioning_fingerprint"]
        is not None
    )


def test_create_materials_step1x_rejects_missing_real_conditioning_before_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_usd = _write_conditioned_source_usd(tmp_path / "asset.usda")
    backend = _Step1XDispatchStubBackend()
    monkeypatch.setattr(
        create_materials_task,
        "_create_step1x_backend",
        lambda _context: backend,
    )

    with pytest.raises(MaterialCreationError, match="explicit seed-material"):
        CreateMaterialsTask().run(
            {
                "source_usd": str(source_usd),
                "output_dir": str(tmp_path / "created"),
                "creation_requests": [_creation_request()],
                "backend": "step1x_material_anything",
            }
        )

    assert backend.calls == []


def test_create_materials_step1x_rejects_non_mapping_request_conditioning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_usd = _write_conditioned_source_usd(tmp_path / "asset.usda")
    backend = _Step1XDispatchStubBackend()
    monkeypatch.setattr(
        create_materials_task,
        "_create_step1x_backend",
        lambda _context: backend,
    )
    request_spec = _creation_request()
    request_spec["conditioning"] = []

    with pytest.raises(TypeError, match="conditioning must be a mapping"):
        CreateMaterialsTask().run(
            {
                "source_usd": str(source_usd),
                "output_dir": str(tmp_path / "created"),
                "creation_requests": [request_spec],
                "backend": "step1x_material_anything",
            }
        )

    assert backend.calls == []


def test_create_materials_unknown_backend_fails_closed_without_fake_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")

    def fake_backend_must_not_be_registered(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("fake backend should not be registered as a fallback")

    monkeypatch.setattr(
        create_materials_task,
        "FakeMaterialCreationBackend",
        fake_backend_must_not_be_registered,
    )

    with pytest.raises(MaterialCreationError) as error:
        CreateMaterialsTask().run(
            {
                "source_usd": str(source_usd),
                "output_dir": str(tmp_path / "created"),
                "creation_requests": [_creation_request()],
                "backend": "missing_backend",
            }
        )

    assert error.value.code is MaterialCreationErrorCode.BACKEND_UNAVAILABLE
    assert error.value.backend == "missing_backend"
    assert not (tmp_path / "created").exists()


def test_create_materials_unknown_backend_fails_before_fail_open_status(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")

    with pytest.raises(MaterialCreationError) as error:
        CreateMaterialsTask().run(
            {
                "source_usd": str(source_usd),
                "output_dir": str(tmp_path / "created"),
                "creation_requests": [_creation_request()],
                "backend": "missing_backend",
                "fail_on_error": False,
            }
        )

    assert error.value.code is MaterialCreationErrorCode.BACKEND_UNAVAILABLE
    assert error.value.backend == "missing_backend"
    assert not (tmp_path / "created" / "material_creation_status.json").exists()


@pytest.mark.parametrize("backend_value", [None, "", " \t\n"])
def test_create_materials_explicit_blank_backend_fails_closed(
    tmp_path: Path,
    backend_value: Any,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")

    with pytest.raises(ValueError, match="backend"):
        CreateMaterialsTask().run(
            {
                "source_usd": str(source_usd),
                "output_dir": str(tmp_path / "created"),
                "creation_requests": [_creation_request()],
                "backend": backend_value,
            }
        )

    assert not (tmp_path / "created").exists()


@pytest.mark.parametrize("backend_alias", ["auto", "auto-for-test"])
def test_create_materials_fake_alias_selects_fake_backend(
    tmp_path: Path,
    backend_alias: str,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")

    result = CreateMaterialsTask().run(
        {
            "source_usd": str(source_usd),
            "output_dir": str(tmp_path / "created"),
            "creation_requests": [_creation_request()],
            "backend": backend_alias,
        }
    )

    assert result["created_material_count"] == 1
    creation_manifest = json.loads(
        Path(result["statuses"][0]["creation_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert creation_manifest["request"]["backend"] == "fake"
    assert creation_manifest["backend_result"]["provenance"]["backend"] == "fake"


def test_create_materials_step1x_config_parsing_helpers(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_data = create_materials_task._step1x_config_data(
        {
            "step1x": {
                "runtime_dir": "runtime",
                "extra_args": "--foo 'bar baz'",
                "timeout_sec": "45",
                "validate_assets": "true",
                "skip_material_anything": "false",
                "require_upscaler": False,
                "strength": "0.25",
            },
            "step1x_material_anything": {
                "command_template": None,
                "model_revisions": ["step1x-test", "ma-test"],
                "required_executables": ["python", "bash"],
                "timeout_sec": "60",
            },
        }
    )

    runtime_config = create_materials_task._step1x_runtime_config(
        _DummyStep1XRuntimeConfig,
        config_data,
        base_dir=config_dir,
    )

    assert runtime_config.runtime_dir == (config_dir / "runtime").resolve()
    assert runtime_config.extra_args == ("--foo", "bar baz")
    assert runtime_config.required_executables == ("python", "bash")
    assert runtime_config.timeout_sec == 60
    assert runtime_config.validate_assets is True
    assert runtime_config.skip_material_anything is False
    assert runtime_config.require_upscaler is False
    assert runtime_config.command_template is None
    assert create_materials_task._step1x_config_data({}) == {}
    assert create_materials_task._optional_config_base_dir(None) is None
    assert create_materials_task._optional_config_base_dir("") is None
    assert create_materials_task._optional_config_base_dir(config_dir) == (
        config_dir.resolve()
    )
    assert (
        create_materials_task._optional_config_path(
            None,
            field_name="runtime_dir",
        )
        is None
    )
    assert create_materials_task._optional_config_path(
        "runtime",
        field_name="runtime_dir",
    ) == Path("runtime")
    assert (
        create_materials_task._string_tuple(
            None,
            field_name="model_revisions",
        )
        == ()
    )
    assert create_materials_task._string_tuple(
        config_data["model_revisions"],
        field_name="model_revisions",
    ) == ("step1x-test", "ma-test")
    assert (
        create_materials_task._float_config_value(
            config_data["strength"],
            field_name="strength",
        )
        == 0.25
    )
    assert create_materials_task._step1x_strength({}) == 0.8
    assert create_materials_task._step1x_strength(config_data) == 0.25
    with pytest.raises(ValueError, match="timeout_sec"):
        create_materials_task._int_config_value("not-an-int", field_name="timeout_sec")
    with pytest.raises(ValueError, match="timeout_sec"):
        create_materials_task._int_config_value(True, field_name="timeout_sec")
    with pytest.raises(ValueError, match="strength"):
        create_materials_task._float_config_value("not-a-float", field_name="strength")
    with pytest.raises(ValueError, match="strength"):
        create_materials_task._float_config_value(False, field_name="strength")
    with pytest.raises(ValueError, match="validate_assets"):
        create_materials_task._bool_config_value("maybe", field_name="validate_assets")
    with pytest.raises(ValueError, match="invalid runtime"):
        create_materials_task._step1x_runtime_config(
            _InvalidStep1XRuntimeConfig,
            {},
        )

    custom_parameters = create_materials_task._step1x_custom_parameters(
        config_data,
        runtime_config,
    )
    assert custom_parameters["skip_material_anything"] is False
    base_revision = create_materials_task._step1x_backend_revision(
        config_data,
        runtime_config=runtime_config,
        model_revisions=("step1x-test", "ma-test"),
        strength=0.25,
        custom_parameters=custom_parameters,
    )
    null_revision = create_materials_task._step1x_backend_revision(
        {**config_data, "backend_revision": None},
        runtime_config=runtime_config,
        model_revisions=("step1x-test", "ma-test"),
        strength=0.25,
        custom_parameters=custom_parameters,
    )
    changed_revision = create_materials_task._step1x_backend_revision(
        config_data,
        runtime_config=runtime_config,
        model_revisions=("step1x-test", "ma-test"),
        strength=0.5,
        custom_parameters=custom_parameters,
    )
    changed_command_revision = create_materials_task._step1x_backend_revision(
        config_data,
        runtime_config=replace(runtime_config, command_template="step1x {prompt}"),
        model_revisions=("step1x-test", "ma-test"),
        strength=0.25,
        custom_parameters=custom_parameters,
    )
    changed_args_revision = create_materials_task._step1x_backend_revision(
        config_data,
        runtime_config=replace(runtime_config, extra_args=("--foo", "different")),
        model_revisions=("step1x-test", "ma-test"),
        strength=0.25,
        custom_parameters=custom_parameters,
    )
    changed_model_dir_revision = create_materials_task._step1x_backend_revision(
        config_data,
        runtime_config=replace(
            runtime_config,
            model_dir=(config_dir / "other-models").resolve(),
        ),
        model_revisions=("step1x-test", "ma-test"),
        strength=0.25,
        custom_parameters=custom_parameters,
    )
    changed_model_revisions_revision = create_materials_task._step1x_backend_revision(
        config_data,
        runtime_config=runtime_config,
        model_revisions=("step1x-next", "ma-test"),
        strength=0.25,
        custom_parameters=custom_parameters,
    )
    default_model_revisions_revision = create_materials_task._step1x_backend_revision(
        config_data,
        runtime_config=runtime_config,
        model_revisions=("step1x-runtime", "material-anything-runtime"),
        strength=0.25,
        custom_parameters=custom_parameters,
    )
    assert base_revision.startswith("step1x-material-creation-adapter.v2+cfg.")
    assert "+cfg." in base_revision
    assert null_revision == base_revision
    assert base_revision != changed_revision
    assert base_revision != changed_command_revision
    assert base_revision != changed_args_revision
    assert base_revision != changed_model_dir_revision
    assert base_revision != changed_model_revisions_revision
    assert base_revision != default_model_revisions_revision


def test_create_materials_constructs_real_step1x_backend(tmp_path: Path) -> None:
    step1x_config = {
        "runtime_dir": "runtime",
        "model_dir": "models",
        "command_template": None,
        "timeout_sec": "60",
        "validate_assets": False,
        "skip_material_anything": False,
        "require_upscaler": True,
        "extra_args": ["--debug"],
        "required_executables": [],
        "strength": "0.4",
        "model_revisions": ["step1x-test", "material-anything-test"],
        "custom_parameters": {"ma_steps": 4},
        "backend_revision": "step1x-test.v1",
    }
    backend = create_materials_task._create_step1x_backend(
        {
            "_config_dir": tmp_path,
            "step1x": step1x_config,
        }
    )
    changed_model_backend = create_materials_task._create_step1x_backend(
        {
            "_config_dir": tmp_path,
            "step1x": {
                **step1x_config,
                "model_revisions": ["step1x-next", "material-anything-test"],
            },
        }
    )
    default_model_backend = create_materials_task._create_step1x_backend({})
    explicit_default_model_backend = create_materials_task._create_step1x_backend(
        {
            "step1x": {
                "model_revisions": [
                    "step1x-runtime",
                    "material-anything-runtime",
                ]
            }
        }
    )

    assert backend.name == "step1x_material_anything"
    assert backend.revision.startswith("step1x-test.v1+cfg.")
    assert backend.revision.count("+cfg.") == 1
    assert backend.revision != changed_model_backend.revision
    assert default_model_backend.revision == explicit_default_model_backend.revision
    assert backend.config.step1x.runtime_dir == (tmp_path / "runtime").resolve()
    assert backend.config.step1x.model_dir == (tmp_path / "models").resolve()
    assert backend.config.step1x.timeout_sec == 60
    assert backend.config.step1x.validate_assets is False
    assert backend.config.step1x.skip_material_anything is False
    assert backend.config.step1x.require_upscaler is True
    assert backend.config.step1x.extra_args == ("--debug",)
    assert backend.config.step1x.required_executables == ()
    assert backend.config.model_revisions == (
        "step1x-test",
        "material-anything-test",
    )
    assert backend.config.strength == 0.4
    assert backend.config.custom_parameters == {
        "ma_steps": 4,
        "skip_material_anything": False,
    }


def test_create_materials_real_step1x_backend_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="strength must be in"):
        create_materials_task._create_step1x_backend({"step1x": {"strength": 2.0}})


def test_create_materials_resolves_config_relative_reference_images(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    absolute_reference = tmp_path / "absolute-request.png"
    spec = _creation_request(
        reference_image_uris=[
            "request.png",
            absolute_reference,
            " https://example.test/request.png ",
        ]
    )
    spec["recipe"]["reference_image_uris"] = [
        "recipe.png",
        " file:///tmp/recipe.png ",
    ]

    result = CreateMaterialsTask().run(
        {
            "source_usd": str(source_usd),
            "output_dir": str(tmp_path / "created"),
            "creation_requests": [spec],
            "backend": "fake",
            "_config_dir": config_dir,
        }
    )
    manifest = json.loads(
        Path(result["statuses"][0]["creation_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    recipe_references = [
        str((config_dir / "recipe.png").resolve()),
        "file:///tmp/recipe.png",
    ]
    request_references = [
        str((config_dir / "request.png").resolve()),
        str(absolute_reference),
        "https://example.test/request.png",
    ]

    assert manifest["request"]["recipe"]["reference_image_uris"] == recipe_references
    assert manifest["request"]["reference_image_uris"] == (
        recipe_references + request_references
    )


@pytest.mark.parametrize("empty_reference", ["", " \t"])
@pytest.mark.parametrize("reference_location", ["request", "recipe"])
def test_create_materials_rejects_empty_config_relative_reference_image(
    tmp_path: Path,
    reference_location: str,
    empty_reference: str,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    spec = _creation_request()
    if reference_location == "request":
        spec["reference_image_uris"] = [empty_reference]
    else:
        spec["recipe"]["reference_image_uris"] = [empty_reference]

    with pytest.raises(ValueError, match="reference_image_uris"):
        CreateMaterialsTask().run(
            {
                "source_usd": str(source_usd),
                "output_dir": str(tmp_path / "created"),
                "creation_requests": [spec],
                "backend": "fake",
                "_config_dir": tmp_path / "configs",
            }
        )


def test_create_materials_polls_cancel_checker_during_backend_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_usd = _write_conditioned_source_usd(tmp_path / "asset.usda")
    checks = 0

    def cancel_checker() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    monkeypatch.setattr(
        create_materials_task,
        "_create_step1x_backend",
        lambda _context: _Step1XWaitForCancelBackend(),
    )
    conditioning = _temporary_real_conditioning_config(tmp_path, source_usd)

    with pytest.raises(MaterialCreationError) as error:
        CreateMaterialsTask().run(
            {
                "source_usd": str(source_usd),
                "output_dir": str(tmp_path / "created"),
                "creation_requests": [_creation_request()],
                "backend": "step1x_material_anything",
                "cancel_checker": cancel_checker,
                "conditioning": conditioning,
            }
        )

    assert error.value.code is MaterialCreationErrorCode.CANCELLED
    assert checks >= 2


def test_create_materials_polls_cancel_checker_during_conditioning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_usd = _write_conditioned_source_usd(tmp_path / "asset.usda")
    checks = 0

    def cancel_checker() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    def wait_for_cancel(*_args: Any, cancel_event: threading.Event, **_kwargs: Any):
        assert cancel_event.wait(timeout=2.0)
        raise MaterialCreationError(
            MaterialCreationErrorCode.CANCELLED,
            "conditioning cancelled",
            backend="step1x_material_anything",
        )

    monkeypatch.setattr(
        create_materials_task,
        "_create_step1x_backend",
        lambda _context: _Step1XWaitForCancelBackend(),
    )
    monkeypatch.setattr(
        create_materials_task,
        "prepare_material_conditioning",
        wait_for_cancel,
    )
    conditioning = _temporary_real_conditioning_config(tmp_path, source_usd)

    with pytest.raises(MaterialCreationError) as error:
        CreateMaterialsTask().run(
            {
                "source_usd": str(source_usd),
                "output_dir": str(tmp_path / "created"),
                "creation_requests": [_creation_request()],
                "backend": "step1x_material_anything",
                "cancel_checker": cancel_checker,
                "conditioning": conditioning,
            }
        )

    assert error.value.code is MaterialCreationErrorCode.CANCELLED
    assert checks >= 2


def test_create_materials_propagates_cancel_checker_error_during_backend_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_usd = _write_conditioned_source_usd(tmp_path / "asset.usda")
    checks = 0

    def cancel_checker() -> bool:
        nonlocal checks
        checks += 1
        if checks >= 2:
            raise RuntimeError("cancel probe failed")
        return False

    monkeypatch.setattr(
        create_materials_task,
        "_create_step1x_backend",
        lambda _context: _Step1XWaitForCancelBackend(),
    )
    conditioning = _temporary_real_conditioning_config(tmp_path, source_usd)

    with pytest.raises(RuntimeError, match="cancel probe failed"):
        CreateMaterialsTask().run(
            {
                "source_usd": str(source_usd),
                "output_dir": str(tmp_path / "created"),
                "creation_requests": [_creation_request()],
                "backend": "step1x_material_anything",
                "cancel_checker": cancel_checker,
                "conditioning": conditioning,
            }
        )

    assert checks >= 2


def test_create_materials_reuses_cached_package_on_resume(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    predictions_path = _write_predictions(tmp_path / "predictions.jsonl")
    context = {
        "source_usd": str(source_usd),
        "predictions_path": str(predictions_path),
        "output_dir": str(tmp_path / "created"),
        "creation_requests": [_creation_request()],
        "backend": "fake",
    }

    first = CreateMaterialsTask().run(context)
    second = CreateMaterialsTask().run(context)

    assert first["statuses"][0]["cache_hit"] is False
    assert second["statuses"][0]["cache_hit"] is True


def test_create_materials_combines_multiple_packages_and_assigns_variants(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                "",
                json.dumps({"id": "/World/Asset/Sensor"}),
                json.dumps(
                    {
                        "id": "/World/Asset/Housing",
                        "materials": "legacy-placeholder",
                    }
                ),
                json.dumps({"id": "/World/Asset/Panel", "materials": {}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    context = {
        "source_usd": str(source_usd),
        "predictions_path": str(predictions_path),
        "output_dir": str(tmp_path / "created"),
        "creation_requests": [
            _creation_request(
                target_prim_paths="/World/Asset/Housing",
                reference_image_uris="file:///tmp/ref.png",
            ),
            _creation_request(
                material_id="warm_gray_rubber",
                name="Warm Gray Rubber",
                prediction_id="/World/Asset/Panel",
                target_prim_paths=["/World/Asset/Panel"],
                color="gray",
                material="rubber",
                finish="matte",
            ),
        ],
        "backend": "fake",
    }

    first = CreateMaterialsTask().run(context)
    second = CreateMaterialsTask().run(context)

    assert first["created_material_count"] == 2
    assert first["assignment_count"] == 2
    combined_library = Path(first["created_material_library_path"])
    assert combined_library.name == "created_materials.usda"
    assert combined_library.is_file()
    layer = Sdf.Layer.FindOrOpen(str(combined_library))
    assert layer
    assert layer.GetPrimAtPath(Sdf.Path("/World/Looks/Satin_Blue_Plastic"))
    assert layer.GetPrimAtPath(Sdf.Path("/World/Looks/Warm_Gray_Rubber"))
    assert second["statuses"][0]["cache_hit"] is True
    assert Path(second["created_material_library_path"]) == combined_library

    assigned_predictions = _read_jsonl(Path(first["predictions_path"]))
    assert assigned_predictions[1]["materials"]["material"] == "Satin Blue Plastic"
    assert assigned_predictions[2]["materials"]["material"] == "Warm Gray Rubber"
    materials_yaml = yaml.safe_load(
        Path(first["created_materials_yaml_path"]).read_text(encoding="utf-8")
    )
    assert materials_yaml["entries"][0]["creation_manifest"] == (
        "packages/satin_blue_plastic/material_creation_manifest.json"
    )
    assert materials_yaml["entries"][1]["creation_manifest"] == (
        "packages/warm_gray_rubber/material_creation_manifest.json"
    )


def test_create_materials_assigns_request_to_all_target_predictions(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "/World/Asset/A", "materials": {}}),
                json.dumps({"id": "/World/Asset/B", "materials": {}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    request = _creation_request(
        prediction_id="",
        target_prim_paths=["/World/Asset/A", "/World/Asset/B"],
    )

    result = CreateMaterialsTask().run(
        {
            "source_usd": str(source_usd),
            "predictions_path": str(predictions_path),
            "output_dir": str(tmp_path / "created"),
            "creation_requests": [request],
        }
    )

    assert result["assignment_count"] == 2
    assigned_predictions = _read_jsonl(Path(result["predictions_path"]))
    assert assigned_predictions[0]["materials"]["material"] == "Satin Blue Plastic"
    assert assigned_predictions[1]["materials"]["material"] == "Satin Blue Plastic"


def test_create_materials_rejects_conflicting_material_reuse_key(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")

    with pytest.raises(ValueError, match="conflicting requests"):
        CreateMaterialsTask().run(
            {
                "source_usd": str(source_usd),
                "output_dir": str(tmp_path / "created"),
                "overwrite": True,
                "creation_requests": [
                    _creation_request(
                        material_id="same_material",
                        name="Same Material",
                        prediction_id="/World/Asset/A",
                        target_prim_paths=["/World/Asset/A"],
                    ),
                    _creation_request(
                        material_id="same_material",
                        name="Same Material",
                        prediction_id="/World/Asset/B",
                        target_prim_paths=["/World/Asset/B"],
                    ),
                ],
            }
        )


def test_create_materials_deduplicates_identical_requests(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    request = _creation_request()

    result = CreateMaterialsTask().run(
        {
            "source_usd": str(source_usd),
            "output_dir": str(tmp_path / "created"),
            "creation_requests": [request, dict(request)],
        }
    )

    assert result["created_material_count"] == 1
    assert len(result["statuses"]) == 1


def test_create_materials_allows_unassigned_request_without_predictions(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    request = _creation_request()
    request.pop("prediction_id")

    result = CreateMaterialsTask().run(
        {
            "source_usd": str(source_usd),
            "output_dir": str(tmp_path / "created"),
            "creation_requests": [request],
        }
    )

    assert result["created_material_count"] == 1
    assert result["assignment_count"] == 0


def test_create_materials_records_backend_failure_when_fail_open(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")

    result = CreateMaterialsTask().run(
        {
            "source_usd": str(source_usd),
            "output_dir": str(tmp_path / "created"),
            "creation_requests": [_creation_request()],
            "fake_behavior": "failure",
            "fail_on_error": False,
        }
    )

    assert result["created_material_count"] == 0
    assert result["created_material_library_path"] is None
    assert result["created_materials_data"]["library_path"] is None
    assert result["statuses"][0]["status"] == "error"
    assert result["statuses"][0]["code"] == "backend_failure"


def test_create_materials_honors_cancel_checker(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")

    with pytest.raises(MaterialCreationError, match="cancelled|cancel"):
        CreateMaterialsTask().run(
            {
                "source_usd": str(source_usd),
                "output_dir": str(tmp_path / "created"),
                "creation_requests": [_creation_request()],
                "cancel_checker": lambda: True,
            }
        )

    assert not (tmp_path / "created" / "packages" / "satin_blue_plastic").exists()


def test_created_material_library_reports_invalid_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = _fake_created_material(tmp_path / "missing.usda", "/World/Looks/Missing")
    with pytest.raises(RuntimeError, match="Could not open"):
        create_materials_task._created_material_library(
            tmp_path,
            [missing, missing],
        )

    empty_path = tmp_path / "empty.usda"
    empty_layer = Sdf.Layer.CreateNew(str(empty_path))
    empty_layer.Save()
    no_prim = _fake_created_material(empty_path, "/World/Looks/Missing")
    with pytest.raises(RuntimeError, match="not found"):
        create_materials_task._created_material_library(
            tmp_path,
            [no_prim, no_prim],
        )

    valid_path = tmp_path / "valid.usda"
    valid_layer = Sdf.Layer.CreateNew(str(valid_path))
    Sdf.CreatePrimInLayer(valid_layer, Sdf.Path("/World/Looks/Valid"))
    valid_layer.Save()
    valid = _fake_created_material(valid_path, "/World/Looks/Valid")
    monkeypatch.setattr(create_materials_task.Sdf, "CopySpec", lambda *args: False)
    with pytest.raises(RuntimeError, match="Could not copy"):
        create_materials_task._created_material_library(
            tmp_path,
            [valid, valid],
        )


def test_executor_create_materials_autowires_downstream_apply(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    predictions_path = _write_predictions(tmp_path / "predictions.jsonl")
    executor = UnifiedPipelineExecutorTask()
    step_config: dict[str, Any] = {
        "source_usd": str(source_usd),
        "output_dir": str(tmp_path / "work" / "created_materials"),
        "creation_requests": [_creation_request()],
    }
    context: dict[str, Any] = {
        "working_dir": str(tmp_path / "work"),
        "cancel_checker": lambda: False,
        "event_listener": SimpleNamespace(event=lambda *_args, **_kwargs: None),
        "step_configs": {"create_materials": step_config, "apply": {}, "refine": {}},
    }
    pipeline_state: dict[str, Any] = {
        "step_outputs": {
            "harmonize_predictions": {"predictions_path": str(predictions_path)}
        }
    }

    outputs = executor._execute_step(
        "create_materials",
        step_config,
        context,
        object_store=None,
        pipeline_state=pipeline_state,
    )
    executor._activate_created_material_library(
        outputs, context, context["step_configs"]
    )

    assert outputs["created_material_count"] == 1
    assert Path(outputs["predictions_path"]).is_file()
    apply_mapping = context["step_configs"]["apply"]["materials_mapping"]
    assert (
        apply_mapping["material_library_path"]
        == outputs["created_material_library_path"]
    )
    assert apply_mapping["Satin Blue Plastic"] == "/World/Looks/Satin_Blue_Plastic"
    refine_mapping = context["step_configs"]["refine"]["apply"]["materials_mapping"]
    assert (
        refine_mapping["material_library_path"]
        == outputs["created_material_library_path"]
    )
    assert refine_mapping["Satin Blue Plastic"] == "/World/Looks/Satin_Blue_Plastic"


def test_executor_create_materials_preserves_prior_generated_materials(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    generated = CreateMaterialsTask().run(
        {
            "source_usd": str(source_usd),
            "output_dir": str(tmp_path / "generated"),
            "creation_requests": [
                _creation_request(
                    material_id="generated_green_paint",
                    name="Generated Green Paint",
                    prediction_id="/World/Asset/Generated",
                    target_prim_paths=["/World/Asset/Generated"],
                    color="green",
                )
            ],
        }
    )
    created = CreateMaterialsTask().run(
        {
            "source_usd": str(source_usd),
            "output_dir": str(tmp_path / "created"),
            "creation_requests": [_creation_request()],
        }
    )
    context = {
        "working_dir": str(tmp_path / "work"),
        "materials_data": {
            "library_path": "/default/materials.usda",
            "entries": [{"name": "Default Steel", "binding": "/World/Looks/Steel"}],
        },
        "default_materials_data": {
            "library_path": "/default/materials.usda",
            "entries": [{"name": "Default Steel", "binding": "/World/Looks/Steel"}],
        },
        "generated_materials_data": generated["created_materials_data"],
    }
    step_configs: dict[str, dict[str, Any]] = {"apply": {}}

    UnifiedPipelineExecutorTask()._activate_created_material_library(
        created,
        context,
        step_configs,
    )

    apply_mapping = step_configs["apply"]["materials_mapping"]
    assert apply_mapping["Generated Green Paint"] == (
        "/World/Looks/Generated_Green_Paint"
    )
    assert apply_mapping["Satin Blue Plastic"] == "/World/Looks/Satin_Blue_Plastic"
    combined_library = Path(apply_mapping["material_library_path"])
    assert combined_library.name == "combined_generated_created_material_library.usda"
    layer = Sdf.Layer.FindOrOpen(str(combined_library))
    assert layer
    assert layer.GetPrimAtPath(Sdf.Path("/World/Looks/Generated_Green_Paint"))
    assert layer.GetPrimAtPath(Sdf.Path("/World/Looks/Satin_Blue_Plastic"))


def test_executor_create_materials_rejects_generated_name_collision(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    generated = CreateMaterialsTask().run(
        {
            "source_usd": str(source_usd),
            "output_dir": str(tmp_path / "generated"),
            "creation_requests": [_creation_request()],
        }
    )
    created = CreateMaterialsTask().run(
        {
            "source_usd": str(source_usd),
            "output_dir": str(tmp_path / "created"),
            "creation_requests": [_creation_request()],
        }
    )

    with pytest.raises(ValueError, match="already exists"):
        UnifiedPipelineExecutorTask()._activate_created_material_library(
            created,
            {
                "working_dir": str(tmp_path / "work"),
                "generated_materials_data": generated["created_materials_data"],
            },
            {"apply": {}},
        )


def test_executor_run_ignores_stale_created_materials_when_rerunning(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    stale_outputs = CreateMaterialsTask().run(
        {
            "source_usd": str(source_usd),
            "output_dir": str(tmp_path / "stale_created"),
            "creation_requests": [_creation_request()],
        }
    )
    pipeline_state = {
        "completed_steps": ["create_materials"],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {"create_materials": stale_outputs},
        "current_step": None,
    }
    context: dict[str, Any] = {
        "steps_to_run": ["create_materials"],
        "working_dir": str(tmp_path / "work"),
        "step_configs": {
            "create_materials": {
                "source_usd": str(source_usd),
                "output_dir": str(tmp_path / "work" / "created_materials"),
                "creation_requests": [_creation_request()],
            },
            "apply": {},
        },
    }

    with patch.object(upe, "_load_pipeline_state", return_value=pipeline_state):
        result = UnifiedPipelineExecutorTask().run(context)

    create_outputs = result["pipeline_results"]["create_materials"]
    assert create_outputs["created_material_count"] == 1
    apply_mapping = result["step_configs"]["apply"]["materials_mapping"]
    assert (
        apply_mapping["material_library_path"]
        == create_outputs["created_material_library_path"]
    )
    current_library = Path(apply_mapping["material_library_path"])
    assert tmp_path / "work" / "created_materials" in current_library.parents
    assert tmp_path / "stale_created" not in current_library.parents
    assert context["step_configs"]["apply"] == {}


def test_executor_created_generated_merge_ignores_unusable_entries(
    tmp_path: Path,
) -> None:
    created_data: dict[str, Any] = {"entries": []}

    result = UnifiedPipelineExecutorTask()._merge_generated_and_created_materials(
        {
            "library_path": "/missing/materials.usda",
            "entries": [
                {"name": FALLBACK_MATERIAL_NAME, "binding": "/World/Looks/Fallback"},
                {"name": 3, "binding": "/World/Looks/Number"},
                {"name": "Missing Binding"},
            ],
        },
        created_data,
        {"working_dir": str(tmp_path / "work")},
        SimpleNamespace(info=lambda *_args, **_kwargs: None),
    )

    assert result is created_data


@pytest.mark.asyncio
async def test_executor_create_materials_rewires_runtime_source_usd_sync_and_async(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    optimized_usd = _write_source_usd(tmp_path / "optimized.usda")
    fixed_usd = _write_source_usd(tmp_path / "fixed.usda")
    executor = UnifiedPipelineExecutorTask()
    sync_config = {
        "source_usd": str(source_usd),
        "output_dir": str(tmp_path / "sync" / "created_materials"),
        "creation_requests": [_creation_request()],
    }

    executor._execute_step(
        "create_materials",
        sync_config,
        {"working_dir": str(tmp_path / "sync")},
        object_store=None,
        pipeline_state={
            "step_outputs": {"optimize_usd": {"optimized_usd_path": str(optimized_usd)}}
        },
    )
    assert sync_config["source_usd"] == str(optimized_usd)

    async_config = {
        "source_usd": str(source_usd),
        "output_dir": str(tmp_path / "async" / "created_materials"),
        "creation_requests": [_creation_request()],
    }
    await executor._aexecute_step(
        "create_materials",
        async_config,
        {"working_dir": str(tmp_path / "async")},
        object_store=None,
        pipeline_state={
            "step_outputs": {
                "validate_input": {"validation_fixed_usd_path": str(fixed_usd)}
            }
        },
    )
    assert async_config["source_usd"] == str(fixed_usd)


@pytest.mark.asyncio
async def test_executor_async_create_materials_runs_in_worker_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    step_config: dict[str, Any] = {
        "source_usd": str(source_usd),
        "output_dir": str(tmp_path / "created_materials"),
        "creation_requests": [_creation_request()],
    }
    to_thread_calls: list[Any] = []

    def execute_create_materials(
        received_config: dict[str, Any],
        _context: dict[str, Any],
    ) -> dict[str, Any]:
        return {"source_usd": received_config["source_usd"]}

    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        to_thread_calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(
        executor, "_execute_create_materials_step", execute_create_materials
    )
    monkeypatch.setattr(upe.asyncio, "to_thread", fake_to_thread)

    result = await executor._aexecute_step(
        "create_materials",
        step_config,
        {"working_dir": str(tmp_path / "work")},
        object_store=None,
        pipeline_state={"step_outputs": {}},
    )

    assert result["source_usd"] == str(source_usd)
    assert to_thread_calls


@pytest.mark.asyncio
async def test_executor_async_cancellation_stops_create_materials_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    worker_started = threading.Event()
    worker_stopped = threading.Event()

    def execute_create_materials(
        _received_config: dict[str, Any],
        received_context: dict[str, Any],
    ) -> dict[str, Any]:
        worker_started.set()
        cancel_checker = received_context["cancel_checker"]
        while not cancel_checker():
            worker_stopped.wait(0.01)
        worker_stopped.set()
        raise RuntimeError("worker stopped after cancellation")

    monkeypatch.setattr(
        executor,
        "_execute_create_materials_step",
        execute_create_materials,
    )
    execution = asyncio.create_task(
        executor._aexecute_step(
            "create_materials",
            {
                "source_usd": str(tmp_path / "asset.usda"),
                "creation_requests": [_creation_request()],
            },
            {"working_dir": str(tmp_path)},
            object_store=None,
            pipeline_state={"step_outputs": {}},
        )
    )
    assert await asyncio.to_thread(worker_started.wait, 1.0)

    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution
    assert worker_stopped.is_set()


def test_executor_create_materials_autowires_prediction_sources(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    working_dir = tmp_path / "work"
    cases = [
        (
            {
                "expand_cluster_predictions": {
                    "predictions_path": "expanded.jsonl",
                }
            },
            "expanded.jsonl",
        ),
        (
            {"validate_predictions": {"predictions_path": "validated.jsonl"}},
            "validated.jsonl",
        ),
        (
            {
                "expand_cluster_predictions": {
                    "predictions_path": "expanded.jsonl",
                },
                "validate_predictions": {"predictions_path": "validated.jsonl"},
            },
            "expanded.jsonl",
        ),
        ({"predict": {"predictions_path": "raw.jsonl"}}, "raw.jsonl"),
        ({"benchmark": {"predictions_path": "benchmark.jsonl"}}, "benchmark.jsonl"),
    ]

    for step_outputs, expected_predictions_path in cases:
        step_config: dict[str, Any] = {}
        executor._autowire_create_materials_step(step_config, step_outputs, working_dir)
        assert step_config["predictions_path"] == expected_predictions_path
        assert step_config["output_predictions_path"] == str(
            working_dir / "created_materials" / "created_predictions.jsonl"
        )

    fallback_config: dict[str, Any] = {}
    executor._autowire_create_materials_step(fallback_config, {}, working_dir)
    assert fallback_config["predictions_path"] == str(
        working_dir / "predictions" / "predictions.jsonl"
    )


def test_executor_run_activates_created_materials(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    context: dict[str, Any] = {
        "steps_to_run": ["create_materials"],
        "working_dir": str(tmp_path / "work"),
        "step_configs": {
            "create_materials": {
                "source_usd": str(source_usd),
                "output_dir": str(tmp_path / "work" / "created_materials"),
                "creation_requests": [_creation_request()],
            },
            "apply": {},
            "refine": {},
        },
    }

    result = UnifiedPipelineExecutorTask().run(context)

    create_outputs = result["pipeline_results"]["create_materials"]
    assert create_outputs["created_material_count"] == 1
    apply_mapping = result["step_configs"]["apply"]["materials_mapping"]
    assert apply_mapping["Satin Blue Plastic"] == "/World/Looks/Satin_Blue_Plastic"
    refine_mapping = result["step_configs"]["refine"]["apply"]["materials_mapping"]
    assert refine_mapping["Satin Blue Plastic"] == "/World/Looks/Satin_Blue_Plastic"
    assert context["step_configs"]["apply"] == {}
    assert context["step_configs"]["refine"] == {}


@pytest.mark.asyncio
async def test_executor_arun_activates_created_materials(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "asset.usda")
    context: dict[str, Any] = {
        "steps_to_run": ["create_materials"],
        "working_dir": str(tmp_path / "work"),
        "step_configs": {
            "create_materials": {
                "source_usd": str(source_usd),
                "output_dir": str(tmp_path / "work" / "created_materials"),
                "creation_requests": [_creation_request()],
            },
            "apply": {},
            "refine": {},
        },
    }

    result = await UnifiedPipelineExecutorTask().arun(context)

    create_outputs = result["pipeline_results"]["create_materials"]
    assert create_outputs["created_material_count"] == 1
    apply_mapping = result["step_configs"]["apply"]["materials_mapping"]
    assert apply_mapping["Satin Blue Plastic"] == "/World/Looks/Satin_Blue_Plastic"
    refine_mapping = result["step_configs"]["refine"]["apply"]["materials_mapping"]
    assert refine_mapping["Satin Blue Plastic"] == "/World/Looks/Satin_Blue_Plastic"
    assert context["step_configs"]["apply"] == {}
    assert context["step_configs"]["refine"] == {}


def _creation_request(
    *,
    material_id: str = "satin_blue_plastic",
    name: str = "Satin Blue Plastic",
    prediction_id: str = "/World/Asset/Housing",
    target_prim_paths: Any = None,
    reference_image_uris: Any = (),
    color: str = "blue",
    material: str = "plastic",
    finish: str = "satin",
) -> dict[str, Any]:
    if target_prim_paths is None:
        target_prim_paths = ["/World/Asset/Housing"]
    return {
        "prediction_id": prediction_id,
        "target_prim_paths": target_prim_paths,
        "reference_image_uris": reference_image_uris,
        "texture_size": 64,
        "seed": 482,
        "recipe": {
            "id": material_id,
            "name": name,
            "description": f"{name} for the requested asset part.",
            "appearance_prompt": f"{name.lower()} with subtle production details",
            "color": color,
            "material": material,
            "finish": finish,
            "base_color_hint": [0.08, 0.18, 0.72],
            "pbr_hints": {"metallic": 0.0, "roughness": 0.42},
            "intended_parts": [
                {
                    "semantic_label": "main housing",
                    "evidence": "WP6 test request",
                    "prim_path_hints": ["/World/Asset/Housing"],
                }
            ],
        },
    }


def _write_source_usd(path: Path) -> Path:
    path.write_text("#usda 1.0\n", encoding="utf-8")
    return path.resolve()


def _write_conditioned_source_usd(path: Path) -> Path:
    path.write_text(
        """#usda 1.0

def Xform "World"
{
    def Xform "Asset"
    {
        def Mesh "Housing"
        {
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [
                (-0.5, -0.5, 0.0),
                (0.5, -0.5, 0.0),
                (0.5, 0.5, 0.0),
                (-0.5, 0.5, 0.0)
            ]
            texCoord2f[] primvars:st = [
                (0, 0), (1, 0), (1, 1), (0, 1)
            ] (
                interpolation = "faceVarying"
            )
        }
    }
}
""",
        encoding="utf-8",
    )
    return path.resolve()


def _temporary_real_conditioning_config(
    tmp_path: Path,
    source_usd: Path,
) -> dict[str, Any]:
    # This temporary OVRTX-shaped payload tests the contract; it is not acceptance evidence.
    seed_dir = tmp_path / "temporary_seed_package"
    texture_dir = seed_dir / "textures"
    texture_dir.mkdir(parents=True)
    source_albedo = texture_dir / "source_albedo.png"
    source_image = Image.new("RGB", (2, 2), (120, 130, 140))
    source_image.putpixel((1, 1), (121, 130, 140))
    source_image.save(source_albedo)
    material_usd = seed_dir / "seed_material.usda"
    material_usd.write_text(
        """#usda 1.0
(
    defaultPrim = "Seed"
)
def Scope "Seed"
{
    def Material "Material"
    {
        token outputs:surface.connect = </Seed/Material/Surface.outputs:surface>
        def Shader "UVReader"
        {
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }
        def Shader "Albedo"
        {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @textures/source_albedo.png@
            float2 inputs:st.connect = </Seed/Material/UVReader.outputs:result>
            float3 outputs:rgb
        }
        def Shader "Surface"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </Seed/Material/Albedo.outputs:rgb>
            float inputs:metallic = 1
            float inputs:roughness = 0.4
            token outputs:surface
        }
    }
}
""",
        encoding="utf-8",
    )
    seed_manifest = seed_dir / "seed_manifest.json"
    seed_manifest.write_text(
        json.dumps(
            {
                "schema_version": REAL_SEED_MATERIAL_SCHEMA_VERSION,
                "package_id": "temporary-task-contract-seed",
                "package_revision": "test-only-revision",
                "material_usd": {
                    "path": material_usd.name,
                    "sha256": _sha256_file(material_usd),
                    "material_prim_path": "/Seed/Material",
                },
                "source_albedo": {
                    "path": "textures/source_albedo.png",
                    "sha256": _sha256_file(source_albedo),
                    "color_space": "srgb",
                    "width": 2,
                    "height": 2,
                },
                "source": {
                    "kind": "approved_s3",
                    "uri": "s3://test-only/source-albedo.png",
                    "etag": "test-only-etag",
                    "last_modified": "2026-06-30T00:00:00Z",
                    "content_type": "image/png",
                    "byte_size": source_albedo.stat().st_size,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    seed_manifest_sha256 = _sha256_file(seed_manifest)
    evidence_dir = tmp_path / "temporary_ovrtx_contract_fixture"
    evidence_dir.mkdir(exist_ok=True)
    render_path = evidence_dir / "front.png"
    Image.new("RGB", (2, 2), (70, 80, 90)).save(render_path)
    request_payload = {
        "source_usd_sha256": _sha256_file(source_usd),
        "seed_manifest_sha256": seed_manifest_sha256,
        "target_prim_paths": ["/World/Asset/Housing"],
        "renderer": {"backend": "ovrtx", "image_width": 512, "image_height": 512},
    }
    request_id = create_materials_task._build_create_request(
        _creation_request(),
        source_usd,
        "step1x_material_anything",
        base_dir=None,
    ).request_id
    request_manifest = evidence_dir / "request_manifest.json"
    request_manifest.write_text(
        json.dumps(
            {
                "schema_version": OVRTX_CONDITIONING_SCHEMA_VERSION,
                "provider": "ovrtx",
                "provider_revision": "temporary-task-contract-fixture",
                "request_id": request_id,
                "simulate": False,
                "request": request_payload,
                "request_sha256": _sha256_json(request_payload),
                "artifacts": [
                    {
                        "kind": "render",
                        "path": render_path.name,
                        "view": "front",
                        "sha256": _sha256_file(render_path),
                        "evidence_source": "renderer_derived",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "evidence_mode": "real_evidence",
        "render_views": [],
        "include_normal": False,
        "include_depth": False,
        "include_segmentation": False,
        "include_source_albedo": True,
        "real_evidence": {
            "seed_manifest_path": seed_manifest.as_posix(),
            "seed_manifest_sha256": seed_manifest_sha256,
            "ovrtx_manifest_path": request_manifest.as_posix(),
        },
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fake_created_material(material_usd_path: Path, material_prim_path: str) -> Any:
    return SimpleNamespace(
        material_usd_path=material_usd_path,
        material_prim_path=material_prim_path,
    )


class _Step1XDispatchStubBackend(RealFakeMaterialCreationBackend):
    def __init__(self) -> None:
        super().__init__()
        self.conditioning: Any | None = None

    @property
    def name(self) -> str:
        return "step1x_material_anything"

    @property
    def revision(self) -> str:
        return "step1x-dispatch-stub.v1"

    def create(
        self,
        request: Any,
        *,
        output_dir: Path,
        conditioning: Any | None = None,
        cancel_event: Any | None = None,
    ) -> BackendMaterialResult:
        if conditioning is None:
            raise MaterialCreationError(
                MaterialCreationErrorCode.INVALID_REQUEST,
                "Step1X dispatch stub requires prepared conditioning",
                backend=self.name,
            )
        conditioning.validate_request(request)
        self.conditioning = conditioning
        result = super().create(
            request,
            output_dir=output_dir,
            conditioning=conditioning,
            cancel_event=cancel_event,
        )
        provenance = MaterialCreationProvenance.for_request(
            request,
            backend=self.name,
            backend_revision=self.revision,
            model_revisions=("step1x-dispatch-stub-model",),
            duration_seconds=0.0,
            conditioning=conditioning,
        )
        return BackendMaterialResult(
            artifacts=result.artifacts,
            provenance=provenance,
            degradations=result.degradations,
            diagnostics=result.diagnostics,
            preview_paths=result.preview_paths,
        )


class _Step1XWaitForCancelBackend(_Step1XDispatchStubBackend):
    def create(
        self,
        request: Any,
        *,
        output_dir: Path,
        conditioning: Any | None = None,
        cancel_event: threading.Event | None = None,
    ) -> BackendMaterialResult:
        if conditioning is None:
            raise MaterialCreationError(
                MaterialCreationErrorCode.INVALID_REQUEST,
                "Step1X dispatch stub requires prepared conditioning",
                backend=self.name,
            )
        if cancel_event is not None and cancel_event.wait(timeout=2.0):
            raise MaterialCreationError(
                MaterialCreationErrorCode.CANCELLED,
                "Step1X dispatch stub cancelled during execution",
                backend=self.name,
            )
        raise AssertionError("cancel checker was not bridged to the backend")


@dataclass(frozen=True)
class _DummyStep1XRuntimeConfig:
    runtime_dir: Path | None = None
    model_dir: Path | None = None
    cache_dir: Path | None = None
    output_dir: Path | None = None
    python_executable: Path | None = None
    edit_script: Path | None = None
    command_template: str | None = None
    timeout_sec: int = 120
    validate_assets: bool = True
    skip_material_anything: bool = False
    require_upscaler: bool = False
    extra_args: tuple[str, ...] = ()
    required_executables: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> _DummyStep1XRuntimeConfig:
        return cls()


class _InvalidStep1XRuntimeConfig:
    @classmethod
    def from_env(cls) -> _InvalidStep1XRuntimeConfig:
        raise ValueError("invalid runtime")


def _write_predictions(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "id": "/World/Asset/Housing",
                "materials": {
                    "material": "__CREATE_NEW__",
                    "reason": "No supplied material matches satin blue plastic.",
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
