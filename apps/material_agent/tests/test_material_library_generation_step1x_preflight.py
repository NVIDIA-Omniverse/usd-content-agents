# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Step1X real-smoke preflight harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from apps.texture_gen_step1x_service.backend import Step1XBackendConfig

from material_agent.material_library_generation import step1x_preflight


def test_preflight_reports_material_anything_and_runtime_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(step1x_preflight, "Step1XBackend", _StubStep1XBackend)
    monkeypatch.setattr(step1x_preflight, "ExternalStep1XRunner", _StubRunner)

    report = step1x_preflight.preflight_step1x_runtime(
        Step1XBackendConfig(
            runtime_dir=tmp_path / "runtime",
            model_dir=tmp_path / "models",
            python_executable=tmp_path / "python",
            skip_material_anything=True,
        ),
        require_material_anything=True,
        require_gpu=True,
    )

    assert report.ready is False
    assert report.status == "blocked"
    codes = {issue.code for issue in report.issues}
    assert "material_anything_default_disabled" in codes
    assert "material_anything_unavailable" in codes
    assert "model_or_checkpoint_missing" in codes
    assert "python_or_executable_missing" in codes
    assert "gpu_cuda_unavailable" in codes
    default_warning = next(
        issue
        for issue in report.issues
        if issue.code == "material_anything_default_disabled"
    )
    assert default_warning.severity == "warning"
    assert report.categories["material_anything"]["default_enabled"] is False
    assert report.categories["material_anything"]["effective_enabled"] is True
    assert report.categories["material_anything"]["enabled"] is True
    assert report.categories["material_anything"]["ready"] is False
    assert report.categories["command_template"]["rendered"] is True
    payload = report.to_dict()
    assert payload["command"] == [
        "python",
        "edit_texture.py",
        "--api-key",
        "<redacted>",
        "--password=<redacted>",
    ]
    assert payload["categories"]["command_template"]["command_preview"] == [
        "python",
        "edit_texture.py",
        "--api-key",
        "<redacted>",
        "--password=<redacted>",
    ]


def test_preflight_default_skip_ma_is_not_a_blocker_when_material_anything_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        step1x_preflight,
        "Step1XBackend",
        _ready_backend_class(gpu_available=True),
    )
    monkeypatch.setattr(step1x_preflight, "ExternalStep1XRunner", _StubRunner)

    report = step1x_preflight.preflight_step1x_runtime(
        Step1XBackendConfig(
            runtime_dir=tmp_path / "runtime",
            model_dir=tmp_path / "models",
            python_executable=tmp_path / "python",
            skip_material_anything=True,
        ),
        require_material_anything=True,
        require_gpu=True,
    )

    assert report.ready is True
    assert report.status == "ready"
    assert report.categories["material_anything"]["default_enabled"] is False
    assert report.categories["material_anything"]["effective_enabled"] is True
    assert {issue.code: issue.severity for issue in report.issues} == {
        "material_anything_default_disabled": "warning"
    }


def test_preflight_command_template_requires_material_anything_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        step1x_preflight,
        "Step1XBackend",
        _command_template_missing_ma_backend_class(gpu_available=True),
    )
    monkeypatch.setattr(step1x_preflight, "ExternalStep1XRunner", _StubRunner)

    report = step1x_preflight.preflight_step1x_runtime(
        Step1XBackendConfig(
            command_template="echo {prompt}",
            runtime_dir=tmp_path / "runtime",
            skip_material_anything=True,
        ),
        require_material_anything=True,
        require_gpu=True,
    )

    assert report.ready is False
    assert report.status == "blocked"
    codes = {issue.code for issue in report.issues}
    assert "material_anything_default_disabled" in codes
    assert "material_anything_unavailable" in codes
    assert "model_or_checkpoint_missing" not in codes
    assert "python_or_executable_missing" not in codes
    assert report.categories["material_anything"]["default_enabled"] is False
    assert report.categories["material_anything"]["effective_enabled"] is True
    assert report.categories["material_anything"]["ready"] is False
    blockers = {issue.code: issue for issue in report.blockers}
    assert blockers["material_anything_unavailable"].detail == {
        "missing": ["Material Anything material_estimator missing."]
    }


def test_redact_command_preview_handles_provider_specific_secret_names() -> None:
    assert step1x_preflight._redact_command_preview(
        [
            "python",
            "edit_texture.py",
            "NVCF_API_KEY=<placeholder-value>",
            "OPENAI_API_KEY=<placeholder-value>",
            "--hf-token",
            "<placeholder-value>",
            "--provider-password-file=<placeholder-path>",
            "--visible",
            "kept",
        ]
    ) == [
        "python",
        "edit_texture.py",
        "NVCF_API_KEY=<redacted>",
        "OPENAI_API_KEY=<redacted>",
        "--hf-token",
        "<redacted>",
        "--provider-password-file=<redacted>",
        "--visible",
        "kept",
    ]


def test_preflight_can_allow_material_anything_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        step1x_preflight,
        "Step1XBackend",
        _ready_backend_class(gpu_available=True),
    )
    monkeypatch.setattr(step1x_preflight, "ExternalStep1XRunner", _StubRunner)

    report = step1x_preflight.preflight_step1x_runtime(
        Step1XBackendConfig(
            runtime_dir=tmp_path / "runtime",
            model_dir=tmp_path / "models",
            python_executable=tmp_path / "python",
            skip_material_anything=True,
        ),
        require_material_anything=False,
        require_gpu=True,
    )

    assert report.ready is True
    assert report.status == "ready"
    assert report.categories["material_anything"]["required"] is False
    assert report.categories["gpu_cuda"]["gpu_available"] is True


def test_preflight_warns_when_gpu_unknown_and_not_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        step1x_preflight,
        "Step1XBackend",
        _ready_backend_class(gpu_available=None),
    )
    monkeypatch.setattr(step1x_preflight, "ExternalStep1XRunner", _StubRunner)

    report = step1x_preflight.preflight_step1x_runtime(
        Step1XBackendConfig(
            runtime_dir=tmp_path / "runtime",
            model_dir=tmp_path / "models",
            python_executable=tmp_path / "python",
            skip_material_anything=False,
        ),
        require_material_anything=True,
        require_gpu=False,
    )

    assert report.ready is True
    assert {issue.code: issue.severity for issue in report.issues} == {
        "gpu_cuda_unknown": "warning"
    }


def test_single_material_smoke_is_blocked_without_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        step1x_preflight,
        "preflight_step1x_runtime",
        lambda *_args, **_kwargs: step1x_preflight.Step1XPreflightReport(
            ready=True,
            status="ready",
            categories={},
            issues=(),
            health={},
        ),
    )

    payload = step1x_preflight.run_single_material_smoke(
        request=object(),  # type: ignore[arg-type]
        conditioning=object(),  # type: ignore[arg-type]
        output_dir=tmp_path,
        config=Step1XBackendConfig(),
        run_real_step1x=False,
        require_gpu=False,
    )

    assert payload["status"] == "blocked"
    assert payload["blockers"][0]["code"] == "real_smoke_not_explicitly_enabled"


def test_single_material_smoke_returns_preflight_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocker = step1x_preflight.Step1XPreflightIssue(
        category=step1x_preflight.Step1XPreflightCategory.RUNTIME,
        code="runtime_missing",
        message="missing runtime",
    )
    monkeypatch.setattr(
        step1x_preflight,
        "preflight_step1x_runtime",
        lambda *_args, **_kwargs: step1x_preflight.Step1XPreflightReport(
            ready=False,
            status="blocked",
            categories={},
            issues=(blocker,),
            health={},
        ),
    )

    payload = step1x_preflight.run_single_material_smoke(
        request=object(),  # type: ignore[arg-type]
        conditioning=object(),  # type: ignore[arg-type]
        output_dir=tmp_path,
        config=Step1XBackendConfig(),
        run_real_step1x=True,
        require_gpu=False,
    )

    assert payload["status"] == "blocked"
    assert payload["blockers"][0]["code"] == "runtime_missing"


def test_single_material_smoke_runs_real_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        step1x_preflight,
        "preflight_step1x_runtime",
        lambda *_args, **_kwargs: step1x_preflight.Step1XPreflightReport(
            ready=True,
            status="ready",
            categories={},
            issues=(),
            health={},
        ),
    )
    monkeypatch.setattr(step1x_preflight, "result_fingerprint", lambda _result: "fp")

    class FakeToDict:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload

        def to_dict(self) -> dict[str, Any]:
            return self.payload

    class FakeResult:
        artifacts = (FakeToDict({"artifact": "ok"}),)
        degradations = (FakeToDict({"degradation": "ok"}),)
        provenance = FakeToDict({"provenance": "ok"})

    class FakeRunner:
        def __init__(self, config: Step1XBackendConfig) -> None:
            assert config.skip_material_anything is False

    class FakeBackend:
        def __init__(self, *, config: Any, runner: FakeRunner) -> None:
            self.config = config
            self.runner = runner

        def create(
            self,
            request: Any,
            *,
            output_dir: Path,
            conditioning: Any,
            cancel_event: Any,
        ) -> FakeResult:
            assert output_dir == tmp_path.resolve()
            assert conditioning == "conditioning"
            assert cancel_event.is_set() is False
            return FakeResult()

    monkeypatch.setattr(step1x_preflight, "ExternalStep1XRunner", FakeRunner)
    monkeypatch.setattr(
        step1x_preflight,
        "Step1XMaterialCreationBackend",
        FakeBackend,
    )

    request = FakeToDict({"request": "ok"})
    payload = step1x_preflight.run_single_material_smoke(
        request=request,  # type: ignore[arg-type]
        conditioning="conditioning",  # type: ignore[arg-type]
        output_dir=tmp_path,
        config=Step1XBackendConfig(skip_material_anything=True),
        run_real_step1x=True,
        require_gpu=False,
    )

    assert payload["status"] == "completed"
    assert payload["request"] == {"request": "ok"}
    assert payload["artifacts"] == [{"artifact": "ok"}]
    assert payload["degradations"] == [{"degradation": "ok"}]
    assert payload["provenance"] == {"provenance": "ok"}
    assert payload["fingerprint"] == "fp"


def test_load_request_and_conditioning_resolve_local_paths(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "source_usd": source.name,
                "target_prim_paths": ["/World/Asset"],
                "recipe": {
                    "name": "Blue Plastic",
                    "description": "blue plastic",
                    "appearance_prompt": "blue plastic",
                },
                "reference_image_uris": ["refs/reference.png"],
            }
        ),
        encoding="utf-8",
    )
    conditioning_path = tmp_path / "conditioning.json"
    conditioning_path.write_text(
        json.dumps(
            {
                "request_id": "mc_000000000000000000000000",
                "target_prim_paths": ["/World/Asset"],
                "reference_image_uris": ["refs/conditioning.png"],
                "artifacts": [
                    {
                        "kind": "scoped_usd",
                        "uri": source.name,
                        "view": "front",
                        "sha256": "0" * 64,
                    },
                    {
                        "kind": "reference_image",
                        "uri": "refs/conditioning.png",
                        "color_space": "raw",
                        "sha256": "1" * 64,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    request = step1x_preflight.load_create_material_request(request_path)
    conditioning = step1x_preflight.load_prepared_conditioning(conditioning_path)

    assert request.source_usd == source.resolve()
    assert request.backend == "step1x_material_anything"
    assert request.reference_image_uris == (
        str((tmp_path / "refs" / "reference.png").resolve()),
    )
    assert conditioning.artifacts[0].uri == str(source.resolve())
    assert conditioning.artifacts[0].color_space is None
    assert conditioning.artifacts[1].uri == str(
        (tmp_path / "refs" / "conditioning.png").resolve()
    )
    assert conditioning.artifacts[1].color_space is not None
    assert conditioning.artifacts[1].color_space.value == "raw"
    assert conditioning.reference_image_uris == (
        str((tmp_path / "refs" / "conditioning.png").resolve()),
    )


def test_preflight_private_helper_branches(tmp_path: Path) -> None:
    object_json = tmp_path / "object.json"
    object_json.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="must contain a JSON object"):
        step1x_preflight._load_json_object(object_json)

    with pytest.raises(TypeError, match="recipe must be a JSON object"):
        step1x_preflight._require_dict({"recipe": []}, "recipe")

    local = tmp_path / "local.usda"
    assert (
        step1x_preflight._resolve_local_reference(
            "https://example.test/a.png", tmp_path
        )
        == "https://example.test/a.png"
    )
    assert step1x_preflight._resolve_local_reference(str(local), tmp_path) == str(local)
    assert step1x_preflight._resolve_local_reference("local.usda", tmp_path) == str(
        local.resolve()
    )
    assert step1x_preflight._resolve_local_path(local.as_uri(), tmp_path) == local
    assert step1x_preflight._resolve_local_path(str(local), tmp_path) == local
    assert step1x_preflight._resolve_local_path("local.usda", tmp_path) == local
    with pytest.raises(ValueError, match="source_usd must be a local path"):
        step1x_preflight._resolve_local_path(
            "https://example.test/source.usda", tmp_path
        )

    assert (
        step1x_preflight._issue_from_missing_input(
            "Material Anything assets missing"
        ).code
        == "material_anything_unavailable"
    )
    assert step1x_preflight._issue_from_missing_input("other missing").code == (
        "runtime_missing"
    )
    assert step1x_preflight._material_anything_capability_ready({}) is True
    assert step1x_preflight._is_sensitive_command_name("--token") is True


def test_probe_command_readiness_reports_empty_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyRunner:
        def __init__(self, _config: Step1XBackendConfig) -> None:
            pass

        def _build_command(self, *_args: Any, **_kwargs: Any) -> list[str]:
            return []

    monkeypatch.setattr(step1x_preflight, "ExternalStep1XRunner", EmptyRunner)

    command, issue = step1x_preflight._probe_command_readiness(
        Step1XBackendConfig(),
        [],
    )

    assert command == []
    assert issue is not None
    assert issue.code == "command_empty"


def test_main_defaults_to_preflight_without_subcommand(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(step1x_preflight, "Step1XBackend", _StubStep1XBackend)
    monkeypatch.setattr(step1x_preflight, "ExternalStep1XRunner", _StubRunner)

    assert step1x_preflight.main([]) == 2

    payload = capsys.readouterr().out
    assert '"status": "blocked"' in payload
    assert "model_or_checkpoint_missing" in payload


def test_main_smoke_without_real_opt_in_does_not_load_request_files(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(step1x_preflight, "Step1XBackend", _StubStep1XBackend)
    monkeypatch.setattr(step1x_preflight, "ExternalStep1XRunner", _StubRunner)

    assert step1x_preflight.main(["smoke"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["blockers"][0]["code"] == "real_smoke_not_explicitly_enabled"


def test_main_smoke_with_real_opt_in_requires_input_files() -> None:
    with pytest.raises(SystemExit) as exc_info:
        step1x_preflight.main(["smoke", "--run-real-step1x"])
    assert exc_info.value.code == 2


def test_main_smoke_with_real_opt_in_loads_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path = tmp_path / "request.json"
    conditioning_path = tmp_path / "conditioning.json"
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        step1x_preflight,
        "load_create_material_request",
        lambda path: {"request_path": path},  # type: ignore[return-value]
    )
    monkeypatch.setattr(
        step1x_preflight,
        "load_prepared_conditioning",
        lambda path: {"conditioning_path": path},  # type: ignore[return-value]
    )
    monkeypatch.setattr(
        step1x_preflight,
        "run_single_material_smoke",
        lambda request, conditioning, **kwargs: {
            "status": "completed",
            "request": request["request_path"].name,
            "conditioning": conditioning["conditioning_path"].name,
            "output_dir": kwargs["output_dir"].name,
        },
    )

    assert (
        step1x_preflight.main(
            [
                "smoke",
                "--run-real-step1x",
                "--request",
                str(request_path),
                "--conditioning",
                str(conditioning_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "completed",
        "request": "request.json",
        "conditioning": "conditioning.json",
        "output_dir": "out",
    }


class _ModelDump:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def model_dump(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return self.payload


class _Health(_ModelDump):
    def __init__(self, *, gpu_available: bool | None) -> None:
        super().__init__({"gpu_available": gpu_available})
        self.gpu_available = gpu_available


class _StubStep1XBackend:
    def __init__(self, config: Step1XBackendConfig) -> None:
        self.config = config

    def health(self) -> _Health:
        return _Health(gpu_available=False)

    def capabilities(self) -> _ModelDump:
        return _ModelDump(
            {
                "external_runtime": {
                    "runtime_dir": str(self.config.runtime_dir or ""),
                    "edit_script": str(self.config.edit_script or ""),
                    "runtime_source": "test",
                    "model_dir": str(self.config.model_dir or ""),
                    "weights_policy": "test",
                    "required_executables": list(self.config.required_executables),
                },
                "material_anything": {
                    "available": False,
                },
            }
        )

    def _missing_runtime_inputs(self) -> list[str]:
        return [
            "TEXTURE_STEP1X_MODEL_DIR is required.",
            "TEXTURE_STEP1X_PYTHON is required.",
        ]


def _ready_backend_class(*, gpu_available: bool | None) -> type[_StubStep1XBackend]:
    class ReadyBackend(_StubStep1XBackend):
        def health(self) -> _Health:
            return _Health(gpu_available=gpu_available)

        def capabilities(self) -> _ModelDump:
            payload = super().capabilities().model_dump()
            payload["material_anything"] = {"available": True}
            return _ModelDump(payload)

        def _missing_runtime_inputs(self) -> list[str]:
            return []

    return ReadyBackend


def _command_template_missing_ma_backend_class(
    *,
    gpu_available: bool | None,
) -> type[_StubStep1XBackend]:
    class CommandTemplateMissingMABackend(_StubStep1XBackend):
        def health(self) -> _Health:
            return _Health(gpu_available=gpu_available)

        def capabilities(self) -> _ModelDump:
            payload = super().capabilities().model_dump()
            payload["material_anything"] = {
                "ready": False,
                "missing": ["Material Anything material_estimator missing."],
            }
            return _ModelDump(payload)

        def _missing_runtime_inputs(self) -> list[str]:
            return []

    return CommandTemplateMissingMABackend


class _StubRunner:
    def __init__(self, _config: Step1XBackendConfig) -> None:
        pass

    def _build_command(self, *_args: Any, **_kwargs: Any) -> list[str]:
        return [
            "python",
            "edit_texture.py",
            "--api-key",
            "TEST_API_KEY_PLACEHOLDER",
            "--password=TEST_PASSWORD_PLACEHOLDER",
        ]
