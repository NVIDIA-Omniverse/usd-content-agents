# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for no-mock/no-fake real-smoke guardrails."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import material_agent.config.real_smoke_guardrails as real_smoke_guardrails
from material_agent.config.real_smoke_guardrails import (
    RealSmokeGuardrailError,
    collect_real_smoke_disqualifiers,
    validate_real_smoke_guardrails,
)


def _real_config() -> dict:
    return {
        "project": {"name": "real-smoke"},
        "input": {"usd_path": "ladder.usd"},
        "materials": {"path": "materials.yaml"},
        "steps": {
            "build_dataset_usd": {
                "enabled": True,
                "renderer": {"backend": "remote"},
            },
            "predict": {
                "enabled": True,
                "vlm": {"backend": "nim", "model": "qwen/example"},
                "llm": {"backend": "nim", "model": "qwen/example"},
            },
            "cluster_prims": {
                "enabled": True,
                "embedding_service": "nim",
            },
            "create_materials": {
                "enabled": False,
                "backend": "fake",
                "fake_behavior": "success",
            },
            "render": {
                "enabled": True,
                "backend": "remote",
            },
        },
    }


def test_real_smoke_guardrails_accept_real_backends_without_credentials() -> None:
    validate_real_smoke_guardrails(_real_config())


def test_real_smoke_guardrails_reject_simulate_mode_flag() -> None:
    with pytest.raises(RealSmokeGuardrailError) as exc_info:
        validate_real_smoke_guardrails(_real_config(), simulate=True)

    assert "simulate_mode" in str(exc_info.value)


def test_real_smoke_guardrails_reject_mock_backend_without_printing_secret() -> None:
    config = _real_config()
    config["steps"]["predict"]["vlm"] = {
        "backend": "mock",
        "api_key": "sk-do-not-print",
    }

    with pytest.raises(RealSmokeGuardrailError) as exc_info:
        validate_real_smoke_guardrails(config)

    message = str(exc_info.value)
    assert "mock_backend" in message
    assert "steps.predict.vlm.backend" in message
    assert "sk-do-not-print" not in message


def test_real_smoke_guardrails_ignore_non_string_backend_value() -> None:
    config = _real_config()
    config["steps"]["predict"]["vlm"]["backend"] = {"name": "nim"}

    validate_real_smoke_guardrails(config)


def test_real_smoke_guardrails_reject_enabled_fake_create_materials() -> None:
    config = _real_config()
    config["steps"]["create_materials"]["enabled"] = True

    disqualifiers = collect_real_smoke_disqualifiers(config)

    codes = {finding.code for finding in disqualifiers}
    assert "fake_backend" in codes
    assert "fake_material_creation_behavior" in codes


@pytest.mark.parametrize("backend_alias", ["auto", "auto-for-test"])
def test_real_smoke_guardrails_reject_fake_create_materials_aliases(
    backend_alias: str,
) -> None:
    config = _real_config()
    create_materials = config["steps"]["create_materials"]
    create_materials["enabled"] = True
    create_materials["backend"] = backend_alias
    create_materials.pop("fake_behavior")

    disqualifiers = collect_real_smoke_disqualifiers(config)

    alias_finding = next(
        finding for finding in disqualifiers if finding.code == "fake_backend_alias"
    )
    assert alias_finding.location == "steps.create_materials.backend"
    assert backend_alias in alias_finding.message


def test_real_smoke_guardrails_reject_default_fake_create_materials_backend() -> None:
    config = _real_config()
    create_materials = config["steps"]["create_materials"]
    create_materials["enabled"] = True
    create_materials.pop("backend")
    create_materials.pop("fake_behavior")

    disqualifiers = collect_real_smoke_disqualifiers(config)

    default_finding = next(
        finding for finding in disqualifiers if finding.code == "fake_backend_default"
    )
    assert default_finding.location == "steps.create_materials.backend"
    assert "defaults to fake" in default_finding.message


def test_real_smoke_guardrails_respect_scheduled_steps() -> None:
    config = _real_config()
    config["steps"]["predict"]["vlm"]["backend"] = "mock"

    validate_real_smoke_guardrails(config, scheduled_steps={"render"})


def test_real_smoke_guardrails_reject_generated_config_backend_declaration(
    tmp_path: Path,
) -> None:
    generated_config = tmp_path / "predict_step.yaml"
    generated_config.write_text(
        """
vlm:
  backend: mock
  api_key: sk-do-not-print
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(RealSmokeGuardrailError) as exc_info:
        validate_real_smoke_guardrails(
            _real_config(), artifact_paths=[generated_config]
        )

    message = str(exc_info.value)
    assert "artifact_mock_or_fake_backend" in message
    assert "sk-do-not-print" not in message


def test_real_smoke_guardrails_reject_jsonl_artifact_backend_declaration(
    tmp_path: Path,
) -> None:
    generated_config = tmp_path / "predict_step.jsonl"
    generated_config.write_text('{"vlm": {"backend": "mock"}}\n', encoding="utf-8")

    with pytest.raises(RealSmokeGuardrailError) as exc_info:
        validate_real_smoke_guardrails(
            _real_config(), artifact_paths=[generated_config]
        )

    assert "artifact_mock_or_fake_backend" in str(exc_info.value)


def test_real_smoke_guardrails_ignore_unparseable_structured_artifact(
    tmp_path: Path,
) -> None:
    generated_config = tmp_path / "predict_step.json"
    generated_config.write_text("{", encoding="utf-8")

    validate_real_smoke_guardrails(_real_config(), artifact_paths=[generated_config])


def test_real_smoke_guardrails_accept_plain_artifact_without_markers(
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    log.write_text("completed real run\n", encoding="utf-8")

    validate_real_smoke_guardrails(_real_config(), artifact_paths=[log])


def test_real_smoke_guardrails_reports_root_string_fake_backend(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "material_creation_status.yaml"
    manifest.write_text("FakeMaterialCreationBackend\n", encoding="utf-8")

    with pytest.raises(RealSmokeGuardrailError) as exc_info:
        validate_real_smoke_guardrails(_real_config(), artifact_paths=[manifest])

    message = str(exc_info.value)
    assert "fake_material_creation_backend" in message
    assert "<root>" in message


def test_real_smoke_guardrails_formats_list_locations(tmp_path: Path) -> None:
    manifest = tmp_path / "material_creation_status.json"
    manifest.write_text('[{"backend": "fake"}]\n', encoding="utf-8")

    with pytest.raises(RealSmokeGuardrailError) as exc_info:
        validate_real_smoke_guardrails(_real_config(), artifact_paths=[manifest])

    assert "[0].backend" in str(exc_info.value)


def test_real_smoke_guardrails_allow_structured_artifact_simulate_false(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "run_metadata.yaml"
    metadata.write_text(
        """
simulate: false
status: completed
""".strip(),
        encoding="utf-8",
    )

    validate_real_smoke_guardrails(_real_config(), artifact_paths=[metadata])


def test_real_smoke_guardrails_reject_structured_artifact_simulate_true(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "run_metadata.yaml"
    metadata.write_text(
        """
simulate: true
status: completed
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(RealSmokeGuardrailError) as exc_info:
        validate_real_smoke_guardrails(_real_config(), artifact_paths=[metadata])

    assert "artifact_simulate_mode" in str(exc_info.value)
    assert "simulate_mode" in str(exc_info.value)


def test_real_smoke_guardrails_reject_backend_class_fake_manifest(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "material_creation_status.json"
    manifest.write_text(
        """
{
  "backend_class": "fake"
}
""".strip(),
        encoding="utf-8",
    )

    disqualifiers = collect_real_smoke_disqualifiers(
        _real_config(), artifact_paths=[manifest]
    )

    codes = {finding.code for finding in disqualifiers}
    assert "artifact_mock_or_fake_backend" in codes
    assert "fake_backend" in codes


def test_real_smoke_guardrails_reject_fake_material_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "material_creation_status.json"
    manifest.write_text(
        """
{
  "backend": "fake",
  "backend_class": "FakeMaterialCreationBackend",
  "provenance": {
    "backend_revision": "fake-material-backend.v1"
  }
}
""".strip(),
        encoding="utf-8",
    )

    disqualifiers = collect_real_smoke_disqualifiers(
        _real_config(), artifact_paths=[manifest]
    )

    codes = {finding.code for finding in disqualifiers}
    assert "artifact_mock_or_fake_backend" in codes
    assert "artifact_fake_backend" in codes
    assert "fake_material_creation_backend" in codes
    assert (
        sum(
            finding.code == "fake_material_creation_backend"
            for finding in disqualifiers
        )
        == 1
    )


def test_real_smoke_guardrails_fail_closed_for_missing_artifact(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.log"

    with pytest.raises(RealSmokeGuardrailError) as exc_info:
        validate_real_smoke_guardrails(_real_config(), artifact_paths=[missing])

    assert "missing_artifact" in str(exc_info.value)


def test_real_smoke_guardrails_fail_closed_for_unsupported_artifact_file(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "preview.png"
    artifact.write_bytes(b"not scanned")

    with pytest.raises(RealSmokeGuardrailError) as exc_info:
        validate_real_smoke_guardrails(_real_config(), artifact_paths=[artifact])

    assert "missing_evidence" in str(exc_info.value)


def test_real_smoke_guardrails_fail_closed_for_unreadable_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "run.log"
    artifact.write_text("completed\n", encoding="utf-8")
    original_read_text = Path.read_text

    def raise_for_artifact(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == artifact:
            raise OSError("cannot read")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raise_for_artifact)

    with pytest.raises(RealSmokeGuardrailError) as exc_info:
        validate_real_smoke_guardrails(_real_config(), artifact_paths=[artifact])

    assert "unreadable_artifact" in str(exc_info.value)


def test_real_smoke_guardrails_skips_non_file_scan_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    monkeypatch.setattr(
        real_smoke_guardrails,
        "_scan_files",
        lambda path: (artifact_dir,),
    )

    validate_real_smoke_guardrails(_real_config(), artifact_paths=[artifact_dir])


def test_real_smoke_guardrails_fail_closed_for_empty_artifact_directory(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "preview.png").write_bytes(b"not scanned")

    with pytest.raises(RealSmokeGuardrailError) as exc_info:
        validate_real_smoke_guardrails(_real_config(), artifact_paths=[artifacts])

    assert "missing_evidence" in str(exc_info.value)
