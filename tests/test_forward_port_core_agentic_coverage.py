# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused regression coverage for forward-ported core hardening branches."""

from __future__ import annotations

import builtins
import errno
import logging
import os
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NoReturn

import numpy as np
import pytest
from filelock import Timeout

import world_understanding.agentic.base_pipeline_executor as executor_module
import world_understanding.agentic.config.context_loader as context_loader
import world_understanding.agentic.config.isolation as isolation
import world_understanding.agentic.config.model_credentials as model_credentials
import world_understanding.agentic.config.unknown_keys as unknown_keys
import world_understanding.agentic.session as session_module
import world_understanding.agentic.usd_tasks.config as usd_config_module
import world_understanding.agentic.usd_tasks.config_optimize_usd as optimize_config_module
import world_understanding.agentic.usd_tasks.config_restore_usd as restore_config_module
import world_understanding.agentic.usd_tasks.optimize_usd as optimize_module
import world_understanding.functions.classification.inference as inference_module
import world_understanding.functions.graphics.render_warp as render_warp
import world_understanding.functions.graphics.usd_scene_analysis as scene_analysis
import world_understanding.utils.artifacts as artifacts
import world_understanding.utils.result_projection as result_projection
from world_understanding.agentic.cli.ingress import _normalize_step_filter
from world_understanding.agentic.events import CollectingEventListener
from world_understanding.utils.llm_parsing import iter_json_dicts_in_text_order
from world_understanding.utils.model_auth import ModelAuthenticationFailure
from world_understanding.validation.usd_rendering import _render_backend_label


class _StringFailure(RuntimeError):
    def __str__(self) -> str:
        raise RuntimeError("provider text unavailable")


def test_optimizer_projection_and_backend_classifier_edges() -> None:
    assert optimize_module._is_local_backend_unavailable(FileNotFoundError())
    assert not optimize_module._is_local_backend_unavailable(_StringFailure())
    assert optimize_module._project_json_value(1.25) == 1.25

    assert optimize_module._safe_prim_path(object()) is None
    assert optimize_module._safe_prim_path("not a prim path[") is None
    assert optimize_module._safe_prim_path("/") is None
    assert not optimize_module._prim_exists(None, "/World")
    assert (
        optimize_module._project_path_list_mapping(
            [], source_stage=None, target_stage=None
        )
        == {}
    )
    assert (
        optimize_module._project_path_mapping([], source_stage=None, target_stage=None)
        == {}
    )
    assert optimize_module._project_operations([42, "split"]) == ["split"]


def test_optimizer_path_mapping_rejects_bad_targets_and_accepts_string_target() -> None:
    from pxr import Usd, UsdGeom

    source = Usd.Stage.CreateInMemory()
    target = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(source, "/World")
    UsdGeom.Xform.Define(source, "/Empty")
    UsdGeom.Xform.Define(source, "/BadTarget")
    UsdGeom.Xform.Define(target, "/Target")

    assert optimize_module._project_path_list_mapping(
        {
            "/World": "/Target",
            "/Missing": ["/Target"],
            "/Empty": [],
            "/BadTarget": ["/Missing"],
        },
        source_stage=source,
        target_stage=target,
    ) == {"/World": ["/Target"]}

    assert optimize_module._project_path_list_mapping(
        {"/World": "/Target"},
        source_stage=source,
        target_stage=target,
    ) == {"/World": ["/Target"]}


def test_optimizer_restores_authored_mass_units(tmp_path: Path) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    source = Usd.Stage.CreateInMemory()
    source_world = UsdGeom.Xform.Define(source, "/World")
    source.SetDefaultPrim(source_world.GetPrim())
    UsdPhysics.SetStageKilogramsPerUnit(source, 2.5)

    output = tmp_path / "optimized.usda"
    optimized = Usd.Stage.CreateNew(str(output))
    optimized_world = UsdGeom.Xform.Define(optimized, "/World")
    optimized.SetDefaultPrim(optimized_world.GetPrim())
    sidecar = tmp_path / f"{output.name}_assets"
    sidecar.mkdir()
    (sidecar / "texture.png").write_bytes(b"sidecar-must-survive")
    asset_path = f"{sidecar.name}/texture.png"
    shader = UsdShade.Shader.Define(optimized, "/World/Shader")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(asset_path))
    optimized.GetRootLayer().Save()

    metadata, restored = optimize_module._restore_optimized_stage_metadata(
        source, output
    )

    assert metadata["restored"] is True
    assert UsdPhysics.GetStageKilogramsPerUnit(restored) == pytest.approx(2.5)
    published = Usd.Stage.Open(str(output))
    assert published is not None
    assert (
        published.GetPrimAtPath("/World/Shader").GetAttribute("inputs:file").Get().path
        == asset_path
    )
    assert (sidecar / "texture.png").read_bytes() == b"sidecar-must-survive"


@pytest.mark.parametrize(
    ("symlink_kind", "message"),
    [
        ("leaf", "symlink USD output"),
        ("parent", "symlink or non-directory ancestor"),
    ],
)
def test_optimizer_metadata_restore_rejects_symlink_republication(
    tmp_path: Path,
    symlink_kind: str,
    message: str,
) -> None:
    from pxr import Usd, UsdGeom

    source = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(source, "/World")

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_output = outside / "optimized.usda"
    optimized = Usd.Stage.CreateNew(str(outside_output))
    UsdGeom.Xform.Define(optimized, "/World")
    optimized.GetRootLayer().Save()
    before = outside_output.read_bytes()

    if symlink_kind == "leaf":
        requested = tmp_path / "requested"
        requested.mkdir()
        output = requested / "optimized.usda"
        output.symlink_to(outside_output)
    else:
        requested = tmp_path / "requested"
        requested.symlink_to(outside, target_is_directory=True)
        output = requested / "optimized.usda"

    with pytest.raises(RuntimeError, match=message):
        optimize_module._restore_optimized_stage_metadata(source, output)

    assert outside_output.read_bytes() == before


@pytest.mark.asyncio
async def test_optimizer_required_output_and_generic_failure_are_value_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = optimize_module.OptimizeUSDTask()
    with pytest.raises(ValueError, match="output_usd_path is required"):
        await task.arun({"input_usd_path": "input.usda"})

    async def fail_impl(*args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError("backend payload with secret")

    monkeypatch.setattr(task, "_arun_impl", fail_impl)
    context = {
        "input_usd_path": "input.usda",
        "output_usd_path": "output.usda",
        "event_listener": CollectingEventListener(),
    }
    with pytest.raises(RuntimeError, match="USD optimization failed") as exc_info:
        await task.arun(context)
    assert "secret" not in str(exc_info.value)
    assert context["optimization_error"] == "USD optimization failed"


@pytest.mark.asyncio
async def test_optimizer_logs_unsupported_settings_without_rendering_values() -> None:
    class Unsupported:
        pass

    listener = CollectingEventListener()
    task = optimize_module.OptimizeUSDTask()
    context = {
        "input_usd_path": "input.usda",
        "output_usd_path": "output.usda",
        "optimization_config": {
            "scene_optimizer_settings": Unsupported(),
            "flatten_prototypes": "invalid",
        },
        "event_listener": listener,
    }

    with pytest.raises(RuntimeError, match="Invalid optimization configuration"):
        await task.arun(context)

    messages = [entry["message"] for entry in listener.logs]
    assert "  Settings: <unsupported>" in messages


def test_pipeline_safe_error_helpers_and_log_filter() -> None:
    with pytest.raises(OSError) as two_paths:
        executor_module._raise_os_error(OSError, errno.EXDEV, "move failed", "a", "b")
    assert two_paths.value.filename == "a"
    assert two_paths.value.filename2 == "b"

    with pytest.raises(OSError) as no_path:
        executor_module._raise_os_error(OSError, errno.EIO, "write failed", None)
    assert no_path.value.filename is None

    long_error = type("X" * 129, (RuntimeError,), {})()
    assert executor_module.safe_exception_category(long_error) == "Exception"

    rejected_constructor_sentinel = "constructor-secret-sentinel"

    class ConstructorFailure(RuntimeError):
        def __init__(self, message: str) -> None:
            del message
            raise TypeError(rejected_constructor_sentinel)

    with pytest.raises(RuntimeError) as constructor_exc:
        executor_module._raise_public_pipeline_exception(
            ConstructorFailure, "safe pipeline failure"
        )
    assert str(constructor_exc.value) == "safe pipeline failure"
    assert constructor_exc.value.__cause__ is None
    assert constructor_exc.value.__context__ is None
    constructor_traceback = "".join(
        traceback.format_exception(
            constructor_exc.type,
            constructor_exc.value,
            constructor_exc.tb,
        )
    )
    assert rejected_constructor_sentinel not in constructor_traceback

    record = logging.LogRecord(
        "filelock",
        logging.DEBUG,
        __file__,
        1,
        "locking %s",
        ("https://user:password@example.test/lock",),
        None,
    )
    assert executor_module._CredentialRedactingLogFilter().filter(record)
    assert record.args == ()
    assert "password" not in record.getMessage()


def test_confined_checkpoint_lock_times_out_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def fake_lock_file(*args: Any, **kwargs: Any) -> Iterator[int]:
        yield 10

    def always_contended(_descriptor: int) -> Iterator[None]:
        raise BlockingIOError()

    monotonic_values = iter((10.0, 11.0))
    monkeypatch.setattr(executor_module, "open_confined_lock_file", fake_lock_file)
    monkeypatch.setattr(executor_module, "exclusive_descriptor_lock", always_contended)
    monkeypatch.setattr(
        executor_module.time, "monotonic", lambda: next(monotonic_values)
    )

    with pytest.raises(Timeout):
        with executor_module._confined_checkpoint_lock(1, "state.lock", timeout=0.1):
            raise AssertionError("lock should not have been acquired")


@pytest.mark.parametrize("failure", [artifacts.ArtifactPathError, TypeError])
def test_checkpoint_invalid_publish_failures_share_stable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: type[Exception],
) -> None:
    rejected_publish_sentinel = "checkpoint-secret-sentinel"

    @contextmanager
    def fail_open(*args: Any, **kwargs: Any) -> Iterator[int]:
        raise failure(rejected_publish_sentinel)
        yield 1

    monkeypatch.setattr(executor_module, "open_confined_directory", fail_open)
    with pytest.raises(RuntimeError) as checkpoint_exc:
        executor_module.save_pipeline_checkpoint({}, tmp_path / "state.json")
    assert str(checkpoint_exc.value) == "Unable to publish a valid pipeline checkpoint"
    assert checkpoint_exc.value.__cause__ is None
    assert checkpoint_exc.value.__context__ is None
    checkpoint_traceback = "".join(
        traceback.format_exception(
            checkpoint_exc.type,
            checkpoint_exc.value,
            checkpoint_exc.tb,
        )
    )
    assert rejected_publish_sentinel not in checkpoint_traceback


def test_pipeline_cleanup_requires_owner_and_rejects_outside_target(
    tmp_path: Path,
) -> None:
    executor = executor_module.BasePipelineExecutor()
    unowned = tmp_path / "unowned" / "work"
    with pytest.raises(ValueError, match="configured ownership root"):
        executor._clean_directories({"working_dir": unowned})

    owned_root = tmp_path / "owned"
    outside = tmp_path / "outside"
    with pytest.raises(ValueError, match="outside the configured cleanup root"):
        executor._clean_directories(
            {"working_dir": outside, "working_dir_base": owned_root}
        )


def test_confined_directory_creation_tolerates_concurrent_creator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    race = tmp_path / "race"
    race.mkdir()
    original_open = artifacts.os.open
    original_mkdir = artifacts.os.mkdir
    forced_missing = False

    def racing_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal forced_missing
        if path == "race" and not forced_missing:
            forced_missing = True
            raise FileNotFoundError(path)
        return original_open(path, flags, *args, **kwargs)

    def racing_mkdir(path: Any, *args: Any, **kwargs: Any) -> None:
        if path == "race":
            raise FileExistsError(path)
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(artifacts.os, "open", racing_open)
    monkeypatch.setattr(artifacts.os, "mkdir", racing_mkdir)

    with artifacts.open_confined_directory(race, create=True) as descriptor:
        assert os.path.samefile(f"/proc/self/fd/{descriptor}", race)


def test_relative_directory_creation_tolerates_concurrent_creator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    original_open = artifacts.os.open
    original_mkdir = artifacts.os.mkdir
    forced_missing = False

    def racing_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal forced_missing
        if path == "child" and not forced_missing:
            forced_missing = True
            raise FileNotFoundError(path)
        return original_open(path, flags, *args, **kwargs)

    def racing_mkdir(path: Any, *args: Any, **kwargs: Any) -> None:
        if path == "child":
            raise FileExistsError(path)
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(artifacts.os, "open", racing_open)
    monkeypatch.setattr(artifacts.os, "mkdir", racing_mkdir)

    with artifacts.open_confined_directory(tmp_path) as root_descriptor:
        with artifacts._open_relative_directory(
            root_descriptor, ("child",), create=True
        ) as child_descriptor:
            assert os.path.samefile(f"/proc/self/fd/{child_descriptor}", child)


def test_confined_readers_reject_directories(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    with artifacts.open_confined_directory(tmp_path) as root_descriptor:
        with pytest.raises(artifacts.ArtifactPathError, match="non-regular"):
            with artifacts.open_confined_regular_file(root_descriptor, "directory"):
                raise AssertionError("directory must not be yielded as a file")

    with pytest.raises(artifacts.ArtifactPathError, match="regular file"):
        with artifacts.open_regular_file_no_follow(directory):
            raise AssertionError("directory must not be yielded as a file")


def test_artifact_lexical_and_listing_early_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert artifacts.visible_local_artifact_key(tmp_path, ".") is None
    assert (
        artifacts.list_confined_artifact_keys(tmp_path, prefix=".pipeline_temp/private")
        == []
    )

    disappearing = tmp_path / "disappearing"
    disappearing.write_bytes(b"data")
    original_open = artifacts.os.open

    def missing_during_walk(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if path == "disappearing":
            raise FileNotFoundError(path)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(artifacts.os, "open", missing_during_walk)
    assert artifacts.list_confined_artifact_keys(tmp_path) == []


def test_confined_non_overwrite_validates_existing_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "result.bin"
    link_calls: list[tuple[str, bool]] = []

    def concurrent_link(
        source: str,
        target: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        del source, src_dir_fd
        link_calls.append((target, follow_symlinks))
        descriptor = os.open(target, os.O_CREAT | os.O_WRONLY, dir_fd=dst_dir_fd)
        os.write(descriptor, b"concurrent")
        os.close(descriptor)
        raise FileExistsError(target)

    monkeypatch.setattr(artifacts.os, "link", concurrent_link)
    with artifacts.open_confined_directory(tmp_path) as root_descriptor:
        assert not artifacts.write_bytes_to_confined(
            root_descriptor, "result.bin", b"replacement", overwrite=False
        )
    assert destination.read_bytes() == b"concurrent"
    assert link_calls == [("result.bin", False)]


def test_delete_missing_nested_artifact_obeys_missing_ok(tmp_path: Path) -> None:
    with artifacts.open_confined_directory(tmp_path) as root_descriptor:
        assert not artifacts.delete_confined_file(
            root_descriptor, "missing/file.bin", missing_ok=True
        )
        with pytest.raises(FileNotFoundError):
            artifacts.delete_confined_file(
                root_descriptor, "missing/file.bin", missing_ok=False
            )


def test_snapshot_prune_tolerates_directory_disappearing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "empty").mkdir()
    original_rmdir = artifacts.os.rmdir

    def disappearing_rmdir(path: Any, *args: Any, **kwargs: Any) -> None:
        if path == "empty":
            raise FileNotFoundError(path)
        original_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(artifacts.os, "rmdir", disappearing_rmdir)
    with artifacts.open_confined_directory(tmp_path) as root_descriptor:
        artifacts.prune_confined_snapshot(root_descriptor, "", set())


def test_remove_confined_tree_rejects_target_outside_owner(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the configured cleanup root"):
        artifacts.remove_confined_tree(tmp_path / "outside", tmp_path / "owner")


def test_model_credential_requirement_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert model_credentials._selector_path(
        "steps.render.vlm", {"provider": "openai"}
    ).endswith(".provider")

    monkeypatch.setattr(
        model_credentials, "get_openai_api_key_for_base_url", lambda *args: "key"
    )
    assert model_credentials._missing_requirement("openai", {}) is None

    monkeypatch.setattr(
        model_credentials, "get_openai_api_key_for_base_url", lambda *args: None
    )
    monkeypatch.setattr(
        model_credentials, "resolve_effective_openai_base_url", lambda value: "local"
    )
    monkeypatch.setattr(
        model_credentials, "is_openai_provider_base_url", lambda value: False
    )
    monkeypatch.setattr(model_credentials, "is_local_base_url", lambda value: True)
    local_requirement = model_credentials._missing_requirement("openai", {})
    assert local_requirement is not None
    assert local_requirement[1] is True
    assert "documented local no-auth endpoint" in local_requirement[0][1]

    monkeypatch.setattr(model_credentials, "is_local_base_url", lambda value: False)
    third_party_requirement = model_credentials._missing_requirement("openai", {})
    assert third_party_requirement == (
        ("endpoint-scoped api_key or api_key_env paired with base_url",),
        True,
    )

    monkeypatch.setattr(
        model_credentials, "is_openai_provider_base_url", lambda value: True
    )
    official_requirement = model_credentials._missing_requirement("openai", {})
    assert official_requirement == (
        model_credentials.API_KEY_ENV_VAR_MAP["openai"],
        False,
    )

    monkeypatch.setattr(
        model_credentials, "get_nim_api_key_for_base_url", lambda *args: None
    )
    monkeypatch.setattr(
        model_credentials, "is_nvidia_provider_base_url", lambda value: False
    )
    monkeypatch.setattr(model_credentials, "is_local_base_url", lambda value: True)
    nim_requirement = model_credentials._missing_requirement(
        "nim", {"base_url": "http://localhost:8000"}
    )
    assert nim_requirement is not None
    assert "documented local no-auth endpoint" in nim_requirement[0][1]

    monkeypatch.setattr(
        model_credentials,
        "_resolved_endpoint_key",
        lambda config: ("endpoint-key", False),
    )
    assert model_credentials._missing_requirement("anthropic", {}) is None
    monkeypatch.setattr(
        model_credentials, "_resolved_endpoint_key", lambda config: (None, False)
    )
    monkeypatch.setattr(
        model_credentials, "get_env_api_key_for_backend", lambda backend: None
    )
    assert model_credentials._missing_requirement("anthropic", {}) == (
        model_credentials.API_KEY_ENV_VAR_MAP["anthropic"],
        False,
    )


def test_model_credential_validation_transforms_and_reports_env_redirect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_credentials.validate_selected_model_credentials(
        {"project": {}, "steps": []},
        tmp_path / "config.yaml",
        (),
        (),
        get_step_defaults=lambda _: {},
    )

    transformed: list[tuple[str, str]] = []

    def transform(
        step_name: str, model_path: str, model_config: dict[str, Any]
    ) -> dict[str, Any]:
        transformed.append((step_name, model_path))
        return {
            **{key: value for key, value in model_config.items() if key != "backend"},
            "provider": "openai",
        }

    monkeypatch.setattr(
        model_credentials, "get_openai_api_key_for_base_url", lambda *args: None
    )
    monkeypatch.setattr(
        model_credentials, "resolve_effective_openai_base_url", lambda value: "local"
    )
    monkeypatch.setattr(
        model_credentials, "is_openai_provider_base_url", lambda value: False
    )
    monkeypatch.setattr(model_credentials, "is_local_base_url", lambda value: True)

    with pytest.raises(ValueError) as exc_info:
        model_credentials.validate_selected_model_credentials(
            {"render": {"enabled": True}},
            tmp_path / "config.yaml",
            (),
            (),
            model_config_iterator=lambda *args: [("render.vlm", {"backend": "nim"})],
            transform_model_config=transform,
        )

    assert transformed == [("render", "render.vlm")]
    assert model_credentials.OPENAI_ENV_REDIRECT_CREDENTIAL_MESSAGE in str(
        exc_info.value
    )
    assert "render.vlm.provider='openai'" in str(exc_info.value)


def test_unknown_key_warnings_cover_non_string_and_nested_sections(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("forward-port-unknown-keys")
    schema = {"section": {"nested": {"known": True}}, "steps": {"render": {}}}
    config = {
        3: {},
        "section": {"nested": {object(): "ignored"}},
        "steps": {object(): {}, "render": {}},
    }
    with caplog.at_level(logging.WARNING, logger=logger.name):
        unknown_keys.warn_unknown_nested_config_keys(
            config,
            schema,
            logger,
            strict_paths=[("section", "nested")],
        )

    assert "<non-string-key>" in caplog.text
    caplog.clear()
    unknown_keys._warn_step_keys([], {}, logger, strict_paths=frozenset())
    assert caplog.records == []


def test_unknown_key_suggestion_requires_clear_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        unknown_keys.difflib,
        "get_close_matches",
        lambda *args, **kwargs: ["enabled", "enable"],
    )

    class FakeMatcher:
        def __init__(self, _junk: Any, _key: str, candidate: str) -> None:
            self.candidate = candidate

        def ratio(self) -> float:
            return 0.95 if self.candidate == "enabled" else 0.70

    monkeypatch.setattr(unknown_keys.difflib, "SequenceMatcher", FakeMatcher)
    assert (
        unknown_keys._unambiguous_suggestion("enabld", ("enabled", "enable"))
        == "enabled"
    )


def test_config_isolation_float_set_and_recursive_tuple() -> None:
    assert isolation.normalize_yaml_config_value(1.5) == 1.5
    source_set = {"b", "a"}
    cloned_set = isolation.clone_config_containers(source_set)
    assert cloned_set == {"a", "b"}
    assert cloned_set is not source_set

    recursive_list: list[Any] = []
    recursive_tuple = (recursive_list,)
    recursive_list.append(recursive_tuple)
    cloned_tuple = isolation.clone_config_containers(recursive_tuple)
    assert cloned_tuple[0][0] is cloned_tuple


def test_result_projection_rejects_non_mapping_cycles_and_defensive_non_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert result_projection.project_result_metadata([]) == {}
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    assert result_projection.project_result_metadata(cyclic) == {}

    original_isinstance = builtins.isinstance

    def defensive_isinstance(value: Any, class_or_tuple: Any) -> bool:
        if class_or_tuple is dict and type(value) is dict:
            return False
        return original_isinstance(value, class_or_tuple)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "isinstance", defensive_isinstance)
        assert result_projection.project_result_metadata({"ok": True}) == {}


def test_usd_data_config_defers_path_inspection_failure_to_shared_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_exists = Path.exists

    def fail_exists(path: Path) -> bool:
        if path.name == "config.yaml":
            raise OSError("inspection failed")
        return real_exists(path)

    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(Path, "exists", fail_exists)
    monkeypatch.setattr(
        usd_config_module,
        "load_config_mapping_from_context",
        lambda *args, **kwargs: (
            {"usd_path": "scene.usda", "output_dir": "output"},
            config_path,
        ),
    )
    monkeypatch.setattr(
        usd_config_module, "log_config_source", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        usd_config_module,
        "create_directory_with_safe_diagnostics",
        lambda *args, **kwargs: None,
    )

    result = usd_config_module.USDDataPrepConfigTask().run(
        {"config_path": config_path, "event_listener": CollectingEventListener()}
    )
    assert result["usd_path"] == config_path.parent / "scene.usda"


_REJECTED_OPTIMIZATION_CONFIG_SENTINEL = "optimization-config-secret-sentinel"


class _SecretOptimizationConfigKey:
    def __repr__(self) -> str:
        return _REJECTED_OPTIMIZATION_CONFIG_SENTINEL


@pytest.mark.parametrize(
    ("optimization_config", "expected_message"),
    [
        pytest.param(
            f"invalid-{_REJECTED_OPTIMIZATION_CONFIG_SENTINEL}",
            "optimization_config must be a mapping",
            id="non-mapping-optimization-config",
        ),
        pytest.param(
            {
                "scene_optimizer_settings": {
                    _SecretOptimizationConfigKey(): True,
                }
            },
            "Invalid scene_optimizer_settings in config",
            id="non-string-secret-bearing-settings-key",
        ),
    ],
)
def test_optimize_config_rejects_non_mapping_shapes(
    monkeypatch: pytest.MonkeyPatch,
    optimization_config: Any,
    expected_message: str,
) -> None:
    task = optimize_config_module.OptimizeUSDConfigTask()
    monkeypatch.setattr(
        task,
        "_load_config",
        lambda context, listener: {
            "input_usd_path": "input.usda",
            "output_usd_path": "output.usda",
            "optimization_config": optimization_config,
        },
    )
    with pytest.raises(ValueError) as config_exc:
        task.run({"event_listener": CollectingEventListener()})
    assert str(config_exc.value) == expected_message
    assert config_exc.value.__cause__ is None
    assert config_exc.value.__context__ is None
    config_traceback = "".join(
        traceback.format_exception(
            config_exc.type,
            config_exc.value,
            config_exc.tb,
        )
    )
    assert _REJECTED_OPTIMIZATION_CONFIG_SENTINEL not in config_traceback


def test_failure_projection_helpers_preserve_errno_without_filename() -> None:
    restore_failure = restore_config_module._RestoreConfigFailure(
        OSError,
        (errno.EIO, "read failed"),
        (errno.EIO, "read failed", None),
    )
    with pytest.raises(OSError) as restore_exc:
        restore_config_module._raise_restore_failure(restore_failure)
    assert restore_exc.value.errno == errno.EIO
    assert restore_exc.value.filename is None

    loader_failure = context_loader._ProjectedFailure(
        OSError,
        (errno.EIO, "read failed"),
        (errno.EIO, "read failed", None),
    )
    with pytest.raises(OSError) as loader_exc:
        context_loader._raise_failure(loader_failure)
    assert loader_exc.value.errno == errno.EIO
    assert loader_exc.value.filename is None


def test_cli_step_filter_accepts_sequence_input() -> None:
    assert _normalize_step_filter(
        ("render", "export"),
        option_name="--only",
        valid_steps=("render", "export"),
    ) == ["render", "export"]


def test_mock_material_json_scanner_recovers_after_invalid_object() -> None:
    from world_understanding.functions.models.backends.public.mock import (
        _extract_json_material_names,
    )

    assert _extract_json_material_names(
        'prefix {invalid then {"material_names": ["Steel", "Wood"]}'
    ) == ["Steel", "Wood"]


def test_session_failure_and_metadata_projection_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = session_module._SessionListFailure(
        OSError,
        (errno.EIO, "listing failed"),
        (errno.EIO, "listing failed", None),
    )
    with pytest.raises(OSError) as exc_info:
        session_module._raise_session_list_failure(failure)
    assert exc_info.value.errno == errno.EIO
    assert exc_info.value.filename is None

    with pytest.raises(OSError) as path_exc:
        session_module._raise_safe_path_inspection_error(
            OSError(errno.EACCES, "secret provider path"),
            path=Path("https://user:password@example.test/session"),
            label="session entry",
        )
    assert path_exc.value.errno == errno.EACCES
    assert "password" not in str(path_exc.value)

    monkeypatch.setattr(
        session_module, "redact_sensitive_config", lambda metadata: "<redacted>"
    )
    assert session_module._project_metadata_for_durable_storage({"token": "x"}) == {}


def test_session_listing_skips_symlink_and_invalid_identifier(tmp_path: Path) -> None:
    valid_target = tmp_path / "target"
    valid_target.mkdir()
    (tmp_path / ".linked").symlink_to(valid_target, target_is_directory=True)
    (tmp_path / "sess").mkdir()

    assert (
        session_module.SessionManager._list_sessions_impl(tmp_path, prefix="sess") == []
    )
    assert all(
        entry["session_id"] != "linked"
        for entry in session_module.SessionManager._list_sessions_impl(
            tmp_path, prefix="."
        )
    )


def test_session_listing_tolerates_metadata_inspection_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_dir = tmp_path / ".valid"
    session_dir.mkdir()
    original = session_module._path_is_symlink_with_safe_diagnostics

    def inspect(path: Path, *, label: str) -> bool:
        if path.name == ".metadata.json":
            raise OSError("metadata inspection failed")
        return original(path, label=label)

    monkeypatch.setattr(
        session_module, "_path_is_symlink_with_safe_diagnostics", inspect
    )
    with caplog.at_level(logging.WARNING, logger=session_module.logger.name):
        sessions = session_module.SessionManager._list_sessions_impl(tmp_path)

    assert sessions[0]["session_id"] == "valid"
    assert sessions[0]["metadata"] == {}
    assert "Unable to inspect session metadata" in caplog.text


def test_material_fingerprint_fail_closed_and_empty_network_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import UsdShade

    class MaterialHandle:
        def GetPrim(self) -> object:
            return object()

    material = MaterialHandle()

    with monkeypatch.context() as patch:
        patch.setattr(UsdShade, "Material", lambda prim: False)
        assert scene_analysis._material_appearance_fingerprint(material) is None

    fake_material = SimpleNamespace(GetSurfaceOutput=lambda: None)
    with monkeypatch.context() as patch:
        patch.setattr(UsdShade, "Material", lambda prim: fake_material)
        assert scene_analysis._material_appearance_fingerprint(material) is None

    output = SimpleNamespace(
        GetConnectedSource=lambda: (SimpleNamespace(GetPrim=lambda: object()),)
    )
    fake_material = SimpleNamespace(GetSurfaceOutput=lambda: output)
    with monkeypatch.context() as patch:
        patch.setattr(UsdShade, "Material", lambda prim: fake_material)
        patch.setattr(UsdShade, "Shader", lambda prim: False)
        assert scene_analysis._material_appearance_fingerprint(material) is None

    fake_shader = SimpleNamespace()
    with monkeypatch.context() as patch:
        patch.setattr(UsdShade, "Material", lambda prim: fake_material)
        patch.setattr(UsdShade, "Shader", lambda prim: fake_shader)
        patch.setattr(
            scene_analysis,
            "_shader_network_fingerprint_values",
            lambda *args, **kwargs: None,
        )
        assert scene_analysis._material_appearance_fingerprint(material) is None
        patch.setattr(
            scene_analysis,
            "_shader_network_fingerprint_values",
            lambda *args, **kwargs: [],
        )
        assert scene_analysis._material_appearance_fingerprint(material) == ""


def test_bound_material_identity_uses_absolute_path_when_network_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prim = SimpleNamespace(
        IsValid=lambda: True,
        GetPath=lambda: "/World/Asset/Looks/Private",
    )
    material = SimpleNamespace(GetPrim=lambda: prim)
    monkeypatch.setattr(
        scene_analysis, "_material_appearance_fingerprint", lambda value: None
    )
    assert (
        scene_analysis._bound_material_identity(material, "/World/Asset")
        == "/World/Asset/Looks/Private"
    )


def test_shader_fingerprint_cycle_and_invalid_connected_shader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import UsdShade

    cyclic_shader = SimpleNamespace(GetPath=lambda: "/Shader")
    assert (
        scene_analysis._shader_network_fingerprint_values(
            cyclic_shader, visited={"/Shader"}
        )
        == []
    )

    bad_prim = object()
    nested_prim = object()

    class FakeInput:
        def __init__(self, prim: object) -> None:
            self.prim = prim

        def GetConnectedSource(self) -> tuple[Any, str, str]:
            return (SimpleNamespace(GetPrim=lambda: self.prim), "rgb", "output")

    nested_shader = SimpleNamespace(
        GetPath=lambda: "/Nested",
        GetShaderId=lambda: "Nested",
        GetPrim=lambda: SimpleNamespace(GetTypeName=lambda: "Shader"),
        GetInputs=lambda: [FakeInput(bad_prim)],
    )
    outer_shader = SimpleNamespace(
        GetPath=lambda: "/Outer",
        GetShaderId=lambda: "Outer",
        GetPrim=lambda: SimpleNamespace(GetTypeName=lambda: "Shader"),
        GetInputs=lambda: [FakeInput(nested_prim)],
    )

    def shader_adapter(prim: object) -> Any:
        return nested_shader if prim is nested_prim else False

    monkeypatch.setattr(UsdShade, "Shader", shader_adapter)
    assert (
        scene_analysis._shader_network_fingerprint_values(outer_shader, visited=set())
        is None
    )


def test_detect_objects_distinguishes_topologies_from_same_source() -> None:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    def define_mesh(path: str) -> None:
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.GetPointsAttr().Set(
            [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)]
        )
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])

    object_paths = [f"/World/Object{name}" for name in "ABCD"]
    for path in object_paths:
        UsdGeom.Xform.Define(stage, path)
        define_mesh(f"{path}/Mesh")
    define_mesh("/World/ObjectC/Mesh2")
    define_mesh("/World/ObjectD/Mesh2")

    objects, groups = scene_analysis.detect_objects(
        stage,
        {
            "sub_usd_files": [
                {
                    "asset_path": "shared.usda",
                    "reference_count": 4,
                    "referencing_prims": object_paths,
                }
            ]
        },
        {},
    )

    assert len(objects) == 4
    source_groups = [group for group in groups if group["source_file"] == "shared.usda"]
    assert {group["instance_count"] for group in source_groups} == {2}
    assert {group["group_name"] for group in source_groups} == {
        "shared_1m_3v_1f",
        "shared_2m_6v_2f",
    }


def test_parallel_classification_rechecks_auth_abort_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AbortAfterPreparation:
        def __init__(self) -> None:
            self.checks = 0

        def is_set(self) -> bool:
            self.checks += 1
            return self.checks >= 2

        def set(self) -> None:
            pass

    monkeypatch.setattr(inference_module, "Event", AbortAfterPreparation)
    monkeypatch.setattr(
        inference_module,
        "classify_object",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("provider must not run after authentication abort")
        ),
    )

    with pytest.raises(ModelAuthenticationFailure):
        inference_module._process_parallel(
            object(),
            [{"id": "entry", "text": "classify"}],
            object(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            1,
            "class",
        )


@pytest.mark.asyncio
async def test_async_classification_rechecks_auth_abort_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AbortAfterPreparation:
        def __init__(self) -> None:
            self.checks = 0

        def is_set(self) -> bool:
            self.checks += 1
            return self.checks >= 2

        def set(self) -> None:
            pass

    async def provider_call(**kwargs: Any) -> NoReturn:
        raise AssertionError("provider must not run after authentication abort")

    monkeypatch.setattr(inference_module.asyncio, "Event", AbortAfterPreparation)
    monkeypatch.setattr(inference_module, "async_classify_object", provider_call)
    with pytest.raises(ModelAuthenticationFailure):
        await inference_module.async_batch_classify_objects(
            object(),
            [{"id": "entry", "text": "classify"}],
            object(),
            max_workers=1,
            max_retries=1,
        )


def test_newton_context_uses_model_bvh_rebuild_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd

    fake_newton = ModuleType("newton")
    fake_geometry = ModuleType("newton.geometry")
    fake_geometry.ShapeFlags = SimpleNamespace(VISIBLE=1)  # type: ignore[attr-defined]
    fake_newton.geometry = fake_geometry  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "newton", fake_newton)
    monkeypatch.setitem(sys.modules, "newton.geometry", fake_geometry)

    class FakeWarp:
        int32 = "int32"
        transform = "transform"
        vec3f = "vec3f"

        @staticmethod
        def array(data: Any, **kwargs: Any) -> Any:
            return np.asarray(data)

    rebuilds: list[Any] = []
    model = SimpleNamespace(bvh_build_shapes=lambda state: rebuilds.append(state))
    state = object()
    context = SimpleNamespace(
        _wu_render_model=model,
        _wu_render_state=state,
        _wu_base_shape_flags=[],
    )
    monkeypatch.setattr(
        render_warp, "_import_warp", lambda: (FakeWarp(), None, None, None)
    )

    assert (
        render_warp._update_newton_model_render_context(
            context,
            render_meshes=[],
            mesh_prims=[],
            time_code=Usd.TimeCode.Default(),
            device="cpu",
            color_boost=1.0,
        )
        == 0
    )
    assert rebuilds == [state]


def test_render_all_cameras_uses_modern_pinhole_ray_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd

    calls: list[tuple[int, int, Any]] = []

    class FakeArray:
        def __init__(self, value: Any) -> None:
            self.value = value

        def reshape(self, shape: Any) -> FakeArray:
            return self

    class FakeWarp:
        float32 = "float32"
        transformf = "transformf"

        @staticmethod
        def init() -> None:
            pass

        @staticmethod
        def array(data: Any, **kwargs: Any) -> FakeArray:
            return FakeArray(data)

    utils = SimpleNamespace(
        compute_camera_rays_pinhole=lambda width, height, *, camera_fovs: (
            calls.append((width, height, camera_fovs)),
            "modern-rays",
        )[1],
        create_color_image_output=lambda *args: object(),
    )
    monkeypatch.setattr(
        render_warp, "_import_warp", lambda: (FakeWarp(), None, None, None)
    )
    monkeypatch.setattr(
        render_warp, "_extract_meshes", lambda *args: ([object()], [object()])
    )
    monkeypatch.setattr(
        render_warp,
        "_setup_render_context",
        lambda **kwargs: SimpleNamespace(utils=utils),
    )
    monkeypatch.setattr(render_warp, "_setup_lights", lambda *args: None)
    monkeypatch.setattr(render_warp, "_compute_camera_fov", lambda *args: 0.5)

    result = render_warp.render_all_cameras(
        Usd.Stage.CreateInMemory(),
        image_width=2,
        image_height=3,
        cameras=["/Camera"],
        frames=",",
        device="cpu",
    )

    assert len(calls) == 1
    assert calls[0][:2] == (2, 3)
    assert result["failed_cameras"] == 1


def test_llm_json_iteration_empty_and_duplicate_candidates() -> None:
    assert list(iter_json_dicts_in_text_order("")) == []
    assert list(iter_json_dicts_in_text_order('{"ok": true} {"ok": true}')) == [
        {"ok": True}
    ]


def test_render_backend_label_preserves_none() -> None:
    assert _render_backend_label(None) is None
