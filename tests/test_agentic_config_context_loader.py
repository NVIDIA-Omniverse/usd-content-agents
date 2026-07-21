# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
import gc
import logging
import threading
import traceback
import weakref
from collections import UserDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn
from unittest.mock import patch

import pytest

from world_understanding.agentic.config import (
    ConfigParseError,
    ConfigStructureError,
    load_config_mapping_from_context,
    log_config_source,
    normalize_yaml_config_value,
)


class _NonCopyableRuntimeClient:
    def __deepcopy__(self, _memo: dict[int, Any]) -> _NonCopyableRuntimeClient:
        raise AssertionError("runtime clients must not be deep-copied")


class _FailingConfigMapping(UserDict[str, Any]):
    def items(self) -> NoReturn:
        raise RuntimeError(f"isolation failure: {self.data['api_key']}")


class _RenderingPath(PurePosixPath):
    render_count = 0

    def __str__(self) -> str:
        type(self).render_count += 1
        return "path-render-secret-713"


def _assert_production_traceback_locals_exclude(
    error: BaseException, sentinel: str
) -> None:
    traceback_frame = error.__traceback__
    production_frames = 0
    while traceback_frame is not None:
        frame = traceback_frame.tb_frame
        if Path(frame.f_code.co_filename).resolve() != Path(__file__).resolve():
            production_frames += 1
            assert sentinel not in repr(frame.f_locals)
        traceback_frame = traceback_frame.tb_next
    assert production_frames > 0


def test_yaml_normalizer_rejects_path_subclasses_without_rendering() -> None:
    path = _RenderingPath("safe-name")
    _RenderingPath.render_count = 0

    with pytest.raises(
        TypeError,
        match="^Unsupported YAML-equivalent configuration value$",
    ) as exc_info:
        normalize_yaml_config_value({"path": path})

    assert _RenderingPath.render_count == 0
    assert "path-render-secret-713" not in str(exc_info.value)


def test_config_dict_is_isolated_retains_credentials_and_uses_anchor(
    tmp_path: Path,
) -> None:
    anchor = tmp_path / "source" / "pipeline.yaml"
    source = UserDict(
        {
            "provider": {
                "api_key": "runtime-only-credential",
                "nested": [{"token": "runtime-only-token"}],
            }
        }
    )

    loaded, config_path = load_config_mapping_from_context(
        {"config_dict": source, "config_path": anchor}
    )

    assert config_path == anchor
    assert loaded == source
    assert loaded["provider"] is not source["provider"]
    assert loaded["provider"]["nested"] is not source["provider"]["nested"]
    loaded["provider"]["nested"][0]["token"] = "changed"
    assert source["provider"]["nested"][0]["token"] == "runtime-only-token"
    assert not anchor.exists()


def test_config_dict_isolation_preserves_opaque_runtime_leaves() -> None:
    runtime_lock = threading.Lock()
    runtime_client = _NonCopyableRuntimeClient()
    source = {
        "runtime": {"lock": runtime_lock, "client": runtime_client},
        "nested": [{"owner": "caller"}],
    }

    loaded, _ = load_config_mapping_from_context({"config_dict": source})
    loaded["nested"][0]["owner"] = "loaded"

    assert loaded["runtime"]["lock"] is runtime_lock
    assert loaded["runtime"]["client"] is runtime_client
    assert source["nested"] == [{"owner": "caller"}]


def test_config_dict_isolation_preserves_recursive_container_topology() -> None:
    recursive_mapping: dict[str, Any] = {}
    recursive_list: list[Any] = []
    recursive_mapping["list"] = recursive_list
    recursive_list.append(recursive_mapping)
    source = {"left": recursive_mapping, "right": recursive_mapping}

    loaded, _ = load_config_mapping_from_context({"config_dict": source})

    assert loaded["left"] is loaded["right"]
    assert loaded["left"] is not recursive_mapping
    assert loaded["left"]["list"] is not recursive_list
    assert loaded["left"]["list"][0] is loaded["left"]


def test_config_dict_isolation_failure_is_value_free_and_detached() -> None:
    sentinel = "config-isolation-secret-713"
    source = _FailingConfigMapping({"api_key": sentinel})

    with pytest.raises(ValueError) as exc_info:
        load_config_mapping_from_context({"config_dict": source})

    assert str(exc_info.value) == "Unable to isolate configuration mapping"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert exc_info.value.__suppress_context__ is True
    assert sentinel not in "".join(traceback.format_exception(exc_info.value))
    _assert_production_traceback_locals_exclude(exc_info.value, sentinel)


def test_default_config_path_is_an_in_memory_anchor(tmp_path: Path) -> None:
    anchor = tmp_path / "defaults" / "agent.yaml"

    loaded, config_path = load_config_mapping_from_context(
        {"config_dict": {"input": "scene.usd"}},
        default_config_path=anchor,
    )

    assert loaded == {"input": "scene.usd"}
    assert config_path == anchor


def test_config_source_logging_uses_one_value_free_taxonomy() -> None:
    messages: list[str] = []
    log_config_source(
        {"config_dict": {"api_key": "never-log"}},
        messages.append,
        label="predict",
    )
    log_config_source(
        {"config_dict": None, "config_path": "config.yaml"},
        messages.append,
        label="predict",
    )

    assert messages == [
        "Loading predict configuration from memory",
        "Loading predict configuration from file",
    ]
    assert "never-log" not in " ".join(messages)


def test_absent_and_none_config_dict_share_file_fallback(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("value: from-file\n", encoding="utf-8")

    for context in (
        {"config_path": config_path},
        {"config_dict": None, "config_path": config_path},
    ):
        loaded, anchor = load_config_mapping_from_context(context)
        assert loaded == {"value": "from-file"}
        assert anchor == config_path


def test_non_mapping_config_dict_has_one_typed_contract() -> None:
    for value in ([], "config", 3, False):
        with pytest.raises(ConfigStructureError) as exc_info:
            load_config_mapping_from_context(
                {"config_dict": value},
            )
        assert str(exc_info.value) == (
            f"config_dict must be a mapping, got {type(value).__name__}"
        )


def test_config_clone_preserves_opaque_leaves_aliases_cycles_and_concurrency() -> None:
    opaque_client = threading.Lock()
    shared: list[object] = []
    source: dict[str, object] = {
        "client": opaque_client,
        "first": shared,
        "second": shared,
    }
    source["self"] = source

    def load_once() -> dict[str, object]:
        loaded, _ = load_config_mapping_from_context({"config_dict": source})
        return loaded

    with ThreadPoolExecutor(max_workers=4) as executor:
        loaded_configs = list(executor.map(lambda _: load_once(), range(8)))

    for loaded in loaded_configs:
        assert loaded is not source
        assert loaded["client"] is opaque_client
        assert loaded["first"] is loaded["second"]
        assert loaded["first"] is not shared
        assert loaded["self"] is loaded
    assert len({id(loaded["first"]) for loaded in loaded_configs}) == len(
        loaded_configs
    )


@pytest.mark.parametrize(
    "context",
    [
        {"config_dict": {}},
        {"config_dict": UserDict()},
    ],
)
def test_empty_in_memory_mapping_requires_opt_in(
    context: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="^Configuration is empty: config.yaml$"):
        load_config_mapping_from_context(context)

    loaded, _ = load_config_mapping_from_context(context, allow_empty=True)
    assert loaded == {}


def test_empty_yaml_requires_opt_in(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="^Configuration is empty:"):
        load_config_mapping_from_context({"config_path": config_path})

    loaded, anchor = load_config_mapping_from_context(
        {"config_path": config_path}, allow_empty=True
    )
    assert loaded == {}
    assert anchor == config_path


def test_mapping_validation_is_value_free_for_memory_and_yaml(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError) as memory_exc:
        load_config_mapping_from_context({"config_dict": ["secret-value"]})
    assert str(memory_exc.value) == "config_dict must be a mapping, got list"
    assert "secret-value" not in str(memory_exc.value)

    config_path = tmp_path / "list.yaml"
    config_path.write_text("- secret-value\n", encoding="utf-8")
    with pytest.raises(ValueError) as file_exc:
        load_config_mapping_from_context({"config_path": config_path})
    assert str(file_exc.value) == (
        "Configuration file must contain a mapping, got list"
    )
    assert "secret-value" not in str(file_exc.value)


def test_missing_path_and_file_options(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="^config_dict or config_path is required in context$",
    ):
        load_config_mapping_from_context({})

    sentinel = "never-disclose-missing-path"
    config_path = tmp_path / f"Authorization: Bearer {sentinel}"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_config_mapping_from_context({"config_path": config_path})
    assert str(exc_info.value) == "Configuration file not found: <redacted>"
    assert sentinel not in str(exc_info.value)

    loaded, anchor = load_config_mapping_from_context(
        {"config_path": config_path},
        allow_missing_file=True,
    )
    assert loaded == {}
    assert anchor == config_path


def test_path_normalized_signed_query_is_redacted_from_loader_errors() -> None:
    sentinel = "loader-normalized-path-secret-713"
    raw_path = f"https://assets.example.test/config.yaml?X-Amz-Signature={sentinel}"

    with pytest.raises(FileNotFoundError) as exc_info:
        load_config_mapping_from_context({"config_path": raw_path})

    assert str(exc_info.value) == "Configuration file not found: <redacted>"
    assert sentinel not in str(exc_info.value)


def test_message_templates_replace_only_documented_tokens(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.yaml"

    with pytest.raises(ValueError) as missing_exc:
        load_config_mapping_from_context(
            {},
            default_config_path=anchor,
            missing_path_message="missing {config_path}; keep {literal}",
        )
    assert str(missing_exc.value) == f"missing {anchor}; keep {{literal}}"

    with pytest.raises(ValueError) as exc_info:
        load_config_mapping_from_context(
            {"config_dict": []},
            default_config_path=anchor,
            config_dict_non_mapping_message=(
                "invalid {type_name} at {config_path}; keep {literal}"
            ),
        )

    assert str(exc_info.value) == f"invalid list at {anchor}; keep {{literal}}"


def test_malformed_yaml_has_no_source_or_path_secret_in_observables(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "ZQX713MalformedAliasCredentialMNP9"
    config_path = tmp_path / f"Authorization: Bearer {sentinel}"
    config_path.write_text(
        f"api_key: {sentinel}\nalias: *{sentinel}\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError) as exc_info:
        load_config_mapping_from_context({"config_path": config_path})

    assert str(exc_info.value) == "Unable to parse configuration file: <redacted>"
    assert isinstance(exc_info.value, ConfigParseError)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert exc_info.value.__suppress_context__ is True
    assert sentinel not in repr((exc_info.value.__cause__, exc_info.value.__context__))
    observable = "\n".join(
        (
            str(exc_info.value),
            "".join(traceback.format_exception(exc_info.value)),
            caplog.text,
        )
    )
    assert sentinel not in observable
    _assert_production_traceback_locals_exclude(exc_info.value, sentinel)


def test_os_error_has_no_source_or_path_secret_in_observables(
    tmp_path: Path,
) -> None:
    sentinel = "ZQX713ReadCredentialMNP9"
    config_path = tmp_path / f"Authorization: Bearer {sentinel}"
    config_path.write_text("value: valid\n", encoding="utf-8")

    with (
        patch.object(Path, "open", side_effect=OSError(f"denied: {sentinel}")),
        pytest.raises(OSError) as exc_info,
    ):
        load_config_mapping_from_context({"config_path": config_path})

    assert str(exc_info.value) == "Unable to read configuration file: <redacted>"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in repr((exc_info.value.__cause__, exc_info.value.__context__))
    assert sentinel not in "".join(traceback.format_exception(exc_info.value))
    _assert_production_traceback_locals_exclude(exc_info.value, sentinel)


@pytest.mark.parametrize(
    ("method_name", "error_type"),
    [("exists", PermissionError), ("open", FileNotFoundError)],
)
def test_os_error_projection_preserves_subclass_errno_and_safe_filename(
    method_name: str,
    error_type: type[OSError],
    tmp_path: Path,
) -> None:
    sentinel = "loader-oserror-path-secret-713"
    config_path = tmp_path / f"user:{sentinel}@config.example.test" / "config.yaml"
    if method_name == "open":
        config_path.parent.mkdir()
        config_path.write_text("value: valid\n", encoding="utf-8")
    error = error_type(errno.EACCES, "denied", str(config_path))

    with (
        patch.object(Path, method_name, side_effect=error),
        pytest.raises(error_type) as exc_info,
    ):
        load_config_mapping_from_context({"config_path": config_path})

    assert exc_info.value.errno == errno.EACCES
    assert exc_info.value.filename == "<redacted>"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in repr((exc_info.value.__cause__, exc_info.value.__context__))
    assert sentinel not in "".join(traceback.format_exception(exc_info.value))


def test_invalid_text_encoding_does_not_retain_source_bytes(
    tmp_path: Path,
) -> None:
    sentinel = b"loader-decode-source-secret-713"
    config_path = tmp_path / "invalid-utf8.yaml"
    config_path.write_bytes(b"api_key: " + sentinel + b"\xff\n")

    with pytest.raises(UnicodeError) as exc_info:
        load_config_mapping_from_context({"config_path": config_path})

    assert exc_info.value.args == (f"Unable to read configuration file: {config_path}",)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel.decode() not in repr(
        (exc_info.value.__cause__, exc_info.value.__context__)
    )
    assert sentinel.decode() not in "".join(traceback.format_exception(exc_info.value))
    _assert_production_traceback_locals_exclude(exc_info.value, sentinel.decode())


def test_projected_parse_error_does_not_retain_parser_exception_graph(
    tmp_path: Path,
) -> None:
    class ParserPayload:
        pass

    payload = ParserPayload()
    payload_ref = weakref.ref(payload)
    holder: list[ParserPayload] = [payload]
    config_path = tmp_path / "config.yaml"
    config_path.write_text("value: ignored\n", encoding="utf-8")

    def failing_loader(_stream: object) -> object:
        error = RuntimeError("parser failure")
        error.payload = holder[0]  # type: ignore[attr-defined]
        raise error

    with pytest.raises(ConfigParseError) as exc_info:
        load_config_mapping_from_context(
            {"config_path": config_path}, file_loader=failing_loader
        )

    del payload
    holder.clear()
    gc.collect()
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert payload_ref() is None
