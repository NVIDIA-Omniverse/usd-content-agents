# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Value-safe loading of workflow configuration mappings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, TextIO

import yaml

from world_understanding.utils.credentials import redact_sensitive_path

from .isolation import clone_config_containers


class ConfigLoadError(ValueError):
    """Base class for value-free configuration loader failures."""


class ConfigSourceError(ConfigLoadError):
    """No usable in-memory or file-backed configuration source was supplied."""


class ConfigParseError(ConfigLoadError):
    """A configuration source could not be parsed safely."""


class ConfigStructureError(ConfigLoadError):
    """A parsed or in-memory configuration has an invalid root structure."""


class ConfigEmptyError(ConfigStructureError):
    """A configuration mapping is empty where content is required."""


ConfigFileLoader = Callable[[TextIO], Any]


@dataclass(frozen=True)
class _ProjectedFailure:
    """Value-safe data needed to recreate an expected public exception."""

    error_type: type[Exception]
    args: tuple[Any, ...]
    os_error: tuple[int | None, str | None, str | None] | None = None
    suppress_context: bool = False


def _capture_failure(error: Exception) -> _ProjectedFailure:
    os_error = (
        (error.errno, error.strerror, error.filename)
        if isinstance(error, OSError)
        else None
    )
    return _ProjectedFailure(
        type(error),
        error.args,
        os_error,
        error.__suppress_context__,
    )


def _raise_failure(failure: _ProjectedFailure) -> NoReturn:
    replacement: Exception
    if failure.os_error is not None:
        error_number, message, filename = failure.os_error
        if filename is not None:
            replacement = failure.error_type(error_number, message, filename)
        elif error_number is not None:
            replacement = failure.error_type(error_number, message)
        else:
            replacement = failure.error_type(*failure.args)
    else:
        replacement = failure.error_type(*failure.args)

    if failure.suppress_context:
        raise replacement from None
    raise replacement


def _isolate_config_mapping(config_value: Mapping[Any, Any]) -> dict[str, Any]:
    """Return an isolated config mapping or one fixed value-free failure."""
    isolated: Any = None
    isolation_failed = False
    try:
        source_mapping = (
            config_value if type(config_value) is dict else dict(config_value.items())
        )
        isolated = clone_config_containers(source_mapping)
    except Exception:
        isolation_failed = True

    if isolation_failed or not isinstance(isolated, dict):
        raise ValueError("Unable to isolate configuration mapping") from None
    return isolated


def _render_message(
    template: str,
    *,
    config_path: Path,
    type_name: str | None = None,
) -> str:
    """Render only the loader's documented, value-free message tokens."""
    safe_path = redact_sensitive_path(config_path)
    message = template.replace("{config_path}", safe_path)
    if type_name is not None:
        message = message.replace("{type_name}", type_name)
    return message


def _project_os_error(
    error_type: type[OSError],
    error_number: int | None,
    *,
    message: str,
    config_path: Path,
) -> OSError:
    """Preserve filesystem semantics with a diagnostic-safe filename."""
    if error_number is None:
        return error_type(message)
    return error_type(error_number, message, redact_sensitive_path(config_path))


def config_source_name(context: Mapping[str, Any]) -> str:
    """Return the shared operator-facing name for the selected source."""
    return "memory" if context.get("config_dict") is not None else "file"


def log_config_source(
    context: Mapping[str, Any],
    log: Callable[[str], Any],
    *,
    label: str,
) -> None:
    """Emit one value-free, consistent configuration-source diagnostic."""
    log(f"Loading {label} configuration from {config_source_name(context)}")


def _load_config_mapping_from_context_impl(
    context: Mapping[str, Any],
    *,
    default_config_path: str | Path = "config.yaml",
    allow_empty: bool = False,
    allow_missing_file: bool = False,
    missing_path_message: str = "config_dict or config_path is required in context",
    missing_file_message: str = "Configuration file not found: {config_path}",
    read_error_message: str = "Unable to read configuration file: {config_path}",
    parse_error_message: str = "Unable to parse configuration file: {config_path}",
    empty_message: str = "Configuration is empty: {config_path}",
    config_dict_non_mapping_message: str = (
        "config_dict must be a mapping, got {type_name}"
    ),
    file_non_mapping_message: str = (
        "Configuration file must contain a mapping, got {type_name}"
    ),
    file_loader: ConfigFileLoader = yaml.safe_load,
) -> tuple[dict[str, Any], Path]:
    """Load an isolated workflow config mapping and its source-path anchor.

    An in-memory ``config_dict`` is authoritative. Its optional ``config_path``
    is retained only as the anchor for resolving relative paths and is never
    opened. When that anchor is absent, ``default_config_path`` is returned.

    The source contract is uniform across agents: an absent ``config_dict`` or
    a value of ``None`` selects the file fallback; a non-``None`` mapping is
    authoritative (including an empty mapping); and every other value is a
    structure error. Empty mappings are accepted only with ``allow_empty``.

    Diagnostic templates may contain ``{config_path}`` and ``{type_name}``.
    Only those documented tokens are substituted, and paths are redacted before
    substitution. YAML and I/O exception details are deliberately discarded so
    source lines, credential values, and sensitive paths cannot reach errors.

    Inline credentials are valid runtime configuration. This loader isolates
    them but does not apply durable-artifact credential policy.
    """
    config_path_value = context.get("config_path")
    config_path = (
        Path(config_path_value) if config_path_value else Path(default_config_path)
    )

    has_config_dict = context.get("config_dict") is not None
    if has_config_dict:
        config_value = context.get("config_dict")
        if not isinstance(config_value, Mapping):
            raise ConfigStructureError(
                _render_message(
                    config_dict_non_mapping_message,
                    config_path=config_path,
                    type_name=type(config_value).__name__,
                )
            )
        config = _isolate_config_mapping(config_value)
    else:
        if not config_path_value:
            raise ConfigSourceError(
                _render_message(missing_path_message, config_path=config_path)
            )

        exists_error: tuple[type[OSError], int | None] | None = None
        try:
            config_exists = config_path.exists()
        except OSError as error:
            exists_error = (type(error), error.errno)
            config_exists = False
        if exists_error is not None:
            raise _project_os_error(
                *exists_error,
                message=_render_message(
                    read_error_message,
                    config_path=config_path,
                ),
                config_path=config_path,
            ) from None

        if not config_exists:
            if allow_missing_file:
                return {}, config_path
            raise FileNotFoundError(
                _render_message(missing_file_message, config_path=config_path)
            )

        loaded: Any = None
        read_error: tuple[type[OSError], int | None] | None = None
        parse_failed = False
        decode_failed = False
        try:
            with config_path.open(encoding="utf-8") as stream:
                loaded = file_loader(stream)
        except OSError as error:
            read_error = (type(error), error.errno)
        except UnicodeError:
            decode_failed = True
        except Exception:
            # Parser implementations may raise built-ins in addition to their
            # documented exception hierarchy. Do not retain the parser object,
            # source buffer, or exception graph on the projected failure.
            parse_failed = True

        # Raise only after leaving the handler. ``raise ... from None`` suppresses
        # display but still keeps ``__context__``; these branches sever it.
        if read_error is not None:
            raise _project_os_error(
                *read_error,
                message=_render_message(
                    read_error_message,
                    config_path=config_path,
                ),
                config_path=config_path,
            ) from None
        if decode_failed:
            raise UnicodeError(
                _render_message(read_error_message, config_path=config_path)
            ) from None
        if parse_failed:
            raise ConfigParseError(
                _render_message(parse_error_message, config_path=config_path)
            ) from None

        if loaded is None:
            config = {}
        elif not isinstance(loaded, Mapping):
            raise ConfigStructureError(
                _render_message(
                    file_non_mapping_message,
                    config_path=config_path,
                    type_name=type(loaded).__name__,
                )
            )
        else:
            config = _isolate_config_mapping(loaded)

    if not config and not allow_empty:
        raise ConfigEmptyError(_render_message(empty_message, config_path=config_path))
    return config, config_path


def load_config_mapping_from_context(
    context: Mapping[str, Any],
    *,
    default_config_path: str | Path = "config.yaml",
    allow_empty: bool = False,
    allow_missing_file: bool = False,
    missing_path_message: str = "config_dict or config_path is required in context",
    missing_file_message: str = "Configuration file not found: {config_path}",
    read_error_message: str = "Unable to read configuration file: {config_path}",
    parse_error_message: str = "Unable to parse configuration file: {config_path}",
    empty_message: str = "Configuration is empty: {config_path}",
    config_dict_non_mapping_message: str = (
        "config_dict must be a mapping, got {type_name}"
    ),
    file_non_mapping_message: str = (
        "Configuration file must contain a mapping, got {type_name}"
    ),
    file_loader: ConfigFileLoader = yaml.safe_load,
) -> tuple[dict[str, Any], Path]:
    """Load config while keeping request-owned values out of public tracebacks."""
    failure: _ProjectedFailure | None = None
    try:
        return _load_config_mapping_from_context_impl(
            context,
            default_config_path=default_config_path,
            allow_empty=allow_empty,
            allow_missing_file=allow_missing_file,
            missing_path_message=missing_path_message,
            missing_file_message=missing_file_message,
            read_error_message=read_error_message,
            parse_error_message=parse_error_message,
            empty_message=empty_message,
            config_dict_non_mapping_message=config_dict_non_mapping_message,
            file_non_mapping_message=file_non_mapping_message,
            file_loader=file_loader,
        )
    except (OSError, ValueError) as error:
        failure = _capture_failure(error)

    del (
        context,
        default_config_path,
        allow_empty,
        allow_missing_file,
        missing_path_message,
        missing_file_message,
        read_error_message,
        parse_error_message,
        empty_message,
        config_dict_non_mapping_message,
        file_non_mapping_message,
        file_loader,
    )
    assert failure is not None
    _raise_failure(failure)


__all__ = [
    "ConfigEmptyError",
    "ConfigFileLoader",
    "ConfigLoadError",
    "ConfigParseError",
    "ConfigSourceError",
    "ConfigStructureError",
    "config_source_name",
    "load_config_mapping_from_context",
    "log_config_source",
]
