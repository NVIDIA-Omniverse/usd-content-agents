# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Secret-safe persistence for Physics Agent worker configuration."""

import asyncio
import logging
import math
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, NoReturn, Protocol

import yaml
from fastapi import HTTPException
from world_understanding.utils.credentials import (
    InlineSecretError,
    ensure_no_inline_secrets,
)
from yaml.tokens import AliasToken

logger = logging.getLogger(__name__)

INVALID_PIPELINE_CONFIG_DETAIL = "Pipeline configuration is invalid"
PIPELINE_CONFIG_WRITE_FAILED_DETAIL = "Failed to persist pipeline configuration"
INVALID_DURABLE_INPUT_DETAIL = "Request content cannot contain inline credentials"


def _raise_http_exception(status_code: int, detail: str) -> NoReturn:
    """Raise a value-free public error from a frame containing only safe data."""
    raise HTTPException(status_code=status_code, detail=detail) from None


def _log_inline_secret_rejection(error: InlineSecretError) -> None:
    """Keep one bounded field path for operators without rendering the error."""
    field_path = error.paths[0] if error.paths else "unavailable"
    logger.warning("durable_input_rejected field_path=%s", field_path)


class _DuplicateMappingKeyError(yaml.constructor.ConstructorError):
    """A YAML mapping defines the same effective key more than once."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects shadowed mapping values."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    """Construct a mapping only when every effective key is unique."""
    loader.flatten_mapping(node)
    seen: set[Any] = set()
    for key_node, _value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in seen:
                raise _DuplicateMappingKeyError(
                    None,
                    None,
                    "duplicate mapping key",
                    key_node.start_mark,
                )
            seen.add(key)
        except TypeError:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "unhashable mapping key",
                key_node.start_mark,
            ) from None
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _is_supported_durable_yaml_value(
    value: Any,
    *,
    active_container_ids: set[int] | None = None,
) -> bool:
    """Accept only acyclic JSON-like values at the durable YAML boundary."""
    if value is None or type(value) in {str, bool, int}:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if not isinstance(value, list | dict):
        # SafeLoader can construct bytes, dates, sets, and other tagged values.
        # Durable request YAML is JSON-like; only None may cross into the
        # canonical persisted form.
        return False

    active_ids = active_container_ids if active_container_ids is not None else set()
    container_id = id(value)
    if container_id in active_ids:
        return False
    active_ids.add(container_id)
    try:
        if isinstance(value, list):
            return all(
                _is_supported_durable_yaml_value(
                    item,
                    active_container_ids=active_ids,
                )
                for item in value
            )
        return all(
            type(key) is str
            and _is_supported_durable_yaml_value(
                item,
                active_container_ids=active_ids,
            )
            for key, item in value.items()
        )
    finally:
        active_ids.remove(container_id)


class SessionCleaner(Protocol):
    """Minimal session cleanup contract needed by request persistence."""

    async def delete_session(self, session_id: str) -> bool:
        """Delete one request-owned session."""
        ...  # pragma: no cover - declaration-only protocol method


def validate_pipeline_config(pipeline_config: dict[str, Any]) -> None:
    """Reject credentials before a worker config reaches durable storage."""
    ensure_no_inline_secrets(
        pipeline_config,
        context="physics pipeline configuration",
    )


def _canonicalize_durable_request_content(
    content: dict[str, Any],
    *,
    documents: dict[str, str],
    context: str,
) -> dict[str, str] | None:
    """Return canonical documents, or ``None`` for a value-free rejection."""
    # The raw representation is a security boundary of its own. YAML parsers
    # discard comments and normally let a later duplicate key shadow an earlier
    # value, so scanning only the parsed structure can miss durable credentials.
    try:
        ensure_no_inline_secrets(
            {
                "content": content,
                "yaml_documents": documents,
            },
            context=context,
        )
    except InlineSecretError as error:
        _log_inline_secret_rejection(error)
        return None

    parsed_documents: dict[str, Any] = {}
    canonical_documents: dict[str, str] = {}
    for name, document in documents.items():
        if not document.strip():
            canonical_documents[name] = document
            continue
        alias_rejected = False
        construction_rejected = False
        syntax_invalid = False
        parsed: Any = None
        try:
            # SafeLoader resolves aliases while constructing objects, and merge
            # aliases can expand one size-bounded request into an unbounded
            # in-memory mapping before canonical serialization. Token scanning
            # is non-constructing, so reject every alias at this boundary first.
            alias_rejected = any(
                isinstance(token, AliasToken) for token in yaml.scan(document)
            )
            if not alias_rejected:
                parsed = yaml.load(document, Loader=_UniqueKeySafeLoader)
        except yaml.constructor.ConstructorError:
            construction_rejected = True
        except yaml.YAMLError:
            syntax_invalid = True

        if alias_rejected or construction_rejected:
            return None
        if syntax_invalid:
            # Route-specific schema validation owns syntax diagnostics. Retain
            # the raw document so it can return the existing value-free 400.
            canonical_documents[name] = document
            continue
        if not _is_supported_durable_yaml_value(parsed):
            return None
        parsed_documents[name] = parsed
        canonical_documents[name] = yaml.safe_dump(
            parsed,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    try:
        ensure_no_inline_secrets(parsed_documents, context=context)
    except InlineSecretError as error:
        _log_inline_secret_rejection(error)
        return None
    return canonical_documents


def validate_durable_request_content(
    content: dict[str, Any],
    *,
    yaml_documents: dict[str, str] | None = None,
    context: str,
) -> dict[str, str]:
    """Reject secrets and canonicalize YAML before any durable session write.

    Pipeline and predict prompts already pass through ``validate_pipeline_config``.
    Refine and tune persist free-form prompts, descriptions, and scenario YAML
    without building that pipeline config first. Scan YAML as raw text so comments
    and shadowed values cannot bypass inspection, then parse it with duplicate-key
    rejection so nested credential keys are inspected structurally. Successful
    documents are returned in canonical form; callers must persist those returned
    bytes rather than the original request text.
    """
    documents = yaml_documents or {}
    canonical_documents = _canonicalize_durable_request_content(
        content,
        documents=documents,
        context=context,
    )
    if canonical_documents is None:
        # The public exception traceback is observable diagnostic state. Remove
        # every caller-provided value from this frame before creating it; the
        # parsing helper has already returned, so its raw locals are not linked.
        del content
        del yaml_documents
        del documents
        del context
        _raise_http_exception(400, INVALID_DURABLE_INPUT_DETAIL)
    return canonical_documents


async def avalidate_durable_request_content(
    content: dict[str, Any],
    *,
    yaml_documents: dict[str, str] | None = None,
    context: str,
) -> dict[str, str]:
    """Validate request-owned content without blocking an async route loop."""
    return await asyncio.to_thread(
        validate_durable_request_content,
        content,
        yaml_documents=yaml_documents,
        context=context,
    )


def write_pipeline_config(
    config_path: Path,
    pipeline_config: dict[str, Any],
) -> None:
    """Reject inline credentials, then atomically persist a worker config.

    Serializing directly into ``config_path`` can truncate a valid config when
    YAML rendering or the final write fails. Build an fsynced sibling first,
    then replace the destination in one filesystem operation.
    """
    validate_pipeline_config(pipeline_config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as config_file:
            temporary_path = Path(config_file.name)
            yaml.safe_dump(pipeline_config, config_file, default_flow_style=False)
            config_file.flush()
            os.fsync(config_file.fileno())
        os.replace(temporary_path, config_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


async def _cleanup_owned_session(
    *,
    session_manager: SessionCleaner,
    session_id: str,
    session_created_here: bool,
) -> None:
    """Best-effort cleanup for a session created by the rejected request."""
    if not session_created_here:
        return
    try:
        deleted = await session_manager.delete_session(session_id)
        if not deleted:
            logger.error("Failed to clean up rejected pipeline session")
    except Exception:  # pragma: no cover - defensive cleanup containment
        logger.error("Failed to clean up rejected pipeline session")


async def build_and_validate_pipeline_config(
    *,
    config_factory: Callable[[], dict[str, Any]],
    session_manager: SessionCleaner,
    session_id: str,
    session_created_here: bool,
) -> dict[str, Any]:
    """Build and validate config before callers acquire scarce resources."""
    pipeline_config: dict[str, Any] | None = None
    failure: tuple[int, str] | None = None
    try:
        pipeline_config = config_factory()
    except ValueError:
        failure = (400, INVALID_PIPELINE_CONFIG_DETAIL)
    except Exception:
        failure = (500, PIPELINE_CONFIG_WRITE_FAILED_DETAIL)

    if failure is not None:
        await _cleanup_owned_session(
            session_manager=session_manager,
            session_id=session_id,
            session_created_here=session_created_here,
        )
        status_code, detail = failure
        # A factory can be a closure over request credentials. Neither it nor a
        # partially produced config may survive in the replacement traceback.
        del failure
        del config_factory
        del pipeline_config
        del session_manager
        del session_id
        del session_created_here
        _raise_http_exception(status_code, detail)

    failure = None
    try:
        assert pipeline_config is not None
        await asyncio.to_thread(validate_pipeline_config, pipeline_config)
    except InlineSecretError:
        failure = (400, INVALID_PIPELINE_CONFIG_DETAIL)
    except Exception:
        failure = (500, PIPELINE_CONFIG_WRITE_FAILED_DETAIL)

    if failure is not None:
        await _cleanup_owned_session(
            session_manager=session_manager,
            session_id=session_id,
            session_created_here=session_created_here,
        )
        status_code, detail = failure
        del failure
        del config_factory
        del pipeline_config
        del session_manager
        del session_id
        del session_created_here
        _raise_http_exception(status_code, detail)
    assert pipeline_config is not None
    return pipeline_config


async def build_and_write_pipeline_config(
    *,
    config_factory: Callable[[], dict[str, Any]],
    config_path: Path,
    session_manager: SessionCleaner,
    session_id: str,
    session_created_here: bool,
) -> dict[str, Any]:
    """Build and persist config with ownership-safe cleanup and fixed errors."""
    pipeline_config: dict[str, Any] | None = None
    build_failure: tuple[int, str] | None = None
    try:
        pipeline_config = await build_and_validate_pipeline_config(
            config_factory=config_factory,
            session_manager=session_manager,
            session_id=session_id,
            session_created_here=session_created_here,
        )
    except HTTPException as error:
        if isinstance(error.detail, str):
            build_failure = (error.status_code, error.detail)
        else:  # pragma: no cover - private callee only emits string details
            build_failure = (500, PIPELINE_CONFIG_WRITE_FAILED_DETAIL)

    if build_failure is not None:
        # build_and_validate_pipeline_config already performed ownership-aware
        # cleanup. Replacing its exception here avoids retaining this wrapper's
        # factory closure while also ensuring the session is deleted only once.
        status_code, detail = build_failure
        del build_failure
        del config_factory
        del pipeline_config
        del config_path
        del session_manager
        del session_id
        del session_created_here
        _raise_http_exception(status_code, detail)

    assert pipeline_config is not None
    failure: tuple[int, str] | None = None
    writer: asyncio.Task[None] | None = None
    try:
        # Keep the secret scanner inseparable from this persistence boundary.
        # Validation before resource admission is only an early rejection;
        # write_pipeline_config revalidates the exact object immediately
        # before it creates or replaces any durable artifact.
        writer = asyncio.create_task(
            asyncio.to_thread(write_pipeline_config, config_path, pipeline_config)
        )
        try:
            await asyncio.shield(writer)
        except asyncio.CancelledError:
            # A worker thread cannot be cancelled. Do not let callers restore
            # an older snapshot while this writer can still replace it again.
            # Drain the exact publication first; cancellation remains the
            # authoritative result once the filesystem mutation quiesces.
            while not writer.done():
                try:
                    await asyncio.shield(writer)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            with suppress(Exception, asyncio.CancelledError):
                writer.result()
            raise
        return pipeline_config
    except InlineSecretError:
        failure = (400, INVALID_PIPELINE_CONFIG_DETAIL)
    except Exception:
        failure = (500, PIPELINE_CONFIG_WRITE_FAILED_DETAIL)

    if failure is not None:
        await _cleanup_owned_session(
            session_manager=session_manager,
            session_id=session_id,
            session_created_here=session_created_here,
        )
        status_code, detail = failure
        del failure
        del build_failure
        del config_factory
        del pipeline_config
        del config_path
        del session_manager
        del session_id
        del session_created_here
        # A completed task retains its exception and that exception's raw
        # traceback. Drop the task before constructing the public traceback so
        # provider diagnostics cannot survive through this frame's locals.
        del writer
        _raise_http_exception(status_code, detail)
    return pipeline_config
