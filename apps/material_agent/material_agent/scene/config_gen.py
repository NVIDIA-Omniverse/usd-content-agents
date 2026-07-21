# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-asset config generation for large-scene pipeline.

Deep-copies the scene config template, injects prim_path scoping,
and forces layer_only output for each sub-asset.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from world_understanding.agentic.config import clone_config_containers
from world_understanding.utils.credentials import (
    create_directory_with_safe_diagnostics,
    ensure_no_inline_secrets,
    find_inline_secret_paths,
    path_exists_with_safe_diagnostics,
    read_text_with_safe_diagnostics,
    redact_sensitive_config,
    redact_sensitive_path,
    resolve_path_with_safe_diagnostics,
)

from .manifest import PayloadGroup, SceneManifest, SubAsset

logger = logging.getLogger(__name__)

_REDACTED_CREDENTIAL = "<redacted>"
_MISSING = object()


class CredentialOverlayError(ValueError):
    """Base class for value-free scene credential-overlay mismatches."""

    code = "credential_overlay_mismatch"


class MissingCredentialSourceError(CredentialOverlayError):
    """The current source config no longer supplies a required credential."""

    code = "credential_source_missing"


class CredentialOverlayShapeError(CredentialOverlayError):
    """Durable config shape cannot accept the current source credential."""

    code = "credential_overlay_shape_mismatch"


def _is_redacted_credential(original: Any, redacted: Any) -> bool:
    """Return whether ``redacted`` replaced an inline credential value."""
    return bool(redacted == _REDACTED_CREDENTIAL and original != redacted)


def _drop_redacted_credentials(original: Any, redacted: Any) -> Any:
    """Copy a config while omitting credential values instead of masking them."""
    if _is_redacted_credential(original, redacted):
        # Mapping values can be omitted entirely by their caller. Sequence
        # values need a non-secret tombstone so their positional identity is
        # retained for the in-memory overlay.
        return None
    if isinstance(original, dict) and isinstance(redacted, dict):
        durable: dict[Any, Any] = {}
        for key, value in original.items():
            redacted_value = redacted[key]
            if _is_redacted_credential(value, redacted_value):
                continue
            durable[key] = _drop_redacted_credentials(value, redacted_value)
        return durable
    if isinstance(original, list) and isinstance(redacted, list):
        return [
            _drop_redacted_credentials(value, redacted[index])
            for index, value in enumerate(original)
        ]
    if isinstance(original, tuple) and isinstance(redacted, tuple):
        return tuple(
            _drop_redacted_credentials(value, redacted[index])
            for index, value in enumerate(original)
        )
    return clone_config_containers(original)


def _credential_token_paths(
    original: Any,
    redacted: Any,
    path: tuple[Any, ...] = (),
) -> tuple[tuple[Any, ...], ...]:
    """Return structural paths for values replaced by constant redaction."""
    if _is_redacted_credential(original, redacted):
        return (path,)
    if isinstance(original, dict) and isinstance(redacted, dict):
        return tuple(
            credential_path
            for key, value in original.items()
            for credential_path in _credential_token_paths(
                value, redacted[key], (*path, key)
            )
        )
    if isinstance(original, list | tuple) and isinstance(redacted, list | tuple):
        return tuple(
            credential_path
            for index, value in enumerate(original)
            for credential_path in _credential_token_paths(
                value, redacted[index], (*path, index)
            )
        )
    return ()


def _value_at_path(value: Any, path: tuple[Any, ...]) -> Any:
    """Return a nested value, or a private sentinel when its shape drifted."""
    current = value
    for token in path:
        if isinstance(current, dict):
            if token not in current:
                return _MISSING
            current = current[token]
        elif isinstance(current, list | tuple) and isinstance(token, int):
            if token < 0 or token >= len(current):
                return _MISSING
            current = current[token]
        else:
            return _MISSING
    return current


def _has_nonsecret_context(value: Any) -> bool:
    """Return whether a projection contains a non-tombstone scalar value."""
    if isinstance(value, dict):
        return any(_has_nonsecret_context(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_has_nonsecret_context(item) for item in value)
    return value is not None


def _credential_binding_matches(
    target: Any,
    source: Any,
    redacted: Any,
    credential_path: tuple[Any, ...],
) -> bool:
    """Prove one credential still belongs to its nearest durable context."""
    parent_path = credential_path[:-1]
    credential_only_context_matches = False
    while True:
        source_parent = _value_at_path(source, parent_path)
        redacted_parent = _value_at_path(redacted, parent_path)
        target_parent = _value_at_path(target, parent_path)

        if isinstance(source_parent, list | tuple) and isinstance(
            redacted_parent, list | tuple
        ):
            if not isinstance(target_parent, list | tuple):
                return False
            projection = _drop_redacted_credentials(source_parent, redacted_parent)
            if list(projection) != list(target_parent):
                return False
            credential_only_context_matches = True
            # A list containing only credential tombstones proves positional
            # identity, but not which provider or endpoint owns the value.
            # Keep climbing until the nearest meaningful durable ancestor so a
            # resumed config cannot bind a current credential to stale sibling
            # context. Lists with their own non-secret members remain a complete
            # local binding and can return immediately.
            if _has_nonsecret_context(projection):
                return True

        if isinstance(source_parent, dict) and isinstance(redacted_parent, dict):
            projection = _drop_redacted_credentials(source_parent, redacted_parent)
            credential_only_context_matches = (
                isinstance(target_parent, dict) and projection == target_parent
            )
            context_projection = clone_config_containers(projection)
            next_token = credential_path[len(parent_path)]
            if isinstance(context_projection, dict):
                context_projection.pop(next_token, None)
            if _has_nonsecret_context(context_projection):
                return isinstance(target_parent, dict) and projection == target_parent

        if not parent_path:
            break
        parent_path = parent_path[:-1]

    return credential_only_context_matches


def _credential_contexts_match(target: Any, source: Any, redacted: Any) -> bool:
    """Return whether every credential is bound to unchanged durable context."""
    return all(
        _credential_binding_matches(target, source, redacted, credential_path)
        for credential_path in _credential_token_paths(source, redacted)
    )


def _durable_config(config: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return a persistence-safe config and the omitted credential field paths."""
    credential_paths = find_inline_secret_paths(config)
    if not credential_paths:
        cloned = clone_config_containers(config)
        if not isinstance(cloned, dict):  # pragma: no cover - input contract
            raise AssertionError("durable configuration clone must be a dictionary")
        return cloned, ()

    redacted = redact_sensitive_config(config)
    durable = _drop_redacted_credentials(config, redacted)
    if not isinstance(durable, dict):  # pragma: no cover - input contract guard
        raise TypeError("Generated pipeline config must be a mapping")
    ensure_no_inline_secrets(durable, context="generated scene pipeline config")
    return durable, credential_paths


def _overlay_inline_credentials(
    target: Any,
    source: Any,
    redacted: Any,
    *,
    _context_validated: bool = False,
) -> Any:
    """Overlay only inline credential values from ``source`` onto ``target``."""
    if (
        not _context_validated
        and isinstance(source, dict | list | tuple)
        and not _credential_contexts_match(target, source, redacted)
    ):
        return target

    if _is_redacted_credential(source, redacted):
        return clone_config_containers(source)

    if isinstance(source, dict) and isinstance(redacted, dict):
        if not isinstance(target, dict):
            return target

        dict_result = clone_config_containers(target)
        for key, source_value in source.items():
            redacted_value = redacted[key]
            if _is_redacted_credential(source_value, redacted_value):
                dict_result[key] = clone_config_containers(source_value)
                continue
            if isinstance(
                source_value, dict | list | tuple
            ) and find_inline_secret_paths(source_value):
                existing = dict_result.get(key)
                overlaid = _overlay_inline_credentials(
                    existing,
                    source_value,
                    redacted_value,
                    _context_validated=True,
                )
                if overlaid is not existing:
                    dict_result[key] = overlaid
        return dict_result

    if isinstance(source, list | tuple) and isinstance(redacted, list | tuple):
        if not isinstance(target, list | tuple):
            return target
        target_items = list(clone_config_containers(target))
        list_result = clone_config_containers(target_items)
        for index, source_value in enumerate(source):
            if index >= len(target_items) or not find_inline_secret_paths(source_value):
                continue
            list_result[index] = _overlay_inline_credentials(
                list_result[index],
                source_value,
                redacted[index],
                _context_validated=True,
            )
        return list_result

    return target


def _write_durable_config(path: Path, config: dict[str, Any]) -> tuple[str, ...]:
    """Atomically persist a generated config with inline credentials omitted."""
    durable, credential_paths = _durable_config(config)
    create_directory_with_safe_diagnostics(
        path.parent,
        label="generated configuration directory",
    )
    diagnostic_path = redact_sensitive_path(path)
    temp_path: Path | None = None
    try:
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                encoding="utf-8",
                delete=False,
            ) as stream:
                temp_path = Path(stream.name)
                yaml.safe_dump(
                    durable,
                    stream,
                    default_flow_style=False,
                    sort_keys=False,
                )
                stream.flush()
                os.fsync(stream.fileno())
            temp_path.replace(path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    except OSError as error:
        # The OS exception may carry either the raw output path or the derived
        # temporary filename. Preserve its subtype and errno while replacing
        # path-bearing diagnostics with the same safe projection used by the
        # rest of the generated-config boundary.
        raise type(error)(
            error.errno,
            "Unable to persist generated configuration",
            diagnostic_path,
        ) from None
    return credential_paths


def _build_sub_asset_config(
    sub_asset: SubAsset,
    scene_config: dict[str, Any],
    output_path: Path,
    scene_config_dir: Path | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Build an isolated per-asset config from the scene config template.

    The generated config:
    - Rebases all relative paths so they resolve correctly from output_path
    - Sets input.prim_path to scope the pipeline to the sub-asset
    - Forces output.layer_only = true and output.flatten_output = false
    - Sets project.name and session_id to a sanitized sub-asset name
    - Removes the scene section (not needed for per-asset runs)

    Args:
        sub_asset: The sub-asset to generate config for.
        scene_config: The scene-level config dict (will be deep-copied).
        output_path: Path anchor for the generated config.
        scene_config_dir: Directory of the original scene config (for rebasing
            relative paths). If None, paths are kept as-is.

    Returns:
        Generated config dictionary. The caller decides what may be persisted.
    """
    config = clone_config_containers(scene_config)

    # Remove scene section (not relevant for per-asset pipeline)
    config.pop("scene", None)

    # Set project identity — use caller-supplied session_id if provided
    # (ensures uniqueness when multiple sub-assets share the same name)
    safe_name = _sanitize_name(session_id if session_id else sub_asset.name)
    workdir_name = _sanitize_name(safe_name)
    project = config.setdefault("project", {})
    project["name"] = safe_name
    project["session_id"] = safe_name

    # Rebase relative paths from scene config dir to generated config dir
    if scene_config_dir is not None:
        output_dir = output_path.parent
        _rebase_paths(
            config,
            resolve_path_with_safe_diagnostics(
                scene_config_dir,
                label="scene configuration directory",
            ),
            resolve_path_with_safe_diagnostics(
                output_dir,
                label="generated configuration directory",
            ),
        )

    # Per-asset pipeline outputs must not share the scene-level working dir.
    # Keep each generated config self-contained beside its YAML file.
    project["working_dir"] = f".{workdir_name}"

    input_section = config.setdefault("input", {})

    # Use extracted USD if available (much smaller than full scene).
    # The extracted USD already contains only this sub-asset's subtree,
    # so prim_path scoping is still needed for the pipeline to find
    # the correct prims within the preserved hierarchy.
    if sub_asset.extracted_usd:
        extracted_path = Path(sub_asset.extracted_usd)
        if path_exists_with_safe_diagnostics(
            extracted_path,
            label="extracted USD path",
        ):
            resolved_extracted_path = resolve_path_with_safe_diagnostics(
                extracted_path,
                label="extracted USD path",
            )
            resolved_output_dir = resolve_path_with_safe_diagnostics(
                output_path.parent,
                label="generated configuration directory",
            )
            try:
                rel = os.path.relpath(resolved_extracted_path, resolved_output_dir)
            except ValueError:
                rel = str(resolved_extracted_path)
            input_section["usd_path"] = rel
            logger.info(
                "  Using extracted USD: %s (instead of full scene)",
                redact_sensitive_path(rel),
            )

    # Set prim_path scoping — needed even with extracted USD because
    # the extracted file preserves the full prim hierarchy
    input_section["prim_path"] = sub_asset.prim_path

    # Force layer-only output (material layer for composition)
    output_section = config.setdefault("output", {})
    output_section["layer_only"] = True
    output_section["flatten_output"] = False

    # Configure per-asset steps for scene mode.
    # The collect step handles unified apply against the master scene,
    # so per-asset apply and render are disabled.
    steps = config.get("steps", {})

    # Disable apply (collect step handles unified apply)
    apply_config = steps.get("apply", {})
    apply_config["enabled"] = False
    apply_config["layer_only"] = True
    apply_config["flatten_output"] = False
    steps["apply"] = apply_config

    # Disable render (collect step handles scene-level render)
    render_config = steps.get("render", {})
    render_config["enabled"] = False
    steps["render"] = render_config

    # Enable restore_usd so predictions use original topology paths (not
    # the SO-optimized ones).  This ensures prim paths in predictions match
    # the base scene and resolve correctly in the unified collect apply.
    restore_config = steps.get("restore_usd", {})
    restore_config["enabled"] = True
    steps["restore_usd"] = restore_config

    # Inject split context into VLM prompt if this asset was produced by
    # splitting a larger container — gives the VLM global context about
    # where this asset fits in the scene hierarchy.
    if sub_asset.split_context:
        _inject_split_context(config, sub_asset)

    return config


def generate_sub_asset_config(
    sub_asset: SubAsset,
    scene_config: dict[str, Any],
    output_path: Path,
    scene_config_dir: Path | None = None,
    session_id: str | None = None,
) -> Path:
    """Generate a credential-safe per-asset YAML config.

    Inline credential values remain in the caller-owned source config and are
    omitted from the generated artifact. ``prepare_sub_asset_runtime_configs``
    restores them only in a per-run in-memory config.
    """
    config = _build_sub_asset_config(
        sub_asset,
        scene_config,
        output_path,
        scene_config_dir=scene_config_dir,
        session_id=session_id,
    )
    sub_asset.config_credential_paths = list(_write_durable_config(output_path, config))
    logger.info(
        "Generated config for '%s': %s",
        sub_asset.name,
        redact_sensitive_path(output_path),
    )
    return output_path


# Path keys in the config that contain file/directory paths needing rebasing
_PATH_KEYS = frozenset(
    {
        "reference_image_uris",
        "usd_path",
        "path",
        "working_dir",
    }
)

_PATH_LIST_KEYS = frozenset(
    {
        "reference_images",
        "reference_image_uris",
        "reference_pdfs",
    }
)

_STEP1X_PATH_KEYS = frozenset(
    {
        "runtime_dir",
        "model_dir",
        "cache_dir",
        "output_dir",
        "python_executable",
        "edit_script",
    }
)
_STEP1X_CONFIG_BLOCKS = frozenset({"step1x", "step1x_material_anything"})


def _rebase_paths(
    config: dict,
    old_base: Path,
    new_base: Path,
    *,
    path: tuple[str, ...] = (),
) -> None:
    """Rebase relative paths in config from old_base to new_base (in-place).

    Walks the config dict recursively. For known path keys, converts
    relative paths so they resolve to the same absolute location from
    the new base directory.
    """
    for key, value in config.items():
        child_path = (*path, key)
        if isinstance(value, dict):
            _rebase_paths(value, old_base, new_base, path=child_path)
        elif isinstance(value, list):
            if key in _PATH_LIST_KEYS:
                for index, item in enumerate(value):
                    if isinstance(item, str):
                        value[index] = _rebase_path_value(item, old_base, new_base)
            else:
                for item in value:
                    if isinstance(item, dict):
                        _rebase_paths(item, old_base, new_base, path=child_path)
        elif isinstance(value, str) and (
            key in _PATH_KEYS or _is_step1x_path_key(key, path)
        ):
            config[key] = _rebase_path_value(value, old_base, new_base)


def _is_step1x_path_key(key: str, path: tuple[str, ...]) -> bool:
    return (
        key in _STEP1X_PATH_KEYS
        and len(path) >= 3
        and path[-3:-1] == ("steps", "create_materials")
        and path[-1] in _STEP1X_CONFIG_BLOCKS
    )


def _rebase_path_value(value: str, old_base: Path, new_base: Path) -> str:
    if value == "" or urlparse(value).scheme or Path(value).is_absolute():
        return value
    abs_path = resolve_path_with_safe_diagnostics(
        old_base / value,
        label="configuration path",
    )
    try:
        return str(os.path.relpath(abs_path, new_base))
    except ValueError:
        # Cross-drive on Windows; fall back to absolute.
        return str(abs_path)


def generate_all_configs(
    manifest: SceneManifest,
    scene_config: dict,
    configs_dir: Path,
    scene_config_dir: Path | None = None,
    names_filter: list[str] | None = None,
) -> SceneManifest:
    """Generate per-asset configs for all processable assets.

    Updates the manifest with config paths and working directories.

    Args:
        manifest: Scene manifest with detected sub-assets.
        scene_config: The scene-level config dict.
        configs_dir: Directory to write generated configs.
        scene_config_dir: Directory of the original scene config (for rebasing
            relative paths).
        names_filter: Optional name/path filter for assets.

    Returns:
        Updated SceneManifest.
    """
    assets = manifest.get_processable_assets(names_filter)
    logger.info(f"Generating configs for {len(assets)} sub-assets")

    # Build unique safe names: append ID suffix when names collide
    safe_names = _unique_safe_names(assets)

    for i, sa in enumerate(assets, 1):
        safe_name = safe_names[sa.id]
        config_path = configs_dir / f"{safe_name}.yaml"

        logger.info(f"[{i}/{len(assets)}] Generating config for '{sa.name}'")
        try:
            generate_sub_asset_config(
                sa,
                scene_config,
                config_path,
                scene_config_dir=scene_config_dir,
                session_id=safe_name,
            )
            sa.config_path = str(config_path)
            # Working dir will be relative to config file location:
            # configs_dir/.{safe_name}
            sa.working_dir = str(configs_dir / f".{safe_name}")
        except Exception:
            logger.exception(f"Failed to generate config for '{sa.name}'")
            sa.status = "failed"

    return manifest


def _load_and_sanitize_generated_config(
    config_path: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Load a generated config and remove credentials left by older versions."""
    diagnostic_path = redact_sensitive_path(config_path)
    if not path_exists_with_safe_diagnostics(
        config_path,
        label="generated configuration",
    ):
        raise FileNotFoundError(f"Generated config not found: {diagnostic_path}")
    try:
        source = read_text_with_safe_diagnostics(
            config_path,
            label="generated configuration",
        )
        loaded = yaml.safe_load(source)
    except (UnicodeError, yaml.YAMLError):
        # Legacy generated files may contain credentials, and parser diagnostics
        # can echo their source lines. Invalid UTF-8 can likewise echo source
        # bytes. Expose only the diagnostic-safe selected artifact path.
        raise ValueError(
            f"Unable to parse generated scene config: {diagnostic_path}"
        ) from None
    if not isinstance(loaded, dict):
        raise ValueError(f"Generated config must contain a mapping: {diagnostic_path}")

    durable, legacy_paths = _durable_config(loaded)
    if legacy_paths:
        _write_durable_config(config_path, durable)
        logger.warning(
            "Removed legacy inline credentials from generated scene config: %s",
            diagnostic_path,
        )
    return durable, legacy_paths


def _missing_credential_error(
    *, config_path: Path, missing_paths: set[str]
) -> MissingCredentialSourceError:
    rendered = _render_credential_paths(missing_paths)
    return MissingCredentialSourceError(
        "Cannot rehydrate generated scene config "
        f"{redact_sensitive_path(config_path)}: source scene config is missing "
        "inline credentials at "
        f"{rendered}"
    )


def _runtime_credential_overlay_error(
    *, config_path: Path, missing_paths: set[str]
) -> CredentialOverlayShapeError:
    """Return a value-free error for credential paths lost to shape drift."""
    rendered = _render_credential_paths(missing_paths)
    return CredentialOverlayShapeError(
        "Cannot rehydrate generated scene config "
        f"{redact_sensitive_path(config_path)}: generated config structure "
        "cannot accept inline "
        f"credentials at {rendered}"
    )


def _render_credential_paths(paths: set[str]) -> str:
    """Render a bounded list of credential field paths without their values."""
    rendered = ", ".join(sorted(paths)[:8])
    if len(paths) > 8:
        rendered += f", and {len(paths) - 8} more"
    return rendered


def prepare_sub_asset_runtime_config(
    sub_asset: SubAsset,
    scene_config: dict[str, Any],
    scene_config_dir: Path | None = None,
) -> dict[str, Any]:
    """Build one in-memory per-asset config from a sanitized artifact + source."""
    if not sub_asset.config_path:
        raise ValueError(f"Sub-asset '{sub_asset.name}' has no config_path set")
    config_path = Path(sub_asset.config_path)
    durable, legacy_paths = _load_and_sanitize_generated_config(config_path)

    project = durable.get("project", {})
    generated_session_id = (
        project.get("session_id") if isinstance(project, dict) else None
    )
    source_config = _build_sub_asset_config(
        sub_asset,
        scene_config,
        config_path,
        scene_config_dir=scene_config_dir,
        session_id=str(generated_session_id or config_path.stem),
    )
    available_paths = set(find_inline_secret_paths(source_config))
    required_paths = set(sub_asset.config_credential_paths) | set(legacy_paths)
    missing_paths = required_paths - available_paths
    if missing_paths:
        raise _missing_credential_error(
            config_path=config_path, missing_paths=missing_paths
        )

    runtime = _overlay_inline_credentials(
        durable, source_config, redact_sensitive_config(source_config)
    )
    if not isinstance(runtime, dict):  # pragma: no cover - input contract guard
        raise TypeError("Runtime scene config must be a mapping")
    runtime_paths = set(find_inline_secret_paths(runtime))
    missing_runtime_paths = (required_paths | available_paths) - runtime_paths
    if missing_runtime_paths:
        raise _runtime_credential_overlay_error(
            config_path=config_path, missing_paths=missing_runtime_paths
        )

    sub_asset.config_credential_paths = sorted(available_paths)
    return runtime


def prepare_sub_asset_runtime_configs(
    manifest: SceneManifest,
    scene_config: dict[str, Any],
    scene_config_dir: Path | None = None,
    names_filter: list[str] | None = None,
    assets: list[SubAsset] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build isolated runtime configs keyed by sub-asset ID."""
    selected_assets = (
        assets if assets is not None else manifest.get_processable_assets(names_filter)
    )
    return {
        sub_asset.id: prepare_sub_asset_runtime_config(
            sub_asset, scene_config, scene_config_dir
        )
        for sub_asset in selected_assets
        if sub_asset.config_path
    }


def _build_payload_config(
    payload_group: PayloadGroup,
    scene_config: dict[str, Any],
    output_path: Path,
    scene_config_dir: Path | None = None,
    sibling_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build an isolated per-payload config from the scene config template.

    The generated config:
    - Sets input.usd_path to the payload file path (no prim_path scoping)
    - Disables: optimize_usd, restore_usd, apply, render
    - Enables: build_dataset_usd, build_dataset_prepare_dataset, predict

    Args:
        payload_group: The payload group to generate config for.
        scene_config: The scene-level config dict (will be deep-copied).
        output_path: Path anchor for the generated config.
        scene_config_dir: Directory of the original scene config (for rebasing
            relative paths). If None, paths are kept as-is.

    Returns:
        Generated config dictionary. The caller decides what may be persisted.
    """
    config = clone_config_containers(scene_config)

    # Remove scene section
    config.pop("scene", None)

    # Set project identity
    safe_name = _sanitize_name(payload_group.group_name)
    workdir_name = _sanitize_name(safe_name)
    project = config.setdefault("project", {})
    project["name"] = safe_name
    project["session_id"] = safe_name

    # Rebase relative paths from scene config dir to generated config dir
    if scene_config_dir is not None:
        output_dir = output_path.parent
        _rebase_paths(
            config,
            resolve_path_with_safe_diagnostics(
                scene_config_dir,
                label="scene configuration directory",
            ),
            resolve_path_with_safe_diagnostics(
                output_dir,
                label="generated configuration directory",
            ),
        )

    # Keep payload outputs isolated from the scene-level working directory and
    # sibling payloads.
    project["working_dir"] = f".{workdir_name}"

    input_section = config.setdefault("input", {})

    # For large payloads with a representative file, use that for
    # SO/render/predict (much smaller). Otherwise use modified copy
    # (parents) or original payload file (leaves).
    if payload_group.representative_path:
        input_file = payload_group.representative_path
    else:
        input_file = payload_group.modified_input_path or payload_group.payload_file
    payload_path = Path(input_file)
    resolved_payload_path = resolve_path_with_safe_diagnostics(
        payload_path,
        label="payload path",
    )
    resolved_output_dir = resolve_path_with_safe_diagnostics(
        output_path.parent,
        label="generated configuration directory",
    )
    try:
        rel = os.path.relpath(resolved_payload_path, resolved_output_dir)
    except ValueError:
        rel = str(resolved_payload_path)
    input_section["usd_path"] = rel

    # No prim_path scoping — process the entire payload file
    input_section.pop("prim_path", None)

    # Force layer-only output
    output_section = config.setdefault("output", {})
    output_section["layer_only"] = True
    output_section["flatten_output"] = False

    # Configure steps for payload mode
    steps = config.get("steps", {})

    # Enable restore_usd so predictions use original topology paths.
    # This ensures the output.usd sublayers the original payload
    # (not the SO-optimized file), preserving the drop-in replacement chain.
    restore_config = steps.get("restore_usd", {})
    restore_config["enabled"] = True
    steps["restore_usd"] = restore_config

    # Enable apply — the output.usd IS the "new version" of this payload.
    # layer_only=True means it sublayers the input (drop-in replacement).
    # skip_instance_check=True because payload instances inherit materials
    # via USD composition — no need to traverse all scene instances.
    apply_config = steps.get("apply", {})
    apply_config["enabled"] = True
    apply_config["layer_only"] = True
    apply_config["flatten_output"] = False
    apply_config["skip_instance_check"] = True
    steps["apply"] = apply_config

    # Disable render
    render_config = steps.get("render", {})
    render_config["enabled"] = False
    steps["render"] = render_config

    # Container payloads (parent with children) have no direct meshes —
    # they are pure assembly layers that stitch child outputs together.
    # Disable SO entirely: the flatten step would resolve all child payload
    # arcs and load the entire hierarchy into memory, potentially OOMing.
    optimize_config = steps.get("optimize_usd", {})
    if payload_group.child_payload_files:
        optimize_config["enabled"] = False
        steps["optimize_usd"] = optimize_config
        logger.info("  Container payload: SO disabled (no direct meshes)")
    elif payload_group.representative_path:
        # Representative payloads: use split-only SO (no deinstance, no dedupe).
        # The representative file contains only prototype source prims, so
        # de-instancing is unnecessary and deduplication is counterproductive.
        so_settings = optimize_config.setdefault("scene_optimizer_settings", {})
        so_settings["enableDeinstance"] = False
        so_settings["enableSplitMeshes"] = True
        so_settings["enableDeduplicate"] = False
        steps["optimize_usd"] = optimize_config
        # Store original payload path so the runner can fix output.usd sublayer
        config["_original_payload_file"] = str(
            resolve_path_with_safe_diagnostics(
                payload_group.payload_file,
                label="original payload path",
            )
        )
        logger.info("  Representative mode: SO split-only (no deinstance, no dedupe)")

    # Inject payload context into VLM system prompt so the VLM knows
    # what kind of object it's looking at (critical for simple payloads
    # like a lone tray or carton that lack visual context in isolation)
    _inject_payload_context(config, payload_group, sibling_names=sibling_names)

    return config


def generate_payload_config(
    payload_group: PayloadGroup,
    scene_config: dict[str, Any],
    output_path: Path,
    scene_config_dir: Path | None = None,
    sibling_names: list[str] | None = None,
) -> Path:
    """Generate a credential-safe per-payload YAML config."""
    config = _build_payload_config(
        payload_group,
        scene_config,
        output_path,
        scene_config_dir=scene_config_dir,
        sibling_names=sibling_names,
    )
    payload_group.config_credential_paths = list(
        _write_durable_config(output_path, config)
    )
    logger.info(
        "Generated payload config for '%s': %s",
        payload_group.group_name,
        redact_sensitive_path(output_path),
    )
    return output_path


def generate_all_payload_configs(
    manifest: SceneManifest,
    scene_config: dict,
    configs_dir: Path,
    scene_config_dir: Path | None = None,
) -> SceneManifest:
    """Generate per-payload configs for all processable payload groups.

    Updates the manifest with config paths and working directories.

    Args:
        manifest: Scene manifest with detected payload groups.
        scene_config: The scene-level config dict.
        configs_dir: Directory to write generated configs.
        scene_config_dir: Directory of the original scene config (for rebasing
            relative paths).

    Returns:
        Updated SceneManifest.
    """
    payloads = manifest.get_processable_payloads()
    if not payloads:
        logger.info("No payload groups to generate configs for")
        return manifest

    logger.info(f"Generating configs for {len(payloads)} payload groups")

    # Build sibling map from the DAG: for each payload, find siblings
    # (other children of the same parent payload). This gives the VLM
    # context about neighboring components in the same system.
    sibling_map = _build_payload_sibling_map(manifest)

    # Use a subdirectory for payload configs to keep them separate
    payload_configs_dir = configs_dir / "payloads"

    for i, pg in enumerate(payloads, 1):
        config_path = payload_configs_dir / f"{pg.group_name}.yaml"

        logger.info(
            f"[{i}/{len(payloads)}] Generating config for payload '{pg.group_name}'"
        )
        try:
            safe_name = _sanitize_name(pg.group_name)
            generate_payload_config(
                pg,
                scene_config,
                config_path,
                scene_config_dir=scene_config_dir,
                sibling_names=sibling_map.get(pg.group_name),
            )
            pg.config_path = str(config_path)
            pg.working_dir = str(payload_configs_dir / f".{safe_name}")
        except Exception:
            logger.exception(f"Failed to generate config for payload '{pg.group_name}'")
            pg.status = "failed"

    return manifest


def _build_payload_sibling_map(manifest: SceneManifest) -> dict[str, list[str]]:
    """Return payload sibling names derived from the manifest DAG."""
    sibling_map: dict[str, list[str]] = {}
    all_pgs = manifest.payload_groups
    file_to_name: dict[str, str] = {}
    for pg in all_pgs:
        resolved = str(Path(pg.payload_file).resolve()) if pg.payload_file else ""
        if resolved:
            file_to_name[resolved] = pg.group_name
    for pg in all_pgs:
        if pg.child_payload_files:
            child_names = []
            for cf in pg.child_payload_files:
                resolved_cf = str(Path(cf).resolve())
                name = file_to_name.get(resolved_cf)
                if name:
                    child_names.append(name)
            for name in child_names:
                sibling_map[name] = child_names
    return sibling_map


def prepare_payload_runtime_config(
    payload_group: PayloadGroup,
    scene_config: dict[str, Any],
    scene_config_dir: Path | None = None,
    sibling_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build one in-memory payload config from a sanitized artifact + source."""
    if not payload_group.config_path:
        raise ValueError(
            f"Payload group '{payload_group.group_name}' has no config_path set"
        )
    config_path = Path(payload_group.config_path)
    durable, legacy_paths = _load_and_sanitize_generated_config(config_path)
    source_config = _build_payload_config(
        payload_group,
        scene_config,
        config_path,
        scene_config_dir=scene_config_dir,
        sibling_names=sibling_names,
    )
    available_paths = set(find_inline_secret_paths(source_config))
    required_paths = set(payload_group.config_credential_paths) | set(legacy_paths)
    missing_paths = required_paths - available_paths
    if missing_paths:
        raise _missing_credential_error(
            config_path=config_path, missing_paths=missing_paths
        )

    runtime = _overlay_inline_credentials(
        durable, source_config, redact_sensitive_config(source_config)
    )
    if not isinstance(runtime, dict):  # pragma: no cover - input contract guard
        raise TypeError("Runtime payload config must be a mapping")
    runtime_paths = set(find_inline_secret_paths(runtime))
    missing_runtime_paths = (required_paths | available_paths) - runtime_paths
    if missing_runtime_paths:
        raise _runtime_credential_overlay_error(
            config_path=config_path, missing_paths=missing_runtime_paths
        )

    payload_group.config_credential_paths = sorted(available_paths)
    return runtime


def prepare_payload_runtime_configs(
    manifest: SceneManifest,
    scene_config: dict[str, Any],
    scene_config_dir: Path | None = None,
    payloads: list[PayloadGroup] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build isolated runtime configs keyed by payload-group ID."""
    sibling_map = _build_payload_sibling_map(manifest)
    groups = payloads if payloads is not None else manifest.get_processable_payloads()
    return {
        payload_group.id: prepare_payload_runtime_config(
            payload_group,
            scene_config,
            scene_config_dir,
            sibling_names=sibling_map.get(payload_group.group_name),
        )
        for payload_group in groups
        if payload_group.config_path
    }


def _inject_payload_context(
    config: dict,
    payload_group: PayloadGroup,
    sibling_names: list[str] | None = None,
) -> None:
    """Append contextual information to the VLM system prompt.

    When a payload is processed in isolation, the VLM loses the spatial
    context it would have in the full scene. For simple objects (a tray,
    a carton, a bracket) this can lead to wrong material guesses.

    This function appends a context block to the VLM system prompt with:
    - The human-readable payload name (e.g., "Tray", "Conveyor_09")
    - The parent system derived from the file path (e.g., "DMS_Shuttle_System")
    - Sibling payload names if available (from DAG parent)

    Args:
        config: The per-payload config dict (modified in place).
        payload_group: The payload group being configured.
        sibling_names: Optional list of sibling payload names (other children
            of the same parent payload in the DAG).
    """
    # Derive human-readable name and parent context from the file path
    payload_path = Path(payload_group.payload_file)
    asset_name = payload_path.stem.replace("_", " ")

    # Walk up the directory tree to find meaningful parent context
    # e.g., .../Assets/Phase_01/DMS_Shuttle_System/Tray/Tray.usd
    #        → parent system: "DMS Shuttle System", category: "Phase 01"
    parent_parts = []
    for parent in payload_path.parents:
        name = parent.name
        if not name or name.lower() in ("assets", "subusd", "subusds", "collected"):
            break
        parent_parts.append(name.replace("_", " "))
    parent_parts.reverse()

    parent_context = ""
    if parent_parts:
        parent_context = " > ".join(parent_parts)

    context_block = (
        "\n\n"
        "IMPORTANT CONTEXT: This object is a component from an industrial/warehouse scene.\n"
    )
    if parent_context:
        context_block += f"Asset hierarchy: {parent_context}\n"
    if sibling_names:
        others = [
            s.replace("_", " ") for s in sibling_names if s != payload_group.group_name
        ]
        if others:
            context_block += f"Sibling components in same system: {', '.join(others)}\n"
    context_block += (
        f'Asset name: "{asset_name}"\n'
        "Use this context to inform your material choices — "
        "industrial/warehouse materials are expected "
        "(e.g., painted metal, powder-coated steel, rubber, plastic).\n"
    )

    # Append to the VLM system prompt
    steps = config.setdefault("steps", {})
    prepare = steps.setdefault("build_dataset_prepare_dataset", {})
    prompts = prepare.setdefault("prompts", {})

    existing_system = prompts.get("vlm_system", "")
    if existing_system:
        prompts["vlm_system"] = existing_system.rstrip() + context_block

    logger.debug(
        f"Injected VLM context for payload '{payload_group.group_name}': "
        f"name='{asset_name}', parent='{parent_context}', "
        f"siblings={len(sibling_names) if sibling_names else 0}"
    )


def _inject_split_context(config: dict, sub_asset: SubAsset) -> None:
    """Append split context to the VLM system prompt.

    When an asset was produced by splitting a larger container, the VLM
    loses the broader context of the parent structure. This injects a
    context block with parent name, ancestor chain, and sibling names
    so the VLM can make informed material choices.

    Args:
        config: The per-asset config dict (modified in place).
        sub_asset: The sub-asset with split_context.
    """
    ctx = sub_asset.split_context
    if not ctx:
        return

    parent_name = ctx.get("parent_name", "").replace("_", " ")
    siblings = ctx.get("sibling_names", [])
    ancestors = ctx.get("ancestors", [])

    # Build human-readable hierarchy
    hierarchy = " > ".join(a.replace("_", " ") for a in ancestors)

    sibling_list = ", ".join(
        s.replace("_", " ") for s in siblings if s != sub_asset.name
    )

    context_block = (
        "\n\nIMPORTANT CONTEXT: This object was extracted from a larger structure.\n"
    )
    if hierarchy:
        context_block += f"Parent hierarchy: {hierarchy}\n"
    if sibling_list:
        context_block += f"Sibling components: {sibling_list}\n"
    context_block += (
        f'This component: "{sub_asset.name.replace("_", " ")}"\n'
        "Use this context to inform your material choices — "
        "materials should be consistent with the parent structure "
        "and neighboring components.\n"
    )

    # Append to the VLM system prompt
    steps = config.setdefault("steps", {})
    prepare = steps.setdefault("build_dataset_prepare_dataset", {})
    prompts = prepare.setdefault("prompts", {})

    existing_system = prompts.get("vlm_system", "")
    if existing_system:
        prompts["vlm_system"] = existing_system.rstrip() + context_block

    logger.debug(
        f"Injected split context for '{sub_asset.name}': "
        f"parent='{parent_name}', siblings={len(siblings)}"
    )


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as directory/file names and session IDs."""
    safe = re.sub(r"[^\w\-]", "_", name)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe.lower() if safe else "unnamed"


def _unique_safe_names(assets: list) -> dict[str, str]:
    """Build a mapping of asset ID → unique safe name.

    When multiple assets share the same sanitized name, appends the asset ID
    as a suffix (e.g. ``default_obj_230``) to disambiguate.
    """
    from collections import Counter

    name_counts = Counter(_sanitize_name(sa.name) for sa in assets)
    result: dict[str, str] = {}
    for sa in assets:
        safe = _sanitize_name(sa.name)
        if name_counts[safe] > 1:
            suffix = _sanitize_name(sa.id)
            safe = f"{safe}_{suffix}"
        result[sa.id] = safe
    return result
