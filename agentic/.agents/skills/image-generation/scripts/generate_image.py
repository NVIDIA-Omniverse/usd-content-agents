#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate or record one image with a provider-neutral result manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

SCHEMA_VERSION = "agentic-image-generation-result.v1"
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    contents = path.read_text(encoding="utf-8")
    if not contents.strip():
        raise ValueError("Prompt file must contain non-whitespace text")
    return contents


def _conditioning_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Conditioning image does not exist: {resolved}")
        with Image.open(resolved) as image:
            width, height = image.size
            image_mode = image.mode
        records.append(
            {
                "path": str(resolved),
                "sha256": _sha256_file(resolved),
                "width": width,
                "height": height,
                "image_mode": image_mode,
            }
        )
    return records


def _provider_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "tool_id": args.tool_id,
        "backend": getattr(args, "backend", None),
        "model": getattr(args, "model", None),
        "base_url": getattr(args, "base_url", None),
        "api_key_env": getattr(args, "api_key_env", None),
    }


def _safe_provider_record(args: argparse.Namespace) -> dict[str, Any]:
    """Return failure metadata without persisting credentials from an invalid URL."""

    record = _provider_record(args)
    try:
        _validate_base_url(record["base_url"])
    except ValueError:
        record["base_url"] = None
    api_key_env = record["api_key_env"]
    if api_key_env is not None and not _ENV_NAME.fullmatch(api_key_env):
        record["api_key_env"] = None
    return record


def _base_manifest(
    args: argparse.Namespace,
    *,
    mode: str,
    status: str,
) -> dict[str, Any]:
    prompt_path = args.prompt_file.resolve()
    _read_prompt(prompt_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "mode": mode,
        "request": {
            "prompt_file": str(prompt_path),
            "prompt_sha256": _sha256_file(prompt_path),
            "conditioning_images": _conditioning_records(list(args.conditioning_image)),
        },
        "provider": _provider_record(args),
        "output": None,
    }


def _image_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Generated image does not exist: {resolved}")
    with Image.open(resolved) as image:
        image.load()
        width, height = image.size
        image_mode = image.mode
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "width": width,
        "height": height,
        "image_mode": image_mode,
        "media_type": "image/png",
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _validate_base_url(base_url: str | None) -> None:
    if not base_url:
        return
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not contain credentials")
    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain params, query, or fragment")


def _explicit_api_key(api_key_env: str | None) -> str | None:
    if not api_key_env:
        return None
    if not _ENV_NAME.fullmatch(api_key_env):
        raise ValueError("api_key_env must be an environment-variable name")
    value = os.getenv(api_key_env)
    if not value:
        raise ValueError(
            f"Image-generation credential is missing from environment: {api_key_env}"
        )
    return value


def _create_world_understanding_model(args: argparse.Namespace) -> Any:
    _validate_base_url(args.base_url)
    kwargs: dict[str, Any] = {}
    if args.model:
        kwargs["model"] = args.model
    if args.base_url:
        kwargs["base_url"] = args.base_url
    if args.timeout_seconds is not None:
        kwargs["timeout"] = args.timeout_seconds
    api_key = _explicit_api_key(args.api_key_env)
    if api_key:
        kwargs["api_key"] = api_key

    if args.backend == "openai_compatible":
        from world_understanding.functions.models.image_generation_models import (
            OpenAICompatibleChatImageGenerationModel,
        )

        return OpenAICompatibleChatImageGenerationModel(
            backend_name=args.backend,
            **kwargs,
        )

    from world_understanding.functions.models.image_generation_models import (
        create_image_generation_model,
    )

    return create_image_generation_model(args.backend, **kwargs)


def _save_generated_image(image: Any, output_path: Path) -> None:
    resolved = output_path.resolve()
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite generated image: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(image, Image.Image):
        raise TypeError("World Understanding image backend did not return a PIL image")
    image.convert("RGB").save(resolved, format="PNG")


def _record_failure(
    args: argparse.Namespace,
    *,
    mode: str,
    error: Exception | None = None,
    reason: str | None = None,
) -> int:
    message = str(error) if error is not None else str(reason)
    api_key_env = getattr(args, "api_key_env", None)
    if api_key_env is not None and _ENV_NAME.fullmatch(api_key_env):
        api_key = os.getenv(api_key_env)
        if api_key:
            message = message.replace(api_key, "[REDACTED]")
    try:
        from world_understanding.utils.credentials import redact_sensitive_config

        message = str(redact_sensitive_config(message))
    except Exception as redaction_error:
        message = (
            "Image generation failed; provider detail could not be recorded "
            "safely because credential redaction was unavailable "
            f"({type(redaction_error).__name__})."
        )
    try:
        manifest = _base_manifest(args, mode=mode, status="failed")
    except Exception as manifest_error:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "mode": mode,
            "request": {
                "prompt_file": str(args.prompt_file.resolve()),
                "prompt_sha256": None,
                "conditioning_images": None,
                "request_inspection_error": type(manifest_error).__name__,
            },
            "provider": _safe_provider_record(args),
            "output": None,
        }
    manifest["provider"] = _safe_provider_record(args)
    manifest["diagnostic"] = {
        "type": type(error).__name__ if error is not None else "ImageGenerationFailure",
        "message": message,
    }
    _write_manifest(args.manifest, manifest)
    return 1


def _run_backend(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest_path}")
    try:
        if args.output.resolve().exists():
            raise FileExistsError(
                f"Refusing to overwrite generated image: {args.output.resolve()}"
            )
        prompt = _read_prompt(args.prompt_file.resolve())
        model = _create_world_understanding_model(args)
        if (
            args.conditioning_image
            and getattr(model, "supports_image_conditioning", None) is not True
        ):
            raise ValueError(
                f"Image-generation backend {args.backend!r} does not support "
                "conditioning images"
            )
        generated = model.generate(
            prompt,
            images=[path.resolve() for path in args.conditioning_image] or None,
        )
        _save_generated_image(generated, args.output)
        manifest = _base_manifest(
            args,
            mode="world_understanding_backend",
            status="completed",
        )
        manifest["provider"]["backend"] = model.backend_name
        manifest["provider"]["model"] = model.model_name
        manifest["output"] = _image_record(args.output)
        _write_manifest(args.manifest, manifest)
        return 0
    except Exception as exc:
        if manifest_path.exists():
            return 1
        return _record_failure(
            args,
            mode="world_understanding_backend",
            error=exc,
        )


def _record_companion(args: argparse.Namespace) -> int:
    try:
        if args.manifest.resolve().exists():
            raise FileExistsError(
                f"Refusing to overwrite manifest: {args.manifest.resolve()}"
            )
        source = args.source_image.resolve()
        output = args.output.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Companion output does not exist: {source}")
        if source != output:
            if output.exists():
                raise FileExistsError(
                    f"Refusing to overwrite generated image: {output}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                image.convert("RGB").save(output, format="PNG")
        else:
            with Image.open(output) as image:
                image.load()

        manifest = _base_manifest(
            args,
            mode="coding_agent_companion",
            status="completed",
        )
        manifest["output"] = _image_record(output)
        _write_manifest(args.manifest, manifest)
        return 0
    except Exception as exc:
        if args.manifest.resolve().exists():
            raise
        return _record_failure(
            args,
            mode="coding_agent_companion",
            error=exc,
        )


def _record_requested_failure(args: argparse.Namespace) -> int:
    return _record_failure(args, mode=args.mode, reason=args.reason)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--conditioning-image", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tool-id", default="world-understanding")
    parser.add_argument("--model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or record one provider-neutral image artifact."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    backend = subparsers.add_parser(
        "backend", help="Generate with an explicit World Understanding backend."
    )
    _add_common_arguments(backend)
    backend.add_argument("--output", type=Path, required=True)
    backend.add_argument("--backend", required=True)
    backend.add_argument("--base-url")
    backend.add_argument("--api-key-env")
    backend.add_argument("--timeout-seconds", type=float)
    backend.set_defaults(handler=_run_backend)

    companion = subparsers.add_parser(
        "record-companion", help="Record an image produced by the companion tool."
    )
    _add_common_arguments(companion)
    companion.set_defaults(tool_id="companion-image-generation")
    companion.add_argument("--source-image", type=Path, required=True)
    companion.add_argument("--output", type=Path, required=True)
    companion.set_defaults(handler=_record_companion)

    failure = subparsers.add_parser(
        "record-failure", help="Record an image-generation failure."
    )
    _add_common_arguments(failure)
    failure.add_argument(
        "--mode",
        choices=("coding_agent_companion", "world_understanding_backend"),
        required=True,
    )
    failure.add_argument("--backend")
    failure.add_argument("--base-url")
    failure.add_argument("--api-key-env")
    failure.add_argument("--reason", required=True)
    failure.set_defaults(handler=_record_requested_failure)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
