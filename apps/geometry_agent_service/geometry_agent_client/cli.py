# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Geometry Agent service and reference-connector CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, cast

from .http import GeometryAgentClient


def _add_service_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--service-url",
        default=os.environ.get("GEOMETRY_AGENT_SERVICE_URL", "http://127.0.0.1:8776"),
    )


def _client(args: argparse.Namespace) -> GeometryAgentClient:
    api_key = os.environ.get("GEOMETRY_AGENT_SERVICE_API_KEY", "")
    return GeometryAgentClient(base_url=args.service_url, api_key=api_key)


ParameterCliValue = bool | int | float | str | dict[str, Any]


def _parameter(raw: str) -> tuple[str, ParameterCliValue]:
    name, separator, encoded = raw.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("parameters must use NAME=JSON_SCALAR")
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("parameter value must be valid JSON") from exc
    if value is None or isinstance(value, list):
        raise argparse.ArgumentTypeError(
            "parameter value must be a JSON scalar or value/unit object"
        )
    if isinstance(value, dict):
        if set(value) not in ({"value"}, {"value", "unit"}):
            raise argparse.ArgumentTypeError(
                "parameter objects may contain only value and optional unit"
            )
        scalar = value.get("value")
        unit = value.get("unit")
        if scalar is None or isinstance(scalar, list | dict):
            raise argparse.ArgumentTypeError(
                "parameter object value must be a JSON scalar"
            )
        if unit is not None and (
            not isinstance(unit, str) or not unit or len(unit) > 64
        ):
            raise argparse.ArgumentTypeError(
                "parameter object unit must be a bounded string"
            )
    return name, value


def _prompt(args: argparse.Namespace) -> str | None:
    prompt = cast(str | None, args.prompt)
    prompt_file = cast(Path | None, args.prompt_file)
    if prompt is not None:
        if len(prompt) > 32_768:
            raise ValueError("Prompt exceeds the 32,768-character service limit")
        return prompt
    if prompt_file is not None:
        raw_prompt_path = prompt_file.expanduser()
        if raw_prompt_path.is_symlink():
            raise ValueError("Prompt path must be a regular file")
        prompt_path = raw_prompt_path.resolve(strict=True)
        if not prompt_path.is_file():
            raise ValueError("Prompt path must be a regular file")
        if prompt_path.stat().st_size > 128 * 1024:
            raise ValueError("Prompt file exceeds the 128 KiB input limit")
        value = prompt_path.read_text(encoding="utf-8")
        if not value.strip():
            raise ValueError("Prompt file is empty")
        if len(value) > 32_768:
            raise ValueError("Prompt exceeds the 32,768-character service limit")
        return value
    return None


def _upload_images(
    client: GeometryAgentClient,
    paths: list[Path],
) -> list[str]:
    source_ids: list[str] = []
    for path in paths:
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        source = client.upload(path, role="reference_image", media_type=media_type)
        source_id = source.get("source_id")
        if not isinstance(source_id, str):
            raise RuntimeError("Image upload omitted its source identity")
        source_ids.append(source_id)
    return source_ids


def _terminal(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("status") == "failed":
        error = job.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        raise RuntimeError(message or "Geometry Agent job failed")
    if job.get("status") != "succeeded":
        raise RuntimeError("Geometry Agent returned a non-terminal job state")
    return job


def _parameters(values: list[tuple[str, ParameterCliValue]]) -> dict[str, Any]:
    names = [name for name, _value in values]
    if len(names) != len(set(names)):
        raise ValueError("Each semantic parameter may be specified only once")
    return dict(values)


def _variant_document(path: Path) -> list[dict[str, Any]]:
    raw_path = path.expanduser()
    if raw_path.is_symlink():
        raise ValueError("Variant document must be a regular file")
    resolved = raw_path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size > 256 * 1024:
        raise ValueError(
            "Variant document must be a regular file no larger than 256 KiB"
        )
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Variant document must contain valid UTF-8 JSON") from exc
    variants = document.get("variants") if isinstance(document, dict) else None
    if not isinstance(variants, list) or not variants or len(variants) > 64:
        raise ValueError(
            "Variant document requires a variants array with 1 to 64 entries"
        )
    if any(not isinstance(item, dict) for item in variants):
        raise ValueError("Every variant must be a JSON object")
    return variants


def _emit(payload: dict[str, Any], output: Path | None = None) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(encoded, end="")
        return
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(destination)


def _providers(args: argparse.Namespace) -> int:
    _emit(_client(args).providers(), args.output)
    return 0


def _generate(args: argparse.Namespace) -> int:
    client = _client(args)
    prompt = _prompt(args)
    image_source_ids = _upload_images(client, args.image)
    parameters = _parameters(args.parameter)
    generation_job = client.wait(
        client.generate(
            {
                "provider_id": args.provider,
                "prompt": prompt,
                "image_source_ids": image_source_ids,
                "parameters": parameters,
                "requested_formats": args.format or ["step"],
                "target_profile": args.target_profile,
            }
        )
    )
    _terminal(generation_job)
    payload: dict[str, Any] = generation_job
    if args.run:
        result = generation_job.get("result")
        source = result.get("source") if isinstance(result, dict) else None
        source_id = source.get("source_id") if isinstance(source, dict) else None
        if not isinstance(source_id, str):
            raise RuntimeError("Generation job omitted its source identity")
        run_job = client.wait(
            client.run(
                {
                    "source_id": source_id,
                    "target_profile": args.target_profile,
                    "target_runtime": args.target_runtime,
                    "runtime_engine": args.runtime_engine,
                    "run_runtime_validation": args.runtime_engine != "none",
                    "render_evidence": args.render_evidence,
                }
            )
        )
        payload = {"generation": generation_job, "geometry_run": run_job}
        _emit(payload, args.output)
        _terminal(run_job)
        return 0
    _emit(payload, args.output)
    return 0


def _revise(args: argparse.Namespace) -> int:
    client = _client(args)
    instructions = _prompt(args)
    image_source_ids = _upload_images(client, args.image)
    parameter_overrides = _parameters(args.parameter)
    job = client.wait(
        client.revise(
            {
                "provider_id": args.provider,
                "source_id": args.source_id,
                "instructions": instructions,
                "image_source_ids": image_source_ids,
                "parameter_overrides": parameter_overrides,
                "requested_formats": args.format or ["step"],
            }
        )
    )
    _emit(_terminal(job), args.output)
    return 0


def _family(args: argparse.Namespace) -> int:
    variants = _variant_document(args.variants)
    client = _client(args)
    job = client.wait(
        client.family(
            {
                "provider_id": args.provider,
                "source_id": args.source_id,
                "variants": variants,
                "requested_formats": args.format or ["step"],
            }
        )
    )
    _emit(job, args.output)
    _terminal(job)
    return 0


def _export(args: argparse.Namespace) -> int:
    client = _client(args)
    job = client.wait(
        client.export(
            {
                "provider_id": args.provider,
                "source_id": args.source_id,
                "requested_formats": args.format,
            }
        )
    )
    _emit(_terminal(job), args.output)
    return 0


def _provider_export(args: argparse.Namespace) -> int:
    client = _client(args)
    job = client.wait(
        client.provider_export(
            {
                "provider_id": args.provider,
                "source_revision": args.source_revision,
                "coordinate_system": {
                    "meters_per_unit": args.meters_per_unit,
                    "up_axis": args.up_axis,
                    "forward_axis": args.forward_axis,
                    "handedness": args.handedness,
                },
                "rights_assertion": args.rights_assertion,
                "license_identifier": args.license_identifier,
                "upstream_edit_uri": args.upstream_edit_uri,
                "requested_formats": args.format,
            }
        )
    )
    _emit(_terminal(job), args.output)
    return 0


def _upload(args: argparse.Namespace) -> int:
    media_type = mimetypes.guess_type(args.source.name)[0] or "application/octet-stream"
    _emit(
        _client(args).upload(
            args.source,
            role=args.role,
            media_type=media_type,
            archive_entrypoint=args.archive_entrypoint,
        ),
        args.output,
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    client = _client(args)
    job = client.wait(
        client.run(
            {
                "source_id": args.source_id,
                "target_profile": args.target_profile,
                "target_runtime": args.target_runtime,
                "runtime_engine": args.runtime_engine,
                "run_runtime_validation": args.runtime_engine != "none",
                "render_evidence": args.render_evidence,
                "render_preset": args.render_preset,
            }
        )
    )
    _emit(_terminal(job), args.output)
    return 0


def _download(args: argparse.Namespace) -> int:
    _client(args).download(
        args.artifact_id,
        args.output,
        expected_sha256=args.expected_sha256,
    )
    print(args.output.expanduser().resolve())
    return 0


def _portable_manifest(bundle: Any, output_dir: Path) -> Path:
    from geometry_authoring_contracts import (
        GeometryArtifactBinding,
        GeometrySourceProvenance,
    )

    output_root = output_dir.resolve(strict=True)
    representations = []
    for item in bundle.representations:
        artifact_path = Path(item.artifact.path).resolve(strict=True)
        if not artifact_path.is_relative_to(output_root):
            raise ValueError("Bundle artifacts must remain inside the output directory")
        relative_path = artifact_path.relative_to(output_root).as_posix()
        representations.append(
            item.model_copy(
                update={
                    "artifact": GeometryArtifactBinding(
                        path=relative_path,
                        sha256=item.artifact.sha256,
                        size_bytes=item.artifact.size_bytes,
                    )
                }
            )
        )
    provenance = GeometrySourceProvenance.model_validate(
        bundle.provenance.model_copy(
            update={
                "input_artifacts": tuple(
                    item.model_copy(update={"path": f"sha256:{item.sha256}"})
                    for item in bundle.provenance.input_artifacts
                )
            }
        )
    )
    portable = bundle.model_copy(
        update={"representations": tuple(representations), "provenance": provenance}
    )
    manifest = output_dir / "geometry.source.json"
    with manifest.open("x", encoding="utf-8") as stream:
        stream.write(portable.model_dump_json(indent=2))
        stream.write("\n")
    return manifest


def _new_output_directory(path: Path) -> Path:
    output_dir = path.expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    return output_dir


def _forgecad_import(args: argparse.Namespace) -> int:
    from geometry_authoring_connectors import ForgeCadArtifactAdapter

    output_dir = _new_output_directory(args.output_dir)
    bundle = ForgeCadArtifactAdapter().import_source_bundle(
        exports=tuple(args.export),
        output_dir=output_dir,
        rights_assertion=args.rights_assertion,
        forge_source=args.forge_source,
        manifest_path=args.manifest,
        units=args.units,
        up_axis=args.up_axis,
    )
    manifest = _portable_manifest(bundle, output_dir)
    _emit(
        {
            "schema_version": bundle.schema_version,
            "bundle_id": bundle.bundle_id,
            "manifest_path": str(manifest),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }
    )
    return 0


def _onshape_api_credentials() -> tuple[str, str]:
    access_key = os.environ.get("ONSHAPE_API_KEY")
    primary_secret = os.environ.get("ONSHAPE_API_SECRET")
    compatible_secret = os.environ.get("ONSHAPE_SECRET")
    if primary_secret and compatible_secret and primary_secret != compatible_secret:
        raise RuntimeError("Onshape API secret environment variables disagree")
    secret_key = primary_secret or compatible_secret
    if not access_key or not secret_key:
        raise RuntimeError(
            "Configure ONSHAPE_API_KEY and ONSHAPE_API_SECRET in the local "
            "execution environment; do not pass credentials in chat or command arguments"
        )
    return access_key, secret_key


def _onshape_export(args: argparse.Namespace) -> int:
    from geometry_authoring_connectors import (
        OnshapeConnector,
        OnshapeVersionExportRequest,
        OnshapeWorkspaceSnapshotRequest,
        canonicalize_materialized_bundle,
    )

    rights_assertion = args.rights_assertion.strip()
    if not rights_assertion:
        raise ValueError("--rights-assertion must not be empty")
    if args.version_id is not None and args.snapshot_name is not None:
        raise ValueError("--snapshot-name is valid only with --workspace-id")
    if args.workspace_id is not None and args.snapshot_name is None:
        raise ValueError("--workspace-id requires --snapshot-name")
    snapshot_request = None
    if args.version_id is None:
        snapshot_request = OnshapeWorkspaceSnapshotRequest(
            document_id=args.document_id,
            workspace_id=args.workspace_id,
            version_name=args.snapshot_name,
        )
    request = OnshapeVersionExportRequest(
        document_id=args.document_id,
        version_id=args.version_id or ("0" * 24),
        element_id=args.element_id,
        element_kind=args.element_kind,
        format=args.format,
        output_filename=f"geometry.{args.format}",
    )
    output_dir = _new_output_directory(args.output_dir)
    access_key, secret_key = _onshape_api_credentials()
    connector = OnshapeConnector(
        api_access_key=access_key,
        api_secret_key=secret_key,
    )
    version_id = args.version_id
    if snapshot_request is not None:
        version_id = connector.create_immutable_version(snapshot_request)
        request = request.model_copy(update={"version_id": version_id})
    request_digest = hashlib.sha256(
        json.dumps(
            request.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    materialized = connector.export(request, output_dir=output_dir)
    bundle = canonicalize_materialized_bundle(
        materialized,
        request_digest=request_digest,
        rights_assertion=rights_assertion,
    )
    manifest = _portable_manifest(bundle, output_dir)
    _emit(
        {
            "schema_version": bundle.schema_version,
            "bundle_id": bundle.bundle_id,
            "source_revision": bundle.source_revision,
            "onshape_version_id": version_id,
            "snapshot_created": args.workspace_id is not None,
            "manifest_path": str(manifest),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
        args.output,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use the public Geometry Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    providers = subparsers.add_parser("providers")
    _add_service_args(providers)
    providers.add_argument("--output", type=Path)
    providers.set_defaults(handler=_providers)

    generate = subparsers.add_parser("generate")
    _add_service_args(generate)
    generate.add_argument("--provider", required=True)
    prompt = generate.add_mutually_exclusive_group()
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    generate.add_argument("--image", type=Path, action="append", default=[])
    generate.add_argument("--parameter", type=_parameter, action="append", default=[])
    generate.add_argument("--format", action="append")
    generate.add_argument(
        "--target-profile",
        default="geometry-agent.insertion-or-fixture-asset.v1",
    )
    generate.add_argument("--run", action="store_true")
    generate.add_argument("--target-runtime", default="isaac-lab")
    generate.add_argument(
        "--runtime-engine", choices=["ovphysx", "fake", "none"], default="none"
    )
    generate.add_argument("--render-evidence", action="store_true")
    generate.add_argument("--output", type=Path)
    generate.set_defaults(handler=_generate)

    revise = subparsers.add_parser("revise")
    _add_service_args(revise)
    revise.add_argument("source_id")
    revise.add_argument("--provider", required=True)
    instructions = revise.add_mutually_exclusive_group()
    instructions.add_argument("--prompt", dest="prompt")
    instructions.add_argument("--prompt-file", type=Path)
    revise.add_argument("--image", type=Path, action="append", default=[])
    revise.add_argument("--parameter", type=_parameter, action="append", default=[])
    revise.add_argument("--format", action="append")
    revise.add_argument("--output", type=Path)
    revise.set_defaults(handler=_revise)

    family = subparsers.add_parser("family")
    _add_service_args(family)
    family.add_argument("source_id")
    family.add_argument("--provider", required=True)
    family.add_argument("--variants", type=Path, required=True)
    family.add_argument("--format", action="append")
    family.add_argument("--output", type=Path)
    family.set_defaults(handler=_family)

    export = subparsers.add_parser("export")
    _add_service_args(export)
    export.add_argument("source_id")
    export.add_argument("--provider", required=True)
    export.add_argument("--format", action="append", required=True)
    export.add_argument("--output", type=Path)
    export.set_defaults(handler=_export)

    provider_export = subparsers.add_parser("export-provider-revision")
    _add_service_args(provider_export)
    provider_export.add_argument("--provider", required=True)
    provider_export.add_argument("--source-revision", required=True)
    provider_export.add_argument("--meters-per-unit", type=float, required=True)
    provider_export.add_argument("--up-axis", choices=["X", "Y", "Z"], required=True)
    provider_export.add_argument(
        "--forward-axis",
        choices=["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
        required=True,
    )
    provider_export.add_argument(
        "--handedness", choices=["left", "right"], default="right"
    )
    provider_export.add_argument("--rights-assertion", required=True)
    provider_export.add_argument("--license-identifier")
    provider_export.add_argument("--upstream-edit-uri")
    provider_export.add_argument("--format", action="append", required=True)
    provider_export.add_argument("--output", type=Path)
    provider_export.set_defaults(handler=_provider_export)

    upload = subparsers.add_parser("upload")
    _add_service_args(upload)
    upload.add_argument("source", type=Path)
    upload.add_argument(
        "--role",
        choices=["geometry_source", "reference_image"],
        default="geometry_source",
    )
    upload.add_argument("--archive-entrypoint")
    upload.add_argument("--output", type=Path)
    upload.set_defaults(handler=_upload)

    run = subparsers.add_parser("run")
    _add_service_args(run)
    run.add_argument("source_id")
    run.add_argument(
        "--target-profile", default="geometry-agent.insertion-or-fixture-asset.v1"
    )
    run.add_argument("--output", type=Path)
    run.add_argument("--target-runtime", default="isaac-lab")
    run.add_argument(
        "--runtime-engine", choices=["ovphysx", "fake", "none"], default="none"
    )
    run.add_argument("--render-evidence", action="store_true")
    run.add_argument(
        "--render-preset",
        choices=["hero", "4view", "six_view", "vertical4", "turntable"],
        default="six_view",
    )
    run.set_defaults(handler=_run)

    download = subparsers.add_parser("download")
    _add_service_args(download)
    download.add_argument("artifact_id")
    download.add_argument("output", type=Path)
    download.add_argument("--expected-sha256")
    download.set_defaults(handler=_download)

    forgecad = subparsers.add_parser("import-forgecad")
    forgecad.add_argument("--export", type=Path, action="append", required=True)
    forgecad.add_argument("--forge-source", type=Path)
    forgecad.add_argument("--manifest", type=Path)
    forgecad.add_argument("--rights-assertion")
    forgecad.add_argument(
        "--units",
        choices=["millimeter", "centimeter", "meter", "inch"],
        default="millimeter",
    )
    forgecad.add_argument("--up-axis", choices=["X", "Y", "Z"], default="Z")
    forgecad.add_argument("--output-dir", type=Path, required=True)
    forgecad.set_defaults(handler=_forgecad_import)

    onshape = subparsers.add_parser("export-onshape")
    onshape.add_argument("--document-id", required=True)
    source_revision = onshape.add_mutually_exclusive_group(required=True)
    source_revision.add_argument("--version-id")
    source_revision.add_argument("--workspace-id")
    onshape.add_argument("--snapshot-name")
    onshape.add_argument("--element-id", required=True)
    onshape.add_argument(
        "--element-kind",
        choices=["partstudio", "assembly"],
        required=True,
    )
    onshape.add_argument(
        "--format",
        choices=["step", "gltf", "obj"],
        default="step",
    )
    onshape.add_argument("--rights-assertion", required=True)
    onshape.add_argument("--output-dir", type=Path, required=True)
    onshape.add_argument("--output", type=Path)
    onshape.set_defaults(handler=_onshape_export)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        status = args.handler(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
