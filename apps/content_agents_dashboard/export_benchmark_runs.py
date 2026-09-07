# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export self-contained benchmark runs for dashboard hosting.

The live benchmark tree contains large intermediate workflow artifacts. This
exporter copies only the standardized run documents and the asset files linked
from ``bundle/run.json`` so a dashboard can be hosted without exposing the raw
agent workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _publish_modes(root: Path) -> None:
    """Make the staged tree readable by whoever serves it.

    Chmoding the staging root alone was not enough. `shutil.copy2` reproduces
    the source mode, and the artifacts this exports are written owner-only:
    the mesh-segmentation runner's `_write_json` defaults to `0o600`, and
    `content_benchmark.util.write_json` materialises through a
    `NamedTemporaryFile` (also `0o600`) before renaming into place. So
    `benchmark_run.json` -- the manifest the dashboard fetches first, whose
    failure drops the entire run -- along with `suite_result.json` and every
    copied `terminal_validation.json` arrived at 0o600 and 403'd for a
    different service account.

    Normalise the whole tree to what a normal umask would have produced.
    """

    root.chmod(0o755)
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        path.chmod(0o755 if path.is_dir() else 0o644)


def _safe_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"unsafe dashboard artifact segment: {value!r}")
    return cleaned


# Every field the renderer writes as an absolute path. `image` and the
# per-channel entries were omitted, so an export still linked into the
# producing machine's run directory -- the exact dangling provenance this
# closure exists to remove.
_RENDER_DEPENDENCY_FIELDS = ("image", "camera", "source_camera", "response")


def _carries_credentialed_endpoint(value: object) -> bool:
    """True when any nested endpoint carries URL userinfo.

    The verbatim exact record ships in a digest-bound sidecar; redaction
    would break the binding, so a credentialed record must not ship at all.
    """
    if isinstance(value, dict):
        endpoint = value.get("endpoint")
        if isinstance(endpoint, str):
            parsed = urlparse(endpoint)
            # Userinfo, query, and fragment can all carry credentials
            # (signed URLs, ?token=...); the render path rejects such base
            # URLs at configuration time and the writers must too.
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                return True
        return any(_carries_credentialed_endpoint(item) for item in value.values())
    if isinstance(value, list):
        return any(_carries_credentialed_endpoint(item) for item in value)
    return False


def _shareable_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Project a provenance record to its shareable evidence fields.

    The verbatim record carries producing-machine absolute paths: the render
    response's output files, and render_metadata's per-stage
    ``stage_preparation[].usd_path`` / ``asset_base_dir``. Keep the renderer
    identity, both digests, and the path-free OVRTX metadata, and bind the
    projection to the exact record with the SHA-256 of its canonical JSON so
    the verbatim metadata preserved in the source run stays integrity-bound
    and recoverable (the report renderer applies the same projection).
    """

    result = {
        key: provenance[key]
        for key in ("renderer", "source_usd_sha256", "image_sha256")
        if key in provenance
    }
    # A remote record's attested OVRTX identity lives only in the render
    # response, which the projection drops wholesale — carry the identity
    # fields (endpoint/engine/protocol_version/status hold no
    # producing-machine paths) so the dashboard's remote attestation gate
    # still verifies exported remote references.
    identity = provenance.get("renderer_identity")
    if not isinstance(identity, dict):
        response = provenance.get("render_response")
        data = response.get("data") if isinstance(response, dict) else None
        results = data.get("results") if isinstance(data, dict) else None
        if (
            isinstance(results, list)
            and len(results) == 1
            and isinstance(results[0], dict)
        ):
            identity = results[0].get("renderer_identity")
    if isinstance(identity, dict):
        endpoint = identity.get("endpoint")
        parsed = urlparse(endpoint) if isinstance(endpoint, str) else None
        # A userinfo-bearing endpoint would publish credentials in the hosted
        # bundle and the exact sidecar; drop the identity (fail closed — the
        # dashboard then demotes the remote reference) rather than carry it.
        if (
            parsed is not None
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
        ):
            result["renderer_identity"] = {
                key: identity[key]
                for key in ("endpoint", "engine", "protocol_version", "status")
                if key in identity
            }
    metadata = provenance.get("render_metadata")
    if isinstance(metadata, dict):
        sanitized = {
            key: value for key, value in metadata.items() if key != "asset_base_dir"
        }
        identity = sanitized.get("renderer_identity")
        if isinstance(identity, dict):
            endpoint = identity.get("endpoint")
            parsed = urlparse(endpoint) if isinstance(endpoint, str) else None
            # A userinfo-bearing endpoint would publish credentials with the
            # shared projection; drop the nested identity rather than carry it.
            if parsed is None or parsed.username or parsed.password:
                sanitized.pop("renderer_identity", None)
        preparation = sanitized.get("stage_preparation")
        if isinstance(preparation, list):
            sanitized["stage_preparation"] = [
                {
                    key: value
                    for key, value in entry.items()
                    if key not in {"usd_path", "asset_base_dir"}
                }
                if isinstance(entry, dict)
                else entry
                for entry in preparation
            ]
        result["render_metadata"] = sanitized
    result["exact_record_sha256"] = hashlib.sha256(
        json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    return result


def _sanitize_suite_result_provenance(
    path: Path, *, source_bundle_dir: Path | None = None
) -> None:
    """Project the reference provenance the scorer stored verbatim.

    The physics scorer carries the unprojected record — including the render
    response's absolute image/USD paths — into ``suite_result.json``, so a
    byte-for-byte copy would publish exactly the producing-machine paths this
    exporter exists to remove.
    """

    if not path.is_file():
        return
    try:
        document = _read_json(path)
    except (OSError, ValueError):
        # Fail closed: the verbatim copy is already staged, and returning
        # here would publish exactly the producing-machine paths this
        # function exists to remove. An unsanitizable document must not ship.
        path.unlink(missing_ok=True)
        return
    changed = False
    for case in document.get("cases") or []:
        if not isinstance(case, dict):
            continue
        metrics = case.get("metrics")
        if isinstance(metrics, dict) and isinstance(
            metrics.get("reference_provenance"), dict
        ):
            record = metrics["reference_provenance"]
            # Always project — the binding marker is data, not proof: a
            # rehydrated or tampered record can carry exact_record_sha256
            # alongside the producing-machine fields the projection exists
            # to strip. But preserve an existing binding rather than
            # recomputing it, or a re-export would replace it with the hash
            # of the projection itself and break the digest binding to the
            # retained verbatim sidecar.
            prior_sha = record.get("exact_record_sha256")
            prior_path = record.get("exact_record_path")
            projected = _shareable_provenance(
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"exact_record_sha256", "exact_record_path"}
                }
            )
            # A prior binding is data, not proof: preserve it only after
            # verifying that the sidecar it names actually digests to it in
            # the source bundle. A foreign or tampered marker is dropped —
            # the record still ships projected, just unbound.
            verified_prior = False
            if (
                isinstance(prior_sha, str)
                and prior_sha
                and isinstance(prior_path, str)
                and prior_path
                and source_bundle_dir is not None
            ):
                sidecar = source_bundle_dir / prior_path
                try:
                    contained = not Path(
                        prior_path
                    ).is_absolute() and sidecar.resolve().is_relative_to(
                        source_bundle_dir.resolve()
                    )
                except OSError:
                    contained = False
                try:
                    loaded = (
                        json.loads(sidecar.read_text(encoding="utf-8"))
                        if contained
                        else None
                    )
                except (OSError, ValueError):
                    loaded = None
                if isinstance(loaded, dict):
                    canonical = json.dumps(
                        loaded,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("utf-8")
                    verified_prior = hashlib.sha256(canonical).hexdigest() == prior_sha
            if verified_prior:
                projected["exact_record_sha256"] = prior_sha
                projected["exact_record_path"] = prior_path
            elif prior_sha is not None or prior_path is not None:
                # An unverifiable binding must not be re-signed; the fresh
                # exact_record_sha256 computed by _shareable_provenance over
                # the projection would be a false binding too, so remove it.
                projected.pop("exact_record_sha256", None)
                projected.pop("exact_record_path", None)
            metrics["reference_provenance"] = projected
            changed = True
    if changed:
        _write_json(path, document)


def _copy_usd_sublayer_packages(
    source_usd: Path,
    destination_usd: Path,
    *,
    allowed_roots: Sequence[Path],
) -> None:
    """Copy the sibling packages a root layer sublayers.

    The material workflow writes `output.usda` whose subLayerPaths reference
    `material_library.usdz[...]` and `source_scene.usdz[...]` beside it, so
    copying the root layer alone produces an advertised Output USD that cannot
    open. USD is optional here on purpose: a missing pxr must degrade the
    export, not fail it.
    """

    try:
        from pxr import Sdf
    except ImportError:
        return
    try:
        layer = Sdf.Layer.FindOrOpen(str(source_usd))
    except Exception:  # noqa: BLE001 - a malformed layer must not stop the export
        return
    if layer is None or not layer.subLayerPaths:
        return
    for sublayer in layer.subLayerPaths:
        # `foo.usdz[inner/path.usda]` -- only the package itself is a file.
        package = str(sublayer).split("[", 1)[0]
        if not package or Path(package).is_absolute():
            continue
        source = (source_usd.parent / package).resolve()
        if not any(source.is_relative_to(root) for root in allowed_roots):
            continue
        if not source.is_file():
            continue
        # Bounding the source is not enough. A layer stored deeply under an
        # allowed root can name `../../..` and still resolve inside it, while
        # the same components walk the destination out of the staging tree --
        # overwriting another staged asset, or a file outside the export
        # entirely. Resolve the target and require it to stay beside the layer
        # it belongs to.
        target = (destination_usd.parent / package).resolve()
        if not target.is_relative_to(destination_usd.parent.resolve()):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _copy_usd_dependency_sidecars(
    source_usd: Path,
    destination_usd: Path,
    *,
    allowed_roots: Sequence[Path],
) -> None:
    """Copy the localized "<name>_assets/" dependency directories a layer uses.

    Both physics collectors localize texture/composition dependencies into a
    sibling "<name>_assets/" directory the root layer references relatively,
    so copying the root layer alone hosts an Output USD with dangling
    references. The sidecar keeps its source name: the exporter renames the
    root layer, but the references inside it still name the original sidecar
    directory. File-by-file with symlink and containment checks, matching the
    collectors' own bundling rules.
    """

    candidates = {
        source_usd.with_name(source_usd.name + "_assets"),
        source_usd.with_name(source_usd.stem + "_assets"),
    }
    # A re-export reads a layer the previous export already renamed (e.g.
    # segmented.usdc) whose internal references still name the ORIGINAL
    # sidecar directory (physics.usdc_assets). Name-derived candidates miss
    # it, so also take sibling *_assets directories, under the same
    # symlink/containment checks.
    try:
        candidates.update(source_usd.parent.glob("*_assets"))
    except OSError:
        pass
    for sidecar in sorted(candidates):
        if sidecar.is_symlink() or not sidecar.is_dir():
            continue
        if not any(sidecar.resolve().is_relative_to(root) for root in allowed_roots):
            continue
        for member in sorted(sidecar.rglob("*")):
            if member.is_symlink() or not member.is_file():
                continue
            if not any(member.resolve().is_relative_to(root) for root in allowed_roots):
                continue
            target = destination_usd.parent / sidecar.name / member.relative_to(sidecar)
            if not target.resolve().is_relative_to(destination_usd.parent.resolve()):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(member, target)


def _export_render_manifest_closure(
    manifest_path: Path,
    *,
    bundle_dir: Path,
    destination_run: Path,
    asset_root: Path,
    allowed_roots: Sequence[Path],
    exported_output_usd: str | None = None,
) -> None:
    """Copy the files a render manifest points at, and rewrite it to match.

    The manifest is the provenance record for every final image, but it
    references per-view camera and render-response files by absolute path.
    Copying it alone produced an export whose provenance links all dangled --
    the exact OVRTX metadata AGENTS.md requires travel with the evidence.
    """

    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError):
        return
    # The manifest names the scene it rendered as an absolute path on the
    # producing machine. Left alone, an export advertises final evidence whose
    # subject cannot be opened. Point it at the copied USD and keep
    # scene_sha256 untouched, so the binding still verifies.
    if exported_output_usd:
        manifest["scene"] = os.path.relpath(
            destination_run / "bundle" / exported_output_usd, manifest_path.parent
        )
    records = manifest.get("renders")
    if not isinstance(records, list):
        _write_json(manifest_path, manifest)
        return
    taken: set[str] = set()
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            continue
        for field in _RENDER_DEPENDENCY_FIELDS:
            source = _source_path(
                record.get(field), bundle_dir=bundle_dir, allowed_roots=allowed_roots
            )
            if source is None or not source.is_file():
                continue
            name = str(record.get("name") or record.get("direction") or index)
            segment = _unique_segment(f"{name}_{field}", taken=taken)
            relative = (
                asset_root
                / "renders"
                / "provenance"
                / (f"{segment}{source.suffix.lower()}")
            )
            copied = _copy_linked_file(
                record.get(field),
                bundle_dir=bundle_dir,
                destination_run=destination_run,
                relative_destination=relative,
                allowed_roots=allowed_roots,
            )
            if copied:
                # Relative to the manifest itself, not to bundle/: the manifest
                # lands beside the asset, so a bundle-relative link does not
                # resolve from where a reader opens it.
                record[field] = os.path.relpath(
                    destination_run / "bundle" / copied, manifest_path.parent
                )
        channels = record.get("channels")
        if isinstance(channels, dict):
            for channel_name, channel in channels.items():
                if not isinstance(channel, dict):
                    continue
                # `raw` is the full-resolution AOV (`.npy`) the render actually
                # produced; `preview` is only its PNG rendering. Copying the
                # preview alone left every normal and depth buffer pointing at
                # the producing machine, so a hosted export could not reproduce
                # the evidence AGENTS.md requires it to carry.
                for channel_field in ("image", "preview", "raw"):
                    source = _source_path(
                        channel.get(channel_field),
                        bundle_dir=bundle_dir,
                        allowed_roots=allowed_roots,
                    )
                    if source is None or not source.is_file():
                        continue
                    segment = _unique_segment(
                        f"{record.get('name') or index}_{channel_name}_{channel_field}",
                        taken=taken,
                    )
                    copied = _copy_linked_file(
                        channel.get(channel_field),
                        bundle_dir=bundle_dir,
                        destination_run=destination_run,
                        relative_destination=asset_root
                        / "renders"
                        / "provenance"
                        / f"{segment}{source.suffix.lower()}",
                        allowed_roots=allowed_roots,
                    )
                    if copied:
                        channel[channel_field] = os.path.relpath(
                            destination_run / "bundle" / copied, manifest_path.parent
                        )
    _write_json(manifest_path, manifest)


def _unique_segment(value: str, *, taken: set[str]) -> str:
    """Sanitize one path segment, disambiguating against segments already used."""

    base = _safe_segment(value)
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}-{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


class UnsafeSourcePathError(ValueError):
    """A bundle referenced a file outside every permitted source root."""


def _source_path(
    value: object,
    *,
    bundle_dir: Path,
    allowed_roots: Sequence[Path],
) -> Path | None:
    """Resolve a bundle-declared path, refusing anything outside the allowlist.

    The export exists to be hosted and shared, so a bundle that names
    ``/etc/passwd`` or escapes upward with ``..`` would otherwise copy local
    files straight into a shareable artifact. Paths are checked after
    resolution, so symlinks cannot be used to step outside a permitted root.
    """

    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    resolved = (path if path.is_absolute() else bundle_dir / path).resolve()
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise UnsafeSourcePathError(
            f"bundle references a file outside the permitted roots: {value!r}"
        )
    return resolved


def _copy_linked_file(
    value: object,
    *,
    bundle_dir: Path,
    destination_run: Path,
    relative_destination: Path,
    allowed_roots: Sequence[Path],
) -> str | None:
    source = _source_path(value, bundle_dir=bundle_dir, allowed_roots=allowed_roots)
    if source is None or not source.is_file():
        return None
    destination = destination_run / "bundle" / relative_destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return relative_destination.as_posix()


def _usd_suffix(
    value: object,
    *,
    bundle_dir: Path,
    allowed_roots: Sequence[Path],
) -> str:
    """Return the source's own USD extension, defaulting to the binary crate."""

    source = _source_path(value, bundle_dir=bundle_dir, allowed_roots=allowed_roots)
    suffix = source.suffix.lower() if source is not None else ""
    return suffix or ".usdc"


def _copy_optional_document(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def export_run(
    source_run: Path,
    destination_root: Path,
    *,
    allowed_roots: Sequence[Path] = (),
) -> Path:
    """Copy one run into a self-contained, shareable export.

    ``allowed_roots`` bounds which files a bundle may pull in. The run
    directory is always permitted; a dataset root has to be named explicitly,
    because assets such as ``source_usd`` legitimately live outside the run.
    """

    source_run = source_run.resolve()
    permitted = [source_run, *(root.resolve() for root in allowed_roots)]
    manifest_path = source_run / "benchmark_run.json"
    bundle_path = source_run / "bundle" / "run.json"
    manifest = _read_json(manifest_path)
    bundle = _read_json(bundle_path)
    workflow = _safe_segment(str(manifest.get("workflow") or bundle.get("workflow")))
    run_id = _safe_segment(str(manifest.get("run_id") or bundle.get("run_id")))
    final_run = destination_root / workflow / run_id
    if final_run.exists():
        raise FileExistsError(
            f"dashboard export already exists: {final_run}; choose a fresh root"
        )

    # Build into a staging directory and move it into place only once the
    # export is complete. A half-written export never appears at the final
    # path, so a failure needs no cleanup there -- and cleanup can never reach
    # a previously completed export.
    final_run.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=final_run.parent))
    destination_run = staging
    try:
        shutil.copy2(manifest_path, destination_run / "benchmark_run.json")
        _copy_optional_document(
            source_run / "score" / "suite_result.json",
            destination_run / "score" / "suite_result.json",
        )
        _sanitize_suite_result_provenance(
            destination_run / "score" / "suite_result.json",
            source_bundle_dir=source_run / "bundle",
        )
        _copy_optional_document(
            source_run / "score" / "report.html",
            destination_run / "score" / "report.html",
        )
        # The report's appendix documents its exact records at this sibling
        # filename; the hosted report must not 404 on its own contract. The
        # exporter is the last line of defense for the shareable artifact,
        # so the sidecar is screened before copying: a stale pre-gate
        # renderer (or a replaced file) could carry credentialed endpoints,
        # and an unparseable or credential-bearing sidecar is omitted.
        report_sidecar = source_run / "score" / "report.html.provenance.exact.json"
        if report_sidecar.is_file():
            try:
                sidecar_records = json.loads(report_sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                sidecar_records = None
            if isinstance(sidecar_records, dict) and not _carries_credentialed_endpoint(
                sidecar_records
            ):
                _copy_optional_document(
                    report_sidecar,
                    destination_run / "score" / "report.html.provenance.exact.json",
                )
        _copy_optional_document(
            source_run / "evaluation" / "material_evaluation.json",
            destination_run / "evaluation" / "material_evaluation.json",
        )

        bundle_dir = bundle_path.parent
        exported_assets: list[dict[str, Any]] = []
        used_asset_segments: set[str] = set()
        for source_asset in bundle.get("assets", []):
            if not isinstance(source_asset, dict):
                continue
            asset = dict(source_asset)
            # Distinct ids can sanitize to the same segment (`part/a` and `part_a`),
            # in which case the later asset would overwrite the earlier one's
            # renders, reports and metrics while both advertised the same
            # export_root. Disambiguate on collision instead of losing evidence.
            asset_id = _unique_segment(
                str(asset.get("asset_id")), taken=used_asset_segments
            )
            asset_root = Path("assets") / asset_id

            references: list[dict[str, Any]] = []
            for index, reference in enumerate(asset.get("references") or [], 1):
                if not isinstance(reference, dict):
                    continue
                label = str(reference.get("label") or f"reference_{index}")
                source = _source_path(
                    reference.get("path"),
                    bundle_dir=bundle_dir,
                    allowed_roots=permitted,
                )
                suffix = source.suffix.lower() if source is not None else ".png"
                relative = (
                    asset_root
                    / "references"
                    / f"{index:02d}_{_safe_segment(label)}{suffix}"
                )
                copied = _copy_linked_file(
                    reference.get("path"),
                    bundle_dir=bundle_dir,
                    destination_run=destination_run,
                    relative_destination=relative,
                    allowed_roots=permitted,
                )
                if copied:
                    record: dict[str, Any] = {"label": label, "path": copied}
                    # The provenance record (renderer identity, OVRTX render
                    # metadata, source USD/image digests) must travel with the
                    # image — but only after revalidating the image digest
                    # against the bytes actually exported. Hash the COPIED
                    # file, not the mutable source: a reference replaced or
                    # retargeted mid-export must not ship provenance that
                    # does not match the published bytes. On mismatch the
                    # image ships without provenance and the dashboard
                    # labels it diagnostic.
                    provenance = reference.get("provenance")
                    exact_record = provenance
                    if isinstance(provenance, dict) and (
                        "exact_record_path" in provenance
                        or "exact_record_sha256" in provenance
                    ):
                        # Re-export of an already exported run: the input
                        # record is the previous export's sanitized
                        # projection, so hashing and shipping IT as the
                        # exact record would promote a lossy copy. Recover
                        # the original exact record from the sidecar the
                        # projection points at — and accept it only when
                        # its canonical bytes hash to the projection's
                        # exact_record_sha256, so a stale or replaced
                        # sidecar cannot be re-signed as canonical. On any
                        # miss, ship the image without provenance (fail
                        # closed) rather than a false exact record.
                        exact_record = None
                        prior_sidecar = _source_path(
                            provenance.get("exact_record_path"),
                            bundle_dir=bundle_dir,
                            allowed_roots=permitted,
                        )
                        if prior_sidecar is not None and prior_sidecar.is_file():
                            try:
                                loaded = json.loads(
                                    prior_sidecar.read_text(encoding="utf-8")
                                )
                            except (OSError, ValueError):
                                loaded = None
                            if isinstance(loaded, dict):
                                canonical = json.dumps(
                                    loaded,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                    ensure_ascii=True,
                                ).encode("utf-8")
                                if hashlib.sha256(
                                    canonical
                                ).hexdigest() == provenance.get("exact_record_sha256"):
                                    exact_record = loaded
                    copied_path = destination_run / "bundle" / copied
                    if isinstance(
                        exact_record, dict
                    ) and _carries_credentialed_endpoint(exact_record):
                        # The verbatim sidecar would publish the credential
                        # the projection just dropped; redaction would break
                        # the digest binding, so the reference ships without
                        # provenance instead.
                        exact_record = None
                    if (
                        isinstance(exact_record, dict)
                        and copied_path.is_file()
                        and exact_record.get("image_sha256")
                        == hashlib.sha256(copied_path.read_bytes()).hexdigest()
                    ):
                        # Display surfaces get the shareable projection: the
                        # full record's render_response AND render_metadata
                        # carry producing-machine absolute paths. But a
                        # digest alone cannot reconstruct the exact record
                        # once the source run is unavailable, so the verbatim
                        # record is preserved in an explicitly-labeled
                        # sidecar whose canonical bytes hash to
                        # exact_record_sha256 (fail-closed evidence rule).
                        record["provenance"] = _shareable_provenance(exact_record)
                        exact_relative = (
                            relative.parent / f"{relative.stem}.provenance.exact.json"
                        )
                        exact_path = destination_run / "bundle" / exact_relative
                        exact_path.parent.mkdir(parents=True, exist_ok=True)
                        exact_path.write_text(
                            json.dumps(
                                exact_record,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=True,
                            ),
                            encoding="utf-8",
                        )
                        record["provenance"]["exact_record_path"] = (
                            exact_relative.as_posix()
                        )
                    references.append(record)
            asset["references"] = references

            renders: dict[str, str] = {}
            # View names come from the agent-written render manifest, whose
            # only dedup is an exact-name suffix that does not survive
            # sanitization. Two names folding to one segment would otherwise
            # overwrite each other, leaving one camera's image shown under two
            # labels in a benchmark whose whole point is visual evidence.
            used_render_segments: set[str] = set()
            for view, value in (asset.get("renders") or {}).items():
                source = _source_path(
                    value, bundle_dir=bundle_dir, allowed_roots=permitted
                )
                suffix = source.suffix.lower() if source is not None else ".png"
                segment = _unique_segment(str(view), taken=used_render_segments)
                relative = asset_root / "renders" / f"{segment}{suffix}"
                copied = _copy_linked_file(
                    value,
                    bundle_dir=bundle_dir,
                    destination_run=destination_run,
                    relative_destination=relative,
                    allowed_roots=permitted,
                )
                if copied:
                    renders[str(view)] = copied
            asset["renders"] = renders

            # USD selects its file-format plugin by extension and the copy is
            # byte-for-byte, so forcing `.usdc` on an ASCII `.usda` layer -- which
            # is what the material workflow records -- produces a file nothing can
            # open. Keep whatever the source actually is.
            asset_source_output = asset.get("output_usd")
            output_usd = _copy_linked_file(
                asset.get("output_usd"),
                bundle_dir=bundle_dir,
                destination_run=destination_run,
                relative_destination=asset_root
                / "output"
                / f"segmented{_usd_suffix(asset.get('output_usd'), bundle_dir=bundle_dir, allowed_roots=permitted)}",
                allowed_roots=permitted,
            )
            asset["output_usd"] = output_usd or ""
            if output_usd:
                source_output = _source_path(
                    asset_source_output,
                    bundle_dir=bundle_dir,
                    allowed_roots=permitted,
                )
                if source_output is not None:
                    _copy_usd_sublayer_packages(
                        source_output,
                        destination_run / "bundle" / output_usd,
                        allowed_roots=permitted,
                    )
                    _copy_usd_dependency_sidecars(
                        source_output,
                        destination_run / "bundle" / output_usd,
                        allowed_roots=permitted,
                    )
            source_usd = _copy_linked_file(
                asset.get("source_usd"),
                bundle_dir=bundle_dir,
                destination_run=destination_run,
                relative_destination=asset_root
                / "input"
                / f"source{_usd_suffix(asset.get('source_usd'), bundle_dir=bundle_dir, allowed_roots=permitted)}",
                allowed_roots=permitted,
            )
            asset["source_usd"] = source_usd or ""

            local_artifacts: dict[str, str] = {}
            used_report_segments: set[str] = set()
            for label, value in (asset.get("local_artifacts") or {}).items():
                if label == "run_dir":
                    continue
                source = _source_path(
                    value, bundle_dir=bundle_dir, allowed_roots=permitted
                )
                if source is None:
                    continue
                suffix = source.suffix.lower()
                segment = _unique_segment(str(label), taken=used_report_segments)
                relative = asset_root / "reports" / f"{segment}{suffix}"
                copied = _copy_linked_file(
                    value,
                    bundle_dir=bundle_dir,
                    destination_run=destination_run,
                    relative_destination=relative,
                    allowed_roots=permitted,
                )
                if copied:
                    local_artifacts[str(label)] = copied
            asset["local_artifacts"] = local_artifacts
            manifest_relative = local_artifacts.get("render_manifest")
            if manifest_relative:
                _export_render_manifest_closure(
                    destination_run / "bundle" / manifest_relative,
                    bundle_dir=bundle_dir,
                    destination_run=destination_run,
                    asset_root=asset_root,
                    allowed_roots=permitted,
                    exported_output_usd=asset.get("output_usd") or None,
                )
            asset["artifact_links"] = []

            metrics_path = destination_run / "bundle" / asset_root / "metrics.json"
            _write_json(metrics_path, dict(asset.get("metrics") or {}))
            # The directory is sanitized but `asset_id` is not -- it is the join
            # key with the scored cases and must stay verbatim. Record where the
            # asset's files actually landed so the dashboard does not have to
            # reimplement this sanitization and drift from it.
            asset["export_root"] = asset_root.as_posix()

            # The dashboard fetches every persisted review with Promise.all, so a
            # single dangling review_path rejects the whole exported run rather
            # than degrading that one asset. Copy it, or clear the pointer.
            review_path = asset.get("review_path")
            if review_path:
                copied_review = _copy_linked_file(
                    review_path,
                    bundle_dir=bundle_dir,
                    destination_run=destination_run,
                    relative_destination=asset_root / "review.json",
                    allowed_roots=permitted,
                )
                if copied_review:
                    asset["review_path"] = copied_review
                else:
                    asset.pop("review_path", None)
            exported_assets.append(asset)

        bundle["assets"] = exported_assets
        _write_json(destination_run / "bundle" / "run.json", bundle)
        _publish_modes(staging)
        staging.replace(final_run)
    except BaseException:
        # Staging is ours alone, so removing it cannot touch a completed
        # export -- which is exactly what the previous cleanup did.
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final_run


def discover_runs(
    artifact_root: Path,
    *,
    workflow: str | None,
    run_id: str | None,
) -> list[Path]:
    candidates = sorted(artifact_root.glob("*/*/benchmark_run.json"))
    runs = []
    for manifest_path in candidates:
        try:
            manifest = _read_json(manifest_path)
        except (OSError, ValueError) as error:
            # main() isolates each run deliberately; discovery raising here
            # aborted the whole batch before that isolation could apply.
            print(f"skipped {manifest_path}: {error}", file=sys.stderr)
            continue
        if workflow and manifest.get("workflow") != workflow:
            continue
        if run_id and manifest.get("run_id") != run_id:
            continue
        if (manifest_path.parent / "bundle" / "run.json").is_file():
            runs.append(manifest_path.parent)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workflow")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--allow-source-root",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "Additional directory a bundle may copy files from. The run "
            "directory is always allowed; name the dataset root here so "
            "assets recorded outside the run (source_usd) are exported."
        ),
    )
    args = parser.parse_args()
    runs = discover_runs(
        args.artifact_root.resolve(),
        workflow=args.workflow,
        run_id=args.run_id,
    )
    if not runs:
        parser.error("no matching standardized benchmark runs found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for source_run in runs:
        try:
            destination = export_run(
                source_run,
                args.output_root.resolve(),
                allowed_roots=args.allow_source_root,
            )
        except (UnsafeSourcePathError, OSError, ValueError) as error:
            # One unexportable run must not abandon the rest of the batch.
            # export_run stages its work, so a failure leaves nothing behind to
            # clean up here -- and nothing this loop could delete by mistake.
            failures += 1
            print(f"skipped {source_run}: {error}", file=sys.stderr)
            continue
        print(destination)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
