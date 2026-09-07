# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VoMP mass-property authoring for the agentic physics workflow."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

PHYSICS_VOMP_RESULT_SCHEMA_VERSION = "content-agent-workflows.physics-vomp-result.v1"


def _vomp_default(name: str) -> Any:
    """Load official VoMP defaults only after a caller enables VoMP."""

    from physics_agent.integrations import vomp_defaults

    return getattr(vomp_defaults, name)


def _default_vomp_revision() -> str:
    return str(_vomp_default("DEFAULT_VOMP_REVISION"))


def _default_vomp_artifact_sha256() -> dict[str, str]:
    return {
        str(name): str(digest)
        for name, digest in dict(_vomp_default("DEFAULT_VOMP_ARTIFACT_SHA256")).items()
    }


def _default_vomp_num_views() -> int:
    return int(_vomp_default("DEFAULT_VOMP_NUM_VIEWS"))


def _default_vomp_image_width() -> int:
    return int(_vomp_default("DEFAULT_VOMP_IMAGE_WIDTH"))


def _default_vomp_image_height() -> int:
    return int(_vomp_default("DEFAULT_VOMP_IMAGE_HEIGHT"))


def _default_vomp_seed() -> int:
    return int(_vomp_default("DEFAULT_VOMP_SEED"))


def _default_vomp_render_mode() -> Literal["rt1", "rt2", "pt"]:
    value = str(_vomp_default("DEFAULT_VOMP_RENDER_MODE"))
    if value not in {"rt1", "rt2", "pt"}:
        raise RuntimeError(f"Unsupported official VoMP render mode: {value!r}")
    return cast(Literal["rt1", "rt2", "pt"], value)


def _default_vomp_num_sensor_updates() -> int:
    return int(_vomp_default("DEFAULT_VOMP_NUM_SENSOR_UPDATES"))


def _default_vomp_material_target() -> Literal[
    "auto", "preview_surface", "openpbr_materialx"
]:
    value = str(_vomp_default("DEFAULT_VOMP_MATERIAL_TARGET"))
    if value not in {"auto", "preview_surface", "openpbr_materialx"}:
        raise RuntimeError(f"Unsupported official VoMP material target: {value!r}")
    return cast(
        Literal["auto", "preview_surface", "openpbr_materialx"],
        value,
    )


class PhysicsVompMassConfig(BaseModel):
    """Opt-in VoMP runtime and render contract for one rigid body."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    runtime_root: Path
    target_prim_path: str | None = None
    python_executable: Path | None = None
    config_path: Path = Path("weights/inference.json")
    expected_revision: str = Field(default_factory=_default_vomp_revision)
    expected_artifact_sha256: dict[str, str] = Field(
        default_factory=_default_vomp_artifact_sha256
    )
    attention_backend: Literal["xformers", "sdpa", "naive"] = "xformers"
    timeout_seconds: float = Field(default=3600.0, gt=0.0)
    max_complete_voxels: int = Field(default=262_144, gt=0)
    num_views: int = Field(default_factory=_default_vomp_num_views, gt=0)
    image_width: int = Field(default_factory=_default_vomp_image_width, gt=0)
    image_height: int = Field(default_factory=_default_vomp_image_height, gt=0)
    seed: int = Field(default_factory=_default_vomp_seed, ge=0, le=2**32 - 1)
    render_mode: Literal["rt1", "rt2", "pt"] = Field(
        default_factory=_default_vomp_render_mode
    )
    num_sensor_updates: int = Field(
        default_factory=_default_vomp_num_sensor_updates,
        gt=0,
    )
    material_target: Literal["auto", "preview_surface", "openpbr_materialx"] = Field(
        default_factory=_default_vomp_material_target
    )
    ovrtx_venv_dir: Path | None = None

    @field_validator("target_prim_path")
    @classmethod
    def _validate_target_prim_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        target = value.strip()
        if not target.startswith("/") or target == "/":
            raise ValueError("VoMP target prim must be an absolute USD prim path")
        return target

    @field_validator("expected_revision")
    @classmethod
    def _validate_revision(cls, value: str) -> str:
        revision = value.strip()
        if len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision
        ):
            raise ValueError("VoMP expected revision must be a full lowercase git SHA")
        return revision

    @field_validator("expected_artifact_sha256")
    @classmethod
    def _validate_artifact_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        expected_keys = set(_default_vomp_artifact_sha256())
        if set(value) != expected_keys:
            raise ValueError(
                "VoMP artifact hashes must pin exactly: "
                + ", ".join(sorted(expected_keys))
            )
        for name, digest in value.items():
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError(
                    f"VoMP artifact hash {name!r} must be lowercase SHA-256"
                )
        return dict(value)


class PhysicsVompMassResult(BaseModel):
    """Canonical VoMP artifacts emitted during agentic finalization."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = PHYSICS_VOMP_RESULT_SCHEMA_VERSION
    target_prim_path: str
    input_usd_path: str
    output_usd_path: str
    output_usd_sha256: str
    provenance_path: str
    provenance_sha256: str
    evidence_dir: str
    evidence_manifest_path: str
    vomp_npz_path: str
    worker_manifest_path: str
    worker_log_path: str
    sample_count: int
    mass_kg: float
    center_of_mass_local_m: tuple[float, float, float]
    diagonal_inertia_kg_m2: tuple[float, float, float]
    principal_axes_wxyz: tuple[float, float, float, float]


def resolve_vomp_target_prim(
    config: PhysicsVompMassConfig,
    decisions: Iterable[object],
) -> str:
    """Resolve an explicit target or the decisions' sole mass-authoring path."""

    targets: set[str] = set()
    for decision in decisions:
        if isinstance(decision, Mapping):
            raw_target = decision.get("mass_authoring_path")
        else:
            raw_target = getattr(decision, "mass_authoring_path", None)
        if isinstance(raw_target, str) and raw_target.strip():
            targets.add(raw_target.strip())
    if config.target_prim_path is not None:
        if targets and config.target_prim_path not in targets:
            raise RuntimeError(
                "Explicit VoMP target is not an accepted mass-authoring path: "
                f"{config.target_prim_path}"
            )
        return config.target_prim_path
    if len(targets) == 1:
        return targets.pop()
    if not targets:
        raise RuntimeError(
            "VoMP target could not be inferred because the physics decisions contain "
            "no mass_authoring_path; pass --vomp-target-prim."
        )
    raise RuntimeError(
        "VoMP target is ambiguous across physics decisions "
        f"({', '.join(sorted(targets))}); pass --vomp-target-prim."
    )


def run_agentic_vomp_mass_authoring(
    *,
    input_usd_path: Path,
    output_usd_path: Path,
    output_dir: Path,
    target_prim_path: str,
    config: PhysicsVompMassConfig,
    artifact_namespace: str | None = None,
) -> PhysicsVompMassResult:
    """Run the attested VoMP adapter and return agentic artifact metadata."""

    from physics_agent.integrations.vomp_pipeline import (
        VompRenderConfig,
        run_vomp_mass_pipeline,
    )
    from physics_agent.integrations.vomp_runtime import VompRuntimeConfig

    source = input_usd_path.expanduser().resolve()
    output = output_usd_path.expanduser().resolve()
    run_root = output_dir.expanduser().resolve()
    runtime_root = config.runtime_root.expanduser().resolve()
    python_executable = config.python_executable or Path(".venv/bin/python")
    if not python_executable.is_absolute():
        python_executable = runtime_root / python_executable
    config_path = config.config_path.expanduser()
    if not config_path.is_absolute():
        config_path = runtime_root / config_path
    ovrtx_venv_dir = config.ovrtx_venv_dir
    if ovrtx_venv_dir is not None:
        ovrtx_venv_dir = ovrtx_venv_dir.expanduser().resolve()

    if artifact_namespace is not None and (
        not artifact_namespace
        or len(artifact_namespace) > 64
        or artifact_namespace[0]
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in artifact_namespace
        )
    ):
        raise ValueError("VoMP artifact namespace contains unsupported characters")
    if artifact_namespace is None:
        work_dir = run_root / "vomp"
        provenance_path = run_root / "raw" / "physics_vomp_mass_properties.json"
    else:
        work_dir = run_root / "vomp" / artifact_namespace
        provenance_path = (
            run_root
            / "raw"
            / "vomp"
            / artifact_namespace
            / "physics_vomp_mass_properties.json"
        )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_vomp_mass_pipeline(
        source,
        output,
        target_prim_path=target_prim_path,
        work_dir=work_dir,
        runtime_config=VompRuntimeConfig(
            runtime_root=runtime_root,
            python_executable=python_executable,
            config_path=config_path,
            expected_revision=config.expected_revision,
            expected_artifact_sha256=config.expected_artifact_sha256,
            attention_backend=config.attention_backend,
            timeout_seconds=config.timeout_seconds,
            max_complete_voxels=config.max_complete_voxels,
        ),
        render_config=VompRenderConfig(
            num_views=config.num_views,
            image_width=config.image_width,
            image_height=config.image_height,
            seed=config.seed,
            render_mode=config.render_mode,
            num_sensor_updates=config.num_sensor_updates,
            material_target=config.material_target,
            ovrtx_venv_dir=str(ovrtx_venv_dir) if ovrtx_venv_dir is not None else None,
        ),
        provenance_path=provenance_path,
    )
    apply_result = result.apply_result
    properties = apply_result.mass_properties
    agentic_result = PhysicsVompMassResult(
        target_prim_path=target_prim_path,
        input_usd_path=str(source),
        output_usd_path=str(apply_result.output_usd_path),
        output_usd_sha256=_sha256(apply_result.output_usd_path),
        provenance_path=str(apply_result.provenance_path),
        provenance_sha256=_sha256(apply_result.provenance_path),
        evidence_dir=str(result.evidence.artifact_dir),
        evidence_manifest_path=str(result.evidence.manifest_path),
        vomp_npz_path=str(result.vomp_npz_path),
        worker_manifest_path=str(result.worker_manifest_path),
        worker_log_path=str(result.worker_log_path),
        sample_count=apply_result.sample_count,
        mass_kg=properties.mass_kg,
        center_of_mass_local_m=properties.center_of_mass_local_m,
        diagonal_inertia_kg_m2=properties.diagonal_inertia_kg_m2,
        principal_axes_wxyz=properties.principal_axes_wxyz,
    )
    verify_vomp_mass_properties(apply_result.output_usd_path, agentic_result)
    return agentic_result


def verify_vomp_mass_properties(
    usd_path: Path | str,
    result: PhysicsVompMassResult,
    *,
    provenance_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Fail unless a USD still carries the exact VoMP-authored mass contract."""

    external_provenance = _load_vomp_provenance(
        result,
        provenance_bytes=provenance_bytes,
    )
    _require_vomp_provenance_matches(
        external_provenance,
        result,
        label="external provenance",
    )

    from pxr import Usd, UsdGeom, UsdPhysics

    source = Path(usd_path).expanduser().resolve()
    stage = Usd.Stage.Open(str(source))
    if not stage:
        raise RuntimeError(f"Unable to open USD for VoMP mass verification: {source}")
    target = stage.GetPrimAtPath(result.target_prim_path)
    if not target or not target.IsValid():
        raise RuntimeError(
            f"VoMP target prim is missing from candidate USD: {result.target_prim_path}"
        )
    if not target.HasAPI(UsdPhysics.MassAPI):
        raise RuntimeError(
            f"VoMP target no longer has MassAPI: {result.target_prim_path}"
        )

    if not UsdGeom.StageHasAuthoredMetersPerUnit(stage):
        raise RuntimeError("VoMP candidate no longer authors metersPerUnit")
    if not UsdPhysics.StageHasAuthoredKilogramsPerUnit(stage):
        raise RuntimeError("VoMP candidate no longer authors kilogramsPerUnit")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    kilograms_per_unit = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise RuntimeError("VoMP candidate has invalid metersPerUnit")
    if not math.isfinite(kilograms_per_unit) or kilograms_per_unit <= 0.0:
        raise RuntimeError("VoMP candidate has invalid kilogramsPerUnit")

    mass_api = UsdPhysics.MassAPI(target)
    mass_stage = mass_api.GetMassAttr().Get()
    center_stage = mass_api.GetCenterOfMassAttr().Get()
    inertia_stage = mass_api.GetDiagonalInertiaAttr().Get()
    axes_stage = mass_api.GetPrincipalAxesAttr().Get()
    if any(
        value is None for value in (mass_stage, center_stage, inertia_stage, axes_stage)
    ):
        raise RuntimeError(
            f"VoMP target has incomplete MassAPI properties: {result.target_prim_path}"
        )

    mass_kg = float(mass_stage) * kilograms_per_unit
    center_local_m = tuple(float(value) * meters_per_unit for value in center_stage)
    inertia_factor = kilograms_per_unit * meters_per_unit * meters_per_unit
    diagonal_inertia_kg_m2 = tuple(
        float(value) * inertia_factor for value in inertia_stage
    )
    axes_imaginary = axes_stage.GetImaginary()
    principal_axes_wxyz = (
        float(axes_stage.GetReal()),
        *(float(value) for value in axes_imaginary),
    )

    _require_vomp_values_close("mass", (mass_kg,), (result.mass_kg,))
    _require_vomp_values_close(
        "center of mass",
        center_local_m,
        result.center_of_mass_local_m,
    )
    _require_vomp_values_close(
        "diagonal inertia",
        diagonal_inertia_kg_m2,
        result.diagonal_inertia_kg_m2,
    )
    if not _vomp_values_close(principal_axes_wxyz, result.principal_axes_wxyz):
        equivalent_negation = tuple(-value for value in result.principal_axes_wxyz)
        if not _vomp_values_close(principal_axes_wxyz, equivalent_negation):
            raise RuntimeError(
                "VoMP principal axes changed after mass authoring: "
                f"expected {result.principal_axes_wxyz}, got {principal_axes_wxyz}"
            )

    provenance = target.GetCustomDataByKey("physicsAgentVomp")
    if not isinstance(provenance, Mapping):
        raise RuntimeError(
            f"VoMP provenance is missing from target: {result.target_prim_path}"
        )
    _require_vomp_provenance_matches(provenance, result, label="USD provenance")
    _require_vomp_embedded_provenance_matches(provenance, external_provenance)

    return {
        "verified": True,
        "usd_path": str(source),
        "target_prim_path": result.target_prim_path,
        "provenance_sha256": result.provenance_sha256,
        "mass_kg": mass_kg,
        "center_of_mass_local_m": list(center_local_m),
        "diagonal_inertia_kg_m2": list(diagonal_inertia_kg_m2),
        "principal_axes_wxyz": list(principal_axes_wxyz),
    }


def _load_vomp_provenance(
    result: PhysicsVompMassResult,
    *,
    provenance_bytes: bytes | None,
) -> Mapping[str, Any]:
    if provenance_bytes is None:
        provenance_path = Path(result.provenance_path).expanduser().resolve(strict=True)
        provenance_bytes = provenance_path.read_bytes()
    actual_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
    if actual_sha256 != result.provenance_sha256:
        raise RuntimeError(
            "VoMP external provenance digest changed: "
            f"expected {result.provenance_sha256}, got {actual_sha256}"
        )
    try:
        payload = json.loads(provenance_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("VoMP external provenance is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("VoMP external provenance must be a JSON object")
    return payload


def _require_vomp_provenance_matches(
    provenance: Mapping[str, Any],
    result: PhysicsVompMassResult,
    *,
    label: str,
) -> None:
    association = provenance.get("association")
    if not isinstance(association, Mapping) or association.get("targetPrimPath") != (
        result.target_prim_path
    ):
        raise RuntimeError(f"VoMP {label} does not match the authored prim")
    provenance_properties = provenance.get("rigidBodyMassProperties")
    if not isinstance(provenance_properties, Mapping):
        raise RuntimeError(f"VoMP rigid-body mass {label} is missing")
    try:
        provenance_mass = float(provenance_properties.get("massKg", math.nan))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"VoMP {label} mass is malformed") from exc
    _require_vomp_values_close(
        f"{label} mass",
        (provenance_mass,),
        (result.mass_kg,),
    )
    for property_label, key, expected in (
        (
            f"{label} center of mass",
            "centerOfMassLocalM",
            result.center_of_mass_local_m,
        ),
        (
            f"{label} diagonal inertia",
            "diagonalInertiaKgM2",
            result.diagonal_inertia_kg_m2,
        ),
        (
            f"{label} principal axes",
            "principalAxesWxyz",
            result.principal_axes_wxyz,
        ),
    ):
        raw_values = provenance_properties.get(key)
        if raw_values is None or isinstance(raw_values, str | bytes | Mapping):
            raise RuntimeError(f"VoMP {property_label} is missing")
        try:
            values = tuple(float(value) for value in raw_values)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"VoMP {property_label} is malformed") from exc
        _require_vomp_values_close(
            property_label,
            values,
            expected,
        )

    voxel_field = provenance.get("voxelField")
    if not isinstance(voxel_field, Mapping) or voxel_field.get("sampleCount") != (
        result.sample_count
    ):
        observed_sample_count = (
            voxel_field.get("sampleCount") if isinstance(voxel_field, Mapping) else None
        )
        raise RuntimeError(
            f"VoMP {label} sample count changed: expected {result.sample_count}, "
            f"got {observed_sample_count!r}"
        )


def _require_vomp_embedded_provenance_matches(
    embedded: Mapping[str, Any],
    external: Mapping[str, Any],
) -> None:
    """Compare every authored provenance field, including renderer metadata."""

    external_json = _vomp_plain_json(external, label="external provenance")
    if not isinstance(external_json, dict):
        raise RuntimeError("VoMP external provenance must be a JSON object")

    exact_renderer_metadata: Any = None
    external_evidence = external_json.get("evidence")
    if isinstance(external_evidence, dict):
        external_rendering = external_evidence.get("rendering")
        if isinstance(external_rendering, dict):
            exact_renderer_metadata = external_rendering.get("rendererMetadata")

    expected_embedded = _vomp_remove_json_nulls(external_json)
    expected_evidence = expected_embedded.get("evidence")
    if isinstance(expected_evidence, dict):
        expected_rendering = expected_evidence.get("rendering")
        if isinstance(expected_rendering, dict):
            expected_rendering.pop("rendererMetadata", None)
            if exact_renderer_metadata is not None:
                expected_rendering["rendererMetadataJson"] = json.dumps(
                    exact_renderer_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )

    embedded_json = _vomp_plain_json(embedded, label="USD provenance")
    expected_canonical = json.dumps(
        expected_embedded,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    embedded_canonical = json.dumps(
        embedded_json,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if embedded_canonical != expected_canonical:
        raise RuntimeError(
            "VoMP USD provenance differs from the complete external provenance"
        )


def _vomp_plain_json(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"VoMP {label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _vomp_plain_json(child, label=label)
            for key, child in value.items()
        }
    if isinstance(value, bytes):
        raise RuntimeError(f"VoMP {label} contains non-JSON bytes")
    if isinstance(value, Iterable):
        return [_vomp_plain_json(child, label=label) for child in value]
    raise RuntimeError(
        f"VoMP {label} contains unsupported value {type(value).__name__}"
    )


def _vomp_remove_json_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _vomp_remove_json_nulls(child)
            for key, child in value.items()
            if child is not None
        }
    if isinstance(value, list):
        return [_vomp_remove_json_nulls(child) for child in value if child is not None]
    return value


def _vomp_values_close(
    actual: Iterable[float],
    expected: Iterable[float],
) -> bool:
    actual_values = tuple(actual)
    expected_values = tuple(expected)
    return len(actual_values) == len(expected_values) and all(
        math.isclose(
            actual_value,
            expected_value,
            rel_tol=2.0e-6,
            abs_tol=1.0e-12,
        )
        for actual_value, expected_value in zip(
            actual_values,
            expected_values,
            strict=True,
        )
    )


def _require_vomp_values_close(
    label: str,
    actual: Iterable[float],
    expected: Iterable[float],
) -> None:
    actual_values = tuple(actual)
    expected_values = tuple(expected)
    if not _vomp_values_close(actual_values, expected_values):
        raise RuntimeError(
            f"VoMP {label} changed after mass authoring: "
            f"expected {expected_values}, got {actual_values}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vomp_contract_payload(config: PhysicsVompMassConfig | None) -> dict[str, Any]:
    """Return the non-secret VoMP portion of the durable agent contract."""

    if config is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "provider": "vomp",
        "target_prim_path": config.target_prim_path,
        "target_selection": (
            "explicit"
            if config.target_prim_path is not None
            else "single_mass_authoring_path"
        ),
        "expected_revision": config.expected_revision,
        "render": {
            "num_views": config.num_views,
            "image_width": config.image_width,
            "image_height": config.image_height,
            "seed": config.seed,
            "render_mode": config.render_mode,
            "num_sensor_updates": config.num_sensor_updates,
            "material_target": config.material_target,
        },
    }


__all__ = [
    "PHYSICS_VOMP_RESULT_SCHEMA_VERSION",
    "PhysicsVompMassConfig",
    "PhysicsVompMassResult",
    "resolve_vomp_target_prim",
    "run_agentic_vomp_mass_authoring",
    "verify_vomp_mass_properties",
    "vomp_contract_payload",
]
