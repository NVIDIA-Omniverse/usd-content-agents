# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rendered swatch evidence and VLM judgment for material refinement."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from world_understanding.agentic.domain_tasks.model_provisioning import (
    ModelProvisioningTask,
)
from world_understanding.functions.graphics.rendering_backend_factory import (
    create_rendering_backend,
)
from world_understanding.utils.credentials import redact_sensitive_config
from world_understanding.utils.llm_parsing import (
    extract_json_from_llm_response,
    extract_labeled_choice,
    extract_labeled_score,
)

from .artifacts import artifact_sha256
from .contracts import (
    MaterialRefinementGoal,
    MaterialRefinementJudgeSettings,
    MaterialRefinementRenderSettings,
)
from .objective import MaterialFeatures

_SWATCH_PRIM_PATH = "/World/MaterialSwatch"
_SWATCH_LATITUDE_SEGMENTS = 64
_SWATCH_LONGITUDE_SEGMENTS = 128
_TARGET_JUDGE_SYSTEM_PROMPT = (
    "You are an expert visual judge of physically based 3D materials."
)
_TARGET_INFERENCE_SYSTEM_PROMPT = (
    "You infer normalized PBR controls for linear USD shader inputs from a "
    "requested material appearance."
)
_TARGET_INFERENCE_PROMPT = """Infer an internal PBR optimization target for the requested appearance.

Requested appearance:
{appearance_prompt}

Material profile: {material_profile}
Source representation: {source_representation}
Supported controls: {supported_controls}

Current candidate controls:
{current_controls}

Previous rendered-material review:
{previous_feedback}

Translate only actionable feedback about the supported controls. Ignore requests
to alter geometry, tessellation, normals, camera, lighting, exposure, background,
shader topology, or unavailable texture/detail controls. Base-color values are
linear RGB shader inputs, not display-encoded sRGB values.

Allowed target ranges:
{control_bounds}

Return exactly one JSON object with this shape:
{{
  "base_color": [0.0, 0.0, 0.0],
  "roughness": 0.0,
  "metallic": 0.0,
  "reasoning": "brief explanation"
}}

Every numeric value must lie inside its corresponding allowed range. These are
internal optimization controls inferred from the appearance prompt, not user
inputs. Do not return markdown or any additional keys.
"""
_TARGET_JUDGE_PROMPT = """Evaluate whether the rendered material swatch matches the requested appearance.

Target appearance:
{appearance_prompt}

Material profile: {material_profile}
Source representation: {source_representation}
Controllable properties in this run: {controllable_properties}

Previously approved material variations:
{comparison_context}

The rendered images are the candidate evidence. Score only appearance dimensions
that the listed controllable properties can change: color, roughness and highlight
response, metallic versus dielectric response, and texture character only when a
texture control is listed. Judge semantic appearance within those capabilities.

The canonical swatch geometry, silhouette tessellation, normals, camera, lighting,
exposure, and background are fixed evaluation-fixture properties. Do not lower the
material score because of them and do not recommend changing them. Do not request
new textures, shader nodes, geometry, or controls that are absent from the listed
controllable properties. Internal proxy metrics are optimization guidance only and
must not override what is visible.

Respond in exactly this format:

**Critique:**
[A complete visual comparison, including what matches and what does not.]

**Score:** [0-10, where {approval_score:g} or higher is acceptable]

**Decision:** [APPROVE only if the score is at least {approval_score:g} and the rendered appearance meets the target, otherwise CONTINUE]

**Improvement Suggestions:**
[If CONTINUE, give concrete generation changes for the next iteration.]
"""


class RenderTaskLike(Protocol):
    """Structural contract used for deterministic render-task injection."""

    def run(
        self,
        context: dict[str, Any],
        object_store: Any = None,
    ) -> dict[str, Any]: ...


class VlmJudgeLike(Protocol):
    """Existing Material Agent VLM image-pair invocation contract."""

    def generate_with_image_caption_pairs(
        self,
        *,
        image_caption_pairs: list[tuple[str, str]],
        final_prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str: ...


@dataclass(frozen=True)
class MaterialRenderEvidence:
    """Canonical swatch stage and render-task evidence for one sweep winner."""

    swatch_usd_path: Path
    flattened_usd_path: Path | None
    rendered_image_paths: tuple[Path, ...]
    backend: str
    rendering_stats: dict[str, Any]
    render_validation: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "swatch_usd_path": self.swatch_usd_path.as_posix(),
            "flattened_usd_path": (
                self.flattened_usd_path.as_posix()
                if self.flattened_usd_path is not None
                else None
            ),
            "rendered_image_paths": [
                path.as_posix() for path in self.rendered_image_paths
            ],
            "backend": self.backend,
            "rendering_stats": _redacted(self.rendering_stats),
            "render_validation": _redacted(list(self.render_validation)),
            "artifact_sha256": {
                "swatch_usd": artifact_sha256(self.swatch_usd_path),
                **(
                    {"flattened_usd": artifact_sha256(self.flattened_usd_path)}
                    if self.flattened_usd_path is not None
                    and self.flattened_usd_path.is_file()
                    else {}
                ),
                **{
                    f"rendered_image_{index}": artifact_sha256(path)
                    for index, path in enumerate(self.rendered_image_paths, start=1)
                },
            },
        }


@dataclass(frozen=True)
class MaterialJudgeVerdict:
    """Complete target-specific VLM response and fail-closed parsed decision."""

    score: float
    decision: str
    decision_parsed: bool
    approved: bool
    reasoning: str
    feedback: str
    raw_response: str
    prompt: str
    system_prompt: str
    image_caption_pairs: tuple[tuple[str, Path], ...]
    provider: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "decision": self.decision,
            "decision_parsed": self.decision_parsed,
            "approved": self.approved,
            "reasoning": self.reasoning,
            "feedback": self.feedback,
            "raw_response": self.raw_response,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "image_caption_pairs": [
                {
                    "caption": caption,
                    "path": path.as_posix(),
                    "sha256": artifact_sha256(path),
                }
                for caption, path in self.image_caption_pairs
            ],
            "provider": dict(self.provider),
        }


@dataclass(frozen=True)
class MaterialTargetInference:
    """Auditable internal PBR controls inferred from an appearance prompt."""

    target_features: MaterialFeatures
    reasoning: str
    raw_response: str
    prompt: str
    system_prompt: str
    provider: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_features": self.target_features.to_dict(),
            "reasoning": self.reasoning,
            "raw_response": self.raw_response,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "provider": dict(self.provider),
        }


def _redacted(value: Any) -> Any:
    projected = redact_sensitive_config(value)
    return projected


def provision_rendering_backend(settings: MaterialRefinementRenderSettings) -> Any:
    """Create the configured shared rendering backend before output cleanup."""

    return create_rendering_backend(settings.backend, settings.to_task_config())


def provision_vlm_judge(settings: MaterialRefinementJudgeSettings) -> VlmJudgeLike:
    """Create the configured VLM through Material Agent model provisioning."""

    if not (settings.vlm.get("backend") or settings.vlm.get("provider")):
        raise ValueError(
            "judge.vlm must configure an existing Material Agent backend/provider"
        )
    return cast(VlmJudgeLike, ModelProvisioningTask().create_vlm(settings.vlm))


def _bounded_inferred_value(
    value: Any,
    *,
    field_name: str,
    bounds: tuple[float, float],
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"inferred {field_name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"inferred {field_name} must be numeric") from error
    minimum, maximum = bounds
    tolerance = 1e-6 * max(1.0, abs(minimum), abs(maximum))
    if (
        not math.isfinite(number)
        or number < minimum - tolerance
        or number > maximum + tolerance
    ):
        raise ValueError(
            f"inferred {field_name}={number} is outside [{minimum}, {maximum}]"
        )
    return min(maximum, max(minimum, number))


def infer_material_target(
    *,
    appearance_prompt: str,
    source_features: MaterialFeatures,
    control_bounds: dict[str, tuple[float, float]],
    settings: MaterialRefinementJudgeSettings,
    vlm_judge: VlmJudgeLike,
    previous_feedback: str | None = None,
    material_profile: str = "unspecified",
    source_representation: str = "scalar_pbr",
    supported_controls: tuple[str, ...] = (
        "base_color",
        "roughness",
        "metallic",
    ),
) -> MaterialTargetInference:
    """Infer bounded internal controls while keeping user input prompt-only."""

    required_bounds = {
        "base_color_r",
        "base_color_g",
        "base_color_b",
        "roughness",
        "metallic",
    }
    missing_bounds = sorted(required_bounds - set(control_bounds))
    if missing_bounds:
        raise ValueError(
            "material target inference is missing control bounds: "
            + ", ".join(missing_bounds)
        )
    prompt = _TARGET_INFERENCE_PROMPT.format(
        appearance_prompt=appearance_prompt,
        material_profile=material_profile,
        source_representation=source_representation,
        supported_controls=", ".join(supported_controls),
        current_controls=json.dumps(source_features.to_dict(), sort_keys=True),
        previous_feedback=str(
            _redacted(previous_feedback or "(No previous visual review.)")
        ),
        control_bounds=json.dumps(
            {
                name: {"min": bounds[0], "max": bounds[1]}
                for name, bounds in sorted(control_bounds.items())
                if name in required_bounds
            },
            sort_keys=True,
        ),
    )
    raw_response = vlm_judge.generate_with_image_caption_pairs(
        image_caption_pairs=[],
        final_prompt=prompt,
        system_prompt=_TARGET_INFERENCE_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=min(settings.max_tokens, 1024),
    )
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("target inference returned an empty response")
    safe_response = str(_redacted(raw_response))
    parsed = extract_json_from_llm_response(
        safe_response,
        expected_keys=("base_color", "roughness", "metallic"),
        log_failures=False,
    )
    if parsed is None:
        raise ValueError("target inference did not return the required JSON object")
    raw_color = parsed.get("base_color")
    if not isinstance(raw_color, list | tuple) or len(raw_color) != 3:
        raise ValueError("inferred base_color must contain three channels")
    target = MaterialFeatures(
        base_color=(
            _bounded_inferred_value(
                raw_color[0],
                field_name="base_color_r",
                bounds=control_bounds["base_color_r"],
            ),
            _bounded_inferred_value(
                raw_color[1],
                field_name="base_color_g",
                bounds=control_bounds["base_color_g"],
            ),
            _bounded_inferred_value(
                raw_color[2],
                field_name="base_color_b",
                bounds=control_bounds["base_color_b"],
            ),
        ),
        roughness=_bounded_inferred_value(
            parsed.get("roughness"),
            field_name="roughness",
            bounds=control_bounds["roughness"],
        ),
        metallic=_bounded_inferred_value(
            parsed.get("metallic"),
            field_name="metallic",
            bounds=control_bounds["metallic"],
        ),
        color_source="appearance_prompt_inference",
    )
    reasoning = parsed.get("reasoning", "")
    return MaterialTargetInference(
        target_features=target,
        reasoning=str(_redacted(reasoning)) if reasoning is not None else "",
        raw_response=safe_response,
        prompt=prompt,
        system_prompt=_TARGET_INFERENCE_SYSTEM_PROMPT,
        provider=settings.provider_evidence(),
    )


def author_material_swatch(
    *,
    material_usd_path: Path,
    material_binding: str,
    output_path: Path,
) -> Path:
    """Author a canonical sphere that binds the generated material library."""

    material_usd_path = material_usd_path.resolve()
    if not material_usd_path.is_file():
        raise FileNotFoundError("winning material USD does not exist")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    if stage is None:
        raise RuntimeError("failed to create material swatch USD")
    stage.GetRootLayer().subLayerPaths.append(material_usd_path.as_posix())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    sphere = UsdGeom.Mesh.Define(stage, _SWATCH_PRIM_PATH)
    latitude_segments = _SWATCH_LATITUDE_SEGMENTS
    longitude_segments = _SWATCH_LONGITUDE_SEGMENTS
    points = [Gf.Vec3f(0.0, 0.0, -1.0)]
    for latitude in range(1, latitude_segments):
        theta = -math.pi / 2.0 + math.pi * latitude / latitude_segments
        ring_radius = math.cos(theta)
        z = math.sin(theta)
        points.extend(
            Gf.Vec3f(
                ring_radius * math.cos(2.0 * math.pi * longitude / longitude_segments),
                ring_radius * math.sin(2.0 * math.pi * longitude / longitude_segments),
                z,
            )
            for longitude in range(longitude_segments)
        )
    top_index = len(points)
    points.append(Gf.Vec3f(0.0, 0.0, 1.0))

    face_vertex_counts: list[int] = []
    face_vertex_indices: list[int] = []
    st_values: list[Gf.Vec2f] = []

    def ring_index(latitude: int, longitude: int) -> int:
        return 1 + (latitude - 1) * longitude_segments + longitude % longitude_segments

    first_ring_v = 1.0 / latitude_segments
    for longitude in range(longitude_segments):
        u0 = longitude / longitude_segments
        u1 = (longitude + 1) / longitude_segments
        face_vertex_counts.append(3)
        face_vertex_indices.extend(
            [0, ring_index(1, longitude + 1), ring_index(1, longitude)]
        )
        st_values.extend(
            [
                Gf.Vec2f((u0 + u1) / 2.0, 0.0),
                Gf.Vec2f(u1, first_ring_v),
                Gf.Vec2f(u0, first_ring_v),
            ]
        )

    for latitude in range(1, latitude_segments - 1):
        v0 = latitude / latitude_segments
        v1 = (latitude + 1) / latitude_segments
        for longitude in range(longitude_segments):
            u0 = longitude / longitude_segments
            u1 = (longitude + 1) / longitude_segments
            face_vertex_counts.append(4)
            face_vertex_indices.extend(
                [
                    ring_index(latitude, longitude),
                    ring_index(latitude, longitude + 1),
                    ring_index(latitude + 1, longitude + 1),
                    ring_index(latitude + 1, longitude),
                ]
            )
            st_values.extend(
                [
                    Gf.Vec2f(u0, v0),
                    Gf.Vec2f(u1, v0),
                    Gf.Vec2f(u1, v1),
                    Gf.Vec2f(u0, v1),
                ]
            )

    last_ring_v = (latitude_segments - 1) / latitude_segments
    for longitude in range(longitude_segments):
        u0 = longitude / longitude_segments
        u1 = (longitude + 1) / longitude_segments
        face_vertex_counts.append(3)
        face_vertex_indices.extend(
            [
                ring_index(latitude_segments - 1, longitude),
                ring_index(latitude_segments - 1, longitude + 1),
                top_index,
            ]
        )
        st_values.extend(
            [
                Gf.Vec2f(u0, last_ring_v),
                Gf.Vec2f(u1, last_ring_v),
                Gf.Vec2f((u0 + u1) / 2.0, 1.0),
            ]
        )

    sphere.CreatePointsAttr(points)
    sphere.CreateFaceVertexCountsAttr(face_vertex_counts)
    sphere.CreateFaceVertexIndicesAttr(face_vertex_indices)
    sphere.CreateNormalsAttr(points)
    sphere.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    sphere.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdGeom.PrimvarsAPI(sphere).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    ).Set(st_values)
    sphere.CreateExtentAttr(
        [
            Gf.Vec3f(-1.0, -1.0, -1.0),
            Gf.Vec3f(1.0, 1.0, 1.0),
        ]
    )

    material = UsdShade.Material.Get(stage, material_binding)
    if not material or not material.GetPrim().IsValid():
        raise ValueError(
            f"winning material binding is missing from its library: {material_binding}"
        )
    if not UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material):
        raise RuntimeError("failed to bind winning material to canonical swatch")

    stage.GetRootLayer().Save()
    return output_path


def render_material_swatch(
    *,
    material_usd_path: Path,
    material_binding: str,
    evidence_dir: Path,
    settings: MaterialRefinementRenderSettings,
    rendering_backend: Any,
    render_task: RenderTaskLike | None = None,
) -> MaterialRenderEvidence:
    """Render one sweep winner through the existing Material Agent RenderTask."""

    evidence_dir.mkdir(parents=True, exist_ok=True)
    swatch_path = author_material_swatch(
        material_usd_path=material_usd_path,
        material_binding=material_binding,
        output_path=evidence_dir / "material_swatch.usda",
    )
    render_config = settings.to_task_config()
    render_config["prim_path"] = _SWATCH_PRIM_PATH
    context = {
        "output_usd_path": swatch_path,
        "output_base_path": evidence_dir,
        "flatten_before_render": True,
        "render_enabled": True,
        "render_config": render_config,
        "rendering_backend": rendering_backend,
    }
    if render_task is None:
        from material_agent.tasks.render import RenderTask

        render_task = RenderTask()
    rendered = render_task.run(context)
    raw_paths = rendered.get("rendered_image_paths", [])
    rendered_paths = tuple(Path(path) for path in raw_paths)
    if rendered.get("rendering_skipped") or not rendered_paths:
        raise RuntimeError("canonical material swatch rendering produced no evidence")
    missing = [path.name for path in rendered_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "canonical material swatch render paths are missing: "
            + ", ".join(sorted(missing))
        )

    raw_flattened = rendered.get("flattened_usd_path")
    raw_validation = rendered.get("render_validation", [])
    validation = (
        tuple(dict(item) for item in raw_validation)
        if isinstance(raw_validation, list)
        else ()
    )
    rendering_stats = dict(rendered.get("rendering_stats", {}))
    return MaterialRenderEvidence(
        swatch_usd_path=swatch_path,
        flattened_usd_path=Path(raw_flattened) if raw_flattened else None,
        rendered_image_paths=rendered_paths,
        backend=str(rendering_stats.get("backend", settings.backend)),
        rendering_stats=rendering_stats,
        render_validation=validation,
    )


def _contact_sheet(paths: tuple[Path, ...], destination: Path) -> Path:
    """Pack bounded visual context into one deterministic RGB image."""

    cell_size = 256
    columns = min(4, max(1, math.ceil(math.sqrt(len(paths)))))
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * cell_size, rows * cell_size), (32, 32, 32))
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            cell = image.convert("RGB")
            cell.thumbnail((cell_size, cell_size))
        x = (index % columns) * cell_size + (cell_size - cell.width) // 2
        y = (index // columns) * cell_size + (cell_size - cell.height) // 2
        sheet.paste(cell, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return destination


def judge_rendered_material(
    *,
    target: MaterialRefinementGoal,
    render_evidence: MaterialRenderEvidence,
    settings: MaterialRefinementJudgeSettings,
    vlm_judge: VlmJudgeLike,
    iteration: int,
    previous_feedback: str | None,
    comparison_image_paths: tuple[Path, ...] = (),
    material_profile: str = "unspecified",
    source_representation: str = "scalar_pbr",
    controllable_properties: tuple[str, ...] = (
        "base_color",
        "roughness",
        "metallic",
    ),
) -> MaterialJudgeVerdict:
    """Ask the existing VLM abstraction to judge rendered target evidence."""

    image_pairs: list[tuple[str, Path]] = []
    image_pairs.extend(
        (f"Rendered Material Swatch - View {index}:", path)
        for index, path in enumerate(render_evidence.rendered_image_paths, start=1)
    )
    for path in comparison_image_paths:
        if not path.is_file():
            raise FileNotFoundError("approved comparison render is unavailable")
    if comparison_image_paths:
        image_pairs.append(
            (
                f"Previously Approved Variations ({len(comparison_image_paths)}):",
                _contact_sheet(
                    comparison_image_paths,
                    render_evidence.swatch_usd_path.parent
                    / "approved_variations_contact_sheet.png",
                ),
            )
        )
    if not image_pairs:
        raise ValueError("rendered material evidence is required for visual judgment")

    prompt = _TARGET_JUDGE_PROMPT.format(
        appearance_prompt=target.appearance_prompt,
        material_profile=material_profile,
        source_representation=source_representation,
        controllable_properties=", ".join(controllable_properties),
        comparison_context=(
            "Reject near-duplicates of these approved variations."
            if comparison_image_paths
            else "(No earlier variation in this set.)"
        ),
        approval_score=settings.score_threshold * 10.0,
    )
    raw_response = vlm_judge.generate_with_image_caption_pairs(
        image_caption_pairs=[
            (caption, path.as_posix()) for caption, path in image_pairs
        ],
        final_prompt=prompt,
        system_prompt=_TARGET_JUDGE_SYSTEM_PROMPT,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("VLM judge returned an empty material verdict")
    safe_response = str(_redacted(raw_response))
    parsed_score = extract_labeled_score(safe_response)
    score = parsed_score if parsed_score is not None else 0.0
    parsed_decision = extract_labeled_choice(
        safe_response,
        "Decision",
        ("continue", "approve"),
        boundary_labels=(
            "Critique",
            "Improvement Suggestion",
            "Improvement Suggestions",
            "Score",
        ),
    )
    decision_parsed = bool(parsed_decision)
    decision = parsed_decision or "continue"
    if score < settings.score_threshold:
        decision = "continue"
    approved = decision_parsed and decision == "approve"
    reasoning = " ".join(safe_response.splitlines()[:3]).strip()
    if len(reasoning) > 200:
        reasoning = reasoning[:197] + "..."
    return MaterialJudgeVerdict(
        score=score,
        decision=decision,
        decision_parsed=decision_parsed,
        approved=approved,
        reasoning=reasoning,
        feedback=safe_response,
        raw_response=safe_response,
        prompt=prompt,
        system_prompt=_TARGET_JUDGE_SYSTEM_PROMPT,
        image_caption_pairs=tuple(image_pairs),
        provider=settings.provider_evidence(),
    )


__all__ = [
    "MaterialJudgeVerdict",
    "MaterialRenderEvidence",
    "MaterialTargetInference",
    "RenderTaskLike",
    "VlmJudgeLike",
    "author_material_swatch",
    "judge_rendered_material",
    "infer_material_target",
    "provision_rendering_backend",
    "provision_vlm_judge",
    "render_material_swatch",
]
