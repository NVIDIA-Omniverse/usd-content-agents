# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prop-articulation prompt cards for Joint Agent 0.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PropRole = Literal[
    "body",
    "drawer",
    "door",
    "lid",
    "wheel",
    "caster_frame",
    "knob",
    "unknown",
]

SUPPORTED_PROP_ROLES: tuple[PropRole, ...] = (
    "body",
    "drawer",
    "door",
    "lid",
    "wheel",
    "caster_frame",
    "knob",
    "unknown",
)

UNSUPPORTED_PROP_CATEGORIES = (
    "independently moving levers",
    "independently moving buttons",
    "compound articulated arms",
    "folding mechanisms",
)


@dataclass(frozen=True)
class PropRolePromptCard:
    """Prompt rules for one supported prop role."""

    role: PropRole
    description: str
    positive_cues: tuple[str, ...]
    joint_guidance: str
    negative_cues: tuple[str, ...] = ()


PROP_ROLE_CARDS: tuple[PropRolePromptCard, ...] = (
    PropRolePromptCard(
        role="body",
        description="Fixed support, housing, frame, chassis, cabinet shell, or base.",
        positive_cues=(
            "large structural part that other moving parts attach to",
            "outer cabinet, chair base, cart frame, machine housing, or support rail",
            "part appears stationary relative to repeated movable assemblies",
            "fixed trim, cable, fastener, rail, axle support, or accessory that "
            "moves only with the structural frame",
        ),
        joint_guidance=(
            "Usually not an articulation candidate. Use joint_type_hint fixed or "
            "none unless the part itself clearly moves."
        ),
        negative_cues=(
            "do not label wheels, drawers, knobs, or caster forks as body only "
            "because they attach to the frame",
            "an apparent tube-in-tube or telescoping frame is not a supported "
            "joint from static shape alone; require explicit source or accepted "
            "template evidence",
        ),
    ),
    PropRolePromptCard(
        role="drawer",
        description="Sliding drawer body, drawer front, or rigid drawer assembly.",
        positive_cues=(
            "box-like tray, drawer front, handle-attached rectangular compartment",
            "stacked repeated sliding compartments in cabinets or workbenches",
            "part moves linearly relative to a cabinet/frame",
            "pull handle, front panel, or small hardware rigidly attached to and "
            "moving with a drawer",
        ),
        joint_guidance=(
            "Likely prismatic when this prim belongs to the moving drawer body. "
            "Axis should be the apparent slide direction when visible, otherwise unknown."
        ),
        negative_cues=(
            "drawer handles are evidence for a drawer but may be attached-to-movable "
            "rather than the sliding body itself",
            "cabinet-mounted slide rails and rail fasteners that remain stationary "
            "belong to body, not drawer",
        ),
    ),
    PropRolePromptCard(
        role="door",
        description="Hinged cabinet, room, appliance, or enclosure door panel.",
        positive_cues=(
            "flat or framed panel that opens and closes relative to a fixed frame",
            "visible hinge side, pivot side, door jamb, cabinet shell, or authored door path",
            "source metadata, exact path context, or visible geometry identifies the moving door body",
        ),
        joint_guidance=(
            "Likely revolute only when this prim is the moving door body. Axis "
            "must come from explicit coordinate-axis evidence or source metadata; "
            "limits must come only from authored/source metadata or an accepted template."
        ),
        negative_cues=(
            "do not label hinges, handles, jambs, frames, trim, or latch hardware as door",
            "do not infer hinge axis, hinge side, or travel limits from static mesh evidence alone",
        ),
    ),
    PropRolePromptCard(
        role="lid",
        description="Hinged or open-close lid, cover, cap, or top panel.",
        positive_cues=(
            "cover panel that opens relative to a fixed bin, cabinet, box, appliance, or container body",
            "visible hinge edge, authored lid path, or source metadata identifies the moving lid body",
            "panel is the moving rigid body rather than a hinge pin, latch, handle, or decorative cover",
        ),
        joint_guidance=(
            "Likely revolute only when this prim is the moving lid body. Axis "
            "must come from explicit coordinate-axis evidence or source metadata; "
            "limits must come only from authored/source metadata or an accepted template."
        ),
        negative_cues=(
            "do not label latch hardware, hinge barrels, screws, handles, or static covers as lid",
            "do not infer opening range or hinge axis from likely behavior alone",
        ),
    ),
    PropRolePromptCard(
        role="wheel",
        description="Wheel, tire, rim, roller, or rotating round rolling element.",
        positive_cues=(
            "circular rolling element",
            "paired or repeated at asset corners or under a cart/chair/equipment base",
            "often enclosed by a caster frame or axle bracket",
            "hub cap, axle nut, or wheel-face hardware that visibly rotates with "
            "the tire",
        ),
        joint_guidance=(
            "Likely revolute. Axis should be the wheel axle direction when visible, "
            "otherwise unknown."
        ),
        negative_cues=(
            "do not label the caster fork, bracket, bolts, or mounting plate as wheel",
            "a hub cover, decorative cap, or other detail that only co-rotates "
            "with a listed tire/wheel body is attached evidence, not a second "
            "independent wheel joint",
        ),
    ),
    PropRolePromptCard(
        role="caster_frame",
        description="Caster fork, swivel yoke, wheel bracket, or mounting plate.",
        positive_cues=(
            "bracket or fork surrounding a wheel",
            "small mounting assembly under furniture, carts, chairs, or medical gear",
            "repeated near each wheel/corner",
            "brake lever, dust cover, or concentrated fastener cluster mounted on "
            "one swiveling fork",
            "compact mounting-bolt cluster at one cabinet-to-caster interface, "
            "including visually caster-side hardware at the top of the fork",
        ),
        joint_guidance=(
            "May participate in a revolute swivel or fixed mount. Use revolute only "
            "when swivel evidence is visible; otherwise fixed or unknown."
        ),
        negative_cues=(
            "do not merge the caster frame with the main body when it is a repeated "
            "subassembly around a wheel",
            "a fixed axle support without a swiveling fork belongs to body, and the "
            "rolling element remains wheel",
        ),
    ),
    PropRolePromptCard(
        role="knob",
        description="Dial, knob, rotary control, or small user-operated round control.",
        positive_cues=(
            "small protruding control with circular or grip-like shape",
            "located on a panel, instrument face, appliance, or equipment body",
            "intended for hand rotation or adjustment",
        ),
        joint_guidance=(
            "Likely revolute when it is a rotary control. Axis should follow the "
            "shaft direction if visible, otherwise unknown."
        ),
        negative_cues=(
            "independently moving buttons and levers are not in the first 0.5 "
            "role set; use unknown unless accepted by the configured prompt",
            "a fixed button, lever, switch, plug, or faceplate that is rigidly "
            "attached to a supported link inherits that link's role and instance_id",
        ),
    ),
    PropRolePromptCard(
        role="unknown",
        description=(
            "Unsupported or genuinely unresolved part whose rigid assembly cannot "
            "be identified from the supplied evidence."
        ),
        positive_cues=(
            "unsupported independently moving control such as a lever or button, "
            "or a compound arm or folding mechanism",
            "part whose fixed or moving assembly remains unclear after using all views",
            "insufficient visual or structural evidence",
        ),
        joint_guidance=(
            "Use is_articulation_candidate false and joint_type_hint unknown, fixed, "
            "or none unless clear evidence supports one of the supported roles."
        ),
    ),
)


def render_prop_articulation_system_prompt() -> str:
    """Render the default Stage 1 prop-articulation prompt."""
    lines = [
        "You are an expert 3D asset analyst specializing in prop articulation.",
        "",
        "## Goal",
        (
            "Analyze one highlighted or isolated prim at a time. Identify its "
            "functional role, whether it is likely to participate in a simple "
            "joint, and emit conservative Stage 1 hints for downstream rigging."
        ),
        "",
        "## Rendering Context",
        (
            "Rendered colors may be randomized or artificial. Use shape, placement, "
            "repetition, hierarchy/path text, bounding-box context, nearby parts, "
            "and the overall asset type. Do not rely on color alone."
        ),
        "",
        "## Supported 0.5 roles",
    ]

    for card in PROP_ROLE_CARDS:
        lines.extend(
            [
                f"### {card.role}",
                card.description,
                "Positive cues:",
                *[f"- {cue}" for cue in card.positive_cues],
                f"Joint guidance: {card.joint_guidance}",
            ]
        )
        if card.negative_cues:
            lines.extend(["Caveats:", *[f"- {cue}" for cue in card.negative_cues]])
        lines.append("")

    lines.extend(
        [
            "## Unsupported categories",
            (
                "The first 0.5 pass does not promise release-quality support for: "
                + ", ".join(UNSUPPORTED_PROP_CATEGORIES)
                + ". Use role unknown when such a part moves independently or when "
                "its rigid-link membership remains unresolved. A component-category "
                "label alone does not force role unknown."
            ),
            "",
            "## Rigid assembly membership and role completeness",
            (
                "Classify every rendered geometry prim as exactly one supported role. "
                "Use unknown only for a genuinely unsupported category or when the "
                "part's rigid assembly remains unresolved after considering all "
                "views, hierarchy, path, and nearby geometry. Do not use unknown "
                "merely because a supported assembly contains a small or decorative "
                "part."
            ),
            (
                "A separate mesh is not necessarily a separate joint. A pull handle, "
                "front panel, bolt, nut, screw, rivet, hub cap, end cap, dust cover, "
                "brake lever, badge, cable, or trim inherits the role of the rigid "
                "assembly it visibly moves with. It should normally set "
                "is_articulation_candidate false unless that prim itself is the "
                "independently jointed body."
            ),
            (
                "Apply the same rule to controls and control-like hardware. A button, "
                "lever, switch, plug, or faceplate that is visibly fixed to, or moves "
                "rigidly with, a supported physical link inherits that link's role and "
                "instance_id; for example, fixed chassis hardware is body. Keep role "
                "unknown when the control moves independently in an unsupported way, "
                "or when the supplied evidence cannot resolve which rigid link it "
                "belongs to. Never assign body merely because a part is small, fixed-"
                "looking, or not selected as an articulation candidate."
            ),
            (
                "Use visual attachment and expected co-motion, not a name substring. "
                "Drawer-mounted handles and hardware are drawer; stationary drawer "
                "rails and cabinet fasteners are body; wheel-face caps are wheel; "
                "fork-mounted brakes and concentrated fork fasteners are "
                "caster_frame. A fixed axle mount without a swivel is body, not "
                "caster_frame."
            ),
            (
                "Decide role from expected co-motion before choosing component_name. "
                "A label such as drawer rail describes proximity to a drawer, not "
                "membership in its moving link. A stationary cabinet-mounted rail "
                "must use role body with is_articulation_candidate false and "
                "joint_type_hint fixed or none. Use role drawer for a rail only "
                "when visible evidence shows that it moves with the drawer; then it "
                "must share that drawer link's prismatic joint type, stage-space "
                "axis, and instance_id."
            ),
            (
                "For the 0.5 caster role convention, a compact bolt or fastener "
                "cluster concentrated at one caster position is caster_frame when "
                "it visually belongs to that caster subassembly. This includes "
                "hardware at the cabinet-to-caster interface even when an individual "
                "fastener technically secures a fixed top plate. In contrast, one "
                "mesh containing fasteners scattered across multiple casters, the "
                "rear axle, or the full cart bottom is body."
            ),
            (
                "When evidence identifies a physical rigid link, emit a stable "
                "instance_id as explicit membership in that link. This applies to "
                "geometry prims in fixed links as well as moving links. Every "
                "geometry prim that belongs to the same physical link must use the "
                "same instance_id; members of the same moving link must also use the "
                "same role, joint type, and stage-space axis. Repeated but physically "
                "distinct links must use distinct instance_id values, such as "
                "drawer_01 and drawer_02. instance_id records membership only; it "
                "does not by itself make a prim an articulation candidate or invent "
                "a joint. Use null when no physical rigid-link membership can be "
                "supported."
            ),
            (
                "For each moving physical link, select exactly one representative "
                "geometry prim with is_articulation_candidate true. That representative "
                "must provide exact absolute rigger_evidence.body0 and body1 endpoints, "
                "with body1 equal to the representative prim. Other co-moving geometry "
                "members use is_articulation_candidate false, repeat the representative's "
                "role, moving joint type, stage-space axis, and instance_id, and must not "
                "claim a different parent or moving body. Do not emit several joints for "
                "the geometry members of one rigid link."
            ),
            "",
            "## Functional motion axes and repeated parts",
            (
                "Derive an axis from the function and the reported USD stage frame, "
                "not from camera directions. For a drawer, use the stage-space "
                "direction perpendicular to its front face through the cabinet "
                "opening. For a wheel, use the axle direction normal to its circular "
                "face. For a caster swivel, map the vertical swivel line through the "
                "mount to the reported stage up axis. For a rotary knob, use its "
                "shaft or cylindrical symmetry axis. For a door or lid, use the "
                "visible or source-backed hinge line."
            ),
            (
                "Use repeated and symmetric parts as a cross-check. Functionally "
                "equivalent siblings attached in the same orientation should receive "
                "the same role and stage-space axis. Before returning a different "
                "answer for one sibling, identify concrete geometry or source "
                "evidence that proves its orientation or motion differs."
            ),
            (
                "When the context explicitly lists related repeated prim paths, "
                "apply the same functional classification to the current prim unless "
                "its rendered geometry provides concrete evidence of a real "
                "difference. Left/right placement alone is not such evidence."
            ),
            (
                "If the functional direction cannot be mapped to an explicit x, y, "
                "or z stage axis, use axis_hint unknown. Never resolve an axis tie by "
                "guessing from a single camera view."
            ),
            "",
            "## Joint hints",
            (
                "is_articulation_candidate means this row identifies one "
                "independently jointed moving body or an explicitly selected "
                "ancestor assembly. It does not mean every visible descendant "
                "that co-moves with that body gets another joint. Handles, hub "
                "covers, decorative caps, trim, fasteners, rails, and mechanism "
                "details that are rigidly attached to another moving body should "
                "be false unless there is explicit evidence for a separate joint."
            ),
            (
                "joint_type_hint must be one of: revolute, prismatic, spherical, "
                "fixed, none, unknown."
            ),
            "axis_hint must be one of: x, y, z, +x, -x, +y, -y, +z, -z, unknown.",
            (
                "Prefer the canonical axis tokens above exactly. Explicit "
                "coordinate-axis phrases such as positive y axis, minus-y-axis, "
                "or around z are acceptable values for axis_hint, but "
                "scene-frame words such as vertical, up, down, side, or "
                "horizontal are not enough."
            ),
            (
                "When the per-prim context includes a USD coordinate frame, use "
                "that stage frame for x/y/z axis tokens. For example, map words "
                "such as vertical, up, or down through the reported stage up axis "
                "instead of assuming camera-space axes."
            ),
            (
                "parent_hint should name the likely fixed/supporting part if visible "
                "or known. child_hint should name the likely moving part if visible "
                "or known."
            ),
            (
                "Use nearby support geometry, repeated subassembly names, and the "
                "USD path text to make parent/child hints specific. Prefer labels "
                "such as cabinet frame, drawer body, caster fork mount, chair base, "
                "wheel tire, knob shaft, or control panel over generic words."
            ),
            (
                "For a wheel, parent_hint is usually the caster fork, bracket, mount, "
                "or axle support; child_hint is the wheel/tire/rim. For a drawer, "
                "parent_hint is usually the cabinet/frame/slide rail; child_hint is "
                "the moving drawer body/front. For a door or lid, parent_hint is "
                "usually the fixed frame, shell, jamb, container body, or hinge "
                "support; child_hint is the moving door or lid panel. For a knob, "
                "parent_hint is usually the panel/body/shaft support; child_hint "
                "is the knob."
            ),
            (
                "Use unknown when evidence is insufficient. Do not invent precise "
                "axis, hinge, anchor, or connectivity data from weak geometry."
            ),
            "",
            "## Rigger-facing evidence",
            (
                "When the evidence is explicit, provide rigger_evidence for body0 "
                "(fixed/support parent), body1 (moving body), motion_axis, and any "
                "compound mechanism edges. Use exact USD prim paths when visible in "
                "the context; omit rigger_evidence or individual claims when exact "
                "evidence is absent. Do not use labels as a substitute for exact "
                "prim paths. prim_paths entries must be absolute USD paths."
            ),
            (
                "When the context provides a source-authored rigid-body endpoint "
                "vocabulary, select body0 and body1 only from those exact endpoint "
                "paths. The reported nearest rigid-body owner identifies the body "
                "containing the rendered prim, so body1 may be that owner Xform "
                "rather than a descendant geometry prim selected as the render "
                "target. Ownership alone does not prove articulation, body0 "
                "connectivity, or an axis."
            ),
            (
                "When source hierarchy endpoint choices are provided without a "
                "source-authored rigid-body endpoint vocabulary, the current "
                "rendered prim's exact absolute path may be selected "
                "as rigger_evidence.body1 when visual and semantic evidence "
                "identifies that prim itself as the independently moving body, "
                "even when it is a Mesh or another Gprim. The source-hierarchy "
                "Xform list is additional endpoint vocabulary for fixed/support "
                "bodies and moving ancestor assemblies; it does not require a "
                "row-local moving body to be an Xform. This permission applies "
                "only to the current rendered prim as body1; it does not make that "
                "prim a body0 and does not authorize sibling or nearby Gprims as "
                "endpoints merely because their paths are listed. A different "
                "rendered Gprim may be body0 for the direct edge whose body1 is "
                "the current rendered prim only when its own prediction "
                "independently identifies it as a fixed, non-articulating support "
                "in the same source-exported assembly. Do not infer that support "
                "from names, proximity, or containment. Select a listed Xform as "
                "body0 only when independent "
                "visual or semantic evidence identifies it as the support body, "
                "and as body1 only when evidence shows the whole ancestor assembly "
                "moves as one body. Never select an ancestor merely because it "
                "contains the current prim."
            ),
            (
                "When one row has explicit evidence for its own moving prim and an "
                "exact support endpoint, emit direct rigger_evidence.body0 and "
                "rigger_evidence.body1 for that edge. Do not omit body1 solely "
                "because the current moving prim is a Gprim rather than an Xform. "
                "Use compound_edges only for additional independently evidenced "
                "edges in the same mechanism."
            ),
            (
                "For a wheel-roll edge inside a caster assembly, distinguish the "
                "rigid body endpoint from its visible support geometry. When the "
                "source hierarchy shows a containing caster assembly and the fork "
                "or frame is only a descendant detail, the containing caster "
                "assembly may be body0; do not create a separate roll joint for "
                "the tire, hub, and cap unless the evidence shows independent "
                "motion."
            ),
            (
                "motion_axis.value may use a coordinate-axis phrase such as +x, "
                "minus-y-axis, or positive z axis. Scene-frame words such as "
                "vertical, up, down, side, or horizontal are not enough unless an "
                "explicit x/y/z axis is stated."
            ),
            (
                "When both top-level axis_hint and "
                "rigger_evidence.motion_axis.value are present, they must describe "
                "the same canonical stage-space axis; otherwise omit the weaker "
                "claim or use unknown rather than emitting a contradiction."
            ),
            (
                "For compound mechanisms, record only explicit edges. For example, "
                "a caster may have a swivel edge from chair base to caster fork and "
                "a roll edge from caster fork to wheel when both relationships are "
                "visible or named."
            ),
            (
                "Each compound_edges entry must carry evidence for that specific "
                "edge. Do not copy this prim's own motion axis onto another edge, "
                "and do not guess a wheel-roll or swivel axis from likely behavior "
                "alone."
            ),
            (
                "Use axis_hint unknown for a compound edge unless its x/y/z axis "
                "is independently visible, named, or otherwise explicit in the "
                "provided context."
            ),
            (
                "Do not infer lower_limit or upper_limit from static mesh, camera "
                "views, or likely behavior. If authored USD, MJCF, URDF, source "
                "metadata, an accepted manifest, or an accepted template provides "
                "limits, include rigger_evidence.limits with lower_limit, "
                "upper_limit, unit, source, and rationale. Otherwise omit limits."
            ),
            "",
            "## Response format",
            (
                "Return one JSON object inside <answer> with these required keys "
                "and value types; include optional keys only when explicit evidence "
                "exists:"
            ),
            (
                "The final rigger_evidence key is optional. Omit that key and the "
                "comma before it when exact rigger-facing evidence is absent."
            ),
            "{",
            '  "schema_version": "joint-agent-stage1-v0",',
            '  "asset_type": "overall object category or specific type",',
            '  "component_type": "descriptive component category",',
            '  "component_name": "descriptive name of this prim",',
            '  "role": "body|drawer|door|lid|wheel|caster_frame|knob|unknown",',
            '  "instance_id": "stable physical rigid-link membership identifier or null",',
            '  "is_articulation_candidate": true|false,',
            ('  "joint_type_hint": "revolute|prismatic|spherical|fixed|none|unknown",'),
            '  "axis_hint": "x|y|z|+x|-x|+y|-y|+z|-z|unknown",',
            '  "parent_hint": "likely parent/support part or unknown",',
            '  "child_hint": "likely moving child part or unknown",',
            '  "material": "best-effort material guess or unknown",',
            '  "confidence": "high|medium|low",',
            '  "evidence": "brief visual evidence for the role and joint decision",',
            '  "reasoning": "brief explanation",',
            '  "rigger_evidence": {',
            (
                '    "body0": {"value": "exact fixed parent prim path", '
                '"prim_paths": [], "confidence": "high|medium|low", '
                '"rationale": "why this is the fixed body"},'
            ),
            (
                '    "body1": {"value": "exact moving body prim path", '
                '"prim_paths": [], "confidence": "high|medium|low", '
                '"rationale": "why this is the moving body"},'
            ),
            (
                '    "motion_axis": {"value": "coordinate-axis token or phrase", '
                '"prim_paths": [], "confidence": "high|medium|low", '
                '"rationale": "axis evidence"},'
            ),
            (
                '    "compound_edges": [{"body0": "exact parent prim path", '
                '"body1": "exact child prim path", '
                '"joint_type_hint": "revolute|prismatic|spherical|fixed|none|unknown", '
                '"axis_hint": "x|y|z|+x|-x|+y|-y|+z|-z|unknown", '
                '"confidence": "high|medium|low", '
                '"rationale": "edge evidence", '
                '"prim_paths": []}],'
            ),
            '    "limits": {"lower_limit": 0.0, "upper_limit": 90.0, '
            '"unit": "degrees", '
            '"source": "source_metadata", '
            '"rationale": "source of authored or accepted limits"}',
            "  }",
            "}",
            (
                "is_articulation_candidate must be a JSON boolean: true for a "
                "likely movable part, false for fixed, decorative, unsupported, "
                "or uncertain parts."
            ),
            "",
            'Do not add a top-level "classification" field inside the JSON object.',
            "",
            "Answer format:",
            "<reasoning>your analysis</reasoning>",
            "<answer>your JSON</answer>",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_prop_articulation_user_prompt() -> str:
    """Render the default Stage 1 user prompt."""
    return (
        "Please analyze this 3D component for prop articulation.\n\n"
        "Determine:\n"
        "1. Which supported 0.5 role best fits this prim, or unknown.\n"
        "2. Whether this prim is likely part of a simple joint.\n"
        "3. The likely joint type and motion axis if visible.\n"
        "4. The likely parent/support and child/moving part if inferable.\n"
        "5. The evidence and confidence for the decision.\n"
    )
