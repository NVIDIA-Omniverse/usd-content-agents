# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restitution priors in ``content_agent_workflows.physics.policy``.

The deterministic fallback profiles run when no agent patch is supplied, so
they must not reintroduce the dead-body restitution values the apply prompt
now argues against (see ``test_physics_apply_prompt_restitution.py`` for the
prompt-side coverage and the benchmark evidence).
"""

from content_agent_workflows.physics.policy import infer_material_profile

# Reference coefficients of restitution for an impact against a hard surface.
# Ordinary rigid materials rebound; only genuinely inelastic parts do not.
RIGID_FAMILY_FLOOR = 0.3


def test_fallback_profiles_rebound_for_rigid_materials() -> None:
    """The deterministic fallback runs when no agent patch is supplied, so it
    must not reintroduce the dead-body values the prompt now avoids."""

    for name in ("metal", "plastic", "glass", "wood", "rubber"):
        profile = infer_material_profile(name)
        assert profile.restitution >= RIGID_FAMILY_FLOOR, (
            f"{name} restitution {profile.restitution} is inelastic"
        )


def test_rubber_still_rebounds_more_than_wood() -> None:
    """Raising the floor must not flatten the families into one value -- the
    ordering is the physical content."""

    assert (
        infer_material_profile("rubber").restitution
        > infer_material_profile("wood").restitution
    )


def test_soft_material_labels_stay_inelastic() -> None:
    """Foam and fabric are the legitimate near-zero-restitution materials the
    prompt calls out; raising the generic fallback must not make them bounce."""

    for label in ("foam pad", "fabric cover", "cloth", "sponge", "seat cushion"):
        profile = infer_material_profile(label)
        assert profile.family == "soft", label
        assert profile.restitution < 0.1, label


def test_unknown_labels_still_get_the_generic_rigid_fallback() -> None:
    assert infer_material_profile("mystery widget").family == "generic"


def test_soft_tokens_match_whole_words_only() -> None:
    """ "cloth" must not classify clothes_rack or clothing_hook — rigid
    furniture — as a 150 kg/m^3 dead-rebound softbody."""

    for label in ("clothes_rack", "clothing hook", "tablecloth holder"):
        assert infer_material_profile(label).family != "soft", label
    assert infer_material_profile("cloth napkin").family == "soft"


def test_explicit_rigid_cues_take_precedence_over_soft_tokens() -> None:
    """A compound rigid name carrying a soft word — "metal pillow block", a
    standard bearing housing — must keep its rigid profile instead of
    becoming a 150 kg/m^3 dead-rebound softbody."""

    assert infer_material_profile("metal pillow block").family == "metal"
    assert infer_material_profile("plastic cushion clip").family == "plastic"


def test_camel_case_soft_names_are_recognized() -> None:
    """Common concatenated USD names must still reach the soft profile."""

    for label in ("FoamPad", "FabricCover", "SeatCushion"):
        assert infer_material_profile(label).family == "soft", label


def test_soft_aliases_and_plurals_are_recognized() -> None:
    """Known soft labels beyond the base tokens — including simple plurals
    and camel-case compounds — must not fall to the elastic generic
    profile."""

    for label in ("cotton", "polyester", "denim", "styrofoam", "SeatCushions"):
        assert infer_material_profile(label).family == "soft", label
