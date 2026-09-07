# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material-inference prompt contract.

The apply prompt's patch schema carried a worked example whose density (1150)
is a plastic value. Across a 16-asset PhysX-Mobility suite the child copied the
plastic reading even on parts it could see were metal -- a textured render of a
silver camera body was still classified `plastic` -- and metal->plastic was 73%
of all material errors. The same anchoring mechanism previously pinned
restitution at ~0.05 until the example value was removed.
"""

from pathlib import Path

from content_workflow_cli.prompts import build_physics_apply_prompt


def _prompt(tmp_path: Path, **overrides: object) -> str:
    kwargs: dict[str, object] = {
        "repo_root": tmp_path,
        "run_dir": tmp_path / "run",
        "usd_path": tmp_path / "asset.usd",
    }
    kwargs.update(overrides)
    return build_physics_apply_prompt(**kwargs)  # type: ignore[arg-type]


def test_schema_example_carries_no_material_specific_density(tmp_path: Path) -> None:
    """A concrete density in the example is a plastic anchor, not a shape hint.

    The anchor must not survive anywhere in the prompt: an earlier revision
    removed `1150.0` from the example but kept prose citing "the example's
    1150 is a plastic value", which reintroduced the exact number this test
    exists to keep out.
    """

    prompt = _prompt(tmp_path)

    assert "1150" not in prompt
    assert "hard-plastic" not in prompt
    assert "density" in prompt


def test_prompt_blocks_the_housing_defaults_to_plastic_prior(tmp_path: Path) -> None:
    """The dominant error class: appliance/device shells read as plastic."""

    prompt = _prompt(
        tmp_path,
        classification_view_labels=["Workbench initial render: classification_top"],
    )

    assert "Do not default a device or appliance housing to plastic" in prompt
    # It must say what to look for, not merely forbid the shortcut.
    assert "specular" in prompt
    assert "brushed" in prompt


def test_prompt_housing_prior_survives_without_views(tmp_path: Path) -> None:
    """On the render-outage/dry-run path no image exists, so the metal-vs-
    plastic guidance must point at names/geometry instead of ordering the
    child to inspect views it was never given -- that invites fabricated view
    citations in the rationale."""

    prompt = _prompt(tmp_path, classification_view_labels=[])

    assert "Do not default a device or appliance housing to plastic" in prompt
    assert "attached views" not in prompt
    assert "no rendered" in prompt and "views are attached to this turn" in prompt


def test_prompt_scales_mass_by_canonical_fill_fractions(tmp_path: Path) -> None:
    """`bounds_m.volume_m3` is a bounding-box volume, so density x volume
    assumes a solid part and overweights a thin glass shell ~25x. The prompt
    must carry the canonical per-family fill fractions from
    ``content_agent_workflows.physics.policy`` -- one source of truth, not a
    diverging copy."""

    from content_agent_workflows.physics import material_volume_fractions

    prompt = _prompt(tmp_path)

    assert "axis-aligned bounding-box volume" in prompt
    for family, fraction in material_volume_fractions().items():
        assert f"{family} {fraction:g}" in prompt


def test_prompt_exempts_unowned_static_fixtures_from_mass(tmp_path: Path) -> None:
    """The canonical policy zeroes density and mass for `unowned_static`
    fixtures; without the exception the fill-fraction formula would direct the
    child to author a large floor's mass onto the simulated body."""

    prompt = _prompt(tmp_path)

    assert "unowned_static" in prompt
    assert "takes no mass" in prompt


def test_prompt_states_component_name_priors(tmp_path: Path) -> None:
    """Glass scored 0-for-21 while parts named `mirror` and `lens` existed."""

    prompt = _prompt(tmp_path)

    for token in ("lens", "mirror", "blade", "gasket"):
        assert token in prompt


def test_prompt_requires_evidence_cited_for_material(tmp_path: Path) -> None:
    """Nothing forced the child to consult the views it was handed; an
    unfalsifiable rationale also makes every future failure undiagnosable."""

    prompt = _prompt(
        tmp_path,
        classification_view_labels=["Workbench initial render: classification_top"],
    )

    assert "cite the evidence" in prompt
    assert "name the view" in prompt
    assert "Inferred from context" in prompt


def test_prompt_evidence_citation_never_names_absent_views(tmp_path: Path) -> None:
    """Without attached views the citation requirement must fall back to
    name/JSON evidence; requiring a view citation when no view exists forces
    the child to fabricate one."""

    prompt = _prompt(tmp_path, classification_view_labels=[])

    assert "cite the evidence" in prompt
    assert "name the view" not in prompt
    assert "JSON evidence field" in prompt
    assert "Inferred from context" in prompt
