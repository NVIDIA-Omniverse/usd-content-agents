# Asset Notices

This file lists every asset file bundled in this repository: USD, USDA, USDC
and USDZ layers, textures, images and animations, environment maps, material
archives, and binary test fixtures. For each it states the source and terms.
Third-party software and its license texts are listed separately in
[`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md).

Assets that come from other distributions keep their original terms; the
repository's [Apache License 2.0](LICENSE) does not relicense them. Asset files
are exempt from per-file SPDX headers, and this file states their source and
terms instead (some NVIDIA-authored USDA fixtures also carry SPDX headers).

## Creative Commons Attribution 4.0

The following are derived from the NVIDIA
[PhysicalAI-SimReady-Warehouse-01](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01)
dataset, Copyright (c) NVIDIA Corporation, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The material is
provided as is, without warranties.

| Asset | Source in the dataset | Changes |
| --- | --- | --- |
| `apps/usd_cli/sample_assets/simready/Spray_Bottle/` | `Props/general/HandManipulation/cleaning_bottle_spray_a` | Converted to OBJ and back to a layered USD payload; albedo texture unchanged, roughness map derived from the source maps, normal and ORM maps not included |
| `assets/images/simready_teaser_steel_rolling_scaffold.gif` | `SM_SteelRollingScaffold_A01_01` | Rendered animation after Content Agents material, texture, and physics passes |
| `apps/material_agent/data/examples/ladder/sources/usd/ladder.usd`, `apps/texture_agent/data/examples/ladder/sources/usd/ladder.usd`, `apps/texture_agent/data/examples/ladder/sources/usd/ladder_uv_ready.usd` | `Props/general/SM_AluminumStepLadder_B05_01` | Material copy: materials removed and layer documentation replaced. Texture copy: re-bound to this repository's OpenPBR library materials. `ladder_uv_ready.usd` only sublayers the texture copy; geometry unchanged |
| `apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg`, `ladder_reference_2.jpeg` | `Props/general/SM_AluminumStepLadder_B05_01` | OVRTX renders of the texture copy above (adaptations of the ladder model) |

The following are derived from NVIDIA Isaac Sim 6.0 SimReady assets, also
published in the NVIDIA
[simready-assets](https://huggingface.co/datasets/nvidia/simready-assets)
dataset, Copyright (c) NVIDIA Corporation, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) as stated by the
license that accompanies each asset. The material is provided as is, without
warranties.

| Asset | Source | Changes |
| --- | --- | --- |
| `apps/physics_agent/data/examples/Container_Gray_C04/` | `SimReady/Industrial/Warehouse/Containers/Container_Gray_C04` (`container_gray_c04`) | Deinstanced; lid and body merged into one rigid body with two convex-hull colliders; fixed joint removed; material bindings repaired; flattened to one self-contained USDC |
| `apps/physics_agent/data/examples/Tire_B01/` | `SimReady/Industrial/Machines/Automotive/Wheels/Tire_B01` (`wheelAssembly_tire_b01`) | Deinstanced and re-materialized by a Content Agents material pass; flattened to USDC; `t_rubber_new_a01_tile_*` textures downscaled from 4K to 1K |

## Apache License 2.0 (NVIDIA SimReady Foundation)

The following are derived from
[NVIDIA SimReady Foundation](https://github.com/NVIDIA/simready-foundation)
sample content, Copyright (c) NVIDIA Corporation, licensed under the Apache
License 2.0.

| Asset | Source | Changes |
| --- | --- | --- |
| `apps/physics_agent/data/examples/Lightbulb01/` | `sample_content/common_assets/props_general/obs_light_bulb_01` | Converted to OBJ and back to a layered USD payload and USDZ package; albedo texture unchanged, normal and ORM maps not included |
| `assets/images/simready_teaser_electricians_toolbox.gif` | `obs_electricians_large_tool_box_a01` | Rendered animation after Content Agents material, texture, and physics passes |
| `assets/images/simready_teaser_ur10.gif` | `robots_general/ur10` | Rendered animation after Content Agents material, texture, and physics passes |

The `agentic/examples/scene/material-pass` fetch scripts download pinned
SimReady Foundation assets, with that repository's license, at run time; they
are not bundled.

## Apache License 2.0 (TheRobotStudio SO-ARM100)

`assets/images/keyboard_typing_rl_polish_4frame_strip.png` shows renders of a
robot-learning replay in which an SO-101 robot arm types on this project's
keyboard asset. The arm is the
[SO-101](https://github.com/TheRobotStudio/SO-ARM100) design by TheRobotStudio,
licensed under the Apache License 2.0, as converted to USD in the NVIDIA Isaac
Sim 6.1 IsaacLab robot assets (`Isaac/IsaacLab/Robots/SO101`). The arm's model
files are not bundled; only the rendered image is.

## AI-generated images

| Asset | Origin | Status |
| --- | --- | --- |
| `assets/images/usd_content_agents_overview.png` | README overview illustration, generated with OpenAI GPT image generation through Codex under an NVIDIA account, from text prompts and keyboard renders produced by this project's workflow | No third-party images were used as inputs |

## MIT No Attribution (NVIDIA PhysicalAI-SimReady-Materials)

The default and example material libraries are derived from
[NVIDIA PhysicalAI-SimReady-Materials](https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials)
v0.1.0, Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES, licensed under
MIT-0, with additional materials authored by NVIDIA:

- `apps/material_agent/data/materials/material_libs_default/materials_libs_v2.usd`
- `apps/material_agent_service/materials/default/materials_libs_v2.usd`
- `apps/material_agent_service/materials/default/default_materials.zip`
- `apps/material_agent_service/examples/custom_materials/materials_libs.usda`
- `apps/material_agent_service/examples/custom_materials.zip`

## CC0 1.0 (Poly Haven)

`world_understanding/data/env_maps/studio.exr` and
`apps/usd_cli/src/usd_core/render/data/studio.exr` are
[Studio Small 08](https://polyhaven.com/a/studio_small_08) by Sergej
Majboroda, Poly Haven, dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). These copies are
the 2K image re-encoded as a DWAB-compressed EXR, obtained through the NVIDIA
`ovrtx` package's test data.

## NVIDIA-authored assets (Apache License 2.0)

The following were created by NVIDIA for this repository and are licensed under
the repository's [Apache License 2.0](LICENSE), Copyright (c) NVIDIA
Corporation.

| Asset | Origin |
| --- | --- |
| `apps/usd_cli/sample_assets/simready/Cube/` (5 layers) | Generated by `apps/usd_cli/scripts/generate_sample_cube.py` |
| `agentic/examples/articulation/drawer/drawer.usda` | Hand-written example scene |
| `agentic/examples/geometry/quickstart/smoke_bracket.usda` | Hand-written example scene |
| `agentic/examples/mesh-segmentation/basic/fused_cart.usda` | Hand-written example scene |
| `agentic/examples/scene/material-pass/mini_workcell.usda` | Hand-written example scene; references SimReady Foundation assets that its fetch scripts download, not bundled |
| `apps/joint_agent_service/tests/test_data/joint_rigger_drawer_stack.usda`, `apps/joint_agent_service/tests/test_data/simple_cube.usda`, `apps/physics_agent_service/tests/test_data/simple_cube.usda`, `apps/texture_agent_service/tests/test_data/materialized_cube.usda` | Hand-written service test fixtures |
| `apps/material_agent/data/materials/material_refinement_examples/omnipbr_mdl.usda`, `apps/material_agent/data/materials/material_refinement_examples/preview_surface.usda` | Hand-written material examples; `omnipbr_mdl.usda` references the renderer's `OmniPBR.mdl`, not bundled |
| `apps/material_agent/data/regression/scene_optimizer/scene_optimizer_test.usda` | Hand-written regression scene |
| `apps/material_agent/data/templates/thumbnail_template.usd` | Material-preview render template (lights and camera) |
| `apps/material_agent_service/examples/large_scene/warehouse.usda` | Hand-written example scene |
| `apps/ovrtx_rendering_api/tests/renders/ovrtx_bug_repro.usda`, `apps/ovrtx_rendering_api/tests/renders/smoke_cube.usda` | Hand-written render test scenes |
| `apps/validation_agent/examples/fixtures/steel_scaffold_behavior/steel_scaffold_rollout.usda` | Hand-written validation fixture |
| `agentic/packages/geometry_repair/tests/fixtures/openvdb10_grid128_canonical_mesh.npz` | Generated test mesh (recipe in `openvdb10_to_13_parity_baseline.json`) |
| `.agents/skills/fixed-pipeline/references/material-agent-cli/evals/fixtures/prefailed-resume/dataset/usd/renders/ladder_side_rail_reference_1.ppm`, `ladder_side_rail_reference_2.ppm` | Hand-typed 2x2 test images |
