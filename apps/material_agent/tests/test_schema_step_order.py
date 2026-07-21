# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard tests for STEP_ORDER, STEP_OUTPUT_DIRS, and step defaults in schema.py.

These tests lock in the expected schema contents so that conflict resolution
during rebases cannot silently drop steps or break ordering invariants.
"""

import subprocess
import sys

from material_agent.config.schema import (
    MUTUALLY_EXCLUSIVE_STEPS,
    STEP_ORDER,
    STEP_OUTPUT_DIRS,
    get_step_defaults,
)


def test_schema_cold_import_does_not_depend_on_api_import_order():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from material_agent.config.schema import get_step_defaults; "
                "print(get_step_defaults('validate_input')['enabled'])"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_cluster_step_defaults_copy_thresholds():
    first = get_step_defaults("cluster_prims")
    first["complexity_thresholds"]["low"][2] = 0.1

    second = get_step_defaults("cluster_prims")

    assert second["complexity_thresholds"]["low"][2] == 0.98


def test_api_pipeline_step_names_include_create_materials():
    from material_agent.api.defaults import PIPELINE_STEP_NAMES

    assert "create_materials" in PIPELINE_STEP_NAMES
    assert PIPELINE_STEP_NAMES.index(
        "harmonize_predictions"
    ) < PIPELINE_STEP_NAMES.index("create_materials")
    assert PIPELINE_STEP_NAMES.index("create_materials") < PIPELINE_STEP_NAMES.index(
        "restore_usd"
    )


class TestStepOrderCompleteness:
    """Verify all expected steps are present in STEP_ORDER."""

    EXPECTED_STEPS = {
        "optimize_usd",
        "render_preview",
        "identify_asset",
        "generate_reference_image",
        "generate_material_library",
        "build_dataset_usd",
        "build_dataset_pdf_vectorstore",
        "build_dataset_prepare_dataset",
        "cluster_prims",
        "predict",
        "benchmark",
        "expand_cluster_predictions",
        "validate_predictions",
        "harmonize_predictions",
        "create_materials",
        "restore_usd",
        "apply",
        "evaluate",
        "refine",
        "render",
    }

    def test_all_expected_steps_present(self):
        """Every expected step must appear in STEP_ORDER."""
        missing = self.EXPECTED_STEPS - set(STEP_ORDER)
        assert not missing, f"Steps missing from STEP_ORDER: {missing}"

    def test_no_duplicates(self):
        """STEP_ORDER must not contain duplicate entries."""
        assert len(STEP_ORDER) == len(set(STEP_ORDER)), (
            f"Duplicate entries in STEP_ORDER: "
            f"{[s for s in STEP_ORDER if STEP_ORDER.count(s) > 1]}"
        )


class TestStepOrderRelativeOrdering:
    """Critical ordering invariants that must hold after any merge."""

    def _assert_before(self, earlier: str, later: str):
        """Assert *earlier* appears before *later* in STEP_ORDER."""
        assert earlier in STEP_ORDER, f"{earlier} not in STEP_ORDER"
        assert later in STEP_ORDER, f"{later} not in STEP_ORDER"
        assert STEP_ORDER.index(earlier) < STEP_ORDER.index(later), (
            f"{earlier} (idx {STEP_ORDER.index(earlier)}) must come before "
            f"{later} (idx {STEP_ORDER.index(later)})"
        )

    def test_optimize_before_build_dataset(self):
        self._assert_before("optimize_usd", "build_dataset_usd")

    def test_build_dataset_before_predict(self):
        self._assert_before("build_dataset_prepare_dataset", "predict")

    def test_generated_material_library_before_dataset_prompting(self):
        self._assert_before(
            "generate_material_library", "build_dataset_prepare_dataset"
        )

    def test_predict_before_validate(self):
        self._assert_before("predict", "validate_predictions")

    def test_validate_before_harmonize(self):
        self._assert_before("validate_predictions", "harmonize_predictions")

    def test_harmonize_before_apply(self):
        self._assert_before("harmonize_predictions", "apply")

    def test_harmonize_before_create_materials(self):
        self._assert_before("harmonize_predictions", "create_materials")

    def test_create_materials_before_restore(self):
        self._assert_before("create_materials", "restore_usd")

    def test_create_materials_before_apply(self):
        self._assert_before("create_materials", "apply")

    def test_cluster_prims_before_predict(self):
        self._assert_before("cluster_prims", "predict")

    def test_predict_before_expand(self):
        self._assert_before("predict", "expand_cluster_predictions")

    def test_restore_before_apply(self):
        self._assert_before("restore_usd", "apply")

    def test_apply_before_render(self):
        self._assert_before("apply", "render")

    def test_apply_before_evaluate(self):
        self._assert_before("apply", "evaluate")


class TestStepOutputDirs:
    """Verify STEP_OUTPUT_DIRS covers all steps that produce output directories."""

    STEPS_WITH_OUTPUT_DIRS = {
        "optimize_usd",
        "render_preview",
        "identify_asset",
        "generate_reference_image",
        "generate_material_library",
        "build_dataset_usd",
        "build_dataset_pdf_vectorstore",
        "build_dataset_prepare_dataset",
        "cluster_prims",
        "predict",
        "benchmark",
        "expand_cluster_predictions",
        "create_materials",
        "evaluate",
        "refine",
        "restore_usd",
        "render",
    }

    def test_all_expected_dirs_present(self):
        missing = self.STEPS_WITH_OUTPUT_DIRS - set(STEP_OUTPUT_DIRS)
        assert not missing, f"Steps missing from STEP_OUTPUT_DIRS: {missing}"

    def test_all_dir_values_are_nonempty_strings(self):
        for step, directory in STEP_OUTPUT_DIRS.items():
            assert isinstance(directory, str) and directory, (
                f"STEP_OUTPUT_DIRS[{step!r}] must be a non-empty string, got {directory!r}"
            )


class TestMutuallyExclusiveSteps:
    """Verify MUTUALLY_EXCLUSIVE_STEPS pairs are correct."""

    def test_predict_benchmark_exclusive(self):
        assert ["predict", "benchmark"] in MUTUALLY_EXCLUSIVE_STEPS

    def test_apply_refine_exclusive(self):
        assert ["apply", "refine"] in MUTUALLY_EXCLUSIVE_STEPS

    def test_all_exclusive_steps_in_step_order(self):
        for group in MUTUALLY_EXCLUSIVE_STEPS:
            for step in group:
                assert step in STEP_ORDER, (
                    f"Mutually exclusive step {step!r} not in STEP_ORDER"
                )


class TestGetStepDefaults:
    """Verify get_step_defaults returns a dict for every step in STEP_ORDER."""

    def test_all_steps_return_dict(self):
        for step in STEP_ORDER:
            defaults = get_step_defaults(step)
            assert isinstance(defaults, dict), (
                f"get_step_defaults({step!r}) returned {type(defaults)}, expected dict"
            )

    def test_unknown_step_returns_enabled_true(self):
        defaults = get_step_defaults("nonexistent_step_xyz")
        assert defaults == {"enabled": True}

    def test_generate_reference_image_public_defaults(self):
        defaults = get_step_defaults("generate_reference_image")

        assert defaults["enabled"] is False
        assert defaults["image_gen"] == {
            "backend": "gemini",
            "model": "gemini-3-pro-image-preview",
        }
        assert defaults["num_images"] == 1

    def test_generate_material_library_public_defaults(self):
        defaults = get_step_defaults("generate_material_library")

        assert defaults["enabled"] is False
        assert defaults["material_generation_plan_path"] is None
        assert defaults["texture_generation"]["texture_size"] == 1024
        assert defaults["write_material_generation_plan"] is True

    def test_create_materials_public_defaults(self):
        defaults = get_step_defaults("create_materials")

        assert defaults["enabled"] is False
        assert defaults["backend"] == "fake"
        assert defaults["creation_requests"] == []
        assert defaults["fail_on_error"] is True

    def test_empty_predictions_fail_closed_by_default(self) -> None:
        assert get_step_defaults("predict")["allow_empty_predictions"] is False
        assert get_step_defaults("benchmark")["allow_empty_predictions"] is False
        assert get_step_defaults("apply")["allow_empty_predictions"] is False
        assert get_step_defaults("refine")["apply"]["allow_empty_predictions"] is False

    def test_unknown_materials_do_not_fail_apply_by_default(self) -> None:
        assert get_step_defaults("apply")["fail_on_unknown_material"] is False
        assert get_step_defaults("refine")["apply"]["fail_on_unknown_material"] is False
